"""Does the time-based Gillespie block heuristic actually hit its reaction target? (issue #13)

`gillespie_steps` wants to run about `sqrt(n)` reactions before reconsidering the engine choice.
Historically it could only ask rebop to advance to a *time*, so it converted the count into a
duration using the total propensity measured on entry::

    time_to_run = sqrt(n) / total_propensity_at_entry

That is exact only if the propensity stays constant across the block.  When the propensity rises
mid-block the same duration buys far more reactions than intended, which keeps the simulator in
Gillespie mode long past the point where batching became the better choice; when it falls, the block
is cut short and the switch overhead is paid more often than necessary.

The forked rebop adds `advance_until_or_reactions`, so a block can be budgeted in reactions
directly.  Both paths now report how many reactions actually fired, which makes the over/undershoot
directly observable rather than inferred.

This script reports, per CRN:

* ``executed / targeted``  under the time-based rule -- 1.0 means the conversion is accurate
* the same ratio under the reaction-based rule, which should be 1.0 by construction (it can fall
  below 1 only when ``t_max`` or a silent configuration ends the block early)
* end-to-end wall-clock under both, so an inaccurate heuristic is only "bad news" if it costs time

Run with::

    python benchmark/gillespie_block_sizing.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from policy_matrix_experiment import build_scenarios
from switching_policy_comparison import Policy, Scenario, _configure_policy, _make_sim

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "gillespie_block_sizing.csv"


def run_once(scenario: Scenario, policy: Policy, seed: int, cap: float,
             by_reactions: bool) -> dict[str, Any]:
    sim = _make_sim(scenario, seed)
    _configure_policy(sim, policy)
    sim.simulator.gillespie_block_by_reactions = by_reactions
    start = time.perf_counter_ns()
    sim.simulator.run(scenario.case.end_time, cap)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    rust = sim.simulator
    switch = rust.switch
    executed = int(rust.gillespie_reactions_executed)
    targeted = int(rust.gillespie_reactions_targeted)
    completed = rust.continuous_time >= scenario.case.end_time * (1 - 1e-9)
    return {
        "scenario": scenario.slug, "family": scenario.family, "policy": policy.name,
        "by_reactions": by_reactions, "seed": seed, "initial_n": scenario.case.initial_n,
        "elapsed_seconds": elapsed, "completed": completed,
        "timed_out": (not completed) and elapsed >= 0.9 * cap,
        "gillespie_calls": int(switch.gillespie_calls),
        "batch_calls": int(switch.batch_calls),
        "reactions_executed": executed, "reactions_targeted": targeted,
        "overshoot": executed / targeted if targeted else math.nan,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="701,702,703")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cap-seconds", type=float, default=10.0)
    parser.add_argument("--probe-repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seeds = [int(p) for p in args.seeds.split(",") if p.strip()]

    scenarios, _ = build_scenarios(probe_repeats=args.probe_repeats, seed=seeds[0])
    # A fixed threshold makes the mode sequence deterministic, so the two block-sizing rules are
    # compared on the same switching policy rather than against the adaptive one.
    policy = Policy("constant_250", "prospective", 2, 250.0)

    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for seed in seeds:
            for scenario in scenarios:
                for by_reactions in (False, True):
                    row = run_once(scenario, policy, seed, args.cap_seconds, by_reactions)
                    row["repeat"] = repeat
                    rows.append(row)
        print(f"  repeat {repeat + 1}/{args.repeats} done", flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}\n")

    def med(slug: str, by_reactions: bool, key: str) -> float:
        vals = [r[key] for r in rows if r["scenario"] == slug
                and r["by_reactions"] == by_reactions and not r["timed_out"]
                and isinstance(r[key], (int, float)) and math.isfinite(float(r[key]))]
        return statistics.median(vals) if vals else math.nan

    print("overshoot = Gillespie reactions actually executed / reactions the block aimed for")
    print("a time-based block is accurate only if the propensity is constant across it\n")
    print(f"{'scenario':<34} {'gill calls':>10} {'overshoot(time)':>16} "
          f"{'overshoot(count)':>17} {'time s':>9} {'count s':>9} {'ratio':>7}")
    print("-" * 108)
    overshoots = []
    speedups = []
    for scenario in scenarios:
        slug = scenario.slug
        calls = med(slug, False, "gillespie_calls")
        o_time = med(slug, False, "overshoot")
        o_count = med(slug, True, "overshoot")
        t_time = med(slug, False, "elapsed_seconds")
        t_count = med(slug, True, "elapsed_seconds")
        if not math.isfinite(t_time) or not math.isfinite(t_count):
            continue
        ratio = t_count / t_time
        if math.isfinite(o_time) and calls and calls > 0:
            overshoots.append(o_time)
        speedups.append(ratio)
        print(f"{slug:<34} {calls:>10.0f} {o_time:>16.3f} {o_count:>17.3f} "
              f"{t_time:>9.4g} {t_count:>9.4g} {ratio:>7.3f}")

    if overshoots:
        print(f"\ntime-based overshoot across CRNs: min {min(overshoots):.3f}  "
              f"median {statistics.median(overshoots):.3f}  max {max(overshoots):.3f}")
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"reaction-based wall-clock vs time-based: geometric mean {geo:.4f}x  "
              f"(best {min(speedups):.3f}x, worst {max(speedups):.3f}x)")


if __name__ == "__main__":
    main()
