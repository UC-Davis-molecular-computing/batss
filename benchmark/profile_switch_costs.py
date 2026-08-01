"""Measure what a batch<->Gillespie mode switch actually costs, before optimizing anything.

Two GitHub issues propose reducing switch cost:

* #11 -- build the ``rebop.Gillespie`` object once in the constructor and call ``set_species`` per
  switch, instead of constructing it (and re-adding every reaction to it) on every entry.
* #13 -- fork rebop so it can run to a specified *number of reactions*.

Before acting on either, this quantifies the cost.  The simulator already reports
``switch.switch_overhead_seconds`` and ``switch.mode_switches``, which bracket exactly the two
routines that run on a transition:

    entering Gillespie -> initialize_gillespie_config()   # allocates, builds the rebop object,
                                                          # and re-adds every reaction
    leaving  Gillespie -> finalize_gillespie()            # sync_urn_from_gillespie()
                                                          #   + reset_k_count()
                                                          #     -> construct_transition_arrays()

Note that ``reset_k_count``/``construct_transition_arrays`` is a *third* cost that neither issue
addresses, so the breakdown below separates the entry side (which #11 targets) from the exit side.

The entry cost is measured independently: ``benchmark_engine_call(gillespie=True)`` reports
``setup_seconds``, which is exactly one ``initialize_gillespie_config``.  Given the average of the
two sides from a real run, the exit cost follows.

Run with::

    python benchmark/profile_switch_costs.py --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import profile_batch_costs as pbc
from policy_matrix_experiment import BASE_SLUGS, FAMILY_OF, build_scenarios
from switching_policy_comparison import Policy, Scenario, _configure_policy, _make_sim

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "switch_cost_profile.csv"


def measure_entry_cost(slug: str, *, repeats: int, seed: int) -> float:
    """Microseconds for one initialize_gillespie_config, from benchmark_engine_call setup."""

    by_slug = {case.case.slug: case for case in pbc.profile_cases()}
    base = by_slug.get(slug)
    if base is None:
        return math.nan
    paired = replace(base, measure_gillespie=True)
    values = []
    pbc._benchmark_once(paired, seed=pbc._stable_seed(seed, "warm"), gillespie=True, reactions=2000)
    for repeat in range(repeats):
        result = pbc._benchmark_once(paired, seed=pbc._stable_seed(seed, slug, repeat),
                                     gillespie=True, reactions=2000)
        values.append(result.setup_seconds * 1e6)
    return statistics.median(values)


def measure_run(scenario: Scenario, policy: Policy, seed: int, cap: float) -> dict[str, Any]:
    sim = _make_sim(scenario, seed)
    _configure_policy(sim, policy)
    start = time.perf_counter_ns()
    sim.simulator.run(scenario.case.end_time, cap)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    switch = sim.simulator.switch
    switches = int(switch.mode_switches)
    overhead = float(switch.switch_overhead_seconds)
    return {
        "scenario": scenario.slug,
        "family": scenario.family,
        "policy": policy.name,
        "seed": seed,
        "initial_n": scenario.case.initial_n,
        "elapsed_seconds": elapsed,
        "switch_overhead_seconds": overhead,
        "mode_switches": switches,
        "batch_calls": int(switch.batch_calls),
        "gillespie_calls": int(switch.gillespie_calls),
        "overhead_fraction": overhead / elapsed if elapsed > 0 else math.nan,
        "us_per_switch": overhead / switches * 1e6 if switches else math.nan,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="601,602,603")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--entry-repeats", type=int, default=25)
    parser.add_argument("--cap-seconds", type=float, default=10.0)
    parser.add_argument("--probe-repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seeds = [int(p) for p in args.seeds.split(",") if p.strip()]

    scenarios, _ = build_scenarios(probe_repeats=args.probe_repeats, seed=seeds[0])
    # Policies that actually switch a lot are the interesting ones; the wall-clock policy probes
    # periodically and so switches most, while a fixed threshold switches only at real crossings.
    policies = [Policy("wallclock_timing", "wallclock", 0, None),
                Policy("constant_250", "prospective", 2, 250.0)]

    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for seed in seeds:
            for scenario in scenarios:
                for policy in policies:
                    row = measure_run(scenario, policy, seed, args.cap_seconds)
                    row["repeat"] = repeat
                    rows.append(row)
        print(f"  repeat {repeat + 1}/{args.repeats} done", flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}\n")

    print("=== switch overhead as a share of total run time ===")
    print(f"{'scenario':<34} {'policy':<18} {'switches':>9} {'us/switch':>10} "
          f"{'overhead s':>11} {'% of run':>9}")
    for policy in policies:
        for scenario in scenarios:
            sel = [r for r in rows if r["scenario"] == scenario.slug
                   and r["policy"] == policy.name]
            if not sel:
                continue
            switches = statistics.median(r["mode_switches"] for r in sel)
            per = [r["us_per_switch"] for r in sel if math.isfinite(r["us_per_switch"])]
            frac = statistics.median(r["overhead_fraction"] for r in sel)
            over = statistics.median(r["switch_overhead_seconds"] for r in sel)
            print(f"{scenario.slug:<34} {policy.name:<18} {switches:>9.0f} "
                  f"{statistics.median(per) if per else float('nan'):>10.2f} "
                  f"{over:>11.5f} {100 * frac:>8.3f}%")

    print("\n=== summary ===")
    for policy in policies:
        sel = [r for r in rows if r["policy"] == policy.name and math.isfinite(r["overhead_fraction"])]
        fracs = [r["overhead_fraction"] for r in sel]
        pers = [r["us_per_switch"] for r in sel if math.isfinite(r["us_per_switch"])]
        print(f"  {policy.name:<18} median {100 * statistics.median(fracs):>6.3f}% of run time, "
              f"max {100 * max(fracs):>6.3f}%   |   median {statistics.median(pers):>7.2f} us/switch, "
              f"max {max(pers):>8.2f} us")

    print("\n=== entry-side cost only (initialize_gillespie_config), what issue #11 targets ===")
    print(f"{'CRN':<34} {'q':>3} {'B':>3} {'entry us':>10}")
    entry: dict[str, float] = {}
    by_slug = {case.case.slug: case for case in pbc.profile_cases()}
    for slug in BASE_SLUGS:
        value = measure_entry_cost(slug, repeats=args.entry_repeats, seed=seeds[0])
        entry[slug] = value
        base = by_slug.get(slug)
        if base is None or not math.isfinite(value):
            continue
        probe = _make_sim(Scenario(slug, FAMILY_OF[slug], base.case), seeds[0])
        print(f"{slug:<34} {int(probe.simulator.debug_q()):>3} "
              f"{int(probe.simulator.debug_output_branches()):>3} {value:>10.2f}")

    finite = [v for v in entry.values() if math.isfinite(v)]
    if finite:
        print(f"\n  entry cost spans {min(finite):.2f} .. {max(finite):.2f} us")
        print("  (an average switch is one entry + one exit, so compare 2x this against the "
              "us/switch column above to see how much of the cost is on the exit side)")


if __name__ == "__main__":
    main()
