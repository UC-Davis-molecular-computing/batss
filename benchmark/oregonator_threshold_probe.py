"""Can any deterministic threshold match the adaptive policy on the stiff oscillator?

On `oregonator_n1e5` the adaptive wall-clock policy beats every deterministic policy by 20-28%.
The frozen-state oracle says batching is *never* locally optimal there: across 16 states sampled
along its trajectory, `score/T*` never exceeds 0.077, peaking at a score of 39.79 against a
break-even near 400-530. So every threshold tested so far (250 and up) executes exactly one batch --
the mandatory initial one -- and then stays in Gillespie for the whole run.

Two possibilities, and they call for different responses:

* A very low threshold (below ~40) *can* batch during the spikes, and doing so is fast. Then a
  deterministic rule can reach the adaptive policy's time, and the frozen-state break-even is simply
  the wrong criterion for stiff oscillators -- it is measured at a fixed state and cannot see that a
  batch call advances far more simulated time than a Gillespie call when the propensity spikes.
* No threshold reaches it. Then the adaptive policy's advantage does not come from its 14 batch
  calls at all, and the gap has some other cause that a threshold cannot address.

This sweeps thresholds far below anything previously tested, on the Oregonator and on the other
CRNs where deterministic policies lose, and reports wall-clock plus the mode split so the two cases
can be told apart.

Run with::

    python benchmark/oregonator_threshold_probe.py --seeds 1001,1002,1003 --repeats 3
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

from switching_policy_comparison import Scenario, _make_sim, comparison_scenarios

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "oregonator_threshold_probe.csv"

HEURISTIC_WALLCLOCK = 0
HEURISTIC_PROSPECTIVE = 2

# Deliberately spans far below the scores these CRNs ever reach, since that is the only region where
# a threshold can make the simulator batch during a spike.
THRESHOLDS = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0, 40.0, 60.0, 100.0, 250.0, 500.0)


def run_once(scenario: Scenario, selector: int, threshold: float | None, seed: int,
             cap: float) -> dict[str, Any]:
    sim = _make_sim(scenario, seed)
    rust = sim.simulator
    rust.heuristic_gillespie_switching = selector
    if threshold is not None:
        rust.proxy_threshold = threshold
    start = time.perf_counter_ns()
    rust.run(scenario.case.end_time, cap)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    switch = rust.switch
    completed = rust.continuous_time >= scenario.case.end_time * (1 - 1e-9)
    total = float(switch.batch_continuous_time) + float(switch.gillespie_continuous_time)
    return {
        "elapsed_seconds": elapsed,
        "timed_out": (not completed) and elapsed >= 0.9 * cap,
        "batch_calls": int(switch.batch_calls),
        "gillespie_calls": int(switch.gillespie_calls),
        "mode_switches": int(switch.mode_switches),
        "gillespie_time_fraction": (float(switch.gillespie_continuous_time) / total
                                    if total > 0 else math.nan),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="1001,1002,1003")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cap-seconds", type=float, default=20.0)
    parser.add_argument("--scenarios", type=str,
                        default="oregonator_n1e5,brusselator_n2e4,rossler_n1e5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seeds = [int(p) for p in args.seeds.split(",") if p.strip()]
    wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    scenarios = [s for s in comparison_scenarios() if s.slug in wanted]

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        print(f"\n=== {scenario.slug}  (n={scenario.case.initial_n:,}, "
              f"t_max={scenario.case.end_time}) ===")
        print(f"{'policy':<18} {'median s':>10} {'vs timing':>10} {'batch':>9} {'gillespie':>10} "
              f"{'switches':>9} {'gill time frac':>15}")

        def measure(label: str, selector: int, threshold: float | None) -> float:
            per_seed = []
            last = None
            for seed in seeds:
                vals = []
                for _ in range(args.repeats):
                    r = run_once(scenario, selector, threshold, seed, args.cap_seconds)
                    if not r["timed_out"]:
                        vals.append(r["elapsed_seconds"])
                    last = r
                if vals:
                    per_seed.append(statistics.median(vals))
                rows.append({"scenario": scenario.slug, "policy": label, "seed": seed,
                             "median_seconds": statistics.median(vals) if vals else math.nan,
                             **{k: v for k, v in (last or {}).items() if k != "elapsed_seconds"}})
            return statistics.median(per_seed) if per_seed else math.nan

        reference = measure("timing", HEURISTIC_WALLCLOCK, None)
        last = rows[-1]
        print(f"{'timing':<18} {reference:>10.5f} {1.0:>10.3f} {last['batch_calls']:>9} "
              f"{last['gillespie_calls']:>10} {last['mode_switches']:>9} "
              f"{last['gillespie_time_fraction']:>15.6f}")
        best_label, best_time = "timing", reference
        for threshold in THRESHOLDS:
            label = f"T={threshold:g}"
            t = measure(label, HEURISTIC_PROSPECTIVE, threshold)
            last = rows[-1]
            if t < best_time:
                best_label, best_time = label, t
            print(f"{label:<18} {t:>10.5f} {t / reference:>10.3f} {last['batch_calls']:>9} "
                  f"{last['gillespie_calls']:>10} {last['mode_switches']:>9} "
                  f"{last['gillespie_time_fraction']:>15.6f}")
        print(f"  best: {best_label} at {best_time:.5f}s "
              f"({best_time / reference:.3f}x the adaptive policy)")

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
