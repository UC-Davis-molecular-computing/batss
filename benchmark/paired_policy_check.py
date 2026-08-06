"""Paired, seed-by-seed comparison of switching policies on a single scenario.

Written for the last outstanding claim that the adaptive wall-clock policy beats deterministic
switching: Rossler, where it led by 1.04-1.15x. Every other such claim has evaporated once mode
signatures were checked, because thresholds that never straddle an encountered score execute the
identical run and their timing differences are noise. On Rossler the earlier sweep found nine
thresholds (0.5 through 60) all executing 161,441 batch calls and zero Gillespie calls -- one run
measured nine times, spanning 11%.

An aggregate median hides that. This compares policies **per seed**, so each comparison is between
runs of the same trajectory, and reports:

* the paired ratio to the adaptive policy for every seed, not just its median
* a sign test over seeds, which does not assume anything about the noise distribution
* the spread across repeats *within* one policy and seed, which is the noise floor the differences
  have to clear
* the mode signature of each policy, so policy-equivalent entries are visible rather than inferred

Run with::

    python benchmark/paired_policy_check.py --scenario rossler_n1e5 --seeds 8 --repeats 3
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

import threshold_cost_model as tcm
from cost_model_head_to_head import Contender, fitted_coefficients
from policy_matrix_experiment import build_scenarios
from switching_policy_comparison import Scenario, _make_sim, comparison_scenarios

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "paired_policy_check.csv"

HEURISTIC_WALLCLOCK = 0
HEURISTIC_PROSPECTIVE = 2
HEURISTIC_COST_MODEL = 3


def run_once(scenario: Scenario, c: Contender, seed: int, cap: float) -> dict[str, Any]:
    sim = _make_sim(scenario, seed)
    rust = sim.simulator
    rust.heuristic_gillespie_switching = c.selector
    if c.threshold is not None:
        rust.proxy_threshold = c.threshold
    if c.batch_coefficients is not None:
        rust.cost_model_batch_coefficients = list(c.batch_coefficients)
        rust.cost_model_gillespie_coefficients = list(c.gillespie_coefficients)
        rust.cost_model_scale = c.scale
    start = time.perf_counter_ns()
    rust.run(scenario.case.end_time, cap)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    s = rust.switch
    completed = rust.continuous_time >= scenario.case.end_time * (1 - 1e-9)
    return {"elapsed_seconds": elapsed,
            "timed_out": (not completed) and elapsed >= 0.9 * cap,
            "signature": (int(s.batch_calls), int(s.gillespie_calls), int(s.mode_switches))}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=str, default="rossler_n1e5")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=1101)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cap-seconds", type=float, default=30.0)
    parser.add_argument("--thresholds", type=str, default="8,100,250,500")
    parser.add_argument("--probe-repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seeds = [args.seed_base + i for i in range(args.seeds)]

    # Look in both scenario sources: the natural-population set, and the boundary-placed set where
    # each CRN is scaled so its trajectory starts near its own break-even. The cost model's claimed
    # advantage lives in the latter, so it has to be checkable here.
    available = list(comparison_scenarios())
    if not any(s.slug == args.scenario for s in available):
        placed, _ = build_scenarios(probe_repeats=args.probe_repeats, seed=args.seed_base)
        available += placed
    matches = [s for s in available if s.slug == args.scenario]
    if not matches:
        raise SystemExit(f"unknown scenario {args.scenario!r}; available: "
                         + ", ".join(sorted(s.slug for s in available)))
    scenario = matches[0]
    cold_b, cold_g = fitted_coefficients([HERE / "batch_cost_profile_timings_allg.csv",
                                          HERE / "batch_cost_profile_timings_allg_seed2.csv"])
    warm_b, warm_g = fitted_coefficients([HERE / "batch_cost_profile_timings_warm.csv",
                                          HERE / "batch_cost_profile_timings_warm_seed2.csv"])
    contenders = [Contender("timing", HEURISTIC_WALLCLOCK)]
    contenders += [Contender(f"T={t:g}", HEURISTIC_PROSPECTIVE, threshold=float(t))
                   for t in (float(x) for x in args.thresholds.split(",") if x.strip())]
    contenders += [
        Contender("cost_model_cold", HEURISTIC_COST_MODEL, batch_coefficients=cold_b,
                  gillespie_coefficients=cold_g, scale=0.368),
        Contender("cost_model_warm", HEURISTIC_COST_MODEL, batch_coefficients=warm_b,
                  gillespie_coefficients=warm_g, scale=1.0),
    ]

    print(f"{scenario.slug}: n={scenario.case.initial_n:,}, t_max={scenario.case.end_time}, "
          f"{len(seeds)} seeds x {args.repeats} repeats\n")
    rows: list[dict[str, Any]] = []
    per_seed: dict[str, dict[int, float]] = {c.name: {} for c in contenders}
    spread: dict[str, list[float]] = {c.name: [] for c in contenders}
    signatures: dict[str, set[tuple[int, int, int]]] = {c.name: set() for c in contenders}
    for seed in seeds:
        for c in contenders:
            vals = []
            for rep in range(args.repeats):
                r = run_once(scenario, c, seed, args.cap_seconds)
                rows.append({"scenario": scenario.slug, "policy": c.name, "seed": seed,
                             "repeat": rep, **r})
                if not r["timed_out"]:
                    vals.append(r["elapsed_seconds"])
                signatures[c.name].add(r["signature"])
            if vals:
                per_seed[c.name][seed] = statistics.median(vals)
                if len(vals) > 1:
                    spread[c.name].append(max(vals) / min(vals))
        print(f"  seed {seed} done", flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.output}\n")

    print("=== noise floor: spread across repeats of the SAME policy and seed ===")
    for c in contenders:
        s = spread[c.name]
        if s:
            print(f"  {c.name:<18} median {statistics.median(s):.3f}x, max {max(s):.3f}x")

    print("\n=== mode signatures (identical signature => identical run) ===")
    for c in contenders:
        sigs = sorted(signatures[c.name])
        shown = sigs[0] if len(sigs) == 1 else f"{len(sigs)} distinct, e.g. {sigs[0]}"
        print(f"  {c.name:<18} {shown}")

    ref = per_seed["timing"]
    print(f"\n=== paired against the adaptive policy, seed by seed ===")
    header = f"  {'policy':<18}" + "".join(f"{s:>9}" for s in seeds) + f"{'median':>9}{'wins':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    summary = {}
    for c in contenders[1:]:
        ratios = [per_seed[c.name][s] / ref[s] for s in seeds
                  if s in per_seed[c.name] and s in ref]
        wins = sum(1 for r in ratios if r < 1.0)
        line = f"  {c.name:<18}" + "".join(f"{r:>9.3f}" for r in ratios)
        line += f"{statistics.median(ratios):>9.3f}{wins:>4}/{len(ratios)}"
        print(line)
        summary[c.name] = {"paired_median": statistics.median(ratios),
                           "wins_vs_timing": wins, "n_seeds": len(ratios),
                           "ratios": ratios}

    print(f"\n=== verdict ===")
    noise = statistics.median([x for c in contenders for x in spread[c.name]] or [math.nan])
    print(f"  within-policy noise floor (median repeat spread): {noise:.3f}x")
    for name, v in summary.items():
        m, w, n = v["paired_median"], v["wins_vs_timing"], v["n_seeds"]
        # A sign test: with no real difference, wins is Binomial(n, 1/2).
        decisive = w == 0 or w == n
        print(f"  {name:<18} paired median {m:.3f}x, beats timing in {w}/{n} seeds"
              f"{'  (consistent)' if decisive else '  (mixed -- not decisive)'}")


if __name__ == "__main__":
    main()
