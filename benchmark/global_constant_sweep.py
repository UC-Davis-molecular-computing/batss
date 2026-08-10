"""Which fixed threshold is actually best across the whole matrix, measured pairwise?

`T = 500` was selected by an early pilot and `T = 250` by a later one, but both selections rest on
comparisons that have since been retracted: they pooled medians across policies that were, on many
scenarios, executing the *identical* run. Paired per-seed checks on two CRNs then found `T = 100`
matching or beating both. So the constant that the cost model is being judged against is not itself
well established, and this re-establishes it.

Method, chosen to avoid the failure mode that produced the retractions:

* **Paired.** Every comparison is between runs of the same scenario *and the same seed*, so it is a
  comparison of trajectories rather than of pooled distributions.
* **Regret against the per-run best**, not against the adaptive policy. The adaptive policy is not
  reproducible -- it produced a different mode signature on every one of 24 runs in an earlier check
  -- so it makes a poor reference.
* **Policy-equivalence is reported, not assumed.** Thresholds that never straddle a score the
  trajectory encounters execute the identical run; scenarios where *every* constant does so
  contribute nothing and are counted separately rather than diluting the averages.
* **Equal weight per CRN family**, so families with many scenarios do not dominate.

Run with::

    python benchmark/global_constant_sweep.py --seeds 6 --repeats 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import measurement_conditions
from cost_model_head_to_head import Contender, fitted_coefficients, run_once
from policy_matrix_experiment import build_scenarios
from switching_policy_comparison import comparison_scenarios

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "global_constant_sweep.csv"

HEURISTIC_WALLCLOCK = 0
HEURISTIC_PROSPECTIVE = 2
HEURISTIC_COST_MODEL = 3

CONSTANTS = (60.0, 100.0, 150.0, 250.0, 400.0, 600.0, 1000.0)


def geometric_mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(finite)))) if finite else math.nan


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed-base", type=int, default=1301)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--probe-repeats", type=int, default=9)
    parser.add_argument("--cap-seconds", type=float, default=8.0)
    parser.add_argument("--override-factors", type=str, default="",
                        help="extra adaptive policies at these override factors. 1.0 makes "
                             "it genuinely greedy (pick the cheaper measured mode); the "
                             "shipped default of 4.0 makes it follow the old proxy rule "
                             "unless the alternative is >4x cheaper")
    parser.add_argument("--eager-probe", action="store_true",
                        help="also race a greedy policy that probes far more often, to see "
                             "whether stale estimates rather than the override gate are "
                             "what limits the adaptive policy")
    parser.add_argument("--constants", type=str, default=None,
                        help="comma-separated fixed thresholds to race (default: the full "
                             "grid). Narrow it to run a focused head-to-head with more "
                             "seeds, which buys statistical power where it matters.")
    parser.add_argument("--cold-timings", action="append", default=None,
                        help="cost CSVs for the cold-fitted model (must match this run's "
                             "machine conditions)")
    parser.add_argument("--warm-timings", action="append", default=None,
                        help="cost CSVs for the warm-fitted model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seeds = [args.seed_base + i for i in range(args.seeds)]
    args.cold_timings = args.cold_timings or [
        HERE / "batch_cost_profile_timings_allg.csv",
        HERE / "batch_cost_profile_timings_allg_seed2.csv"]
    args.warm_timings = args.warm_timings or [
        HERE / "batch_cost_profile_timings_warm.csv",
        HERE / "batch_cost_profile_timings_warm_seed2.csv"]

    # Stamp the machine conditions. Timings taken on battery cannot be compared with
    # timings taken on AC -- the processor is clocked lower -- and a draining battery
    # throttles progressively across a long run. Paired within-run comparisons survive
    # that; comparisons against numbers from another session do not.
    conditions = measurement_conditions.snapshot()
    print(f"measurement conditions: {measurement_conditions.describe(conditions)}")
    if conditions.get("power_source") == "on_battery":
        print("  NOTE: on battery. Results here are internally comparable, and comparable with "
              "other runs made on battery, but not with runs made on AC power.")

    # The coefficients must come from cost data measured under the SAME machine conditions as the
    # run being scored. Batch and Gillespie do not slow by the same factor when conditions change --
    # measured 1.365x versus 1.221x between a battery session and an AC one -- so T* itself shifts,
    # and coefficients fitted elsewhere are miscalibrated.
    cold_b, cold_g = fitted_coefficients([Path(p) for p in args.cold_timings])
    warm_b, warm_g = fitted_coefficients([Path(p) for p in args.warm_timings])
    # The adaptive policy as shipped is a poor stand-in for "optimal": with override_factor = 4 it
    # follows the OLD PROXY RULE unless measurement shows the alternative is more than 4x cheaper,
    # so whenever the better engine is only 1.5-3x better it never switches. Racing several
    # override factors finds the best measurement-based policy actually available, which is the
    # reference the deterministic rules deserve to be judged against. Factor 1.0 is genuinely
    # greedy: pick whichever mode measures cheaper.
    contenders = [Contender("timing", HEURISTIC_WALLCLOCK)]
    contenders += [Contender(f"timing_ovr{f:g}", HEURISTIC_WALLCLOCK, override_factor=f)
                   for f in (float(x) for x in args.override_factors.split(",") if x.strip())]
    if args.eager_probe:
        contenders.append(Contender("timing_ovr1_eager", HEURISTIC_WALLCLOCK, override_factor=1.0,
                                    probe_interval=64, probe_interval_committed=512))
    chosen = ([float(x) for x in args.constants.split(",") if x.strip()]
              if args.constants else list(CONSTANTS))
    contenders += [Contender(f"T={c:g}", HEURISTIC_PROSPECTIVE, threshold=c) for c in chosen]
    contenders += [
        Contender("cost_model_cold", HEURISTIC_COST_MODEL, batch_coefficients=cold_b,
                  gillespie_coefficients=cold_g, scale=0.368),
        Contender("cost_model_warm", HEURISTIC_COST_MODEL, batch_coefficients=warm_b,
                  gillespie_coefficients=warm_g, scale=1.0),
    ]
    constants = [c.name for c in contenders if c.name.startswith("T=")]

    placed, _ = build_scenarios(probe_repeats=args.probe_repeats, seed=seeds[0])
    names = {s.case.spec.name for s in placed}
    scenarios = placed + [s for s in comparison_scenarios() if s.case.spec.name not in names]
    kept = []
    for s in scenarios:
        if run_once(s, contenders[0], seeds[0], args.cap_seconds)["timed_out"]:
            print(f"  dropping {s.slug}: exceeds the {args.cap_seconds:g}s cap")
        else:
            kept.append(s)
    scenarios = kept
    print(f"\n{len(scenarios)} scenarios x {len(contenders)} policies x {len(seeds)} seeds "
          f"x {args.repeats} repeats\n")

    rows: list[dict[str, Any]] = []
    # time[policy][scenario][seed]
    times: dict[str, dict[str, dict[int, float]]] = {c.name: {} for c in contenders}
    raw: dict[str, dict[str, dict[int, list[float]]]] = {c.name: {} for c in contenders}
    signatures: dict[str, dict[str, set]] = {c.name: {} for c in contenders}
    for c in contenders:
        for s in scenarios:
            times[c.name][s.slug] = {}
            raw[c.name][s.slug] = {seed: [] for seed in seeds}
            signatures[c.name][s.slug] = set()

    # Rotate the order the policies are executed in. Measuring them in a fixed order gives whichever
    # runs first the benefit of any monotonic drift -- thermal, or a battery draining across a long
    # session -- and in the earlier harnesses that was always the adaptive policy. Rotating means no
    # policy sits systematically at the front.
    order_counter = 0
    for i, scenario in enumerate(scenarios, 1):
        for seed in seeds:
            for rep in range(args.repeats):
                rotation = order_counter % len(contenders)
                order_counter += 1
                for c in contenders[rotation:] + contenders[:rotation]:
                    r = run_once(scenario, c, seed, args.cap_seconds)
                    rows.append({"scenario": scenario.slug, "family": scenario.family,
                                 "policy": c.name, "seed": seed, "repeat": rep,
                                 "execution_position": (contenders[rotation:]
                                                        + contenders[:rotation]).index(c), **r})
                    signatures[c.name][scenario.slug].add(
                        (r["batch_calls"], r["gillespie_calls"], r["mode_switches"]))
                    if not r["timed_out"] and not r.get("panicked"):
                        raw[c.name][scenario.slug][seed].append(r["elapsed_seconds"])
        for c in contenders:
            for seed in seeds:
                vals = raw[c.name][scenario.slug][seed]
                if vals:
                    times[c.name][scenario.slug][seed] = statistics.median(vals)
        print(f"  [{i}/{len(scenarios)}] {scenario.slug}", flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.output}\n")
    panics = [r for r in rows if r.get("panicked")]
    if panics:
        # A simulator panic is a defect in batss, not a measurement artifact, so surface it
        # loudly rather than letting it vanish into a lower sample count.
        print(f"=== {len(panics)} runs PANICKED in the simulator and were excluded ===")
        for scen in dict.fromkeys(r["scenario"] for r in panics):
            hit = [r for r in panics if r["scenario"] == scen]
            print(f"  {scen}: {len(hit)} runs; {hit[0]['error'][:150]}")
        print()

    # Which scenarios can distinguish the constants at all? Compare the *sets* of mode signatures
    # each constant produced: if every constant produced the same set, they all executed the same
    # runs and the scenario says nothing about which threshold is better.
    discriminating = []
    for s in scenarios:
        per_constant = [frozenset(signatures[p][s.slug]) for p in constants
                        if signatures[p][s.slug]]
        if len(set(per_constant)) > 1:
            discriminating.append(s)
    print(f"=== {len(discriminating)}/{len(scenarios)} scenarios distinguish the constants "
          f"(the rest run identically whatever the threshold, so they only add noise) ===\n")

    def report(subset, label):
        if not subset:
            return
        families = sorted({s.family for s in subset})
        print(f"--- {label} (n={len(subset)}) ---")
        print(f"  {'policy':<18} {'equal-family regret':>20} {'worst scenario':>16} "
              f"{'vs timing':>11}")
        out = {}
        for c in contenders:
            per_scenario, vs_timing = {}, {}
            for s in subset:
                # paired: best policy for this scenario AND seed, then median over seeds
                reg, rel = [], []
                for seed in seeds:
                    have = [times[p.name][s.slug].get(seed) for p in contenders]
                    have = [v for v in have if v is not None]
                    mine = times[c.name][s.slug].get(seed)
                    t = times["timing"][s.slug].get(seed)
                    if mine is None or not have:
                        continue
                    reg.append(mine / min(have))
                    if t:
                        rel.append(mine / t)
                if reg:
                    per_scenario[s.slug] = statistics.median(reg)
                if rel:
                    vs_timing[s.slug] = statistics.median(rel)
            fam = [geometric_mean([per_scenario[s.slug] for s in subset
                                   if s.family == f and s.slug in per_scenario])
                   for f in families]
            fam = [x for x in fam if math.isfinite(x)]
            out[c.name] = {"regret": geometric_mean(fam),
                           "worst": max(per_scenario.values()) if per_scenario else math.nan,
                           "vs_timing": geometric_mean(list(vs_timing.values()))}
            v = out[c.name]
            print(f"  {c.name:<18} {v['regret']:>19.3f}x {v['worst']:>15.3f}x "
                  f"{v['vs_timing']:>10.3f}x")
        best = min(out, key=lambda k: out[k]["regret"])
        best_c = min((k for k in out if k.startswith("T=")), key=lambda k: out[k]["regret"])
        best_w = min((k for k in out if k.startswith("T=")), key=lambda k: out[k]["worst"])
        print(f"  -> best overall: {best};  best constant by regret: {best_c};  "
              f"by worst case: {best_w}\n")
        return out

    everything = report(scenarios, "all scenarios")
    disc = report(discriminating, "only scenarios that distinguish the constants")

    args.output.with_suffix(".json").write_text(json.dumps(
        {"seeds": seeds, "repeats": args.repeats, "constants": list(CONSTANTS),
         "measurement_conditions": conditions,
         "all": everything, "discriminating": disc,
         "discriminating_scenarios": [s.slug for s in discriminating]}, indent=2), encoding="utf-8")
    print(f"wrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
