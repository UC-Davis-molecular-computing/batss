"""Wide end-to-end policy comparison: many CRN structures, each near its own switching boundary.

`end_to_end_threshold_check.py` could not settle whether structure-aware thresholds pay off,
because seven of its ten scenarios were *policy-equivalent* -- every deterministic threshold
executed the identical mode sequence, so the comparison was measuring nothing but noise.  Its three
discriminating scenarios came from only two CRN families and spanned just 461..1009 in predicted
threshold, which is far too narrow to expose a structural effect.

This experiment fixes both problems at once:

1. **Many more structures.**  Every distinct CRN available in the benchmark suite is used: the five
   real ones, the three matched Shrinking topologies, the split-channel family that varies ``B``,
   the dense-support family that varies ``q`` (whose break-even is as low as 223), and the collision
   family that varies ``o`` and ``g`` (whose break-even reaches 1524).
2. **Populations placed on the boundary.**  Composition is invariant under scaling ``n``, so
   ``score`` scales exactly as ``sqrt(n)``; for each CRN the population is solved so the trajectory
   *starts* at a chosen multiple of its own break-even, and therefore crosses it as it evolves.
   This is what "different initial population sizes for each CRN" is for -- the sizes are chosen per
   CRN so the switching decision is contested, not picked from a fixed grid.

Because a wide matrix wastes time on scenarios that cannot discriminate, a cheap pilot pass runs
first and keeps only scenarios that (a) complete under the time cap for every policy and (b) show at
least two distinct mode signatures across the deterministic policies.  Everything screened out is
reported with its reason -- no silent truncation.

Run with::

    python benchmark/policy_matrix_experiment.py --seeds 301,302,303,304,305 --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import near_boundary_states as nbs
import profile_batch_costs as pbc
import threshold_cost_model as tcm
from switching_policy_comparison import Policy, Scenario, _configure_policy, _make_sim
from threshold_model import experiment_cases

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "policy_matrix_experiment.csv"

# End-to-end runs execute whole trajectories, so the population that puts a CRN on its boundary can
# still be far too large to simulate.  Anything above this is dropped and reported.
N_CAP = 200_000_000
N_FLOOR = 20_000
TARGET_RATIOS = (0.5, 1.0, 2.0)

# Every distinct structure the suite provides.  Composition variants are excluded because they
# duplicate a structure already present; the o=3 collision cases are included but usually fail the
# population cap, which is itself a reported finding.
BASE_SLUGS = (
    # five real benchmark CRNs
    "oregonator", "rossler", "dimerization", "lotka_volterra", "brusselator",
    # matched Shrinking topologies (vary R and B at fixed generator)
    "shrinking", "shrinking_no_b_decay", "shrinking_split_b_decay",
    # channel-count family (varies B only)
    "shrinking_b_decay_channels_1", "shrinking_b_decay_channels_2",
    "shrinking_b_decay_channels_4", "shrinking_b_decay_channels_8",
    # dense-support family (varies q only) -- the low break-even extreme
    "dense_support_q4", "dense_support_q6", "dense_support_q8", "dense_support_q12",
    # collision family (varies o and g) -- the high break-even extreme
    "collision_o2_g0_n1000000", "collision_o2_g1_n1000000", "collision_o2_g2_n1000000",
    "collision_o3_g0_n1000000",
)

FAMILY_OF = {
    "oregonator": "oregonator", "rossler": "rossler", "dimerization": "dimerization",
    "lotka_volterra": "lotka_volterra", "brusselator": "brusselator",
    "shrinking": "shrinking", "shrinking_no_b_decay": "shrinking",
    "shrinking_split_b_decay": "shrinking",
    "shrinking_b_decay_channels_1": "branch_count",
    "shrinking_b_decay_channels_2": "branch_count",
    "shrinking_b_decay_channels_4": "branch_count",
    "shrinking_b_decay_channels_8": "branch_count",
    "dense_support_q4": "dense_support", "dense_support_q6": "dense_support",
    "dense_support_q8": "dense_support", "dense_support_q12": "dense_support",
    "collision_o2_g0_n1000000": "collision", "collision_o2_g1_n1000000": "collision",
    "collision_o2_g2_n1000000": "collision", "collision_o3_g0_n1000000": "collision",
}


def build_scenarios(*, probe_repeats: int, seed: int) -> tuple[list[Scenario], list[dict[str, Any]]]:
    """Place each CRN at populations where its trajectory starts near its own break-even."""

    by_slug = {case.case.slug: case for case in pbc.profile_cases()}
    scenarios: list[Scenario] = []
    report: list[dict[str, Any]] = []
    for slug in BASE_SLUGS:
        base = by_slug.get(slug)
        if base is None:
            report.append({"base": slug, "status": "unknown slug"})
            continue
        paired = replace(base, measure_gillespie=True)
        score_ref = nbs._score_at(paired, nbs.pbc._stable_seed(seed, "score", slug))
        t_star = nbs._measure_t_star(paired, repeats=probe_repeats, seed=seed,
                                     gillespie_reactions=5000)
        if not (math.isfinite(score_ref) and math.isfinite(t_star)) or score_ref <= 0:
            report.append({"base": slug, "status": "no paired measurement"})
            continue
        reference_n = base.case.initial_n
        for ratio in TARGET_RATIOS:
            target = reference_n * (ratio * t_star / score_ref) ** 2
            entry = {"base": slug, "family": FAMILY_OF[slug], "target_ratio": ratio,
                     "T_star_probe": t_star, "wanted_n": target}
            if target > N_CAP:
                entry["status"] = f"needs n={target:.3g} > cap {N_CAP:.3g}"
                report.append(entry)
                continue
            population = int(max(target, N_FLOOR))
            entry["n"] = population
            entry["status"] = "candidate"
            report.append(entry)
            scenarios.append(Scenario(
                f"{slug}_r{ratio:g}".replace(".", ""),
                FAMILY_OF[slug],
                replace(base.case, initial_n=population),
            ))
    return scenarios, report


def cost_model_threshold(scenario: Scenario, batch_fit: dict[str, Any],
                         gillespie_fit: dict[str, Any], seed: int) -> float:
    sim = _make_sim(scenario, seed)
    rust = sim.simulator
    order, gen = int(rust.debug_o()), int(rust.debug_g())
    prospective_n = int(rust.debug_prospective_n())
    expected_length = (math.sqrt(math.pi / (2.0 * order * (order + gen))) * math.sqrt(prospective_n)
                       if prospective_n > 0 and order > 0 else 0.0)
    row = {"prospective_N": str(prospective_n), "q": str(int(rust.debug_q())), "o": str(order),
           "g": str(gen), "score": str(float(rust.prospective_batch_score())),
           "expected_batch_length": str(expected_length),
           "output_branches_B": str(int(rust.debug_output_branches()))}
    batch = sum(tcm.batch_features(row)[n] * batch_fit["coefficients"][n] for n in tcm.BATCH_FEATURES)
    gill = sum(tcm.gillespie_features(row, horizon=5000.0)[n] * gillespie_fit["coefficients"][n]
               for n in tcm.GILLESPIE_FEATURES)
    return batch / gill


def run_once(scenario: Scenario, policy: Policy, seed: int, cap: float) -> dict[str, Any]:
    sim = _make_sim(scenario, seed)
    _configure_policy(sim, policy)
    start = time.perf_counter_ns()
    sim.simulator.run(scenario.case.end_time, cap)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    switch = sim.simulator.switch
    completed = sim.simulator.continuous_time >= scenario.case.end_time * (1 - 1e-9)
    return {
        "scenario": scenario.slug, "family": scenario.family, "policy": policy.name,
        "threshold": "" if policy.threshold is None else policy.threshold, "seed": seed,
        "initial_n": scenario.case.initial_n, "elapsed_seconds": elapsed, "completed": completed,
        "timed_out": (not completed) and elapsed >= 0.9 * cap,
        "batch_calls": int(switch.batch_calls), "gillespie_calls": int(switch.gillespie_calls),
        "mode_switches": int(switch.mode_switches),
    }


def geometric_mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(finite)))) if finite else math.nan


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="301,302,303,304,305")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--probe-repeats", type=int, default=15)
    parser.add_argument("--cap-seconds", type=float, default=20.0)
    parser.add_argument("--constants", type=str, default="250,500,1000",
                        help="comma-separated fixed thresholds to test")
    parser.add_argument("--cost-model-scales", type=str, default="1",
                        help="comma-separated multipliers applied to the predicted T_hat. The "
                             "frozen-state oracle excludes setup and switching costs, so the "
                             "end-to-end optimum can sit below the local crossover; scaling "
                             "separates a wrong *level* from a wrong *shape*.")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seeds = [int(p) for p in args.seeds.split(",") if p.strip()]

    training = [HERE / "batch_cost_profile_timings_allg.csv",
                HERE / "batch_cost_profile_timings_allg_seed2.csv"]
    rows_train = tcm.usable(tcm.load_rows(training))
    batch_fit = tcm.fit_cost(rows_train, "batch", horizon=5000.0)
    gillespie_fit = tcm.fit_cost(rows_train, "gillespie", horizon=5000.0)

    print(f"placing {len(BASE_SLUGS)} CRN structures on their own boundaries "
          f"at ratios {TARGET_RATIOS} ...\n")
    scenarios, placement = build_scenarios(probe_repeats=args.probe_repeats, seed=seeds[0])
    print(f"{'base':<32} {'ratio':>6} {'T* probe':>9} {'n':>14}  status")
    for entry in placement:
        print(f"{entry['base']:<32} {entry.get('target_ratio', float('nan')):>6.2g} "
              f"{entry.get('T_star_probe', float('nan')):>9.4g} "
              f"{entry.get('n', float('nan')):>14.6g}  {entry['status']}")
    print(f"\n{len(scenarios)} candidate scenarios\n")

    thresholds = {s.slug: cost_model_threshold(s, batch_fit, gillespie_fit, seeds[0])
                  for s in scenarios}
    constants = [float(v) for v in args.constants.split(",") if v.strip()]
    scales = [float(v) for v in args.cost_model_scales.split(",") if v.strip()]
    policies = [Policy("wallclock_timing", "wallclock", 0, None)]
    policies += [Policy(f"constant_{c:g}", "prospective", 2, c) for c in constants]
    policies += [Policy(f"cost_model_x{s:g}" if s != 1 else "cost_model", "prospective", 2, None)
                 for s in scales]
    scale_of = {(f"cost_model_x{s:g}" if s != 1 else "cost_model"): s for s in scales}
    det = [p.name for p in policies if p.name != "wallclock_timing"]

    def resolved(policy: Policy, scenario: Scenario) -> Policy:
        if policy.name in scale_of:
            return Policy(policy.name, policy.kind, policy.selector,
                          thresholds[scenario.slug] * scale_of[policy.name])
        return policy

    # ---- pilot: drop scenarios that time out or cannot tell the policies apart ----
    print("pilot pass (1 seed, 1 repeat) to screen for feasibility and discrimination ...")
    keep: list[Scenario] = []
    screen: list[dict[str, Any]] = []
    for scenario in scenarios:
        signatures, slow, timed_out = {}, 0.0, False
        for policy in policies:
            row = run_once(scenario, resolved(policy, scenario), seeds[0], args.cap_seconds)
            slow = max(slow, row["elapsed_seconds"])
            timed_out = timed_out or row["timed_out"] or not row["completed"]
            if policy.name in det:
                signatures[policy.name] = (row["batch_calls"], row["gillespie_calls"],
                                           row["mode_switches"])
        distinct = len(set(signatures.values()))
        entry = {"scenario": scenario.slug, "family": scenario.family,
                 "initial_n": scenario.case.initial_n, "T_hat": thresholds[scenario.slug],
                 "slowest_seconds": slow, "distinct_policies": distinct}
        if timed_out:
            entry["status"] = "timed out"
        elif distinct < 2:
            entry["status"] = "policy-equivalent (cannot discriminate)"
        else:
            entry["status"] = "kept"
            keep.append(scenario)
        screen.append(entry)
        print(f"  {scenario.slug:<34} n={scenario.case.initial_n:>11,} "
              f"T_hat={thresholds[scenario.slug]:>6.0f} slowest={slow:>7.3f}s "
              f"distinct={distinct}  {entry['status']}")

    print(f"\nkept {len(keep)} of {len(scenarios)} scenarios")
    for reason in ("timed out", "policy-equivalent (cannot discriminate)"):
        n = sum(1 for e in screen if e["status"] == reason)
        print(f"  dropped {n:>2} : {reason}")
    (args.output.with_name(args.output.stem + "_screen.json")).write_text(
        json.dumps({"placement": placement, "screen": screen,
                    "thresholds": thresholds}, indent=2, default=str), encoding="utf-8")
    if args.pilot_only or not keep:
        print("\npilot-only; stopping")
        return

    # ---- full protocol on the surviving scenarios ----
    est = sum(e["slowest_seconds"] for e in screen if e["status"] == "kept")
    print(f"\nfull run: {len(keep)} scenarios x {len(policies)} policies x {len(seeds)} seeds "
          f"x {args.repeats} repeats  (rough estimate {est * len(seeds) * args.repeats / 60:.1f} min)\n")
    results: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for seed in seeds:
            for scenario in keep:
                for policy in policies:
                    row = run_once(scenario, resolved(policy, scenario), seed, args.cap_seconds)
                    row["repeat"] = repeat
                    results.append(row)
        print(f"  repeat {repeat + 1}/{args.repeats} done", flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote {args.output}\n")

    def summarize(slug: str, policy_name: str) -> float:
        per_seed = []
        for seed in seeds:
            vals = [r["elapsed_seconds"] for r in results if r["scenario"] == slug
                    and r["policy"] == policy_name and r["seed"] == seed and not r["timed_out"]]
            if vals:
                per_seed.append(statistics.median(vals))
        return statistics.median(per_seed) if per_seed else math.nan

    names = [p.name for p in policies]
    print("median seconds (ratio to timing); lower is better\n")
    header = f"{'scenario':<34}" + "".join(f"{n:>22}" for n in names)
    print(header)
    print("-" * len(header))
    ratios = {n: {} for n in names}
    for scenario in keep:
        ref = summarize(scenario.slug, "wallclock_timing")
        line = f"{scenario.slug:<34}"
        for name in names:
            v = summarize(scenario.slug, name)
            ratios[name][scenario.slug] = v / ref if ref > 0 else math.nan
            line += f"{v:>13.5g}({ratios[name][scenario.slug]:>5.2f}x)"
        print(line)

    best = {s.slug: min(summarize(s.slug, n) for n in names
                        if math.isfinite(summarize(s.slug, n))) for s in keep}
    families = sorted({s.family for s in keep})
    print(f"\n{'policy':<20} {'equal-family':>14} {'family-worst':>14} {'worst scenario':>16} "
          f"{'vs best tested':>16} {'worst regret':>14}")
    summary = {}
    for name in names:
        per_family = [geometric_mean([ratios[name][s.slug] for s in keep if s.family == fam])
                      for fam in families]
        reg = {s.slug: summarize(s.slug, name) / best[s.slug] for s in keep}
        reg_family = [geometric_mean([reg[s.slug] for s in keep if s.family == fam])
                      for fam in families]
        summary[name] = {"central": geometric_mean(per_family), "family_worst": max(per_family),
                         "worst_scenario": max(ratios[name].values()),
                         "regret_central": geometric_mean(reg_family),
                         "regret_worst": max(reg.values()),
                         "per_scenario_ratio": ratios[name], "per_scenario_regret": reg}
        s = summary[name]
        print(f"{name:<20} {s['central']:>13.3f}x {s['family_worst']:>13.3f}x "
              f"{s['worst_scenario']:>15.3f}x {s['regret_central']:>15.3f}x "
              f"{s['regret_worst']:>13.3f}x")

    (args.output.with_suffix(".json")).write_text(
        json.dumps({"seeds": seeds, "repeats": args.repeats, "thresholds": thresholds,
                    "families": families, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
