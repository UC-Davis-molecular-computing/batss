"""Does a single global threshold match the timing policy end to end, or does structure matter?

The frozen-state experiments say the break-even exchange rate ``T*`` varies about 15x across CRN
structures, which predicts that no one constant can be right everywhere.  But frozen states measure
only the *local* crossover with setup excluded.  A whole trajectory also pays mode duration, engine
rebuild and switching costs, and it spends most of its time in states nowhere near the boundary --
so a constant can be badly wrong locally and still finish in about the same wall-clock time.

This script settles that directly.  It runs each benchmark scenario end to end under:

* ``wallclock_timing``  -- the adaptive measured-wall-clock policy, used as the reference for
  "optimal".  It is a *proxy* for optimal, not an oracle: it bootstraps from the old proxy rule,
  probes periodically, and needs a 4x measured advantage before overriding.
* ``constant_670``      -- the single exchange rate quoted from one CRN, applied unconditionally.
* ``constant_500``      -- the threshold the earlier pilot selected.
* ``cost_model``        -- a *per-CRN* threshold ``T_hat = C_batch / C_Gillespie`` predicted by
  ``threshold_cost_model.py`` and evaluated at each scenario's initial state.

The last one is a CRN-static approximation of the state-dependent model: ``proxy_threshold`` is a
single float per run, so testing a genuinely state-dependent ``T_hat(x)`` would require implementing
it in Rust.  What it does test is whether the *between-CRN* variation the cost model predicts is
real and worth acting on.

Timing follows the existing harness: construction is excluded, one raw ``BatchSimulator.run`` call
is timed, the median is taken across repeats within a seed and then across seeds.

Run with::

    python benchmark/end_to_end_threshold_check.py --seeds 201,202,203,204,205 --repeats 2
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

import threshold_cost_model as tcm
from switching_policy_comparison import (
    Policy,
    Scenario,
    _configure_policy,
    _make_sim,
    comparison_scenarios,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "end_to_end_threshold_check.csv"
CAP_SECONDS = 60.0


def fit_cost_model(training: Sequence[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = tcm.usable(tcm.load_rows(list(training)))
    return (tcm.fit_cost(rows, "batch", horizon=5000.0),
            tcm.fit_cost(rows, "gillespie", horizon=5000.0))


def predicted_threshold_for(scenario: Scenario, batch_fit: dict[str, Any],
                            gillespie_fit: dict[str, Any], seed: int) -> float:
    """Evaluate T_hat at the scenario's initial state, giving one threshold per CRN."""

    sim = _make_sim(scenario, seed)
    rust = sim.simulator
    order = int(rust.debug_o())
    prospective_n = int(rust.debug_prospective_n())
    expected_length = (
        math.sqrt(math.pi / (2.0 * order * (order + int(rust.debug_g())))) * math.sqrt(prospective_n)
        if prospective_n > 0 and order > 0 else 0.0
    )
    row = {
        "prospective_N": str(prospective_n),
        "q": str(int(rust.debug_q())),
        "o": str(order),
        "g": str(int(rust.debug_g())),
        "score": str(float(rust.prospective_batch_score())),
        "expected_batch_length": str(expected_length),
        "output_branches_B": str(int(rust.debug_output_branches())),
    }
    batch = sum(tcm.batch_features(row)[name] * batch_fit["coefficients"][name]
                for name in tcm.BATCH_FEATURES)
    gillespie = sum(tcm.gillespie_features(row, horizon=5000.0)[name]
                    * gillespie_fit["coefficients"][name]
                    for name in tcm.GILLESPIE_FEATURES)
    return batch / gillespie


def run_once(scenario: Scenario, policy: Policy, seed: int) -> dict[str, Any]:
    sim = _make_sim(scenario, seed)
    _configure_policy(sim, policy)
    start = time.perf_counter_ns()
    sim.simulator.run(scenario.case.end_time, CAP_SECONDS)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    switch = sim.simulator.switch
    completed = sim.simulator.continuous_time >= scenario.case.end_time * (1 - 1e-9)
    total_time = float(switch.batch_continuous_time) + float(switch.gillespie_continuous_time)
    return {
        "scenario": scenario.slug,
        "family": scenario.family,
        "policy": policy.name,
        "threshold": "" if policy.threshold is None else policy.threshold,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "completed": completed,
        "timed_out": (not completed) and elapsed >= 0.9 * CAP_SECONDS,
        "batch_calls": int(switch.batch_calls),
        "gillespie_calls": int(switch.gillespie_calls),
        "mode_switches": int(switch.mode_switches),
        "gillespie_time_fraction": (float(switch.gillespie_continuous_time) / total_time
                                    if total_time > 0 else math.nan),
    }


def geometric_mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(finite)))) if finite else math.nan


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="201,202,203,204,205")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training", type=Path, action="append", default=None,
                        help="timing CSVs the cost model is fit on (default: the two _allg files)")
    args = parser.parse_args(argv)

    training = args.training or [HERE / "batch_cost_profile_timings_allg.csv",
                                 HERE / "batch_cost_profile_timings_allg_seed2.csv"]
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    scenarios = comparison_scenarios()

    print("fitting the cost model on the frozen-state profile matrix ...")
    batch_fit, gillespie_fit = fit_cost_model(training)

    print(f"\n{'scenario':<30} {'initial n':>12} {'T_hat (cost model)':>20}")
    thresholds: dict[str, float] = {}
    for scenario in scenarios:
        value = predicted_threshold_for(scenario, batch_fit, gillespie_fit, seeds[0])
        thresholds[scenario.slug] = value
        print(f"{scenario.slug:<30} {scenario.case.initial_n:>12,} {value:>20.0f}")

    policies = [
        Policy("wallclock_timing", "wallclock", 0, None),
        Policy("constant_670", "prospective", 2, 670.0),
        Policy("constant_500", "prospective", 2, 500.0),
        Policy("cost_model", "prospective", 2, None),  # threshold filled per scenario
    ]

    # Warm up so first-touch costs do not land on whichever policy happens to run first.
    for scenario in scenarios:
        run_once(scenario, policies[0], seeds[0])

    rows: list[dict[str, Any]] = []
    total = len(scenarios) * len(policies) * len(seeds) * args.repeats
    done = 0
    for repeat in range(args.repeats):
        for seed in seeds:
            for scenario in scenarios:
                for policy in policies:
                    if policy.name == "cost_model":
                        policy = Policy(policy.name, policy.kind, policy.selector,
                                        thresholds[scenario.slug])
                    row = run_once(scenario, policy, seed)
                    row["repeat"] = repeat
                    rows.append(row)
                    done += 1
        print(f"  repeat {repeat + 1}/{args.repeats} done ({done}/{total} runs)", flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}\n")

    # median across repeats within a seed, then median across seeds
    def summarize(scenario_slug: str, policy_name: str) -> float:
        per_seed = []
        for seed in seeds:
            values = [r["elapsed_seconds"] for r in rows
                      if r["scenario"] == scenario_slug and r["policy"] == policy_name
                      and r["seed"] == seed and not r["timed_out"]]
            if values:
                per_seed.append(statistics.median(values))
        return statistics.median(per_seed) if per_seed else math.nan

    names = [p.name for p in policies]
    print("Each cell is: median seconds (ratio to timing). Lower is better; timing is 1.00x.\n")
    header = f"{'scenario':<30}" + "".join(f"{n:>24}" for n in names)
    print(header)
    print("-" * len(header))
    ratios: dict[str, dict[str, float]] = {n: {} for n in names}
    for scenario in scenarios:
        reference = summarize(scenario.slug, "wallclock_timing")
        line = f"{scenario.slug:<30}"
        for name in names:
            value = summarize(scenario.slug, name)
            ratio = value / reference if reference > 0 else math.nan
            ratios[name][scenario.slug] = ratio
            line += f"{value:>14.5g} ({ratio:>4.2f}x)"
        print(line)

    # The timing policy is an adaptive baseline, not an oracle -- it can lose to every fixed
    # threshold.  So also score each policy against the best result any tested policy achieved on
    # that scenario, which is the closest thing here to "how far from the achievable optimum".
    best_per_scenario = {
        s.slug: min(summarize(s.slug, n) for n in names
                    if math.isfinite(summarize(s.slug, n)))
        for s in scenarios
    }
    regrets: dict[str, dict[str, float]] = {n: {} for n in names}
    for name in names:
        for s in scenarios:
            value = summarize(s.slug, name)
            regrets[name][s.slug] = value / best_per_scenario[s.slug]

    families = sorted({s.family for s in scenarios})
    print(f"\n{'policy':<22} {'equal-family central':>22} {'family-worst':>14} {'worst scenario':>16}"
          f" {'vs best tested':>16} {'worst regret':>14}")
    summary: dict[str, Any] = {}
    for name in names:
        per_family = [geometric_mean([ratios[name][s.slug] for s in scenarios if s.family == fam])
                      for fam in families]
        central = geometric_mean(per_family)
        worst_family = max(per_family)
        worst_scenario = max(ratios[name].values())
        regret_central = geometric_mean(
            [geometric_mean([regrets[name][s.slug] for s in scenarios if s.family == fam])
             for fam in families])
        regret_worst = max(regrets[name].values())
        summary[name] = {"central": central, "worst_family": worst_family,
                         "worst_scenario": worst_scenario,
                         "regret_vs_best_tested": regret_central,
                         "worst_regret_vs_best_tested": regret_worst,
                         "per_scenario": ratios[name]}
        print(f"{name:<22} {central:>21.3f}x {worst_family:>13.3f}x {worst_scenario:>15.3f}x"
              f" {regret_central:>15.3f}x {regret_worst:>13.3f}x")

    (args.output.with_suffix(".json")).write_text(
        json.dumps({"thresholds": thresholds, "seeds": seeds, "repeats": args.repeats,
                    "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
