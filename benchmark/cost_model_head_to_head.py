"""How long does a CRN actually take to run, under each switching policy, as really implemented?

Every earlier comparison fed a *fixed* threshold to the Rust selector, computed once in Python from
the initial state. This one uses `HEURISTIC_COST_MODEL` (selector 3), which recomputes the threshold
from the live configuration at every decision -- the production form. The only thing Python supplies
is the fitted coefficient vector.

The measured quantity is the one that matters: wall-clock seconds for one `BatchSimulator.run` to
reach `t_max`. Construction is excluded; the timer covers the run call alone.

Policies raced:

* ``timing``          -- the adaptive wall-clock policy, the incumbent default
* ``constant_250`` / ``constant_500`` -- the best fixed thresholds found earlier
* ``cost_model_cold`` -- coefficients fitted on cold-measured costs, scaled by the empirical 0.368
* ``cost_model_warm`` -- coefficients fitted on warm-measured costs, unscaled
* ``cost_model_warm_scaled`` -- warm coefficients with the residual correction still applied

The cold/warm pair is the point of the experiment. Fitted costs were historically measured one call
at a time on a freshly built simulator, i.e. always a *first* call on cold data structures. Batch
pays far more for that than Gillespie (measured: batch 0.617x warm/cold, Gillespie 0.942x), which
inflates the fitted ratio. If warm coefficients need less correction to win, the scale factor was
mostly a measurement artifact rather than a property of the switching problem.

Run with::

    python benchmark/cost_model_head_to_head.py --seeds 901,902,903,904 --repeats 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import threshold_cost_model as tcm
from policy_matrix_experiment import build_scenarios
from switching_policy_comparison import Scenario, _make_sim, comparison_scenarios

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "cost_model_head_to_head.csv"

HEURISTIC_WALLCLOCK = 0
HEURISTIC_PROSPECTIVE = 2
HEURISTIC_COST_MODEL = 3


@dataclass(frozen=True)
class Contender:
    name: str
    selector: int
    threshold: float | None = None
    batch_coefficients: tuple[float, ...] | None = None
    gillespie_coefficients: tuple[float, ...] | None = None
    scale: float = 1.0
    # Tuning for the adaptive wall-clock policy. Left as None the simulator keeps its shipped
    # defaults; the important one is override_factor, whose default of 4.0 means the policy follows
    # the old proxy rule unless the alternative measures more than 4x cheaper -- so it is not a
    # "pick the cheaper engine" policy at all, and is a poor stand-in for optimal until tuned.
    override_factor: float | None = None
    probe_interval: int | None = None
    probe_interval_committed: int | None = None
    ema_alpha: float | None = None


def fitted_coefficients(paths: Sequence[Path]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rows = tcm.usable(tcm.load_rows(list(paths)))
    batch = tcm.fit_cost(rows, "batch", horizon=5000.0)
    gillespie = tcm.fit_cost(rows, "gillespie", horizon=5000.0)
    return (tuple(batch["coefficients"][n] for n in tcm.BATCH_FEATURES),
            tuple(gillespie["coefficients"][n] for n in tcm.GILLESPIE_FEATURES))


def run_once(scenario: Scenario, contender: Contender, seed: int, cap: float) -> dict[str, Any]:
    """Time one full run under one policy.

    A run can also *panic*: the collision sampler carries a precision assertion that fires at large
    populations (observed at n = 23,394,833 on Lotka-Volterra, with a claimed error of 6.1e-7). That
    is a simulator bug rather than a harness one, but a panic reaching Python as a
    ``PanicException`` would otherwise abort a multi-hour sweep at whatever scenario happened to
    trigger it, losing every later measurement. Record it as a failed run and carry on, so the
    failure is visible in the output instead of destroying it.
    """

    try:
        return _run_once_inner(scenario, contender, seed, cap)
    except BaseException as exc:  # PanicException does not derive from Exception
        if type(exc).__name__ not in ("PanicException", "RuntimeError"):
            raise
        return {
            "scenario": scenario.slug, "family": scenario.family, "policy": contender.name,
            "seed": seed, "initial_n": scenario.case.initial_n,
            "elapsed_seconds": math.nan, "completed": False, "timed_out": False,
            "panicked": True, "error": " ".join(str(exc).split())[:400],
            "batch_calls": 0, "gillespie_calls": 0, "mode_switches": 0,
        }


def _run_once_inner(scenario: Scenario, contender: Contender, seed: int,
                    cap: float) -> dict[str, Any]:
    sim = _make_sim(scenario, seed)
    rust = sim.simulator
    rust.heuristic_gillespie_switching = contender.selector
    if contender.threshold is not None:
        rust.proxy_threshold = contender.threshold
    if contender.batch_coefficients is not None:
        rust.cost_model_batch_coefficients = list(contender.batch_coefficients)
        rust.cost_model_gillespie_coefficients = list(contender.gillespie_coefficients)
        rust.cost_model_scale = contender.scale
    # Set these on the simulator, not on `rust.switch`: that property hands back a snapshot, so
    # assigning to its fields would be silently discarded and the policy would run untuned.
    if contender.override_factor is not None:
        rust.wdt_override_factor = contender.override_factor
    if contender.probe_interval is not None:
        rust.wdt_probe_interval = contender.probe_interval
    if contender.probe_interval_committed is not None:
        rust.wdt_probe_interval_committed = contender.probe_interval_committed
    if contender.ema_alpha is not None:
        rust.wdt_ema_alpha = contender.ema_alpha
    start = time.perf_counter_ns()
    rust.run(scenario.case.end_time, cap)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    switch = rust.switch
    completed = rust.continuous_time >= scenario.case.end_time * (1 - 1e-9)
    return {
        "scenario": scenario.slug, "family": scenario.family, "policy": contender.name,
        "seed": seed, "initial_n": scenario.case.initial_n,
        "elapsed_seconds": elapsed, "completed": completed,
        "timed_out": (not completed) and elapsed >= 0.9 * cap,
        "panicked": False, "error": "",
        "batch_calls": int(switch.batch_calls), "gillespie_calls": int(switch.gillespie_calls),
        "mode_switches": int(switch.mode_switches),
    }


def geometric_mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(finite)))) if finite else math.nan


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="901,902,903,904")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--probe-repeats", type=int, default=9)
    parser.add_argument("--cap-seconds", type=float, default=10.0)
    parser.add_argument("--residual-scale", type=float, default=0.56,
                        help="correction still applied to the warm model, if the warm measurement "
                             "only accounts for part of the empirical scale factor")
    parser.add_argument("--include-natural", action="store_true",
                        help="also race the CRNs at their natural populations. Boundary "
                             "placement drops every order-3 CRN, so this is the only way "
                             "to get end-to-end o=3 coverage, though those runs do not sit "
                             "near their own break-even")
    parser.add_argument("--skip-slow", action="store_true",
                        help="drop scenarios that hit the time cap; they contribute nothing "
                             "to the analysis but cost cap-seconds on every run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seeds = [int(p) for p in args.seeds.split(",") if p.strip()]

    cold_b, cold_g = fitted_coefficients([HERE / "batch_cost_profile_timings_allg.csv",
                                          HERE / "batch_cost_profile_timings_allg_seed2.csv"])
    warm_b, warm_g = fitted_coefficients([HERE / "batch_cost_profile_timings_warm.csv",
                                          HERE / "batch_cost_profile_timings_warm_seed2.csv"])

    contenders = [
        Contender("timing", HEURISTIC_WALLCLOCK),
        Contender("constant_250", HEURISTIC_PROSPECTIVE, threshold=250.0),
        Contender("constant_500", HEURISTIC_PROSPECTIVE, threshold=500.0),
        Contender("cost_model_cold", HEURISTIC_COST_MODEL, batch_coefficients=cold_b,
                  gillespie_coefficients=cold_g, scale=0.368),
        Contender("cost_model_warm", HEURISTIC_COST_MODEL, batch_coefficients=warm_b,
                  gillespie_coefficients=warm_g, scale=1.0),
        Contender(f"cost_model_warm_x{args.residual_scale:g}", HEURISTIC_COST_MODEL,
                  batch_coefficients=warm_b, gillespie_coefficients=warm_g,
                  scale=args.residual_scale),
    ]

    scenarios, _ = build_scenarios(probe_repeats=args.probe_repeats, seed=seeds[0])
    if args.include_natural:
        # Boundary placement drops every order-3 CRN: Brusselator cannot reach its break-even below
        # the population cap, and the collision o=3 family times out there. Their *natural*
        # scenarios still run fine, so including them buys end-to-end order-3 coverage, at the cost
        # of those runs not sitting near their own boundary.
        placed = {s.case.spec.name for s in scenarios}
        scenarios = scenarios + [s for s in comparison_scenarios()
                                 if s.case.spec.name not in placed]
    if args.skip_slow:
        # A scenario that hits the cap under any policy contributes nothing to the analysis but
        # costs cap-seconds on every one of its runs. Drop it once rather than repeatedly.
        kept = []
        for scenario in scenarios:
            probe = run_once(scenario, contenders[0], seeds[0], args.cap_seconds)
            if probe["timed_out"]:
                print(f"  dropping {scenario.slug}: exceeds the {args.cap_seconds:g}s cap")
            else:
                kept.append(scenario)
        scenarios = kept
    print(f"\n{len(scenarios)} scenarios x {len(contenders)} policies x {len(seeds)} seeds "
          f"x {args.repeats} repeats")
    orders = sorted({int(_make_sim(s, seeds[0]).simulator.debug_o()) for s in scenarios})
    print(f"reaction orders covered: {orders}\n")

    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for seed in seeds:
            for scenario in scenarios:
                for contender in contenders:
                    row = run_once(scenario, contender, seed, args.cap_seconds)
                    row["repeat"] = repeat
                    rows.append(row)
        print(f"  repeat {repeat + 1}/{args.repeats} done", flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.output}\n")

    names = [c.name for c in contenders]

    def summarize(slug: str, policy: str) -> float:
        per_seed = []
        for seed in seeds:
            vals = [r["elapsed_seconds"] for r in rows if r["scenario"] == slug
                    and r["policy"] == policy and r["seed"] == seed and not r["timed_out"]]
            if vals:
                per_seed.append(statistics.median(vals))
        return statistics.median(per_seed) if per_seed else math.nan

    usable = [s for s in scenarios
              if all(math.isfinite(summarize(s.slug, n)) for n in names)]
    print(f"{len(usable)} scenarios completed under every policy "
          f"({len(scenarios) - len(usable)} dropped for timing out)\n")

    header = f"{'scenario':<32}" + "".join(f"{n[:20]:>21}" for n in names)
    print(header)
    print("-" * len(header))
    ratios = {n: {} for n in names}
    for s in usable:
        ref = summarize(s.slug, "timing")
        line = f"{s.slug:<32}"
        for n in names:
            v = summarize(s.slug, n)
            ratios[n][s.slug] = v / ref
            line += f"{v:>13.5g}({ratios[n][s.slug]:>5.2f}x)"
        print(line)

    best = {s.slug: min(summarize(s.slug, n) for n in names) for s in usable}
    families = sorted({s.family for s in usable})
    print(f"\n{'policy':<26} {'equal-family':>14} {'family-worst':>14} {'worst scenario':>16}"
          f" {'vs best tested':>16} {'worst regret':>14}")
    summary = {}
    for n in names:
        per_family = [geometric_mean([ratios[n][s.slug] for s in usable if s.family == fam])
                      for fam in families]
        reg = {s.slug: summarize(s.slug, n) / best[s.slug] for s in usable}
        reg_family = [geometric_mean([reg[s.slug] for s in usable if s.family == fam])
                      for fam in families]
        summary[n] = {"central": geometric_mean(per_family), "family_worst": max(per_family),
                      "worst_scenario": max(ratios[n].values()),
                      "regret_central": geometric_mean(reg_family),
                      "regret_worst": max(reg.values())}
        v = summary[n]
        print(f"{n:<26} {v['central']:>13.3f}x {v['family_worst']:>13.3f}x "
              f"{v['worst_scenario']:>15.3f}x {v['regret_central']:>15.3f}x "
              f"{v['regret_worst']:>13.3f}x")

    args.output.with_suffix(".json").write_text(json.dumps(
        {"seeds": seeds, "repeats": args.repeats, "summary": summary,
         "cold_batch": cold_b, "cold_gillespie": cold_g,
         "warm_batch": warm_b, "warm_gillespie": warm_g}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
