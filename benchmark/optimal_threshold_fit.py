"""Fit the cost model to the *end-to-end optimal* threshold instead of the frozen-state break-even.

`threshold_cost_model.py` fits `T_hat = C_batch / C_Gillespie` against `T*`, the break-even measured
from frozen states.  It predicts `T*` well (0.94 correlation out-of-family) but loses end to end
unless the prediction is halved, and that 0.5 is a fitted constant with no established mechanism.

The likely reason is that `T*` answers the wrong question.  It says which engine is cheaper *at this
instant*, whereas a policy needs the fixed threshold that minimises *total wall-clock* over a whole
trajectory.  This experiment measures the latter directly and asks how it relates to `T_hat`:

    T_opt = argmin_T  wallclock( run(scenario, threshold = T) )
    alpha = T_opt / T_hat

* If `alpha` is constant across CRN families, the 0.5 is a real universal calibration constant and
  can be shipped deliberately rather than as a fudge.
* If `alpha` varies structurally, the cost model is missing a term, and the residuals say which
  feature it tracks.

Two things make the measurement non-trivial, and both are handled here rather than averaged over:

1. **Thresholds are not a continuous control.**  If two thresholds never straddle a score the
   trajectory encounters, they execute the identical mode sequence and consume the identical random
   stream, so timing differences between them are pure noise.  The sweep therefore first groups
   thresholds into equivalence classes by mode signature, and times each class once.  The optimum is
   consequently an *interval* of thresholds, not a point; `alpha` is reported as the geometric
   centre of the winning interval, with the interval itself recorded.
2. **The grid is relative to each CRN's own `T_hat`**, so the sweep resolves `alpha` directly and
   spends its samples where each scenario's decision actually changes.

Run with::

    python benchmark/optimal_threshold_fit.py --seeds 801,802,803,804 --repeats 2
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
from policy_matrix_experiment import build_scenarios, cost_model_threshold
from switching_policy_comparison import Policy, Scenario, _configure_policy, _make_sim

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "optimal_threshold_fit.csv"

# Multipliers of each scenario's own T_hat. Log-spaced by ~1.4, spanning an eightfold range below
# T_hat and a fourfold range above, which brackets both the fitted 0.5 and the unscaled model.
ALPHAS = (0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)


def run_once(scenario: Scenario, threshold: float, seed: int, cap: float) -> dict[str, Any]:
    sim = _make_sim(scenario, seed)
    _configure_policy(sim, Policy(f"T{threshold:g}", "prospective", 2, threshold))
    start = time.perf_counter_ns()
    sim.simulator.run(scenario.case.end_time, cap)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    switch = sim.simulator.switch
    completed = sim.simulator.continuous_time >= scenario.case.end_time * (1 - 1e-9)
    return {
        "elapsed_seconds": elapsed,
        "timed_out": (not completed) and elapsed >= 0.9 * cap,
        # The mode sequence a run executed. Two thresholds with the same signature ran identically.
        "signature": (int(switch.batch_calls), int(switch.gillespie_calls),
                      int(switch.mode_switches)),
    }


def geometric_mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(finite)))) if finite else math.nan


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="801,802,803,804")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--probe-repeats", type=int, default=9)
    parser.add_argument("--cap-seconds", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="classes within this fraction of the fastest count as tied, since the "
                             "measured timing floor is comparable to the gaps between them")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    seeds = [int(p) for p in args.seeds.split(",") if p.strip()]

    training = [HERE / "batch_cost_profile_timings_allg.csv",
                HERE / "batch_cost_profile_timings_allg_seed2.csv"]
    rows_train = tcm.usable(tcm.load_rows(training))
    batch_fit = tcm.fit_cost(rows_train, "batch", horizon=5000.0)
    gillespie_fit = tcm.fit_cost(rows_train, "gillespie", horizon=5000.0)

    scenarios, _ = build_scenarios(probe_repeats=args.probe_repeats, seed=seeds[0])
    t_hat = {s.slug: cost_model_threshold(s, batch_fit, gillespie_fit, seeds[0]) for s in scenarios}
    print(f"{len(scenarios)} scenarios; sweeping {len(ALPHAS)} multipliers of each one's T_hat\n")

    records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        th = t_hat[scenario.slug]
        # --- pass 1: which multipliers are actually distinct policies? ---
        signatures: dict[float, tuple[int, int, int]] = {}
        censored = False
        for alpha in ALPHAS:
            row = run_once(scenario, alpha * th, seeds[0], args.cap_seconds)
            if row["timed_out"]:
                censored = True
                break
            signatures[alpha] = row["signature"]
        if censored or len(signatures) < len(ALPHAS):
            print(f"  {scenario.slug:<34} SKIPPED (timed out during classification)")
            continue

        classes: dict[tuple[int, int, int], list[float]] = {}
        for alpha, sig in signatures.items():
            classes.setdefault(sig, []).append(alpha)
        if len(classes) < 2:
            print(f"  {scenario.slug:<34} SKIPPED (one policy across the whole sweep)")
            continue

        # --- pass 2: time one representative per class, properly ---
        timings: dict[tuple[int, int, int], float] = {}
        for sig, alphas in classes.items():
            rep = alphas[len(alphas) // 2]
            per_seed = []
            for seed in seeds:
                vals = [run_once(scenario, rep * th, seed, args.cap_seconds)["elapsed_seconds"]
                        for _ in range(args.repeats)]
                per_seed.append(statistics.median(vals))
            timings[sig] = statistics.median(per_seed)
            records.append({"scenario": scenario.slug, "family": scenario.family,
                            "T_hat": th, "alpha_rep": rep, "alphas_in_class": len(alphas),
                            "batch_calls": sig[0], "gillespie_calls": sig[1],
                            "mode_switches": sig[2], "median_seconds": timings[sig]})

        best_sig = min(timings, key=lambda s: timings[s])
        best_time = timings[best_sig]
        # Picking the strict argmin over ~8 classes would be noise-dominated: the timing floor is
        # about 10% and neighbouring classes often differ by less than that. Treat every class
        # within `tol` of the best as tied, and report the optimum as the span of multipliers those
        # tied classes cover. That is the honest statement -- the optimum is a plateau, and its
        # width is itself the interesting quantity.
        tied = [sig for sig, t in timings.items() if t <= best_time * (1.0 + args.tolerance)]
        best_alphas = sorted(a for sig in tied for a in classes[sig])
        runner_up = sorted(timings.values())[1] if len(timings) > 1 else math.inf
        alpha_opt = math.sqrt(best_alphas[0] * best_alphas[-1])
        edge = best_alphas[0] == ALPHAS[0] or best_alphas[-1] == ALPHAS[-1]
        worst = max(timings.values())
        results.append({
            "scenario": scenario.slug, "family": scenario.family, "T_hat": th,
            "alpha_opt": alpha_opt, "alpha_lo": best_alphas[0], "alpha_hi": best_alphas[-1],
            "T_opt": alpha_opt * th, "distinct_policies": len(classes),
            "best_seconds": best_time, "worst_seconds": worst,
            "spread": worst / best_time, "edge_censored": edge,
            "tied_classes": len(tied), "runner_up_ratio": runner_up / best_time,
        })
        print(f"  {scenario.slug:<34} T_hat={th:>6.0f}  policies={len(classes):>2}  "
              f"alpha_opt={alpha_opt:>5.2f} [{best_alphas[0]:g},{best_alphas[-1]:g}] "
              f"tied={len(tied)}  spread={worst / best_time:>5.2f}x"
              f"  2nd={runner_up / best_time:>4.2f}x{'  EDGE' if edge else ''}")

    if not results:
        raise SystemExit("no scenario produced a usable optimum")

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nwrote {args.output}")

    # ---------------- analysis ----------------
    usable = [r for r in results if not r["edge_censored"]]
    alphas = np.array([r["alpha_opt"] for r in usable])
    ths = np.array([r["T_hat"] for r in usable])
    print(f"\n=== is alpha = T_opt / T_hat constant? ({len(usable)} uncensored of "
          f"{len(results)} scenarios) ===")
    print(f"  geometric mean alpha : {geometric_mean(alphas):.3f}")
    print(f"  10-90 percentile     : {np.percentile(alphas, 10):.3f} .. "
          f"{np.percentile(alphas, 90):.3f}")
    print(f"  full range           : {alphas.min():.3f} .. {alphas.max():.3f}")
    print(f"  spread (max/min)     : {alphas.max() / alphas.min():.2f}x")

    print("\n  per family:")
    for fam in sorted({r["family"] for r in usable}):
        sel = [r["alpha_opt"] for r in usable if r["family"] == fam]
        print(f"    {fam:<22} n={len(sel):<3} geo-mean alpha = {geometric_mean(sel):.3f}"
              f"   ({min(sel):.2f} .. {max(sel):.2f})")

    # A constant alpha means log T_opt = log alpha + 1.0 * log T_hat. A slope away from 1 means the
    # model is mis-scaled with magnitude, not merely offset.
    design = np.column_stack([np.ones(len(usable)), np.log(ths)])
    coef, *_ = np.linalg.lstsq(design, np.log(alphas * ths), rcond=None)
    resid = np.log(alphas * ths) - design @ coef
    print(f"\n  log T_opt = {coef[0]:.3f} + {coef[1]:.3f} * log T_hat")
    print(f"    slope 1.0 would mean a pure constant offset (alpha independent of magnitude)")
    print(f"    residual sd = {resid.std():.3f} in log units "
          f"(= {math.exp(resid.std()):.2f}x multiplicative)")

    print("\n=== does alpha track any structural feature? (corr of log alpha vs feature) ===")
    feats = {}
    for r in usable:
        sim = _make_sim(next(s for s in scenarios if s.slug == r["scenario"]), seeds[0])
        rust = sim.simulator
        feats.setdefault("q", []).append(int(rust.debug_q()))
        feats.setdefault("o", []).append(int(rust.debug_o()))
        feats.setdefault("g", []).append(int(rust.debug_g()))
        feats.setdefault("B", []).append(int(rust.debug_output_branches()))
        feats.setdefault("log2_N", []).append(math.log2(int(rust.debug_prospective_n())))
        feats.setdefault("log_T_hat", []).append(math.log(r["T_hat"]))
    log_alpha = np.log(alphas)
    for name, vals in feats.items():
        v = np.array(vals, dtype=float)
        if v.std() == 0:
            print(f"    {name:<12} (no variation across the surviving scenarios -- "
                  f"all = {v[0]:g}, so nothing can be learned about it here)")
            continue
        c = float(np.corrcoef(log_alpha, v)[0, 1])
        flag = "   <-- systematic" if abs(c) > 0.5 else ""
        print(f"    {name:<12} corr = {c:+.3f}{flag}")

    widths = np.array([r["alpha_hi"] / r["alpha_lo"] for r in usable])
    print(f"\n=== how well determined is each optimum? ===")
    print(f"  plateau width (alpha_hi/alpha_lo): median {np.median(widths):.2f}x, "
          f"max {widths.max():.2f}x")
    print(f"  runner-up within {100 * args.tolerance:.0f}% of best in "
          f"{sum(1 for r in usable if r['runner_up_ratio'] <= 1 + args.tolerance)}"
          f"/{len(usable)} scenarios")
    print(f"  best-vs-worst policy spread: median "
          f"{np.median([r['spread'] for r in usable]):.2f}x  "
          f"(if this were ~1x, the threshold would not matter at all)")

    args.output.with_suffix(".json").write_text(json.dumps(
        {"alphas": ALPHAS, "seeds": seeds, "repeats": args.repeats, "results": results,
         "geometric_mean_alpha": geometric_mean(alphas),
         "log_fit": {"intercept": coef[0], "slope": coef[1], "residual_sd": float(resid.std())}},
        indent=2), encoding="utf-8")
    print(f"\nwrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
