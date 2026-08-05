"""Are the fitted engine costs measured cold, while a real run pays warm ones?

`profile_batch_costs.py` builds a **fresh simulator for every timed call**, so each measurement is
the first engine call on brand-new data structures: cold caches, untrained branch predictor,
first-touch page faults on the urn and the transition arrays. A real run pays those once and then
executes thousands of warm calls.

That matters because the fitted threshold is a *ratio*. Batch touches large structures (transition
arrays, the urn, `q^o` terminal lanes) while Gillespie touches a small dense rate vector, so if
batch suffers more from a cold start then `C_batch / C_Gillespie` is over-measured, `T*` is
over-estimated, and the empirically optimal policy threshold sits below it -- which is the sign and
roughly the magnitude of the unexplained scale factor.

Protocol, per CRN and per engine:

* **cold** -- the existing protocol: one timed call per freshly constructed simulator.
* **warm** -- many calls on a *single* simulator, discarding the first few and taking the median of
  the rest.

The confound is that repeated calls advance the trajectory, so the state (and hence the true cost)
drifts. It is kept small by choosing large populations, where one call changes a vanishing fraction
of the population, and by reporting the warm series so drift is visible rather than assumed away.

Run with::

    python benchmark/warm_vs_cold_engine_cost.py --cold-repeats 61 --warm-calls 60 --discard 8
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import profile_batch_costs as pbc
from threshold_model import _make_sim

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "warm_vs_cold_engine_cost.csv"

CASES = (
    "shrinking", "shrinking_b_decay_channels_8",
    "dense_support_q4", "dense_support_q12",
    "collision_o2_g0_n100000000", "collision_o3_g0_n100000000",
    "dimerization", "lotka_volterra", "oregonator", "rossler", "brusselator",
)


def _fresh(profile_case: pbc.ProfileCase, seed: int) -> Any:
    case = profile_case.case
    return _make_sim(case, case.spec.inits_from_n(case.initial_n), seed)


def cold_series(profile_case: pbc.ProfileCase, *, gillespie: bool, reactions: int,
                repeats: int, seed: int) -> list[float]:
    """One timed call per fresh simulator -- exactly what the fitted costs were measured with."""
    out = []
    for i in range(repeats):
        sim = _fresh(profile_case, pbc._stable_seed(seed, "cold", i))
        r = sim.simulator.benchmark_engine_call(gillespie, reactions if gillespie else None)
        secs = r.engine_seconds + r.postprocess_seconds
        out.append(secs / r.total_reactions * 1e6 if gillespie else secs * 1e6)
    return out


def warm_series(profile_case: pbc.ProfileCase, *, gillespie: bool, reactions: int,
                calls: int, seed: int) -> list[float]:
    """Many calls on ONE simulator -- what a real run actually executes."""
    sim = _fresh(profile_case, pbc._stable_seed(seed, "warm"))
    out = []
    for _ in range(calls):
        r = sim.simulator.benchmark_engine_call(gillespie, reactions if gillespie else None)
        secs = r.engine_seconds + r.postprocess_seconds
        if r.total_reactions <= 0:
            break
        out.append(secs / r.total_reactions * 1e6 if gillespie else secs * 1e6)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-repeats", type=int, default=61)
    parser.add_argument("--warm-calls", type=int, default=60)
    parser.add_argument("--discard", type=int, default=8,
                        help="warm calls dropped before taking the median, to exclude the warm-up")
    parser.add_argument("--gillespie-reactions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    by_slug = {c.case.slug: c for c in pbc.profile_cases()}
    rows: list[dict[str, Any]] = []
    print(f"{'case':<32} {'batch cold':>11} {'batch warm':>11} {'gill cold':>11} "
          f"{'gill warm':>11} {'T* cold':>9} {'T* warm':>9} {'warm/cold':>10}")
    print("-" * 112)
    for slug in CASES:
        base = by_slug.get(slug)
        if base is None:
            continue
        pc = replace(base, measure_gillespie=True)
        try:
            bc = cold_series(pc, gillespie=False, reactions=0,
                             repeats=args.cold_repeats, seed=args.seed)
            bw = warm_series(pc, gillespie=False, reactions=0,
                             calls=args.warm_calls, seed=args.seed)
            gc = cold_series(pc, gillespie=True, reactions=args.gillespie_reactions,
                             repeats=max(9, args.cold_repeats // 5), seed=args.seed)
            gw = warm_series(pc, gillespie=True, reactions=args.gillespie_reactions,
                             calls=max(9, args.warm_calls // 3), seed=args.seed)
        except Exception as exc:
            print(f"{slug:<32} FAILED: {exc!r}")
            continue
        if len(bw) <= args.discard or len(gw) <= 2:
            print(f"{slug:<32} SKIPPED (warm series too short: batch {len(bw)}, gill {len(gw)})")
            continue

        b_cold, g_cold = statistics.median(bc), statistics.median(gc)
        b_warm = statistics.median(bw[args.discard:])
        g_warm = statistics.median(gw[min(2, len(gw) - 1):])
        t_cold, t_warm = b_cold / g_cold, b_warm / g_warm
        rows.append({"case": slug, "batch_cold_us": b_cold, "batch_warm_us": b_warm,
                     "gillespie_cold_us": g_cold, "gillespie_warm_us": g_warm,
                     "T_star_cold": t_cold, "T_star_warm": t_warm,
                     "warm_over_cold": t_warm / t_cold,
                     "batch_first_call_us": bw[0], "batch_warm_plateau_us": b_warm,
                     "warm_calls_used": len(bw)})
        print(f"{slug:<32} {b_cold:>11.3f} {b_warm:>11.3f} {g_cold:>11.5f} {g_warm:>11.5f} "
              f"{t_cold:>9.0f} {t_warm:>9.0f} {t_warm / t_cold:>10.3f}")

    if not rows:
        raise SystemExit("no case produced a usable comparison")
    with args.output.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.output}")

    ratios = [r["warm_over_cold"] for r in rows]
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"\n=== does measuring warm lower the break-even? ===")
    print(f"  geometric mean T*_warm / T*_cold = {geo:.3f}   (range {min(ratios):.3f} .. "
          f"{max(ratios):.3f})")
    print(f"  the unexplained end-to-end scale factor is alpha = 0.368 [0.304, 0.445]")
    if geo <= 0.55:
        print("  -> warm measurement accounts for most of it: T* should be measured warm")
    elif geo <= 0.85:
        print("  -> warm measurement accounts for PART of it; something else remains")
    else:
        print("  -> warm measurement does NOT explain it; this hypothesis is refuted too")

    print(f"\n=== how much of the warm-up is in the first call? (batch) ===")
    print(f"{'case':<32} {'first call':>12} {'plateau':>10} {'first/plateau':>14}")
    for r in rows:
        print(f"{r['case']:<32} {r['batch_first_call_us']:>12.3f} "
              f"{r['batch_warm_plateau_us']:>10.3f} "
              f"{r['batch_first_call_us'] / r['batch_warm_plateau_us']:>14.2f}x")


if __name__ == "__main__":
    main()
