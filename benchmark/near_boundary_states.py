"""Build frozen states that sit *near* their own batch/Gillespie break-even, and time both engines.

Motivation.  The profile matrix in ``profile_batch_costs.py`` was designed to isolate engine costs,
not to sit near the switching boundary.  Only 8 of its 37 states lie within 2x of their break-even;
the rest are 10x-800x away, where every candidate threshold in a wide band makes the same decision.
Decision **regret** is therefore degenerate on that matrix -- the separate-cost ratio, ``T_logN``,
``T = 500`` and even the oracle ``T*`` all score exactly 1.0000 -- so regret cannot rank predictors.

This module fixes that.  For each base CRN it solves for the population at which the prospective
score lands at a requested multiple of the measured break-even, then measures both engines there at
full fidelity.  The resulting states are a **held-out decision test set**: fit the cost model on the
original profile matrix, then evaluate regret here, where being wrong actually costs something.

The search is exact rather than iterative-by-luck.  ``inits_from_n`` scales every initial count
proportionally and ``_make_sim`` sets ``volume = initial_n``, so composition -- and hence the active
probability -- is invariant under scaling ``n``.  Since ``E[L] proportional to sqrt(N)``, the score
obeys ``score(n) = score(n_ref) sqrt(n / n_ref)`` exactly (verified to 4 significant figures).
Solving ``score(n) = ratio * T*`` therefore gives

    n = n_ref * (ratio * T* / score(n_ref))^2

and only ``T*`` needs re-measuring, because it drifts weakly (logarithmically) with ``N``.

Run with::

    python benchmark/near_boundary_states.py --repeats 301 --seed 1 \
        --output benchmark/near_boundary_timings.csv
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import profile_batch_costs as pbc
from threshold_model import _make_sim

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "near_boundary_timings.csv"

# Population bounds.  batss reaches about 1e10; below 1e3 the batch engine is meaningless.
N_MIN = 1_000
N_MAX = 10**10

# Bracket the boundary from both sides so a predictor that is merely biased high or low is caught.
TARGET_RATIOS = (0.6, 1.0, 1.6)

# One representative case per structural axis, so the test set spans the same features the cost
# model claims to explain without simply re-timing all 37 training states.
BASE_SLUGS = (
    "collision_o2_g0_n1000000",
    "collision_o2_g2_n1000000",
    "collision_o3_g0_n1000000",
    "collision_o3_g2_n1000000",
    "dense_support_q4",
    "dense_support_q8",
    "dense_support_q12",
    "shrinking_b_decay_channels_1",
    "shrinking_b_decay_channels_4",
    "shrinking_b_decay_channels_8",
    "shrinking",
    "shrinking_no_b_decay",
    "shrinking_split_b_decay",
    "dimerization",
    "lotka_volterra",
    "rossler",
    "oregonator",
    "brusselator",
)


def _score_at(profile_case: pbc.ProfileCase, seed: int) -> float:
    case = profile_case.case
    config = case.spec.inits_from_n(case.initial_n)
    sim = _make_sim(case, config, seed)
    return float(sim.simulator.prospective_batch_score())


def _measure_t_star(profile_case: pbc.ProfileCase, *, repeats: int, seed: int,
                    gillespie_reactions: int) -> float:
    """Median paired batch/Gillespie cost ratio, using the same pairing as ``collect_timings``."""

    batch_us: list[float] = []
    gillespie_us: list[float] = []
    pbc._benchmark_once(profile_case, seed=pbc._stable_seed(seed, "warm"), gillespie=False, reactions=0)
    pbc._benchmark_once(profile_case, seed=pbc._stable_seed(seed, "warm"), gillespie=True,
                        reactions=gillespie_reactions)
    for repeat in range(repeats):
        trial_seed = pbc._stable_seed(seed, profile_case.case.slug, repeat)
        # Alternate order so monotonic clock drift cannot bias the ratio in one direction.
        if repeat % 2:
            gill = pbc._benchmark_once(profile_case, seed=trial_seed, gillespie=True,
                                       reactions=gillespie_reactions)
            batch = pbc._benchmark_once(profile_case, seed=trial_seed, gillespie=False, reactions=0)
        else:
            batch = pbc._benchmark_once(profile_case, seed=trial_seed, gillespie=False, reactions=0)
            gill = pbc._benchmark_once(profile_case, seed=trial_seed, gillespie=True,
                                       reactions=gillespie_reactions)
        batch_us.append((batch.engine_seconds + batch.postprocess_seconds) * 1e6)
        if gill.total_reactions > 0:
            gillespie_us.append(
                (gill.engine_seconds + gill.postprocess_seconds) / gill.total_reactions * 1e6
            )
    if not gillespie_us:
        return math.nan
    return statistics.median(batch_us) / statistics.median(gillespie_us)


def _with_population(profile_case: pbc.ProfileCase, n: int, slug: str) -> pbc.ProfileCase:
    return replace(
        profile_case,
        case=replace(profile_case.case, slug=slug, initial_n=int(n)),
        measure_gillespie=True,
    )


def solve_population(profile_case: pbc.ProfileCase, ratio: float, *, probe_repeats: int,
                     seed: int, gillespie_reactions: int, rounds: int = 3) -> dict[str, Any]:
    """Find the population whose prospective score is ``ratio`` times its break-even ``T*``."""

    n = float(profile_case.case.initial_n)
    clamped = False
    t_star = math.nan
    score = math.nan
    for _ in range(rounds):
        probe = _with_population(profile_case, int(n), profile_case.case.slug + "_probe")
        score = _score_at(probe, pbc._stable_seed(seed, "score", int(n)))
        t_star = _measure_t_star(probe, repeats=probe_repeats, seed=seed,
                                 gillespie_reactions=gillespie_reactions)
        if not (math.isfinite(score) and math.isfinite(t_star)) or score <= 0:
            break
        # score scales exactly as sqrt(n), so this solve is a single exact Newton step in log n.
        target = n * (ratio * t_star / score) ** 2
        bounded = min(max(target, N_MIN), N_MAX)
        clamped = bounded != target
        if abs(math.log10(bounded / n)) < 0.01:
            n = bounded
            break
        n = bounded
        if clamped:
            break

    probe = _with_population(profile_case, int(n), profile_case.case.slug + "_probe")
    score = _score_at(probe, pbc._stable_seed(seed, "score", int(n)))
    achieved = score / t_star if math.isfinite(t_star) and t_star > 0 else math.nan
    return {
        "n": int(n),
        "score": score,
        "t_star_probe": t_star,
        "achieved_ratio": achieved,
        "clamped": clamped,
    }


def build_cases(*, probe_repeats: int, seed: int, gillespie_reactions: int,
                slugs: Sequence[str]) -> tuple[list[pbc.ProfileCase], list[dict[str, Any]]]:
    by_slug = {case.case.slug: case for case in pbc.profile_cases()}
    unknown = [slug for slug in slugs if slug not in by_slug]
    if unknown:
        raise SystemExit(f"unknown base case slug(s): {', '.join(unknown)}")

    selected: list[pbc.ProfileCase] = []
    report: list[dict[str, Any]] = []
    seen_populations: dict[str, set[int]] = {}
    for slug in slugs:
        base = by_slug[slug]
        for ratio in TARGET_RATIOS:
            solved = solve_population(base, ratio, probe_repeats=probe_repeats, seed=seed,
                                      gillespie_reactions=gillespie_reactions)
            entry = {"base": slug, "target_ratio": ratio, **solved}
            report.append(entry)
            achieved = solved["achieved_ratio"]
            if not math.isfinite(achieved):
                entry["status"] = "no paired measurement"
                continue
            # Reject states the population bounds could not bring into the useful band.
            if not (0.35 <= achieved <= 3.0):
                entry["status"] = f"unreachable within [{N_MIN:g}, {N_MAX:g}]"
                continue
            population = solved["n"]
            already = seen_populations.setdefault(slug, set())
            # Two target ratios can land on the same population; keep one copy.
            if any(abs(math.log10(population / other)) < 0.02 for other in already):
                entry["status"] = "duplicate population"
                continue
            already.add(population)
            entry["status"] = "kept"
            label = f"nb_{slug}_r{ratio:g}".replace(".", "")
            selected.append(_with_population(base, population, label))
    return selected, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=301,
                        help="final high-fidelity timing repeats per case")
    parser.add_argument("--probe-repeats", type=int, default=15,
                        help="cheap repeats used while solving for the population")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for the population search; used once so every timing seed "
                             "measures the *same* populations and the passes stay poolable")
    parser.add_argument("--timing-seeds", type=str, default="1",
                        help="comma-separated seeds for the final timing pass; each writes "
                             "<output stem>_seed<k>.csv except the first, which uses --output")
    parser.add_argument("--gillespie-reactions", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slugs", type=str,
                        help="comma-separated base case slugs (default: the built-in selection)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    slugs = tuple(s.strip() for s in args.slugs.split(",")) if args.slugs else BASE_SLUGS

    print(f"solving populations for {len(slugs)} base CRNs x {len(TARGET_RATIOS)} target ratios "
          f"({args.probe_repeats} probe repeats)\n")
    cases, report = build_cases(probe_repeats=args.probe_repeats, seed=args.seed,
                                gillespie_reactions=args.gillespie_reactions, slugs=slugs)

    print(f"{'base case':<32} {'want':>5} {'n':>12} {'score':>10} {'T* probe':>10} "
          f"{'got':>6}  status")
    for entry in report:
        print(f"{entry['base']:<32} {entry['target_ratio']:>5.2g} {entry['n']:>12.4g} "
              f"{entry['score']:>10.4g} {entry['t_star_probe']:>10.4g} "
              f"{entry['achieved_ratio']:>6.3g}  {entry.get('status', '?')}")

    kept = sum(1 for entry in report if entry.get("status") == "kept")
    print(f"\nkept {kept} near-boundary states; timing them at {args.repeats} repeats\n")
    if not cases:
        raise SystemExit("no near-boundary states were reachable")

    timing_seeds = [int(part) for part in args.timing_seeds.split(",") if part.strip()]
    output = args.output.resolve()
    for index, timing_seed in enumerate(timing_seeds):
        destination = output if index == 0 else output.with_name(
            f"{output.stem}_seed{timing_seed}{output.suffix}"
        )
        print(f"--- timing pass seed={timing_seed} -> {destination.name} ---")
        pbc.collect_timings(cases, repeats=args.repeats, seed=timing_seed,
                            gillespie_reactions=args.gillespie_reactions,
                            output=destination)


if __name__ == "__main__":
    main()
