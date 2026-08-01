"""Profile the cost drivers of one batch and measure matched engine costs.

This is an offline diagnostic for the deterministic switching experiment.  It deliberately keeps
Flame instrumentation separate from the timings used to estimate the oracle threshold

    T* = batch steady seconds / Gillespie steady seconds per exact reaction.

Run the two modes with different extension builds::

    # Component attribution (the report requires Flame support).
    maturin develop --release --features flm
    python benchmark/profile_batch_costs.py components

    # Uninstrumented engine timings.
    maturin develop --release
    python benchmark/profile_batch_costs.py timings

``components`` constructs every fresh simulator before clearing the thread-local Flame recorder,
then runs exactly one batch on each simulator.  Consequently construction spans do not leak into
the component totals.  ``timings`` benchmarks a batch for every case and also benchmarks Gillespie
for the matched Shrinking/branch-count controls.  The collision grid is batch-only in timing mode:
its purpose is to isolate how N, o, and g affect batch internals, not to fit a useful T*.

Outputs are written directly in this directory by default:

* ``batch_cost_profile_components.csv``
* ``batch_cost_profile_timings.csv``
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from threshold_model import ExperimentCase, _make_sim, experiment_cases

import batss as bt
from batss.benchmarking import CRNSpec


HERE = Path(__file__).resolve().parent
DEFAULT_COMPONENTS_CSV = HERE / "batch_cost_profile_components.csv"
DEFAULT_TIMINGS_CSV = HERE / "batch_cost_profile_timings.csv"
MATCHED_N = 2_000_000


@dataclass(frozen=True)
class ProfileCase:
    """One initial configuration plus labels for the controlled comparison."""

    case: ExperimentCase
    family: str
    branch_channels: int | None = None
    measure_gillespie: bool = False


def _matched_shrinking_cases() -> list[ProfileCase]:
    wanted = {"shrinking", "shrinking_no_b_decay", "shrinking_split_b_decay"}
    cases = [case for case in experiment_cases("matched") if case.slug in wanted]
    return [
        ProfileCase(
            replace(case, initial_n=MATCHED_N),
            family="matched_shrinking",
            measure_gillespie=True,
        )
        for case in cases
    ]


def _branch_case(channels: int) -> ProfileCase:
    """Split B -> None into generator-equivalent channels whose rates sum to 100."""

    a, b = bt.species(f"A_m{channels} B_m{channels}")
    reactions = [(a >> None).k(100.0)]
    reactions.extend((b >> None).k(100.0 / channels) for _ in range(channels))
    reactions.append((2 * a >> 2 * b).k(1.0))
    spec = CRNSpec(
        name=f"Shrinking B decay split {channels} way(s)",
        rxns=reactions,
        inits_from_n=lambda n, a=a, b=b: {a: 3 * n // 5, b: 2 * n // 5},
        benchmark_end_time=0.1,
    )
    case = ExperimentCase(
        slug=f"shrinking_b_decay_channels_{channels}",
        spec=spec,
        initial_n=MATCHED_N,
        end_time=0.1,
    )
    return ProfileCase(
        case,
        family="branch_count_control",
        branch_channels=channels,
        measure_gillespie=True,
    )


def _collision_case(order: int, generativity: int, real_n: int) -> ProfileCase:
    """Create a one-channel CRN with exactly the requested maximum order and generativity."""

    a, b = bt.species(f"A_o{order}_g{generativity}_n{real_n} B_o{order}_g{generativity}_n{real_n}")
    reaction = (order * a >> (order + generativity) * b).k(1.0)
    spec = CRNSpec(
        name=f"Collision grid o={order}, g={generativity}, n={real_n}",
        rxns=[reaction],
        inits_from_n=lambda n, a=a: {a: n},
        benchmark_end_time=1.0,
    )
    case = ExperimentCase(
        slug=f"collision_o{order}_g{generativity}_n{real_n}",
        spec=spec,
        initial_n=real_n,
        end_time=1.0,
    )
    return ProfileCase(case, family="collision_grid")


def _dense_support_case(total_q: int) -> ProfileCase:
    """Keep only A populated while growing the declared dense q-by-q lane structure."""

    if total_q < 4:
        raise ValueError("total_q includes A, B, K, and W and must be at least 4")
    a, b = bt.species(f"A_q{total_q} B_q{total_q}")
    spectator_count = total_q - 4
    spectators = (
        bt.species(" ".join(f"D{i}_q{total_q}" for i in range(spectator_count)))
        if spectator_count
        else ()
    )
    reactions = [(2 * a >> 2 * b).k(1.0)]
    reactions.extend((2 * spectator >> 2 * spectator).k(0.0) for spectator in spectators)
    spec = CRNSpec(
        name=f"Dense support control q={total_q}",
        rxns=reactions,
        inits_from_n=lambda n, a=a: {a: n},
        benchmark_end_time=1.0,
    )
    case = ExperimentCase(
        slug=f"dense_support_q{total_q}",
        spec=spec,
        initial_n=MATCHED_N,
        end_time=1.0,
    )
    return ProfileCase(case, family="dense_support_control")


def _composition_case(a_fraction: float) -> ProfileCase:
    """Keep no-B-decay Shrinking fixed while changing which groups are occupied."""

    fraction_label = f"{a_fraction:.2f}".replace(".", "")
    a, b = bt.species(f"A_f{fraction_label} B_f{fraction_label}")
    spec = CRNSpec(
        name=f"Shrinking no B decay composition A={a_fraction:.0%}",
        rxns=[(a >> None).k(100.0), (2 * a >> 2 * b).k(1.0)],
        inits_from_n=lambda n, a=a, b=b, fraction=a_fraction: {
            a: round(fraction * n),
            b: n - round(fraction * n),
        },
        benchmark_end_time=0.1,
    )
    case = ExperimentCase(
        slug=f"shrinking_no_b_decay_a_fraction_{fraction_label}",
        spec=spec,
        initial_n=MATCHED_N,
        end_time=0.1,
    )
    return ProfileCase(case, family="composition_control")


def _real_crn_initial_cases() -> list[ProfileCase]:
    wanted = {"oregonator", "rossler", "dimerization", "lotka_volterra", "brusselator"}
    return [
        ProfileCase(case, family="real_crn_initial")
        for case in experiment_cases("full")
        if case.slug in wanted
    ]


def profile_cases() -> list[ProfileCase]:
    cases = _matched_shrinking_cases()
    cases.extend(_branch_case(channels) for channels in (1, 2, 4, 8))
    cases.extend(
        _collision_case(order, generativity, real_n)
        for order in (2, 3)
        for generativity in (0, 1, 2)
        for real_n in (10_000, 1_000_000, 100_000_000)
    )
    cases.extend(_dense_support_case(total_q) for total_q in (4, 6, 8, 12))
    cases.extend(_composition_case(a_fraction) for a_fraction in (0.60, 0.10, 0.01))
    cases.extend(_real_crn_initial_cases())
    return cases


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(map(str, (base_seed, *parts))).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _metadata(profile_case: ProfileCase, sim: Any) -> dict[str, Any]:
    rust = sim.simulator
    prospective_n = int(rust.debug_prospective_n())
    order = int(rust.debug_o())
    generativity = int(rust.debug_g())
    score = float(rust.prospective_batch_score())
    expected_batch_length = (
        math.sqrt(math.pi / (2.0 * order * (order + generativity))) * math.sqrt(prospective_n)
        if prospective_n > 0 and order > 0
        else 0.0
    )
    active_probability = score / expected_batch_length if expected_batch_length else 0.0
    return {
        "case": profile_case.case.slug,
        "name": profile_case.case.spec.name,
        "family": profile_case.family,
        "branch_channels": profile_case.branch_channels,
        "initial_real_N": profile_case.case.initial_n,
        "prospective_N": prospective_n,
        "q": int(rust.debug_q()),
        "o": order,
        "g": generativity,
        "q_power_o": int(rust.debug_q()) ** order,
        "reactant_sets_R": int(rust.debug_reactant_sets()),
        "output_branches_B": int(rust.debug_output_branches()),
        "score": score,
        "expected_batch_length": expected_batch_length,
        "active_probability": active_probability,
    }


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else math.nan


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty profile table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_PROFILE_LINE = re.compile(
    r"^(?P<indent> *)(?P<name>.+?):\s+"
    r"(?P<milliseconds>[0-9]+(?:\.[0-9]+)?) ms "
    r"\((?P<count>[0-9]+) calls, (?P<microseconds>[0-9]+(?:\.[0-9]+)?) us/call\)$"
)


def _parse_profile(path: Path) -> dict[str, dict[str, float | int]]:
    """Parse the precise count/us-per-call Flame report, aggregating equal leaf names."""

    if not path.exists():
        raise RuntimeError(
            "write_profile produced no file; rebuild with `maturin develop --release --features flm`"
        )
    aggregate: dict[str, dict[str, float | int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PROFILE_LINE.match(line)
        if match is None:
            continue
        name = match.group("name").strip()
        milliseconds = float(match.group("milliseconds"))
        count = int(match.group("count"))
        entry = aggregate.setdefault(name, {"milliseconds": 0.0, "count": 0})
        entry["milliseconds"] = float(entry["milliseconds"]) + milliseconds
        entry["count"] = int(entry["count"]) + count
    if "batch step" not in aggregate:
        raise RuntimeError(
            f"{path} contains no precise `batch step` record; rebuild the current Rust source with the flm feature"
        )
    for entry in aggregate.values():
        count = int(entry["count"])
        entry["microseconds_per_call"] = float(entry["milliseconds"]) * 1_000.0 / count
    return aggregate


_COMPONENT_COLUMNS = {
    "batch step": "batch_step_component_us_per_call",
    "sample_coll": "sample_collision_free_length_us_per_call",
    "sample batch clock": "sample_batch_clock_us_per_call",
    "checkpoint rejection sampling": "checkpoint_rejection_us_per_call",
    "sample batch": "sample_multidimensional_batch_us_per_call",
    "process batch": "process_dense_batch_lanes_us_per_call",
    "sample collision": "sample_terminal_collision_us_per_call",
    "urn commit and sort": "urn_commit_and_sort_us_per_call",
    "finish batch step": "finish_batch_step_us_per_call",
}


def collect_components(
    cases: Sequence[ProfileCase], *, repeats: int, seed: int, output: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical_positions = {profile_case.case.slug: index for index, profile_case in enumerate(cases)}
    traversal = list(cases)
    if seed % 2 == 0:
        traversal.reverse()
    for case_index, profile_case in enumerate(traversal, start=1):
        config = profile_case.case.spec.inits_from_n(profile_case.case.initial_n)
        # Construct first, then clear once: transition-array construction must not be attributed to
        # a steady batch. Every simulator is used for exactly one engine call.
        sims = [
            _make_sim(
                profile_case.case,
                config,
                _stable_seed(seed, profile_case.case.slug, repeat),
            )
            for repeat in range(repeats)
        ]
        row = _metadata(profile_case, sims[0])
        row["profile_seed"] = seed
        if not hasattr(sims[0].simulator, "clear_profile"):
            raise RuntimeError("the loaded extension predates the profiling API; rebuild the Rust extension")
        sims[0].simulator.clear_profile()
        results = [sim.simulator.benchmark_engine_call(False) for sim in sims]
        with tempfile.TemporaryDirectory(prefix="batss-profile-") as temporary:
            report_path = Path(temporary) / "flame.txt"
            sims[0].simulator.write_profile(str(report_path))
            components = _parse_profile(report_path)

        row.update(
            {
                "profile_repeats": repeats,
                "batch_engine_us_per_call": _median(result.engine_seconds * 1e6 for result in results),
                "batch_postprocess_us_per_call": _median(
                    result.postprocess_seconds * 1e6 for result in results
                ),
                "batch_steady_us_per_call": _median(
                    (result.engine_seconds + result.postprocess_seconds) * 1e6 for result in results
                ),
                "total_reactions_per_batch": _median(result.total_reactions for result in results),
                "active_reactions_per_batch": _median(result.active_reactions for result in results),
                "active_reaction_fraction_realized": (
                    sum(result.active_reactions for result in results)
                    / sum(result.total_reactions for result in results)
                ),
            }
        )
        for component_name, column_name in _COMPONENT_COLUMNS.items():
            component = components.get(component_name)
            row[column_name] = (
                float(component["microseconds_per_call"]) if component is not None else math.nan
            )
        row["all_flame_components_json"] = json.dumps(
            {
                name: {
                    "calls": int(values["count"]),
                    "us_per_call": float(values["microseconds_per_call"]),
                }
                for name, values in sorted(components.items())
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append(row)
        print(
            f"[{case_index}/{len(traversal)}] {profile_case.case.slug}: "
            f"batch={row['batch_steady_us_per_call']:.3f} us, "
            f"sample_coll={row['sample_collision_free_length_us_per_call']:.3f} us",
            flush=True,
        )
    rows.sort(key=lambda row: canonical_positions[str(row["case"])])
    _write_csv(output, rows)
    print(f"wrote {output}")
    return rows


def _benchmark_once(profile_case: ProfileCase, *, seed: int, gillespie: bool, reactions: int) -> Any:
    config = profile_case.case.spec.inits_from_n(profile_case.case.initial_n)
    sim = _make_sim(profile_case.case, config, seed)
    return sim.simulator.benchmark_engine_call(gillespie, reactions if gillespie else None)


def collect_timings(
    cases: Sequence[ProfileCase],
    *,
    repeats: int,
    seed: int,
    gillespie_reactions: int,
    output: Path,
) -> list[dict[str, Any]]:
    # Pay first-use costs before collecting samples. The warm-up uses a fresh simulator, just as
    # every measured call does, and exercises both engines wherever T* will be calculated.
    metadata_rows: dict[str, dict[str, Any]] = {}
    batch_results_by_case: dict[str, list[Any]] = {}
    gillespie_results_by_case: dict[str, list[Any]] = {}
    for profile_case in cases:
        slug = profile_case.case.slug
        config = profile_case.case.spec.inits_from_n(profile_case.case.initial_n)
        probe = _make_sim(profile_case.case, config, _stable_seed(seed, slug, "metadata"))
        metadata_rows[slug] = _metadata(profile_case, probe)
        metadata_rows[slug]["profile_seed"] = seed
        batch_results_by_case[slug] = []
        gillespie_results_by_case[slug] = []
        warm_seed = _stable_seed(seed, slug, "warm")
        _benchmark_once(profile_case, seed=warm_seed, gillespie=False, reactions=0)
        if profile_case.measure_gillespie:
            _benchmark_once(
                profile_case,
                seed=warm_seed,
                gillespie=True,
                reactions=gillespie_reactions,
            )

    # Sweep the complete case matrix once per repeat instead of exhausting one case at a time.
    # Rotating the starting case makes every CRN occupy different positions in the sweep, reducing
    # confounding from CPU frequency, temperature, and other slow monotonic drift.
    ordered_cases = list(cases)
    for repeat in range(repeats):
        rotation = repeat % len(ordered_cases)
        repeat_cases = ordered_cases[rotation:] + ordered_cases[:rotation]
        for profile_case in repeat_cases:
            slug = profile_case.case.slug
            trial_seed = _stable_seed(seed, slug, repeat)
            # Alternate paired execution order to reduce monotonic clock/thermal drift. Each call
            # still receives a fresh simulator and the same seed.
            if profile_case.measure_gillespie and repeat % 2:
                gillespie_results_by_case[slug].append(
                    _benchmark_once(
                        profile_case,
                        seed=trial_seed,
                        gillespie=True,
                        reactions=gillespie_reactions,
                    )
                )
                batch_results_by_case[slug].append(
                    _benchmark_once(profile_case, seed=trial_seed, gillespie=False, reactions=0)
                )
            else:
                batch_results_by_case[slug].append(
                    _benchmark_once(profile_case, seed=trial_seed, gillespie=False, reactions=0)
                )
                if profile_case.measure_gillespie:
                    gillespie_results_by_case[slug].append(
                        _benchmark_once(
                            profile_case,
                            seed=trial_seed,
                            gillespie=True,
                            reactions=gillespie_reactions,
                        )
                    )
        print(f"timing repeat {repeat + 1}/{repeats}", flush=True)

    rows: list[dict[str, Any]] = []
    for case_index, profile_case in enumerate(cases, start=1):
        slug = profile_case.case.slug
        row = metadata_rows[slug]
        batch_results = batch_results_by_case[slug]
        gillespie_results = gillespie_results_by_case[slug]
        batch_engine = _median(result.engine_seconds for result in batch_results)
        batch_postprocess = _median(result.postprocess_seconds for result in batch_results)
        batch_steady = _median(
            result.engine_seconds + result.postprocess_seconds for result in batch_results
        )
        row.update(
            {
                "timing_repeats": repeats,
                "batch_engine_us_per_call": batch_engine * 1e6,
                "batch_postprocess_us_per_call": batch_postprocess * 1e6,
                "batch_steady_us_per_call": batch_steady * 1e6,
                "batch_total_reactions": _median(result.total_reactions for result in batch_results),
                "batch_active_reactions": _median(result.active_reactions for result in batch_results),
            }
        )
        if gillespie_results:
            per_reaction = [
                (result.engine_seconds + result.postprocess_seconds) / result.total_reactions
                for result in gillespie_results
                if result.total_reactions > 0
            ]
            gillespie_seconds_per_reaction = _median(per_reaction)
            row.update(
                {
                    "gillespie_requested_reactions": gillespie_reactions,
                    "gillespie_actual_reactions": _median(
                        result.total_reactions for result in gillespie_results
                    ),
                    "gillespie_setup_us_per_block": _median(
                        result.setup_seconds * 1e6 for result in gillespie_results
                    ),
                    "gillespie_steady_us_per_reaction": gillespie_seconds_per_reaction * 1e6,
                    "threshold_T_star": batch_steady / gillespie_seconds_per_reaction,
                }
            )
        else:
            row.update(
                {
                    "gillespie_requested_reactions": "",
                    "gillespie_actual_reactions": "",
                    "gillespie_setup_us_per_block": "",
                    "gillespie_steady_us_per_reaction": "",
                    "threshold_T_star": "",
                }
            )
        rows.append(row)
        threshold_text = (
            f", T*={row['threshold_T_star']:.3f}" if isinstance(row["threshold_T_star"], float) else ""
        )
        print(
            f"[{case_index}/{len(cases)}] {profile_case.case.slug}: "
            f"batch={row['batch_steady_us_per_call']:.3f} us{threshold_text}",
            flush=True,
        )
    _write_csv(output, rows)
    print(f"wrote {output}")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("components", "timings"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gillespie-reactions", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats is None:
        args.repeats = 40 if args.mode == "components" else 9
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.gillespie_reactions < 1:
        parser.error("--gillespie-reactions must be positive")
    if args.output is None:
        args.output = DEFAULT_COMPONENTS_CSV if args.mode == "components" else DEFAULT_TIMINGS_CSV
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cases = profile_cases()
    if args.mode == "components":
        collect_components(cases, repeats=args.repeats, seed=args.seed, output=args.output.resolve())
    else:
        collect_timings(
            cases,
            repeats=args.repeats,
            seed=args.seed,
            gillespie_reactions=args.gillespie_reactions,
            output=args.output.resolve(),
        )


if __name__ == "__main__":
    main()
