"""Collect and fit a deterministic CRN-structured switching threshold.

This is an offline experiment for issue #14.  It deliberately does *not* alter the simulator's
production switching policy.  The experiment has three stages:

1. run deterministic reference trajectories and freeze representative configurations;
2. benchmark one batch and one exact-Gillespie block repeatedly from each frozen configuration;
3. fit the proposed model

       T = theta_0 + theta_1 log2(N) + theta_2 q**o + theta_3 R + theta_4 B
           + theta_5 o log2(N) [g > 0]

   where T is the measured cost of a full steady-state batch divided by the measured Gillespie
   cost per exact reaction.

The paired engine timings happen inside Rust via ``BatchSimulator.benchmark_engine_call``.  K
preparation and Gillespie construction are reported in the raw data but excluded from T: those are
switching costs, whereas this model is intended to predict steady engine throughput.  Normal batch
postprocessing and amortized Gillespie-to-urn synchronization are included in both costs.

Examples (run from the repository root after rebuilding the Rust extension)::

    python benchmark/threshold_model.py run --preset quick
    python benchmark/threshold_model.py run --preset matched
    python benchmark/threshold_model.py run --preset full
    python benchmark/threshold_model.py collect --preset full --repeats 31
    python benchmark/threshold_model.py fit --preset full

The quick preset uses the three within-run regime candidates requested for the first experiment:
Oregonator, Rössler-Willamowski, and Shrinking.  The matched preset adds two controlled Shrinking
variants at the same population and duration: one removes B decay, and one splits B decay into two
identical half-rate channels.  Splitting preserves the CTMC generator while increasing the stored
output-branch count B.  The full preset instead adds Dimerization, Lotka-Volterra, and the order-3
Brusselator for broad structural coverage.  Results are
written under ``benchmark/results`` (which is gitignored).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Make ``python benchmark/threshold_model.py`` use the in-tree Python package even when the project
# has not been installed in editable mode.  The compiled extension must still have been rebuilt.
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import batss as bt  # noqa: E402
from batss.benchmarking import CRNSpec  # noqa: E402


FEATURE_NAMES = (
    "intercept",
    "log2_N",
    "q_power_o",
    "reactant_sets_R",
    "output_branches_B",
    "generative_o_log2_N",
)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ExperimentCase:
    """A CRN plus the single reference trajectory used to freeze states."""

    slug: str
    spec: CRNSpec
    initial_n: int
    end_time: float


@dataclass
class FrozenState:
    crn: str
    state_id: int
    trajectory_time: float
    initial_n: int
    config: dict[Any, int]
    config_by_name: dict[str, int]
    real_n: int
    prospective_n: int
    score: float
    active_probability: float
    expected_batch_length: float
    q: int
    o: int
    g: int
    reactant_sets_R: int
    output_branches_B: int


def experiment_cases(preset: str) -> list[ExperimentCase]:
    """Define the requested dynamic CRNs and optional structural-coverage CRNs."""

    x1, x2, x3 = bt.species("X1 X2 X3")
    oregonator = CRNSpec(
        name="Oregonator",
        rxns=[
            (x2 >> x1).k(0.0871),
            (x2 + x1 >> None).k(1000),
            (x1 >> 2 * x1 + x3).k(520),
            (2 * x1 >> None).k(40),
            (x3 >> x2).k(443.324),
            (x3 >> None).k(2.676),
        ],
        inits_from_n=lambda n: {
            x1: int(0.0029 * n),
            x2: int(0.5358 * n),
            x3: int(0.0034 * n),
        },
        benchmark_end_time=1.0,
    )

    r1, r2, r3 = bt.species("X1 X2 X3")
    rossler = CRNSpec(
        name="Rössler-Willamowski",
        rxns=[
            (r1 >> 2 * r1).k(30),
            (2 * r1 >> r1).k(0.5),
            (r2 + r1 >> 2 * r2).k(1),
            (r2 >> None).k(10),
            (r1 + r3 >> None).k(1),
            (r3 >> 2 * r3).k(16.5),
            (2 * r3 >> r3).k(0.5),
        ],
        inits_from_n=lambda n: {r1: n // 3, r2: n // 3, r3: n - 2 * (n // 3)},
        benchmark_end_time=1.0,
    )

    a, b = bt.species("A B")
    shrinking = CRNSpec(
        name="Shrinking",
        rxns=[(a >> None).k(100), (b >> None).k(100), (2 * a >> 2 * b).k(1)],
        inits_from_n=lambda n: {a: n // 2 + n // 10, b: n // 2 - n // 10},
        benchmark_end_time=0.05,
    )

    no_b_a, no_b_b = bt.species("A B")
    shrinking_no_b_decay = CRNSpec(
        name="Shrinking no B decay",
        rxns=[(no_b_a >> None).k(100), (2 * no_b_a >> 2 * no_b_b).k(1)],
        inits_from_n=lambda n: {no_b_a: 3 * n // 5, no_b_b: 2 * n // 5},
        benchmark_end_time=0.1,
    )

    split_a, split_b = bt.species("A B")
    shrinking_split_b_decay = CRNSpec(
        name="Shrinking split B decay",
        # The two B -> None channels have the same update and rates summing to 100, so they
        # preserve the original CTMC generator while increasing the stored branch count B.
        rxns=[
            (split_a >> None).k(100),
            (split_b >> None).k(50),
            (split_b >> None).k(50),
            (2 * split_a >> 2 * split_b).k(1),
        ],
        inits_from_n=lambda n: {split_a: 3 * n // 5, split_b: 2 * n // 5},
        benchmark_end_time=0.1,
    )

    # These three do not all cross modes within one trajectory.  Their purpose is structural
    # identifiability: q, o, R, B, and g are constant within a CRN, so additional CRNs are needed
    # to estimate the corresponding coefficients.
    m, d = bt.species("M D")
    dimerization = CRNSpec(
        name="Dimerization",
        rxns=[(m + m >> d).k(1), (d >> m + m).k(1)],
        inits_from_n=lambda n: {m: n},
        benchmark_end_time=0.5,
    )

    prey, predator = bt.species("R F")
    lotka = CRNSpec(
        name="Lotka-Volterra",
        rxns=[
            (prey + predator >> 2 * predator).k(1),
            (prey >> 2 * prey).k(1),
            (predator >> None).k(1),
        ],
        inits_from_n=lambda n: {prey: n // 2, predator: n - n // 2},
        benchmark_end_time=1.0,
    )

    ba, bb, bx, by = bt.species("A B X Y")
    brusselator = CRNSpec(
        name="Order-3 Brusselator",
        rxns=[
            (ba >> ba + bx).k(1),
            (2 * bx + by >> 3 * bx).k(1),
            (bb + bx >> bb + by).k(1),
            (bx >> None).k(1),
        ],
        inits_from_n=lambda n: {ba: n, bb: 3 * n, bx: n, by: n},
        benchmark_end_time=40.0,
    )

    # The trajectory lengths match the gallery/notebook examples where possible.  Shrinking uses
    # the notebook's t=0.1 trajectory and a larger population so one run spans several decades.
    dynamic = [
        ExperimentCase("oregonator", oregonator, 100_000, 5.0),
        ExperimentCase("rossler", rossler, 100_000, 8.0),
        ExperimentCase(
            "shrinking",
            shrinking,
            2_000_000 if preset == "quick" else 100_000_000,
            0.1,
        ),
    ]
    if preset == "quick":
        return dynamic
    if preset == "matched":
        return dynamic + [
            ExperimentCase("shrinking_no_b_decay", shrinking_no_b_decay, 100_000_000, 0.1),
            ExperimentCase("shrinking_split_b_decay", shrinking_split_b_decay, 100_000_000, 0.1),
        ]
    return dynamic + [
        ExperimentCase("dimerization", dimerization, 100_000, 2.0),
        ExperimentCase("lotka_volterra", lotka, 100_000, 20.0),
        ExperimentCase("brusselator", brusselator, 20_000, 20.0),
    ]


def _species_for_spec(spec: CRNSpec) -> list[Any]:
    by_name: dict[str, Any] = {}
    for reaction in spec.rxns:
        for specie in (*reaction.reactants.species, *reaction.products.species):
            by_name.setdefault(specie.name, specie)
    return list(by_name.values())


def _make_sim(case: ExperimentCase, config: dict[Any, int], seed: int) -> Any:
    # The volume remains the *original* initial_n at every frozen state.  Using the shrinking
    # state's current count as volume would silently change all higher-order rate constants.
    return bt.Simulation(
        config,
        case.spec.rxns,
        simulator_method="crn",
        continuous_time=True,
        seed=seed,
        volume=case.initial_n,
    )


def _row_to_config(row: Any, species: Sequence[Any]) -> dict[Any, int]:
    values_by_name = {getattr(column, "name", str(column)): int(value) for column, value in row.items()}
    return {specie: values_by_name.get(specie.name, 0) for specie in species}


def _state_metadata(
    case: ExperimentCase,
    config: dict[Any, int],
    trajectory_time: float,
    seed: int,
) -> FrozenState | None:
    sim = _make_sim(case, config, seed)
    rust = sim.simulator
    real_propensity = float(rust.debug_p_active())
    if not (real_propensity > 0.0 and math.isfinite(real_propensity)):
        return None

    prospective_n = int(rust.debug_prospective_n())
    o = int(rust.debug_o())
    g = int(rust.debug_g())
    score = float(rust.prospective_batch_score())
    expected_length = (
        math.sqrt(math.pi / (2.0 * o * (o + g))) * math.sqrt(prospective_n) if prospective_n > 0 and o > 0 else 0.0
    )
    active_probability = score / expected_length if expected_length > 0 else 0.0
    config_by_name = {specie.name: int(count) for specie, count in config.items() if count}
    return FrozenState(
        crn=case.slug,
        state_id=-1,
        trajectory_time=float(trajectory_time),
        initial_n=case.initial_n,
        config=config,
        config_by_name=config_by_name,
        real_n=sum(config.values()),
        prospective_n=prospective_n,
        score=score,
        active_probability=active_probability,
        expected_batch_length=expected_length,
        q=int(rust.debug_q()),
        o=o,
        g=g,
        reactant_sets_R=int(rust.debug_reactant_sets()),
        output_branches_B=int(rust.debug_output_branches()),
    )


def _select_spanning_states(candidates: Sequence[FrozenState], target: int) -> list[FrozenState]:
    """Farthest-point sample in (trajectory time, log score), preserving spikes and endpoints."""

    if len(candidates) <= target:
        selected = list(candidates)
    else:
        times = np.asarray([state.trajectory_time for state in candidates], dtype=float)
        scores = np.log1p(np.asarray([max(state.score, 0.0) for state in candidates], dtype=float))

        def normalize(values: np.ndarray) -> np.ndarray:
            span = float(values.max() - values.min())
            return (values - values.min()) / span if span > 0 else np.zeros_like(values)

        coordinates = np.column_stack((normalize(times), normalize(scores)))
        seed_order = [0, len(candidates) - 1, int(np.argmin(scores)), int(np.argmax(scores))]
        chosen: set[int] = set()
        for index in seed_order:
            chosen.add(index)
            if len(chosen) == target:
                break
        while len(chosen) < target:
            remaining = [index for index in range(len(candidates)) if index not in chosen]
            nearest = []
            for index in remaining:
                distance = min(float(np.sum((coordinates[index] - coordinates[other]) ** 2)) for other in chosen)
                nearest.append((distance, -index, index))
            chosen.add(max(nearest)[2])
        selected = [candidates[index] for index in sorted(chosen)]

    for state_id, state in enumerate(selected):
        state.state_id = state_id
    return selected


def capture_frozen_states(
    case: ExperimentCase,
    *,
    seed: int,
    states_per_crn: int,
    candidate_factor: int,
    trajectory_threshold: float,
) -> list[FrozenState]:
    candidate_count = max(states_per_crn, states_per_crn * candidate_factor)
    sim = _make_sim(case, case.spec.inits_from_n(case.initial_n), seed)
    # This fixed number is only a deterministic way to generate a valid exact trajectory.  It is
    # never used as a label or as the fitted threshold.  The paired timing oracle below decides
    # which engine is faster at each frozen state.
    sim.simulator.heuristic_gillespie_switching = 2
    sim.simulator.proxy_threshold = trajectory_threshold
    interval = case.end_time / max(candidate_count - 1, 1)
    sim.run(case.end_time, interval, stopping_interval=interval, timer=False)

    species = _species_for_spec(case.spec)
    candidates: list[FrozenState] = []
    seen: set[tuple[int, ...]] = set()
    for trajectory_time, row in sim.history.iterrows():
        config = _row_to_config(row, species)
        fingerprint = tuple(config[specie] for specie in species)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        metadata = _state_metadata(case, config, float(trajectory_time), seed)
        if metadata is not None:
            candidates.append(metadata)
    if not candidates:
        raise RuntimeError(f"{case.slug}: reference trajectory produced no active frozen states")
    selected = _select_spanning_states(candidates, states_per_crn)
    score_min = min(state.score for state in selected)
    score_max = max(state.score for state in selected)
    print(
        f"{case.slug}: selected {len(selected)}/{len(candidates)} active states "
        f"(score {score_min:.4g}..{score_max:.4g})",
        flush=True,
    )
    return selected


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(map(str, (base_seed, *parts))).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    # pyo3 accepts a u64, but staying below 2**63 is friendlier to Python/numpy tooling.
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _benchmark_result_dict(result: Any) -> dict[str, Any]:
    return {
        "preparation_seconds": float(result.preparation_seconds),
        "setup_seconds": float(result.setup_seconds),
        "engine_seconds": float(result.engine_seconds),
        "postprocess_seconds": float(result.postprocess_seconds),
        "steady_seconds": float(result.engine_seconds + result.postprocess_seconds),
        "continuous_time_advanced": float(result.continuous_time_advanced),
        "total_reactions": int(result.total_reactions),
        "active_reactions": int(result.active_reactions),
        "k_rebuilt": bool(result.k_rebuilt),
    }


def _one_engine_call(
    case: ExperimentCase,
    state: FrozenState,
    *,
    seed: int,
    gillespie: bool,
    gillespie_reactions: int | None,
) -> dict[str, Any]:
    sim = _make_sim(case, state.config, seed)
    result = sim.simulator.benchmark_engine_call(gillespie, gillespie_reactions)
    return _benchmark_result_dict(result)


def _median_of_means(values: Iterable[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(finite) == 0:
        return math.inf
    groups = max(1, int(math.sqrt(len(finite))))
    group_means = [float(np.mean(group)) for group in np.array_split(finite, groups) if len(group)]
    return float(np.median(group_means))


def benchmark_frozen_state(
    case: ExperimentCase,
    state: FrozenState,
    *,
    seed: int,
    repeats: int,
    gillespie_reactions: int | None,
    raw_file: Any,
) -> dict[str, Any]:
    # Pay one throwaway call for each path.  Rust measures only engine internals, but this also
    # removes first-use allocator/page effects from the paired samples.
    warm_seed = _stable_seed(seed, case.slug, state.state_id, "warm")
    _one_engine_call(
        case,
        state,
        seed=warm_seed,
        gillespie=False,
        gillespie_reactions=gillespie_reactions,
    )
    _one_engine_call(
        case,
        state,
        seed=warm_seed,
        gillespie=True,
        gillespie_reactions=gillespie_reactions,
    )

    batch_trials: list[dict[str, Any]] = []
    gillespie_trials: list[dict[str, Any]] = []
    for repeat in range(repeats):
        trial_seed = _stable_seed(seed, case.slug, state.state_id, repeat)
        order = (True, False) if repeat % 2 else (False, True)
        results: dict[bool, dict[str, Any]] = {}
        for is_gillespie in order:
            results[is_gillespie] = _one_engine_call(
                case,
                state,
                seed=trial_seed,
                gillespie=is_gillespie,
                gillespie_reactions=gillespie_reactions,
            )
        batch = results[False]
        gillespie = results[True]
        if gillespie["total_reactions"] <= 0:
            raise RuntimeError(f"{case.slug} state {state.state_id}: Gillespie made no reaction")
        batch_trials.append(batch)
        gillespie_trials.append(gillespie)
        raw_file.write(
            json.dumps(
                {
                    "crn": case.slug,
                    "state_id": state.state_id,
                    "trajectory_time": state.trajectory_time,
                    "repeat": repeat,
                    "seed": trial_seed,
                    "config": state.config_by_name,
                    "batch": batch,
                    "gillespie": gillespie,
                },
                sort_keys=True,
            )
            + "\n"
        )
        raw_file.flush()

    batch_seconds = _median_of_means(trial["steady_seconds"] for trial in batch_trials)
    gillespie_seconds_per_reaction = _median_of_means(
        trial["steady_seconds"] / trial["total_reactions"] for trial in gillespie_trials
    )
    active_reactions = _median_of_means(trial["active_reactions"] for trial in batch_trials)
    total_reactions = _median_of_means(trial["total_reactions"] for trial in batch_trials)
    observed_batch_seconds_per_active = _median_of_means(
        trial["steady_seconds"] / trial["active_reactions"] for trial in batch_trials if trial["active_reactions"] > 0
    )
    threshold = batch_seconds / gillespie_seconds_per_reaction
    # This experiment tests the proposed decomposition C_B / (p E[L]) versus C_G. Therefore the
    # timing oracle's mode boundary is exactly p E[L] = C_B/C_G, i.e. score = T*. Realized active
    # reactions are retained as a diagnostic, but are too sparse at low p to label a state from a
    # handful of batches reliably (an Oregonator batch can quite legitimately realize zero).
    oracle_mode = "batch" if state.score > threshold else "gillespie"
    observed_oracle_mode = (
        "batch" if observed_batch_seconds_per_active < gillespie_seconds_per_reaction else "gillespie"
    )

    return {
        "crn": case.slug,
        "state_id": state.state_id,
        "trajectory_time": state.trajectory_time,
        "initial_n": case.initial_n,
        "config": json.dumps(state.config_by_name, sort_keys=True, separators=(",", ":")),
        "real_n": state.real_n,
        "prospective_n": state.prospective_n,
        "score": state.score,
        "active_probability": state.active_probability,
        "expected_batch_length": state.expected_batch_length,
        "q": state.q,
        "o": state.o,
        "g": state.g,
        "q_power_o": float(state.q**state.o),
        "reactant_sets_R": state.reactant_sets_R,
        "output_branches_B": state.output_branches_B,
        "batch_steady_seconds": batch_seconds,
        "gillespie_seconds_per_reaction": gillespie_seconds_per_reaction,
        "threshold_T_star": threshold,
        "batch_total_reactions": total_reactions,
        "batch_active_reactions": active_reactions,
        "batch_seconds_per_active": observed_batch_seconds_per_active,
        "oracle_mode": oracle_mode,
        "observed_oracle_mode": observed_oracle_mode,
        "batch_preparation_seconds": _median_of_means(trial["preparation_seconds"] for trial in batch_trials),
        "batch_postprocess_seconds": _median_of_means(trial["postprocess_seconds"] for trial in batch_trials),
        "gillespie_preparation_seconds": _median_of_means(trial["preparation_seconds"] for trial in gillespie_trials),
        "gillespie_setup_seconds": _median_of_means(trial["setup_seconds"] for trial in gillespie_trials),
        "gillespie_postprocess_seconds": _median_of_means(trial["postprocess_seconds"] for trial in gillespie_trials),
        "batch_k_rebuild_fraction": float(np.mean([trial["k_rebuilt"] for trial in batch_trials])),
        "gillespie_k_rebuild_fraction": float(np.mean([trial["k_rebuilt"] for trial in gillespie_trials])),
        "repeats": repeats,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect(
    cases: Sequence[ExperimentCase],
    *,
    output_prefix: Path,
    seed: int,
    repeats: int,
    states_per_crn: int,
    candidate_factor: int,
    trajectory_threshold: float,
    gillespie_reactions: int | None,
) -> list[dict[str, Any]]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_prefix.with_name(output_prefix.name + "_trials.jsonl")
    state_path = output_prefix.with_name(output_prefix.name + "_states.csv")
    aggregate_rows: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for case in cases:
            states = capture_frozen_states(
                case,
                seed=_stable_seed(seed, case.slug, "trajectory"),
                states_per_crn=states_per_crn,
                candidate_factor=candidate_factor,
                trajectory_threshold=trajectory_threshold,
            )
            for index, state in enumerate(states, 1):
                print(
                    f"  {case.slug} state {index}/{len(states)} at t={state.trajectory_time:.5g}",
                    flush=True,
                )
                aggregate_rows.append(
                    benchmark_frozen_state(
                        case,
                        state,
                        seed=seed,
                        repeats=repeats,
                        gillespie_reactions=gillespie_reactions,
                        raw_file=raw_file,
                    )
                )
                # Preserve completed aggregate states if a long full run is interrupted.
                _write_csv(state_path, aggregate_rows)
    print(f"wrote {raw_path}")
    print(f"wrote {state_path}")
    return aggregate_rows


def _numeric_csv_rows(path: Path) -> list[dict[str, Any]]:
    integer_fields = {
        "state_id",
        "initial_n",
        "real_n",
        "prospective_n",
        "q",
        "o",
        "g",
        "reactant_sets_R",
        "output_branches_B",
        "repeats",
    }
    text_fields = {"crn", "config", "oracle_mode", "observed_oracle_mode"}
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    converted: list[dict[str, Any]] = []
    for row in rows:
        converted.append(
            {
                key: (value if key in text_fields else int(value) if key in integer_fields else float(value))
                for key, value in row.items()
            }
        )
    return converted


def _design_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                1.0,
                math.log2(float(row["prospective_n"])),
                float(row["q_power_o"]),
                float(row["reactant_sets_R"]),
                float(row["output_branches_B"]),
                float(row["o"]) * math.log2(float(row["prospective_n"])) * (float(row["g"]) > 0),
            ]
            for row in rows
        ],
        dtype=float,
    )


def _matrix_diagnostics(matrix: np.ndarray, feature_names: Sequence[str] = FEATURE_NAMES) -> dict[str, Any]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    raw_condition = float(np.linalg.cond(matrix))
    variable = matrix[:, 1:]
    scales = np.std(variable, axis=0)
    scaled = np.zeros_like(variable)
    nonconstant = scales > 0
    if len(variable):
        scaled[:, nonconstant] = (variable[:, nonconstant] - np.mean(variable[:, nonconstant], axis=0)) / scales[
            nonconstant
        ]
    standardized = np.column_stack((np.ones(len(matrix)), scaled))
    standardized_rank = int(np.linalg.matrix_rank(standardized))
    standardized_condition = float(np.linalg.cond(standardized))
    return {
        "rows": len(matrix),
        "columns": matrix.shape[1],
        "rank": rank,
        "singular_values": singular_values.tolist(),
        "raw_condition_number": raw_condition,
        "standardized_rank": standardized_rank,
        "standardized_condition_number": standardized_condition,
        "constant_feature_columns": [
            feature_names[index + 1] for index, is_variable in enumerate(nonconstant) if not is_variable
        ],
    }


def _regret_metrics(
    rows: Sequence[dict[str, Any]], predictions: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluated: list[dict[str, Any]] = []
    regrets: list[float] = []
    correct = 0
    clipped = 0
    for row, raw_prediction in zip(rows, predictions):
        threshold = max(float(raw_prediction), 0.0)
        clipped += raw_prediction < 0
        chosen = "batch" if float(row["score"]) > threshold else "gillespie"
        # Express both engine costs in units of C_G. Since T*=C_B/C_G and score=pE[L],
        # the expected batch cost per active reaction is T*/score; Gillespie's is 1.
        batch_cost = float(row["threshold_T_star"]) / float(row["score"])
        gillespie_cost = 1.0
        oracle = str(row["oracle_mode"])
        chosen_cost = batch_cost if chosen == "batch" else gillespie_cost
        regret = chosen_cost / min(batch_cost, gillespie_cost)
        correct += chosen == oracle
        regrets.append(regret)
        evaluated.append(
            {
                "raw_threshold": float(raw_prediction),
                "threshold": threshold,
                "chosen_mode": chosen,
                "oracle_mode": oracle,
                "regret": regret,
            }
        )
    finite_regrets = np.asarray([value for value in regrets if math.isfinite(value)], dtype=float)
    return (
        {
            "states": len(rows),
            "mode_accuracy": correct / len(rows),
            "mean_regret": float(np.mean(finite_regrets)),
            "geometric_mean_regret": float(np.exp(np.mean(np.log(finite_regrets)))),
            "p90_regret": float(np.quantile(finite_regrets, 0.90)),
            "p95_regret": float(np.quantile(finite_regrets, 0.95)),
            "worst_regret": float(np.max(finite_regrets)),
            "states_over_1_25x_regret": int(np.sum(finite_regrets > 1.25)),
            "states_over_1_5x_regret": int(np.sum(finite_regrets > 1.5)),
            "negative_threshold_fraction_clipped_to_zero": clipped / len(rows),
        },
        evaluated,
    )


def _regime_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for crn in sorted({str(row["crn"]) for row in rows}):
        group = sorted(
            (row for row in rows if row["crn"] == crn),
            key=lambda row: (float(row["trajectory_time"]), int(row["state_id"])),
        )
        modes = [str(row["oracle_mode"]) for row in group]
        transitions = sum(left != right for left, right in zip(modes, modes[1:]))
        summary[crn] = {
            "states": len(group),
            "batch_optimal": modes.count("batch"),
            "gillespie_optimal": modes.count("gillespie"),
            "sampled_mode_transitions": transitions,
            "mixed_regime": len(set(modes)) > 1,
            "score_range": [
                min(float(row["score"]) for row in group),
                max(float(row["score"]) for row in group),
            ],
            "threshold_range": [
                min(float(row["threshold_T_star"]) for row in group),
                max(float(row["threshold_T_star"]) for row in group),
            ],
        }
    return summary


def _threshold_fit_metrics(target: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    residuals = target - predictions
    return {
        "rmse": float(math.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "median_absolute_error": float(np.median(np.abs(residuals))),
        "target_range": [float(np.min(target)), float(np.max(target))],
    }


def _best_constant_threshold(
    rows: Sequence[dict[str, Any]],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Select a threshold using training rows only and a regret-aware objective."""

    candidates = sorted({0.0, *(float(row["score"]) for row in rows)})
    best: tuple[tuple[float, float, float, float], float, dict[str, Any], list[dict[str, Any]]] | None = None
    for threshold in candidates:
        metrics, evaluations = _regret_metrics(rows, np.full(len(rows), threshold))
        criterion = (
            metrics["geometric_mean_regret"],
            metrics["worst_regret"],
            metrics["mean_regret"],
            threshold,
        )
        candidate = (criterion, threshold, metrics, evaluations)
        if best is None or criterion < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def _grouped_constant_predictions(
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    groups = sorted({str(row["crn"]) for row in rows})
    predictions = np.full(len(rows), np.nan)
    folds: dict[str, Any] = {}
    for held_out in groups:
        train_indices = [index for index, row in enumerate(rows) if row["crn"] != held_out]
        test_indices = [index for index, row in enumerate(rows) if row["crn"] == held_out]
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        threshold, train_metrics, _ = _best_constant_threshold(train_rows)
        fold_predictions = np.full(len(test_rows), threshold)
        predictions[test_indices] = fold_predictions
        test_metrics, _ = _regret_metrics(test_rows, fold_predictions)
        folds[held_out] = {
            "selected_threshold": threshold,
            "training_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
    overall, evaluations = _regret_metrics(rows, predictions)
    return predictions, overall, evaluations, folds


def _grouped_ols_predictions(
    rows: Sequence[dict[str, Any]],
    design: np.ndarray,
    target: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], dict[str, Any], bool]:
    groups = sorted({str(row["crn"]) for row in rows})
    predictions = np.full(len(rows), np.nan)
    folds: dict[str, Any] = {}
    all_identifiable = True
    for held_out in groups:
        train_indices = [index for index, row in enumerate(rows) if row["crn"] != held_out]
        test_indices = [index for index, row in enumerate(rows) if row["crn"] == held_out]
        train_design = design[train_indices]
        train_target = target[train_indices]
        diagnostics = _matrix_diagnostics(train_design, feature_names)
        identifiable = diagnostics["rank"] == len(feature_names)
        all_identifiable &= identifiable
        coefficients, _, _, _ = np.linalg.lstsq(train_design, train_target, rcond=None)
        fold_predictions = design[test_indices] @ coefficients
        predictions[test_indices] = fold_predictions
        test_rows = [rows[index] for index in test_indices]
        test_metrics, _ = _regret_metrics(test_rows, fold_predictions)
        folds[held_out] = {
            "train_matrix": diagnostics,
            "identifiable": identifiable,
            "prediction_status": "identified" if identifiable else "diagnostic_pseudoinverse_only",
            "coefficients": dict(zip(feature_names, coefficients.tolist())),
            "test_metrics": test_metrics,
        }
    overall, evaluations = _regret_metrics(rows, predictions)
    return predictions, overall, evaluations, folds, all_identifiable


def _fixed_threshold_report(
    rows: Sequence[dict[str, Any]], threshold: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = np.full(len(rows), threshold)
    metrics, evaluations = _regret_metrics(rows, predictions)
    folds: dict[str, Any] = {}
    for crn in sorted({str(row["crn"]) for row in rows}):
        group_rows = [row for row in rows if row["crn"] == crn]
        group_metrics, _ = _regret_metrics(group_rows, np.full(len(group_rows), threshold))
        folds[crn] = {"test_metrics": group_metrics}
    return (
        {
            "threshold": threshold,
            "in_sample_decision": metrics,
            "leave_one_crn_out": {
                "overall": metrics,
                "folds": folds,
                "note": "No parameters are trained, so grouped evaluation equals all-data evaluation.",
            },
        },
        evaluations,
    )


def fit_model(rows: list[dict[str, Any]], *, output_prefix: Path) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("at least two frozen states are required")
    design = _design_matrix(rows)
    target = np.asarray([float(row["threshold_T_star"]) for row in rows], dtype=float)

    # Proposed six-feature model.
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    in_sample_metrics, in_sample_evaluations = _regret_metrics(rows, fitted)
    (
        loo_predictions,
        loo_diagnostic_metrics,
        loo_evaluations,
        folds,
        all_folds_identifiable,
    ) = _grouped_ols_predictions(rows, design, target, FEATURE_NAMES)

    # Baseline 1: one constant chosen on training data using the same regret objective as evaluation.
    constant_threshold, constant_in_metrics, constant_in_evaluations = _best_constant_threshold(rows)
    (
        constant_loo_predictions,
        constant_loo_metrics,
        constant_loo_evaluations,
        constant_folds,
    ) = _grouped_constant_predictions(rows)

    # Baseline 2: the population-only OLS model T = beta_0 + beta_1 log2(N).
    log_n_names = ("intercept", "log2_N")
    log_n_design = design[:, :2]
    log_n_coefficients, _, _, _ = np.linalg.lstsq(log_n_design, target, rcond=None)
    log_n_fitted = log_n_design @ log_n_coefficients
    log_n_in_metrics, log_n_in_evaluations = _regret_metrics(rows, log_n_fitted)
    (
        log_n_loo_predictions,
        log_n_loo_diagnostic_metrics,
        log_n_loo_evaluations,
        log_n_folds,
        log_n_all_folds_identifiable,
    ) = _grouped_ols_predictions(rows, log_n_design, target, log_n_names)

    # Previously promising constants are useful pre-registered reference points, not trained models.
    fixed_200_report, fixed_200_evaluations = _fixed_threshold_report(rows, 200.0)
    fixed_300_report, fixed_300_evaluations = _fixed_threshold_report(rows, 300.0)
    fixed_500_report, fixed_500_evaluations = _fixed_threshold_report(rows, 500.0)

    regimes = _regime_summary(rows)
    diagnostics = _matrix_diagnostics(design)
    log_n_diagnostics = _matrix_diagnostics(log_n_design, log_n_names)
    mixed_crns = [crn for crn, values in regimes.items() if values["mixed_regime"]]
    full_cv_status = "identified" if all_folds_identifiable else "diagnostic_pseudoinverse_only"
    log_n_cv_status = "identified" if log_n_all_folds_identifiable else "diagnostic_pseudoinverse_only"

    report = {
        "formula": ("T = theta_0 + theta_1*log2(N) + theta_2*q**o + theta_3*R + theta_4*B + theta_5*o*log2(N)*[g>0]"),
        "target_definition": (
            "(batch engine + batch postprocess seconds) / ((Gillespie engine + urn-sync seconds) / exact reactions)"
        ),
        "excluded_switch_costs": ["K preparation", "Gillespie object/reaction construction"],
        "feature_names": list(FEATURE_NAMES),
        "matrix": diagnostics,
        "coefficients": dict(zip(FEATURE_NAMES, coefficients.tolist())),
        "threshold_fit": _threshold_fit_metrics(target, fitted),
        "in_sample_decision": in_sample_metrics,
        "leave_one_crn_out": {
            # A rank-deficient fold cannot validate the unregularized structural model. Retain its
            # minimum-norm predictions only as a diagnostic, never as the identified CV result.
            "overall": loo_diagnostic_metrics if all_folds_identifiable else None,
            "status": full_cv_status,
            "unavailable_reason": (
                None
                if all_folds_identifiable
                else "At least one training fold is rank deficient for the six-feature model."
            ),
            "diagnostic_pseudoinverse_overall": loo_diagnostic_metrics,
            "folds": folds,
        },
        "baselines": {
            "training_selected_constant": {
                "selection_objective": (
                    "minimize training geometric-mean regret; tie-break worst regret, mean regret, then lower threshold"
                ),
                "all_data_selected_threshold": constant_threshold,
                "threshold_fit": _threshold_fit_metrics(target, np.full(len(rows), constant_threshold)),
                "in_sample_decision": constant_in_metrics,
                "leave_one_crn_out": {
                    "overall": constant_loo_metrics,
                    "folds": constant_folds,
                },
            },
            "log2_n_ols": {
                "formula": "T = beta_0 + beta_1*log2(N)",
                "feature_names": list(log_n_names),
                "matrix": log_n_diagnostics,
                "coefficients": dict(zip(log_n_names, log_n_coefficients.tolist())),
                "threshold_fit": _threshold_fit_metrics(target, log_n_fitted),
                "in_sample_decision": log_n_in_metrics,
                "leave_one_crn_out": {
                    "overall": (log_n_loo_diagnostic_metrics if log_n_all_folds_identifiable else None),
                    "status": log_n_cv_status,
                    "unavailable_reason": (
                        None
                        if log_n_all_folds_identifiable
                        else "At least one training fold is rank deficient for 1+log2(N)."
                    ),
                    "diagnostic_pseudoinverse_overall": log_n_loo_diagnostic_metrics,
                    "folds": log_n_folds,
                },
            },
            "fixed_200": fixed_200_report,
            "fixed_300": fixed_300_report,
            "fixed_500": fixed_500_report,
        },
        "within_trajectory_regimes": regimes,
        "sufficiency": {
            "full_model_identifiable_in_all_data": diagnostics["rank"] == len(FEATURE_NAMES),
            "full_model_identifiable_in_every_leave_one_crn_out_training_fold": all_folds_identifiable,
            "crns_with_both_oracle_modes": mixed_crns,
            "has_two_or_more_mixed_regime_crns": len(mixed_crns) >= 2,
            "interpretation": (
                "A full-rank all-data fit estimates one set of coefficients for these CRNs; "
                "full-rank grouped folds are the stronger check that coefficients are not "
                "identified only by a single held-out structure. Mixed-regime CRNs test the "
                "switching decision but do not by themselves identify CRN-static coefficients."
            ),
        },
    }

    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        augmented = dict(row)
        augmented.update(
            {
                # Preserve the original full-model output columns.
                "fit_threshold": in_sample_evaluations[index]["threshold"],
                "fit_mode": in_sample_evaluations[index]["chosen_mode"],
                "fit_regret": in_sample_evaluations[index]["regret"],
                "loo_threshold": loo_evaluations[index]["threshold"],
                "loo_mode": loo_evaluations[index]["chosen_mode"],
                "loo_regret": loo_evaluations[index]["regret"],
                "loo_prediction_status": full_cv_status,
                # Add directly comparable baseline predictions.
                "constant_fit_threshold": constant_in_evaluations[index]["threshold"],
                "constant_fit_mode": constant_in_evaluations[index]["chosen_mode"],
                "constant_fit_regret": constant_in_evaluations[index]["regret"],
                "constant_loo_threshold": constant_loo_evaluations[index]["threshold"],
                "constant_loo_mode": constant_loo_evaluations[index]["chosen_mode"],
                "constant_loo_regret": constant_loo_evaluations[index]["regret"],
                "log2_n_fit_threshold": log_n_in_evaluations[index]["threshold"],
                "log2_n_fit_mode": log_n_in_evaluations[index]["chosen_mode"],
                "log2_n_fit_regret": log_n_in_evaluations[index]["regret"],
                "log2_n_loo_threshold": log_n_loo_evaluations[index]["threshold"],
                "log2_n_loo_mode": log_n_loo_evaluations[index]["chosen_mode"],
                "log2_n_loo_regret": log_n_loo_evaluations[index]["regret"],
                "log2_n_loo_prediction_status": log_n_cv_status,
                "fixed_200_mode": fixed_200_evaluations[index]["chosen_mode"],
                "fixed_200_regret": fixed_200_evaluations[index]["regret"],
                "fixed_300_mode": fixed_300_evaluations[index]["chosen_mode"],
                "fixed_300_regret": fixed_300_evaluations[index]["regret"],
                "fixed_500_mode": fixed_500_evaluations[index]["chosen_mode"],
                "fixed_500_regret": fixed_500_evaluations[index]["regret"],
            }
        )
        prediction_rows.append(augmented)

    report_path = output_prefix.with_name(output_prefix.name + "_fit.json")
    prediction_path = output_prefix.with_name(output_prefix.name + "_predictions.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(prediction_path, prediction_rows)
    print(f"wrote {report_path}")
    print(f"wrote {prediction_path}")
    print(
        f"full model: design rank {diagnostics['rank']}/{len(FEATURE_NAMES)}, "
        f"grouped-CV status={full_cv_status}, diagnostic mean/worst regret "
        f"{loo_diagnostic_metrics['mean_regret']:.3g}/{loo_diagnostic_metrics['worst_regret']:.3g}"
    )
    print(
        f"selected constant: T={constant_threshold:.4g}, grouped-CV mean/worst regret "
        f"{constant_loo_metrics['mean_regret']:.3g}/{constant_loo_metrics['worst_regret']:.3g}"
    )
    print(
        f"1+log2(N): grouped-CV status={log_n_cv_status}, mean/worst regret "
        f"{log_n_loo_diagnostic_metrics['mean_regret']:.3g}/"
        f"{log_n_loo_diagnostic_metrics['worst_regret']:.3g}"
    )
    for crn, values in regimes.items():
        print(
            f"  {crn}: batch={values['batch_optimal']}, "
            f"gillespie={values['gillespie_optimal']}, "
            f"sampled transitions={values['sampled_mode_transitions']}"
        )
    return report


def _output_prefix(args: argparse.Namespace) -> Path:
    if args.output_prefix is not None:
        return args.output_prefix.resolve()
    return DEFAULT_RESULTS_DIR / f"threshold_model_{args.preset}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run", "collect", "fit"), default="run")
    parser.add_argument(
        "--preset",
        choices=("quick", "matched", "full"),
        default="quick",
        help="quick=3 CRNs; matched=5 controlled dynamic CRNs; full=broad 6-CRN coverage",
    )
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--input-csv", type=Path, help="aggregate states CSV for the fit action")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, help="paired timing repeats per frozen state")
    parser.add_argument("--states-per-crn", type=int)
    parser.add_argument("--candidate-factor", type=int, default=5)
    parser.add_argument(
        "--trajectory-threshold",
        type=float,
        default=200.0,
        help="deterministic selector-2 threshold used only to generate reference trajectories",
    )
    parser.add_argument(
        "--gillespie-reactions",
        type=int,
        help="exact reactions per oracle block (default: sqrt(current real count))",
    )
    parser.add_argument(
        "--crns",
        help="comma-separated slugs to collect (default: every CRN in the selected preset)",
    )
    args = parser.parse_args(argv)
    if args.repeats is None:
        args.repeats = {"quick": 7, "matched": 9, "full": 25}[args.preset]
    if args.states_per_crn is None:
        args.states_per_crn = {"quick": 8, "matched": 8, "full": 24}[args.preset]
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.states_per_crn < 2:
        parser.error("--states-per-crn must be at least 2")
    if args.candidate_factor < 1:
        parser.error("--candidate-factor must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prefix = _output_prefix(args)
    if args.action in ("run", "collect"):
        cases = experiment_cases(args.preset)
        if args.crns:
            wanted = {slug.strip() for slug in args.crns.split(",") if slug.strip()}
            known = {case.slug for case in cases}
            unknown = wanted - known
            if unknown:
                raise SystemExit(f"unknown CRN slug(s) for {args.preset}: {', '.join(sorted(unknown))}")
            cases = [case for case in cases if case.slug in wanted]
        rows = collect(
            cases,
            output_prefix=prefix,
            seed=args.seed,
            repeats=args.repeats,
            states_per_crn=args.states_per_crn,
            candidate_factor=args.candidate_factor,
            trajectory_threshold=args.trajectory_threshold,
            gillespie_reactions=args.gillespie_reactions,
        )
    else:
        input_csv = (
            args.input_csv.resolve() if args.input_csv is not None else prefix.with_name(prefix.name + "_states.csv")
        )
        rows = _numeric_csv_rows(input_csv)
    if args.action in ("run", "fit"):
        fit_model(rows, output_prefix=prefix)


if __name__ == "__main__":
    main()
