"""End-to-end comparison of deterministic and wall-clock switching policies.

This issue-#14 research harness times one uninterrupted ``BatchSimulator.run`` call per trial.
It compares the deterministic prospective score at a threshold sweep with the current wall-clock
heuristic and with forced batch/Gillespie anchors.  Pilot and final phases use disjoint seeds.

Typical workflow from the repository root::

    python benchmark/switching_policy_comparison.py pilot --overwrite-phase
    python benchmark/switching_policy_comparison.py final --overwrite-phase
    python benchmark/switching_policy_comparison.py analyze --phase final

All outputs live directly in ``benchmark/`` so they can be reviewed and committed deliberately.
The timing matrix is resumable by (phase, scenario, policy, seed, repeat).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PYTHON_ROOT = REPO_ROOT / "python"
for search_path in (PYTHON_ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import batss as bt  # noqa: E402
from threshold_model import ExperimentCase, experiment_cases  # noqa: E402


RUNS_PATH = SCRIPT_DIR / "switching_policy_comparison_runs.csv"
OUTPUT_PREFIX = SCRIPT_DIR / "switching_policy_comparison"
METADATA_PATH = SCRIPT_DIR / "switching_policy_comparison_metadata.json"
SELECTION_PATH = SCRIPT_DIR / "switching_policy_comparison_selection.json"

FORCED_BATCH_THRESHOLD = 0.0
FORCED_GILLESPIE_THRESHOLD = 1.0e12
PILOT_THRESHOLDS = (
    0.0,
    20.0,
    50.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
    400.0,
    500.0,
    750.0,
    1_000.0,
    1_500.0,
    2_000.0,
    FORCED_GILLESPIE_THRESHOLD,
)

RUN_FIELDS = (
    "phase",
    "scenario",
    "family",
    "crn",
    "initial_n",
    "end_time",
    "policy",
    "policy_kind",
    "heuristic_selector",
    "threshold",
    "seed",
    "repeat",
    "run_order",
    "started_utc",
    "setup_seconds",
    "elapsed_seconds",
    "completed",
    "timed_out",
    "continuous_time",
    "continuous_time_error",
    "real_n_final",
    "full_n_final",
    "final_mode_gillespie",
    "silent",
    "batch_calls",
    "gillespie_calls",
    "mode_switches",
    "k_resets",
    "batch_continuous_time",
    "gillespie_continuous_time",
    "batch_wallclock_seconds",
    "gillespie_wallclock_seconds",
    "switch_overhead_seconds",
    "unaccounted_outer_seconds",
    "full_config_sha256",
    "mode_signature_sha256",
    "source_fingerprint",
    "error",
)


@dataclass(frozen=True)
class Scenario:
    slug: str
    family: str
    case: ExperimentCase


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    selector: int
    threshold: float | None


def _replace_case(base: ExperimentCase, *, initial_n: int, end_time: float) -> ExperimentCase:
    return ExperimentCase(base.slug, base.spec, initial_n, end_time)


def comparison_scenarios() -> list[Scenario]:
    matched = {case.slug: case for case in experiment_cases("matched")}
    broad = {case.slug: case for case in experiment_cases("full")}

    scenarios = [
        Scenario(
            "oregonator_n1e5", "oregonator", _replace_case(matched["oregonator"], initial_n=100_000, end_time=1.0)
        ),
        Scenario("rossler_n1e5", "rossler", _replace_case(matched["rossler"], initial_n=100_000, end_time=1.0)),
        Scenario("shrinking_n2e6", "shrinking", _replace_case(matched["shrinking"], initial_n=2_000_000, end_time=0.1)),
        Scenario(
            "shrinking_no_b_decay_n2e6",
            "shrinking",
            _replace_case(matched["shrinking_no_b_decay"], initial_n=2_000_000, end_time=0.1),
        ),
        Scenario(
            "shrinking_split_b_decay_n2e6",
            "shrinking",
            _replace_case(matched["shrinking_split_b_decay"], initial_n=2_000_000, end_time=0.1),
        ),
        Scenario(
            "dimerization_n1e8",
            "dimerization",
            _replace_case(broad["dimerization"], initial_n=100_000_000, end_time=0.5),
        ),
    ]
    lotka = broad["lotka_volterra"]
    for n in (100_000, 1_000_000, 10_000_000):
        scenarios.append(
            Scenario(
                f"lotka_volterra_n{n}",
                "lotka_volterra",
                _replace_case(lotka, initial_n=n, end_time=1.0),
            )
        )
    scenarios.append(
        Scenario(
            "brusselator_n2e4",
            "brusselator",
            _replace_case(broad["brusselator"], initial_n=20_000, end_time=20.0),
        )
    )
    return scenarios


def _threshold_label(value: float) -> str:
    if value == FORCED_BATCH_THRESHOLD:
        return "forced_batch"
    if value == FORCED_GILLESPIE_THRESHOLD:
        return "forced_gillespie_after_initial_batch"
    if value.is_integer():
        return f"threshold_{int(value)}"
    return f"threshold_{value:g}".replace(".", "p")


def prospective_policy(threshold: float) -> Policy:
    return Policy(_threshold_label(threshold), "prospective", 2, threshold)


def wallclock_policy() -> Policy:
    return Policy("wallclock_timing", "wallclock", 0, None)


def pilot_policies(thresholds: Sequence[float] = PILOT_THRESHOLDS) -> list[Policy]:
    return [prospective_policy(float(value)) for value in thresholds] + [wallclock_policy()]


def _snap_below(selected: float, factor: float, candidates: Sequence[float]) -> float:
    target = selected / factor
    choices = [value for value in candidates if 0.0 < value < selected and value <= target]
    return max(choices) if choices else target


def _snap_above(selected: float, factor: float, candidates: Sequence[float]) -> float:
    target = selected * factor
    choices = [value for value in candidates if selected < value < FORCED_GILLESPIE_THRESHOLD and value >= target]
    return min(choices) if choices else target


def final_thresholds(selected: float) -> list[float]:
    tunable = [value for value in PILOT_THRESHOLDS if 0.0 < value < FORCED_GILLESPIE_THRESHOLD]
    values = [
        FORCED_BATCH_THRESHOLD,
        _snap_below(selected, 4.0, tunable),
        _snap_below(selected, 2.0, tunable),
        selected,
        _snap_above(selected, 2.0, tunable),
        _snap_above(selected, 4.0, tunable),
        FORCED_GILLESPIE_THRESHOLD,
    ]
    return list(dict.fromkeys(float(value) for value in values))


def final_policies(selected: float) -> list[Policy]:
    return [prospective_policy(value) for value in final_thresholds(selected)] + [wallclock_policy()]


def _policy_role(phase: str, policy_name: str, threshold: float, selected: float | None) -> str:
    if policy_name == "wallclock_timing":
        return "timing_reference"
    if threshold == FORCED_BATCH_THRESHOLD:
        return "forced_batch_anchor"
    if threshold == FORCED_GILLESPIE_THRESHOLD:
        return "forced_gillespie_anchor"
    if phase != "final" or selected is None:
        return "pilot_candidate"
    if threshold == selected:
        return "selected_T"
    if threshold < selected:
        lower = sorted(value for value in final_thresholds(selected) if 0.0 < value < selected)
        return "low_far_T_l" if lower and threshold == lower[0] else "low_near_T_l"
    higher = sorted(value for value in final_thresholds(selected) if selected < value < FORCED_GILLESPIE_THRESHOLD)
    return "high_far_T_h" if higher and threshold == higher[-1] else "high_near_T_h"


def _csv_numbers(text: str, converter: Any) -> tuple[Any, ...]:
    values = tuple(converter(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _stable_int(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        REPO_ROOT / "src" / "simulator_crn.rs",
        SCRIPT_DIR / "threshold_model.py",
        REPO_ROOT / "python" / "batss" / "simulation.py",
    ):
        digest.update(str(path.relative_to(REPO_ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _extension_info() -> dict[str, Any]:
    spec = importlib.util.find_spec("batss.batss_rust.batss_rust")
    if spec is None or spec.origin is None:
        return {"path": None, "sha256": None}
    path = Path(spec.origin).resolve()
    return {"path": str(path), "sha256": _sha256_file(path) if path.is_file() else None}


def _metadata() -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "source_fingerprint": _source_fingerprint(),
        "extension": _extension_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pilot_thresholds": list(PILOT_THRESHOLDS),
        "pilot_default_seeds": [1, 2, 3],
        "final_default_seeds": list(range(101, 107)),
        "default_repeats": 2,
        "timer_scope": "one uninterrupted raw BatchSimulator.run call; construction excluded",
        "anchor_note": "forced Gillespie still executes the simulator's mandatory initial batch",
        "selection_note": "wall-clock policy is a reference and never selects T",
        "scenarios": [
            {
                "scenario": scenario.slug,
                "family": scenario.family,
                "crn": scenario.case.spec.name,
                "initial_n": scenario.case.initial_n,
                "end_time": scenario.case.end_time,
            }
            for scenario in comparison_scenarios()
        ],
    }


def _make_sim(scenario: Scenario, seed: int) -> Any:
    case = scenario.case
    return bt.Simulation(
        case.spec.inits_from_n(case.initial_n),
        case.spec.rxns,
        simulator_method="crn",
        continuous_time=True,
        seed=seed,
        volume=case.initial_n,
    )


def _configure_policy(sim: Any, policy: Policy) -> None:
    sim.simulator.heuristic_gillespie_switching = policy.selector
    if policy.threshold is not None:
        sim.simulator.proxy_threshold = policy.threshold


def _config_hash(sim: Any) -> str:
    config = np.asarray(sim.simulator.config, dtype="<u8")
    return hashlib.sha256(config.tobytes()).hexdigest()


def _mode_hash(sim: Any) -> str:
    switch = sim.simulator.switch
    payload = {
        "batch_calls": int(switch.batch_calls),
        "gillespie_calls": int(switch.gillespie_calls),
        "mode_switches": int(switch.mode_switches),
        "batch_continuous_time": float(switch.batch_continuous_time),
        "gillespie_continuous_time": float(switch.gillespie_continuous_time),
        "final_mode_gillespie": bool(sim.simulator.do_gillespie),
        "k_resets": int(sim.simulator.k_resets),
        "real_n": int(sim.simulator.n),
        "continuous_time": float(sim.simulator.continuous_time),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _empty_run_row(
    phase: str,
    scenario: Scenario,
    policy: Policy,
    seed: int,
    repeat: int,
    run_order: int,
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "scenario": scenario.slug,
        "family": scenario.family,
        "crn": scenario.case.spec.name,
        "initial_n": scenario.case.initial_n,
        "end_time": scenario.case.end_time,
        "policy": policy.name,
        "policy_kind": policy.kind,
        "heuristic_selector": policy.selector,
        "threshold": "" if policy.threshold is None else policy.threshold,
        "seed": seed,
        "repeat": repeat,
        "run_order": run_order,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "setup_seconds": math.nan,
        "elapsed_seconds": math.nan,
        "completed": False,
        "timed_out": False,
        "continuous_time": math.nan,
        "continuous_time_error": math.nan,
        "real_n_final": "",
        "full_n_final": "",
        "final_mode_gillespie": "",
        "silent": "",
        "batch_calls": "",
        "gillespie_calls": "",
        "mode_switches": "",
        "k_resets": "",
        "batch_continuous_time": math.nan,
        "gillespie_continuous_time": math.nan,
        "batch_wallclock_seconds": math.nan,
        "gillespie_wallclock_seconds": math.nan,
        "switch_overhead_seconds": math.nan,
        "unaccounted_outer_seconds": math.nan,
        "full_config_sha256": "",
        "mode_signature_sha256": "",
        "source_fingerprint": fingerprint,
        "error": "",
    }


def _run_one(
    phase: str,
    scenario: Scenario,
    policy: Policy,
    seed: int,
    repeat: int,
    run_order: int,
    cap_seconds: float,
    fingerprint: str,
) -> dict[str, Any]:
    row = _empty_run_row(phase, scenario, policy, seed, repeat, run_order, fingerprint)
    sim: Any | None = None
    try:
        setup_start = time.perf_counter_ns()
        sim = _make_sim(scenario, seed)
        _configure_policy(sim, policy)
        row["setup_seconds"] = (time.perf_counter_ns() - setup_start) / 1e9

        run_start = time.perf_counter_ns()
        sim.simulator.run(scenario.case.end_time, cap_seconds)
        elapsed = (time.perf_counter_ns() - run_start) / 1e9
        row["elapsed_seconds"] = elapsed

        rust = sim.simulator
        switch = rust.switch
        continuous_time = float(rust.continuous_time)
        tolerance = 1e-12 * max(1.0, abs(scenario.case.end_time))
        completed = continuous_time + tolerance >= scenario.case.end_time
        row.update(
            {
                "completed": completed,
                "timed_out": not completed and elapsed >= 0.9 * cap_seconds,
                "continuous_time": continuous_time,
                "continuous_time_error": continuous_time - scenario.case.end_time,
                "real_n_final": int(rust.n),
                "full_n_final": int(rust.n_including_extra_species),
                "final_mode_gillespie": bool(rust.do_gillespie),
                "silent": bool(rust.silent),
                "batch_calls": int(switch.batch_calls),
                "gillespie_calls": int(switch.gillespie_calls),
                "mode_switches": int(switch.mode_switches),
                "k_resets": int(rust.k_resets),
                "batch_continuous_time": float(switch.batch_continuous_time),
                "gillespie_continuous_time": float(switch.gillespie_continuous_time),
                "batch_wallclock_seconds": float(switch.batch_wallclock_seconds),
                "gillespie_wallclock_seconds": float(switch.gillespie_wallclock_seconds),
                "switch_overhead_seconds": float(switch.switch_overhead_seconds),
                "full_config_sha256": _config_hash(sim),
                "mode_signature_sha256": _mode_hash(sim),
            }
        )
        accounted = (
            float(row["batch_wallclock_seconds"])
            + float(row["gillespie_wallclock_seconds"])
            + float(row["switch_overhead_seconds"])
        )
        row["unaccounted_outer_seconds"] = elapsed - accounted
        if not completed:
            row["error"] = "wall-clock cap reached before t_max"
    except Exception as error:  # keep a long matrix resumable after one failed cell
        row["error"] = f"{type(error).__name__}: {error}"
        if sim is not None:
            try:
                row["continuous_time"] = float(sim.simulator.continuous_time)
            except Exception:
                pass
    return row


def _read_runs(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _rewrite_without_phase(path: Path, phase: str) -> None:
    rows = [row for row in _read_runs(path) if row.get("phase") != phase]
    if not rows:
        if path.exists():
            path.unlink()
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _warm_up(scenarios: Sequence[Scenario]) -> None:
    scenario = next(item for item in scenarios if item.family == "dimerization")
    warm_case = ExperimentCase(scenario.case.slug, scenario.case.spec, 10_000, 0.001)
    warm_scenario = Scenario("warmup", "warmup", warm_case)
    for policy in (prospective_policy(500.0), wallclock_policy()):
        sim = _make_sim(warm_scenario, 9_999)
        _configure_policy(sim, policy)
        sim.simulator.run(warm_case.end_time, 10.0)


def run_matrix(
    *,
    phase: str,
    scenarios: Sequence[Scenario],
    policies: Sequence[Policy],
    seeds: Sequence[int],
    repeats: int,
    runs_path: Path,
    cap_seconds: float,
    overwrite_phase: bool,
    max_runs: int | None,
    dry_run: bool,
    warm_up: bool,
) -> int:
    if overwrite_phase and not dry_run:
        _rewrite_without_phase(runs_path, phase)
    existing = _read_runs(runs_path)
    completed_keys = {
        (row["phase"], row["scenario"], row["policy"], int(row["seed"]), int(row["repeat"])) for row in existing
    }
    run_order = max((int(row["run_order"]) for row in existing), default=0)
    fingerprint = _source_fingerprint()

    blocks = [(scenario, seed, repeat) for scenario in scenarios for seed in seeds for repeat in range(repeats)]
    random.Random(_stable_int("blocks", phase)).shuffle(blocks)
    jobs: list[tuple[Scenario, Policy, int, int]] = []
    for scenario, seed, repeat in blocks:
        block_policies = list(policies)
        random.Random(_stable_int("policies", phase, scenario.slug, seed, repeat)).shuffle(block_policies)
        jobs.extend((scenario, policy, seed, repeat) for policy in block_policies)
    pending = [job for job in jobs if (phase, job[0].slug, job[1].name, job[2], job[3]) not in completed_keys]
    if max_runs is not None:
        pending = pending[:max_runs]
    print(
        f"{phase}: {len(pending)} new runs ({len(scenarios)} scenarios, {len(policies)} policies, "
        f"{len(seeds)} seeds, {repeats} repeats)",
        flush=True,
    )
    if dry_run:
        for scenario, policy, seed, repeat in pending:
            print(scenario.slug, policy.name, seed, repeat)
        return 0
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(_metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if warm_up and pending:
        _warm_up(scenarios)

    write_header = not runs_path.exists() or runs_path.stat().st_size == 0
    new_runs = 0
    with runs_path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RUN_FIELDS)
        if write_header:
            writer.writeheader()
        for scenario, policy, seed, repeat in pending:
            run_order += 1
            row = _run_one(
                phase,
                scenario,
                policy,
                seed,
                repeat,
                run_order,
                cap_seconds,
                fingerprint,
            )
            writer.writerow(row)
            output.flush()
            new_runs += 1
            status = "ok" if row["completed"] else "FAILED"
            print(
                f"[{run_order}] {scenario.slug} {policy.name} seed={seed} rep={repeat}: "
                f"{float(row['elapsed_seconds']):.4g}s {status}",
                flush=True,
            )
    return new_runs


def _as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        return 0


def _truth(value: str) -> bool:
    return value.strip().lower() == "true"


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else math.nan


def _bootstrap_median_ci(values: Sequence[float], seed: int, draws: int = 2_000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(draws, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _geomean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value) and value > 0.0]
    return math.exp(sum(math.log(value) for value in finite) / len(finite)) if finite else math.inf


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hierarchical_runtime(rows: Sequence[dict[str, str]]) -> tuple[float, list[float]]:
    seed_medians: list[float] = []
    for seed in sorted({_as_int(row, "seed") for row in rows}):
        values = [_as_float(row, "elapsed_seconds") for row in rows if _as_int(row, "seed") == seed]
        values = [value for value in values if math.isfinite(value)]
        if values:
            seed_medians.append(statistics.median(values))
    return (statistics.median(seed_medians) if seed_medians else math.inf), seed_medians


def summarize_phase(
    *,
    phase: str,
    runs_path: Path,
    output_prefix: Path,
    selection_path: Path,
    selected_threshold: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    all_rows = [row for row in _read_runs(runs_path) if row["phase"] == phase]
    if not all_rows:
        raise ValueError(f"no {phase!r} rows in {runs_path}")
    scenario_lookup = {scenario.slug: scenario for scenario in comparison_scenarios()}
    policies = sorted({row["policy"] for row in all_rows})
    scenario_names = sorted({row["scenario"] for row in all_rows})

    base: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario_name in scenario_names:
        scenario = scenario_lookup[scenario_name]
        for policy_name in policies:
            group = [row for row in all_rows if row["scenario"] == scenario_name and row["policy"] == policy_name]
            if not group:
                continue
            complete = [row for row in group if _truth(row["completed"]) and not row["error"]]
            runtime, seed_medians = _hierarchical_runtime(complete)
            raw_times = [_as_float(row, "elapsed_seconds") for row in complete]
            ci_lo, ci_hi = _bootstrap_median_ci(seed_medians, _stable_int(phase, scenario_name, policy_name))
            replay_groups = 0
            replay_matches = 0
            for seed in sorted({_as_int(row, "seed") for row in complete}):
                seed_rows = [row for row in complete if _as_int(row, "seed") == seed]
                if len(seed_rows) >= 2:
                    replay_groups += 1
                    config_hashes = {row["full_config_sha256"] for row in seed_rows}
                    mode_hashes = {row["mode_signature_sha256"] for row in seed_rows}
                    replay_matches += len(config_hashes) == 1 and len(mode_hashes) == 1
            threshold = _as_float(group[0], "threshold")
            base[(scenario_name, policy_name)] = {
                "phase": phase,
                "scenario": scenario_name,
                "family": scenario.family,
                "crn": scenario.case.spec.name,
                "initial_n": scenario.case.initial_n,
                "end_time": scenario.case.end_time,
                "policy": policy_name,
                "policy_kind": group[0]["policy_kind"],
                "policy_role": _policy_role(phase, policy_name, threshold, selected_threshold),
                "threshold": "" if not math.isfinite(threshold) else threshold,
                "run_count": len(group),
                "complete_count": len(complete),
                "timeout_count": sum(_truth(row["timed_out"]) for row in group),
                "error_count": sum(bool(row["error"]) for row in group),
                "seed_count": len(seed_medians),
                "median_seconds": runtime,
                "q1_seconds": _quantile(raw_times, 0.25),
                "q3_seconds": _quantile(raw_times, 0.75),
                "mad_seconds": statistics.median([abs(value - statistics.median(raw_times)) for value in raw_times])
                if raw_times
                else math.nan,
                "seed_bootstrap_ci95_low": ci_lo,
                "seed_bootstrap_ci95_high": ci_hi,
                "median_batch_continuous_fraction": statistics.median(
                    [_as_float(row, "batch_continuous_time") / scenario.case.end_time for row in complete]
                )
                if complete
                else math.nan,
                "median_gillespie_continuous_fraction": statistics.median(
                    [_as_float(row, "gillespie_continuous_time") / scenario.case.end_time for row in complete]
                )
                if complete
                else math.nan,
                "median_batch_calls": statistics.median([_as_int(row, "batch_calls") for row in complete])
                if complete
                else math.nan,
                "median_gillespie_calls": statistics.median([_as_int(row, "gillespie_calls") for row in complete])
                if complete
                else math.nan,
                "median_mode_switches": statistics.median([_as_int(row, "mode_switches") for row in complete])
                if complete
                else math.nan,
                "median_switch_overhead_seconds": statistics.median(
                    [_as_float(row, "switch_overhead_seconds") for row in complete]
                )
                if complete
                else math.nan,
                "replay_seed_groups": replay_groups,
                "replay_matching_groups": replay_matches,
                "replay_match_fraction": replay_matches / replay_groups if replay_groups else math.nan,
                "unique_final_hashes": len({row["full_config_sha256"] for row in complete}),
                "unique_mode_hashes": len({row["mode_signature_sha256"] for row in complete}),
            }

    case_rows: list[dict[str, Any]] = []
    selected_name = _threshold_label(selected_threshold) if selected_threshold is not None else None
    for scenario_name in scenario_names:
        scenario_entries = [entry for (name, _), entry in base.items() if name == scenario_name]
        valid_times = [entry["median_seconds"] for entry in scenario_entries if math.isfinite(entry["median_seconds"])]
        best_observed = min(valid_times, default=math.inf)
        batch_anchor = base.get((scenario_name, "forced_batch"), {}).get("median_seconds", math.inf)
        gill_anchor = base.get((scenario_name, "forced_gillespie_after_initial_batch"), {}).get(
            "median_seconds", math.inf
        )
        anchor_best = min(batch_anchor, gill_anchor)
        wallclock = base.get((scenario_name, "wallclock_timing"), {}).get("median_seconds", math.inf)
        selected = (
            base.get((scenario_name, selected_name), {}).get("median_seconds", math.inf) if selected_name else math.inf
        )
        for entry in scenario_entries:
            runtime = entry["median_seconds"]
            augmented = dict(entry)
            augmented.update(
                {
                    "anchor_best_seconds": anchor_best,
                    "anchor_normalized_ratio": runtime / anchor_best if math.isfinite(anchor_best) else math.inf,
                    "ratio_to_wallclock": runtime / wallclock if math.isfinite(wallclock) else math.nan,
                    "ratio_to_selected_threshold": runtime / selected if math.isfinite(selected) else math.nan,
                    "best_observed_seconds": best_observed,
                    "best_observed_ratio": runtime / best_observed if math.isfinite(best_observed) else math.inf,
                }
            )
            case_rows.append(augmented)

    policy_rows: list[dict[str, Any]] = []
    for policy_name in policies:
        entries = [row for row in case_rows if row["policy"] == policy_name]
        family_anchor: dict[str, float] = {}
        family_wallclock: dict[str, float] = {}
        for family in sorted({row["family"] for row in entries}):
            family_entries = [row for row in entries if row["family"] == family]
            family_anchor[family] = max(float(row["anchor_normalized_ratio"]) for row in family_entries)
            finite_wall = [
                float(row["ratio_to_wallclock"])
                for row in family_entries
                if math.isfinite(float(row["ratio_to_wallclock"]))
            ]
            family_wallclock[family] = max(finite_wall) if finite_wall else math.inf
        ratios = list(family_anchor.values())
        logs = sorted((math.log(value) for value in ratios), reverse=True)
        top_two = statistics.mean(logs[:2]) if logs else math.inf
        worst_entry = max(entries, key=lambda row: float(row["anchor_normalized_ratio"]))
        threshold = entries[0]["threshold"] if entries else ""
        switch_values = [
            float(row["median_mode_switches"]) for row in entries if math.isfinite(float(row["median_mode_switches"]))
        ]
        policy_rows.append(
            {
                "phase": phase,
                "policy": policy_name,
                "policy_kind": entries[0]["policy_kind"],
                "policy_role": entries[0]["policy_role"],
                "threshold": threshold,
                "scenario_count": len(entries),
                "family_count": len(family_anchor),
                "complete_runs": sum(int(row["complete_count"]) for row in entries),
                "total_runs": sum(int(row["run_count"]) for row in entries),
                "equal_family_geomean_anchor_ratio": _geomean(ratios),
                "top_two_family_log_objective": top_two,
                "p90_family_anchor_ratio": _quantile(ratios, 0.90),
                "worst_family_anchor_ratio": max(ratios, default=math.inf),
                "worst_family": max(family_anchor, key=family_anchor.get) if family_anchor else "",
                "worst_scenario": worst_entry["scenario"],
                "worst_scenario_anchor_ratio": worst_entry["anchor_normalized_ratio"],
                "equal_family_geomean_ratio_to_wallclock": _geomean(family_wallclock.values()),
                "families_over_1p1": sum(value > 1.1 for value in ratios),
                "families_over_1p25": sum(value > 1.25 for value in ratios),
                "families_over_1p5": sum(value > 1.5 for value in ratios),
                "families_over_2": sum(value > 2.0 for value in ratios),
                "median_mode_switches": statistics.median(switch_values) if switch_values else math.nan,
            }
        )

    selection: dict[str, Any] | None = None
    if phase == "pilot":
        candidates = [
            row
            for row in policy_rows
            if row["policy_kind"] == "prospective"
            and 0.0 < float(row["threshold"]) < FORCED_GILLESPIE_THRESHOLD
            and int(row["complete_runs"]) == int(row["total_runs"])
        ]
        if not candidates:
            raise ValueError("pilot has no complete tunable threshold")
        best_objective = min(float(row["top_two_family_log_objective"]) for row in candidates)
        plateau = [
            row for row in candidates if float(row["top_two_family_log_objective"]) <= best_objective + math.log(1.05)
        ]
        center = _geomean(float(row["threshold"]) for row in plateau)
        selected_row = min(
            plateau,
            key=lambda row: (
                round(abs(math.log(float(row["threshold"]) / center)), 12),
                float(row["equal_family_geomean_anchor_ratio"]),
                float(row["median_mode_switches"]),
                abs(float(row["threshold"]) - 500.0),
            ),
        )
        selected_value = float(selected_row["threshold"])
        selection = {
            "selected_threshold": selected_value,
            "selected_policy": selected_row["policy"],
            "best_top_two_family_log_objective": best_objective,
            "plateau_relative_tolerance": 1.05,
            "plateau_thresholds": [float(row["threshold"]) for row in plateau],
            "plateau_geometric_center": center,
            "final_thresholds": final_thresholds(selected_value),
            "final_policies": [policy.name for policy in final_policies(selected_value)],
            "wallclock_excluded_from_selection": True,
            "selection_generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_fingerprint": _source_fingerprint(),
        }

    case_path = output_prefix.with_name(output_prefix.name + f"_{phase}_case_summary.csv")
    policy_path = output_prefix.with_name(output_prefix.name + f"_{phase}_policy_summary.csv")
    _write_rows(case_path, case_rows)
    _write_rows(policy_path, policy_rows)
    print(f"wrote {case_path}")
    print(f"wrote {policy_path}")
    if selection is not None:
        selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"selected T={selection['selected_threshold']:g}; wrote {selection_path}")
    return case_rows, policy_rows, selection


def _selected_threshold(value: float | None, selection_path: Path) -> float:
    if value is not None:
        return value
    if not selection_path.is_file():
        raise ValueError(f"missing pilot selection: {selection_path}; run pilot first or pass --selected-threshold")
    return float(json.loads(selection_path.read_text(encoding="utf-8"))["selected_threshold"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pilot", "final", "analyze", "list"))
    parser.add_argument("--phase", choices=("pilot", "final"), help="phase for analyze")
    parser.add_argument("--runs", type=Path, default=RUNS_PATH)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument("--selected-threshold", type=float)
    parser.add_argument("--thresholds", type=lambda text: _csv_numbers(text, float))
    parser.add_argument("--seeds", type=lambda text: _csv_numbers(text, int))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cases", help="comma-separated scenario subset")
    parser.add_argument("--cap-seconds", type=float, default=30.0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--overwrite-phase", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.cap_seconds <= 0:
        parser.error("--cap-seconds must be positive")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scenarios = comparison_scenarios()
    if args.cases:
        wanted = {part.strip() for part in args.cases.split(",") if part.strip()}
        known = {scenario.slug for scenario in scenarios}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"unknown scenario(s): {', '.join(sorted(unknown))}")
        scenarios = [scenario for scenario in scenarios if scenario.slug in wanted]

    if args.action == "list":
        for scenario in scenarios:
            print(
                f"{scenario.slug}: family={scenario.family}, n={scenario.case.initial_n}, "
                f"end={scenario.case.end_time}, crn={scenario.case.spec.name}"
            )
        return

    runs_path = args.runs.resolve()
    output_prefix = args.output_prefix.resolve()
    selection_path = args.selection.resolve()
    if args.action == "analyze":
        phase = args.phase or "final"
        selected = None
        if phase == "final":
            selected = _selected_threshold(args.selected_threshold, selection_path)
        summarize_phase(
            phase=phase,
            runs_path=runs_path,
            output_prefix=output_prefix,
            selection_path=selection_path,
            selected_threshold=selected,
        )
        return

    if args.action == "pilot":
        phase = "pilot"
        seeds = args.seeds or (1, 2, 3)
        thresholds = args.thresholds or PILOT_THRESHOLDS
        policies = pilot_policies(thresholds)
        selected = None
    else:
        phase = "final"
        seeds = args.seeds or tuple(range(101, 107))
        selected = _selected_threshold(args.selected_threshold, selection_path)
        policies = final_policies(selected)

    run_matrix(
        phase=phase,
        scenarios=scenarios,
        policies=policies,
        seeds=seeds,
        repeats=args.repeats,
        runs_path=runs_path,
        cap_seconds=args.cap_seconds,
        overwrite_phase=args.overwrite_phase,
        max_runs=args.max_runs,
        dry_run=args.dry_run,
        warm_up=not args.no_warmup,
    )
    if not args.dry_run and args.max_runs is None:
        summarize_phase(
            phase=phase,
            runs_path=runs_path,
            output_prefix=output_prefix,
            selection_path=selection_path,
            selected_threshold=selected,
        )


if __name__ == "__main__":
    main()
