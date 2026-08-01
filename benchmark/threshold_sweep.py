"""Threshold sweep for the deterministic batch/Gillespie switching experiments.

This is an issue-#14 research harness, not part of batss's public API.  It reuses the
authoritative CRN definitions in ``generate_gallery_figures.py`` and writes resumable
JSON Lines records outside ``benchmark/data`` so it cannot overwrite the gallery's
committed runtime caches.

Typical invocations from the repository root::

    python benchmark/threshold_sweep.py --preset quick --heuristic prospective
    python benchmark/threshold_sweep.py --preset full --heuristic prospective
    python benchmark/threshold_sweep.py --preset full --heuristic proxy

The quick preset is deliberately bounded: 32 short runs at most, a 5-second cap per
run, and a 120-second cap for the invocation.  The full preset evaluates every
threshold requested in issue #14 over three sizes and three seeds per CRN; it can take
hours, so it must be selected explicitly.

Selector 2 is experimental and older extensions silently interpret unknown selectors
as the wall-clock policy.  To prevent invalid results, ``--heuristic prospective``
requires the new ``prospective_batch_score`` method to exist on ``BatchSimulator``.
Rebuild the Rust extension before running this harness after editing ``src/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
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
PYTHON_DIR = REPO_ROOT / "python"

# Prefer the checked-out Python package and make the sibling gallery module importable
# whether this file is invoked as ``python benchmark/threshold_sweep.py`` or as a module.
for search_path in (PYTHON_DIR, SCRIPT_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

import batss  # noqa: E402  (the checkout path is installed above)
from batss.benchmarking import CRNSpec, _batss_sim  # noqa: E402
from generate_gallery_figures import gallery_specs  # noqa: E402


HEURISTIC_SELECTORS = {
    "proxy": 1,
    "prospective": 2,
}

FULL_THRESHOLDS = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    300.0,
    500.0,
    750.0,
    1_000.0,
    1_500.0,
    2_000.0,
    3_000.0,
    5_000.0,
)

SPEC_ORDER = (
    "dimerization",
    "oregonator",
    "lotka_volterra",
    "rossler",
)

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "thresholds": (20.0, 200.0, 500.0, 2_000.0),
        "sizes": {
            "dimerization": (100_000,),
            "oregonator": (10_000,),
            "lotka_volterra": (100_000,),
            "rossler": (1_000,),
        },
        "seeds": (1,),
        "repeats": 2,
        "end_time_scale": 0.1,
        "max_run_seconds": 5.0,
        "max_total_seconds": 120.0,
    },
    "full": {
        "thresholds": FULL_THRESHOLDS,
        "sizes": {
            "dimerization": (100_000, 10_000_000, 1_000_000_000),
            "oregonator": (1_000, 10_000, 100_000),
            "lotka_volterra": (100_000, 10_000_000, 1_000_000_000),
            "rossler": (1_000, 10_000, 100_000),
        },
        "seeds": (1, 2, 3),
        "repeats": 2,
        "end_time_scale": 1.0,
        "max_run_seconds": 300.0,
        "max_total_seconds": None,
    },
}

DEFAULT_OUTPUT = SCRIPT_DIR / "results" / "issue14_threshold_sweep.jsonl"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunCase:
    spec: CRNSpec
    slug: str
    heuristic: str
    selector: int
    population_size: int
    seed: int
    threshold: float
    repeat: int
    end_time: float
    history_samples: int
    max_run_seconds: float


def _csv_values(text: str, convert: Any) -> tuple[Any, ...]:
    values = tuple(convert(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _csv_floats(text: str) -> tuple[float, ...]:
    try:
        values = _csv_values(text, float)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("thresholds must be finite and non-negative")
    return values


def _csv_positive_ints(text: str) -> tuple[int, ...]:
    try:
        values = _csv_values(text, int)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _csv_specs(text: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower().replace("-", "_") for part in text.split(",") if part.strip())
    unknown = sorted(set(values) - set(SPEC_ORDER))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown spec(s): {', '.join(unknown)}; choose from {', '.join(SPEC_ORDER)}"
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="quick",
        help="quick is bounded and small; full uses the complete issue-#14 matrix",
    )
    parser.add_argument(
        "--heuristic",
        choices=tuple(HEURISTIC_SELECTORS),
        default="prospective",
        help="deterministic score to sweep (selector 2 requires a freshly rebuilt extension)",
    )
    parser.add_argument(
        "--specs",
        type=_csv_specs,
        help=f"comma-separated subset of: {', '.join(SPEC_ORDER)}",
    )
    parser.add_argument("--thresholds", type=_csv_floats, help="comma-separated threshold override")
    parser.add_argument(
        "--sizes",
        type=_csv_positive_ints,
        help="comma-separated population sizes applied to every selected CRN",
    )
    parser.add_argument("--seeds", type=_csv_positive_ints, help="comma-separated seed override")
    parser.add_argument("--repeats", type=int, help="fresh-simulator repeats per case (default: 2)")
    parser.add_argument(
        "--end-time-scale",
        type=float,
        help="multiply each CRN's authoritative benchmark_end_time by this value",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=1,
        help="equally spaced trajectory checkpoints, excluding t=0 (default: 1 for clean timings)",
    )
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        help="approximate wall-clock cap for each run; checked between Rust engine calls",
    )
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        help="stop starting new runs after this invocation-level budget (0 means unlimited)",
    )
    parser.add_argument("--max-runs", type=int, help="maximum new (non-cached) runs this invocation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="resumable JSONL output path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="discard an existing output file instead of resuming it",
    )
    parser.add_argument("--no-warmup", action="store_true", help="skip the one untimed warm-up run")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix without running or writing")
    return parser


def _authoritative_specs() -> dict[str, CRNSpec]:
    specs = {slug: spec for spec, slug, _end, _loc, _aspect in gallery_specs() if slug in SPEC_ORDER}
    missing = sorted(set(SPEC_ORDER) - set(specs))
    if missing:
        raise RuntimeError(f"gallery_specs() is missing expected CRNs: {', '.join(missing)}")
    return specs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "src" / "simulator_crn.rs",
        REPO_ROOT / "python" / "batss" / "benchmarking.py",
        SCRIPT_DIR / "generate_gallery_figures.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _extension_info() -> dict[str, Any]:
    module_spec = importlib.util.find_spec("batss.batss_rust.batss_rust")
    origin = Path(module_spec.origin).resolve() if module_spec is not None and module_spec.origin else None
    if origin is None or not origin.is_file():
        return {"path": None, "sha256": None, "mtime_ns": None, "older_than_rust_source": None}
    rust_source = REPO_ROOT / "src" / "simulator_crn.rs"
    mtime_ns = origin.stat().st_mtime_ns
    return {
        "path": str(origin),
        "sha256": _sha256_file(origin),
        "mtime_ns": mtime_ns,
        "older_than_rust_source": mtime_ns < rust_source.stat().st_mtime_ns,
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_name(state: Any) -> str:
    name = getattr(state, "name", None)
    return str(name if name is not None else state)


def _state_snapshot(sim: Any) -> tuple[list[str], list[int], list[str], list[int]]:
    full_counts = np.asarray(sim.simulator.config, dtype="<u8")
    full_names = [_state_name(state) for state in sim.state_list]
    visible_indices = sim._visible_indices
    if visible_indices is None:
        visible_indices = list(range(len(full_counts)))
    visible_names = [full_names[index] for index in visible_indices]
    visible_counts = [int(full_counts[index]) for index in visible_indices]
    return visible_names, visible_counts, full_names, [int(value) for value in full_counts]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _case_payload(
    case: RunCase,
    source_fingerprint: str,
    extension_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "extension_sha256": extension_sha256,
        "spec": case.slug,
        "spec_name": case.spec.name,
        "heuristic": case.heuristic,
        "selector": case.selector,
        "population_size": case.population_size,
        "seed": case.seed,
        "threshold": case.threshold,
        "end_time": case.end_time,
        "history_samples": case.history_samples,
        "max_run_seconds": case.max_run_seconds,
    }


def _case_id(case_payload: dict[str, Any]) -> str:
    return _json_hash(case_payload)[:24]


def _run_id(case_id: str, repeat: int) -> str:
    return _json_hash({"case_id": case_id, "repeat": repeat})[:24]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines):
                print(f"warning: ignoring incomplete final JSONL line {index} in {path}", file=sys.stderr)
                break
            raise RuntimeError(f"invalid JSON on line {index} of {path}")
        if isinstance(record, dict):
            records.append(record)
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    last_error: OSError | None = None
    for attempt in range(5):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            return
        except OSError as error:
            last_error = error
            time.sleep(0.2 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _build_cases(args: argparse.Namespace, specs: dict[str, CRNSpec]) -> list[RunCase]:
    preset = PRESETS[args.preset]
    selected_specs: Sequence[str] = args.specs or SPEC_ORDER
    thresholds: Sequence[float] = args.thresholds or preset["thresholds"]
    seeds: Sequence[int] = args.seeds or preset["seeds"]
    repeats = args.repeats if args.repeats is not None else preset["repeats"]
    end_time_scale = args.end_time_scale if args.end_time_scale is not None else preset["end_time_scale"]
    max_run_seconds = args.max_run_seconds if args.max_run_seconds is not None else preset["max_run_seconds"]

    if repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.history_samples <= 0:
        raise ValueError("--history-samples must be positive")
    if not math.isfinite(end_time_scale) or end_time_scale <= 0.0:
        raise ValueError("--end-time-scale must be finite and positive")
    if not math.isfinite(max_run_seconds) or max_run_seconds <= 0.0:
        raise ValueError("--max-run-seconds must be finite and positive")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")

    selector = HEURISTIC_SELECTORS[args.heuristic]
    cases: list[RunCase] = []
    for slug in selected_specs:
        spec = specs[slug]
        sizes: Iterable[int] = args.sizes or preset["sizes"][slug]
        for population_size in sizes:
            for seed in seeds:
                for threshold in thresholds:
                    for repeat in range(repeats):
                        cases.append(
                            RunCase(
                                spec=spec,
                                slug=slug,
                                heuristic=args.heuristic,
                                selector=selector,
                                population_size=population_size,
                                seed=seed,
                                threshold=threshold,
                                repeat=repeat,
                                end_time=spec.benchmark_end_time * end_time_scale,
                                history_samples=args.history_samples,
                                max_run_seconds=max_run_seconds,
                            )
                        )
    return cases


def _verify_heuristic_available(case: RunCase) -> None:
    if case.heuristic != "prospective":
        return
    probe = _batss_sim(case.spec, min(case.population_size, 1_000), case.seed)
    if not hasattr(probe.simulator, "prospective_batch_score"):
        raise RuntimeError(
            "the loaded Rust extension does not expose prospective_batch_score(); "
            "rebuild it with `maturin develop --release` before using selector 2"
        )


def _warm_up(case: RunCase) -> None:
    sim = _batss_sim(case.spec, min(case.population_size, 1_000), case.seed)
    sim.simulator.heuristic_gillespie_switching = case.selector
    sim.simulator.proxy_threshold = case.threshold
    warm_end = min(case.end_time, 1e-4)
    sim.simulator.run(warm_end, min(case.max_run_seconds, 1.0))


def _run_case(case: RunCase) -> dict[str, Any]:
    total_start = time.perf_counter()
    sim = _batss_sim(case.spec, case.population_size, case.seed)
    sim.simulator.heuristic_gillespie_switching = case.selector
    sim.simulator.proxy_threshold = case.threshold
    if int(sim.simulator.heuristic_gillespie_switching) != case.selector:
        raise RuntimeError(f"simulator rejected heuristic selector {case.selector}")
    if not math.isclose(float(sim.simulator.proxy_threshold), case.threshold, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(f"simulator rejected proxy threshold {case.threshold}")
    setup_seconds = time.perf_counter() - total_start

    visible_names, visible_counts, full_names, full_counts = _state_snapshot(sim)
    trajectory_times = [float(sim.simulator.continuous_time).hex()]
    trajectory_counts = [visible_counts]

    run_start = time.perf_counter()
    deadline = run_start + case.max_run_seconds
    for sample_index in range(1, case.history_samples + 1):
        target = case.end_time * sample_index / case.history_samples
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            break
        sim.simulator.run(target, remaining)
        actual_time = float(sim.simulator.continuous_time)
        _, visible_counts, _, _ = _state_snapshot(sim)
        trajectory_times.append(actual_time.hex())
        trajectory_counts.append(visible_counts)
        tolerance = max(1e-15, abs(target) * 1e-12)
        if actual_time < target - tolerance:
            break
    run_seconds = time.perf_counter() - run_start
    total_seconds = time.perf_counter() - total_start

    visible_names, visible_counts, full_names, full_counts = _state_snapshot(sim)
    final_time = float(sim.simulator.continuous_time)
    tolerance = max(1e-15, abs(case.end_time) * 1e-12)
    reached_end = final_time >= case.end_time - tolerance
    switch: Any = sim.simulator.switch

    batch_seconds = float(switch.batch_wallclock_seconds)
    gillespie_seconds = float(switch.gillespie_wallclock_seconds)
    switch_seconds = float(switch.switch_overhead_seconds)
    batch_time = float(switch.batch_continuous_time)
    gillespie_time = float(switch.gillespie_continuous_time)
    simulated_time = batch_time + gillespie_time
    engine_seconds = batch_seconds + gillespie_seconds + switch_seconds
    mode_signature = {
        "batch_calls": int(switch.batch_calls),
        "gillespie_calls": int(switch.gillespie_calls),
        "mode_switches": int(switch.mode_switches),
        "final_do_gillespie": bool(sim.simulator.do_gillespie),
    }

    return {
        "status": "ok" if reached_end else "timeout",
        "reached_end_time": reached_end,
        "final_continuous_time": final_time,
        "setup_wallclock_seconds": setup_seconds,
        "run_wallclock_seconds": run_seconds,
        "total_wallclock_seconds": total_seconds,
        "batch_wallclock_seconds": batch_seconds,
        "gillespie_wallclock_seconds": gillespie_seconds,
        "switch_overhead_seconds": switch_seconds,
        "unaccounted_run_wallclock_seconds": run_seconds - engine_seconds,
        "batch_continuous_time": batch_time,
        "gillespie_continuous_time": gillespie_time,
        "batch_continuous_fraction": _ratio(batch_time, simulated_time),
        "gillespie_continuous_fraction": _ratio(gillespie_time, simulated_time),
        "batch_wallclock_per_continuous_time": _ratio(batch_seconds, batch_time),
        "gillespie_wallclock_per_continuous_time": _ratio(gillespie_seconds, gillespie_time),
        **mode_signature,
        "mode_signature_hash": _json_hash(mode_signature),
        "final_config": list(zip(visible_names, visible_counts)),
        "final_config_hash": _json_hash({"species": visible_names, "counts": visible_counts}),
        "full_state_hash": _json_hash({"states": full_names, "counts": full_counts}),
        "trajectory_hash": _json_hash(
            {"species": visible_names, "times_hex": trajectory_times, "counts": trajectory_counts}
        ),
        "trajectory_points": len(trajectory_times),
    }


def _reference_key(record: dict[str, Any]) -> tuple[str, int] | None:
    case_id = record.get("case_id")
    repeat = record.get("repeat")
    if not isinstance(case_id, str) or not isinstance(repeat, int):
        return None
    return case_id, repeat


def _replay_comparison(result: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    if reference is None:
        return {
            "replay_reference_run_id": None,
            "replay_hashes_match_repeat_zero": None,
            "replay_mode_signature_matches_repeat_zero": None,
            "observed_exact_replay": None,
        }
    hashes_match = all(
        result[field] == reference.get(field)
        for field in ("trajectory_hash", "final_config_hash", "full_state_hash")
    )
    mode_match = result["mode_signature_hash"] == reference.get("mode_signature_hash")
    return {
        "replay_reference_run_id": reference.get("run_id"),
        "replay_hashes_match_repeat_zero": hashes_match,
        "replay_mode_signature_matches_repeat_zero": mode_match,
        "observed_exact_replay": hashes_match and mode_match,
    }


def _print_matrix(args: argparse.Namespace, cases: Sequence[RunCase], pending: int) -> None:
    print(f"preset={args.preset} heuristic={args.heuristic} output={args.output.resolve()}")
    print(f"planned runs={len(cases)} pending runs={pending}")
    for slug in SPEC_ORDER:
        spec_cases = [case for case in cases if case.slug == slug]
        if not spec_cases:
            continue
        sizes = sorted({case.population_size for case in spec_cases})
        thresholds = sorted({case.threshold for case in spec_cases})
        seeds = sorted({case.seed for case in spec_cases})
        repeats = max(case.repeat for case in spec_cases) + 1
        print(
            f"  {slug}: n={sizes}, thresholds={thresholds}, seeds={seeds}, "
            f"repeats={repeats}, end_time={spec_cases[0].end_time:g}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    specs = _authoritative_specs()
    cases = _build_cases(args, specs)
    source_fingerprint = _source_fingerprint()
    extension_info = _extension_info()

    output = args.output.resolve()
    existing_records = [] if args.overwrite else _read_jsonl(output)
    completed_run_ids = {
        record["run_id"]
        for record in existing_records
        if record.get("record_type") == "run"
        and record.get("status") in {"ok", "timeout"}
        and isinstance(record.get("run_id"), str)
    }

    identified_cases: list[tuple[RunCase, dict[str, Any], str, str]] = []
    for case in cases:
        payload = _case_payload(case, source_fingerprint, extension_info["sha256"])
        case_id = _case_id(payload)
        identified_cases.append((case, payload, case_id, _run_id(case_id, case.repeat)))
    pending_cases = [item for item in identified_cases if item[3] not in completed_run_ids]
    if args.max_runs is not None:
        pending_cases = pending_cases[: args.max_runs]

    _print_matrix(args, cases, len(pending_cases))
    if args.dry_run:
        return 0

    if args.overwrite and output.exists():
        output.unlink()
        existing_records = []

    if not pending_cases:
        print("nothing to do; all selected run IDs are already checkpointed")
        return 0

    _verify_heuristic_available(pending_cases[0][0])
    if extension_info["older_than_rust_source"]:
        raise RuntimeError(
            f"loaded extension {extension_info['path']} is older than src/simulator_crn.rs; "
            "rebuild with `maturin develop --release` before benchmarking"
        )

    invocation_record = {
        "record_type": "invocation",
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "preset": args.preset,
        "heuristic": args.heuristic,
        "planned_runs": len(cases),
        "pending_runs": len(pending_cases),
        "source_fingerprint": source_fingerprint,
        "extension": extension_info,
        "git_commit": _git_commit(),
        "batss_version": getattr(batss, "__version__", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    _append_jsonl(output, invocation_record)

    if not args.no_warmup:
        print("warming up (untimed)...", flush=True)
        _warm_up(pending_cases[0][0])

    # References include resumed repeat-zero rows and are updated as new repeat-zero rows finish.
    references: dict[str, dict[str, Any]] = {}
    for record in existing_records:
        key = _reference_key(record)
        if key is not None and key[1] == 0 and record.get("status") == "ok":
            references[key[0]] = record

    max_total_seconds = (
        args.max_total_seconds
        if args.max_total_seconds is not None
        else PRESETS[args.preset]["max_total_seconds"]
    )
    if max_total_seconds is not None and max_total_seconds <= 0.0:
        max_total_seconds = None
    invocation_start = time.perf_counter()
    failures = 0

    for position, (case, payload, case_id, run_id) in enumerate(pending_cases, start=1):
        if max_total_seconds is not None and time.perf_counter() - invocation_start >= max_total_seconds:
            print(f"stopping before run {position}: {max_total_seconds:g}s total budget reached")
            break

        label = (
            f"[{position}/{len(pending_cases)}] {case.slug} n={case.population_size:g} "
            f"seed={case.seed} threshold={case.threshold:g} repeat={case.repeat}"
        )
        print(label, flush=True)
        base_record = {
            "record_type": "run",
            **payload,
            "case_id": case_id,
            "run_id": run_id,
            "repeat": case.repeat,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            result = _run_case(case)
            reference = references.get(case_id) if case.repeat != 0 else None
            record = {**base_record, **result, **_replay_comparison(result, reference)}
            if case.repeat == 0 and result["status"] == "ok":
                references[case_id] = record
            replay = record["observed_exact_replay"]
            replay_label = "" if replay is None else f" replay={'match' if replay else 'MISMATCH'}"
            print(
                f"  {result['status']} {result['run_wallclock_seconds']:.4g}s "
                f"G-dt={result['gillespie_continuous_fraction']} "
                f"hash={result['trajectory_hash'][:12]}{replay_label}",
                flush=True,
            )
        except Exception as error:
            failures += 1
            record = {
                **base_record,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(f"  ERROR {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        _append_jsonl(output, record)

    print(f"results: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
