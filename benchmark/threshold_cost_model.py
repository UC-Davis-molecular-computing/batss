"""Fit the batch and Gillespie engine costs separately, then take their ratio as the threshold.

This implements the "Revised research hypothesis" of ``GILLESPIE_SWITCH_LOGIC.md``: earlier work
regressed the break-even threshold ``T*`` directly on CRN features and found the structural
coefficients unstable, because ``T* = C_batch / C_Gillespie`` mixes two different cost models into
one signed regression where ``R`` and ``B`` are nearly collinear.  Here the two costs are fit
independently against their own measured targets with **nonnegative** coefficients -- every term is
a real amount of work, so no coefficient may be negative -- and the ratio is formed only afterwards.

Inputs are the paired engine timings written by ``profile_batch_costs.py timings
--measure-gillespie-all``, which measures both engines for every profile case.  The older paired
CSVs only measured Gillespie for the two Shrinking families, which cannot identify the denominator's
``q`` and ``o`` dependence.

Targets:

* ``batch_steady_us_per_call``        -- microseconds for one batch call (engine + postprocess)
* ``gillespie_steady_us_per_reaction``-- microseconds per exact reaction (engine + amortized sync)
* ``threshold_T_star``               -- their ratio, the measured break-even score

Run with::

    python benchmark/threshold_cost_model.py --timings benchmark/batch_cost_profile_timings_allg.csv
    python benchmark/threshold_cost_model.py --timings ...allg.csv --timings ...allg_seed2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------------
# Nonnegative least squares
# --------------------------------------------------------------------------------------------

def nnls(design: np.ndarray, target: np.ndarray, *, max_iter: int = 500) -> np.ndarray:
    """Solve ``min ||design @ w - target||`` subject to ``w >= 0``.

    Uses scipy when available and otherwise falls back to a projected-gradient solver so the
    harness has no hard scipy dependency.
    """

    try:
        from scipy.optimize import nnls as scipy_nnls
    except ImportError:
        pass
    else:
        return scipy_nnls(design, target, maxiter=max_iter * design.shape[1])[0]

    # Projected gradient descent with a Lipschitz step size.
    gram = design.T @ design
    rhs = design.T @ target
    step = 1.0 / (np.linalg.norm(gram, 2) + 1e-12)
    weights = np.zeros(design.shape[1])
    for _ in range(max_iter * 200):
        gradient = gram @ weights - rhs
        updated = np.maximum(weights - step * gradient, 0.0)
        if np.allclose(updated, weights, rtol=1e-12, atol=1e-14):
            break
        weights = updated
    return weights


# --------------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------------

def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    return float(value)


def batch_features(row: dict[str, str]) -> dict[str, float]:
    """Nonnegative work terms for one batch call.

    ``C_batch = C_collision(N, o, g) + a_sample sum_d q^d + a_scan q^o + a_branch W_lanes
                + C_finish(q)``
    """

    n = _f(row, "prospective_N")
    q = _f(row, "q")
    o = _f(row, "o")
    g = _f(row, "g")
    log2_n = math.log2(n) if n > 0 else 0.0
    score = _f(row, "score")
    expected_length = _f(row, "expected_batch_length")
    branches = _f(row, "output_branches_B")

    # Recursive dense sampling visits 1 + q + ... + q^(o-1) nodes doing ~q work each, so the node
    # work is sum_{d=1}^{o} q^d; the terminal scan over q^o lanes is counted separately.
    recursive_nodes = sum(q**d for d in range(1, int(o) + 1))

    # Expected occupied ordered lanes.  With E[L] draws spread over q^o lanes, occupancy saturates:
    # a batch cannot occupy more lanes than it has draws.  This is the doc's W_lanes stand-in that
    # static R cannot provide, and it is computable prospectively.
    lanes = q**o
    occupied_lanes = lanes * (1.0 - math.exp(-expected_length / lanes)) if lanes > 0 else 0.0

    return {
        "const": 1.0,
        "log2_N": log2_n,
        "o": o,
        "o_log2_N": o * log2_n,
        "g_pos": 1.0 if g >= 1 else 0.0,
        "g": g,
        "g_log2_N": g * log2_n,
        "recursive_nodes": recursive_nodes,
        "q_power_o": lanes,
        "occupied_lanes": occupied_lanes,
        "active_reactions": max(score, 0.0),
        "branch_alternatives": branches,
        "q": q,
    }


def gillespie_features(row: dict[str, str], *, horizon: float) -> dict[str, float]:
    """Nonnegative work terms for one exact Gillespie reaction in rebop's dense engine.

    ``C_G = c0 + c_rate B (q - 2) + c_order sum_channels(order) + c_select B + c_jump (q - 2)
            + C_sync(q) / H``
    """

    q = _f(row, "q")
    o = _f(row, "o")
    branches = _f(row, "output_branches_B")
    real_species = max(q - 2.0, 0.0)  # q counts the filler species K and W

    return {
        "const": 1.0,
        "B": branches,
        "B_real_species": branches * real_species,
        "channel_order": branches * o,  # padded order bounds the real reactant order per channel
        "real_species": real_species,
        "sync_per_reaction": q / horizon,
    }


BATCH_FEATURES = list(batch_features({"prospective_N": "1", "q": "1", "o": "1", "g": "0",
                                      "score": "0", "expected_batch_length": "0",
                                      "output_branches_B": "0"}))
GILLESPIE_FEATURES = list(gillespie_features({"q": "1", "o": "1", "output_branches_B": "0"},
                                             horizon=1.0))


# --------------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------------

def load_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    """Read one or more timing CSVs, averaging repeated cases across seed passes."""

    by_case: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                slug = row["case"]
                if slug not in by_case:
                    by_case[slug] = []
                    order.append(slug)
                by_case[slug].append(row)

    merged: list[dict[str, str]] = []
    numeric = ("batch_steady_us_per_call", "gillespie_steady_us_per_reaction", "threshold_T_star",
               "batch_engine_us_per_call", "batch_postprocess_us_per_call")
    for slug in order:
        group = by_case[slug]
        row = dict(group[0])
        for key in numeric:
            values = [float(item[key]) for item in group if item.get(key, "") not in ("", None)]
            row[key] = str(sum(values) / len(values)) if values else ""
        row["seed_passes"] = str(len(group))
        merged.append(row)
    return merged


def load_trajectory_states(paths: Sequence[Path]) -> list[dict[str, str]]:
    """Adapt `threshold_model.py`'s frozen-state CSVs into the row schema used here.

    This exists because of a sampling gap. `profile_batch_costs.py`, which produced the training
    data for the original fit, times every case from its CRN's **initial** configuration. A running
    policy never sees those states after step one -- it spends the whole trajectory at intermediate
    configurations whose composition, active fraction, and K have all drifted. `threshold_model.py`
    samples exactly those: `capture_frozen_states` walks a real trajectory and freezes states along
    it, then times both engines there with the same `benchmark_engine_call` oracle.

    Fitting the cost model on this distribution instead tests whether the empirical `alpha` needed
    to make the model win end-to-end is a real physical constant or an artifact of having trained on
    initial configurations only.

    Column names and units differ (seconds rather than microseconds), so they are translated here
    rather than by changing either harness.
    """

    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("threshold_T_star"):
                    continue
                rows.append({
                    # One fit row per frozen state; family is the CRN, so leave-one-family-out and
                    # the per-family reporting still group the way they do for the profile matrix.
                    "case": f"{row['crn']}_s{row.get('state_id', '?')}",
                    "family": row["crn"],
                    "trajectory_time": row.get("trajectory_time", ""),
                    "prospective_N": row["prospective_n"],
                    "q": row["q"], "o": row["o"], "g": row["g"],
                    "score": row["score"],
                    "expected_batch_length": row["expected_batch_length"],
                    "output_branches_B": row["output_branches_B"],
                    "batch_steady_us_per_call": str(float(row["batch_steady_seconds"]) * 1e6),
                    "gillespie_steady_us_per_reaction":
                        str(float(row["gillespie_seconds_per_reaction"]) * 1e6),
                    "threshold_T_star": row["threshold_T_star"],
                })
    return rows


def usable(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    keep = []
    for row in rows:
        if row.get("gillespie_steady_us_per_reaction", "") in ("", None):
            continue
        if not math.isfinite(_f(row, "threshold_T_star")):
            continue
        keep.append(row)
    return keep


# --------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------

def _design(rows: Sequence[dict[str, str]], kind: str, *, horizon: float) -> np.ndarray:
    if kind == "batch":
        return np.array([[batch_features(row)[name] for name in BATCH_FEATURES] for row in rows])
    return np.array([[gillespie_features(row, horizon=horizon)[name]
                      for name in GILLESPIE_FEATURES] for row in rows])


def _relative_fit(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    ratio = prediction / target
    return {
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratio)))),
        "max_over_prediction": float(np.max(ratio)),
        "max_under_prediction": float(np.min(ratio)),
        "median_abs_relative_error": float(np.median(np.abs(ratio - 1.0))),
        "r2": float(1.0 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2)),
    }


def fit_cost(rows: Sequence[dict[str, str]], kind: str, *, horizon: float,
             weighted: bool = True) -> dict[str, Any]:
    """Fit one engine cost with nonnegative coefficients.

    Costs span an order of magnitude, so the fit minimizes *relative* error by scaling each row by
    ``1 / target``.  An unweighted absolute fit would let the most expensive cases dominate.
    """

    target_key = ("batch_steady_us_per_call" if kind == "batch"
                  else "gillespie_steady_us_per_reaction")
    design = _design(rows, kind, horizon=horizon)
    target = np.array([_f(row, target_key) for row in rows])

    if weighted:
        scale = 1.0 / target
        weights = nnls(design * scale[:, None], target * scale)
    else:
        weights = nnls(design, target)

    prediction = design @ weights
    names = BATCH_FEATURES if kind == "batch" else GILLESPIE_FEATURES
    return {
        "kind": kind,
        "coefficients": {name: float(value) for name, value in zip(names, weights)},
        "active_terms": [name for name, value in zip(names, weights) if value > 0],
        "fit": _relative_fit(target, prediction),
        "weights": weights,
        "prediction": prediction,
        "target": target,
    }


# --------------------------------------------------------------------------------------------
# Decision regret
# --------------------------------------------------------------------------------------------

def regret(score: float, t_star: float, predicted_threshold: float) -> float:
    """Cost ratio of the engine a policy picks versus the engine that is actually cheaper.

    Batch costs ``C_batch / score`` per real reaction and Gillespie costs ``C_G``, so their ratio is
    ``t_star / score``.  Batch is optimal exactly when ``score > t_star``.
    """

    if not (math.isfinite(score) and math.isfinite(t_star)) or score <= 0 or t_star <= 0:
        return math.nan
    chooses_batch = score >= predicted_threshold
    return max(1.0, t_star / score) if chooses_batch else max(1.0, score / t_star)


def evaluate_policy(rows: Sequence[dict[str, str]], thresholds: np.ndarray) -> dict[str, Any]:
    regrets = np.array([regret(_f(row, "score"), _f(row, "threshold_T_star"), threshold)
                        for row, threshold in zip(rows, thresholds)])
    finite = regrets[np.isfinite(regrets)]
    misclassified = [
        {"case": row["case"], "family": row["family"], "score": _f(row, "score"),
         "T_star": _f(row, "threshold_T_star"), "T_predicted": float(threshold),
         "regret": float(value)}
        for row, threshold, value in zip(rows, thresholds, regrets) if value > 1.0 + 1e-9
    ]
    return {
        "mean_regret": float(np.mean(finite)),
        "geometric_mean_regret": float(np.exp(np.mean(np.log(finite)))),
        "worst_regret": float(np.max(finite)),
        "misclassified_states": len(misclassified),
        "total_states": int(len(finite)),
        "worst_cases": sorted(misclassified, key=lambda item: -item["regret"])[:6],
    }


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------

def predict_threshold(rows: Sequence[dict[str, str]], batch_fit: dict[str, Any],
                      gillespie_fit: dict[str, Any], *, horizon: float) -> np.ndarray:
    batch = _design(rows, "batch", horizon=horizon) @ batch_fit["weights"]
    gillespie = _design(rows, "gillespie", horizon=horizon) @ gillespie_fit["weights"]
    return batch / gillespie


def leave_one_family_out(rows: Sequence[dict[str, str]], *, horizon: float) -> dict[str, Any]:
    families = sorted({row["family"] for row in rows})
    folds = {}
    all_predictions = np.zeros(len(rows))
    for family in families:
        train = [row for row in rows if row["family"] != family]
        test_index = [i for i, row in enumerate(rows) if row["family"] == family]
        test = [rows[i] for i in test_index]
        batch_fit = fit_cost(train, "batch", horizon=horizon)
        gillespie_fit = fit_cost(train, "gillespie", horizon=horizon)
        predicted = predict_threshold(test, batch_fit, gillespie_fit, horizon=horizon)
        all_predictions[test_index] = predicted
        folds[family] = {
            "held_out_states": len(test),
            "policy": evaluate_policy(test, predicted),
            "threshold_range": [float(predicted.min()), float(predicted.max())],
            "batch_active_terms": batch_fit["active_terms"],
            "gillespie_active_terms": gillespie_fit["active_terms"],
        }
    return {"folds": folds, "pooled": evaluate_policy(rows, all_predictions),
            "predictions": all_predictions}


def best_constant(rows: Sequence[dict[str, str]]) -> float:
    """The single fixed threshold with the lowest mean regret on these states.

    This is the fair constant baseline: it is *selected on the training set* and then applied
    unchanged to held-out states, exactly like the fitted model.
    """

    scores = np.array([_f(row, "score") for row in rows])
    stars = np.array([_f(row, "threshold_T_star") for row in rows])
    grid = np.unique(np.concatenate([stars, scores, np.geomspace(50, 5000, 400)]))
    best, best_cost = 500.0, math.inf
    for candidate in grid:
        cost = float(np.mean([regret(s, t, candidate) for s, t in zip(scores, stars)]))
        if cost < best_cost - 1e-12:
            best, best_cost = float(candidate), cost
    return best


def report_heldout(train: Sequence[dict[str, str]], test: Sequence[dict[str, str]], *,
                   horizon: float) -> dict[str, Any]:
    """Fit on ``train`` only, then score every predictor on the untouched ``test`` states."""

    batch_fit = fit_cost(train, "batch", horizon=horizon)
    gillespie_fit = fit_cost(train, "gillespie", horizon=horizon)
    predicted = predict_threshold(test, batch_fit, gillespie_fit, horizon=horizon)

    stars = np.array([_f(row, "threshold_T_star") for row in test])
    scores = np.array([_f(row, "score") for row in test])
    log2_n = np.array([math.log2(_f(row, "prospective_N")) for row in test])

    proximity = scores / stars
    in_band = int(np.sum((proximity >= 0.5) & (proximity <= 2.0)))
    print(f"held-out states: {len(test)}   within 2x of break-even: {in_band}/{len(test)}"
          f"   prefer batch: {int(np.sum(scores > stars))}")
    print(f"   score/T* spans {proximity.min():.3g} .. {proximity.max():.3g}")
    print(f"   T* spans {stars.min():.1f} .. {stars.max():.1f}\n")

    train_log2_n = np.array([math.log2(_f(row, "prospective_N")) for row in train])
    train_stars = np.array([_f(row, "threshold_T_star") for row in train])
    design = np.column_stack([np.ones(len(train)), train_log2_n])
    logn_coef = np.linalg.lstsq(design, train_stars, rcond=None)[0]
    selected_constant = best_constant(train)

    candidates = {
        "separate-cost ratio (fit on train)": predicted,
        f"constant selected on train (T={selected_constant:.0f})":
            np.full(len(test), selected_constant),
        "constant T=500": np.full(len(test), 500.0),
        f"T_logN fit on train ({logn_coef[0]:.0f}+{logn_coef[1]:.2f} log2N)":
            design_apply(logn_coef, log2_n),
        "oracle T* (unattainable)": stars,
    }

    print(f"{'predictor':<46} {'mean':>8} {'worst':>8} {'bad':>5} {'medRelEr':>9}")
    results: dict[str, Any] = {}
    for label, thresholds in candidates.items():
        policy = evaluate_policy(test, thresholds)
        rel = float(np.median(np.abs(np.asarray(thresholds) / stars - 1.0)))
        results[label] = {**policy, "median_relative_threshold_error": rel}
        print(f"{label:<46} {policy['mean_regret']:>8.4f} {policy['worst_regret']:>8.4f} "
              f"{policy['misclassified_states']:>5} {rel:>9.4f}")
    return {"results": results, "predicted": predicted, "in_band": in_band,
            "selected_constant": selected_constant}


def design_apply(coefficients: np.ndarray, log2_n: np.ndarray) -> np.ndarray:
    return coefficients[0] + coefficients[1] * log2_n


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, action="append", required=True,
                        help="paired timing CSV; repeat the flag to average seed passes")
    parser.add_argument("--evaluate-timings", type=Path, action="append",
                        help="held-out paired timing CSV(s) used only for evaluation, never for "
                             "fitting; intended for the near-boundary decision test set")
    parser.add_argument("--trajectory-states", type=Path, action="append",
                        help="fit on states sampled ALONG a trajectory instead of initial "
                             "configurations: pass threshold_model.py's *_states.csv. These are "
                             "the states a running policy actually meets, so a fit here tests "
                             "whether the empirical scale factor is physical or a sampling artifact")
    parser.add_argument("--evaluate-trajectory-states", type=Path, action="append",
                        help="held-out trajectory-state CSV(s), same format as --trajectory-states")
    parser.add_argument("--horizon", type=float, default=5000.0,
                        help="Gillespie reactions the sync cost is amortized over (the block size)")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    rows = usable(load_rows(args.timings)) if args.timings else []
    if args.trajectory_states:
        trajectory = usable(load_trajectory_states(args.trajectory_states))
        print(f"trajectory-sampled states: {len(trajectory)} "
              f"across {len(set(r['family'] for r in trajectory))} CRNs "
              f"(states met during a run, not initial configurations)")
        rows = rows + trajectory
    if not rows:
        raise SystemExit("give at least one of --timings or --trajectory-states")
    print(f"usable paired states: {len(rows)} "
          f"across {len(set(row['family'] for row in rows))} families\n")

    batch_fit = fit_cost(rows, "batch", horizon=args.horizon)
    gillespie_fit = fit_cost(rows, "gillespie", horizon=args.horizon)

    for fit in (batch_fit, gillespie_fit):
        print(f"--- {fit['kind']} cost (nonnegative fit) ---")
        for name, value in fit["coefficients"].items():
            marker = "" if value > 0 else "   (dropped)"
            print(f"    {name:<22} {value:>14.6g}{marker}")
        metrics = fit["fit"]
        print(f"    R^2={metrics['r2']:.4f}  median|rel err|={metrics['median_abs_relative_error']:.4f}  "
              f"range=[{metrics['max_under_prediction']:.3f}, {metrics['max_over_prediction']:.3f}]x")
        print()

    predicted = predict_threshold(rows, batch_fit, gillespie_fit, horizon=args.horizon)
    measured = np.array([_f(row, "threshold_T_star") for row in rows])
    print("--- ratio model T_hat = C_batch / C_Gillespie (in sample) ---")
    print(f"    T_hat range   {predicted.min():.1f} .. {predicted.max():.1f}")
    print(f"    T* range      {measured.min():.1f} .. {measured.max():.1f}")
    print(f"    correlation   {np.corrcoef(predicted, measured)[0, 1]:.4f}")
    in_sample = evaluate_policy(rows, predicted)
    print(f"    regret mean={in_sample['mean_regret']:.4f} worst={in_sample['worst_regret']:.4f} "
          f"({in_sample['misclassified_states']}/{in_sample['total_states']} misclassified)\n")

    grouped = leave_one_family_out(rows, horizon=args.horizon)
    print("--- leave-one-family-out ---")
    for family, fold in grouped["folds"].items():
        policy = fold["policy"]
        print(f"    {family:<22} n={fold['held_out_states']:<3} "
              f"mean={policy['mean_regret']:.4f} worst={policy['worst_regret']:.4f} "
              f"bad={policy['misclassified_states']}")
    pooled = grouped["pooled"]
    print(f"    {'POOLED':<22} n={pooled['total_states']:<3} "
          f"mean={pooled['mean_regret']:.4f} worst={pooled['worst_regret']:.4f} "
          f"bad={pooled['misclassified_states']}\n")

    print("--- baselines on the same states ---")
    scores = np.array([_f(row, "score") for row in rows])
    log2_n = np.array([math.log2(_f(row, "prospective_N")) for row in rows])
    baselines = {
        "constant T=500": np.full(len(rows), 500.0),
        "constant T=250": np.full(len(rows), 250.0),
        "T_logN seed1 (460.2 + 8.35 log2 N)": 460.2 + 8.35 * log2_n,
        "T_logN seed2 (506.0 + 7.56 log2 N)": 506.0 + 7.56 * log2_n,
        "oracle T*": measured,
    }
    rows_out = {}
    for label, thresholds in baselines.items():
        policy = evaluate_policy(rows, thresholds)
        rows_out[label] = policy
        print(f"    {label:<38} mean={policy['mean_regret']:.4f} "
              f"worst={policy['worst_regret']:.4f} bad={policy['misclassified_states']}")
    print(f"    {'separate-cost ratio (grouped)':<38} mean={pooled['mean_regret']:.4f} "
          f"worst={pooled['worst_regret']:.4f} bad={pooled['misclassified_states']}")

    if pooled["worst_cases"]:
        print("\n--- worst grouped misclassifications ---")
        for item in pooled["worst_cases"]:
            print(f"    {item['case']:<36} score={item['score']:>9.1f} T*={item['T_star']:>8.1f} "
                  f"T_hat={item['T_predicted']:>8.1f} regret={item['regret']:.3f}")

    heldout = None
    test_rows: list[dict[str, str]] = []
    if args.evaluate_timings:
        test_rows += usable(load_rows(args.evaluate_timings))
    if args.evaluate_trajectory_states:
        test_rows += usable(load_trajectory_states(args.evaluate_trajectory_states))
    if test_rows:
        print("\n" + "=" * 78)
        print("HELD-OUT DECISION TEST (fit uses training rows only)")
        print("=" * 78)
        heldout = report_heldout(rows, test_rows, horizon=args.horizon)

    if args.json_out:
        payload = {
            "states": len(rows),
            "horizon": args.horizon,
            "batch_cost": {k: v for k, v in batch_fit.items()
                           if k not in ("weights", "prediction", "target")},
            "gillespie_cost": {k: v for k, v in gillespie_fit.items()
                               if k not in ("weights", "prediction", "target")},
            "in_sample_policy": in_sample,
            "leave_one_family_out": {"folds": grouped["folds"], "pooled": grouped["pooled"]},
            "baselines": rows_out,
            "per_state": [
                {"case": row["case"], "family": row["family"], "score": _f(row, "score"),
                 "T_star": _f(row, "threshold_T_star"), "T_hat_in_sample": float(a),
                 "T_hat_grouped": float(b)}
                for row, a, b in zip(rows, predicted, grouped["predictions"])
            ],
        }
        if heldout is not None:
            payload["held_out_near_boundary"] = {
                "states_within_2x_of_break_even": heldout["in_band"],
                "selected_constant": heldout["selected_constant"],
                "results": heldout["results"],
            }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
