#!/usr/bin/env python3
"""Clean-room V5 recomputation and preregistered negative controls.

This module intentionally does not import any V5 package module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import time
from typing import Any

import numpy as np


ROWS_PER_EPISODE = 38
EPISODES = 1600
ROWS = ROWS_PER_EPISODE * EPISODES
BASE_FLOPS = 70_529_190
V1_FLOPS = 669_184
ADAPTER_FLOPS = 264_960
FEATURE_FLOPS = 3_801
SCORE_FLOPS = 4_184
GATE_FLOPS = FEATURE_FLOPS + SCORE_FLOPS
GATE_NONFLOPS = 5
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_831_999_991
HISTOGRAM_SEED = 2_831_888_881
SEEDED_MIXTURE_SEED = 2_831_777_771


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = average_ranks(left)
    y = average_ranks(right)
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / denominator)


def calls_from_scores(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    result = np.ones(len(scores), dtype=np.int64)
    active = np.arange(len(scores), dtype=np.int64)
    for stage in range(3):
        local = scores[active, stage]
        if not np.isfinite(local).all():
            raise RuntimeError("active score is nonfinite")
        active = active[local > thresholds[stage]]
        result[active] += 1
    return result


def per_episode_mean(values: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty(EPISODES, dtype=np.float64)
    for episode in range(EPISODES):
        selected = values[episode_id == episode]
        if len(selected) != ROWS_PER_EPISODE:
            raise RuntimeError(f"episode {episode} does not contain 38 rows")
        output[episode] = math.fsum(map(float, selected)) / ROWS_PER_EPISODE
    return output


def best_mixture(loss: np.ndarray, mean_depth: float) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for lower in range(1, 5):
        for upper in range(lower, 5):
            if mean_depth < lower or mean_depth > upper:
                continue
            weight = 0.0 if lower == upper else (mean_depth - lower) / (upper - lower)
            vector = (1.0 - weight) * loss[:, lower - 1] + weight * loss[:, upper - 1]
            candidate = {
                "mean_loss": float(math.fsum(map(float, vector)) / len(vector)),
                "loss": vector,
                "depth_lower": lower,
                "depth_upper": upper,
                "weight_upper": float(weight),
            }
            key = (
                candidate["mean_loss"],
                candidate["depth_lower"],
                candidate["depth_upper"],
            )
            if best is None or key < (
                best["mean_loss"],
                best["depth_lower"],
                best["depth_upper"],
            ):
                best = candidate
    if best is None:
        raise RuntimeError("no mixture brackets requested mean depth")
    return best


def linear_quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def synthetic_validation() -> dict[str, Any]:
    fixture_scores = np.asarray(
        [
            [0.1, np.nan, np.nan],
            [0.6, 0.1, np.nan],
            [0.6, 0.7, 0.1],
            [0.6, 0.7, 0.8],
        ],
        dtype=np.float64,
    )
    calls = calls_from_scores(fixture_scores, np.asarray([0.5, 0.5, 0.5]))
    calls_ok = np.array_equal(calls, np.asarray([1, 2, 3, 4]))

    fixture_loss = np.asarray(
        [
            [4.0, 3.0, 2.0, 1.0],
            [8.0, 7.0, 6.0, 5.0],
        ]
    )
    mixture = best_mixture(fixture_loss, 1.25)
    expected = 0.75 * fixture_loss[:, 0] + 0.25 * fixture_loss[:, 1]
    mixture_ok = (
        mixture["depth_lower"] == 1
        and mixture["depth_upper"] == 2
        and mixture["weight_upper"] == 0.25
        and np.array_equal(mixture["loss"], expected)
    )

    ranks_ok = spearman(np.asarray([30.0, 10.0, 20.0]), np.asarray([3.0, 1.0, 2.0])) == 1.0
    tie_ranks_ok = np.array_equal(
        average_ranks(np.asarray([2.0, 1.0, 2.0, 3.0])),
        np.asarray([2.5, 1.0, 2.5, 4.0]),
    )
    quantile_fixture = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
    quantile_ok = linear_quantile(quantile_fixture, 0.25) == 10.0
    checks = {
        "sequential_calls_hand_fixture": calls_ok,
        "analytic_mixture_hand_fixture": mixture_ok,
        "average_rank_hand_fixture": tie_ranks_ok,
        "spearman_hand_fixture": ranks_ok,
        "linear_quantile_hand_fixture": quantile_ok,
    }
    return {"passed": all(checks.values()), "checks": checks}


def compare_scalar(
    observed: float, expected: float, *, atol: float, rtol: float
) -> dict[str, Any]:
    delta = float(observed) - float(expected)
    allowed = atol + rtol * abs(float(expected))
    return {
        "observed": float(observed),
        "expected": float(expected),
        "delta": delta,
        "absolute_delta": abs(delta),
        "allowed": allowed,
        "pass": abs(delta) <= allowed,
    }


def sequential_random_calls(
    rng: np.random.Generator,
    base_calls: np.ndarray,
    strata: np.ndarray | None,
) -> np.ndarray:
    """Random nested decisions preserving stage counts within each stratum."""
    result = np.ones(len(base_calls), dtype=np.int8)
    if strata is None:
        strata = np.zeros(len(base_calls), dtype=np.int64)
    for value in np.unique(strata):
        members = np.flatnonzero(strata == value)
        active = members.copy()
        for stage in range(1, 4):
            required = int(np.count_nonzero(base_calls[members] > stage))
            if required > len(active):
                raise RuntimeError("invalid nested stage count")
            if required == 0:
                active = active[:0]
                continue
            active = rng.permutation(active)[:required]
            result[active] += 1
    return result


def policy_benefit(
    raw_loss: np.ndarray, analytic_loss: np.ndarray, calls: np.ndarray
) -> float:
    selected = raw_loss[np.arange(len(calls)), calls.astype(np.int64) - 1]
    return float(np.mean(analytic_loss - selected, dtype=np.float64))


def summarize_controls(values: np.ndarray, actual: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    extreme = int(np.count_nonzero(values >= actual))
    return {
        "repetitions": len(values),
        "actual_real_allocation_benefit": actual,
        "control_mean": float(values.mean()),
        "control_std": float(values.std()),
        "control_quantiles": {
            "0.01": float(np.quantile(values, 0.01)),
            "0.05": float(np.quantile(values, 0.05)),
            "0.50": float(np.quantile(values, 0.50)),
            "0.95": float(np.quantile(values, 0.95)),
            "0.99": float(np.quantile(values, 0.99)),
        },
        "control_at_least_as_favorable_count": extreme,
        "plus_one_one_sided_p": (extreme + 1) / (len(values) + 1),
        "diagnostic_pass_p_lt_0_05": (extreme + 1) / (len(values) + 1) < 0.05,
    }


def run_negative_controls(
    *,
    raw_loss: np.ndarray,
    target: np.ndarray,
    dense: np.ndarray,
    sparse: np.ndarray,
    calls: np.ndarray,
    scores: np.ndarray,
    episode_id: np.ndarray,
    model_step: np.ndarray,
    analytic: dict[str, Any],
    repetitions: int,
    target_repetitions: int,
) -> dict[str, Any]:
    actual = float(np.mean(analytic["loss"] - np.mean((sparse - target) ** 2, axis=1)))
    controls: dict[str, Any] = {}

    within_episode = np.empty(repetitions, dtype=np.float64)
    rng = np.random.default_rng(510001)
    for index in range(repetitions):
        randomized = sequential_random_calls(rng, calls, episode_id)
        within_episode[index] = policy_benefit(raw_loss, analytic["loss"], randomized)
    controls["within_episode_stage_stratum_permutation"] = {
        "master_seed": 510001,
        "constraints": "nested reachability and every episode's exact four-depth histogram",
        **summarize_controls(within_episode, actual),
    }

    global_values = np.empty(repetitions, dtype=np.float64)
    rng = np.random.default_rng(510002)
    for index in range(repetitions):
        randomized = sequential_random_calls(rng, calls, None)
        global_values[index] = policy_benefit(raw_loss, analytic["loss"], randomized)
    controls["random_global_matched_call_policy"] = {
        "master_seed": 510002,
        "constraints": "nested reachability and exact global call histogram",
        **summarize_controls(global_values, actual),
    }

    gate_evaluations = int(np.minimum(calls, 3).sum())
    solver_calls = int(calls.sum())
    fractional_calls = solver_calls + gate_evaluations * GATE_FLOPS / ADAPTER_FLOPS
    integer_target = math.ceil(fractional_calls)
    template = best_mixture(raw_loss, integer_target / len(calls))
    lower = int(template["depth_lower"])
    upper = int(template["depth_upper"])
    number_upper = (
        0
        if lower == upper
        else math.ceil(
            (integer_target - lower * len(calls)) / (upper - lower)
        )
    )
    mixed = np.empty(repetitions, dtype=np.float64)
    rng = np.random.default_rng(510003)
    for index in range(repetitions):
        randomized = np.full(len(calls), lower, dtype=np.int8)
        if number_upper:
            randomized[rng.permutation(len(calls))[:number_upper]] = upper
        mixed[index] = policy_benefit(raw_loss, analytic["loss"], randomized)
    controls["constant_mixed_weakly_more_compute_policy"] = {
        "master_seed": 510003,
        "constraints": {
            "depth_lower": lower,
            "depth_upper": upper,
            "number_upper": number_upper,
            "integer_total_calls": integer_target,
        },
        **summarize_controls(mixed, actual),
    }

    by_step = np.empty(repetitions, dtype=np.float64)
    rng = np.random.default_rng(510005)
    for index in range(repetitions):
        randomized = sequential_random_calls(rng, calls, model_step)
        by_step[index] = policy_benefit(raw_loss, analytic["loss"], randomized)
    controls["broken_score_gain_association_by_model_step"] = {
        "master_seed": 510005,
        "constraints": "nested reachability and exact call histogram within each model step",
        **summarize_controls(by_step, actual),
    }

    reverse_calls = np.ones(len(calls), dtype=np.int8)
    active = np.arange(len(calls))
    for stage in range(3):
        required = int(np.count_nonzero(calls > stage + 1))
        local_order = np.argsort(scores[active, stage], kind="mergesort")
        active = active[local_order[:required]]
        reverse_calls[active] += 1
    reverse_benefit = policy_benefit(raw_loss, analytic["loss"], reverse_calls)
    controls["reverse_score_ordering"] = {
        "master_seed": 510006,
        "repetitions": 1,
        "constraints": "exact global reached-stage counts and nested reachability",
        "benefit": reverse_benefit,
        "less_favorable_than_real": reverse_benefit < actual,
    }

    # Target alignment control.  For a weighted expected analytic mixture,
    # analytic loss minus adaptive loss has a fixed prediction-norm term and
    # a target dot-product term.  This avoids recomputing five complete MSE
    # tensors per permutation.
    lower = int(analytic["depth_lower"]) - 1
    upper = int(analytic["depth_upper"]) - 1
    weight = float(analytic["weight_upper"])
    weighted_prediction = (1.0 - weight) * dense[:, lower] + weight * dense[:, upper]
    weighted_sqnorm = (
        (1.0 - weight) * np.sum(dense[:, lower] ** 2, axis=1, dtype=np.float64)
        + weight * np.sum(dense[:, upper] ** 2, axis=1, dtype=np.float64)
    )
    sparse_sqnorm = np.sum(sparse**2, axis=1, dtype=np.float64)
    fixed_term = float(np.mean((weighted_sqnorm - sparse_sqnorm) / target.shape[1]))
    prediction_delta = (weighted_prediction - sparse).reshape(
        EPISODES, ROWS_PER_EPISODE, target.shape[1]
    )
    target_episode = target.reshape(EPISODES, ROWS_PER_EPISODE, target.shape[1])
    target_values = np.empty(target_repetitions, dtype=np.float64)
    rng = np.random.default_rng(510004)
    for index in range(target_repetitions):
        order = rng.permutation(EPISODES)
        dot = np.einsum(
            "erd,erd->",
            target_episode[order],
            prediction_delta,
            optimize=True,
            dtype=np.float64,
        )
        target_values[index] = fixed_term - 2.0 * dot / (ROWS * target.shape[1])
    controls["target_episode_alignment_permutation"] = {
        "master_seed": 510004,
        "constraints": "whole 38-transition episode blocks; prediction/call/compute fixed",
        **summarize_controls(target_values, actual),
    }

    return {
        "schema_version": 1,
        "control_role": "diagnostic_falsification_not_new_confirmation",
        "actual_raw_vs_analytic_benefit": actual,
        "controls": controls,
        "all_random_families_at_least_1000": all(
            item.get("repetitions", 0) >= 1000
            for item in controls.values()
            if item.get("repetitions", 0) != 1
        ),
        "all_random_families_diagnostic_pass": all(
            item.get("diagnostic_pass_p_lt_0_05", True)
            for item in controls.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--negative-output", required=True, type=pathlib.Path)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--target-repetitions", type=int, default=1000)
    args = parser.parse_args()
    start = time.time()
    package = args.package.resolve()

    synthetic = synthetic_validation()
    if not synthetic["passed"]:
        raise SystemExit("synthetic clean-room validation failed")

    execution_path = package / "data/v5_confirmation_execution.npz"
    with np.load(execution_path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(package / "freeze/whitening.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    with np.load(package / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        thresholds_storage = stored["thresholds"].astype(np.float64)
        thresholds_runtime = stored["thresholds"].astype(np.float32)
    with np.load(
        package / "metrics/v5_confirmation_episode_metrics.npz", allow_pickle=False
    ) as stored:
        recorded_episode_metrics = {name: stored[name].copy() for name in stored.files}
    with np.load(package / "metrics/bootstrap_replicates.npz", allow_pickle=False) as stored:
        recorded_bootstrap = {name: stored[name].copy() for name in stored.files}

    target_f32 = arrays["target"]
    dense_f32 = arrays["dense_exits"]
    sparse_f32 = arrays["sparse_selected"]
    scores_f32 = arrays["scores"]
    target = target_f32.astype(np.float64)
    dense = dense_f32.astype(np.float64)
    sparse = sparse_f32.astype(np.float64)
    scores = scores_f32.astype(np.float64)
    recorded_calls = arrays["calls"].astype(np.int64)
    episode_id = arrays["episode_id"].astype(np.int64)
    model_step = arrays["model_step"].astype(np.int64)

    shape_checks = {
        "target": target.shape == (ROWS, 192),
        "dense_exits": dense.shape == (ROWS, 4, 192),
        "sparse_selected": sparse.shape == (ROWS, 192),
        "calls": recorded_calls.shape == (ROWS,),
        "scores": scores.shape == (ROWS, 3),
        "features": arrays["features"].shape == (ROWS, 3, 1046),
        "episode_id": episode_id.shape == (ROWS,),
        "model_step": model_step.shape == (ROWS,),
    }
    episode_order_exact = np.array_equal(
        episode_id, np.repeat(np.arange(EPISODES), ROWS_PER_EPISODE)
    )
    model_step_order_exact = np.array_equal(
        model_step, np.tile(np.arange(3, 41), EPISODES)
    )
    finite_checks = {
        "target": bool(np.isfinite(target).all()),
        "dense": bool(np.isfinite(dense).all()),
        "sparse": bool(np.isfinite(sparse).all()),
        "reached_scores": all(
            bool(np.isfinite(scores[recorded_calls > stage, stage]).all())
            for stage in range(3)
        ),
        "unreached_scores_nan": all(
            bool(np.isnan(scores[recorded_calls <= stage, stage]).all())
            for stage in range(3)
        ),
        "reached_features": all(
            bool(np.isfinite(arrays["features"][recorded_calls > stage, stage]).all())
            for stage in range(3)
        ),
        "unreached_features_nan": all(
            bool(np.isnan(arrays["features"][recorded_calls <= stage, stage]).all())
            for stage in range(3)
        ),
    }

    calls_runtime = calls_from_scores(scores_f32, thresholds_runtime)
    calls_storage = calls_from_scores(scores, thresholds_storage)
    active_margins = []
    for stage in range(3):
        active = recorded_calls > stage
        active_margins.append(
            float(np.min(np.abs(scores[active, stage] - thresholds_storage[stage])))
        )

    difference = dense - target[:, None, :]
    raw_loss = np.mean(difference * difference, axis=2, dtype=np.float64)
    white_difference = np.einsum(
        "nkd,df->nkf", difference, whitening, optimize=False
    )
    white_loss = np.mean(white_difference * white_difference, axis=2, dtype=np.float64)
    adaptive_raw = np.mean((sparse - target) ** 2, axis=1, dtype=np.float64)
    adaptive_white_delta = np.einsum(
        "nd,df->nf", sparse - target, whitening, optimize=False
    )
    adaptive_white = np.mean(
        adaptive_white_delta * adaptive_white_delta, axis=1, dtype=np.float64
    )
    selected_dense = dense[np.arange(ROWS), recorded_calls - 1]
    sparse_dense_delta = np.abs(sparse - selected_dense)

    gate_evaluations = int(np.minimum(recorded_calls, 3).sum())
    refiner_calls = int(recorded_calls.sum())
    common_flops = ROWS * (BASE_FLOPS + V1_FLOPS)
    adaptive_total_flops = (
        common_flops
        + (refiner_calls - ROWS) * ADAPTER_FLOPS
        + gate_evaluations * GATE_FLOPS
    )
    analytic_total_calls = (
        refiner_calls + gate_evaluations * GATE_FLOPS / ADAPTER_FLOPS
    )
    analytic_mean_calls = analytic_total_calls / ROWS
    raw_analytic = best_mixture(raw_loss, analytic_mean_calls)
    white_analytic = best_mixture(white_loss, analytic_mean_calls)

    integer_target_calls = math.ceil(analytic_total_calls)
    seeded_shape = best_mixture(raw_loss, integer_target_calls / ROWS)
    lower = int(seeded_shape["depth_lower"])
    upper = int(seeded_shape["depth_upper"])
    number_upper = (
        0
        if lower == upper
        else math.ceil(
            (integer_target_calls - lower * ROWS) / (upper - lower)
        )
    )
    seeded_calls = np.full(ROWS, lower, dtype=np.int64)
    seeded_rng = np.random.default_rng(SEEDED_MIXTURE_SEED)
    seeded_calls[seeded_rng.permutation(ROWS)[:number_upper]] = upper
    seeded_raw = raw_loss[np.arange(ROWS), seeded_calls - 1]
    seeded_flops = (
        common_flops + (int(seeded_calls.sum()) - ROWS) * ADAPTER_FLOPS
    )

    histogram_calls = np.empty(ROWS, dtype=np.int64)
    histogram_rng = np.random.default_rng(HISTOGRAM_SEED)
    histograms_exact = True
    for episode in range(EPISODES):
        indices = np.flatnonzero(episode_id == episode)
        histogram_calls[indices] = recorded_calls[indices][
            histogram_rng.permutation(ROWS_PER_EPISODE)
        ]
        histograms_exact &= bool(
            np.array_equal(
                np.bincount(histogram_calls[indices], minlength=5),
                np.bincount(recorded_calls[indices], minlength=5),
            )
        )
    histogram_raw = raw_loss[np.arange(ROWS), histogram_calls - 1]
    histogram_white = white_loss[np.arange(ROWS), histogram_calls - 1]

    contrasts = {
        "raw_vs_analytic": per_episode_mean(
            raw_analytic["loss"] - adaptive_raw, episode_id
        ),
        "raw_vs_seeded": per_episode_mean(seeded_raw - adaptive_raw, episode_id),
        "raw_vs_fixed_d1": per_episode_mean(
            raw_loss[:, 0] - adaptive_raw, episode_id
        ),
        "raw_vs_within_episode_histogram": per_episode_mean(
            histogram_raw - adaptive_raw, episode_id
        ),
        "native_whitened_vs_analytic": per_episode_mean(
            white_analytic["loss"] - adaptive_white, episode_id
        ),
        "native_whitened_vs_within_episode_histogram": per_episode_mean(
            histogram_white - adaptive_white, episode_id
        ),
    }
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampling = bootstrap_rng.integers(
        0,
        EPISODES,
        size=(BOOTSTRAP_REPLICATES, EPISODES),
        dtype=np.int32,
    )
    bootstrap = {
        name: np.mean(values[sampling], axis=1, dtype=np.float64)
        for name, values in contrasts.items()
    }
    intervals = {
        name: {
            "estimate": float(np.mean(values, dtype=np.float64)),
            "lower": linear_quantile(bootstrap[name], 0.025),
            "upper": linear_quantile(bootstrap[name], 0.975),
        }
        for name, values in contrasts.items()
    }

    stages = []
    for stage in range(3):
        reached = np.isfinite(scores[:, stage])
        raw_gain = raw_loss[:, stage] - raw_loss[:, stage + 1]
        white_gain = white_loss[:, stage] - white_loss[:, stage + 1]
        combined = 0.5 * (
            raw_gain / (raw_gain.std() + 1e-12)
            + white_gain / (white_gain.std() + 1e-12)
        )
        rho = spearman(scores[reached, stage], combined[reached])
        stages.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "spearman_score_gain_rho": rho,
                "positive_sign": bool(np.isfinite(rho) and rho > 0),
            }
        )

    claimed_analysis = json.loads((package / "analysis_result.json").read_text())
    claimed_decision = json.loads((package / "decision.json").read_text())
    claimed_compute = json.loads(
        (package / "metrics/compute_ledger_realized.json").read_text()
    )
    metric_comparisons = {}
    for name, result in intervals.items():
        raw_scale = name.startswith("raw_")
        atol = 5e-12 if raw_scale else 5e-10
        rtol = 1e-9 if raw_scale else 1e-8
        metric_comparisons[name] = {
            field: compare_scalar(
                result[field],
                claimed_analysis["criteria"][name][field],
                atol=atol,
                rtol=rtol,
            )
            for field in ("estimate", "lower", "upper")
        }
    episode_metric_comparisons = {}
    expected_episode = {
        "episode_id": np.arange(EPISODES, dtype=np.int32),
        "adaptive_raw": per_episode_mean(adaptive_raw, episode_id),
        "adaptive_native_whitened": per_episode_mean(adaptive_white, episode_id),
        **contrasts,
    }
    for name, values in expected_episode.items():
        recorded = recorded_episode_metrics[name]
        exact = np.array_equal(values, recorded)
        max_abs = (
            0.0
            if exact
            else float(np.max(np.abs(values.astype(np.float64) - recorded.astype(np.float64))))
        )
        episode_metric_comparisons[name] = {
            "exact": exact,
            "max_abs": max_abs,
            "pass": exact or max_abs <= (1e-12 if name != "adaptive_native_whitened" else 5e-10),
        }
    bootstrap_comparisons = {}
    for name, values in bootstrap.items():
        recorded = recorded_bootstrap[name]
        exact = np.array_equal(values, recorded)
        max_abs = 0.0 if exact else float(np.max(np.abs(values - recorded)))
        bootstrap_comparisons[name] = {
            "exact": exact,
            "max_abs": max_abs,
            "pass": exact or max_abs <= (5e-12 if name.startswith("raw_") else 5e-10),
        }

    stage_comparisons = []
    for observed, expected in zip(stages, claimed_analysis["stagewise_rank"]):
        stage_comparisons.append(
            {
                "stage": observed["stage"],
                "reached_exact": observed["reached_rows"] == expected["reached_rows"],
                "rho": compare_scalar(
                    observed["spearman_score_gain_rho"],
                    expected["spearman_score_gain_rho"],
                    atol=1e-10,
                    rtol=0.0,
                ),
                "sign_exact": observed["positive_sign"] == expected["positive_sign"],
            }
        )

    individual_pass = all(item["lower"] > 0 for item in intervals.values())
    simultaneous_pass = all(
        intervals[name]["lower"] > 0
        for name in ("raw_vs_analytic", "native_whitened_vs_analytic")
    )
    stagewise_pass = all(item["positive_sign"] for item in stages)
    integrity_pass = all(shape_checks.values()) and all(finite_checks.values())
    integrity_pass &= episode_order_exact and model_step_order_exact
    integrity_pass &= np.array_equal(calls_runtime, recorded_calls)
    integrity_pass &= histograms_exact and seeded_flops >= adaptive_total_flops
    terminal = (
        "v5_confirmation_passed"
        if integrity_pass and individual_pass and simultaneous_pass and stagewise_pass
        else "v5_execution_invalid"
        if not integrity_pass
        else "v5_confirmation_failed"
    )

    # Sensitivity to omitted later-adapter arithmetic in the inherited
    # historical FLOP convention.  704 is a strict lower bound from the two
    # linear biases (320), alpha multiply (192), and residual add (192), before
    # assigning any cost to GELU.
    compute_sensitivity = []
    for omitted in (0, 704, 832, 1024, 2048, 4096, 8192):
        cost = ADAPTER_FLOPS + omitted
        mean = (refiner_calls + gate_evaluations * GATE_FLOPS / cost) / ROWS
        mixture = best_mixture(raw_loss, mean)
        effect = float(np.mean(mixture["loss"] - adaptive_raw))
        compute_sensitivity.append(
            {
                "additional_ops_per_later_adapter_beyond_declared": omitted,
                "later_adapter_cost": cost,
                "analytic_mean_calls": mean,
                "raw_vs_analytic_effect": effect,
                "positive": effect > 0,
            }
        )

    loo = {}
    for name, values in contrasts.items():
        total = math.fsum(map(float, values))
        estimates = (total - values) / (len(values) - 1)
        loo[name] = {
            "minimum": float(estimates.min()),
            "maximum": float(estimates.max()),
            "all_positive": bool(np.all(estimates > 0)),
            "most_influential_episode": int(
                np.argmax(np.abs(estimates - values.mean()))
            ),
            "maximum_absolute_shift": float(
                np.max(np.abs(estimates - values.mean()))
            ),
        }
    raw_values = contrasts["raw_vs_analytic"]
    block_means = [
        float(np.mean(raw_values[start : start + 100]))
        for start in range(0, EPISODES, 100)
    ]
    accumulation = {
        "numpy_mean": float(np.mean(raw_values, dtype=np.float64)),
        "math_fsum_forward": math.fsum(map(float, raw_values)) / len(raw_values),
        "math_fsum_reverse": math.fsum(map(float, raw_values[::-1])) / len(raw_values),
        "numpy_mean_reversed": float(np.mean(raw_values[::-1], dtype=np.float64)),
    }
    quantile_sensitivity = {}
    for name, values in bootstrap.items():
        quantile_sensitivity[name] = {
            method: float(np.quantile(values, 0.025, method=method))
            for method in ("linear", "lower", "higher", "midpoint", "nearest")
        }

    result = {
        "schema_version": 1,
        "implementation": "clean-room; no imports from analysis.py, independent_verify.py, or V5 helpers",
        "created_unix_ns": time.time_ns(),
        "runtime_seconds": time.time() - start,
        "source": {
            "package": str(package),
            "execution_sha256": sha256(execution_path),
        },
        "synthetic_harness": synthetic,
        "structure": {
            "shape_checks": shape_checks,
            "episode_order_exact": episode_order_exact,
            "model_step_order_exact": model_step_order_exact,
            "finite_checks": finite_checks,
        },
        "calls": {
            "runtime_float32_threshold_calls_exact": bool(
                np.array_equal(calls_runtime, recorded_calls)
            ),
            "storage_float64_threshold_calls_exact": bool(
                np.array_equal(calls_storage, recorded_calls)
            ),
            "runtime_vs_storage_calls_exact": bool(
                np.array_equal(calls_runtime, calls_storage)
            ),
            "minimum_active_score_threshold_margin_by_stage": active_margins,
            "call_histogram": np.bincount(recorded_calls, minlength=5)[1:].tolist(),
            "reached_stage_counts": [
                int(np.count_nonzero(recorded_calls > stage)) for stage in range(3)
            ],
        },
        "losses": {
            "adaptive_raw_mse": float(np.mean(adaptive_raw)),
            "adaptive_native_whitened_mse": float(np.mean(adaptive_white)),
            "sparse_vs_dense_selected_max_abs": float(sparse_dense_delta.max()),
            "sparse_vs_dense_selected_mean_abs": float(sparse_dense_delta.mean()),
        },
        "compute": {
            "base_model_calls": ROWS,
            "refiner_model_calls": refiner_calls,
            "gate_evaluations": gate_evaluations,
            "gate_flops_per_evaluation": GATE_FLOPS,
            "gate_nonflops_per_evaluation": GATE_NONFLOPS,
            "adaptive_total_flops": adaptive_total_flops,
            "claimed_adaptive_total_flops": claimed_compute["adaptive_total_flops"],
            "exact_match_to_claimed": adaptive_total_flops
            == claimed_compute["adaptive_total_flops"],
            "seeded_total_flops": seeded_flops,
            "seeded_weakly_more_compute": seeded_flops >= adaptive_total_flops,
            "seeded_minus_adaptive_flops": seeded_flops - adaptive_total_flops,
            "analytic_equivalent_mean_calls": analytic_mean_calls,
            "raw_analytic": {
                key: value for key, value in raw_analytic.items() if key != "loss"
            },
            "native_whitened_analytic": {
                key: value for key, value in white_analytic.items() if key != "loss"
            },
            "inherited_convention_sensitivity": compute_sensitivity,
        },
        "criteria": intervals,
        "metric_comparisons": metric_comparisons,
        "episode_metric_comparisons": episode_metric_comparisons,
        "bootstrap_replicate_comparisons": bootstrap_comparisons,
        "stagewise_rank": stages,
        "stagewise_comparisons": stage_comparisons,
        "terminal_mapping": {
            "individual_pass": individual_pass,
            "simultaneous_pass": simultaneous_pass,
            "stagewise_pass": stagewise_pass,
            "integrity_pass_clean_room_subset": integrity_pass,
            "recomputed_terminal": terminal,
            "claimed_analysis_terminal": claimed_analysis["proposed_terminal_outcome"],
            "claimed_decision_terminal": claimed_decision["terminal_outcome"],
            "exact_terminal_match": terminal
            == claimed_analysis["proposed_terminal_outcome"]
            == claimed_decision["terminal_outcome"],
        },
        "sensitivity": {
            "leave_one_episode_out": loo,
            "raw_vs_analytic_100_episode_contiguous_block_means": block_means,
            "raw_vs_analytic_accumulation_orders": accumulation,
            "bootstrap_lower_quantile_methods": quantile_sensitivity,
        },
        "all_primary_agreement_checks_pass": (
            all(
                item[field]["pass"]
                for item in metric_comparisons.values()
                for field in ("estimate", "lower", "upper")
            )
            and all(item["pass"] for item in episode_metric_comparisons.values())
            and all(item["pass"] for item in bootstrap_comparisons.values())
            and all(
                item["reached_exact"]
                and item["rho"]["pass"]
                and item["sign_exact"]
                for item in stage_comparisons
            )
            and adaptive_total_flops == claimed_compute["adaptive_total_flops"]
            and terminal == claimed_decision["terminal_outcome"]
        ),
    }
    atomic_json(args.output, result)

    negative = run_negative_controls(
        raw_loss=raw_loss,
        target=target,
        dense=dense,
        sparse=sparse,
        calls=recorded_calls,
        scores=scores,
        episode_id=episode_id,
        model_step=model_step,
        analytic=raw_analytic,
        repetitions=args.repetitions,
        target_repetitions=args.target_repetitions,
    )
    negative["created_unix_ns"] = time.time_ns()
    negative["runtime_seconds_total_including_clean_room"] = time.time() - start
    atomic_json(args.negative_output, negative)
    print(
        json.dumps(
            {
                "clean_room_pass": result["all_primary_agreement_checks_pass"],
                "terminal": terminal,
                "negative_controls_pass": negative[
                    "all_random_families_diagnostic_pass"
                ],
                "runtime_seconds": time.time() - start,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
