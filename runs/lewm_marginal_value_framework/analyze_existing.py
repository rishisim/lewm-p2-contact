#!/usr/bin/env python3
"""Independent exploratory recomputation from accepted, lossless LeWM arrays.

This script never invokes an environment or model. It reads accepted artifacts
outside this directory and writes only derived exploratory artifacts beside
itself.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CUBE = REPO / "runs/lewm_v5_readiness_program/v5_package_versions/v004"
PILOT = REPO / "runs/lewm_pusht_replication_pilot"
BINARY = REPO / "runs/lewm_pusht_binary_confirmation"
NUMERIC = ROOT / "numeric"
FIGURES = ROOT / "figures"

CUBE_BASE = 70_529_190
CUBE_DEPTH1 = 669_184
CUBE_ADAPTER = 264_960
CUBE_GATE = 7_985
PUSHT_BASE = 70_383_192
PUSHT_ACTION_NORM = 60
PUSHT_REFINER = 650_432
PUSHT_GATE = 7_715

CUBE_RETENTION = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
PUSHT_FRACTIONS = tuple(x / 10 for x in range(10))
OVERHEAD_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key)) for key in fields})


def losses(dense: np.ndarray, target: np.ndarray, whitening: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    n, depths, width = dense.shape
    raw = np.empty((n, depths), dtype=np.float64)
    white = np.empty((n, depths), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        difference = dense[start:stop].astype(np.float64) - target[start:stop, None, :].astype(np.float64)
        raw[start:stop] = np.square(difference).mean(axis=2)
        transformed = difference.reshape(-1, width) @ whitening
        white[start:stop] = np.square(transformed.reshape(stop - start, depths, width)).mean(axis=2)
    return raw, white


def single_loss(prediction: np.ndarray, target: np.ndarray, whitening: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    n = len(target)
    raw = np.empty(n, dtype=np.float64)
    white = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        difference = prediction[start:stop].astype(np.float64) - target[start:stop].astype(np.float64)
        raw[start:stop] = np.square(difference).mean(axis=1)
        transformed = difference @ whitening
        white[start:stop] = np.square(transformed).mean(axis=1)
    return raw, white


def episode_means(values: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    unique = np.unique(episode_id)
    return np.asarray([np.asarray(values)[episode_id == identifier].mean() for identifier in unique], dtype=np.float64)


def strongest_mixture(loss_matrix: np.ndarray, mean_depth: float, maximum_depth: int) -> dict[str, Any] | None:
    candidates: list[tuple[float, np.ndarray, int, int, float]] = []
    for lower in range(1, maximum_depth + 1):
        for upper in range(lower, maximum_depth + 1):
            if lower - 1e-12 <= mean_depth <= upper + 1e-12:
                weight = 0.0 if lower == upper else (mean_depth - lower) / (upper - lower)
                vector = (1.0 - weight) * loss_matrix[:, lower - 1] + weight * loss_matrix[:, upper - 1]
                candidates.append((float(vector.mean()), vector, lower, upper, float(weight)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[2], item[3]))
    mean_loss, vector, lower, upper, weight = candidates[0]
    return {
        "mean_loss": mean_loss,
        "loss": vector,
        "depth_lower": lower,
        "depth_upper": upper,
        "weight_upper": weight,
        "feasible_pair_count": len(candidates),
    }


def deterministic_topk(score: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    k = min(max(int(k), 0), len(indices))
    if k == 0:
        return indices[:0]
    order = np.lexsort((indices, -np.asarray(score)[indices]))
    return indices[order[:k]]


def half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def causal_features_numpy(history: np.ndarray, actions: np.ndarray, current: np.ndarray, update: np.ndarray) -> np.ndarray:
    """Independent NumPy implementation of the accepted PushT causal boundary."""
    history = np.asarray(history, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    update = np.asarray(update, dtype=np.float32)
    epsilon = np.finfo(np.float32).eps
    last = history[:, -1]
    gap = current - last
    norm = lambda value, axis: np.linalg.vector_norm(value, axis=axis, keepdims=True).astype(np.float32)
    current_norm = norm(current, 1)
    update_norm = norm(update, 1)
    last_norm = norm(last, 1)
    gap_norm = norm(gap, 1)
    relative = update_norm / np.maximum(current_norm, epsilon)
    update_current = np.sum(update * current, axis=1, keepdims=True) / np.maximum(update_norm * current_norm, epsilon)
    update_gap = np.sum(update * gap, axis=1, keepdims=True) / np.maximum(update_norm * gap_norm, epsilon)
    history_change = np.linalg.vector_norm(history[:, 1:] - history[:, :-1], axis=2).astype(np.float32)
    action_change = np.linalg.vector_norm(actions[:, 1:] - actions[:, :-1], axis=2).astype(np.float32)
    result = np.concatenate(
        (
            history.reshape(len(history), -1), actions.reshape(len(history), -1), current, update,
            current_norm, update_norm, relative, last_norm, gap_norm, update_current, update_gap,
            history_change, action_change,
        ),
        axis=1,
    ).astype(np.float32)
    if result.shape[1] != 1001 or not np.isfinite(result).all():
        raise RuntimeError("PushT independent causal feature contract failed")
    return result


def pusht_heads(features: np.ndarray, gate: dict[str, np.ndarray], stage: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = (features.astype(np.float64) - gate["feature_mean"][stage]) / gate["feature_scale"][stage]
    heads = z @ gate["weights"][stage].T
    return heads[:, 0], heads[:, 1], np.minimum(heads[:, 0], heads[:, 1])


def add_calibration(
    dataset: str,
    stage: int,
    predicted_raw: np.ndarray,
    predicted_white: np.ndarray,
    score: np.ndarray,
    gain_raw: np.ndarray,
    gain_white: np.ndarray,
    calibration_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
) -> None:
    for endpoint, predicted, observed in (
        ("raw", predicted_raw, gain_raw),
        ("whitened", predicted_white, gain_white),
    ):
        predicted = np.asarray(predicted, dtype=np.float64)
        observed = np.asarray(observed, dtype=np.float64)
        slope, intercept = np.polyfit(predicted, observed, 1)
        rank_rows.append(
            {
                "dataset": dataset,
                "stage": stage,
                "endpoint": endpoint,
                "rows": len(predicted),
                "predicted_gain_mean": predicted.mean(),
                "observed_gain_mean": observed.mean(),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "rmse": np.sqrt(np.mean(np.square(predicted - observed))),
                "predicted_vs_observed_spearman": safe_spearman(predicted, observed),
                "gate_score_vs_observed_spearman": safe_spearman(score, observed),
            }
        )
        ordered = np.lexsort((np.arange(len(predicted)), predicted))
        for decile, rows in enumerate(np.array_split(ordered, 10), start=1):
            calibration_rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "endpoint": endpoint,
                    "decile": decile,
                    "rows": len(rows),
                    "predicted_gain_mean": predicted[rows].mean(),
                    "observed_gain_mean": observed[rows].mean(),
                    "score_mean": np.asarray(score)[rows].mean(),
                }
            )


def stage_oracle_rows(
    dataset: str,
    calls: np.ndarray,
    scores: np.ndarray,
    raw: np.ndarray,
    white: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stage in range(scores.shape[1]):
        # The pilot deliberately stored dense counterfactual scores at later
        # stages, so finiteness alone does not identify executable reach. Calls
        # do: stage 1 is reached at depth >=1, stage 2 at depth >=2, and so on.
        reached = np.flatnonzero((calls > stage) & np.isfinite(scores[:, stage]))
        selected = reached[calls[reached] > stage + 1]
        k = len(selected)
        for endpoint, matrix in (("raw", raw), ("whitened", white)):
            gain = matrix[reached, stage] - matrix[reached, stage + 1]
            local_selected = np.isin(reached, selected, assume_unique=True)
            learned = float(gain[local_selected].sum())
            random = float(k / len(reached) * gain.sum())
            oracle = float(np.sort(gain)[::-1][:k].sum()) if k else 0.0
            denominator = oracle - random
            output.append(
                {
                    "dataset": dataset,
                    "scope": "stage_local_reached_rows",
                    "stage": stage + 1,
                    "endpoint": endpoint,
                    "reached_rows": len(reached),
                    "continued_rows": k,
                    "learned_total_gain": learned,
                    "random_expected_total_gain": random,
                    "oracle_total_gain": oracle,
                    "allocation_uplift_over_random": learned - random,
                    "oracle_uplift_over_random": denominator,
                    "fraction_oracle_uplift_captured": (learned - random) / denominator if denominator > 0 else float("nan"),
                }
            )
    return output


def schedule_diagnostic(
    dataset: str,
    endpoint: str,
    matrix: np.ndarray,
    mixture: dict[str, Any],
    target_flops: int,
    fixed_cost: int,
    per_depth_cost: int,
    depth1_in_fixed: bool,
    seed: int,
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    n = len(matrix)
    lower = int(mixture["depth_lower"])
    upper = int(mixture["depth_upper"])
    if lower == upper:
        candidates = [("exact", 0)]
    else:
        target_units = (target_flops - fixed_cost) / per_depth_cost + (n if depth1_in_fixed else 0)
        upper_float = (target_units - lower * n) / (upper - lower)
        candidates = [("floor", math.floor(upper_float)), ("ceil", math.ceil(upper_float))]
    result: list[dict[str, Any]] = []
    permutation = np.random.default_rng(seed).permutation(n)
    for label, number_upper in candidates:
        number_upper = min(max(int(number_upper), 0), n)
        depths = np.full(n, lower, dtype=np.int8)
        if number_upper:
            depths[permutation[:number_upper]] = upper
        selected_loss = matrix[np.arange(n), depths - 1]
        total_depth_units = int(depths.sum())
        additional_units = total_depth_units - (n if depth1_in_fixed else 0)
        schedule_flops = int(fixed_cost + additional_units * per_depth_cost)
        key = f"{dataset}_{endpoint}_{label}"
        arrays[f"{key}_depths"] = depths
        arrays[f"{key}_loss"] = selected_loss
        result.append(
            {
                "dataset": dataset,
                "endpoint": endpoint,
                "rounding": label,
                "seed": seed,
                "depth_lower": lower,
                "depth_upper": upper,
                "number_upper": number_upper,
                "total_depth_units": total_depth_units,
                "target_flops": target_flops,
                "schedule_flops": schedule_flops,
                "schedule_minus_target_flops": schedule_flops - target_flops,
                "analytic_mixture_loss": mixture["mean_loss"],
                "realized_seeded_loss": selected_loss.mean(),
            }
        )
    return result


def cube_analysis(calibration_rows: list[dict[str, Any]], rank_rows: list[dict[str, Any]], oracle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    execution = CUBE / "data/v5_confirmation_execution.npz"
    with np.load(execution, allow_pickle=False) as stored:
        calls = stored["calls"].astype(np.int64)
        scores = stored["scores"].astype(np.float64)
        episode_id = stored["episode_id"].astype(np.int64)
    with np.load(CUBE / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        compiled = {key: stored[key].copy() for key in stored.files}
    with np.load(CUBE / "freeze/gate_fit.npz", allow_pickle=False) as stored:
        fit = {key: stored[key].copy() for key in stored.files}
    with np.load(CUBE / "freeze/whitening.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)

    raw_head = np.full(scores.shape, np.nan, dtype=np.float64)
    white_head = np.full(scores.shape, np.nan, dtype=np.float64)
    with np.load(execution, allow_pickle=False) as stored:
        features = stored["features"]
        for stage in range(3):
            finite = np.isfinite(scores[:, stage])
            indices = np.flatnonzero(finite)
            if finite.all():
                raw_value = features[:, stage, :] @ compiled["a_raw"][stage] + compiled["b_raw"][stage]
                white_value = features[:, stage, :] @ compiled["a_white"][stage] + compiled["b_white"][stage]
            else:
                raw_value = features[indices, stage, :] @ compiled["a_raw"][stage] + compiled["b_raw"][stage]
                white_value = features[indices, stage, :] @ compiled["a_white"][stage] + compiled["b_white"][stage]
            raw_head[indices, stage] = raw_value
            white_head[indices, stage] = white_value
        del features
    gc.collect()
    recomputed_scores = np.minimum(raw_head, white_head)
    finite = np.isfinite(scores)
    score_max_abs = float(np.max(np.abs(recomputed_scores[finite] - scores[finite])))
    stored_thresholds = compiled["thresholds"].astype(np.float32).astype(np.float64)
    stored_score_calls = np.ones(len(calls), dtype=np.int64)
    recomputed_head_calls = np.ones(len(calls), dtype=np.int64)
    active = np.arange(len(calls))
    for stage in range(3):
        active = active[scores[active, stage] > stored_thresholds[stage]]
        stored_score_calls[active] += 1
    active = np.arange(len(calls))
    for stage in range(3):
        active = active[recomputed_scores[active, stage] > stored_thresholds[stage]]
        recomputed_head_calls[active] += 1

    with np.load(execution, allow_pickle=False) as stored:
        target = stored["target"].copy()
        dense = stored["dense_exits"].copy()
        sparse = stored["sparse_selected"].copy()
    raw, white = losses(dense, target, whitening)
    sparse_raw, sparse_white = single_loss(sparse, target, whitening)
    row = np.arange(len(calls))
    selected_raw = raw[row, calls - 1]
    selected_white = white[row, calls - 1]

    for stage in range(3):
        reached = np.flatnonzero(np.isfinite(scores[:, stage]))
        predicted_raw = raw_head[reached, stage] * fit["raw_target_std"][stage] + fit["raw_target_mean"][stage]
        predicted_white = white_head[reached, stage] * fit["white_target_std"][stage] + fit["white_target_mean"][stage]
        add_calibration(
            "cube_v5", stage + 1, predicted_raw, predicted_white, scores[reached, stage],
            raw[reached, stage] - raw[reached, stage + 1],
            white[reached, stage] - white[reached, stage + 1], calibration_rows, rank_rows,
        )
    oracle_rows.extend(stage_oracle_rows("cube_v5", calls, scores, raw, white))

    frontier: list[dict[str, Any]] = []
    episode_arrays: dict[str, np.ndarray] = {"episode_id": np.unique(episode_id), "retention": np.asarray(CUBE_RETENTION)}
    ep_learned_raw: list[np.ndarray] = []
    ep_learned_white: list[np.ndarray] = []
    ep_comp_raw: list[np.ndarray] = []
    ep_comp_white: list[np.ndarray] = []
    common = len(calls) * (CUBE_BASE + CUBE_DEPTH1)
    fixed_depth1 = common
    for retention in CUBE_RETENTION:
        policy_calls = np.ones(len(calls), dtype=np.int64)
        active = np.arange(len(calls))
        gate_evaluations = 0
        continued_by_stage: list[int] = []
        for stage in range(3):
            gate_evaluations += len(active)
            eligible = active[scores[active, stage] > stored_thresholds[stage]]
            k = half_up(retention * len(eligible))
            active = deterministic_topk(scores[:, stage], eligible, k)
            policy_calls[active] += 1
            continued_by_stage.append(len(active))
        learned_raw = raw[row, policy_calls - 1]
        learned_white = white[row, policy_calls - 1]
        total_flops = int(common + (policy_calls.sum() - len(calls)) * CUBE_ADAPTER + gate_evaluations * CUBE_GATE)
        mean_depth = 1.0 + (total_flops - common) / (len(calls) * CUBE_ADAPTER)
        comp_raw = strongest_mixture(raw, mean_depth, 4)
        comp_white = strongest_mixture(white, mean_depth, 4)
        assert comp_raw is not None and comp_white is not None
        frontier.append(
            {
                "dataset": "cube_v5",
                "policy_family": "conservative_frozen-continuation_topk_thinning",
                "retention_fraction": retention,
                "frozen_operating_point": retention == 1.0,
                "total_counted_flops": total_flops,
                "counted_flops_per_transition": total_flops / len(calls),
                "extra_mflops_per_transition_over_fixed_depth1": (total_flops - fixed_depth1) / len(calls) / 1e6,
                "equivalent_transition_independent_mean_depth": mean_depth,
                "gate_evaluations": gate_evaluations,
                "refiner_calls": int(policy_calls.sum()),
                "depth1_rows": int(np.sum(policy_calls == 1)),
                "depth2_rows": int(np.sum(policy_calls == 2)),
                "depth3_rows": int(np.sum(policy_calls == 3)),
                "depth4_rows": int(np.sum(policy_calls == 4)),
                "continued_stage1": continued_by_stage[0],
                "continued_stage2": continued_by_stage[1],
                "continued_stage3": continued_by_stage[2],
                "learned_raw_mse": learned_raw.mean(),
                "ti_envelope_raw_mse": comp_raw["mean_loss"],
                "raw_benefit_vs_ti_envelope": comp_raw["mean_loss"] - learned_raw.mean(),
                "raw_ti_depth_lower": comp_raw["depth_lower"],
                "raw_ti_depth_upper": comp_raw["depth_upper"],
                "learned_whitened_mse": learned_white.mean(),
                "ti_envelope_whitened_mse": comp_white["mean_loss"],
                "whitened_benefit_vs_ti_envelope": comp_white["mean_loss"] - learned_white.mean(),
                "whitened_ti_depth_lower": comp_white["depth_lower"],
                "whitened_ti_depth_upper": comp_white["depth_upper"],
            }
        )
        ep_learned_raw.append(episode_means(learned_raw, episode_id))
        ep_learned_white.append(episode_means(learned_white, episode_id))
        ep_comp_raw.append(episode_means(comp_raw["loss"], episode_id))
        ep_comp_white.append(episode_means(comp_white["loss"], episode_id))
    episode_arrays.update(
        learned_raw=np.stack(ep_learned_raw), learned_whitened=np.stack(ep_learned_white),
        ti_raw=np.stack(ep_comp_raw), ti_whitened=np.stack(ep_comp_white),
    )
    np.savez_compressed(NUMERIC / "cube_frontier_episode_arrays.npz", **episode_arrays)

    frozen_total = int(common + (calls.sum() - len(calls)) * CUBE_ADAPTER + np.minimum(calls, 3).sum() * CUBE_GATE)
    frozen_mean = 1 + (frozen_total - common) / (len(calls) * CUBE_ADAPTER)
    frozen_mix_raw = strongest_mixture(raw, frozen_mean, 4)
    frozen_mix_white = strongest_mixture(white, frozen_mean, 4)
    assert frozen_mix_raw is not None and frozen_mix_white is not None
    return {
        "frontier": frontier,
        "raw": raw,
        "white": white,
        "calls": calls,
        "scores": scores,
        "episode_id": episode_id,
        "fixed_depth1_flops": fixed_depth1,
        "frozen_total_flops": frozen_total,
        "frozen_gate_evaluations": int(np.minimum(calls, 3).sum()),
        "frozen_selected_raw": selected_raw,
        "frozen_selected_white": selected_white,
        "frozen_mix_raw": frozen_mix_raw,
        "frozen_mix_white": frozen_mix_white,
        "audit": {
            "source_sha256": sha256(execution),
            "rows": len(calls),
            "episodes": len(np.unique(episode_id)),
            "rows_per_episode_unique": np.unique(np.bincount(episode_id)).tolist(),
            "score_finite_counts": np.isfinite(scores).sum(axis=0).tolist(),
            "score_recomputation_max_abs": score_max_abs,
            "stored_score_calls_exact": bool(np.array_equal(stored_score_calls, calls)),
            "recomputed_head_calls_exact": bool(np.array_equal(recomputed_head_calls, calls)),
            "call_histogram": np.bincount(calls, minlength=5)[1:5].tolist(),
            "dense_selected_vs_sparse_raw_mean_abs": float(np.mean(np.abs(selected_raw - sparse_raw))),
            "dense_selected_vs_sparse_white_mean_abs": float(np.mean(np.abs(selected_white - sparse_white))),
            "sparse_raw_mean": sparse_raw.mean(),
            "sparse_whitened_mean": sparse_white.mean(),
            "dense_selected_raw_mean": selected_raw.mean(),
            "dense_selected_whitened_mean": selected_white.mean(),
            "analytic_raw_mean": frozen_mix_raw["mean_loss"],
            "analytic_whitened_mean": frozen_mix_white["mean_loss"],
            "frozen_total_flops": frozen_total,
            "retention_one_calls_exact": bool(np.array_equal(
                np.asarray([frontier[-1][f"depth{depth}_rows"] for depth in range(1, 5)]),
                np.bincount(calls, minlength=5)[1:5],
            )),
        },
    }


def pusht_dataset(
    dataset: str,
    evaluation_path: Path,
    input_path: Path | None,
    binary: bool,
    calibration_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    with np.load(PILOT / "FROZEN_GATE.npz", allow_pickle=False) as stored:
        gate = {key: stored[key].copy() for key in stored.files}
    with np.load(PILOT / "FIT_ARTIFACTS.npz", allow_pickle=False) as stored:
        fit = {key: stored[key].copy() for key in stored.files}
    with np.load(evaluation_path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if input_path is not None:
        with np.load(input_path, allow_pickle=False) as stored:
            inputs = {key: stored[key].copy() for key in ("history", "actions", "base", "target")}
        if not np.array_equal(inputs["target"], arrays["target"]):
            raise RuntimeError("PushT pilot evaluation target drift")
    else:
        inputs = {key: arrays[key] for key in ("history", "actions", "base", "target")}

    target = arrays["target"]
    dense = arrays["dense_exits"]
    calls = arrays["calls"].astype(np.int64)
    episode_id = arrays["episode_id"].astype(np.int64)
    raw, white = losses(dense, target, fit["whitening_matrix"].astype(np.float64))
    normalized_actions = ((inputs["actions"].astype(np.float64) - fit["action_mean"]) / fit["action_scale"]).astype(np.float32)
    stages = 1 if binary else 3
    score_matrix = np.full((len(calls), stages), np.nan, dtype=np.float64)
    raw_heads = np.full_like(score_matrix, np.nan)
    white_heads = np.full_like(score_matrix, np.nan)
    for stage in range(stages):
        current = dense[:, stage]
        update = arrays["stage1_update"] if binary else arrays["dense_updates"][:, stage]
        features = causal_features_numpy(inputs["history"], normalized_actions, current, update)
        head_raw, head_white, score = pusht_heads(features, gate, stage)
        raw_heads[:, stage] = head_raw
        white_heads[:, stage] = head_white
        score_matrix[:, stage] = score
        predicted_raw = head_raw * gate["raw_gain_scale"][stage] + gate["raw_gain_mean"][stage]
        predicted_white = head_white * gate["white_gain_scale"][stage] + gate["white_gain_mean"][stage]
        reached = np.arange(len(calls)) if binary else np.flatnonzero(calls > stage)
        add_calibration(
            dataset, stage + 1, predicted_raw[reached], predicted_white[reached], score[reached],
            raw[reached, stage] - raw[reached, stage + 1],
            white[reached, stage] - white[reached, stage + 1], calibration_rows, rank_rows,
        )

    stored_score = arrays["stage1_score"][:, None] if binary else arrays["scores"]
    score_max_abs = float(np.max(np.abs(score_matrix - stored_score)))
    recomputed_calls = np.ones(len(calls), dtype=np.int64)
    active = np.arange(len(calls))
    for stage in range(stages):
        active = active[score_matrix[active, stage] > gate["thresholds"][stage]]
        recomputed_calls[active] += 1
    oracle_rows.extend(stage_oracle_rows(dataset, calls, stored_score.astype(np.float64), raw, white))
    row = np.arange(len(calls))
    selected_raw = raw[row, calls - 1]
    selected_white = white[row, calls - 1]
    gate_evaluations = len(calls) if binary else int(np.minimum(calls, 3).sum())
    base_common = len(calls) * (PUSHT_BASE + PUSHT_ACTION_NORM)
    total_flops = int(base_common + calls.sum() * PUSHT_REFINER + gate_evaluations * PUSHT_GATE)
    mean_depth = (total_flops - base_common) / (len(calls) * PUSHT_REFINER)
    maximum_depth = 2 if binary else 4
    mix_raw = strongest_mixture(raw, mean_depth, maximum_depth)
    mix_white = strongest_mixture(white, mean_depth, maximum_depth)
    assert mix_raw is not None and mix_white is not None
    result: dict[str, Any] = {
        "raw": raw,
        "white": white,
        "calls": calls,
        "scores": stored_score.astype(np.float64),
        "episode_id": episode_id,
        "fixed_depth1_flops": int(base_common + len(calls) * PUSHT_REFINER),
        "frozen_total_flops": total_flops,
        "frozen_gate_evaluations": gate_evaluations,
        "frozen_selected_raw": selected_raw,
        "frozen_selected_white": selected_white,
        "frozen_mix_raw": mix_raw,
        "frozen_mix_white": mix_white,
        "audit": {
            "source_sha256": sha256(evaluation_path),
            "rows": len(calls),
            "episodes": len(np.unique(episode_id)),
            "rows_per_episode_unique": np.unique(np.bincount(episode_id)).tolist(),
            "score_recomputation_max_abs": score_max_abs,
            "recomputed_calls_exact": bool(np.array_equal(recomputed_calls, calls)),
            "call_histogram": np.bincount(calls, minlength=maximum_depth + 1)[1:].tolist(),
            "adaptive_raw_mean": selected_raw.mean(),
            "adaptive_whitened_mean": selected_white.mean(),
            "analytic_raw_mean": mix_raw["mean_loss"],
            "analytic_whitened_mean": mix_white["mean_loss"],
            "frozen_total_flops": total_flops,
        },
    }
    if not binary:
        return result

    frozen_k = int(np.sum(calls == 2))
    maximum_k = int(math.floor(len(calls) * (1.0 - PUSHT_GATE / PUSHT_REFINER)))
    counts = {0, maximum_k, frozen_k}
    counts.update(half_up(fraction * len(calls)) for fraction in PUSHT_FRACTIONS)
    counts = {k for k in counts if 0 <= k <= maximum_k}
    frontier: list[dict[str, Any]] = []
    ep: dict[str, list[np.ndarray]] = {
        "learned_raw": [], "learned_whitened": [], "ti_raw": [], "ti_whitened": [],
        "oracle_raw": [], "oracle_whitened": [],
    }
    stage_score = stored_score[:, 0]
    all_rows = np.arange(len(calls))
    for k in sorted(counts):
        learned_rows = deterministic_topk(stage_score, all_rows, k)
        learned_mask = np.zeros(len(calls), dtype=bool)
        learned_mask[learned_rows] = True
        policy_calls = 1 + learned_mask.astype(np.int64)
        learned_raw = raw[row, policy_calls - 1]
        learned_white = white[row, policy_calls - 1]
        policy_total = int(base_common + policy_calls.sum() * PUSHT_REFINER + len(calls) * PUSHT_GATE)
        policy_mean = (policy_total - base_common) / (len(calls) * PUSHT_REFINER)
        comparator_raw = strongest_mixture(raw, policy_mean, 2)
        comparator_white = strongest_mixture(white, policy_mean, 2)
        assert comparator_raw is not None and comparator_white is not None
        oracle_values: dict[str, np.ndarray] = {}
        capture: dict[str, float] = {}
        for endpoint, matrix in (("raw", raw), ("whitened", white)):
            gain = matrix[:, 0] - matrix[:, 1]
            oracle_rows_k = deterministic_topk(gain, all_rows, k)
            oracle_mask = np.zeros(len(calls), dtype=bool)
            oracle_mask[oracle_rows_k] = True
            oracle_loss = np.where(oracle_mask, matrix[:, 1], matrix[:, 0])
            oracle_values[endpoint] = oracle_loss
            learned_gain = float(gain[learned_mask].sum())
            random_gain = float(k / len(calls) * gain.sum())
            oracle_gain = float(gain[oracle_mask].sum())
            denominator = oracle_gain - random_gain
            capture[endpoint] = (learned_gain - random_gain) / denominator if denominator > 0 else float("nan")
        frontier.append(
            {
                "dataset": dataset,
                "policy_family": "stage1_score_exact_topk",
                "optional_depth2_rows": k,
                "allocation_fraction": k / len(calls),
                "frozen_operating_point": k == frozen_k,
                "total_counted_flops": policy_total,
                "counted_flops_per_transition": policy_total / len(calls),
                "extra_mflops_per_transition_over_fixed_depth1": (policy_total - result["fixed_depth1_flops"]) / len(calls) / 1e6,
                "equivalent_transition_independent_mean_depth": policy_mean,
                "gate_evaluations": len(calls),
                "refiner_calls": int(policy_calls.sum()),
                "depth1_rows": int(np.sum(policy_calls == 1)),
                "depth2_rows": int(np.sum(policy_calls == 2)),
                "learned_raw_mse": learned_raw.mean(),
                "ti_envelope_raw_mse": comparator_raw["mean_loss"],
                "oracle_raw_mse": oracle_values["raw"].mean(),
                "raw_benefit_vs_ti_envelope": comparator_raw["mean_loss"] - learned_raw.mean(),
                "raw_fraction_oracle_allocation_uplift_captured": capture["raw"],
                "learned_whitened_mse": learned_white.mean(),
                "ti_envelope_whitened_mse": comparator_white["mean_loss"],
                "oracle_whitened_mse": oracle_values["whitened"].mean(),
                "whitened_benefit_vs_ti_envelope": comparator_white["mean_loss"] - learned_white.mean(),
                "whitened_fraction_oracle_allocation_uplift_captured": capture["whitened"],
            }
        )
        ep["learned_raw"].append(episode_means(learned_raw, episode_id))
        ep["learned_whitened"].append(episode_means(learned_white, episode_id))
        ep["ti_raw"].append(episode_means(comparator_raw["loss"], episode_id))
        ep["ti_whitened"].append(episode_means(comparator_white["loss"], episode_id))
        ep["oracle_raw"].append(episode_means(oracle_values["raw"], episode_id))
        ep["oracle_whitened"].append(episode_means(oracle_values["whitened"], episode_id))
    np.savez_compressed(
        NUMERIC / "pusht_binary_frontier_episode_arrays.npz",
        episode_id=np.unique(episode_id), optional_depth2_rows=np.asarray(sorted(counts), dtype=np.int64),
        **{key: np.stack(value) for key, value in ep.items()},
    )
    result["frontier"] = frontier
    result["audit"]["maximum_topk_count_under_fixed_depth2_cost"] = maximum_k
    result["audit"]["frozen_topk_reproduces_calls"] = bool(
        np.array_equal(
            1 + np.isin(all_rows, deterministic_topk(stage_score, all_rows, frozen_k)).astype(np.int64), calls
        )
    )
    return result


def overhead_rows(dataset: str, result: dict[str, Any], adapter: int, base_common: int, maximum_depth: int, depth1_in_common: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    n = len(result["calls"])
    extra_calls = int(result["calls"].sum() - (n if depth1_in_common else 0))
    gate = CUBE_GATE if dataset == "cube_v5" else PUSHT_GATE
    for multiplier in OVERHEAD_MULTIPLIERS:
        total = int(round(base_common + extra_calls * adapter + result["frozen_gate_evaluations"] * gate * multiplier))
        mean_depth = (total - base_common) / (n * adapter) + (1 if depth1_in_common else 0)
        for endpoint, matrix, learned in (
            ("raw", result["raw"], result["frozen_selected_raw"]),
            ("whitened", result["white"], result["frozen_selected_white"]),
        ):
            comparator = strongest_mixture(matrix, mean_depth, maximum_depth)
            output.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "gate_overhead_multiplier": multiplier,
                    "total_counted_flops": total,
                    "equivalent_ti_mean_depth": mean_depth,
                    "comparator_feasible": comparator is not None,
                    "learned_loss": learned.mean(),
                    "ti_envelope_loss": comparator["mean_loss"] if comparator is not None else float("nan"),
                    "benefit_vs_ti_envelope": comparator["mean_loss"] - learned.mean() if comparator is not None else float("nan"),
                }
            )
    return output


def fixed_depth_rows(dataset: str, result: dict[str, Any], maximum_depth: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    n = len(result["calls"])
    for depth in range(1, maximum_depth + 1):
        if dataset == "cube_v5":
            total = n * (CUBE_BASE + CUBE_DEPTH1 + (depth - 1) * CUBE_ADAPTER)
        else:
            total = n * (PUSHT_BASE + PUSHT_ACTION_NORM + depth * PUSHT_REFINER)
        for endpoint, matrix in (("raw", result["raw"]), ("whitened", result["white"])):
            output.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "depth": depth,
                    "total_counted_flops": total,
                    "counted_flops_per_transition": total / n,
                    "mean_loss": matrix[:, depth - 1].mean(),
                }
            )
    return output


def make_plots(cube: dict[str, Any], binary: dict[str, Any], pilot: dict[str, Any], calibration_rows: list[dict[str, Any]], overhead: list[dict[str, Any]]) -> None:
    colors = {"learned": "#154360", "ti": "#D35400", "oracle": "#1E8449"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for row_index, (dataset, result, title) in enumerate((
        ("cube_v5", cube, "Cube v5 (conservative thinning)"),
        ("pusht_binary", binary, "PushT binary (exact score top-k)"),
    )):
        rows = result["frontier"]
        x = np.asarray([item["extra_mflops_per_transition_over_fixed_depth1"] for item in rows])
        for column, (endpoint, label) in enumerate((("raw", "Raw latent MSE"), ("whitened", "Whitened latent MSE"))):
            ax = axes[row_index, column]
            learned = np.asarray([item[f"learned_{endpoint}_mse"] for item in rows])
            ti = np.asarray([item[f"ti_envelope_{endpoint}_mse"] for item in rows])
            ax.plot(x, learned, "o-", color=colors["learned"], label="causal score allocation")
            ax.plot(x, ti, "s--", color=colors["ti"], label="TI lower envelope")
            if dataset == "pusht_binary":
                oracle = np.asarray([item[f"oracle_{endpoint}_mse"] for item in rows])
                ax.plot(x, oracle, "^:", color=colors["oracle"], label="outcome-oracle top-k")
            frozen = np.flatnonzero([item["frozen_operating_point"] for item in rows])
            ax.scatter(x[frozen], learned[frozen], marker="*", s=150, color="#7D3C98", zorder=5, label="accepted operating point")
            ax.set_title(f"{title}: {label}")
            ax.set_xlabel("Extra counted MFLOPs / transition over gate-free depth 1")
            ax.set_ylabel(label)
            ax.grid(alpha=0.22)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            if row_index == 0 and column == 0:
                ax.legend(fontsize=8)
    fig.savefig(FIGURES / "quality_compute_frontiers.png", dpi=180)
    fig.savefig(FIGURES / "quality_compute_frontiers.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), constrained_layout=True)
    ax = axes[0, 0]
    width = 0.24
    for index, (name, result, color) in enumerate((
        ("Cube v5", cube, "#154360"), ("PushT 4-depth pilot", pilot, "#A04000"), ("PushT binary", binary, "#1E8449"),
    )):
        hist = np.bincount(result["calls"], minlength=5)[1:5].astype(float)
        hist /= hist.sum()
        ax.bar(np.arange(1, 5) + (index - 1) * width, hist, width=width, label=name, color=color)
    ax.set_xticks(range(1, 5))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Executed refinement depth")
    ax.set_ylabel("Fraction of transitions")
    ax.set_title("Accepted allocation histograms")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    for axis, (dataset, title) in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]),
        (("cube_v5", "Cube v5 calibration"), ("pusht_pilot", "PushT 4-depth pilot calibration"), ("pusht_binary", "PushT binary calibration")),
    ):
        subset = [row for row in calibration_rows if row["dataset"] == dataset]
        endpoint_color = {"raw": "#154360", "whitened": "#D35400"}
        markers = {1: "o", 2: "s", 3: "^"}
        values: list[float] = []
        for endpoint in ("raw", "whitened"):
            for stage in sorted({int(row["stage"]) for row in subset}):
                rows = [row for row in subset if row["endpoint"] == endpoint and int(row["stage"]) == stage]
                if not rows:
                    continue
                x = np.asarray([row["predicted_gain_mean"] for row in rows])
                y = np.asarray([row["observed_gain_mean"] for row in rows])
                values.extend(x.tolist() + y.tolist())
                axis.plot(x, y, marker=markers[stage], linewidth=1, markersize=4,
                          color=endpoint_color[endpoint], alpha=0.55 + 0.15 * stage,
                          label=f"{endpoint}, stage {stage}")
        low, high = min(values), max(values)
        axis.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=1, label="perfect calibration")
        axis.axhline(0, color="#999999", linewidth=0.7)
        axis.axvline(0, color="#999999", linewidth=0.7)
        axis.set_xlabel("Mean predicted marginal gain")
        axis.set_ylabel("Mean observed marginal gain")
        axis.set_title(title + " (exploratory deciles)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)
    fig.savefig(FIGURES / "allocation_and_calibration.png", dpi=180)
    fig.savefig(FIGURES / "allocation_and_calibration.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    dataset_style = {
        "cube_v5": ("Cube v5", "o", "#154360"),
        "pusht_pilot": ("PushT 4-depth pilot", "s", "#A04000"),
        "pusht_binary": ("PushT binary", "^", "#1E8449"),
    }
    for ax, endpoint in zip(axes, ("raw", "whitened")):
        for dataset, (label, marker, color) in dataset_style.items():
            rows = [row for row in overhead if row["dataset"] == dataset and row["endpoint"] == endpoint and row["comparator_feasible"]]
            ax.plot([row["gate_overhead_multiplier"] for row in rows], [row["benefit_vs_ti_envelope"] for row in rows],
                    marker=marker, color=color, label=label)
        ax.axhline(0, color="#555555", linewidth=1)
        ax.axvline(1, color="#7D3C98", linestyle="--", linewidth=1, label="accepted ledger" if endpoint == "raw" else None)
        ax.set_xlabel("Gate FLOP multiplier")
        ax.set_ylabel("TI-envelope MSE minus causal-policy MSE")
        ax.set_title(f"Frozen allocation overhead sensitivity: {endpoint}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(FIGURES / "gate_overhead_sensitivity.png", dpi=180)
    fig.savefig(FIGURES / "gate_overhead_sensitivity.pdf")
    plt.close(fig)


def main() -> None:
    NUMERIC.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    calibration_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    cube = cube_analysis(calibration_rows, rank_rows, oracle_rows)
    pilot = pusht_dataset(
        "pusht_pilot", PILOT / "EVALUATION_ARRAYS.npz", PILOT / "data/evaluation.npz", False,
        calibration_rows, rank_rows, oracle_rows,
    )
    binary = pusht_dataset(
        "pusht_binary", BINARY / "EVALUATION_ARRAYS.npz", None, True,
        calibration_rows, rank_rows, oracle_rows,
    )

    overhead: list[dict[str, Any]] = []
    overhead.extend(overhead_rows("cube_v5", cube, CUBE_ADAPTER, len(cube["calls"]) * (CUBE_BASE + CUBE_DEPTH1), 4, True))
    overhead.extend(overhead_rows("pusht_pilot", pilot, PUSHT_REFINER, len(pilot["calls"]) * (PUSHT_BASE + PUSHT_ACTION_NORM), 4, False))
    overhead.extend(overhead_rows("pusht_binary", binary, PUSHT_REFINER, len(binary["calls"]) * (PUSHT_BASE + PUSHT_ACTION_NORM), 2, False))

    fixed: list[dict[str, Any]] = []
    fixed.extend(fixed_depth_rows("cube_v5", cube, 4))
    fixed.extend(fixed_depth_rows("pusht_pilot", pilot, 4))
    fixed.extend(fixed_depth_rows("pusht_binary", binary, 2))

    schedule_arrays: dict[str, np.ndarray] = {}
    schedule_rows: list[dict[str, Any]] = []
    for offset, (dataset, result, fixed_cost, per_depth, depth1_in_fixed) in enumerate((
        ("cube_v5", cube, len(cube["calls"]) * (CUBE_BASE + CUBE_DEPTH1), CUBE_ADAPTER, True),
        ("pusht_pilot", pilot, len(pilot["calls"]) * (PUSHT_BASE + PUSHT_ACTION_NORM), PUSHT_REFINER, False),
        ("pusht_binary", binary, len(binary["calls"]) * (PUSHT_BASE + PUSHT_ACTION_NORM), PUSHT_REFINER, False),
    )):
        for endpoint_index, (endpoint, matrix, mixture) in enumerate((
            ("raw", result["raw"], result["frozen_mix_raw"]),
            ("whitened", result["white"], result["frozen_mix_white"]),
        )):
            schedule_rows.extend(schedule_diagnostic(
                dataset, endpoint, matrix, mixture, result["frozen_total_flops"], fixed_cost,
                per_depth, depth1_in_fixed, 93_710 + 100 * offset + endpoint_index, schedule_arrays,
            ))
    np.savez_compressed(NUMERIC / "finite_schedule_arrays.npz", **schedule_arrays)

    write_csv(NUMERIC / "cube_frontier.csv", cube["frontier"])
    write_csv(NUMERIC / "pusht_binary_frontier.csv", binary["frontier"])
    write_csv(NUMERIC / "fixed_depth_points.csv", fixed)
    write_csv(NUMERIC / "calibration_deciles.csv", calibration_rows)
    write_csv(NUMERIC / "rank_and_calibration_summary.csv", rank_rows)
    write_csv(NUMERIC / "oracle_headroom.csv", oracle_rows)
    write_csv(NUMERIC / "gate_overhead_sensitivity.csv", overhead)
    write_csv(NUMERIC / "finite_integer_scheduling.csv", schedule_rows)
    write_json(NUMERIC / "pusht_four_depth_pilot_point.json", clean({
        "chronology": "four-depth pilot preceded and motivated the later binary confirmation",
        "scientific_label": "process-valid pilot not supported because fit-whitened co-primary point benefit was negative",
        **pilot["audit"],
        "raw_benefit_vs_ti_envelope": pilot["frozen_mix_raw"]["mean_loss"] - pilot["frozen_selected_raw"].mean(),
        "whitened_benefit_vs_ti_envelope": pilot["frozen_mix_white"]["mean_loss"] - pilot["frozen_selected_white"].mean(),
    }))

    accepted_cube = json.loads((CUBE / "analysis_result.json").read_text())
    accepted_pilot = json.loads((PILOT / "RESULTS.json").read_text())["evaluation"]
    accepted_binary = json.loads((BINARY / "DECISION.json").read_text())
    audit = {
        "method": "independent float64 recomputation from accepted lossless arrays; no model or environment execution",
        "source_artifacts_read_only": True,
        "fresh_outcomes_generated": False,
        "cube": cube["audit"],
        "pusht_four_depth_pilot": pilot["audit"],
        "pusht_binary": binary["audit"],
        "headline_deltas_recomputed_minus_accepted": {
            "cube_adaptive_raw": cube["audit"]["sparse_raw_mean"] - accepted_cube["adaptive_raw_mse"],
            "cube_adaptive_whitened": cube["audit"]["sparse_whitened_mean"] - accepted_cube["adaptive_planoracle_native_whitened_mse"],
            "cube_analytic_raw": cube["audit"]["analytic_raw_mean"] - accepted_cube["compute"]["raw_analytic_mixture"]["mean_loss"],
            "cube_total_flops": cube["audit"]["frozen_total_flops"] - accepted_cube["compute"]["adaptive_total_flops"],
            "pilot_adaptive_raw": pilot["audit"]["adaptive_raw_mean"] - accepted_pilot["adaptive_raw_mse"],
            "pilot_adaptive_whitened": pilot["audit"]["adaptive_whitened_mean"] - accepted_pilot["adaptive_fit_whitened_mse"],
            "pilot_total_flops": pilot["audit"]["frozen_total_flops"] - accepted_pilot["compute"]["adaptive_total_counted_flops"],
            "binary_adaptive_raw": binary["audit"]["adaptive_raw_mean"] - accepted_binary["absolute_episode_loss_intervals"]["adaptive_raw"]["mean"],
            "binary_adaptive_whitened": binary["audit"]["adaptive_whitened_mean"] - accepted_binary["absolute_episode_loss_intervals"]["adaptive_fit_whitened"]["mean"],
            "binary_total_flops": binary["audit"]["frozen_total_flops"] - accepted_binary["compute"]["adaptive_total_counted_flops"],
        },
        "validation": {
            "cube_call_reconstruction": cube["audit"]["stored_score_calls_exact"],
            "cube_independent_head_call_reconstruction": cube["audit"]["recomputed_head_calls_exact"],
            "cube_frontier_endpoint_reconstruction": cube["audit"]["retention_one_calls_exact"],
            "pusht_pilot_call_reconstruction": pilot["audit"]["recomputed_calls_exact"],
            "pusht_binary_call_reconstruction": binary["audit"]["recomputed_calls_exact"],
            "pusht_binary_topk_frozen_reconstruction": binary["audit"]["frozen_topk_reproduces_calls"],
            "all_headline_absolute_deltas_at_most_1e-12": None,
            "all_compute_deltas_zero": None,
        },
        "intentional_missingness": {
            "cube_scores": "NaN exactly for stages not reached by the frozen sparse policy; this prohibits more-permissive retrospective sweeps",
            "pusht_scores": "all required pilot scores and all binary stage-1 scores are finite",
        },
    }
    absolute_keys = [key for key in audit["headline_deltas_recomputed_minus_accepted"] if not key.endswith("flops")]
    compute_keys = [key for key in audit["headline_deltas_recomputed_minus_accepted"] if key.endswith("flops")]
    audit["validation"]["all_headline_absolute_deltas_at_most_1e-12"] = all(
        abs(audit["headline_deltas_recomputed_minus_accepted"][key]) <= 1e-12 for key in absolute_keys
    )
    audit["validation"]["all_compute_deltas_zero"] = all(
        audit["headline_deltas_recomputed_minus_accepted"][key] == 0 for key in compute_keys
    )
    if not all(audit["validation"].values()):
        raise RuntimeError(f"independent validation failed: {audit['validation']}")
    write_json(NUMERIC / "independent_audit.json", clean(audit))

    make_plots(cube, binary, pilot, calibration_rows, overhead)
    write_json(ROOT / "FIGURE_MANIFEST.json", {
        "quality_compute_frontiers": {
            "png": "figures/quality_compute_frontiers.png", "pdf": "figures/quality_compute_frontiers.pdf",
            "claim": "Exploratory consumed-cohort quality-versus-counted-compute comparison against the transition-independent lower envelope; Cube is conservative thinning, PushT is exact top-k.",
        },
        "allocation_and_calibration": {
            "png": "figures/allocation_and_calibration.png", "pdf": "figures/allocation_and_calibration.pdf",
            "claim": "Accepted depth histograms and exploratory predicted-versus-observed marginal-gain deciles.",
        },
        "gate_overhead_sensitivity": {
            "png": "figures/gate_overhead_sensitivity.png", "pdf": "figures/gate_overhead_sensitivity.pdf",
            "claim": "Frozen allocation repriced under gate-overhead multipliers; allocation and outcomes fixed.",
        },
    })
    print(json.dumps({
        "cube_audit": cube["audit"],
        "pilot_audit": pilot["audit"],
        "binary_audit": binary["audit"],
        "outputs": len(list(NUMERIC.iterdir())) + len(list(FIGURES.iterdir())),
    }, indent=2, default=clean))


if __name__ == "__main__":
    main()
