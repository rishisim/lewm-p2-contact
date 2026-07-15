"""Fold-isolated routing evaluation and discovery decision utilities."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import common
import students


SUPPORTED_CALLS = np.asarray([1, 2, 3, 4], dtype=np.int64)


def prior_policy() -> Any:
    _, _, policy, _ = common.load_prior_modules()
    return policy


def gains_from_losses(losses: np.ndarray) -> np.ndarray:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("losses must have four fixed exits")
    return values[:, :-1] - values[:, 1:]


def calibrate_price(scores: np.ndarray, target_mean_calls: float) -> dict[str, Any]:
    return prior_policy().calibrate_compute_price(scores, float(target_mean_calls), exit_calls=SUPPORTED_CALLS)


def sequential_calls(scores: np.ndarray, price: float) -> np.ndarray:
    return prior_policy().causal_sequential_stopping(scores, compute_price=float(price), exit_calls=SUPPORTED_CALLS)


def selected_loss(losses: np.ndarray, calls: np.ndarray) -> np.ndarray:
    return prior_policy().gather_exit_values(losses, calls, SUPPORTED_CALLS)


def inner_routing_utility(scores: np.ndarray, losses: np.ndarray, budgets: Sequence[float]) -> dict[str, Any]:
    """Decision-aligned hyperparameter objective on strict inner OOF scores."""
    values = np.asarray(losses, dtype=np.float64)
    rows = []
    for budget in budgets:
        calibrated = calibrate_price(scores, float(budget))
        calls = sequential_calls(scores, float(calibrated["compute_price"]))
        adaptive = selected_loss(values, calls)
        # Exact histogram shuffle removes local allocation while retaining calls.
        null_calls = prior_policy().randomized_histogram_control(calls, 9011 + int(round(100 * float(budget))))
        null = selected_loss(values, null_calls)
        oracle_price = calibrate_price(gains_from_losses(values), float(calls.mean()))
        oracle_calls = sequential_calls(gains_from_losses(values), float(oracle_price["compute_price"]))
        oracle = selected_loss(values, oracle_calls)
        rows.append({
            "target_mean_calls": float(budget),
            "realized_mean_calls": float(calls.mean()),
            "allocation_benefit_vs_histogram": float((null - adaptive).mean()),
            "routing_regret_vs_causal_oracle": float((adaptive - oracle).mean()),
        })
    utility = float(np.mean([row["allocation_benefit_vs_histogram"] - 0.1 * row["routing_regret_vs_causal_oracle"] for row in rows]))
    return {"utility": utility, "operating_points": rows}


def evaluate_outer_fold(
    *,
    train_scores_oof: np.ndarray,
    eval_scores: np.ndarray,
    train_losses: np.ndarray,
    eval_losses: np.ndarray,
    eval_white_losses: np.ndarray,
    eval_episodes: np.ndarray,
    budgets: Sequence[float],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Apply fold-frozen prices/baselines to never-trained-on outer rows."""
    policy = prior_policy()
    result = {}
    for budget_index, budget in enumerate(budgets):
        calibration = calibrate_price(train_scores_oof, float(budget))
        price = float(calibration["compute_price"])
        calls = sequential_calls(eval_scores, price)
        adaptive = selected_loss(eval_losses, calls)
        white_adaptive = selected_loss(eval_white_losses, calls)
        exact = policy.strongest_transition_independent_baseline(
            train_losses,
            SUPPORTED_CALLS,
            n=len(calls),
            target_total_calls=int(calls.sum()),
            seed=int(seed) + 1009 * budget_index,
        )
        matched_calls = np.asarray(exact["selected_calls"], dtype=np.int64)
        matched = selected_loss(eval_losses, matched_calls)
        white_matched = selected_loss(eval_white_losses, matched_calls)
        analytic = policy.optimal_expected_mixture(train_losses, SUPPORTED_CALLS, float(calls.mean()))
        expected = np.einsum("nd,d->n", eval_losses, analytic["probabilities"], optimize=False)
        white_expected = np.einsum("nd,d->n", eval_white_losses, analytic["probabilities"], optimize=False)
        histogram_calls = policy.randomized_histogram_control(calls, int(seed) + 2003 * budget_index)
        histogram = selected_loss(eval_losses, histogram_calls)
        white_histogram = selected_loss(eval_white_losses, histogram_calls)
        permuted_scores = policy.permute_critic_scores(eval_scores, int(seed) + 3001 * budget_index, episode_ids=eval_episodes)
        permuted_calls = sequential_calls(permuted_scores, price)
        permuted = selected_loss(eval_losses, permuted_calls)
        permuted_baseline = policy.strongest_transition_independent_baseline(
            train_losses,
            SUPPORTED_CALLS,
            n=len(permuted_calls),
            target_total_calls=int(permuted_calls.sum()),
            seed=int(seed) + 4001 * budget_index,
        )
        permuted_matched = selected_loss(eval_losses, permuted_baseline["selected_calls"])
        oracle_calibration = calibrate_price(gains_from_losses(eval_losses), float(calls.mean()))
        oracle_calls = sequential_calls(gains_from_losses(eval_losses), float(oracle_calibration["compute_price"]))
        oracle = selected_loss(eval_losses, oracle_calls)
        result[f"b{float(budget):.2f}"] = {
            "target_mean_calls": float(budget),
            "price": price,
            "price_fit_mean_calls": float(calibration["realized_mean_calls"]),
            "calls": calls,
            "matched_calls": matched_calls,
            "histogram_calls": histogram_calls,
            "permuted_calls": permuted_calls,
            "oracle_calls": oracle_calls,
            "adaptive_loss": adaptive,
            "matched_loss": matched,
            "analytic_loss": expected,
            "histogram_loss": histogram,
            "permuted_loss": permuted,
            "permuted_matched_loss": permuted_matched,
            "oracle_loss": oracle,
            "white_adaptive_loss": white_adaptive,
            "white_matched_loss": white_matched,
            "white_analytic_loss": white_expected,
            "white_histogram_loss": white_histogram,
            "episodes": np.asarray(eval_episodes, dtype=np.int64),
            "exact_call_audit": exact["audit"],
            "analytic_probabilities": analytic["probabilities"],
        }
    return result


def _average_repeat_arrays(repeat_payloads: Sequence[Mapping[str, np.ndarray]], name: str) -> np.ndarray:
    return np.stack([np.asarray(payload[name], dtype=np.float64) for payload in repeat_payloads], axis=0).mean(0)


def aggregate_repeated_point(
    repeats: Sequence[Mapping[str, Any]],
    *,
    episode_ids: np.ndarray,
    fixed_losses: np.ndarray,
    fixed_white_losses: np.ndarray,
    family: str,
    feature_count: int,
    hidden_dims: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(repeats) < 2:
        raise ValueError("repeated CV aggregation requires at least two repeats")
    policy = prior_policy()
    adaptive = _average_repeat_arrays(repeats, "adaptive_loss")
    matched = _average_repeat_arrays(repeats, "matched_loss")
    analytic = _average_repeat_arrays(repeats, "analytic_loss")
    histogram = _average_repeat_arrays(repeats, "histogram_loss")
    permuted = _average_repeat_arrays(repeats, "permuted_loss")
    permuted_matched = _average_repeat_arrays(repeats, "permuted_matched_loss")
    oracle = _average_repeat_arrays(repeats, "oracle_loss")
    white_adaptive = _average_repeat_arrays(repeats, "white_adaptive_loss")
    white_matched = _average_repeat_arrays(repeats, "white_matched_loss")
    calls_per_repeat = [np.asarray(payload["calls"], dtype=np.int64) for payload in repeats]
    mean_calls = float(np.mean([calls.mean() for calls in calls_per_repeat]))
    mean_gate_decisions = float(np.mean([np.minimum(calls, 3).mean() for calls in calls_per_repeat]))
    gate = students.gate_cost(family, feature_count, hidden_dims)
    cfg = common.load_config()
    adaptive_total_flops = (
        float(cfg["base_predict_flops"])
        + float(cfg["v1_call_flops"])
        + (mean_calls - 1.0) * float(cfg["stage_adapter_flops"])
        + mean_gate_decisions * float(gate["total_incremental_gate_flops_per_evaluated_decision"])
    )
    baseline_total_flops = (
        float(cfg["base_predict_flops"])
        + float(cfg["v1_call_flops"])
        + (mean_calls - 1.0) * float(cfg["stage_adapter_flops"])
    )
    def ci(candidate: np.ndarray, baseline: np.ndarray, offset: int) -> dict[str, Any]:
        return policy.clustered_paired_loss_ci(
            candidate, baseline, episode_ids,
            samples=int(bootstrap_samples), seed=int(bootstrap_seed) + offset,
        )
    repeat_benefits = [float((np.asarray(item["matched_loss"]) - np.asarray(item["adaptive_loss"])).mean()) for item in repeats]
    return {
        "mean_calls": mean_calls,
        "total_calls_by_repeat": [int(calls.sum()) for calls in calls_per_repeat],
        "exact_matched_calls_by_repeat": [bool(item["exact_call_audit"]["exact_total_match"]) for item in repeats],
        "raw_mse": float(adaptive.mean()),
        "matched_raw_mse": float(matched.mean()),
        "analytic_raw_mse": float(analytic.mean()),
        "histogram_raw_mse": float(histogram.mean()),
        "fixed_d1_raw_mse": float(fixed_losses[:, 0].mean()),
        "oracle_raw_mse": float(oracle.mean()),
        "routing_regret_vs_causal_oracle": float((adaptive - oracle).mean()),
        "vs_matched_randomized": ci(adaptive, matched, 0),
        "vs_analytic_mixture": ci(adaptive, analytic, 11),
        "vs_fixed_d1": ci(adaptive, fixed_losses[:, 0], 17),
        "vs_histogram_null": ci(adaptive, histogram, 23),
        "score_permutation_vs_own_matched": ci(permuted, permuted_matched, 29),
        "whitened_mse": float(white_adaptive.mean()),
        "whitened_vs_matched": ci(white_adaptive, white_matched, 31),
        "whitened_vs_fixed_d1": ci(white_adaptive, fixed_white_losses[:, 0], 37),
        "repeat_matched_benefits": repeat_benefits,
        "repeat_direction_consistent": bool(all(value > 0 for value in repeat_benefits)),
        "gate_cost": gate,
        "mean_gate_decisions": mean_gate_decisions,
        "adaptive_total_flops_per_transition": adaptive_total_flops,
        "matched_total_flops_per_transition": baseline_total_flops,
    }


def ranking_diagnostics(scores: np.ndarray, realized_gains: np.ndarray) -> list[dict[str, Any]]:
    policy = prior_policy()
    return [
        {"decision_depth": depth + 1, **policy.gain_ranking_calibration(scores[:, depth], realized_gains[:, depth], bins=5)}
        for depth in range(3)
    ]


def mark_global_frontier(families: Mapping[str, Any], fixed_raw: Sequence[float], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for depth, raw in enumerate(fixed_raw, start=1):
        total = float(config["base_predict_flops"] + config["v1_call_flops"] + (depth - 1) * config["stage_adapter_flops"])
        rows.append({"kind": "fixed", "name": f"d{depth}", "mean_calls": float(depth), "raw_mse": float(raw), "total_flops": total})
    for family, payload in families.items():
        for key, point in payload["operating_points"].items():
            rows.append({"kind": "adaptive", "name": f"{family}:{key}", "mean_calls": point["mean_calls"], "raw_mse": point["raw_mse"], "total_flops": point["adaptive_total_flops_per_transition"]})
            rows.append({"kind": "matched", "name": f"{family}:{key}:matched", "mean_calls": point["mean_calls"], "raw_mse": point["matched_raw_mse"], "total_flops": point["matched_total_flops_per_transition"]})
    call_mask = prior_policy().nondominated_mask([row["mean_calls"] for row in rows], [row["raw_mse"] for row in rows])
    flop_mask = prior_policy().nondominated_mask([row["total_flops"] for row in rows], [row["raw_mse"] for row in rows])
    for row, call_ok, flop_ok in zip(rows, call_mask, flop_mask, strict=True):
        row["call_nondominated"] = bool(call_ok)
        row["flop_nondominated"] = bool(flop_ok)
        if row["kind"] == "adaptive":
            family, key = row["name"].split(":", 1)
            point = families[family]["operating_points"][key]
            point["call_nondominated"] = bool(call_ok)
            point["flop_nondominated"] = bool(flop_ok)
    return rows


def discovery_point_passes(point: Mapping[str, Any]) -> bool:
    return bool(
        all(point[name]["ci_low"] > 0 for name in (
            "vs_matched_randomized", "vs_analytic_mixture", "vs_fixed_d1", "vs_histogram_null"
        ))
        and point["whitened_vs_matched"]["mean_benefit"] > 0
        and point.get("call_nondominated", False)
        and point.get("flop_nondominated", False)
        and point["repeat_direction_consistent"]
        and all(point["exact_matched_calls_by_repeat"])
        and point["gate_cost"]["within_budget"]
    )
