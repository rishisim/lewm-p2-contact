#!/usr/bin/env python3
"""Executable causal feature graph with explicit common-subexpression reuse."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def semantic_feature_names(
    *, latent_dim: int, action_dim: int, history_len: int
) -> tuple[str, ...]:
    names = [
        f"history_latent_t{time}_d{dimension}"
        for time in range(history_len)
        for dimension in range(latent_dim)
    ]
    names += [
        f"normalized_action_t{time}_d{dimension}"
        for time in range(history_len)
        for dimension in range(action_dim)
    ]
    names += [f"current_prediction_d{dimension}" for dimension in range(latent_dim)]
    names += [f"last_update_d{dimension}" for dimension in range(latent_dim)]
    names += [
        "current_norm",
        "last_update_norm",
        "relative_update_norm",
        "last_history_norm",
        "current_history_distance",
        "update_current_cosine",
        "update_history_distance_cosine",
    ]
    names += [
        f"history_change_t{time}_t{time + 1}_norm"
        for time in range(history_len - 1)
    ]
    names += [
        f"action_change_t{time}_t{time + 1}_norm"
        for time in range(history_len - 1)
    ]
    if len(set(names)) != len(names):
        raise RuntimeError("semantic feature names are not unique")
    return tuple(names)


def _validate(
    history: Tensor,
    action_history: Tensor,
    current_prediction: Tensor,
    last_update: Tensor,
) -> tuple[int, int, int]:
    if history.ndim != 3 or action_history.ndim != 3:
        raise ValueError("history and actions must have rank three")
    if current_prediction.ndim != 2 or last_update.ndim != 2:
        raise ValueError("prediction and update must have rank two")
    batch, history_len, latent_dim = history.shape
    if batch <= 0 or history_len <= 0 or latent_dim <= 0:
        raise ValueError("feature dimensions must be positive")
    if action_history.shape[:2] != (batch, history_len):
        raise ValueError("action history shape mismatch")
    if current_prediction.shape != (batch, latent_dim):
        raise ValueError("current prediction shape mismatch")
    if last_update.shape != current_prediction.shape:
        raise ValueError("last update shape mismatch")
    inputs = (history, action_history, current_prediction, last_update)
    if len({value.device for value in inputs}) != 1 or len({value.dtype for value in inputs}) != 1:
        raise ValueError("feature inputs must share device and dtype")
    if not all(torch.is_floating_point(value) for value in inputs):
        raise ValueError("feature inputs must be floating point")
    return history_len, latent_dim, action_history.shape[2]


def build_counted_causal_features(
    history: Tensor,
    action_history: Tensor,
    current_prediction: Tensor,
    last_update: Tensor,
    *,
    return_names: bool = False,
) -> Tensor | tuple[Tensor, tuple[str, ...]]:
    """Build the frozen 1,046 features with every shared norm evaluated once.

    Inputs are detached before all arithmetic.  No future, target, contact,
    episode, batch-reduction, or privileged argument exists.
    """
    history_len, latent_dim, action_dim = _validate(
        history, action_history, current_prediction, last_update
    )
    history = history.detach()
    action_history = action_history.detach()
    current_prediction = current_prediction.detach()
    last_update = last_update.detach()
    epsilon = torch.finfo(current_prediction.dtype).eps

    last_history = history[:, -1]
    gap = current_prediction - last_history

    # Explicit common subexpressions: each norm appears exactly once in the
    # executable graph and is reused by ratios/cosines below.
    current_norm = torch.linalg.vector_norm(current_prediction, dim=-1, keepdim=True)
    update_norm = torch.linalg.vector_norm(last_update, dim=-1, keepdim=True)
    last_history_norm = torch.linalg.vector_norm(last_history, dim=-1, keepdim=True)
    gap_norm = torch.linalg.vector_norm(gap, dim=-1, keepdim=True)
    relative_update = update_norm / current_norm.clamp_min(epsilon)

    update_current_dot = (last_update * current_prediction).sum(dim=-1, keepdim=True)
    update_current_denominator = (update_norm * current_norm).clamp_min(epsilon)
    update_current_cosine = update_current_dot / update_current_denominator

    update_gap_dot = (last_update * gap).sum(dim=-1, keepdim=True)
    update_gap_denominator = (update_norm * gap_norm).clamp_min(epsilon)
    update_gap_cosine = update_gap_dot / update_gap_denominator

    summaries = [
        current_norm,
        update_norm,
        relative_update,
        last_history_norm,
        gap_norm,
        update_current_cosine,
        update_gap_cosine,
    ]
    if history_len > 1:
        history_change = history[:, 1:] - history[:, :-1]
        action_change = action_history[:, 1:] - action_history[:, :-1]
        summaries.append(torch.linalg.vector_norm(history_change, dim=2))
        summaries.append(torch.linalg.vector_norm(action_change, dim=2))

    features = torch.cat(
        (
            history.flatten(1),
            action_history.flatten(1),
            current_prediction,
            last_update,
            *summaries,
        ),
        dim=1,
    )
    names = semantic_feature_names(
        latent_dim=latent_dim, action_dim=action_dim, history_len=history_len
    )
    if features.shape != (len(history), len(names)):
        raise RuntimeError("feature/name width mismatch")
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("causal features contain nonfinite values")
    return (features, names) if return_names else features


def executable_operation_graph(
    *, latent_dim: int = 192, action_dim: int = 25, history_len: int = 3,
    gate_width: int = 1049,
) -> dict[str, Any]:
    """Return the primitive graph corresponding to the code above."""
    changes = history_len - 1
    nodes: list[dict[str, Any]] = []

    def node(section: str, name: str, primitive: str, count: int, *, nonflop: bool = False) -> None:
        nodes.append(
            {
                "section": section,
                "name": name,
                "primitive": primitive,
                "count_per_reached_row": int(count),
                "accounting": "non_flop" if nonflop else "flop",
            }
        )

    node("feature", "current_minus_last_history", "subtraction", latent_dim)
    for name in ("current_norm", "update_norm", "last_history_norm", "gap_norm"):
        node("feature", f"{name}_squares", "multiplication", latent_dim)
        node("feature", f"{name}_reduction", "reduction_add", latent_dim - 1)
        node("feature", f"{name}_root", "sqrt", 1)
    node("feature", "relative_norm_clamp", "comparison_min", 1, nonflop=True)
    node("feature", "relative_norm_ratio", "division", 1)
    for name in ("update_current_cosine", "update_gap_cosine"):
        node("feature", f"{name}_dot_products", "multiplication", latent_dim)
        node("feature", f"{name}_dot_reduction", "reduction_add", latent_dim - 1)
        node("feature", f"{name}_norm_product", "multiplication", 1)
        node("feature", f"{name}_denominator_clamp", "comparison_min", 1, nonflop=True)
        node("feature", f"{name}_ratio", "division", 1)
    node("feature", "history_change_subtractions", "subtraction", changes * latent_dim)
    node("feature", "history_change_squares", "multiplication", changes * latent_dim)
    node("feature", "history_change_reductions", "reduction_add", changes * (latent_dim - 1))
    node("feature", "history_change_roots", "sqrt", changes)
    node("feature", "action_change_subtractions", "subtraction", changes * action_dim)
    node("feature", "action_change_squares", "multiplication", changes * action_dim)
    node("feature", "action_change_reductions", "reduction_add", changes * (action_dim - 1))
    node("feature", "action_change_roots", "sqrt", changes)
    for head in ("raw_head", "native_whitened_head"):
        node("gate_score", f"{head}_products", "affine_multiplication", gate_width)
        node("gate_score", f"{head}_reduction", "affine_reduction_add", gate_width - 1)
        node("gate_score", f"{head}_bias", "affine_bias_add", 1)
    node("gate_score", "dual_head_minimum", "comparison_min", 1, nonflop=True)
    node("gate_score", "continue_threshold", "comparison", 1, nonflop=True)

    feature_flops = sum(
        item["count_per_reached_row"]
        for item in nodes
        if item["section"] == "feature" and item["accounting"] == "flop"
    )
    score_flops = sum(
        item["count_per_reached_row"]
        for item in nodes
        if item["section"] == "gate_score" and item["accounting"] == "flop"
    )
    nonflops = sum(
        item["count_per_reached_row"]
        for item in nodes
        if item["accounting"] == "non_flop"
    )
    return {
        "schema_version": 1,
        "implementation": "build_counted_causal_features",
        "dimensions": {
            "latent": latent_dim,
            "action": action_dim,
            "history": history_len,
            "causal_features": len(
                semantic_feature_names(
                    latent_dim=latent_dim, action_dim=action_dim, history_len=history_len
                )
            ),
            "gate_width_including_depth_one_hot": gate_width,
        },
        "zero_flop_operations": ["detach", "view", "flatten", "concatenate", "index", "scatter", "one_hot_assignment"],
        "nodes": nodes,
        "totals": {
            "feature_flops_per_reached_evaluation": feature_flops,
            "dual_affine_score_flops_per_reached_evaluation": score_flops,
            "total_flops_per_reached_evaluation": feature_flops + score_flops,
            "nonflop_comparison_min_per_reached_evaluation": nonflops,
        },
    }

