#!/usr/bin/env python3
"""Frozen V5 causal feature graph and explicit operation accounting.

The arithmetic in :func:`build_counted_causal_features` is a source-level
carry-forward of V5 package v004.  The domain-robust study changes only the
number of affine gate heads; it does not change, normalize, or augment the
1,046 causal features.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


LATENT_DIM = 192
ACTION_DIM = 25
HISTORY_LEN = 3
FEATURE_DIM = 1046
VALID_HEAD_COUNTS = frozenset({2, 8})


def semantic_feature_names(
    *, latent_dim: int, action_dim: int, history_len: int
) -> tuple[str, ...]:
    """Return the frozen semantic order of every causal feature."""

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


FROZEN_FEATURE_NAMES = semantic_feature_names(
    latent_dim=LATENT_DIM,
    action_dim=ACTION_DIM,
    history_len=HISTORY_LEN,
)
if len(FROZEN_FEATURE_NAMES) != FEATURE_DIM:
    raise RuntimeError("frozen causal feature width drift")


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
    if len({value.device for value in inputs}) != 1 or len(
        {value.dtype for value in inputs}
    ) != 1:
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
    """Build the frozen V5 features with each shared norm evaluated once.

    Inputs are detached before arithmetic.  There is deliberately no target,
    contact, state, motion, phase, reward, success, episode, or batch-reduction
    argument.  The four tensors are the complete gate-information boundary.
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

    # This block intentionally mirrors V5 v004 exactly.  In particular, the
    # norms are common subexpressions reused by ratios and cosines.
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
        latent_dim=latent_dim,
        action_dim=action_dim,
        history_len=history_len,
    )
    if features.shape != (len(history), len(names)):
        raise RuntimeError("feature/name width mismatch")
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("causal features contain nonfinite values")
    return (features, names) if return_names else features


def assert_frozen_feature_inputs(
    history: Tensor,
    action_history: Tensor,
    current_prediction: Tensor,
    last_update: Tensor,
) -> None:
    """Fail closed unless inputs have the exact V5 gate dimensions."""

    history_len, latent_dim, action_dim = _validate(
        history, action_history, current_prediction, last_update
    )
    if (history_len, latent_dim, action_dim) != (
        HISTORY_LEN,
        LATENT_DIM,
        ACTION_DIM,
    ):
        raise RuntimeError(
            "frozen feature dimensions drift: "
            f"observed={(history_len, latent_dim, action_dim)}"
        )


def executable_operation_graph(
    *,
    latent_dim: int = LATENT_DIM,
    action_dim: int = ACTION_DIM,
    history_len: int = HISTORY_LEN,
    gate_width: int = FEATURE_DIM,
    head_count: int = 2,
) -> dict[str, Any]:
    """Return the primitive graph for features plus a min-ensemble score."""

    if head_count not in VALID_HEAD_COUNTS:
        raise ValueError("head_count must be the preregistered pooled 2 or envelope 8")
    changes = history_len - 1
    nodes: list[dict[str, Any]] = []

    def node(
        section: str,
        name: str,
        primitive: str,
        count: int,
        *,
        nonflop: bool = False,
    ) -> None:
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
        node(
            "feature",
            f"{name}_denominator_clamp",
            "comparison_min",
            1,
            nonflop=True,
        )
        node("feature", f"{name}_ratio", "division", 1)
    node("feature", "history_change_subtractions", "subtraction", changes * latent_dim)
    node("feature", "history_change_squares", "multiplication", changes * latent_dim)
    node(
        "feature",
        "history_change_reductions",
        "reduction_add",
        changes * (latent_dim - 1),
    )
    node("feature", "history_change_roots", "sqrt", changes)
    node("feature", "action_change_subtractions", "subtraction", changes * action_dim)
    node("feature", "action_change_squares", "multiplication", changes * action_dim)
    node(
        "feature",
        "action_change_reductions",
        "reduction_add",
        changes * (action_dim - 1),
    )
    node("feature", "action_change_roots", "sqrt", changes)
    for head_index in range(head_count):
        prefix = f"head_{head_index:02d}"
        node("gate_score", f"{prefix}_products", "affine_multiplication", gate_width)
        node(
            "gate_score",
            f"{prefix}_reduction",
            "affine_reduction_add",
            gate_width - 1,
        )
        node("gate_score", f"{prefix}_bias", "affine_bias_add", 1)
    node(
        "gate_score",
        "headwise_minimum_reduction",
        "comparison_min",
        head_count - 1,
        nonflop=True,
    )
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
                    latent_dim=latent_dim,
                    action_dim=action_dim,
                    history_len=history_len,
                )
            ),
            "stage_specific_gate_width": gate_width,
            "affine_head_count": head_count,
        },
        "zero_flop_operations": [
            "detach",
            "view",
            "flatten",
            "concatenate",
            "index",
            "scatter",
            "stage_coefficient_selection",
        ],
        "nodes": nodes,
        "totals": {
            "feature_flops_per_reached_evaluation": feature_flops,
            "affine_score_flops_per_head_per_reached_evaluation": (
                score_flops // head_count
            ),
            "all_affine_score_flops_per_reached_evaluation": score_flops,
            "total_flops_per_reached_evaluation": feature_flops + score_flops,
            "nonflop_operations_per_reached_evaluation": nonflops,
        },
    }
