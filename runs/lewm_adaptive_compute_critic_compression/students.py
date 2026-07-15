"""Detached cheap-student models, teacher cross-fitting, and gate accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import common


class SharedDepthRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        dimensions = [int(input_dim), *(int(value) for value in hidden_dims), 1]
        layers: list[nn.Module] = []
        for left, right in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(left, right), nn.GELU()))
        layers.append(nn.Linear(dimensions[-2], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


@dataclass
class FrozenStudent:
    family: str
    config: dict[str, Any]
    feature_indices: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_center: float
    target_scale: float
    input_dim: int
    hidden_dims: tuple[int, ...]
    state_dict: dict[str, torch.Tensor]
    seed: int
    training_rows: int

    def build(self, device: torch.device) -> SharedDepthRegressor:
        model = SharedDepthRegressor(self.input_dim, self.hidden_dims)
        model.load_state_dict(self.state_dict, strict=True)
        return model.to(device).eval().requires_grad_(False)

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenStudent":
        return cls(**dict(payload))


def _depth_encoded(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (3, 1046):
        raise ValueError("features must have shape [N,3,1046]")
    selected = values[:, :, indices]
    n, stages, width = selected.shape
    depth = np.broadcast_to(np.eye(stages, dtype=np.float32)[None], (n, stages, stages))
    return np.concatenate((selected, depth), axis=2).reshape(n * stages, width + stages)


def _flatten_targets(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("targets must have shape [N,3]")
    return array.reshape(-1)


def fit_student(
    features: np.ndarray,
    realized_gains: np.ndarray,
    teacher_scores: np.ndarray,
    *,
    family: str,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
    feature_selection: Sequence[int] | None = None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> FrozenStudent:
    """Fit from detached NumPy only; no solver object is accepted."""
    if torch.is_tensor(features) or torch.is_tensor(realized_gains) or torch.is_tensor(teacher_scores):
        raise TypeError("student training requires detached NumPy arrays")
    indices = common.feature_indices(family, feature_selection)
    x = _depth_encoded(features, indices)
    realized = _flatten_targets(realized_gains)
    teacher = _flatten_targets(teacher_scores)
    if len(x) != len(realized) or not np.isfinite(x).all() or not np.isfinite(realized).all() or not np.isfinite(teacher).all():
        raise ValueError("invalid/nonfinite student training arrays")
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(0, dtype=np.float64).astype(np.float32), 1e-5)
    objective = str(config.get("objective", "realized"))
    teacher_weight = float(config.get("teacher_weight", 0.0)) if objective != "realized" else 0.0
    target = (1.0 - teacher_weight) * realized + teacher_weight * teacher
    center = float(np.median(target))
    scale = float(max(np.median(np.abs(target - center)) * 1.4826, 1e-6))
    tx = torch.from_numpy((x - mean) / std)
    ty = torch.from_numpy((target - center) / scale)
    anchor = torch.from_numpy((realized - center) / scale)
    hidden_dims = tuple(int(value) for value in config.get("hidden_dims", ()))
    torch.manual_seed(int(seed))
    model = SharedDepthRegressor(x.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(config.get("weight_decay", 0.0))
    )
    rng = np.random.default_rng(int(seed))
    rank_weight = float(config.get("rank_weight", 0.0)) if objective == "distilled_rank" else 0.0
    for _ in range(int(epochs)):
        order = rng.permutation(len(x))
        model.train()
        for start in range(0, len(order), int(batch_size)):
            index = order[start : start + int(batch_size)]
            bx = tx[index].to(device)
            by = ty[index].to(device)
            ba = anchor[index].to(device)
            prediction = model(bx)
            regression = F.smooth_l1_loss(prediction, by, beta=0.5)
            # Realized gain always anchors distillation, even at high teacher weight.
            loss = regression + 0.25 * F.smooth_l1_loss(prediction, ba, beta=0.5)
            if rank_weight > 0 and len(index) > 1:
                perm = torch.roll(torch.arange(len(index), device=device), shifts=1)
                desired = torch.sign(by - by[perm])
                informative = desired != 0
                if bool(informative.any()):
                    margin = desired[informative] * (prediction[informative] - prediction[perm][informative])
                    importance = (by[informative] - by[perm][informative]).abs().clamp_max(5.0)
                    loss = loss + rank_weight * (F.softplus(-margin) * importance).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite student loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return FrozenStudent(
        family=family,
        config=dict(config),
        feature_indices=indices.copy(),
        feature_mean=mean,
        feature_std=std,
        target_center=center,
        target_scale=scale,
        input_dim=x.shape[1],
        hidden_dims=hidden_dims,
        state_dict=state,
        seed=int(seed),
        training_rows=int(len(x)),
    )


def predict_student(
    student: FrozenStudent,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    x = _depth_encoded(features, student.feature_indices)
    model = student.build(device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), int(batch_size)):
            batch = torch.from_numpy((x[start : start + batch_size] - student.feature_mean) / student.feature_std).to(device)
            chunks.append(model(batch).cpu().numpy())
    values = np.concatenate(chunks).astype(np.float64) * student.target_scale + student.target_center
    return values.reshape(len(features), 3)


def fit_teacher(
    features: np.ndarray,
    realized_gains: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> FrozenStudent:
    config = {"hidden_dims": list(hidden_dims), "weight_decay": 1e-4, "objective": "realized"}
    return fit_student(
        features,
        realized_gains,
        realized_gains,
        family="full_linear",
        config=config,
        seed=seed,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def crossfit_teacher(
    features: np.ndarray,
    realized_gains: np.ndarray,
    episodes: np.ndarray,
    *,
    folds: int,
    seed: int,
    teacher_seeds: Sequence[int],
    device: torch.device,
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    assignment = common.episode_assignment(episodes, folds, seed)
    oof = np.full_like(realized_gains, np.nan, dtype=np.float64)
    provenance = []
    for fold in range(int(folds)):
        train = assignment != fold
        held = ~train
        if set(np.unique(episodes[train])) & set(np.unique(episodes[held])):
            raise RuntimeError("teacher cross-fit episode leakage")
        model = fit_teacher(
            features[train], realized_gains[train],
            seed=int(teacher_seeds[fold % len(teacher_seeds)]) + fold * 1009,
            device=device, hidden_dims=hidden_dims, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate,
        )
        oof[held] = predict_student(model, features[held], device)
        provenance.append({
            "fold": fold,
            "train_episodes": np.unique(episodes[train]),
            "held_episodes": np.unique(episodes[held]),
            "held_rows": int(held.sum()),
        })
    if not np.isfinite(oof).all():
        raise RuntimeError("cross-fitted teacher left missing labels")
    return oof, provenance


def stable_feature_selection(
    features: np.ndarray,
    target: np.ndarray,
    episodes: np.ndarray,
    *,
    k: int,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fold-local stability ranking over raw causal coordinates only."""
    if not 1 <= int(k) <= 1035:
        raise ValueError("sparse k must select only raw coordinates")
    x = np.asarray(features[:, :, :1035], dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    assignment = common.episode_assignment(episodes, folds, seed)
    selected_per_fold = []
    correlations = []
    for fold in range(int(folds)):
        train = assignment != fold
        flat_x = x[train].reshape(-1, x.shape[2])
        flat_y = y[train].reshape(-1)
        centered_y = flat_y - flat_y.mean()
        centered_x = flat_x - flat_x.mean(0)
        x_energy = np.einsum("nf,nf->f", centered_x, centered_x, optimize=False)
        y_energy = float(np.einsum("n,n->", centered_y, centered_y, optimize=False))
        denominator = np.sqrt(x_energy * y_energy).clip(min=1e-12)
        numerator = np.einsum("nf,n->f", centered_x, centered_y, optimize=False)
        corr = np.abs(numerator / denominator)
        corr[~np.isfinite(corr)] = 0.0
        top = np.argsort(-corr, kind="stable")[: int(k)]
        selected_per_fold.append(top)
        correlations.append(corr)
    frequency = np.zeros(1035, dtype=np.float64)
    for chosen in selected_per_fold:
        frequency[chosen] += 1.0
    frequency /= folds
    mean_corr = np.mean(correlations, axis=0)
    # Frequency is primary; attribution magnitude breaks stability ties.
    order = np.lexsort((np.arange(1035), -mean_corr, -frequency))
    selected = np.sort(order[: int(k)]).astype(np.int64)
    return selected, {
        "eligible_raw_features": 1035,
        "selected": selected,
        "selection_frequency": frequency[selected],
        "mean_abs_correlation": mean_corr[selected],
        "fold_selected": selected_per_fold,
        "fit_episode_count": int(len(np.unique(episodes))),
        "fold_local": True,
    }


def model_parameter_count(input_dim: int, hidden_dims: Sequence[int]) -> int:
    dims = [int(input_dim), *(int(v) for v in hidden_dims), 1]
    return int(sum(left * right + right for left, right in zip(dims[:-1], dims[1:])))


def gate_cost(family: str, feature_count: int, hidden_dims: Sequence[int]) -> dict[str, Any]:
    """Conservative max per evaluated decision, including d1 update creation."""
    feature_count = int(feature_count)
    depth_width = 3
    input_dim = feature_count + depth_width
    # d1 update = current - d0. Later adapters already expose their update.
    update_arithmetic = 192
    # Exact arithmetic count for the 11 summary tail in the immutable builder:
    # four norms, two cosines, gap, two history deltas, two action deltas, ratio.
    summary_arithmetic = 3806 if family in ("full_linear", "distilled_rank_linear", "low_rank", "compact11_control") else 0
    normalization = 2 * input_dim
    dims = [input_dim, *(int(value) for value in hidden_dims), 1]
    layers = 0
    activations = 0
    for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
        layers += 2 * left * right + right  # multiply/add pairs plus bias adds
        if index < len(dims) - 2:
            activations += 10 * right  # explicit GELU arithmetic estimate
    control_flow = 1
    total = update_arithmetic + summary_arithmetic + normalization + layers + activations + control_flow
    parameters = model_parameter_count(input_dim, hidden_dims)
    return {
        "solver_flops": 0,
        "feature_construction_flops": update_arithmetic + summary_arithmetic,
        "normalization_flops": normalization,
        "student_linear_flops": layers,
        "activation_flops_estimate": activations,
        "control_flow_flops": control_flow,
        "total_incremental_gate_flops_per_evaluated_decision": total,
        "hard_budget": 66240,
        "within_budget": total <= 66240,
        "parameters": parameters,
        "parameter_bytes_float32": 4 * parameters,
        "feature_bytes_float32": 4 * input_dim,
        "input_dim_with_depth_encoding": input_dim,
        "notes": "Arithmetic FLOPs are not latency or energy; GELU is a conservative 10-operation estimate.",
    }


def audit_candidate_costs(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for family, candidates in config["families"].items():
        rows = []
        for candidate in candidates:
            count = int(candidate.get("sparse_k", len(common.feature_indices(family)) if family != "stable_sparse" else 1))
            cost = gate_cost(family, count, candidate.get("hidden_dims", ()))
            rows.append({"config": dict(candidate), "cost": cost})
            if not cost["within_budget"]:
                raise RuntimeError(f"candidate exceeds hard gate budget: {family} {candidate}")
        result[family] = rows
    return result
