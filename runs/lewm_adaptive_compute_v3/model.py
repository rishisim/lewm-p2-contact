"""Joint shared refiner and strictly local causal halting gate for V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MAX_DEPTH = 4
LATENT_DIM = 192
HISTORY_LEN = 3
ACTION_DIM = 25
V1_REFINER_SHA256 = "388a82fc30c96921083bfa4f2578cb296e6c3544510953d17578cf766cdf10f1"


@dataclass(frozen=True)
class CallStats:
    processed_rows: int
    block_invocations: int
    gate_rows: int = 0


class SharedResidualRefiner(nn.Module):
    """V1-compatible recurrent residual block, trainable in V3."""

    def __init__(self, latent_dim=192, action_dim=25, history_len=3,
                 hidden_dim=256, iteration_dim=16) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)
        self.hidden_dim = int(hidden_dim)
        self.iteration_dim = int(iteration_dim)
        self.max_depth = MAX_DEPTH
        self.iteration_embedding = nn.Embedding(MAX_DEPTH, iteration_dim)
        input_dim = history_len * latent_dim + history_len * action_dim + latent_dim + iteration_dim
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _validate(self, history: Tensor, action: Tensor, base: Tensor) -> None:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.history_len, self.latent_dim):
            raise ValueError("invalid latent history shape")
        if action.ndim != 3 or tuple(action.shape[1:]) != (self.history_len, self.action_dim):
            raise ValueError("invalid action history shape")
        if base.ndim != 2 or tuple(base.shape) != (len(history), self.latent_dim):
            raise ValueError("invalid base prediction shape")
        if not (history.device == action.device == base.device):
            raise ValueError("refiner inputs must share a device")

    def update(self, history: Tensor, action: Tensor, current: Tensor, depth: int) -> Tensor:
        if depth < 0 or depth >= MAX_DEPTH:
            raise ValueError("call index outside 0..3")
        iteration = self.iteration_embedding(torch.full(
            (len(current),), depth, dtype=torch.long, device=current.device
        ))
        values = torch.cat((history.flatten(1), action.flatten(1), current, iteration), dim=1)
        return self.block(values)

    def forward_all(self, history: Tensor, action: Tensor, base: Tensor,
                    max_depth: int = MAX_DEPTH) -> tuple[list[Tensor], list[Tensor], CallStats]:
        self._validate(history, action, base)
        if max_depth < 1 or max_depth > MAX_DEPTH:
            raise ValueError("max_depth must be in 1..4")
        exits = [base]
        updates = []
        current = base
        for depth in range(max_depth):
            delta = self.update(history, action, current, depth)
            current = current + delta
            updates.append(delta)
            exits.append(current)
        return exits, updates, CallStats(len(base) * max_depth, max_depth)

    def forward_selected(self, history: Tensor, action: Tensor, base: Tensor,
                         depths: Tensor) -> tuple[Tensor, CallStats]:
        self._validate(history, action, base)
        depths = torch.as_tensor(depths, dtype=torch.long, device=base.device)
        if depths.shape != (len(base),) or bool(((depths < 0) | (depths > MAX_DEPTH)).any()):
            raise ValueError("selected depths must have shape [batch] and lie in 0..4")
        current = base
        invocations = 0
        for depth in range(MAX_DEPTH):
            active = torch.nonzero(depths > depth, as_tuple=False).flatten()
            if not len(active):
                break
            delta = self.update(history[active], action[active], current[active], depth)
            current = current.index_copy(0, active, current[active] + delta)
            invocations += 1
        return current, CallStats(int(depths.sum()), invocations)


def causal_feature_names() -> tuple[str, ...]:
    names = []
    names += [f"history_{t}_{d}" for t in range(HISTORY_LEN) for d in range(LATENT_DIM)]
    names += [f"action_{t}_{d}" for t in range(HISTORY_LEN) for d in range(ACTION_DIM)]
    names += [f"current_{d}" for d in range(LATENT_DIM)]
    names += [f"last_update_{d}" for d in range(LATENT_DIM)]
    names += [
        "current_norm", "update_norm", "relative_update_norm", "last_history_norm",
        "distance_to_last_history", "update_current_cosine", "update_history_gap_cosine",
        "history_delta_01_norm", "history_delta_12_norm", "action_delta_01_norm",
        "action_delta_12_norm",
    ]
    return tuple(names)


def build_causal_features(history: Tensor, action: Tensor, current: Tensor,
                          last_update: Tensor) -> Tensor:
    """Target-free, row-local gate features. No batch reductions are used."""
    if history.ndim != 3 or action.ndim != 3 or current.ndim != 2 or last_update.shape != current.shape:
        raise ValueError("invalid causal feature inputs")
    if len(history) != len(action) or len(history) != len(current):
        raise ValueError("causal feature batch sizes differ")
    eps = 1e-6
    gap = current - history[:, -1]
    norm = lambda x: torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    cosine = lambda x, y: (x * y).sum(-1, keepdim=True) / (norm(x) * norm(y)).clamp_min(eps)
    summary = torch.cat((
        norm(current), norm(last_update), norm(last_update) / norm(current).clamp_min(eps),
        norm(history[:, -1]), norm(gap), cosine(last_update, current), cosine(last_update, gap),
        norm(history[:, 1] - history[:, 0]), norm(history[:, 2] - history[:, 1]),
        norm(action[:, 1] - action[:, 0]), norm(action[:, 2] - action[:, 1]),
    ), dim=1)
    result = torch.cat((history.flatten(1), action.flatten(1), current, last_update, summary), dim=1)
    if result.shape[1] != len(causal_feature_names()) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("invalid causal feature result")
    return result


class LocalGainGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), iteration_dim=8) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.iteration_embedding = nn.Embedding(3, int(iteration_dim))
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.register_buffer("gain_scale", torch.tensor(1e-4))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + iteration_dim, hidden_dims[0]), nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.GELU(),
            nn.Linear(hidden_dims[1], 1),
        )

    @torch.no_grad()
    def fit_normalization(self, features: Tensor, gains: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("normalization feature shape mismatch")
        self.feature_mean.copy_(features.mean(0))
        self.feature_std.copy_(features.std(0, unbiased=False).clamp_min(1e-6))
        centered = gains.flatten() - gains.flatten().median()
        scale = centered.abs().median().mul(1.4826).clamp_min(1e-6)
        self.gain_scale.copy_(scale)

    def forward(self, features: Tensor, decision_depth: int) -> Tensor:
        if decision_depth not in (1, 2, 3):
            raise ValueError("decision depth must be 1, 2, or 3")
        standardized = (features - self.feature_mean) / self.feature_std
        emb = self.iteration_embedding(torch.full(
            (len(features),), decision_depth - 1, dtype=torch.long, device=features.device
        ))
        return self.mlp(torch.cat((standardized, emb), dim=1)).squeeze(1) * self.gain_scale


class AdaptiveLeWM(nn.Module):
    def __init__(self, refiner: SharedResidualRefiner, gate: LocalGainGate) -> None:
        super().__init__()
        self.refiner = refiner
        self.gate = gate

    def dense(self, history: Tensor, action: Tensor, base: Tensor,
              max_depth: int = MAX_DEPTH) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        exits, updates, _ = self.refiner.forward_all(history, action, base, max_depth)
        scores = []
        for depth in range(1, max_depth):
            scores.append(self.gate(build_causal_features(history, action, exits[depth], updates[depth - 1]), depth))
        return exits, updates, scores


def joint_loss(model: AdaptiveLeWM, history: Tensor, action: Tensor, base: Tensor,
               target: Tensor, max_depth: int, cfg: Mapping[str, object]) -> tuple[Tensor, dict[str, float]]:
    exits, updates, scores = model.dense(history, action, base, max_depth)
    losses = torch.stack([F.mse_loss(value, target, reduction="none").mean(1) for value in exits[1:]], 1)
    deep = losses.mean()
    monotonic = torch.relu(losses[:, 1:] - losses[:, :-1]).mean() if max_depth > 1 else deep.new_zeros(())
    update = torch.stack([value.square().mean() for value in updates]).mean()
    gate_reg = deep.new_zeros(())
    gate_price = deep.new_zeros(())
    gate_rank = deep.new_zeros(())
    if scores:
        actual = losses[:, :-1] - losses[:, 1:]
        predicted = torch.stack(scores, 1)
        scale = model.gate.gain_scale.clamp_min(1e-8)
        gate_reg = F.smooth_l1_loss(predicted / scale, actual.detach() / scale,
                                    beta=float(cfg["gate_smooth_l1_beta"]))
        temperature = float(cfg["gate_temperature"]) * scale
        price_terms = []
        for price in cfg["training_prices"]:
            p = float(price)
            logits = (predicted - p) / temperature
            labels = (actual.detach() > p).to(predicted.dtype)
            price_terms.append(F.binary_cross_entropy_with_logits(logits, labels))
        gate_price = torch.stack(price_terms).mean()
        if len(predicted) > 1:
            rolled = torch.roll(torch.arange(len(predicted), device=predicted.device), 1)
            diff_y = actual.detach() - actual.detach()[rolled]
            diff_p = predicted - predicted[rolled]
            mask = diff_y.abs() > 1e-12
            if bool(mask.any()):
                gate_rank = F.softplus(-diff_y[mask].sign() * diff_p[mask] / scale).mean()
    total = (
        float(cfg["deep_supervision_weight"]) * deep
        + float(cfg["monotonic_penalty_weight"]) * monotonic
        + float(cfg["update_penalty_weight"]) * update
        + float(cfg["gate_regression_weight"]) * gate_reg
        + float(cfg["gate_price_weight"]) * gate_price
        + float(cfg["gate_ranking_weight"]) * gate_rank
    )
    parts = {k: float(v.detach()) for k, v in {
        "total": total, "deep": deep, "monotonic": monotonic, "update": update,
        "gate_regression": gate_reg, "gate_price": gate_price, "gate_ranking": gate_rank,
    }.items()}
    return total, parts


def refiner_flops_per_call(refiner: SharedResidualRefiner) -> int:
    return int(sum(2 * m.in_features * m.out_features for m in refiner.block if isinstance(m, nn.Linear)))


def gate_flops_per_decision(gate: LocalGainGate) -> int:
    return int(sum(2 * m.in_features * m.out_features for m in gate.mlp if isinstance(m, nn.Linear)))
