"""Independent cross-fitted local marginal-gain critics.

All solver inputs must be detached before reaching this module. The public
training API accepts NumPy arrays, making a critic-to-solver autograd path
structurally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GainMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        dims = [int(input_dim), *(int(v) for v in hidden_dims), 1]
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-2], dims[1:-1]):
            layers.extend((nn.Linear(left, right), nn.GELU()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(1)


@dataclass
class FrozenCritic:
    state_dict: dict[str, torch.Tensor]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    gain_center: float
    gain_scale: float
    input_dim: int
    hidden_dims: tuple[int, ...]
    fold: int
    seed: int

    def build(self, device: torch.device) -> GainMLP:
        model = GainMLP(self.input_dim, self.hidden_dims)
        model.load_state_dict(self.state_dict, strict=True)
        return model.to(device).eval().requires_grad_(False)


def _episode_folds(episodes: np.ndarray, folds: int, seed: int) -> np.ndarray:
    episodes = np.asarray(episodes, dtype=np.int64)
    unique = np.unique(episodes)
    if folds < 2 or folds > len(unique):
        raise ValueError("invalid number of episode folds")
    shuffled = np.random.default_rng(seed).permutation(unique)
    mapping = {int(ep): int(i % folds) for i, ep in enumerate(shuffled)}
    return np.asarray([mapping[int(ep)] for ep in episodes], dtype=np.int64)


def _fit_one(
    features: np.ndarray,
    gains: np.ndarray,
    *,
    hidden_dims: Sequence[int],
    seed: int,
    fold: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> FrozenCritic:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(gains, dtype=np.float32).reshape(-1)
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = x.std(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    center = float(np.median(y))
    scale = float(max(np.median(np.abs(y - center)) * 1.4826, 1e-6))
    tx = torch.from_numpy((x - mean) / std)
    ty = torch.from_numpy((y - center) / scale)
    positive = y > 0
    pos_weight = min(5.0, float((~positive).sum() / max(positive.sum(), 1)))
    weights = torch.from_numpy(np.where(positive, max(1.0, pos_weight), 1.0).astype(np.float32))
    torch.manual_seed(seed)
    model = GainMLP(x.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    for _ in range(int(epochs)):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), int(batch_size)):
            index = order[start : start + int(batch_size)]
            bx = tx[index].to(device)
            by = ty[index].to(device)
            bw = weights[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (F.smooth_l1_loss(model(bx), by, reduction="none", beta=0.5) * bw).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite critic loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return FrozenCritic(state, mean, std, center, scale, x.shape[1], tuple(hidden_dims), fold, seed)


def predict_critic(critic: FrozenCritic, features: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    model = critic.build(device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy((x[start : start + batch_size] - critic.feature_mean) / critic.feature_std).to(device)
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64) * critic.gain_scale + critic.gain_center


def fit_cross_fitted_critics(
    features: np.ndarray,
    gains: np.ndarray,
    episodes: np.ndarray,
    *,
    folds: int,
    seeds: Sequence[int],
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[list[FrozenCritic], np.ndarray]:
    """Fit one held-episode-fold model per fold and return strict OOF scores."""
    if torch.is_tensor(features) or torch.is_tensor(gains):
        raise TypeError("critic training requires detached NumPy arrays")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(gains, dtype=np.float32).reshape(-1)
    ep = np.asarray(episodes, dtype=np.int64).reshape(-1)
    if len(x) != len(y) or len(y) != len(ep) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid critic arrays")
    assignment = _episode_folds(ep, int(folds), int(seeds[0]))
    models: list[FrozenCritic] = []
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(int(folds)):
        train = assignment != fold
        held = ~train
        critic = _fit_one(
            x[train], y[train], hidden_dims=hidden_dims,
            seed=int(seeds[fold % len(seeds)]) + fold * 1009, fold=fold,
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, device=device,
        )
        models.append(critic)
        oof[held] = predict_critic(critic, x[held], device)
    if not np.isfinite(oof).all():
        raise RuntimeError("cross-fitting left missing predictions")
    return models, oof


def ensemble_scores(models: Sequence[FrozenCritic], features: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.stack([predict_critic(model, features, device) for model in models], axis=1)
    return predictions.mean(1), predictions.std(1, ddof=0)


def permute_gain_by_episode(gains: np.ndarray, episodes: np.ndarray, seed: int) -> np.ndarray:
    """Label-free null: shuffle complete equal-length episode gain blocks."""
    gains = np.asarray(gains)
    episodes = np.asarray(episodes)
    unique = np.unique(episodes)
    groups = [np.flatnonzero(episodes == ep) for ep in unique]
    lengths = {len(group) for group in groups}
    if len(lengths) != 1:
        raise ValueError("episode-block permutation requires equal group sizes")
    shuffled = np.random.default_rng(seed).permutation(len(groups))
    result = np.empty_like(gains)
    for destination, source in enumerate(shuffled):
        result[groups[destination]] = gains[groups[source]]
    return result


def serialize_critics(by_depth: Mapping[int, Sequence[FrozenCritic]]) -> dict[int, list[dict[str, object]]]:
    return {int(depth): [critic.__dict__ for critic in critics] for depth, critics in by_depth.items()}


def deserialize_critics(payload: Mapping[int, Sequence[Mapping[str, object]]]) -> dict[int, list[FrozenCritic]]:
    return {int(depth): [FrozenCritic(**dict(item)) for item in values] for depth, values in payload.items()}
