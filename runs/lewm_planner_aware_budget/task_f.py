"""Task F candidate-local critic contract, metrics, and frozen fitting."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from task_e import ranking_metrics, selected_change

ALLOWED_FEATURES = frozenset({"native_latent_goal_cost", "candidate_actions_flat"})
FORBIDDEN_TOKENS = ("simulator", "state", "geometry", "contact", "future", "success", "role", "seed", "start", "rank", "bank")

def validate_feature_schema(features: list[str]) -> None:
    if set(features) != ALLOWED_FEATURES:
        raise ValueError("Task F feature allowlist violation")
    if any(token in name.lower() for name in features for token in FORBIDDEN_TOKENS):
        raise ValueError("forbidden inference feature")

def feature_matrix(native_cost: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    native_cost, candidates = np.asarray(native_cost, np.float64), np.asarray(candidates, np.float64)
    if native_cost.ndim != 1 or candidates.shape != (len(native_cost), 5, 10):
        raise ValueError("expected candidate-local [N] and [N,5,10] tensors")
    output = np.column_stack((native_cost, candidates.reshape(len(candidates), -1)))
    if not np.isfinite(output).all(): raise ValueError("non-finite inference feature")
    return output

@dataclass(frozen=True)
class LinearCritic:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    def score(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, float)
        if x.ndim != 2 or x.shape[1] != len(self.mean): raise ValueError("invalid feature shape")
        result = ((x - self.mean) / self.scale) @ self.weight + self.bias
        if not np.isfinite(result).all(): raise ValueError("non-finite critic score")
        return result
    def payload(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "weight": self.weight.tolist(), "bias": self.bias}
    def sha256(self) -> str:
        import json
        return hashlib.sha256((json.dumps(self.payload(),sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()

def fit_linear(banks: list[tuple[np.ndarray, np.ndarray]], *, top_tail: int, epochs: int, learning_rate: float, weight_decay: float, seed: int) -> LinearCritic:
    if not banks: raise ValueError("fit banks required")
    x = np.concatenate([b[0] for b in banks]); y = np.concatenate([b[1] for b in banks])
    mean, scale = x.mean(0), x.std(0)
    if not np.isfinite(scale).all() or np.any(scale == 0): raise ValueError("fit-only normalization is degenerate")
    torch.manual_seed(seed)
    model = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Each bank contributes equally; pairs are a loss construction, not an inferential unit.
    for _ in range(epochs):
        optimizer.zero_grad(); losses=[]
        for bx, by in banks:
            tx=torch.as_tensor((bx-mean)/scale,dtype=torch.float32); ty=torch.as_tensor(by,dtype=torch.float32)
            score=model(tx).squeeze(1); top=torch.argsort(ty,stable=True)[:top_tail]
            delta=score[top,None]-score[None,:]; target=(ty[top,None]-ty[None,:]).sign()
            mask=target.ne(0); losses.append(F.softplus(-target[mask]*delta[mask]).mean())
        torch.stack(losses).mean().backward(); optimizer.step()
    return LinearCritic(mean,scale,model.weight.detach().numpy()[0].astype(float),float(model.bias.detach()))

def bootstrap_start(values: np.ndarray, *, draws: int, seed: int) -> dict:
    values=np.asarray(values,float)
    if values.ndim != 1 or not np.isfinite(values).all(): raise ValueError("invalid cluster values")
    rng=np.random.default_rng(seed); sampled=values[rng.integers(0,len(values),(draws,len(values)))].mean(1)
    return {"estimate":float(values.mean()),"ci95":np.quantile(sampled,[.025,.975]).tolist(),"draws":draws}

def choice_comparison(native: np.ndarray, critic: np.ndarray, simulator: np.ndarray, candidates: np.ndarray, tolerance: float) -> dict:
    return selected_change(native,critic,simulator,candidates,tolerance)
