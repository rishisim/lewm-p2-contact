"""Local threshold policies, matched baselines, and uncertainty for V3."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def local_depths(scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 3 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite [N,3]")
    depth = np.ones(len(scores), dtype=np.int64)
    active = np.ones(len(scores), dtype=bool)
    for column in range(3):
        keep = active & (scores[:, column] > threshold)
        depth[keep] += 1
        active &= keep
    return depth


def calibrate_threshold(scores: np.ndarray, raw_losses: np.ndarray, target_mean: float,
                        grid_size: int) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    raw_losses = np.asarray(raw_losses, dtype=np.float64)
    finite = np.sort(np.unique(scores.ravel()))
    candidates = np.concatenate(([np.inf], (finite[:-1] + finite[1:]) / 2, [-np.inf]))
    if len(candidates) > grid_size:
        indices = np.linspace(0, len(candidates) - 1, grid_size).round().astype(int)
        candidates = candidates[np.unique(indices)]
    best = None
    for threshold in candidates:
        depths = local_depths(scores, float(threshold))
        loss = float(raw_losses[np.arange(len(depths)), depths].mean())
        key = (abs(float(depths.mean()) - target_mean), loss, -float(threshold))
        if best is None or key < best[0]:
            best = (key, threshold, depths, loss)
    assert best is not None
    return {
        "threshold": float(best[1]), "target_mean_depth": float(target_mean),
        "calibration_mean_depth": float(best[2].mean()),
        "calibration_total_calls": int(best[2].sum()), "calibration_raw_mse": float(best[3]),
    }


def gather(losses: np.ndarray, depths: Iterable[int]) -> np.ndarray:
    depths = np.asarray(depths, dtype=np.int64)
    losses = np.asarray(losses)
    if losses.ndim != 2 or losses.shape[0] != len(depths):
        raise ValueError("loss/depth shape mismatch")
    return losses[np.arange(len(depths)), depths]


def histogram(depths: Iterable[int]) -> dict[int, int]:
    return {int(k): int(v) for k, v in sorted(Counter(np.asarray(depths).tolist()).items())}


def strongest_mixture(n: int, total_calls: int, calibration_means: np.ndarray,
                      seed: int) -> np.ndarray:
    """Best calibration-selected mixture over depths 1,2,4 at exact calls."""
    extra = int(total_calls) - n
    if extra < 0 or extra > 3 * n:
        raise ValueError("infeasible matched call budget")
    best = None
    for n4 in range(min(n, extra // 3) + 1):
        n2 = extra - 3 * n4
        n1 = n - n2 - n4
        if n1 < 0 or n2 < 0:
            continue
        objective = n1 * calibration_means[1] + n2 * calibration_means[2] + n4 * calibration_means[4]
        key = (float(objective), n4, n2)
        if best is None or key < best[0]:
            best = (key, n1, n2, n4)
    if best is None:
        raise RuntimeError("no exact mixture found")
    result = np.asarray([1] * best[1] + [2] * best[2] + [4] * best[3], dtype=np.int64)
    np.random.default_rng(seed).shuffle(result)
    if len(result) != n or int(result.sum()) != total_calls:
        raise RuntimeError("matched mixture accounting failure")
    return result


def histogram_randomized(depths: np.ndarray, seed: int) -> np.ndarray:
    result = np.asarray(depths, dtype=np.int64).copy()
    np.random.default_rng(seed).shuffle(result)
    return result


def histogram_permutation(depths: np.ndarray, episode_ids: np.ndarray, seed: int) -> np.ndarray:
    """Independent episode-stratified permutation preserving the global histogram."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(depths)), np.asarray(episode_ids)))
    shift = max(1, len(depths) // 3)
    permuted = np.empty_like(depths)
    permuted[order] = np.asarray(depths)[np.roll(order, shift)]
    if histogram(permuted) != histogram(depths):
        raise RuntimeError("permutation changed histogram")
    return permuted


def oracle_exact(losses: np.ndarray, total_calls: int) -> np.ndarray:
    """Diagnostic multiple-choice DP over local depths 1..4."""
    losses = np.asarray(losses, dtype=np.float64)
    n = len(losses)
    extra = int(total_calls) - n
    if losses.shape[1] < 5 or extra < 0 or extra > 3 * n:
        raise ValueError("invalid oracle inputs")
    neg_inf = -np.inf
    utility = losses[:, [1]] - losses[:, 1:5]
    dp = np.full(extra + 1, neg_inf)
    dp[0] = 0.0
    choices = np.full((n, extra + 1), -1, dtype=np.int8)
    for i in range(n):
        nxt = np.full_like(dp, neg_inf)
        selected = np.full(extra + 1, -1, dtype=np.int8)
        for cost in range(4):
            if cost > extra:
                continue
            candidate = dp[:extra + 1 - cost] + utility[i, cost]
            current = nxt[cost:]
            better = candidate > current
            current[better] = candidate[better]
            selected[cost:][better] = cost
        dp = nxt
        choices[i] = selected
    if not np.isfinite(dp[extra]):
        raise RuntimeError("oracle budget infeasible")
    result = np.empty(n, dtype=np.int64)
    remaining = extra
    for i in range(n - 1, -1, -1):
        cost = int(choices[i, remaining])
        if cost < 0:
            raise RuntimeError("oracle backtrace failed")
        result[i] = cost + 1
        remaining -= cost
    if remaining != 0 or int(result.sum()) != total_calls:
        raise RuntimeError("oracle accounting failure")
    return result


def clustered_ci(values: np.ndarray, episodes: np.ndarray, samples: int,
                 seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes)
    unique = np.unique(episodes)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    groups = {ep: np.flatnonzero(episodes == ep) for ep in unique}
    for i in range(samples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        total = 0.0
        count = 0
        for ep in selected:
            index = groups[ep]
            total += float(values[index].sum())
            count += len(index)
        means[i] = total / count
    return {
        "mean_benefit": float(values.mean()), "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)), "n": int(len(values)),
        "n_episodes": int(len(unique)), "bootstrap_samples": int(samples),
    }


def nondominated(points: list[dict[str, float | str]]) -> list[bool]:
    result = []
    for i, point in enumerate(points):
        dominated = any(
            j != i and other["mean_calls"] <= point["mean_calls"]
            and other["raw_mse"] <= point["raw_mse"]
            and (other["mean_calls"] < point["mean_calls"] or other["raw_mse"] < point["raw_mse"])
            for j, other in enumerate(points)
        )
        result.append(not dominated)
    return result
