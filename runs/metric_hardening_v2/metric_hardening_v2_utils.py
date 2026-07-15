"""Utilities for the metric-hardening v2 recompute.

This module is intentionally local to ``runs/metric_hardening_v2``. It reuses
the statistical conventions from the first metric-hardening pass, while adding
the v2-specific action matching, whitening variants, and density matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EPS = 1e-8


def finite_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def trimmed_mean(values: Iterable[float] | np.ndarray, proportion_to_cut: float = 0.1) -> float:
    arr = np.sort(finite_array(values))
    if len(arr) == 0:
        return float("nan")
    trim = int(math.floor(len(arr) * proportion_to_cut))
    if trim == 0 or 2 * trim >= len(arr):
        return float(arr.mean())
    return float(arr[trim:-trim].mean())


def probability_superiority(left: Iterable[float] | np.ndarray, right: Iterable[float] | np.ndarray) -> float:
    left_arr = finite_array(left)
    right_arr = np.sort(finite_array(right))
    if len(left_arr) == 0 or len(right_arr) == 0:
        return float("nan")
    less = np.searchsorted(right_arr, left_arr, side="left")
    less_equal = np.searchsorted(right_arr, left_arr, side="right")
    ties = less_equal - less
    return float((less.sum() + 0.5 * ties.sum()) / (len(left_arr) * len(right_arr)))


def describe_values(values: Iterable[float] | np.ndarray) -> dict[str, float]:
    arr = finite_array(values)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "trimmed_mean": trimmed_mean(arr),
    }


def contrast_values(left: Iterable[float] | np.ndarray, right: Iterable[float] | np.ndarray) -> dict[str, float]:
    left_arr = finite_array(left)
    right_arr = finite_array(right)
    p_superiority = probability_superiority(left_arr, right_arr)
    out = {
        "left_n": int(len(left_arr)),
        "right_n": int(len(right_arr)),
        "left_mean": float(left_arr.mean()) if len(left_arr) else float("nan"),
        "right_mean": float(right_arr.mean()) if len(right_arr) else float("nan"),
        "left_median": float(np.median(left_arr)) if len(left_arr) else float("nan"),
        "right_median": float(np.median(right_arr)) if len(right_arr) else float("nan"),
        "left_trimmed_mean": trimmed_mean(left_arr),
        "right_trimmed_mean": trimmed_mean(right_arr),
        "p_superiority": p_superiority,
        "cliffs_delta": 2.0 * p_superiority - 1.0 if np.isfinite(p_superiority) else float("nan"),
    }
    out["delta_mean"] = out["left_mean"] - out["right_mean"] if len(left_arr) and len(right_arr) else float("nan")
    out["delta_median"] = out["left_median"] - out["right_median"] if len(left_arr) and len(right_arr) else float("nan")
    out["delta_trimmed_mean"] = (
        out["left_trimmed_mean"] - out["right_trimmed_mean"] if len(left_arr) and len(right_arr) else float("nan")
    )
    return out


def percentile_ci(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    arr = finite_array([row.get(key, float("nan")) for row in rows])
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def attach_cis(row: dict[str, object], boot_rows: list[dict[str, float]], metrics: Iterable[str]) -> dict[str, object]:
    out = dict(row)
    for metric in metrics:
        low, high = percentile_ci(boot_rows, metric)
        out[f"{metric}_ci_low"] = low
        out[f"{metric}_ci_high"] = high
    return out


def cluster_positions(df: pd.DataFrame, episode_col: str) -> list[np.ndarray]:
    reset = df.reset_index(drop=True)
    return [group.index.to_numpy(dtype=np.int64) for _, group in reset.groupby(episode_col, sort=False)]


def bootstrap_sample_positions(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if not clusters:
        return np.asarray([], dtype=np.int64)
    selected = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in selected])


def clustered_group_summary(
    df: pd.DataFrame,
    *,
    mask: np.ndarray,
    value_col: str,
    n_bootstrap: int,
    seed: int,
    episode_col: str,
) -> dict[str, float]:
    work = df.reset_index(drop=True)
    mask_arr = np.asarray(mask, dtype=bool)
    point = describe_values(work.loc[mask_arr, value_col].to_numpy(dtype=np.float64))
    clusters = [idx for idx in cluster_positions(work, episode_col) if bool(mask_arr[idx].any())]
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = bootstrap_sample_positions(clusters, rng)
        if len(idx) == 0:
            boot_rows.append(describe_values([]))
            continue
        boot_rows.append(describe_values(work.iloc[idx].loc[mask_arr[idx], value_col].to_numpy(dtype=np.float64)))
    return attach_cis(point, boot_rows, ["mean", "median", "trimmed_mean"])


def clustered_contrast_summary(
    df: pd.DataFrame,
    *,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    value_col: str,
    n_bootstrap: int,
    seed: int,
    episode_col: str,
) -> dict[str, float]:
    work = df.reset_index(drop=True)
    left_arr = np.asarray(left_mask, dtype=bool)
    right_arr = np.asarray(right_mask, dtype=bool)
    point = contrast_values(
        work.loc[left_arr, value_col].to_numpy(dtype=np.float64),
        work.loc[right_arr, value_col].to_numpy(dtype=np.float64),
    )
    clusters = [idx for idx in cluster_positions(work, episode_col) if bool(left_arr[idx].any() or right_arr[idx].any())]
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = bootstrap_sample_positions(clusters, rng)
        if len(idx) == 0:
            boot_rows.append(contrast_values([], []))
            continue
        boot = work.iloc[idx]
        boot_rows.append(
            contrast_values(
                boot.loc[left_arr[idx], value_col].to_numpy(dtype=np.float64),
                boot.loc[right_arr[idx], value_col].to_numpy(dtype=np.float64),
            )
        )
    return attach_cis(point, boot_rows, ["delta_mean", "delta_median", "delta_trimmed_mean", "p_superiority"])


def per_dim_std(calibration_values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    std = np.asarray(calibration_values, dtype=np.float64).std(axis=0)
    return np.where(std < floor, 1.0, std)


def standardize_fit(train_values: np.ndarray, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_values, dtype=np.float64)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < floor, 1.0, std)
    return mean, std


def standardize_apply(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / np.asarray(std, dtype=np.float64)


def rms_scaled(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("values must be a 2D array")
    return arr / math.sqrt(max(arr.shape[1], 1))


def balanced_joint_features(state_z: np.ndarray, action_z: np.ndarray) -> np.ndarray:
    state = rms_scaled(state_z)
    action = rms_scaled(action_z)
    if state.shape[0] != action.shape[0]:
        raise ValueError("state and action row counts must match")
    return np.concatenate([state, action], axis=1)


@dataclass(frozen=True)
class WhiteningTransform:
    name: str
    family: str
    mean: np.ndarray
    transform: np.ndarray
    eigvals: np.ndarray
    used_eigvals: np.ndarray
    retained_dim: int
    condition_number: float
    variance_retained: float
    floor_value: float


def _covariance_eigh(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return mean, eigvals[order], eigvecs[:, order]


def fit_whitening_transform(
    values: np.ndarray,
    *,
    name: str,
    family: str,
    floor_fraction: float | None = None,
    pca_variance: float | None = None,
    eps: float = 1e-6,
) -> WhiteningTransform:
    mean, eigvals_desc, eigvecs_desc = _covariance_eigh(values)
    positive = eigvals_desc[eigvals_desc > 1e-12]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    total = float(np.maximum(eigvals_desc, 0.0).sum())

    if pca_variance is not None:
        if not (0.0 < pca_variance <= 1.0):
            raise ValueError("pca_variance must be in (0, 1]")
        nonneg = np.maximum(eigvals_desc, 0.0)
        cumulative = np.cumsum(nonneg) / total if total > 0.0 else np.zeros_like(nonneg)
        retained = int(np.searchsorted(cumulative, pca_variance, side="left") + 1)
        retained = min(max(retained, 1), len(eigvals_desc))
        used = eigvals_desc[:retained]
        vecs = eigvecs_desc[:, :retained]
        floor_value = max(float(used.max()) * eps, eps) if len(used) else eps
        inv_sqrt = 1.0 / np.sqrt(np.maximum(used, floor_value))
        transform = vecs * inv_sqrt
        variance_retained = float(nonneg[:retained].sum() / total) if total > 0.0 else float("nan")
        return WhiteningTransform(
            name=name,
            family=family,
            mean=mean,
            transform=transform,
            eigvals=eigvals_desc,
            used_eigvals=np.maximum(used, floor_value),
            retained_dim=retained,
            condition_number=condition,
            variance_retained=variance_retained,
            floor_value=float(floor_value),
        )

    max_eig = float(eigvals_desc.max()) if len(eigvals_desc) else 0.0
    if floor_fraction is None:
        floor_value = max(max_eig * eps, eps)
    else:
        floor_value = max(max_eig * float(floor_fraction), eps)
    used = np.maximum(eigvals_desc, floor_value)
    inv_sqrt = 1.0 / np.sqrt(used)
    transform = eigvecs_desc * inv_sqrt
    return WhiteningTransform(
        name=name,
        family=family,
        mean=mean,
        transform=transform,
        eigvals=eigvals_desc,
        used_eigvals=used,
        retained_dim=int(len(eigvals_desc)),
        condition_number=condition,
        variance_retained=1.0,
        floor_value=float(floor_value),
    )


def apply_transform(values: np.ndarray, transform: WhiteningTransform) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - transform.mean) @ transform.transform


def apply_delta_transform(delta: np.ndarray, transform: WhiteningTransform) -> np.ndarray:
    return np.asarray(delta, dtype=np.float64) @ transform.transform


@dataclass(frozen=True)
class NeighborResult:
    indices: np.ndarray
    distances: np.ndarray
    valid: np.ndarray


def cross_episode_neighbors(
    x: np.ndarray,
    episode_keys: Iterable[object],
    *,
    k: int,
    k_min: int,
) -> NeighborResult:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x must be 2D")
    n = arr.shape[0]
    keys = np.asarray(list(episode_keys), dtype=object)
    if len(keys) != n:
        raise ValueError("episode_keys length must match x")
    if k < 1 or k_min < 1:
        raise ValueError("k and k_min must be positive")
    if n <= 1:
        return NeighborResult(
            indices=np.full((n, k), -1, dtype=np.int64),
            distances=np.full((n, k), np.nan, dtype=np.float64),
            valid=np.zeros(n, dtype=bool),
        )

    _, counts = np.unique(keys, return_counts=True)
    query_k = min(n, max(k + 1, int(counts.max()) + k + 2, 64))
    tree = cKDTree(arr)
    final_idx = np.full((n, k), -1, dtype=np.int64)
    final_dist = np.full((n, k), np.nan, dtype=np.float64)

    unresolved = np.arange(n)
    while len(unresolved):
        dists, idxs = tree.query(arr[unresolved], k=query_k)
        if query_k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        still = []
        for row_pos, anchor in enumerate(unresolved):
            valid_mask = (idxs[row_pos] != anchor) & (keys[idxs[row_pos]] != keys[anchor])
            keep_idx = idxs[row_pos][valid_mask][:k]
            keep_dist = dists[row_pos][valid_mask][:k]
            if len(keep_idx) >= k_min:
                final_idx[anchor, : len(keep_idx)] = keep_idx
                final_dist[anchor, : len(keep_dist)] = keep_dist
            else:
                still.append(anchor)
        if not still or query_k >= n:
            break
        unresolved = np.asarray(still, dtype=np.int64)
        query_k = min(n, query_k * 2)

    valid = np.sum(final_idx >= 0, axis=1) >= k_min
    return NeighborResult(indices=final_idx, distances=final_dist, valid=valid)


def euclidean_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64), axis=1)


def local_sensitivity_frame(
    *,
    x: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    episode_keys: Iterable[object],
    regimes: Iterable[str],
    row_ids: Iterable[object],
    k: int,
    k_min: int,
    action_z: np.ndarray | None = None,
    neighbors: NeighborResult | None = None,
    extra_cols: dict[str, Iterable[object]] | None = None,
) -> pd.DataFrame:
    row_ids_list = list(row_ids)
    episode_keys_list = list(episode_keys)
    regimes_arr = np.asarray(list(regimes), dtype=object)
    if neighbors is None:
        neighbors = cross_episode_neighbors(x, episode_keys_list, k=k, k_min=k_min)
    y_arr = np.asarray(y, dtype=np.float64)
    r_arr = np.asarray(residuals, dtype=np.float64)
    action_arr = None if action_z is None else rms_scaled(action_z)
    rows = []
    extra_values = {key: list(value) for key, value in (extra_cols or {}).items()}
    for i in range(len(y_arr)):
        idx = neighbors.indices[i]
        idx = idx[idx >= 0]
        base = {
            "row_id": row_ids_list[i],
            "episode_key": str(episode_keys_list[i]),
            "regime": str(regimes_arr[i]),
            "valid_neighbors": bool(len(idx) >= k_min),
            "n_neighbors": int(len(idx)),
            "neighbor_radius": float("nan"),
            "input_spread": float("nan"),
            "successor_spread": float("nan"),
            "residual_spread": float("nan"),
            "local_expansion": float("nan"),
            "model_sensitivity": float("nan"),
            "residual_norm": float(np.linalg.norm(r_arr[i])),
            "action_neighbor_distance": float("nan"),
        }
        for key, values in extra_values.items():
            base[key] = values[i]
        if len(idx) < k_min:
            rows.append(base)
            continue
        in_dist = neighbors.distances[i][: len(idx)]
        out_dist = euclidean_rows(y_arr[idx], y_arr[i])
        res_dist = euclidean_rows(r_arr[idx], r_arr[i])
        input_spread = float(np.median(in_dist))
        successor_spread = float(np.median(out_dist))
        residual_spread = float(np.median(res_dist))
        base.update(
            {
                "neighbor_radius": input_spread,
                "input_spread": input_spread,
                "successor_spread": successor_spread,
                "residual_spread": residual_spread,
                "local_expansion": successor_spread / max(input_spread, EPS),
                "model_sensitivity": residual_spread / max(input_spread, EPS),
            }
        )
        if action_arr is not None:
            base["action_neighbor_distance"] = float(np.median(euclidean_rows(action_arr[idx], action_arr[i])))
        rows.append(base)
    return pd.DataFrame(rows)


def summarize_sensitivity_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    r_item: str,
    scale: str,
    right_regime: str,
    n_bootstrap: int,
    seed: int,
    extra_value_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_cols = [
        "neighbor_radius",
        "action_neighbor_distance",
        "local_expansion",
        "model_sensitivity",
        "residual_norm",
    ]
    for col in extra_value_cols or []:
        if col not in value_cols:
            value_cols.append(col)
    valid = frame.loc[frame["valid_neighbors"].astype(bool)].reset_index(drop=True)
    group_rows = []
    contrast_rows = []
    for regime_index, regime in enumerate(frame["regime"].drop_duplicates().tolist()):
        all_mask = frame["regime"].eq(regime).to_numpy(dtype=bool)
        valid_mask = valid["regime"].eq(regime).to_numpy(dtype=bool)
        for value_index, value_col in enumerate(value_cols):
            if value_col not in valid.columns:
                continue
            row = clustered_group_summary(
                valid,
                mask=valid_mask,
                value_col=value_col,
                n_bootstrap=n_bootstrap,
                seed=seed + regime_index * 1000 + value_index,
                episode_col="episode_key",
            )
            row.update(
                {
                    "dataset": dataset,
                    "r_item": r_item,
                    "scale": scale,
                    "regime": regime,
                    "metric": value_col,
                    "n_anchors": int(all_mask.sum()),
                    "n_usable": int(valid_mask.sum()),
                    "n_below_5_neighbors": int(all_mask.sum() - valid_mask.sum()),
                    "n_episodes": int(valid.loc[valid_mask, "episode_key"].nunique()),
                }
            )
            group_rows.append(row)

    left_mask = valid["regime"].eq("interaction").to_numpy(dtype=bool)
    right_mask = valid["regime"].eq(right_regime).to_numpy(dtype=bool)
    for value_index, value_col in enumerate(value_cols):
        if value_col not in valid.columns:
            continue
        row = clustered_contrast_summary(
            valid,
            left_mask=left_mask,
            right_mask=right_mask,
            value_col=value_col,
            n_bootstrap=n_bootstrap,
            seed=seed + 50000 + value_index,
            episode_col="episode_key",
        )
        row.update(
            {
                "dataset": dataset,
                "r_item": r_item,
                "scale": scale,
                "contrast": f"interaction_vs_{right_regime}",
                "metric": value_col,
                "left_episodes": int(valid.loc[left_mask, "episode_key"].nunique()),
                "right_episodes": int(valid.loc[right_mask, "episode_key"].nunique()),
            }
        )
        contrast_rows.append(row)
    return pd.DataFrame(group_rows), pd.DataFrame(contrast_rows)


def contrast_pass(contrasts: pd.DataFrame, *, metric: str, scale: str | None = None) -> bool:
    rows = contrasts.loc[contrasts["metric"].eq(metric)]
    if scale is not None and "scale" in rows.columns:
        rows = rows.loc[rows["scale"].eq(scale)]
    if rows.empty:
        return False
    row = rows.iloc[0]
    return bool(
        np.isfinite(float(row["delta_trimmed_mean"]))
        and float(row["delta_trimmed_mean"]) > 0.0
        and np.isfinite(float(row["delta_trimmed_mean_ci_low"]))
        and float(row["delta_trimmed_mean_ci_low"]) > 0.0
    )


def coverage_ok(frame: pd.DataFrame, *, regime: str = "interaction", min_rows: int = 30, min_episodes: int = 5) -> bool:
    valid = frame.loc[frame["valid_neighbors"].astype(bool) & frame["regime"].eq(regime)]
    return bool(len(valid) >= min_rows and valid["episode_key"].nunique() >= min_episodes)


def cube_action_blocks_from_arrays(
    records: pd.DataFrame,
    *,
    actions: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    episode_ids: np.ndarray,
    frameskip: int = 5,
    raw_action_dim: int = 5,
) -> np.ndarray:
    out = np.full((len(records), frameskip * raw_action_dim), np.nan, dtype=np.float64)
    seen_keys: set[tuple[int, int]] = set()
    for pos, row in enumerate(records.itertuples(index=False)):
        episode_ordinal = int(row.episode_ordinal)
        episode_id = int(row.episode_id)
        transition_block = int(row.transition_block)
        raw_step = int(row.raw_step)
        model_step = int(row.model_step)
        if episode_ordinal < 0 or episode_ordinal >= len(offsets):
            raise ValueError(f"episode_ordinal out of range: {episode_ordinal}")
        if int(episode_ids[episode_ordinal]) != episode_id:
            raise ValueError(f"episode id mismatch for ordinal {episode_ordinal}: {episode_id}")
        if raw_step != model_step * frameskip:
            raise ValueError(f"raw_step/model_step mismatch for row {pos}: {raw_step} vs {model_step * frameskip}")
        if transition_block != model_step - 1:
            raise ValueError(f"transition_block/model_step mismatch for row {pos}")
        key = (episode_ordinal, transition_block)
        if key in seen_keys:
            raise ValueError(f"duplicate action join key: {key}")
        seen_keys.add(key)
        raw_start = int(offsets[episode_ordinal]) + transition_block * frameskip
        raw_stop = raw_start + frameskip
        ep_stop = int(offsets[episode_ordinal] + lengths[episode_ordinal])
        if raw_start < int(offsets[episode_ordinal]) or raw_stop > ep_stop:
            raise ValueError(f"action block outside episode bounds for row {pos}")
        block = np.asarray(actions[raw_start:raw_stop], dtype=np.float64).reshape(-1)
        if block.shape[0] != frameskip * raw_action_dim:
            raise ValueError(f"action block has wrong shape for row {pos}: {block.shape}")
        out[pos] = block
    if not np.isfinite(out).all():
        raise ValueError("non-finite action join output")
    return out


def cube_kinematics_from_arrays(
    records: pd.DataFrame,
    *,
    effector_pos: np.ndarray,
    block_pos: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    block_quat: np.ndarray | None = None,
    block_yaw: np.ndarray | None = None,
    frameskip: int = 5,
) -> tuple[np.ndarray, str]:
    rows = []
    pose_source = "quat"
    for pos, row in enumerate(records.itertuples(index=False)):
        episode_ordinal = int(row.episode_ordinal)
        raw_step = int(row.raw_step)
        model_step = int(row.model_step)
        if raw_step != model_step * frameskip:
            raise ValueError(f"raw_step/model_step mismatch for row {pos}")
        raw_index = int(offsets[episode_ordinal]) + raw_step
        ep_stop = int(offsets[episode_ordinal] + lengths[episode_ordinal])
        if raw_index >= ep_stop:
            raise ValueError(f"kinematic raw index outside episode bounds for row {pos}")
        eff = np.asarray(effector_pos[raw_index], dtype=np.float64).reshape(-1)
        block = np.asarray(block_pos[raw_index], dtype=np.float64).reshape(-1)
        if block_quat is not None:
            quat = np.asarray(block_quat[raw_index], dtype=np.float64).reshape(-1)
            norm = float(np.linalg.norm(quat))
            quat = quat / norm if norm > EPS else quat
            pose = quat
        elif block_yaw is not None:
            yaw = float(np.asarray(block_yaw[raw_index]).reshape(-1)[0])
            pose = np.asarray([math.sin(yaw), math.cos(yaw)], dtype=np.float64)
            pose_source = "yaw_sincos"
        else:
            raise ValueError("Either block_quat or block_yaw must be provided")
        rows.append(np.concatenate([eff, block, pose]))
    return np.asarray(rows, dtype=np.float64), pose_source


def _normalize_quat(values: np.ndarray) -> np.ndarray:
    quat = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > EPS else quat


def cube_augmented_kinematics_from_arrays(
    records: pd.DataFrame,
    *,
    effector_pos: np.ndarray,
    block_pos: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    block_quat: np.ndarray | None = None,
    block_yaw: np.ndarray | None = None,
    native_velocity_arrays: dict[str, np.ndarray] | None = None,
    gripper_arrays: dict[str, np.ndarray] | None = None,
    frameskip: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build input-time Cube kinematic features for R3v2.

    For a prediction row whose target model step is ``model_step``, the input
    state is the previous model step ``transition_block``. Features therefore
    use raw index ``transition_block * frameskip`` and finite differences from
    the preceding observed model step. This avoids target-time pose leakage.
    """

    velocity_arrays = native_velocity_arrays or {}
    grip_arrays = gripper_arrays or {}
    rows = []
    feature_names = [
        "effector_pos_x",
        "effector_pos_y",
        "effector_pos_z",
        "block_pos_x",
        "block_pos_y",
        "block_pos_z",
    ]
    pose_source = "quat" if block_quat is not None else "yaw_sincos"
    if block_quat is not None:
        feature_names.extend([f"block_quat_{i}" for i in range(4)])
    elif block_yaw is not None:
        feature_names.extend(["block_yaw_sin", "block_yaw_cos"])
    else:
        raise ValueError("Either block_quat or block_yaw must be provided")
    feature_names.extend(
        [
            "effector_delta_x",
            "effector_delta_y",
            "effector_delta_z",
            "block_delta_x",
            "block_delta_y",
            "block_delta_z",
        ]
    )
    if block_quat is not None:
        feature_names.extend([f"block_quat_delta_{i}" for i in range(4)])
    else:
        feature_names.extend(["block_yaw_sin_delta", "block_yaw_cos_delta"])
    for key, values in velocity_arrays.items():
        dim = int(np.asarray(values).reshape((np.asarray(values).shape[0], -1)).shape[1])
        feature_names.extend([f"native_velocity:{key}_{i}" for i in range(dim)])
    for key, values in grip_arrays.items():
        dim = int(np.asarray(values).reshape((np.asarray(values).shape[0], -1)).shape[1])
        feature_names.extend([f"gripper:{key}_{i}" for i in range(dim)])

    for pos, row in enumerate(records.itertuples(index=False)):
        episode_ordinal = int(row.episode_ordinal)
        model_step = int(row.model_step)
        transition_block = int(row.transition_block)
        if transition_block != model_step - 1:
            raise ValueError(f"transition_block/model_step mismatch for row {pos}")
        input_raw = transition_block * frameskip
        prev_raw = input_raw - frameskip
        if prev_raw < 0:
            raise ValueError(f"no previous observed model step for row {pos}")
        raw_index = int(offsets[episode_ordinal]) + input_raw
        prev_index = int(offsets[episode_ordinal]) + prev_raw
        ep_start = int(offsets[episode_ordinal])
        ep_stop = int(offsets[episode_ordinal] + lengths[episode_ordinal])
        if prev_index < ep_start or raw_index >= ep_stop:
            raise ValueError(f"augmented kinematic index outside episode bounds for row {pos}")

        eff = np.asarray(effector_pos[raw_index], dtype=np.float64).reshape(-1)
        eff_prev = np.asarray(effector_pos[prev_index], dtype=np.float64).reshape(-1)
        block = np.asarray(block_pos[raw_index], dtype=np.float64).reshape(-1)
        block_prev = np.asarray(block_pos[prev_index], dtype=np.float64).reshape(-1)
        parts = [eff, block]
        if block_quat is not None:
            pose = _normalize_quat(block_quat[raw_index])
            pose_prev = _normalize_quat(block_quat[prev_index])
            pose_delta = pose - pose_prev
        else:
            yaw = float(np.asarray(block_yaw[raw_index]).reshape(-1)[0])
            yaw_prev = float(np.asarray(block_yaw[prev_index]).reshape(-1)[0])
            pose = np.asarray([math.sin(yaw), math.cos(yaw)], dtype=np.float64)
            pose_prev = np.asarray([math.sin(yaw_prev), math.cos(yaw_prev)], dtype=np.float64)
            pose_delta = pose - pose_prev
        parts.extend([pose, eff - eff_prev, block - block_prev, pose_delta])
        for values in velocity_arrays.values():
            parts.append(np.asarray(values[raw_index], dtype=np.float64).reshape(-1))
        for values in grip_arrays.values():
            parts.append(np.asarray(values[raw_index], dtype=np.float64).reshape(-1))
        rows.append(np.concatenate(parts))

    out = np.asarray(rows, dtype=np.float64)
    if len(feature_names) != out.shape[1]:
        raise ValueError(f"feature name count mismatch: {len(feature_names)} vs {out.shape[1]}")
    provenance = {
        "pose_source": pose_source,
        "input_time": "transition_block * 5",
        "finite_difference": "input observed model step minus previous observed model step",
        "native_velocity_keys": sorted(velocity_arrays.keys()),
        "gripper_keys": sorted(grip_arrays.keys()),
        "feature_dim": int(out.shape[1]),
        "feature_names": feature_names,
    }
    return out, provenance


def matched_pairs_by_radius(
    frame: pd.DataFrame,
    *,
    radius_edges: np.ndarray,
    left_regime: str = "interaction",
    right_regime: str = "free",
) -> pd.DataFrame:
    valid = frame.loc[frame["valid_neighbors"].astype(bool)].copy().reset_index(drop=True)
    edges = np.asarray(radius_edges, dtype=np.float64)
    if len(edges) < 2:
        return pd.DataFrame()
    valid["radius_bin"] = np.searchsorted(edges[1:-1], valid["neighbor_radius"].to_numpy(dtype=np.float64), side="right")
    pair_rows = []
    for radius_bin, group in valid.groupby("radius_bin", sort=True):
        left_rows = group.loc[group["regime"].eq(left_regime)].sort_values("neighbor_radius")
        right_rows = group.loc[group["regime"].eq(right_regime)].copy()
        available = set(right_rows.index.tolist())
        for left_index, left_row in left_rows.iterrows():
            if not available:
                break
            candidates = right_rows.loc[list(available)]
            candidates = candidates.loc[candidates["episode_key"].astype(str) != str(left_row["episode_key"])]
            if candidates.empty:
                continue
            gaps = np.abs(candidates["neighbor_radius"].to_numpy(dtype=np.float64) - float(left_row["neighbor_radius"]))
            right_index = candidates.index[int(np.argmin(gaps))]
            right_row = valid.loc[right_index]
            available.remove(right_index)
            pair_rows.append(
                {
                    "radius_bin": int(radius_bin),
                    "left_row_id": left_row["row_id"],
                    "right_row_id": right_row["row_id"],
                    "left_episode_key": left_row["episode_key"],
                    "right_episode_key": right_row["episode_key"],
                    "left_radius": float(left_row["neighbor_radius"]),
                    "right_radius": float(right_row["neighbor_radius"]),
                    "abs_radius_gap": float(abs(float(left_row["neighbor_radius"]) - float(right_row["neighbor_radius"]))),
                    "left_local_expansion": float(left_row["local_expansion"]),
                    "right_local_expansion": float(right_row["local_expansion"]),
                    "left_model_sensitivity": float(left_row["model_sensitivity"]),
                    "right_model_sensitivity": float(right_row["model_sensitivity"]),
                }
            )
    return pd.DataFrame(pair_rows)


def cross_fitted_knn_probe_frame(
    *,
    x: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    episode_keys: Iterable[object],
    regimes: Iterable[str],
    row_ids: Iterable[object],
    k: int,
    k_min: int,
    n_folds: int = 5,
) -> pd.DataFrame:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    r_arr = np.asarray(residuals, dtype=np.float64)
    keys = np.asarray(list(episode_keys), dtype=object)
    regimes_arr = np.asarray(list(regimes), dtype=object)
    row_ids_list = list(row_ids)
    unique_keys = np.asarray(sorted(set(keys.astype(str))), dtype=object)
    fold_by_key = {key: i % n_folds for i, key in enumerate(unique_keys)}
    folds = np.asarray([fold_by_key[str(key)] for key in keys], dtype=np.int64)
    rows = []
    for fold in range(n_folds):
        eval_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)
        if len(eval_idx) == 0 or len(train_idx) < k_min:
            continue
        query_k = min(len(train_idx), max(k, k_min))
        tree = cKDTree(x_arr[train_idx])
        dists, local_idxs = tree.query(x_arr[eval_idx], k=query_k)
        if query_k == 1:
            dists = dists[:, None]
            local_idxs = local_idxs[:, None]
        baseline_r = r_arr[train_idx].mean(axis=0)
        baseline_y = y_arr[train_idx].mean(axis=0)
        for row_pos, anchor in enumerate(eval_idx):
            keep_local = local_idxs[row_pos][:k]
            keep_dist = dists[row_pos][:k]
            keep = train_idx[keep_local]
            if len(keep) < k_min:
                valid = False
                neighbor_radius = float("nan")
                pred_r = np.full(r_arr.shape[1], np.nan, dtype=np.float64)
                pred_y = np.full(y_arr.shape[1], np.nan, dtype=np.float64)
            else:
                valid = True
                neighbor_radius = float(np.median(keep_dist[: len(keep)]))
                pred_r = r_arr[keep].mean(axis=0)
                pred_y = y_arr[keep].mean(axis=0)
            residual_pred_mse = float(np.mean((r_arr[anchor] - pred_r) ** 2)) if valid else float("nan")
            residual_baseline_mse = float(np.mean((r_arr[anchor] - baseline_r) ** 2))
            floor = float(np.mean((y_arr[anchor] - pred_y) ** 2)) if valid else float("nan")
            successor_baseline_mse = float(np.mean((y_arr[anchor] - baseline_y) ** 2))
            rows.append(
                {
                    "row_id": row_ids_list[anchor],
                    "episode_key": str(keys[anchor]),
                    "regime": str(regimes_arr[anchor]),
                    "fold": int(fold),
                    "valid_neighbors": bool(valid),
                    "n_neighbors": int(len(keep)) if valid else 0,
                    "neighbor_radius": neighbor_radius,
                    "actual_residual_norm": float(np.linalg.norm(r_arr[anchor])),
                    "predicted_residual_norm": float(np.linalg.norm(pred_r)) if valid else float("nan"),
                    "residual_prediction_mse": residual_pred_mse,
                    "residual_baseline_mse": residual_baseline_mse,
                    "residual_predictability": 1.0 - residual_pred_mse / max(residual_baseline_mse, EPS)
                    if valid
                    else float("nan"),
                    "conditional_mean_floor": floor,
                    "successor_baseline_mse": successor_baseline_mse,
                    "conditional_mean_gain": 1.0 - floor / max(successor_baseline_mse, EPS) if valid else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def matched_pair_metric_summary(
    pairs: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if pairs.empty:
        return {
            "metric": metric,
            "n_pairs": 0,
            "eligible_episodes": 0,
            "delta_trimmed_mean": float("nan"),
            "delta_trimmed_mean_ci_low": float("nan"),
            "delta_trimmed_mean_ci_high": float("nan"),
            "p_superiority": float("nan"),
        }
    left_col = f"left_{metric}"
    right_col = f"right_{metric}"
    work = pairs.reset_index(drop=True).copy()
    work["delta"] = work[left_col].to_numpy(dtype=np.float64) - work[right_col].to_numpy(dtype=np.float64)
    point = contrast_values(work[left_col], work[right_col])
    point["paired_delta_trimmed_mean"] = trimmed_mean(work["delta"])
    point["paired_delta_median"] = float(np.median(work["delta"]))
    clusters = cluster_positions(work, "left_episode_key")
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = bootstrap_sample_positions(clusters, rng)
        if len(idx) == 0:
            boot_rows.append(contrast_values([], []))
            boot_rows[-1]["paired_delta_trimmed_mean"] = float("nan")
            boot_rows[-1]["paired_delta_median"] = float("nan")
            continue
        boot = work.iloc[idx]
        row = contrast_values(boot[left_col], boot[right_col])
        row["paired_delta_trimmed_mean"] = trimmed_mean(boot["delta"])
        row["paired_delta_median"] = float(np.median(boot["delta"]))
        boot_rows.append(row)
    point = attach_cis(
        point,
        boot_rows,
        ["delta_mean", "delta_median", "delta_trimmed_mean", "p_superiority", "paired_delta_trimmed_mean", "paired_delta_median"],
    )
    point.update(
        {
            "metric": metric,
            "n_pairs": int(len(work)),
            "eligible_episodes": int(work["left_episode_key"].nunique()),
            "median_abs_radius_gap": float(np.median(work["abs_radius_gap"])),
            "p90_abs_radius_gap": float(np.percentile(work["abs_radius_gap"], 90.0)),
        }
    )
    return point


def k_sweep_pass(rows: pd.DataFrame, *, metric: str) -> bool:
    work = rows.loc[rows["metric"].eq(metric)]
    if len(work) != 4:
        return False
    deltas = work["delta_trimmed_mean"].to_numpy(dtype=np.float64)
    lows = work["delta_trimmed_mean_ci_low"].to_numpy(dtype=np.float64)
    return bool(np.isfinite(deltas).all() and (deltas > 0.0).all() and np.sum(lows > 0.0) >= 3)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
