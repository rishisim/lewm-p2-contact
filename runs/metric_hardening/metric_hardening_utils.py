"""Utilities for the metric-hardening diagnostic pilot.

All helpers are local to ``runs/metric_hardening`` so the existing diagnostic
scripts and run directories remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon


EPS = 1e-8


def finite_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def trimmed_mean(values: Iterable[float] | np.ndarray, proportion_to_cut: float = 0.1) -> float:
    """Return a trimmed mean.

    Adapted from ``runs/lewm_transfer/lewm_transfer_stats.py``.
    """

    arr = np.sort(finite_array(values))
    if len(arr) == 0:
        return float("nan")
    trim = int(math.floor(len(arr) * proportion_to_cut))
    if trim == 0 or 2 * trim >= len(arr):
        return float(arr.mean())
    return float(arr[trim:-trim].mean())


def probability_superiority(left: Iterable[float] | np.ndarray, right: Iterable[float] | np.ndarray) -> float:
    """P(left > right), with ties counted as half.

    Adapted from ``runs/lewm_transfer/lewm_transfer_stats.py``.
    """

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
    out = {
        "left_n": int(len(left_arr)),
        "right_n": int(len(right_arr)),
        "left_mean": float(left_arr.mean()) if len(left_arr) else float("nan"),
        "right_mean": float(right_arr.mean()) if len(right_arr) else float("nan"),
        "left_median": float(np.median(left_arr)) if len(left_arr) else float("nan"),
        "right_median": float(np.median(right_arr)) if len(right_arr) else float("nan"),
        "left_trimmed_mean": trimmed_mean(left_arr),
        "right_trimmed_mean": trimmed_mean(right_arr),
        "p_superiority": probability_superiority(left_arr, right_arr),
    }
    out["cliffs_delta"] = 2.0 * out["p_superiority"] - 1.0 if np.isfinite(out["p_superiority"]) else float("nan")
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


def cluster_indices(df: pd.DataFrame, episode_col: str = "episode_ordinal") -> list[np.ndarray]:
    return [group.index.to_numpy() for _, group in df.groupby(episode_col, sort=False)]


def bootstrap_sample_indices(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if not clusters:
        return np.asarray([], dtype=np.int64)
    selected = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in selected])


def clustered_group_summary(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    value_col: str,
    n_bootstrap: int,
    seed: int,
    episode_col: str = "episode_ordinal",
) -> dict[str, float]:
    point = describe_values(df.loc[mask, value_col].to_numpy(dtype=np.float64))
    clusters = [idx for idx in cluster_indices(df, episode_col) if bool(mask.loc[idx].any())]
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = bootstrap_sample_indices(clusters, rng)
        if len(idx) == 0:
            boot_rows.append(describe_values([]))
            continue
        boot_mask = mask.loc[idx].to_numpy(dtype=bool)
        boot_rows.append(describe_values(df.loc[idx].loc[boot_mask, value_col].to_numpy(dtype=np.float64)))
    return attach_cis(point, boot_rows, ["mean", "median", "trimmed_mean"])


def clustered_contrast_summary(
    df: pd.DataFrame,
    *,
    left_mask: pd.Series,
    right_mask: pd.Series,
    value_col: str,
    n_bootstrap: int,
    seed: int,
    episode_col: str = "episode_ordinal",
) -> dict[str, float]:
    point = contrast_values(
        df.loc[left_mask, value_col].to_numpy(dtype=np.float64),
        df.loc[right_mask, value_col].to_numpy(dtype=np.float64),
    )
    clusters = [
        idx for idx in cluster_indices(df, episode_col) if bool(left_mask.loc[idx].any()) or bool(right_mask.loc[idx].any())
    ]
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = bootstrap_sample_indices(clusters, rng)
        if len(idx) == 0:
            boot_rows.append(contrast_values([], []))
            continue
        boot_left = left_mask.loc[idx].to_numpy(dtype=bool)
        boot_right = right_mask.loc[idx].to_numpy(dtype=bool)
        boot_df = df.loc[idx]
        boot_rows.append(
            contrast_values(
                boot_df.loc[boot_left, value_col].to_numpy(dtype=np.float64),
                boot_df.loc[boot_right, value_col].to_numpy(dtype=np.float64),
            )
        )
    return attach_cis(point, boot_rows, ["delta_mean", "delta_median", "delta_trimmed_mean", "p_superiority"])


def per_dim_std(calibration_values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    std = np.asarray(calibration_values, dtype=np.float64).std(axis=0)
    return np.where(std < floor, 1.0, std)


def scaled_mse(delta: np.ndarray, scale: np.ndarray) -> np.ndarray:
    arr = np.asarray(delta, dtype=np.float64) / np.asarray(scale, dtype=np.float64)
    return np.mean(arr**2, axis=1)


def euclidean_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64), axis=1)


@dataclass(frozen=True)
class NeighborResult:
    indices: np.ndarray
    distances: np.ndarray
    valid: np.ndarray


def cross_episode_neighbors(
    x: np.ndarray,
    episode_keys: Iterable[object],
    *,
    k: int = 10,
    k_min: int = 5,
) -> NeighborResult:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x must be 2D")
    n = arr.shape[0]
    keys = np.asarray(list(episode_keys), dtype=object)
    if len(keys) != n:
        raise ValueError("episode_keys length must match x")
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


def local_sensitivity_frame(
    *,
    x: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    episode_keys: Iterable[object],
    regimes: Iterable[str],
    row_ids: Iterable[object],
    k: int = 10,
    k_min: int = 5,
) -> pd.DataFrame:
    row_ids_list = list(row_ids)
    episode_keys_list = list(episode_keys)
    neighbors = cross_episode_neighbors(x, episode_keys, k=k, k_min=k_min)
    y_arr = np.asarray(y, dtype=np.float64)
    r_arr = np.asarray(residuals, dtype=np.float64)
    regimes_arr = np.asarray(list(regimes), dtype=object)
    rows = []
    for i in range(len(y_arr)):
        idx = neighbors.indices[i]
        idx = idx[idx >= 0]
        if len(idx) < k_min:
            rows.append(
                {
                    "row_id": row_ids_list[i],
                    "episode_ordinal": episode_keys_list[i],
                    "regime": str(regimes_arr[i]),
                    "valid_neighbors": False,
                    "n_neighbors": int(len(idx)),
                    "neighbor_radius": float("nan"),
                    "input_spread": float("nan"),
                    "successor_spread": float("nan"),
                    "residual_spread": float("nan"),
                    "local_expansion": float("nan"),
                    "model_sensitivity": float("nan"),
                    "residual_norm": float(np.linalg.norm(r_arr[i])),
                }
            )
            continue
        in_dist = neighbors.distances[i][: len(idx)]
        out_dist = euclidean_rows(y_arr[idx], y_arr[i])
        res_dist = euclidean_rows(r_arr[idx], r_arr[i])
        input_spread = float(np.median(in_dist))
        successor_spread = float(np.median(out_dist))
        residual_spread = float(np.median(res_dist))
        rows.append(
            {
                "row_id": row_ids_list[i],
                "episode_ordinal": episode_keys_list[i],
                "regime": str(regimes_arr[i]),
                "valid_neighbors": True,
                "n_neighbors": int(len(idx)),
                "neighbor_radius": input_spread,
                "input_spread": input_spread,
                "successor_spread": successor_spread,
                "residual_spread": residual_spread,
                "local_expansion": successor_spread / max(input_spread, EPS),
                "model_sensitivity": residual_spread / max(input_spread, EPS),
                "residual_norm": float(np.linalg.norm(r_arr[i])),
            }
        )
    return pd.DataFrame(rows)


def add_quantile_bins(values: np.ndarray, *, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray]:
    finite = finite_array(values)
    if len(finite) == 0:
        return np.full(len(values), -1, dtype=np.int64), np.asarray([], dtype=np.float64)
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return np.full(len(values), -1, dtype=np.int64), edges
    bins = np.searchsorted(edges[1:-1], np.asarray(values, dtype=np.float64), side="right")
    return bins.astype(np.int64), edges


def double_matched_pairs(
    df: pd.DataFrame,
    *,
    value_col: str,
    magnitude_col: str,
    left: str = "interaction",
    right: str = "transport_free",
) -> tuple[pd.DataFrame, np.ndarray]:
    work = df.copy()
    bins, edges = add_quantile_bins(work[magnitude_col].to_numpy(dtype=np.float64), n_bins=5)
    work["_magnitude_bin"] = bins
    if len(np.unique(bins[bins >= 0])) < 3:
        return pd.DataFrame(), edges

    pair_rows = []
    for (episode, decile, mag_bin), group in work.groupby(["episode_ordinal", "phase_decile", "_magnitude_bin"], sort=False):
        if int(mag_bin) < 0:
            continue
        left_rows = group.loc[group["regime"].eq(left)].sort_values(magnitude_col)
        right_rows = group.loc[group["regime"].eq(right)].copy()
        if left_rows.empty or right_rows.empty:
            continue
        available = set(right_rows.index.tolist())
        for left_index, left_row in left_rows.iterrows():
            if not available:
                break
            candidates = right_rows.loc[list(available)]
            gaps = np.abs(candidates[magnitude_col].to_numpy(dtype=np.float64) - float(left_row[magnitude_col]))
            right_index = candidates.index[int(np.argmin(gaps))]
            right_row = work.loc[right_index]
            available.remove(right_index)
            pair_rows.append(
                {
                    "episode_ordinal": episode,
                    "phase_decile": int(decile),
                    "magnitude_bin": int(mag_bin),
                    "left_row_id": int(left_row["row_id"]),
                    "right_row_id": int(right_row["row_id"]),
                    "left_value": float(left_row[value_col]),
                    "right_value": float(right_row[value_col]),
                    "delta": float(left_row[value_col] - right_row[value_col]),
                    "left_magnitude": float(left_row[magnitude_col]),
                    "right_magnitude": float(right_row[magnitude_col]),
                    "abs_magnitude_gap": float(abs(float(left_row[magnitude_col]) - float(right_row[magnitude_col]))),
                    "left_position": float(left_row["normalized_position"]),
                    "right_position": float(right_row["normalized_position"]),
                }
            )
    return pd.DataFrame(pair_rows), edges


def paired_delta_summary(
    pairs: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    episode_col: str = "episode_ordinal",
) -> dict[str, float]:
    if pairs.empty:
        return {
            "eligible_episodes": 0,
            "n_pairs": 0,
            "delta_trimmed_mean": float("nan"),
            "delta_trimmed_mean_ci_low": float("nan"),
            "delta_trimmed_mean_ci_high": float("nan"),
            "median_episode_delta": float("nan"),
            "fraction_episode_positive": float("nan"),
            "fraction_pair_positive": float("nan"),
            "p_superiority": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_p_greater": float("nan"),
            "abs_magnitude_gap_median": float("nan"),
        }
    point = describe_values(pairs["delta"].to_numpy(dtype=np.float64))
    clusters = cluster_indices(pairs, episode_col)
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = bootstrap_sample_indices(clusters, rng)
        boot_rows.append(describe_values(pairs.loc[idx, "delta"].to_numpy(dtype=np.float64)))
    point = attach_cis(point, boot_rows, ["trimmed_mean", "median", "mean"])
    episode_deltas = pairs.groupby(episode_col, sort=False)["delta"].median()
    try:
        result = wilcoxon(episode_deltas.to_numpy(dtype=np.float64), alternative="greater", zero_method="zsplit", method="auto")
        stat = float(result.statistic)
        pvalue = float(result.pvalue)
    except Exception:
        stat = float("nan")
        pvalue = float("nan")
    return {
        "eligible_episodes": int(len(episode_deltas)),
        "n_pairs": int(len(pairs)),
        "delta_trimmed_mean": point["trimmed_mean"],
        "delta_trimmed_mean_ci_low": point["trimmed_mean_ci_low"],
        "delta_trimmed_mean_ci_high": point["trimmed_mean_ci_high"],
        "delta_median": point["median"],
        "delta_median_ci_low": point["median_ci_low"],
        "delta_median_ci_high": point["median_ci_high"],
        "median_episode_delta": float(np.median(episode_deltas)),
        "fraction_episode_positive": float((episode_deltas > 0.0).mean()),
        "fraction_pair_positive": float((pairs["delta"].to_numpy(dtype=np.float64) > 0.0).mean()),
        "p_superiority": probability_superiority(pairs["left_value"], pairs["right_value"]),
        "wilcoxon_statistic": stat,
        "wilcoxon_p_greater": pvalue,
        "abs_magnitude_gap_median": float(np.median(pairs["abs_magnitude_gap"])),
        "abs_magnitude_gap_p90": float(np.percentile(pairs["abs_magnitude_gap"], 90.0)),
    }


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
