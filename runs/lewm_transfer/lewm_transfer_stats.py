"""Statistics helpers for the LeWM latent-transfer diagnostic pilot."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


BOOTSTRAP_CI_LOW = 2.5
BOOTSTRAP_CI_HIGH = 97.5


def finite_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def trimmed_mean(values: Iterable[float] | np.ndarray, proportion_to_cut: float = 0.1) -> float:
    """Return a 10% trimmed mean by default.

    Adapted from le-wm/diagnostics/fetch_contact_compute/experiment_utils.py.
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

    Adapted from le-wm/diagnostics/fetch_contact_compute/experiment_utils.py.
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


def mean_dim_mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"shape mismatch: {left_arr.shape} != {right_arr.shape}")
    if left_arr.ndim < 2:
        raise ValueError("mean_dim_mse expects at least 2D arrays with rows and latent dimensions")
    return np.mean((left_arr - right_arr) ** 2, axis=1)


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
    out.update(
        {
            "delta_mean": out["left_mean"] - out["right_mean"]
            if len(left_arr) and len(right_arr)
            else float("nan"),
            "delta_median": out["left_median"] - out["right_median"]
            if len(left_arr) and len(right_arr)
            else float("nan"),
            "delta_trimmed_mean": out["left_trimmed_mean"] - out["right_trimmed_mean"]
            if len(left_arr) and len(right_arr)
            else float("nan"),
        }
    )
    return out


def percentile_ci(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    arr = finite_array([row.get(key, float("nan")) for row in rows])
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, BOOTSTRAP_CI_LOW)), float(np.percentile(arr, BOOTSTRAP_CI_HIGH))


def attach_cis(row: dict[str, object], boot_rows: list[dict[str, float]], metrics: Iterable[str]) -> dict[str, object]:
    out = dict(row)
    for metric in metrics:
        low, high = percentile_ci(boot_rows, metric)
        out[f"{metric}_ci_low"] = low
        out[f"{metric}_ci_high"] = high
    return out


def episode_clusters(df: pd.DataFrame, episode_col: str = "episode_ordinal") -> list[np.ndarray]:
    return [group.index.to_numpy() for _, group in df.groupby(episode_col, sort=False)]


def bootstrap_sample_indices(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    if not clusters:
        return np.asarray([], dtype=np.int64)
    selected = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in selected])


def cluster_bootstrap_group(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    value_col: str,
    n_bootstrap: int,
    seed: int,
    episode_col: str = "episode_ordinal",
) -> tuple[dict[str, float], list[dict[str, float]]]:
    point = describe_values(df.loc[mask, value_col].to_numpy(dtype=np.float64))
    clusters = [
        group.index.to_numpy()
        for _, group in df.groupby(episode_col, sort=False)
        if bool(mask.loc[group.index].any())
    ]
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = bootstrap_sample_indices(clusters, rng)
        if len(idx) == 0:
            boot_rows.append(describe_values([]))
            continue
        boot_mask = mask.loc[idx].to_numpy(dtype=bool)
        boot_rows.append(describe_values(df.loc[idx].loc[boot_mask, value_col].to_numpy(dtype=np.float64)))
    return point, boot_rows


def cluster_bootstrap_contrast(
    df: pd.DataFrame,
    *,
    left_mask: pd.Series,
    right_mask: pd.Series,
    value_col: str,
    n_bootstrap: int,
    seed: int,
    episode_col: str = "episode_ordinal",
) -> tuple[dict[str, float], list[dict[str, float]]]:
    point = contrast_values(
        df.loc[left_mask, value_col].to_numpy(dtype=np.float64),
        df.loc[right_mask, value_col].to_numpy(dtype=np.float64),
    )
    clusters = [
        group.index.to_numpy()
        for _, group in df.groupby(episode_col, sort=False)
        if bool(left_mask.loc[group.index].any()) or bool(right_mask.loc[group.index].any())
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
    return point, boot_rows


def wilcoxon_greater(values: Iterable[float] | np.ndarray) -> tuple[float, float]:
    arr = finite_array(values)
    if len(arr) == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(arr, alternative="greater", zero_method="zsplit", method="auto")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float("nan"), float("nan")


def make_regime_summary(
    df: pd.DataFrame,
    *,
    regimes: Iterable[str],
    value_cols: Iterable[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for regime_index, regime in enumerate(regimes):
        regime_mask = df["regime"].eq(regime)
        for value_index, value_col in enumerate(value_cols):
            point, boot = cluster_bootstrap_group(
                df,
                mask=regime_mask,
                value_col=value_col,
                n_bootstrap=n_bootstrap,
                seed=seed + 1000 * regime_index + value_index,
            )
            row = {
                "regime": regime,
                "metric": value_col,
                "n_episodes": int(df.loc[regime_mask, "episode_ordinal"].nunique()),
                **point,
            }
            rows.append(attach_cis(row, boot, ["mean", "median", "trimmed_mean"]))
    return pd.DataFrame(rows)


def make_regime_contrasts(
    df: pd.DataFrame,
    *,
    contrasts: Iterable[tuple[str, str, str]],
    value_cols: Iterable[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for contrast_index, (name, left, right) in enumerate(contrasts):
        left_mask = df["regime"].eq(left)
        right_mask = df["regime"].eq(right)
        for value_index, value_col in enumerate(value_cols):
            point, boot = cluster_bootstrap_contrast(
                df,
                left_mask=left_mask,
                right_mask=right_mask,
                value_col=value_col,
                n_bootstrap=n_bootstrap,
                seed=seed + 1000 * contrast_index + value_index,
            )
            row = {
                "contrast": name,
                "left_regime": left,
                "right_regime": right,
                "metric": value_col,
                **point,
            }
            rows.append(
                attach_cis(
                    row,
                    boot,
                    ["delta_mean", "delta_median", "delta_trimmed_mean", "p_superiority"],
                )
            )
    return pd.DataFrame(rows)


def add_phase_decile(df: pd.DataFrame, position_col: str = "normalized_position") -> pd.DataFrame:
    out = df.copy()
    decile = np.floor(out[position_col].to_numpy(dtype=np.float64) * 10.0).astype(int)
    out["phase_decile"] = np.clip(decile, 0, 9)
    return out


def decile_adjusted_contrast(df: pd.DataFrame, *, value_col: str, left: str, right: str) -> dict[str, float]:
    decile_rows = []
    for decile, group in df.groupby("phase_decile", sort=True):
        left_values = group.loc[group["regime"].eq(left), value_col].to_numpy(dtype=np.float64)
        right_values = group.loc[group["regime"].eq(right), value_col].to_numpy(dtype=np.float64)
        if len(left_values) == 0 or len(right_values) == 0:
            continue
        metric = contrast_values(left_values, right_values)
        weight = min(metric["left_n"], metric["right_n"])
        decile_rows.append((int(decile), weight, metric))
    if not decile_rows:
        return {
            "n_deciles": 0,
            "weight_sum": 0,
            "delta_mean": float("nan"),
            "delta_median": float("nan"),
            "delta_trimmed_mean": float("nan"),
            "p_superiority": float("nan"),
        }
    weight_sum = float(sum(weight for _, weight, _ in decile_rows))
    out = {"n_deciles": len(decile_rows), "weight_sum": weight_sum}
    for key in ["delta_mean", "delta_median", "delta_trimmed_mean", "p_superiority"]:
        out[key] = float(sum(weight * row[key] for _, weight, row in decile_rows) / weight_sum)
    return out


def make_phase_decile_contrasts(
    df: pd.DataFrame,
    *,
    value_col: str,
    left: str,
    right: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decile_rows = []
    for decile, group in df.groupby("phase_decile", sort=True):
        metrics = contrast_values(
            group.loc[group["regime"].eq(left), value_col].to_numpy(dtype=np.float64),
            group.loc[group["regime"].eq(right), value_col].to_numpy(dtype=np.float64),
        )
        decile_rows.append({"phase_decile": int(decile), "metric": value_col, **metrics})

    point = decile_adjusted_contrast(df, value_col=value_col, left=left, right=right)
    clusters = episode_clusters(df)
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = bootstrap_sample_indices(clusters, rng)
        boot_rows.append(decile_adjusted_contrast(df.loc[idx], value_col=value_col, left=left, right=right))
    adjusted = attach_cis(
        {"contrast": f"{left}_vs_{right}_phase_decile_adjusted", "metric": value_col, **point},
        boot_rows,
        ["delta_mean", "delta_median", "delta_trimmed_mean", "p_superiority"],
    )
    return pd.DataFrame(decile_rows), pd.DataFrame([adjusted])


def make_matched_position_pairs(
    df: pd.DataFrame,
    *,
    value_col: str,
    left: str = "interaction",
    right: str = "free",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    for episode, group in df.groupby("episode_ordinal", sort=False):
        for decile, decile_group in group.groupby("phase_decile", sort=True):
            left_rows = decile_group.loc[decile_group["regime"].eq(left)].sort_values("normalized_position")
            right_rows = decile_group.loc[decile_group["regime"].eq(right)].copy()
            if len(left_rows) == 0 or len(right_rows) == 0:
                continue
            available = set(right_rows.index.tolist())
            for left_index, left_row in left_rows.iterrows():
                if not available:
                    break
                candidates = right_rows.loc[list(available)]
                distances = np.abs(
                    candidates["normalized_position"].to_numpy(dtype=np.float64)
                    - float(left_row["normalized_position"])
                )
                right_index = candidates.index[int(np.argmin(distances))]
                right_row = df.loc[right_index]
                available.remove(right_index)
                pair_rows.append(
                    {
                        "episode_ordinal": int(episode),
                        "phase_decile": int(decile),
                        "left_row_id": int(left_row["row_id"]),
                        "right_row_id": int(right_row["row_id"]),
                        "left_value": float(left_row[value_col]),
                        "right_value": float(right_row[value_col]),
                        "delta": float(left_row[value_col] - right_row[value_col]),
                        "left_position": float(left_row["normalized_position"]),
                        "right_position": float(right_row["normalized_position"]),
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        summary = pd.DataFrame(
            [
                {
                    "metric": value_col,
                    "eligible_episodes": 0,
                    "n_pairs": 0,
                    "median_episode_delta": float("nan"),
                    "fraction_episode_positive": float("nan"),
                    "wilcoxon_statistic": float("nan"),
                    "wilcoxon_p_greater": float("nan"),
                }
            ]
        )
        return pairs, summary
    episode_deltas = pairs.groupby("episode_ordinal", sort=False)["delta"].median()
    stat, pvalue = wilcoxon_greater(episode_deltas.to_numpy(dtype=np.float64))
    summary = pd.DataFrame(
        [
            {
                "metric": value_col,
                "eligible_episodes": int(len(episode_deltas)),
                "n_pairs": int(len(pairs)),
                "median_episode_delta": float(np.median(episode_deltas)),
                "fraction_episode_positive": float((episode_deltas > 0.0).mean()),
                "wilcoxon_statistic": stat,
                "wilcoxon_p_greater": pvalue,
            }
        ]
    )
    return pairs, summary


def make_within_episode_pairs(
    df: pd.DataFrame,
    *,
    value_cols: Iterable[str],
    left: str = "interaction",
    right: str = "free",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = []
    summary_rows = []
    for value_col in value_cols:
        value_pair_rows = []
        for episode, group in df.groupby("episode_ordinal", sort=False):
            left_values = finite_array(group.loc[group["regime"].eq(left), value_col].to_numpy(dtype=np.float64))
            right_values = finite_array(group.loc[group["regime"].eq(right), value_col].to_numpy(dtype=np.float64))
            if len(left_values) == 0 or len(right_values) == 0:
                continue
            row = {
                "episode_ordinal": int(episode),
                "metric": value_col,
                "left_median": float(np.median(left_values)),
                "right_median": float(np.median(right_values)),
            }
            row["delta"] = row["left_median"] - row["right_median"]
            value_pair_rows.append(row)
            pair_rows.append(row)
        deltas = np.asarray([row["delta"] for row in value_pair_rows], dtype=np.float64)
        stat, pvalue = wilcoxon_greater(deltas)
        summary_rows.append(
            {
                "metric": value_col,
                "eligible_episodes": int(len(deltas)),
                "fraction_episode_positive": float((deltas > 0.0).mean()) if len(deltas) else float("nan"),
                "median_delta": float(np.median(deltas)) if len(deltas) else float("nan"),
                "wilcoxon_statistic": stat,
                "wilcoxon_p_greater": pvalue,
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


@dataclass(frozen=True)
class ResidualStructure:
    directional_consistency: float
    top1_variance: float
    top3_variance: float
    top5_variance: float
    top10_variance: float
    effective_rank: float
    n: int


def residual_structure(residuals: np.ndarray) -> ResidualStructure:
    arr = np.asarray(residuals, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return ResidualStructure(*(float("nan") for _ in range(6)), n=0)
    norms = np.linalg.norm(arr, axis=1)
    denom = float(np.mean(norms)) if len(norms) else float("nan")
    directional = float(np.linalg.norm(arr.mean(axis=0)) / denom) if denom and np.isfinite(denom) else float("nan")
    centered = arr - arr.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return ResidualStructure(directional, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), arr.shape[0])
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2
    total = float(variances.sum())
    if total <= 0.0:
        ratios = np.zeros_like(variances)
    else:
        ratios = variances / total

    def topk(k: int) -> float:
        return float(ratios[: min(k, len(ratios))].sum()) if len(ratios) else float("nan")

    positive = ratios[ratios > 0.0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else float("nan")
    return ResidualStructure(
        directional_consistency=directional,
        top1_variance=topk(1),
        top3_variance=topk(3),
        top5_variance=topk(5),
        top10_variance=topk(10),
        effective_rank=effective_rank,
        n=int(arr.shape[0]),
    )


def make_residual_structure_table(
    df: pd.DataFrame,
    residuals: np.ndarray,
    *,
    regimes: Iterable[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for regime_index, regime in enumerate(regimes):
        mask = df["regime"].eq(regime)
        row_ids = df.loc[mask, "row_id"].to_numpy(dtype=np.int64)
        point = residual_structure(residuals[row_ids]).__dict__
        clusters = [
            group.index.to_numpy()
            for _, group in df.groupby("episode_ordinal", sort=False)
            if bool(mask.loc[group.index].any())
        ]
        rng = np.random.default_rng(seed + 1000 * regime_index)
        boot_rows = []
        for _ in range(n_bootstrap):
            idx = bootstrap_sample_indices(clusters, rng)
            if len(idx) == 0:
                boot_rows.append(residual_structure(np.empty((0, residuals.shape[1]))).__dict__)
                continue
            boot_row_ids = df.loc[idx].loc[mask.loc[idx].to_numpy(dtype=bool), "row_id"].to_numpy(dtype=np.int64)
            boot_rows.append(residual_structure(residuals[boot_row_ids]).__dict__)
        base = {"regime": regime, **point}
        rows.append(
            attach_cis(
                base,
                boot_rows,
                [
                    "directional_consistency",
                    "top1_variance",
                    "top3_variance",
                    "top5_variance",
                    "top10_variance",
                    "effective_rank",
                ],
            )
        )
    return pd.DataFrame(rows)


def latent_isotropy(target_latents: np.ndarray) -> dict[str, float]:
    latents = np.asarray(target_latents, dtype=np.float64)
    if latents.ndim != 2 or latents.shape[0] < 2:
        return {
            "latent_dim": int(latents.shape[1]) if latents.ndim == 2 else 0,
            "condition_number": float("nan"),
            "per_dim_variance_min": float("nan"),
            "per_dim_variance_max": float("nan"),
            "per_dim_variance_spread": float("nan"),
        }
    cov = np.cov(latents, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    positive = eigvals[eigvals > 1e-12]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    variances = np.var(latents, axis=0, ddof=1)
    positive_var = variances[variances > 1e-12]
    spread = float(positive_var.max() / positive_var.min()) if len(positive_var) else float("inf")
    return {
        "latent_dim": int(latents.shape[1]),
        "condition_number": condition,
        "per_dim_variance_min": float(positive_var.min()) if len(positive_var) else float("nan"),
        "per_dim_variance_max": float(positive_var.max()) if len(positive_var) else float("nan"),
        "per_dim_variance_spread": spread,
    }


def whitening_matrix(calibration_latents: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    latents = np.asarray(calibration_latents, dtype=np.float64)
    mean = latents.mean(axis=0)
    cov = np.cov(latents, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    floor = max(float(eigvals.max()) * eps, eps)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, floor))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        transform = (eigvecs * inv_sqrt) @ eigvecs.T
    return mean, transform


def apply_whitened_mse(delta: np.ndarray, transform: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        whitened = np.asarray(delta, dtype=np.float64) @ transform
    return np.mean(whitened**2, axis=1)


def boolean_gate(value: bool, reason: str) -> dict[str, object]:
    return {"passed": bool(value), "reason": reason}


def summarize_gate_table(gates: dict[str, dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([{"gate": key, **value} for key, value in gates.items()])
