#!/usr/bin/env python3
"""Stage 2: recurrent re-read refinement and adaptive compute gate."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import PRIMARY_REGIMES
from experiment_utils import (
    DEFAULT_DATA_DIR,
    ExampleSet,
    build_examples,
    find_records,
    load_records,
    masks_from_train_keys,
    standardize,
    trimmed_mean,
)


DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/recurrent_refiner")
DEFAULT_STAGE0_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/fullstate_ablation")
DEFAULT_STAGE1_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv")
DEPTHS = [1, 2, 4, 8]
DEFAULT_GATE_FRACTIONS = {1: 1.0 / 6.0, 2: 0.25, 4: 1.0 / 3.0, 8: 0.25}
GATE_FEATURE_CLIP = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", action="append", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--split-keys", type=Path, default=None)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--context-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--gate-ridge", type=float, default=1e-2)
    parser.add_argument("--gate-target-budget", type=float, default=4.0)
    parser.add_argument("--gate-fractions", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=31415)
    parser.add_argument("--random-baseline-samples", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[fetch-recurrent] {message}", flush=True)


def load_gatekeeping(stage0_dir: Path, stage1_dir: Path, split_keys_path: Path | None) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    split_path = split_keys_path or stage0_dir / "split_keys.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split key file: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    stage0_decision_path = stage0_dir / "decision.json"
    stage1_decision_path = stage1_dir / "decision.json"
    stage0 = json.loads(stage0_decision_path.read_text(encoding="utf-8")) if stage0_decision_path.exists() else {}
    stage1 = json.loads(stage1_decision_path.read_text(encoding="utf-8")) if stage1_decision_path.exists() else {}
    if stage1 and not stage1.get("stage2_allowed", False):
        raise RuntimeError(f"Stage 1 did not allow Stage 2: {stage1}")
    return split, stage0, stage1


def choose_device(torch, requested: str):
    if requested == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def parse_gate_fractions(raw: str) -> dict[int, float]:
    if not raw.strip():
        return dict(DEFAULT_GATE_FRACTIONS)
    fractions: dict[int, float] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        key, value = part.split(":", 1)
        depth = int(key.strip())
        if depth not in DEPTHS:
            raise ValueError(f"Unsupported depth in gate fractions: {depth}")
        fractions[depth] = float(value.strip())
    missing = set(DEPTHS) - set(fractions)
    if missing:
        raise ValueError(f"Gate fractions missing depths: {sorted(missing)}")
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("Gate fractions must sum to a positive value")
    return {depth: fractions[depth] / total for depth in DEPTHS}


def gate_fraction_budget(fractions: dict[int, float]) -> float:
    return float(sum(depth * fractions[depth] for depth in DEPTHS))


def calibrate_depth_thresholds(scores: np.ndarray, fractions: dict[int, float]) -> dict[str, object]:
    """Fit score thresholds on train scores only; higher scores get more compute."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("Gate scores must be a 1D array")
    if not np.isfinite(scores).all():
        raise ValueError("Gate scores must be finite")
    cumulative = []
    running = 0.0
    for depth in DEPTHS[:-1]:
        running += fractions[depth]
        cumulative.append(min(max(running, 0.0), 1.0))
    thresholds = [float(np.quantile(scores, q)) for q in cumulative]
    assigned = assign_depths_from_thresholds(scores, thresholds)
    return {
        "thresholds": thresholds,
        "fractions": {str(depth): float(fractions[depth]) for depth in DEPTHS},
        "train_mean_depth": float(np.mean(assigned)),
    }


def assign_depths_from_thresholds(scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    bins = np.searchsorted(np.asarray(thresholds, dtype=np.float64), scores, side="right")
    return np.asarray([DEPTHS[int(idx)] for idx in bins], dtype=np.int64)


def uniform_curve_at_budget(mse_by_depth: pd.DataFrame, budget: float) -> np.ndarray:
    """Interpolate fixed-depth errors at a realized uniform compute budget."""

    if budget <= DEPTHS[0]:
        return mse_by_depth[str(DEPTHS[0])].to_numpy(dtype=np.float64)
    if budget >= DEPTHS[-1]:
        return mse_by_depth[str(DEPTHS[-1])].to_numpy(dtype=np.float64)
    lower = DEPTHS[0]
    upper = DEPTHS[-1]
    for left, right in zip(DEPTHS[:-1], DEPTHS[1:]):
        if left <= budget <= right:
            lower, upper = left, right
            break
    weight = (budget - lower) / float(upper - lower)
    return (
        (1.0 - weight) * mse_by_depth[str(lower)].to_numpy(dtype=np.float64)
        + weight * mse_by_depth[str(upper)].to_numpy(dtype=np.float64)
    )


def fit_ridge_gate(x_train_norm: np.ndarray, benefit: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    x = np.asarray(x_train_norm, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -GATE_FEATURE_CLIP, GATE_FEATURE_CLIP)
    y = np.asarray(benefit, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    center = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    z = (x - center) / scale
    z = np.clip(z, -GATE_FEATURE_CLIP, GATE_FEATURE_CLIP)
    x_aug = np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(ridge)
    reg[0, 0] = 0.0
    xtx = np.einsum("ni,nj->ij", x_aug, x_aug, optimize=True)
    xty = np.einsum("ni,n->i", x_aug, y, optimize=True)
    coef = np.linalg.solve(xtx + reg, xty)
    return {
        "intercept": np.asarray([coef[0]], dtype=np.float32),
        "coef": coef[1:].astype(np.float32),
        "center": center.reshape(-1).astype(np.float32),
        "scale": scale.reshape(-1).astype(np.float32),
    }


def score_ridge_gate(x_norm: np.ndarray, gate: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(x_norm, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -GATE_FEATURE_CLIP, GATE_FEATURE_CLIP)
    center = gate.get("center")
    scale = gate.get("scale")
    if center is not None and scale is not None:
        x = (x - center.astype(np.float64)) / scale.astype(np.float64)
    x = np.clip(x, -GATE_FEATURE_CLIP, GATE_FEATURE_CLIP)
    return np.einsum("ni,i->n", x, gate["coef"].astype(np.float64), optimize=True) + float(gate["intercept"][0])


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index, dtype=bool)
    series = df[col]
    if series.dtype == object:
        return series.astype(str).str.lower().isin({"true", "1", "yes"})
    return series.fillna(False).astype(bool)


def cluster_bootstrap_values(
    df: pd.DataFrame,
    *,
    value_fn,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    point = value_fn(df)
    clusters = [group.index.to_numpy() for _, group in df.groupby(["env_id", "episode_id"], sort=False)]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))])
        rows.append(value_fn(df.loc[idx]))
    return point, rows


def ci_from_rows(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    arr = np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def attach_ci(row: dict[str, object], boot_rows: list[dict[str, float]], keys: list[str]) -> dict[str, object]:
    out = dict(row)
    for key in keys:
        low, high = ci_from_rows(boot_rows, key)
        out[f"{key}_ci_low"] = low
        out[f"{key}_ci_high"] = high
    return out


def error_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "trimmed_mean": trimmed_mean(arr),
    }


def depth_curve_rows(predictions: pd.DataFrame, *, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []
    scopes = [("all", predictions)] + [(str(env_id), group.copy()) for env_id, group in predictions.groupby("env_id", sort=True)]
    for scope_index, (scope, scope_df) in enumerate(scopes):
        for regime in PRIMARY_REGIMES:
            if regime == "boundary_or_invalid":
                continue
            mask = scope_df["primary_regime"].astype(str) == regime
            if not mask.any():
                continue
            for depth in DEPTHS:
                depth_col = f"mse_k{depth}"

                def value_fn(frame: pd.DataFrame, col: str = depth_col) -> dict[str, float]:
                    return error_summary(frame[col].to_numpy(dtype=np.float64))

                point, boot = cluster_bootstrap_values(
                    scope_df.loc[mask].copy(),
                    value_fn=value_fn,
                    n_bootstrap=n_bootstrap,
                    seed=seed + scope_index * 1000 + len(curve_rows),
                )
                row = {
                    "scope": scope,
                    "primary_regime": regime,
                    "depth_k": depth,
                    **point,
                }
                row = attach_ci(row, boot, ["mean", "median", "trimmed_mean"])
                curve_rows.append(row)
                for metric in ["mean", "median", "trimmed_mean"]:
                    ci_rows.append(
                        {
                            "table": "depth_curve",
                            "scope": scope,
                            "primary_regime": regime,
                            "depth_k": depth,
                            "metric": metric,
                            "ci_low": row[f"{metric}_ci_low"],
                            "ci_high": row[f"{metric}_ci_high"],
                        }
                    )
    return pd.DataFrame(curve_rows), pd.DataFrame(ci_rows)


def fixed_depth_contrast_rows(predictions: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("all", predictions)] + [(str(env_id), group.copy()) for env_id, group in predictions.groupby("env_id", sort=True)]
    for scope_index, (scope, scope_df) in enumerate(scopes):
        interaction = bool_series(scope_df, "gripper_object_contact")
        free = scope_df["primary_regime"].astype(str) == "free_motion"
        if not interaction.any() or not free.any():
            continue
        for depth in DEPTHS:
            col = f"mse_k{depth}"

            def value_fn(frame: pd.DataFrame, depth_col: str = col) -> dict[str, float]:
                left = frame.loc[bool_series(frame, "gripper_object_contact"), depth_col].to_numpy(dtype=np.float64)
                right = frame.loc[frame["primary_regime"].astype(str) == "free_motion", depth_col].to_numpy(dtype=np.float64)
                return {
                    "left_mean": float(np.mean(left)),
                    "right_mean": float(np.mean(right)),
                    "delta_mean": float(np.mean(left) - np.mean(right)),
                    "left_trimmed_mean": trimmed_mean(left),
                    "right_trimmed_mean": trimmed_mean(right),
                    "delta_trimmed_mean": trimmed_mean(left) - trimmed_mean(right),
                }

            point, boot = cluster_bootstrap_values(
                scope_df.loc[interaction | free].copy(),
                value_fn=value_fn,
                n_bootstrap=n_bootstrap,
                seed=seed + 5000 + scope_index * 100 + depth,
            )
            row = {
                "scope": scope,
                "contrast": "interaction_manipulation_contact_vs_free_motion",
                "depth_k": depth,
                "left_n": int(interaction.sum()),
                "right_n": int(free.sum()),
                **point,
            }
            row = attach_ci(row, boot, ["delta_mean", "delta_trimmed_mean"])
            rows.append(row)
    return pd.DataFrame(rows)


def paired_delta_metrics(df: pd.DataFrame, left_col: str, right_col: str) -> dict[str, float]:
    delta = df[left_col].to_numpy(dtype=np.float64) - df[right_col].to_numpy(dtype=np.float64)
    return {
        "n": int(len(delta)),
        "left_mean": float(df[left_col].mean()),
        "right_mean": float(df[right_col].mean()),
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_trimmed_mean": trimmed_mean(delta),
    }


def paired_delta_with_ci(
    df: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    point, boot = cluster_bootstrap_values(
        df,
        value_fn=lambda frame: paired_delta_metrics(frame, left_col, right_col),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return attach_ci(point, boot, ["delta_mean", "delta_median", "delta_trimmed_mean"])


def make_adaptive_metrics(
    val_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("adaptive_vs_uniform_realized_budget_interp", "adaptive_mse", "uniform_realized_budget_mse", True),
        ("adaptive_vs_uniform_target_k4", "adaptive_mse", "mse_k4", False),
        ("adaptive_vs_permuted_same_budget", "adaptive_mse", "permuted_depth_mse", False),
        ("adaptive_vs_random_fraction_budget", "adaptive_mse", "random_fraction_mse", False),
    ]
    for idx, (name, left_col, right_col, headline) in enumerate(comparisons):
        row = {
            "comparison": name,
            "headline_metric": bool(headline),
            **paired_delta_with_ci(
                val_df,
                left_col=left_col,
                right_col=right_col,
                n_bootstrap=n_bootstrap,
                seed=seed + idx,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def random_depth_baselines(
    mse_by_depth: pd.DataFrame,
    adaptive_depth: np.ndarray,
    *,
    fractions: dict[int, float],
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    permuted_depth = adaptive_depth.copy()
    rng.shuffle(permuted_depth)
    random_depth = rng.choice(np.asarray(DEPTHS), size=len(adaptive_depth), p=np.asarray([fractions[d] for d in DEPTHS]))

    def gather(depths: np.ndarray) -> np.ndarray:
        out = np.empty(len(depths), dtype=np.float64)
        for depth in DEPTHS:
            mask = depths == depth
            out[mask] = mse_by_depth.loc[mask, str(depth)].to_numpy(dtype=np.float64)
        return out

    sample_rows = []
    for sample_index in range(samples):
        sample_depth = adaptive_depth.copy()
        rng.shuffle(sample_depth)
        sample_mse = gather(sample_depth)
        sample_rows.append(
            {
                "sample_index": sample_index,
                "baseline": "permuted_adaptive_depth",
                "mean_depth": float(np.mean(sample_depth)),
                "mean_mse": float(np.mean(sample_mse)),
                "trimmed_mean_mse": trimmed_mean(sample_mse),
            }
        )
    return gather(permuted_depth), gather(random_depth), pd.DataFrame(sample_rows)


def make_oracle_metrics(val_df: pd.DataFrame, *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    point = paired_delta_with_ci(
        val_df,
        left_col="oracle_best_mse",
        right_col="uniform_realized_budget_mse",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return pd.DataFrame(
        [
            {
                "comparison": "oracle_best_depth_vs_uniform_realized_budget",
                "diagnostic_only": True,
                "note": "Biased ceiling: chooses best depth using held-out target error.",
                **point,
            }
        ]
    )


def make_decision(curves: pd.DataFrame, adaptive: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, object]:
    all_curves = curves.loc[curves["scope"] == "all"].copy()
    interaction = all_curves.loc[all_curves["primary_regime"] == "gripper_object_contact"].set_index("depth_k")
    impact = all_curves.loc[all_curves["primary_regime"] == "impact_onset"].set_index("depth_k")
    free = all_curves.loc[all_curves["primary_regime"] == "free_motion"].set_index("depth_k")
    headline = adaptive.loc[adaptive["headline_metric"].astype(bool)].iloc[0]
    headline_high = float(headline.get("delta_mean_ci_high", float("nan")))
    headline_delta = float(headline.get("delta_mean", float("nan")))

    interaction_source = "gripper_object_contact"
    if len(impact) and (len(interaction) == 0 or float(impact.loc[1, "mean"]) > float(interaction.loc[1, "mean"])):
        interaction_source = "impact_onset"
        interaction = impact

    if len(interaction) and 1 in interaction.index and 8 in interaction.index:
        interaction_improvement = float(interaction.loc[1, "mean"] - interaction.loc[8, "mean"])
    else:
        interaction_improvement = float("nan")
    if len(free) and 1 in free.index and 8 in free.index:
        free_improvement = float(free.loc[1, "mean"] - free.loc[8, "mean"])
    else:
        free_improvement = float("nan")

    best_depths = (
        all_curves.sort_values(["scope", "primary_regime", "mean"])
        .groupby(["scope", "primary_regime"], as_index=False)
        .first()[["primary_regime", "depth_k", "mean"]]
        .to_dict("records")
    )
    adaptive_win = bool(np.isfinite(headline_high) and headline_high < 0.0)
    interaction_decreases = bool(np.isfinite(interaction_improvement) and interaction_improvement > 0.0)
    free_flatter = bool(
        np.isfinite(interaction_improvement)
        and np.isfinite(free_improvement)
        and abs(free_improvement) < max(abs(interaction_improvement), 1e-12)
    )
    if adaptive_win and interaction_decreases and free_flatter:
        decision = "adaptive_compute_supported_pilot"
        reason = (
            "The recurrent refiner shows lower interaction error with more compute and the train-calibrated "
            "gate beats realized-budget uniform compute on held-out episodes."
        )
    elif interaction_decreases and not adaptive_win:
        decision = "compute_reducible_but_gate_not_supported"
        reason = (
            "The fixed-depth curves show some compute reducibility, but the train-calibrated adaptive gate "
            "does not beat compute-matched uniform compute with a CI excluding zero."
        )
    else:
        decision = "no_adaptive_compute_signature"
        reason = "The recurrent refiner does not produce the desired held-out compute-reducibility signature."
    return {
        "decision": decision,
        "reason": reason,
        "adaptive_headline_delta_mean": headline_delta,
        "adaptive_headline_delta_mean_ci": [
            float(headline.get("delta_mean_ci_low", float("nan"))),
            headline_high,
        ],
        "adaptive_mean_depth": float(val_df["adaptive_depth"].mean()),
        "interaction_curve_source": interaction_source,
        "interaction_k1_minus_k8_mean": interaction_improvement,
        "free_k1_minus_k8_mean": free_improvement,
        "best_depth_by_regime": best_depths,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_summary(
    path: Path,
    *,
    decision: dict[str, object],
    curves: pd.DataFrame,
    adaptive: pd.DataFrame,
    fixed_contrasts: pd.DataFrame,
    oracle: pd.DataFrame,
    gate_config: dict[str, object],
) -> None:
    curve_cols = [
        "scope",
        "primary_regime",
        "depth_k",
        "mean",
        "mean_ci_low",
        "mean_ci_high",
        "trimmed_mean",
        "trimmed_mean_ci_low",
        "trimmed_mean_ci_high",
    ]
    adaptive_cols = [
        "comparison",
        "headline_metric",
        "left_mean",
        "right_mean",
        "delta_mean",
        "delta_mean_ci_low",
        "delta_mean_ci_high",
        "delta_trimmed_mean",
        "delta_trimmed_mean_ci_low",
        "delta_trimmed_mean_ci_high",
    ]
    contrast_cols = [
        "scope",
        "depth_k",
        "left_mean",
        "right_mean",
        "delta_mean",
        "delta_mean_ci_low",
        "delta_mean_ci_high",
        "delta_trimmed_mean",
        "delta_trimmed_mean_ci_low",
        "delta_trimmed_mean_ci_high",
    ]
    lines = [
        "# Fetch Stage 2 Recurrent Refiner",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Reason: {decision['reason']}",
        f"- Adaptive mean depth on val: `{decision['adaptive_mean_depth']:.6g}`",
        f"- Gate train mean depth: `{gate_config['train_mean_depth']:.6g}`",
        f"- Gate thresholds calibrated on train only: `{gate_config['thresholds']}`",
        "",
        "The gate uses only prediction-time feature vectors from the same input/history/action/full-state features used by the model.",
        "It does not use next-state-derived regime labels or object-velocity discontinuities.",
        "",
        "## Error vs Compute",
        "",
        markdown_table(curves.loc[curves["scope"] == "all", [col for col in curve_cols if col in curves]]),
        "",
        "## Fixed-Depth Interaction Contrast",
        "",
        markdown_table(fixed_contrasts.loc[fixed_contrasts["scope"] == "all", [col for col in contrast_cols if col in fixed_contrasts]]),
        "",
        "## Adaptive Gate",
        "",
        markdown_table(adaptive[[col for col in adaptive_cols if col in adaptive]]),
        "",
        "## Oracle Diagnostic Upper Bound",
        "",
        markdown_table(oracle),
        "",
        "Oracle metrics are biased ceilings only and are not used as the main effect size.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def import_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def build_recurrent_refiner(nn, input_dim: int, target_dim: int, context_dim: int, hidden_dim: int, max_depth: int):
    import torch

    class ReReadRefiner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_depth = max_depth
            self.context = nn.Sequential(
                nn.Linear(input_dim, context_dim),
                nn.ReLU(),
                nn.Linear(context_dim, context_dim),
                nn.ReLU(),
            )
            self.init_hidden = nn.Linear(context_dim, hidden_dim)
            self.init_pred = nn.Linear(context_dim, target_dim)
            self.cell = nn.GRUCell(context_dim + target_dim, hidden_dim)
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_dim + context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, target_dim),
            )

        def forward(self, x, return_depths: list[int]):
            context = self.context(x)
            hidden = torch.tanh(self.init_hidden(context))
            pred = self.init_pred(context)
            outputs = {}
            wanted = set(return_depths)
            for depth in range(1, self.max_depth + 1):
                hidden = self.cell(torch.cat([context, pred], dim=-1), hidden)
                pred = self.pred_head(torch.cat([hidden, context], dim=-1))
                if depth in wanted:
                    outputs[depth] = pred
            return outputs

    return ReReadRefiner()


def train_refiner(
    *,
    examples: ExampleSet,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    args: argparse.Namespace,
):
    torch, nn, DataLoader, TensorDataset = import_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(torch, args.device)
    print_step(f"device={device} seed={args.seed}")

    x_norm, x_mean, x_std = standardize(examples.x[train_mask], examples.x)
    y_norm, y_mean, y_std = standardize(examples.y[train_mask], examples.y)
    x_train = torch.as_tensor(x_norm[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(y_norm[train_mask], dtype=torch.float32)
    x_val = torch.as_tensor(x_norm[val_mask], dtype=torch.float32)
    y_val = torch.as_tensor(y_norm[val_mask], dtype=torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = build_recurrent_refiner(
        nn,
        input_dim=examples.x.shape[1],
        target_dim=examples.y.shape[1],
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        max_depth=max(DEPTHS),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb, DEPTHS)
            loss = torch.stack([torch.mean((outputs[depth] - yb) ** 2) for depth in DEPTHS]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            outputs = model(x_val.to(device), DEPTHS)
            val_losses = {f"val_loss_k{depth}": float(torch.mean((outputs[depth] - y_val.to(device)) ** 2).detach().cpu()) for depth in DEPTHS}
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            **val_losses,
        }
        history.append(row)
        if (epoch + 1) == 1 or (epoch + 1) % max(1, args.epochs // 5) == 0:
            loss_text = " ".join(f"k{depth}={row[f'val_loss_k{depth}']:.6g}" for depth in DEPTHS)
            print_step(f"epoch={epoch + 1} train_loss={row['train_loss']:.6g} {loss_text}")
    normalizers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, normalizers, pd.DataFrame(history), device, torch, x_norm


def predict_refiner_depths(
    *,
    model,
    normalizers: dict[str, np.ndarray],
    x: np.ndarray,
    device,
    torch,
    chunk_size: int = 8192,
) -> dict[int, np.ndarray]:
    x_norm = (x - normalizers["x_mean"]) / normalizers["x_std"]
    chunks: dict[int, list[np.ndarray]] = {depth: [] for depth in DEPTHS}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_norm), chunk_size):
            xb = torch.as_tensor(x_norm[start : start + chunk_size], dtype=torch.float32, device=device)
            outputs = model(xb, DEPTHS)
            for depth in DEPTHS:
                pred_norm = outputs[depth].detach().cpu().numpy()
                pred = pred_norm * normalizers["y_std"] + normalizers["y_mean"]
                chunks[depth].append(pred.astype(np.float32))
    return {depth: np.concatenate(parts, axis=0) for depth, parts in chunks.items()}


def mse_frame(predictions: dict[int, np.ndarray], targets: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({str(depth): np.mean((predictions[depth] - targets) ** 2, axis=1) for depth in DEPTHS})


def build_prediction_frame(meta: pd.DataFrame, mse_by_depth: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = meta.reset_index(drop=True).copy()
    out["split"] = split_name
    for depth in DEPTHS:
        out[f"mse_k{depth}"] = mse_by_depth[str(depth)].to_numpy(dtype=np.float64)
    out["oracle_k"] = mse_by_depth.astype(float).idxmin(axis=1).astype(int).to_numpy()
    out["oracle_best_mse"] = mse_by_depth.min(axis=1).to_numpy(dtype=np.float64)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and any(args.output_dir.glob("*")):
        raise FileExistsError(f"{args.output_dir} already has outputs; pass --force to overwrite")

    split_payload, stage0_decision, stage1_decision = load_gatekeeping(args.stage0_dir, args.stage1_dir, args.split_keys)
    train_keys = split_payload.get("train_keys")
    if not train_keys:
        raise ValueError("Stage 2 requires train_keys from the Stage 0 split file")

    records = find_records(args.data_dir, args.records)
    print_step(f"loading {len(records)} record file(s)")
    df = load_records(records, history_size=args.history_size)
    examples = build_examples(df, history_size=args.history_size, include_shifted_fullstate=True)
    train_mask, val_mask = masks_from_train_keys(examples.meta, train_keys)
    print_step(f"examples={len(examples.x)} train={int(train_mask.sum())} val={int(val_mask.sum())}")

    model, normalizers, history, device, torch, x_norm = train_refiner(
        examples=examples,
        train_mask=train_mask,
        val_mask=val_mask,
        args=args,
    )
    history.to_csv(args.output_dir / "training_history.csv", index=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "normalizers": normalizers,
        "feature_cols": examples.feature_cols,
        "target_cols": examples.target_cols,
        "state_cols": examples.state_cols,
        "action_cols": examples.action_cols,
        "fullstate_cols": examples.fullstate_cols,
        "history_size": args.history_size,
        "context_dim": args.context_dim,
        "hidden_dim": args.hidden_dim,
        "depths": DEPTHS,
        "seed": args.seed,
        "created_at_unix": time.time(),
    }
    torch.save(checkpoint, args.output_dir / "recurrent_refiner.pt")

    all_predictions = predict_refiner_depths(
        model=model,
        normalizers=normalizers,
        x=examples.x,
        device=device,
        torch=torch,
    )
    train_mse = mse_frame({depth: values[train_mask] for depth, values in all_predictions.items()}, examples.y[train_mask])
    val_mse = mse_frame({depth: values[val_mask] for depth, values in all_predictions.items()}, examples.y[val_mask])
    train_pred_df = build_prediction_frame(examples.meta.loc[train_mask], train_mse, "train")
    val_pred_df = build_prediction_frame(examples.meta.loc[val_mask], val_mse, "val")

    gate_fractions = parse_gate_fractions(args.gate_fractions)
    fraction_budget = gate_fraction_budget(gate_fractions)
    if abs(fraction_budget - args.gate_target_budget) > 1e-6:
        print_step(
            f"gate fraction budget={fraction_budget:.6g} differs from requested target={args.gate_target_budget:.6g}; using fractions"
        )
    train_benefit = train_mse["1"].to_numpy(dtype=np.float64) - train_mse["8"].to_numpy(dtype=np.float64)
    gate = fit_ridge_gate(x_norm[train_mask], train_benefit, args.gate_ridge)
    train_scores = score_ridge_gate(x_norm[train_mask], gate)
    val_scores = score_ridge_gate(x_norm[val_mask], gate)
    gate_config = calibrate_depth_thresholds(train_scores, gate_fractions)
    train_depths = assign_depths_from_thresholds(train_scores, gate_config["thresholds"])
    val_depths = assign_depths_from_thresholds(val_scores, gate_config["thresholds"])
    gate_config.update(
        {
            "target_budget": float(args.gate_target_budget),
            "fraction_budget": fraction_budget,
            "val_mean_depth": float(np.mean(val_depths)),
            "ridge": float(args.gate_ridge),
            "score_source": "ridge prediction of train-only k1_minus_k8 benefit from prediction-time feature vector",
            "forbidden_features": [
                "next_state_*",
                "object_velocity_discontinuity",
                "primary_regime",
                "impact_onset",
                "any held-out error or target-derived quantity at validation time",
            ],
        }
    )

    def gather_depth_mse(mse_by_depth: pd.DataFrame, depths: np.ndarray) -> np.ndarray:
        out = np.empty(len(depths), dtype=np.float64)
        for depth in DEPTHS:
            mask = depths == depth
            out[mask] = mse_by_depth.loc[mask, str(depth)].to_numpy(dtype=np.float64)
        return out

    train_pred_df["gate_score"] = train_scores
    train_pred_df["adaptive_depth"] = train_depths
    train_pred_df["adaptive_mse"] = gather_depth_mse(train_mse, train_depths)
    val_pred_df["gate_score"] = val_scores
    val_pred_df["adaptive_depth"] = val_depths
    val_pred_df["adaptive_mse"] = gather_depth_mse(val_mse, val_depths)
    val_pred_df["uniform_realized_budget_mse"] = uniform_curve_at_budget(val_mse, float(np.mean(val_depths)))
    val_pred_df["uniform_target_budget_mse"] = uniform_curve_at_budget(val_mse, args.gate_target_budget)
    permuted_mse, random_fraction_mse, random_samples = random_depth_baselines(
        val_mse,
        val_depths,
        fractions=gate_fractions,
        samples=args.random_baseline_samples,
        seed=args.bootstrap_seed + 7000,
    )
    val_pred_df["permuted_depth_mse"] = permuted_mse
    val_pred_df["random_fraction_mse"] = random_fraction_mse

    train_pred_df.to_csv(args.output_dir / "train_predictions_by_depth.csv.gz", index=False, compression="gzip")
    val_pred_df.to_csv(args.output_dir / "val_predictions_by_depth.csv.gz", index=False, compression="gzip")
    random_samples.to_csv(args.output_dir / "random_depth_permutation_samples.csv", index=False)
    np.savez_compressed(
        args.output_dir / "predictions_by_depth_val.npz",
        **{f"pred_k{depth}": all_predictions[depth][val_mask].astype(np.float32) for depth in DEPTHS},
        targets=examples.y[val_mask].astype(np.float32),
        adaptive_depth=val_depths.astype(np.int64),
    )

    curves, curve_cis = depth_curve_rows(val_pred_df, n_bootstrap=args.bootstrap_samples, seed=args.bootstrap_seed)
    fixed_contrasts = fixed_depth_contrast_rows(val_pred_df, n_bootstrap=args.bootstrap_samples, seed=args.bootstrap_seed + 11000)
    adaptive_metrics = make_adaptive_metrics(val_pred_df, n_bootstrap=args.bootstrap_samples, seed=args.bootstrap_seed + 22000)
    oracle_metrics = make_oracle_metrics(val_pred_df, n_bootstrap=args.bootstrap_samples, seed=args.bootstrap_seed + 33000)
    decision = make_decision(curves, adaptive_metrics, val_pred_df)

    curves.to_csv(args.output_dir / "depth_curve_by_regime.csv", index=False)
    fixed_contrasts.to_csv(args.output_dir / "fixed_depth_interaction_contrasts.csv", index=False)
    adaptive_metrics.to_csv(args.output_dir / "adaptive_gate_metrics.csv", index=False)
    oracle_metrics.to_csv(args.output_dir / "oracle_upper_bound_metrics.csv", index=False)
    curve_cis.to_csv(args.output_dir / "bootstrap_cis.csv", index=False)
    (args.output_dir / "gate_config.json").write_text(json.dumps(gate_config, indent=2), encoding="utf-8")
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    run_config = {
        "records": [str(path) for path in records],
        "stage0_dir": str(args.stage0_dir),
        "stage0_decision": stage0_decision,
        "stage1_dir": str(args.stage1_dir),
        "stage1_decision": stage1_decision,
        "history_size": args.history_size,
        "context_dim": args.context_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "depths": DEPTHS,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "random_baseline_samples": args.random_baseline_samples,
        "feature_source": "shifted previous-row qpos/qvel plus observation/action/history",
        "gate_feature_source": "prediction-time feature vector only; thresholds calibrated on train split only",
        "created_at_unix": time.time(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    write_summary(
        args.output_dir / "summary.md",
        decision=decision,
        curves=curves,
        adaptive=adaptive_metrics,
        fixed_contrasts=fixed_contrasts,
        oracle=oracle_metrics,
        gate_config=gate_config,
    )
    print_step(f"decision={decision['decision']}")
    print_step(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
