#!/usr/bin/env python3
"""Recompute Fetch Stage 1 after a persistence/state-delta control."""

from __future__ import annotations

import argparse
import json
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
    bias_variance_components,
    build_examples,
    find_records,
    load_records,
    masks_from_train_keys,
    probability_superiority,
    trimmed_mean,
)


DEFAULT_STAGE0_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/fullstate_ablation")
DEFAULT_ENSEMBLE_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv")
DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv_persistence_control")
SCALES = ["raw", "targetnorm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", action="append", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=8675309)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[fetch-persistence] {message}", flush=True)


def example_keys(meta: pd.DataFrame) -> pd.Series:
    return meta[["env_id", "episode_id", "step_idx"]].astype(str).agg("::".join, axis=1)


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index, dtype=bool)
    series = df[col]
    if series.dtype == object:
        return series.astype(str).str.lower().isin({"true", "1", "yes"})
    return series.fillna(False).astype(bool)


def persistence_mse(current_state: np.ndarray, target_state: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    current = np.asarray(current_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    if current.shape != target.shape:
        raise ValueError(f"current/target shape mismatch: {current.shape} vs {target.shape}")
    diff = target - current
    if scale is not None:
        denom = np.asarray(scale, dtype=np.float64).reshape(1, -1)
        denom = np.where(denom < 1e-6, 1.0, denom)
        diff = diff / denom
    return np.mean(diff**2, axis=1)


def current_state_from_examples(examples: ExampleSet, history_size: int) -> np.ndarray:
    indices = []
    for state_col in examples.state_cols:
        feature_name = f"hist_{history_size - 1}:{state_col}"
        try:
            indices.append(examples.feature_cols.index(feature_name))
        except ValueError as exc:
            raise ValueError(f"Missing current-state feature {feature_name}") from exc
    return examples.x[:, indices]


def load_stage_inputs(args: argparse.Namespace) -> tuple[ExampleSet, np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    split_path = args.stage0_dir / "split_keys.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing Stage 0 split file: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_keys = split.get("train_keys")
    if not train_keys:
        raise ValueError("Stage 0 split file does not contain train_keys")
    stage1_decision_path = args.ensemble_dir / "decision.json"
    stage1_decision = json.loads(stage1_decision_path.read_text(encoding="utf-8")) if stage1_decision_path.exists() else {}

    records = find_records(args.data_dir, args.records)
    df = load_records(records, history_size=args.history_size)
    examples = build_examples(df, history_size=args.history_size, include_shifted_fullstate=True)
    train_mask, val_mask = masks_from_train_keys(examples.meta, train_keys)
    return examples, train_mask, val_mask, split, stage1_decision


def verify_val_order(examples: ExampleSet, val_mask: np.ndarray, ensemble_dir: Path) -> pd.DataFrame:
    val_meta = examples.meta.loc[val_mask].reset_index(drop=True)
    saved = pd.read_csv(ensemble_dir / "val_decomposition.csv.gz")
    rebuilt_keys = example_keys(val_meta).tolist()
    saved_keys = example_keys(saved).tolist()
    if rebuilt_keys != saved_keys:
        for idx, (left, right) in enumerate(zip(rebuilt_keys, saved_keys)):
            if left != right:
                raise AssertionError(f"Val example order mismatch at {idx}: rebuilt={left} saved={right}")
        raise AssertionError("Val example order mismatch with different lengths")
    return saved


def add_scale_components(
    out: pd.DataFrame,
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    current_state: np.ndarray,
    scale_name: str,
    target_scale: np.ndarray | None,
) -> pd.DataFrame:
    if target_scale is None:
        scaled_predictions = predictions
        scaled_targets = targets
    else:
        denom = np.where(target_scale.reshape(1, 1, -1) < 1e-6, 1.0, target_scale.reshape(1, 1, -1))
        scaled_predictions = predictions / denom
        scaled_targets = targets / denom.reshape(1, -1)
    components = bias_variance_components(scaled_predictions, scaled_targets)
    persistence = persistence_mse(current_state, targets, target_scale)

    suffix = f"_{scale_name}"
    out[f"persistence_mse{suffix}"] = persistence
    out[f"bias2_mse{suffix}"] = components["bias2_mse"].to_numpy(dtype=np.float64)
    out[f"variance_mse{suffix}"] = components["variance_mse"].to_numpy(dtype=np.float64)
    out[f"heldout_error_mse{suffix}"] = components["heldout_error_mse"].to_numpy(dtype=np.float64)
    out[f"decomposition_residual_mse{suffix}"] = components["decomposition_residual_mse"].to_numpy(dtype=np.float64)
    out[f"excess_heldout_mse{suffix}"] = out[f"heldout_error_mse{suffix}"] - out[f"persistence_mse{suffix}"]
    out[f"excess_bias2_mse{suffix}"] = out[f"bias2_mse{suffix}"] - out[f"persistence_mse{suffix}"]
    out[f"excess_variance_mse{suffix}"] = out[f"variance_mse{suffix}"]
    out[f"excess_decomposition_residual_mse{suffix}"] = (
        out[f"excess_heldout_mse{suffix}"]
        - out[f"excess_bias2_mse{suffix}"]
        - out[f"excess_variance_mse{suffix}"]
    )
    return out


def value_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "trimmed_mean": trimmed_mean(arr),
    }


def group_stats(df: pd.DataFrame, scale: str) -> dict[str, float]:
    suffix = f"_{scale}"
    out = {"n": int(len(df))}
    for metric in [
        "heldout_error_mse",
        "persistence_mse",
        "excess_heldout_mse",
        "bias2_mse",
        "variance_mse",
        "excess_bias2_mse",
    ]:
        stats = value_stats(df[f"{metric}{suffix}"].to_numpy(dtype=np.float64))
        for key, value in stats.items():
            if key == "n":
                continue
            out[f"{metric}_{key}"] = value
    error_mean = out["excess_heldout_mse_mean"]
    if np.isfinite(error_mean) and abs(error_mean) > 1e-12:
        out["excess_bias_error_share"] = out["excess_bias2_mse_mean"] / error_mean
        out["variance_error_share"] = out["variance_mse_mean"] / error_mean
    else:
        out["excess_bias_error_share"] = float("nan")
        out["variance_error_share"] = float("nan")
    return out


def contrast_stats(df: pd.DataFrame, left_mask: pd.Series, right_mask: pd.Series, scale: str) -> dict[str, float]:
    suffix = f"_{scale}"
    left = df.loc[left_mask]
    right = df.loc[right_mask]
    left_stats = group_stats(left, scale)
    right_stats = group_stats(right, scale)
    out: dict[str, float] = {"left_n": int(len(left)), "right_n": int(len(right))}
    for metric in [
        "heldout_error_mse",
        "persistence_mse",
        "excess_heldout_mse",
        "bias2_mse",
        "variance_mse",
        "excess_bias2_mse",
    ]:
        for stat in ["mean", "median", "trimmed_mean"]:
            key = f"{metric}_{stat}"
            out[f"left_{key}"] = left_stats[key]
            out[f"right_{key}"] = right_stats[key]
            out[f"delta_{key}"] = left_stats[key] - right_stats[key]
    left_excess = left[f"excess_heldout_mse{suffix}"].to_numpy(dtype=np.float64)
    right_excess = right[f"excess_heldout_mse{suffix}"].to_numpy(dtype=np.float64)
    p_superiority = probability_superiority(left_excess, right_excess)
    out["excess_p_superiority"] = p_superiority
    out["excess_cliffs_delta"] = 2.0 * p_superiority - 1.0
    delta_excess = out["delta_excess_heldout_mse_mean"]
    if np.isfinite(delta_excess) and abs(delta_excess) > 1e-12:
        out["excess_bias_delta_share"] = out["delta_excess_bias2_mse_mean"] / delta_excess
        out["variance_delta_share"] = out["delta_variance_mse_mean"] / delta_excess
    else:
        out["excess_bias_delta_share"] = float("nan")
        out["variance_delta_share"] = float("nan")
    return out


def bootstrap_ci(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    arr = np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def attach_cis(row: dict[str, object], boot_rows: list[dict[str, float]], metrics: list[str]) -> dict[str, object]:
    out = dict(row)
    for metric in metrics:
        low, high = bootstrap_ci(boot_rows, metric)
        out[f"{metric}_ci_low"] = low
        out[f"{metric}_ci_high"] = high
    return out


def cluster_bootstrap_contrast(
    df: pd.DataFrame,
    *,
    left_mask: pd.Series,
    right_mask: pd.Series,
    scale: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    point = contrast_stats(df, left_mask, right_mask, scale)
    clusters = [
        group.index.to_numpy()
        for _, group in df.groupby(["env_id", "episode_id"], sort=False)
        if bool(left_mask.loc[group.index].any()) or bool(right_mask.loc[group.index].any())
    ]
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))])
        boot_df = df.loc[idx]
        boot_rows.append(
            contrast_stats(
                boot_df,
                pd.Series(left_mask.loc[idx].to_numpy(dtype=bool), index=boot_df.index),
                pd.Series(right_mask.loc[idx].to_numpy(dtype=bool), index=boot_df.index),
                scale,
            )
        )
    return point, boot_rows


def cluster_bootstrap_group(
    df: pd.DataFrame,
    *,
    group_mask: pd.Series,
    scale: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    point = group_stats(df.loc[group_mask], scale)
    clusters = [
        group.index.to_numpy()
        for _, group in df.groupby(["env_id", "episode_id"], sort=False)
        if bool(group_mask.loc[group.index].any())
    ]
    rng = np.random.default_rng(seed)
    boot_rows = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))])
        boot_df = df.loc[idx]
        boot_mask = pd.Series(group_mask.loc[idx].to_numpy(dtype=bool), index=boot_df.index)
        boot_rows.append(group_stats(boot_df.loc[boot_mask], scale))
    return point, boot_rows


def summarize(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contrast_rows = []
    regime_rows = []
    ci_rows = []
    contrast_ci_metrics = [
        "delta_heldout_error_mse_mean",
        "delta_persistence_mse_mean",
        "delta_excess_heldout_mse_mean",
        "delta_excess_heldout_mse_median",
        "delta_excess_heldout_mse_trimmed_mean",
        "delta_excess_bias2_mse_mean",
        "delta_variance_mse_mean",
        "excess_p_superiority",
        "excess_cliffs_delta",
        "excess_bias_delta_share",
        "variance_delta_share",
    ]
    group_ci_metrics = [
        "heldout_error_mse_mean",
        "persistence_mse_mean",
        "excess_heldout_mse_mean",
        "excess_heldout_mse_trimmed_mean",
        "excess_bias2_mse_mean",
        "variance_mse_mean",
    ]
    scopes = [("all", df)] + [(str(env_id), group.copy()) for env_id, group in df.groupby("env_id", sort=True)]
    for scale_index, scale in enumerate(SCALES):
        for scope_index, (scope, scope_df) in enumerate(scopes):
            interaction = bool_series(scope_df, "gripper_object_contact")
            free = scope_df["primary_regime"].astype(str) == "free_motion"
            if interaction.any() and free.any():
                point, boot = cluster_bootstrap_contrast(
                    scope_df,
                    left_mask=interaction,
                    right_mask=free,
                    scale=scale,
                    n_bootstrap=args.bootstrap_samples,
                    seed=args.bootstrap_seed + scale_index * 10000 + scope_index,
                )
                row = {
                    "scale": scale,
                    "scope": scope,
                    "contrast": "interaction_manipulation_contact_vs_free_motion",
                    "left_label": "gripper_object_contact",
                    "right_label": "primary_regime=free_motion",
                    **point,
                }
                row = attach_cis(row, boot, contrast_ci_metrics)
                contrast_rows.append(row)
                for metric in contrast_ci_metrics:
                    ci_rows.append(
                        {
                            "table": "primary_persistence_contrast",
                            "scale": scale,
                            "scope": scope,
                            "label": row["contrast"],
                            "metric": metric,
                            "ci_low": row[f"{metric}_ci_low"],
                            "ci_high": row[f"{metric}_ci_high"],
                        }
                    )

            for regime in PRIMARY_REGIMES:
                if regime == "boundary_or_invalid":
                    continue
                mask = scope_df["primary_regime"].astype(str) == regime
                if not mask.any():
                    continue
                point, boot = cluster_bootstrap_group(
                    scope_df,
                    group_mask=mask,
                    scale=scale,
                    n_bootstrap=args.bootstrap_samples,
                    seed=args.bootstrap_seed + 20000 + scale_index * 10000 + scope_index * 100 + len(regime_rows),
                )
                row = {
                    "scale": scale,
                    "scope": scope,
                    "primary_regime": regime,
                    **point,
                }
                row = attach_cis(row, boot, group_ci_metrics)
                regime_rows.append(row)
                for metric in group_ci_metrics:
                    ci_rows.append(
                        {
                            "table": "regime_persistence_summary",
                            "scale": scale,
                            "scope": scope,
                            "label": regime,
                            "metric": metric,
                            "ci_low": row[f"{metric}_ci_low"],
                            "ci_high": row[f"{metric}_ci_high"],
                        }
                    )
    return pd.DataFrame(contrast_rows), pd.DataFrame(regime_rows), pd.DataFrame(ci_rows)


def make_decision(contrast: pd.DataFrame) -> dict[str, object]:
    raw = contrast.loc[(contrast["scale"] == "raw") & (contrast["scope"] == "all")]
    norm = contrast.loc[(contrast["scale"] == "targetnorm") & (contrast["scope"] == "all")]
    if len(raw) == 0:
        return {"decision": "no_decision_missing_primary", "reason": "Missing raw all-scope primary contrast."}
    row = raw.iloc[0]
    raw_delta = float(row["delta_excess_heldout_mse_trimmed_mean"])
    raw_low = float(row["delta_excess_heldout_mse_trimmed_mean_ci_low"])
    raw_mean_delta = float(row["delta_excess_heldout_mse_mean"])
    raw_mean_low = float(row["delta_excess_heldout_mse_mean_ci_low"])
    raw_bias_share = float(row["excess_bias_delta_share"])
    raw_bias_low = float(row["delta_excess_bias2_mse_mean_ci_low"])
    raw_var_share = float(row["variance_delta_share"])
    norm_payload: dict[str, object] = {}
    norm_survives = None
    if len(norm):
        norm_row = norm.iloc[0]
        norm_payload = {
            "targetnorm_delta_excess_trimmed_mean": float(norm_row["delta_excess_heldout_mse_trimmed_mean"]),
            "targetnorm_delta_excess_trimmed_mean_ci": [
                float(norm_row["delta_excess_heldout_mse_trimmed_mean_ci_low"]),
                float(norm_row["delta_excess_heldout_mse_trimmed_mean_ci_high"]),
            ],
            "targetnorm_excess_bias_delta_share": float(norm_row["excess_bias_delta_share"]),
        }
        norm_survives = bool(
            norm_payload["targetnorm_delta_excess_trimmed_mean"] > 0.0
            and norm_payload["targetnorm_delta_excess_trimmed_mean_ci"][0] > 0.0
        )

    raw_trimmed_survives = bool(raw_delta > 0.0 and raw_low > 0.0)
    raw_mean_survives = bool(raw_mean_delta > 0.0 and raw_mean_low > 0.0)
    bias_dominated = bool(raw_mean_survives and raw_bias_share >= 0.65 and raw_bias_low > 0.0 and raw_var_share < 0.5)
    if raw_trimmed_survives and raw_mean_survives and bias_dominated and (norm_survives is not False):
        decision = "persistence_corrected_signal_survives"
        reason = "Interaction/free excess error remains positive after subtracting the state-delta persistence baseline, and the corrected contrast is still bias-dominated."
    elif raw_trimmed_survives:
        decision = "persistence_control_undercuts_stage1_bias_claim"
        reason = (
            "A robust raw trimmed excess-error contrast remains positive, but the mean-based excess contrast "
            "and/or target-normalized control do not support the original bias-dominance claim."
        )
    else:
        decision = "persistence_control_collapses_fetch_signal"
        reason = "Interaction/free excess error no longer has a positive clustered CI after subtracting the state-delta persistence baseline."
    return {
        "decision": decision,
        "reason": reason,
        "raw_delta_excess_trimmed_mean": raw_delta,
        "raw_delta_excess_trimmed_mean_ci": [
            raw_low,
            float(row["delta_excess_heldout_mse_trimmed_mean_ci_high"]),
        ],
        "raw_delta_excess_mean": float(row["delta_excess_heldout_mse_mean"]),
        "raw_delta_excess_mean_ci": [
            float(row["delta_excess_heldout_mse_mean_ci_low"]),
            float(row["delta_excess_heldout_mse_mean_ci_high"]),
        ],
        "raw_delta_excess_bias2_mean": float(row["delta_excess_bias2_mse_mean"]),
        "raw_delta_excess_bias2_mean_ci": [
            raw_bias_low,
            float(row["delta_excess_bias2_mse_mean_ci_high"]),
        ],
        "raw_delta_variance_mean": float(row["delta_variance_mse_mean"]),
        "raw_delta_variance_mean_ci": [
            float(row["delta_variance_mse_mean_ci_low"]),
            float(row["delta_variance_mse_mean_ci_high"]),
        ],
        "raw_excess_bias_delta_share": raw_bias_share,
        "raw_variance_delta_share": raw_var_share,
        "raw_excess_p_superiority": float(row["excess_p_superiority"]),
        "raw_excess_cliffs_delta": float(row["excess_cliffs_delta"]),
        **norm_payload,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_summary(path: Path, *, decision: dict[str, object], contrast: pd.DataFrame, regimes: pd.DataFrame, run_config: dict[str, object]) -> None:
    contrast_cols = [
        "scale",
        "scope",
        "left_n",
        "right_n",
        "left_heldout_error_mse_mean",
        "right_heldout_error_mse_mean",
        "delta_heldout_error_mse_mean",
        "left_persistence_mse_mean",
        "right_persistence_mse_mean",
        "delta_persistence_mse_mean",
        "left_excess_heldout_mse_trimmed_mean",
        "right_excess_heldout_mse_trimmed_mean",
        "delta_excess_heldout_mse_trimmed_mean",
        "delta_excess_heldout_mse_trimmed_mean_ci_low",
        "delta_excess_heldout_mse_trimmed_mean_ci_high",
        "delta_excess_bias2_mse_mean",
        "delta_excess_bias2_mse_mean_ci_low",
        "delta_excess_bias2_mse_mean_ci_high",
        "delta_variance_mse_mean",
        "delta_variance_mse_mean_ci_low",
        "delta_variance_mse_mean_ci_high",
        "excess_bias_delta_share",
        "variance_delta_share",
        "excess_p_superiority",
        "excess_cliffs_delta",
    ]
    regime_cols = [
        "scale",
        "scope",
        "primary_regime",
        "n",
        "heldout_error_mse_mean",
        "persistence_mse_mean",
        "excess_heldout_mse_trimmed_mean",
        "excess_heldout_mse_trimmed_mean_ci_low",
        "excess_heldout_mse_trimmed_mean_ci_high",
        "excess_bias2_mse_mean",
        "variance_mse_mean",
    ]
    lines = [
        "# Fetch Stage 1 Persistence-Controlled Recompute",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Reason: {decision['reason']}",
        f"- Bootstrap samples: `{run_config['bootstrap_samples']}` clustered over episodes",
        f"- Source ensemble: `{run_config['ensemble_dir']}`",
        "",
        "The persistence baseline predicts `next_state` with the current input state and uses mean squared state delta.",
        "Excess error is `model_error - persistence_error`. For the finite ensemble, corrected bias uses `bias2 - persistence_error`; member variance is unchanged.",
        "The raw scale audits the original Stage 1 result directly; targetnorm divides each state dimension by its train-split target standard deviation before computing MSE.",
        "",
        "## Primary Interaction vs Free",
        "",
        markdown_table(contrast.loc[contrast["scope"] == "all", [col for col in contrast_cols if col in contrast]]),
        "",
        "## Secondary MECE Regimes",
        "",
        markdown_table(regimes.loc[regimes["scope"] == "all", [col for col in regime_cols if col in regimes]]),
        "",
        "This is a diagnostic control on the Fetch state-space proxy, not evidence from the latent LeWM model.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and any(args.output_dir.glob("*")):
        raise FileExistsError(f"{args.output_dir} already has outputs; pass --force to overwrite")

    print_step("rebuilding matched full-state examples")
    examples, train_mask, val_mask, split_payload, stage1_decision = load_stage_inputs(args)
    saved_decomp = verify_val_order(examples, val_mask, args.ensemble_dir)
    current_state = current_state_from_examples(examples, args.history_size)
    current_val = current_state[val_mask]
    target_val = examples.y[val_mask]
    target_std = examples.y[train_mask].std(axis=0)
    target_std = np.where(target_std < 1e-6, 1.0, target_std)

    npz = np.load(args.ensemble_dir / "member_predictions_val.npz")
    predictions = np.asarray(npz["predictions"], dtype=np.float64)
    targets = np.asarray(npz["targets"], dtype=np.float64)
    if predictions.shape[1:] != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {predictions.shape} vs {targets.shape}")
    if targets.shape != target_val.shape or not np.allclose(targets, target_val, rtol=1e-5, atol=1e-6):
        raise AssertionError("Rebuilt targets do not match ensemble member_predictions_val.npz targets")

    out = saved_decomp.drop(
        columns=[col for col in saved_decomp.columns if col in {"bias2_mse", "variance_mse", "heldout_error_mse", "decomposition_residual_mse"}]
    ).copy()
    out = add_scale_components(
        out,
        predictions=predictions,
        targets=targets,
        current_state=current_val,
        scale_name="raw",
        target_scale=None,
    )
    out = add_scale_components(
        out,
        predictions=predictions,
        targets=targets,
        current_state=current_val,
        scale_name="targetnorm",
        target_scale=target_std,
    )
    raw_resid = float(np.max(np.abs(out["decomposition_residual_mse_raw"].to_numpy(dtype=np.float64))))
    raw_excess_resid = float(np.max(np.abs(out["excess_decomposition_residual_mse_raw"].to_numpy(dtype=np.float64))))
    print_step(f"rows={len(out)} raw_residual_abs_max={raw_resid:.3g} raw_excess_residual_abs_max={raw_excess_resid:.3g}")

    out.to_csv(args.output_dir / "val_persistence_decomposition.csv.gz", index=False, compression="gzip")
    contrast, regimes, cis = summarize(out, args)
    contrast.to_csv(args.output_dir / "primary_persistence_contrast.csv", index=False)
    regimes.to_csv(args.output_dir / "regime_persistence_summary.csv", index=False)
    cis.to_csv(args.output_dir / "bootstrap_cis.csv", index=False)
    decision = make_decision(contrast)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    run_config = {
        "records": [str(path) for path in find_records(args.data_dir, args.records)],
        "stage0_dir": str(args.stage0_dir),
        "ensemble_dir": str(args.ensemble_dir),
        "stage1_decision": stage1_decision,
        "history_size": args.history_size,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "member_predictions": str(args.ensemble_dir / "member_predictions_val.npz"),
        "split_train_episode_count": split_payload.get("train_episode_count"),
        "split_val_episode_count": split_payload.get("val_episode_count"),
        "targetnorm_source": "train-split target next_state standard deviation",
        "created_at_unix": time.time(),
        "raw_residual_abs_max": raw_resid,
        "raw_excess_residual_abs_max": raw_excess_resid,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    write_summary(args.output_dir / "summary.md", decision=decision, contrast=contrast, regimes=regimes, run_config=run_config)
    print_step(f"decision={decision['decision']}")
    print_step(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
