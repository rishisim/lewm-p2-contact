#!/usr/bin/env python3
"""Run the metric-hardening diagnostic from existing artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRANSFER_ROOT = ROOT / "runs" / "lewm_transfer"
FETCH_ROOT = ROOT / "le-wm" / "diagnostics" / "fetch_contact_compute"
for path in [HERE, TRANSFER_ROOT, FETCH_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_utils import build_examples, load_records, masks_from_train_keys, standardize
from lewm_transfer_stats import add_phase_decile, apply_whitened_mse, make_matched_position_pairs, make_phase_decile_contrasts, whitening_matrix
from metric_hardening_utils import (
    clustered_contrast_summary,
    clustered_group_summary,
    double_matched_pairs,
    local_sensitivity_frame,
    paired_delta_summary,
    per_dim_std,
    scaled_mse,
    write_csv,
)


DEFAULT_OUTPUT_DIR = ROOT / "runs" / "metric_hardening"
CUBE_INPUT_DIR = ROOT / "runs" / "lewm_transfer" / "cube"
CUBE_RELABEL_DIR = CUBE_INPUT_DIR / "relabel_motion"
FETCH_ENSEMBLE_DIR = FETCH_ROOT / "runs" / "ensemble_bv"
FETCH_PERSISTENCE_DIR = FETCH_ROOT / "runs" / "ensemble_bv_persistence_control"
FETCH_FULLSTATE_DIR = FETCH_ROOT / "runs" / "fullstate_ablation"
FETCH_DATA_PATHS = [
    FETCH_ROOT / "data" / "swm__FetchPushDense-v3" / "records.csv.gz",
    FETCH_ROOT / "data" / "swm__FetchSlideDense-v3" / "records.csv.gz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=260727)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--neighbor-k-min", type=int, default=5)
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[metric-hardening] {message}", flush=True)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "(empty)"
    present = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(present) + " |", "| " + " | ".join(["---"] * len(present)) + " |"]
    for _, row in df[present].head(max_rows).iterrows():
        values = []
        for col in present:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_cube_records() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    records = pd.read_csv(CUBE_RELABEL_DIR / "step_records_relabel_motion.csv")
    if "phase_decile" not in records.columns:
        records = add_phase_decile(records)
    npz = np.load(CUBE_INPUT_DIR / "residuals.npz")
    arrays = {key: np.asarray(npz[key]) for key in npz.files}
    row_ids = arrays["row_id"]
    if not np.array_equal(row_ids, np.arange(len(row_ids))):
        raise AssertionError("Cube residuals.npz row_id is not contiguous; update row mapping before analysis")
    if records["row_id"].max() >= len(row_ids):
        raise AssertionError("Cube records reference row_id outside residuals.npz")
    return records, arrays


def add_cube_metric_columns(records: pd.DataFrame, arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, np.ndarray | float]]:
    out = records.copy()
    target = arrays["target_latents"].astype(np.float64)
    pred = arrays["pred_latents"].astype(np.float64)
    prev = arrays["prev_latents"].astype(np.float64)
    residuals = arrays["residuals"].astype(np.float64)
    calibration_ids = out.loc[out["split"].eq("calibration"), "row_id"].to_numpy(dtype=np.int64)
    target_mean, transform = whitening_matrix(target[calibration_ids])
    target_std = per_dim_std(target[calibration_ids])

    out["mse_model_whitened_recomputed"] = apply_whitened_mse(pred - target, transform)
    out["mse_persistence_whitened_recomputed"] = apply_whitened_mse(prev - target, transform)
    out["excess_whitened_recomputed"] = out["mse_model_whitened_recomputed"] - out["mse_persistence_whitened_recomputed"]
    compare_cols = [
        ("mse_model_whitened", "mse_model_whitened_recomputed"),
        ("mse_persistence_whitened", "mse_persistence_whitened_recomputed"),
        ("excess_whitened", "excess_whitened_recomputed"),
    ]
    max_abs = 0.0
    for existing, recomputed in compare_cols:
        diff = np.nanmax(np.abs(out[existing].to_numpy(dtype=np.float64) - out[recomputed].to_numpy(dtype=np.float64)))
        max_abs = max(max_abs, float(diff))
    if max_abs > 1e-4:
        raise AssertionError(f"Cube whitened recompute disagrees with saved columns; max_abs_diff={max_abs:.6g}")

    out["mse_model_targetnorm"] = scaled_mse(pred - target, target_std)
    out["mse_persistence_targetnorm"] = scaled_mse(prev - target, target_std)
    out["excess_targetnorm"] = out["mse_model_targetnorm"] - out["mse_persistence_targetnorm"]
    out["persistence_mag_whitened"] = np.sqrt(np.maximum(out["mse_persistence_whitened_recomputed"].to_numpy(dtype=np.float64), 0.0))
    out["persistence_mag_targetnorm"] = np.sqrt(np.maximum(out["mse_persistence_targetnorm"].to_numpy(dtype=np.float64), 0.0))
    transforms = {
        "target_mean": target_mean,
        "whitening": transform,
        "target_std": target_std,
        "whitened_recompute_max_abs_diff": max_abs,
    }
    return out, transforms


def metric_pass(row: pd.Series) -> bool:
    return bool(
        len(row)
        and np.isfinite(float(row["delta_trimmed_mean"]))
        and float(row["delta_trimmed_mean"]) > 0.0
        and float(row["delta_trimmed_mean_ci_low"]) > 0.0
    )


def matched_pass(row: pd.Series) -> bool:
    return bool(
        len(row)
        and np.isfinite(float(row["median_episode_delta"]))
        and float(row["median_episode_delta"]) > 0.0
        and float(row["fraction_episode_positive"]) > 0.5
        and float(row["wilcoxon_p_greater"]) < 0.05
    )


def run_cube_part1(
    records: pd.DataFrame,
    *,
    output_dir: Path,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    heldout = records.loc[records["split"].eq("heldout")].copy()
    contrast_rows = []
    position_pair_frames = []
    double_pair_frames = []
    pass_flags: dict[str, object] = {}

    metric_specs = [
        ("whitened", "excess_whitened_recomputed", "persistence_mag_whitened"),
        ("targetnorm", "excess_targetnorm", "persistence_mag_targetnorm"),
    ]
    for idx, (scale, value_col, magnitude_col) in enumerate(metric_specs):
        _, phase_adjusted = make_phase_decile_contrasts(
            heldout,
            value_col=value_col,
            left="interaction",
            right="transport_free",
            n_bootstrap=n_bootstrap,
            seed=seed + 1000 * idx,
        )
        phase_row = phase_adjusted.iloc[0].to_dict()
        phase_row.update({"scale": scale, "metric": value_col, "test": "phase_decile_adjusted"})
        contrast_rows.append(phase_row)

        pairs, matched_summary = make_matched_position_pairs(
            heldout,
            value_col=value_col,
            left="interaction",
            right="transport_free",
        )
        matched_row = matched_summary.iloc[0].to_dict()
        matched_row.update({"scale": scale, "metric": value_col, "test": "matched_position"})
        contrast_rows.append(matched_row)
        if not pairs.empty:
            pair_copy = pairs.copy()
            pair_copy["scale"] = scale
            pair_copy["metric"] = value_col
            position_pair_frames.append(pair_copy)

        double_pairs, edges = double_matched_pairs(
            heldout,
            value_col=value_col,
            magnitude_col=magnitude_col,
            left="interaction",
            right="transport_free",
        )
        double_summary = paired_delta_summary(double_pairs, n_bootstrap=n_bootstrap, seed=seed + 2000 + idx)
        double_summary.update(
            {
                "scale": scale,
                "metric": value_col,
                "test": "phase_persistence_magnitude_double_match",
                "magnitude_col": magnitude_col,
                "n_magnitude_bins": max(0, len(edges) - 1),
            }
        )
        contrast_rows.append(double_summary)
        if not double_pairs.empty:
            double_copy = double_pairs.copy()
            double_copy["scale"] = scale
            double_copy["metric"] = value_col
            double_copy["magnitude_col"] = magnitude_col
            double_pair_frames.append(double_copy)

        phase_ok = metric_pass(pd.Series(phase_row))
        matched_ok = matched_pass(pd.Series(matched_row))
        double_ok = metric_pass(pd.Series(double_summary)) and matched_pass(pd.Series(double_summary))
        pass_flags[f"{scale}_phase_pass"] = phase_ok
        pass_flags[f"{scale}_matched_position_pass"] = matched_ok
        pass_flags[f"{scale}_double_match_pass"] = double_ok
        pass_flags[f"{scale}_metric_pass"] = bool(phase_ok and matched_ok)

    part1_all = bool(
        pass_flags["whitened_metric_pass"]
        and pass_flags["targetnorm_metric_pass"]
        and pass_flags["whitened_double_match_pass"]
        and pass_flags["targetnorm_double_match_pass"]
    )
    pass_flags["part1_survives_all_controls"] = part1_all

    contrasts = pd.DataFrame(contrast_rows)
    double_pairs_all = pd.concat(double_pair_frames, ignore_index=True) if double_pair_frames else pd.DataFrame()
    position_pairs_all = pd.concat(position_pair_frames, ignore_index=True) if position_pair_frames else pd.DataFrame()
    write_csv(output_dir / "cube" / "part1_metric_contrasts.csv", contrasts)
    write_csv(output_dir / "cube" / "matched_position_pairs.csv", position_pairs_all)
    write_csv(output_dir / "cube" / "double_matched_pairs.csv", double_pairs_all)
    return contrasts, double_pairs_all, pass_flags


def make_summary_tables(
    frame: pd.DataFrame,
    *,
    dataset: str,
    right_regime: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_cols = ["neighbor_radius", "local_expansion", "model_sensitivity", "residual_norm"]
    valid = frame.loc[frame["valid_neighbors"].astype(bool)].copy()
    group_rows = []
    contrast_rows = []
    for scale_index, scale in enumerate(valid["scale"].drop_duplicates().tolist()):
        scale_df = valid.loc[valid["scale"].eq(scale)].copy()
        for regime_index, regime in enumerate(scale_df["regime"].drop_duplicates().tolist()):
            mask = scale_df["regime"].eq(regime)
            for value_index, value_col in enumerate(value_cols):
                row = clustered_group_summary(
                    scale_df,
                    mask=mask,
                    value_col=value_col,
                    n_bootstrap=n_bootstrap,
                    seed=seed + scale_index * 10000 + regime_index * 100 + value_index,
                )
                row.update(
                    {
                        "dataset": dataset,
                        "scale": scale,
                        "regime": regime,
                        "metric": value_col,
                        "n_episodes": int(scale_df.loc[mask, "episode_ordinal"].nunique()),
                    }
                )
                group_rows.append(row)
        left_mask = scale_df["regime"].eq("interaction")
        right_mask = scale_df["regime"].eq(right_regime)
        for value_index, value_col in enumerate(value_cols):
            row = clustered_contrast_summary(
                scale_df,
                left_mask=left_mask,
                right_mask=right_mask,
                value_col=value_col,
                n_bootstrap=n_bootstrap,
                seed=seed + 50000 + scale_index * 1000 + value_index,
            )
            row.update(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "contrast": f"interaction_vs_{right_regime}",
                    "metric": value_col,
                }
            )
            contrast_rows.append(row)
    return pd.DataFrame(group_rows), pd.DataFrame(contrast_rows)


def run_cube_sensitivity(
    records: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    transforms: dict[str, np.ndarray | float],
    *,
    output_dir: Path,
    n_bootstrap: int,
    seed: int,
    k: int,
    k_min: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout = records.loc[records["split"].eq("heldout")].copy()
    row_ids = heldout["row_id"].to_numpy(dtype=np.int64)
    prev = arrays["prev_latents"].astype(np.float64)[row_ids]
    target = arrays["target_latents"].astype(np.float64)[row_ids]
    residuals = arrays["residuals"].astype(np.float64)[row_ids]
    target_mean = np.asarray(transforms["target_mean"], dtype=np.float64)
    transform = np.asarray(transforms["whitening"], dtype=np.float64)
    target_std = np.asarray(transforms["target_std"], dtype=np.float64)
    specs = [
        ("raw", prev, target, residuals),
        ("whitened", (prev - target_mean) @ transform, (target - target_mean) @ transform, residuals @ transform),
        ("targetnorm", (prev - target_mean) / target_std, (target - target_mean) / target_std, residuals / target_std),
    ]
    frames = []
    for scale, x, y, res in specs:
        print_step(f"cube sensitivity scale={scale}")
        frame = local_sensitivity_frame(
            x=x,
            y=y,
            residuals=res,
            episode_keys=heldout["episode_ordinal"].to_numpy(),
            regimes=heldout["regime"].to_numpy(),
            row_ids=row_ids,
            k=k,
            k_min=k_min,
        )
        frame["dataset"] = "cube"
        frame["scale"] = scale
        frames.append(frame)
    all_frame = pd.concat(frames, ignore_index=True)
    group_summary, contrasts = make_summary_tables(
        all_frame,
        dataset="cube",
        right_regime="transport_free",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(output_dir / "cube" / "sensitivity_anchor_metrics.csv", all_frame)
    write_csv(output_dir / "cube" / "sensitivity_regime_summary.csv", group_summary)
    write_csv(output_dir / "cube" / "sensitivity_contrasts.csv", contrasts)
    return group_summary, contrasts


def load_fetch_examples() -> tuple[object, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    with (FETCH_FULLSTATE_DIR / "split_keys.json").open("r", encoding="utf-8") as f:
        split_payload = json.load(f)
    records = load_records(FETCH_DATA_PATHS, history_size=3)
    examples = build_examples(records, history_size=3, include_shifted_fullstate=True)
    train_mask, val_mask = masks_from_train_keys(examples.meta, split_payload["train_keys"])
    val_meta = examples.meta.loc[val_mask].reset_index(drop=True)
    saved_meta = pd.read_csv(FETCH_ENSEMBLE_DIR / "val_decomposition.csv.gz")
    compare_cols = ["env_id", "episode_id", "step_idx"]
    if len(saved_meta) != len(val_meta):
        raise AssertionError(f"Fetch saved validation rows mismatch: {len(saved_meta)} vs {len(val_meta)}")
    for col in compare_cols:
        left = saved_meta[col].astype(str).to_numpy()
        right = val_meta[col].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise AssertionError(f"Fetch validation ordering mismatch in column {col}")
    npz = np.load(FETCH_ENSEMBLE_DIR / "member_predictions_val.npz")
    targets = np.asarray(npz["targets"], dtype=np.float64)
    if targets.shape != examples.y[val_mask].shape or not np.allclose(targets, examples.y[val_mask], rtol=1e-5, atol=1e-6):
        raise AssertionError("Fetch validation targets do not match member_predictions_val.npz")
    return examples, train_mask, val_mask, saved_meta, np.asarray(npz["predictions"], dtype=np.float64)


def run_fetch_sensitivity(
    *,
    output_dir: Path,
    n_bootstrap: int,
    seed: int,
    k: int,
    k_min: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_step("rebuilding Fetch fullstate examples for sensitivity")
    examples, train_mask, val_mask, val_meta, predictions = load_fetch_examples()
    x_all_norm, _, _ = standardize(examples.x[train_mask], examples.x)
    x_norm = x_all_norm[val_mask]
    x_raw = examples.x[val_mask].astype(np.float64)
    y_raw = examples.y[val_mask].astype(np.float64)
    target_mean = examples.y[train_mask].mean(axis=0)
    target_std = per_dim_std(examples.y[train_mask])
    y_norm = (y_raw - target_mean) / target_std
    ensemble_mean = predictions.mean(axis=0)
    residual_raw = ensemble_mean - y_raw
    residual_norm = residual_raw / target_std

    interaction = val_meta["gripper_object_contact"].astype(bool).to_numpy()
    primary = val_meta["primary_regime"].astype(str).to_numpy()
    regimes = np.where(interaction, "interaction", np.where(primary == "free_motion", "free", primary))
    episode_keys = val_meta[["env_id", "episode_id"]].astype(str).agg("::".join, axis=1).to_numpy()
    row_ids = val_meta[["env_id", "episode_id", "step_idx"]].astype(str).agg("::".join, axis=1).to_numpy()

    specs = [
        ("raw_fullstate", x_raw, y_raw, residual_raw),
        ("normalized_fullstate", x_norm, y_norm, residual_norm),
    ]
    frames = []
    for scale, x, y, res in specs:
        print_step(f"fetch sensitivity scale={scale}")
        frame = local_sensitivity_frame(
            x=x,
            y=y,
            residuals=res,
            episode_keys=episode_keys,
            regimes=regimes,
            row_ids=row_ids,
            k=k,
            k_min=k_min,
        )
        frame["dataset"] = "fetch"
        frame["scale"] = scale
        frames.append(frame)
    all_frame = pd.concat(frames, ignore_index=True)
    group_summary, contrasts = make_summary_tables(
        all_frame,
        dataset="fetch",
        right_regime="free",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(output_dir / "fetch" / "sensitivity_anchor_metrics.csv", all_frame)
    write_csv(output_dir / "fetch" / "sensitivity_regime_summary.csv", group_summary)
    write_csv(output_dir / "fetch" / "sensitivity_contrasts.csv", contrasts)
    return group_summary, contrasts


def contrast_pass(contrasts: pd.DataFrame, *, dataset: str, scale: str, metric: str) -> bool:
    row = contrasts.loc[
        contrasts["dataset"].eq(dataset) & contrasts["scale"].eq(scale) & contrasts["metric"].eq(metric)
    ]
    if row.empty:
        return False
    row0 = row.iloc[0]
    return bool(
        np.isfinite(float(row0["delta_trimmed_mean"]))
        and float(row0["delta_trimmed_mean"]) > 0.0
        and float(row0["delta_trimmed_mean_ci_low"]) > 0.0
    )


def build_decision(
    *,
    part1_flags: dict[str, object],
    cube_contrasts: pd.DataFrame,
    fetch_contrasts: pd.DataFrame,
) -> dict[str, object]:
    cube_2a = contrast_pass(cube_contrasts, dataset="cube", scale="whitened", metric="local_expansion")
    cube_2b = contrast_pass(cube_contrasts, dataset="cube", scale="whitened", metric="model_sensitivity")
    fetch_2a = contrast_pass(fetch_contrasts, dataset="fetch", scale="normalized_fullstate", metric="local_expansion")
    fetch_2b = contrast_pass(fetch_contrasts, dataset="fetch", scale="normalized_fullstate", metric="model_sensitivity")
    confirmed = bool(part1_flags["part1_survives_all_controls"] or cube_2a or cube_2b)
    if confirmed:
        verdict = "confirmed_motion_free_contact_difficulty"
    else:
        verdict = "no_motion_free_contact_difficulty_found_current_data"
    return {
        "verdict": verdict,
        "part1_survives_all_controls": bool(part1_flags["part1_survives_all_controls"]),
        "cube_normalized_local_expansion_pass": cube_2a,
        "cube_normalized_model_sensitivity_pass": cube_2b,
        "fetch_convergent_local_expansion_pass": fetch_2a,
        "fetch_convergent_model_sensitivity_pass": fetch_2b,
        "fetch_convergent_support": bool(fetch_2a or fetch_2b),
        "part1_flags": part1_flags,
    }


def write_report(
    path: Path,
    *,
    decision: dict[str, object],
    cube_part1: pd.DataFrame,
    cube_summary: pd.DataFrame,
    cube_contrasts: pd.DataFrame,
    fetch_summary: pd.DataFrame,
    fetch_contrasts: pd.DataFrame,
    provenance: dict[str, object],
) -> None:
    lines = [
        "# Metric Hardening Diagnostic Report",
        "",
        "## Verdict",
        "",
        f"Decision: `{decision['verdict']}`.",
        "",
        (
            "This is a diagnostic recompute from existing artifacts only. No new data collection, benchmark, "
            "encoding, model training, or model calls were run."
        ),
        "",
        "Main gate components:",
        "",
        f"- Part 1 all-controls Cube matched-phase excess: `{decision['part1_survives_all_controls']}`",
        f"- Cube normalized true-dynamics local expansion: `{decision['cube_normalized_local_expansion_pass']}`",
        f"- Cube normalized model sensitivity: `{decision['cube_normalized_model_sensitivity_pass']}`",
        f"- Fetch convergent support: `{decision['fetch_convergent_support']}`",
        "",
        "## Cube Part 1",
        "",
        markdown_table(
            cube_part1,
            [
                "test",
                "scale",
                "metric",
                "delta_trimmed_mean",
                "delta_trimmed_mean_ci_low",
                "delta_trimmed_mean_ci_high",
                "p_superiority",
                "median_episode_delta",
                "fraction_episode_positive",
                "wilcoxon_p_greater",
                "n_pairs",
            ],
            max_rows=12,
        ),
        "",
        "## Cube Sensitivity Contrasts",
        "",
        markdown_table(
            cube_contrasts,
            [
                "scale",
                "metric",
                "delta_trimmed_mean",
                "delta_trimmed_mean_ci_low",
                "delta_trimmed_mean_ci_high",
                "p_superiority",
                "left_n",
                "right_n",
            ],
            max_rows=20,
        ),
        "",
        "## Cube Density Summary",
        "",
        markdown_table(
            cube_summary.loc[cube_summary["metric"].eq("neighbor_radius")],
            ["scale", "regime", "n", "n_episodes", "median", "trimmed_mean", "trimmed_mean_ci_low", "trimmed_mean_ci_high"],
            max_rows=20,
        ),
        "",
        "## Fetch Sensitivity Contrasts",
        "",
        markdown_table(
            fetch_contrasts,
            [
                "scale",
                "metric",
                "delta_trimmed_mean",
                "delta_trimmed_mean_ci_low",
                "delta_trimmed_mean_ci_high",
                "p_superiority",
                "left_n",
                "right_n",
            ],
            max_rows=20,
        ),
        "",
        "## Fetch Density Summary",
        "",
        markdown_table(
            fetch_summary.loc[fetch_summary["metric"].eq("neighbor_radius")],
            ["scale", "regime", "n", "n_episodes", "median", "trimmed_mean", "trimmed_mean_ci_low", "trimmed_mean_ci_high"],
            max_rows=20,
        ),
        "",
        "## Assumptions And Failure Modes",
        "",
        "- Neighbor sets are cross-episode only, which reduces temporal leakage but does not guarantee identical hidden state or action intent.",
        "- Cube sensitivity is latent-state based because the saved Cube artifacts do not include the full action-conditioned model input.",
        "- Local expansion can reflect multimodality, unobserved variables, sparse neighborhoods, or metric geometry; it is not causal proof by itself.",
        "- Fetch is a state-space proxy cross-check, not a latent LeWM result and not sufficient for the main confirm-data gate.",
        "- Density differences should be read alongside expansion/sensitivity; larger interaction neighbor radii weaken the interpretation.",
        "",
        "## Provenance",
        "",
        f"- Cube rows: `{provenance['cube_rows']}` total, `{provenance['cube_heldout_rows']}` heldout",
        f"- Cube whitening recompute max absolute difference: `{provenance['cube_whitened_recompute_max_abs_diff']:.6g}`",
        f"- Fetch validation rows: `{provenance['fetch_val_rows']}`",
        f"- Bootstrap samples: `{provenance['bootstrap_samples']}`",
        f"- Neighbor k/min: `{provenance['neighbor_k']}` / `{provenance['neighbor_k_min']}`",
        "",
        "## Output Files",
        "",
        "- `runs/metric_hardening/cube/part1_metric_contrasts.csv`",
        "- `runs/metric_hardening/cube/double_matched_pairs.csv`",
        "- `runs/metric_hardening/cube/sensitivity_regime_summary.csv`",
        "- `runs/metric_hardening/cube/sensitivity_contrasts.csv`",
        "- `runs/metric_hardening/fetch/sensitivity_regime_summary.csv`",
        "- `runs/metric_hardening/fetch/sensitivity_contrasts.csv`",
        "- `runs/metric_hardening/decision.json`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cube").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fetch").mkdir(parents=True, exist_ok=True)

    print_step("loading Cube artifacts")
    cube_records, cube_arrays = load_cube_records()
    cube_records, cube_transforms = add_cube_metric_columns(cube_records, cube_arrays)
    write_csv(args.output_dir / "cube" / "step_records_metric_hardened.csv", cube_records)
    print_step(f"cube rows={len(cube_records)} heldout={int(cube_records['split'].eq('heldout').sum())}")

    print_step("running Cube Part 1")
    cube_part1, _, part1_flags = run_cube_part1(
        cube_records,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    print_step("running Cube sensitivity")
    cube_summary, cube_contrasts = run_cube_sensitivity(
        cube_records,
        cube_arrays,
        cube_transforms,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 100000,
        k=args.neighbor_k,
        k_min=args.neighbor_k_min,
    )

    print_step("running Fetch sensitivity")
    fetch_summary, fetch_contrasts = run_fetch_sensitivity(
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 200000,
        k=args.neighbor_k,
        k_min=args.neighbor_k_min,
    )

    decision = build_decision(part1_flags=part1_flags, cube_contrasts=cube_contrasts, fetch_contrasts=fetch_contrasts)
    provenance = {
        "cube_rows": int(len(cube_records)),
        "cube_heldout_rows": int(cube_records["split"].eq("heldout").sum()),
        "cube_whitened_recompute_max_abs_diff": float(cube_transforms["whitened_recompute_max_abs_diff"]),
        "fetch_val_rows": int(pd.read_csv(FETCH_ENSEMBLE_DIR / "val_decomposition.csv.gz", usecols=["env_id"]).shape[0]),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "neighbor_k": int(args.neighbor_k),
        "neighbor_k_min": int(args.neighbor_k_min),
    }
    write_json(args.output_dir / "decision.json", {"decision": decision, "provenance": provenance})
    write_report(
        args.output_dir / "REPORT.md",
        decision=decision,
        cube_part1=cube_part1,
        cube_summary=cube_summary,
        cube_contrasts=cube_contrasts,
        fetch_summary=fetch_summary,
        fetch_contrasts=fetch_contrasts,
        provenance=provenance,
    )
    print_step(f"decision={decision['verdict']}")
    print_step(f"report={args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
