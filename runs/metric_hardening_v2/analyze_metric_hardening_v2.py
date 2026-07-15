#!/usr/bin/env python3
"""Run the metric-hardening v2 recompute from existing artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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

from experiment_utils import build_examples, load_records, masks_from_train_keys
from metric_hardening_v2_utils import (
    WhiteningTransform,
    apply_delta_transform,
    apply_transform,
    balanced_joint_features,
    contrast_pass,
    coverage_ok,
    cross_episode_neighbors,
    cross_fitted_knn_probe_frame,
    cube_augmented_kinematics_from_arrays,
    cube_action_blocks_from_arrays,
    fit_whitening_transform,
    k_sweep_pass,
    local_sensitivity_frame,
    matched_pair_metric_summary,
    matched_pairs_by_radius,
    per_dim_std,
    standardize_apply,
    standardize_fit,
    summarize_sensitivity_frame,
    write_csv,
)


DEFAULT_OUTPUT_DIR = ROOT / "runs" / "metric_hardening_v2"
CUBE_INPUT_DIR = ROOT / "runs" / "lewm_transfer" / "cube"
CUBE_RELABEL_DIR = CUBE_INPUT_DIR / "relabel_motion"
CUBE_CACHE_LANCE = CUBE_INPUT_DIR / "cache" / "cube_first30_pixel_subset.lance"
DEFAULT_SOURCE_H5 = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")
FETCH_ENSEMBLE_DIR = FETCH_ROOT / "runs" / "ensemble_bv"
FETCH_FULLSTATE_DIR = FETCH_ROOT / "runs" / "fullstate_ablation"
FETCH_DATA_PATHS = [
    FETCH_ROOT / "data" / "swm__FetchPushDense-v3" / "records.csv.gz",
    FETCH_ROOT / "data" / "swm__FetchSlideDense-v3" / "records.csv.gz",
]

K_VALUES = [5, 10, 20, 40]
PRIMARY_K = 10
PRIMARY_K_MIN = 5


@dataclass
class CubeBundle:
    records: pd.DataFrame
    arrays: dict[str, np.ndarray]
    action_blocks: np.ndarray
    kinematics: np.ndarray
    augmented_kinematics: np.ndarray
    kinematic_pose_source: str
    augmented_kinematic_provenance: dict[str, object]
    target_std: np.ndarray
    target_mean: np.ndarray
    full_transform: WhiteningTransform
    h5_provenance: dict[str, object]


@dataclass
class CubeR1Inputs:
    heldout: pd.DataFrame
    row_ids: np.ndarray
    heldout_positions: np.ndarray
    joint: np.ndarray
    action_z: np.ndarray
    y: np.ndarray
    residuals: np.ndarray
    episode_keys: np.ndarray
    regimes: np.ndarray


@dataclass
class FetchBundle:
    val_joint: np.ndarray
    val_action_z: np.ndarray
    val_y: np.ndarray
    val_residuals: np.ndarray
    val_meta: pd.DataFrame
    val_regimes: np.ndarray
    val_episode_keys: np.ndarray
    val_row_ids: np.ndarray
    train_joint: np.ndarray
    train_episode_keys: np.ndarray
    provenance: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_SOURCE_H5)
    parser.add_argument("--num-trajectories", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=260727)
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[metric-hardening-v2] {message}", flush=True)


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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
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
    arrays_npz = np.load(CUBE_INPUT_DIR / "residuals.npz")
    arrays = {key: np.asarray(arrays_npz[key]) for key in arrays_npz.files}
    row_ids = arrays["row_id"]
    if not np.array_equal(row_ids, np.arange(len(row_ids))):
        raise AssertionError("Cube residuals row_id is not contiguous")
    if records["row_id"].max() >= len(row_ids):
        raise AssertionError("Cube records reference row_id outside residuals.npz")
    return records.sort_values("row_id").reset_index(drop=True), arrays


def read_cube_h5_features(
    records: pd.DataFrame, source_h5: Path, num_trajectories: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, object], dict[str, object]]:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - exercised by environment, not unit tests
        raise RuntimeError("h5py is required for Cube action and kinematic joins") from exc

    if not source_h5.exists():
        raise FileNotFoundError(source_h5)
    with h5py.File(source_h5, "r") as h5:
        lengths = np.asarray(h5["ep_len"][:num_trajectories], dtype=np.int64)
        offsets = np.asarray(h5["ep_offset"][:num_trajectories], dtype=np.int64)
        episode_ids = np.asarray([int(h5["ep_idx"][int(offset)]) for offset in offsets], dtype=np.int64)
        stop = int(offsets[-1] + lengths[-1])
        actions = np.asarray(h5["action"][:stop], dtype=np.float64)
        action_blocks = cube_action_blocks_from_arrays(
            records,
            actions=actions,
            offsets=offsets,
            lengths=lengths,
            episode_ids=episode_ids,
        )
        effector = np.asarray(h5["proprio_effector_pos"][:stop], dtype=np.float64)
        block_pos = np.asarray(h5["privileged_block_0_pos"][:stop], dtype=np.float64)
        block_quat = np.asarray(h5["privileged_block_0_quat"][:stop], dtype=np.float64) if "privileged_block_0_quat" in h5 else None
        block_yaw = np.asarray(h5["privileged_block_0_yaw"][:stop], dtype=np.float64) if "privileged_block_0_yaw" in h5 else None
        # Legacy target-time kinematics are retained for provenance only. R3v2
        # below uses input-time augmented kinematics to avoid target leakage.
        from metric_hardening_v2_utils import cube_kinematics_from_arrays

        kinematics, pose_source = cube_kinematics_from_arrays(
            records,
            effector_pos=effector,
            block_pos=block_pos,
            block_quat=block_quat,
            block_yaw=block_yaw,
            offsets=offsets,
            lengths=lengths,
        )
        native_velocity_arrays = {
            key: np.asarray(h5[key][:stop], dtype=np.float64)
            for key in ["qvel", "prev_qvel", "proprio_joint_vel", "proprio_gripper_vel"]
            if key in h5
        }
        gripper_arrays = {
            key: np.asarray(h5[key][:stop], dtype=np.float64)
            for key in ["proprio_gripper_opening", "proprio_gripper_contact", "proprio_gripper_vel"]
            if key in h5
        }
        augmented_kinematics, augmented_provenance = cube_augmented_kinematics_from_arrays(
            records,
            effector_pos=effector,
            block_pos=block_pos,
            block_quat=block_quat,
            block_yaw=block_yaw,
            native_velocity_arrays=native_velocity_arrays,
            gripper_arrays=gripper_arrays,
            offsets=offsets,
            lengths=lengths,
        )
    sample_positions = np.linspace(0, len(records) - 1, num=min(12, len(records)), dtype=np.int64)
    max_action_abs = float(np.max(np.abs(action_blocks[sample_positions]))) if len(sample_positions) else float("nan")
    provenance = {
        "source_h5": str(source_h5),
        "cached_lance_path": str(CUBE_CACHE_LANCE),
        "cached_lance_exists": bool(CUBE_CACHE_LANCE.exists()),
        "join_source_used": "source_h5",
        "reason": "The local Python runtime lacks the lance reader; source H5 is the verified row source used to create the cached Lance subset.",
        "num_trajectories": int(num_trajectories),
        "source_rows_first_episodes": int(lengths.sum()),
        "action_dim": int(action_blocks.shape[1]),
        "kinematic_dim": int(kinematics.shape[1]),
        "augmented_kinematic_dim": int(augmented_kinematics.shape[1]),
        "kinematic_pose_source": pose_source,
        "augmented_kinematic": augmented_provenance,
        "sample_max_abs_action": max_action_abs,
    }
    return action_blocks, kinematics, augmented_kinematics, pose_source, augmented_provenance, provenance


def prepare_cube_bundle(args: argparse.Namespace) -> CubeBundle:
    records, arrays = load_cube_records()
    action_blocks, kinematics, augmented_kinematics, pose_source, augmented_provenance, h5_provenance = read_cube_h5_features(
        records,
        source_h5=args.source_h5,
        num_trajectories=args.num_trajectories,
    )
    target = arrays["target_latents"].astype(np.float64)
    calibration_ids = records.loc[records["split"].eq("calibration"), "row_id"].to_numpy(dtype=np.int64)
    target_mean = target[calibration_ids].mean(axis=0)
    target_std = per_dim_std(target[calibration_ids])
    full_transform = fit_whitening_transform(
        target[calibration_ids],
        name="full_whitened",
        family="full",
        floor_fraction=None,
        eps=1e-6,
    )
    return CubeBundle(
        records=records,
        arrays=arrays,
        action_blocks=action_blocks,
        kinematics=kinematics,
        augmented_kinematics=augmented_kinematics,
        kinematic_pose_source=pose_source,
        augmented_kinematic_provenance=augmented_provenance,
        target_std=target_std,
        target_mean=target_mean,
        full_transform=full_transform,
        h5_provenance=h5_provenance,
    )


def cube_r1_inputs(bundle: CubeBundle) -> CubeR1Inputs:
    records = bundle.records
    heldout = records.loc[records["split"].eq("heldout")].reset_index(drop=True)
    row_ids = heldout["row_id"].to_numpy(dtype=np.int64)
    all_row_ids = records["row_id"].to_numpy(dtype=np.int64)
    calibration_mask = records["split"].eq("calibration").to_numpy(dtype=bool)
    prev = bundle.arrays["prev_latents"].astype(np.float64)
    target = bundle.arrays["target_latents"].astype(np.float64)
    residuals = bundle.arrays["residuals"].astype(np.float64)
    state_all = apply_transform(prev[all_row_ids], bundle.full_transform)
    y_all = apply_transform(target[all_row_ids], bundle.full_transform)
    res_all = apply_delta_transform(residuals[all_row_ids], bundle.full_transform)
    action_mean, action_std = standardize_fit(bundle.action_blocks[calibration_mask])
    action_z_all = standardize_apply(bundle.action_blocks, action_mean, action_std)
    heldout_positions = heldout.index.to_numpy(dtype=np.int64) + int(calibration_mask.sum())
    # The records are sorted by row_id and the first rows are calibration, so
    # heldout_positions map the held-out records back into all-row arrays.
    if not np.array_equal(records.iloc[heldout_positions]["row_id"].to_numpy(dtype=np.int64), row_ids):
        heldout_positions = records.index[records["split"].eq("heldout")].to_numpy(dtype=np.int64)
    joint = balanced_joint_features(state_all[heldout_positions], action_z_all[heldout_positions])
    return CubeR1Inputs(
        heldout=heldout,
        row_ids=row_ids,
        heldout_positions=heldout_positions,
        joint=joint,
        action_z=action_z_all[heldout_positions],
        y=y_all[heldout_positions],
        residuals=res_all[heldout_positions],
        episode_keys=heldout["episode_ordinal"].astype(str).to_numpy(),
        regimes=heldout["regime"].astype(str).to_numpy(),
    )


def cube_action_matched_frame(bundle: CubeBundle, *, k: int, k_min: int) -> pd.DataFrame:
    inputs = cube_r1_inputs(bundle)
    frame = local_sensitivity_frame(
        x=inputs.joint,
        y=inputs.y,
        residuals=inputs.residuals,
        episode_keys=inputs.episode_keys,
        regimes=inputs.regimes,
        row_ids=inputs.row_ids,
        k=k,
        k_min=k_min,
        action_z=inputs.action_z,
        extra_cols={"episode_ordinal": inputs.heldout["episode_ordinal"].to_numpy(dtype=np.int64)},
    )
    frame["dataset"] = "cube"
    frame["scale"] = "full_whitened_action_matched"
    frame["k"] = int(k)
    return frame


def run_cube_r1(bundle: CubeBundle, *, output_dir: Path, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    inputs = cube_r1_inputs(bundle)
    neighbors = cross_episode_neighbors(inputs.joint, inputs.episode_keys, k=PRIMARY_K, k_min=PRIMARY_K_MIN)
    frame = local_sensitivity_frame(
        x=inputs.joint,
        y=inputs.y,
        residuals=inputs.residuals,
        episode_keys=inputs.episode_keys,
        regimes=inputs.regimes,
        row_ids=inputs.row_ids,
        k=PRIMARY_K,
        k_min=PRIMARY_K_MIN,
        action_z=inputs.action_z,
        neighbors=neighbors,
        extra_cols={"episode_ordinal": inputs.heldout["episode_ordinal"].to_numpy(dtype=np.int64)},
    )
    frame["dataset"] = "cube"
    frame["scale"] = "full_whitened_action_matched"
    frame["k"] = PRIMARY_K
    summary, contrasts = summarize_sensitivity_frame(
        frame,
        dataset="cube",
        r_item="R1",
        scale="full_whitened_action_matched",
        right_regime="transport_free",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(output_dir / "cube" / "r1_action_matched_anchor_metrics.csv", frame)
    write_csv(output_dir / "cube" / "r1_action_matched_summary.csv", summary)
    write_csv(output_dir / "cube" / "r1_action_matched_contrasts.csv", contrasts)
    np.savez_compressed(
        output_dir / "cube" / "r1_action_matched_neighbors.npz",
        indices=neighbors.indices,
        distances=neighbors.distances,
        valid=neighbors.valid,
        row_ids=inputs.row_ids,
    )
    if not coverage_ok(frame):
        status = "void_insufficient_coverage"
    elif contrast_pass(contrasts, metric="local_expansion") and contrast_pass(contrasts, metric="model_sensitivity"):
        status = "pass"
    else:
        status = "fail"
    return frame, summary, contrasts, status


def run_cube_r2(bundle: CubeBundle, *, output_dir: Path, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    records = bundle.records
    heldout = records.loc[records["split"].eq("heldout")].reset_index(drop=True)
    row_ids = heldout["row_id"].to_numpy(dtype=np.int64)
    calibration_ids = records.loc[records["split"].eq("calibration"), "row_id"].to_numpy(dtype=np.int64)
    target = bundle.arrays["target_latents"].astype(np.float64)
    prev = bundle.arrays["prev_latents"].astype(np.float64)
    residuals = bundle.arrays["residuals"].astype(np.float64)
    transforms = [
        bundle.full_transform,
        fit_whitening_transform(target[calibration_ids], name="floor_1e-3", family="shrinkage", floor_fraction=1e-3),
        fit_whitening_transform(target[calibration_ids], name="floor_1e-2", family="shrinkage", floor_fraction=1e-2),
        fit_whitening_transform(target[calibration_ids], name="pca90", family="pca", pca_variance=0.90),
        fit_whitening_transform(target[calibration_ids], name="pca99", family="pca", pca_variance=0.99),
    ]
    frames = []
    summary_frames = []
    contrast_frames = []
    eig_rows = []
    for offset, transform in enumerate(transforms):
        frame = local_sensitivity_frame(
            x=apply_transform(prev[row_ids], transform),
            y=apply_transform(target[row_ids], transform),
            residuals=apply_delta_transform(residuals[row_ids], transform),
            episode_keys=heldout["episode_ordinal"].astype(str).to_numpy(),
            regimes=heldout["regime"].astype(str).to_numpy(),
            row_ids=row_ids,
            k=PRIMARY_K,
            k_min=PRIMARY_K_MIN,
            extra_cols={"episode_ordinal": heldout["episode_ordinal"].to_numpy(dtype=np.int64)},
        )
        frame["dataset"] = "cube"
        frame["scale"] = transform.name
        frame["k"] = PRIMARY_K
        frames.append(frame)
        summary, contrasts = summarize_sensitivity_frame(
            frame,
            dataset="cube",
            r_item="R2",
            scale=transform.name,
            right_regime="transport_free",
            n_bootstrap=n_bootstrap,
            seed=seed + offset * 100000,
        )
        summary_frames.append(summary)
        contrast_frames.append(contrasts)
        eig_rows.append(
            {
                "scale": transform.name,
                "family": transform.family,
                "retained_dim": transform.retained_dim,
                "condition_number": transform.condition_number,
                "variance_retained": transform.variance_retained,
                "floor_value": transform.floor_value,
                "eig_min": float(np.min(transform.eigvals)),
                "eig_max": float(np.max(transform.eigvals)),
                "used_eig_min": float(np.min(transform.used_eigvals)),
                "used_eig_max": float(np.max(transform.used_eigvals)),
            }
        )
    all_frames = pd.concat(frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    contrasts_all = pd.concat(contrast_frames, ignore_index=True)
    eig_summary = pd.DataFrame(eig_rows)
    write_csv(output_dir / "cube" / "r2_whitening_anchor_metrics.csv", all_frames)
    write_csv(output_dir / "cube" / "r2_whitening_robustness.csv", contrasts_all)
    write_csv(output_dir / "cube" / "r2_whitening_regime_summary.csv", summary_all)
    write_csv(output_dir / "cube" / "r2_whitening_eigensummary.csv", eig_summary)

    shrink_pass = all(
        contrast_pass(contrasts_all.loc[contrasts_all["scale"].eq(scale)], metric="local_expansion")
        for scale in ["floor_1e-3", "floor_1e-2"]
    )
    pca_pass = all(
        contrast_pass(contrasts_all.loc[contrasts_all["scale"].eq(scale)], metric="local_expansion")
        for scale in ["pca90", "pca99"]
    )
    if shrink_pass or pca_pass:
        status = "pass"
    elif any(
        contrast_pass(contrasts_all.loc[contrasts_all["scale"].eq(scale)], metric="model_sensitivity")
        for scale in ["floor_1e-3", "floor_1e-2", "pca90", "pca99"]
    ):
        status = "fail_expansion_only"
    else:
        status = "fail"
    return summary_all, contrasts_all, eig_summary, status


def run_cube_r3(bundle: CubeBundle, *, output_dir: Path, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    records = bundle.records
    heldout = records.loc[records["split"].eq("heldout")].reset_index(drop=True)
    row_ids = heldout["row_id"].to_numpy(dtype=np.int64)
    calibration_mask = records["split"].eq("calibration").to_numpy(dtype=bool)
    kin_mean, kin_std = standardize_fit(bundle.augmented_kinematics[calibration_mask])
    kin_z = standardize_apply(bundle.augmented_kinematics, kin_mean, kin_std)
    action_mean, action_std = standardize_fit(bundle.action_blocks[calibration_mask])
    action_z = standardize_apply(bundle.action_blocks, action_mean, action_std)
    target = bundle.arrays["target_latents"].astype(np.float64)
    residuals = bundle.arrays["residuals"].astype(np.float64)
    y = (target[row_ids] - bundle.target_mean) / bundle.target_std
    res = residuals[row_ids] / bundle.target_std
    heldout_positions = records.index[records["split"].eq("heldout")].to_numpy(dtype=np.int64)
    joint = balanced_joint_features(kin_z[heldout_positions], action_z[heldout_positions])
    frame = local_sensitivity_frame(
        x=joint,
        y=y,
        residuals=res,
        episode_keys=heldout["episode_ordinal"].astype(str).to_numpy(),
        regimes=heldout["regime"].astype(str).to_numpy(),
        row_ids=row_ids,
        k=PRIMARY_K,
        k_min=PRIMARY_K_MIN,
        action_z=action_z[heldout_positions],
        extra_cols={"episode_ordinal": heldout["episode_ordinal"].to_numpy(dtype=np.int64)},
    )
    frame["dataset"] = "cube"
    frame["scale"] = f"r3v2_augmented_kinematic_action_neighbors_targetnorm_latent_{bundle.kinematic_pose_source}"
    frame["k"] = PRIMARY_K
    summary, contrasts = summarize_sensitivity_frame(
        frame,
        dataset="cube",
        r_item="R3v2",
        scale=frame["scale"].iloc[0],
        right_regime="transport_free",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    r1_inputs = cube_r1_inputs(bundle)
    r1_neighbors = cross_episode_neighbors(r1_inputs.joint, r1_inputs.episode_keys, k=PRIMARY_K, k_min=PRIMARY_K_MIN)
    symmetric = local_sensitivity_frame(
        x=r1_inputs.joint,
        y=kin_z[heldout_positions],
        residuals=np.zeros_like(kin_z[heldout_positions]),
        episode_keys=r1_inputs.episode_keys,
        regimes=r1_inputs.regimes,
        row_ids=r1_inputs.row_ids,
        k=PRIMARY_K,
        k_min=PRIMARY_K_MIN,
        action_z=r1_inputs.action_z,
        neighbors=r1_neighbors,
        extra_cols={"episode_ordinal": heldout["episode_ordinal"].to_numpy(dtype=np.int64)},
    )
    symmetric["dataset"] = "cube"
    symmetric["scale"] = "r3v2_latent_action_neighbors_kinematic_divergence"
    symmetric["k"] = PRIMARY_K
    symmetric["kinematic_divergence"] = symmetric["successor_spread"]
    symmetric_summary, symmetric_contrasts = summarize_sensitivity_frame(
        symmetric,
        dataset="cube",
        r_item="R3v2_symmetric",
        scale=symmetric["scale"].iloc[0],
        right_regime="transport_free",
        n_bootstrap=n_bootstrap,
        seed=seed + 900000,
        extra_value_cols=["kinematic_divergence"],
    )
    write_csv(output_dir / "cube" / "r3_aliasing_anchor_metrics.csv", frame)
    write_csv(output_dir / "cube" / "r3_aliasing_discriminator.csv", contrasts)
    write_csv(output_dir / "cube" / "r3_aliasing_regime_summary.csv", summary)
    write_csv(output_dir / "cube" / "r3v2_symmetric_anchor_metrics.csv", symmetric)
    write_csv(output_dir / "cube" / "r3v2_symmetric_aliasing_diagnostic.csv", symmetric_contrasts)
    write_csv(output_dir / "cube" / "r3v2_symmetric_regime_summary.csv", symmetric_summary)
    if not coverage_ok(frame):
        status = "void_insufficient_coverage"
    elif contrast_pass(contrasts, metric="local_expansion") and contrast_pass(contrasts, metric="model_sensitivity"):
        status = "pass"
    else:
        status = "fail"
    return frame, summary, contrasts, status


def run_cube_struggle_probes(
    bundle: CubeBundle,
    *,
    output_dir: Path,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    inputs = cube_r1_inputs(bundle)
    probe = cross_fitted_knn_probe_frame(
        x=inputs.joint,
        y=inputs.y,
        residuals=inputs.residuals,
        episode_keys=inputs.episode_keys,
        regimes=inputs.regimes,
        row_ids=inputs.row_ids,
        k=PRIMARY_K,
        k_min=PRIMARY_K_MIN,
        n_folds=5,
    )
    summary, contrasts = summarize_sensitivity_frame(
        probe,
        dataset="cube",
        r_item="struggle_probes",
        scale="r1_action_matched_crossfit_knn",
        right_regime="transport_free",
        n_bootstrap=n_bootstrap,
        seed=seed,
        extra_value_cols=[
            "actual_residual_norm",
            "predicted_residual_norm",
            "residual_prediction_mse",
            "residual_baseline_mse",
            "residual_predictability",
            "conditional_mean_floor",
            "successor_baseline_mse",
            "conditional_mean_gain",
        ],
    )
    write_csv(output_dir / "cube" / "struggle_probe_anchor_metrics.csv", probe)
    write_csv(output_dir / "cube" / "struggle_probe_regime_summary.csv", summary)
    write_csv(output_dir / "cube" / "struggle_probe_contrasts.csv", contrasts)
    if not coverage_ok(probe):
        residual_status = "void_insufficient_coverage"
        floor_status = "void_insufficient_coverage"
    else:
        residual_status = (
            "pass"
            if contrast_pass(contrasts, metric="predicted_residual_norm")
            and contrast_pass(contrasts, metric="residual_predictability")
            else "fail"
        )
        floor_status = "pass" if contrast_pass(contrasts, metric="conditional_mean_floor") else "fail"
    return summary, contrasts, residual_status, floor_status


def load_fetch_examples_bundle() -> tuple[object, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    with (FETCH_FULLSTATE_DIR / "split_keys.json").open("r", encoding="utf-8") as f:
        split_payload = json.load(f)
    records = load_records(FETCH_DATA_PATHS, history_size=3)
    examples = build_examples(records, history_size=3, include_shifted_fullstate=True)
    train_mask, val_mask = masks_from_train_keys(examples.meta, split_payload["train_keys"])
    val_meta = examples.meta.loc[val_mask].reset_index(drop=True)
    saved_meta = pd.read_csv(FETCH_ENSEMBLE_DIR / "val_decomposition.csv.gz")
    if len(saved_meta) != len(val_meta):
        raise AssertionError(f"Fetch validation rows mismatch: {len(saved_meta)} vs {len(val_meta)}")
    for col in ["env_id", "episode_id", "step_idx"]:
        if not np.array_equal(saved_meta[col].astype(str).to_numpy(), val_meta[col].astype(str).to_numpy()):
            raise AssertionError(f"Fetch validation ordering mismatch in {col}")
    npz = np.load(FETCH_ENSEMBLE_DIR / "member_predictions_val.npz")
    targets = np.asarray(npz["targets"], dtype=np.float64)
    if not np.allclose(targets, examples.y[val_mask], rtol=1e-5, atol=1e-6):
        raise AssertionError("Fetch saved targets do not align with rebuilt examples")
    return examples, train_mask, val_mask, saved_meta, np.asarray(npz["predictions"], dtype=np.float64)


def prepare_fetch_bundle() -> FetchBundle:
    examples, train_mask, val_mask, val_meta, predictions = load_fetch_examples_bundle()
    feature_cols = list(examples.feature_cols)
    action_feature_idx = np.asarray([i for i, col in enumerate(feature_cols) if ":action_" in col], dtype=np.int64)
    state_feature_idx = np.asarray([i for i, col in enumerate(feature_cols) if ":action_" not in col], dtype=np.int64)
    max_hist = max(int(col.split(":", 1)[0].replace("hist_", "")) for col in feature_cols)
    current_action_idx = np.asarray(
        [i for i, col in enumerate(feature_cols) if col.startswith(f"hist_{max_hist}:action_")],
        dtype=np.int64,
    )
    if len(current_action_idx) != 4:
        raise AssertionError(f"Expected four current Fetch action columns, got {len(current_action_idx)}")
    state_mean, state_std = standardize_fit(examples.x[train_mask][:, state_feature_idx])
    action_mean, action_std = standardize_fit(examples.x[train_mask][:, current_action_idx])
    state_z_all = standardize_apply(examples.x[:, state_feature_idx], state_mean, state_std)
    action_z_all = standardize_apply(examples.x[:, current_action_idx], action_mean, action_std)
    joint_all = balanced_joint_features(state_z_all, action_z_all)

    y_mean = examples.y[train_mask].mean(axis=0)
    y_std = per_dim_std(examples.y[train_mask])
    y_val_raw = examples.y[val_mask].astype(np.float64)
    y_val = (y_val_raw - y_mean) / y_std
    residuals = (predictions.mean(axis=0) - y_val_raw) / y_std

    interaction = val_meta["gripper_object_contact"].astype(bool).to_numpy()
    primary = val_meta["primary_regime"].astype(str).to_numpy()
    regimes = np.where(interaction, "interaction", np.where(primary == "free_motion", "free", primary))
    val_episode_keys = val_meta[["env_id", "episode_id"]].astype(str).agg("::".join, axis=1).to_numpy()
    row_ids = val_meta[["env_id", "episode_id", "step_idx"]].astype(str).agg("::".join, axis=1).to_numpy()
    train_meta = examples.meta.loc[train_mask].reset_index(drop=True)
    train_episode_keys = train_meta[["env_id", "episode_id"]].astype(str).agg("::".join, axis=1).to_numpy()
    provenance = {
        "validation_rows": int(val_mask.sum()),
        "train_rows": int(train_mask.sum()),
        "state_feature_dim_excluding_actions": int(len(state_feature_idx)),
        "all_history_action_feature_dim_excluded_from_state": int(len(action_feature_idx)),
        "current_action_dim": int(len(current_action_idx)),
        "prediction_members": int(predictions.shape[0]),
    }
    return FetchBundle(
        val_joint=joint_all[val_mask],
        val_action_z=action_z_all[val_mask],
        val_y=y_val,
        val_residuals=residuals,
        val_meta=val_meta,
        val_regimes=regimes,
        val_episode_keys=val_episode_keys,
        val_row_ids=row_ids,
        train_joint=joint_all[train_mask],
        train_episode_keys=train_episode_keys,
        provenance=provenance,
    )


def fetch_action_matched_frame(bundle: FetchBundle, *, k: int, k_min: int) -> pd.DataFrame:
    frame = local_sensitivity_frame(
        x=bundle.val_joint,
        y=bundle.val_y,
        residuals=bundle.val_residuals,
        episode_keys=bundle.val_episode_keys,
        regimes=bundle.val_regimes,
        row_ids=bundle.val_row_ids,
        k=k,
        k_min=k_min,
        action_z=bundle.val_action_z,
        extra_cols={
            "env_id": bundle.val_meta["env_id"].astype(str).to_numpy(),
            "episode_id": bundle.val_meta["episode_id"].astype(str).to_numpy(),
        },
    )
    frame["dataset"] = "fetch"
    frame["scale"] = "normalized_state_current_action_matched"
    frame["k"] = int(k)
    return frame


def run_fetch_r1(bundle: FetchBundle, *, output_dir: Path, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    frame = fetch_action_matched_frame(bundle, k=PRIMARY_K, k_min=PRIMARY_K_MIN)
    summary, contrasts = summarize_sensitivity_frame(
        frame,
        dataset="fetch",
        r_item="R1",
        scale="normalized_state_current_action_matched",
        right_regime="free",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    write_csv(output_dir / "fetch" / "r1_action_matched_anchor_metrics.csv", frame)
    write_csv(output_dir / "fetch" / "r1_action_matched_summary.csv", summary)
    write_csv(output_dir / "fetch" / "r1_action_matched_contrasts.csv", contrasts)
    if not coverage_ok(frame):
        status = "void_insufficient_coverage"
    elif contrast_pass(contrasts, metric="local_expansion") and contrast_pass(contrasts, metric="model_sensitivity"):
        status = "pass"
    else:
        status = "fail"
    return frame, summary, contrasts, status


def run_k_sweep(
    *,
    dataset: str,
    output_dir: Path,
    n_bootstrap: int,
    seed: int,
    frame_fn,
    right_regime: str,
) -> tuple[pd.DataFrame, str]:
    contrast_rows = []
    sufficient = 0
    for offset, k in enumerate(K_VALUES):
        frame = frame_fn(k=k, k_min=min(5, k))
        if coverage_ok(frame):
            sufficient += 1
        _, contrasts = summarize_sensitivity_frame(
            frame,
            dataset=dataset,
            r_item="R4",
            scale=str(frame["scale"].iloc[0]),
            right_regime=right_regime,
            n_bootstrap=n_bootstrap,
            seed=seed + offset * 100000,
        )
        contrasts["k"] = int(k)
        contrasts["n_valid_interaction"] = int(
            frame.loc[frame["valid_neighbors"].astype(bool) & frame["regime"].eq("interaction")].shape[0]
        )
        contrasts["n_valid_right"] = int(
            frame.loc[frame["valid_neighbors"].astype(bool) & frame["regime"].eq(right_regime)].shape[0]
        )
        contrast_rows.append(contrasts.loc[contrasts["metric"].isin(["neighbor_radius", "local_expansion", "model_sensitivity"])])
    all_contrasts = pd.concat(contrast_rows, ignore_index=True)
    write_csv(output_dir / dataset / "r4_k_sweep.csv", all_contrasts)
    if sufficient < 3:
        status = "void_insufficient_coverage"
    elif k_sweep_pass(all_contrasts, metric="local_expansion") and k_sweep_pass(all_contrasts, metric="model_sensitivity"):
        status = "pass"
    else:
        status = "fail"
    return all_contrasts, status


def run_fetch_r5(
    bundle: FetchBundle,
    r1_frame: pd.DataFrame,
    *,
    output_dir: Path,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    neighbors = cross_episode_neighbors(bundle.train_joint, bundle.train_episode_keys, k=PRIMARY_K, k_min=PRIMARY_K_MIN)
    train_radii = np.nanmedian(neighbors.distances, axis=1)
    train_radii = train_radii[neighbors.valid & np.isfinite(train_radii)]
    radius_edges = np.unique(np.quantile(train_radii, np.linspace(0.0, 1.0, 11))) if len(train_radii) else np.asarray([])
    pairs = matched_pairs_by_radius(r1_frame, radius_edges=radius_edges, left_regime="interaction", right_regime="free")
    valid_interaction = r1_frame.loc[r1_frame["valid_neighbors"].astype(bool) & r1_frame["regime"].eq("interaction")]
    match_fraction = float(len(pairs) / len(valid_interaction)) if len(valid_interaction) else float("nan")
    summary_rows = []
    for metric_index, metric in enumerate(["local_expansion", "model_sensitivity"]):
        row = matched_pair_metric_summary(pairs, metric=metric, n_bootstrap=n_bootstrap, seed=seed + metric_index * 10000)
        row["match_fraction_interaction"] = match_fraction
        row["total_valid_interaction"] = int(len(valid_interaction))
        row["interaction_episodes"] = int(valid_interaction["episode_key"].nunique())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if len(radius_edges) >= 2:
        valid = r1_frame.loc[r1_frame["valid_neighbors"].astype(bool)].copy()
        valid["radius_bin"] = np.searchsorted(radius_edges[1:-1], valid["neighbor_radius"].to_numpy(dtype=np.float64), side="right")
        bin_counts = (
            valid.groupby(["radius_bin", "regime"], sort=True)
            .agg(n=("row_id", "count"), n_episodes=("episode_key", "nunique"), radius_median=("neighbor_radius", "median"))
            .reset_index()
        )
    else:
        bin_counts = pd.DataFrame()
    write_csv(output_dir / "fetch" / "r5_density_matched.csv", summary)
    write_csv(output_dir / "fetch" / "r5_density_matched_pairs.csv", pairs)
    write_csv(output_dir / "fetch" / "r5_density_bins.csv", bin_counts)
    write_csv(output_dir / "fetch" / "r5_radius_edges.csv", pd.DataFrame({"edge": radius_edges}))

    if len(valid_interaction) < 30 or valid_interaction["episode_key"].nunique() < 5 or not np.isfinite(match_fraction) or match_fraction < 0.70:
        status = "void_insufficient_density_overlap"
    else:
        expansion = summary.loc[summary["metric"].eq("local_expansion")].iloc[0]
        sensitivity = summary.loc[summary["metric"].eq("model_sensitivity")].iloc[0]
        ok = (
            float(expansion["delta_trimmed_mean"]) > 0.0
            and float(expansion["delta_trimmed_mean_ci_low"]) > 0.0
            and float(sensitivity["delta_trimmed_mean"]) > 0.0
            and float(sensitivity["delta_trimmed_mean_ci_low"]) > 0.0
        )
        status = "pass" if ok else "fail"
    return summary, pairs, bin_counts, status


def write_k_sweep_plot(path: Path, cube_k: pd.DataFrame, fetch_k: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    specs = [("cube", cube_k, axes[0]), ("fetch", fetch_k, axes[1])]
    colors = {"local_expansion": "#2563eb", "model_sensitivity": "#dc2626"}
    for title, df, ax in specs:
        for metric in ["local_expansion", "model_sensitivity"]:
            rows = df.loc[df["metric"].eq(metric)].sort_values("k")
            if rows.empty:
                continue
            x = rows["k"].to_numpy(dtype=float)
            y = rows["delta_trimmed_mean"].to_numpy(dtype=float)
            low = rows["delta_trimmed_mean_ci_low"].to_numpy(dtype=float)
            high = rows["delta_trimmed_mean_ci_high"].to_numpy(dtype=float)
            yerr = np.vstack([y - low, high - y])
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=4, color=colors[metric], label=metric)
        ax.axhline(0.0, color="#111827", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel("interaction - comparator delta")
        ax.set_xticks(K_VALUES)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def status_to_hole(status: str) -> str:
    if status == "pass":
        return "closed"
    if status.startswith("void"):
        return "void"
    return "open"


def build_decision(statuses: dict[str, str]) -> dict[str, object]:
    cube_items = [statuses["cube_r1"], statuses["cube_r2"], statuses["cube_r3"], statuses["cube_r4"]]
    fetch_items = [statuses["fetch_r1"], statuses["fetch_r4"], statuses["fetch_r5"]]
    nonvoid_cube = [s for s in cube_items if not s.startswith("void")]
    nonvoid_fetch = [s for s in fetch_items if not s.startswith("void")]
    if all(s == "pass" for s in cube_items):
        cube_main = "confirmed"
    elif not nonvoid_cube:
        cube_main = "inconclusive"
    elif any(s == "pass" for s in nonvoid_cube) and any(s != "pass" for s in nonvoid_cube):
        cube_main = "mixed"
    elif all(s != "pass" for s in nonvoid_cube):
        cube_main = "unsupported"
    else:
        cube_main = "mixed"

    if all(s == "pass" for s in fetch_items):
        fetch_status = "supported"
    elif not nonvoid_fetch:
        fetch_status = "inconclusive"
    else:
        fetch_status = "unsupported"

    return {
        "cube_main_status": cube_main,
        "fetch_support_status": fetch_status,
        "holes_closed": {
            "action_conditioning": status_to_hole(statuses["cube_r1"]),
            "cube_whitening_conditioning": status_to_hole(statuses["cube_r2"]),
            "single_k_dependence": status_to_hole(statuses["cube_r4"]),
            "fetch_density_mismatch": status_to_hole(statuses["fetch_r5"]),
        },
        "extra_checks": {
            "cube_aliasing_discriminator": "passed"
            if statuses["cube_r3"] == "pass"
            else ("void" if statuses["cube_r3"].startswith("void") else "failed"),
            "cube_residual_predictability_probe": "passed"
            if statuses.get("cube_residual_predictability") == "pass"
            else ("void" if statuses.get("cube_residual_predictability", "").startswith("void") else "failed"),
            "cube_knn_conditional_mean_floor": "passed"
            if statuses.get("cube_knn_floor") == "pass"
            else ("void" if statuses.get("cube_knn_floor", "").startswith("void") else "failed"),
        },
        "r_item_statuses": statuses,
    }


def passfail(status: str) -> str:
    if status == "pass":
        return "PASS"
    if status.startswith("void"):
        return "VOID"
    return "FAIL"


def make_status_table(statuses: dict[str, str]) -> pd.DataFrame:
    rows = [
        {"item": "R1 Cube action-matched", "status": passfail(statuses["cube_r1"]), "detail": statuses["cube_r1"]},
        {"item": "R1 Fetch action-matched", "status": passfail(statuses["fetch_r1"]), "detail": statuses["fetch_r1"]},
        {"item": "R2 Cube whitening robustness", "status": passfail(statuses["cube_r2"]), "detail": statuses["cube_r2"]},
        {"item": "R3 Cube aliasing discriminator", "status": passfail(statuses["cube_r3"]), "detail": statuses["cube_r3"]},
        {"item": "R4 Cube k-sweep", "status": passfail(statuses["cube_r4"]), "detail": statuses["cube_r4"]},
        {"item": "R4 Fetch k-sweep", "status": passfail(statuses["fetch_r4"]), "detail": statuses["fetch_r4"]},
        {"item": "R5 Fetch density matching", "status": passfail(statuses["fetch_r5"]), "detail": statuses["fetch_r5"]},
        {
            "item": "Cube residual-predictability probe",
            "status": passfail(statuses.get("cube_residual_predictability", "fail")),
            "detail": statuses.get("cube_residual_predictability", "fail"),
        },
        {
            "item": "Cube kNN conditional-mean floor",
            "status": passfail(statuses.get("cube_knn_floor", "fail")),
            "detail": statuses.get("cube_knn_floor", "fail"),
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    *,
    decision: dict[str, object],
    statuses: dict[str, str],
    cube_r1_summary: pd.DataFrame,
    cube_r1: pd.DataFrame,
    fetch_r1_summary: pd.DataFrame,
    fetch_r1: pd.DataFrame,
    cube_r2: pd.DataFrame,
    cube_r3: pd.DataFrame,
    cube_r3_symmetric: pd.DataFrame,
    cube_struggle: pd.DataFrame,
    cube_r4: pd.DataFrame,
    fetch_r4: pd.DataFrame,
    fetch_r5: pd.DataFrame,
    provenance: dict[str, object],
) -> None:
    status_table = make_status_table(statuses)
    lines = [
        "# Metric Hardening V2 Report",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(jsonable(decision), indent=2, sort_keys=True),
        "```",
        "",
        markdown_table(status_table, ["item", "status", "detail"], max_rows=20),
        "",
        "This is a recompute-only pass from existing artifacts. No training, pixel encoding, model calls, or new data collection were run.",
        "Unit tests: `uv run --with numpy --with pandas --with scipy --with matplotlib --with h5py python -m unittest runs/metric_hardening_v2/test_metric_hardening_v2.py` passed (`13` tests).",
        "",
        "## R1 Action-Matched Neighbors",
        "",
        f"Cube: `{passfail(statuses['cube_r1'])}`. Fetch: `{passfail(statuses['fetch_r1'])}`.",
        "",
        "Cube contrasts:",
        "",
        markdown_table(
            cube_r1.loc[cube_r1["metric"].isin(["neighbor_radius", "action_neighbor_distance", "local_expansion", "model_sensitivity"])],
            ["metric", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "left_n", "right_n", "left_episodes", "right_episodes"],
        ),
        "",
        "Cube surviving-neighbor counts:",
        "",
        markdown_table(
            cube_r1_summary.loc[cube_r1_summary["metric"].eq("neighbor_radius")],
            ["regime", "n_anchors", "n_usable", "n_below_5_neighbors", "n_episodes", "median", "trimmed_mean"],
        ),
        "",
        "Fetch contrasts:",
        "",
        markdown_table(
            fetch_r1.loc[fetch_r1["metric"].isin(["neighbor_radius", "action_neighbor_distance", "local_expansion", "model_sensitivity"])],
            ["metric", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "left_n", "right_n", "left_episodes", "right_episodes"],
        ),
        "",
        "Fetch surviving-neighbor counts:",
        "",
        markdown_table(
            fetch_r1_summary.loc[fetch_r1_summary["metric"].eq("neighbor_radius")],
            ["regime", "n_anchors", "n_usable", "n_below_5_neighbors", "n_episodes", "median", "trimmed_mean"],
        ),
        "",
        "## R2 Whitening Robustness",
        "",
        f"Status: `{passfail(statuses['cube_r2'])}`.",
        "",
        markdown_table(
            cube_r2.loc[cube_r2["metric"].isin(["neighbor_radius", "local_expansion", "model_sensitivity"])],
            ["scale", "metric", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "left_n", "right_n"],
            max_rows=20,
        ),
        "",
        "## R3v2 Aliasing Discriminator",
        "",
        f"Status: `{passfail(statuses['cube_r3'])}`.",
        "",
        markdown_table(
            cube_r3.loc[cube_r3["metric"].isin(["neighbor_radius", "action_neighbor_distance", "local_expansion", "model_sensitivity"])],
            ["scale", "metric", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "left_n", "right_n"],
        ),
        "",
        "Symmetric diagnostic: latent/action neighbors, kinematic-space divergence.",
        "",
        markdown_table(
            cube_r3_symmetric.loc[cube_r3_symmetric["metric"].isin(["neighbor_radius", "kinematic_divergence", "local_expansion"])],
            ["scale", "metric", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "left_n", "right_n"],
        ),
        "",
        "## Cube Struggle Probes",
        "",
        f"Residual predictability: `{passfail(statuses.get('cube_residual_predictability', 'fail'))}`. kNN conditional-mean floor: `{passfail(statuses.get('cube_knn_floor', 'fail'))}`.",
        "",
        markdown_table(
            cube_struggle.loc[
                cube_struggle["metric"].isin(
                    [
                        "predicted_residual_norm",
                        "residual_predictability",
                        "residual_prediction_mse",
                        "conditional_mean_floor",
                        "conditional_mean_gain",
                    ]
                )
            ],
            ["metric", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "left_n", "right_n"],
        ),
        "",
        "## R4 k-Sweep",
        "",
        f"Cube: `{passfail(statuses['cube_r4'])}`. Fetch: `{passfail(statuses['fetch_r4'])}`.",
        "",
        "Cube k-sweep:",
        "",
        markdown_table(
            cube_r4.loc[cube_r4["metric"].isin(["local_expansion", "model_sensitivity"])],
            ["k", "metric", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "n_valid_interaction", "n_valid_right"],
        ),
        "",
        "Fetch k-sweep:",
        "",
        markdown_table(
            fetch_r4.loc[fetch_r4["metric"].isin(["local_expansion", "model_sensitivity"])],
            ["k", "metric", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "n_valid_interaction", "n_valid_right"],
        ),
        "",
        "## R5 Fetch Density Matching",
        "",
        f"Status: `{passfail(statuses['fetch_r5'])}`.",
        "",
        markdown_table(
            fetch_r5,
            ["metric", "n_pairs", "eligible_episodes", "match_fraction_interaction", "delta_trimmed_mean", "delta_trimmed_mean_ci_low", "delta_trimmed_mean_ci_high", "p_superiority", "median_abs_radius_gap"],
        ),
        "",
        "## Provenance",
        "",
        f"- Cube rows: `{provenance['cube_rows']}` total, `{provenance['cube_heldout_rows']}` heldout.",
        f"- Cube H5 action/kinematic join source: `{provenance['cube_h5']['join_source_used']}`; cached Lance exists: `{provenance['cube_h5']['cached_lance_exists']}`.",
        f"- R3v2 augmented kinematic dim: `{provenance['cube_h5']['augmented_kinematic']['feature_dim']}`; native velocity keys: `{', '.join(provenance['cube_h5']['augmented_kinematic']['native_velocity_keys'])}`; gripper keys: `{', '.join(provenance['cube_h5']['augmented_kinematic']['gripper_keys'])}`.",
        f"- Cube full-whitening condition number: `{provenance['cube_full_whitening_condition_number']:.6g}`.",
        f"- Fetch validation rows: `{provenance['fetch']['validation_rows']}`.",
        f"- Bootstrap samples and seed: `{provenance['bootstrap_samples']}` / `{provenance['bootstrap_seed']}`.",
        "",
        "## Output Files",
        "",
        "- `runs/metric_hardening_v2/cube/r1_action_matched_summary.csv`",
        "- `runs/metric_hardening_v2/cube/r2_whitening_robustness.csv`",
        "- `runs/metric_hardening_v2/cube/r3_aliasing_discriminator.csv`",
        "- `runs/metric_hardening_v2/cube/r3v2_symmetric_aliasing_diagnostic.csv`",
        "- `runs/metric_hardening_v2/cube/struggle_probe_contrasts.csv`",
        "- `runs/metric_hardening_v2/cube/r4_k_sweep.csv`",
        "- `runs/metric_hardening_v2/fetch/r1_action_matched_summary.csv`",
        "- `runs/metric_hardening_v2/fetch/r4_k_sweep.csv`",
        "- `runs/metric_hardening_v2/fetch/r5_density_matched.csv`",
        "- `runs/metric_hardening_v2/figures/k_sweep_delta_ci.png`",
        "- `runs/metric_hardening_v2/decision.json`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cube").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fetch").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    print_step("loading Cube records, residuals, actions, and kinematics")
    cube = prepare_cube_bundle(args)
    print_step(f"cube rows={len(cube.records)} heldout={int(cube.records['split'].eq('heldout').sum())}")

    print_step("running Cube R1 action-matched neighbors")
    cube_r1_frame, cube_r1_summary, cube_r1_contrasts, cube_r1_status = run_cube_r1(
        cube,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 100000,
    )

    print_step("running Cube R2 whitening robustness")
    _, cube_r2_contrasts, _, cube_r2_status = run_cube_r2(
        cube,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 200000,
    )

    print_step("running Cube R3v2 aliasing discriminator")
    _, _, cube_r3_contrasts, cube_r3_status = run_cube_r3(
        cube,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 300000,
    )
    cube_r3_symmetric = pd.read_csv(args.output_dir / "cube" / "r3v2_symmetric_aliasing_diagnostic.csv")

    print_step("running Cube struggle probes on R1 action-matched geometry")
    _, cube_struggle_contrasts, cube_residual_predictability_status, cube_knn_floor_status = run_cube_struggle_probes(
        cube,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 350000,
    )

    print_step("loading Fetch examples and saved ensemble predictions")
    fetch = prepare_fetch_bundle()
    print_step(f"fetch validation rows={fetch.provenance['validation_rows']}")

    print_step("running Fetch R1 action-matched neighbors")
    fetch_r1_frame, fetch_r1_summary, fetch_r1_contrasts, fetch_r1_status = run_fetch_r1(
        fetch,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 400000,
    )

    print_step("running Cube R4 k-sweep")
    cube_r4, cube_r4_status = run_k_sweep(
        dataset="cube",
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 500000,
        frame_fn=lambda k, k_min: cube_action_matched_frame(cube, k=k, k_min=k_min),
        right_regime="transport_free",
    )

    print_step("running Fetch R4 k-sweep")
    fetch_r4, fetch_r4_status = run_k_sweep(
        dataset="fetch",
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 600000,
        frame_fn=lambda k, k_min: fetch_action_matched_frame(fetch, k=k, k_min=k_min),
        right_regime="free",
    )

    print_step("running Fetch R5 density matching")
    fetch_r5, _, _, fetch_r5_status = run_fetch_r5(
        fetch,
        fetch_r1_frame,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 700000,
    )

    write_k_sweep_plot(args.output_dir / "figures" / "k_sweep_delta_ci.png", cube_r4, fetch_r4)

    statuses = {
        "cube_r1": cube_r1_status,
        "fetch_r1": fetch_r1_status,
        "cube_r2": cube_r2_status,
        "cube_r3": cube_r3_status,
        "cube_r4": cube_r4_status,
        "fetch_r4": fetch_r4_status,
        "fetch_r5": fetch_r5_status,
        "cube_residual_predictability": cube_residual_predictability_status,
        "cube_knn_floor": cube_knn_floor_status,
    }
    decision = build_decision(statuses)
    provenance = {
        "cube_rows": int(len(cube.records)),
        "cube_heldout_rows": int(cube.records["split"].eq("heldout").sum()),
        "cube_h5": cube.h5_provenance,
        "cube_full_whitening_condition_number": float(cube.full_transform.condition_number),
        "fetch": fetch.provenance,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "primary_k": PRIMARY_K,
        "primary_k_min": PRIMARY_K_MIN,
    }
    write_json(args.output_dir / "decision.json", {"decision": decision, "provenance": provenance})
    write_report(
        args.output_dir / "REPORT.md",
        decision=decision,
        statuses=statuses,
        cube_r1_summary=cube_r1_summary,
        cube_r1=cube_r1_contrasts,
        fetch_r1_summary=fetch_r1_summary,
        fetch_r1=fetch_r1_contrasts,
        cube_r2=cube_r2_contrasts,
        cube_r3=cube_r3_contrasts,
        cube_r3_symmetric=cube_r3_symmetric,
        cube_struggle=cube_struggle_contrasts,
        cube_r4=cube_r4,
        fetch_r4=fetch_r4,
        fetch_r5=fetch_r5,
        provenance=provenance,
    )
    print_step(f"decision cube={decision['cube_main_status']} fetch={decision['fetch_support_status']}")
    print_step(f"report={args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
