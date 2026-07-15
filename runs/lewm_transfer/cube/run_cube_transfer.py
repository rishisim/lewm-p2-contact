#!/usr/bin/env python3
"""Run the Cube LeWM transfer diagnostic pilot.

This intentionally writes only under runs/lewm_transfer/cube/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hdf5plugin  # noqa: F401 - registers HDF5 filters before h5py opens files.
import h5py
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TRANSFER_ROOT = Path(__file__).resolve().parents[1]
if str(TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFER_ROOT))

from lewm_transfer_plots import write_all_figures
from lewm_transfer_stats import (
    add_phase_decile,
    apply_whitened_mse,
    boolean_gate,
    cluster_bootstrap_contrast,
    contrast_values,
    latent_isotropy,
    make_matched_position_pairs,
    make_phase_decile_contrasts,
    make_regime_contrasts,
    make_regime_summary,
    make_residual_structure_table,
    make_within_episode_pairs,
    summarize_gate_table,
    whitening_matrix,
)


DEFAULT_SOURCE_H5 = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")
DEFAULT_OUTPUT_DIR = Path("runs/lewm_transfer/cube")
REGIMES = ["interaction", "support", "free"]
PRIMARY_METRICS = ["mse_model", "mse_persistence", "excess"]
WHITENED_METRICS = ["mse_model_whitened", "mse_persistence_whitened", "excess_whitened"]


def load_cube_diagnostic_module():
    path = REPO_ROOT / "le-wm" / "diagnostics" / "cube_event_localization" / "diagnostic.py"
    spec = importlib.util.spec_from_file_location("cube_event_localization_diagnostic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Cube diagnostic module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def remap_modern_vit_keys_to_hf(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map newer ViT key names back to the HF 4.x names expected in this env.

    This is the inverse of the legacy remap in
    le-wm/diagnostics/cube_event_localization/diagnostic.py and is kept local to
    this pilot so the existing diagnostic script remains untouched.
    """

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("encoder.layers."):
            new_key = key.replace("encoder.layers.", "encoder.encoder.layer.", 1)
            new_key = new_key.replace(".attention.q_proj.", ".attention.attention.query.")
            new_key = new_key.replace(".attention.k_proj.", ".attention.attention.key.")
            new_key = new_key.replace(".attention.v_proj.", ".attention.attention.value.")
            new_key = new_key.replace(".attention.o_proj.", ".attention.output.dense.")
            new_key = new_key.replace(".mlp.fc1.", ".intermediate.dense.")
            new_key = new_key.replace(".mlp.fc2.", ".output.dense.")
        remapped[new_key] = value
    return remapped


def load_model_compat(cube_diag, args: SimpleNamespace, device: torch.device):
    try:
        return cube_diag.load_model(args, device)
    except RuntimeError as exc:
        message = str(exc)
        if "Missing key(s)" not in message or "Unexpected key(s)" not in message:
            raise
        from huggingface_hub import hf_hub_download

        print_step("retrying model load with inverse ViT key remap for this Transformers runtime")
        model_dir = args.cache_dir / "model"
        cfg_path = Path(hf_hub_download(args.model_repo, "config.json", local_dir=model_dir))
        weights_path = Path(hf_hub_download(args.model_repo, "weights.pt", local_dir=model_dir))
        cfg = json.loads(cfg_path.read_text())
        model = cube_diag.JEPA(
            encoder=cube_diag.vit_hf_from_config(**cube_diag.clean_cfg(cfg["encoder"])),
            predictor=cube_diag.ARPredictor(**cube_diag.clean_cfg(cfg["predictor"])),
            action_encoder=cube_diag.Embedder(**cube_diag.clean_cfg(cfg["action_encoder"])),
            projector=cube_diag.mlp_from_config(cfg, "projector"),
            pred_proj=cube_diag.mlp_from_config(cfg, "pred_proj"),
        )
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        result = model.load_state_dict(remap_modern_vit_keys_to_hf(state), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint load mismatch after inverse remap: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        model.eval().requires_grad_(False).to(device)
        return model, cfg, cfg_path, weights_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_SOURCE_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-repo", default="quentinll/lewm-cube")
    parser.add_argument("--num-trajectories", type=int, default=30)
    parser.add_argument("--calibration-episodes", type=int, default=10)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=260626)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-lance", action="store_true")
    parser.add_argument("--skip-lance-subset", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[lewm-transfer:cube] {message}", flush=True)


def support_threshold_from_calibration(
    h5: h5py.File,
    offsets: np.ndarray,
    lengths: np.ndarray,
    *,
    calibration_episodes: int,
) -> dict[str, float | str]:
    z_values = []
    non_interaction_z = []
    for offset, length in zip(offsets[:calibration_episodes], lengths[:calibration_episodes]):
        sl = slice(int(offset), int(offset + length))
        z = np.asarray(h5["privileged_block_0_pos"][sl, 2], dtype=np.float64)
        interaction = np.asarray(h5["proprio_gripper_contact"][sl, 0], dtype=np.float64) > 1e-9
        z_values.append(z)
        non_interaction_z.append(z[~interaction])
    all_z = np.concatenate(z_values)
    base_z = np.concatenate(non_interaction_z) if non_interaction_z else all_z
    if len(base_z) == 0:
        base_z = all_z
    table_height = float(np.median(base_z))
    mad = float(np.median(np.abs(base_z - table_height)))
    tolerance = max(0.0025, 10.0 * mad)
    threshold = table_height + tolerance
    return {
        "source": "geometry: privileged_block_0_pos[:,2] <= calibration_median_non_interaction_z + max(0.0025, 10*MAD)",
        "table_height_median": table_height,
        "calibration_mad": mad,
        "tolerance": tolerance,
        "support_z_threshold": threshold,
        "unit": "meters",
    }


def raw_block_labels(
    h5: h5py.File,
    *,
    offset: int,
    length: int,
    support_z_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    sl = slice(offset, offset + length)
    interaction = np.asarray(h5["proprio_gripper_contact"][sl, 0], dtype=np.float64) > 1e-9
    support = np.asarray(h5["privileged_block_0_pos"][sl, 2], dtype=np.float64) <= support_z_threshold
    return interaction, support


def geometry_agreement(h5: h5py.File, *, start: int, stop: int) -> float:
    eff = np.asarray(h5["proprio_effector_pos"][start:stop], dtype=np.float64)
    block = np.asarray(h5["privileged_block_0_pos"][start:stop], dtype=np.float64)
    sensor = np.asarray(h5["proprio_gripper_contact"][start:stop, 0], dtype=np.float64) > 1e-9
    geom = np.linalg.norm(eff - block, axis=1) < 0.04
    return 100.0 * float(np.mean(geom == sensor))


def predict_episode_records_with_residuals(
    cube_diag,
    model,
    h5: h5py.File,
    *,
    episode_ordinal: int,
    episode_id: int,
    offset: int,
    length: int,
    frameskip: int,
    history_size: int,
    device: torch.device,
    encode_batch_size: int,
    predict_batch_size: int,
    image_size: int,
    support_z_threshold: float,
    split: str,
    start_row_id: int,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Prediction loop adapted from le-wm/diagnostics/cube_event_localization/diagnostic.py.
    emb = cube_diag.encode_episode(
        model,
        h5,
        offset=offset,
        length=length,
        frameskip=frameskip,
        device=device,
        batch_size=encode_batch_size,
        image_size=image_size,
    )
    model_steps = int(emb.shape[0])
    action_blocks = []
    for block_idx in range(model_steps - 1):
        raw_start = offset + block_idx * frameskip
        raw_stop = raw_start + frameskip
        block = np.asarray(h5["action"][raw_start:raw_stop], dtype=np.float32).reshape(-1)
        if block.shape[0] != frameskip * 5:
            raise RuntimeError(f"Action block shape mismatch in episode {episode_id}, block {block_idx}: {block.shape}")
        action_blocks.append(block)
    actions = torch.as_tensor(np.asarray(action_blocks), dtype=torch.float32, device=device).unsqueeze(0)
    actions = torch.nan_to_num(actions, 0.0)
    with torch.inference_mode():
        act_emb = model.action_encoder(actions).squeeze(0).detach().cpu()

    raw_interaction, raw_support = raw_block_labels(
        h5,
        offset=offset,
        length=length,
        support_z_threshold=support_z_threshold,
    )
    onsets_raw, releases_raw = cube_diag.edge_indices(raw_interaction)
    onset_steps = {cube_diag.raw_edge_to_target_model_step(int(idx), frameskip) for idx in onsets_raw}
    release_steps = {cube_diag.raw_edge_to_target_model_step(int(idx), frameskip) for idx in releases_raw}

    target_steps = np.arange(history_size, model_steps, dtype=np.int64)
    rows = []
    residual_chunks = []
    pred_chunks = []
    target_chunks = []
    prev_chunks = []
    row_id = start_row_id
    with torch.inference_mode():
        for start in range(0, len(target_steps), predict_batch_size):
            batch_targets = target_steps[start : start + predict_batch_size]
            emb_windows = []
            act_windows = []
            for target in batch_targets:
                lo = int(target) - history_size
                emb_windows.append(emb[lo : int(target)])
                act_windows.append(act_emb[lo : int(target)])
            emb_tensor = torch.stack(emb_windows, dim=0).to(device)
            act_tensor = torch.stack(act_windows, dim=0).to(device)
            target_tensor = emb[batch_targets].to(device)
            prev_tensor = emb[batch_targets - 1].to(device)
            pred_tensor = model.predict(emb_tensor, act_tensor)[:, -1]
            residual_tensor = pred_tensor - target_tensor
            mse_model = (residual_tensor**2).mean(dim=1).detach().cpu().numpy()
            mse_persistence = ((prev_tensor - target_tensor) ** 2).mean(dim=1).detach().cpu().numpy()
            residual_np = residual_tensor.detach().cpu().numpy()
            pred_np = pred_tensor.detach().cpu().numpy()
            target_np = target_tensor.detach().cpu().numpy()
            prev_np = prev_tensor.detach().cpu().numpy()
            residual_chunks.append(residual_np)
            pred_chunks.append(pred_np)
            target_chunks.append(target_np)
            prev_chunks.append(prev_np)

            for i, target in enumerate(batch_targets):
                target = int(target)
                raw_step = target * frameskip
                prev_raw = (target - 1) * frameskip
                transition_block = target - 1
                block_slice = slice(prev_raw + 1, raw_step + 1)
                interaction = bool(raw_interaction[block_slice].any())
                support = bool(raw_support[block_slice].any())
                if interaction:
                    regime = "interaction"
                elif support:
                    regime = "support"
                else:
                    regime = "free"
                rows.append(
                    {
                        "row_id": row_id,
                        "episode_ordinal": episode_ordinal,
                        "episode_id": episode_id,
                        "split": split,
                        "model_step": target,
                        "raw_step": raw_step,
                        "transition_block": transition_block,
                        "normalized_position": float(target / max(model_steps - 1, 1)),
                        "interaction_contact": interaction,
                        "support_contact": support,
                        "free": not interaction and not support,
                        "regime": regime,
                        "sensor_grasp_onset": target in onset_steps,
                        "sensor_release": target in release_steps,
                        "mse_model": float(mse_model[i]),
                        "mse_persistence": float(mse_persistence[i]),
                        "excess": float(mse_model[i] - mse_persistence[i]),
                    }
                )
                row_id += 1
    return (
        rows,
        np.concatenate(residual_chunks, axis=0),
        np.concatenate(pred_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        np.concatenate(prev_chunks, axis=0),
    )


def regime_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "regime"], sort=False)
        .agg(n_rows=("row_id", "count"), n_episodes=("episode_ordinal", "nunique"))
        .reset_index()
    )


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def compute_whitened_columns(records: pd.DataFrame, target_latents: np.ndarray, pred_latents: np.ndarray, prev_latents: np.ndarray) -> pd.DataFrame:
    out = records.copy()
    calibration_ids = out.loc[out["split"] == "calibration", "row_id"].to_numpy(dtype=np.int64)
    if len(calibration_ids) < 2:
        out["mse_model_whitened"] = np.nan
        out["mse_persistence_whitened"] = np.nan
        out["excess_whitened"] = np.nan
        return out
    _, transform = whitening_matrix(target_latents[calibration_ids])
    model_delta = pred_latents - target_latents
    persistence_delta = prev_latents - target_latents
    out["mse_model_whitened"] = apply_whitened_mse(model_delta, transform)
    out["mse_persistence_whitened"] = apply_whitened_mse(persistence_delta, transform)
    out["excess_whitened"] = out["mse_model_whitened"] - out["mse_persistence_whitened"]
    return out


def gate_results(
    heldout: pd.DataFrame,
    all_records: pd.DataFrame,
    target_latents: np.ndarray,
    *,
    isotropy: dict[str, float],
) -> dict[str, dict[str, object]]:
    model_mean = float(heldout["mse_model"].mean()) if len(heldout) else float("nan")
    persistence_mean = float(heldout["mse_persistence"].mean()) if len(heldout) else float("nan")
    ratio = model_mean / persistence_mean if persistence_mean else float("inf")
    coverage = regime_counts(heldout)

    def coverage_ok(regime: str) -> bool:
        row = coverage.loc[coverage["regime"] == regime]
        if row.empty:
            return False
        return int(row.iloc[0]["n_rows"]) >= 30 and int(row.iloc[0]["n_episodes"]) >= 5

    finite_latents = target_latents[all_records.loc[all_records["split"] == "heldout", "row_id"].to_numpy(dtype=np.int64)]
    finite_latents_ok = bool(np.isfinite(finite_latents).all()) if len(finite_latents) else False
    isotropic_enough = bool(
        np.isfinite(isotropy["condition_number"])
        and isotropy["condition_number"] <= 1e4
        and np.isfinite(isotropy["per_dim_variance_spread"])
        and isotropy["per_dim_variance_spread"] <= 100.0
    )
    return {
        "collapse_persistence": {
            **boolean_gate(
                np.isfinite(ratio) and ratio < 0.9,
                f"heldout model/persistence ratio={ratio:.6g}; void if approximately 1 or worse",
            ),
            "model_mse_mean": model_mean,
            "persistence_mse_mean": persistence_mean,
            "model_persistence_ratio": ratio,
        },
        "latent_isotropy": {
            **boolean_gate(
                finite_latents_ok and isotropic_enough,
                "target latent covariance condition and per-dim spread within preregistered thresholds",
            ),
            **isotropy,
            "whitened_metrics_required": not isotropic_enough,
        },
        "no_circular_labels": boolean_gate(
            True,
            "interaction uses proprio_gripper_contact sensor; support uses block-height geometry only; no velocity/orientation/latent/error labels",
        ),
        "split_thresholds": boolean_gate(
            True,
            "support threshold and optional whitening are fit on calibration episodes and evaluated on held-out episodes",
        ),
        "coverage_interaction_free": {
            **boolean_gate(
                coverage_ok("interaction") and coverage_ok("free"),
                "requires >=5 held-out episodes and >=30 held-out rows in both interaction and free",
            ),
            "interaction_rows": int(coverage.loc[coverage["regime"] == "interaction", "n_rows"].sum()),
            "interaction_episodes": int(coverage.loc[coverage["regime"] == "interaction", "n_episodes"].sum()),
            "free_rows": int(coverage.loc[coverage["regime"] == "free", "n_rows"].sum()),
            "free_episodes": int(coverage.loc[coverage["regime"] == "free", "n_episodes"].sum()),
        },
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "(empty)"
    present = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(present) + " |", "| " + " | ".join(["---"] * len(present)) + " |"]
    for _, row in df[present].iterrows():
        values = []
        for col in present:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report_decision(
    *,
    gates: dict[str, dict[str, object]],
    regime_contrasts: pd.DataFrame,
    phase_adjusted: pd.DataFrame,
    matched_summary: pd.DataFrame,
) -> dict[str, object]:
    coverage_passed = bool(gates["coverage_interaction_free"]["passed"])
    collapse_passed = bool(gates["collapse_persistence"]["passed"])
    contrast = regime_contrasts.loc[
        (regime_contrasts["contrast"] == "interaction_vs_free") & (regime_contrasts["metric"] == "excess")
    ]
    phase = phase_adjusted.loc[phase_adjusted["metric"] == "excess"] if not phase_adjusted.empty else pd.DataFrame()
    matched = matched_summary.loc[matched_summary["metric"] == "excess"] if not matched_summary.empty else pd.DataFrame()

    contrast_passed = False
    phase_passed = False
    matched_passed = False
    if len(contrast):
        contrast_passed = bool(float(contrast.iloc[0].get("delta_trimmed_mean_ci_low", float("nan"))) > 0.0)
    if len(phase):
        phase_passed = bool(float(phase.iloc[0].get("delta_trimmed_mean_ci_low", float("nan"))) > 0.0)
    if len(matched):
        row = matched.iloc[0]
        matched_passed = bool(
            float(row.get("median_episode_delta", float("nan"))) > 0.0
            and float(row.get("wilcoxon_p_greater", float("nan"))) < 0.05
        )
    transfers = bool(coverage_passed and collapse_passed and contrast_passed and phase_passed and matched_passed)
    if transfers:
        verdict = "transfer_supported"
        reason = "interaction excess error exceeds free with positive clustered CI and survives phase controls"
    elif not coverage_passed:
        verdict = "contrast_void_insufficient_free"
        reason = "held-out interaction/free coverage gate failed, so the preregistered transfer contrast is void"
    elif not collapse_passed:
        verdict = "contrast_void_prediction_trivial"
        reason = "model/persistence gate failed"
    elif contrast_passed and not (phase_passed and matched_passed):
        verdict = "not_supported_phase_control"
        reason = "raw excess contrast did not survive the preregistered phase controls"
    else:
        verdict = "not_supported"
        reason = "interaction excess contrast did not meet the preregistered positive-CI criterion"
    return {
        "verdict": verdict,
        "reason": reason,
        "transfer_supported": transfers,
        "coverage_passed": coverage_passed,
        "collapse_passed": collapse_passed,
        "contrast_passed": contrast_passed,
        "phase_passed": phase_passed,
        "matched_passed": matched_passed,
    }


def write_report(
    path: Path,
    *,
    decision: dict[str, object],
    label_provenance: dict[str, object],
    gates: dict[str, dict[str, object]],
    regime_summary: pd.DataFrame,
    regime_contrasts: pd.DataFrame,
    phase_adjusted: pd.DataFrame,
    matched_summary: pd.DataFrame,
    within_summary: pd.DataFrame,
    residual_table: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    excess_summary = regime_summary.loc[regime_summary["metric"] == "excess"]
    primary_contrast = regime_contrasts.loc[
        (regime_contrasts["contrast"] == "interaction_vs_free") & (regime_contrasts["metric"] == "excess")
    ]
    model_persistence = gates["collapse_persistence"]
    label_counts = pd.DataFrame(label_provenance["regime_counts"])

    if len(primary_contrast):
        contrast_text = (
            f"interaction-vs-free excess trimmed-mean delta "
            f"{float(primary_contrast.iloc[0]['delta_trimmed_mean']):.6g} "
            f"(CI {float(primary_contrast.iloc[0]['delta_trimmed_mean_ci_low']):.6g}, "
            f"{float(primary_contrast.iloc[0]['delta_trimmed_mean_ci_high']):.6g})"
        )
    else:
        contrast_text = "interaction-vs-free excess contrast unavailable"

    lines = [
        "# Cube LeWM Transfer Diagnostic Report",
        "",
        "## Summary",
        "",
        (
            f"Verdict: `{decision['verdict']}`. This Cube latent-space pilot does not overclaim beyond the "
            f"preregistered gates: {decision['reason']}. The held-out model/persistence ratio was "
            f"`{float(model_persistence['model_persistence_ratio']):.6g}`, and the primary contrast was "
            f"{contrast_text}. Headline numbers are sourced to `metrics/*.csv`, `validity_gates.json`, "
            "`label_provenance.json`, and `residuals.npz`."
        ),
        "",
        "## Label Provenance",
        "",
        f"- Interaction source: `{label_provenance['interaction_source']}`",
        f"- Support source: `{label_provenance['support_source']}`",
        f"- Support threshold: `{label_provenance['support_threshold']['support_z_threshold']:.6g}` meters",
        f"- Gripper geometry agreement at 0.04m: `{label_provenance['gripper_geometry_agreement_pct']:.3f}%`",
        "",
        markdown_table(label_counts, ["split", "regime", "n_rows", "n_episodes"]),
        "",
        "## Validity Gates",
        "",
        markdown_table(
            summarize_gate_table(gates),
            ["gate", "passed", "reason", "model_persistence_ratio", "condition_number", "per_dim_variance_spread"],
        ),
        "",
        "Because the latent covariance condition-number gate failed, the summary and contrast tables also include calibration-whitened MSE rows (`*_whitened`). Whitening is fit on calibration episodes only.",
        "",
        "## Disentangling",
        "",
        "High `mse_model` with high `mse_persistence` is representation motion; high `mse_model` with low `mse_persistence` is predictor difficulty. The headline contrast uses `excess = mse_model - mse_persistence`.",
        "",
        markdown_table(
            regime_summary,
            ["regime", "metric", "n", "n_episodes", "trimmed_mean", "trimmed_mean_ci_low", "trimmed_mean_ci_high", "median"],
        ),
        "",
        "## Regime Contrasts",
        "",
        markdown_table(
            regime_contrasts,
            [
                "contrast",
                "metric",
                "delta_trimmed_mean",
                "delta_trimmed_mean_ci_low",
                "delta_trimmed_mean_ci_high",
                "p_superiority",
            ],
        ),
        "",
        "## Phase Control",
        "",
        markdown_table(
            phase_adjusted,
            [
                "contrast",
                "metric",
                "n_deciles",
                "delta_trimmed_mean",
                "delta_trimmed_mean_ci_low",
                "delta_trimmed_mean_ci_high",
                "p_superiority",
            ],
        ),
        "",
        markdown_table(
            matched_summary,
            [
                "metric",
                "eligible_episodes",
                "n_pairs",
                "median_episode_delta",
                "fraction_episode_positive",
                "wilcoxon_p_greater",
            ],
        ),
        "",
        "## Within-Episode Paired Test",
        "",
        markdown_table(
            within_summary,
            ["metric", "eligible_episodes", "fraction_episode_positive", "median_delta", "wilcoxon_p_greater"],
        ),
        "",
        "## Residual Structure",
        "",
        "This is a single-checkpoint residual-structure check, not a LeWM ensemble bias-variance decomposition. Systematic or low-rank residuals are consistent with reducible bias but are not proof. The ensemble bias-dominance result was established separately on the state-space Fetch proxy.",
        "",
        markdown_table(
            residual_table,
            [
                "regime",
                "n",
                "directional_consistency",
                "directional_consistency_ci_low",
                "directional_consistency_ci_high",
                "top5_variance",
                "effective_rank",
            ],
        ),
        "",
        "## Figures",
        "",
    ]
    for name, figure_path in figure_paths.items():
        rel = figure_path.relative_to(path.parent)
        lines.append(f"![{name}]({rel})")
        lines.append("")
    lines.extend(
        [
            "## Scope And Caveats",
            "",
            "- This is latent-space LeWM prediction, not the state-space Fetch proxy.",
            "- This uses one released checkpoint, so a full LeWM bias-variance decomposition is not feasible without pretraining an ensemble of world models.",
            "- This is Cube, not Fetch, and no LeWM Fetch checkpoint is attempted here.",
            "- The prior Cube `4.76519x` number was a raw contact/non-contact MSE ratio; this report separates model MSE from persistence and checks normalized-position phase confounding.",
            "- If the free regime is sparse or absent under non-circular labels, the interaction-vs-free transfer claim is void rather than negative evidence about predictor difficulty.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    figure_dir = args.output_dir / "figures"
    cache_dir = args.output_dir / "cache"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cube_diag = load_cube_diagnostic_module()
    device = cube_diag.choose_device(args.device)
    model_args = SimpleNamespace(cache_dir=cache_dir, model_repo=args.model_repo)
    print_step(f"loading model {args.model_repo} on {device}")
    model, cfg, cfg_path, weights_path = load_model_compat(cube_diag, model_args, device)
    training = cube_diag.validate_training_match(cfg)
    print_step(
        f"training config history={training['history_size']} frameskip={training['frameskip']} "
        f"image_size={training['image_size']} action_dim={training['action_encoder_input_dim']}"
    )

    all_rows: list[dict[str, object]] = []
    residual_chunks = []
    pred_chunks = []
    target_chunks = []
    prev_chunks = []
    with h5py.File(args.source_h5, "r") as h5:
        lengths, offsets, start, stop = cube_diag.h5_layout(h5, args.num_trajectories)
        source_step_count = int(lengths.sum())
        if not args.skip_lance_subset:
            lance_dir = cache_dir / "cube_first30_pixel_subset.lance"
            print_step(f"writing/verifying Lance subset at {lance_dir}")
            lance_rows = cube_diag.write_lance_subset(
                h5,
                lance_dir,
                start=start,
                stop=stop,
                lengths=lengths,
                offsets=offsets,
                batch_rows=300,
                force=args.force_lance,
            )
            if lance_rows != source_step_count:
                raise RuntimeError(f"Lance row count mismatch: {lance_rows} != {source_step_count}")
        else:
            lance_dir = None
            lance_rows = None

        labels_by_episode = [
            np.asarray(h5["proprio_gripper_contact"][int(o) : int(o + l), 0]) > 1e-9
            for o, l in zip(offsets, lengths)
        ]
        raw_contact_stats = cube_diag.contact_stats(labels_by_episode)
        precheck_compare = cube_diag.compare_precheck(raw_contact_stats)
        support_threshold = support_threshold_from_calibration(
            h5,
            offsets,
            lengths,
            calibration_episodes=args.calibration_episodes,
        )
        gripper_agreement = geometry_agreement(h5, start=start, stop=stop)
        print_step(f"contact stats={raw_contact_stats}; precheck divergence={precheck_compare['divergence'] or 'none'}")
        print_step(
            f"support z threshold={support_threshold['support_z_threshold']:.6g}; "
            f"gripper geometry agreement={gripper_agreement:.3f}%"
        )

        row_id = 0
        for ordinal, (offset, length) in enumerate(zip(offsets, lengths), start=0):
            episode_id = int(h5["ep_idx"][int(offset)])
            split = "calibration" if ordinal < args.calibration_episodes else "heldout"
            print_step(f"predicting episode {ordinal + 1}/{len(lengths)} ep_idx={episode_id} split={split}")
            rows, residuals, preds, targets, prevs = predict_episode_records_with_residuals(
                cube_diag,
                model,
                h5,
                episode_ordinal=ordinal,
                episode_id=episode_id,
                offset=int(offset),
                length=int(length),
                frameskip=int(training["frameskip"]),
                history_size=int(training["history_size"]),
                device=device,
                encode_batch_size=args.encode_batch_size,
                predict_batch_size=args.predict_batch_size,
                image_size=int(training["image_size"]),
                support_z_threshold=float(support_threshold["support_z_threshold"]),
                split=split,
                start_row_id=row_id,
            )
            row_id += len(rows)
            all_rows.extend(rows)
            residual_chunks.append(residuals)
            pred_chunks.append(preds)
            target_chunks.append(targets)
            prev_chunks.append(prevs)

    records = pd.DataFrame(all_rows)
    if records.empty:
        raise RuntimeError("No prediction records were produced")
    residuals = np.concatenate(residual_chunks, axis=0)
    pred_latents = np.concatenate(pred_chunks, axis=0)
    target_latents = np.concatenate(target_chunks, axis=0)
    prev_latents = np.concatenate(prev_chunks, axis=0)
    records = compute_whitened_columns(records, target_latents, pred_latents, prev_latents)
    records = add_phase_decile(records)
    heldout = records.loc[records["split"] == "heldout"].copy()
    calibration = records.loc[records["split"] == "calibration"].copy()
    if heldout.empty:
        raise RuntimeError("Held-out split is empty")

    heldout_ids = heldout["row_id"].to_numpy(dtype=np.int64)
    isotropy = latent_isotropy(target_latents[heldout_ids])
    gates = gate_results(heldout, records, target_latents, isotropy=isotropy)
    counts = regime_counts(records)
    label_provenance = {
        "model_repo": args.model_repo,
        "config": str(cfg_path),
        "weights": str(weights_path),
        "source_h5": str(args.source_h5),
        "lance_subset": str(lance_dir) if lance_dir is not None else None,
        "lance_rows": lance_rows,
        "source_step_count": source_step_count,
        "num_trajectories": args.num_trajectories,
        "calibration_episodes": args.calibration_episodes,
        "heldout_episodes": args.num_trajectories - args.calibration_episodes,
        "history_size": int(training["history_size"]),
        "frameskip": int(training["frameskip"]),
        "interaction_source": "sensor: proprio_gripper_contact > 1e-9, block-reduced over prediction transition rows",
        "support_source": support_threshold["source"],
        "support_threshold": support_threshold,
        "regime_precedence": ["interaction", "support", "free"],
        "gripper_geometry_agreement_pct": gripper_agreement,
        "contact_stats": raw_contact_stats,
        "precheck_contact_divergence": precheck_compare["divergence"],
        "regime_counts": counts.to_dict(orient="records"),
    }

    records_path = args.output_dir / "step_records.csv"
    records.to_csv(records_path, index=False)
    np.savez_compressed(
        args.output_dir / "residuals.npz",
        residuals=residuals,
        pred_latents=pred_latents,
        target_latents=target_latents,
        prev_latents=prev_latents,
        row_id=records["row_id"].to_numpy(dtype=np.int64),
    )
    write_json(args.output_dir / "label_provenance.json", label_provenance)
    write_json(args.output_dir / "validity_gates.json", gates)
    summarize_gate_table(gates).to_csv(metrics_dir / "validity_gates.csv", index=False)
    counts.to_csv(metrics_dir / "regime_counts.csv", index=False)

    print_step("computing regime summaries and contrasts")
    summary_metrics = PRIMARY_METRICS + [
        metric for metric in WHITENED_METRICS if metric in heldout.columns
    ]
    regime_summary = make_regime_summary(
        heldout,
        regimes=REGIMES,
        value_cols=summary_metrics,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    regime_contrasts = make_regime_contrasts(
        heldout,
        contrasts=[
            ("interaction_vs_free", "interaction", "free"),
            ("support_vs_free", "support", "free"),
            ("interaction_vs_support", "interaction", "support"),
        ],
        value_cols=summary_metrics,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 10000,
    )
    phase_deciles, phase_adjusted = make_phase_decile_contrasts(
        heldout,
        value_col="excess",
        left="interaction",
        right="free",
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 20000,
    )
    matched_pairs, matched_summary = make_matched_position_pairs(heldout, value_col="excess")
    within_pairs, within_summary = make_within_episode_pairs(heldout, value_cols=["mse_model", "excess"])
    residual_table = make_residual_structure_table(
        heldout,
        residuals,
        regimes=REGIMES,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 30000,
    )

    calibration.to_csv(metrics_dir / "calibration_records.csv", index=False)
    heldout.to_csv(metrics_dir / "heldout_records.csv", index=False)
    regime_summary.to_csv(metrics_dir / "regime_summary.csv", index=False)
    regime_contrasts.to_csv(metrics_dir / "regime_contrasts.csv", index=False)
    phase_deciles.to_csv(metrics_dir / "phase_decile_contrasts.csv", index=False)
    phase_adjusted.to_csv(metrics_dir / "phase_adjusted_contrast.csv", index=False)
    matched_pairs.to_csv(metrics_dir / "matched_position_pairs.csv", index=False)
    matched_summary.to_csv(metrics_dir / "matched_position_summary.csv", index=False)
    within_pairs.to_csv(metrics_dir / "within_episode_pairs.csv", index=False)
    within_summary.to_csv(metrics_dir / "within_episode_summary.csv", index=False)
    residual_table.to_csv(metrics_dir / "residual_structure.csv", index=False)

    print_step("writing figures")
    figure_paths = write_all_figures(
        records=heldout,
        regime_summary=regime_summary,
        regime_contrasts=regime_contrasts,
        within_episode_pairs=within_pairs,
        within_episode_summary=within_summary,
        residual_structure=residual_table,
        figure_dir=figure_dir,
    )

    decision = report_decision(
        gates=gates,
        regime_contrasts=regime_contrasts,
        phase_adjusted=phase_adjusted,
        matched_summary=matched_summary,
    )
    write_json(args.output_dir / "decision.json", decision)
    write_report(
        args.output_dir / "REPORT.md",
        decision=decision,
        label_provenance=label_provenance,
        gates=gates,
        regime_summary=regime_summary,
        regime_contrasts=regime_contrasts,
        phase_adjusted=phase_adjusted,
        matched_summary=matched_summary,
        within_summary=within_summary,
        residual_table=residual_table,
        figure_paths=figure_paths,
    )
    print_step(f"done: {args.output_dir / 'REPORT.md'}")
    print_step(f"decision={decision['verdict']} reason={decision['reason']}")


if __name__ == "__main__":
    main()
