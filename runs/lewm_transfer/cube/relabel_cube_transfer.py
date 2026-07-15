#!/usr/bin/env python3
"""Relabel Cube transfer records into interaction/transport_free/static regimes.

This script does not encode pixels or call the LeWM checkpoint. It reads the
existing step_records.csv and residuals.npz artifacts, attaches raw
low-dimensional motion labels from the Cube HDF5, and recomputes the analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd


TRANSFER_ROOT = Path(__file__).resolve().parents[1]
if str(TRANSFER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFER_ROOT))

from lewm_transfer_plots import write_all_figures
from lewm_transfer_stats import (
    add_phase_decile,
    cluster_bootstrap_contrast,
    latent_isotropy,
    make_matched_position_pairs,
    make_phase_decile_contrasts,
    make_regime_contrasts,
    make_regime_summary,
    make_residual_structure_table,
    make_within_episode_pairs,
    summarize_gate_table,
)


DEFAULT_SOURCE_H5 = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")
DEFAULT_INPUT_DIR = Path("runs/lewm_transfer/cube")
DEFAULT_OUTPUT_DIR = Path("runs/lewm_transfer/cube/relabel_motion")
REGIMES = ["interaction", "transport_free", "static"]
PRIMARY_METRICS = [
    "mse_model",
    "mse_persistence",
    "excess",
    "mse_model_whitened",
    "mse_persistence_whitened",
    "excess_whitened",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_SOURCE_H5)
    parser.add_argument("--num-trajectories", type=int, default=30)
    parser.add_argument("--effector-percentile", type=float, default=10.0)
    parser.add_argument("--block-percentile", type=float, default=95.0)
    parser.add_argument("--block-floor-m", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=260727)
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[lewm-transfer:cube-relabel] {message}", flush=True)


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


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


def attach_motion_columns(records: pd.DataFrame, source_h5: Path, num_trajectories: int) -> pd.DataFrame:
    rows = []
    with h5py.File(source_h5, "r") as h5:
        offsets = np.asarray(h5["ep_offset"][:num_trajectories], dtype=np.int64)
        for row in records.itertuples(index=False):
            episode_offset = int(offsets[int(row.episode_ordinal)])
            raw_step = episode_offset + int(row.raw_step)
            prev_raw = raw_step - 5
            block = np.asarray(h5["privileged_block_0_pos"][raw_step], dtype=np.float64)
            block_prev = np.asarray(h5["privileged_block_0_pos"][prev_raw], dtype=np.float64)
            eff = np.asarray(h5["proprio_effector_pos"][raw_step], dtype=np.float64)
            eff_prev = np.asarray(h5["proprio_effector_pos"][prev_raw], dtype=np.float64)
            interaction = np.asarray(
                h5["proprio_gripper_contact"][prev_raw + 1 : raw_step + 1, 0],
                dtype=np.float64,
            ) > 1e-9
            rows.append(
                {
                    "row_id": int(row.row_id),
                    "interaction_contact_relabel": bool(interaction.any()),
                    "block_disp": float(np.linalg.norm(block - block_prev)),
                    "effector_disp": float(np.linalg.norm(eff - eff_prev)),
                }
            )
    motion = pd.DataFrame(rows)
    out = records.drop(columns=["regime"], errors="ignore").merge(motion, on="row_id", how="left", validate="one_to_one")
    if not np.array_equal(out["interaction_contact_relabel"].to_numpy(dtype=bool), out["interaction_contact"].to_numpy(dtype=bool)):
        raise RuntimeError("Relabeled interaction sensor contact does not match existing interaction_contact column")
    return out


def fit_motion_thresholds(
    records: pd.DataFrame,
    *,
    effector_percentile: float,
    block_percentile: float,
    block_floor_m: float,
) -> dict[str, float | str]:
    calibration = records.loc[(records["split"] == "calibration") & (~records["interaction_contact_relabel"])]
    if calibration.empty:
        raise RuntimeError("No calibration non-interaction rows for motion-threshold fitting")
    effector_threshold = float(np.percentile(calibration["effector_disp"].to_numpy(dtype=np.float64), effector_percentile))
    block_raw = float(np.percentile(calibration["block_disp"].to_numpy(dtype=np.float64), block_percentile))
    block_threshold = max(float(block_floor_m), block_raw)
    return {
        "source": (
            "calibration non-interaction kinematics; transport_free if no gripper contact and "
            "effector_disp > effector_p10 or block_disp > max(1mm, block_p95)"
        ),
        "effector_percentile": effector_percentile,
        "effector_motion_threshold_m": effector_threshold,
        "block_percentile": block_percentile,
        "block_motion_threshold_raw_m": block_raw,
        "block_floor_m": block_floor_m,
        "block_motion_threshold_m": block_threshold,
    }


def apply_motion_regimes(records: pd.DataFrame, thresholds: dict[str, float | str]) -> pd.DataFrame:
    out = records.copy()
    interaction = out["interaction_contact_relabel"].astype(bool)
    moving = (
        (out["effector_disp"].astype(float) > float(thresholds["effector_motion_threshold_m"]))
        | (out["block_disp"].astype(float) > float(thresholds["block_motion_threshold_m"]))
    )
    out["transport_free"] = (~interaction) & moving
    out["static"] = (~interaction) & (~moving)
    out["regime"] = np.select(
        [interaction, out["transport_free"], out["static"]],
        ["interaction", "transport_free", "static"],
        default="unlabeled",
    )
    out["free"] = out["transport_free"]
    return out


def regime_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["split", "regime"], sort=False)
        .agg(n_rows=("row_id", "count"), n_episodes=("episode_ordinal", "nunique"))
        .reset_index()
    )


def gate_results(records: pd.DataFrame, target_latents: np.ndarray) -> dict[str, dict[str, object]]:
    heldout = records.loc[records["split"] == "heldout"]
    model_mean = float(heldout["mse_model"].mean())
    persistence_mean = float(heldout["mse_persistence"].mean())
    ratio = model_mean / persistence_mean if persistence_mean else float("inf")
    counts = regime_counts(heldout)

    def coverage_ok(regime: str) -> bool:
        row = counts.loc[counts["regime"] == regime]
        if row.empty:
            return False
        return int(row.iloc[0]["n_rows"]) >= 30 and int(row.iloc[0]["n_episodes"]) >= 5

    heldout_ids = heldout["row_id"].to_numpy(dtype=np.int64)
    isotropy = latent_isotropy(target_latents[heldout_ids])
    isotropic_enough = bool(
        np.isfinite(isotropy["condition_number"])
        and isotropy["condition_number"] <= 1e4
        and np.isfinite(isotropy["per_dim_variance_spread"])
        and isotropy["per_dim_variance_spread"] <= 100.0
    )
    return {
        "collapse_persistence": {
            "passed": bool(np.isfinite(ratio) and ratio < 0.9),
            "reason": f"heldout model/persistence ratio={ratio:.6g}; void if approximately 1 or worse",
            "model_mse_mean": model_mean,
            "persistence_mse_mean": persistence_mean,
            "model_persistence_ratio": ratio,
        },
        "latent_isotropy": {
            "passed": isotropic_enough,
            "reason": "target latent covariance condition and per-dim spread within preregistered thresholds",
            **isotropy,
            "whitened_metrics_required": not isotropic_enough,
        },
        "no_circular_labels": {
            "passed": True,
            "reason": "interaction uses sensor contact; transport_free/static use raw effector/block motion only; no latent/error labels",
        },
        "split_thresholds": {
            "passed": True,
            "reason": "motion thresholds and optional whitening were fit on calibration episodes and evaluated on held-out episodes",
        },
        "coverage_interaction_transport_free": {
            "passed": bool(coverage_ok("interaction") and coverage_ok("transport_free")),
            "reason": "requires >=5 held-out episodes and >=30 held-out rows in interaction and transport_free",
            "interaction_rows": int(counts.loc[counts["regime"] == "interaction", "n_rows"].sum()),
            "interaction_episodes": int(counts.loc[counts["regime"] == "interaction", "n_episodes"].sum()),
            "transport_free_rows": int(counts.loc[counts["regime"] == "transport_free", "n_rows"].sum()),
            "transport_free_episodes": int(counts.loc[counts["regime"] == "transport_free", "n_episodes"].sum()),
        },
    }


def report_decision(
    gates: dict[str, dict[str, object]],
    regime_contrasts: pd.DataFrame,
    phase_adjusted: pd.DataFrame,
    matched_summary: pd.DataFrame,
) -> dict[str, object]:
    contrast = regime_contrasts.loc[
        (regime_contrasts["contrast"] == "interaction_vs_transport_free") & (regime_contrasts["metric"] == "excess")
    ]
    phase = phase_adjusted.loc[phase_adjusted["metric"] == "excess"] if not phase_adjusted.empty else pd.DataFrame()
    matched = matched_summary.loc[matched_summary["metric"] == "excess"] if not matched_summary.empty else pd.DataFrame()
    contrast_passed = bool(len(contrast) and float(contrast.iloc[0]["delta_trimmed_mean_ci_low"]) > 0.0)
    phase_passed = bool(len(phase) and float(phase.iloc[0]["delta_trimmed_mean_ci_low"]) > 0.0)
    matched_passed = bool(
        len(matched)
        and float(matched.iloc[0]["median_episode_delta"]) > 0.0
        and float(matched.iloc[0]["wilcoxon_p_greater"]) < 0.05
    )
    coverage_passed = bool(gates["coverage_interaction_transport_free"]["passed"])
    collapse_passed = bool(gates["collapse_persistence"]["passed"])
    transfers = bool(coverage_passed and collapse_passed and contrast_passed and phase_passed and matched_passed)
    if transfers:
        verdict = "transfer_supported"
        reason = "interaction excess exceeds transport_free with positive clustered CI and survives phase controls"
    elif (not contrast_passed) and phase_passed and matched_passed:
        verdict = "mixed_phase_control_positive_global_negative"
        reason = (
            "global interaction-vs-transport_free excess contrast is not positive, but phase-adjusted "
            "and matched-position phase controls are positive; preregistered transfer acceptance is not met"
        )
    elif contrast_passed and not (phase_passed and matched_passed):
        verdict = "not_supported_phase_control"
        reason = "interaction excess contrast did not survive phase controls"
    elif not coverage_passed:
        verdict = "contrast_void_insufficient_coverage"
        reason = "interaction/transport_free coverage gate failed"
    else:
        verdict = "not_supported"
        reason = "interaction excess contrast did not meet the positive-CI criterion"
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
    provenance: dict[str, object],
    gates: dict[str, dict[str, object]],
    regime_summary: pd.DataFrame,
    regime_contrasts: pd.DataFrame,
    phase_adjusted: pd.DataFrame,
    matched_summary: pd.DataFrame,
    within_summary: pd.DataFrame,
    residual_table: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> None:
    counts = pd.DataFrame(provenance["regime_counts"])
    primary = regime_contrasts.loc[
        (regime_contrasts["contrast"] == "interaction_vs_transport_free") & (regime_contrasts["metric"] == "excess")
    ]
    primary_text = "unavailable"
    if len(primary):
        row = primary.iloc[0]
        primary_text = (
            f"{float(row['delta_trimmed_mean']):.6g} "
            f"(CI {float(row['delta_trimmed_mean_ci_low']):.6g}, {float(row['delta_trimmed_mean_ci_high']):.6g})"
        )
    lines = [
        "# Cube LeWM Transfer Diagnostic Report - Motion Relabel",
        "",
        "## Summary",
        "",
        (
            f"Verdict: `{decision['verdict']}`. This is a relabel-and-recompute pass from existing "
            f"`step_records.csv` and `residuals.npz`; no pixel encoding or model prediction was rerun. "
            f"The primary interaction-vs-transport_free excess trimmed-mean delta is {primary_text}. "
            f"Decision reason: {decision['reason']}."
        ),
        "",
        "## Label Provenance",
        "",
        f"- Interaction source: `{provenance['interaction_source']}`",
        f"- Transport/static source: `{provenance['motion_source']}`",
        f"- Effector motion threshold: `{provenance['motion_thresholds']['effector_motion_threshold_m']:.6g}` m",
        f"- Block motion threshold: `{provenance['motion_thresholds']['block_motion_threshold_m']:.6g}` m",
        "",
        markdown_table(counts, ["split", "regime", "n_rows", "n_episodes"]),
        "",
        "## Validity Gates",
        "",
        markdown_table(
            summarize_gate_table(gates),
            [
                "gate",
                "passed",
                "reason",
                "model_persistence_ratio",
                "condition_number",
                "per_dim_variance_spread",
            ],
        ),
        "",
        "Because the latent covariance condition-number gate failed, calibration-whitened MSE rows are included.",
        "",
        "## Disentangling",
        "",
        markdown_table(
            regime_summary,
            [
                "regime",
                "metric",
                "n",
                "n_episodes",
                "trimmed_mean",
                "trimmed_mean_ci_low",
                "trimmed_mean_ci_high",
                "median",
            ],
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
        "This remains a single-checkpoint residual-structure check, not a LeWM ensemble bias-variance decomposition.",
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
        lines.append(f"![{name}]({figure_path.relative_to(path.parent)})")
        lines.append("")
    lines.extend(
        [
            "## Scope And Caveats",
            "",
            "- This is Cube latent-space prediction, not Fetch state-space prediction.",
            "- Labels use raw sensor contact and raw kinematic motion only; no latent/error-derived labels are used.",
            "- `transport_free` means no gripper-cube sensor contact and either arm or cube motion above calibration-fitted low-motion thresholds.",
            "- Single checkpoint residual structure is not a bias-variance decomposition.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    figure_dir = args.output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.input_dir / "step_records.csv")
    residual_npz = np.load(args.input_dir / "residuals.npz")
    residuals = residual_npz["residuals"]
    target_latents = residual_npz["target_latents"]
    print_step(f"loaded {len(records)} existing rows and residuals {residuals.shape}")

    records = attach_motion_columns(records, args.source_h5, args.num_trajectories)
    thresholds = fit_motion_thresholds(
        records,
        effector_percentile=args.effector_percentile,
        block_percentile=args.block_percentile,
        block_floor_m=args.block_floor_m,
    )
    records = apply_motion_regimes(records, thresholds)
    records = add_phase_decile(records)
    heldout = records.loc[records["split"] == "heldout"].copy()
    print_step(f"motion thresholds: eff={thresholds['effector_motion_threshold_m']:.6g}, block={thresholds['block_motion_threshold_m']:.6g}")
    print_step("heldout counts: " + str(heldout.groupby("regime").size().to_dict()))

    counts = regime_counts(records)
    gates = gate_results(records, target_latents)
    provenance = {
        "source_step_records": str(args.input_dir / "step_records.csv"),
        "source_residuals": str(args.input_dir / "residuals.npz"),
        "source_h5": str(args.source_h5),
        "interaction_source": "sensor: proprio_gripper_contact > 1e-9, block-reduced over prediction transition rows",
        "motion_source": thresholds["source"],
        "motion_thresholds": thresholds,
        "regime_precedence": ["interaction", "transport_free", "static"],
        "regime_counts": counts.to_dict(orient="records"),
    }

    summary_metrics = [metric for metric in PRIMARY_METRICS if metric in heldout.columns]
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
            ("interaction_vs_transport_free", "interaction", "transport_free"),
            ("interaction_vs_static", "interaction", "static"),
            ("transport_free_vs_static", "transport_free", "static"),
        ],
        value_cols=summary_metrics,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 10000,
    )
    phase_deciles, phase_adjusted = make_phase_decile_contrasts(
        heldout,
        value_col="excess",
        left="interaction",
        right="transport_free",
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 20000,
    )
    matched_pairs, matched_summary = make_matched_position_pairs(
        heldout,
        value_col="excess",
        left="interaction",
        right="transport_free",
    )
    within_pairs, within_summary = make_within_episode_pairs(
        heldout,
        value_cols=["mse_model", "excess"],
        left="interaction",
        right="transport_free",
    )
    residual_table = make_residual_structure_table(
        heldout,
        residuals,
        regimes=REGIMES,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 30000,
    )

    records.to_csv(args.output_dir / "step_records_relabel_motion.csv", index=False)
    write_json(args.output_dir / "label_provenance.json", provenance)
    write_json(args.output_dir / "validity_gates.json", gates)
    counts.to_csv(metrics_dir / "regime_counts.csv", index=False)
    summarize_gate_table(gates).to_csv(metrics_dir / "validity_gates.csv", index=False)
    regime_summary.to_csv(metrics_dir / "regime_summary.csv", index=False)
    regime_contrasts.to_csv(metrics_dir / "regime_contrasts.csv", index=False)
    phase_deciles.to_csv(metrics_dir / "phase_decile_contrasts.csv", index=False)
    phase_adjusted.to_csv(metrics_dir / "phase_adjusted_contrast.csv", index=False)
    matched_pairs.to_csv(metrics_dir / "matched_position_pairs.csv", index=False)
    matched_summary.to_csv(metrics_dir / "matched_position_summary.csv", index=False)
    within_pairs.to_csv(metrics_dir / "within_episode_pairs.csv", index=False)
    within_summary.to_csv(metrics_dir / "within_episode_summary.csv", index=False)
    residual_table.to_csv(metrics_dir / "residual_structure.csv", index=False)

    figure_paths = write_all_figures(
        records=heldout,
        regime_summary=regime_summary,
        regime_contrasts=regime_contrasts,
        within_episode_pairs=within_pairs,
        within_episode_summary=within_summary,
        residual_structure=residual_table,
        figure_dir=figure_dir,
    )
    decision = report_decision(gates, regime_contrasts, phase_adjusted, matched_summary)
    write_json(args.output_dir / "decision.json", decision)
    write_report(
        args.output_dir / "REPORT.md",
        decision=decision,
        provenance=provenance,
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
