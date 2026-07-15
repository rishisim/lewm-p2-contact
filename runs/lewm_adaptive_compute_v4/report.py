#!/usr/bin/env python3
"""Create the V4 paper figures, machine-readable tables, and narrative report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common


COLORS = {
    "adaptive": "#1565C0",
    "baseline": "#616161",
    "exact": "#C62828",
    "secondary": "#EF6C00",
    "oracle": "#2E7D32",
}


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = common.ROOT / "figures" / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _comparison_point(item: dict[str, Any]) -> tuple[float, float, float]:
    return item["mean_benefit"], item["ci_low"], item["ci_high"]


def make_figures(primary: dict[str, Any], physical: dict[str, Any], runtime: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fixed = primary["fixed_exit_raw_mse"]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    depths = np.arange(1, 5)
    ax.plot(depths, [fixed[f"d{i}"] for i in depths], marker="o", color=COLORS["baseline"])
    ax.axhline(primary["adaptive_raw_mse"], color=COLORS["adaptive"], label="adaptive b1.25")
    ax.set(xticks=depths, xlabel="Solver depth", ylabel="Raw latent MSE", title="Frozen fixed exits and V4 adaptive policy")
    ax.legend(frameon=False)
    _save(fig, "01_fixed_exits")

    frontier = primary["frontier"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for row in frontier:
        color = COLORS["adaptive"] if row["kind"] == "adaptive" else (
            COLORS["exact"] if row["kind"] == "primary_exact_total_flop_mixture" else COLORS["baseline"]
        )
        marker = "*" if row["kind"] == "adaptive" else "o"
        axes[0].scatter(row["mean_block_calls"], row["raw_mse"], color=color, marker=marker, s=70 if marker == "*" else 30)
        axes[1].scatter(row["fully_counted_flops_per_transition"], row["raw_mse"], color=color, marker=marker, s=70 if marker == "*" else 30)
    axes[0].set(xlabel="Mean solver block calls", ylabel="Raw latent MSE", title="Block-call frontier")
    axes[1].set(xlabel="Fully counted FLOPs / transition", ylabel="Raw latent MSE", title="Total-FLOP frontier")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    _save(fig, "02_call_flop_frontiers")

    calibration = common.read_json(common.COMPRESSION / "metrics/calibration_once.json")
    names = [
        "V3 equal-call calibration",
        "V4 exact-total analytic",
        "V4 exact-total seeded",
        "V4 fixed d1",
        "V4 histogram",
        "V4 whitened exact-total",
    ]
    items = [
        calibration["vs_analytic_mixture"],
        primary["comparisons"]["raw_vs_exact_total_flop_analytic"],
        primary["comparisons"]["raw_vs_exact_total_flop_seeded"],
        primary["comparisons"]["raw_vs_fixed_d1"],
        primary["comparisons"]["raw_vs_histogram"],
        primary["comparisons"]["whitened_vs_exact_total_flop_analytic"],
    ]
    points = np.asarray([_comparison_point(item) for item in items])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(names))
    ax.errorbar(points[:, 0], y, xerr=np.vstack((points[:, 0] - points[:, 1], points[:, 2] - points[:, 0])), fmt="o", color=COLORS["adaptive"], capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(yticks=y, yticklabels=names, xlabel="Benefit (baseline MSE − adaptive MSE)", title="Episode-clustered 95% intervals")
    ax.invert_yaxis()
    _save(fig, "03_confirmation_benefit_cis")

    histogram = primary["call_histogram"]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(depths, [histogram[str(i)] for i in depths], color=COLORS["adaptive"])
    ax.set(xticks=depths, xlabel="Selected depth", ylabel="Transitions", title=f"V4 call histogram (mean={primary['realized_mean_calls']:.3f})")
    _save(fig, "04_call_histogram")

    outcome_path = common.ROOT / "data/confirmation_outcome_once.npz"
    with np.load(outcome_path, allow_pickle=False) as stored:
        losses = stored["losses"].astype(np.float64)
        calls = stored["calls"].astype(np.int64)
    selected = losses[np.arange(len(calls)), calls - 1]
    gain = losses[:, 0] - selected
    quantiles = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for depth in depths:
        values = gain[calls == depth]
        if len(values):
            ax.plot(quantiles, np.quantile(values, quantiles), label=f"d{depth} (n={len(values)})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Quantile", ylabel="Realized cumulative gain vs d1", title="Gain quantiles by selected depth")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, "05_gain_quantiles")

    selected_paths = [
        "fixed_d1",
        "fixed_d4",
        "exact_total_flop_seeded_integer_mixture",
        "adaptive_dense_all_exits",
        "adaptive_reference_sparse",
        "adaptive_optimized_sparse",
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for path in selected_paths:
        rows = sorted(
            [row for row in runtime["timings"] if row["path"] == path and "all_seconds" in row],
            key=lambda row: row["batch_size"],
        )
        ax.plot([row["batch_size"] for row in rows], [1000 * row["median_seconds"] for row in rows], marker="o", label=path.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set(xlabel="Batch size", ylabel="Median synchronized latency (ms)", title=f"MPS runtime ({runtime['deployment_status']})")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    _save(fig, "06_latency")

    regimes = physical["regimes"]
    names = [item["regime"] for item in regimes]
    calls_est = [item["calls_ci"]["estimate"] for item in regimes]
    benefit_est = [item["prediction_benefit_ci"]["estimate"] for item in regimes]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].bar(names, calls_est, color=COLORS["adaptive"])
    axes[0].set(ylabel="Mean calls", title="Allocation by physical regime")
    axes[1].bar(names, benefit_est, color=COLORS["oracle"])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(ylabel="Benefit vs exact-total analytic", title="Prediction benefit by regime")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    _save(fig, "07_physical_regimes")


def _ci(value: dict[str, Any]) -> str:
    return f"{value['mean_benefit']:.8g} [{value['ci_low']:.8g}, {value['ci_high']:.8g}]"


def write_reports(primary: dict[str, Any], physical: dict[str, Any], runtime: dict[str, Any]) -> None:
    decision = common.read_json(common.ROOT / "decision.json")
    comp = primary["comparisons"]
    report = f"""# LeWM adaptive computation V4 confirmatory study

## Outcome

The preregistered one-shot decision is **{decision['decision']}**. The frozen b1.25 policy selected {primary['realized_mean_calls']:.6f} solver calls per transition and achieved raw latent MSE {primary['adaptive_raw_mse']:.9f}. Its primary benefit versus the strongest transition-independent analytic mixture at exactly the same fully counted total FLOPs was {_ci(comp['raw_vs_exact_total_flop_analytic'])}. The conservative hash-seeded integer mixture was rounded upward by {primary['exact_total_flop_budget']['integer_baseline_minus_adaptive_flops']} aggregate FLOPs and gave benefit {_ci(comp['raw_vs_exact_total_flop_seeded'])}.

The discovery-whitened benefit versus the analytic exact-total-FLOP comparator was {_ci(comp['whitened_vs_exact_total_flop_analytic'])}. Under the frozen rule, a positive interval crossing zero is raw-only confirmation; a positive lower bound is a full pass; a negative point estimate is a failure.

Runtime status is **{runtime['deployment_status']}**. This is reported separately and does not alter the statistical/FLOP verdict. Reliable energy measurement was unavailable.

## Primary checks

- Fixed exits d1–d4 raw MSE: {', '.join(f"d{i}={primary['fixed_exit_raw_mse'][f'd{i}']:.9f}" for i in range(1,5))}.
- Exact-total analytic comparison: {_ci(comp['raw_vs_exact_total_flop_analytic'])}.
- Exact-total seeded integer comparison: {_ci(comp['raw_vs_exact_total_flop_seeded'])}.
- Fixed d1 comparison: {_ci(comp['raw_vs_fixed_d1'])}.
- Exact histogram randomization: {_ci(comp['raw_vs_histogram'])}.
- Equal-call analytic decomposition (secondary): {_ci(comp['raw_vs_equal_call_analytic_secondary'])}.
- Block-call nondominated: {primary['raw_criteria']['adaptive_block_call_nondominated']}.
- Fully counted FLOP nondominated: {primary['raw_criteria']['adaptive_fully_counted_flop_nondominated']}.
- Mechanical validity suite: {primary['validity']['passed']}.
- Oracle headroom: {primary['oracle_headroom']:.9g} raw MSE.

## Physical robustness

Physical labels were attached only after the primary verdict. The frozen motion/phase/action-controlled result is **{physical['claim']}**. The fine-bin matched contact-minus-noncontact call contrast was {physical['contact_vs_noncontact_matched']['fine']['calls']['estimate_contact_minus_noncontact']:.6g} with 95% CI [{physical['contact_vs_noncontact_matched']['fine']['calls']['ci_low']:.6g}, {physical['contact_vs_noncontact_matched']['fine']['calls']['ci_high']:.6g}]. The matched prediction-benefit contrast was {physical['contact_vs_noncontact_matched']['fine']['prediction_benefit']['estimate_contact_minus_noncontact']:.6g} with CI [{physical['contact_vs_noncontact_matched']['fine']['prediction_benefit']['ci_low']:.6g}, {physical['contact_vs_noncontact_matched']['fine']['prediction_benefit']['ci_high']:.6g}]. A null contrast limits the contact-specific claim but does not invalidate overall compute-responsive allocation.

## Scope and caveats

This study confirms or rejects adaptive computation only for the frozen released visual latent physical world model, frozen stagewise solver, frozen full-feature linear gate, and one preregistered operating point. It does not establish a first adaptive world model; LoopWM remains relevant adaptive-depth world-model work in text environments. The released LeWM checkpoint has no original pretraining episode manifest, so pretraining membership of newly generated episodes cannot be mechanically disproven. The documented new seeds, exact raw hashes, and duplicate audit establish generation-time freshness against every accessible prior episode outside the permanently untouched V3 test set.

No solver, student, normalization, whitening transform, price, feature, depth encoding, or stopping rule was trained, selected, or recalibrated from V4 outcomes. Smoke episodes were physically separate and excluded forever. All 300 confirmation episodes were evaluated once after the exclusive access receipt.
"""
    (common.ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    commands = common.read_json(common.ROOT / "config.json")["exact_commands"]
    readme = """# V4 confirmatory artifact

This directory contains the preregistered one-shot LeWM V4 confirmation. Prior experiment trees are immutable; all new files live here. `PREREGISTRATION.md` and `decision_rule.json` define the frozen analysis, `decision.json` is the mechanical verdict, `REPORT.md` is the narrative readout, and `artifact_manifest.json` hashes the complete deliverable.

## Exact execution order

""" + "\n".join(f"{index}. `{command}`" for index, command in enumerate(commands, 1)) + "\n"
    (common.ROOT / "README.md").write_text(readme, encoding="utf-8")


def run() -> dict[str, Any]:
    primary = common.read_json(common.ROOT / "metrics/primary_confirmation.json")
    physical = common.read_json(common.ROOT / "metrics/physical_regimes.json")
    latency = common.read_json(common.ROOT / "metrics/runtime.json")
    make_figures(primary, physical, latency)
    write_reports(primary, physical, latency)
    return {
        "report": str(common.ROOT / "REPORT.md"),
        "figures": len(list((common.ROOT / "figures").glob("*.png"))),
        "decision": common.read_json(common.ROOT / "decision.json")["decision"],
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
