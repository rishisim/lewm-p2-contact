#!/usr/bin/env python3
"""Reconcile terminal artifacts and generate the three manuscript figures.

The script is deliberately read-only with respect to experiment and synthesis
artifacts.  Its only outputs are the three PDF files in manuscript/figures/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


MANUSCRIPT = Path(__file__).resolve().parent
ROOT = MANUSCRIPT.parent
FIGURES = MANUSCRIPT / "figures"

INDEX_PATH = ROOT / "runs/lewm_paper_evidence_synthesis/EVIDENCE_INDEX.json"
CUBE_DECISION = ROOT / "runs/lewm_v5_readiness_program/v5_package_versions/v004/decision.json"
CUBE_BOOT = ROOT / "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/bootstrap_replicates.npz"
CUBE_EPISODES = ROOT / "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/v5_confirmation_episode_metrics.npz"
CUBE_RANK = ROOT / "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/stagewise_ranking.json"
SHIFT_DECISION = ROOT / "runs/lewm_v5_generalization/attempts/v005/decision.json"
SHIFT_BOOT = ROOT / "runs/lewm_v5_generalization/attempts/v005/metrics/bootstrap_replicates.npz"
PILOT_DECISION = ROOT / "runs/lewm_pusht_replication_pilot/PILOT_DECISION.json"
PILOT_ARRAYS = ROOT / "runs/lewm_pusht_replication_pilot/EVALUATION_ARRAYS.npz"
BINARY_DECISION = ROOT / "runs/lewm_pusht_binary_confirmation/DECISION.json"
BINARY_ARRAYS = ROOT / "runs/lewm_pusht_binary_confirmation/EVALUATION_ARRAYS.npz"
BRIDGE_RESULTS = ROOT / "runs/lewm_frozen_gate_planning_bridge/RESULTS.json"
BRIDGE_B = ROOT / "runs/lewm_frozen_gate_planning_bridge/phase_b_metrics.npz"
DECOMP_RESULTS = ROOT / "runs/lewm_planning_bridge_decomposition/RESULTS.json"
DECOMP_ARRAYS = ROOT / "runs/lewm_planning_bridge_decomposition/decomposition_metrics.npz"

BLUE = "#2864A6"
LIGHT_BLUE = "#79A9D1"
ORANGE = "#D17A22"
RED = "#B4473E"
GREEN = "#39845C"
GRAY = "#6F7782"
LIGHT_GRAY = "#D8DCE2"
BLACK = "#20242A"
DEPTH_COLORS = ["#3A78B8", "#7CB6D8", "#E4A05B", "#B65A4A"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, *, atol=1e-14, rtol=1e-11):
    if not np.isclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{actual!r} != {expected!r}")


def verify_sources() -> None:
    """Hash-check all inputs and independently reproduce plotted summaries."""
    index = load_json(INDEX_PATH)
    catalog = index["source_catalog"]
    used_ids = [
        "cube_decision", "cube_episode_metrics", "cube_bootstrap_replicates",
        "cube_stagewise_ranking", "shift_decision", "shift_bootstrap_replicates",
        "planning_bridge_results", "planning_phase_b_metrics",
        "planning_decomposition_results", "planning_decomposition_metrics",
        "pusht_pilot_decision", "pusht_pilot_arrays",
        "pusht_binary_decision", "pusht_binary_arrays",
    ]
    for source_id in used_ids:
        record = catalog[source_id]
        path = ROOT / record["path"]
        actual = sha256(path)
        if actual != record["sha256"]:
            raise AssertionError(f"hash mismatch for {source_id}: {actual}")

    cube = load_json(CUBE_DECISION)
    ce = np.load(CUBE_EPISODES, allow_pickle=False)
    cb = np.load(CUBE_BOOT, allow_pickle=False)
    assert ce["episode_id"].size == 1600
    assert sum(cube["compute"]["call_histogram"]) == 60800
    assert cube["compute"]["call_histogram"] == [50973, 7267, 1773, 787]
    close(ce["raw_vs_analytic"].mean(), 6.769003361969305e-06)
    close(ce["native_whitened_vs_analytic"].mean(), 0.0006086220971507569)
    close(np.quantile(cb["raw_vs_analytic"], 0.025), 5.423955197986978e-06)
    close(np.quantile(cb["native_whitened_vs_analytic"], 0.025), 0.0005135570380185851)
    assert cube["compute"]["adaptive_total_flops"] == (
        60800 * 70_529_190 + 60800 * 669_184
        + (73974 - 60800) * 264_960 + 73187 * 7_985
    ) == 4_332_936_120_435

    shifts = load_json(SHIFT_DECISION)
    sb = np.load(SHIFT_BOOT, allow_pickle=False)
    regime_specs = {
        "markov_oracle": ([-2.5137786633814877e-06, 0.00017736180662955358], [103059, 6748, 3507, 686]),
        "plan_action_noise_0p2": ([6.690627724886398e-06, 0.0006314822713329762], [96236, 12960, 3261, 1543]),
        "plan_random_action_0p1": ([9.599565687949847e-08, 0.00042630600368814966], [96637, 12788, 3166, 1409]),
    }
    for regime, (effects, hist) in regime_specs.items():
        assert shifts["compute"][regime]["call_histogram"] == hist
        for metric, expected in zip(("raw_vs_analytic", "fixed_whitened_vs_analytic"), effects):
            rec = shifts["individual_intervals"][regime][metric]
            reps = sb[f"{regime}__{metric}"]
            close(rec["estimate"], expected)
            close(np.quantile(reps, 0.025), rec["lower"])
            close(np.quantile(reps, 0.975), rec["upper"])
            close(
                np.quantile(reps, 0.05 / 6),
                shifts["simultaneous_co_primary"][regime][metric]["lower"],
            )
        assert sum(hist) == 114000

    pilot = load_json(PILOT_DECISION)
    pa = np.load(PILOT_ARRAYS, allow_pickle=False)
    assert pa["episode_raw_vs_primary_analytic"].size == 80
    close(pa["episode_raw_vs_primary_analytic"].mean(), 0.003909796985389183)
    close(pa["episode_fit_whitened_vs_primary_analytic"].mean(), -0.00030990012922212844)
    assert np.bincount(pa["calls"], minlength=5)[1:].tolist() == [1089, 42, 46, 263]
    assert pilot["compute"]["adaptive_total_counted_flops"] == (
        1440 * 70_383_192 + 1440 * 60 + 2363 * 650_432 + 2100 * 7_715
    ) == 102_905_055_196

    binary = load_json(BINARY_DECISION)
    ba = np.load(BINARY_ARRAYS, allow_pickle=False)
    assert ba["episode_ordinal"].size == 240
    close(ba["episode_raw_vs_exact_compute_analytic"].mean(), 0.002733782040519924)
    close(ba["episode_fit_whitened_vs_exact_compute_analytic"].mean(), 0.0015692368266779146)
    assert np.bincount(ba["calls"], minlength=3)[1:].tolist() == [3265, 1055]
    assert binary["compute"]["adaptive_total_counted_flops"] == (
        4320 * 70_383_192 + 4320 * 60 + 5375 * 650_432 + 4320 * 7_715
    ) == 307_585_049_440

    bridge = load_json(BRIDGE_RESULTS)
    phase_b = np.load(BRIDGE_B, allow_pickle=False)
    assert phase_b["case_start_index"].size == 3400
    assert bridge["phase_b"]["episode_count"] == 100
    assert phase_b["condition"].tolist()[:2] == ["adaptive", "matched"]
    close((phase_b["raw_mse"][0, 4] - phase_b["raw_mse"][1, 4]).mean(), -6.597279931162478e-05)
    close((phase_b["whitened_mse"][0, 4] - phase_b["whitened_mse"][1, 4]).mean(), -0.0011870892398969934)

    decomp = load_json(DECOMP_RESULTS)
    da = np.load(DECOMP_ARRAYS, allow_pickle=False)
    assert da["adaptive_minus_matched_start_effect_raw"].size == 20
    close(da["adaptive_minus_matched_start_effect_raw"].mean(), -2.7352993921345853e-06)
    close(da["adaptive_minus_matched_start_effect_whitened"].mean(), -0.00014018127730549418)
    assert int(da["informative_start"].sum()) == 0
    assert int((da["physical_outcome_range"] == 0).sum()) == 4
    assert int((da["physical_outcome_range"] <= 1e-12).sum()) == 10
    assert decomp["candidate_set_informativeness"]["informative_start_count"] == 0


def configure_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=10,
            va="bottom", ha="left")


def interval_row(ax, y, estimate, low, high, *, color=BLUE, marker="o", filled=True,
                 simultaneous=None, size=38, zorder=3):
    ax.plot([low, high], [y, y], color=GRAY, linewidth=1.5, solid_capstyle="round", zorder=1)
    ax.plot([low, low], [y - 0.055, y + 0.055], color=GRAY, linewidth=0.8, zorder=1)
    ax.plot([high, high], [y - 0.055, y + 0.055], color=GRAY, linewidth=0.8, zorder=1)
    ax.scatter([estimate], [y], s=size, marker=marker, facecolor=color if filled else "white",
               edgecolor=color, linewidth=1.2, zorder=zorder)
    if simultaneous is not None:
        ax.plot([simultaneous, simultaneous], [y - 0.11, y + 0.11], color=BLACK,
                linewidth=1.5, zorder=4)


def figure_one() -> None:
    cube = load_json(CUBE_DECISION)
    pilot = load_json(PILOT_DECISION)
    binary = load_json(BINARY_DECISION)
    rank = load_json(CUBE_RANK)

    fig = plt.figure(figsize=(7.45, 5.65), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.0], width_ratios=[0.82, 1.35])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    # A: locked Cube co-primary endpoints, normalized by comparator MSE.
    cube_specs = [
        ("Raw MSE", "raw_vs_analytic", "raw_analytic_mixture"),
        ("Native-whitened MSE", "native_whitened_vs_analytic", "native_whitened_analytic_mixture"),
    ]
    for y, (_, endpoint, comparator_key) in enumerate(cube_specs[::-1]):
        rec = cube["criteria"][endpoint]
        denom = cube["compute"][comparator_key]["mean_loss"]
        values = [100 * rec[key] / denom for key in ("estimate", "lower", "upper")]
        interval_row(ax_a, y, *values, simultaneous=values[1], color=BLUE)
    ax_a.axvline(0, color=BLACK, linewidth=0.8)
    ax_a.set_yticks([0, 1], [x[0] for x in cube_specs[::-1]])
    ax_a.set_xlabel("Benefit (% of comparator MSE)\nright favors adaptive")
    ax_a.set_title("Frozen Cube confirmation")
    ax_a.text(0.02, 0.02, "1,600 episodes · 60,800 transitions\n4,332,936,120,435 counted FLOPs · passed",
              transform=ax_a.transAxes, fontsize=6.5, va="bottom", color=GRAY)
    ax_a.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55)
    panel_label(ax_a, "A")

    # B: preserve the PushT discovery-to-confirmation chronology.
    rows = [
        ("Pilot · raw", pilot["primary"]["raw"], pilot["compute"]["raw_primary_analytic_mixture"]["mean_loss"], ORANGE, "o", True, None),
        ("Pilot · fit-whitened", pilot["primary"]["fit_whitened"], pilot["compute"]["fit_whitened_primary_analytic_mixture"]["mean_loss"], ORANGE, "o", True, None),
        ("Binary confirmation · raw", binary["co_primary"]["raw"], binary["compute"]["raw_strongest_pairwise_mixture"]["mean_episode_loss"], BLUE, "s", True, "lower"),
        ("Binary confirmation · fit-whitened", binary["co_primary"]["fit_whitened"], binary["compute"]["fit_whitened_strongest_pairwise_mixture"]["mean_episode_loss"], BLUE, "s", True, "lower"),
    ]
    y_positions = [3.2, 2.2, 0.8, -0.2]
    for y, (label, rec, denom, color, marker, filled, _) in zip(y_positions, rows):
        est = rec.get("mean_benefit", rec.get("mean"))
        low = rec.get("ci_low", rec.get("exploratory_95_ci_low"))
        high = rec.get("ci_high", rec.get("exploratory_95_ci_high"))
        vals = 100 * np.array([est, low, high]) / denom
        sim = vals[1] if "Binary" in label else None
        interval_row(ax_b, y, *vals, simultaneous=sim, color=color, marker=marker, filled=filled)
    ax_b.axvline(0, color=BLACK, linewidth=0.8)
    ax_b.axhline(1.5, color=LIGHT_GRAY, linewidth=0.9)
    ax_b.text(0.01, 0.53, "pilot outcome → binary rule derived → fresh cohort",
              transform=ax_b.transAxes, fontsize=6.5, color=GRAY, va="bottom")
    ax_b.set_yticks(y_positions, [row[0] for row in rows])
    ax_b.set_xlabel("Benefit (% of comparator MSE)\nright favors adaptive")
    ax_b.set_title("PushT: negative pilot before fresh confirmation")
    ax_b.text(0.99, 0.02, "Pilot: 80 episodes, exploratory 95% intervals\nBinary: 240 episodes, 307,585,049,440 FLOPs",
              transform=ax_b.transAxes, fontsize=6.5, va="bottom", ha="right", color=GRAY)
    ax_b.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55)
    panel_label(ax_b, "B")

    # C: realized allocation, with routing-rank summaries.
    allocations = np.array([
        cube["compute"]["call_histogram"],
        binary["compute"]["call_histogram_depth_1_to_2"] + [0, 0],
    ], dtype=float)
    allocations /= allocations.sum(axis=1, keepdims=True)
    left = np.zeros(2)
    for depth in range(4):
        ax_c.barh([1, 0], allocations[:, depth] * 100, left=left * 100, height=0.55,
                  color=DEPTH_COLORS[depth], edgecolor="white", linewidth=0.6,
                  label=f"depth {depth + 1}")
        left += allocations[:, depth]
    cube_rhos = ", ".join(f"{s['spearman_score_gain_rho']:.2f}" for s in rank["stages"])
    ax_c.text(100.7, 1, rf"stage $\rho$: {cube_rhos}", va="center", fontsize=7)
    ax_c.text(100.7, 0, rf"stage-1 $\rho$: {binary['stage1_score_gain_rank']['combined_score_gain_spearman']:.2f}",
              va="center", fontsize=7)
    ax_c.set_xlim(0, 132)
    ax_c.set_yticks([1, 0], ["Cube confirmation", "PushT binary confirmation"])
    ax_c.set_xlabel("Realized transition allocation (%)")
    ax_c.set_title("Allocation and routing summaries", loc="left")
    ax_c.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.48))
    panel_label(ax_c, "C")

    fig.savefig(FIGURES / "figure1_confirmed_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_two() -> None:
    decision = load_json(SHIFT_DECISION)
    boots = np.load(SHIFT_BOOT, allow_pickle=False)
    regimes = ["markov_oracle", "plan_action_noise_0p2", "plan_random_action_0p1"]
    labels = ["Markov policy", "PlanOracle noise 0.2", "Random action 0.1"]

    fig = plt.figure(figsize=(7.45, 5.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.65, 1.0])
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_white = fig.add_subplot(grid[0, 1])
    ax_alloc = fig.add_subplot(grid[1, :])

    for ax, metric, title, comparator_key in [
        (ax_raw, "raw_vs_analytic", "Raw MSE", "raw_analytic_mixture"),
        (ax_white, "fixed_whitened_vs_analytic", "Fixed-whitened MSE", "fixed_whitened_analytic_mixture"),
    ]:
        for y, regime in enumerate(regimes[::-1]):
            rec = decision["individual_intervals"][regime][metric]
            sim = decision["simultaneous_co_primary"][regime][metric]["lower"]
            denom = decision["compute"][regime][comparator_key]["mean_loss"]
            vals = 100 * np.array([rec["estimate"], rec["lower"], rec["upper"], sim]) / denom
            supported = sim > 0
            interval_row(ax, y, vals[0], vals[1], vals[2], simultaneous=vals[3],
                         color=GREEN if supported else RED, filled=supported)
        ax.axvline(0, color=BLACK, linewidth=0.8)
        ax.set_yticks(range(3), labels[::-1])
        ax.set_xlabel("Benefit (% of comparator MSE)\nright favors adaptive")
        ax.set_title(title)
        ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55)
    panel_label(ax_raw, "A")
    panel_label(ax_white, "B")
    ax_raw.text(0.01, 0.02, "line: individual 95% interval\nblack cap: simultaneous lower bound",
                transform=ax_raw.transAxes, fontsize=6.5, color=GRAY, va="bottom")
    ax_white.text(0.99, 0.02, "filled: simultaneous lower bound > 0\n3,000 episodes per regime",
                  transform=ax_white.transAxes, fontsize=6.5, color=GRAY, va="bottom", ha="right")

    all_labels = ["Cube native", *labels]
    histograms = [
        [50973, 7267, 1773, 787],
        decision["compute"]["markov_oracle"]["call_histogram"],
        decision["compute"]["plan_action_noise_0p2"]["call_histogram"],
        decision["compute"]["plan_random_action_0p1"]["call_histogram"],
    ]
    allocations = np.array(histograms, dtype=float)
    allocations /= allocations.sum(axis=1, keepdims=True)
    left = np.zeros(4)
    positions = np.arange(4)[::-1]
    for depth in range(4):
        ax_alloc.barh(positions, allocations[:, depth] * 100, left=left * 100, height=0.58,
                      color=DEPTH_COLORS[depth], edgecolor="white", linewidth=0.6,
                      label=f"depth {depth + 1}")
        left += allocations[:, depth]
    ax_alloc.set_yticks(positions, all_labels)
    ax_alloc.set_xlim(0, 100)
    ax_alloc.set_xlabel("Realized transition allocation (%)")
    ax_alloc.set_title("Frozen-gate allocation under distribution shifts", loc="left")
    ax_alloc.legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.48))
    panel_label(ax_alloc, "C")

    fig.savefig(FIGURES / "figure2_cube_shift_map.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_three() -> None:
    phase_b = np.load(BRIDGE_B, allow_pickle=False)
    decomp = np.load(DECOMP_ARRAYS, allow_pickle=False)
    decomp_results = load_json(DECOMP_RESULTS)
    horizons = np.arange(1, 6)
    raw_h = np.array([(phase_b["raw_mse"][0, h] - phase_b["raw_mse"][1, h]).mean() for h in range(5)])
    white_h = np.array([(phase_b["whitened_mse"][0, h] - phase_b["whitened_mse"][1, h]).mean() for h in range(5)])

    fig = plt.figure(figsize=(7.45, 6.25), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25])
    top = outer[0].subgridspec(1, 4, width_ratios=[1.0, 1.0, 0.9, 0.9])
    ax_ar = fig.add_subplot(top[0, 0])
    ax_aw = fig.add_subplot(top[0, 1])
    ax_br = fig.add_subplot(top[0, 2])
    ax_bw = fig.add_subplot(top[0, 3])
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.55, 0.9])
    ax_range = fig.add_subplot(bottom[0, 0])
    ax_groups = fig.add_subplot(bottom[0, 1])

    for ax, values, title, scale in [
        (ax_ar, raw_h, "Raw", 1e5),
        (ax_aw, white_h, "Whitened", 1e3),
    ]:
        ax.plot(horizons, values * scale, color=BLUE, marker="o", linewidth=1.5, markersize=4)
        ax.axhline(0, color=BLACK, linewidth=0.8)
        ax.set_xticks(horizons)
        ax.set_xlabel("Horizon")
        ax.set_title(title)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55)
    ax_ar.set_ylabel(r"Adaptive $-$ matched ($\times 10^{-5}$)")
    ax_aw.set_ylabel(r"Adaptive $-$ matched ($\times 10^{-3}$)")
    ax_ar.text(0.02, 0.02, "negative favors adaptive\nexploratory; no intervals",
               transform=ax_ar.transAxes, fontsize=6.5, color=GRAY, va="bottom")
    panel_label(ax_ar, "A")

    candidate_specs = [
        (ax_br, "Raw", "adaptive_minus_matched_start_effect_raw", 1e5,
         decomp_results["candidate_rollout_fidelity"]["adaptive_minus_matched"]["raw"]),
        (ax_bw, "Whitened", "adaptive_minus_matched_start_effect_whitened", 1e3,
         decomp_results["candidate_rollout_fidelity"]["adaptive_minus_matched"]["whitened"]),
    ]
    for ax, title, key, scale, rec in candidate_specs:
        mean = rec["mean"] * scale
        low, high = np.array(rec["mean_bootstrap_95_interval"]) * scale
        interval_row(ax, 0, mean, low, high, color=ORANGE, marker="s", filled=True)
        ax.axvline(0, color=BLACK, linewidth=0.8)
        ax.set_yticks([0], ["20 starts"])
        ax.set_xlabel(r"Adaptive $-$ matched")
        ax.set_title(title + (r" ($\times 10^{-5}$)" if scale == 1e5 else r" ($\times 10^{-3}$)"))
        ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55)
    ax_br.text(0.02, 0.02, "paired cluster-bootstrap\n95% interval",
               transform=ax_br.transAxes, fontsize=6.2, color=GRAY, va="bottom")
    panel_label(ax_br, "B")

    ranges_mm = decomp["physical_outcome_range"] * 1000.0
    exactly_constant = ranges_mm == 0
    floor = 1e-14
    plotted = np.maximum(ranges_mm, floor)
    x = np.arange(1, 21)
    ax_range.scatter(x[~exactly_constant], plotted[~exactly_constant], color=GRAY, s=25,
                     marker="o", label="nonzero range")
    ax_range.scatter(x[exactly_constant], np.full(exactly_constant.sum(), floor), color=RED,
                     s=34, marker="x", linewidth=1.4, label="exactly constant")
    ax_range.axhline(0.1, color=ORANGE, linestyle="--", linewidth=1.0,
                     label="0.1 mm tolerance")
    ax_range.set_yscale("log")
    ax_range.set_xticks([1, 5, 10, 15, 20])
    ax_range.set_xlabel("Independent planning start")
    ax_range.set_ylabel("Physical outcome range (mm; log scale)")
    ax_range.set_ylim(floor / 4, 100)
    ax_range.grid(axis="y", which="major", color=LIGHT_GRAY, linewidth=0.55)
    ax_range.legend(frameon=False, loc="upper left")
    ax_range.text(0.99, 0.04, "10/20 ≤ 10⁻⁹ mm\n4/20 exactly constant",
                  transform=ax_range.transAxes, ha="right", va="bottom", fontsize=6.7, color=GRAY)
    panel_label(ax_range, "C")

    groups = decomp["meaningfully_distinct_outcomes"]
    ax_groups.barh(x, groups, color=np.where(groups >= 5, GREEN, LIGHT_BLUE), height=0.72)
    ax_groups.axvline(5, color=ORANGE, linestyle="--", linewidth=1.0)
    ax_groups.set_ylim(0.3, 20.7)
    ax_groups.invert_yaxis()
    ax_groups.set_yticks([1, 5, 10, 15, 20])
    ax_groups.set_xlabel("Tolerance-distinct groups")
    ax_groups.set_ylabel("Start")
    ax_groups.set_title("Candidate-set resolution")
    ax_groups.text(0.98, 0.04, "0/20 informative starts\nmean rank ρ = 0.024\nmean top-5 overlap = 0.633",
                   transform=ax_groups.transAxes, ha="right", va="bottom", fontsize=6.7, color=GRAY)
    ax_groups.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55)

    fig.suptitle("Exploratory composition did not supply planning evidence", fontsize=10, fontweight="bold")
    fig.savefig(FIGURES / "figure3_multistep_planning.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    verify_sources()
    configure_style()
    figure_one()
    figure_two()
    figure_three()
    print("verified indexed inputs and wrote exactly three manuscript figures")


if __name__ == "__main__":
    main()
