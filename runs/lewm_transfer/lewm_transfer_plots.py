"""Plot helpers for the LeWM latent-transfer diagnostic pilot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIME_COLORS = {
    "interaction": "#2563eb",
    "transport_free": "#0891b2",
    "static": "#a16207",
    "support": "#16a34a",
    "free": "#6b7280",
}
REGIME_ORDER = ["interaction", "transport_free", "static", "support", "free"]


def _ci_bounds(row: pd.Series, metric: str) -> tuple[float, float]:
    value = float(row[metric])
    low = float(row.get(f"{metric}_ci_low", value))
    high = float(row.get(f"{metric}_ci_high", value))
    return max(0.0, value - low), max(0.0, high - value)


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not written or is empty: {path}")


def plot_model_vs_persistence(summary: pd.DataFrame, path: Path) -> None:
    regimes = [r for r in REGIME_ORDER if r in set(summary["regime"])]
    metrics = ["mse_model", "mse_persistence"]
    labels = ["model", "persistence"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(regimes), dtype=float)
    width = 0.34
    offsets = [-width / 2, width / 2]
    colors = ["#2563eb", "#94a3b8"]
    for metric, label, offset, color in zip(metrics, labels, offsets, colors):
        values = []
        yerr_low = []
        yerr_high = []
        for regime in regimes:
            row = summary.loc[(summary["regime"] == regime) & (summary["metric"] == metric)].iloc[0]
            values.append(float(row["trimmed_mean"]))
            low, high = _ci_bounds(row, "trimmed_mean")
            yerr_low.append(low)
            yerr_high.append(high)
        ax.bar(x + offset, values, width=width, color=color, alpha=0.92, label=label)
        ax.errorbar(x + offset, values, yerr=[yerr_low, yerr_high], fmt="none", ecolor="#111827", capsize=3)
    ax.set_xticks(x, [r.replace("_", " ") for r in regimes])
    ax.set_ylabel("Trimmed mean latent MSE")
    ax.set_title("Model vs persistence by regime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _finish(fig, path)


def plot_excess_bars(summary: pd.DataFrame, path: Path) -> None:
    rows = summary.loc[summary["metric"] == "excess"].copy()
    rows = rows.sort_values("trimmed_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = [REGIME_COLORS.get(r, "#64748b") for r in rows["regime"]]
    values = rows["trimmed_mean"].to_numpy(dtype=float)
    lows = []
    highs = []
    for _, row in rows.iterrows():
        low, high = _ci_bounds(row, "trimmed_mean")
        lows.append(low)
        highs.append(high)
    ax.bar(rows["regime"].str.replace("_", " "), values, color=colors, alpha=0.92)
    ax.errorbar(np.arange(len(rows)), values, yerr=[lows, highs], fmt="none", ecolor="#111827", capsize=3)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_ylabel("Trimmed mean excess MSE")
    ax.set_title("Persistence-adjusted excess error")
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(values[np.isfinite(values)])
    if len(arr) == 0:
        return arr, arr
    y = np.arange(1, len(arr) + 1) / len(arr)
    return arr, y


def plot_interaction_free_distribution(records: pd.DataFrame, contrasts: pd.DataFrame, path: Path) -> None:
    interaction = records.loc[records["regime"] == "interaction", "excess"].to_numpy(dtype=float)
    right_regime = "transport_free" if "transport_free" in set(records["regime"]) else "free"
    free = records.loc[records["regime"] == right_regime, "excess"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    ax = axes[0]
    if len(interaction):
        x, y = _ecdf(interaction)
        ax.plot(x, y, color=REGIME_COLORS["interaction"], linewidth=2, label=f"interaction n={len(interaction)}")
    if len(free):
        x, y = _ecdf(free)
        ax.plot(x, y, color=REGIME_COLORS[right_regime], linewidth=2, label=f"{right_regime} n={len(free)}")
    ax.set_xlabel("Excess MSE")
    ax.set_ylabel("ECDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    groups = []
    labels = []
    colors = []
    if len(interaction):
        groups.append(interaction)
        labels.append("interaction")
        colors.append(REGIME_COLORS["interaction"])
    if len(free):
        groups.append(free)
        labels.append(right_regime)
        colors.append(REGIME_COLORS[right_regime])
    if groups:
        parts = ax.violinplot(groups, showmeans=False, showmedians=True, showextrema=True)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.25)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel("Excess MSE")
    ax.grid(axis="y", alpha=0.25)
    p_row = contrasts.loc[
        (contrasts["contrast"].isin(["interaction_vs_transport_free", "interaction_vs_free"]))
        & (contrasts["metric"] == "excess")
    ]
    p_text = "P(superiority)=n/a"
    if len(p_row):
        p_value = float(p_row.iloc[0]["p_superiority"])
        if np.isfinite(p_value):
            p_text = f"P(superiority)={p_value:.3f}"
    ax.text(0.02, 0.98, p_text, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    fig.suptitle(f"Interaction vs {right_regime} excess-error distribution")
    _finish(fig, path)


def plot_error_vs_position(records: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for regime in REGIME_ORDER:
        group = records.loc[records["regime"] == regime]
        if group.empty:
            continue
        curves = (
            group.groupby("phase_decile", sort=True)["excess"]
            .agg(["mean", "std", "count"])
            .rename(columns={"count": "n"})
            .reset_index()
        )
        x = (curves["phase_decile"].to_numpy(dtype=float) + 0.5) / 10.0
        y = curves["mean"].to_numpy(dtype=float)
        sem = curves["std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(curves["n"].clip(lower=1))
        color = REGIME_COLORS.get(regime, "#64748b")
        ax.plot(x, y, marker="o", linewidth=2, color=color, label=regime)
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel("Normalized trajectory position")
    ax.set_ylabel("Mean excess MSE")
    ax.set_title("Excess error vs trajectory phase")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finish(fig, path)


def plot_within_episode_scatter(pair_rows: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    if "metric" in pair_rows.columns:
        rows = pair_rows.loc[pair_rows["metric"] == "mse_model"].copy()
    else:
        rows = pd.DataFrame()
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    if not rows.empty:
        x = rows["right_median"].to_numpy(dtype=float)
        y = rows["left_median"].to_numpy(dtype=float)
        positive = (x > 0.0) & (y > 0.0)
        ax.scatter(x[positive], y[positive], s=36, color="#2563eb", alpha=0.8)
        lo = float(min(x[positive].min(), y[positive].min())) if positive.any() else 1e-6
        hi = float(max(x[positive].max(), y[positive].max())) if positive.any() else 1.0
        ax.plot([lo, hi], [lo, hi], color="#111827", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Free per-episode median model MSE")
    ax.set_ylabel("Interaction per-episode median model MSE")
    ax.set_title("Within-episode paired medians")
    ax.grid(alpha=0.25, which="both")
    summary_row = summary.loc[summary["metric"] == "mse_model"]
    text = "eligible episodes=0"
    if len(summary_row):
        row = summary_row.iloc[0]
        frac = row.get("fraction_episode_positive", np.nan)
        p_value = row.get("wilcoxon_p_greater", np.nan)
        text = f"above={100.0 * frac:.1f}%\nWilcoxon p={p_value:.3g}" if np.isfinite(frac) else "eligible episodes=0"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    _finish(fig, path)


def plot_residual_structure(residual_table: pd.DataFrame, path: Path) -> None:
    regimes = [r for r in REGIME_ORDER if r in set(residual_table["regime"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    ax = axes[0]
    values = []
    lows = []
    highs = []
    for regime in regimes:
        row = residual_table.loc[residual_table["regime"] == regime].iloc[0]
        values.append(float(row["directional_consistency"]))
        low, high = _ci_bounds(row, "directional_consistency")
        lows.append(low)
        highs.append(high)
    ax.bar(regimes, values, color=[REGIME_COLORS.get(r, "#64748b") for r in regimes], alpha=0.92)
    ax.errorbar(np.arange(len(regimes)), values, yerr=[lows, highs], fmt="none", ecolor="#111827", capsize=3)
    ax.set_ylabel("||mean residual|| / mean ||residual||")
    ax.set_title("Directional consistency")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    values = []
    lows = []
    highs = []
    for regime in regimes:
        row = residual_table.loc[residual_table["regime"] == regime].iloc[0]
        values.append(float(row["effective_rank"]))
        low, high = _ci_bounds(row, "effective_rank")
        lows.append(low)
        highs.append(high)
    ax.bar(regimes, values, color=[REGIME_COLORS.get(r, "#64748b") for r in regimes], alpha=0.92)
    ax.errorbar(np.arange(len(regimes)), values, yerr=[lows, highs], fmt="none", ecolor="#111827", capsize=3)
    ax.set_ylabel("Effective rank")
    ax.set_title("Residual spectrum")
    ax.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def write_all_figures(
    *,
    records: pd.DataFrame,
    regime_summary: pd.DataFrame,
    regime_contrasts: pd.DataFrame,
    within_episode_pairs: pd.DataFrame,
    within_episode_summary: pd.DataFrame,
    residual_structure: pd.DataFrame,
    figure_dir: Path,
) -> dict[str, Path]:
    outputs = {
        "model_vs_persistence": figure_dir / "01_model_vs_persistence_by_regime.png",
        "excess_bars": figure_dir / "02_excess_error_by_regime.png",
        "interaction_free_distribution": figure_dir / "03_interaction_free_excess_distribution.png",
        "error_vs_position": figure_dir / "04_excess_error_vs_position.png",
        "within_episode_scatter": figure_dir / "05_within_episode_mse_scatter.png",
        "residual_structure": figure_dir / "06_residual_structure_by_regime.png",
    }
    plot_model_vs_persistence(regime_summary, outputs["model_vs_persistence"])
    plot_excess_bars(regime_summary, outputs["excess_bars"])
    plot_interaction_free_distribution(records, regime_contrasts, outputs["interaction_free_distribution"])
    plot_error_vs_position(records, outputs["error_vs_position"])
    plot_within_episode_scatter(within_episode_pairs, within_episode_summary, outputs["within_episode_scatter"])
    plot_residual_structure(residual_structure, outputs["residual_structure"])
    return outputs
