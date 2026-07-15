"""Noninteractive figures for the LeWM adaptive-compute pilot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "depth0": "#4b5563",
    "uniform": "#2563eb",
    "oracle": "#dc2626",
    "adaptive": "#16a34a",
    "random": "#d97706",
    "permuted": "#7c3aed",
}


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not written: {path}")
    return path


def plot_depth_curve(depth_rows: list[dict[str, object]], path: Path) -> Path:
    depths = np.asarray([int(row["depth"]) for row in depth_rows])
    raw = np.asarray([float(row["raw_mean_loss"]) for row in depth_rows])
    white = np.asarray([float(row["whitened_mean_loss"]) for row in depth_rows])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(depths, raw, marker="o", color=COLORS["uniform"], linewidth=2)
    axes[0].set_title("Raw latent error versus depth")
    axes[0].set_ylabel("Mean per-dimension MSE")
    axes[1].plot(depths, white, marker="o", color=COLORS["permuted"], linewidth=2)
    axes[1].set_title("Calibration-whitened error")
    axes[1].set_ylabel("Mean whitened MSE")
    for axis in axes:
        axis.set_xlabel("Shared refinement-block calls")
        axis.set_xticks(depths)
        axis.grid(alpha=0.25)
    return _finish(fig, path)


def plot_gain_distribution(
    raw_losses: np.ndarray,
    depths: tuple[int, ...],
    path: Path,
) -> Path:
    losses = np.asarray(raw_losses, dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.3, 4.0))
    baseline = losses[:, 0]
    for column, depth in enumerate(depths[1:], start=1):
        gain = baseline - losses[:, column]
        lo, hi = np.quantile(gain, [0.01, 0.99])
        clipped = np.clip(gain, lo, hi)
        axis.hist(clipped, bins=55, density=True, histtype="step", linewidth=1.8, label=f"depth {depth}")
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_title("Same-transition compute benefit (1st–99th percentile clipped)")
    axis.set_xlabel(r"$G_i(k) =$ depth-0 MSE $-$ depth-k MSE")
    axis.set_ylabel("Density")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    return _finish(fig, path)


def plot_budget_curve(budget_rows: list[dict[str, object]], path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(7.3, 4.2))
    strategies = sorted({str(row["strategy"]) for row in budget_rows})
    for strategy in strategies:
        rows = sorted(
            [row for row in budget_rows if str(row["strategy"]) == strategy],
            key=lambda row: float(row["mean_calls"]),
        )
        axis.plot(
            [float(row["mean_calls"]) for row in rows],
            [float(row["raw_mean_loss"]) for row in rows],
            marker="o",
            label=strategy,
            color=COLORS.get(strategy),
            linewidth=2 if strategy in {"uniform", "oracle", "adaptive"} else 1.3,
        )
    axis.set_title("Matched-call allocation curves")
    axis.set_xlabel("Mean shared-block calls per transition")
    axis.set_ylabel("Mean raw latent MSE (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    return _finish(fig, path)


def plot_allocation_by_regime(
    allocations: np.ndarray,
    regimes: np.ndarray,
    depths: tuple[int, ...],
    path: Path,
) -> Path:
    selected = np.asarray(allocations, dtype=np.int64)
    labels = np.asarray(regimes).astype(str)
    regime_order = [name for name in ("impact", "interaction", "transport_free", "static") if np.any(labels == name)]
    if not regime_order:
        regime_order = sorted(np.unique(labels).tolist())
    fractions = np.zeros((len(regime_order), len(depths)), dtype=np.float64)
    for row, regime in enumerate(regime_order):
        mask = labels == regime
        for column, depth in enumerate(depths):
            fractions[row, column] = np.mean(selected[mask] == depth) if np.any(mask) else 0.0
    fig, axis = plt.subplots(figsize=(7.8, 4.2))
    bottom = np.zeros(len(regime_order))
    palette = ["#d1d5db", "#93c5fd", "#3b82f6", "#1e3a8a"]
    for column, depth in enumerate(depths):
        axis.bar(regime_order, fractions[:, column], bottom=bottom, label=f"depth {depth}", color=palette[column])
        bottom += fractions[:, column]
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of transitions")
    axis.set_title("Post-hoc compute allocation by physical regime")
    axis.legend(frameon=False, ncol=len(depths))
    axis.tick_params(axis="x", rotation=15)
    return _finish(fig, path)


def write_all_figures(
    *,
    depth_rows: list[dict[str, object]],
    budget_rows: list[dict[str, object]],
    raw_losses: np.ndarray,
    depths: tuple[int, ...],
    allocation: np.ndarray,
    regimes: np.ndarray,
    output_dir: Path,
) -> dict[str, str]:
    paths = {
        "depth_curve": plot_depth_curve(depth_rows, output_dir / "01_error_vs_depth.png"),
        "gain_distribution": plot_gain_distribution(raw_losses, depths, output_dir / "02_compute_benefit_distribution.png"),
        "budget_curve": plot_budget_curve(budget_rows, output_dir / "03_budget_curves.png"),
        "allocation_by_regime": plot_allocation_by_regime(
            allocation, regimes, depths, output_dir / "04_allocation_by_regime.png"
        ),
    }
    return {name: str(path) for name, path in paths.items()}

