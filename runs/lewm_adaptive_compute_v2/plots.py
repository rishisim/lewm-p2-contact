"""Deterministic, noninteractive figures for LeWM adaptive-compute V2.

The plotting API consumes already-aggregated, target-scored metric rows.  It
does not load experiment caches or outcomes itself.  This keeps figure inputs
explicit and lets the experiment driver hash the exact source tables used by
each figure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEPTHS = (1, 2, 4)
POLICY_ORDER = ("adaptive", "uniform_d2", "fixed_d1", "random", "permutation", "oracle")
POLICY_LABELS = {
    "adaptive": "Adaptive",
    "uniform_d2": "Uniform depth 2",
    "fixed_d1": "Fixed depth 1",
    "random": "Random exact budget",
    "permutation": "Histogram permutation",
    "oracle": "Sequential oracle",
}
COLORS = {
    "adaptive": "#2563eb",
    "uniform_d2": "#9ca3af",
    "fixed_d1": "#d1d5db",
    "random": "#d97706",
    "permutation": "#7c3aed",
    "oracle": "#111827",
    "depth1": "#dbeafe",
    "depth2": "#60a5fa",
    "depth4": "#1e3a8a",
    "ink": "#1f2937",
    "grid": "#d1d5db",
}


def _normalise_json(value: Any) -> Any:
    """Convert common scalar/container types to strict, canonical JSON data."""

    if isinstance(value, Mapping):
        return {str(key): _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalise_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Figure source data must contain only finite values")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported figure source value: {type(value).__name__}")


def source_data_integrity(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return a stable SHA-256, row count, and columns for a source table.

    Row order is intentionally significant: the digest identifies the exact
    ordered table handed to a plot, not merely an unordered set of values.
    """

    normalised = _normalise_json(list(rows))
    columns = sorted({key for row in normalised for key in row})
    payload = json.dumps(normalised, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": len(normalised),
        "columns": columns,
    }


def _require_rows(
    rows: Sequence[Mapping[str, object]], required: Sequence[str], source_name: str
) -> list[dict[str, object]]:
    copied = [dict(row) for row in rows]
    if not copied:
        raise ValueError(f"{source_name} must not be empty")
    for index, row in enumerate(copied):
        missing = [column for column in required if column not in row]
        if missing:
            raise ValueError(f"{source_name}[{index}] missing columns: {missing}")
    # This also rejects NaN and infinity before Matplotlib receives them.
    source_data_integrity(copied)
    return copied


def _finite_float(value: object, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _ordered(values: Sequence[str], preferred: Sequence[str]) -> list[str]:
    seen = set(values)
    return [value for value in preferred if value in seen] + sorted(seen.difference(preferred))


def _style_axis(axis: plt.Axes, *, x_grid: bool = False, y_grid: bool = True) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="x" if x_grid else "y", color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    if not y_grid and not x_grid:
        axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors=COLORS["ink"])


def _finish(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", metadata={"Software": "LeWM V2 plots.py"})
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not written: {path}")
    decoded = plt.imread(path)
    if decoded.ndim not in (2, 3) or decoded.shape[0] < 400 or decoded.shape[1] < 600:
        raise RuntimeError(f"Figure failed PNG decode/dimension audit: {path} {decoded.shape}")
    return path


def png_integrity(path: Path) -> dict[str, object]:
    """Decode a PNG and return byte/dimension metadata for artifact auditing."""

    path = Path(path)
    decoded = plt.imread(path)
    if decoded.ndim not in (2, 3):
        raise ValueError(f"Unexpected decoded PNG shape: {decoded.shape}")
    payload = path.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "width_px": int(decoded.shape[1]),
        "height_px": int(decoded.shape[0]),
        "channels": 1 if decoded.ndim == 2 else int(decoded.shape[2]),
    }


def plot_policy_performance(
    policy_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> Path:
    """Plot raw/whitened policy losses and paired differences with 95% CIs.

    ``policy_rows`` columns: ``policy``, ``raw_mean_loss``,
    ``whitened_mean_loss``. ``comparison_rows`` columns: ``comparison``,
    ``raw_mean_difference``, ``raw_ci_lower``, ``raw_ci_upper``, and the
    analogous three ``whitened_*`` columns.  Comparison labels must state the
    subtraction direction; the plot does not silently reinterpret signs.
    """

    policies = _require_rows(
        policy_rows, ("policy", "raw_mean_loss", "whitened_mean_loss"), "policy_rows"
    )
    comparisons = _require_rows(
        comparison_rows,
        (
            "comparison",
            "raw_mean_difference",
            "raw_ci_lower",
            "raw_ci_upper",
            "whitened_mean_difference",
            "whitened_ci_lower",
            "whitened_ci_upper",
        ),
        "comparison_rows",
    )
    policy_names = _ordered([str(row["policy"]) for row in policies], POLICY_ORDER)
    by_policy = {str(row["policy"]): row for row in policies}
    if len(by_policy) != len(policies):
        raise ValueError("policy_rows must contain one row per policy")

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    for column, (prefix, title, ylabel) in enumerate(
        (
            ("raw", "Policy raw latent loss", "Mean per-dimension MSE"),
            ("whitened", "Policy calibration-whitened loss", "Mean whitened MSE"),
        )
    ):
        axis = axes[0, column]
        values = [_finite_float(by_policy[name][f"{prefix}_mean_loss"], f"{name} loss") for name in policy_names]
        if any(value < 0 for value in values):
            raise ValueError("Policy losses must be nonnegative")
        bars = axis.bar(
            np.arange(len(policy_names)),
            values,
            color=[COLORS.get(name, "#64748b") for name in policy_names],
            edgecolor=COLORS["ink"],
            linewidth=0.6,
        )
        axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        axis.set_xticks(np.arange(len(policy_names)), [POLICY_LABELS.get(name, name) for name in policy_names], rotation=20, ha="right")
        axis.set_ylim(bottom=0)
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_ylabel(ylabel)
        _style_axis(axis)

        difference_key = f"{prefix}_mean_difference"
        lower_key = f"{prefix}_ci_lower"
        upper_key = f"{prefix}_ci_upper"
        contrast_axis = axes[1, column]
        labels = [str(row["comparison"]) for row in comparisons]
        means = np.asarray([_finite_float(row[difference_key], difference_key) for row in comparisons])
        lowers = np.asarray([_finite_float(row[lower_key], lower_key) for row in comparisons])
        uppers = np.asarray([_finite_float(row[upper_key], upper_key) for row in comparisons])
        if np.any(lowers > means) or np.any(means > uppers):
            raise ValueError(f"{prefix} comparison confidence intervals must contain their means")
        positions = np.arange(len(comparisons))
        contrast_axis.errorbar(
            means,
            positions,
            xerr=np.vstack((means - lowers, uppers - means)),
            fmt="o",
            color=COLORS["adaptive"],
            ecolor=COLORS["ink"],
            capsize=3,
            linewidth=1.2,
        )
        contrast_axis.axvline(0.0, color=COLORS["ink"], linestyle="--", linewidth=1)
        contrast_axis.set_yticks(positions, labels)
        contrast_axis.invert_yaxis()
        contrast_axis.set_title(f"Paired {prefix} differences (episode-clustered 95% CI)", loc="left", fontweight="semibold")
        contrast_axis.set_xlabel("Difference; sign follows label")
        _style_axis(contrast_axis, x_grid=True, y_grid=False)
    fig.suptitle("Sequential gate policy performance", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.965, "Untouched test split; lower loss is better. Error bars are paired episode-clustered intervals.", fontsize=9, color="#4b5563")
    return _finish(fig, path)


def plot_gate_diagnostics(
    calibration_rows: Sequence[Mapping[str, object]],
    quantile_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> Path:
    """Plot calibration ranking correlations and observed benefit by score quintile.

    ``calibration_rows`` columns: ``benefit`` (``b12`` or ``b14``),
    ``pearson``, ``spearman``, ``rmse``, ``mae``. ``quantile_rows`` columns:
    ``benefit``, ``quantile``, ``mean_predicted``, ``mean_actual``, ``count``.
    The predicted mean is retained in source integrity metadata; the chart uses
    it only to establish the preregistered score order because standardized
    scores and raw benefits need not share units.
    """

    calibration = _require_rows(
        calibration_rows, ("benefit", "pearson", "spearman", "rmse", "mae"), "calibration_rows"
    )
    quantiles = _require_rows(
        quantile_rows,
        ("benefit", "quantile", "mean_predicted", "mean_actual", "count"),
        "quantile_rows",
    )
    benefits = _ordered([str(row["benefit"]) for row in calibration], ("b12", "b14"))
    by_benefit = {str(row["benefit"]): row for row in calibration}
    if len(by_benefit) != len(calibration):
        raise ValueError("calibration_rows must contain one row per benefit")
    if set(str(row["benefit"]) for row in quantiles) != set(benefits):
        raise ValueError("Calibration and quantile benefit sets must match")

    fig = plt.figure(figsize=(12.2, 4.4), constrained_layout=True)
    grid = fig.add_gridspec(1, len(benefits) + 1, width_ratios=[0.95] + [1.25] * len(benefits))
    corr_axis = fig.add_subplot(grid[0, 0])
    positions = np.arange(len(benefits))
    width = 0.34
    pearson = [_finite_float(by_benefit[name]["pearson"], "pearson") for name in benefits]
    spearman = [_finite_float(by_benefit[name]["spearman"], "spearman") for name in benefits]
    if any(abs(value) > 1.0 for value in pearson + spearman):
        raise ValueError("Correlations must lie in [-1, 1]")
    corr_axis.bar(positions - width / 2, pearson, width, label="Pearson", color="#93c5fd", edgecolor=COLORS["ink"], linewidth=0.5)
    corr_axis.bar(positions + width / 2, spearman, width, label="Spearman", color=COLORS["adaptive"], edgecolor=COLORS["ink"], linewidth=0.5)
    corr_axis.axhline(0.0, color=COLORS["ink"], linewidth=0.9)
    corr_axis.set_xticks(positions, benefits)
    corr_axis.set_ylim(-1.0, 1.0)
    corr_axis.set_ylabel("Correlation")
    corr_axis.set_title("Calibration ranking", loc="left", fontweight="semibold")
    corr_axis.legend(frameon=False, fontsize=8)
    _style_axis(corr_axis)

    for column, benefit in enumerate(benefits, start=1):
        axis = fig.add_subplot(grid[0, column])
        rows = sorted(
            [row for row in quantiles if str(row["benefit"]) == benefit],
            key=lambda row: int(row["quantile"]),
        )
        if [int(row["quantile"]) for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"{benefit} quantiles must be consecutive starting at 1")
        if len(rows) != 5:
            raise ValueError(f"{benefit} must have exactly five predicted-score quantiles")
        predicted = np.asarray([_finite_float(row["mean_predicted"], "mean_predicted") for row in rows])
        if np.any(np.diff(predicted) < -1e-12):
            raise ValueError(f"{benefit} quantiles must be ordered by nondecreasing predicted score")
        actual = np.asarray([_finite_float(row["mean_actual"], "mean_actual") for row in rows])
        counts = np.asarray([int(row["count"]) for row in rows])
        if np.any(counts <= 0):
            raise ValueError("Quantile counts must be positive")
        axis.plot(range(1, 6), actual, marker="o", color=COLORS["adaptive"], linewidth=2)
        axis.axhline(0.0, color=COLORS["ink"], linestyle="--", linewidth=0.9)
        axis.set_xticks(range(1, 6))
        axis.set_xlabel("Predicted-score quintile (low to high)")
        axis.set_ylabel("Mean actual raw-MSE benefit")
        axis.set_title(f"{benefit}: benefit by score quintile", loc="left", fontweight="semibold")
        metric_row = by_benefit[benefit]
        annotation = (
            f"RMSE {_finite_float(metric_row['rmse'], 'rmse'):.3g}\n"
            f"MAE {_finite_float(metric_row['mae'], 'mae'):.3g}\n"
            f"n={int(counts.sum()):,}"
        )
        axis.text(0.03, 0.97, annotation, transform=axis.transAxes, va="top", fontsize=8, color="#4b5563")
        _style_axis(axis)
    fig.suptitle("Sequential gate calibration diagnostics", x=0.01, ha="left", fontsize=15, fontweight="bold")
    return _finish(fig, path)


def plot_exact_allocation(
    allocation_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> Path:
    """Plot exact depth histograms and validate call totals.

    ``allocation_rows`` columns: ``policy``, ``depth``, ``count``,
    ``total_calls``. There must be one row for every policy/depth pair for
    depths 1, 2, and 4. Repeated ``total_calls`` values are audited against
    ``sum(depth * count)``.
    """

    rows = _require_rows(allocation_rows, ("policy", "depth", "count", "total_calls"), "allocation_rows")
    policies = _ordered([str(row["policy"]) for row in rows], POLICY_ORDER)
    fig, axis = plt.subplots(figsize=(10.2, 5.0), constrained_layout=True)
    positions = np.arange(len(policies))
    bottoms = np.zeros(len(policies), dtype=float)
    totals: list[int] = []
    sample_counts: list[int] = []
    for policy in policies:
        selected = [row for row in rows if str(row["policy"]) == policy]
        depth_map = {int(row["depth"]): row for row in selected}
        if set(depth_map) != set(DEPTHS) or len(selected) != len(DEPTHS):
            raise ValueError(f"{policy} must have exactly one allocation row for each depth in {DEPTHS}")
        counts = [int(depth_map[depth]["count"]) for depth in DEPTHS]
        if any(count < 0 for count in counts) or sum(counts) <= 0:
            raise ValueError(f"{policy} allocation counts must be nonnegative with positive total")
        declared = {int(row["total_calls"]) for row in selected}
        if len(declared) != 1:
            raise ValueError(f"{policy} total_calls must be constant across depth rows")
        actual_calls = sum(depth * count for depth, count in zip(DEPTHS, counts))
        if actual_calls != next(iter(declared)):
            raise ValueError(f"{policy} total_calls mismatch: declared {declared}, computed {actual_calls}")
        totals.append(actual_calls)
        sample_counts.append(sum(counts))

    for depth in DEPTHS:
        counts = []
        fractions = []
        for policy, sample_count in zip(policies, sample_counts):
            row = next(row for row in rows if str(row["policy"]) == policy and int(row["depth"]) == depth)
            count = int(row["count"])
            counts.append(count)
            fractions.append(count / sample_count)
        bars = axis.bar(
            positions,
            fractions,
            bottom=bottoms,
            label=f"depth {depth}",
            color=COLORS[f"depth{depth}"],
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, count, fraction in zip(bars, counts, fractions):
            if fraction >= 0.08:
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{count:,}", ha="center", va="center", fontsize=8, color=COLORS["ink"])
        bottoms += np.asarray(fractions)
    labels = [f"{POLICY_LABELS.get(policy, policy)}\n{calls:,} calls" for policy, calls in zip(policies, totals)]
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of transitions (counts labeled)")
    axis.set_title("Exact final-depth allocation histograms", loc="left", fontweight="semibold")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    _style_axis(axis)
    return _finish(fig, path)


def plot_posthoc_regimes(
    regime_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> Path:
    """Plot explicitly post-hoc allocation and raw benefit by physical regime.

    ``regime_rows`` columns: ``regime``, ``depth``, ``count``, ``fraction``,
    ``raw_benefit_vs_uniform``, and ``n``. Benefit and n repeat across a
    regime's depth rows and are checked for consistency.
    """

    rows = _require_rows(
        regime_rows,
        ("regime", "depth", "count", "fraction", "raw_benefit_vs_uniform", "n"),
        "regime_rows",
    )
    preferred = ("impact", "contact", "transport_free", "static")
    regimes = _ordered([str(row["regime"]) for row in rows], preferred)
    fractions = np.zeros((len(regimes), len(DEPTHS)), dtype=float)
    benefits = np.zeros(len(regimes), dtype=float)
    counts_n = np.zeros(len(regimes), dtype=int)
    for regime_index, regime in enumerate(regimes):
        selected = [row for row in rows if str(row["regime"]) == regime]
        depth_map = {int(row["depth"]): row for row in selected}
        if set(depth_map) != set(DEPTHS) or len(selected) != len(DEPTHS):
            raise ValueError(f"{regime} must have exactly one row for each depth in {DEPTHS}")
        n_values = {int(row["n"]) for row in selected}
        benefit_values = {_finite_float(row["raw_benefit_vs_uniform"], "raw_benefit_vs_uniform") for row in selected}
        if len(n_values) != 1 or len(benefit_values) != 1:
            raise ValueError(f"{regime} repeated n/benefit values must agree")
        n = next(iter(n_values))
        if n <= 0:
            raise ValueError("Regime n must be positive")
        regime_counts = np.asarray([int(depth_map[depth]["count"]) for depth in DEPTHS])
        regime_fractions = np.asarray([_finite_float(depth_map[depth]["fraction"], "fraction") for depth in DEPTHS])
        if np.any(regime_counts < 0) or int(regime_counts.sum()) != n:
            raise ValueError(f"{regime} depth counts must sum to n")
        if np.any(regime_fractions < 0) or not np.isclose(regime_fractions.sum(), 1.0, atol=1e-8):
            raise ValueError(f"{regime} fractions must sum to one")
        if not np.allclose(regime_fractions, regime_counts / n, atol=1e-8):
            raise ValueError(f"{regime} fractions must equal count / n")
        fractions[regime_index] = regime_fractions
        benefits[regime_index] = next(iter(benefit_values))
        counts_n[regime_index] = n

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    positions = np.arange(len(regimes))
    bottom = np.zeros(len(regimes))
    for column, depth in enumerate(DEPTHS):
        axes[0].bar(
            positions,
            fractions[:, column],
            bottom=bottom,
            color=COLORS[f"depth{depth}"],
            label=f"depth {depth}",
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += fractions[:, column]
    labels = [f"{regime.replace('_', ' ')}\n(n={n:,})" for regime, n in zip(regimes, counts_n)]
    axes[0].set_xticks(positions, labels, rotation=12, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of transitions")
    axes[0].set_title("Adaptive allocation by regime", loc="left", fontweight="semibold")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    _style_axis(axes[0])

    benefit_bars = axes[1].barh(positions, benefits, color=COLORS["adaptive"], edgecolor=COLORS["ink"], linewidth=0.6)
    axes[1].axvline(0.0, color=COLORS["ink"], linestyle="--", linewidth=1)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Uniform depth-2 raw MSE − adaptive raw MSE")
    axes[1].set_title("Raw benefit by regime", loc="left", fontweight="semibold")
    axes[1].bar_label(benefit_bars, fmt="%.3g", padding=3, fontsize=8)
    _style_axis(axes[1], x_grid=True, y_grid=False)
    fig.suptitle("Post-hoc physical-regime interpretation", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.95, "Descriptive only; regimes were attached after gate predictions and do not alter the primary decision.", fontsize=9, color="#4b5563")
    return _finish(fig, path)


def write_all_figures(
    *,
    policy_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    quantile_rows: Sequence[Mapping[str, object]],
    allocation_rows: Sequence[Mapping[str, object]],
    regime_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    """Write all preregistered V2 figures and an integrity manifest.

    Required columns are documented on the four plotting functions.  Returned
    ``sources`` hashes identify the exact ordered input rows.  The JSON
    manifest uses strict JSON and records decoded PNG hashes and dimensions.
    """

    output_dir = Path(output_dir)
    sources = {
        "policy_rows": source_data_integrity(policy_rows),
        "comparison_rows": source_data_integrity(comparison_rows),
        "calibration_rows": source_data_integrity(calibration_rows),
        "quantile_rows": source_data_integrity(quantile_rows),
        "allocation_rows": source_data_integrity(allocation_rows),
        "regime_rows": source_data_integrity(regime_rows),
    }
    paths = {
        "policy_performance": plot_policy_performance(
            policy_rows, comparison_rows, output_dir / "01_policy_losses_and_comparisons.png"
        ),
        "gate_diagnostics": plot_gate_diagnostics(
            calibration_rows, quantile_rows, output_dir / "02_gate_calibration_quantiles.png"
        ),
        "exact_allocation": plot_exact_allocation(
            allocation_rows, output_dir / "03_exact_allocation_histogram.png"
        ),
        "posthoc_regimes": plot_posthoc_regimes(
            regime_rows, output_dir / "04_posthoc_regimes.png"
        ),
    }
    pngs = {name: png_integrity(path) for name, path in paths.items()}
    manifest_path = output_dir / "figure_integrity.json"
    manifest = {
        "schema_version": 1,
        "sources": sources,
        "figures": {name: {"path": path.name, **pngs[name]} for name, path in paths.items()},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sources": sources,
        "pngs": pngs,
        "manifest": str(manifest_path),
    }
