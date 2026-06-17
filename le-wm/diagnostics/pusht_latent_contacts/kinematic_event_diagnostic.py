#!/usr/bin/env python3
"""Diagnose whether PushT MSE spikes align with kinematic discontinuities."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, fisher_exact, wilcoxon


DEFAULT_RECORDS = Path("le-wm/diagnostics/pusht_latent_contacts/prediction_errors.csv")
DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/pusht_latent_contacts/kinematic_events")
DEFAULT_H5_CANDIDATES = [
    Path("le-wm/.cache/diagnostic/dataset/pusht_subset_n30_fs5_seed0.h5"),
    Path(
        "/Users/rishisim/Documents/research/World Models/"
        "le-wm/.cache/diagnostic/dataset/pusht_subset_n30_fs5_seed0.h5"
    ),
]


@dataclass(frozen=True)
class EventSpec:
    key: str
    name: str
    feature: str
    threshold_col: str
    event_col: str
    marker: str
    color: str


EVENTS = [
    EventSpec(
        key="block_kinematic_discontinuity",
        name="Block velocity/acceleration discontinuity",
        feature="block_acceleration_proxy",
        threshold_col="block_acceleration_proxy_p90",
        event_col="event_block_kinematic_discontinuity",
        marker="o",
        color="#d62728",
    ),
    EventSpec(
        key="pusher_acceleration_proxy",
        name="Pusher acceleration proxy",
        feature="pusher_acceleration_proxy",
        threshold_col="pusher_acceleration_proxy_p90",
        event_col="event_pusher_acceleration_proxy",
        marker="^",
        color="#2ca02c",
    ),
    EventSpec(
        key="joint_acceleration_proxy",
        name="Joint acceleration proxy",
        feature="joint_acceleration_proxy",
        threshold_col="joint_acceleration_proxy_p90",
        event_col="event_joint_acceleration_proxy",
        marker="s",
        color="#9467bd",
    ),
    EventSpec(
        key="orientation_delta",
        name="Orientation delta",
        feature="orientation_delta",
        threshold_col="orientation_delta_p90",
        event_col="event_orientation_delta",
        marker="D",
        color="#ff7f0e",
    ),
    EventSpec(
        key="late_trajectory_position",
        name="Normalized trajectory position",
        feature="normalized_position",
        threshold_col="normalized_position_p90",
        event_col="event_late_trajectory_position",
        marker="x",
        color="#6b7280",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--dataset-h5", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--event-percentile", type=float, default=90.0)
    parser.add_argument("--high-mse-percentile", type=float, default=90.0)
    parser.add_argument("--window", type=int, default=20)
    return parser.parse_args()


def find_h5(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else DEFAULT_H5_CANDIDATES
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    searched = "\n".join(f"- {p}" for p in candidates if p is not None)
    raise FileNotFoundError(f"No PushT subset HDF5 found. Searched:\n{searched}")


def load_states(h5_path: Path) -> dict[int, np.ndarray]:
    states_by_episode: dict[int, np.ndarray] = {}
    with h5py.File(h5_path, "r") as h5:
        if "state" not in h5:
            raise KeyError(f"{h5_path} does not contain a 'state' dataset")
        lengths = np.asarray(h5["ep_len"])
        offsets = np.asarray(h5["ep_offset"])
        states = h5["state"]
        for episode_id, (offset, length) in enumerate(zip(offsets, lengths)):
            start = int(offset)
            stop = start + int(length)
            states_by_episode[int(episode_id)] = np.asarray(states[start:stop], dtype=np.float64)
    return states_by_episode


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def wrapped_angle_delta(angle_t: float, angle_prev: float) -> float:
    delta = angle_t - angle_prev
    return float(abs(math.atan2(math.sin(delta), math.cos(delta))))


def attach_kinematic_features(
    records: pd.DataFrame,
    states_by_episode: dict[int, np.ndarray],
    *,
    frameskip: int,
) -> pd.DataFrame:
    rows = []
    dropped = 0
    for row in records.itertuples(index=False):
        episode_id = int(row.episode_id)
        states = states_by_episode[episode_id]
        raw_step = int(row.raw_step)
        prev_step = raw_step - frameskip
        prev2_step = raw_step - 2 * frameskip
        if prev2_step < 0 or raw_step >= len(states):
            dropped += 1
            continue

        state_t = states[raw_step]
        state_prev = states[prev_step]
        state_prev2 = states[prev2_step]

        block_velocity = state_t[2:4] - state_prev[2:4]
        block_velocity_prev = state_prev[2:4] - state_prev2[2:4]
        block_acceleration = state_t[2:4] - 2.0 * state_prev[2:4] + state_prev2[2:4]
        pusher_acceleration = state_t[0:2] - 2.0 * state_prev[0:2] + state_prev2[0:2]

        block_velocity_discontinuity = float(np.linalg.norm(block_velocity - block_velocity_prev))
        block_acceleration_proxy = float(np.linalg.norm(block_acceleration))
        pusher_acceleration_proxy = float(np.linalg.norm(pusher_acceleration))
        joint_acceleration_proxy = float(
            math.sqrt(block_acceleration_proxy**2 + pusher_acceleration_proxy**2)
        )
        orientation_delta = wrapped_angle_delta(float(state_t[4]), float(state_prev[4]))
        normalized_position = raw_step / max(len(states) - 1, 1)

        rows.append(
            {
                "episode_id": episode_id,
                "model_step": int(row.model_step),
                "raw_step": raw_step,
                "transition_block": int(row.transition_block),
                "mse": float(row.mse),
                "n_contacts": float(row.n_contacts),
                "rel_contact": float(row.rel_contact),
                "block_velocity_discontinuity": block_velocity_discontinuity,
                "block_acceleration_proxy": block_acceleration_proxy,
                "pusher_acceleration_proxy": pusher_acceleration_proxy,
                "joint_acceleration_proxy": joint_acceleration_proxy,
                "orientation_delta": orientation_delta,
                "normalized_position": float(normalized_position),
            }
        )

    enriched = pd.DataFrame(rows)
    enriched.attrs["dropped_insufficient_history"] = dropped
    return enriched


def add_event_labels(
    df: pd.DataFrame,
    *,
    event_percentile: float,
    high_mse_percentile: float,
) -> tuple[pd.DataFrame, dict[str, float], float]:
    out = df.copy()
    thresholds: dict[str, float] = {}
    for spec in EVENTS:
        threshold = float(np.percentile(out[spec.feature], event_percentile))
        thresholds[spec.feature] = threshold
        out[spec.threshold_col] = threshold
        out[spec.event_col] = out[spec.feature] >= threshold

    high_mse_threshold = float(np.percentile(out["mse"], high_mse_percentile))
    out["high_mse_p90"] = high_mse_threshold
    out["high_mse"] = out["mse"] >= high_mse_threshold
    return out, thresholds, high_mse_threshold


def aligned_curve(df: pd.DataFrame, spec: EventSpec, *, window: int) -> pd.DataFrame:
    pieces = []
    event_rows = df.loc[df[spec.event_col], ["episode_id", "model_step"]].reset_index(drop=True)
    for event_id, event in event_rows.iterrows():
        episode_id = int(event["episode_id"])
        event_step = int(event["model_step"])
        group = df.loc[df["episode_id"] == episode_id, ["episode_id", "model_step", "mse"]].copy()
        group["event_id"] = int(event_id)
        group["event_model_step"] = event_step
        group["rel_event"] = group["model_step"] - event_step
        pieces.append(group.loc[group["rel_event"].between(-window, window)])

    if not pieces:
        return pd.DataFrame(columns=["rel_event", "mean", "std", "n", "ci95"])

    aligned = pd.concat(pieces, ignore_index=True)
    summary = (
        aligned.groupby("rel_event")["mse"]
        .agg(["mean", "std", "count"])
        .rename(columns={"count": "n"})
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    return summary


def aligned_lift(curve: pd.DataFrame) -> float:
    if len(curve) == 0:
        return float("nan")
    by_rel = curve.set_index("rel_event")
    if -5 not in by_rel.index or 0 not in by_rel.index:
        return float("nan")
    baseline = float(by_rel.loc[-5, "mean"])
    return float(by_rel.loc[0, "mean"]) / baseline if baseline else float("nan")


def curve_values(curve: pd.DataFrame) -> dict[str, float | bool]:
    if len(curve) == 0:
        return {
            "mean_mse_tminus5": float("nan"),
            "mean_mse_t0": float("nan"),
            "mean_mse_tplus5": float("nan"),
            "localized_peak_pass": False,
        }
    by_rel = curve.set_index("rel_event")
    tminus5 = float(by_rel.loc[-5, "mean"]) if -5 in by_rel.index else float("nan")
    t0 = float(by_rel.loc[0, "mean"]) if 0 in by_rel.index else float("nan")
    tplus5 = float(by_rel.loc[5, "mean"]) if 5 in by_rel.index else float("nan")
    post = by_rel.loc[(by_rel.index >= 1) & (by_rel.index <= 10), "mean"]
    localized = (
        np.isfinite(tminus5)
        and np.isfinite(t0)
        and np.isfinite(tplus5)
        and len(post) >= 3
        and t0 > tminus5
        and t0 >= tplus5
        and t0 >= float(post.mean())
    )
    return {
        "mean_mse_tminus5": tminus5,
        "mean_mse_t0": t0,
        "mean_mse_tplus5": tplus5,
        "localized_peak_pass": bool(localized),
    }


def fit_logistic_loglik(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    def objective(beta: np.ndarray) -> float:
        if not np.all(np.isfinite(beta)):
            return 1e100
        beta = np.clip(beta, -30.0, 30.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = x @ beta
        if not np.all(np.isfinite(logits)):
            return 1e100
        return float(np.sum(np.logaddexp(0.0, logits) - y * logits))

    bounds = [(-30.0, 30.0)] * x.shape[1]
    result = minimize(
        objective,
        np.zeros(x.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not result.success:
        result = minimize(
            objective,
            np.zeros(x.shape[1], dtype=np.float64),
            method="Powell",
            bounds=bounds,
        )
    beta = np.asarray(result.x, dtype=np.float64)
    return beta, -objective(beta)


def position_adjusted_logistic(df: pd.DataFrame, spec: EventSpec) -> dict[str, float | bool]:
    y = df["high_mse"].astype(float).to_numpy()
    event = df[spec.event_col].astype(float).to_numpy()
    position = df["normalized_position"].astype(float).to_numpy()
    null_x = np.column_stack([np.ones(len(df)), position])
    full_x = np.column_stack([np.ones(len(df)), event, position])
    try:
        null_beta, null_ll = fit_logistic_loglik(y, null_x)
        full_beta, full_ll = fit_logistic_loglik(y, full_x)
        lr_stat = max(0.0, 2.0 * (full_ll - null_ll))
        p_value = float(chi2.sf(lr_stat, df=1))
        adjusted_or = float(math.exp(np.clip(full_beta[1], -30, 30)))
        position_or = float(math.exp(np.clip(full_beta[2], -30, 30)))
        return {
            "position_adjusted_odds_ratio": adjusted_or,
            "position_adjusted_lr_p": p_value,
            "position_adjusted_position_or": position_or,
            "position_adjusted_pass": adjusted_or > 2.0 and p_value < 0.05,
        }
    except Exception:
        return {
            "position_adjusted_odds_ratio": float("nan"),
            "position_adjusted_lr_p": float("nan"),
            "position_adjusted_position_or": float("nan"),
            "position_adjusted_pass": False,
        }


def trajectory_level_metrics(df: pd.DataFrame, spec: EventSpec) -> dict[str, float | bool]:
    deltas = []
    odds_ratios = []
    for _, group in df.groupby("episode_id", sort=False):
        event = group[spec.event_col].astype(bool)
        high = group["high_mse"].astype(bool)
        if event.any():
            event_rate = float(high[event].mean())
            non_event_rate = float(high[~event].mean()) if (~event).any() else 0.0
            deltas.append(event_rate - non_event_rate)

        a = int((event & high).sum())
        b = int((event & ~high).sum())
        c = int((~event & high).sum())
        d = int((~event & ~high).sum())
        if event.any() and (~event).any():
            odds_ratios.append(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))

    if deltas:
        try:
            wilcoxon_p = float(wilcoxon(deltas, alternative="greater", zero_method="zsplit").pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        mean_delta = float(np.mean(deltas))
    else:
        wilcoxon_p = float("nan")
        mean_delta = float("nan")

    return {
        "trajectory_event_episodes": len(deltas),
        "trajectory_mean_high_mse_delta": mean_delta,
        "trajectory_median_odds_ratio": float(np.median(odds_ratios)) if odds_ratios else float("nan"),
        "trajectory_wilcoxon_p": wilcoxon_p,
        "trajectory_pass": bool(np.isfinite(mean_delta) and mean_delta > 0.0 and wilcoxon_p < 0.05),
    }


def enrichment_metrics(
    df: pd.DataFrame,
    spec: EventSpec,
    *,
    threshold: float,
    lift: float,
    curve: pd.DataFrame,
) -> dict[str, object]:
    event = df[spec.event_col].astype(bool)
    high = df["high_mse"].astype(bool)
    a = int((event & high).sum())
    b = int((event & ~high).sum())
    c = int((~event & high).sum())
    d = int((~event & ~high).sum())
    odds_ratio, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")

    event_count = int(event.sum())
    high_count = int(high.sum())
    event_rate = 100.0 * event_count / len(df)
    p_high_given_event = a / event_count if event_count else float("nan")
    p_event_given_high = a / high_count if high_count else float("nan")
    trajectories = int(df.loc[event, "episode_id"].nunique())
    enrichment_pass = odds_ratio > 2.0 and fisher_p < 0.05
    lift_pass = bool(np.isfinite(lift) and lift > 1.3)
    rate_pass = 5.0 <= event_rate <= 30.0
    curve_info = curve_values(curve)
    adjusted = position_adjusted_logistic(df, spec)
    trajectory = trajectory_level_metrics(df, spec)

    if (
        rate_pass
        and enrichment_pass
        and lift_pass
        and adjusted["position_adjusted_pass"]
        and trajectory["trajectory_pass"]
        and curve_info["localized_peak_pass"]
    ):
        verdict = "explains"
    elif enrichment_pass or lift_pass or adjusted["position_adjusted_pass"] or trajectory["trajectory_pass"]:
        verdict = "partial/promising"
    else:
        verdict = "no"

    return {
        "event": spec.name,
        "feature": spec.feature,
        "threshold_p90": threshold,
        "event_count": event_count,
        "event_rate_pct": event_rate,
        "trajectories": trajectories,
        "high_mse_count": high_count,
        "event_and_high_mse": a,
        "p_high_mse_given_event": p_high_given_event,
        "p_event_given_high_mse": p_event_given_high,
        "odds_ratio": float(odds_ratio),
        "fisher_p_greater": float(fisher_p),
        "aligned_lift_t0_over_tminus5": lift,
        **curve_info,
        **adjusted,
        **trajectory,
        "rate_pass": rate_pass,
        "enrichment_pass": enrichment_pass,
        "aligned_lift_pass": lift_pass,
        "verdict": verdict,
    }


def plot_raster(df: pd.DataFrame, path: Path) -> None:
    episodes = sorted(df["episode_id"].unique())
    ncols = 3
    nrows = int(math.ceil(len(episodes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(8, nrows * 1.8)), sharex=False, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, episode_id in zip(axes_flat, episodes):
        group = df.loc[df["episode_id"] == episode_id].sort_values("model_step")
        ax.plot(group["model_step"], group["mse"], color="#1f2937", linewidth=1.2)
        ymax = max(float(group["mse"].max()), 1e-9)
        for spec in EVENTS:
            events = group.loc[group[spec.event_col]]
            if len(events):
                ax.scatter(
                    events["model_step"],
                    np.full(len(events), ymax * 1.05),
                    s=18,
                    marker=spec.marker,
                    color=spec.color,
                    alpha=0.85,
                    label=spec.name,
                )
        ax.set_title(f"Episode {episode_id}", fontsize=9)
        ax.grid(alpha=0.2)

    for ax in axes_flat[len(episodes) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2, fontsize=9)
    fig.supxlabel("Model step")
    fig.supylabel("One-step MSE")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_aligned_average(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for spec in EVENTS:
        curve = curves[spec.key]
        if len(curve) == 0:
            continue
        x = curve["rel_event"].to_numpy(dtype=np.float64)
        y = curve["mean"].to_numpy(dtype=np.float64)
        ci = curve["ci95"].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=spec.color, linewidth=2, label=spec.name)
        ax.fill_between(x, y - ci, y + ci, color=spec.color, alpha=0.14, linewidth=0)
    ax.axvline(0, color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("Model timestep relative to event")
    ax.set_ylabel("Mean one-step MSE")
    ax.set_title("All-event aligned MSE average")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def table_markdown(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def interpret(metrics: list[dict[str, object]]) -> str:
    explainers = [row for row in metrics if row["verdict"] == "explains"]
    partials = [row for row in metrics if row["verdict"] == "partial/promising"]
    if explainers:
        best = max(explainers, key=lambda row: float(row["odds_ratio"]))
        return (
            f"{best['event']} best explains the MSE spikes: it passes the event-rate, "
            "high-MSE enrichment, aligned-lift, position-control, trajectory-level, "
            "and localized-peak criteria."
        )
    if partials:
        best = max(partials, key=lambda row: float(row["position_adjusted_odds_ratio"]))
        return (
            "No kinematic mechanism cleanly explains the spikes under the stricter controls. "
            f"The strongest partial signal is {best['event']}; treat it as promising, "
            "unconfirmed evidence until the position and trajectory-level caveats are resolved."
        )
    return "None of the three kinematic mechanisms cleanly explains the MSE spikes."


def self_checks(df: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> list[str]:
    checks = []
    block_equal = bool(
        np.allclose(df["block_velocity_discontinuity"], df["block_acceleration_proxy"])
    )
    checks.append(f"block velocity discontinuity equals block acceleration proxy: {block_equal}")

    wrapped = wrapped_angle_delta(math.pi - 0.01, -math.pi + 0.01)
    checks.append(f"angle wrapping pi-to-minus-pi smoke value: {wrapped:.6g}")
    checks.append(f"angle wrapping smoke pass: {wrapped < 0.05}")

    for spec in EVENTS:
        event_count = int(df[spec.event_col].sum())
        aligned_n0 = 0
        curve = curves[spec.key]
        if len(curve) and 0 in set(curve["rel_event"]):
            aligned_n0 = int(curve.loc[curve["rel_event"] == 0, "n"].iloc[0])
        checks.append(f"{spec.key} all-event alignment n@0 equals event count: {aligned_n0 == event_count}")
    checks.append("normalized_position uses raw_step / (episode_raw_length - 1): True")
    return checks


def main() -> None:
    args = parse_args()
    h5_path = find_h5(args.dataset_h5)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.records)
    states_by_episode = load_states(h5_path)
    enriched = attach_kinematic_features(records, states_by_episode, frameskip=args.frameskip)
    labeled, thresholds, high_mse_threshold = add_event_labels(
        enriched,
        event_percentile=args.event_percentile,
        high_mse_percentile=args.high_mse_percentile,
    )

    curves = {spec.key: aligned_curve(labeled, spec, window=args.window) for spec in EVENTS}
    lifts = {spec.key: aligned_lift(curves[spec.key]) for spec in EVENTS}
    metrics = [
        enrichment_metrics(
            labeled,
            spec,
            threshold=thresholds[spec.feature],
            lift=lifts[spec.key],
            curve=curves[spec.key],
        )
        for spec in EVENTS
    ]
    checks = self_checks(labeled, curves)

    records_path = args.output_dir / "kinematic_event_records.csv"
    labeled.to_csv(records_path, index=False)

    for spec in EVENTS:
        curves[spec.key].to_csv(args.output_dir / f"{spec.key}_aligned_curve.csv", index=False)

    raster_path = args.output_dir / "kinematic_event_raster.png"
    aligned_path = args.output_dir / "kinematic_event_aligned_average.png"
    plot_raster(labeled, raster_path)
    plot_aligned_average(curves, aligned_path)

    metric_columns = [
        "event",
        "threshold_p90",
        "event_rate_pct",
        "trajectories",
        "p_high_mse_given_event",
        "p_event_given_high_mse",
        "odds_ratio",
        "fisher_p_greater",
        "aligned_lift_t0_over_tminus5",
        "mean_mse_tplus5",
        "localized_peak_pass",
        "position_adjusted_odds_ratio",
        "position_adjusted_lr_p",
        "trajectory_mean_high_mse_delta",
        "trajectory_wilcoxon_p",
        "verdict",
    ]
    summary = [
        "# Kinematic Event Diagnostic",
        "",
        f"- Records: `{args.records}`",
        f"- State source: `{h5_path}`",
        f"- Input prediction records: {len(records)}",
        f"- Records retained after requiring t, t-1, t-2 states: {len(labeled)}",
        f"- Dropped for insufficient history/state: {enriched.attrs['dropped_insufficient_history']}",
        f"- Event percentile: p{args.event_percentile:g}",
        f"- High-MSE threshold: p{args.high_mse_percentile:g} = {high_mse_threshold:.6g}",
        f"- Window: [-{args.window}, +{args.window}] model steps",
        f"- Raster: `{raster_path.name}`",
        f"- All-event aligned average: `{aligned_path.name}`",
        "",
        "## Metric Note",
        "",
        (
            "`block_velocity_discontinuity` and `block_acceleration_proxy` are the same "
            "second-difference norm on the strided block-position series; both columns are "
            "written for auditability, but the event is reported once."
        ),
        "",
        "## Enrichment And Aligned Lift",
        "",
        table_markdown(metrics, metric_columns),
        "",
        "## Position And Cluster Controls",
        "",
        (
            "The row-level Fisher p-values are retained for continuity but are optimistic "
            "because records cluster within trajectories. The `position_adjusted_*` columns "
            "fit `high_mse ~ event + normalized_position`; the trajectory columns aggregate "
            "within episodes before testing whether event steps have higher high-MSE rates."
        ),
        "",
        "## Verification Checks",
        "",
        "\n".join(f"- {check}" for check in checks),
        "",
        "## Interpretation",
        "",
        interpret(metrics),
        "",
    ]
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
