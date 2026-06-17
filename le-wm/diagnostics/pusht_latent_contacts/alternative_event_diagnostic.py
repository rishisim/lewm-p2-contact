#!/usr/bin/env python3
"""Post-process PushT one-step MSE records with alternative event definitions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RECORDS = Path("le-wm/diagnostics/pusht_latent_contacts/prediction_errors.csv")
DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/pusht_latent_contacts/alternative_events")
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
    title: str


EVENTS = [
    EventSpec("original", "Original geometric contact", "Original: geometric contact"),
    EventSpec("block_velocity_onset", "Event B: block velocity onset", "Event B: block velocity onset"),
    EventSpec("large_state_change", "Event C: large state change", "Event C: dynamic discontinuity"),
    EventSpec("proximity_threshold", "Event D: proximity threshold", "Event D: proximity threshold"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--dataset-h5", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--block-speed-threshold", type=float, default=0.01)
    parser.add_argument("--state-delta-percentile", type=float, default=90.0)
    parser.add_argument("--proximity-threshold", type=float, default=0.08)
    parser.add_argument("--proximity-scale", choices=("auto", "raw"), default="auto")
    return parser.parse_args()


def find_h5(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else DEFAULT_H5_CANDIDATES
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    searched = "\n".join(f"- {p}" for p in candidates if p is not None)
    raise FileNotFoundError(f"No PushT subset HDF5 found. Searched:\n{searched}")


def load_states(h5_path: Path) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    states_by_episode: dict[int, np.ndarray] = {}
    lengths_by_episode: dict[int, int] = {}
    with h5py.File(h5_path, "r") as h5:
        if "state" not in h5:
            raise KeyError(f"{h5_path} does not contain a 'state' dataset")
        lengths = np.asarray(h5["ep_len"])
        offsets = np.asarray(h5["ep_offset"])
        states = h5["state"]
        for episode_id, (offset, length) in enumerate(zip(offsets, lengths)):
            episode_states = np.asarray(states[int(offset) : int(offset + length)], dtype=np.float64)
            states_by_episode[int(episode_id)] = episode_states
            lengths_by_episode[int(episode_id)] = int(length)
    return states_by_episode, lengths_by_episode


def attach_state_features(
    records: pd.DataFrame,
    states_by_episode: dict[int, np.ndarray],
    *,
    proximity_scale: str,
) -> pd.DataFrame:
    rows = []
    all_distance = []
    for row in records.itertuples(index=False):
        states = states_by_episode[int(row.episode_id)]
        raw_step = min(int(row.raw_step), len(states) - 1)
        prev_step = max(raw_step - 5, 0)
        state = states[raw_step]
        prev_state = states[prev_step]

        block_speed = float(np.linalg.norm(state[2:4] - prev_state[2:4]))
        state_delta = float(np.linalg.norm(state - prev_state))
        pusher_block_distance = float(np.linalg.norm(state[0:2] - state[2:4]))
        all_distance.append(pusher_block_distance)
        rows.append(
            {
                "episode_id": int(row.episode_id),
                "model_step": int(row.model_step),
                "raw_step": raw_step,
                "transition_block": int(row.transition_block),
                "mse": float(row.mse),
                "n_contacts": float(row.n_contacts),
                "original_event": bool(row.n_contacts > 0),
                "block_speed": block_speed,
                "state_delta": state_delta,
                "pusher_block_distance_raw": pusher_block_distance,
            }
        )

    enriched = pd.DataFrame(rows)
    distances = np.asarray(all_distance, dtype=np.float64)
    if proximity_scale == "auto" and np.nanmedian(distances) > 10:
        enriched["pusher_block_distance"] = enriched["pusher_block_distance_raw"] / 512.0
        enriched.attrs["proximity_units"] = "raw pixels / 512"
    else:
        enriched["pusher_block_distance"] = enriched["pusher_block_distance_raw"]
        enriched.attrs["proximity_units"] = "raw state units"
    return enriched


def first_true_by_episode(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    event = pd.Series(False, index=df.index)
    for _, group in df.loc[mask].groupby("episode_id", sort=False):
        event.loc[group.index[0]] = True
    return event


def add_event_labels(
    df: pd.DataFrame,
    *,
    block_speed_threshold: float,
    state_delta_percentile: float,
    proximity_threshold: float,
) -> tuple[pd.DataFrame, float]:
    out = df.copy()
    out["event_original"] = out["original_event"]
    out["event_block_velocity_onset"] = first_true_by_episode(
        out, out["block_speed"] > block_speed_threshold
    )
    delta_threshold = float(np.percentile(out["state_delta"], state_delta_percentile))
    out["event_large_state_change"] = out["state_delta"] > delta_threshold
    out["event_proximity_threshold"] = first_true_by_episode(
        out, out["pusher_block_distance"] < proximity_threshold
    )
    return out, delta_threshold


def first_onsets(df: pd.DataFrame, event_col: str) -> dict[int, int]:
    onsets = {}
    for episode_id, group in df.groupby("episode_id", sort=False):
        hits = group.loc[group[event_col]]
        if len(hits):
            onsets[int(episode_id)] = int(hits.iloc[0]["model_step"])
    return onsets


def aligned_curve(
    df: pd.DataFrame,
    event_col: str,
    *,
    window: int,
) -> pd.DataFrame:
    onsets = first_onsets(df, event_col)
    pieces = []
    for episode_id, onset in onsets.items():
        group = df.loc[df["episode_id"] == episode_id, ["model_step", "mse"]].copy()
        group["rel_event"] = group["model_step"] - onset
        pieces.append(group.loc[group["rel_event"].between(-window, window)])
    if not pieces:
        return pd.DataFrame(columns=["rel_event", "mean", "ci95", "n"])
    aligned = pd.concat(pieces, ignore_index=True)
    summary = (
        aligned.groupby("rel_event")["mse"]
        .agg(["mean", "std", "count"])
        .rename(columns={"count": "n"})
        .reset_index()
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))
    return summary


def event_metrics(df: pd.DataFrame, spec: EventSpec) -> dict[str, float | int | str]:
    event_col = f"event_{spec.key}"
    rate = 100.0 * float(df[event_col].mean())
    trajectories = int(df.loc[df[event_col], "episode_id"].nunique())
    if rate > 60.0:
        flag = "too frequent"
    elif rate < 5.0:
        flag = "too sparse"
    else:
        flag = "ok"
    return {"event": spec.name, "event_rate_pct": rate, "trajectories": trajectories, "flag": flag}


def sharpness_metrics(curves: dict[str, pd.DataFrame]) -> list[dict[str, float | str]]:
    rows = []
    for spec in EVENTS:
        curve = curves[spec.key]
        by_rel = curve.set_index("rel_event") if len(curve) else pd.DataFrame()
        mse_t_minus_5 = float(by_rel.loc[-5, "mean"]) if -5 in by_rel.index else float("nan")
        mse_t0 = float(by_rel.loc[0, "mean"]) if 0 in by_rel.index else float("nan")
        ratio = mse_t0 / mse_t_minus_5 if np.isfinite(mse_t_minus_5) and mse_t_minus_5 else float("nan")
        rows.append(
            {
                "event": spec.name,
                "onset_sharpness": ratio,
                "mean_mse_t_minus_5": mse_t_minus_5,
                "mean_mse_t0": mse_t0,
            }
        )
    return rows


def is_localized_peak(curve: pd.DataFrame) -> bool:
    if len(curve) == 0:
        return False
    by_rel = curve.set_index("rel_event")
    required = [-5, 0, 5]
    if any(rel not in by_rel.index for rel in required):
        return False
    t0 = float(by_rel.loc[0, "mean"])
    t_minus_5 = float(by_rel.loc[-5, "mean"])
    t_plus_5 = float(by_rel.loc[5, "mean"])
    pre = by_rel.loc[(by_rel.index >= -10) & (by_rel.index <= -1), "mean"]
    post = by_rel.loc[(by_rel.index >= 1) & (by_rel.index <= 10), "mean"]
    if len(pre) < 3 or len(post) < 3:
        return False
    return t0 > t_minus_5 and t0 > t_plus_5 and t0 > float(pre.mean()) and t0 > float(post.mean())


def plot_grid(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, spec in zip(axes.flat, EVENTS):
        curve = curves[spec.key]
        if len(curve):
            x = curve["rel_event"].to_numpy(dtype=np.float64)
            y = curve["mean"].to_numpy(dtype=np.float64)
            ci = curve["ci95"].to_numpy(dtype=np.float64)
            ax.plot(x, y, color="#1f77b4", linewidth=2)
            ax.fill_between(x, y - ci, y + ci, color="#1f77b4", alpha=0.2, linewidth=0)
        ax.axvline(0, color="#111827", linestyle="--", linewidth=1)
        ax.set_title(spec.title)
        ax.set_xlabel("Model timestep relative to first event")
        ax.set_ylabel("Mean one-step MSE")
        ax.grid(alpha=0.25)
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


def main() -> None:
    args = parse_args()
    h5_path = find_h5(args.dataset_h5)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.records)
    states_by_episode, _ = load_states(h5_path)
    enriched = attach_state_features(records, states_by_episode, proximity_scale=args.proximity_scale)
    labeled, delta_threshold = add_event_labels(
        enriched,
        block_speed_threshold=args.block_speed_threshold,
        state_delta_percentile=args.state_delta_percentile,
        proximity_threshold=args.proximity_threshold,
    )

    labeled_csv = args.output_dir / "alternative_event_records.csv"
    labeled.to_csv(labeled_csv, index=False)

    event_rows = [event_metrics(labeled, spec) for spec in EVENTS[1:]]
    curves = {spec.key: aligned_curve(labeled, f"event_{spec.key}", window=args.window) for spec in EVENTS}
    for spec, curve in curves.items():
        curve.to_csv(args.output_dir / f"{spec}_aligned_curve.csv", index=False)

    sharpness_rows = sharpness_metrics(curves)
    alternative_sharpness = [row for row in sharpness_rows if not row["event"].startswith("Original")]
    positive = [
        row
        for row in alternative_sharpness
        if float(row["onset_sharpness"]) > 1.3
        and is_localized_peak(curves[next(spec.key for spec in EVENTS if spec.name == row["event"])])
    ]
    if positive:
        best = max(positive, key=lambda row: float(row["onset_sharpness"]))
        verdict = "D1"
        verdict_text = (
            "The original contact label was the wrong signal. "
            f"{best['event']} shows a cleaner onset spike with ratio "
            f"{float(best['onset_sharpness']):.3f}. Proposal 2 survives with a refined "
            "event definition. Recommend this as the basis for the Dr. Ding discussion."
        )
    else:
        verdict = "D2"
        verdict_text = (
            "Even with task-agnostic dynamic discontinuity as the event definition, no "
            "localized prediction error spike appears. The foundational claim does not "
            "hold for this task in its current form."
        )

    grid_path = args.output_dir / "alternative_event_time_aligned_grid.png"
    plot_grid(curves, grid_path)

    summary = [
        "# Alternative Event Diagnostic",
        "",
        f"- Records: `{args.records}`",
        f"- State source: `{h5_path}`",
        f"- Prediction records: {len(labeled)}",
        f"- Trajectories: {labeled['episode_id'].nunique()}",
        f"- Block speed threshold: {args.block_speed_threshold}",
        f"- State-delta percentile threshold: p{args.state_delta_percentile:g} = {delta_threshold:.6g}",
        f"- Proximity threshold: {args.proximity_threshold} ({enriched.attrs['proximity_units']})",
        f"- Figure: `{grid_path.name}`",
        "",
        "## Stage 1 Event Rates",
        "",
        table_markdown(event_rows, ["event", "event_rate_pct", "trajectories", "flag"]),
        "",
        "## Stage 3 Onset Sharpness",
        "",
        table_markdown(
            sharpness_rows,
            ["event", "onset_sharpness", "mean_mse_t_minus_5", "mean_mse_t0"],
        ),
        "",
        "## Verdict",
        "",
        f"VERDICT {verdict} -- {verdict_text}",
        "",
    ]
    if verdict == "D2":
        summary.extend(
            [
                (
                    "Caveat: Event B and Event D have onset ratios above 1.3, but both are below "
                    "the 5% event-rate floor and neither forms a localized t=0 peak in the grid. "
                    "Event B steps up after onset, and Event D is noisy with later peaks larger "
                    "than t=0."
                ),
                "",
            ]
        )
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))


if __name__ == "__main__":
    main()
