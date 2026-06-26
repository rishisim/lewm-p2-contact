#!/usr/bin/env python3
"""Analyze whether Fetch forward-model error concentrates at interaction contact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path("le-wm/diagnostics/fetch_contact_compute/runs/forward_error_probe/predictions.csv.gz")
DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/interaction_error_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val", choices=("train", "val", "all"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[fetch-interaction] {message}", flush=True)


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].fillna(False).astype(bool)


def filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all" or "split" not in df:
        return df.copy()
    out = df.loc[df["split"] == split].copy()
    if len(out) == 0:
        print_step(f"split={split!r} is empty; falling back to all predictions")
        return df.copy()
    return out


def ci95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def add_error_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    interaction = bool_series(out, "gripper_object_contact")
    support = bool_series(out, "object_support_contact")
    out["interaction_contact"] = interaction
    out["support_only_contact"] = support & ~interaction
    out["no_task_contact"] = ~support & ~interaction
    out["non_interaction"] = ~interaction
    out["error_group"] = np.select(
        [out["interaction_contact"], out["support_only_contact"], out["no_task_contact"]],
        ["interaction_contact", "support_only_contact", "no_task_contact"],
        default="other",
    )
    return out


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env_label, env_df in [("all", df), *list(df.groupby("env_id", sort=True))]:
        for group_name in ["interaction_contact", "support_only_contact", "no_task_contact"]:
            group = env_df.loc[env_df["error_group"] == group_name]
            if len(group) == 0:
                continue
            clustered = group.groupby(["env_id", "episode_id"])["mse"].median().to_numpy(dtype=np.float64)
            rows.append(
                {
                    "env_id": env_label,
                    "error_group": group_name,
                    "row_count": int(len(group)),
                    "episode_count": int(group[["env_id", "episode_id"]].drop_duplicates().shape[0]),
                    "mse_mean": float(group["mse"].mean()),
                    "mse_median": float(group["mse"].median()),
                    "episode_median_ci95": ci95(clustered),
                }
            )
    return pd.DataFrame(rows)


def wilcoxon_pvalue(diffs: np.ndarray, alternative: str) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
    if len(diffs) == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(diffs, alternative=alternative, zero_method="wilcox", method="auto")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        abs_diff = np.abs(diffs)
        order = np.argsort(abs_diff)
        ranks = np.empty_like(abs_diff)
        sorted_abs = abs_diff[order]
        start = 0
        while start < len(sorted_abs):
            end = start + 1
            while end < len(sorted_abs) and sorted_abs[end] == sorted_abs[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        w_plus = float(ranks[diffs > 0].sum())
        n = len(diffs)
        mean = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        z = (w_plus - mean) / math.sqrt(var)
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        if alternative == "greater":
            pvalue = 1.0 - cdf
        elif alternative == "less":
            pvalue = cdf
        else:
            pvalue = 2.0 * min(cdf, 1.0 - cdf)
        return w_plus, float(max(0.0, min(1.0, pvalue)))


def episode_contrast(
    df: pd.DataFrame,
    *,
    contrast: str,
    left_label: str,
    right_label: str,
    left_mask: pd.Series,
    right_mask: pd.Series,
    alternative: str,
) -> pd.DataFrame:
    rows = []
    scopes = [("all", df)] + list(df.groupby("env_id", sort=True))
    for env_label, env_df in scopes:
        pairs = []
        env_left = left_mask.loc[env_df.index]
        env_right = right_mask.loc[env_df.index]
        for key, group in env_df.groupby(["env_id", "episode_id"], sort=False):
            group_left = group.loc[env_left.loc[group.index], "mse"]
            group_right = group.loc[env_right.loc[group.index], "mse"]
            if len(group_left) == 0 or len(group_right) == 0:
                continue
            left_median = float(group_left.median())
            right_median = float(group_right.median())
            ratio = left_median / right_median if right_median > 0 else float("nan")
            pairs.append(
                {
                    "env_id": key[0],
                    "episode_id": key[1],
                    "left_median": left_median,
                    "right_median": right_median,
                    "diff": left_median - right_median,
                    "ratio": ratio,
                }
            )
        pair_df = pd.DataFrame(pairs)
        if len(pair_df) == 0:
            continue
        statistic, pvalue = wilcoxon_pvalue(pair_df["diff"].to_numpy(dtype=np.float64), alternative=alternative)
        rows.append(
            {
                "scope": env_label,
                "contrast": contrast,
                "left_label": left_label,
                "right_label": right_label,
                "episode_count": int(len(pair_df)),
                "left_higher_episode_rate": float((pair_df["diff"] > 0).mean()),
                "median_left_mse": float(pair_df["left_median"].median()),
                "median_right_mse": float(pair_df["right_median"].median()),
                "median_ratio": float(pair_df["ratio"].replace([np.inf, -np.inf], np.nan).median()),
                "median_diff": float(pair_df["diff"].median()),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": pvalue,
                "alternative": alternative,
            }
        )
    return pd.DataFrame(rows)


def build_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    interaction = df["interaction_contact"].astype(bool)
    support_only = df["support_only_contact"].astype(bool)
    no_task = df["no_task_contact"].astype(bool)
    contrasts = [
        episode_contrast(
            df,
            contrast="interaction_vs_non_interaction",
            left_label="interaction_contact",
            right_label="non_interaction",
            left_mask=interaction,
            right_mask=~interaction,
            alternative="greater",
        ),
        episode_contrast(
            df,
            contrast="support_only_vs_no_task_contact",
            left_label="support_only_contact",
            right_label="no_task_contact",
            left_mask=support_only,
            right_mask=no_task,
            alternative="less",
        ),
    ]
    return pd.concat([frame for frame in contrasts if len(frame)], ignore_index=True)


def plot_group_summary(summary: pd.DataFrame, path: Path) -> None:
    data = summary.loc[summary["env_id"] != "all"].copy()
    if len(data) == 0:
        data = summary.copy()
    pivot = data.pivot(index="env_id", columns="error_group", values="mse_median")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot.plot(kind="bar", ax=ax, color=["#1d4ed8", "#64748b", "#16a34a"])
    ax.set_xlabel("")
    ax.set_ylabel("Held-out median one-step MSE")
    ax.set_title("Prediction error by interaction-contact group")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_contrast_ratios(contrasts: pd.DataFrame, path: Path) -> None:
    data = contrasts.loc[contrasts["contrast"] == "interaction_vs_non_interaction"].copy()
    data = data.loc[data["scope"] != "all"]
    if len(data) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(data["scope"], data["median_ratio"], color="#1d4ed8")
    ax.axhline(1.0, color="#111827", linewidth=1)
    ax.set_ylabel("Within-episode median error ratio")
    ax.set_title("Interaction contact vs non-interaction")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def result_payload(summary: pd.DataFrame, contrasts: pd.DataFrame) -> dict[str, object]:
    interaction = contrasts.loc[contrasts["contrast"] == "interaction_vs_non_interaction"]
    all_interaction = interaction.loc[interaction["scope"] == "all"]
    env_interaction = interaction.loc[interaction["scope"] != "all"]
    supported = False
    if len(all_interaction) and len(env_interaction):
        supported = bool(
            all_interaction["median_ratio"].iloc[0] > 1.0
            and all_interaction["left_higher_episode_rate"].iloc[0] > 0.5
            and all_interaction["wilcoxon_p"].iloc[0] < 0.05
            and (env_interaction["median_ratio"] > 1.0).all()
        )

    support = contrasts.loc[contrasts["contrast"] == "support_only_vs_no_task_contact"]
    all_support = support.loc[support["scope"] == "all"]
    support_opposite = bool(len(all_support) and all_support["median_ratio"].iloc[0] < 1.0)
    return {
        "status": "descriptive_pass" if supported else "inconclusive",
        "interaction_error_supported": supported,
        "support_only_contact_lower_than_no_task_contact": support_opposite,
        "group_summary_rows": int(len(summary)),
        "contrast_rows": int(len(contrasts)),
    }


def write_summary(path: Path, result: dict[str, object], summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    lines = [
        "# Fetch Interaction-Error Analysis",
        "",
        f"- Status: `{result['status']}`",
        f"- Interaction-contact error elevation supported: `{result['interaction_error_supported']}`",
        f"- Support-only contact lower than no-task contact: `{result['support_only_contact_lower_than_no_task_contact']}`",
        "",
        "This analysis supports only the narrow descriptive claim: held-out",
        "one-step forward-model error concentrates at active gripper-object",
        "interaction/manipulation contact steps. It does not evaluate an adaptive",
        "compute mechanism.",
        "",
        "## Group Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Within-Episode Contrasts",
        "",
        contrasts.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and any(args.output_dir.glob("*")):
        raise FileExistsError(f"{args.output_dir} already has outputs; pass --force to overwrite")
    df = pd.read_csv(args.predictions)
    df = filter_split(df, args.split)
    if "mse" not in df:
        raise ValueError(f"{args.predictions} must contain an mse column from train_forward_error_probe.py")
    valid = bool_series(df, "is_valid_model_step") & np.isfinite(df["mse"].astype(float))
    df = add_error_groups(df.loc[valid].copy())
    summary = group_summary(df)
    contrasts = build_contrasts(df)
    result = result_payload(summary, contrasts)

    summary.to_csv(args.output_dir / "interaction_group_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "within_episode_contrasts.csv", index=False)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    plot_group_summary(summary, args.output_dir / "interaction_error_by_group.png")
    plot_contrast_ratios(contrasts, args.output_dir / "interaction_ratio_by_env.png")
    write_summary(args.output_dir / "summary.md", result, summary, contrasts)
    print_step(f"status={result['status']} wrote {args.output_dir}")


if __name__ == "__main__":
    main()
