#!/usr/bin/env python3
"""Stage 1: ensemble bias/variance decomposition for Fetch diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import PRIMARY_REGIMES
from experiment_utils import (
    DEFAULT_DATA_DIR,
    ExampleSet,
    bias_variance_components,
    build_examples,
    find_records,
    load_records,
    masks_from_train_keys,
    predict_values,
    train_mlp,
    trimmed_mean,
)


DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/ensemble_bv")
DEFAULT_STAGE0_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/fullstate_ablation")
COMPONENT_COLS = ["bias2_mse", "variance_mse", "heldout_error_mse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", action="append", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--split-keys", type=Path, default=None)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--member-seeds", default="")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2718)
    parser.add_argument("--min-bias-delta-share", type=float, default=0.65)
    parser.add_argument("--max-variance-delta-share", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[fetch-ensemble-bv] {message}", flush=True)


def parse_member_seeds(raw: str, *, ensemble_size: int, base_seed: int) -> list[int]:
    if raw.strip():
        seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        seeds = [base_seed + i * 9973 for i in range(ensemble_size)]
    if len(seeds) < 5:
        raise ValueError("Stage 1 requires N >= 5 ensemble members")
    if len(seeds) != ensemble_size:
        raise ValueError(f"Expected {ensemble_size} seeds, got {len(seeds)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Ensemble member seeds must be unique")
    return seeds


def load_stage0_metadata(stage0_dir: Path, split_keys_path: Path | None) -> tuple[dict[str, object], dict[str, object]]:
    split_path = split_keys_path or stage0_dir / "split_keys.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split key file: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    decision_path = stage0_dir / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    if decision and not decision.get("stage1_allowed", False):
        raise RuntimeError(f"Stage 0 did not allow Stage 1: {decision}")
    return split, decision


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index, dtype=bool)
    series = df[col]
    if series.dtype == object:
        return series.astype(str).str.lower().isin({"true", "1", "yes"})
    return series.fillna(False).astype(bool)


def finite_val_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    valid = bool_series(df, "is_valid_model_step")
    for col in COMPONENT_COLS:
        valid &= np.isfinite(df[col].astype(float))
    return df.loc[valid].copy()


def component_stats(values: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {"n": int(len(values))}
    for col in COMPONENT_COLS:
        arr = values[col].to_numpy(dtype=np.float64)
        out[f"{col}_mean"] = float(np.mean(arr)) if len(arr) else float("nan")
        out[f"{col}_median"] = float(np.median(arr)) if len(arr) else float("nan")
        out[f"{col}_trimmed_mean"] = trimmed_mean(arr)
    error = out["heldout_error_mse_mean"]
    if np.isfinite(error) and error > 0:
        out["bias_error_share"] = out["bias2_mse_mean"] / error
        out["variance_error_share"] = out["variance_mse_mean"] / error
    else:
        out["bias_error_share"] = float("nan")
        out["variance_error_share"] = float("nan")
    residual = values["decomposition_residual_mse"].to_numpy(dtype=np.float64)
    out["decomposition_residual_abs_max"] = float(np.max(np.abs(residual))) if len(residual) else float("nan")
    return out


def contrast_stats(df: pd.DataFrame, left_mask: pd.Series, right_mask: pd.Series) -> dict[str, float]:
    left = df.loc[left_mask]
    right = df.loc[right_mask]
    left_stats = component_stats(left)
    right_stats = component_stats(right)
    out: dict[str, float] = {
        "left_n": left_stats["n"],
        "right_n": right_stats["n"],
    }
    for col in COMPONENT_COLS:
        for suffix in ["mean", "median", "trimmed_mean"]:
            key = f"{col}_{suffix}"
            out[f"left_{key}"] = left_stats[key]
            out[f"right_{key}"] = right_stats[key]
            out[f"delta_{key}"] = left_stats[key] - right_stats[key]
    delta_error = out["delta_heldout_error_mse_mean"]
    if np.isfinite(delta_error) and abs(delta_error) > 1e-12:
        out["bias_delta_share"] = out["delta_bias2_mse_mean"] / delta_error
        out["variance_delta_share"] = out["delta_variance_mse_mean"] / delta_error
    else:
        out["bias_delta_share"] = float("nan")
        out["variance_delta_share"] = float("nan")
    out["left_bias_error_share"] = left_stats["bias_error_share"]
    out["left_variance_error_share"] = left_stats["variance_error_share"]
    out["right_bias_error_share"] = right_stats["bias_error_share"]
    out["right_variance_error_share"] = right_stats["variance_error_share"]
    return out


def bootstrap_ci(values: list[dict[str, float]], key: str) -> tuple[float, float]:
    arr = np.asarray([row.get(key, float("nan")) for row in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def attach_bootstrap_cis(row: dict[str, object], boot_rows: list[dict[str, float]], keys: list[str]) -> dict[str, object]:
    out = dict(row)
    for key in keys:
        low, high = bootstrap_ci(boot_rows, key)
        out[f"{key}_ci_low"] = low
        out[f"{key}_ci_high"] = high
    return out


def cluster_bootstrap_contrast(
    df: pd.DataFrame,
    *,
    left_mask: pd.Series,
    right_mask: pd.Series,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    point = contrast_stats(df, left_mask, right_mask)
    clusters = [
        group.index.to_numpy()
        for _, group in df.groupby(["env_id", "episode_id"], sort=False)
        if bool(left_mask.loc[group.index].any()) or bool(right_mask.loc[group.index].any())
    ]
    rng = np.random.default_rng(seed)
    boot_rows: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))])
        boot_df = df.loc[idx]
        boot_rows.append(
            contrast_stats(
                boot_df,
                pd.Series(left_mask.loc[idx].to_numpy(dtype=bool), index=boot_df.index),
                pd.Series(right_mask.loc[idx].to_numpy(dtype=bool), index=boot_df.index),
            )
        )
    return point, boot_rows


def cluster_bootstrap_group(
    df: pd.DataFrame,
    *,
    group_mask: pd.Series,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    point = component_stats(df.loc[group_mask])
    clusters = [
        group.index.to_numpy()
        for _, group in df.groupby(["env_id", "episode_id"], sort=False)
        if bool(group_mask.loc[group.index].any())
    ]
    rng = np.random.default_rng(seed)
    boot_rows: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))])
        boot_df = df.loc[idx]
        boot_mask = pd.Series(group_mask.loc[idx].to_numpy(dtype=bool), index=boot_df.index)
        boot_rows.append(component_stats(boot_df.loc[boot_mask]))
    return point, boot_rows


def episode_count(df: pd.DataFrame, mask: pd.Series) -> int:
    return int(df.loc[mask, ["env_id", "episode_id"]].drop_duplicates().shape[0])


def add_ci_rows(ci_rows: list[dict[str, object]], *, table: str, row: dict[str, object], keys: list[str]) -> None:
    for key in keys:
        ci_rows.append(
            {
                "table": table,
                "scope": row.get("scope"),
                "label": row.get("contrast") or row.get("primary_regime"),
                "metric": key,
                "ci_low": row.get(f"{key}_ci_low"),
                "ci_high": row.get(f"{key}_ci_high"),
            }
        )


def analyze_decomposition(
    decomposition: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = finite_val_decomposition(decomposition)
    primary_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []
    scopes = [("all", df)] + [(str(env_id), group.copy()) for env_id, group in df.groupby("env_id", sort=True)]
    contrast_ci_keys = [
        "delta_bias2_mse_mean",
        "delta_variance_mse_mean",
        "delta_heldout_error_mse_mean",
        "bias_delta_share",
        "variance_delta_share",
    ]
    group_ci_keys = [
        "bias2_mse_mean",
        "variance_mse_mean",
        "heldout_error_mse_mean",
        "bias_error_share",
        "variance_error_share",
    ]

    for scope_index, (scope, scope_df) in enumerate(scopes):
        interaction = bool_series(scope_df, "gripper_object_contact")
        free = scope_df["primary_regime"].astype(str) == "free_motion"
        if interaction.any() and free.any():
            point, boot_rows = cluster_bootstrap_contrast(
                scope_df,
                left_mask=interaction,
                right_mask=free,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed + scope_index,
            )
            row = {
                "scope": scope,
                "contrast": "interaction_manipulation_contact_vs_free_motion",
                "left_label": "gripper_object_contact",
                "right_label": "primary_regime=free_motion",
                "left_episode_count": episode_count(scope_df, interaction),
                "right_episode_count": episode_count(scope_df, free),
                **point,
            }
            row = attach_bootstrap_cis(row, boot_rows, contrast_ci_keys)
            primary_rows.append(row)
            add_ci_rows(ci_rows, table="primary_contrast_bv", row=row, keys=contrast_ci_keys)

        for regime in PRIMARY_REGIMES:
            if regime == "boundary_or_invalid":
                continue
            mask = scope_df["primary_regime"].astype(str) == regime
            if not mask.any():
                continue
            point, boot_rows = cluster_bootstrap_group(
                scope_df,
                group_mask=mask,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed + 1000 + scope_index * 100 + len(regime_rows),
            )
            row = {
                "scope": scope,
                "primary_regime": regime,
                "episode_count": episode_count(scope_df, mask),
                **point,
            }
            row = attach_bootstrap_cis(row, boot_rows, group_ci_keys)
            regime_rows.append(row)
            add_ci_rows(ci_rows, table="regime_bv_summary", row=row, keys=group_ci_keys)
    return pd.DataFrame(primary_rows), pd.DataFrame(regime_rows), pd.DataFrame(ci_rows)


def make_decision(
    primary: pd.DataFrame,
    *,
    min_bias_delta_share: float,
    max_variance_delta_share: float,
) -> dict[str, object]:
    all_rows = primary.loc[primary["scope"] == "all"]
    if len(all_rows) == 0:
        return {
            "decision": "no_go_stop",
            "stage2_allowed": False,
            "reason": "Missing all-scope primary bias/variance contrast.",
        }
    row = all_rows.iloc[0]
    delta_error = float(row["delta_heldout_error_mse_mean"])
    delta_bias = float(row["delta_bias2_mse_mean"])
    delta_variance = float(row["delta_variance_mse_mean"])
    bias_share = float(row["bias_delta_share"])
    variance_share = float(row["variance_delta_share"])
    delta_error_low = float(row.get("delta_heldout_error_mse_mean_ci_low", float("nan")))
    delta_bias_low = float(row.get("delta_bias2_mse_mean_ci_low", float("nan")))
    delta_variance_low = float(row.get("delta_variance_mse_mean_ci_low", float("nan")))

    if not np.isfinite(delta_error) or delta_error <= 0.0 or delta_error_low <= 0.0:
        decision = "no_go_stop"
        reason = "The full-state ensemble does not retain a robust positive interaction/free held-out error contrast."
    elif np.isfinite(variance_share) and variance_share >= max_variance_delta_share and delta_variance_low > 0.0:
        decision = "no_go_variance_dominated"
        reason = (
            "The surviving interaction/free error contrast is variance-dominated; pivot toward stochastic or "
            f"multimodal world-model framing instead of adaptive depth. variance_delta_share={variance_share:.3g}."
        )
    elif np.isfinite(bias_share) and bias_share >= min_bias_delta_share and delta_bias_low > 0.0:
        decision = "go_stage2_recurrent_refiner"
        reason = (
            "The surviving interaction/free error contrast is bias-dominated after full-state input; "
            f"bias_delta_share={bias_share:.3g} >= {min_bias_delta_share:.3g}."
        )
    else:
        decision = "no_go_inconclusive"
        reason = (
            "The interaction/free contrast survives, but the bias/variance split is not clean enough to justify "
            "Stage 2 as an adaptive-depth test."
        )
    return {
        "decision": decision,
        "stage2_allowed": bool(decision.startswith("go_")),
        "reason": reason,
        "delta_heldout_error_mse_mean": delta_error,
        "delta_bias2_mse_mean": delta_bias,
        "delta_variance_mse_mean": delta_variance,
        "bias_delta_share": bias_share,
        "variance_delta_share": variance_share,
        "delta_heldout_error_mse_mean_ci": [
            delta_error_low,
            float(row.get("delta_heldout_error_mse_mean_ci_high", float("nan"))),
        ],
        "delta_bias2_mse_mean_ci": [
            delta_bias_low,
            float(row.get("delta_bias2_mse_mean_ci_high", float("nan"))),
        ],
        "delta_variance_mse_mean_ci": [
            delta_variance_low,
            float(row.get("delta_variance_mse_mean_ci_high", float("nan"))),
        ],
        "left_bias_error_share": float(row.get("left_bias_error_share", float("nan"))),
        "left_variance_error_share": float(row.get("left_variance_error_share", float("nan"))),
        "right_bias_error_share": float(row.get("right_bias_error_share", float("nan"))),
        "right_variance_error_share": float(row.get("right_variance_error_share", float("nan"))),
        "min_bias_delta_share": min_bias_delta_share,
        "max_variance_delta_share": max_variance_delta_share,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_summary(
    path: Path,
    *,
    decision: dict[str, object],
    primary: pd.DataFrame,
    regimes: pd.DataFrame,
    args: argparse.Namespace,
    member_seeds: list[int],
) -> None:
    primary_cols = [
        "scope",
        "left_n",
        "right_n",
        "left_bias2_mse_mean",
        "left_variance_mse_mean",
        "left_heldout_error_mse_mean",
        "right_bias2_mse_mean",
        "right_variance_mse_mean",
        "right_heldout_error_mse_mean",
        "delta_bias2_mse_mean",
        "delta_bias2_mse_mean_ci_low",
        "delta_bias2_mse_mean_ci_high",
        "delta_variance_mse_mean",
        "delta_variance_mse_mean_ci_low",
        "delta_variance_mse_mean_ci_high",
        "delta_heldout_error_mse_mean",
        "delta_heldout_error_mse_mean_ci_low",
        "delta_heldout_error_mse_mean_ci_high",
        "bias_delta_share",
        "variance_delta_share",
    ]
    regime_cols = [
        "scope",
        "primary_regime",
        "n",
        "bias2_mse_mean",
        "bias2_mse_mean_ci_low",
        "bias2_mse_mean_ci_high",
        "variance_mse_mean",
        "variance_mse_mean_ci_low",
        "variance_mse_mean_ci_high",
        "heldout_error_mse_mean",
        "heldout_error_mse_mean_ci_low",
        "heldout_error_mse_mean_ci_high",
        "bias_error_share",
        "variance_error_share",
    ]
    lines = [
        "# Fetch Stage 1 Ensemble Bias/Variance",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Stage 2 allowed: `{decision['stage2_allowed']}`",
        f"- Reason: {decision['reason']}",
        f"- Ensemble members: `{args.ensemble_size}`",
        f"- Member seeds: `{member_seeds}`",
        f"- Bootstrap samples: `{args.bootstrap_samples}` clustered over episodes",
        f"- Features: leak-safe shifted full-state input from Stage 0",
        "",
        "Bias^2 is the ensemble-mean squared error. Variance is member disagreement around the ensemble mean.",
        "With one observed next state per input, irreducible environment noise is not separately identifiable; the reported held-out error is the finite-ensemble mean member error.",
        "",
        "## Primary Interaction vs Free Motion",
        "",
        markdown_table(primary.loc[primary["scope"] == "all", [col for col in primary_cols if col in primary]]),
        "",
        "## Secondary MECE Regimes",
        "",
        markdown_table(regimes.loc[regimes["scope"] == "all", [col for col in regime_cols if col in regimes]]),
        "",
        "This remains a diagnostic pilot and is only a gate for Stage 2.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_member_checkpoint(
    *,
    member_dir: Path,
    model: object,
    normalizers: dict[str, np.ndarray],
    examples: ExampleSet,
    seed: int,
    args: argparse.Namespace,
    torch: object,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "normalizers": normalizers,
        "feature_cols": examples.feature_cols,
        "target_cols": examples.target_cols,
        "state_cols": examples.state_cols,
        "action_cols": examples.action_cols,
        "fullstate_cols": examples.fullstate_cols,
        "history_size": args.history_size,
        "hidden_dim": args.hidden_dim,
        "seed": seed,
        "created_at_unix": time.time(),
    }
    torch.save(checkpoint, member_dir / "forward_probe.pt")


def train_member(
    *,
    member_index: int,
    seed: int,
    examples: ExampleSet,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> np.ndarray:
    member_name = f"member_{member_index:02d}_seed_{seed}"
    member_dir = output_dir / member_name
    member_dir.mkdir(parents=True, exist_ok=True)
    print_step(f"{member_name}: train={int(train_mask.sum())} val={int(val_mask.sum())}")
    model, normalizers, history, device, torch = train_mlp(
        x=examples.x,
        y=examples.y,
        train_mask=train_mask,
        val_mask=val_mask,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=seed,
        device_name=args.device,
        log_fn=lambda msg: print_step(f"{member_name}: {msg}"),
    )
    predictions = predict_values(
        model=model,
        normalizers=normalizers,
        x=examples.x,
        device=device,
        torch=torch,
    ).astype(np.float32)
    member_mse = np.mean((predictions - examples.y) ** 2, axis=1)
    member_output = pd.concat(
        [
            examples.meta.reset_index(drop=True),
            pd.DataFrame(
                {
                    "member_index": member_index,
                    "seed": seed,
                    "split": np.where(train_mask, "train", "val"),
                    "mse": member_mse,
                }
            ),
        ],
        axis=1,
    )
    member_output.to_csv(member_dir / "predictions.csv.gz", index=False, compression="gzip")
    history.to_csv(member_dir / "training_history.csv", index=False)
    write_member_checkpoint(
        member_dir=member_dir,
        model=model,
        normalizers=normalizers,
        examples=examples,
        seed=seed,
        args=args,
        torch=torch,
    )
    return predictions[val_mask]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and any(args.output_dir.glob("*")):
        raise FileExistsError(f"{args.output_dir} already has outputs; pass --force to overwrite")

    member_seeds = parse_member_seeds(args.member_seeds, ensemble_size=args.ensemble_size, base_seed=args.base_seed)
    split_payload, stage0_decision = load_stage0_metadata(args.stage0_dir, args.split_keys)
    train_keys = split_payload.get("train_keys")
    if not train_keys:
        raise ValueError("Stage 1 requires train_keys from the Stage 0 split file")

    records = find_records(args.data_dir, args.records)
    print_step(f"loading {len(records)} record file(s)")
    df = load_records(records, history_size=args.history_size)
    examples = build_examples(df, history_size=args.history_size, include_shifted_fullstate=True)
    train_mask, val_mask = masks_from_train_keys(examples.meta, train_keys)
    print_step(f"examples={len(examples.x)} train={int(train_mask.sum())} val={int(val_mask.sum())}")

    run_config = {
        "records": [str(path) for path in records],
        "stage0_dir": str(args.stage0_dir),
        "stage0_decision": stage0_decision,
        "split_keys": str(args.split_keys or args.stage0_dir / "split_keys.json"),
        "history_size": args.history_size,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "ensemble_size": args.ensemble_size,
        "member_seeds": member_seeds,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "min_bias_delta_share": args.min_bias_delta_share,
        "max_variance_delta_share": args.max_variance_delta_share,
        "feature_source": "shifted previous-row qpos/qvel plus observation/action/history",
        "created_at_unix": time.time(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_count": int(examples.x.shape[1]),
                "target_count": int(examples.y.shape[1]),
                "feature_cols": examples.feature_cols,
                "target_cols": examples.target_cols,
                "fullstate_cols": examples.fullstate_cols,
                "fullstate_source": "shifted previous-row qpos/qvel; same-row qpos/qvel are excluded as post-step leakage",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    val_predictions = []
    for member_index, seed in enumerate(member_seeds):
        val_predictions.append(
            train_member(
                member_index=member_index,
                seed=seed,
                examples=examples,
                train_mask=train_mask,
                val_mask=val_mask,
                args=args,
                output_dir=args.output_dir,
            )
        )
    prediction_array = np.stack(val_predictions, axis=0).astype(np.float32)
    val_targets = examples.y[val_mask].astype(np.float32)
    np.savez_compressed(
        args.output_dir / "member_predictions_val.npz",
        predictions=prediction_array,
        targets=val_targets,
        member_seeds=np.asarray(member_seeds, dtype=np.int64),
    )

    components = bias_variance_components(prediction_array, val_targets)
    decomposition = pd.concat([examples.meta.loc[val_mask].reset_index(drop=True), components], axis=1)
    decomposition.to_csv(args.output_dir / "val_decomposition.csv.gz", index=False, compression="gzip")
    primary, regimes, ci = analyze_decomposition(
        decomposition,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    primary.to_csv(args.output_dir / "primary_contrast_bv.csv", index=False)
    regimes.to_csv(args.output_dir / "regime_bv_summary.csv", index=False)
    ci.to_csv(args.output_dir / "bootstrap_cis.csv", index=False)
    decision = make_decision(
        primary,
        min_bias_delta_share=args.min_bias_delta_share,
        max_variance_delta_share=args.max_variance_delta_share,
    )
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_summary(
        args.output_dir / "summary.md",
        decision=decision,
        primary=primary,
        regimes=regimes,
        args=args,
        member_seeds=member_seeds,
    )
    print_step(f"decision={decision['decision']} stage2_allowed={decision['stage2_allowed']}")
    print_step(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
