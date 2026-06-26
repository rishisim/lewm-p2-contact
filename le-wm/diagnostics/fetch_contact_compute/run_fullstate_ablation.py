#!/usr/bin/env python3
"""Stage 0: full-state ablation for Fetch interaction-error diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
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
    attach_ci,
    build_examples,
    cluster_bootstrap_contrast,
    cluster_bootstrap_group,
    find_records,
    load_records,
    masks_from_train_keys,
    split_by_episode,
    train_mlp,
    predict_mse,
)


DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/fullstate_ablation")
VARIANTS = [
    ("baseline_observation", False),
    ("full_state_shifted", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", action="append", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--min-residual-ratio", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[fetch-fullstate] {message}", flush=True)


def example_keys(meta: pd.DataFrame) -> pd.Series:
    return meta[["env_id", "episode_id", "step_idx"]].astype(str).agg("::".join, axis=1)


def subset_examples(examples: ExampleSet, keys: set[str], ordered_keys: list[str]) -> ExampleSet:
    meta = examples.meta.copy()
    meta["_example_key"] = example_keys(meta)
    key_to_pos = {key: pos for pos, key in enumerate(meta["_example_key"].tolist()) if key in keys}
    positions = np.asarray([key_to_pos[key] for key in ordered_keys], dtype=np.int64)
    out_meta = meta.iloc[positions].drop(columns=["_example_key"]).reset_index(drop=True)
    return replace(
        examples,
        x=examples.x[positions],
        y=examples.y[positions],
        meta=out_meta,
    )


def align_example_sets(example_sets: dict[str, ExampleSet]) -> dict[str, ExampleSet]:
    key_lists = {name: example_keys(examples.meta).tolist() for name, examples in example_sets.items()}
    common = set.intersection(*(set(keys) for keys in key_lists.values()))
    if not common:
        raise ValueError("No common examples across ablation variants")
    reference_name = next(iter(example_sets))
    ordered = [key for key in key_lists[reference_name] if key in common]
    aligned = {name: subset_examples(examples, common, ordered) for name, examples in example_sets.items()}
    lengths = {name: len(examples.x) for name, examples in aligned.items()}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"Aligned example counts differ: {lengths}")
    return aligned


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index, dtype=bool)
    series = df[col]
    if series.dtype == object:
        return series.astype(str).str.lower().isin({"true", "1", "yes"})
    return series.fillna(False).astype(bool)


def finite_val_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.loc[predictions["split"] == "val"].copy()
    valid = bool_series(out, "is_valid_model_step") & np.isfinite(out["mse"].astype(float))
    return out.loc[valid].copy()


def episode_count(df: pd.DataFrame, mask: pd.Series) -> int:
    return int(df.loc[mask, ["env_id", "episode_id"]].drop_duplicates().shape[0])


def add_ci_rows(
    ci_rows: list[dict[str, object]],
    *,
    table: str,
    row: dict[str, object],
    metrics: list[str],
) -> None:
    for metric in metrics:
        ci_rows.append(
            {
                "table": table,
                "variant": row.get("variant"),
                "scope": row.get("scope"),
                "label": row.get("contrast") or row.get("primary_regime"),
                "metric": metric,
                "ci_low": row.get(f"{metric}_ci_low"),
                "ci_high": row.get(f"{metric}_ci_high"),
            }
        )


def analyze_predictions(
    predictions: pd.DataFrame,
    *,
    variant: str,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    df = finite_val_predictions(predictions)
    primary_rows: list[dict[str, object]] = []
    mece_rows: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []
    scopes = [("all", df)] + [(str(env_id), group.copy()) for env_id, group in df.groupby("env_id", sort=True)]

    contrast_metrics = [
        "delta_mean",
        "delta_median",
        "delta_trimmed_mean",
        "p_superiority",
        "cliffs_delta",
    ]
    group_metrics = ["left_mean", "left_median", "left_trimmed_mean"]

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
                "variant": variant,
                "scope": scope,
                "contrast": "interaction_manipulation_contact_vs_free_motion",
                "left_label": "gripper_object_contact",
                "right_label": "primary_regime=free_motion",
                "left_episode_count": episode_count(scope_df, interaction),
                "right_episode_count": episode_count(scope_df, free),
                **point,
            }
            row = attach_ci(row, boot_rows, contrast_metrics)
            primary_rows.append(row)
            add_ci_rows(ci_rows, table="primary_contrast_metrics", row=row, metrics=contrast_metrics)

        free_mask = scope_df["primary_regime"].astype(str) == "free_motion"
        for regime in PRIMARY_REGIMES:
            if regime == "boundary_or_invalid":
                continue
            regime_mask = scope_df["primary_regime"].astype(str) == regime
            if not regime_mask.any():
                continue
            point, group_boot = cluster_bootstrap_group(
                scope_df,
                group_mask=regime_mask,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed + 1000 + scope_index * 100 + len(mece_rows),
            )
            row = {
                "variant": variant,
                "scope": scope,
                "primary_regime": regime,
                "episode_count": episode_count(scope_df, regime_mask),
                **point,
            }
            row = attach_ci(row, group_boot, group_metrics)
            if regime != "free_motion" and free_mask.any():
                contrast, contrast_boot = cluster_bootstrap_contrast(
                    scope_df,
                    left_mask=regime_mask,
                    right_mask=free_mask,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed + 2000 + scope_index * 100 + len(mece_rows),
                )
                for key, value in contrast.items():
                    if key.startswith("right_") or key.startswith("delta_") or key in {"p_superiority", "cliffs_delta"}:
                        row[key] = value
                row = attach_ci(row, contrast_boot, contrast_metrics)
            mece_rows.append(row)
            add_ci_rows(ci_rows, table="mece_regime_metrics", row=row, metrics=group_metrics)
            if regime != "free_motion":
                add_ci_rows(ci_rows, table="mece_regime_metrics", row=row, metrics=contrast_metrics)
    return primary_rows, mece_rows, ci_rows


def train_variant(
    *,
    name: str,
    examples: ExampleSet,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
):
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    print_step(f"{name}: examples={len(examples.x)} train={int(train_mask.sum())} val={int(val_mask.sum())}")
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
        seed=args.seed,
        device_name=args.device,
        log_fn=lambda msg: print_step(f"{name}: {msg}"),
    )
    pred_df = predict_mse(
        model=model,
        normalizers=normalizers,
        x=examples.x,
        y=examples.y,
        device=device,
        torch=torch,
    )
    pred_df["split"] = np.where(train_mask, "train", "val")
    pred_df["variant"] = name
    output = pd.concat([examples.meta.reset_index(drop=True), pred_df], axis=1)
    output.to_csv(variant_dir / "predictions.csv.gz", index=False, compression="gzip")
    history.to_csv(variant_dir / "training_history.csv", index=False)
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
        "variant": name,
        "created_at_unix": time.time(),
    }
    torch.save(checkpoint, variant_dir / "forward_probe.pt")
    (variant_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "variant": name,
                "feature_count": int(examples.x.shape[1]),
                "target_count": int(examples.y.shape[1]),
                "feature_cols": examples.feature_cols,
                "target_cols": examples.target_cols,
                "fullstate_cols": examples.fullstate_cols,
                "fullstate_source": (
                    "shifted previous-row qpos/qvel, because same-row recorded qpos/qvel are post-step target-time"
                    if examples.fullstate_cols
                    else "none"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, history


def make_decision(primary: pd.DataFrame, *, min_residual_ratio: float) -> dict[str, object]:
    all_rows = primary.loc[primary["scope"] == "all"].set_index("variant")
    required = {"baseline_observation", "full_state_shifted"}
    if not required <= set(all_rows.index):
        return {
            "decision": "no_go_stop",
            "reason": "Missing all-scope primary contrast rows for one or more variants.",
            "stage1_allowed": False,
        }

    baseline = all_rows.loc["baseline_observation"]
    full = all_rows.loc["full_state_shifted"]
    baseline_delta = float(baseline["delta_trimmed_mean"])
    full_delta = float(full["delta_trimmed_mean"])
    residual_ratio = full_delta / baseline_delta if baseline_delta > 0 else float("nan")
    baseline_supported = bool(
        baseline_delta > 0.0
        and float(baseline.get("delta_trimmed_mean_ci_low", float("nan"))) > 0.0
        and float(baseline.get("cliffs_delta_ci_low", float("nan"))) > 0.0
    )
    full_survives = bool(
        full_delta > 0.0
        and float(full.get("delta_trimmed_mean_ci_low", float("nan"))) > 0.0
        and float(full.get("cliffs_delta_ci_low", float("nan"))) > 0.0
        and np.isfinite(residual_ratio)
        and residual_ratio >= min_residual_ratio
    )
    if baseline_supported and full_survives:
        decision = "go_stage1_ensemble_bv"
        reason = (
            "The interaction/free error spike survives leak-safe full-state input with positive clustered CIs "
            f"and residual_ratio={residual_ratio:.3g} >= {min_residual_ratio:.3g}."
        )
    elif not baseline_supported:
        decision = "no_go_stop"
        reason = "The baseline interaction/free spike is not robust under the configured clustered CI rule."
    else:
        decision = "no_go_stop"
        reason = (
            "The interaction/free spike largely collapses or becomes non-robust under leak-safe full-state input; "
            "treat this as primarily an observability/state-information result for now."
        )
    return {
        "decision": decision,
        "reason": reason,
        "stage1_allowed": bool(decision.startswith("go_")),
        "baseline_delta_trimmed_mean": baseline_delta,
        "full_state_delta_trimmed_mean": full_delta,
        "residual_ratio": residual_ratio,
        "min_residual_ratio": min_residual_ratio,
        "baseline_delta_trimmed_mean_ci": [
            float(baseline.get("delta_trimmed_mean_ci_low", float("nan"))),
            float(baseline.get("delta_trimmed_mean_ci_high", float("nan"))),
        ],
        "full_state_delta_trimmed_mean_ci": [
            float(full.get("delta_trimmed_mean_ci_low", float("nan"))),
            float(full.get("delta_trimmed_mean_ci_high", float("nan"))),
        ],
        "baseline_cliffs_delta": float(baseline.get("cliffs_delta", float("nan"))),
        "full_state_cliffs_delta": float(full.get("cliffs_delta", float("nan"))),
        "baseline_p_superiority": float(baseline.get("p_superiority", float("nan"))),
        "full_state_p_superiority": float(full.get("p_superiority", float("nan"))),
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
    mece: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    primary_cols = [
        "variant",
        "scope",
        "left_n",
        "right_n",
        "left_trimmed_mean",
        "right_trimmed_mean",
        "delta_trimmed_mean",
        "delta_trimmed_mean_ci_low",
        "delta_trimmed_mean_ci_high",
        "p_superiority",
        "cliffs_delta",
    ]
    mece_cols = [
        "variant",
        "scope",
        "primary_regime",
        "left_n",
        "left_trimmed_mean",
        "left_trimmed_mean_ci_low",
        "left_trimmed_mean_ci_high",
        "delta_trimmed_mean",
        "delta_trimmed_mean_ci_low",
        "delta_trimmed_mean_ci_high",
        "cliffs_delta",
    ]
    all_primary = primary.loc[primary["scope"] == "all", [col for col in primary_cols if col in primary]]
    all_mece = mece.loc[mece["scope"] == "all", [col for col in mece_cols if col in mece]]
    lines = [
        "# Fetch Stage 0 Full-State Ablation",
        "",
        f"- Decision: `{decision['decision']}`",
        f"- Stage 1 allowed: `{decision['stage1_allowed']}`",
        f"- Reason: {decision['reason']}",
        f"- Bootstrap samples: `{args.bootstrap_samples}` clustered over episodes",
        f"- Full-state features: leak-safe shifted previous-row `qpos_*`/`qvel_*` input features",
        "",
        "Same-row recorded qpos/qvel were not used as features because the collector writes them after `env.step(action)`,",
        "which makes them target-time state for the current prediction row.",
        "",
        "## Primary Interaction vs Free Motion",
        "",
        markdown_table(all_primary),
        "",
        "## Secondary MECE Regimes",
        "",
        markdown_table(all_mece),
        "",
        "This is a diagnostic pilot. The decision is only a gate for whether to spend Stage 1/2 effort.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and any(args.output_dir.glob("*")):
        raise FileExistsError(f"{args.output_dir} already has outputs; pass --force to overwrite")

    records = find_records(args.data_dir, args.records)
    print_step(f"loading {len(records)} record file(s)")
    df = load_records(records, history_size=args.history_size)
    example_sets = {
        name: build_examples(df, history_size=args.history_size, include_shifted_fullstate=include_fullstate)
        for name, include_fullstate in VARIANTS
    }
    example_sets = align_example_sets(example_sets)
    train_mask, val_mask, train_keys, val_keys = split_by_episode(
        example_sets["baseline_observation"].meta,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    split_payload = {
        "seed": args.seed,
        "train_frac": args.train_frac,
        "train_episode_count": len(train_keys),
        "val_episode_count": len(val_keys),
        "train_keys": train_keys,
        "val_keys": val_keys,
        "aligned_examples": int(len(example_sets["baseline_observation"].x)),
    }
    (args.output_dir / "split_keys.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    all_primary: list[dict[str, object]] = []
    all_mece: list[dict[str, object]] = []
    all_ci: list[dict[str, object]] = []
    run_config = {
        "records": [str(path) for path in records],
        "history_size": args.history_size,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "min_residual_ratio": args.min_residual_ratio,
        "variants": [name for name, _ in VARIANTS],
        "fullstate_feature_source": "shifted previous-row qpos/qvel; same-row qpos/qvel are post-step and excluded as leakage",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    for variant_index, (name, _) in enumerate(VARIANTS):
        examples = example_sets[name]
        variant_train_mask, variant_val_mask = masks_from_train_keys(examples.meta, train_keys)
        if not np.array_equal(train_mask, variant_train_mask) or not np.array_equal(val_mask, variant_val_mask):
            raise AssertionError(f"Split masks differ after alignment for {name}")
        predictions, _ = train_variant(
            name=name,
            examples=examples,
            train_mask=variant_train_mask,
            val_mask=variant_val_mask,
            args=args,
            output_dir=args.output_dir,
        )
        primary_rows, mece_rows, ci_rows = analyze_predictions(
            predictions,
            variant=name,
            n_bootstrap=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + variant_index * 10000,
        )
        all_primary.extend(primary_rows)
        all_mece.extend(mece_rows)
        all_ci.extend(ci_rows)

    primary = pd.DataFrame(all_primary)
    mece = pd.DataFrame(all_mece)
    ci = pd.DataFrame(all_ci)
    primary.to_csv(args.output_dir / "primary_contrast_metrics.csv", index=False)
    mece.to_csv(args.output_dir / "mece_regime_metrics.csv", index=False)
    ci.to_csv(args.output_dir / "bootstrap_cis.csv", index=False)
    decision = make_decision(primary, min_residual_ratio=args.min_residual_ratio)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_summary(args.output_dir / "summary.md", decision=decision, primary=primary, mece=mece, args=args)
    print_step(f"decision={decision['decision']} stage1_allowed={decision['stage1_allowed']}")
    print_step(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
