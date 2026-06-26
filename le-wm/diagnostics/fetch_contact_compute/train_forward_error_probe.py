#!/usr/bin/env python3
"""Train a single-depth one-step forward predictor on Fetch pilot data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import assign_event_taxonomy


DEFAULT_DATA_DIR = Path("le-wm/diagnostics/fetch_contact_compute/data")
DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/fetch_contact_compute/runs/forward_error_probe")


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
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[fetch-forward] {message}", flush=True)


def import_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def choose_device(torch, requested: str):
    if requested == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def numeric_suffix_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in df.columns if col.startswith(prefix) and col.rsplit("_", 1)[1].isdigit()]
    return sorted(cols, key=lambda name: int(name.rsplit("_", 1)[1]))


def find_records(args: argparse.Namespace) -> list[Path]:
    if args.records:
        return args.records
    records = sorted(args.data_dir.glob("*/records.csv.gz"))
    if not records:
        raise FileNotFoundError(f"No records.csv.gz files found under {args.data_dir}; run collection first")
    return records


def load_records(paths: list[Path], history_size: int) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "primary_regime" not in df:
            df, _ = assign_event_taxonomy(df, history_size=history_size)
        df["source_records"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["env_id", "episode_id", "step_idx"]).reset_index(drop=True)


def build_examples(
    df: pd.DataFrame,
    history_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str], list[str]]:
    state_cols = numeric_suffix_cols(df, "state_")
    action_cols = numeric_suffix_cols(df, "action_")
    next_state_cols = numeric_suffix_cols(df, "next_state_")
    if not state_cols or not action_cols or not next_state_cols:
        raise ValueError("Records must contain state_*, action_*, and next_state_* columns")

    metadata_cols = [
        "env_id",
        "episode_id",
        "step_idx",
        "primary_regime",
        "is_valid_model_step",
        "gripper_object_contact",
        "object_support_contact",
        "task_contact_count",
        "gripper_object_contact_onset",
        "object_support_contact_onset",
        "object_table_impact_onset",
        "impact_onset",
        "high_contact_impulse",
        "object_velocity_discontinuity",
        "post_contact_response_window",
        "sliding_contact",
        "object_speed",
        "object_velocity_delta",
        "max_contact_impulse_proxy",
        "contact_impulse_delta",
        "task_contact_pairs",
        "new_task_contact_pairs",
    ]
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    features = []
    targets = []
    meta_rows = []

    for _, group in df.groupby(["env_id", "episode_id"], sort=False):
        group = group.sort_values("step_idx")
        values = group[state_cols + action_cols].to_numpy(dtype=np.float32)
        target_values = group[next_state_cols].to_numpy(dtype=np.float32)
        valid = group.get("is_valid_model_step", pd.Series(True, index=group.index)).astype(bool).to_numpy()
        for pos in range(history_size - 1, len(group)):
            if not valid[pos]:
                continue
            window = values[pos - history_size + 1 : pos + 1]
            if not np.isfinite(window).all() or not np.isfinite(target_values[pos]).all():
                continue
            features.append(window.reshape(-1))
            targets.append(target_values[pos])
            meta_rows.append(group.iloc[pos][metadata_cols].to_dict())

    if not features:
        raise ValueError("No valid prediction examples were produced")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        pd.DataFrame(meta_rows),
        state_cols,
        action_cols,
        next_state_cols,
    )


def split_by_episode(meta: pd.DataFrame, train_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    keys = meta[["env_id", "episode_id"]].astype(str).agg("::".join, axis=1).to_numpy()
    unique = np.unique(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    if len(unique) < 2:
        train_keys = set(unique)
    else:
        n_train = int(round(len(unique) * train_frac))
        n_train = min(max(n_train, 1), len(unique) - 1)
        train_keys = set(unique[:n_train])
    train_mask = np.asarray([key in train_keys for key in keys], dtype=bool)
    val_mask = ~train_mask
    if not val_mask.any():
        val_mask = train_mask.copy()
    return train_mask, val_mask


def standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (values - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def build_model(nn, input_dim: int, target_dim: int, hidden_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, target_dim),
    )


def train_model(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
):
    torch, nn, DataLoader, TensorDataset = import_torch()
    device = choose_device(torch, args.device)
    print_step(f"device={device}")

    x_norm, x_mean, x_std = standardize(x[train_mask], x)
    y_norm, y_mean, y_std = standardize(y[train_mask], y)
    x_train = torch.as_tensor(x_norm[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(y_norm[train_mask], dtype=torch.float32)
    x_val = torch.as_tensor(x_norm[val_mask], dtype=torch.float32)
    y_val = torch.as_tensor(y_norm[val_mask], dtype=torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = build_model(nn, x.shape[1], y.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_pred = model(x_val.to(device))
            val_loss = torch.mean((val_pred - y_val.to(device)) ** 2)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_loss": float(val_loss.detach().cpu()),
        }
        history.append(row)
        if (epoch + 1) == 1 or (epoch + 1) % max(1, args.epochs // 5) == 0:
            print_step(f"epoch={epoch + 1} train_loss={row['train_loss']:.6g} val_loss={row['val_loss']:.6g}")

    normalizers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, normalizers, pd.DataFrame(history), device, torch


def predict_all(model, normalizers: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, device, torch) -> pd.DataFrame:
    x_norm = (x - normalizers["x_mean"]) / normalizers["x_std"]
    x_tensor = torch.as_tensor(x_norm, dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        pred_norm = model(x_tensor).detach().cpu().numpy()
    pred = pred_norm * normalizers["y_std"] + normalizers["y_mean"]
    return pd.DataFrame({"mse": np.mean((pred - y) ** 2, axis=1)})


def write_summary(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Fetch One-Step Forward Error Probe",
        "",
        f"- Records: `{payload['records']}`",
        f"- Examples: `{payload['num_examples']}`",
        f"- Train examples: `{payload['train_examples']}`",
        f"- Validation examples: `{payload['val_examples']}`",
        f"- Final train loss: `{payload['final_train_loss']:.6g}`",
        f"- Final validation loss: `{payload['final_val_loss']:.6g}`",
        f"- Predictions: `{payload['predictions_path']}`",
        f"- Checkpoint: `{payload['checkpoint_path']}`",
        "",
        "This probe is intentionally single-depth. It estimates where prediction",
        "error concentrates; it does not test or implement adaptive computation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv.gz"
    checkpoint_path = args.output_dir / "forward_error_probe.pt"
    if not args.force and (predictions_path.exists() or checkpoint_path.exists()):
        raise FileExistsError(f"{args.output_dir} already has outputs; pass --force to overwrite")

    records = find_records(args)
    print_step(f"loading {len(records)} record file(s)")
    df = load_records(records, args.history_size)
    x, y, meta, state_cols, action_cols, next_state_cols = build_examples(df, args.history_size)
    train_mask, val_mask = split_by_episode(meta, args.train_frac, args.seed)
    print_step(f"examples={len(x)} train={int(train_mask.sum())} val={int(val_mask.sum())}")

    model, normalizers, history, device, torch = train_model(args, x, y, train_mask, val_mask)
    pred_df = predict_all(model, normalizers, x, y, device, torch)
    pred_df["split"] = np.where(train_mask, "train", "val")
    output = pd.concat([meta.reset_index(drop=True), pred_df], axis=1)
    output.to_csv(predictions_path, index=False, compression="gzip")
    history.to_csv(args.output_dir / "training_history.csv", index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "normalizers": normalizers,
        "state_cols": state_cols,
        "action_cols": action_cols,
        "next_state_cols": next_state_cols,
        "history_size": args.history_size,
        "hidden_dim": args.hidden_dim,
        "created_at_unix": time.time(),
    }
    torch.save(checkpoint, checkpoint_path)

    summary_payload = {
        "records": [str(path) for path in records],
        "num_examples": int(len(x)),
        "train_examples": int(train_mask.sum()),
        "val_examples": int(val_mask.sum()),
        "final_train_loss": float(history["train_loss"].iloc[-1]),
        "final_val_loss": float(history["val_loss"].iloc[-1]),
        "predictions_path": str(predictions_path),
        "checkpoint_path": str(checkpoint_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_summary(args.output_dir / "summary.md", summary_payload)
    print_step(f"wrote {predictions_path}")


if __name__ == "__main__":
    main()
