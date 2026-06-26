"""Experiment helpers for Fetch adaptive-compute diagnostics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from common import assign_event_taxonomy


DEFAULT_DATA_DIR = Path("le-wm/diagnostics/fetch_contact_compute/data")
META_COLS = [
    "env_id",
    "episode_id",
    "step_idx",
    "source_records",
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


@dataclass(frozen=True)
class ExampleSet:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    feature_cols: list[str]
    target_cols: list[str]
    state_cols: list[str]
    action_cols: list[str]
    fullstate_cols: list[str]


def numeric_suffix_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in df.columns if col.startswith(prefix) and col.rsplit("_", 1)[1].isdigit()]
    return sorted(cols, key=lambda name: int(name.rsplit("_", 1)[1]))


def find_records(data_dir: Path, records: Iterable[Path] | None = None) -> list[Path]:
    if records:
        return [Path(path) for path in records]
    found = sorted(Path(data_dir).glob("*/records.csv.gz"))
    if not found:
        raise FileNotFoundError(f"No records.csv.gz files found under {data_dir}")
    return found


def load_records(paths: Iterable[Path], *, history_size: int) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "primary_regime" not in df:
            df, _ = assign_event_taxonomy(df, history_size=history_size)
        df["source_records"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("No record frames were loaded")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["env_id", "episode_id", "step_idx"]).reset_index(drop=True)


def add_shifted_input_fullstate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add input-time qpos/qvel features from the previous post-step row.

    The existing collector stores qpos/qvel after env.step(action), so same-row
    values are target-time and would leak. For step t > 0, the previous row's
    post-step qpos/qvel are the simulator state available before action t.
    """

    qpos_cols = numeric_suffix_cols(df, "qpos_")
    qvel_cols = numeric_suffix_cols(df, "qvel_")
    if not qpos_cols or not qvel_cols:
        raise ValueError("Full-state ablation requires qpos_* and qvel_* columns")

    out = df.copy()
    shifted_cols: list[str] = []
    for src_col in qpos_cols + qvel_cols:
        if src_col.startswith("qpos_"):
            dst_col = src_col.replace("qpos_", "input_qpos_", 1)
        else:
            dst_col = src_col.replace("qvel_", "input_qvel_", 1)
        shifted_cols.append(dst_col)
        out[dst_col] = np.nan

    for _, group in out.groupby(["env_id", "episode_id"], sort=False):
        group = group.sort_values("step_idx")
        shifted = group[qpos_cols + qvel_cols].shift(1)
        shifted.columns = shifted_cols
        out.loc[group.index, shifted_cols] = shifted.to_numpy(dtype=np.float32)
    return out, shifted_cols


def build_examples(
    df: pd.DataFrame,
    *,
    history_size: int,
    include_shifted_fullstate: bool,
) -> ExampleSet:
    working = df
    fullstate_cols: list[str] = []
    if include_shifted_fullstate:
        working, fullstate_cols = add_shifted_input_fullstate(df)

    state_cols = numeric_suffix_cols(working, "state_")
    action_cols = numeric_suffix_cols(working, "action_")
    target_cols = numeric_suffix_cols(working, "next_state_")
    if not state_cols or not action_cols or not target_cols:
        raise ValueError("Records must contain state_*, action_*, and next_state_* columns")

    row_feature_cols = state_cols + action_cols + fullstate_cols
    meta_cols = [col for col in META_COLS if col in working.columns]
    features = []
    targets = []
    meta_rows = []

    for _, group in working.groupby(["env_id", "episode_id"], sort=False):
        group = group.sort_values("step_idx")
        row_features = group[row_feature_cols].to_numpy(dtype=np.float32)
        target_values = group[target_cols].to_numpy(dtype=np.float32)
        valid = group.get("is_valid_model_step", pd.Series(True, index=group.index)).astype(bool).to_numpy()
        for pos in range(history_size - 1, len(group)):
            if not valid[pos]:
                continue
            window = row_features[pos - history_size + 1 : pos + 1]
            target = target_values[pos]
            if not np.isfinite(window).all() or not np.isfinite(target).all():
                continue
            features.append(window.reshape(-1))
            targets.append(target)
            meta_rows.append(group.iloc[pos][meta_cols].to_dict())

    if not features:
        raise ValueError("No valid prediction examples were produced")
    feature_cols = [f"hist_{offset}:{name}" for offset in range(history_size) for name in row_feature_cols]
    return ExampleSet(
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(targets, dtype=np.float32),
        meta=pd.DataFrame(meta_rows),
        feature_cols=feature_cols,
        target_cols=target_cols,
        state_cols=state_cols,
        action_cols=action_cols,
        fullstate_cols=fullstate_cols,
    )


def episode_key_frame(meta: pd.DataFrame) -> pd.Series:
    return meta[["env_id", "episode_id"]].astype(str).agg("::".join, axis=1)


def split_by_episode(
    meta: pd.DataFrame,
    *,
    train_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    keys = episode_key_frame(meta).to_numpy()
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
    val_keys = sorted(set(keys[val_mask]))
    return train_mask, val_mask, sorted(train_keys), val_keys


def masks_from_train_keys(meta: pd.DataFrame, train_keys: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    train_key_set = set(train_keys)
    keys = episode_key_frame(meta).to_numpy()
    train_mask = np.asarray([key in train_key_set for key in keys], dtype=bool)
    return train_mask, ~train_mask


def standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (values - mean) / std, mean.astype(np.float32), std.astype(np.float32)


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


def build_mlp(nn, input_dim: int, target_dim: int, hidden_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, target_dim),
    )


def train_mlp(
    *,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    log_fn: Callable[[str], None],
) -> tuple[object, dict[str, np.ndarray], pd.DataFrame, object, object]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(torch, device_name)
    log_fn(f"device={device} seed={seed}")

    x_norm, x_mean, x_std = standardize(x[train_mask], x)
    y_norm, y_mean, y_std = standardize(y[train_mask], y)
    x_train = torch.as_tensor(x_norm[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(y_norm[train_mask], dtype=torch.float32)
    x_val = torch.as_tensor(x_norm[val_mask], dtype=torch.float32)
    y_val = torch.as_tensor(y_norm[val_mask], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = build_mlp(nn, x.shape[1], y.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
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
        if (epoch + 1) == 1 or (epoch + 1) % max(1, epochs // 5) == 0:
            log_fn(f"epoch={epoch + 1} train_loss={row['train_loss']:.6g} val_loss={row['val_loss']:.6g}")
    normalizers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, normalizers, pd.DataFrame(history), device, torch


def predict_mse(
    *,
    model: object,
    normalizers: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    device: object,
    torch: object,
    chunk_size: int = 8192,
) -> pd.DataFrame:
    pred = predict_values(
        model=model,
        normalizers=normalizers,
        x=x,
        device=device,
        torch=torch,
        chunk_size=chunk_size,
    )
    return pd.DataFrame({"mse": np.mean((pred - y) ** 2, axis=1)})


def predict_values(
    *,
    model: object,
    normalizers: dict[str, np.ndarray],
    x: np.ndarray,
    device: object,
    torch: object,
    chunk_size: int = 8192,
) -> np.ndarray:
    x_norm = (x - normalizers["x_mean"]) / normalizers["x_std"]
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_norm), chunk_size):
            xb = torch.as_tensor(x_norm[start : start + chunk_size], dtype=torch.float32, device=device)
            pred_norm = model(xb).detach().cpu().numpy()
            chunks.append(pred_norm)
    return np.concatenate(chunks, axis=0) * normalizers["y_std"] + normalizers["y_mean"]


def bias_variance_components(predictions: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    """Return finite-ensemble bias/variance components for one target sample each.

    With one observed next state per input, irreducible environment noise is not
    separately identifiable. The finite-ensemble identity reported here is:
    mean_member_mse = bias2_mse + variance_mse + numerical_residual.
    """

    preds = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if preds.ndim != 3:
        raise ValueError("predictions must have shape (members, examples, target_dims)")
    if y.ndim != 2:
        raise ValueError("targets must have shape (examples, target_dims)")
    if preds.shape[1:] != y.shape:
        raise ValueError(f"Prediction/target shape mismatch: {preds.shape} vs {y.shape}")
    ensemble_mean = preds.mean(axis=0)
    bias2 = np.mean((ensemble_mean - y) ** 2, axis=1)
    variance = np.mean(np.mean((preds - ensemble_mean[None, :, :]) ** 2, axis=2), axis=0)
    heldout_error = np.mean(np.mean((preds - y[None, :, :]) ** 2, axis=2), axis=0)
    residual = heldout_error - bias2 - variance
    return pd.DataFrame(
        {
            "bias2_mse": bias2,
            "variance_mse": variance,
            "heldout_error_mse": heldout_error,
            "decomposition_residual_mse": residual,
        }
    )


def trimmed_mean(values: np.ndarray, proportion_to_cut: float = 0.1) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    arr = np.sort(arr)
    trim = int(math.floor(len(arr) * proportion_to_cut))
    if trim == 0:
        return float(arr.mean())
    if 2 * trim >= len(arr):
        return float(arr.mean())
    return float(arr[trim:-trim].mean())


def probability_superiority(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    right_sorted = np.sort(right)
    less = np.searchsorted(right_sorted, left, side="left")
    less_equal = np.searchsorted(right_sorted, left, side="right")
    ties = less_equal - less
    return float((less.sum() + 0.5 * ties.sum()) / (len(left) * len(right)))


def error_metrics(left: np.ndarray, right: np.ndarray | None = None) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    left = left[np.isfinite(left)]
    result = {
        "left_n": int(len(left)),
        "left_mean": float(left.mean()) if len(left) else float("nan"),
        "left_median": float(np.median(left)) if len(left) else float("nan"),
        "left_trimmed_mean": trimmed_mean(left),
    }
    if right is None:
        return result
    right = np.asarray(right, dtype=np.float64)
    right = right[np.isfinite(right)]
    p_superiority = probability_superiority(left, right)
    result.update(
        {
            "right_n": int(len(right)),
            "right_mean": float(right.mean()) if len(right) else float("nan"),
            "right_median": float(np.median(right)) if len(right) else float("nan"),
            "right_trimmed_mean": trimmed_mean(right),
            "delta_mean": float(left.mean() - right.mean()) if len(left) and len(right) else float("nan"),
            "delta_median": float(np.median(left) - np.median(right)) if len(left) and len(right) else float("nan"),
            "delta_trimmed_mean": trimmed_mean(left) - trimmed_mean(right),
            "p_superiority": p_superiority,
            "cliffs_delta": 2.0 * p_superiority - 1.0,
        }
    )
    return result


def bootstrap_ci(values: list[dict[str, float]], key: str) -> tuple[float, float]:
    arr = np.asarray([row[key] for row in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def cluster_bootstrap_contrast(
    df: pd.DataFrame,
    *,
    left_mask: pd.Series,
    right_mask: pd.Series,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    point = error_metrics(
        df.loc[left_mask, "mse"].to_numpy(dtype=np.float64),
        df.loc[right_mask, "mse"].to_numpy(dtype=np.float64),
    )
    clusters = [
        group.index.to_numpy()
        for _, group in df.groupby(["env_id", "episode_id"], sort=False)
        if bool(left_mask.loc[group.index].any()) or bool(right_mask.loc[group.index].any())
    ]
    rng = np.random.default_rng(seed)
    boot_rows: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[i] for i in selected])
        boot_df = df.loc[idx]
        boot_left = left_mask.loc[idx].to_numpy(dtype=bool)
        boot_right = right_mask.loc[idx].to_numpy(dtype=bool)
        boot_rows.append(
            error_metrics(
                boot_df.loc[boot_left, "mse"].to_numpy(dtype=np.float64),
                boot_df.loc[boot_right, "mse"].to_numpy(dtype=np.float64),
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
    point = error_metrics(df.loc[group_mask, "mse"].to_numpy(dtype=np.float64))
    clusters = [
        group.index.to_numpy()
        for _, group in df.groupby(["env_id", "episode_id"], sort=False)
        if bool(group_mask.loc[group.index].any())
    ]
    rng = np.random.default_rng(seed)
    boot_rows: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[i] for i in selected])
        boot_df = df.loc[idx]
        boot_mask = group_mask.loc[idx].to_numpy(dtype=bool)
        boot_rows.append(error_metrics(boot_df.loc[boot_mask, "mse"].to_numpy(dtype=np.float64)))
    return point, boot_rows


def attach_ci(row: dict[str, object], boot_rows: list[dict[str, float]], metrics: Iterable[str]) -> dict[str, object]:
    out = dict(row)
    for metric in metrics:
        low, high = bootstrap_ci(boot_rows, metric)
        out[f"{metric}_ci_low"] = low
        out[f"{metric}_ci_high"] = high
    return out
