#!/usr/bin/env python3
"""Run Cube event-localization diagnostics for the released pixel LeWM checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import py_compile
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hdf5plugin  # noqa: F401 - registers HDF5 compression filters before h5py opens files.
import h5py
import lance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
from huggingface_hub import hf_hub_download
from scipy.optimize import minimize
from scipy.stats import chi2, fisher_exact, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from module import ARPredictor, Embedder, MLP
from transformers import ViTConfig, ViTModel
from jepa import JEPA


MODEL_REPO = "quentinll/lewm-cube"
DEFAULT_SOURCE_H5 = Path("/tmp/lewm_cube_stage0/cube_single_expert.h5")
DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/cube_event_localization")
DEFAULT_CACHE_DIR = Path("le-wm/.cache/cube_event_diagnostic")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRECHECK = {
    "trajectories": 30,
    "total_onsets": 70,
    "median_onsets_per_traj": 2.0,
    "median_bout_length": 42.0,
    "contact_fraction": 0.4235489220563847,
    "bouts_per_episode": 2.0,
}


@dataclass(frozen=True)
class EventSpec:
    key: str
    name: str
    feature: str
    event_col: str
    tag: str
    marker: str
    color: str
    threshold_percentile: float | None = None


EVENTS = [
    EventSpec(
        key="sensor_grasp_onset",
        name="Sensor grasp onset",
        feature="sensor_grasp_onset",
        event_col="event_sensor_grasp_onset",
        tag="USABLE",
        marker="o",
        color="#2563eb",
    ),
    EventSpec(
        key="sensor_release",
        name="Sensor release",
        feature="sensor_release",
        event_col="event_sensor_release",
        tag="USABLE",
        marker="v",
        color="#0891b2",
    ),
    EventSpec(
        key="effector_kinematic_change",
        name="Effector velocity/accel change",
        feature="effector_acceleration_proxy",
        event_col="event_effector_kinematic_change",
        tag="USABLE",
        marker="^",
        color="#16a34a",
        threshold_percentile=90.0,
    ),
    EventSpec(
        key="block_kinematic_discontinuity",
        name="Block velocity/accel discontinuity",
        feature="block_acceleration_proxy",
        event_col="event_block_kinematic_discontinuity",
        tag="CIRCULAR",
        marker="s",
        color="#dc2626",
        threshold_percentile=90.0,
    ),
    EventSpec(
        key="orientation_delta",
        name="Orientation delta",
        feature="orientation_delta",
        event_col="event_orientation_delta",
        tag="CIRCULAR-ish",
        marker="D",
        color="#ea580c",
        threshold_percentile=90.0,
    ),
    EventSpec(
        key="normalized_trajectory_position",
        name="Normalized trajectory position",
        feature="normalized_position",
        event_col="event_normalized_trajectory_position",
        tag="CONTROL",
        marker="x",
        color="#6b7280",
        threshold_percentile=90.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_SOURCE_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--num-trajectories", type=int, default=30)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--high-mse-percentile", type=float, default=90.0)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-lance", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[cube-event] {message}", flush=True)


def clean_cfg(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "_target_"}


def choose_device(requested: str) -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print_step("MPS requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print_step("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def vit_hf_from_config(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> ViTModel:
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
    }
    if size not in size_configs:
        raise ValueError(f"Unknown ViT size {size!r}")
    params = dict(size_configs[size])
    params["intermediate_size"] = params["hidden_size"] * 4
    params["image_size"] = image_size
    params["patch_size"] = patch_size
    params.update(kwargs)
    if pretrained:
        model = ViTModel.from_pretrained(
            f"google/vit-{size}-patch{patch_size}-{image_size}",
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    else:
        model = ViTModel(ViTConfig(**params), add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model


def mlp_from_config(cfg: dict, key: str) -> MLP:
    mlp_cfg = clean_cfg(cfg[key])
    norm_cfg = mlp_cfg.pop("norm_fn", None)
    norm_fn = torch.nn.LayerNorm
    if isinstance(norm_cfg, dict):
        target = norm_cfg.get("_target_", "")
        if target.endswith("BatchNorm1d"):
            norm_fn = torch.nn.BatchNorm1d
        elif target.endswith("LayerNorm"):
            norm_fn = torch.nn.LayerNorm
    elif norm_cfg is None:
        norm_fn = None
    return MLP(norm_fn=norm_fn, **mlp_cfg)


def remap_legacy_vit_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("encoder.encoder.layer."):
            new_key = key.replace("encoder.encoder.layer.", "encoder.layers.", 1)
            new_key = new_key.replace(".attention.attention.query.", ".attention.q_proj.")
            new_key = new_key.replace(".attention.attention.key.", ".attention.k_proj.")
            new_key = new_key.replace(".attention.attention.value.", ".attention.v_proj.")
            new_key = new_key.replace(".attention.output.dense.", ".attention.o_proj.")
            new_key = new_key.replace(".intermediate.dense.", ".mlp.fc1.")
            new_key = new_key.replace(".output.dense.", ".mlp.fc2.")
        remapped[new_key] = value
    return remapped


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[JEPA, dict, Path, Path]:
    model_dir = args.cache_dir / "model"
    cfg_path = Path(hf_hub_download(args.model_repo, "config.json", local_dir=model_dir))
    weights_path = Path(hf_hub_download(args.model_repo, "weights.pt", local_dir=model_dir))
    cfg = json.loads(cfg_path.read_text())
    model = JEPA(
        encoder=vit_hf_from_config(**clean_cfg(cfg["encoder"])),
        predictor=ARPredictor(**clean_cfg(cfg["predictor"])),
        action_encoder=Embedder(**clean_cfg(cfg["action_encoder"])),
        projector=mlp_from_config(cfg, "projector"),
        pred_proj=mlp_from_config(cfg, "pred_proj"),
    )
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = model.load_state_dict(remap_legacy_vit_keys(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    model.eval().requires_grad_(False).to(device)
    return model, cfg, cfg_path, weights_path


def validate_training_match(cfg: dict) -> dict[str, object]:
    predictor_frames = int(cfg["predictor"]["num_frames"])
    action_input_dim = int(cfg["action_encoder"]["input_dim"])
    image_size = int(cfg["encoder"]["image_size"])
    raw_action_dim = 5
    if action_input_dim % raw_action_dim != 0:
        raise RuntimeError(
            f"action_encoder.input_dim={action_input_dim} is not divisible by Cube raw action dim {raw_action_dim}"
        )
    frameskip = action_input_dim // raw_action_dim
    if predictor_frames != 3:
        raise RuntimeError(f"Unexpected Cube predictor history num_frames={predictor_frames}; expected 3")
    if frameskip != 5:
        raise RuntimeError(f"Unexpected Cube action-block frameskip={frameskip}; expected 5 from input_dim=25")
    if image_size != 224:
        raise RuntimeError(f"Unexpected Cube encoder image_size={image_size}; expected 224")
    return {
        "history_size": predictor_frames,
        "frameskip": frameskip,
        "raw_action_dim": raw_action_dim,
        "action_encoder_input_dim": action_input_dim,
        "image_size": image_size,
        "pixel_normalization": f"uint8/255 then ImageNet mean={IMAGENET_MEAN}, std={IMAGENET_STD}",
    }


def pixel_transform(pixels: np.ndarray, device: torch.device, image_size: int) -> torch.Tensor:
    x = torch.as_tensor(pixels, dtype=torch.uint8)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB pixels, got shape {tuple(x.shape)}")
    x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
    return x.to(device)


def h5_layout(h5: h5py.File, n: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    lengths = np.asarray(h5["ep_len"][:n], dtype=np.int64)
    offsets = np.asarray(h5["ep_offset"][:n], dtype=np.int64)
    if len(lengths) != n or len(offsets) != n:
        raise RuntimeError(f"Requested {n} trajectories but source exposes only {len(lengths)} lengths")
    if int(offsets[0]) != 0:
        raise RuntimeError(f"Expected first pre-check trajectory offset 0, got {int(offsets[0])}")
    stop = int(offsets[-1] + lengths[-1])
    source_rows = int(lengths.sum())
    if stop != source_rows:
        raise RuntimeError(f"First {n} episodes are not contiguous: stop={stop}, sum(ep_len)={source_rows}")
    return lengths, offsets, 0, stop


def fixed_tensor(values: np.ndarray) -> pa.FixedShapeTensorArray:
    return pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(values))


def write_lance_subset(
    h5: h5py.File,
    lance_dir: Path,
    *,
    start: int,
    stop: int,
    lengths: np.ndarray,
    offsets: np.ndarray,
    batch_rows: int = 100,
    force: bool = False,
) -> int:
    if lance_dir.exists() and force:
        shutil.rmtree(lance_dir)
    if lance_dir.exists():
        ds = lance.dataset(lance_dir)
        rows = ds.count_rows()
        if rows == stop - start:
            return rows
        shutil.rmtree(lance_dir)

    keys = [
        "pixels",
        "observation",
        "proprio_gripper_contact",
        "proprio_effector_pos",
        "privileged_block_0_pos",
        "privileged_block_0_quat",
        "privileged_block_0_yaw",
        "action",
        "ep_idx",
        "step_idx",
        "qpos",
        "qvel",
    ]
    length_by_episode = {i: int(length) for i, length in enumerate(lengths)}
    offset_by_episode = {i: int(offset) for i, offset in enumerate(offsets)}
    mode = "overwrite"
    for batch_start in range(start, stop, batch_rows):
        batch_stop = min(stop, batch_start + batch_rows)
        data = {}
        for key in keys:
            values = h5[key][batch_start:batch_stop]
            if values.dtype.kind in ("S", "U", "O"):
                data[key] = pa.array([x.decode() if isinstance(x, bytes) else str(x) for x in values])
            elif values.ndim > 1:
                data[key] = fixed_tensor(values)
            else:
                data[key] = pa.array(values)
        ep_values = np.asarray(h5["ep_idx"][batch_start:batch_stop], dtype=np.int64)
        data["episode_length"] = pa.array([length_by_episode[int(ep)] for ep in ep_values], type=pa.int64())
        data["episode_offset"] = pa.array([offset_by_episode[int(ep)] for ep in ep_values], type=pa.int64())
        batch = pa.record_batch(data)
        lance.write_dataset(batch, lance_dir, mode=mode, max_rows_per_group=100)
        mode = "append"
    return lance.dataset(lance_dir).count_rows()


def contact_stats(labels_by_episode: list[np.ndarray]) -> dict[str, object]:
    all_labels = []
    onset_counts = []
    bout_counts = []
    bout_lengths = []
    for labels in labels_by_episode:
        labels = np.asarray(labels, dtype=bool)
        prev = np.r_[False, labels[:-1]]
        onsets = np.flatnonzero(labels & ~prev)
        offsets = np.flatnonzero(~labels & prev)
        if labels[-1]:
            offsets = np.r_[offsets, len(labels)]
        onset_counts.append(len(onsets))
        bout_counts.append(len(onsets))
        for start, end in zip(onsets, offsets):
            bout_lengths.append(int(end - start))
        all_labels.append(labels)
    concat = np.concatenate(all_labels)
    return {
        "total_onsets": int(sum(onset_counts)),
        "median_onsets_per_traj": float(np.median(onset_counts)),
        "contact_fraction": float(concat.mean()),
        "median_bout_length": float(np.median(bout_lengths)) if bout_lengths else 0.0,
        "bouts_per_episode": float(np.median(bout_counts)),
    }


def compare_precheck(stats: dict[str, object]) -> dict[str, object]:
    divergence = []
    if int(stats["total_onsets"]) != PRECHECK["total_onsets"]:
        divergence.append("total_onsets")
    for key, tolerance in [
        ("median_onsets_per_traj", 0.01),
        ("contact_fraction", 0.01),
        ("median_bout_length", 0.1),
        ("bouts_per_episode", 0.01),
    ]:
        if abs(float(stats[key]) - float(PRECHECK[key])) > tolerance:
            divergence.append(key)
    return {
        "precheck": PRECHECK,
        "full_data": stats,
        "divergence": divergence,
    }


def wrapped_angle_delta(angle_t: float, angle_prev: float) -> float:
    delta = angle_t - angle_prev
    return float(abs(math.atan2(math.sin(delta), math.cos(delta))))


def edge_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=bool)
    prev = np.r_[False, labels[:-1]]
    onsets = np.flatnonzero(labels & ~prev)
    releases = np.flatnonzero(~labels & prev)
    return onsets, releases


def raw_edge_to_target_model_step(raw_idx: int, frameskip: int) -> int:
    return int((max(raw_idx, 1) - 1) // frameskip + 1)


def encode_episode(
    model: JEPA,
    h5: h5py.File,
    *,
    offset: int,
    length: int,
    frameskip: int,
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> torch.Tensor:
    obs_rows = offset + np.arange(1 + (length - 1) // frameskip, dtype=np.int64) * frameskip
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(obs_rows), batch_size):
            rows = obs_rows[i : i + batch_size]
            pixels = h5["pixels"][rows]
            x = pixel_transform(pixels, device, image_size)
            out = model.encode({"pixels": x.unsqueeze(0)})
            chunks.append(out["emb"].squeeze(0).detach().cpu())
    return torch.cat(chunks, dim=0)


def predict_episode_records(
    model: JEPA,
    h5: h5py.File,
    *,
    episode_ordinal: int,
    episode_id: int,
    offset: int,
    length: int,
    frameskip: int,
    history_size: int,
    device: torch.device,
    encode_batch_size: int,
    predict_batch_size: int,
    image_size: int,
) -> list[dict[str, object]]:
    emb = encode_episode(
        model,
        h5,
        offset=offset,
        length=length,
        frameskip=frameskip,
        device=device,
        batch_size=encode_batch_size,
        image_size=image_size,
    )
    model_steps = emb.shape[0]
    action_blocks = []
    for block_idx in range(model_steps - 1):
        raw_start = offset + block_idx * frameskip
        raw_stop = raw_start + frameskip
        block = np.asarray(h5["action"][raw_start:raw_stop], dtype=np.float32).reshape(-1)
        if block.shape[0] != frameskip * 5:
            raise RuntimeError(f"Action block shape mismatch in episode {episode_id}, block {block_idx}: {block.shape}")
        action_blocks.append(block)
    actions = torch.as_tensor(np.asarray(action_blocks), dtype=torch.float32, device=device).unsqueeze(0)
    actions = torch.nan_to_num(actions, 0.0)
    with torch.inference_mode():
        act_emb = model.action_encoder(actions).squeeze(0).detach().cpu()

    raw_contact = np.asarray(h5["proprio_gripper_contact"][offset : offset + length, 0], dtype=np.float64)
    contact = raw_contact > 1e-9
    onsets_raw, releases_raw = edge_indices(contact)
    onset_steps = {raw_edge_to_target_model_step(int(idx), frameskip) for idx in onsets_raw}
    release_steps = {raw_edge_to_target_model_step(int(idx), frameskip) for idx in releases_raw}

    effector = np.asarray(h5["proprio_effector_pos"][offset : offset + length], dtype=np.float64)
    block_pos = np.asarray(h5["privileged_block_0_pos"][offset : offset + length], dtype=np.float64)
    block_yaw = np.asarray(h5["privileged_block_0_yaw"][offset : offset + length, 0], dtype=np.float64)

    target_steps = np.arange(history_size, model_steps, dtype=np.int64)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(target_steps), predict_batch_size):
            batch_targets = target_steps[start : start + predict_batch_size]
            emb_windows = []
            act_windows = []
            for target in batch_targets:
                lo = int(target) - history_size
                emb_windows.append(emb[lo:int(target)])
                act_windows.append(act_emb[lo:int(target)])
            emb_tensor = torch.stack(emb_windows, dim=0).to(device)
            act_tensor = torch.stack(act_windows, dim=0).to(device)
            gt = emb[batch_targets].to(device)
            pred = model.predict(emb_tensor, act_tensor)[:, -1]
            mse = ((pred - gt) ** 2).mean(dim=1).detach().cpu().numpy()
            identity = ((emb[batch_targets] - emb[batch_targets - 1]) ** 2).mean(dim=1).numpy()

            for target, err, ident in zip(batch_targets, mse, identity):
                target = int(target)
                raw_step = target * frameskip
                prev_raw = (target - 1) * frameskip
                prev2_raw = (target - 2) * frameskip
                transition_block = target - 1
                block_contact = bool(contact[prev_raw + 1 : raw_step + 1].any())

                eff_acc = effector[raw_step] - 2.0 * effector[prev_raw] + effector[prev2_raw]
                block_acc = block_pos[raw_step] - 2.0 * block_pos[prev_raw] + block_pos[prev2_raw]
                yaw_delta = wrapped_angle_delta(float(block_yaw[raw_step]), float(block_yaw[prev_raw]))
                rows.append(
                    {
                        "episode_ordinal": episode_ordinal,
                        "episode_id": episode_id,
                        "model_step": target,
                        "raw_step": raw_step,
                        "transition_block": transition_block,
                        "mse": float(err),
                        "identity_mse": float(ident),
                        "contact": block_contact,
                        "sensor_grasp_onset": target in onset_steps,
                        "sensor_release": target in release_steps,
                        "effector_acceleration_proxy": float(np.linalg.norm(eff_acc)),
                        "block_acceleration_proxy": float(np.linalg.norm(block_acc)),
                        "orientation_delta": yaw_delta,
                        "normalized_position": float(target / max(model_steps - 1, 1)),
                    }
                )
    return rows


def add_event_labels(df: pd.DataFrame, high_mse_percentile: float) -> tuple[pd.DataFrame, dict[str, float], float]:
    out = df.copy()
    thresholds: dict[str, float] = {}
    for spec in EVENTS:
        if spec.threshold_percentile is None:
            out[spec.event_col] = out[spec.feature].astype(bool)
            thresholds[spec.key] = float("nan")
        else:
            threshold = float(np.percentile(out[spec.feature], spec.threshold_percentile))
            thresholds[spec.key] = threshold
            out[spec.event_col] = out[spec.feature] >= threshold
    high_threshold = float(np.percentile(out["mse"], high_mse_percentile))
    out["high_mse_p90"] = high_threshold
    out["high_mse"] = out["mse"] >= high_threshold
    return out, thresholds, high_threshold


def aligned_curve(df: pd.DataFrame, event_col: str, *, window: int) -> pd.DataFrame:
    pieces = []
    event_rows = df.loc[df[event_col], ["episode_ordinal", "model_step"]].reset_index(drop=True)
    for event_id, event in event_rows.iterrows():
        episode = int(event["episode_ordinal"])
        event_step = int(event["model_step"])
        group = df.loc[df["episode_ordinal"] == episode, ["episode_ordinal", "model_step", "mse"]].copy()
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
    return float(by_rel.loc[0, "mean"] / baseline) if baseline else float("nan")


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
    pre = by_rel.loc[(by_rel.index >= -10) & (by_rel.index <= -1), "mean"]
    post = by_rel.loc[(by_rel.index >= 1) & (by_rel.index <= 10), "mean"]
    localized = (
        np.isfinite(tminus5)
        and np.isfinite(t0)
        and len(pre) >= 3
        and len(post) >= 3
        and t0 > float(pre.mean())
        and (not np.isfinite(tplus5) or t0 >= tplus5)
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
    result = minimize(objective, np.zeros(x.shape[1], dtype=np.float64), method="L-BFGS-B", bounds=bounds)
    if not result.success:
        result = minimize(objective, np.zeros(x.shape[1], dtype=np.float64), method="Powell", bounds=bounds)
    beta = np.asarray(result.x, dtype=np.float64)
    return beta, -objective(beta)


def position_adjusted_logistic(df: pd.DataFrame, spec: EventSpec) -> dict[str, float | bool]:
    y = df["high_mse"].astype(float).to_numpy()
    event = df[spec.event_col].astype(float).to_numpy()
    position = df["normalized_position"].astype(float).to_numpy()
    null_x = np.column_stack([np.ones(len(df)), position])
    full_x = np.column_stack([np.ones(len(df)), event, position])
    try:
        _, null_ll = fit_logistic_loglik(y, null_x)
        full_beta, full_ll = fit_logistic_loglik(y, full_x)
        lr_stat = max(0.0, 2.0 * (full_ll - null_ll))
        p_value = float(chi2.sf(lr_stat, df=1))
        adjusted_or = float(math.exp(np.clip(full_beta[1], -30, 30)))
        position_or = float(math.exp(np.clip(full_beta[2], -30, 30)))
        return {
            "position_adjusted_odds_ratio": adjusted_or,
            "position_adjusted_lr_p": p_value,
            "position_adjusted_position_or": position_or,
            "position_adjusted_pass": bool(adjusted_or > 1.0 and p_value < 0.05),
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
    for _, group in df.groupby("episode_ordinal", sort=False):
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
    adjusted = position_adjusted_logistic(df, spec)
    trajectory = trajectory_level_metrics(df, spec)
    curve_info = curve_values(curve)
    explains = bool(adjusted["position_adjusted_pass"] and trajectory["trajectory_pass"])
    promising = bool(
        not explains
        and (
            adjusted["position_adjusted_pass"]
            or trajectory["trajectory_pass"]
            or curve_info["localized_peak_pass"]
            or (np.isfinite(lift) and lift > 1.3)
            or (float(odds_ratio) > 1.0 and float(fisher_p) < 0.05)
        )
    )
    return {
        "event": spec.name,
        "feature": spec.feature,
        "threshold": threshold,
        "event_count": int(event.sum()),
        "event_rate_pct": 100.0 * float(event.mean()),
        "raw_odds_ratio": float(odds_ratio),
        "fisher_p_greater": float(fisher_p),
        "aligned_lift_t0_over_tminus5": lift,
        **curve_info,
        **adjusted,
        **trajectory,
        "tag": spec.tag,
        "verdict": "explains" if explains else "promising" if promising else "no",
    }


def position_only_check(df: pd.DataFrame) -> dict[str, float]:
    y = df["high_mse"].astype(float).to_numpy()
    position = df["normalized_position"].astype(float).to_numpy()
    null_x = np.ones((len(df), 1), dtype=np.float64)
    full_x = np.column_stack([np.ones(len(df)), position])
    null_beta, null_ll = fit_logistic_loglik(y, null_x)
    full_beta, full_ll = fit_logistic_loglik(y, full_x)
    lr = max(0.0, 2.0 * (full_ll - null_ll))
    return {
        "position_only_odds_ratio": float(math.exp(np.clip(full_beta[1], -30, 30))),
        "position_only_lr_p": float(chi2.sf(lr, df=1)),
    }


def plot_curve(curve: pd.DataFrame, path: Path, title: str, color: str = "#2563eb") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if len(curve):
        x = curve["rel_event"].to_numpy(dtype=np.float64)
        y = curve["mean"].to_numpy(dtype=np.float64)
        ci = curve["ci95"].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=color, linewidth=2)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.18, linewidth=0)
        ax.axvline(0, color="#111827", linestyle="--", linewidth=1)
    else:
        ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Model step relative to event")
    ax.set_ylabel("Mean one-step latent MSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_all_curves(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for spec in EVENTS:
        curve = curves[spec.key]
        if len(curve) == 0:
            continue
        x = curve["rel_event"].to_numpy(dtype=np.float64)
        y = curve["mean"].to_numpy(dtype=np.float64)
        ci = curve["ci95"].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=spec.color, linewidth=2, label=spec.name)
        ax.fill_between(x, y - ci, y + ci, color=spec.color, alpha=0.12, linewidth=0)
    ax.axvline(0, color="#111827", linestyle="--", linewidth=1)
    ax.set_xlabel("Model step relative to event")
    ax.set_ylabel("Mean one-step latent MSE")
    ax.set_title("Cube event-aligned MSE averages")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_raster(df: pd.DataFrame, path: Path) -> None:
    episodes = sorted(df["episode_ordinal"].unique())
    ncols = 3
    nrows = int(math.ceil(len(episodes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(8, nrows * 1.8)), sharex=False, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, episode in zip(axes_flat, episodes):
        group = df.loc[df["episode_ordinal"] == episode].sort_values("model_step")
        ax.plot(group["model_step"], group["mse"], color="#111827", linewidth=1.2)
        ymax = max(float(group["mse"].max()), 1e-12)
        for spec in EVENTS:
            ev = group.loc[group[spec.event_col]]
            if len(ev):
                ax.scatter(
                    ev["model_step"],
                    np.full(len(ev), ymax * 1.05),
                    s=18,
                    marker=spec.marker,
                    color=spec.color,
                    alpha=0.85,
                    label=spec.name,
                )
        ax.set_title(f"Episode {int(episode)}", fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes_flat[len(episodes) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3, fontsize=8)
    fig.supxlabel("Model step")
    fig.supylabel("One-step latent MSE")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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


def first_onset_and_second_release_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grasp_indices = []
    release_indices = []
    for _, group in df.groupby("episode_ordinal", sort=False):
        onsets = group.index[group["event_sensor_grasp_onset"]].tolist()
        releases = group.index[group["event_sensor_release"]].tolist()
        if onsets:
            grasp_indices.append(onsets[0])
        if len(releases) >= 2:
            release_indices.append(releases[1])
        elif releases:
            release_indices.append(releases[-1])
    grasp_df = df.copy()
    release_df = df.copy()
    grasp_df["phase_grasp_onset_bout1"] = False
    release_df["phase_release_bout2"] = False
    grasp_df.loc[grasp_indices, "phase_grasp_onset_bout1"] = True
    release_df.loc[release_indices, "phase_release_bout2"] = True
    return grasp_df, release_df


def verify_outputs(paths: Iterable[Path]) -> list[str]:
    checks = []
    for path in paths:
        exists = path.exists()
        nonempty = path.stat().st_size > 0 if exists else False
        checks.append(f"{path.name}: exists={exists}, nonempty={nonempty}, size={path.stat().st_size if exists else 0}")
    return checks


def main() -> None:
    args = parse_args()
    py_compile.compile(__file__, doraise=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    lance_dir = args.output_dir / "cube_first30_pixel_subset.lance"

    device = choose_device(args.device)
    model, cfg, cfg_path, weights_path = load_model(args, device)
    training = validate_training_match(cfg)
    print_step(f"device={device}; MPS fallback={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    print_step(f"checkpoint config={cfg_path}; weights={weights_path}")
    print_step(
        "training match: "
        f"history={training['history_size']} frameskip={training['frameskip']} "
        f"action_input_dim={training['action_encoder_input_dim']} image_size={training['image_size']}"
    )
    print_step(f"pixel normalization: {training['pixel_normalization']}")
    print_step("loader.pin_memory=False for this diagnostic; inference tensors are moved explicitly")

    with h5py.File(args.source_h5, "r") as h5:
        lengths, offsets, start, stop = h5_layout(h5, args.num_trajectories)
        pixel_shape = (stop - start, *tuple(h5["pixels"].shape[1:]))
        pixel_dtype = str(h5["pixels"].dtype)
        source_step_count = int(lengths.sum())
        print_step(f"pixel subset shape={pixel_shape}, dtype={pixel_dtype}")
        print_step(f"frame count check: pixels={pixel_shape[0]}, sum(ep_len)={source_step_count}")
        if pixel_shape[0] != source_step_count:
            raise RuntimeError(f"Pixel frame count mismatch: {pixel_shape[0]} != {source_step_count}")

        lance_rows = write_lance_subset(
            h5,
            lance_dir,
            start=start,
            stop=stop,
            lengths=lengths,
            offsets=offsets,
            batch_rows=300,
            force=args.force_lance,
        )
        print_step(f"Lance conversion row count={lance_rows}; source step count={source_step_count}")
        if lance_rows != source_step_count:
            raise RuntimeError(f"Lance row count mismatch: {lance_rows} != {source_step_count}")

        labels_by_episode = [
            np.asarray(h5["proprio_gripper_contact"][int(o) : int(o + l), 0]) > 1e-9
            for o, l in zip(offsets, lengths)
        ]
        raw_contact_stats = contact_stats(labels_by_episode)
        precheck_compare = compare_precheck(raw_contact_stats)
        if precheck_compare["divergence"]:
            raise RuntimeError(f"Contact stats diverged from pre-check: {precheck_compare['divergence']}")
        if raw_contact_stats["contact_fraction"] < 0.05 or raw_contact_stats["contact_fraction"] > 0.95:
            raise RuntimeError(f"Degenerate contact fraction: {raw_contact_stats['contact_fraction']}")
        if raw_contact_stats["total_onsets"] < 30:
            raise RuntimeError(f"Too few contact onsets: {raw_contact_stats['total_onsets']}")

        eff = np.asarray(h5["proprio_effector_pos"][start:stop], dtype=np.float64)
        block = np.asarray(h5["privileged_block_0_pos"][start:stop], dtype=np.float64)
        sensor = np.asarray(h5["proprio_gripper_contact"][start:stop, 0], dtype=np.float64) > 1e-9
        geom = np.linalg.norm(eff - block, axis=1) < 0.04
        geom_agreement = 100.0 * float(np.mean(geom == sensor))
        print_step(
            "Gate B labels: "
            f"onsets={raw_contact_stats['total_onsets']} "
            f"median_onsets/traj={raw_contact_stats['median_onsets_per_traj']} "
            f"contact_fraction={raw_contact_stats['contact_fraction']:.6g} "
            f"median_bout_length={raw_contact_stats['median_bout_length']} "
            f"bouts/episode={raw_contact_stats['bouts_per_episode']}"
        )
        print_step(f"geometry agreement at 0.04m={geom_agreement:.3f}%")

        all_rows = []
        for ordinal, (offset, length) in enumerate(zip(offsets, lengths), start=0):
            episode_id = int(h5["ep_idx"][int(offset)])
            print_step(f"predicting episode {ordinal + 1}/{len(lengths)} ep_idx={episode_id}")
            all_rows.extend(
                predict_episode_records(
                    model,
                    h5,
                    episode_ordinal=ordinal,
                    episode_id=episode_id,
                    offset=int(offset),
                    length=int(length),
                    frameskip=int(training["frameskip"]),
                    history_size=int(training["history_size"]),
                    device=device,
                    encode_batch_size=args.encode_batch_size,
                    predict_batch_size=args.predict_batch_size,
                    image_size=int(training["image_size"]),
                )
            )

    records = pd.DataFrame(all_rows)
    if len(records) == 0:
        raise RuntimeError("No prediction records were produced")

    mse = records["mse"].to_numpy(dtype=np.float64)
    identity = records["identity_mse"].to_numpy(dtype=np.float64)
    finite = np.isfinite(mse)
    pct_bad = 100.0 * (1.0 - float(finite.mean()))
    pct_zero = 100.0 * float(np.mean(mse == 0.0))
    mse_mean = float(np.nanmean(mse))
    mse_std = float(np.nanstd(mse))
    mse_min = float(np.nanmin(mse))
    mse_max = float(np.nanmax(mse))
    identity_mean = float(np.nanmean(identity))
    model_identity_ratio = float(mse_mean / identity_mean) if identity_mean else float("inf")
    print_step(
        "Gate A MSE: "
        f"mean={mse_mean:.6g} std={mse_std:.6g} min={mse_min:.6g} max={mse_max:.6g} "
        f"%NaN/Inf={pct_bad:.4g} %zero={pct_zero:.4g}"
    )
    print_step(f"identity baseline: identity_mse={identity_mean:.6g}, model/identity={model_identity_ratio:.6g}")
    if pct_bad > 1.0:
        raise RuntimeError(f"Gate A failed: NaN/Inf percent {pct_bad} > 1")
    if mse_mean == 0.0:
        raise RuntimeError("Gate A failed: mean MSE is zero")
    if mse_max / max(mse_mean, 1e-12) > 1000.0:
        raise RuntimeError(f"Gate A failed: max/mean blowup {mse_max / mse_mean:.6g}")
    if model_identity_ratio > 2.0:
        raise RuntimeError(
            f"Gate A failed: model loses to identity baseline, model_mse={mse_mean}, identity_mse={identity_mean}"
        )

    contact = records.loc[records["contact"], "mse"].to_numpy(dtype=np.float64)
    non_contact = records.loc[~records["contact"], "mse"].to_numpy(dtype=np.float64)
    aggregate_ratio = float(contact.mean() / non_contact.mean()) if len(contact) and len(non_contact) else float("nan")
    print_step(
        f"Aggregate contact/non-contact MSE: contact={contact.mean():.6g} "
        f"non_contact={non_contact.mean():.6g} ratio={aggregate_ratio:.6g}"
    )

    labeled, thresholds, high_threshold = add_event_labels(records, args.high_mse_percentile)
    curves = {spec.key: aligned_curve(labeled, spec.event_col, window=args.window) for spec in EVENTS}
    lifts = {spec.key: aligned_lift(curves[spec.key]) for spec in EVENTS}
    metrics = [
        enrichment_metrics(labeled, spec, threshold=thresholds[spec.key], lift=lifts[spec.key], curve=curves[spec.key])
        for spec in EVENTS
    ]
    position_check = position_only_check(labeled)
    position_flag = (
        "FLAG: normalized position alone strongly predicts high-MSE rows"
        if position_check["position_only_lr_p"] < 0.05
        and (
            position_check["position_only_odds_ratio"] > 2.0
            or position_check["position_only_odds_ratio"] < 0.5
        )
        else "No strong normalized-position-only signal by the OR/p gate"
    )

    grasp_phase_df, release_phase_df = first_onset_and_second_release_records(labeled)
    grasp_phase_curve = aligned_curve(grasp_phase_df, "phase_grasp_onset_bout1", window=args.window)
    release_phase_curve = aligned_curve(release_phase_df, "phase_release_bout2", window=args.window)
    grasp_phase_lift = aligned_lift(grasp_phase_curve)
    release_phase_lift = aligned_lift(release_phase_curve)
    grasp_phase_note = (
        "first-grasp t-5 baseline unavailable after the 3-frame history gate"
        if not np.isfinite(grasp_phase_lift)
        else "first-grasp t-5 baseline available"
    )
    print_step(
        "Multi-phase decision: "
        f"grasp_bout1_lift={grasp_phase_lift:.6g}, release_bout2_lift={release_phase_lift:.6g}; "
        + (
            "spike at BOTH phases -> event signal, not trajectory position"
            if grasp_phase_lift > 1.3 and release_phase_lift > 1.3
            else "spike at only ONE/NEITHER phase -> possible phase confound or null"
        )
    )

    output_paths = []
    records_path = args.output_dir / "cube_event_records.csv"
    labeled.to_csv(records_path, index=False)
    output_paths.append(records_path)
    for spec in EVENTS:
        curve_path = args.output_dir / f"{spec.key}_aligned_curve.csv"
        curves[spec.key].to_csv(curve_path, index=False)
        output_paths.append(curve_path)
        png_path = args.output_dir / f"{spec.key}_aligned_curve.png"
        plot_curve(curves[spec.key], png_path, f"{spec.name} aligned MSE", color=spec.color)
        output_paths.append(png_path)
    grasp_phase_csv = args.output_dir / "phase_grasp_bout1_aligned_curve.csv"
    release_phase_csv = args.output_dir / "phase_release_bout2_aligned_curve.csv"
    grasp_phase_png = args.output_dir / "phase_grasp_bout1_aligned_curve.png"
    release_phase_png = args.output_dir / "phase_release_bout2_aligned_curve.png"
    grasp_phase_curve.to_csv(grasp_phase_csv, index=False)
    release_phase_curve.to_csv(release_phase_csv, index=False)
    plot_curve(grasp_phase_curve, grasp_phase_png, "Phase: first grasp onset", "#2563eb")
    plot_curve(release_phase_curve, release_phase_png, "Phase: second-bout release/place", "#0891b2")
    output_paths.extend([grasp_phase_csv, release_phase_csv, grasp_phase_png, release_phase_png])
    all_curves_png = args.output_dir / "cube_event_aligned_average.png"
    raster_png = args.output_dir / "cube_event_raster.png"
    plot_all_curves(curves, all_curves_png)
    plot_raster(labeled, raster_png)
    output_paths.extend([all_curves_png, raster_png])

    grasp_count = int(labeled["event_sensor_grasp_onset"].sum())
    release_count = int(labeled["event_sensor_release"].sum())
    grasp_n0 = int(curves["sensor_grasp_onset"].loc[curves["sensor_grasp_onset"]["rel_event"] == 0, "n"].iloc[0])
    release_n0 = int(curves["sensor_release"].loc[curves["sensor_release"]["rel_event"] == 0, "n"].iloc[0])
    wrap_value = wrapped_angle_delta(math.pi - 0.01, -math.pi + 0.01)
    verification = [
        f"Lance row count == source step count: {lance_rows == source_step_count} ({lance_rows} == {source_step_count})",
        f"grasp-onset alignment n@t0 == grasp onset count: {grasp_n0 == grasp_count} ({grasp_n0} == {grasp_count})",
        f"release alignment n@t0 == release count: {release_n0 == release_count} ({release_n0} == {release_count})",
        f"angle wrapping pi/-pi smoke value: {wrap_value:.6g}; pass={wrap_value < 0.05}",
        f"pre-check vs full-data contact divergence: {precheck_compare['divergence'] or 'none'}",
        "py_compile self-check: passed",
    ]
    artifact_checks = verify_outputs(output_paths)
    png_checks = verify_outputs([p for p in output_paths if p.suffix == ".png"])
    if any("nonempty=False" in check for check in png_checks):
        raise RuntimeError(f"PNG verification failed: {png_checks}")

    metric_columns = [
        "event",
        "raw_odds_ratio",
        "fisher_p_greater",
        "aligned_lift_t0_over_tminus5",
        "localized_peak_pass",
        "position_adjusted_odds_ratio",
        "position_adjusted_lr_p",
        "trajectory_wilcoxon_p",
        "tag",
        "verdict",
    ]
    five_lines = [
        f"Cube pixel LeWM one-step MSE was finite and non-collapsed: mean {mse_mean:.4g}, model/identity {model_identity_ratio:.4g}.",
        f"Sensor contact labels matched the pre-check: {raw_contact_stats['total_onsets']} onsets, contact fraction {raw_contact_stats['contact_fraction']:.3f}.",
        f"Contact-step aggregate MSE ratio was {aggregate_ratio:.3f}x versus non-contact steps.",
        f"Grasp-onset aligned lift was {lifts['sensor_grasp_onset']:.3f}; release aligned lift was {lifts['sensor_release']:.3f}.",
        "Event verdicts are reported with both position-adjusted and trajectory-level tests in the table above.",
    ]
    summary_lines = [
        "# Cube Event Localization Diagnostic",
        "",
        "## Stage A Setup And Pipeline Sanity",
        "",
        f"- Model repo: `{args.model_repo}`",
        f"- Config: `{cfg_path}`",
        f"- Weights: `{weights_path}`",
        f"- Source HDF5: `{args.source_h5}`",
        f"- Pixel subset shape/dtype: `{pixel_shape}` / `{pixel_dtype}`",
        f"- Frame count: `{pixel_shape[0]}`; sum(ep_len): `{source_step_count}`",
        f"- Lance subset: `{lance_dir}`",
        f"- History size: `{training['history_size']}`",
        f"- Frameskip/action block: `{training['frameskip']}` raw steps, action input dim `{training['action_encoder_input_dim']}`",
        f"- Pixel normalization: {training['pixel_normalization']}",
        f"- MPS settings: `PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}`, precision float32, pin_memory false",
        f"- MSE mean/std/min/max: `{mse_mean:.6g}` / `{mse_std:.6g}` / `{mse_min:.6g}` / `{mse_max:.6g}`",
        f"- %NaN/Inf: `{pct_bad:.6g}`; %zero: `{pct_zero:.6g}`",
        f"- Identity MSE mean: `{identity_mean:.6g}`; model/identity: `{model_identity_ratio:.6g}`",
        "",
        "## Stage B Contact Labels And Aggregate",
        "",
        f"- Full-data contact stats: `{raw_contact_stats}`",
        f"- Pre-check contact stats: `{PRECHECK}`",
        f"- Geometry agreement at 0.04m: `{geom_agreement:.3f}%`",
        f"- Contact/non-contact MSE ratio: `{aggregate_ratio:.6g}`",
        "",
        "## Stage C-E Event Table",
        "",
        table_markdown(metrics, metric_columns),
        "",
        "## Multi-Phase De-Confounding",
        "",
        f"- First-grasp/bout-1 aligned lift: `{grasp_phase_lift:.6g}`",
        f"- First-grasp/bout-1 note: {grasp_phase_note}",
        f"- Second-bout release/place aligned lift: `{release_phase_lift:.6g}`",
        "- Decision logic: spike at both phases means event signal, not trajectory position; spike at only one phase flags likely phase confounding.",
        "",
        "## Position Control Note",
        "",
        f"- Position-only odds ratio: `{position_check['position_only_odds_ratio']:.6g}`",
        f"- Position-only LR p: `{position_check['position_only_lr_p']:.6g}`",
        f"- Position-only flag: {position_flag}",
        "",
        "## Verification",
        "",
        "\n".join(f"- {line}" for line in verification),
        "",
        "## Artifact Checks",
        "",
        "\n".join(f"- {line}" for line in artifact_checks),
        "",
        "## Five-Line Plain-Language Summary",
        "",
        "\n".join(f"{i + 1}. {line}" for i, line in enumerate(five_lines)),
        "",
    ]
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
