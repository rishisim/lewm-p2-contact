"""Strict released-LeWM loading and fresh Cube latent extraction."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hdf5plugin  # noqa: F401 - registers filters used by the Cube HDF5 file.
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
LEWM_ROOT = REPO_ROOT / "le-wm"
if str(LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(LEWM_ROOT))

from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelContract:
    latent_dim: int
    history_size: int
    frameskip: int
    raw_action_dim: int
    blocked_action_dim: int
    image_size: int


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _clean_cfg(values: dict) -> dict:
    return {key: value for key, value in values.items() if key != "_target_"}


def _vit_hf_from_config(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    pretrained: bool = False,
    use_mask_token: bool = True,
    **kwargs,
) -> ViTModel:
    if pretrained:
        raise ValueError("Released LeWM must be instantiated without external pretrained weights")
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
    }
    if size not in size_configs:
        raise ValueError(f"Unsupported ViT size {size!r}")
    params = dict(size_configs[size])
    params.update(
        intermediate_size=params["hidden_size"] * 4,
        image_size=image_size,
        patch_size=patch_size,
        **kwargs,
    )
    model = ViTModel(ViTConfig(**params), add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model


def _mlp_from_config(cfg: dict, key: str) -> MLP:
    mlp_cfg = _clean_cfg(cfg[key])
    norm_cfg = mlp_cfg.pop("norm_fn", None)
    norm_fn = None
    if isinstance(norm_cfg, dict):
        target = str(norm_cfg.get("_target_", ""))
        if target.endswith("BatchNorm1d"):
            norm_fn = torch.nn.BatchNorm1d
        elif target.endswith("LayerNorm"):
            norm_fn = torch.nn.LayerNorm
    return MLP(norm_fn=norm_fn, **mlp_cfg)


def _legacy_to_modern(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
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


def _modern_to_legacy(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        if key.startswith("encoder.layers."):
            new_key = key.replace("encoder.layers.", "encoder.encoder.layer.", 1)
            new_key = new_key.replace(".attention.q_proj.", ".attention.attention.query.")
            new_key = new_key.replace(".attention.k_proj.", ".attention.attention.key.")
            new_key = new_key.replace(".attention.v_proj.", ".attention.attention.value.")
            new_key = new_key.replace(".attention.o_proj.", ".attention.output.dense.")
            new_key = new_key.replace(".mlp.fc1.", ".intermediate.dense.")
            new_key = new_key.replace(".mlp.fc2.", ".output.dense.")
        remapped[new_key] = value
    return remapped


def load_frozen_lewm(
    config_path: Path,
    weights_path: Path,
    device: torch.device,
) -> tuple[JEPA, ModelContract, dict[str, object]]:
    """Load the released checkpoint strictly, allowing only known ViT key renames."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model = JEPA(
        encoder=_vit_hf_from_config(**_clean_cfg(cfg["encoder"])),
        predictor=ARPredictor(**_clean_cfg(cfg["predictor"])),
        action_encoder=Embedder(**_clean_cfg(cfg["action_encoder"])),
        projector=_mlp_from_config(cfg, "projector"),
        pred_proj=_mlp_from_config(cfg, "pred_proj"),
    )
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint did not contain a state dictionary")

    expected_keys = set(model.state_dict())
    variants = [
        ("identity", state),
        ("legacy_to_modern", _legacy_to_modern(state)),
        ("modern_to_legacy", _modern_to_legacy(state)),
    ]
    matches = [(name, candidate) for name, candidate in variants if set(candidate) == expected_keys]
    # A rename helper can be an identity when its source prefix is absent. In
    # that case the candidates are tensor-for-tensor identical and the first
    # (least transformed) strict match is unambiguous.
    unique_matches: list[tuple[str, dict[str, torch.Tensor]]] = []
    for name, candidate in matches:
        if not any(
            candidate.keys() == prior.keys()
            and all(candidate[key].data_ptr() == prior[key].data_ptr() for key in candidate)
            for _, prior in unique_matches
        ):
            unique_matches.append((name, candidate))
    if len(unique_matches) != 1:
        diagnostics = {
            name: {
                "missing": sorted(expected_keys - set(candidate))[:8],
                "unexpected": sorted(set(candidate) - expected_keys)[:8],
            }
            for name, candidate in variants
        }
        raise RuntimeError(f"No unique strict checkpoint key match: {diagnostics}")
    key_mapping, matched_state = unique_matches[0]
    result = model.load_state_dict(matched_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    model.eval().requires_grad_(False).to(device)

    raw_action_dim = 5
    blocked_action_dim = int(cfg["action_encoder"]["input_dim"])
    if blocked_action_dim % raw_action_dim:
        raise RuntimeError("Cube action encoder input is not divisible by the five raw action dimensions")
    contract = ModelContract(
        latent_dim=int(cfg["predictor"]["output_dim"]),
        history_size=int(cfg["predictor"]["num_frames"]),
        frameskip=blocked_action_dim // raw_action_dim,
        raw_action_dim=raw_action_dim,
        blocked_action_dim=blocked_action_dim,
        image_size=int(cfg["encoder"]["image_size"]),
    )
    expected = ModelContract(192, 3, 5, 5, 25, 224)
    if contract != expected:
        raise RuntimeError(f"Unexpected released Cube model contract: {contract}; expected {expected}")
    provenance = {
        "config_path": str(config_path.resolve()),
        "weights_path": str(weights_path.resolve()),
        "weights_sha256": sha256_file(weights_path),
        "strict_key_mapping": key_mapping,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "all_parameters_frozen": bool(all(not parameter.requires_grad for parameter in model.parameters())),
        "contract": contract.__dict__,
    }
    return model, contract, provenance


def pixel_transform(pixels: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(pixels), dtype=torch.uint8)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected [frames,H,W,3] uint8 pixels, got {tuple(x.shape)}")
    x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
    return x.to(device)


def global_action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray, int]:
    """Reproduce the fixed dataset-wide normalizer used to train released LeWM."""
    actions = np.asarray(h5["action"][:], dtype=np.float32)
    actions = actions[~np.isnan(actions).any(axis=1)]
    if len(actions) == 0:
        raise RuntimeError("No finite Cube actions")
    mean = actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = actions.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    if np.any(std < 1e-6):
        raise RuntimeError(f"Degenerate Cube action standard deviation: {std}")
    return mean, std, int(len(actions))


def _encode_episode(
    model: JEPA,
    h5: h5py.File,
    offset: int,
    length: int,
    contract: ModelContract,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model_steps = 1 + (length - 1) // contract.frameskip
    obs_rows = offset + np.arange(model_steps, dtype=np.int64) * contract.frameskip
    pixels = h5["pixels"][obs_rows]
    chunks = []
    with torch.inference_mode():
        for start in range(0, model_steps, batch_size):
            batch = pixel_transform(pixels[start : start + batch_size], contract.image_size, device)
            encoded = model.encode({"pixels": batch.unsqueeze(0)})["emb"].squeeze(0)
            chunks.append(encoded.detach().cpu())
    latents = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    if latents.shape != (model_steps, contract.latent_dim):
        raise RuntimeError(f"Encoded latent shape {latents.shape} violates {contract}")
    return latents


def _predict_base(
    model: JEPA,
    histories: np.ndarray,
    action_histories: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(histories), batch_size):
            history = torch.from_numpy(histories[start : start + batch_size]).to(device)
            actions = torch.from_numpy(action_histories[start : start + batch_size]).to(device)
            act_emb = model.action_encoder(actions)
            pred = model.predict(history, act_emb)[:, -1]
            chunks.append(pred.detach().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def deterministic_episode_split(
    total_episodes: int,
    excluded: Iterable[int],
    seed: int,
    train_count: int,
    calibration_count: int,
    test_count: int,
) -> dict[str, np.ndarray]:
    excluded_set = {int(value) for value in excluded}
    eligible = np.asarray([value for value in range(total_episodes) if value not in excluded_set], dtype=np.int64)
    requested = train_count + calibration_count + test_count
    if requested > len(eligible):
        raise ValueError(f"Requested {requested} episodes from {len(eligible)} eligible")
    selected = np.random.default_rng(seed).choice(eligible, size=requested, replace=False)
    splits = {
        "train": selected[:train_count],
        "calibration": selected[train_count : train_count + calibration_count],
        "test": selected[train_count + calibration_count :],
    }
    sets = {name: set(values.tolist()) for name, values in splits.items()}
    if sets["train"] & sets["calibration"] or sets["train"] & sets["test"] or sets["calibration"] & sets["test"]:
        raise AssertionError("Episode split overlap")
    if any(values & excluded_set for values in sets.values()):
        raise AssertionError("Excluded episode entered split")
    return splits


def extract_fresh_cube_cache(
    *,
    source_h5: Path,
    config_path: Path,
    weights_path: Path,
    output_npz: Path,
    split_manifest_path: Path,
    split_episodes: dict[str, np.ndarray],
    device: torch.device,
    encode_batch_size: int = 64,
    predict_batch_size: int = 512,
) -> dict[str, object]:
    """Encode fresh episodes once and cache only causal inputs, targets, and labels."""
    model, contract, model_provenance = load_frozen_lewm(config_path, weights_path, device)
    split_code_by_name = {"train": 0, "calibration": 1, "test": 2}
    split_by_episode = {
        int(episode): name for name, values in split_episodes.items() for episode in np.asarray(values).tolist()
    }
    if len(split_by_episode) != sum(len(values) for values in split_episodes.values()):
        raise AssertionError("Split episodes are not unique")

    history_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    episode_chunks: list[np.ndarray] = []
    step_chunks: list[np.ndarray] = []
    split_chunks: list[np.ndarray] = []
    interaction_chunks: list[np.ndarray] = []
    impact_chunks: list[np.ndarray] = []
    effector_disp_chunks: list[np.ndarray] = []
    block_disp_chunks: list[np.ndarray] = []
    phase_chunks: list[np.ndarray] = []
    trajectory_records: list[dict[str, object]] = []

    start_time = time.perf_counter()
    with h5py.File(source_h5, "r", swmr=True) as h5:
        total_episodes = len(h5["ep_offset"])
        if max(split_by_episode) >= total_episodes:
            raise ValueError("Requested episode outside Cube source")
        action_mean, action_std, action_stats_rows = global_action_stats(h5)
        for episode in sorted(split_by_episode):
            offset = int(h5["ep_offset"][episode])
            length = int(h5["ep_len"][episode])
            if length != 201:
                raise RuntimeError(f"Cube episode {episode} has unexpected length {length}")
            source_episode = np.asarray(h5["ep_idx"][offset : offset + length], dtype=np.int64)
            if not np.all(source_episode == episode):
                raise RuntimeError(f"Cube episode index mismatch at ordinal {episode}")

            latents = _encode_episode(model, h5, offset, length, contract, device, encode_batch_size)
            model_steps = len(latents)
            action_rows = (model_steps - 1) * contract.frameskip
            actions_raw = np.asarray(h5["action"][offset : offset + action_rows], dtype=np.float32)
            if np.isnan(actions_raw).any():
                raise RuntimeError(f"NaN action entered episode {episode} prediction window")
            normalized = (actions_raw - action_mean.reshape(1, -1)) / action_std.reshape(1, -1)
            action_blocks = normalized.reshape(model_steps - 1, contract.blocked_action_dim).astype(np.float32)

            n_examples = model_steps - contract.history_size
            histories = np.stack(
                [latents[index : index + contract.history_size] for index in range(n_examples)], axis=0
            )
            action_histories = np.stack(
                [action_blocks[index : index + contract.history_size] for index in range(n_examples)], axis=0
            )
            targets = latents[contract.history_size :]
            target_steps = np.arange(contract.history_size, model_steps, dtype=np.int64)

            raw_contact = np.asarray(
                h5["proprio_gripper_contact"][offset : offset + length, 0], dtype=np.float64
            ) > 1e-9
            interaction = []
            impact = []
            effector_disp = []
            block_disp = []
            for target_step in target_steps:
                raw_target = int(target_step * contract.frameskip)
                raw_previous = raw_target - contract.frameskip
                current_block_contact = bool(raw_contact[raw_previous + 1 : raw_target + 1].any())
                prior_start = max(0, raw_previous - contract.frameskip + 1)
                prior_block_contact = bool(raw_contact[prior_start : raw_previous + 1].any())
                interaction.append(current_block_contact)
                impact.append(current_block_contact and not prior_block_contact)
                eff_now = np.asarray(h5["proprio_effector_pos"][offset + raw_target], dtype=np.float64)
                eff_prev = np.asarray(h5["proprio_effector_pos"][offset + raw_previous], dtype=np.float64)
                block_now = np.asarray(h5["privileged_block_0_pos"][offset + raw_target], dtype=np.float64)
                block_prev = np.asarray(h5["privileged_block_0_pos"][offset + raw_previous], dtype=np.float64)
                effector_disp.append(float(np.linalg.norm(eff_now - eff_prev)))
                block_disp.append(float(np.linalg.norm(block_now - block_prev)))

            split_name = split_by_episode[episode]
            history_chunks.append(histories.astype(np.float32))
            action_chunks.append(action_histories.astype(np.float32))
            target_chunks.append(targets.astype(np.float32))
            episode_chunks.append(np.full(n_examples, episode, dtype=np.int64))
            step_chunks.append(target_steps)
            split_chunks.append(np.full(n_examples, split_code_by_name[split_name], dtype=np.int8))
            interaction_chunks.append(np.asarray(interaction, dtype=np.bool_))
            impact_chunks.append(np.asarray(impact, dtype=np.bool_))
            effector_disp_chunks.append(np.asarray(effector_disp, dtype=np.float32))
            block_disp_chunks.append(np.asarray(block_disp, dtype=np.float32))
            phase_chunks.append((target_steps / max(model_steps - 1, 1)).astype(np.float32))
            trajectory_records.append(
                {
                    "episode_id": episode,
                    "split": split_name,
                    "raw_length": length,
                    "model_steps": model_steps,
                    "examples": n_examples,
                }
            )

    history = np.concatenate(history_chunks)
    action = np.concatenate(action_chunks)
    target = np.concatenate(target_chunks)
    episode_id = np.concatenate(episode_chunks)
    model_step = np.concatenate(step_chunks)
    split = np.concatenate(split_chunks)
    base_pred = _predict_base(model, history, action, device, predict_batch_size)
    synchronize(device)
    elapsed = time.perf_counter() - start_time

    expected_shapes = {
        "history": (len(target), contract.history_size, contract.latent_dim),
        "action": (len(target), contract.history_size, contract.blocked_action_dim),
        "base_pred": (len(target), contract.latent_dim),
        "target": (len(target), contract.latent_dim),
    }
    arrays = {"history": history, "action": action, "base_pred": base_pred, "target": target}
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape or not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"Invalid {name}: shape={arrays[name].shape}, expected={expected_shape}")
    keys = np.stack([episode_id, model_step], axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise RuntimeError("Duplicate episode/model-step key")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        history=history,
        action=action,
        base_pred=base_pred,
        target=target,
        episode_id=episode_id,
        model_step=model_step,
        split=split,
        interaction=np.concatenate(interaction_chunks),
        impact=np.concatenate(impact_chunks),
        effector_disp=np.concatenate(effector_disp_chunks),
        block_disp=np.concatenate(block_disp_chunks),
        normalized_phase=np.concatenate(phase_chunks),
        action_mean=action_mean,
        action_std=action_std,
    )
    manifest = {
        "selection": {
            "seed": 260713,
            "excluded_episode_ordinals": list(range(30)),
            "split_episode_ordinals": {
                name: [int(value) for value in np.asarray(values).tolist()]
                for name, values in split_episodes.items()
            },
        },
        "source": {
            "h5_path": str(source_h5.resolve()),
            "h5_size_bytes": int(source_h5.stat().st_size),
            "episode_count": 10000,
            "row_count": 2010000,
        },
        "model": model_provenance,
        "preprocessing": {
            "pixel_normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "action_normalization": "released-training dataset-wide finite action mean/std (unbiased std)",
            "action_stats_rows": action_stats_rows,
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
        },
        "cache": {
            "path": str(output_npz.resolve()),
            "examples": int(len(target)),
            "shapes": {name: list(array.shape) for name, array in arrays.items()},
            "split_examples": {
                name: int(np.sum(split == code)) for name, code in split_code_by_name.items()
            },
            "elapsed_seconds": elapsed,
        },
        "trajectory_records": trajectory_records,
    }
    split_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
