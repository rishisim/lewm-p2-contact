"""Strict frozen-model loading and causality-preserving Cube extraction for V2.

The primary cache intentionally contains only latent/action histories, the
frozen base prediction, targets used later for scoring, and row identifiers.
Physical signals are read by :func:`extract_post_prediction_labels` in a
separate, explicitly post-prediction step.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hdf5plugin  # noqa: F401 - register Cube HDF5 compression filters.
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
FROZEN_ACTION_MEAN = np.asarray(
    [
        0.010884696617722511,
        -0.003141433000564575,
        0.002646582666784525,
        0.00042392866453155875,
        0.1592525690793991,
    ],
    dtype=np.float32,
)
FROZEN_ACTION_STD = np.asarray(
    [
        0.28941991925239563,
        0.39371708035469055,
        0.6431366801261902,
        0.3928017318248749,
        0.25030744075775146,
    ],
    dtype=np.float32,
)
FROZEN_ACTION_STATS_ROWS = 2_000_000

EXPECTED_BASE_CONFIG_SHA256 = "4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999"
EXPECTED_BASE_WEIGHTS_SHA256 = "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89"
EXPECTED_REFINER_SHA256 = "388a82fc30c96921083bfa4f2578cb296e6c3544510953d17578cf766cdf10f1"
EXPECTED_V1_FULL_MANIFEST_SHA256 = "3cc9bd4d7df229b0a4111a49933e98c5b72e916b5b5f98d7d5e023fc30ed544d"
EXPECTED_V1_SMOKE_MANIFEST_SHA256 = "f5a4da6c0f37d6222e83085327e30a1b70f0ace6f96e9c97d37c1067d7b65940"

V2_SELECTION_SEED = 260813
DIAGNOSTIC_EPISODES = tuple(range(30))
SMOKE_EPISODES = tuple(range(36, 42))
SPLIT_NAMES = ("train", "calibration", "test")
SPLIT_CODES = {"train": 0, "calibration": 1, "test": 2}


@dataclass(frozen=True)
class ModelContract:
    latent_dim: int
    history_size: int
    frameskip: int
    raw_action_dim: int
    blocked_action_dim: int
    image_size: int


EXPECTED_MODEL_CONTRACT = ModelContract(192, 3, 5, 5, 25, 224)


@dataclass(frozen=True)
class FrozenModuleAudit:
    state_sha256: str
    parameter_count: int
    buffer_count: int
    training: bool
    all_parameters_frozen: bool
    no_parameter_gradients: bool


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected_sha256: str, *, label: str = "file") -> str:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch for {Path(path).resolve()}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash all named parameters and buffers in a device-independent order."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def audit_frozen_module(module: torch.nn.Module) -> FrozenModuleAudit:
    return FrozenModuleAudit(
        state_sha256=module_state_sha256(module),
        parameter_count=int(sum(parameter.numel() for parameter in module.parameters())),
        buffer_count=int(sum(buffer.numel() for buffer in module.buffers())),
        training=bool(module.training),
        all_parameters_frozen=bool(all(not parameter.requires_grad for parameter in module.parameters())),
        no_parameter_gradients=bool(all(parameter.grad is None for parameter in module.parameters())),
    )


def assert_frozen_module(
    module: torch.nn.Module,
    *,
    expected: FrozenModuleAudit | None = None,
    label: str = "module",
) -> FrozenModuleAudit:
    audit = audit_frozen_module(module)
    if audit.training:
        raise RuntimeError(f"Frozen {label} is in training mode")
    if not audit.all_parameters_frozen:
        raise RuntimeError(f"Frozen {label} has parameters requiring gradients")
    if not audit.no_parameter_gradients:
        raise RuntimeError(f"Frozen {label} accumulated parameter gradients")
    if expected is not None and audit != expected:
        raise RuntimeError(f"Frozen {label} audit changed: before={expected}, after={audit}")
    return audit


def verify_byte_identical_files(left: Path, right: Path) -> str:
    left_hash = sha256_file(left)
    right_hash = sha256_file(right)
    if left_hash != right_hash or Path(left).stat().st_size != Path(right).stat().st_size:
        raise RuntimeError(f"Files are not byte-identical: {left} vs {right}")
    return left_hash


def _strict_int_list(values: object, *, label: str, total_episodes: int) -> list[int]:
    if not isinstance(values, list):
        raise RuntimeError(f"{label} must be a JSON list")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"{label} contains a non-integer ordinal: {value!r}")
        if not 0 <= value < total_episodes:
            raise RuntimeError(f"{label} contains out-of-range ordinal {value}")
        parsed.append(value)
    if len(parsed) != len(set(parsed)):
        raise RuntimeError(f"{label} contains duplicate episode ordinals")
    return parsed


def parse_prior_episode_manifest(
    path: Path,
    *,
    expected_sha256: str,
    total_episodes: int = 10_000,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Read one frozen V1 manifest and strictly parse its three episode splits."""
    manifest_hash = verify_file_hash(path, expected_sha256, label="prior manifest")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_splits = payload["selection"]["split_episode_ordinals"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Malformed prior episode manifest {path}") from exc
    if not isinstance(raw_splits, dict) or set(raw_splits) != set(SPLIT_NAMES):
        raise RuntimeError(f"Prior manifest {path} must contain exactly splits {SPLIT_NAMES}")
    splits = {
        name: _strict_int_list(
            raw_splits[name], label=f"{Path(path).name}:{name}", total_episodes=total_episodes
        )
        for name in SPLIT_NAMES
    }
    flattened = [episode for name in SPLIT_NAMES for episode in splits[name]]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError(f"Prior manifest {path} has cross-split episode overlap")
    provenance = {
        "path": str(Path(path).resolve()),
        "sha256": manifest_hash,
        "split_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "episode_count": len(flattened),
    }
    return splits, provenance


def load_prior_episode_exclusions(
    full_manifest_path: Path,
    smoke_manifest_path: Path,
    *,
    full_manifest_sha256: str = EXPECTED_V1_FULL_MANIFEST_SHA256,
    smoke_manifest_sha256: str = EXPECTED_V1_SMOKE_MANIFEST_SHA256,
    total_episodes: int = 10_000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the frozen 636-episode V2 prior-exclusion union and provenance."""
    full, full_provenance = parse_prior_episode_manifest(
        full_manifest_path,
        expected_sha256=full_manifest_sha256,
        total_episodes=total_episodes,
    )
    smoke, smoke_provenance = parse_prior_episode_manifest(
        smoke_manifest_path,
        expected_sha256=smoke_manifest_sha256,
        total_episodes=total_episodes,
    )
    full_set = {value for values in full.values() for value in values}
    smoke_set = {value for values in smoke.values() for value in values}
    diagnostic_set = set(DIAGNOSTIC_EPISODES)
    if len(full_set) != 600:
        raise RuntimeError(f"Frozen V1 full manifest contains {len(full_set)} episodes, expected 600")
    if smoke_set != set(range(30, 36)):
        raise RuntimeError(f"Frozen V1 smoke manifest episodes changed: {sorted(smoke_set)}")
    if full_set & (diagnostic_set | smoke_set):
        raise RuntimeError("V1 full episodes overlap diagnostic or V1 smoke episodes")
    union = np.asarray(sorted(diagnostic_set | smoke_set | full_set), dtype=np.int64)
    if len(union) != 636:
        raise RuntimeError(f"Prior exclusion union has {len(union)} episodes, expected 636")
    provenance = {
        "diagnostic_episode_ordinals": list(DIAGNOSTIC_EPISODES),
        "full_manifest": full_provenance,
        "smoke_manifest": smoke_provenance,
        "union_count": int(len(union)),
        "union_sha256": hashlib.sha256(union.tobytes()).hexdigest(),
    }
    return union, provenance


def deterministic_episode_split(
    total_episodes: int,
    excluded: Iterable[int],
    seed: int,
    train_count: int,
    calibration_count: int,
    test_count: int,
    *,
    reserved: Iterable[int] = (),
) -> dict[str, np.ndarray]:
    """Select disjoint episodes while preserving NumPy RNG return order."""
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    excluded_set = {int(value) for value in excluded}
    reserved_set = {int(value) for value in reserved}
    if any(value < 0 or value >= total_episodes for value in excluded_set | reserved_set):
        raise ValueError("Excluded/reserved episode outside source range")
    if any(
        isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0
        for count in (train_count, calibration_count, test_count)
    ):
        raise ValueError("Split counts must be nonnegative integers")
    forbidden = excluded_set | reserved_set
    eligible = np.asarray(
        [value for value in range(total_episodes) if value not in forbidden], dtype=np.int64
    )
    requested = train_count + calibration_count + test_count
    if requested > len(eligible):
        raise ValueError(f"Requested {requested} episodes from {len(eligible)} eligible")
    selected = np.random.default_rng(seed).choice(eligible, size=requested, replace=False)
    splits = {
        "train": selected[:train_count],
        "calibration": selected[train_count : train_count + calibration_count],
        "test": selected[train_count + calibration_count :],
    }
    validate_split_episodes(splits, total_episodes=total_episodes, forbidden=forbidden)
    return splits


def validate_split_episodes(
    split_episodes: Mapping[str, Iterable[int]],
    *,
    total_episodes: int,
    forbidden: Iterable[int] = (),
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, np.ndarray]:
    if set(split_episodes) != set(SPLIT_NAMES):
        raise ValueError(f"Splits must be exactly {SPLIT_NAMES}")
    normalized: dict[str, np.ndarray] = {}
    for name in SPLIT_NAMES:
        raw_values = list(split_episodes[name])
        values = np.asarray(raw_values, dtype=np.int64) if not raw_values else np.asarray(raw_values)
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(f"Split {name} must contain one-dimensional integer ordinals")
        values = values.astype(np.int64, copy=False)
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Split {name} contains duplicates")
        if np.any((values < 0) | (values >= total_episodes)):
            raise ValueError(f"Split {name} contains out-of-range episodes")
        if expected_counts is not None and len(values) != int(expected_counts[name]):
            raise ValueError(f"Split {name} has {len(values)} episodes, expected {expected_counts[name]}")
        normalized[name] = values
    sets = {name: set(values.tolist()) for name, values in normalized.items()}
    if any(sets[left] & sets[right] for index, left in enumerate(SPLIT_NAMES) for right in SPLIT_NAMES[index + 1 :]):
        raise ValueError("Episode splits overlap")
    forbidden_set = {int(value) for value in forbidden}
    if any(sets[name] & forbidden_set for name in SPLIT_NAMES):
        raise ValueError("Forbidden episode entered split")
    return normalized


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
    *,
    expected_config_sha256: str | None = None,
    expected_weights_sha256: str | None = None,
) -> tuple[JEPA, ModelContract, dict[str, object]]:
    """Strictly load a released LeWM and freeze every parameter."""
    config_hash = sha256_file(config_path)
    weights_hash = sha256_file(weights_path)
    if expected_config_sha256 is not None:
        verify_file_hash(config_path, expected_config_sha256, label="base config")
    if expected_weights_sha256 is not None:
        verify_file_hash(weights_path, expected_weights_sha256, label="base weights")
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = JEPA(
        encoder=_vit_hf_from_config(**_clean_cfg(cfg["encoder"])),
        predictor=ARPredictor(**_clean_cfg(cfg["predictor"])),
        action_encoder=Embedder(**_clean_cfg(cfg["action_encoder"])),
        projector=_mlp_from_config(cfg, "projector"),
        pred_proj=_mlp_from_config(cfg, "pred_proj"),
    )
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise TypeError("Checkpoint did not contain a string-keyed state dictionary")

    expected_keys = set(model.state_dict())
    variants = [
        ("identity", state),
        ("legacy_to_modern", _legacy_to_modern(state)),
        ("modern_to_legacy", _modern_to_legacy(state)),
    ]
    matches = [(name, candidate) for name, candidate in variants if set(candidate) == expected_keys]
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
            f"Strict checkpoint mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.eval().requires_grad_(False).to(device)

    raw_action_dim = 5
    blocked_action_dim = int(cfg["action_encoder"]["input_dim"])
    if blocked_action_dim % raw_action_dim:
        raise RuntimeError("Cube action encoder input is not divisible by five raw action dimensions")
    contract = ModelContract(
        latent_dim=int(cfg["predictor"]["output_dim"]),
        history_size=int(cfg["predictor"]["num_frames"]),
        frameskip=blocked_action_dim // raw_action_dim,
        raw_action_dim=raw_action_dim,
        blocked_action_dim=blocked_action_dim,
        image_size=int(cfg["encoder"]["image_size"]),
    )
    if contract != EXPECTED_MODEL_CONTRACT:
        raise RuntimeError(f"Unexpected released Cube model contract: {contract}")
    audit = assert_frozen_module(model, label="base LeWM")
    if audit.parameter_count != 18_034_628:
        raise RuntimeError(f"Unexpected base parameter count {audit.parameter_count}")
    provenance = {
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_hash,
        "weights_path": str(Path(weights_path).resolve()),
        "weights_sha256": weights_hash,
        "strict_key_mapping": key_mapping,
        "contract": asdict(contract),
        "frozen_audit": asdict(audit),
    }
    return model, contract, provenance


def load_frozen_refiner(
    refiner: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    *,
    expected_sha256: str = EXPECTED_REFINER_SHA256,
    expected_seed: int = 260713,
    expected_parameter_count: int | None = 335_360,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Strictly load the V1-selected shared refiner into a supplied V2 module."""
    checkpoint_hash = verify_file_hash(checkpoint_path, expected_sha256, label="refiner checkpoint")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise RuntimeError("Refiner checkpoint must contain a state_dict")
    if payload.get("seed") != expected_seed:
        raise RuntimeError(
            f"Refiner checkpoint seed mismatch: expected {expected_seed}, got {payload.get('seed')}"
        )
    state = payload["state_dict"]
    expected_keys = set(refiner.state_dict())
    if set(state) != expected_keys:
        raise RuntimeError(
            "Strict refiner checkpoint mismatch: "
            f"missing={sorted(expected_keys - set(state))}, unexpected={sorted(set(state) - expected_keys)}"
        )
    result = refiner.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Strict refiner load unexpectedly reported key differences")
    refiner.eval().requires_grad_(False).to(device)
    audit = assert_frozen_module(refiner, label="shared refiner")
    if expected_parameter_count is not None and audit.parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Unexpected refiner parameter count {audit.parameter_count}; expected {expected_parameter_count}"
        )
    provenance = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_seed": int(payload["seed"]),
        "strict_key_match": True,
        "frozen_audit": asdict(audit),
    }
    return refiner, provenance


def pixel_transform(pixels: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(pixels), dtype=torch.uint8)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected [frames,H,W,3] pixels, got {tuple(x.shape)}")
    x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.shape[-2:] != (image_size, image_size):
        x = F.interpolate(
            x, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True
        )
    return x.to(device)


def global_action_stats(h5: h5py.File) -> tuple[np.ndarray, np.ndarray, int]:
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
            prediction = model.predict(history, model.action_encoder(actions))[:, -1]
            chunks.append(prediction.detach().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def validate_cache_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    split_episodes: Mapping[str, Iterable[int]],
    contract: ModelContract,
    expected_examples_per_episode: int = 38,
) -> dict[str, object]:
    """Audit causal-cache shape, finiteness, keys, temporal order, and split mapping."""
    required = {"history", "action", "base_pred", "target", "episode_id", "model_step", "split"}
    if set(arrays) != required:
        raise RuntimeError(f"Cache arrays must be exactly {sorted(required)}; got {sorted(arrays)}")
    n_rows = len(np.asarray(arrays["target"]))
    expected_shapes = {
        "history": (n_rows, contract.history_size, contract.latent_dim),
        "action": (n_rows, contract.history_size, contract.blocked_action_dim),
        "base_pred": (n_rows, contract.latent_dim),
        "target": (n_rows, contract.latent_dim),
        "episode_id": (n_rows,),
        "model_step": (n_rows,),
        "split": (n_rows,),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise RuntimeError(f"Invalid {name} shape {value.shape}; expected {shape}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RuntimeError(f"Cache array {name} contains nonfinite values")
    episode_id = np.asarray(arrays["episode_id"], dtype=np.int64)
    model_step = np.asarray(arrays["model_step"], dtype=np.int64)
    split_codes = np.asarray(arrays["split"], dtype=np.int64)
    keys = np.stack([episode_id, model_step], axis=1)
    if len(np.unique(keys, axis=0)) != n_rows:
        raise RuntimeError("Duplicate episode/model-step cache key")
    planned_values = [int(value) for values in split_episodes.values() for value in values]
    validation_total_episodes = max(
        (int(episode_id.max()) + 1) if n_rows else 1,
        (max(planned_values) + 1) if planned_values else 1,
    )
    normalized_splits = validate_split_episodes(
        split_episodes,
        total_episodes=validation_total_episodes,
    )
    expected_episode_to_split = {
        int(episode): SPLIT_CODES[name]
        for name, values in normalized_splits.items()
        for episode in values.tolist()
    }
    if set(episode_id.tolist()) != set(expected_episode_to_split):
        raise RuntimeError("Extracted episode set differs from planned split set")
    for episode, expected_code in expected_episode_to_split.items():
        indices = np.flatnonzero(episode_id == episode)
        if len(indices) != expected_examples_per_episode:
            raise RuntimeError(
                f"Episode {episode} has {len(indices)} examples, expected {expected_examples_per_episode}"
            )
        if not np.all(split_codes[indices] == expected_code):
            raise RuntimeError(f"Episode {episode} has incorrect split code")
        expected_steps = np.arange(
            contract.history_size,
            contract.history_size + expected_examples_per_episode,
            dtype=np.int64,
        )
        if not np.array_equal(model_step[indices], expected_steps):
            raise RuntimeError(f"Episode {episode} cache rows are not in expected temporal order")
    return {
        "examples": n_rows,
        "episode_count": len(expected_episode_to_split),
        "split_examples": {
            name: int(np.sum(split_codes == SPLIT_CODES[name])) for name in SPLIT_NAMES
        },
        "shapes": {name: list(np.asarray(value).shape) for name, value in arrays.items()},
    }


def extract_fresh_cube_cache(
    *,
    source_h5: Path,
    config_path: Path,
    weights_path: Path,
    output_npz: Path,
    split_manifest_path: Path,
    split_episodes: Mapping[str, Iterable[int]],
    selection_seed: int,
    excluded_episode_ordinals: Iterable[int],
    reserved_episode_ordinals: Iterable[int] = (),
    prior_manifest_provenance: Mapping[str, object] | None = None,
    device: torch.device,
    encode_batch_size: int = 64,
    predict_batch_size: int = 512,
    expected_source_episodes: int = 10_000,
    expected_source_rows: int = 2_010_000,
) -> dict[str, object]:
    """Encode planned Cube episodes without reading any physical-label dataset."""
    forbidden = sorted(
        {int(value) for value in excluded_episode_ordinals}
        | {int(value) for value in reserved_episode_ordinals}
    )
    split_episodes = validate_split_episodes(
        split_episodes,
        total_episodes=expected_source_episodes,
        forbidden=forbidden,
    )
    model, contract, model_provenance = load_frozen_lewm(
        config_path,
        weights_path,
        device,
        expected_config_sha256=EXPECTED_BASE_CONFIG_SHA256,
        expected_weights_sha256=EXPECTED_BASE_WEIGHTS_SHA256,
    )
    model_audit_before = assert_frozen_module(model, label="base LeWM")
    file_hashes_before = {
        "config": sha256_file(config_path),
        "weights": sha256_file(weights_path),
    }
    split_by_episode = {
        int(episode): name for name, values in split_episodes.items() for episode in values.tolist()
    }

    history_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    episode_chunks: list[np.ndarray] = []
    step_chunks: list[np.ndarray] = []
    split_chunks: list[np.ndarray] = []
    trajectory_records: list[dict[str, object]] = []

    start_time = time.perf_counter()
    with h5py.File(source_h5, "r", swmr=True) as h5:
        total_episodes = len(h5["ep_offset"])
        if total_episodes != expected_source_episodes or len(h5["ep_idx"]) != expected_source_rows:
            raise RuntimeError(
                f"Unexpected Cube layout: episodes={total_episodes}, rows={len(h5['ep_idx'])}"
            )
        # Frozen released/V1 preprocessing constants. Do not refit any
        # normalization statistic after the V2 split is selected.
        action_mean = FROZEN_ACTION_MEAN.copy()
        action_std = FROZEN_ACTION_STD.copy()
        action_stats_rows = FROZEN_ACTION_STATS_ROWS
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
            actions_raw = np.asarray(
                h5["action"][offset : offset + (model_steps - 1) * contract.frameskip],
                dtype=np.float32,
            )
            if np.isnan(actions_raw).any():
                raise RuntimeError(f"NaN action entered episode {episode} prediction window")
            normalized = (actions_raw - action_mean.reshape(1, -1)) / action_std.reshape(1, -1)
            action_blocks = normalized.reshape(model_steps - 1, contract.blocked_action_dim).astype(np.float32)
            n_examples = model_steps - contract.history_size
            histories = np.stack(
                [latents[index : index + contract.history_size] for index in range(n_examples)]
            ).astype(np.float32)
            action_histories = np.stack(
                [action_blocks[index : index + contract.history_size] for index in range(n_examples)]
            ).astype(np.float32)
            targets = latents[contract.history_size :].astype(np.float32)
            target_steps = np.arange(contract.history_size, model_steps, dtype=np.int64)
            split_name = split_by_episode[episode]
            history_chunks.append(histories)
            action_chunks.append(action_histories)
            target_chunks.append(targets)
            episode_chunks.append(np.full(n_examples, episode, dtype=np.int64))
            step_chunks.append(target_steps)
            split_chunks.append(np.full(n_examples, SPLIT_CODES[split_name], dtype=np.int8))
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
    elapsed_seconds = time.perf_counter() - start_time

    arrays = {
        "history": history,
        "action": action,
        "base_pred": base_pred,
        "target": target,
        "episode_id": episode_id,
        "model_step": model_step,
        "split": split,
    }
    cache_audit = validate_cache_arrays(
        arrays,
        split_episodes=split_episodes,
        contract=contract,
        expected_examples_per_episode=38,
    )
    model_audit_after = assert_frozen_module(
        model, expected=model_audit_before, label="base LeWM"
    )
    file_hashes_after = {
        "config": sha256_file(config_path),
        "weights": sha256_file(weights_path),
    }
    if file_hashes_after != file_hashes_before:
        raise RuntimeError("Frozen base config or weights changed during extraction")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        **arrays,
        action_mean=action_mean,
        action_std=action_std,
    )
    manifest = {
        "selection": {
            "seed": int(selection_seed),
            "excluded_episode_ordinals": sorted({int(value) for value in excluded_episode_ordinals}),
            "reserved_episode_ordinals": sorted({int(value) for value in reserved_episode_ordinals}),
            "split_episode_ordinals": {
                name: [int(value) for value in split_episodes[name].tolist()] for name in SPLIT_NAMES
            },
            "prior_manifests": dict(prior_manifest_provenance or {}),
        },
        "source": {
            "h5_path": str(Path(source_h5).resolve()),
            "h5_size_bytes": int(Path(source_h5).stat().st_size),
            "episode_count": expected_source_episodes,
            "row_count": expected_source_rows,
            "base_pretraining_episode_membership": "unknown",
        },
        "model": model_provenance,
        "freeze_audit": {
            "before": asdict(model_audit_before),
            "after": asdict(model_audit_after),
            "file_hashes_before": file_hashes_before,
            "file_hashes_after": file_hashes_after,
        },
        "preprocessing": {
            "pixel_normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "action_normalization": "frozen released/V1 dataset-wide finite action mean/std (unbiased std); not refit on V2",
            "action_normalization_provenance": "runs/lewm_adaptive_compute_v1/cache/split_manifest.json and fresh_cube_inputs.npz",
            "action_stats_rows": action_stats_rows,
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
        },
        "cache": {
            "path": str(Path(output_npz).resolve()),
            "sha256": sha256_file(output_npz),
            "contains_physical_labels": False,
            "elapsed_seconds": elapsed_seconds,
            **cache_audit,
        },
        "trajectory_records": trajectory_records,
    }
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return manifest


def extract_post_prediction_labels(
    source_h5: Path,
    episode_id: np.ndarray,
    model_step: np.ndarray,
    *,
    frameskip: int = 5,
) -> dict[str, np.ndarray]:
    """Read post-hoc physical signals for already-frozen prediction row keys.

    This function is deliberately separate from :func:`extract_fresh_cube_cache`.
    It returns sensor contact/impact and displacement signals; calibration-fit
    motion thresholds used to derive ``transport_free`` and ``static`` remain a
    downstream analysis responsibility.
    """
    episode_id = np.asarray(episode_id)
    model_step = np.asarray(model_step)
    if episode_id.ndim != 1 or model_step.shape != episode_id.shape:
        raise ValueError("episode_id and model_step must be matching one-dimensional arrays")
    if episode_id.dtype.kind not in "iu" or model_step.dtype.kind not in "iu":
        raise ValueError("episode_id and model_step must be integer arrays")
    if frameskip <= 0:
        raise ValueError("frameskip must be positive")
    keys = np.stack([episode_id, model_step], axis=1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Post-prediction label keys must be unique")

    interaction = np.empty(len(keys), dtype=np.bool_)
    impact = np.empty(len(keys), dtype=np.bool_)
    effector_disp = np.empty(len(keys), dtype=np.float32)
    block_disp = np.empty(len(keys), dtype=np.float32)
    normalized_phase = np.empty(len(keys), dtype=np.float32)
    with h5py.File(source_h5, "r", swmr=True) as h5:
        total_episodes = len(h5["ep_offset"])
        for episode in np.unique(episode_id.astype(np.int64)):
            if episode < 0 or episode >= total_episodes:
                raise ValueError(f"Episode {episode} outside Cube source")
            indices = np.flatnonzero(episode_id == episode)
            offset = int(h5["ep_offset"][episode])
            length = int(h5["ep_len"][episode])
            if not np.all(np.asarray(h5["ep_idx"][offset : offset + length]) == episode):
                raise RuntimeError(f"Cube episode index mismatch at ordinal {episode}")
            raw_contact = np.asarray(
                h5["proprio_gripper_contact"][offset : offset + length, 0], dtype=np.float64
            ) > 1e-9
            max_model_step = (length - 1) // frameskip
            for index in indices:
                step = int(model_step[index])
                raw_target = step * frameskip
                raw_previous = raw_target - frameskip
                if step <= 0 or raw_target >= length:
                    raise ValueError(f"Invalid model step {step} for episode {episode}")
                current_contact = bool(raw_contact[raw_previous + 1 : raw_target + 1].any())
                prior_start = max(0, raw_previous - frameskip + 1)
                prior_contact = bool(raw_contact[prior_start : raw_previous + 1].any())
                interaction[index] = current_contact
                impact[index] = current_contact and not prior_contact
                eff_now = np.asarray(h5["proprio_effector_pos"][offset + raw_target], dtype=np.float64)
                eff_prev = np.asarray(h5["proprio_effector_pos"][offset + raw_previous], dtype=np.float64)
                block_now = np.asarray(h5["privileged_block_0_pos"][offset + raw_target], dtype=np.float64)
                block_prev = np.asarray(h5["privileged_block_0_pos"][offset + raw_previous], dtype=np.float64)
                effector_disp[index] = np.linalg.norm(eff_now - eff_prev)
                block_disp[index] = np.linalg.norm(block_now - block_prev)
                normalized_phase[index] = step / max(max_model_step, 1)
    return {
        "interaction": interaction,
        "impact": impact,
        "effector_disp": effector_disp,
        "block_disp": block_disp,
        "normalized_phase": normalized_phase,
    }
