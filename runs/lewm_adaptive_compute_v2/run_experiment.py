#!/usr/bin/env python3
"""Run the frozen LeWM V2 sequential causal-gate experiment.

The ordering in :func:`main` is part of the preregistration.  In particular,
the final-test gate features, predictions, and exact-budget allocation are
frozen before final-test targets are read for scoring.  The target-informed
oracle is constructed only after that scoring boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import allocation as allocation_lib  # noqa: E402
import gate as gate_lib  # noqa: E402
import integrity as integrity_lib  # noqa: E402
import model_io as model_io_lib  # noqa: E402
import plots as plots_lib  # noqa: E402
import refiner as refiner_lib  # noqa: E402


DEFAULT_SOURCE_H5 = Path(
    "/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/"
    "cube_single_expert.h5"
)
DEFAULT_MODEL_CONFIG = REPO_ROOT / "runs/lewm_transfer/cube/cache/model/config.json"
DEFAULT_MODEL_WEIGHTS = REPO_ROOT / "runs/lewm_transfer/cube/cache/model/weights.pt"
DEFAULT_REFINER = REPO_ROOT / (
    "runs/lewm_adaptive_compute_v1/checkpoints/refiner_seed_260713.pt"
)
V1_FULL_MANIFEST = REPO_ROOT / "runs/lewm_adaptive_compute_v1/cache/split_manifest.json"
V1_SMOKE_MANIFEST = REPO_ROOT / (
    "runs/lewm_adaptive_compute_v1/smoke_run/cache/split_manifest.json"
)

EXPECTED_HASHES = {
    "base_config": "4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999",
    "base_weights": "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89",
    "refiner": "388a82fc30c96921083bfa4f2578cb296e6c3544510953d17578cf766cdf10f1",
    "v1_full_manifest": "3cc9bd4d7df229b0a4111a49933e98c5b72e916b5b5f98d7d5e023fc30ed544d",
    "v1_smoke_manifest": "f5a4da6c0f37d6222e83085327e30a1b70f0ace6f96e9c97d37c1067d7b65940",
}
FINAL_DEPTHS = (1, 2, 4)
SPLIT_CODES = {"train": 0, "calibration": 1, "test": 2}
EXPECTED_TRANSITIONS_PER_EPISODE = 38
EXPECTED_PRIOR_EXCLUSION_COUNT = 636
REFINER_FLOPS_PER_CALL = 669_184


def print_step(message: str) -> None:
    print(f"[lewm-adaptive-v2] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "full_config.json")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_SOURCE_H5)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-weights", type=Path, default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--refiner-checkpoint", type=Path, default=DEFAULT_REFINER)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-gate", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("strict JSON artifact contains a nonfinite float")
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([_jsonable(dict(row)) for row in rows])
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"nonfinite numeric value in {path.name}")
    frame.to_csv(path, index=False)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if hasattr(model_io_lib, "sha256_file"):
        return str(model_io_lib.sha256_file(path))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"frozen {label} SHA-256 mismatch: expected {expected}, got {observed}"
        )
    return observed


def state_tensor_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_arrays(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_frozen_module(module: nn.Module, label: str) -> dict[str, Any]:
    if module.training:
        raise RuntimeError(f"{label} must remain in eval mode")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError(f"{label} has trainable parameters")
    if any(parameter.grad is not None for parameter in module.parameters()):
        raise RuntimeError(f"{label} accumulated a gradient")
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in module.parameters())),
        "all_parameters_frozen": True,
        "no_parameter_gradients": True,
        "state_tensor_sha256": state_tensor_hash(module),
    }


def _read_manifest_split(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        raw = payload["selection"]["split_episode_ordinals"]
        return {name: [int(value) for value in raw[name]] for name in SPLIT_CODES}
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot parse prior Cube split manifest {path}") from exc


def prior_exclusions() -> tuple[np.ndarray, dict[str, Any]]:
    if hasattr(model_io_lib, "load_prior_episode_exclusions"):
        excluded, provenance = model_io_lib.load_prior_episode_exclusions(
            V1_FULL_MANIFEST,
            V1_SMOKE_MANIFEST,
            full_manifest_sha256=EXPECTED_HASHES["v1_full_manifest"],
            smoke_manifest_sha256=EXPECTED_HASHES["v1_smoke_manifest"],
            total_episodes=10_000,
        )
        if len(excluded) != EXPECTED_PRIOR_EXCLUSION_COUNT:
            raise RuntimeError("model_io returned an unexpected prior-exclusion count")
        return excluded, dict(provenance)
    full_hash = require_hash(
        V1_FULL_MANIFEST, EXPECTED_HASHES["v1_full_manifest"], "V1 full manifest"
    )
    smoke_hash = require_hash(
        V1_SMOKE_MANIFEST, EXPECTED_HASHES["v1_smoke_manifest"], "V1 smoke manifest"
    )
    full = _read_manifest_split(V1_FULL_MANIFEST)
    smoke = _read_manifest_split(V1_SMOKE_MANIFEST)
    excluded = set(range(30))
    for split in (full, smoke):
        for values in split.values():
            excluded.update(values)
    if len(excluded) != EXPECTED_PRIOR_EXCLUSION_COUNT:
        raise RuntimeError(
            f"prior exclusion union has {len(excluded)} episodes, expected "
            f"{EXPECTED_PRIOR_EXCLUSION_COUNT}"
        )
    if not set(range(30)).issubset(excluded) or not set(range(30, 36)).issubset(excluded):
        raise RuntimeError("prior diagnostic/smoke exclusions are incomplete")
    return np.asarray(sorted(excluded), dtype=np.int64), {
        "diagnostic_episode_ordinals": list(range(30)),
        "excluded_episode_ordinals": sorted(excluded),
        "excluded_count": len(excluded),
        "source_manifests": {
            "v1_full": {"path": str(V1_FULL_MANIFEST.resolve()), "sha256": full_hash},
            "v1_smoke": {"path": str(V1_SMOKE_MANIFEST.resolve()), "sha256": smoke_hash},
        },
        "other_prior_cube_manifests_found": False,
    }


def validate_split_isolation(split: Mapping[str, np.ndarray]) -> None:
    if hasattr(allocation_lib, "validate_split_isolation"):
        allocation_lib.validate_split_isolation(split)
    sets = {name: set(np.asarray(values, dtype=np.int64).tolist()) for name, values in split.items()}
    for left in SPLIT_CODES:
        for right in SPLIT_CODES:
            if left < right and sets[left] & sets[right]:
                raise RuntimeError(f"episode overlap between {left} and {right}")


def configured_split(
    cfg: Mapping[str, Any], excluded: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    smoke = np.asarray(cfg.get("reserved_smoke_episodes", [36, 37, 38, 39, 40, 41]), dtype=np.int64)
    if cfg["mode"] == "full":
        if not np.array_equal(smoke, np.arange(36, 42, dtype=np.int64)):
            raise RuntimeError("full configuration changed the frozen smoke reservation")
        split = model_io_lib.deterministic_episode_split(
            total_episodes=10_000,
            excluded=excluded,
            reserved=smoke,
            seed=int(cfg["selection_seed"]),
            train_count=int(cfg["train_episodes"]),
            calibration_count=int(cfg["calibration_episodes"]),
            test_count=int(cfg["test_episodes"]),
        )
        seed: int | None = int(cfg["selection_seed"])
    elif cfg["mode"] == "smoke":
        split = {
            name: np.asarray(cfg[f"{name}_episode_ordinals"], dtype=np.int64)
            for name in SPLIT_CODES
        }
        seed = None
        selected = np.concatenate(list(split.values()))
        if set(selected.tolist()) != set(range(36, 42)):
            raise RuntimeError("smoke split must use only reserved episodes 36..41")
    else:
        raise ValueError(f"unknown run mode {cfg['mode']!r}")

    validate_split_isolation(split)
    selected_set = set(np.concatenate(list(split.values())).tolist())
    if selected_set & set(excluded.tolist()):
        raise RuntimeError("V2 split intersects prior exclusions")
    if cfg["mode"] == "full" and selected_set & set(range(36, 42)):
        raise RuntimeError("full split intersects reserved smoke episodes")
    expected_counts = (
        {"train": 420, "calibration": 90, "test": 90}
        if cfg["mode"] == "full"
        else {"train": 4, "calibration": 1, "test": 1}
    )
    for name, expected in expected_counts.items():
        if len(split[name]) != expected or len(np.unique(split[name])) != expected:
            raise RuntimeError(f"{name} episode count/isolation mismatch")
    return split, {
        "selection_seed": seed,
        "reserved_smoke_episode_ordinals": list(range(36, 42)),
        "split_episode_ordinals": {name: values.tolist() for name, values in split.items()},
        "split_episode_counts": {name: len(values) for name, values in split.items()},
    }


def _extract_cache(
    *,
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    split: Mapping[str, np.ndarray],
    cache_path: Path,
    manifest_path: Path,
    device: torch.device,
    excluded: np.ndarray,
    exclusion_provenance: Mapping[str, Any],
) -> None:
    """Adapter for the V2/V1-compatible frozen Cube extractor."""
    extractor = getattr(model_io_lib, "extract_fresh_cube_cache", None)
    if extractor is None:
        extractor = getattr(model_io_lib, "extract_cube_cache")
    extractor(
        source_h5=args.source_h5,
        config_path=args.model_config,
        weights_path=args.model_weights,
        output_npz=cache_path,
        split_manifest_path=manifest_path,
        split_episodes=split,
        selection_seed=int(cfg.get("selection_seed", 260813)),
        excluded_episode_ordinals=excluded,
        reserved_episode_ordinals=(range(36, 42) if cfg["mode"] == "full" else ()),
        prior_manifest_provenance=exclusion_provenance,
        device=device,
        encode_batch_size=int(cfg["encode_batch_size"]),
        predict_batch_size=int(cfg["predict_batch_size"]),
    )


def load_cache(path: Path, expected_split: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    required = {"history", "action", "base_pred", "target", "episode_id", "model_step", "split"}
    missing = required - set(arrays)
    if missing:
        raise RuntimeError(f"latent cache is missing {sorted(missing)}")
    n = len(arrays["target"])
    if any(len(arrays[key]) != n for key in required):
        raise RuntimeError("latent cache arrays have inconsistent row counts")
    if arrays["history"].shape[1:] != (3, 192):
        raise RuntimeError(f"unexpected history shape {arrays['history'].shape}")
    if arrays["action"].shape[1:] != (3, 25):
        raise RuntimeError(f"unexpected action shape {arrays['action'].shape}")
    if arrays["base_pred"].shape[1:] != (192,) or arrays["target"].shape[1:] != (192,):
        raise RuntimeError("unexpected prediction/target latent dimension")
    for key in ("history", "action", "base_pred", "target"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"nonfinite latent cache tensor: {key}")
    for name, code in SPLIT_CODES.items():
        mask = arrays["split"] == code
        observed = set(np.unique(arrays["episode_id"][mask]).tolist())
        expected = set(np.asarray(expected_split[name]).tolist())
        if observed != expected:
            raise RuntimeError(f"cached {name} episodes differ from frozen split")
        counts = Counter(arrays["episode_id"][mask].tolist())
        if not counts or set(counts.values()) != {EXPECTED_TRANSITIONS_PER_EPISODE}:
            raise RuntimeError(f"expected 38 transitions per {name} episode")
        keys = np.stack((arrays["episode_id"][mask], arrays["model_step"][mask]), axis=1)
        if len(np.unique(keys, axis=0)) != len(keys):
            raise RuntimeError(f"duplicate episode/model-step row key in {name}")
        for episode in expected:
            steps = arrays["model_step"][mask & (arrays["episode_id"] == episode)]
            if len(steps) != EXPECTED_TRANSITIONS_PER_EPISODE or np.any(np.diff(steps) <= 0):
                raise RuntimeError(f"non-temporal row order for episode {episode}")
    validate_split_isolation(expected_split)
    return arrays


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise RuntimeError("refiner checkpoint contains no state_dict")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()):
        raise RuntimeError("refiner checkpoint state_dict is malformed")
    return payload  # type: ignore[return-value]


def load_frozen_refiner(path: Path, device: torch.device) -> nn.Module:
    loader = getattr(refiner_lib, "load_frozen_v1_refiner", None)
    if loader is not None:
        model, _ = loader(path, device=device, expected_sha256=EXPECTED_HASHES["refiner"])
        return model
    model = refiner_lib.SharedResidualRefiner(
        latent_dim=192, action_dim=25, history_len=3, hidden_dim=256, iteration_dim=16
    )
    state = _checkpoint_state(path)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict refiner checkpoint load failed")
    model.eval().requires_grad_(False).to(device)
    audit = validate_frozen_module(model, "refiner")
    if audit["parameter_count"] != 335_360:
        raise RuntimeError(
            f"refiner parameter count {audit['parameter_count']} != frozen 335360"
        )
    return model


def _batch_inputs(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        torch.from_numpy(np.ascontiguousarray(arrays[key][indices])).to(device)
        for key in ("history", "action", "base_pred")
    )  # type: ignore[return-value]


def evaluate_dense_exits(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return z1/z2/z4; each dense batch intentionally computes four calls."""
    result = np.empty((len(indices), 3, 192), dtype=np.float32)
    processed_rows = 0
    invocations = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            history, action, z0 = _batch_inputs(arrays, batch_indices, device)
            outputs, stats = model(history, action, z0, depths=FINAL_DEPTHS, return_stats=True)
            for column, depth in enumerate(FINAL_DEPTHS):
                result[start : start + len(batch_indices), column] = outputs[depth].cpu().numpy()
            processed_rows += int(stats.processed_rows)
            invocations += int(stats.block_invocations)
    expected = 4 * len(indices)
    if processed_rows != expected:
        raise RuntimeError(f"dense exit accounting {processed_rows} != {expected}")
    return result, {"processed_rows": processed_rows, "block_invocations": invocations}


def first_call_outputs(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.empty((len(indices), 192), dtype=np.float32)
    processed_rows = 0
    invocations = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            history, action, z0 = _batch_inputs(arrays, batch_indices, device)
            if hasattr(model, "forward_first"):
                z1, stats = model.forward_first(history, action, z0, return_stats=True)
            else:
                outputs, stats = model(history, action, z0, depths=(1,), return_stats=True)
                z1 = outputs[1]
            result[start : start + len(batch_indices)] = z1.cpu().numpy()
            processed_rows += int(stats.processed_rows)
            invocations += int(stats.block_invocations)
    if processed_rows != len(indices):
        raise RuntimeError("the first refiner call was not executed exactly once per row")
    return result, {"processed_rows": processed_rows, "block_invocations": invocations}


def build_features_numpy(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray, z1: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    builder = getattr(refiner_lib, "build_causal_post_call_features", None)
    if builder is None:
        builder = getattr(refiner_lib, "build_post_call_features", None)
    if builder is None:
        builder = getattr(refiner_lib, "build_causal_gate_features")
    chunks: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    for start in range(0, len(indices), 4096):
        batch_indices = indices[start : start + 4096]
        history = torch.from_numpy(np.ascontiguousarray(arrays["history"][batch_indices]))
        action = torch.from_numpy(np.ascontiguousarray(arrays["action"][batch_indices]))
        z0 = torch.from_numpy(np.ascontiguousarray(arrays["base_pred"][batch_indices]))
        z1_batch = torch.from_numpy(np.ascontiguousarray(z1[start : start + len(batch_indices)]))
        features, names = builder(history, action, z0, z1_batch, return_names=True)
        names = tuple(names)
        if hasattr(refiner_lib, "validate_causal_feature_names"):
            refiner_lib.validate_causal_feature_names(names)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("causal gate feature-name drift")
        chunks.append(np.asarray(features.cpu(), dtype=np.float32))
    if feature_names is None:
        raise RuntimeError("cannot construct features for an empty split")
    values = np.concatenate(chunks)
    if values.shape[0] != len(indices) or not np.isfinite(values).all():
        raise RuntimeError("invalid causal feature matrix")
    return values, feature_names


def save_feature_cache(
    path: Path,
    *,
    split_indices: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    z1_by_split: Mapping[str, np.ndarray],
    features_by_split: Mapping[str, np.ndarray],
    feature_names: Sequence[str],
    binding: Mapping[str, Any],
) -> None:
    payload: dict[str, np.ndarray] = {
        "feature_names": np.asarray(feature_names),
        "binding_json": np.asarray(
            json.dumps(_jsonable(binding), sort_keys=True, separators=(",", ":"), allow_nan=False)
        ),
    }
    for name in SPLIT_CODES:
        indices = split_indices[name]
        payload[f"{name}_episode_id"] = arrays["episode_id"][indices]
        payload[f"{name}_model_step"] = arrays["model_step"][indices]
        payload[f"{name}_z1"] = z1_by_split[name]
        payload[f"{name}_features"] = features_by_split[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_feature_cache(
    path: Path,
    *,
    split_indices: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    expected_binding: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as stored:
        observed_binding = json.loads(str(stored["binding_json"].item()))
        if observed_binding != _jsonable(expected_binding):
            raise RuntimeError("causal feature cache provenance binding mismatch")
        names = tuple(str(value) for value in stored["feature_names"].tolist())
        z1 = {name: stored[f"{name}_z1"] for name in SPLIT_CODES}
        features = {name: stored[f"{name}_features"] for name in SPLIT_CODES}
        for name in SPLIT_CODES:
            indices = split_indices[name]
            if not np.array_equal(stored[f"{name}_episode_id"], arrays["episode_id"][indices]):
                raise RuntimeError(f"cached {name} feature episode IDs drifted")
            if not np.array_equal(stored[f"{name}_model_step"], arrays["model_step"][indices]):
                raise RuntimeError(f"cached {name} feature row keys drifted")
    if not names or any(not np.isfinite(values).all() for values in (*z1.values(), *features.values())):
        raise RuntimeError("invalid causal feature cache")
    return z1, features, names


def gate_training_config(cfg: Mapping[str, Any]) -> Any:
    return gate_lib.GateTrainingConfig(
        learning_rate=float(cfg["gate_learning_rate"]),
        weight_decay=float(cfg["gate_weight_decay"]),
        batch_size=int(cfg["gate_batch_size"]),
        max_epochs=int(cfg["gate_max_epochs"]),
        gradient_clip=float(cfg["gradient_clip"]),
        smooth_l1_beta=float(cfg["gate_smooth_l1_beta"]),
        ranking_weight=float(cfg["gate_ranking_weight"]),
        early_stopping_patience=int(cfg["gate_patience"]),
        early_stopping_min_delta=float(cfg["gate_min_delta"]),
        calibration_pair_seed=260813,
    )


def _gate_result_from_loaded(loaded: Any) -> Any:
    return gate_lib.GateSeedResult(
        seed=int(loaded.seed),
        model=loaded.model,
        input_transform=loaded.input_transform,
        target_transform=loaded.target_transform,
        best_epoch=int(loaded.best_epoch),
        best_calibration_objective=float(loaded.best_calibration_objective),
        epochs_completed=0,
        history=[],
        trainable_parameters=int(sum(p.numel() for p in loaded.model.parameters())),
    )


def train_or_load_gates(
    *,
    cfg: Mapping[str, Any],
    train_features: np.ndarray,
    train_benefits: np.ndarray,
    calibration_features: np.ndarray,
    calibration_benefits: np.ndarray,
    feature_names: Sequence[str],
    checkpoint_dir: Path,
    force: bool,
    device: torch.device,
    binding: Mapping[str, Any],
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, str]]]:
    training_cfg = gate_training_config(cfg)
    seed_order = [int(seed) for seed in cfg["gate_seeds"]]
    results = []
    histories: list[dict[str, Any]] = []
    checkpoint_audits: dict[int, dict[str, str]] = {}
    for seed in seed_order:
        checkpoint_path = checkpoint_dir / f"gate_seed_{seed}.pt"
        binding_path = checkpoint_dir / f"gate_seed_{seed}.binding.json"
        if checkpoint_path.exists() and not force:
            if not binding_path.is_file():
                raise RuntimeError(
                    f"gate checkpoint {checkpoint_path.name} has no provenance binding; rerun with --force-gate"
                )
            observed_binding = json.loads(binding_path.read_text(encoding="utf-8"))
            expected_binding = {**_jsonable(binding), "seed": seed}
            if observed_binding != expected_binding:
                raise RuntimeError(
                    f"gate checkpoint {checkpoint_path.name} provenance mismatch; rerun with --force-gate"
                )
            loaded = gate_lib.load_gate_checkpoint(
                checkpoint_path,
                expected_input_dim=train_features.shape[1],
                expected_feature_names=feature_names,
                expected_seed=seed,
                device=device,
            )
            result = _gate_result_from_loaded(loaded)
            checkpoint_audits[seed] = {
                "file_sha256": str(loaded.file_sha256),
                "state_tensor_hash": str(loaded.state_tensor_hash),
            }
        else:
            print_step(f"training preregistered gate seed {seed}")
            result = gate_lib.train_gate_seed(
                train_features,
                train_benefits,
                calibration_features,
                calibration_benefits,
                seed=seed,
                config=training_cfg,
                device=device,
            )
            checkpoint_audits[seed] = gate_lib.save_gate_checkpoint(
                checkpoint_path, result, feature_names=feature_names
            )
            write_json(binding_path, {**_jsonable(binding), "seed": seed})
            histories.extend(dict(row) for row in result.history)
        results.append(result)

    selected, diagnostics = gate_lib.select_gate_seed(
        results,
        calibration_features,
        calibration_benefits,
        seed_order=seed_order,
    )
    return selected, diagnostics, histories, checkpoint_audits


def run_continuation(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    z1: np.ndarray,
    allocation: np.ndarray,
    device: torch.device,
    batch_size: int,
    repeats: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_continuation = int(np.sum(allocation - 1, dtype=np.int64))
    elapsed: list[float] = []
    first_output: np.ndarray | None = None
    first_invocations = 0
    for repeat in range(repeats):
        chunks: list[np.ndarray] = []
        calls = 0
        invocations = 0
        model_io_lib.synchronize(device)
        start_time = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                history = torch.from_numpy(np.ascontiguousarray(arrays["history"][batch_indices])).to(device)
                action = torch.from_numpy(np.ascontiguousarray(arrays["action"][batch_indices])).to(device)
                z1_batch = torch.from_numpy(np.ascontiguousarray(z1[start : start + len(batch_indices)])).to(device)
                depths = torch.from_numpy(
                    np.ascontiguousarray(allocation[start : start + len(batch_indices)])
                ).to(device)
                if not hasattr(model, "continue_selected"):
                    raise RuntimeError("V2 refiner lacks staged continuation API")
                output, stats = model.continue_selected(
                    history, action, z1_batch, depths, return_stats=True
                )
                chunks.append(output.cpu().numpy())
                calls += int(stats.processed_rows)
                invocations += int(stats.block_invocations)
        model_io_lib.synchronize(device)
        elapsed.append(time.perf_counter() - start_time)
        if calls != expected_continuation:
            raise RuntimeError(
                f"continuation calls={calls}, expected={expected_continuation}"
            )
        if repeat == 0:
            first_output = np.concatenate(chunks).astype(np.float32)
            first_invocations = invocations
    assert first_output is not None
    return first_output, {
        "samples": len(indices),
        "first_calls": len(indices),
        "continuation_calls": expected_continuation,
        "total_calls": len(indices) + expected_continuation,
        "batched_continuation_invocations": first_invocations,
        "latency_seconds_repeats": elapsed,
        "latency_seconds_median": float(np.median(elapsed)),
        "device": str(device),
    }


def median_timed(repeats: int, device: torch.device, function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    values = []
    first = None
    for repeat in range(repeats):
        model_io_lib.synchronize(device)
        start = time.perf_counter()
        value = function()
        model_io_lib.synchronize(device)
        values.append(time.perf_counter() - start)
        if repeat == 0:
            first = value
    return first, {
        "latency_seconds_repeats": values,
        "latency_seconds_median": float(np.median(values)),
        "device": str(device),
    }


def refiner_consistency_audit(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    audit_indices = indices[: min(32, len(indices))]
    history, action, z0 = _batch_inputs(arrays, audit_indices, device)
    with torch.inference_mode():
        first = model(history, action, z0, depths=FINAL_DEPTHS)
        second = model(history, action, z0, depths=FINAL_DEPTHS)
        if any(not torch.equal(first[depth], second[depth]) for depth in FINAL_DEPTHS):
            raise RuntimeError("repeated dense refiner outputs are not bitwise identical")
        z1, first_stats = model.forward_first(history, action, z0, return_stats=True)
        staged = {}
        staged_calls = int(first_stats.processed_rows)
        max_pruned_abs = 0.0
        for depth in FINAL_DEPTHS:
            selected = torch.full((len(audit_indices),), depth, dtype=torch.long, device=device)
            output, stats = model.continue_selected(
                history, action, z1, selected, return_stats=True
            )
            staged[depth] = output
            staged_calls += int(stats.processed_rows)
            if not torch.equal(output, first[depth]):
                raise RuntimeError(f"staged uniform depth-{depth} output is not bitwise dense-prefix identical")
        mixed_depths = np.resize(np.asarray(FINAL_DEPTHS, dtype=np.int64), len(audit_indices))
        mixed, stats = model.continue_selected(
            history,
            action,
            z1,
            torch.from_numpy(mixed_depths).to(device),
            return_stats=True,
        )
        dense_mixed = torch.empty_like(mixed)
        for depth in FINAL_DEPTHS:
            mask = torch.from_numpy(mixed_depths == depth).to(device)
            dense_mixed[mask] = first[depth][mask]
        max_pruned_abs = float(torch.max(torch.abs(mixed - dense_mixed)).item())
        if max_pruned_abs > 2e-5:
            raise RuntimeError(f"mixed staged/dense mismatch {max_pruned_abs} exceeds 2e-5")
        expected_mixed_calls = int(np.sum(mixed_depths - 1, dtype=np.int64))
        if int(stats.processed_rows) != expected_mixed_calls:
            raise RuntimeError("mixed staged continuation accounting mismatch")
    return {
        "audit_rows": len(audit_indices),
        "repeated_dense_bitwise_equal": True,
        "staged_uniform_bitwise_equal": True,
        "mixed_pruned_max_abs": max_pruned_abs,
        "mixed_pruned_tolerance": 2e-5,
        "mixed_continuation_calls": expected_mixed_calls,
        "auxiliary_staged_calls": staged_calls,
    }


def _ci(values: np.ndarray, episodes: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, Any]:
    result = allocation_lib.paired_cluster_bootstrap(
        values,
        episodes,
        n_bootstrap=int(cfg["bootstrap_samples"]),
        seed=int(cfg["bootstrap_seed"]),
    )
    return {
        "mean_benefit": float(result.estimate),
        "ci_low": float(result.lower),
        "ci_high": float(result.upper),
        "n": int(result.n_samples),
        "n_episodes": int(result.n_episodes),
        "bootstrap_samples": int(result.n_bootstrap),
    }


def safe_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if rank:
        x = pd.Series(x).rank(method="average").to_numpy()
        y = pd.Series(y).rank(method="average").to_numpy()
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else 0.0


def benefit_diagnostic_rows(
    scope: str, predicted: np.ndarray, actual: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for column, benefit in enumerate(("b12", "b14")):
        error = predicted[:, column] - actual[:, column]
        rows.append(
            {
                "scope": scope,
                "benefit": benefit,
                "pearson": safe_correlation(predicted[:, column], actual[:, column]),
                "spearman": safe_correlation(
                    predicted[:, column], actual[:, column], rank=True
                ),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "predicted_mean": float(predicted[:, column].mean()),
                "actual_mean": float(actual[:, column].mean()),
            }
        )
    return rows


def policy_and_comparison_rows(
    *,
    raw_losses: np.ndarray,
    whitened_losses: np.ndarray,
    allocations: Mapping[str, np.ndarray],
    episode_ids: np.ndarray,
    cfg: Mapping[str, Any],
    raw_overrides: Mapping[str, np.ndarray] | None = None,
    whitened_overrides: Mapping[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    selected_raw = {
        name: allocation_lib.gather_depths(raw_losses, allocation)
        for name, allocation in allocations.items()
    }
    selected_white = {
        name: allocation_lib.gather_depths(whitened_losses, allocation)
        for name, allocation in allocations.items()
    }
    for name, values in (raw_overrides or {}).items():
        selected_raw[name] = np.asarray(values, dtype=np.float64)
    for name, values in (whitened_overrides or {}).items():
        selected_white[name] = np.asarray(values, dtype=np.float64)
    rows = []
    for name, selected in allocations.items():
        rows.append(
            {
                "policy": name,
                "raw_mean_loss": float(selected_raw[name].mean()),
                "whitened_mean_loss": float(selected_white[name].mean()),
                "depth_1": int(np.sum(selected == 1)),
                "depth_2": int(np.sum(selected == 2)),
                "depth_4": int(np.sum(selected == 4)),
                "first_calls": len(selected),
                "continuation_calls": int(np.sum(selected - 1, dtype=np.int64)),
                "total_calls": int(np.sum(selected, dtype=np.int64)),
                "mean_calls": float(np.mean(selected)),
                "diagnostic_only": name == "oracle",
            }
        )
    definitions = {
        "adaptive_vs_uniform_d2": ("uniform_d2", "adaptive"),
        "adaptive_vs_fixed_d1": ("fixed_d1", "adaptive"),
        "adaptive_vs_random": ("random", "adaptive"),
        "adaptive_vs_permutation": ("permutation", "adaptive"),
        "oracle_regret": ("adaptive", "oracle"),
        "oracle_advantage_vs_uniform_d2": ("uniform_d2", "oracle"),
    }
    comparison_rows = []
    for comparison, (left, right) in definitions.items():
        for metric, values in (("raw", selected_raw), ("whitened", selected_white)):
            row = {
                "comparison": comparison,
                "metric": metric,
                **_ci(values[left] - values[right], episode_ids, cfg),
                "positive_favors": "adaptive" if comparison.startswith("adaptive_vs") else (
                    "oracle" if comparison in {"oracle_regret", "oracle_advantage_vs_uniform_d2"} else "left"
                ),
                "diagnostic_only": comparison.startswith("oracle"),
            }
            comparison_rows.append(row)
    return rows, comparison_rows, selected_raw, selected_white


def procedural_oracle_comparison_rows(
    *,
    raw_losses: np.ndarray,
    whitened_losses: np.ndarray,
    adaptive_raw: np.ndarray,
    adaptive_white: np.ndarray,
    episode_ids: np.ndarray,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bootstrap the diagnostic oracle as an outcome-dependent procedure.

    The raw-MSE oracle is re-optimized inside every episode-clustered
    replicate.  The replicate's whitened oracle score gathers with that same
    raw-selected allocation; whitened loss never selects an oracle policy.
    """
    raw = np.asarray(raw_losses, dtype=np.float64)
    white = np.asarray(whitened_losses, dtype=np.float64)
    episodes = np.asarray(episode_ids)
    clusters = [np.flatnonzero(episodes == episode) for episode in np.unique(episodes)]
    n_bootstrap = int(cfg["bootstrap_samples"])
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    bootstrap = {
        ("oracle_regret", "raw"): np.empty(n_bootstrap, dtype=np.float64),
        ("oracle_regret", "whitened"): np.empty(n_bootstrap, dtype=np.float64),
        ("oracle_advantage_vs_uniform_d2", "raw"): np.empty(n_bootstrap, dtype=np.float64),
        ("oracle_advantage_vs_uniform_d2", "whitened"): np.empty(n_bootstrap, dtype=np.float64),
    }
    for replicate in range(n_bootstrap):
        sampled_clusters = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[int(index)] for index in sampled_clusters])
        replicate_allocation = allocation_lib.sequential_oracle_exact_budget_allocation(raw[rows])
        oracle_raw = allocation_lib.gather_depths(raw[rows], replicate_allocation)
        oracle_white = allocation_lib.gather_depths(white[rows], replicate_allocation)
        bootstrap[("oracle_regret", "raw")][replicate] = float(
            np.mean(adaptive_raw[rows] - oracle_raw)
        )
        bootstrap[("oracle_regret", "whitened")][replicate] = float(
            np.mean(adaptive_white[rows] - oracle_white)
        )
        bootstrap[("oracle_advantage_vs_uniform_d2", "raw")][replicate] = float(
            np.mean(raw[rows, 1] - oracle_raw)
        )
        bootstrap[("oracle_advantage_vs_uniform_d2", "whitened")][replicate] = float(
            np.mean(white[rows, 1] - oracle_white)
        )

    full_allocation = allocation_lib.sequential_oracle_exact_budget_allocation(raw)
    full_oracle_raw = allocation_lib.gather_depths(raw, full_allocation)
    full_oracle_white = allocation_lib.gather_depths(white, full_allocation)
    estimates = {
        ("oracle_regret", "raw"): float(np.mean(adaptive_raw - full_oracle_raw)),
        ("oracle_regret", "whitened"): float(np.mean(adaptive_white - full_oracle_white)),
        ("oracle_advantage_vs_uniform_d2", "raw"): float(np.mean(raw[:, 1] - full_oracle_raw)),
        ("oracle_advantage_vs_uniform_d2", "whitened"): float(np.mean(white[:, 1] - full_oracle_white)),
    }
    result = []
    for (comparison, metric), values in bootstrap.items():
        lower, upper = np.quantile(values, [0.025, 0.975])
        result.append(
            {
                "comparison": comparison,
                "metric": metric,
                "mean_benefit": estimates[(comparison, metric)],
                "ci_low": float(lower),
                "ci_high": float(upper),
                "n": len(raw),
                "n_episodes": len(clusters),
                "bootstrap_samples": n_bootstrap,
                "positive_favors": "oracle",
                "diagnostic_only": True,
                "bootstrap_procedure": "episode_cluster_resample_then_reoptimize_raw_oracle",
            }
        )
    return result


def fit_physical_thresholds(labels: Mapping[str, np.ndarray], calibration_mask: np.ndarray) -> dict[str, float]:
    interaction = np.asarray(labels["interaction"], dtype=bool)
    noncontact = calibration_mask & ~interaction
    if not np.any(noncontact):
        raise RuntimeError("no calibration non-contact transitions for post-hoc thresholds")
    return {
        "effector_motion_threshold_m": float(
            np.percentile(np.asarray(labels["effector_disp"])[noncontact], 10.0)
        ),
        "block_motion_threshold_m": max(
            0.001, float(np.percentile(np.asarray(labels["block_disp"])[noncontact], 95.0))
        ),
    }


def apply_physical_regimes(
    labels: Mapping[str, np.ndarray], indices: np.ndarray, thresholds: Mapping[str, float]
) -> np.ndarray:
    contact = np.asarray(labels["interaction"])[indices].astype(bool)
    impact = np.asarray(labels["impact"])[indices].astype(bool)
    moving = (
        np.asarray(labels["effector_disp"])[indices] > thresholds["effector_motion_threshold_m"]
    ) | (np.asarray(labels["block_disp"])[indices] > thresholds["block_motion_threshold_m"])
    return np.select(
        [impact, contact, (~contact) & moving],
        ["impact", "contact", "transport_free"],
        default="static",
    )


def regime_rows(
    regimes: np.ndarray,
    allocation: np.ndarray,
    raw_losses: np.ndarray,
    selected_raw: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for regime in ("impact", "contact", "transport_free", "static"):
        mask = regimes == regime
        if not np.any(mask):
            continue
        rows.append(
            {
                "regime": regime,
                "n": int(mask.sum()),
                "mean_depth": float(allocation[mask].mean()),
                "depth_1_fraction": float(np.mean(allocation[mask] == 1)),
                "depth_2_fraction": float(np.mean(allocation[mask] == 2)),
                "depth_4_fraction": float(np.mean(allocation[mask] == 4)),
                "mean_gain_vs_depth1": float((raw_losses[mask, 0] - selected_raw[mask]).mean()),
                "post_hoc_only": True,
            }
        )
    return rows


def run_uniform_depth2_latency(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    repeats: int,
) -> dict[str, Any]:
    elapsed = []
    invocations = 0
    for repeat in range(repeats):
        calls = 0
        invocation_count = 0
        model_io_lib.synchronize(device)
        start_time = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                history, action, z0 = _batch_inputs(arrays, batch_indices, device)
                _, stats = model(history, action, z0, depths=(2,), return_stats=True)
                calls += int(stats.processed_rows)
                invocation_count += int(stats.block_invocations)
        model_io_lib.synchronize(device)
        elapsed.append(time.perf_counter() - start_time)
        if calls != 2 * len(indices):
            raise RuntimeError("uniform depth-2 execution did not use exactly 2N calls")
        if repeat == 0:
            invocations = invocation_count
    return {
        "samples": len(indices),
        "first_calls": len(indices),
        "continuation_calls": len(indices),
        "total_calls": 2 * len(indices),
        "batched_block_invocations": invocations,
        "latency_seconds_repeats": elapsed,
        "latency_seconds_median": float(np.median(elapsed)),
        "device": str(device),
    }


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "(none)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.7g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    decision: Mapping[str, Any],
    policy_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    quantile_rows: Sequence[Mapping[str, Any]],
    regime_summary: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    figures: Mapping[str, str],
) -> None:
    raw_comparisons = [row for row in comparison_rows if row["metric"] == "raw"]
    whitened_comparisons = [
        row for row in comparison_rows if row["metric"] == "whitened"
    ]
    smoke_note = (
        "\nThis is the reserved six-episode engineering smoke run; it supports no "
        "scientific claim.\n"
        if decision["mode"] == "smoke"
        else ""
    )
    report = f"""# LeWM Adaptive Compute V2 Report

## Frozen decision

**`{decision['verdict']}`** — {decision['reason']}
{smoke_note}
The confirmatory policy executed the first frozen-refiner call for every
transition, then assigned final depths from `{{1,2,4}}` at exactly `2N` total
calls. The primary claim is transition-selective latent computation, not
physical grounding or contact-specific compute benefit.

## Untouched-test policies

{markdown_table(policy_rows, ['policy', 'raw_mean_loss', 'whitened_mean_loss', 'depth_1', 'depth_2', 'depth_4', 'total_calls', 'mean_calls'])}

## Paired episode-clustered comparisons

Positive benefit favors the named adaptive/oracle side. Intervals are 95%
episode-clustered percentile intervals with the preregistered bootstrap seed.

{markdown_table(raw_comparisons, ['comparison', 'mean_benefit', 'ci_low', 'ci_high', 'n_episodes', 'diagnostic_only'])}

Whitened latent MSE is a calibration-fit sensitivity analysis and does not
override the raw-MSE decision.

{markdown_table(whitened_comparisons, ['comparison', 'mean_benefit', 'ci_low', 'ci_high', 'n_episodes', 'diagnostic_only'])}

## Gate ranking diagnostics

{markdown_table(diagnostic_rows, ['scope', 'benefit', 'pearson', 'spearman', 'rmse', 'mae'])}

Benefit-by-score quantiles are in `metrics/gate_benefit_quantiles.csv`. Only
calibration diagnostics selected the seed; test diagnostics are descriptive.

## Post-hoc physical interpretation

Contact, impact, transport-free, and static labels were attached only after
gate predictions and the exact test allocation were frozen. These results
cannot alter the primary verdict. V1 did not show contact-specific compute
benefit.

{markdown_table(regime_summary, ['regime', 'n', 'mean_depth', 'depth_1_fraction', 'depth_2_fraction', 'depth_4_fraction', 'mean_gain_vs_depth1'])}

## Validity and provenance

- Released base config SHA-256: `{provenance['frozen_files']['base_config']['before']}`.
- Released base weights SHA-256: `{provenance['frozen_files']['base_weights']['before']}`.
- V1-selected refiner SHA-256: `{provenance['frozen_files']['refiner_source']['before']}`.
- The copied V2 refiner is byte-identical to the V1 source.
- Base and refiner state/frozen audits passed before and after use.
- All prior diagnostic/V1 episodes and reserved smoke episodes were excluded
  from the confirmatory split; all splits are episode-disjoint.
- Gate features contain only causal history/action, z0, z1, the full first
  update, and preregistered convergence/alignment summaries.
- Original LeWM pretraining episode membership remains unknown.
- The sequential target-informed oracle is diagnostic only.

Configuration and machine-readable provenance are in `configuration.json` and
`provenance.json`.

## Compute

- Frozen refiner dense-linear FLOPs per call: `{decision['compute']['refiner_flops_per_call']}`.
- Gate parameters: `{decision['compute']['gate_parameter_count']}`.
- Gate dense-linear FLOPs per transition: `{decision['compute']['gate_flops_per_transition']}`.
- Adaptive total refiner calls: `{decision['compute']['adaptive_total_calls']}`.
- Gate allocation overhead median: `{decision['compute']['latency']['gate_feature_mlp_allocation']['latency_seconds_median']:.7g}` s.

## Figures

"""
    figure_paths = figures.get("paths", figures)
    for name, figure_path in figure_paths.items():
        relative = Path(figure_path).resolve().relative_to(path.parent.resolve())
        report += f"\n![{name}]({relative.as_posix()})\n"
    path.write_text(report, encoding="utf-8")


def frozen_verdict(
    policy_rows: Sequence[Mapping[str, Any]], comparison_rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    raw = {
        str(row["comparison"]): row
        for row in comparison_rows
        if row["metric"] == "raw"
    }
    oracle = raw["oracle_advantage_vs_uniform_d2"]
    if float(oracle["ci_low"]) <= 0.0:
        return (
            "no_sequential_headroom",
            "The sequentially feasible diagnostic oracle did not beat uniform depth 2 with a positive clustered-CI lower bound.",
        )
    adaptive_uniform = raw["adaptive_vs_uniform_d2"]
    adaptive_fixed = raw["adaptive_vs_fixed_d1"]
    by_policy = {str(row["policy"]): row for row in policy_rows}
    budget_matched = (
        int(by_policy["adaptive"]["total_calls"])
        == int(by_policy["uniform_d2"]["total_calls"])
        == 2 * int(by_policy["adaptive"]["first_calls"])
    )
    success = all(
        float(row["mean_benefit"]) > 0 and float(row["ci_low"]) > 0
        for row in (adaptive_uniform, adaptive_fixed)
    ) and budget_matched
    if success:
        return (
            "sequential_gate_success",
            "The frozen sequential gate beat uniform depth 2 and fixed depth 1 in raw MSE with positive clustered-CI lower bounds at the exact matched budget.",
        )
    return (
        "sequential_headroom_gate_failed",
        "Sequential oracle headroom exists, but the frozen gate failed at least one preregistered raw-MSE comparison or exact-budget requirement.",
    )


def validate_configuration(cfg: Mapping[str, Any]) -> None:
    common = {
        "encode_batch_size": 64,
        "predict_batch_size": 512,
        "gate_batch_size": 256,
        "gate_max_epochs": 150,
        "gate_patience": 20,
        "gate_min_delta": 0.00001,
        "gate_learning_rate": 0.0003,
        "gate_weight_decay": 0.0001,
        "gate_dropout": 0.05,
        "gate_hidden_dims": [128, 64],
        "gate_smooth_l1_beta": 0.5,
        "gate_ranking_weight": 0.25,
        "gate_seeds": [260813, 260814, 260815],
        "gradient_clip": 1.0,
        "final_depths": [1, 2, 4],
        "mean_call_budget": 2,
        "random_seed": 260816,
        "permutation_seed": 260817,
        "bootstrap_samples": 2000,
        "bootstrap_seed": 260818,
        "latency_repeats": 5,
    }
    expected_full = {
        "mode": "full",
        "selection_seed": 260813,
        "train_episodes": 420,
        "calibration_episodes": 90,
        "test_episodes": 90,
        "reserved_smoke_episodes": [36, 37, 38, 39, 40, 41],
        **common,
    }
    expected_smoke = {
        "mode": "smoke",
        "train_episode_ordinals": [36, 37, 38, 39],
        "calibration_episode_ordinals": [40],
        "test_episode_ordinals": [41],
        **{
            **common,
            "encode_batch_size": 32,
            "predict_batch_size": 128,
            "gate_batch_size": 64,
            "gate_max_epochs": 3,
            "gate_patience": 3,
            "gate_seeds": [260813],
            "bootstrap_samples": 100,
            "latency_repeats": 2,
        },
    }
    expected = expected_full if cfg.get("mode") == "full" else expected_smoke
    if dict(cfg) != expected:
        missing = sorted(set(expected) - set(cfg))
        extra = sorted(set(cfg) - set(expected))
        changed = sorted(key for key in set(cfg) & set(expected) if cfg[key] != expected[key])
        raise RuntimeError(
            f"configuration differs from the frozen exact schema: missing={missing}, "
            f"extra={extra}, changed={changed}"
        )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    validate_configuration(cfg)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(EXPERIMENT_DIR.resolve())
    except ValueError as exc:
        raise RuntimeError("V2 output directory must remain inside the isolated V2 run directory") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    for directory in (cache_dir, checkpoint_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    v1_root = REPO_ROOT / "runs/lewm_adaptive_compute_v1"
    v1_hashes_before = tree_hashes(v1_root)
    frozen_file_hashes_before = {
        "base_config": require_hash(args.model_config, EXPECTED_HASHES["base_config"], "base config"),
        "base_weights": require_hash(args.model_weights, EXPECTED_HASHES["base_weights"], "base weights"),
        "refiner_source": require_hash(args.refiner_checkpoint, EXPECTED_HASHES["refiner"], "V1 refiner"),
    }
    excluded, exclusion_provenance = prior_exclusions()
    split_episodes, split_provenance = configured_split(cfg, excluded)
    split_plan = {
        **split_provenance,
        "prior_exclusions": exclusion_provenance,
        "base_pretraining_episode_membership": "unknown",
    }
    write_json(cache_dir / "planned_split.json", split_plan)
    write_json(output_dir / "configuration.json", cfg)

    device = model_io_lib.choose_device(args.device)
    print_step(f"mode={cfg['mode']} device={device} output={output_dir}")
    cache_path = cache_dir / "cube_causal_inputs.npz"
    manifest_path = cache_dir / "split_manifest.json"
    if args.force_extract or not cache_path.exists() or not manifest_path.exists():
        print_step("extracting the frozen episode split with strict released-base checks")
        _extract_cache(
            args=args,
            cfg=cfg,
            split=split_episodes,
            cache_path=cache_path,
            manifest_path=manifest_path,
            device=device,
            excluded=excluded,
            exclusion_provenance=exclusion_provenance,
        )
    arrays = load_cache(cache_path, split_episodes)
    extraction_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if extraction_manifest.get("cache", {}).get("sha256") != sha256_file(cache_path):
        raise RuntimeError("cached causal inputs do not match their extraction manifest hash")
    manifest_splits = extraction_manifest.get("selection", {}).get("split_episode_ordinals", {})
    if any(
        manifest_splits.get(name) != [int(value) for value in split_episodes[name].tolist()]
        for name in SPLIT_CODES
    ):
        raise RuntimeError("extraction manifest split differs from the frozen planned split")
    if args.extract_only:
        print_step(f"extraction complete: {cache_path}")
        return {"verdict": "extract_only", "mode": cfg["mode"]}

    copied_refiner = checkpoint_dir / "refiner_seed_260713.pt"
    if not copied_refiner.exists():
        shutil.copyfile(args.refiner_checkpoint, copied_refiner)
    require_hash(copied_refiner, EXPECTED_HASHES["refiner"], "copied V2 refiner")
    model_io_lib.verify_byte_identical_files(args.refiner_checkpoint, copied_refiner)
    refiner = load_frozen_refiner(copied_refiner, device)
    refiner_before = validate_frozen_module(refiner, "refiner")

    split_indices = {
        name: np.flatnonzero(arrays["split"] == code) for name, code in SPLIT_CODES.items()
    }
    refiner_checks = refiner_consistency_audit(
        refiner, arrays, split_indices["train"], device
    )

    # Development splits: first call/features and train/calibration targets are
    # permitted before gate fitting.  No test target is accessed in this block.
    z1_by_split: dict[str, np.ndarray] = {}
    features_by_split: dict[str, np.ndarray] = {}
    feature_names: tuple[str, ...] | None = None
    first_call_accounting: dict[str, Any] = {}
    for name in ("train", "calibration"):
        z1, stats = first_call_outputs(
            refiner,
            arrays,
            split_indices[name],
            device,
            int(cfg["predict_batch_size"]),
        )
        features, names = build_features_numpy(arrays, split_indices[name], z1)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("train/calibration feature contract mismatch")
        z1_by_split[name] = z1
        features_by_split[name] = features
        first_call_accounting[name] = stats
    assert feature_names is not None

    dense_by_split: dict[str, np.ndarray] = {}
    raw_by_split: dict[str, np.ndarray] = {}
    for name in ("train", "calibration"):
        exits, _ = evaluate_dense_exits(
            refiner,
            arrays,
            split_indices[name],
            device,
            int(cfg["predict_batch_size"]),
        )
        if not np.array_equal(exits[:, 0], z1_by_split[name]):
            raise RuntimeError(f"{name} first-call output differs from dense depth-1 prefix")
        dense_by_split[name] = exits
        raw_by_split[name] = allocation_lib.latent_mse_by_depth(
            arrays["target"][split_indices[name]], exits
        )
    train_benefits = allocation_lib.marginal_continuation_benefits(raw_by_split["train"])
    calibration_benefits = allocation_lib.marginal_continuation_benefits(
        raw_by_split["calibration"]
    )
    whitening = allocation_lib.fit_whitening(arrays["target"][split_indices["calibration"]])
    np.savez_compressed(
        cache_dir / "calibration_whitening.npz",
        mean=whitening.mean,
        matrix=whitening.matrix,
        eigenvalues=whitening.eigenvalues,
        used_eigenvalues=whitening.used_eigenvalues,
        floor=np.asarray(whitening.floor),
    )

    feature_binding = {
        "schema_version": 1,
        "source_cache_sha256": sha256_file(cache_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "refiner_checkpoint_sha256": sha256_file(copied_refiner),
        "planned_split_sha256": sha256_json(split_plan),
        "feature_names_sha256": sha256_json(list(feature_names)),
    }
    gate_binding = {
        **feature_binding,
        "gate_config_sha256": sha256_json(
            {
                key: cfg[key]
                for key in cfg
                if key.startswith("gate_") or key == "gradient_clip"
            }
        ),
        "train_calibration_features_sha256": sha256_arrays(
            [features_by_split["train"], features_by_split["calibration"]]
        ),
        "train_calibration_benefits_sha256": sha256_arrays(
            [train_benefits, calibration_benefits]
        ),
    }

    selected_gate, seed_diagnostics, gate_history, gate_checkpoint_audits = train_or_load_gates(
        cfg=cfg,
        train_features=features_by_split["train"],
        train_benefits=train_benefits,
        calibration_features=features_by_split["calibration"],
        calibration_benefits=calibration_benefits,
        feature_names=feature_names,
        checkpoint_dir=checkpoint_dir,
        force=bool(args.force_gate),
        device=device,
        binding=gate_binding,
    )
    selected_gate.model.zero_grad(set_to_none=True)
    selected_gate.model.eval().requires_grad_(False)
    gate_before = validate_frozen_module(selected_gate.model, "selected gate")
    selected_gate_checkpoint = checkpoint_dir / f"gate_seed_{selected_gate.seed}.pt"
    primary_gate_checkpoint = checkpoint_dir / "sequential_gate_selected.pt"
    shutil.copyfile(selected_gate_checkpoint, primary_gate_checkpoint)
    gate_lib.load_gate_checkpoint(
        primary_gate_checkpoint,
        expected_input_dim=len(feature_names),
        expected_feature_names=feature_names,
        expected_seed=int(selected_gate.seed),
        expected_sha256=sha256_file(selected_gate_checkpoint),
        device=device,
    )
    if gate_history:
        write_rows(metrics_dir / "gate_training_history.csv", gate_history)
    write_json(metrics_dir / "gate_seed_diagnostics.json", seed_diagnostics)
    write_json(
        metrics_dir / "gate_seed_selection.json",
        {
            "selected_seed": int(selected_gate.seed),
            "selected_best_epoch": int(selected_gate.best_epoch),
            "seed_order": [int(value) for value in cfg["gate_seeds"]],
            "selection_rule": "largest mean Spearman; ties within 1e-12 use smaller calibration objective then seed order",
            "checkpoint_audits": gate_checkpoint_audits,
        },
    )
    calibration_predictions = gate_lib.predict_gate(
        selected_gate,
        features_by_split["calibration"],
        device=device,
        batch_size=int(cfg["predict_batch_size"]),
    )
    calibration_diagnostic_rows = benefit_diagnostic_rows(
        "calibration", calibration_predictions, calibration_benefits
    )
    calibration_quantiles = gate_lib.benefit_quantile_rows(
        calibration_predictions, calibration_benefits, quantiles=5
    )
    calibration_quantiles = [dict(scope="calibration", **row) for row in calibration_quantiles]

    # Untouched-test freeze boundary.  Only target-free z1/features/predictions
    # are used to freeze allocations.  Test targets are deliberately assigned
    # to a local name only after all non-oracle allocations are persisted.
    test_indices = split_indices["test"]
    timed_first, first_latency = median_timed(
        int(cfg["latency_repeats"]),
        device,
        lambda: first_call_outputs(
            refiner, arrays, test_indices, device, int(cfg["predict_batch_size"])
        ),
    )
    z1_test, first_test_stats = timed_first
    first_latency.update(first_test_stats)
    test_features, test_feature_names = build_features_numpy(arrays, test_indices, z1_test)
    if test_feature_names != feature_names:
        raise RuntimeError("test causal feature contract differs after gate freeze")
    test_predictions = gate_lib.predict_gate(
        selected_gate,
        test_features,
        device=device,
        batch_size=int(cfg["predict_batch_size"]),
    )
    n_test = len(test_indices)
    adaptive_allocation = allocation_lib.sequential_exact_budget_allocation(
        test_predictions, n_samples=n_test
    )
    allocation_lib.validate_sequential_exact_budget(adaptive_allocation, n_test)
    nonoracle_allocations = {
        "adaptive": adaptive_allocation,
        "uniform_d2": np.full(n_test, 2, dtype=np.int64),
        "fixed_d1": np.full(n_test, 1, dtype=np.int64),
        "random": allocation_lib.random_sequential_exact_budget_allocation(
            n_test, rng=int(cfg["random_seed"])
        ),
        "permutation": allocation_lib.permuted_allocation(
            adaptive_allocation, rng=int(cfg["permutation_seed"])
        ),
    }
    for name in ("adaptive", "uniform_d2", "random", "permutation"):
        allocation_lib.validate_sequential_exact_budget(nonoracle_allocations[name], n_test)
    if allocation_lib.depth_histogram(nonoracle_allocations["permutation"]) != allocation_lib.depth_histogram(adaptive_allocation):
        raise RuntimeError("permutation did not preserve the adaptive histogram")
    allocation_freeze_path = metrics_dir / "test_allocations_frozen_pre_outcome.npz"
    np.savez_compressed(allocation_freeze_path, predictions=test_predictions, **nonoracle_allocations)
    allocation_freeze_hash = sha256_file(allocation_freeze_path)

    z1_by_split["test"] = z1_test
    features_by_split["test"] = test_features
    first_call_accounting["test"] = first_test_stats
    feature_cache_path = cache_dir / "causal_post_call_features.npz"
    save_feature_cache(
        feature_cache_path,
        split_indices=split_indices,
        arrays=arrays,
        z1_by_split=z1_by_split,
        features_by_split=features_by_split,
        feature_names=feature_names,
        binding=feature_binding,
    )

    def gate_overhead() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values, names = build_features_numpy(arrays, test_indices, z1_test)
        if names != feature_names:
            raise RuntimeError("timed gate feature contract drift")
        predicted = gate_lib.predict_gate(
            selected_gate, values, device=device, batch_size=int(cfg["predict_batch_size"])
        )
        selected = allocation_lib.sequential_exact_budget_allocation(predicted, n_samples=n_test)
        return values, predicted, selected

    timed_gate, gate_latency = median_timed(
        int(cfg["latency_repeats"]), device, gate_overhead
    )
    if not np.array_equal(timed_gate[2], adaptive_allocation):
        raise RuntimeError("timed gate/allocation differs from frozen test allocation")

    # First outcome inspection begins here, after the allocation-freeze artifact.
    test_exits, test_dense_stats = evaluate_dense_exits(
        refiner, arrays, test_indices, device, int(cfg["predict_batch_size"])
    )
    if not np.array_equal(test_exits[:, 0], z1_test):
        raise RuntimeError("test staged first output differs from dense depth-1 prefix")
    test_targets = arrays["target"][test_indices]
    test_raw = allocation_lib.latent_mse_by_depth(test_targets, test_exits)
    test_white = allocation_lib.latent_mse_by_depth(
        test_targets, test_exits, whitening=whitening
    )
    adaptive_outputs, adaptive_latency = run_continuation(
        refiner,
        arrays,
        test_indices,
        z1_test,
        adaptive_allocation,
        device,
        int(cfg["predict_batch_size"]),
        int(cfg["latency_repeats"]),
    )
    dense_adaptive = allocation_lib.gather_depths(test_exits, adaptive_allocation)
    adaptive_dense_max_abs = float(np.max(np.abs(adaptive_outputs - dense_adaptive)))
    if adaptive_dense_max_abs > 2e-5:
        raise RuntimeError(
            f"adaptive staged/dense output mismatch {adaptive_dense_max_abs} exceeds 2e-5"
        )
    adaptive_raw = np.mean(
        (adaptive_outputs.astype(np.float64) - test_targets.astype(np.float64)) ** 2,
        axis=1,
    )
    adaptive_white = allocation_lib.whitened_mse(
        adaptive_outputs, test_targets, whitening
    )
    uniform_latency = run_uniform_depth2_latency(
        refiner,
        arrays,
        test_indices,
        device,
        int(cfg["predict_batch_size"]),
        int(cfg["latency_repeats"]),
    )

    # Diagnostic oracle is intentionally constructed only now, after adaptive
    # allocation freeze and test scoring.
    oracle_allocation = allocation_lib.sequential_oracle_exact_budget_allocation(test_raw)
    allocations = {**nonoracle_allocations, "oracle": oracle_allocation}
    policy_rows, comparison_rows, selected_raw, selected_white = policy_and_comparison_rows(
        raw_losses=test_raw,
        whitened_losses=test_white,
        allocations=allocations,
        episode_ids=arrays["episode_id"][test_indices],
        cfg=cfg,
        raw_overrides={"adaptive": adaptive_raw},
        whitened_overrides={"adaptive": adaptive_white},
    )
    comparison_rows = [
        row
        for row in comparison_rows
        if row["comparison"]
        not in {"oracle_regret", "oracle_advantage_vs_uniform_d2"}
    ] + procedural_oracle_comparison_rows(
        raw_losses=test_raw,
        whitened_losses=test_white,
        adaptive_raw=adaptive_raw,
        adaptive_white=adaptive_white,
        episode_ids=arrays["episode_id"][test_indices],
        cfg=cfg,
    )
    test_benefits = allocation_lib.marginal_continuation_benefits(test_raw)
    test_diagnostic_rows = benefit_diagnostic_rows("test_descriptive", test_predictions, test_benefits)
    test_quantiles = gate_lib.benefit_quantile_rows(test_predictions, test_benefits, quantiles=5)
    test_quantiles = [dict(scope="test_descriptive", **row) for row in test_quantiles]
    diagnostic_rows = calibration_diagnostic_rows + test_diagnostic_rows
    quantile_rows = calibration_quantiles + test_quantiles

    # Physical labels are first opened after predictions and allocations froze.
    physical_labels = model_io_lib.extract_post_prediction_labels(
        args.source_h5, arrays["episode_id"], arrays["model_step"], frameskip=5
    )
    thresholds = fit_physical_thresholds(
        physical_labels, np.asarray(arrays["split"] == SPLIT_CODES["calibration"])
    )
    test_regimes = apply_physical_regimes(physical_labels, test_indices, thresholds)
    posthoc_rows = regime_rows(
        test_regimes,
        adaptive_allocation,
        test_raw,
        selected_raw["adaptive"],
    )

    allocation_rows = [
        {
            "policy": policy,
            "depth": depth,
            "count": count,
            "fraction": count / n_test,
            "total_calls": int(np.sum(selected, dtype=np.int64)),
        }
        for policy, selected in allocations.items()
        for depth, count in allocation_lib.depth_histogram(selected).items()
    ]
    write_rows(metrics_dir / "policy_metrics.csv", policy_rows)
    write_rows(metrics_dir / "paired_comparisons.csv", comparison_rows)
    write_rows(metrics_dir / "gate_correlations.csv", diagnostic_rows)
    write_rows(metrics_dir / "gate_benefit_quantiles.csv", quantile_rows)
    write_rows(metrics_dir / "adaptive_depth_histogram.csv", allocation_rows)
    write_rows(metrics_dir / "posthoc_regime_allocation.csv", posthoc_rows)
    write_json(metrics_dir / "posthoc_regime_thresholds.json", thresholds)
    np.savez_compressed(
        metrics_dir / "test_metrics.npz",
        raw_losses=test_raw,
        whitened_losses=test_white,
        episode_id=arrays["episode_id"][test_indices],
        model_step=arrays["model_step"][test_indices],
        predicted_benefits=test_predictions,
        actual_benefits=test_benefits,
        adaptive_outputs=adaptive_outputs,
        adaptive_allocation=adaptive_allocation,
        oracle_allocation=oracle_allocation,
        physical_regime=test_regimes,
    )
    npz_audits = {
        str(path.relative_to(output_dir)): integrity_lib.audit_npz_finite(path)
        for path in (
            cache_path,
            cache_dir / "calibration_whitening.npz",
            feature_cache_path,
            allocation_freeze_path,
            metrics_dir / "test_metrics.npz",
        )
    }

    comparison_plot_rows = []
    for comparison in dict.fromkeys(str(row["comparison"]) for row in comparison_rows):
        by_metric = {
            str(row["metric"]): row
            for row in comparison_rows
            if row["comparison"] == comparison
        }
        comparison_plot_rows.append(
            {
                "comparison": comparison,
                "raw_mean_difference": by_metric["raw"]["mean_benefit"],
                "raw_ci_lower": by_metric["raw"]["ci_low"],
                "raw_ci_upper": by_metric["raw"]["ci_high"],
                "whitened_mean_difference": by_metric["whitened"]["mean_benefit"],
                "whitened_ci_lower": by_metric["whitened"]["ci_low"],
                "whitened_ci_upper": by_metric["whitened"]["ci_high"],
            }
        )
    calibration_plot_rows = calibration_diagnostic_rows
    quantile_plot_rows = [
        {
            "benefit": row["benefit"],
            "quantile": int(row["predicted_score_quantile"]) + 1,
            "mean_predicted": row["mean_predicted_benefit"],
            "mean_actual": row["mean_actual_benefit"],
            "count": row["n"],
        }
        for row in calibration_quantiles
    ]
    regime_plot_rows = []
    uniform_raw = selected_raw["uniform_d2"]
    for summary in posthoc_rows:
        regime = str(summary["regime"])
        mask = test_regimes == regime
        histogram = allocation_lib.depth_histogram(adaptive_allocation[mask])
        benefit_vs_uniform = float((uniform_raw[mask] - selected_raw["adaptive"][mask]).mean())
        for depth, count in histogram.items():
            regime_plot_rows.append(
                {
                    "regime": regime,
                    "depth": depth,
                    "count": count,
                    "fraction": count / int(mask.sum()),
                    "raw_benefit_vs_uniform": benefit_vs_uniform,
                    "n": int(mask.sum()),
                }
            )
    figures = plots_lib.write_all_figures(
        policy_rows=policy_rows,
        comparison_rows=comparison_plot_rows,
        calibration_rows=calibration_plot_rows,
        quantile_rows=quantile_plot_rows,
        allocation_rows=allocation_rows,
        regime_rows=regime_plot_rows,
        output_dir=figures_dir,
    )
    figure_audits = figures["pngs"]

    computed_refiner_flops = int(
        sum(
            2 * module.in_features * module.out_features
            for module in refiner.block
            if isinstance(module, nn.Linear)
        )
    )
    if computed_refiner_flops != REFINER_FLOPS_PER_CALL:
        raise RuntimeError(
            f"refiner FLOP recomputation {computed_refiner_flops} != {REFINER_FLOPS_PER_CALL}"
        )
    gate_parameter_count = int(gate_lib.gate_parameter_count(len(feature_names)))
    gate_flops = int(gate_lib.gate_dense_linear_flops(len(feature_names)))
    adaptive_calls = int(np.sum(adaptive_allocation, dtype=np.int64))
    compute = {
        "device": str(device),
        "refiner_flops_per_call": REFINER_FLOPS_PER_CALL,
        "refiner_flops_per_call_recomputed": computed_refiner_flops,
        "gate_parameter_count": gate_parameter_count,
        "gate_flops_per_transition": gate_flops,
        "adaptive_first_calls": n_test,
        "adaptive_continuation_calls": int(np.sum(adaptive_allocation - 1, dtype=np.int64)),
        "adaptive_total_calls": adaptive_calls,
        "adaptive_refiner_flops": REFINER_FLOPS_PER_CALL * adaptive_calls,
        "test_gate_total_flops": gate_flops * n_test,
        "adaptive_refiner_plus_gate_flops": REFINER_FLOPS_PER_CALL * adaptive_calls + gate_flops * n_test,
        "dense_exit_diagnostic_calls": int(test_dense_stats["processed_rows"]),
        "latency": {
            "first_call": first_latency,
            "gate_feature_mlp_allocation": gate_latency,
            "adaptive_continuation": adaptive_latency,
            "uniform_depth2_execution": uniform_latency,
        },
    }

    refiner_after = validate_frozen_module(refiner, "refiner")
    if refiner_before != refiner_after:
        raise RuntimeError("frozen refiner state/audit changed during V2")
    gate_after = validate_frozen_module(selected_gate.model, "selected gate")
    if gate_before != gate_after:
        raise RuntimeError("frozen selected-gate state/audit changed during V2")
    frozen_file_hashes_after = {
        "base_config": require_hash(args.model_config, EXPECTED_HASHES["base_config"], "base config after"),
        "base_weights": require_hash(args.model_weights, EXPECTED_HASHES["base_weights"], "base weights after"),
        "refiner_source": require_hash(args.refiner_checkpoint, EXPECTED_HASHES["refiner"], "V1 refiner after"),
    }
    if frozen_file_hashes_before != frozen_file_hashes_after:
        raise RuntimeError("frozen source hashes changed during V2")
    v1_hashes_after = tree_hashes(v1_root)
    if v1_hashes_before != v1_hashes_after:
        raise RuntimeError("V1 artifact tree changed during V2")

    verdict, reason = frozen_verdict(policy_rows, comparison_rows)
    provenance = {
        "mode": cfg["mode"],
        "source": extraction_manifest.get("source", {}),
        "base_model": extraction_manifest.get("model", {}),
        "base_freeze_audit": extraction_manifest.get("freeze_audit", {}),
        "split": split_plan,
        "frozen_files": {
            name: {"path": str(path.resolve()), "before": frozen_file_hashes_before[name], "after": frozen_file_hashes_after[name]}
            for name, path in {
                "base_config": args.model_config,
                "base_weights": args.model_weights,
                "refiner_source": args.refiner_checkpoint,
            }.items()
        },
        "copied_refiner": {
            "path": str(copied_refiner),
            "sha256": sha256_file(copied_refiner),
            "byte_identical_to_source": True,
            "frozen_audit_before": refiner_before,
            "frozen_audit_after": refiner_after,
        },
        "refiner_consistency": refiner_checks,
        "first_call_accounting": first_call_accounting,
        "causal_feature_cache": {
            "path": str(feature_cache_path),
            "sha256": sha256_file(feature_cache_path),
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
        },
        "gate": {
            "selected_seed": int(selected_gate.seed),
            "selected_epoch": int(selected_gate.best_epoch),
            "checkpoint_path": str(primary_gate_checkpoint),
            "checkpoint_sha256": sha256_file(primary_gate_checkpoint),
            "frozen_audit_before": gate_before,
            "frozen_audit_after": gate_after,
            "provenance_binding": gate_binding,
        },
        "allocation_freeze": {
            "path": str(allocation_freeze_path),
            "sha256": allocation_freeze_hash,
            "created_before_test_target_scoring": True,
        },
        "v1_artifact_tree_unchanged": True,
        "v1_artifact_count": len(v1_hashes_before),
        "figure_audits": figure_audits,
        "finite_npz_audits": npz_audits,
        "base_pretraining_episode_membership": "unknown",
    }
    write_json(output_dir / "provenance.json", provenance)
    decision = {
        "verdict": verdict,
        "reason": reason,
        "mode": cfg["mode"],
        "confirmatory": cfg["mode"] == "full",
        "selected_gate_seed": int(selected_gate.seed),
        "selected_gate_epoch": int(selected_gate.best_epoch),
        "policy_metrics": policy_rows,
        "paired_comparisons": comparison_rows,
        "compute": compute,
        "validity": {
            "strict_base_hash_and_load": True,
            "strict_refiner_hash_keys_and_freeze": True,
            "base_and_refiner_unchanged": True,
            "refiner_output_consistency": True,
            "first_call_for_every_transition": True,
            "causal_features_only": True,
            "episode_splits_and_exclusions_valid": True,
            "gate_selected_on_calibration_only": True,
            "test_allocation_frozen_before_outcome": True,
            "adaptive_uniform_exact_2n_calls": True,
            "strict_finite_artifacts": True,
            "posthoc_labels_after_allocation": True,
            "v1_artifacts_unchanged": True,
        },
        "artifacts": {
            "provenance": str(output_dir / "provenance.json"),
            "metrics": str(metrics_dir),
            "figures": figures,
            "causal_feature_cache": str(feature_cache_path),
            "allocation_freeze": str(allocation_freeze_path),
        },
    }
    write_json(output_dir / "decision.json", decision)
    write_report(
        output_dir / "REPORT.md",
        decision=decision,
        policy_rows=policy_rows,
        comparison_rows=comparison_rows,
        diagnostic_rows=diagnostic_rows,
        quantile_rows=quantile_rows,
        regime_summary=posthoc_rows,
        provenance=provenance,
        figures=figures,
    )
    # Re-read every JSON written by the driver under strict parsing.
    for json_path in output_dir.rglob("*.json"):
        json.loads(json_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    print_step(f"complete: {verdict}")
    return decision


def main() -> None:
    args = parse_args()
    requested_output = args.output_dir.resolve()
    try:
        requested_output.relative_to(EXPERIMENT_DIR.resolve())
    except ValueError as exc:
        # Never create or write to an unsafe path, including V1, even when
        # reporting an invalid invocation.
        failure = {
            "verdict": "invalid",
            "reason": "RuntimeError: V2 output directory must remain inside the isolated V2 run directory",
            "confirmatory": False,
            "validity": {"pipeline_completed": False},
        }
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False), file=sys.stderr)
        raise RuntimeError(failure["reason"]) from exc
    try:
        decision = _run(args)
    except Exception as exc:
        output_dir = requested_output
        output_dir.mkdir(parents=True, exist_ok=True)
        verdict = "blocked" if isinstance(exc, FileNotFoundError) else "invalid"
        failure = {
            "verdict": verdict,
            "reason": f"{type(exc).__name__}: {exc}",
            "confirmatory": False,
            "validity": {"pipeline_completed": False},
        }
        write_json(output_dir / "decision.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False), file=sys.stderr)
        raise
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
