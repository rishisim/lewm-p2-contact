#!/usr/bin/env python3
"""Compact PushT-specific helpers for the bounded LeWM replication pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
LEWM_ROOT = Path("/Users/rishisim/Documents/research/World Models/le-wm")
MODEL_ROOT = LEWM_ROOT / ".cache/diagnostic/model/quentinll--lewm-pusht"
MODEL_CONFIG = MODEL_ROOT / "config.json"
MODEL_WEIGHTS = MODEL_ROOT / "weights.pt"
GENERATION_PYTHON = LEWM_ROOT / ".venv/bin/python"

LATENT_DIM = 192
ACTION_DIM = 10
RAW_ACTION_DIM = 2
HISTORY = 3
FRAMESKIP = 5
MAX_DEPTH = 4
DEPTHS = (1, 2, 3, 4)


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def module_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def append_note(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (ROOT / "NOTES.md").open("a") as stream:
        stream.write(f"\n- {stamp} — {message}\n")


def update_status(phase: str, status: str = "running", **extra: Any) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        "phase": phase,
        "updated_unix_ns": time.time_ns(),
        **extra,
    }
    atomic_json(ROOT / "RUN_STATUS.json", payload)


def git_snapshot() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    tracked_diff = subprocess.run(
        ["git", "diff", "--name-only"], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return {"head": head, "status": status, "tracked_diff": tracked_diff}


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_base_model(device: torch.device) -> nn.Module:
    """Instantiate only from the two local cached model files."""
    if str(LEWM_ROOT) not in sys.path:
        sys.path.insert(0, str(LEWM_ROOT))
    import diagnose_pusht_latent_contacts as diagnostic

    model = diagnostic.instantiate_model(MODEL_CONFIG, MODEL_WEIGHTS, device)
    model.eval().requires_grad_(False)
    return model


def preprocess_pixels(pixels: np.ndarray, device: torch.device) -> Tensor:
    if str(LEWM_ROOT) not in sys.path:
        sys.path.insert(0, str(LEWM_ROOT))
    import diagnose_pusht_latent_contacts as diagnostic

    return diagnostic.preprocess_pixels(pixels, device, 224)


def _info_array(infos: Mapping[str, Any], key: str) -> np.ndarray:
    value = infos[key]
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value[0, 0]).copy()


def rollout_episode(seed: int, *, max_steps: int = 100) -> dict[str, np.ndarray | int | bool]:
    """Generate one fresh trajectory directly from swm/PushT-v1."""
    import stable_worldmodel as swm
    from stable_worldmodel.envs.pusht import WeakPolicy

    world = swm.World(
        "swm/PushT-v1", num_envs=1, image_shape=(224, 224),
        max_episode_steps=max_steps, render_mode="rgb_array",
    )
    policy = WeakPolicy(dist_constraint=100, seed=int(seed))
    world.set_policy(policy)
    pixels: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    contacts: list[float] = []
    rewards: list[float] = []
    terminateds: list[bool] = []
    truncateds: list[bool] = []
    try:
        world.reset(seed=int(seed))
        pixels.append(_info_array(world.infos, "pixels").astype(np.uint8, copy=False))
        states.append(_info_array(world.infos, "state").astype(np.float64, copy=False))
        for _ in range(max_steps):
            action_batch = np.asarray(policy.get_action(world.infos), dtype=np.float32)
            _, reward, terminated, truncated, infos = world.envs.step(action_batch)
            world.infos = infos
            actions.append(action_batch[0].copy())
            rewards.append(float(reward[0]))
            terminateds.append(bool(terminated[0]))
            truncateds.append(bool(truncated[0]))
            contacts.append(float(_info_array(infos, "n_contacts")))
            pixels.append(_info_array(infos, "pixels").astype(np.uint8, copy=False))
            states.append(_info_array(infos, "state").astype(np.float64, copy=False))
            if terminated[0] or truncated[0]:
                break
    finally:
        world.close()

    raw_actions = np.asarray(actions, dtype=np.float32)
    raw_pixels = np.asarray(pixels, dtype=np.uint8)
    raw_states = np.asarray(states, dtype=np.float64)
    blocks = len(raw_actions) // FRAMESKIP
    obs_indices = np.arange(blocks + 1, dtype=np.int64) * FRAMESKIP
    strided_pixels = raw_pixels[obs_indices]
    blocked_actions = raw_actions[: blocks * FRAMESKIP].reshape(blocks, ACTION_DIM)
    return {
        "seed": int(seed),
        "pixels": strided_pixels,
        "blocked_actions": blocked_actions,
        "raw_actions": raw_actions,
        "raw_states": raw_states,
        "contacts": np.asarray(contacts, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminated": np.asarray(terminateds, dtype=np.bool_),
        "truncated": np.asarray(truncateds, dtype=np.bool_),
        "raw_steps": int(len(raw_actions)),
        "success": bool(any(terminateds)),
    }


def trajectory_digest(episode: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in (
        "pixels", "blocked_actions", "raw_actions", "raw_states", "contacts",
        "rewards", "terminated", "truncated",
    ):
        digest.update(key.encode())
        digest.update(array_digest(np.asarray(episode[key])).encode())
    return digest.hexdigest()


def encode_pixel_sequences(
    model: nn.Module, episodes: Sequence[Mapping[str, Any]], device: torch.device,
    *, batch_size: int = 64,
) -> list[np.ndarray]:
    lengths = [len(np.asarray(item["pixels"])) for item in episodes]
    all_pixels = np.concatenate([np.asarray(item["pixels"]) for item in episodes], axis=0)
    chunks: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(all_pixels), batch_size):
            batch = preprocess_pixels(all_pixels[start : start + batch_size], device)
            encoded = model.encode({"pixels": batch.unsqueeze(0)})["emb"].squeeze(0)
            chunks.append(encoded.detach().cpu())
    joined = torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)
    if joined.shape[1] != LATENT_DIM or not np.isfinite(joined).all():
        raise RuntimeError(f"invalid encoded latent contract: {joined.shape}")
    outputs: list[np.ndarray] = []
    cursor = 0
    for length in lengths:
        outputs.append(joined[cursor : cursor + length].copy())
        cursor += length
    return outputs


def transition_arrays(
    model: nn.Module, episodes: Sequence[Mapping[str, Any]], latents: Sequence[np.ndarray],
    episode_ordinals: Sequence[int], device: torch.device, *, predict_batch_size: int = 512,
) -> dict[str, np.ndarray]:
    histories: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_id: list[int] = []
    episode_seed: list[int] = []
    model_step: list[int] = []
    for item, embedding, ordinal in zip(episodes, latents, episode_ordinals):
        blocked = np.asarray(item["blocked_actions"], dtype=np.float32)
        for target_step in range(HISTORY, len(embedding)):
            start = target_step - HISTORY
            histories.append(embedding[start:target_step])
            actions.append(blocked[start:target_step])
            targets.append(embedding[target_step])
            episode_id.append(int(ordinal))
            episode_seed.append(int(item["seed"]))
            model_step.append(target_step)
    if not histories:
        raise RuntimeError("role produced no eligible one-step transitions")
    history_array = np.asarray(histories, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    base_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(history_array), predict_batch_size):
            history = torch.from_numpy(history_array[start : start + predict_batch_size]).to(device)
            action = torch.from_numpy(action_array[start : start + predict_batch_size]).to(device)
            action_embedding = model.action_encoder(action)
            base = model.predict(history, action_embedding)[:, -1]
            base_chunks.append(base.detach().cpu().numpy().astype(np.float32))
    base_array = np.concatenate(base_chunks, axis=0)
    arrays = {
        "episode_id": np.asarray(episode_id, dtype=np.int32),
        "episode_seed": np.asarray(episode_seed, dtype=np.int64),
        "model_step": np.asarray(model_step, dtype=np.int16),
        "history": history_array,
        "actions": action_array,
        "base": base_array,
        "target": target_array,
    }
    expected = (len(history_array), HISTORY, LATENT_DIM)
    if history_array.shape != expected or action_array.shape != (len(history_array), HISTORY, ACTION_DIM):
        raise RuntimeError("history/action blocking contract failed")
    if base_array.shape != target_array.shape or base_array.shape[1] != LATENT_DIM:
        raise RuntimeError("base/target latent contract failed")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("nonfinite role array")
    return arrays


def concatenate_transition_batches(batches: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = tuple(batches[0])
    if any(tuple(batch) != names for batch in batches):
        raise RuntimeError("transition batch schema drift")
    return {name: np.concatenate([batch[name] for batch in batches], axis=0) for name in names}


def pack_context(episodes: Sequence[Mapping[str, Any]], episode_ordinals: Sequence[int]) -> dict[str, np.ndarray]:
    count = len(episodes)
    max_steps = max(int(item["raw_steps"]) for item in episodes)
    states = np.full((count, max_steps + 1, 7), np.nan, dtype=np.float64)
    actions = np.full((count, max_steps, RAW_ACTION_DIM), np.nan, dtype=np.float32)
    contacts = np.full((count, max_steps), np.nan, dtype=np.float32)
    rewards = np.full((count, max_steps), np.nan, dtype=np.float32)
    for row, item in enumerate(episodes):
        length = int(item["raw_steps"])
        states[row, : length + 1] = np.asarray(item["raw_states"])
        actions[row, :length] = np.asarray(item["raw_actions"])
        contacts[row, :length] = np.asarray(item["contacts"])
        rewards[row, :length] = np.asarray(item["rewards"])
    return {
        "episode_id": np.asarray(episode_ordinals, dtype=np.int32),
        "seed": np.asarray([int(item["seed"]) for item in episodes], dtype=np.int64),
        "raw_steps": np.asarray([int(item["raw_steps"]) for item in episodes], dtype=np.int16),
        "success": np.asarray([bool(item["success"]) for item in episodes], dtype=np.bool_),
        "states": states,
        "raw_actions": actions,
        "contacts": contacts,
        "rewards": rewards,
    }


class StagewiseRefiner(nn.Module):
    """One recurrent action-conditioned residual block with exits 1--4."""

    def __init__(
        self, latent_dim: int = LATENT_DIM, action_dim: int = ACTION_DIM,
        history: int = HISTORY, hidden: int = 256, iteration_dim: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history = history
        self.hidden = hidden
        self.iteration_dim = iteration_dim
        self.iteration_embedding = nn.Embedding(MAX_DEPTH, iteration_dim)
        width = history * latent_dim + history * action_dim + latent_dim + iteration_dim
        self.input_width = width
        self.block = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def step(self, history: Tensor, actions: Tensor, current: Tensor, iteration: int) -> tuple[Tensor, Tensor]:
        ids = torch.full((len(current),), int(iteration), dtype=torch.long, device=current.device)
        value = torch.cat(
            (history.flatten(1), actions.flatten(1), current, self.iteration_embedding(ids)), dim=1,
        )
        update = self.block(value)
        return current + update, update

    def dense(self, history: Tensor, actions: Tensor, base: Tensor) -> tuple[Tensor, Tensor]:
        outputs: list[Tensor] = []
        updates: list[Tensor] = []
        current = base
        for iteration in range(MAX_DEPTH):
            current, update = self.step(history, actions, current, iteration)
            outputs.append(current)
            updates.append(update)
        return torch.stack(outputs, dim=1), torch.stack(updates, dim=1)


def action_normalization(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(arrays["actions"], dtype=np.float64).reshape(-1, ACTION_DIM)
    mean = actions.mean(axis=0)
    scale = actions.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def normalize_actions(actions: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((actions.astype(np.float64) - mean) / scale).astype(np.float32)


def fit_whitening(target: np.ndarray, floor_ratio: float = 1e-3) -> tuple[np.ndarray, dict[str, Any]]:
    value = np.asarray(target, dtype=np.float64)
    mean = value.mean(axis=0)
    centered = value - mean
    covariance = centered.T @ centered / max(len(value) - 1, 1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()) * floor_ratio, 1e-10)
    used = np.maximum(eigenvalues, floor)
    matrix = (vectors * (1.0 / np.sqrt(used))) @ vectors.T
    details = {
        "fit_rows": len(value), "floor_ratio": floor_ratio, "floor": floor,
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "used_condition_number": float(used.max() / used.min()),
    }
    if not np.isfinite(matrix).all():
        raise RuntimeError("nonfinite fit whitening")
    return matrix, {**details, "mean": mean, "eigenvalues": eigenvalues, "used": used}


def evaluate_refiner_dense(
    model: StagewiseRefiner, arrays: Mapping[str, np.ndarray], action_mean: np.ndarray,
    action_scale: np.ndarray, device: torch.device, *, batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    exits: list[np.ndarray] = []
    updates: list[np.ndarray] = []
    normalized = normalize_actions(arrays["actions"], action_mean, action_scale)
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            history = torch.from_numpy(np.ascontiguousarray(arrays["history"][start : start + batch_size])).to(device)
            actions = torch.from_numpy(np.ascontiguousarray(normalized[start : start + batch_size])).to(device)
            base = torch.from_numpy(np.ascontiguousarray(arrays["base"][start : start + batch_size])).to(device)
            out, upd = model.dense(history, actions, base)
            exits.append(out.detach().cpu().numpy().astype(np.float32))
            updates.append(upd.detach().cpu().numpy().astype(np.float32))
    joined_exits = np.concatenate(exits)
    joined_updates = np.concatenate(updates)
    if joined_exits.shape != (len(normalized), MAX_DEPTH, LATENT_DIM):
        raise RuntimeError("dense refiner exit shape failure")
    if not np.isfinite(joined_exits).all() or not np.isfinite(joined_updates).all():
        raise RuntimeError("nonfinite refiner output")
    return joined_exits, joined_updates


def raw_white_losses(exits: np.ndarray, target: np.ndarray, whitening: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    difference = exits.astype(np.float64) - target.astype(np.float64)[:, None, :]
    raw = np.square(difference).mean(axis=2)
    transformed = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    white = np.square(transformed).mean(axis=2)
    return raw, white


def causal_feature_names() -> tuple[str, ...]:
    names = [f"history_latent_t{t}_d{d}" for t in range(HISTORY) for d in range(LATENT_DIM)]
    names += [f"normalized_action_t{t}_d{d}" for t in range(HISTORY) for d in range(ACTION_DIM)]
    names += [f"current_prediction_d{d}" for d in range(LATENT_DIM)]
    names += [f"last_update_d{d}" for d in range(LATENT_DIM)]
    names += [
        "current_norm", "last_update_norm", "relative_update_norm", "last_history_norm",
        "current_history_distance", "update_current_cosine", "update_history_distance_cosine",
    ]
    names += [f"history_change_t{t}_t{t+1}_norm" for t in range(HISTORY - 1)]
    names += [f"action_change_t{t}_t{t+1}_norm" for t in range(HISTORY - 1)]
    return tuple(names)


def build_causal_features(history: Tensor, actions: Tensor, current: Tensor, update: Tensor) -> Tensor:
    """No target/future/simulator argument exists at this boundary."""
    history = history.detach()
    actions = actions.detach()
    current = current.detach()
    update = update.detach()
    epsilon = torch.finfo(current.dtype).eps
    last = history[:, -1]
    gap = current - last
    current_norm = torch.linalg.vector_norm(current, dim=1, keepdim=True)
    update_norm = torch.linalg.vector_norm(update, dim=1, keepdim=True)
    last_norm = torch.linalg.vector_norm(last, dim=1, keepdim=True)
    gap_norm = torch.linalg.vector_norm(gap, dim=1, keepdim=True)
    relative = update_norm / current_norm.clamp_min(epsilon)
    update_current = (update * current).sum(1, keepdim=True) / (update_norm * current_norm).clamp_min(epsilon)
    update_gap = (update * gap).sum(1, keepdim=True) / (update_norm * gap_norm).clamp_min(epsilon)
    history_change = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=2)
    action_change = torch.linalg.vector_norm(actions[:, 1:] - actions[:, :-1], dim=2)
    features = torch.cat(
        (
            history.flatten(1), actions.flatten(1), current, update, current_norm, update_norm,
            relative, last_norm, gap_norm, update_current, update_gap, history_change, action_change,
        ), dim=1,
    )
    if features.shape[1] != len(causal_feature_names()):
        raise RuntimeError("causal feature width drift")
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("nonfinite causal features")
    return features


def feature_arrays(
    arrays: Mapping[str, np.ndarray], exits: np.ndarray, updates: np.ndarray,
    action_mean: np.ndarray, action_scale: np.ndarray, *, batch_size: int = 1024,
) -> np.ndarray:
    normalized = normalize_actions(arrays["actions"], action_mean, action_scale)
    width = len(causal_feature_names())
    result = np.empty((len(normalized), 3, width), dtype=np.float32)
    for start in range(0, len(normalized), batch_size):
        stop = min(start + batch_size, len(normalized))
        history = torch.from_numpy(np.ascontiguousarray(arrays["history"][start:stop]))
        actions = torch.from_numpy(np.ascontiguousarray(normalized[start:stop]))
        for stage in range(3):
            current = torch.from_numpy(np.ascontiguousarray(exits[start:stop, stage]))
            update = torch.from_numpy(np.ascontiguousarray(updates[start:stop, stage]))
            result[start:stop, stage] = build_causal_features(history, actions, current, update).numpy()
    return result


def fit_stagewise_ridge(
    features: np.ndarray, raw_gain: np.ndarray, white_gain: np.ndarray,
    regularizations: Sequence[float],
) -> dict[str, np.ndarray]:
    feature_mean = features.astype(np.float64).mean(axis=0)
    feature_scale = features.astype(np.float64).std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    raw_mean = raw_gain.mean(axis=0)
    white_mean = white_gain.mean(axis=0)
    raw_scale = raw_gain.std(axis=0) + 1e-12
    white_scale = white_gain.std(axis=0) + 1e-12
    width = features.shape[2]
    weights = np.empty((len(regularizations), 3, 2, width), dtype=np.float64)
    for stage in range(3):
        z = (features[:, stage].astype(np.float64) - feature_mean[stage]) / feature_scale[stage]
        y = np.column_stack(
            ((raw_gain[:, stage] - raw_mean[stage]) / raw_scale[stage],
             (white_gain[:, stage] - white_mean[stage]) / white_scale[stage])
        )
        gram = z.T @ z
        cross = z.T @ y
        eigenvalues, vectors = np.linalg.eigh(gram)
        projected = vectors.T @ cross
        for index, alpha in enumerate(regularizations):
            solution = vectors @ (projected / (eigenvalues[:, None] + float(alpha)))
            weights[index, stage] = solution.T
    values = {
        "regularizations": np.asarray(regularizations, dtype=np.float64),
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "raw_gain_mean": raw_mean, "raw_gain_scale": raw_scale,
        "white_gain_mean": white_mean, "white_gain_scale": white_scale,
        "weights": weights,
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise RuntimeError("nonfinite gate fit")
    return values


def gate_scores(features: np.ndarray, fitted: Mapping[str, np.ndarray], regularization_index: int) -> np.ndarray:
    output = np.empty((len(features), 3), dtype=np.float64)
    for stage in range(3):
        z = (features[:, stage].astype(np.float64) - fitted["feature_mean"][stage]) / fitted["feature_scale"][stage]
        heads = z @ fitted["weights"][regularization_index, stage].T
        output[:, stage] = np.minimum(heads[:, 0], heads[:, 1])
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite gate score")
    return output


def sequential_calls(scores: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    calls = np.ones(len(scores), dtype=np.int64)
    active = np.ones(len(scores), dtype=bool)
    reached: list[np.ndarray] = []
    for stage in range(3):
        reached.append(active.copy())
        active &= scores[:, stage] > thresholds[stage]
        calls += active.astype(np.int64)
    return calls, reached


def episode_means(values: np.ndarray, episode_id: np.ndarray, episode_count: int | None = None) -> np.ndarray:
    identifiers = np.unique(episode_id) if episode_count is None else np.arange(episode_count)
    output = []
    for identifier in identifiers:
        selected = values[episode_id == identifier]
        if len(selected) == 0:
            raise RuntimeError(f"episode {identifier} has no prediction transitions")
        output.append(float(selected.mean()))
    return np.asarray(output, dtype=np.float64)


def strongest_analytic_mixture(
    losses: np.ndarray, mean_depth: float, episode_id: np.ndarray | None = None,
    episode_count: int | None = None,
) -> dict[str, Any]:
    best: tuple[float, np.ndarray, int, int, float] | None = None
    for lower in DEPTHS:
        for upper in DEPTHS:
            if upper < lower or not lower <= mean_depth <= upper:
                continue
            weight = 0.0 if lower == upper else (mean_depth - lower) / (upper - lower)
            values = (1.0 - weight) * losses[:, lower - 1] + weight * losses[:, upper - 1]
            objective = (
                float(values.mean())
                if episode_id is None
                else float(episode_means(values, episode_id, episode_count).mean())
            )
            candidate = (objective, values, lower, upper, float(weight))
            if best is None or candidate[0] < best[0] or (
                candidate[0] == best[0] and candidate[2:4] < best[2:4]
            ):
                best = candidate
    if best is None:
        raise RuntimeError(f"no feasible analytic mixture for mean depth {mean_depth}")
    return {
        "mean_loss": best[0], "loss": best[1], "depth_lower": best[2],
        "depth_upper": best[3], "weight_upper": best[4],
    }


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def bootstrap_interval(
    episode_values: np.ndarray, *, seed: int, replicates: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    values = np.asarray(episode_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)), dtype=np.int32)
    sampled = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "mean_benefit": float(values.mean()),
        "ci_low": float(np.quantile(sampled, alpha / 2)),
        "ci_high": float(np.quantile(sampled, 1 - alpha / 2)),
        "confidence": confidence, "bootstrap_replicates": replicates,
        "bootstrap_seed": seed, "episode_count": len(values), "exploratory": True,
    }


def refiner_flops(hidden: int = 256, iteration_dim: int = 16) -> dict[str, int]:
    width = HISTORY * LATENT_DIM + HISTORY * ACTION_DIM + LATENT_DIM + iteration_dim
    linear = 2 * width * hidden + 2 * hidden * hidden + 2 * hidden * LATENT_DIM
    gelu = 8 * hidden * 2
    residual_add = LATENT_DIM
    return {
        "linear_flops_per_call": linear,
        "gelu_flops_per_call": gelu,
        "residual_add_flops_per_call": residual_add,
        "total_flops_per_call": linear + gelu + residual_add,
    }


def gate_operation_ledger() -> dict[str, int]:
    changes = HISTORY - 1
    feature = LATENT_DIM
    feature += 4 * (LATENT_DIM + (LATENT_DIM - 1) + 1)
    feature += 1
    feature += 2 * (LATENT_DIM + (LATENT_DIM - 1) + 1 + 1)
    feature += changes * (LATENT_DIM + LATENT_DIM + (LATENT_DIM - 1) + 1)
    feature += changes * (ACTION_DIM + ACTION_DIM + (ACTION_DIM - 1) + 1)
    width = len(causal_feature_names())
    score = 2 * (2 * width)
    return {
        "feature_width": width,
        "feature_flops_per_reached_evaluation": feature,
        "dual_affine_score_flops_per_reached_evaluation": score,
        "total_flops_per_reached_evaluation": feature + score,
        "nonflop_comparison_min_operations_per_reached_evaluation": 5,
        "action_normalization_flops_per_transition": 2 * HISTORY * ACTION_DIM,
    }


def base_predictor_counted_flops() -> dict[str, int]:
    """Fixed linear/attention-core ledger for one 3-token base prediction."""
    tokens, dim, action_in, action_smooth, action_hidden = 3, 192, 10, 10, 768
    heads, head_dim, inner, mlp, layers = 16, 64, 16 * 64, 2048, 6
    action = (
        2 * action_in * action_smooth * tokens
        + 2 * action_smooth * action_hidden * tokens
        + 2 * action_hidden * dim * tokens
    )
    # The causal attention pair count for T=3 is 1+2+3=6.
    attention_pairs = tokens * (tokens + 1) // 2
    per_layer = (
        2 * dim * (6 * dim) * tokens
        + 2 * dim * (3 * inner) * tokens
        + 2 * heads * attention_pairs * head_dim * 2
        + 2 * inner * dim * tokens
        + 2 * dim * mlp * tokens
        + 2 * mlp * dim * tokens
    )
    pred_projection = 2 * dim * 2048 * tokens + 2 * 2048 * dim * tokens
    total = action + layers * per_layer + pred_projection
    return {
        "action_encoder_linear_conv_flops_per_transition": action,
        "predictor_transformer_linear_attention_flops_per_transition": layers * per_layer,
        "prediction_projection_linear_flops_per_transition": pred_projection,
        "total_flops_per_transition": total,
    }


def summarize_context(contexts: Iterable[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for context in contexts:
        role = str(np.asarray(context["role_name"]).item()) if "role_name" in context else "unknown"
        lengths = context["raw_steps"].astype(int)
        contact_counts = []
        reward_means = []
        block_motion = []
        for row, length in enumerate(lengths):
            contact_counts.append(float(np.nansum(context["contacts"][row, :length] > 0)))
            reward_means.append(float(np.nanmean(context["rewards"][row, :length])))
            states = context["states"][row, : length + 1]
            block_motion.append(float(np.linalg.norm(np.diff(states[:, 2:4], axis=0), axis=1).sum()))
        summaries[role] = {
            "episode_count": len(lengths), "mean_raw_steps": float(lengths.mean()),
            "success_fraction": float(context["success"].mean()),
            "mean_contact_positive_steps": float(np.mean(contact_counts)),
            "mean_reward": float(np.mean(reward_means)),
            "mean_total_block_motion_pixels": float(np.mean(block_motion)),
        }
    return summaries
