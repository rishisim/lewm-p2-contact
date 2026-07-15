"""Shared, isolation-first utilities for critic-compression discovery."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PRIOR = REPO / "runs/lewm_adaptive_compute_discovery"
TRAIN_CACHE = PRIOR / "cache/v3_train_only.npz"
SOLVER_CHECKPOINT = PRIOR / "checkpoints/stagewise_seed_261102.pt"
PREPARED_CACHE = ROOT / "cache/stagewise_train_outputs.npz"
PREPARED_MANIFEST = ROOT / "cache/stagewise_train_outputs_manifest.json"


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
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def load_config() -> dict[str, Any]:
    return json.loads((ROOT / "config.json").read_text())


def load_prior_modules() -> tuple[Any, Any, Any, Any]:
    """Load immutable discovery code without copying or modifying it."""
    prior_text = str(PRIOR)
    if prior_text not in sys.path:
        sys.path.insert(0, prior_text)
    for name in ("data_isolation", "models", "policy"):
        if name not in sys.modules:
            __import__(name)
    spec = importlib.util.spec_from_file_location("critic_compression_prior_runner", PRIOR / "run_discovery.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable discovery runner")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner, sys.modules["models"], sys.modules["policy"], sys.modules["data_isolation"]


def load_train_arrays() -> dict[str, np.ndarray]:
    cfg = load_config()
    if sha256_file(TRAIN_CACHE) != cfg["train_cache_sha256"]:
        raise RuntimeError("isolated train-only cache hash drift")
    with np.load(TRAIN_CACHE, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    _, _, _, isolation = load_prior_modules()
    audit = isolation.assert_isolated_cache(arrays, "train")
    if audit["episodes"] != 420 or audit["test_episodes_present"]:
        raise RuntimeError("train-only cache isolation audit failed")
    return arrays


def load_frozen_solver(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, Any, Any]:
    cfg = load_config()
    if sha256_file(SOLVER_CHECKPOINT) != cfg["selected_solver_checkpoint_sha256"]:
        raise RuntimeError("selected stagewise checkpoint hash drift")
    runner, models, policy, _ = load_prior_modules()
    prior_cfg = json.loads((PRIOR / "config.json").read_text())
    v1 = runner.load_v1_refiner(device)
    model = runner.build_solver("stagewise", int(cfg["selected_solver_seed"]), prior_cfg, v1, device)
    payload = torch.load(SOLVER_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload["family"] != "stagewise" or payload["accepted_depth"] != 4:
        raise RuntimeError("selected checkpoint contract mismatch")
    if payload["anchor_sha256"] != cfg["selected_solver_anchor_state_sha256"]:
        raise RuntimeError("selected checkpoint anchor hash mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.set_trainable_stage(None)
    model.to(device).eval().requires_grad_(False)
    return model, v1, runner, models


def episode_assignment(episode_ids: np.ndarray, folds: int, seed: int) -> np.ndarray:
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    unique = np.unique(episode_ids)
    if not 2 <= int(folds) <= len(unique):
        raise ValueError("invalid grouped fold count")
    shuffled = np.random.default_rng(int(seed)).permutation(unique)
    mapping = {int(ep): int(index % folds) for index, ep in enumerate(shuffled)}
    assigned = np.asarray([mapping[int(ep)] for ep in episode_ids], dtype=np.int16)
    for episode in unique:
        if len(np.unique(assigned[episode_ids == episode])) != 1:
            raise RuntimeError("episode crossed a fold")
    return assigned


def fit_whitening(target: np.ndarray) -> dict[str, np.ndarray | float]:
    values = np.asarray(target, dtype=np.float64)
    mean = values.mean(0)
    centered = values - mean
    covariance = np.einsum("ni,nj->ij", centered, centered, optimize=False) / max(len(values) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()) * 1e-6, 1e-12)
    matrix = np.einsum(
        "ik,k,jk->ij",
        eigenvectors,
        1.0 / np.sqrt(np.maximum(eigenvalues, floor)),
        eigenvectors,
        optimize=False,
    )
    return {"mean": mean, "matrix": matrix, "eigenvalues": eigenvalues, "floor": floor}


def whitened_losses(target: np.ndarray, exits: np.ndarray, whitening: Mapping[str, Any]) -> np.ndarray:
    difference = np.asarray(exits, dtype=np.float64) - np.asarray(target, dtype=np.float64)[:, None, :]
    transformed = np.einsum("nkd,df->nkf", difference, whitening["matrix"], optimize=False)
    result = np.square(transformed).mean(2)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite whitened loss")
    return result


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


@torch.inference_mode()
def prepare_solver_cache(device: torch.device, batch_size: int = 1024) -> dict[str, Any]:
    """Mechanically freeze/audit the solver and cache causal discovery tensors."""
    cfg = load_config()
    if PREPARED_CACHE.exists() or PREPARED_MANIFEST.exists():
        if not (PREPARED_CACHE.exists() and PREPARED_MANIFEST.exists()):
            raise RuntimeError("partial prepared-cache artifact")
        manifest = json.loads(PREPARED_MANIFEST.read_text())
        if manifest["cache_sha256"] != sha256_file(PREPARED_CACHE):
            raise RuntimeError("prepared-cache hash drift")
        return manifest

    arrays = load_train_arrays()
    model, v1, runner, models = load_frozen_solver(device)
    solver_hash_before = state_hash(model)
    v1_hash_before = state_hash(v1)
    checkpoint_hash_before = sha256_file(SOLVER_CHECKPOINT)
    n = len(arrays["episode_id"])
    exits = np.empty((n, 4, 192), dtype=np.float32)
    features = np.empty((n, 3, int(cfg["feature_dim"])), dtype=np.float32)
    all_indices = np.arange(n, dtype=np.int64)
    d0_identity = True
    d0_bitwise = True
    d1_bitwise = True
    cursor = 0
    for part in iter_batches(all_indices, batch_size):
        h = torch.as_tensor(np.ascontiguousarray(arrays["history"][part]), device=device)
        a = torch.as_tensor(np.ascontiguousarray(arrays["action"][part]), device=device)
        z = torch.as_tensor(np.ascontiguousarray(arrays["base_pred"][part]), device=device)
        output, updates = model(h, a, z, max_depth=4, return_updates=True)
        expected = v1(h, a, z, depths=(0, 1))
        d0_identity = d0_identity and output[0] is z
        d0_bitwise = d0_bitwise and bool(torch.equal(output[0], expected[0]))
        d1_bitwise = d1_bitwise and bool(torch.equal(output[1], expected[1]))
        stacked = torch.stack([output[depth] for depth in (1, 2, 3, 4)], dim=1)
        b = len(part)
        exits[cursor : cursor + b] = stacked.cpu().numpy()
        for stage, depth in enumerate((1, 2, 3)):
            causal = models.build_causal_features(h, a, output[depth], updates[depth])
            if causal.shape[1] != int(cfg["feature_dim"]) or causal.requires_grad:
                raise RuntimeError("causal feature contract failed")
            features[cursor : cursor + b, stage] = causal.cpu().numpy()
        cursor += b
    target = np.asarray(arrays["target"], dtype=np.float32)
    losses = np.square(exits - target[:, None, :]).mean(2).astype(np.float32)
    if not np.isfinite(features).all() or not np.isfinite(losses).all():
        raise RuntimeError("prepared data contain nonfinite values")

    # Reproduce the already-inspected 84-episode split mechanically, without
    # using it for selection in this new all-420 nested protocol.
    _, _, _, isolation = load_prior_modules()
    old_split = isolation.grouped_discovery_split(84, 261013)
    old_val = np.isin(arrays["episode_id"], old_split["internal_validation"])
    old_means = losses[old_val].mean(0, dtype=np.float64)
    expected_means = np.asarray([
        0.003120224690064788,
        0.0030725307296961546,
        0.0030579364392906427,
        0.003048931946977973,
    ])
    # The immutable report accumulated batch sums through MPS; this fresh path
    # materializes per-row float32 losses before a float64 mean. Bitwise anchors
    # are checked separately, while the MSE reproduction allows only 5e-9.
    if not np.allclose(old_means, expected_means, rtol=0.0, atol=5e-9):
        raise RuntimeError(f"established fixed-exit result did not reproduce: {[repr(v) for v in old_means]}")

    ROOT.joinpath("cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PREPARED_CACHE,
        exits=exits,
        losses=losses,
        features=features,
        episode_id=np.asarray(arrays["episode_id"], dtype=np.int64),
        model_step=np.asarray(arrays["model_step"], dtype=np.int64),
    )
    solver_hash_after = state_hash(model)
    v1_hash_after = state_hash(v1)
    checkpoint_hash_after = sha256_file(SOLVER_CHECKPOINT)
    audit = {
        "schema_version": 1,
        "cache": str(PREPARED_CACHE),
        "cache_sha256": sha256_file(PREPARED_CACHE),
        "rows": n,
        "episodes": int(len(np.unique(arrays["episode_id"]))),
        "feature_dim": int(features.shape[2]),
        "old_internal_validation_fixed_exit_raw_mse": old_means,
        "d0_identity": d0_identity,
        "d0_bitwise": d0_bitwise,
        "d1_bitwise": d1_bitwise,
        "solver_state_sha256_before": solver_hash_before,
        "solver_state_sha256_after": solver_hash_after,
        "v1_state_sha256_before": v1_hash_before,
        "v1_state_sha256_after": v1_hash_after,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "no_solver_state_change": solver_hash_before == solver_hash_after,
        "no_v1_state_change": v1_hash_before == v1_hash_after,
        "no_checkpoint_change": checkpoint_hash_before == checkpoint_hash_after,
        "all_solver_gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
        "combined_v3_target_npz_opened": False,
        "v3_test_targets_loaded": False,
    }
    if not all(
        audit[key]
        for key in (
            "d0_identity", "d0_bitwise", "d1_bitwise", "no_solver_state_change",
            "no_v1_state_change", "no_checkpoint_change", "all_solver_gradients_absent",
        )
    ):
        raise RuntimeError("solver preservation audit failed")
    write_json(PREPARED_MANIFEST, audit)
    return audit


def load_prepared() -> dict[str, np.ndarray]:
    if not PREPARED_CACHE.exists() or not PREPARED_MANIFEST.exists():
        raise RuntimeError("run prepare before discovery")
    manifest = json.loads(PREPARED_MANIFEST.read_text())
    if manifest["cache_sha256"] != sha256_file(PREPARED_CACHE):
        raise RuntimeError("prepared-cache hash drift")
    with np.load(PREPARED_CACHE, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if arrays["features"].shape != (15960, 3, 1046):
        raise RuntimeError("prepared feature shape mismatch")
    return arrays


def feature_indices(family: str, selected: Sequence[int] | None = None) -> np.ndarray:
    if family in ("full_linear", "distilled_rank_linear", "low_rank"):
        return np.arange(1046, dtype=np.int64)
    if family == "stable_sparse":
        if selected is None:
            raise ValueError("sparse family requires selected feature indices")
        result = np.asarray(selected, dtype=np.int64)
        if result.ndim != 1 or not len(result) or len(np.unique(result)) != len(result):
            raise ValueError("invalid sparse feature indices")
        return result
    if family == "in_model_halting":
        # 576 history + 75 action, then current prediction and last update.
        return np.arange(651, 1035, dtype=np.int64)
    if family == "compact11_control":
        return np.arange(1035, 1046, dtype=np.int64)
    raise ValueError(f"unknown family {family}")


FORBIDDEN_FEATURE_TOKENS = {
    "target", "future", "contact", "impact", "regime", "label", "loss",
    "gain", "oracle", "outcome", "truth", "global", "episode",
}


def causal_feature_audit() -> dict[str, Any]:
    _, models, _, _ = load_prior_modules()
    names = models.causal_feature_names(latent_dim=192, action_dim=25, history_len=3)
    rejected = []
    for name in names:
        tokens = set(name.lower().split("_"))
        if tokens & FORBIDDEN_FEATURE_TOKENS:
            rejected.append(name)
    result = {
        "feature_count": len(names),
        "unique": len(set(names)) == len(names),
        "forbidden_names": rejected,
        "transition_local_builder_has_no_target_argument": "target" not in models.build_causal_features.__code__.co_varnames,
    }
    result["passed"] = bool(result["feature_count"] == 1046 and result["unique"] and not rejected and result["transition_local_builder_has_no_target_argument"])
    return result
