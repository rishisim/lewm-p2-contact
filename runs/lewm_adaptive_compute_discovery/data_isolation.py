"""Mechanical discovery isolation for the V3 episode split.

This module intentionally has no function capable of returning V3 test data.
The old V3 combined NPZ is treated only as an opaque file whose hash may be
checked; it is never opened with NumPy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V3_MANIFEST = REPO / "runs/lewm_adaptive_compute_v3/cache/split_manifest.json"
V3_COMBINED_CACHE = REPO / "runs/lewm_adaptive_compute_v3/cache/cube_inputs.npz"
PINNED = {
    "manifest_sha256": "4b9344b6ff2acce4c58c3b296ba2017b0122f8f8158f45088239712ce94b7030",
    "opaque_combined_cache_sha256": "21377c283136008ae07aa4d1e00407ab57eaba5ddda7d5b1c740c47e55afaba0",
    "train_episode_sha256": "3a17d970038a52efdb553322a2b60e08a1eaa952566b6591be56fff2ae0269ea",
    "calibration_episode_sha256": "a69d88220e6c08d75ebb29b2eec4153965ae36368cb29084e8bb8913ee85282d",
    "test_episode_sha256": "178d1e0fb6d0ffd563e9f71b2a87473576990d99a92be402744ece866e0f1dc7",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_digest(values: Sequence[int]) -> str:
    canonical = np.asarray(sorted(int(v) for v in values), dtype="<i4").tobytes()
    return hashlib.sha256(canonical).hexdigest()


def load_pinned_v3_episode_sets() -> dict[str, np.ndarray]:
    if sha256_file(V3_MANIFEST) != PINNED["manifest_sha256"]:
        raise RuntimeError("V3 split manifest drift")
    payload = json.loads(V3_MANIFEST.read_text())
    raw = payload["selection"]["split_episode_ordinals"]
    result = {name: np.asarray(raw[name], dtype=np.int64) for name in ("train", "calibration", "test")}
    expected = {"train": 420, "calibration": 90, "test": 90}
    for name, values in result.items():
        if len(values) != expected[name] or len(np.unique(values)) != len(values):
            raise RuntimeError(f"invalid pinned V3 {name} episode set")
        if episode_digest(values) != PINNED[f"{name}_episode_sha256"]:
            raise RuntimeError(f"pinned V3 {name} episode digest mismatch")
    if any(set(result[a]) & set(result[b]) for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test"))):
        raise RuntimeError("pinned V3 partitions overlap")
    return result


def grouped_discovery_split(validation_episodes: int, seed: int) -> dict[str, np.ndarray]:
    sets = load_pinned_v3_episode_sets()
    train = sets["train"]
    if not 1 <= int(validation_episodes) < len(train):
        raise ValueError("invalid internal-validation episode count")
    order = np.random.default_rng(int(seed)).permutation(train)
    internal = np.sort(order[: int(validation_episodes)])
    fit = np.sort(order[int(validation_episodes) :])
    if set(fit) & set(internal) or set(np.concatenate((fit, internal))) != set(train):
        raise RuntimeError("grouped discovery split failed")
    return {"discovery_fit": fit, "internal_validation": internal}


def extraction_splits(role: str) -> dict[str, np.ndarray]:
    """Return the only source-HDF5 episode allowlists discovery may extract."""
    sets = load_pinned_v3_episode_sets()
    empty = np.empty(0, dtype=np.int64)
    if role == "train":
        selected = sets["train"]
        result = {"train": selected.copy(), "calibration": empty.copy(), "test": empty.copy()}
    elif role == "calibration_once":
        selected = sets["calibration"]
        result = {"train": empty.copy(), "calibration": selected.copy(), "test": empty.copy()}
    else:
        raise ValueError("role must be 'train' or 'calibration_once'; test access is forbidden")
    if set(selected) & set(sets["test"]):
        raise RuntimeError("test episode entered extraction allowlist")
    return result


def assert_isolated_cache(arrays: Mapping[str, np.ndarray], role: str) -> dict[str, object]:
    allowed = extraction_splits(role)
    expected = set(np.concatenate(tuple(allowed.values())).tolist())
    observed = set(np.asarray(arrays["episode_id"], dtype=np.int64).tolist())
    if observed != expected:
        raise RuntimeError(f"{role} cache episode set is not exactly its allowlist")
    forbidden = set(load_pinned_v3_episode_sets()["test"].tolist())
    if observed & forbidden:
        raise RuntimeError("V3 test episode found in isolated discovery cache")
    required = {"history", "action", "base_pred", "target", "episode_id", "model_step", "split"}
    if required - set(arrays):
        raise RuntimeError("isolated cache is incomplete")
    for name in ("history", "action", "base_pred", "target"):
        if not np.isfinite(np.asarray(arrays[name])).all():
            raise RuntimeError(f"nonfinite isolated cache array {name}")
    return {"role": role, "episodes": len(observed), "rows": len(arrays["episode_id"]), "test_episodes_present": False}


def create_calibration_receipt(path: Path, tournament_sha256: str) -> dict[str, str]:
    """Irreversibly consume the single calibration access before target I/O."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "consumed", "tournament_sha256": str(tournament_sha256)}
    try:
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError("V3 calibration access has already been consumed") from exc
    return payload


def assert_combined_cache_never_opened() -> dict[str, str]:
    """Hash the old cache opaquely; never call np.load on it."""
    observed = sha256_file(V3_COMBINED_CACHE)
    if observed != PINNED["opaque_combined_cache_sha256"]:
        raise RuntimeError("opaque V3 combined cache hash mismatch")
    return {"path": str(V3_COMBINED_CACHE), "sha256": observed, "opened_with_numpy": "false"}
