"""Shared, fail-closed utilities for the Cube distribution-contract study.

The scientific runtime is not copied or retrained here.  This module exposes
the already-sealed V4 constants and loaders, while keeping every new artifact
under ``runs/lewm_adaptive_compute_distribution_contract``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


STUDY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STUDY_ROOT.parents[1]
V4_ROOT = REPO_ROOT / "runs/lewm_adaptive_compute_v4"
DISCOVERY_ROOT = REPO_ROOT / "runs/lewm_adaptive_compute_discovery"
COMPRESSION_ROOT = REPO_ROOT / "runs/lewm_adaptive_compute_critic_compression"
SOURCE_H5 = Path(
    "/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/"
    "cube_single_expert.h5"
)
GENERATION_PYTHON = Path(
    "/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python"
)
EVALUATION_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Re-export the immutable V4 contract so the unchanged V4 runtime can import
# this module under the plain name ``common`` and see exactly the same values.
_v4_common = _load_module("distribution_contract_v4_common", V4_ROOT / "common.py")
for _name in dir(_v4_common):
    if not _name.startswith("__") and _name not in globals():
        globals()[_name] = getattr(_v4_common, _name)


_RUNTIME: Any | None = None
_MODEL_IO: Any | None = None
_ISOLATION: Any | None = None


def load_runtime() -> Any:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _load_module(
            "distribution_contract_frozen_runtime", V4_ROOT / "runtime.py"
        )
    return _RUNTIME


def load_model_io() -> Any:
    global _MODEL_IO
    if _MODEL_IO is None:
        _MODEL_IO = _load_module(
            "distribution_contract_model_io",
            REPO_ROOT / "runs/lewm_adaptive_compute_v2/model_io.py",
        )
    return _MODEL_IO


def load_isolation() -> Any:
    global _ISOLATION
    if _ISOLATION is None:
        _ISOLATION = _load_module(
            "distribution_contract_data_isolation",
            DISCOVERY_ROOT / "data_isolation.py",
        )
    return _ISOLATION


def study_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_study_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray], *, exclusive: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise RuntimeError(f"artifact already exists: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def canonical_int_digest(values: Iterable[int]) -> str:
    array = np.asarray(sorted({int(value) for value in values}), dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def recursive_integers(value: Any) -> set[int]:
    result: set[int] = set()
    if isinstance(value, bool):
        return result
    if isinstance(value, int):
        result.add(int(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            result.update(recursive_integers(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(recursive_integers(item))
    return result


def load_allowed_cache(role: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if role == "offline_discovery":
        path = DISCOVERY_ROOT / "cache/v3_train_only.npz"
        isolation_role = "train"
    elif role == "offline_calibration":
        path = COMPRESSION_ROOT / "cache/v3_calibration_only.npz"
        isolation_role = "calibration_once"
    else:
        raise ValueError("role must be offline_discovery or offline_calibration")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    audit = load_isolation().assert_isolated_cache(arrays, isolation_role)
    sets = load_isolation().load_pinned_v3_episode_sets()
    observed = set(np.asarray(arrays["episode_id"], dtype=np.int64).tolist())
    test = set(np.asarray(sets["test"], dtype=np.int64).tolist())
    if observed & test:
        raise RuntimeError("V3 test episode entered allowed cache")
    return arrays, {
        **audit,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "v3_test_episode_intersection": [],
        "v3_test_targets_opened": False,
    }


def role_raw_paths(phase: str, policy: str) -> tuple[Path, Path]:
    if phase not in {"smoke", "main"} or policy not in {"plan_oracle", "markov_oracle"}:
        raise ValueError("invalid phase/policy")
    data = STUDY_ROOT / "data" / f"{phase}_{policy}_raw"
    manifest = STUDY_ROOT / "data" / f"{phase}_{policy}_manifest.json"
    return data, manifest


def assert_pre_generation_seal() -> dict[str, Any]:
    path = STUDY_ROOT / "audit/pre_generation_seal.json"
    if not path.exists():
        raise RuntimeError("pre-generation seal is missing")
    payload = study_json(path)
    if payload.get("status") != "frozen_before_any_smoke_or_main_rollout":
        raise RuntimeError("invalid pre-generation seal")
    for relative, expected in payload["sealed_files"].items():
        candidate = REPO_ROOT / relative
        if sha256_file(candidate) != expected:
            raise RuntimeError(f"sealed source drift: {candidate}")
    return payload


def v3_test_set() -> set[int]:
    return set(load_isolation().load_pinned_v3_episode_sets()["test"].astype(int).tolist())


FEATURE_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("history_latents", 0, 576),
    ("normalized_actions", 576, 651),
    ("current_prediction", 651, 843),
    ("last_update", 843, 1035),
    ("scalar_summaries", 1035, 1046),
    ("depth_one_hot", 1046, 1049),
)

BLOCK_STAT_NAMES = ("mean", "std", "l2_rms", "abs_max", "abs_gt3_rate")


def dataset_manifest_base(*, role: str, episodes: int, rows: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": role,
        "episode_count": int(episodes),
        "row_count": int(rows),
        "v3_test_episode_intersection": [],
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }

