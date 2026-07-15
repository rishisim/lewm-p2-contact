#!/usr/bin/env python3
"""Cycle-006 constants and fail-closed atomic artifact helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent
PROGRAM_ROOT = ROOT.parents[1]
REPO_ROOT = ROOT.parents[3]
SOURCE_DISCOVERY = REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_discovery"
PRIOR_INVALID = REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_audit_repair"
DISTRIBUTION = REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract"
V4_ROOT = REPO_ROOT / "runs/lewm_adaptive_compute_v4"
DISCOVERY_CODE = REPO_ROOT / "runs/lewm_adaptive_compute_discovery"
EVALUATION_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")
GENERATION_PYTHON = Path(
    "/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python"
)

LATENT_DIM = 192
ACTION_DIM = 25
HISTORY_LEN = 3
FEATURE_DIM = 1046
GATE_WIDTH = 1046
ROWS_PER_EPISODE = 38
BASE_FLOPS = 70_529_190
V1_FLOPS = 669_184
ADAPTER_FLOPS = 264_960
ROLE_COUNTS = {"fit": 240, "selection": 120, "smoke": 12, "prospective": 300}
ROLE_BASE_SEEDS = {
    "fit": 1_916_010_000,
    "selection": 1_916_020_000,
    "smoke": 1_916_030_000,
    "prospective": 1_916_040_000,
}

# Carried forward unchanged from the cycle-001 pre-data seal before any
# cycle-004 smoke. These are the repository's tighter selected-execution
# tolerances; the independent ceiling was fixed from pre-cycle-001 evidence.
NUMERICAL_RTOL = 2e-6
NUMERICAL_ATOL = 2e-7
NUMERICAL_MAX_ABS = 4.76837158203125e-7

EXPECTED_SOURCE_HASHES = {
    "whitening": "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796",
    "base_config": "4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999",
    "base_weights": "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89",
    "stagewise_refiner": "e63277943a356f3e28b4c4d1a1eb56acc8771fc07eccccdca67f878fed5ba782",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    encoded = (
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray], *, exclusive: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive and path.exists():
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sequential_calls(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("scores must have shape [rows,3]")
    limits = np.asarray(thresholds, dtype=np.float64)
    if limits.shape != (3,) or not np.isfinite(limits).all():
        raise ValueError("thresholds must be a finite length-three vector")
    calls = np.ones(len(values), dtype=np.int64)
    active = np.ones(len(values), dtype=bool)
    for stage in range(3):
        if not np.isfinite(values[active, stage]).all():
            raise ValueError("active gate scores must be finite")
        active &= values[:, stage] > limits[stage]
        calls += active.astype(np.int64)
    return calls


def verify_expected_sources() -> dict[str, str]:
    paths = {
        "whitening": SOURCE_DISCOVERY / "fit_only_normalization_whitening.npz",
        "base_config": REPO_ROOT / "runs/lewm_transfer/cube/cache/model/config.json",
        "base_weights": REPO_ROOT / "runs/lewm_transfer/cube/cache/model/weights.pt",
        "stagewise_refiner": DISCOVERY_CODE / "checkpoints/stagewise_seed_261102.pt",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"frozen source drift: {observed}")
    return observed
