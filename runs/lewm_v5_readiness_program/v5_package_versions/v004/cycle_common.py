#!/usr/bin/env python3
"""V5 package v004 constants and fail-closed atomic artifact helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent
PROGRAM_ROOT = ROOT.parents[1]
REPO_ROOT = ROOT.parents[3]
SOURCE_DISCOVERY = REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_discovery"
PRIOR_INVALID = REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_audit_repair"
SOURCE_CYCLE = PROGRAM_ROOT / "cycles/cycle_007_mps_threshold_dtype_repair"
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
PACKAGE_VERSION = "v004"
V5_SAMPLE_SIZE = 1600
ROLE_COUNTS = {"package_smoke": 12, "v5_confirmation": V5_SAMPLE_SIZE}
ROLE_BASE_SEEDS = {
    "package_smoke": 2_730_010_000,
    "v5_confirmation": 2_730_100_000,
}

PYTHON_VERSION = "3.10.20"
RUNTIME_PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "stable-worldmodel",
    "ogbench",
    "mujoco",
    "gymnasium",
    "transformers",
)
EXPECTED_RUNTIME_PACKAGES = {
    "evaluation": {
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "torch": "2.12.0",
        "stable-worldmodel": None,
        "ogbench": None,
        "mujoco": None,
        "gymnasium": None,
        "transformers": "5.10.2",
    },
    "generation": {
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "torch": "2.12.0",
        "stable-worldmodel": "0.1.0",
        "ogbench": "1.2.1",
        "mujoco": "3.10.0",
        "gymnasium": "1.3.0",
        "transformers": "4.50.0",
    },
}

SOURCE_CYCLE_HASHES = {
    "decision": "ed73eab576eb448f6eb4c1edbe4d1646f022424c43b232f98f1d0e545966d21c",
    "artifact_manifest": "6cee6f18cb98835450a013a3b61c83e4df332a801032341d036f3de061a22b5c",
    "independent_verification": "ee719a50d4bffe4df7fff9a74f21712564e2f2d7c09be21a627c36bd88323a72",
    "compiled_gate": "1eb9253073dbc6aad52c1589ac6b0643fec2aa5388cb0c7fd474a5f0114c8d57",
    "gate_fit": "d13cf89bc3d5f22d89612c30dd555bb50aa392bd465eba335e0917cbe860ff97",
    "gate_freeze": "11fedb55363539aea683d2960224b1e6b3d946f52027e6d18952a2ad4e480c3d",
    "gate_freeze_seal": "90b44528119b576d742d91572eab0f442dca748e5966b47a3a156cb02dd6bb36",
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


def installed_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_snapshot(role: str) -> dict[str, Any]:
    if role not in ("evaluation", "generation"):
        raise ValueError(f"unknown runtime role: {role}")
    expected_executable = EVALUATION_PYTHON if role == "evaluation" else GENERATION_PYTHON
    return {
        "role": role,
        "sys_executable": sys.executable,
        "expected_executable": str(expected_executable),
        "python_version": platform.python_version(),
        "python_version_info": list(sys.version_info[:3]),
        "packages": installed_package_versions(),
    }


def assert_runtime_contract(role: str) -> dict[str, Any]:
    snapshot = runtime_snapshot(role)
    expected_executable = EVALUATION_PYTHON if role == "evaluation" else GENERATION_PYTHON
    if Path(sys.executable) != expected_executable:
        raise RuntimeError(
            f"{role} interpreter contract violation: expected {expected_executable}, "
            f"observed {sys.executable}"
        )
    if snapshot["python_version"] != PYTHON_VERSION:
        raise RuntimeError(f"{role} Python version drift: {snapshot['python_version']}")
    if snapshot["packages"] != EXPECTED_RUNTIME_PACKAGES[role]:
        raise RuntimeError(f"{role} package version drift: {snapshot['packages']}")
    contract_path = ROOT / "runtime_contract.json"
    if contract_path.exists():
        contract = read_json(contract_path)
        recorded = contract["runtimes"][role]
        if (
            recorded["sys_executable"] != snapshot["sys_executable"]
            or recorded["python_version"] != snapshot["python_version"]
            or recorded["packages"] != snapshot["packages"]
        ):
            raise RuntimeError(f"{role} sealed runtime contract drift")
    return snapshot


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
        "whitening": ROOT / "freeze/whitening.npz",
        "base_config": REPO_ROOT / "runs/lewm_transfer/cube/cache/model/config.json",
        "base_weights": REPO_ROOT / "runs/lewm_transfer/cube/cache/model/weights.pt",
        "stagewise_refiner": DISCOVERY_CODE / "checkpoints/stagewise_seed_261102.pt",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"frozen source drift: {observed}")
    return observed
