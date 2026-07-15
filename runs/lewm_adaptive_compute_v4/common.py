"""Fail-closed shared utilities for the preregistered LeWM V4 study.

This module contains no target-dependent choices.  Constants below are copied
from sealed discovery artifacts and are rechecked against their byte hashes at
every phase boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DISCOVERY = REPO / "runs/lewm_adaptive_compute_discovery"
COMPRESSION = REPO / "runs/lewm_adaptive_compute_critic_compression"
V3 = REPO / "runs/lewm_adaptive_compute_v3"

SOLVER_CHECKPOINT = DISCOVERY / "checkpoints/stagewise_seed_261102.pt"
STUDENT_CHECKPOINT = COMPRESSION / "checkpoints/final_student.pt"
TOURNAMENT = COMPRESSION / "audit/frozen_tournament.json"
WHITENING = COMPRESSION / "checkpoints/discovery_whitening.npz"
BASE_CONFIG = REPO / "runs/lewm_transfer/cube/cache/model/config.json"
BASE_WEIGHTS = REPO / "runs/lewm_transfer/cube/cache/model/weights.pt"
SOURCE_H5 = Path(
    "/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/"
    "cube_single_expert.h5"
)
GENERATION_PYTHON = Path(
    "/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python"
)
EVALUATION_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")

EXPECTED = {
    "solver_checkpoint_sha256": "e63277943a356f3e28b4c4d1a1eb56acc8771fc07eccccdca67f878fed5ba782",
    "student_checkpoint_sha256": "fb59da4fd2a3895679a2cbba006c2a1f4c6619968b594f9ab1117531560f6302",
    "tournament_sha256": "5f474b17a37cc05c39b66495aad0baceb7c7ec4913eb3d6b2966eae230e52b66",
    "whitening_sha256": "6ac9f8a463921e8fc158ba2aca543f07dc46931a3f98b4d7ef9f10efd4add54f",
    "base_config_sha256": "4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999",
    "base_weights_sha256": "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89",
    "v3_split_manifest_sha256": "4b9344b6ff2acce4c58c3b296ba2017b0122f8f8158f45088239712ce94b7030",
    "v3_combined_cache_opaque_sha256": "21377c283136008ae07aa4d1e00407ab57eaba5ddda7d5b1c740c47e55afaba0",
}

FAMILY = "full_linear"
OPERATING_POINT = "b1.25"
COMPUTE_PRICE = 0.00014899746351320145
SUPPORTED_CALLS = np.asarray([1, 2, 3, 4], dtype=np.int64)
GATE_PARAMETERS = 1_050
GATE_FLOPS_PER_DECISION = 8_196
BASE_PREDICT_FLOPS = 70_529_190
V1_CALL_FLOPS = 669_184
STAGE_ADAPTER_FLOPS = 264_960
FEATURE_DIM = 1_046
LATENT_DIM = 192
HISTORY = 3
FRAMESKIP = 5
RAW_ACTION_DIM = 5
BLOCKED_ACTION_DIM = 25
RAW_EPISODE_ROWS = 201
MODEL_STEPS = 41
EXAMPLES_PER_EPISODE = 38

# Frozen released-training action normalizer from the isolated V2 extractor.
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


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes so array hashes are unambiguous."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def combined_array_sha256(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(array_sha256(np.asarray(value))))
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


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(data)
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, observed {observed}: {path}"
        )
    return observed


def verify_frozen_objects() -> dict[str, Any]:
    checks = {
        "solver_checkpoint": (SOLVER_CHECKPOINT, EXPECTED["solver_checkpoint_sha256"]),
        "student_checkpoint": (STUDENT_CHECKPOINT, EXPECTED["student_checkpoint_sha256"]),
        "tournament": (TOURNAMENT, EXPECTED["tournament_sha256"]),
        "whitening": (WHITENING, EXPECTED["whitening_sha256"]),
        "base_config": (BASE_CONFIG, EXPECTED["base_config_sha256"]),
        "base_weights": (BASE_WEIGHTS, EXPECTED["base_weights_sha256"]),
    }
    result = {}
    for name, (path, expected) in checks.items():
        result[name] = {
            "path": str(path.resolve()),
            "sha256": verify_hash(path, expected, name),
            "bytes": int(path.stat().st_size),
        }
    tournament = read_json(TOURNAMENT)
    contract = {
        "family": tournament["family"] == FAMILY,
        "operating_point": tournament["operating_point"] == OPERATING_POINT,
        "compute_price": float(tournament["compute_price"]) == COMPUTE_PRICE,
        "parameters": int(tournament["gate_cost"]["parameters"]) == GATE_PARAMETERS,
        "gate_flops": int(
            tournament["gate_cost"]["total_incremental_gate_flops_per_evaluated_decision"]
        )
        == GATE_FLOPS_PER_DECISION,
    }
    if not all(contract.values()):
        raise RuntimeError(f"frozen tournament contract mismatch: {contract}")
    result["contract"] = contract
    return result


def load_config() -> dict[str, Any]:
    return read_json(ROOT / "config.json")


def role_paths(role: str) -> tuple[Path, Path]:
    if role not in {"smoke", "confirmation"}:
        raise ValueError("role must be smoke or confirmation")
    directory = ROOT / "data" / f"{role}_raw"
    manifest = ROOT / "data" / f"{role}_data_manifest.json"
    return directory, manifest


def iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def exact_total_flop_budget(
    calls: np.ndarray, *, conservative_integer: bool = True
) -> dict[str, Any]:
    """Translate adaptive gate overhead into baseline adapter-call budget.

    The analytic target is exactly FLOP matched.  The integer target rounds up,
    deliberately granting the randomized baseline weakly more compute.
    """
    selected = np.asarray(calls, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0 or not np.isin(selected, SUPPORTED_CALLS).all():
        raise ValueError("calls must be a nonempty supported-call vector")
    n = len(selected)
    solver_calls = int(selected.sum(dtype=np.int64))
    gate_decisions = int(np.minimum(selected, 3).sum(dtype=np.int64))
    gate_flops = gate_decisions * GATE_FLOPS_PER_DECISION
    fractional_calls = solver_calls + gate_flops / STAGE_ADAPTER_FLOPS
    integer_calls = int(math.ceil(fractional_calls) if conservative_integer else round(fractional_calls))
    if not n <= integer_calls <= 4 * n:
        raise RuntimeError("exact-total-FLOP comparator is outside supported depth range")
    baseline_extra = (integer_calls - solver_calls) * STAGE_ADAPTER_FLOPS
    residual = baseline_extra - gate_flops
    if conservative_integer and not 0 <= residual < STAGE_ADAPTER_FLOPS:
        raise RuntimeError("conservative exact-FLOP residual accounting failed")
    common_total = n * (BASE_PREDICT_FLOPS + V1_CALL_FLOPS)
    adaptive_total = (
        common_total
        + (solver_calls - n) * STAGE_ADAPTER_FLOPS
        + gate_flops
    )
    integer_baseline_total = common_total + (integer_calls - n) * STAGE_ADAPTER_FLOPS
    return {
        "rows": n,
        "adaptive_solver_total_calls": solver_calls,
        "adaptive_gate_decisions": gate_decisions,
        "adaptive_gate_total_flops": gate_flops,
        "analytic_target_total_calls": float(fractional_calls),
        "analytic_target_mean_calls": float(fractional_calls / n),
        "integer_target_total_calls": integer_calls,
        "integer_target_mean_calls": float(integer_calls / n),
        "integer_rounding": "ceiling_conservative_for_adaptive_claim",
        "integer_baseline_minus_adaptive_flops": int(residual),
        "adaptive_total_flops": int(adaptive_total),
        "integer_baseline_total_flops": int(integer_baseline_total),
        "analytic_exact_flop_match": True,
        "integer_baseline_weakly_more_compute": bool(integer_baseline_total >= adaptive_total),
    }


def source_environment() -> dict[str, Any]:
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    return {
        "timestamp_policy": "timestamps are excluded from sealed scientific inputs",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git_head": git_head,
        "git_status_short": status,
        "cwd": str(REPO),
    }


def collect_file_hashes(paths: Iterable[Path], *, base: Path | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in sorted({Path(path).resolve() for path in paths}):
        if not raw.is_file():
            raise FileNotFoundError(raw)
        key = str(raw.relative_to(base.resolve())) if base is not None and raw.is_relative_to(base.resolve()) else str(raw)
        result[key] = sha256_file(raw)
    return result


def verify_seal(path: Path) -> dict[str, Any]:
    seal = read_json(path)
    for raw_path, expected in seal["files"].items():
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = REPO / candidate
        verify_hash(candidate, expected, f"sealed file {raw_path}")
    return seal


def file_mode_read_only(path: Path) -> None:
    Path(path).chmod(0o444)


__all__ = [name for name in globals() if name.isupper()] + [
    "array_sha256",
    "canonical_json_bytes",
    "collect_file_hashes",
    "combined_array_sha256",
    "exact_total_flop_budget",
    "file_mode_read_only",
    "iter_batches",
    "jsonable",
    "load_config",
    "read_json",
    "role_paths",
    "sha256_bytes",
    "sha256_file",
    "source_environment",
    "verify_frozen_objects",
    "verify_hash",
    "verify_seal",
    "write_json",
]
