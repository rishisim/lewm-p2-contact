#!/usr/bin/env python3
"""Non-scientific paths, immutable contracts, hashing, and atomic I/O."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ATTEMPT_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ATTEMPT_ROOT.parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]

DGP_MATRIX_PATH = ATTEMPT_ROOT / "DGP_MATRIX.json"
CANDIDATE_GRID_PATH = ATTEMPT_ROOT / "candidate_grid.json"
POWER_RULE_PATH = ATTEMPT_ROOT / "power_rule.json"
OUTCOME_MAPPING_PATH = ATTEMPT_ROOT / "outcome_mapping.json"
COHORT_LEDGER_PATH = ATTEMPT_ROOT / "cohort_seed_ledger.json"
CONTROLLER_PATH = ATTEMPT_ROOT / "version_forward_transaction.py"

REGIME_ORDER = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ROLE_ORDER = ("fit", "selection", "smoke", "confirmation")
ROWS_PER_EPISODE = 38
LATENT_DIM = 192
ACTION_DIM = 25
HISTORY_LEN = 3
FEATURE_DIM = 1046

BASE_FLOPS_PER_ROW = 70_529_190
MANDATORY_DEPTH1_FLOPS_PER_ROW = 669_184
ADDITIONAL_REFINER_FLOPS_PER_CALL = 264_960
FEATURE_FLOPS_PER_REACHED_DECISION = 3_801
AFFINE_HEAD_FLOPS = 2_092


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_verified_controller() -> dict[str, Any]:
    """Return controller state only after its ledger/checkpoint audit passes.

    Scientific entry points use this instead of trusting mutable ``STATE.json``
    fields directly.  The controller is intentionally invoked as a separate
    stdlib-only process so generation and evaluation environments share the
    same fail-closed chronology check without importing each other's runtimes.
    """

    completed = subprocess.run(
        [sys.executable, str(CONTROLLER_PATH), "status"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"controller status verification failed: {detail}")
    try:
        state = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("controller status emitted non-JSON output") from exc
    if not isinstance(state, dict):
        raise RuntimeError("controller status is not a JSON object")
    return state


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_repo(path: Path) -> str:
    resolved = Path(path).resolve(strict=False)
    return str(resolved.relative_to(REPO_ROOT.resolve()))


def resolve_repo_relative(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"unsafe repository-relative path: {value}")
    resolved = (REPO_ROOT / candidate).resolve(strict=False)
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise RuntimeError(f"path escapes repository: {value}")
    return resolved


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temporary(
    temporary: Path, destination: Path, *, exclusive: bool
) -> None:
    """Publish a fully written same-directory inode with true no-replace."""

    if exclusive:
        # link(2) creates the destination name atomically and fails with
        # EEXIST for every extant directory entry, including dangling links.
        os.link(temporary, destination, follow_symlinks=False)
    else:
        # Mutable controller/helper artifacts intentionally retain replace
        # semantics; durability still requires the directory entry fsync.
        os.replace(temporary, destination)
    _fsync_parent_directory(destination)


def atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(temporary, path, exclusive=exclusive)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray], *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        _publish_temporary(temporary, path, exclusive=exclusive)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(path)


def validate_normative_contracts() -> dict[str, Any]:
    dgp = read_json(DGP_MATRIX_PATH)
    grid = read_json(CANDIDATE_GRID_PATH)
    power = read_json(POWER_RULE_PATH)
    mapping = read_json(OUTCOME_MAPPING_PATH)
    if tuple(dgp["regime_order"]) != REGIME_ORDER:
        raise RuntimeError("DGP order drift")
    if int(grid["candidate_count"]) != 24:
        raise RuntimeError("candidate count drift")
    if int(mapping["family_size"]) != 8 or int(power["family_size"]) != 8:
        raise RuntimeError("simultaneous family drift")
    return {
        "DGP_MATRIX": sha256_file(DGP_MATRIX_PATH),
        "candidate_grid": sha256_file(CANDIDATE_GRID_PATH),
        "power_rule": sha256_file(POWER_RULE_PATH),
        "outcome_mapping": sha256_file(OUTCOME_MAPPING_PATH),
    }
