#!/usr/bin/env python3
"""Fail-closed shared utilities for LeWM V5 zero-shot generalization v005."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ATTEMPT_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ATTEMPT_ROOT.parents[1]
REPO_ROOT = ATTEMPT_ROOT.parents[3]
V5_ROOT = (
    REPO_ROOT
    / "runs/lewm_v5_readiness_program/v5_package_versions/v004"
)
V5_VERSIONS_ROOT = V5_ROOT.parent
DISTRIBUTION_ROOT = REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract"

ATTEMPT_VERSION = "v005"
ORIGINAL_REQUIRED_GIT_HEAD = "d20ad79c4db7f564f5007513634bec4b9a785605"
EXPECTED_GIT_HEAD = "86e4bb150de0d1546a3cf57f3ef301294eb04368"
EVALUATION_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")
GENERATION_PYTHON = Path(
    "/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python"
)

STATE_MACHINE = (
    "BOOTSTRAP_AUDIT",
    "DESIGN_AND_POWER",
    "IMPLEMENT_PACKAGE",
    "PRESEAL_QUALIFY",
    "PRE_OUTCOME_SEAL",
    "EXCLUDED_REGIME_SMOKE",
    "FRESH_COHORT_GENERATION",
    "SPARSE_DENSE_EXECUTION",
    "SEALED_ANALYSIS",
    "LATENCY_AND_RESOURCE_REPORTING",
    "INDEPENDENT_VERIFICATION",
    "TERMINAL",
    "FOLLOW_ON_TASK_CREATION",
)

TERMINAL_LABELS = (
    "zero_shot_generalization_supported",
    "zero_shot_generalization_partial",
    "zero_shot_generalization_failed",
    "generalization_execution_invalid",
)

REGIMES: dict[str, dict[str, Any]] = {
    "markov_oracle": {
        "label": "policy_structure_markov_oracle",
        "policy_type": "markov_oracle",
        "action_noise": 0.1,
        "p_random_action": 0.0,
        "noise_smoothing": 0.5,
        "min_norm": 0.4,
        "single_changed_factor": "policy_type: plan_oracle -> markov_oracle",
    },
    "plan_action_noise_0p2": {
        "label": "action_noise_0p1_to_0p2",
        "policy_type": "plan_oracle",
        "action_noise": 0.2,
        "p_random_action": 0.0,
        "noise_smoothing": 0.5,
        "min_norm": 0.4,
        "single_changed_factor": "action_noise: 0.1 -> 0.2",
    },
    "plan_random_action_0p1": {
        "label": "random_action_contamination_0p1",
        "policy_type": "plan_oracle",
        "action_noise": 0.1,
        "p_random_action": 0.1,
        "noise_smoothing": 0.5,
        "min_norm": 0.4,
        "single_changed_factor": "p_random_action: 0.0 -> 0.1",
    },
}

SMOKE_EPISODES_PER_REGIME = 6
TARGET_EPISODES_PER_REGIME = 3_000
REPLACEMENTS_PER_REGIME = 200
ROWS_PER_EPISODE = 38
BOOTSTRAP_REPLICATES = 20_000
FAMILYWISE_ALPHA = 0.05
CO_PRIMARY_ENDPOINTS = ("raw_vs_analytic", "fixed_whitened_vs_analytic")
SIMULTANEOUS_FAMILY_SIZE = len(REGIMES) * len(CO_PRIMARY_ENDPOINTS)
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / SIMULTANEOUS_FAMILY_SIZE

LATENT_DIM = 192
ACTION_DIM = 25
HISTORY_LEN = 3
FEATURE_DIM = 1046
BASE_FLOPS = 70_529_190
V1_FLOPS = 669_184
ADAPTER_FLOPS = 264_960
GATE_FEATURE_FLOPS = 3_801
GATE_HEAD_FLOPS = 4_184
GATE_TOTAL_FLOPS = 7_985
GATE_NONFLOP_OPS = 5
NUMERICAL_RTOL = 2e-6
NUMERICAL_ATOL = 2e-7
NUMERICAL_MAX_ABS = 4.76837158203125e-7

V5_REQUIRED_HASHES = {
    "decision.json": "69bb8af8d80d7e18aeabc5430963c57b1bc91be603745d5a0d5f1a855d61fbe5",
    "audit/independent_verification.json": "d30e58466184ac98c5643ea6242e1fb1d95cc2f820c001be16aee3556903c3f0",
    "analysis_result.json": "3d6ce72bdb1c9e2330654d2822d592b8a387399cc44370931989accbf874eecc",
    "audit/pre_v5_seal.json": "c2853b726cf2e596ab3fda47ae1cae5aba1308f8f0c6409732acf4dc430677d6",
    "artifact_manifest.json": "cad97a07712eba7219a354d4105ec69d20eb85ba08fcefdaa1a6570cb21c2526",
    "data/v5_confirmation_raw_manifest.json": "10f3fcafeca92f4624693382c873802ef5eaf3013c6b066912e8533d6b4ef374",
    "data/v5_confirmation_execution.npz": "93aa013b550696e5a2ea962a680df29d411b94f6ff45caf9eb98a438727e636e",
    "data/v5_confirmation_execution_manifest.json": "3afb5afac96505d7869b7e8f5a768573afd13b5e96ccec173956ad612ca5ab48",
    "audit/v5_confirmation_input_seal.json": "b1696240bf80b293fee5274e7c1279b27a214f5e0ed537f01a476a9deb3cd119",
    "metrics/v5_confirmation_latency.json": "db4031e1472aa399dc5deeb7dfd288469d55b2c22ec0b4913e480244b68e98a9",
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


def combined_array_sha256(values: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
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


def atomic_json(
    path: Path, payload: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    encoded = (
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    exclusive: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
        delete=False,
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def relative_to_repo(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))


def append_ledger(event: str, **fields: Any) -> None:
    path = STUDY_ROOT / "RESEARCH_LEDGER.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_unix_ns": time.time_ns(),
        "attempt": ATTEMPT_VERSION,
        "event": event,
        **jsonable(fields),
    }
    encoded = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def initialize_state() -> dict[str, Any]:
    path = STUDY_ROOT / "STATE.json"
    if path.exists():
        state = read_json(path)
        if state.get("state_machine") != list(STATE_MACHINE):
            raise RuntimeError("existing generalization state-machine drift")
        if state.get("active_attempt") != ATTEMPT_VERSION:
            raise RuntimeError("existing active attempt is not v005")
        return state
    now = time.time_ns()
    state = {
        "schema_version": 1,
        "program": "LeWM V5 zero-shot adaptive-compute generalization",
        "objective": (
            "Determine the zero-shot external-validity envelope of the "
            "V5-confirmed frozen LeWM adaptive-compute policy."
        ),
        "expected_git_head": EXPECTED_GIT_HEAD,
        "active_attempt": ATTEMPT_VERSION,
        "active_attempt_path": relative_to_repo(ATTEMPT_ROOT),
        "attempt_history": [
            {
                "version": ATTEMPT_VERSION,
                "path": relative_to_repo(ATTEMPT_ROOT),
                "status": "active_zero_outcome_version_forward",
                "created_unix_ns": now,
            }
        ],
        "state_machine": list(STATE_MACHINE),
        "current_state": "BOOTSTRAP_AUDIT",
        "completed_states": [],
        "target_episode_count_per_regime": TARGET_EPISODES_PER_REGIME,
        "target_outcome_episodes_generated": 0,
        "target_outcome_episodes_executed": 0,
        "target_outcomes_opened_for_analysis": False,
        "smoke_episode_count_per_regime": SMOKE_EPISODES_PER_REGIME,
        "terminal_label": None,
        "process_valid": None,
        "last_verified_checkpoint": None,
        "next_action": "complete bootstrap audit",
        "created_unix_ns": now,
        "updated_unix_ns": now,
    }
    atomic_json(path, state, exclusive=True)
    append_ledger(
        "state_initialized",
        current_state=state["current_state"],
        state_machine=state["state_machine"],
        target_episode_count_per_regime=TARGET_EPISODES_PER_REGIME,
    )
    return state


def update_state_fields(**fields: Any) -> dict[str, Any]:
    path = STUDY_ROOT / "STATE.json"
    state = read_json(path)
    state.update(jsonable(fields))
    state["updated_unix_ns"] = time.time_ns()
    atomic_json(path, state)
    append_ledger("state_fields_updated", fields=fields)
    return state


def complete_state(
    completed: str,
    next_state: str,
    *,
    evidence_path: Path,
    checkpoint_name: str,
    next_action: str,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if completed not in STATE_MACHINE or next_state not in STATE_MACHINE:
        raise RuntimeError("unknown state transition")
    state_path = STUDY_ROOT / "STATE.json"
    state = read_json(state_path)
    if state["current_state"] != completed:
        if completed in state["completed_states"] and state["current_state"] == next_state:
            return state
        raise RuntimeError(
            f"state transition mismatch: expected {completed}, "
            f"observed {state['current_state']}"
        )
    if not evidence_path.exists():
        raise RuntimeError(f"missing state evidence: {evidence_path}")
    evidence_hash = sha256_file(evidence_path)
    completed_states = list(state["completed_states"])
    completed_states.append(completed)
    state.update(
        {
            "current_state": next_state,
            "completed_states": completed_states,
            "last_verified_checkpoint": {
                "name": checkpoint_name,
                "evidence_path": relative_to_repo(evidence_path),
                "evidence_sha256": evidence_hash,
                "created_unix_ns": time.time_ns(),
            },
            "next_action": next_action,
            "updated_unix_ns": time.time_ns(),
        }
    )
    if extra_fields:
        state.update(jsonable(extra_fields))
    atomic_json(state_path, state)
    append_ledger(
        "state_completed",
        completed_state=completed,
        next_state=next_state,
        evidence_path=relative_to_repo(evidence_path),
        evidence_sha256=evidence_hash,
        checkpoint_name=checkpoint_name,
        extra_fields=extra_fields or {},
    )
    return state


def mark_terminal(
    label: str,
    *,
    decision_path: Path,
    process_valid: bool,
) -> dict[str, Any]:
    if label not in TERMINAL_LABELS:
        raise RuntimeError(f"invalid terminal label: {label}")
    state_path = STUDY_ROOT / "STATE.json"
    state = read_json(state_path)
    if state["current_state"] != "TERMINAL":
        raise RuntimeError("terminal label can be recorded only in TERMINAL state")
    state["terminal_label"] = label
    state["process_valid"] = bool(process_valid)
    state["last_verified_checkpoint"] = {
        "name": f"{ATTEMPT_VERSION}_{label}",
        "evidence_path": relative_to_repo(decision_path),
        "evidence_sha256": sha256_file(decision_path),
        "created_unix_ns": time.time_ns(),
    }
    state["next_action"] = "create the preregistered follow-on project task"
    state["updated_unix_ns"] = time.time_ns()
    atomic_json(state_path, state)
    append_ledger(
        "scientific_terminal_decision_recorded",
        terminal_label=label,
        process_valid=process_valid,
        decision_path=relative_to_repo(decision_path),
        decision_sha256=sha256_file(decision_path),
    )
    return state


def finalize_follow_on(follow_on_path: Path) -> dict[str, Any]:
    state_path = STUDY_ROOT / "STATE.json"
    state = read_json(state_path)
    if state["current_state"] != "TERMINAL" or not state.get("terminal_label"):
        raise RuntimeError("follow-on creation requires a terminal decision")
    completed = list(state["completed_states"])
    if "TERMINAL" not in completed:
        completed.append("TERMINAL")
    if "FOLLOW_ON_TASK_CREATION" not in completed:
        completed.append("FOLLOW_ON_TASK_CREATION")
    state["completed_states"] = completed
    state["next_action"] = (
        "none; process-valid generalization result and follow-on task are complete"
    )
    state["last_verified_checkpoint"] = {
        "name": f"{ATTEMPT_VERSION}_follow_on_task_created",
        "evidence_path": relative_to_repo(follow_on_path),
        "evidence_sha256": sha256_file(follow_on_path),
        "created_unix_ns": time.time_ns(),
    }
    state["updated_unix_ns"] = time.time_ns()
    atomic_json(state_path, state)
    append_ledger(
        "follow_on_task_created",
        path=relative_to_repo(follow_on_path),
        sha256=sha256_file(follow_on_path),
    )
    return state


def installed_versions() -> dict[str, str | None]:
    names = (
        "numpy",
        "scipy",
        "torch",
        "stable-worldmodel",
        "ogbench",
        "mujoco",
        "gymnasium",
        "transformers",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def assert_runtime_contract(role: str) -> dict[str, Any]:
    if role not in ("evaluation", "generation"):
        raise RuntimeError(f"unknown runtime role: {role}")
    contract = read_json(V5_ROOT / "runtime_contract.json")
    recorded = contract["runtimes"][role]
    expected_executable = (
        EVALUATION_PYTHON if role == "evaluation" else GENERATION_PYTHON
    )
    snapshot = {
        "role": role,
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_version_info": list(sys.version_info[:3]),
        "packages": installed_versions(),
    }
    checks = {
        "executable": Path(sys.executable) == expected_executable,
        "python_version": snapshot["python_version"]
        == recorded["python_version"],
        "packages": snapshot["packages"] == recorded["packages"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{role} runtime contract failure: checks={checks}, "
            f"snapshot={snapshot}, recorded={recorded}"
        )
    return {**snapshot, "checks": checks}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_V5_RUNNER: Any | None = None


def load_v5_runner() -> Any:
    global _V5_RUNNER
    if _V5_RUNNER is None:
        if str(V5_ROOT) not in sys.path:
            sys.path.insert(0, str(V5_ROOT))
        _V5_RUNNER = load_module(
            "lewm_v5_frozen_runner_generalization",
            V5_ROOT / "runner.py",
        )
    return _V5_RUNNER


def verify_pre_outcome_seal() -> dict[str, Any]:
    path = ATTEMPT_ROOT / "audit/pre_outcome_seal.json"
    if not path.exists():
        raise RuntimeError("pre-outcome seal is missing")
    seal = read_json(path)
    bad = []
    for raw_path, expected in seal["sealed_files"].items():
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if not candidate.exists() or sha256_file(candidate) != expected:
            bad.append(raw_path)
    checks = {
        "status": seal.get("status")
        == "frozen_before_any_v005_excluded_smoke_or_target_episode",
        "attempt": seal.get("attempt") == ATTEMPT_VERSION,
        "target_outcomes_at_seal": seal.get("target_outcome_episodes_at_seal")
        == 0,
        "smoke_outcomes_at_seal": seal.get("smoke_episodes_at_seal") == 0,
        "sample_size": seal.get("target_episodes_per_regime")
        == TARGET_EPISODES_PER_REGIME,
        "regimes": seal.get("regimes") == list(REGIMES),
        "all_hashes_match": not bad,
    }
    if not all(checks.values()):
        raise RuntimeError(f"pre-outcome seal invalid: checks={checks}, bad={bad}")
    return {
        "passed": True,
        "checks": checks,
        "bad_hashes": bad,
        "seal_sha256": sha256_file(path),
        "sealed_file_count": len(seal["sealed_files"]),
    }


def role_count(phase: str) -> int:
    if phase == "smoke":
        return SMOKE_EPISODES_PER_REGIME
    if phase == "target":
        return TARGET_EPISODES_PER_REGIME
    raise RuntimeError(f"unknown phase: {phase}")


def raw_directory(phase: str, regime: str) -> Path:
    return ATTEMPT_ROOT / "data" / phase / regime / "raw"


def raw_manifest_path(phase: str, regime: str) -> Path:
    return ATTEMPT_ROOT / "data" / phase / regime / "raw_manifest.json"


def execution_directory(phase: str, regime: str) -> Path:
    return ATTEMPT_ROOT / "data" / phase / regime / "execution"


def execution_manifest_path(phase: str, regime: str) -> Path:
    return ATTEMPT_ROOT / "data" / phase / regime / "execution_manifest.json"


def latency_input_path(regime: str) -> Path:
    return ATTEMPT_ROOT / "data" / "target" / regime / "latency_inputs.npz"


def canonical_digest(values: Iterable[Any]) -> str:
    payload = "\n".join(str(value) for value in sorted(set(values), key=str)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sequential_calls(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or limits.shape != (3,):
        raise RuntimeError("invalid score or threshold shape")
    calls = np.ones(len(values), dtype=np.int64)
    active = np.ones(len(values), dtype=bool)
    for stage in range(3):
        if not np.isfinite(values[active, stage]).all():
            raise RuntimeError("nonfinite active gate score")
        active &= values[:, stage] > limits[stage]
        calls += active.astype(np.int64)
    return calls
