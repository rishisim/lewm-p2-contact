#!/usr/bin/env python3
"""Fail-closed, resumable orchestration for the sealed v001 development path.

This driver never generates an episode.  It advances only from already
materialized role artifacts after byte-level manifest verification, and it
opens fit or selection aggregates only inside their prospectively assigned
state.  Every state transition is delegated to the durable study controller.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import compile_gate
import fit_inheritance
import fit_select
import power_analysis
import preseal
from counted_features import ACTION_DIM, FEATURE_DIM, HISTORY_LEN, LATENT_DIM
from input_loader import array_sha256
from inherited_authorization import (
    ACTIVE_ATTEMPT,
    INHERITANCE_SEAL_PATH,
    SCIENCE_ATTEMPT,
    SOURCE_PRE_SELECTION_SEAL_PATH,
    SOURCE_ATTEMPT as VERSION_FORWARD_SOURCE_ATTEMPT,
    _EARLY_ROOT_GUARD_TOKEN,
    _ROLE_COUNT_RECOVERY_GUARD_TOKEN,
    InheritedAuthorizationError,
    verify_inherited_pre_data_authorization,
)
from runtime_contract import (
    ATTEMPT_ROOT,
    EVALUATION_PYTHON,
    REPO_ROOT,
    sha256_file,
    verify_here,
)


ATTEMPT = ACTIVE_ATTEMPT
STUDY_ROOT = ATTEMPT_ROOT.parents[1]
CONTROLLER_ADAPTER_PATH = ATTEMPT_ROOT / "version_forward_transaction.py"
ROOT_PROGRAM_PATH = STUDY_ROOT / "program.py"
STATE_PATH = STUDY_ROOT / "STATE.json"
LEDGER_PATH = STUDY_ROOT / "RESEARCH_LEDGER.jsonl"
REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ROLE_EPISODES_PER_DGP = {"fit": 300, "selection": 500}
ROLE_STATES = {"fit": "FIT_COHORTS", "selection": "SELECTION_COHORTS"}
ROLE_COUNTER_FIELDS = {
    "fit": "fit_outcome_episodes",
    "selection": "selection_outcome_episodes",
}
ROLE_RAW_SEALS = {
    "fit": (INHERITANCE_SEAL_PATH, "PRE_OUTCOME_SEAL"),
    "selection": (INHERITANCE_SEAL_PATH, "PRE_OUTCOME_SEAL"),
}
ROLE_EXECUTION_SEALS = {
    "fit": (INHERITANCE_SEAL_PATH, "PRE_OUTCOME_SEAL"),
    "selection": (
        SOURCE_PRE_SELECTION_SEAL_PATH,
        "PRE_SELECTION_SEAL",
    ),
}
ROWS_PER_EPISODE = 38
AGGREGATE_KEYS = (
    "episode_slot",
    "model_step",
    "target",
    "exits",
    "production_features",
)
DEVELOPMENT_PART_KEYS = frozenset(
    {
        *AGGREGATE_KEYS,
        "history",
        "action_history",
        "stage_current",
        "stage_update",
    }
)
DEVELOPMENT_PART_CONTRACT = {
    "episode_slot": ([ROWS_PER_EPISODE], "int32"),
    "model_step": ([ROWS_PER_EPISODE], "int16"),
    "target": ([ROWS_PER_EPISODE, LATENT_DIM], "float32"),
    "exits": ([ROWS_PER_EPISODE, 4, LATENT_DIM], "float32"),
    "production_features": ([ROWS_PER_EPISODE, 3, FEATURE_DIM], "float32"),
    "history": ([ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM], "float32"),
    "action_history": ([ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM], "float32"),
    "stage_current": ([ROWS_PER_EPISODE, 3, LATENT_DIM], "float32"),
    "stage_update": ([ROWS_PER_EPISODE, 3, LATENT_DIM], "float32"),
}
DEVELOPMENT_SIDECAR_KEYS = frozenset(
    {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "role",
        "regime",
        "slot",
        "episode_id",
        "raw_path",
        "raw_sha256",
        "raw_sidecar_path",
        "raw_sidecar_sha256",
        "input_loader_audit",
        "evaluation",
        "target_use",
        "contact_or_privileged_materialized",
        "primitive_feature_reconstruction_exact",
        "no_gradients",
        "complete",
        "path",
        "sha256",
        "bytes",
        "arrays",
    }
)
INPUT_LOADER_AUDIT_KEYS = frozenset(
    {
        "loaded_keys",
        "pixels_shape",
        "pixels_dtype",
        "action_shape",
        "action_dtype",
        "pixels_finite",
        "modeled_actions_finite",
        "terminal_action_nan_sentinel",
        "array_sha256",
        "archive_keys",
        "archive_sha256",
        "arrays_materialized",
        "non_input_arrays_materialized",
    }
)
FORBIDDEN_KEY_TOKENS = (
    "contact",
    "privileged",
    "qpos",
    "qvel",
    "motion",
    "phase",
    "reward",
    "success",
)
ALLOWED_NEGATIVE_ASSERTIONS = frozenset({"contact_or_privileged_materialized"})
FROZEN_FACADE_SOURCE_HASHES = {
    str(
        (
            REPO_ROOT
            / "runs/lewm_adaptive_compute_v2/model_io.py"
        ).resolve(strict=True)
    ): "5505e529ba17aeb0c641e7d4e8dc511d0ed61decdc13a596915ba8964b954874",
    str(
        (
            REPO_ROOT
            / "runs/lewm_adaptive_compute_v4/runtime.py"
        ).resolve(strict=True)
    ): "801b160ad9c457f7eabb4b92395c90cb7eb2f9d1809a627bd0202df8a210bfdf",
    str(
        (
            REPO_ROOT
            / "runs/lewm_v5_readiness_program/v5_package_versions/v004/runner.py"
        ).resolve(strict=True)
    ): "ebcbe74b409867c538460b53b5ddb69a05c9163832ec4ab4eb2a3ea484ab1174",
}

FITTED_PATH = STUDY_ROOT / "attempts/v006/fit/fitted_candidates.npz"
FIT_LOCK_PATH = STUDY_ROOT / "attempts/v006/fit/fit_lock.json"
SELECTION_LEDGER_PATH = ATTEMPT_ROOT / "selection/selection_ledger.json"
GATE_FIT_PATH = ATTEMPT_ROOT / "freeze/gate_fit.npz"
COMPILED_GATE_PATH = ATTEMPT_ROOT / "freeze/compiled_gate.npz"
COMPILER_MANIFEST_PATH = ATTEMPT_ROOT / "freeze/compiled_gate_manifest.json"
GATE_FREEZE_PATH = ATTEMPT_ROOT / "freeze/gate_freeze.json"
FIT_POWER_SUMMARY_PATH = ATTEMPT_ROOT / "metrics/fit_power_summary.json"
SELECTION_POWER_SUMMARY_PATH = ATTEMPT_ROOT / "metrics/selection_power_summary.json"
POWER_PATH = ATTEMPT_ROOT / "power_analysis.json"
POWER_RULE_PATH = ATTEMPT_ROOT / "power_rule.json"
DGP_MATRIX_PATH = ATTEMPT_ROOT / "DGP_MATRIX.json"
COHORT_LEDGER_PATH = ATTEMPT_ROOT / "cohort_seed_ledger.json"
REPLACEMENT_REGISTRY_PATH = ATTEMPT_ROOT / "data/replacement_registry.json"
ANALYSIS_RESULT_PATH = ATTEMPT_ROOT / "analysis_result.json"
ANALYSIS_INVALID_PATH = ATTEMPT_ROOT / "audit/analysis_execution_invalid.json"
SCIENTIFIC_REPLAY_LAUNCHER_PATH = (
    ATTEMPT_ROOT / "scientific_replay_launcher.py"
)
SCIENTIFIC_REPLAY_AUDIT_ROOT = ATTEMPT_ROOT / "audit/scientific_replay"
SCIENTIFIC_REPLAY_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "attempt",
        "role",
        "regime",
        "raw_manifest",
        "execution_manifest",
        "compiled_gate",
        "episode_count",
        "row_count",
        "episodes",
        "aggregate",
        "runtime_audit",
        "source_hashes",
        "module_before",
        "module_after",
        "input_loader_agreement_exact",
        "every_persisted_tensor_exact",
        "aggregate_exact",
        "target_loss_computed",
        "loss_or_effect_used_for_acceptance",
        "contact_or_privileged_materialized",
        "no_gradients",
        "artifact_tree_unchanged",
        "read_only",
        "passed",
    }
)
SCIENTIFIC_REPLAY_ROLE_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "attempt",
        "role",
        "state",
        "authorization_state_sha256",
        "manifest_verification_sha256",
        "regime_order",
        "episodes_per_regime",
        "episode_count",
        "row_count",
        "regimes",
        "source_hashes",
        "input_loader_agreement_exact",
        "every_persisted_tensor_exact",
        "all_development_aggregates_exact",
        "all_runtime_mps_exact",
        "target_loss_computed",
        "loss_or_effect_used_for_acceptance",
        "contact_or_privileged_materialized",
        "no_gradients",
        "artifact_trees_unchanged",
        "read_only",
        "passed",
    }
)
SCIENTIFIC_REPLAY_EPISODE_KEYS = frozenset(
    {
        "slot",
        "episode_id",
        "raw",
        "raw_sidecar",
        "execution_part",
        "execution_sidecar",
        "input_loader_audit_sha256",
        "array_sha256",
        "every_persisted_tensor_exact",
    }
)
SCIENTIFIC_REPLAY_QUALIFICATION_KEYS = frozenset(
    {
        "artifact_type",
        "path",
        "sha256",
        "role",
        "state",
        "authorization_state_sha256",
        "manifest_verification_sha256",
        "episode_count",
        "row_count",
        "passed",
    }
)
V5_FIXED_WHITENING_PATH = (
    REPO_ROOT
    / "runs/lewm_v5_readiness_program/v5_package_versions/v004/freeze/whitening.npz"
)

AUDIT_PATHS = {
    "FIT_COHORTS": ATTEMPT_ROOT / "audit/fit_cohorts.json",
    "FIT_LOCK": ATTEMPT_ROOT / "audit/fit_lock_checkpoint.json",
    "SELECTION_COHORTS": ATTEMPT_ROOT / "audit/selection_cohorts.json",
    "CANDIDATE_SELECTION": ATTEMPT_ROOT / "audit/candidate_selection.json",
    "GATE_FREEZE": ATTEMPT_ROOT / "audit/gate_freeze_checkpoint.json",
    "CONFIRMATION_POWER_AND_COHORT_FREEZE": (
        ATTEMPT_ROOT / "audit/confirmation_power_and_cohort_freeze.json"
    ),
}
DEFAULT_EVIDENCE_PATHS = {
    "IMPLEMENTATION_COMPLETE": preseal.IMPLEMENTATION_REPORT_PATH,
    "PRESEAL_QUALIFICATION": preseal.QUALIFICATION_REPORT_PATH,
    "PRE_OUTCOME_SEAL": INHERITANCE_SEAL_PATH,
    **AUDIT_PATHS,
    "PRE_SELECTION_SEAL": SOURCE_PRE_SELECTION_SEAL_PATH,
    "PRE_CONFIRMATION_PACKAGE_SEAL": preseal.PRE_CONFIRMATION_SEAL_PATH,
}


class WorkflowError(RuntimeError):
    """The workflow encountered a fail-closed contract violation."""


_INHERITED_FIT_RECOVERY_GUARD = object()


class Controller(Protocol):
    def status(self) -> dict[str, Any]: ...

    def advance(
        self,
        completed_state: str,
        evidence: Path,
        checkpoint_name: str,
        next_action: str,
    ) -> dict[str, Any]: ...

    def update_counts(self, **counts: int | bool) -> dict[str, Any]: ...

    def stage_early_scientific_failure(self, mode: str) -> dict[str, Any]: ...

    def stage_postconfirmation_integrity_failure(self) -> dict[str, Any]: ...


def _require_inherited_presealed_verifier_contract(
    state: Mapping[str, Any], contract_path: Path
) -> None:
    """Adapt the root early-stop guard to authenticated v008 inheritance."""

    if (
        state.get("active_attempt") != ATTEMPT
        or state.get("active_attempt_path") != relative_to_repo(ATTEMPT_ROOT)
        or state.get("current_state")
        not in {"CANDIDATE_SELECTION", "CONFIRMATION_POWER_AND_COHORT_FREEZE"}
    ):
        raise WorkflowError("inherited early-stop contract has invalid controller state")
    contract = _regular_file(Path(contract_path))
    expected_contract = {
        "CANDIDATE_SELECTION": ATTEMPT_ROOT
        / "verifier_contract_no_candidate.json",
        "CONFIRMATION_POWER_AND_COHORT_FREEZE": ATTEMPT_ROOT
        / "verifier_contract_power_infeasible.json",
    }[str(state["current_state"])]
    if contract.resolve() != expected_contract.resolve():
        raise WorkflowError("early-stop contract is not the exact active v008 contract")
    try:
        inherited = verify_inherited_pre_data_authorization(
            expected_current_state=str(state["current_state"]),
            state=state,
            authorized_early_guard_token=_EARLY_ROOT_GUARD_TOKEN,
            authorized_early_verifier_contract_path=contract,
        )
    except InheritedAuthorizationError as error:
        raise WorkflowError("early-stop inherited pre-data lineage failed") from error

    inheritance_path = _regular_file(INHERITANCE_SEAL_PATH)
    inheritance_sha256 = sha256_file(inheritance_path)
    transaction_path = _regular_file(
        ATTEMPT_ROOT / "audit/version_forward_transaction_receipt.json"
    )
    transaction = inherited.get("version_forward_transaction_receipt")
    if (
        inherited.get("passed") is not True
        or inherited.get("active_attempt") != ATTEMPT
        or inherited.get("science_attempt") != SCIENCE_ATTEMPT
        or inherited.get("state") != state.get("current_state")
        or inherited.get("seal_path") != relative_to_repo(inheritance_path)
        or inherited.get("seal_sha256") != inheritance_sha256
        or inherited.get("seal_checkpoint_state") != "PRE_OUTCOME_SEAL"
        or not isinstance(transaction, Mapping)
        or transaction.get("passed") is not True
        or transaction.get("path") != relative_to_repo(transaction_path)
        or transaction.get("sha256") != sha256_file(transaction_path)
    ):
        raise WorkflowError("early-stop inherited authorization result drift")

    seal = read_json(inheritance_path)
    source_link = seal.get("source_pre_data_seal")
    sealed_files = seal.get("sealed_files")
    seal_checks = {
        "schema": seal.get("schema_version") == 1,
        "passed": seal.get("passed") is True,
        "attempt": seal.get("attempt") == ATTEMPT,
        "source": seal.get("source_attempt") == SCIENCE_ATTEMPT,
        "science": seal.get("science_attempt") == SCIENCE_ATTEMPT,
        "target": seal.get("target_attempt") == ATTEMPT,
        "checkpoint": seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL",
        "kind": seal.get("authorization_kind")
        == "zero_outcome_version_forward_inherited_pre_data",
        "source_link": isinstance(source_link, Mapping)
        and set(source_link) == {"path", "sha256"},
        "sealed_files": isinstance(sealed_files, Mapping),
        "contract": isinstance(sealed_files, Mapping)
        and sealed_files.get(relative_to_repo(contract)) == sha256_file(contract),
    }
    if not all(seal_checks.values()):
        raise WorkflowError(f"early-stop inheritance seal drift: {seal_checks}")

    completed = state.get("completed_states")
    checkpoints = state.get("verified_checkpoints")
    if not isinstance(completed, list) or not isinstance(checkpoints, list):
        raise WorkflowError("early-stop inherited checkpoint chronology is absent")
    try:
        index = completed.index("PRE_OUTCOME_SEAL")
        checkpoint = checkpoints[index]
    except (ValueError, IndexError) as error:
        raise WorkflowError("inherited PRE_OUTCOME checkpoint is absent") from error
    if not isinstance(checkpoint, Mapping) or not isinstance(source_link, Mapping):
        raise WorkflowError("inherited PRE_OUTCOME checkpoint schema drift")
    source_path = _regular_file(resolve_repo_relative(source_link["path"]))
    checkpoint_checks = {
        "authorized_source_path": inherited.get("source_pre_data_seal_path")
        == source_link["path"],
        "authorized_source_hash": inherited.get("source_pre_data_seal_sha256")
        == source_link["sha256"],
        "source_attempt": checkpoint.get("source_attempt") == SCIENCE_ATTEMPT,
        "evidence_path": checkpoint.get("evidence_path") == source_link["path"],
        "evidence_hash": checkpoint.get("evidence_sha256")
        == source_link["sha256"]
        == sha256_file(source_path),
        "lineage": checkpoint.get("verification_lineage") == "direct_checkpoint",
    }
    if not all(checkpoint_checks.values()):
        raise WorkflowError(
            f"inherited PRE_OUTCOME checkpoint drift: {checkpoint_checks}"
        )


def _load_controller() -> Controller:
    spec = importlib.util.spec_from_file_location(
        "lewm_domain_robust_gate_v008_controller_adapter",
        CONTROLLER_ADAPTER_PATH,
    )
    if spec is None or spec.loader is None:
        raise WorkflowError("cannot load the durable study controller")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "_require_presealed_verifier_contract"):
        raise WorkflowError("v008 controller adapter lacks its early-stop contract guard")
    module._require_presealed_verifier_contract = (  # type: ignore[attr-defined]
        _require_inherited_presealed_verifier_contract
    )
    return module


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkflowError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise WorkflowError(f"expected JSON object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_to_repo(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as error:
        raise WorkflowError(f"path escapes repository: {path}") from error


def resolve_repo_relative(raw: Any) -> Path:
    if not isinstance(raw, str):
        raise WorkflowError("artifact path is not a string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
        raise WorkflowError(f"noncanonical repository-relative path: {raw}")
    path = (REPO_ROOT / relative).resolve(strict=False)
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise WorkflowError(f"artifact path escapes repository: {raw}")
    return path


def _regular_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise WorkflowError(f"required regular non-symlink file is absent: {path}")
    return path


def _atomic_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable workflow artifact already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"immutable workflow artifact already exists: {path}")
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_or_verify_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    intended = dict(value)
    if path.exists():
        existing = read_json(path)
        existing_invariant = {
            key: item for key, item in existing.items() if key != "created_unix_ns"
        }
        intended_invariant = {
            key: item for key, item in intended.items() if key != "created_unix_ns"
        }
        if existing_invariant != intended_invariant:
            raise WorkflowError(f"immutable workflow artifact drift: {path}")
        return existing
    _atomic_json_exclusive(path, intended)
    return intended


def _assert_no_forbidden_keys(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        if location.endswith(".frozen_facade.sources"):
            observed = {str(key): item for key, item in value.items()}
            if observed != FROZEN_FACADE_SOURCE_HASHES:
                raise WorkflowError(
                    f"frozen facade provenance map drift at {location}"
                )
            for raw_path, expected_sha256 in observed.items():
                path = Path(raw_path)
                if (
                    not path.is_absolute()
                    or path.resolve(strict=True) != path
                    or not path.is_file()
                    or path.is_symlink()
                    or sha256_file(path) != expected_sha256
                ):
                    raise WorkflowError(
                        f"frozen facade provenance record drift at {location}"
                    )
            return
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_KEY_TOKENS):
                if not (
                    lowered in ALLOWED_NEGATIVE_ASSERTIONS and item is False
                ):
                    raise WorkflowError(
                        f"forbidden preterminal key at {location}: {key}"
                    )
            _assert_no_forbidden_keys(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_forbidden_keys(item, location=f"{location}[{index}]")


def _verify_link(
    raw_path: Any,
    raw_sha256: Any,
    *,
    required_root: Path | None = None,
) -> dict[str, Any]:
    path = _regular_file(resolve_repo_relative(raw_path))
    if required_root is not None and not path.resolve().is_relative_to(
        required_root.resolve()
    ):
        raise WorkflowError(f"artifact is outside its required role root: {path}")
    observed = sha256_file(path)
    if not isinstance(raw_sha256, str) or observed != raw_sha256:
        raise WorkflowError(f"artifact hash drift: {path}")
    return {
        "path": str(raw_path),
        "sha256": observed,
        "bytes": path.stat().st_size,
        "device": int(path.stat().st_dev),
        "inode": int(path.stat().st_ino),
    }


def _npz_member_names(path: Path) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = tuple(item.filename for item in archive.infolist())
    except (OSError, zipfile.BadZipFile) as error:
        raise WorkflowError(f"invalid NPZ/ZIP container: {path}") from error
    if len(names) != len(set(names)) or any(
        name.startswith("/") or ".." in Path(name).parts for name in names
    ):
        raise WorkflowError(f"unsafe or duplicate NPZ members: {path}")
    return names


def _validate_input_loader_audit(
    value: Any, *, raw_record: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != INPUT_LOADER_AUDIT_KEYS:
        raise WorkflowError("development input-loader audit schema drift")
    hashes = value.get("array_sha256")
    raw_arrays = raw_record.get("arrays")
    pixels_shape = value.get("pixels_shape")
    action_shape = value.get("action_shape")
    checks = {
        "loaded_keys": value.get("loaded_keys") == ["action", "pixels"],
        "archive_keys": value.get("archive_keys") == ["action", "pixels"],
        "materialized": value.get("arrays_materialized")
        == ["action", "pixels"],
        "pixels_shape": isinstance(pixels_shape, list)
        and pixels_shape == [201, 224, 224, 3]
        and all(type(item) is int for item in pixels_shape),
        "pixels_dtype": value.get("pixels_dtype") == "uint8",
        "action_shape": isinstance(action_shape, list)
        and action_shape == [201, 5]
        and all(type(item) is int for item in action_shape),
        "action_dtype": value.get("action_dtype") == "float32",
        "pixels_finite": value.get("pixels_finite") is True,
        "actions_finite": value.get("modeled_actions_finite") is True,
        "terminal_nan": value.get("terminal_action_nan_sentinel") is True,
        "non_input": value.get("non_input_arrays_materialized") is False,
        "archive_hash": value.get("archive_sha256")
        == raw_record.get("raw_sha256"),
        "hash_schema": isinstance(hashes, Mapping)
        and set(hashes) == {"action", "pixels"},
        "raw_array_schema": isinstance(raw_arrays, Mapping)
        and set(raw_arrays) == {"action", "pixels"},
    }
    if not all(checks.values()):
        raise WorkflowError(f"development input-loader audit drift: {checks}")
    assert isinstance(hashes, Mapping) and isinstance(raw_arrays, Mapping)
    expected_raw_arrays = {
        "action": {
            "shape": [201, 5],
            "dtype": "float32",
            "sha256": hashes["action"],
        },
        "pixels": {
            "shape": [201, 224, 224, 3],
            "dtype": "uint8",
            "sha256": hashes["pixels"],
        },
    }
    if dict(raw_arrays) != expected_raw_arrays:
        raise WorkflowError("raw array/input-loader audit cross-link drift")
    return dict(value)


def _numpy_causal_features(
    history: np.ndarray,
    actions: np.ndarray,
    current: np.ndarray,
    update: np.ndarray,
) -> np.ndarray:
    epsilon = np.finfo(np.float32).eps
    last_history = history[:, -1]
    gap = current - last_history

    def norm(value: np.ndarray) -> np.ndarray:
        return np.sqrt(
            np.sum(np.square(value, dtype=np.float32), axis=-1, keepdims=True)
        ).astype(np.float32)

    current_norm = norm(current)
    update_norm = norm(update)
    last_history_norm = norm(last_history)
    gap_norm = norm(gap)
    relative = update_norm / np.maximum(current_norm, epsilon)
    update_current = np.sum(update * current, axis=-1, keepdims=True) / np.maximum(
        update_norm * current_norm, epsilon
    )
    update_gap = np.sum(update * gap, axis=-1, keepdims=True) / np.maximum(
        update_norm * gap_norm, epsilon
    )
    history_change = norm(history[:, 1:] - history[:, :-1]).squeeze(-1)
    action_change = norm(actions[:, 1:] - actions[:, :-1]).squeeze(-1)
    return np.concatenate(
        (
            history.reshape(len(history), -1),
            actions.reshape(len(actions), -1),
            current,
            update,
            current_norm,
            update_norm,
            relative,
            last_history_norm,
            gap_norm,
            update_current,
            update_gap,
            history_change,
            action_change,
        ),
        axis=1,
    ).astype(np.float32)


def _verify_development_part_arrays(
    path: Path,
    metadata: Mapping[str, Any],
    *,
    expected_slot: int,
) -> dict[str, Any]:
    observed: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != DEVELOPMENT_PART_KEYS:
                raise WorkflowError("development NPZ member-key drift")
            for name, (expected_shape, expected_dtype) in (
                DEVELOPMENT_PART_CONTRACT.items()
            ):
                value = stored[name].copy()
                if list(value.shape) != expected_shape or str(value.dtype) != expected_dtype:
                    raise WorkflowError(
                        f"development NPZ shape/dtype drift: {name}"
                    )
                if array_sha256(value) != metadata[name]["sha256"]:
                    raise WorkflowError(f"development NPZ array hash drift: {name}")
                observed[name] = value
    except WorkflowError:
        raise
    except Exception as error:
        raise WorkflowError(f"cannot independently open development part: {path}") from error

    if not np.array_equal(
        observed["episode_slot"],
        np.full(ROWS_PER_EPISODE, expected_slot, dtype=np.int32),
    ) or not np.array_equal(
        observed["model_step"], np.arange(3, 41, dtype=np.int16)
    ):
        raise WorkflowError("development row identity/order drift")
    finite_names = DEVELOPMENT_PART_KEYS - {"episode_slot", "model_step"}
    if any(not np.isfinite(observed[name]).all() for name in finite_names):
        raise WorkflowError("development part contains nonfinite model arrays")
    if not np.array_equal(
        observed["stage_current"], observed["exits"][:, :3]
    ):
        raise WorkflowError("development exits/stage-current drift")
    for stage in (1, 2):
        if not np.array_equal(
            observed["stage_current"][:, stage],
            observed["stage_current"][:, stage - 1]
            + observed["stage_update"][:, stage],
        ):
            raise WorkflowError("development stage update/output drift")

    reconstructed = np.stack(
        [
            _numpy_causal_features(
                observed["history"],
                observed["action_history"],
                observed["stage_current"][:, stage],
                observed["stage_update"][:, stage],
            )
            for stage in range(3)
        ],
        axis=1,
    )
    persisted_features = observed["production_features"]
    copied_width = HISTORY_LEN * LATENT_DIM + HISTORY_LEN * ACTION_DIM + 2 * LATENT_DIM
    if not np.array_equal(
        persisted_features[:, :, :copied_width],
        reconstructed[:, :, :copied_width],
    ) or not np.allclose(
        persisted_features[:, :, copied_width:],
        reconstructed[:, :, copied_width:],
        rtol=2e-5,
        atol=2e-6,
    ):
        raise WorkflowError("development causal feature reconstruction drift")
    difference = observed["exits"].astype(np.float64) - observed["target"].astype(
        np.float64
    )[:, None, :]
    endpoint = np.square(difference).mean(axis=2)
    gain = endpoint[:, :3] - endpoint[:, 1:]
    if not np.isfinite(endpoint).all() or not np.isfinite(gain).all():
        raise WorkflowError("development endpoint/gain reconstruction drift")
    return {
        "exact_member_set": True,
        "exact_shapes_dtypes_hashes": True,
        "row_identity_exact": True,
        "primitive_feature_reconstruction": "exact_copied_features_allclose_summaries",
        "endpoint_shape": list(endpoint.shape),
        "gain_shape": list(gain.shape),
        "raw_endpoint_sha256": array_sha256(endpoint),
        "raw_gain_sha256": array_sha256(gain),
    }


RAW_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "complete",
        "role",
        "regime",
        "regime_specification",
        "episode_count",
        "episodes",
        "replacements_used",
        "raw_archive_members",
        "raw_pixels_contract",
        "raw_action_contract",
        "role_isolation",
        "smoke_permanently_excluded",
        "retention_contract",
        "dgp_matrix_path",
        "dgp_matrix_sha256",
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "authorization_seal",
        "replacement_registry_path",
        "replacement_registry_scope",
        "orphan_policy",
    }
)
EXECUTION_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "role",
        "regime",
        "complete",
        "episode_count",
        "row_count",
        "episodes",
        "aggregate_role_path",
        "aggregate_role_sha256",
        "aggregate_recovery",
        "aggregate_role_keys",
        "raw_manifest_path",
        "raw_manifest_sha256",
        "source_raw_manifest_sha256",
        "authorization",
        "source_bindings",
        "loaded_input_keys",
        "contact_or_privileged_materialized",
        "target_role_isolation",
        "causal_primitive_contract",
        "module_before",
        "module_after",
        "base_provenance",
        "frozen_facade",
        "no_gradients",
    }
)
EXECUTION_RECORD_KEYS = frozenset(
    {
        "slot",
        "episode_id",
        "path",
        "sha256",
        "sidecar_path",
        "sidecar_sha256",
        "source_raw_path",
        "source_raw_sha256",
        "source_raw_sidecar_path",
        "source_raw_sidecar_sha256",
        "call_histogram",
    }
)
SOURCE_BINDING_KEYS = frozenset(
    {
        "dgp_matrix",
        "cohort_seed_ledger",
        "execution_authorization_seal",
        "raw_authorization_seal",
        "counted_feature_source",
        "pricing_source",
        "runner_source",
        "replacement_registry",
    }
)
AGGREGATE_RECOVERY_KEYS = frozenset(
    {
        "status",
        "exact_keys_verified",
        "exact_shapes_dtypes_content_verified",
        "array_sha256",
        "file_sha256",
    }
)


def _verify_source_bindings(
    value: Any, *, expected_paths: Mapping[str, Path]
) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != SOURCE_BINDING_KEYS:
        raise WorkflowError("development source-binding membership drift")
    if set(expected_paths) != SOURCE_BINDING_KEYS:
        raise WorkflowError("internal development source-binding map drift")
    records: list[dict[str, Any]] = []
    identities: dict[tuple[int, int], Path] = {}
    for name in sorted(value):
        item = value[name]
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise WorkflowError(f"invalid source binding: {name}")
        expected_path = _regular_file(Path(expected_paths[name]))
        expected_path = expected_path.resolve()
        expected_link = {
            "path": relative_to_repo(expected_path),
            "sha256": sha256_file(expected_path),
        }
        if dict(item) != expected_link:
            raise WorkflowError(f"development source binding path/hash drift: {name}")
        metadata = expected_path.stat()
        if int(metadata.st_nlink) != 1:
            raise WorkflowError(f"development source binding has inode alias: {name}")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        previous_path = identities.get(identity)
        if previous_path is not None and previous_path != expected_path:
            raise WorkflowError(
                "development source bindings alias one inode: "
                f"{previous_path} and {expected_path}"
            )
        identities[identity] = expected_path
        records.append(_verify_link(item["path"], item["sha256"]))
    return records


def _verify_role_regime(
    role: str,
    regime: str,
    *,
    expected_episodes: int,
    expected_raw_seal: Mapping[str, str],
    expected_execution_seal: Mapping[str, str],
    expected_state_sha256: str,
    data_attempt: str = ATTEMPT,
    data_attempt_root: Path = ATTEMPT_ROOT,
    configuration_root: Path = ATTEMPT_ROOT,
) -> dict[str, Any]:
    root = data_attempt_root / "data" / role / regime
    dgp_matrix_path = configuration_root / "DGP_MATRIX.json"
    cohort_ledger_path = configuration_root / "cohort_seed_ledger.json"
    replacement_registry_path = configuration_root / "data/replacement_registry.json"
    raw_manifest_path = _regular_file(root / "raw_manifest.json")
    execution_manifest_path = _regular_file(root / "execution_manifest.json")
    raw = read_json(raw_manifest_path)
    execution = read_json(execution_manifest_path)
    if set(raw) != RAW_MANIFEST_KEYS:
        raise WorkflowError(f"raw manifest schema drift: {role}/{regime}")
    if set(execution) != EXECUTION_MANIFEST_KEYS:
        raise WorkflowError(f"execution manifest schema drift: {role}/{regime}")
    _assert_no_forbidden_keys(raw, location=f"raw.{role}.{regime}")
    _assert_no_forbidden_keys(execution, location=f"execution.{role}.{regime}")

    raw_records = raw.get("episodes")
    execution_records = execution.get("episodes")
    raw_header = {
        "schema": type(raw.get("schema_version")) is int
        and raw.get("schema_version") == 1,
        "attempt": raw.get("attempt") == data_attempt,
        "complete": raw.get("complete") is True,
        "role": raw.get("role") == role,
        "regime": raw.get("regime") == regime,
        "count": raw.get("episode_count") == expected_episodes,
        "records": isinstance(raw_records, list)
        and len(raw_records) == expected_episodes,
        "members": raw.get("raw_archive_members") == ["action", "pixels"],
        "pixels_contract": raw.get("raw_pixels_contract")
        == "uint8 [201,224,224,3]",
        "action_contract": raw.get("raw_action_contract")
        == "float32 [201,5]; 200 finite plus terminal NaN",
        "role_isolation": raw.get("role_isolation") is True,
        "not_smoke": raw.get("smoke_permanently_excluded") is False,
        "retention": raw.get("retention_contract")
        == "pixels_action_shapes_finiteness_and_local_step_count_only",
        "authorization": raw.get("authorization_seal") == expected_raw_seal,
        "dgp_path": raw.get("dgp_matrix_path")
        == relative_to_repo(dgp_matrix_path),
        "dgp_hash": raw.get("dgp_matrix_sha256")
        == sha256_file(_regular_file(dgp_matrix_path)),
        "ledger_path": raw.get("cohort_seed_ledger_path")
        == relative_to_repo(cohort_ledger_path),
        "ledger_hash": raw.get("cohort_seed_ledger_sha256")
        == sha256_file(_regular_file(cohort_ledger_path)),
        "replacement_registry": raw.get("replacement_registry_path")
        == relative_to_repo(replacement_registry_path),
        "replacement_scope": raw.get("replacement_registry_scope")
        == "append_only_all_roles_and_all_dgps",
        "orphan_policy": raw.get("orphan_policy")
        == (
            "adopt raw-only archive only after exact closed intent, current "
            "authorization, prospective ledger, canonical nonlink paths, and full "
            "two-array recomputation; otherwise stop"
        ),
    }
    dgp_matrix = read_json(_regular_file(dgp_matrix_path))
    raw_header["regime_specification"] = (
        isinstance(dgp_matrix.get("regimes"), Mapping)
        and raw.get("regime_specification") == dgp_matrix["regimes"].get(regime)
    )
    if not all(raw_header.values()):
        raise WorkflowError(f"raw manifest header failed: {role}/{regime}: {raw_header}")
    if not isinstance(execution_records, list):
        raise WorkflowError("execution records are not a list")
    execution_header = {
        "schema": type(execution.get("schema_version")) is int
        and execution.get("schema_version") == 1,
        "attempt": execution.get("attempt") == data_attempt,
        "complete": execution.get("complete") is True,
        "role": execution.get("role") == role,
        "regime": execution.get("regime") == regime,
        "count": execution.get("episode_count") == expected_episodes,
        "row_count": execution.get("row_count")
        == expected_episodes * ROWS_PER_EPISODE,
        "records": len(execution_records) == expected_episodes,
        "aggregate_keys": execution.get("aggregate_role_keys")
        == list(AGGREGATE_KEYS),
        "raw_manifest_path": execution.get("raw_manifest_path")
        == relative_to_repo(raw_manifest_path),
        "raw_manifest_hash": execution.get("raw_manifest_sha256")
        == sha256_file(raw_manifest_path)
        and execution.get("source_raw_manifest_sha256")
        == sha256_file(raw_manifest_path),
        "loaded_keys": execution.get("loaded_input_keys")
        == ["action", "pixels"],
        "contact_free": execution.get("contact_or_privileged_materialized")
        is False,
        "target_isolation": execution.get("target_role_isolation") == role,
        "module_frozen": execution.get("module_before")
        == execution.get("module_after")
        and isinstance(execution.get("module_after"), Mapping)
        and execution["module_after"].get("passed") is True,
        "no_gradients": execution.get("no_gradients") is True,
    }
    if not all(execution_header.values()):
        raise WorkflowError(
            f"execution manifest header failed: {role}/{regime}: {execution_header}"
        )

    authorization = execution.get("authorization")
    expected_authorization_keys = {
        "state",
        "active_attempt",
        "state_sha256",
        "seal_path",
        "seal_sha256",
        "seal_checkpoint_state",
        "science_attempt",
        "raw_authorization_seal_path",
        "raw_authorization_seal_sha256",
        "exact_locked_snapshot_unchanged",
    }
    if (
        not isinstance(authorization, Mapping)
        or set(authorization) != expected_authorization_keys
    ):
        raise WorkflowError("execution authorization is absent")
    authorization_checks = {
        "state": authorization.get("state") == ROLE_STATES[role],
        "attempt": authorization.get("active_attempt") == data_attempt,
        "state_hash": authorization.get("state_sha256")
        == expected_state_sha256,
        "seal_path": authorization.get("seal_path")
        == expected_execution_seal["path"],
        "seal_hash": authorization.get("seal_sha256")
        == expected_execution_seal["sha256"],
        "seal_state": authorization.get("seal_checkpoint_state")
        == expected_execution_seal["checkpoint_state"],
        "science_attempt": authorization.get("science_attempt")
        == SCIENCE_ATTEMPT,
        "raw_seal_path": authorization.get("raw_authorization_seal_path")
        == expected_raw_seal["path"],
        "raw_seal_hash": authorization.get("raw_authorization_seal_sha256")
        == expected_raw_seal["sha256"],
        "locked_snapshot": authorization.get("exact_locked_snapshot_unchanged")
        is True,
    }
    if not all(authorization_checks.values()):
        raise WorkflowError(
            f"execution authorization drift: {role}/{regime}: {authorization_checks}"
        )

    verified_files: list[dict[str, Any]] = [
        {
            "path": relative_to_repo(raw_manifest_path),
            "sha256": sha256_file(raw_manifest_path),
            "bytes": raw_manifest_path.stat().st_size,
        },
        {
            "path": relative_to_repo(execution_manifest_path),
            "sha256": sha256_file(execution_manifest_path),
            "bytes": execution_manifest_path.stat().st_size,
        },
    ]
    raw_ids: list[str] = []
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for expected_slot, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise WorkflowError("raw episode record is not an object")
        episode_id = str(record.get("episode_id"))
        if (
            record.get("role") != role
            or record.get("regime") != regime
            or record.get("complete") is not True
            or int(record.get("slot", -1)) != expected_slot
            or record.get("raw_archive_members") != ["action", "pixels"]
            or record.get("authorization_seal") != expected_raw_seal
            or not isinstance(record.get("arrays"), Mapping)
            or set(record["arrays"]) != {"action", "pixels"}
        ):
            raise WorkflowError(f"raw episode contract drift: {role}/{regime}/{expected_slot}")
        _assert_no_forbidden_keys(
            record, location=f"raw_record.{role}.{regime}.{expected_slot}"
        )
        raw_link = _verify_link(
            record.get("raw_path"), record.get("raw_sha256"), required_root=root / "raw"
        )
        raw_path = resolve_repo_relative(record["raw_path"])
        expected_raw_path = root / "raw" / f"{episode_id}.npz"
        if raw_path != expected_raw_path.resolve():
            raise WorkflowError("raw episode path is not canonical for its identifier")
        sidecar_path = _regular_file(raw_path.with_suffix(".json"))
        if read_json(sidecar_path) != dict(record):
            raise WorkflowError("raw sidecar differs from its manifest record")
        intent_link = _verify_link(
            record.get("persistence_intent_path"),
            record.get("persistence_intent_sha256"),
            required_root=data_attempt_root
            / "data/persistence_intents"
            / role
            / regime,
        )
        verified_files.extend(
            [
                raw_link,
                {
                    "path": relative_to_repo(sidecar_path),
                    "sha256": sha256_file(sidecar_path),
                    "bytes": sidecar_path.stat().st_size,
                },
                intent_link,
            ]
        )
        if episode_id in raw_by_id:
            raise WorkflowError("duplicate raw episode identifier")
        raw_ids.append(episode_id)
        raw_by_id[episode_id] = record
    if raw.get("replacements_used") != sum(
        item.get("replacement_used") is True for item in raw_records
    ):
        raise WorkflowError("raw replacement count drift")

    execution_ids: list[str] = []
    semantic_summaries: list[dict[str, Any]] = []
    for expected_slot, record in enumerate(execution_records):
        if not isinstance(record, Mapping) or set(record) != EXECUTION_RECORD_KEYS:
            raise WorkflowError("execution episode record schema drift")
        episode_id = str(record.get("episode_id"))
        if int(record.get("slot", -1)) != expected_slot or episode_id not in raw_by_id:
            raise WorkflowError("execution slot or identity drift")
        if record.get("call_histogram") is not None:
            raise WorkflowError("development execution unexpectedly contains routing calls")
        part_link = _verify_link(
            record.get("path"), record.get("sha256"), required_root=root / "execution"
        )
        sidecar_link = _verify_link(
            record.get("sidecar_path"),
            record.get("sidecar_sha256"),
            required_root=root / "execution",
        )
        part_path = resolve_repo_relative(record["path"])
        sidecar_path = resolve_repo_relative(record["sidecar_path"])
        if (
            part_path != (root / "execution" / f"{episode_id}.npz").resolve()
            or sidecar_path != (root / "execution" / f"{episode_id}.json").resolve()
        ):
            raise WorkflowError("execution episode path is not identifier-canonical")
        raw_record = raw_by_id[episode_id]
        raw_loader_audit = _validate_input_loader_audit(
            raw_record.get("input_loader_audit"), raw_record=raw_record
        )
        raw_sidecar_path = resolve_repo_relative(record["source_raw_sidecar_path"])
        source_checks = {
            "raw_path": record.get("source_raw_path") == raw_record.get("raw_path"),
            "raw_hash": record.get("source_raw_sha256")
            == raw_record.get("raw_sha256"),
            "raw_sidecar_path": raw_sidecar_path
            == resolve_repo_relative(raw_record["raw_path"]).with_suffix(".json"),
            "raw_sidecar_hash": record.get("source_raw_sidecar_sha256")
            == sha256_file(raw_sidecar_path),
        }
        if not all(source_checks.values()):
            raise WorkflowError(f"execution/raw cross-link drift: {source_checks}")
        sidecar = read_json(sidecar_path)
        if set(sidecar) != DEVELOPMENT_SIDECAR_KEYS:
            raise WorkflowError("development execution sidecar schema drift")
        _assert_no_forbidden_keys(
            sidecar, location=f"execution_sidecar.{role}.{regime}.{expected_slot}"
        )
        arrays = sidecar.get("arrays")
        sidecar_checks = {
            "header": type(sidecar.get("schema_version")) is int
            and sidecar.get("schema_version") == 1
            and sidecar.get("attempt") == data_attempt
            and sidecar.get("role") == role
            and sidecar.get("regime") == regime
            and type(sidecar.get("created_unix_ns")) is int
            and int(sidecar["created_unix_ns"]) > 0
            and sidecar.get("complete") is True,
            "identity": sidecar.get("slot") == expected_slot
            and sidecar.get("episode_id") == episode_id,
            "part": sidecar.get("path") == record.get("path")
            and sidecar.get("sha256") == record.get("sha256")
            and sidecar.get("bytes") == part_path.stat().st_size,
            "raw": sidecar.get("raw_path") == record.get("source_raw_path")
            and sidecar.get("raw_sha256") == record.get("source_raw_sha256")
            and sidecar.get("raw_sidecar_path")
            == record.get("source_raw_sidecar_path")
            and sidecar.get("raw_sidecar_sha256")
            == record.get("source_raw_sidecar_sha256"),
            "loader_audit": sidecar.get("input_loader_audit")
            == raw_loader_audit,
            "evaluation": sidecar.get("evaluation")
            == "four dense exits and all three frozen causal feature stages",
            "target_use": sidecar.get("target_use")
            == f"permitted only for isolated {role}",
            "contact_free": sidecar.get("contact_or_privileged_materialized")
            is False,
            "primitive_reconstruction": sidecar.get(
                "primitive_feature_reconstruction_exact"
            )
            is True,
            "no_gradients": sidecar.get("no_gradients") is True,
            "array_contract": isinstance(arrays, Mapping)
            and set(arrays) == DEVELOPMENT_PART_KEYS,
        }
        if not all(sidecar_checks.values()):
            raise WorkflowError(f"execution sidecar contract failed: {sidecar_checks}")
        for name, item in arrays.items():
            expected_shape, expected_dtype = DEVELOPMENT_PART_CONTRACT[name]
            if (
                not isinstance(item, Mapping)
                or set(item) != {"shape", "dtype", "sha256"}
                or item.get("shape") != expected_shape
                or item.get("dtype") != expected_dtype
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
            ):
                raise WorkflowError(f"execution array contract drift: {name}")
        semantic_summaries.append(
            _verify_development_part_arrays(
                part_path,
                arrays,
                expected_slot=expected_slot,
            )
        )
        verified_files.extend([part_link, sidecar_link])
        execution_ids.append(episode_id)
    if execution_ids != raw_ids:
        raise WorkflowError("raw/execution episode order differs")

    aggregate_link = _verify_link(
        execution.get("aggregate_role_path"),
        execution.get("aggregate_role_sha256"),
        required_root=root,
    )
    aggregate_path = resolve_repo_relative(execution["aggregate_role_path"])
    if aggregate_path != (root / "role.npz").resolve():
        raise WorkflowError("aggregate role path drift")
    if set(_npz_member_names(aggregate_path)) != {
        f"{name}.npy" for name in AGGREGATE_KEYS
    }:
        raise WorkflowError("aggregate NPZ member drift")
    recovery = execution.get("aggregate_recovery")
    if not isinstance(recovery, Mapping) or set(recovery) != AGGREGATE_RECOVERY_KEYS:
        raise WorkflowError("aggregate interruption-recovery audit schema drift")
    recovery_hashes = recovery.get("array_sha256")
    recovery_checks = {
        "status": recovery.get("status")
        in {
            "new_complete_aggregate_written_and_verified",
            "existing_interruption_complete_aggregate_adopted",
            "concurrent_complete_aggregate_adopted",
        },
        "keys": recovery.get("exact_keys_verified") is True,
        "content": recovery.get("exact_shapes_dtypes_content_verified") is True,
        "array_hashes": isinstance(recovery_hashes, Mapping)
        and set(recovery_hashes) == set(AGGREGATE_KEYS)
        and all(
            isinstance(value, str) and len(value) == 64
            for value in recovery_hashes.values()
        ),
        "file_hash": recovery.get("file_sha256") == aggregate_link["sha256"],
    }
    if not all(recovery_checks.values()):
        raise WorkflowError(f"aggregate interruption-recovery audit failed: {recovery_checks}")
    verified_files.append(aggregate_link)
    expected_binding_paths = {
        "dgp_matrix": dgp_matrix_path,
        "cohort_seed_ledger": cohort_ledger_path,
        "execution_authorization_seal": resolve_repo_relative(
            expected_execution_seal["path"]
        ),
        "raw_authorization_seal": resolve_repo_relative(
            expected_raw_seal["path"]
        ),
        "counted_feature_source": configuration_root / "counted_features.py",
        "pricing_source": configuration_root / "flops.py",
        "runner_source": configuration_root / "runner.py",
        "replacement_registry": replacement_registry_path,
    }
    source_bindings = execution.get("source_bindings")
    verified_files.extend(
        _verify_source_bindings(
            source_bindings,
            expected_paths=expected_binding_paths,
        )
    )
    if not isinstance(source_bindings, Mapping):
        raise WorkflowError("development source bindings are absent")
    if source_bindings["raw_authorization_seal"] != {
        "path": expected_raw_seal["path"],
        "sha256": expected_raw_seal["sha256"],
    }:
        raise WorkflowError("raw source binding does not equal raw authorization")
    if source_bindings["execution_authorization_seal"] != {
        "path": expected_execution_seal["path"],
        "sha256": expected_execution_seal["sha256"],
    }:
        raise WorkflowError(
            "execution source binding does not equal execution authorization"
        )
    return {
        "regime": regime,
        "episode_count": expected_episodes,
        "row_count": expected_episodes * ROWS_PER_EPISODE,
        "raw_manifest": {
            "path": relative_to_repo(raw_manifest_path),
            "sha256": sha256_file(raw_manifest_path),
        },
        "execution_manifest": {
            "path": relative_to_repo(execution_manifest_path),
            "sha256": sha256_file(execution_manifest_path),
        },
        "aggregate": {
            "path": relative_to_repo(aggregate_path),
            "sha256": sha256_file(aggregate_path),
            "bytes": aggregate_path.stat().st_size,
        },
        "part_semantic_summaries": semantic_summaries,
        "development_part_arrays_opened": True,
        "aggregate_arrays_opened": False,
        "episode_ids_sha256": canonical_sha256(raw_ids),
        "verified_file_count": len(verified_files),
        "verified_bytes": sum(int(item["bytes"]) for item in verified_files),
        "verified_file_index_sha256": canonical_sha256(
            [
                {key: item[key] for key in ("path", "sha256", "bytes")}
                for item in sorted(verified_files, key=lambda item: item["path"])
            ]
        ),
    }


def _controller_state_object_sha256(value: Mapping[str, Any]) -> str:
    """Reproduce the controller's indented, sorted, newline-terminated state bytes."""

    encoded = (
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_ast_sha256(path: Path) -> str:
    path = _regular_file(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise WorkflowError(f"cannot parse controller source: {path}") from error
    return hashlib.sha256(
        ast.dump(tree, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _verify_recovery_adapter_marker(
    value: Any,
    *,
    event_count: int,
    head_sha256: str,
    state_without_marker: Mapping[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "authorization_kind",
        "adapter_source_path",
        "adapter_source_sha256",
        "adapter_source_ast_sha256",
        "root_program_path",
        "root_program_sha256",
        "root_program_ast_sha256",
        "ledger_event_count",
        "ledger_head_sha256",
        "operation_sha256",
        "state_binding_sha256",
    }
    adapter = _regular_file(CONTROLLER_ADAPTER_PATH)
    root_program = _regular_file(ROOT_PROGRAM_PATH)
    operation = value.get("operation_sha256") if isinstance(value, Mapping) else None
    state_binding = (
        value.get("state_binding_sha256") if isinstance(value, Mapping) else None
    )
    marker_without_state_binding = dict(value) if isinstance(value, Mapping) else {}
    marker_without_state_binding.pop("state_binding_sha256", None)
    expected_state_binding = (
        canonical_sha256(
            {
                "marker_without_state_binding": marker_without_state_binding,
                "state_without_marker": dict(state_without_marker),
            }
        )
        if state_without_marker is not None
        else None
    )
    checks = {
        "schema": isinstance(value, Mapping) and set(value) == expected_keys,
        "version": isinstance(value, Mapping)
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 2,
        "authorization": isinstance(value, Mapping)
        and value.get("authorization_kind")
        == "receipt_bound_v008_root_controller_adapter",
        "adapter_path": isinstance(value, Mapping)
        and value.get("adapter_source_path")
        == relative_to_repo(adapter),
        "adapter_hash": isinstance(value, Mapping)
        and value.get("adapter_source_sha256") == sha256_file(adapter),
        "adapter_ast": isinstance(value, Mapping)
        and value.get("adapter_source_ast_sha256") == _source_ast_sha256(adapter),
        "root_path": isinstance(value, Mapping)
        and value.get("root_program_path") == relative_to_repo(root_program),
        "root_hash": isinstance(value, Mapping)
        and value.get("root_program_sha256") == sha256_file(root_program),
        "root_ast": isinstance(value, Mapping)
        and value.get("root_program_ast_sha256")
        == _source_ast_sha256(root_program),
        "ledger_count": isinstance(value, Mapping)
        and type(value.get("ledger_event_count")) is int
        and value.get("ledger_event_count") == event_count,
        "ledger_head": isinstance(value, Mapping)
        and value.get("ledger_head_sha256") == head_sha256,
        "operation": isinstance(operation, str)
        and len(operation) == 64
        and all(character in "0123456789abcdef" for character in operation),
        "state_binding": isinstance(state_binding, str)
        and len(state_binding) == 64
        and all(
            character in "0123456789abcdef" for character in state_binding
        )
        and (
            expected_state_binding is None
            or state_binding == expected_state_binding
        ),
    }
    if not all(checks.values()):
        raise WorkflowError(f"{label} adapter marker drift: {checks}")
    return dict(value)


def _common_role_authorization_state_sha256(role: str) -> str:
    observed: list[str] = []
    for regime in REGIMES:
        path = _regular_file(
            ATTEMPT_ROOT / "data" / role / regime / "execution_manifest.json"
        )
        manifest = read_json(path)
        authorization = manifest.get("authorization")
        value = (
            authorization.get("state_sha256")
            if isinstance(authorization, Mapping)
            else None
        )
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise WorkflowError(
                f"invalid execution authorization state hash: {role}/{regime}"
            )
        observed.append(value)
    if len(set(observed)) != 1:
        raise WorkflowError(
            f"{role} execution manifests do not share one pre-count authorization"
        )
    return observed[0]


def _verify_role_count_update_recovery(
    role: str,
    state: Mapping[str, Any],
    *,
    pre_count_authorization_sha256: str | None,
    inherited_fit_recovery_guard: object | None = None,
) -> dict[str, Any]:
    """Authenticate the sole recoverable crash window after role-count commit.

    The execution manifests bind the exact controller state before count
    registration.  Recovery is allowed only by reversing one final, exact
    outcome-count ledger event and reproducing those pre-count state bytes.
    """

    state_name = ROLE_STATES[role]
    counter = ROLE_COUNTER_FIELDS[role]
    expected_key = (
        "expected_fit_episode_count"
        if role == "fit"
        else "expected_selection_episode_count"
    )
    expected = state.get(expected_key)
    if (
        state.get("active_attempt") != ATTEMPT
        or state.get("current_state") != state_name
        or type(expected) is not int
        or type(state.get(counter)) is not int
        or state.get(counter) != expected
    ):
        raise WorkflowError("role-count recovery has invalid state/count identity")

    counter_expectations: dict[str, int | bool] = {
        "fit_outcome_episodes": (
            expected
            if role == "fit"
            else state.get("expected_fit_episode_count", -1)
        ),
        "selection_outcome_episodes": expected if role == "selection" else 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    bad_counters = {
        name: {"expected": value, "observed": state.get(name)}
        for name, value in counter_expectations.items()
        if type(state.get(name)) is not type(value) or state.get(name) != value
    }
    if bad_counters:
        raise WorkflowError(
            f"role-count recovery has changed other/later counters: {bad_counters}"
        )
    if any(
        state.get(name) is not None
        for name in (
            "early_scientific_failure",
            "postconfirmation_integrity_failure",
            "terminal_label",
        )
    ):
        raise WorkflowError("role-count recovery cannot follow a terminal branch")

    audit_path = ATTEMPT_ROOT / "audit" / f"{role}_cohorts.json"
    audit_present = os.path.lexists(audit_path)
    audit_relative = relative_to_repo(audit_path)
    completed = state.get("completed_states")
    checkpoints = state.get("verified_checkpoints")
    if (
        not isinstance(completed, list)
        or state_name in completed
        or not isinstance(checkpoints, list)
        or any(
            not isinstance(item, Mapping)
            or item.get("evidence_path") == audit_relative
            or item.get("name") == f"{ATTEMPT}_{role}_cohorts_verified"
            for item in checkpoints
        )
    ):
        raise WorkflowError("role-count recovery found a role checkpoint")

    state_file = read_json(_regular_file(STATE_PATH))
    if state_file != dict(state):
        raise WorkflowError("role-count recovery state differs from controller status")
    current_state_sha256 = sha256_file(STATE_PATH)
    if current_state_sha256 == pre_count_authorization_sha256:
        raise WorkflowError("role-count recovery was requested without a state mismatch")

    ledger_path = _regular_file(LEDGER_PATH)
    try:
        ledger_payload = ledger_path.read_bytes()
        lines = ledger_payload.splitlines(keepends=True)
        records = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkflowError("cannot authenticate the recovery ledger") from error
    event_count = state.get("ledger_event_count")
    if (
        type(event_count) is not int
        or event_count < 2
        or len(records) != event_count
        or not all(isinstance(item, dict) for item in records[-2:])
    ):
        raise WorkflowError("role-count recovery ledger length/type drift")
    previous = records[-2]
    event = records[-1]
    event_keys = {
        "event",
        "attempt",
        "fields",
        "prior_controller_adapter_marker",
        "created_unix_ns",
        "seq",
        "prev_sha256",
        "record_sha256",
    }
    event_payload = dict(event)
    observed_event_hash = event_payload.pop("record_sha256", None)
    event_checks = {
        "exact_schema": set(event) == event_keys,
        "event": event.get("event") == "outcome_counts_updated",
        "attempt": event.get("attempt") == ATTEMPT,
        "fields": event.get("fields") == {counter: expected},
        "sequence": event.get("seq") == event_count,
        "created": type(event.get("created_unix_ns")) is int
        and type(state.get("updated_unix_ns")) is int
        and event.get("created_unix_ns") == state.get("updated_unix_ns"),
        "head": observed_event_hash == state.get("ledger_head_sha256"),
        "hash": observed_event_hash == canonical_sha256(event_payload),
    }
    previous_payload = dict(previous)
    observed_previous_hash = previous_payload.pop("record_sha256", None)
    previous_checks = {
        "sequence": previous.get("seq") == event_count - 1,
        "created": type(previous.get("created_unix_ns")) is int,
        "hash": observed_previous_hash == canonical_sha256(previous_payload),
        "link": observed_previous_hash == event.get("prev_sha256"),
    }
    if not all(event_checks.values()) or not all(previous_checks.values()):
        raise WorkflowError(
            "role-count recovery ledger event drift: "
            f"event={event_checks}, previous={previous_checks}"
        )

    expected_event_suffix = (
        json.dumps(
            event,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    if lines[-1] != expected_event_suffix:
        raise WorkflowError(
            "role-count recovery final ledger suffix is not canonical"
        )
    base_ledger_payload = b"".join(lines[:-1])

    current_state_without_marker = json.loads(json.dumps(dict(state)))
    current_state_without_marker.pop("v008_durable_controller_adapter", None)
    current_adapter_marker = _verify_recovery_adapter_marker(
        state.get("v008_durable_controller_adapter"),
        event_count=event_count,
        head_sha256=str(observed_event_hash),
        state_without_marker=current_state_without_marker,
        label="current role-count recovery",
    )

    pre_count_state_without_marker = dict(current_state_without_marker)
    pre_count_state_without_marker[counter] = 0
    pre_count_state_without_marker["updated_unix_ns"] = previous[
        "created_unix_ns"
    ]
    pre_count_state_without_marker["ledger_event_count"] = event_count - 1
    pre_count_state_without_marker["ledger_head_sha256"] = observed_previous_hash
    prior_adapter_marker = _verify_recovery_adapter_marker(
        event.get("prior_controller_adapter_marker"),
        event_count=event_count - 1,
        head_sha256=str(observed_previous_hash),
        state_without_marker=pre_count_state_without_marker,
        label="prior role-count recovery",
    )

    pre_count_state = dict(pre_count_state_without_marker)
    pre_count_state["v008_durable_controller_adapter"] = prior_adapter_marker
    operation_context = {
        "base_state_object_sha256": canonical_sha256(pre_count_state),
        "base_ledger_sha256": hashlib.sha256(base_ledger_payload).hexdigest(),
        "expected_suffix_sha256": hashlib.sha256(
            expected_event_suffix
        ).hexdigest(),
        "intended_state_object_sha256": canonical_sha256(
            current_state_without_marker
        ),
    }
    expected_operation_sha256 = canonical_sha256(operation_context)
    if current_adapter_marker.get("operation_sha256") != expected_operation_sha256:
        raise WorkflowError(
            "current role-count recovery adapter operation/context drift"
        )
    reconstructed_sha256 = _controller_state_object_sha256(pre_count_state)
    inherited_reconstruction = (
        pre_count_authorization_sha256 is None
        and role == "fit"
        and inherited_fit_recovery_guard is _INHERITED_FIT_RECOVERY_GUARD
    )
    if pre_count_authorization_sha256 is None and not inherited_reconstruction:
        raise WorkflowError("role-count recovery lacks pre-count authorization")
    if (
        pre_count_authorization_sha256 is not None
        and reconstructed_sha256 != pre_count_authorization_sha256
    ):
        raise WorkflowError(
            "role-count recovery cannot reconstruct the manifests' pre-count state"
        )
    effective_pre_count_sha256 = reconstructed_sha256
    base_recovery = {
        "passed": True,
        "role": role,
        "state": state_name,
        "counter_field": counter,
        "counter_value": expected,
        "pre_count_authorization_sha256": effective_pre_count_sha256,
        "current_state_sha256": current_state_sha256,
        "count_update_event_seq": event_count,
        "count_update_event_sha256": observed_event_hash,
        "count_update_event_fields": {counter: expected},
        "prior_controller_adapter_marker_sha256": canonical_sha256(
            prior_adapter_marker
        ),
        "current_controller_adapter_marker_sha256": canonical_sha256(
            state["v008_durable_controller_adapter"]
        ),
        "prior_state_binding_sha256": prior_adapter_marker[
            "state_binding_sha256"
        ],
        "current_state_binding_sha256": current_adapter_marker[
            "state_binding_sha256"
        ],
        "count_operation_sha256": expected_operation_sha256,
        "count_operation_context": operation_context,
        "no_role_audit_or_checkpoint_before_recovery": True,
        "no_role_checkpoint_before_recovery": True,
        "role_audit_already_published": False,
        "existing_role_audit_sha256": None,
        "existing_role_audit_created_unix_ns": None,
        "existing_role_audit_verification_kind": None,
        "all_other_later_counters_unchanged": True,
        "exactly_one_controller_event_after_authorized_state": True,
    }
    existing_audit_sha256: str | None = None
    existing_audit_created_unix_ns: int | None = None
    existing_audit_verification_kind: str | None = None
    if audit_present:
        audit = read_json(_regular_file(audit_path))
        audit_fields = {
            "schema_version",
            "attempt",
            "checkpoint_state",
            "created_unix_ns",
            "passed",
            "checks",
            "controller_status_sha256",
            "outcome_counts",
            "evidence",
            "scientific_objects_changed",
            "confirmation_outcomes_opened_by_workflow",
        }
        audit_checks = audit.get("checks")
        audit_evidence = audit.get("evidence")
        prior_verification = (
            audit_evidence.get("role_verification")
            if isinstance(audit_evidence, Mapping)
            else None
        )
        prior_recovery = (
            prior_verification.get("role_count_update_recovery")
            if isinstance(prior_verification, Mapping)
            else None
        )
        if prior_recovery is None:
            existing_audit_verification_kind = "pre_count_verification"
            prior_recovery_valid = True
        elif (
            isinstance(prior_recovery, Mapping)
            and dict(prior_recovery) == base_recovery
        ):
            existing_audit_verification_kind = "count_recovery_verification"
            prior_recovery_valid = True
        else:
            prior_recovery_valid = False
        expected_audit_checks = {
            "all_four_regimes_verified": True,
            "exact_episode_count": True,
            "exact_row_count": True,
            "controller_count_exact": True,
            "aggregate_arrays_unopened": True,
            "scientific_replay_qualified": True,
            "confirmation_unopened": True,
        }
        expected_outcome_counts = {
            name: state.get(name)
            for name in (
                "fit_outcome_episodes",
                "selection_outcome_episodes",
                "smoke_outcome_episodes",
                "confirmation_outcome_episodes_generated",
                "confirmation_outcome_episodes_executed",
                "confirmation_outcomes_opened_for_analysis",
            )
        }
        audit_contract = {
            "fields": set(audit) == audit_fields,
            "schema": audit.get("schema_version") == 1,
            "attempt": audit.get("attempt") == ATTEMPT,
            "state": audit.get("checkpoint_state") == state_name,
            "timestamp": type(audit.get("created_unix_ns")) is int
            and int(audit["created_unix_ns"])
            >= int(event["created_unix_ns"]),
            "passed": audit.get("passed") is True,
            "checks": audit_checks == expected_audit_checks,
            "controller": audit.get("controller_status_sha256")
            == canonical_sha256(dict(state)),
            "counts": audit.get("outcome_counts") == expected_outcome_counts,
            "evidence_fields": isinstance(audit_evidence, Mapping)
            and set(audit_evidence) == {"role_verification"},
            "scientific_unchanged": audit.get("scientific_objects_changed") is False,
            "confirmation_unopened": audit.get(
                "confirmation_outcomes_opened_by_workflow"
            )
            == 0,
            "prior_verification": isinstance(prior_verification, Mapping)
            and prior_verification.get("passed") is True
            and prior_verification.get("role") == role
            and prior_verification.get("regime_order") == list(REGIMES)
            and prior_verification.get("episodes_per_regime")
            == ROLE_EPISODES_PER_DGP[role]
            and prior_verification.get("episode_count") == expected
            and prior_verification.get("authorization_state_sha256")
            == effective_pre_count_sha256
            and prior_recovery_valid
            and prior_verification.get(
                "npz_arrays_opened_during_manifest_verification"
            )
            is True
            and prior_verification.get("development_part_arrays_opened") is True
            and prior_verification.get("aggregate_arrays_opened") is False,
        }
        if isinstance(prior_verification, Mapping):
            _validate_role_replay_qualification(role, prior_verification)
        if not all(audit_contract.values()):
            raise WorkflowError(
                "role-count recovery existing role audit drift: "
                f"{audit_contract}"
            )
        existing_audit_sha256 = sha256_file(audit_path)
        existing_audit_created_unix_ns = int(audit["created_unix_ns"])
    recovery = dict(base_recovery)
    if audit_present:
        recovery.update(
            {
                "no_role_audit_or_checkpoint_before_recovery": False,
                "role_audit_already_published": True,
                "existing_role_audit_sha256": existing_audit_sha256,
                "existing_role_audit_created_unix_ns": (
                    existing_audit_created_unix_ns
                ),
                "existing_role_audit_verification_kind": (
                    existing_audit_verification_kind
                ),
            }
        )
    return recovery


def _verify_inherited_fit_manifests(state: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the immutable v003 fit role under the v008 controller lineage."""

    expected_total = ROLE_EPISODES_PER_DGP["fit"] * len(REGIMES)
    if (
        state.get("active_attempt") != ATTEMPT
        or state.get("current_state") != ROLE_STATES["fit"]
        or state.get("expected_fit_episode_count") != expected_total
        or type(state.get("fit_outcome_episodes")) is not int
        or state.get("fit_outcome_episodes") not in {0, expected_total}
    ):
        raise WorkflowError("inherited fit verification has invalid controller state")
    try:
        inventory_result = fit_inheritance.verify_inventory()
    except fit_inheritance.FitInheritanceError as error:
        raise WorkflowError("immutable fit-role inventory verification failed") from error
    inheritance_path = _regular_file(INHERITANCE_SEAL_PATH)
    inheritance = read_json(inheritance_path)
    inventory_link = inheritance.get("inherited_fit_role")
    expected_inventory_link = {
        "source_attempt": fit_inheritance.SOURCE_ATTEMPT,
        "inventory_path": relative_to_repo(fit_inheritance.INVENTORY_PATH),
        "inventory_sha256": sha256_file(_regular_file(fit_inheritance.INVENTORY_PATH)),
        "file_count": fit_inheritance.EXPECTED_FILE_COUNT,
        "episode_count": expected_total,
        "outcome_arrays_opened": False,
        "passed": True,
    }
    if (
        inventory_result.get("passed") is not True
        or inventory_result.get("episode_count") != expected_total
        or inventory_result.get("file_count") != fit_inheritance.EXPECTED_FILE_COUNT
        or inventory_link != expected_inventory_link
        or inheritance.get("attempt") != ATTEMPT
        or inheritance.get("source_attempt") != VERSION_FORWARD_SOURCE_ATTEMPT
        or inheritance.get("target_attempt") != ATTEMPT
        or inheritance.get("passed") is not True
    ):
        raise WorkflowError("fit-role inventory is not bound by the active inheritance seal")
    source_invalidity = read_json(_regular_file(fit_inheritance.SOURCE_INVALIDITY_PATH))
    source_state_sha256 = source_invalidity.get("frozen_evidence", {}).get(
        "controller_state_sha256_at_failure"
    )
    if not isinstance(source_state_sha256, str) or len(source_state_sha256) != 64:
        raise WorkflowError("source fit authorization-state hash is absent")
    source_seal_link = {
        "path": relative_to_repo(fit_inheritance.SOURCE_SEAL_PATH),
        "sha256": sha256_file(_regular_file(fit_inheritance.SOURCE_SEAL_PATH)),
        "checkpoint_state": "PRE_OUTCOME_SEAL",
    }
    current_state_sha256 = sha256_file(_regular_file(STATE_PATH))
    recovery: dict[str, Any] | None = None
    if state.get("fit_outcome_episodes") == 0:
        authorization_state_sha256 = current_state_sha256
    else:
        recovery = _verify_role_count_update_recovery(
            "fit",
            state,
            pre_count_authorization_sha256=None,
            inherited_fit_recovery_guard=_INHERITED_FIT_RECOVERY_GUARD,
        )
        authorization_state_sha256 = str(
            recovery["pre_count_authorization_sha256"]
        )
    regimes = {
        regime: _verify_role_regime(
            "fit",
            regime,
            expected_episodes=ROLE_EPISODES_PER_DGP["fit"],
            expected_raw_seal=source_seal_link,
            expected_execution_seal=source_seal_link,
            expected_state_sha256=source_state_sha256,
            data_attempt=fit_inheritance.SOURCE_ATTEMPT,
            data_attempt_root=fit_inheritance.SOURCE_ROOT,
            configuration_root=fit_inheritance.SOURCE_ROOT,
        )
        for regime in REGIMES
    }
    aggregate_paths = {
        regime: regimes[regime]["aggregate"]["path"] for regime in REGIMES
    }
    aggregate_hashes = {
        path: regimes[regime]["aggregate"]["sha256"]
        for regime, path in aggregate_paths.items()
    }
    result = {
        "role": "fit",
        "regime_order": list(REGIMES),
        "episodes_per_regime": ROLE_EPISODES_PER_DGP["fit"],
        "episode_count": sum(item["episode_count"] for item in regimes.values()),
        "row_count": sum(item["row_count"] for item in regimes.values()),
        "authorization_seal": {
            "path": relative_to_repo(inheritance_path),
            "sha256": sha256_file(inheritance_path),
            "checkpoint_state": "PRE_OUTCOME_SEAL",
        },
        "raw_authorization_seal": source_seal_link,
        "execution_authorization_seal": source_seal_link,
        "authorization_state_sha256": authorization_state_sha256,
        "source_authorization_state_sha256": source_state_sha256,
        "inherited_fit_role": expected_inventory_link,
        "role_count_update_recovery": recovery,
        "regimes": regimes,
        "aggregate_paths": aggregate_paths,
        "aggregate_hashes": dict(sorted(aggregate_hashes.items())),
        "development_part_arrays_opened": True,
        "aggregate_arrays_opened": False,
        "npz_arrays_opened_during_manifest_verification": True,
        "passed": True,
    }
    if result["episode_count"] != expected_total:
        raise WorkflowError("verified inherited fit total count drift")
    if (
        sha256_file(_regular_file(STATE_PATH)) != current_state_sha256
        or read_json(STATE_PATH) != dict(state)
    ):
        raise WorkflowError("controller state changed during inherited fit verification")
    return result


def verify_role_manifests(role: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every raw/execution/aggregate byte without loading outcome arrays."""

    if role not in ROLE_EPISODES_PER_DGP:
        raise WorkflowError(f"unsupported development role: {role}")
    if role == "fit":
        return _verify_inherited_fit_manifests(state)
    if state.get("active_attempt") != ATTEMPT or state.get("current_state") != ROLE_STATES[role]:
        raise WorkflowError(f"{role} manifests are unauthorized in the current state")
    expected_per_dgp = ROLE_EPISODES_PER_DGP[role]
    expected_total = expected_per_dgp * len(REGIMES)
    state_expected = state.get(
        "expected_fit_episode_count" if role == "fit" else "expected_selection_episode_count"
    )
    if state_expected != expected_total:
        raise WorkflowError(f"controller {role} count differs from the frozen design")
    raw_seal_path, raw_seal_state = ROLE_RAW_SEALS[role]
    execution_seal_path, execution_seal_state = ROLE_EXECUTION_SEALS[role]
    recovery_requested = state.get(ROLE_COUNTER_FIELDS[role], 0) != 0
    inherited_kwargs: dict[str, Any] = {}
    if recovery_requested:
        inherited_kwargs = {
            "authorized_role_count_recovery_guard_token": (
                _ROLE_COUNT_RECOVERY_GUARD_TOKEN
            ),
            "authorized_role_count_recovery": role,
        }
    try:
        inherited = verify_inherited_pre_data_authorization(
            expected_current_state=ROLE_STATES[role],
            **inherited_kwargs,
        )
    except InheritedAuthorizationError as error:
        raise WorkflowError(
            f"{role} raw manifests lack authenticated inherited pre-data lineage"
        ) from error
    raw_seal = read_json(_regular_file(raw_seal_path))
    transaction_path = _regular_file(
        ATTEMPT_ROOT / "audit/version_forward_transaction_receipt.json"
    )
    transaction = inherited.get("version_forward_transaction_receipt")
    inherited_recovery = inherited.get("role_count_recovery_authorization")
    independent_recomputation = inherited.get("independent_recomputation")
    if (
        raw_seal.get("passed") is not True
        or raw_seal.get("attempt") != ATTEMPT
        or raw_seal.get("science_attempt") != SCIENCE_ATTEMPT
        or raw_seal.get("checkpoint_state") != raw_seal_state
        or inherited.get("seal_path") != relative_to_repo(raw_seal_path)
        or inherited.get("seal_sha256") != sha256_file(raw_seal_path)
        or not isinstance(transaction, Mapping)
        or transaction.get("passed") is not True
        or transaction.get("path") != relative_to_repo(transaction_path)
        or transaction.get("sha256") != sha256_file(transaction_path)
        or not isinstance(independent_recomputation, Mapping)
        or independent_recomputation.get("authorized_role_count_recovery")
        != (role if recovery_requested else None)
        or (
            recovery_requested
            and (
                not isinstance(inherited_recovery, Mapping)
                or inherited_recovery.get("passed") is not True
                or inherited_recovery.get("role") != role
                or inherited_recovery.get("state") != ROLE_STATES[role]
                or inherited_recovery.get("counter_field")
                != ROLE_COUNTER_FIELDS[role]
                or inherited_recovery.get("counter_value") != expected_total
                or inherited_recovery.get("no_later_output_conflicts") is not True
            )
        )
        or (not recovery_requested and inherited_recovery is not None)
    ):
        raise WorkflowError(f"invalid {role} raw authorization seal")
    raw_seal_link = {
        "path": relative_to_repo(raw_seal_path),
        "sha256": sha256_file(raw_seal_path),
        "checkpoint_state": raw_seal_state,
    }
    execution_seal = read_json(_regular_file(execution_seal_path))
    if (
        execution_seal.get("passed") is not True
        or execution_seal.get("attempt") != ATTEMPT
        or execution_seal.get("checkpoint_state") != execution_seal_state
        or (
            role == "selection"
            and execution_seal.get("pre_data_seal_sha256")
            != inherited.get("seal_sha256")
        )
    ):
        raise WorkflowError(f"invalid {role} execution authorization seal")
    execution_seal_link = {
        "path": relative_to_repo(execution_seal_path),
        "sha256": sha256_file(execution_seal_path),
        "checkpoint_state": execution_seal_state,
    }
    if role == "selection" and not any(
        checkpoint.get("source_attempt") == ATTEMPT
        and checkpoint.get("evidence_path") == execution_seal_link["path"]
        and checkpoint.get("evidence_sha256") == execution_seal_link["sha256"]
        for checkpoint in state.get("verified_checkpoints", [])
    ):
        raise WorkflowError("selection execution seal lacks a direct v008 checkpoint")
    current_state_sha256 = sha256_file(_regular_file(STATE_PATH))
    if (
        inherited.get("authenticated_state_sha256") != current_state_sha256
        or _controller_state_object_sha256(state) != current_state_sha256
    ):
        raise WorkflowError(
            f"{role} workflow and inherited authorization states differ"
        )
    recovery: dict[str, Any] | None = None
    if state.get(ROLE_COUNTER_FIELDS[role], 0) == 0:
        role_audit_path = ATTEMPT_ROOT / "audit" / f"{role}_cohorts.json"
        if os.path.lexists(role_audit_path):
            raise WorkflowError(
                f"{role} audit exists before its exact count commit"
            )
        expected_state_sha256 = current_state_sha256
    else:
        expected_state_sha256 = _common_role_authorization_state_sha256(role)
        recovery = _verify_role_count_update_recovery(
            role,
            state,
            pre_count_authorization_sha256=expected_state_sha256,
        )
        if (
            not isinstance(inherited_recovery, Mapping)
            or inherited_recovery.get("pre_count_state_sha256")
            != expected_state_sha256
            or inherited_recovery.get("count_update_event_seq")
            != recovery.get("count_update_event_seq")
            or inherited_recovery.get("count_update_event_sha256")
            != recovery.get("count_update_event_sha256")
            or inherited_recovery.get(
                "prior_controller_adapter_marker_sha256"
            )
            != recovery.get("prior_controller_adapter_marker_sha256")
            or inherited_recovery.get("role_audit_sha256")
            != recovery.get("existing_role_audit_sha256")
            or inherited_recovery.get("role_audit_created_unix_ns")
            != recovery.get("existing_role_audit_created_unix_ns")
        ):
            raise WorkflowError(
                f"{role} inherited and workflow count-recovery proofs differ"
            )
    regimes = {
        regime: _verify_role_regime(
            role,
            regime,
            expected_episodes=expected_per_dgp,
            expected_raw_seal=raw_seal_link,
            expected_execution_seal=execution_seal_link,
            expected_state_sha256=expected_state_sha256,
        )
        for regime in REGIMES
    }
    aggregate_paths = {
        regime: regimes[regime]["aggregate"]["path"] for regime in REGIMES
    }
    aggregate_hashes = {
        path: regimes[regime]["aggregate"]["sha256"]
        for regime, path in aggregate_paths.items()
    }
    result = {
        "role": role,
        "regime_order": list(REGIMES),
        "episodes_per_regime": expected_per_dgp,
        "episode_count": sum(item["episode_count"] for item in regimes.values()),
        "row_count": sum(item["row_count"] for item in regimes.values()),
        "authorization_seal": execution_seal_link,
        "raw_authorization_seal": raw_seal_link,
        "execution_authorization_seal": execution_seal_link,
        "authorization_state_sha256": expected_state_sha256,
        "role_count_update_recovery": recovery,
        "regimes": regimes,
        "aggregate_paths": aggregate_paths,
        "aggregate_hashes": dict(sorted(aggregate_hashes.items())),
        "development_part_arrays_opened": True,
        "aggregate_arrays_opened": False,
        "npz_arrays_opened_during_manifest_verification": True,
        "passed": True,
    }
    if result["episode_count"] != expected_total:
        raise WorkflowError(f"verified {role} total count drift")
    if (
        sha256_file(_regular_file(STATE_PATH)) != current_state_sha256
        or read_json(STATE_PATH) != dict(state)
    ):
        raise WorkflowError(f"controller state changed during {role} verification")
    return result


def role_inputs_complete(role: str) -> bool:
    if role == "fit":
        if not fit_inheritance.INVENTORY_PATH.is_file():
            return False
        paths = [
            fit_inheritance.SOURCE_FIT_ROOT / regime / name
            for regime in REGIMES
            for name in ("raw_manifest.json", "execution_manifest.json", "role.npz")
        ]
        return all(path.is_file() and not path.is_symlink() for path in paths)
    paths = [
        ATTEMPT_ROOT / "data" / role / regime / name
        for regime in REGIMES
        for name in ("raw_manifest.json", "execution_manifest.json")
    ]
    return all(path.is_file() and not path.is_symlink() for path in paths)


def _scientific_replay_manifest_binding(
    role: str, verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Select only stable, pre-count artifact identities from role verification."""

    regimes = verification.get("regimes")
    if not isinstance(regimes, Mapping) or set(regimes) != set(REGIMES):
        raise WorkflowError(f"{role} replay lacks four manifest-verification regimes")
    stable_regimes: dict[str, Any] = {}
    for regime in REGIMES:
        item = regimes[regime]
        if not isinstance(item, Mapping):
            raise WorkflowError(f"{role}/{regime} manifest verification is not an object")
        raw = item.get("raw_manifest")
        execution = item.get("execution_manifest")
        aggregate = item.get("aggregate")
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"path", "sha256"}
            or not isinstance(execution, Mapping)
            or set(execution) != {"path", "sha256"}
            or not isinstance(aggregate, Mapping)
            or set(aggregate) != {"path", "sha256", "bytes"}
            or type(item.get("episode_count")) is not int
            or type(item.get("row_count")) is not int
            or not isinstance(item.get("episode_ids_sha256"), str)
            or not isinstance(item.get("verified_file_index_sha256"), str)
        ):
            raise WorkflowError(
                f"{role}/{regime} stable manifest-verification schema drift"
            )
        stable_regimes[regime] = {
            "raw_manifest": dict(raw),
            "execution_manifest": dict(execution),
            "aggregate": dict(aggregate),
            "episode_count": item["episode_count"],
            "row_count": item["row_count"],
            "episode_ids_sha256": item["episode_ids_sha256"],
            "verified_file_index_sha256": item["verified_file_index_sha256"],
        }
    authorization = verification.get("authorization_state_sha256")
    if (
        not isinstance(authorization, str)
        or len(authorization) != 64
        or any(character not in "0123456789abcdef" for character in authorization)
    ):
        raise WorkflowError(f"{role} replay authorization-state hash drift")
    return {
        "role": role,
        "state": ROLE_STATES[role],
        "regime_order": list(REGIMES),
        "episodes_per_regime": verification.get("episodes_per_regime"),
        "episode_count": verification.get("episode_count"),
        "row_count": verification.get("row_count"),
        "authorization_state_sha256": authorization,
        "regimes": stable_regimes,
    }


def _run_scientific_replay_launcher(role: str, regime: str) -> dict[str, Any]:
    launcher = _regular_file(SCIENTIFIC_REPLAY_LAUNCHER_PATH)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            str(EVALUATION_PYTHON),
            "-B",
            str(launcher),
            "--role",
            role,
            "--regime",
            regime,
            "--inside-exact-runtime",
        ],
        cwd=ATTEMPT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkflowError(
            f"independent scientific replay failed for {role}/{regime}: "
            f"{completed.stderr.strip()}"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise WorkflowError(
            f"independent scientific replay emitted non-single JSON: {role}/{regime}"
        )
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise WorkflowError(
            f"independent scientific replay emitted invalid JSON: {role}/{regime}"
        ) from error
    if not isinstance(value, dict):
        raise WorkflowError("independent scientific replay result is not an object")
    return value


def _validate_scientific_replay_result(
    role: str,
    regime: str,
    result: Mapping[str, Any],
    stable: Mapping[str, Any],
) -> None:
    if set(result) != SCIENTIFIC_REPLAY_RESULT_KEYS:
        raise WorkflowError(f"scientific replay result schema drift: {role}/{regime}")
    if type(result.get("schema_version")) is not int:
        raise WorkflowError("scientific replay schema_version is not an exact integer")
    for field in ("episode_count", "row_count"):
        if type(result.get(field)) is not int:
            raise WorkflowError(
                f"scientific replay {field} is not an exact integer: {role}/{regime}"
            )
    for field in (
        "input_loader_agreement_exact",
        "every_persisted_tensor_exact",
        "target_loss_computed",
        "loss_or_effect_used_for_acceptance",
        "contact_or_privileged_materialized",
        "no_gradients",
        "artifact_tree_unchanged",
        "read_only",
        "passed",
    ):
        if type(result.get(field)) is not bool:
            raise WorkflowError(
                f"scientific replay {field} is not an exact boolean: {role}/{regime}"
            )
    expected_regime = stable["regimes"][regime]
    runtime = result.get("runtime_audit")
    runtime_mps = runtime.get("mps") if isinstance(runtime, Mapping) else None
    aggregate = result.get("aggregate")
    episodes = result.get("episodes")
    checks = {
        "schema": result.get("schema_version") == 1,
        "kind": result.get("artifact_type")
        == "v008_independent_scientific_replay_regime",
        "attempt": result.get("attempt") == ATTEMPT,
        "role": result.get("role") == role,
        "regime": result.get("regime") == regime,
        "raw": result.get("raw_manifest") == expected_regime["raw_manifest"],
        "execution": result.get("execution_manifest")
        == expected_regime["execution_manifest"],
        "episode_count": result.get("episode_count")
        == expected_regime["episode_count"],
        "row_count": result.get("row_count") == expected_regime["row_count"],
        "episodes": isinstance(episodes, list)
        and len(episodes) == expected_regime["episode_count"],
        "aggregate": isinstance(aggregate, Mapping)
        and set(aggregate) == {"applicable", "path", "sha256", "arrays", "exact"},
        "runtime": isinstance(runtime, Mapping)
        and runtime.get("passed") is True
        and runtime.get("requested_role") == "independent_verification",
        "mps": isinstance(runtime_mps, Mapping)
        and runtime_mps.get("required") is True
        and runtime_mps.get("built") is True
        and runtime_mps.get("available") is True,
        "sources": isinstance(result.get("source_hashes"), Mapping)
        and bool(result["source_hashes"]),
        "module": isinstance(result.get("module_before"), Mapping)
        and result.get("module_before") == result.get("module_after")
        and result["module_before"].get("passed") is True,
        "loader": result.get("input_loader_agreement_exact") is True,
        "tensors": result.get("every_persisted_tensor_exact") is True,
        "target_loss": result.get("target_loss_computed") is False,
        "effect_acceptance": result.get("loss_or_effect_used_for_acceptance")
        is False,
        "contact": result.get("contact_or_privileged_materialized") is False,
        "gradients": result.get("no_gradients") is True,
        "unchanged": result.get("artifact_tree_unchanged") is True,
        "read_only": result.get("read_only") is True,
        "passed": result.get("passed") is True,
    }
    if role in ("fit", "selection"):
        checks["compiled_gate"] = result.get("compiled_gate") is None
        checks["aggregate_link"] = (
            isinstance(aggregate, Mapping)
            and aggregate.get("applicable") is True
            and aggregate.get("path") == expected_regime["aggregate"]["path"]
            and aggregate.get("sha256") == expected_regime["aggregate"]["sha256"]
            and aggregate.get("exact") is True
            and result.get("aggregate_exact") is True
        )
    else:
        checks["compiled_gate"] = isinstance(result.get("compiled_gate"), Mapping)
        checks["aggregate_link"] = (
            isinstance(aggregate, Mapping)
            and aggregate
            == {
                "applicable": False,
                "path": None,
                "sha256": None,
                "arrays": None,
                "exact": None,
            }
            and result.get("aggregate_exact") is None
        )
    if isinstance(episodes, list):
        for index, episode in enumerate(episodes):
            if (
                not isinstance(episode, Mapping)
                or set(episode) != SCIENTIFIC_REPLAY_EPISODE_KEYS
                or type(episode.get("slot")) is not int
                or episode.get("slot") != index
                or not isinstance(episode.get("episode_id"), str)
                or not isinstance(episode.get("input_loader_audit_sha256"), str)
                or not isinstance(episode.get("array_sha256"), Mapping)
                or episode.get("every_persisted_tensor_exact") is not True
            ):
                checks[f"episode_{index}"] = False
                break
    if not all(checks.values()):
        raise WorkflowError(
            f"scientific replay result failed: {role}/{regime}: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )


def qualify_role_scientific_replay(
    role: str,
    state: Mapping[str, Any],
    verification: Mapping[str, Any],
    *,
    launcher: Any = _run_scientific_replay_launcher,
) -> dict[str, Any]:
    """Rerun and immutably bind all role episodes before count acceptance."""

    if role not in ROLE_STATES or state.get("current_state") != ROLE_STATES[role]:
        raise WorkflowError(f"scientific replay is outside the exact {role} role state")
    stable = _scientific_replay_manifest_binding(role, verification)
    manifest_verification_sha256 = canonical_sha256(stable)
    regime_records: dict[str, Any] = {}
    common_sources: dict[str, str] | None = None
    total_episodes = 0
    total_rows = 0
    for regime in REGIMES:
        result = _operation_result(
            launcher(role, regime),
            operation=f"independent scientific replay {role}/{regime}",
        )
        _validate_scientific_replay_result(role, regime, result, stable)
        path = SCIENTIFIC_REPLAY_AUDIT_ROOT / f"{role}_{regime}.json"
        write_or_verify_json(path, result)
        sources = {str(key): str(value) for key, value in result["source_hashes"].items()}
        if common_sources is None:
            common_sources = sources
        elif sources != common_sources:
            raise WorkflowError(f"scientific replay source hashes differ across {role} DGPs")
        link = {"path": relative_to_repo(path), "sha256": sha256_file(path)}
        regime_records[regime] = {
            "result": link,
            "raw_manifest": dict(result["raw_manifest"]),
            "execution_manifest": dict(result["execution_manifest"]),
            "aggregate": dict(result["aggregate"]),
            "episode_count": result["episode_count"],
            "row_count": result["row_count"],
        }
        total_episodes += result["episode_count"]
        total_rows += result["row_count"]
    role_audit = {
        "schema_version": 1,
        "artifact_type": "v008_independent_scientific_replay_role",
        "attempt": ATTEMPT,
        "role": role,
        "state": ROLE_STATES[role],
        "authorization_state_sha256": stable["authorization_state_sha256"],
        "manifest_verification_sha256": manifest_verification_sha256,
        "regime_order": list(REGIMES),
        "episodes_per_regime": verification["episodes_per_regime"],
        "episode_count": total_episodes,
        "row_count": total_rows,
        "regimes": regime_records,
        "source_hashes": common_sources,
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "all_development_aggregates_exact": True,
        "all_runtime_mps_exact": True,
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_trees_unchanged": True,
        "read_only": True,
        "passed": True,
    }
    if (
        total_episodes != verification.get("episode_count")
        or total_rows != verification.get("row_count")
    ):
        raise WorkflowError(f"{role} scientific replay count differs from manifest verification")
    role_path = ATTEMPT_ROOT / "audit" / f"{role}_scientific_replay.json"
    write_or_verify_json(role_path, role_audit)
    return {
        "artifact_type": "v008_independent_scientific_replay_qualification",
        "path": relative_to_repo(role_path),
        "sha256": sha256_file(role_path),
        "role": role,
        "state": ROLE_STATES[role],
        "authorization_state_sha256": stable["authorization_state_sha256"],
        "manifest_verification_sha256": manifest_verification_sha256,
        "episode_count": total_episodes,
        "row_count": total_rows,
        "passed": True,
    }


def _validate_role_replay_qualification(
    role: str, verification: Mapping[str, Any]
) -> dict[str, Any]:
    qualification = verification.get("scientific_replay_qualification")
    if (
        not isinstance(qualification, Mapping)
        or set(qualification) != SCIENTIFIC_REPLAY_QUALIFICATION_KEYS
    ):
        raise WorkflowError(f"{role} scientific replay qualification schema drift")
    stable = _scientific_replay_manifest_binding(role, verification)
    expected_digest = canonical_sha256(stable)
    path = _regular_file(resolve_repo_relative(qualification.get("path")))
    audit = read_json(path)
    if set(audit) != SCIENTIFIC_REPLAY_ROLE_AUDIT_KEYS:
        raise WorkflowError(f"{role} scientific replay role-audit schema drift")
    for field in ("schema_version", "episodes_per_regime", "episode_count", "row_count"):
        if type(audit.get(field)) is not int:
            raise WorkflowError(f"{role} replay role-audit {field} type drift")
    for field in (
        "input_loader_agreement_exact",
        "every_persisted_tensor_exact",
        "all_development_aggregates_exact",
        "all_runtime_mps_exact",
        "target_loss_computed",
        "loss_or_effect_used_for_acceptance",
        "contact_or_privileged_materialized",
        "no_gradients",
        "artifact_trees_unchanged",
        "read_only",
        "passed",
    ):
        if type(audit.get(field)) is not bool:
            raise WorkflowError(f"{role} replay role-audit {field} type drift")
    expected_qualification = {
        "artifact_type": "v008_independent_scientific_replay_qualification",
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
        "role": role,
        "state": ROLE_STATES[role],
        "authorization_state_sha256": stable["authorization_state_sha256"],
        "manifest_verification_sha256": expected_digest,
        "episode_count": verification.get("episode_count"),
        "row_count": verification.get("row_count"),
        "passed": True,
    }
    audit_checks = {
        "qualification": dict(qualification) == expected_qualification,
        "schema": audit.get("schema_version") == 1,
        "kind": audit.get("artifact_type")
        == "v008_independent_scientific_replay_role",
        "attempt": audit.get("attempt") == ATTEMPT,
        "role": audit.get("role") == role,
        "state": audit.get("state") == ROLE_STATES[role],
        "authorization": audit.get("authorization_state_sha256")
        == stable["authorization_state_sha256"],
        "manifest": audit.get("manifest_verification_sha256") == expected_digest,
        "regime_order": audit.get("regime_order") == list(REGIMES),
        "episodes_per_regime": audit.get("episodes_per_regime")
        == verification.get("episodes_per_regime"),
        "episode_count": audit.get("episode_count")
        == verification.get("episode_count"),
        "row_count": audit.get("row_count") == verification.get("row_count"),
        "regimes": isinstance(audit.get("regimes"), Mapping)
        and set(audit["regimes"]) == set(REGIMES),
        "sources": isinstance(audit.get("source_hashes"), Mapping)
        and bool(audit["source_hashes"]),
        "loader": audit.get("input_loader_agreement_exact") is True,
        "tensors": audit.get("every_persisted_tensor_exact") is True,
        "aggregates": audit.get("all_development_aggregates_exact") is True,
        "mps": audit.get("all_runtime_mps_exact") is True,
        "target_loss": audit.get("target_loss_computed") is False,
        "effect_acceptance": audit.get("loss_or_effect_used_for_acceptance") is False,
        "contact": audit.get("contact_or_privileged_materialized") is False,
        "gradients": audit.get("no_gradients") is True,
        "unchanged": audit.get("artifact_trees_unchanged") is True,
        "read_only": audit.get("read_only") is True,
        "passed": audit.get("passed") is True,
    }
    audit_regimes = audit.get("regimes")
    if isinstance(audit_regimes, Mapping) and set(audit_regimes) == set(REGIMES):
        for regime in REGIMES:
            item = audit_regimes[regime]
            expected_item = stable["regimes"][regime]
            if (
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "result",
                    "raw_manifest",
                    "execution_manifest",
                    "aggregate",
                    "episode_count",
                    "row_count",
                }
                or item.get("raw_manifest") != expected_item["raw_manifest"]
                or item.get("execution_manifest")
                != expected_item["execution_manifest"]
                or item.get("aggregate", {}).get("path")
                != expected_item["aggregate"]["path"]
                or item.get("aggregate", {}).get("sha256")
                != expected_item["aggregate"]["sha256"]
                or type(item.get("episode_count")) is not int
                or item.get("episode_count") != expected_item["episode_count"]
                or type(item.get("row_count")) is not int
                or item.get("row_count") != expected_item["row_count"]
                or not isinstance(item.get("result"), Mapping)
                or set(item["result"]) != {"path", "sha256"}
            ):
                audit_checks[f"regime_{regime}"] = False
                continue
            result_path = _regular_file(
                resolve_repo_relative(item["result"].get("path"))
            )
            result = read_json(result_path)
            try:
                _validate_scientific_replay_result(role, regime, result, stable)
            except WorkflowError:
                audit_checks[f"regime_{regime}"] = False
                continue
            audit_checks[f"regime_{regime}"] = (
                sha256_file(result_path) == item["result"].get("sha256")
                and result.get("source_hashes") == audit.get("source_hashes")
            )
    if not all(audit_checks.values()):
        raise WorkflowError(
            f"{role} scientific replay qualification failed: "
            f"{[name for name, passed in audit_checks.items() if not passed]}"
        )
    return dict(qualification)


def _path_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return dict(
        sorted((relative_to_repo(_regular_file(path)), sha256_file(path)) for path in paths)
    )


def _verify_checkpoint_json(path: Path, state: str) -> dict[str, Any]:
    value = read_json(_regular_file(path))
    if (
        value.get("schema_version") != 1
        or value.get("attempt") != ATTEMPT
        or value.get("checkpoint_state") != state
        or value.get("passed") is not True
    ):
        raise WorkflowError(f"invalid existing checkpoint artifact: {path}")
    return value


def _identity_enrichment() -> dict[str, str]:
    freeze = read_json(_regular_file(GATE_FREEZE_PATH))
    return {
        "attempt": ATTEMPT,
        "selected_candidate_id": str(freeze["selected_candidate_id"]),
        "selected_candidate_object_sha256": str(
            freeze["selected_candidate_object_sha256"]
        ),
        "fit_lock_sha256": sha256_file(FIT_LOCK_PATH),
        "selection_ledger_sha256": sha256_file(SELECTION_LEDGER_PATH),
        "gate_freeze_sha256": sha256_file(GATE_FREEZE_PATH),
    }


class DefaultOperations:
    """Scientific operations executed only by their state-specific handlers."""

    def implementation(self) -> dict[str, Any]:
        if preseal.IMPLEMENTATION_REPORT_PATH.exists():
            return _verify_checkpoint_json(
                preseal.IMPLEMENTATION_REPORT_PATH, "IMPLEMENTATION_COMPLETE"
            )
        return preseal.implementation_complete()

    def qualification(self) -> dict[str, Any]:
        if preseal.QUALIFICATION_REPORT_PATH.exists():
            return _verify_checkpoint_json(
                preseal.QUALIFICATION_REPORT_PATH, "PRESEAL_QUALIFICATION"
            )
        return preseal.preseal_qualification()

    def pre_data_seal(self) -> dict[str, Any]:
        if preseal.PRE_DATA_SEAL_PATH.exists():
            preseal.verify_seal(preseal.PRE_DATA_SEAL_PATH, "PRE_OUTCOME_SEAL")
            return _verify_checkpoint_json(
                preseal.PRE_DATA_SEAL_PATH, "PRE_OUTCOME_SEAL"
            )
        return preseal.seal_pre_data()

    def fit(self, role_verification: Mapping[str, Any]) -> dict[str, Any]:
        verify_here("fit", include_external_hashes=True)
        aggregate_paths = {
            regime: resolve_repo_relative(role_verification["aggregate_paths"][regime])
            for regime in REGIMES
        }
        expected_hashes = dict(role_verification["aggregate_hashes"])
        if _path_hashes(list(aggregate_paths.values())) != expected_hashes:
            raise WorkflowError("fit aggregates changed after manifest verification")
        if FITTED_PATH.exists() or FIT_LOCK_PATH.exists():
            if not (FITTED_PATH.is_file() and FIT_LOCK_PATH.is_file()):
                raise WorkflowError("partial immutable fit output")
            lock = fit_select.verify_fit_lock(FITTED_PATH, FIT_LOCK_PATH)
        else:
            whitening = fit_select.load_fixed_whitening(V5_FIXED_WHITENING_PATH)
            lock = fit_select.fit_and_seal(
                fit_select.load_per_dgp_npz(aggregate_paths),
                whitening,
                fitted_path=FITTED_PATH,
                lock_path=FIT_LOCK_PATH,
                fit_input_hashes=expected_hashes,
                fixed_whitening_sha256=fit_select.V5_FIXED_WHITENING_SHA256,
            )
        checks = {
            "input_hashes": lock.get("fit_input_hashes") == expected_hashes,
            "fixed_whitening": lock.get("fixed_whitening_sha256")
            == fit_select.V5_FIXED_WHITENING_SHA256
            == sha256_file(V5_FIXED_WHITENING_PATH),
            "fitted_hash": lock.get("fitted_candidates_sha256")
            == sha256_file(FITTED_PATH),
            "selection_unopened": lock.get("selection_input_opened_before_lock")
            is False,
            "no_refit": lock.get("selected_head_refit_permitted") is False,
            "candidate_count": lock.get("candidate_count") == 24,
        }
        if not all(checks.values()):
            raise WorkflowError(f"fit lock cross-link failed: {checks}")
        return lock

    def pre_selection_seal(self) -> dict[str, Any]:
        if preseal.PRE_SELECTION_SEAL_PATH.exists():
            preseal.verify_seal(
                preseal.PRE_SELECTION_SEAL_PATH, "PRE_SELECTION_SEAL"
            )
            return _verify_checkpoint_json(
                preseal.PRE_SELECTION_SEAL_PATH, "PRE_SELECTION_SEAL"
            )
        return preseal.seal_pre_selection()

    def select(self, role_verification: Mapping[str, Any]) -> dict[str, Any]:
        verify_here("selection", include_external_hashes=True)
        aggregate_paths = {
            regime: resolve_repo_relative(role_verification["aggregate_paths"][regime])
            for regime in REGIMES
        }
        expected_hashes = dict(role_verification["aggregate_hashes"])
        if _path_hashes(list(aggregate_paths.values())) != expected_hashes:
            raise WorkflowError("selection aggregates changed after manifest verification")
        cohort_ledger = read_json(_regular_file(COHORT_LEDGER_PATH))
        comparator_ids = fit_select.comparator_seeds_from_ledger(cohort_ledger)
        if SELECTION_LEDGER_PATH.exists():
            ledger = read_json(SELECTION_LEDGER_PATH)
        else:
            if any(
                path.exists()
                for path in (GATE_FIT_PATH, FIT_POWER_SUMMARY_PATH, SELECTION_POWER_SUMMARY_PATH)
            ):
                raise WorkflowError("selection outputs exist without their immutable ledger")
            ledger = fit_select.select_after_fit_lock(
                fitted_path=FITTED_PATH,
                fit_lock_path=FIT_LOCK_PATH,
                selection_loader=lambda: fit_select.load_per_dgp_npz(aggregate_paths),
                comparator_seeds=comparator_ids,
                selection_input_hashes=expected_hashes,
                selection_ledger_path=SELECTION_LEDGER_PATH,
                gate_fit_path=GATE_FIT_PATH,
                fit_power_summary_path=None,
                selection_power_summary_path=None,
            )
        selected = ledger.get("selected_candidate_id")
        selection_checks = {
            "status": ledger.get("status")
            == "selection_complete_no_selected_head_refit",
            "candidate_count": ledger.get("candidate_count") == 24,
            "no_refit": ledger.get("selected_head_refit_after_selection") is False,
            "runtime_equivalence": ledger.get(
                "all_24_selection_call_traces_reconstruct_exactly"
            )
            is True,
            "fit_lock": ledger.get("fit_lock_sha256") == sha256_file(FIT_LOCK_PATH),
            "fitted": ledger.get("fitted_candidates_sha256")
            == sha256_file(FITTED_PATH),
            "selection_inputs": ledger.get("selection_input_hashes")
            == expected_hashes,
            "comparator_ids": ledger.get("comparator_rng_ids") == comparator_ids,
            "confirmation_unused": ledger.get("prior_confirmation_outcome_episodes_used")
            == 0,
            "gate_presence": GATE_FIT_PATH.is_file() if selected else not GATE_FIT_PATH.exists(),
            "summary_deferred": not FIT_POWER_SUMMARY_PATH.exists()
            and not SELECTION_POWER_SUMMARY_PATH.exists(),
        }
        if not all(selection_checks.values()):
            raise WorkflowError(f"selection ledger cross-link failed: {selection_checks}")
        return ledger

    def freeze(self) -> dict[str, Any]:
        verify_here("selection", include_external_hashes=True)
        compiled_exists = COMPILED_GATE_PATH.exists()
        manifest_exists = COMPILER_MANIFEST_PATH.exists()
        if compiled_exists != manifest_exists:
            raise WorkflowError("partial immutable compiler output")
        if not compiled_exists:
            compile_gate.compile_gate_file(
                GATE_FIT_PATH, COMPILED_GATE_PATH, COMPILER_MANIFEST_PATH
            )
        if GATE_FREEZE_PATH.exists():
            freeze = read_json(GATE_FREEZE_PATH)
        else:
            freeze = fit_select.seal_gate_freeze(
                fitted_path=FITTED_PATH,
                fit_lock_path=FIT_LOCK_PATH,
                selection_ledger_path=SELECTION_LEDGER_PATH,
                gate_fit_path=GATE_FIT_PATH,
                compiled_gate_path=COMPILED_GATE_PATH,
                compiled_gate_manifest_path=COMPILER_MANIFEST_PATH,
                gate_freeze_path=GATE_FREEZE_PATH,
            )
        identity = _identity_enrichment()
        freeze_checks = {
            "status": freeze.get("status")
            == "selected_gate_frozen_before_smoke_or_confirmation",
            "candidate": freeze.get("selected_candidate_id")
            == identity["selected_candidate_id"],
            "object": freeze.get("selected_candidate_object_sha256")
            == identity["selected_candidate_object_sha256"],
            "fit_lock": freeze.get("fit_lock_sha256")
            == identity["fit_lock_sha256"],
            "selection": freeze.get("selection_ledger_sha256")
            == identity["selection_ledger_sha256"],
            "gate_fit": freeze.get("gate_fit_sha256") == sha256_file(GATE_FIT_PATH),
            "compiled": freeze.get("compiled_gate_sha256")
            == sha256_file(COMPILED_GATE_PATH),
            "compiler": freeze.get("compiled_gate_manifest_sha256")
            == sha256_file(COMPILER_MANIFEST_PATH),
            "semantic_compile": freeze.get(
                "compiled_runtime_arrays_bitwise_equal_selected_gate"
            )
            is True,
            "no_refit": freeze.get("selected_head_refit_after_selection") is False,
        }
        if not all(freeze_checks.values()):
            raise WorkflowError(f"gate freeze cross-link failed: {freeze_checks}")

        summaries_exist = FIT_POWER_SUMMARY_PATH.exists(), SELECTION_POWER_SUMMARY_PATH.exists()
        if summaries_exist[0] != summaries_exist[1]:
            raise WorkflowError("partial immutable power-summary output")
        if not summaries_exist[0]:
            fitted = fit_select.load_fitted_candidates(FITTED_PATH)
            ledger = read_json(SELECTION_LEDGER_PATH)
            fit_summary, selection_summary = fit_select.selected_power_summaries(
                fitted, ledger
            )
            fit_summary.update(identity)
            selection_summary.update(identity)
            _atomic_json_exclusive(FIT_POWER_SUMMARY_PATH, fit_summary)
            _atomic_json_exclusive(SELECTION_POWER_SUMMARY_PATH, selection_summary)
        for path, role in (
            (FIT_POWER_SUMMARY_PATH, "fit"),
            (SELECTION_POWER_SUMMARY_PATH, "selection"),
        ):
            summary = read_json(path)
            if summary.get("role") != role or any(
                summary.get(key) != value for key, value in identity.items()
            ):
                raise WorkflowError(f"power-summary identity drift: {path}")
        return freeze

    def power(self) -> dict[str, Any]:
        if POWER_PATH.exists():
            result = read_json(POWER_PATH)
        else:
            result = power_analysis.run_binding_power(
                fit_summary_path=FIT_POWER_SUMMARY_PATH,
                selection_summary_path=SELECTION_POWER_SUMMARY_PATH,
                fit_lock_path=FIT_LOCK_PATH,
                selection_ledger_path=SELECTION_LEDGER_PATH,
                gate_freeze_path=GATE_FREEZE_PATH,
                power_rule_path=POWER_RULE_PATH,
                output_path=POWER_PATH,
            )
            if not POWER_PATH.is_file():
                raise WorkflowError("identity-bound power API did not seal a result")
        return result

    def pre_confirmation_seal(self) -> dict[str, Any]:
        if preseal.PRE_CONFIRMATION_SEAL_PATH.exists():
            preseal.verify_seal(
                preseal.PRE_CONFIRMATION_SEAL_PATH,
                "PRE_CONFIRMATION_PACKAGE_SEAL",
            )
            return _verify_checkpoint_json(
                preseal.PRE_CONFIRMATION_SEAL_PATH,
                "PRE_CONFIRMATION_PACKAGE_SEAL",
            )
        return preseal.seal_pre_confirmation()


def _artifact_record(path: Path) -> dict[str, Any]:
    artifact = _regular_file(path)
    return {
        "path": relative_to_repo(artifact),
        "sha256": sha256_file(artifact),
        "bytes": int(artifact.stat().st_size),
    }


def _operation_result(value: Any, *, operation: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowError(f"{operation} did not return a JSON object")
    return dict(value)


def _checkpoint_object(
    checkpoint_state: str,
    controller_state: Mapping[str, Any],
    *,
    checks: Mapping[str, bool],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_checks = {str(key): bool(value) for key, value in checks.items()}
    if not normalized_checks or not all(normalized_checks.values()):
        failed = [key for key, value in normalized_checks.items() if not value]
        raise WorkflowError(
            f"{checkpoint_state} audit cannot pass: {failed or ['no_checks']}"
        )
    return {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "checkpoint_state": checkpoint_state,
        "created_unix_ns": time.time_ns(),
        "passed": True,
        "checks": normalized_checks,
        "controller_status_sha256": canonical_sha256(dict(controller_state)),
        "outcome_counts": {
            key: controller_state.get(key)
            for key in (
                "fit_outcome_episodes",
                "selection_outcome_episodes",
                "smoke_outcome_episodes",
                "confirmation_outcome_episodes_generated",
                "confirmation_outcome_episodes_executed",
                "confirmation_outcomes_opened_for_analysis",
            )
        },
        "evidence": dict(evidence),
        "scientific_objects_changed": False,
        "confirmation_outcomes_opened_by_workflow": 0,
    }


def _validate_operation_checkpoint(
    value: Mapping[str, Any], checkpoint_state: str
) -> None:
    checks = {
        "schema": value.get("schema_version") == 1,
        "attempt": value.get("attempt") == ATTEMPT,
        "state": value.get("checkpoint_state") == checkpoint_state,
        "passed": value.get("passed") is True,
    }
    if not all(checks.values()):
        raise WorkflowError(
            f"{checkpoint_state} operation checkpoint invalid: "
            f"{[key for key, passed in checks.items() if not passed]}"
        )


def _validate_power_identity(result: Mapping[str, Any]) -> dict[str, str]:
    identity = result.get("selected_gate_identity")
    inputs = result.get("inputs")
    if not isinstance(identity, Mapping) or not isinstance(inputs, Mapping):
        raise WorkflowError("power result lacks selected-gate identity inputs")
    paths = {
        "fit_lock": FIT_LOCK_PATH,
        "selection_ledger": SELECTION_LEDGER_PATH,
        "gate_freeze": GATE_FREEZE_PATH,
    }
    hash_fields = {
        "fit_lock": "fit_lock_sha256",
        "selection_ledger": "selection_ledger_sha256",
        "gate_freeze": "gate_freeze_sha256",
    }
    for name, path in paths.items():
        item = inputs.get(name)
        if not isinstance(item, Mapping):
            raise WorkflowError(f"power result lacks {name} input binding")
        observed = sha256_file(_regular_file(path))
        if (
            item.get("path") != relative_to_repo(path)
            or item.get("sha256") != observed
            or identity.get(hash_fields[name]) != observed
        ):
            raise WorkflowError(f"power result {name} identity drift")
    summaries = {
        "fit_summary": FIT_POWER_SUMMARY_PATH,
        "selection_summary": SELECTION_POWER_SUMMARY_PATH,
    }
    for name, path in summaries.items():
        item = inputs.get(name)
        if not isinstance(item, Mapping) or (
            item.get("path") != relative_to_repo(path)
            or item.get("sha256") != sha256_file(_regular_file(path))
        ):
            raise WorkflowError(f"power result {name} binding drift")
        summary = read_json(path)
        if any(
            summary.get(field) != identity.get(field)
            for field in (
                "selected_candidate_id",
                "selected_candidate_object_sha256",
                "fit_lock_sha256",
                "selection_ledger_sha256",
                "gate_freeze_sha256",
            )
        ):
            raise WorkflowError(f"power result {name} selected-gate drift")
    if identity.get("selected_candidate_id") != read_json(
        GATE_FREEZE_PATH
    ).get("selected_candidate_id"):
        raise WorkflowError("power result selected candidate differs from gate freeze")
    return {str(key): str(value) for key, value in identity.items()}


class Workflow:
    """Advance at most one verified state at a time, or resume to a boundary."""

    def __init__(
        self,
        controller: Controller | None = None,
        operations: Any | None = None,
        *,
        evidence_paths: Mapping[str, Path] | None = None,
        manifest_verifier: Any = verify_role_manifests,
        replay_qualifier: Any | None = None,
        inputs_complete: Any = role_inputs_complete,
    ) -> None:
        self.controller = controller if controller is not None else _load_controller()
        self.operations = operations if operations is not None else DefaultOperations()
        self.evidence_paths = dict(DEFAULT_EVIDENCE_PATHS)
        if evidence_paths is not None:
            self.evidence_paths.update(
                {str(key): Path(value) for key, value in evidence_paths.items()}
            )
        self.manifest_verifier = manifest_verifier
        self.replay_qualifier = (
            qualify_role_scientific_replay
            if replay_qualifier is None
            else replay_qualifier
        )
        self.inputs_complete = inputs_complete

    def status(self) -> dict[str, Any]:
        state = _operation_result(self.controller.status(), operation="controller status")
        if state.get("active_attempt") != ATTEMPT:
            raise WorkflowError("workflow is not authorized for the active attempt")
        if not isinstance(state.get("current_state"), str):
            raise WorkflowError("controller status lacks a current state")
        return state

    def _require_state(self, expected: str) -> dict[str, Any]:
        state = self.status()
        if state.get("current_state") != expected:
            raise WorkflowError(
                f"state changed during workflow step: expected={expected} "
                f"observed={state.get('current_state')}"
            )
        return state

    def _advance(
        self,
        state_name: str,
        evidence_path: Path,
        *,
        checkpoint_name: str,
        next_action: str,
    ) -> dict[str, Any]:
        self._require_state(state_name)
        result = _operation_result(
            self.controller.advance(
                state_name, evidence_path, checkpoint_name, next_action
            ),
            operation=f"advance {state_name}",
        )
        if state_name not in result.get("completed_states", []):
            raise WorkflowError(f"controller did not record {state_name} exactly once")
        if result.get("current_state") == state_name:
            raise WorkflowError(f"controller did not advance from {state_name}")
        return result

    def _seal_operation_checkpoint(
        self,
        state_name: str,
        result: Mapping[str, Any],
        *,
        checkpoint_name: str,
        next_action: str,
    ) -> dict[str, Any]:
        _validate_operation_checkpoint(result, state_name)
        path = self.evidence_paths[state_name]
        write_or_verify_json(path, result)
        return self._advance(
            state_name,
            path,
            checkpoint_name=checkpoint_name,
            next_action=next_action,
        )

    def _ensure_role_count(
        self,
        role: str,
        verification: Mapping[str, Any],
    ) -> dict[str, Any]:
        state_name = ROLE_STATES[role]
        state = self._require_state(state_name)
        counter = (
            "fit_outcome_episodes"
            if role == "fit"
            else "selection_outcome_episodes"
        )
        expected_key = (
            "expected_fit_episode_count"
            if role == "fit"
            else "expected_selection_episode_count"
        )
        expected_raw = state.get(expected_key)
        if type(expected_raw) is not int:
            raise WorkflowError(f"{role} expected count has invalid type")
        expected = expected_raw
        if (
            verification.get("passed") is not True
            or verification.get("role") != role
            or verification.get("episode_count") != expected
            or verification.get("episodes_per_regime")
            != ROLE_EPISODES_PER_DGP[role]
            or verification.get("regime_order") != list(REGIMES)
            or verification.get("npz_arrays_opened_during_manifest_verification")
            is not True
            or verification.get("development_part_arrays_opened") is not True
            or verification.get("aggregate_arrays_opened") is not False
        ):
            raise WorkflowError(f"{role} verification cannot authorize a counter")
        _validate_role_replay_qualification(role, verification)
        current_raw = state.get(counter)
        if type(current_raw) is not int:
            raise WorkflowError(f"{role} controller counter has invalid type")
        current = current_raw
        if current == 0:
            if verification.get("role_count_update_recovery") is not None:
                raise WorkflowError(
                    f"{role} pre-count verification carried a recovery proof"
                )
            self.controller.update_counts(**{counter: expected})
        elif current == expected:
            recovery = verification.get("role_count_update_recovery")
            published_audit = (
                isinstance(recovery, Mapping)
                and recovery.get("role_audit_already_published") is True
                and isinstance(recovery.get("existing_role_audit_sha256"), str)
                and type(recovery.get("existing_role_audit_created_unix_ns"))
                is int
                and recovery.get("existing_role_audit_verification_kind")
                in {"pre_count_verification", "count_recovery_verification"}
            )
            if not (
                isinstance(recovery, Mapping)
                and recovery.get("passed") is True
                and recovery.get("role") == role
                and recovery.get("state") == state_name
                and recovery.get("counter_field") == counter
                and recovery.get("counter_value") == expected
                and recovery.get("count_update_event_fields")
                == {counter: expected}
                and recovery.get("pre_count_authorization_sha256")
                == verification.get("authorization_state_sha256")
                and recovery.get("current_state_sha256")
                == sha256_file(_regular_file(STATE_PATH))
                and recovery.get("no_role_checkpoint_before_recovery") is True
                and (
                    recovery.get(
                        "no_role_audit_or_checkpoint_before_recovery"
                    )
                    is True
                    and recovery.get("role_audit_already_published") is False
                    and recovery.get("existing_role_audit_sha256") is None
                    and recovery.get("existing_role_audit_created_unix_ns")
                    is None
                    and recovery.get("existing_role_audit_verification_kind")
                    is None
                    or published_audit
                )
                and recovery.get("all_other_later_counters_unchanged") is True
                and recovery.get("exactly_one_controller_event_after_authorized_state")
                is True
            ):
                raise WorkflowError(
                    f"{role} exact counter lacks authenticated crash recovery"
                )
        else:
            raise WorkflowError(
                f"{role} controller counter is neither zero nor exact: {current}"
            )
        updated = self._require_state(state_name)
        if updated.get(counter) != expected:
            raise WorkflowError(f"{role} controller counter did not reach exact total")
        return updated

    def _role_checkpoint(self, role: str) -> dict[str, Any]:
        state_name = ROLE_STATES[role]
        state = self._require_state(state_name)
        if not self.inputs_complete(role):
            return {
                "action": "waiting_for_role_manifests",
                "role": role,
                "state": state_name,
                "advanced": False,
            }
        verification = _operation_result(
            self.manifest_verifier(role, state),
            operation=f"verify {role} manifests",
        )
        replay_qualification = _operation_result(
            self.replay_qualifier(role, state, verification),
            operation=f"qualify {role} independent scientific replay",
        )
        verification = dict(verification)
        verification["scientific_replay_qualification"] = replay_qualification
        _validate_role_replay_qualification(role, verification)
        updated = self._ensure_role_count(role, verification)
        counter = (
            "fit_outcome_episodes"
            if role == "fit"
            else "selection_outcome_episodes"
        )
        expected = ROLE_EPISODES_PER_DGP[role] * len(REGIMES)
        recovery = verification.get("role_count_update_recovery")
        audit_already_published = (
            isinstance(recovery, Mapping)
            and recovery.get("role_audit_already_published") is True
        )
        audit_verification = dict(verification)
        if audit_already_published:
            verification_kind = recovery.get(
                "existing_role_audit_verification_kind"
            )
            if verification_kind == "pre_count_verification":
                # The ordinary pre-count verification preceded the count event.
                prior_recovery: dict[str, Any] | None = None
            elif verification_kind == "count_recovery_verification":
                # A prior recovery observed the count event before publishing
                # the immutable audit.  Reconstruct that exact no-audit proof.
                prior_recovery = dict(recovery)
                prior_recovery.update(
                    {
                        "no_role_audit_or_checkpoint_before_recovery": True,
                        "role_audit_already_published": False,
                        "existing_role_audit_sha256": None,
                        "existing_role_audit_created_unix_ns": None,
                        "existing_role_audit_verification_kind": None,
                    }
                )
            else:
                raise WorkflowError(
                    f"{role} published audit has unknown verification lineage"
                )
            audit_verification["role_count_update_recovery"] = prior_recovery
        audit = _checkpoint_object(
            state_name,
            updated,
            checks={
                "all_four_regimes_verified": set(
                    verification.get("regimes", {})
                )
                == set(REGIMES),
                "exact_episode_count": verification.get("episode_count")
                == expected,
                "exact_row_count": verification.get("row_count")
                == expected * ROWS_PER_EPISODE,
                "controller_count_exact": updated.get(counter) == expected,
                "aggregate_arrays_unopened": verification.get(
                    "aggregate_arrays_opened"
                )
                is False,
                "scientific_replay_qualified": verification.get(
                    "scientific_replay_qualification", {}
                ).get("passed")
                is True,
                "confirmation_unopened": updated.get(
                    "confirmation_outcomes_opened_for_analysis"
                )
                is False,
            },
            evidence={"role_verification": audit_verification},
        )
        path = self.evidence_paths[state_name]
        if audit_already_published:
            expected_hash = recovery.get("existing_role_audit_sha256")
            existing_timestamp = recovery.get(
                "existing_role_audit_created_unix_ns"
            )
            if type(existing_timestamp) is not int:
                raise WorkflowError(
                    f"{role} published audit lacks its exact timestamp"
                )
            audit["created_unix_ns"] = existing_timestamp
            existing_path = _regular_file(path)
            persisted_audit = read_json(existing_path)
            canonical_existing = (
                json.dumps(persisted_audit, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            canonical_intended = (
                json.dumps(audit, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            if (
                not isinstance(expected_hash, str)
                or existing_path.read_bytes() != canonical_existing
                or canonical_existing != canonical_intended
                or persisted_audit != audit
                or sha256_file(existing_path) != expected_hash
            ):
                raise WorkflowError(
                    f"{role} published-audit crash recovery byte drift"
                )
        else:
            persisted_audit = write_or_verify_json(path, audit)
        next_action = (
            "fit and seal all bounded candidates from verified fit aggregates"
            if role == "fit"
            else "select once from the verified selection aggregates without refit"
        )
        after = self._advance(
            state_name,
            path,
            checkpoint_name=f"{ATTEMPT}_{role}_cohorts_verified",
            next_action=next_action,
        )
        return {
            "action": "role_checkpoint_advanced",
            "role": role,
            "state": state_name,
            "advanced": True,
            "current_state": after["current_state"],
        }

    def _prior_role_verification(self, state_name: str) -> dict[str, Any]:
        audit = _verify_checkpoint_json(self.evidence_paths[state_name], state_name)
        evidence = audit.get("evidence")
        if not isinstance(evidence, Mapping):
            raise WorkflowError(f"{state_name} audit lacks evidence")
        verification = evidence.get("role_verification")
        if not isinstance(verification, Mapping) or verification.get("passed") is not True:
            raise WorkflowError(f"{state_name} audit lacks verified role manifests")
        return dict(verification)

    def _fit_lock(self) -> dict[str, Any]:
        state = self._require_state("FIT_LOCK")
        verification = self._prior_role_verification("FIT_COHORTS")
        lock = _operation_result(
            self.operations.fit(verification), operation="fit and seal"
        )
        checks = {
            "status": lock.get("status")
            == "all_24_candidates_fit_compiled_and_locked_before_selection_open",
            "candidate_count": lock.get("candidate_count") == 24,
            "fit_lock_present": FIT_LOCK_PATH.is_file(),
            "fitted_candidates_present": FITTED_PATH.is_file(),
            "fitted_hash": FITTED_PATH.is_file()
            and lock.get("fitted_candidates_sha256") == sha256_file(FITTED_PATH),
            "selection_unopened": lock.get("selection_input_opened_before_lock")
            is False,
            "refit_forbidden": lock.get("selected_head_refit_permitted") is False,
            "input_hashes": lock.get("fit_input_hashes")
            == verification.get("aggregate_hashes"),
        }
        audit = _checkpoint_object(
            "FIT_LOCK",
            state,
            checks=checks,
            evidence={
                "fit_lock": _artifact_record(FIT_LOCK_PATH),
                "fitted_candidates": _artifact_record(FITTED_PATH),
                "fit_cohorts_audit": _artifact_record(
                    self.evidence_paths["FIT_COHORTS"]
                ),
            },
        )
        path = self.evidence_paths["FIT_LOCK"]
        write_or_verify_json(path, audit)
        after = self._advance(
            "FIT_LOCK",
            path,
            checkpoint_name=f"{ATTEMPT}_fit_lock",
            next_action="seal the fixed fit lock before selection outcomes",
        )
        return {
            "action": "fit_lock_advanced",
            "advanced": True,
            "current_state": after["current_state"],
        }

    def _candidate_selection(self) -> dict[str, Any]:
        state = self._require_state("CANDIDATE_SELECTION")
        verification = self._prior_role_verification("SELECTION_COHORTS")
        ledger = _operation_result(
            self.operations.select(verification), operation="candidate selection"
        )
        if ledger.get("selected_candidate_id") is None:
            checks = {
                "status": ledger.get("status")
                == "selection_complete_no_selected_head_refit",
                "candidate_count": ledger.get("candidate_count") == 24,
                "eligible_count_zero": ledger.get("eligible_count") == 0,
                "selected_index_null": ledger.get("selected_candidate_index") is None,
                "no_refit": ledger.get("selected_head_refit_after_selection")
                is False,
                "confirmation_unused": ledger.get(
                    "prior_confirmation_outcome_episodes_used"
                )
                == 0,
            }
            if not all(checks.values()):
                raise WorkflowError(f"invalid no-candidate result: {checks}")
            staged = _operation_result(
                self.controller.stage_early_scientific_failure("no_candidate"),
                operation="stage no-candidate terminal branch",
            )
            return {
                "action": "early_scientific_failure_staged",
                "mode": "no_candidate",
                "advanced": True,
                "current_state": staged.get("current_state"),
            }
        selected_id = ledger.get("selected_candidate_id")
        checks = {
            "status": ledger.get("status")
            == "selection_complete_no_selected_head_refit",
            "candidate_count": ledger.get("candidate_count") == 24,
            "eligible": isinstance(ledger.get("eligible_count"), int)
            and int(ledger["eligible_count"]) > 0,
            "selected_id": isinstance(selected_id, str) and bool(selected_id),
            "selected_index": isinstance(ledger.get("selected_candidate_index"), int),
            "no_refit": ledger.get("selected_head_refit_after_selection") is False,
            "confirmation_unused": ledger.get(
                "prior_confirmation_outcome_episodes_used"
            )
            == 0,
            "selection_inputs": ledger.get("selection_input_hashes")
            == verification.get("aggregate_hashes"),
            "ledger_present": SELECTION_LEDGER_PATH.is_file(),
            "gate_fit_present": GATE_FIT_PATH.is_file(),
        }
        audit = _checkpoint_object(
            "CANDIDATE_SELECTION",
            state,
            checks=checks,
            evidence={
                "selection_ledger": _artifact_record(SELECTION_LEDGER_PATH),
                "gate_fit": _artifact_record(GATE_FIT_PATH),
                "selected_candidate_id": selected_id,
                "selection_cohorts_audit": _artifact_record(
                    self.evidence_paths["SELECTION_COHORTS"]
                ),
            },
        )
        path = self.evidence_paths["CANDIDATE_SELECTION"]
        write_or_verify_json(path, audit)
        after = self._advance(
            "CANDIDATE_SELECTION",
            path,
            checkpoint_name=f"{ATTEMPT}_candidate_selection",
            next_action="compile the selected fit-only gate and freeze semantic identity",
        )
        return {
            "action": "candidate_selection_advanced",
            "advanced": True,
            "current_state": after["current_state"],
        }

    def _gate_freeze(self) -> dict[str, Any]:
        state = self._require_state("GATE_FREEZE")
        freeze = _operation_result(self.operations.freeze(), operation="gate freeze")
        identity = _identity_enrichment()
        checks = {
            "status": freeze.get("status")
            == "selected_gate_frozen_before_smoke_or_confirmation",
            "candidate": freeze.get("selected_candidate_id")
            == identity["selected_candidate_id"],
            "candidate_object": freeze.get("selected_candidate_object_sha256")
            == identity["selected_candidate_object_sha256"],
            "fit_lock": freeze.get("fit_lock_sha256")
            == identity["fit_lock_sha256"],
            "selection_ledger": freeze.get("selection_ledger_sha256")
            == identity["selection_ledger_sha256"],
            "compiled_semantics": freeze.get(
                "compiled_runtime_arrays_bitwise_equal_selected_gate"
            )
            is True,
            "no_refit": freeze.get("selected_head_refit_after_selection") is False,
            "confirmation_zero": freeze.get("confirmation_episodes_at_freeze") == 0,
            "fit_summary_postfreeze": FIT_POWER_SUMMARY_PATH.is_file(),
            "selection_summary_postfreeze": SELECTION_POWER_SUMMARY_PATH.is_file(),
        }
        audit = _checkpoint_object(
            "GATE_FREEZE",
            state,
            checks=checks,
            evidence={
                "gate_freeze": _artifact_record(GATE_FREEZE_PATH),
                "compiled_gate": _artifact_record(COMPILED_GATE_PATH),
                "compiler_manifest": _artifact_record(COMPILER_MANIFEST_PATH),
                "fit_power_summary": _artifact_record(FIT_POWER_SUMMARY_PATH),
                "selection_power_summary": _artifact_record(
                    SELECTION_POWER_SUMMARY_PATH
                ),
                "selected_gate_identity": identity,
            },
        )
        path = self.evidence_paths["GATE_FREEZE"]
        write_or_verify_json(path, audit)
        after = self._advance(
            "GATE_FREEZE",
            path,
            checkpoint_name=f"{ATTEMPT}_gate_freeze",
            next_action="run the identity-bound frozen confirmation power rule once",
        )
        return {
            "action": "gate_freeze_advanced",
            "advanced": True,
            "current_state": after["current_state"],
        }

    def _power(self) -> dict[str, Any]:
        state = self._require_state("CONFIRMATION_POWER_AND_COHORT_FREEZE")
        result = _operation_result(self.operations.power(), operation="binding power")
        try:
            persisted = read_json(_regular_file(POWER_PATH))
            if canonical_sha256(persisted) != canonical_sha256(result):
                raise WorkflowError(
                    "binding-power operation result differs from its immutable artifact"
                )

            fitted_path = _regular_file(FITTED_PATH)
            fitted_sha256 = sha256_file(fitted_path)
            fit_lock = read_json(_regular_file(FIT_LOCK_PATH))
            selection_ledger = read_json(_regular_file(SELECTION_LEDGER_PATH))
            if (
                fit_lock.get("fitted_candidates_sha256") != fitted_sha256
                or selection_ledger.get("fitted_candidates_sha256")
                != fitted_sha256
            ):
                raise WorkflowError(
                    "binding-power source does not match the fit/selection locks"
                )
            fitted = fit_select.load_fitted_candidates(fitted_path)
            expected_fit_summary, expected_selection_summary = (
                fit_select.selected_power_summaries(fitted, selection_ledger)
            )
            expected_identity = _identity_enrichment()
            expected_fit_summary.update(expected_identity)
            expected_selection_summary.update(expected_identity)
            observed_fit_summary = read_json(_regular_file(FIT_POWER_SUMMARY_PATH))
            observed_selection_summary = read_json(
                _regular_file(SELECTION_POWER_SUMMARY_PATH)
            )
            summary_checks = {
                "fit": canonical_sha256(observed_fit_summary)
                == canonical_sha256(expected_fit_summary),
                "selection": canonical_sha256(observed_selection_summary)
                == canonical_sha256(expected_selection_summary),
            }
            if not all(summary_checks.values()):
                raise WorkflowError(
                    "stored power summaries differ from exact selected-gate moments: "
                    f"{summary_checks}"
                )

            frozen_moments = power_analysis.load_frozen_power_rule_moments(
                _regular_file(POWER_RULE_PATH)
            )
            expected_power = power_analysis.compute_binding_power(
                expected_fit_summary,
                expected_selection_summary,
                frozen_moments,
            )
            identity_artifacts = power_analysis.validate_identity_artifacts(
                expected_power["selected_gate_identity"],
                fit_lock_path=FIT_LOCK_PATH,
                selection_ledger_path=SELECTION_LEDGER_PATH,
                gate_freeze_path=GATE_FREEZE_PATH,
            )
            expected_power["inputs"] = {
                "fit_summary": {
                    "path": relative_to_repo(FIT_POWER_SUMMARY_PATH),
                    "sha256": sha256_file(FIT_POWER_SUMMARY_PATH),
                },
                "selection_summary": {
                    "path": relative_to_repo(SELECTION_POWER_SUMMARY_PATH),
                    "sha256": sha256_file(SELECTION_POWER_SUMMARY_PATH),
                },
                "power_rule": {
                    "path": relative_to_repo(POWER_RULE_PATH),
                    "sha256": sha256_file(POWER_RULE_PATH),
                    "expected_sha256": power_analysis.POWER_RULE_SHA256,
                    "endpoint_moments_object_sha256": (
                        power_analysis.POWER_RULE_MOMENTS_SHA256
                    ),
                },
                **identity_artifacts,
            }
            gate_freeze = read_json(_regular_file(GATE_FREEZE_PATH))
            created_unix_ns = result.get("created_unix_ns")
            if (
                type(created_unix_ns) is not int
                or type(gate_freeze.get("created_unix_ns")) is not int
                or created_unix_ns <= int(gate_freeze["created_unix_ns"])
            ):
                raise WorkflowError(
                    "binding-power chronology is not strictly after gate freeze"
                )
            expected_power["created_unix_ns"] = created_unix_ns
            if canonical_sha256(result) != canonical_sha256(expected_power):
                raise WorkflowError(
                    "binding-power artifact is not the exact prospective recomputation"
                )
        except WorkflowError:
            raise
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
            raise WorkflowError(
                "binding-power prospective recomputation failed closed"
            ) from error
        identity = _validate_power_identity(result)
        common_checks = {
            "schema": result.get("schema_version") == 1,
            "attempt": result.get("attempt") == ATTEMPT,
            "status": result.get("status")
            == "binding_post_selection_power_result",
            "binding": result.get("binding") is True,
            "identity": bool(identity.get("selected_candidate_id")),
            "no_sequential_expansion": result.get("no_sequential_expansion") is True,
            "confirmation_unopened": result.get("fresh_confirmation_outcomes_opened")
            == 0,
        }
        if not all(common_checks.values()):
            raise WorkflowError(f"power common contract failed: {common_checks}")
        if result.get("feasible") is False:
            failure_checks = {
                "passed_false": result.get("passed") is False,
                "decision": result.get("decision")
                == "power_infeasible_no_confirmation",
                "selected_n_null": result.get(
                    "selected_confirmation_episodes_per_regime"
                )
                is None,
                "counts_null": result.get("confirmation_episode_count_per_regime")
                is None,
                "generation_unauthorized": result.get(
                    "confirmation_generation_authorized_by_power"
                )
                is False,
                "terminal_label": result.get("terminal_label_if_infeasible")
                == "domain_robust_gate_failed",
            }
            if not all(failure_checks.values()):
                raise WorkflowError(f"invalid power-infeasible result: {failure_checks}")
            staged = _operation_result(
                self.controller.stage_early_scientific_failure("power_infeasible"),
                operation="stage power-infeasible terminal branch",
            )
            return {
                "action": "early_scientific_failure_staged",
                "mode": "power_infeasible",
                "advanced": True,
                "current_state": staged.get("current_state"),
            }
        n = result.get("selected_confirmation_episodes_per_regime")
        counts = result.get("confirmation_episode_count_per_regime")
        feasible_checks = {
            "feasible": result.get("feasible") is True,
            "passed": result.get("passed") is True,
            "decision": result.get("decision") == "confirmation_size_fixed",
            "n_on_grid": isinstance(n, int) and n in range(500, 4501, 500),
            "same_n": isinstance(counts, Mapping)
            and set(counts) == set(REGIMES)
            and all(value == n for value in counts.values()),
            "generation_authorized": result.get(
                "confirmation_generation_authorized_by_power"
            )
            is True,
            "prefix_only": result.get("fixed_prefix_only") is True,
        }
        if not all(feasible_checks.values()):
            raise WorkflowError(f"invalid feasible power result: {feasible_checks}")
        assert isinstance(n, int)
        audit = _checkpoint_object(
            "CONFIRMATION_POWER_AND_COHORT_FREEZE",
            state,
            checks={**common_checks, **feasible_checks},
            evidence={
                "power_analysis": _artifact_record(POWER_PATH),
                "selected_gate_identity": identity,
                "confirmation_episode_count_per_regime": dict(counts),
            },
        )
        audit["fixed_confirmation_episode_count"] = len(REGIMES) * n
        audit["fixed_confirmation_episodes_per_regime"] = n
        path = self.evidence_paths["CONFIRMATION_POWER_AND_COHORT_FREEZE"]
        write_or_verify_json(path, audit)
        after = self._advance(
            "CONFIRMATION_POWER_AND_COHORT_FREEZE",
            path,
            checkpoint_name=f"{ATTEMPT}_confirmation_power_and_cohort_freeze",
            next_action="seal the complete fixed confirmation package before smoke",
        )
        return {
            "action": "confirmation_power_advanced",
            "advanced": True,
            "fixed_confirmation_episodes_per_regime": n,
            "current_state": after["current_state"],
        }

    def _sealed_analysis_boundary(self) -> dict[str, Any]:
        self._require_state("SEALED_ANALYSIS")
        if not ANALYSIS_INVALID_PATH.exists():
            return {
                "action": "run_sealed_analysis",
                "advanced": False,
                "state": "SEALED_ANALYSIS",
                "analysis_result_present": ANALYSIS_RESULT_PATH.is_file(),
            }
        invalid = read_json(_regular_file(ANALYSIS_INVALID_PATH))
        checks = {
            "attempt": invalid.get("attempt") == ATTEMPT,
            "state": invalid.get("checkpoint_state") == "SEALED_ANALYSIS",
            "passed_false": invalid.get("passed") is False,
            "process_invalid": invalid.get("process_valid") is False,
            "integrity_failed": invalid.get("integrity_passed") is False,
            "execution_invalid": invalid.get("execution_invalid") is True,
            "opened": invalid.get("confirmation_outcomes_opened_for_analysis")
            is True,
            "result_presence": invalid.get("analysis_result_present")
            is ANALYSIS_RESULT_PATH.is_file(),
        }
        if not all(checks.values()):
            raise WorkflowError(f"analysis invalidity artifact failed: {checks}")
        staged = _operation_result(
            self.controller.stage_postconfirmation_integrity_failure(),
            operation="stage analysis execution-invalid branch",
        )
        return {
            "action": "postconfirmation_execution_invalid_staged",
            "advanced": True,
            "current_state": staged.get("current_state"),
            "source": _artifact_record(ANALYSIS_INVALID_PATH),
        }

    def step(self) -> dict[str, Any]:
        state = self.status()
        current = state["current_state"]
        if current == "IMPLEMENTATION_COMPLETE":
            result = _operation_result(
                self.operations.implementation(), operation="implementation audit"
            )
            after = self._seal_operation_checkpoint(
                current,
                result,
                checkpoint_name=f"{ATTEMPT}_implementation_complete",
                next_action="run preseal qualification without opening outcomes",
            )
        elif current == "PRESEAL_QUALIFICATION":
            result = _operation_result(
                self.operations.qualification(), operation="preseal qualification"
            )
            after = self._seal_operation_checkpoint(
                current,
                result,
                checkpoint_name=f"{ATTEMPT}_preseal_qualification",
                next_action="seal every pre-outcome source and verifier contract",
            )
        elif current == "PRE_OUTCOME_SEAL":
            result = _operation_result(
                self.operations.pre_data_seal(), operation="pre-outcome seal"
            )
            after = self._seal_operation_checkpoint(
                current,
                result,
                checkpoint_name=f"{ATTEMPT}_pre_outcome_seal",
                next_action="materialize the fixed fit role only",
            )
        elif current == "FIT_COHORTS":
            return self._role_checkpoint("fit")
        elif current == "FIT_LOCK":
            return self._fit_lock()
        elif current == "PRE_SELECTION_SEAL":
            result = _operation_result(
                self.operations.pre_selection_seal(),
                operation="pre-selection seal",
            )
            after = self._seal_operation_checkpoint(
                current,
                result,
                checkpoint_name=f"{ATTEMPT}_pre_selection_seal",
                next_action="materialize the disjoint fixed selection role only",
            )
        elif current == "SELECTION_COHORTS":
            return self._role_checkpoint("selection")
        elif current == "CANDIDATE_SELECTION":
            return self._candidate_selection()
        elif current == "GATE_FREEZE":
            return self._gate_freeze()
        elif current == "CONFIRMATION_POWER_AND_COHORT_FREEZE":
            return self._power()
        elif current == "PRE_CONFIRMATION_PACKAGE_SEAL":
            result = _operation_result(
                self.operations.pre_confirmation_seal(),
                operation="pre-confirmation package seal",
            )
            after = self._seal_operation_checkpoint(
                current,
                result,
                checkpoint_name=f"{ATTEMPT}_pre_confirmation_package_seal",
                next_action="run the permanently excluded mechanical smoke once",
            )
        elif current == "SEALED_ANALYSIS":
            return self._sealed_analysis_boundary()
        else:
            return {
                "action": "external_state_boundary",
                "state": current,
                "advanced": False,
                "next_action": state.get("next_action"),
            }
        return {
            "action": "checkpoint_advanced",
            "state": current,
            "advanced": True,
            "current_state": after.get("current_state"),
        }

    def resume(self, *, max_transitions: int = 32) -> dict[str, Any]:
        if max_transitions < 1:
            raise WorkflowError("max transitions must be positive")
        events: list[dict[str, Any]] = []
        for _ in range(max_transitions):
            before = self.status()["current_state"]
            event = self.step()
            events.append(event)
            after = self.status()["current_state"]
            if not event.get("advanced") or after == before:
                break
            if event.get("mode") in {"no_candidate", "power_infeasible"}:
                break
            if event.get("action") == "postconfirmation_execution_invalid_staged":
                break
        return {
            "attempt": ATTEMPT,
            "transition_count": sum(bool(event.get("advanced")) for event in events),
            "events": events,
            "status": self.status(),
        }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("step")
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--max-transitions", type=int, default=32)
    verify_parser = subparsers.add_parser("verify-role")
    verify_parser.add_argument("role", choices=("fit", "selection"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    workflow = Workflow()
    if arguments.command == "status":
        result = workflow.status()
    elif arguments.command == "step":
        result = workflow.step()
    elif arguments.command == "resume":
        result = workflow.resume(max_transitions=arguments.max_transitions)
    else:
        state = workflow.status()
        result = verify_role_manifests(arguments.role, state)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
