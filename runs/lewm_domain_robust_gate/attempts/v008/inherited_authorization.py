#!/usr/bin/env python3
"""Authenticate the transitive v001 -> ... -> v007 -> v008 authorization.

This module is deliberately stdlib-only and outcome-blind.  It authenticates
the immutable v008 inheritance seal against the durable controller state, the
complete append-only ledger chain, the v001 procedural-invalidity evidence,
and every inherited checkpoint annotation.  Scientific entry points call it
before selecting a seed tuple or creating an output path.
"""

from __future__ import annotations

import ast
import hashlib
import fcntl
import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ATTEMPT_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ATTEMPT_ROOT.parents[1]
REPO_ROOT = ATTEMPT_ROOT.parents[3]

ACTIVE_ATTEMPT = "v008"
SCIENCE_ATTEMPT = "v001"
INTERMEDIATE_ATTEMPT = "v004"
SOURCE_PARENT_ATTEMPT = "v005"
SOURCE_ATTEMPT = "v006"
ACTIVATION_SOURCE_ATTEMPT = "v007"
FIT_SOURCE_ATTEMPT = "v003"
PRIOR_ATTEMPT = "v002"
INHERITANCE_SEAL_PATH = ATTEMPT_ROOT / "audit/pre_data_inheritance_seal.json"
SOURCE_PRE_DATA_SEAL_PATH = (
    STUDY_ROOT / "attempts/v007/audit/pre_data_inheritance_seal.json"
)
SOURCE_PRE_SELECTION_SEAL_PATH = (
    STUDY_ROOT / "attempts/v006/audit/pre_selection_seal.json"
)
SELECTION_PRE_DATA_SEAL_PATH = (
    STUDY_ROOT / "attempts/v006/audit/pre_data_inheritance_seal.json"
)
SOURCE_PARENT_PRE_DATA_SEAL_PATH = (
    STUDY_ROOT / "attempts/v005/audit/pre_data_inheritance_seal.json"
)
INTERMEDIATE_PRE_DATA_SEAL_PATH = (
    STUDY_ROOT / "attempts/v004/audit/pre_data_inheritance_seal.json"
)
FIT_PRE_DATA_SEAL_PATH = (
    STUDY_ROOT / "attempts/v003/audit/pre_data_inheritance_seal.json"
)
PRIOR_PRE_DATA_SEAL_PATH = (
    STUDY_ROOT / "attempts/v002/audit/pre_data_inheritance_seal.json"
)
INVALIDITY_PATH = (
    STUDY_ROOT / "attempts/v007/audit/v007_procedural_invalidity.json"
)
INVALIDITY_DRAFT_PATH = (
    STUDY_ROOT / "attempts/v007/audit/v007_procedural_invalidity_draft.json"
)
SOURCE_PARENT_INVALIDITY_PATH = (
    STUDY_ROOT / "attempts/v005/audit/v005_procedural_invalidity.json"
)
INTERMEDIATE_INVALIDITY_PATH = (
    STUDY_ROOT / "attempts/v004/audit/v004_procedural_invalidity.json"
)
FIT_INVALIDITY_PATH = (
    STUDY_ROOT / "attempts/v003/audit/v003_procedural_invalidity.json"
)
FIT_TRANSACTION_RECEIPT_PATH = (
    STUDY_ROOT / "attempts/v003/audit/version_forward_transaction_receipt.json"
)
INTERMEDIATE_TRANSACTION_RECEIPT_PATH = (
    STUDY_ROOT / "attempts/v004/audit/version_forward_transaction_receipt.json"
)
PRIOR_INVALIDITY_PATH = (
    STUDY_ROOT / "attempts/v002/audit/v002_procedural_invalidity_v2.json"
)
PRIOR_TRANSACTION_RECEIPT_PATH = (
    STUDY_ROOT / "attempts/v002/audit/version_forward_transaction_receipt.json"
)
SOURCE_TRANSACTION_RECEIPT_PATH = (
    STUDY_ROOT / "attempts/v007/audit/version_forward_transaction_receipt.json"
)
SOURCE_PARENT_TRANSACTION_RECEIPT_PATH = (
    STUDY_ROOT / "attempts/v005/audit/version_forward_transaction_receipt.json"
)
SOURCE_CONTROLLER_ADAPTER_PATH = (
    STUDY_ROOT / "attempts/v007/version_forward_transaction.py"
)
SELECTION_CONTROLLER_ADAPTER_PATH = (
    STUDY_ROOT / "attempts/v006/version_forward_transaction.py"
)
STATE_PATH = STUDY_ROOT / "STATE.json"
LEDGER_PATH = STUDY_ROOT / "RESEARCH_LEDGER.jsonl"
LEDGER_GENESIS_PATH = STUDY_ROOT / "LEDGER_CHAIN_GENESIS.json"
INDEPENDENT_VERIFIER_PATH = ATTEMPT_ROOT / "verify_version_forward.py"
TRANSACTION_RECEIPT_PATH = (
    ATTEMPT_ROOT / "audit/version_forward_transaction_receipt.json"
)
TRANSACTION_SOURCE_PATH = ATTEMPT_ROOT / "version_forward_transaction.py"
PROGRAM_PATH = STUDY_ROOT / "program.py"
CONTROLLER_ADAPTER_PATH = TRANSACTION_SOURCE_PATH
STATE_TRANSACTION_PATH = STUDY_ROOT / "STATE_TRANSACTION.json"
PENDING_STAGING_PATH = STUDY_ROOT / ".STATE_TRANSACTION.json.v008-staging"
STATE_STAGING_PATH = STUDY_ROOT / ".STATE.json.v008-staging"
RECEIPT_JOURNAL_PATH = STUDY_ROOT / "VERSION_FORWARD_TRANSACTION.json"
RECEIPT_JOURNAL_STAGING_PATH = (
    STUDY_ROOT / ".VERSION_FORWARD_TRANSACTION.json.v008-staging"
)
RECEIPT_STAGING_PATH = (
    ATTEMPT_ROOT
    / "audit/.version_forward_transaction_receipt.json.v008-staging"
)
RECEIPT_CONTEXT_EVENT_FIELD = "version_forward_receipt_context_sha256"
SOURCE_ADAPTER_MARKER_FIELD = "source_controller_adapter_marker"
SOURCE_ADAPTER_STATE_KEY = "v007_durable_controller_adapter"
PROGRAM_LOCK_PATH = STUDY_ROOT / ".program.lock"
_EARLY_ROOT_GUARD_TOKEN = object()
_ROLE_COUNT_RECOVERY_GUARD_TOKEN = object()

INHERITED_STATES = (
    "BOOTSTRAP_AUDIT",
    "DIAGNOSTIC_ACCOUNT",
    "PREREGISTRATION_AND_POWER",
    "IMPLEMENTATION_COMPLETE",
    "PRESEAL_QUALIFICATION",
    "PRE_OUTCOME_SEAL",
    "FIT_COHORTS",
    "FIT_LOCK",
    "PRE_SELECTION_SEAL",
)
ZERO_COUNTS = {
    "fit_outcome_episodes": 1200,
    "selection_outcome_episodes": 0,
    "smoke_outcome_episodes": 0,
    "confirmation_outcome_episodes_generated": 0,
    "confirmation_outcome_episodes_executed": 0,
    "confirmation_outcomes_opened_for_analysis": False,
}
REQUIRED_EQUIVALENCE = {
    "source_hash_equivalent": True,
    "normalized_ast_equivalent": True,
    "scientific_object_hash_equivalent": True,
    "configuration_hash_equivalent": True,
    "scientific_changes": False,
    "attempt_parameterization_verified": True,
    "runtime_modules_accept_active_attempt": True,
    "independent_verifier_accepts_active_attempt": True,
}
VERIFIER_PARTITION_NAMES = (
    "exact_hash",
    "normalized_ast",
    "canonical_contracts",
    "procedural_only",
    "new_lineage_support",
    "lineage_support",
)


class InheritedAuthorizationError(RuntimeError):
    """The v001-through-v008 immutable authorization drifted."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _exact_typed_mapping(value: Any, expected: Mapping[str, Any]) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(expected)
        and all(
            type(value.get(key)) is type(item) and value.get(key) == item
            for key, item in expected.items()
        )
    )


def _validate_inheritance_seal_scalar_types(value: Any) -> None:
    """Reject every JSON numeric alias in the inheritance-seal schema."""

    integer_fields = {
        "bytes",
        "confirmation_outcome_episodes_executed",
        "confirmation_outcome_episodes_generated",
        "created_unix_ns",
        "event_count",
        "file_count",
        "episode_count",
        "fit_outcome_episodes",
        "ledger_event_count",
        "schema_version",
        "science_attempt_exact_path_count",
        "selection_outcome_episodes",
        "smoke_outcome_episodes",
        "source_attempt_allowed_suffix_path_count",
        "source_top_level_unit_count",
        "target_top_level_unit_count",
        "unchanged_top_level_symbol_count",
        "unchanged_top_level_unit_count",
    }
    boolean_fields = {
        "all_inherited_checkpoints_authenticated",
        "all_rehashed",
        "ast_equivalent",
        "attempt_parameterization_verified",
        "authoritative",
        "complete_scope_live_walk",
        "complete_source_partition",
        "configuration_hash_equivalent",
        "confirmation_outcomes_opened_for_analysis",
        "contracts_canonically_equivalent",
        "hashes_equivalent",
        "hdf5_contents_opened",
        "independent_verifier_accepts_active_attempt",
        "invalidity_authenticated",
        "expected_boundary_outcome_counts",
        "no_confirmation_artifacts",
        "normalized_ast_equivalent",
        "outcome_arrays_opened",
        "passed",
        "procedural_invalidity_confirmed",
        "root_controls_authenticated",
        "runtime_modules_accept_active_attempt",
        "scientific_changes",
        "scientific_object_hash_equivalent",
        "scientific_objects_equivalent",
        "source_equivalent",
        "source_hash_equivalent",
        "source_pre_data_seal_authenticated",
        "source_pre_selection_seal_authenticated",
        "source_seal_bound",
        "source_sealed_files_rehashed",
        "state_transaction_present",
        "study_root_namespace_complete",
        "v008_manifest_closure",
        "v3_targets_opened",
        "zero_confirmation_outcomes_at_version_forward",
        "zero_outcome_counters",
    }
    stack: list[tuple[str | None, Any]] = [(None, value)]
    while stack:
        key, item = stack.pop()
        if isinstance(item, Mapping):
            if not all(isinstance(child_key, str) for child_key in item):
                raise InheritedAuthorizationError(
                    "inheritance seal contains a non-string object key"
                )
            stack.extend((str(child_key), child) for child_key, child in item.items())
            continue
        if isinstance(item, list):
            stack.extend((None, child) for child in item)
            continue
        if key in integer_fields:
            if type(item) is not int:
                raise InheritedAuthorizationError(
                    f"inheritance seal integer type drift: {key}"
                )
        elif key in boolean_fields:
            if type(item) is not bool:
                raise InheritedAuthorizationError(
                    f"inheritance seal boolean type drift: {key}"
                )
        elif type(item) in (bool, int, float):
            raise InheritedAuthorizationError(
                f"inheritance seal unexpected numeric scalar: {key}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path) -> Path:
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise InheritedAuthorizationError(f"required lineage file is absent: {path}") from error
    metadata = path.lstat()
    if not stat.S_ISREG(mode) or path.is_symlink() or metadata.st_nlink != 1:
        raise InheritedAuthorizationError(f"lineage file is not regular: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    path = _regular_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InheritedAuthorizationError(f"cannot decode lineage JSON: {path}") from error
    if not isinstance(value, dict):
        raise InheritedAuthorizationError(f"lineage JSON is not an object: {path}")
    return value


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve(strict=True).relative_to(
            REPO_ROOT.resolve(strict=True)
        ).as_posix()
    except ValueError as error:
        raise InheritedAuthorizationError(f"lineage path escapes repository: {path}") from error


def _resolve_relative(raw: Any, *, required_root: Path | None = None) -> Path:
    if not isinstance(raw, str):
        raise InheritedAuthorizationError("lineage path is not a string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
        raise InheritedAuthorizationError(f"noncanonical lineage path: {raw}")
    path = (REPO_ROOT / relative).resolve(strict=False)
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise InheritedAuthorizationError(f"lineage path escapes repository: {raw}")
    if required_root is not None and not path.is_relative_to(required_root.resolve()):
        raise InheritedAuthorizationError(f"lineage path has the wrong owner: {raw}")
    return path


def _file_link(value: Any, *, expected_path: Path) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise InheritedAuthorizationError("lineage file link schema drift")
    path = _resolve_relative(value["path"])
    if path != expected_path.resolve():
        raise InheritedAuthorizationError(f"lineage file link path drift: {path}")
    observed = sha256_file(_regular_file(path))
    if value["sha256"] != observed:
        raise InheritedAuthorizationError(f"lineage file link hash drift: {path}")
    return {"path": str(value["path"]), "sha256": observed}


def _verify_sealed_files(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise InheritedAuthorizationError("inheritance seal lacks sealed_files")
    observed: dict[str, str] = {}
    for raw, expected in sorted(value.items()):
        path = _resolve_relative(raw)
        digest = sha256_file(_regular_file(path))
        if not isinstance(expected, str) or expected != digest:
            raise InheritedAuthorizationError(f"inherited sealed-file hash drift: {raw}")
        observed[str(raw)] = digest
    required = (
        ATTEMPT_ROOT / "inherited_authorization.py",
        ATTEMPT_ROOT / "generator.py",
        ATTEMPT_ROOT / "runner.py",
        ATTEMPT_ROOT / "workflow.py",
        ATTEMPT_ROOT / "checkpoints.py",
        ATTEMPT_ROOT / "verify_version_forward.py",
        ATTEMPT_ROOT / "DGP_MATRIX.json",
        ATTEMPT_ROOT / "cohort_seed_ledger.json",
    )
    missing = [
        _relative(path)
        for path in required
        if observed.get(_relative(path)) != sha256_file(_regular_file(path))
    ]
    if missing:
        raise InheritedAuthorizationError(
            f"inheritance seal omits required v008 runtime inputs: {missing}"
        )
    return observed


def _active_seal_verifier_projection(
    seal: Mapping[str, Any], sealed_files: Mapping[str, str]
) -> dict[str, Any]:
    """Derive compact receipt claims from the freshly rehashed active seal."""

    raw_sealed = seal.get("sealed_files")
    if (
        type(raw_sealed) is not dict
        or type(sealed_files) is not dict
        or dict(raw_sealed) != dict(sealed_files)
    ):
        raise InheritedAuthorizationError(
            "active inheritance sealed-file projection drift"
        )
    closure = seal.get("v008_manifest_closure")
    discovered = closure.get("discovered_paths") if type(closure) is dict else None
    if (
        type(discovered) is not list
        or not discovered
        or any(type(item) is not str or not item for item in discovered)
        or discovered != sorted(discovered)
        or len(discovered) != len(set(discovered))
    ):
        raise InheritedAuthorizationError(
            "active inheritance manifest-closure path drift"
        )
    partitions = seal.get("source_partitions")
    if type(partitions) is not dict or set(partitions) != set(
        VERIFIER_PARTITION_NAMES
    ):
        raise InheritedAuthorizationError(
            "active inheritance source-partition schema drift"
        )
    members: dict[str, list[str]] = {}
    for name in VERIFIER_PARTITION_NAMES:
        records = partitions.get(name)
        if type(records) is not list:
            raise InheritedAuthorizationError(
                f"active inheritance partition type drift: {name}"
            )
        names: list[str] = []
        for record in records:
            relative_path = (
                record.get("relative_path") if type(record) is dict else None
            )
            if type(relative_path) is not str or not relative_path:
                raise InheritedAuthorizationError(
                    f"active inheritance partition record drift: {name}"
                )
            relative = Path(relative_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != relative_path
            ):
                raise InheritedAuthorizationError(
                    f"active inheritance partition path is noncanonical: {relative_path}"
                )
            names.append(relative_path)
        if names != sorted(names) or len(names) != len(set(names)):
            raise InheritedAuthorizationError(
                f"active inheritance partition membership drift: {name}"
            )
        members[name] = names
    flattened = [
        relative_path
        for name in VERIFIER_PARTITION_NAMES
        for relative_path in members[name]
    ]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != discovered:
        raise InheritedAuthorizationError(
            "active inheritance partition/manifest-closure projection drift"
        )
    source_paths = sorted(
        set(discovered) - set(members["new_lineage_support"])
    )
    complete = seal.get("complete_source_partition")
    if (
        type(complete) is not dict
        or set(complete)
        != {
            "source_paths",
            "target_paths",
            "partition_members",
            "overlap",
            "unpartitioned_source",
            "unpartitioned_target",
            "passed",
        }
        or type(complete.get("source_paths")) is not list
        or complete["source_paths"] != source_paths
        or type(complete.get("target_paths")) is not list
        or complete["target_paths"] != discovered
        or type(complete.get("partition_members")) is not dict
        or complete["partition_members"] != members
        or complete.get("overlap") != []
        or complete.get("unpartitioned_source") != []
        or complete.get("unpartitioned_target") != []
        or complete.get("passed") is not True
    ):
        raise InheritedAuthorizationError(
            "active inheritance complete source-partition projection drift"
        )
    independent = {
        "source_path_count": len(source_paths),
        "target_path_count": len(discovered),
        "partition_counts": {
            name: len(members[name]) for name in VERIFIER_PARTITION_NAMES
        },
    }
    return {
        "partition_file_count": len(discovered),
        "sealed_file_count": len(sealed_files),
        "independent_partitions": independent,
    }


def _read_only_snapshot(root: Path) -> dict[str, tuple[Any, ...]]:
    """Record path/type/inode/size/mtime without following directory links."""

    root = root.resolve(strict=True)
    result: dict[str, tuple[Any, ...]] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in sorted((*directory_names, *file_names)):
            path = base / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            result[relative] = (
                stat.S_IFMT(info.st_mode),
                int(info.st_dev),
                int(info.st_ino),
                int(info.st_size),
                int(info.st_mtime_ns),
            )
    return result


def _verify_post_forward_recomputation(
    seal_path: Path,
    *,
    authorized_early_state: str | None = None,
    authorized_role_count_recovery: str | None = None,
    expected_partition_file_count: int | None = None,
    expected_sealed_file_count: int | None = None,
) -> dict[str, Any]:
    """Run and parse the separately written verifier without permitting output."""

    script = _regular_file(INDEPENDENT_VERIFIER_PATH)
    before = _read_only_snapshot(STUDY_ROOT)
    command = [
        sys.executable,
        str(script),
        "post-forward",
        "--study-root",
        str(STUDY_ROOT),
        "--seal",
        str(seal_path),
    ]
    if authorized_early_state is not None:
        command.extend(
            ["--authorized-early-verifier-state", authorized_early_state]
        )
    if authorized_role_count_recovery is not None:
        command.extend(
            ["--authorized-role-count-recovery", authorized_role_count_recovery]
        )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    after = _read_only_snapshot(STUDY_ROOT)
    if before != after:
        raise InheritedAuthorizationError(
            "independent version-forward verification mutated the study tree"
        )
    if completed.returncode != 0 or completed.stderr.strip():
        raise InheritedAuthorizationError(
            "independent post-forward verifier rejected inherited authorization"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise InheritedAuthorizationError(
            "independent post-forward verifier emitted a noncanonical result"
        )
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise InheritedAuthorizationError(
            "independent post-forward verifier emitted invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise InheritedAuthorizationError(
            "independent post-forward verifier result is not an object"
        )
    expected = {
        "passed": True,
        "phase": "post-forward",
        "attempt": ACTIVE_ATTEMPT,
        "active_attempt": ACTIVE_ATTEMPT,
        "science_attempt": SCIENCE_ATTEMPT,
        "seal_path": _relative(seal_path),
        "seal_sha256": sha256_file(seal_path),
        "authorized_early_verifier_state": authorized_early_state,
        "authorized_role_count_recovery": authorized_role_count_recovery,
        "outcome_arrays_opened": False,
        "output_paths_created": 0,
        "read_only": True,
    }
    if expected_partition_file_count is not None:
        if type(expected_partition_file_count) is not int:
            raise InheritedAuthorizationError(
                "expected partition-file count has structural type drift"
            )
        expected["partition_file_count"] = expected_partition_file_count
    if expected_sealed_file_count is not None:
        if type(expected_sealed_file_count) is not int:
            raise InheritedAuthorizationError(
                "expected sealed-file count has structural type drift"
            )
        expected["sealed_file_count"] = expected_sealed_file_count
    if set(value) != set(expected) or any(
        value.get(key) != expected_value
        or (
            type(expected_value) in (bool, int)
            and type(value.get(key)) is not type(expected_value)
        )
        for key, expected_value in expected.items()
    ):
        raise InheritedAuthorizationError(
            "independent post-forward verifier result identity/completeness drift"
        )
    return value


def _verify_ledger_chain(
    *, ledger_path: Path, genesis_path: Path, state: Mapping[str, Any]
) -> list[dict[str, Any]]:
    genesis = _read_json(genesis_path)
    raw = _regular_file(ledger_path).read_bytes()
    try:
        prefix_bytes = int(genesis["legacy_prefix_bytes"])
        legacy_count = int(genesis["legacy_prefix_event_count"])
        expected_prefix_hash = str(genesis["legacy_prefix_sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise InheritedAuthorizationError("ledger genesis schema drift") from error
    prefix = raw[:prefix_bytes]
    if len(prefix) != prefix_bytes or hashlib.sha256(prefix).hexdigest() != expected_prefix_hash:
        raise InheritedAuthorizationError("ledger immutable prefix drift")
    lines = raw.splitlines(keepends=True)
    if len(lines) < legacy_count or b"".join(lines[:legacy_count]) != prefix:
        raise InheritedAuthorizationError("ledger prefix event boundary drift")
    events: list[dict[str, Any]] = []
    previous = f"legacy:{expected_prefix_hash}"
    for sequence, encoded in enumerate(lines, start=1):
        if not encoded.endswith(b"\n"):
            raise InheritedAuthorizationError(f"ledger event {sequence} is not line complete")
        try:
            event = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise InheritedAuthorizationError(f"ledger event {sequence} is invalid JSON") from error
        if not isinstance(event, dict):
            raise InheritedAuthorizationError(f"ledger event {sequence} is not an object")
        if sequence > legacy_count:
            payload = dict(event)
            observed = payload.pop("record_sha256", None)
            expected = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
            if (
                observed != expected
                or type(payload.get("seq")) is not int
                or payload.get("seq") != sequence
                or not isinstance(payload.get("prev_sha256"), str)
                or payload.get("prev_sha256") != previous
            ):
                raise InheritedAuthorizationError(f"ledger hash-chain drift at event {sequence}")
            previous = expected
        events.append(event)
    if state.get("ledger_event_count") != len(events) or state.get(
        "ledger_head_sha256"
    ) != previous:
        raise InheritedAuthorizationError("STATE/ledger head mismatch")
    return events


def _controller_state_file_sha256(value: Mapping[str, Any]) -> str:
    encoded = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _verify_role_count_recovery_authorization(
    *,
    role: str,
    controller: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate one final count commit before its immutable checkpoint."""

    specifications = {
        "fit": {
            "state": "FIT_COHORTS",
            "counter": "fit_outcome_episodes",
            "expected": 1200,
            "episodes_per_regime": 300,
            "audit": ATTEMPT_ROOT / "audit/fit_cohorts.json",
            "later": (
                ATTEMPT_ROOT / "audit/fit_lock_checkpoint.json",
                ATTEMPT_ROOT / "audit/pre_selection_manifest.json",
                ATTEMPT_ROOT / "audit/pre_selection_seal.json",
                ATTEMPT_ROOT / "audit/selection_cohorts.json",
                ATTEMPT_ROOT / "data/selection",
                ATTEMPT_ROOT / "data/smoke",
                ATTEMPT_ROOT / "data/confirmation",
            ),
        },
        "selection": {
            "state": "SELECTION_COHORTS",
            "counter": "selection_outcome_episodes",
            "expected": 2000,
            "episodes_per_regime": 500,
            "audit": ATTEMPT_ROOT / "audit/selection_cohorts.json",
            "later": (
                ATTEMPT_ROOT / "audit/candidate_selection.json",
                ATTEMPT_ROOT / "audit/gate_freeze_checkpoint.json",
                ATTEMPT_ROOT / "audit/confirmation_power_and_cohort_freeze.json",
                ATTEMPT_ROOT / "audit/pre_confirmation_package_seal.json",
                ATTEMPT_ROOT / "data/smoke",
                ATTEMPT_ROOT / "data/confirmation",
            ),
        },
    }
    specification = specifications.get(role)
    if specification is None:
        raise InheritedAuthorizationError("unknown role-count recovery role")
    state_name = str(specification["state"])
    counter = str(specification["counter"])
    expected = int(specification["expected"])
    expected_counters = {
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": expected if role == "selection" else 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    state_machine = (
        "BOOTSTRAP_AUDIT",
        "DIAGNOSTIC_ACCOUNT",
        "PREREGISTRATION_AND_POWER",
        "IMPLEMENTATION_COMPLETE",
        "PRESEAL_QUALIFICATION",
        "PRE_OUTCOME_SEAL",
        "FIT_COHORTS",
        "FIT_LOCK",
        "PRE_SELECTION_SEAL",
        "SELECTION_COHORTS",
        "CANDIDATE_SELECTION",
        "GATE_FREEZE",
        "CONFIRMATION_POWER_AND_COHORT_FREEZE",
        "PRE_CONFIRMATION_PACKAGE_SEAL",
        "EXCLUDED_MECHANICAL_SMOKE",
        "CONFIRMATION_GENERATION",
        "CONFIRMATION_EXECUTION",
        "CONFIRMATION_INPUT_SEAL",
        "SEALED_ANALYSIS",
        "LATENCY_AND_RESOURCE_REPORTING",
        "INDEPENDENT_VERIFICATION",
        "TERMINAL",
        "POST_TERMINAL_REPORTING",
    )
    current_index = state_machine.index(state_name)
    checkpoints = controller.get("verified_checkpoints")
    state_checks = {
        "state": controller.get("current_state") == state_name,
        "counter": type(controller.get(counter)) is int
        and controller.get(counter) == expected,
        "counters": all(
            type(controller.get(name)) is type(value)
            and controller.get(name) == value
            for name, value in expected_counters.items()
        ),
        "state_machine": tuple(controller.get("state_machine", ())) == state_machine,
        "completed_prefix": controller.get("completed_states")
        == list(state_machine[:current_index]),
        "checkpoint_prefix": isinstance(checkpoints, list)
        and len(checkpoints) == current_index,
        "terminal_unset": controller.get("terminal_label") is None
        and controller.get("confirmation_terminal") is False
        and controller.get("early_scientific_failure") is None
        and controller.get("postconfirmation_integrity_failure") is None
        and controller.get("skipped_states") in (None, []),
    }
    if not all(state_checks.values()):
        raise InheritedAuthorizationError(
            f"role-count recovery controller state drift: {state_checks}"
        )
    audit_path = Path(specification["audit"])
    audit_relative = _relative(audit_path) if audit_path.exists() else (
        audit_path.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    )
    if any(
        not isinstance(item, Mapping)
        or item.get("evidence_path") == audit_relative
        or item.get("name") == f"{ACTIVE_ATTEMPT}_{role}_cohorts_verified"
        for item in checkpoints
    ):
        raise InheritedAuthorizationError(
            "role-count recovery already has a role checkpoint"
        )

    if len(events) < 2:
        raise InheritedAuthorizationError("role-count recovery ledger is too short")
    event = dict(events[-1])
    event_body = dict(event)
    observed_hash = event_body.pop("record_sha256", None)
    event_checks = {
        "schema": set(event)
        == {
            "event",
            "attempt",
            "fields",
            "prior_controller_adapter_marker",
            "created_unix_ns",
            "seq",
            "prev_sha256",
            "record_sha256",
            "v008_durable_adapter_transaction",
        },
        "event": event.get("event") == "outcome_counts_updated",
        "attempt": event.get("attempt") == ACTIVE_ATTEMPT,
        "fields": event.get("fields") == {counter: expected},
        "sequence": type(event.get("seq")) is int
        and type(controller.get("ledger_event_count")) is int
        and event.get("seq") == len(events)
        == controller.get("ledger_event_count"),
        "created": type(event.get("created_unix_ns")) is int
        and type(controller.get("updated_unix_ns")) is int
        and event.get("created_unix_ns") == controller.get("updated_unix_ns"),
        "head": observed_hash == controller.get("ledger_head_sha256"),
        "hash": observed_hash == hashlib.sha256(_canonical_bytes(event_body)).hexdigest(),
    }
    previous = events[-2]
    previous_hash = previous.get("record_sha256")
    previous_checks = {
        "hash": isinstance(previous_hash, str),
        "link": event.get("prev_sha256") == previous_hash,
        "created": type(previous.get("created_unix_ns")) is int,
    }
    if not all(event_checks.values()) or not all(previous_checks.values()):
        raise InheritedAuthorizationError(
            "role-count recovery ledger event drift: "
            f"event={event_checks}, previous={previous_checks}"
        )
    pre_count_state = dict(controller)
    pre_count_state[counter] = 0
    pre_count_state["updated_unix_ns"] = previous["created_unix_ns"]
    pre_count_state["ledger_event_count"] = len(events) - 1
    pre_count_state["ledger_head_sha256"] = previous_hash
    pre_count_state.pop("v008_durable_controller_adapter", None)
    prior_adapter_marker = _verify_controller_adapter_marker_value(
        event.get("prior_controller_adapter_marker"),
        event_count=len(events) - 1,
        head_sha256=str(previous_hash),
        label="prior role-count recovery",
        state_without_marker=pre_count_state,
    )
    pre_count_state["v008_durable_controller_adapter"] = prior_adapter_marker
    pre_count_state_sha256 = _controller_state_file_sha256(pre_count_state)

    regimes = (
        "native_plan",
        "markov_oracle",
        "plan_action_noise_0p2",
        "plan_random_action_0p1",
    )
    manifest_links: list[dict[str, str]] = []
    for regime in regimes:
        manifest_path = _regular_file(
            ATTEMPT_ROOT / "data" / role / regime / "execution_manifest.json"
        )
        manifest = _read_json(manifest_path)
        authorization = manifest.get("authorization")
        manifest_checks = {
            "attempt": manifest.get("attempt") == ACTIVE_ATTEMPT,
            "role": manifest.get("role") == role,
            "regime": manifest.get("regime") == regime,
            "complete": manifest.get("complete") is True,
            "count": manifest.get("episode_count")
            == int(specification["episodes_per_regime"]),
            "authorization": isinstance(authorization, Mapping)
            and authorization.get("active_attempt") == ACTIVE_ATTEMPT
            and authorization.get("state") == state_name
            and authorization.get("state_sha256") == pre_count_state_sha256,
        }
        if not all(manifest_checks.values()):
            raise InheritedAuthorizationError(
                f"role-count recovery execution authorization drift: "
                f"{role}/{regime}: {manifest_checks}"
            )
        manifest_links.append(
            {"path": _relative(manifest_path), "sha256": sha256_file(manifest_path)}
        )

    audit_sha256: str | None = None
    audit_created_unix_ns: int | None = None
    if os.path.lexists(audit_path):
        audit = _read_json(audit_path)
        canonical = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        if audit_path.read_bytes() != canonical:
            raise InheritedAuthorizationError(
                "role-count recovery audit is not canonical JSON"
            )
        if (
            type(audit.get("created_unix_ns")) is not int
            or int(audit["created_unix_ns"])
            < int(event["created_unix_ns"])
        ):
            raise InheritedAuthorizationError(
                "role-count recovery audit chronology drift"
            )
        audit_created_unix_ns = int(audit["created_unix_ns"])
        audit_sha256 = sha256_file(audit_path)
    conflicts = [
        _relative(path) if path.exists() and path.is_file() else str(path)
        for path in specification["later"]
        if os.path.lexists(path)
    ]
    if conflicts:
        raise InheritedAuthorizationError(
            f"role-count recovery has later output conflicts: {conflicts}"
        )
    return {
        "passed": True,
        "role": role,
        "state": state_name,
        "counter_field": counter,
        "counter_value": expected,
        "count_update_event_seq": len(events),
        "count_update_event_sha256": observed_hash,
        "pre_count_state_sha256": pre_count_state_sha256,
        "prior_controller_adapter_marker_sha256": hashlib.sha256(
            _canonical_bytes(prior_adapter_marker)
        ).hexdigest(),
        "execution_manifests": manifest_links,
        "role_audit_path": audit_relative,
        "role_audit_sha256": audit_sha256,
        "role_audit_created_unix_ns": audit_created_unix_ns,
        "no_later_output_conflicts": True,
    }


def _ast_sha256(path: Path) -> str:
    path = _regular_file(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise InheritedAuthorizationError(
            f"cannot parse transaction-bound source: {path}"
        ) from error
    dump = ast.dump(tree, include_attributes=False).encode("utf-8")
    return hashlib.sha256(dump).hexdigest()


def _verify_source_controller_adapter_marker_value(
    marker: Any,
    *,
    event_count: int,
    head_sha256: str,
    state_without_marker: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the rolled-over marker to the immutable v007 adapter and state."""

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
    adapter = _regular_file(SOURCE_CONTROLLER_ADAPTER_PATH)
    root_program = _regular_file(PROGRAM_PATH)
    marker_core = dict(marker) if isinstance(marker, Mapping) else {}
    state_binding = marker_core.pop("state_binding_sha256", None)
    checks = {
        "schema": isinstance(marker, Mapping) and set(marker) == expected_keys,
        "version": isinstance(marker, Mapping)
        and type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 2,
        "authorization": isinstance(marker, Mapping)
        and marker.get("authorization_kind")
        == "receipt_bound_v007_root_controller_adapter",
        "adapter_path": isinstance(marker, Mapping)
        and marker.get("adapter_source_path") == _relative(adapter),
        "adapter_hash": isinstance(marker, Mapping)
        and marker.get("adapter_source_sha256") == sha256_file(adapter),
        "adapter_ast": isinstance(marker, Mapping)
        and marker.get("adapter_source_ast_sha256") == _ast_sha256(adapter),
        "root_path": isinstance(marker, Mapping)
        and marker.get("root_program_path") == _relative(root_program),
        "root_hash": isinstance(marker, Mapping)
        and marker.get("root_program_sha256") == sha256_file(root_program),
        "root_ast": isinstance(marker, Mapping)
        and marker.get("root_program_ast_sha256") == _ast_sha256(root_program),
        "ledger_count": isinstance(marker, Mapping)
        and type(marker.get("ledger_event_count")) is int
        and marker.get("ledger_event_count") == event_count,
        "ledger_head": isinstance(marker, Mapping)
        and marker.get("ledger_head_sha256") == head_sha256,
        "operation": isinstance(marker, Mapping)
        and isinstance(marker.get("operation_sha256"), str)
        and len(marker["operation_sha256"]) == 64
        and all(
            character in "0123456789abcdef"
            for character in marker["operation_sha256"]
        ),
        "state_binding": isinstance(state_binding, str)
        and state_binding
        == hashlib.sha256(
            _canonical_bytes(
                {
                    "marker_without_state_binding": marker_core,
                    "state_without_marker": state_without_marker,
                }
            )
        ).hexdigest(),
    }
    if not all(checks.values()):
        raise InheritedAuthorizationError(
            f"source v007 controller adapter authorization drift: {checks}"
        )
    return dict(marker)


def _verify_controller_adapter_marker_value(
    marker: Any,
    *,
    event_count: int,
    head_sha256: str,
    label: str,
    state_without_marker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently bind one marker to the exact receipt-bound v008 adapter."""

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
    root_program = _regular_file(PROGRAM_PATH)
    operation = marker.get("operation_sha256") if isinstance(marker, Mapping) else None
    operation_is_sha256 = (
        isinstance(operation, str)
        and len(operation) == 64
        and all(character in "0123456789abcdef" for character in operation)
    )
    marker_without_binding = dict(marker) if isinstance(marker, Mapping) else {}
    observed_state_binding = marker_without_binding.pop(
        "state_binding_sha256", None
    )
    state_binding_is_sha256 = (
        isinstance(observed_state_binding, str)
        and len(observed_state_binding) == 64
        and all(
            character in "0123456789abcdef"
            for character in observed_state_binding
        )
    )
    checks = {
        "schema": isinstance(marker, Mapping) and set(marker) == expected_keys,
        "version": isinstance(marker, Mapping)
        and type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 2,
        "authorization": isinstance(marker, Mapping)
        and marker.get("authorization_kind")
        == "receipt_bound_v008_root_controller_adapter",
        "adapter_path": isinstance(marker, Mapping)
        and marker.get("adapter_source_path") == _relative(adapter),
        "adapter_hash": isinstance(marker, Mapping)
        and marker.get("adapter_source_sha256") == sha256_file(adapter),
        "adapter_ast": isinstance(marker, Mapping)
        and marker.get("adapter_source_ast_sha256") == _ast_sha256(adapter),
        "root_path": isinstance(marker, Mapping)
        and marker.get("root_program_path") == _relative(root_program),
        "root_hash": isinstance(marker, Mapping)
        and marker.get("root_program_sha256") == sha256_file(root_program),
        "root_ast": isinstance(marker, Mapping)
        and marker.get("root_program_ast_sha256") == _ast_sha256(root_program),
        "ledger_count": isinstance(marker, Mapping)
        and type(marker.get("ledger_event_count")) is int
        and marker.get("ledger_event_count") == event_count,
        "ledger_head": isinstance(marker, Mapping)
        and marker.get("ledger_head_sha256") == head_sha256,
        "operation": operation_is_sha256,
        "state_binding": state_binding_is_sha256
        and (
            state_without_marker is None
            or observed_state_binding
            == hashlib.sha256(
                _canonical_bytes(
                    {
                        "marker_without_state_binding": marker_without_binding,
                        "state_without_marker": state_without_marker,
                    }
                )
            ).hexdigest()
        ),
    }
    if not all(checks.values()):
        raise InheritedAuthorizationError(
            f"{label} v008 controller adapter authorization drift: {checks}"
        )
    return dict(marker)


def _verify_controller_adapter_marker(
    controller: Mapping[str, Any],
    *,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently bind STATE to the exact receipt-bound v008 adapter."""

    state_without_marker = json.loads(json.dumps(dict(controller)))
    state_without_marker.pop("v008_durable_controller_adapter", None)
    return _verify_controller_adapter_marker_value(
        controller.get("v008_durable_controller_adapter"),
        event_count=len(events),
        head_sha256=str(controller.get("ledger_head_sha256")),
        label="current STATE",
        state_without_marker=state_without_marker,
    )


def _verify_adapter_count_event_bindings(
    events: Sequence[Mapping[str, Any]],
) -> int:
    """Verify v008 predecessors while preserving authenticated v006 history."""

    verified = 0
    field = "prior_controller_adapter_marker"
    for event in events:
        is_v008_count = (
            event.get("attempt") == ACTIVE_ATTEMPT
            and event.get("event") == "outcome_counts_updated"
        )
        if field in event and not is_v008_count:
            marker = event.get(field)
            expected_keys = {
                "schema_version", "authorization_kind", "adapter_source_path",
                "adapter_source_sha256", "adapter_source_ast_sha256",
                "root_program_path", "root_program_sha256",
                "root_program_ast_sha256", "ledger_event_count",
                "ledger_head_sha256", "operation_sha256",
                "state_binding_sha256",
            }
            adapter = _regular_file(SELECTION_CONTROLLER_ADAPTER_PATH)
            program = _regular_file(PROGRAM_PATH)
            sequence = event.get("seq")
            previous = event.get("prev_sha256")
            historical = {
                "event": event.get("attempt") == SOURCE_ATTEMPT
                and event.get("event") == "outcome_counts_updated",
                "schema": isinstance(marker, Mapping)
                and set(marker) == expected_keys
                and type(marker.get("schema_version")) is int
                and marker.get("schema_version") == 2,
                "authorization": isinstance(marker, Mapping)
                and marker.get("authorization_kind")
                == "receipt_bound_v006_root_controller_adapter",
                "adapter": isinstance(marker, Mapping)
                and marker.get("adapter_source_path") == _relative(adapter)
                and marker.get("adapter_source_sha256") == sha256_file(adapter)
                and marker.get("adapter_source_ast_sha256") == _ast_sha256(adapter),
                "program": isinstance(marker, Mapping)
                and marker.get("root_program_path") == _relative(program)
                and marker.get("root_program_sha256") == sha256_file(program)
                and marker.get("root_program_ast_sha256") == _ast_sha256(program),
                "ledger": isinstance(marker, Mapping)
                and type(sequence) is int
                and isinstance(previous, str)
                and marker.get("ledger_event_count") == sequence - 1
                and marker.get("ledger_head_sha256") == previous,
                "digests": isinstance(marker, Mapping)
                and all(
                    isinstance(marker.get(key), str)
                    and len(marker[key]) == 64
                    and all(character in "0123456789abcdef" for character in marker[key])
                    for key in ("operation_sha256", "state_binding_sha256")
                ),
            }
            if not all(historical.values()):
                raise InheritedAuthorizationError(
                    f"historical v006 adapter marker drift: {historical}"
                )
        if not is_v008_count:
            continue
        if field not in event:
            raise InheritedAuthorizationError(
                "v008 count event lacks its prior adapter marker"
            )
        sequence = event.get("seq")
        previous = event.get("prev_sha256")
        if type(sequence) is not int or not isinstance(previous, str):
            raise InheritedAuthorizationError(
                "v008 count event predecessor identity drift"
            )
        _verify_controller_adapter_marker_value(
            event[field],
            event_count=sequence - 1,
            head_sha256=previous,
            label="v008 count-event predecessor",
        )
        verified += 1
    return verified


def _snapshot_record(
    value: Any,
    *,
    expected_path: Path,
    extra_keys: frozenset[str] = frozenset(),
    require_current_identity: bool,
) -> dict[str, Any]:
    base_keys = {"path", "sha256", "bytes", "device", "inode", "nlink"}
    if not isinstance(value, Mapping) or set(value) != base_keys | set(extra_keys):
        raise InheritedAuthorizationError(
            f"transaction snapshot schema drift: {expected_path}"
        )
    if value.get("path") != _relative(expected_path):
        raise InheritedAuthorizationError(
            f"transaction snapshot path drift: {expected_path}"
        )
    for key in ("bytes", "device", "inode", "nlink"):
        if type(value.get(key)) is not int or int(value[key]) < 0:
            raise InheritedAuthorizationError(
                f"transaction snapshot metadata drift: {expected_path}/{key}"
            )
    if value.get("nlink") != 1 or not isinstance(value.get("sha256"), str):
        raise InheritedAuthorizationError(
            f"transaction snapshot identity drift: {expected_path}"
        )
    if require_current_identity:
        path = _regular_file(expected_path)
        metadata = path.lstat()
        observed = {
            "path": _relative(path),
            "sha256": sha256_file(path),
            "bytes": metadata.st_size,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "nlink": metadata.st_nlink,
        }
        if any(value.get(key) != item for key, item in observed.items()):
            raise InheritedAuthorizationError(
                f"transaction snapshot current identity drift: {expected_path}"
            )
    return dict(value)


def _stable_snapshot_record(
    value: Any,
    *,
    expected_path: Path,
    extra_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Validate one inode-independent historical receipt record."""

    base_keys = {"path", "sha256", "bytes"}
    if not isinstance(value, Mapping) or set(value) != base_keys | set(extra_keys):
        raise InheritedAuthorizationError(
            f"stable transaction snapshot schema drift: {expected_path}"
        )
    digest = value.get("sha256")
    if (
        value.get("path") != _relative(expected_path)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(value.get("bytes")) is not int
        or int(value["bytes"]) < 0
    ):
        raise InheritedAuthorizationError(
            f"stable transaction snapshot identity drift: {expected_path}"
        )
    return dict(value)


def _verify_receipt_state_snapshot(
    value: Any,
    *,
    expected_attempt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _stable_snapshot_record(
        value,
        expected_path=STATE_PATH,
        extra_keys=frozenset({"object_sha256", "object"}),
    )
    state = record.get("object")
    if not isinstance(state, Mapping):
        raise InheritedAuthorizationError("transaction snapshot STATE object is absent")
    state_object = dict(state)
    encoded = (
        json.dumps(state_object, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    if (
        record.get("sha256") != hashlib.sha256(encoded).hexdigest()
        or record.get("bytes") != len(encoded)
        or record.get("object_sha256")
        != hashlib.sha256(_canonical_bytes(state_object)).hexdigest()
        or state_object.get("active_attempt") != expected_attempt
        or state_object.get("current_state") != "SELECTION_COHORTS"
        or state_object.get("confirmation_terminal") is not False
        or not _exact_typed_mapping(
            {key: state_object.get(key) for key in ZERO_COUNTS}, ZERO_COUNTS
        )
    ):
        raise InheritedAuthorizationError(
            f"transaction {expected_attempt} STATE snapshot drift"
        )
    return record, state_object


def _verify_receipt_verifier_result(
    value: Any,
    *,
    phase: str,
    seal_relative: str,
    seal_sha256: str,
    expected_projection: Mapping[str, Any],
) -> None:
    wrapper_keys = (
        {"producer", "standalone", "independent_partitions"}
        if phase == "pre-forward"
        else {"standalone", "independent_partitions"}
    )
    if not isinstance(value, Mapping) or set(value) != wrapper_keys:
        raise InheritedAuthorizationError(
            f"transaction {phase} verifier wrapper schema drift"
        )
    if phase == "pre-forward":
        producer = value.get("producer")
        producer_keys = {
            "passed",
            "phase",
            "attempt",
            "seal_path",
            "seal_sha256",
            "state_sha256",
            "ledger_sha256",
            "outcome_arrays_opened",
            "read_only",
        }
        if not isinstance(producer, Mapping) or set(producer) != producer_keys:
            raise InheritedAuthorizationError(
                "transaction producer closed schema drift"
            )
        if not (
            producer.get("passed") is True
            and producer.get("phase") == phase
            and producer.get("attempt") == ACTIVE_ATTEMPT
            and producer.get("seal_path") == seal_relative
            and producer.get("seal_sha256") == seal_sha256
            and isinstance(producer.get("state_sha256"), str)
            and len(producer["state_sha256"]) == 64
            and isinstance(producer.get("ledger_sha256"), str)
            and len(producer["ledger_sha256"]) == 64
            and producer.get("outcome_arrays_opened") is False
            and producer.get("read_only") is True
        ):
            raise InheritedAuthorizationError("transaction producer result drift")
    standalone = value.get("standalone")
    standalone_keys = {
        "passed",
        "phase",
        "attempt",
        "active_attempt",
        "science_attempt",
        "seal_path",
        "seal_sha256",
        "partition_file_count",
        "sealed_file_count",
        "authorized_early_verifier_state",
        "authorized_role_count_recovery",
        "outcome_arrays_opened",
        "output_paths_created",
        "read_only",
    }
    if not isinstance(standalone, Mapping) or set(standalone) != standalone_keys:
        raise InheritedAuthorizationError(
            f"transaction standalone {phase} closed schema drift"
        )
    if not (
        standalone.get("passed") is True
        and standalone.get("phase") == phase
        and standalone.get("attempt") == ACTIVE_ATTEMPT
        and standalone.get("active_attempt") == ACTIVE_ATTEMPT
        and standalone.get("science_attempt") == SCIENCE_ATTEMPT
        and standalone.get("seal_path") == seal_relative
        and standalone.get("seal_sha256") == seal_sha256
        and type(standalone.get("partition_file_count")) is int
        and standalone["partition_file_count"]
        == expected_projection["partition_file_count"]
        and type(standalone.get("sealed_file_count")) is int
        and standalone["sealed_file_count"]
        == expected_projection["sealed_file_count"]
        and standalone.get("authorized_early_verifier_state") is None
        and standalone.get("authorized_role_count_recovery") is None
        and standalone.get("outcome_arrays_opened") is False
        and type(standalone.get("output_paths_created")) is int
        and standalone.get("output_paths_created") == 0
        and standalone.get("read_only") is True
    ):
        raise InheritedAuthorizationError(
            f"transaction standalone {phase} result drift"
        )
    independent = value.get("independent_partitions")
    partition_names = {
        "exact_hash",
        "normalized_ast",
        "canonical_contracts",
        "procedural_only",
        "new_lineage_support",
        "lineage_support",
    }
    if (
        not isinstance(independent, Mapping)
        or set(independent)
        != {"source_path_count", "target_path_count", "partition_counts"}
        or type(independent.get("source_path_count")) is not int
        or int(independent["source_path_count"]) <= 0
        or type(independent.get("target_path_count")) is not int
        or int(independent["target_path_count"]) <= 0
        or not isinstance(independent.get("partition_counts"), Mapping)
        or set(independent["partition_counts"]) != partition_names
        or any(
            type(count) is not int or int(count) < 0
            for count in independent["partition_counts"].values()
        )
        or sum(int(count) for count in independent["partition_counts"].values())
        != int(independent["target_path_count"])
        or _canonical_bytes(independent)
        != _canonical_bytes(expected_projection["independent_partitions"])
    ):
        raise InheritedAuthorizationError(
            f"transaction independent {phase} result drift"
        )


def _verify_version_forward_transaction_receipt(
    *,
    seal_relative: str,
    seal_sha256: str,
    invalidity_relative: str,
    invalidity_sha256: str,
    expected_source_marker: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    expected_verifier_projection: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = _regular_file(TRANSACTION_RECEIPT_PATH)
    receipt = _read_json(receipt_path)
    receipt_payload = receipt_path.read_bytes()
    if receipt_payload != (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"):
        raise InheritedAuthorizationError(
            "version-forward transaction receipt is not canonical JSON"
        )
    expected_keys = {
        "schema_version",
        "artifact_type",
        "authorization_kind",
        "source_attempt",
        "target_attempt",
        "resume_state",
        "journal_created_unix_ns",
        "forward_created_unix_ns",
        "created_unix_ns",
        "receipt_context_sha256",
        "receipt_context",
        "seal",
        "invalidity",
        "invalidity_draft",
        "prior_receipt",
        "root_program",
        "transaction_source",
        "pre_snapshot",
        "pre_verifiers",
        "controller_return",
        "controller_return_sha256",
        "post_snapshot",
        "post_verifiers",
        "outcome_counts",
        "state_transaction_absent",
        "receipt_journal_absent",
        "passed",
    }
    journal_created = receipt.get("journal_created_unix_ns")
    forward_created = receipt.get("forward_created_unix_ns")
    receipt_created = receipt.get("created_unix_ns")
    receipt_context_sha256 = receipt.get("receipt_context_sha256")
    residue_paths = (
        STATE_TRANSACTION_PATH,
        PENDING_STAGING_PATH,
        STATE_STAGING_PATH,
        RECEIPT_JOURNAL_PATH,
        RECEIPT_JOURNAL_STAGING_PATH,
        RECEIPT_STAGING_PATH,
    )
    header = {
        "schema": set(receipt) == expected_keys
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 2,
        "artifact": receipt.get("artifact_type")
        == "atomic_zero_confirmation_outcome_version_forward_transaction_receipt",
        "authorization": receipt.get("authorization_kind")
        == "locked_v007_to_v008_inherited_pre_selection_forward",
        "source": receipt.get("source_attempt") == ACTIVATION_SOURCE_ATTEMPT,
        "target": receipt.get("target_attempt") == ACTIVE_ATTEMPT,
        "resume": receipt.get("resume_state") == "SELECTION_COHORTS",
        "created": type(journal_created) is int
        and type(forward_created) is int
        and type(receipt_created) is int
        and int(journal_created) > 0
        and forward_created == journal_created + 1
        and receipt_created == journal_created + 2,
        "context": isinstance(receipt_context_sha256, str)
        and len(receipt_context_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in receipt_context_sha256
        )
        and isinstance(receipt.get("receipt_context"), Mapping)
        and hashlib.sha256(
            _canonical_bytes(receipt["receipt_context"])
        ).hexdigest()
        == receipt_context_sha256,
        "zero": _exact_typed_mapping(receipt.get("outcome_counts"), ZERO_COUNTS),
        "transaction_absent": receipt.get("state_transaction_absent") is True
        and not any(os.path.lexists(path) for path in residue_paths[:3]),
        "journal_absent": receipt.get("receipt_journal_absent") is True
        and not any(os.path.lexists(path) for path in residue_paths[3:]),
        "passed": receipt.get("passed") is True,
    }
    if not all(header.values()):
        raise InheritedAuthorizationError(
            f"version-forward transaction receipt header drift: {header}"
        )

    fixed = {
        "seal": (INHERITANCE_SEAL_PATH, frozenset()),
        "invalidity": (INVALIDITY_PATH, frozenset()),
        "invalidity_draft": (INVALIDITY_DRAFT_PATH, frozenset()),
        "prior_receipt": (SOURCE_TRANSACTION_RECEIPT_PATH, frozenset()),
        "root_program": (PROGRAM_PATH, frozenset({"ast_sha256"})),
        "transaction_source": (
            TRANSACTION_SOURCE_PATH,
            frozenset({"ast_sha256"}),
        ),
    }
    fixed_records: dict[str, dict[str, Any]] = {}
    for name, (path, extra) in fixed.items():
        record = _snapshot_record(
            receipt.get(name),
            expected_path=path,
            extra_keys=extra,
            require_current_identity=True,
        )
        if extra and record.get("ast_sha256") != _ast_sha256(path):
            raise InheritedAuthorizationError(
                f"transaction source AST binding drift: {name}"
            )
        fixed_records[name] = record
    if (
        fixed_records["seal"]["path"] != seal_relative
        or fixed_records["seal"]["sha256"] != seal_sha256
        or fixed_records["invalidity"]["path"] != invalidity_relative
        or fixed_records["invalidity"]["sha256"] != invalidity_sha256
    ):
        raise InheritedAuthorizationError("transaction seal/invalidity binding drift")

    pre_snapshot = receipt.get("pre_snapshot")
    post_snapshot = receipt.get("post_snapshot")
    if (
        not isinstance(pre_snapshot, Mapping)
        or set(pre_snapshot) != {"state", "ledger", "genesis"}
        or not isinstance(post_snapshot, Mapping)
        or set(post_snapshot) != {"state", "ledger", "genesis"}
    ):
        raise InheritedAuthorizationError("transaction snapshot schema drift")
    _pre_state_record, pre_state = _verify_receipt_state_snapshot(
        pre_snapshot["state"], expected_attempt=ACTIVATION_SOURCE_ATTEMPT
    )
    _post_state_record, post_state = _verify_receipt_state_snapshot(
        post_snapshot["state"], expected_attempt=ACTIVE_ATTEMPT
    )
    ledger_extra = frozenset({"event_count", "head_sha256", "ledger_sha256"})
    pre_ledger = _stable_snapshot_record(
        pre_snapshot["ledger"],
        expected_path=LEDGER_PATH,
        extra_keys=ledger_extra,
    )
    post_ledger = _stable_snapshot_record(
        post_snapshot["ledger"],
        expected_path=LEDGER_PATH,
        extra_keys=ledger_extra,
    )
    pre_genesis = _stable_snapshot_record(
        pre_snapshot["genesis"],
        expected_path=LEDGER_GENESIS_PATH,
    )
    post_genesis = _stable_snapshot_record(
        post_snapshot["genesis"],
        expected_path=LEDGER_GENESIS_PATH,
    )
    if pre_genesis != post_genesis:
        raise InheritedAuthorizationError("transaction ledger-genesis identity drift")
    genesis_path = _regular_file(LEDGER_GENESIS_PATH)
    genesis_payload = genesis_path.read_bytes()
    current_genesis = {
        "path": _relative(genesis_path),
        "sha256": hashlib.sha256(genesis_payload).hexdigest(),
        "bytes": len(genesis_payload),
    }
    if pre_genesis != current_genesis:
        raise InheritedAuthorizationError("transaction ledger-genesis content drift")
    ledger_bytes = _regular_file(LEDGER_PATH).read_bytes()
    for label, record in (("pre", pre_ledger), ("post", post_ledger)):
        size = record["bytes"]
        prefix = ledger_bytes[:size]
        if (
            len(prefix) != size
            or hashlib.sha256(prefix).hexdigest() != record.get("sha256")
            or record.get("ledger_sha256") != record.get("sha256")
            or type(record.get("event_count")) is not int
            or int(record["event_count"]) <= 0
            or not isinstance(record.get("head_sha256"), str)
        ):
            raise InheritedAuthorizationError(
                f"transaction historical {label} ledger prefix drift"
            )
    if (
        post_ledger["event_count"] != pre_ledger["event_count"] + 1
        or post_ledger["bytes"] <= pre_ledger["bytes"]
        or pre_state.get("ledger_event_count") != pre_ledger["event_count"]
        or pre_state.get("ledger_head_sha256") != pre_ledger["head_sha256"]
        or post_state.get("ledger_event_count") != post_ledger["event_count"]
        or post_state.get("ledger_head_sha256") != post_ledger["head_sha256"]
        or len(events) < post_ledger["event_count"]
        or events[post_ledger["event_count"] - 1].get("record_sha256")
        != post_ledger["head_sha256"]
    ):
        raise InheritedAuthorizationError("transaction ledger/state chronology drift")
    forward_event = events[post_ledger["event_count"] - 1]
    expected_suffix = _canonical_bytes(forward_event) + b"\n"
    pre_without_source_marker = json.loads(json.dumps(pre_state))
    source_marker = pre_without_source_marker.pop(SOURCE_ADAPTER_STATE_KEY, None)
    verified_source_marker = _verify_source_controller_adapter_marker_value(
        source_marker,
        event_count=pre_ledger["event_count"],
        head_sha256=str(pre_ledger["head_sha256"]),
        state_without_marker=pre_without_source_marker,
    )
    if dict(verified_source_marker) != dict(expected_source_marker):
        raise InheritedAuthorizationError(
            "transaction source marker differs from inheritance seal"
        )
    if (
        ledger_bytes[pre_ledger["bytes"] : post_ledger["bytes"]]
        != expected_suffix
        or forward_event.get("event")
        != "zero_confirmation_outcome_version_forward"
        or forward_event.get(RECEIPT_CONTEXT_EVENT_FIELD)
        != receipt_context_sha256
        or forward_event.get(SOURCE_ADAPTER_MARKER_FIELD)
        != verified_source_marker
        or SOURCE_ADAPTER_STATE_KEY in post_state
    ):
        raise InheritedAuthorizationError(
            "transaction forward event/context commitment drift"
        )
    post_without_marker = json.loads(json.dumps(post_state))
    observed_post_marker = post_without_marker.pop(
        "v008_durable_controller_adapter", None
    )
    expected_operation_sha256 = hashlib.sha256(
        _canonical_bytes(
            {
                "base_state_object_sha256": hashlib.sha256(
                    _canonical_bytes(pre_state)
                ).hexdigest(),
                "base_ledger_sha256": pre_ledger["sha256"],
                "expected_suffix_sha256": hashlib.sha256(
                    expected_suffix
                ).hexdigest(),
                "intended_state_object_sha256": hashlib.sha256(
                    _canonical_bytes(post_without_marker)
                ).hexdigest(),
            }
        )
    ).hexdigest()
    expected_post_marker: dict[str, Any] = {
        "schema_version": 2,
        "authorization_kind": "receipt_bound_v008_root_controller_adapter",
        "adapter_source_path": fixed_records["transaction_source"]["path"],
        "adapter_source_sha256": fixed_records["transaction_source"]["sha256"],
        "adapter_source_ast_sha256": fixed_records["transaction_source"][
            "ast_sha256"
        ],
        "root_program_path": fixed_records["root_program"]["path"],
        "root_program_sha256": fixed_records["root_program"]["sha256"],
        "root_program_ast_sha256": fixed_records["root_program"][
            "ast_sha256"
        ],
        "ledger_event_count": post_ledger["event_count"],
        "ledger_head_sha256": post_ledger["head_sha256"],
        "operation_sha256": expected_operation_sha256,
    }
    expected_post_marker["state_binding_sha256"] = hashlib.sha256(
        _canonical_bytes(
            {
                "marker_without_state_binding": expected_post_marker,
                "state_without_marker": post_without_marker,
            }
        )
    ).hexdigest()
    if observed_post_marker != expected_post_marker:
        raise InheritedAuthorizationError(
            "transaction post-state adapter operation binding drift"
        )
    controller_return = receipt.get("controller_return")
    if (
        not isinstance(controller_return, Mapping)
        or dict(controller_return) != post_state
        or receipt.get("controller_return_sha256")
        != hashlib.sha256(_canonical_bytes(post_state)).hexdigest()
    ):
        raise InheritedAuthorizationError("transaction controller-return binding drift")
    lineage = post_state.get("version_forward_lineage")
    prior_lineage = pre_state.get("version_forward_lineage")
    if (
        not isinstance(lineage, list)
        or not isinstance(prior_lineage, list)
        or len(lineage) != 7
        or lineage[:-1] != prior_lineage
        or len(prior_lineage) != 6
        or lineage[-1].get("old_attempt") != ACTIVATION_SOURCE_ATTEMPT
        or lineage[-1].get("new_attempt") != ACTIVE_ATTEMPT
        or lineage[-1].get("resume_state") != "SELECTION_COHORTS"
        or lineage[-1].get("equivalence_path") != seal_relative
        or lineage[-1].get("equivalence_sha256") != seal_sha256
        or lineage[-1].get("invalidity_path") != invalidity_relative
        or lineage[-1].get("invalidity_sha256") != invalidity_sha256
    ):
        raise InheritedAuthorizationError("transaction post-state lineage drift")
    _verify_receipt_verifier_result(
        receipt.get("pre_verifiers"),
        phase="pre-forward",
        seal_relative=seal_relative,
        seal_sha256=seal_sha256,
        expected_projection=expected_verifier_projection,
    )
    _verify_receipt_verifier_result(
        receipt.get("post_verifiers"),
        phase="post-forward",
        seal_relative=seal_relative,
        seal_sha256=seal_sha256,
        expected_projection=expected_verifier_projection,
    )
    if (
        receipt["pre_verifiers"].get("independent_partitions")
        != receipt["post_verifiers"].get("independent_partitions")
        or receipt["pre_verifiers"]["producer"].get("state_sha256")
        != pre_snapshot["state"]["sha256"]
        or receipt["pre_verifiers"]["producer"].get("ledger_sha256")
        != pre_ledger["sha256"]
    ):
        raise InheritedAuthorizationError(
            "transaction verifier/snapshot cross-link drift"
        )
    context = receipt["receipt_context"]
    context_keys = {
        "schema_version",
        "source_attempt",
        "target_attempt",
        "resume_state",
        "journal_created_unix_ns",
        "forward_created_unix_ns",
        "receipt_created_unix_ns",
        "fixed_inputs",
        "pre_snapshot",
        "pre_verifiers",
        "controller_proposal",
        "predicted_post_verifiers",
    }
    context_header = {
        "schema": set(context) == context_keys
        and type(context.get("schema_version")) is int
        and context.get("schema_version") == 1,
        "source": context.get("source_attempt") == ACTIVATION_SOURCE_ATTEMPT,
        "target": context.get("target_attempt") == ACTIVE_ATTEMPT,
        "resume": context.get("resume_state") == "SELECTION_COHORTS",
        "timestamps": context.get("journal_created_unix_ns") == journal_created
        and context.get("forward_created_unix_ns") == forward_created
        and context.get("receipt_created_unix_ns") == receipt_created,
        "fixed": context.get("fixed_inputs")
        == {name: receipt[name] for name in fixed},
        "pre_snapshot": context.get("pre_snapshot") == pre_snapshot,
        "pre_verifiers": context.get("pre_verifiers")
        == receipt["pre_verifiers"],
        "post_verifiers": context.get("predicted_post_verifiers")
        == receipt["post_verifiers"],
    }
    if not all(context_header.values()):
        raise InheritedAuthorizationError(
            f"transaction precommit context drift: {context_header}"
        )
    proposal = context.get("controller_proposal")
    if (
        not isinstance(proposal, Mapping)
        or set(proposal) != {"state", "events"}
        or not isinstance(proposal.get("state"), Mapping)
        or not isinstance(proposal.get("events"), list)
        or len(proposal["events"]) != 1
        or not isinstance(proposal["events"][0], Mapping)
    ):
        raise InheritedAuthorizationError(
            "transaction controller proposal schema drift"
        )
    raw_event = dict(forward_event)
    for key in (
        "seq",
        "prev_sha256",
        "record_sha256",
        RECEIPT_CONTEXT_EVENT_FIELD,
    ):
        raw_event.pop(key, None)
    expected_raw_state = json.loads(json.dumps(post_state))
    expected_raw_state.pop("v008_durable_controller_adapter", None)
    expected_raw_state["ledger_event_count"] = pre_ledger["event_count"]
    expected_raw_state["ledger_head_sha256"] = pre_ledger["head_sha256"]
    if (
        proposal["events"] != [raw_event]
        or dict(proposal["state"]) != expected_raw_state
        or raw_event.get("created_unix_ns") != forward_created
        or proposal["state"].get("updated_unix_ns") != forward_created
    ):
        raise InheritedAuthorizationError(
            "transaction controller proposal/commit replay drift"
        )
    rebuilt_receipt = {
        "schema_version": 2,
        "artifact_type": (
            "atomic_zero_confirmation_outcome_version_forward_transaction_receipt"
        ),
        "authorization_kind": "locked_v007_to_v008_inherited_pre_selection_forward",
        "source_attempt": ACTIVATION_SOURCE_ATTEMPT,
        "target_attempt": ACTIVE_ATTEMPT,
        "resume_state": "SELECTION_COHORTS",
        "journal_created_unix_ns": journal_created,
        "forward_created_unix_ns": forward_created,
        "created_unix_ns": receipt_created,
        "receipt_context_sha256": receipt_context_sha256,
        "receipt_context": context,
        "seal": context["fixed_inputs"]["seal"],
        "invalidity": context["fixed_inputs"]["invalidity"],
        "invalidity_draft": context["fixed_inputs"]["invalidity_draft"],
        "prior_receipt": context["fixed_inputs"]["prior_receipt"],
        "root_program": context["fixed_inputs"]["root_program"],
        "transaction_source": context["fixed_inputs"]["transaction_source"],
        "pre_snapshot": context["pre_snapshot"],
        "pre_verifiers": context["pre_verifiers"],
        "controller_return": dict(controller_return),
        "controller_return_sha256": hashlib.sha256(
            _canonical_bytes(controller_return)
        ).hexdigest(),
        "post_snapshot": post_snapshot,
        "post_verifiers": context["predicted_post_verifiers"],
        "outcome_counts": {
            key: controller_return[key] for key in ZERO_COUNTS
        },
        "state_transaction_absent": True,
        "receipt_journal_absent": True,
        "passed": True,
    }
    if receipt != rebuilt_receipt:
        raise InheritedAuthorizationError(
            "transaction receipt differs from its precommit projection"
        )
    return {
        "path": _relative(receipt_path),
        "sha256": sha256_file(receipt_path),
        "receipt_context_sha256": receipt_context_sha256,
        "pre_state_sha256": pre_snapshot["state"]["sha256"],
        "post_state_sha256": post_snapshot["state"]["sha256"],
        "pre_ledger_sha256": pre_ledger["sha256"],
        "post_ledger_sha256": post_ledger["sha256"],
        "post_ledger_event_count": post_ledger["event_count"],
        "passed": True,
    }


def _checkpoint_projection(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_name": checkpoint.get("name"),
        "source_attempt": checkpoint.get("source_attempt"),
        "evidence_path": checkpoint.get("evidence_path"),
        "evidence_sha256": checkpoint.get("evidence_sha256"),
    }


def _verify_inherited_pre_data_authorization_locked(
    *,
    expected_current_state: str,
    state: Mapping[str, Any] | None = None,
    seal_path: Path = INHERITANCE_SEAL_PATH,
    ledger_path: Path = LEDGER_PATH,
    genesis_path: Path = LEDGER_GENESIS_PATH,
    authorized_early_state: str | None = None,
    authorized_role_count_recovery: str | None = None,
) -> dict[str, Any]:
    """Return an authenticated fit/selection-raw authorization record.

    Ordinary production callers obtain ``state`` through the root controller's
    read-only status command.  The sole early-terminal adapter supplies the
    root-held controller object through a module-private capability; the outer
    entry point requires it to equal the current ``STATE.json`` object/bytes.
    """

    if state is None:
        raise InheritedAuthorizationError(
            "locked inherited authorization requires an authenticated state snapshot"
        )
    controller = dict(state)
    current = controller.get("current_state")
    early_states = {
        "CANDIDATE_SELECTION",
        "CONFIRMATION_POWER_AND_COHORT_FREEZE",
    }
    expected_fit = (
        0
        if current == "FIT_COHORTS"
        and authorized_role_count_recovery is None
        else 1200
    )
    expected_selection = (
        2000
        if current in early_states
        or authorized_role_count_recovery == "selection"
        else 0
    )
    expected_attempt_path = _relative(ATTEMPT_ROOT)
    state_checks = {
        "active_attempt": controller.get("active_attempt") == ACTIVE_ATTEMPT,
        "active_attempt_path": controller.get("active_attempt_path")
        == expected_attempt_path,
        "current_state": current == expected_current_state,
        "terminal_unset": controller.get("terminal_label") is None,
        "confirmation_not_terminal": controller.get("confirmation_terminal") is False,
        "outcome_counts": (
            type(controller.get("fit_outcome_episodes")) is int
            and controller.get("fit_outcome_episodes") == expected_fit
            and type(controller.get("selection_outcome_episodes")) is int
            and controller.get("selection_outcome_episodes")
            == expected_selection
            and type(controller.get("smoke_outcome_episodes")) is int
            and controller.get("smoke_outcome_episodes") == 0
            and type(controller.get("confirmation_outcome_episodes_generated"))
            is int
            and controller.get("confirmation_outcome_episodes_generated") == 0
            and type(controller.get("confirmation_outcome_episodes_executed"))
            is int
            and controller.get("confirmation_outcome_episodes_executed") == 0
            and controller.get("confirmation_outcomes_opened_for_analysis")
            is False
        ),
        "completed_fit_size": (
            (
                current == "FIT_COHORTS"
                and authorized_role_count_recovery is None
            )
            or (
                type(controller.get("expected_fit_episode_count")) is int
                and controller.get("expected_fit_episode_count") == 1200
            )
        ),
        "completed_selection_size": (
            (
                current not in early_states
                and authorized_role_count_recovery != "selection"
            )
            or (
                type(controller.get("expected_selection_episode_count")) is int
                and controller.get("expected_selection_episode_count") == 2000
            )
        ),
    }
    if not all(state_checks.values()):
        raise InheritedAuthorizationError(
            f"controller does not authorize inherited source use: {state_checks}"
        )
    ordinary_role = {
        "FIT_COHORTS": "fit",
        "SELECTION_COHORTS": "selection",
    }.get(str(current))
    if (
        authorized_early_state is None
        and authorized_role_count_recovery is None
        and ordinary_role is not None
        and os.path.lexists(
            ATTEMPT_ROOT / "audit" / f"{ordinary_role}_cohorts.json"
        )
    ):
        raise InheritedAuthorizationError(
            "ordinary role authorization found a pre-count role audit"
        )

    seal_path = _regular_file(seal_path).resolve()
    seal_relative = _relative(seal_path)
    seal_sha256 = sha256_file(seal_path)
    seal = _read_json(seal_path)
    _validate_inheritance_seal_scalar_types(seal)
    header = {
        "schema": type(seal.get("schema_version")) is int
        and seal.get("schema_version") == 1,
        "attempt": seal.get("attempt") == ACTIVE_ATTEMPT,
        "source": seal.get("source_attempt") == ACTIVATION_SOURCE_ATTEMPT,
        "target": seal.get("target_attempt") == ACTIVE_ATTEMPT,
        "science": seal.get("science_attempt") == SCIENCE_ATTEMPT,
        "checkpoint": seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL",
        "resume": seal.get("resume_state") == "FIT_COHORTS",
        "kind": seal.get("authorization_kind")
        == "zero_outcome_version_forward_inherited_pre_data",
        "passed": seal.get("passed") is True,
        "invalidity": seal.get("procedural_invalidity_confirmed") is True,
        "zero_confirmation": seal.get("zero_confirmation_outcomes_at_version_forward")
        is True,
        **{
            key: seal.get(key) is expected
            for key, expected in REQUIRED_EQUIVALENCE.items()
        },
    }
    if not all(header.values()):
        raise InheritedAuthorizationError(
            f"inheritance seal header/equivalence drift: {header}"
        )
    counts = seal.get("outcome_counts_at_seal")
    if not _exact_typed_mapping(counts, ZERO_COUNTS):
        raise InheritedAuthorizationError("inheritance seal outcome counts are not exact zero")
    if not isinstance(seal.get("source_controller_adapter_marker"), Mapping):
        raise InheritedAuthorizationError(
            "inheritance seal lacks the authenticated v005 source marker"
        )
    sealed_files = _verify_sealed_files(seal.get("sealed_files"))
    expected_verifier_projection = _active_seal_verifier_projection(
        seal, sealed_files
    )
    invalidity_link = _file_link(
        seal.get("invalidity_evidence"), expected_path=INVALIDITY_PATH
    )
    source_seal_link = _file_link(
        seal.get("source_pre_data_seal"), expected_path=SOURCE_PRE_DATA_SEAL_PATH
    )
    raw_invalidity_draft = seal.get("superseded_invalidity_draft")
    if (
        not isinstance(raw_invalidity_draft, Mapping)
        or set(raw_invalidity_draft) != {"path", "sha256", "authoritative"}
    ):
        raise InheritedAuthorizationError(
            "superseded v005 invalidity draft link schema drift"
        )
    invalidity_draft_link = _file_link(
        {
            "path": raw_invalidity_draft.get("path"),
            "sha256": raw_invalidity_draft.get("sha256"),
        },
        expected_path=INVALIDITY_DRAFT_PATH,
    )
    source_receipt_link = _file_link(
        seal.get("source_version_forward_transaction_receipt"),
        expected_path=SOURCE_TRANSACTION_RECEIPT_PATH,
    )
    if raw_invalidity_draft.get("authoritative") is not False:
        raise InheritedAuthorizationError(
            "superseded v005 invalidity draft became authoritative"
        )
    invalidity = _read_json(INVALIDITY_PATH)
    invalidity_checks = {
        "attempt": invalidity.get("attempt") == SOURCE_ATTEMPT,
        "procedural": invalidity.get("procedural_invalidity") is True,
        "authoritative": invalidity.get("authoritative") is True,
        "unconditioned": invalidity.get("scientific_outcomes_conditioned_on")
        is False,
        "generated": type(
            invalidity.get("confirmation_outcome_episodes_generated")
        )
        is int
        and invalidity.get("confirmation_outcome_episodes_generated") == 0,
        "executed": type(
            invalidity.get("confirmation_outcome_episodes_executed")
        )
        is int
        and invalidity.get("confirmation_outcome_episodes_executed") == 0,
        "unopened": invalidity.get("confirmation_outcomes_opened_for_analysis") is False,
    }
    if not all(invalidity_checks.values()):
        raise InheritedAuthorizationError(
            f"v005 invalidity evidence drift: {invalidity_checks}"
        )
    prior_draft = invalidity.get("superseded_draft")
    if (
        not isinstance(prior_draft, Mapping)
        or prior_draft.get("authoritative") is not False
        or prior_draft.get("path") != invalidity_draft_link["path"]
        or prior_draft.get("sha256") != invalidity_draft_link["sha256"]
    ):
        raise InheritedAuthorizationError(
            "authoritative v005 invalidity lost its draft cross-link"
        )
    source_seal = _read_json(SOURCE_PRE_DATA_SEAL_PATH)
    source_seal_checks = {
        "attempt": source_seal.get("attempt") == SOURCE_ATTEMPT,
        "source": source_seal.get("source_attempt") == INTERMEDIATE_ATTEMPT,
        "target": source_seal.get("target_attempt") == SOURCE_ATTEMPT,
        "science": source_seal.get("science_attempt") == SCIENCE_ATTEMPT,
        "checkpoint": source_seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL",
        "passed": source_seal.get("passed") is True,
        "zero": _exact_typed_mapping(
            source_seal.get("outcome_counts_at_seal"), ZERO_COUNTS
        ),
    }
    if not all(source_seal_checks.values()):
        raise InheritedAuthorizationError(
            f"source pre-data seal drift: {source_seal_checks}"
        )
    intermediate_pre_data_seal_link = _file_link(
        source_seal.get("source_pre_data_seal"),
        expected_path=INTERMEDIATE_PRE_DATA_SEAL_PATH,
    )
    intermediate_pre_data_seal = _read_json(INTERMEDIATE_PRE_DATA_SEAL_PATH)
    intermediate_pre_data_seal_checks = {
        "attempt": intermediate_pre_data_seal.get("attempt")
        == INTERMEDIATE_ATTEMPT,
        "source": intermediate_pre_data_seal.get("source_attempt")
        == FIT_SOURCE_ATTEMPT,
        "target": intermediate_pre_data_seal.get("target_attempt")
        == INTERMEDIATE_ATTEMPT,
        "science": intermediate_pre_data_seal.get("science_attempt")
        == SCIENCE_ATTEMPT,
        "checkpoint": intermediate_pre_data_seal.get("checkpoint_state")
        == "PRE_OUTCOME_SEAL",
        "passed": intermediate_pre_data_seal.get("passed") is True,
        "zero": _exact_typed_mapping(
            intermediate_pre_data_seal.get("outcome_counts_at_seal"), ZERO_COUNTS
        ),
    }
    if not all(intermediate_pre_data_seal_checks.values()):
        raise InheritedAuthorizationError(
            "intermediate pre-data seal drift: "
            f"{intermediate_pre_data_seal_checks}"
        )
    fit_pre_data_seal_link = _file_link(
        intermediate_pre_data_seal.get("source_pre_data_seal"),
        expected_path=FIT_PRE_DATA_SEAL_PATH,
    )
    fit_pre_data_seal = _read_json(FIT_PRE_DATA_SEAL_PATH)
    fit_pre_data_seal_checks = {
        "attempt": fit_pre_data_seal.get("attempt") == FIT_SOURCE_ATTEMPT,
        "source": fit_pre_data_seal.get("source_attempt") == PRIOR_ATTEMPT,
        "target": fit_pre_data_seal.get("target_attempt") == FIT_SOURCE_ATTEMPT,
        "science": fit_pre_data_seal.get("science_attempt") == SCIENCE_ATTEMPT,
        "checkpoint": fit_pre_data_seal.get("checkpoint_state")
        == "PRE_OUTCOME_SEAL",
        "passed": fit_pre_data_seal.get("passed") is True,
        "zero": _exact_typed_mapping(
            fit_pre_data_seal.get("outcome_counts_at_seal"), ZERO_COUNTS
        ),
    }
    if not all(fit_pre_data_seal_checks.values()):
        raise InheritedAuthorizationError(
            f"fit-source pre-data seal drift: {fit_pre_data_seal_checks}"
        )
    prior_pre_data_seal_link = _file_link(
        fit_pre_data_seal.get("source_pre_data_seal"),
        expected_path=PRIOR_PRE_DATA_SEAL_PATH,
    )
    prior_pre_data_seal = _read_json(PRIOR_PRE_DATA_SEAL_PATH)
    prior_pre_data_seal_checks = {
        "attempt": prior_pre_data_seal.get("attempt") == PRIOR_ATTEMPT,
        "source": prior_pre_data_seal.get("source_attempt") == SCIENCE_ATTEMPT,
        "target": prior_pre_data_seal.get("target_attempt") == PRIOR_ATTEMPT,
        "science": prior_pre_data_seal.get("science_attempt") == SCIENCE_ATTEMPT,
        "checkpoint": prior_pre_data_seal.get("checkpoint_state")
        == "PRE_OUTCOME_SEAL",
        "passed": prior_pre_data_seal.get("passed") is True,
        "zero": _exact_typed_mapping(
            prior_pre_data_seal.get("outcome_counts_at_seal"), ZERO_COUNTS
        ),
    }
    if not all(prior_pre_data_seal_checks.values()):
        raise InheritedAuthorizationError(
            f"prior pre-data seal drift: {prior_pre_data_seal_checks}"
        )
    original_pre_data_seal_path = (
        STUDY_ROOT / "attempts/v001/audit/pre_data_seal.json"
    )
    original_invalidity_path = (
        STUDY_ROOT / "attempts/v001/audit/v001_procedural_invalidity.json"
    )
    original_pre_data_seal_link = _file_link(
        prior_pre_data_seal.get("source_pre_data_seal"),
        expected_path=original_pre_data_seal_path,
    )
    original_invalidity_link = _file_link(
        prior_pre_data_seal.get("invalidity_evidence"),
        expected_path=original_invalidity_path,
    )
    prior_invalidity_link = _file_link(
        fit_pre_data_seal.get("invalidity_evidence"),
        expected_path=PRIOR_INVALIDITY_PATH,
    )
    fit_invalidity_link = _file_link(
        intermediate_pre_data_seal.get("invalidity_evidence"),
        expected_path=FIT_INVALIDITY_PATH,
    )
    intermediate_invalidity_link = _file_link(
        source_seal.get("invalidity_evidence"),
        expected_path=INTERMEDIATE_INVALIDITY_PATH,
    )
    source_receipt = _read_json(SOURCE_TRANSACTION_RECEIPT_PATH)
    if (
        source_receipt.get("source_attempt") != INTERMEDIATE_ATTEMPT
        or source_receipt.get("target_attempt") != SOURCE_ATTEMPT
        or source_receipt.get("passed") is not True
        or source_receipt_link["sha256"]
        != sha256_file(SOURCE_TRANSACTION_RECEIPT_PATH)
    ):
        raise InheritedAuthorizationError("v005 source transaction receipt drift")
    fit_receipt_link = _file_link(
        intermediate_pre_data_seal.get(
            "source_version_forward_transaction_receipt"
        ),
        expected_path=FIT_TRANSACTION_RECEIPT_PATH,
    )
    fit_receipt = _read_json(FIT_TRANSACTION_RECEIPT_PATH)
    if (
        fit_receipt.get("source_attempt") != PRIOR_ATTEMPT
        or fit_receipt.get("target_attempt") != FIT_SOURCE_ATTEMPT
        or fit_receipt.get("passed") is not True
        or fit_receipt_link["sha256"]
        != sha256_file(FIT_TRANSACTION_RECEIPT_PATH)
    ):
        raise InheritedAuthorizationError("v003 fit-source transaction receipt drift")
    intermediate_receipt_link = _file_link(
        source_seal.get("source_version_forward_transaction_receipt"),
        expected_path=INTERMEDIATE_TRANSACTION_RECEIPT_PATH,
    )
    intermediate_receipt = _read_json(INTERMEDIATE_TRANSACTION_RECEIPT_PATH)
    if (
        intermediate_receipt.get("source_attempt") != FIT_SOURCE_ATTEMPT
        or intermediate_receipt.get("target_attempt") != INTERMEDIATE_ATTEMPT
        or intermediate_receipt.get("passed") is not True
        or intermediate_receipt_link["sha256"]
        != sha256_file(INTERMEDIATE_TRANSACTION_RECEIPT_PATH)
    ):
        raise InheritedAuthorizationError(
            "v004 intermediate transaction receipt drift"
        )
    prior_receipt_link = _file_link(
        fit_pre_data_seal.get("source_version_forward_transaction_receipt"),
        expected_path=PRIOR_TRANSACTION_RECEIPT_PATH,
    )
    prior_receipt = _read_json(PRIOR_TRANSACTION_RECEIPT_PATH)
    if (
        prior_receipt.get("source_attempt") != SCIENCE_ATTEMPT
        or prior_receipt.get("target_attempt") != PRIOR_ATTEMPT
        or prior_receipt.get("passed") is not True
        or prior_receipt_link["sha256"]
        != sha256_file(PRIOR_TRANSACTION_RECEIPT_PATH)
    ):
        raise InheritedAuthorizationError("v002 transaction receipt drift")

    history = controller.get("attempt_history")
    if (
        not isinstance(history, list)
        or [item.get("version") for item in history]
        != [
            SCIENCE_ATTEMPT,
            PRIOR_ATTEMPT,
            FIT_SOURCE_ATTEMPT,
            INTERMEDIATE_ATTEMPT,
            SOURCE_ATTEMPT,
            ACTIVE_ATTEMPT,
        ]
        or "invalid_zero_confirmation_outcome_procedural"
        != history[0].get("status")
        or history[0].get("invalidity_evidence_path")
        != original_invalidity_link["path"]
        or history[0].get("invalidity_evidence_sha256")
        != original_invalidity_link["sha256"]
        or history[1].get("status")
        != "invalid_zero_confirmation_outcome_procedural"
        or history[1].get("invalidity_evidence_path")
        != prior_invalidity_link["path"]
        or history[1].get("invalidity_evidence_sha256")
        != prior_invalidity_link["sha256"]
        or history[1].get("version_forward_evidence_path")
        != prior_pre_data_seal_link["path"]
        or history[1].get("version_forward_evidence_sha256")
        != prior_pre_data_seal_link["sha256"]
        or history[1].get("attempt_parameterization_verified") is not True
        or history[2].get("status")
        != "invalid_zero_confirmation_outcome_procedural"
        or history[2].get("invalidity_evidence_path")
        != fit_invalidity_link["path"]
        or history[2].get("invalidity_evidence_sha256")
        != fit_invalidity_link["sha256"]
        or history[2].get("version_forward_evidence_path")
        != fit_pre_data_seal_link["path"]
        or history[2].get("version_forward_evidence_sha256")
        != fit_pre_data_seal_link["sha256"]
        or history[2].get("attempt_parameterization_verified") is not True
        or history[3].get("status")
        != "invalid_zero_confirmation_outcome_procedural"
        or history[3].get("invalidity_evidence_path")
        != intermediate_invalidity_link["path"]
        or history[3].get("invalidity_evidence_sha256")
        != intermediate_invalidity_link["sha256"]
        or history[3].get("version_forward_evidence_path")
        != intermediate_pre_data_seal_link["path"]
        or history[3].get("version_forward_evidence_sha256")
        != intermediate_pre_data_seal_link["sha256"]
        or history[3].get("attempt_parameterization_verified") is not True
        or history[4].get("status")
        != "invalid_zero_confirmation_outcome_procedural"
        or history[4].get("invalidity_evidence_path") != invalidity_link["path"]
        or history[4].get("invalidity_evidence_sha256")
        != invalidity_link["sha256"]
        or history[4].get("version_forward_evidence_path")
        != source_seal_link["path"]
        or history[4].get("version_forward_evidence_sha256")
        != source_seal_link["sha256"]
        or history[4].get("attempt_parameterization_verified") is not True
        or history[5].get("status")
        != "active_zero_confirmation_outcome_version_forward"
        or history[5].get("version_forward_evidence_path") != seal_relative
        or history[5].get("version_forward_evidence_sha256") != seal_sha256
        or history[5].get("attempt_parameterization_verified") is not True
    ):
        raise InheritedAuthorizationError("attempt-history version-forward lineage drift")

    lineage = controller.get("version_forward_lineage")
    prior_lineage = seal.get("prior_version_forward_lineage")
    if (
        not isinstance(lineage, list)
        or len(lineage) != 5
        or not all(isinstance(item, Mapping) for item in lineage)
        or not isinstance(prior_lineage, list)
        or len(prior_lineage) != 4
        or lineage[:-1] != prior_lineage
    ):
        raise InheritedAuthorizationError("STATE version-forward lineage prefix drift")
    oldest_edge = lineage[0]
    oldest_edge_checks = {
        "source": oldest_edge.get("old_attempt") == SCIENCE_ATTEMPT,
        "target": oldest_edge.get("new_attempt") == PRIOR_ATTEMPT,
        "resume": oldest_edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": oldest_edge.get("invalidity_path")
        == original_invalidity_link["path"],
        "invalidity_hash": oldest_edge.get("invalidity_sha256")
        == original_invalidity_link["sha256"],
        "seal_path": oldest_edge.get("equivalence_path")
        == prior_pre_data_seal_link["path"],
        "seal_hash": oldest_edge.get("equivalence_sha256")
        == prior_pre_data_seal_link["sha256"],
        "parameterized": oldest_edge.get("attempt_parameterization_verified")
        is True,
    }
    if not all(oldest_edge_checks.values()):
        raise InheritedAuthorizationError(
            f"oldest STATE version-forward edge drift: {oldest_edge_checks}"
        )
    fit_edge = lineage[1]
    fit_edge_checks = {
        "source": fit_edge.get("old_attempt") == PRIOR_ATTEMPT,
        "target": fit_edge.get("new_attempt") == FIT_SOURCE_ATTEMPT,
        "resume": fit_edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": fit_edge.get("invalidity_path")
        == prior_invalidity_link["path"],
        "invalidity_hash": fit_edge.get("invalidity_sha256")
        == prior_invalidity_link["sha256"],
        "seal_path": fit_edge.get("equivalence_path")
        == fit_pre_data_seal_link["path"],
        "seal_hash": fit_edge.get("equivalence_sha256")
        == fit_pre_data_seal_link["sha256"],
        "parameterized": fit_edge.get("attempt_parameterization_verified")
        is True,
    }
    if not all(fit_edge_checks.values()):
        raise InheritedAuthorizationError(
            f"fit-source STATE version-forward edge drift: {fit_edge_checks}"
        )
    intermediate_edge = lineage[2]
    intermediate_edge_checks = {
        "source": intermediate_edge.get("old_attempt") == FIT_SOURCE_ATTEMPT,
        "target": intermediate_edge.get("new_attempt") == INTERMEDIATE_ATTEMPT,
        "resume": intermediate_edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": intermediate_edge.get("invalidity_path")
        == fit_invalidity_link["path"],
        "invalidity_hash": intermediate_edge.get("invalidity_sha256")
        == fit_invalidity_link["sha256"],
        "seal_path": intermediate_edge.get("equivalence_path")
        == intermediate_pre_data_seal_link["path"],
        "seal_hash": intermediate_edge.get("equivalence_sha256")
        == intermediate_pre_data_seal_link["sha256"],
        "parameterized": intermediate_edge.get("attempt_parameterization_verified")
        is True,
    }
    if not all(intermediate_edge_checks.values()):
        raise InheritedAuthorizationError(
            "intermediate STATE version-forward edge drift: "
            f"{intermediate_edge_checks}"
        )
    source_edge = lineage[3]
    source_edge_checks = {
        "source": source_edge.get("old_attempt") == INTERMEDIATE_ATTEMPT,
        "target": source_edge.get("new_attempt") == SOURCE_ATTEMPT,
        "resume": source_edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": source_edge.get("invalidity_path")
        == intermediate_invalidity_link["path"],
        "invalidity_hash": source_edge.get("invalidity_sha256")
        == intermediate_invalidity_link["sha256"],
        "seal_path": source_edge.get("equivalence_path")
        == source_seal_link["path"],
        "seal_hash": source_edge.get("equivalence_sha256")
        == source_seal_link["sha256"],
        "parameterized": source_edge.get("attempt_parameterization_verified")
        is True,
    }
    if not all(source_edge_checks.values()):
        raise InheritedAuthorizationError(
            f"source STATE version-forward edge drift: {source_edge_checks}"
        )
    edge = lineage[-1]
    edge_checks = {
        "source": edge.get("old_attempt") == SOURCE_ATTEMPT,
        "target": edge.get("new_attempt") == ACTIVE_ATTEMPT,
        "resume": edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": edge.get("invalidity_path") == invalidity_link["path"],
        "invalidity_hash": edge.get("invalidity_sha256") == invalidity_link["sha256"],
        "seal_path": edge.get("equivalence_path") == seal_relative,
        "seal_hash": edge.get("equivalence_sha256") == seal_sha256,
        "parameterized": edge.get("attempt_parameterization_verified") is True,
    }
    if not all(edge_checks.values()):
        raise InheritedAuthorizationError(f"STATE version-forward edge drift: {edge_checks}")

    completed = controller.get("completed_states")
    checkpoints = controller.get("verified_checkpoints")
    if not isinstance(completed, list) or not isinstance(checkpoints, list) or len(completed) != len(checkpoints):
        raise InheritedAuthorizationError("controller checkpoint chronology schema drift")
    if completed[: len(INHERITED_STATES)] != list(INHERITED_STATES):
        raise InheritedAuthorizationError("inherited completed-state prefix drift")
    inherited = checkpoints[: len(INHERITED_STATES)]
    prior_annotation = {
        "attempt": PRIOR_ATTEMPT,
        "equivalence_evidence_path": prior_pre_data_seal_link["path"],
        "equivalence_evidence_sha256": prior_pre_data_seal_link["sha256"],
    }
    fit_annotation = {
        "attempt": FIT_SOURCE_ATTEMPT,
        "equivalence_evidence_path": fit_pre_data_seal_link["path"],
        "equivalence_evidence_sha256": fit_pre_data_seal_link["sha256"],
    }
    intermediate_annotation = {
        "attempt": INTERMEDIATE_ATTEMPT,
        "equivalence_evidence_path": intermediate_pre_data_seal_link["path"],
        "equivalence_evidence_sha256": intermediate_pre_data_seal_link["sha256"],
    }
    source_annotation = {
        "attempt": SOURCE_ATTEMPT,
        "equivalence_evidence_path": source_seal_link["path"],
        "equivalence_evidence_sha256": source_seal_link["sha256"],
    }
    active_annotation = {
        "attempt": ACTIVE_ATTEMPT,
        "equivalence_evidence_path": seal_relative,
        "equivalence_evidence_sha256": seal_sha256,
    }
    for checkpoint in inherited:
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("source_attempt") != SCIENCE_ATTEMPT
            or checkpoint.get("verification_lineage") != "direct_checkpoint"
            or checkpoint.get("inherited_into_attempts")
            != [
                prior_annotation,
                fit_annotation,
                intermediate_annotation,
                source_annotation,
                active_annotation,
            ]
        ):
            raise InheritedAuthorizationError("inherited checkpoint annotation drift")
        evidence = _resolve_relative(
            checkpoint.get("evidence_path"),
            required_root=STUDY_ROOT / "attempts/v001",
        )
        if sha256_file(_regular_file(evidence)) != checkpoint.get("evidence_sha256"):
            raise InheritedAuthorizationError("inherited checkpoint evidence hash drift")
    source_checkpoint = inherited[-1]
    if (
        source_checkpoint.get("evidence_path")
        != original_pre_data_seal_link["path"]
        or source_checkpoint.get("evidence_sha256")
        != original_pre_data_seal_link["sha256"]
    ):
        raise InheritedAuthorizationError("inherited PRE_OUTCOME_SEAL checkpoint drift")
    projection = [_checkpoint_projection(item) for item in inherited]
    if (
        oldest_edge.get("inherited_verified_checkpoints") != projection
        or fit_edge.get("inherited_verified_checkpoints") != projection
        or intermediate_edge.get("inherited_verified_checkpoints") != projection
        or source_edge.get("inherited_verified_checkpoints") != projection
        or edge.get("inherited_verified_checkpoints") != projection
        or seal.get("inherited_verified_checkpoints") != projection
    ):
        raise InheritedAuthorizationError("STATE inherited-checkpoint projection drift")
    direct_states = [
        completed[index]
        for index, checkpoint in enumerate(checkpoints)
        if checkpoint.get("source_attempt") == ACTIVE_ATTEMPT
    ]
    if direct_states and direct_states[0] != "FIT_COHORTS":
        raise InheritedAuthorizationError("first direct v008 checkpoint is not FIT_COHORTS")

    events = _verify_ledger_chain(
        ledger_path=ledger_path, genesis_path=genesis_path, state=controller
    )
    controller_adapter = _verify_controller_adapter_marker(
        controller,
        events=events,
    )
    adapter_bound_count_events = _verify_adapter_count_event_bindings(events)
    role_count_recovery: dict[str, Any] | None = None
    if authorized_role_count_recovery is not None:
        if authorized_early_state is not None:
            raise InheritedAuthorizationError(
                "early and role-count recovery authorizations are mutually exclusive"
            )
        role_count_recovery = _verify_role_count_recovery_authorization(
            role=authorized_role_count_recovery,
            controller=controller,
            events=events,
        )
    edge_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "zero_confirmation_outcome_version_forward"
    ]
    if len(edge_indexes) != 5:
        raise InheritedAuthorizationError(
            "ledger does not contain the exact five version-forward edges"
        )
    (
        oldest_edge_index,
        fit_edge_index,
        intermediate_edge_index,
        source_edge_index,
        edge_index,
    ) = edge_indexes
    oldest_ledger_edge = events[oldest_edge_index]
    if (
        oldest_ledger_edge.get("attempt") != PRIOR_ATTEMPT
        or oldest_ledger_edge.get("old_attempt") != SCIENCE_ATTEMPT
        or oldest_ledger_edge.get("new_attempt") != PRIOR_ATTEMPT
        or oldest_ledger_edge.get("resume_state") != "FIT_COHORTS"
        or oldest_ledger_edge.get("invalidity_path")
        != original_invalidity_link["path"]
        or oldest_ledger_edge.get("invalidity_sha256")
        != original_invalidity_link["sha256"]
        or oldest_ledger_edge.get("equivalence_path")
        != prior_pre_data_seal_link["path"]
        or oldest_ledger_edge.get("equivalence_sha256")
        != prior_pre_data_seal_link["sha256"]
        or oldest_ledger_edge.get("inherited_verified_checkpoints") != projection
        or oldest_ledger_edge.get("attempt_parameterization_verified") is not True
    ):
        raise InheritedAuthorizationError("oldest ledger version-forward edge drift")
    fit_ledger_edge = events[fit_edge_index]
    fit_ledger_checks = {
        "attempt": fit_ledger_edge.get("attempt") == FIT_SOURCE_ATTEMPT,
        "source": fit_ledger_edge.get("old_attempt") == PRIOR_ATTEMPT,
        "target": fit_ledger_edge.get("new_attempt") == FIT_SOURCE_ATTEMPT,
        "resume": fit_ledger_edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": fit_ledger_edge.get("invalidity_path")
        == prior_invalidity_link["path"],
        "invalidity_hash": fit_ledger_edge.get("invalidity_sha256")
        == prior_invalidity_link["sha256"],
        "seal_path": fit_ledger_edge.get("equivalence_path")
        == fit_pre_data_seal_link["path"],
        "seal_hash": fit_ledger_edge.get("equivalence_sha256")
        == fit_pre_data_seal_link["sha256"],
        "projection": fit_ledger_edge.get("inherited_verified_checkpoints")
        == projection,
        "parameterized": fit_ledger_edge.get(
            "attempt_parameterization_verified"
        )
        is True,
    }
    if not all(fit_ledger_checks.values()):
        raise InheritedAuthorizationError(
            f"fit-source ledger version-forward edge drift: {fit_ledger_checks}"
        )
    intermediate_ledger_edge = events[intermediate_edge_index]
    intermediate_ledger_checks = {
        "attempt": intermediate_ledger_edge.get("attempt")
        == INTERMEDIATE_ATTEMPT,
        "source": intermediate_ledger_edge.get("old_attempt")
        == FIT_SOURCE_ATTEMPT,
        "target": intermediate_ledger_edge.get("new_attempt")
        == INTERMEDIATE_ATTEMPT,
        "resume": intermediate_ledger_edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": intermediate_ledger_edge.get("invalidity_path")
        == fit_invalidity_link["path"],
        "invalidity_hash": intermediate_ledger_edge.get("invalidity_sha256")
        == fit_invalidity_link["sha256"],
        "seal_path": intermediate_ledger_edge.get("equivalence_path")
        == intermediate_pre_data_seal_link["path"],
        "seal_hash": intermediate_ledger_edge.get("equivalence_sha256")
        == intermediate_pre_data_seal_link["sha256"],
        "projection": intermediate_ledger_edge.get(
            "inherited_verified_checkpoints"
        )
        == projection,
        "parameterized": intermediate_ledger_edge.get(
            "attempt_parameterization_verified"
        )
        is True,
    }
    if not all(intermediate_ledger_checks.values()):
        raise InheritedAuthorizationError(
            "intermediate ledger version-forward edge drift: "
            f"{intermediate_ledger_checks}"
        )
    source_ledger_edge = events[source_edge_index]
    source_ledger_checks = {
        "attempt": source_ledger_edge.get("attempt") == SOURCE_ATTEMPT,
        "source": source_ledger_edge.get("old_attempt") == INTERMEDIATE_ATTEMPT,
        "target": source_ledger_edge.get("new_attempt") == SOURCE_ATTEMPT,
        "resume": source_ledger_edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": source_ledger_edge.get("invalidity_path")
        == intermediate_invalidity_link["path"],
        "invalidity_hash": source_ledger_edge.get("invalidity_sha256")
        == intermediate_invalidity_link["sha256"],
        "seal_path": source_ledger_edge.get("equivalence_path")
        == source_seal_link["path"],
        "seal_hash": source_ledger_edge.get("equivalence_sha256")
        == source_seal_link["sha256"],
        "projection": source_ledger_edge.get("inherited_verified_checkpoints")
        == projection,
        "parameterized": source_ledger_edge.get(
            "attempt_parameterization_verified"
        )
        is True,
    }
    if not all(source_ledger_checks.values()):
        raise InheritedAuthorizationError(
            f"source ledger version-forward edge drift: {source_ledger_checks}"
        )
    ledger_edge = events[edge_index]
    ledger_checks = {
        "attempt": ledger_edge.get("attempt") == ACTIVE_ATTEMPT,
        "source": ledger_edge.get("old_attempt") == SOURCE_ATTEMPT,
        "target": ledger_edge.get("new_attempt") == ACTIVE_ATTEMPT,
        "resume": ledger_edge.get("resume_state") == "FIT_COHORTS",
        "invalidity_path": ledger_edge.get("invalidity_path") == invalidity_link["path"],
        "invalidity_hash": ledger_edge.get("invalidity_sha256") == invalidity_link["sha256"],
        "seal_path": ledger_edge.get("equivalence_path") == seal_relative,
        "seal_hash": ledger_edge.get("equivalence_sha256") == seal_sha256,
        "projection": ledger_edge.get("inherited_verified_checkpoints") == projection,
        "parameterized": ledger_edge.get("attempt_parameterization_verified") is True,
    }
    if not all(ledger_checks.values()):
        raise InheritedAuthorizationError(f"ledger version-forward edge drift: {ledger_checks}")
    if any(
        event.get("attempt") != SCIENCE_ATTEMPT
        for event in events[:oldest_edge_index]
    ):
        raise InheritedAuthorizationError("pre-v002 ledger event attempt drift")
    if any(
        event.get("attempt") != PRIOR_ATTEMPT
        for event in events[oldest_edge_index:fit_edge_index]
    ):
        raise InheritedAuthorizationError("v002 ledger event attempt drift")
    if any(
        event.get("attempt") != FIT_SOURCE_ATTEMPT
        for event in events[fit_edge_index:intermediate_edge_index]
    ):
        raise InheritedAuthorizationError("v003 ledger event attempt drift")
    if any(
        event.get("attempt") != INTERMEDIATE_ATTEMPT
        for event in events[intermediate_edge_index:source_edge_index]
    ):
        raise InheritedAuthorizationError("v004 ledger event attempt drift")
    if any(
        event.get("attempt") != SOURCE_ATTEMPT
        for event in events[source_edge_index:edge_index]
    ):
        raise InheritedAuthorizationError("v005 ledger event attempt drift")
    if any(event.get("attempt") != ACTIVE_ATTEMPT for event in events[edge_index:]):
        raise InheritedAuthorizationError("post-forward ledger event attempt drift")

    transaction_receipt = _verify_version_forward_transaction_receipt(
        seal_relative=seal_relative,
        seal_sha256=seal_sha256,
        invalidity_relative=invalidity_link["path"],
        invalidity_sha256=invalidity_link["sha256"],
        expected_source_marker=seal.get("source_controller_adapter_marker"),
        events=events,
        expected_verifier_projection=expected_verifier_projection,
    )

    independent = _verify_post_forward_recomputation(
        seal_path,
        authorized_early_state=authorized_early_state,
        authorized_role_count_recovery=authorized_role_count_recovery,
        expected_partition_file_count=expected_verifier_projection[
            "partition_file_count"
        ],
        expected_sealed_file_count=expected_verifier_projection[
            "sealed_file_count"
        ],
    )

    return {
        "passed": True,
        "active_attempt": ACTIVE_ATTEMPT,
        "science_attempt": SCIENCE_ATTEMPT,
        "state": expected_current_state,
        "seal_path": seal_relative,
        "seal_sha256": seal_sha256,
        "seal_checkpoint_state": "PRE_OUTCOME_SEAL",
        "invalidity_path": invalidity_link["path"],
        "invalidity_sha256": invalidity_link["sha256"],
        "source_pre_data_seal_path": source_seal_link["path"],
        "source_pre_data_seal_sha256": source_seal_link["sha256"],
        "inherited_checkpoint_count": len(inherited),
        "sealed_file_count": len(sealed_files),
        "version_forward_ledger_sequence": int(ledger_edge["seq"]),
        "version_forward_transaction_receipt": transaction_receipt,
        "controller_adapter_authorization": controller_adapter,
        "adapter_bound_count_event_count": adapter_bound_count_events,
        "role_count_recovery_authorization": role_count_recovery,
        "independent_recomputation": {
            "passed": True,
            "seal_path": independent["seal_path"],
            "seal_sha256": independent["seal_sha256"],
            "active_attempt": independent["active_attempt"],
            "science_attempt": independent["science_attempt"],
            "authorized_early_verifier_state": independent[
                "authorized_early_verifier_state"
            ],
            "authorized_role_count_recovery": independent[
                "authorized_role_count_recovery"
            ],
            "output_paths_created": independent["output_paths_created"],
        },
    }


def _verify_inherited_pre_data_authorization_locked(
    *,
    expected_current_state: str,
    state: Mapping[str, Any] | None = None,
    seal_path: Path = INHERITANCE_SEAL_PATH,
    ledger_path: Path = LEDGER_PATH,
    genesis_path: Path = LEDGER_GENESIS_PATH,
    authorized_early_state: str | None = None,
    authorized_role_count_recovery: str | None = None,
) -> dict[str, Any]:
    """Authenticate the v008 selection-boundary lineage and receipt."""

    if state is None:
        raise InheritedAuthorizationError(
            "locked inherited authorization requires an authenticated state snapshot"
        )
    controller = dict(state)
    current = controller.get("current_state")
    early_states = {
        "CANDIDATE_SELECTION",
        "CONFIRMATION_POWER_AND_COHORT_FREEZE",
    }
    if current != expected_current_state:
        raise InheritedAuthorizationError("controller current-state identity drift")
    if authorized_early_state is None and authorized_role_count_recovery is None:
        if current != "SELECTION_COHORTS":
            raise InheritedAuthorizationError(
                "ordinary inherited authorization is selection-boundary only"
            )
    elif authorized_early_state is not None:
        if current != authorized_early_state or current not in early_states:
            raise InheritedAuthorizationError(
                "authorized early-state identity drift"
            )
    elif (
        authorized_role_count_recovery != "selection"
        or current != "SELECTION_COHORTS"
    ):
        raise InheritedAuthorizationError(
            "role-count recovery is not the exact selection transition"
        )

    expected_selection = (
        2000
        if current in early_states
        or authorized_role_count_recovery == "selection"
        else 0
    )
    state_checks = {
        "attempt": controller.get("active_attempt") == ACTIVE_ATTEMPT,
        "attempt_path": controller.get("active_attempt_path")
        == _relative(ATTEMPT_ROOT),
        "terminal": controller.get("terminal_label") is None
        and controller.get("confirmation_terminal") is False,
        "fit": type(controller.get("fit_outcome_episodes")) is int
        and controller.get("fit_outcome_episodes") == 1200,
        "selection": type(controller.get("selection_outcome_episodes")) is int
        and controller.get("selection_outcome_episodes") == expected_selection,
        "smoke": type(controller.get("smoke_outcome_episodes")) is int
        and controller.get("smoke_outcome_episodes") == 0,
        "confirmation_generated": type(
            controller.get("confirmation_outcome_episodes_generated")
        )
        is int
        and controller.get("confirmation_outcome_episodes_generated") == 0,
        "confirmation_executed": type(
            controller.get("confirmation_outcome_episodes_executed")
        )
        is int
        and controller.get("confirmation_outcome_episodes_executed") == 0,
        "confirmation_unopened": controller.get(
            "confirmation_outcomes_opened_for_analysis"
        )
        is False,
        "fit_design": controller.get("expected_fit_episode_count") == 1200,
        "selection_design": controller.get("expected_selection_episode_count")
        == 2000,
    }
    if not all(state_checks.values()):
        raise InheritedAuthorizationError(
            f"controller selection-boundary provenance drift: {state_checks}"
        )
    if (
        current == "SELECTION_COHORTS"
        and authorized_role_count_recovery is None
        and os.path.lexists(ATTEMPT_ROOT / "audit/selection_cohorts.json")
    ):
        raise InheritedAuthorizationError(
            "ordinary selection authorization found a pre-count role audit"
        )

    seal_path = _regular_file(seal_path).resolve()
    seal_relative = _relative(seal_path)
    seal_sha256 = sha256_file(seal_path)
    seal = _read_json(seal_path)
    _validate_inheritance_seal_scalar_types(seal)
    header = {
        "schema": type(seal.get("schema_version")) is int
        and seal.get("schema_version") == 1,
        "attempt": seal.get("attempt") == ACTIVE_ATTEMPT,
        "source": seal.get("source_attempt") == ACTIVATION_SOURCE_ATTEMPT,
        "target": seal.get("target_attempt") == ACTIVE_ATTEMPT,
        "science": seal.get("science_attempt") == SCIENCE_ATTEMPT,
        "checkpoint": seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL",
        "resume": seal.get("resume_state") == "SELECTION_COHORTS",
        "kind": seal.get("authorization_kind")
        == "zero_confirmation_outcome_version_forward_inherited_pre_selection",
        "passed": seal.get("passed") is True,
        "invalidity": seal.get("procedural_invalidity_confirmed") is True,
        "zero_confirmation": seal.get(
            "zero_confirmation_outcomes_at_version_forward"
        )
        is True,
        **{
            key: seal.get(key) is expected
            for key, expected in REQUIRED_EQUIVALENCE.items()
        },
    }
    if not all(header.values()):
        raise InheritedAuthorizationError(
            f"inheritance seal header/equivalence drift: {header}"
        )
    if not _exact_typed_mapping(seal.get("outcome_counts_at_seal"), ZERO_COUNTS):
        raise InheritedAuthorizationError(
            "inheritance seal selection-boundary counts drift"
        )
    sealed_files = _verify_sealed_files(seal.get("sealed_files"))
    expected_verifier_projection = _active_seal_verifier_projection(
        seal, sealed_files
    )
    invalidity_link = _file_link(
        seal.get("invalidity_evidence"), expected_path=INVALIDITY_PATH
    )
    source_seal_link = _file_link(
        seal.get("source_pre_data_seal"),
        expected_path=SOURCE_PRE_DATA_SEAL_PATH,
    )
    source_preselection_link = _file_link(
        seal.get("source_pre_selection_seal"),
        expected_path=SOURCE_PRE_SELECTION_SEAL_PATH,
    )
    source_receipt_link = _file_link(
        seal.get("source_version_forward_transaction_receipt"),
        expected_path=SOURCE_TRANSACTION_RECEIPT_PATH,
    )
    raw_draft = seal.get("superseded_invalidity_draft")
    if not isinstance(raw_draft, Mapping) or raw_draft.get("authoritative") is not False:
        raise InheritedAuthorizationError("invalidity draft authority drift")
    draft_link = _file_link(
        {"path": raw_draft.get("path"), "sha256": raw_draft.get("sha256")},
        expected_path=INVALIDITY_DRAFT_PATH,
    )

    invalidity = _read_json(INVALIDITY_PATH)
    defect = invalidity.get("defect")
    selection_evidence = invalidity.get("selection_evidence")
    if not (
        invalidity.get("attempt") == ACTIVATION_SOURCE_ATTEMPT
        and invalidity.get("procedural_invalidity") is True
        and invalidity.get("authoritative") is True
        and invalidity.get("scientific_outcomes_conditioned_on") is False
        and invalidity.get("selection_or_later_outcomes_opened") is False
        and _exact_typed_mapping(
            invalidity.get("controller_outcome_counts"), ZERO_COUNTS
        )
        and isinstance(defect, Mapping)
        and defect.get("classification")
        == "independent_verifier_contract_schema_rejects_authenticated_inherited_selection_boundary_paths"
        and defect.get("reproduced_without_outcome_arrays") is True
        and selection_evidence
        == {
            "controller_selection_outcome_episode_count": 0,
            "generation_output_file_count": 0,
            "seed_tuples_consumed": 0,
            "selection_outcome_arrays_opened": False,
            "worlds_constructed": 0,
        }
    ):
        raise InheritedAuthorizationError(
            "v007 procedural-invalidity provenance drift"
        )
    prior_draft = invalidity.get("superseded_draft")
    if not (
        isinstance(prior_draft, Mapping)
        and prior_draft.get("path") == draft_link["path"]
        and prior_draft.get("sha256") == draft_link["sha256"]
        and prior_draft.get("authoritative") is False
    ):
        raise InheritedAuthorizationError(
            "v007 invalidity draft cross-link drift"
        )

    source_seal = _read_json(SOURCE_PRE_DATA_SEAL_PATH)
    if not (
        source_seal.get("attempt") == ACTIVATION_SOURCE_ATTEMPT
        and source_seal.get("source_attempt") == SOURCE_ATTEMPT
        and source_seal.get("target_attempt") == ACTIVATION_SOURCE_ATTEMPT
        and source_seal.get("resume_state") == "SELECTION_COHORTS"
        and source_seal.get("passed") is True
        and _exact_typed_mapping(
            source_seal.get("outcome_counts_at_seal"),
            ZERO_COUNTS,
        )
    ):
        raise InheritedAuthorizationError("v007 source pre-data seal drift")
    source_preselection = _read_json(SOURCE_PRE_SELECTION_SEAL_PATH)
    if not (
        source_preselection.get("attempt") == SOURCE_ATTEMPT
        and source_preselection.get("checkpoint_state") == "PRE_SELECTION_SEAL"
        and source_preselection.get("passed") is True
        and source_preselection.get("pre_data_seal_sha256")
        == sha256_file(SELECTION_PRE_DATA_SEAL_PATH)
        and _exact_typed_mapping(
            source_preselection.get("outcome_counts_at_seal"), ZERO_COUNTS
        )
        and source_preselection_link["sha256"]
        == sha256_file(SOURCE_PRE_SELECTION_SEAL_PATH)
    ):
        raise InheritedAuthorizationError(
            "v006 pre-selection checkpoint provenance drift"
        )
    source_receipt = _read_json(SOURCE_TRANSACTION_RECEIPT_PATH)
    if not (
        source_receipt.get("source_attempt") == SOURCE_ATTEMPT
        and source_receipt.get("target_attempt") == ACTIVATION_SOURCE_ATTEMPT
        and source_receipt.get("resume_state") == "SELECTION_COHORTS"
        and source_receipt.get("passed") is True
        and source_receipt_link["sha256"]
        == sha256_file(SOURCE_TRANSACTION_RECEIPT_PATH)
    ):
        raise InheritedAuthorizationError("v007 source receipt drift")

    expected_versions = [
        SCIENCE_ATTEMPT,
        PRIOR_ATTEMPT,
        FIT_SOURCE_ATTEMPT,
        INTERMEDIATE_ATTEMPT,
        SOURCE_PARENT_ATTEMPT,
        SOURCE_ATTEMPT,
        ACTIVATION_SOURCE_ATTEMPT,
        ACTIVE_ATTEMPT,
    ]
    history = controller.get("attempt_history")
    lineage = controller.get("version_forward_lineage")
    prior_lineage = seal.get("prior_version_forward_lineage")
    if not (
        isinstance(history, list)
        and [item.get("version") for item in history if isinstance(item, Mapping)]
        == expected_versions
        and history[-2].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[-2].get("invalidity_evidence_path")
        == invalidity_link["path"]
        and history[-2].get("invalidity_evidence_sha256")
        == invalidity_link["sha256"]
        and history[-1].get("status")
        == "active_zero_confirmation_outcome_version_forward"
        and history[-1].get("version_forward_evidence_path") == seal_relative
        and history[-1].get("version_forward_evidence_sha256") == seal_sha256
        and isinstance(lineage, list)
        and len(lineage) == 7
        and isinstance(prior_lineage, list)
        and len(prior_lineage) == 6
        and lineage[:-1] == prior_lineage
    ):
        raise InheritedAuthorizationError(
            "v001-through-v008 controller lineage drift"
        )
    expected_edges = [
        (SCIENCE_ATTEMPT, PRIOR_ATTEMPT, "FIT_COHORTS"),
        (PRIOR_ATTEMPT, FIT_SOURCE_ATTEMPT, "FIT_COHORTS"),
        (FIT_SOURCE_ATTEMPT, INTERMEDIATE_ATTEMPT, "FIT_COHORTS"),
        (INTERMEDIATE_ATTEMPT, SOURCE_PARENT_ATTEMPT, "FIT_COHORTS"),
        (SOURCE_PARENT_ATTEMPT, SOURCE_ATTEMPT, "FIT_COHORTS"),
        (SOURCE_ATTEMPT, ACTIVATION_SOURCE_ATTEMPT, "SELECTION_COHORTS"),
        (ACTIVATION_SOURCE_ATTEMPT, ACTIVE_ATTEMPT, "SELECTION_COHORTS"),
    ]
    for edge, (old, new, resume) in zip(lineage, expected_edges):
        if not (
            isinstance(edge, Mapping)
            and edge.get("old_attempt") == old
            and edge.get("new_attempt") == new
            and edge.get("resume_state") == resume
            and edge.get("attempt_parameterization_verified") is True
        ):
            raise InheritedAuthorizationError(
                "controller deterministic edge chronology drift"
            )
    edge = lineage[-1]
    if not (
        edge.get("invalidity_path") == invalidity_link["path"]
        and edge.get("invalidity_sha256") == invalidity_link["sha256"]
        and edge.get("equivalence_path") == seal_relative
        and edge.get("equivalence_sha256") == seal_sha256
    ):
        raise InheritedAuthorizationError("active version-forward edge drift")

    completed = controller.get("completed_states")
    checkpoints = controller.get("verified_checkpoints")
    if not (
        isinstance(completed, list)
        and isinstance(checkpoints, list)
        and completed[: len(INHERITED_STATES)] == list(INHERITED_STATES)
        and len(checkpoints) == len(completed)
    ):
        raise InheritedAuthorizationError(
            "controller checkpoint chronology drift"
        )
    inherited = checkpoints[: len(INHERITED_STATES)]
    projection = [_checkpoint_projection(item) for item in inherited]
    if (
        seal.get("inherited_verified_checkpoints") != projection
        or edge.get("inherited_verified_checkpoints") != projection
    ):
        raise InheritedAuthorizationError(
            "inherited checkpoint projection drift"
        )
    for index, checkpoint in enumerate(inherited):
        expected_source = SCIENCE_ATTEMPT if index < 6 else SOURCE_ATTEMPT
        evidence = _regular_file(
            _resolve_relative(checkpoint.get("evidence_path"))
        )
        if not (
            isinstance(checkpoint, Mapping)
            and checkpoint.get("source_attempt") == expected_source
            and checkpoint.get("verification_lineage") == "direct_checkpoint"
            and sha256_file(evidence) == checkpoint.get("evidence_sha256")
        ):
            raise InheritedAuthorizationError(
                "inherited checkpoint evidence drift"
            )
    direct_states = [
        completed[index]
        for index, checkpoint in enumerate(checkpoints)
        if checkpoint.get("source_attempt") == ACTIVE_ATTEMPT
    ]
    if direct_states and direct_states[0] != "SELECTION_COHORTS":
        raise InheritedAuthorizationError(
            "first direct v008 checkpoint is not SELECTION_COHORTS"
        )

    events = _verify_ledger_chain(
        ledger_path=ledger_path, genesis_path=genesis_path, state=controller
    )
    edge_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "zero_confirmation_outcome_version_forward"
    ]
    if len(edge_indexes) != 7:
        raise InheritedAuthorizationError(
            "ledger does not contain the exact seven version-forward edges"
        )
    for index, (old, new, resume) in zip(edge_indexes, expected_edges):
        record = events[index]
        if not (
            record.get("attempt") == new
            and record.get("old_attempt") == old
            and record.get("new_attempt") == new
            and record.get("resume_state") == resume
        ):
            raise InheritedAuthorizationError(
                "ledger deterministic edge chronology drift"
            )
    ledger_edge = events[edge_indexes[-1]]
    if not (
        ledger_edge.get("invalidity_path") == invalidity_link["path"]
        and ledger_edge.get("invalidity_sha256") == invalidity_link["sha256"]
        and ledger_edge.get("equivalence_path") == seal_relative
        and ledger_edge.get("equivalence_sha256") == seal_sha256
        and ledger_edge.get("source_controller_adapter_marker")
        == seal.get("source_controller_adapter_marker")
    ):
        raise InheritedAuthorizationError(
            "active ledger edge provenance drift"
        )
    controller_adapter = _verify_controller_adapter_marker(
        controller, events=events
    )
    adapter_bound_count_events = _verify_adapter_count_event_bindings(events)
    role_count_recovery: dict[str, Any] | None = None
    if authorized_role_count_recovery is not None:
        role_count_recovery = _verify_role_count_recovery_authorization(
            role=authorized_role_count_recovery,
            controller=controller,
            events=events,
        )
    transaction_receipt = _verify_version_forward_transaction_receipt(
        seal_relative=seal_relative,
        seal_sha256=seal_sha256,
        invalidity_relative=invalidity_link["path"],
        invalidity_sha256=invalidity_link["sha256"],
        expected_source_marker=seal.get("source_controller_adapter_marker"),
        events=events,
        expected_verifier_projection=expected_verifier_projection,
    )
    independent = _verify_post_forward_recomputation(
        seal_path,
        authorized_early_state=authorized_early_state,
        authorized_role_count_recovery=authorized_role_count_recovery,
        expected_partition_file_count=expected_verifier_projection[
            "partition_file_count"
        ],
        expected_sealed_file_count=expected_verifier_projection[
            "sealed_file_count"
        ],
    )
    return {
        "passed": True,
        "active_attempt": ACTIVE_ATTEMPT,
        "science_attempt": SCIENCE_ATTEMPT,
        "state": expected_current_state,
        "seal_path": seal_relative,
        "seal_sha256": seal_sha256,
        "seal_checkpoint_state": "PRE_OUTCOME_SEAL",
        "invalidity_path": invalidity_link["path"],
        "invalidity_sha256": invalidity_link["sha256"],
        "source_pre_data_seal_path": source_seal_link["path"],
        "source_pre_data_seal_sha256": source_seal_link["sha256"],
        "source_pre_selection_seal_path": source_preselection_link["path"],
        "source_pre_selection_seal_sha256": source_preselection_link["sha256"],
        "inherited_checkpoint_count": len(inherited),
        "sealed_file_count": len(sealed_files),
        "version_forward_ledger_sequence": int(ledger_edge["seq"]),
        "version_forward_transaction_receipt": transaction_receipt,
        "controller_adapter_authorization": controller_adapter,
        "adapter_bound_count_event_count": adapter_bound_count_events,
        "role_count_recovery_authorization": role_count_recovery,
        "independent_recomputation": {
            "passed": True,
            "seal_path": independent["seal_path"],
            "seal_sha256": independent["seal_sha256"],
            "active_attempt": independent["active_attempt"],
            "science_attempt": independent["science_attempt"],
            "authorized_early_verifier_state": independent[
                "authorized_early_verifier_state"
            ],
            "authorized_role_count_recovery": independent[
                "authorized_role_count_recovery"
            ],
            "output_paths_created": independent["output_paths_created"],
        },
    }


@contextmanager
def _shared_program_lock(path: Path = PROGRAM_LOCK_PATH) -> Any:
    """Hold the controller lock shared while every authorization byte is used."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise InheritedAuthorizationError(
                f"controller lock is not a regular file: {path}"
            )
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _exact_byte_snapshot(paths: Sequence[Path]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for raw in paths:
        path = _regular_file(Path(raw)).resolve()
        result[str(path)] = path.read_bytes()
    return result


def _verify_direct_selection_seal(
    *,
    path: Path,
    controller: Mapping[str, Any],
    inheritance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the direct selection execution seal to the inherited raw seal."""

    path = _regular_file(path).resolve()
    value = _read_json(path)
    path_relative = _relative(path)
    digest = sha256_file(path)
    checks = {
        "schema": type(value.get("schema_version")) is int
        and value.get("schema_version") == 1,
        "passed": value.get("passed") is True,
        "attempt": value.get("attempt") == SOURCE_ATTEMPT,
        "checkpoint": value.get("checkpoint_state") == "PRE_SELECTION_SEAL",
        "inheritance_hash": value.get("pre_data_seal_sha256")
        == inheritance.get("source_pre_data_seal_sha256"),
        "selection_unopened": type(
            value.get("selection_arrays_opened_by_sealer")
        )
        is int
        and value.get("selection_arrays_opened_by_sealer") == 0,
        "refit_forbidden": value.get("selected_head_refit_permitted") is False,
    }
    counts = value.get("outcome_counts_at_seal")
    checks["counts"] = (
        isinstance(counts, Mapping)
        and type(counts.get("fit_outcome_episodes")) is int
        and counts.get("fit_outcome_episodes") == 1200
        and type(counts.get("selection_outcome_episodes")) is int
        and counts.get("selection_outcome_episodes") == 0
        and type(counts.get("smoke_outcome_episodes")) is int
        and counts.get("smoke_outcome_episodes") == 0
        and type(counts.get("confirmation_outcome_episodes_generated")) is int
        and counts.get("confirmation_outcome_episodes_generated") == 0
        and type(counts.get("confirmation_outcome_episodes_executed")) is int
        and counts.get("confirmation_outcome_episodes_executed") == 0
        and counts.get("confirmation_outcomes_opened_for_analysis") is False
    )
    checkpoint_matches = [
        item
        for item in controller.get("verified_checkpoints", [])
        if isinstance(item, Mapping)
        and item.get("source_attempt") == SOURCE_ATTEMPT
        and item.get("evidence_path") == path_relative
        and item.get("evidence_sha256") == digest
    ]
    checks["unique_direct_checkpoint"] = len(checkpoint_matches) == 1
    if not all(checks.values()):
        raise InheritedAuthorizationError(
            f"direct selection seal/inheritance binding drift: {checks}"
        )
    return {
        "path": path_relative,
        "sha256": digest,
        "checkpoint_state": "PRE_SELECTION_SEAL",
        "pre_data_seal_sha256": str(inheritance["source_pre_data_seal_sha256"]),
        "inherited_by_seal_sha256": str(inheritance["seal_sha256"]),
    }


def _read_verified_controller_via_adapter() -> dict[str, Any]:
    """Read strict controller status only through the receipt-bound v008 adapter."""

    adapter = _regular_file(CONTROLLER_ADAPTER_PATH)
    completed = subprocess.run(
        [sys.executable, str(adapter), "status"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    lines = completed.stdout.splitlines()
    if completed.returncode != 0 or completed.stderr.strip() or len(lines) != 1:
        raise InheritedAuthorizationError(
            "receipt-bound controller adapter rejected status"
        )
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise InheritedAuthorizationError(
            "receipt-bound controller adapter emitted invalid status"
        ) from error
    if not isinstance(value, dict):
        raise InheritedAuthorizationError(
            "receipt-bound controller adapter status is not an object"
        )
    return value


def verify_inherited_pre_data_authorization(
    *,
    expected_current_state: str,
    state: Mapping[str, Any] | None = None,
    seal_path: Path = INHERITANCE_SEAL_PATH,
    ledger_path: Path = LEDGER_PATH,
    genesis_path: Path = LEDGER_GENESIS_PATH,
    direct_selection_seal_path: Path | None = None,
    authorized_early_guard_token: object | None = None,
    authorized_early_verifier_contract_path: Path | None = None,
    authorized_role_count_recovery_guard_token: object | None = None,
    authorized_role_count_recovery: str | None = None,
) -> dict[str, Any]:
    """Authenticate inherited lineage under one exact, read-only lock snapshot.

    Ordinary controller status runs before acquiring the shared lock because the
    controller itself takes that lock exclusively.  The authenticated early guard
    instead runs while its caller already owns the exclusive lock and therefore
    must not reacquire it.  In either path the supplied/status object must equal
    current ``STATE.json`` and every state/ledger/seal byte is compared before and
    after independent recomputation.  Selection execution may additionally bind
    its direct seal in this same snapshot.
    """

    injected = state is not None
    early_guard = authorized_early_guard_token is _EARLY_ROOT_GUARD_TOKEN
    role_recovery_guard = (
        authorized_role_count_recovery_guard_token
        is _ROLE_COUNT_RECOVERY_GUARD_TOKEN
    )
    if authorized_early_guard_token is not None and not early_guard:
        raise InheritedAuthorizationError("invalid early-verifier guard token")
    if (
        authorized_role_count_recovery_guard_token is not None
        and not role_recovery_guard
    ):
        raise InheritedAuthorizationError("invalid role-count recovery guard token")
    if early_guard and role_recovery_guard:
        raise InheritedAuthorizationError(
            "early and role-count recovery guards are mutually exclusive"
        )
    if role_recovery_guard != (authorized_role_count_recovery is not None):
        raise InheritedAuthorizationError(
            "role-count recovery guard and role must be supplied together"
        )
    role_recovery_states = {"selection": "SELECTION_COHORTS"}
    if role_recovery_guard and role_recovery_states.get(
        authorized_role_count_recovery
    ) != expected_current_state:
        raise InheritedAuthorizationError(
            "role-count recovery guard is not bound to its exact role/state"
        )
    if role_recovery_guard and authorized_early_verifier_contract_path is not None:
        raise InheritedAuthorizationError(
            "role-count recovery cannot carry an early-verifier contract"
        )
    early_contracts = {
        "CANDIDATE_SELECTION": ATTEMPT_ROOT
        / "verifier_contract_no_candidate.json",
        "CONFIRMATION_POWER_AND_COHORT_FREEZE": ATTEMPT_ROOT
        / "verifier_contract_power_infeasible.json",
    }
    if early_guard:
        expected_contract = early_contracts.get(expected_current_state)
        if (
            not injected
            or expected_contract is None
            or authorized_early_verifier_contract_path is None
            or Path(authorized_early_verifier_contract_path).resolve()
            != expected_contract.resolve()
        ):
            raise InheritedAuthorizationError(
                "early-verifier guard is not bound to its exact state/contract"
            )
    elif not role_recovery_guard and (
        authorized_early_verifier_contract_path is not None
        or expected_current_state != "SELECTION_COHORTS"
    ):
        raise InheritedAuthorizationError(
            "inherited authorization state is outside its exact permitted contract"
        )
    if injected:
        controller = dict(state or {})
    else:
        controller = _read_verified_controller_via_adapter()

    tracked = [
        Path(seal_path),
        Path(ledger_path),
        Path(genesis_path),
        INVALIDITY_PATH,
        INVALIDITY_DRAFT_PATH,
        SOURCE_PRE_DATA_SEAL_PATH,
        SOURCE_PRE_SELECTION_SEAL_PATH,
        SELECTION_PRE_DATA_SEAL_PATH,
        SOURCE_TRANSACTION_RECEIPT_PATH,
        SOURCE_CONTROLLER_ADAPTER_PATH,
        SELECTION_CONTROLLER_ADAPTER_PATH,
        INDEPENDENT_VERIFIER_PATH,
        TRANSACTION_RECEIPT_PATH,
        TRANSACTION_SOURCE_PATH,
        PROGRAM_PATH,
    ]
    if not STATE_PATH.exists():
        raise InheritedAuthorizationError(
            "current on-disk STATE.json is required for inherited authorization"
        )
    tracked.append(STATE_PATH)
    if direct_selection_seal_path is not None:
        tracked.append(Path(direct_selection_seal_path))
    if role_recovery_guard:
        recovery_role = str(authorized_role_count_recovery)
        tracked.extend(
            ATTEMPT_ROOT
            / "data"
            / recovery_role
            / regime
            / "execution_manifest.json"
            for regime in (
                "native_plan",
                "markov_oracle",
                "plan_action_noise_0p2",
                "plan_random_action_0p1",
            )
        )
        role_audit = ATTEMPT_ROOT / "audit" / f"{recovery_role}_cohorts.json"
        if os.path.lexists(role_audit):
            tracked.append(role_audit)

    lock_path = PROGRAM_LOCK_PATH
    lock_context = nullcontext() if early_guard else _shared_program_lock(lock_path)
    with lock_context:
        before = _exact_byte_snapshot(tracked)
        disk_state = _read_json(STATE_PATH)
        if disk_state != controller:
            raise InheritedAuthorizationError(
                "injected/controller state and current locked STATE.json differ"
            )
        state_bytes = before[str(STATE_PATH.resolve())]

        result = _verify_inherited_pre_data_authorization_locked(
            expected_current_state=expected_current_state,
            state=controller,
            seal_path=seal_path,
            ledger_path=ledger_path,
            genesis_path=genesis_path,
            authorized_early_state=(
                expected_current_state if early_guard else None
            ),
            authorized_role_count_recovery=(
                authorized_role_count_recovery
                if role_recovery_guard
                else None
            ),
        )
        if direct_selection_seal_path is not None:
            if expected_current_state != "SELECTION_COHORTS":
                raise InheritedAuthorizationError(
                    "a direct selection seal is valid only in SELECTION_COHORTS"
                )
            result["direct_selection_seal"] = _verify_direct_selection_seal(
                path=Path(direct_selection_seal_path),
                controller=controller,
                inheritance=result,
            )

        after = _exact_byte_snapshot(tracked)
        if after != before:
            raise InheritedAuthorizationError(
                "state/ledger/seal bytes changed during inherited authorization"
            )
        result["authenticated_state_sha256"] = hashlib.sha256(state_bytes).hexdigest()
        result["program_lock_path"] = _relative(lock_path)
        result["exact_locked_snapshot_unchanged"] = True
        return result


__all__: Sequence[str] = (
    "ACTIVE_ATTEMPT",
    "SCIENCE_ATTEMPT",
    "INHERITANCE_SEAL_PATH",
    "TRANSACTION_RECEIPT_PATH",
    "InheritedAuthorizationError",
    "verify_inherited_pre_data_authorization",
)
