#!/usr/bin/env python3
"""Receipt-bound durable adapter for every v008 controller operation.

The adapter owns ``.program.lock`` across recovery, authenticated import of
the unchanged root controller, mutation, durable ledger/state publication, and
status verification.  The initial v007 -> v008 edge additionally spans the
independent pre/post equivalence checks and exclusive receipt.  Later v008
operations are accepted only when the receipt, adapter marker, root source,
adapter source, STATE, and newline-framed ledger agree exactly.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
import stat
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence

sys.dont_write_bytecode = True

SCIENCE_ATTEMPT = "v001"
INTERMEDIATE_ATTEMPT = "v004"
SOURCE_PARENT_ATTEMPT = "v005"
ACTIVATION_SOURCE_ATTEMPT = "v006"
SOURCE_ATTEMPT = "v007"
TARGET_ATTEMPT = "v008"
RESUME_STATE = "SELECTION_COHORTS"
ATTEMPT_ROOT = Path(__file__).resolve().parent
TRANSACTION_SOURCE_PATH = Path(__file__).resolve()
STUDY_ROOT = ATTEMPT_ROOT.parents[1]
REPOSITORY_ROOT = STUDY_ROOT.parents[1]
LOCK_PATH = STUDY_ROOT / ".program.lock"
STATE_PATH = STUDY_ROOT / "STATE.json"
LEDGER_PATH = STUDY_ROOT / "RESEARCH_LEDGER.jsonl"
GENESIS_PATH = STUDY_ROOT / "LEDGER_CHAIN_GENESIS.json"
TRANSACTION_PATH = STUDY_ROOT / "STATE_TRANSACTION.json"
PROGRAM_PATH = STUDY_ROOT / "program.py"
SEAL_PATH = ATTEMPT_ROOT / "audit/pre_data_inheritance_seal.json"
INVALIDITY_PATH = (
    STUDY_ROOT / "attempts/v007/audit/v007_procedural_invalidity.json"
)
INVALIDITY_DRAFT_PATH = (
    STUDY_ROOT / "attempts/v007/audit/v007_procedural_invalidity_draft.json"
)
PRIOR_RECEIPT_PATH = (
    STUDY_ROOT / "attempts/v007/audit/version_forward_transaction_receipt.json"
)
RECEIPT_PATH = ATTEMPT_ROOT / "audit/version_forward_transaction_receipt.json"
PENDING_STAGING_PATH = STUDY_ROOT / ".STATE_TRANSACTION.json.v008-staging"
STATE_STAGING_PATH = STUDY_ROOT / ".STATE.json.v008-staging"
RECEIPT_JOURNAL_PATH = STUDY_ROOT / "VERSION_FORWARD_TRANSACTION.json"
RECEIPT_JOURNAL_STAGING_PATH = (
    STUDY_ROOT / ".VERSION_FORWARD_TRANSACTION.json.v008-staging"
)
RECEIPT_STAGING_PATH = (
    ATTEMPT_ROOT / "audit/.version_forward_transaction_receipt.json.v008-staging"
)

PRIOR_ADAPTER_STATE_KEY = "v007_durable_controller_adapter"
SOURCE_ADAPTER_MARKER_FIELD = "source_controller_adapter_marker"
ADAPTER_STATE_KEY = "v008_durable_controller_adapter"
ADAPTER_ARTIFACT_TYPE = "v008_durable_controller_transaction"
ADAPTER_AUTHORIZATION_KIND = "receipt_bound_v008_root_controller_adapter"
PRIOR_ADAPTER_MARKER_FIELD = "prior_controller_adapter_marker"
RECEIPT_CONTEXT_EVENT_FIELD = "version_forward_receipt_context_sha256"
ADAPTER_TRANSACTION_ENVELOPE_FIELD = "v008_durable_adapter_transaction"
ADAPTER_TRANSACTION_ENVELOPE_KIND = (
    "receipt_anchored_v008_descendant_adapter_transaction"
)
RECEIPT_JOURNAL_ARTIFACT_TYPE = "v008_version_forward_receipt_context_journal"
_ADAPTER_LOCAL = threading.local()

# A no-op in production.  Tests replace this hook to emulate process death
# immediately after individual durability boundaries.  Recovery never depends
# on the hook and therefore exercises the same persisted bytes as production.
_DURABILITY_TEST_HOOK: Callable[[str], None] = lambda _boundary: None

ZERO_OUTCOMES: dict[str, int | bool] = {
    "fit_outcome_episodes": 1200,
    "selection_outcome_episodes": 0,
    "smoke_outcome_episodes": 0,
    "confirmation_outcome_episodes_generated": 0,
    "confirmation_outcome_episodes_executed": 0,
    "confirmation_outcomes_opened_for_analysis": False,
}
NUMERIC_OUTCOME_KEYS = frozenset(
    {
        "fit_outcome_episodes",
        "selection_outcome_episodes",
        "smoke_outcome_episodes",
        "confirmation_outcome_episodes_generated",
        "confirmation_outcome_episodes_executed",
    }
)
BOOLEAN_OUTCOME_KEY = "confirmation_outcomes_opened_for_analysis"
VERIFIER_PARTITION_NAMES = (
    "exact_hash",
    "normalized_ast",
    "canonical_contracts",
    "procedural_only",
    "new_lineage_support",
    "lineage_support",
)

CONTROLLER_STATE_MACHINE = (
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
EARLY_FAILURE_TRANSITIONS = {
    "no_candidate": {
        "trigger_state": "CANDIDATE_SELECTION",
        "source_relative": "selection/selection_ledger.json",
        "contract_relative": "verifier_contract_no_candidate.json",
    },
    "power_infeasible": {
        "trigger_state": "CONFIRMATION_POWER_AND_COHORT_FREEZE",
        "source_relative": "power_analysis.json",
        "contract_relative": "verifier_contract_power_infeasible.json",
    },
}
POSTCONFIRMATION_FAILURE_TRANSITIONS = {
    "SEALED_ANALYSIS": {
        "source": "analysis_execution",
        "source_relative": "audit/analysis_execution_invalid.json",
        "audit_hash_keys": (
            "analysis_execution_invalid",
            "analysis_execution",
            "analysis_failure",
        ),
    },
    "LATENCY_AND_RESOURCE_REPORTING": {
        "source": "latency_resource",
        "source_relative": "metrics/latency_and_resources.json",
        "audit_hash_keys": (
            "latency_resource",
            "latency_and_resources",
            "latency_report",
        ),
    },
}


class VersionForwardTransactionError(RuntimeError):
    """The atomic version-forward transaction was not fully authenticated."""


def _validate_outcome_counter_types(state: Mapping[str, Any]) -> None:
    bad_numeric = {
        key: state.get(key)
        for key in NUMERIC_OUTCOME_KEYS
        if type(state.get(key)) is not int or int(state[key]) < 0
    }
    if bad_numeric or type(state.get(BOOLEAN_OUTCOME_KEY)) is not bool:
        raise VersionForwardTransactionError(
            "controller outcome-counter type drift: "
            f"numeric={bad_numeric}, boolean={state.get(BOOLEAN_OUTCOME_KEY)!r}"
        )


def _validate_count_assignments(counts: Mapping[str, Any]) -> dict[str, int | bool]:
    if not counts or not set(counts).issubset(
        NUMERIC_OUTCOME_KEYS | {BOOLEAN_OUTCOME_KEY}
    ):
        raise VersionForwardTransactionError("unsupported adapter outcome counter")
    invalid = {
        key: value
        for key, value in counts.items()
        if (
            key in NUMERIC_OUTCOME_KEYS
            and (type(value) is not int or int(value) < 0)
        )
        or (key == BOOLEAN_OUTCOME_KEY and type(value) is not bool)
    }
    if invalid:
        raise VersionForwardTransactionError(
            f"adapter outcome-counter type drift: {invalid}"
        )
    return dict(counts)


def _validate_count_transition_constraints(
    base_state: Mapping[str, Any], assignments: Mapping[str, int | bool]
) -> None:
    allowed_states = {
        "fit_outcome_episodes": {"FIT_COHORTS"},
        "selection_outcome_episodes": {"SELECTION_COHORTS"},
        "smoke_outcome_episodes": {"EXCLUDED_MECHANICAL_SMOKE"},
        "confirmation_outcome_episodes_generated": {
            "CONFIRMATION_GENERATION", "CONFIRMATION_EXECUTION"
        },
        "confirmation_outcome_episodes_executed": {
            "CONFIRMATION_EXECUTION", "CONFIRMATION_INPUT_SEAL"
        },
        "confirmation_outcomes_opened_for_analysis": {"SEALED_ANALYSIS"},
    }
    expected_keys = {
        "fit_outcome_episodes": "expected_fit_episode_count",
        "selection_outcome_episodes": "expected_selection_episode_count",
        "smoke_outcome_episodes": "expected_smoke_episode_count",
        "confirmation_outcome_episodes_generated": (
            "expected_confirmation_episode_count"
        ),
        "confirmation_outcome_episodes_executed": (
            "expected_confirmation_episode_count"
        ),
    }
    if (
        base_state.get("early_scientific_failure") is not None
        or base_state.get("terminal_label") is not None
    ):
        raise VersionForwardTransactionError(
            "adapter count transition follows a branch or terminal mutation"
        )
    for key, value in assignments.items():
        old = base_state.get(key)
        if base_state.get("current_state") not in allowed_states[key]:
            raise VersionForwardTransactionError(
                f"adapter counter {key} changed in a forbidden state"
            )
        if (
            type(value) is bool
            and (type(old) is not bool or old is True and value is False)
        ) or (
            type(value) is int
            and (type(old) is not int or value < old)
        ):
            raise VersionForwardTransactionError(
                f"adapter counter {key} moved backward or changed type"
            )
        expected_key = expected_keys.get(key)
        if (
            expected_key is not None
            and expected_key in base_state
            and value > base_state[expected_key]
        ):
            raise VersionForwardTransactionError(
                f"adapter counter {key} exceeds the preregistered size"
            )


def _validate_state_ledger_types(state: Mapping[str, Any], *, label: str) -> None:
    """Reject bool/float aliases in controller structural ledger fields."""

    if (
        type(state.get("ledger_event_count")) is not int
        or int(state["ledger_event_count"]) <= 0
        or not isinstance(state.get("ledger_head_sha256"), str)
        or not state["ledger_head_sha256"]
    ):
        raise VersionForwardTransactionError(
            f"{label} controller ledger structural type drift"
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(REPOSITORY_ROOT.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise VersionForwardTransactionError(f"path outside repository: {path}") from error


def _snapshot_file(path: Path) -> tuple[dict[str, Any], bytes]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VersionForwardTransactionError(f"cannot open authenticated file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise VersionForwardTransactionError(f"authenticated file is linked or non-regular: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise VersionForwardTransactionError(f"authenticated file changed while open: {path}")
        payload = b"".join(chunks)
        return {
            "path": _relative(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
            "nlink": int(before.st_nlink),
        }, payload
    finally:
        os.close(descriptor)


def _json_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VersionForwardTransactionError(f"invalid JSON in {label}") from error
    if not isinstance(value, dict):
        raise VersionForwardTransactionError(f"non-object JSON in {label}")
    return value


def _authenticated_active_seal_verifier_projection(
    *, seal_relative: str, seal_sha256: str
) -> dict[str, Any]:
    """Rebuild the receipt count projection from the live authenticated seal.

    The standalone and separately implemented partition verifiers intentionally
    report compact counts.  A durable receipt must not turn those counts into
    self-authenticating claims, so every local consumer derives the same values
    again from the hash-bound active seal.  Sealed inputs are rehashed here;
    partition membership is cross-linked to the seal's independently produced
    manifest closure and complete/disjoint partition projection.
    """

    seal_record, seal_payload = _snapshot_file(SEAL_PATH)
    if (
        seal_record["path"] != seal_relative
        or seal_record["sha256"] != seal_sha256
    ):
        raise VersionForwardTransactionError(
            "active inheritance seal identity drift during verifier projection"
        )
    seal = _json_payload(seal_payload, "active inheritance seal")
    if seal_payload != _pretty_bytes(seal):
        raise VersionForwardTransactionError(
            "active inheritance seal is not canonical JSON"
        )

    sealed_files = seal.get("sealed_files")
    if type(sealed_files) is not dict or not sealed_files:
        raise VersionForwardTransactionError(
            "active inheritance seal sealed-file map drift"
        )
    verified_sealed_paths: set[str] = set()
    for raw, expected_sha256 in sorted(sealed_files.items()):
        if (
            type(raw) is not str
            or not raw
            or not _is_sha256(expected_sha256)
        ):
            raise VersionForwardTransactionError(
                "active inheritance seal sealed-file entry drift"
            )
        relative = Path(raw)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != raw
        ):
            raise VersionForwardTransactionError(
                f"active inheritance seal path is noncanonical: {raw}"
            )
        path = REPOSITORY_ROOT / relative
        observed, _payload = _snapshot_file(path)
        if observed["path"] != raw or observed["sha256"] != expected_sha256:
            raise VersionForwardTransactionError(
                f"active inheritance sealed-file hash drift: {raw}"
            )
        if raw in verified_sealed_paths:
            raise VersionForwardTransactionError(
                f"duplicate active inheritance sealed-file path: {raw}"
            )
        verified_sealed_paths.add(raw)

    closure = seal.get("v008_manifest_closure")
    discovered = closure.get("discovered_paths") if type(closure) is dict else None
    if (
        type(discovered) is not list
        or not discovered
        or any(type(item) is not str or not item for item in discovered)
        or discovered != sorted(discovered)
        or len(discovered) != len(set(discovered))
    ):
        raise VersionForwardTransactionError(
            "active inheritance seal manifest-closure path drift"
        )

    partitions = seal.get("source_partitions")
    if type(partitions) is not dict or set(partitions) != set(
        VERIFIER_PARTITION_NAMES
    ):
        raise VersionForwardTransactionError(
            "active inheritance seal source-partition schema drift"
        )
    members: dict[str, list[str]] = {}
    for name in VERIFIER_PARTITION_NAMES:
        records = partitions.get(name)
        if type(records) is not list:
            raise VersionForwardTransactionError(
                f"active inheritance seal partition type drift: {name}"
            )
        names: list[str] = []
        for record in records:
            relative_path = (
                record.get("relative_path") if type(record) is dict else None
            )
            if type(relative_path) is not str or not relative_path:
                raise VersionForwardTransactionError(
                    f"active inheritance seal partition record drift: {name}"
                )
            relative = Path(relative_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != relative_path
            ):
                raise VersionForwardTransactionError(
                    f"active inheritance partition path is noncanonical: {relative_path}"
                )
            names.append(relative_path)
        if names != sorted(names) or len(names) != len(set(names)):
            raise VersionForwardTransactionError(
                f"active inheritance partition membership drift: {name}"
            )
        members[name] = names

    flattened = [
        relative_path
        for name in VERIFIER_PARTITION_NAMES
        for relative_path in members[name]
    ]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != discovered:
        raise VersionForwardTransactionError(
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
        raise VersionForwardTransactionError(
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
        "sealed_file_count": len(verified_sealed_paths),
        "independent_partitions": independent,
    }


def _ledger_summary(payload: bytes, genesis_payload: bytes) -> dict[str, Any]:
    genesis = _json_payload(genesis_payload, "ledger genesis")
    prefix_bytes = genesis.get("legacy_prefix_bytes")
    legacy_count = genesis.get("legacy_prefix_event_count")
    if type(prefix_bytes) is not int or type(legacy_count) is not int:
        raise VersionForwardTransactionError("ledger genesis count types drift")
    if hashlib.sha256(payload[:prefix_bytes]).hexdigest() != genesis.get("legacy_prefix_sha256"):
        raise VersionForwardTransactionError("ledger immutable prefix drift")
    if not payload or not payload.endswith(b"\n"):
        raise VersionForwardTransactionError("ledger is empty or lacks final newline framing")
    lines = payload.splitlines()
    if len(lines) < legacy_count:
        raise VersionForwardTransactionError("ledger prefix event count drift")
    previous = f"legacy:{genesis['legacy_prefix_sha256']}"
    for sequence, line in enumerate(lines[legacy_count:], start=legacy_count + 1):
        value = _json_payload(line, f"ledger line {sequence}")
        if line != _canonical_bytes(value):
            raise VersionForwardTransactionError(
                f"ledger line {sequence} is not canonical JSON"
            )
        observed = value.pop("record_sha256", None)
        if (
            not _is_sha256(observed)
            or observed != _canonical_sha256(value)
            or type(value.get("seq")) is not int
            or value.get("seq") != sequence
            or not isinstance(value.get("prev_sha256"), str)
            or value.get("prev_sha256") != previous
        ):
            raise VersionForwardTransactionError(f"ledger chain drift at {sequence}")
        previous = str(observed)
    return {
        "event_count": len(lines),
        "head_sha256": previous,
        "ledger_sha256": hashlib.sha256(payload).hexdigest(),
        "ledger_bytes": len(payload),
    }


def _state_ledger_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    state_record, state_payload = _snapshot_file(STATE_PATH)
    ledger_record, ledger_payload = _snapshot_file(LEDGER_PATH)
    genesis_record, genesis_payload = _snapshot_file(GENESIS_PATH)
    state = _json_payload(state_payload, "STATE.json")
    _validate_state_ledger_types(state, label="STATE snapshot")
    ledger = _ledger_summary(ledger_payload, genesis_payload)
    state_record["object_sha256"] = _canonical_sha256(state)
    # The immutable receipt must carry the historical state object as well as
    # its exact on-disk byte hash.  Without this copy, later workflow states
    # could not independently reconstruct and authenticate the state bytes at
    # the version-forward boundary.
    state_record["object"] = state
    ledger_record.update(ledger)
    if (
        state.get("ledger_event_count") != ledger["event_count"]
        or state.get("ledger_head_sha256") != ledger["head_sha256"]
    ):
        raise VersionForwardTransactionError("STATE/ledger binding drift")
    return {
        "state": state_record,
        "ledger": ledger_record,
        "genesis": genesis_record,
    }, state


def _stable_receipt_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    state = snapshot["state"]
    ledger = snapshot["ledger"]
    genesis = snapshot["genesis"]
    return {
        "state": {
            key: state[key]
            for key in ("path", "sha256", "bytes", "object_sha256", "object")
        },
        "ledger": {
            key: ledger[key]
            for key in (
                "path",
                "sha256",
                "bytes",
                "event_count",
                "head_sha256",
                "ledger_sha256",
            )
        },
        "genesis": {
            key: genesis[key] for key in ("path", "sha256", "bytes")
        },
    }


def _validate_stable_receipt_snapshot(
    value: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "state",
        "ledger",
        "genesis",
    }:
        raise VersionForwardTransactionError(
            f"transaction {label} stable snapshot schema drift"
        )
    snapshot = dict(value)
    state_record = snapshot["state"]
    ledger_record = snapshot["ledger"]
    genesis_record = snapshot["genesis"]
    if (
        not isinstance(state_record, Mapping)
        or set(state_record)
        != {"path", "sha256", "bytes", "object_sha256", "object"}
        or state_record.get("path") != _relative(STATE_PATH)
        or not _is_sha256(state_record.get("sha256"))
        or type(state_record.get("bytes")) is not int
        or not isinstance(state_record.get("object"), Mapping)
        or not _is_sha256(state_record.get("object_sha256"))
    ):
        raise VersionForwardTransactionError(
            f"transaction {label} stable STATE record drift"
        )
    state_payload = _pretty_bytes(state_record["object"])
    if (
        len(state_payload) != state_record["bytes"]
        or hashlib.sha256(state_payload).hexdigest() != state_record["sha256"]
        or _canonical_sha256(state_record["object"])
        != state_record["object_sha256"]
    ):
        raise VersionForwardTransactionError(
            f"transaction {label} stable STATE bytes drift"
        )
    if (
        not isinstance(ledger_record, Mapping)
        or set(ledger_record)
        != {
            "path",
            "sha256",
            "bytes",
            "event_count",
            "head_sha256",
            "ledger_sha256",
        }
        or ledger_record.get("path") != _relative(LEDGER_PATH)
        or not _is_sha256(ledger_record.get("sha256"))
        or ledger_record.get("ledger_sha256") != ledger_record.get("sha256")
        or type(ledger_record.get("bytes")) is not int
        or int(ledger_record["bytes"]) <= 0
        or type(ledger_record.get("event_count")) is not int
        or int(ledger_record["event_count"]) <= 0
        or not isinstance(ledger_record.get("head_sha256"), str)
    ):
        raise VersionForwardTransactionError(
            f"transaction {label} stable ledger record drift"
        )
    if (
        not isinstance(genesis_record, Mapping)
        or set(genesis_record) != {"path", "sha256", "bytes"}
        or genesis_record.get("path") != _relative(GENESIS_PATH)
        or not _is_sha256(genesis_record.get("sha256"))
        or type(genesis_record.get("bytes")) is not int
        or int(genesis_record["bytes"]) <= 0
    ):
        raise VersionForwardTransactionError(
            f"transaction {label} stable genesis record drift"
        )
    state = state_record["object"]
    if (
        state.get("ledger_event_count") != ledger_record["event_count"]
        or state.get("ledger_head_sha256") != ledger_record["head_sha256"]
    ):
        raise VersionForwardTransactionError(
            f"transaction {label} stable STATE/ledger drift"
        )
    return json.loads(json.dumps(snapshot))


def _predicted_post_snapshot(
    pending: Mapping[str, Any],
    *,
    base_ledger_payload: bytes,
    pre_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    intended = pending["intended_state"]
    state_payload = _pretty_bytes(intended)
    suffix = str(pending["expected_suffix"]).encode("ascii")
    ledger_payload = base_ledger_payload + suffix
    return {
        "state": {
            "path": _relative(STATE_PATH),
            "sha256": hashlib.sha256(state_payload).hexdigest(),
            "bytes": len(state_payload),
            "object_sha256": _canonical_sha256(intended),
            "object": json.loads(json.dumps(intended)),
        },
        "ledger": {
            "path": _relative(LEDGER_PATH),
            "sha256": hashlib.sha256(ledger_payload).hexdigest(),
            "bytes": len(ledger_payload),
            "event_count": pending["target_event_count"],
            "head_sha256": pending["target_head_sha256"],
            "ledger_sha256": hashlib.sha256(ledger_payload).hexdigest(),
        },
        "genesis": json.loads(json.dumps(pre_snapshot["genesis"])),
    }


def _validate_pre_state(state: Mapping[str, Any]) -> None:
    _validate_outcome_counter_types(state)
    _validate_state_ledger_types(state, label="pre-forward STATE")
    observed = {key: state.get(key) for key in ZERO_OUTCOMES}
    if (
        state.get("active_attempt") != SOURCE_ATTEMPT
        or state.get("current_state") != RESUME_STATE
        or state.get("confirmation_terminal") is not False
        or observed != ZERO_OUTCOMES
        or state.get("completed_states")
        != list(CONTROLLER_STATE_MACHINE[:9])
        or state.get("expected_fit_episode_count") != 1200
        or state.get("expected_selection_episode_count") != 2000
        or state.get("expected_smoke_episode_count") != 24
        or state.get("last_verified_checkpoint", {}).get("name")
        != "v006_pre_selection_seal"
    ):
        raise VersionForwardTransactionError(
            "pre-forward state is not the exact verified v007 selection "
            f"boundary: {observed}"
        )


def _validate_post_state(
    state: Mapping[str, Any],
    controller_return: Mapping[str, Any],
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
) -> None:
    _validate_outcome_counter_types(state)
    _validate_state_ledger_types(state, label="post-forward STATE")
    observed = {key: state.get(key) for key in ZERO_OUTCOMES}
    if (
        state.get("active_attempt") != TARGET_ATTEMPT
        or state.get("current_state") != RESUME_STATE
        or state.get("confirmation_terminal") is not False
        or observed != ZERO_OUTCOMES
        or dict(controller_return) != dict(state)
        or PRIOR_ADAPTER_STATE_KEY in state
        or post_snapshot["ledger"]["event_count"]
        != pre_snapshot["ledger"]["event_count"] + 1
    ):
        raise VersionForwardTransactionError("post-forward controller result/state drift")
    _validate_adapter_marker(state, post_snapshot["ledger"])
    lineage = state.get("version_forward_lineage")
    pre_state = pre_snapshot.get("state", {}).get("object", {})
    prior_lineage = pre_state.get("version_forward_lineage")
    if (
        not isinstance(lineage, list)
        or len(lineage) != 7
        or not isinstance(prior_lineage, list)
        or lineage[:-1] != prior_lineage
    ):
        raise VersionForwardTransactionError("post-forward lineage prefix/edge drift")
    edge = lineage[-1]
    if (
        not isinstance(edge, Mapping)
        or edge.get("old_attempt") != SOURCE_ATTEMPT
        or edge.get("new_attempt") != TARGET_ATTEMPT
        or edge.get("resume_state") != RESUME_STATE
        or edge.get("equivalence_path") != _relative(SEAL_PATH)
        or edge.get("equivalence_sha256")
        != _snapshot_file(SEAL_PATH)[0]["sha256"]
        or edge.get("invalidity_path") != _relative(INVALIDITY_PATH)
        or edge.get("invalidity_sha256")
        != _snapshot_file(INVALIDITY_PATH)[0]["sha256"]
    ):
        raise VersionForwardTransactionError("post-forward lineage edge identity drift")
    history = state.get("attempt_history")
    pre_history = pre_state.get("attempt_history")
    expected_prior_history = json.loads(json.dumps(pre_history))
    if isinstance(expected_prior_history, list):
        matching_source = [
            item
            for item in expected_prior_history
            if isinstance(item, dict)
            and item.get("version") == SOURCE_ATTEMPT
        ]
        if len(matching_source) == 1:
            matching_source[0]["status"] = (
                "invalid_zero_confirmation_outcome_procedural"
            )
            matching_source[0]["invalidity_evidence_path"] = _relative(
                INVALIDITY_PATH
            )
            matching_source[0]["invalidity_evidence_sha256"] = _snapshot_file(
                INVALIDITY_PATH
            )[0]["sha256"]
    if (
        not isinstance(history, list)
        or not isinstance(pre_history, list)
        or history[:-1] != expected_prior_history
        or [item.get("version") for item in history if isinstance(item, Mapping)]
        != [
            SCIENCE_ATTEMPT,
            "v002",
            "v003",
            INTERMEDIATE_ATTEMPT,
            SOURCE_PARENT_ATTEMPT,
            ACTIVATION_SOURCE_ATTEMPT,
            SOURCE_ATTEMPT,
            TARGET_ATTEMPT,
        ]
        or history[-1].get("status")
        != "active_zero_confirmation_outcome_version_forward"
        or history[-1].get("version_forward_evidence_path")
        != _relative(SEAL_PATH)
        or history[-1].get("version_forward_evidence_sha256")
        != _snapshot_file(SEAL_PATH)[0]["sha256"]
    ):
        raise VersionForwardTransactionError("post-forward attempt history drift")
    residues = [
        str(path)
        for path in (TRANSACTION_PATH, PENDING_STAGING_PATH, STATE_STAGING_PATH)
        if os.path.lexists(path)
    ]
    if residues:
        raise VersionForwardTransactionError(
            f"controller left transaction/staging residue: {residues}"
        )


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise VersionForwardTransactionError(f"cannot load authenticated module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _source_ast_sha256(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    return hashlib.sha256(
        ast.dump(ast.parse(source, filename=str(path)), include_attributes=False).encode("utf-8")
    ).hexdigest()


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _durability_boundary(name: str) -> None:
    _DURABILITY_TEST_HOOK(name)


def _fsync_directory(path: Path, boundary: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
        _durability_boundary(boundary)
    finally:
        os.close(descriptor)


@contextmanager
def _adapter_lock() -> Iterator[None]:
    if getattr(_ADAPTER_LOCAL, "held", False):
        raise VersionForwardTransactionError("controller adapter lock is non-reentrant")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _ADAPTER_LOCAL.held = True
        yield
    finally:
        _ADAPTER_LOCAL.held = False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _require_adapter_lock() -> None:
    if not getattr(_ADAPTER_LOCAL, "held", False):
        raise VersionForwardTransactionError(
            "controller adapter mutation/recovery requires .program.lock"
        )


def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise VersionForwardTransactionError(f"missing {label}: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise VersionForwardTransactionError(
            f"linked, aliased, or non-regular {label}: {path}"
        )
    return path.read_bytes()


def _canonical_object_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _regular_bytes(path, label)
    value = _json_payload(payload, label)
    if payload != _pretty_bytes(value):
        raise VersionForwardTransactionError(f"noncanonical {label}: {path}")
    return value, payload


def _source_record(path: Path) -> dict[str, Any]:
    record, _payload = _snapshot_file(path)
    return {**record, "ast_sha256": _source_ast_sha256(path)}


def _write_staging_file(path: Path, payload: bytes, *, prefix: str) -> None:
    if os.path.lexists(path):
        raise VersionForwardTransactionError(f"stale {prefix} staging file exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise VersionForwardTransactionError(f"short {prefix} staging write")
        _durability_boundary(f"{prefix}_write")
        os.fsync(descriptor)
        _durability_boundary(f"{prefix}_fsync")
    finally:
        os.close(descriptor)


def _journal_alias_bytes(
    path: Path, *, label: str
) -> tuple[bytes, os.stat_result]:
    """Read one journal publication alias without following or tolerating drift."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VersionForwardTransactionError(
            f"cannot open {label}: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink not in (1, 2):
            raise VersionForwardTransactionError(
                f"linked, aliased, or non-regular {label}: {path}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_nlink,
        ):
            raise VersionForwardTransactionError(
                f"{label} changed while it was authenticated"
            )
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _fsync_exact_publication_alias(
    path: Path,
    expected: bytes,
    *,
    label: str,
    boundary: str,
) -> None:
    """Make already-readable crash residue durable before it authorizes use."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VersionForwardTransactionError(
            f"cannot open recovery {label}: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink not in (1, 2):
            raise VersionForwardTransactionError(
                f"linked, aliased, or non-regular recovery {label}: {path}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        if b"".join(chunks) != expected:
            raise VersionForwardTransactionError(
                f"recovery {label} differs from its authenticated bytes"
            )
        os.fsync(descriptor)
        _durability_boundary(boundary)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_nlink,
        ):
            raise VersionForwardTransactionError(
                f"recovery {label} changed during inode fsync"
            )
    finally:
        os.close(descriptor)


def _complete_receipt_journal_publication(
    expected: bytes, *, recovery: bool
) -> None:
    """Atomically create, authenticate, and close the journal hard-link pair."""

    prefix = "recovery_receipt_journal" if recovery else "receipt_journal"
    final_exists = os.path.lexists(RECEIPT_JOURNAL_PATH)
    staging_exists = os.path.lexists(RECEIPT_JOURNAL_STAGING_PATH)
    final_meta: os.stat_result | None = None
    staging_meta: os.stat_result | None = None
    if final_exists:
        payload, final_meta = _journal_alias_bytes(
            RECEIPT_JOURNAL_PATH, label="version-forward receipt journal"
        )
        if payload != expected:
            raise VersionForwardTransactionError(
                "competing version-forward receipt journal differs from staging"
            )
    if staging_exists:
        payload, staging_meta = _journal_alias_bytes(
            RECEIPT_JOURNAL_STAGING_PATH,
            label="version-forward receipt journal staging",
        )
        if payload != expected:
            raise VersionForwardTransactionError(
                "version-forward receipt journal staging content drift"
            )
    if final_meta is not None and staging_meta is not None:
        if (
            (final_meta.st_dev, final_meta.st_ino)
            != (staging_meta.st_dev, staging_meta.st_ino)
            or final_meta.st_nlink != 2
            or staging_meta.st_nlink != 2
        ):
            raise VersionForwardTransactionError(
                "receipt journal and staging are competing aliases"
            )
    elif final_meta is not None:
        if final_meta.st_nlink != 1:
            raise VersionForwardTransactionError(
                "published receipt journal has a hidden alias"
            )
    elif staging_meta is not None:
        if staging_meta.st_nlink != 1:
            raise VersionForwardTransactionError(
                "unpublished receipt journal staging has a hidden alias"
            )
    else:
        raise VersionForwardTransactionError(
            "receipt journal publication lacks both final and staging files"
        )

    if recovery:
        _fsync_exact_publication_alias(
            (
                RECEIPT_JOURNAL_STAGING_PATH
                if staging_meta is not None
                else RECEIPT_JOURNAL_PATH
            ),
            expected,
            label="receipt journal publication alias",
            boundary="recovery_receipt_journal_inode_fsync",
        )

    if final_meta is None:
        try:
            os.link(RECEIPT_JOURNAL_STAGING_PATH, RECEIPT_JOURNAL_PATH)
        except FileExistsError as error:
            raise VersionForwardTransactionError(
                "competing receipt journal appeared during exclusive publication"
            ) from error
        _durability_boundary(f"{prefix}_link")
        _fsync_directory(
            RECEIPT_JOURNAL_PATH.parent, f"{prefix}_link_parent_fsync"
        )
        final_payload, final_meta = _journal_alias_bytes(
            RECEIPT_JOURNAL_PATH, label="version-forward receipt journal"
        )
        staging_payload, staging_meta = _journal_alias_bytes(
            RECEIPT_JOURNAL_STAGING_PATH,
            label="version-forward receipt journal staging",
        )
        if (
            final_payload != expected
            or staging_payload != expected
            or (final_meta.st_dev, final_meta.st_ino)
            != (staging_meta.st_dev, staging_meta.st_ino)
            or final_meta.st_nlink != 2
            or staging_meta.st_nlink != 2
        ):
            raise VersionForwardTransactionError(
                "receipt journal hard-link publication identity drift"
            )

    if os.path.lexists(RECEIPT_JOURNAL_STAGING_PATH):
        RECEIPT_JOURNAL_STAGING_PATH.unlink()
        _durability_boundary(f"{prefix}_staging_unlink")
        _fsync_directory(
            RECEIPT_JOURNAL_PATH.parent,
            f"{prefix}_staging_unlink_parent_fsync",
        )
    final_payload, final_meta = _journal_alias_bytes(
        RECEIPT_JOURNAL_PATH, label="version-forward receipt journal"
    )
    if final_payload != expected or final_meta.st_nlink != 1:
        raise VersionForwardTransactionError(
            "receipt journal publication did not close its exact alias"
        )
    # Also covers recovery from a final-only directory entry whose prior
    # staging unlink may not have reached stable storage before process death.
    _fsync_directory(
        RECEIPT_JOURNAL_PATH.parent, f"{prefix}_publication_parent_fsync"
    )


def _publish_receipt_journal(journal: Mapping[str, Any]) -> None:
    _require_adapter_lock()
    if os.path.lexists(RECEIPT_JOURNAL_PATH) or os.path.lexists(
        RECEIPT_JOURNAL_STAGING_PATH
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal publication is not exclusive"
        )
    _write_staging_file(
        RECEIPT_JOURNAL_STAGING_PATH,
        _pretty_bytes(journal),
        prefix="receipt_journal_staging",
    )
    _complete_receipt_journal_publication(
        _pretty_bytes(journal), recovery=False
    )


def _unlink_receipt_journal(expected_journal: Mapping[str, Any]) -> None:
    _require_adapter_lock()
    if os.path.lexists(RECEIPT_JOURNAL_STAGING_PATH):
        raise VersionForwardTransactionError(
            "receipt journal cleanup found a staging alias"
        )
    expected = _pretty_bytes(expected_journal)
    observed, metadata = _journal_alias_bytes(
        RECEIPT_JOURNAL_PATH,
        label="version-forward receipt journal cleanup target",
    )
    if observed != expected or metadata.st_nlink != 1:
        raise VersionForwardTransactionError(
            "receipt journal cleanup target identity drift"
        )
    RECEIPT_JOURNAL_PATH.unlink()
    _durability_boundary("receipt_journal_unlink")
    _fsync_directory(
        RECEIPT_JOURNAL_PATH.parent, "receipt_journal_unlink_parent_fsync"
    )


def _publish_or_recover_exact_receipt(journal: Mapping[str, Any]) -> None:
    _require_adapter_lock()
    expected = _pretty_bytes(journal["predicted_receipt"])
    if hashlib.sha256(expected).hexdigest() != journal[
        "predicted_receipt_sha256"
    ]:
        raise VersionForwardTransactionError(
            "receipt publication journal hash drift"
        )

    def exact_regular(path: Path, label: str) -> os.stat_result:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise VersionForwardTransactionError(
                f"cannot open journal-bound {label}: {path}"
            ) from error
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink not in (1, 2):
                raise VersionForwardTransactionError(
                    f"linked, aliased, or non-regular {label}: {path}"
                )
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_nlink,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_nlink,
            ):
                raise VersionForwardTransactionError(
                    f"journal-bound {label} changed while open"
                )
            if b"".join(chunks) != expected:
                raise VersionForwardTransactionError(
                    f"competing or partial {label} differs from journal"
                )
            return after
        finally:
            os.close(descriptor)

    receipt_exists = os.path.lexists(RECEIPT_PATH)
    staging_exists = os.path.lexists(RECEIPT_STAGING_PATH)
    recovering_existing_alias = receipt_exists or staging_exists
    receipt_meta = (
        exact_regular(RECEIPT_PATH, "version-forward receipt")
        if receipt_exists
        else None
    )
    staging_meta = (
        exact_regular(RECEIPT_STAGING_PATH, "receipt staging file")
        if staging_exists
        else None
    )
    if receipt_meta is not None and staging_meta is not None:
        if (
            (receipt_meta.st_dev, receipt_meta.st_ino)
            != (staging_meta.st_dev, staging_meta.st_ino)
            or receipt_meta.st_nlink != 2
            or staging_meta.st_nlink != 2
        ):
            raise VersionForwardTransactionError(
                "receipt and staging files are competing aliases"
            )
    elif receipt_meta is not None and receipt_meta.st_nlink != 1:
        raise VersionForwardTransactionError(
            "completed version-forward receipt has unexpected links"
        )
    elif staging_meta is not None and staging_meta.st_nlink != 1:
        raise VersionForwardTransactionError(
            "unpublished receipt staging file has unexpected links"
        )
    if recovering_existing_alias:
        _fsync_exact_publication_alias(
            RECEIPT_STAGING_PATH if staging_meta is not None else RECEIPT_PATH,
            expected,
            label="version-forward receipt publication alias",
            boundary="receipt_recovery_inode_fsync",
        )
    if receipt_meta is None and staging_meta is None:
        RECEIPT_STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
        _write_staging_file(
            RECEIPT_STAGING_PATH, expected, prefix="receipt_staging"
        )
        staging_meta = RECEIPT_STAGING_PATH.lstat()
    if receipt_meta is None:
        try:
            os.link(RECEIPT_STAGING_PATH, RECEIPT_PATH)
        except FileExistsError as error:
            raise VersionForwardTransactionError(
                "competing version-forward receipt appeared during publication"
            ) from error
        _durability_boundary("receipt_link")
        _fsync_directory(RECEIPT_PATH.parent, "receipt_link_parent_fsync")
        receipt_meta = RECEIPT_PATH.lstat()
        staging_meta = RECEIPT_STAGING_PATH.lstat()
        if (
            (receipt_meta.st_dev, receipt_meta.st_ino)
            != (staging_meta.st_dev, staging_meta.st_ino)
            or receipt_meta.st_nlink != 2
        ):
            raise VersionForwardTransactionError(
                "published receipt hard-link identity drift"
            )
    if os.path.lexists(RECEIPT_STAGING_PATH):
        RECEIPT_STAGING_PATH.unlink()
        _durability_boundary("receipt_staging_unlink")
        _fsync_directory(
            RECEIPT_PATH.parent, "receipt_staging_unlink_parent_fsync"
        )
    final_meta = exact_regular(RECEIPT_PATH, "version-forward receipt")
    if final_meta.st_nlink != 1:
        raise VersionForwardTransactionError(
            "version-forward receipt publication did not close its link"
        )
    # This barrier is unconditional.  In particular, a recovery invocation
    # that finds only the final nlink-1 receipt must make a prior staging
    # unlink durable before it may remove the journal in another directory.
    _fsync_directory(
        RECEIPT_PATH.parent, "receipt_publication_recovery_parent_fsync"
    )


def _publish_pending(pending: Mapping[str, Any]) -> None:
    _require_adapter_lock()
    if os.path.lexists(TRANSACTION_PATH):
        raise VersionForwardTransactionError("unreconciled durable transaction exists")
    payload = _pretty_bytes(pending)
    _write_staging_file(PENDING_STAGING_PATH, payload, prefix="pending_staging")
    os.replace(PENDING_STAGING_PATH, TRANSACTION_PATH)
    _durability_boundary("pending_rename")
    _fsync_directory(TRANSACTION_PATH.parent, "pending_parent_fsync")


def _publish_state(state: Mapping[str, Any]) -> None:
    _require_adapter_lock()
    payload = _pretty_bytes(state)
    _write_staging_file(STATE_STAGING_PATH, payload, prefix="state_staging")
    os.replace(STATE_STAGING_PATH, STATE_PATH)
    _durability_boundary("state_rename")
    _fsync_directory(STATE_PATH.parent, "state_parent_fsync")


def _unlink_pending() -> None:
    _require_adapter_lock()
    TRANSACTION_PATH.unlink()
    _durability_boundary("pending_unlink")
    _fsync_directory(TRANSACTION_PATH.parent, "pending_unlink_parent_fsync")


def _preview_record(
    event: Mapping[str, Any], sequence: int, previous: str
) -> dict[str, Any]:
    if type(event.get("created_unix_ns")) is not int or int(
        event["created_unix_ns"]
    ) <= 0:
        raise VersionForwardTransactionError(
            "controller event lacks an exact positive fixed timestamp"
        )
    payload = dict(event)
    if "seq" in payload or "prev_sha256" in payload or "record_sha256" in payload:
        raise VersionForwardTransactionError(
            "controller event pre-populates adapter-owned chain fields"
        )
    payload["seq"] = sequence
    payload["prev_sha256"] = previous
    payload["record_sha256"] = _canonical_sha256(payload)
    return payload


def _operation_sha256(
    *,
    base_state: Mapping[str, Any],
    base_ledger_sha256: str,
    expected_suffix_sha256: str,
    intended_without_marker: Mapping[str, Any],
) -> str:
    return _canonical_sha256(
        {
            "base_state_object_sha256": _canonical_sha256(base_state),
            "base_ledger_sha256": base_ledger_sha256,
            "expected_suffix_sha256": expected_suffix_sha256,
            "intended_state_object_sha256": _canonical_sha256(
                intended_without_marker
            ),
        }
    )


def _adapter_marker(
    *,
    event_count: int,
    head_sha256: str,
    operation_sha256: str,
    state_without_marker: Mapping[str, Any],
    root_record: Mapping[str, Any],
    adapter_record: Mapping[str, Any],
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema_version": 2,
        "authorization_kind": ADAPTER_AUTHORIZATION_KIND,
        "adapter_source_path": adapter_record["path"],
        "adapter_source_sha256": adapter_record["sha256"],
        "adapter_source_ast_sha256": adapter_record["ast_sha256"],
        "root_program_path": root_record["path"],
        "root_program_sha256": root_record["sha256"],
        "root_program_ast_sha256": root_record["ast_sha256"],
        "ledger_event_count": event_count,
        "ledger_head_sha256": head_sha256,
        "operation_sha256": operation_sha256,
    }
    marker["state_binding_sha256"] = _canonical_sha256(
        {
            "marker_without_state_binding": marker,
            "state_without_marker": state_without_marker,
        }
    )
    return marker


def _validate_adapter_marker(
    state: Mapping[str, Any], chain: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_state_ledger_types(state, label="adapter-authorized STATE")
    marker = state.get(ADAPTER_STATE_KEY)
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
    state_without_marker = json.loads(json.dumps(dict(state)))
    state_without_marker.pop(ADAPTER_STATE_KEY, None)
    root = _source_record(PROGRAM_PATH)
    adapter = _source_record(TRANSACTION_SOURCE_PATH)
    marker_without_binding = dict(marker) if isinstance(marker, Mapping) else {}
    observed_state_binding = marker_without_binding.pop(
        "state_binding_sha256", None
    )
    checks = {
        "schema": isinstance(marker, Mapping) and set(marker) == expected_keys,
        "version": isinstance(marker, Mapping)
        and type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 2,
        "authorization": isinstance(marker, Mapping)
        and marker.get("authorization_kind") == ADAPTER_AUTHORIZATION_KIND,
        "adapter_path": isinstance(marker, Mapping)
        and marker.get("adapter_source_path") == adapter["path"],
        "adapter_hash": isinstance(marker, Mapping)
        and marker.get("adapter_source_sha256") == adapter["sha256"],
        "adapter_ast": isinstance(marker, Mapping)
        and marker.get("adapter_source_ast_sha256") == adapter["ast_sha256"],
        "root_path": isinstance(marker, Mapping)
        and marker.get("root_program_path") == root["path"],
        "root_hash": isinstance(marker, Mapping)
        and marker.get("root_program_sha256") == root["sha256"],
        "root_ast": isinstance(marker, Mapping)
        and marker.get("root_program_ast_sha256") == root["ast_sha256"],
        "count": isinstance(marker, Mapping)
        and type(marker.get("ledger_event_count")) is int
        and marker.get("ledger_event_count") == chain.get("event_count")
        == state.get("ledger_event_count"),
        "head": isinstance(marker, Mapping)
        and marker.get("ledger_head_sha256") == chain.get("head_sha256")
        == state.get("ledger_head_sha256"),
        "operation": isinstance(marker, Mapping)
        and _is_sha256(marker.get("operation_sha256")),
        "state_binding": _is_sha256(observed_state_binding)
        and observed_state_binding
        == _canonical_sha256(
            {
                "marker_without_state_binding": marker_without_binding,
                "state_without_marker": state_without_marker,
            }
        ),
    }
    if not all(checks.values()):
        raise VersionForwardTransactionError(
            f"v008 controller adapter authorization drift: {checks}"
        )
    return dict(marker)


def _validate_source_adapter_marker(
    state: Mapping[str, Any], chain: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate the live v007 marker from its receipt and ledger suffix."""

    marker = state.get(PRIOR_ADAPTER_STATE_KEY)
    expected_keys = {
        "schema_version", "authorization_kind", "adapter_source_path",
        "adapter_source_sha256", "adapter_source_ast_sha256",
        "root_program_path", "root_program_sha256", "root_program_ast_sha256",
        "ledger_event_count", "ledger_head_sha256", "operation_sha256",
        "state_binding_sha256",
    }
    if not isinstance(marker, Mapping) or set(marker) != expected_keys:
        raise VersionForwardTransactionError("source v007 adapter marker schema drift")
    adapter_path = REPOSITORY_ROOT / str(marker["adapter_source_path"])
    root_path = REPOSITORY_ROOT / str(marker["root_program_path"])
    marker_core = dict(marker)
    state_binding = marker_core.pop("state_binding_sha256", None)
    state_without_marker = json.loads(json.dumps(dict(state)))
    state_without_marker.pop(PRIOR_ADAPTER_STATE_KEY, None)
    checks = {
        "schema": type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 2,
        "authorization": marker.get("authorization_kind")
        == "receipt_bound_v007_root_controller_adapter",
        "adapter_path": marker.get("adapter_source_path")
        == "runs/lewm_domain_robust_gate/attempts/v007/version_forward_transaction.py",
        "adapter_hash": adapter_path.is_file()
        and marker.get("adapter_source_sha256") == _snapshot_file(adapter_path)[0]["sha256"],
        "adapter_ast": adapter_path.is_file()
        and marker.get("adapter_source_ast_sha256") == _source_ast_sha256(adapter_path),
        "root_path": root_path.resolve(strict=False) == PROGRAM_PATH.resolve(strict=True),
        "root_hash": marker.get("root_program_sha256")
        == _snapshot_file(PROGRAM_PATH)[0]["sha256"],
        "root_ast": marker.get("root_program_ast_sha256")
        == _source_ast_sha256(PROGRAM_PATH),
        "count": marker.get("ledger_event_count") == chain.get("event_count")
        == state.get("ledger_event_count"),
        "head": marker.get("ledger_head_sha256") == chain.get("head_sha256")
        == state.get("ledger_head_sha256"),
        "operation": _is_sha256(marker.get("operation_sha256")),
        "state_binding": state_binding
        == _canonical_sha256(
            {
                "marker_without_state_binding": marker_core,
                "state_without_marker": state_without_marker,
            }
        ),
    }
    if not all(checks.values()):
        raise VersionForwardTransactionError(
            f"source v007 adapter marker authorization drift: {checks}"
        )
    receipt_record, receipt_payload = _snapshot_file(PRIOR_RECEIPT_PATH)
    receipt = _json_payload(receipt_payload, "historical v007 receipt")
    post = receipt.get("post_snapshot")
    post_state = post.get("state") if isinstance(post, Mapping) else None
    post_ledger = post.get("ledger") if isinstance(post, Mapping) else None
    transaction_source = receipt.get("transaction_source")
    if (
        receipt_record.get("sha256")
        != "f038c25974c3ddd1bf7ea4ae9e9be68d1230a359653d6ba3c90558e9034b0818"
        or receipt.get("source_attempt") != ACTIVATION_SOURCE_ATTEMPT
        or receipt.get("target_attempt") != SOURCE_ATTEMPT
        or receipt.get("resume_state") != "SELECTION_COHORTS"
        or receipt.get("passed") is not True
        or not isinstance(post_state, Mapping)
        or not isinstance(post_state.get("object"), Mapping)
        or post_state.get("object_sha256")
        != _canonical_sha256(post_state["object"])
        or post_state["object"].get("active_attempt") != SOURCE_ATTEMPT
        or post_state["object"].get("current_state") != "SELECTION_COHORTS"
        or {
            key: post_state["object"].get(key) for key in ZERO_OUTCOMES
        }
        != ZERO_OUTCOMES
        or not isinstance(post_ledger, Mapping)
        or post_ledger.get("event_count") != 17
        or post_ledger.get("head_sha256")
        != "7486d4096773da9450cf145ca321e2eaabb907e4cc50cee460dc19d6f8714d45"
        or post_ledger.get("sha256")
        != "a6a9374dddddfc20254d8fd532252c045cec8a7c4ab31a06ea7a703de542d0db"
        or post_ledger.get("ledger_sha256") != post_ledger.get("sha256")
        or not isinstance(transaction_source, Mapping)
        or transaction_source.get("sha256") != marker.get("adapter_source_sha256")
    ):
        raise VersionForwardTransactionError(
            "historical v007 receipt boundary drift"
        )

    # The activation receipt is an immutable prefix.  Independently replay the
    # later v007 deterministic transactions from that exact prefix
    # to the supplied selection-boundary STATE instead of treating the receipt
    # as if it were a snapshot of mutable current state.
    ledger_payload = _regular_bytes(LEDGER_PATH, "research ledger")
    lines = ledger_payload.splitlines(keepends=True)
    source_count = state.get("ledger_event_count")
    if (
        type(source_count) is not int
        or source_count <= 0
        or source_count > len(lines)
        or any(not line.endswith(b"\n") for line in lines[:source_count])
    ):
        raise VersionForwardTransactionError(
            "live v007 ledger-prefix framing drift"
        )
    source_ledger_payload = b"".join(lines[:source_count])
    genesis_payload = _snapshot_file(GENESIS_PATH)[1]
    source_chain = _ledger_summary(source_ledger_payload, genesis_payload)
    if (
        source_chain.get("event_count") != chain.get("event_count")
        or source_chain.get("head_sha256") != chain.get("head_sha256")
        or source_chain.get("ledger_sha256") != chain.get("ledger_sha256")
        or source_chain.get("ledger_bytes") != chain.get("ledger_bytes")
    ):
        raise VersionForwardTransactionError(
            "live v007 ledger-prefix chain drift"
        )
    source_adapter = _load_module(
        "v007_source_adapter_replay",
        STUDY_ROOT / "attempts/v007/version_forward_transaction.py",
    )
    replay = getattr(source_adapter, "_validate_descendant_adapter_chain", None)
    if not callable(replay):
        raise VersionForwardTransactionError(
            "sealed v007 adapter lacks descendant replay"
        )
    try:
        replay(
            receipt_post_state=post_state["object"],
            receipt_post_ledger=post_ledger,
            current_state=state,
            ledger_payload=source_ledger_payload,
        )
    except Exception as error:
        raise VersionForwardTransactionError(
            "live v007 adapter ledger suffix failed scientific replay"
        ) from error
    return dict(marker)


def _bind_prior_marker_to_count_events(
    base_state: Mapping[str, Any],
    proposed_state: Mapping[str, Any],
    chain: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind the exact pre-count adapter marker into the sole count event.

    Count recovery must reproduce the exact pre-count STATE bytes.  The
    current adapter marker changes on every commit, so reversing only the
    count, timestamp, and ledger fields is insufficient.  A bounded copy of
    the immediately prior marker in the count event makes that one-step
    inverse exact without constructing a recursive marker chain.
    """

    bound = [dict(event) for event in events]
    if any(PRIOR_ADAPTER_MARKER_FIELD in event for event in bound):
        raise VersionForwardTransactionError(
            "controller event pre-populates adapter-owned prior marker"
        )
    count_events = [
        event for event in bound if event.get("event") == "outcome_counts_updated"
    ]
    if not count_events:
        return bound
    if len(bound) != 1 or len(count_events) != 1:
        raise VersionForwardTransactionError(
            "outcome-count transaction must contain exactly one event"
        )
    event = count_events[0]
    fields = event.get("fields")
    if (
        set(event) != {"event", "attempt", "fields", "created_unix_ns"}
        or event.get("attempt") != TARGET_ATTEMPT
        or not isinstance(fields, Mapping)
        or base_state.get("active_attempt") != TARGET_ATTEMPT
    ):
        raise VersionForwardTransactionError(
            "outcome-count event is not the exact v008 controller schema"
        )
    validated_fields = _validate_count_assignments(fields)
    if len(validated_fields) != 1:
        raise VersionForwardTransactionError(
            "outcome-count transaction must update exactly one counter"
        )
    _validate_count_transition_constraints(base_state, validated_fields)
    expected_state = json.loads(json.dumps(dict(base_state)))
    expected_state.update(validated_fields)
    expected_state["updated_unix_ns"] = event["created_unix_ns"]
    if dict(proposed_state) != expected_state:
        raise VersionForwardTransactionError(
            "outcome-count proposed STATE is not the exact base transition"
        )
    event[PRIOR_ADAPTER_MARKER_FIELD] = _validate_adapter_marker(
        base_state, chain
    )
    return bound


def _adapter_transaction_id(
    base_state: Mapping[str, Any],
    proposed_state_sha256: str,
    events: Sequence[Mapping[str, Any]],
) -> str:
    """Name one adapter transaction without depending on its ledger framing."""

    return _canonical_sha256(
        {
            "schema_version": 1,
            "authorization_kind": ADAPTER_TRANSACTION_ENVELOPE_KIND,
            "base_state_sha256": _canonical_sha256(base_state),
            "proposed_state_sha256": proposed_state_sha256,
            "events": [dict(event) for event in events],
        }
    )


def _bind_adapter_transaction_envelopes(
    base_state: Mapping[str, Any],
    proposed_state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind every record to one exact adapter-owned pre-state transaction."""

    bound = [dict(event) for event in events]
    if not bound or any(
        ADAPTER_TRANSACTION_ENVELOPE_FIELD in event for event in bound
    ):
        raise VersionForwardTransactionError(
            "controller event pre-populates adapter-owned transaction envelope"
        )
    base = json.loads(json.dumps(dict(base_state)))
    base_sha256 = _canonical_sha256(base)
    proposal = json.loads(json.dumps(dict(proposed_state)))
    if _canonical_bytes(proposal.get(ADAPTER_STATE_KEY)) != _canonical_bytes(
        base.get(ADAPTER_STATE_KEY)
    ):
        raise VersionForwardTransactionError(
            "controller proposal altered the adapter-owned base marker"
        )
    proposal.pop(ADAPTER_STATE_KEY, None)
    if (
        proposal.get("ledger_event_count") != base.get("ledger_event_count")
        or proposal.get("ledger_head_sha256")
        != base.get("ledger_head_sha256")
    ):
        raise VersionForwardTransactionError(
            "controller proposal pre-populates adapter-owned ledger state"
        )
    proposed_state_sha256 = _canonical_sha256(proposal)
    transaction_id = _adapter_transaction_id(
        base, proposed_state_sha256, bound
    )
    event_count = len(bound)
    for index, event in enumerate(bound):
        envelope: dict[str, Any] = {
            "schema_version": 1,
            "authorization_kind": ADAPTER_TRANSACTION_ENVELOPE_KIND,
            "transaction_id": transaction_id,
            "event_index": index,
            "event_count": event_count,
            "base_state_sha256": base_sha256,
            "proposed_state_sha256": proposed_state_sha256,
        }
        if index == 0:
            envelope["base_state"] = base
        event[ADAPTER_TRANSACTION_ENVELOPE_FIELD] = envelope
    return bound


def _validate_adapter_transaction_envelopes(
    base_state: Mapping[str, Any],
    intended_state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and remove pending envelopes before controller-event checks."""

    if not events:
        raise VersionForwardTransactionError(
            "pending adapter transaction has no enveloped events"
        )
    base = json.loads(json.dumps(dict(base_state)))
    base_sha256 = _canonical_sha256(base)
    proposal = json.loads(json.dumps(dict(intended_state)))
    proposal.pop(ADAPTER_STATE_KEY, None)
    proposal["ledger_event_count"] = base.get("ledger_event_count")
    proposal["ledger_head_sha256"] = base.get("ledger_head_sha256")
    proposed_state_sha256 = _canonical_sha256(proposal)
    stripped: list[dict[str, Any]] = []
    transaction_id: str | None = None
    event_count = len(events)
    for index, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping):
            raise VersionForwardTransactionError(
                "pending adapter transaction event schema drift"
            )
        event = dict(raw_event)
        envelope = event.pop(ADAPTER_TRANSACTION_ENVELOPE_FIELD, None)
        expected_keys = {
            "schema_version",
            "authorization_kind",
            "transaction_id",
            "event_index",
            "event_count",
            "base_state_sha256",
            "proposed_state_sha256",
        }
        if index == 0:
            expected_keys.add("base_state")
        if (
            not isinstance(envelope, Mapping)
            or set(envelope) != expected_keys
            or type(envelope.get("schema_version")) is not int
            or envelope.get("schema_version") != 1
            or envelope.get("authorization_kind")
            != ADAPTER_TRANSACTION_ENVELOPE_KIND
            or not _is_sha256(envelope.get("transaction_id"))
            or type(envelope.get("event_index")) is not int
            or envelope.get("event_index") != index
            or type(envelope.get("event_count")) is not int
            or envelope.get("event_count") != event_count
            or envelope.get("base_state_sha256") != base_sha256
            or envelope.get("proposed_state_sha256")
            != proposed_state_sha256
            or (
                index == 0
                and (
                    not isinstance(envelope.get("base_state"), Mapping)
                    or _canonical_bytes(envelope["base_state"])
                    != _canonical_bytes(base)
                )
            )
        ):
            raise VersionForwardTransactionError(
                "pending adapter transaction envelope drift"
            )
        if transaction_id is None:
            transaction_id = str(envelope["transaction_id"])
        elif envelope.get("transaction_id") != transaction_id:
            raise VersionForwardTransactionError(
                "pending adapter transaction envelope identity drift"
            )
        stripped.append(event)
    if transaction_id != _adapter_transaction_id(
        base, proposed_state_sha256, stripped
    ):
        raise VersionForwardTransactionError(
            "pending adapter transaction envelope content drift"
        )
    return stripped


def _validate_bound_count_events(
    base_state: Mapping[str, Any],
    intended_state: Mapping[str, Any],
    chain: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    target_event_count: int,
    target_head_sha256: str,
) -> None:
    """Independently rederive every adapter-owned count-event binding."""

    count_events = [
        event for event in events if event.get("event") == "outcome_counts_updated"
    ]
    if count_events:
        if len(events) != 1 or len(count_events) != 1:
            raise VersionForwardTransactionError(
                "pending outcome-count transaction is not singular"
            )
        event = count_events[0]
        expected_keys = {
            "event",
            "attempt",
            "fields",
            "created_unix_ns",
            PRIOR_ADAPTER_MARKER_FIELD,
        }
        expected_marker = _validate_adapter_marker(base_state, chain)
        fields = event.get("fields")
        if (
            set(event) != expected_keys
            or event.get("attempt") != TARGET_ATTEMPT
            or not isinstance(fields, Mapping)
            or event.get(PRIOR_ADAPTER_MARKER_FIELD) != expected_marker
        ):
            raise VersionForwardTransactionError(
                "pending outcome-count prior adapter marker drift"
            )
        validated_fields = _validate_count_assignments(fields)
        if len(validated_fields) != 1:
            raise VersionForwardTransactionError(
                "pending outcome-count transition is not singular"
            )
        expected_state = json.loads(json.dumps(dict(base_state)))
        expected_state.update(validated_fields)
        expected_state["updated_unix_ns"] = event["created_unix_ns"]
        expected_state["ledger_event_count"] = target_event_count
        expected_state["ledger_head_sha256"] = target_head_sha256
        expected_state.pop(ADAPTER_STATE_KEY, None)
        observed_state = dict(intended_state)
        observed_state.pop(ADAPTER_STATE_KEY, None)
        if observed_state != expected_state:
            raise VersionForwardTransactionError(
                "pending outcome-count intended STATE transition drift"
            )
        return
    if any(PRIOR_ADAPTER_MARKER_FIELD in event for event in events):
        raise VersionForwardTransactionError(
            "non-count event carries an adapter-owned prior marker"
        )


def _pending_digest_v2(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("transaction_sha256", None)
    return _canonical_sha256(payload)


def _bind_receipt_context_to_version_forward_event(
    base_state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    receipt_context_sha256: str | None,
) -> list[dict[str, Any]]:
    bound = [dict(event) for event in events]
    if any(RECEIPT_CONTEXT_EVENT_FIELD in event for event in bound):
        raise VersionForwardTransactionError(
            "controller event pre-populates adapter-owned receipt context"
        )
    forward = [
        event
        for event in bound
        if event.get("event") == "zero_confirmation_outcome_version_forward"
    ]
    if not forward:
        if receipt_context_sha256 is not None:
            raise VersionForwardTransactionError(
                "receipt context supplied for a non-forward controller event"
            )
        return bound
    if (
        len(bound) != 1
        or len(forward) != 1
        or base_state.get("active_attempt") != SOURCE_ATTEMPT
        or not _is_sha256(receipt_context_sha256)
    ):
        raise VersionForwardTransactionError(
            "version-forward event lacks one exact receipt context"
        )
    forward[0][RECEIPT_CONTEXT_EVENT_FIELD] = receipt_context_sha256
    return bound


def _rollover_source_adapter_marker(
    base_state: Mapping[str, Any],
    proposed_state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    chain: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Remove only the authenticated v007 marker and bind it into the edge."""

    marker = _validate_source_adapter_marker(base_state, chain)
    if proposed_state.get(PRIOR_ADAPTER_STATE_KEY) != marker:
        raise VersionForwardTransactionError(
            "root version-forward proposal did not preserve the exact source marker"
        )
    if ADAPTER_STATE_KEY in proposed_state:
        raise VersionForwardTransactionError(
            "root version-forward proposal pre-populated the v008 adapter marker"
        )
    transformed_state = json.loads(json.dumps(dict(proposed_state)))
    removed = transformed_state.pop(PRIOR_ADAPTER_STATE_KEY, None)
    if removed != marker:
        raise VersionForwardTransactionError("source marker rollover removal drift")
    transformed_events = [dict(event) for event in events]
    if (
        len(transformed_events) != 1
        or transformed_events[0].get("event")
        != "zero_confirmation_outcome_version_forward"
        or transformed_events[0].get("attempt") != TARGET_ATTEMPT
        or SOURCE_ADAPTER_MARKER_FIELD in transformed_events[0]
    ):
        raise VersionForwardTransactionError(
            "source marker rollover requires one exact forward event"
        )
    transformed_events[0][SOURCE_ADAPTER_MARKER_FIELD] = marker
    return transformed_state, transformed_events


def _validate_bound_receipt_context_event(
    base_state: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    forward = [
        event
        for event in events
        if event.get("event") == "zero_confirmation_outcome_version_forward"
    ]
    if forward:
        if (
            len(events) != 1
            or len(forward) != 1
            or base_state.get("active_attempt") != SOURCE_ATTEMPT
            or not _is_sha256(forward[0].get(RECEIPT_CONTEXT_EVENT_FIELD))
            or forward[0].get(SOURCE_ADAPTER_MARKER_FIELD)
            != base_state.get(PRIOR_ADAPTER_STATE_KEY)
        ):
            raise VersionForwardTransactionError(
                "pending version-forward receipt context drift"
            )
        return
    if any(RECEIPT_CONTEXT_EVENT_FIELD in event for event in events):
        raise VersionForwardTransactionError(
            "non-forward event carries an adapter-owned receipt context"
        )
    if any(SOURCE_ADAPTER_MARKER_FIELD in event for event in events):
        raise VersionForwardTransactionError(
            "non-forward event carries a source adapter marker"
        )


def _assemble_pending(
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    base_state: Mapping[str, Any],
    base_state_payload: bytes,
    genesis_record: Mapping[str, Any],
    genesis_payload: bytes,
    ledger_payload: bytes,
    receipt_context_sha256: str | None,
) -> dict[str, Any]:
    if not events:
        raise VersionForwardTransactionError("controller transaction has no events")
    _validate_outcome_counter_types(base_state)
    _validate_outcome_counter_types(state)
    _validate_state_ledger_types(base_state, label="pending base STATE")
    _validate_state_ledger_types(state, label="pending proposed STATE")
    chain = _ledger_summary(ledger_payload, genesis_payload)
    if (
        state.get("ledger_event_count") != chain["event_count"]
        or state.get("ledger_head_sha256") != chain["head_sha256"]
        or base_state.get("ledger_event_count") != chain["event_count"]
        or base_state.get("ledger_head_sha256") != chain["head_sha256"]
    ):
        raise VersionForwardTransactionError(
            "controller transaction base does not match strict ledger framing"
        )
    if base_state.get("active_attempt") == TARGET_ATTEMPT:
        _validate_adapter_marker(base_state, chain)
    elif base_state.get("active_attempt") == SOURCE_ATTEMPT:
        source_marker = _validate_source_adapter_marker(base_state, chain)
        if PRIOR_ADAPTER_STATE_KEY in state or ADAPTER_STATE_KEY in state:
            raise VersionForwardTransactionError(
                "forward proposal retained or pre-populated an adapter marker"
            )
        if (
            len(events) != 1
            or events[0].get(SOURCE_ADAPTER_MARKER_FIELD) != source_marker
        ):
            raise VersionForwardTransactionError(
                "forward event lacks its exact source adapter marker"
            )
    else:
        raise VersionForwardTransactionError("unsupported controller lineage base")
    context_bound_events = _bind_receipt_context_to_version_forward_event(
        base_state, events, receipt_context_sha256
    )
    bound_events = _bind_prior_marker_to_count_events(
        base_state, state, chain, context_bound_events
    )
    if base_state.get("active_attempt") == TARGET_ATTEMPT:
        bound_events = _bind_adapter_transaction_envelopes(
            base_state, state, bound_events
        )
    previews: list[dict[str, Any]] = []
    previous = str(chain["head_sha256"])
    for offset, event in enumerate(bound_events, start=1):
        preview = _preview_record(
            event, int(chain["event_count"]) + offset, previous
        )
        previews.append(preview)
        previous = str(preview["record_sha256"])
    suffix = b"".join(_canonical_bytes(record) + b"\n" for record in previews)
    suffix_sha256 = hashlib.sha256(suffix).hexdigest()
    intended_without_marker = json.loads(json.dumps(dict(state)))
    intended_without_marker["ledger_event_count"] = int(chain["event_count"]) + len(
        previews
    )
    intended_without_marker["ledger_head_sha256"] = previous
    intended_without_marker.pop(ADAPTER_STATE_KEY, None)
    operation_sha256 = _operation_sha256(
        base_state=base_state,
        base_ledger_sha256=hashlib.sha256(ledger_payload).hexdigest(),
        expected_suffix_sha256=suffix_sha256,
        intended_without_marker=intended_without_marker,
    )
    root_record = _source_record(PROGRAM_PATH)
    adapter_record = _source_record(TRANSACTION_SOURCE_PATH)
    intended = dict(intended_without_marker)
    intended[ADAPTER_STATE_KEY] = _adapter_marker(
        event_count=intended["ledger_event_count"],
        head_sha256=previous,
        operation_sha256=operation_sha256,
        state_without_marker=intended_without_marker,
        root_record=root_record,
        adapter_record=adapter_record,
    )
    pending: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": ADAPTER_ARTIFACT_TYPE,
        "authorization_kind": ADAPTER_AUTHORIZATION_KIND,
        "created_unix_ns": min(
            int(event["created_unix_ns"]) for event in bound_events
        ),
        "source_attempt": base_state.get("active_attempt"),
        "target_attempt": intended.get("active_attempt"),
        "root_program": root_record,
        "adapter_source": adapter_record,
        "ledger_genesis": genesis_record,
        "base_state": base_state,
        "base_state_bytes": len(base_state_payload),
        "base_state_file_sha256": hashlib.sha256(base_state_payload).hexdigest(),
        "base_state_object_sha256": _canonical_sha256(base_state),
        "base_ledger_bytes": len(ledger_payload),
        "base_ledger_sha256": hashlib.sha256(ledger_payload).hexdigest(),
        "base_event_count": chain["event_count"],
        "base_head_sha256": chain["head_sha256"],
        "events": bound_events,
        "expected_records": previews,
        "expected_suffix": suffix.decode("ascii"),
        "expected_suffix_sha256": suffix_sha256,
        "intended_state": intended,
        "target_event_count": intended["ledger_event_count"],
        "target_head_sha256": intended["ledger_head_sha256"],
        "operation_sha256": operation_sha256,
    }
    pending["transaction_sha256"] = _pending_digest_v2(pending)
    return pending


def _build_pending(
    state: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    _require_adapter_lock()
    base_state, base_state_payload = _canonical_object_file(STATE_PATH, "STATE.json")
    genesis_record, genesis_payload = _snapshot_file(GENESIS_PATH)
    ledger_payload = _snapshot_file(LEDGER_PATH)[1]
    if base_state.get("active_attempt") == SOURCE_ATTEMPT:
        state, events = _rollover_source_adapter_marker(
            base_state,
            state,
            events,
            _ledger_summary(ledger_payload, genesis_payload),
        )
    return _assemble_pending(
        state,
        events,
        base_state=base_state,
        base_state_payload=base_state_payload,
        genesis_record=genesis_record,
        genesis_payload=genesis_payload,
        ledger_payload=ledger_payload,
        receipt_context_sha256=getattr(
            _ADAPTER_LOCAL, "receipt_context_sha256", None
        ),
    )


def _validate_source_record(
    value: Any, *, expected_path: Path, label: str
) -> dict[str, Any]:
    expected_keys = {
        "path",
        "sha256",
        "bytes",
        "device",
        "inode",
        "nlink",
        "ast_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise VersionForwardTransactionError(f"{label} source-record schema drift")
    if (
        not isinstance(value.get("path"), str)
        or value.get("path") != _relative(expected_path)
        or not _is_sha256(value.get("sha256"))
        or not _is_sha256(value.get("ast_sha256"))
        or any(
            type(value.get(key)) is not int
            for key in ("bytes", "device", "inode", "nlink")
        )
        or any(int(value[key]) < 0 for key in ("bytes", "device", "inode"))
        or value.get("nlink") != 1
    ):
        raise VersionForwardTransactionError(
            f"{label} source-record metadata type drift"
        )
    observed = _source_record(expected_path)
    if _canonical_bytes(value) != _canonical_bytes(observed):
        raise VersionForwardTransactionError(f"{label} source-record identity drift")
    return dict(value)


def _validate_pending(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "artifact_type",
        "authorization_kind",
        "created_unix_ns",
        "source_attempt",
        "target_attempt",
        "root_program",
        "adapter_source",
        "ledger_genesis",
        "base_state",
        "base_state_bytes",
        "base_state_file_sha256",
        "base_state_object_sha256",
        "base_ledger_bytes",
        "base_ledger_sha256",
        "base_event_count",
        "base_head_sha256",
        "events",
        "expected_records",
        "expected_suffix",
        "expected_suffix_sha256",
        "intended_state",
        "target_event_count",
        "target_head_sha256",
        "operation_sha256",
        "transaction_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise VersionForwardTransactionError("pending adapter transaction schema drift")
    pending = dict(value)
    if not (
        type(pending.get("schema_version")) is int
        and pending.get("schema_version") == 2
        and pending.get("artifact_type") == ADAPTER_ARTIFACT_TYPE
        and pending.get("authorization_kind") == ADAPTER_AUTHORIZATION_KIND
        and type(pending.get("created_unix_ns")) is int
        and int(pending["created_unix_ns"]) > 0
        and pending.get("source_attempt") in (SOURCE_ATTEMPT, TARGET_ATTEMPT)
        and pending.get("target_attempt") == TARGET_ATTEMPT
        and pending.get("transaction_sha256") == _pending_digest_v2(pending)
    ):
        raise VersionForwardTransactionError("pending adapter transaction header drift")
    _validate_source_record(
        pending["root_program"], expected_path=PROGRAM_PATH, label="root program"
    )
    _validate_source_record(
        pending["adapter_source"],
        expected_path=TRANSACTION_SOURCE_PATH,
        label="adapter",
    )
    genesis_record = pending.get("ledger_genesis")
    current_genesis, _payload = _snapshot_file(GENESIS_PATH)
    genesis_keys = {"path", "sha256", "bytes", "device", "inode", "nlink"}
    if (
        not isinstance(genesis_record, Mapping)
        or set(genesis_record) != genesis_keys
        or not isinstance(genesis_record.get("path"), str)
        or genesis_record.get("path") != _relative(GENESIS_PATH)
        or not _is_sha256(genesis_record.get("sha256"))
        or any(
            type(genesis_record.get(key)) is not int
            for key in ("bytes", "device", "inode", "nlink")
        )
        or any(
            int(genesis_record[key]) < 0
            for key in ("bytes", "device", "inode")
        )
        or genesis_record.get("nlink") != 1
    ):
        raise VersionForwardTransactionError(
            "pending ledger-genesis schema/type drift"
        )
    if _canonical_bytes(genesis_record) != _canonical_bytes(current_genesis):
        raise VersionForwardTransactionError("pending ledger-genesis identity drift")
    base_state = pending.get("base_state")
    intended = pending.get("intended_state")
    events = pending.get("events")
    records = pending.get("expected_records")
    if (
        not isinstance(base_state, Mapping)
        or not isinstance(intended, Mapping)
        or not isinstance(events, list)
        or not events
        or not all(isinstance(event, Mapping) for event in events)
        or not isinstance(records, list)
        or len(records) != len(events)
        or not all(isinstance(record, Mapping) for record in records)
    ):
        raise VersionForwardTransactionError("pending adapter transaction object drift")
    _validate_outcome_counter_types(base_state)
    _validate_outcome_counter_types(intended)
    _validate_state_ledger_types(base_state, label="pending base STATE")
    _validate_state_ledger_types(intended, label="pending intended STATE")
    base_bytes = _pretty_bytes(base_state)
    if not (
        type(pending.get("base_state_bytes")) is int
        and pending["base_state_bytes"] == len(base_bytes)
        and pending.get("base_state_file_sha256")
        == hashlib.sha256(base_bytes).hexdigest()
        and pending.get("base_state_object_sha256") == _canonical_sha256(base_state)
        and type(pending.get("base_ledger_bytes")) is int
        and int(pending["base_ledger_bytes"]) > 0
        and _is_sha256(pending.get("base_ledger_sha256"))
        and type(pending.get("base_event_count")) is int
        and int(pending["base_event_count"]) > 0
        and isinstance(pending.get("base_head_sha256"), str)
        and base_state.get("active_attempt") == pending.get("source_attempt")
        and type(pending.get("target_event_count")) is int
        and int(pending["target_event_count"]) > 0
    ):
        raise VersionForwardTransactionError("pending adapter base binding drift")
    base_chain = {
        "event_count": int(pending["base_event_count"]),
        "head_sha256": str(pending["base_head_sha256"]),
    }
    if pending.get("source_attempt") == TARGET_ATTEMPT:
        controller_events = _validate_adapter_transaction_envelopes(
            base_state, intended, events
        )
    else:
        if any(
            ADAPTER_TRANSACTION_ENVELOPE_FIELD in event for event in events
        ):
            raise VersionForwardTransactionError(
                "initial version-forward event carries a descendant envelope"
            )
        controller_events = [dict(event) for event in events]
    _validate_bound_receipt_context_event(base_state, controller_events)
    _validate_bound_count_events(
        base_state,
        intended,
        base_chain,
        controller_events,
        target_event_count=int(pending["target_event_count"]),
        target_head_sha256=str(pending["target_head_sha256"]),
    )
    previous = str(pending["base_head_sha256"])
    recomputed: list[dict[str, Any]] = []
    for offset, event in enumerate(events, start=1):
        preview = _preview_record(
            event, int(pending["base_event_count"]) + offset, previous
        )
        recomputed.append(preview)
        previous = str(preview["record_sha256"])
    suffix = b"".join(_canonical_bytes(record) + b"\n" for record in recomputed)
    try:
        declared_suffix = str(pending["expected_suffix"]).encode("ascii")
    except UnicodeEncodeError as error:
        raise VersionForwardTransactionError("pending suffix is not ASCII") from error
    if not (
        _canonical_bytes(records) == _canonical_bytes(recomputed)
        and declared_suffix == suffix
        and pending.get("expected_suffix_sha256")
        == hashlib.sha256(suffix).hexdigest()
        and pending.get("target_event_count")
        == int(pending["base_event_count"]) + len(recomputed)
        and pending.get("target_head_sha256") == previous
        and intended.get("ledger_event_count") == pending["target_event_count"]
        and intended.get("ledger_head_sha256") == pending["target_head_sha256"]
        and intended.get("active_attempt") == pending["target_attempt"]
    ):
        raise VersionForwardTransactionError("pending expected ledger/state drift")
    without_marker = dict(intended)
    marker = without_marker.pop(ADAPTER_STATE_KEY, None)
    operation = _operation_sha256(
        base_state=base_state,
        base_ledger_sha256=str(pending["base_ledger_sha256"]),
        expected_suffix_sha256=str(pending["expected_suffix_sha256"]),
        intended_without_marker=without_marker,
    )
    expected_marker = _adapter_marker(
        event_count=int(pending["target_event_count"]),
        head_sha256=str(pending["target_head_sha256"]),
        operation_sha256=operation,
        state_without_marker=without_marker,
        root_record=pending["root_program"],
        adapter_record=pending["adapter_source"],
    )
    if (
        pending.get("operation_sha256") != operation
        or marker != expected_marker
    ):
        raise VersionForwardTransactionError("pending adapter marker drift")
    if pending.get("source_attempt") == TARGET_ATTEMPT:
        proposed = json.loads(json.dumps(dict(intended)))
        proposed.pop(ADAPTER_STATE_KEY, None)
        proposed["ledger_event_count"] = base_state["ledger_event_count"]
        proposed["ledger_head_sha256"] = base_state["ledger_head_sha256"]
        _validate_replayed_controller_transition(
            base_state=base_state,
            proposed_state=proposed,
            events=controller_events,
        )
    return pending


def _validate_replayed_controller_transition(
    *,
    base_state: Mapping[str, Any],
    proposed_state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> None:
    """Recheck the bounded controller semantics for one historical adapter group."""

    def active_json_object(relative: Any, label: str) -> tuple[Path, dict[str, Any]]:
        if not isinstance(relative, str):
            raise VersionForwardTransactionError(
                f"replayed {label} path is absent"
            )
        try:
            path = (REPOSITORY_ROOT / relative).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise VersionForwardTransactionError(
                f"replayed {label} is absent"
            ) from error
        if not path.is_relative_to(ATTEMPT_ROOT.resolve()):
            raise VersionForwardTransactionError(
                f"replayed {label} escapes the active attempt"
            )
        record, payload = _snapshot_file(path)
        return path, _json_payload(payload, f"replayed {label}")

    def proves_integrity_failure(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered in {
                    "execution_invalid", "integrity_failure", "integrity_failed",
                } and child is True:
                    return True
                if lowered in {
                    "integrity_valid", "integrity_passed", "process_valid",
                    "all_integrity_checks_passed",
                } and child is False:
                    return True
                if proves_integrity_failure(child):
                    return True
        elif isinstance(value, list):
            return any(proves_integrity_failure(item) for item in value)
        return False

    def linked_sha256(value: Any) -> Any:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return value.get("sha256")
        return None

    base = json.loads(json.dumps(dict(base_state)))
    base_marker = base.pop(ADAPTER_STATE_KEY, None)
    proposal = json.loads(json.dumps(dict(proposed_state)))
    sequence = [event.get("event") for event in events]
    if not events or any(
        type(event.get("created_unix_ns")) is not int
        or int(event["created_unix_ns"]) <= 0
        or event.get("attempt") != TARGET_ATTEMPT
        for event in events
    ):
        raise VersionForwardTransactionError(
            "replayed controller-event identity/type drift"
        )
    if sequence == ["outcome_counts_updated"]:
        event = events[0]
        fields = event.get("fields")
        if (
            set(event)
            != {
                "event", "attempt", "fields", PRIOR_ADAPTER_MARKER_FIELD,
                "created_unix_ns",
            }
            or not isinstance(fields, Mapping)
            or len(fields) != 1
            or event.get(PRIOR_ADAPTER_MARKER_FIELD) != base_marker
        ):
            raise VersionForwardTransactionError(
                "replayed count controller-event schema drift"
            )
        validated = _validate_count_assignments(fields)
        if len(validated) != 1:
            raise VersionForwardTransactionError(
                "replayed count assignment is not singular"
            )
        _validate_count_transition_constraints(base_state, validated)
        expected = json.loads(json.dumps(base))
        expected.update(validated)
        expected["updated_unix_ns"] = event["created_unix_ns"]
        if _canonical_bytes(proposal) != _canonical_bytes(expected):
            raise VersionForwardTransactionError(
                "replayed count controller transition drift"
            )
        return
    if sequence == ["scientific_terminal_decision_recorded"]:
        event = events[0]
        if (
            set(event)
            != {
                "event", "attempt", "terminal_label", "process_valid",
                "decision_path", "decision_sha256", "created_unix_ns",
            }
            or base.get("current_state") != "TERMINAL"
            or base.get("early_scientific_failure") is not None
            or base.get("terminal_label") is not None
            or event.get("terminal_label")
            not in {
                "domain_robust_gate_confirmed", "domain_robust_gate_partial",
                "domain_robust_gate_failed",
            }
            or type(event.get("process_valid")) is not bool
            or not isinstance(event.get("decision_path"), str)
            or not _is_sha256(event.get("decision_sha256"))
        ):
            raise VersionForwardTransactionError(
                "replayed terminal controller-event schema drift"
            )
        decision_path, decision_object = active_json_object(
            event["decision_path"], "terminal decision"
        )
        if (
            _snapshot_file(decision_path)[0]["sha256"]
            != event["decision_sha256"]
            or decision_object.get("attempt") != TARGET_ATTEMPT
            or decision_object.get("terminal_label")
            != event["terminal_label"]
            or type(decision_object.get("process_valid")) is not bool
            or decision_object.get("process_valid")
            is not event["process_valid"]
        ):
            raise VersionForwardTransactionError(
                "replayed terminal decision evidence drift"
            )
        expected = json.loads(json.dumps(base))
        expected.update(
            {
                "terminal_label": event["terminal_label"],
                "process_valid": event["process_valid"],
                "scientific_terminal": True,
                "confirmation_terminal": True,
                "terminal_decision_path": event["decision_path"],
                "terminal_decision_sha256": event["decision_sha256"],
                "updated_unix_ns": event["created_unix_ns"],
            }
        )
        if _canonical_bytes(proposal) != _canonical_bytes(expected):
            raise VersionForwardTransactionError(
                "replayed terminal controller transition drift"
            )
        return
    if sequence == ["state_completed"]:
        event = events[0]
        ordinary = {
            "event", "attempt", "completed_state", "next_state",
            "checkpoint_name", "evidence_path", "evidence_sha256",
            "created_unix_ns",
        }
        integrity = ordinary | {
            "postconfirmation_integrity_failure", "integrity_source",
            "skipped_states",
        }
        evidence_path = event.get("evidence_path")
        evidence = (
            (REPOSITORY_ROOT / evidence_path).resolve(strict=True)
            if isinstance(evidence_path, str)
            else None
        )
        evidence_record, evidence_payload = (
            _snapshot_file(evidence) if evidence is not None else ({}, b"")
        )
        evidence_object = (
            _json_payload(evidence_payload, "state-completed evidence")
            if evidence_payload
            else {}
        )
        if not (
            set(event) in (ordinary, integrity)
            and event.get("completed_state") == base.get("current_state")
            and isinstance(event.get("checkpoint_name"), str)
            and _is_sha256(event.get("evidence_sha256"))
            and evidence is not None
            and evidence.is_relative_to(ATTEMPT_ROOT.resolve())
            and evidence_record.get("sha256")
            == event.get("evidence_sha256")
        ):
            raise VersionForwardTransactionError(
                "replayed state-completed controller transition drift"
            )
        completed_state = str(event["completed_state"])
        created = int(event["created_unix_ns"])
        expected = json.loads(json.dumps(base))
        completed_before = base.get("completed_states")
        checkpoints_before = base.get("verified_checkpoints")
        if not isinstance(completed_before, list) or not isinstance(
            checkpoints_before, list
        ):
            raise VersionForwardTransactionError(
                "replayed state-completed controller history drift"
            )
        if set(event) == ordinary:
            if completed_state not in CONTROLLER_STATE_MACHINE:
                raise VersionForwardTransactionError(
                    "replayed state-completed state-machine drift"
                )
            index = CONTROLLER_STATE_MACHINE.index(completed_state)
            final_transition = completed_state == "POST_TERMINAL_REPORTING"
            next_state = (
                None if final_transition else CONTROLLER_STATE_MACHINE[index + 1]
            )
            if (
                event.get("next_state") != next_state
                or evidence_object.get("passed") is not True
                or evidence_object.get("attempt") != TARGET_ATTEMPT
                or evidence_object.get("checkpoint_state") != completed_state
                or (
                    completed_state == "INDEPENDENT_VERIFICATION"
                    and evidence_object.get("terminal_label")
                    == "domain_robust_gate_execution_invalid"
                )
                or not isinstance(proposal.get("next_action"), str)
                or (
                    base.get("early_scientific_failure") is not None
                    and completed_state != "POST_TERMINAL_REPORTING"
                )
                or (
                    base.get("postconfirmation_integrity_failure") is not None
                    and completed_state != "POST_TERMINAL_REPORTING"
                )
                or (
                    base.get("terminal_label") is not None
                    and completed_state
                    not in {"TERMINAL", "POST_TERMINAL_REPORTING"}
                )
            ):
                raise VersionForwardTransactionError(
                    "replayed ordinary state-completed authorization drift"
                )
            checkpoint = {
                "name": event["checkpoint_name"],
                "created_unix_ns": created,
                "evidence_path": evidence_path,
                "evidence_sha256": event["evidence_sha256"],
                "source_attempt": TARGET_ATTEMPT,
                "verification_lineage": "direct_checkpoint",
            }
            expected["completed_states"] = completed_before + [completed_state]
            expected["current_state"] = (
                "POST_TERMINAL_REPORTING" if final_transition else next_state
            )
            expected["last_verified_checkpoint"] = checkpoint
            expected["verified_checkpoints"] = checkpoints_before + [checkpoint]
            expected["next_action"] = proposal["next_action"]
            if final_transition:
                if (
                    base.get("terminal_label") is None
                    or base.get("process_valid") is None
                    or base.get("post_terminal_reporting_complete") is True
                    or proposal.get("next_action")
                    != "program_complete_no_further_scientific_or_reporting_transition"
                ):
                    raise VersionForwardTransactionError(
                        "replayed post-terminal reporting completion drift"
                    )
                expected["post_terminal_reporting_complete"] = True
            if completed_state == "PREREGISTRATION_AND_POWER":
                registered = evidence_object.get("registered_counts")
                fixed = {
                    "expected_fit_episode_count": 1200,
                    "expected_selection_episode_count": 2000,
                    "expected_smoke_episode_count": 24,
                    "maximum_confirmation_episode_count": 18000,
                }
                if registered != fixed or any(
                    type(registered[key]) is not int for key in fixed
                ):
                    raise VersionForwardTransactionError(
                        "replayed preregistered role-count projection drift"
                    )
                expected.update(fixed)
            if completed_state == "CONFIRMATION_POWER_AND_COHORT_FREEZE":
                per_regime = evidence_object.get(
                    "fixed_confirmation_episodes_per_regime"
                )
                total = evidence_object.get("fixed_confirmation_episode_count")
                if not (
                    type(per_regime) is int
                    and per_regime in range(500, 4501, 500)
                    and type(total) is int
                    and total == 4 * per_regime
                ):
                    raise VersionForwardTransactionError(
                        "replayed fixed confirmation-size projection drift"
                    )
                expected["expected_confirmation_episodes_per_regime"] = per_regime
                expected["expected_confirmation_episode_count"] = total
        else:
            spec = POSTCONFIRMATION_FAILURE_TRANSITIONS.get(completed_state)
            skipped = (
                list(
                    CONTROLLER_STATE_MACHINE[
                        CONTROLLER_STATE_MACHINE.index(completed_state) + 1:
                        CONTROLLER_STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
                    ]
                )
                if spec is not None
                else None
            )
            expected_path = (
                _relative(ATTEMPT_ROOT / str(spec["source_relative"]))
                if spec is not None
                else None
            )
            if not (
                spec is not None
                and event.get("next_state") == "INDEPENDENT_VERIFICATION"
                and event.get("postconfirmation_integrity_failure") is True
                and event.get("integrity_source") == spec["source"]
                and event.get("skipped_states") == skipped
                and evidence_path == expected_path
                and evidence_object.get("attempt") == TARGET_ATTEMPT
                and evidence_object.get("checkpoint_state") == completed_state
                and evidence_object.get("passed") is False
                and base.get("early_scientific_failure") is None
                and base.get("terminal_label") is None
            ):
                raise VersionForwardTransactionError(
                    "replayed integrity-failure staging drift"
                )
            expected_confirmation = base.get(
                "expected_confirmation_episode_count"
            )
            if not (
                type(expected_confirmation) is int
                and expected_confirmation > 0
                and type(base.get("expected_smoke_episode_count")) is int
                and type(base.get("smoke_outcome_episodes")) is int
                and base.get("smoke_outcome_episodes")
                == base.get("expected_smoke_episode_count")
                and type(base.get("confirmation_outcome_episodes_generated"))
                is int
                and base.get("confirmation_outcome_episodes_generated")
                == expected_confirmation
                and type(base.get("confirmation_outcome_episodes_executed"))
                is int
                and base.get("confirmation_outcome_episodes_executed")
                == expected_confirmation
                and base.get("confirmation_outcomes_opened_for_analysis") is True
                and proves_integrity_failure(evidence_object)
            ):
                raise VersionForwardTransactionError(
                    "replayed integrity fixed-open proof drift"
                )
            if completed_state == "SEALED_ANALYSIS":
                analysis_result = ATTEMPT_ROOT / "analysis_result.json"
                if not (
                    type(
                        evidence_object.get(
                            "confirmation_outcome_episodes_generated"
                        )
                    ) is int
                    and evidence_object.get(
                        "confirmation_outcome_episodes_generated"
                    ) == expected_confirmation
                    and type(
                        evidence_object.get(
                            "confirmation_outcome_episodes_executed"
                        )
                    ) is int
                    and evidence_object.get(
                        "confirmation_outcome_episodes_executed"
                    ) == expected_confirmation
                    and evidence_object.get(
                        "confirmation_outcomes_opened_for_analysis"
                    ) is True
                    and type(evidence_object.get("analysis_result_present")) is bool
                    and evidence_object.get("analysis_result_present")
                    is analysis_result.is_file()
                    and isinstance(evidence_object.get("error_type"), str)
                    and bool(evidence_object["error_type"])
                    and isinstance(evidence_object.get("error"), str)
                    and bool(evidence_object["error"])
                    and evidence_object.get("scientific_objects_changed") is False
                ):
                    raise VersionForwardTransactionError(
                        "replayed analysis-invalid evidence drift"
                    )
            checkpoint = {
                "name": f"{TARGET_ATTEMPT}_{spec['source']}_integrity_failure",
                "created_unix_ns": created,
                "evidence_path": evidence_path,
                "evidence_sha256": event["evidence_sha256"],
                "source_attempt": TARGET_ATTEMPT,
                "verification_lineage": (
                    "direct_postconfirmation_integrity_checkpoint"
                ),
            }
            if event.get("checkpoint_name") != checkpoint["name"]:
                raise VersionForwardTransactionError(
                    "replayed integrity-failure checkpoint-name drift"
                )
            expected["completed_states"] = completed_before + [completed_state]
            expected["verified_checkpoints"] = checkpoints_before + [checkpoint]
            expected["last_verified_checkpoint"] = checkpoint
            expected["current_state"] = "INDEPENDENT_VERIFICATION"
            expected["postconfirmation_integrity_failure"] = {
                "status": "awaiting_independent_verification",
                "source": spec["source"],
                "trigger_state": completed_state,
                "source_path": evidence_path,
                "source_sha256": event["evidence_sha256"],
                "audit_hash_keys": list(spec["audit_hash_keys"]),
                "skipped_states": skipped,
                "staged_unix_ns": created,
            }
            expected["skipped_states"] = [
                {
                    "state": state_name,
                    "source": spec["source"],
                    "reason": "postconfirmation_integrity_failure_short_circuit",
                    "source_path": evidence_path,
                    "source_sha256": event["evidence_sha256"],
                    "recorded_unix_ns": created,
                }
                for state_name in skipped
            ]
            expected["next_action"] = (
                "run the read-only confirmation-mode independent verifier and "
                "capture the execution-invalid audit; do not retry or alter "
                "confirmation evidence"
            )
        expected["updated_unix_ns"] = created
        if _canonical_bytes(proposal) != _canonical_bytes(expected):
            raise VersionForwardTransactionError(
                "replayed state-completed unrelated STATE drift"
            )
        return
    if sequence == ["preregistered_early_scientific_failure_staged"]:
        event = events[0]
        keys = {
                "event", "attempt", "mode", "trigger_state",
                "trigger_evidence_path", "trigger_evidence_sha256",
                "verifier_contract_path", "verifier_contract_sha256",
                "skipped_states", "smoke_outcome_episodes",
                "confirmation_outcome_episodes_generated",
                "confirmation_outcome_episodes_executed",
                "confirmation_outcomes_opened_for_analysis",
                "created_unix_ns",
        }
        mode = event.get("mode")
        spec = EARLY_FAILURE_TRANSITIONS.get(mode)
        trigger = spec.get("trigger_state") if spec is not None else None
        skipped = (
            list(
                CONTROLLER_STATE_MACHINE[
                    CONTROLLER_STATE_MACHINE.index(str(trigger)) + 1:
                    CONTROLLER_STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
                ]
            )
            if trigger in CONTROLLER_STATE_MACHINE
            else None
        )
        source_path = (
            _relative(ATTEMPT_ROOT / str(spec["source_relative"]))
            if spec is not None
            else None
        )
        contract_path = (
            _relative(ATTEMPT_ROOT / str(spec["contract_relative"]))
            if spec is not None
            else None
        )
        if not (
            set(event) == keys
            and spec is not None
            and event.get("trigger_state") == trigger == base.get("current_state")
            and event.get("trigger_evidence_path") == source_path
            and event.get("verifier_contract_path") == contract_path
            and _is_sha256(event.get("trigger_evidence_sha256"))
            and _is_sha256(event.get("verifier_contract_sha256"))
            and event.get("skipped_states") == skipped
            and type(event.get("smoke_outcome_episodes")) is int
            and event.get("smoke_outcome_episodes") == 0
            and type(event.get("confirmation_outcome_episodes_generated")) is int
            and event.get("confirmation_outcome_episodes_generated") == 0
            and type(event.get("confirmation_outcome_episodes_executed")) is int
            and event.get("confirmation_outcome_episodes_executed") == 0
            and event.get("confirmation_outcomes_opened_for_analysis") is False
            and base.get("early_scientific_failure") is None
            and base.get("terminal_label") is None
            and type(base.get("smoke_outcome_episodes")) is int
            and base.get("smoke_outcome_episodes") == 0
            and type(base.get("confirmation_outcome_episodes_generated")) is int
            and base.get("confirmation_outcome_episodes_generated") == 0
            and type(base.get("confirmation_outcome_episodes_executed")) is int
            and base.get("confirmation_outcome_episodes_executed") == 0
            and base.get("confirmation_outcomes_opened_for_analysis") is False
        ):
            raise VersionForwardTransactionError(
                "replayed early-failure controller transition drift"
            )
        for path_value, digest_value in (
            (source_path, event["trigger_evidence_sha256"]),
            (contract_path, event["verifier_contract_sha256"]),
        ):
            resolved = (REPOSITORY_ROOT / str(path_value)).resolve(strict=True)
            if (
                not resolved.is_relative_to(ATTEMPT_ROOT.resolve())
                or _snapshot_file(resolved)[0]["sha256"] != digest_value
            ):
                raise VersionForwardTransactionError(
                    "replayed early-failure evidence drift"
                )
        source_file, source_object = active_json_object(
            source_path, "early-failure source"
        )
        contract_file, contract_object = active_json_object(
            contract_path, "early-failure verifier contract"
        )
        contract_path_key = (
            "selection_ledger" if mode == "no_candidate" else "power_freeze"
        )
        contract_paths = contract_object.get("paths")
        if not (
            _snapshot_file(source_file)[0]["sha256"]
            == event["trigger_evidence_sha256"]
            and _snapshot_file(contract_file)[0]["sha256"]
            == event["verifier_contract_sha256"]
            and type(contract_object.get("schema_version")) is int
            and contract_object.get("schema_version") == 1
            and contract_object.get("attempt") == TARGET_ATTEMPT
            and contract_object.get("attempt_root") == _relative(ATTEMPT_ROOT)
            and contract_object.get("mode") == mode
            and isinstance(contract_paths, Mapping)
            and contract_paths.get(contract_path_key) == source_path
        ):
            raise VersionForwardTransactionError(
                "replayed early-failure contract binding drift"
            )
        if mode == "no_candidate":
            source_semantics = (
                source_object.get("status")
                == "selection_complete_no_selected_head_refit"
                and type(source_object.get("candidate_count")) is int
                and source_object.get("candidate_count") == 24
                and type(source_object.get("eligible_count")) is int
                and source_object.get("eligible_count") == 0
                and source_object.get("selected_candidate_id") is None
                and source_object.get("selected_candidate_index") is None
                and source_object.get("selected_head_refit_after_selection") is False
                and type(
                    source_object.get(
                        "prior_confirmation_outcome_episodes_used", 0
                    )
                ) is int
                and source_object.get(
                    "prior_confirmation_outcome_episodes_used", 0
                ) == 0
            )
        else:
            source_semantics = (
                source_object.get("status")
                == "binding_post_selection_power_result"
                and source_object.get("decision")
                == "power_infeasible_no_confirmation"
                and source_object.get("feasible") is False
                and source_object.get("passed") is False
                and source_object.get(
                    "confirmation_generation_authorized_by_power"
                ) is False
                and source_object.get(
                    "selected_confirmation_episodes_per_regime"
                ) is None
                and source_object.get("confirmation_episode_count_per_regime")
                is None
                and source_object.get("terminal_label_if_infeasible")
                == "domain_robust_gate_failed"
                and type(
                    source_object.get("fresh_confirmation_outcomes_opened")
                ) is int
                and source_object.get("fresh_confirmation_outcomes_opened") == 0
            )
        if not source_semantics:
            raise VersionForwardTransactionError(
                "replayed early-failure source eligibility drift"
            )
        if not (
            type(base.get("expected_fit_episode_count")) is int
            and type(base.get("fit_outcome_episodes")) is int
            and base.get("fit_outcome_episodes")
            == base.get("expected_fit_episode_count")
            and type(base.get("expected_selection_episode_count")) is int
            and type(base.get("selection_outcome_episodes")) is int
            and base.get("selection_outcome_episodes")
            == base.get("expected_selection_episode_count")
        ):
            raise VersionForwardTransactionError(
                "replayed early-failure role completeness drift"
            )
        forbidden_roots = (
            "data/smoke", "data/confirmation", "execution/smoke",
            "execution/confirmation", "metrics/smoke", "metrics/confirmation",
        )
        forbidden_files = (
            "audit/pre_confirmation_manifest.json",
            "audit/pre_confirmation_package_seal.json",
            "audit/excluded_mechanical_smoke.json",
            "audit/confirmation_generation.json",
            "audit/confirmation_execution.json",
            "audit/confirmation_input_seal.json", "analysis_result.json",
            "bootstrap_replicates.npz", "bootstrap_summary.json",
        )
        if not (
            all(
                not root.exists()
                or not any(path.is_file() for path in root.rglob("*"))
                for root in (ATTEMPT_ROOT / relative for relative in forbidden_roots)
            )
            and all(not (ATTEMPT_ROOT / relative).exists() for relative in forbidden_files)
        ):
            raise VersionForwardTransactionError(
                "replayed early-failure later-role artifact drift"
            )
        created = int(event["created_unix_ns"])
        expected = json.loads(json.dumps(base))
        expected["early_scientific_failure"] = {
            "mode": mode,
            "status": "awaiting_independent_verification",
            "trigger_state": trigger,
            "trigger_evidence_path": source_path,
            "trigger_evidence_sha256": event["trigger_evidence_sha256"],
            "verifier_contract_path": contract_path,
            "verifier_contract_sha256": event["verifier_contract_sha256"],
            "skipped_states": skipped,
            "staged_unix_ns": created,
            "required_terminal_label": "domain_robust_gate_failed",
            "required_process_valid": True,
        }
        expected["skipped_states"] = [
            {
                "state": state_name,
                "mode": mode,
                "reason": "preregistered_process_valid_early_scientific_failure",
                "trigger_evidence_path": source_path,
                "trigger_evidence_sha256": event["trigger_evidence_sha256"],
                "recorded_unix_ns": created,
            }
            for state_name in skipped
        ]
        expected["current_state"] = "INDEPENDENT_VERIFICATION"
        expected["next_action"] = (
            "run the standalone read-only verifier in the exact staged mode and "
            "capture audit/independent_verification.json before finalizing"
        )
        expected["updated_unix_ns"] = created
        if _canonical_bytes(proposal) != _canonical_bytes(expected):
            raise VersionForwardTransactionError(
                "replayed early-failure unrelated STATE drift"
            )
        return
    if sequence == [
        "state_completed", "scientific_terminal_decision_recorded",
        "state_completed",
    ]:
        first, decision, last = events
        ordinary = {
            "event", "attempt", "completed_state", "next_state",
            "checkpoint_name", "evidence_path", "evidence_sha256",
            "created_unix_ns",
        }
        terminal = {
            "event", "attempt", "terminal_label", "process_valid",
            "decision_path", "decision_sha256", "created_unix_ns",
        }
        early = decision.get("terminal_label") == "domain_robust_gate_failed"
        expected_first = ordinary | (
            {"early_failure_mode"}
            if early
            else {"postconfirmation_integrity_failure"}
        )
        expected_decision = terminal | (
            {"confirmation_terminal", "early_failure_mode"} if early else set()
        )
        expected_last = ordinary | (
            {"early_failure_mode", "skipped_states"}
            if early
            else {"postconfirmation_integrity_failure"}
        )
        created = first.get("created_unix_ns")
        if not (
            set(first) == expected_first
            and set(decision) == expected_decision
            and set(last) == expected_last
            and created == decision.get("created_unix_ns")
            == last.get("created_unix_ns")
            and first.get("completed_state") == "INDEPENDENT_VERIFICATION"
            and first.get("next_state") == "TERMINAL"
            and last.get("completed_state") == "TERMINAL"
            and last.get("next_state") == "POST_TERMINAL_REPORTING"
            and base.get("current_state") == "INDEPENDENT_VERIFICATION"
            and base.get("terminal_label") is None
            and _is_sha256(first.get("evidence_sha256"))
            and _is_sha256(last.get("evidence_sha256"))
            and decision.get("decision_path") == last.get("evidence_path")
            and decision.get("decision_sha256") == last.get("evidence_sha256")
            and (
                (
                    early
                    and decision.get("process_valid") is True
                    and decision.get("confirmation_terminal") is False
                    and decision.get("early_failure_mode")
                    == first.get("early_failure_mode")
                    == last.get("early_failure_mode")
                )
                or (
                    not early
                    and decision.get("terminal_label")
                    == "domain_robust_gate_execution_invalid"
                    and decision.get("process_valid") is False
                    and first.get("postconfirmation_integrity_failure") is True
                    and last.get("postconfirmation_integrity_failure") is True
                )
            )
        ):
            raise VersionForwardTransactionError(
                "replayed three-event terminal controller transition drift"
            )
        for item in (first, last):
            path = (REPOSITORY_ROOT / str(item["evidence_path"])).resolve(
                strict=True
            )
            if (
                not path.is_relative_to(ATTEMPT_ROOT.resolve())
                or _snapshot_file(path)[0]["sha256"]
                != item["evidence_sha256"]
            ):
                raise VersionForwardTransactionError(
                    "replayed three-event terminal evidence drift"
                )
        exact_audit_path = _relative(
            ATTEMPT_ROOT / "audit/independent_verification.json"
        )
        exact_decision_path = _relative(ATTEMPT_ROOT / "decision.json")
        if (
            first.get("evidence_path") != exact_audit_path
            or last.get("evidence_path") != exact_decision_path
        ):
            raise VersionForwardTransactionError(
                "replayed three-event terminal evidence path drift"
            )
        audit_file, audit_object = active_json_object(
            first["evidence_path"], "three-event independent audit"
        )
        decision_file, decision_object = active_json_object(
            last["evidence_path"], "three-event terminal decision"
        )
        if not (
            _snapshot_file(audit_file)[0]["sha256"]
            == first["evidence_sha256"]
            and _snapshot_file(decision_file)[0]["sha256"]
            == last["evidence_sha256"]
            and audit_object.get("attempt") == TARGET_ATTEMPT
            and decision_object.get("attempt") == TARGET_ATTEMPT
            and decision_object.get("terminal_label")
            == decision["terminal_label"]
            and type(decision_object.get("process_valid")) is bool
            and decision_object.get("process_valid")
            is decision["process_valid"]
        ):
            raise VersionForwardTransactionError(
                "replayed three-event terminal JSON binding drift"
            )
        checkpoints_before = base.get("verified_checkpoints")
        completed_before = base.get("completed_states")
        if not isinstance(checkpoints_before, list) or not isinstance(
            completed_before, list
        ):
            raise VersionForwardTransactionError(
                "replayed three-event terminal history drift"
            )
        if early:
            mode = decision["early_failure_mode"]
            staged = base.get("early_scientific_failure")
            contract_file, contract_object = active_json_object(
                staged.get("verifier_contract_path")
                if isinstance(staged, Mapping)
                else None,
                "early-terminal verifier contract",
            )
            contract_record = audit_object.get("verifier_contract")
            source_hashes = audit_object.get("source_hashes")
            source_key = (
                "selection_ledger" if mode == "no_candidate" else "power_freeze"
            )
            if not (
                mode in EARLY_FAILURE_TRANSITIONS
                and isinstance(staged, Mapping)
                and staged.get("mode") == mode
                and staged.get("status") == "awaiting_independent_verification"
                and base.get("postconfirmation_integrity_failure") is None
                and type(audit_object.get("schema_version")) is int
                and audit_object.get("schema_version") == 1
                and audit_object.get("mode") == mode
                and audit_object.get("passed") is True
                and audit_object.get("terminal_label")
                == "domain_robust_gate_failed"
                and audit_object.get("read_only_verifier") is True
                and isinstance(audit_object.get("checks"), Mapping)
                and bool(audit_object["checks"])
                and isinstance(contract_record, Mapping)
                and contract_record.get("path")
                == staged.get("verifier_contract_path")
                and contract_record.get("sha256")
                == _snapshot_file(contract_file)[0]["sha256"]
                == staged.get("verifier_contract_sha256")
                and type(contract_object.get("schema_version")) is int
                and contract_object.get("schema_version") == 1
                and contract_object.get("attempt") == TARGET_ATTEMPT
                and contract_object.get("attempt_root") == _relative(ATTEMPT_ROOT)
                and contract_object.get("mode") == mode
                and isinstance(source_hashes, Mapping)
                and linked_sha256(source_hashes.get(source_key))
                == staged.get("trigger_evidence_sha256")
                and type(decision_object.get("schema_version")) is int
                and decision_object.get("schema_version") == 1
                and decision_object.get("passed") is True
                and decision_object.get("checkpoint_state") == "TERMINAL"
                and decision_object.get("early_failure_mode") == mode
                and decision_object.get("trigger_evidence_path")
                == staged.get("trigger_evidence_path")
                and decision_object.get("trigger_evidence_sha256")
                == staged.get("trigger_evidence_sha256")
                and decision_object.get("independent_verification_path")
                == first.get("evidence_path")
                and decision_object.get("independent_verification_sha256")
                == first.get("evidence_sha256")
                and type(decision_object.get("smoke_outcome_episodes")) is int
                and decision_object.get("smoke_outcome_episodes") == 0
                and type(
                    decision_object.get(
                        "confirmation_outcome_episodes_generated"
                    )
                ) is int
                and decision_object.get(
                    "confirmation_outcome_episodes_generated"
                ) == 0
                and type(
                    decision_object.get(
                        "confirmation_outcome_episodes_executed"
                    )
                ) is int
                and decision_object.get(
                    "confirmation_outcome_episodes_executed"
                ) == 0
                and decision_object.get(
                    "confirmation_outcomes_opened_for_analysis"
                ) is False
                and last.get("skipped_states") == staged.get("skipped_states")
                and first.get("checkpoint_name")
                == f"{TARGET_ATTEMPT}_{mode}_independent_verification_passed"
                and last.get("checkpoint_name")
                == f"{TARGET_ATTEMPT}_{mode}_terminal_decision_recorded"
            ):
                raise VersionForwardTransactionError(
                    "replayed early terminal-state authorization drift"
                )
            forbidden_roots = (
                "data/smoke", "data/confirmation", "execution/smoke",
                "execution/confirmation", "metrics/smoke", "metrics/confirmation",
            )
            forbidden_files = (
                "audit/pre_confirmation_manifest.json",
                "audit/pre_confirmation_package_seal.json",
                "audit/excluded_mechanical_smoke.json",
                "audit/confirmation_generation.json",
                "audit/confirmation_execution.json",
                "audit/confirmation_input_seal.json", "analysis_result.json",
                "bootstrap_replicates.npz", "bootstrap_summary.json",
            )
            if not (
                type(base.get("expected_fit_episode_count")) is int
                and type(base.get("fit_outcome_episodes")) is int
                and base.get("fit_outcome_episodes")
                == base.get("expected_fit_episode_count")
                and type(base.get("expected_selection_episode_count")) is int
                and type(base.get("selection_outcome_episodes")) is int
                and base.get("selection_outcome_episodes")
                == base.get("expected_selection_episode_count")
                and type(base.get("smoke_outcome_episodes")) is int
                and base.get("smoke_outcome_episodes") == 0
                and type(base.get("confirmation_outcome_episodes_generated")) is int
                and base.get("confirmation_outcome_episodes_generated") == 0
                and type(base.get("confirmation_outcome_episodes_executed")) is int
                and base.get("confirmation_outcome_episodes_executed") == 0
                and base.get("confirmation_outcomes_opened_for_analysis") is False
                and all(
                    not root.exists()
                    or not any(path.is_file() for path in root.rglob("*"))
                    for root in (
                        ATTEMPT_ROOT / relative for relative in forbidden_roots
                    )
                )
                and all(
                    not (ATTEMPT_ROOT / relative).exists()
                    for relative in forbidden_files
                )
            ):
                raise VersionForwardTransactionError(
                    "replayed early terminal later-role isolation drift"
                )
            lineage = "direct_early_scientific_failure_checkpoint"
        else:
            expected_confirmation = base.get(
                "expected_confirmation_episode_count"
            )
            contract_file, contract_object = active_json_object(
                _relative(ATTEMPT_ROOT / "verifier_contract.json"),
                "execution-invalid verifier contract",
            )
            contract_record = audit_object.get("verifier_contract")
            capture_record = audit_object.get("capture")
            source_hashes = audit_object.get("source_hashes")
            staged_integrity = base.get("postconfirmation_integrity_failure")
            if not (
                base.get("early_scientific_failure") is None
                and type(expected_confirmation) is int
                and expected_confirmation > 0
                and type(base.get("expected_smoke_episode_count")) is int
                and type(base.get("smoke_outcome_episodes")) is int
                and base.get("smoke_outcome_episodes")
                == base.get("expected_smoke_episode_count")
                and type(base.get("confirmation_outcome_episodes_generated")) is int
                and base.get("confirmation_outcome_episodes_generated")
                == expected_confirmation
                and type(base.get("confirmation_outcome_episodes_executed")) is int
                and base.get("confirmation_outcome_episodes_executed")
                == expected_confirmation
                and base.get("confirmation_outcomes_opened_for_analysis") is True
                and type(audit_object.get("schema_version")) is int
                and audit_object.get("schema_version") == 1
                and audit_object.get("mode") == "confirmation"
                and audit_object.get("read_only_verifier") is True
                and audit_object.get("execution_invalid") is True
                and audit_object.get("terminal_label")
                == "domain_robust_gate_execution_invalid"
                and proves_integrity_failure(audit_object)
                and isinstance(contract_record, Mapping)
                and contract_record.get("path")
                == _relative(ATTEMPT_ROOT / "verifier_contract.json")
                and contract_record.get("sha256")
                == _snapshot_file(contract_file)[0]["sha256"]
                and type(contract_object.get("schema_version")) is int
                and contract_object.get("schema_version") == 1
                and contract_object.get("attempt") == TARGET_ATTEMPT
                and contract_object.get("attempt_root") == _relative(ATTEMPT_ROOT)
                and contract_object.get("mode") == "confirmation"
                and isinstance(capture_record, Mapping)
                and capture_record.get("captured_exclusively") is True
                and capture_record.get("wrapper_integrity_passed") is True
                and type(decision_object.get("schema_version")) is int
                and decision_object.get("schema_version") == 1
                and decision_object.get("passed") is True
                and decision_object.get("checkpoint_state") == "TERMINAL"
                and decision_object.get("terminal_basis")
                == "postconfirmation_integrity_failure"
                and decision_object.get("independent_verification_path")
                == first.get("evidence_path")
                and decision_object.get("independent_verification_sha256")
                == first.get("evidence_sha256")
                and type(
                    decision_object.get(
                        "confirmation_outcome_episodes_generated"
                    )
                ) is int
                and decision_object.get(
                    "confirmation_outcome_episodes_generated"
                ) == expected_confirmation
                and type(
                    decision_object.get(
                        "confirmation_outcome_episodes_executed"
                    )
                ) is int
                and decision_object.get(
                    "confirmation_outcome_episodes_executed"
                ) == expected_confirmation
                and decision_object.get(
                    "confirmation_outcomes_opened_for_analysis"
                ) is True
                and
                first.get("checkpoint_name")
                == f"{TARGET_ATTEMPT}_execution_invalid_independent_audit"
                and last.get("checkpoint_name")
                == f"{TARGET_ATTEMPT}_execution_invalid_terminal_decision"
            ):
                raise VersionForwardTransactionError(
                    "replayed execution-invalid base/evidence drift"
                )
            if isinstance(staged_integrity, Mapping):
                candidates = staged_integrity.get("audit_hash_keys")
                if not (
                    audit_object.get("passed") is True
                    and isinstance(source_hashes, Mapping)
                    and isinstance(candidates, list)
                    and bool(candidates)
                    and any(
                        linked_sha256(source_hashes.get(name))
                        == staged_integrity.get("source_sha256")
                        for name in candidates
                    )
                ):
                    raise VersionForwardTransactionError(
                        "replayed execution-invalid staged-source drift"
                    )
            elif not (
                staged_integrity is None
                and audit_object.get("passed") is False
                and capture_record.get("scientific_verifier_passed") is False
                and type(capture_record.get("verifier_returncode")) is int
                and capture_record.get("verifier_returncode") != 0
                and isinstance(audit_object.get("error_type"), str)
                and bool(audit_object["error_type"])
                and isinstance(audit_object.get("error"), str)
                and bool(audit_object["error"])
            ):
                raise VersionForwardTransactionError(
                    "replayed execution-invalid verifier-failure drift"
                )
            lineage = "direct_postconfirmation_integrity_checkpoint"
        audit_checkpoint = {
            "name": first["checkpoint_name"],
            "created_unix_ns": created,
            "evidence_path": first["evidence_path"],
            "evidence_sha256": first["evidence_sha256"],
            "source_attempt": TARGET_ATTEMPT,
            "verification_lineage": lineage,
        }
        decision_checkpoint = {
            "name": last["checkpoint_name"],
            "created_unix_ns": created,
            "evidence_path": last["evidence_path"],
            "evidence_sha256": last["evidence_sha256"],
            "source_attempt": TARGET_ATTEMPT,
            "verification_lineage": lineage,
        }
        expected = json.loads(json.dumps(base))
        expected["completed_states"] = completed_before + [
            "INDEPENDENT_VERIFICATION", "TERMINAL"
        ]
        expected["verified_checkpoints"] = checkpoints_before + [
            audit_checkpoint, decision_checkpoint
        ]
        expected["last_verified_checkpoint"] = decision_checkpoint
        expected["current_state"] = "POST_TERMINAL_REPORTING"
        expected["terminal_label"] = decision["terminal_label"]
        expected["process_valid"] = decision["process_valid"]
        expected["scientific_terminal"] = True
        expected["confirmation_terminal"] = not early
        expected["terminal_decision_path"] = decision["decision_path"]
        expected["terminal_decision_sha256"] = decision["decision_sha256"]
        if early:
            expected["terminal_basis"] = f"preregistered_{mode}"
            early_record = expected["early_scientific_failure"]
            early_record.update(
                {
                    "status": "terminal_recorded",
                    "independent_verification_path": first["evidence_path"],
                    "independent_verification_sha256": first["evidence_sha256"],
                    "decision_path": last["evidence_path"],
                    "decision_sha256": last["evidence_sha256"],
                    "terminal_recorded_unix_ns": created,
                }
            )
            expected["next_action"] = (
                "write the concise terminal report, robustness map, audit, "
                "limitations, and next project-scoped task; then checkpoint "
                "POST_TERMINAL_REPORTING"
            )
        else:
            expected["terminal_basis"] = "postconfirmation_integrity_failure"
            integrity_record = expected.get("postconfirmation_integrity_failure")
            if not isinstance(integrity_record, dict):
                integrity_record = {
                    "source": "independent_verifier",
                    "trigger_state": "INDEPENDENT_VERIFICATION",
                    "skipped_states": [],
                }
                expected["postconfirmation_integrity_failure"] = integrity_record
                expected["skipped_states"] = []
            integrity_record.update(
                {
                    "status": "terminal_recorded",
                    "independent_verification_path": first["evidence_path"],
                    "independent_verification_sha256": first["evidence_sha256"],
                    "decision_path": last["evidence_path"],
                    "decision_sha256": last["evidence_sha256"],
                    "terminal_recorded_unix_ns": created,
                }
            )
            expected["next_action"] = (
                "report the immutable execution-invalid result and integrity "
                "diagnosis; do not retry or reinterpret it as a scientific "
                "partial/failure"
            )
        expected["updated_unix_ns"] = created
        if _canonical_bytes(proposal) != _canonical_bytes(expected):
            raise VersionForwardTransactionError(
                "replayed three-event terminal unrelated STATE drift"
            )
        return
    raise VersionForwardTransactionError(
        "replayed adapter controller-event sequence drift"
    )


def _validate_descendant_adapter_chain(
    *,
    receipt_post_state: Mapping[str, Any],
    receipt_post_ledger: Mapping[str, Any],
    current_state: Mapping[str, Any],
    ledger_payload: bytes,
) -> None:
    """Replay every post-forward adapter transaction from its exact base STATE.

    Each transaction's first ledger record carries its complete pre-transaction
    STATE; every record carries an exact transaction id/index/count envelope.
    The next transaction's base STATE (or the live STATE for the final group)
    supplies the prior transaction's target, allowing its suffix, operation
    digest, resulting marker, state binding, and end count/head to be rederived
    without trusting the mutable current marker as its own authorization.
    """

    post_count = receipt_post_ledger.get("event_count")
    post_bytes = receipt_post_ledger.get("bytes")
    if (
        type(post_count) is not int
        or post_count <= 0
        or type(post_bytes) is not int
        or post_bytes <= 0
        or not isinstance(receipt_post_state, Mapping)
        or not isinstance(current_state, Mapping)
    ):
        raise VersionForwardTransactionError(
            "descendant adapter-chain receipt boundary drift"
        )
    lines = ledger_payload.splitlines(keepends=True)
    active_edges: list[int] = []
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            continue
        event = _json_payload(
            line[:-1], f"descendant adapter ledger record {index + 1}"
        )
        if (
            event.get("event") == "zero_confirmation_outcome_version_forward"
            and event.get("attempt") == TARGET_ATTEMPT
        ):
            active_edges.append(index)
    post_head = _json_payload(
        lines[post_count - 1][:-1], "receipt-bound active forward record"
    ) if post_count <= len(lines) and lines[post_count - 1].endswith(b"\n") else {}
    if (
        type(current_state.get("ledger_event_count")) is not int
        or len(lines) != current_state.get("ledger_event_count")
        or post_count > len(lines)
        or any(not line.endswith(b"\n") for line in lines)
        or len(ledger_payload[:post_bytes]) != post_bytes
        or hashlib.sha256(ledger_payload[:post_bytes]).hexdigest()
        != receipt_post_ledger.get("sha256")
        or b"".join(lines[:post_count]) != ledger_payload[:post_bytes]
        or post_bytes != len(b"".join(lines[:post_count]))
        or len(active_edges) != 1
        or post_count != active_edges[0] + 1
        or receipt_post_ledger.get("head_sha256")
        != post_head.get("record_sha256")
        or receipt_post_state.get("ledger_event_count") != post_count
        or receipt_post_state.get("ledger_head_sha256")
        != receipt_post_ledger.get("head_sha256")
    ):
        raise VersionForwardTransactionError(
            "descendant adapter-chain ledger boundary drift"
        )
    expected_base = json.loads(json.dumps(dict(receipt_post_state)))
    if post_count == len(lines):
        if _canonical_bytes(current_state) != _canonical_bytes(expected_base):
            raise VersionForwardTransactionError(
                "receipt post-state differs from current initial adapter state"
            )
        return

    root_record = _source_record(PROGRAM_PATH)
    adapter_record = _source_record(TRANSACTION_SOURCE_PATH)
    cursor = post_count
    while cursor < len(lines):
        first = _json_payload(
            lines[cursor][:-1],
            f"descendant adapter ledger record {cursor + 1}",
        )
        first_envelope = first.get(ADAPTER_TRANSACTION_ENVELOPE_FIELD)
        first_keys = {
            "schema_version",
            "authorization_kind",
            "transaction_id",
            "event_index",
            "event_count",
            "base_state_sha256",
            "proposed_state_sha256",
            "base_state",
        }
        if (
            lines[cursor] != _canonical_bytes(first) + b"\n"
            or not isinstance(first_envelope, Mapping)
            or set(first_envelope) != first_keys
            or type(first_envelope.get("schema_version")) is not int
            or first_envelope.get("schema_version") != 1
            or first_envelope.get("authorization_kind")
            != ADAPTER_TRANSACTION_ENVELOPE_KIND
            or not _is_sha256(first_envelope.get("transaction_id"))
            or type(first_envelope.get("event_index")) is not int
            or first_envelope.get("event_index") != 0
            or type(first_envelope.get("event_count")) is not int
            or first_envelope.get("event_count") <= 0
            or not _is_sha256(first_envelope.get("base_state_sha256"))
            or not _is_sha256(
                first_envelope.get("proposed_state_sha256")
            )
            or not isinstance(first_envelope.get("base_state"), Mapping)
        ):
            raise VersionForwardTransactionError(
                "descendant adapter transaction first-envelope drift"
            )
        event_count = int(first_envelope["event_count"])
        end = cursor + event_count
        if end > len(lines):
            raise VersionForwardTransactionError(
                "descendant adapter transaction exceeds ledger boundary"
            )
        base_state = dict(first_envelope["base_state"])
        _validate_state_ledger_types(
            base_state, label="descendant adapter base STATE"
        )
        _validate_outcome_counter_types(base_state)
        if (
            _canonical_bytes(base_state) != _canonical_bytes(expected_base)
            or first_envelope.get("base_state_sha256")
            != _canonical_sha256(base_state)
            or base_state.get("active_attempt") != TARGET_ATTEMPT
            or base_state.get("ledger_event_count") != cursor
            or base_state.get("ledger_head_sha256")
            != _json_payload(
                lines[cursor - 1][:-1],
                f"descendant adapter predecessor record {cursor}",
            ).get("record_sha256")
        ):
            raise VersionForwardTransactionError(
                "descendant adapter transaction base-state drift"
            )

        stripped_events: list[dict[str, Any]] = []
        transaction_id = str(first_envelope["transaction_id"])
        base_sha256 = str(first_envelope["base_state_sha256"])
        last_record: dict[str, Any] | None = None
        for offset in range(event_count):
            absolute = cursor + offset
            record = _json_payload(
                lines[absolute][:-1],
                f"descendant adapter ledger record {absolute + 1}",
            )
            if lines[absolute] != _canonical_bytes(record) + b"\n":
                raise VersionForwardTransactionError(
                    "descendant adapter ledger record is noncanonical"
                )
            envelope = record.get(ADAPTER_TRANSACTION_ENVELOPE_FIELD)
            expected_keys = {
                "schema_version",
                "authorization_kind",
                "transaction_id",
                "event_index",
                "event_count",
                "base_state_sha256",
                "proposed_state_sha256",
            }
            if offset == 0:
                expected_keys.add("base_state")
            if (
                not isinstance(envelope, Mapping)
                or set(envelope) != expected_keys
                or type(envelope.get("schema_version")) is not int
                or envelope.get("schema_version") != 1
                or envelope.get("authorization_kind")
                != ADAPTER_TRANSACTION_ENVELOPE_KIND
                or envelope.get("transaction_id") != transaction_id
                or type(envelope.get("event_index")) is not int
                or envelope.get("event_index") != offset
                or type(envelope.get("event_count")) is not int
                or envelope.get("event_count") != event_count
                or envelope.get("base_state_sha256") != base_sha256
                or envelope.get("proposed_state_sha256")
                != first_envelope.get("proposed_state_sha256")
                or (
                    offset == 0
                    and _canonical_bytes(envelope.get("base_state"))
                    != _canonical_bytes(base_state)
                )
                or type(record.get("seq")) is not int
                or record.get("seq") != absolute + 1
                or not isinstance(record.get("prev_sha256"), str)
                or not _is_sha256(record.get("record_sha256"))
            ):
                raise VersionForwardTransactionError(
                    "descendant adapter transaction envelope/record drift"
                )
            record_without_digest = dict(record)
            observed_digest = record_without_digest.pop("record_sha256")
            if observed_digest != _canonical_sha256(record_without_digest):
                raise VersionForwardTransactionError(
                    "descendant adapter ledger record hash drift"
                )
            stripped = dict(record)
            stripped.pop(ADAPTER_TRANSACTION_ENVELOPE_FIELD)
            stripped.pop("seq")
            stripped.pop("prev_sha256")
            stripped.pop("record_sha256")
            if (
                type(stripped.get("created_unix_ns")) is not int
                or stripped.get("created_unix_ns") <= 0
                or stripped.get("attempt") != TARGET_ATTEMPT
            ):
                raise VersionForwardTransactionError(
                    "descendant adapter controller-event type drift"
                )
            stripped_events.append(stripped)
            last_record = record
        event_sequence = [event.get("event") for event in stripped_events]
        permitted_singletons = {
            "state_completed",
            "outcome_counts_updated",
            "preregistered_early_scientific_failure_staged",
            "scientific_terminal_decision_recorded",
        }
        if not (
            (len(event_sequence) == 1 and event_sequence[0] in permitted_singletons)
            or event_sequence
            == [
                "state_completed",
                "scientific_terminal_decision_recorded",
                "state_completed",
            ]
        ):
            raise VersionForwardTransactionError(
                "descendant adapter controller-event sequence drift"
            )
        count_events = [
            event
            for event in stripped_events
            if event.get("event") == "outcome_counts_updated"
        ]
        if count_events:
            if (
                len(stripped_events) != 1
                or _canonical_bytes(
                    count_events[0].get(PRIOR_ADAPTER_MARKER_FIELD)
                )
                != _canonical_bytes(base_state.get(ADAPTER_STATE_KEY))
            ):
                raise VersionForwardTransactionError(
                    "descendant count event prior-marker drift"
                )
        elif any(
            PRIOR_ADAPTER_MARKER_FIELD in event for event in stripped_events
        ):
            raise VersionForwardTransactionError(
                "non-count descendant event carries a prior marker"
            )
        if transaction_id != _adapter_transaction_id(
            base_state,
            str(first_envelope["proposed_state_sha256"]),
            stripped_events,
        ):
            raise VersionForwardTransactionError(
                "descendant adapter transaction id drift"
            )
        assert last_record is not None

        if end < len(lines):
            next_record = _json_payload(
                lines[end][:-1],
                f"descendant adapter ledger record {end + 1}",
            )
            next_envelope = next_record.get(
                ADAPTER_TRANSACTION_ENVELOPE_FIELD
            )
            if (
                not isinstance(next_envelope, Mapping)
                or next_envelope.get("event_index") != 0
                or not isinstance(next_envelope.get("base_state"), Mapping)
            ):
                raise VersionForwardTransactionError(
                    "descendant adapter transaction grouping gap"
                )
            result_state = dict(next_envelope["base_state"])
        else:
            result_state = dict(current_state)
        _validate_state_ledger_types(
            result_state, label="descendant adapter target STATE"
        )
        _validate_outcome_counter_types(result_state)
        last_head = str(last_record["record_sha256"])
        if (
            result_state.get("active_attempt") != TARGET_ATTEMPT
            or result_state.get("ledger_event_count") != end
            or result_state.get("ledger_head_sha256") != last_head
        ):
            raise VersionForwardTransactionError(
                "descendant adapter transaction target-state drift"
            )
        state_without_marker = json.loads(json.dumps(result_state))
        observed_marker = state_without_marker.pop(ADAPTER_STATE_KEY, None)
        proposed_state = json.loads(json.dumps(state_without_marker))
        proposed_state["ledger_event_count"] = base_state[
            "ledger_event_count"
        ]
        proposed_state["ledger_head_sha256"] = base_state[
            "ledger_head_sha256"
        ]
        if _canonical_sha256(proposed_state) != first_envelope.get(
            "proposed_state_sha256"
        ):
            raise VersionForwardTransactionError(
                "descendant adapter proposed-state commitment drift"
            )
        _validate_replayed_controller_transition(
            base_state=base_state,
            proposed_state=proposed_state,
            events=stripped_events,
        )
        group_suffix = b"".join(lines[cursor:end])
        operation_sha256 = _operation_sha256(
            base_state=base_state,
            base_ledger_sha256=hashlib.sha256(
                b"".join(lines[:cursor])
            ).hexdigest(),
            expected_suffix_sha256=hashlib.sha256(group_suffix).hexdigest(),
            intended_without_marker=state_without_marker,
        )
        expected_marker = _adapter_marker(
            event_count=end,
            head_sha256=last_head,
            operation_sha256=operation_sha256,
            state_without_marker=state_without_marker,
            root_record=root_record,
            adapter_record=adapter_record,
        )
        if _canonical_bytes(observed_marker) != _canonical_bytes(
            expected_marker
        ):
            raise VersionForwardTransactionError(
                "descendant adapter transaction marker derivation drift"
            )
        expected_base = result_state
        cursor = end


def _ledger_progress_for_pending(
    pending: Mapping[str, Any], ledger_payload: bytes
) -> tuple[bytes, str]:
    base_size = int(pending["base_ledger_bytes"])
    if len(ledger_payload) < base_size:
        raise VersionForwardTransactionError("ledger is shorter than pending base")
    base = ledger_payload[:base_size]
    if hashlib.sha256(base).hexdigest() != pending["base_ledger_sha256"]:
        raise VersionForwardTransactionError("pending ledger base prefix drift")
    genesis_payload = _snapshot_file(GENESIS_PATH)[1]
    base_chain = _ledger_summary(base, genesis_payload)
    if (
        base_chain["event_count"] != pending["base_event_count"]
        or base_chain["head_sha256"] != pending["base_head_sha256"]
    ):
        raise VersionForwardTransactionError("pending ledger base chain drift")
    suffix = str(pending["expected_suffix"]).encode("ascii")
    remainder = ledger_payload[base_size:]
    if not suffix.startswith(remainder):
        raise VersionForwardTransactionError(
            "ledger suffix is not an exact prefix of the pending transaction"
        )
    if not remainder:
        progress = "zero"
    elif remainder == suffix:
        progress = "complete"
    else:
        progress = "partial"
    return remainder, progress


def _require_initial_forward_journal_binding(
    pending: Mapping[str, Any],
) -> None:
    """Forbid recovery of the initial forward outside its durable journal."""

    if pending.get("source_attempt") != SOURCE_ATTEMPT:
        return
    if (
        not os.path.lexists(RECEIPT_JOURNAL_PATH)
        or os.path.lexists(RECEIPT_JOURNAL_STAGING_PATH)
    ):
        raise VersionForwardTransactionError(
            "initial version-forward pending lacks its sole durable receipt journal"
        )
    journal, _payload = _canonical_object_file(
        RECEIPT_JOURNAL_PATH, "initial version-forward receipt journal"
    )
    validated = _validate_receipt_journal_locked(journal)
    if validated.get("predicted_controller_transaction") != dict(pending):
        raise VersionForwardTransactionError(
            "initial version-forward pending differs from its durable journal"
        )


def _complete_pending(pending_value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_adapter_lock()
    pending = _validate_pending(pending_value)
    _require_initial_forward_journal_binding(pending)
    base_state = dict(pending["base_state"])
    intended = dict(pending["intended_state"])
    state, state_payload = _canonical_object_file(STATE_PATH, "STATE.json")
    base_state_payload = _pretty_bytes(base_state)
    intended_payload = _pretty_bytes(intended)
    if state_payload not in (base_state_payload, intended_payload):
        raise VersionForwardTransactionError(
            "STATE.json is neither the authenticated transaction base nor target"
        )
    if os.path.lexists(STATE_STAGING_PATH):
        staged, staged_payload = _canonical_object_file(
            STATE_STAGING_PATH, "STATE staging file"
        )
        if staged != intended or staged_payload != intended_payload:
            raise VersionForwardTransactionError("STATE staging file is ambiguous")
    ledger_payload = _regular_bytes(LEDGER_PATH, "research ledger")
    remainder, progress = _ledger_progress_for_pending(pending, ledger_payload)
    suffix = str(pending["expected_suffix"]).encode("ascii")
    if progress == "partial" and remainder and not remainder.endswith(b"\n"):
        complete_bytes = remainder.rfind(b"\n") + 1
        descriptor = os.open(LEDGER_PATH, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.ftruncate(
                descriptor, int(pending["base_ledger_bytes"]) + complete_bytes
            )
            _durability_boundary("ledger_partial_truncate")
            os.fsync(descriptor)
            _durability_boundary("ledger_partial_truncate_fsync")
        finally:
            os.close(descriptor)
        remainder = remainder[:complete_bytes]
    missing = suffix[len(remainder) :]
    if missing:
        descriptor = os.open(LEDGER_PATH, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.lseek(descriptor, 0, os.SEEK_END)
            written = os.write(descriptor, missing)
            if written != len(missing):
                raise VersionForwardTransactionError(
                    "short durable research-ledger suffix write"
                )
            _durability_boundary("ledger_suffix_write")
            os.fsync(descriptor)
            _durability_boundary("ledger_suffix_fsync")
        finally:
            os.close(descriptor)
    else:
        descriptor = os.open(LEDGER_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
            _durability_boundary("ledger_complete_fsync")
        finally:
            os.close(descriptor)
    final_ledger = _regular_bytes(LEDGER_PATH, "research ledger")
    final_chain = _ledger_summary(final_ledger, _snapshot_file(GENESIS_PATH)[1])
    if (
        final_chain["event_count"] != pending["target_event_count"]
        or final_chain["head_sha256"] != pending["target_head_sha256"]
        or final_ledger[: int(pending["base_ledger_bytes"])]
        != ledger_payload[: int(pending["base_ledger_bytes"])]
    ):
        raise VersionForwardTransactionError("reconciled ledger target drift")
    if state_payload == base_state_payload:
        if os.path.lexists(STATE_STAGING_PATH):
            _fsync_exact_publication_alias(
                STATE_STAGING_PATH,
                intended_payload,
                label="controller STATE staging",
                boundary="recovery_state_inode_fsync",
            )
            os.replace(STATE_STAGING_PATH, STATE_PATH)
            _durability_boundary("recovery_state_rename")
            _fsync_directory(STATE_PATH.parent, "recovery_state_parent_fsync")
        else:
            _publish_state(intended)
    elif os.path.lexists(STATE_STAGING_PATH):
        raise VersionForwardTransactionError(
            "target STATE and a second STATE staging file coexist"
        )
    installed, _installed_payload = _canonical_object_file(STATE_PATH, "STATE.json")
    if installed != intended:
        raise VersionForwardTransactionError("durable STATE target drift")
    _validate_outcome_counter_types(installed)
    _validate_adapter_marker(installed, final_chain)
    _fsync_exact_publication_alias(
        STATE_PATH,
        intended_payload,
        label="installed controller STATE",
        boundary="recovery_state_final_inode_fsync",
    )
    _fsync_directory(
        STATE_PATH.parent, "recovery_state_publication_parent_fsync"
    )
    _unlink_pending()
    return installed, final_chain


def _recover_durable_transaction_locked() -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconcile before strict ledger decoding; never infer unauthenticated bytes."""

    _require_adapter_lock()
    pending_exists = os.path.lexists(TRANSACTION_PATH)
    pending_staging_exists = os.path.lexists(PENDING_STAGING_PATH)
    if pending_exists and pending_staging_exists:
        raise VersionForwardTransactionError(
            "pending and pending-staging transactions coexist"
        )
    if not pending_exists and pending_staging_exists:
        staged, _payload = _canonical_object_file(
            PENDING_STAGING_PATH, "pending transaction staging"
        )
        pending = _validate_pending(staged)
        _require_initial_forward_journal_binding(pending)
        state, _state_payload = _canonical_object_file(STATE_PATH, "STATE.json")
        ledger = _regular_bytes(LEDGER_PATH, "research ledger")
        if (
            state != pending["base_state"]
            or len(ledger) != pending["base_ledger_bytes"]
            or hashlib.sha256(ledger).hexdigest()
            != pending["base_ledger_sha256"]
        ):
            raise VersionForwardTransactionError(
                "unpublished pending staging file is not at its exact base"
            )
        _fsync_exact_publication_alias(
            PENDING_STAGING_PATH,
            _pretty_bytes(pending),
            label="controller pending staging",
            boundary="recovery_pending_inode_fsync",
        )
        os.replace(PENDING_STAGING_PATH, TRANSACTION_PATH)
        _durability_boundary("recovery_pending_rename")
        _fsync_directory(
            TRANSACTION_PATH.parent, "recovery_pending_parent_fsync"
        )
        pending_exists = True
    if not pending_exists:
        if os.path.lexists(STATE_STAGING_PATH):
            raise VersionForwardTransactionError(
                "STATE staging exists without an authenticated pending transaction"
            )
        state, _payload = _canonical_object_file(STATE_PATH, "STATE.json")
        _validate_state_ledger_types(state, label="recovered current STATE")
        ledger = _regular_bytes(LEDGER_PATH, "research ledger")
        chain = _ledger_summary(ledger, _snapshot_file(GENESIS_PATH)[1])
        if (
            state.get("ledger_event_count") != chain["event_count"]
            or state.get("ledger_head_sha256") != chain["head_sha256"]
        ):
            raise VersionForwardTransactionError(
                "STATE/ledger diverged without an authenticated pending transaction"
            )
        if state.get("active_attempt") == TARGET_ATTEMPT:
            _validate_outcome_counter_types(state)
            _validate_adapter_marker(state, chain)
        return state, chain
    pending, _payload = _canonical_object_file(
        TRANSACTION_PATH, "pending adapter transaction"
    )
    validated_pending = _validate_pending(pending)
    _fsync_exact_publication_alias(
        TRANSACTION_PATH,
        _pretty_bytes(validated_pending),
        label="published controller pending transaction",
        boundary="recovery_pending_final_inode_fsync",
    )
    _fsync_directory(
        TRANSACTION_PATH.parent,
        "recovery_pending_publication_parent_fsync",
    )
    return _complete_pending(validated_pending)


def _durable_commit_state_with_events(
    state: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    _require_adapter_lock()
    if any(
        os.path.lexists(path)
        for path in (TRANSACTION_PATH, PENDING_STAGING_PATH, STATE_STAGING_PATH)
    ):
        raise VersionForwardTransactionError(
            "controller mutation began with transaction residue"
        )
    pending = _validate_pending(_build_pending(state, events))
    receipt_context = getattr(_ADAPTER_LOCAL, "receipt_context_sha256", None)
    if receipt_context is not None:
        journal, _payload = _canonical_object_file(
            RECEIPT_JOURNAL_PATH, "version-forward receipt journal"
        )
        if (
            journal.get("receipt_context_sha256") != receipt_context
            or journal.get("transaction_sha256")
            != _receipt_journal_digest(journal)
            or journal.get("predicted_controller_transaction") != pending
        ):
            raise VersionForwardTransactionError(
                "live controller transaction differs from durable receipt journal"
            )
    _publish_pending(pending)
    installed, _chain = _complete_pending(pending)
    return installed


def _require_presealed_verifier_contract(*_args: Any, **_kwargs: Any) -> None:
    """Workflow replaces this narrow hook before early-terminal delegation."""

    raise VersionForwardTransactionError(
        "early-terminal controller guard was not supplied by sealed v008 workflow"
    )


def _preverify() -> dict[str, Any]:
    producer = _load_module("v008_forward_producer", ATTEMPT_ROOT / "version_forward_prepare.py")
    verifier = _load_module("v008_forward_standalone", ATTEMPT_ROOT / "verify_version_forward.py")
    independent = _load_module("v008_forward_independent", ATTEMPT_ROOT / "independent_verify.py")
    producer_result = producer.verify_existing_inheritance_seal(SEAL_PATH, STUDY_ROOT)
    standalone_result = verifier.verify(SEAL_PATH, STUDY_ROOT, phase="pre-forward")
    seal_object = _json_payload(_snapshot_file(SEAL_PATH)[1], "inheritance seal")
    policy = independent.PathPolicy(
        REPOSITORY_ROOT,
        _relative(ATTEMPT_ROOT),
        _relative(STUDY_ROOT),
    )
    partition_result = independent.verify_equivalence_partitions(seal_object, policy)
    return {
        "producer": producer_result,
        "standalone": standalone_result,
        "independent_partitions": partition_result,
    }


def _postverify() -> dict[str, Any]:
    verifier = _load_module("v008_forward_standalone_post", ATTEMPT_ROOT / "verify_version_forward.py")
    independent = _load_module("v008_forward_independent_post", ATTEMPT_ROOT / "independent_verify.py")
    journal = _json_payload(
        _snapshot_file(RECEIPT_JOURNAL_PATH)[1],
        "postverification receipt journal",
    )
    receipt_context_sha256 = journal.get("receipt_context_sha256")
    if not _is_sha256(receipt_context_sha256):
        raise VersionForwardTransactionError(
            "postverification transaction context authorization drift"
        )
    standalone_result = verifier.verify(
        SEAL_PATH,
        STUDY_ROOT,
        phase="post-forward",
        authorized_transaction_context_sha256=receipt_context_sha256,
    )
    seal_object = _json_payload(_snapshot_file(SEAL_PATH)[1], "inheritance seal")
    policy = independent.PathPolicy(
        REPOSITORY_ROOT,
        _relative(ATTEMPT_ROOT),
        _relative(STUDY_ROOT),
    )
    partition_result = independent.verify_equivalence_partitions(seal_object, policy)
    return {
        "standalone": standalone_result,
        "independent_partitions": partition_result,
    }


def _strict_chain_for_root() -> dict[str, Any]:
    payload = _regular_bytes(LEDGER_PATH, "research ledger")
    return _ledger_summary(payload, _snapshot_file(GENESIS_PATH)[1])


def _load_authenticated_program_locked(*, require_receipt: bool) -> ModuleType:
    _require_adapter_lock()
    if require_receipt:
        verify_transaction_receipt()
    program_before = _source_record(PROGRAM_PATH)
    adapter_before = _source_record(TRANSACTION_SOURCE_PATH)
    if not require_receipt:
        seal = _json_payload(_snapshot_file(SEAL_PATH)[1], "inheritance seal")
        root_record = seal.get("root_controls", {}).get("program.py")
        sealed_files = seal.get("sealed_files")
        if (
            not isinstance(root_record, Mapping)
            or root_record.get("sha256") != program_before["sha256"]
            or not isinstance(sealed_files, Mapping)
            or sealed_files.get(adapter_before["path"]) != adapter_before["sha256"]
        ):
            raise VersionForwardTransactionError(
                "initial controller adapter/root sources are not inheritance-seal bound"
            )
    program = _load_module("domain_robust_gate_root_controller", PROGRAM_PATH)
    if program.CONTROLLER_LOCK_PATH.resolve() != LOCK_PATH.resolve():
        raise VersionForwardTransactionError("root controller lock path drift")
    path_checks = {
        "state": Path(program.STATE_PATH).resolve() == STATE_PATH.resolve(),
        "transaction": Path(program.STATE_TRANSACTION_PATH).resolve()
        == TRANSACTION_PATH.resolve(),
        "ledger": Path(program.LEDGER_PATH).resolve() == LEDGER_PATH.resolve(),
        "genesis": Path(program.LEDGER_GENESIS_PATH).resolve()
        == GENESIS_PATH.resolve(),
    }
    if not all(path_checks.values()):
        raise VersionForwardTransactionError(
            f"root controller storage-path drift: {path_checks}"
        )
    if (
        program_before != _source_record(PROGRAM_PATH)
        or adapter_before != _source_record(TRANSACTION_SOURCE_PATH)
    ):
        raise VersionForwardTransactionError(
            "controller adapter/root source changed during authenticated import"
        )
    program.verify_ledger_chain = _strict_chain_for_root
    program._commit_state_with_events = _durable_commit_state_with_events
    program._require_presealed_verifier_contract = (
        _require_presealed_verifier_contract
    )
    return program


def _fixed_time_source(created_unix_ns: int) -> Any:
    class FixedTime:
        @staticmethod
        def time_ns() -> int:
            return created_unix_ns

    return FixedTime()


def _preview_controller_proposal_locked(
    base_state: Mapping[str, Any], *, created_unix_ns: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the authenticated root transition without publishing bytes."""

    _require_adapter_lock()
    if type(created_unix_ns) is not int or created_unix_ns <= 0:
        raise VersionForwardTransactionError(
            "prospective version-forward timestamp is invalid"
        )
    program = _load_authenticated_program_locked(require_receipt=False)
    public = getattr(program, "version_forward", None)
    raw = getattr(public, "__wrapped__", None)
    if raw is None or not callable(raw):
        raise VersionForwardTransactionError(
            "root version-forward preview entry point drift"
        )
    captured: dict[str, Any] = {}

    def capture(
        state: dict[str, Any], events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if captured:
            raise VersionForwardTransactionError(
                "root version-forward preview committed more than once"
            )
        captured["state"] = json.loads(json.dumps(state))
        captured["events"] = json.loads(json.dumps(events))
        return json.loads(json.dumps(state))

    program._status_locked = lambda: json.loads(json.dumps(dict(base_state)))
    program._commit_state_with_events = capture
    program.time = _fixed_time_source(created_unix_ns)
    if getattr(program._LOCK_LOCAL, "held", False):
        raise VersionForwardTransactionError(
            "root controller preview lock-local state was pre-held"
        )
    program._LOCK_LOCAL.held = True
    try:
        result = raw(TARGET_ATTEMPT, INVALIDITY_PATH, SEAL_PATH)
    finally:
        program._LOCK_LOCAL.held = False
    if (
        not isinstance(result, dict)
        or set(captured) != {"state", "events"}
        or result != captured["state"]
        or not isinstance(captured["events"], list)
        or not captured["events"]
    ):
        raise VersionForwardTransactionError(
            "root version-forward preview result drift"
        )
    return dict(captured["state"]), list(captured["events"])


def _predicted_post_verifiers(
    pre_verifiers: Mapping[str, Any]
) -> dict[str, Any]:
    standalone = json.loads(json.dumps(pre_verifiers["standalone"]))
    standalone["phase"] = "post-forward"
    return {
        "standalone": standalone,
        "independent_partitions": json.loads(
            json.dumps(pre_verifiers["independent_partitions"])
        ),
    }


def _receipt_context_core(
    *,
    journal_created_unix_ns: int,
    forward_created_unix_ns: int,
    receipt_created_unix_ns: int,
    fixed_inputs: Mapping[str, Any],
    pre_snapshot: Mapping[str, Any],
    pre_verifiers: Mapping[str, Any],
    controller_proposal: Mapping[str, Any],
    predicted_post_verifiers: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_attempt": SOURCE_ATTEMPT,
        "target_attempt": TARGET_ATTEMPT,
        "resume_state": RESUME_STATE,
        "journal_created_unix_ns": journal_created_unix_ns,
        "forward_created_unix_ns": forward_created_unix_ns,
        "receipt_created_unix_ns": receipt_created_unix_ns,
        "fixed_inputs": json.loads(json.dumps(dict(fixed_inputs))),
        "pre_snapshot": json.loads(json.dumps(dict(pre_snapshot))),
        "pre_verifiers": json.loads(json.dumps(dict(pre_verifiers))),
        "controller_proposal": json.loads(
            json.dumps(dict(controller_proposal))
        ),
        "predicted_post_verifiers": json.loads(
            json.dumps(dict(predicted_post_verifiers))
        ),
    }


def _build_receipt_value(
    *,
    context: Mapping[str, Any],
    receipt_context_sha256: str,
    controller_return: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
    post_verifiers: Mapping[str, Any],
) -> dict[str, Any]:
    fixed = context["fixed_inputs"]
    return {
        "schema_version": 2,
        "artifact_type": (
            "atomic_zero_confirmation_outcome_version_forward_transaction_receipt"
        ),
        "authorization_kind": "locked_v007_to_v008_inherited_pre_selection_forward",
        "source_attempt": SOURCE_ATTEMPT,
        "target_attempt": TARGET_ATTEMPT,
        "resume_state": RESUME_STATE,
        "journal_created_unix_ns": context["journal_created_unix_ns"],
        "forward_created_unix_ns": context["forward_created_unix_ns"],
        "created_unix_ns": context["receipt_created_unix_ns"],
        "receipt_context_sha256": receipt_context_sha256,
        "receipt_context": json.loads(json.dumps(dict(context))),
        "seal": fixed["seal"],
        "invalidity": fixed["invalidity"],
        "invalidity_draft": fixed["invalidity_draft"],
        "prior_receipt": fixed["prior_receipt"],
        "root_program": fixed["root_program"],
        "transaction_source": fixed["transaction_source"],
        "pre_snapshot": context["pre_snapshot"],
        "pre_verifiers": context["pre_verifiers"],
        "controller_return": json.loads(json.dumps(dict(controller_return))),
        "controller_return_sha256": _canonical_sha256(controller_return),
        "post_snapshot": json.loads(json.dumps(dict(post_snapshot))),
        "post_verifiers": json.loads(json.dumps(dict(post_verifiers))),
        "outcome_counts": {
            key: controller_return[key] for key in ZERO_OUTCOMES
        },
        "state_transaction_absent": True,
        "receipt_journal_absent": True,
        "passed": True,
    }


def _receipt_journal_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("transaction_sha256", None)
    return _canonical_sha256(payload)


def _build_receipt_journal_locked(
    *, prospective_created_unix_ns: int | None = None
) -> dict[str, Any]:
    _require_adapter_lock()
    rich_pre_snapshot, pre_state = _state_ledger_snapshot()
    _validate_pre_state(pre_state)
    fixed_paths = {
        "seal": SEAL_PATH,
        "invalidity": INVALIDITY_PATH,
        "invalidity_draft": INVALIDITY_DRAFT_PATH,
        "prior_receipt": PRIOR_RECEIPT_PATH,
        "root_program": PROGRAM_PATH,
        "transaction_source": TRANSACTION_SOURCE_PATH,
    }
    fixed_inputs: dict[str, Any] = {}
    for name, path in fixed_paths.items():
        record = _snapshot_file(path)[0]
        if name in {"root_program", "transaction_source"}:
            record["ast_sha256"] = _source_ast_sha256(path)
        fixed_inputs[name] = record
    seal = _json_payload(_snapshot_file(SEAL_PATH)[1], "inheritance seal")
    if (
        seal.get("root_controls", {}).get("program.py", {}).get("sha256")
        != fixed_inputs["root_program"]["sha256"]
        or seal.get("invalidity_evidence", {}).get("sha256")
        != fixed_inputs["invalidity"]["sha256"]
        or seal.get("superseded_invalidity_draft", {}).get("sha256")
        != fixed_inputs["invalidity_draft"]["sha256"]
        or seal.get("source_version_forward_transaction_receipt", {}).get("sha256")
        != fixed_inputs["prior_receipt"]["sha256"]
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal inputs are not seal-bound"
        )
    expected_projection = _authenticated_active_seal_verifier_projection(
        seal_relative=str(fixed_inputs["seal"]["path"]),
        seal_sha256=str(fixed_inputs["seal"]["sha256"]),
    )
    pre_verifiers = _preverify()
    stable_rich_snapshot, stable_state = _state_ledger_snapshot()
    fixed_stable: dict[str, Any] = {}
    for name, path in fixed_paths.items():
        record = _snapshot_file(path)[0]
        if name in {"root_program", "transaction_source"}:
            record["ast_sha256"] = _source_ast_sha256(path)
        fixed_stable[name] = record
    if (
        stable_rich_snapshot != rich_pre_snapshot
        or stable_state != pre_state
        or fixed_stable != fixed_inputs
    ):
        raise VersionForwardTransactionError(
            "receipt-journal snapshot changed during preverification"
        )
    _validate_receipt_verifier_result(
        pre_verifiers,
        phase="pre-forward",
        seal_relative=str(fixed_inputs["seal"]["path"]),
        seal_sha256=str(fixed_inputs["seal"]["sha256"]),
        expected_projection=expected_projection,
    )
    journal_created = (
        time.time_ns()
        if prospective_created_unix_ns is None
        else prospective_created_unix_ns
    )
    if type(journal_created) is not int or journal_created <= 0:
        raise VersionForwardTransactionError(
            "receipt-journal prospective timestamp is invalid"
        )
    forward_created = journal_created + 1
    receipt_created = journal_created + 2
    pre_snapshot = _stable_receipt_snapshot(rich_pre_snapshot)
    raw_proposal_state, raw_proposal_events = _preview_controller_proposal_locked(
        pre_state, created_unix_ns=forward_created
    )
    proposal_state, proposal_events = _rollover_source_adapter_marker(
        pre_state,
        raw_proposal_state,
        raw_proposal_events,
        rich_pre_snapshot["ledger"],
    )
    post_verifiers = _predicted_post_verifiers(pre_verifiers)
    proposal = {"state": proposal_state, "events": proposal_events}
    context = _receipt_context_core(
        journal_created_unix_ns=journal_created,
        forward_created_unix_ns=forward_created,
        receipt_created_unix_ns=receipt_created,
        fixed_inputs=fixed_inputs,
        pre_snapshot=pre_snapshot,
        pre_verifiers=pre_verifiers,
        controller_proposal=proposal,
        predicted_post_verifiers=post_verifiers,
    )
    context_sha256 = _canonical_sha256(context)
    ledger_payload = _snapshot_file(LEDGER_PATH)[1]
    genesis_record, genesis_payload = _snapshot_file(GENESIS_PATH)
    pending = _assemble_pending(
        proposal_state,
        proposal_events,
        base_state=pre_state,
        base_state_payload=_pretty_bytes(pre_state),
        genesis_record=genesis_record,
        genesis_payload=genesis_payload,
        ledger_payload=ledger_payload,
        receipt_context_sha256=context_sha256,
    )
    post_snapshot = _predicted_post_snapshot(
        pending,
        base_ledger_payload=ledger_payload,
        pre_snapshot=pre_snapshot,
    )
    receipt = _build_receipt_value(
        context=context,
        receipt_context_sha256=context_sha256,
        controller_return=pending["intended_state"],
        post_snapshot=post_snapshot,
        post_verifiers=post_verifiers,
    )
    journal: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": RECEIPT_JOURNAL_ARTIFACT_TYPE,
        "authorization_kind": ADAPTER_AUTHORIZATION_KIND,
        "receipt_context": context,
        "receipt_context_sha256": context_sha256,
        "controller_proposal": proposal,
        "predicted_controller_transaction": pending,
        "predicted_post_snapshot": post_snapshot,
        "predicted_post_verifiers": post_verifiers,
        "predicted_receipt": receipt,
        "predicted_receipt_sha256": hashlib.sha256(
            _pretty_bytes(receipt)
        ).hexdigest(),
    }
    journal["transaction_sha256"] = _receipt_journal_digest(journal)
    return journal


def _validate_receipt_journal_locked(value: Any) -> dict[str, Any]:
    _require_adapter_lock()
    expected_keys = {
        "schema_version",
        "artifact_type",
        "authorization_kind",
        "receipt_context",
        "receipt_context_sha256",
        "controller_proposal",
        "predicted_controller_transaction",
        "predicted_post_snapshot",
        "predicted_post_verifiers",
        "predicted_receipt",
        "predicted_receipt_sha256",
        "transaction_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise VersionForwardTransactionError(
            "version-forward receipt journal closed schema drift"
        )
    journal = dict(value)
    if (
        type(journal.get("schema_version")) is not int
        or journal.get("schema_version") != 1
        or journal.get("artifact_type") != RECEIPT_JOURNAL_ARTIFACT_TYPE
        or journal.get("authorization_kind") != ADAPTER_AUTHORIZATION_KIND
        or journal.get("transaction_sha256")
        != _receipt_journal_digest(journal)
        or not _is_sha256(journal.get("receipt_context_sha256"))
        or not _is_sha256(journal.get("predicted_receipt_sha256"))
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal header drift"
        )
    context = journal.get("receipt_context")
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
    if not isinstance(context, Mapping) or set(context) != context_keys:
        raise VersionForwardTransactionError(
            "version-forward receipt context schema drift"
        )
    timestamps = (
        context.get("journal_created_unix_ns"),
        context.get("forward_created_unix_ns"),
        context.get("receipt_created_unix_ns"),
    )
    if (
        type(context.get("schema_version")) is not int
        or context.get("schema_version") != 1
        or context.get("source_attempt") != SOURCE_ATTEMPT
        or context.get("target_attempt") != TARGET_ATTEMPT
        or context.get("resume_state") != RESUME_STATE
        or any(type(item) is not int or int(item) <= 0 for item in timestamps)
        or timestamps[1] != timestamps[0] + 1
        or timestamps[2] != timestamps[0] + 2
        or _canonical_sha256(context)
        != journal.get("receipt_context_sha256")
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt context identity/timestamp drift"
        )
    fixed_inputs = context.get("fixed_inputs")
    if not isinstance(fixed_inputs, Mapping) or set(fixed_inputs) != {
        "seal",
        "invalidity",
        "invalidity_draft",
        "prior_receipt",
        "root_program",
        "transaction_source",
    }:
        raise VersionForwardTransactionError(
            "version-forward receipt context input schema drift"
        )
    for name, path in {
        "seal": SEAL_PATH,
        "invalidity": INVALIDITY_PATH,
        "invalidity_draft": INVALIDITY_DRAFT_PATH,
        "prior_receipt": PRIOR_RECEIPT_PATH,
        "root_program": PROGRAM_PATH,
        "transaction_source": TRANSACTION_SOURCE_PATH,
    }.items():
        observed = _snapshot_file(path)[0]
        if name in {"root_program", "transaction_source"}:
            observed["ast_sha256"] = _source_ast_sha256(path)
        if fixed_inputs.get(name) != observed:
            raise VersionForwardTransactionError(
                f"version-forward receipt context input drift: {name}"
            )
    pre_snapshot = _validate_stable_receipt_snapshot(
        context.get("pre_snapshot"), label="journal pre"
    )
    pre_state = pre_snapshot["state"]["object"]
    _validate_pre_state(pre_state)
    current_ledger = _regular_bytes(LEDGER_PATH, "research ledger")
    pre_ledger_record = pre_snapshot["ledger"]
    pre_bytes = int(pre_ledger_record["bytes"])
    pre_ledger = current_ledger[:pre_bytes]
    if (
        len(pre_ledger) != pre_bytes
        or hashlib.sha256(pre_ledger).hexdigest()
        != pre_ledger_record["sha256"]
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal pre-ledger prefix drift"
        )
    genesis_record, genesis_payload = _snapshot_file(GENESIS_PATH)
    if (
        pre_snapshot["genesis"]
        != {
            key: genesis_record[key] for key in ("path", "sha256", "bytes")
        }
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal genesis drift"
        )
    pre_chain = _ledger_summary(pre_ledger, genesis_payload)
    if (
        pre_chain["event_count"] != pre_ledger_record["event_count"]
        or pre_chain["head_sha256"] != pre_ledger_record["head_sha256"]
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal pre-ledger chain drift"
        )
    pre_verifiers = context.get("pre_verifiers")
    seal_record = fixed_inputs["seal"]
    expected_projection = _authenticated_active_seal_verifier_projection(
        seal_relative=str(seal_record["path"]),
        seal_sha256=str(seal_record["sha256"]),
    )
    _validate_receipt_verifier_result(
        pre_verifiers,
        phase="pre-forward",
        seal_relative=str(seal_record["path"]),
        seal_sha256=str(seal_record["sha256"]),
        expected_projection=expected_projection,
    )
    raw_proposal_state, raw_proposal_events = _preview_controller_proposal_locked(
        pre_state, created_unix_ns=int(timestamps[1])
    )
    proposal_state, proposal_events = _rollover_source_adapter_marker(
        pre_state, raw_proposal_state, raw_proposal_events, pre_chain
    )
    proposal = {"state": proposal_state, "events": proposal_events}
    post_verifiers = _predicted_post_verifiers(pre_verifiers)
    expected_context = _receipt_context_core(
        journal_created_unix_ns=int(timestamps[0]),
        forward_created_unix_ns=int(timestamps[1]),
        receipt_created_unix_ns=int(timestamps[2]),
        fixed_inputs=fixed_inputs,
        pre_snapshot=pre_snapshot,
        pre_verifiers=pre_verifiers,
        controller_proposal=proposal,
        predicted_post_verifiers=post_verifiers,
    )
    context_sha256 = _canonical_sha256(expected_context)
    if (
        _canonical_bytes(context) != _canonical_bytes(expected_context)
        or journal.get("receipt_context_sha256") != context_sha256
        or _canonical_bytes(journal.get("controller_proposal"))
        != _canonical_bytes(proposal)
        or _canonical_bytes(journal.get("predicted_post_verifiers"))
        != _canonical_bytes(post_verifiers)
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal independently replayed context drift"
        )
    pending = _assemble_pending(
        proposal_state,
        proposal_events,
        base_state=pre_state,
        base_state_payload=_pretty_bytes(pre_state),
        genesis_record=genesis_record,
        genesis_payload=genesis_payload,
        ledger_payload=pre_ledger,
        receipt_context_sha256=context_sha256,
    )
    stored_pending = _validate_pending(
        journal.get("predicted_controller_transaction")
    )
    if _canonical_bytes(stored_pending) != _canonical_bytes(pending):
        raise VersionForwardTransactionError(
            "version-forward receipt journal controller prediction drift"
        )
    post_snapshot = _predicted_post_snapshot(
        pending,
        base_ledger_payload=pre_ledger,
        pre_snapshot=pre_snapshot,
    )
    if _canonical_bytes(journal.get("predicted_post_snapshot")) != _canonical_bytes(
        post_snapshot
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal post-snapshot prediction drift"
        )
    receipt = _build_receipt_value(
        context=expected_context,
        receipt_context_sha256=context_sha256,
        controller_return=pending["intended_state"],
        post_snapshot=post_snapshot,
        post_verifiers=post_verifiers,
    )
    if (
        _canonical_bytes(journal.get("predicted_receipt"))
        != _canonical_bytes(receipt)
        or journal.get("predicted_receipt_sha256")
        != hashlib.sha256(_pretty_bytes(receipt)).hexdigest()
    ):
        raise VersionForwardTransactionError(
            "version-forward receipt journal exact receipt prediction drift"
        )
    _validate_receipt_closed_schema(receipt)
    return journal


def _invoke_root_locked(
    operation: str,
    *args: Any,
    require_receipt: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    _require_adapter_lock()
    recovered_state, recovered_chain = _recover_durable_transaction_locked()
    if require_receipt:
        if recovered_state.get("active_attempt") != TARGET_ATTEMPT:
            raise VersionForwardTransactionError(
                "v008 controller adapter requires active v008 lineage"
            )
        _validate_adapter_marker(recovered_state, recovered_chain)
    program = _load_authenticated_program_locked(require_receipt=require_receipt)
    fixed_forward_time = getattr(_ADAPTER_LOCAL, "forward_created_unix_ns", None)
    if operation == "version_forward" and fixed_forward_time is not None:
        if type(fixed_forward_time) is not int or fixed_forward_time <= 0:
            raise VersionForwardTransactionError(
                "version-forward fixed controller timestamp drift"
            )
        program.time = _fixed_time_source(fixed_forward_time)
    if operation == "status":
        raw: Callable[..., dict[str, Any]] | None = getattr(
            program, "_status_locked", None
        )
    else:
        public = getattr(program, operation, None)
        raw = getattr(public, "__wrapped__", None)
    if raw is None or not callable(raw):
        raise VersionForwardTransactionError(
            f"root controller narrow operation wrapper drift: {operation}"
        )
    if getattr(program._LOCK_LOCAL, "held", False):
        raise VersionForwardTransactionError("root controller lock-local state was pre-held")
    program._LOCK_LOCAL.held = True
    try:
        result = raw(*args, **kwargs)
    finally:
        program._LOCK_LOCAL.held = False
    if not isinstance(result, dict):
        raise VersionForwardTransactionError("root controller returned a non-object")
    final_state, final_chain = _recover_durable_transaction_locked()
    if dict(result) != final_state:
        raise VersionForwardTransactionError(
            "root controller return differs from durably installed STATE"
        )
    _validate_adapter_marker(final_state, final_chain)
    if require_receipt:
        verify_transaction_receipt()
    return result


def _invoke_controller() -> dict[str, Any]:
    return _invoke_root_locked(
        "version_forward",
        TARGET_ATTEMPT,
        INVALIDITY_PATH,
        SEAL_PATH,
        require_receipt=False,
    )


def _adapter_operation(
    operation: str, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    with _adapter_lock():
        return _invoke_root_locked(
            operation, *args, require_receipt=True, **kwargs
        )


def status() -> dict[str, Any]:
    return _adapter_operation("status")


def advance(
    completed_state: str,
    evidence: Path,
    checkpoint_name: str,
    next_action: str,
) -> dict[str, Any]:
    return _adapter_operation(
        "advance",
        completed_state,
        Path(evidence).resolve(),
        checkpoint_name,
        next_action,
    )


def update_counts(**counts: int | bool) -> dict[str, Any]:
    validated = _validate_count_assignments(counts)
    return _adapter_operation("update_counts", **validated)


def stage_early_scientific_failure(mode: str) -> dict[str, Any]:
    return _adapter_operation("stage_early_scientific_failure", mode)


def finalize_early_scientific_failure(mode: str) -> dict[str, Any]:
    return _adapter_operation("finalize_early_scientific_failure", mode)


def stage_postconfirmation_integrity_failure() -> dict[str, Any]:
    return _adapter_operation("stage_postconfirmation_integrity_failure")


def finalize_postconfirmation_execution_invalid() -> dict[str, Any]:
    return _adapter_operation("finalize_postconfirmation_execution_invalid")


def record_terminal(
    label: str, decision: Path, process_valid: bool
) -> dict[str, Any]:
    if type(process_valid) is not bool:
        raise VersionForwardTransactionError("process_valid must be exact boolean")
    return _adapter_operation(
        "record_terminal", label, Path(decision).resolve(), process_valid
    )


def version_forward(
    new_attempt: str, invalidity: Path, equivalence: Path
) -> dict[str, Any]:
    if (
        new_attempt != TARGET_ATTEMPT
        or Path(invalidity).resolve() != INVALIDITY_PATH.resolve()
        or Path(equivalence).resolve() != SEAL_PATH.resolve()
    ):
        raise VersionForwardTransactionError(
            "adapter permits only the sealed exact v007-to-v008 forward edge"
        )
    return execute_version_forward()


def _load_or_recover_receipt_journal_locked() -> dict[str, Any] | None:
    _require_adapter_lock()
    journal_exists = os.path.lexists(RECEIPT_JOURNAL_PATH)
    staging_exists = os.path.lexists(RECEIPT_JOURNAL_STAGING_PATH)
    if not journal_exists and not staging_exists:
        return None
    final_payload: bytes | None = None
    staging_payload: bytes | None = None
    final_meta: os.stat_result | None = None
    staging_meta: os.stat_result | None = None
    if journal_exists:
        final_payload, final_meta = _journal_alias_bytes(
            RECEIPT_JOURNAL_PATH, label="version-forward receipt journal"
        )
    if staging_exists:
        staging_payload, staging_meta = _journal_alias_bytes(
            RECEIPT_JOURNAL_STAGING_PATH,
            label="version-forward receipt journal staging",
        )
    if final_meta is not None and staging_meta is not None:
        if (
            final_payload != staging_payload
            or (final_meta.st_dev, final_meta.st_ino)
            != (staging_meta.st_dev, staging_meta.st_ino)
            or final_meta.st_nlink != 2
            or staging_meta.st_nlink != 2
        ):
            raise VersionForwardTransactionError(
                "receipt journal and staging are competing aliases"
            )
    elif final_meta is not None and final_meta.st_nlink != 1:
        raise VersionForwardTransactionError(
            "published receipt journal has a hidden alias"
        )
    elif staging_meta is not None and staging_meta.st_nlink != 1:
        raise VersionForwardTransactionError(
            "unpublished receipt journal staging has a hidden alias"
        )
    payload = final_payload if final_payload is not None else staging_payload
    if payload is None:
        raise VersionForwardTransactionError(
            "receipt journal publication has no readable alias"
        )
    journal = _json_payload(payload, "version-forward receipt journal")
    if payload != _pretty_bytes(journal):
        raise VersionForwardTransactionError(
            "version-forward receipt journal is not canonical JSON"
        )
    validated = _validate_receipt_journal_locked(journal)
    if staging_exists:
        if any(
            os.path.lexists(path)
            for path in (
                TRANSACTION_PATH,
                PENDING_STAGING_PATH,
                STATE_STAGING_PATH,
                RECEIPT_PATH,
                RECEIPT_STAGING_PATH,
            )
        ):
            raise VersionForwardTransactionError(
                "unpublished receipt journal staging has later transaction residue"
            )
        current, _state = _state_ledger_snapshot()
        if _stable_receipt_snapshot(current) != validated["receipt_context"][
            "pre_snapshot"
        ]:
            raise VersionForwardTransactionError(
                "unpublished receipt journal staging is not at its exact pre-state"
            )
    # This is deliberately unconditional for a recovered journal.  A crash
    # immediately after staging unlink can leave only the exact final alias
    # while the directory unlink itself still needs a new durability barrier.
    _complete_receipt_journal_publication(
        _pretty_bytes(validated), recovery=True
    )
    return validated


def _reconcile_controller_transaction_from_journal_locked(
    journal: Mapping[str, Any]
) -> None:
    _require_adapter_lock()
    predicted = journal["predicted_controller_transaction"]
    pending_exists = os.path.lexists(TRANSACTION_PATH)
    pending_staging_exists = os.path.lexists(PENDING_STAGING_PATH)
    if pending_exists and pending_staging_exists:
        raise VersionForwardTransactionError(
            "controller pending and staging coexist under receipt journal"
        )
    if pending_exists or pending_staging_exists:
        path = TRANSACTION_PATH if pending_exists else PENDING_STAGING_PATH
        observed, _payload = _canonical_object_file(
            path, "journal-bound controller pending transaction"
        )
        if observed != predicted:
            raise VersionForwardTransactionError(
                "controller pending differs from receipt-journal prediction"
            )
        _recover_durable_transaction_locked()
    elif os.path.lexists(STATE_STAGING_PATH):
        raise VersionForwardTransactionError(
            "controller STATE staging exists without its journal-bound pending"
        )


def _resume_receipt_journal_locked(
    journal: Mapping[str, Any]
) -> dict[str, Any]:
    _require_adapter_lock()
    validated = _validate_receipt_journal_locked(journal)
    pre = validated["receipt_context"]["pre_snapshot"]

    # A crash can leave the journal-bound initial pending transaction durable
    # while STATE and the ledger are still exactly at the prospective
    # pre-state.  Re-run the live read-only pre-verifiers *before* reconciling
    # that pending transaction: reconciliation is the first irreversible
    # STATE/ledger mutation, and a coherently reprojected journal must not be
    # allowed to defer discovery of verifier drift until after it.
    state_record, _state_payload = _snapshot_file(STATE_PATH)
    ledger_record, _ledger_payload = _snapshot_file(LEDGER_PATH)
    genesis_record, _genesis_payload = _snapshot_file(GENESIS_PATH)
    exactly_at_pre_files = all(
        observed["sha256"] == expected["sha256"]
        and observed["bytes"] == expected["bytes"]
        for observed, expected in (
            (state_record, pre["state"]),
            (ledger_record, pre["ledger"]),
            (genesis_record, pre["genesis"]),
        )
    )
    preverified_at_exact_pre = False
    if exactly_at_pre_files:
        live_pre_verifiers = _preverify()
        if _canonical_bytes(live_pre_verifiers) != _canonical_bytes(
            validated["receipt_context"]["pre_verifiers"]
        ):
            raise VersionForwardTransactionError(
                "live preverification differs from the durable receipt context"
            )
        stable_after_preverify, _state_after_preverify = _state_ledger_snapshot()
        if _canonical_bytes(_stable_receipt_snapshot(stable_after_preverify)) != (
            _canonical_bytes(pre)
        ):
            raise VersionForwardTransactionError(
                "pre-forward state changed during live preverification"
            )
        preverified_at_exact_pre = True

    _reconcile_controller_transaction_from_journal_locked(validated)
    current_rich, _current_state = _state_ledger_snapshot()
    current = _stable_receipt_snapshot(current_rich)
    post = validated["predicted_post_snapshot"]
    if current == pre:
        if os.path.lexists(RECEIPT_PATH) or os.path.lexists(
            RECEIPT_STAGING_PATH
        ):
            raise VersionForwardTransactionError(
                "receipt exists before its predicted controller commit"
            )
        if not preverified_at_exact_pre:
            raise VersionForwardTransactionError(
                "exact pre-state was not live-preverified before controller commit"
            )
        _ADAPTER_LOCAL.receipt_context_sha256 = validated[
            "receipt_context_sha256"
        ]
        _ADAPTER_LOCAL.forward_created_unix_ns = validated["receipt_context"][
            "forward_created_unix_ns"
        ]
        try:
            controller_return = _invoke_controller()
        finally:
            if hasattr(_ADAPTER_LOCAL, "receipt_context_sha256"):
                delattr(_ADAPTER_LOCAL, "receipt_context_sha256")
            if hasattr(_ADAPTER_LOCAL, "forward_created_unix_ns"):
                delattr(_ADAPTER_LOCAL, "forward_created_unix_ns")
        if _canonical_bytes(controller_return) != _canonical_bytes(
            validated["predicted_controller_transaction"]["intended_state"]
        ):
            raise VersionForwardTransactionError(
                "live controller return differs from receipt-journal prediction"
            )
        current_rich, _current_state = _state_ledger_snapshot()
        current = _stable_receipt_snapshot(current_rich)
    if current != post:
        raise VersionForwardTransactionError(
            "receipt journal found neither its exact pre-state nor exact post-state"
        )
    recovering_receipt_publication = os.path.lexists(
        RECEIPT_PATH
    ) or os.path.lexists(RECEIPT_STAGING_PATH)
    if recovering_receipt_publication:
        # Production closure rejects dot-staging files and nlink-2 aliases.
        # Close only the journal-predicted bytes before rerunning it.
        _publish_or_recover_exact_receipt(validated)
    actual_post_verifiers = _postverify()
    if _canonical_bytes(actual_post_verifiers) != _canonical_bytes(
        validated["predicted_post_verifiers"]
    ):
        raise VersionForwardTransactionError(
            "post-forward verifier result differs from prospective journal"
        )
    stable_after_verify, _state_after_verify = _state_ledger_snapshot()
    if _stable_receipt_snapshot(stable_after_verify) != post:
        raise VersionForwardTransactionError(
            "post-forward state changed during receipt verification"
        )
    if not recovering_receipt_publication:
        _publish_or_recover_exact_receipt(validated)
    verify_transaction_receipt(_allow_receipt_journal=True)
    _unlink_receipt_journal(validated)
    return verify_transaction_receipt()


def execute_version_forward() -> dict[str, Any]:
    descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if getattr(_ADAPTER_LOCAL, "held", False):
            raise VersionForwardTransactionError(
                "version-forward adapter lock is unexpectedly re-entered"
            )
        _ADAPTER_LOCAL.held = True
        journal = _load_or_recover_receipt_journal_locked()
        if journal is not None:
            return _resume_receipt_journal_locked(journal)
        if os.path.lexists(RECEIPT_STAGING_PATH):
            raise VersionForwardTransactionError(
                "receipt staging exists without its durable context journal"
            )
        if os.path.lexists(RECEIPT_PATH):
            return verify_transaction_receipt()
        if any(
            os.path.lexists(path)
            for path in (TRANSACTION_PATH, PENDING_STAGING_PATH, STATE_STAGING_PATH)
        ):
            raise VersionForwardTransactionError(
                "controller transaction exists without a receipt context journal"
            )
        current, state = _state_ledger_snapshot()
        _validate_pre_state(state)
        journal = _build_receipt_journal_locked()
        if _stable_receipt_snapshot(current) != journal["receipt_context"][
            "pre_snapshot"
        ]:
            raise VersionForwardTransactionError(
                "pre-state changed before receipt-journal publication"
            )
        _publish_receipt_journal(journal)
        return _resume_receipt_journal_locked(journal)
    finally:
        for name in ("receipt_context_sha256", "forward_created_unix_ns"):
            if hasattr(_ADAPTER_LOCAL, name):
                delattr(_ADAPTER_LOCAL, name)
        _ADAPTER_LOCAL.held = False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _receipt_snapshot_record(
    value: Any,
    *,
    expected_path: Path,
    extra_keys: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    base_keys = {"path", "sha256", "bytes", "device", "inode", "nlink"}
    if not isinstance(value, Mapping) or set(value) != base_keys | set(extra_keys):
        raise VersionForwardTransactionError(
            f"transaction snapshot schema drift: {expected_path}"
        )
    if value.get("path") != _relative(expected_path) or not _is_sha256(
        value.get("sha256")
    ):
        raise VersionForwardTransactionError(
            f"transaction snapshot path/hash drift: {expected_path}"
        )
    for key in ("bytes", "device", "inode", "nlink"):
        if type(value.get(key)) is not int or int(value[key]) < 0:
            raise VersionForwardTransactionError(
                f"transaction snapshot metadata type drift: {expected_path}/{key}"
            )
    if value.get("nlink") != 1:
        raise VersionForwardTransactionError(
            f"transaction snapshot link-count drift: {expected_path}"
        )
    return dict(value)


def _validate_receipt_verifier_result(
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
        raise VersionForwardTransactionError(
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
            raise VersionForwardTransactionError(
                "transaction producer result closed schema drift"
            )
        if not (
            producer.get("passed") is True
            and producer.get("phase") == phase
            and producer.get("attempt") == TARGET_ATTEMPT
            and producer.get("seal_path") == seal_relative
            and producer.get("seal_sha256") == seal_sha256
            and _is_sha256(producer.get("state_sha256"))
            and _is_sha256(producer.get("ledger_sha256"))
            and producer.get("outcome_arrays_opened") is False
            and producer.get("read_only") is True
        ):
            raise VersionForwardTransactionError("transaction producer result drift")
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
        raise VersionForwardTransactionError(
            f"transaction standalone {phase} closed schema drift"
        )
    if not (
        standalone.get("passed") is True
        and standalone.get("phase") == phase
        and standalone.get("attempt") == TARGET_ATTEMPT
        and standalone.get("active_attempt") == TARGET_ATTEMPT
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
        and standalone.get("output_paths_created") == 0
        and type(standalone.get("output_paths_created")) is int
        and standalone.get("read_only") is True
    ):
        raise VersionForwardTransactionError(
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
        raise VersionForwardTransactionError(
            f"transaction independent {phase} result drift"
        )


def _validate_receipt_closed_schema(receipt: Any) -> dict[str, Any]:
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
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise VersionForwardTransactionError("transaction receipt closed schema drift")
    value = dict(receipt)
    outcome_counts = value.get("outcome_counts")
    exact_count_types = (
        isinstance(outcome_counts, Mapping)
        and set(outcome_counts) == set(ZERO_OUTCOMES)
        and all(
            (
                type(outcome_counts[key]) is bool
                if type(expected) is bool
                else type(outcome_counts[key]) is int
            )
            for key, expected in ZERO_OUTCOMES.items()
        )
    )
    if not (
        type(value.get("schema_version")) is int
        and value.get("schema_version") == 2
        and value.get("artifact_type")
        == "atomic_zero_confirmation_outcome_version_forward_transaction_receipt"
        and value.get("authorization_kind")
        == "locked_v007_to_v008_inherited_pre_selection_forward"
        and value.get("source_attempt") == SOURCE_ATTEMPT
        and value.get("target_attempt") == TARGET_ATTEMPT
        and value.get("resume_state") == RESUME_STATE
        and type(value.get("journal_created_unix_ns")) is int
        and type(value.get("forward_created_unix_ns")) is int
        and type(value.get("created_unix_ns")) is int
        and int(value["created_unix_ns"]) > 0
        and value["forward_created_unix_ns"]
        == value["journal_created_unix_ns"] + 1
        and value["created_unix_ns"] == value["journal_created_unix_ns"] + 2
        and _is_sha256(value.get("receipt_context_sha256"))
        and isinstance(value.get("receipt_context"), Mapping)
        and _canonical_sha256(value["receipt_context"])
        == value["receipt_context_sha256"]
        and exact_count_types
        and dict(outcome_counts) == ZERO_OUTCOMES
        and value.get("state_transaction_absent") is True
        and value.get("receipt_journal_absent") is True
        and value.get("passed") is True
        and isinstance(value.get("controller_return"), Mapping)
        and _is_sha256(value.get("controller_return_sha256"))
    ):
        raise VersionForwardTransactionError("transaction receipt header/type drift")
    fixed = {
        "seal": (SEAL_PATH, frozenset()),
        "invalidity": (INVALIDITY_PATH, frozenset()),
        "invalidity_draft": (INVALIDITY_DRAFT_PATH, frozenset()),
        "prior_receipt": (PRIOR_RECEIPT_PATH, frozenset()),
        "root_program": (PROGRAM_PATH, frozenset({"ast_sha256"})),
        "transaction_source": (
            TRANSACTION_SOURCE_PATH,
            frozenset({"ast_sha256"}),
        ),
    }
    fixed_records: dict[str, dict[str, Any]] = {}
    for name, (path, extra) in fixed.items():
        record = _receipt_snapshot_record(
            value.get(name), expected_path=path, extra_keys=extra
        )
        if extra and (
            not _is_sha256(record.get("ast_sha256"))
            or record.get("ast_sha256") != _source_ast_sha256(path)
        ):
            raise VersionForwardTransactionError(
                f"transaction receipt AST binding drift: {name}"
            )
        fixed_records[name] = record
    seal_relative = fixed_records["seal"]["path"]
    seal_sha256 = fixed_records["seal"]["sha256"]
    expected_projection = _authenticated_active_seal_verifier_projection(
        seal_relative=seal_relative,
        seal_sha256=seal_sha256,
    )
    pre_snapshot = _validate_stable_receipt_snapshot(
        value.get("pre_snapshot"), label="pre"
    )
    post_snapshot = _validate_stable_receipt_snapshot(
        value.get("post_snapshot"), label="post"
    )
    _validate_receipt_verifier_result(
        value["pre_verifiers"],
        phase="pre-forward",
        seal_relative=seal_relative,
        seal_sha256=seal_sha256,
        expected_projection=expected_projection,
    )
    _validate_receipt_verifier_result(
        value["post_verifiers"],
        phase="post-forward",
        seal_relative=seal_relative,
        seal_sha256=seal_sha256,
        expected_projection=expected_projection,
    )
    context = value["receipt_context"]
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
    if (
        set(context) != context_keys
        or type(context.get("schema_version")) is not int
        or context.get("schema_version") != 1
        or context.get("source_attempt") != SOURCE_ATTEMPT
        or context.get("target_attempt") != TARGET_ATTEMPT
        or context.get("resume_state") != RESUME_STATE
        or type(context.get("journal_created_unix_ns")) is not int
        or type(context.get("forward_created_unix_ns")) is not int
        or type(context.get("receipt_created_unix_ns")) is not int
        or context.get("journal_created_unix_ns")
        != value["journal_created_unix_ns"]
        or context.get("forward_created_unix_ns")
        != value["forward_created_unix_ns"]
        or context.get("receipt_created_unix_ns")
        != value["created_unix_ns"]
        or _canonical_bytes(context.get("fixed_inputs"))
        != _canonical_bytes({name: value[name] for name in fixed})
        or _canonical_bytes(context.get("pre_snapshot"))
        != _canonical_bytes(pre_snapshot)
        or _canonical_bytes(context.get("pre_verifiers"))
        != _canonical_bytes(value["pre_verifiers"])
        or _canonical_bytes(context.get("predicted_post_verifiers"))
        != _canonical_bytes(value["post_verifiers"])
    ):
        raise VersionForwardTransactionError(
            "transaction receipt precommit context cross-link drift"
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
        raise VersionForwardTransactionError(
            "transaction receipt controller proposal schema drift"
        )
    pre_state = pre_snapshot["state"]["object"]
    _validate_pre_state(pre_state)
    current_ledger = _regular_bytes(LEDGER_PATH, "research ledger")
    pre_ledger = pre_snapshot["ledger"]
    pre_ledger_bytes = int(pre_ledger["bytes"])
    base_ledger = current_ledger[:pre_ledger_bytes]
    if (
        len(base_ledger) != pre_ledger_bytes
        or hashlib.sha256(base_ledger).hexdigest() != pre_ledger["sha256"]
    ):
        raise VersionForwardTransactionError(
            "transaction receipt pre-ledger prefix drift"
        )
    genesis_record, genesis_payload = _snapshot_file(GENESIS_PATH)
    stable_genesis = {
        key: genesis_record[key] for key in ("path", "sha256", "bytes")
    }
    if pre_snapshot["genesis"] != stable_genesis:
        raise VersionForwardTransactionError(
            "transaction receipt ledger-genesis drift"
        )
    pending = _assemble_pending(
        proposal["state"],
        proposal["events"],
        base_state=pre_state,
        base_state_payload=_pretty_bytes(pre_state),
        genesis_record=genesis_record,
        genesis_payload=genesis_payload,
        ledger_payload=base_ledger,
        receipt_context_sha256=str(value["receipt_context_sha256"]),
    )
    derived_post_snapshot = _predicted_post_snapshot(
        pending,
        base_ledger_payload=base_ledger,
        pre_snapshot=pre_snapshot,
    )
    if (
        _canonical_bytes(value.get("controller_return"))
        != _canonical_bytes(pending["intended_state"])
        or _canonical_bytes(post_snapshot)
        != _canonical_bytes(derived_post_snapshot)
    ):
        raise VersionForwardTransactionError(
            "transaction receipt post-state differs from its ledger-bound proposal"
        )
    rebuilt = _build_receipt_value(
        context=context,
        receipt_context_sha256=str(value["receipt_context_sha256"]),
        controller_return=value["controller_return"],
        post_snapshot=post_snapshot,
        post_verifiers=value["post_verifiers"],
    )
    if _canonical_bytes(rebuilt) != _canonical_bytes(value):
        raise VersionForwardTransactionError(
            "transaction receipt differs from its precommit context projection"
        )
    return value


def verify_transaction_receipt(
    *, _allow_receipt_journal: bool = False
) -> dict[str, Any]:
    receipt_record, receipt_payload = _snapshot_file(RECEIPT_PATH)
    decoded_receipt = _json_payload(
        receipt_payload, "version-forward transaction receipt"
    )
    if receipt_payload != _pretty_bytes(decoded_receipt):
        raise VersionForwardTransactionError(
            "version-forward transaction receipt is not canonical JSON"
        )
    receipt = _validate_receipt_closed_schema(decoded_receipt)
    if (
        receipt.get("schema_version") != 2
        or receipt.get("artifact_type")
        != "atomic_zero_confirmation_outcome_version_forward_transaction_receipt"
        or receipt.get("authorization_kind")
        != "locked_v007_to_v008_inherited_pre_selection_forward"
        or receipt.get("source_attempt") != SOURCE_ATTEMPT
        or receipt.get("target_attempt") != TARGET_ATTEMPT
        or receipt.get("resume_state") != RESUME_STATE
        or receipt.get("passed") is not True
        or receipt.get("state_transaction_absent") is not True
        or receipt.get("receipt_journal_absent") is not True
        or any(
            os.path.lexists(path)
            for path in (TRANSACTION_PATH, PENDING_STAGING_PATH, STATE_STAGING_PATH)
        )
        or os.path.lexists(RECEIPT_STAGING_PATH)
        or (
            not _allow_receipt_journal
            and any(
                os.path.lexists(path)
                for path in (
                    RECEIPT_JOURNAL_PATH,
                    RECEIPT_JOURNAL_STAGING_PATH,
                )
            )
        )
    ):
        raise VersionForwardTransactionError("transaction receipt header drift")
    pre_snapshot = receipt.get("pre_snapshot")
    post_snapshot = receipt.get("post_snapshot")
    if not isinstance(pre_snapshot, Mapping) or not isinstance(post_snapshot, Mapping):
        raise VersionForwardTransactionError("transaction receipt snapshots are absent")
    pre_state = pre_snapshot.get("state", {}).get("object")
    post_state = post_snapshot.get("state", {}).get("object")
    if not isinstance(pre_state, Mapping) or not isinstance(post_state, Mapping):
        raise VersionForwardTransactionError("transaction receipt state objects are absent")
    _validate_pre_state(pre_state)
    _validate_post_state(
        post_state,
        receipt.get("controller_return", {}),
        pre_snapshot,
        post_snapshot,
    )
    if (
        receipt["pre_verifiers"].get("independent_partitions")
        != receipt["post_verifiers"].get("independent_partitions")
        or receipt["pre_verifiers"]["producer"].get("state_sha256")
        != pre_snapshot["state"].get("sha256")
        or receipt["pre_verifiers"]["producer"].get("ledger_sha256")
        != pre_snapshot["ledger"].get("sha256")
    ):
        raise VersionForwardTransactionError(
            "transaction verifier/snapshot cross-link drift"
        )
    for label, snapshot in (("pre", pre_snapshot), ("post", post_snapshot)):
        state_record = snapshot.get("state")
        if not isinstance(state_record, Mapping):
            raise VersionForwardTransactionError(f"transaction {label} STATE snapshot drift")
        state_object = state_record.get("object")
        encoded = json.dumps(state_object, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if (
            state_record.get("sha256") != hashlib.sha256(encoded).hexdigest()
            or state_record.get("bytes") != len(encoded)
            or state_record.get("object_sha256") != _canonical_sha256(state_object)
        ):
            raise VersionForwardTransactionError(
                f"transaction {label} STATE byte/object binding drift"
            )
        ledger_record = snapshot.get("ledger")
        if (
            not isinstance(ledger_record, Mapping)
            or state_object.get("ledger_event_count")
            != ledger_record.get("event_count")
            or state_object.get("ledger_head_sha256")
            != ledger_record.get("head_sha256")
            or ledger_record.get("ledger_sha256") != ledger_record.get("sha256")
        ):
            raise VersionForwardTransactionError(
                f"transaction {label} STATE/ledger binding drift"
            )
    current, state = _state_ledger_snapshot()
    if state.get("active_attempt") != TARGET_ATTEMPT:
        raise VersionForwardTransactionError("current controller is not on receipt target attempt")
    _validate_outcome_counter_types(state)
    current_marker = _validate_adapter_marker(state, current["ledger"])
    post_ledger_record = post_snapshot.get("ledger", {})
    if (
        state.get("ledger_event_count") == post_ledger_record.get("event_count")
        and state.get("ledger_head_sha256")
        == post_ledger_record.get("head_sha256")
        and (
            _canonical_bytes(state) != _canonical_bytes(post_state)
            or _canonical_bytes(current_marker)
            != _canonical_bytes(post_state.get(ADAPTER_STATE_KEY))
        )
    ):
        raise VersionForwardTransactionError(
            "current initial adapter marker differs from receipt post-state"
        )
    current_ledger_payload = _snapshot_file(LEDGER_PATH)[1]
    for label, snapshot in (("pre", pre_snapshot), ("post", post_snapshot)):
        ledger_record = snapshot.get("ledger")
        if not isinstance(ledger_record, Mapping) or type(ledger_record.get("bytes")) is not int:
            raise VersionForwardTransactionError(
                f"transaction {label}-ledger snapshot drift"
            )
        historical = current_ledger_payload[: ledger_record["bytes"]]
        if (
            len(historical) != ledger_record["bytes"]
            or hashlib.sha256(historical).hexdigest() != ledger_record.get("sha256")
        ):
            raise VersionForwardTransactionError(
                f"transaction {label}-ledger prefix is stale"
            )
    post_ledger = post_snapshot.get("ledger")
    if not isinstance(post_ledger, Mapping) or type(post_ledger.get("bytes")) is not int:
        raise VersionForwardTransactionError("transaction post-ledger snapshot drift")
    historical = current_ledger_payload[: post_ledger["bytes"]]
    if (
        len(historical) != post_ledger["bytes"]
        or hashlib.sha256(historical).hexdigest() != post_ledger.get("sha256")
        or current["ledger"]["event_count"] < post_ledger.get("event_count", 0)
    ):
        raise VersionForwardTransactionError("transaction post-ledger prefix is stale")
    _validate_descendant_adapter_chain(
        receipt_post_state=post_state,
        receipt_post_ledger=post_ledger,
        current_state=state,
        ledger_payload=current_ledger_payload,
    )
    post_lines = historical.splitlines()
    if (
        len(post_lines) != post_ledger.get("event_count")
        or not post_lines
    ):
        raise VersionForwardTransactionError(
            "transaction post-ledger event framing drift"
        )
    forward_event = _json_payload(post_lines[-1], "version-forward ledger event")
    if (
        forward_event.get("event")
        != "zero_confirmation_outcome_version_forward"
        or forward_event.get(RECEIPT_CONTEXT_EVENT_FIELD)
        != receipt.get("receipt_context_sha256")
        or forward_event.get(SOURCE_ADAPTER_MARKER_FIELD)
        != pre_state.get(PRIOR_ADAPTER_STATE_KEY)
    ):
        raise VersionForwardTransactionError(
            "transaction receipt lacks its precommit ledger context binding"
        )
    if receipt.get("controller_return_sha256") != _canonical_sha256(receipt.get("controller_return")):
        raise VersionForwardTransactionError("transaction receipt controller-return hash drift")
    bindings = {
        "seal": SEAL_PATH,
        "invalidity": INVALIDITY_PATH,
        "invalidity_draft": INVALIDITY_DRAFT_PATH,
        "prior_receipt": PRIOR_RECEIPT_PATH,
        "root_program": PROGRAM_PATH,
        "transaction_source": TRANSACTION_SOURCE_PATH,
    }
    for name, path in bindings.items():
        observed, _payload = _snapshot_file(path)
        record = receipt.get(name)
        if not isinstance(record, Mapping) or any(record.get(key) != observed[key] for key in observed):
            raise VersionForwardTransactionError(f"transaction receipt binding drift: {name}")
        if name in {"root_program", "transaction_source"} and record.get("ast_sha256") != _source_ast_sha256(path):
            raise VersionForwardTransactionError(f"transaction receipt AST drift: {name}")
    return {
        "passed": True,
        "attempt": TARGET_ATTEMPT,
        "receipt_path": _relative(RECEIPT_PATH),
        "receipt_sha256": receipt_record["sha256"],
        "state_sha256": current["state"]["sha256"],
        "ledger_sha256": current["ledger"]["sha256"],
        "outcome_arrays_opened": False,
        "read_only": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("execute")
    subparsers.add_parser("verify-receipt")
    subparsers.add_parser("status")
    advance_parser = subparsers.add_parser("advance")
    advance_parser.add_argument("completed_state")
    advance_parser.add_argument("evidence", type=Path)
    advance_parser.add_argument("checkpoint_name")
    advance_parser.add_argument("next_action")
    count_parser = subparsers.add_parser("counts")
    count_parser.add_argument("assignments", nargs="+")
    forward_parser = subparsers.add_parser("version-forward")
    forward_parser.add_argument("new_attempt")
    forward_parser.add_argument("invalidity", type=Path)
    forward_parser.add_argument("equivalence", type=Path)
    terminal_parser = subparsers.add_parser("terminal")
    terminal_parser.add_argument("label")
    terminal_parser.add_argument("decision", type=Path)
    terminal_parser.add_argument("--execution-invalid", action="store_true")
    early_stage = subparsers.add_parser("stage-early-failure")
    early_stage.add_argument("mode")
    early_finalize = subparsers.add_parser("finalize-early-failure")
    early_finalize.add_argument("mode")
    subparsers.add_parser("stage-execution-invalid")
    subparsers.add_parser("finalize-execution-invalid")
    arguments = parser.parse_args(argv)
    if arguments.command == "execute":
        result = execute_version_forward()
    elif arguments.command == "verify-receipt":
        result = verify_transaction_receipt()
    elif arguments.command == "status":
        result = status()
    elif arguments.command == "advance":
        result = advance(
            arguments.completed_state,
            arguments.evidence,
            arguments.checkpoint_name,
            arguments.next_action,
        )
    elif arguments.command == "counts":
        parsed: dict[str, int | bool] = {}
        for assignment in arguments.assignments:
            if assignment.count("=") != 1:
                raise VersionForwardTransactionError(
                    f"invalid counter assignment: {assignment}"
                )
            key, raw = assignment.split("=", 1)
            parsed[key] = (
                raw.lower() == "true"
                if raw.lower() in ("true", "false")
                else int(raw)
            )
        result = update_counts(**parsed)
    elif arguments.command == "version-forward":
        result = version_forward(
            arguments.new_attempt, arguments.invalidity, arguments.equivalence
        )
    elif arguments.command == "terminal":
        result = record_terminal(
            arguments.label,
            arguments.decision,
            not arguments.execution_invalid,
        )
    elif arguments.command == "stage-early-failure":
        result = stage_early_scientific_failure(arguments.mode)
    elif arguments.command == "finalize-early-failure":
        result = finalize_early_scientific_failure(arguments.mode)
    elif arguments.command == "stage-execution-invalid":
        result = stage_postconfirmation_integrity_failure()
    else:
        result = finalize_postconfirmation_execution_invalid()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
