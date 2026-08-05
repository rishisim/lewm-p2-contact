#!/usr/bin/env python3
"""Fail-closed durable controller for the domain-robust gate study.

This controller owns only state transitions and evidence hashes. Scientific
programs write immutable evidence in the active versioned attempt and then ask
this controller to advance exactly one preregistered transition.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ROOT
REPO_ROOT = ROOT.parents[1]
STATE_PATH = ROOT / "STATE.json"
STATE_TRANSACTION_PATH = ROOT / "STATE_TRANSACTION.json"
LEDGER_PATH = ROOT / "RESEARCH_LEDGER.jsonl"
LEDGER_GENESIS_PATH = ROOT / "LEDGER_CHAIN_GENESIS.json"
CONTROLLER_LOCK_PATH = ROOT / ".program.lock"
_LOCK_LOCAL = threading.local()

STATE_MACHINE = (
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

TERMINAL_LABELS = (
    "domain_robust_gate_confirmed",
    "domain_robust_gate_partial",
    "domain_robust_gate_failed",
    "domain_robust_gate_execution_invalid",
)

EARLY_FAILURE_SPECS = {
    "no_candidate": {
        "trigger_state": "CANDIDATE_SELECTION",
        "source_relative": "selection/selection_ledger.json",
        "contract_relative": "verifier_contract_no_candidate.json",
        "contract_path_key": "selection_ledger",
        "audit_hash_key": "selection_ledger",
    },
    "power_infeasible": {
        "trigger_state": "CONFIRMATION_POWER_AND_COHORT_FREEZE",
        "source_relative": "power_analysis.json",
        "contract_relative": "verifier_contract_power_infeasible.json",
        "contract_path_key": "power_freeze",
        "audit_hash_key": "power_freeze",
    },
}
CONFIRMATION_VERIFIER_CONTRACT_RELATIVE = "verifier_contract.json"
EARLY_VERIFIER_AUDIT_RELATIVE = "audit/independent_verification.json"
EARLY_DECISION_RELATIVE = "decision.json"
EARLY_FORBIDDEN_ARTIFACT_ROOTS = (
    "data/smoke",
    "data/confirmation",
    "execution/smoke",
    "execution/confirmation",
    "metrics/smoke",
    "metrics/confirmation",
)
EARLY_FORBIDDEN_ARTIFACT_FILES = (
    "audit/pre_confirmation_manifest.json",
    "audit/pre_confirmation_package_seal.json",
    "audit/excluded_mechanical_smoke.json",
    "audit/confirmation_generation.json",
    "audit/confirmation_execution.json",
    "audit/confirmation_input_seal.json",
    "analysis_result.json",
    "bootstrap_replicates.npz",
    "bootstrap_summary.json",
)
POSTCONFIRMATION_INTEGRITY_SOURCE_RELATIVE = "metrics/latency_and_resources.json"
POSTCONFIRMATION_FAILURE_SPECS = {
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
        "source_relative": POSTCONFIRMATION_INTEGRITY_SOURCE_RELATIVE,
        "audit_hash_keys": (
            "latency_resource",
            "latency_and_resources",
            "latency_report",
        ),
    },
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(f"immutable artifact already exists: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@contextmanager
def _controller_lock() -> Iterator[None]:
    """Serialize reconciliation and whole read-modify-commit operations."""

    if getattr(_LOCK_LOCAL, "held", False):
        raise RuntimeError("controller lock is deliberately non-reentrant")
    CONTROLLER_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(CONTROLLER_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _LOCK_LOCAL.held = True
        yield
    finally:
        _LOCK_LOCAL.held = False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _require_controller_lock() -> None:
    if not getattr(_LOCK_LOCAL, "held", False):
        raise RuntimeError("controller mutation/reconciliation requires the global lock")


def _exclusive_operation(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def locked(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with _controller_lock():
            return function(*args, **kwargs)

    return locked


def verify_ledger_chain() -> dict[str, Any]:
    genesis = read_json(LEDGER_GENESIS_PATH)
    raw = LEDGER_PATH.read_bytes()
    prefix_bytes = int(genesis["legacy_prefix_bytes"])
    prefix = raw[:prefix_bytes]
    if len(prefix) != prefix_bytes:
        raise RuntimeError("research ledger is shorter than its immutable prefix")
    if hashlib.sha256(prefix).hexdigest() != genesis["legacy_prefix_sha256"]:
        raise RuntimeError("research ledger immutable prefix drift")
    lines = raw.splitlines()
    legacy_count = int(genesis["legacy_prefix_event_count"])
    if len(lines) < legacy_count:
        raise RuntimeError("research ledger lost a prefix event")
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError("research ledger contains a non-object record")
    previous = f"legacy:{genesis['legacy_prefix_sha256']}"
    for sequence, line in enumerate(lines[legacy_count:], start=legacy_count + 1):
        record = json.loads(line)
        observed_hash = record.pop("record_sha256", None)
        expected_hash = hashlib.sha256(canonical_bytes(record)).hexdigest()
        if observed_hash != expected_hash:
            raise RuntimeError(f"research ledger hash drift at sequence {sequence}")
        if record.get("seq") != sequence:
            raise RuntimeError(f"research ledger sequence drift at {sequence}")
        if record.get("prev_sha256") != previous:
            raise RuntimeError(f"research ledger predecessor drift at {sequence}")
        previous = observed_hash
    return {"event_count": len(lines), "head_sha256": previous}


def append_ledger(event: dict[str, Any]) -> dict[str, Any]:
    descriptor = os.open(LEDGER_PATH, os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        chain = verify_ledger_chain()
        payload = dict(event)
        payload.setdefault("created_unix_ns", time.time_ns())
        payload["seq"] = int(chain["event_count"]) + 1
        payload["prev_sha256"] = chain["head_sha256"]
        record_hash = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        payload["record_sha256"] = record_hash
        line = canonical_bytes(payload) + b"\n"
        written = os.write(descriptor, line)
        if written != len(line):
            raise RuntimeError("short append to research ledger")
        os.fsync(descriptor)
        verified = verify_ledger_chain()
        if verified["head_sha256"] != record_hash:
            raise RuntimeError("research ledger append did not verify")
        return verified
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _preview_ledger_record(
    event: dict[str, Any], sequence: int, previous: str
) -> dict[str, Any]:
    if "created_unix_ns" not in event:
        raise RuntimeError("transactional ledger event lacks a fixed timestamp")
    payload = dict(event)
    payload["seq"] = sequence
    payload["prev_sha256"] = previous
    payload["record_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def _pending_digest(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("transaction_sha256", None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _durable_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _commit_state_with_events(
    state: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    _require_controller_lock()
    if not events:
        raise ValueError("state transaction has no ledger events")
    if STATE_TRANSACTION_PATH.exists():
        raise RuntimeError("unreconciled state transaction already exists")
    base = verify_ledger_chain()
    if (
        state.get("ledger_event_count") != base["event_count"]
        or state.get("ledger_head_sha256") != base["head_sha256"]
    ):
        raise RuntimeError("state transaction base does not match the ledger")
    previews: list[dict[str, Any]] = []
    previous = base["head_sha256"]
    for offset, event in enumerate(events, start=1):
        preview = _preview_ledger_record(
            event, int(base["event_count"]) + offset, previous
        )
        previews.append(preview)
        previous = preview["record_sha256"]
    intended = json.loads(json.dumps(state))
    intended["ledger_event_count"] = int(base["event_count"]) + len(events)
    intended["ledger_head_sha256"] = previous
    pending = {
        "schema_version": 1,
        "base_event_count": base["event_count"],
        "base_head_sha256": base["head_sha256"],
        "events": events,
        "expected_records": previews,
        "intended_state": intended,
        "target_event_count": intended["ledger_event_count"],
        "target_head_sha256": intended["ledger_head_sha256"],
    }
    pending["transaction_sha256"] = _pending_digest(pending)
    atomic_json(STATE_TRANSACTION_PATH, pending, exclusive=True)
    chain = base
    for event, expected in zip(events, previews, strict=True):
        chain = append_ledger(event)
        if (
            chain["event_count"] != expected["seq"]
            or chain["head_sha256"] != expected["record_sha256"]
        ):
            raise RuntimeError("ledger append differs from durable transaction preview")
    atomic_json(STATE_PATH, intended)
    _durable_unlink(STATE_TRANSACTION_PATH)
    return intended


def _reconcile_state_transaction(
    state: dict[str, Any], chain: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_controller_lock()
    if not STATE_TRANSACTION_PATH.exists():
        return state, chain
    pending = read_json(STATE_TRANSACTION_PATH)
    if (
        pending.get("schema_version") != 1
        or pending.get("transaction_sha256") != _pending_digest(pending)
    ):
        raise RuntimeError("pending state transaction authentication failed")
    events = pending.get("events")
    previews = pending.get("expected_records")
    intended = pending.get("intended_state")
    if (
        not isinstance(events, list)
        or not isinstance(previews, list)
        or len(events) != len(previews)
        or not isinstance(intended, dict)
        or not events
    ):
        raise RuntimeError("pending state transaction schema is invalid")
    base_count = int(pending["base_event_count"])
    base_head = str(pending["base_head_sha256"])
    if int(pending["target_event_count"]) != base_count + len(events):
        raise RuntimeError("pending state transaction target count drift")
    previous = base_head
    for offset, (event, observed) in enumerate(zip(events, previews, strict=True), start=1):
        expected = _preview_ledger_record(event, base_count + offset, previous)
        if observed != expected:
            raise RuntimeError("pending state transaction record preview drift")
        previous = expected["record_sha256"]
    if previous != pending["target_head_sha256"]:
        raise RuntimeError("pending state transaction target head drift")
    state_is_base = (
        state.get("ledger_event_count") == base_count
        and state.get("ledger_head_sha256") == base_head
    )
    state_is_target = state == intended
    if not state_is_base and not state_is_target:
        raise RuntimeError("STATE.json is neither the transaction base nor target")
    progress = int(chain["event_count"]) - base_count
    if progress < 0 or progress > len(events):
        raise RuntimeError("ledger progress is outside the pending transaction")
    expected_progress_head = base_head if progress == 0 else previews[progress - 1]["record_sha256"]
    if chain["head_sha256"] != expected_progress_head:
        raise RuntimeError("ledger head does not match pending transaction progress")
    if state_is_target:
        if progress != len(events):
            raise RuntimeError("target state was installed before its ledger transaction")
    else:
        for event, expected in zip(events[progress:], previews[progress:], strict=True):
            chain = append_ledger(event)
            if (
                chain["event_count"] != expected["seq"]
                or chain["head_sha256"] != expected["record_sha256"]
            ):
                raise RuntimeError("pending transaction reconciliation append drift")
        atomic_json(STATE_PATH, intended)
        state = intended
        chain = verify_ledger_chain()
    _durable_unlink(STATE_TRANSACTION_PATH)
    return state, chain


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _attempt_root(state: dict[str, Any]) -> Path:
    return (REPO_ROOT / state["active_attempt_path"]).resolve()


def _attempt_history_paths(state: dict[str, Any]) -> dict[str, Path]:
    history = state.get("attempt_history")
    if not isinstance(history, list):
        raise RuntimeError("attempt_history is not a list")
    result: dict[str, Path] = {}
    for item in history:
        if not isinstance(item, dict):
            raise RuntimeError("attempt_history contains a non-object")
        version = item.get("version")
        path = item.get("path")
        if not isinstance(version, str) or not isinstance(path, str) or version in result:
            raise RuntimeError("attempt_history version/path lineage is invalid")
        resolved = (REPO_ROOT / path).resolve()
        if not resolved.is_relative_to(STUDY_ROOT.resolve() / "attempts"):
            raise RuntimeError("attempt_history path is outside the study attempt root")
        result[version] = resolved
    return result


def _checkpoint_source_attempt(
    state: dict[str, Any], checkpoint: dict[str, Any]
) -> str:
    paths = _attempt_history_paths(state)
    declared = checkpoint.get("source_attempt")
    if declared is not None:
        if declared not in paths:
            raise RuntimeError("checkpoint source_attempt is absent from attempt history")
        return str(declared)
    evidence = (REPO_ROOT / checkpoint["evidence_path"]).resolve()
    matches = [version for version, root in paths.items() if evidence.is_relative_to(root)]
    if len(matches) != 1:
        raise RuntimeError("legacy checkpoint source attempt cannot be inferred uniquely")
    return matches[0]


def _expected_early_completed_states(state: dict[str, Any]) -> list[str]:
    early = state["early_scientific_failure"]
    trigger_index = STATE_MACHINE.index(early["trigger_state"])
    result = list(STATE_MACHINE[:trigger_index])
    if early["status"] == "terminal_recorded":
        result.extend(("INDEPENDENT_VERIFICATION", "TERMINAL"))
        if state.get("post_terminal_reporting_complete") is True:
            result.append("POST_TERMINAL_REPORTING")
    return result


def verify_state_shape(state: dict[str, Any]) -> None:
    if tuple(state.get("state_machine", ())) != STATE_MACHINE:
        raise RuntimeError("state-machine definition drift")
    if state.get("current_state") not in STATE_MACHINE:
        raise RuntimeError("unknown current state")
    completed = state.get("completed_states")
    if not isinstance(completed, list):
        raise RuntimeError("completed_states is not a list")
    early = state.get("early_scientific_failure")
    integrity = state.get("postconfirmation_integrity_failure")
    if early is not None and integrity is not None:
        raise RuntimeError("early and post-confirmation failure branches cannot coexist")
    final_reporting_complete = state.get("post_terminal_reporting_complete") is True
    if early is None and integrity is None:
        current_index = STATE_MACHINE.index(state["current_state"])
        expected_completed = (
            list(STATE_MACHINE)
            if final_reporting_complete
            else list(STATE_MACHINE[:current_index])
        )
        if final_reporting_complete and state["current_state"] != "POST_TERMINAL_REPORTING":
            raise RuntimeError("completed final reporting has an invalid current state")
        if completed != expected_completed:
            raise RuntimeError("completed-state prefix is not exact")
        if state.get("skipped_states") not in (None, []):
            raise RuntimeError("ordinary path unexpectedly records skipped states")
    elif early is not None:
        if not isinstance(early, dict) or early.get("mode") not in EARLY_FAILURE_SPECS:
            raise RuntimeError("early scientific-failure record is invalid")
        spec = EARLY_FAILURE_SPECS[early["mode"]]
        if early.get("trigger_state") != spec["trigger_state"]:
            raise RuntimeError("early scientific-failure trigger state drift")
        trigger_index = STATE_MACHINE.index(spec["trigger_state"])
        independent_index = STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
        expected_skipped = list(STATE_MACHINE[trigger_index + 1 : independent_index])
        if early.get("skipped_states") != expected_skipped:
            raise RuntimeError("early scientific-failure skipped-state set drift")
        records = state.get("skipped_states")
        if not isinstance(records, list) or [item.get("state") for item in records] != expected_skipped:
            raise RuntimeError("skipped-state transparency records are incomplete")
        if any(
            not isinstance(item, dict)
            or item.get("mode") != early["mode"]
            or item.get("reason") != "preregistered_process_valid_early_scientific_failure"
            for item in records
        ):
            raise RuntimeError("skipped-state transparency record drift")
        if completed != _expected_early_completed_states(state):
            raise RuntimeError("early-path completed states are not exact")
        if early.get("status") == "awaiting_independent_verification":
            if state["current_state"] != "INDEPENDENT_VERIFICATION":
                raise RuntimeError("early path is not staged at independent verification")
            if state.get("terminal_label") is not None or state.get("process_valid") is not None:
                raise RuntimeError("unverified early failure was marked terminal")
        elif early.get("status") == "terminal_recorded":
            if state["current_state"] != "POST_TERMINAL_REPORTING":
                raise RuntimeError("verified early failure did not enter final reporting")
            if state.get("terminal_label") != "domain_robust_gate_failed":
                raise RuntimeError("early scientific failure has the wrong terminal label")
            if state.get("process_valid") is not True:
                raise RuntimeError("early scientific failure is not process-valid")
            if state.get("confirmation_terminal") is not False:
                raise RuntimeError("early scientific failure was mislabeled as confirmation")
        else:
            raise RuntimeError("unknown early scientific-failure status")
    else:
        if not isinstance(integrity, dict):
            raise RuntimeError("post-confirmation integrity record is invalid")
        trigger = integrity.get("trigger_state")
        if trigger not in {
            "SEALED_ANALYSIS",
            "LATENCY_AND_RESOURCE_REPORTING",
            "INDEPENDENT_VERIFICATION",
        }:
            raise RuntimeError("post-confirmation integrity trigger state is invalid")
        trigger_index = STATE_MACHINE.index(trigger)
        independent_index = STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
        expected_skipped = list(STATE_MACHINE[trigger_index + 1 : independent_index])
        if integrity.get("skipped_states") != expected_skipped:
            raise RuntimeError("post-confirmation skipped-state set drift")
        records = state.get("skipped_states")
        if (
            not isinstance(records, list)
            or [item.get("state") for item in records] != expected_skipped
            or any(
                not isinstance(item, dict)
                or item.get("source") != integrity.get("source")
                or item.get("reason")
                != "postconfirmation_integrity_failure_short_circuit"
                for item in records
            )
        ):
            raise RuntimeError("post-confirmation skipped-state transparency drift")
        expected_completed = list(STATE_MACHINE[: trigger_index + 1])
        if integrity.get("status") == "awaiting_independent_verification":
            if trigger == "INDEPENDENT_VERIFICATION":
                raise RuntimeError("verifier failure cannot be staged before it occurs")
            if state["current_state"] != "INDEPENDENT_VERIFICATION":
                raise RuntimeError("integrity failure is not staged at verification")
            if state.get("terminal_label") is not None:
                raise RuntimeError("unverified integrity failure was marked terminal")
        elif integrity.get("status") == "terminal_recorded":
            if trigger != "INDEPENDENT_VERIFICATION":
                expected_completed.append("INDEPENDENT_VERIFICATION")
            expected_completed.append("TERMINAL")
            if final_reporting_complete:
                expected_completed.append("POST_TERMINAL_REPORTING")
            if (
                state["current_state"] != "POST_TERMINAL_REPORTING"
                or state.get("terminal_label")
                != "domain_robust_gate_execution_invalid"
                or state.get("process_valid") is not False
                or state.get("confirmation_terminal") is not True
            ):
                raise RuntimeError("execution-invalid terminal state is inconsistent")
        else:
            raise RuntimeError("unknown post-confirmation integrity status")
        if completed != expected_completed:
            raise RuntimeError("post-confirmation failure checkpoints are not exact")
    if state.get("terminal_label") is not None and state["terminal_label"] not in TERMINAL_LABELS:
        raise RuntimeError("unknown terminal label")
    expected_attempt_path = f"runs/lewm_domain_robust_gate/attempts/{state['active_attempt']}"
    if state.get("active_attempt_path") != expected_attempt_path:
        raise RuntimeError("active attempt path drift")


def verify_checkpoints(state: dict[str, Any]) -> None:
    checkpoints = state.get("verified_checkpoints", [])
    if len(checkpoints) != len(state["completed_states"]):
        raise RuntimeError("verified checkpoint count does not match completed states")
    for checkpoint in checkpoints:
        path = REPO_ROOT / checkpoint["evidence_path"]
        if not path.is_file() or sha256_file(path) != checkpoint["evidence_sha256"]:
            raise RuntimeError(f"verified checkpoint drift: {path}")
        source_attempt = _checkpoint_source_attempt(state, checkpoint)
        source_root = _attempt_history_paths(state)[source_attempt]
        if not path.resolve().is_relative_to(source_root):
            raise RuntimeError("checkpoint evidence is outside its source-attempt lineage")
    expected_last = checkpoints[-1] if checkpoints else None
    if state.get("last_verified_checkpoint") != expected_last:
        raise RuntimeError("last verified checkpoint pointer drift")


def _assert_no_early_later_role_evidence(state: dict[str, Any]) -> None:
    expected_counts: dict[str, int | bool] = {
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    for key, expected in expected_counts.items():
        if state.get(key) != expected:
            raise RuntimeError(f"early scientific failure has nonzero later-role state: {key}")
    if state.get("fit_outcome_episodes") != state.get("expected_fit_episode_count"):
        raise RuntimeError("early scientific failure lacks the fixed complete fit cohort")
    if state.get("selection_outcome_episodes") != state.get(
        "expected_selection_episode_count"
    ):
        raise RuntimeError("early scientific failure lacks the fixed complete selection cohort")
    attempt_root = _attempt_root(state)
    for relative_root in EARLY_FORBIDDEN_ARTIFACT_ROOTS:
        root = attempt_root / relative_root
        if root.exists() and any(path.is_file() for path in root.rglob("*")):
            raise RuntimeError(f"early scientific failure has forbidden later-role artifacts: {root}")
    for relative_file in EARLY_FORBIDDEN_ARTIFACT_FILES:
        path = attempt_root / relative_file
        if path.exists():
            raise RuntimeError(f"early scientific failure has forbidden artifact: {path}")


def _verify_terminal_links(state: dict[str, Any]) -> None:
    decision_path = state.get("terminal_decision_path")
    decision_sha256 = state.get("terminal_decision_sha256")
    if decision_path is None and decision_sha256 is None:
        return
    if not isinstance(decision_path, str) or not isinstance(decision_sha256, str):
        raise RuntimeError("terminal decision link is incomplete")
    path = REPO_ROOT / decision_path
    if not path.is_file() or sha256_file(path) != decision_sha256:
        raise RuntimeError("immutable terminal decision drift")


def _status_locked() -> dict[str, Any]:
    _require_controller_lock()
    state = read_json(STATE_PATH)
    chain = verify_ledger_chain()
    state, chain = _reconcile_state_transaction(state, chain)
    verify_state_shape(state)
    verify_checkpoints(state)
    _verify_terminal_links(state)
    if state.get("early_scientific_failure") is not None:
        _assert_no_early_later_role_evidence(state)
    if state.get("ledger_event_count") != chain["event_count"]:
        raise RuntimeError("state/ledger event-count mismatch")
    if state.get("ledger_head_sha256") != chain["head_sha256"]:
        raise RuntimeError("state/ledger head mismatch")
    return state


def status() -> dict[str, Any]:
    with _controller_lock():
        return _status_locked()


@_exclusive_operation
def advance(completed_state: str, evidence: Path, checkpoint_name: str, next_action: str) -> dict[str, Any]:
    state = _status_locked()
    early = state.get("early_scientific_failure")
    if early is not None and completed_state != "POST_TERMINAL_REPORTING":
        raise RuntimeError("early scientific-failure path requires its narrow controller command")
    if (
        state.get("postconfirmation_integrity_failure") is not None
        and completed_state != "POST_TERMINAL_REPORTING"
    ):
        raise RuntimeError(
            "post-confirmation integrity failure requires its narrow controller command"
        )
    if state.get("terminal_label") is not None and completed_state not in {
        "TERMINAL",
        "POST_TERMINAL_REPORTING",
    }:
        raise RuntimeError("scientific terminal decision is immutable")
    if state["current_state"] != completed_state:
        raise RuntimeError(
            f"out-of-order transition: current={state['current_state']} requested={completed_state}"
        )
    if not evidence.is_file():
        raise FileNotFoundError(evidence)
    attempt_root = (REPO_ROOT / state["active_attempt_path"]).resolve()
    if not evidence.resolve().is_relative_to(attempt_root):
        raise RuntimeError("checkpoint evidence is outside the active attempt")
    evidence_object = read_json(evidence)
    if evidence_object.get("passed") is not True:
        raise RuntimeError("checkpoint evidence does not pass")
    if evidence_object.get("attempt") != state["active_attempt"]:
        raise RuntimeError("checkpoint evidence attempt mismatch")
    if evidence_object.get("checkpoint_state") != completed_state:
        raise RuntimeError("checkpoint evidence state mismatch")
    if (
        completed_state == "INDEPENDENT_VERIFICATION"
        and evidence_object.get("terminal_label")
        == "domain_robust_gate_execution_invalid"
    ):
        raise RuntimeError(
            "execution-invalid verifier evidence requires its narrow finalizer"
        )
    index = STATE_MACHINE.index(completed_state)
    final_transition = completed_state == "POST_TERMINAL_REPORTING"
    if final_transition:
        if state.get("terminal_label") is None or state.get("process_valid") is None:
            raise RuntimeError("post-terminal reporting cannot complete before a terminal decision")
        if state.get("post_terminal_reporting_complete") is True:
            raise RuntimeError("post-terminal reporting is already complete")
        next_state = None
    else:
        next_state = STATE_MACHINE[index + 1]
    created = time.time_ns()
    checkpoint = {
        "name": checkpoint_name,
        "created_unix_ns": created,
        "evidence_path": relative(evidence),
        "evidence_sha256": sha256_file(evidence),
        "source_attempt": state["active_attempt"],
        "verification_lineage": "direct_checkpoint",
    }
    state["completed_states"].append(completed_state)
    state["current_state"] = (
        "POST_TERMINAL_REPORTING" if final_transition else next_state
    )
    state["last_verified_checkpoint"] = checkpoint
    state["verified_checkpoints"].append(checkpoint)
    state["next_action"] = (
        "program_complete_no_further_scientific_or_reporting_transition"
        if final_transition
        else next_action
    )
    if final_transition:
        state["post_terminal_reporting_complete"] = True
    if completed_state == "PREREGISTRATION_AND_POWER":
        registered = evidence_object.get("registered_counts", {})
        expected_registered = {
            "expected_fit_episode_count": 1200,
            "expected_selection_episode_count": 2000,
            "expected_smoke_episode_count": 24,
            "maximum_confirmation_episode_count": 18000,
        }
        if registered != expected_registered:
            raise RuntimeError("preregistered role counts do not match the fixed design")
        state.update(registered)
    if completed_state == "CONFIRMATION_POWER_AND_COHORT_FREEZE":
        fixed_total = evidence_object.get("fixed_confirmation_episode_count")
        fixed_per_regime = evidence_object.get("fixed_confirmation_episodes_per_regime")
        if not isinstance(fixed_per_regime, int) or fixed_per_regime not in range(500, 4501, 500):
            raise RuntimeError("fixed confirmation size is outside the preregistered grid")
        if fixed_total != 4 * fixed_per_regime:
            raise RuntimeError("fixed confirmation total is inconsistent across DGPs")
        state["expected_confirmation_episodes_per_regime"] = fixed_per_regime
        state["expected_confirmation_episode_count"] = fixed_total
    state["updated_unix_ns"] = created
    return _commit_state_with_events(
        state,
        [
            {
                "event": "state_completed",
                "attempt": state["active_attempt"],
                "completed_state": completed_state,
                "next_state": next_state,
                "checkpoint_name": checkpoint_name,
                "evidence_path": checkpoint["evidence_path"],
                "evidence_sha256": checkpoint["evidence_sha256"],
                "created_unix_ns": created,
            }
        ],
    )


@_exclusive_operation
def update_counts(**counts: int | bool) -> dict[str, Any]:
    allowed = {
        "fit_outcome_episodes",
        "selection_outcome_episodes",
        "smoke_outcome_episodes",
        "confirmation_outcome_episodes_generated",
        "confirmation_outcome_episodes_executed",
        "confirmation_outcomes_opened_for_analysis",
    }
    if not counts or not set(counts).issubset(allowed):
        raise ValueError("unsupported outcome counter")
    state = _status_locked()
    if state.get("early_scientific_failure") is not None:
        raise RuntimeError("outcome counters are immutable on an early scientific-failure path")
    if state.get("terminal_label") is not None:
        raise RuntimeError("outcome counters are immutable after a terminal decision")
    allowed_states = {
        "fit_outcome_episodes": {"FIT_COHORTS"},
        "selection_outcome_episodes": {"SELECTION_COHORTS"},
        "smoke_outcome_episodes": {"EXCLUDED_MECHANICAL_SMOKE"},
        "confirmation_outcome_episodes_generated": {"CONFIRMATION_GENERATION", "CONFIRMATION_EXECUTION"},
        "confirmation_outcome_episodes_executed": {"CONFIRMATION_EXECUTION", "CONFIRMATION_INPUT_SEAL"},
        "confirmation_outcomes_opened_for_analysis": {"SEALED_ANALYSIS"},
    }
    for key, value in counts.items():
        if state["current_state"] not in allowed_states[key]:
            raise RuntimeError(f"counter {key} cannot change in state {state['current_state']}")
        old = state.get(key, False if isinstance(value, bool) else 0)
        if isinstance(value, bool):
            if old is True and value is False:
                raise RuntimeError(f"boolean outcome state cannot move backward: {key}")
        elif int(value) < int(old):
            raise RuntimeError(f"outcome counter cannot decrease: {key}")
        expected_key = {
            "fit_outcome_episodes": "expected_fit_episode_count",
            "selection_outcome_episodes": "expected_selection_episode_count",
            "smoke_outcome_episodes": "expected_smoke_episode_count",
            "confirmation_outcome_episodes_generated": "expected_confirmation_episode_count",
            "confirmation_outcome_episodes_executed": "expected_confirmation_episode_count",
        }.get(key)
        if expected_key and expected_key in state and int(value) > int(state[expected_key]):
            raise RuntimeError(f"counter exceeds preregistered size: {key}")
        state[key] = value
    state["updated_unix_ns"] = time.time_ns()
    return _commit_state_with_events(
        state,
        [
            {
                "event": "outcome_counts_updated",
                "attempt": state["active_attempt"],
                "fields": counts,
                "created_unix_ns": state["updated_unix_ns"],
            }
        ],
    )


def _exact_active_attempt_file(state: dict[str, Any], relative_path: str) -> Path:
    path = (_attempt_root(state) / relative_path).resolve()
    if not path.is_relative_to(_attempt_root(state)):
        raise RuntimeError("early scientific-failure path escapes the active attempt")
    return path


def _resolved_contract_path(raw: Any) -> Path:
    if not isinstance(raw, str):
        raise RuntimeError("verifier contract path is absent")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("verifier contract contains a noncanonical path")
    return (REPO_ROOT / candidate).resolve()


def _validate_early_source(mode: str, source: dict[str, Any]) -> None:
    if mode == "no_candidate":
        checks = {
            "selection_status": source.get("status")
            == "selection_complete_no_selected_head_refit",
            "candidate_count": source.get("candidate_count") == 24,
            "eligible_count_zero": source.get("eligible_count") == 0,
            "selected_id_null": source.get("selected_candidate_id") is None,
            "selected_index_null": source.get("selected_candidate_index") is None,
            "no_selected_refit": source.get("selected_head_refit_after_selection") is False,
            "confirmation_unused": source.get("prior_confirmation_outcome_episodes_used", 0)
            == 0,
        }
    else:
        checks = {
            "power_status": source.get("status")
            == "binding_post_selection_power_result",
            "decision": source.get("decision") == "power_infeasible_no_confirmation",
            "feasible_false": source.get("feasible") is False,
            "passed_false": source.get("passed") is False,
            "generation_unauthorized": source.get(
                "confirmation_generation_authorized_by_power"
            )
            is False,
            "selected_n_null": source.get(
                "selected_confirmation_episodes_per_regime"
            )
            is None,
            "selected_counts_null": source.get("confirmation_episode_count_per_regime")
            is None,
            "terminal_mapping": source.get("terminal_label_if_infeasible")
            == "domain_robust_gate_failed",
            "confirmation_unopened": source.get("fresh_confirmation_outcomes_opened")
            == 0,
        }
    if not all(checks.values()):
        raise RuntimeError(
            f"{mode} source does not prove the preregistered scientific failure: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )


def _require_presealed_verifier_contract(
    state: dict[str, Any], contract_path: Path
) -> None:
    """Prove the mode-specific contract was hash-sealed before fit outcomes."""

    if "PRE_OUTCOME_SEAL" not in state["completed_states"]:
        raise RuntimeError("verifier contract was not sealed before fresh outcomes")
    seal_path = _exact_active_attempt_file(state, "audit/pre_data_seal.json")
    if not seal_path.is_file():
        raise FileNotFoundError(seal_path)
    completed_index = state["completed_states"].index("PRE_OUTCOME_SEAL")
    checkpoint = state["verified_checkpoints"][completed_index]
    if (
        (REPO_ROOT / checkpoint["evidence_path"]).resolve() != seal_path
        or checkpoint["evidence_sha256"] != sha256_file(seal_path)
    ):
        raise RuntimeError("PRE_OUTCOME_SEAL checkpoint does not bind pre_data_seal.json")
    seal = read_json(seal_path)
    sealed_files = seal.get("sealed_files")
    if (
        seal.get("passed") is not True
        or seal.get("attempt") != state["active_attempt"]
        or seal.get("checkpoint_state") != "PRE_OUTCOME_SEAL"
        or not isinstance(sealed_files, dict)
        or sealed_files.get(relative(contract_path)) != sha256_file(contract_path)
    ):
        raise RuntimeError(
            "mode-specific verifier contract is absent from the pre-outcome seal"
        )


@_exclusive_operation
def stage_early_scientific_failure(mode: str) -> dict[str, Any]:
    if mode not in EARLY_FAILURE_SPECS:
        raise ValueError("early scientific-failure mode is not preregistered")
    state = _status_locked()
    if state.get("early_scientific_failure") is not None:
        raise RuntimeError("an early scientific-failure path is already staged")
    if state.get("terminal_label") is not None:
        raise RuntimeError("scientific terminal decision is immutable")
    spec = EARLY_FAILURE_SPECS[mode]
    if state["current_state"] != spec["trigger_state"]:
        raise RuntimeError(
            f"{mode} may only be staged from {spec['trigger_state']}"
        )
    _assert_no_early_later_role_evidence(state)
    source_path = _exact_active_attempt_file(state, spec["source_relative"])
    contract_path = _exact_active_attempt_file(state, spec["contract_relative"])
    if not source_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError(source_path if not source_path.is_file() else contract_path)
    source = read_json(source_path)
    _validate_early_source(mode, source)
    _require_presealed_verifier_contract(state, contract_path)
    contract = read_json(contract_path)
    if (
        contract.get("schema_version") != 1
        or contract.get("attempt") != state["active_attempt"]
        or contract.get("mode") != mode
        or contract.get("attempt_root") != state["active_attempt_path"]
    ):
        raise RuntimeError("early verifier contract identity or mode mismatch")
    paths = contract.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError("early verifier contract path map is absent")
    contracted_source = _resolved_contract_path(paths.get(spec["contract_path_key"]))
    if contracted_source != source_path:
        raise RuntimeError("early verifier contract is not bound to the exact source")

    created = time.time_ns()
    trigger_index = STATE_MACHINE.index(spec["trigger_state"])
    independent_index = STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
    skipped = list(STATE_MACHINE[trigger_index + 1 : independent_index])
    source_relative = relative(source_path)
    source_sha256 = sha256_file(source_path)
    state["early_scientific_failure"] = {
        "mode": mode,
        "status": "awaiting_independent_verification",
        "trigger_state": spec["trigger_state"],
        "trigger_evidence_path": source_relative,
        "trigger_evidence_sha256": source_sha256,
        "verifier_contract_path": relative(contract_path),
        "verifier_contract_sha256": sha256_file(contract_path),
        "skipped_states": skipped,
        "staged_unix_ns": created,
        "required_terminal_label": "domain_robust_gate_failed",
        "required_process_valid": True,
    }
    state["skipped_states"] = [
        {
            "state": skipped_state,
            "mode": mode,
            "reason": "preregistered_process_valid_early_scientific_failure",
            "trigger_evidence_path": source_relative,
            "trigger_evidence_sha256": source_sha256,
            "recorded_unix_ns": created,
        }
        for skipped_state in skipped
    ]
    state["current_state"] = "INDEPENDENT_VERIFICATION"
    state["next_action"] = (
        "run the standalone read-only verifier in the exact staged mode and "
        "capture audit/independent_verification.json before finalizing"
    )
    state["updated_unix_ns"] = created
    return _commit_state_with_events(
        state,
        [
            {
                "event": "preregistered_early_scientific_failure_staged",
                "attempt": state["active_attempt"],
                "mode": mode,
                "trigger_state": spec["trigger_state"],
                "trigger_evidence_path": source_relative,
                "trigger_evidence_sha256": source_sha256,
                "verifier_contract_path": relative(contract_path),
                "verifier_contract_sha256": sha256_file(contract_path),
                "skipped_states": skipped,
                "smoke_outcome_episodes": 0,
                "confirmation_outcome_episodes_generated": 0,
                "confirmation_outcome_episodes_executed": 0,
                "confirmation_outcomes_opened_for_analysis": False,
                "created_unix_ns": created,
            }
        ],
    )


def _audit_hash_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("sha256"), str):
        return str(value["sha256"])
    return None


def _validate_early_audit(
    state: dict[str, Any], mode: str, audit_path: Path
) -> dict[str, Any]:
    early = state["early_scientific_failure"]
    audit = read_json(audit_path)
    required = {
        "schema_version": audit.get("schema_version") == 1,
        "attempt": audit.get("attempt") == state["active_attempt"],
        "mode": audit.get("mode") == mode,
        "passed": audit.get("passed") is True,
        "terminal_label": audit.get("terminal_label")
        == "domain_robust_gate_failed",
        "read_only": audit.get("read_only_verifier") is True,
        # Real verifier checks deliberately contain hashes, counts, summaries,
        # and recomputed records rather than a boolean-only tree.  ``passed``
        # plus the explicit immutable links below is the fail-closed contract.
        "checks_record": isinstance(audit.get("checks"), dict)
        and bool(audit.get("checks")),
    }
    contract_record = audit.get("verifier_contract")
    required["contract_record"] = isinstance(contract_record, dict)
    if isinstance(contract_record, dict):
        required["contract_path"] = contract_record.get("path") == early[
            "verifier_contract_path"
        ]
        required["contract_hash"] = contract_record.get("sha256") == early[
            "verifier_contract_sha256"
        ]
    source_hashes = audit.get("source_hashes")
    spec = EARLY_FAILURE_SPECS[mode]
    required["source_hashes"] = isinstance(source_hashes, dict)
    if isinstance(source_hashes, dict):
        required["source_hash"] = _audit_hash_value(
            source_hashes.get(spec["audit_hash_key"])
        ) == early["trigger_evidence_sha256"]
    if not all(required.values()):
        raise RuntimeError(
            "independent early-failure audit mismatch: "
            f"{[name for name, passed in required.items() if not passed]}"
        )
    return audit


def _validate_early_decision(
    state: dict[str, Any], mode: str, decision_path: Path, audit_path: Path
) -> dict[str, Any]:
    early = state["early_scientific_failure"]
    decision = read_json(decision_path)
    expected = {
        "schema": decision.get("schema_version") == 1,
        "attempt": decision.get("attempt") == state["active_attempt"],
        "passed": decision.get("passed") is True,
        "checkpoint": decision.get("checkpoint_state") == "TERMINAL",
        "mode": decision.get("early_failure_mode") == mode,
        "label": decision.get("terminal_label") == "domain_robust_gate_failed",
        "process_valid": decision.get("process_valid") is True,
        "trigger_path": decision.get("trigger_evidence_path")
        == early["trigger_evidence_path"],
        "trigger_hash": decision.get("trigger_evidence_sha256")
        == early["trigger_evidence_sha256"],
        "audit_path": decision.get("independent_verification_path")
        == relative(audit_path),
        "audit_hash": decision.get("independent_verification_sha256")
        == sha256_file(audit_path),
        "smoke_zero": decision.get("smoke_outcome_episodes") == 0,
        "confirmation_generated_zero": decision.get(
            "confirmation_outcome_episodes_generated"
        )
        == 0,
        "confirmation_executed_zero": decision.get(
            "confirmation_outcome_episodes_executed"
        )
        == 0,
        "confirmation_unopened": decision.get(
            "confirmation_outcomes_opened_for_analysis"
        )
        is False,
    }
    if not all(expected.values()):
        raise RuntimeError(
            "immutable early-failure decision mismatch: "
            f"{[name for name, passed in expected.items() if not passed]}"
        )
    return decision


@_exclusive_operation
def finalize_early_scientific_failure(mode: str) -> dict[str, Any]:
    if mode not in EARLY_FAILURE_SPECS:
        raise ValueError("early scientific-failure mode is not preregistered")
    state = _status_locked()
    early = state.get("early_scientific_failure")
    if not isinstance(early, dict) or early.get("mode") != mode:
        raise RuntimeError("no matching staged early scientific failure exists")
    if early.get("status") != "awaiting_independent_verification":
        raise RuntimeError("early scientific-failure decision is immutable")
    if state["current_state"] != "INDEPENDENT_VERIFICATION":
        raise RuntimeError("early failure is not awaiting independent verification")
    audit_path = _exact_active_attempt_file(state, EARLY_VERIFIER_AUDIT_RELATIVE)
    decision_path = _exact_active_attempt_file(state, EARLY_DECISION_RELATIVE)
    if not audit_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError(audit_path if not audit_path.is_file() else decision_path)
    _validate_early_audit(state, mode, audit_path)
    _validate_early_decision(state, mode, decision_path, audit_path)
    _assert_no_early_later_role_evidence(state)

    created = time.time_ns()
    audit_checkpoint = {
        "name": f"{state['active_attempt']}_{mode}_independent_verification_passed",
        "created_unix_ns": created,
        "evidence_path": relative(audit_path),
        "evidence_sha256": sha256_file(audit_path),
        "source_attempt": state["active_attempt"],
        "verification_lineage": "direct_early_scientific_failure_checkpoint",
    }
    decision_checkpoint = {
        "name": f"{state['active_attempt']}_{mode}_terminal_decision_recorded",
        "created_unix_ns": created,
        "evidence_path": relative(decision_path),
        "evidence_sha256": sha256_file(decision_path),
        "source_attempt": state["active_attempt"],
        "verification_lineage": "direct_early_scientific_failure_checkpoint",
    }
    state["completed_states"].append("INDEPENDENT_VERIFICATION")
    state["verified_checkpoints"].append(audit_checkpoint)
    state["completed_states"].append("TERMINAL")
    state["verified_checkpoints"].append(decision_checkpoint)
    state["last_verified_checkpoint"] = decision_checkpoint
    state["current_state"] = "POST_TERMINAL_REPORTING"
    state["terminal_label"] = "domain_robust_gate_failed"
    state["process_valid"] = True
    state["scientific_terminal"] = True
    state["confirmation_terminal"] = False
    state["terminal_basis"] = f"preregistered_{mode}"
    state["terminal_decision_path"] = relative(decision_path)
    state["terminal_decision_sha256"] = sha256_file(decision_path)
    state["early_scientific_failure"].update(
        {
            "status": "terminal_recorded",
            "independent_verification_path": relative(audit_path),
            "independent_verification_sha256": sha256_file(audit_path),
            "decision_path": relative(decision_path),
            "decision_sha256": sha256_file(decision_path),
            "terminal_recorded_unix_ns": created,
        }
    )
    state["next_action"] = (
        "write the concise terminal report, robustness map, audit, limitations, "
        "and next project-scoped task; then checkpoint POST_TERMINAL_REPORTING"
    )
    state["updated_unix_ns"] = created
    events = [
        {
            "event": "state_completed",
            "attempt": state["active_attempt"],
            "completed_state": "INDEPENDENT_VERIFICATION",
            "next_state": "TERMINAL",
            "checkpoint_name": audit_checkpoint["name"],
            "evidence_path": audit_checkpoint["evidence_path"],
            "evidence_sha256": audit_checkpoint["evidence_sha256"],
            "early_failure_mode": mode,
            "created_unix_ns": created,
        },
        {
            "event": "scientific_terminal_decision_recorded",
            "attempt": state["active_attempt"],
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
            "confirmation_terminal": False,
            "early_failure_mode": mode,
            "decision_path": decision_checkpoint["evidence_path"],
            "decision_sha256": decision_checkpoint["evidence_sha256"],
            "created_unix_ns": created,
        },
        {
            "event": "state_completed",
            "attempt": state["active_attempt"],
            "completed_state": "TERMINAL",
            "next_state": "POST_TERMINAL_REPORTING",
            "checkpoint_name": decision_checkpoint["name"],
            "evidence_path": decision_checkpoint["evidence_path"],
            "evidence_sha256": decision_checkpoint["evidence_sha256"],
            "early_failure_mode": mode,
            "skipped_states": early["skipped_states"],
            "created_unix_ns": created,
        },
    ]
    return _commit_state_with_events(state, events)


def _assert_fixed_confirmation_opened(state: dict[str, Any]) -> None:
    expected = state.get("expected_confirmation_episode_count")
    if not isinstance(expected, int) or expected <= 0:
        raise RuntimeError("fixed confirmation size is absent")
    required = {
        "smoke": state.get("smoke_outcome_episodes")
        == state.get("expected_smoke_episode_count"),
        "generated": state.get("confirmation_outcome_episodes_generated") == expected,
        "executed": state.get("confirmation_outcome_episodes_executed") == expected,
        "opened": state.get("confirmation_outcomes_opened_for_analysis") is True,
    }
    if not all(required.values()):
        raise RuntimeError(
            "execution-invalid branch requires the complete fixed confirmation: "
            f"{[name for name, passed in required.items() if not passed]}"
        )


def _object_proves_integrity_failure(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {
                "execution_invalid",
                "integrity_failure",
                "integrity_failed",
            } and child is True:
                return True
            if lowered in {
                "integrity_valid",
                "integrity_passed",
                "process_valid",
                "all_integrity_checks_passed",
            } and child is False:
                return True
            if _object_proves_integrity_failure(child):
                return True
    elif isinstance(value, list):
        return any(_object_proves_integrity_failure(item) for item in value)
    return False


@_exclusive_operation
def stage_postconfirmation_integrity_failure() -> dict[str, Any]:
    state = _status_locked()
    trigger_state = state["current_state"]
    spec = POSTCONFIRMATION_FAILURE_SPECS.get(trigger_state)
    if spec is None:
        raise RuntimeError(
            "post-confirmation integrity failure may only be staged from "
            "SEALED_ANALYSIS or LATENCY_AND_RESOURCE_REPORTING"
        )
    if state.get("terminal_label") is not None:
        raise RuntimeError("scientific terminal decision is immutable")
    _assert_fixed_confirmation_opened(state)
    source_path = _exact_active_attempt_file(state, spec["source_relative"])
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = read_json(source_path)
    if (
        source.get("attempt") != state["active_attempt"]
        or source.get("checkpoint_state") != trigger_state
        or source.get("passed") is not False
        or not _object_proves_integrity_failure(source)
    ):
        raise RuntimeError("source does not prove a post-confirmation integrity failure")
    if trigger_state == "SEALED_ANALYSIS":
        expected = state["expected_confirmation_episode_count"]
        analysis_result_path = _exact_active_attempt_file(state, "analysis_result.json")
        analysis_checks = {
            "generated_fixed": source.get("confirmation_outcome_episodes_generated")
            == expected,
            "executed_fixed": source.get("confirmation_outcome_episodes_executed")
            == expected,
            "opened": source.get("confirmation_outcomes_opened_for_analysis") is True,
            "state_opened": state.get("confirmation_outcomes_opened_for_analysis")
            is True,
            "result_presence_recorded": source.get("analysis_result_present")
            is analysis_result_path.is_file(),
            "error_type": isinstance(source.get("error_type"), str)
            and bool(source.get("error_type")),
            "error": isinstance(source.get("error"), str)
            and bool(source.get("error")),
            "science_unchanged": source.get("scientific_objects_changed") is False,
        }
        if not all(analysis_checks.values()):
            raise RuntimeError(
                "analysis execution-invalid artifact is incomplete: "
                f"{[name for name, passed in analysis_checks.items() if not passed]}"
            )
    created = time.time_ns()
    checkpoint = {
        "name": f"{state['active_attempt']}_{spec['source']}_integrity_failure",
        "created_unix_ns": created,
        "evidence_path": relative(source_path),
        "evidence_sha256": sha256_file(source_path),
        "source_attempt": state["active_attempt"],
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    state["completed_states"].append(trigger_state)
    state["verified_checkpoints"].append(checkpoint)
    state["last_verified_checkpoint"] = checkpoint
    trigger_index = STATE_MACHINE.index(trigger_state)
    independent_index = STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
    skipped = list(STATE_MACHINE[trigger_index + 1 : independent_index])
    state["current_state"] = "INDEPENDENT_VERIFICATION"
    state["postconfirmation_integrity_failure"] = {
        "status": "awaiting_independent_verification",
        "source": spec["source"],
        "trigger_state": trigger_state,
        "source_path": relative(source_path),
        "source_sha256": sha256_file(source_path),
        "audit_hash_keys": list(spec["audit_hash_keys"]),
        "skipped_states": skipped,
        "staged_unix_ns": created,
    }
    state["skipped_states"] = [
        {
            "state": skipped_state,
            "source": spec["source"],
            "reason": "postconfirmation_integrity_failure_short_circuit",
            "source_path": relative(source_path),
            "source_sha256": sha256_file(source_path),
            "recorded_unix_ns": created,
        }
        for skipped_state in skipped
    ]
    state["next_action"] = (
        "run the read-only confirmation-mode independent verifier and capture the "
        "execution-invalid audit; do not retry or alter confirmation evidence"
    )
    state["updated_unix_ns"] = created
    return _commit_state_with_events(
        state,
        [
            {
                "event": "state_completed",
                "attempt": state["active_attempt"],
                "completed_state": trigger_state,
                "next_state": "INDEPENDENT_VERIFICATION",
                "checkpoint_name": checkpoint["name"],
                "evidence_path": checkpoint["evidence_path"],
                "evidence_sha256": checkpoint["evidence_sha256"],
                "postconfirmation_integrity_failure": True,
                "integrity_source": spec["source"],
                "skipped_states": skipped,
                "created_unix_ns": created,
            }
        ],
    )


def _validate_execution_invalid_audit(
    state: dict[str, Any], audit_path: Path
) -> dict[str, Any]:
    audit = read_json(audit_path)
    contract_path = _exact_active_attempt_file(
        state, CONFIRMATION_VERIFIER_CONTRACT_RELATIVE
    )
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = read_json(contract_path)
    contract_record = audit.get("verifier_contract")
    capture = audit.get("capture")
    required = {
        "schema": audit.get("schema_version") == 1,
        "attempt": audit.get("attempt") == state["active_attempt"],
        "mode": audit.get("mode") == "confirmation",
        "read_only": audit.get("read_only_verifier") is True,
        "contract_schema": contract.get("schema_version") == 1,
        "contract_attempt": contract.get("attempt") == state["active_attempt"],
        "contract_attempt_root": contract.get("attempt_root")
        == state["active_attempt_path"],
        "contract_mode": contract.get("mode") == "confirmation",
        "contract_record": isinstance(contract_record, dict),
        "capture_record": isinstance(capture, dict),
    }
    if isinstance(contract_record, dict):
        required["contract_path"] = contract_record.get("path") == relative(
            contract_path
        )
        required["contract_hash"] = contract_record.get("sha256") == sha256_file(
            contract_path
        )
    if isinstance(capture, dict):
        required["captured_exclusively"] = capture.get("captured_exclusively") is True
        required["wrapper_integrity"] = capture.get("wrapper_integrity_passed") is True
    staged = state.get("postconfirmation_integrity_failure")
    if isinstance(staged, dict):
        required["verifier_passed"] = audit.get("passed") is True
        required["label"] = (
            audit.get("terminal_label")
            == "domain_robust_gate_execution_invalid"
        )
        required["integrity_failure"] = _object_proves_integrity_failure(audit)
        hashes = audit.get("source_hashes")
        candidates = staged.get("audit_hash_keys")
        required["staged_source_hash"] = (
            isinstance(hashes, dict)
            and isinstance(candidates, list)
            and bool(candidates)
            and any(
                _audit_hash_value(hashes.get(name)) == staged["source_sha256"]
                for name in candidates
            )
        )
    else:
        required["verifier_itself_failed"] = audit.get("passed") is False
        required["capture_records_failure"] = (
            isinstance(capture, dict)
            and capture.get("scientific_verifier_passed") is False
            and isinstance(capture.get("verifier_returncode"), int)
            and int(capture["verifier_returncode"]) != 0
        )
        required["failure_recorded"] = (
            isinstance(audit.get("error_type"), str)
            and bool(audit.get("error_type"))
            and isinstance(audit.get("error"), str)
            and bool(audit.get("error"))
        )
    if not all(required.values()):
        raise RuntimeError(
            "execution-invalid independent audit mismatch: "
            f"{[name for name, passed in required.items() if not passed]}"
        )
    return audit


@_exclusive_operation
def finalize_postconfirmation_execution_invalid() -> dict[str, Any]:
    state = _status_locked()
    if state["current_state"] != "INDEPENDENT_VERIFICATION":
        raise RuntimeError("execution-invalid finalization requires independent verification")
    if state.get("terminal_label") is not None:
        raise RuntimeError("scientific terminal decision is immutable")
    _assert_fixed_confirmation_opened(state)
    audit_path = _exact_active_attempt_file(state, EARLY_VERIFIER_AUDIT_RELATIVE)
    decision_path = _exact_active_attempt_file(state, EARLY_DECISION_RELATIVE)
    if not audit_path.is_file() or not decision_path.is_file():
        raise FileNotFoundError(audit_path if not audit_path.is_file() else decision_path)
    _validate_execution_invalid_audit(state, audit_path)
    decision = read_json(decision_path)
    expected_confirmation = state["expected_confirmation_episode_count"]
    decision_checks = {
        "schema": decision.get("schema_version") == 1,
        "attempt": decision.get("attempt") == state["active_attempt"],
        "passed": decision.get("passed") is True,
        "checkpoint": decision.get("checkpoint_state") == "TERMINAL",
        "basis": decision.get("terminal_basis")
        == "postconfirmation_integrity_failure",
        "label": decision.get("terminal_label")
        == "domain_robust_gate_execution_invalid",
        "process_invalid": decision.get("process_valid") is False,
        "audit_path": decision.get("independent_verification_path")
        == relative(audit_path),
        "audit_hash": decision.get("independent_verification_sha256")
        == sha256_file(audit_path),
        "generated_fixed": decision.get("confirmation_outcome_episodes_generated")
        == expected_confirmation,
        "executed_fixed": decision.get("confirmation_outcome_episodes_executed")
        == expected_confirmation,
        "opened": decision.get("confirmation_outcomes_opened_for_analysis") is True,
    }
    if not all(decision_checks.values()):
        raise RuntimeError(
            "execution-invalid terminal decision mismatch: "
            f"{[name for name, passed in decision_checks.items() if not passed]}"
        )
    created = time.time_ns()
    audit_checkpoint = {
        "name": f"{state['active_attempt']}_execution_invalid_independent_audit",
        "created_unix_ns": created,
        "evidence_path": relative(audit_path),
        "evidence_sha256": sha256_file(audit_path),
        "source_attempt": state["active_attempt"],
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    decision_checkpoint = {
        "name": f"{state['active_attempt']}_execution_invalid_terminal_decision",
        "created_unix_ns": created,
        "evidence_path": relative(decision_path),
        "evidence_sha256": sha256_file(decision_path),
        "source_attempt": state["active_attempt"],
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    state["completed_states"].extend(("INDEPENDENT_VERIFICATION", "TERMINAL"))
    state["verified_checkpoints"].extend((audit_checkpoint, decision_checkpoint))
    state["last_verified_checkpoint"] = decision_checkpoint
    state["current_state"] = "POST_TERMINAL_REPORTING"
    state["terminal_label"] = "domain_robust_gate_execution_invalid"
    state["process_valid"] = False
    state["scientific_terminal"] = True
    state["confirmation_terminal"] = True
    state["terminal_basis"] = "postconfirmation_integrity_failure"
    state["terminal_decision_path"] = relative(decision_path)
    state["terminal_decision_sha256"] = sha256_file(decision_path)
    integrity = state.get("postconfirmation_integrity_failure")
    if not isinstance(integrity, dict):
        integrity = {
            "source": "independent_verifier",
            "trigger_state": "INDEPENDENT_VERIFICATION",
            "skipped_states": [],
        }
        state["postconfirmation_integrity_failure"] = integrity
        state["skipped_states"] = []
    integrity.update(
        {
            "status": "terminal_recorded",
            "independent_verification_path": relative(audit_path),
            "independent_verification_sha256": sha256_file(audit_path),
            "decision_path": relative(decision_path),
            "decision_sha256": sha256_file(decision_path),
            "terminal_recorded_unix_ns": created,
        }
    )
    state["next_action"] = (
        "report the immutable execution-invalid result and integrity diagnosis; "
        "do not retry or reinterpret it as a scientific partial/failure"
    )
    state["updated_unix_ns"] = created
    events = [
        {
            "event": "state_completed",
            "attempt": state["active_attempt"],
            "completed_state": "INDEPENDENT_VERIFICATION",
            "next_state": "TERMINAL",
            "checkpoint_name": audit_checkpoint["name"],
            "evidence_path": audit_checkpoint["evidence_path"],
            "evidence_sha256": audit_checkpoint["evidence_sha256"],
            "postconfirmation_integrity_failure": True,
            "created_unix_ns": created,
        },
        {
            "event": "scientific_terminal_decision_recorded",
            "attempt": state["active_attempt"],
            "terminal_label": "domain_robust_gate_execution_invalid",
            "process_valid": False,
            "decision_path": decision_checkpoint["evidence_path"],
            "decision_sha256": decision_checkpoint["evidence_sha256"],
            "created_unix_ns": created,
        },
        {
            "event": "state_completed",
            "attempt": state["active_attempt"],
            "completed_state": "TERMINAL",
            "next_state": "POST_TERMINAL_REPORTING",
            "checkpoint_name": decision_checkpoint["name"],
            "evidence_path": decision_checkpoint["evidence_path"],
            "evidence_sha256": decision_checkpoint["evidence_sha256"],
            "postconfirmation_integrity_failure": True,
            "created_unix_ns": created,
        },
    ]
    return _commit_state_with_events(state, events)


@_exclusive_operation
def version_forward(new_attempt: str, invalidity: Path, equivalence: Path) -> dict[str, Any]:
    state = _status_locked()
    if state.get("early_scientific_failure") is not None:
        raise RuntimeError("version-forward cannot replace a process-valid scientific result")
    if state.get("confirmation_outcome_episodes_generated", 0) != 0:
        raise RuntimeError("version-forward forbidden after confirmation generation")
    if state.get("confirmation_outcome_episodes_executed", 0) != 0:
        raise RuntimeError("version-forward forbidden after confirmation execution")
    if state.get("confirmation_outcomes_opened_for_analysis") is not False:
        raise RuntimeError("version-forward forbidden after confirmation outcome access")
    old_attempt = state["active_attempt"]
    old_number = int(old_attempt.removeprefix("v"))
    if new_attempt != f"v{old_number + 1:03d}":
        raise RuntimeError("new attempt is not the next immutable version")
    old_root = (REPO_ROOT / state["active_attempt_path"]).resolve()
    new_root = (STUDY_ROOT / "attempts" / new_attempt).resolve()
    if any(path.is_file() for path in (old_root / "data/confirmation").glob("**/*")):
        raise RuntimeError("confirmation artifacts exist; version-forward forbidden")
    if not new_root.is_dir():
        raise RuntimeError("new versioned attempt must already exist")
    if not invalidity.resolve().is_relative_to(old_root):
        raise RuntimeError("invalidity evidence must belong to the old attempt")
    if not equivalence.resolve().is_relative_to(new_root):
        raise RuntimeError("equivalence evidence must belong to the new attempt")
    invalidity_object = read_json(invalidity)
    equivalence_object = read_json(equivalence)
    invalidity_checks = {
        "attempt": invalidity_object.get("attempt") == old_attempt,
        "procedural_invalidity": invalidity_object.get("procedural_invalidity") is True,
        "confirmation_generated_zero": invalidity_object.get("confirmation_outcome_episodes_generated") == 0,
        "confirmation_executed_zero": invalidity_object.get("confirmation_outcome_episodes_executed") == 0,
        "confirmation_unopened": invalidity_object.get("confirmation_outcomes_opened_for_analysis") is False,
    }
    equivalence_checks = {
        "attempt": equivalence_object.get("attempt") == new_attempt,
        "source_hash": equivalence_object.get("source_hash_equivalent") is True,
        "normalized_ast": equivalence_object.get("normalized_ast_equivalent") is True,
        "scientific_objects": equivalence_object.get("scientific_object_hash_equivalent") is True,
        "configuration": equivalence_object.get("configuration_hash_equivalent") is True,
        "science_unchanged": equivalence_object.get("scientific_changes") is False,
        "resume_state": equivalence_object.get("resume_state") == state["current_state"],
        "attempt_parameterization": equivalence_object.get(
            "attempt_parameterization_verified"
        )
        is True,
        "runtime_modules": equivalence_object.get(
            "runtime_modules_accept_active_attempt"
        )
        is True,
        "independent_verifier": equivalence_object.get(
            "independent_verifier_accepts_active_attempt"
        )
        is True,
        "passed": equivalence_object.get("passed") is True,
    }
    if not all(invalidity_checks.values()) or not all(equivalence_checks.values()):
        raise RuntimeError("version-forward evidence is incomplete or non-equivalent")
    inherited_checkpoint_records = []
    for checkpoint in state["verified_checkpoints"]:
        source_attempt = _checkpoint_source_attempt(state, checkpoint)
        checkpoint["source_attempt"] = source_attempt
        checkpoint.setdefault("verification_lineage", "direct_checkpoint")
        inherited = checkpoint.setdefault("inherited_into_attempts", [])
        if not isinstance(inherited, list):
            raise RuntimeError("checkpoint inherited-attempt lineage is invalid")
        record = {
            "attempt": new_attempt,
            "equivalence_evidence_path": relative(equivalence),
            "equivalence_evidence_sha256": sha256_file(equivalence),
        }
        if record not in inherited:
            inherited.append(record)
        inherited_checkpoint_records.append(
            {
                "checkpoint_name": checkpoint["name"],
                "source_attempt": source_attempt,
                "evidence_path": checkpoint["evidence_path"],
                "evidence_sha256": checkpoint["evidence_sha256"],
            }
        )
    state["last_verified_checkpoint"] = (
        state["verified_checkpoints"][-1] if state["verified_checkpoints"] else None
    )
    created = time.time_ns()
    for item in state["attempt_history"]:
        if item["version"] == old_attempt:
            item["status"] = "invalid_zero_confirmation_outcome_procedural"
            item["invalidity_evidence_path"] = relative(invalidity)
            item["invalidity_evidence_sha256"] = sha256_file(invalidity)
            break
    state["attempt_history"].append(
        {
            "version": new_attempt,
            "path": relative(new_root),
            "status": "active_zero_confirmation_outcome_version_forward",
            "created_unix_ns": created,
            "version_forward_evidence_path": relative(equivalence),
            "version_forward_evidence_sha256": sha256_file(equivalence),
            "attempt_parameterization_verified": True,
        }
    )
    state["active_attempt"] = new_attempt
    state["active_attempt_path"] = relative(new_root)
    lineage_record = {
        "old_attempt": old_attempt,
        "new_attempt": new_attempt,
        "resume_state": state["current_state"],
        "invalidity_path": relative(invalidity),
        "invalidity_sha256": sha256_file(invalidity),
        "equivalence_path": relative(equivalence),
        "equivalence_sha256": sha256_file(equivalence),
        "inherited_verified_checkpoints": inherited_checkpoint_records,
        "attempt_parameterization_verified": True,
    }
    state.setdefault("version_forward_lineage", []).append(lineage_record)
    state["updated_unix_ns"] = created
    return _commit_state_with_events(
        state,
        [
            {
                "event": "zero_confirmation_outcome_version_forward",
                "attempt": new_attempt,
                "old_attempt": old_attempt,
                "new_attempt": new_attempt,
                "invalidity_path": relative(invalidity),
                "invalidity_sha256": sha256_file(invalidity),
                "equivalence_path": relative(equivalence),
                "equivalence_sha256": sha256_file(equivalence),
                "resume_state": state["current_state"],
                "inherited_verified_checkpoints": inherited_checkpoint_records,
                "attempt_parameterization_verified": True,
                "created_unix_ns": created,
            }
        ],
    )


@_exclusive_operation
def record_terminal(label: str, decision: Path, process_valid: bool) -> dict[str, Any]:
    state = _status_locked()
    if state.get("early_scientific_failure") is not None:
        raise RuntimeError("early scientific failure requires its narrow finalizer")
    if state["current_state"] != "TERMINAL":
        raise RuntimeError("terminal decision requested before TERMINAL state")
    if state.get("terminal_label") is not None:
        raise RuntimeError("scientific terminal decision is immutable")
    if label not in TERMINAL_LABELS:
        raise ValueError("terminal label is not preregistered")
    if label == "domain_robust_gate_execution_invalid":
        raise RuntimeError(
            "execution-invalid decisions require the post-confirmation integrity finalizer"
        )
    if not decision.is_file():
        raise FileNotFoundError(decision)
    attempt_root = (REPO_ROOT / state["active_attempt_path"]).resolve()
    if not decision.resolve().is_relative_to(attempt_root):
        raise RuntimeError("terminal decision is outside the active attempt")
    decision_object = read_json(decision)
    if decision_object.get("terminal_label") != label:
        raise RuntimeError("decision label mismatch")
    if bool(decision_object.get("process_valid")) != bool(process_valid):
        raise RuntimeError("decision process-validity mismatch")
    created = time.time_ns()
    state["terminal_label"] = label
    state["process_valid"] = bool(process_valid)
    state["scientific_terminal"] = True
    state["confirmation_terminal"] = True
    state["terminal_decision_path"] = relative(decision)
    state["terminal_decision_sha256"] = sha256_file(decision)
    state["updated_unix_ns"] = created
    return _commit_state_with_events(
        state,
        [
            {
                "event": "scientific_terminal_decision_recorded",
                "attempt": state["active_attempt"],
                "terminal_label": label,
                "process_valid": bool(process_valid),
                "decision_path": state["terminal_decision_path"],
                "decision_sha256": state["terminal_decision_sha256"],
                "created_unix_ns": created,
            }
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    advance_parser = subparsers.add_parser("advance")
    advance_parser.add_argument("completed_state", choices=STATE_MACHINE)
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
    terminal_parser.add_argument("label", choices=TERMINAL_LABELS)
    terminal_parser.add_argument("decision", type=Path)
    terminal_parser.add_argument("--execution-invalid", action="store_true")
    early_stage = subparsers.add_parser("stage-early-failure")
    early_stage.add_argument("mode", choices=tuple(EARLY_FAILURE_SPECS))
    early_finalize = subparsers.add_parser("finalize-early-failure")
    early_finalize.add_argument("mode", choices=tuple(EARLY_FAILURE_SPECS))
    subparsers.add_parser("stage-execution-invalid")
    subparsers.add_parser("finalize-execution-invalid")
    arguments = parser.parse_args()

    if arguments.command == "status":
        result = status()
    elif arguments.command == "advance":
        result = advance(
            arguments.completed_state,
            arguments.evidence.resolve(),
            arguments.checkpoint_name,
            arguments.next_action,
        )
    elif arguments.command == "counts":
        parsed: dict[str, int | bool] = {}
        for assignment in arguments.assignments:
            key, raw_value = assignment.split("=", 1)
            parsed[key] = raw_value.lower() == "true" if raw_value.lower() in ("true", "false") else int(raw_value)
        result = update_counts(**parsed)
    elif arguments.command == "version-forward":
        result = version_forward(
            arguments.new_attempt,
            arguments.invalidity.resolve(),
            arguments.equivalence.resolve(),
        )
    elif arguments.command == "stage-early-failure":
        result = stage_early_scientific_failure(arguments.mode)
    elif arguments.command == "finalize-early-failure":
        result = finalize_early_scientific_failure(arguments.mode)
    elif arguments.command == "stage-execution-invalid":
        result = stage_postconfirmation_integrity_failure()
    elif arguments.command == "finalize-execution-invalid":
        result = finalize_postconfirmation_execution_invalid()
    else:
        result = record_terminal(
            arguments.label,
            arguments.decision.resolve(),
            not arguments.execution_invalid,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
