#!/usr/bin/env python3
"""Atomic, append-only control records for the V5 confirmation recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROGRAM_ROOT = ROOT.parent
REPO_ROOT = PROGRAM_ROOT.parents[1]
STATE_PATH = ROOT / "STATE.json"
POINTER_PATH = ROOT / "CURRENT_POINTER.json"
LEDGER_PATH = ROOT / "LEDGER.md"
V001_SUMMARY_PATH = ROOT / "v001_execution_invalid.json"
V001_ROOT = PROGRAM_ROOT / "v5_package_versions/v001"

EXPECTED_HEAD = "d20ad79c4db7f564f5007513634bec4b9a785605"
EXPECTED_V001_FAILURE_SHA256 = (
    "2a94c67335084fad28e431c7f630aca9005d77a6822995818dc72cf0d2a305f0"
)
EXPECTED_V001_SEAL_SHA256 = (
    "ca3e01672a8cbf0042426c63f35c2297e55e4f2592cbc2eae31e4aeff509a4dd"
)
STATE_MACHINE = [
    "DIAGNOSE",
    "IMPLEMENT_PACKAGE",
    "PRESEAL_QUALIFY",
    "PRE_V5_SEAL",
    "EXCLUDED_PACKAGE_SMOKE",
    "PACKAGE_READY",
    "CONFIRMATION_GENERATION",
    "CONFIRMATION_EXECUTION",
    "SEALED_ANALYSIS",
    "LATENCY",
    "INDEPENDENT_AUDIT",
    "TERMINAL",
]
HISTORICAL_RECORDS = [
    "V5_READY.json",
    "PROGRAM_STATE.json",
    "CURRENT_POINTER.json",
    "RESEARCH_LEDGER.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now_ns() -> int:
    return time.time_ns()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def append_ledger(heading: str, body: str) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    prefix = "" if LEDGER_PATH.exists() else "# LeWM V5 confirmation recovery ledger\n\n"
    encoded = f"{prefix}## {utc_now()} — {heading}\n\n{body.rstrip()}\n\n"
    descriptor = os.open(LEDGER_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def historical_hashes() -> dict[str, str]:
    return {name: sha256_file(PROGRAM_ROOT / name) for name in HISTORICAL_RECORDS}


def verify_historical_hashes(expected: dict[str, str]) -> None:
    observed = historical_hashes()
    if observed != expected:
        raise RuntimeError(f"historical program record drift: {observed}")


def v001_diagnosis() -> dict[str, Any]:
    failure_path = V001_ROOT / "audit/v5_confirmation_generation_failures.json"
    seal_path = V001_ROOT / "audit/pre_v5_seal.json"
    if sha256_file(failure_path) != EXPECTED_V001_FAILURE_SHA256:
        raise RuntimeError("v001 generation-failure evidence drift")
    if sha256_file(seal_path) != EXPECTED_V001_SEAL_SHA256:
        raise RuntimeError("v001 pre-V5 seal drift")
    failure = read_json(failure_path)
    failures = failure.get("failures", [])
    raw_directory = V001_ROOT / "data/v5_confirmation_raw"
    raw_files = sorted(str(path.relative_to(REPO_ROOT)) for path in raw_directory.rglob("*") if path.is_file())
    forbidden = [
        V001_ROOT / "data/v5_confirmation_raw_manifest.json",
        V001_ROOT / "data/v5_confirmation_execution.npz",
        V001_ROOT / "data/v5_confirmation_execution_manifest.json",
        V001_ROOT / "audit/v5_confirmation_input_seal.json",
        V001_ROOT / "analysis_result.json",
        V001_ROOT / "metrics/v5_confirmation_latency.json",
        V001_ROOT / "audit/independent_verification.json",
        V001_ROOT / "decision.json",
    ]
    seed_tuples = {
        (int(item["env_seed"]), int(item["policy_seed"]), int(item["oracle_np_seed"]))
        for item in failures
    }
    attempted_ids = {str(item["attempted_seed_source_episode_id"]) for item in failures}
    checks = {
        "failure_count_exact_101": len(failures) == 101,
        "all_failures_slot_zero": {int(item["slot"]) for item in failures} == {0},
        "all_exception_types_module_not_found": {
            str(item["exception_type"]) for item in failures
        } == {"ModuleNotFoundError"},
        "all_messages_exact": {str(item["exception_message"]) for item in failures}
        == {"No module named 'stable_worldmodel'"},
        "all_inspection_flags_false": all(
            item.get("outcome_loss_contact_or_success_inspected") is False
            for item in failures
        ),
        "all_attempted_identifiers_unique": len(attempted_ids) == 101,
        "all_attempted_seed_tuples_unique": len(seed_tuples) == 101,
        "confirmation_raw_directory_empty": not raw_files,
        "no_confirmation_output_artifacts": not any(path.exists() for path in forbidden),
        "pre_v5_seal_hash_exact": sha256_file(seal_path) == EXPECTED_V001_SEAL_SHA256,
        "failure_record_hash_exact": sha256_file(failure_path)
        == EXPECTED_V001_FAILURE_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v001 diagnosis failed: {checks}")
    return {
        "schema_version": 1,
        "created_unix_ns": now_ns(),
        "source_task_id": "019f67dd-913e-7922-a8fd-01747ae44a0d",
        "package_version": "v001",
        "package_path": str(V001_ROOT.relative_to(REPO_ROOT)),
        "operational_terminal_label": "v5_execution_invalid",
        "scientific_interpretation": (
            "pre-outcome procedural execution invalidity caused by generation being invoked "
            "with the sealed evaluation-only interpreter; not a statistical retry"
        ),
        "confirmation_outcome_count": 0,
        "raw_episode_count": 0,
        "modeled_row_count": 0,
        "failed_attempt_count": 101,
        "failed_slot_set": [0],
        "failure_record_path": str(failure_path.relative_to(REPO_ROOT)),
        "failure_record_sha256": sha256_file(failure_path),
        "pre_v5_seal_path": str(seal_path.relative_to(REPO_ROOT)),
        "pre_v5_seal_sha256": sha256_file(seal_path),
        "checks": checks,
        "no_analysis_or_decision_artifact_fabricated": True,
        "all_v001_identifiers_treated_as_consumed": True,
    }


def initialize() -> None:
    if any(path.exists() for path in (STATE_PATH, POINTER_PATH, V001_SUMMARY_PATH)):
        raise RuntimeError("recovery control root is already initialized")
    current_head = git_head()
    if current_head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected Git HEAD before recovery: {current_head}")
    diagnosis = v001_diagnosis()
    immutable_hashes = historical_hashes()
    atomic_json(V001_SUMMARY_PATH, diagnosis, exclusive=True)
    summary_sha256 = sha256_file(V001_SUMMARY_PATH)
    created = now_ns()
    state = {
        "schema_version": 1,
        "created_unix_ns": created,
        "updated_unix_ns": created,
        "program": "LeWM V5 confirmation recovery and one-shot completion",
        "state_machine": STATE_MACHINE,
        "current_state": "DIAGNOSE",
        "completed_states": ["DIAGNOSE"],
        "active_package_version": None,
        "active_package_path": None,
        "last_verified_checkpoint": {
            "name": "v001_zero_outcome_execution_invalidity_independently_verified",
            "created_unix_ns": diagnosis["created_unix_ns"],
            "evidence_path": str(V001_SUMMARY_PATH.relative_to(REPO_ROOT)),
            "evidence_sha256": summary_sha256,
        },
        "next_action": "materialize and qualify the minimal operational-only package v002 repair",
        "confirmation_outcome_count": 0,
        "confirmation_cohort_complete": False,
        "terminal_label": None,
        "expected_git_head": EXPECTED_HEAD,
        "observed_git_head": current_head,
        "immutable_historical_record_hashes": immutable_hashes,
    }
    atomic_json(STATE_PATH, state, exclusive=True)
    pointer = {
        "schema_version": 1,
        "created_unix_ns": created,
        "updated_unix_ns": created,
        "status": "v001_execution_invalid_zero_outcomes_verified",
        "recovery_state_path": str(STATE_PATH.relative_to(REPO_ROOT)),
        "recovery_state_sha256": sha256_file(STATE_PATH),
        "v001_execution_invalid_path": str(V001_SUMMARY_PATH.relative_to(REPO_ROOT)),
        "v001_execution_invalid_sha256": summary_sha256,
        "active_package": None,
        "terminal_evidence": None,
        "confirmation_outcome_count": 0,
    }
    atomic_json(POINTER_PATH, pointer, exclusive=True)
    append_ledger(
        "DIAGNOSE completed",
        (
            "Independently verified package v001 as `v5_execution_invalid` before any "
            "confirmation outcome: 0/1,600 raw episodes, 0/60,800 modeled rows, and no "
            "analysis, latency, independent-audit, or decision artifact. The immutable "
            f"failure record has SHA-256 `{EXPECTED_V001_FAILURE_SHA256}` and contains "
            "exactly 101 slot-0 `ModuleNotFoundError` attempts, all without outcome, loss, "
            "contact, reward, or success inspection. The pre-V5 seal verifies at SHA-256 "
            f"`{EXPECTED_V001_SEAL_SHA256}`. All 1,712 v001 seed tuples are treated as "
            "consumed. Version-forward repair is operational and pre-outcome, not a "
            "statistical retry."
        ),
    )
    print(json.dumps({"state": state, "pointer": pointer}, sort_keys=True))


def checkpoint(args: argparse.Namespace) -> None:
    state = read_json(STATE_PATH)
    verify_historical_hashes(state["immutable_historical_record_hashes"])
    current_head = git_head()
    if current_head != state["expected_git_head"]:
        raise RuntimeError(f"Git HEAD drift: {current_head}")
    if args.state not in STATE_MACHINE:
        raise RuntimeError(f"unknown recovery state: {args.state}")
    if args.outcomes < int(state["confirmation_outcome_count"]):
        raise RuntimeError("confirmation outcome count cannot decrease")
    completed = list(state.get("completed_states", []))
    if args.complete_state and args.state not in completed:
        completed.append(args.state)
    package_path = (
        f"runs/lewm_v5_readiness_program/v5_package_versions/{args.package_version}"
        if args.package_version
        else None
    )
    state.update(
        {
            "updated_unix_ns": now_ns(),
            "current_state": args.state,
            "completed_states": completed,
            "active_package_version": args.package_version,
            "active_package_path": package_path,
            "last_verified_checkpoint": {
                "name": args.checkpoint,
                "created_unix_ns": now_ns(),
                "evidence_path": args.evidence_path,
                "evidence_sha256": args.evidence_sha256,
            },
            "next_action": args.next_action,
            "confirmation_outcome_count": args.outcomes,
            "confirmation_cohort_complete": args.cohort_complete,
            "terminal_label": args.terminal_label,
            "observed_git_head": current_head,
        }
    )
    atomic_json(STATE_PATH, state)
    pointer = read_json(POINTER_PATH)
    active_package = None
    if args.package_version:
        active_package = {
            "version": args.package_version,
            "path": package_path,
            "checkpoint": args.checkpoint,
            "evidence_path": args.evidence_path,
            "evidence_sha256": args.evidence_sha256,
        }
    terminal = None
    if args.terminal_label:
        terminal = {
            "label": args.terminal_label,
            "path": args.evidence_path,
            "sha256": args.evidence_sha256,
        }
    pointer.update(
        {
            "updated_unix_ns": now_ns(),
            "status": args.checkpoint,
            "recovery_state_sha256": sha256_file(STATE_PATH),
            "active_package": active_package,
            "terminal_evidence": terminal,
            "confirmation_outcome_count": args.outcomes,
        }
    )
    atomic_json(POINTER_PATH, pointer)
    append_ledger(args.ledger_heading, args.ledger_body)
    print(json.dumps({"state": state, "pointer": pointer}, sort_keys=True))


def verify() -> None:
    state = read_json(STATE_PATH)
    pointer = read_json(POINTER_PATH)
    summary = read_json(V001_SUMMARY_PATH)
    verify_historical_hashes(state["immutable_historical_record_hashes"])
    current_head = git_head()
    checks = {
        "git_head": current_head == state["expected_git_head"],
        "v001_summary_hash": sha256_file(V001_SUMMARY_PATH)
        == pointer["v001_execution_invalid_sha256"],
        "v001_failure_hash": summary["failure_record_sha256"]
        == EXPECTED_V001_FAILURE_SHA256,
        "v001_seal_hash": summary["pre_v5_seal_sha256"] == EXPECTED_V001_SEAL_SHA256,
        "outcome_count_nonnegative": int(state["confirmation_outcome_count"]) >= 0,
        "known_state": state["current_state"] in STATE_MACHINE,
    }
    if not all(checks.values()):
        raise RuntimeError(f"recovery control verification failed: {checks}")
    print(json.dumps({"passed": True, "checks": checks, "state": state}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("initialize")
    subparsers.add_parser("verify")
    update = subparsers.add_parser("checkpoint")
    update.add_argument("--state", required=True)
    update.add_argument("--package-version")
    update.add_argument("--checkpoint", required=True)
    update.add_argument("--evidence-path")
    update.add_argument("--evidence-sha256")
    update.add_argument("--next-action", required=True)
    update.add_argument("--outcomes", type=int, required=True)
    update.add_argument("--cohort-complete", action="store_true")
    update.add_argument("--terminal-label")
    update.add_argument("--complete-state", action="store_true")
    update.add_argument("--ledger-heading", required=True)
    update.add_argument("--ledger-body", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "initialize":
        initialize()
    elif args.command == "checkpoint":
        checkpoint(args)
    else:
        verify()


if __name__ == "__main__":
    main()
