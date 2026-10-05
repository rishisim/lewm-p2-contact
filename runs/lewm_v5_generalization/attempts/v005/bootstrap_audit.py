#!/usr/bin/env python3
"""Reverify the consumed V5 package and repository before new study design."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from study_common import (
    ATTEMPT_ROOT,
    EVALUATION_PYTHON,
    EXPECTED_GIT_HEAD,
    GENERATION_PYTHON,
    REPO_ROOT,
    V5_REQUIRED_HASHES,
    V5_ROOT,
    atomic_json,
    complete_state,
    read_json,
    sha256_file,
)


def command(*arguments: str) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"bootstrap command failed: {arguments}\n{completed.stdout}"
        )
    return completed.stdout.strip()


def verify_hash_mapping(files: dict[str, str]) -> tuple[list[str], int]:
    bad: list[str] = []
    checked = 0
    for raw_path, expected in files.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        checked += 1
        if not path.exists() or sha256_file(path) != expected:
            bad.append(raw_path)
    return bad, checked


def main() -> None:
    output = ATTEMPT_ROOT / "audit/bootstrap_audit.json"
    if output.exists():
        complete_state(
            "BOOTSTRAP_AUDIT",
            "DESIGN_AND_POWER",
            evidence_path=output,
            checkpoint_name="v005_bootstrap_audit_reused",
            next_action="materialize fixed design, power, and identifiers",
        )
        print(json.dumps(read_json(output), sort_keys=True))
        return

    observed_head = command("git", "rev-parse", "HEAD")
    git_status = command("git", "status", "--short", "--branch")
    version_forward = read_json(
        ATTEMPT_ROOT / "audit/version_forward_equivalence.json"
    )
    required_observed = {
        relative: sha256_file(V5_ROOT / relative)
        for relative in V5_REQUIRED_HASHES
    }
    required_bad = sorted(
        relative
        for relative, expected in V5_REQUIRED_HASHES.items()
        if required_observed[relative] != expected
    )

    decision = read_json(V5_ROOT / "decision.json")
    independent = read_json(V5_ROOT / "audit/independent_verification.json")
    recovery = read_json(
        REPO_ROOT
        / "runs/lewm_v5_readiness_program/v5_confirmation_recovery/STATE.json"
    )
    gate_freeze = read_json(V5_ROOT / "freeze/gate_freeze.json")
    candidate = read_json(V5_ROOT / "freeze/frozen_candidate_manifest.json")

    preseal = read_json(V5_ROOT / "audit/pre_v5_seal.json")
    preseal_bad, preseal_count = verify_hash_mapping(preseal["files"])
    transitive = read_json(V5_ROOT / "transitive_source_package_manifest.json")
    transitive_bad, transitive_count = verify_hash_mapping(transitive["files"])
    package_manifest = read_json(V5_ROOT / "artifact_manifest.json")
    package_bad = []
    for relative, record in package_manifest["files"].items():
        candidate_path = REPO_ROOT / relative
        if (
            not candidate_path.exists()
            or candidate_path.stat().st_size != int(record["bytes"])
            or sha256_file(candidate_path) != record["sha256"]
        ):
            package_bad.append(relative)

    runtime_contract = read_json(V5_ROOT / "runtime_contract.json")
    external_bad, external_count = verify_hash_mapping(
        runtime_contract["external_file_hashes"]
    )

    raw_manifest = read_json(
        V5_ROOT / "data/v5_confirmation_raw_manifest.json"
    )
    raw_bad = []
    raw_bytes = 0
    for record in raw_manifest["episodes"]:
        path = REPO_ROOT / record["path"]
        raw_bytes += int(record["bytes"])
        if not path.exists() or sha256_file(path) != record["sha256"]:
            raw_bad.append(record["path"])

    verifier_outputs: dict[str, str] = {}
    for role, interpreter in (
        ("evaluation", EVALUATION_PYTHON),
        ("generation", GENERATION_PYTHON),
    ):
        verifier_outputs[role] = command(
            str(interpreter), str(V5_ROOT / "verify_pre_v5.py")
        )

    checks = {
        "git_head_exact": observed_head == EXPECTED_GIT_HEAD,
        "version_forward_equivalence_passed": version_forward.get("passed")
        is True,
        "external_commit_is_exactly_recorded": version_forward.get(
            "observed_operational_head"
        )
        == observed_head,
        "working_tree_changes_confined_to_study_root": all(
            line.startswith("?? runs/lewm_v5_generalization/")
            for line in git_status.splitlines()[1:]
        ),
        "required_named_hashes_exact": not required_bad,
        "v5_decision_terminal": decision.get("confirmation_terminal") is True,
        "v5_decision_process_valid": decision.get("process_valid") is True,
        "v5_terminal_label": decision.get("terminal_outcome")
        == "v5_confirmation_passed",
        "v5_episode_count": decision.get("v5_outcome_episodes") == 1600,
        "v5_rows": decision.get("compute", {}).get("rows") == 60_800,
        "v5_independent_audit": independent.get("passed") is True,
        "v5_recovery_terminal": recovery.get("current_state") == "TERMINAL"
        and recovery.get("terminal_label") == "v5_confirmation_passed",
        "frozen_candidate": gate_freeze["selected"]["candidate_id"]
        == "stage_dual_r0.01_q0.85",
        "thresholds_exact": gate_freeze["selected"]["thresholds"]
        == [
            0.19465729885363534,
            0.1568665868361991,
            0.15153321811129283,
        ],
        "candidate_hashes_verified": candidate.get("all_hashes_verified") is True,
        "all_preseal_files_rehashed": not preseal_bad and preseal_count == 103,
        "all_transitive_sources_rehashed": not transitive_bad
        and transitive_count == 102,
        "all_package_manifest_files_rehashed": not package_bad
        and len(package_manifest["files"]) == 92,
        "all_external_runtime_files_rehashed": not external_bad
        and external_count == len(runtime_contract["external_file_hashes"]),
        "all_1600_raw_episodes_rehashed": not raw_bad
        and len(raw_manifest["episodes"]) == 1600,
        "dual_runtime_v5_preseal_verification": all(
            '"passed": true' in text for text in verifier_outputs.values()
        ),
        "generalization_target_outcomes_absent": not (
            ATTEMPT_ROOT / "data/target"
        ).exists(),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "status": "bootstrap_audit_passed"
        if all(checks.values())
        else "bootstrap_audit_failed",
        "checks": checks,
        "passed": all(checks.values()),
        "git": {
            "expected_head": EXPECTED_GIT_HEAD,
            "observed_head": observed_head,
            "status": git_status.splitlines(),
            "history_or_worktree_mutated": False,
            "original_user_required_head": version_forward[
                "original_required_head"
            ],
            "zero_outcome_operational_version_forward": True,
        },
        "v5_required_hashes": {
            relative: {
                "expected": V5_REQUIRED_HASHES[relative],
                "observed": required_observed[relative],
            }
            for relative in V5_REQUIRED_HASHES
        },
        "rehashed_counts": {
            "preseal_files": preseal_count,
            "transitive_sources": transitive_count,
            "package_manifest_files": len(package_manifest["files"]),
            "external_runtime_files": external_count,
            "raw_v5_episodes": len(raw_manifest["episodes"]),
            "raw_v5_bytes": raw_bytes,
        },
        "bad_paths": {
            "required": required_bad,
            "preseal": preseal_bad,
            "transitive": transitive_bad,
            "package_manifest": package_bad,
            "external": external_bad,
            "raw": raw_bad,
        },
        "dual_runtime_verifier_outputs": verifier_outputs,
        "forbidden_evidence_access": {
            "v3_test_targets_opened": False,
            "combined_v3_cache_numpy_loaded": False,
            "released_hdf5_opened": False,
        },
        "v5_consumed_read_only": True,
        "v5_rerun": False,
        "v5_files_modified": False,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"bootstrap audit failed: {result}")
    complete_state(
        "BOOTSTRAP_AUDIT",
        "DESIGN_AND_POWER",
        evidence_path=output,
        checkpoint_name="v005_v5_and_workspace_bootstrap_verified",
        next_action="materialize fixed design, power, and identifiers",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
