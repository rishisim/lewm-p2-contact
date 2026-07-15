#!/usr/bin/env python3
"""Finalize provenance, hash every deliverable, and verify the manifest."""

from __future__ import annotations

import json
from pathlib import Path

import common


def _files() -> list[Path]:
    excluded = {
        common.ROOT / "artifact_manifest.json",
        common.ROOT / "audit/artifact_manifest_verification.json",
    }
    return sorted(
        path
        for path in common.ROOT.rglob("*")
        if path.is_file()
        and path not in excluded
        and "__pycache__" not in path.parts
        and not path.name.startswith(".")
    )


def run() -> dict[str, object]:
    final_tests = common.ROOT / "logs/final_tests.json"
    if not final_tests.exists() or not common.read_json(final_tests)["passed"]:
        raise RuntimeError("final post-report tests have not passed")
    required = [
        common.ROOT / "REPORT.md",
        common.ROOT / "README.md",
        common.ROOT / "PREREGISTRATION.md",
        common.ROOT / "decision.json",
        common.ROOT / "metrics/primary_confirmation.json",
        common.ROOT / "metrics/physical_regimes.json",
        common.ROOT / "metrics/runtime.json",
        common.ROOT / "metrics/flops.json",
        common.ROOT / "audit/freshness_duplicate_audit.json",
        common.ROOT / "audit/confirmation_access_receipt.json",
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"missing final deliverable: {path}")
    verification = {
        "schema_version": 1,
        "status": "ready_for_artifact_manifest",
        "final_tests_sha256": common.sha256_file(final_tests),
        "frozen_objects": common.verify_frozen_objects(),
        "phase0_seal_sha256": common.sha256_file(common.ROOT / "audit/phase0_seal.json"),
        "preconfirmation_seal_sha256": common.sha256_file(
            common.ROOT / "audit/pre_confirmation_seal.json"
        ),
        "confirmation_receipt_sha256": common.sha256_file(
            common.ROOT / "audit/confirmation_access_receipt.json"
        ),
        "confirmation_outcome_sha256": common.sha256_file(
            common.ROOT / "data/confirmation_outcome_once.npz"
        ),
        "decision": common.read_json(common.ROOT / "decision.json")["decision"],
        "runtime_deployment_status": common.read_json(common.ROOT / "decision.json")[
            "runtime_deployment_status"
        ],
        "v3_test_targets_opened": False,
        "prior_artifacts_mutated": False,
        "commit_or_push_performed": False,
    }
    common.write_json(common.ROOT / "audit/final_verification.json", verification)
    manifest_path = common.ROOT / "artifact_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("artifact manifest already exists")
    records = []
    for path in _files():
        records.append(
            {
                "path": str(path.relative_to(common.REPO)),
                "sha256": common.sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "complete_v4_artifact_manifest",
        "root": str(common.ROOT.relative_to(common.REPO)),
        "file_count": len(records),
        "files": records,
        "manifest_self_excluded_to_avoid_recursive_hash": True,
        "v3_test_targets_opened": False,
    }
    common.write_json(manifest_path, manifest, exclusive=True)
    failures = []
    for record in records:
        path = common.REPO / record["path"]
        observed = common.sha256_file(path)
        if observed != record["sha256"] or path.stat().st_size != record["bytes"]:
            failures.append({"path": record["path"], "observed_sha256": observed})
    result = {
        "schema_version": 1,
        "passed": not failures,
        "manifest_path": str(manifest_path.relative_to(common.REPO)),
        "manifest_sha256": common.sha256_file(manifest_path),
        "verified_entries": len(records),
        "failures": failures,
        "manifest_self_hash_recorded_here": True,
        "v3_test_targets_opened": False,
    }
    common.write_json(
        common.ROOT / "audit/artifact_manifest_verification.json", result, exclusive=True
    )
    if failures:
        raise RuntimeError(f"artifact verification failures: {failures}")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
