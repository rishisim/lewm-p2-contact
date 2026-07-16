#!/usr/bin/env python3
"""Finalize immutable package v004 after excluded smoke and zero V5 outcomes."""

from __future__ import annotations

import json
import time

from cycle_common import (
    REPO_ROOT,
    ROOT,
    SOURCE_CYCLE_HASHES,
    assert_runtime_contract,
    atomic_json,
    read_json,
    sha256_file,
)
from verify_pre_v5 import verify as verify_pre_v5


EXCLUDED_FROM_MANIFEST = {
    "artifact_manifest.json",
    "audit/final_package_verification.json",
}


def durable_files() -> list:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(ROOT)) not in EXCLUDED_FROM_MANIFEST
        and not path.name.startswith(".")
    )


def main() -> None:
    assert_runtime_contract("evaluation")
    ready_path = ROOT / "PACKAGE_READY.json"
    manifest_path = ROOT / "artifact_manifest.json"
    verification_path = ROOT / "audit/final_package_verification.json"
    if any(path.exists() for path in (ready_path, manifest_path, verification_path)):
        raise RuntimeError("package finalization artifacts are immutable and already exist")
    preseal = verify_pre_v5()
    smoke = read_json(ROOT / "audit/package_smoke_verification.json")
    power = read_json(ROOT / "power_analysis.json")
    candidate = read_json(ROOT / "freeze/frozen_candidate_manifest.json")
    forbidden = [
        ROOT / relative
        for relative in read_json(ROOT / "expected_artifact_roles.json")[
            "forbidden_at_v5_ready"
        ]
    ]
    confirmation_glob = list((ROOT / "data").glob("v5_confirmation*"))
    if (
        not smoke["passed"]
        or smoke["v5_outcome_episodes"] != 0
        or not power["passed"]
        or power["selected_v5_sample_size"] != 1600
        or not candidate["all_hashes_verified"]
        or confirmation_glob
        or any(path.exists() for path in forbidden)
    ):
        raise RuntimeError("package cannot finalize: smoke, power, freeze, or zero-outcome check failed")
    ready = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "status": "v5_package_ready",
        "package_version": "v004",
        "accepted_discovery_cycle": 7,
        "accepted_discovery_source_hashes": SOURCE_CYCLE_HASHES,
        "selected_candidate_id": candidate["candidate_id"],
        "frozen_claim": candidate["claim"],
        "frozen_candidate_manifest_sha256": sha256_file(
            ROOT / "freeze/frozen_candidate_manifest.json"
        ),
        "pre_v5_seal_sha256": preseal["seal_sha256"],
        "excluded_package_smoke_passed": True,
        "excluded_package_smoke_episode_count": 12,
        "package_smoke_verification_sha256": sha256_file(
            ROOT / "audit/package_smoke_verification.json"
        ),
        "selected_v5_sample_size": 1600,
        "primary_joint_power_lower": power["primary_power"][
            "joint_power_lower_by_union_bound"
        ],
        "stress_joint_power_lower": power["stress_power"][
            "joint_power_lower_by_union_bound"
        ],
        "artifact_manifest_path": "artifact_manifest.json",
        "confirmation_has_not_been_launched": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(ready_path, ready, exclusive=True)
    files = {
        str(path.relative_to(REPO_ROOT)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in durable_files()
    }
    manifest = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "status": "v5_package_ready",
        "package_version": "v004",
        "file_count": len(files),
        "files": files,
        "pre_v5_seal_sha256": preseal["seal_sha256"],
        "package_smoke_verification_sha256": ready[
            "package_smoke_verification_sha256"
        ],
        "accepted_discovery_final_manifest_sha256": SOURCE_CYCLE_HASHES[
            "artifact_manifest"
        ],
        "v5_outcome_episodes": 0,
    }
    atomic_json(manifest_path, manifest, exclusive=True)
    expected = set(files)
    actual = {str(path.relative_to(REPO_ROOT)) for path in durable_files()}
    bad = [
        relative
        for relative, record in files.items()
        if not (REPO_ROOT / relative).exists()
        or sha256_file(REPO_ROOT / relative) != record["sha256"]
    ]
    verification = {
        "schema_version": 1,
        "passed": expected == actual and not bad,
        "expected_path_set_equals_actual": expected == actual,
        "bad_hashes": bad,
        "file_count": len(files),
        "manifest_sha256": sha256_file(manifest_path),
        "status": "v5_package_ready",
        "v5_outcome_episodes": 0,
    }
    atomic_json(verification_path, verification, exclusive=True)
    if not verification["passed"]:
        raise RuntimeError(f"final package manifest verification failed: {verification}")
    print(json.dumps({"ready": ready, "verification": verification}, sort_keys=True))


if __name__ == "__main__":
    main()
