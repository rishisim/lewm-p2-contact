#!/usr/bin/env python3
"""Create the preregistration seal before detailed outcome inspection."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib


BASE = pathlib.Path(__file__).resolve().parents[1]
SEALED_NAMES = [
    "AUDIT_PLAN.md",
    "THREAT_MODEL.md",
    "TEST_MATRIX.json",
    "SOURCE_SNAPSHOT.json",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def main() -> int:
    matrix = json.loads((BASE / "TEST_MATRIX.json").read_text(encoding="utf-8"))
    snapshot = json.loads((BASE / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    tests = matrix["tests"]
    if len(tests) != len({test["id"] for test in tests}):
        raise SystemExit("duplicate test id")
    if any(test["status"] != "not_run" for test in tests):
        raise SystemExit("test matrix contains a result before sealing")
    if any(test["post_hoc"] for test in tests):
        raise SystemExit("preseal matrix contains post-hoc test")

    sealed_files = {}
    for name in SEALED_NAMES:
        data = (BASE / name).read_bytes()
        sealed_files[name] = {"sha256": sha256_bytes(data), "bytes": len(data)}

    seal = {
        "schema_version": 1,
        "seal_role": "pre_outcome_detail_audit_plan_seal",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": BASE.name,
        "known_outcome_scope_at_seal": (
            "Only the user-supplied headline claims; no detailed per-episode "
            "result arrays or tables inspected."
        ),
        "hash_algorithm": "sha256",
        "sealed_files": sealed_files,
        "source_snapshot_aggregate_sha256": snapshot["summary"]["aggregate_sha256"],
        "predeclared_commitments": {
            "test_count": len(tests),
            "test_ids": [test["id"] for test in tests],
            "tests_include_expected_falsification_behavior": all(
                bool(test["falsification"]) for test in tests
            ),
            "numerical_tolerances_location": "AUDIT_PLAN.md#predeclared-numerical-tolerances",
            "finding_severities": ["Blocker", "Major", "Minor"],
            "terminal_outcomes": [
                "v5_validity_supported",
                "v5_validity_supported_with_caveats",
                "v5_validity_invalidated",
                "v5_validity_audit_inconclusive",
            ],
            "terminal_mapping_location": "AUDIT_PLAN.md#terminal-decision-mapping",
            "negative_control_master_seeds": {
                "score_or_decision_permutation": 510001,
                "random_matched_call": 510002,
                "constant_mixed_matched_compute": 510003,
                "target_block_permutation": 510004,
                "broken_score_gain_association": 510005,
                "ordering_reversal_perturbation": 510006,
                "synthetic_fixture_seeds": [730001, 730002, 730003],
                "sensitivity_bootstrap": 880001,
            },
            "negative_control_repetitions_per_random_family": 2000,
            "minimum_negative_control_repetitions": 1000,
            "mandatory_for_support_test_ids": [
                test["id"] for test in tests if test["mandatory_for_support"]
            ],
            "mandatory_for_invalidated": (
                "validated Blocker, synthetic harness pass, second-method "
                "reproduction where technically possible, documented causal path"
            ),
            "mandatory_for_inconclusive": (
                "exact unresolved essential evidence/disagreement, alternatives "
                "attempted, and credible Blocker-concealing surface documented"
            ),
            "post_hoc_tests_must_be_separately_labeled": True,
        },
    }
    seal["sealed_payload_sha256"] = sha256_bytes(canonical_bytes(seal))
    output = BASE / "audit_plan_seal.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    for name in [*SEALED_NAMES, "audit_plan_seal.json"]:
        os.chmod(BASE / name, 0o444)
    print(seal["sealed_payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
