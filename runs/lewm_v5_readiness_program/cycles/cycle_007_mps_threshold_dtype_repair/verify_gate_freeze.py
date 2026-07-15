#!/usr/bin/env python3
"""Read-only verifier for the post-selection gate-freeze seal."""

from __future__ import annotations

import json
from pathlib import Path

from cycle_common import (
    REPO_ROOT,
    ROOT,
    SOURCE_CYCLE,
    SOURCE_CYCLE_HASHES,
    read_json,
    sha256_file,
)


def verify() -> dict:
    path = ROOT / "freeze/gate_freeze_seal.json"
    if not path.exists():
        raise RuntimeError("gate-freeze seal is missing")
    seal = read_json(path)
    bad = [
        relative
        for relative, expected in seal["files"].items()
        if not (REPO_ROOT / relative).exists()
        or sha256_file(REPO_ROOT / relative) != expected
    ]
    checks = {
        "status": seal["status"] == "immutable_gate_freeze_before_smoke_and_prospective",
        "fit_selection_isolation": seal["fit_selection_isolation_passed"] is True,
        "zero_smoke_at_seal": seal["smoke_episodes_at_seal"] == 0,
        "zero_prospective_at_seal": seal["prospective_episodes_at_seal"] == 0,
        "zero_v5": seal["v5_outcome_episodes"] == 0,
        "all_hashes": not bad,
        "source_cycle_decision_hash": sha256_file(SOURCE_CYCLE / "decision.json")
        == SOURCE_CYCLE_HASHES["decision"],
        "source_cycle_final_manifest_hash": sha256_file(
            SOURCE_CYCLE / "artifact_manifest.json"
        )
        == SOURCE_CYCLE_HASHES["artifact_manifest"],
        "local_compiled_gate_exact_copy": sha256_file(
            ROOT / "freeze/compiled_gate.npz"
        )
        == SOURCE_CYCLE_HASHES["compiled_gate"],
        "local_gate_fit_exact_copy": sha256_file(ROOT / "freeze/gate_fit.npz")
        == SOURCE_CYCLE_HASHES["gate_fit"],
        "local_gate_freeze_exact_copy": sha256_file(
            ROOT / "freeze/gate_freeze.json"
        )
        == SOURCE_CYCLE_HASHES["gate_freeze"],
        "local_gate_freeze_seal_exact_copy": sha256_file(path)
        == SOURCE_CYCLE_HASHES["gate_freeze_seal"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"gate-freeze seal invalid: checks={checks}, bad={bad}")
    return {
        "schema_version": 1,
        "passed": True,
        "checks": checks,
        "bad_hashes": bad,
        "seal_sha256": sha256_file(path),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
