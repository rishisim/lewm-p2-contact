#!/usr/bin/env python3
"""Read-only verifier for the pre-fit cycle-006 design seal."""

from __future__ import annotations

import json
from pathlib import Path

from cycle_common import REPO_ROOT, ROOT, read_json, sha256_file


def verify() -> dict:
    path = ROOT / "audit/design_seal.json"
    if not path.exists():
        raise RuntimeError("design seal missing")
    seal = read_json(path)
    bad = []
    for relative, expected in seal["files"].items():
        source = Path(relative)
        if not source.is_absolute():
            source = REPO_ROOT / source
        if not source.exists() or sha256_file(source) != expected:
            bad.append(relative)
    checks = {
        "status": seal["status"]
        == "frozen_before_any_cycle_006_fit_selection_smoke_or_prospective_episode",
        "zero_fit": seal["fit_episodes_at_seal"] == 0,
        "zero_selection": seal["selection_episodes_at_seal"] == 0,
        "zero_smoke": seal["smoke_episodes_at_seal"] == 0,
        "zero_prospective": seal["prospective_episodes_at_seal"] == 0,
        "zero_v5": seal["v5_outcome_episodes"] == 0,
        "all_hashes": not bad,
        "code_complete": seal[
            "all_generation_evaluation_fit_selection_sparse_analysis_audit_and_manifest_code_materialized"
        ],
    }
    if not all(checks.values()):
        raise RuntimeError(f"design seal invalid: checks={checks}, bad={bad}")
    return {
        "schema_version": 1,
        "passed": True,
        "checks": checks,
        "bad_hashes": bad,
        "sealed_file_count": len(seal["files"]),
        "seal_sha256": sha256_file(path),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
