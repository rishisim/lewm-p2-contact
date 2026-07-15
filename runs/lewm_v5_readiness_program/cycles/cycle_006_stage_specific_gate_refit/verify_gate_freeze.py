#!/usr/bin/env python3
"""Read-only verifier for the post-selection gate-freeze seal."""

from __future__ import annotations

import json
from pathlib import Path

from cycle_common import REPO_ROOT, ROOT, read_json, sha256_file


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
