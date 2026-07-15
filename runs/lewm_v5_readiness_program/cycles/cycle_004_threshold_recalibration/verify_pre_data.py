#!/usr/bin/env python3
"""Read-only verifier for the cycle-004 pre-data seal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cycle_common import REPO_ROOT, ROOT, read_json, sha256_file


def verify() -> dict[str, Any]:
    seal_path = ROOT / "audit/pre_data_seal.json"
    if not seal_path.exists():
        raise RuntimeError("pre-data seal is missing")
    seal = read_json(seal_path)
    bad = []
    for raw_path, expected in seal["files"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() or sha256_file(path) != expected:
            bad.append(raw_path)
    checks = {
        "status_frozen": seal.get("status") == "frozen_before_any_cycle_004_smoke_or_prospective_episode",
        "zero_new_raw_at_seal": seal.get("zero_new_raw_episode_files_at_seal") is True,
        "zero_prospective_at_seal": seal.get("prospective_episodes_at_seal") == 0,
        "zero_v5_at_seal": seal.get("v5_outcome_episodes") == 0,
        "all_hashes_match": not bad,
    }
    if not all(checks.values()):
        raise RuntimeError(f"pre-data seal invalid: checks={checks}, bad={bad}")
    return {
        "passed": True,
        "checks": checks,
        "bad_hashes": bad,
        "sealed_file_count": len(seal["files"]),
        "seal_sha256": sha256_file(seal_path),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
