#!/usr/bin/env python3
"""Read-only verifier for the immutable v001 pre-V5 seal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cycle_common import REPO_ROOT, ROOT, read_json, sha256_file


def verify() -> dict[str, Any]:
    seal_path = ROOT / "audit/pre_v5_seal.json"
    if not seal_path.exists():
        raise RuntimeError("pre-V5 seal is missing")
    seal = read_json(seal_path)
    bad = []
    for raw_path, expected in seal["files"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() or sha256_file(path) != expected:
            bad.append(raw_path)
    checks = {
        "status_frozen": seal.get("status")
        == "frozen_before_any_v001_package_smoke_or_v5_confirmation_episode",
        "package_version": seal.get("package_version") == "v001",
        "accepted_cycle": seal.get("accepted_discovery_cycle") == 7,
        "package_smoke_zero_at_seal": seal.get("package_smoke_episodes_at_seal") == 0,
        "v5_zero_at_seal": seal.get("v5_outcome_episodes_at_seal") == 0,
        "v5_size": seal.get("v5_confirmation_episode_count") == 1600,
        "power_passed": seal.get("power_analysis_passed") is True,
        "normative_code_complete": seal.get("normative_code_complete") is True,
        "all_hashes_match": not bad,
    }
    if not all(checks.values()):
        raise RuntimeError(f"pre-V5 seal invalid: checks={checks}, bad={bad}")
    return {
        "passed": True,
        "checks": checks,
        "bad_hashes": bad,
        "sealed_file_count": len(seal["files"]),
        "seal_sha256": sha256_file(seal_path),
        "v5_outcome_episodes_at_seal": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
