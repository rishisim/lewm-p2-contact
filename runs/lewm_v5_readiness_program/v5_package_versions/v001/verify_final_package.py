#!/usr/bin/env python3
"""Read-only final v001 manifest and path-set verifier."""

from __future__ import annotations

import json

from cycle_common import REPO_ROOT, ROOT, read_json, sha256_file


EXCLUDED = {"artifact_manifest.json", "audit/final_package_verification.json"}


def durable_files() -> set[str]:
    return {
        str(path.relative_to(REPO_ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(ROOT)) not in EXCLUDED
        and not path.name.startswith(".")
    }


def verify() -> dict:
    manifest_path = ROOT / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    expected = set(manifest["files"])
    actual = durable_files()
    bad = [
        relative
        for relative, record in manifest["files"].items()
        if not (REPO_ROOT / relative).exists()
        or sha256_file(REPO_ROOT / relative) != record["sha256"]
    ]
    result = {
        "passed": expected == actual and not bad,
        "path_set_complete": expected == actual,
        "bad_hashes": bad,
        "file_count": len(expected),
        "manifest_sha256": sha256_file(manifest_path),
        "status": manifest["status"],
        "v5_outcome_episodes": manifest["v5_outcome_episodes"],
    }
    if not result["passed"]:
        raise RuntimeError(f"final package verification failed: {result}")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
