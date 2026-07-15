#!/usr/bin/env python3
"""Read-only final cycle manifest and path-set verifier."""

from __future__ import annotations

import json

from cycle_common import REPO_ROOT, ROOT, read_json, sha256_file


def durable_files() -> set[str]:
    return {
        str(path.relative_to(REPO_ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name not in {"artifact_manifest.json", "artifact_manifest_verification.json"}
        and not path.name.startswith(".")
    }


def main() -> None:
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
        "manifest_sha256": sha256_file(manifest_path),
        "terminal_outcome": manifest["terminal_outcome"],
        "v5_outcome_episodes": manifest["v5_outcome_episodes"],
    }
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

