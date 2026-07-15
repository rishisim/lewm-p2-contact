#!/usr/bin/env python3
"""Create a fail-closed manifest for the completed negative development attempt."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
MANIFEST = ROOT / "artifact_manifest.json"
VERIFICATION = ROOT / "artifact_manifest_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    excluded = {MANIFEST, VERIFICATION}
    actual = sorted(path for path in ROOT.rglob("*") if path.is_file() and path not in excluded)
    files = {str(path.relative_to(REPO_ROOT)): sha256(path) for path in actual}
    outcome = json.loads((ROOT / "DEVELOPMENT_OUTCOME.json").read_text())
    manifest = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "root": str(ROOT.relative_to(REPO_ROOT)),
        "outcome": outcome["outcome"],
        "files": files,
        "path_set_complete": True,
        "new_episode_count": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(MANIFEST, manifest)
    bad = [relative for relative, expected in files.items() if sha256(REPO_ROOT / relative) != expected]
    current = {
        str(path.relative_to(REPO_ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path not in {MANIFEST, VERIFICATION}
    }
    verification = {
        "schema_version": 1,
        "passed": not bad and current == set(files),
        "bad_hashes": bad,
        "path_set_complete": current == set(files),
        "file_count": len(files),
        "manifest_sha256": sha256(MANIFEST),
        "outcome": outcome["outcome"],
        "v5_outcome_episodes": 0,
    }
    atomic_json(VERIFICATION, verification)
    if not verification["passed"]:
        raise RuntimeError(f"development manifest failed: {verification}")
    print(json.dumps(verification, sort_keys=True))


if __name__ == "__main__":
    main()
