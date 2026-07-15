#!/usr/bin/env python3
"""Read-only final verifier; writes nothing and detects unrecorded files."""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "artifact_manifest.json").read_text())
expected = {r["path"]: r for r in manifest["records"]}
actual = {p.relative_to(HERE).as_posix(): p for p in HERE.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}
errors = []
if set(actual) != set(expected):
    errors.append({"unrecorded": sorted(set(actual) - set(expected)), "missing": sorted(set(expected) - set(actual))})
for name in sorted(set(actual) & set(expected)):
    b = actual[name].read_bytes()
    if len(b) != expected[name]["bytes"] or hashlib.sha256(b).hexdigest() != expected[name]["sha256"]:
        errors.append({"mismatch": name})
print(json.dumps({"status": "pass" if not errors else "fail", "record_count": len(expected), "errors": errors}, sort_keys=True))
sys.exit(1 if errors else 0)
