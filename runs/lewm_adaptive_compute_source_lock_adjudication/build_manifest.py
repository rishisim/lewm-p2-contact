#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXCLUDED = {"artifact_manifest.json"}

def main():
    records = []
    for p in sorted(x for x in HERE.rglob("*") if x.is_file() and x.relative_to(HERE).as_posix() not in EXCLUDED):
        b = p.read_bytes()
        records.append({"path": p.relative_to(HERE).as_posix(), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
    out = {"schema_version": 1, "policy": {"eligible": "every regular file recursively", "excluded": {"artifact_manifest.json": "self-referential final manifest"}}, "record_count": len(records), "records": records}
    (HERE / "artifact_manifest.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__": main()
