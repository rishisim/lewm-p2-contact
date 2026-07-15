#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/"artifact_manifest.json").read_text())
bad=[]
for r in m["records"]:
 p=ROOT/r["path"]
 if not p.is_file() or p.stat().st_size!=r["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest()!=r["sha256"]: bad.append(r["path"])
out={"status":"pass" if not bad else "fail","record_count":len(m["records"]),"hash_mismatches":bad,"manifest_modified":False}
print(json.dumps(out,sort_keys=True))
if bad: sys.exit(1)
