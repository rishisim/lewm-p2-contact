#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXCLUDE={"artifact_manifest.json"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
for p in sorted(ROOT.rglob("*")):
    rel=p.relative_to(ROOT)
    if not p.is_file() or p.name in EXCLUDE or rel.parts[0] in {".venv","source_repo"}: continue
    files.append({"path":str(rel),"bytes":p.stat().st_size,"sha256":sha(p)})
out={"schema_version":1,"records":files,"record_count":len(files),"total_bytes":sum(x["bytes"] for x in files),
     "exclusions":{".venv/":"represented by environment_provenance.json, environment_requirements.txt, and environment_sync.log","source_repo/":"represented by detached commit, uv.lock hash, and selected hashes in environment_provenance.json","source_repo/.git/":"immutable git commit representation"}}
(ROOT/"artifact_manifest.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
