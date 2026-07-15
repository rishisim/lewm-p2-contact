#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
exclude={'artifact_manifest.json','final_verification.json'}
records=[]
for p in sorted(x for x in R.rglob('*') if x.is_file() and '.git' not in x.parts and x.relative_to(R).as_posix() not in exclude):
 b=p.read_bytes(); records.append({'path':p.relative_to(R).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
out={'schema_version':1,'record_count':len(records),'exclusions':sorted(exclude),'records':records}
(R/'artifact_manifest.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
