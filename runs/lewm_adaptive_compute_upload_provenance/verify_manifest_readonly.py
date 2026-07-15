#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent; M=json.loads((R/'artifact_manifest.json').read_text()); expected={x['path']:x for x in M['records']}; exclude=set(M['exclusions'])
actual={p.relative_to(R).as_posix():p for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and p.relative_to(R).as_posix() not in exclude}
checks={'path_set':set(actual)==set(expected),'record_count':len(actual)==M['record_count'],'hashes':all(actual[k].stat().st_size==v['bytes'] and hashlib.sha256(actual[k].read_bytes()).hexdigest()==v['sha256'] for k,v in expected.items() if k in actual)}
out={'checks':checks,'passed':all(checks.values()),'unrecorded':sorted(set(actual)-set(expected)),'missing':sorted(set(expected)-set(actual))}
print(json.dumps(out,indent=2,sort_keys=True)); sys.exit(0 if out['passed'] else 1)
