#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    out=ROOT/'artifact_manifest.json'
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p==out or '/historical/source_repo/' in p.as_posix(): continue
        files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    head='a265229cb29688715651cbd831a3b4c10b8f98b4'
    payload={'schema_version':1,'status':'complete','records':files,'record_count':len(files),'total_bytes':sum(x['bytes'] for x in files),'excluded_materialized_source_tree':{'path':'historical/source_repo','git_commit':head,'reason':'represented by immutable git commit rather than per-file manifest'},'v3_test_targets_opened':False,'v5_confirmation_episodes':0}
    out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'records':len(files),'bytes':payload['total_bytes']}))
if __name__=='__main__': main()
