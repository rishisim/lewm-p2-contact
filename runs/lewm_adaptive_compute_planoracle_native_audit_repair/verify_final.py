#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 m=json.loads((R/'artifact_manifest.json').read_text());expected=set(m['files']);actual={str(p.relative_to(REPO)) for p in R.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','artifact_manifest_verification.json')};bad=[p for p,e in m['files'].items() if not (REPO/p).exists() or sha(REPO/p)!=e['sha256']];ok=expected==actual and not bad
 print(json.dumps({'passed':ok,'path_set_complete':expected==actual,'bad_hashes':bad,'manifest_sha256':sha(R/'artifact_manifest.json')},sort_keys=True));raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
