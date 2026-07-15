#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def verify():
 s=json.loads((R/'audit/pre_data_seal.json').read_text());bad=[]
 for p,e in s['files'].items():
  q=REPO/p
  if not q.exists() or sha(q)!=e:bad.append(p)
 ok=s['status']=='frozen_before_any_smoke_or_prospective_data' and s['zero_new_raw_episode_files_at_seal'] and not bad
 if not ok:raise RuntimeError('pre-data seal invalid: '+repr(bad))
 return {'passed':True,'files':len(s['files']),'seal_sha256':sha(R/'audit/pre_data_seal.json')}
if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
