#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.loads((R/'decision.json').read_text());m=json.loads((R/'data/prospective_evaluated_manifest.json').read_text());assert sha(R/'data/prospective_evaluated.npz')==m['sha256']
with np.load(R/'data/prospective_evaluated.npz') as z, np.load(R/'freeze/gate_contract.npz') as c, np.load(R/'freeze/gate_weights.npz') as w:
 target=z['target'];exits=z['exits'];ids=z['episode_id'];x=np.c_[z['features'].reshape(-1,1046).astype(float),np.tile(np.eye(3),(len(target),1))];q=(x-c['feature_mean'])/c['feature_std'];score=np.minimum(q@w['wr'],q@w['ww']).reshape(-1,3);W=c['whitening_matrix']
s=json.loads((R/'freeze/gate_freeze.json').read_text())['selected'];calls=np.ones(len(score),np.int8);active=np.ones(len(score),bool)
for j in range(3):active&=score[:,j]>s['threshold'];calls+=active
diff=exits.astype(float)-target[:,None,:];raw=np.square(diff).mean(2);white=np.square(np.einsum('nkd,df->nkf',diff,W)).mean(2);idx=np.arange(len(calls));ar=raw[idx,calls-1];aw=white[idx,calls-1];gd=np.minimum(calls,3).sum();equiv=(calls.sum()+gd*s['gate_flops_per_evaluation']/264960)/len(calls);w2=equiv-1;rm=(1-w2)*raw[:,0]+w2*raw[:,1];wm=(1-w2)*white[:,0]+w2*white[:,1]
def ep(v):return np.array([v[ids==i].mean() for i in np.unique(ids)])
assert abs(ep(rm-ar).mean()-d['criteria']['raw_vs_analytic']['estimate'])<1e-15
assert abs(ep(wm-aw).mean()-d['criteria']['white_vs_analytic']['estimate'])<1e-15
assert d['terminal_outcome']=='planoracle_prospective_discovery_passed' and all(v['lower']>0 for v in d['criteria'].values()) and all(x['rho']>0 for x in d['stagewise_rank'])
out={'passed':True,'terminal_outcome_recomputed':d['terminal_outcome'],'raw_analytic_benefit_recomputed':ep(rm-ar).mean(),'white_analytic_benefit_recomputed':ep(wm-aw).mean(),'call_histogram_recomputed':np.bincount(calls,minlength=5)[1:].tolist(),'report_not_trusted':True}
(R/'audit/independent_recomputation.json').write_text(json.dumps(out,indent=2,sort_keys=True,default=float)+'\n');print(json.dumps(out,default=float))
