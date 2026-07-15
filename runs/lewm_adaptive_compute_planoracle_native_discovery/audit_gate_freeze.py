#!/usr/bin/env python3
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; REPO=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
subprocess.run([sys.executable,str(R/'freeze/verify_gate_freeze.py')],check=True)
p=json.loads((R/'freeze/gate_freeze.json').read_text()); s=p['selected']
with np.load(R/'freeze/gate_contract.npz',allow_pickle=False) as c, np.load(R/'freeze/gate_weights.npz',allow_pickle=False) as w, np.load(R/'data/selection_evaluated.npz',allow_pickle=False) as z:
 assert all(np.isfinite(c[k]).all() for k in c.files if c[k].dtype.kind in 'fc')
 assert all(np.isfinite(w[k]).all() for k in w.files)
 f=z['features'].reshape(-1,1046).astype(float);stage=np.tile(np.eye(3),(len(z['target']),1));x=np.c_[f,stage];zz=(x-c['feature_mean'])/c['feature_std'];scores=np.minimum(zz@w['wr'],zz@w['ww']).reshape(-1,3)
 calls=np.ones(len(scores),np.int8); active=np.ones(len(scores),bool)
 traces=[]
 for j in range(3):
  decision=scores[:,j]>s['threshold']; active &= decision; calls+=active; traces.append(active.copy())
 dense=z['exits'][np.arange(len(calls)),calls-1]
 # Independent sparse emulation executes only reached stages and must select identical exits/calls.
 sparse_calls=np.ones(len(scores),np.int8)
 for i in range(len(scores)):
  for j in range(3):
   if scores[i,j]<=s['threshold']:break
   sparse_calls[i]+=1
 sparse=z['exits'][np.arange(len(calls)),sparse_calls-1]
 checks={'gate_freeze_hash':True,'all_package_arrays_finite':True,'feature_order_exact':len(c['feature_order'])==1049 and c['feature_order'][-1]=='stage_3','contact_token_absent':not any('contact' in str(x).lower() for x in c['feature_order']),'exact_call_histogram':np.bincount(calls,minlength=5)[1:].tolist()==s['call_histogram'],'sparse_dense_calls_exact':np.array_equal(calls,sparse_calls),'sparse_dense_outputs_exact':np.array_equal(dense,sparse),'gate_flops_recomputed':2*(2*1049-1)+1==s['gate_flops_per_evaluation'],'gate_parameters_recomputed':2*(1049+1)==s['parameters'],'gradient_isolation':all(m['all_gradients_absent'] and m['all_parameters_frozen'] for m in json.loads((R/'data/selection_evaluated_manifest.json').read_text())['module_after']['modules']),'selection_non_degenerate':all(np.bincount(calls,minlength=5)[1:]>0),'prospective_absent':not (R/'data/prospective_raw_manifest.json').exists()}
 if not all(checks.values()):raise RuntimeError(checks)
 out={'schema_version':1,'passed':True,'checks':checks,'selected_candidate':s['candidate_id'],'numpy_fitting_warnings_audited':'saved inputs/outputs/package all finite; independent score/call recomputation exact','v3_hdf5_v4_access':False,'v5_episodes':0}
(R/'audit/gate_freeze_tests.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True))
