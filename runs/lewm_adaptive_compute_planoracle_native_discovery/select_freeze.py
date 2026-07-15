#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]
BASE=70_529_190; V1=669_184; ADAPTER=264_960; D=1049
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,default=float)+'\n')
def load(role):
 p=ROOT/'data'/f'{role}_evaluated.npz';m=json.loads((ROOT/'data'/f'{role}_evaluated_manifest.json').read_text())
 if sha(p)!=m['sha256']:raise RuntimeError('evaluated drift')
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def losses(a,W):
 dif=a['exits'].astype(np.float64)-a['target'][:,None,:]
 raw=np.square(dif).mean(2); white=np.square(np.einsum('nkd,df->nkf',dif,W,optimize=True)).mean(2)
 return raw,white
def design(a):
 f=a['features'].reshape(-1,1046).astype(np.float64); stage=np.tile(np.eye(3), (len(a['target']),1));return np.concatenate([f,stage],1)
def calls(scores,thr):
 s=scores.reshape(-1,3);c=np.ones(len(s),np.int8);active=np.ones(len(s),bool)
 for j in range(3):active &= s[:,j]>thr;c+=active
 return c
def gate_flops(family):return 2*D-1 if family=='stagewise_linear' else (2*D*16+2*16+2*16+1 if family=='shared_mlp_h16' else 2*(2*D-1)+1)
def analytic(loss,c,gf):
 n=len(c);gd=np.minimum(c,3).sum(); mean_calls=(c.sum()+gd*gf/ADAPTER)/n;lo=int(np.floor(mean_calls));hi=int(np.ceil(mean_calls));w=mean_calls-lo
 comp=(1-w)*loss[:,lo-1]+w*loss[:,hi-1] if hi!=lo else loss[:,lo-1]
 return comp,mean_calls,int(n*(BASE+V1)+(c.sum()-n)*ADAPTER+gd*gf)
def epi(v,ids):return np.array([v[ids==i].mean() for i in np.unique(ids)])
def main():
 fit=load('fit');sel=load('selection'); X=design(fit); Xs=design(sel)
 mean=X.mean(0);std=X.std(0);std[std<1e-6]=1.;Z=(X-mean)/std;Zs=(Xs-mean)/std
 target=fit['target'].astype(np.float64);mu=target.mean(0);cov=np.cov(target-mu,rowvar=False);ev,V=np.linalg.eigh(cov);pos=ev[ev>0];floor=max(1e-8,1e-4*np.median(pos));W=(V*(1/np.sqrt(np.maximum(ev,floor))))@V.T
 np.savez_compressed(ROOT/'fit_only_normalization_whitening.npz',feature_mean=mean,feature_std=std,target_mean=mu,whitening_matrix=W,eigenvalues=ev,eigenvalue_floor=floor)
 fr,fw=losses(fit,W);sr,sw=losses(sel,W);yr=(fr[:,:-1]-fr[:,1:]).reshape(-1);yw=(fw[:,:-1]-fw[:,1:]).reshape(-1)
 # Equalize scales so the joint scorer cannot ignore either declared objective.
 yavg=.5*(yr/(np.std(yr)+1e-12)+yw/(np.std(yw)+1e-12)); grid=json.loads((ROOT/'candidate_grid.json').read_text())['candidates']; ledger=[]; weights={}
 rng=np.random.default_rng(1750777777); R=rng.normal(0,1/np.sqrt(D),size=(D,16)); rb=rng.normal(0,.01,size=16); H=np.tanh(Z@R+rb); Hs=np.tanh(Zs@R+rb)
 import torch
 def ridge(A,y,reg):
  at=torch.tensor(A,dtype=torch.float64);yt=torch.tensor(y,dtype=torch.float64);I=torch.eye(A.shape[1],dtype=torch.float64);return torch.linalg.solve(at.T@at+reg*I,at.T@yt).numpy()
 cache={}
 for cnd in grid:
  fam=cnd['family'];reg=cnd['regularization']; key=(fam,reg)
  if key not in cache:
   if fam=='stagewise_linear': w=ridge(Z,yavg,reg); pred=Z@w; ps=Zs@w; state={'w':w}
   elif fam=='shared_mlp_h16': w=ridge(np.c_[H,np.ones(len(H))],yavg,reg);pred=np.c_[H,np.ones(len(H))]@w;ps=np.c_[Hs,np.ones(len(Hs))]@w;state={'R':R,'rb':rb,'w':w}
   else:
    wr=ridge(Z,yr/(np.std(yr)+1e-12),reg);ww=ridge(Z,yw/(np.std(yw)+1e-12),reg);pred=np.minimum(Z@wr,Z@ww);ps=np.minimum(Zs@wr,Zs@ww);state={'wr':wr,'ww':ww}
   cache[key]=(pred,ps,state)
  pred,ps,state=cache[key];thr=float(np.quantile(pred,cnd['threshold_quantile']));cc=calls(ps,thr);gf=gate_flops(fam); ar,mc,total=analytic(sr,cc,gf);aw,_,_=analytic(sw,cc,gf);idx=np.arange(len(cc));ad_r=sr[idx,cc-1];ad_w=sw[idx,cc-1];br=epi(ar-ad_r,sel['episode_id']);bw=epi(aw-ad_w,sel['episode_id']);hist=np.bincount(cc,minlength=5)[1:].tolist();nondeg=all(x>0 for x in hist) and 1<cc.mean()<4
  rec={**cnd,'threshold':thr,'parameters':int(D+1 if fam=='stagewise_linear' else D*16+16+17 if fam=='shared_mlp_h16' else 2*(D+1)),'gate_flops_per_evaluation':gf,'selection_raw_benefit':float(br.mean()),'selection_native_whitened_benefit':float(bw.mean()),'mean_calls':float(cc.mean()),'total_flops':total,'call_histogram':hist,'nondegenerate':nondeg,'qualified':bool(br.mean()>0 and bw.mean()>0 and nondeg),'minimum_benefit':float(min(br.mean(),bw.mean()))}
  ledger.append(rec);weights[cnd['candidate_id']]=state
 dump(ROOT/'metrics/candidate_selection_ledger.json',{'candidates':ledger,'selection_episodes':60,'fit_episodes':120,'targets_not_used_in_features':True})
 qualified=[x for x in ledger if x['qualified']]
 if not qualified:
  dump(ROOT/'decision.json',{'terminal_outcome':'gate_family_selection_failed','prospective_generated':False,'V5_authorized':False});print('gate_family_selection_failed');return
 qualified.sort(key=lambda x:(-x['minimum_benefit'],x['total_flops'],x['parameters'],x['candidate_id']));chosen=qualified[0];state=weights[chosen['candidate_id']]
 np.savez_compressed(ROOT/'freeze/gate_weights.npz',**state)
 feature_order=np.array([f'causal_{i}' for i in range(1046)]+['stage_1','stage_2','stage_3'])
 np.savez_compressed(ROOT/'freeze/gate_contract.npz',feature_mean=mean,feature_std=std,target_mean=mu,whitening_matrix=W,feature_order=feature_order)
 package={'schema_version':1,'status':'frozen_before_prospective_generation','selected':chosen,'decision_code_sha256':sha(Path(__file__)),'gate_weights_sha256':sha(ROOT/'freeze/gate_weights.npz'),'gate_contract_sha256':sha(ROOT/'freeze/gate_contract.npz'),'fit_only_whitening_sha256':sha(ROOT/'fit_only_normalization_whitening.npz'),'seed_ledger_sha256':sha(ROOT/'cohort_seed_ledger.json'),'selection_ledger_sha256':sha(ROOT/'metrics/candidate_selection_ledger.json'),'base_and_refiner_hashes':json.loads((ROOT/'resolved_frozen_objects.json').read_text()),'causal_feature_order':feature_order.tolist(),'contact_excluded':True,'v3_hdf5_v4_used':False,'v5_episodes':0}
 dump(ROOT/'freeze/gate_freeze.json',package)
 files=[ROOT/'freeze/gate_freeze.json',ROOT/'freeze/gate_weights.npz',ROOT/'freeze/gate_contract.npz',ROOT/'fit_only_normalization_whitening.npz',ROOT/'cohort_seed_ledger.json',ROOT/'metrics/candidate_selection_ledger.json',Path(__file__)]
 dump(ROOT/'freeze/gate_freeze_seal.json',{'status':'immutable_gate_freeze_before_prospective','files':{str(p.relative_to(REPO)):sha(p) for p in files}})
 verifier='''#!/usr/bin/env python3\nimport hashlib,json\nfrom pathlib import Path\nr=Path(__file__).resolve().parents[3];s=json.loads((Path(__file__).parent/'gate_freeze_seal.json').read_text())\nfor p,e in s['files'].items():\n if hashlib.sha256((r/p).read_bytes()).hexdigest()!=e:raise SystemExit('FAIL '+p)\nprint('gate_freeze_verified')\n''';(ROOT/'freeze/verify_gate_freeze.py').write_text(verifier);os.chmod(ROOT/'freeze/verify_gate_freeze.py',0o755)
 print(json.dumps(chosen,sort_keys=True))
if __name__=='__main__':main()
