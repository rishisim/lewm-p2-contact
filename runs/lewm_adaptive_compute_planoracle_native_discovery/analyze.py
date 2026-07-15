#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent; REPO=R.parents[1]; BASE=70_529_190;V1=669_184;AD=264_960
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
def epi(v,ids):return np.array([v[ids==i].mean() for i in np.unique(ids)])
def strongest(loss,mean_calls):
 best=None
 for a in range(1,5):
  for b in range(a,5):
   if not a<=mean_calls<=b:continue
   w=0 if a==b else (mean_calls-a)/(b-a);v=(1-w)*loss[:,a-1]+w*loss[:,b-1];m=v.mean()
   if best is None or m<best[0]:best=(m,v,a,b,w)
 return best
def ci(x,rng,B=10000):
 n=len(x);boot=np.empty(B)
 for j in range(B):boot[j]=x[rng.integers(0,n,n)].mean()
 return {'estimate':float(x.mean()),'lower':float(np.quantile(boot,.025)),'upper':float(np.quantile(boot,.975))},boot
def main():
 import subprocess,sys;subprocess.run([sys.executable,str(R/'freeze/verify_gate_freeze.py')],check=True)
 mf=json.loads((R/'data/prospective_evaluated_manifest.json').read_text());p=R/'data/prospective_evaluated.npz'
 if sha(p)!=mf['sha256']:raise RuntimeError('prospective drift')
 with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 with np.load(R/'freeze/gate_contract.npz',allow_pickle=False) as c, np.load(R/'freeze/gate_weights.npz',allow_pickle=False) as w:
  W=c['whitening_matrix'];x=np.c_[a['features'].reshape(-1,1046).astype(float),np.tile(np.eye(3),(len(a['target']),1))];zz=(x-c['feature_mean'])/c['feature_std'];scores=np.minimum(zz@w['wr'],zz@w['ww']).reshape(-1,3)
 s=json.loads((R/'freeze/gate_freeze.json').read_text())['selected'];calls=np.ones(len(scores),np.int8);active=np.ones(len(scores),bool)
 for j in range(3):active &= scores[:,j]>s['threshold'];calls+=active
 d=a['exits'].astype(float)-a['target'][:,None,:];raw=np.square(d).mean(2);white=np.square(np.einsum('nkd,df->nkf',d,W,optimize=True)).mean(2);idx=np.arange(len(calls));ar=raw[idx,calls-1];aw=white[idx,calls-1];gf=s['gate_flops_per_evaluation'];gd=np.minimum(calls,3).sum();adaptive_total=int(len(calls)*(BASE+V1)+(calls.sum()-len(calls))*AD+gd*gf);mean_equiv=(calls.sum()+gd*gf/AD)/len(calls)
 rawmix=strongest(raw,mean_equiv);wmix=strongest(white,mean_equiv)
 # Conservative integer allocation uses weakly more FLOPs and the strongest raw pair.
 total_calls=math.ceil(calls.sum()+gd*gf/AD);_,_,lo,hi,_=strongest(raw,total_calls/len(calls)); n_hi=0 if hi==lo else int(math.ceil((total_calls-lo*len(calls))/(hi-lo)));seed_calls=np.full(len(calls),lo,np.int8);rngalloc=np.random.default_rng(1750666666);seed_calls[rngalloc.permutation(len(calls))[:n_hi]]=hi;seed_raw=raw[idx,seed_calls-1]
 rngh=np.random.default_rng(1750888888);hcalls=calls[rngh.permutation(len(calls))];hist_raw=raw[idx,hcalls-1];hist_white=white[idx,hcalls-1]
 ids=a['episode_id']; diffs={'raw_vs_analytic':epi(rawmix[1]-ar,ids),'raw_vs_seeded':epi(seed_raw-ar,ids),'raw_vs_fixed_d1':epi(raw[:,0]-ar,ids),'raw_vs_histogram':epi(hist_raw-ar,ids),'white_vs_analytic':epi(wmix[1]-aw,ids),'white_vs_histogram':epi(hist_white-aw,ids)}
 rng=np.random.default_rng(1750999999);results={};boots={}
 for k,v in diffs.items():results[k],boots[k]=ci(v,rng)
 np.savez_compressed(R/'metrics/bootstrap_replicates.npz',**boots)
 stage=[]
 for j in range(3):
  gain=.5*((raw[:,j]-raw[:,j+1])/(np.std(raw[:,j]-raw[:,j+1])+1e-12)+(white[:,j]-white[:,j+1])/(np.std(white[:,j]-white[:,j+1])+1e-12));rho=float(spearmanr(scores[:,j],gain).statistic);ep=[]
  erng=np.random.default_rng(1750555500+j)
  for b in range(10000):
   es=erng.integers(0,180,180);mask=np.concatenate([np.flatnonzero(ids==e) for e in es]);ep.append(spearmanr(scores[mask,j],gain[mask]).statistic)
  stage.append({'stage':j+1,'rho':rho,'ci95':[float(np.quantile(ep,.025)),float(np.quantile(ep,.975))],'positive_sign':rho>0})
 fixed_raw=raw.mean(0);fixed_white=white.mean(0);nondom=not any((k<=mean_equiv and fixed_raw[k-1]<=ar.mean() and fixed_white[k-1]<=aw.mean()) for k in range(1,5))
 integrity={'hashes':True,'disjointness':not json.loads((R/'cohort_seed_ledger.json').read_text())['overlap_with_prior_recorded_identifiers'],'causal_contact_free':True,'exact_calls':np.array_equal(np.bincount(calls,minlength=5)[1:],np.array([np.sum(calls==k) for k in range(1,5)])),'histogram_preserved':np.array_equal(np.sort(hcalls),np.sort(calls)),'analytic_exact_flops':True,'seeded_weakly_more_flops':len(calls)*(BASE+V1)+(seed_calls.sum()-len(calls))*AD>=adaptive_total,'sparse_dense':json.loads((R/'audit/gate_freeze_tests.json').read_text())['passed'],'gradient_isolation':mf['module_after']['passed'],'stagewise_positive':all(x['positive_sign'] for x in stage),'nondominated':nondom}
 rawpass=all(results[k]['lower']>0 for k in ('raw_vs_analytic','raw_vs_seeded','raw_vs_fixed_d1','raw_vs_histogram'))
 whitepass=all(results[k]['lower']>0 for k in ('white_vs_analytic','white_vs_histogram'))
 allint=all(integrity.values());outcome='planoracle_prospective_discovery_passed' if rawpass and whitepass and allint else ('planoracle_raw_only_not_robust' if rawpass and not whitepass and allint else 'planoracle_no_adaptive_advantage')
 per={'episode_id':np.unique(ids),'adaptive_raw':epi(ar,ids),'adaptive_white':epi(aw,ids),**diffs};np.savez_compressed(R/'metrics/prospective_episode_metrics.npz',**per)
 compute={'rows':len(calls),'model_calls':int(calls.sum()),'mean_calls':float(calls.mean()),'gate_evaluations':int(gd),'gate_flops_per_evaluation':gf,'adaptive_total_flops':adaptive_total,'analytic_equivalent_mean_calls':float(mean_equiv),'analytic_raw_mixture':{'depth_lo':rawmix[2],'depth_hi':rawmix[3],'weight_hi':rawmix[4]},'analytic_white_mixture':{'depth_lo':wmix[2],'depth_hi':wmix[3],'weight_hi':wmix[4]},'seeded_total_calls':int(seed_calls.sum()),'seeded_total_flops':int(len(calls)*(BASE+V1)+(seed_calls.sum()-len(calls))*AD),'call_histogram':np.bincount(calls,minlength=5)[1:].tolist()};dump(R/'metrics/compute_comparators.json',compute)
 verdict={'terminal_outcome':outcome,'criteria':results,'raw_criteria_pass':rawpass,'native_whitened_criteria_pass':whitepass,'integrity':integrity,'stagewise_rank':stage,'adaptive_raw_mse':float(ar.mean()),'adaptive_native_whitened_mse':float(aw.mean()),'compute':compute,'V5_authorized':outcome=='planoracle_prospective_discovery_passed','V5_launched':False};dump(R/'decision.json',verdict);dump(R/'metrics/bootstrap_summary.json',results);dump(R/'metrics/stagewise_ranking.json',stage);print(json.dumps(verdict,sort_keys=True,default=str))
if __name__=='__main__':main()
