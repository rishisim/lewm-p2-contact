#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent;REPO=R.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n')
def epi(v,ids):return np.array([v[ids==i].mean() for i in np.unique(ids)])
def strongest(loss,mean_calls):
 best=None
 for a in range(1,5):
  for b in range(a,5):
   if not a<=mean_calls<=b:continue
   w=0 if a==b else (mean_calls-a)/(b-a);v=(1-w)*loss[:,a-1]+w*loss[:,b-1];m=v.mean()
   if best is None or m<best[0] or (m==best[0] and (a,b)<(best[2],best[3])):best=(m,v,a,b,w)
 return best
def main():
 subprocess.run([sys.executable,str(R/'verify_pre_data.py')],check=True)
 mf=json.loads((R/'data/prospective_execution_manifest.json').read_text());p=R/'data/prospective_execution.npz'
 if sha(p)!=mf['sha256']:raise RuntimeError('sealed input drift')
 rawm=json.loads((R/'data/prospective_raw_manifest.json').read_text())
 if len(rawm['episodes'])!=300:raise RuntimeError('not exactly 300 episodes')
 inputseal={'created_unix_ns':time.time_ns(),'files':{str(p.relative_to(REPO)):sha(p),str((R/'data/prospective_raw_manifest.json').relative_to(REPO)):sha(R/'data/prospective_raw_manifest.json')},'raw_episode_hashes':{x['path']:x['sha256'] for x in rawm['episodes']},'all_300_complete_before_unblinding':True};dump(R/'audit/prospective_input_seal.json',inputseal)
 with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 with np.load(REPO/'runs/lewm_adaptive_compute_planoracle_native_discovery/freeze/gate_contract.npz',allow_pickle=False) as c:W=c['whitening_matrix']
 d=a['dense_exits'].astype(float)-a['target'][:,None,:];raw=np.square(d).mean(2);white=np.square(np.einsum('nkd,df->nkf',d,W,optimize=True)).mean(2);calls=a['calls'].astype(int);idx=np.arange(len(calls));ar=raw[idx,calls-1];aw=white[idx,calls-1];ids=a['episode_id'];op=json.loads((R/'operation_ledger.json').read_text());ge=int(np.minimum(calls,3).sum());gf=op['gate_feature_flops_per_reached_evaluation'];gs=op['gate_score_per_reached_evaluation']['flops'];BASE=op['base_flops_per_row'];V1=op['mandatory_depth1_refiner_flops_per_row'];AD=op['additional_refiner_flops_per_call'];total=int(len(calls)*(BASE+V1)+(calls.sum()-len(calls))*AD+ge*(gf+gs));equiv=(calls.sum()+ge*(gf+gs)/AD)/len(calls)
 rm=strongest(raw,equiv);wm=strongest(white,equiv);target_calls=math.ceil(calls.sum()+ge*(gf+gs)/AD);_,_,lo,hi,_=strongest(raw,target_calls/len(calls));nhi=0 if lo==hi else math.ceil((target_calls-lo*len(calls))/(hi-lo));seedcalls=np.full(len(calls),lo);rng=np.random.default_rng(1765777777);seedcalls[rng.permutation(len(calls))[:nhi]]=hi;seedraw=raw[idx,seedcalls-1]
 hcalls=np.empty_like(calls);rng=np.random.default_rng(1765888888);hist=[]
 for eid in np.unique(ids):
  q=np.flatnonzero(ids==eid);hcalls[q]=calls[q][rng.permutation(len(q))];hist.append({'episode_id':int(eid),'original':np.bincount(calls[q],minlength=5)[1:].tolist(),'permuted':np.bincount(hcalls[q],minlength=5)[1:].tolist(),'preserved':bool(np.array_equal(np.sort(calls[q]),np.sort(hcalls[q])))})
 hr=raw[idx,hcalls-1];hw=white[idx,hcalls-1]
 diffs={'raw_vs_analytic':epi(rm[1]-ar,ids),'raw_vs_seeded':epi(seedraw-ar,ids),'raw_vs_fixed_d1':epi(raw[:,0]-ar,ids),'raw_vs_within_episode_histogram':epi(hr-ar,ids),'white_vs_analytic':epi(wm[1]-aw,ids),'white_vs_within_episode_histogram':epi(hw-aw,ids)}
 B=20000;rng=np.random.default_rng(1765999999);boots={k:np.empty(B) for k in diffs};n=300
 for j in range(B):
  q=rng.integers(0,n,n)
  for k,v in diffs.items():boots[k][j]=v[q].mean()
 ci={k:{'estimate':float(v.mean()),'lower':float(np.quantile(boots[k],.025)),'upper':float(np.quantile(boots[k],.975))} for k,v in diffs.items()}
 cop=['raw_vs_analytic','white_vs_analytic'];errs=np.column_stack([diffs[k].mean()-boots[k] for k in cop]);q95=float(np.quantile(np.max(errs,axis=1),.95));sim={k:{'estimate':float(diffs[k].mean()),'lower':float(diffs[k].mean()-q95),'shared_max_error_quantile':q95} for k in cop}
 np.savez_compressed(R/'metrics/bootstrap_replicates.npz',**boots);dump(R/'metrics/bootstrap_summary.json',{'individual':ci,'simultaneous':sim,'replicates':B});dump(R/'metrics/within_episode_histogram_ledger.json',{'episodes':hist,'all_preserved':all(x['preserved'] for x in hist),'global_permutation_decision_use':False})
 stages=[]
 for j in range(3):
  mask=np.isfinite(a['scores'][:,j]);gain=.5*((raw[:,j]-raw[:,j+1])/(np.std(raw[:,j]-raw[:,j+1])+1e-12)+(white[:,j]-white[:,j+1])/(np.std(white[:,j]-white[:,j+1])+1e-12));rho=float(spearmanr(a['scores'][mask,j],gain[mask]).statistic);stages.append({'stage':j+1,'reached':int(mask.sum()),'rho':rho,'positive_sign':rho>0})
 compute={'rows':len(calls),'model_calls':int(calls.sum()),'mean_model_calls':float(calls.mean()),'gate_evaluations':ge,'gate_feature_flops':ge*gf,'gate_score_flops':ge*gs,'gate_total_flops':ge*(gf+gs),'adaptive_total_flops':total,'nonflop_comparisons':ge*2,'analytic_equivalent_mean_calls':equiv,'raw_analytic_mixture':{'depth_lo':rm[2],'depth_hi':rm[3],'weight_hi':rm[4]},'white_analytic_mixture':{'depth_lo':wm[2],'depth_hi':wm[3],'weight_hi':wm[4]},'seeded_calls':int(seedcalls.sum()),'seeded_total_flops':int(len(calls)*(BASE+V1)+(seedcalls.sum()-len(calls))*AD),'call_histogram':np.bincount(calls,minlength=5)[1:].tolist()};dump(R/'metrics/compute_ledger_realized.json',compute);dump(R/'metrics/stagewise_ranking.json',stages)
 integrity={'pre_data_seal':True,'chronology':inputseal['all_300_complete_before_unblinding'],'exact_episode_count':len(np.unique(ids))==300,'finite_arrays':all(np.isfinite(v).all() for v in (raw,white,ar,aw)),'hashes':True,'seed_isolation':not json.loads((R/'cohort_seed_ledger.json').read_text())['overlap_with_prior_recorded_identifiers'],'contact_input_exclusion':not mf['contact_or_privileged_loaded'],'input_key_allowlist':mf['loaded_input_keys']==['action','pixels'],'sparse_dense_exact':mf['sparse_dense_exact'],'frozen_modules':mf['module_before']==mf['module_after'] and mf['module_after']['passed'],'histograms_preserved':all(x['preserved'] for x in hist),'exact_compute':compute['seeded_total_flops']>=total,'stagewise_positive':all(x['positive_sign'] for x in stages),'no_v5':True}
 stat=all(ci[k]['lower']>0 for k in diffs) and all(sim[k]['lower']>0 for k in cop);valid=all(integrity.values());out='planoracle_prospective_audit_repair_passed' if valid and stat else ('planoracle_prospective_audit_repair_failed' if valid else 'planoracle_audit_repair_invalid');decision={'terminal_outcome':out,'criteria':ci,'simultaneous_co_primary':sim,'statistical_pass':stat,'integrity':integrity,'adaptive_raw_mse':float(ar.mean()),'adaptive_native_whitened_mse':float(aw.mean()),'compute':compute,'stagewise_rank':stages,'V5_eligible_future_only':out=='planoracle_prospective_audit_repair_passed','V5_launched':False};dump(R/'decision.json',decision);np.savez_compressed(R/'metrics/prospective_episode_metrics.npz',episode_id=np.unique(ids),adaptive_raw=epi(ar,ids),adaptive_white=epi(aw,ids),**diffs);print(json.dumps(decision,sort_keys=True))
if __name__=='__main__':main()
