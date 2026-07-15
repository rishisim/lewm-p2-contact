#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent; REPO=R.parents[1]
SRC=REPO/'runs/lewm_adaptive_compute_planoracle_native_discovery'
DIST=REPO/'runs/lewm_adaptive_compute_distribution_contract'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def ints(v):
 out=set()
 if isinstance(v,bool):return out
 if isinstance(v,int):out.add(v)
 elif isinstance(v,dict):
  for x in v.values():out|=ints(x)
 elif isinstance(v,list):
  for x in v:out|=ints(x)
 return out

def equivalence():
 with np.load(SRC/'freeze/gate_contract.npz',allow_pickle=False) as c, np.load(SRC/'freeze/gate_weights.npz',allow_pickle=False) as w:
  mean=c['feature_mean'].astype(np.float64);std=c['feature_std'].astype(np.float64);wr=w['wr'].astype(np.float64);ww=w['ww'].astype(np.float64)
 ar=(wr/std).astype(np.float32);br=np.float32(-(mean/std)@wr);aw=(ww/std).astype(np.float32);bw=np.float32(-(mean/std)@ww)
 rows=[];ok=True;minmargin=float('inf')
 threshold=0.18128031821449414
 for role in ('fit','selection','prospective'):
  with np.load(SRC/f'data/{role}_evaluated.npz',allow_pickle=False) as z:f=z['features'].reshape(-1,1046).astype(np.float64);n=len(z['target'])
  x=np.c_[f,np.tile(np.eye(3), (n,1))]
  ref=np.minimum(((x-mean)/std)@wr,((x-mean)/std)@ww).reshape(n,3)
  xf=x.astype(np.float32);fold=np.minimum(xf@ar+br,xf@aw+bw).reshape(n,3)
  def calls(s):
   q=np.ones(len(s),np.int8);active=np.ones(len(s),bool)
   for j in range(3):active&=s[:,j]>threshold;q+=active
   return q
  cr,cf=calls(ref),calls(fold);same=np.array_equal(cr,cf);ok&=same;minmargin=min(minmargin,float(np.min(np.abs(ref-threshold))))
  rows.append({'role':role,'rows':n,'stage_decisions_equal':bool(np.array_equal(ref>threshold,fold>threshold)),'calls_equal':same,'selected_exits_equal':same,'call_histogram_equal':bool(np.array_equal(np.bincount(cr,minlength=5),np.bincount(cf,minlength=5))),'max_abs_score_delta':float(np.max(np.abs(ref-fold)))})
 np.savez(R/'freeze/compiled_gate.npz',a_raw=ar,b_raw=br,a_white=aw,b_white=bw,threshold=np.float64(threshold))
 led={'schema_version':1,'branch_rule':'compiled float32 affine iff exact decisions/calls/exits/histograms on all consumed fit, selection, prior prospective rows','chosen':'compiled_float32_folded_affine' if ok else 'explicit_float32_standardization','dtype':'float32','all_exact':ok,'minimum_float64_threshold_margin':minmargin,'datasets':rows,'no_tuning':True}
 dump(R/'audit/implementation_equivalence.json',led)
 if not ok:raise RuntimeError('fallback required but not implemented fail-closed')

def main():
 if (R/'audit/pre_data_seal.json').exists():raise RuntimeError('already sealed')
 for p,e in {'freeze/gate_weights.npz':'a9e42f735a55b10e8bdac594cc0cc152cde0b66781fa942749e5c903dc5a33a8','freeze/gate_contract.npz':'2a953b7f441098c552fa8d725e581ab570c8692b3747b409004e57844520a869','fit_only_normalization_whitening.npz':'515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796'}.items():
  q=SRC/p
  if sha(q)!=e:raise RuntimeError('source artifact drift '+p)
 (R/'audit').mkdir(parents=True,exist_ok=True);(R/'freeze').mkdir(exist_ok=True);(R/'data').mkdir(exist_ok=True);(R/'metrics').mkdir(exist_ok=True)
 equivalence()
 prior=set()
 for p in (REPO/'runs').glob('**/*.json'):
  if R in p.parents:continue
  try:prior|=ints(json.loads(p.read_text()))
  except:pass
 base=1765000000
 roles={}
 for role,n,off in [('smoke',12,0),('prospective',300,10000),('replacement',60,50000)]:
  roles[role]=[{'role':role,'slot':i,'episode_id':f'par-{role[:4]}-{i:03d}','env_seed':base+off+i,'policy_seed':base+100000+off+i,'oracle_np_seed':base+200000+off+i} for i in range(n)]
 new={x[k] for a in roles.values() for x in a for k in ('env_seed','policy_seed','oracle_np_seed')}
 overlap=sorted(new&prior)
 if overlap:raise RuntimeError(f'prior seed overlap {overlap[:5]}')
 ledger={'schema_version':1,'roles':roles,'all_identifiers_unique':len(new)==3*sum(map(len,roles.values())),'prior_numeric_identifier_count':len(prior),'overlap_with_prior_recorded_identifiers':overlap,'replacement_rule':'only exception or mechanically malformed rollout; never outcome, loss, contact, or success; replacements consumed in order','bootstrap_seed':1765999999,'histogram_seed':1765888888,'seeded_mixture_seed':1765777777,'stagewise_bootstrap_seeds':[1765666601,1765666602,1765666603]}
 dump(R/'cohort_seed_ledger.json',ledger)
 protocol={'schema_version':1,'study':'PlanOracle native audit-repair prospective discovery','not_v5':True,'selected_gate':'minimax_dual_linear-r0.01-q0.85','threshold':0.18128031821449414,'sample_sizes':{'excluded_smoke':12,'prospective':300},'rows_per_episode':38,'dgp':{'stable-worldmodel':'0.1.0','ogbench':'1.2.1','mujoco':'3.10.0','gymnasium':'1.3.0','environment':'swm/OGBCube-v0','agents':1,'image':[224,224],'environment_steps':200,'frameskip':5,'history':3,'model_steps':[3,40],'policy':'PlanOracle','action_noise':0.1,'p_random_action':0,'noise_smoothing':0.5,'min_norm':0.4,'deterministic_reset_amendment':True},'input_key_allowlist':['pixels','action'],'forbidden_gate_model_keys':['contact','observation','qpos','qvel','privileged_target_block_pos','privileged_target_block_yaw','reward','terminated','truncated'],'unblinding':'only after 300 raw episodes, sparse traces, and dense shadows complete and input-hash sealed','bootstrap':{'unit':'episode','replicates':20000,'interval':'equal-tailed percentile 95%','tie':'strict lower bound > 0','simultaneous':'bootstrap max-t: use 95th percentile of max of centered estimate-minus-bootstrap-mean errors for raw and whitened analytic contrasts; lower=estimate-q95'},'criteria':['raw_vs_analytic','raw_vs_seeded','raw_vs_fixed_d1','raw_vs_within_episode_histogram','white_vs_analytic','white_vs_within_episode_histogram'],'stagewise_rank':'Spearman score/gain rho must be >0 at each reached stage; ties use scipy average ranks','latency':'post-decision only; synchronized MPS; full sparse path','contact_posthoc_only':True,'exclusions':'only predeclared smoke and mechanical generation corruption; no outcome exclusions','stopping':'exactly 300 valid prospective episodes; no sequential expansion'}
 dump(R/'protocol.json',protocol)
 outcome={'pass':'planoracle_prospective_audit_repair_passed','failure':'planoracle_prospective_audit_repair_failed','invalid':'planoracle_audit_repair_invalid','pass_requires':'all six individual lower bounds, both simultaneous analytic lower bounds, stagewise signs, and every integrity check','failure_when':'valid sealed run but statistical criterion fails','invalid_when':'any seal, DGP, chronology, isolation, sparse/dense, hash, or exact-compute failure'};dump(R/'outcome_mapping.json',outcome)
 prereg='''# Preregistration: PlanOracle native audit-repair\n\nThis is a prospective audit-repair discovery run, not V5 or confirmation. The source minimax dual-linear gate, normalization, whitening, threshold, features, model, solver, and prices are immutable. Twelve post-seal smoke episodes are excluded. Exactly 300 fresh PlanOracle episodes are generated from the sealed ledger. The evaluator loads only `pixels` and `action`; contact and privileged values cannot enter execution, exclusions, stopping, statistics, or terminal mapping.\n\nAll raw generation, sparse online execution, dense shadows, and their hashes must complete before loss or gate-outcome analysis. The adaptive output comes from the actual sparse stage loop; dense execution is comparator-only. Six individual episode-bootstrap lower bounds and simultaneous raw/whitened analytic lower bounds must exceed zero, with all integrity checks, for a pass. A valid statistical miss is failure; any process defect is invalid. No tuning, candidate search, or V5 launch is permitted.\n''';(R/'PREREGISTRATION.md').write_text(prereg)
 op={'schema_version':1,'convention':'one scalar add, subtract, multiply, divide, or comparison is one FLOP except comparisons/min are reported non-FLOP; tensor views/concatenation/indexing are zero FLOP','base_flops_per_row':70529190,'mandatory_depth1_refiner_flops_per_row':669184,'additional_refiner_flops_per_call':264960,'feature_graph':{'history_flatten_view':0,'actions_flatten_view':0,'current_view':0,'last_update_view':0,'scalar_summaries':{'current_mean':192,'current_std_population':576,'update_mean':192,'update_std_population':576,'current_l2_rms':576,'update_l2_rms':576,'current_abs_mean':384,'update_abs_mean':384,'current_update_dot_mean':384,'update_abs_max':192,'total':4032},'concatenate_and_depth_onehot':0},'gate_score_per_reached_evaluation':{'two_affine_heads_each_1049_multiplies_1048_adds_plus_bias':4196,'min_nonflop':1,'threshold_comparison_nonflop':1,'flops':4196},'gate_feature_flops_per_reached_evaluation':4032,'gate_total_flops_per_reached_evaluation':8228,'biases_explicitly_counted':2,'independent_symbolic_check':'2*(1049 multiplications + 1048 reduction adds + 1 bias add)=4196; features=4032 from sealed scalar expressions','nonflop_per_gate_evaluation':2};dump(R/'operation_ledger.json',op)
 report='''# REPORT\n\nStatus: pre-data frozen; results not yet evaluated.\n\n## Facts\n\nSee machine-readable artifacts.\n\n## Interpretations\n\nPending the one prospective evaluation.\n\n## Process validity\n\nThe pre-data seal and chronology audit govern validity.\n\n## Limitations\n\nOne simulator/model distribution; discovery evidence rather than confirmation.\n\n## Outcome mapping\n\nSee `outcome_mapping.json`.\n''';(R/'REPORT.md').write_text(report)
 local=[R/x for x in ('prepare.py','runner.py','analyze.py','verify_pre_data.py','verify_final.py','finalize.py','PREREGISTRATION.md','REPORT.md','protocol.json','outcome_mapping.json','cohort_seed_ledger.json','operation_ledger.json','audit/implementation_equivalence.json','freeze/compiled_gate.npz')]
 deps=[DIST/'common.py',DIST/'generator.py',DIST/'generator_seedfix.py',REPO/'runs/lewm_adaptive_compute_v4/common.py',REPO/'runs/lewm_adaptive_compute_v4/runtime.py',REPO/'runs/lewm_adaptive_compute_v2/model_io.py',REPO/'runs/lewm_adaptive_compute_discovery/models.py',REPO/'runs/lewm_adaptive_compute_discovery/run_discovery.py',REPO/'runs/lewm_adaptive_compute_discovery/critics.py',REPO/'runs/lewm_adaptive_compute_discovery/policy.py']
 objs=[SRC/'freeze/gate_weights.npz',SRC/'freeze/gate_contract.npz',SRC/'fit_only_normalization_whitening.npz',REPO/'runs/lewm_transfer/cube/cache/model/config.json']
 files=local+deps+objs
 missing=[str(x) for x in files if not x.exists()]
 if missing:raise RuntimeError('missing seal inputs '+str(missing))
 versions={}
 for n in ('numpy','scipy','torch','stable-worldmodel','ogbench','mujoco','gymnasium'):
  try:versions[n]=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError:versions[n]=None
 manifest={'schema_version':1,'files':{str(p.relative_to(REPO)):sha(p) for p in files},'packages':versions,'frozen_object_expected_hashes':{'base_weights':'2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89','base_config':'4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999','stagewise_refiner':'e63277943a356f3e28b4c4d1a1eb56acc8771fc07eccccdca67f878fed5ba782'},'actual_import_trace_scope':'all transitive local sources loaded by runner through distribution common/runtime/discovery modules'};dump(R/'transitive_dependency_manifest.json',manifest)
 files.append(R/'transitive_dependency_manifest.json')
 raw=list((R/'data').glob('**/*.npz'))
 if raw:raise RuntimeError('raw episode existed at seal time')
 seal={'schema_version':1,'status':'frozen_before_any_smoke_or_prospective_data','created_unix_ns':time.time_ns(),'files':{str(p.relative_to(REPO)):sha(p) for p in files},'zero_new_raw_episode_files_at_seal':True,'zero_count':0,'prospective_targets_opened':False,'v3_test_targets_opened':False,'hdf5_opened':False,'v5_episodes':0}
 dump(R/'audit/pre_data_seal.json',seal);os.chmod(R/'audit/pre_data_seal.json',0o444)
 print(json.dumps({'sealed_files':len(files),'seal_sha256':sha(R/'audit/pre_data_seal.json'),'equivalence':'compiled_float32_folded_affine'}))
if __name__=='__main__':main()
