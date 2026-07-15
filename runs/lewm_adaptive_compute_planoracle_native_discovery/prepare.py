#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, platform, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]
PRIOR=REPO/'runs/lewm_adaptive_compute_distribution_contract'
V4=REPO/'runs/lewm_adaptive_compute_v4'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def main():
 if ROOT.exists() and any(p.name != 'prepare.py' for p in ROOT.iterdir()): raise RuntimeError('run directory already initialized')
 ROOT.mkdir(parents=True,exist_ok=True); (ROOT/'audit').mkdir(); (ROOT/'data').mkdir(); (ROOT/'metrics').mkdir(); (ROOT/'freeze').mkdir()
 roles={}; base=1750000000
 specs=[('smoke',12),('fit',120),('selection',60),('prospective',180),('replacement',48)]
 k=0
 for role,n in specs:
  roles[role]=[]
  for i in range(n):
   roles[role].append({'role':role,'slot':i,'episode_id':f'pon-{role}-{i:03d}','env_seed':base+k,'policy_seed':base+100000+k,'oracle_np_seed':base+200000+k}); k+=1
 ledger={'schema_version':1,'roles':roles,'all_numeric_identifiers_unique':True,'all_episode_ids_unique':True,'prior_identifier_audit_method':'recursive scan of prior JSON manifests; byte hashes only for opaque binary artifacts','failure_replacement_rule':'only exception or mechanically malformed rollout; never outcome/loss/contact/success'}
 prior=set()
 for p in (REPO/'runs').glob('lewm_adaptive_compute*/**/*.json'):
  if ROOT in p.parents: continue
  try:
   def walk(x):
    if isinstance(x,bool): return
    if isinstance(x,int): prior.add(x)
    elif isinstance(x,dict):
     for v in x.values(): walk(v)
    elif isinstance(x,list):
     for v in x: walk(v)
   walk(json.loads(p.read_text()))
  except Exception: pass
 fresh={v for rr in roles.values() for s in rr for v in (s['env_seed'],s['policy_seed'],s['oracle_np_seed'])}
 ledger['prior_numeric_identifier_count']=len(prior); ledger['overlap_with_prior_recorded_identifiers']=sorted(fresh&prior)
 if ledger['overlap_with_prior_recorded_identifiers'] or len(fresh)!=sum(3*len(x) for x in roles.values()): raise RuntimeError('seed disjointness failed')
 dump(ROOT/'cohort_seed_ledger.json',ledger)
 grid=[]
 for family in ('stagewise_linear','shared_mlp_h16','minimax_dual_linear'):
  for reg in (1e-3,1e-2):
   for q in (0.55,0.70,0.85): grid.append({'candidate_id':f'{family}-r{reg:g}-q{q:.2f}','family':family,'regularization':reg,'threshold_quantile':q})
 dump(ROOT/'candidate_grid.json',{'schema_version':1,'candidates':grid,'tie_break':['largest minimum selection benefit (raw, native-whitened)','lower total FLOPs','fewer parameters','lexicographic candidate_id'],'maximum_hidden_units':32,'finite_candidate_count':len(grid)})
 mapping={'benefit_definition':'comparator episode mean MSE minus adaptive episode mean MSE; positive favors adaptive','bootstrap':'episode resampling, 10000 replicates, percentile [0.025,0.975], seed 1750999999','selection_qualifier':['raw benefit vs analytic exact-total-FLOP mixture > 0','native-whitened benefit vs analytic exact-total-FLOP mixture > 0','all call depths represented and mean calls strictly between 1 and 4','causal/contact/gradient/sparse-dense audits pass'],'prospective_pass':['lower CI > 0 for raw vs analytic','lower CI > 0 for raw vs conservative seeded','lower CI > 0 for raw vs fixed depth 1','lower CI > 0 for raw vs histogram randomization','lower CI > 0 for native-whitened vs analytic','lower CI > 0 for native-whitened vs histogram randomization','nondominated in calls and total FLOPs','all integrity audits pass','positive stagewise score/gain rank correlation at every stage'],'terminal_mapping':{'all_pass':'planoracle_prospective_discovery_passed','raw_all_but_native_failure':'planoracle_raw_only_not_robust','other_valid_prospective_failure':'planoracle_no_adaptive_advantage','no_selection_candidate':'gate_family_selection_failed','DGP_or_integrity_failure':'planoracle_dgp_execution_invalid'}}
 dump(ROOT/'outcome_mapping.json',mapping)
 protocol={'schema_version':1,'study':'PlanOracle-native adaptive-depth prospective discovery','DGP':{'packages':{'stable-worldmodel':'resolved at generation','ogbench':'resolved at generation','mujoco':'resolved at generation','gymnasium':'resolved at generation'},'environment':'swm/OGBCube-v0','constructor':'single-agent 224x224, 200 steps, frameskip 5','policy':'PlanOracle action_noise=.1 p_random_action=0 noise_smoothing=.5 min_norm=.4','reset':'distribution-contract deterministic reset amendment','history':3,'targets':'next-step frozen-base latent, model steps 3..40','action_normalization':'frozen released-model normalizer'},'roles':{'smoke':12,'fit':120,'selection':60,'prospective':180},'whitening':'fit targets only: centered covariance eigendecomposition with eigenvalue floor max(1e-8, 1e-4*median positive eigenvalue); matrix V diag(1/sqrt(clipped eigenvalues)) V^T; frozen before selection','features':'1046 causal features from history latents, normalized actions, current prediction, last update, scalar summaries; stage one-hot; no contact/state/future/target','frozen_objects':'base and stagewise depth-4 refiner from prior sealed runtime; inference-only, no gradients','compute':'base=70,529,190 FLOPs/row; mandatory depth1=669,184; each later adapter=264,960. Gate FLOPs derived from actual selected architecture: linear scalar head=2D-1; shared MLP=2D*H+2H+2H+1 including affine normalization and activations; dual linear=2*(2D-1)+1 minimax. Evaluate gate at each reached decision. Analytic comparator linearly interpolates adjacent fixed depths to exact adaptive total FLOPs. Conservative seeded comparator uses ceil-equivalent total calls, weakly more FLOPs. Histogram comparator independently permutes adaptive calls within episode using seed 1750888888.','contact':'post-hoc only and not collected for decision artifacts','selection':'fit candidates on fit only, thresholds are fit-score quantiles; evaluate locked selection once; require positive point benefits raw and native-whitened plus audits; fixed tie-break','prospective':'generated only after selected gate freeze seal verifies; one evaluation; no peeking/expansion/retuning','latency':'synchronized MPS after statistical verdict only; excluded from decision','forbidden':['released HDF5 access','V3 target/cache access','V4 selection or confirmation use','V5 launch','generator reconstruction/distribution matching claim','first adaptive world model claim'],'related_work':'LoopWM is relevant prior adaptive-computation work in text environments.'}
 dump(ROOT/'protocol.json',protocol)
 (ROOT/'PREREGISTRATION.md').write_text('# PlanOracle-native adaptive-computation discovery preregistration\n\nThis prospective program freezes the complete DGP, split, candidate grid, fit-only whitening, exact-FLOP comparators, bootstrap, audits, and terminal mapping before any new rollout. Existing PlanOracle/V4/HDF5 artifacts are development context only and supply no episode, whitening statistic, target, threshold, or selection outcome. Contact is excluded from all pre-freeze and decision paths. V5 will not be launched. See `protocol.json`, `cohort_seed_ledger.json`, `candidate_grid.json`, and `outcome_mapping.json` for the normative machine-readable contract.\n')
 frozen={'base_config':V4/'../lewm_transfer/cube/cache/model/config.json','base_weights':V4/'../lewm_transfer/cube/cache/model/weights.pt','solver':REPO/'runs/lewm_adaptive_compute_discovery/checkpoints/stagewise_seed_261102.pt'}
 resolved={k:{'path':str(p.resolve()),'sha256':sha(p.resolve()),'bytes':p.resolve().stat().st_size} for k,p in frozen.items()}
 dump(ROOT/'resolved_frozen_objects.json',resolved)
 sealed=[ROOT/'PREREGISTRATION.md',ROOT/'protocol.json',ROOT/'cohort_seed_ledger.json',ROOT/'candidate_grid.json',ROOT/'outcome_mapping.json',ROOT/'resolved_frozen_objects.json',Path(__file__).resolve()]
 seal={'schema_version':1,'status':'frozen_before_any_new_episode_or_fit','files':{str(p.relative_to(REPO)):sha(p) for p in sealed},'new_episode_files_at_seal':0,'released_hdf5_opened':False,'v3_targets_opened':False,'v5_episodes':0}
 dump(ROOT/'audit/pre_generation_seal.json',seal)
 verifier='''#!/usr/bin/env python3\nimport hashlib,json\nfrom pathlib import Path\nr=Path(__file__).resolve().parents[2]; s=json.loads((Path(__file__).parent/'pre_generation_seal.json').read_text())\nfor p,e in s['files'].items():\n h=hashlib.sha256((r/p).read_bytes()).hexdigest()\n if h!=e: raise SystemExit(f"FAIL {p} {h} != {e}")\nprint("pre_generation_seal_verified")\n'''
 (ROOT/'audit/verify_pre_generation.py').write_text(verifier); os.chmod(ROOT/'audit/verify_pre_generation.py',0o755)
 print(json.dumps(seal,sort_keys=True))
if __name__=='__main__': main()
