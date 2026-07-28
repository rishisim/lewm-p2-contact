#!/usr/bin/env python3
"""Generate Task F banks, fit, select, and evaluate without reopening roles."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import gymnasium as gym
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]; WORK=ROOT/"work/task_f"
sys.path[:0]=[str(REPO/"le-wm"),str(ROOT)]
import stable_worldmodel as swm
from omegaconf import OmegaConf
import eval as lewm_eval
from pusht_cem_adapter import PushTRefinedCostModel
from run_task_b import load_base
from task_c_population import PrefixComparableCEMSolver, PopulationConfig
from task_e import array_sha256
from run_task_e import CaptureModel, configure_solver, label_candidate, make_info, restore_env
from task_f import LinearCritic, choice_comparison, feature_matrix, fit_linear, ranking_metrics

def atomic_json(path:Path,value:dict):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n"); tmp.replace(path)
def config(): return json.loads((ROOT/"task_f_config.json").read_text())
def manifest(): return json.loads((ROOT/"task_f_manifest.json").read_text())
def device():
 if not torch.backends.mps.is_available(): raise RuntimeError("Task F inherits the verified MPS stack")
 return torch.device("mps")
def paths(role,row,seed):
 stem=WORK/role/f"row{row}_seed{seed}"; return stem.with_suffix(".npz"),stem.with_suffix(".json")
def native_scores(info,candidates,dev):
 model=PushTRefinedCostModel(load_base(dev),None,0).to(dev).eval()
 tensor=torch.from_numpy(candidates[None]).to(dev); expanded={k:v.unsqueeze(1).expand(1,len(candidates),*v.shape[1:]) for k,v in info.items()}
 with torch.inference_mode(): value=model.get_cost(expanded,tensor)
 out=value.detach().cpu().numpy()[0].astype(np.float64)
 if not np.isfinite(out).all(): raise RuntimeError("non-finite native cost")
 return out
def generate(role):
 cfg=config(); man=manifest()
 if role=="evaluation" and not (WORK/"selected_model.json").exists(): raise RuntimeError("evaluation role remains unopened until selection artifact is frozen")
 ds=lewm_eval.get_dataset(OmegaConf.load(REPO/"le-wm/config/eval/pusht.yaml"),json.loads((ROOT/"config.json").read_text())["planner_qualification"]["dataset"]["name"]); transform=lewm_eval.processors["pusht"]
 dev=device()
 for entry in man[role]:
  for seed in man["candidate_seeds"]:
   npz,meta=paths(role,entry["row_id"],seed)
   if npz.exists() and meta.exists(): continue
   info=make_info(ds,transform,entry,dev); base=PushTRefinedCostModel(load_base(dev),None,0).to(dev).eval(); capture=CaptureModel(base)
   solver=PrefixComparableCEMSolver(model=capture,population=PopulationConfig.create(300,elites=38),device=dev,candidate_seed=seed); configure_solver(solver); solver.solve(info,start_id=entry["row_id"],replan_index=0)
   candidates=capture.calls[-1][0]; native=native_scores(info,candidates,dev); env=gym.make("swm/PushT-v1",max_episode_steps=100)
   try:
    a,_=restore_env(env,entry); b,_=restore_env(env,entry)
    if not np.array_equal(a,b): raise RuntimeError("reset reconstruction mismatch")
    labels=[label_candidate(env,entry,c) for c in candidates]
    rng=np.random.default_rng(cfg["roles"]["cohort_seed"]+entry["row_id"]+seed); replay=[]
    for index in np.sort(rng.choice(len(candidates),cfg["screen"]["replay_per_bank"],replace=False)):
     again=label_candidate(env,entry,candidates[index]); first=labels[int(index)]
     replay.append({"index":int(index),"actions_exact":again["executed_sha256"]==first["executed_sha256"],"states_exact":np.array_equal(again["states"],first["states"]),"costs_exact":np.array_equal(again["costs"],first["costs"])})
   finally: env.close()
   if not all(x["actions_exact"] and x["states_exact"] and x["costs_exact"] for x in replay): raise RuntimeError("deterministic replay failure")
   sim=np.asarray([x["cumulative_cost"] for x in labels]); valid=np.asarray([x["valid"] for x in labels],bool); feat=feature_matrix(native,candidates)
   if valid.mean()<cfg["screen"]["minimum_valid_fraction"]: raise RuntimeError("too few valid labels")
   tmp=npz.with_suffix(".tmp.npz"); npz.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(tmp,candidates=candidates,native_cost=native,simulator_cost=sim,valid=valid,features=feat); tmp.replace(npz)
   atomic_json(meta,{"role":role,"row_id":entry["row_id"],"episode_id":entry["episode_id"],"candidate_seed":seed,"candidate_count":len(candidates),"valid_count":int(valid.sum()),"candidate_tensor_sha256":array_sha256(candidates),"native_cost_sha256":array_sha256(native),"replay_checks":replay,"npz_sha256":hashlib.sha256(npz.read_bytes()).hexdigest()})
def load_banks(role):
 result=[]
 for meta in sorted((WORK/role).glob("*.json")):
  rec=json.loads(meta.read_text()); data=np.load(meta.with_suffix(".npz"),allow_pickle=False); result.append((rec,data))
 return result
def fit():
 cfg=config(); banks=[]
 for _,d in load_banks("fit"):
  good=d["valid"]; banks.append((d["features"][good],d["simulator_cost"][good]))
 c=cfg["critic"]; critic=fit_linear(banks,top_tail=c["top_tail"],epochs=c["epochs"],learning_rate=c["learning_rate"],weight_decay=c["weight_decay"],seed=c["training_seed"])
 artifact=WORK/"artifacts"/f"linear_{critic.sha256()}.json"; atomic_json(artifact,critic.payload()); atomic_json(WORK/"fit_artifact.json",{"artifact":str(artifact),"sha256":critic.sha256(),"fit_bank_count":len(banks)})
def read_critic():
 record=json.loads((WORK/"fit_artifact.json").read_text()); p=json.loads(Path(record["artifact"]).read_text()); critic=LinearCritic(np.asarray(p["mean"]),np.asarray(p["scale"]),np.asarray(p["weight"]),float(p["bias"]))
 if critic.sha256()!=record["sha256"]: raise RuntimeError("critic artifact hash mismatch")
 return critic,record
def score_role(role,critic):
 records=[]
 for meta,d in load_banks(role):
  good=d["valid"]; native=d["native_cost"][good]; sim=d["simulator_cost"][good]; cand=d["candidates"][good]; pred=critic.score(d["features"][good])
  records.append({"row_id":meta["row_id"],"episode_id":meta["episode_id"],"candidate_seed":meta["candidate_seed"],"native":ranking_metrics(native,sim),"linear":ranking_metrics(pred,sim),"change":choice_comparison(native,pred,sim,cand,config()["analysis"]["helpful_margin"]),"range":float(np.ptp(sim)),"selected_native_cost":float(sim[np.argmin(native)]),"selected_linear_cost":float(sim[np.argmin(pred)])})
 return records
def latency(role, critic):
 """Measure the declared boundary: native rollout + feature build + scalar head."""
 ds=lewm_eval.get_dataset(OmegaConf.load(REPO/"le-wm/config/eval/pusht.yaml"),json.loads((ROOT/"config.json").read_text())["planner_qualification"]["dataset"]["name"])
 transform=lewm_eval.processors["pusht"]; entries={x["row_id"]:x for x in manifest()[role]}; dev=device(); critic_times=[]; end_times=[]
 for meta,d in load_banks(role):
  entry=entries[meta["row_id"]]; candidates=d["candidates"]
  # Warm-up is intentionally excluded from the fixed repetition summaries.
  info=make_info(ds,transform,entry,dev); values=native_scores(info,candidates,dev); features=feature_matrix(values,candidates); critic.score(features)
  for _ in range(7):
   started=time.perf_counter_ns(); critic.score(features); critic_times.append((time.perf_counter_ns()-started)/1e6)
   started=time.perf_counter_ns(); values=native_scores(info,candidates,dev); critic.score(feature_matrix(values,candidates)); end_times.append((time.perf_counter_ns()-started)/1e6)
 def summary(values): return {"median_ms":float(np.median(values)),"p95_ms":float(np.quantile(values,.95)),"samples":len(values),"boundary":"300-candidate fixed final bank"}
 return {"critic_only":summary(critic_times),"end_to_end":summary(end_times),"operation_overhead":{"linear_parameters":52,"per_candidate_multiply_adds":51,"exact_complete_flops_claim":False}}
def select():
 critic,art=read_critic(); records=score_role("selection",critic); native=np.mean([x["native"]["top_choice_regret"] for x in records]); linear=np.mean([x["linear"]["top_choice_regret"] for x in records]); chosen="linear" if linear<native else "native"
 atomic_json(WORK/"selected_model.json",{"chosen":chosen,"selection_mean_regret":{"native":float(native),"linear":float(linear)},"tie_break":"native","artifact":art,"selection_bank_count":len(records)})
def evaluate():
 frozen=json.loads((WORK/"selected_model.json").read_text()); critic,_=read_critic(); records=score_role("evaluation",critic); atomic_json(WORK/"evaluation_records.json",{"frozen_selection":frozen,"records":records})
 atomic_json(WORK/"latency.json",latency("evaluation",critic))
def main():
 p=argparse.ArgumentParser(); p.add_argument("command",choices=("generate","fit","select","evaluate")); p.add_argument("--role",choices=("pilot","fit","selection","evaluation")); a=p.parse_args()
 if a.command=="generate":
  if not a.role: p.error("--role required");
  generate(a.role)
 elif a.command=="fit": fit()
 elif a.command=="select": select()
 else: evaluate()
if __name__=="__main__": main()
