#!/usr/bin/env python3
"""Seal fresh Task F roles before candidate generation."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]
sys.path[:0]=[str(REPO/"le-wm"),str(ROOT)]
from omegaconf import OmegaConf
import eval as lewm_eval
from task_d import canonical_bytes, file_sha256, freeze_cohort

def main():
 cfg=json.loads((ROOT/"task_f_config.json").read_text()); roles=cfg["roles"]
 dataset=lewm_eval.get_dataset(OmegaConf.load(REPO/"le-wm/config/eval/pusht.yaml"),json.loads((ROOT/"config.json").read_text())["planner_qualification"]["dataset"]["name"])
 total=sum(roles[k] for k in ("pilot_starts","fit_starts","selection_starts","evaluation_starts"))
 raw=freeze_cohort(dataset,lewm_eval,{"task_a":json.loads((ROOT/"config.json").read_text())["planner_qualification"],"cohort_seed":roles["cohort_seed"],"pilot_starts":0,"sealed_starts":total+128})
 prior=set()
 for path in (ROOT/"task_d_cohort.json",ROOT/"task_e_manifest.json"):
  data=json.loads(path.read_text()); prior|={int(x["episode_id"]) for key in ("pilot","sealed") for x in data[key]}
 rows=[x for x in raw["sealed"] if int(x["episode_id"]) not in prior][:total]
 if len(rows)!=total: raise RuntimeError("insufficient fresh Task F source episodes")
 cursor=0; payload={"schema_version":1,"status":"frozen_before_task_f_outcomes","exclusions":raw["exclusions"]|{"task_d_and_e_episodes":sorted(prior)},"candidate_seeds":roles["candidate_seeds"],"provenance":{"config_sha256":file_sha256(ROOT/"task_f_config.json"),"protocol_sha256":file_sha256(ROOT/"TASK_F_PROTOCOL.md"),"task_e_manifest_sha256":file_sha256(ROOT/"task_e_manifest.json"),"source_commit":cfg["provenance"]["task_e_commit"]}}
 for role,key in (("pilot","pilot_starts"),("fit","fit_starts"),("selection","selection_starts"),("evaluation","evaluation_starts")):
  payload[role]=rows[cursor:cursor+roles[key]]; cursor+=roles[key]
 payload["episode_set_sha256"]=hashlib.sha256(np.asarray([r["episode_id"] for r in rows],dtype="<i8").tobytes()).hexdigest()
 all_eps=[r["episode_id"] for r in rows]
 if len(all_eps)!=len(set(all_eps)) or prior.intersection(all_eps): raise RuntimeError("role isolation failure")
 out=ROOT/"task_f_manifest.json"
 if out.exists() and json.loads(out.read_text())!=payload: raise RuntimeError("refusing to overwrite frozen manifest")
 if not out.exists(): out.write_bytes(canonical_bytes(payload))
 print(json.dumps({"manifest":str(out),"sha256":file_sha256(out)}))
if __name__=="__main__": main()
