#!/usr/bin/env python3
"""Aggregate frozen Task F evaluation without candidate-IID inference."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; WORK=ROOT/"work/task_f"
from task_f import bootstrap_start
def main():
 cfg=json.loads((ROOT/"task_f_config.json").read_text()); data=json.loads((WORK/"evaluation_records.json").read_text()); latency=json.loads((WORK/"latency.json").read_text()); rows=data["records"]
 by_start={}
 for row in rows: by_start.setdefault(row["row_id"],[]).append(row)
 diffs=np.asarray([np.mean([x["linear"]["top_choice_regret"]-x["native"]["top_choice_regret"] for x in group]) for group in by_start.values()])
 change=[x["change"] for x in rows]; helpful=sum(x["direction"]=="improves" for x in change); harmful=sum(x["direction"]=="harms" for x in change)
 seed={str(s):float(np.mean([x["linear"]["top_choice_regret"]-x["native"]["top_choice_regret"] for x in rows if x["candidate_seed"]==s])) for s in cfg["roles"]["candidate_seeds"]}
 # Regret fraction makes catastrophic and tail definitions independent of cost scale.
 frac={head:np.asarray([x[head]["top_choice_regret"]/x["range"] if x["range"] else np.nan for x in rows]) for head in ("native","linear")}
 primary=bootstrap_start(diffs,draws=cfg["analysis"]["bootstrap_draws"],seed=cfg["analysis"]["bootstrap_seed"])
 selected=data["frozen_selection"]["chosen"]; valid=bool(np.isfinite(diffs).all())
 latency_ok=latency["critic_only"]["median_ms"]<=cfg["analysis"]["max_median_overhead_ms"] and latency["critic_only"]["p95_ms"]<=cfg["analysis"]["max_p95_overhead_ms"]
 passes= selected=="linear" and primary["ci95"][1]<0 and all(v<0 for v in seed.values()) and helpful>harmful and np.nanmean(frac["linear"]<=cfg["analysis"]["catastrophic_regret_fraction"])>=np.nanmean(frac["native"]<=cfg["analysis"]["catastrophic_regret_fraction"]) and np.nanquantile(frac["linear"],.9)<=np.nanquantile(frac["native"],.9) and latency_ok
 result={"schema_version":1,"status":"complete","scope":"final_bank_reranking_only","selected_head":selected,"primary_linear_minus_native_regret":primary,"per_seed_mean_difference":seed,"choice_changes":{"helpful":helpful,"harmful":harmful,"tied":len(change)-helpful-harmful,"regret_weighted_benefit":float(-sum(x["simulator_cost_delta"] for x in change if x["changed"]))},"safety":{"native_catastrophic_fraction":float(np.nanmean(frac["native"]>=cfg["analysis"]["catastrophic_regret_fraction"])),"linear_catastrophic_fraction":float(np.nanmean(frac["linear"]>=cfg["analysis"]["catastrophic_regret_fraction"])),"native_worst_tail":float(np.nanquantile(frac["native"],.9)),"linear_worst_tail":float(np.nanquantile(frac["linear"],.9))},"latency":latency,"decision":"PASS" if passes and valid else "FAIL","next_step":"small closed-loop critic-in-CEM confirmation" if passes and valid else "retain native depth-0/population-300; stop planner/refinement branch"}
 (ROOT/"task_f_results.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"decision":result["decision"]}))
if __name__=="__main__": main()
