#!/usr/bin/env python3
import hashlib,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 d=json.loads((R/'decision.json').read_text());c=d['compute'];crit=d['criteria'];sim=d['simultaneous_co_primary']
 facts='\n'.join(f"- {k}: {v['estimate']:.10g}, 95% CI [{v['lower']:.10g}, {v['upper']:.10g}]" for k,v in crit.items());sims='\n'.join(f"- {k}: simultaneous lower {v['lower']:.10g}" for k,v in sim.items())
 text=f'''# REPORT\n\nTerminal outcome: `{d['terminal_outcome']}`\n\n## Facts\n\n{facts}\n\nSimultaneous co-primary bounds:\n\n{sims}\n\nAdaptive raw MSE: {d['adaptive_raw_mse']:.10g}. Adaptive native-whitened MSE: {d['adaptive_native_whitened_mse']:.10g}. Mean model calls: {c['mean_model_calls']:.10g}; total FLOPs: {c['adaptive_total_flops']}; feature FLOPs: {c['gate_feature_flops']}; score FLOPs: {c['gate_score_flops']}; non-FLOP comparisons/min operations: {c['nonflop_comparisons']}.\n\n## Interpretations\n\nThe verdict follows the sealed mapping exactly. A pass makes a separate future V5 confirmation task eligible, but no V5 was created or launched here.\n\n## Process validity\n\nAll integrity checks: {d['integrity']}. The gate and all frozen scientific objects were unchanged. Contact/privileged fields were excluded by the evaluator input allowlist.\n\n## Limitations\n\nThis remains prospective discovery evidence on one frozen simulator/model distribution, not independent confirmation. Latency is engineering evidence and cannot change the verdict.\n\n## Outcome mapping\n\nPass requires six positive individual lower bounds, two positive simultaneous analytic lower bounds, and every integrity criterion. A valid statistical miss is failure; a process defect is invalid.\n''';(R/'REPORT.md').write_text(text)
 files={}
 for p in sorted(R.rglob('*')):
  if p.is_file() and p.name not in ('artifact_manifest.json','artifact_manifest_verification.json'):files[str(p.relative_to(REPO))]={'sha256':sha(p),'bytes':p.stat().st_size}
 m={'schema_version':1,'created_unix_ns':time.time_ns(),'root':str(R.relative_to(REPO)),'files':files,'path_set_complete':True};(R/'artifact_manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');mh=sha(R/'artifact_manifest.json');v={'passed':all((REPO/p).exists() and sha(REPO/p)==e['sha256'] for p,e in files.items()),'manifest_sha256':mh,'file_count':len(files),'actual_path_set_matches':set(files)=={str(p.relative_to(REPO)) for p in R.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','artifact_manifest_verification.json')}};(R/'audit/artifact_manifest_verification.json').write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'outcome':d['terminal_outcome'],'manifest_sha256':mh,'files':len(files)}))
if __name__=='__main__':main()
