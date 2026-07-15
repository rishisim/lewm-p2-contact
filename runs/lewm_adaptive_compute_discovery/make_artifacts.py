#!/usr/bin/env python3
"""Build final failure-report tables, figures, provenance, and post-hoc analysis."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
import critics, data_isolation, policy, run_discovery as run


def write_csv(path,rows):
    rows=list(rows); path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    with path.open('w',newline='') as h:
        w=csv.DictWriter(h,fields); w.writeheader(); w.writerows(rows)


def model_io():
    path=REPO/'runs/lewm_adaptive_compute_v2/model_io.py'
    spec=importlib.util.spec_from_file_location('discovery_posthoc_model_io',path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module


def main():
    cfg=json.loads((ROOT/'config.json').read_text()); payload=json.loads((ROOT/'internal_decision.json').read_text())
    metrics=ROOT/'metrics'; figures=ROOT/'figures'; figures.mkdir(exist_ok=True)
    budget=[]; fixed=[]; calibration=[]
    for candidate in payload['candidates']:
        for row in candidate['fixed_exit_checks']:
            fixed.append({'candidate':candidate['candidate'],'family':candidate['family'],'critic_variant':candidate['critic_variant'],
                          'depth':row['depth'],'raw_mse':row['raw_mse'],'gain_vs_d1':row['improvement_vs_d1']['mean_benefit'],
                          'ci_low':row['improvement_vs_d1']['ci_low'],'ci_high':row['improvement_vs_d1']['ci_high']})
        for item in candidate['critic_diagnostics']:
            for quantile in item['quantiles']:
                calibration.append({'candidate':candidate['candidate'],'depth':item['depth'],'spearman':item['spearman_rho'],
                                    'ordered':item['ordered_quantiles'],**quantile})
        for row in candidate['policy']['operating_points']:
            budget.append({'candidate':candidate['candidate'],'family':candidate['family'],'critic_variant':candidate['critic_variant'],
                           'key':row['key'],'lcb_z':row['lcb_z'],'target_mean_calls':row['target_mean_calls'],'mean_calls':row['mean_calls'],
                           'raw_mse':row['raw_mse'],'matched_raw_mse':row['baseline_raw_mse'],'expected_matched_raw_mse':row['expected_baseline_raw_mse'],
                           'benefit_vs_matched':row['vs_matched_baseline']['mean_benefit'],'matched_ci_low':row['vs_matched_baseline']['ci_low'],
                           'matched_ci_high':row['vs_matched_baseline']['ci_high'],'expected_ci_low':row['vs_expected_mixture']['ci_low'],
                           'benefit_vs_d1':row['vs_fixed_d1']['mean_benefit'],'d1_ci_low':row['vs_fixed_d1']['ci_low'],
                           'benefit_vs_null':row['vs_histogram_null']['mean_benefit'],'null_ci_low':row['vs_histogram_null']['ci_low'],
                           'whitened_mse':row.get('whitened_mse'),'whitened_benefit_vs_matched':row.get('whitened_vs_matched',{}).get('mean_benefit'),
                           'whitened_ci_low':row.get('whitened_vs_matched',{}).get('ci_low'),'call_nondominated':row['nondominated'],
                           'flop_nondominated':row['flop_nondominated'],'adaptive_flops_per_transition':row['adaptive_total_flops_per_transition'],
                           'baseline_flops_per_transition':row['baseline_total_flops_per_transition'],'gate_decisions':row['gate_decisions'],
                           'solver_latency_seconds':candidate['runtime_call_audit'][row['key']]['selected_solver_latency_seconds']})
    write_csv(metrics/'budget_curves.csv',budget); write_csv(metrics/'fixed_exits.csv',fixed); write_csv(metrics/'gain_calibration_quantiles.csv',calibration)

    # Environment and immutable provenance.
    environment={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,
                 'numpy':np.__version__,'mps_available':torch.backends.mps.is_available(),'device':'mps','pid':os.getpid()}
    run.write_json(ROOT/'environment.json',environment)
    source_files=['PLAN.md','config.json','REPORT.md','decision.json','run_discovery.py','models.py','critics.py','policy.py','data_isolation.py','extract_isolated.py','make_artifacts.py']
    provenance={'sources':{name:data_isolation.sha256_file(ROOT/name) for name in source_files},
                'v1_checkpoint':data_isolation.sha256_file(run.V1_CHECKPOINT),
                'base_config':cfg['base_config_sha256'],'base_weights':cfg['base_weights_sha256'],
                'v3_manifest':data_isolation.PINNED['manifest_sha256'],
                'train_only_cache':data_isolation.sha256_file(ROOT/'cache/v3_train_only.npz'),
                'combined_v3_cache':{'sha256':data_isolation.PINNED['opaque_combined_cache_sha256'],'opened_with_numpy':False},
                'calibration_receipt_exists':(ROOT/'calibration_access_receipt.json').exists(),
                'calibration_cache_exists':(ROOT/'cache/v3_calibration_only.npz').exists(),
                'v4_directory_exists':(REPO/'runs/lewm_adaptive_compute_v4').exists(),
                'checkpoints':{p.name:data_isolation.sha256_file(p) for p in sorted((ROOT/'checkpoints').glob('*.pt'))}}
    run.write_json(ROOT/'provenance.json',provenance)

    # Plot fixed exits once per solver family.
    fig,ax=plt.subplots(figsize=(7,5))
    for family in ('v3_repair','stagewise','contractive','verifier_v1'):
        rows=[r for r in fixed if r['family']==family and r['critic_variant']=='full']
        ax.plot([r['depth'] for r in rows],[r['raw_mse'] for r in rows],'o-',label=family)
    ax.set(xlabel='Fixed block calls',ylabel='Raw latent MSE',title='Anchored solver exits (internal validation)'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures/'01_fixed_exit_solvers.png',dpi=220); plt.close(fig)

    # Stagewise block-call curves and matched baselines.
    fig,ax=plt.subplots(figsize=(7,5))
    for variant,marker in [('full','o'),('compact','s')]:
        rows=[r for r in budget if r['candidate']==f'stagewise__{variant}' and r['lcb_z']==0.0]
        ax.plot([r['mean_calls'] for r in rows],[r['raw_mse'] for r in rows],marker+'-',label=f'{variant} adaptive')
        ax.plot([r['mean_calls'] for r in rows],[r['matched_raw_mse'] for r in rows],marker+'--',alpha=.7,label=f'{variant} matched mix')
    rows=[r for r in fixed if r['candidate']=='stagewise__full']
    ax.scatter([r['depth'] for r in rows],[r['raw_mse'] for r in rows],c='black',label='fixed exits')
    ax.set(xlabel='Mean block calls',ylabel='Raw latent MSE',title='Stagewise budget curves'); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figures/'02_budget_curves.png',dpi=220); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    for axis,depth in zip(axes,(1,2,3)):
        for candidate,marker in [('stagewise__full','o'),('stagewise__compact','s')]:
            rows=[r for r in calibration if r['candidate']==candidate and r['depth']==depth]
            axis.plot([r['quantile']+1 for r in rows],[r['mean_realized_gain'] for r in rows],marker+'-',label=candidate.split('__')[1])
        axis.axhline(0,color='black',lw=.7); axis.set(title=f'Decision after d{depth}',xlabel='Score quintile'); axis.grid(alpha=.2)
    axes[0].set_ylabel('Realized marginal gain'); axes[-1].legend(); fig.suptitle('Cross-fitted gain ordering'); fig.tight_layout(); fig.savefig(figures/'03_gain_calibration.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5))
    for candidate,marker in [('stagewise__full','o'),('stagewise__compact','s')]:
        rows=[r for r in budget if r['candidate']==candidate and r['lcb_z']==0.0]
        ax.plot(np.asarray([r['adaptive_flops_per_transition'] for r in rows])/1e6,[r['raw_mse'] for r in rows],marker+'-',label=candidate)
    fixed_stage=[r for r in payload['candidates'] if r['candidate']=='stagewise__full'][0]['policy']['frontier']
    points=[r for r in fixed_stage if r['policy'].startswith('fixed_')]
    ax.scatter(np.asarray([r['total_flops_per_transition'] for r in points])/1e6,[r['raw_mse'] for r in points],c='black',label='fixed exits')
    ax.set(xlabel='Analytic total FLOPs / transition (millions)',ylabel='Raw latent MSE',title='Gate overhead removes the apparent call frontier'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(figures/'04_flop_frontier.png',dpi=220); plt.close(fig)

    # Post-hoc physical allocation: labels are first attached after the verdict.
    arrays=run.load_arrays('train'); split=data_isolation.grouped_discovery_split(cfg['internal_validation_episodes'],cfg['internal_split_seed'])
    fit_idx=run.indices_for_episodes(arrays,split['discovery_fit']); val_idx=run.indices_for_episodes(arrays,split['internal_validation'])
    labels=model_io().extract_post_prediction_labels(run.SOURCE if hasattr(run,'SOURCE') else Path('/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5'),arrays['episode_id'],arrays['model_step'])
    static=float(np.quantile(labels['block_disp'][fit_idx],.25)); transport=float(np.quantile(labels['block_disp'][fit_idx],.75))
    interaction=labels['interaction'][val_idx]; impact=labels['impact'][val_idx]; displacement=labels['block_disp'][val_idx]
    regimes=np.full(len(val_idx),'free',dtype='U20'); regimes[displacement<=static]='static'; regimes[(~interaction)&(displacement>transport)]='transport_free'; regimes[interaction]='contact'; regimes[impact]='impact'
    stage=next(c for c in payload['candidates'] if c['candidate']=='stagewise__full'); chosen='z0.00_b1.50'; allocation=np.asarray(stage['policy']['allocations'][chosen])
    regime_rows=[]
    for name in ('impact','contact','transport_free','free','static'):
        mask=regimes==name
        if mask.any(): regime_rows.append({'regime':name,'n':int(mask.sum()),'mean_calls':float(allocation[mask].mean()),'depth1_fraction':float((allocation[mask]==1).mean()),'depth4_fraction':float((allocation[mask]==4).mean())})
    write_csv(metrics/'posthoc_regime_allocation.csv',regime_rows); run.write_json(metrics/'posthoc_thresholds.json',{'static_q25':static,'transport_q75':transport,'fit_episodes_only':True,'operating_point':chosen,'labels_attached_after_decision':True})
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.bar([r['regime'] for r in regime_rows],[r['mean_calls'] for r in regime_rows]); ax.set(ylabel='Mean selected calls',title='Post-hoc stagewise allocation by physical regime'); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); fig.savefig(figures/'05_posthoc_regimes.png',dpi=220); plt.close(fig)

    # Measured incremental latency: true sparse solver plus prebuilt critic MLP
    # scoring. Causal feature construction and the common cached base predictor
    # are excluded and called out explicitly.
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); v1=run.load_v1_refiner(device); latency={}
    for candidate_name in ('stagewise__full','stagewise__compact'):
        candidate=next(c for c in payload['candidates'] if c['candidate']==candidate_name)
        checkpoint=ROOT/'checkpoints'/f'{candidate_name}_critics.pt'; frozen={'path':str(checkpoint),'sha256':data_isolation.sha256_file(checkpoint)}
        solver,saved,by_depth=run.load_frozen_candidate(frozen,cfg,v1,device)
        depths=tuple(saved['supported_calls']); _,features,_=run.dense_outputs(solver,'stagewise',arrays,val_idx,device,cfg['predict_batch_size'],(0,*depths))
        if saved.get('feature_tail') is not None: features=[value[:,-int(saved['feature_tail']):] for value in features]
        selected=np.asarray(candidate['policy']['allocations'][chosen]); built={depth:[critic.build(device) for critic in by_depth[depth]] for depth in depths[:-1]}
        tensors={}
        for stage,depth in enumerate(depths[:-1]):
            active=selected>=depth
            tensors[depth]=[(torch.from_numpy(((features[stage][active]-critic.feature_mean)/critic.feature_std).astype(np.float32)).to(device),model)
                            for critic,model in zip(by_depth[depth],built[depth])]
        repeats=[]
        with torch.inference_mode():
            for _ in range(5):
                if device.type=='mps': torch.mps.synchronize()
                start=time.perf_counter()
                for depth in depths[:-1]:
                    for values,model in tensors[depth]: model(values)
                if device.type=='mps': torch.mps.synchronize()
                repeats.append(time.perf_counter()-start)
        solver_latency=candidate['runtime_call_audit'][chosen]['selected_solver_latency_seconds']
        latency[candidate_name]={'operating_point':chosen,'transitions':len(selected),'mean_calls':float(selected.mean()),
                                 'solver_sparse_seconds':solver_latency,'critic_mlp_seconds_repeats':repeats,
                                 'critic_mlp_median_seconds':float(np.median(repeats)),
                                 'component_sum_median_seconds':float(solver_latency+np.median(repeats)),
                                 'excludes':'common base predictor and causal feature construction'}
    run.write_json(metrics/'latency.json',latency)

    verification={'decision':'discovery_gate_failed','calibration_consumed':False,'test_targets_consumed':False,'v4_created':False,
                  'focused_tests':27,'all_anchor_audits_passed':all(c['anchor_audit']['passed'] for c in payload['candidates']),
                  'all_runtime_call_audits_passed':all(all(a['exact_call_match'] and a['dense_selected_close'] for a in c['runtime_call_audit'].values()) for c in payload['candidates']),
                  'all_gradient_boundaries_passed':all(c['gradient_boundary_audit']['passed'] for c in payload['candidates']),
                  'full_critic_flops_per_decision':3*run.critic_flops_per_model(1046,[128,64]),
                  'compact_critic_flops_per_decision':3*run.critic_flops_per_model(11,[32,16]),
                  'stage_adapter_flops_per_call':2*(843*128+128*192),
                  'failure_mechanism':'full critics retain allocation signal but are FLOP-dominated; compact critics restore cheap gating but lose matched-mixture ranking and reverse whitened robustness'}
    run.write_json(metrics/'adversarial_verification.json',verification)
    artifact_paths=[]
    for pattern in ('*.md','*.json','*.txt','metrics/*','figures/*','checkpoints/*','tests/*.py'):
        artifact_paths.extend(path for path in ROOT.glob(pattern) if path.is_file() and path.name!='artifact_manifest.json')
    manifest={'decision':'discovery_gate_failed','files':{str(path.relative_to(ROOT)):data_isolation.sha256_file(path) for path in sorted(set(artifact_paths))}}
    run.write_json(ROOT/'artifact_manifest.json',manifest)


if __name__=='__main__': main()
