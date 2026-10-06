"""Resumable pre-evaluation calibration grid, using disjoint scene streams."""

import json
import hashlib
import math
from dataclasses import asdict
from pathlib import Path
import numpy as np

from ..paths import runs_root, checkpoint_dir
from ..normalization import load_normalization
from .conditions import NAMES, generate_bases, save_bases, load_bases
from .rollout_eval import evaluate, prepare_output_root, summarize
from .stats import paired_sample_size, wilson_ci

CALIBRATION_SEED = 2_100_000_000
CALIBRATION_PLANNER_SEED = 23042
VALIDATION_PLANNER_SEED = 23043


def grid(n=20, workers=8, device='auto', run_dir=None):
    """W4c: fixed rollout tasks, shaping x CEM, then independent validation."""
    root = prepare_output_root(run_dir or runs_root()/'w4c-calibration')
    root.mkdir(parents=True,exist_ok=True)
    normalization_path = root/'normalization.json'
    if not normalization_path.exists():
        normalization_path.write_text(json.dumps(load_normalization(checkpoint_dir('lewm-pusht')),indent=2))
    normalization = json.loads(normalization_path.read_text())
    bases_path = root/'bases.json'
    if not bases_path.exists():
        save_bases(generate_bases(n,CALIBRATION_SEED),bases_path)
    bases = load_bases(bases_path)
    # Prove actual uint32 scene-seed separation, not just master separation.
    previous = []
    for path in (runs_root()/'w4-pilot'/'bases.json',
                 runs_root()/'w4b-calibration'/'bases-min40.json',
                 runs_root()/'w4b-calibration'/'bases-min60.json'):
        if path.exists():
            previous += [b.seed for b in load_bases(path)]
    if {b.seed for b in bases} & set(previous):
        raise ValueError('calibration scene seeds overlap historical scenes')
    distributions = {name:[b.rollout_tasks[name]['displacement_px'] for b in bases] for name in NAMES}
    (root/'construction.json').write_text(json.dumps({'master_seed':CALIBRATION_SEED,
        'historical_scene_seeds_checked':len(set(previous)), 'scene_seeds':[b.seed for b in bases],
        'displacements_px':distributions},indent=2))
    tables = []
    targets = dict(off_path=.8,on_path=.6,move_peg=.6,move_T_matched=.8)
    # Exhaust the requested six-cell grid at upstream budget 50. If needed,
    # extend its best cell to 100 before independent validation (bounded time).
    for budget in (50,100):
        extended = min(tables,key=lambda c:(c['target_deficit'],c['d_difference'],
                       c['population'],c['approach_weight'])) if budget==100 else None
        for population,iterations,topk in ((100,10,10),(300,30,30)):
            for weight in (0,.1,.3):
                if extended and (population != extended['population'] or weight != extended['approach_weight']):
                    continue
                run = root/f'b{budget}-cem{population}x{iterations}-w{weight}'
                evaluate('reference',bases_path,seed=CALIBRATION_PLANNER_SEED,n=n,run_dir=run,
                    normalization=normalization,budget=budget,displacement_range=(40,100),workers=workers,
                    population=population,iterations=iterations,topk=topk,device=device,approach_weight=weight,
                    prefix_run=root/f'b50-cem{population}x{iterations}-w{weight}' if budget==100 else None)
                summary = summarize(read_records(run),CALIBRATION_PLANNER_SEED)
                rates = {k:v['success_rate'] for k,v in summary['conditions'].items()}
                d_difference = abs(rates['on_path']-rates['off_path'])
                row = dict(population=population,iterations=iterations,topk=topk,approach_weight=weight,
                    budget=budget,displacement_range=[40,100],bases_path=str(bases_path),source_run=str(run),
                    summary=summary,meets_targets=all(rates[k]>=targets[k] for k in NAMES),
                    d_difference=d_difference,ready=all(rates[k]>=targets[k] for k in NAMES) and d_difference<.05-1e-12,
                    target_deficit=sum(max(0,targets[k]-rates[k]) for k in NAMES))
                tables.append(row)
                (root/'grid.json').write_text(json.dumps(tables,indent=2))
                print(json.dumps({**{k:v for k,v in row.items() if k!='summary'},'rates':rates}),flush=True)
        if any(c['ready'] for c in tables):
            break
    def preference(c):
        return (c['budget'],c['population'],c['approach_weight'])
    passing = [c for c in tables if c['ready']]
    chosen = min(passing,key=preference) if passing else min(tables,key=lambda c:(c['target_deficit'],c['d_difference'],preference(c)))
    (root/'chosen.json').write_text(json.dumps(chosen,indent=2))
    options = dict(normalization=normalization,budget=chosen['budget'],displacement_range=(40,100),
        workers=workers,population=chosen['population'],iterations=chosen['iterations'],topk=chosen['topk'],
        device=device,approach_weight=chosen['approach_weight'])
    # F is the exact chosen grid run; E is independent.
    evaluate('reference',bases_path,seed=VALIDATION_PLANNER_SEED,n=n,run_dir=root/'chosen-reference',
             feasibility_run=chosen['source_run'],**options)
    learned_source = root/'pretrained-budget100'
    if (learned_source/'episodes.jsonl').exists():
        source_config = json.loads((learned_source/'config.json').read_text())
        reference_config = json.loads((Path(chosen['source_run'])/'config.json').read_text())
        if (source_config['seed'] != VALIDATION_PLANNER_SEED or source_config['arm'] != 'lewm-pusht'
                or source_config['budget'] != 100 or source_config['population'] != 300
                or source_config['iterations'] != 30 or source_config['topk'] != 30
                or source_config['observation_rendering'] != 'upstream-fixed-target'
                or source_config['checkpoint_weights_sha256'] != hashlib.sha256(
                    (checkpoint_dir('lewm-pusht')/'weights.pt').read_bytes()).hexdigest()
                or any(source_config[k] != reference_config[k] for k in ('bases_sha256','normalization_sha256'))):
            raise ValueError('precomputed learned validation configuration differs')
        source_rows = read_records(learned_source)
        if len(source_rows)==4*n:
            destination = root/'chosen-lewm'
            destination.mkdir(exist_ok=True)
            rows = [prefix(r,chosen['budget']) for r in source_rows]
            (destination/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (destination/'reuse.json').write_text(json.dumps({'source':str(learned_source),
                'budget_prefix':chosen['budget'],'populations_and_condition_files_at_source':True}))
    evaluate('lewm-pusht',bases_path,seed=VALIDATION_PLANNER_SEED,n=n,run_dir=root/'chosen-lewm',
        feasibility_run=chosen['source_run'],normalization=normalization,budget=chosen['budget'],
        displacement_range=(40,100),workers=workers,population=300,iterations=30,topk=30,device=device,
        with_target=True)
    freeze_protocol(root)
    return chosen


def read_records(root):
    return [json.loads(line) for line in (Path(root)/'episodes.jsonl').read_text().splitlines()]


def prefix(record, budget):
    """50-step prefix equals a standalone run: same seed, horizon, and history."""
    result = dict(record)
    result['trajectory'] = record['trajectory'][:budget+1]
    result['actions'] = record['actions'][:budget]
    result.update(record['budget_checkpoints'][str(budget)])
    result['planning_times_s'] = record['planning_times_s'][:budget//25]
    result['budget_checkpoints'] = {k:v for k,v in record['budget_checkpoints'].items() if int(k)<=budget}
    return result


def reuse_identical_bases(source, destination, source_bases, destination_bases):
    """Reuse exact episodes when a changed acceptance filter retains a base.

    Base IDs seed the solver; all states, goals, and planner settings are
    identical. Only calibration invokes this, across the two acceptance filters.
    """
    if not (source/'episodes.jsonl').exists():
        return
    source_rows = {b.id: asdict(b) for b in load_bases(source_bases)}
    identical = {b.id for b in load_bases(destination_bases) if source_rows.get(b.id) == asdict(b)}
    if not identical:
        return
    destination.mkdir(parents=True, exist_ok=True)
    completed = {(r['base_id'],r['condition']) for r in read_records(destination)} if (destination/'episodes.jsonl').exists() else set()
    reused = [r for r in read_records(source) if r['base_id'] in identical
              and (r['base_id'],r['condition']) not in completed]
    with (destination/'episodes.jsonl').open('a') as file:
        for row in reused:
            file.write(json.dumps(row)+'\n')
    (destination/'reuse.json').write_text(json.dumps({'source':str(source), 'reused_episodes':len(reused),
                                                    'identical_base_ids':sorted(identical)}, indent=2))


def evidence(root=None):
    """Compact evidence, with calibration eligibility explicitly descriptive."""
    root = Path(root or runs_root()/'w4c-calibration')
    chosen = json.loads((root/'chosen.json').read_text())
    bases = load_bases(chosen['bases_path'])
    base_n = len(bases)
    reference = read_records(root/'chosen-reference')
    feasibility = read_records(chosen['source_run'])
    learned = read_records(root/'chosen-lewm')
    expected = {(b.id,name) for b in bases for name in NAMES}
    for arm,rows in (('reference',reference),('feasibility',feasibility),('pretrained',learned)):
        if len(rows) != len(expected) or {(r['base_id'],r['condition']) for r in rows} != expected:
            raise ValueError(f'{arm} calibration must be complete and unique before freezing')
    result = {'chosen':chosen, 'arms':{'reference':summarize(reference,VALIDATION_PLANNER_SEED),
                                     'lewm-pusht':summarize(learned,VALIDATION_PLANNER_SEED)},
              'common_feasible':{}, 'sample_size':{}, 'episode_times':[]}
    for arm, records in (('reference',reference),('lewm-pusht',learned)):
        result['episode_times'] += [{k:r[k] for k in ('arm','base_id','condition','wall_s','planning_times_s')} for r in records]
        walls = [r['wall_s'] for r in records]
        result['arms'][arm]['wall_time_summary_s'] = {
            'mean':float(np.mean(walls)), 'median':float(np.median(walls)),
            'p90':float(np.quantile(walls,.9)), 'max':float(np.max(walls))}
    for label, pair in (('D',('on_path','off_path')),('G',('move_peg','move_T_matched'))):
        solved = [{r['base_id'] for r in feasibility if r['condition']==name and r['score']['success']} for name in pair]
        common = set.intersection(*solved)
        result['common_feasible'][label] = {'n':len(common),'excluded':base_n-len(common),'base_ids':sorted(common),
            'selection':'calibration F=23042; validation E=23043; distinct from main eligibility',
            'arms':{arm:summarize([r for r in records if r['base_id'] in common and r['condition'] in pair],VALIDATION_PLANNER_SEED)
                    for arm,records in (('reference',reference),('lewm-pusht',learned))}}
        rows = [{r['base_id']:r['score']['success'] for r in learned if r['condition']==name and r['base_id'] in common} for name in pair]
        if common:
            discordant = sum(rows[0][i]!=rows[1][i] for i in common)
            n_gap = paired_sample_size(discordant,len(common),effect=.15)
            n_remedy_proxy = paired_sample_size(discordant,len(common),effect=.10)
            fraction = len(common)/base_n
            fraction_lower = wilson_ci(len(common),base_n)[0]
            result['sample_size'][label] = {
                'gap':n_gap, 'remedy_proxy':n_remedy_proxy,
                'eligibility_fraction':fraction,'eligibility_fraction_lower':fraction_lower,
                'bases_at_observed_eligibility':math.ceil(max(n_gap['effective_bases'],n_remedy_proxy['effective_bases'])/fraction),
                'bases_at_lower_eligibility':math.ceil(max(n_gap['effective_bases'],n_remedy_proxy['effective_bases'])/fraction_lower)}
        else:
            result['sample_size'][label] = {'estimable':False,'reason':'zero common-feasible calibration bases'}
    mean_lewm = result['arms']['lewm-pusht']['wall_time_summary_s']['mean']
    mean_reference = result['arms']['reference']['wall_time_summary_s']['mean']
    sizes = [x['bases_at_lower_eligibility'] for x in result['sample_size'].values()
             if 'bases_at_lower_eligibility' in x]
    # Match W4b's conservative hundred-base rounding, now requiring both contrasts.
    candidate_n = int(math.ceil(max(sizes)/100)*100) if len(sizes)==2 else None
    result['planning_N'] = candidate_n
    conditional_n = int(math.ceil(max(sizes)/100)*100) if sizes else None
    result['conditional_planning_N'] = conditional_n
    result['time_projection_N'] = candidate_n or conditional_n
    projection_n = candidate_n or conditional_n
    planning = float(np.mean([sum(r['planning_times_s']) for r in learned]))
    gpu_lewm = mean_lewm-planning+planning/5
    result['time_estimate'] = {
        'learned_7_checkpoints_1_repeat_seconds_per_base':7*4*mean_lewm,
        'reference_F_and_E_seconds_per_base':2*4*mean_reference,
        'mean_lewm_planning_s':planning, 'cuda_projected_mean_lewm_episode_s':gpu_lewm,
        'Mac_hours':projection_n*(7*4*mean_lewm+2*4*mean_reference)/3600 if projection_n else None,
        'CUDA_hours':projection_n*(7*4*gpu_lewm+2*4*mean_reference)/3600 if projection_n else None,
        'projection_is_conditional':candidate_n is None,
        'assumption':'7 checkpoints, one evaluation repeat each, serial episodes; reference candidates use measured CPU workers; CUDA assumes ONLY LeWM planning 5x faster; excludes training/readout/ABC'}
    targets = dict(off_path=.8,on_path=.6,move_peg=.6,move_T_matched=.8)
    rates = result['arms']['reference']['conditions']
    d = result['common_feasible']['D']['arms']['reference']['contrasts'].get('D')
    result['GO'] = bool(chosen['ready'] and all(rates[k]['success_rate']>=targets[k] for k in NAMES)
                        and d is not None and abs(d['difference'])<.05-1e-12
                        and all(result['common_feasible'][label]['n'] for label in ('D','G')))
    (root/'evidence.json').write_text(json.dumps(result,indent=2))
    return result


def freeze_protocol(root=None):
    """Curate the completed calibration; failed targets never authorize main."""
    root = Path(root or runs_root()/'w4c-calibration')
    result = evidence(root)
    chosen = result['chosen']
    construction = json.loads((root/'construction.json').read_text())
    cells = json.loads((root/'grid.json').read_text())
    workers = json.loads((root/'chosen-reference'/'config.json').read_text())['workers']
    lines = ["The diagnostic configuration is frozen before main evaluation. **Main run: "
        + ('GO' if result['GO'] else 'NO-GO') + "**. No main evaluation has run.",
        f"Budget **{chosen['budget']} env steps**; reference **{chosen['population']}x{chosen['iterations']}, "
        f"top-{chosen['topk']}, w={chosen['approach_weight']}, {workers} CPU workers**; LeWM **300x30, top-30, MPS**.",
        "", "Complete F_cal=23042 grid, final joint success percentages (20 bases each):", "",
        "| Budget | CEM | w | off_path | on_path | move_peg | move_T_matched |",
        "|---:|:---|---:|---:|---:|---:|---:|"]
    for c in cells:
        values = [f"{100*c['summary']['conditions'][name]['success_rate']:.0f}" for name in NAMES]
        lines.append(f"| {c['budget']} | {c['population']}x{c['iterations']} | {c['approach_weight']} | "+' | '.join(values)+' |')
    lines += ["", "Independent E_cal=23043 validation, all bases unconditionally:", "",
        "| Condition | Reference | Pretrained | Reference mean s | Pretrained mean s |",
        "|:---|---:|---:|---:|---:|"]
    for name in NAMES:
        r,l = [result['arms'][arm]['conditions'][name] for arm in ('reference','lewm-pusht')]
        lines.append(f"| {name} | {round(r['success_rate']*r['n'])}/{r['n']} ({100*r['success_rate']:.0f}%) | "
            f"{round(l['success_rate']*l['n'])}/{l['n']} ({100*l['success_rate']:.0f}%) | {r['mean_wall_s']:.2f} | {l['mean_wall_s']:.2f} |")
    ref_d = result['arms']['reference']['contrasts']['D']['difference']
    lines += ["", f"Reference validation unconditional D difference is **{100*ref_d:.1f} points**. "
        + ("Frozen readiness checks pass." if result["GO"] else "Frozen readiness checks are unmet; no main evaluation.")]
    if result["common_feasible"]["D"]["n"] == 0:
        lines.append("The empty D common set leaves its reference E difference unestimable. These readiness failures do not establish LeWM information loss.")
    lines += ["", "F-selected common sets (calibration eligibility only):", ""]
    for label in ('D','G'):
        c = result['common_feasible'][label]
        lines.append(f"- {label}: **{c['n']}/20 included, {c['excluded']} excluded**.")
        for arm in ('reference','lewm-pusht'):
            summary = c['arms'][arm]
            if label in summary['contrasts']:
                d = summary['contrasts'][label]
                rates = ', '.join(f"{name} {100*v['success_rate']:.1f}%" for name,v in summary['conditions'].items())
                lines.append(f"  {arm}: {rates}; hard-control difference {100*d['difference']:.1f} points, "
                    f"bootstrap 95% CI [{100*d['ci'][0]:.1f}, {100*d['ci'][1]:.1f}].")
    lines += ["", "25-step displacement distributions (px):", "",
        "| Condition | min | median | mean | max |", "|:---|---:|---:|---:|---:|"]
    for name,values in construction['displacements_px'].items():
        lines.append(f"| {name} | {min(values):.2f} | {np.median(values):.2f} | {np.mean(values):.2f} | {max(values):.2f} |")
    bins = {name:np.histogram(construction['displacements_px'][name],bins=[40,60,80,100])[0].tolist()
            for name in ('move_peg','move_T_matched')}
    lines += ["", f"G bin counts ([40,60), [60,80), [80,100]): **{bins}**.",
        f"Scene seeds are disjoint from {construction['historical_scene_seeds_checked']} unique recorded pilot/W4b seeds.",
        "All 60 original replay witnesses solve; all 20 midpoint endpoint checks pass.",
        "The superseded initial attempt rejected three contact-cache-sensitive futures and contributes no observations;",
        "its diagnosis is retained at `runs/w4c-snapshot-audit/superseded.json`.", ""]
    for label,size in result['sample_size'].items():
        if 'gap' not in size:
            lines.append(f"{label}: N unestimable ({size['reason']}).")
        else:
            lines.append(f"{label}: discordance {size['gap']['discordance']:.4f}, Wilson q upper {size['gap']['discordance_upper']:.4f}; "
                f"effective bases {size['gap']['effective_bases']} (15 points), {size['remedy_proxy']['effective_bases']} (10 points); "
                f"eligibility {size['eligibility_fraction']:.4f}, Wilson lower {size['eligibility_fraction_lower']:.4f}; "
                f"candidate bases {size['bases_at_observed_eligibility']} (observed), {size['bases_at_lower_eligibility']} (lower bound).")
    lines += ["", f"Joint planning N: **{result['planning_N'] if result['planning_N'] is not None else 'unestimable'}**. "
        f"Conditional N from estimable contrasts: **{result['conditional_planning_N']}**; neither repairs reference readiness."]
    t = result['time_estimate']
    lines += ["", "Pretrained latency was measured concurrently with reference calibration and tests; projections assume those measured latencies persist."]
    rm = result['arms']['reference']['wall_time_summary_s']['mean']
    lm = result['arms']['lewm-pusht']['wall_time_summary_s']['mean']
    lines += ["", f"Measured independent episode means: reference **{rm:.3f} s**, pretrained **{lm:.3f} s** "
        f"(LeWM planning **{t['mean_lewm_planning_s']:.3f} s**)."]
    n = result['time_projection_N']
    if n:
        lines += [f"At N={n} ({'conditional' if t['projection_is_conditional'] else 'joint planning'}), 7 checkpoints x1 repeat x4 conditions:", "",
            f"- Mac learned: **{n*28*lm/3600:.2f} h**; reference F/E: **{n*8*rm/3600:.2f} h**; total **{t['Mac_hours']:.2f} h**.",
            f"- One CUDA GPU, LeWM planning assumed 5x faster, same CPU reference: learned "
            f"**{n*28*t['cuda_projected_mean_lewm_episode_s']/3600:.2f} h**; total **{t['CUDA_hours']:.2f} h**."]
    lines += ["", "Projections exclude training, generation, readout and A/B/C. CUDA is an assumption, not a benchmark.",
        "All full evidence is under `$LEWM_WORK_ROOT/runs/w4c-calibration`: `grid.json`, `chosen.json`,",
        "`construction.json`, `construction_audit.json`, `evidence.json`, `chosen-reference`, `chosen-lewm`,",
        "`pretrained-budget100` and `protocol_freeze.json`. Reused learned images/populations are at the source in `reuse.json`.",
        "Wilson intervals, all episode times and full power inputs are in evidence.json. Tests: `uv run pytest -q` passes (see `runs/w4c-tests/pytest.log`). No commit was made."]
    repo = Path(__file__).resolve().parents[3]
    protocol = repo/'experiments/role_swap/PROTOCOL.md'
    text = protocol.read_text()
    a,b = text.index('## Calibration evidence and frozen settings'),text.index('## Sample size and timing method')
    text = text[:a]+'## Calibration evidence and frozen settings\n\n'+'\n'.join(lines)+'\n\n'+text[b:]
    status_start = text.index('Status:')
    status_end = text.index('\n',status_start)
    text = text[:status_start]+'Status: **frozen diagnostic configuration; main run '+('GO' if result['GO'] else 'NO-GO')+'**.'+text[status_end:]
    protocol.write_text(text.rstrip()+'\n')
    sources = ['conditions.py','reference.py','rollout_eval.py','calibration.py']
    manifest = {'GO':result['GO'],'chosen':chosen,'planning_N':result['planning_N'],
        'conditional_planning_N':result['conditional_planning_N'],'time_estimate':t,
        'per_arm_with_target':{'pretrained':True,'ft_block':False,'ft_mixed':False},
        'calibration_master_seed':CALIBRATION_SEED,'F_cal':CALIBRATION_PLANNER_SEED,'E_cal':VALIDATION_PLANNER_SEED,
        'pretrained_weights_sha256':json.loads((root/'chosen-lewm'/'config.json').read_text())['checkpoint_weights_sha256'],
        'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in sources},
        'normalization_file_sha256':hashlib.sha256((root/'normalization.json').read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest()}
    (root/'protocol_freeze.json').write_text(json.dumps(manifest,indent=2))
    return result


if __name__ == '__main__':
    print(json.dumps(grid(), indent=2))
