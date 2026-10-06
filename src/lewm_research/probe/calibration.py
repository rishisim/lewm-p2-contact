"""Resumable pre-evaluation calibration grid, using disjoint scene streams."""

import json
import math
from dataclasses import asdict
from pathlib import Path
import numpy as np

from ..paths import runs_root, checkpoint_dir
from ..normalization import load_normalization
from .conditions import NAMES, generate_bases, make_conditions, save_bases, load_bases, score_trajectory
from .rollout_eval import evaluate, prepare_output_root, summarize
from .stats import paired_sample_size, wilson_ci

CALIBRATION_SEED = 2_000_000_000
CALIBRATION_PLANNER_SEED = 22042
VALIDATION_PLANNER_SEED = 22043


def grid(n=20, workers=8, device='auto', run_dir=None):
    root = prepare_output_root(run_dir or runs_root() / 'w4b-calibration')
    root.mkdir(parents=True, exist_ok=True)
    normalization_path = root / 'normalization.json'
    if not normalization_path.exists():
        normalization_path.write_text(json.dumps(load_normalization(checkpoint_dir('lewm-pusht')), indent=2))
    normalization = json.loads(normalization_path.read_text())
    tables = []
    for minimum in (60, 40):
        bases_path = root / f'bases-min{minimum}.json'
        if not bases_path.exists():
            # Both displacement bands must be valid on each retained base.
            pool = generate_bases(n, CALIBRATION_SEED, min_t_displacement=minimum)
            bases = []
            for base in pool:
                try:
                    make_conditions(base, (40,80), render_images=False)
                except ValueError:
                    continue
                bases.append(base)
                if len(bases) == n:
                    break
            if len(bases) != n:
                raise RuntimeError('not enough bases valid in both displacement bands')
            save_bases(bases, bases_path)
        for population, iterations, topk in ((100,10,10), (300,30,30)):
            for displacement in ((60,100), (40,80)):
                run = root / f'min{minimum}-cem{population}x{iterations}-d{displacement[0]}'
                names = 'all' if displacement[0] == 60 else 'move_peg,move_T_matched'
                if minimum == 40:
                    previous = root / f'min60-cem{population}x{iterations}-d{displacement[0]}'
                    reuse_identical_bases(previous, run, root/'bases-min60.json', bases_path)

                evaluate('reference', bases_path, conditions=names, seed=CALIBRATION_PLANNER_SEED,
                         n=n, run_dir=run, normalization=normalization, budget=100,
                         displacement_range=displacement, workers=workers,
                         population=population, iterations=iterations, topk=topk, device=device)
                records = read_records(run)
                if displacement[0] == 40:
                    shared = root / f'min{minimum}-cem{population}x{iterations}-d60'
                    records += [r for r in read_records(shared) if r['condition'] in ('off_path','on_path')]
                for budget in (50,100):
                    rows = [prefix(r, budget) for r in records]
                    summary = summarize(rows, CALIBRATION_PLANNER_SEED)
                    rates = {k: v['success_rate'] for k,v in summary['conditions'].items()}
                    targets = dict(off_path=.8, on_path=.6, move_peg=.6, move_T_matched=.8)
                    tables.append({'minimum':minimum, 'population':population, 'iterations':iterations,
                                   'topk':topk, 'displacement_range':list(displacement), 'budget':budget,
                                   'bases_path':str(bases_path), 'source_run':str(run),
                                   'summary':summary, 'meets_targets':all(rates[k]>=targets[k] for k in NAMES),
                                   'target_deficit':sum(max(0,targets[k]-rates[k]) for k in NAMES)})
                (root / 'grid.json').write_text(json.dumps(tables, indent=2))
                print(json.dumps({k:v for k,v in tables[-1].items() if k not in ('summary',)}), flush=True)
    # Least-changed among passing configs. If none pass, minimize summed deficit,
    # then favor original settings. Keep upstream budget 50 first among passes.
    def preference(c):
        changes = (c['minimum'] != 60) + (c['displacement_range'] != [60,100]) + (c['population'] != 100)
        return (c['budget'] != 50, changes, c['population'], c['minimum'] != 60,
                c['displacement_range'] != [60,100])
    passing = [c for c in tables if c['meets_targets']]
    chosen = min(passing, key=preference) if passing else min(tables, key=lambda c:(c['target_deficit'],preference(c)))
    (root / 'chosen.json').write_text(json.dumps(chosen,indent=2))
    # Seed F is calibration's grid seed. Use a separate seed E after selection,
    # so common-set reference results are not true by construction.
    feasibility = root/'chosen-feasibility'
    feasibility.mkdir(parents=True, exist_ok=True)
    if not (feasibility/'episodes.jsonl').exists():
        rows = chosen_reference_rows(root, chosen)
        (feasibility/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (feasibility/'reuse.json').write_text(json.dumps({'grid_source':chosen['source_run'],
                                                        'budget_prefix':chosen['budget']}))
    reference_options = dict(normalization=normalization, budget=chosen['budget'],
                             displacement_range=chosen['displacement_range'], workers=workers,
                             population=chosen['population'], iterations=chosen['iterations'],
                             topk=chosen['topk'], device=device)
    evaluate('reference', chosen['bases_path'], seed=CALIBRATION_PLANNER_SEED, n=n,
             run_dir=feasibility, **reference_options)
    evaluate('reference', chosen['bases_path'], seed=VALIDATION_PLANNER_SEED, n=n,
             run_dir=root/'chosen-reference', feasibility_run=feasibility, **reference_options)
    evaluate('lewm-pusht', chosen['bases_path'], seed=VALIDATION_PLANNER_SEED, n=n,
             run_dir=root/'chosen-lewm', feasibility_run=feasibility,
             normalization=normalization, budget=chosen['budget'],
             displacement_range=chosen['displacement_range'], workers=workers,
             population=300, iterations=30, topk=30, device=device)
    evidence(root)
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


def chosen_reference_rows(root, chosen):
    records = read_records(chosen['source_run'])
    if chosen['displacement_range'][0] == 40:
        shared = root/f"min{chosen['minimum']}-cem{chosen['population']}x{chosen['iterations']}-d60"
        records += [r for r in read_records(shared) if r['condition'] in ('off_path','on_path')]
    return [prefix(r,chosen['budget']) for r in records]


def evidence(root=None):
    """Compact evidence, with calibration eligibility explicitly descriptive."""
    root = Path(root or runs_root()/'w4b-calibration')
    chosen = json.loads((root/'chosen.json').read_text())
    base_n = len(load_bases(chosen['bases_path']))
    reference = read_records(root/'chosen-reference')
    feasibility = read_records(root/'chosen-feasibility')
    learned = read_records(root/'chosen-lewm')
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
            'selection':'calibration F=22042; validation E=22043; distinct from main eligibility',
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
    result['time_estimate'] = {'learned_3_arms_3_seeds_seconds_per_base':3*3*4*mean_lewm,
                             'reference_F_and_E_seconds_per_base':2*4*mean_reference,
                             'assumption':'serial episodes; reference candidates use measured worker count; excludes training/readout/ABC'}
    (root/'evidence.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    print(json.dumps(grid(), indent=2))
