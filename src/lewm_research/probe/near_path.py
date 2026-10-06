"""Bounded W4d lateral-offset calibration; W4c controls retain exact provenance."""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..paths import runs_root, checkpoint_dir
from .calibration import read_records, CALIBRATION_PLANNER_SEED as F, VALIDATION_PLANNER_SEED as E
from .conditions import load_bases, save_bases, with_near_path, naive_disturbance
from .rollout_eval import evaluate, prepare_output_root, summarize
from .stats import paired_sample_size, wilson_ci

OFFSETS = (40, 55, 70, 85)
CONTROLS = ('off_path', 'move_peg', 'move_T_matched')
OPTIONS = dict(budget=50, population=300, iterations=30, topk=30, workers=8,
               device='mps', with_target=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def select_offset(table):
    """Maximize witness disturbance subject to the specified common-set bar.

    F common-set differences are identically zero by selection and cannot
    substitute for the independent E check. Do not inspect E for selection.
    """
    feasible = [r for r in table if r['success_rate_all_bases'] >= .6 and r['naive_disturbance_rate'] > 0]
    if not feasible:
        return None
    strict = [r for r in feasible if r['reference_common_difference_abs'] < .05 - 1e-12]
    if strict:
        return min(strict, key=lambda r: (-r['naive_disturbance_rate'], r['L']))
    return min(feasible, key=lambda r: (r['reference_common_difference_abs'], -r['naive_disturbance_rate'], r['L']))


def _controls(source, bases, normalization, seed, arm):
    config = json.loads((source / 'config.json').read_text())
    expected = {'arm': arm, 'seed': seed, 'budget': 50, 'population': 300,
                'iterations': 30, 'topk': 30, 'with_target': True}
    if arm == 'reference':
        expected['approach_weight'] = .1
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(f'W4c source planner mismatch: {source}')
    norm_hash = hashlib.sha256(json.dumps(normalization, sort_keys=True).encode()).hexdigest()
    if config['normalization_sha256'] != norm_hash:
        raise ValueError('W4c normalization mismatch')
    if arm != 'reference' and config['checkpoint_weights_sha256'] != hashlib.sha256(
            (checkpoint_dir(arm) / 'weights.pt').read_bytes()).hexdigest():
        raise ValueError('W4c pretrained weights changed')
    # W4c controls must be exactly the source population, not just matching IDs.
    original_path = source.parent / 'bases.json'
    if config['bases_sha256'] != hashlib.sha256(original_path.read_bytes()).hexdigest():
        raise ValueError('W4c base source changed')
    if [asdict(b) for b in load_bases(original_path)] != [asdict(b) for b in bases]:
        raise ValueError('W4c source bases differ')
    rows = [r for r in read_records(source) if r['condition'] in CONTROLS]
    if len(rows) != len(bases) * 3 or {(r['base_id'], r['condition']) for r in rows} != {
            (b.id, c) for b in bases for c in CONTROLS}:
        raise ValueError('W4c controls incomplete or duplicate')
    return rows


def _complete(rows, bases):
    expected = {(b.id, 'near_path') for b in bases}
    if len(rows) != len(expected) or {(r['base_id'], r['condition']) for r in rows} != expected:
        raise ValueError('near_path episodes incomplete or duplicate')


def grid(run_dir=None):
    root = prepare_output_root(run_dir or runs_root() / 'w4d-calibration')
    root.mkdir(parents=True, exist_ok=True)
    source = runs_root() / 'w4c-calibration'
    bases = load_bases(source / 'bases.json')
    normalization = json.loads((source / 'normalization.json').read_text())
    write_json(root / 'normalization.json', normalization)
    f_source = source / 'b50-cem300x30-w0.1'
    control_f = _controls(f_source, bases, normalization, F, 'reference')
    control_e = _controls(source / 'chosen-reference', bases, normalization, E, 'reference')
    control_l = _controls(source / 'chosen-lewm', bases, normalization, E, 'lewm-pusht')
    write_json(root / 'reuse.json', {'source_bases': str(source / 'bases.json'),
        'source_bases_sha256': hashlib.sha256((source / 'bases.json').read_bytes()).hexdigest(),
        'controls': list(CONTROLS), 'F_reference': str(f_source),
        'E_reference': str(source / 'chosen-reference'), 'E_pretrained': str(source / 'chosen-lewm'),
        'source_condition_files_and_populations_retained_at_source': True})
    table = []
    for L in OFFSETS:
        cell = root / f'L{L}'
        cell.mkdir(exist_ok=True)
        modified, excluded = [], []
        for base in bases:
            try:
                modified.append(with_near_path(base, L))
            except ValueError as exc:
                excluded.append({'base_id': base.id, 'reason': str(exc)})
        path = cell / 'bases.json'
        save_bases(modified, path)
        naive = [naive_disturbance(b) for b in modified]
        write_json(cell / 'construction.json', {'L': L, 'valid': len(modified), 'excluded': excluded,
            'naive_replays': naive, 'disturbance_definition': 'maximum displacement >10 px over original 25-step witness'})
        if modified:
            evaluate('reference', path, conditions='near_path', n=len(modified), seed=F,
                run_dir=cell / 'reference-F', normalization=normalization, approach_weight=.1, **OPTIONS)
            rows = read_records(cell / 'reference-F')
            _complete(rows, modified)
        else:
            rows = []
        ids = {b.id for b in modified}
        off = [r for r in control_f if r['condition'] == 'off_path' and r['base_id'] in ids]
        successes = sum(r['score']['success'] for r in rows)
        rate = successes / len(modified) if modified else 0
        common = {r['base_id'] for r in rows if r['score']['success']} & {r['base_id'] for r in off if r['score']['success']}
        row = {'L': L, 'n_valid': len(modified), 'n_total': len(bases), 'successes': successes,
               'success_rate': rate, 'success_rate_all_bases': successes / len(bases),
               'naive_disturbed': sum(r['disturbed'] for r in naive),
               'naive_disturbance_rate': sum(r['disturbed'] for r in naive) / len(modified) if modified else 0,
               'off_path_rate_same_valid_bases': np.mean([r['score']['success'] for r in off]).item() if off else None,
               'reference_difference_abs': abs(rate - np.mean([r['score']['success'] for r in off])) if off else 1,
               'reference_common_n': len(common), 'reference_common_difference_abs': 0 if common else 1,
               'source_run': str(cell / 'reference-F'), 'bases_path': str(path)}
        table.append(row)
        write_json(root / 'grid.json', table)
        print(json.dumps(row), flush=True)
    chosen = select_offset(table)
    write_json(root / 'chosen.json', chosen)
    if chosen:
        path = Path(chosen['bases_path'])
        modified = load_bases(path)
        evaluate('reference', path, conditions='near_path', seed=E, n=len(modified),
                 run_dir=root / 'near-reference-E', normalization=normalization, approach_weight=.1, **OPTIONS)
        evaluate('lewm-pusht', path, conditions='near_path', seed=E, n=len(modified),
                 run_dir=root / 'near-pretrained-E', normalization=normalization, **OPTIONS)
        near_f = read_records(Path(chosen['source_run']))
        near_e = read_records(root / 'near-reference-E')
        near_l = read_records(root / 'near-pretrained-E')
        for rows in (near_f, near_e, near_l):
            _complete(rows, modified)
    else:
        near_f, near_e, near_l = [], [], []
    feasibility, reference, learned = control_f + near_f, control_e + near_e, control_l + near_l
    for name, rows in (('reference-F', feasibility), ('reference-E', reference), ('pretrained-E', learned)):
        destination = root / name
        destination.mkdir(exist_ok=True)
        (destination / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        write_json(destination / 'summary.json', summarize(rows, E if name != 'reference-F' else F))
    result = evidence(bases, chosen, feasibility, reference, learned)
    write_json(root / 'evidence.json', result)
    print(json.dumps(result), flush=True)
    return result


def evidence(bases, chosen, feasibility, reference, learned):
    result = {'chosen_L': chosen['L'] if chosen else None,
        'reference': summarize(reference, E), 'pretrained': summarize(learned, E),
        'common_feasible': {}, 'sample_size': {}, 'primary_contrasts': []}
    for label, pair in (('D_prime', ('near_path', 'off_path')), ('G', ('move_peg', 'move_T_matched'))):
        solved = [{r['base_id'] for r in feasibility if r['condition'] == name and r['score']['success']} for name in pair]
        common = set.intersection(*solved)
        arm_summaries = {arm: summarize([r for r in rows if r['base_id'] in common and r['condition'] in pair], E)
                         for arm, rows in (('reference', reference), ('pretrained', learned))}
        contrast = arm_summaries['reference']['contrasts'].get(label)
        rates = result['reference']['conditions']
        targets = (.6, .8)
        ready = bool(common and contrast and abs(contrast['difference']) < .05 - 1e-12
                     and all(rates.get(name, {}).get('success_rate', 0) >= target for name, target in zip(pair, targets)))
        result['common_feasible'][label] = {'n': len(common), 'excluded': len(bases) - len(common),
            'base_ids': sorted(common), 'arms': arm_summaries, 'ready': ready,
            'selection': 'F=23042; independent E=23043; calibration only'}
        if label == 'G' or (chosen and ready):
            result['primary_contrasts'].append(label)
        if common:
            values = [{r['base_id']: r['score']['success'] for r in learned if r['condition'] == name} for name in pair]
            discordant = sum(values[0][i] != values[1][i] for i in common)
            gap, remedy = [paired_sample_size(discordant, len(common), effect=e) for e in (.15, .10)]
            fraction, lower = len(common) / len(bases), wilson_ci(len(common), len(bases))[0]
            effective = max(gap['effective_bases'], remedy['effective_bases'])
            result['sample_size'][label] = {'gap': gap, 'remedy_proxy': remedy,
                'eligibility_fraction': fraction, 'eligibility_fraction_lower': lower,
                'bases_at_observed_eligibility': math.ceil(effective / fraction),
                'bases_at_lower_eligibility': math.ceil(effective / lower)}
    used = result['primary_contrasts']
    result['D_dropped'] = 'D_prime' not in used
    result['GO'] = all(result['common_feasible'][label]['ready'] for label in used)
    sizes = [result['sample_size'][label]['bases_at_lower_eligibility'] for label in used if label in result['sample_size']]
    N = math.ceil(max(sizes) / 100) * 100 if len(sizes) == len(used) else None
    result['planning_N'] = N
    conditions = ['off_path', 'near_path', 'move_peg', 'move_T_matched'] if not result['D_dropped'] else ['move_peg', 'move_T_matched']
    r = [row for row in reference if row['condition'] in conditions]
    l = [row for row in learned if row['condition'] in conditions]
    rm, lm = np.mean([x['wall_s'] for x in r]).item(), np.mean([x['wall_s'] for x in l]).item()
    result['time_estimate'] = {'checkpoints': ['lewm-pusht', 'ft_block_s0', 'ft_block_s1', 'ft_mixed_s0', 'ft_mixed_s1'],
        'repeats': 1, 'conditions': conditions, 'reference_mean_episode_s': rm, 'pretrained_mean_episode_s': lm,
        'pretrained_mean_planning_s': np.mean([sum(x['planning_times_s']) for x in l]).item(),
        'learned_episodes': 5 * len(conditions) * N if N else None,
        'reference_F_E_episodes': 2 * len(conditions) * N if N else None,
        'Mac_learned_hours': 5 * len(conditions) * N * lm / 3600 if N else None,
        'Mac_reference_hours': 2 * len(conditions) * N * rm / 3600 if N else None,
        'Mac_total_hours': len(conditions) * N * (5 * lm + 2 * rm) / 3600 if N else None,
        'assumption': 'Measured Mac episode means; pretrained is a latency proxy for all five checkpoints. Serial episodes, eight reference CPU candidate workers. Excludes training/generation/readout/ABC. Controls reused from W4c.'}
    return result


def freeze_protocol(run_dir=None):
    """Update the canonical protocol only after complete W4d evidence exists."""
    root = Path(run_dir or runs_root() / 'w4d-calibration')
    result = json.loads((root / 'evidence.json').read_text())
    table = json.loads((root / 'grid.json').read_text())
    chosen = json.loads((root / 'chosen.json').read_text())
    if [r['L'] for r in table] != list(OFFSETS):
        raise ValueError('all four bounded offsets required before freezing')
    source_bases = load_bases(runs_root() / 'w4c-calibration' / 'bases.json')
    if chosen:
        selected_bases = load_bases(chosen['bases_path'])
        for name in ('near-reference-E', 'near-pretrained-E'):
            _complete(read_records(root / name), selected_bases)
        common_ids = set(result['common_feasible']['D_prime']['base_ids'])
        naive = json.loads((Path(chosen['bases_path']).parent / 'construction.json').read_text())['naive_replays']
        disturbed = sum(r['disturbed'] for r in naive if r['base_id'] in common_ids)
        result['common_feasible']['D_prime']['naive_disturbance'] = {
            'disturbed': disturbed, 'n': len(common_ids),
            'rate': disturbed / len(common_ids) if common_ids else None}
    for name in ('reference-F', 'reference-E', 'pretrained-E'):
        rows = read_records(root / name)
        controls = [r for r in rows if r['condition'] in CONTROLS]
        if len(controls) != 3 * len(source_bases) or len({(r['base_id'], r['condition']) for r in controls}) != len(controls):
            raise ValueError('complete unique controls required before freezing')
    write_json(root / 'evidence.json', result)
    status = 'GO' if result['GO'] else 'NO-GO'
    used_d = not result['D_dropped']
    primary = "D' = near_path vs off_path; G = move_peg vs move_T_matched" if used_d else 'G = move_peg vs move_T_matched only; D is dropped'
    repo = Path(__file__).resolve().parents[3]
    protocol = repo / 'experiments/role_swap/PROTOCOL.md'
    old = protocol.read_text()
    semantics = old[old.index('## Semantics, rendering, normalization, and scoring'):old.index('## Seeds')]
    semantics = semantics.replace("`$LEWM_WORK_ROOT/runs/w4c-calibration/normalization.json`", "`$LEWM_WORK_ROOT/runs/w4d-calibration/normalization.json` (exact W4c copy)")
    bar = old[old.index('## Frozen decision bar'):].replace('D or G shows', "D' or G shows")
    lines = ['# Role-swap Step-4 protocol — W4d', '',
        f'Status: **{status} for the primary contrasts below**. No main evaluation has run.',
        f'Primary contrasts: **{primary}**.',
        'W4d is one bounded redesign of W4c D. G and the physical planner/scoring are unchanged.',
        'The prior W4c midpoint `on_path` is retained as an explicitly requested diagnostic,',
        'not a primary condition: its independent reference success was 0/20.', '',
        '## Task construction', '',
        'Sample the same non-overlapping agent/T/clutter-peg scene with unchanged physics.',
        'Run 300 physical steps of BlockWeakPolicy. D uses the first 25-step window',
        'whose start agent is within 40 px of T, T translation is >=40 px, and peg',
        'position is stationary throughout (absolute tolerance 1e-6 px). Retain the',
        'exact start snapshot including body velocities; goals use the real t+25',
        'agent/T states. `off_path` retains the original peg, >=80 px from the T',
        'center start-goal segment. `on_path` places the peg at its exact midpoint.',
        '`near_path` uses the same D start and goal, replacing only the peg with',
        'a perpendicular offset L from the segment midpoint. Its goal peg is fixed',
        'there; peg linear/angular velocities are zero. A per-base seeded side',
        '(NumPy default_rng(scene_seed), choice([-1,1])) is preferred independently',
        'of L and outcomes; try the opposite side only if geometry is invalid.',
        'Require no peg overlap with agent/T at both endpoints and center within',
        '[45,467]^2. Do not jitter, alter goals, or replace calibration bases.',
        'Record bases where neither side fits; exclude them from D eligibility.', '',
        'For G, restore the original scene and run PegWeakPolicy (30 px box) for',
        '300 steps. Select the first 25-step window with agent within 40 px of peg,',
        'peg translation 40–100 px and T pose stationary throughout (1e-6 tolerance).',
        '`move_peg` uses its real future agent/peg, with T fixed. `move_T_matched`',
        'uses a separate block window in the same scene, agent within 40 px of T,',
        'T translation in the same 20 px peg displacement bin ([40,60), [60,80),',
        '[80,100]), wrapped rotation <pi/9 and stationary peg.',
        'D/G share scene IDs but may have different exact starts and agent goals.',
        'Replay original off_path/move_peg/move_T_matched witnesses from fresh',
        'simulator spaces before scene acceptance; require unchanged final joint',
        'success. This rejects Chipmunk contact-cache-sensitive restored futures.',
        'Witnesses never initialize either planner. For near_path the original D',
        'witness is instead replayed with the relocated peg to measure **naive',
        'disturbance**: maximum peg displacement >10 px during all 25 steps.',
        'This establishes whether the obstacle affects that witness, not whether',
        'near_path is solvable; reference F/E establish planner feasibility.', '',
        'New main scenes use `generate_bases(N, 300000000, near_path_distance=L)`',
        'when D is retained; geometry-invalid candidates are rejected. Calibration',
        'uses `with_near_path` on the exact 20 stored W4c bases. Legacy BaseScene',
        'files remain readable but require their original revision for evaluation.', '',
        semantics.strip(), '', '## Seeds and bounded calibration selection', '',
        'Reuse the exact W4c calibration master 2,100,000,000 and its 20 accepted',
        'scene IDs (previous pilot/W4b disjointness remains verified). F_cal=23042',
        'alone selects L; E_cal=23043 independently validates reference and pretrained.',
        'This is calibration, not main evaluation or main feasibility. Main master',
        '300,000,000 / [300,000,000,300,999,999], F=101 and E=202 remain reserved.',
        'Main learned planning namespace 42 is shared across the **five trained',
        'checkpoints**: lewm-pusht, ft_block_s0, ft_block_s1, ft_mixed_s0, ft_mixed_s1.',
        'One evaluation repeat each. Episode seeds are low 32 bits of',
        'SHA256(namespace:base_id:condition), identical across arms.', '',
        'Test L={40,55,70,85} px only. Reference is frozen at 50 env steps,',
        '300x30 CEM, top-30, approach w=0.1 and eight CPU candidate workers.',
        'LeWM is 50 steps, 300x30, top-30 on MPS with upstream fixed decoration.',
        'Original W4c control episodes are reused after exact base, config,',
        'normalization and checkpoint-hash checks; source condition/population',
        'artifacts remain at their original run roots. Only near_path is new.',
        'A candidate requires near_path F success >=60% of the original 20 bases',
        '(geometry exclusions count against coverage) and nonzero naive disturbance.',
        'Among candidates meeting the reference F common-set |near-off| <5-point',
        'bar, maximize naive disturbance; tie by smaller L. Report the unconditional',
        'difference as a companion, not an additional selection screen.',
        'If none meets the common-set bar, minimize its difference, then maximize naive',
        'disturbance, then smaller L. A bounded fallback is diagnostic and cannot',
        'relax the independent E <5-point frozen bar on the F common feasible set.',
        'F common-set success differences are zero by construction, not validation.',
        'Do not select a second L after inspecting E. Retain D only if chosen L',
        'passes independent E targets (off >=80%, near >=60%), nonempty F common',
        'eligibility and reference E absolute difference <5 points. Otherwise drop D',
        'and use G only. G requires move_peg >=60%, matched >=80%, nonempty common',
        'eligibility and reference E absolute difference <5 points. GO applies only',
        'to the contrasts retained and does not authorize a main run automatically.', '',
        '## Calibration evidence and frozen settings', '',
        '| L (px) | Geometry valid | Reference F success | Success / all 20 | Naive disturbance | F abs near-off (points) |',
        '|---:|---:|---:|---:|---:|---:|']
    for cell in table:
        lines.append(f"| {cell['L']} | {cell['n_valid']}/20 | {cell['successes']}/{cell['n_valid']} ({100*cell['success_rate']:.1f}%) | "
            f"{100*cell['success_rate_all_bases']:.1f}% | {cell['naive_disturbed']}/{cell['n_valid']} ({100*cell['naive_disturbance_rate']:.1f}%) | "
            f"{100*cell['reference_difference_abs']:.1f} |")
    if chosen:
        lines += ['', f"Chosen **L={chosen['L']} px**; naive disturbance **{chosen['naive_disturbed']}/{chosen['n_valid']} "
            f"({100*chosen['naive_disturbance_rate']:.1f}%)**. F unconditional abs difference "
            f"**{100*chosen['reference_difference_abs']:.1f} points**."]
    else:
        lines += ['', '**No feasible L in the bounded sweep; D is dropped. G-only.**']
    if result['D_dropped'] and chosen:
        lines += ['**Chosen L fails independent D readiness; D is dropped. G-only.**']
    lines += ['', 'Independent E=23043 validation (all available bases; unchanged controls reused):', '',
        '| Condition | Reference | Pretrained | Reference mean s | Pretrained mean s |',
        '|:---|---:|---:|---:|---:|']
    for name in ('off_path', 'near_path', 'move_peg', 'move_T_matched'):
        if name not in result['reference']['conditions']:
            continue
        r, l = [result[arm]['conditions'][name] for arm in ('reference', 'pretrained')]
        lines.append(f"| {name} | {round(r['success_rate']*r['n'])}/{r['n']} ({100*r['success_rate']:.1f}%) | "
            f"{round(l['success_rate']*l['n'])}/{l['n']} ({100*l['success_rate']:.1f}%) | {r['mean_wall_s']:.2f} | {l['mean_wall_s']:.2f} |")
    lines += ['', 'F-selected common feasible sets (calibration eligibility only):', '']
    for label in ('D_prime', 'G'):
        c = result['common_feasible'][label]
        display = "D'" if label == 'D_prime' else label
        lines.append(f"- {display}: **{c['n']}/20 included, {c['excluded']} excluded**; independent readiness {'passes' if c['ready'] else 'fails'}.")
        if label == 'D_prime' and c.get('naive_disturbance', {}).get('n'):
            naive = c['naive_disturbance']
            lines.append(f"  Naive disturbance on this common set: **{naive['disturbed']}/{naive['n']} ({100*naive['rate']:.1f}%)**; "
                'eligibility removes some witness-disturbing bases, limiting D interaction coverage.')
        for arm in ('reference', 'pretrained'):
            summary = c['arms'][arm]
            if label not in summary['contrasts']:
                continue
            d = summary['contrasts'][label]
            rates = ', '.join(f"{name} {100*v['success_rate']:.1f}%" for name, v in summary['conditions'].items())
            lines.append(f"  {arm}: {rates}; hard-control difference **{100*d['difference']:.1f} points**, "
                f"paired bootstrap 95% CI [{100*d['ci'][0]:.1f}, {100*d['ci'][1]:.1f}].")
    lines += ['', f'**{status}** for {primary}. No main evaluation has run.', '', '## Sample size and measured Mac timing', '',
        'Use the inherited paired-binary approximation separately on each retained',
        'F-selected common set. q is pretrained hard/control discordance; use its',
        'Wilson 95% upper bound (at least the effect). Effective bases for delta in',
        '{0.15,0.10} = ceil((z_0.975+z_0.80)^2*(q_upper-delta^2)/delta^2). Divide',
        'the larger count by the Wilson lower bound on F eligibility. Take the',
        'maximum over retained contrasts and round up to the next hundred. The',
        '10-point count is a proxy, not measured cross-arm power. This powers CI',
        'exclusion of zero, not the full compound bar or guaranteed fine-tuned effects.', '']
    for label in result['primary_contrasts']:
        size = result['sample_size'].get(label)
        if not size:
            lines.append(f'{label}: N unestimable (empty common feasible set).')
            continue
        lines.append(f"{label}: discordance {size['gap']['discordance']:.4f}, Wilson q upper {size['gap']['discordance_upper']:.4f}; "
            f"effective bases {size['gap']['effective_bases']} (15 points), {size['remedy_proxy']['effective_bases']} (10 points); "
            f"eligibility {size['eligibility_fraction']:.4f}, Wilson lower {size['eligibility_fraction_lower']:.4f}; "
            f"candidate bases {size['bases_at_observed_eligibility']} (observed), {size['bases_at_lower_eligibility']} (lower bound).")
    t = result['time_estimate']
    lines += ['', f"Planning **N={result['planning_N']} candidate bases**; five checkpoints x1 repeat x{len(t['conditions'])} primary conditions.",
        f"Measured independent episode means: reference **{t['reference_mean_episode_s']:.3f} s**, pretrained **{t['pretrained_mean_episode_s']:.3f} s** "
        f"(LeWM planning **{t['pretrained_mean_planning_s']:.3f} s**)."]
    if result['planning_N']:
        lines += [f"Mac learned: **{t['learned_episodes']} episodes, {t['Mac_learned_hours']:.2f} h**; "
            f"reference F/E: **{t['reference_F_E_episodes']} episodes, {t['Mac_reference_hours']:.2f} h**; "
            f"total **{t['Mac_total_hours']:.2f} h**."]
    lines += [t['assumption'], 'Calibration base generation and contact-cache witness filtering are additional unprojected work.', '',
        'Full evidence: `$LEWM_WORK_ROOT/runs/w4d-calibration` (`grid.json`,',
        '`chosen.json`, `L*/construction.json`, `L*/reference-F`, `near-reference-E`,',
        '`near-pretrained-E`, merged `reference-F/reference-E/pretrained-E`,',
        '`reuse.json`, `evidence.json`, `protocol_freeze.json`, `driver.log`).',
        'Reproduce the bounded workflow with `python -m lewm_research.probe.near_path`',
        'using the project environment; resume rejects changed episode configs.',
        'Tests and validation are recorded under `$LEWM_WORK_ROOT/runs/w4d-tests`.',
        'No commit was made.', '', bar.strip()]
    protocol.write_text('\n'.join(lines).rstrip() + '\n')
    write_json(root / 'protocol_freeze.json', {'GO': result['GO'], 'chosen_L': result['chosen_L'],
        'primary_contrasts': result['primary_contrasts'], 'planning_N': result['planning_N'],
        'time_estimate': t, 'F_cal': F, 'E_cal': E,
        'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ('conditions.py', 'rollout_eval.py', 'near_path.py')},
        'normalization_sha256': hashlib.sha256(json.dumps(json.loads((root / 'normalization.json').read_text()), sort_keys=True).encode()).hexdigest(),
        'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest()})
    return result


if __name__ == '__main__':
    grid()
    freeze_protocol()
