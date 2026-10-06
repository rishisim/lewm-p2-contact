"""W4d placement, witness metric, selection, and primary-condition checks."""
from dataclasses import asdict

import numpy as np
import pytest

from lewm_research.envs.pusht_peg import PushTPeg
from lewm_research.probe.conditions import (
    generate_bases, with_near_path, naive_disturbance,
    segment_distance, save_bases,
)
from lewm_research.probe.near_path import select_offset
from lewm_research.probe.rollout_eval import evaluate


@pytest.fixture(scope='module')
def base():
    return generate_bases(1, 0)[0]


def test_near_geometry_and_original_future(base):
    original = asdict(base)
    modified = with_near_path(base, 55)
    assert original == asdict(base)
    assert asdict(modified) == asdict(with_near_path(base, 55))
    off = base.rollout_tasks['off_path']
    task = modified.rollout_tasks['near_path']
    start = np.array(off['snapshot']['bodies']['block']['position'])
    goal = np.array(off['goal_state'][2:4])
    xy = np.array(task['snapshot']['bodies']['peg']['position'])
    assert segment_distance(xy, start, goal) == pytest.approx(55)
    assert np.dot(xy - (start + goal)/2, goal - start) == pytest.approx(0, abs=1e-8)
    assert task['goal_state'][:7] == off['goal_state'][:7]
    assert task['snapshot']['bodies']['agent'] == off['snapshot']['bodies']['agent']
    assert task['snapshot']['bodies']['block'] == off['snapshot']['bodies']['block']
    assert task['witness_actions'] == off['witness_actions']
    assert 'witness_replay_score' not in task
    env = PushTPeg(with_target=False)
    try:
        env.reset(seed=0)
        env.restore_snapshot(task['snapshot'])
        assert not env.peg_overlaps(xy)
        env._set_state(np.array(task['goal_state']))
        assert not env.peg_overlaps(xy)
        assert np.all((xy >= 45) & (xy <= 467))
    finally:
        env.close()
    for name in base.rollout_tasks:
        assert modified.rollout_tasks[name] == base.rollout_tasks[name]
    generated = generate_bases(1, 0, near_path_distance=55)[0]
    assert asdict(generated) == asdict(modified)


def test_invalid_offset_and_no_silent_jitter(base):
    for L in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match='positive finite'):
            with_near_path(base, L)
    with pytest.raises(ValueError, match='no non-overlapping'):
        with_near_path(base, 1000)


def test_naive_metric_is_original_witness_maximum(base):
    modified = with_near_path(base, 55)
    metric = naive_disturbance(modified)
    env = PushTPeg(with_target=False)
    try:
        env.reset(seed=0)
        env.restore_snapshot(modified.rollout_tasks['near_path']['snapshot'])
        xy = np.array(env.peg.position)
        distances = [0.]
        for action in base.rollout_tasks['off_path']['witness_actions']:
            env.step(np.array(action, dtype=np.float32))
            distances.append(np.linalg.norm(np.array(env.peg.position) - xy))
        assert metric['max_peg_displacement_px'] == pytest.approx(max(distances))
        assert metric['disturbed'] == (max(distances) > 10)
        assert metric == naive_disturbance(modified)
    finally:
        env.close()


def test_bounded_selection():
    def row(L, success, naive, diff):
        return dict(L=L, success_rate_all_bases=success,
                    naive_disturbance_rate=naive, reference_common_difference_abs=diff)
    table = [row(40, .55, .9, 0), row(55, .8, .5, 0), row(70, .9, .25, 0), row(85, .9, .8, .1)]
    assert select_offset(table)['L'] == 55
    assert select_offset([row(40, .6, .8, .1), row(55, .7, .2, .05)])['L'] == 55
    assert select_offset([row(40, .55, 1, 0), row(55, .9, 0, 0)]) is None


def test_primary_defaults_and_independent_feasibility(base, tmp_path, monkeypatch):
    monkeypatch.setenv('LEWM_WORK_ROOT', str(tmp_path))
    root = tmp_path / 'runs'
    root.mkdir()
    path = root / 'bases.json'
    save_bases([with_near_path(base, 55)], path)
    options = dict(population=4, iterations=1, topk=2, workers=1, normalization={
        'columns': {'action': {'mean': [0,0], 'std': [.2,.2]},
                    'proprio': {'mean': [0,0,0,0], 'std': [1,1,1,1]}}})
    first = evaluate('reference', path, n=1, seed=1, run_dir=root / 'F', **options)
    assert set(first['unconditional']['conditions']) == {'off_path', 'near_path', 'move_peg', 'move_T_matched'}
    assert 'D_prime' in first['unconditional']['contrasts']
    assert 'D' not in first['unconditional']['contrasts']
    second = evaluate('reference', path, n=1, seed=2, run_dir=root / 'E', feasibility_run=root / 'F', **options)
    assert set(second['common_feasible']) == {'D_prime', 'G'}
    assert second == evaluate('reference', path, n=1, seed=2, run_dir=root / 'E', feasibility_run=root / 'F', **options)
