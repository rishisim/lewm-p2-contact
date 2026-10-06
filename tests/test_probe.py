"""Fast physical, serialization, solver, and statistical checks for W4."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from lewm_research.envs.pusht_peg import PushTPeg, PEG_COLOR
from lewm_research.probe.conditions import (
    _free, generate_bases, make_conditions, segment_distance, save_bases, load_bases,
    save_conditions, load_conditions, score_trajectory, role_cost, role_cost_endpoint,
)
from lewm_research.probe.reference import SimulatorCost
from lewm_research.probe.rollout_eval import Planner, run_episode, PersistedScaler, episode_seed
from lewm_research.probe.stats import wilson_ci, paired_cluster_bootstrap, spearman_bootstrap, top1_regret


@pytest.fixture(scope="module")
def base():
    return generate_bases(1, 0)[0]


@pytest.mark.parametrize("scene_seed", [0, 1, 2])
def test_rollout_geometry_and_reachability(scene_seed):
    """Witnesses must solve original futures; on-path changes geometry only."""
    base = generate_bases(1, scene_seed)[0]
    env = PushTPeg(with_target=False)
    env.reset(seed=0)
    try:
        conditions = make_conditions(base)
        starts = {}
        for name,c in conditions.items():
            env._setup()
            env.restore_snapshot(c.start_snapshot)
            start = env._get_obs().copy()
            starts[name] = start
            object_xy = start[7:9] if name=="move_peg" else start[2:4]
            assert np.linalg.norm(start[:2]-object_xy) <= 40
            if name != "on_path":
                trajectory = [start]
                for action in base.rollout_tasks[name]['witness_actions']:
                    trajectory.append(env.step(action)[0]['state'].copy())
                assert score_trajectory(trajectory,c)['success']
            else:
                assert not env.peg_overlaps(start[7:9])
                env._set_state(np.asarray(c.goal_state))
                assert not env.peg_overlaps(c.goal_state[7:9])
        off,on = conditions['off_path'],conditions['on_path']
        assert segment_distance(starts['off_path'][7:9],starts['off_path'][2:4],off.goal_state[2:4])>=80
        np.testing.assert_allclose(starts['on_path'][7:9],
            (starts['off_path'][2:4]+np.asarray(off.goal_state[2:4]))/2)
        for name in ('move_peg','move_T_matched'):
            assert 40<=base.rollout_tasks[name]['displacement_px']<=100
        assert base.rollout_tasks['move_peg']['displacement_bin']==base.rollout_tasks['move_T_matched']['displacement_bin']
        angle = abs((conditions['move_T_matched'].goal_state[4]-starts['move_T_matched'][4]+np.pi)%(2*np.pi)-np.pi)
        assert angle < np.pi/9
    finally:
        env.close()


def test_determinism_and_roundtrip(base, tmp_path):
    from dataclasses import replace
    with pytest.raises(ValueError,match="archived pre-W4c"):
        make_conditions(replace(base,rollout_tasks=None))
    assert asdict(generate_bases(1, 0)[0]) == asdict(base)
    save_bases([base], tmp_path / "bases.json")
    assert asdict(load_bases(tmp_path / "bases.json")[0]) == asdict(base)
    conditions = make_conditions(base)
    save_conditions(conditions, tmp_path / "conditions.json")
    loaded = load_conditions(tmp_path / "conditions.json")
    for name, c in conditions.items():
        assert loaded[name].start_snapshot == c.start_snapshot
        assert loaded[name].goal_state == c.goal_state
        np.testing.assert_array_equal(loaded[name].goal_image, c.goal_image)
        xy = np.asarray(c.goal_state[7:9]) * 224 / 512
        np.testing.assert_allclose(c.goal_image[round(xy[1]), round(xy[0])], PEG_COLOR, atol=5)


@pytest.mark.parametrize("radius", [15, 45])
def test_base_radius_roundtrip_and_legacy(base, tmp_path, radius):
    path = tmp_path/"bases.json"
    scene = replace(base, peg_radius=radius)
    save_bases([scene], path)
    assert json.loads(path.read_text())[0]["peg_radius"] == radius
    assert asdict(load_bases(path)[0]) == asdict(scene)
    legacy = asdict(scene)
    del legacy["peg_radius"]
    path.write_text(json.dumps([legacy]))
    assert load_bases(path)[0].peg_radius == 15


def test_conditions_render_base_radius(base, tmp_path):
    conditions = make_conditions(replace(base, peg_radius=45), with_target=False)
    env = PushTPeg(resolution=224, with_target=False, peg_radius=45)
    try:
        env.reset(seed=0)
        for condition in conditions.values():
            assert condition.peg_radius == 45
            np.testing.assert_array_equal(condition.goal_image, env.render_state(condition.goal_state))
        assert not np.array_equal(conditions["move_peg"].goal_image,
                                  make_conditions(base, with_target=False)["move_peg"].goal_image)
    finally:
        env.close()
    save_conditions(conditions, tmp_path/"conditions.json")
    assert all(c.peg_radius == 45 for c in load_conditions(tmp_path/"conditions.json").values())


def test_generate_passes_radius_to_gym(monkeypatch):
    from lewm_research.probe import conditions
    def make(name, **kwargs):
        assert name == "swm/PushTPeg-v1" and kwargs["peg_radius"] == 45
        raise RuntimeError("radius reached gym")
    monkeypatch.setattr(conditions.gym, "make", make)
    with pytest.raises(RuntimeError, match="radius reached gym"):
        generate_bases(1, 0, peg_radius=45)


def test_scoring_away_and_back(base):
    c = make_conditions(base)["off_path"]
    endpoint = np.asarray(c.goal_state)
    away = endpoint.copy()
    away[7] += 25
    scores = score_trajectory([away, endpoint], c)
    assert scores["t_success"] and not scores["peg_preserved"] and not scores["success"]
    assert not scores["ever_success"]
    assert role_cost([away, endpoint], c) == pytest.approx(50)
    assert role_cost_endpoint([away, endpoint], c) == pytest.approx(0)
    wrap = endpoint.copy()
    wrap[4] += 2 * np.pi
    assert score_trajectory([wrap], c)["success"]
    peg = make_conditions(base)["move_peg"]
    assert score_trajectory([peg.goal_state], peg)["success"]
    moved_t = np.asarray(peg.goal_state).copy()
    moved_t[2] += 21
    assert not score_trajectory([moved_t], peg)["success"]


@pytest.mark.parametrize("radius", [15, 45])
def test_simulator_manual_and_pool(base, radius):
    base = replace(base, peg_radius=radius)
    c = make_conditions(base)["off_path"]
    stats = {"columns": {"action": {"mean": [0.1, -0.2], "std": [0.5, 0.7]}}}
    candidate = torch.zeros(1, 2, 5, 10)
    env = PushTPeg(with_target=False, peg_radius=radius)
    env.reset(seed=0)
    env.restore_snapshot(c.start_snapshot)
    trajectory = [env._get_obs().copy()]
    for _ in range(25):
        trajectory.append(env.step(np.array([0.1, -0.2]))[0]["state"].copy())
    env.close()
    population_costs = []
    for workers in (1, 2):
        cost = SimulatorCost(c, stats, workers)
        try:
            np.testing.assert_allclose(cost.get_cost({}, candidate), role_cost(trajectory, c), rtol=1e-6)
            if workers == 1:
                from lewm_research.probe import reference
                assert reference._WORKER_ENV.peg_radius == radius
            candidates = torch.randn(1, 8, 5, 10, generator=torch.Generator().manual_seed(8))
            first = cost.get_cost({}, candidates)
            torch.testing.assert_close(first, cost.get_cost({}, candidates), rtol=0, atol=0)
            population_costs.append(first)
            cost.prior_max = 1000
            assert cost.get_cost({}, candidate).min() >= 2000
        finally:
            cost.close()
    torch.testing.assert_close(*population_costs, rtol=0, atol=0)


def test_controlled_loop(base, tmp_path):
    c = make_conditions(base)["off_path"]
    planner = Planner("reference", population=4, iterations=1, topk=2)
    try:
        a = run_episode(base, c, planner, 7, tmp_path / "population.npz")
        b = run_episode(base, c, planner, 7)
        assert a["trajectory"] == b["trajectory"]
        assert a["actions"] == b["actions"]
        assert len(a["trajectory"]) == 51 and len(a["actions"]) == 50
        assert len(a["planning_times_s"]) == 2
        longer = run_episode(base, c, planner, 7, budget=100)
        extended = run_episode(base,c,planner,7,budget=100,prefix_record=a,
                               prefix_population=tmp_path / "population.npz")
        assert extended["trajectory"] == longer["trajectory"]
        assert extended["actions"] == longer["actions"]
        assert extended["score"] == longer["score"]
        assert longer["trajectory"][:51] == a["trajectory"]
        assert longer["actions"][:50] == a["actions"]
        assert longer["budget_checkpoints"]["50"]["score"] == a["score"]
        bank = np.load(tmp_path / "population.npz")
        assert bank["call0_candidates"].shape == (1, 4, 5, 10)
    finally:
        planner.close()


def test_stats():
    np.testing.assert_allclose(wilson_ci(5, 10), [0.23659309, 0.76340691], atol=1e-8)
    result = paired_cluster_bootstrap([1, 1, 0], [0, 0, 0], ["a", "a", "b"], seed=4)
    assert result == paired_cluster_bootstrap([1, 1, 0], [0, 0, 0], ["a", "a", "b"], seed=4)
    assert result["difference"] == pytest.approx(2 / 3)
    assert result["bases"] == 2
    assert spearman_bootstrap([1, 2, 3, 4], [4, 3, 2, 1], samples=50)["rho"] == -1
    assert top1_regret([3, 1, 2], [0, 5, 2]) == 5
    assert episode_seed(42, "base", "on_path") == episode_seed(42, "base", "on_path")
    scaler = PersistedScaler({"columns": {"action": {"mean": [1, 2], "std": [3, 4]}}}, "action")
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(np.array([[2, 5]]))), [[2, 5]])


def test_resume_and_separate_feasibility(base, tmp_path, monkeypatch):
    from lewm_research.probe.rollout_eval import evaluate

    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    tmp_path = tmp_path / "runs"
    tmp_path.mkdir()
    bases = tmp_path / "bases.json"
    save_bases([base], bases)
    options = dict(population=4, iterations=1, topk=2, workers=1, normalization={
        "columns": {"action": {"mean": [0,0], "std": [.2,.2]},
                    "proprio": {"mean": [0,0,0,0], "std": [1,1,1,1]}}})
    first = evaluate("reference", bases, n=1, run_dir=tmp_path / "F", seed=1, **options)
    path = tmp_path/"F/config.json"
    legacy = json.loads(path.read_text())
    del legacy["peg_radius"]
    path.write_text(json.dumps(legacy))
    resumed = evaluate("reference", bases, n=1, run_dir=tmp_path / "F", seed=1, **options)
    assert first == resumed
    path.write_text(json.dumps(legacy))
    assert len((tmp_path / "F" / "episodes.jsonl").read_text().splitlines()) == 4
    extended = evaluate("reference", bases, n=1, run_dir=tmp_path / "F100", seed=1,
                        budget=100, prefix_run=tmp_path / "F", **options)
    assert extended["unconditional"]["episodes"] == 4
    assert extended == evaluate("reference", bases, n=1, run_dir=tmp_path / "F100", seed=1,
                               budget=100, prefix_run=tmp_path / "F", **options)
    with pytest.raises(ValueError, match="prefix config differs"):
        evaluate("reference", bases, n=1, run_dir=tmp_path / "wrong-prefix", seed=2,
                 budget=100, prefix_run=tmp_path / "F", **options)
    with pytest.raises(ValueError, match="resume configuration"):
        evaluate("reference", bases, n=1, run_dir=tmp_path / "F", seed=2, **options)
    second = evaluate("reference", bases, n=1, run_dir=tmp_path / "E", seed=2,
                      feasibility_run=tmp_path / "F", **options)
    assert second["common_feasible"]["D"]["feasibility_seed"] == 1
    with pytest.raises(ValueError, match="match feasibility planner"):
        evaluate("reference", bases, n=1, run_dir=tmp_path / "shaped-E", seed=2,
                 feasibility_run=tmp_path / "F", approach_weight=.1, **options)


def test_no_peg_has_no_physics_or_pixels():
    from stable_worldmodel.envs.pusht.env import PushT
    env = PushTPeg(peg_enabled=False, with_target=False)
    upstream = PushT(with_target=False)
    state = np.array([120, 120, 280, 240, 0.3, 0, 0, -1000, -1000.])
    try:
        obs, info = env.reset(seed=7, options={"state": state, "goal_state": state})
        upstream.reset(seed=7, options={"state": state[:7], "goal_state": state[:7]})
        # Compare exact pose initialization, without the upstream setter's dt step.
        upstream.agent.position = tuple(state[:2])
        upstream.block.angle = state[4]
        upstream.block.position = tuple(state[2:4])
        assert env.peg is None
        np.testing.assert_array_equal(env.goal_state[7:9], [-1000, -1000])
        assert len(env.space.bodies) == len(upstream.space.bodies)
        assert obs["state"].shape == (9,)
        np.testing.assert_array_equal(obs["state"][7:], [-1000, -1000])
        np.testing.assert_array_equal(obs["proprio"], state[[0,1,5,6]])
        np.testing.assert_array_equal(env.render(), upstream.render())
        np.testing.assert_array_equal(env.render_state(state), upstream.render())
        snapshot = env.get_snapshot()
        assert set(snapshot["bodies"]) == {"agent", "block"}
        action = np.array([0.1, 0.2])
        a = env.step(action)[0]["state"]
        env.restore_snapshot(snapshot)
        b = env.step(action)[0]["state"]
        np.testing.assert_allclose(a, b, atol=1e-8)
    finally:
        env.close()
        upstream.close()


def test_probe_outputs_reject_repo_and_other_roots(tmp_path, monkeypatch):
    from lewm_research.probe.rollout_eval import prepare_output_root
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    assert prepare_output_root(tmp_path / "runs" / "probe") == tmp_path / "runs" / "probe"
    with pytest.raises(ValueError, match="repository"):
        prepare_output_root("runs/probe")
    with pytest.raises(ValueError, match="runs_root"):
        prepare_output_root(tmp_path / "other")


def test_fixed_upstream_marker_is_independent_of_task_goal():
    from stable_worldmodel.envs.pusht.env import PushT
    env = PushTPeg(peg_enabled=False, with_target=True,
                   render_target_pose=(256,256,np.pi/4))
    upstream = PushT(with_target=True)
    state = np.array([120,120,280,240,.3,0,0,-1000,-1000.])
    try:
        env.reset(seed=7, options={"state":state, "goal_state":state})
        upstream.reset(seed=7, options={"state":state[:7], "goal_state":state[:7]})
        upstream.agent.position = tuple(state[:2])
        upstream.block.angle = state[4]
        upstream.block.position = tuple(state[2:4])
        np.testing.assert_array_equal(env.render(), upstream.render())
        np.testing.assert_array_equal(env.render_state(state), upstream.render())
        old_pixels = env.render().copy()
        changed_goal = state.copy()
        changed_goal[2:4] += 80
        env.set_goal(changed_goal)
        # Goal fields change; the observation's dataset decoration does not.
        np.testing.assert_array_equal(env.goal_pose, changed_goal[2:5])
        np.testing.assert_array_equal(env.render(), old_pixels)
    finally:
        env.close()
        upstream.close()


def test_reuse_requires_identical_base_not_just_id(base, tmp_path):
    from dataclasses import replace
    from lewm_research.probe.calibration import reuse_identical_bases
    source, destination = tmp_path/"source", tmp_path/"destination"
    source.mkdir()
    save_bases([base],tmp_path/"a.json")
    save_bases([replace(base, agent_xy=[1,2])],tmp_path/"b.json")
    (source/"episodes.jsonl").write_text(json.dumps({"base_id":base.id,"condition":"off_path"})+"\n")
    reuse_identical_bases(source,destination,tmp_path/"a.json",tmp_path/"b.json")
    assert not destination.exists()
    save_bases([base],tmp_path/"b.json")
    reuse_identical_bases(source,destination,tmp_path/"a.json",tmp_path/"b.json")
    reuse_identical_bases(source,destination,tmp_path/"a.json",tmp_path/"b.json")
    assert len((destination/"episodes.jsonl").read_text().splitlines()) == 1


def test_reference_ipc_payload_excludes_unused_rendered_snapshots(base, monkeypatch):
    import pickle
    import lewm_research.probe.reference as reference
    condition = make_conditions(base)["off_path"]
    original = reference._simulate
    checked = []
    def check_payload(task):
        # A full 224x224 RGB frame in nested lists is hundreds of KB and gets
        # copied per CEM chunk. Physical state + role spec should stay tiny.
        assert len(pickle.dumps(task)) < 20_000
        checked.append(True)
        return original(task)
    monkeypatch.setattr(reference,"_simulate",check_payload)
    cost = SimulatorCost(condition,{"columns":{"action":{"mean":[0,0],"std":[.2,.2]}}})
    try:
        result = cost.get_cost({},torch.zeros(1,2,5,10))
        assert checked == [True,True]
        torch.testing.assert_close(result[0,0],result[0,1])
    finally:
        cost.close()


def test_reference_only_approach_shaping(base):
    c = make_conditions(base)['move_peg']
    stats = {'columns':{'action':{'mean':[0,0],'std':[1,1]}}}
    candidate = torch.zeros(1,2,5,10)
    costs = []
    for w in (0,.3):
        cost = SimulatorCost(c,stats,approach_weight=w)
        try:
            costs.append(cost.get_cost({},candidate))
        finally:
            cost.close()
    env = PushTPeg(with_target=False)
    env.reset(seed=0)
    env.restore_snapshot(c.start_snapshot)
    for _ in range(25):
        state = env.step([0,0])[0]['state']
    env.close()
    torch.testing.assert_close(costs[1]-costs[0],torch.full_like(costs[0],.3*np.linalg.norm(state[:2]-state[7:9])))
    with pytest.raises(ValueError,match='reference-only'):
        Planner('lewm-pusht',approach_weight=.1)


def test_per_arm_rendering_preserves_physics(base):
    yes,no = make_conditions(base,with_target=True),make_conditions(base,with_target=False)
    for name in yes:
        assert yes[name].goal_state==no[name].goal_state
        assert yes[name].start_snapshot['bodies']==no[name].start_snapshot['bodies']
        assert np.any(yes[name].goal_image != no[name].goal_image)


def test_calibration_refuses_incomplete_freeze(base, tmp_path):
    from lewm_research.probe.calibration import evidence
    save_bases([base],tmp_path/'bases.json')
    for name in ('F','chosen-reference','chosen-lewm'):
        (tmp_path/name).mkdir()
        (tmp_path/name/'episodes.jsonl').write_text('')
    (tmp_path/'chosen.json').write_text(json.dumps({'bases_path':str(tmp_path/'bases.json'),
                                                  'source_run':str(tmp_path/'F')}))
    with pytest.raises(ValueError,match='complete and unique'):
        evidence(tmp_path)
