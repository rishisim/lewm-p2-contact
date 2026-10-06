"""Fast physical, serialization, solver, and statistical checks for W4."""

from dataclasses import asdict
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


def test_geometry_100_seeds():
    env = PushTPeg(with_target=False)
    env.reset(seed=0)
    try:
        # Property test on accepted scenes across 100 independent seed streams.
        for seed in range(100):
            b = generate_bases(1, seed)[0]
            conditions = make_conditions(b)
            start = np.array([*b.agent_xy, *b.block_pose, 0, 0, *b.peg_xy])
            assert _free(env, start)
            assert np.linalg.norm(np.asarray(b.t_goal_pose[:2]) - start[2:4]) >= 60
            off, on, peg, matched = (conditions[k] for k in ("off_path", "on_path", "move_peg", "move_T_matched"))
            assert segment_distance(b.peg_xy, b.block_pose[:2], b.t_goal_pose[:2]) >= 80
            assert segment_distance(on.scoring_spec["peg_start"], b.block_pose[:2], b.t_goal_pose[:2]) < 1e-8
            v = np.asarray(peg.goal_state[7:9]) - start[7:9]
            assert 60 <= np.linalg.norm(v) <= 100
            np.testing.assert_allclose(np.asarray(matched.goal_state[2:4]) - start[2:4], v)
            for c in conditions.values():
                env.restore_snapshot(c.start_snapshot)
                assert _free(env, env._get_obs())
                assert _free(env, np.asarray(c.goal_state))
            assert matched.goal_state[4] == start[4]
            assert peg.goal_state[:2] == matched.goal_state[:2] == b.agent_xy
    finally:
        env.close()


def test_determinism_and_roundtrip(base, tmp_path):
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


def test_simulator_manual_and_pool(base):
    c = make_conditions(base)["off_path"]
    stats = {"columns": {"action": {"mean": [0.1, -0.2], "std": [0.5, 0.7]}}}
    candidate = torch.zeros(1, 2, 5, 10)
    env = PushTPeg(with_target=False)
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
    resumed = evaluate("reference", bases, n=1, run_dir=tmp_path / "F", seed=1, **options)
    assert first == resumed
    assert len((tmp_path / "F" / "episodes.jsonl").read_text().splitlines()) == 4
    with pytest.raises(ValueError, match="resume configuration"):
        evaluate("reference", bases, n=1, run_dir=tmp_path / "F", seed=2, **options)
    second = evaluate("reference", bases, n=1, run_dir=tmp_path / "E", seed=2,
                      feasibility_run=tmp_path / "F", **options)
    assert second["common_feasible"]["D"]["feasibility_seed"] == 1


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
