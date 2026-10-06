"""Fast correctness gates for the PushT-Peg environment and collection."""

import json

import gymnasium as gym
import numpy as np
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

import lewm_research.envs  # noqa: F401 - registers the Gymnasium id
from lewm_research.data.collect import collect
from lewm_research.envs.pusht_peg import PEG_COLOR, PushTPeg
from lewm_research.policies.weak import BlockWeakPolicy, MixedPolicy, PegWeakPolicy


def test_snapshot_and_render():
    env = PushTPeg(with_target=False)
    try:
        state, info = env.reset(seed=5, options={"peg_xy": (300, 300)})
        before = env.get_snapshot()
        altered = state["state"].copy()
        altered[7:9] = (100, 100)
        image = env.render_state(altered)
        assert image.shape == (224, 224, 3)
        assert np.allclose(env._get_obs(), state["state"])
        assert np.allclose(env.get_snapshot()["goal_state"], before["goal_state"])
        pixel = image[round(100 * 224 / 512), round(100 * 224 / 512)]
        np.testing.assert_allclose(pixel, PEG_COLOR, atol=5)
        assert np.array_equal(env.render(), env.render())
        actions = np.random.default_rng(8).uniform(-1, 1, (50, 2)).astype(np.float32)
        first = [env.step(action)[0]["state"].copy() for action in actions]
        env.restore_snapshot(before)
        second = [env.step(action)[0]["state"].copy() for action in actions]
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-8)
    finally:
        env.close()


def test_proprio_and_no_termination():
    env = PushTPeg()
    try:
        obs, info = env.reset(seed=1, options={"peg_xy": (100, 100)})
        np.testing.assert_array_equal(obs["proprio"], obs["state"][[0, 1, 5, 6]])
        np.testing.assert_array_equal(info["goal_proprio"], info["goal_state"][[0, 1, 5, 6]])
        env.set_goal(obs["state"])
        obs, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert not terminated
        np.testing.assert_array_equal(obs["proprio"], obs["state"][[0, 1, 5, 6]])
        np.testing.assert_allclose(info["peg_pos"], obs["state"][7:9])
    finally:
        env.close()


def test_uniform_placement_200_seeds():
    env = PushTPeg(with_target=False)
    try:
        for seed in range(200):
            obs, _ = env.reset(seed=seed, options={"peg_placement": "uniform"})
            xy = obs["state"][7:9]
            assert np.all((xy >= 45) & (xy <= 467))
            assert not env.peg_overlaps(xy)
    finally:
        env.close()


def test_policies_and_registration():
    env = gym.make("swm/PushTPeg-v1", with_target=False)
    try:
        env.reset(seed=2, options={"peg_xy": (100, 100), "agent_xy": (300, 300),
                                   "block_pose": (400, 400, 0)})
        for cls in (BlockWeakPolicy, PegWeakPolicy, MixedPolicy):
            policy = cls(seed=2)
            policy.set_env(env)
            if isinstance(policy, MixedPolicy):
                assert policy.begin_episode(0) in {"block", "peg"}
                assert 0 in policy.choices
            actions = np.array([policy.get_action()[0] for _ in range(20)])
            assert actions.shape == (20, 2)
            assert np.all(np.abs(actions) <= 1)
        peg = PegWeakPolicy(seed=2)
        peg.set_env(env)
        action = peg.get_action()[0]
        desired = np.asarray(env.unwrapped.agent.position) + action * 100
        assert np.all(desired >= 0) and np.all(desired <= 200)
    finally:
        env.close()


def test_mixed_policy_peg_limit_scales_with_radius():
    for kwargs, expected in (({}, 30), ({"peg_radius": 45}, 60)):
        env = gym.make("swm/PushTPeg-v1", with_target=False, **kwargs)
        try:
            policy = MixedPolicy(p_peg=1, seed=2)
            policy.set_env(env)
            policy.begin_episode(0)
            assert policy._limit(env.unwrapped, 0) == expected
            override = MixedPolicy(p_peg=1, peg_dist_constraint=42, seed=2)
            override.set_env(env)
            override.begin_episode(0)
            assert override._limit(env.unwrapped, 0) == 42
            policy.current_choice = "block"
            assert policy._limit(env.unwrapped, 0) == 100
        finally:
            env.close()


def test_collection_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    metadata = collect("tiny", episodes=2, steps=10, policy="mixed", seed=4, workers=2)
    dataset = HDF5Dataset(path=metadata["dataset"])
    assert dataset.lengths.tolist() == [10, 10]
    assert {"pixels", "action", "proprio", "state", "ep_idx", "step_idx"} <= set(dataset.column_names)
    episode = dataset.load_episode(0)
    assert tuple(episode["pixels"].shape) == (10, 3, 224, 224)
    assert tuple(episode["action"].shape) == (10, 2)
    assert tuple(episode["proprio"].shape) == (10, 4)
    assert tuple(episode["state"].shape) == (10, 9)
    assert json.loads((tmp_path / "stable-worldmodel/datasets/tiny.json").read_text())["frames"] == 20


def test_clutter_placement_keeps_distance():
    from lewm_research.envs.pusht_peg import CLUTTER_MIN_AGENT_DIST, CLUTTER_MIN_BLOCK_DIST

    env = PushTPeg(with_target=False)
    try:
        for seed in range(100):
            obs, _ = env.reset(seed=seed, options={"peg_placement": "clutter"})
            state = obs["state"]
            assert np.linalg.norm(state[7:9] - state[2:4]) >= CLUTTER_MIN_BLOCK_DIST
            assert np.linalg.norm(state[7:9] - state[0:2]) >= CLUTTER_MIN_AGENT_DIST
            assert not env.peg_overlaps(state[7:9])
    finally:
        env.close()


def test_collection_peg_radius(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    images = []
    for name, kwargs in (("default", {}), ("small", {"peg_radius": 15}), ("large", {"peg_radius": 45})):
        metadata = collect(name, episodes=1, steps=1, seed=4, **kwargs)
        radius = kwargs.get("peg_radius", 15)
        assert metadata["peg_radius"] == radius
        written = json.loads((tmp_path / f"stable-worldmodel/datasets/{name}.json").read_text())
        assert written["peg_radius"] == radius
        episode = HDF5Dataset(path=metadata["dataset"]).load_episode(0)
        images.append(np.asarray(episode["pixels"])[0].transpose(1, 2, 0))
    np.testing.assert_array_equal(images[0], images[1])
    counts = [np.all(image == PEG_COLOR, axis=-1).sum() for image in images]
    assert counts[2] > counts[1] > 0


def test_collection_rows_pair_observation_with_action_taken_from_it(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    metadata = collect("aligned", episodes=1, steps=30, policy="block", seed=7, workers=1)
    episode = HDF5Dataset(path=metadata["dataset"]).load_episode(0)
    state = np.asarray(episode["state"])
    action = np.asarray(episode["action"])
    # Row 0 is the reset state (agent at rest); action t moves the agent toward t+1.
    np.testing.assert_allclose(state[0, 5:7], 0.0)
    move = state[1:, :2] - state[:-1, :2]
    cos = np.sum(move * action[:-1], axis=1) / (np.linalg.norm(move, axis=1) * np.linalg.norm(action[:-1], axis=1) + 1e-9)
    assert np.mean(cos) > 0.5


def test_peg_radius_render_and_physics():
    options = {"agent_xy": (100,100), "block_pose": (250,250,0), "peg_xy": (400,100)}
    envs = [PushTPeg(with_target=False, **kwargs) for kwargs in ({}, {"peg_radius":15}, {"peg_radius":45})]
    try:
        images = []
        for env in envs:
            obs, _ = env.reset(seed=0, options=options)
            image = env._render_frame("rgb_array").copy()
            images.append(image)
            assert next(iter(env.peg.shapes)).radius == env.peg_radius
            assert env.peg.moment == env.peg_radius**2/2
            xy = env._sample_peg()
            assert np.all((xy >= 30+env.peg_radius) & (xy <= 482-env.peg_radius))
            np.testing.assert_array_equal(env.render_state(obs["state"]), image)
        assert envs[0].peg_radius == 15
        np.testing.assert_array_equal(images[0], images[1])
        counts = [np.all(image == PEG_COLOR, axis=-1).sum() for image in images]
        assert 7 < counts[2]/counts[0] < 12
        for env in envs:
            assert env.peg_overlaps((120,100))
        assert not envs[0].peg_overlaps((160,100))
        assert envs[2].peg_overlaps((160,100))
    finally:
        for env in envs:
            env.close()
