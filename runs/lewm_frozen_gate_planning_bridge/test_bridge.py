from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import bridge
import phase_c_generate


def test_fixed_configuration_is_bounded_and_preselected() -> None:
    config = bridge.make_config()
    assert len(config["phase_a"]["episode_ids"]) == 8
    assert len(config["phase_b"]["episode_ids"]) == 100
    assert config["phase_b"]["episode_ids"][0].endswith("0000")
    assert config["phase_b"]["episode_ids"][-1].endswith("0099")
    assert config["phase_b"]["go_criterion"]["k5_whitened_relative_margin"] == 0.05


def test_permutations_are_deterministic_and_horizon_specific() -> None:
    first = bridge.fixed_permutation(3400, 0)
    repeat = bridge.fixed_permutation(3400, 0)
    next_horizon = bridge.fixed_permutation(3400, 1)
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, next_horizon)
    assert np.array_equal(np.sort(first), np.arange(3400))


def test_permutation_preserves_exact_depth_histogram() -> None:
    calls = np.tile(np.asarray([1, 1, 1, 2, 3, 4], dtype=np.int8), 20)
    assigned = calls[bridge.fixed_permutation(len(calls), 2)]
    assert np.array_equal(np.bincount(calls), np.bincount(assigned))
    assert int(calls.sum()) == int(assigned.sum())
    assert int(np.minimum(calls, 3).sum()) == int(np.minimum(assigned, 3).sum())


def test_case_window_contract() -> None:
    latents = np.arange(2 * 41 * 192, dtype=np.float32).reshape(2, 41, 192)
    blocks = np.arange(2 * 40 * 25, dtype=np.float32).reshape(2, 40, 25)
    episode, start, history, actions, targets = bridge.case_arrays(latents, blocks)
    assert episode.shape == start.shape == (68,)
    assert history.shape == (68, 3, 192)
    assert actions.shape == (5, 68, 3, 25)
    assert targets.shape == (5, 68, 192)
    row = 33
    assert np.array_equal(history[row], latents[0, 33:36])
    assert np.array_equal(actions[4, row], blocks[0, 37:40])
    assert np.array_equal(targets[4, row], latents[0, 40])


def test_loss_functions_match_direct_definitions() -> None:
    prediction = np.asarray([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    target = np.asarray([[0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    whitening = np.eye(2, dtype=np.float64)
    expected = np.square(prediction.astype(np.float64) - target).mean(axis=1)
    assert np.array_equal(bridge.raw_mse(prediction, target), expected)
    assert np.array_equal(bridge.whitened_mse(prediction, target, whitening), expected)


def test_source_does_not_name_forbidden_data_paths() -> None:
    source = Path(bridge.__file__).read_text(encoding="utf-8")
    forbidden_fragments = ("combined_v3", "cube_single_expert.h5", "v3_test_targets.npz")
    executable_source = source.split('"prohibited_inputs_opened"', maxsplit=1)[0]
    assert all(fragment not in executable_source for fragment in forbidden_fragments)


def test_phase_c_identifiers_and_seeds_are_fixed_and_unique() -> None:
    config = phase_c_generate.pilot_config()
    assert config["start_count"] == 20
    assert config["candidate_count_per_start"] == 64
    assert len({item["start_id"] for item in config["starts"]}) == 20
    assert len({item["env_seed"] for item in config["starts"]}) == 20


def test_phase_c_candidate_pool_is_deterministic_and_preserves_gripper() -> None:
    baseline = np.linspace(-0.8, 0.8, 125, dtype=np.float32).reshape(25, 5)
    low = np.full(5, -1.0, dtype=np.float32)
    high = np.full(5, 1.0, dtype=np.float32)
    first = phase_c_generate.make_candidates(baseline, low, high, 17)
    repeat = phase_c_generate.make_candidates(baseline, low, high, 17)
    assert np.array_equal(first, repeat)
    assert np.array_equal(first[0], baseline)
    assert np.array_equal(first[:, :, 4], np.repeat(baseline[None, :, 4], 64, axis=0))
