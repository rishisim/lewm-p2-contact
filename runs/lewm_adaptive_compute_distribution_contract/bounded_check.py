#!/usr/bin/env python3
"""One-trajectory replay/action-semantics check after reconstruction failure.

This diagnostic is intentionally unavailable before the mechanical decision and
does not generate another policy trajectory.  It reuses the first frozen main
PlanOracle seed and the already-recorded 200 actions, then verifies that the
same installed environment reproduces the captured transition sequence.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

import common
import generator


def _load_raw(path: common.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def run() -> dict[str, Any]:
    common.assert_pre_generation_seal()
    output = common.STUDY_ROOT / "audit/bounded_reconstruction_check.json"
    if output.exists():
        return common.study_json(output)
    decision = common.study_json(common.STUDY_ROOT / "decision.json")
    if decision["decision"] != "generator_reconstruction_failed":
        raise RuntimeError(
            "bounded reconstruction check is authorized only after "
            "generator_reconstruction_failed"
        )

    manifest = common.study_json(
        common.role_raw_paths("main", "plan_oracle")[1]
    )
    record = manifest["episodes"][0]
    raw_path = common.REPO_ROOT / record["path"]
    raw = _load_raw(raw_path)
    actions = np.asarray(raw["action"][:200], dtype=np.float32)
    terminal_action = np.asarray(raw["action"][200])
    if actions.shape != (200, common.RAW_ACTION_DIM) or not np.isnan(terminal_action).all():
        raise RuntimeError("recorded action convention is malformed")

    world, policy = generator.make_world("plan_oracle")
    try:
        policy.set_seed(int(record["policy_seed"]))
        world.reset(seed=int(record["env_seed"]), options=None)
        np.random.seed(int(record["oracle_np_seed"]))
        states = [generator._v4_generator._capture(world.infos)]
        for action in actions:
            batch = action[None, :]
            _, reward, term, trunc, infos = world.envs.step(batch)
            world.infos = infos
            world.rewards = reward
            world.terminateds = term
            world.truncateds = trunc
            states.append(generator._v4_generator._capture(infos))
        replay = generator._v4_generator._stack_records(states)
        action_space = world.envs.action_space
        action_space_low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
        action_space_high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
    finally:
        world.close()

    comparisons: dict[str, Any] = {}
    for name, replayed in replay.items():
        if name not in raw:
            continue
        expected = np.asarray(raw[name])
        replayed = np.asarray(replayed)
        if expected.shape != replayed.shape:
            comparisons[name] = {
                "shape_equal": False,
                "expected_shape": list(expected.shape),
                "replayed_shape": list(replayed.shape),
            }
            continue
        if np.issubdtype(expected.dtype, np.number):
            delta = np.abs(expected.astype(np.float64) - replayed.astype(np.float64))
            max_abs = float(delta.max(initial=0.0))
        else:
            max_abs = 0.0 if np.array_equal(expected, replayed) else float("inf")
        comparisons[name] = {
            "shape_equal": True,
            "array_equal": bool(np.array_equal(expected, replayed)),
            "max_abs_difference": max_abs,
            "expected_sha256": common.array_sha256(expected),
            "replayed_sha256": common.array_sha256(replayed),
        }

    required = ("pixels", "observation", "qpos", "qvel")
    replay_exact = all(
        comparisons.get(name, {}).get("array_equal", False) for name in required
    )
    source = common.study_json(common.STUDY_ROOT / "source_manifest.json")
    result = {
        "schema_version": 1,
        "classification": "smallest_bounded_post-decision_replay_and_action-semantics_check",
        "authorized_by_decision": decision["decision"],
        "new_environment_seeds": 0,
        "new_policy_trajectories": 0,
        "replayed_existing_trajectory_count": 1,
        "replayed_role": "main_plan_oracle",
        "replayed_slot": int(record["slot"]),
        "reused_env_seed": int(record["env_seed"]),
        "raw_file_sha256_verified": common.sha256_file(raw_path) == record["file_sha256"],
        "recorded_initial_state_sha256": record["initial_state_sha256"],
        "replayed_initial_state_sha256": generator._initial_state_digest(states[0]),
        "initial_state_exact": record["initial_state_sha256"]
        == generator._initial_state_digest(states[0]),
        "transition_replay": {
            "required_exact_arrays": list(required),
            "required_arrays_exact": replay_exact,
            "arrays": comparisons,
        },
        "action_semantics": {
            "recorded_action_shape": list(actions.shape),
            "terminal_action_is_nan_sentinel": bool(np.isnan(terminal_action).all()),
            "action_space_low": action_space_low.tolist(),
            "action_space_high": action_space_high.tolist(),
            "all_actions_inside_environment_space": bool(
                np.all(actions >= action_space_low) and np.all(actions <= action_space_high)
            ),
            "raw_min": actions.min(0).astype(float).tolist(),
            "raw_max": actions.max(0).astype(float).tolist(),
            "no_action_rescaling_during_replay": True,
        },
        "installed_versions": manifest["package_versions"],
        "installed_controlling_source_hashes": source[
            "installed_controlling_sources"
        ],
        "upstream_public_horizon": 1001,
        "released_local_horizon": 200,
        "upstream_local_horizon_mismatch_remains": True,
        "gate_or_model_tuned": False,
        "additional_target_cache_opened": False,
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
        "passed": bool(
            replay_exact
            and record["initial_state_sha256"]
            == generator._initial_state_digest(states[0])
            and np.all(actions >= action_space_low)
            and np.all(actions <= action_space_high)
        ),
        "interpretation": (
            "Exact replay supports deterministic local capture and direct action semantics; "
            "it does not resolve a remaining mismatch to the released offline generator."
            if replay_exact
            else "Replay mismatch indicates a local determinism, capture, or action-semantics issue."
        ),
    }
    common.write_study_json(output, result, exclusive=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
