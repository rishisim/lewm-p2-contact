#!/usr/bin/env python3
"""Generate the fixed 20-start Cube pilot and execute 64 common candidates."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import bridge


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V5 = REPO / "runs/lewm_v5_readiness_program/v5_package_versions/v004"
GENERATION_PYTHON = Path(
    "/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python"
)
START_COUNT = 20
CANDIDATE_COUNT = 64
WARMUP_STEPS = 75
HISTORY_STEPS = (65, 70, 75)
FUTURE_STEPS = 25
NOISE_SCALES = (0.04, 0.08, 0.16, 0.24)
MATCHED_SEED = 3_814_000_000


def pilot_config() -> dict[str, Any]:
    starts = []
    for index in range(START_COUNT):
        starts.append(
            {
                "start_id": f"pilot-cube-{index:03d}",
                "env_seed": 3_810_000_000 + index,
                "policy_seed": 3_811_000_000 + index,
                "oracle_np_seed": 3_812_000_000 + index,
                "candidate_seed": 3_813_000_000 + index,
                "start_definition": "simulator state after 75 fixed PlanOracle warm-up actions",
            }
        )
    return {
        "schema_version": 1,
        "scientific_role": "pilot_only_candidate_ranking",
        "fixed_before_simulator_outcomes_or_model_costs": True,
        "start_count": START_COUNT,
        "candidate_count_per_start": CANDIDATE_COUNT,
        "candidate_horizon_raw_steps": FUTURE_STEPS,
        "candidate_horizon_action_blocks": 5,
        "history_frame_steps": list(HISTORY_STEPS),
        "history_past_action_steps": [65, 75],
        "candidate_pool": {
            "procedure": (
                "candidate 0 is the frozen PlanOracle actions at steps 75:100; candidates 1:63 "
                "add candidate-seeded blockwise Gaussian noise to motion/yaw coordinates only, "
                "preserve the PlanOracle gripper coordinate, and clip to simulator action bounds"
            ),
            "noise_scales_cycled": list(NOISE_SCALES),
            "condition_independent": True,
            "identical_candidates_for_all_model_conditions": True,
            "no_candidate_exclusion_or_replacement_by_outcome": True,
        },
        "goal": {
            "predicted_cost": "raw terminal latent MSE to a simulator-rendered goal image latent",
            "goal_image": (
                "render the identical warm-start simulator state with the physical target block "
                "placed at its fixed target pose, then restore the full MuJoCo state"
            ),
            "real_terminal_error": "3D Euclidean target-block position error in meters",
            "top_k": 5,
        },
        "matched_allocation": {
            "seed": MATCHED_SEED,
            "rule": "permute adaptive depths across the 64 candidates separately by start and rollout step",
            "exact_depth_histogram_refiner_calls_and_reached_gate_evaluations": True,
        },
        "planner_ranking_promising_rule": {
            "adaptive_mean_spearman_strictly_greater_than_matched": True,
            "adaptive_mean_real_regret_strictly_less_than_matched": True,
            "adaptive_mean_top5_overlap_at_least_matched": True,
            "finite_and_exact_compute_required": True,
        },
        "starts": starts,
        "no_large_mpc": True,
        "no_new_confirmation_cohort": True,
    }


def initialize_config() -> dict[str, Any]:
    phase_b = bridge.read_json(ROOT / "RESULTS.json")
    if not phase_b["phase_b"]["go_to_phase_c"]:
        raise RuntimeError("Phase-C pilot is forbidden because Phase B did not pass")
    path = ROOT / "PHASE_C_PILOT_CONFIG.json"
    expected = pilot_config()
    if path.exists():
        if bridge.read_json(path) != expected:
            raise RuntimeError("existing Phase-C pilot configuration drift")
    else:
        bridge.write_json(path, expected, exclusive=True)
        bridge.append_note(
            "Before any pilot simulator outcome or model cost, fixed 20 pilot start/goal seed "
            "tuples, 64 candidates per start, the PlanOracle-centered condition-independent "
            "proposal rule, top-k=5, matched seed, and planner-ranking decision rule."
        )
    return expected


def load_generation_modules() -> tuple[Any, Any, Any, Any]:
    if Path(sys.executable) != GENERATION_PYTHON:
        raise RuntimeError(f"expected sealed generation interpreter {GENERATION_PYTHON}")
    package_path = str(V5)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    cycle_common = importlib.import_module("cycle_common")
    runtime = cycle_common.assert_runtime_contract("generation")
    distribution_path = str(REPO / "runs/lewm_adaptive_compute_distribution_contract")
    if distribution_path not in sys.path:
        sys.path.insert(0, distribution_path)
    generator = importlib.import_module("generator")
    seedfix = importlib.import_module("generator_seedfix")
    import mujoco

    return generator, seedfix, mujoco, runtime


def step_world(world: Any, action: np.ndarray) -> None:
    _, reward, terminated, truncated, infos = world.envs.step(action[None])
    world.infos = infos
    world.rewards = reward
    world.terminateds = terminated
    world.truncateds = truncated


def pixel(world: Any) -> np.ndarray:
    value = np.asarray(world.infos["pixels"])[0, 0]
    if value.shape != (224, 224, 3) or value.dtype != np.uint8:
        raise RuntimeError("pilot pixel contract drift")
    return value.copy()


def state_digest(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_candidates(
    baseline: np.ndarray, low: np.ndarray, high: np.ndarray, seed: int
) -> np.ndarray:
    if baseline.shape != (FUTURE_STEPS, 5):
        raise ValueError("baseline action shape drift")
    candidates = np.repeat(baseline[None], CANDIDATE_COUNT, axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    for candidate in range(1, CANDIDATE_COUNT):
        scale = NOISE_SCALES[(candidate - 1) % len(NOISE_SCALES)]
        block_noise = rng.normal(0.0, scale, size=(5, 4)).astype(np.float32)
        candidates[candidate, :, :4] += np.repeat(block_noise, 5, axis=0)
        candidates[candidate] = np.clip(candidates[candidate], low, high)
        candidates[candidate, :, 4] = baseline[:, 4]
    hashes = {state_digest([candidate]) for candidate in candidates}
    if len(hashes) != CANDIDATE_COUNT:
        raise RuntimeError("candidate pool contains exact duplicates")
    return candidates


def render_goal(env: Any, snapshot: Any, mujoco: Any) -> np.ndarray:
    mujoco.mj_copyData(env._data, env._model, snapshot)
    target_mocap = env._cube_target_mocap_ids[env._target_block]
    joint = env._data.joint(f"object_joint_{env._target_block}")
    joint.qpos[:3] = env._data.mocap_pos[target_mocap]
    joint.qpos[3:] = env._data.mocap_quat[target_mocap]
    mujoco.mj_forward(env._model, env._data)
    goal = np.asarray(env.render()).copy()
    mujoco.mj_copyData(env._data, env._model, snapshot)
    mujoco.mj_forward(env._model, env._data)
    if goal.shape != (224, 224, 3) or goal.dtype != np.uint8:
        raise RuntimeError(f"goal render contract drift: {goal.shape} {goal.dtype}")
    return goal


def restore(env: Any, snapshot: Any, mujoco: Any) -> None:
    mujoco.mj_copyData(env._data, env._model, snapshot)
    env._prev_qpos = env._data.qpos.copy()
    env._prev_qvel = env._data.qvel.copy()
    env._success = False
    env._reset_next_step = False
    mujoco.mj_forward(env._model, env._data)


def run() -> dict[str, Any]:
    config = initialize_config()
    output = ROOT / "phase_c_simulator.npz"
    manifest_path = ROOT / "PHASE_C_SIMULATOR.json"
    if output.exists() or manifest_path.exists():
        raise RuntimeError("Phase-C simulator artifacts already exist")
    generator, seedfix, mujoco, runtime = load_generation_modules()
    start_ids = np.asarray([item["start_id"] for item in config["starts"]], dtype="U20")
    initial_pixels = np.empty((START_COUNT, 3, 224, 224, 3), dtype=np.uint8)
    goal_pixels = np.empty((START_COUNT, 224, 224, 3), dtype=np.uint8)
    past_actions = np.empty((START_COUNT, 10, 5), dtype=np.float32)
    candidate_actions = np.empty(
        (START_COUNT, CANDIDATE_COUNT, FUTURE_STEPS, 5), dtype=np.float32
    )
    real_error = np.empty((START_COUNT, CANDIDATE_COUNT), dtype=np.float64)
    terminal_position = np.empty((START_COUNT, CANDIDATE_COUNT, 3), dtype=np.float64)
    start_position = np.empty((START_COUNT, 3), dtype=np.float64)
    target_position = np.empty((START_COUNT, 3), dtype=np.float64)
    state_hashes = np.empty(START_COUNT, dtype="U64")
    restore_repeat_error = np.empty(START_COUNT, dtype=np.float64)
    world, policy = generator.make_world("plan_oracle")
    try:
        env = world.envs.envs[0].unwrapped
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        for start_index, spec in enumerate(config["starts"]):
            policy.set_seed(int(spec["policy_seed"]))
            seedfix.deterministic_environment_reset(
                world, world.reset, int(spec["env_seed"])
            )
            np.random.seed(int(spec["oracle_np_seed"]))
            frames: dict[int, np.ndarray] = {}
            actions: list[np.ndarray] = []
            snapshot = None
            for action_index in range(WARMUP_STEPS + FUTURE_STEPS):
                action = np.asarray(policy.get_action(world.infos), dtype=np.float32)[0]
                actions.append(action.copy())
                step_world(world, action)
                step_count = action_index + 1
                if step_count in HISTORY_STEPS:
                    frames[step_count] = pixel(world)
                if step_count == WARMUP_STEPS:
                    snapshot = mujoco.MjData(env._model)
                    mujoco.mj_copyData(snapshot, env._model, env._data)
                    start_position[start_index] = np.asarray(
                        world.infos["privileged/block_0_pos"]
                    )[0, 0]
                    target_position[start_index] = np.asarray(
                        world.infos["privileged/target_block_pos"]
                    )[0, 0]
            if snapshot is None or set(frames) != set(HISTORY_STEPS):
                raise RuntimeError("failed to capture fixed warm-start history")
            action_array = np.asarray(actions, dtype=np.float32)
            initial_pixels[start_index] = np.stack([frames[step] for step in HISTORY_STEPS])
            past_actions[start_index] = action_array[65:75]
            goal_pixels[start_index] = render_goal(env, snapshot, mujoco)
            candidates = make_candidates(
                action_array[75:100], low, high, int(spec["candidate_seed"])
            )
            candidate_actions[start_index] = candidates
            state_hashes[start_index] = state_digest(
                [
                    initial_pixels[start_index],
                    past_actions[start_index],
                    start_position[start_index],
                    target_position[start_index],
                    snapshot.qpos,
                    snapshot.qvel,
                ]
            )
            for candidate_index, candidate in enumerate(candidates):
                restore(env, snapshot, mujoco)
                info = None
                for action in candidate:
                    _, _, _, _, info = env.step(action)
                assert info is not None
                position = np.asarray(info["privileged/block_0_pos"], dtype=np.float64)
                target = np.asarray(info["privileged/target_block_pos"], dtype=np.float64)
                terminal_position[start_index, candidate_index] = position
                real_error[start_index, candidate_index] = np.linalg.vector_norm(
                    position - target
                )
            restore(env, snapshot, mujoco)
            repeat_info = None
            for action in candidates[0]:
                _, _, _, _, repeat_info = env.step(action)
            assert repeat_info is not None
            repeat_position = np.asarray(
                repeat_info["privileged/block_0_pos"], dtype=np.float64
            )
            restore_repeat_error[start_index] = np.max(
                np.abs(repeat_position - terminal_position[start_index, 0])
            )
            if restore_repeat_error[start_index] != 0.0:
                raise RuntimeError("full-state restore was not exactly repeatable")
            print(
                f"Phase C simulator {start_index + 1}/{START_COUNT} "
                f"real-error range={real_error[start_index].min():.5f}..{real_error[start_index].max():.5f}",
                flush=True,
            )
    finally:
        world.close()
    if not np.isfinite(real_error).all() or np.any(real_error < 0):
        raise RuntimeError("invalid pilot real outcomes")
    bridge.write_npz(
        output,
        {
            "start_id": start_ids,
            "env_seed": np.asarray([item["env_seed"] for item in config["starts"]], dtype=np.int64),
            "policy_seed": np.asarray(
                [item["policy_seed"] for item in config["starts"]], dtype=np.int64
            ),
            "oracle_np_seed": np.asarray(
                [item["oracle_np_seed"] for item in config["starts"]], dtype=np.int64
            ),
            "candidate_seed": np.asarray(
                [item["candidate_seed"] for item in config["starts"]], dtype=np.int64
            ),
            "initial_pixels": initial_pixels,
            "goal_pixels": goal_pixels,
            "past_actions": past_actions,
            "candidate_actions": candidate_actions,
            "real_terminal_error": real_error,
            "real_terminal_position": terminal_position,
            "start_position": start_position,
            "target_position": target_position,
            "initial_state_sha256": state_hashes,
            "restore_repeat_max_abs": restore_repeat_error,
        },
    )
    manifest = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "scientific_role": "pilot_only",
        "path": str(output.relative_to(REPO)),
        "sha256": bridge.sha256_file(output),
        "runtime": runtime,
        "config_path": str((ROOT / "PHASE_C_PILOT_CONFIG.json").relative_to(REPO)),
        "config_sha256": bridge.sha256_file(ROOT / "PHASE_C_PILOT_CONFIG.json"),
        "consumed_start_ids": start_ids.tolist(),
        "consumed_seed_tuples": config["starts"],
        "start_count": START_COUNT,
        "candidate_count_per_start": CANDIDATE_COUNT,
        "candidate_outcomes": START_COUNT * CANDIDATE_COUNT,
        "identical_full_state_per_candidate": True,
        "exact_restore_repeatability": True,
        "performance_or_success_based_exclusions": 0,
        "model_or_gate_loaded_during_simulator_generation": False,
        "v3_test_targets_opened": False,
        "released_hdf5_opened": False,
    }
    bridge.write_json(manifest_path, manifest, exclusive=True)
    bridge.append_note(
        "Generated and executed all 1,280 fixed pilot candidates across the 20 now-consumed "
        "start/goal identifiers. Full MuJoCo state restoration repeated candidate 0 exactly for "
        "every start; no performance-based exclusion occurred."
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "generate"))
    args = parser.parse_args()
    result = initialize_config() if args.command == "init" else run()
    print(json.dumps(bridge.jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
