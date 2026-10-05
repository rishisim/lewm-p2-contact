#!/usr/bin/env python3
"""Replay one consumed Cube pilot start and emit one bounded terminal-pixel batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PRIOR = REPO / "runs/lewm_frozen_gate_planning_bridge"
sys.path.insert(0, str(PRIOR))

import phase_c_generate as prior_generate  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def run(start_index: int, output: Path) -> dict[str, Any]:
    config = read_json(ROOT / "CONFIG.json")
    pilot = read_json(PRIOR / "PHASE_C_PILOT_CONFIG.json")
    if not 0 <= start_index < int(config["consumed_pilot"]["start_count"]):
        raise ValueError("start index outside fixed consumed cohort")
    expected_spec = config["consumed_pilot"]["seed_tuples"][start_index]
    prior_spec = pilot["starts"][start_index]
    for key in ("start_id", "env_seed", "policy_seed", "oracle_np_seed", "candidate_seed"):
        if expected_spec[key] != prior_spec[key]:
            raise RuntimeError(f"consumed identifier drift for {key}")

    simulator_path = PRIOR / "phase_c_simulator.npz"
    with np.load(simulator_path, allow_pickle=False) as stored:
        prior = {name: stored[name][start_index].copy() for name in stored.files}
    if str(prior["start_id"]) != expected_spec["start_id"]:
        raise RuntimeError("stored start identifier drift")
    for key in ("env_seed", "policy_seed", "oracle_np_seed", "candidate_seed"):
        if int(prior[key]) != int(expected_spec[key]):
            raise RuntimeError(f"stored seed drift for {key}")

    generator, seedfix, mujoco, runtime = prior_generate.load_generation_modules()
    world, policy = generator.make_world("plan_oracle")
    try:
        env = world.envs.envs[0].unwrapped
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        policy.set_seed(int(expected_spec["policy_seed"]))
        seedfix.deterministic_environment_reset(
            world, world.reset, int(expected_spec["env_seed"])
        )
        np.random.seed(int(expected_spec["oracle_np_seed"]))
        frames: dict[int, np.ndarray] = {}
        actions: list[np.ndarray] = []
        snapshot = None
        reconstructed_start_position = None
        reconstructed_target_position = None
        for action_index in range(prior_generate.WARMUP_STEPS + prior_generate.FUTURE_STEPS):
            action = np.asarray(policy.get_action(world.infos), dtype=np.float32)[0]
            actions.append(action.copy())
            prior_generate.step_world(world, action)
            step_count = action_index + 1
            if step_count in prior_generate.HISTORY_STEPS:
                frames[step_count] = prior_generate.pixel(world)
            if step_count == prior_generate.WARMUP_STEPS:
                snapshot = mujoco.MjData(env._model)
                mujoco.mj_copyData(snapshot, env._model, env._data)
                reconstructed_start_position = np.asarray(
                    world.infos["privileged/block_0_pos"]
                )[0, 0].copy()
                reconstructed_target_position = np.asarray(
                    world.infos["privileged/target_block_pos"]
                )[0, 0].copy()
        if snapshot is None or set(frames) != set(prior_generate.HISTORY_STEPS):
            raise RuntimeError("failed to reconstruct fixed warm-start state")
        if reconstructed_start_position is None or reconstructed_target_position is None:
            raise RuntimeError("missing reconstructed privileged verification state")

        action_array = np.asarray(actions, dtype=np.float32)
        reconstructed_initial_pixels = np.stack(
            [frames[step] for step in prior_generate.HISTORY_STEPS]
        )
        reconstructed_past_actions = action_array[65:75]
        reconstructed_goal_pixels = prior_generate.render_goal(env, snapshot, mujoco)
        reconstructed_candidates = prior_generate.make_candidates(
            action_array[75:100], low, high, int(expected_spec["candidate_seed"])
        )
        reconstructed_state_hash = prior_generate.state_digest(
            [
                reconstructed_initial_pixels,
                reconstructed_past_actions,
                reconstructed_start_position,
                reconstructed_target_position,
                snapshot.qpos,
                snapshot.qvel,
            ]
        )

        exact_checks = {
            "initial_pixels": np.array_equal(reconstructed_initial_pixels, prior["initial_pixels"]),
            "goal_pixels": np.array_equal(reconstructed_goal_pixels, prior["goal_pixels"]),
            "past_actions": np.array_equal(reconstructed_past_actions, prior["past_actions"]),
            "candidate_actions": np.array_equal(
                reconstructed_candidates, prior["candidate_actions"]
            ),
            "initial_state_sha256": reconstructed_state_hash
            == str(prior["initial_state_sha256"]),
        }
        if not all(exact_checks.values()):
            failed = [name for name, passed in exact_checks.items() if not passed]
            raise RuntimeError(f"exact simulator reconstruction failed: {failed}")

        position_atol = float(
            config["reproduction_tolerances"]["simulator_position_and_error_atol_m"]
        )
        start_position_max_abs = max_abs(
            reconstructed_start_position, prior["start_position"]
        )
        target_position_max_abs = max_abs(
            reconstructed_target_position, prior["target_position"]
        )
        if start_position_max_abs > position_atol or target_position_max_abs > position_atol:
            raise RuntimeError("warm-start or target position reconstruction exceeded tolerance")

        candidate_count = int(config["consumed_pilot"]["candidate_count_per_start"])
        terminal_pixels = np.empty((candidate_count, 224, 224, 3), dtype=np.uint8)
        terminal_position = np.empty((candidate_count, 3), dtype=np.float64)
        terminal_error = np.empty(candidate_count, dtype=np.float64)
        restore_state_max_abs = 0.0
        for candidate_index, candidate in enumerate(reconstructed_candidates):
            prior_generate.restore(env, snapshot, mujoco)
            restore_state_max_abs = max(
                restore_state_max_abs,
                max_abs(env._data.qpos, snapshot.qpos),
                max_abs(env._data.qvel, snapshot.qvel),
            )
            info = None
            for action in candidate:
                _, _, _, _, info = env.step(action)
            if info is None:
                raise RuntimeError("candidate execution returned no final info")
            pixel = np.asarray(env.render()).copy()
            if pixel.shape != (224, 224, 3) or pixel.dtype != np.uint8:
                raise RuntimeError(f"terminal render contract drift: {pixel.shape} {pixel.dtype}")
            position = np.asarray(info["privileged/block_0_pos"], dtype=np.float64)
            target = np.asarray(info["privileged/target_block_pos"], dtype=np.float64)
            terminal_pixels[candidate_index] = pixel
            terminal_position[candidate_index] = position
            terminal_error[candidate_index] = np.linalg.vector_norm(position - target)

        terminal_position_max_abs = max_abs(
            terminal_position, prior["real_terminal_position"]
        )
        terminal_error_max_abs = max_abs(terminal_error, prior["real_terminal_error"])
        if terminal_position_max_abs > position_atol or terminal_error_max_abs > position_atol:
            raise RuntimeError(
                "replayed terminal position/error exceeded fixed tolerance: "
                f"position={terminal_position_max_abs:.3g}, error={terminal_error_max_abs:.3g}"
            )

        prior_generate.restore(env, snapshot, mujoco)
        repeat_info = None
        for action in reconstructed_candidates[0]:
            _, _, _, _, repeat_info = env.step(action)
        if repeat_info is None:
            raise RuntimeError("repeat candidate returned no final info")
        repeat_position = np.asarray(
            repeat_info["privileged/block_0_pos"], dtype=np.float64
        )
        repeat_pixel = np.asarray(env.render()).copy()
        restore_repeat_position_max_abs = max_abs(repeat_position, terminal_position[0])
        restore_repeat_pixel_exact = bool(np.array_equal(repeat_pixel, terminal_pixels[0]))
        if restore_repeat_position_max_abs != 0.0 or not restore_repeat_pixel_exact:
            raise RuntimeError("full-state restore did not exactly repeat candidate 0")
    finally:
        world.close()

    arrays = {
        "start_id": np.asarray(expected_spec["start_id"], dtype="U20"),
        "start_index": np.asarray(start_index, dtype=np.int16),
        "terminal_pixels": terminal_pixels,
        "terminal_pixels_sha256": np.asarray(sha256_array(terminal_pixels), dtype="U64"),
        "real_terminal_position": terminal_position,
        "real_terminal_error": terminal_error,
        "initial_state_sha256": np.asarray(reconstructed_state_hash, dtype="U64"),
        "start_position_max_abs": np.asarray(start_position_max_abs, dtype=np.float64),
        "target_position_max_abs": np.asarray(target_position_max_abs, dtype=np.float64),
        "terminal_position_max_abs": np.asarray(
            terminal_position_max_abs, dtype=np.float64
        ),
        "terminal_error_max_abs": np.asarray(terminal_error_max_abs, dtype=np.float64),
        "restore_state_max_abs": np.asarray(restore_state_max_abs, dtype=np.float64),
        "restore_repeat_position_max_abs": np.asarray(
            restore_repeat_position_max_abs, dtype=np.float64
        ),
        "restore_repeat_pixel_exact": np.asarray(restore_repeat_pixel_exact, dtype=np.bool_),
        "candidate_actions_bitwise_exact": np.asarray(
            exact_checks["candidate_actions"], dtype=np.bool_
        ),
    }
    write_npz_atomic(output, arrays)
    return {
        "start_id": expected_spec["start_id"],
        "runtime": runtime,
        "candidate_actions_bitwise_exact": exact_checks["candidate_actions"],
        "terminal_position_max_abs": terminal_position_max_abs,
        "terminal_error_max_abs": terminal_error_max_abs,
        "restore_repeat_exact": True,
        "bounded_pixel_batch_shape": list(terminal_pixels.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start_index", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.start_index, args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
