#!/usr/bin/env python3
"""Collect a bounded StableWM Fetch interaction-error pilot dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    GRIPPER_GEOMS,
    OBJECT_GEOM,
    assign_event_taxonomy,
    is_gripper_object_pair,
    is_object_support_pair,
    is_task_object_pair,
    pair_key,
    pairs_to_text,
)


DEFAULT_ENVS = ["swm/FetchSlideDense-v3", "swm/FetchPushDense-v3"]
DEFAULT_OUTPUT_DIR = Path("le-wm/diagnostics/fetch_contact_compute/data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", dest="envs", default=None)
    parser.add_argument("--num-trajectories", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--render-size", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--response-window", type=int, default=3)
    parser.add_argument("--skip-pixels", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[fetch-collect] {message}", flush=True)


def import_env_stack():
    if platform.system() == "Darwin" and os.environ.get("MUJOCO_GL") == "egl":
        print_step("unsetting MUJOCO_GL=egl on macOS; Fetch runs with the native MuJoCo backend here")
        os.environ.pop("MUJOCO_GL", None)
    import gymnasium as gym
    import mujoco
    import stable_worldmodel.envs  # noqa: F401 - registers swm/* ids.

    return gym, mujoco


def geom_name(mujoco, model, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name or f"geom_{int(geom_id)}"


def contact_force_norm(mujoco, model, data, contact_index: int) -> float:
    force = np.zeros(6, dtype=np.float64)
    try:
        mujoco.mj_contactForce(model, data, int(contact_index), force)
    except Exception:
        return 0.0
    return float(np.linalg.norm(force[:3]))


def summarize_contacts(mujoco, model, data) -> dict[str, object]:
    all_pairs: set[str] = set()
    task_pairs: set[str] = set()
    gripper_pairs: set[str] = set()
    support_pairs: set[str] = set()
    max_force = 0.0
    sum_force = 0.0
    max_task_force = 0.0
    sum_task_force = 0.0
    max_gripper_force = 0.0
    max_support_force = 0.0
    timestep = float(getattr(model.opt, "timestep", 1.0))

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        name_a = geom_name(mujoco, model, int(contact.geom1))
        name_b = geom_name(mujoco, model, int(contact.geom2))
        key = pair_key(name_a, name_b)
        force_norm = contact_force_norm(mujoco, model, data, contact_index)
        all_pairs.add(key)
        max_force = max(max_force, force_norm)
        sum_force += force_norm

        if is_task_object_pair(key):
            task_pairs.add(key)
            max_task_force = max(max_task_force, force_norm)
            sum_task_force += force_norm
        if is_gripper_object_pair(key):
            gripper_pairs.add(key)
            max_gripper_force = max(max_gripper_force, force_norm)
        if is_object_support_pair(key):
            support_pairs.add(key)
            max_support_force = max(max_support_force, force_norm)

    return {
        "ncon_total": int(data.ncon),
        "contact_pairs": pairs_to_text(all_pairs),
        "task_contact_pairs": pairs_to_text(task_pairs),
        "task_contact_count": int(len(task_pairs)),
        "gripper_object_contact": bool(gripper_pairs),
        "object_support_contact": bool(support_pairs),
        "gripper_object_pairs": pairs_to_text(gripper_pairs),
        "object_support_pairs": pairs_to_text(support_pairs),
        "max_contact_force_norm": float(max_force),
        "sum_contact_force_norm": float(sum_force),
        "max_task_contact_force_norm": float(max_task_force),
        "sum_task_contact_force_norm": float(sum_task_force),
        "max_gripper_object_force_norm": float(max_gripper_force),
        "max_object_support_force_norm": float(max_support_force),
        "max_contact_impulse_proxy": float(max_task_force * timestep),
        "sum_contact_impulse_proxy": float(sum_task_force * timestep),
    }


def state_from_info(obs: np.ndarray, info: dict[str, object]) -> np.ndarray:
    return np.asarray(info.get("state", obs), dtype=np.float32).reshape(-1)


def object_pos(state: np.ndarray) -> np.ndarray:
    if state.shape[0] < 6:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(state[3:6], dtype=np.float32)


def gripper_pos(state: np.ndarray) -> np.ndarray:
    if state.shape[0] < 3:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(state[:3], dtype=np.float32)


def goal_pos(state: np.ndarray, info: dict[str, object]) -> np.ndarray:
    goal = info.get("goal_state")
    if goal is not None:
        arr = np.asarray(goal, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 3:
            return arr[:3]
    if state.shape[0] >= 3:
        return np.asarray(state[-3:], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def scripted_fetch_action(state: np.ndarray, info: dict[str, object], rng: np.random.Generator, noise_std: float) -> np.ndarray:
    grip = gripper_pos(state)
    obj = object_pos(state)
    goal = goal_pos(state, info)
    direction = goal - obj
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction[:2]))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm

    behind = obj - 0.07 * direction
    behind[2] = obj[2] + 0.015
    push_target = obj + 0.12 * direction
    push_target[2] = obj[2] + 0.015

    target = behind if np.linalg.norm((grip - behind)[:2]) > 0.035 else push_target
    action_xyz = 7.0 * (target - grip)
    action = np.zeros(4, dtype=np.float32)
    action[:3] = action_xyz
    action[3] = 0.0
    if noise_std > 0:
        action[:3] += rng.normal(0.0, noise_std, size=3).astype(np.float32)
    return np.clip(action, -1.0, 1.0)


def flatten_prefixed(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return {f"{prefix}_{i}": float(v) for i, v in enumerate(arr)}


def collect_env(args: argparse.Namespace, env_id: str) -> tuple[pd.DataFrame, np.ndarray | None, dict[str, object]]:
    gym, mujoco = import_env_stack()
    env_seed_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(env_id)) % 100000
    rng = np.random.default_rng(args.seed + env_seed_offset)
    env = gym.make(env_id, render_mode="rgb_array", resolution=args.render_size)
    records: list[dict[str, object]] = []
    pixels: list[np.ndarray] = []
    state_dim = None
    action_dim = None
    qpos_dim = None
    qvel_dim = None
    geom_names: list[str] = []

    try:
        model = env.unwrapped.model
        data = env.unwrapped.data
        geom_names = [geom_name(mujoco, model, i) for i in range(model.ngeom)]
        qpos_dim = int(data.qpos.shape[0])
        qvel_dim = int(data.qvel.shape[0])

        for episode_id in range(args.num_trajectories):
            obs, info = env.reset(seed=args.seed + episode_id)
            state = state_from_info(obs, info)
            state_dim = int(state.shape[0])
            action_dim = int(env.action_space.shape[0])
            last_object_pos = object_pos(state)
            last_object_vel = np.zeros(3, dtype=np.float32)

            for step_idx in range(args.max_steps):
                action = scripted_fetch_action(state, info, rng, args.noise_std)
                obs_next, reward, terminated, truncated, info_next = env.step(action)
                next_state = state_from_info(obs_next, info_next)
                current_object_pos = object_pos(next_state)
                object_vel = current_object_pos - last_object_pos
                object_vel_delta = object_vel - last_object_vel
                contact_summary = summarize_contacts(mujoco, model, data)

                row: dict[str, object] = {
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "object_pos_x": float(current_object_pos[0]),
                    "object_pos_y": float(current_object_pos[1]),
                    "object_pos_z": float(current_object_pos[2]),
                    "object_vel_x": float(object_vel[0]),
                    "object_vel_y": float(object_vel[1]),
                    "object_vel_z": float(object_vel[2]),
                    "object_velocity_delta": float(np.linalg.norm(object_vel_delta)),
                    "object_speed": float(np.linalg.norm(object_vel)),
                    "state_norm": float(np.linalg.norm(next_state)),
                    "reset_artifact": bool(step_idx == 0),
                    **contact_summary,
                    **flatten_prefixed("state", state),
                    **flatten_prefixed("next_state", next_state),
                    **flatten_prefixed("action", action),
                    **flatten_prefixed("qpos", data.qpos),
                    **flatten_prefixed("qvel", data.qvel),
                }
                records.append(row)

                if not args.skip_pixels:
                    frame = env.render()
                    if frame is not None:
                        pixels.append(np.asarray(frame, dtype=np.uint8))

                state = next_state
                info = info_next
                last_object_pos = current_object_pos
                last_object_vel = object_vel
                if terminated or truncated:
                    break
            if (episode_id + 1) % max(1, args.num_trajectories // 10) == 0:
                print_step(f"{env_id}: collected {episode_id + 1}/{args.num_trajectories} trajectories")
    finally:
        env.close()

    df = pd.DataFrame(records)
    df, thresholds = assign_event_taxonomy(
        df,
        history_size=args.history_size,
        response_window=args.response_window,
    )
    pixel_array = np.stack(pixels, axis=0) if pixels else None
    metadata = {
        "env_id": env_id,
        "num_trajectories_requested": args.num_trajectories,
        "max_steps": args.max_steps,
        "rows": int(len(df)),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "qpos_dim": qpos_dim,
        "qvel_dim": qvel_dim,
        "render_size": args.render_size,
        "pixels_stored": pixel_array is not None,
        "pixel_rows": int(pixel_array.shape[0]) if pixel_array is not None else 0,
        "geom_names": geom_names,
        "tracked_object_geom": OBJECT_GEOM,
        "tracked_gripper_geoms": sorted(GRIPPER_GEOMS),
        "taxonomy_thresholds": thresholds.as_dict(),
        "primary_regime_counts": df["primary_regime"].value_counts().to_dict(),
        "created_at_unix": time.time(),
    }
    return df, pixel_array, metadata


def write_env_outputs(output_dir: Path, env_id: str, df: pd.DataFrame, pixels: np.ndarray | None, metadata: dict[str, object], *, force: bool) -> None:
    safe_name = env_id.replace("/", "__")
    env_dir = output_dir / safe_name
    env_dir.mkdir(parents=True, exist_ok=True)
    records_path = env_dir / "records.csv.gz"
    metadata_path = env_dir / "metadata.json"
    pixels_path = env_dir / "pixels.npz"
    if not force and (records_path.exists() or metadata_path.exists() or pixels_path.exists()):
        raise FileExistsError(f"{env_dir} already has outputs; pass --force to overwrite")
    df.to_csv(records_path, index=False, compression="gzip")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if pixels is not None:
        np.savez_compressed(pixels_path, pixels=pixels)
    elif pixels_path.exists() and force:
        pixels_path.unlink()
    print_step(f"wrote {records_path} rows={len(df)}")
    print_step(f"wrote {metadata_path}")
    if pixels is not None:
        print_step(f"wrote {pixels_path} shape={pixels.shape}")


def main() -> None:
    args = parse_args()
    envs = args.envs or DEFAULT_ENVS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for env_id in envs:
        print_step(f"collecting {env_id}")
        df, pixels, metadata = collect_env(args, env_id)
        write_env_outputs(args.output_dir, env_id, df, pixels, metadata, force=args.force)


if __name__ == "__main__":
    main()
