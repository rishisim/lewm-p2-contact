#!/usr/bin/env python3
"""Generate role-isolated, seed-frozen OGBench Cube episodes.

Run this file only with the preregistered generation interpreter.  Episode
outcomes and success never affect retention.  Only an exception or a malformed
environment trajectory invokes the frozen next-unused replacement-seed rule.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

import common


INFO_ALIASES: dict[str, tuple[str, ...]] = {
    "pixels": ("pixels",),
    "observation": ("observation",),
    "qpos": ("qpos",),
    "qvel": ("qvel",),
    "control": ("control",),
    "prev_qpos": ("prev_qpos",),
    "prev_qvel": ("prev_qvel",),
    "proprio_effector_pos": ("proprio/effector_pos", "proprio_effector_pos"),
    "proprio_effector_yaw": ("proprio/effector_yaw", "proprio_effector_yaw"),
    "proprio_gripper_contact": (
        "proprio/gripper_contact",
        "proprio_gripper_contact",
    ),
    "proprio_gripper_opening": (
        "proprio/gripper_opening",
        "proprio_gripper_opening",
    ),
    "proprio_gripper_vel": ("proprio/gripper_vel", "proprio_gripper_vel"),
    "proprio_joint_pos": ("proprio/joint_pos", "proprio_joint_pos"),
    "proprio_joint_vel": ("proprio/joint_vel", "proprio_joint_vel"),
    "privileged_block_0_pos": (
        "privileged/block_0_pos",
        "privileged_block_0_pos",
    ),
    "privileged_block_0_quat": (
        "privileged/block_0_quat",
        "privileged_block_0_quat",
    ),
    "privileged_block_0_yaw": (
        "privileged/block_0_yaw",
        "privileged_block_0_yaw",
    ),
    "privileged_target_block": (
        "privileged/target_block",
        "privileged_target_block",
    ),
    "privileged_target_block_pos": (
        "privileged/target_block_pos",
        "privileged_target_block_pos",
    ),
    "privileged_target_block_yaw": (
        "privileged/target_block_yaw",
        "privileged_target_block_yaw",
    ),
    "privileged_target_task": (
        "privileged/target_task",
        "privileged_target_task",
    ),
    "success": ("success",),
    "step_idx": ("step_idx",),
    "time": ("time",),
}
REQUIRED_CAPTURED = {
    "pixels",
    "observation",
    "qpos",
    "qvel",
    "proprio_effector_pos",
    "proprio_gripper_contact",
    "privileged_block_0_pos",
    "privileged_block_0_quat",
    "step_idx",
}


def _unwrap(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        result = value
        if result.shape[:2] == (1, 1):
            result = result[0, 0]
        elif result.shape[:1] == (1,):
            result = result[0]
        return np.asarray(result).copy()
    if isinstance(value, list) and len(value) == 1:
        result = value[0]
        if isinstance(result, list) and len(result) == 1:
            result = result[0]
        return result
    return value


def _capture(infos: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for output_name, aliases in INFO_ALIASES.items():
        for source_name in aliases:
            if source_name in infos:
                result[output_name] = _unwrap(infos[source_name])
                break
    missing = REQUIRED_CAPTURED - set(result)
    if missing:
        raise RuntimeError(
            f"Cube environment info is missing required fields {sorted(missing)}; "
            f"available={sorted(infos)}"
        )
    return result


def _stack_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if len(records) != common.RAW_EPISODE_ROWS:
        raise RuntimeError(f"expected {common.RAW_EPISODE_ROWS} states, got {len(records)}")
    shared = set(records[0])
    if any(set(record) != shared for record in records):
        raise RuntimeError("environment info schema changed during an episode")
    result: dict[str, np.ndarray] = {}
    for name in sorted(shared):
        values = [record[name] for record in records]
        try:
            result[name] = np.stack([np.asarray(value) for value in values])
        except (TypeError, ValueError):
            result[name] = np.asarray([str(value) for value in values], dtype="U64")
    return result


def _validate_episode(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if arrays["pixels"].shape != (common.RAW_EPISODE_ROWS, 224, 224, 3):
        raise RuntimeError(f"unexpected pixels shape {arrays['pixels'].shape}")
    if arrays["pixels"].dtype != np.uint8:
        raise RuntimeError(f"unexpected pixels dtype {arrays['pixels'].dtype}")
    if arrays["action"].shape != (common.RAW_EPISODE_ROWS, common.RAW_ACTION_DIM):
        raise RuntimeError(f"unexpected action shape {arrays['action'].shape}")
    if not np.isfinite(arrays["action"][:-1]).all() or not np.isnan(arrays["action"][-1]).all():
        raise RuntimeError("action alignment requires 200 finite actions and one terminal NaN row")
    for name, value in arrays.items():
        if value.shape[0] != common.RAW_EPISODE_ROWS:
            raise RuntimeError(f"{name} has wrong row count {value.shape}")
        if value.dtype.kind in "fc" and name != "action" and not np.isfinite(value).all():
            raise RuntimeError(f"nonfinite generated array {name}")
    steps = np.asarray(arrays["step_idx"]).reshape(common.RAW_EPISODE_ROWS, -1)[:, 0]
    if not np.array_equal(steps.astype(np.int64), np.arange(common.RAW_EPISODE_ROWS)):
        raise RuntimeError(f"environment step_idx is not 0..200: {steps[:5]} ... {steps[-5:]}")
    return {
        "rows": common.RAW_EPISODE_ROWS,
        "finite_nonterminal_actions": True,
        "terminal_action_nan_sentinel": True,
        "step_index_exact": True,
        "success_final": bool(np.asarray(arrays["success"][-1]).reshape(-1)[0])
        if "success" in arrays
        else None,
    }


def _make_world() -> tuple[Any, Any]:
    import stable_worldmodel as swm
    from stable_worldmodel.envs.ogbench import ExpertPolicy

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=200,
        image_shape=(224, 224),
        env_type="single",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
    )
    policy = ExpertPolicy(
        policy_type="markov_oracle",
        action_noise=0.1,
        p_random_action=0.0,
        noise_smoothing=0.5,
        min_norm=0.4,
        seed=None,
    )
    world.set_policy(policy)
    return world, policy


def generate_episode(
    world: Any,
    policy: Any,
    *,
    env_seed: int,
    policy_seed: int,
    trajectory_id: str,
    role: str,
    slot: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    policy.set_seed(int(policy_seed))
    world.reset(seed=int(env_seed), options=None)
    states = [_capture(world.infos)]
    actions = []
    rewards = [0.0]
    terminated = [False]
    truncated = [False]
    for _ in range(200):
        action = np.asarray(policy.get_action(world.infos), dtype=np.float32)
        if action.shape != (1, common.RAW_ACTION_DIM):
            raise RuntimeError(f"unexpected expert action shape {action.shape}")
        actions.append(action[0].copy())
        _, reward, term, trunc, infos = world.envs.step(action)
        world.infos = infos
        world.rewards = reward
        world.terminateds = term
        world.truncateds = trunc
        states.append(_capture(infos))
        rewards.append(float(reward[0]))
        terminated.append(bool(term[0]))
        truncated.append(bool(trunc[0]))
    arrays = _stack_records(states)
    arrays["action"] = np.vstack(
        (np.asarray(actions, dtype=np.float32), np.full((1, common.RAW_ACTION_DIM), np.nan, dtype=np.float32))
    )
    arrays["reward"] = np.asarray(rewards, dtype=np.float32)
    arrays["terminated"] = np.asarray(terminated, dtype=np.bool_)
    arrays["truncated"] = np.asarray(truncated, dtype=np.bool_)
    arrays["trajectory_id"] = np.full(common.RAW_EPISODE_ROWS, trajectory_id, dtype="U32")
    arrays["role"] = np.full(common.RAW_EPISODE_ROWS, role, dtype="U16")
    arrays["slot"] = np.full(common.RAW_EPISODE_ROWS, int(slot), dtype=np.int32)
    arrays["env_seed"] = np.full(common.RAW_EPISODE_ROWS, int(env_seed), dtype=np.int64)
    arrays["policy_seed"] = np.full(common.RAW_EPISODE_ROWS, int(policy_seed), dtype=np.int64)
    audit = _validate_episode(arrays)
    return arrays, audit


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _episode_record(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    role: str,
    slot: int,
    trajectory_id: str,
    env_seed: int,
    policy_seed: int,
    primary_seed_used: bool,
    generation_audit: Mapping[str, Any],
) -> dict[str, Any]:
    hashes = {name: common.array_sha256(value) for name, value in arrays.items()}
    return {
        "role": role,
        "slot": int(slot),
        "trajectory_id": trajectory_id,
        "env_seed": int(env_seed),
        "policy_seed": int(policy_seed),
        "primary_seed_used": bool(primary_seed_used),
        "path": str(path.relative_to(common.REPO)),
        "file_sha256": common.sha256_file(path),
        "file_bytes": int(path.stat().st_size),
        "observation_sha256": hashes["observation"],
        "pixels_sha256": hashes["pixels"],
        "actions_sha256": hashes["action"],
        "trajectory_identifier_sha256": hashes["trajectory_id"],
        "combined_observation_action_sha256": common.combined_array_sha256(
            [arrays["pixels"], arrays["observation"], arrays["action"]]
        ),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashes[name]}
            for name, value in arrays.items()
        },
        "generation_audit": dict(generation_audit),
    }


def _versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "stable-worldmodel",
        "ogbench",
        "mujoco",
        "gymnasium",
        "numpy",
        "torch",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_existing_record(sidecar: Path, raw_path: Path) -> dict[str, Any] | None:
    if not sidecar.exists() and not raw_path.exists():
        return None
    if not (sidecar.exists() and raw_path.exists()):
        raise RuntimeError(f"partial episode artifact: {raw_path}")
    record = common.read_json(sidecar)
    if common.sha256_file(raw_path) != record["file_sha256"]:
        raise RuntimeError(f"existing generated episode hash drift: {raw_path}")
    return record


def run(role: str) -> dict[str, Any]:
    if role == "smoke":
        common.verify_seal(common.ROOT / "audit/phase0_seal.json")
    else:
        common.verify_seal(common.ROOT / "audit/pre_confirmation_seal.json")
        smoke_manifest = common.ROOT / "data/smoke_data_manifest.json"
        if not smoke_manifest.exists() or not common.read_json(smoke_manifest)["complete"]:
            raise RuntimeError("confirmation generation requires the completed 12-episode smoke")
    seed_manifest = common.read_json(common.ROOT / "seed_manifest.json")
    slots = seed_manifest["roles"][role]["episodes"]
    replacements = iter(seed_manifest["roles"][role]["replacement_pool"])
    raw_dir, aggregate_path = common.role_paths(role)
    if aggregate_path.exists():
        aggregate = common.read_json(aggregate_path)
        if not aggregate.get("complete") or len(aggregate["episodes"]) != len(slots):
            raise RuntimeError("existing aggregate generation manifest is incomplete")
        for record in aggregate["episodes"]:
            path = common.REPO / record["path"]
            if common.sha256_file(path) != record["file_sha256"]:
                raise RuntimeError(f"generated data drift: {path}")
        return aggregate

    failure_path = common.ROOT / "audit" / f"{role}_generation_failures.json"
    failures = common.read_json(failure_path) if failure_path.exists() else []
    used_replacements = {int(item["replacement_env_seed"]) for item in failures}
    replacement_list = [item for item in seed_manifest["roles"][role]["replacement_pool"]]
    replacements = iter(
        [item for item in replacement_list if int(item["env_seed"]) not in used_replacements]
    )
    records = []
    world = policy = None
    try:
        for slot_spec in slots:
            slot = int(slot_spec["slot"])
            trajectory_id = str(slot_spec["trajectory_id"])
            raw_path = raw_dir / f"{trajectory_id}.npz"
            sidecar = raw_dir / f"{trajectory_id}.json"
            existing = _load_existing_record(sidecar, raw_path)
            if existing is not None:
                records.append(existing)
                continue
            candidate = dict(slot_spec["primary"])
            primary = True
            while True:
                try:
                    if world is None:
                        world, policy = _make_world()
                    arrays, episode_audit = generate_episode(
                        world,
                        policy,
                        env_seed=int(candidate["env_seed"]),
                        policy_seed=int(candidate["policy_seed"]),
                        trajectory_id=trajectory_id,
                        role=role,
                        slot=slot,
                    )
                    _write_npz_atomic(raw_path, arrays)
                    record = _episode_record(
                        raw_path,
                        arrays,
                        role=role,
                        slot=slot,
                        trajectory_id=trajectory_id,
                        env_seed=int(candidate["env_seed"]),
                        policy_seed=int(candidate["policy_seed"]),
                        primary_seed_used=primary,
                        generation_audit=episode_audit,
                    )
                    common.write_json(sidecar, record, exclusive=True)
                    common.file_mode_read_only(raw_path)
                    common.file_mode_read_only(sidecar)
                    records.append(record)
                    print(f"generated {role} slot {slot + 1}/{len(slots)}", flush=True)
                    break
                except Exception as exc:
                    if world is not None:
                        try:
                            world.close()
                        except Exception:
                            pass
                    world = policy = None
                    try:
                        replacement = next(replacements)
                    except StopIteration as exhausted:
                        raise RuntimeError("predeclared replacement seed pool exhausted") from exhausted
                    failure = {
                        "role": role,
                        "slot": slot,
                        "trajectory_id": trajectory_id,
                        "failed_env_seed": int(candidate["env_seed"]),
                        "failed_policy_seed": int(candidate["policy_seed"]),
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                        "replacement_env_seed": int(replacement["env_seed"]),
                        "replacement_policy_seed": int(replacement["policy_seed"]),
                        "reason_class": "environment_generation_failure_only",
                    }
                    failures.append(failure)
                    common.write_json(failure_path, failures)
                    candidate = dict(replacement)
                    primary = False
    finally:
        if world is not None:
            world.close()

    records.sort(key=lambda item: int(item["slot"]))
    if len(records) != len(slots) or [int(item["slot"]) for item in records] != list(range(len(slots))):
        raise RuntimeError("role generation did not produce exactly its frozen slots")
    combined = [item["combined_observation_action_sha256"] for item in records]
    if len(set(combined)) != len(combined):
        raise RuntimeError("duplicate episodes occurred within generated role")
    aggregate = {
        "schema_version": 1,
        "role": role,
        "complete": True,
        "episode_count": len(records),
        "expected_episode_count": 12 if role == "smoke" else 300,
        "raw_rows_per_episode": common.RAW_EPISODE_ROWS,
        "environment": common.load_config()["environment_generation"],
        "package_versions": _versions(),
        "seed_manifest_sha256": common.sha256_file(common.ROOT / "seed_manifest.json"),
        "phase0_seal_sha256": common.sha256_file(common.ROOT / "audit/phase0_seal.json"),
        "pre_confirmation_seal_sha256": common.sha256_file(
            common.ROOT / "audit/pre_confirmation_seal.json"
        )
        if role == "confirmation"
        else None,
        "episodes": records,
        "generation_failures": failures,
        "performance_based_exclusions": 0,
        "success_based_exclusions": 0,
        "complete_role_digest": common.sha256_bytes(
            "".join(item["file_sha256"] for item in records).encode("ascii")
        ),
    }
    if aggregate["episode_count"] != aggregate["expected_episode_count"]:
        raise RuntimeError("wrong generated episode count")
    common.write_json(aggregate_path, aggregate, exclusive=True)
    common.file_mode_read_only(aggregate_path)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("smoke", "confirmation"))
    args = parser.parse_args()
    result = run(args.role)
    print(
        json.dumps(
            {
                "role": args.role,
                "complete": result["complete"],
                "episodes": result["episode_count"],
                "generation_failures": len(result["generation_failures"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
