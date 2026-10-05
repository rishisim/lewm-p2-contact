#!/usr/bin/env python3
"""Fresh regime-isolated Cube rollout generation with outcome-blind retention."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from study_common import (
    ATTEMPT_ROOT,
    REGIMES,
    REPO_ROOT,
    STUDY_ROOT,
    append_ledger,
    assert_runtime_contract,
    atomic_json,
    atomic_npz,
    combined_array_sha256,
    load_v5_runner,
    raw_directory,
    raw_manifest_path,
    read_json,
    relative_to_repo,
    role_count,
    sha256_file,
    update_state_fields,
    verify_pre_outcome_seal,
)


INFO_ALIASES: dict[str, tuple[str, ...]] = {
    "pixels": ("pixels",),
    "observation": ("observation",),
    "qpos": ("qpos",),
    "qvel": ("qvel",),
    "control": ("control",),
    "prev_qpos": ("prev_qpos",),
    "prev_qvel": ("prev_qvel",),
    "proprio_effector_pos": (
        "proprio/effector_pos",
        "proprio_effector_pos",
    ),
    "proprio_effector_yaw": (
        "proprio/effector_yaw",
        "proprio_effector_yaw",
    ),
    "proprio_gripper_contact": (
        "proprio/gripper_contact",
        "proprio_gripper_contact",
    ),
    "proprio_gripper_opening": (
        "proprio/gripper_opening",
        "proprio_gripper_opening",
    ),
    "proprio_gripper_vel": (
        "proprio/gripper_vel",
        "proprio_gripper_vel",
    ),
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
CORE_CAPTURED = {
    "pixels",
    "observation",
    "qpos",
    "qvel",
    "privileged_target_block_pos",
    "privileged_target_block_yaw",
    "step_idx",
}


class GenerationPreflightError(RuntimeError):
    """Global failure that consumes no smoke or target identifier."""


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


def capture(infos: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for output_name, aliases in INFO_ALIASES.items():
        for source_name in aliases:
            if source_name in infos:
                result[output_name] = _unwrap(infos[source_name])
                break
    missing = CORE_CAPTURED - set(result)
    if missing:
        raise RuntimeError(
            f"Cube environment is missing core non-contact fields: {sorted(missing)}"
        )
    return result


def stack_records(records: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[str]]:
    if len(records) != 201:
        raise RuntimeError(f"expected 201 states, observed {len(records)}")
    shared = set.intersection(*(set(record) for record in records))
    missing = CORE_CAPTURED - shared
    if missing:
        raise RuntimeError(f"core capture schema changed: {sorted(missing)}")
    union = set.union(*(set(record) for record in records))
    dropped = sorted(union - shared)
    arrays: dict[str, np.ndarray] = {}
    for name in sorted(shared):
        values = [record[name] for record in records]
        try:
            arrays[name] = np.stack([np.asarray(value) for value in values])
        except (TypeError, ValueError):
            # Optional post-hoc fields can never cause replacement or exclusion.
            arrays[name] = np.asarray([str(value) for value in values], dtype="U64")
    return arrays, dropped


def core_validate(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    pixels = arrays["pixels"]
    actions = arrays["action"]
    if pixels.shape != (201, 224, 224, 3) or pixels.dtype != np.uint8:
        raise RuntimeError(f"unexpected pixel tensor: {pixels.shape} {pixels.dtype}")
    if actions.shape != (201, 5):
        raise RuntimeError(f"unexpected action tensor: {actions.shape}")
    if not np.isfinite(actions[:200]).all() or not np.isnan(actions[200]).all():
        raise RuntimeError("action alignment is not 200 finite plus terminal NaN")
    steps = np.asarray(arrays["step_idx"]).reshape(201, -1)[:, 0]
    if not np.array_equal(steps.astype(np.int64), np.arange(201)):
        raise RuntimeError("environment step index is not exactly 0..200")
    if any(np.asarray(value).shape[0] != 201 for value in arrays.values()):
        raise RuntimeError("generated array row-count drift")
    return {
        "rows": 201,
        "pixels_shape_dtype_valid": True,
        "finite_nonterminal_actions": True,
        "terminal_action_nan_sentinel": True,
        "step_index_exact": True,
        "contact_motion_phase_reward_success_inspected": False,
    }


def make_world(regime: str) -> tuple[Any, Any]:
    if regime not in REGIMES:
        raise RuntimeError(f"unknown regime: {regime}")
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
    specification = REGIMES[regime]
    policy = ExpertPolicy(
        policy_type=specification["policy_type"],
        action_noise=specification["action_noise"],
        p_random_action=specification["p_random_action"],
        noise_smoothing=specification["noise_smoothing"],
        min_norm=specification["min_norm"],
        seed=None,
    )
    world.set_policy(policy)
    observed = {
        "policy_type": policy.type,
        "action_noise": float(policy.action_noise),
        "p_random_action": float(policy.p_random_action),
        "noise_smoothing": float(policy.noise_smoothing),
        "min_norm": float(policy.min_norm),
    }
    expected = {
        "policy_type": specification["policy_type"],
        "action_noise": specification["action_noise"],
        "p_random_action": specification["p_random_action"],
        "noise_smoothing": specification["noise_smoothing"],
        "min_norm": specification["min_norm"],
    }
    if observed != expected:
        world.close()
        raise RuntimeError(f"constructed policy drift: {observed} != {expected}")
    return world, policy


def seed_action_spaces(world: Any, seed: int) -> list[str]:
    seeded = []
    candidates = [
        ("vector_action_space", getattr(world.envs, "action_space", None)),
        (
            "unwrapped_action_space",
            getattr(world.envs.envs[0].unwrapped, "action_space", None),
        ),
    ]
    seen: set[int] = set()
    for name, space in candidates:
        if space is None or id(space) in seen:
            continue
        seen.add(id(space))
        if not callable(getattr(space, "seed", None)):
            raise RuntimeError(f"{name} has no deterministic seed method")
        space.seed(int(seed))
        seeded.append(name)
    if not seeded:
        raise RuntimeError("no action space RNG was seeded")
    return seeded


def generate_episode(
    world: Any,
    policy: Any,
    *,
    regime: str,
    phase: str,
    slot: int,
    episode_id: str,
    env_seed: int,
    policy_seed: int,
    oracle_np_seed: int,
    action_space_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    v5_runner = load_v5_runner()
    _, _, seedfix = v5_runner.distribution_modules()
    policy.set_seed(int(policy_seed))
    reset_metadata = seedfix.deterministic_environment_reset(
        world, world.reset, int(env_seed)
    )
    action_spaces = seed_action_spaces(world, int(action_space_seed))
    np.random.seed(int(oracle_np_seed))
    first = capture(world.infos)
    states = [first]
    actions: list[np.ndarray] = []
    rewards = [0.0]
    terminated = [False]
    truncated = [False]
    for _ in range(200):
        action = np.asarray(policy.get_action(world.infos), dtype=np.float32)
        if action.shape != (1, 5):
            raise RuntimeError(f"unexpected expert action shape: {action.shape}")
        actions.append(action[0].copy())
        _, reward, term, trunc, infos = world.envs.step(action)
        world.infos = infos
        world.rewards = reward
        world.terminateds = term
        world.truncateds = trunc
        states.append(capture(infos))
        # Values are recorded but never inspected or used for retention.
        rewards.append(float(reward[0]))
        terminated.append(bool(term[0]))
        truncated.append(bool(trunc[0]))
    arrays, dropped_optional = stack_records(states)
    arrays["action"] = np.vstack(
        (
            np.asarray(actions, dtype=np.float32),
            np.full((1, 5), np.nan, dtype=np.float32),
        )
    )
    arrays["reward"] = np.asarray(rewards, dtype=np.float32)
    arrays["terminated"] = np.asarray(terminated, dtype=np.bool_)
    arrays["truncated"] = np.asarray(truncated, dtype=np.bool_)
    arrays["trajectory_id"] = np.full(201, episode_id, dtype="U32")
    arrays["phase"] = np.full(201, phase, dtype="U8")
    arrays["regime"] = np.full(201, regime, dtype="U32")
    arrays["policy_type"] = np.full(
        201, REGIMES[regime]["policy_type"], dtype="U16"
    )
    arrays["slot"] = np.full(201, slot, dtype=np.int32)
    arrays["env_seed"] = np.full(201, env_seed, dtype=np.int64)
    arrays["policy_seed"] = np.full(201, policy_seed, dtype=np.int64)
    arrays["oracle_np_seed"] = np.full(201, oracle_np_seed, dtype=np.int64)
    arrays["action_space_seed"] = np.full(
        201, action_space_seed, dtype=np.int64
    )
    validation = core_validate(arrays)
    initial_digest = combined_array_sha256(
        [
            np.asarray(first["pixels"]),
            np.asarray(first["observation"]),
            np.asarray(first["qpos"]),
            np.asarray(first["qvel"]),
            np.asarray(first["privileged_target_block_pos"]),
            np.asarray(first["privileged_target_block_yaw"]),
        ]
    )
    audit = {
        **validation,
        "initial_state_sha256": initial_digest,
        "reset_metadata": reset_metadata,
        "action_space_seed": int(action_space_seed),
        "action_spaces_seeded": action_spaces,
        "rng_activation_order": (
            "policy.set_seed; deterministic V5 seed-forwarded reset; "
            "action-space seed; numpy oracle seed; first policy action"
        ),
        "optional_schema_fields_dropped_without_replacement": dropped_optional,
        "full_environment_fields_recorded_for_post_terminal_interpretation": True,
        "posthoc_fields_never_validated_for_retention": True,
        "outcome_conditioned_retention": False,
    }
    return arrays, audit


def preflight(phase: str, regime: str) -> tuple[Any, Any, dict[str, Any]]:
    world = None
    try:
        runtime = assert_runtime_contract("generation")
        seal = verify_pre_outcome_seal()
        if phase == "target":
            smoke = read_json(ATTEMPT_ROOT / "audit/excluded_regime_smoke.json")
            if not smoke.get("passed"):
                raise RuntimeError("target generation requires passing excluded smoke")
        world, policy = make_world(regime)
        return world, policy, {"runtime": runtime, "seal": seal}
    except BaseException as error:
        if world is not None:
            world.close()
        raise GenerationPreflightError(
            f"global {phase}/{regime} preflight failed before identifier use: {error}"
        ) from error


def record_preflight_failure(
    phase: str, regime: str, error: BaseException
) -> None:
    path = ATTEMPT_ROOT / f"audit/{phase}_{regime}_generation_preflight_failure.json"
    if path.exists():
        return
    atomic_json(
        path,
        {
            "schema_version": 1,
            "created_unix_ns": time.time_ns(),
            "phase": phase,
            "regime": regime,
            "classification": "package_level_preflight",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "identifiers_consumed": 0,
            "smoke_or_target_episodes_generated": 0,
            "target_loss_contact_motion_phase_reward_success_inspected": False,
        },
        exclusive=True,
    )


def validate_existing_manifest(
    phase: str, regime: str, expected: int
) -> dict[str, Any] | None:
    path = raw_manifest_path(phase, regime)
    if not path.exists():
        return None
    manifest = read_json(path)
    if (
        not manifest.get("complete")
        or manifest.get("episode_count") != expected
        or manifest.get("regime") != regime
        or manifest.get("phase") != phase
    ):
        raise RuntimeError(f"incomplete existing raw manifest: {path}")
    for record in manifest["episodes"]:
        candidate = REPO_ROOT / record["path"]
        if sha256_file(candidate) != record["sha256"]:
            raise RuntimeError(f"raw hash drift: {candidate}")
    return manifest


def all_generated_target_count() -> int:
    total = 0
    for regime in REGIMES:
        directory = raw_directory("target", regime)
        if directory.exists():
            total += len(list(directory.glob("*.json")))
    return total


def generate(phase: str, regime: str) -> dict[str, Any]:
    if phase not in ("smoke", "target") or regime not in REGIMES:
        raise RuntimeError("invalid generation role")
    world, policy, preflight_audit = preflight(phase, regime)
    expected = role_count(phase)
    ledger = read_json(ATTEMPT_ROOT / "cohort_seed_ledger.json")
    primary = ledger["roles"][regime][phase]
    replacements = ledger["roles"][regime]["replacement"]
    if len(primary) != expected:
        world.close()
        raise RuntimeError("sealed cohort count mismatch")
    existing_manifest = validate_existing_manifest(phase, regime, expected)
    if existing_manifest is not None:
        world.close()
        return existing_manifest

    directory = raw_directory(phase, regime)
    directory.mkdir(parents=True, exist_ok=True)
    failure_path = ATTEMPT_ROOT / f"audit/{phase}_{regime}_generation_failures.json"
    failures = (
        read_json(failure_path).get("failures", [])
        if failure_path.exists()
        else []
    )
    failed_tuples = {
        (
            int(item["env_seed"]),
            int(item["policy_seed"]),
            int(item["oracle_np_seed"]),
            int(item["action_space_seed"]),
        )
        for item in failures
    }
    used_replacements: set[tuple[int, int, int, int]] = set()
    for sidecar in directory.glob("*.json"):
        item = read_json(sidecar)
        if item.get("replacement_used"):
            used_replacements.add(
                (
                    int(item["env_seed"]),
                    int(item["policy_seed"]),
                    int(item["oracle_np_seed"]),
                    int(item["action_space_seed"]),
                )
            )
    records = []
    try:
        for index, specification in enumerate(primary):
            raw_path = directory / f"{specification['episode_id']}.npz"
            sidecar_path = directory / f"{specification['episode_id']}.json"
            if raw_path.exists() or sidecar_path.exists():
                if not (raw_path.exists() and sidecar_path.exists()):
                    raise RuntimeError(f"partial raw artifact: {raw_path}")
                record = read_json(sidecar_path)
                if sha256_file(raw_path) != record["sha256"]:
                    raise RuntimeError(f"raw sidecar drift: {raw_path}")
                records.append(record)
                continue

            candidates = [(dict(specification), False)] + [
                (dict(item), True) for item in replacements
            ]
            for candidate, replacement_used in candidates:
                seed_tuple = (
                    int(candidate["env_seed"]),
                    int(candidate["policy_seed"]),
                    int(candidate["oracle_np_seed"]),
                    int(candidate["action_space_seed"]),
                )
                if seed_tuple in failed_tuples or (
                    replacement_used and seed_tuple in used_replacements
                ):
                    continue
                if world is None:
                    world, policy = make_world(regime)
                try:
                    arrays, generation_audit = generate_episode(
                        world,
                        policy,
                        regime=regime,
                        phase=phase,
                        slot=int(specification["slot"]),
                        episode_id=str(specification["episode_id"]),
                        env_seed=seed_tuple[0],
                        policy_seed=seed_tuple[1],
                        oracle_np_seed=seed_tuple[2],
                        action_space_seed=seed_tuple[3],
                    )
                    atomic_npz(raw_path, arrays)
                    record = {
                        "schema_version": 1,
                        "created_unix_ns": time.time_ns(),
                        "phase": phase,
                        "regime": regime,
                        "slot": int(specification["slot"]),
                        "episode_id": str(specification["episode_id"]),
                        "seed_source_episode_id": str(candidate["episode_id"]),
                        "replacement_used": replacement_used,
                        "env_seed": seed_tuple[0],
                        "policy_seed": seed_tuple[1],
                        "oracle_np_seed": seed_tuple[2],
                        "action_space_seed": seed_tuple[3],
                        "path": relative_to_repo(raw_path),
                        "sha256": sha256_file(raw_path),
                        "bytes": raw_path.stat().st_size,
                        "initial_state_sha256": generation_audit[
                            "initial_state_sha256"
                        ],
                        "generation_audit": generation_audit,
                        "model_gate_arrays_loaded_during_generation": [],
                        "target_loss_contact_motion_phase_reward_success_inspected": False,
                        "outcome_conditioned_retention": False,
                    }
                    atomic_json(sidecar_path, record, exclusive=True)
                    records.append(record)
                    if replacement_used:
                        used_replacements.add(seed_tuple)
                    if phase == "target" and (
                        (index + 1) % 50 == 0 or index + 1 == expected
                    ):
                        update_state_fields(
                            target_outcome_episodes_generated=all_generated_target_count()
                        )
                    print(
                        f"generated {phase} {regime} {index + 1}/{expected}",
                        flush=True,
                    )
                    break
                except BaseException as error:
                    failures.append(
                        {
                            "created_unix_ns": time.time_ns(),
                            "phase": phase,
                            "regime": regime,
                            "slot": int(specification["slot"]),
                            "episode_id": str(specification["episode_id"]),
                            "attempted_seed_source_episode_id": str(
                                candidate["episode_id"]
                            ),
                            "env_seed": seed_tuple[0],
                            "policy_seed": seed_tuple[1],
                            "oracle_np_seed": seed_tuple[2],
                            "action_space_seed": seed_tuple[3],
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                            "traceback": traceback.format_exc(),
                            "target_loss_contact_motion_phase_reward_success_inspected": False,
                            "outcome_conditioned_replacement": False,
                        }
                    )
                    failed_tuples.add(seed_tuple)
                    atomic_json(
                        failure_path,
                        {
                            "schema_version": 1,
                            "replacement_rule": (
                                "mechanical exception only; outcome fields never inspected"
                            ),
                            "failures": failures,
                        },
                    )
                    if world is not None:
                        world.close()
                    world = policy = None
            else:
                raise RuntimeError(
                    f"replacement pool exhausted for {phase}/{regime}"
                )
    finally:
        if world is not None:
            world.close()

    if len(records) != expected or len(
        {item["episode_id"] for item in records}
    ) != expected:
        raise RuntimeError("generated cohort is not exact and unique")
    manifest = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "phase": phase,
        "regime": regime,
        "regime_specification": REGIMES[regime],
        "complete": True,
        "episode_count": expected,
        "episodes": records,
        "replacement_count": sum(
            bool(item["replacement_used"]) for item in records
        ),
        "input_allowlist_for_model_gate": ["action", "pixels"],
        "smoke_permanently_excluded": phase == "smoke",
        "target_outcomes_inspected_during_generation": False,
        "contact_motion_phase_reward_success_used_for_retention": False,
        "v3_test_targets_opened": False,
        "combined_v3_cache_numpy_loaded": False,
        "released_hdf5_opened": False,
        "generation_runtime": preflight_audit["runtime"],
        "pre_outcome_seal_sha256": preflight_audit["seal"]["seal_sha256"],
        "global_preflight_before_identifier_use": True,
    }
    path = raw_manifest_path(phase, regime)
    atomic_json(path, manifest, exclusive=True)
    append_ledger(
        "raw_cohort_complete",
        phase=phase,
        regime=regime,
        episodes=expected,
        replacements=manifest["replacement_count"],
        manifest_path=relative_to_repo(path),
        manifest_sha256=sha256_file(path),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "target"))
    parser.add_argument("regime", choices=tuple(REGIMES))
    arguments = parser.parse_args()
    try:
        result = generate(arguments.phase, arguments.regime)
    except GenerationPreflightError as error:
        record_preflight_failure(arguments.phase, arguments.regime, error)
        raise
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
