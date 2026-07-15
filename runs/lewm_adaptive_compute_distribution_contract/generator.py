#!/usr/bin/env python3
"""Generate paired PlanOracle/MarkovOracle Cube rollouts from frozen seeds."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import traceback
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

import common


_v4_generator = common._load_module(
    "distribution_contract_v4_generator_helpers", common.V4_ROOT / "generator.py"
)


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("stable-worldmodel", "ogbench", "mujoco", "gymnasium", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def make_world(policy_type: str) -> tuple[Any, Any]:
    import stable_worldmodel as swm
    from stable_worldmodel.envs.ogbench import ExpertPolicy

    if policy_type not in {"plan_oracle", "markov_oracle"}:
        raise ValueError("unsupported policy type")
    cfg = common.study_json(common.STUDY_ROOT / "config.json")
    world_cfg = cfg["environment_constructor"]
    world = swm.World(
        world_cfg["environment"],
        num_envs=world_cfg["num_envs"],
        max_episode_steps=world_cfg["max_episode_steps"],
        image_shape=tuple(world_cfg["image_shape"]),
        env_type=world_cfg["env_type"],
        multiview=world_cfg["multiview"],
        width=world_cfg["width"],
        height=world_cfg["height"],
        visualize_info=world_cfg["visualize_info"],
        terminate_at_goal=world_cfg["terminate_at_goal"],
        mode=world_cfg["mode"],
    )
    policy_cfg = cfg["policy_constructors"][policy_type]
    policy = ExpertPolicy(
        policy_type=policy_type,
        action_noise=policy_cfg["action_noise"],
        p_random_action=policy_cfg["p_random_action"],
        noise_smoothing=policy_cfg["noise_smoothing"],
        min_norm=policy_cfg["min_norm"],
        seed=None,
    )
    world.set_policy(policy)
    return world, policy


def _initial_state_digest(state: Mapping[str, Any]) -> str:
    return common.combined_array_sha256(
        [
            np.asarray(state["pixels"]),
            np.asarray(state["observation"]),
            np.asarray(state["qpos"]),
            np.asarray(state["qvel"]),
            np.asarray(state["privileged_target_block_pos"]),
            np.asarray(state["privileged_target_block_yaw"]),
        ]
    )


def generate_episode(
    world: Any,
    policy: Any,
    *,
    phase: str,
    policy_type: str,
    slot: int,
    trajectory_id: str,
    env_seed: int,
    policy_seed: int,
    oracle_np_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    policy.set_seed(int(policy_seed))
    world.reset(seed=int(env_seed), options=None)
    # OGBench 1.2.1 oracle implementations draw plan keyframes/noise and
    # Markov final poses from np.random.  Seed only after reset so the policy
    # stream cannot perturb paired environment initialization.
    np.random.seed(int(oracle_np_seed))
    first = _v4_generator._capture(world.infos)
    states = [first]
    actions: list[np.ndarray] = []
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
        states.append(_v4_generator._capture(infos))
        rewards.append(float(reward[0]))
        terminated.append(bool(term[0]))
        truncated.append(bool(trunc[0]))
    arrays = _v4_generator._stack_records(states)
    arrays["action"] = np.vstack(
        [
            np.asarray(actions, dtype=np.float32),
            np.full((1, common.RAW_ACTION_DIM), np.nan, dtype=np.float32),
        ]
    )
    arrays["reward"] = np.asarray(rewards, dtype=np.float32)
    arrays["terminated"] = np.asarray(terminated, dtype=np.bool_)
    arrays["truncated"] = np.asarray(truncated, dtype=np.bool_)
    arrays["trajectory_id"] = np.full(common.RAW_EPISODE_ROWS, trajectory_id, dtype="U32")
    arrays["phase"] = np.full(common.RAW_EPISODE_ROWS, phase, dtype="U8")
    arrays["policy_type"] = np.full(common.RAW_EPISODE_ROWS, policy_type, dtype="U16")
    arrays["slot"] = np.full(common.RAW_EPISODE_ROWS, slot, dtype=np.int32)
    arrays["env_seed"] = np.full(common.RAW_EPISODE_ROWS, env_seed, dtype=np.int64)
    arrays["policy_seed"] = np.full(common.RAW_EPISODE_ROWS, policy_seed, dtype=np.int64)
    arrays["oracle_np_seed"] = np.full(
        common.RAW_EPISODE_ROWS, oracle_np_seed, dtype=np.int64
    )
    audit = _v4_generator._validate_episode(arrays)
    audit.update(
        {
            "initial_state_sha256": _initial_state_digest(first),
            "rng_activation_order": "policy.set_seed; world.reset(env_seed); numpy.random.seed(oracle_np_seed); first policy action",
            "numpy_global_oracle_rng_explicitly_seeded": True,
        }
    )
    return arrays, audit


def _record(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    phase: str,
    policy_type: str,
    slot: int,
    trajectory_id: str,
    env_seed: int,
    policy_seed: int,
    oracle_np_seed: int,
    primary_seed_used: bool,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    hashes = {name: common.array_sha256(value) for name, value in arrays.items()}
    return {
        "phase": phase,
        "policy_type": policy_type,
        "slot": slot,
        "trajectory_id": trajectory_id,
        "env_seed": env_seed,
        "policy_seed": policy_seed,
        "oracle_np_seed": oracle_np_seed,
        "primary_seed_used": bool(primary_seed_used),
        "path": str(path.relative_to(common.REPO_ROOT)),
        "file_sha256": common.sha256_file(path),
        "file_bytes": path.stat().st_size,
        "observation_sha256": hashes["observation"],
        "pixels_sha256": hashes["pixels"],
        "actions_sha256": hashes["action"],
        "combined_observation_action_sha256": common.combined_array_sha256(
            [arrays["pixels"], arrays["observation"], arrays["action"]]
        ),
        "initial_state_sha256": audit["initial_state_sha256"],
        "arrays": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "sha256": hashes[name],
            }
            for name, value in arrays.items()
        },
        "generation_audit": dict(audit),
    }


def _load_existing(sidecar: Path, raw_path: Path) -> dict[str, Any] | None:
    if not sidecar.exists() and not raw_path.exists():
        return None
    if not (sidecar.exists() and raw_path.exists()):
        raise RuntimeError(f"partial generated artifact: {raw_path}")
    record = common.study_json(sidecar)
    if common.sha256_file(raw_path) != record["file_sha256"]:
        raise RuntimeError(f"generated raw hash drift: {raw_path}")
    return record


def run(phase: str, policy_type: str) -> dict[str, Any]:
    seal = common.assert_pre_generation_seal()
    seed_manifest = common.study_json(common.STUDY_ROOT / "seed_manifest.json")
    slots = seed_manifest["roles"][phase][policy_type]
    raw_dir, manifest_path = common.role_raw_paths(phase, policy_type)
    if manifest_path.exists():
        existing = common.study_json(manifest_path)
        if not existing.get("complete") or existing.get("episode_count") != len(slots):
            raise RuntimeError("existing generation manifest is incomplete")
        for record in existing["episodes"]:
            if common.sha256_file(common.REPO_ROOT / record["path"]) != record["file_sha256"]:
                raise RuntimeError("existing generated episode drift")
        return existing
    if phase == "main":
        for smoke_policy in ("plan_oracle", "markov_oracle"):
            smoke_manifest = common.role_raw_paths("smoke", smoke_policy)[1]
            if not smoke_manifest.exists() or not common.study_json(smoke_manifest).get("complete"):
                raise RuntimeError("main generation requires both complete smoke roles")
        smoke_audit = common.STUDY_ROOT / "audit/smoke_pairing.json"
        if not smoke_audit.exists() or not common.study_json(smoke_audit).get("passed"):
            raise RuntimeError("main generation requires passing smoke pairing audit")
    raw_dir.mkdir(parents=True, exist_ok=True)
    failure_path = common.STUDY_ROOT / "audit" / f"{phase}_{policy_type}_failures.json"
    failures = common.study_json(failure_path) if failure_path.exists() else []
    used_replacements = {int(item["replacement_env_seed"]) for item in failures}
    replacements = iter(
        [
            item
            for item in seed_manifest["roles"]["replacement"][policy_type]
            if int(item["env_seed"]) not in used_replacements
        ]
    )
    records: list[dict[str, Any]] = []
    world = policy = None
    try:
        for spec in slots:
            slot = int(spec["slot"])
            trajectory_id = str(spec["trajectory_id"])
            raw_path = raw_dir / f"{trajectory_id}.npz"
            sidecar = raw_dir / f"{trajectory_id}.json"
            existing = _load_existing(sidecar, raw_path)
            if existing is not None:
                records.append(existing)
                continue
            candidate = dict(spec)
            primary = True
            while True:
                try:
                    if world is None:
                        world, policy = make_world(policy_type)
                    arrays, audit = generate_episode(
                        world,
                        policy,
                        phase=phase,
                        policy_type=policy_type,
                        slot=slot,
                        trajectory_id=trajectory_id,
                        env_seed=int(candidate["env_seed"]),
                        policy_seed=int(candidate["policy_seed"]),
                        oracle_np_seed=int(candidate["oracle_np_seed"]),
                    )
                    common.atomic_npz(raw_path, arrays)
                    record = _record(
                        raw_path,
                        arrays,
                        phase=phase,
                        policy_type=policy_type,
                        slot=slot,
                        trajectory_id=trajectory_id,
                        env_seed=int(candidate["env_seed"]),
                        policy_seed=int(candidate["policy_seed"]),
                        oracle_np_seed=int(candidate["oracle_np_seed"]),
                        primary_seed_used=primary,
                        audit=audit,
                    )
                    common.write_study_json(sidecar, record, exclusive=True)
                    records.append(record)
                    print(
                        f"generated {phase} {policy_type} episode {slot + 1}/{len(slots)}",
                        flush=True,
                    )
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
                        raise RuntimeError("predeclared replacement pool exhausted") from exhausted
                    failure = {
                        "phase": phase,
                        "policy_type": policy_type,
                        "slot": slot,
                        "trajectory_id": trajectory_id,
                        "failed_env_seed": int(candidate["env_seed"]),
                        "failed_policy_seed": int(candidate["policy_seed"]),
                        "failed_oracle_np_seed": int(candidate["oracle_np_seed"]),
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "traceback": traceback.format_exc(),
                        "replacement_env_seed": int(replacement["env_seed"]),
                        "replacement_policy_seed": int(replacement["policy_seed"]),
                        "replacement_oracle_np_seed": int(replacement["oracle_np_seed"]),
                        "reason_class": "exception_or_mechanical_malformed_trajectory_only",
                    }
                    failures.append(failure)
                    common.write_study_json(failure_path, failures)
                    candidate = dict(replacement)
                    primary = False
    finally:
        if world is not None:
            world.close()
    records.sort(key=lambda item: int(item["slot"]))
    if [int(item["slot"]) for item in records] != list(range(len(slots))):
        raise RuntimeError("generated slots are incomplete or reordered")
    combined = [item["combined_observation_action_sha256"] for item in records]
    if len(set(combined)) != len(combined):
        raise RuntimeError("exact duplicate within generated role")
    manifest = {
        **common.dataset_manifest_base(
            role=f"{phase}_{policy_type}", episodes=len(records), rows=len(records) * 201
        ),
        "complete": True,
        "phase": phase,
        "policy_type": policy_type,
        "smoke_permanently_excluded_from_all_model_and_decision_metrics": phase == "smoke",
        "source_namespace": "fresh_seeded_swm_ogbcube_rollout",
        "package_versions": _versions(),
        "config_sha256": common.sha256_file(common.STUDY_ROOT / "config.json"),
        "seed_manifest_sha256": common.sha256_file(
            common.STUDY_ROOT / "seed_manifest.json"
        ),
        "pre_generation_seal_sha256": common.sha256_file(
            common.STUDY_ROOT / "audit/pre_generation_seal.json"
        ),
        "sealed_file_count": len(seal["sealed_files"]),
        "episodes": records,
        "generation_failures": failures,
        "performance_based_exclusions": 0,
        "success_based_exclusions": 0,
        "complete_role_digest": common.sha256_bytes(
            "".join(record["file_sha256"] for record in records).encode("ascii")
        ),
    }
    common.write_study_json(manifest_path, manifest, exclusive=True)
    return manifest


def audit_pairing(phase: str) -> dict[str, Any]:
    plan_path = common.role_raw_paths(phase, "plan_oracle")[1]
    markov_path = common.role_raw_paths(phase, "markov_oracle")[1]
    plan = common.study_json(plan_path)
    markov = common.study_json(markov_path)
    expected = 12 if phase == "smoke" else 30
    plan_by_slot = {int(item["slot"]): item for item in plan["episodes"]}
    markov_by_slot = {int(item["slot"]): item for item in markov["episodes"]}
    paired_slots = sorted(set(plan_by_slot) & set(markov_by_slot))
    rows = []
    for slot in paired_slots:
        left, right = plan_by_slot[slot], markov_by_slot[slot]
        rows.append(
            {
                "slot": slot,
                "env_seed": int(left["env_seed"]),
                "environment_seed_equal": int(left["env_seed"]) == int(right["env_seed"]),
                "initial_state_sha256_equal": left["initial_state_sha256"]
                == right["initial_state_sha256"],
                "wrapper_policy_seeds_distinct": int(left["policy_seed"])
                != int(right["policy_seed"]),
                "oracle_np_seeds_distinct": int(left["oracle_np_seed"])
                != int(right["oracle_np_seed"]),
                "raw_paths_distinct": left["path"] != right["path"],
                "trajectory_ids_distinct": left["trajectory_id"] != right["trajectory_id"],
            }
        )
    checks = {
        "paired_count_exact": len(rows) == expected,
        "identical_environment_seeds": all(item["environment_seed_equal"] for item in rows),
        "identical_initial_states": all(item["initial_state_sha256_equal"] for item in rows),
        "separate_wrapper_rng": all(item["wrapper_policy_seeds_distinct"] for item in rows),
        "separate_oracle_np_rng": all(item["oracle_np_seeds_distinct"] for item in rows),
        "physically_separate_paths": all(item["raw_paths_distinct"] for item in rows),
        "distinct_trajectory_ids": all(item["trajectory_ids_distinct"] for item in rows),
        "v3_test_targets_opened": False,
    }
    passed = all(value for key, value in checks.items() if key != "v3_test_targets_opened")
    output = {
        "schema_version": 1,
        "phase": phase,
        "passed": passed,
        "checks": checks,
        "pairs": rows,
        "plan_manifest_sha256": common.sha256_file(plan_path),
        "markov_manifest_sha256": common.sha256_file(markov_path),
        "v5_confirmation_episodes": 0,
    }
    path = common.STUDY_ROOT / "audit" / f"{phase}_pairing.json"
    common.write_study_json(path, output, exclusive=not path.exists())
    if not passed:
        raise RuntimeError(f"paired generator audit failed: {checks}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("phase", choices=("smoke", "main"))
    generate.add_argument("policy", choices=("plan_oracle", "markov_oracle"))
    audit = sub.add_parser("audit-pairs")
    audit.add_argument("phase", choices=("smoke", "main"))
    args = parser.parse_args()
    if args.command == "generate":
        result = run(args.phase, args.policy)
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "phase": args.phase,
                    "policy": args.policy,
                    "episodes": result["episode_count"],
                    "failures": len(result["generation_failures"]),
                },
                sort_keys=True,
            )
        )
    else:
        result = audit_pairing(args.phase)
        print(json.dumps({"phase": args.phase, "passed": result["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()

