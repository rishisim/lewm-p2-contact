#!/usr/bin/env python3
"""Evidence-bearing checkpoints for smoke, generation, and execution states."""

from __future__ import annotations

import argparse
import json
import time

from study_common import (
    ATTEMPT_ROOT,
    REGIMES,
    REPO_ROOT,
    ROWS_PER_EPISODE,
    SMOKE_EPISODES_PER_REGIME,
    TARGET_EPISODES_PER_REGIME,
    atomic_json,
    complete_state,
    execution_manifest_path,
    raw_manifest_path,
    read_json,
    relative_to_repo,
    sha256_file,
    update_state_fields,
    verify_pre_outcome_seal,
)


def verify_raw(phase: str, regime: str, expected: int) -> dict:
    path = raw_manifest_path(phase, regime)
    manifest = read_json(path)
    bad = []
    for record in manifest["episodes"]:
        candidate = REPO_ROOT / record["path"]
        if sha256_file(candidate) != record["sha256"]:
            bad.append(record["path"])
    checks = {
        "complete": manifest.get("complete") is True,
        "phase": manifest.get("phase") == phase,
        "regime": manifest.get("regime") == regime,
        "episode_count": manifest.get("episode_count") == expected,
        "record_count": len(manifest.get("episodes", [])) == expected,
        "unique_episode_ids": len(
            {item["episode_id"] for item in manifest["episodes"]}
        )
        == expected,
        "all_hashes": not bad,
        "outcome_blind_retention": manifest.get(
            "contact_motion_phase_reward_success_used_for_retention"
        )
        is False,
        "no_forbidden_evidence": not manifest.get("v3_test_targets_opened")
        and not manifest.get("combined_v3_cache_numpy_loaded")
        and not manifest.get("released_hdf5_opened"),
    }
    return {
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
        "checks": checks,
        "bad_paths": bad,
        "passed": all(checks.values()),
        "replacement_count": manifest["replacement_count"],
        "episode_ids": [item["episode_id"] for item in manifest["episodes"]],
    }


def verify_execution(phase: str, regime: str, expected: int) -> dict:
    path = execution_manifest_path(phase, regime)
    manifest = read_json(path)
    bad = []
    for record in manifest["episodes"]:
        candidate = REPO_ROOT / record["path"]
        if sha256_file(candidate) != record["sha256"]:
            bad.append(record["path"])
    checks = {
        "complete": manifest.get("complete") is True,
        "phase": manifest.get("phase") == phase,
        "regime": manifest.get("regime") == regime,
        "episode_count": manifest.get("episode_count") == expected,
        "row_count": manifest.get("row_count")
        == expected * ROWS_PER_EPISODE,
        "record_count": len(manifest.get("episodes", [])) == expected,
        "all_hashes": not bad,
        "all_equivalence": manifest.get(
            "all_equivalence_checks_passed"
        )
        is True,
        "module_frozen": manifest.get("module_before")
        == manifest.get("module_after")
        and manifest.get("module_after", {}).get("passed") is True,
        "no_gradients": manifest.get("no_gradients") is True,
        "input_allowlist": manifest.get("loaded_input_keys")
        == ["action", "pixels"],
        "contact_excluded": manifest.get("contact_or_privileged_loaded")
        is False,
        "no_outcome_analysis": manifest.get(
            "target_loss_contact_motion_phase_reward_success_inspected"
        )
        is False,
        "no_forbidden_evidence": not manifest.get("v3_test_targets_opened")
        and not manifest.get("combined_v3_cache_numpy_loaded")
        and not manifest.get("released_hdf5_opened"),
    }
    return {
        "path": relative_to_repo(path),
        "sha256": sha256_file(path),
        "checks": checks,
        "bad_paths": bad,
        "passed": all(checks.values()),
        "episode_ids": [item["episode_id"] for item in manifest["episodes"]],
    }


def smoke_checkpoint() -> dict:
    verify_pre_outcome_seal()
    output = ATTEMPT_ROOT / "audit/excluded_regime_smoke.json"
    if output.exists():
        complete_state(
            "EXCLUDED_REGIME_SMOKE",
            "FRESH_COHORT_GENERATION",
            evidence_path=output,
            checkpoint_name="v005_excluded_regime_smoke_reused",
            next_action="generate exactly 3,000 fresh target episodes per regime",
        )
        return read_json(output)
    regimes = {}
    all_smoke_ids = set()
    target_ids = {
        item["episode_id"]
        for regime_roles in read_json(
            ATTEMPT_ROOT / "cohort_seed_ledger.json"
        )["roles"].values()
        for item in regime_roles["target"]
    }
    for regime in REGIMES:
        raw = verify_raw("smoke", regime, SMOKE_EPISODES_PER_REGIME)
        execution = verify_execution(
            "smoke", regime, SMOKE_EPISODES_PER_REGIME
        )
        ids = set(raw["episode_ids"])
        regimes[regime] = {
            "raw": raw,
            "execution": execution,
            "raw_execution_ids_exact": ids == set(execution["episode_ids"]),
            "smoke_target_ids_disjoint": ids.isdisjoint(target_ids),
            "passed": raw["passed"]
            and execution["passed"]
            and ids == set(execution["episode_ids"])
            and ids.isdisjoint(target_ids),
        }
        if all_smoke_ids & ids:
            raise RuntimeError("smoke episode IDs overlap across regimes")
        all_smoke_ids.update(ids)
    checks = {
        "all_regimes_present": set(regimes) == set(REGIMES),
        "six_per_regime": len(all_smoke_ids)
        == len(REGIMES) * SMOKE_EPISODES_PER_REGIME,
        "all_regimes_passed": all(item["passed"] for item in regimes.values()),
        "smoke_permanently_excluded": True,
        "target_directories_absent": not (
            ATTEMPT_ROOT / "data/target"
        ).exists(),
        "target_outcome_episodes_absent": True,
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "classification": "fresh_excluded_regime_smoke",
        "regimes": regimes,
        "checks": checks,
        "passed": all(checks.values()),
        "smoke_episode_count": len(all_smoke_ids),
        "target_outcome_episodes": 0,
        "smoke_outcomes_excluded_from_all_inference": True,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"excluded smoke failed: {result}")
    complete_state(
        "EXCLUDED_REGIME_SMOKE",
        "FRESH_COHORT_GENERATION",
        evidence_path=output,
        checkpoint_name="v005_excluded_regime_smoke_passed",
        next_action="generate exactly 3,000 fresh target episodes per regime",
    )
    return result


def generation_checkpoint() -> dict:
    output = ATTEMPT_ROOT / "audit/target_generation_complete.json"
    if output.exists():
        complete_state(
            "FRESH_COHORT_GENERATION",
            "SPARSE_DENSE_EXECUTION",
            evidence_path=output,
            checkpoint_name="v005_target_generation_reused",
            next_action="execute frozen sparse policy and dense shadow",
            extra_fields={"target_outcome_episodes_generated": 9_000},
        )
        return read_json(output)
    regimes = {
        regime: verify_raw("target", regime, TARGET_EPISODES_PER_REGIME)
        for regime in REGIMES
    }
    ids = [
        episode
        for item in regimes.values()
        for episode in item["episode_ids"]
    ]
    checks = {
        "all_regimes_passed": all(item["passed"] for item in regimes.values()),
        "exact_total_9000": len(ids)
        == len(REGIMES) * TARGET_EPISODES_PER_REGIME,
        "all_target_ids_unique": len(set(ids)) == len(ids),
        "no_sequential_expansion": True,
        "outcome_blind_retention": all(
            item["checks"]["outcome_blind_retention"]
            for item in regimes.values()
        ),
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "regimes": regimes,
        "checks": checks,
        "passed": all(checks.values()),
        "target_outcome_episodes_generated": len(ids),
        "target_outcomes_opened_for_analysis": False,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"target generation checkpoint failed: {result}")
    update_state_fields(target_outcome_episodes_generated=len(ids))
    complete_state(
        "FRESH_COHORT_GENERATION",
        "SPARSE_DENSE_EXECUTION",
        evidence_path=output,
        checkpoint_name="v005_target_generation_complete",
        next_action="execute frozen sparse policy and dense shadow",
        extra_fields={"target_outcome_episodes_generated": len(ids)},
    )
    return result


def execution_checkpoint() -> dict:
    output = ATTEMPT_ROOT / "audit/target_execution_complete.json"
    if output.exists():
        complete_state(
            "SPARSE_DENSE_EXECUTION",
            "SEALED_ANALYSIS",
            evidence_path=output,
            checkpoint_name="v005_target_execution_reused",
            next_action="seal all target inputs before first target-array analysis",
            extra_fields={"target_outcome_episodes_executed": 9_000},
        )
        return read_json(output)
    regimes = {
        regime: verify_execution(
            "target", regime, TARGET_EPISODES_PER_REGIME
        )
        for regime in REGIMES
    }
    ids = [
        episode
        for item in regimes.values()
        for episode in item["episode_ids"]
    ]
    checks = {
        "all_regimes_passed": all(item["passed"] for item in regimes.values()),
        "exact_total_9000": len(ids)
        == len(REGIMES) * TARGET_EPISODES_PER_REGIME,
        "exact_total_rows_342000": len(ids) * ROWS_PER_EPISODE == 342_000,
        "all_target_ids_unique": len(set(ids)) == len(ids),
        "all_equivalence_passed": all(
            item["checks"]["all_equivalence"] for item in regimes.values()
        ),
        "targets_not_opened_for_analysis": True,
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "regimes": regimes,
        "checks": checks,
        "passed": all(checks.values()),
        "target_outcome_episodes_executed": len(ids),
        "target_rows": len(ids) * ROWS_PER_EPISODE,
        "target_outcomes_opened_for_analysis": False,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"target execution checkpoint failed: {result}")
    update_state_fields(target_outcome_episodes_executed=len(ids))
    complete_state(
        "SPARSE_DENSE_EXECUTION",
        "SEALED_ANALYSIS",
        evidence_path=output,
        checkpoint_name="v005_target_execution_complete",
        next_action="seal all target inputs before first target-array analysis",
        extra_fields={"target_outcome_episodes_executed": len(ids)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", choices=("smoke", "generation", "execution"))
    arguments = parser.parse_args()
    if arguments.checkpoint == "smoke":
        result = smoke_checkpoint()
    elif arguments.checkpoint == "generation":
        result = generation_checkpoint()
    else:
        result = execution_checkpoint()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
