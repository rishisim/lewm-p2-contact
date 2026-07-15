#!/usr/bin/env python3
"""Independent verification of post-seal compatibility amendments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import common


def _verify_seal(path: Path) -> dict[str, Any]:
    payload = common.study_json(path)
    mismatches = []
    for relative, expected in payload["sealed_files"].items():
        candidate = common.REPO_ROOT / relative
        if common.sha256_file(candidate) != expected:
            mismatches.append(relative)
    return {
        "path": str(path.relative_to(common.REPO_ROOT)),
        "sha256": common.sha256_file(path),
        "sealed_file_count": len(payload["sealed_files"]),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _initial_digest(raw: dict[str, np.ndarray]) -> str:
    return common.combined_array_sha256(
        [
            raw["pixels"][0],
            raw["observation"][0],
            raw["qpos"][0],
            raw["qvel"][0],
            raw["privileged_target_block_pos"][0],
            raw["privileged_target_block_yaw"][0],
        ]
    )


def _raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def run() -> dict[str, Any]:
    output = common.STUDY_ROOT / "audit/amendment_independent_verification.json"
    if output.exists():
        raise RuntimeError("amendment verification already exists")
    seals = [
        _verify_seal(common.STUDY_ROOT / "audit/pre_generation_seal.json"),
        _verify_seal(common.STUDY_ROOT / "audit/protocol_amendment_01_seal.json"),
        _verify_seal(
            common.STUDY_ROOT / "audit/v4_manifest_schema_compatibility_seal.json"
        ),
        _verify_seal(
            common.STUDY_ROOT / "audit/v4_raw_loss_arithmetic_compatibility_seal.json"
        ),
    ]
    failed_smoke = common.study_json(
        common.STUDY_ROOT / "audit/smoke_pairing_pre_amendment_failed.json"
    )
    amended_smoke = common.study_json(
        common.STUDY_ROOT / "audit/smoke_pairing.json"
    )
    plan = common.study_json(common.role_raw_paths("main", "plan_oracle")[1])
    markov = common.study_json(common.role_raw_paths("main", "markov_oracle")[1])
    plan_by_slot = {int(item["slot"]): item for item in plan["episodes"][:30]}
    markov_by_slot = {int(item["slot"]): item for item in markov["episodes"]}
    pair_rows = []
    for slot in range(30):
        left = plan_by_slot[slot]
        right = markov_by_slot[slot]
        left_raw = _raw(common.REPO_ROOT / left["path"])
        right_raw = _raw(common.REPO_ROOT / right["path"])
        left_digest = _initial_digest(left_raw)
        right_digest = _initial_digest(right_raw)
        left_reset = left["generation_audit"]["environment_reset_amendment"]
        right_reset = right["generation_audit"]["environment_reset_amendment"]
        pair_rows.append(
            {
                "slot": slot,
                "env_seed_exact": left["env_seed"] == right["env_seed"],
                "independent_initial_digest_exact": left_digest == right_digest,
                "digests_match_manifests": (
                    left_digest == left["initial_state_sha256"]
                    and right_digest == right["initial_state_sha256"]
                ),
                "variation_values_digest_exact": (
                    left_reset["variation_values_sha256"]
                    == right_reset["variation_values_sha256"]
                ),
                "wrapper_seed_distinct": left["policy_seed"] != right["policy_seed"],
                "oracle_seed_distinct": left["oracle_np_seed"] != right["oracle_np_seed"],
            }
        )

    with np.load(
        common.STUDY_ROOT / "data/v4_markov_evaluation.npz", allow_pickle=False
    ) as stored:
        v4 = {name: stored[name].copy() for name in stored.files}
    with np.load(
        common.V4_ROOT / "data/confirmation_outcome_once.npz", allow_pickle=False
    ) as stored:
        prior = {name: stored[name].copy() for name in stored.files}
    bounded_path = common.STUDY_ROOT / "audit/bounded_reconstruction_check.json"
    bounded = common.study_json(bounded_path)
    report_manifest = common.study_json(
        common.STUDY_ROOT / "audit/report_manifest.json"
    )
    decision = common.study_json(common.STUDY_ROOT / "decision.json")
    postdecision = common.study_json(
        common.STUDY_ROOT / "postdecision_completion.json"
    )
    checks = {
        "all_seals_recomputed_exact": all(item["passed"] for item in seals),
        "failed_smoke_evidence_retained": (
            not failed_smoke["passed"]
            and not failed_smoke["checks"]["identical_initial_states"]
            and len(failed_smoke["pairs"]) == 12
        ),
        "amended_smoke_honestly_labels_original_failure": (
            amended_smoke["passed"]
            and not amended_smoke["original_pairing_passed"]
            and not amended_smoke["original_initial_states_equal"]
            and len(amended_smoke["pairs"]) == 12
        ),
        "all_30_main_pairs_independently_exact": all(
            all(value for key, value in row.items() if key != "slot")
            for row in pair_rows
        ),
        "main_counts_exact_no_replacements": (
            len(plan["episodes"]) == 90
            and len(markov["episodes"]) == 30
            and not plan["generation_failures"]
            and not markov["generation_failures"]
        ),
        "v4_calls_exact": bool(np.array_equal(v4["calls"], prior["calls"])),
        "v4_scores_exact": bool(np.array_equal(v4["scores"], prior["scores"])),
        "v4_legacy_raw_losses_exact": bool(
            np.array_equal(v4["legacy_v4_losses_d0_d4"][:, 1:], prior["losses"])
        ),
        "v4_whitened_losses_exact": bool(
            np.array_equal(v4["whitened_losses_d0_d4"][:, 1:], prior["whitened_losses"])
        ),
        "bounded_replay_exact_zero_new_episodes": (
            bounded["passed"]
            and bounded["new_environment_seeds"] == 0
            and bounded["new_policy_trajectories"] == 0
            and bounded["transition_replay"]["required_arrays_exact"]
        ),
        "report_hashes_current": (
            common.sha256_file(common.STUDY_ROOT / "REPORT.md")
            == report_manifest["report_sha256"]
            and common.sha256_file(common.STUDY_ROOT / "README.md")
            == report_manifest["readme_sha256"]
        ),
        "decision_and_postdecision_forbid_v5": (
            not decision["future_entirely_fresh_v5_confirmation_authorized"]
            and not decision["v5_launched"]
            and not postdecision["future_entirely_fresh_v5_confirmation_authorized"]
            and not postdecision["v5_launched"]
        ),
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    passed = all(
        value
        for key, value in checks.items()
        if key not in {"v3_test_targets_opened", "v5_confirmation_episodes"}
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "seals": seals,
        "main_pair_recomputation": pair_rows,
        "bounded_replay_sha256": common.sha256_file(bounded_path),
        "v4_canonical_vs_legacy_max_abs": float(
            np.max(
                np.abs(
                    v4["losses_d0_d4"] - v4["legacy_v4_losses_d0_d4"]
                )
            )
        ),
    }
    if not passed:
        raise RuntimeError(f"amendment independent verification failed: {checks}")
    common.write_study_json(output, result, exclusive=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
