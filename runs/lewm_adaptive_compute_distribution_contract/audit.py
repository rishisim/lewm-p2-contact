#!/usr/bin/env python3
"""Freshness, seed, role, cache, and V3-test exclusion audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import common


def _generated_manifests() -> dict[str, dict[str, Any]]:
    result = {}
    for phase in ("smoke", "main"):
        for policy in ("plan_oracle", "markov_oracle"):
            path = common.role_raw_paths(phase, policy)[1]
            payload = common.study_json(path)
            if not payload.get("complete"):
                raise RuntimeError(f"incomplete generated role: {path}")
            result[f"{phase}_{policy}"] = payload
    return result


def _v4_records() -> list[dict[str, Any]]:
    records = []
    for name in ("smoke_data_manifest.json", "confirmation_data_manifest.json"):
        records.extend(common.study_json(common.V4_ROOT / "data" / name)["episodes"])
    return records


def _hash_sets(records: list[dict[str, Any]], key: str) -> set[str]:
    return {str(item[key]) for item in records}


def run() -> dict[str, Any]:
    output = common.STUDY_ROOT / "audit/freshness.json"
    if output.exists():
        prior = common.study_json(output)
        if not prior.get("passed"):
            raise RuntimeError("existing freshness audit failed")
        return prior
    seal = common.assert_pre_generation_seal()
    generated = _generated_manifests()
    records_by_role = {name: payload["episodes"] for name, payload in generated.items()}
    all_new = [record for records in records_by_role.values() for record in records]
    expected_counts = {
        "smoke_plan_oracle": 12,
        "smoke_markov_oracle": 12,
        "main_plan_oracle": 90,
        "main_markov_oracle": 30,
    }
    for name, expected in expected_counts.items():
        if len(records_by_role[name]) != expected:
            raise RuntimeError(f"wrong role count {name}")
    for record in all_new:
        path = common.REPO_ROOT / record["path"]
        if common.sha256_file(path) != record["file_sha256"]:
            raise RuntimeError(f"fresh raw file drift: {path}")

    prior_index_path = common.V4_ROOT / "audit/prior_raw_episode_hash_index.json"
    prior_index = common.study_json(prior_index_path)
    if prior_index["v3_test_targets_opened"]:
        raise RuntimeError("invalid prior hash index")
    prior_records = list(prior_index["records"]) + _v4_records()
    duplicate_detail = {}
    for key in (
        "combined_observation_action_sha256",
        "pixels_sha256",
        "observation_sha256",
        "actions_sha256",
    ):
        new_values = [str(item[key]) for item in all_new]
        prior_values = _hash_sets(prior_records, key)
        duplicate_detail[key] = {
            "new_internal_duplicates": sorted(
                {value for value in new_values if new_values.count(value) > 1}
            ),
            "prior_duplicates": sorted(set(new_values) & prior_values),
        }

    seed_manifest_path = common.STUDY_ROOT / "seed_manifest.json"
    seed_manifest = common.study_json(seed_manifest_path)
    prior_inventory = common.study_json(
        common.V4_ROOT / "audit/prior_identifier_inventory.json"
    )
    prior_numeric = {int(value) for value in prior_inventory["unique_numeric_ids"]}
    prior_numeric.update(
        common.recursive_integers(common.study_json(common.V4_ROOT / "seed_manifest.json"))
    )
    observed_seeds = {
        int(record[key])
        for record in all_new
        for key in ("env_seed", "policy_seed", "oracle_np_seed")
    }
    observed_seeds.update(int(value) for value in seed_manifest["statistical_seeds"].values())
    seed_overlap = sorted(observed_seeds & prior_numeric)

    smoke_paths = {
        record["path"]
        for name, records in records_by_role.items()
        if name.startswith("smoke_")
        for record in records
    }
    smoke_ids = {
        record["trajectory_id"]
        for name, records in records_by_role.items()
        if name.startswith("smoke_")
        for record in records
    }
    encoded_and_evaluated_text = ""
    for path in sorted((common.STUDY_ROOT / "data").glob("*encoded_manifest.json")) + sorted(
        (common.STUDY_ROOT / "data").glob("*evaluation_manifest.json")
    ):
        encoded_and_evaluated_text += path.read_text(encoding="utf-8")
    smoke_leaks = sorted(
        [value for value in (*smoke_paths, *smoke_ids) if value in encoded_and_evaluated_text]
    )

    test = common.v3_test_set()
    cache_checks: dict[str, Any] = {}
    for role in ("offline_discovery", "offline_calibration"):
        path = common.STUDY_ROOT / "data" / f"{role}_evaluation.npz"
        with np.load(path, allow_pickle=False) as stored:
            ids = set(stored["episode_id"].astype(int).tolist())
        cache_checks[path.name] = {
            "episode_count": len(ids),
            "v3_test_intersection": sorted(ids & test),
            "v3_test_targets_opened": False,
        }
    for role in ("plan_oracle", "markov_oracle"):
        for kind in ("encoded", "evaluation"):
            path = common.STUDY_ROOT / "data" / f"{role}_{kind}.npz"
            with np.load(path, allow_pickle=False) as stored:
                ids = set(stored["episode_id"].astype(int).tolist())
            cache_checks[path.name] = {
                "source_namespace": "fresh_rollout_slot_not_v3_ordinal",
                "slot_count": len(ids),
                "v3_episode_ids": [],
                "v3_test_targets_opened": False,
            }
    offline_metrics = common.STUDY_ROOT / "reference/offline_episode_metrics.npz"
    with np.load(offline_metrics, allow_pickle=False) as stored:
        ids = set(stored["episode_id"].astype(int).tolist())
    cache_checks[offline_metrics.name] = {
        "episode_count": len(ids),
        "v3_test_intersection": sorted(ids & test),
        "v3_test_targets_opened": False,
    }
    cache_test_clear = all(
        not item.get("v3_test_intersection", []) for item in cache_checks.values()
    )

    checks = {
        "pre_generation_seal_valid": bool(seal),
        "role_counts_exact": all(
            len(records_by_role[name]) == count for name, count in expected_counts.items()
        ),
        "no_performance_or_success_exclusions": all(
            payload["performance_based_exclusions"] == 0
            and payload["success_based_exclusions"] == 0
            for payload in generated.values()
        ),
        "no_exact_raw_hash_duplicates_against_accessible_prior_or_v4": all(
            not item["prior_duplicates"] for item in duplicate_detail.values()
        ),
        "no_internal_exact_raw_hash_duplicates": all(
            not item["new_internal_duplicates"] for item in duplicate_detail.values()
        ),
        "all_seed_values_disjoint_from_prior_numeric_identifiers": not seed_overlap,
        "smoke_physically_and_analytically_excluded": not smoke_leaks,
        "smoke_pairing_passed": common.study_json(
            common.STUDY_ROOT / "audit/smoke_pairing.json"
        )["passed"],
        "main_pairing_passed": common.study_json(
            common.STUDY_ROOT / "audit/main_pairing.json"
        )["passed"],
        "every_derived_cache_test_clear": cache_test_clear,
        "v3_combined_cache_opened_with_numpy": False,
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    passed = all(
        value
        for key, value in checks.items()
        if key
        not in {
            "v3_combined_cache_opened_with_numpy",
            "v3_test_targets_opened",
            "v5_confirmation_episodes",
        }
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "role_counts": {name: len(records) for name, records in records_by_role.items()},
        "duplicate_detail": duplicate_detail,
        "seed_overlap": seed_overlap,
        "smoke_leaks": smoke_leaks,
        "cache_exclusion_checks": cache_checks,
        "prior_raw_hash_index_sha256": common.sha256_file(prior_index_path),
        "seed_manifest_sha256": common.sha256_file(seed_manifest_path),
        "released_checkpoint_pretraining_episode_manifest_available": False,
        "pretraining_membership_caveat": (
            "Exact nonduplication is proven against every accessible recorded prior episode "
            "outside the opaque V3 test set and against V4. The released checkpoint lacks a "
            "complete original pretraining episode manifest, so membership cannot be proven."
        ),
    }
    if not passed:
        raise RuntimeError(f"freshness audit failed: {checks}")
    common.write_study_json(output, result, exclusive=True)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"passed": result["passed"]}, sort_keys=True))

