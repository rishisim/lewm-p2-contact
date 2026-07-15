#!/usr/bin/env python3
"""Freshness, duplicate, role-isolation, and allowlist audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import hdf5plugin  # noqa: F401
import h5py
import numpy as np

import common


def prior_episode_ids() -> tuple[list[int], list[int]]:
    inventory = common.read_json(common.ROOT / "audit/prior_identifier_inventory.json")
    forbidden_test = set(map(int, inventory["v3_test_episode_ids"]))
    candidates = set()
    for record in inventory["records"]:
        field = str(record["field"]).lower()
        value = int(record["value"])
        if "episode" in field and 0 <= value < 10_000:
            candidates.add(value)
    candidates.update(range(30))  # explicitly recorded Cube event diagnostic
    accessible = sorted(candidates - forbidden_test)
    return accessible, sorted(forbidden_test)


def build_prior_hash_index() -> dict[str, Any]:
    output = common.ROOT / "audit/prior_raw_episode_hash_index.json"
    if output.exists():
        payload = common.read_json(output)
        if payload["v3_test_targets_opened"]:
            raise RuntimeError("invalid prior hash index claims V3 test access")
        return payload
    accessible, forbidden = prior_episode_ids()
    records = []
    with h5py.File(common.SOURCE_H5, "r", swmr=True) as h5:
        for index, episode in enumerate(accessible):
            if episode in forbidden:
                raise RuntimeError("V3 test episode reached prior raw hash loop")
            offset = int(h5["ep_offset"][episode])
            length = int(h5["ep_len"][episode])
            if length != common.RAW_EPISODE_ROWS:
                raise RuntimeError(f"unexpected prior episode length {episode}: {length}")
            pixels = np.asarray(h5["pixels"][offset : offset + length], dtype=np.uint8)
            observation = np.asarray(h5["observation"][offset : offset + length])
            action = np.asarray(h5["action"][offset : offset + length], dtype=np.float32)
            records.append(
                {
                    "episode_id": episode,
                    "pixels_sha256": common.array_sha256(pixels),
                    "observation_sha256": common.array_sha256(observation),
                    "actions_sha256": common.array_sha256(action),
                    "combined_observation_action_sha256": common.combined_array_sha256(
                        [pixels, observation, action]
                    ),
                }
            )
            if (index + 1) % 100 == 0:
                print(f"hashed prior accessible episode {index + 1}/{len(accessible)}", flush=True)
    payload = {
        "schema_version": 1,
        "source_h5": str(common.SOURCE_H5),
        "source_h5_bytes": int(common.SOURCE_H5.stat().st_size),
        "accessible_recorded_episode_count": len(accessible),
        "accessible_recorded_episode_ids": accessible,
        "v3_test_episode_ids_opaque": forbidden,
        "v3_test_targets_opened": False,
        "records": records,
    }
    common.write_json(output, payload, exclusive=True)
    return payload


def load_generated(role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, manifest_path = common.role_paths(role)
    manifest = common.read_json(manifest_path)
    expected = 12 if role == "smoke" else 300
    if not manifest["complete"] or manifest["episode_count"] != expected:
        raise RuntimeError(f"incomplete generated {role} role")
    records = manifest["episodes"]
    for record in records:
        path = common.REPO / record["path"]
        if common.sha256_file(path) != record["file_sha256"]:
            raise RuntimeError(f"generated episode file hash drift: {path}")
        if record["role"] != role:
            raise RuntimeError("generated role label mismatch")
    return manifest, records


def audit() -> dict[str, Any]:
    inventory = common.read_json(common.ROOT / "audit/prior_identifier_inventory.json")
    recorded_ids = set(map(int, inventory["unique_numeric_ids"]))
    prior = build_prior_hash_index()
    smoke_manifest, smoke = load_generated("smoke")
    confirmation_manifest, confirmation = load_generated("confirmation")
    smoke_hashes = {item["combined_observation_action_sha256"] for item in smoke}
    confirmation_hashes = {
        item["combined_observation_action_sha256"] for item in confirmation
    }
    prior_hashes = {
        item["combined_observation_action_sha256"] for item in prior["records"]
    }
    all_new_seeds = {
        int(item[key])
        for item in (*smoke, *confirmation)
        for key in ("env_seed", "policy_seed")
    }
    seed_overlap = sorted(all_new_seeds & recorded_ids)
    cross_role_duplicates = sorted(smoke_hashes & confirmation_hashes)
    prior_duplicates = sorted((smoke_hashes | confirmation_hashes) & prior_hashes)
    path_overlap = sorted(
        {item["path"] for item in smoke} & {item["path"] for item in confirmation}
    )
    trajectory_overlap = sorted(
        {item["trajectory_id"] for item in smoke}
        & {item["trajectory_id"] for item in confirmation}
    )
    checks = {
        "seed_nonoverlap_with_all_recorded_numeric_ids": not seed_overlap,
        "no_exact_observation_action_duplicate_against_accessible_prior": not prior_duplicates,
        "no_smoke_confirmation_exact_duplicate": not cross_role_duplicates,
        "physical_paths_disjoint": not path_overlap,
        "trajectory_identifiers_disjoint": not trajectory_overlap,
        "smoke_count_exact": len(smoke) == 12,
        "confirmation_count_exact": len(confirmation) == 300,
        "v3_test_targets_opened": False,
        "performance_based_exclusions_absent": smoke_manifest["performance_based_exclusions"]
        == 0
        and confirmation_manifest["performance_based_exclusions"] == 0,
    }
    passed = all(value is True for key, value in checks.items() if key != "v3_test_targets_opened") and checks[
        "v3_test_targets_opened"
    ] is False
    result = {
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "seed_overlap": seed_overlap,
        "prior_exact_duplicates": prior_duplicates,
        "cross_role_exact_duplicates": cross_role_duplicates,
        "path_overlap": path_overlap,
        "trajectory_overlap": trajectory_overlap,
        "prior_raw_hash_index_sha256": common.sha256_file(
            common.ROOT / "audit/prior_raw_episode_hash_index.json"
        ),
        "smoke_data_manifest_sha256": common.sha256_file(
            common.ROOT / "data/smoke_data_manifest.json"
        ),
        "confirmation_data_manifest_sha256": common.sha256_file(
            common.ROOT / "data/confirmation_data_manifest.json"
        ),
        "released_checkpoint_pretraining_manifest_available": False,
        "pretraining_membership_caveat": (
            "The released LeWM checkpoint has no original pretraining episode manifest; "
            "mechanical nonmembership cannot be proven even though V4 trajectories were "
            "generated now from new frozen seeds and are not exact duplicates of every "
            "accessible recorded prior episode outside the untouched V3 test set."
        ),
    }
    if not passed:
        raise RuntimeError(f"freshness/isolation audit failed: {checks}")
    common.write_json(common.ROOT / "audit/freshness_duplicate_audit.json", result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("index", "audit"))
    args = parser.parse_args()
    if args.mode == "index":
        result = build_prior_hash_index()
        print(
            json.dumps(
                {
                    "accessible_recorded_episode_count": result[
                        "accessible_recorded_episode_count"
                    ],
                    "v3_test_targets_opened": False,
                },
                sort_keys=True,
            )
        )
    else:
        result = audit()
        print(json.dumps({"passed": result["passed"], "v3_test_targets_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
