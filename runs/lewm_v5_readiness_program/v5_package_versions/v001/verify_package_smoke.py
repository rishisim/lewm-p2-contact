#!/usr/bin/env python3
"""Verify the 12 excluded package-smoke episodes without confirmation use."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cycle_common import (
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    REPO_ROOT,
    ROOT,
    atomic_json,
    read_json,
    sequential_calls,
    sha256_file,
)
from verify_pre_v5 import verify as verify_pre_v5


def main() -> None:
    output = ROOT / "audit/package_smoke_verification.json"
    if output.exists():
        raise RuntimeError("package-smoke verification is immutable and already exists")
    preseal = verify_pre_v5()
    raw_manifest = read_json(ROOT / "data/package_smoke_raw_manifest.json")
    execution_manifest = read_json(ROOT / "data/package_smoke_execution_manifest.json")
    latency = read_json(ROOT / "metrics/package_smoke_latency.json")
    ledger = read_json(ROOT / "cohort_seed_ledger.json")
    execution_path = ROOT / "data/package_smoke_execution.npz"
    if sha256_file(execution_path) != execution_manifest["sha256"]:
        raise RuntimeError("package-smoke execution hash drift")
    raw_directory = ROOT / "data/package_smoke_raw"
    expected_npz = {Path(record["path"]).name for record in raw_manifest["episodes"]}
    actual_npz = {path.name for path in raw_directory.glob("*.npz")}
    actual_json = {path.with_suffix(".npz").name for path in raw_directory.glob("*.json")}
    raw_hashes = []
    ledger_primary = {item["episode_id"]: item for item in ledger["roles"]["package_smoke"]}
    ledger_replacements = {item["episode_id"]: item for item in ledger["roles"]["replacement"]}
    seed_records_valid = True
    for record in raw_manifest["episodes"]:
        path = REPO_ROOT / record["path"]
        raw_hashes.append(sha256_file(path) == record["sha256"])
        primary = ledger_primary.get(record["episode_id"])
        source_pool = ledger_replacements if record["replacement_used"] else ledger_primary
        source = source_pool.get(record["seed_source_episode_id"])
        seed_records_valid &= bool(
            primary is not None
            and source is not None
            and int(primary["slot"]) == int(record["slot"])
            and all(
                int(record[key]) == int(source[key])
                for key in ("env_seed", "policy_seed", "oracle_np_seed")
            )
        )
    with np.load(execution_path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    calls = arrays["calls"].astype(np.int64)
    scores = arrays["scores"].astype(np.float64)
    features = arrays["features"].astype(np.float64)
    episode_ids = arrays["episode_id"].astype(np.int64)
    with np.load(ROOT / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        thresholds = stored["thresholds"].astype(np.float64)
    reconstructed = sequential_calls(scores, thresholds)
    reached = np.arange(3)[None, :] < np.minimum(calls, 3)[:, None]
    feature_reached = np.broadcast_to(reached[:, :, None], features.shape)
    graph = read_json(ROOT / "audit/flop_derivation_graph.json")
    symbolic = read_json(ROOT / "audit/flop_derivation_symbolic.json")
    forbidden = [ROOT / relative for relative in read_json(
        ROOT / "expected_artifact_roles.json"
    )["forbidden_at_v5_ready"]]
    confirmation_data_files = list((ROOT / "data").glob("v5_confirmation*"))
    seal_created = read_json(ROOT / "audit/pre_v5_seal.json")["created_unix_ns"]
    checks = {
        "pre_v5_seal": preseal["passed"],
        "seal_precedes_every_smoke_episode": int(seal_created)
        < min(int(item["created_unix_ns"]) for item in raw_manifest["episodes"]),
        "exact_12_raw_episodes": raw_manifest["episode_count"] == 12
        and len(raw_manifest["episodes"]) == 12,
        "exact_456_execution_rows": execution_manifest["rows"] == 456
        and len(calls) == 456,
        "episode_identifiers_exact": np.array_equal(np.unique(episode_ids), np.arange(12))
        and all(np.sum(episode_ids == index) == 38 for index in range(12)),
        "raw_path_set_complete": expected_npz == actual_npz == actual_json
        and len(expected_npz) == 12,
        "raw_hashes": all(raw_hashes),
        "seed_records_match_frozen_ledger": seed_records_valid,
        "seed_isolation": not ledger["overlap_with_prior_recorded_numeric_identifiers"]
        and not ledger["overlap_with_prior_recorded_string_identifiers"],
        "calls_in_range": bool(np.all((calls >= 1) & (calls <= 4))),
        "calls_reconstructed_exactly": bool(np.array_equal(reconstructed, calls)),
        "reached_scores_finite_unreached_nan": bool(
            np.isfinite(scores[reached]).all() and np.isnan(scores[~reached]).all()
        ),
        "reached_features_finite_unreached_nan": bool(
            np.isfinite(features[feature_reached]).all()
            and np.isnan(features[~feature_reached]).all()
        ),
        "model_outputs_finite": bool(
            all(
                np.isfinite(arrays[name]).all()
                for name in ("target", "dense_exits", "sparse_selected")
            )
        ),
        "input_allowlist": execution_manifest["loaded_input_keys"] == ["action", "pixels"],
        "contact_and_privileged_excluded": not execution_manifest[
            "contact_or_privileged_loaded"
        ],
        "v3_and_hdf5_excluded": not execution_manifest["v3_test_targets_opened"]
        and not execution_manifest["released_hdf5_opened"],
        "frozen_modules_no_gradients": execution_manifest["module_before"]
        == execution_manifest["module_after"]
        and execution_manifest["module_after"]["passed"]
        and execution_manifest["no_gradients"],
        "same_path_and_manual_api_exact": all(
            item["same_sparse_path_outputs_calls_scores_features_bitwise_exact"]
            and item["manual_sparse_vs_forward_selected_bitwise_exact"]
            for item in execution_manifest["equivalence"]
        ),
        "sparse_dense_calls_histograms_exact": all(
            item["sparse_vs_dense_calls_exact"] and item["call_histogram_exact"]
            for item in execution_manifest["equivalence"]
        ),
        "sparse_dense_latents_within_contract": all(
            item["cross_batch_allclose"]
            and item["cross_batch_max_abs_ceiling_passed"]
            and item["cross_batch_max_abs"] <= NUMERICAL_MAX_ABS
            for item in execution_manifest["equivalence"]
        )
        and execution_manifest["numerical_contract"]
        == {
            "rtol": NUMERICAL_RTOL,
            "atol": NUMERICAL_ATOL,
            "maximum_absolute_error_ceiling": NUMERICAL_MAX_ABS,
            "calls_scores_same_path_and_forward_selected_remain_exact": True,
        },
        "two_flop_derivations_agree": graph["passed"]
        and symbolic["passed"]
        and graph["totals"] == symbolic["totals"],
        "synchronized_latency_passed_separately": latency["passed"]
        and latency["actual_latency_separate_from_fully_counted_flops"]
        and latency["statistical_and_flop_verdict_independent_of_latency"],
        "zero_confirmation_artifacts": not confirmation_data_files
        and not any(path.exists() for path in forbidden),
        "zero_v5_outcomes": raw_manifest["v5_outcome_episodes"] == 0
        and execution_manifest["v5_outcome_episodes"] == 0
        and latency["v5_outcome_episodes"] == 0,
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "checks": checks,
        "call_histogram": np.bincount(calls, minlength=5)[1:].tolist(),
        "maximum_sparse_dense_absolute_error": max(
            item["cross_batch_max_abs"] for item in execution_manifest["equivalence"]
        ),
        "raw_manifest_sha256": sha256_file(ROOT / "data/package_smoke_raw_manifest.json"),
        "execution_sha256": sha256_file(execution_path),
        "execution_manifest_sha256": sha256_file(
            ROOT / "data/package_smoke_execution_manifest.json"
        ),
        "latency_sha256": sha256_file(ROOT / "metrics/package_smoke_latency.json"),
        "passed": all(checks.values()),
        "excluded_package_smoke_episode_count": 12,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"package smoke verification failed: {result}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
