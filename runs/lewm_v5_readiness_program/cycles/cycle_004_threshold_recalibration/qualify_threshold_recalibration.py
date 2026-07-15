#!/usr/bin/env python3
"""Fail-closed qualification of the isolated threshold-only recalibration."""

from __future__ import annotations

import json

from cycle_common import (
    EXPECTED_SOURCE_HASHES,
    ROOT,
    SOURCE_DISCOVERY,
    THRESHOLD,
    atomic_json,
    read_json,
    sha256_file,
    verify_expected_sources,
)


def main() -> None:
    output_path = ROOT / "audit/threshold_recalibration_qualification.json"
    if output_path.exists():
        raise RuntimeError("threshold qualification already exists")
    protocol_path = ROOT / "development/threshold_selection_protocol.json"
    result_path = ROOT / "development/threshold_selection_result.json"
    diagnosis_path = ROOT / "development/cycle_001_cycle_003_diagnosis.json"
    selected_path = ROOT / "development/selected_threshold.json"
    protocol = read_json(protocol_path)
    result = read_json(result_path)
    diagnosis = read_json(diagnosis_path)
    selected = read_json(selected_path)
    source_hashes = verify_expected_sources()
    selected_metrics = result["selected_on_cycle_001"]
    validation = result["one_shot_cycle_003_validation"]
    checks = {
        "protocol_precedes_counterfactual_access": protocol[
            "written_before_counterfactual_target_access"
        ],
        "selection_script_hash_matches_protocol": sha256_file(
            ROOT / "select_threshold.py"
        )
        == protocol["script_sha256"],
        "finite_five_candidate_design": protocol["candidate_count"] == 5
        and len(protocol["candidate_thresholds"]) == 5,
        "all_candidates_upward_and_counterfactually_complete": min(
            protocol["candidate_thresholds"]
        )
        == protocol["candidate_thresholds"][0]
        and protocol["candidate_thresholds"] == sorted(protocol["candidate_thresholds"]),
        "threshold_only_change": protocol["highest_information_change"]
        == "threshold_only",
        "cycle_001_selection_cycle_003_one_shot_validation": "cycle_001"
        in protocol["selection_cohort"]
        and "cycle_003" in protocol["validation_cohort"],
        "selection_result_bound_to_protocol": result["protocol_sha256"]
        == sha256_file(protocol_path),
        "selection_passed": selected_metrics["eligible"],
        "one_shot_validation_passed": result["validation_passed"]
        and validation["eligible"],
        "threshold_frozen_exact": selected["threshold"] == THRESHOLD
        and selected_metrics["threshold"] == THRESHOLD
        and validation["threshold"] == THRESHOLD,
        "selection_result_hash_bound": selected["selection_result_sha256"]
        == sha256_file(result_path),
        "diagnosis_hash_bound": selected["diagnosis_sha256"]
        == sha256_file(diagnosis_path),
        "all_required_diagnostic_dimensions": set(
            diagnosis["diagnostic_dimensions_complete"]
        )
        == {
            "heterogeneity",
            "calibration",
            "compute_price",
            "stagewise_gains",
            "gate_margins",
            "DGP_stability",
        },
        "all_selection_lower_bounds_positive": all(
            value["lower"] > 0 for value in selected_metrics["criteria"].values()
        ),
        "all_validation_lower_bounds_positive": all(
            value["lower"] > 0 for value in validation["criteria"].values()
        ),
        "all_selection_stage_signs_positive": all(
            value["positive_sign"] for value in selected_metrics["stagewise"]
        ),
        "all_validation_stage_signs_positive": all(
            value["positive_sign"] for value in validation["stagewise"]
        ),
        "base_refiner_gate_whitening_unchanged": selected[
            "base_refiner_gate_whitening_unchanged"
        ]
        and source_hashes == EXPECTED_SOURCE_HASHES,
        "gate_weights_still_original_hash": sha256_file(
            SOURCE_DISCOVERY / "freeze/gate_weights.npz"
        )
        == EXPECTED_SOURCE_HASHES["gate_weights"],
        "no_new_cycle_data_before_qualification": not (ROOT / "data").exists(),
        "fixed_new_sample_sizes": selected["requires_new_sparse_smoke"] == 12
        and selected["requires_new_prospective"] == 300
        and protocol["no_sequential_expansion"],
        "zero_v5_outcomes": selected["v5_outcome_episodes"] == 0
        and result["v5_outcome_episodes"] == 0
        and diagnosis["v5_outcome_episodes"] == 0,
    }
    audit = {
        "schema_version": 1,
        "change": "threshold_only",
        "old_threshold": selected["old_threshold"],
        "new_threshold": selected["threshold"],
        "candidate_count": protocol["candidate_count"],
        "selection_protocol_sha256": sha256_file(protocol_path),
        "selection_result_sha256": sha256_file(result_path),
        "diagnosis_sha256": sha256_file(diagnosis_path),
        "selected_threshold_sha256": sha256_file(selected_path),
        "checks": checks,
        "passed": all(checks.values()),
        "new_smoke_episodes_used": 0,
        "new_prospective_episodes_used": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output_path, audit, exclusive=True)
    if not audit["passed"]:
        raise RuntimeError(f"threshold qualification failed: {checks}")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
