#!/usr/bin/env python3
"""Freeze and quantify the cross-batch float32 numerical contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cycle_common import (
    DISCOVERY_CODE,
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    REPO_ROOT,
    ROOT,
    SOURCE_DISCOVERY,
    atomic_json,
    read_json,
    sha256_file,
)


def collect_key(value: Any, key: str, output: list[float]) -> None:
    if isinstance(value, dict):
        if key in value:
            output.append(float(value[key]))
        for item in value.values():
            collect_key(item, key, output)
    elif isinstance(value, list):
        for item in value:
            collect_key(item, key, output)


def main() -> None:
    diagnosis = read_json(ROOT / "audit/diagnosis_input.json")
    runtime = read_json(ROOT / "audit/runtime_qualification.json")
    consumed_paths = [
        DISCOVERY_CODE / "metrics/internal_stagewise__full.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_v4/metrics/runtime.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/metrics/latency.json",
    ]
    consumed_differences = [
        float(
            diagnosis["first_consumed_smoke_episode"][
                "sparse_dense_maximum_absolute_difference"
            ]
        )
    ]
    for path in consumed_paths:
        payload = json.loads(path.read_text())
        collect_key(payload, "dense_selected_max_abs_difference", consumed_differences)
        collect_key(payload, "dense_reference_max_abs", consumed_differences)
        collect_key(payload, "dense_optimized_max_abs", consumed_differences)
        collect_key(payload, "dense_reference_max_abs_output", consumed_differences)
        collect_key(payload, "dense_optimized_max_abs_output", consumed_differences)
    runtime_maxima = [
        float(item["cross_batch_max_abs"]) for item in runtime["devices"]
    ]
    consumed_differences.extend(runtime_maxima)
    observed_maximum = max(consumed_differences)

    test_path = DISCOVERY_CODE / "tests/test_models.py"
    test_text = test_path.read_text()
    repository_standard_present = "rtol=2e-6" in test_text and "atol=2e-7" in test_text
    decision = read_json(SOURCE_DISCOVERY / "decision.json")
    raw_mse = float(decision["adaptive_raw_mse"])
    white_mse = float(decision["adaptive_native_whitened_mse"])
    with np.load(SOURCE_DISCOVERY / "freeze/gate_contract.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    whitening_operator_norm = float(np.linalg.svd(whitening, compute_uv=False)[0])
    raw_perturbation = NUMERICAL_MAX_ABS
    white_perturbation = whitening_operator_norm * NUMERICAL_MAX_ABS
    raw_mse_shift_bound = 2 * raw_perturbation * np.sqrt(raw_mse) + raw_perturbation**2
    white_mse_shift_bound = 2 * white_perturbation * np.sqrt(white_mse) + white_perturbation**2

    primary_impact = {}
    for name, interval in decision["criteria"].items():
        scale = "native_whitened" if name.startswith("white_") else "raw"
        bound = white_mse_shift_bound if scale == "native_whitened" else raw_mse_shift_bound
        primary_impact[name] = {
            "scale": scale,
            "consumed_estimate": float(interval["estimate"]),
            "consumed_lower_bound": float(interval["lower"]),
            "maximum_contrast_shift_bound": float(bound),
            "fraction_of_consumed_effect_estimate": float(bound / interval["estimate"]),
            "fraction_of_consumed_positive_lower_bound": float(bound / interval["lower"]),
        }

    contract = {
        "schema_version": 1,
        "status": "carried_forward_unchanged_before_new_cycle_002_smoke",
        "scope": "cross-batch-shape selected latent tensors only",
        "tensor_contract": {
            "torch_allclose_rtol": NUMERICAL_RTOL,
            "torch_allclose_atol": NUMERICAL_ATOL,
            "independent_maximum_absolute_error_ceiling": NUMERICAL_MAX_ABS,
            "both_allclose_and_ceiling_required": True,
        },
        "still_bitwise_exact": [
            "call decisions",
            "stopping depths",
            "call histograms",
            "gate scores and features when the same sparse path is repeated",
            "manual sparse output versus StagewiseResidualCascade.forward_selected",
            "depth-one selected rows",
        ],
        "consumed_evidence": {
            "largest_observed_cross_batch_max_abs": observed_maximum,
            "ceiling_multiple_of_largest_observed": NUMERICAL_MAX_ABS / observed_maximum,
            "sources": [
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in consumed_paths
            ],
            "runtime_qualification_path": str(
                (ROOT / "audit/runtime_qualification.json").relative_to(REPO_ROOT)
            ),
            "runtime_tested_devices": runtime["tested_devices"],
            "repository_selected_execution_test_path": str(test_path.relative_to(REPO_ROOT)),
            "repository_selected_execution_test_sha256": sha256_file(test_path),
            "repository_rtol_2e_6_atol_2e_7_present": repository_standard_present,
        },
        "scale_justification": {
            "latent_raw_rmse": float(np.sqrt(raw_mse)),
            "ceiling_fraction_of_latent_raw_rmse": float(NUMERICAL_MAX_ABS / np.sqrt(raw_mse)),
            "native_whitening_operator_norm": whitening_operator_norm,
            "raw_per_row_mse_shift_bound": float(raw_mse_shift_bound),
            "native_whitened_per_row_mse_shift_bound": float(white_mse_shift_bound),
            "bound_formula": "|MSE(p+e,t)-MSE(p,t)| <= 2*RMSE(p-t)*RMSE(e)+RMSE(e)^2; raw RMSE(e)<=ceiling and whitened RMSE(e)<=||W||2*ceiling",
            "every_consumed_primary_contrast": primary_impact,
        },
        "acceptance_checks": {
            "ceiling_exceeds_or_equals_all_consumed_diagnostics": observed_maximum <= NUMERICAL_MAX_ABS,
            "ceiling_is_exactly_two_times_largest_consumed_discrepancy": np.isclose(
                NUMERICAL_MAX_ABS / observed_maximum, 2.0
            ),
            "repository_tighter_tolerances_adopted": repository_standard_present,
            "cpu_mps_runtime_qualification_passed": bool(runtime["passed"]),
            "raw_bound_below_two_percent_of_smallest_consumed_raw_effect": max(
                item["fraction_of_consumed_effect_estimate"]
                for item in primary_impact.values()
                if item["scale"] == "raw"
            ) < 0.02,
            "white_bound_below_five_percent_of_smallest_consumed_white_effect": max(
                item["fraction_of_consumed_effect_estimate"]
                for item in primary_impact.values()
                if item["scale"] == "native_whitened"
            ) < 0.05,
        },
        "tolerance_never_applies_to_calls_or_gate_repeatability": True,
        "prospective_episodes_used_to_choose_contract": 0,
        "cycle_001_prospective_used_to_change_contract": False,
        "v5_outcome_episodes": 0,
    }
    contract["passed"] = all(contract["acceptance_checks"].values())
    if not contract["passed"]:
        raise RuntimeError(f"numerical contract qualification failed: {contract['acceptance_checks']}")
    output_path = ROOT / "numerical_equivalence_contract.json"
    atomic_json(output_path, contract, exclusive=True)
    print(json.dumps(json.loads(output_path.read_text()), sort_keys=True))


if __name__ == "__main__":
    main()
