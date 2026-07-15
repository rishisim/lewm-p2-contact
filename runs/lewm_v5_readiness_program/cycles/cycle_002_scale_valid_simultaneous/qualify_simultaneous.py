#!/usr/bin/env python3
"""Qualify the single scale-valid simultaneous-inference change on consumed data."""

from __future__ import annotations

import json
import math

import numpy as np

from cycle_common import ROOT, atomic_json, read_json, sha256_file


def manual_linear_quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    weight = location - lower
    return float(ordered[lower] + weight * (ordered[upper] - ordered[lower]))


def main() -> None:
    output_path = ROOT / "audit/simultaneous_inference_qualification.json"
    if output_path.exists():
        raise RuntimeError("simultaneous qualification already exists")
    diagnosis = read_json(ROOT / "development/cycle_001_consumed_diagnosis.json")
    source = ROOT.parent / "cycle_001_audit_repair"
    with np.load(source / "metrics/bootstrap_replicates.npz", allow_pickle=False) as stored:
        raw = stored["raw_vs_analytic"].astype(np.float64)
        white = stored["native_whitened_vs_analytic"].astype(np.float64)
    endpoint = {"raw_vs_analytic": raw, "native_whitened_vs_analytic": white}
    numpy_bounds = {name: float(np.quantile(values, 0.025)) for name, values in endpoint.items()}
    manual_bounds = {
        name: manual_linear_quantile(values, 0.025) for name, values in endpoint.items()
    }
    declared = diagnosis["simultaneous_inference_diagnosis"][
        "bonferroni_familywise_95_lower_bounds_diagnostic_only"
    ]
    analyze_text = (ROOT / "analyze.py").read_text()
    independent_text = (ROOT / "independent_verify.py").read_text()
    checks = {
        "source_cycle_is_consumed_failure": diagnosis["source_cohort_is_permanently_consumed"]
        and diagnosis["source_terminal_outcome"] == "cycle_prospective_discovery_failed",
        "exactly_one_candidate": diagnosis["bounded_change"]["candidate_count"] == 1,
        "policy_and_dgp_unchanged": all(
            not diagnosis["bounded_change"][name]
            for name in (
                "candidate_policy_changed",
                "base_changed",
                "refiner_changed",
                "gate_changed",
                "threshold_changed",
                "whitening_changed",
                "dgp_changed",
            )
        ),
        "endpoint_scale_ratio_exceeds_50": diagnosis["simultaneous_inference_diagnosis"][
            "bootstrap_sd_ratio_native_whitened_over_raw"
        ] > 50,
        "two_fixed_endpoints": set(endpoint)
        == {"raw_vs_analytic", "native_whitened_vs_analytic"},
        "bonferroni_union_bound_familywise_alpha": 2 * 0.025 <= 0.05,
        "numpy_and_manual_quantiles_agree": all(
            abs(numpy_bounds[name] - manual_bounds[name]) <= 2e-14 for name in endpoint
        ),
        "diagnostic_bounds_reproduced": all(
            abs(numpy_bounds[name] - float(declared[name])) <= 2e-14 for name in endpoint
        ),
        "analysis_uses_fixed_method": "bonferroni_two_endpoint_one_sided_percentile"
        in analyze_text
        and "np.max(centered_errors" not in analyze_text,
        "independent_verifier_has_separate_quantile_implementation": "independent_linear_quantile"
        in independent_text
        and "bonferroni_two_endpoint_one_sided_percentile" in independent_text,
        "individual_criteria_unchanged": diagnosis["bounded_change"][
            "individual_criteria_unchanged"
        ],
        "new_design_has_no_sequential_expansion": diagnosis["bounded_change"][
            "no_sequential_expansion"
        ],
    }
    result = {
        "schema_version": 1,
        "method": "Bonferroni two-fixed-endpoint one-sided percentile lower bounds",
        "family_size": 2,
        "familywise_alpha": 0.05,
        "per_endpoint_alpha": 0.025,
        "coverage_argument": "For events E_raw and E_white that their respective lower bound exceeds the true effect, P(E_raw union E_white) <= P(E_raw)+P(E_white) <= .025+.025=.05; no common unit, independence, or correlation assumption is used.",
        "consumed_diagnostic_bounds_only": numpy_bounds,
        "checks": checks,
        "passed": all(checks.values()),
        "analysis_source_sha256": sha256_file(ROOT / "analyze.py"),
        "independent_source_sha256": sha256_file(ROOT / "independent_verify.py"),
        "new_smoke_episodes_used": 0,
        "new_prospective_episodes_used": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output_path, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"simultaneous inference qualification failed: {checks}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
