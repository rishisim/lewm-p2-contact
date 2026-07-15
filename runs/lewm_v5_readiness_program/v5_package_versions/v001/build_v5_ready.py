#!/usr/bin/env python3
"""Create the program-level V5_READY record after final package verification."""

from __future__ import annotations

import json
import time

from cycle_common import PROGRAM_ROOT, ROOT, SOURCE_CYCLE, atomic_json, read_json, sha256_file
from verify_final_package import verify as verify_final_package
from verify_pre_v5 import verify as verify_pre_v5


def main() -> None:
    output = PROGRAM_ROOT / "V5_READY.json"
    if output.exists():
        raise RuntimeError("V5_READY.json is immutable and already exists")
    final = verify_final_package()
    preseal = verify_pre_v5()
    ready = read_json(ROOT / "PACKAGE_READY.json")
    smoke = read_json(ROOT / "audit/package_smoke_verification.json")
    power = read_json(ROOT / "power_analysis.json")
    candidate = read_json(ROOT / "freeze/frozen_candidate_manifest.json")
    discovery = read_json(SOURCE_CYCLE / "decision.json")
    discovery_audit = read_json(SOURCE_CYCLE / "audit/independent_verification.json")
    forbidden = [
        ROOT / relative
        for relative in read_json(ROOT / "expected_artifact_roles.json")[
            "forbidden_at_v5_ready"
        ]
    ]
    zero_outcomes = (
        not list((ROOT / "data").glob("v5_confirmation*"))
        and not any(path.exists() for path in forbidden)
        and ready["v5_outcome_episodes"] == 0
        and smoke["v5_outcome_episodes"] == 0
        and final["v5_outcome_episodes"] == 0
    )
    criteria = {
        "fresh_prospective_discovery_passed": discovery["terminal_outcome"]
        == "cycle_prospective_discovery_passed",
        "independent_discovery_reproduction_passed": discovery_audit["passed"],
        "integrity_and_isolation_audits_passed": all(discovery["integrity"].values()),
        "one_candidate_and_claim_frozen": candidate["all_hashes_verified"]
        and candidate["candidate_id"] == "stage_dual_r0.01_q0.85",
        "power_at_least_90_percent_for_both_co_primary_joint_design": power["passed"]
        and power["stress_power"]["joint_power_lower_by_union_bound"] >= 0.90,
        "complete_preregistration_cohort_runner_analysis_decision_and_verifier": all(
            (ROOT / path).exists()
            for path in (
                "PREREGISTRATION.md",
                "DGP.json",
                "cohort_seed_ledger.json",
                "runner.py",
                "analysis.py",
                "outcome_mapping.json",
                "independent_verify.py",
                "transitive_source_package_manifest.json",
            )
        ),
        "pre_v5_seal_verifies": preseal["passed"],
        "excluded_package_smoke_passed": smoke["passed"],
        "final_package_manifest_verifies": final["passed"],
        "zero_v5_outcome_episodes": zero_outcomes,
    }
    if not all(criteria.values()):
        raise RuntimeError(f"V5_READY criteria not satisfied: {criteria}")
    payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "status": "V5_READY",
        "criteria": criteria,
        "accepted_discovery_cycle": 7,
        "accepted_discovery_path": str(SOURCE_CYCLE),
        "accepted_discovery_terminal_outcome": discovery["terminal_outcome"],
        "accepted_discovery_effects": {
            "raw_vs_analytic": discovery["criteria"]["raw_vs_analytic"],
            "native_whitened_vs_analytic": discovery["criteria"][
                "native_whitened_vs_analytic"
            ],
            "simultaneous_co_primary": discovery["simultaneous_co_primary"],
        },
        "accepted_discovery_compute": discovery["compute"],
        "source_hashes": {
            "accepted_decision": sha256_file(SOURCE_CYCLE / "decision.json"),
            "accepted_independent_verification": sha256_file(
                SOURCE_CYCLE / "audit/independent_verification.json"
            ),
            "accepted_final_manifest": sha256_file(
                SOURCE_CYCLE / "artifact_manifest.json"
            ),
            "compiled_gate": sha256_file(ROOT / "freeze/compiled_gate.npz"),
            "gate_fit": sha256_file(ROOT / "freeze/gate_fit.npz"),
            "whitening": sha256_file(ROOT / "freeze/whitening.npz"),
            "frozen_candidate_manifest": sha256_file(
                ROOT / "freeze/frozen_candidate_manifest.json"
            ),
        },
        "frozen_candidate_id": candidate["candidate_id"],
        "frozen_claim": candidate["claim"],
        "v5_package_version": "v001",
        "v5_package_path": str(ROOT),
        "selected_v5_sample_size": power["selected_v5_sample_size"],
        "power_assumptions": {
            "method": power["method"],
            "alpha_per_endpoint": power["alpha_per_endpoint"],
            "effect_definition": power["effect_definition"],
            "conservative_effects": power["conservative_effects"],
            "primary_design_effect": power["primary_design_effect"],
            "primary_joint_power_lower": power["primary_power"][
                "joint_power_lower_by_union_bound"
            ],
            "stress_design_effect": power["stress_design_effect"],
            "stress_joint_power_lower": power["stress_power"][
                "joint_power_lower_by_union_bound"
            ],
        },
        "remaining_irreducible_statistical_uncertainty": power[
            "remaining_uncertainty"
        ],
        "pre_v5_seal_sha256": preseal["seal_sha256"],
        "package_smoke_verification_sha256": sha256_file(
            ROOT / "audit/package_smoke_verification.json"
        ),
        "package_final_manifest_sha256": final["manifest_sha256"],
        "package_final_file_count": final["file_count"],
        "v5_confirmation_launched": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output, payload, exclusive=True)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
