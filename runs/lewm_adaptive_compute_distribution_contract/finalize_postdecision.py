#!/usr/bin/env python3
"""Record completion of the preregistered bounded failure diagnostic."""

from __future__ import annotations

import json

import common
import evaluate_v4_compat
import evaluate_v4_numeric_compat
import generator_seedfix


def run() -> dict:
    common.assert_pre_generation_seal()
    generator_seedfix.assert_amendment_seal()
    evaluate_v4_compat.assert_seal()
    evaluate_v4_numeric_compat.assert_seal()
    output = common.STUDY_ROOT / "postdecision_completion.json"
    if output.exists():
        raise RuntimeError("post-decision completion already exists")
    decision = common.study_json(common.STUDY_ROOT / "decision.json")
    bounded_path = common.STUDY_ROOT / "audit/bounded_reconstruction_check.json"
    bounded = common.study_json(bounded_path)
    if decision["decision"] != "generator_reconstruction_failed":
        raise RuntimeError("post-decision replay is not mapped from this verdict")
    if not (
        bounded["passed"]
        and bounded["new_environment_seeds"] == 0
        and bounded["new_policy_trajectories"] == 0
        and bounded["replayed_existing_trajectory_count"] == 1
        and bounded["transition_replay"]["required_arrays_exact"]
    ):
        raise RuntimeError("bounded replay/action-semantics check did not pass exactly")
    result = {
        "schema_version": 1,
        "mechanical_decision": decision["decision"],
        "preregistered_failure_branch_completed": True,
        "bounded_replay_passed": True,
        "bounded_replay_sha256": common.sha256_file(bounded_path),
        "new_environment_seeds_in_diagnostic": 0,
        "new_policy_trajectories_in_diagnostic": 0,
        "exact_pixels_observation_qpos_qvel_replay": True,
        "direct_action_semantics_verified": True,
        "remaining_unresolved_generator_differences": [
            "released offline HDF5 lacks its exact generation commit/config/seed manifest",
            "public upstream command horizon is 1001 while released local episodes use 200 actions",
            "installed stable-worldmodel Cube reset semantics differ from the public OGBench generator path",
        ],
        "gate_interpretation": (
            "The frozen gate passed every raw/FLOP/ranking criterion but had negative "
            "whitened point benefit. Because the distribution contract failed first, "
            "this is not a matched-distribution gate verdict and cannot motivate tuning here."
        ),
        "next_highest_information_step": (
            "Obtain or reconstruct the exact released HDF5 generator provenance "
            "(environment implementation, reset semantics, horizon, command/config, and "
            "seed handling), then preregister a new bounded generator-reconstruction study. "
            "Do not tune the gate or launch V5."
        ),
        "future_entirely_fresh_v5_confirmation_authorized": False,
        "v5_launched": False,
        "confirmatory_claim": False,
        "contact_aware_claim": False,
        "v3_test_targets_opened": False,
    }
    common.write_study_json(output, result, exclusive=True)

    report_path = common.STUDY_ROOT / "REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    old = (
        "Next step: Run only the smallest bounded environment/version/action-semantics "
        "or replay check; do not tune the gate.\n"
    )
    new = (
        "The preregistered smallest bounded check is complete and passed exact replay; "
        "it did not resolve the offline-generator mismatch.\n\n"
        "Next highest-information step: obtain or reconstruct the exact released HDF5 "
        "generator provenance (environment implementation, reset semantics, horizon, "
        "command/config, and seed handling), then preregister a new bounded generator-"
        "reconstruction study. Do not tune the gate or launch V5.\n"
    )
    if report.count(old) != 1:
        raise RuntimeError("final report post-decision marker changed unexpectedly")
    report_path.write_text(report.replace(old, new), encoding="utf-8")
    readme_path = common.STUDY_ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme += (
        "\n`postdecision_completion.json` records the completed one-trajectory replay "
        "and the final no-V5 recommendation.\n"
    )
    readme_path.write_text(readme, encoding="utf-8")
    report_manifest_path = common.STUDY_ROOT / "audit/report_manifest.json"
    report_manifest = common.study_json(report_manifest_path)
    report_manifest.update(
        {
            "report_sha256": common.sha256_file(report_path),
            "readme_sha256": common.sha256_file(readme_path),
            "postdecision_completion_sha256": common.sha256_file(output),
            "postdecision_completion_applied": True,
        }
    )
    common.write_study_json(report_manifest_path, report_manifest)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
