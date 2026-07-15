#!/usr/bin/env python3
"""Independent recomputation, focused tests, and final artifact hash audit."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import analyze
import common


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def independent_recomputation() -> dict[str, Any]:
    output = common.STUDY_ROOT / "audit/independent_recomputation.json"
    if output.exists():
        prior = common.study_json(output)
        if not prior.get("passed"):
            raise RuntimeError("existing independent recomputation failed")
        return prior
    plan_input = _load(common.STUDY_ROOT / "data/plan_oracle_encoded.npz")
    plan = _load(common.STUDY_ROOT / "data/plan_oracle_evaluation.npz")
    gate = common.study_json(common.STUDY_ROOT / "metrics/gate_prospective.json")
    comparison = common.study_json(
        common.STUDY_ROOT / "metrics/distribution_comparison.json"
    )
    decision = common.study_json(common.STUDY_ROOT / "decision.json")
    d0 = np.square(
        plan_input["base_pred"].astype(np.float64) - plan_input["target"].astype(np.float64)
    ).mean(1)
    calls = np.ones(len(plan["scores"]), dtype=np.int64)
    active = np.ones(len(calls), dtype=bool)
    for stage in range(3):
        active &= plan["scores"][:, stage] > common.COMPUTE_PRICE
        calls += active.astype(np.int64)
    solver_calls = int(calls.sum())
    gate_decisions = int(np.minimum(calls, 3).sum())
    gate_flops = gate_decisions * 8_196
    fractional_calls = solver_calls + gate_flops / 264_960
    common_total = len(calls) * (70_529_190 + 669_184)
    adaptive_total = common_total + (solver_calls - len(calls)) * 264_960 + gate_flops
    reported_budget = gate["exact_total_flop_budget"]

    losses = plan["losses_d0_d4"][:, 1:]
    adaptive = losses[np.arange(len(losses)), calls - 1]
    probabilities = np.asarray(gate["analytic_probabilities"], dtype=np.float64)
    analytic = np.einsum("nd,d->n", losses, probabilities, optimize=False)
    episode = plan["episode_id"].astype(np.int64)
    _, recomputed_replicates = analyze._cluster_ci(
        analytic - adaptive,
        episode,
        seed=int(
            common.study_json(common.STUDY_ROOT / "seed_manifest.json")[
                "statistical_seeds"
            ]["gate_exact_analytic"]
        ),
    )
    bootstrap = _load(common.STUDY_ROOT / "metrics/bootstrap_replicates.npz")

    episode_metrics = _load(common.STUDY_ROOT / "metrics/dataset_episode_metrics.npz")
    role_names = episode_metrics["role_names"].astype(str).tolist()
    plan_code = role_names.index("plan_oracle")
    metric_names = episode_metrics["metric_names"].astype(str).tolist()
    matrix = episode_metrics["metrics"][episode_metrics["role_code"] == plan_code]
    bands = common.study_json(common.STUDY_ROOT / "reference/reference_bands.json")
    core_names = bands["core_metrics"]
    center = np.asarray([bands["bands"][name]["offline_mean"] for name in core_names])
    se = np.asarray(
        [bands["bands"][name]["fresh_90_standard_error"] for name in core_names]
    )
    indices = [metric_names.index(name) for name in core_names]
    max_z = float(np.max(np.abs((matrix[:, indices].mean(0) - center) / np.maximum(se, 1e-15))))
    mapped = analyze.map_decision(
        bool(decision["validity_passed"]),
        bool(decision["distribution_matched"]),
        bool(decision["frozen_gate_prospective_discovery_passed"]),
    )

    expected_counts = {
        "offline_discovery": (420, 15_960),
        "offline_calibration": (90, 3_420),
        "plan_oracle": (90, 3_420),
        "markov_oracle": (30, 1_140),
        "v4_markov": (300, 11_400),
    }
    row_count_checks = {}
    for role, (episodes, rows) in expected_counts.items():
        values = _load(common.STUDY_ROOT / "data" / f"{role}_evaluation.npz")
        row_count_checks[role] = {
            "rows": len(values["episode_id"]),
            "episodes": len(np.unique(values["episode_id"])),
            "passed": len(values["episode_id"]) == rows
            and len(np.unique(values["episode_id"])) == episodes,
        }
    checks = {
        "d0_losses_exact": bool(np.array_equal(d0, plan["losses_d0_d4"][:, 0])),
        "calls_exact_from_scores": bool(np.array_equal(calls, plan["calls"])),
        "solver_total_calls_exact": solver_calls == reported_budget["adaptive_solver_total_calls"],
        "gate_decisions_exact": gate_decisions == reported_budget["adaptive_gate_decisions"],
        "gate_flops_exact": gate_flops == reported_budget["adaptive_gate_total_flops"],
        "fractional_calls_exact": abs(
            fractional_calls - reported_budget["analytic_target_total_calls"]
        )
        < 1e-12,
        "adaptive_total_flops_exact": adaptive_total == reported_budget["adaptive_total_flops"],
        "bootstrap_replicates_exact": bool(
            np.array_equal(
                recomputed_replicates,
                bootstrap["raw_vs_exact_total_flop_analytic"],
            )
        ),
        "joint_max_z_exact": abs(max_z - comparison["plan_joint_max_abs_z"]) < 1e-12,
        "decision_mapping_exact": mapped == decision["decision"],
        "all_row_counts_exact": all(item["passed"] for item in row_count_checks.values()),
        "frozen_hashes_exact": bool(common.verify_frozen_objects()),
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
        "row_count_checks": row_count_checks,
        "recomputed": {
            "d0_mean": float(d0.mean()),
            "total_calls": solver_calls,
            "gate_decisions": gate_decisions,
            "gate_flops": gate_flops,
            "fractional_exact_total_calls": fractional_calls,
            "adaptive_total_flops": adaptive_total,
            "joint_max_abs_z": max_z,
            "decision": mapped,
        },
    }
    if not passed:
        raise RuntimeError(f"independent recomputation failed: {checks}")
    common.write_study_json(output, result, exclusive=True)
    return result


def run_tests() -> dict[str, Any]:
    output = common.STUDY_ROOT / "test_logs.txt"
    command = [
        str(common.EVALUATION_PYTHON),
        "-m",
        "pytest",
        str(common.STUDY_ROOT / "tests"),
        "-q",
    ]
    completed = subprocess.run(
        command,
        cwd=common.REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    text = "$ " + " ".join(command) + "\n" + completed.stdout
    output.write_text(text, encoding="utf-8")
    result = {
        "command": command,
        "exit_code": completed.returncode,
        "log_path": str(output.relative_to(common.REPO_ROOT)),
        "log_sha256": common.sha256_file(output),
        "passed": completed.returncode == 0,
    }
    common.write_study_json(common.STUDY_ROOT / "audit/final_tests.json", result)
    if completed.returncode:
        raise RuntimeError(f"focused tests failed; see {output}")
    return result


def artifact_manifest() -> dict[str, Any]:
    output = common.STUDY_ROOT / "artifact_manifest.json"
    verification_path = common.STUDY_ROOT / "audit/artifact_manifest_verification.json"
    if output.exists() or verification_path.exists():
        raise RuntimeError("artifact manifest already exists")
    excluded = {
        output.resolve(),
        verification_path.resolve(),
    }
    files = [
        path
        for path in common.STUDY_ROOT.rglob("*")
        if path.is_file()
        and path.resolve() not in excluded
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
    ]
    records = [
        {
            "path": str(path.relative_to(common.REPO_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": common.sha256_file(path),
        }
        for path in sorted(files)
    ]
    payload = {
        "schema_version": 1,
        "scope": "all_study_files_except_this_manifest_and_its_followup_verification",
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "records": records,
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    common.write_study_json(output, payload, exclusive=True)
    mismatches = []
    for record in payload["records"]:
        path = common.REPO_ROOT / record["path"]
        observed = common.sha256_file(path)
        if observed != record["sha256"] or path.stat().st_size != record["bytes"]:
            mismatches.append(record["path"])
    verification = {
        "schema_version": 1,
        "passed": not mismatches,
        "manifest_path": str(output.relative_to(common.REPO_ROOT)),
        "manifest_sha256": common.sha256_file(output),
        "verified_file_count": len(records),
        "mismatches": mismatches,
    }
    common.write_study_json(verification_path, verification, exclusive=True)
    if mismatches:
        raise RuntimeError(f"artifact hash audit failed: {mismatches}")
    return verification


def run_all() -> dict[str, Any]:
    independent = independent_recomputation()
    tests = run_tests()
    verification = artifact_manifest()
    return {
        "independent_recomputation_passed": independent["passed"],
        "tests_passed": tests["passed"],
        "artifact_hash_audit_passed": verification["passed"],
        "artifact_manifest_sha256": verification["manifest_sha256"],
    }


if __name__ == "__main__":
    result = run_all()
    print(json.dumps(result, sort_keys=True))

