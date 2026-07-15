#!/usr/bin/env python3
"""Run repair unit tests and the repository's selected-execution standards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from counted_features import (
    build_counted_causal_features,
    executable_operation_graph,
    semantic_feature_names,
)
from cycle_common import ACTION_DIM, DISCOVERY_CODE, HISTORY_LEN, LATENT_DIM, REPO_ROOT, ROOT, atomic_json


def main() -> None:
    output_path = ROOT / "audit/preseal_tests.json"
    if output_path.exists():
        raise RuntimeError("preseal test artifact already exists")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "runs/lewm_adaptive_compute_discovery/tests/test_models.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "runs/lewm_adaptive_compute_distribution_contract/tests/test_contract.py",
        ],
    ]
    subprocess_results = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess_results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "passed": completed.returncode == 0,
            }
        )

    if str(DISCOVERY_CODE) not in sys.path:
        sys.path.insert(0, str(DISCOVERY_CODE))
    import models

    torch.manual_seed(260715)
    history = torch.randn(17, HISTORY_LEN, LATENT_DIM)
    actions = torch.randn(17, HISTORY_LEN, ACTION_DIM)
    current = torch.randn(17, LATENT_DIM)
    update = torch.randn(17, LATENT_DIM)
    expected = models.build_causal_features(history, actions, current, update)
    actual = build_counted_causal_features(history, actions, current, update)
    semantic = semantic_feature_names(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, history_len=HISTORY_LEN
    )
    internal_checks = {
        "cpu_feature_shape": tuple(actual.shape) == (17, 1046),
        "cpu_feature_order_values_bitwise_exact": bool(torch.equal(actual, expected)),
        "feature_names_unique_and_complete": len(semantic) == 1046 and len(set(semantic)) == 1046,
        "gradient_boundary_detached": not actual.requires_grad,
        "operation_graph_feature_flops": executable_operation_graph()["totals"][
            "feature_flops_per_reached_evaluation"
        ]
        == 3801,
        "operation_graph_gate_score_flops": executable_operation_graph()["totals"][
            "dual_affine_score_flops_per_reached_evaluation"
        ]
        == 4184,
        "operation_graph_total_flops": executable_operation_graph()["totals"][
            "total_flops_per_reached_evaluation"
        ]
        == 7985,
        "operation_graph_nonflops": executable_operation_graph()["totals"][
            "nonflop_comparison_min_per_reached_evaluation"
        ]
        == 5,
    }
    if torch.backends.mps.is_available():
        mps_inputs = [value.to("mps") for value in (history, actions, current, update)]
        mps_expected = models.build_causal_features(*mps_inputs)
        mps_actual = build_counted_causal_features(*mps_inputs)
        internal_checks["mps_feature_values_bitwise_exact"] = bool(
            torch.equal(mps_actual, mps_expected)
        )
        internal_checks["mps_feature_values_within_repository_tolerance"] = bool(
            torch.allclose(mps_actual, mps_expected, rtol=2e-6, atol=2e-7)
        )
    result = {
        "schema_version": 1,
        "subprocess_tests": subprocess_results,
        "internal_checks": internal_checks,
        "passed": all(item["passed"] for item in subprocess_results)
        and all(internal_checks.values()),
        "consumed_or_synthetic_inputs_only": True,
        "prospective_episodes": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output_path, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"preseal tests failed: {result}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
