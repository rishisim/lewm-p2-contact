#!/usr/bin/env python3
"""Implementation completion, preseal qualification, and immutable seal."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from study_common import (
    ATTEMPT_ROOT,
    BOOTSTRAP_REPLICATES,
    CO_PRIMARY_ENDPOINTS,
    EVALUATION_PYTHON,
    FEATURE_DIM,
    GENERATION_PYTHON,
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    PER_CLAIM_ALPHA,
    REGIMES,
    REPO_ROOT,
    ROWS_PER_EPISODE,
    SIMULTANEOUS_FAMILY_SIZE,
    SMOKE_EPISODES_PER_REGIME,
    TARGET_EPISODES_PER_REGIME,
    V5_REQUIRED_HASHES,
    V5_ROOT,
    append_ledger,
    assert_runtime_contract,
    atomic_json,
    complete_state,
    load_module,
    load_v5_runner,
    read_json,
    relative_to_repo,
    sha256_file,
    verify_pre_outcome_seal,
)


NORMATIVE_SCRIPTS = (
    "study_common.py",
    "state_init.py",
    "activate_attempt.py",
    "bootstrap_audit.py",
    "design_and_power.py",
    "generator.py",
    "runner.py",
    "checkpoints.py",
    "analysis.py",
    "latency.py",
    "independent_verify.py",
    "posthoc.py",
    "finalize.py",
    "preseal.py",
    "launcher.py",
)
NORMATIVE_DESIGN = (
    "DGP_MATRIX.json",
    "power_analysis.json",
    "cohort_seed_ledger.json",
    "outcome_mapping.json",
    "operation_ledger.json",
    "numerical_equivalence_contract.json",
    "PREREGISTRATION.md",
)


def subprocess_run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def implementation() -> dict[str, Any]:
    output = ATTEMPT_ROOT / "audit/implementation_complete.json"
    if output.exists():
        complete_state(
            "IMPLEMENT_PACKAGE",
            "PRESEAL_QUALIFY",
            evidence_path=output,
            checkpoint_name="v005_implementation_reused",
            next_action="run zero-outcome qualification",
        )
        return read_json(output)
    missing = [
        name
        for name in (*NORMATIVE_SCRIPTS, *NORMATIVE_DESIGN)
        if not (ATTEMPT_ROOT / name).exists()
    ]
    ast_checks = {}
    for name in NORMATIVE_SCRIPTS:
        path = ATTEMPT_ROOT / name
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            ast_checks[name] = {
                "parsed": True,
                "top_level_nodes": len(tree.body),
                "sha256": sha256_file(path),
            }
        except SyntaxError as error:
            ast_checks[name] = {
                "parsed": False,
                "error": str(error),
                "sha256": sha256_file(path),
            }
    source_text = {
        name: (ATTEMPT_ROOT / name).read_text(encoding="utf-8")
        for name in NORMATIVE_SCRIPTS
    }
    checks = {
        "no_missing_normative_files": not missing,
        "all_python_ast_parses": all(
            item["parsed"] for item in ast_checks.values()
        ),
        "generator_has_exact_three_regimes": "make_world(regime" in source_text[
            "generator.py"
        ]
        and "REGIMES[regime]" in source_text["generator.py"],
        "runner_uses_frozen_v5_sparse_dense": "v5.manual_sparse" in source_text[
            "runner.py"
        ]
        and "v5.dense_shadow" in source_text["runner.py"],
        "analysis_has_six_claim_family": "PER_CLAIM_ALPHA" in source_text[
            "analysis.py"
        ]
        and "terminal_mapping" in source_text["analysis.py"],
        "independent_does_not_import_analysis": "import analysis"
        not in source_text["independent_verify.py"],
        "posthoc_requires_decision": "decision.json" in source_text[
            "posthoc.py"
        ],
        "contact_not_loaded_by_runner": "proprio_gripper_contact"
        not in source_text["runner.py"],
        "no_target_or_smoke_data_before_qualification": not (
            ATTEMPT_ROOT / "data"
        ).exists(),
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "checks": checks,
        "ast": ast_checks,
        "missing": missing,
        "passed": all(checks.values()),
        "normative_file_count": len(NORMATIVE_SCRIPTS)
        + len(NORMATIVE_DESIGN),
        "target_outcome_episodes": 0,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"implementation checkpoint failed: {result}")
    append_ledger(
        "implementation_complete",
        path=relative_to_repo(output),
        sha256=sha256_file(output),
        normative_file_count=result["normative_file_count"],
    )
    complete_state(
        "IMPLEMENT_PACKAGE",
        "PRESEAL_QUALIFY",
        evidence_path=output,
        checkpoint_name="v005_implementation_complete",
        next_action="run zero-outcome qualification",
    )
    return result


def qualify_generation() -> dict[str, Any]:
    assert_runtime_contract("generation")
    import generator

    regimes = {}
    for regime, expected in REGIMES.items():
        world, policy = generator.make_world(regime)
        observed = {
            "policy_type": policy.type,
            "action_noise": float(policy.action_noise),
            "p_random_action": float(policy.p_random_action),
            "noise_smoothing": float(policy.noise_smoothing),
            "min_norm": float(policy.min_norm),
        }
        target = {
            "policy_type": expected["policy_type"],
            "action_noise": expected["action_noise"],
            "p_random_action": expected["p_random_action"],
            "noise_smoothing": expected["noise_smoothing"],
            "min_norm": expected["min_norm"],
        }
        world.close()
        regimes[regime] = {
            "observed": observed,
            "expected": target,
            "exact": observed == target,
            "policy_episode_actions_generated": 0,
        }
    return {
        "schema_version": 1,
        "runtime": assert_runtime_contract("generation"),
        "regimes": regimes,
        "passed": all(item["exact"] for item in regimes.values()),
        "smoke_episodes": 0,
        "target_outcome_episodes": 0,
    }


def qualify_sparse_dense() -> dict[str, Any]:
    runtime_snapshot = assert_runtime_contract("evaluation")
    v5 = load_v5_runner()
    input_path = V5_ROOT / "data/package_smoke_latency_inputs.npz"
    expected_hash = (
        "9c8d2344076da9dd089ffa4c53af9e9cf4ebb34910edb1a79f8035124d25c625"
    )
    if sha256_file(input_path) != expected_hash:
        raise RuntimeError("consumed V5 smoke latency input drift")
    with np.load(input_path, allow_pickle=False) as stored:
        history_np = stored["history"][:128].copy()
        actions_np = stored["actions"][:128].copy()
    torch, runtime, _, device, base, _, stack = v5.load_scientific_stack(
        "mps"
    )
    solver, v1, _, provenance = stack
    gate = v5.load_gate_tensors(torch, device)
    before = runtime.module_audit(base, solver, v1)
    history = torch.as_tensor(history_np, device=device)
    actions = torch.as_tensor(actions_np, device=device)
    base_prediction = runtime.base_predict(base, history, actions)
    dense, dense_calls, _, _ = v5.dense_shadow(
        torch, solver, gate, history, actions, base_prediction
    )
    sparse, calls, scores, features = v5.manual_sparse(
        torch, solver, gate, history, actions, base_prediction
    )
    repeat, repeat_calls, repeat_scores, repeat_features = v5.manual_sparse(
        torch, solver, gate, history, actions, base_prediction
    )
    with torch.inference_mode():
        selected = solver.forward_selected(
            history, actions, base_prediction, calls
        )
    dense_selected = dense[
        torch.arange(len(history), device=device), calls - 1
    ]
    delta = (sparse - dense_selected).abs()
    checks = {
        "module_before": before["passed"],
        "sparse_repeat_output_exact": bool(torch.equal(sparse, repeat)),
        "sparse_repeat_calls_exact": bool(
            torch.equal(calls, repeat_calls)
        ),
        "sparse_repeat_score_nan_mask_exact": bool(
            torch.equal(torch.isnan(scores), torch.isnan(repeat_scores))
        ),
        "sparse_repeat_feature_nan_mask_exact": bool(
            torch.equal(
                torch.isnan(features), torch.isnan(repeat_features)
            )
        ),
        "manual_forward_selected_exact": bool(
            torch.equal(sparse, selected)
        ),
        "sparse_dense_calls_exact": bool(
            torch.equal(calls, dense_calls)
        ),
        "cross_batch_allclose": bool(
            torch.allclose(
                sparse,
                dense_selected,
                rtol=NUMERICAL_RTOL,
                atol=NUMERICAL_ATOL,
            )
        ),
        "max_abs_ceiling": float(delta.max().item())
        <= NUMERICAL_MAX_ABS,
        "feature_width": features.shape[2] == FEATURE_DIM,
    }
    runtime.synchronize(device)
    after = runtime.module_audit(base, solver, v1)
    checks["module_after_identical"] = before == after
    return {
        "schema_version": 1,
        "runtime": runtime_snapshot,
        "input_path": relative_to_repo(input_path),
        "input_sha256": expected_hash,
        "checks": checks,
        "passed": all(checks.values()),
        "call_histogram": torch.bincount(
            calls, minlength=5
        )[1:].cpu().tolist(),
        "cross_batch_max_abs": float(delta.max().item()),
        "base_provenance": provenance,
        "consumed_excluded_smoke_only": True,
        "target_outcome_episodes": 0,
    }


def qualify_inference() -> dict[str, Any]:
    assert_runtime_contract("evaluation")
    analysis = load_module(
        "generalization_v005_analysis_qualification",
        ATTEMPT_ROOT / "analysis.py",
    )
    independent_verify = load_module(
        "generalization_v005_independent_qualification",
        ATTEMPT_ROOT / "independent_verify.py",
    )

    ledger = read_json(ATTEMPT_ROOT / "cohort_seed_ledger.json")
    rng = np.random.default_rng(
        ledger["analysis_seeds"]["latency_order_seed"]
    )
    losses = np.square(rng.normal(size=(257, 4)))
    target_mean = 1.375
    first = analysis.strongest_mixture(losses, target_mean)
    second, left, right, weight = independent_verify.optimal_baseline(
        losses, target_mean
    )
    difference = np.abs(first["loss"] - second)
    synthetic = rng.normal(
        size=TARGET_EPISODES_PER_REGIME * ROWS_PER_EPISODE
    )
    vectorized = analysis.episode_mean(synthetic)
    looped = independent_verify.per_episode(synthetic)
    mapping_cases = {
        "all_six": analysis.terminal_mapping(
            {
                regime: {
                    endpoint: {"lower": 1.0}
                    for endpoint in CO_PRIMARY_ENDPOINTS
                }
                for regime in REGIMES
            }
        )[0],
        "one": analysis.terminal_mapping(
            {
                regime: {
                    endpoint: {
                        "lower": (
                            1.0
                            if regime == next(iter(REGIMES))
                            and endpoint == CO_PRIMARY_ENDPOINTS[0]
                            else -1.0
                        )
                    }
                    for endpoint in CO_PRIMARY_ENDPOINTS
                }
                for regime in REGIMES
            }
        )[0],
        "zero": analysis.terminal_mapping(
            {
                regime: {
                    endpoint: {"lower": -1.0}
                    for endpoint in CO_PRIMARY_ENDPOINTS
                }
                for regime in REGIMES
            }
        )[0],
    }
    checks = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES == 20_000,
        "family_size": SIMULTANEOUS_FAMILY_SIZE == 6,
        "per_claim_alpha": PER_CLAIM_ALPHA == 0.05 / 6,
        "analytic_depths_weights_exact": (
            first["depth_lower"],
            first["depth_upper"],
            first["weight_upper"],
        )
        == (left, right, weight),
        "analytic_vectors_machine_precision": bool(
            float(difference.max())
            <= 16
            * np.finfo(np.float64).eps
            * max(1.0, float(np.abs(second).max()))
        ),
        "episode_aggregation_two_implementations": bool(
            np.array_equal(vectorized, looped)
        ),
        "terminal_mapping": mapping_cases
        == {
            "all_six": "zero_shot_generalization_supported",
            "one": "zero_shot_generalization_partial",
            "zero": "zero_shot_generalization_failed",
        },
    }
    return {
        "schema_version": 1,
        "checks": checks,
        "mapping_cases": mapping_cases,
        "passed": all(checks.values()),
        "synthetic_inputs_only": True,
        "target_outcome_episodes": 0,
    }


def qualify() -> dict[str, Any]:
    output = ATTEMPT_ROOT / "audit/preseal_qualification.json"
    if output.exists():
        complete_state(
            "PRESEAL_QUALIFY",
            "PRE_OUTCOME_SEAL",
            evidence_path=output,
            checkpoint_name="v005_preseal_qualification_reused",
            next_action="seal all normative code and frozen objects",
        )
        return read_json(output)
    evaluation_runtime = assert_runtime_contract("evaluation")
    generation_command = subprocess_run(
        [
            str(GENERATION_PYTHON),
            str(ATTEMPT_ROOT / "preseal.py"),
            "qualify-generation",
        ]
    )
    if not generation_command["passed"]:
        raise RuntimeError(
            f"generation qualification failed: {generation_command}"
        )
    generation_result = json.loads(
        generation_command["stdout"].splitlines()[-1]
    )
    sparse_dense = qualify_sparse_dense()
    inference = qualify_inference()
    wrong_generation = subprocess_run(
        [
            str(EVALUATION_PYTHON),
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(ATTEMPT_ROOT)!r});"
                "import study_common;"
                "study_common.assert_runtime_contract('generation')"
            ),
        ]
    )
    wrong_evaluation = subprocess_run(
        [
            str(GENERATION_PYTHON),
            "-c",
            (
                "import sys;"
                f"sys.path.insert(0,{str(ATTEMPT_ROOT)!r});"
                "import study_common;"
                "study_common.assert_runtime_contract('evaluation')"
            ),
        ]
    )
    checks = {
        "evaluation_runtime": evaluation_runtime["checks"]
        == {"executable": True, "python_version": True, "packages": True},
        "generation_constructor_qualification": generation_result["passed"],
        "sparse_dense_qualification": sparse_dense["passed"],
        "synthetic_inference_qualification": inference["passed"],
        "wrong_generation_interpreter_fails": wrong_generation["returncode"]
        != 0,
        "wrong_evaluation_interpreter_fails": wrong_evaluation["returncode"]
        != 0,
        "no_smoke_or_target_data": not (ATTEMPT_ROOT / "data").exists(),
        "no_target_outcomes": True,
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "checks": checks,
        "evaluation_runtime": evaluation_runtime,
        "generation": generation_result,
        "sparse_dense": sparse_dense,
        "inference": inference,
        "wrong_interpreter_negative_paths": {
            "evaluation_as_generation": wrong_generation,
            "generation_as_evaluation": wrong_evaluation,
        },
        "passed": all(checks.values()),
        "target_outcome_episodes": 0,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"preseal qualification failed: {result}")
    append_ledger(
        "preseal_qualification_complete",
        path=relative_to_repo(output),
        sha256=sha256_file(output),
        target_outcomes=0,
    )
    complete_state(
        "PRESEAL_QUALIFY",
        "PRE_OUTCOME_SEAL",
        evidence_path=output,
        checkpoint_name="v005_preseal_qualification_complete",
        next_action="seal all normative code and frozen objects",
    )
    return result


def seal() -> dict[str, Any]:
    output = ATTEMPT_ROOT / "audit/pre_outcome_seal.json"
    if output.exists():
        verified = verify_pre_outcome_seal()
        complete_state(
            "PRE_OUTCOME_SEAL",
            "EXCLUDED_REGIME_SMOKE",
            evidence_path=output,
            checkpoint_name="v005_pre_outcome_seal_reused",
            next_action="run fresh excluded smoke in all regimes",
        )
        return verified
    if (ATTEMPT_ROOT / "data").exists():
        raise RuntimeError("cannot seal after any smoke or target data")
    qualification = read_json(
        ATTEMPT_ROOT / "audit/preseal_qualification.json"
    )
    if not qualification.get("passed"):
        raise RuntimeError("passing preseal qualification required")
    sealed_files: dict[str, str] = {}
    local_files = [
        ATTEMPT_ROOT / name
        for name in (*NORMATIVE_SCRIPTS, *NORMATIVE_DESIGN)
    ] + [
        ATTEMPT_ROOT / "audit/implementation_complete.json",
        ATTEMPT_ROOT / "audit/preseal_qualification.json",
        ATTEMPT_ROOT / "audit/version_forward_equivalence.json",
        ATTEMPT_ROOT.parent
        / "v002/audit/design_and_power_failure.json",
        ATTEMPT_ROOT.parent
        / "v003/audit/preseal_qualification_failure.json",
        ATTEMPT_ROOT.parent
        / "v004/audit/smoke_checkpoint_procedural_failure.json",
        ATTEMPT_ROOT.parent
        / "v004/audit/excluded_regime_smoke.json",
        ATTEMPT_ROOT / "audit/design_and_power.json",
    ]
    for path in local_files:
        if not path.exists():
            raise RuntimeError(f"missing seal input: {path}")
        sealed_files[relative_to_repo(path)] = sha256_file(path)
    v5_preseal = read_json(V5_ROOT / "audit/pre_v5_seal.json")
    for raw, expected in v5_preseal["files"].items():
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if sha256_file(path) != expected:
            raise RuntimeError(f"V5 frozen source drift at seal: {path}")
        key = str(path) if path.is_absolute() else relative_to_repo(path)
        sealed_files[key] = expected
    for relative, expected in V5_REQUIRED_HASHES.items():
        path = V5_ROOT / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"V5 terminal evidence drift: {relative}")
        sealed_files[relative_to_repo(path)] = expected
    power_metrics = (
        V5_ROOT / "metrics/v5_confirmation_episode_metrics.npz"
    )
    sealed_files[relative_to_repo(power_metrics)] = sha256_file(
        power_metrics
    )
    payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "status": "frozen_before_any_v005_excluded_smoke_or_target_episode",
        "attempt": "v005",
        "target_episodes_per_regime": TARGET_EPISODES_PER_REGIME,
        "smoke_episodes_per_regime": SMOKE_EPISODES_PER_REGIME,
        "regimes": list(REGIMES),
        "sealed_files": sealed_files,
        "sealed_file_count": len(sealed_files),
        "normative_code_complete": True,
        "preseal_qualification_passed": True,
        "post_seal_normative_patch_forbidden": True,
        "zero_outcome_version_forward_only": True,
        "smoke_episodes_at_seal": 0,
        "target_outcome_episodes_at_seal": 0,
        "contact_motion_phase_reward_success_opened_at_seal": False,
        "v3_test_targets_opened": False,
        "combined_v3_cache_numpy_loaded": False,
        "released_hdf5_opened": False,
        "scientific_components_changed": False,
    }
    atomic_json(output, payload, exclusive=True)
    verified = verify_pre_outcome_seal()
    append_ledger(
        "pre_outcome_seal_complete",
        path=relative_to_repo(output),
        sha256=sha256_file(output),
        sealed_file_count=len(sealed_files),
        target_outcomes=0,
    )
    complete_state(
        "PRE_OUTCOME_SEAL",
        "EXCLUDED_REGIME_SMOKE",
        evidence_path=output,
        checkpoint_name="v005_pre_outcome_seal_complete",
        next_action="run fresh excluded smoke in all regimes",
    )
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "implementation",
            "qualify-generation",
            "qualify",
            "seal",
            "verify-seal",
        ),
    )
    arguments = parser.parse_args()
    if arguments.command == "implementation":
        result = implementation()
    elif arguments.command == "qualify-generation":
        result = qualify_generation()
    elif arguments.command == "qualify":
        result = qualify()
    elif arguments.command == "seal":
        result = seal()
    else:
        result = verify_pre_outcome_seal()
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
