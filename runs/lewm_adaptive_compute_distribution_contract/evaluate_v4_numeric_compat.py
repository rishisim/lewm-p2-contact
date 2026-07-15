#!/usr/bin/env python3
"""Recompute V4 with explicit legacy and study-wide raw-loss arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import common
import evaluate
import evaluate_v4_compat


NOTE = common.STUDY_ROOT / "audit/v4_raw_loss_arithmetic_compatibility.json"
SEAL = common.STUDY_ROOT / "audit/v4_raw_loss_arithmetic_compatibility_seal.json"
OUTPUT = common.STUDY_ROOT / "data/v4_markov_evaluation.npz"
MANIFEST = common.STUDY_ROOT / "data/v4_markov_evaluation_manifest.json"


def seal() -> dict[str, Any]:
    common.assert_pre_generation_seal()
    evaluate_v4_compat.assert_seal()
    if OUTPUT.exists() or MANIFEST.exists() or NOTE.exists() or SEAL.exists():
        raise RuntimeError("V4 numeric compatibility artifacts or outputs already exist")
    prior_path = common.V4_ROOT / "data/confirmation_outcome_once.npz"
    payload = {
        "schema_version": 1,
        "status": "frozen_before_materializing_v4_diagnostic_recomputation",
        "diagnosis": (
            "V4 evaluate.py line 259 subtracts and squares float32 exits/targets, "
            "then casts the mean to float64; the distribution-contract evaluator "
            "casts exits to float64 before subtraction and squaring"
        ),
        "failed_first_recomputation_materialized_output": False,
        "read_only_diagnostic": {
            "calls_exact": True,
            "score_max_abs": 0.0,
            "whitened_max_abs": 0.0,
            "canonical_double_vs_v4_raw_loss_max_abs": 2.774697009932936e-7,
        },
        "fixed_audit_rule": (
            "retain canonical float64-residual raw losses for cross-role metrics; also "
            "recompute V4's float32-residual legacy losses and require exact d1-d4 "
            "equality, exact whitened losses, exact scores, and exact calls"
        ),
        "v4_outcome_sha256": common.sha256_file(prior_path),
        "v4_evaluator_sha256": common.sha256_file(common.V4_ROOT / "evaluate.py"),
        "thresholds_models_gate_or_stopping_rule_changed": False,
        "v4_used_for_training_selection_tuning_or_thresholds": False,
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    common.write_study_json(NOTE, payload, exclusive=True)
    paths = (
        Path(__file__).resolve(),
        NOTE,
        evaluate_v4_compat.SEAL,
        common.V4_ROOT / "evaluate.py",
    )
    result = {
        "schema_version": 1,
        "status": "frozen_before_materializing_v4_diagnostic_recomputation",
        "sealed_files": {
            str(path.relative_to(common.REPO_ROOT)): common.sha256_file(path)
            for path in paths
        },
        "v4_outcome_sha256": common.sha256_file(prior_path),
        "v3_test_targets_opened": False,
    }
    common.write_study_json(SEAL, result, exclusive=True)
    return result


def assert_seal() -> dict[str, Any]:
    result = common.study_json(SEAL)
    for relative, expected in result["sealed_files"].items():
        if common.sha256_file(common.REPO_ROOT / relative) != expected:
            raise RuntimeError(f"V4 numeric compatibility seal drift: {relative}")
    prior = common.V4_ROOT / "data/confirmation_outcome_once.npz"
    if common.sha256_file(prior) != result["v4_outcome_sha256"]:
        raise RuntimeError("V4 outcome drift")
    return result


@torch.inference_mode()
def run(device_name: str) -> dict[str, Any]:
    assert_seal()
    if OUTPUT.exists() or MANIFEST.exists():
        raise RuntimeError("V4 diagnostic evaluation already exists")
    arrays, input_manifest = evaluate_v4_compat.load_v4("v4_markov")
    runtime = common.load_runtime()
    device = runtime.choose_device(device_name)
    solver, v1, models = runtime.load_solver(device)
    gate = runtime.load_gate(device)
    whitening = runtime.load_whitening()
    before = runtime.module_audit(solver, v1, gate.model)
    n = len(arrays["target"])
    losses = np.empty((n, 5), dtype=np.float64)
    legacy_losses = np.empty((n, 5), dtype=np.float64)
    white = np.empty((n, 5), dtype=np.float64)
    scores = np.empty((n, 3), dtype=np.float64)
    calls = np.empty(n, dtype=np.int64)
    updates_out = np.empty((n, 4), dtype=np.float32)
    shape = (n, 3, len(common.FEATURE_BLOCKS), len(common.BLOCK_STAT_NAMES))
    raw_stats = np.empty(shape, dtype=np.float32)
    z_stats = np.empty(shape, dtype=np.float32)
    raw_digest = hashlib.sha256()
    z_digest = hashlib.sha256()
    d0_identity = d0_bitwise = d1_bitwise = True
    sparse_call_exact = True
    sparse_output_max = 0.0
    started = time.perf_counter()
    for batch_index, part in enumerate(evaluate._iter(n, 1024)):
        history = torch.as_tensor(
            np.ascontiguousarray(arrays["history"][part]), device=device
        )
        action = torch.as_tensor(
            np.ascontiguousarray(arrays["action"][part]), device=device
        )
        base = torch.as_tensor(
            np.ascontiguousarray(arrays["base_pred"][part]), device=device
        )
        target = np.asarray(arrays["target"][part], dtype=np.float32)
        outputs, updates = solver(
            history, action, base, max_depth=4, return_updates=True
        )
        expected = v1(history, action, base, depths=(0, 1))
        d0_identity &= outputs[0] is base
        d0_bitwise &= bool(torch.equal(outputs[0], expected[0]))
        d1_bitwise &= bool(torch.equal(outputs[1], expected[1]))
        exits = torch.stack([outputs[depth] for depth in range(5)], 1).cpu().numpy()
        # Study-wide convention, shared with all four other roles.
        losses[part] = np.square(
            exits.astype(np.float64) - target[:, None, :]
        ).mean(2)
        # Preserved V4 convention from runs/lewm_adaptive_compute_v4/evaluate.py:259.
        legacy_losses[part] = np.square(exits - target[:, None, :]).mean(2).astype(
            np.float64
        )
        white[part] = evaluate._whitened(target, exits, whitening["matrix"])
        for depth in range(1, 5):
            updates_out[part, depth - 1] = torch.linalg.vector_norm(
                updates[depth], dim=1
            ).cpu().numpy()
        local_scores = []
        for stage, depth in enumerate((1, 2, 3)):
            score, causal = runtime.gate_score(
                gate, models, history, action, outputs[depth], updates[depth], stage
            )
            score_np = score.cpu().numpy().astype(np.float64)
            causal_np = causal.cpu().numpy().astype(np.float32)
            one_hot = np.zeros((len(part), 3), dtype=np.float32)
            one_hot[:, stage] = 1.0
            encoded = np.concatenate([causal_np, one_hot], axis=1)
            normalized = (
                encoded - gate.feature_mean.detach().cpu().numpy()
            ) / gate.feature_std.detach().cpu().numpy()
            raw_digest.update(np.ascontiguousarray(encoded).tobytes())
            z_digest.update(np.ascontiguousarray(normalized).tobytes())
            for block_index, (_, begin, end) in enumerate(common.FEATURE_BLOCKS):
                raw_stats[part, stage, block_index] = evaluate._block_stats(
                    encoded[:, begin:end]
                )
                z_stats[part, stage, block_index] = evaluate._block_stats(
                    normalized[:, begin:end]
                )
            local_scores.append(score_np)
        score_batch = np.stack(local_scores, axis=1)
        scores[part] = score_batch
        call_batch = runtime.sequential_calls_numpy(
            score_batch, common.COMPUTE_PRICE
        )
        calls[part] = call_batch
        reference, reference_calls = runtime.sparse_adaptive_reference(
            solver, gate, models, history, action, base
        )
        optimized, optimized_calls = runtime.sparse_adaptive_optimized(
            solver, gate, models, history, action, base
        )
        dense = exits[np.arange(len(part)), call_batch]
        sparse_call_exact &= bool(
            np.array_equal(call_batch, reference_calls.cpu().numpy())
            and np.array_equal(call_batch, optimized_calls.cpu().numpy())
        )
        sparse_output_max = max(
            sparse_output_max,
            float(np.max(np.abs(dense - reference.cpu().numpy()))),
            float(np.max(np.abs(dense - optimized.cpu().numpy()))),
        )
        if (batch_index + 1) % 5 == 0 or part[-1] == n - 1:
            print(f"evaluated v4_markov rows {int(part[-1]) + 1}/{n}", flush=True)
    runtime.synchronize(device)
    after = runtime.module_audit(solver, v1, gate.model)
    with np.load(
        common.V4_ROOT / "data/confirmation_outcome_once.npz", allow_pickle=False
    ) as stored:
        prior = {name: stored[name].copy() for name in stored.files}
    recomputation = {
        "calls_exact": bool(np.array_equal(calls, prior["calls"])),
        "scores_exact": bool(np.array_equal(scores, prior["scores"])),
        "scores_max_abs": float(np.max(np.abs(scores - prior["scores"]))),
        "legacy_d1_d4_losses_exact": bool(
            np.array_equal(legacy_losses[:, 1:], prior["losses"])
        ),
        "legacy_d1_d4_losses_max_abs": float(
            np.max(np.abs(legacy_losses[:, 1:] - prior["losses"]))
        ),
        "whitened_d1_d4_exact": bool(
            np.array_equal(white[:, 1:], prior["whitened_losses"])
        ),
        "whitened_d1_d4_max_abs": float(
            np.max(np.abs(white[:, 1:] - prior["whitened_losses"]))
        ),
        "canonical_double_vs_legacy_raw_max_abs": float(
            np.max(np.abs(losses - legacy_losses))
        ),
        "prior_outcome_sha256": common.sha256_file(
            common.V4_ROOT / "data/confirmation_outcome_once.npz"
        ),
        "canonical_cross_role_loss_convention": "float64 residual then square",
        "preserved_v4_legacy_loss_convention": "float32 residual and square then cast",
        "diagnostic_only": True,
    }
    recomputation["passed"] = bool(
        recomputation["calls_exact"]
        and recomputation["scores_exact"]
        and recomputation["legacy_d1_d4_losses_exact"]
        and recomputation["whitened_d1_d4_exact"]
    )
    checks = {
        "d0_identity": d0_identity,
        "d0_bitwise_v1": d0_bitwise,
        "d1_bitwise_v1": d1_bitwise,
        "sparse_calls_exact": sparse_call_exact,
        "sparse_outputs_equivalent": sparse_output_max <= 2e-5,
        "module_state_unchanged": before == after,
        "all_modules_frozen_no_gradients": bool(before["passed"] and after["passed"]),
        "causal_feature_signature_excludes_target": (
            "target" not in models.build_causal_features.__code__.co_varnames
        ),
        "supported_calls_only": bool(np.isin(calls, common.SUPPORTED_CALLS).all()),
        "v4_recompute_exact_under_preserved_arithmetic": recomputation["passed"],
        "v3_test_targets_opened": False,
    }
    passed = all(
        value for key, value in checks.items() if key != "v3_test_targets_opened"
    )
    if not passed:
        raise RuntimeError(f"V4 numeric compatibility evaluation failed: {checks}")
    arrays_out = {
        "losses_d0_d4": losses,
        "legacy_v4_losses_d0_d4": legacy_losses,
        "whitened_losses_d0_d4": white,
        "marginal_gains_d0_d4": losses[:, :-1] - losses[:, 1:],
        "scores": scores,
        "calls": calls,
        "update_norms_d1_d4": updates_out,
        "raw_feature_block_stats": raw_stats,
        "normalized_feature_block_stats": z_stats,
        "episode_id": np.asarray(arrays["episode_id"], dtype=np.int64),
        "model_step": np.asarray(arrays["model_step"], dtype=np.int64),
    }
    for name in (
        "interaction",
        "impact",
        "effector_disp",
        "block_disp",
        "action_magnitude",
        "normalized_phase",
    ):
        if name in arrays:
            arrays_out[name] = np.asarray(arrays[name])
    common.atomic_npz(OUTPUT, arrays_out)
    result = {
        **common.dataset_manifest_base(role="v4_markov", episodes=300, rows=n),
        "status": "complete_frozen_evaluation_with_explicit_v4_numeric_compatibility",
        "evaluation_path": str(OUTPUT.relative_to(common.REPO_ROOT)),
        "evaluation_sha256": common.sha256_file(OUTPUT),
        "input_manifest": input_manifest,
        "checks": checks,
        "passed": passed,
        "sparse_output_max_abs": sparse_output_max,
        "raw_feature_stream_sha256": raw_digest.hexdigest(),
        "normalized_feature_stream_sha256": z_digest.hexdigest(),
        "feature_blocks": [item[0] for item in common.FEATURE_BLOCKS],
        "feature_block_stats": list(common.BLOCK_STAT_NAMES),
        "gate_parameters": common.GATE_PARAMETERS,
        "gate_flops_per_evaluated_decision": common.GATE_FLOPS_PER_DECISION,
        "compute_price": common.COMPUTE_PRICE,
        "module_before": before,
        "module_after": after,
        "elapsed_seconds": time.perf_counter() - started,
        "v4_recomputation": recomputation,
        "v4_outcomes_used_for_thresholds_or_selection": False,
        "numeric_compatibility_sha256": common.sha256_file(NOTE),
    }
    common.write_study_json(MANIFEST, result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "run"))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    result = seal() if args.command == "seal" else run(args.device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
