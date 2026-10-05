#!/usr/bin/env python3
"""Actual sparse execution with an independently accounted dense shadow."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from study_common import (
    ATTEMPT_ROOT,
    FEATURE_DIM,
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    REGIMES,
    REPO_ROOT,
    ROWS_PER_EPISODE,
    append_ledger,
    assert_runtime_contract,
    atomic_json,
    atomic_npz,
    execution_directory,
    execution_manifest_path,
    latency_input_path,
    load_v5_runner,
    raw_manifest_path,
    read_json,
    relative_to_repo,
    role_count,
    sha256_file,
    update_state_fields,
    verify_pre_outcome_seal,
)


def existing_manifest(
    phase: str, regime: str, expected: int
) -> dict[str, Any] | None:
    path = execution_manifest_path(phase, regime)
    if not path.exists():
        return None
    manifest = read_json(path)
    if (
        not manifest.get("complete")
        or manifest.get("episode_count") != expected
        or manifest.get("phase") != phase
        or manifest.get("regime") != regime
    ):
        raise RuntimeError(f"incomplete execution manifest: {path}")
    for record in manifest["episodes"]:
        part = REPO_ROOT / record["path"]
        if sha256_file(part) != record["sha256"]:
            raise RuntimeError(f"execution part drift: {part}")
    return manifest


def executed_target_count() -> int:
    total = 0
    for regime in REGIMES:
        directory = execution_directory("target", regime)
        if directory.exists():
            total += len(list(directory.glob("*.json")))
    return total


def exact_with_nan(torch: Any, left: Any, right: Any) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if not torch.is_floating_point(left):
        return bool(torch.equal(left, right))
    left_nan = torch.isnan(left)
    right_nan = torch.isnan(right)
    return bool(
        torch.equal(left_nan, right_nan)
        and torch.equal(left[~left_nan], right[~right_nan])
    )


def execute(phase: str, regime: str, device_name: str) -> dict[str, Any]:
    if phase not in ("smoke", "target") or regime not in REGIMES:
        raise RuntimeError("invalid execution role")
    runtime_snapshot = assert_runtime_contract("evaluation")
    seal = verify_pre_outcome_seal()
    if phase == "target":
        smoke = read_json(ATTEMPT_ROOT / "audit/excluded_regime_smoke.json")
        if not smoke.get("passed"):
            raise RuntimeError("target execution requires passing excluded smoke")
    expected = role_count(phase)
    prior = existing_manifest(phase, regime, expected)
    if prior is not None:
        return prior
    raw_manifest = read_json(raw_manifest_path(phase, regime))
    if (
        not raw_manifest.get("complete")
        or raw_manifest.get("episode_count") != expected
    ):
        raise RuntimeError("complete raw manifest required")
    output_directory = execution_directory(phase, regime)
    output_directory.mkdir(parents=True, exist_ok=True)
    failure_path = ATTEMPT_ROOT / f"audit/{phase}_{regime}_execution_failure.json"

    v5 = load_v5_runner()
    torch, runtime, model_io, device, base, contract, stack = (
        v5.load_scientific_stack(device_name)
    )
    solver, v1, _, provenance = stack
    common, _, _ = v5.distribution_modules()
    gate = v5.load_gate_tensors(torch, device)
    before = runtime.module_audit(base, solver, v1)
    if not before["passed"]:
        raise RuntimeError("frozen module audit failed before execution")
    records = []
    latency_history: list[np.ndarray] = []
    latency_actions: list[np.ndarray] = []
    latency_calls: list[np.ndarray] = []
    try:
        for index, raw_record in enumerate(raw_manifest["episodes"]):
            episode_id = str(raw_record["episode_id"])
            part_path = output_directory / f"{episode_id}.npz"
            sidecar_path = output_directory / f"{episode_id}.json"
            if part_path.exists() or sidecar_path.exists():
                if not (part_path.exists() and sidecar_path.exists()):
                    raise RuntimeError(f"partial execution part: {part_path}")
                record = read_json(sidecar_path)
                if sha256_file(part_path) != record["sha256"]:
                    raise RuntimeError(f"execution sidecar drift: {part_path}")
                records.append(record)
                continue
            raw_path = REPO_ROOT / raw_record["path"]
            if sha256_file(raw_path) != raw_record["sha256"]:
                raise RuntimeError(f"raw episode drift: {raw_path}")
            loaded, loader_audit = v5.load_model_gate_inputs(raw_path)
            if (
                loader_audit["loaded_keys"] != ["action", "pixels"]
                or loader_audit["contact_or_privileged_loaded"]
            ):
                raise RuntimeError("model/gate input isolation failed")
            history, actions, target, base_prediction = (
                v5.prepare_episode_tensors(
                    torch,
                    runtime,
                    model_io,
                    device,
                    base,
                    contract,
                    common,
                    loaded,
                )
            )
            dense, dense_calls, dense_scores, _ = v5.dense_shadow(
                torch, solver, gate, history, actions, base_prediction
            )
            sparse, calls, scores, features = v5.manual_sparse(
                torch, solver, gate, history, actions, base_prediction
            )
            repeat_sparse, repeat_calls, repeat_scores, repeat_features = (
                v5.manual_sparse(
                    torch, solver, gate, history, actions, base_prediction
                )
            )
            with torch.inference_mode():
                selected_api = solver.forward_selected(
                    history, actions, base_prediction, calls
                )
            selected_dense = dense[
                torch.arange(len(history), device=device), calls - 1
            ]
            delta = (sparse - selected_dense).abs()
            depth_one = calls == 1
            checks = {
                "same_sparse_path_output_exact": bool(
                    torch.equal(sparse, repeat_sparse)
                ),
                "same_sparse_path_calls_exact": bool(
                    torch.equal(calls, repeat_calls)
                ),
                "same_sparse_path_scores_exact": exact_with_nan(
                    torch, scores, repeat_scores
                ),
                "same_sparse_path_features_exact": exact_with_nan(
                    torch, features, repeat_features
                ),
                "manual_sparse_forward_selected_exact": bool(
                    torch.equal(sparse, selected_api)
                ),
                "sparse_dense_calls_exact": bool(
                    torch.equal(calls, dense_calls)
                ),
                "sparse_dense_histogram_exact": bool(
                    torch.equal(
                        torch.bincount(calls, minlength=5),
                        torch.bincount(dense_calls, minlength=5),
                    )
                ),
                "depth_one_selected_rows_exact": bool(
                    not depth_one.any()
                    or torch.equal(sparse[depth_one], selected_dense[depth_one])
                ),
                "cross_batch_allclose": bool(
                    torch.allclose(
                        sparse,
                        selected_dense,
                        rtol=NUMERICAL_RTOL,
                        atol=NUMERICAL_ATOL,
                    )
                ),
                "cross_batch_max_abs_ceiling": bool(
                    float(delta.max().item()) <= NUMERICAL_MAX_ABS
                ),
                "dense_score_calls_exact": bool(
                    torch.equal(calls, dense_calls)
                ),
                "feature_shape": tuple(features.shape)
                == (ROWS_PER_EPISODE, 3, FEATURE_DIM),
                "target_loss_never_computed": True,
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"sparse/dense contract failed for {episode_id}: {checks}"
                )
            arrays = {
                "target": target.cpu().numpy().astype(np.float32),
                "dense_exits": dense.cpu().numpy().astype(np.float32),
                "sparse_selected": sparse.cpu().numpy().astype(np.float32),
                "calls": calls.cpu().numpy().astype(np.int8),
                "scores": scores.cpu().numpy().astype(np.float32),
                "features": features.cpu().numpy().astype(np.float32),
                "model_step": np.arange(3, 41, dtype=np.int16),
                "slot": np.full(
                    ROWS_PER_EPISODE, int(raw_record["slot"]), dtype=np.int32
                ),
            }
            atomic_npz(part_path, arrays)
            record = {
                "schema_version": 1,
                "created_unix_ns": time.time_ns(),
                "phase": phase,
                "regime": regime,
                "episode_id": episode_id,
                "slot": int(raw_record["slot"]),
                "raw_path": raw_record["path"],
                "raw_sha256": raw_record["sha256"],
                "path": relative_to_repo(part_path),
                "sha256": sha256_file(part_path),
                "bytes": part_path.stat().st_size,
                "loaded_input_keys": loader_audit["loaded_keys"],
                "contact_or_privileged_loaded": False,
                "equivalence": {
                    "checks": checks,
                    "cross_batch_max_abs": float(delta.max().item()),
                    "cross_batch_mean_abs": float(delta.mean().item()),
                    "cross_batch_rmse": float(
                        torch.sqrt(torch.square(sparse - selected_dense).mean()).item()
                    ),
                    "call_histogram": torch.bincount(
                        calls, minlength=5
                    )[1:].cpu().tolist(),
                    "passed": all(checks.values()),
                },
                "target_loss_contact_motion_phase_reward_success_inspected": False,
                "no_gradients": True,
            }
            atomic_json(sidecar_path, record, exclusive=True)
            records.append(record)
            if sum(len(item) for item in latency_history) < 1_024:
                latency_history.append(
                    history.cpu().numpy().astype(np.float32)
                )
                latency_actions.append(
                    actions.cpu().numpy().astype(np.float32)
                )
                latency_calls.append(calls.cpu().numpy().astype(np.int8))
            if phase == "target" and (
                (index + 1) % 50 == 0 or index + 1 == expected
            ):
                update_state_fields(
                    target_outcome_episodes_executed=executed_target_count()
                )
            print(
                f"executed {phase} {regime} {index + 1}/{expected}",
                flush=True,
            )
        runtime.synchronize(device)
        after = runtime.module_audit(base, solver, v1)
        if before != after or not after["passed"]:
            raise RuntimeError("frozen module or gradient state changed")
    except BaseException as error:
        if not failure_path.exists():
            atomic_json(
                failure_path,
                {
                    "schema_version": 1,
                    "created_unix_ns": time.time_ns(),
                    "phase": phase,
                    "regime": regime,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_execution_parts": len(records),
                    "target_loss_contact_motion_phase_reward_success_inspected": False,
                    "prospective_statistical_analysis_performed": False,
                },
                exclusive=True,
            )
        raise

    if len(records) != expected or len(
        {item["episode_id"] for item in records}
    ) != expected:
        raise RuntimeError("execution cohort is not exact and unique")
    if phase == "target" and sum(len(item) for item in latency_history) < 1_024:
        latency_history = []
        latency_actions = []
        latency_calls = []
        for raw_record, execution_record in zip(
            raw_manifest["episodes"], records
        ):
            raw_path = REPO_ROOT / raw_record["path"]
            loaded, loader_audit = v5.load_model_gate_inputs(raw_path)
            if loader_audit["loaded_keys"] != ["action", "pixels"]:
                raise RuntimeError("latency recovery input isolation failed")
            history, actions, _, _ = v5.prepare_episode_tensors(
                torch,
                runtime,
                model_io,
                device,
                base,
                contract,
                common,
                loaded,
            )
            with np.load(
                REPO_ROOT / execution_record["path"], allow_pickle=False
            ) as stored:
                recovered_calls = stored["calls"].astype(np.int8)
            latency_history.append(
                history.cpu().numpy().astype(np.float32)
            )
            latency_actions.append(
                actions.cpu().numpy().astype(np.float32)
            )
            latency_calls.append(recovered_calls)
            if sum(len(item) for item in latency_history) >= 1_024:
                break
    latency_path = latency_input_path(regime)
    history_array = np.concatenate(latency_history)[:1_024]
    action_array = np.concatenate(latency_actions)[:1_024]
    call_array = np.concatenate(latency_calls)[:1_024]
    if phase == "target":
        atomic_npz(
            latency_path,
            {
                "history": history_array,
                "actions": action_array,
                "expected_calls": call_array,
            },
            exclusive=True,
        )
    manifest = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "phase": phase,
        "regime": regime,
        "complete": True,
        "episode_count": expected,
        "row_count": expected * ROWS_PER_EPISODE,
        "episodes": records,
        "actual_adaptive_output": "manual sparse execution",
        "dense_execution_role": "numerical shadow and fixed-depth comparator only",
        "all_equivalence_checks_passed": all(
            item["equivalence"]["passed"] for item in records
        ),
        "loaded_input_keys": ["action", "pixels"],
        "contact_or_privileged_loaded": False,
        "target_loss_contact_motion_phase_reward_success_inspected": False,
        "v3_test_targets_opened": False,
        "combined_v3_cache_numpy_loaded": False,
        "released_hdf5_opened": False,
        "no_gradients": True,
        "module_before": before,
        "module_after": after,
        "base_provenance": provenance,
        "runtime": runtime_snapshot,
        "pre_outcome_seal_sha256": seal["seal_sha256"],
        "latency_input_path": (
            relative_to_repo(latency_path) if phase == "target" else None
        ),
        "latency_input_sha256": (
            sha256_file(latency_path) if phase == "target" else None
        ),
    }
    path = execution_manifest_path(phase, regime)
    atomic_json(path, manifest, exclusive=True)
    append_ledger(
        "execution_cohort_complete",
        phase=phase,
        regime=regime,
        episodes=expected,
        rows=expected * ROWS_PER_EPISODE,
        manifest_path=relative_to_repo(path),
        manifest_sha256=sha256_file(path),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "target"))
    parser.add_argument("regime", choices=tuple(REGIMES))
    parser.add_argument("--device", default="mps")
    arguments = parser.parse_args()
    result = execute(arguments.phase, arguments.regime, arguments.device)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
