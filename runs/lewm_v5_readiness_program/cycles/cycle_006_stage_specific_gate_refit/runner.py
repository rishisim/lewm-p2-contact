#!/usr/bin/env python3
"""Sealed generator, actual sparse executor, and separately accounted dense shadow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from counted_features import build_counted_causal_features
from cycle_common import (
    ACTION_DIM,
    FEATURE_DIM,
    HISTORY_LEN,
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    REPO_ROOT,
    ROLE_COUNTS,
    ROOT,
    ROWS_PER_EPISODE,
    atomic_json,
    atomic_npz,
    read_json,
    sha256_file,
)
from input_loader import INPUT_ALLOWLIST, load_model_gate_inputs


def verify_pre_data() -> dict[str, Any]:
    from verify_pre_data import verify

    return verify()


def verify_design() -> dict[str, Any]:
    from verify_design import verify

    return verify()


def verify_role_permission(role: str) -> dict[str, Any]:
    return verify_design() if role in ("fit", "selection") else verify_pre_data()


def distribution_modules() -> tuple[Any, Any, Any]:
    distribution = REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract"
    if str(distribution) not in sys.path:
        sys.path.insert(0, str(distribution))
    import common
    import generator
    import generator_seedfix

    return common, generator, generator_seedfix


def load_scientific_stack(device_name: str) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    common, _, _ = distribution_modules()
    runtime = common.load_runtime()
    model_io = common.load_model_io()
    device = runtime.choose_device(device_name)
    base, contract, provenance = runtime.load_base_model(device)
    solver, v1, models = runtime.load_solver(device)
    return torch, runtime, model_io, device, base, contract, (solver, v1, models, provenance)


def load_gate_tensors(torch: Any, device: Any) -> dict[str, Any]:
    with np.load(ROOT / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        gate = {
            "a_raw": torch.as_tensor(stored["a_raw"], device=device),
            "b_raw": torch.as_tensor(stored["b_raw"], device=device),
            "a_white": torch.as_tensor(stored["a_white"], device=device),
            "b_white": torch.as_tensor(stored["b_white"], device=device),
            "thresholds": torch.as_tensor(stored["thresholds"], dtype=torch.float64, device=device),
        }
    if gate["a_raw"].shape != (3, FEATURE_DIM) or gate["a_white"].shape != (3, FEATURE_DIM):
        raise RuntimeError("compiled stage-specific gate shape drift")
    if gate["thresholds"].shape != (3,) or not bool(torch.isfinite(gate["thresholds"]).all().item()):
        raise RuntimeError("compiled threshold vector drift")
    return gate


def score_gate(
    torch: Any,
    gate: dict[str, Any],
    history: Any,
    actions: Any,
    current: Any,
    update: Any,
    stage: int,
) -> tuple[Any, Any]:
    if stage not in (0, 1, 2):
        raise ValueError("gate stage must be zero, one, or two")
    features = build_counted_causal_features(history, actions, current, update)
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError("counted causal feature width drift")
    raw_score = features @ gate["a_raw"][stage] + gate["b_raw"][stage]
    white_score = features @ gate["a_white"][stage] + gate["b_white"][stage]
    return torch.minimum(raw_score, white_score), features


def manual_sparse(
    torch: Any,
    solver: Any,
    gate: dict[str, Any],
    history: Any,
    actions: Any,
    base_prediction: Any,
) -> tuple[Any, Any, Any, Any]:
    """The actual adaptive path: later adapters run only on active rows."""
    with torch.inference_mode():
        anchored = solver.anchor(history, actions, base_prediction)
        current = anchored[1]
        last_update = current - base_prediction
        calls = torch.ones(len(history), dtype=torch.long, device=history.device)
        scores = torch.full((len(history), 3), float("nan"), device=history.device)
        features = torch.full(
            (len(history), 3, FEATURE_DIM), float("nan"), device=history.device
        )
        active = torch.arange(len(history), device=history.device)
        for stage, adapter in enumerate(solver.adapters):
            if not active.numel():
                break
            local_score, local_features = score_gate(
                torch,
                gate,
                history.index_select(0, active),
                actions.index_select(0, active),
                current.index_select(0, active),
                last_update.index_select(0, active),
                stage,
            )
            scores[active, stage] = local_score
            features[active, stage] = local_features
            active = active[local_score > gate["thresholds"][stage]]
            if not active.numel():
                break
            preceding = current.index_select(0, active)
            update = adapter(
                history.index_select(0, active),
                actions.index_select(0, active),
                preceding,
            )
            current = current.index_copy(0, active, preceding + update)
            last_update = last_update.index_copy(0, active, update)
            calls[active] += 1
    return current, calls, scores, features


def dense_shadow(
    torch: Any,
    solver: Any,
    gate: dict[str, Any],
    history: Any,
    actions: Any,
    base_prediction: Any,
) -> tuple[Any, Any, Any, Any]:
    """Full-batch all-exit shadow; never supplies the adaptive result."""
    with torch.inference_mode():
        outputs, updates = solver(
            history, actions, base_prediction, max_depth=4, return_updates=True
        )
        exits = torch.stack([outputs[depth] for depth in (1, 2, 3, 4)], dim=1)
        score_parts = []
        feature_parts = []
        for stage, depth in enumerate((1, 2, 3)):
            score, feature = score_gate(
                torch, gate, history, actions, outputs[depth], updates[depth], stage
            )
            score_parts.append(score)
            feature_parts.append(feature)
        scores = torch.stack(score_parts, dim=1)
        features = torch.stack(feature_parts, dim=1)
        calls = torch.ones(len(history), dtype=torch.long, device=history.device)
        active = torch.ones(len(history), dtype=torch.bool, device=history.device)
        for stage in range(3):
            active &= scores[:, stage] > gate["thresholds"][stage]
            calls += active.long()
    return exits, calls, scores, features


def tensors_exact_with_nan(torch: Any, left: Any, right: Any) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_nan = torch.isnan(left) if torch.is_floating_point(left) else torch.zeros_like(left, dtype=torch.bool)
    right_nan = torch.isnan(right) if torch.is_floating_point(right) else torch.zeros_like(right, dtype=torch.bool)
    if not torch.equal(left_nan, right_nan):
        return False
    return bool(torch.equal(left[~left_nan], right[~right_nan]))


def prepare_episode_tensors(
    torch: Any,
    runtime: Any,
    model_io: Any,
    device: Any,
    base: Any,
    contract: Any,
    common: Any,
    loaded: dict[str, np.ndarray],
) -> tuple[Any, Any, Any, Any]:
    pixels = loaded["pixels"][::5]
    raw_actions = loaded["action"][:200].astype(np.float32)
    if len(pixels) != 41:
        raise RuntimeError("frameskip did not produce 41 modeled frames")
    latent_parts = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 64):
            transformed = model_io.pixel_transform(
                pixels[start : start + 64], contract.image_size, device
            ).unsqueeze(0)
            latent_parts.append(base.encode({"pixels": transformed})["emb"].squeeze(0).cpu())
    latent = torch.cat(latent_parts).numpy().astype(np.float32)
    normalized = (raw_actions - common.FROZEN_ACTION_MEAN) / common.FROZEN_ACTION_STD
    blocks = normalized.reshape(40, ACTION_DIM)
    history = np.stack([latent[index : index + HISTORY_LEN] for index in range(ROWS_PER_EPISODE)])
    actions = np.stack([blocks[index : index + HISTORY_LEN] for index in range(ROWS_PER_EPISODE)])
    target = latent[HISTORY_LEN:]
    history_tensor = torch.as_tensor(history, device=device)
    action_tensor = torch.as_tensor(actions, device=device)
    with torch.inference_mode():
        base_prediction = runtime.base_predict(base, history_tensor, action_tensor)
    return history_tensor, action_tensor, torch.as_tensor(target, device=device), base_prediction


def _validate_existing_raw_manifest(role: str, expected: int) -> dict[str, Any] | None:
    path = ROOT / f"data/{role}_raw_manifest.json"
    if not path.exists():
        return None
    manifest = read_json(path)
    if not manifest.get("complete") or manifest.get("episode_count") != expected:
        raise RuntimeError("existing raw manifest is incomplete")
    for record in manifest["episodes"]:
        candidate = REPO_ROOT / record["path"]
        if sha256_file(candidate) != record["sha256"]:
            raise RuntimeError(f"raw episode hash drift: {candidate}")
    return manifest


def generate(role: str) -> dict[str, Any]:
    verify_role_permission(role)
    ledger = read_json(ROOT / "cohort_seed_ledger.json")
    specs = ledger["roles"][role]
    expected = ROLE_COUNTS[role]
    if len(specs) != expected:
        raise RuntimeError("sealed cohort size mismatch")
    existing_manifest = _validate_existing_raw_manifest(role, expected)
    if existing_manifest is not None:
        return existing_manifest

    common, generator, seedfix = distribution_modules()
    output_directory = ROOT / f"data/{role}_raw"
    output_directory.mkdir(parents=True, exist_ok=True)
    failure_path = ROOT / f"audit/{role}_generation_failures.json"
    failures = read_json(failure_path).get("failures", []) if failure_path.exists() else []
    failed_seed_tuples = {
        (int(item["env_seed"]), int(item["policy_seed"]), int(item["oracle_np_seed"]))
        for item in failures
    }
    replacements = ledger["roles"]["replacement"]
    used_replacements: set[tuple[int, int, int]] = set()
    for prior_sidecar in ROOT.glob("data/*_raw/*.json"):
        prior_record = read_json(prior_sidecar)
        if prior_record.get("replacement_used"):
            used_replacements.add(
                (
                    int(prior_record["env_seed"]),
                    int(prior_record["policy_seed"]),
                    int(prior_record["oracle_np_seed"]),
                )
            )
    records = []
    world = policy = None
    try:
        for index, primary in enumerate(specs):
            raw_path = output_directory / f"{primary['episode_id']}.npz"
            sidecar_path = output_directory / f"{primary['episode_id']}.json"
            if raw_path.exists() or sidecar_path.exists():
                if not (raw_path.exists() and sidecar_path.exists()):
                    raise RuntimeError(f"partial raw artifact: {raw_path}")
                record = read_json(sidecar_path)
                if sha256_file(raw_path) != record["sha256"]:
                    raise RuntimeError(f"raw sidecar hash drift: {raw_path}")
                records.append(record)
                if record.get("replacement_used"):
                    used_replacements.add(
                        (
                            int(record["env_seed"]),
                            int(record["policy_seed"]),
                            int(record["oracle_np_seed"]),
                        )
                    )
                continue

            candidates = [(dict(primary), False)] + [
                (dict(item), True) for item in replacements
            ]
            for candidate, replacement_used in candidates:
                seed_tuple = (
                    int(candidate["env_seed"]),
                    int(candidate["policy_seed"]),
                    int(candidate["oracle_np_seed"]),
                )
                if seed_tuple in failed_seed_tuples or (replacement_used and seed_tuple in used_replacements):
                    continue
                try:
                    if world is None:
                        world, policy = generator.make_world("plan_oracle")
                    arrays, generation_audit = seedfix.generate_episode_seedfixed(
                        world,
                        policy,
                        phase=role,
                        policy_type="plan_oracle",
                        slot=int(primary["slot"]),
                        trajectory_id=str(primary["episode_id"]),
                        env_seed=seed_tuple[0],
                        policy_seed=seed_tuple[1],
                        oracle_np_seed=seed_tuple[2],
                    )
                    atomic_npz(raw_path, arrays)
                    record = {
                        "role": role,
                        "slot": int(primary["slot"]),
                        "episode_id": str(primary["episode_id"]),
                        "seed_source_episode_id": str(candidate["episode_id"]),
                        "replacement_used": replacement_used,
                        "env_seed": seed_tuple[0],
                        "policy_seed": seed_tuple[1],
                        "oracle_np_seed": seed_tuple[2],
                        "path": str(raw_path.relative_to(REPO_ROOT)),
                        "sha256": sha256_file(raw_path),
                        "bytes": raw_path.stat().st_size,
                        "created_unix_ns": time.time_ns(),
                        "initial_state_sha256": generation_audit["initial_state_sha256"],
                        "deterministic_reset_amendment": bool(
                            generation_audit["environment_reset_seed_forwarding_bug_corrected"]
                        ),
                        "generator_records_full_environment_for_posthoc_audit": True,
                        "model_gate_arrays_loaded_during_generation": [],
                    }
                    atomic_json(sidecar_path, record, exclusive=True)
                    records.append(record)
                    if replacement_used:
                        used_replacements.add(seed_tuple)
                    print(f"generated {role} {index + 1}/{expected}", flush=True)
                    break
                except BaseException as error:
                    failures.append(
                        {
                            "role": role,
                            "slot": int(primary["slot"]),
                            "attempted_seed_source_episode_id": str(candidate["episode_id"]),
                            "env_seed": seed_tuple[0],
                            "policy_seed": seed_tuple[1],
                            "oracle_np_seed": seed_tuple[2],
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                            "created_unix_ns": time.time_ns(),
                            "outcome_loss_contact_or_success_inspected": False,
                        }
                    )
                    failed_seed_tuples.add(seed_tuple)
                    atomic_json(
                        failure_path,
                        {
                            "schema_version": 1,
                            "replacement_rule": "mechanical exception only; never outcome, loss, contact, or success",
                            "failures": failures,
                        },
                    )
                    if world is not None:
                        world.close()
                    world = policy = None
            else:
                raise RuntimeError("fresh replacement seed pool exhausted")
    finally:
        if world is not None:
            world.close()

    if len(records) != expected or len({item["episode_id"] for item in records}) != expected:
        raise RuntimeError("generated cohort is not exact and unique")
    manifest = {
        "schema_version": 1,
        "role": role,
        "complete": True,
        "episode_count": expected,
        "created_unix_ns": time.time_ns(),
        "episodes": records,
        "input_allowlist_for_later_model_gate_execution": sorted(INPUT_ALLOWLIST),
        "prospective_outcomes_inspected_during_generation": False,
        "v3_test_targets_opened": False,
        "released_hdf5_opened": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / f"data/{role}_raw_manifest.json", manifest, exclusive=True)
    return manifest


def evaluate_development(role: str, device_name: str) -> dict[str, Any]:
    """Evaluate every dense exit and causal feature for isolated fit/selection roles."""
    if role not in ("fit", "selection"):
        raise ValueError("development evaluation is restricted to fit or selection")
    verify_design()
    expected = ROLE_COUNTS[role]
    output_path = ROOT / f"data/{role}_evaluated.npz"
    manifest_path = ROOT / f"data/{role}_evaluated_manifest.json"
    if output_path.exists() or manifest_path.exists():
        if not (output_path.exists() and manifest_path.exists()):
            raise RuntimeError("partial development evaluation artifact")
        existing = read_json(manifest_path)
        if sha256_file(output_path) != existing["sha256"]:
            raise RuntimeError("development evaluation hash drift")
        return existing
    raw_manifest = _validate_existing_raw_manifest(role, expected)
    if raw_manifest is None:
        raise RuntimeError("complete raw manifest required before development evaluation")
    try:
        torch, runtime, model_io, device, base, contract, stack = load_scientific_stack(device_name)
        solver, v1, _, provenance = stack
        common, _, _ = distribution_modules()
        module_before = runtime.module_audit(base, solver, v1)
        if not module_before["passed"]:
            raise RuntimeError("frozen module audit failed before development evaluation")
        targets = []
        exits_parts = []
        feature_parts = []
        episode_parts = []
        step_parts = []
        loaded_keys: set[str] = set()
        for episode_index, record in enumerate(raw_manifest["episodes"]):
            raw_path = REPO_ROOT / record["path"]
            if sha256_file(raw_path) != record["sha256"]:
                raise RuntimeError("raw development episode drift")
            loaded, loader_audit = load_model_gate_inputs(raw_path)
            loaded_keys.update(loader_audit["loaded_keys"])
            if loader_audit["contact_or_privileged_loaded"]:
                raise RuntimeError("development input isolation failure")
            history, actions, target, base_prediction = prepare_episode_tensors(
                torch, runtime, model_io, device, base, contract, common, loaded
            )
            with torch.inference_mode():
                outputs, updates = solver(
                    history, actions, base_prediction, max_depth=4, return_updates=True
                )
                exits = torch.stack([outputs[depth] for depth in (1, 2, 3, 4)], dim=1)
                features = torch.stack(
                    [
                        build_counted_causal_features(
                            history, actions, outputs[depth], updates[depth]
                        )
                        for depth in (1, 2, 3)
                    ],
                    dim=1,
                )
            if exits.shape != (ROWS_PER_EPISODE, 4, 192) or features.shape != (
                ROWS_PER_EPISODE,
                3,
                FEATURE_DIM,
            ):
                raise RuntimeError("dense development tensor shape drift")
            targets.append(target.cpu().numpy().astype(np.float32))
            exits_parts.append(exits.cpu().numpy().astype(np.float32))
            feature_parts.append(features.cpu().numpy().astype(np.float32))
            episode_parts.append(np.full(ROWS_PER_EPISODE, episode_index, dtype=np.int32))
            step_parts.append(np.arange(3, 41, dtype=np.int16))
            print(f"evaluated {role} {episode_index + 1}/{expected}", flush=True)
        runtime.synchronize(device)
        module_after = runtime.module_audit(base, solver, v1)
        if module_before != module_after or not module_after["passed"]:
            raise RuntimeError("frozen module or gradient state changed")
        arrays = {
            "target": np.concatenate(targets),
            "exits": np.concatenate(exits_parts),
            "features": np.concatenate(feature_parts),
            "episode_id": np.concatenate(episode_parts),
            "model_step": np.concatenate(step_parts),
        }
        expected_rows = expected * ROWS_PER_EPISODE
        if len(arrays["target"]) != expected_rows:
            raise RuntimeError("development row count mismatch")
        atomic_npz(output_path, arrays)
        manifest = {
            "schema_version": 1,
            "role": role,
            "created_unix_ns": time.time_ns(),
            "path": str(output_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(output_path),
            "rows": expected_rows,
            "episode_count": expected,
            "evaluation": "all four dense exits and all three counted causal feature stages",
            "loaded_input_keys": sorted(loaded_keys),
            "contact_or_privileged_loaded": False,
            "targets_used_only_for_declared_fit_or_selection": True,
            "v3_test_targets_opened": False,
            "released_hdf5_opened": False,
            "base_provenance": provenance,
            "module_before": module_before,
            "module_after": module_after,
            "no_gradients": True,
            "v5_outcome_episodes": 0,
        }
        atomic_json(manifest_path, manifest, exclusive=True)
        return manifest
    except BaseException as error:
        failure_path = ROOT / f"audit/{role}_evaluation_failure.json"
        if not failure_path.exists():
            atomic_json(
                failure_path,
                {
                    "schema_version": 1,
                    "role": role,
                    "created_unix_ns": time.time_ns(),
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": traceback.format_exc(),
                    "v5_outcome_episodes": 0,
                },
                exclusive=True,
            )
        raise


def execute(role: str, device_name: str) -> dict[str, Any]:
    verify_pre_data()
    expected = ROLE_COUNTS[role]
    output_path = ROOT / f"data/{role}_execution.npz"
    manifest_path = ROOT / f"data/{role}_execution_manifest.json"
    if output_path.exists() or manifest_path.exists():
        if not (output_path.exists() and manifest_path.exists()):
            raise RuntimeError("partial execution artifact")
        existing = read_json(manifest_path)
        if sha256_file(output_path) != existing["sha256"]:
            raise RuntimeError("execution artifact hash drift")
        return existing
    raw_manifest = _validate_existing_raw_manifest(role, expected)
    if raw_manifest is None:
        raise RuntimeError("complete raw manifest required before execution")

    try:
        torch, runtime, model_io, device, base, contract, stack = load_scientific_stack(device_name)
        solver, v1, models, provenance = stack
        common, _, _ = distribution_modules()
        gate = load_gate_tensors(torch, device)
        module_before = runtime.module_audit(base, solver, v1)
        if not module_before["passed"]:
            raise RuntimeError("frozen module audit failed before execution")

        targets: list[np.ndarray] = []
        dense_exits: list[np.ndarray] = []
        sparse_selected: list[np.ndarray] = []
        calls_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        episode_parts: list[np.ndarray] = []
        step_parts: list[np.ndarray] = []
        latency_history: list[np.ndarray] = []
        latency_actions: list[np.ndarray] = []
        equivalence = []
        loaded_keys: set[str] = set()

        for episode_index, record in enumerate(raw_manifest["episodes"]):
            raw_path = REPO_ROOT / record["path"]
            if sha256_file(raw_path) != record["sha256"]:
                raise RuntimeError("raw episode drift before execution")
            loaded, loader_audit = load_model_gate_inputs(raw_path)
            loaded_keys.update(loader_audit["loaded_keys"])
            if loader_audit["contact_or_privileged_loaded"]:
                raise RuntimeError("input isolation failure")
            history, actions, target, base_prediction = prepare_episode_tensors(
                torch, runtime, model_io, device, base, contract, common, loaded
            )

            dense, dense_calls, dense_scores, _ = dense_shadow(
                torch, solver, gate, history, actions, base_prediction
            )
            sparse, calls, scores, features = manual_sparse(
                torch, solver, gate, history, actions, base_prediction
            )
            sparse_repeat, calls_repeat, scores_repeat, features_repeat = manual_sparse(
                torch, solver, gate, history, actions, base_prediction
            )
            with torch.inference_mode():
                selected_api = solver.forward_selected(history, actions, base_prediction, calls)
            selected_dense = dense[
                torch.arange(len(history), device=device), calls - 1
            ]

            same_path_exact = (
                torch.equal(sparse, sparse_repeat)
                and torch.equal(calls, calls_repeat)
                and tensors_exact_with_nan(torch, scores, scores_repeat)
                and tensors_exact_with_nan(torch, features, features_repeat)
            )
            manual_api_exact = bool(torch.equal(sparse, selected_api))
            calls_exact = bool(torch.equal(calls, dense_calls))
            histogram_exact = bool(
                torch.equal(
                    torch.bincount(calls, minlength=5),
                    torch.bincount(dense_calls, minlength=5),
                )
            )
            difference = (sparse - selected_dense).abs()
            max_abs = float(difference.max().item())
            mean_abs = float(difference.mean().item())
            rmse = float(torch.sqrt(torch.square(sparse - selected_dense).mean()).item())
            numerical_close = bool(
                torch.allclose(
                    sparse,
                    selected_dense,
                    rtol=NUMERICAL_RTOL,
                    atol=NUMERICAL_ATOL,
                )
            )
            ceiling_passed = max_abs <= NUMERICAL_MAX_ABS
            depth_one = calls == 1
            depth_one_exact = bool(
                not depth_one.any() or torch.equal(sparse[depth_one], selected_dense[depth_one])
            )
            raw_sparse_loss = torch.square(sparse - target).mean(1)
            raw_dense_loss = torch.square(selected_dense - target).mean(1)
            maximum_row_mse_impact = float(
                (raw_sparse_loss - raw_dense_loss).abs().max().item()
            )
            passed = all(
                (
                    same_path_exact,
                    manual_api_exact,
                    calls_exact,
                    histogram_exact,
                    numerical_close,
                    ceiling_passed,
                    depth_one_exact,
                )
            )
            equivalence.append(
                {
                    "episode_index": episode_index,
                    "episode_id": record["episode_id"],
                    "same_sparse_path_outputs_calls_scores_features_bitwise_exact": same_path_exact,
                    "manual_sparse_vs_forward_selected_bitwise_exact": manual_api_exact,
                    "sparse_vs_dense_calls_exact": calls_exact,
                    "call_histogram_exact": histogram_exact,
                    "depth_one_selected_rows_bitwise_exact": depth_one_exact,
                    "cross_batch_allclose": numerical_close,
                    "cross_batch_max_abs_ceiling_passed": ceiling_passed,
                    "cross_batch_max_abs": max_abs,
                    "cross_batch_mean_abs": mean_abs,
                    "cross_batch_rmse": rmse,
                    "maximum_observed_per_row_raw_mse_impact": maximum_row_mse_impact,
                    "passed": passed,
                }
            )
            if not passed:
                raise RuntimeError(f"sparse execution contract failed at {record['episode_id']}")

            targets.append(target.cpu().numpy().astype(np.float32))
            dense_exits.append(dense.cpu().numpy().astype(np.float32))
            sparse_selected.append(sparse.cpu().numpy().astype(np.float32))
            calls_parts.append(calls.cpu().numpy().astype(np.int8))
            score_parts.append(scores.cpu().numpy().astype(np.float32))
            feature_parts.append(features.cpu().numpy().astype(np.float32))
            episode_parts.append(np.full(ROWS_PER_EPISODE, episode_index, dtype=np.int32))
            step_parts.append(np.arange(3, 41, dtype=np.int16))
            if sum(len(item) for item in latency_history) < 1024:
                latency_history.append(history.cpu().numpy().astype(np.float32))
                latency_actions.append(actions.cpu().numpy().astype(np.float32))
            print(f"executed {role} {episode_index + 1}/{expected}", flush=True)

        runtime.synchronize(device)
        module_after = runtime.module_audit(base, solver, v1)
        if module_before != module_after or not module_after["passed"]:
            raise RuntimeError("frozen module or gradient state changed")

        arrays = {
            "target": np.concatenate(targets),
            "dense_exits": np.concatenate(dense_exits),
            "sparse_selected": np.concatenate(sparse_selected),
            "calls": np.concatenate(calls_parts),
            "scores": np.concatenate(score_parts),
            "features": np.concatenate(feature_parts),
            "episode_id": np.concatenate(episode_parts),
            "model_step": np.concatenate(step_parts),
        }
        expected_rows = expected * ROWS_PER_EPISODE
        if len(arrays["calls"]) != expected_rows:
            raise RuntimeError("execution row count mismatch")
        atomic_npz(output_path, arrays)

        latency_path = ROOT / f"data/{role}_latency_inputs.npz"
        history_array = np.concatenate(latency_history)[:1024]
        action_array = np.concatenate(latency_actions)[:1024]
        atomic_npz(
            latency_path,
            {
                "history": history_array,
                "actions": action_array,
                "expected_calls": arrays["calls"][: len(history_array)],
            },
        )
        manifest = {
            "schema_version": 1,
            "role": role,
            "created_unix_ns": time.time_ns(),
            "path": str(output_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(output_path),
            "rows": expected_rows,
            "episode_count": expected,
            "actual_adaptive_output": "manual sparse execution",
            "dense_execution_role": "separately accounted comparator and numerical shadow only",
            "loaded_input_keys": sorted(loaded_keys),
            "contact_or_privileged_loaded": False,
            "v3_test_targets_opened": False,
            "released_hdf5_opened": False,
            "v5_outcome_episodes": 0,
            "no_gradients": True,
            "module_before": module_before,
            "module_after": module_after,
            "base_provenance": provenance,
            "numerical_contract": {
                "rtol": NUMERICAL_RTOL,
                "atol": NUMERICAL_ATOL,
                "maximum_absolute_error_ceiling": NUMERICAL_MAX_ABS,
                "calls_scores_same_path_and_forward_selected_remain_exact": True,
            },
            "all_equivalence_checks_passed": all(item["passed"] for item in equivalence),
            "equivalence": equivalence,
            "latency_input_path": str(latency_path.relative_to(REPO_ROOT)),
            "latency_input_sha256": sha256_file(latency_path),
        }
        atomic_json(manifest_path, manifest, exclusive=True)
        return manifest
    except BaseException as error:
        failure_path = ROOT / f"audit/{role}_execution_failure.json"
        if not failure_path.exists():
            atomic_json(
                failure_path,
                {
                    "schema_version": 1,
                    "role": role,
                    "created_unix_ns": time.time_ns(),
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": traceback.format_exc(),
                    "prospective_statistical_analysis_performed": False,
                    "v5_outcome_episodes": 0,
                },
                exclusive=True,
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("role", choices=("fit", "selection", "smoke", "prospective"))
    development_parser = subparsers.add_parser("evaluate-development")
    development_parser.add_argument("role", choices=("fit", "selection"))
    development_parser.add_argument("--device", default="auto")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("role", choices=("smoke", "prospective"))
    execute_parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if arguments.command == "generate":
        result = generate(arguments.role)
    elif arguments.command == "evaluate-development":
        result = evaluate_development(arguments.role, arguments.device)
    else:
        result = execute(arguments.role, arguments.device)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
