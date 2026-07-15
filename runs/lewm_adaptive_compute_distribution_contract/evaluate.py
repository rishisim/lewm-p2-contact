#!/usr/bin/env python3
"""Encode and evaluate offline/fresh/V4 data with the unchanged frozen runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch

import common


EVAL_ROLES = (
    "offline_discovery",
    "offline_calibration",
    "plan_oracle",
    "markov_oracle",
    "v4_markov",
)


def _iter(n: int, size: int):
    for start in range(0, n, size):
        yield np.arange(start, min(start + size, n), dtype=np.int64)


def _physical(raw: Mapping[str, np.ndarray], model_steps: np.ndarray) -> dict[str, np.ndarray]:
    contact = np.asarray(raw["proprio_gripper_contact"], dtype=np.float64).reshape(201, -1)[:, 0] > 1e-9
    effector = np.asarray(raw["proprio_effector_pos"], dtype=np.float64)
    block = np.asarray(raw["privileged_block_0_pos"], dtype=np.float64)
    action = np.asarray(raw["action"], dtype=np.float32)
    n = len(model_steps)
    interaction = np.empty(n, dtype=np.bool_)
    impact = np.empty(n, dtype=np.bool_)
    effector_disp = np.empty(n, dtype=np.float32)
    block_disp = np.empty(n, dtype=np.float32)
    action_magnitude = np.empty(n, dtype=np.float32)
    for index, step in enumerate(model_steps.astype(np.int64)):
        target = int(step * common.FRAMESKIP)
        previous = target - common.FRAMESKIP
        now_contact = bool(contact[previous + 1 : target + 1].any())
        prior_start = max(0, previous - common.FRAMESKIP + 1)
        prior_contact = bool(contact[prior_start : previous + 1].any())
        interaction[index] = now_contact
        impact[index] = now_contact and not prior_contact
        effector_disp[index] = np.linalg.norm(effector[target] - effector[previous])
        block_disp[index] = np.linalg.norm(block[target] - block[previous])
        action_magnitude[index] = np.sqrt(
            np.square(action[previous:target].astype(np.float64)).sum(1).mean()
        )
    return {
        "interaction": interaction,
        "impact": impact,
        "effector_disp": effector_disp,
        "block_disp": block_disp,
        "action_magnitude": action_magnitude,
        "normalized_phase": model_steps.astype(np.float32) / 40.0,
    }


@torch.inference_mode()
def encode_fresh(policy_type: str, device_name: str) -> dict[str, Any]:
    common.assert_pre_generation_seal()
    if policy_type not in {"plan_oracle", "markov_oracle"}:
        raise ValueError("invalid fresh policy")
    _, raw_manifest_path = common.role_raw_paths("main", policy_type)
    manifest = common.study_json(raw_manifest_path)
    expected = 90 if policy_type == "plan_oracle" else 30
    if not manifest.get("complete") or manifest.get("episode_count") != expected:
        raise RuntimeError("fresh raw manifest incomplete")
    if manifest.get("phase") != "main" or manifest.get(
        "smoke_permanently_excluded_from_all_model_and_decision_metrics"
    ):
        raise RuntimeError("main encoder received a smoke manifest")
    output = common.STUDY_ROOT / "data" / f"{policy_type}_encoded.npz"
    output_manifest = common.STUDY_ROOT / "data" / f"{policy_type}_encoded_manifest.json"
    if output.exists() or output_manifest.exists():
        if not (output.exists() and output_manifest.exists()):
            raise RuntimeError("partial encoded artifact")
        prior = common.study_json(output_manifest)
        if common.sha256_file(output) != prior["encoded_sha256"]:
            raise RuntimeError("encoded cache drift")
        return prior
    runtime = common.load_runtime()
    device = runtime.choose_device(device_name)
    model_io = common.load_model_io()
    base, contract, provenance = runtime.load_base_model(device)
    before = runtime.module_audit(base)
    histories: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    env_seeds: list[np.ndarray] = []
    policy_seeds: list[np.ndarray] = []
    oracle_seeds: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    physical: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "interaction",
            "impact",
            "effector_disp",
            "block_disp",
            "action_magnitude",
            "normalized_phase",
        )
    }
    start_time = time.perf_counter()
    for position, record in enumerate(manifest["episodes"]):
        path = common.REPO_ROOT / record["path"]
        if common.sha256_file(path) != record["file_sha256"]:
            raise RuntimeError(f"fresh raw episode drift: {path}")
        if "smoke" in record["path"] or record["phase"] != "main":
            raise RuntimeError("smoke episode entered fresh encoder")
        with np.load(path, allow_pickle=False) as stored:
            raw = {name: stored[name] for name in stored.files}
        pixels = np.asarray(raw["pixels"])[:: common.FRAMESKIP]
        chunks: list[torch.Tensor] = []
        for part in _iter(len(pixels), 64):
            transformed = model_io.pixel_transform(pixels[part], contract.image_size, device)
            chunks.append(base.encode({"pixels": transformed.unsqueeze(0)})["emb"].squeeze(0).cpu())
        latent = torch.cat(chunks).numpy().astype(np.float32)
        if latent.shape != (common.MODEL_STEPS, common.LATENT_DIM):
            raise RuntimeError(f"unexpected latent shape {latent.shape}")
        raw_action = np.asarray(raw["action"][:-1], dtype=np.float32)
        normalized = (raw_action - common.FROZEN_ACTION_MEAN) / common.FROZEN_ACTION_STD
        blocks = normalized.reshape(40, common.BLOCKED_ACTION_DIM).astype(np.float32)
        n = common.EXAMPLES_PER_EPISODE
        histories.append(
            np.stack([latent[index : index + common.HISTORY] for index in range(n)]).astype(np.float32)
        )
        actions.append(
            np.stack([blocks[index : index + common.HISTORY] for index in range(n)]).astype(np.float32)
        )
        targets.append(latent[common.HISTORY :].astype(np.float32))
        episode_ids.append(np.full(n, int(record["slot"]), dtype=np.int64))
        env_seeds.append(np.full(n, int(record["env_seed"]), dtype=np.int64))
        policy_seeds.append(np.full(n, int(record["policy_seed"]), dtype=np.int64))
        oracle_seeds.append(np.full(n, int(record["oracle_np_seed"]), dtype=np.int64))
        model_steps = np.arange(common.HISTORY, common.MODEL_STEPS, dtype=np.int64)
        steps.append(model_steps)
        labels = _physical(raw, model_steps)
        for name, value in labels.items():
            physical[name].append(value)
        if (position + 1) % 10 == 0 or position + 1 == expected:
            print(f"encoded {policy_type} episode {position + 1}/{expected}", flush=True)
    history = np.concatenate(histories)
    action = np.concatenate(actions)
    target = np.concatenate(targets)
    predictions = []
    for part in _iter(len(history), 1024):
        h = torch.as_tensor(np.ascontiguousarray(history[part]), device=device)
        a = torch.as_tensor(np.ascontiguousarray(action[part]), device=device)
        predictions.append(runtime.base_predict(base, h, a).cpu().numpy())
    runtime.synchronize(device)
    arrays = {
        "history": history,
        "action": action,
        "base_pred": np.concatenate(predictions).astype(np.float32),
        "target": target,
        "episode_id": np.concatenate(episode_ids),
        "model_step": np.concatenate(steps),
        "env_seed": np.concatenate(env_seeds),
        "policy_seed": np.concatenate(policy_seeds),
        "oracle_np_seed": np.concatenate(oracle_seeds),
        **{name: np.concatenate(value) for name, value in physical.items()},
    }
    expected_rows = expected * common.EXAMPLES_PER_EPISODE
    if any(len(value) != expected_rows for value in arrays.values()):
        raise RuntimeError("fresh encoded row-count mismatch")
    if any(value.dtype.kind in "fc" and not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("nonfinite fresh encoded cache")
    after = runtime.module_audit(base)
    if before != after or not before["passed"]:
        raise RuntimeError("base model changed during encoding")
    common.atomic_npz(output, arrays)
    result = {
        **common.dataset_manifest_base(role=policy_type, episodes=expected, rows=expected_rows),
        "status": "complete_fresh_discovery_encoding",
        "source_namespace": "fresh_seeded_swm_ogbcube_rollout",
        "source_manifest": str(raw_manifest_path.relative_to(common.REPO_ROOT)),
        "source_manifest_sha256": common.sha256_file(raw_manifest_path),
        "encoded_path": str(output.relative_to(common.REPO_ROOT)),
        "encoded_sha256": common.sha256_file(output),
        "elapsed_seconds": time.perf_counter() - start_time,
        "model_contract": vars(contract),
        "base_provenance": provenance,
        "base_module_before": before,
        "base_module_after": after,
        "preprocessing_exact": True,
        "smoke_paths_present": False,
        "posthoc_labels_present_but_never_passed_to_solver_or_gate": True,
    }
    common.write_study_json(output_manifest, result, exclusive=True)
    return result


def _load_eval_inputs(role: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if role in {"offline_discovery", "offline_calibration"}:
        arrays, audit = common.load_allowed_cache(role)
        return arrays, audit
    if role in {"plan_oracle", "markov_oracle"}:
        path = common.STUDY_ROOT / "data" / f"{role}_encoded.npz"
        manifest_path = common.STUDY_ROOT / "data" / f"{role}_encoded_manifest.json"
        manifest = common.study_json(manifest_path)
        if common.sha256_file(path) != manifest["encoded_sha256"]:
            raise RuntimeError("fresh encoded input drift")
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in stored.files}
        return arrays, manifest
    if role == "v4_markov":
        path = common.V4_ROOT / "data/confirmation_encoded.npz"
        manifest_path = common.V4_ROOT / "data/confirmation_encoded_manifest.json"
        manifest = common.study_json(manifest_path)
        if common.sha256_file(path) != manifest["encoded_sha256"]:
            raise RuntimeError("V4 encoded input drift")
        with np.load(path, allow_pickle=False) as stored:
            original = {name: stored[name].copy() for name in stored.files}
        original["episode_id"] = original.pop("episode_slot").astype(np.int64)
        return original, {
            **manifest,
            "role": "v4_markov_diagnostic_only",
            "v3_test_targets_opened": False,
        }
    raise ValueError(f"unknown evaluation role {role}")


def _block_stats(value: np.ndarray) -> np.ndarray:
    value64 = np.asarray(value, dtype=np.float64)
    return np.stack(
        [
            value64.mean(1),
            value64.std(1),
            np.sqrt(np.square(value64).mean(1)),
            np.abs(value64).max(1),
            (np.abs(value64) > 3.0).mean(1),
        ],
        axis=1,
    ).astype(np.float32)


def _whitened(target: np.ndarray, exits: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    difference = np.asarray(exits, dtype=np.float64) - np.asarray(target, dtype=np.float64)[:, None, :]
    transformed = np.einsum("nkd,df->nkf", difference, matrix, optimize=False)
    return np.square(transformed).mean(2)


@torch.inference_mode()
def evaluate_role(role: str, device_name: str) -> dict[str, Any]:
    if role not in EVAL_ROLES:
        raise ValueError("invalid role")
    if role not in {"offline_discovery", "offline_calibration"}:
        common.assert_pre_generation_seal()
    output = common.STUDY_ROOT / "data" / f"{role}_evaluation.npz"
    manifest_path = common.STUDY_ROOT / "data" / f"{role}_evaluation_manifest.json"
    if output.exists() or manifest_path.exists():
        if not (output.exists() and manifest_path.exists()):
            raise RuntimeError("partial evaluation artifact")
        prior = common.study_json(manifest_path)
        if common.sha256_file(output) != prior["evaluation_sha256"]:
            raise RuntimeError("evaluation artifact drift")
        return prior
    arrays, input_manifest = _load_eval_inputs(role)
    runtime = common.load_runtime()
    device = runtime.choose_device(device_name)
    solver, v1, models = runtime.load_solver(device)
    gate = runtime.load_gate(device)
    whitening = runtime.load_whitening()
    before = runtime.module_audit(solver, v1, gate.model)
    n = len(arrays["target"])
    losses = np.empty((n, 5), dtype=np.float64)
    white = np.empty((n, 5), dtype=np.float64)
    scores = np.empty((n, 3), dtype=np.float64)
    calls = np.empty(n, dtype=np.int64)
    update_norms = np.empty((n, 4), dtype=np.float32)
    raw_stats = np.empty(
        (n, 3, len(common.FEATURE_BLOCKS), len(common.BLOCK_STAT_NAMES)), dtype=np.float32
    )
    z_stats = np.empty_like(raw_stats)
    raw_digest = hashlib.sha256()
    z_digest = hashlib.sha256()
    d0_identity = d0_bitwise = d1_bitwise = True
    sparse_call_exact = True
    sparse_output_max = 0.0
    start_time = time.perf_counter()
    for batch_index, part in enumerate(_iter(n, 1024)):
        history = torch.as_tensor(np.ascontiguousarray(arrays["history"][part]), device=device)
        action = torch.as_tensor(np.ascontiguousarray(arrays["action"][part]), device=device)
        base = torch.as_tensor(np.ascontiguousarray(arrays["base_pred"][part]), device=device)
        target = np.asarray(arrays["target"][part], dtype=np.float32)
        outputs, updates = solver(history, action, base, max_depth=4, return_updates=True)
        expected = v1(history, action, base, depths=(0, 1))
        d0_identity &= outputs[0] is base
        d0_bitwise &= bool(torch.equal(outputs[0], expected[0]))
        d1_bitwise &= bool(torch.equal(outputs[1], expected[1]))
        exits = torch.stack([outputs[depth] for depth in range(5)], 1).cpu().numpy()
        losses[part] = np.square(exits.astype(np.float64) - target[:, None, :]).mean(2)
        white[part] = _whitened(target, exits, whitening["matrix"])
        for depth in range(1, 5):
            update_norms[part, depth - 1] = torch.linalg.vector_norm(
                updates[depth], dim=1
            ).cpu().numpy()
        local_scores: list[np.ndarray] = []
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
                raw_stats[part, stage, block_index] = _block_stats(encoded[:, begin:end])
                z_stats[part, stage, block_index] = _block_stats(normalized[:, begin:end])
            local_scores.append(score_np)
        score_batch = np.stack(local_scores, axis=1)
        scores[part] = score_batch
        call_batch = runtime.sequential_calls_numpy(score_batch, common.COMPUTE_PRICE)
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
            print(f"evaluated {role} rows {int(part[-1]) + 1}/{n}", flush=True)
    runtime.synchronize(device)
    after = runtime.module_audit(solver, v1, gate.model)
    gains = losses[:, :-1] - losses[:, 1:]
    arrays_out = {
        "losses_d0_d4": losses,
        "whitened_losses_d0_d4": white,
        "marginal_gains_d0_d4": gains,
        "scores": scores,
        "calls": calls,
        "update_norms_d1_d4": update_norms,
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
    if not np.isfinite(losses).all() or not np.isfinite(white).all() or not np.isfinite(scores).all():
        raise RuntimeError("nonfinite frozen evaluation")
    causal_signature = "target" not in models.build_causal_features.__code__.co_varnames
    checks = {
        "d0_identity": d0_identity,
        "d0_bitwise_v1": d0_bitwise,
        "d1_bitwise_v1": d1_bitwise,
        "sparse_calls_exact": sparse_call_exact,
        "sparse_outputs_equivalent": sparse_output_max <= 2e-5,
        "module_state_unchanged": before == after,
        "all_modules_frozen_no_gradients": bool(before["passed"] and after["passed"]),
        "causal_feature_signature_excludes_target": causal_signature,
        "supported_calls_only": bool(np.isin(calls, common.SUPPORTED_CALLS).all()),
        "v3_test_targets_opened": False,
    }
    v4_recompute: dict[str, Any] | None = None
    if role == "v4_markov":
        prior_path = common.V4_ROOT / "data/confirmation_outcome_once.npz"
        with np.load(prior_path, allow_pickle=False) as stored:
            prior = {name: stored[name] for name in stored.files}
        v4_recompute = {
            "calls_exact": bool(np.array_equal(calls, prior["calls"])),
            "scores_max_abs": float(np.max(np.abs(scores - prior["scores"]))),
            "d1_d4_losses_max_abs": float(
                np.max(np.abs(losses[:, 1:] - prior["losses"]))
            ),
            "whitened_d1_d4_max_abs": float(
                np.max(np.abs(white[:, 1:] - prior["whitened_losses"]))
            ),
            "prior_outcome_sha256": common.sha256_file(prior_path),
            "diagnostic_only": True,
        }
        v4_recompute["passed"] = bool(
            v4_recompute["calls_exact"]
            and v4_recompute["scores_max_abs"] < 1e-10
            and v4_recompute["d1_d4_losses_max_abs"] < 1e-12
            and v4_recompute["whitened_d1_d4_max_abs"] < 1e-12
        )
        checks["v4_recompute_exact"] = v4_recompute["passed"]
    passed = all(value for key, value in checks.items() if key != "v3_test_targets_opened")
    if not passed:
        raise RuntimeError(f"frozen evaluation audit failed for {role}: {checks}")
    common.atomic_npz(output, arrays_out)
    unique_episodes = np.unique(arrays_out["episode_id"])
    test_intersection = (
        sorted(set(unique_episodes.astype(int).tolist()) & common.v3_test_set())
        if role.startswith("offline_")
        else []
    )
    if test_intersection:
        raise RuntimeError("test episode in derived evaluation cache")
    result = {
        **common.dataset_manifest_base(
            role=role, episodes=len(unique_episodes), rows=n
        ),
        "status": "complete_frozen_evaluation",
        "evaluation_path": str(output.relative_to(common.REPO_ROOT)),
        "evaluation_sha256": common.sha256_file(output),
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
        "elapsed_seconds": time.perf_counter() - start_time,
        "v4_recomputation": v4_recompute,
        "v4_outcomes_used_for_thresholds_or_selection": False,
    }
    common.write_study_json(manifest_path, result, exclusive=True)
    return result


def _reconstruct_raw_actions(action_windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(action_windows, dtype=np.float32)
    if windows.shape != (38, 3, 25):
        raise ValueError(f"unexpected episode action windows {windows.shape}")
    blocks = np.concatenate([windows[0], windows[1:, -1]], axis=0).reshape(40, 25)
    normalized = blocks.reshape(200, 5)
    return normalized * common.FROZEN_ACTION_STD + common.FROZEN_ACTION_MEAN


RAW_METRIC_NAMES = (
    "pixel_r_mean",
    "pixel_g_mean",
    "pixel_b_mean",
    "pixel_r_std",
    "pixel_g_std",
    "pixel_b_std",
    "pixel_temporal_abs_delta_mean",
    "observation_mean",
    "observation_std",
    "observation_norm_mean",
    "raw_action_abs_ge_0_99_rate",
    "raw_action_rms",
    "raw_action_coord4_mean",
    "normalized_action_abs_gt3_rate",
    "contact_row_rate",
    "impact_row_rate",
    "effector_displacement_mean",
    "block_displacement_mean",
    "block_motion_rate_gt_1e_6",
    "phase_early_rate",
    "phase_middle_rate",
    "phase_late_rate",
    "target_task_cube_rate",
    "final_success",
)


def _raw_metrics(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    pixels = np.asarray(raw["pixels"], dtype=np.uint8)[:: common.FRAMESKIP].astype(np.float64) / 255.0
    flat = pixels.reshape(-1, 3)
    observation = np.asarray(raw["observation"], dtype=np.float64)
    action = np.asarray(raw["action"], dtype=np.float32)[:200]
    normalized = (action - common.FROZEN_ACTION_MEAN) / common.FROZEN_ACTION_STD
    model_steps = np.arange(common.HISTORY, common.MODEL_STEPS, dtype=np.int64)
    physical = _physical(raw, model_steps)
    phase = physical["normalized_phase"]
    task = np.asarray(raw.get("privileged_target_task", np.full(201, "cube"))).astype(str)
    success = np.asarray(raw.get("success", np.zeros(201, dtype=bool))).reshape(201, -1)[:, 0]
    values = [
        *flat.mean(0).tolist(),
        *flat.std(0).tolist(),
        float(np.abs(np.diff(pixels, axis=0)).mean()),
        float(observation.mean()),
        float(observation.std()),
        float(np.linalg.vector_norm(observation, axis=1).mean()),
        float((np.abs(action) >= 0.99).mean()),
        float(np.sqrt(np.square(action.astype(np.float64)).sum(1)).mean()),
        float(action[:, 4].mean()),
        float((np.abs(normalized) > 3.0).mean()),
        float(physical["interaction"].mean()),
        float(physical["impact"].mean()),
        float(physical["effector_disp"].mean()),
        float(physical["block_disp"].mean()),
        float((physical["block_disp"] > 1e-6).mean()),
        float((phase < 1 / 3).mean()),
        float(((phase >= 1 / 3) & (phase < 2 / 3)).mean()),
        float((phase >= 2 / 3).mean()),
        float(np.mean(task == "cube")),
        float(success[-1]),
    ]
    if len(values) != len(RAW_METRIC_NAMES):
        raise RuntimeError("raw metric schema mismatch")
    return np.asarray(values, dtype=np.float64)


def raw_reference() -> dict[str, Any]:
    output = common.STUDY_ROOT / "reference/raw_offline_episode_metrics.npz"
    audit_path = common.STUDY_ROOT / "audit/raw_reference_access.json"
    if output.exists() or audit_path.exists():
        if not (output.exists() and audit_path.exists()):
            raise RuntimeError("partial raw reference artifact")
        prior = common.study_json(audit_path)
        if common.sha256_file(output) != prior["raw_metrics_sha256"]:
            raise RuntimeError("raw reference drift")
        return prior
    discovery, discovery_audit = common.load_allowed_cache("offline_discovery")
    calibration, calibration_audit = common.load_allowed_cache("offline_calibration")
    test = common.v3_test_set()
    roles = []
    ids = []
    metrics = []
    action_alignment_max = 0.0
    datasets_accessed = [
        "ep_offset",
        "ep_len",
        "ep_idx",
        "pixels",
        "observation",
        "action",
        "qpos",
        "qvel",
        "proprio_gripper_contact",
        "proprio_effector_pos",
        "privileged_block_0_pos",
        "privileged_target_block_pos",
        "privileged_target_block_yaw",
        "privileged_target_task",
        "success",
    ]
    started = time.perf_counter()
    with h5py.File(common.SOURCE_H5, "r", swmr=True) as h5:
        for role_code, (role, arrays) in enumerate(
            (("offline_discovery", discovery), ("offline_calibration", calibration))
        ):
            unique = np.unique(arrays["episode_id"]).astype(np.int64)
            if set(unique.tolist()) & test:
                raise RuntimeError("test episode reached raw reference loop")
            for position, episode in enumerate(unique):
                if int(episode) in test:
                    raise RuntimeError("test episode reached HDF5 raw read")
                offset = int(h5["ep_offset"][episode])
                length = int(h5["ep_len"][episode])
                if length != 201 or not np.all(
                    np.asarray(h5["ep_idx"][offset : offset + length]) == episode
                ):
                    raise RuntimeError("offline raw episode layout mismatch")
                raw = {
                    name: np.asarray(h5[name][offset : offset + length])
                    for name in datasets_accessed
                    if name not in {"ep_offset", "ep_len", "ep_idx"}
                }
                row_mask = np.asarray(arrays["episode_id"]) == episode
                cache_actions = _reconstruct_raw_actions(arrays["action"][row_mask])
                source_actions = np.asarray(raw["action"][:200], dtype=np.float32)
                action_alignment_max = max(
                    action_alignment_max,
                    float(np.max(np.abs(cache_actions - source_actions))),
                )
                metrics.append(_raw_metrics(raw))
                ids.append(int(episode))
                roles.append(role_code)
                if (position + 1) % 25 == 0 or position + 1 == len(unique):
                    print(
                        f"raw-reference {role} episode {position + 1}/{len(unique)}",
                        flush=True,
                    )
    arrays_out = {
        "role_code": np.asarray(roles, dtype=np.int8),
        "episode_id": np.asarray(ids, dtype=np.int64),
        "metric_names": np.asarray(RAW_METRIC_NAMES, dtype="U64"),
        "metrics": np.stack(metrics),
    }
    common.atomic_npz(output, arrays_out)
    audit = {
        "schema_version": 1,
        "status": "complete_allowlisted_raw_input_and_posthoc_descriptor_read",
        "raw_metrics_path": str(output.relative_to(common.REPO_ROOT)),
        "raw_metrics_sha256": common.sha256_file(output),
        "episode_count": len(ids),
        "discovery_episode_count": int(np.sum(arrays_out["role_code"] == 0)),
        "calibration_episode_count": int(np.sum(arrays_out["role_code"] == 1)),
        "datasets_accessed": datasets_accessed,
        "targets_supplied_to_models_from_h5": False,
        "raw_h5_access_authorized_only_by_isolated_cache_episode_ids": True,
        "v3_test_episode_intersection": sorted(set(ids) & test),
        "v3_test_targets_opened": False,
        "combined_v3_cache_opened_with_numpy": False,
        "action_reconstruction_vs_h5_max_abs": action_alignment_max,
        "discovery_isolation": discovery_audit,
        "calibration_isolation": calibration_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "contact_impact_state_motion_phase_task_success_posthoc_only": True,
    }
    if audit["v3_test_episode_intersection"] or action_alignment_max > 1e-5:
        raise RuntimeError(f"raw reference isolation/alignment audit failed: {audit}")
    common.write_study_json(audit_path, audit, exclusive=True)
    return audit


def summarize_generated_raw(role: str) -> dict[str, Any]:
    if role not in {"plan_oracle", "markov_oracle", "v4_markov"}:
        raise ValueError("invalid raw summary role")
    common.assert_pre_generation_seal()
    output = common.STUDY_ROOT / "data" / f"{role}_raw_episode_metrics.npz"
    manifest_path = common.STUDY_ROOT / "data" / f"{role}_raw_episode_metrics_manifest.json"
    if output.exists() or manifest_path.exists():
        if not (output.exists() and manifest_path.exists()):
            raise RuntimeError("partial raw summary")
        prior = common.study_json(manifest_path)
        if common.sha256_file(output) != prior["raw_metrics_sha256"]:
            raise RuntimeError("raw summary drift")
        return prior
    if role == "v4_markov":
        raw_manifest_path = common.V4_ROOT / "data/confirmation_data_manifest.json"
        source = common.study_json(raw_manifest_path)
        records = source["episodes"]
        diagnostic_only = True
    else:
        raw_manifest_path = common.role_raw_paths("main", role)[1]
        source = common.study_json(raw_manifest_path)
        records = source["episodes"]
        diagnostic_only = False
    values = []
    ids = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        path = common.REPO_ROOT / record["path"]
        if common.sha256_file(path) != record["file_sha256"]:
            raise RuntimeError("raw summary input drift")
        with np.load(path, allow_pickle=False) as stored:
            needed = {
                name: stored[name]
                for name in (
                    "pixels",
                    "observation",
                    "action",
                    "proprio_gripper_contact",
                    "proprio_effector_pos",
                    "privileged_block_0_pos",
                    "privileged_target_task",
                    "success",
                )
                if name in stored.files
            }
        values.append(_raw_metrics(needed))
        ids.append(int(record["slot"]))
        if (index + 1) % 20 == 0 or index + 1 == len(records):
            print(f"raw-summary {role} episode {index + 1}/{len(records)}", flush=True)
    arrays_out = {
        "episode_id": np.asarray(ids, dtype=np.int64),
        "metric_names": np.asarray(RAW_METRIC_NAMES, dtype="U64"),
        "metrics": np.stack(values),
    }
    common.atomic_npz(output, arrays_out)
    result = {
        **common.dataset_manifest_base(role=role, episodes=len(ids), rows=len(ids)),
        "status": "complete_raw_input_and_posthoc_composition_summary",
        "raw_metrics_path": str(output.relative_to(common.REPO_ROOT)),
        "raw_metrics_sha256": common.sha256_file(output),
        "source_manifest": str(raw_manifest_path.relative_to(common.REPO_ROOT)),
        "source_manifest_sha256": common.sha256_file(raw_manifest_path),
        "posthoc_labels_never_used_for_primary_adaptive_decision": True,
        "diagnostic_only": diagnostic_only,
        "elapsed_seconds": time.perf_counter() - started,
    }
    common.write_study_json(manifest_path, result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    encode = sub.add_parser("encode")
    encode.add_argument("policy", choices=("plan_oracle", "markov_oracle"))
    encode.add_argument("--device", default="auto")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("role", choices=EVAL_ROLES)
    evaluate.add_argument("--device", default="auto")
    sub.add_parser("raw-reference")
    raw = sub.add_parser("summarize-raw")
    raw.add_argument("role", choices=("plan_oracle", "markov_oracle", "v4_markov"))
    args = parser.parse_args()
    if args.command == "encode":
        result = encode_fresh(args.policy, args.device)
    elif args.command == "evaluate":
        result = evaluate_role(args.role, args.device)
    elif args.command == "raw-reference":
        result = raw_reference()
    else:
        result = summarize_generated_raw(args.role)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

