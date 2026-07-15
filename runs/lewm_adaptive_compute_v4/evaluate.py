#!/usr/bin/env python3
"""Encode and evaluate a complete V4 role without exposing partial metrics."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import common
import runtime


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"one-shot output already exists: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _raw_manifest(role: str) -> dict[str, Any]:
    _, path = common.role_paths(role)
    payload = common.read_json(path)
    expected = 12 if role == "smoke" else 300
    if payload.get("role") != role or not payload.get("complete") or payload.get("episode_count") != expected:
        raise RuntimeError(f"invalid/incomplete {role} raw manifest")
    return payload


def _episode_physical(raw: Mapping[str, np.ndarray], model_steps: np.ndarray) -> dict[str, np.ndarray]:
    contact_raw = np.asarray(raw["proprio_gripper_contact"], dtype=np.float64).reshape(
        common.RAW_EPISODE_ROWS, -1
    )[:, 0] > 1e-9
    effector = np.asarray(raw["proprio_effector_pos"], dtype=np.float64)
    block = np.asarray(raw["privileged_block_0_pos"], dtype=np.float64)
    actions = np.asarray(raw["action"], dtype=np.float32)
    interaction = np.empty(len(model_steps), dtype=np.bool_)
    impact = np.empty(len(model_steps), dtype=np.bool_)
    effector_disp = np.empty(len(model_steps), dtype=np.float32)
    block_disp = np.empty(len(model_steps), dtype=np.float32)
    action_magnitude = np.empty(len(model_steps), dtype=np.float32)
    phase = np.asarray(model_steps, dtype=np.float32) / 40.0
    for index, step in enumerate(model_steps.astype(np.int64)):
        target = int(step * common.FRAMESKIP)
        previous = target - common.FRAMESKIP
        current_contact = bool(contact_raw[previous + 1 : target + 1].any())
        prior_start = max(0, previous - common.FRAMESKIP + 1)
        prior_contact = bool(contact_raw[prior_start : previous + 1].any())
        interaction[index] = current_contact
        impact[index] = current_contact and not prior_contact
        effector_disp[index] = np.linalg.norm(effector[target] - effector[previous])
        block_disp[index] = np.linalg.norm(block[target] - block[previous])
        action_magnitude[index] = np.sqrt(
            np.square(actions[previous:target].astype(np.float64)).sum(1).mean()
        )
    regime = np.full(len(model_steps), 3, dtype=np.int8)  # other_free_motion
    regime[~interaction & (block_disp <= 0.0)] = 2  # static
    regime[interaction & ~impact] = 1  # contact
    regime[impact] = 0
    return {
        "interaction": interaction,
        "impact": impact,
        "effector_disp": effector_disp,
        "block_disp": block_disp,
        "action_magnitude": action_magnitude,
        "normalized_phase": phase,
        "regime_code": regime,
    }


@torch.inference_mode()
def encode_role(role: str, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if role == "confirmation":
        receipt = common.ROOT / "audit/confirmation_access_receipt.json"
        if not receipt.exists():
            raise RuntimeError("confirmation target/next-latent access requires the one-shot receipt")
        status = common.read_json(receipt).get("status")
        if status != "consumed_before_any_confirmatory_target_or_next_latent_io":
            raise RuntimeError("invalid confirmation access receipt")
    manifest = _raw_manifest(role)
    base_model, contract, base_provenance = runtime.load_base_model(device)
    before = runtime.module_audit(base_model)
    if not before["passed"]:
        raise RuntimeError("base model was not frozen before encoding")
    model_io = runtime.load_model_io()
    histories: list[np.ndarray] = []
    action_histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_slots: list[np.ndarray] = []
    env_seeds: list[np.ndarray] = []
    policy_seeds: list[np.ndarray] = []
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
            "regime_code",
        )
    }
    for position, record in enumerate(manifest["episodes"]):
        path = common.REPO / record["path"]
        if common.sha256_file(path) != record["file_sha256"]:
            raise RuntimeError(f"raw episode hash drift: {path}")
        with np.load(path, allow_pickle=False) as stored:
            raw = {name: stored[name] for name in stored.files}
        if str(raw["role"][0]) != role or int(raw["slot"][0]) != int(record["slot"]):
            raise RuntimeError("raw role/slot allowlist violation")
        pixels = raw["pixels"][:: common.FRAMESKIP]
        chunks = []
        for start in range(0, len(pixels), 64):
            batch = model_io.pixel_transform(pixels[start : start + 64], contract.image_size, device)
            encoded = base_model.encode({"pixels": batch.unsqueeze(0)})["emb"].squeeze(0)
            chunks.append(encoded.detach().cpu())
        latents = torch.cat(chunks).numpy().astype(np.float32)
        if latents.shape != (common.MODEL_STEPS, common.LATENT_DIM):
            raise RuntimeError(f"unexpected encoded latent shape {latents.shape}")
        raw_actions = np.asarray(raw["action"][:-1], dtype=np.float32)
        if raw_actions.shape != (200, common.RAW_ACTION_DIM) or not np.isfinite(raw_actions).all():
            raise RuntimeError("invalid raw action sequence")
        normalized = (raw_actions - common.FROZEN_ACTION_MEAN) / common.FROZEN_ACTION_STD
        blocks = normalized.reshape(common.MODEL_STEPS - 1, common.BLOCKED_ACTION_DIM).astype(np.float32)
        n = common.EXAMPLES_PER_EPISODE
        history = np.stack([latents[index : index + common.HISTORY] for index in range(n)]).astype(np.float32)
        action = np.stack([blocks[index : index + common.HISTORY] for index in range(n)]).astype(np.float32)
        target = latents[common.HISTORY :].astype(np.float32)
        model_steps = np.arange(common.HISTORY, common.MODEL_STEPS, dtype=np.int64)
        histories.append(history)
        action_histories.append(action)
        targets.append(target)
        episode_slots.append(np.full(n, int(record["slot"]), dtype=np.int32))
        env_seeds.append(np.full(n, int(record["env_seed"]), dtype=np.int64))
        policy_seeds.append(np.full(n, int(record["policy_seed"]), dtype=np.int64))
        steps.append(model_steps)
        episode_labels = _episode_physical(raw, model_steps)
        for name, value in episode_labels.items():
            physical[name].append(value)
        if (position + 1) % (1 if role == "smoke" else 10) == 0:
            print(f"encoded {role} episode {position + 1}/{len(manifest['episodes'])}", flush=True)
    history_all = np.concatenate(histories)
    action_all = np.concatenate(action_histories)
    target_all = np.concatenate(targets)
    predictions = []
    for part in common.iter_batches(np.arange(len(history_all)), 1024):
        history = torch.as_tensor(np.ascontiguousarray(history_all[part]), device=device)
        action = torch.as_tensor(np.ascontiguousarray(action_all[part]), device=device)
        predictions.append(runtime.base_predict(base_model, history, action).cpu().numpy())
    runtime.synchronize(device)
    base_pred = np.concatenate(predictions).astype(np.float32)
    after = runtime.module_audit(base_model)
    arrays = {
        "history": history_all,
        "action": action_all,
        "base_pred": base_pred,
        "target": target_all,
        "episode_slot": np.concatenate(episode_slots),
        "env_seed": np.concatenate(env_seeds),
        "policy_seed": np.concatenate(policy_seeds),
        "model_step": np.concatenate(steps),
        **{name: np.concatenate(values) for name, values in physical.items()},
    }
    expected_rows = (12 if role == "smoke" else 300) * common.EXAMPLES_PER_EPISODE
    if any(len(value) != expected_rows for value in arrays.values()):
        raise RuntimeError("encoded array row-count mismatch")
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RuntimeError(f"nonfinite encoded array {name}")
    audit = {
        "role": role,
        "episodes": len(manifest["episodes"]),
        "rows": expected_rows,
        "model_contract": asdict(contract),
        "base_provenance": base_provenance,
        "base_module_before": before,
        "base_module_after": after,
        "base_state_unchanged": before == after,
        "action_normalization": "frozen_released_training_mean_std",
        "physical_labels_attached_during_cache_construction_but_not_passed_to_solver_or_gate": True,
        "v3_test_targets_opened": False,
    }
    if not before["passed"] or not after["passed"] or before != after:
        raise RuntimeError("base-model preservation audit failed")
    return arrays, audit


def whitened_losses(target: np.ndarray, exits: np.ndarray) -> np.ndarray:
    whitening = runtime.load_whitening()
    difference = np.asarray(exits, dtype=np.float64) - np.asarray(target, dtype=np.float64)[:, None, :]
    transformed = np.einsum("nkd,df->nkf", difference, whitening["matrix"], optimize=False)
    result = np.square(transformed).mean(2)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite whitened loss")
    return result


@torch.inference_mode()
def evaluate_arrays(
    arrays: Mapping[str, np.ndarray], role: str, device: torch.device
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    solver, v1, models = runtime.load_solver(device)
    gate = runtime.load_gate(device)
    before = runtime.module_audit(solver, v1, gate.model)
    n = len(arrays["target"])
    exits = np.empty((n, 4, common.LATENT_DIM), dtype=np.float32)
    scores = np.empty((n, 3), dtype=np.float64)
    calls = np.empty(n, dtype=np.int64)
    sparse_reference_output = np.empty((n, common.LATENT_DIM), dtype=np.float32)
    sparse_reference_calls = np.empty(n, dtype=np.int64)
    sparse_optimized_output = np.empty_like(sparse_reference_output)
    sparse_optimized_calls = np.empty(n, dtype=np.int64)
    feature_finite = True
    feature_requires_grad = False
    d0_identity = d0_bitwise = d1_bitwise = True
    for part in common.iter_batches(np.arange(n), 1024):
        history = torch.as_tensor(np.ascontiguousarray(arrays["history"][part]), device=device)
        action = torch.as_tensor(np.ascontiguousarray(arrays["action"][part]), device=device)
        base = torch.as_tensor(np.ascontiguousarray(arrays["base_pred"][part]), device=device)
        output, updates = solver(history, action, base, max_depth=4, return_updates=True)
        expected = v1(history, action, base, depths=(0, 1))
        d0_identity &= output[0] is base
        d0_bitwise &= bool(torch.equal(output[0], expected[0]))
        d1_bitwise &= bool(torch.equal(output[1], expected[1]))
        dense_selected, dense_calls, dense_scores, dense_features = runtime.dense_adaptive(
            solver, gate, models, history, action, base
        )
        reference, reference_calls = runtime.sparse_adaptive_reference(
            solver, gate, models, history, action, base
        )
        optimized, optimized_calls = runtime.sparse_adaptive_optimized(
            solver, gate, models, history, action, base
        )
        exits[part] = torch.stack([output[depth] for depth in (1, 2, 3, 4)], 1).cpu().numpy()
        scores[part] = dense_scores.cpu().numpy().astype(np.float64)
        calls[part] = dense_calls.cpu().numpy()
        sparse_reference_output[part] = reference.cpu().numpy()
        sparse_reference_calls[part] = reference_calls.cpu().numpy()
        sparse_optimized_output[part] = optimized.cpu().numpy()
        sparse_optimized_calls[part] = optimized_calls.cpu().numpy()
        feature_finite &= bool(torch.isfinite(dense_features).all())
        feature_requires_grad |= bool(dense_features.requires_grad)
    runtime.synchronize(device)
    target = np.asarray(arrays["target"], dtype=np.float32)
    losses = np.square(exits - target[:, None, :]).mean(2).astype(np.float64)
    white = whitened_losses(target, exits)
    selected_dense = exits[np.arange(n), calls - 1]
    recomputed_calls = runtime.sequential_calls_numpy(scores, common.COMPUTE_PRICE)
    after = runtime.module_audit(solver, v1, gate.model)
    sparse = {
        "dense_vs_reference_calls_exact": bool(np.array_equal(calls, sparse_reference_calls)),
        "dense_vs_optimized_calls_exact": bool(np.array_equal(calls, sparse_optimized_calls)),
        "reference_vs_optimized_calls_exact": bool(
            np.array_equal(sparse_reference_calls, sparse_optimized_calls)
        ),
        "calls_match_independent_numpy_policy": bool(np.array_equal(calls, recomputed_calls)),
        "dense_vs_reference_output_max_abs": float(
            np.max(np.abs(selected_dense - sparse_reference_output))
        ),
        "dense_vs_optimized_output_max_abs": float(
            np.max(np.abs(selected_dense - sparse_optimized_output))
        ),
        "reference_vs_optimized_output_max_abs": float(
            np.max(np.abs(sparse_reference_output - sparse_optimized_output))
        ),
        "dense_vs_reference_outputs_equivalent": bool(
            np.allclose(selected_dense, sparse_reference_output, rtol=2e-5, atol=2e-6)
        ),
        "dense_vs_optimized_outputs_equivalent": bool(
            np.allclose(selected_dense, sparse_optimized_output, rtol=2e-5, atol=2e-6)
        ),
    }
    sparse["passed"] = bool(
        all(
            sparse[name]
            for name in (
                "dense_vs_reference_calls_exact",
                "dense_vs_optimized_calls_exact",
                "reference_vs_optimized_calls_exact",
                "calls_match_independent_numpy_policy",
                "dense_vs_reference_outputs_equivalent",
                "dense_vs_optimized_outputs_equivalent",
            )
        )
    )
    audit = {
        "role": role,
        "rows": n,
        "finite_losses": bool(np.isfinite(losses).all() and np.isfinite(white).all()),
        "finite_scores": bool(np.isfinite(scores).all()),
        "exact_supported_calls": bool(np.isin(calls, common.SUPPORTED_CALLS).all()),
        "d0_identity": d0_identity,
        "d0_bitwise": d0_bitwise,
        "d1_bitwise": d1_bitwise,
        "causal_feature_signature_excludes_target": True,
        "causal_features_finite": feature_finite,
        "causal_features_require_grad": feature_requires_grad,
        "solver_gate_module_before": before,
        "solver_gate_module_after": after,
        "solver_gate_state_unchanged": before == after,
        "gate_metadata": gate.metadata,
        "gate_parameters": common.GATE_PARAMETERS,
        "gate_flops_per_evaluated_decision": common.GATE_FLOPS_PER_DECISION,
        "compute_price": common.COMPUTE_PRICE,
        "sparse_equivalence": sparse,
        "no_gradient_audit": bool(before["passed"] and after["passed"]),
        "v3_test_targets_opened": False,
    }
    audit["passed"] = bool(
        audit["finite_losses"]
        and audit["finite_scores"]
        and audit["exact_supported_calls"]
        and audit["d0_identity"]
        and audit["d0_bitwise"]
        and audit["d1_bitwise"]
        and audit["causal_features_finite"]
        and not audit["causal_features_require_grad"]
        and audit["solver_gate_state_unchanged"]
        and audit["no_gradient_audit"]
        and sparse["passed"]
    )
    if not audit["passed"]:
        raise RuntimeError(f"frozen evaluation validity audit failed: {audit}")
    outcome = {
        "losses": losses,
        "whitened_losses": white,
        "scores": scores,
        "calls": calls,
        "selected_output": selected_dense,
        **{
            name: np.asarray(arrays[name])
            for name in (
                "episode_slot",
                "env_seed",
                "policy_seed",
                "model_step",
                "interaction",
                "impact",
                "effector_disp",
                "block_disp",
                "action_magnitude",
                "normalized_phase",
                "regime_code",
            )
        },
    }
    return outcome, audit


def run(role: str, device: torch.device) -> dict[str, Any]:
    if role == "smoke":
        common.verify_seal(common.ROOT / "audit/phase0_seal.json")
    else:
        common.verify_seal(common.ROOT / "audit/pre_confirmation_seal.json")
        if not (common.ROOT / "audit/freshness_duplicate_audit.json").exists():
            raise RuntimeError("confirmation evaluation requires the passed freshness audit")
        if not common.read_json(common.ROOT / "audit/freshness_duplicate_audit.json")["passed"]:
            raise RuntimeError("freshness audit did not pass")
    encoded_path = common.ROOT / "data" / f"{role}_encoded.npz"
    encoded_manifest_path = common.ROOT / "data" / f"{role}_encoded_manifest.json"
    outcome_path = common.ROOT / "data" / f"{role}_outcome_once.npz"
    outcome_manifest_path = common.ROOT / "data" / f"{role}_outcome_manifest.json"
    if any(path.exists() for path in (encoded_path, encoded_manifest_path, outcome_path, outcome_manifest_path)):
        raise RuntimeError(f"{role} encoding/evaluation has already been consumed")
    arrays, encoding_audit = encode_role(role, device)
    _atomic_npz(encoded_path, arrays)
    encoded_manifest = {
        "schema_version": 1,
        "role": role,
        "cache_path": str(encoded_path.relative_to(common.REPO)),
        "cache_sha256": common.sha256_file(encoded_path),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": common.array_sha256(value),
            }
            for name, value in arrays.items()
        },
        "audit": encoding_audit,
    }
    common.write_json(encoded_manifest_path, encoded_manifest, exclusive=True)
    outcome, evaluation_audit = evaluate_arrays(arrays, role, device)
    _atomic_npz(outcome_path, outcome)
    outcome_manifest = {
        "schema_version": 1,
        "role": role,
        "status": "complete_frozen_policy_evaluated_once",
        "target_metrics_not_printed_during_run": True,
        "outcome_path": str(outcome_path.relative_to(common.REPO)),
        "outcome_sha256": common.sha256_file(outcome_path),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": common.array_sha256(value),
            }
            for name, value in outcome.items()
        },
        "encoded_manifest_sha256": common.sha256_file(encoded_manifest_path),
        "evaluation_audit": evaluation_audit,
        "v3_test_targets_opened": False,
    }
    common.write_json(outcome_manifest_path, outcome_manifest, exclusive=True)
    common.file_mode_read_only(encoded_path)
    common.file_mode_read_only(encoded_manifest_path)
    common.file_mode_read_only(outcome_path)
    common.file_mode_read_only(outcome_manifest_path)
    if role == "smoke":
        completion = {
            "schema_version": 1,
            "status": "smoke_complete_and_forever_excluded_from_confirmation",
            "episodes": 12,
            "rows": 12 * common.EXAMPLES_PER_EPISODE,
            "evaluation_passed": True,
            "outcome_manifest_sha256": common.sha256_file(outcome_manifest_path),
            "may_influence_scientific_choices": False,
        }
        common.write_json(common.ROOT / "audit/smoke_completion.json", completion, exclusive=True)
    return {
        "role": role,
        "episodes": 12 if role == "smoke" else 300,
        "rows": len(outcome["calls"]),
        "evaluation_passed": evaluation_audit["passed"],
        "metrics_withheld": role == "confirmation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("smoke", "confirmation"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run(args.role, runtime.choose_device(args.device))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
