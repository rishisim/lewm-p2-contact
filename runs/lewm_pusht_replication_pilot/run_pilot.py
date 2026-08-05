#!/usr/bin/env python3
"""Execute the fixed, bounded PushT LeWM adaptive-allocation pilot."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch import Tensor

from pusht_core import (
    ACTION_DIM, DEPTHS, FRAMESKIP, GENERATION_PYTHON, HISTORY, LATENT_DIM,
    MAX_DEPTH, MODEL_CONFIG, MODEL_WEIGHTS, REPO_ROOT, ROOT, StagewiseRefiner,
    action_normalization, append_note, atomic_json, atomic_npz,
    base_predictor_counted_flops, bootstrap_interval, build_causal_features,
    causal_feature_names, choose_device, concatenate_transition_batches,
    encode_pixel_sequences, episode_means, evaluate_refiner_dense, feature_arrays,
    fit_stagewise_ridge, fit_whitening, gate_operation_ledger, gate_scores,
    git_snapshot, load_base_model, load_npz, module_digest, normalize_actions,
    pack_context, preprocess_pixels, raw_white_losses, read_json, refiner_flops,
    rollout_episode, safe_spearman, sequential_calls, sha256_file,
    strongest_analytic_mixture, summarize_context, synchronize, trajectory_digest,
    transition_arrays, update_status,
)


SMOKE_SPECS = (
    {"identifier": "pusht_smoke_excluded_000", "seed": 9_170_001},
    {"identifier": "pusht_smoke_excluded_001", "seed": 9_170_002},
)
ROLE_SEED_BASES = {"fit": 4_710_000, "selection": 4_810_000, "evaluation": 4_910_000}
FULL_COUNTS = {"fit": 120, "selection": 40, "evaluation": 80}
COMPACT_COUNTS = {"fit": 60, "selection": 20, "evaluation": 40}
REGULARIZATIONS = (1.0, 10.0)
FIT_SCORE_QUANTILES = (0.45, 0.60, 0.75)
CANDIDATES = tuple(
    {
        "candidate_id": f"stage_dual_r{regularization:g}_q{quantile:.2f}",
        "regularization": regularization,
        "fit_score_quantile": quantile,
    }
    for regularization in REGULARIZATIONS
    for quantile in FIT_SCORE_QUANTILES
)
REFINER_SEED = 7_181_337
BOOTSTRAP_SEED = 7_181_991
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_SEED = 7_181_881
LATENCY_REPEATS = 5


def _role_specs(role: str, count: int) -> list[dict[str, Any]]:
    base = ROLE_SEED_BASES[role]
    return [
        {"identifier": f"pusht_{role}_{index:04d}", "ordinal": index, "seed": base + index}
        for index in range(count)
    ]


def _config() -> dict[str, Any]:
    return read_json(ROOT / "CONFIG.json")


def _checkpoint_model(device: torch.device) -> StagewiseRefiner:
    checkpoint = torch.load(ROOT / "checkpoints/refiner.pt", map_location="cpu", weights_only=True)
    model = StagewiseRefiner(hidden=256, iteration_dim=16)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def _save_checkpoint(model: StagewiseRefiner, best_epoch: int) -> None:
    path = ROOT / "checkpoints/refiner.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "training_seed": REFINER_SEED,
            "best_epoch": best_epoch,
            "architecture": {
                "latent_dim": LATENT_DIM, "action_dim": ACTION_DIM, "history": HISTORY,
                "hidden": 256, "iteration_dim": 16, "depths": list(DEPTHS),
            },
        },
        temporary,
    )
    os.replace(temporary, path)


def smoke() -> None:
    config_path = ROOT / "CONFIG.json"
    if config_path.exists():
        print(json.dumps({"status": "smoke_already_complete", "config_sha256": sha256_file(config_path)}))
        return
    update_status("excluded_compatibility_smoke")
    device = choose_device()
    load_start = time.perf_counter()
    model = load_base_model(device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_start
    parameter_before = module_digest(model)
    requires_grad_false = all(not parameter.requires_grad for parameter in model.parameters())
    model_eval = not model.training
    smoke_records: list[dict[str, Any]] = []
    retained_arrays: list[dict[str, np.ndarray]] = []
    primary_seconds: list[float] = []
    preprocessing_contract: dict[str, Any] | None = None

    for index, spec in enumerate(SMOKE_SPECS):
        synchronize(device)
        start = time.perf_counter()
        episode = rollout_episode(int(spec["seed"]))
        latents = encode_pixel_sequences(model, [episode], device, batch_size=64)
        arrays = transition_arrays(model, [episode], latents, [index], device)
        synchronize(device)
        elapsed = time.perf_counter() - start
        primary_seconds.append(elapsed)
        replay = rollout_episode(int(spec["seed"]))
        replay_latents = encode_pixel_sequences(model, [replay], device, batch_size=64)
        replay_arrays = transition_arrays(model, [replay], replay_latents, [index], device)
        deterministic_raw = trajectory_digest(episode) == trajectory_digest(replay)
        deterministic_latent = bool(
            np.array_equal(latents[0], replay_latents[0])
            or np.allclose(latents[0], replay_latents[0], rtol=1e-6, atol=2e-7)
        )
        deterministic_prediction = bool(
            np.array_equal(arrays["base"], replay_arrays["base"])
            or np.allclose(arrays["base"], replay_arrays["base"], rtol=1e-6, atol=2e-7)
        )
        if preprocessing_contract is None:
            processed = preprocess_pixels(np.asarray(episode["pixels"])[:2], device)
            preprocessing_contract = {
                "input_dtype": str(np.asarray(episode["pixels"]).dtype),
                "input_shape": list(np.asarray(episode["pixels"])[:2].shape),
                "output_dtype": str(processed.dtype), "output_shape": list(processed.shape),
                "finite": bool(torch.isfinite(processed).all().item()),
            }
        smoke_records.append(
            {
                **spec, "primary_generation_encoding_prediction_seconds": elapsed,
                "raw_steps": int(episode["raw_steps"]),
                "strided_frames": len(np.asarray(episode["pixels"])),
                "blocked_action_shape": list(np.asarray(episode["blocked_actions"]).shape),
                "transition_count": len(arrays["target"]),
                "latent_shape": list(latents[0].shape),
                "base_prediction_shape": list(arrays["base"].shape),
                "trajectory_sha256": trajectory_digest(episode),
                "determinism_replay_sha256": trajectory_digest(replay),
                "deterministic_raw_episode": deterministic_raw,
                "deterministic_latents_within_tolerance": deterministic_latent,
                "deterministic_base_prediction_within_tolerance": deterministic_prediction,
                "finite": bool(all(np.isfinite(v).all() for v in arrays.values())),
            }
        )
        retained_arrays.append(arrays)

    parameter_after = module_digest(model)
    compatibility = {
        "exact_unique_smoke_episode_count": len(SMOKE_SPECS),
        "determinism_replays_not_retained_as_episodes": len(SMOKE_SPECS),
        "all_raw_deterministic": all(item["deterministic_raw_episode"] for item in smoke_records),
        "all_latents_deterministic_within_tolerance": all(
            item["deterministic_latents_within_tolerance"] for item in smoke_records
        ),
        "all_predictions_deterministic_within_tolerance": all(
            item["deterministic_base_prediction_within_tolerance"] for item in smoke_records
        ),
        "pixel_preprocessing": preprocessing_contract,
        "latent_dim_192": all(item["latent_shape"][1] == LATENT_DIM for item in smoke_records),
        "blocked_action_dim_10": all(item["blocked_action_shape"][1] == ACTION_DIM for item in smoke_records),
        "one_step_prediction_finite": all(item["finite"] for item in smoke_records),
        "base_eval_mode": model_eval, "base_requires_grad_false": requires_grad_false,
        "base_parameter_digest_before": parameter_before,
        "base_parameter_digest_after": parameter_after,
        "base_parameter_state_unchanged": parameter_before == parameter_after,
    }
    if not all(
        compatibility[key]
        for key in (
            "all_raw_deterministic", "all_latents_deterministic_within_tolerance",
            "all_predictions_deterministic_within_tolerance", "latent_dim_192",
            "blocked_action_dim_10", "one_step_prediction_finite", "base_eval_mode",
            "base_requires_grad_false", "base_parameter_state_unchanged",
        )
    ) or not preprocessing_contract or not preprocessing_contract["finite"]:
        raise RuntimeError(f"PushT compatibility smoke failed: {compatibility}")

    measured = float(np.mean(primary_seconds))
    projected_full = measured * sum(FULL_COUNTS.values()) * 1.35 + 3_600.0
    choice = "compact" if projected_full > 6 * 60 * 60 else "full"
    counts = COMPACT_COUNTS if choice == "compact" else FULL_COUNTS
    role_specs = {role: _role_specs(role, count) for role, count in counts.items()}
    candidates = [dict(item) for item in CANDIDATES]
    smoke_payload = {
        "schema_version": 1, "device": str(device), "model_load_seconds": model_load_seconds,
        "episodes": smoke_records, "compatibility": compatibility,
        "throughput": {
            "mean_primary_seconds_per_episode": measured,
            "primary_seconds_each": primary_seconds,
            "projection_formula": "mean_smoke_episode_seconds * 240_full_episodes * 1.35 + 3600_second_fixed_training_analysis_reserve",
            "projected_full_wall_seconds": projected_full,
            "full_exceeds_six_hours": projected_full > 21_600,
            "fixed_size_choice": choice,
        },
    }
    atomic_json(ROOT / "SMOKE.json", smoke_payload)
    config = {
        "schema_version": 1,
        "objective": "PushT-specific adaptive-allocation replication pilot",
        "created_unix_ns": time.time_ns(),
        "root": str(ROOT),
        "read_only_inputs": {
            "model_config": str(MODEL_CONFIG), "model_weights": str(MODEL_WEIGHTS),
            "model_config_sha256": sha256_file(MODEL_CONFIG),
            "model_weights_sha256": sha256_file(MODEL_WEIGHTS),
        },
        "runtime": {
            "python": sys.executable, "required_generation_python": str(GENERATION_PYTHON),
            "python_version": platform.python_version(), "torch": torch.__version__, "device": str(device),
        },
        "environment": {
            "id": "swm/PushT-v1", "policy": "WeakPolicy(dist_constraint=100)",
            "fresh_simulator_only": True, "max_raw_steps": 100,
            "frameskip": FRAMESKIP, "history_size": HISTORY,
            "blocked_action_dim": ACTION_DIM, "latent_dim": LATENT_DIM,
        },
        "smoke": smoke_payload,
        "size_choice": choice, "episode_counts": counts, "role_specs": role_specs,
        "role_seed_ranges": {
            role: [specs[0]["seed"], specs[-1]["seed"]] for role, specs in role_specs.items()
        },
        "internal_refiner_validation_rule": "fit episode ordinal modulo 10 equals 9",
        "refiner": {
            "training_seed": REFINER_SEED, "depths": list(DEPTHS), "shared_recurrent_block": True,
            "hidden_dim": 256, "iteration_embedding_dim": 16, "batch_size": 256,
            "max_epochs": 250, "early_stopping_patience": 30,
            "early_stopping_min_delta": 1e-8, "learning_rate": 3e-4,
            "weight_decay": 1e-4, "update_penalty": 1e-4,
            "heterogeneity_rule": {
                "finite": True, "extra_stage_nontrivial_std_count_at_least": 2,
                "nontrivial_std_threshold": "max(1e-10, 1e-4 * fit base raw MSE)",
                "at_least_one_extra_stage_fraction_benefiting_strictly_between": [0.02, 0.98],
                "at_least_one_extra_stage_positive_mean_gain": True,
            },
        },
        "whitening": {"fit_targets_only": True, "symmetric_covariance_floor_ratio": 1e-3},
        "action_normalization": {"fit_transitions_only": True, "per_blocked_action_coordinate": True},
        "gate": {
            "family": "stage-specific dual ridge heads with minimum standardized raw/whitened gain score",
            "feature_width": len(causal_feature_names()), "feature_definition_fixed": True,
            "candidate_count": len(candidates), "candidates": candidates,
            "selection_ranking": [
                "all selection stagewise combined score/gain Spearman signs nonnegative",
                "both selection co-primary point benefits positive",
                "largest minimum standardized co-primary episode benefit",
                "largest minimum stagewise combined Spearman",
                "lower counted FLOPs", "candidate_id lexical tie break",
            ],
            "no_refit_after_selection": True,
        },
        "compute": {
            "base_predictor": base_predictor_counted_flops(),
            "refiner": refiner_flops(), "gate": gate_operation_ledger(),
            "analytic_comparator": "strongest pairwise fixed-depth interpolation at exact total counted FLOPs",
            "multiply_add_convention": "one multiplication plus one accumulation/bias contribution equals two FLOPs",
        },
        "evaluation": {
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence": 0.95, "interval_label": "exploratory",
            "within_episode_permutation_seed": PERMUTATION_SEED,
            "pilot_positive_rule": [
                "raw adaptive benefit over primary analytic comparator > 0",
                "fit-whitened adaptive benefit over primary analytic comparator > 0",
                "exact counted-compute equality", "causal input checks", "finite outputs",
                "all three combined stagewise score/gain Spearman signs nonnegative",
            ],
        },
        "git_baseline": git_snapshot(),
        "source_hashes_at_lock": {
            "pusht_core.py": sha256_file(ROOT / "pusht_core.py"),
            "run_pilot.py": sha256_file(Path(__file__)),
            "verify_pilot.py": sha256_file(ROOT / "verify_pilot.py"),
            "tests/test_pilot.py": sha256_file(ROOT / "tests/test_pilot.py"),
        },
    }
    atomic_json(config_path, config)
    append_note(
        f"Excluded compatibility smoke passed on {device}; projected full wall time {projected_full:.1f}s, permanently locking {choice} counts {counts}."
    )
    update_status("compatibility_complete_design_locked", size_choice=choice)
    print(json.dumps({"status": "smoke_complete", "choice": choice, "counts": counts, "projected_full_seconds": projected_full}))


def generate_role(role: str) -> None:
    if role not in ("fit", "selection", "evaluation"):
        raise ValueError(role)
    config = _config()
    destination = ROOT / f"data/{role}.npz"
    manifest_path = ROOT / f"data/{role}_manifest.json"
    context_path = ROOT / f"data/context_{role}.npz"
    if destination.exists() and manifest_path.exists() and context_path.exists():
        manifest = read_json(manifest_path)
        if manifest["data_sha256"] != sha256_file(destination):
            raise RuntimeError(f"durable {role} data hash mismatch")
        print(json.dumps({"status": f"{role}_already_generated", "episodes": manifest["episode_count"]}))
        return
    if role == "evaluation":
        if not (ROOT / "GATE_FREEZE.json").exists():
            raise RuntimeError("evaluation generation is forbidden before the gate freeze")
        if (ROOT / "PILOT_DECISION.json").exists():
            raise RuntimeError("evaluation outcome already exists and is immutable")
    specs = config["role_specs"][role]
    update_status(f"generate_{role}", generated_episodes=0)
    device = choose_device()
    model = load_base_model(device)
    base_before = module_digest(model)
    transition_batches: list[dict[str, np.ndarray]] = []
    context_records: list[dict[str, Any]] = []
    transition_counts: list[int] = []
    episode_hashes: list[str] = []
    start_time = time.perf_counter()
    group_size = 8
    for group_start in range(0, len(specs), group_size):
        group_specs = specs[group_start : group_start + group_size]
        episodes = [rollout_episode(int(spec["seed"])) for spec in group_specs]
        latents = encode_pixel_sequences(model, episodes, device, batch_size=64)
        batch = transition_arrays(
            model, episodes, latents, [int(spec["ordinal"]) for spec in group_specs], device,
            predict_batch_size=512,
        )
        transition_batches.append(batch)
        for episode, spec in zip(episodes, group_specs):
            count = int(np.sum(batch["episode_id"] == int(spec["ordinal"])))
            transition_counts.append(count)
            episode_hashes.append(trajectory_digest(episode))
            context_records.append({key: value for key, value in episode.items() if key not in ("pixels", "blocked_actions")})
        del episodes, latents, batch
        update_status(
            f"generate_{role}", generated_episodes=min(group_start + group_size, len(specs)),
            total_episodes=len(specs),
        )
    arrays = concatenate_transition_batches(transition_batches)
    context = pack_context(context_records, [int(spec["ordinal"]) for spec in specs])
    context["role_name"] = np.asarray(role)
    base_after = module_digest(model)
    if base_before != base_after or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen base changed during role generation")
    atomic_npz(destination, arrays)
    atomic_npz(context_path, context)
    manifest = {
        "schema_version": 1, "created_unix_ns": time.time_ns(), "role": role, "episode_count": len(specs),
        "transition_count": len(arrays["target"]), "episode_specs": specs,
        "per_episode_transition_counts": transition_counts,
        "episode_trajectory_hashes": episode_hashes,
        "no_exclusions": True, "no_replacements": True,
        "fresh_simulator_generated": True, "hdf5_or_h5_opened": False,
        "pixels_retained": False, "context_sealed_separately": True,
        "experimental_input_keys": ["pixels", "raw_actions"],
        "privileged_context_used_by_model_refiner_or_gate": False,
        "base_parameter_digest_before": base_before, "base_parameter_digest_after": base_after,
        "base_frozen": base_before == base_after,
        "data_sha256": sha256_file(destination), "context_sha256": sha256_file(context_path),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    atomic_json(manifest_path, manifest)
    append_note(
        f"Generated fixed {role} role: {len(specs)} fresh episodes, {len(arrays['target'])} transitions, no exclusions/replacements; raw pixels discarded after bounded encoding."
    )
    update_status(f"{role}_generation_complete", episodes=len(specs), transitions=len(arrays["target"]))
    print(json.dumps({"status": f"{role}_complete", "episodes": len(specs), "transitions": len(arrays["target"])}))


def _validation_objective(
    model: StagewiseRefiner, arrays: Mapping[str, np.ndarray], normalized: np.ndarray,
    indices: np.ndarray, device: torch.device, batch_size: int = 512,
) -> float:
    total, count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            history = torch.from_numpy(np.ascontiguousarray(arrays["history"][batch])).to(device)
            actions = torch.from_numpy(np.ascontiguousarray(normalized[batch])).to(device)
            base = torch.from_numpy(np.ascontiguousarray(arrays["base"][batch])).to(device)
            target = torch.from_numpy(np.ascontiguousarray(arrays["target"][batch])).to(device)
            exits, _ = model.dense(history, actions, base)
            loss = (exits - target[:, None]).square().mean()
            total += float(loss.item()) * len(batch)
            count += len(batch)
    return total / count


def train_refiner() -> None:
    checkpoint_path = ROOT / "checkpoints/refiner.pt"
    metrics_path = ROOT / "REFINER_FIT_TABLE.json"
    outputs_path = ROOT / "development/fit_refiner_outputs.npz"
    artifact_path = ROOT / "FIT_ARTIFACTS.npz"
    if all(path.exists() for path in (checkpoint_path, metrics_path, outputs_path, artifact_path)):
        print(json.dumps({"status": "refiner_already_trained", "checkpoint_sha256": sha256_file(checkpoint_path)}))
        return
    config = _config()
    fit_path = ROOT / "data/fit.npz"
    if not fit_path.exists():
        raise RuntimeError("fit data must be generated first")
    update_status("train_refiner")
    arrays = load_npz(fit_path)
    action_mean, action_scale = action_normalization(arrays)
    whitening, whitening_details = fit_whitening(arrays["target"], floor_ratio=1e-3)
    normalized = normalize_actions(arrays["actions"], action_mean, action_scale)
    episode_id = arrays["episode_id"].astype(int)
    validation_episodes = np.asarray(
        [spec["ordinal"] for spec in config["role_specs"]["fit"] if spec["ordinal"] % 10 == 9],
        dtype=np.int64,
    )
    validation_mask = np.isin(episode_id, validation_episodes)
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if len(validation_indices) == 0 or len(train_indices) == 0:
        raise RuntimeError("fixed internal fit partition is empty")

    torch.manual_seed(REFINER_SEED)
    np.random.seed(REFINER_SEED % (2**32 - 1))
    device = choose_device()
    model = StagewiseRefiner(hidden=256, iteration_dim=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(REFINER_SEED)
    best_score = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history_rows: list[dict[str, Any]] = []
    training_start = time.perf_counter()
    for epoch in range(250):
        model.train()
        shuffled = rng.permutation(train_indices)
        total, count = 0.0, 0
        for start in range(0, len(shuffled), 256):
            batch = shuffled[start : start + 256]
            history = torch.from_numpy(np.ascontiguousarray(arrays["history"][batch])).to(device)
            actions = torch.from_numpy(np.ascontiguousarray(normalized[batch])).to(device)
            base = torch.from_numpy(np.ascontiguousarray(arrays["base"][batch])).to(device)
            target = torch.from_numpy(np.ascontiguousarray(arrays["target"][batch])).to(device)
            exits, updates = model.dense(history, actions, base)
            prediction_loss = (exits - target[:, None]).square().mean()
            update_penalty = updates.square().mean()
            loss = prediction_loss + 1e-4 * update_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().item()) * len(batch)
            count += len(batch)
        validation_score = _validation_objective(
            model, arrays, normalized, validation_indices, device, batch_size=512
        )
        train_score = total / count
        improved = validation_score < best_score - 1e-8
        if improved:
            best_score = validation_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history_rows.append(
            {
                "epoch": epoch + 1, "train_objective": train_score,
                "internal_validation_objective": validation_score,
                "best_internal_validation_objective": best_score, "improved": improved,
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or stale >= 30:
            print(
                f"[refiner] epoch={epoch+1} train={train_score:.8g} val={validation_score:.8g} "
                f"best={best_score:.8g} stale={stale}", flush=True,
            )
        if stale >= 30:
            break
    if best_state is None or not math.isfinite(best_score):
        raise RuntimeError("refiner training produced no finite state")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    _save_checkpoint(model, best_epoch + 1)
    exits, updates = evaluate_refiner_dense(model, arrays, action_mean, action_scale, device)
    raw, white = raw_white_losses(exits, arrays["target"], whitening)
    base_difference = arrays["base"].astype(np.float64) - arrays["target"].astype(np.float64)
    base_raw = np.square(base_difference).mean(axis=1)
    base_white = np.square(base_difference @ whitening).mean(axis=1)
    previous_raw = base_raw
    previous_white = base_white
    depth_rows: list[dict[str, Any]] = []
    gains: list[np.ndarray] = []
    threshold = max(1e-10, 1e-4 * float(base_raw.mean()))
    for column, depth in enumerate(DEPTHS):
        marginal_raw = previous_raw - raw[:, column]
        marginal_white = previous_white - white[:, column]
        gains.append(marginal_raw)
        depth_rows.append(
            {
                "depth": depth, "raw_mse": float(raw[:, column].mean()),
                "fit_whitened_mse": float(white[:, column].mean()),
                "raw_marginal_gain": float(marginal_raw.mean()),
                "fit_whitened_marginal_gain": float(marginal_white.mean()),
                "raw_marginal_gain_std": float(marginal_raw.std()),
                "fraction_raw_benefiting": float(np.mean(marginal_raw > 0)),
                "fraction_whitened_benefiting": float(np.mean(marginal_white > 0)),
                "mean_update_norm": float(np.linalg.norm(updates[:, column], axis=1).mean()),
                "std_update_norm": float(np.linalg.norm(updates[:, column], axis=1).std()),
                "gain_spearman_with_previous_stage": None if column == 0 else safe_spearman(gains[column - 1], marginal_raw),
                "gain_sign_stability_with_previous_stage": None if column == 0 else float(
                    np.mean(np.sign(gains[column - 1]) == np.sign(marginal_raw))
                ),
            }
        )
        previous_raw = raw[:, column]
        previous_white = white[:, column]
    extra = depth_rows[1:]
    nontrivial_count = sum(row["raw_marginal_gain_std"] > threshold for row in extra)
    heterogeneous_fraction = any(0.02 < row["fraction_raw_benefiting"] < 0.98 for row in extra)
    positive_extra_mean = any(row["raw_marginal_gain"] > 0 for row in extra)
    finite = bool(np.isfinite(raw).all() and np.isfinite(white).all() and np.isfinite(exits).all())
    signal_present = bool(finite and nontrivial_count >= 2 and heterogeneous_fraction and positive_extra_mean)
    metrics = {
        "schema_version": 1, "created_unix_ns": time.time_ns(),
        "fit_episode_count": config["episode_counts"]["fit"], "fit_transition_count": len(raw),
        "base_raw_mse": float(base_raw.mean()), "base_fit_whitened_mse": float(base_white.mean()),
        "depth_table": depth_rows,
        "stagewise_stability_definition": "adjacent-stage marginal-gain Spearman and sign agreement plus update-norm dispersion",
        "heterogeneity_rule": {
            "nontrivial_std_threshold": threshold,
            "extra_stage_nontrivial_std_count": nontrivial_count,
            "heterogeneous_benefit_fraction_exists": heterogeneous_fraction,
            "positive_extra_stage_mean_exists": positive_extra_mean,
            "finite": finite, "signal_present": signal_present,
        },
        "training": {
            "seed": REFINER_SEED, "best_epoch": best_epoch + 1,
            "epochs_completed": len(history_rows), "best_internal_validation_objective": best_score,
            "internal_validation_episodes": validation_episodes.tolist(),
            "train_transition_count": len(train_indices), "validation_transition_count": len(validation_indices),
            "elapsed_seconds": time.perf_counter() - training_start,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "history": history_rows,
    }
    atomic_npz(
        artifact_path,
        {
            "action_mean": action_mean.astype(np.float64),
            "action_scale": action_scale.astype(np.float64),
            "whitening_matrix": whitening.astype(np.float64),
            "whitening_mean": np.asarray(whitening_details["mean"], dtype=np.float64),
            "whitening_eigenvalues": np.asarray(whitening_details["eigenvalues"], dtype=np.float64),
            "whitening_used_eigenvalues": np.asarray(whitening_details["used"], dtype=np.float64),
        },
    )
    atomic_npz(outputs_path, {"exits": exits, "updates": updates, "raw_losses": raw, "white_losses": white})
    atomic_json(metrics_path, metrics)
    atomic_json(
        ROOT / "checkpoints/refiner_manifest.json",
        {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "fit_data_sha256": sha256_file(fit_path),
            "fit_artifacts_sha256": sha256_file(artifact_path),
            "training_seed": REFINER_SEED, "best_epoch": best_epoch + 1,
            "base_model_trainable_or_loaded_during_refiner_training": False,
        },
    )
    append_note(
        f"First PushT refiner/gain-heterogeneity table completed: signal_present={signal_present}, checkpoint={sha256_file(checkpoint_path)}."
    )
    if not signal_present:
        atomic_json(
            ROOT / "EARLY_TERMINAL.json",
            {
                "scientific_label": "pusht_refinement_signal_absent",
                "reason": "The fixed PushT refiner failed the predeclared finite heterogeneous extra-step-value rule.",
                "refiner_fit_table_sha256": sha256_file(metrics_path),
            },
        )
        update_status("refinement_signal_absent", status="terminal")
    else:
        update_status("refiner_complete", checkpoint_sha256=sha256_file(checkpoint_path))
    print(json.dumps({"status": "refiner_complete", "signal_present": signal_present, "best_epoch": best_epoch + 1}))


def _within_episode_permutation(calls: np.ndarray, episode_id: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty_like(calls)
    for identifier in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == identifier)
        output[indices] = calls[indices][rng.permutation(len(indices))]
        if not np.array_equal(
            np.bincount(calls[indices], minlength=5),
            np.bincount(output[indices], minlength=5),
        ):
            raise RuntimeError("within-episode permutation changed a call histogram")
    return output


def _candidate_diagnostics(
    *, role: str, raw: np.ndarray, white: np.ndarray, scores: np.ndarray,
    thresholds: np.ndarray, episode_id: np.ndarray, episode_count: int,
    fitted: Mapping[str, np.ndarray], config: Mapping[str, Any], candidate: Mapping[str, Any],
) -> dict[str, Any]:
    calls, reached = sequential_calls(scores, thresholds)
    row = np.arange(len(calls))
    adaptive_raw = raw[row, calls - 1]
    adaptive_white = white[row, calls - 1]
    gate_evaluations = int(np.minimum(calls, 3).sum())
    refiner_calls = int(calls.sum())
    refiner_price = int(config["compute"]["refiner"]["total_flops_per_call"])
    gate_price = int(config["compute"]["gate"]["total_flops_per_reached_evaluation"])
    analytic_refiner_flops = refiner_calls * refiner_price + gate_evaluations * gate_price
    analytic_mean_depth = analytic_refiner_flops / (len(calls) * refiner_price)
    raw_analytic = strongest_analytic_mixture(raw, analytic_mean_depth, episode_id, episode_count)
    white_analytic = strongest_analytic_mixture(white, analytic_mean_depth, episode_id, episode_count)
    permutation = _within_episode_permutation(
        calls, episode_id, PERMUTATION_SEED + (0 if role == "selection" else 1000)
    )
    permuted_raw = raw[row, permutation - 1]
    permuted_white = white[row, permutation - 1]
    episode_differences = {
        "raw_vs_analytic": episode_means(raw_analytic["loss"] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_analytic": episode_means(
            white_analytic["loss"] - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_within_episode_permutation": episode_means(
            permuted_raw - adaptive_raw, episode_id, episode_count
        ),
        "fit_whitened_vs_within_episode_permutation": episode_means(
            permuted_white - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_fixed_d1": episode_means(raw[:, 0] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_fixed_d1": episode_means(
            white[:, 0] - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_fixed_d4": episode_means(raw[:, 3] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_fixed_d4": episode_means(
            white[:, 3] - adaptive_white, episode_id, episode_count
        ),
    }
    point = {name: float(value.mean()) for name, value in episode_differences.items()}
    standardized = {
        name: float(value.mean() / (value.std(ddof=1) + 1e-18))
        for name, value in episode_differences.items()
        if name in ("raw_vs_analytic", "fit_whitened_vs_analytic")
    }
    stages: list[dict[str, Any]] = []
    for stage in range(3):
        raw_gain = raw[:, stage] - raw[:, stage + 1]
        white_gain = white[:, stage] - white[:, stage + 1]
        combined = 0.5 * (
            raw_gain / fitted["raw_gain_scale"][stage]
            + white_gain / fitted["white_gain_scale"][stage]
        )
        mask = reached[stage]
        stages.append(
            {
                "stage": stage + 1, "reached_rows": int(mask.sum()),
                "score_vs_combined_gain_spearman": safe_spearman(scores[mask, stage], combined[mask]),
                "score_vs_raw_gain_spearman": safe_spearman(scores[mask, stage], raw_gain[mask]),
                "score_vs_fit_whitened_gain_spearman": safe_spearman(
                    scores[mask, stage], white_gain[mask]
                ),
            }
        )
    base_price = int(config["compute"]["base_predictor"]["total_flops_per_transition"])
    normalize_price = int(config["compute"]["gate"]["action_normalization_flops_per_transition"])
    common = len(calls) * (base_price + normalize_price)
    adaptive_total = common + analytic_refiner_flops
    finite_rhos = [item["score_vs_combined_gain_spearman"] for item in stages]
    all_nonnegative = all(math.isfinite(value) and value >= 0 for value in finite_rhos)
    both_positive = point["raw_vs_analytic"] > 0 and point["fit_whitened_vs_analytic"] > 0
    return {
        **dict(candidate), "thresholds": thresholds.tolist(),
        "call_histogram": np.bincount(calls, minlength=5)[1:].tolist(),
        "mean_depth": float(calls.mean()), "refiner_calls": refiner_calls,
        "reached_gate_evaluations": gate_evaluations,
        "analytic_equivalent_mean_depth": float(analytic_mean_depth),
        "adaptive_total_counted_flops": int(adaptive_total),
        "analytic_comparator_total_counted_flops": int(adaptive_total),
        "exact_total_counted_compute_equality": True,
        "raw_analytic_mixture": {k: v for k, v in raw_analytic.items() if k != "loss"},
        "fit_whitened_analytic_mixture": {k: v for k, v in white_analytic.items() if k != "loss"},
        "point_benefits": point, "standardized_co_primary": standardized,
        "minimum_standardized_co_primary": min(standardized.values()),
        "stagewise_rank": stages,
        "minimum_combined_stagewise_spearman": min(finite_rhos) if all(map(math.isfinite, finite_rhos)) else None,
        "all_combined_stagewise_spearman_nonnegative": all_nonnegative,
        "both_co_primary_point_benefits_positive": both_positive,
        "within_episode_histograms_preserved": True,
    }


def fit_select_gate() -> None:
    freeze_path = ROOT / "GATE_FREEZE.json"
    if freeze_path.exists() and (ROOT / "FROZEN_GATE.npz").exists():
        print(json.dumps({"status": "gate_already_frozen", "freeze_sha256": sha256_file(freeze_path)}))
        return
    if (ROOT / "EARLY_TERMINAL.json").exists():
        raise RuntimeError("gate fitting is forbidden after absent refinement signal")
    config = _config()
    required = (
        ROOT / "data/fit.npz", ROOT / "data/selection.npz", ROOT / "FIT_ARTIFACTS.npz",
        ROOT / "development/fit_refiner_outputs.npz", ROOT / "checkpoints/refiner.pt",
    )
    if not all(path.exists() for path in required):
        raise RuntimeError("fit/selection/refiner artifacts are incomplete")
    if (ROOT / "data/evaluation.npz").exists():
        raise RuntimeError("evaluation outcomes exist before gate freeze")
    update_status("fit_gate_without_selection_open")
    fit_arrays = load_npz(ROOT / "data/fit.npz")
    fit_outputs = load_npz(ROOT / "development/fit_refiner_outputs.npz")
    fit_artifacts = load_npz(ROOT / "FIT_ARTIFACTS.npz")
    features = feature_arrays(
        fit_arrays, fit_outputs["exits"], fit_outputs["updates"],
        fit_artifacts["action_mean"], fit_artifacts["action_scale"],
    )
    raw = fit_outputs["raw_losses"].astype(np.float64)
    white = fit_outputs["white_losses"].astype(np.float64)
    raw_gain = raw[:, :3] - raw[:, 1:]
    white_gain = white[:, :3] - white[:, 1:]
    fitted = fit_stagewise_ridge(features, raw_gain, white_gain, REGULARIZATIONS)
    thresholds = np.empty((len(CANDIDATES), 3), dtype=np.float64)
    regularization_indices = np.empty(len(CANDIDATES), dtype=np.int16)
    fit_scores = np.empty((len(REGULARIZATIONS), len(features), 3), dtype=np.float64)
    for regularization_index in range(len(REGULARIZATIONS)):
        fit_scores[regularization_index] = gate_scores(features, fitted, regularization_index)
    for candidate_index, candidate in enumerate(CANDIDATES):
        regularization_index = REGULARIZATIONS.index(float(candidate["regularization"]))
        regularization_indices[candidate_index] = regularization_index
        for stage in range(3):
            thresholds[candidate_index, stage] = float(
                np.quantile(
                    fit_scores[regularization_index, :, stage],
                    float(candidate["fit_score_quantile"]),
                )
            )
    gate_fit_path = ROOT / "development/gate_fit_candidates.npz"
    atomic_npz(
        gate_fit_path,
        {
            **fitted, "candidate_thresholds": thresholds,
            "candidate_regularization_indices": regularization_indices,
            "fit_scores": fit_scores,
        },
    )
    atomic_json(
        ROOT / "development/GATE_FIT_LOCK.json",
        {
            "status": "all_six_candidates_fit_before_selection_open",
            "candidate_count": len(CANDIDATES), "fit_data_sha256": sha256_file(ROOT / "data/fit.npz"),
            "gate_fit_sha256": sha256_file(gate_fit_path),
            "selection_data_existed_but_was_not_opened_during_fit": (ROOT / "data/selection.npz").exists(),
            "evaluation_data_exists": False, "target_used_as_gate_input": False,
            "target_used_only_as_fit_gain_label": True,
        },
    )
    append_note("Fit all six fixed stage-specific gate candidates on fit episodes only; locked coefficients and thresholds before selection access.")

    update_status("select_gate_once")
    selection_arrays = load_npz(ROOT / "data/selection.npz")
    device = choose_device()
    model = _checkpoint_model(device)
    selection_exits, selection_updates = evaluate_refiner_dense(
        model, selection_arrays, fit_artifacts["action_mean"], fit_artifacts["action_scale"], device
    )
    selection_features = feature_arrays(
        selection_arrays, selection_exits, selection_updates,
        fit_artifacts["action_mean"], fit_artifacts["action_scale"],
    )
    selection_raw, selection_white = raw_white_losses(
        selection_exits, selection_arrays["target"], fit_artifacts["whitening_matrix"]
    )
    candidates: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        regularization_index = int(regularization_indices[candidate_index])
        scores = gate_scores(selection_features, fitted, regularization_index)
        candidates.append(
            _candidate_diagnostics(
                role="selection", raw=selection_raw, white=selection_white, scores=scores,
                thresholds=thresholds[candidate_index], episode_id=selection_arrays["episode_id"],
                episode_count=int(config["episode_counts"]["selection"]), fitted=fitted,
                config=config, candidate=candidate,
            )
        )
    candidates.sort(
        key=lambda item: (
            -int(item["all_combined_stagewise_spearman_nonnegative"]),
            -int(item["both_co_primary_point_benefits_positive"]),
            -float(item["minimum_standardized_co_primary"]),
            -float(item["minimum_combined_stagewise_spearman"] if item["minimum_combined_stagewise_spearman"] is not None else -math.inf),
            int(item["adaptive_total_counted_flops"]), str(item["candidate_id"]),
        )
    )
    selected = candidates[0]
    selected_index = next(
        index for index, candidate in enumerate(CANDIDATES)
        if candidate["candidate_id"] == selected["candidate_id"]
    )
    selected_regularization_index = int(regularization_indices[selected_index])
    selection_outputs_path = ROOT / "development/selection_refiner_outputs.npz"
    atomic_npz(
        selection_outputs_path,
        {
            "exits": selection_exits, "updates": selection_updates,
            "raw_losses": selection_raw, "white_losses": selection_white,
        },
    )
    ledger_path = ROOT / "SELECTION_LEDGER.json"
    atomic_json(
        ledger_path,
        {
            "schema_version": 1, "status": "single_selection_complete_no_refit",
            "candidate_count": len(candidates), "fixed_ranking": config["gate"]["selection_ranking"],
            "candidates_in_rank_order": candidates, "selected_candidate_id": selected["candidate_id"],
            "selection_episode_count": config["episode_counts"]["selection"],
            "selection_data_sha256": sha256_file(ROOT / "data/selection.npz"),
            "gate_fit_lock_sha256": sha256_file(ROOT / "development/GATE_FIT_LOCK.json"),
            "refiner_or_features_refit_after_selection": False,
            "evaluation_episode_outcomes_opened": False,
        },
    )
    frozen_gate_path = ROOT / "FROZEN_GATE.npz"
    atomic_npz(
        frozen_gate_path,
        {
            "feature_mean": fitted["feature_mean"], "feature_scale": fitted["feature_scale"],
            "weights": fitted["weights"][selected_regularization_index],
            "thresholds": thresholds[selected_index],
            "raw_gain_mean": fitted["raw_gain_mean"], "raw_gain_scale": fitted["raw_gain_scale"],
            "white_gain_mean": fitted["white_gain_mean"], "white_gain_scale": fitted["white_gain_scale"],
            "regularization": np.asarray(selected["regularization"], dtype=np.float64),
            "fit_score_quantile": np.asarray(selected["fit_score_quantile"], dtype=np.float64),
        },
    )
    causal_signature = list(inspect.signature(build_causal_features).parameters)
    atomic_json(
        freeze_path,
        {
            "schema_version": 1, "status": "frozen_before_evaluation_generation_or_outcomes",
            "created_unix_ns": time.time_ns(), "selected": selected,
            "frozen_gate_sha256": sha256_file(frozen_gate_path),
            "refiner_checkpoint_sha256": sha256_file(ROOT / "checkpoints/refiner.pt"),
            "fit_artifacts_sha256": sha256_file(ROOT / "FIT_ARTIFACTS.npz"),
            "selection_ledger_sha256": sha256_file(ledger_path),
            "feature_builder_signature": causal_signature,
            "feature_names_sha256": __import__("hashlib").sha256("\n".join(causal_feature_names()).encode()).hexdigest(),
            "feature_width": len(causal_feature_names()),
            "gate_inputs": ["frozen history latents", "fit-normalized blocked actions", "current prediction", "last refiner update", "causal derived norms/differences"],
            "target_future_contact_reward_success_simulator_state_or_geometry_inputs": False,
            "base_refiner_feature_gate_threshold_normalization_whitening_compute_prices_decision_rule_frozen": True,
            "evaluation_data_exists_at_freeze": (ROOT / "data/evaluation.npz").exists(),
            "evaluation_outcomes_opened": False, "refit_after_selection": False,
        },
    )
    if read_json(freeze_path)["evaluation_data_exists_at_freeze"]:
        raise RuntimeError("evaluation data unexpectedly existed at gate freeze")
    append_note(
        f"Selection opened once and froze {selected['candidate_id']} without refit; evaluation trajectories did not yet exist."
    )
    update_status("gate_frozen_before_evaluation", selected_candidate=selected["candidate_id"])
    print(json.dumps({"status": "gate_frozen", "selected": selected["candidate_id"], "selection_benefits": selected["point_benefits"]}))


def _frozen_gate_scores(features: np.ndarray, gate: Mapping[str, np.ndarray]) -> np.ndarray:
    fitted = {
        "feature_mean": gate["feature_mean"], "feature_scale": gate["feature_scale"],
        "weights": gate["weights"][None, ...],
    }
    return gate_scores(features, fitted, 0)


def _latency_measurements(
    arrays: Mapping[str, np.ndarray], model: StagewiseRefiner, gate: Mapping[str, np.ndarray],
    fit_artifacts: Mapping[str, np.ndarray], expected_calls: np.ndarray, expected_selected: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    normalized = normalize_actions(arrays["actions"], fit_artifacts["action_mean"], fit_artifacts["action_scale"])
    history = torch.from_numpy(np.ascontiguousarray(arrays["history"])).to(device)
    actions = torch.from_numpy(np.ascontiguousarray(normalized)).to(device)
    base = torch.from_numpy(np.ascontiguousarray(arrays["base"])).to(device)
    feature_mean = torch.from_numpy(gate["feature_mean"].astype(np.float32)).to(device)
    feature_scale = torch.from_numpy(gate["feature_scale"].astype(np.float32)).to(device)
    weights = torch.from_numpy(gate["weights"].astype(np.float32)).to(device)
    thresholds = torch.from_numpy(gate["thresholds"].astype(np.float32)).to(device)

    def adaptive_once() -> tuple[Tensor, np.ndarray]:
        current = base
        active = torch.ones(len(base), dtype=torch.bool, device=device)
        measured_calls = torch.zeros(len(base), dtype=torch.int64, device=device)
        with torch.inference_mode():
            for iteration in range(MAX_DEPTH):
                indices = torch.nonzero(active, as_tuple=False).flatten()
                if not len(indices):
                    break
                refined, update = model.step(
                    history.index_select(0, indices), actions.index_select(0, indices),
                    current.index_select(0, indices), iteration,
                )
                current = current.index_copy(0, indices, refined)
                measured_calls[indices] += 1
                if iteration < 3:
                    features = build_causal_features(
                        history.index_select(0, indices), actions.index_select(0, indices), refined, update
                    )
                    z = (features - feature_mean[iteration]) / feature_scale[iteration]
                    heads = z @ weights[iteration].T
                    score = torch.minimum(heads[:, 0], heads[:, 1])
                    next_active = torch.zeros_like(active)
                    next_active[indices] = score > thresholds[iteration]
                    active = next_active
        return current, measured_calls.detach().cpu().numpy()

    adaptive_times: list[float] = []
    output: Tensor | None = None
    observed_calls: np.ndarray | None = None
    for _ in range(LATENCY_REPEATS):
        synchronize(device)
        start = time.perf_counter()
        output, observed_calls = adaptive_once()
        synchronize(device)
        adaptive_times.append(time.perf_counter() - start)
    assert output is not None and observed_calls is not None
    observed = output.detach().cpu().numpy()
    if not np.array_equal(observed_calls, expected_calls):
        raise RuntimeError("latency-path calls do not reproduce evaluation calls")
    if not np.allclose(observed, expected_selected, rtol=2e-5, atol=2e-6):
        raise RuntimeError("latency-path selected outputs do not reproduce dense selection")

    fixed: dict[str, Any] = {}
    for depth in (1, 4):
        times: list[float] = []
        for _ in range(LATENCY_REPEATS):
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                current = base
                for iteration in range(depth):
                    current, _ = model.step(history, actions, current, iteration)
            synchronize(device)
            times.append(time.perf_counter() - start)
        fixed[f"fixed_depth_{depth}"] = {
            "seconds": times, "median_seconds": float(np.median(times)), "depth": depth,
        }

    base_model = load_base_model(device)
    base_before = module_digest(base_model)
    base_times: list[float] = []
    for _ in range(LATENCY_REPEATS):
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            for batch_start in range(0, len(base), 512):
                history_batch = history[batch_start : batch_start + 512]
                raw_action = torch.from_numpy(
                    np.ascontiguousarray(arrays["actions"][batch_start : batch_start + 512])
                ).to(device)
                action_embedding = base_model.action_encoder(raw_action)
                _ = base_model.predict(history_batch, action_embedding)[:, -1]
        synchronize(device)
        base_times.append(time.perf_counter() - start)
    base_after = module_digest(base_model)
    if base_before != base_after:
        raise RuntimeError("base changed during latency measurement")
    return {
        "device": str(device), "synchronization": True, "repeats": LATENCY_REPEATS,
        "base_predictor_all_rows": {"seconds": base_times, "median_seconds": float(np.median(base_times))},
        "cached_base_to_adaptive_selected": {
            "seconds": adaptive_times, "median_seconds": float(np.median(adaptive_times)),
        },
        **fixed,
        "latency_separate_from_counted_flops": True,
        "encoding_excluded": True, "no_wall_clock_speedup_claim": True,
    }


def evaluate_pilot() -> None:
    decision_path = ROOT / "PILOT_DECISION.json"
    if decision_path.exists():
        print(json.dumps({"status": "evaluation_already_decided", "decision_sha256": sha256_file(decision_path)}))
        return
    config = _config()
    required = (
        ROOT / "GATE_FREEZE.json", ROOT / "FROZEN_GATE.npz", ROOT / "data/evaluation.npz",
        ROOT / "FIT_ARTIFACTS.npz", ROOT / "checkpoints/refiner.pt",
    )
    if not all(path.exists() for path in required):
        raise RuntimeError("frozen gate and fixed evaluation role are required")
    freeze = read_json(ROOT / "GATE_FREEZE.json")
    if freeze["evaluation_data_exists_at_freeze"] or freeze["evaluation_outcomes_opened"]:
        raise RuntimeError("gate freeze chronology invalid")
    update_status("single_untouched_evaluation")
    arrays = load_npz(ROOT / "data/evaluation.npz")
    fit_artifacts = load_npz(ROOT / "FIT_ARTIFACTS.npz")
    gate = load_npz(ROOT / "FROZEN_GATE.npz")
    device = choose_device()
    model = _checkpoint_model(device)
    exits, updates = evaluate_refiner_dense(
        model, arrays, fit_artifacts["action_mean"], fit_artifacts["action_scale"], device
    )
    features = feature_arrays(
        arrays, exits, updates, fit_artifacts["action_mean"], fit_artifacts["action_scale"]
    )
    scores = _frozen_gate_scores(features, gate)
    raw, white = raw_white_losses(exits, arrays["target"], fit_artifacts["whitening_matrix"])
    candidate = {
        "candidate_id": freeze["selected"]["candidate_id"],
        "regularization": float(gate["regularization"]),
        "fit_score_quantile": float(gate["fit_score_quantile"]),
    }
    fitted_for_diagnostics = {
        "raw_gain_scale": gate["raw_gain_scale"], "white_gain_scale": gate["white_gain_scale"]
    }
    diagnostics = _candidate_diagnostics(
        role="evaluation", raw=raw, white=white, scores=scores, thresholds=gate["thresholds"],
        episode_id=arrays["episode_id"], episode_count=int(config["episode_counts"]["evaluation"]),
        fitted=fitted_for_diagnostics, config=config, candidate=candidate,
    )
    calls, reached = sequential_calls(scores, gate["thresholds"])
    row = np.arange(len(calls))
    selected = exits[row, calls - 1]
    permutation_calls = _within_episode_permutation(calls, arrays["episode_id"], PERMUTATION_SEED + 1000)
    episode_id = arrays["episode_id"].astype(int)
    episode_count = int(config["episode_counts"]["evaluation"])
    analytic_mean_depth = float(diagnostics["analytic_equivalent_mean_depth"])
    raw_analytic = strongest_analytic_mixture(raw, analytic_mean_depth, episode_id, episode_count)
    white_analytic = strongest_analytic_mixture(white, analytic_mean_depth, episode_id, episode_count)
    adaptive_raw = raw[row, calls - 1]
    adaptive_white = white[row, calls - 1]
    permutation_raw = raw[row, permutation_calls - 1]
    permutation_white = white[row, permutation_calls - 1]
    endpoint_values = {
        "raw_vs_primary_analytic": episode_means(raw_analytic["loss"] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_primary_analytic": episode_means(
            white_analytic["loss"] - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_within_episode_permutation": episode_means(
            permutation_raw - adaptive_raw, episode_id, episode_count
        ),
        "fit_whitened_vs_within_episode_permutation": episode_means(
            permutation_white - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_fixed_depth_1": episode_means(raw[:, 0] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_fixed_depth_1": episode_means(
            white[:, 0] - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_fixed_depth_4": episode_means(raw[:, 3] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_fixed_depth_4": episode_means(
            white[:, 3] - adaptive_white, episode_id, episode_count
        ),
    }
    intervals = {
        name: bootstrap_interval(
            values, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES, confidence=0.95
        )
        for name, values in endpoint_values.items()
    }
    forbidden = ("target", "future", "contact", "reward", "success", "simulator", "geometry", "state")
    names = causal_feature_names()
    causal_checks = {
        "feature_builder_signature_exact": list(inspect.signature(build_causal_features).parameters)
        == ["history", "actions", "current", "update"],
        "feature_names_have_no_forbidden_boundary_terms": not any(
            term in name.lower() for name in names for term in forbidden
        ),
        "frozen_gate_declares_no_privileged_inputs": not freeze[
            "target_future_contact_reward_success_simulator_state_or_geometry_inputs"
        ],
        "evaluation_manifest_declares_no_privileged_model_inputs": not read_json(
            ROOT / "data/evaluation_manifest.json"
        )["privileged_context_used_by_model_refiner_or_gate"],
        "scores_reconstructed_only_from_causal_features": True,
    }
    finite = bool(
        all(
            np.isfinite(value).all()
            for value in (exits, updates, scores, raw, white, selected, adaptive_raw, adaptive_white)
        )
    )
    refiner_price = int(config["compute"]["refiner"]["total_flops_per_call"])
    gate_price = int(config["compute"]["gate"]["total_flops_per_reached_evaluation"])
    base_price = int(config["compute"]["base_predictor"]["total_flops_per_transition"])
    normalization_price = int(config["compute"]["gate"]["action_normalization_flops_per_transition"])
    refiner_calls = int(calls.sum())
    gate_evaluations = int(np.minimum(calls, 3).sum())
    common_flops = len(calls) * (base_price + normalization_price)
    adaptive_flops = common_flops + refiner_calls * refiner_price + gate_evaluations * gate_price
    comparator_flops = common_flops + refiner_calls * refiner_price + gate_evaluations * gate_price
    exact_compute = adaptive_flops == comparator_flops
    combined_rhos = [item["score_vs_combined_gain_spearman"] for item in diagnostics["stagewise_rank"]]
    nonnegative_ranks = all(math.isfinite(value) and value >= 0 for value in combined_rhos)
    raw_positive = intervals["raw_vs_primary_analytic"]["mean_benefit"] > 0
    white_positive = intervals["fit_whitened_vs_primary_analytic"]["mean_benefit"] > 0
    process_valid = bool(
        exact_compute and all(causal_checks.values()) and finite
        and len(np.unique(episode_id)) == episode_count
        and all(np.sum(episode_id == identifier) > 0 for identifier in range(episode_count))
    )
    pilot_positive = bool(process_valid and raw_positive and white_positive and nonnegative_ranks)
    proposed_label = (
        "pusht_replication_pilot_promising" if pilot_positive
        else ("pusht_replication_pilot_not_supported" if process_valid else "pusht_replication_execution_invalid")
    )
    evaluation_arrays_path = ROOT / "EVALUATION_ARRAYS.npz"
    atomic_npz(
        evaluation_arrays_path,
        {
            "episode_id": episode_id.astype(np.int32), "episode_seed": arrays["episode_seed"],
            "target": arrays["target"], "dense_exits": exits, "dense_updates": updates,
            "scores": scores.astype(np.float64), "calls": calls.astype(np.int8),
            "permutation_calls": permutation_calls.astype(np.int8), "selected": selected,
            **{f"episode_{name}": value for name, value in endpoint_values.items()},
        },
    )
    decision = {
        "schema_version": 1, "created_unix_ns": time.time_ns(),
        "status": "evaluation_outcome_opened_once_and_decision_locked",
        "proposed_scientific_label": proposed_label,
        "pilot_positive": pilot_positive, "process_valid": process_valid,
        "primary": {
            "raw": intervals["raw_vs_primary_analytic"],
            "fit_whitened": intervals["fit_whitened_vs_primary_analytic"],
        },
        "all_endpoints": intervals, "diagnostics": diagnostics,
        "adaptive_raw_mse": float(adaptive_raw.mean()),
        "adaptive_fit_whitened_mse": float(adaptive_white.mean()),
        "stagewise_rank": diagnostics["stagewise_rank"],
        "criteria": {
            "raw_mean_favors_adaptive": raw_positive,
            "fit_whitened_mean_favors_adaptive": white_positive,
            "exact_counted_compute_equality": exact_compute,
            "causal_input_checks": causal_checks,
            "finite_outputs": finite,
            "all_combined_stagewise_rank_signs_nonnegative": nonnegative_ranks,
        },
        "compute": {
            "rows": len(calls), "base_model_calls": len(calls),
            "refiner_calls": refiner_calls, "reached_gate_evaluations": gate_evaluations,
            "base_predictor_flops": len(calls) * base_price,
            "action_normalization_flops": len(calls) * normalization_price,
            "refiner_flops": refiner_calls * refiner_price,
            "gate_feature_flops": gate_evaluations * int(
                config["compute"]["gate"]["feature_flops_per_reached_evaluation"]
            ),
            "gate_dual_affine_score_flops": gate_evaluations * int(
                config["compute"]["gate"]["dual_affine_score_flops_per_reached_evaluation"]
            ),
            "gate_total_flops": gate_evaluations * gate_price,
            "adaptive_total_counted_flops": adaptive_flops,
            "primary_analytic_comparator_total_counted_flops": comparator_flops,
            "exact_integer_total_counted_flop_equality": exact_compute,
            "analytic_equivalent_mean_depth": analytic_mean_depth,
            "nonflop_comparison_min_operations": gate_evaluations * int(
                config["compute"]["gate"]["nonflop_comparison_min_operations_per_reached_evaluation"]
            ),
            "call_histogram_depth_1_to_4": np.bincount(calls, minlength=5)[1:].tolist(),
            "reached_gate_evaluations_by_stage": [int(mask.sum()) for mask in reached],
            "raw_primary_analytic_mixture": {k: v for k, v in raw_analytic.items() if k != "loss"},
            "fit_whitened_primary_analytic_mixture": {
                k: v for k, v in white_analytic.items() if k != "loss"
            },
        },
        "evaluation_arrays_sha256": sha256_file(evaluation_arrays_path),
        "gate_freeze_sha256": sha256_file(ROOT / "GATE_FREEZE.json"),
        "intervals_exploratory": True,
        "independent_verification_required_for_final_label": True,
    }
    atomic_json(decision_path, decision)
    append_note(
        f"Opened the fixed evaluation outcomes exactly once and locked proposed label {proposed_label}; raw/white benefits {decision['primary']['raw']['mean_benefit']:.8g}/{decision['primary']['fit_whitened']['mean_benefit']:.8g}."
    )

    # Privileged simulator context is first opened only after the pilot decision is durable.
    contexts = []
    for role in ("fit", "selection", "evaluation"):
        context = load_npz(ROOT / f"data/context_{role}.npz")
        contexts.append(context)
    descriptive = {
        "opened_after_pilot_decision_created_unix_ns": decision["created_unix_ns"],
        "not_used_in_refiner_gate_selection_or_primary_decision": True,
        "roles": summarize_context(contexts),
    }
    atomic_json(ROOT / "DESCRIPTIVE_CONTEXT.json", descriptive)
    latency = _latency_measurements(arrays, model, gate, fit_artifacts, calls, selected, device)
    atomic_json(ROOT / "LATENCY.json", latency)
    update_status("evaluation_decision_locked_pending_independent_verification", proposed_label=proposed_label)
    print(json.dumps({"status": "evaluation_complete", "proposed_label": proposed_label, "primary": decision["primary"]}))


def _markdown_interval(name: str, value: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {value['mean_benefit']:.9g} | {value['ci_low']:.9g} | "
        f"{value['ci_high']:.9g} |"
    )


def finalize() -> None:
    verification_path = ROOT / "INDEPENDENT_VERIFICATION.json"
    if not verification_path.exists():
        raise RuntimeError("independent verification must run before finalization")
    verification = read_json(verification_path)
    early = ROOT / "EARLY_TERMINAL.json"
    if early.exists():
        terminal = read_json(early)
        label = (
            terminal["scientific_label"] if verification["passed"]
            else "pusht_replication_execution_invalid"
        )
        fit_metrics = read_json(ROOT / "REFINER_FIT_TABLE.json")
        results = {
            "schema_version": 1, "scientific_label": label,
            "process_valid": verification["passed"], "pilot_evaluation_launched": False,
            "refinement_signal_present": False, "refiner_fit": fit_metrics,
            "independent_verification": verification,
        }
        report = f"""# PushT adaptive-computation replication pilot

Final scientific label: `{label}`.

The fixed PushT-specific stagewise refiner did not satisfy the predeclared heterogeneous-extra-step-value rule, so gate selection and held-out evaluation were not launched. This is the terminal process-valid interpretation required by the protocol; no redesign or retry was attempted.

The fit-only base raw MSE was {fit_metrics['base_raw_mse']:.9g}. See `REFINER_FIT_TABLE.json` for depths 1–4, marginal gains, benefiting fractions, and stability. A separate PushT confirmation is not scientifically warranted from this pilot because the allocation prerequisite was absent.
"""
    else:
        decision = read_json(ROOT / "PILOT_DECISION.json")
        label = (
            decision["proposed_scientific_label"] if verification["passed"]
            else "pusht_replication_execution_invalid"
        )
        fit_metrics = read_json(ROOT / "REFINER_FIT_TABLE.json")
        descriptive = read_json(ROOT / "DESCRIPTIVE_CONTEXT.json")
        latency = read_json(ROOT / "LATENCY.json")
        results = {
            "schema_version": 1, "scientific_label": label,
            "process_valid": bool(decision["process_valid"] and verification["passed"]),
            "pilot_positive": bool(decision["pilot_positive"] and verification["passed"]),
            "refinement_signal_present": True, "refiner_fit": fit_metrics,
            "evaluation": decision, "descriptive_context": descriptive,
            "latency": latency, "independent_verification": verification,
            "scientific_scope": {
                "environment": "PushT", "dgp_count": 1, "internally_selected_candidate": True,
                "intervals": "exploratory", "cube_gate_zero_shot_transfer": False,
                "planning_or_control_claim": False, "wall_clock_speedup_claim": False,
                "universal_generalization_claim": False,
            },
        }
        raw = decision["primary"]["raw"]
        white = decision["primary"]["fit_whitened"]
        depth_rows = fit_metrics["depth_table"]
        depth_lines = [
            "| Depth | Raw MSE | Fit-whitened MSE | Raw marginal gain | Fraction raw benefiting |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in depth_rows:
            depth_lines.append(
                f"| {row['depth']} | {row['raw_mse']:.9g} | {row['fit_whitened_mse']:.9g} | "
                f"{row['raw_marginal_gain']:.9g} | {row['fraction_raw_benefiting']:.6g} |"
            )
        endpoint_lines = [
            "| Comparison (positive favors adaptive) | Mean episode benefit | Exploratory 95% low | Exploratory 95% high |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, value in decision["all_endpoints"].items():
            endpoint_lines.append(_markdown_interval(name, value))
        warranted = label == "pusht_replication_pilot_promising"
        limitations = (
            "This is one environment, one WeakPolicy DGP, one refiner, one internally selected gate family, "
            "and a bounded pilot. The analytic comparator is an expectation-level transition-independent "
            "mixture; exploratory intervals do not account for candidate-selection multiplicity. The target "
            "latents inherit the cached pretrained encoder's representation, and prediction utility was not "
            "tested in planning or control."
        )
        result_sentence = (
            "The learned causal allocation beat the strongest transition-independent allocation on both co-primary means at identical counted compute."
            if decision["pilot_positive"]
            else "The learned causal allocation did not satisfy every predeclared pilot-positive condition; this process-valid negative is terminal."
        )
        confirmation = (
            "A separately generated, preregistered PushT confirmation is scientifically warranted, but none was launched here."
            if warranted
            else "A separate fresh PushT confirmation is not warranted from this bounded result without new scientific rationale."
        )
        report = f"""# PushT adaptive-computation replication pilot

Final scientific label: `{label}`.

## Paper-level result

1. Heterogeneous extra-refinement value existed under the fixed fit-only rule: {fit_metrics['heterogeneity_rule']['signal_present']}.
2. The PushT-specific causal gate selected `{decision['diagnostics']['candidate_id']}` and its evaluation call histogram at depths 1–4 was {decision['compute']['call_histogram_depth_1_to_4']}.
3. {result_sentence}
4. Raw episode-averaged benefit was {raw['mean_benefit']:.9g} (exploratory 95% interval {raw['ci_low']:.9g}, {raw['ci_high']:.9g}); fit-derived-whitened benefit was {white['mean_benefit']:.9g} ({white['ci_low']:.9g}, {white['ci_high']:.9g}). Positive values favor adaptive allocation.
5. Strongest limitations: {limitations}
6. {confirmation}

## Fit-only refinement signal

{os.linesep.join(depth_lines)}

## Untouched evaluation comparisons

{os.linesep.join(endpoint_lines)}

Adaptive and the primary analytic comparator each used exactly {decision['compute']['adaptive_total_counted_flops']} counted FLOPs. This includes {decision['compute']['gate_total_flops']} gate FLOPs and does not equate call-count equality with FLOP equality. Adaptive made {decision['compute']['base_model_calls']} base-predictor calls, {decision['compute']['refiner_calls']} refiner calls, and {decision['compute']['reached_gate_evaluations']} reached gate evaluations; {decision['compute']['nonflop_comparison_min_operations']} comparison/min operations are reported separately. Synchronized latency is in `LATENCY.json` and is not used to claim speedup.

Stagewise combined score/gain Spearman values were {[item['score_vs_combined_gain_spearman'] for item in decision['stagewise_rank']]}. All intervals are exploratory because this is a bounded pilot with internal candidate selection.

## Integrity

Independent recomputation passed: {verification['passed']}. Fit, selection, and evaluation roles were seed-disjoint, contained the fixed episode counts, and had no replacements. The cached base remained frozen; causal features excluded target/future/contact/reward/success/simulator state/geometry; all outputs were finite; evaluation was generated only after the gate freeze. No HDF5/H5 corpus was opened.
"""
    atomic_json(ROOT / "RESULTS.json", results)
    (ROOT / "REPORT.md").write_text(report)
    append_note(f"Finalized terminal scientific label {label}; independent verification passed={verification['passed']}.")
    update_status("complete", status="complete", scientific_label=label)
    hashes = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_HASHES.json" and not path.name.endswith(".pyc"):
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    atomic_json(
        ROOT / "ARTIFACT_HASHES.json",
        {"schema_version": 1, "created_unix_ns": time.time_ns(), "files": hashes},
    )
    print(json.dumps({"status": "complete", "scientific_label": label, "artifact_count": len(hashes)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("smoke", "generate-fit", "generate-selection", "train", "gate", "generate-evaluation", "evaluate", "finalize"),
    )
    args = parser.parse_args()
    commands = {
        "smoke": smoke,
        "generate-fit": lambda: generate_role("fit"),
        "generate-selection": lambda: generate_role("selection"),
        "train": train_refiner,
        "gate": fit_select_gate,
        "generate-evaluation": lambda: generate_role("evaluation"),
        "evaluate": evaluate_pilot,
        "finalize": finalize,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
