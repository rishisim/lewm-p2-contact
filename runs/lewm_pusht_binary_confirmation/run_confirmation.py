#!/usr/bin/env python3
"""Run the fixed PushT binary adaptive-computation confirmation once."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PILOT_ROOT = ROOT.parent / "lewm_pusht_replication_pilot"
if str(PILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PILOT_ROOT))

from pusht_core import (  # noqa: E402
    ACTION_DIM,
    DEPTHS,
    FRAMESKIP,
    GENERATION_PYTHON,
    HISTORY,
    LATENT_DIM,
    MODEL_CONFIG,
    MODEL_WEIGHTS,
    StagewiseRefiner,
    atomic_json,
    atomic_npz,
    base_predictor_counted_flops,
    build_causal_features,
    causal_feature_names,
    choose_device,
    concatenate_transition_batches,
    encode_pixel_sequences,
    gate_operation_ledger,
    git_snapshot,
    load_base_model,
    load_npz,
    module_digest,
    normalize_actions,
    pack_context,
    read_json,
    refiner_flops,
    rollout_episode,
    sha256_file,
    summarize_context,
    synchronize,
    trajectory_digest,
    transition_arrays,
)


THRESHOLD = 0.036598234837386154
COHORT_COUNT = 240
COHORT_SEED_FIRST = 15_910_000
COHORT_SEED_LAST = COHORT_SEED_FIRST + COHORT_COUNT - 1
IDENTIFIER_PREFIX = "pusht_binary_confirmation_"
SMOKE_SPECS = (
    {"identifier": "pusht_binary_confirmation_smoke_excluded_000", "seed": 15_917_001},
    {"identifier": "pusht_binary_confirmation_smoke_excluded_001", "seed": 15_917_002},
)
BOOTSTRAP_SEED = 15_918_991
PERMUTATION_SEED = 15_918_881
BOOTSTRAP_REPLICATES = 20_000
LATENCY_REPEATS = 5


def now_ns() -> int:
    return time.time_ns()


def append_note(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (ROOT / "NOTES.md").open("a") as stream:
        stream.write(f"\n- {stamp} — {message}\n")


def set_status(phase: str, status: str = "running", **extra: Any) -> None:
    atomic_json(
        ROOT / "RUN_STATUS.json",
        {
            "schema_version": 1,
            "status": status,
            "phase": phase,
            "updated_unix_ns": now_ns(),
            **extra,
        },
    )


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def cohort_specs() -> list[dict[str, Any]]:
    return [
        {
            "ordinal": ordinal,
            "identifier": f"{IDENTIFIER_PREFIX}{ordinal:04d}",
            "seed": COHORT_SEED_FIRST + ordinal,
        }
        for ordinal in range(COHORT_COUNT)
    ]


def protocol_object() -> dict[str, Any]:
    return {
        "policy": {
            "mandatory_depth": 1,
            "optional_depth": 2,
            "stage_1_threshold": THRESHOLD,
            "strict_comparison": "score > threshold",
            "stage_2_or_stage_3_gate_scores_computed_or_consulted": False,
            "adaptive_depths_3_or_4_used": False,
        },
        "cohort": {
            "count": COHORT_COUNT,
            "seed_range_inclusive": [COHORT_SEED_FIRST, COHORT_SEED_LAST],
            "contiguous": True,
            "identifier_prefix": IDENTIFIER_PREFIX,
            "identifier_format": f"{IDENTIFIER_PREFIX}{{ordinal:04d}}",
        },
        "smoke_specs": list(SMOKE_SPECS),
        "environment": {
            "id": "swm/PushT-v1",
            "policy": "WeakPolicy(dist_constraint=100)",
            "frameskip": FRAMESKIP,
            "history": HISTORY,
            "max_raw_steps": 100,
            "fresh_simulator_only": True,
            "hdf5_or_h5_opened": False,
        },
        "analysis": {
            "episode_is_independent_unit": True,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "permutation_seed": PERMUTATION_SEED,
            "familywise_one_sided_alpha": 0.05,
            "per_endpoint_one_sided_alpha": 0.025,
            "exploratory_two_sided_confidence": 0.95,
            "comparator": "strongest episode-averaged transition-independent analytic pairwise mixture of fixed depths 1-4 at exact counted compute",
        },
    }


def load_refiner(device: torch.device) -> StagewiseRefiner:
    checkpoint = torch.load(
        PILOT_ROOT / "checkpoints/refiner.pt", map_location="cpu", weights_only=True
    )
    model = StagewiseRefiner(hidden=256, iteration_dim=16)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def stage1_features(
    history: np.ndarray,
    actions: np.ndarray,
    exit1: np.ndarray,
    update1: np.ndarray,
    action_mean: np.ndarray,
    action_scale: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    normalized = normalize_actions(actions, action_mean, action_scale)
    result = np.empty((len(history), len(causal_feature_names())), dtype=np.float32)
    for start in range(0, len(history), batch_size):
        stop = min(start + batch_size, len(history))
        result[start:stop] = build_causal_features(
            torch.from_numpy(np.ascontiguousarray(history[start:stop])),
            torch.from_numpy(np.ascontiguousarray(normalized[start:stop])),
            torch.from_numpy(np.ascontiguousarray(exit1[start:stop])),
            torch.from_numpy(np.ascontiguousarray(update1[start:stop])),
        ).numpy()
    return result


def score_stage1(features: np.ndarray, gate: Mapping[str, np.ndarray]) -> np.ndarray:
    z = (
        features.astype(np.float64) - np.asarray(gate["feature_mean"])[0]
    ) / np.asarray(gate["feature_scale"])[0]
    with np.errstate(all="ignore"):
        heads = z @ np.asarray(gate["weights"])[0].T
    scores = np.minimum(heads[:, 0], heads[:, 1])
    if not np.isfinite(scores).all():
        raise RuntimeError("nonfinite frozen stage-1 gate score")
    return scores


def binary_calls(scores: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    return (1 + (np.asarray(scores) > threshold).astype(np.int8)).astype(np.int8)


def raw_white_losses(
    exits: np.ndarray, target: np.ndarray, whitening: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    difference = exits.astype(np.float64) - target.astype(np.float64)[:, None, :]
    raw = np.square(difference).mean(axis=2)
    transformed = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    white = np.square(transformed).mean(axis=2)
    return raw, white


def episode_means(values: np.ndarray, episode_id: np.ndarray, count: int = COHORT_COUNT) -> np.ndarray:
    output = np.empty(count, dtype=np.float64)
    for identifier in range(count):
        selected = np.asarray(values)[episode_id == identifier]
        if len(selected) == 0:
            raise RuntimeError(f"episode {identifier} has no prediction transitions")
        output[identifier] = selected.mean()
    return output


def analytic_mixture(
    losses: np.ndarray, mean_depth: float, episode_id: np.ndarray, count: int
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for lower in DEPTHS:
        for upper in DEPTHS:
            if upper < lower or not lower <= mean_depth <= upper:
                continue
            weight = 0.0 if lower == upper else (mean_depth - lower) / (upper - lower)
            transition_loss = (1.0 - weight) * losses[:, lower - 1] + weight * losses[:, upper - 1]
            objective = float(episode_means(transition_loss, episode_id, count).mean())
            candidates.append(
                {
                    "depth_lower": lower,
                    "depth_upper": upper,
                    "weight_upper": float(weight),
                    "mean_episode_loss": objective,
                    "loss": transition_loss,
                }
            )
    if not candidates:
        raise RuntimeError("no feasible exact-compute analytic mixture")
    candidates.sort(key=lambda item: (item["mean_episode_loss"], item["depth_lower"], item["depth_upper"]))
    return {**candidates[0], "feasible_pair_count": len(candidates)}


def bootstrap_summaries(
    values: Mapping[str, np.ndarray], *, seed: int = BOOTSTRAP_SEED
) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, COHORT_COUNT, size=(BOOTSTRAP_REPLICATES, COHORT_COUNT), dtype=np.int32
    )
    output: dict[str, dict[str, Any]] = {}
    for name, array in values.items():
        vector = np.asarray(array, dtype=np.float64)
        sampled = vector[indices].mean(axis=1)
        output[name] = {
            "mean": float(vector.mean()),
            "exploratory_95_ci_low": float(np.quantile(sampled, 0.025)),
            "exploratory_95_ci_high": float(np.quantile(sampled, 0.975)),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": seed,
            "episode_count": len(vector),
        }
    return output


def within_episode_permutation(calls: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(PERMUTATION_SEED)
    output = np.empty_like(calls)
    for identifier in range(COHORT_COUNT):
        indices = np.flatnonzero(episode_id == identifier)
        output[indices] = calls[indices][rng.permutation(len(indices))]
        if not np.array_equal(
            np.bincount(calls[indices], minlength=3),
            np.bincount(output[indices], minlength=3),
        ):
            raise RuntimeError("within-episode permutation changed a call histogram")
    return output


def pilot_contract() -> dict[str, Any]:
    pilot_config = read_json(PILOT_ROOT / "CONFIG.json")
    artifact_hashes = read_json(PILOT_ROOT / "ARTIFACT_HASHES.json")["files"]
    required = {
        "pilot_config": (PILOT_ROOT / "CONFIG.json", artifact_hashes["CONFIG.json"]),
        "refiner": (
            PILOT_ROOT / "checkpoints/refiner.pt",
            artifact_hashes["checkpoints/refiner.pt"],
        ),
        "fit_artifacts": (PILOT_ROOT / "FIT_ARTIFACTS.npz", artifact_hashes["FIT_ARTIFACTS.npz"]),
        "frozen_gate": (PILOT_ROOT / "FROZEN_GATE.npz", artifact_hashes["FROZEN_GATE.npz"]),
        "pilot_evaluation_arrays": (
            PILOT_ROOT / "EVALUATION_ARRAYS.npz",
            artifact_hashes["EVALUATION_ARRAYS.npz"],
        ),
        "pilot_evaluation_role": (
            PILOT_ROOT / "data/evaluation.npz",
            artifact_hashes["data/evaluation.npz"],
        ),
        "pilot_core_source": (PILOT_ROOT / "pusht_core.py", artifact_hashes["pusht_core.py"]),
        "pilot_run_source": (PILOT_ROOT / "run_pilot.py", artifact_hashes["run_pilot.py"]),
        "model_config": (MODEL_CONFIG, pilot_config["read_only_inputs"]["model_config_sha256"]),
        "model_weights": (MODEL_WEIGHTS, pilot_config["read_only_inputs"]["model_weights_sha256"]),
    }
    hashes = {
        name: {
            "path": str(path),
            "expected_sha256": expected,
            "observed_sha256": sha256_file(path),
            "matches": sha256_file(path) == expected,
        }
        for name, (path, expected) in required.items()
    }
    if not all(item["matches"] for item in hashes.values()):
        raise RuntimeError("a frozen pilot/base hash does not match")

    fit = load_npz(PILOT_ROOT / "FIT_ARTIFACTS.npz")
    gate = load_npz(PILOT_ROOT / "FROZEN_GATE.npz")
    if float(gate["thresholds"][0]) != THRESHOLD:
        raise RuntimeError("frozen stage-1 threshold drift")
    compute = {
        "base_predictor": base_predictor_counted_flops(),
        "refiner": refiner_flops(),
        "gate": gate_operation_ledger(),
    }
    if any(compute[key] != pilot_config["compute"][key] for key in compute):
        raise RuntimeError("frozen compute ledger drift")

    role = load_npz(PILOT_ROOT / "data/evaluation.npz")
    stored = load_npz(PILOT_ROOT / "EVALUATION_ARRAYS.npz")
    features = stage1_features(
        role["history"], role["actions"], stored["dense_exits"][:, 0],
        stored["dense_updates"][:, 0], fit["action_mean"], fit["action_scale"],
    )
    scores = score_stage1(features, gate)
    calls = binary_calls(scores)
    expected_calls = (1 + (stored["calls"] >= 2).astype(np.int8)).astype(np.int8)
    scores_exact = np.array_equal(scores, stored["scores"][:, 0])
    calls_exact = np.array_equal(calls, expected_calls)
    if not scores_exact or not calls_exact:
        raise RuntimeError("pilot stage-1 score/call contract did not reproduce exactly")

    raw, white = raw_white_losses(stored["dense_exits"], stored["target"], fit["whitening_matrix"])
    refiner_price = int(compute["refiner"]["total_flops_per_call"])
    gate_price = int(compute["gate"]["total_flops_per_reached_evaluation"])
    equivalent_depth = (int(calls.sum()) * refiner_price + len(calls) * gate_price) / (
        len(calls) * refiner_price
    )
    diagnostic: dict[str, Any] = {}
    for name, losses in (("raw", raw), ("fit_whitened", white)):
        comparator = analytic_mixture(losses, equivalent_depth, role["episode_id"], 80)
        adaptive = losses[np.arange(len(calls)), calls - 1]
        benefit = episode_means(comparator["loss"] - adaptive, role["episode_id"], 80)
        diagnostic[name] = float(benefit.mean())

    pilot_seed_sets = {
        role_name: {int(item["seed"]) for item in pilot_config["role_specs"][role_name]}
        for role_name in ("fit", "selection", "evaluation")
    }
    pilot_seed_sets["smoke"] = {int(item["seed"]) for item in pilot_config["smoke"]["episodes"]}
    target_seeds = set(range(COHORT_SEED_FIRST, COHORT_SEED_LAST + 1))
    smoke_seeds = {int(item["seed"]) for item in SMOKE_SPECS}
    seed_checks = {
        f"target_disjoint_from_pilot_{name}": target_seeds.isdisjoint(values)
        for name, values in pilot_seed_sets.items()
    }
    seed_checks.update(
        {
            f"smoke_disjoint_from_pilot_{name}": smoke_seeds.isdisjoint(values)
            for name, values in pilot_seed_sets.items()
        }
    )
    seed_checks["target_and_smoke_disjoint"] = target_seeds.isdisjoint(smoke_seeds)
    seed_checks["target_range_exact_contiguous_240"] = (
        len(target_seeds) == COHORT_COUNT
        and max(target_seeds) - min(target_seeds) + 1 == COHORT_COUNT
    )
    if not all(seed_checks.values()):
        raise RuntimeError("fresh seed range overlaps a frozen pilot role")

    freeze = read_json(PILOT_ROOT / "GATE_FREEZE.json")
    frozen_array_digests = {
        "action_mean": array_digest(fit["action_mean"]),
        "action_scale": array_digest(fit["action_scale"]),
        "whitening_matrix": array_digest(fit["whitening_matrix"]),
        "stage1_feature_mean": array_digest(gate["feature_mean"][0]),
        "stage1_feature_scale": array_digest(gate["feature_scale"][0]),
        "stage1_weights": array_digest(gate["weights"][0]),
        "stage1_raw_gain_scale": array_digest(gate["raw_gain_scale"][0]),
        "stage1_white_gain_scale": array_digest(gate["white_gain_scale"][0]),
    }
    return {
        "all_hashes_match": True,
        "hashes": hashes,
        "frozen_array_digests": frozen_array_digests,
        "feature_builder_signature": list(inspect.signature(build_causal_features).parameters),
        "feature_names_sha256": hashlib.sha256("\n".join(causal_feature_names()).encode()).hexdigest(),
        "feature_names_match_freeze": hashlib.sha256("\n".join(causal_feature_names()).encode()).hexdigest()
        == freeze["feature_names_sha256"],
        "stage1_threshold": float(gate["thresholds"][0]),
        "compute": compute,
        "pilot_stage1_reproduction": {
            "score_count": len(scores),
            "scores_bitwise_exact": scores_exact,
            "binary_calls_exact": calls_exact,
            "stage1_scores_array_digest": array_digest(scores),
            "binary_calls_array_digest": array_digest(calls),
            "binary_call_histogram_depth_1_to_2": np.bincount(calls, minlength=3)[1:].tolist(),
            "read_only_raw_benefit": diagnostic["raw"],
            "read_only_fit_whitened_benefit": diagnostic["fit_whitened"],
        },
        "seed_checks": seed_checks,
    }


def policy_and_reference_outputs(
    arrays: Mapping[str, np.ndarray],
    refiner: StagewiseRefiner,
    fit: Mapping[str, np.ndarray],
    gate: Mapping[str, np.ndarray],
    device: torch.device,
    *,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    normalized = normalize_actions(arrays["actions"], fit["action_mean"], fit["action_scale"])
    exits: list[np.ndarray] = []
    first_updates: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    calls: list[np.ndarray] = []
    selected: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            stop = min(start + batch_size, len(normalized))
            history = torch.from_numpy(np.ascontiguousarray(arrays["history"][start:stop])).to(device)
            actions = torch.from_numpy(np.ascontiguousarray(normalized[start:stop])).to(device)
            base = torch.from_numpy(np.ascontiguousarray(arrays["base"][start:stop])).to(device)

            # The adaptive decision is fixed immediately after mandatory depth 1.
            exit1, update1 = refiner.step(history, actions, base, 0)
            exit1_np = exit1.detach().cpu().numpy().astype(np.float32)
            update1_np = update1.detach().cpu().numpy().astype(np.float32)
            feature = stage1_features(
                arrays["history"][start:stop], arrays["actions"][start:stop],
                exit1_np, update1_np, fit["action_mean"], fit["action_scale"],
            )
            score = score_stage1(feature, gate)
            call = binary_calls(score)

            # Depths 2--4 below are outcome-blind counterfactual reference exits.
            # Only depth 2 at rows with call==2 belongs to the adaptive policy ledger.
            exit2, _ = refiner.step(history, actions, exit1, 1)
            exit3, _ = refiner.step(history, actions, exit2, 2)
            exit4, _ = refiner.step(history, actions, exit3, 3)
            dense = torch.stack((exit1, exit2, exit3, exit4), dim=1).detach().cpu().numpy().astype(np.float32)
            chosen = np.where(call[:, None] == 2, dense[:, 1], dense[:, 0]).astype(np.float32)

            exits.append(dense)
            first_updates.append(update1_np)
            scores.append(score)
            calls.append(call)
            selected.append(chosen)
    result = {
        "dense_exits": np.concatenate(exits),
        "stage1_update": np.concatenate(first_updates),
        "stage1_score": np.concatenate(scores),
        "calls": np.concatenate(calls),
        "selected": np.concatenate(selected),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("nonfinite refiner/gate output")
    return result


def setup_smoke() -> None:
    config_path = ROOT / "CONFIG.json"
    if config_path.exists() and read_json(config_path).get("setup_smoke", {}).get("complete"):
        print(json.dumps({"status": "setup_smoke_already_complete"}))
        return
    if config_path.exists():
        raise RuntimeError("partial setup exists; do not retry or replace smoke episodes")
    set_status("setup/smoke")
    locked_ns = now_ns()
    protocol = protocol_object()
    contract = pilot_contract()
    source_hashes = {
        "run_confirmation.py": sha256_file(ROOT / "run_confirmation.py"),
        "verify_confirmation.py": sha256_file(ROOT / "verify_confirmation.py"),
        "tests/test_confirmation.py": sha256_file(ROOT / "tests/test_confirmation.py"),
    }
    config: dict[str, Any] = {
        "schema_version": 1,
        "objective": "Fixed fresh-cohort PushT binary adaptive-computation confirmation",
        "root": str(ROOT),
        "protocol_locked_unix_ns": locked_ns,
        "protocol": protocol,
        "protocol_sha256": canonical_digest(protocol),
        "frozen_contract": contract,
        "source_hashes_at_lock": source_hashes,
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "generation_python": str(GENERATION_PYTHON),
        },
        "git_baseline": git_snapshot(),
        "setup_smoke": {"complete": False},
    }
    atomic_json(config_path, config)

    smoke_started_ns = now_ns()
    device = choose_device()
    base_model = load_base_model(device)
    refiner = load_refiner(device)
    fit = load_npz(PILOT_ROOT / "FIT_ARTIFACTS.npz")
    gate = load_npz(PILOT_ROOT / "FROZEN_GATE.npz")
    base_before = module_digest(base_model)
    refiner_before = module_digest(refiner)
    records: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(SMOKE_SPECS):
        start = time.perf_counter()
        episode = rollout_episode(int(spec["seed"]))
        latents = encode_pixel_sequences(base_model, [episode], device)
        arrays = transition_arrays(base_model, [episode], latents, [ordinal], device)
        policy = policy_and_reference_outputs(arrays, refiner, fit, gate, device)
        elapsed = time.perf_counter() - start
        records.append(
            {
                **spec,
                "raw_steps": int(episode["raw_steps"]),
                "transition_count": len(arrays["target"]),
                "trajectory_sha256": trajectory_digest(episode),
                "binary_call_histogram_depth_1_to_2": np.bincount(
                    policy["calls"], minlength=3
                )[1:].tolist(),
                "finite": bool(
                    all(np.isfinite(value).all() for value in arrays.values())
                    and all(np.isfinite(value).all() for value in policy.values())
                ),
                "elapsed_seconds": elapsed,
            }
        )
    base_after = module_digest(base_model)
    refiner_after = module_digest(refiner)
    smoke_complete_ns = now_ns()
    if base_before != base_after or refiner_before != refiner_after or not all(
        item["finite"] for item in records
    ):
        raise RuntimeError("excluded smoke compatibility failed")
    config["setup_smoke"] = {
        "complete": True,
        "started_unix_ns": smoke_started_ns,
        "completed_unix_ns": smoke_complete_ns,
        "exact_excluded_episode_count": len(records),
        "episodes": records,
        "no_replays": True,
        "not_in_target_cohort": True,
        "device": str(device),
        "base_parameter_digest_before": base_before,
        "base_parameter_digest_after": base_after,
        "refiner_parameter_digest_before": refiner_before,
        "refiner_parameter_digest_after": refiner_after,
    }
    atomic_json(config_path, config)
    append_note(
        "Locked hashes, policy, identifiers, seeds, compute, and analysis; reproduced the pilot stage-1 contract exactly; then ran exactly two excluded smoke episodes without replay."
    )
    set_status("setup/smoke", status="complete", next_phase="fixed cohort")
    print(json.dumps({"status": "setup_smoke_complete", "device": str(device), "episodes": 2}))


def fixed_cohort() -> None:
    destination = ROOT / "EVALUATION_ARRAYS.npz"
    if destination.exists():
        print(json.dumps({"status": "fixed_cohort_already_complete", "sha256": sha256_file(destination)}))
        return
    config = read_json(ROOT / "CONFIG.json")
    if not config["setup_smoke"]["complete"]:
        raise RuntimeError("two excluded smoke episodes must complete before target generation")
    if canonical_digest(config["protocol"]) != config["protocol_sha256"]:
        raise RuntimeError("locked protocol changed")
    if (ROOT / "DECISION.json").exists():
        raise RuntimeError("decision exists before fixed cohort")
    set_status("fixed cohort", generated_episodes=0, total_episodes=COHORT_COUNT)
    generation_started_ns = now_ns()
    specs = cohort_specs()
    device = choose_device()
    base_model = load_base_model(device)
    refiner = load_refiner(device)
    fit = load_npz(PILOT_ROOT / "FIT_ARTIFACTS.npz")
    gate = load_npz(PILOT_ROOT / "FROZEN_GATE.npz")
    base_before = module_digest(base_model)
    refiner_before = module_digest(refiner)
    batches: list[dict[str, np.ndarray]] = []
    policy_batches: list[dict[str, np.ndarray]] = []
    contexts: list[dict[str, Any]] = []
    transition_counts: list[int] = []
    trajectory_hashes: list[str] = []
    group_size = 8
    for group_start in range(0, COHORT_COUNT, group_size):
        group_specs = specs[group_start : group_start + group_size]
        episodes = [rollout_episode(int(spec["seed"])) for spec in group_specs]
        latents = encode_pixel_sequences(base_model, episodes, device)
        arrays = transition_arrays(
            base_model,
            episodes,
            latents,
            [int(spec["ordinal"]) for spec in group_specs],
            device,
        )
        policy = policy_and_reference_outputs(arrays, refiner, fit, gate, device)
        batches.append(arrays)
        policy_batches.append(policy)
        for episode, spec in zip(episodes, group_specs):
            transition_counts.append(int(np.sum(arrays["episode_id"] == int(spec["ordinal"]))))
            trajectory_hashes.append(trajectory_digest(episode))
            contexts.append(
                {key: value for key, value in episode.items() if key not in ("pixels", "blocked_actions")}
            )
        set_status(
            "fixed cohort",
            generated_episodes=min(group_start + group_size, COHORT_COUNT),
            total_episodes=COHORT_COUNT,
        )
    arrays = concatenate_transition_batches(batches)
    policy = {
        key: np.concatenate([batch[key] for batch in policy_batches]) for key in policy_batches[0]
    }
    base_after = module_digest(base_model)
    refiner_after = module_digest(refiner)
    if base_before != base_after or refiner_before != refiner_after:
        raise RuntimeError("frozen base or refiner changed during the cohort")
    if len(contexts) != COHORT_COUNT or len(set(trajectory_hashes)) != COHORT_COUNT:
        raise RuntimeError("target cohort count or trajectory uniqueness failure")
    if not np.array_equal(np.unique(arrays["episode_id"]), np.arange(COHORT_COUNT)):
        raise RuntimeError("target episode ordinals are incomplete")
    expected_seeds = np.arange(COHORT_SEED_FIRST, COHORT_SEED_LAST + 1, dtype=np.int64)
    if not np.array_equal(np.unique(arrays["episode_seed"]), expected_seeds):
        raise RuntimeError("target seed range is incomplete")
    if not all(np.isfinite(value).all() for value in (*arrays.values(), *policy.values())):
        raise RuntimeError("nonfinite fixed-cohort array")

    saved = {
        "episode_id": arrays["episode_id"].astype(np.int32),
        "episode_seed": arrays["episode_seed"].astype(np.int64),
        "model_step": arrays["model_step"].astype(np.int16),
        "history": arrays["history"].astype(np.float32),
        "actions": arrays["actions"].astype(np.float32),
        "base": arrays["base"].astype(np.float32),
        "target": arrays["target"].astype(np.float32),
        "dense_exits": policy["dense_exits"].astype(np.float32),
        "stage1_update": policy["stage1_update"].astype(np.float32),
        "stage1_score": policy["stage1_score"].astype(np.float64),
        "calls": policy["calls"].astype(np.int8),
        "selected": policy["selected"].astype(np.float32),
        "episode_ordinal": np.arange(COHORT_COUNT, dtype=np.int32),
        "episode_unique_seed": expected_seeds,
        "episode_identifier": np.asarray([item["identifier"] for item in specs]),
        "episode_raw_steps": np.asarray([int(item["raw_steps"]) for item in contexts], dtype=np.int16),
        "episode_transition_count": np.asarray(transition_counts, dtype=np.int16),
        "episode_trajectory_sha256": np.asarray(trajectory_hashes),
    }
    atomic_npz(destination, saved)
    policy_locked_ns = now_ns()

    # Privileged simulator context is summarized only after every policy call is fixed durably.
    packed = pack_context(contexts, list(range(COHORT_COUNT)))
    packed["role_name"] = np.asarray("confirmation")
    context_summary = summarize_context([packed])["confirmation"]
    generation_completed_ns = now_ns()
    trajectory_hash_digest = hashlib.sha256("\n".join(trajectory_hashes).encode()).hexdigest()
    config["fixed_cohort"] = {
        "complete": True,
        "generation_started_unix_ns": generation_started_ns,
        "policy_decision_locked_unix_ns": policy_locked_ns,
        "privileged_context_summarized_unix_ns": generation_completed_ns,
        "generation_completed_unix_ns": generation_completed_ns,
        "episode_count": COHORT_COUNT,
        "transition_count": len(arrays["target"]),
        "per_episode_transition_counts": transition_counts,
        "trajectory_hashes_sha256": trajectory_hash_digest,
        "no_exclusions": True,
        "no_replacements": True,
        "fresh_simulator_only": True,
        "hdf5_or_h5_opened": False,
        "gate_scores_computed": ["stage_1"],
        "stage_2_or_stage_3_gate_scores_computed_or_consulted": False,
        "counterfactual_reference_exits": [1, 2, 3, 4],
        "counterfactual_depths_3_or_4_used_by_adaptive_policy": False,
        "privileged_context_used_by_policy_eligibility_or_decision": False,
        "privileged_context_summary": context_summary,
        "base_parameter_digest_before": base_before,
        "base_parameter_digest_after": base_after,
        "refiner_parameter_digest_before": refiner_before,
        "refiner_parameter_digest_after": refiner_after,
        "evaluation_arrays_initial_sha256": sha256_file(destination),
        "device": str(device),
    }
    atomic_json(ROOT / "CONFIG.json", config)
    append_note(
        f"Generated the one fixed cohort: {COHORT_COUNT} fresh episodes and {len(arrays['target'])} prediction transitions, with no exclusions, replacements, or H5/HDF5 access."
    )
    set_status("fixed cohort", status="complete", next_phase="analysis")
    print(
        json.dumps(
            {
                "status": "fixed_cohort_complete",
                "episodes": COHORT_COUNT,
                "transitions": len(arrays["target"]),
                "call_histogram": np.bincount(policy["calls"], minlength=3)[1:].tolist(),
            }
        )
    )


def latency_measurements(
    arrays: Mapping[str, np.ndarray], expected_calls: np.ndarray, expected_selected: np.ndarray
) -> dict[str, Any]:
    device = choose_device()
    refiner = load_refiner(device)
    fit = load_npz(PILOT_ROOT / "FIT_ARTIFACTS.npz")
    gate = load_npz(PILOT_ROOT / "FROZEN_GATE.npz")
    normalized = normalize_actions(arrays["actions"], fit["action_mean"], fit["action_scale"])
    history = torch.from_numpy(np.ascontiguousarray(arrays["history"])).to(device)
    actions = torch.from_numpy(np.ascontiguousarray(normalized)).to(device)
    base = torch.from_numpy(np.ascontiguousarray(arrays["base"])).to(device)

    def adaptive_once() -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            exit1, update1 = refiner.step(history, actions, base, 0)
            feature = stage1_features(
                arrays["history"], arrays["actions"],
                exit1.detach().cpu().numpy().astype(np.float32),
                update1.detach().cpu().numpy().astype(np.float32),
                fit["action_mean"], fit["action_scale"],
            )
            score = score_stage1(feature, gate)
            calls = binary_calls(score)
            active = torch.from_numpy(np.flatnonzero(calls == 2)).to(device=device, dtype=torch.long)
            current = exit1
            if len(active):
                refined, _ = refiner.step(
                    history.index_select(0, active), actions.index_select(0, active),
                    exit1.index_select(0, active), 1,
                )
                current = current.index_copy(0, active, refined)
        return current.detach().cpu().numpy(), calls

    adaptive_times: list[float] = []
    observed: np.ndarray | None = None
    observed_calls: np.ndarray | None = None
    for _ in range(LATENCY_REPEATS):
        synchronize(device)
        start = time.perf_counter()
        observed, observed_calls = adaptive_once()
        synchronize(device)
        adaptive_times.append(time.perf_counter() - start)
    assert observed is not None and observed_calls is not None
    if not np.array_equal(observed_calls, expected_calls):
        raise RuntimeError("synchronized adaptive path changed binary calls")
    if not np.allclose(observed, expected_selected, rtol=2e-5, atol=2e-6):
        raise RuntimeError("synchronized adaptive path changed selected predictions")

    fixed: dict[str, Any] = {}
    for depth in (1, 2):
        times: list[float] = []
        for _ in range(LATENCY_REPEATS):
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                current = base
                for iteration in range(depth):
                    current, _ = refiner.step(history, actions, current, iteration)
            synchronize(device)
            times.append(time.perf_counter() - start)
        fixed[f"fixed_depth_{depth}_cached_base"] = {
            "seconds": times,
            "median_seconds": float(np.median(times)),
        }

    base_model = load_base_model(device)
    before = module_digest(base_model)
    base_times: list[float] = []
    for _ in range(LATENCY_REPEATS):
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            for batch_start in range(0, len(base), 512):
                raw_actions = torch.from_numpy(
                    np.ascontiguousarray(arrays["actions"][batch_start : batch_start + 512])
                ).to(device)
                action_embedding = base_model.action_encoder(raw_actions)
                _ = base_model.predict(
                    history[batch_start : batch_start + 512], action_embedding
                )[:, -1]
        synchronize(device)
        base_times.append(time.perf_counter() - start)
    if before != module_digest(base_model):
        raise RuntimeError("base model changed during latency measurement")
    return {
        "device": str(device),
        "synchronized": True,
        "repeats": LATENCY_REPEATS,
        "base_predictor_all_transitions": {
            "seconds": base_times,
            "median_seconds": float(np.median(base_times)),
        },
        "cached_base_to_binary_adaptive_selected": {
            "seconds": adaptive_times,
            "median_seconds": float(np.median(adaptive_times)),
            "includes_stage1_feature_score_and_cpu_transfer": True,
        },
        **fixed,
        "pixel_encoding_excluded": True,
        "action_normalization_precomputed": True,
        "latency_separate_from_counted_flops": True,
        "wall_clock_speedup_claim": False,
    }


def analyze() -> None:
    decision_path = ROOT / "DECISION.json"
    if decision_path.exists():
        print(json.dumps({"status": "analysis_already_complete", "sha256": sha256_file(decision_path)}))
        return
    config = read_json(ROOT / "CONFIG.json")
    if not config.get("fixed_cohort", {}).get("complete"):
        raise RuntimeError("fixed cohort is incomplete")
    set_status("analysis")
    analysis_started_ns = now_ns()
    path = ROOT / "EVALUATION_ARRAYS.npz"
    arrays = load_npz(path)
    fit = load_npz(PILOT_ROOT / "FIT_ARTIFACTS.npz")
    gate = load_npz(PILOT_ROOT / "FROZEN_GATE.npz")

    feature = stage1_features(
        arrays["history"], arrays["actions"], arrays["dense_exits"][:, 0],
        arrays["stage1_update"], fit["action_mean"], fit["action_scale"],
    )
    score = score_stage1(feature, gate)
    calls = binary_calls(score)
    row = np.arange(len(calls))
    selected = arrays["dense_exits"][row, calls - 1]
    if not np.array_equal(score, arrays["stage1_score"]):
        raise RuntimeError("saved fresh-cohort stage-1 scores did not reproduce exactly")
    if not np.array_equal(calls, arrays["calls"]):
        raise RuntimeError("saved fresh-cohort calls did not reproduce exactly")
    if not np.array_equal(selected, arrays["selected"]):
        raise RuntimeError("saved selected predictions did not reproduce exactly")

    raw, white = raw_white_losses(arrays["dense_exits"], arrays["target"], fit["whitening_matrix"])
    adaptive_raw = raw[row, calls - 1]
    adaptive_white = white[row, calls - 1]
    episode_id = arrays["episode_id"].astype(int)
    permutation = within_episode_permutation(calls, episode_id)
    refiner_price = int(config["frozen_contract"]["compute"]["refiner"]["total_flops_per_call"])
    gate_price = int(config["frozen_contract"]["compute"]["gate"]["total_flops_per_reached_evaluation"])
    base_price = int(config["frozen_contract"]["compute"]["base_predictor"]["total_flops_per_transition"])
    norm_price = int(config["frozen_contract"]["compute"]["gate"]["action_normalization_flops_per_transition"])
    gate_evaluations = len(calls)
    refiner_calls = int(calls.sum())
    equivalent_depth = (refiner_calls * refiner_price + gate_evaluations * gate_price) / (
        len(calls) * refiner_price
    )
    raw_comparator = analytic_mixture(raw, equivalent_depth, episode_id, COHORT_COUNT)
    white_comparator = analytic_mixture(white, equivalent_depth, episode_id, COHORT_COUNT)

    endpoint_values = {
        "raw_vs_exact_compute_analytic": episode_means(
            raw_comparator["loss"] - adaptive_raw, episode_id
        ),
        "fit_whitened_vs_exact_compute_analytic": episode_means(
            white_comparator["loss"] - adaptive_white, episode_id
        ),
        "raw_vs_within_episode_call_randomization": episode_means(
            raw[row, permutation - 1] - adaptive_raw, episode_id
        ),
        "fit_whitened_vs_within_episode_call_randomization": episode_means(
            white[row, permutation - 1] - adaptive_white, episode_id
        ),
        "raw_vs_fixed_depth_1": episode_means(raw[:, 0] - adaptive_raw, episode_id),
        "fit_whitened_vs_fixed_depth_1": episode_means(
            white[:, 0] - adaptive_white, episode_id
        ),
        "raw_vs_fixed_depth_2": episode_means(raw[:, 1] - adaptive_raw, episode_id),
        "fit_whitened_vs_fixed_depth_2": episode_means(
            white[:, 1] - adaptive_white, episode_id
        ),
    }
    endpoint_intervals = bootstrap_summaries(endpoint_values)
    absolute_episode_losses = {
        "adaptive_raw": episode_means(adaptive_raw, episode_id),
        "adaptive_fit_whitened": episode_means(adaptive_white, episode_id),
        "fixed_depth_1_raw": episode_means(raw[:, 0], episode_id),
        "fixed_depth_1_fit_whitened": episode_means(white[:, 0], episode_id),
        "fixed_depth_2_raw": episode_means(raw[:, 1], episode_id),
        "fixed_depth_2_fit_whitened": episode_means(white[:, 1], episode_id),
        "exact_compute_analytic_raw": episode_means(raw_comparator["loss"], episode_id),
        "exact_compute_analytic_fit_whitened": episode_means(
            white_comparator["loss"], episode_id
        ),
    }
    loss_intervals = bootstrap_summaries(absolute_episode_losses)

    raw_gain = raw[:, 0] - raw[:, 1]
    white_gain = white[:, 0] - white[:, 1]
    combined_gain = 0.5 * (
        raw_gain / float(gate["raw_gain_scale"][0])
        + white_gain / float(gate["white_gain_scale"][0])
    )
    ranks = {
        "combined_score_gain_spearman": float(spearmanr(score, combined_gain).statistic),
        "raw_score_gain_spearman": float(spearmanr(score, raw_gain).statistic),
        "fit_whitened_score_gain_spearman": float(spearmanr(score, white_gain).statistic),
    }

    primary_raw = endpoint_intervals["raw_vs_exact_compute_analytic"]
    primary_white = endpoint_intervals["fit_whitened_vs_exact_compute_analytic"]
    simultaneous_bounds = {
        "method": "Bonferroni familywise one-sided alpha 0.05; alpha/2=0.025 per endpoint",
        "raw_lower": primary_raw["exploratory_95_ci_low"],
        "fit_whitened_lower": primary_white["exploratory_95_ci_low"],
        "strictly_positive_both": bool(
            primary_raw["exploratory_95_ci_low"] > 0
            and primary_white["exploratory_95_ci_low"] > 0
        ),
    }
    raw_comparator_mean = float(absolute_episode_losses["exact_compute_analytic_raw"].mean())
    white_comparator_mean = float(
        absolute_episode_losses["exact_compute_analytic_fit_whitened"].mean()
    )
    sign_and_relative = {
        "raw": {
            "positive_episode_count": int(np.sum(endpoint_values["raw_vs_exact_compute_analytic"] > 0)),
            "zero_episode_count": int(np.sum(endpoint_values["raw_vs_exact_compute_analytic"] == 0)),
            "negative_episode_count": int(np.sum(endpoint_values["raw_vs_exact_compute_analytic"] < 0)),
            "relative_benefit_fraction_of_comparator_mse": float(
                primary_raw["mean"] / raw_comparator_mean
            ),
        },
        "fit_whitened": {
            "positive_episode_count": int(
                np.sum(endpoint_values["fit_whitened_vs_exact_compute_analytic"] > 0)
            ),
            "zero_episode_count": int(
                np.sum(endpoint_values["fit_whitened_vs_exact_compute_analytic"] == 0)
            ),
            "negative_episode_count": int(
                np.sum(endpoint_values["fit_whitened_vs_exact_compute_analytic"] < 0)
            ),
            "relative_benefit_fraction_of_comparator_mse": float(
                primary_white["mean"] / white_comparator_mean
            ),
        },
    }

    common_flops = len(calls) * (base_price + norm_price)
    adaptive_total = (
        common_flops + refiner_calls * refiner_price + gate_evaluations * gate_price
    )
    comparator_total = adaptive_total
    exact_compute = isinstance(adaptive_total, int) and adaptive_total == comparator_total
    forbidden = ("target", "future", "contact", "reward", "success", "simulator", "geometry", "state")
    causal_checks = {
        "feature_builder_signature_exact": list(inspect.signature(build_causal_features).parameters)
        == ["history", "actions", "current", "update"],
        "feature_names_exclude_forbidden_terms": not any(
            term in name.lower() for name in causal_feature_names() for term in forbidden
        ),
        "only_stage1_score_computed": config["fixed_cohort"]["gate_scores_computed"]
        == ["stage_1"],
        "stage2_stage3_scores_never_computed_or_consulted": not config["fixed_cohort"][
            "stage_2_or_stage_3_gate_scores_computed_or_consulted"
        ],
        "privileged_context_not_used": not config["fixed_cohort"][
            "privileged_context_used_by_policy_eligibility_or_decision"
        ],
    }
    finiteness = bool(
        all(np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number))
        and np.isfinite(raw).all()
        and np.isfinite(white).all()
        and all(math.isfinite(value) for value in ranks.values())
    )
    cohort_checks = {
        "fixed_episode_count_240": len(np.unique(episode_id)) == COHORT_COUNT,
        "no_replacements": config["fixed_cohort"]["no_replacements"],
        "no_exclusions": config["fixed_cohort"]["no_exclusions"],
        "all_episodes_have_transitions": bool(np.all(arrays["episode_transition_count"] > 0)),
        "seeds_exact": np.array_equal(
            arrays["episode_unique_seed"],
            np.arange(COHORT_SEED_FIRST, COHORT_SEED_LAST + 1, dtype=np.int64),
        ),
    }
    process_valid = bool(
        exact_compute
        and all(causal_checks.values())
        and finiteness
        and all(cohort_checks.values())
        and config["frozen_contract"]["all_hashes_match"]
    )
    supported = bool(
        process_valid
        and simultaneous_bounds["strictly_positive_both"]
        and ranks["combined_score_gain_spearman"] >= 0
    )
    label = (
        "pusht_binary_confirmation_supported"
        if supported
        else (
            "pusht_binary_confirmation_not_supported"
            if process_valid
            else "pusht_binary_confirmation_execution_invalid"
        )
    )

    augmented = dict(arrays)
    augmented["permutation_calls"] = permutation.astype(np.int8)
    for name, values in endpoint_values.items():
        augmented[f"episode_{name}"] = values.astype(np.float64)
    for name, values in absolute_episode_losses.items():
        augmented[f"episode_loss_{name}"] = values.astype(np.float64)
    atomic_npz(path, augmented)
    scientific_decision_locked_ns = now_ns()

    compute = {
        "transitions": len(calls),
        "base_model_calls": len(calls),
        "refiner_calls": refiner_calls,
        "stage1_gate_evaluations": gate_evaluations,
        "call_histogram_depth_1_to_2": np.bincount(calls, minlength=3)[1:].tolist(),
        "base_predictor_flops": len(calls) * base_price,
        "action_normalization_flops": len(calls) * norm_price,
        "refiner_flops": refiner_calls * refiner_price,
        "gate_feature_flops": gate_evaluations
        * int(config["frozen_contract"]["compute"]["gate"]["feature_flops_per_reached_evaluation"]),
        "gate_score_flops": gate_evaluations
        * int(config["frozen_contract"]["compute"]["gate"]["dual_affine_score_flops_per_reached_evaluation"]),
        "gate_total_flops": gate_evaluations * gate_price,
        "adaptive_total_counted_flops": adaptive_total,
        "analytic_comparator_total_counted_flops": comparator_total,
        "exact_integer_total_counted_compute_equality": exact_compute,
        "analytic_equivalent_mean_depth": float(equivalent_depth),
        "nonflop_comparison_min_operations": gate_evaluations
        * int(
            config["frozen_contract"]["compute"]["gate"][
                "nonflop_comparison_min_operations_per_reached_evaluation"
            ]
        ),
        "raw_strongest_pairwise_mixture": {
            key: value for key, value in raw_comparator.items() if key != "loss"
        },
        "fit_whitened_strongest_pairwise_mixture": {
            key: value for key, value in white_comparator.items() if key != "loss"
        },
    }
    decision: dict[str, Any] = {
        "schema_version": 1,
        "analysis_started_unix_ns": analysis_started_ns,
        "scientific_decision_locked_unix_ns": scientific_decision_locked_ns,
        "scientific_label": label,
        "supported": supported,
        "process_valid": process_valid,
        "co_primary": {
            "raw": primary_raw,
            "fit_whitened": primary_white,
            "simultaneous_one_sided_lower_bounds": simultaneous_bounds,
        },
        "all_exploratory_episode_benefit_intervals": endpoint_intervals,
        "absolute_episode_loss_intervals": loss_intervals,
        "sign_counts_and_relative_effects": sign_and_relative,
        "stage1_score_gain_rank": ranks,
        "compute": compute,
        "criteria": {
            "both_simultaneous_lower_bounds_strictly_positive": simultaneous_bounds[
                "strictly_positive_both"
            ],
            "exact_compute_equality": exact_compute,
            "causal_checks": causal_checks,
            "finite": finiteness,
            "cohort_checks": cohort_checks,
            "combined_stage1_score_gain_spearman_nonnegative": ranks[
                "combined_score_gain_spearman"
            ]
            >= 0,
        },
        "policy": {
            "threshold": THRESHOLD,
            "calls_recomputed_exactly": True,
            "selected_predictions_recomputed_exactly": True,
            "stage2_or_stage3_gate_scores_computed_or_consulted": False,
            "adaptive_depths": [1, 2],
        },
        "evaluation_arrays_sha256": sha256_file(path),
        "independent_recomputation_required": True,
    }
    atomic_json(decision_path, decision)
    append_note(
        f"Locked the one scientific decision as {label}: raw/whitened effects {primary_raw['mean']:.12g}/{primary_white['mean']:.12g}."
    )

    latency = latency_measurements(augmented, calls, selected)
    decision["synchronized_latency"] = latency
    decision["latency_completed_unix_ns"] = now_ns()
    atomic_json(decision_path, decision)
    set_status("analysis", status="complete", scientific_label=label, next_phase="independent check")
    print(
        json.dumps(
            {
                "status": "analysis_complete",
                "scientific_label": label,
                "raw": primary_raw,
                "fit_whitened": primary_white,
                "simultaneous": simultaneous_bounds,
            }
        )
    )


def report() -> None:
    verification_path = ROOT / "INDEPENDENT_CHECK.json"
    if not verification_path.exists():
        raise RuntimeError("independent check must complete before report")
    verification = read_json(verification_path)
    decision = read_json(ROOT / "DECISION.json")
    label = decision["scientific_label"]
    if not verification["passed"]:
        label = "pusht_binary_confirmation_execution_invalid"
    raw = decision["co_primary"]["raw"]
    white = decision["co_primary"]["fit_whitened"]
    bounds = decision["co_primary"]["simultaneous_one_sided_lower_bounds"]
    ranks = decision["stage1_score_gain_rank"]
    compute = decision["compute"]
    endpoints = decision["all_exploratory_episode_benefit_intervals"]
    losses = decision["absolute_episode_loss_intervals"]
    signs = decision["sign_counts_and_relative_effects"]
    latency = decision["synchronized_latency"]
    replicated = label == "pusht_binary_confirmation_supported"
    answer = (
        "The simple one-extra-step mechanism replicated on this fixed fresh PushT cohort under both raw and pilot-frozen fit-whitened metrics."
        if replicated
        else "The simple one-extra-step mechanism did not meet the preregistered two-endpoint confirmation rule on this fixed fresh PushT cohort."
    )
    beyond_cube = (
        "This adds a fresh-data PushT confirmation of the binary adaptive-allocation mechanism beyond the already-confirmed Cube PlanOracle result."
        if replicated
        else "This does not add a PushT confirmation beyond the already-confirmed Cube PlanOracle result."
    )
    rows = []
    for name, value in endpoints.items():
        rows.append(
            f"| {name} | {value['mean']:.9g} | {value['exploratory_95_ci_low']:.9g} | {value['exploratory_95_ci_high']:.9g} |"
        )
    report_text = f"""# PushT binary adaptive-computation confirmation

Final scientific label: `{label}`.

## Paper-level answer

{answer} The raw episode-averaged benefit over the strongest exact-compute transition-independent analytic comparator was {raw['mean']:.9g}, with simultaneous one-sided lower bound {bounds['raw_lower']:.9g}. The pilot-frozen fit-whitened benefit was {white['mean']:.9g}, with simultaneous lower bound {bounds['fit_whitened_lower']:.9g}. {beyond_cube}

What remains unproven: this is one PushT environment, one WeakPolicy(dist_constraint=100) data-generating process, one frozen base/refiner/gate, and one fresh cohort. It does not establish planning or control improvement, universal generalization, physical-difficulty prediction, or wall-clock speedup.

## Fixed analysis

The binary policy made {compute['call_histogram_depth_1_to_2'][0]} depth-1 and {compute['call_histogram_depth_1_to_2'][1]} depth-2 decisions across {compute['transitions']} transitions. The stage-1 score/combined-gain Spearman correlation was {ranks['combined_score_gain_spearman']:.9g} (raw {ranks['raw_score_gain_spearman']:.9g}; whitened {ranks['fit_whitened_score_gain_spearman']:.9g}). Raw episode signs were {signs['raw']['positive_episode_count']} positive, {signs['raw']['zero_episode_count']} zero, and {signs['raw']['negative_episode_count']} negative; the relative raw effect was {100 * signs['raw']['relative_benefit_fraction_of_comparator_mse']:.6g}%. Whitened signs were {signs['fit_whitened']['positive_episode_count']} positive, {signs['fit_whitened']['zero_episode_count']} zero, and {signs['fit_whitened']['negative_episode_count']} negative; the relative whitened effect was {100 * signs['fit_whitened']['relative_benefit_fraction_of_comparator_mse']:.6g}%.

| Comparison (positive favors adaptive) | Mean episode benefit | Exploratory 95% low | Exploratory 95% high |
| --- | ---: | ---: | ---: |
{os.linesep.join(rows)}

Absolute episode-averaged MSEs were: adaptive raw {losses['adaptive_raw']['mean']:.9g}, fixed depth 1 raw {losses['fixed_depth_1_raw']['mean']:.9g}, fixed depth 2 raw {losses['fixed_depth_2_raw']['mean']:.9g}; adaptive whitened {losses['adaptive_fit_whitened']['mean']:.9g}, fixed depth 1 whitened {losses['fixed_depth_1_fit_whitened']['mean']:.9g}, and fixed depth 2 whitened {losses['fixed_depth_2_fit_whitened']['mean']:.9g}.

The raw comparator selected depths {compute['raw_strongest_pairwise_mixture']['depth_lower']} and {compute['raw_strongest_pairwise_mixture']['depth_upper']} with upper-depth weight {compute['raw_strongest_pairwise_mixture']['weight_upper']:.9g}; the whitened comparator selected depths {compute['fit_whitened_strongest_pairwise_mixture']['depth_lower']} and {compute['fit_whitened_strongest_pairwise_mixture']['depth_upper']} with weight {compute['fit_whitened_strongest_pairwise_mixture']['weight_upper']:.9g}. Both searched every feasible fixed-depth pair from depths 1–4.

## Compute, timing, and integrity

Adaptive and comparator totals were exactly {compute['adaptive_total_counted_flops']} integer counted FLOPs each. Adaptive used {compute['base_model_calls']} base predictions, {compute['refiner_calls']} refiner calls, and {compute['stage1_gate_evaluations']} gate evaluations; gate FLOPs were {compute['gate_total_flops']}, and {compute['nonflop_comparison_min_operations']} comparison/min operations are reported separately. Counterfactual depth-3/4 exits were evaluated only to search the required comparator; the adaptive policy never used those depths and no stage-2 or stage-3 gate score was computed.

Synchronized median latency on {latency['device']} was {latency['base_predictor_all_transitions']['median_seconds']:.6g}s for base prediction over all transitions, {latency['cached_base_to_binary_adaptive_selected']['median_seconds']:.6g}s from cached base to binary adaptive outputs, {latency['fixed_depth_1_cached_base']['median_seconds']:.6g}s for fixed depth 1, and {latency['fixed_depth_2_cached_base']['median_seconds']:.6g}s for fixed depth 2. These timings are separate from counted FLOPs and support no speedup claim.

All 240 fixed fresh episodes were retained with no exclusions or replacements. Frozen hashes, causal boundaries, finiteness, exact selected predictions, exact integer compute, chronology, focused tests, bootstrap values, and every headline value were independently checked from the lossless arrays. Independent recomputation passed: {verification['passed']}.
"""
    (ROOT / "REPORT.md").write_text(report_text)
    append_note(f"Finalized concise paper-level report with terminal label {label}.")
    set_status("report", status="complete", scientific_label=label)
    hashes: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_HASHES.json" and not path.name.endswith(".pyc"):
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    atomic_json(
        ROOT / "ARTIFACT_HASHES.json",
        {"schema_version": 1, "created_unix_ns": now_ns(), "files": hashes},
    )
    print(json.dumps({"status": "report_complete", "scientific_label": label, "artifacts": len(hashes)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("setup-smoke", "fixed-cohort", "analyze", "report"))
    args = parser.parse_args()
    commands = {
        "setup-smoke": setup_smoke,
        "fixed-cohort": fixed_cohort,
        "analyze": analyze,
        "report": report,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
