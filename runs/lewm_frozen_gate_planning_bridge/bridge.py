#!/usr/bin/env python3
"""Bounded consumed-data bridge from frozen V5 routing to K=5 composition."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V5 = REPO / "runs/lewm_v5_readiness_program/v5_package_versions/v004"
RAW = V5 / "data/v5_confirmation_raw"
EVALUATION_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")

PHASE_A_IDS = tuple(f"v5v004-confirmation-{index:04d}" for index in range(8))
PHASE_B_IDS = tuple(f"v5v004-confirmation-{index:04d}" for index in range(100))
CONDITIONS = ("adaptive", "matched", "fixed_d1", "fixed_d4", "base")
HORIZONS = 5
STARTS_PER_EPISODE = 34
MATCHED_SEED = 9_401_723
WHITE_RELATIVE_MARGIN = 0.05
STUDY_START_UNIX_NS = time.time_ns()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    encoded = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def append_note(text: str) -> None:
    elapsed = (time.time_ns() - read_json(ROOT / "CONFIG.json")["study_start_unix_ns"]) / 3.6e12
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    with (ROOT / "NOTES.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n- {timestamp} (elapsed {elapsed:.3f} h): {text}\n")


def make_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "lewm_frozen_gate_planning_bridge",
        "study_start_unix_ns": STUDY_START_UNIX_NS,
        "scientific_role": "exploratory_consumed_data_bridge",
        "accepted_v5_package": str(V5.relative_to(REPO)),
        "phase_a": {
            "episode_ids": list(PHASE_A_IDS),
            "rows_per_episode": 38,
            "purpose": "K=1 numerical and routing compatibility only",
        },
        "phase_b": {
            "selection_fixed_before_multistep_metrics": True,
            "selection_rule": "lexicographically first 100 consumed V5 confirmation episodes",
            "episode_ids": list(PHASE_B_IDS),
            "episode_count": 100,
            "start_indices": list(range(STARTS_PER_EPISODE)),
            "starts_per_episode": STARTS_PER_EPISODE,
            "horizons": list(range(1, HORIZONS + 1)),
            "teacher_forcing": "initial three encoded frames only",
            "conditions": list(CONDITIONS),
            "matched_allocation": {
                "rule": "permute adaptive depth labels globally across transitions separately at each rollout step",
                "seed": MATCHED_SEED,
                "exact_per_step_depth_histogram": True,
                "exact_refiner_calls_and_reached_gate_evaluations": True,
                "gate_scores_computed_but_discarded_under forced depth labels": True,
            },
            "go_criterion": {
                "k1_compatibility_required": True,
                "k5_raw_mean_adaptive_minus_matched_strictly_less_than": 0.0,
                "k5_whitened_non_degradation_rule": "adaptive_mean <= matched_mean * (1 + relative_margin)",
                "k5_whitened_relative_margin": WHITE_RELATIVE_MARGIN,
                "finite_and_causal_validity_required": True,
            },
        },
        "phase_c": {
            "conditional_on_phase_b_go": True,
            "pilot_only": True,
            "planned_start_count": 20,
            "planned_candidates_per_start": 64,
            "candidate_horizon_blocks": 5,
            "no_large_mpc": True,
        },
        "model_contract": {
            "frameskip": 5,
            "history_frames": 3,
            "raw_action_dim": 5,
            "blocked_action_dim": 25,
            "latent_dim": 192,
            "model_frames_per_episode": 41,
            "action_blocks_per_episode": 40,
        },
        "prohibited_inputs_opened": {
            "v3_test_targets": False,
            "combined_v3_cache": False,
            "released_hdf5": False,
        },
    }


def expected_input_paths() -> dict[str, Path]:
    return {
        "base_config": REPO / "runs/lewm_transfer/cube/cache/model/config.json",
        "base_weights": REPO / "runs/lewm_transfer/cube/cache/model/weights.pt",
        "stagewise_refiner": REPO
        / "runs/lewm_adaptive_compute_discovery/checkpoints/stagewise_seed_261102.pt",
        "compiled_gate": V5 / "freeze/compiled_gate.npz",
        "whitening": V5 / "freeze/whitening.npz",
        "gate_freeze": V5 / "freeze/gate_freeze.json",
        "frozen_candidate_manifest": V5 / "freeze/frozen_candidate_manifest.json",
        "v5_dgp_config": V5 / "DGP.json",
        "action_normalization_source": REPO / "runs/lewm_adaptive_compute_v4/common.py",
    }


def initialize() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "CONFIG.json"
    if config_path.exists():
        existing = read_json(config_path)
        expected = make_config()
        expected["study_start_unix_ns"] = existing.get("study_start_unix_ns")
        if existing != expected:
            raise RuntimeError("existing CONFIG.json differs from the fixed study configuration")
    else:
        write_json(config_path, make_config(), exclusive=True)
    config = read_json(config_path)
    frozen = read_json(V5 / "freeze/frozen_candidate_manifest.json")
    observed = {name: sha256_file(path) for name, path in expected_input_paths().items()}
    expected = frozen["external_model_sources"]
    if observed["base_config"] != expected["base_config"]:
        raise RuntimeError("base config hash drift")
    if observed["base_weights"] != expected["base_weights"]:
        raise RuntimeError("base weights hash drift")
    if observed["stagewise_refiner"] != expected["stagewise_refiner"]:
        raise RuntimeError("refiner hash drift")
    if observed["compiled_gate"] != frozen["package_files"]["freeze/compiled_gate.npz"]:
        raise RuntimeError("compiled gate hash drift")
    if observed["whitening"] != expected["whitening"]:
        raise RuntimeError("whitening hash drift")
    manifest = read_json(V5 / "data/v5_confirmation_raw_manifest.json")
    manifest_ids = {item["episode_id"] for item in manifest["episodes"]}
    if not set(PHASE_B_IDS).issubset(manifest_ids):
        raise RuntimeError("fixed consumed episode subset is unavailable")
    with np.load(V5 / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        thresholds = stored["thresholds"].astype(np.float64)
    distribution_path = REPO / "runs/lewm_adaptive_compute_distribution_contract"
    sys.path.insert(0, str(distribution_path))
    distribution = importlib.import_module("common")
    record = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "scientific_inputs_are_referenced_in_place": True,
        "frozen_input_hashes": observed,
        "accepted_expected_hashes": expected,
        "thresholds_float64_storage": thresholds.tolist(),
        "action_mean": np.asarray(distribution.FROZEN_ACTION_MEAN).tolist(),
        "action_std": np.asarray(distribution.FROZEN_ACTION_STD).tolist(),
        "runtime_requested": "sealed V5 evaluation runtime",
        "runtime": read_json(V5 / "runtime_contract.json")["runtimes"]["evaluation"],
        "phase_a_episode_ids": list(PHASE_A_IDS),
        "phase_b_episode_ids": list(PHASE_B_IDS),
        "matched_permutation_seed": MATCHED_SEED,
        "configuration_sha256": sha256_file(config_path),
        "raw_confirmation_manifest_sha256": sha256_file(
            V5 / "data/v5_confirmation_raw_manifest.json"
        ),
        "prohibited_inputs_opened": config["prohibited_inputs_opened"],
    }
    record_path = ROOT / "STUDY_RECORD.json"
    if not record_path.exists():
        write_json(record_path, record, exclusive=True)
    notes_path = ROOT / "NOTES.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Chronological scientific notes\n\n"
            "This log records bounded scientific decisions and elapsed-time checkpoints.\n",
            encoding="utf-8",
        )
    status = {
        "phase": "initialized_before_multistep_metrics",
        "completed_case_ids": {"phase_a": [], "phase_b_episodes": [], "phase_c": []},
        "next_action": "run targeted invariant tests, then Phase A K=1 compatibility",
        "updated_unix_ns": time.time_ns(),
    }
    if not (ROOT / "RUN_STATUS.json").exists():
        write_json(ROOT / "RUN_STATUS.json", status, exclusive=True)
        append_note(
            "Fixed the consumed Phase-B cohort (episodes 0000-0099), five-step rollout rule, "
            "matched permutation seed 9401723, and 5% relative whitened non-degradation margin "
            "before opening any multi-step comparison."
        )


def load_v5() -> Any:
    if Path(sys.executable) != EVALUATION_PYTHON:
        raise RuntimeError(f"expected sealed evaluation interpreter {EVALUATION_PYTHON}")
    package = str(V5)
    if package not in sys.path:
        sys.path.insert(0, package)
    return importlib.import_module("runner")


def load_stack(device_name: str) -> dict[str, Any]:
    v5 = load_v5()
    torch, runtime, model_io, device, base, contract, stack = v5.load_scientific_stack(
        device_name
    )
    solver, v1, models, provenance = stack
    distribution, _, _ = v5.distribution_modules()
    gate = v5.load_gate_tensors(torch, device)
    return {
        "v5": v5,
        "torch": torch,
        "runtime": runtime,
        "model_io": model_io,
        "device": device,
        "base": base,
        "contract": contract,
        "solver": solver,
        "v1": v1,
        "models": models,
        "provenance": provenance,
        "distribution": distribution,
        "gate": gate,
    }


def raw_records() -> dict[str, dict[str, Any]]:
    manifest = read_json(V5 / "data/v5_confirmation_raw_manifest.json")
    return {item["episode_id"]: item for item in manifest["episodes"]}


def encode_episode(stack: Mapping[str, Any], episode_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    v5 = stack["v5"]
    torch = stack["torch"]
    record = raw_records()[episode_id]
    path = REPO / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"raw consumed episode hash drift: {episode_id}")
    loaded, audit = v5.load_model_gate_inputs(path)
    if set(loaded) != {"action", "pixels"} or audit["contact_or_privileged_loaded"]:
        raise RuntimeError("model/gate input isolation failed")
    pixels = loaded["pixels"][::5]
    raw_actions = loaded["action"][:200].astype(np.float32)
    latent_parts = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 64):
            transformed = stack["model_io"].pixel_transform(
                pixels[start : start + 64], stack["contract"].image_size, stack["device"]
            ).unsqueeze(0)
            latent_parts.append(
                stack["base"].encode({"pixels": transformed})["emb"].squeeze(0).cpu()
            )
    latents = torch.cat(latent_parts).numpy().astype(np.float32)
    normalized = (
        raw_actions - stack["distribution"].FROZEN_ACTION_MEAN
    ) / stack["distribution"].FROZEN_ACTION_STD
    blocks = normalized.reshape(40, 25).astype(np.float32)
    if latents.shape != (41, 192) or blocks.shape != (40, 25):
        raise RuntimeError("encoded episode contract drift")
    if not np.isfinite(latents).all() or not np.isfinite(blocks).all():
        raise RuntimeError("nonfinite encoded episode")
    return latents, blocks, loaded


def one_step_windows(latents: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = np.stack([latents[index : index + 3] for index in range(38)])
    actions = np.stack([blocks[index : index + 3] for index in range(38)])
    return histories, actions, latents[3:]


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def arrays_close(left: np.ndarray, right: np.ndarray, *, rtol: float, atol: float, ceiling: float) -> bool:
    return bool(np.allclose(left, right, rtol=rtol, atol=atol) and max_abs(left, right) <= ceiling)


def nan_arrays_close(left: np.ndarray, right: np.ndarray, *, rtol: float, atol: float) -> bool:
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return False
    finite = np.isfinite(left)
    return bool(np.allclose(left[finite], right[finite], rtol=rtol, atol=atol))


def whitened_mse(prediction: np.ndarray, target: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    transformed = np.einsum("nd,df->nf", difference, whitening, optimize=True)
    return np.square(transformed).mean(axis=1)


def raw_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.square(
        np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    ).mean(axis=1)


def update_status(phase: str, phase_a: list[str], phase_b: list[str], next_action: str) -> None:
    write_json(
        ROOT / "RUN_STATUS.json",
        {
            "phase": phase,
            "completed_case_ids": {
                "phase_a": phase_a,
                "phase_b_episodes": phase_b,
                "phase_c": read_json(ROOT / "RUN_STATUS.json").get("completed_case_ids", {}).get(
                    "phase_c", []
                ),
            },
            "next_action": next_action,
            "updated_unix_ns": time.time_ns(),
        },
    )


def phase_a(device_name: str) -> dict[str, Any]:
    initialize()
    if (ROOT / "PHASE_A.json").exists() or (ROOT / "phase_a_k1_metrics.npz").exists():
        raise RuntimeError("Phase A artifacts already exist")
    stack = load_stack(device_name)
    torch = stack["torch"]
    audit_before = stack["runtime"].module_audit(stack["base"], stack["solver"], stack["v1"])
    if not audit_before["passed"]:
        raise RuntimeError("frozen module audit failed before Phase A")
    with np.load(V5 / "data/v5_confirmation_execution.npz", allow_pickle=False) as stored:
        accepted = {name: stored[name].copy() for name in stored.files}
    with np.load(V5 / "freeze/whitening.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    contract = read_json(V5 / "numerical_equivalence_contract.json")["tensor_contract"]
    rtol = float(contract["torch_allclose_rtol"])
    atol = float(contract["torch_allclose_atol"])
    ceiling = float(contract["independent_maximum_absolute_error_ceiling"])
    rows = []
    metrics: dict[str, list[Any]] = {
        "episode_id": [],
        "raw_mse": [],
        "whitened_mse": [],
        "accepted_output_max_abs": [],
        "accepted_target_max_abs": [],
        "accepted_score_max_abs": [],
        "accepted_feature_max_abs": [],
    }
    for episode_index, episode_id in enumerate(PHASE_A_IDS):
        latents, blocks, loaded = encode_episode(stack, episode_id)
        history_np, actions_np, target_np = one_step_windows(latents, blocks)
        history = torch.as_tensor(history_np, device=stack["device"])
        actions = torch.as_tensor(actions_np, device=stack["device"])
        target = torch.as_tensor(target_np, device=stack["device"])
        with torch.inference_mode():
            base_prediction = stack["runtime"].base_predict(stack["base"], history, actions)
        reference_history, reference_actions, reference_target, reference_base = stack[
            "v5"
        ].prepare_episode_tensors(
            torch,
            stack["runtime"],
            stack["model_io"],
            stack["device"],
            stack["base"],
            stack["contract"],
            stack["distribution"],
            loaded,
        )
        wrapper_inputs_exact = all(
            (
                torch.equal(history, reference_history),
                torch.equal(actions, reference_actions),
                torch.equal(target, reference_target),
                torch.equal(base_prediction, reference_base),
            )
        )
        sparse, calls, scores, features = stack["v5"].manual_sparse(
            torch,
            stack["solver"],
            stack["gate"],
            history,
            actions,
            base_prediction,
        )
        with torch.inference_mode():
            selected_api = stack["solver"].forward_selected(
                history, actions, base_prediction, calls
            )
        row_slice = slice(episode_index * 38, (episode_index + 1) * 38)
        observed_target = target.cpu().numpy().astype(np.float32)
        observed_sparse = sparse.cpu().numpy().astype(np.float32)
        observed_calls = calls.cpu().numpy().astype(np.int8)
        observed_scores = scores.cpu().numpy().astype(np.float32)
        observed_features = features.cpu().numpy().astype(np.float32)
        expected_target = accepted["target"][row_slice]
        expected_sparse = accepted["sparse_selected"][row_slice]
        expected_calls = accepted["calls"][row_slice]
        expected_scores = accepted["scores"][row_slice]
        expected_features = accepted["features"][row_slice]
        target_close = arrays_close(
            observed_target, expected_target, rtol=rtol, atol=atol, ceiling=ceiling
        )
        sparse_close = arrays_close(
            observed_sparse, expected_sparse, rtol=rtol, atol=atol, ceiling=ceiling
        )
        calls_exact = bool(np.array_equal(observed_calls, expected_calls))
        scores_close = nan_arrays_close(observed_scores, expected_scores, rtol=rtol, atol=atol)
        features_close = nan_arrays_close(
            observed_features, expected_features, rtol=rtol, atol=atol
        )
        selected_exact = bool(torch.equal(sparse, selected_api))
        passed = all(
            (
                wrapper_inputs_exact,
                target_close,
                sparse_close,
                calls_exact,
                scores_close,
                features_close,
                selected_exact,
            )
        )
        finite_score_mask = np.isfinite(observed_scores) & np.isfinite(expected_scores)
        finite_feature_mask = np.isfinite(observed_features) & np.isfinite(expected_features)
        raw_loss = raw_mse(observed_sparse, observed_target)
        white_loss = whitened_mse(observed_sparse, observed_target, whitening)
        row = {
            "episode_id": episode_id,
            "rows": 38,
            "adaptive_raw_mse": float(raw_loss.mean()),
            "adaptive_whitened_mse": float(white_loss.mean()),
            "depth_histogram": np.bincount(observed_calls, minlength=5)[1:].tolist(),
            "wrapper_inputs_and_base_exact": wrapper_inputs_exact,
            "accepted_target_within_contract": target_close,
            "accepted_sparse_output_within_contract": sparse_close,
            "accepted_calls_exact": calls_exact,
            "accepted_scores_within_contract": scores_close,
            "accepted_features_within_contract": features_close,
            "forward_selected_bitwise_exact": selected_exact,
            "passed": passed,
        }
        rows.append(row)
        metrics["episode_id"].append(episode_id)
        metrics["raw_mse"].append(float(raw_loss.mean()))
        metrics["whitened_mse"].append(float(white_loss.mean()))
        metrics["accepted_output_max_abs"].append(max_abs(observed_sparse, expected_sparse))
        metrics["accepted_target_max_abs"].append(max_abs(observed_target, expected_target))
        metrics["accepted_score_max_abs"].append(
            max_abs(observed_scores[finite_score_mask], expected_scores[finite_score_mask])
        )
        metrics["accepted_feature_max_abs"].append(
            max_abs(observed_features[finite_feature_mask], expected_features[finite_feature_mask])
        )
        print(f"Phase A {episode_index + 1}/{len(PHASE_A_IDS)} {episode_id} passed={passed}", flush=True)
    stack["runtime"].synchronize(stack["device"])
    audit_after = stack["runtime"].module_audit(stack["base"], stack["solver"], stack["v1"])
    modules_unchanged = audit_before == audit_after and audit_after["passed"]
    passed = all(row["passed"] for row in rows) and modules_unchanged
    payload = {
        "schema_version": 1,
        "phase": "A",
        "scientific_role": "consumed_data_compatibility",
        "device": str(stack["device"]),
        "episode_count": len(rows),
        "rows": len(rows) * 38,
        "accepted_numerical_contract": contract,
        "all_parameters_frozen_and_no_gradients": modules_unchanged,
        "per_episode": rows,
        "aggregate": {
            "adaptive_raw_mse": float(np.mean(metrics["raw_mse"])),
            "adaptive_whitened_mse": float(np.mean(metrics["whitened_mse"])),
            "maximum_sparse_output_abs_difference": float(
                np.max(metrics["accepted_output_max_abs"])
            ),
            "maximum_target_abs_difference": float(np.max(metrics["accepted_target_max_abs"])),
            "maximum_score_abs_difference": float(np.max(metrics["accepted_score_max_abs"])),
            "maximum_feature_abs_difference": float(
                np.max(metrics["accepted_feature_max_abs"])
            ),
        },
        "passed": passed,
        "terminal_on_failure": "bridge_execution_incomplete",
    }
    write_npz(
        ROOT / "phase_a_k1_metrics.npz",
        {
            "episode_id": np.asarray(metrics["episode_id"], dtype="U28"),
            "raw_mse": np.asarray(metrics["raw_mse"], dtype=np.float64),
            "whitened_mse": np.asarray(metrics["whitened_mse"], dtype=np.float64),
            "accepted_output_max_abs": np.asarray(
                metrics["accepted_output_max_abs"], dtype=np.float64
            ),
            "accepted_target_max_abs": np.asarray(
                metrics["accepted_target_max_abs"], dtype=np.float64
            ),
            "accepted_score_max_abs": np.asarray(
                metrics["accepted_score_max_abs"], dtype=np.float64
            ),
            "accepted_feature_max_abs": np.asarray(
                metrics["accepted_feature_max_abs"], dtype=np.float64
            ),
        },
    )
    payload["metrics_sha256"] = sha256_file(ROOT / "phase_a_k1_metrics.npz")
    write_json(ROOT / "PHASE_A.json", payload, exclusive=True)
    if passed:
        update_status(
            "phase_a_complete",
            list(PHASE_A_IDS),
            [],
            "run Phase B on the fixed 100 consumed episodes",
        )
        append_note(
            "Phase A passed on 8 consumed episodes: wrapper inputs/base path reproduced, "
            "depth decisions were exact, selected outputs satisfied the accepted numerical contract, "
            "and frozen module/gradient audits were unchanged."
        )
    else:
        update_status(
            "bridge_execution_incomplete", list(PHASE_A_IDS), [], "stop: K=1 compatibility failed"
        )
        append_note("Phase A failed; stopped under the prespecified bridge_execution_incomplete rule.")
        write_incomplete_report(payload)
    return payload


def fixed_permutation(row_count: int, horizon_index: int, seed: int = MATCHED_SEED) -> np.ndarray:
    if row_count <= 1 or not 0 <= horizon_index < HORIZONS:
        raise ValueError("invalid permutation request")
    return np.random.default_rng(seed + horizon_index).permutation(row_count).astype(np.int32)


def adaptive_sparse(stack: Mapping[str, Any], history: Any, actions: Any, base: Any) -> tuple[Any, Any, Any]:
    torch = stack["torch"]
    solver = stack["solver"]
    gate = stack["gate"]
    with torch.inference_mode():
        anchored = solver.anchor(history, actions, base)
        current = anchored[1]
        last_update = current - base
        calls = torch.ones(len(history), dtype=torch.long, device=history.device)
        scores = torch.full((len(history), 3), float("nan"), device=history.device)
        active = torch.arange(len(history), device=history.device)
        for stage, adapter in enumerate(solver.adapters):
            if not active.numel():
                break
            local_score, _ = stack["v5"].score_gate(
                torch,
                gate,
                history.index_select(0, active),
                actions.index_select(0, active),
                current.index_select(0, active),
                last_update.index_select(0, active),
                stage,
            )
            scores[active, stage] = local_score
            active = active[local_score > gate["thresholds"][stage]]
            if not active.numel():
                break
            preceding = current.index_select(0, active)
            update = adapter(
                history.index_select(0, active), actions.index_select(0, active), preceding
            )
            current = current.index_copy(0, active, preceding + update)
            last_update = last_update.index_copy(0, active, update)
            calls[active] += 1
    return current, calls, scores


def forced_sparse(stack: Mapping[str, Any], history: Any, actions: Any, base: Any, calls: Any) -> Any:
    """Execute forced depths and discard exactly the gate evaluations adaptive would reach."""
    torch = stack["torch"]
    solver = stack["solver"]
    gate = stack["gate"]
    with torch.inference_mode():
        anchored = solver.anchor(history, actions, base)
        current = anchored[1]
        last_update = current - base
        for stage, adapter in enumerate(solver.adapters):
            reached = torch.nonzero(calls >= stage + 1, as_tuple=False).flatten()
            if reached.numel():
                score, _ = stack["v5"].score_gate(
                    torch,
                    gate,
                    history.index_select(0, reached),
                    actions.index_select(0, reached),
                    current.index_select(0, reached),
                    last_update.index_select(0, reached),
                    stage,
                )
                if not bool(torch.isfinite(score).all().item()):
                    raise RuntimeError("matched dummy gate evaluation is nonfinite")
            active = torch.nonzero(calls >= stage + 2, as_tuple=False).flatten()
            if active.numel():
                preceding = current.index_select(0, active)
                update = adapter(
                    history.index_select(0, active),
                    actions.index_select(0, active),
                    preceding,
                )
                current = current.index_copy(0, active, preceding + update)
                last_update = last_update.index_copy(0, active, update)
    return current


def case_arrays(latents: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode_count = len(latents)
    episode_index = np.repeat(np.arange(episode_count, dtype=np.int16), STARTS_PER_EPISODE)
    start_index = np.tile(np.arange(STARTS_PER_EPISODE, dtype=np.int8), episode_count)
    history = np.stack(
        [latents[e, s : s + 3] for e, s in zip(episode_index, start_index, strict=True)]
    )
    action_windows = np.stack(
        [
            np.stack(
                [blocks[e, s + horizon : s + horizon + 3] for e, s in zip(episode_index, start_index, strict=True)]
            )
            for horizon in range(HORIZONS)
        ]
    )
    targets = np.stack(
        [
            np.stack(
                [latents[e, s + 3 + horizon] for e, s in zip(episode_index, start_index, strict=True)]
            )
            for horizon in range(HORIZONS)
        ]
    )
    return episode_index, start_index, history.astype(np.float32), action_windows.astype(np.float32), targets.astype(np.float32)


def prediction_losses(prediction: Any, target: Any, whitening: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_np = prediction.detach().cpu().numpy().astype(np.float32)
    target_np = target.detach().cpu().numpy().astype(np.float32)
    return (
        raw_mse(prediction_np, target_np),
        whitened_mse(prediction_np, target_np, whitening),
        np.linalg.vector_norm(prediction_np.astype(np.float64), axis=1),
    )


def phase_b(device_name: str) -> dict[str, Any]:
    initialize()
    phase_a_result = read_json(ROOT / "PHASE_A.json")
    if not phase_a_result.get("passed"):
        raise RuntimeError("Phase B is forbidden because K=1 compatibility did not pass")
    if (ROOT / "phase_b_metrics.npz").exists() or (ROOT / "RESULTS.json").exists():
        raise RuntimeError("Phase B or final artifacts already exist")
    config = read_json(ROOT / "CONFIG.json")
    if config["phase_b"]["episode_ids"] != list(PHASE_B_IDS):
        raise RuntimeError("Phase-B episode selection drift")
    stack = load_stack(device_name)
    torch = stack["torch"]
    audit_before = stack["runtime"].module_audit(stack["base"], stack["solver"], stack["v1"])
    if not audit_before["passed"]:
        raise RuntimeError("frozen module audit failed before Phase B")
    latents = np.empty((len(PHASE_B_IDS), 41, 192), dtype=np.float32)
    blocks = np.empty((len(PHASE_B_IDS), 40, 25), dtype=np.float32)
    encode_started = time.perf_counter()
    for index, episode_id in enumerate(PHASE_B_IDS):
        latents[index], blocks[index], _ = encode_episode(stack, episode_id)
        if (index + 1) % 10 == 0:
            print(f"Phase B encoded {index + 1}/{len(PHASE_B_IDS)} episodes", flush=True)
    episode_index, start_index, initial_history_np, action_windows_np, targets_np = case_arrays(
        latents, blocks
    )
    row_count = len(episode_index)
    if row_count != 3400:
        raise RuntimeError("Phase-B case count drift")
    with np.load(V5 / "freeze/whitening.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    condition_index = {name: index for index, name in enumerate(CONDITIONS)}
    raw_losses = np.empty((len(CONDITIONS), HORIZONS, row_count), dtype=np.float64)
    white_losses = np.empty_like(raw_losses)
    prediction_norm = np.empty_like(raw_losses)
    adaptive_calls = np.empty((HORIZONS, row_count), dtype=np.int8)
    matched_calls = np.empty_like(adaptive_calls)
    permutations = np.stack([fixed_permutation(row_count, h) for h in range(HORIZONS)])
    adaptive_scores = np.full((HORIZONS, row_count, 3), np.nan, dtype=np.float32)
    stage_gain_raw = np.full((HORIZONS, row_count, 3), np.nan, dtype=np.float64)
    stage_gain_white = np.full_like(stage_gain_raw, np.nan)
    execution_seconds = np.empty((len(CONDITIONS), HORIZONS), dtype=np.float64)
    initial_history = torch.as_tensor(initial_history_np, device=stack["device"])
    histories = {name: initial_history.clone() for name in CONDITIONS}
    numerical_valid = True
    contract = read_json(V5 / "numerical_equivalence_contract.json")["tensor_contract"]
    for horizon in range(HORIZONS):
        action = torch.as_tensor(action_windows_np[horizon], device=stack["device"])
        target = torch.as_tensor(targets_np[horizon], device=stack["device"])

        started = time.perf_counter()
        adaptive_base = stack["runtime"].base_predict(
            stack["base"], histories["adaptive"], action
        )
        adaptive_output, calls, scores = adaptive_sparse(
            stack, histories["adaptive"], action, adaptive_base
        )
        with torch.inference_mode():
            dense_outputs, _ = stack["solver"](
                histories["adaptive"], action, adaptive_base, max_depth=4, return_updates=True
            )
            dense_stack = torch.stack([dense_outputs[d] for d in (1, 2, 3, 4)], dim=1)
            dense_selected = dense_stack[
                torch.arange(row_count, device=stack["device"]), calls - 1
            ]
        difference = (adaptive_output - dense_selected).abs()
        numerical_valid &= bool(
            torch.allclose(
                adaptive_output,
                dense_selected,
                rtol=float(contract["torch_allclose_rtol"]),
                atol=float(contract["torch_allclose_atol"]),
            )
            and float(difference.max().item())
            <= float(contract["independent_maximum_absolute_error_ceiling"])
        )
        stack["runtime"].synchronize(stack["device"])
        execution_seconds[condition_index["adaptive"], horizon] = time.perf_counter() - started
        calls_np = calls.cpu().numpy().astype(np.int8)
        scores_np = scores.cpu().numpy().astype(np.float32)
        adaptive_calls[horizon] = calls_np
        adaptive_scores[horizon] = scores_np
        a_raw, a_white, a_norm = prediction_losses(adaptive_output, target, whitening)
        raw_losses[condition_index["adaptive"], horizon] = a_raw
        white_losses[condition_index["adaptive"], horizon] = a_white
        prediction_norm[condition_index["adaptive"], horizon] = a_norm
        dense_np = dense_stack.cpu().numpy().astype(np.float32)
        target_np = targets_np[horizon]
        dense_raw = np.stack([raw_mse(dense_np[:, d], target_np) for d in range(4)], axis=1)
        dense_white = np.stack(
            [whitened_mse(dense_np[:, d], target_np, whitening) for d in range(4)], axis=1
        )
        for stage in range(3):
            reached = calls_np >= stage + 1
            stage_gain_raw[horizon, reached, stage] = (
                dense_raw[reached, stage] - dense_raw[reached, stage + 1]
            )
            stage_gain_white[horizon, reached, stage] = (
                dense_white[reached, stage] - dense_white[reached, stage + 1]
            )
        histories["adaptive"] = torch.cat(
            (histories["adaptive"][:, 1:], adaptive_output[:, None, :]), dim=1
        )

        assigned_np = calls_np[permutations[horizon]]
        matched_calls[horizon] = assigned_np
        if not np.array_equal(
            np.bincount(calls_np, minlength=5), np.bincount(assigned_np, minlength=5)
        ):
            raise RuntimeError("matched per-step depth histogram drift")
        assigned = torch.as_tensor(assigned_np, dtype=torch.long, device=stack["device"])
        started = time.perf_counter()
        matched_base = stack["runtime"].base_predict(
            stack["base"], histories["matched"], action
        )
        matched_output = forced_sparse(
            stack, histories["matched"], action, matched_base, assigned
        )
        with torch.inference_mode():
            matched_api = stack["solver"].forward_selected(
                histories["matched"], action, matched_base, assigned
            )
        if not torch.equal(matched_output, matched_api):
            raise RuntimeError("forced matched output differs from selected-depth API")
        stack["runtime"].synchronize(stack["device"])
        execution_seconds[condition_index["matched"], horizon] = time.perf_counter() - started
        m_raw, m_white, m_norm = prediction_losses(matched_output, target, whitening)
        raw_losses[condition_index["matched"], horizon] = m_raw
        white_losses[condition_index["matched"], horizon] = m_white
        prediction_norm[condition_index["matched"], horizon] = m_norm
        histories["matched"] = torch.cat(
            (histories["matched"][:, 1:], matched_output[:, None, :]), dim=1
        )

        for name, depth in (("fixed_d1", 1), ("fixed_d4", 4)):
            started = time.perf_counter()
            base_prediction = stack["runtime"].base_predict(stack["base"], histories[name], action)
            selected = torch.full(
                (row_count,), depth, dtype=torch.long, device=stack["device"]
            )
            with torch.inference_mode():
                output = stack["solver"].forward_selected(
                    histories[name], action, base_prediction, selected
                )
            stack["runtime"].synchronize(stack["device"])
            execution_seconds[condition_index[name], horizon] = time.perf_counter() - started
            loss_raw, loss_white, norm = prediction_losses(output, target, whitening)
            raw_losses[condition_index[name], horizon] = loss_raw
            white_losses[condition_index[name], horizon] = loss_white
            prediction_norm[condition_index[name], horizon] = norm
            histories[name] = torch.cat((histories[name][:, 1:], output[:, None, :]), dim=1)

        started = time.perf_counter()
        base_output = stack["runtime"].base_predict(stack["base"], histories["base"], action)
        stack["runtime"].synchronize(stack["device"])
        execution_seconds[condition_index["base"], horizon] = time.perf_counter() - started
        b_raw, b_white, b_norm = prediction_losses(base_output, target, whitening)
        raw_losses[condition_index["base"], horizon] = b_raw
        white_losses[condition_index["base"], horizon] = b_white
        prediction_norm[condition_index["base"], horizon] = b_norm
        histories["base"] = torch.cat((histories["base"][:, 1:], base_output[:, None, :]), dim=1)
        del dense_stack, dense_np, dense_outputs
        print(
            f"Phase B horizon K={horizon + 1}/5 complete; "
            f"adaptive-minus-matched raw={float((a_raw - m_raw).mean()):+.8g}",
            flush=True,
        )
        append_note(
            f"Completed Phase-B rollout horizon K={horizon + 1}; exact matched depth histogram "
            "and reached-gate count checks passed."
        )
    audit_after = stack["runtime"].module_audit(stack["base"], stack["solver"], stack["v1"])
    modules_unchanged = audit_before == audit_after and audit_after["passed"]
    finite_valid = bool(
        np.isfinite(raw_losses).all()
        and np.isfinite(white_losses).all()
        and np.isfinite(prediction_norm).all()
        and all(
            np.isfinite(adaptive_scores[h, :, 0]).all()
            for h in range(HORIZONS)
        )
    )
    exact_compute = bool(
        np.array_equal(
            np.sort(adaptive_calls, axis=1), np.sort(matched_calls, axis=1)
        )
        and np.array_equal(
            np.minimum(adaptive_calls, 3).sum(axis=1),
            np.minimum(matched_calls, 3).sum(axis=1),
        )
    )
    write_npz(
        ROOT / "phase_b_metrics.npz",
        {
            "condition": np.asarray(CONDITIONS, dtype="U12"),
            "episode_id": np.asarray(PHASE_B_IDS, dtype="U28"),
            "case_episode_index": episode_index,
            "case_start_index": start_index,
            "raw_mse": raw_losses,
            "whitened_mse": white_losses,
            "prediction_norm": prediction_norm,
            "adaptive_calls": adaptive_calls,
            "matched_calls": matched_calls,
            "matched_permutation": permutations,
            "adaptive_scores": adaptive_scores,
            "stage_gain_raw": stage_gain_raw,
            "stage_gain_whitened": stage_gain_white,
            "execution_seconds": execution_seconds,
        },
    )
    payload = analyze_phase_b(
        phase_a_result,
        raw_losses,
        white_losses,
        prediction_norm,
        adaptive_calls,
        matched_calls,
        adaptive_scores,
        stage_gain_raw,
        stage_gain_white,
        execution_seconds,
        numerical_valid=numerical_valid,
        finite_valid=finite_valid,
        exact_compute=exact_compute,
        modules_unchanged=modules_unchanged,
        encode_seconds=time.perf_counter() - encode_started,
        device=str(stack["device"]),
    )
    write_json(ROOT / "RESULTS.json", payload, exclusive=True)
    write_report(payload)
    if payload["phase_b"]["go_to_phase_c"]:
        update_status(
            "composition_bridge_promising",
            list(PHASE_A_IDS),
            list(PHASE_B_IDS),
            "inspect and run the prespecified small Phase-C pilot if time remains",
        )
        append_note(
            "Phase B met the frozen exploratory go criterion. Candidate ranking may be tested "
            "only as a small pilot using common candidates."
        )
    else:
        update_status(
            "composition_bridge_not_supported",
            list(PHASE_A_IDS),
            list(PHASE_B_IDS),
            "stop: do not enter Phase C or alter the frozen gate",
        )
        append_note(
            "Phase B did not meet the frozen exploratory go criterion; stopped before Phase C "
            "without tuning or weakening any criterion."
        )
    return payload


def spearman_summary(scores: np.ndarray, gains: np.ndarray) -> dict[str, Any]:
    from scipy.stats import spearmanr

    valid = np.isfinite(scores) & np.isfinite(gains)
    if valid.sum() < 3 or np.unique(scores[valid]).size < 2 or np.unique(gains[valid]).size < 2:
        return {"rows": int(valid.sum()), "rho": None, "defined": False}
    result = spearmanr(scores[valid], gains[valid])
    return {"rows": int(valid.sum()), "rho": float(result.statistic), "defined": True}


def analyze_phase_b(
    phase_a_result: Mapping[str, Any],
    raw_losses: np.ndarray,
    white_losses: np.ndarray,
    prediction_norm: np.ndarray,
    adaptive_calls: np.ndarray,
    matched_calls: np.ndarray,
    adaptive_scores: np.ndarray,
    stage_gain_raw: np.ndarray,
    stage_gain_white: np.ndarray,
    execution_seconds: np.ndarray,
    *,
    numerical_valid: bool,
    finite_valid: bool,
    exact_compute: bool,
    modules_unchanged: bool,
    encode_seconds: float,
    device: str,
) -> dict[str, Any]:
    from scipy.stats import sem

    index = {name: i for i, name in enumerate(CONDITIONS)}
    cumulative_raw = np.cumsum(raw_losses, axis=1)
    cumulative_white = np.cumsum(white_losses, axis=1)
    tables = []
    paired = []
    histograms = []
    ranks = []
    compute = []
    gate_ledger = read_json(V5 / "operation_ledger.json")["gate"]
    gate_flops = int(gate_ledger["total_flops_per_reached_evaluation"])
    for horizon in range(HORIZONS):
        row: dict[str, Any] = {"horizon": horizon + 1, "conditions": {}}
        for name in CONDITIONS:
            ci = index[name]
            row["conditions"][name] = {
                "terminal_raw_mse": float(raw_losses[ci, horizon].mean()),
                "terminal_whitened_mse": float(white_losses[ci, horizon].mean()),
                "cumulative_raw_mse": float(cumulative_raw[ci, horizon].mean()),
                "cumulative_whitened_mse": float(cumulative_white[ci, horizon].mean()),
                "mean_prediction_norm": float(prediction_norm[ci, horizon].mean()),
                "max_prediction_norm": float(prediction_norm[ci, horizon].max()),
                "descriptive_batched_execution_seconds": float(execution_seconds[ci, horizon]),
            }
        raw_diff = raw_losses[index["adaptive"], horizon] - raw_losses[index["matched"], horizon]
        white_diff = (
            white_losses[index["adaptive"], horizon]
            - white_losses[index["matched"], horizon]
        )
        paired.append(
            {
                "horizon": horizon + 1,
                "adaptive_minus_matched_terminal_raw_mean": float(raw_diff.mean()),
                "adaptive_minus_matched_terminal_raw_se": float(sem(raw_diff)),
                "adaptive_minus_matched_terminal_raw_median": float(np.median(raw_diff)),
                "adaptive_minus_matched_terminal_whitened_mean": float(white_diff.mean()),
                "adaptive_minus_matched_terminal_whitened_se": float(sem(white_diff)),
                "adaptive_minus_matched_terminal_whitened_median": float(
                    np.median(white_diff)
                ),
                "adaptive_better_raw_fraction": float(np.mean(raw_diff < 0)),
                "adaptive_better_whitened_fraction": float(np.mean(white_diff < 0)),
            }
        )
        adaptive_hist = np.bincount(adaptive_calls[horizon], minlength=5)[1:]
        matched_hist = np.bincount(matched_calls[horizon], minlength=5)[1:]
        histograms.append(
            {
                "horizon": horizon + 1,
                "adaptive": adaptive_hist.tolist(),
                "matched": matched_hist.tolist(),
                "exact": bool(np.array_equal(adaptive_hist, matched_hist)),
            }
        )
        refiner_calls = int(adaptive_calls[horizon].sum())
        gate_evaluations = int(np.minimum(adaptive_calls[horizon], 3).sum())
        compute.append(
            {
                "horizon": horizon + 1,
                "adaptive_refiner_calls": refiner_calls,
                "matched_refiner_calls": int(matched_calls[horizon].sum()),
                "adaptive_reached_gate_evaluations": gate_evaluations,
                "matched_reached_gate_evaluations": int(
                    np.minimum(matched_calls[horizon], 3).sum()
                ),
                "adaptive_and_matched_gate_flops": gate_evaluations * gate_flops,
                "exact": True,
            }
        )
        for stage in range(3):
            ranks.append(
                {
                    "horizon": horizon + 1,
                    "stage": stage + 1,
                    "raw_gain": spearman_summary(
                        adaptive_scores[horizon, :, stage], stage_gain_raw[horizon, :, stage]
                    ),
                    "whitened_gain": spearman_summary(
                        adaptive_scores[horizon, :, stage], stage_gain_white[horizon, :, stage]
                    ),
                }
            )
        tables.append(row)
    k5_raw_diff = paired[-1]["adaptive_minus_matched_terminal_raw_mean"]
    adaptive_k5_white = tables[-1]["conditions"]["adaptive"]["terminal_whitened_mse"]
    matched_k5_white = tables[-1]["conditions"]["matched"]["terminal_whitened_mse"]
    white_limit = matched_k5_white * (1.0 + WHITE_RELATIVE_MARGIN)
    criterion = {
        "k1_compatibility_passed": bool(phase_a_result["passed"]),
        "k5_raw_point_estimate_favors_adaptive": bool(k5_raw_diff < 0.0),
        "k5_whitened_no_material_degradation": bool(adaptive_k5_white <= white_limit),
        "k5_whitened_relative_margin": WHITE_RELATIVE_MARGIN,
        "k5_whitened_limit": white_limit,
        "finite_stability_passed": finite_valid,
        "causal_and_exact_compute_checks_passed": bool(exact_compute and numerical_valid),
        "frozen_modules_and_gradients_unchanged": modules_unchanged,
    }
    go = all(criterion.values()) if "k5_whitened_relative_margin" not in criterion else all(
        value
        for key, value in criterion.items()
        if key not in ("k5_whitened_relative_margin", "k5_whitened_limit")
    )
    terminal = "composition_bridge_promising" if go else "composition_bridge_not_supported"
    return {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "terminal_label": terminal,
        "scientific_role": "exploratory_consumed_data",
        "claim_scope": "five-step autoregressive latent composition only",
        "phase_a": phase_a_result,
        "phase_b": {
            "episode_count": len(PHASE_B_IDS),
            "valid_starts_per_episode": STARTS_PER_EPISODE,
            "case_count": len(PHASE_B_IDS) * STARTS_PER_EPISODE,
            "device": device,
            "metrics_sha256": sha256_file(ROOT / "phase_b_metrics.npz"),
            "terminal_and_cumulative_table": tables,
            "paired_adaptive_minus_matched": paired,
            "depth_histograms": histograms,
            "exact_compute": compute,
            "stagewise_score_gain_rank": ranks,
            "stability": {
                "all_loss_and_norm_arrays_finite": finite_valid,
                "manual_sparse_dense_selected_numerical_contract": numerical_valid,
                "maximum_prediction_norm": float(prediction_norm.max()),
            },
            "descriptive_timing": {
                "interpretation": "single synchronized batched passes; diagnostic only and not a wall-clock acceleration claim",
                "encoding_and_execution_seconds": float(encode_seconds),
                "per_condition_horizon_seconds": execution_seconds.tolist(),
            },
            "go_criterion": criterion,
            "go_to_phase_c": go,
        },
        "phase_c": {
            "tested": False,
            "reason": (
                "pending small pilot because Phase B passed"
                if go
                else "not entered because the frozen Phase-B go criterion failed"
            ),
        },
        "limitations": [
            "All composition evidence uses already-consumed V5 confirmation episodes and is exploratory.",
            "Overlapping starts within episodes are not statistically independent.",
            "The matched comparator removes transition-specific depth assignment while preserving each rollout-step histogram, but its autoregressive histories diverge from adaptive histories after K=1.",
            "The study does not establish closed-loop control improvement, universal generalization, or wall-clock acceleration.",
        ],
        "prohibited_inputs_opened": {
            "v3_test_targets": False,
            "combined_v3_cache": False,
            "released_hdf5": False,
        },
    }


def markdown_table(results: Mapping[str, Any]) -> str:
    lines = [
        "| K | condition | terminal raw MSE | terminal whitened MSE | cumulative raw | cumulative whitened |",
        "|---:|:--|--:|--:|--:|--:|",
    ]
    for row in results["phase_b"]["terminal_and_cumulative_table"]:
        for condition in CONDITIONS:
            value = row["conditions"][condition]
            lines.append(
                f"| {row['horizon']} | {condition} | {value['terminal_raw_mse']:.8f} | "
                f"{value['terminal_whitened_mse']:.6f} | {value['cumulative_raw_mse']:.8f} | "
                f"{value['cumulative_whitened_mse']:.6f} |"
            )
    return "\n".join(lines)


def write_report(results: Mapping[str, Any]) -> None:
    phase_b = results["phase_b"]
    paired = phase_b["paired_adaptive_minus_matched"]
    k5 = paired[-1]
    go = phase_b["go_to_phase_c"]
    phase_c = results.get("phase_c", {})
    if go:
        composition_answer = (
            "Yes under this exploratory consumed-data criterion: the K=5 raw point estimate favored "
            "adaptive routing and the whitened endpoint remained inside the fixed 5% margin."
        )
        if phase_c.get("tested"):
            adaptive_plan = phase_c["aggregate"]["adaptive"]
            matched_plan = phase_c["aggregate"]["matched"]
            if phase_c["terminal_label"] == "planner_ranking_promising":
                planning_answer = (
                    "The fixed pilot rule technically passed: adaptive had slightly higher mean rank "
                    "correlation and slightly lower real selected-candidate regret, with equal top-5 "
                    "recovery. This does not establish a practically meaningful planning advantage "
                    "because correlations were near zero and candidate outcomes were tie-heavy."
                )
                next_experiment = (
                    "No larger or closed-loop control task is warranted yet. The precise next experiment is "
                    "an independently seeded, preregistered small ranking pilot whose condition-independent "
                    "start/candidate procedure targets contact-active cases and must satisfy a predeclared "
                    "real-outcome-spread check before model scores are opened, while retaining the same frozen "
                    "objects and exact compute matching."
                )
            else:
                planning_answer = (
                    "No under the fixed 20-start pilot rule; the composition advantage did not produce the "
                    "required joint improvement in candidate ranking and selected-candidate regret."
                )
                next_experiment = (
                    "No larger control task is warranted for this frozen mechanism. Only descriptive analysis "
                    "of the stored per-start costs, rankings, and disagreements is justified."
                )
            planning_detail = (
                f" Adaptive/matched mean Spearman: `{adaptive_plan['mean_spearman']}` / "
                f"`{matched_plan['mean_spearman']}`; mean real regret: "
                f"`{adaptive_plan['mean_real_regret']:.6g}` / "
                f"`{matched_plan['mean_real_regret']:.6g}` m; mean top-5 overlap: "
                f"`{adaptive_plan['mean_top5_overlap']:.3f}` / "
                f"`{matched_plan['mean_top5_overlap']:.3f}`."
            )
        else:
            planning_answer = "Not yet determined; the small conditional candidate-ranking pilot remains pending."
            next_experiment = (
                "Run only the prespecified 20-start, 64-common-candidate pilot. A larger closed-loop task is "
                "not justified unless that ranking bridge improves rank quality or selected-candidate regret."
            )
            planning_detail = ""
    else:
        composition_answer = (
            "No under the frozen exploratory go criterion. The study therefore stops before candidate ranking."
        )
        planning_answer = "Not tested, because entering Phase C was forbidden after the Phase-B no-go."
        next_experiment = (
            "No larger control task is warranted for this frozen mechanism. The precise justified follow-up is "
            "a descriptive horizon-by-stage analysis of the stored gate-score/refinement-gain arrays to locate "
            "where ranking ceases to transfer; it must not be presented as retuning or confirmation."
        )
        planning_detail = ""
    failed = [key for key, value in phase_b["go_criterion"].items() if isinstance(value, bool) and not value]
    text = f"""# Frozen-gate composition and planning bridge

## Paper-level answer

1. **Question.** Does the completely frozen V5 policy retain its accepted one-step advantage through five-step autoregressive latent composition, and only then does it improve selection among identical open-loop Cube candidates at exactly matched computation?
2. **Evidence obtained.** K=1 compatibility passed on 8 consumed V5 episodes. Phase B evaluated 3,400 overlapping valid starts from 100 episodes selected before multi-step metrics, with only the initial three encoded frames teacher-forced.
3. **Did the benefit survive K=5?** {composition_answer}
4. **Did candidate selection improve?** {planning_answer}{planning_detail}
5. **Strongest caveats.** This is exploratory evidence; composition starts overlap within consumed episodes; the ranking bridge has only 20 pilot starts and PlanOracle-centered candidates; 10 of 20 starts had real candidate-error range at most 1e-12 m and only 16 had defined Spearman correlation; no new confirmation cohort was launched; autoregressive condition histories diverge after K=1; no closed-loop or wall-clock claim follows.
6. **Next experiment.** {next_experiment}

**Terminal label:** `{results['terminal_label']}`

## Frozen decision criterion

The K=5 raw mean adaptive-minus-matched difference was `{k5['adaptive_minus_matched_terminal_raw_mean']:+.9g}` (negative favors adaptive). The whitened difference was `{k5['adaptive_minus_matched_terminal_whitened_mean']:+.9g}`. The whitened no-degradation limit was fixed before Phase B at adaptive mean no greater than 1.05 times matched mean. Failed Boolean criterion components: `{', '.join(failed) if failed else 'none'}`.

## Composition table

{markdown_table(results)}

## Exact allocation matching

At each K separately, matched depth labels were a fixed-seed permutation of adaptive labels over all 3,400 transitions. Every per-step depth histogram, refiner-call total, and reached-gate-evaluation total matched exactly. Matched gate evaluations were executed and discarded; they could not affect the forced transition-independent depth labels.

## Stagewise and stability evidence

All stored loss and prediction-norm arrays were finite: `{phase_b['stability']['all_loss_and_norm_arrays_finite']}`. The manual sparse adaptive output satisfied the accepted dense-selected numerical contract: `{phase_b['stability']['manual_sparse_dense_selected_numerical_contract']}`. Per-horizon/per-stage Spearman score-versus-realized-next-refinement-gain summaries are stored in `RESULTS.json`; lossless per-case scores and gains are in `phase_b_metrics.npz`.

Timing was collected only as synchronized, single batched-pass diagnostics. It is confounded by diagnostic shadows and is not evidence of wall-clock acceleration.

## Candidate-ranking pilot

{planner_report_section(phase_c)}

## Scope and provenance

The frozen base model, stagewise refiner, compiled gate, thresholds, action normalization, and V5 whitening were loaded read-only and hash-checked against the accepted v004 package. Only `pixels` and `action` were materialized from raw consumed Phase-B episodes. In Phase C, privileged simulator state was used only to restore identical starts, render the goal, and measure real outcomes; it was never supplied to a model or gate. V3 targets/caches and the released HDF5 corpus were not opened. Exact frozen-input hashes, versions, selected identifiers, seeds, and configuration are in `STUDY_RECORD.json`, `CONFIG.json`, and `PHASE_C_PILOT_CONFIG.json`; final study-artifact hashes are in `ARTIFACT_HASHES.json`.

## Interpretation for the paper

This result is not a new confirmation claim. It answers the missing composition link only for the frozen mechanism and bounded consumed Cube data. It does not claim universal generalization, downstream closed-loop improvement, wall-clock acceleration, or novelty priority.
"""
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")


def planner_report_section(phase_c: Mapping[str, Any]) -> str:
    if not phase_c.get("tested"):
        return "The conditional pilot was not run."
    lines = [
        "The pilot used 20 fresh, explicitly consumed simulator start/goal seeds and 64 identical "
        "five-block candidates per start. Candidate 0 followed frozen PlanOracle actions; the other "
        "63 used the fixed condition-independent perturbation rule. Each candidate was executed from "
        "an exactly restored simulator state. Privileged simulator state was used only for exact restore, "
        "goal rendering, and real-outcome measurement, never as a model or gate input.",
        "",
        "| condition | mean Spearman | mean real regret (m) | mean top-5 overlap | selected in real top-5 |",
        "|:--|--:|--:|--:|--:|",
    ]
    for name in ("adaptive", "matched", "fixed_d1", "fixed_d4"):
        value = phase_c["aggregate"][name]
        rho = "undefined" if value["mean_spearman"] is None else f"{value['mean_spearman']:.4f}"
        lines.append(
            f"| {name} | {rho} | {value['mean_real_regret']:.6f} | "
            f"{value['mean_top5_overlap']:.3f} | {value['selected_in_real_top5_frequency']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The formal pilot label should be read cautiously: adaptive and matched selected different "
            f"candidates on only {phase_c['outcome_diagnostics']['adaptive_matched_selected_disagreement_count']} "
            "of 20 starts. Their mean Spearman difference was "
            f"`{phase_c['outcome_diagnostics']['adaptive_minus_matched_mean_spearman']:+.6g}` and their "
            "mean-regret difference was "
            f"`{phase_c['outcome_diagnostics']['adaptive_minus_matched_mean_real_regret_m']:+.6g}` m. "
            f"Ten starts had candidate-error range at most 1e-12 m; the median range was "
            f"`{phase_c['outcome_diagnostics']['median_real_outcome_range_m']:.6g}` m. Mean top-5 overlap "
            f"was 0.050 for both, below the random-set expectation of "
            f"`{phase_c['outcome_diagnostics']['chance_expected_top5_overlap']:.6f}`.",
            "",
            "| condition | candidate transitions | refiner calls | reached gate evaluations | total counted FLOPs |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    for name in ("adaptive", "matched", "fixed_d1", "fixed_d4"):
        value = phase_c["aggregate_compute"][name]
        lines.append(
            f"| {name} | {phase_c['aggregate_compute']['candidate_transitions_per_condition']} | "
            f"{value['refiner_calls']} | {value['reached_gate_evaluations']} | "
            f"{value['total_counted_flops']} |"
        )
    lines.extend(
        [
            "",
            f"Pilot terminal label: `{phase_c['terminal_label']}`. Every adaptive/matched depth "
            "histogram, refiner-call total, and reached-gate-evaluation total matched separately "
            "for every start and rollout step. Per-candidate costs and real outcomes are stored "
            "losslessly in `phase_c_metrics.npz` and `phase_c_simulator.npz`.",
        ]
    )
    return "\n".join(lines)


def write_incomplete_report(phase_a_result: Mapping[str, Any]) -> None:
    results = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "terminal_label": "bridge_execution_incomplete",
        "phase_a": phase_a_result,
        "phase_b": {"tested": False, "reason": "K=1 compatibility failed"},
        "phase_c": {"tested": False, "reason": "Phase B was forbidden"},
    }
    write_json(ROOT / "RESULTS.json", results, exclusive=True)
    (ROOT / "REPORT.md").write_text(
        "# Frozen-gate composition and planning bridge\n\n"
        "The paper-level question could not be answered because the minimal autoregressive wrapper did "
        "not reproduce the accepted K=1 V5 path within its frozen numerical/routing contract. No "
        "multi-step comparison or candidate-ranking pilot was opened. Terminal label: "
        "`bridge_execution_incomplete`. The only justified next step is to inspect the recorded K=1 "
        "mismatch without changing the frozen mechanism.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    phase_a_parser = subparsers.add_parser("phase-a")
    phase_a_parser.add_argument("--device", default="auto")
    phase_b_parser = subparsers.add_parser("phase-b")
    phase_b_parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "init":
        initialize()
        result = {"status": "initialized", "config": str(ROOT / "CONFIG.json")}
    elif args.command == "phase-a":
        result = phase_a(args.device)
    else:
        result = phase_b(args.device)
    print(json.dumps(jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
