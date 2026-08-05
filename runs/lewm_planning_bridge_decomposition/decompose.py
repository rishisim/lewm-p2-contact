#!/usr/bin/env python3
"""Exact reproduction and descriptive decomposition of the consumed 20x64 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PRIOR = REPO / "runs/lewm_frozen_gate_planning_bridge"
V5 = REPO / "runs/lewm_v5_readiness_program/v5_package_versions/v004"
MODEL_PATH = ROOT / "model_reproduction.npz"
ACTUAL_PATH = ROOT / "actual_terminal_latents.npz"
METRICS_PATH = ROOT / "decomposition_metrics.npz"
RESULTS_PATH = ROOT / "RESULTS.json"
CONDITIONS = ("adaptive", "matched", "fixed_d1", "fixed_d4")
START_COUNT = 20
CANDIDATE_COUNT = 64
HORIZONS = 5
LATENT_DIM = 192


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


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


def write_json_atomic(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    encoded = json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_text_atomic(path, encoded, exclusive=exclusive)


def write_text_atomic(path: Path, text: str, *, exclusive: bool = False) -> None:
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and path.exists():
            raise RuntimeError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_npz_atomic(
    path: Path, arrays: Mapping[str, np.ndarray], *, exclusive: bool = False
) -> None:
    if exclusive and path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    if exclusive and path.exists():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"refusing to overwrite {path}")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def append_note(text: str) -> None:
    config = read_json(ROOT / "CONFIG.json")
    elapsed = (time.time_ns() - int(config["study_start_unix_ns"])) / 3.6e12
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    with (ROOT / "NOTES.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n- {timestamp} (elapsed {elapsed:.3f} h): {text}\n")


def set_status(field: str, value: bool = True) -> None:
    allowed = {"configured", "simulator_verified", "model_verified", "analyzed", "complete"}
    if field not in allowed:
        raise ValueError(f"invalid status field: {field}")
    status = read_json(ROOT / "RUN_STATUS.json")
    if set(status) != allowed:
        raise RuntimeError("RUN_STATUS.json must remain minimal")
    status[field] = bool(value)
    write_json_atomic(ROOT / "RUN_STATUS.json", status)


def prior_modules() -> tuple[Any, Any, Any]:
    if str(PRIOR) not in sys.path:
        sys.path.insert(0, str(PRIOR))
    import bridge
    import phase_c_evaluate
    import phase_c_generate

    return bridge, phase_c_generate, phase_c_evaluate


def load_prior_simulation() -> dict[str, np.ndarray]:
    with np.load(PRIOR / "phase_c_simulator.npz", allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def load_prior_metrics() -> dict[str, np.ndarray]:
    with np.load(PRIOR / "phase_c_metrics.npz", allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def verify_source_hashes() -> dict[str, str]:
    config = read_json(ROOT / "CONFIG.json")
    observed: dict[str, str] = {}
    for section in ("frozen_source_hashes_sha256", "prior_bridge_hashes_sha256"):
        for relative, expected in config[section].items():
            path = REPO / relative
            actual = sha256_file(path)
            observed[relative] = actual
            if actual != expected:
                raise RuntimeError(f"source hash drift: {relative}: {actual} != {expected}")

    configured_freeze = {
        Path(relative).name
        for relative in config["frozen_source_hashes_sha256"]
        if "/freeze/" in relative
    }
    actual_freeze = {path.name for path in (V5 / "freeze").iterdir() if path.is_file()}
    if configured_freeze != actual_freeze:
        raise RuntimeError(
            f"frozen-object inventory drift: configured={sorted(configured_freeze)}, "
            f"actual={sorted(actual_freeze)}"
        )

    frozen_manifest = read_json(V5 / "freeze/frozen_candidate_manifest.json")
    frozen_hashes = config["frozen_source_hashes_sha256"]
    external_paths = {
        "base_config": "runs/lewm_transfer/cube/cache/model/config.json",
        "base_weights": "runs/lewm_transfer/cube/cache/model/weights.pt",
        "stagewise_refiner": "runs/lewm_adaptive_compute_discovery/checkpoints/stagewise_seed_261102.pt",
        "whitening": "runs/lewm_v5_readiness_program/v5_package_versions/v004/freeze/whitening.npz",
    }
    for name, relative in external_paths.items():
        if frozen_manifest["external_model_sources"][name] != frozen_hashes[relative]:
            raise RuntimeError(f"accepted frozen manifest mismatch for {name}")
    for relative_within_v5, expected in frozen_manifest["package_files"].items():
        relative = f"runs/lewm_v5_readiness_program/v5_package_versions/v004/{relative_within_v5}"
        if frozen_hashes[relative] != expected:
            raise RuntimeError(f"accepted package-file hash mismatch for {relative}")

    prior_hashes = read_json(PRIOR / "ARTIFACT_HASHES.json")["artifacts"]
    for relative, expected in config["prior_bridge_hashes_sha256"].items():
        name = Path(relative).name
        if name in prior_hashes and prior_hashes[name] != expected:
            raise RuntimeError(f"prior artifact manifest mismatch for {name}")

    simulation = load_prior_simulation()
    metrics = load_prior_metrics()
    expected_ids = np.asarray(
        [item["start_id"] for item in config["consumed_pilot"]["seed_tuples"]],
        dtype="U20",
    )
    if not np.array_equal(simulation["start_id"], expected_ids):
        raise RuntimeError("stored simulator start identifiers differ from sealed configuration")
    if not np.array_equal(metrics["start_id"], expected_ids):
        raise RuntimeError("stored model start identifiers differ from sealed configuration")
    for key in ("env_seed", "policy_seed", "oracle_np_seed", "candidate_seed"):
        expected = np.asarray(
            [item[key] for item in config["consumed_pilot"]["seed_tuples"]], dtype=np.int64
        )
        if not np.array_equal(simulation[key], expected):
            raise RuntimeError(f"stored consumed seed array drift: {key}")
    if not np.array_equal(metrics["condition"], np.asarray(CONDITIONS, dtype="U12")):
        raise RuntimeError("prior condition ordering drift")
    if not np.array_equal(metrics["real_terminal_error"], simulation["real_terminal_error"]):
        raise RuntimeError("prior simulator/model physical-outcome arrays disagree")
    if simulation["candidate_actions"].shape != (START_COUNT, CANDIDATE_COUNT, 25, 5):
        raise RuntimeError("stored candidate action shape drift")
    return observed


def whitened_mse(
    prediction: np.ndarray, target: np.ndarray, whitening: np.ndarray
) -> np.ndarray:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    original_shape = difference.shape[:-1]
    transformed = np.einsum(
        "nd,df->nf", difference.reshape(-1, difference.shape[-1]), whitening, optimize=True
    )
    return np.square(transformed).mean(axis=1).reshape(original_shape)


def raw_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.square(
        np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    ).mean(axis=-1)


def validate_model_reproduction() -> dict[str, np.ndarray]:
    config = read_json(ROOT / "CONFIG.json")
    prior = load_prior_metrics()
    with np.load(MODEL_PATH, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    required_shapes = {
        "condition": (4,),
        "start_id": (20,),
        "initial_latent": (20, 3, 192),
        "goal_latent": (20, 192),
        "predicted_terminal_latent": (4, 20, 64, 192),
        "predicted_goal_cost_raw": (4, 20, 64),
        "predicted_goal_cost_whitened": (4, 20, 64),
        "adaptive_calls": (20, 5, 64),
        "matched_calls": (20, 5, 64),
        "matched_permutation": (20, 5, 64),
        "selected_candidate_index_raw": (4, 20),
    }
    for name, shape in required_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise RuntimeError(f"model reproduction artifact shape drift: {name}")
    tolerances = config["reproduction_tolerances"]
    difference = max_abs(arrays["predicted_goal_cost_raw"], prior["predicted_terminal_cost"])
    close = np.allclose(
        arrays["predicted_goal_cost_raw"],
        prior["predicted_terminal_cost"],
        rtol=float(tolerances["model_cost_rtol"]),
        atol=float(tolerances["model_cost_atol"]),
    )
    if not close or difference > float(tolerances["model_cost_independent_max_abs_ceiling"]):
        raise RuntimeError(f"stored model reproduction no longer satisfies contract: {difference}")
    for name in ("adaptive_calls", "matched_calls", "matched_permutation"):
        if not np.array_equal(arrays[name], prior[name]):
            raise RuntimeError(f"stored model reproduction integer-array drift: {name}")
    if not np.array_equal(
        arrays["selected_candidate_index_raw"], prior["selected_candidate_index"]
    ):
        raise RuntimeError("stored model selected indices do not reproduce prior pilot")
    if not np.isfinite(arrays["predicted_terminal_latent"]).all():
        raise RuntimeError("nonfinite reproduced terminal latent")
    return arrays


def reproduce_model(stack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if MODEL_PATH.exists():
        arrays = validate_model_reproduction()
        set_status("model_verified")
        print("Model reproduction artifact already verified", flush=True)
        return arrays

    bridge, phase_c_generate, phase_c_evaluate = prior_modules()
    simulation = load_prior_simulation()
    prior = load_prior_metrics()
    torch = stack["torch"]
    audit_before = stack["runtime"].module_audit(stack["base"], stack["solver"], stack["v1"])
    if not audit_before["passed"]:
        raise RuntimeError("frozen module audit failed before model reproduction")

    initial_latents = phase_c_evaluate.encode_pixels(
        stack, simulation["initial_pixels"].reshape(-1, 224, 224, 3)
    ).reshape(START_COUNT, 3, LATENT_DIM)
    goal_latents = phase_c_evaluate.encode_pixels(stack, simulation["goal_pixels"])
    mean = stack["distribution"].FROZEN_ACTION_MEAN
    std = stack["distribution"].FROZEN_ACTION_STD
    past_blocks = ((simulation["past_actions"] - mean) / std).reshape(
        START_COUNT, 2, 25
    ).astype(np.float32)
    candidate_blocks = ((simulation["candidate_actions"] - mean) / std).reshape(
        START_COUNT, CANDIDATE_COUNT, HORIZONS, 25
    ).astype(np.float32)

    terminal_latents = np.empty(
        (len(CONDITIONS), START_COUNT, CANDIDATE_COUNT, LATENT_DIM), dtype=np.float32
    )
    adaptive_calls = np.empty((START_COUNT, HORIZONS, CANDIDATE_COUNT), dtype=np.int8)
    matched_calls = np.empty_like(adaptive_calls)
    permutations = np.empty_like(adaptive_calls)
    condition_index = {name: index for index, name in enumerate(CONDITIONS)}

    for start_index in range(START_COUNT):
        initial = torch.as_tensor(
            np.repeat(initial_latents[start_index][None], CANDIDATE_COUNT, axis=0),
            device=stack["device"],
        )
        histories = {name: initial.clone() for name in CONDITIONS}
        all_blocks = np.concatenate(
            (
                np.repeat(past_blocks[start_index][None], CANDIDATE_COUNT, axis=0),
                candidate_blocks[start_index],
            ),
            axis=1,
        )
        for horizon in range(HORIZONS):
            action = torch.as_tensor(
                np.ascontiguousarray(all_blocks[:, horizon : horizon + 3]),
                device=stack["device"],
            )
            adaptive_base = stack["runtime"].base_predict(
                stack["base"], histories["adaptive"], action
            )
            adaptive_output, calls, _ = bridge.adaptive_sparse(
                stack, histories["adaptive"], action, adaptive_base
            )
            calls_np = calls.cpu().numpy().astype(np.int8)
            adaptive_calls[start_index, horizon] = calls_np
            histories["adaptive"] = torch.cat(
                (histories["adaptive"][:, 1:], adaptive_output[:, None, :]), dim=1
            )

            permutation = phase_c_evaluate.matched_permutation(start_index, horizon)
            assigned_np = calls_np[permutation]
            permutations[start_index, horizon] = permutation
            matched_calls[start_index, horizon] = assigned_np
            if not np.array_equal(
                np.bincount(calls_np, minlength=5),
                np.bincount(assigned_np, minlength=5),
            ):
                raise RuntimeError("matched depth histogram failed exact equality")
            assigned = torch.as_tensor(
                assigned_np, dtype=torch.long, device=stack["device"]
            )
            matched_base = stack["runtime"].base_predict(
                stack["base"], histories["matched"], action
            )
            matched_output = bridge.forced_sparse(
                stack, histories["matched"], action, matched_base, assigned
            )
            histories["matched"] = torch.cat(
                (histories["matched"][:, 1:], matched_output[:, None, :]), dim=1
            )

            for name, depth in (("fixed_d1", 1), ("fixed_d4", 4)):
                base_prediction = stack["runtime"].base_predict(
                    stack["base"], histories[name], action
                )
                selected = torch.full(
                    (CANDIDATE_COUNT,), depth, dtype=torch.long, device=stack["device"]
                )
                with torch.inference_mode():
                    output = stack["solver"].forward_selected(
                        histories[name], action, base_prediction, selected
                    )
                histories[name] = torch.cat(
                    (histories[name][:, 1:], output[:, None, :]), dim=1
                )

        for name in CONDITIONS:
            terminal_latents[condition_index[name], start_index] = (
                histories[name][:, -1].detach().cpu().numpy().astype(np.float32)
            )
        print(f"Frozen model reproduction {start_index + 1}/{START_COUNT}", flush=True)

    stack["runtime"].synchronize(stack["device"])
    audit_after = stack["runtime"].module_audit(stack["base"], stack["solver"], stack["v1"])
    if audit_before != audit_after or not audit_after["passed"]:
        raise RuntimeError("frozen module/gradient audit changed during reproduction")

    goal_broadcast = goal_latents[None, :, None, :]
    raw_cost = raw_mse(terminal_latents, goal_broadcast)
    with np.load(V5 / "freeze/whitening.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    white_cost = whitened_mse(terminal_latents, goal_broadcast, whitening)
    selected_raw = np.argmin(raw_cost, axis=2).astype(np.int16)
    tolerance = read_json(ROOT / "CONFIG.json")["reproduction_tolerances"]
    raw_cost_difference = max_abs(raw_cost, prior["predicted_terminal_cost"])
    raw_cost_close = np.allclose(
        raw_cost,
        prior["predicted_terminal_cost"],
        rtol=float(tolerance["model_cost_rtol"]),
        atol=float(tolerance["model_cost_atol"]),
    )
    if (
        not raw_cost_close
        or raw_cost_difference > float(tolerance["model_cost_independent_max_abs_ceiling"])
    ):
        raise RuntimeError(
            f"original predicted goal costs failed numerical reproduction: {raw_cost_difference}"
        )
    for name, observed in (
        ("adaptive_calls", adaptive_calls),
        ("matched_calls", matched_calls),
        ("matched_permutation", permutations),
    ):
        if not np.array_equal(observed, prior[name]):
            raise RuntimeError(f"original lossless integer array failed reproduction: {name}")
    if not np.array_equal(selected_raw, prior["selected_candidate_index"]):
        raise RuntimeError("original selected candidate indices failed exact reproduction")
    if not np.isfinite(terminal_latents).all() or not np.isfinite(white_cost).all():
        raise RuntimeError("nonfinite model reproduction output")

    arrays = {
        "condition": np.asarray(CONDITIONS, dtype="U12"),
        "start_id": simulation["start_id"],
        "initial_latent": initial_latents,
        "goal_latent": goal_latents,
        "predicted_terminal_latent": terminal_latents,
        "predicted_goal_cost_raw": raw_cost,
        "predicted_goal_cost_whitened": white_cost,
        "adaptive_calls": adaptive_calls,
        "matched_calls": matched_calls,
        "matched_permutation": permutations,
        "selected_candidate_index_raw": selected_raw,
        "prior_goal_cost_max_abs": np.asarray(raw_cost_difference, dtype=np.float64),
        "frozen_modules_unchanged": np.asarray(True, dtype=np.bool_),
    }
    write_npz_atomic(MODEL_PATH, arrays, exclusive=True)
    validate_model_reproduction()
    set_status("model_verified")
    append_note(
        "Recomputed all four frozen five-step conditions before generating actual-terminal "
        f"latents. Calls, matched permutations, and selected indices were exact; prior raw "
        f"goal costs reproduced with maximum absolute difference {raw_cost_difference:.3g}, "
        "inside the predeclared V5 numerical contract."
    )
    return arrays


def validate_progress(path: Path, start_index: int, expected_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    if str(arrays["start_id"]) != expected_id or int(arrays["start_index"]) != start_index:
        raise RuntimeError(f"simulator progress identity drift: {path}")
    if arrays["actual_terminal_latent"].shape != (CANDIDATE_COUNT, LATENT_DIM):
        raise RuntimeError(f"simulator progress latent shape drift: {path}")
    if not np.isfinite(arrays["actual_terminal_latent"]).all():
        raise RuntimeError(f"nonfinite simulator progress latent: {path}")
    if not bool(arrays["candidate_actions_bitwise_exact"]):
        raise RuntimeError(f"candidate reconstruction was not exact: {path}")
    if not bool(arrays["restore_repeat_pixel_exact"]):
        raise RuntimeError(f"restore repeat was not exact: {path}")
    return arrays


def validate_actual_latents() -> dict[str, np.ndarray]:
    with np.load(ACTUAL_PATH, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    required_shapes = {
        "start_id": (20,),
        "actual_terminal_latent": (20, 64, 192),
        "terminal_pixel_sha256": (20, 64),
        "real_terminal_position": (20, 64, 3),
        "real_terminal_error": (20, 64),
        "terminal_position_max_abs": (20,),
        "terminal_error_max_abs": (20,),
    }
    for name, shape in required_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise RuntimeError(f"actual-terminal artifact shape drift: {name}")
    simulation = load_prior_simulation()
    if not np.array_equal(arrays["start_id"], simulation["start_id"]):
        raise RuntimeError("actual-terminal start identifier drift")
    tolerance = float(
        read_json(ROOT / "CONFIG.json")["reproduction_tolerances"][
            "simulator_position_and_error_atol_m"
        ]
    )
    if max_abs(arrays["real_terminal_position"], simulation["real_terminal_position"]) > tolerance:
        raise RuntimeError("actual-terminal positions no longer reproduce prior simulator")
    if max_abs(arrays["real_terminal_error"], simulation["real_terminal_error"]) > tolerance:
        raise RuntimeError("actual-terminal errors no longer reproduce prior simulator")
    if not np.isfinite(arrays["actual_terminal_latent"]).all():
        raise RuntimeError("nonfinite actual terminal latent")
    return arrays


def replay_and_encode(stack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if ACTUAL_PATH.exists():
        arrays = validate_actual_latents()
        set_status("simulator_verified")
        print("Actual-terminal latent artifact already verified", flush=True)
        return arrays

    _, _, phase_c_evaluate = prior_modules()
    config = read_json(ROOT / "CONFIG.json")
    expected_ids = [item["start_id"] for item in config["consumed_pilot"]["seed_tuples"]]
    progress_dir = ROOT / "_progress"
    progress_dir.mkdir(exist_ok=True)
    generation_python = config["runtime"]["generation_python"]
    simulator_script = ROOT / "simulator_replay.py"

    for start_index, start_id in enumerate(expected_ids):
        progress_path = progress_dir / f"actual-terminal-{start_index:03d}.npz"
        if progress_path.exists():
            validate_progress(progress_path, start_index, start_id)
            print(f"Simulator/encoding progress {start_index + 1}/{START_COUNT} resumed", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix=".terminal-pixels-", dir=ROOT) as temporary_dir:
            pixel_path = Path(temporary_dir) / "terminal_pixels.npz"
            completed = subprocess.run(
                [generation_python, str(simulator_script), str(start_index), str(pixel_path)],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"simulator replay failed for {start_id}:\n{completed.stdout}"
                )
            with np.load(pixel_path, allow_pickle=False) as stored:
                batch = {name: stored[name].copy() for name in stored.files}
            if str(batch["start_id"]) != start_id:
                raise RuntimeError("bounded pixel batch identity drift")
            pixels = batch.pop("terminal_pixels")
            if pixels.shape != (CANDIDATE_COUNT, 224, 224, 3) or pixels.dtype != np.uint8:
                raise RuntimeError("bounded terminal pixel batch contract drift")
            if sha256_array(pixels) != str(batch["terminal_pixels_sha256"]):
                raise RuntimeError("bounded terminal pixel batch hash mismatch")
            actual_latent = phase_c_evaluate.encode_pixels(stack, pixels)
            candidate_pixel_hashes = np.asarray(
                [sha256_array(pixel) for pixel in pixels], dtype="U64"
            )
            progress = {
                "start_id": np.asarray(start_id, dtype="U20"),
                "start_index": np.asarray(start_index, dtype=np.int16),
                "actual_terminal_latent": actual_latent.astype(np.float32),
                "terminal_pixel_sha256": candidate_pixel_hashes,
                "terminal_pixel_batch_sha256": batch["terminal_pixels_sha256"],
                "real_terminal_position": batch["real_terminal_position"],
                "real_terminal_error": batch["real_terminal_error"],
                "initial_state_sha256": batch["initial_state_sha256"],
                "start_position_max_abs": batch["start_position_max_abs"],
                "target_position_max_abs": batch["target_position_max_abs"],
                "terminal_position_max_abs": batch["terminal_position_max_abs"],
                "terminal_error_max_abs": batch["terminal_error_max_abs"],
                "restore_state_max_abs": batch["restore_state_max_abs"],
                "restore_repeat_position_max_abs": batch[
                    "restore_repeat_position_max_abs"
                ],
                "restore_repeat_pixel_exact": batch["restore_repeat_pixel_exact"],
                "candidate_actions_bitwise_exact": batch[
                    "candidate_actions_bitwise_exact"
                ],
            }
            write_npz_atomic(progress_path, progress, exclusive=True)
            del pixels
        validate_progress(progress_path, start_index, start_id)
        print(
            f"Simulator replay and bounded terminal encoding {start_index + 1}/{START_COUNT}",
            flush=True,
        )

    progress = [
        validate_progress(
            progress_dir / f"actual-terminal-{index:03d}.npz", index, expected_ids[index]
        )
        for index in range(START_COUNT)
    ]
    arrays = {
        "start_id": np.asarray(expected_ids, dtype="U20"),
        "actual_terminal_latent": np.stack(
            [item["actual_terminal_latent"] for item in progress]
        ),
        "terminal_pixel_sha256": np.stack(
            [item["terminal_pixel_sha256"] for item in progress]
        ),
        "terminal_pixel_batch_sha256": np.asarray(
            [str(item["terminal_pixel_batch_sha256"]) for item in progress], dtype="U64"
        ),
        "real_terminal_position": np.stack(
            [item["real_terminal_position"] for item in progress]
        ),
        "real_terminal_error": np.stack([item["real_terminal_error"] for item in progress]),
        "initial_state_sha256": np.asarray(
            [str(item["initial_state_sha256"]) for item in progress], dtype="U64"
        ),
        "start_position_max_abs": np.asarray(
            [float(item["start_position_max_abs"]) for item in progress], dtype=np.float64
        ),
        "target_position_max_abs": np.asarray(
            [float(item["target_position_max_abs"]) for item in progress], dtype=np.float64
        ),
        "terminal_position_max_abs": np.asarray(
            [float(item["terminal_position_max_abs"]) for item in progress], dtype=np.float64
        ),
        "terminal_error_max_abs": np.asarray(
            [float(item["terminal_error_max_abs"]) for item in progress], dtype=np.float64
        ),
        "restore_state_max_abs": np.asarray(
            [float(item["restore_state_max_abs"]) for item in progress], dtype=np.float64
        ),
        "restore_repeat_position_max_abs": np.asarray(
            [float(item["restore_repeat_position_max_abs"]) for item in progress],
            dtype=np.float64,
        ),
    }
    write_npz_atomic(ACTUAL_PATH, arrays, exclusive=True)
    validate_actual_latents()
    for path in progress_dir.glob("actual-terminal-*.npz"):
        path.unlink()
    progress_dir.rmdir()
    set_status("simulator_verified")
    append_note(
        "Deterministically reconstructed all 20 consumed starts and all stored candidate actions "
        "bitwise exactly. Replayed all 1,280 outcomes within the fixed 1e-12 m tolerance, "
        "verified exact candidate-0 state restoration, encoded terminal renders one 64-image "
        "batch at a time, and retained no terminal pixels."
    )
    return arrays


def tolerance_groups(values: np.ndarray, tolerance: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("tolerance grouping requires a finite nonempty vector")
    order = np.argsort(values, kind="stable")
    groups = np.empty(len(values), dtype=np.int16)
    group = 0
    group_minimum = float(values[order[0]])
    groups[order[0]] = group
    for index in order[1:]:
        value = float(values[index])
        if value - group_minimum > tolerance:
            group += 1
            group_minimum = value
        groups[index] = group
    return groups


def stable_top_k(values: np.ndarray, top_k: int = 5) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or not 0 < top_k <= len(values):
        raise ValueError("invalid stable top-k request")
    return np.argsort(values, kind="stable")[:top_k]


def stable_group_top_k(groups: np.ndarray, top_k: int = 5) -> np.ndarray:
    groups = np.asarray(groups)
    candidates = np.arange(len(groups))
    return np.lexsort((candidates, groups))[:top_k]


def overlap_fraction(left: np.ndarray, right: np.ndarray) -> float:
    return float(len(set(left.tolist()) & set(right.tolist())) / len(left))


def physical_rank_metrics(
    cost: np.ndarray,
    physical_error: np.ndarray,
    groups: np.ndarray,
    top_k: int = 5,
) -> dict[str, Any]:
    cost = np.asarray(cost, dtype=np.float64)
    physical_error = np.asarray(physical_error, dtype=np.float64)
    groups = np.asarray(groups)
    if np.unique(cost).size < 2 or np.unique(groups).size < 2:
        rho = None
    else:
        value = spearmanr(cost, groups).statistic
        rho = float(value) if np.isfinite(value) else None
    predicted_top = stable_top_k(cost, top_k)
    physical_top = stable_group_top_k(groups, top_k)
    selected = int(predicted_top[0])
    exact_best = float(np.min(physical_error))
    exact_regret = float(physical_error[selected] - exact_best)
    meaningful_regret = 0.0 if int(groups[selected]) == 0 else exact_regret
    return {
        "spearman": rho,
        "top5_overlap": overlap_fraction(predicted_top, physical_top),
        "selected_index": selected,
        "selected_physical_error_m": float(physical_error[selected]),
        "selected_physical_regret_m": exact_regret,
        "selected_meaningful_physical_regret_m": meaningful_regret,
        "selected_in_best_physical_group": bool(int(groups[selected]) == 0),
    }


def target_rank_metrics(
    predicted_cost: np.ndarray, target_cost: np.ndarray, top_k: int = 5
) -> dict[str, Any]:
    predicted_cost = np.asarray(predicted_cost, dtype=np.float64)
    target_cost = np.asarray(target_cost, dtype=np.float64)
    if np.unique(predicted_cost).size < 2 or np.unique(target_cost).size < 2:
        rho = None
    else:
        value = spearmanr(predicted_cost, target_cost).statistic
        rho = float(value) if np.isfinite(value) else None
    predicted_top = stable_top_k(predicted_cost, top_k)
    target_top = stable_top_k(target_cost, top_k)
    selected = int(predicted_top[0])
    return {
        "spearman": rho,
        "top5_overlap": overlap_fraction(predicted_top, target_top),
        "selected_index": selected,
        "selected_target_cost": float(target_cost[selected]),
        "selected_target_cost_regret": float(target_cost[selected] - np.min(target_cost)),
    }


def mean_or_none(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def summarize_physical_metrics(
    per_start: Sequence[Mapping[str, Any]], informative: np.ndarray
) -> dict[str, Any]:
    defined = [item["spearman"] for item in per_start if item["spearman"] is not None]
    informative_items = [item for item, keep in zip(per_start, informative) if keep]
    informative_defined = [
        item["spearman"] for item in informative_items if item["spearman"] is not None
    ]
    return {
        "defined_spearman_starts_all": len(defined),
        "mean_spearman_all_defined": mean_or_none(defined),
        "mean_top5_overlap_all": float(np.mean([item["top5_overlap"] for item in per_start])),
        "mean_selected_physical_regret_m_all": float(
            np.mean([item["selected_physical_regret_m"] for item in per_start])
        ),
        "informative_start_count": len(informative_items),
        "defined_spearman_informative_starts": len(informative_defined),
        "mean_spearman_informative_defined": mean_or_none(informative_defined),
        "mean_top5_overlap_informative": (
            float(np.mean([item["top5_overlap"] for item in informative_items]))
            if informative_items
            else None
        ),
        "mean_selected_physical_regret_m_informative": (
            float(
                np.mean(
                    [item["selected_physical_regret_m"] for item in informative_items]
                )
            )
            if informative_items
            else None
        ),
    }


def summarize_target_metrics(
    per_start: Sequence[Mapping[str, Any]], informative: np.ndarray
) -> dict[str, Any]:
    defined = [item["spearman"] for item in per_start if item["spearman"] is not None]
    informative_items = [item for item, keep in zip(per_start, informative) if keep]
    informative_defined = [
        item["spearman"] for item in informative_items if item["spearman"] is not None
    ]
    return {
        "defined_spearman_starts_all": len(defined),
        "mean_spearman_all_defined": mean_or_none(defined),
        "mean_top5_overlap_all": float(np.mean([item["top5_overlap"] for item in per_start])),
        "mean_selected_target_cost_regret_all": float(
            np.mean([item["selected_target_cost_regret"] for item in per_start])
        ),
        "defined_spearman_informative_starts": len(informative_defined),
        "mean_spearman_informative_defined": mean_or_none(informative_defined),
        "mean_top5_overlap_informative": (
            float(np.mean([item["top5_overlap"] for item in informative_items]))
            if informative_items
            else None
        ),
    }


def bootstrap_effect(
    values: np.ndarray, resamples: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    sampled = values[resamples]
    mean_samples = sampled.mean(axis=1)
    median_samples = np.median(sampled, axis=1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mean_bootstrap_95_interval": np.quantile(mean_samples, [0.025, 0.975]).tolist(),
        "median_bootstrap_95_interval": np.quantile(
            median_samples, [0.025, 0.975]
        ).tolist(),
        "starts_favoring_adaptive": int(np.sum(values < 0.0)),
        "starts_favoring_matched": int(np.sum(values > 0.0)),
        "tied_starts": int(np.sum(values == 0.0)),
        "independent_cluster_count": int(len(values)),
    }


def bootstrap_mean_interval(values: Sequence[float], seed: int, replicates: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[resamples].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def compute_accounting(
    adaptive_calls: np.ndarray, matched_calls: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ledger = read_json(V5 / "operation_ledger.json")
    base_flops = int(ledger["model"]["base_flops_per_row"])
    depth1_flops = int(ledger["model"]["mandatory_depth1_refiner_flops_per_row"])
    adapter_flops = int(ledger["model"]["each_additional_adapter_flops_per_row"])
    gate_flops = int(ledger["gate"]["total_flops_per_reached_evaluation"])

    calls = np.empty((4, START_COUNT, HORIZONS), dtype=np.int64)
    reached = np.empty_like(calls)
    histograms = np.zeros((4, START_COUNT, HORIZONS, 4), dtype=np.int64)
    calls[0] = adaptive_calls.sum(axis=2)
    calls[1] = matched_calls.sum(axis=2)
    calls[2] = CANDIDATE_COUNT
    calls[3] = 4 * CANDIDATE_COUNT
    reached[0] = np.minimum(adaptive_calls, 3).sum(axis=2)
    reached[1] = np.minimum(matched_calls, 3).sum(axis=2)
    reached[2:] = 0
    for start in range(START_COUNT):
        for horizon in range(HORIZONS):
            histograms[0, start, horizon] = np.bincount(
                adaptive_calls[start, horizon], minlength=5
            )[1:]
            histograms[1, start, horizon] = np.bincount(
                matched_calls[start, horizon], minlength=5
            )[1:]
            histograms[2, start, horizon] = [CANDIDATE_COUNT, 0, 0, 0]
            histograms[3, start, horizon] = [0, 0, 0, CANDIDATE_COUNT]
            if not np.array_equal(
                histograms[0, start, horizon], histograms[1, start, horizon]
            ):
                raise RuntimeError("adaptive/matched per-start-step depth histogram mismatch")
            if calls[0, start, horizon] != calls[1, start, horizon]:
                raise RuntimeError("adaptive/matched per-start-step refiner calls mismatch")
            if reached[0, start, horizon] != reached[1, start, horizon]:
                raise RuntimeError("adaptive/matched per-start-step reached-gate mismatch")

    transitions = np.full_like(calls, CANDIDATE_COUNT)
    counted_flops = (
        transitions * (base_flops + depth1_flops)
        + (calls - transitions) * adapter_flops
        + reached * gate_flops
    )
    aggregate: dict[str, Any] = {}
    for ci, name in enumerate(CONDITIONS):
        total_transitions = START_COUNT * HORIZONS * CANDIDATE_COUNT
        total_calls = int(calls[ci].sum())
        total_reached = int(reached[ci].sum())
        components = {
            "candidate_transitions": total_transitions,
            "depth_histogram": histograms[ci].sum(axis=(0, 1)).tolist(),
            "refiner_calls": total_calls,
            "reached_gate_evaluations": total_reached,
            "base_flops": total_transitions * base_flops,
            "mandatory_depth1_flops": total_transitions * depth1_flops,
            "additional_refiner_flops": (total_calls - total_transitions) * adapter_flops,
            "gate_flops": total_reached * gate_flops,
            "total_counted_flops": int(counted_flops[ci].sum()),
        }
        aggregate[name] = components

    per_start_step = []
    for start in range(START_COUNT):
        for horizon in range(HORIZONS):
            row = {"start_index": start, "horizon": horizon + 1, "conditions": {}}
            for ci, name in enumerate(CONDITIONS):
                row["conditions"][name] = {
                    "depth_histogram": histograms[ci, start, horizon].tolist(),
                    "refiner_calls": int(calls[ci, start, horizon]),
                    "reached_gate_evaluations": int(reached[ci, start, horizon]),
                    "counted_flops": int(counted_flops[ci, start, horizon]),
                }
            row["adaptive_matched_exact"] = True
            per_start_step.append(row)
    return {
        "aggregate": aggregate,
        "per_start_step": per_start_step,
        "adaptive_matched_exact_at_all_100_start_steps": True,
    }, {
        "depth_histogram": histograms,
        "refiner_calls": calls,
        "reached_gate_evaluations": reached,
        "counted_flops": counted_flops,
    }


def render_first_table(results: Mapping[str, Any]) -> str:
    raw = results["candidate_rollout_fidelity"]["adaptive_minus_matched"]["raw"]
    white = results["candidate_rollout_fidelity"]["adaptive_minus_matched"]["whitened"]
    alignment = results["goal_cost_alignment"]["actual_terminal_ceiling"]["raw"]["aggregate"]
    info = results["candidate_set_informativeness"]
    decisions = results["decision_table"]
    return f"""# First decomposition table

Produced {time.strftime('%Y-%m-%d %H:%M:%S %Z')} from the fixed consumed 20x64 pilot.

| link | fixed metric | result | descriptive decision |
|:--|:--|:--|:--|
| candidate rollout transfer | adaptive-minus-matched raw terminal MSE | {raw['mean']:+.8g} (95% cluster interval {raw['mean_bootstrap_95_interval'][0]:+.8g}, {raw['mean_bootstrap_95_interval'][1]:+.8g}) | {decisions['candidate_rollout_transfer']['status']} |
| candidate rollout transfer | adaptive-minus-matched whitened terminal MSE | {white['mean']:+.8g} (95% cluster interval {white['mean_bootstrap_95_interval'][0]:+.8g}, {white['mean_bootstrap_95_interval'][1]:+.8g}) | {decisions['candidate_rollout_transfer']['status']} |
| goal-cost alignment ceiling | actual-latent raw cost vs physical error where rank is defined | {alignment['defined_spearman_starts_all']} starts; mean Spearman {alignment['mean_spearman_all_defined']}; all-start mean top-5 overlap {alignment['mean_top5_overlap_all']} vs chance {info['chance_top5_overlap_fraction']:.6f} | {decisions['goal_cost_alignment']['status']} |
| candidate informativeness | informative starts under fixed rule | {info['informative_start_count']}/20 | {decisions['candidate_informativeness']['status']} |
| downstream planning evidence | all required links plus selection change | {decisions['downstream_planning_evidence']['explanation']} | {decisions['downstream_planning_evidence']['status']} |
"""


def render_report(results: Mapping[str, Any]) -> str:
    rollout = results["candidate_rollout_fidelity"]
    raw = rollout["adaptive_minus_matched"]["raw"]
    white = rollout["adaptive_minus_matched"]["whitened"]
    goal = results["goal_cost_alignment"]["actual_terminal_ceiling"]
    raw_goal = goal["raw"]["aggregate"]
    white_goal = goal["whitened"]["aggregate"]
    info = results["candidate_set_informativeness"]
    decisions = results["decision_table"]
    selection = results["selection_utility"]["raw_goal_cost"]
    next_experiment = results["next_experiment"]
    lines = [
        "# Planning bridge decomposition",
        "",
        "## Paper-level answer",
        "",
        f"1. **Does the K=5 adaptive advantage transfer to these off-policy candidates at exactly matched compute?** DIRECTIONALLY YES, BUT WEAKLY: adaptive-minus-matched terminal prediction MSE was {raw['mean']:+.8g} raw and {white['mean']:+.8g} whitened, satisfying the sealed point-estimate rule. Both paired 95% start-cluster intervals crossed zero: [{raw['mean_bootstrap_95_interval'][0]:+.8g}, {raw['mean_bootstrap_95_interval'][1]:+.8g}] and [{white['mean_bootstrap_95_interval'][0]:+.8g}, {white['mean_bootstrap_95_interval'][1]:+.8g}].",
        f"2. **Can the frozen goal-latent cost rank real block outcomes with the actual future observation?** NOT SUPPORTED IN THIS POOL: no start met the predeclared informativeness rule. Among the {raw_goal['defined_spearman_starts_all']} subthreshold starts where a physical rank was at least defined, the actual-terminal raw goal cost had mean Spearman {raw_goal['mean_spearman_all_defined']:.6f} and all-start stable top-5 overlap {raw_goal['mean_top5_overlap_all']:.3f} versus chance {info['chance_top5_overlap_fraction']:.6f}; the whitened analogues were {white_goal['mean_spearman_all_defined']:.6f} and {white_goal['mean_top5_overlap_all']:.3f}.",
        f"3. **Is any planning/control claim now justified?** {decisions['downstream_planning_evidence']['status'].upper()}. {decisions['downstream_planning_evidence']['explanation']}",
        "4. **Strongest caveats.** This is a descriptive decomposition of a consumed 20-start pilot; starts are the independent clusters; the candidate pool is PlanOracle-centered and tie-heavy; physical error is block-position error only; stable tie breaking cannot make uninformative starts evidential; intervals are descriptive; and no closed-loop control was run.",
        (
            f"5. **Next experiment.** {next_experiment}"
            if next_experiment is not None
            else "5. **Next experiment.** None is proposed: the fixed decision rule permits a follow-on only if downstream planning evidence is supported."
        ),
        "",
        "## Exact reproduction",
        "",
        f"All configured frozen-object and prior-artifact SHA-256 hashes verified. The 20 start identifiers and seed tuples were exact; warm-start pixels, goal pixels, past actions, reconstructed 64-candidate action arrays, and state hashes were bitwise exact. Replayed terminal block positions/errors differed from the stored arrays by at most {results['exact_reproduction']['simulator_terminal_position_max_abs_m']:.3g} m / {results['exact_reproduction']['simulator_terminal_error_max_abs_m']:.3g} m. Candidate 0 restored and repeated exactly for every start.",
        "",
        f"All original model calls, matched permutations, and raw selected indices reproduced exactly. The maximum absolute difference in the original lossless predicted raw goal-cost array was {results['exact_reproduction']['model_prior_goal_cost_max_abs']:.3g}, inside the predeclared V5 numerical contract.",
        "",
        "Terminal observations were rendered and encoded one 64-image start batch at a time. Only compact float32 actual terminal latents and pixel hashes were retained; no terminal-pixel archive remains.",
        "",
        "## Candidate-rollout prediction fidelity",
        "",
        "| condition | mean raw terminal latent MSE | mean V5-whitened terminal latent MSE | refiner calls | reached gate evaluations | counted FLOPs |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for name in CONDITIONS:
        condition = rollout["conditions"][name]
        compute = results["compute_accounting"]["aggregate"][name]
        lines.append(
            f"| {name} | {condition['mean_raw_mse']:.8f} | {condition['mean_whitened_mse']:.8f} | {compute['refiner_calls']} | {compute['reached_gate_evaluations']} | {compute['total_counted_flops']} |"
        )
    lines.extend(
        [
            "",
            "Paired effects use each start's mean over its fixed 64 candidates, so the independent sample size is 20.",
            "",
            "| metric | mean adaptive-minus-matched | median | 95% bootstrap interval for mean | 95% interval for median | starts favoring adaptive |",
            "|:--|--:|--:|:--|:--|--:|",
            f"| raw MSE | {raw['mean']:+.8g} | {raw['median']:+.8g} | [{raw['mean_bootstrap_95_interval'][0]:+.8g}, {raw['mean_bootstrap_95_interval'][1]:+.8g}] | [{raw['median_bootstrap_95_interval'][0]:+.8g}, {raw['median_bootstrap_95_interval'][1]:+.8g}] | {raw['starts_favoring_adaptive']}/20 |",
            f"| whitened MSE | {white['mean']:+.8g} | {white['median']:+.8g} | [{white['mean_bootstrap_95_interval'][0]:+.8g}, {white['mean_bootstrap_95_interval'][1]:+.8g}] | [{white['median_bootstrap_95_interval'][0]:+.8g}, {white['median_bootstrap_95_interval'][1]:+.8g}] | {white['starts_favoring_adaptive']}/20 |",
            "",
            "Adaptive and matched have identical depth histograms, refiner calls, reached-gate evaluations, and counted FLOPs separately at all 20x5 start/rollout-step cells. The lossless per-cell arrays are in `decomposition_metrics.npz`.",
            "",
            "## Latent-goal-cost alignment ceiling",
            "",
            "| actual-terminal cost | starts with defined physical rank | mean Spearman on defined starts | all-start mean top-5 overlap | chance | all-start mean selected physical regret (m) |",
            "|:--|--:|--:|--:|--:|--:|",
            f"| raw | {raw_goal['defined_spearman_starts_all']} | {raw_goal['mean_spearman_all_defined']:.6f} | {raw_goal['mean_top5_overlap_all']:.3f} | {info['chance_top5_overlap_fraction']:.6f} | {raw_goal['mean_selected_physical_regret_m_all']:.8g} |",
            f"| whitened | {white_goal['defined_spearman_starts_all']} | {white_goal['mean_spearman_all_defined']:.6f} | {white_goal['mean_top5_overlap_all']:.3f} | {info['chance_top5_overlap_fraction']:.6f} | {white_goal['mean_selected_physical_regret_m_all']:.8g} |",
            "",
            "No start met the fixed >=5-group informativeness rule, so these nine defined correlations are descriptive diagnostics, not a clean representation verdict. Still, using the actual future observation removes rollout error and did not reveal a useful physical ranking signal.",
            "",
            "By contrast, model-predicted costs tracked their corresponding actual-terminal latent costs well, even though neither tracked physical error usefully:",
            "",
            "| condition | raw: model vs actual-cost Spearman | raw top-5 overlap | whitened Spearman | whitened top-5 overlap |",
            "|:--|--:|--:|--:|--:|",
            *[
                f"| {name} | {results['goal_cost_alignment']['model_comparisons']['raw'][name]['versus_actual_terminal_latent_goal_cost']['aggregate']['mean_spearman_all_defined']:.6f} | {results['goal_cost_alignment']['model_comparisons']['raw'][name]['versus_actual_terminal_latent_goal_cost']['aggregate']['mean_top5_overlap_all']:.3f} | {results['goal_cost_alignment']['model_comparisons']['whitened'][name]['versus_actual_terminal_latent_goal_cost']['aggregate']['mean_spearman_all_defined']:.6f} | {results['goal_cost_alignment']['model_comparisons']['whitened'][name]['versus_actual_terminal_latent_goal_cost']['aggregate']['mean_top5_overlap_all']:.3f} |"
                for name in CONDITIONS
            ],
            "",
            "## Candidate-set informativeness",
            "",
            f"The fixed rule marked {info['informative_start_count']}/20 starts informative; {info['exactly_constant_start_count']} were exactly constant and {info['range_at_most_physical_tolerance_start_count']} had physical range at most 0.1 mm. Overall informativeness is **{decisions['candidate_informativeness']['status']}** under the sealed 15/20 rule. The 10 constant/tie-heavy starts are reported but are not evidence for or against planning quality.",
            "",
            "| start | range (m) | meaningful groups | tied fraction | top-5 boundary tie size | informative | raw adaptive-match regret contribution to mean (m) | raw selections differ |",
            "|:--|--:|--:|--:|--:|:--:|--:|:--:|",
        ]
    )
    raw_rows = {row["start_id"]: row for row in selection["per_start"]}
    for row in info["per_start"]:
        utility = raw_rows[row["start_id"]]
        lines.append(
            f"| {row['start_id']} | {row['physical_outcome_range_m']:.8g} | {row['meaningfully_distinct_outcomes']} | {row['fraction_tied_candidates']:.3f} | {row['physical_top5_boundary_tie_size']} | {'yes' if row['informative'] else 'no'} | {utility['adaptive_minus_matched_regret_contribution_to_20_start_mean_m']:+.8g} | {'yes' if utility['selected_different'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Adaptive and matched raw selections differed on {selection['adaptive_matched_selected_disagreement_count_all']}/20 starts and {selection['adaptive_matched_selected_disagreement_count_informative']} informative starts. Their all-start mean physical-regret difference was {selection['adaptive_minus_matched_mean_physical_regret_m_all']:+.8g} m. The per-start contributions above sum to that mean difference.",
            "",
            "## Link-by-link decision table",
            "",
            "| link | decision | reason |",
            "|:--|:--|:--|",
        ]
    )
    for key in (
        "candidate_rollout_transfer",
        "goal_cost_alignment",
        "candidate_informativeness",
        "downstream_planning_evidence",
    ):
        item = decisions[key]
        lines.append(f"| {key.replace('_', ' ')} | {item['status']} | {item['explanation']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The paper's chain is heterogeneous value of refinement -> learned equal-compute allocation -> multi-step prediction fidelity -> decision utility. V5 v004 established the first two links and the prior bridge gave a nominal positive five-step composition result. This decomposition diagnoses only the missing bridge from prediction fidelity to decision utility; it neither fits a new objective nor supplies downstream-control evidence.",
            "",
            "All intervals and ranks are descriptive because the pilot was already consumed and contains only 20 independent start clusters.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze() -> dict[str, Any]:
    if RESULTS_PATH.exists() or METRICS_PATH.exists():
        if not (RESULTS_PATH.exists() and METRICS_PATH.exists()):
            raise RuntimeError("partial final analysis artifacts exist")
        set_status("analyzed")
        return read_json(RESULTS_PATH)

    config = read_json(ROOT / "CONFIG.json")
    model = validate_model_reproduction()
    actual = validate_actual_latents()
    simulation = load_prior_simulation()
    prior_metrics = load_prior_metrics()
    actual_latent = actual["actual_terminal_latent"].astype(np.float32)
    predicted_latent = model["predicted_terminal_latent"].astype(np.float32)
    goal_latent = model["goal_latent"].astype(np.float32)
    physical_error = simulation["real_terminal_error"].astype(np.float64)
    with np.load(V5 / "freeze/whitening.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)

    rollout_raw = raw_mse(predicted_latent, actual_latent[None])
    rollout_white = whitened_mse(predicted_latent, actual_latent[None], whitening)
    actual_goal_raw = raw_mse(actual_latent, goal_latent[:, None, :])
    actual_goal_white = whitened_mse(
        actual_latent, goal_latent[:, None, :], whitening
    )
    predicted_goal_raw = model["predicted_goal_cost_raw"].astype(np.float64)
    predicted_goal_white = model["predicted_goal_cost_whitened"].astype(np.float64)

    physical_tolerance = float(config["tie_rules"]["physical_error_absolute_tolerance_m"])
    physical_group = np.stack(
        [tolerance_groups(physical_error[start], physical_tolerance) for start in range(START_COUNT)]
    )
    distinct = physical_group.max(axis=1).astype(np.int16) + 1
    ranges = np.ptp(physical_error, axis=1)
    tied_fraction = np.empty(START_COUNT, dtype=np.float64)
    boundary_tie_size = np.empty(START_COUNT, dtype=np.int16)
    informative = np.empty(START_COUNT, dtype=np.bool_)
    candidate_rows = []
    for start in range(START_COUNT):
        counts = np.bincount(physical_group[start])
        tied_fraction[start] = float(
            np.mean(counts[physical_group[start]] >= 2)
        )
        physical_top = stable_group_top_k(physical_group[start])
        boundary_tie_size[start] = int(counts[physical_group[start, physical_top[-1]]])
        informative[start] = bool(ranges[start] > physical_tolerance and distinct[start] >= 5)
        candidate_rows.append(
            {
                "start_id": str(simulation["start_id"][start]),
                "physical_outcome_min_m": float(np.min(physical_error[start])),
                "physical_outcome_max_m": float(np.max(physical_error[start])),
                "physical_outcome_range_m": float(ranges[start]),
                "exactly_distinct_outcomes": int(np.unique(physical_error[start]).size),
                "meaningfully_distinct_outcomes": int(distinct[start]),
                "fraction_tied_candidates": float(tied_fraction[start]),
                "physical_top5_boundary_tie_size": int(boundary_tie_size[start]),
                "informative": bool(informative[start]),
            }
        )

    bootstrap = config["bootstrap"]
    rng = np.random.default_rng(int(bootstrap["seed"]))
    resamples = rng.integers(
        0, START_COUNT, size=(int(bootstrap["replicates"]), START_COUNT)
    )
    per_start_raw = rollout_raw.mean(axis=2)
    per_start_white = rollout_white.mean(axis=2)
    raw_effect_vector = per_start_raw[0] - per_start_raw[1]
    white_effect_vector = per_start_white[0] - per_start_white[1]
    raw_effect = bootstrap_effect(raw_effect_vector, resamples)
    white_effect = bootstrap_effect(white_effect_vector, resamples)
    rollout_supported = bool(raw_effect["mean"] < 0.0 and white_effect["mean"] < 0.0)
    rollout_conditions = {
        name: {
            "mean_raw_mse": float(np.mean(rollout_raw[ci])),
            "median_raw_mse": float(np.median(rollout_raw[ci])),
            "mean_whitened_mse": float(np.mean(rollout_white[ci])),
            "median_whitened_mse": float(np.median(rollout_white[ci])),
        }
        for ci, name in enumerate(CONDITIONS)
    }

    actual_ceiling: dict[str, Any] = {}
    actual_ceiling_spearman = np.full((2, START_COUNT), np.nan, dtype=np.float64)
    actual_ceiling_top5 = np.empty((2, START_COUNT), dtype=np.float64)
    actual_ceiling_regret = np.empty((2, START_COUNT), dtype=np.float64)
    for space_index, (space, costs) in enumerate(
        (("raw", actual_goal_raw), ("whitened", actual_goal_white))
    ):
        per_start = []
        for start in range(START_COUNT):
            item = physical_rank_metrics(
                costs[start], physical_error[start], physical_group[start]
            )
            item["start_id"] = str(simulation["start_id"][start])
            per_start.append(item)
            if item["spearman"] is not None:
                actual_ceiling_spearman[space_index, start] = item["spearman"]
            actual_ceiling_top5[space_index, start] = item["top5_overlap"]
            actual_ceiling_regret[space_index, start] = item[
                "selected_physical_regret_m"
            ]
        aggregate = summarize_physical_metrics(per_start, informative)
        informative_rhos = [
            item["spearman"]
            for item, keep in zip(per_start, informative)
            if keep and item["spearman"] is not None
        ]
        interval = bootstrap_mean_interval(
            informative_rhos, int(bootstrap["seed"]), int(bootstrap["replicates"])
        )
        actual_ceiling[space] = {
            "aggregate": aggregate,
            "informative_mean_spearman_bootstrap_95_interval": interval,
            "per_start": per_start,
        }

    raw_alignment = actual_ceiling["raw"]
    raw_alignment_aggregate = raw_alignment["aggregate"]
    raw_alignment_interval = raw_alignment[
        "informative_mean_spearman_bootstrap_95_interval"
    ]
    alignment_supported = bool(
        informative.sum() > 0
        and raw_alignment_aggregate["mean_spearman_informative_defined"] is not None
        and raw_alignment_interval[0] > 0.0
        and raw_alignment_aggregate["mean_top5_overlap_informative"]
        > CANDIDATE_COUNT ** -1 * 5
    )

    model_comparisons: dict[str, Any] = {}
    model_vs_physical_spearman = np.full((2, 4, START_COUNT), np.nan, dtype=np.float64)
    model_vs_actual_spearman = np.full_like(model_vs_physical_spearman, np.nan)
    model_vs_physical_top5 = np.empty((2, 4, START_COUNT), dtype=np.float64)
    model_vs_actual_top5 = np.empty_like(model_vs_physical_top5)
    for space_index, (space, predicted_cost, actual_cost) in enumerate(
        (
            ("raw", predicted_goal_raw, actual_goal_raw),
            ("whitened", predicted_goal_white, actual_goal_white),
        )
    ):
        by_condition: dict[str, Any] = {}
        for ci, name in enumerate(CONDITIONS):
            physical_rows = []
            actual_rows = []
            for start in range(START_COUNT):
                physical_item = physical_rank_metrics(
                    predicted_cost[ci, start],
                    physical_error[start],
                    physical_group[start],
                )
                actual_item = target_rank_metrics(
                    predicted_cost[ci, start], actual_cost[start]
                )
                physical_item["start_id"] = str(simulation["start_id"][start])
                actual_item["start_id"] = str(simulation["start_id"][start])
                physical_rows.append(physical_item)
                actual_rows.append(actual_item)
                if physical_item["spearman"] is not None:
                    model_vs_physical_spearman[space_index, ci, start] = physical_item[
                        "spearman"
                    ]
                if actual_item["spearman"] is not None:
                    model_vs_actual_spearman[space_index, ci, start] = actual_item[
                        "spearman"
                    ]
                model_vs_physical_top5[space_index, ci, start] = physical_item[
                    "top5_overlap"
                ]
                model_vs_actual_top5[space_index, ci, start] = actual_item["top5_overlap"]
            by_condition[name] = {
                "versus_physical_error": {
                    "aggregate": summarize_physical_metrics(physical_rows, informative),
                    "per_start": physical_rows,
                },
                "versus_actual_terminal_latent_goal_cost": {
                    "aggregate": summarize_target_metrics(actual_rows, informative),
                    "per_start": actual_rows,
                },
            }
        model_comparisons[space] = by_condition

    informative_count = int(np.sum(informative))
    informativeness_supported = informative_count >= 15
    chance_overlap = 5 / CANDIDATE_COUNT

    selection_utility: dict[str, Any] = {}
    selection_indices = np.empty((2, 4, START_COUNT), dtype=np.int16)
    selection_regret = np.empty((2, 4, START_COUNT), dtype=np.float64)
    for space_index, (space, predicted_cost) in enumerate(
        (("raw_goal_cost", predicted_goal_raw), ("whitened_goal_cost", predicted_goal_white))
    ):
        for ci in range(4):
            selection_indices[space_index, ci] = np.argmin(
                predicted_cost[ci], axis=1
            ).astype(np.int16)
            for start in range(START_COUNT):
                selected = selection_indices[space_index, ci, start]
                selection_regret[space_index, ci, start] = (
                    physical_error[start, selected] - np.min(physical_error[start])
                )
        effect = selection_regret[space_index, 0] - selection_regret[space_index, 1]
        different = selection_indices[space_index, 0] != selection_indices[space_index, 1]
        rows = []
        for start in range(START_COUNT):
            rows.append(
                {
                    "start_id": str(simulation["start_id"][start]),
                    "informative": bool(informative[start]),
                    "adaptive_selected_index": int(selection_indices[space_index, 0, start]),
                    "matched_selected_index": int(selection_indices[space_index, 1, start]),
                    "selected_different": bool(different[start]),
                    "adaptive_physical_regret_m": float(selection_regret[space_index, 0, start]),
                    "matched_physical_regret_m": float(selection_regret[space_index, 1, start]),
                    "adaptive_minus_matched_physical_regret_m": float(effect[start]),
                    "adaptive_minus_matched_regret_contribution_to_20_start_mean_m": float(
                        effect[start] / START_COUNT
                    ),
                }
            )
        selection_utility[space] = {
            "adaptive_matched_selected_disagreement_count_all": int(np.sum(different)),
            "adaptive_matched_selected_disagreement_count_informative": int(
                np.sum(different & informative)
            ),
            "adaptive_minus_matched_mean_physical_regret_m_all": float(np.mean(effect)),
            "adaptive_minus_matched_mean_physical_regret_m_informative": (
                float(np.mean(effect[informative])) if informative_count else None
            ),
            "per_start": rows,
        }
    if not np.array_equal(selection_indices[0], prior_metrics["selected_candidate_index"]):
        raise RuntimeError("analysis raw selection indices drifted from exact reproduction")

    compute, compute_arrays = compute_accounting(
        model["adaptive_calls"], model["matched_calls"]
    )
    raw_selection = selection_utility["raw_goal_cost"]
    selection_link = bool(
        raw_selection["adaptive_matched_selected_disagreement_count_informative"] >= 1
        and raw_selection["adaptive_minus_matched_mean_physical_regret_m_informative"]
        is not None
        and raw_selection["adaptive_minus_matched_mean_physical_regret_m_informative"] < 0.0
    )
    downstream_supported = bool(
        rollout_supported
        and alignment_supported
        and informativeness_supported
        and selection_link
    )

    decision_table = {
        "candidate_rollout_transfer": {
            "status": "supported" if rollout_supported else "unsupported",
            "explanation": (
                "adaptive start-cluster mean terminal MSE is lower than matched in both spaces under the sealed directional rule, but both descriptive 95% intervals cross zero"
                if rollout_supported
                else "adaptive start-cluster mean terminal MSE is not lower than matched in both raw and whitened spaces"
            ),
        },
        "goal_cost_alignment": {
            "status": "supported" if alignment_supported else "unsupported",
            "explanation": (
                "the actual-terminal raw goal cost clears the fixed positive-rank interval and above-chance top-5 rules on informative starts"
                if alignment_supported
                else f"no useful alignment is supported: {informative_count}/20 starts are informative, while the {actual_ceiling['raw']['aggregate']['defined_spearman_starts_all']} subthreshold rank-defined starts have raw mean Spearman {actual_ceiling['raw']['aggregate']['mean_spearman_all_defined']:.6f} and all-start top-5 overlap {actual_ceiling['raw']['aggregate']['mean_top5_overlap_all']:.3f} versus 0.078125 chance"
            ),
        },
        "candidate_informativeness": {
            "status": "supported" if informativeness_supported else "unsupported",
            "explanation": f"{informative_count}/20 starts meet the fixed >=5 meaningful groups and >0.1 mm range rule; 15/20 were required",
        },
        "downstream_planning_evidence": {
            "status": "supported" if downstream_supported else "unsupported",
            "explanation": (
                "all three links are supported and adaptive changes an informative-start selection with lower informative-start mean regret"
                if downstream_supported
                else "goal-cost alignment and candidate informativeness remain unsupported, and adaptive changes no informative-start selection; the nominal prior ranking label does not justify planning or control"
            ),
        },
    }

    results = {
        "schema_version": 1,
        "scientific_role": "descriptive_decomposition_of_consumed_pilot",
        "exact_reproduction": {
            "all_configured_source_hashes_verified": True,
            "all_start_identifiers_and_seed_tuples_exact": True,
            "candidate_actions_bitwise_exact_all_starts": True,
            "warm_start_pixels_goal_pixels_past_actions_and_state_hashes_exact": True,
            "restore_repeat_exact_all_starts": True,
            "simulator_terminal_position_max_abs_m": float(
                np.max(actual["terminal_position_max_abs"])
            ),
            "simulator_terminal_error_max_abs_m": float(
                np.max(actual["terminal_error_max_abs"])
            ),
            "model_prior_goal_cost_max_abs": float(model["prior_goal_cost_max_abs"]),
            "model_calls_permutations_and_selected_indices_exact": True,
            "frozen_modules_and_gradients_unchanged": True,
        },
        "candidate_rollout_fidelity": {
            "conditions": rollout_conditions,
            "adaptive_minus_matched": {"raw": raw_effect, "whitened": white_effect},
            "cluster_definition": "each of 20 starts; 64 fixed candidates averaged within start",
            "bootstrap": bootstrap,
        },
        "goal_cost_alignment": {
            "actual_terminal_ceiling": actual_ceiling,
            "model_comparisons": model_comparisons,
            "interpretation": "actual-terminal ceiling removes rollout prediction error and isolates frozen goal-cost representation alignment",
        },
        "diagnostic_summary": {
            "rollout_transfer": "adaptive has a tiny directional mean-MSE edge over matched in both spaces, with intervals crossing zero",
            "model_cost_tracks_actual_latent_cost": {
                "adaptive_raw_mean_spearman": model_comparisons["raw"]["adaptive"]["versus_actual_terminal_latent_goal_cost"]["aggregate"]["mean_spearman_all_defined"],
                "adaptive_raw_mean_top5_overlap": model_comparisons["raw"]["adaptive"]["versus_actual_terminal_latent_goal_cost"]["aggregate"]["mean_top5_overlap_all"],
                "adaptive_whitened_mean_spearman": model_comparisons["whitened"]["adaptive"]["versus_actual_terminal_latent_goal_cost"]["aggregate"]["mean_spearman_all_defined"],
                "adaptive_whitened_mean_top5_overlap": model_comparisons["whitened"]["adaptive"]["versus_actual_terminal_latent_goal_cost"]["aggregate"]["mean_top5_overlap_all"],
            },
            "actual_latent_cost_tracks_physical_error": {
                "rank_defined_start_count": actual_ceiling["raw"]["aggregate"]["defined_spearman_starts_all"],
                "raw_mean_spearman_defined": actual_ceiling["raw"]["aggregate"]["mean_spearman_all_defined"],
                "raw_all_start_mean_top5_overlap": actual_ceiling["raw"]["aggregate"]["mean_top5_overlap_all"],
                "whitened_mean_spearman_defined": actual_ceiling["whitened"]["aggregate"]["mean_spearman_all_defined"],
                "whitened_all_start_mean_top5_overlap": actual_ceiling["whitened"]["aggregate"]["mean_top5_overlap_all"],
            },
            "interpretation": "the model ranks its own latent goal-cost target substantially better than that target ranks physical outcomes, but zero starts meet the fixed informativeness rule, so the consumed pool cannot isolate representation failure cleanly",
        },
        "candidate_set_informativeness": {
            "physical_tolerance_m": physical_tolerance,
            "informative_start_rule": config["decision_table"]["candidate_informativeness"][
                "informative_start_rule"
            ],
            "informative_start_count": informative_count,
            "exactly_constant_start_count": int(np.sum(ranges == 0.0)),
            "range_at_most_1e-12_m_start_count": int(np.sum(ranges <= 1e-12)),
            "range_at_most_physical_tolerance_start_count": int(
                np.sum(ranges <= physical_tolerance)
            ),
            "chance_top5_overlap_fraction": chance_overlap,
            "constant_or_uninformative_starts_are_not_planning_evidence": True,
            "per_start": candidate_rows,
        },
        "selection_utility": selection_utility,
        "compute_accounting": compute,
        "decision_table": decision_table,
        "downstream_planning_claim_justified": downstream_supported,
        "next_experiment": (
            "Run one preregistered, independently seeded contact-active 20x64 ranking cohort with the same frozen objects and exact compute matching, opening model costs only after at least 15 starts satisfy the same >=5-group/>0.1 mm informativeness rule."
            if downstream_supported
            else None
        ),
        "strongest_caveats": [
            "The 20-start pilot is already consumed; all intervals are descriptive.",
            "The candidate pool is PlanOracle-centered and was not designed to ensure contact-active outcome spread.",
            "Ten or more starts are constant or tie-heavy under physically meaningful tolerances and cannot support ranking conclusions.",
            "Physical utility is terminal target-block position error only.",
            "No fitting, new cohort, candidate filtering, threshold change, or closed-loop control was performed.",
        ],
        "prohibited_inputs_opened": config["prohibited_inputs_opened"],
        "arrays": {
            "model_reproduction": {
                "path": str(MODEL_PATH.relative_to(REPO)),
                "sha256": sha256_file(MODEL_PATH),
            },
            "actual_terminal_latents": {
                "path": str(ACTUAL_PATH.relative_to(REPO)),
                "sha256": sha256_file(ACTUAL_PATH),
                "terminal_pixels_retained": False,
            },
        },
    }

    metric_arrays = {
        "condition": np.asarray(CONDITIONS, dtype="U12"),
        "start_id": simulation["start_id"],
        "physical_terminal_error": physical_error,
        "physical_tolerance_group": physical_group,
        "physical_outcome_range": ranges,
        "meaningfully_distinct_outcomes": distinct,
        "fraction_tied_candidates": tied_fraction,
        "physical_top5_boundary_tie_size": boundary_tie_size,
        "informative_start": informative,
        "rollout_terminal_mse_raw": rollout_raw,
        "rollout_terminal_mse_whitened": rollout_white,
        "rollout_start_mean_mse_raw": per_start_raw,
        "rollout_start_mean_mse_whitened": per_start_white,
        "adaptive_minus_matched_start_effect_raw": raw_effect_vector,
        "adaptive_minus_matched_start_effect_whitened": white_effect_vector,
        "actual_terminal_goal_cost_raw": actual_goal_raw,
        "actual_terminal_goal_cost_whitened": actual_goal_white,
        "predicted_goal_cost_raw": predicted_goal_raw,
        "predicted_goal_cost_whitened": predicted_goal_white,
        "actual_ceiling_spearman": actual_ceiling_spearman,
        "actual_ceiling_top5_overlap": actual_ceiling_top5,
        "actual_ceiling_physical_regret": actual_ceiling_regret,
        "model_vs_physical_spearman": model_vs_physical_spearman,
        "model_vs_actual_goal_cost_spearman": model_vs_actual_spearman,
        "model_vs_physical_top5_overlap": model_vs_physical_top5,
        "model_vs_actual_goal_cost_top5_overlap": model_vs_actual_top5,
        "selected_candidate_index": selection_indices,
        "selected_physical_regret": selection_regret,
        "compute_depth_histogram": compute_arrays["depth_histogram"],
        "compute_refiner_calls": compute_arrays["refiner_calls"],
        "compute_reached_gate_evaluations": compute_arrays["reached_gate_evaluations"],
        "compute_counted_flops": compute_arrays["counted_flops"],
    }
    write_npz_atomic(METRICS_PATH, metric_arrays, exclusive=True)
    results["arrays"]["decomposition_metrics"] = {
        "path": str(METRICS_PATH.relative_to(REPO)),
        "sha256": sha256_file(METRICS_PATH),
    }
    write_json_atomic(RESULTS_PATH, results, exclusive=True)
    first_table = render_first_table(results)
    write_text_atomic(ROOT / "FIRST_DECOMPOSITION_TABLE.md", first_table, exclusive=True)
    write_text_atomic(ROOT / "REPORT.md", render_report(results), exclusive=True)
    set_status("analyzed")
    append_note(
        "Produced the first decomposition table and completed the fixed analyses of rollout "
        "fidelity, actual-latent goal-cost alignment, and candidate-set informativeness. "
        f"Link decisions: { {key: value['status'] for key, value in decision_table.items()} }."
    )
    return results


def refresh_reporting() -> dict[str, Any]:
    """Refresh wording/summary fields without changing sealed metrics or decisions."""
    results = read_json(RESULTS_PATH)
    info = results["candidate_set_informativeness"]
    if "range_at_most_1e12_start_count" in info:
        info["range_at_most_1e-12_m_start_count"] = info.pop(
            "range_at_most_1e12_start_count"
        )
    raw_effect = results["candidate_rollout_fidelity"]["adaptive_minus_matched"]["raw"]
    white_effect = results["candidate_rollout_fidelity"]["adaptive_minus_matched"][
        "whitened"
    ]
    ceiling = results["goal_cost_alignment"]["actual_terminal_ceiling"]
    raw_ceiling = ceiling["raw"]["aggregate"]
    white_ceiling = ceiling["whitened"]["aggregate"]
    comparisons = results["goal_cost_alignment"]["model_comparisons"]
    adaptive_raw_actual = comparisons["raw"]["adaptive"][
        "versus_actual_terminal_latent_goal_cost"
    ]["aggregate"]
    adaptive_white_actual = comparisons["whitened"]["adaptive"][
        "versus_actual_terminal_latent_goal_cost"
    ]["aggregate"]
    results["diagnostic_summary"] = {
        "rollout_transfer": (
            "adaptive has a tiny directional mean-MSE edge over matched in both spaces, "
            "with both descriptive 95% intervals crossing zero"
        ),
        "model_cost_tracks_actual_latent_cost": {
            "adaptive_raw_mean_spearman": adaptive_raw_actual[
                "mean_spearman_all_defined"
            ],
            "adaptive_raw_mean_top5_overlap": adaptive_raw_actual[
                "mean_top5_overlap_all"
            ],
            "adaptive_whitened_mean_spearman": adaptive_white_actual[
                "mean_spearman_all_defined"
            ],
            "adaptive_whitened_mean_top5_overlap": adaptive_white_actual[
                "mean_top5_overlap_all"
            ],
        },
        "actual_latent_cost_tracks_physical_error": {
            "rank_defined_start_count": raw_ceiling["defined_spearman_starts_all"],
            "raw_mean_spearman_defined": raw_ceiling["mean_spearman_all_defined"],
            "raw_all_start_mean_top5_overlap": raw_ceiling["mean_top5_overlap_all"],
            "whitened_mean_spearman_defined": white_ceiling[
                "mean_spearman_all_defined"
            ],
            "whitened_all_start_mean_top5_overlap": white_ceiling[
                "mean_top5_overlap_all"
            ],
        },
        "interpretation": (
            "the model ranks its own latent goal-cost target substantially better than that "
            "target ranks physical outcomes, but zero starts meet the fixed informativeness "
            "rule, so the consumed pool cannot isolate representation failure cleanly"
        ),
    }
    decisions = results["decision_table"]
    decisions["candidate_rollout_transfer"]["explanation"] = (
        "adaptive start-cluster mean terminal MSE is lower than matched in both spaces under "
        "the sealed directional rule, but both descriptive 95% intervals cross zero"
    )
    decisions["goal_cost_alignment"]["explanation"] = (
        f"no useful alignment is supported: {info['informative_start_count']}/20 starts are "
        f"informative, while the {raw_ceiling['defined_spearman_starts_all']} subthreshold "
        f"rank-defined starts have raw mean Spearman "
        f"{raw_ceiling['mean_spearman_all_defined']:.6f} and all-start top-5 overlap "
        f"{raw_ceiling['mean_top5_overlap_all']:.3f} versus 0.078125 chance"
    )
    decisions["downstream_planning_evidence"]["explanation"] = (
        "goal-cost alignment and candidate informativeness remain unsupported, and adaptive "
        "changes no informative-start selection; the nominal prior ranking label does not "
        "justify planning or control"
    )
    results["reporting_clarification"] = {
        "scientific_rules_or_metric_arrays_changed": False,
        "reason": (
            "when zero starts met the sealed informativeness rule, report all rank-defined "
            "subthreshold diagnostics explicitly rather than displaying null informative-only "
            "summaries"
        ),
        "raw_rollout_effect_unchanged": raw_effect["mean"],
        "whitened_rollout_effect_unchanged": white_effect["mean"],
    }
    write_json_atomic(RESULTS_PATH, results)
    write_text_atomic(ROOT / "FIRST_DECOMPOSITION_TABLE.md", render_first_table(results))
    write_text_atomic(ROOT / "REPORT.md", render_report(results))
    append_note(
        "Corrected report presentation after the sealed informativeness rule yielded 0/20 "
        "informative starts: retained all metric arrays and decisions unchanged, and exposed "
        "the nine rank-defined subthreshold ceiling diagnostics instead of null informative-only "
        "summaries."
    )
    return {
        "metrics_or_rules_changed": False,
        "decision_statuses": {
            key: value["status"] for key, value in decisions.items()
        },
    }


def verify_final() -> dict[str, Any]:
    verify_source_hashes()
    model = validate_model_reproduction()
    actual = validate_actual_latents()
    results = read_json(RESULTS_PATH)
    with np.load(METRICS_PATH, allow_pickle=False) as stored:
        metrics = {name: stored[name].copy() for name in stored.files}
    expected_shapes = {
        "rollout_terminal_mse_raw": (4, 20, 64),
        "rollout_terminal_mse_whitened": (4, 20, 64),
        "actual_terminal_goal_cost_raw": (20, 64),
        "actual_terminal_goal_cost_whitened": (20, 64),
        "compute_depth_histogram": (4, 20, 5, 4),
        "compute_refiner_calls": (4, 20, 5),
        "compute_reached_gate_evaluations": (4, 20, 5),
        "compute_counted_flops": (4, 20, 5),
    }
    for name, shape in expected_shapes.items():
        if metrics[name].shape != shape:
            raise RuntimeError(f"final metric shape drift: {name}")
        if not np.isfinite(metrics[name]).all():
            raise RuntimeError(f"nonfinite final metric array: {name}")
    for name, record in results["arrays"].items():
        path = REPO / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"result array hash drift: {name}")
    if not np.array_equal(model["start_id"], actual["start_id"]):
        raise RuntimeError("model/actual start identifier mismatch")
    status = read_json(ROOT / "RUN_STATUS.json")
    return {
        "source_hashes_verified": True,
        "arrays_verified": True,
        "status": status,
        "decision_table": results["decision_table"],
    }


def finalize(focused_test_count: int) -> dict[str, Any]:
    verification = verify_final()
    status = verification["status"]
    for field in ("configured", "simulator_verified", "model_verified", "analyzed"):
        if not status[field]:
            raise RuntimeError(f"cannot finalize before {field}")
    if (ROOT / "ARTIFACT_HASHES.json").exists():
        raise RuntimeError("ARTIFACT_HASHES.json already exists")
    set_status("complete")
    append_note(
        f"Focused tests passed ({focused_test_count}); final artifact verification passed and "
        "the descriptive decomposition is complete."
    )
    artifacts = {}
    for path in sorted(ROOT.iterdir()):
        if not path.is_file() or path.name == "ARTIFACT_HASHES.json":
            continue
        if path.name.startswith(".") or path.suffix in {".pyc"}:
            continue
        artifacts[path.name] = sha256_file(path)
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "self_excluded": True,
        "focused_test_count": int(focused_test_count),
        "artifacts": artifacts,
        "created_unix_ns": time.time_ns(),
    }
    write_json_atomic(ROOT / "ARTIFACT_HASHES.json", payload, exclusive=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("run", "model", "simulator", "analyze", "refresh", "verify", "finalize"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--focused-test-count", type=int, default=0)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(jsonable(verify_final()), sort_keys=True))
        return
    if args.command == "finalize":
        if args.focused_test_count <= 0:
            raise RuntimeError("finalize requires a positive --focused-test-count")
        print(json.dumps(jsonable(finalize(args.focused_test_count)), sort_keys=True))
        return
    if args.command == "refresh":
        verify_source_hashes()
        print(json.dumps(jsonable(refresh_reporting()), sort_keys=True))
        return
    verify_source_hashes()
    if args.command == "analyze":
        print(json.dumps(jsonable(analyze()), sort_keys=True))
        return
    bridge, _, _ = prior_modules()
    stack = bridge.load_stack(args.device)
    if args.command in ("run", "model"):
        reproduce_model(stack)
    if args.command in ("run", "simulator"):
        if not read_json(ROOT / "RUN_STATUS.json")["model_verified"]:
            raise RuntimeError("model reproduction must pass before actual-terminal generation")
        replay_and_encode(stack)
    if args.command == "run":
        analyze()
    print(json.dumps({"command": args.command, "status": read_json(ROOT / "RUN_STATUS.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
