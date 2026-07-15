#!/usr/bin/env python3
"""Run the preregistered direct LeWM variable-depth refinement pilot."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from allocation import (  # noqa: E402
    exact_budget_optimal_allocation,
    fit_whitening,
    gather_depths,
    latent_mse_by_depth,
    paired_cluster_bootstrap,
    permute_allocation,
    predicted_gain_exact_budget_allocation,
    random_exact_budget_allocation,
    robust_relative_gain,
    summarize_allocation,
    total_block_calls,
    validate_split_isolation,
)
from model_io import (  # noqa: E402
    choose_device,
    deterministic_episode_split,
    extract_fresh_cube_cache,
    synchronize,
)
from plots import write_all_figures  # noqa: E402
from refiner import (  # noqa: E402
    EXITS,
    SharedResidualRefiner,
    build_causal_gate_features,
    validate_causal_feature_names,
)


DEFAULT_SOURCE_H5 = Path(
    "/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5"
)
DEFAULT_MODEL_CONFIG = EXPERIMENT_DIR.parent / "lewm_transfer" / "cube" / "cache" / "model" / "config.json"
DEFAULT_MODEL_WEIGHTS = EXPERIMENT_DIR.parent / "lewm_transfer" / "cube" / "cache" / "model" / "weights.pt"
SPLIT_CODES = {"train": 0, "calibration": 1, "test": 2}
DEPTHS = tuple(EXITS)


def print_step(message: str) -> None:
    print(f"[lewm-adaptive] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "full_config.json")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_SOURCE_H5)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-weights", type=Path, default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def trimmed_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(math.floor(len(array) * fraction))
    if 2 * trim >= len(array):
        return float(array.mean())
    return float(array[trim : len(array) - trim].mean())


def configured_split(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    if cfg["mode"] == "full":
        excluded_bounds = cfg["excluded_episode_ordinals"]
        excluded = range(int(excluded_bounds[0]), int(excluded_bounds[1]) + 1)
        split = deterministic_episode_split(
            total_episodes=10_000,
            excluded=excluded,
            seed=int(cfg["seed"]),
            train_count=int(cfg["train_episodes"]),
            calibration_count=int(cfg["calibration_episodes"]),
            test_count=int(cfg["test_episodes"]),
        )
    elif cfg["mode"] == "smoke":
        split = {
            "train": np.asarray(cfg["train_episode_ordinals"], dtype=np.int64),
            "calibration": np.asarray(cfg["calibration_episode_ordinals"], dtype=np.int64),
            "test": np.asarray(cfg["test_episode_ordinals"], dtype=np.int64),
        }
    else:
        raise ValueError(f"Unknown run mode {cfg['mode']!r}")
    validate_split_isolation(split)
    return split


def load_cache(path: Path, expected_split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    required = {
        "history",
        "action",
        "base_pred",
        "target",
        "episode_id",
        "model_step",
        "split",
        "interaction",
        "impact",
        "effector_disp",
        "block_disp",
        "normalized_phase",
    }
    missing = required - set(arrays)
    if missing:
        raise RuntimeError(f"Latent cache is missing {sorted(missing)}")
    n = len(arrays["target"])
    for key in required:
        if len(arrays[key]) != n:
            raise RuntimeError(f"Cache length mismatch for {key}")
    for name, code in SPLIT_CODES.items():
        observed = set(np.unique(arrays["episode_id"][arrays["split"] == code]).tolist())
        expected = set(np.asarray(expected_split[name]).tolist())
        if observed != expected:
            raise RuntimeError(f"Cached {name} episodes do not match frozen split")
        counts = pd.Series(arrays["episode_id"][arrays["split"] == code]).value_counts()
        if len(counts) == 0 or not np.all(counts.to_numpy() == 38):
            raise RuntimeError(f"Expected exactly 38 examples per {name} episode")
    validate_split_isolation(
        {
            name: np.unique(arrays["episode_id"][arrays["split"] == code])
            for name, code in SPLIT_CODES.items()
        }
    )
    for key in ("history", "action", "base_pred", "target"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"Nonfinite cache tensor: {key}")
    return arrays


def make_refiner(cfg: dict[str, Any], arrays: dict[str, np.ndarray], seed: int) -> SharedResidualRefiner:
    torch.manual_seed(seed)
    return SharedResidualRefiner(
        latent_dim=int(arrays["target"].shape[1]),
        action_dim=int(arrays["action"].shape[2]),
        history_len=int(arrays["history"].shape[1]),
        hidden_dim=int(cfg["refiner_hidden_dim"]),
        iteration_dim=int(cfg["iteration_embedding_dim"]),
    )


def _batch_tensors(
    arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        torch.from_numpy(np.ascontiguousarray(arrays[key][indices])).to(device)
        for key in ("history", "action", "base_pred", "target")
    )  # type: ignore[return-value]


def evaluation_objective(
    model: SharedResidualRefiner,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            history, action, base, target = _batch_tensors(arrays, batch_indices, device)
            outputs = model(history, action, base, depths=(1, 2, 4))
            per_depth = torch.stack([(outputs[depth] - target).square().mean() for depth in (1, 2, 4)])
            total += float(per_depth.mean().item()) * len(batch_indices)
            count += len(batch_indices)
    return total / count


def train_one_seed(
    *,
    seed: int,
    cfg: dict[str, Any],
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    device: torch.device,
    checkpoint_dir: Path,
    force: bool,
) -> tuple[SharedResidualRefiner, dict[str, Any], list[dict[str, Any]]]:
    checkpoint_path = checkpoint_dir / f"refiner_seed_{seed}.pt"
    summary_path = checkpoint_dir / f"refiner_seed_{seed}.json"
    model = make_refiner(cfg, arrays, seed).to(device)
    if checkpoint_path.exists() and summary_path.exists() and not force:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.to(device)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return model, summary, []

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    batch_size = int(cfg["train_batch_size"])
    best_score = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        shuffled = rng.permutation(train_indices)
        total_loss = 0.0
        total_examples = 0
        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            history, action, base, target = _batch_tensors(arrays, batch_indices, device)
            outputs, updates = model.forward_with_updates(history, action, base)
            prediction_loss = torch.stack(
                [(outputs[depth] - target).square().mean() for depth in (1, 2, 4)]
            ).mean()
            update_penalty = torch.stack([update.square().mean() for update in updates]).mean()
            loss = prediction_loss + float(cfg["update_penalty"]) * update_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch_indices)
            total_examples += len(batch_indices)

        calibration_score = evaluation_objective(
            model, arrays, calibration_indices, device, batch_size
        )
        train_score = total_loss / total_examples
        improved = calibration_score < best_score - float(cfg["early_stopping_min_delta"])
        if improved:
            best_score = calibration_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_objective": train_score,
                "calibration_objective": calibration_score,
                "best_calibration_objective": best_score,
                "improved": improved,
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or stale >= int(cfg["early_stopping_patience"]):
            print_step(
                f"seed={seed} epoch={epoch + 1} train={train_score:.6g} "
                f"cal={calibration_score:.6g} best={best_score:.6g} stale={stale}"
            )
        if stale >= int(cfg["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Refiner training never produced a finite best state")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "seed": seed}, checkpoint_path)
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(rows),
        "best_calibration_objective": best_score,
        "elapsed_seconds": time.perf_counter() - start_time,
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    write_json(summary_path, summary)
    return model, summary, rows


def evaluate_exits(
    model: SharedResidualRefiner,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = np.empty((len(indices), len(DEPTHS), arrays["target"].shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            history, action, base, _ = _batch_tensors(arrays, batch_indices, device)
            exits = model(history, action, base, depths=DEPTHS)
            for column, depth in enumerate(DEPTHS):
                output[start : start + len(batch_indices), column] = exits[depth].detach().cpu().numpy()
    expected_base = arrays["base_pred"][indices]
    if not np.array_equal(output[:, 0], expected_base):
        max_abs = float(np.max(np.abs(output[:, 0] - expected_base)))
        raise RuntimeError(f"Depth-0 identity failed: max_abs={max_abs}")
    return output


def stage_a_analysis(
    *,
    raw_losses: np.ndarray,
    whitened_losses: np.ndarray,
    relative_gain: np.ndarray,
    episode_ids: np.ndarray,
    cfg: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    n = len(raw_losses)
    bootstrap_kwargs = {
        "n_bootstrap": int(cfg["bootstrap_samples"]),
        "seed": int(cfg["bootstrap_seed"]),
    }
    depth_rows: list[dict[str, Any]] = []
    response_passes = []
    for column, depth in enumerate(DEPTHS):
        gain = raw_losses[:, 0] - raw_losses[:, column]
        ci = paired_cluster_bootstrap(gain, episode_ids, **bootstrap_kwargs)
        row = {
            "scope": label,
            "depth": depth,
            "raw_mean_loss": float(raw_losses[:, column].mean()),
            "raw_median_loss": float(np.median(raw_losses[:, column])),
            "raw_trimmed_mean_loss": trimmed_mean(raw_losses[:, column]),
            "whitened_mean_loss": float(whitened_losses[:, column].mean()),
            "mean_gain": ci.estimate,
            "gain_ci_low": ci.lower,
            "gain_ci_high": ci.upper,
            "mean_robust_relative_gain": float(relative_gain[:, column].mean()),
            "n": n,
            "n_episodes": int(len(np.unique(episode_ids))),
        }
        depth_rows.append(row)
        if depth > 0:
            response_passes.append(ci.lower > 0.0)

    budget_rows: list[dict[str, Any]] = [
        {
            "scope": label,
            "mean_budget": 0,
            "mean_calls": 0.0,
            "strategy": "depth0",
            "total_calls": 0,
            "raw_mean_loss": float(raw_losses[:, 0].mean()),
            "whitened_mean_loss": float(whitened_losses[:, 0].mean()),
        }
    ]
    allocations: dict[str, np.ndarray] = {"depth0_b0": np.zeros(n, dtype=np.int64)}
    headroom_passes = []
    threshold = 0.005 * float(raw_losses[:, 0].mean())
    for mean_budget in (1, 2):
        total_budget = mean_budget * n
        uniform = np.full(n, mean_budget, dtype=np.int64)
        oracle = exact_budget_optimal_allocation(raw_losses, DEPTHS, total_budget)
        random = random_exact_budget_allocation(
            n, DEPTHS, total_budget, np.random.default_rng(int(cfg["bootstrap_seed"]) + mean_budget)
        )
        permuted = permute_allocation(
            oracle, np.random.default_rng(int(cfg["bootstrap_seed"]) + 100 + mean_budget)
        )
        strategy_allocations = {
            "uniform": uniform,
            "oracle": oracle,
            "random": random,
            "permuted": permuted,
        }
        uniform_loss = gather_depths(raw_losses, uniform, DEPTHS)
        for strategy, allocation in strategy_allocations.items():
            selected_raw = gather_depths(raw_losses, allocation, DEPTHS)
            selected_white = gather_depths(whitened_losses, allocation, DEPTHS)
            summary = summarize_allocation(raw_losses, allocation, DEPTHS)
            comparison = paired_cluster_bootstrap(
                uniform_loss - selected_raw, episode_ids, **bootstrap_kwargs
            )
            budget_rows.append(
                {
                    "scope": label,
                    "mean_budget": mean_budget,
                    "mean_calls": float(summary["mean_calls"]),
                    "strategy": strategy,
                    "total_calls": int(summary["total_calls"]),
                    "raw_mean_loss": float(selected_raw.mean()),
                    "whitened_mean_loss": float(selected_white.mean()),
                    "mean_gain_vs_depth0": float((raw_losses[:, 0] - selected_raw).mean()),
                    "mean_advantage_vs_uniform": comparison.estimate,
                    "advantage_vs_uniform_ci_low": comparison.lower,
                    "advantage_vs_uniform_ci_high": comparison.upper,
                    "depth_histogram": json.dumps(summary["depth_histogram"], sort_keys=True),
                    "diagnostic_only": strategy == "oracle",
                }
            )
            allocations[f"{strategy}_b{mean_budget}"] = allocation
        oracle_delta = uniform_loss - gather_depths(raw_losses, oracle, DEPTHS)
        oracle_ci = paired_cluster_bootstrap(oracle_delta, episode_ids, **bootstrap_kwargs)
        headroom_passes.append(oracle_ci.lower > 0.0 and oracle_ci.estimate >= threshold)

    result = {
        "scope": label,
        "compute_response_passed": bool(any(response_passes)),
        "oracle_headroom_passed": bool(any(headroom_passes)),
        "meaningful_headroom_threshold": threshold,
        "n": n,
        "n_episodes": int(len(np.unique(episode_ids))),
    }
    return result, depth_rows, budget_rows, allocations


def causal_features_numpy(arrays: dict[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    chunks = []
    names: tuple[str, ...] | None = None
    for start in range(0, len(indices), 4096):
        batch = indices[start : start + 4096]
        history = torch.from_numpy(np.ascontiguousarray(arrays["history"][batch]))
        action = torch.from_numpy(np.ascontiguousarray(arrays["action"][batch]))
        base = torch.from_numpy(np.ascontiguousarray(arrays["base_pred"][batch]))
        features, batch_names = build_causal_gate_features(history, action, base, return_names=True)
        validate_causal_feature_names(batch_names)
        if names is None:
            names = batch_names
        elif names != batch_names:
            raise RuntimeError("Causal feature name drift")
        chunks.append(features.numpy().astype(np.float64))
    if names is None:
        raise RuntimeError("No causal gate examples")
    return np.concatenate(chunks), names


def fit_ridge_gate(features: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    y_mean = y.mean(axis=0)
    standardized = (x - mean) / scale
    gram = np.einsum("ni,nj->ij", standardized, standardized, optimize=False)
    gram = gram + float(alpha) * np.eye(standardized.shape[1])
    rhs = np.einsum("ni,nk->ik", standardized, y - y_mean, optimize=False)
    coef = np.linalg.solve(gram, rhs)
    return {"feature_mean": mean, "feature_scale": scale, "target_mean": y_mean, "coef": coef, "alpha": alpha}


def predict_ridge_gate(model: dict[str, np.ndarray | float], features: np.ndarray) -> np.ndarray:
    standardized = (features - model["feature_mean"]) / model["feature_scale"]  # type: ignore[operator]
    return np.einsum(
        "ni,ik->nk", standardized, model["coef"], optimize=False  # type: ignore[arg-type]
    ) + model["target_mean"]  # type: ignore[operator]


def gate_calibration_rows(predicted: np.ndarray, actual: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for column, depth in enumerate((1, 2, 4)):
        pred = predicted[:, column]
        truth = actual[:, column]
        corr = float(np.corrcoef(pred, truth)[0, 1]) if np.std(pred) > 0 and np.std(truth) > 0 else float("nan")
        rows.append(
            {
                "depth": depth,
                "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                "pearson_correlation": corr,
                "predicted_mean_gain": float(pred.mean()),
                "actual_mean_gain": float(truth.mean()),
            }
        )
    return rows


def select_gate_budget(
    predicted_calibration_gains: np.ndarray,
    calibration_losses: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[int, list[dict[str, Any]], dict[int, np.ndarray]]:
    rows = []
    allocations: dict[int, np.ndarray] = {}
    n = len(calibration_losses)
    for mean_budget in [int(value) for value in cfg["gate_candidate_mean_budgets"]]:
        allocation = predicted_gain_exact_budget_allocation(
            predicted_calibration_gains, mean_budget * n, DEPTHS
        )
        allocations[mean_budget] = allocation
        adaptive_loss = gather_depths(calibration_losses, allocation, DEPTHS)
        uniform_loss = calibration_losses[:, DEPTHS.index(mean_budget)]
        row = {
            "mean_budget": mean_budget,
            "total_calls": total_block_calls(allocation, DEPTHS),
            "adaptive_mean_loss": float(adaptive_loss.mean()),
            "uniform_mean_loss": float(uniform_loss.mean()),
            "depth0_mean_loss": float(calibration_losses[:, 0].mean()),
            "adaptive_advantage_vs_uniform": float((uniform_loss - adaptive_loss).mean()),
            "adaptive_gain_vs_depth0": float((calibration_losses[:, 0] - adaptive_loss).mean()),
            "qualifies": bool(adaptive_loss.mean() < calibration_losses[:, 0].mean()),
        }
        rows.append(row)
    qualifying = [row for row in rows if row["qualifies"]]
    if qualifying:
        best = max(
            qualifying,
            key=lambda row: (float(row["adaptive_advantage_vs_uniform"]), -int(row["mean_budget"])),
        )
    else:
        best = min(rows, key=lambda row: int(row["mean_budget"]))
    return int(best["mean_budget"]), rows, allocations


def run_selected_inference(
    model: SharedResidualRefiner,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    allocation: np.ndarray,
    device: torch.device,
    batch_size: int,
    repeats: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected_calls = int(np.sum(allocation))
    measured = []
    first_output = None
    first_calls = None
    model.eval()
    with torch.inference_mode():
        for repeat in range(repeats):
            synchronize(device)
            start_time = time.perf_counter()
            chunks = []
            calls = 0
            invocations = 0
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                history, action, base, _ = _batch_tensors(arrays, batch_indices, device)
                selected = torch.from_numpy(allocation[start : start + len(batch_indices)]).to(device)
                output, stats = model.forward_selected(
                    history, action, base, selected, return_stats=True
                )
                calls += stats.processed_rows
                invocations += stats.block_invocations
                chunks.append(output.detach().cpu().numpy())
            synchronize(device)
            elapsed = time.perf_counter() - start_time
            if calls != expected_calls:
                raise RuntimeError(f"Selected inference calls={calls}, expected={expected_calls}")
            measured.append(elapsed)
            if repeat == 0:
                first_output = np.concatenate(chunks).astype(np.float32)
                first_calls = calls
                first_invocations = invocations
    assert first_output is not None and first_calls is not None
    return first_output, {
        "total_block_calls": first_calls,
        "batched_block_invocations": first_invocations,
        "latency_seconds_repeats": measured,
        "latency_seconds_median": float(np.median(measured)),
        "samples": len(indices),
        "mean_calls": expected_calls / len(indices),
        "device": str(device),
    }


def dense_linear_flops_per_call(model: SharedResidualRefiner) -> int:
    return int(
        sum(2 * module.in_features * module.out_features for module in model.block if isinstance(module, nn.Linear))
    )


def fit_physical_regimes(arrays: dict[str, np.ndarray], calibration_indices: np.ndarray) -> dict[str, float]:
    noninteraction = calibration_indices[~arrays["interaction"][calibration_indices].astype(bool)]
    if len(noninteraction) == 0:
        raise RuntimeError("No calibration noninteraction transitions for post-hoc regime thresholds")
    return {
        "effector_motion_threshold_m": float(np.percentile(arrays["effector_disp"][noninteraction], 10.0)),
        "block_motion_threshold_m": max(0.001, float(np.percentile(arrays["block_disp"][noninteraction], 95.0))),
    }


def apply_physical_regimes(
    arrays: dict[str, np.ndarray], indices: np.ndarray, thresholds: dict[str, float]
) -> np.ndarray:
    interaction = arrays["interaction"][indices].astype(bool)
    impact = arrays["impact"][indices].astype(bool)
    moving = (
        arrays["effector_disp"][indices] > thresholds["effector_motion_threshold_m"]
    ) | (arrays["block_disp"][indices] > thresholds["block_motion_threshold_m"])
    return np.select(
        [impact, interaction, (~interaction) & moving],
        ["impact", "interaction", "transport_free"],
        default="static",
    )


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(not applicable)"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    decision: dict[str, Any],
    depth_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    figures: dict[str, str],
    cfg: dict[str, Any],
) -> None:
    verdict = decision["verdict"]
    best_depth = min(depth_rows[1:], key=lambda row: float(row["raw_mean_loss"])) if len(depth_rows) > 1 else depth_rows[0]
    oracle_rows = [row for row in budget_rows if row["strategy"] == "oracle"]
    best_oracle = max(oracle_rows, key=lambda row: float(row.get("mean_advantage_vs_uniform", -np.inf)))
    gate_section = markdown_table(
        gate_rows,
        ["comparison", "mean_benefit", "ci_low", "ci_high", "passed"],
    )
    smoke_caveat = (
        "\n## Engineering smoke caveat\n\nThis six-episode run validates the pipeline only; "
        "its one-episode calibration/test summaries do not support scientific claims.\n"
        if cfg["mode"] == "smoke"
        else ""
    )
    report = f"""# Direct LeWM Adaptive Compute Pilot

## Decision

**`{verdict}`** — {decision['reason']}

The primary frozen-LeWM depth-0 raw MSE was `{depth_rows[0]['raw_mean_loss']:.6g}`.
The best fixed depth was `{best_depth['depth']}` with raw MSE
`{best_depth['raw_mean_loss']:.6g}` and same-transition gain
`{best_depth['mean_gain']:.6g}` (episode-clustered 95% CI
`[{best_depth['gain_ci_low']:.6g}, {best_depth['gain_ci_high']:.6g}]`). The
largest diagnostic oracle advantage over matched uniform compute was
`{best_oracle['mean_advantage_vs_uniform']:.6g}` at mean budget
`{best_oracle['mean_budget']}` calls (CI
`[{best_oracle['advantage_vs_uniform_ci_low']:.6g}, {best_oracle['advantage_vs_uniform_ci_high']:.6g}]`).

## Validity

- Depth 0 is bitwise identical to the direct released LeWM prediction cache.
- Released checkpoint loading was strict; all 18,034,628 base parameters were frozen.
- Train/calibration/test episodes are pairwise disjoint and prior Cube diagnostic episodes 0--29 were excluded.
- Refiner and gate inputs contain only latent/action history, the current prediction, and iteration information.
- Whitening, primary-seed choice, physical-regime thresholds, and any gate budget were calibration-only.
- Test metrics were computed only after calibration screening and gate/budget freezing.
- Oracle allocations use target error and are diagnostic upper bounds, never gate features.
- Latent MSE is not interpreted as proof of physical grounding.

The released checkpoint has no episode-level pretraining manifest. These test
episodes are fresh to the contact diagnostics and refiner, but cannot be proven
unseen during original LeWM pretraining.
{smoke_caveat}

## Error versus depth

{markdown_table(depth_rows, ['depth', 'raw_mean_loss', 'whitened_mean_loss', 'mean_gain', 'gain_ci_low', 'gain_ci_high', 'mean_robust_relative_gain'])}

## Matched-call allocation

{markdown_table(budget_rows, ['mean_budget', 'strategy', 'total_calls', 'raw_mean_loss', 'mean_gain_vs_depth0', 'mean_advantage_vs_uniform', 'advantage_vs_uniform_ci_low', 'advantage_vs_uniform_ci_high'])}

## Learned causal gate

{gate_section}

## Post-hoc physical regimes

Contact, impact, transport-free motion, and static labels below are used only
for interpretation; they were unavailable to both the refiner and gate.

{markdown_table(regime_rows, ['regime', 'n', 'mean_selected_depth', 'mean_gain_vs_depth0'])}

## Compute and provenance

- Trainable shared-refiner parameters: `{decision['compute']['trainable_parameters']}`.
- Estimated dense-linear FLOPs per refinement-block call: `{decision['compute']['flops_per_call']}`.
- Selected-policy calls: `{decision['compute']['selected_total_calls']}`.
- Selected-policy estimated dense-linear FLOPs: `{decision['compute']['selected_estimated_flops']}`.
- Selected-policy median measured latency: `{decision['compute'].get('selected_latency_seconds_median', float('nan')):.6g}` seconds on `{decision['compute']['device']}`.
- Checkpoint SHA-256: `{provenance['model']['weights_sha256']}`.
- Source HDF5: `{provenance['source']['h5_path']}`.
- Split/cache manifest: `cache/split_manifest.json`.
- Configuration: `{cfg['mode']}_config.json`.

## Figures

![Error versus depth](figures/01_error_vs_depth.png)

![Compute benefit distribution](figures/02_compute_benefit_distribution.png)

![Matched budget curves](figures/03_budget_curves.png)

![Allocation by physical regime](figures/04_allocation_by_regime.png)

## Limitations and next gate

This is one released Cube checkpoint and one small shared-refiner family. A
negative result rules out this pilot architecture/objective, not all iterative
LeWM computation. A positive oracle result without a learned-gate result calls
for better causal uncertainty/convergence features, not target-derived regime
features. A learned-gate success should next be replicated across checkpoint
seeds and another physical domain before control or novelty claims.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    cache_dir = output_dir / "cache"
    checkpoint_dir = output_dir / "checkpoints"
    for directory in (metrics_dir, figures_dir, cache_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    split_episodes = configured_split(cfg)
    cache_path = cache_dir / "fresh_cube_inputs.npz"
    manifest_path = cache_dir / "split_manifest.json"
    device = choose_device(args.device)
    print_step(f"mode={cfg['mode']} device={device} output={output_dir}")
    if args.force_extract or not cache_path.exists():
        print_step("strictly loading released LeWM and extracting frozen episode split")
        extract_fresh_cube_cache(
            source_h5=args.source_h5,
            config_path=args.model_config,
            weights_path=args.model_weights,
            output_npz=cache_path,
            split_manifest_path=manifest_path,
            split_episodes=split_episodes,
            device=device,
            encode_batch_size=int(cfg["encode_batch_size"]),
            predict_batch_size=int(cfg["predict_batch_size"]),
        )
    arrays = load_cache(cache_path, split_episodes)
    if args.extract_only:
        print_step(f"extraction complete: {cache_path}")
        return
    provenance = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_indices = {
        name: np.flatnonzero(arrays["split"] == code) for name, code in SPLIT_CODES.items()
    }
    train_indices = split_indices["train"]
    calibration_indices = split_indices["calibration"]
    test_indices = split_indices["test"]

    all_training_rows: list[dict[str, Any]] = []
    seed_models: dict[int, SharedResidualRefiner] = {}
    seed_summaries = []
    for seed in [int(value) for value in cfg["training_seeds"]]:
        print_step(f"training/loading refiner seed {seed}")
        model, summary, history_rows = train_one_seed(
            seed=seed,
            cfg=cfg,
            arrays=arrays,
            train_indices=train_indices,
            calibration_indices=calibration_indices,
            device=device,
            checkpoint_dir=checkpoint_dir,
            force=args.force_train,
        )
        seed_models[seed] = model
        seed_summaries.append(summary)
        all_training_rows.extend(history_rows)
    if all_training_rows:
        write_rows(metrics_dir / "training_history.csv", all_training_rows)
    write_json(metrics_dir / "seed_training_summary.json", seed_summaries)

    primary_summary = min(
        seed_summaries,
        key=lambda row: (
            float(row["best_calibration_objective"]),
            [int(v) for v in cfg["training_seeds"]].index(int(row["seed"])),
        ),
    )
    primary_seed = int(primary_summary["seed"])
    primary = seed_models[primary_seed]
    print_step(f"primary seed selected on calibration only: {primary_seed}")

    whitening = fit_whitening(arrays["target"][calibration_indices])
    np.savez_compressed(
        cache_dir / "calibration_whitening.npz",
        mean=whitening.mean,
        matrix=whitening.matrix,
        eigenvalues=whitening.eigenvalues,
        used_eigenvalues=whitening.used_eigenvalues,
        floor=np.asarray(whitening.floor),
    )

    # Calibration Stage A screen happens before any test prediction is scored.
    calibration_exits = evaluate_exits(
        primary, arrays, calibration_indices, device, int(cfg["predict_batch_size"])
    )
    calibration_targets = arrays["target"][calibration_indices]
    calibration_raw = latent_mse_by_depth(calibration_targets, calibration_exits, DEPTHS)
    calibration_white = latent_mse_by_depth(
        calibration_targets, calibration_exits, DEPTHS, whitening=whitening
    )
    calibration_relative = robust_relative_gain(calibration_raw, calibration_raw[:, 0], DEPTHS)
    calibration_stage_a, _, calibration_budget_rows, _ = stage_a_analysis(
        raw_losses=calibration_raw,
        whitened_losses=calibration_white,
        relative_gain=calibration_relative,
        episode_ids=arrays["episode_id"][calibration_indices],
        cfg=cfg,
        label="calibration",
    )
    write_json(metrics_dir / "calibration_stage_a_screen.json", calibration_stage_a)
    write_rows(metrics_dir / "calibration_budget_curves.csv", calibration_budget_rows)

    gate_screened_in = bool(
        calibration_stage_a["compute_response_passed"]
        and calibration_stage_a["oracle_headroom_passed"]
    )
    gate_model = None
    gate_feature_names: tuple[str, ...] = ()
    selected_gate_budget = None
    gate_calibration_table: list[dict[str, Any]] = []
    budget_selection_rows: list[dict[str, Any]] = []
    if gate_screened_in:
        print_step("calibration shows response and oracle headroom; fitting causal ridge gate")
        train_exits = evaluate_exits(
            primary, arrays, train_indices, device, int(cfg["predict_batch_size"])
        )
        train_raw = latent_mse_by_depth(arrays["target"][train_indices], train_exits, DEPTHS)
        train_gains = train_raw[:, [0]] - train_raw[:, 1:]
        train_features, gate_feature_names = causal_features_numpy(arrays, train_indices)
        calibration_features, calibration_names = causal_features_numpy(arrays, calibration_indices)
        if gate_feature_names != calibration_names:
            raise RuntimeError("Train/calibration causal feature mismatch")
        gate_model = fit_ridge_gate(
            train_features, train_gains, float(cfg["gate_ridge_alpha"])
        )
        calibration_predicted_gains = predict_ridge_gate(gate_model, calibration_features)
        calibration_actual_gains = calibration_raw[:, [0]] - calibration_raw[:, 1:]
        gate_calibration_table = gate_calibration_rows(
            calibration_predicted_gains, calibration_actual_gains
        )
        selected_gate_budget, budget_selection_rows, _ = select_gate_budget(
            calibration_predicted_gains, calibration_raw, cfg
        )
        np.savez_compressed(
            checkpoint_dir / "causal_ridge_gate.npz",
            feature_names=np.asarray(gate_feature_names),
            feature_mean=gate_model["feature_mean"],
            feature_scale=gate_model["feature_scale"],
            target_mean=gate_model["target_mean"],
            coef=gate_model["coef"],
            alpha=np.asarray(gate_model["alpha"]),
            selected_mean_budget=np.asarray(selected_gate_budget),
        )
        write_rows(metrics_dir / "gate_calibration.csv", gate_calibration_table)
        write_rows(metrics_dir / "gate_budget_selection.csv", budget_selection_rows)
    else:
        print_step("calibration Stage A screen failed; learned gate will not be fit")

    # Final test evaluation begins only after every model/gate/budget choice is frozen.
    print_step("running single frozen test evaluation")
    test_exits = evaluate_exits(primary, arrays, test_indices, device, int(cfg["predict_batch_size"]))
    test_targets = arrays["target"][test_indices]
    test_episodes = arrays["episode_id"][test_indices]
    test_raw = latent_mse_by_depth(test_targets, test_exits, DEPTHS)
    test_white = latent_mse_by_depth(test_targets, test_exits, DEPTHS, whitening=whitening)
    test_relative = robust_relative_gain(test_raw, calibration_raw[:, 0], DEPTHS)
    test_stage_a, depth_rows, budget_rows, test_allocations = stage_a_analysis(
        raw_losses=test_raw,
        whitened_losses=test_white,
        relative_gain=test_relative,
        episode_ids=test_episodes,
        cfg=cfg,
        label="test",
    )

    # Secondary seed curves are fixed-seed sensitivity only; they never select the primary.
    seed_depth_rows = []
    for seed, model in seed_models.items():
        exits = test_exits if seed == primary_seed else evaluate_exits(
            model, arrays, test_indices, device, int(cfg["predict_batch_size"])
        )
        losses = latent_mse_by_depth(test_targets, exits, DEPTHS)
        for column, depth in enumerate(DEPTHS):
            seed_depth_rows.append(
                {"seed": seed, "primary": seed == primary_seed, "depth": depth, "raw_mean_loss": float(losses[:, column].mean())}
            )
    write_rows(metrics_dir / "seed_depth_sensitivity.csv", seed_depth_rows)

    gate_comparison_rows: list[dict[str, Any]] = []
    selected_allocation = None
    selected_outputs = None
    selected_latency: dict[str, Any] = {}
    uniform_latency: dict[str, Any] = {}
    if gate_screened_in and gate_model is not None and selected_gate_budget is not None:
        test_features, test_feature_names = causal_features_numpy(arrays, test_indices)
        if test_feature_names != gate_feature_names:
            raise RuntimeError("Frozen gate feature mismatch on test")
        test_predicted_gains = predict_ridge_gate(gate_model, test_features)
        selected_allocation = predicted_gain_exact_budget_allocation(
            test_predicted_gains, selected_gate_budget * len(test_indices), DEPTHS
        )
        selected_outputs, selected_latency = run_selected_inference(
            primary,
            arrays,
            test_indices,
            selected_allocation,
            device,
            int(cfg["predict_batch_size"]),
        )
        gathered = gather_depths(test_exits, selected_allocation, DEPTHS)
        max_abs = float(np.max(np.abs(selected_outputs - gathered)))
        if max_abs > 2e-5:
            raise RuntimeError(f"Pruned adaptive output mismatch vs dense exits: {max_abs}")
        selected_loss = np.mean((selected_outputs.astype(np.float64) - test_targets) ** 2, axis=1)
        uniform_allocation = np.full(len(test_indices), selected_gate_budget, dtype=np.int64)
        uniform_outputs, uniform_latency = run_selected_inference(
            primary,
            arrays,
            test_indices,
            uniform_allocation,
            device,
            int(cfg["predict_batch_size"]),
        )
        uniform_loss = np.mean((uniform_outputs.astype(np.float64) - test_targets) ** 2, axis=1)
        random_allocation = random_exact_budget_allocation(
            len(test_indices), DEPTHS, selected_gate_budget * len(test_indices), np.random.default_rng(260713)
        )
        permuted = permute_allocation(selected_allocation, np.random.default_rng(260714))
        oracle = test_allocations[f"oracle_b{selected_gate_budget}"]
        comparison_values = {
            "adaptive_vs_depth0": test_raw[:, 0] - selected_loss,
            "adaptive_vs_uniform": uniform_loss - selected_loss,
            "adaptive_vs_random": gather_depths(test_raw, random_allocation, DEPTHS) - selected_loss,
            "adaptive_vs_permuted": gather_depths(test_raw, permuted, DEPTHS) - selected_loss,
            "oracle_regret": selected_loss - gather_depths(test_raw, oracle, DEPTHS),
        }
        for comparison, values in comparison_values.items():
            ci = paired_cluster_bootstrap(
                values,
                test_episodes,
                n_bootstrap=int(cfg["bootstrap_samples"]),
                seed=int(cfg["bootstrap_seed"]),
            )
            passed = ci.lower > 0.0 if comparison in {"adaptive_vs_depth0", "adaptive_vs_uniform"} else None
            gate_comparison_rows.append(
                {
                    "comparison": comparison,
                    "mean_benefit": ci.estimate,
                    "ci_low": ci.lower,
                    "ci_high": ci.upper,
                    "passed": passed,
                }
            )
        selected_summary = summarize_allocation(test_raw, selected_allocation, DEPTHS)
        budget_rows.append(
            {
                "scope": "test",
                "mean_budget": selected_gate_budget,
                "mean_calls": float(selected_summary["mean_calls"]),
                "strategy": "adaptive",
                "total_calls": int(selected_summary["total_calls"]),
                "raw_mean_loss": float(selected_loss.mean()),
                "whitened_mean_loss": float(
                    np.mean(
                        np.einsum(
                            "nd,df->nf",
                            selected_outputs - test_targets,
                            whitening.matrix,
                            optimize=False,
                        )
                        ** 2
                    )
                ),
                "mean_gain_vs_depth0": float((test_raw[:, 0] - selected_loss).mean()),
                "depth_histogram": json.dumps(selected_summary["depth_histogram"], sort_keys=True),
                "diagnostic_only": False,
            }
        )

        # Calibration-frozen predicted-benefit quantiles.
        predicted_selected_gain = np.maximum(test_predicted_gains.max(axis=1), 0.0)
        quantile = pd.qcut(predicted_selected_gain, q=5, labels=False, duplicates="drop")
        quantile_rows = []
        for value in sorted(np.unique(quantile)):
            mask = np.asarray(quantile == value)
            quantile_rows.append(
                {
                    "predicted_benefit_quantile": int(value),
                    "n": int(mask.sum()),
                    "mean_predicted_gain": float(predicted_selected_gain[mask].mean()),
                    "mean_actual_gain": float((test_raw[mask, 0] - selected_loss[mask]).mean()),
                    "mean_depth": float(selected_allocation[mask].mean()),
                }
            )
        write_rows(metrics_dir / "gate_predicted_benefit_quantiles.csv", quantile_rows)

    if not test_stage_a["compute_response_passed"]:
        verdict = "no_compute_response"
        reason = "No fixed refinement depth had positive raw-MSE gain with a positive episode-clustered 95% CI."
    elif not test_stage_a["oracle_headroom_passed"]:
        verdict = "compute_helpful_but_not_heterogeneous"
        reason = "Extra LeWM refinement helped, but the exact-budget oracle did not clear the preregistered allocation-headroom gate."
    elif not gate_screened_in:
        verdict = "selective_headroom_gate_failed"
        reason = "Test oracle headroom exists, but the causal gate was not fit because the calibration-only Stage A screen failed."
    else:
        required = {row["comparison"]: row for row in gate_comparison_rows}
        gate_passed = bool(
            required.get("adaptive_vs_depth0", {}).get("passed", False)
            and required.get("adaptive_vs_uniform", {}).get("passed", False)
        )
        if gate_passed:
            verdict = "adaptive_beats_original_and_uniform"
            reason = "The frozen causal gate beat depth 0 and matched-call uniform refinement with positive clustered-CI lower bounds."
        else:
            verdict = "selective_headroom_gate_failed"
            reason = "The exact-budget oracle showed selective headroom, but the frozen causal gate failed at least one required test comparison."

    thresholds = fit_physical_regimes(arrays, calibration_indices)
    test_regimes = apply_physical_regimes(arrays, test_indices, thresholds)
    if selected_allocation is None:
        if test_stage_a["oracle_headroom_passed"]:
            oracle_rows = [row for row in budget_rows if row["strategy"] == "oracle"]
            chosen = max(oracle_rows, key=lambda row: float(row["mean_advantage_vs_uniform"]))
            figure_allocation = test_allocations[f"oracle_b{int(chosen['mean_budget'])}"]
            figure_allocation_source = "diagnostic_oracle"
        else:
            best_depth = min(depth_rows[1:], key=lambda row: float(row["raw_mean_loss"]))
            figure_allocation = np.full(len(test_indices), int(best_depth["depth"]), dtype=np.int64)
            figure_allocation_source = "best_fixed_depth"
    else:
        figure_allocation = selected_allocation
        figure_allocation_source = "learned_adaptive"
    regime_rows = []
    figure_selected_loss = gather_depths(test_raw, figure_allocation, DEPTHS)
    for regime in ("impact", "interaction", "transport_free", "static"):
        mask = test_regimes == regime
        if not np.any(mask):
            continue
        regime_rows.append(
            {
                "regime": regime,
                "n": int(mask.sum()),
                "n_episodes": int(len(np.unique(test_episodes[mask]))),
                "mean_selected_depth": float(figure_allocation[mask].mean()),
                "mean_gain_vs_depth0": float((test_raw[mask, 0] - figure_selected_loss[mask]).mean()),
                "allocation_source": figure_allocation_source,
            }
        )

    write_rows(metrics_dir / "depth_curves.csv", depth_rows)
    write_rows(metrics_dir / "budget_curves.csv", budget_rows)
    write_rows(metrics_dir / "gate_comparisons.csv", gate_comparison_rows)
    write_rows(metrics_dir / "allocation_by_regime.csv", regime_rows)
    write_json(metrics_dir / "posthoc_regime_thresholds.json", thresholds)
    np.savez_compressed(
        metrics_dir / "test_metrics.npz",
        raw_losses=test_raw,
        whitened_losses=test_white,
        robust_relative_gain=test_relative,
        episode_id=test_episodes,
        model_step=arrays["model_step"][test_indices],
        selected_allocation=figure_allocation,
        physical_regime=test_regimes,
    )

    figures = write_all_figures(
        depth_rows=depth_rows,
        budget_rows=budget_rows,
        raw_losses=test_raw,
        depths=DEPTHS,
        allocation=figure_allocation,
        regimes=test_regimes,
        output_dir=figures_dir,
    )

    flops_per_call = dense_linear_flops_per_call(primary)
    selected_total_calls = int(np.sum(figure_allocation))
    compute = {
        "device": str(device),
        "trainable_parameters": int(sum(parameter.numel() for parameter in primary.parameters())),
        "flops_per_call": flops_per_call,
        "selected_total_calls": selected_total_calls,
        "selected_estimated_flops": int(flops_per_call * selected_total_calls),
        "allocation_source": figure_allocation_source,
        "selected_latency_seconds_median": selected_latency.get("latency_seconds_median", float("nan")),
        "selected_latency": selected_latency,
        "uniform_latency": uniform_latency,
    }
    decision = {
        "verdict": verdict,
        "reason": reason,
        "mode": cfg["mode"],
        "primary_seed": primary_seed,
        "calibration_stage_a": calibration_stage_a,
        "test_stage_a": test_stage_a,
        "gate_screened_in": gate_screened_in,
        "gate_screen_failure_reason": None if gate_screened_in else "gate_not_fit_calibration_screen",
        "selected_gate_mean_budget": selected_gate_budget,
        "gate_comparisons": gate_comparison_rows,
        "validity": {
            "depth0_bitwise_equal": True,
            "strict_checkpoint_load": True,
            "base_frozen": bool(provenance["model"]["all_parameters_frozen"]),
            "episode_split_isolated": True,
            "prior_episodes_excluded": True,
            "causal_gate_features_only": True,
            "calibration_fit_whitening": True,
            "test_evaluated_after_freeze": True,
            "fresh_relative_to_prior_diagnostics_not_base_pretraining": True,
        },
        "compute": compute,
        "artifacts": {
            "metrics": str(metrics_dir),
            "figures": figures,
            "cache_manifest": str(manifest_path),
        },
    }
    write_json(output_dir / "decision.json", decision)
    write_report(
        output_dir / "REPORT.md",
        decision=decision,
        depth_rows=depth_rows,
        budget_rows=budget_rows,
        gate_rows=gate_comparison_rows,
        regime_rows=regime_rows,
        provenance=provenance,
        figures=figures,
        cfg=cfg,
    )
    print_step(f"complete: {verdict}")
    print(json.dumps(_jsonable(decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
