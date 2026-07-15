#!/usr/bin/env python3
"""Run isolated adaptive-compute discovery and the sealed calibration judge."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import critics
import data_isolation
from extract_isolated import extract, load_isolated
import models
import policy

V1_CHECKPOINT = REPO / "runs/lewm_adaptive_compute_v1/checkpoints/refiner_seed_260713.pt"
V1_REFINER_SOURCE = REPO / "runs/lewm_adaptive_compute_v1/refiner.py"


def log(message: str) -> None:
    print(f"[adaptive-discovery] {message}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [jsonable(dict(row)) for row in rows]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fit_whitening(target: np.ndarray) -> dict[str, np.ndarray | float]:
    values = np.asarray(target, dtype=np.float64)
    mean = values.mean(0)
    centered = values - mean
    # The local Accelerate/BLAS path can emit nonfinite warnings for this
    # ill-conditioned covariance; retain the finite einsum path used in V3.
    covariance = np.einsum("ni,nj->ij", centered, centered, optimize=False) / max(len(values) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max()) * 1e-6, 1e-12)
    matrix = np.einsum("ik,k,jk->ij", eigenvectors, 1.0 / np.sqrt(np.maximum(eigenvalues, floor)), eigenvectors, optimize=False)
    return {"mean": mean, "matrix": matrix, "eigenvalues": eigenvalues, "floor": floor}


def whitened_losses(target: np.ndarray, exits: np.ndarray, whitening: Mapping[str, Any]) -> np.ndarray:
    difference = np.asarray(exits, dtype=np.float64) - np.asarray(target, dtype=np.float64)[:, None, :]
    transformed = np.einsum("nkd,df->nkf", difference, whitening["matrix"], optimize=False)
    result = np.square(transformed).mean(2)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite whitened loss")
    return result


BASE_PREDICT_FLOPS = 70_529_190
V1_CALL_FLOPS = 669_184


def solver_flops(family: str, selected_calls: np.ndarray) -> int:
    calls = np.asarray(selected_calls, dtype=np.int64)
    if family in ("v3_repair", "verifier_v1"):
        return int(calls.sum()) * V1_CALL_FLOPS
    later = 2 * (843 * 128 + 128 * 192)
    return int(len(calls)) * V1_CALL_FLOPS + int((calls - 1).sum()) * later


def critic_flops_per_model(feature_dim: int, hidden_dims: Sequence[int]) -> int:
    dimensions = [int(feature_dim), *(int(v) for v in hidden_dims), 1]
    return int(sum(2 * left * right for left, right in zip(dimensions[:-1], dimensions[1:])))


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    result = torch.device(name)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return result


def load_v1_refiner(device: torch.device) -> nn.Module:
    if data_isolation.sha256_file(V1_CHECKPOINT) != json.loads((ROOT / "config.json").read_text())["v1_refiner_sha256"]:
        raise RuntimeError("V1 refiner checkpoint hash mismatch")
    spec = importlib.util.spec_from_file_location("discovery_v1_refiner", V1_REFINER_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load V1 refiner source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    refiner = module.SharedResidualRefiner()
    payload = torch.load(V1_CHECKPOINT, map_location="cpu", weights_only=True)
    result = refiner.load_state_dict(payload["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys or payload.get("seed") != 260713:
        raise RuntimeError("V1 strict load failed")
    return refiner.to(device).eval().requires_grad_(False)


def load_arrays(role: str) -> dict[str, np.ndarray]:
    name = "v3_train_only.npz" if role == "train" else "v3_calibration_only.npz"
    arrays, _ = load_isolated(ROOT / "cache" / name, role)
    return arrays


def indices_for_episodes(arrays: Mapping[str, np.ndarray], episodes: Sequence[int]) -> np.ndarray:
    result = np.flatnonzero(np.isin(arrays["episode_id"], np.asarray(episodes, dtype=np.int64)))
    if not len(result):
        raise RuntimeError("episode selection produced no rows")
    return result


def batches(indices: np.ndarray, size: int):
    for start in range(0, len(indices), int(size)):
        yield indices[start : start + int(size)]


def tensors(arrays: Mapping[str, np.ndarray], index: np.ndarray, device: torch.device, target: bool = True):
    values = [torch.as_tensor(np.ascontiguousarray(arrays[name][index]), device=device) for name in ("history", "action", "base_pred")]
    if target:
        values.append(torch.as_tensor(np.ascontiguousarray(arrays["target"][index]), device=device))
    return values


def build_solver(family: str, seed: int, cfg: Mapping[str, Any], v1: nn.Module, device: torch.device) -> nn.Module:
    torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
    if family == "stagewise":
        model = models.StagewiseResidualCascade(v1, later_stages=3, hidden_dim=int(cfg["adapter_hidden_dim"]))
    elif family == "v3_repair":
        model = models.V3RepairControl(v1)
    elif family == "contractive":
        model = models.ContractiveSharedResidualCascade(v1, max_depth=4, hidden_dim=int(cfg["contractive_hidden_dim"]), relaxation=0.5)
    else:
        raise ValueError(f"unknown solver family {family}")
    return model.to(device)


def output_depths(family: str, accepted_depth: int = 4) -> tuple[int, ...]:
    if family == "verifier_v1":
        return (0, 1, 2, 4)
    return tuple(range(0, accepted_depth + 1))


def forward_solver(model: nn.Module, family: str, h: torch.Tensor, a: torch.Tensor, z: torch.Tensor, maximum: int = 4):
    if family == "verifier_v1":
        return model(h, a, z, depths=(0, 1, 2, 4))
    return model(h, a, z, max_depth=maximum)


@torch.no_grad()
def mean_exit_losses(model: nn.Module, family: str, arrays: Mapping[str, np.ndarray], index: np.ndarray,
                     device: torch.device, batch_size: int, depths: Sequence[int]) -> dict[int, float]:
    sums = {int(depth): 0.0 for depth in depths}; count = 0
    model.eval()
    for part in batches(index, batch_size):
        h, a, z, y = tensors(arrays, part, device)
        outputs = forward_solver(model, family, h, a, z, max(depths))
        for depth in depths:
            sums[int(depth)] += float((outputs[int(depth)] - y).square().mean(1).sum().cpu())
        count += len(part)
    return {depth: value / count for depth, value in sums.items()}


def episode_caps(episode_ids: np.ndarray, epoch: int, seed: int, maximum: int) -> np.ndarray:
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(int(seed) + int(epoch) * 100003)
    mapping = {int(ep): int(cap) for ep, cap in zip(unique, rng.integers(2, maximum + 1, size=len(unique)), strict=True)}
    return np.asarray([mapping[int(ep)] for ep in episode_ids], dtype=np.int64)


def train_joint_later(model: nn.Module, family: str, arrays: Mapping[str, np.ndarray], fit_idx: np.ndarray,
                      val_idx: np.ndarray, seed: int, cfg: Mapping[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    params = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg["solver_lr"]), weight_decay=float(cfg["solver_weight_decay"]))
    depths = tuple(d for d in output_depths(family) if d >= 2)
    best_state = copy.deepcopy(model.state_dict())
    initial = mean_exit_losses(model, family, arrays, val_idx, device, int(cfg["predict_batch_size"]), depths)
    best_objective = float(np.mean(list(initial.values())))
    best_epoch = 0; stale = 0; history = [{"epoch": 0, "validation_objective": best_objective, **{f"val_d{k}":v for k,v in initial.items()}}]
    rng = np.random.default_rng(seed)
    fit_episodes = arrays["episode_id"][fit_idx]
    for epoch in range(1, int(cfg["solver_max_epochs"]) + 1):
        model.train(); order = rng.permutation(fit_idx)
        cap_for_row = episode_caps(fit_episodes, epoch, seed, 4)
        cap_map = {int(row): int(cap) for row, cap in zip(fit_idx, cap_for_row, strict=True)}
        train_sum = 0.0; seen = 0
        for part in batches(order, int(cfg["train_batch_size"])):
            h, a, z, y = tensors(arrays, part, device)
            outputs = forward_solver(model, family, h, a, z, 4)
            caps = torch.as_tensor([cap_map[int(row)] for row in part], device=device)
            terms = []
            previous_loss = (outputs[1] - y).square().mean(1).detach()
            for depth in depths:
                row_loss = (outputs[depth] - y).square().mean(1)
                active = caps >= depth
                if bool(active.any()):
                    terms.append(row_loss[active].mean())
                    # Earlier-exit loss is a detached constant: the V3 sign bug is impossible.
                    terms.append(0.1 * torch.relu(row_loss[active] - previous_loss[active].detach()).mean())
                previous_loss = row_loss.detach()
            loss = torch.stack(terms).sum() / max(len(depths), 1)
            if family == "contractive":
                stable = model.stability_losses(outputs)
                loss = loss + 0.1 * stable["shortcut_consistency"] + 0.1 * stable["contraction_violation"]
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(params, float(cfg["gradient_clip"])); optimizer.step()
            train_sum += float(loss.detach()) * len(part); seen += len(part)
        values = mean_exit_losses(model, family, arrays, val_idx, device, int(cfg["predict_batch_size"]), depths)
        objective = float(np.mean(list(values.values())))
        history.append({"epoch": epoch, "train_objective": train_sum / seen, "validation_objective": objective, **{f"val_d{k}":v for k,v in values.items()}})
        if objective < best_objective - 1e-8:
            best_objective = objective; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= int(cfg["solver_patience"]):
            break
    model.load_state_dict(best_state, strict=True); model.eval().requires_grad_(False)
    return model, {"best_epoch": best_epoch, "epoch0_objective": history[0]["validation_objective"], "best_validation_objective": best_objective, "history": history}


def train_stagewise(model: nn.Module, arrays: Mapping[str, np.ndarray], fit_idx: np.ndarray, val_idx: np.ndarray,
                    seed: int, cfg: Mapping[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any], int]:
    stage_records = []; accepted_depth = 1
    rng = np.random.default_rng(seed)
    val_episodes = arrays["episode_id"][val_idx]
    for depth in (2, 3, 4):
        model.set_trainable_stage(depth); model.train()
        epoch0_state = copy.deepcopy(model.state_dict())
        initial = mean_exit_losses(model, "stagewise", arrays, val_idx, device, int(cfg["predict_batch_size"]), (depth - 1, depth))
        best_state = epoch0_state; best_loss = initial[depth]; best_epoch = 0; stale = 0; history = []
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=float(cfg["solver_lr"]), weight_decay=float(cfg["solver_weight_decay"]))
        for epoch in range(1, int(cfg["solver_max_epochs"]) + 1):
            order = rng.permutation(fit_idx); train_sum = 0.0
            for part in batches(order, int(cfg["train_batch_size"])):
                h, a, z, y = tensors(arrays, part, device)
                outputs = model(h, a, z, max_depth=depth)
                loss = (outputs[depth] - y).square().mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(params, float(cfg["gradient_clip"])); optimizer.step()
                train_sum += float(loss.detach()) * len(part)
            current = mean_exit_losses(model, "stagewise", arrays, val_idx, device, int(cfg["predict_batch_size"]), (depth,))[depth]
            history.append({"epoch": epoch, "train_mse": train_sum / len(fit_idx), "validation_mse": current})
            if current < best_loss - 1e-8:
                best_loss = current; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict()); stale = 0
            else:
                stale += 1
            if stale >= int(cfg["solver_patience"]): break
        model.load_state_dict(best_state, strict=True)
        dense, _, losses = dense_outputs(model, "stagewise", arrays, val_idx, device, int(cfg["predict_batch_size"]), tuple(range(0, depth + 1)), need_features=False)
        benefit = losses[:, depth - 1] - losses[:, depth]
        ci = policy.clustered_bootstrap_ci(benefit, val_episodes, samples=int(cfg["bootstrap_samples_internal"]), seed=int(cfg["bootstrap_seed"]) + depth)
        accepted = bool(ci["ci_low"] > 0)
        stage_records.append({"depth": depth, "best_epoch": best_epoch, "epoch0_validation_mse": initial[depth], "best_validation_mse": best_loss, "improvement": ci, "accepted": accepted, "history": history})
        if not accepted:
            model.load_state_dict(epoch0_state, strict=True)
            break
        accepted_depth = depth
    model.set_trainable_stage(None); model.eval().requires_grad_(False)
    return model, {"stages": stage_records}, accepted_depth


@torch.no_grad()
def dense_outputs(model: nn.Module, family: str, arrays: Mapping[str, np.ndarray], index: np.ndarray,
                  device: torch.device, batch_size: int, depths: Sequence[int], need_features: bool = True):
    depth_values = tuple(int(v) for v in depths)
    exits = np.empty((len(index), len(depth_values), 192), dtype=np.float32)
    features: list[list[np.ndarray]] = [[] for _ in range(max(len(depth_values) - 2, 0))]
    cursor = 0; model.eval()
    for part in batches(index, batch_size):
        h, a, z, y = tensors(arrays, part, device)
        output = forward_solver(model, family, h, a, z, max(depth_values))
        stacked = torch.stack([output[d] for d in depth_values], 1)
        n = len(part); exits[cursor:cursor+n] = stacked.cpu().numpy()
        if need_features:
            # Policies always begin at call one; depth zero is not a critic exit.
            supported = depth_values[1:]
            for stage, current_depth in enumerate(supported[:-1]):
                previous_depth = depth_values[depth_values.index(current_depth) - 1]
                value = models.build_causal_features(h, a, output[current_depth], output[current_depth] - output[previous_depth])
                features[stage].append(value.cpu().numpy())
        cursor += n
    target = np.asarray(arrays["target"][index], dtype=np.float32)
    losses = np.square(exits - target[:, None, :]).mean(2)
    feature_arrays = [np.concatenate(chunks) for chunks in features] if need_features else []
    return exits, feature_arrays, losses


def anchor_audit(model: nn.Module, family: str, v1: nn.Module, arrays: Mapping[str, np.ndarray], index: np.ndarray,
                 device: torch.device, batch_size: int) -> dict[str, Any]:
    for part in batches(index[: min(len(index), batch_size)], batch_size):
        h, a, z = tensors(arrays, part, device, target=False)
        expected = v1(h, a, z, depths=(0, 1))
        actual = forward_solver(model, family, h, a, z, 1)
        return {"d0_identity": actual[0] is z, "d0_bitwise": bool(torch.equal(actual[0], expected[0])), "d1_bitwise": bool(torch.equal(actual[1], expected[1])), "passed": bool(actual[0] is z and torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1]))}
    raise RuntimeError("empty anchor audit")


def fit_critics_for_candidate(model: nn.Module, family: str, depths: Sequence[int], arrays: Mapping[str, np.ndarray],
                              fit_idx: np.ndarray, val_idx: np.ndarray, cfg: Mapping[str, Any], device: torch.device,
                              *, feature_tail: int | None = None, hidden_dims: Sequence[int] | None = None):
    for parameter in model.parameters():
        parameter.grad = None
    solver_hash_before = state_hash(model)
    fit_exits, fit_features, fit_losses = dense_outputs(model, family, arrays, fit_idx, device, int(cfg["predict_batch_size"]), depths)
    val_exits, val_features, val_losses = dense_outputs(model, family, arrays, val_idx, device, int(cfg["predict_batch_size"]), depths)
    if feature_tail is not None:
        fit_features = [value[:, -int(feature_tail):] for value in fit_features]
        val_features = [value[:, -int(feature_tail):] for value in val_features]
    hidden_dims = tuple(hidden_dims or cfg["critic_hidden_dims"])
    fit_episodes = arrays["episode_id"][fit_idx]; val_episodes = arrays["episode_id"][val_idx]
    supported = tuple(depths[1:]); by_depth = {}; oof_columns = []; diagnostics = []; autograd_audits = []
    val_mean_columns = []; val_std_columns = []; fit_mean_columns = []; fit_std_columns = []
    for stage, current_depth in enumerate(supported[:-1]):
        realized = fit_losses[:, stage + 1] - fit_losses[:, stage + 2]
        fitted, oof = critics.fit_cross_fitted_critics(
            fit_features[stage], realized, fit_episodes,
            folds=int(cfg["critic_folds"]), seeds=cfg["critic_seeds"], hidden_dims=hidden_dims,
            epochs=int(cfg["critic_max_epochs"]), batch_size=int(cfg["train_batch_size"]),
            lr=float(cfg["critic_lr"]), weight_decay=float(cfg["critic_weight_decay"]), device=device)
        by_depth[int(current_depth)] = fitted; oof_columns.append(oof)
        probe = fitted[0]
        probe_model = probe.build(device).requires_grad_(True)
        probe_x = torch.from_numpy(((fit_features[stage][:8] - probe.feature_mean) / probe.feature_std).astype(np.float32)).to(device)
        autograd_audits.append({"depth": int(current_depth), **models.assert_critic_gradient_boundary(
            probe_model(probe_x), solver=model, critic=probe_model, retain_graph=False).as_dict()})
        diagnostics.append({"depth": int(current_depth), **policy.gain_ranking_calibration(oof, realized, bins=5)})
        mean, std = critics.ensemble_scores(fitted, val_features[stage], device); val_mean_columns.append(mean); val_std_columns.append(std)
        # Strict out-of-fold scores, rather than in-sample ensemble predictions,
        # calibrate compute prices. OOF uncertainty is unavailable, so zero is
        # frozen for price calibration while deployment LCB uses ensemble spread.
        fit_mean_columns.append(oof); fit_std_columns.append(np.zeros_like(oof))
    solver_hash_after = state_hash(model)
    if solver_hash_after != solver_hash_before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("critic fitting changed or connected to frozen solver")
    return {
        "critics": by_depth, "fit_losses": fit_losses[:, 1:], "val_losses": val_losses[:, 1:],
        "fit_scores_mean": np.stack(fit_mean_columns, 1), "fit_scores_std": np.stack(fit_std_columns, 1),
        "val_scores_mean": np.stack(val_mean_columns, 1), "val_scores_std": np.stack(val_std_columns, 1),
        "oof_scores": np.stack(oof_columns, 1), "diagnostics": diagnostics,
        "fit_episodes": fit_episodes, "val_episodes": val_episodes,
        "fit_exits": fit_exits[:, 1:], "val_exits": val_exits[:, 1:],
        "gradient_boundary_audit": {"solver_state_before": solver_hash_before, "solver_state_after": solver_hash_after,
                                    "solver_parameters_require_grad": False, "solver_parameter_grads_present": False,
                                    "critic_api_requires_detached_numpy": True, "autograd_by_depth": autograd_audits,
                                    "passed": bool(all(item["passed"] for item in autograd_audits))},
        "feature_tail": feature_tail, "feature_dim": int(fit_features[0].shape[1]), "hidden_dims": list(hidden_dims),
    }


def evaluate_policies(candidate: Mapping[str, Any], cfg: Mapping[str, Any], phase: str) -> dict[str, Any]:
    supported = np.asarray(candidate["supported_calls"], dtype=np.int64)
    fit_losses = candidate["fit_losses"]; eval_losses = candidate["eval_losses"]
    fit_mean = candidate["fit_scores_mean"]; fit_std = candidate["fit_scores_std"]
    eval_mean = candidate["eval_scores_mean"]; eval_std = candidate["eval_scores_std"]
    episodes = candidate["eval_episodes"]
    bootstrap = int(cfg["bootstrap_samples_internal"] if phase == "internal" else cfg["bootstrap_samples_judge"])
    family = str(candidate["family"]); critic_models_per_depth = int(candidate["critic_models_per_depth"])
    gate_flops_each = critic_flops_per_model(int(candidate["feature_dim"]), candidate["hidden_dims"])
    fit_white = candidate.get("fit_white_losses"); eval_white = candidate.get("eval_white_losses")
    rows = []; comparisons = []; allocations = {}; fixed_rows = []
    fixed_d1 = eval_losses[:, 0]
    for column, calls in enumerate(supported):
        fixed_alloc = np.full(len(eval_losses), int(calls), dtype=np.int64)
        fixed_rows.append({"policy": f"fixed_d{calls}", "mean_calls": float(calls), "raw_mse": float(eval_losses[:, column].mean()),
                           "total_flops_per_transition": BASE_PREDICT_FLOPS + solver_flops(family, fixed_alloc) / len(fixed_alloc)})
    for lcb_z in cfg["critic_lcb_z"]:
        fit_score = fit_mean - float(lcb_z) * fit_std
        eval_score = eval_mean - float(lcb_z) * eval_std
        for target in cfg["target_mean_calls"]:
            if float(target) > supported.max():
                continue
            calibrated = policy.calibrate_compute_price(fit_score, float(target), exit_calls=supported)
            price = float(calibrated["compute_price"])
            selected = policy.causal_sequential_stopping(eval_score, compute_price=price, exit_calls=supported)
            adaptive = policy.gather_exit_values(eval_losses, selected, supported)
            baseline = policy.strongest_transition_independent_baseline(
                fit_losses, supported, n=len(selected), target_total_calls=int(selected.sum()),
                seed=int(cfg["selection_seed"]) + int(round(float(target) * 100)) + int(round(float(lcb_z) * 1000)))
            baseline_loss = policy.gather_exit_values(eval_losses, baseline["selected_calls"], supported)
            expected_baseline_loss = np.einsum("nd,d->n", eval_losses, baseline["probabilities"], optimize=False)
            perm_scores = policy.permute_critic_scores(eval_score, int(cfg["selection_seed"]) + 77, episode_ids=episodes)
            perm_calls = policy.causal_sequential_stopping(perm_scores, compute_price=price, exit_calls=supported)
            perm_loss = policy.gather_exit_values(eval_losses, perm_calls, supported)
            perm_baseline = policy.strongest_transition_independent_baseline(
                fit_losses, supported, n=len(perm_calls), target_total_calls=int(perm_calls.sum()),
                seed=int(cfg["selection_seed"]) + 909 + int(round(float(target) * 100)))
            perm_baseline_loss = policy.gather_exit_values(eval_losses, perm_baseline["selected_calls"], supported)
            # Exact histogram control is the clean matched-call label-free null.
            null_calls = policy.randomized_histogram_control(selected, int(cfg["selection_seed"]) + 88)
            null_loss = policy.gather_exit_values(eval_losses, null_calls, supported)
            key = f"z{float(lcb_z):.2f}_b{float(target):.2f}"
            allocations[key] = selected
            vs_baseline = policy.clustered_paired_loss_ci(adaptive, baseline_loss, episodes, samples=bootstrap, seed=int(cfg["bootstrap_seed"]))
            vs_expected = policy.clustered_paired_loss_ci(adaptive, expected_baseline_loss, episodes, samples=bootstrap, seed=int(cfg["bootstrap_seed"]) + 11)
            vs_d1 = policy.clustered_paired_loss_ci(adaptive, fixed_d1, episodes, samples=bootstrap, seed=int(cfg["bootstrap_seed"]) + 1)
            vs_null = policy.clustered_paired_loss_ci(adaptive, null_loss, episodes, samples=bootstrap, seed=int(cfg["bootstrap_seed"]) + 2)
            perm_vs_baseline = policy.clustered_paired_loss_ci(perm_loss, perm_baseline_loss, episodes, samples=bootstrap, seed=int(cfg["bootstrap_seed"]) + 3)
            gate_decisions = int(sum(np.count_nonzero(selected >= int(current)) for current in supported[:-1]))
            adaptive_flops = BASE_PREDICT_FLOPS + (solver_flops(family, selected) + gate_decisions * critic_models_per_depth * gate_flops_each) / len(selected)
            baseline_flops = BASE_PREDICT_FLOPS + solver_flops(family, baseline["selected_calls"]) / len(selected)
            row = {"key": key, "lcb_z": float(lcb_z), "target_mean_calls": float(target), "mean_calls": float(selected.mean()),
                   "total_calls": int(selected.sum()), "raw_mse": float(adaptive.mean()), "baseline_raw_mse": float(baseline_loss.mean()),
                   "expected_baseline_raw_mse": float(expected_baseline_loss.mean()), "null_raw_mse": float(null_loss.mean()),
                   "permuted_score_mean_calls": float(perm_calls.mean()), "permuted_score_raw_mse": float(perm_loss.mean()),
                   "permuted_vs_own_matched_baseline": perm_vs_baseline,
                   "exact_baseline_calls": bool(baseline["audit"]["exact_total_match"]),
                   "adaptive_call_audit": policy.call_audit(selected, supported_calls=supported, target_total_calls=int(selected.sum())),
                   "baseline_call_audit": baseline["audit"], "vs_matched_baseline": vs_baseline,
                   "vs_expected_mixture": vs_expected, "vs_fixed_d1": vs_d1, "vs_histogram_null": vs_null,
                   "gate_decisions": gate_decisions, "gate_models_per_decision": critic_models_per_depth,
                   "gate_flops_per_model": gate_flops_each, "adaptive_total_flops_per_transition": adaptive_flops,
                   "baseline_total_flops_per_transition": baseline_flops,
                   "compute_price": price, "calibration_fit_mean_calls": float(calibrated["realized_mean_calls"])}
            if fit_white is not None and eval_white is not None:
                white_adaptive = policy.gather_exit_values(eval_white, selected, supported)
                white_baseline = policy.gather_exit_values(eval_white, baseline["selected_calls"], supported)
                row["whitened_mse"] = float(white_adaptive.mean())
                row["whitened_vs_matched"] = policy.clustered_paired_loss_ci(white_adaptive, white_baseline, episodes, samples=bootstrap, seed=int(cfg["bootstrap_seed"]) + 21)
                row["whitened_vs_fixed_d1"] = policy.clustered_paired_loss_ci(white_adaptive, eval_white[:, 0], episodes, samples=bootstrap, seed=int(cfg["bootstrap_seed"]) + 22)
            rows.append(row)
            comparisons.extend([{"key":key,"comparison":"adaptive_vs_matched",**vs_baseline}, {"key":key,"comparison":"adaptive_vs_fixed_d1",**vs_d1}, {"key":key,"comparison":"adaptive_vs_null",**vs_null}])
    frontier = fixed_rows + [{"policy":"adaptive","key":r["key"],"mean_calls":r["mean_calls"],"raw_mse":r["raw_mse"],"total_flops_per_transition":r["adaptive_total_flops_per_transition"]} for r in rows] + [{"policy":"matched_mixture","key":r["key"],"mean_calls":r["mean_calls"],"raw_mse":r["baseline_raw_mse"],"total_flops_per_transition":r["baseline_total_flops_per_transition"]} for r in rows]
    mask = policy.nondominated_mask([r["mean_calls"] for r in frontier], [r["raw_mse"] for r in frontier])
    for item, flag in zip(frontier, mask, strict=True): item["nondominated"] = bool(flag)
    flop_mask = policy.nondominated_mask([r["total_flops_per_transition"] for r in frontier], [r["raw_mse"] for r in frontier])
    for item, flag in zip(frontier, flop_mask, strict=True): item["flop_nondominated"] = bool(flag)
    for row in rows:
        row["nondominated"] = any(item.get("key") == row["key"] and item["policy"] == "adaptive" and item["nondominated"] for item in frontier)
        row["flop_nondominated"] = any(item.get("key") == row["key"] and item["policy"] == "adaptive" and item["flop_nondominated"] for item in frontier)
    return {"operating_points": rows, "comparisons": comparisons, "frontier": frontier, "allocations": allocations}


@torch.no_grad()
def audit_runtime_allocations(model: nn.Module, arrays: Mapping[str,np.ndarray], index: np.ndarray,
                              dense_exits: np.ndarray, supported: Sequence[int], policy_result: Mapping[str,Any],
                              device: torch.device, batch_size: int) -> dict[str,Any]:
    mapping={int(depth):column for column,depth in enumerate(supported)}; audits={}
    for key,selected_raw in policy_result["allocations"].items():
        selected=np.asarray(selected_raw,dtype=np.int64); actual=[]; processed=0; cursor=0; start=time.perf_counter()
        for part in batches(index,batch_size):
            h,a,z=tensors(arrays,part,device,target=False)
            chosen=torch.as_tensor(selected[cursor:cursor+len(part)],device=device)
            value,stats=model.forward_selected(h,a,z,chosen,return_stats=True)
            actual.append(value.cpu().numpy()); processed+=int(stats.processed_rows); cursor+=len(part)
        if device.type=="mps": torch.mps.synchronize()
        elapsed=time.perf_counter()-start
        actual_array=np.concatenate(actual)
        expected=np.stack([dense_exits[row,mapping[int(depth)]] for row,depth in enumerate(selected)],axis=0)
        maximum=float(np.max(np.abs(actual_array-expected)))
        audits[key]={"processed_rows":processed,"expected_processed_rows":int(selected.sum()),
                     "exact_call_match":bool(processed==int(selected.sum())),"dense_selected_max_abs_difference":maximum,
                     "dense_selected_close":bool(np.allclose(actual_array,expected,rtol=2e-5,atol=2e-6)),
                     "selected_solver_latency_seconds":elapsed,"transitions":len(selected)}
        if not audits[key]["exact_call_match"] or not audits[key]["dense_selected_close"]:
            raise RuntimeError(f"runtime adaptive audit failed for {key}")
    return audits


def select_internal_candidate(summary: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = summary["critic_diagnostics"]
    critic_pass = bool(diagnostics) and all((item.get("spearman_rho") or -1) > 0 and item["ordered_quantiles"] and item["top_bottom_realized_gain_gap"] > 0 for item in diagnostics)
    fixed = summary["fixed_exit_checks"]
    later_pass = any(item["depth"] > 1 and item["improvement_vs_d1"]["ci_low"] > 0 for item in fixed)
    valid_points = [row for row in summary["policy"]["operating_points"]
                    if row["exact_baseline_calls"]
                    and row["adaptive_call_audit"]["exact_total_match"]
                    and row["vs_matched_baseline"]["ci_low"] > 0
                    and row["vs_expected_mixture"]["ci_low"] > 0
                    and row["vs_fixed_d1"]["ci_low"] > 0
                    and row["vs_histogram_null"]["ci_low"] > 0
                    and row["nondominated"] and row["flop_nondominated"]]
    anchor = summary["anchor_audit"]["passed"]
    passed = bool(anchor and later_pass and critic_pass and valid_points)
    best = min(valid_points, key=lambda row: (row["raw_mse"], row["mean_calls"])) if valid_points else None
    return {"anchor_pass": anchor, "later_fixed_exit_pass": later_pass, "critic_pass": critic_pass,
            "matched_pareto_points": [row["key"] for row in valid_points], "passed": passed,
            "selected_operating_point": best["key"] if best else None}


def fixed_exit_checks(losses: np.ndarray, supported: Sequence[int], episodes: np.ndarray, cfg: Mapping[str, Any], samples: int) -> list[dict[str, Any]]:
    rows = []
    for column, depth in enumerate(supported):
        ci = policy.clustered_paired_loss_ci(losses[:, column], losses[:, 0], episodes, samples=samples, seed=int(cfg["bootstrap_seed"]) + int(depth))
        rows.append({"depth": int(depth), "raw_mse": float(losses[:, column].mean()), "improvement_vs_d1": ci})
    return rows


def fit_phase(device: torch.device) -> dict[str, Any]:
    cfg = json.loads((ROOT / "config.json").read_text())
    if (ROOT / "frozen_tournament.json").exists() or (ROOT / "calibration_access_receipt.json").exists():
        raise RuntimeError("discovery fit cannot be rerun after tournament freeze/calibration access")
    opaque = data_isolation.assert_combined_cache_never_opened()
    extract("train", str(device)); arrays = load_arrays("train")
    split = data_isolation.grouped_discovery_split(int(cfg["internal_validation_episodes"]), int(cfg["internal_split_seed"]))
    fit_idx = indices_for_episodes(arrays, split["discovery_fit"]); val_idx = indices_for_episodes(arrays, split["internal_validation"])
    whitening = fit_whitening(arrays["target"][fit_idx])
    v1 = load_v1_refiner(device); anchor_hash = state_hash(v1)
    pilot_summaries = []; checkpoint_dir = ROOT / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_models: dict[tuple[str,int], tuple[nn.Module,int,dict[str,Any]]] = {}
    for family in ("v3_repair", "stagewise", "contractive"):
        seeds = cfg["solver_seeds"] if family == "stagewise" else cfg["solver_seeds"][:2]
        for seed in seeds:
            log(f"training pilot={family} seed={seed}")
            model = build_solver(family, int(seed), cfg, v1, device)
            architectural_trainable = (sum(p.numel() for adapter in model.adapters for p in adapter.parameters())
                                       if family == "stagewise" else
                                       sum(p.numel() for p in model.parameters() if p.requires_grad))
            if family == "stagewise":
                model, training, accepted_depth = train_stagewise(model, arrays, fit_idx, val_idx, int(seed), cfg, device)
            else:
                model, training = train_joint_later(model, family, arrays, fit_idx, val_idx, int(seed), cfg, device); accepted_depth = 4
            audit = anchor_audit(model, family, v1, arrays, val_idx, device, int(cfg["predict_batch_size"]))
            if not audit["passed"] or state_hash(v1) != anchor_hash:
                raise RuntimeError("bitwise V1 anchor changed during solver training")
            depths = output_depths(family, accepted_depth)
            means = mean_exit_losses(model, family, arrays, val_idx, device, int(cfg["predict_batch_size"]), depths)
            checkpoint = checkpoint_dir / f"{family}_seed_{seed}.pt"
            torch.save({"family":family,"seed":int(seed),"accepted_depth":accepted_depth,"state_dict":model.state_dict(),"training":training,"anchor_sha256":anchor_hash}, checkpoint)
            record = {"family":family,"seed":int(seed),"accepted_depth":accepted_depth,"anchor_audit":audit,"validation_exit_mse":means,
                      "trainable_parameters":architectural_trainable,
                      "total_parameters":sum(p.numel() for p in model.parameters()),"checkpoint":str(checkpoint),"checkpoint_sha256":data_isolation.sha256_file(checkpoint),"training":training}
            pilot_summaries.append(record); seed_models[(family,int(seed))] = (model,accepted_depth,record)
    # Frozen V1 verifier-only control is solver seed-independent.
    verifier = copy.deepcopy(v1).eval().requires_grad_(False)
    verifier_record = {"family":"verifier_v1","seed":260713,"accepted_depth":4,
                       "anchor_audit":anchor_audit(verifier,"verifier_v1",v1,arrays,val_idx,device,int(cfg["predict_batch_size"])),
                       "validation_exit_mse":mean_exit_losses(verifier,"verifier_v1",arrays,val_idx,device,int(cfg["predict_batch_size"]),(0,1,2,4)),
                       "trainable_parameters":0,"total_parameters":sum(p.numel() for p in verifier.parameters())}
    pilot_summaries.append(verifier_record)

    # Pick one solver seed per architecture using fixed-exit validation only,
    # then train critics after that solver is frozen.
    finalist_payload = {}; candidate_summaries = []
    families = ("v3_repair","stagewise","contractive","verifier_v1")
    for family in families:
        records = [r for r in pilot_summaries if r["family"] == family]
        chosen = min(records, key=lambda r: min(float(v) for k,v in r["validation_exit_mse"].items() if int(k) >= 1))
        if family == "verifier_v1": model = verifier; accepted_depth = 4
        else: model, accepted_depth, _ = seed_models[(family,int(chosen["seed"]))]
        depths = output_depths(family, accepted_depth)
        variants = [("full", None, cfg["critic_hidden_dims"]),
                    ("compact", int(cfg["compact_critic_feature_tail"]), cfg["compact_critic_hidden_dims"])]
        for variant, feature_tail, hidden_dims in variants:
            candidate_name = f"{family}__{variant}"
            critic_result = fit_critics_for_candidate(model,family,depths,arrays,fit_idx,val_idx,cfg,device,
                                                       feature_tail=feature_tail,hidden_dims=hidden_dims)
            fit_white = whitened_losses(arrays["target"][fit_idx], critic_result["fit_exits"], whitening)
            val_white = whitened_losses(arrays["target"][val_idx], critic_result["val_exits"], whitening)
            evaluation_input = {**critic_result,"family":family,"supported_calls":depths[1:],"eval_losses":critic_result["val_losses"],
                                "eval_scores_mean":critic_result["val_scores_mean"],"eval_scores_std":critic_result["val_scores_std"],
                                "eval_episodes":critic_result["val_episodes"],"fit_white_losses":fit_white,"eval_white_losses":val_white,
                                "critic_models_per_depth":int(cfg["critic_folds"])}
            policy_result = evaluate_policies(evaluation_input,cfg,"internal")
            runtime_audit = audit_runtime_allocations(model,arrays,val_idx,critic_result["val_exits"],depths[1:],policy_result,device,int(cfg["predict_batch_size"]))
            checks = fixed_exit_checks(critic_result["val_losses"],depths[1:],critic_result["val_episodes"],cfg,int(cfg["bootstrap_samples_internal"]))
            summary = {"candidate":candidate_name,"family":family,"critic_variant":variant,"selected_seed":int(chosen["seed"]),
                       "accepted_depth":accepted_depth,"supported_calls":list(depths[1:]),"anchor_audit":chosen["anchor_audit"],
                       "fixed_exit_checks":checks,"critic_diagnostics":critic_result["diagnostics"],
                       "gradient_boundary_audit":critic_result["gradient_boundary_audit"],"runtime_call_audit":runtime_audit,"policy":policy_result}
            summary["internal_decision"] = select_internal_candidate(summary,cfg)
            candidate_summaries.append(summary)
            critic_path = checkpoint_dir / f"{candidate_name}_critics.pt"
            torch.save({"family":family,"critic_variant":variant,"feature_tail":feature_tail,"hidden_dims":list(hidden_dims),
                        "feature_dim":critic_result["feature_dim"],"seed":int(chosen["seed"]),"accepted_depth":accepted_depth,
                        "solver_state_dict":model.state_dict(),"critics":critics.serialize_critics(critic_result["critics"]),
                        "supported_calls":list(depths[1:]),"fit_losses":critic_result["fit_losses"],
                        "fit_white_losses":fit_white,"whitening":whitening,
                        "gradient_boundary_audit":critic_result["gradient_boundary_audit"],
                        "fit_scores_mean":critic_result["fit_scores_mean"],"fit_scores_std":critic_result["fit_scores_std"]},critic_path)
            finalist_payload[candidate_name] = {"path":str(critic_path.resolve()),"sha256":data_isolation.sha256_file(critic_path),
                                               "solver_family":family,"critic_variant":variant,"selected_seed":int(chosen["seed"]),
                                               "accepted_depth":accepted_depth,"supported_calls":list(depths[1:]),
                                               "internal_pass":summary["internal_decision"]["passed"]}
            write_json(ROOT / "metrics" / f"internal_{candidate_name}.json", summary)
            log(f"internal pilot={candidate_name} pass={summary['internal_decision']['passed']}")
    advancing = [family for family,payload in finalist_payload.items() if payload["internal_pass"]]
    fit_decision = {"phase":"internal_discovery","advancing_candidates":advancing,"discovery_internal_pass":bool(advancing),
                    "opaque_v3_cache_audit":opaque,"v1_anchor_state_sha256":anchor_hash,"pilots":pilot_summaries,"candidates":candidate_summaries}
    write_json(ROOT / "internal_decision.json", fit_decision)
    write_csv(ROOT / "metrics" / "pilot_seed_summary.csv", [{k:v for k,v in row.items() if k not in ("training",)} for row in pilot_summaries])
    if advancing:
        source_names = ("run_discovery.py","policy.py","critics.py","models.py","data_isolation.py","extract_isolated.py")
        tournament = {"schema_version":1,"status":"frozen_before_v3_calibration_target_access","config_sha256":data_isolation.sha256_file(ROOT/"config.json"),
                      "plan_sha256":data_isolation.sha256_file(ROOT/"PLAN.md"),"v3_manifest_sha256":data_isolation.PINNED["manifest_sha256"],
                      "source_sha256":{name:data_isolation.sha256_file(ROOT/name) for name in source_names},
                      "candidates":{name:finalist_payload[name] for name in advancing},"selection_rule":"among candidates passing every internal gate, require all final discovery gates and choose lowest calibration raw MSE at a passing nondominated operating point; ties lower calls then family name",
                      "test_targets":"forbidden","budgets":cfg["target_mean_calls"],"critic_lcb_z":cfg["critic_lcb_z"],"bootstrap_samples":cfg["bootstrap_samples_judge"],"bootstrap_seed":cfg["bootstrap_seed"]}
        write_json(ROOT / "frozen_tournament.json", tournament)
        write_json(ROOT / "frozen_tournament.sha256.json", {"sha256":data_isolation.sha256_file(ROOT/"frozen_tournament.json")})
    else:
        write_json(ROOT / "decision.json", {"decision":"discovery_gate_failed","reason":"no candidate passed the episode-held-out internal discovery criteria","v4_created":False,"v3_calibration_consumed":False})
        write_report(fit_decision, None)
    return fit_decision


def load_frozen_candidate(payload: Mapping[str,Any], cfg: Mapping[str,Any], v1: nn.Module, device: torch.device):
    path = Path(payload["path"])
    if data_isolation.sha256_file(path) != payload["sha256"]: raise RuntimeError("frozen candidate drift")
    saved = torch.load(path,map_location="cpu",weights_only=False)
    family = str(saved["family"])
    if family == "verifier_v1": model=copy.deepcopy(v1)
    else: model=build_solver(family,int(saved["seed"]),cfg,v1,device)
    model.load_state_dict(saved["solver_state_dict"],strict=True); model.to(device).eval().requires_grad_(False)
    return model, saved, critics.deserialize_critics(saved["critics"])


def judge_phase(device: torch.device) -> dict[str,Any]:
    tournament_path=ROOT/"frozen_tournament.json"
    if not tournament_path.exists(): raise RuntimeError("no advancing frozen tournament; calibration judge is forbidden")
    tournament_hash=data_isolation.sha256_file(tournament_path)
    expected=json.loads((ROOT/"frozen_tournament.sha256.json").read_text())["sha256"]
    if tournament_hash!=expected: raise RuntimeError("frozen tournament hash drift")
    if data_isolation.sha256_file(ROOT/"config.json") != json.loads(tournament_path.read_text())["config_sha256"]:
        raise RuntimeError("frozen config drift")
    if data_isolation.sha256_file(ROOT/"PLAN.md") != json.loads(tournament_path.read_text())["plan_sha256"]:
        raise RuntimeError("frozen plan drift")
    tournament = json.loads(tournament_path.read_text())
    for name, expected_source in tournament["source_sha256"].items():
        if data_isolation.sha256_file(ROOT/name) != expected_source:
            raise RuntimeError(f"frozen judge source drift: {name}")
    data_isolation.create_calibration_receipt(ROOT/"calibration_access_receipt.json",tournament_hash)
    cfg=json.loads((ROOT/"config.json").read_text())
    extract("calibration_once",str(device)); arrays=load_arrays("calibration_once")
    index=np.arange(len(arrays["episode_id"])); episodes=arrays["episode_id"]
    v1=load_v1_refiner(device); results=[]
    for candidate_name,payload in tournament["candidates"].items():
        model,saved,by_depth=load_frozen_candidate(payload,cfg,v1,device)
        family=str(saved["family"])
        depths=tuple(int(v) for v in saved["supported_calls"]); full_depths=(0,*depths)
        exits,features,losses=dense_outputs(model,family,arrays,index,device,int(cfg["predict_batch_size"]),full_depths)
        if saved.get("feature_tail") is not None:
            features=[value[:,-int(saved["feature_tail"]):] for value in features]
        score_mean=[]; score_std=[]
        for stage,current in enumerate(depths[:-1]):
            mean,std=critics.ensemble_scores(by_depth[int(current)],features[stage],device); score_mean.append(mean); score_std.append(std)
        white_losses=whitened_losses(arrays["target"][index],exits[:,1:],saved["whitening"])
        input_payload={"family":family,"supported_calls":depths,"fit_losses":saved["fit_losses"],"fit_scores_mean":saved["fit_scores_mean"],"fit_scores_std":saved["fit_scores_std"],
                       "eval_losses":losses[:,1:],"eval_scores_mean":np.stack(score_mean,1),"eval_scores_std":np.stack(score_std,1),"eval_episodes":episodes,
                       "fit_white_losses":saved["fit_white_losses"],"eval_white_losses":white_losses,
                       "feature_dim":int(saved["feature_dim"]),"hidden_dims":saved["hidden_dims"],"critic_models_per_depth":int(cfg["critic_folds"])}
        policies=evaluate_policies(input_payload,cfg,"judge")
        runtime_audit=audit_runtime_allocations(model,arrays,index,exits[:,1:],depths,policies,device,int(cfg["predict_batch_size"]))
        diagnostics=[]
        for stage,current in enumerate(depths[:-1]):
            realized=losses[:,stage+1]-losses[:,stage+2]
            diagnostics.append({"depth":current,**policy.gain_ranking_calibration(input_payload["eval_scores_mean"][:,stage],realized,bins=5)})
        checks=fixed_exit_checks(losses[:,1:],depths,episodes,cfg,int(cfg["bootstrap_samples_judge"]))
        anchor=anchor_audit(model,family,v1,arrays,index,device,int(cfg["predict_batch_size"]))
        summary={"candidate":candidate_name,"family":family,"critic_variant":saved["critic_variant"],"selected_seed":int(saved["seed"]),"supported_calls":list(depths),"anchor_audit":anchor,
                 "fixed_exit_checks":checks,"critic_diagnostics":diagnostics,"gradient_boundary_audit":saved["gradient_boundary_audit"],
                 "runtime_call_audit":runtime_audit,"policy":policies}
        summary["final_discovery_decision"]=select_internal_candidate(summary,cfg)
        results.append(summary); write_json(ROOT/"metrics"/f"judge_{family}.json",summary)
    passing=[r for r in results if r["final_discovery_decision"]["passed"]]
    if passing:
        choices=[]
        for result in passing:
            key=result["final_discovery_decision"]["selected_operating_point"]
            row=next(v for v in result["policy"]["operating_points"] if v["key"]==key)
            choices.append((row["raw_mse"],row["mean_calls"],result["candidate"],key))
        _,_,winner,key=min(choices)
        decision_name="discovery_gate_passed"
    else:
        winner=None; key=None; decision_name="discovery_gate_failed"
    decision={"decision":decision_name,"winner":winner,"operating_point":key,"v3_calibration_consumed_once":True,"v3_test_targets_consumed":False,
              "v4_authorized":bool(passing),"tournament_sha256":tournament_hash,"candidate_results":results}
    write_json(ROOT/"decision.json",decision); write_report(json.loads((ROOT/"internal_decision.json").read_text()),decision)
    return decision


def write_report(internal: Mapping[str,Any], judge: Mapping[str,Any]|None) -> None:
    if judge is None:
        verdict="discovery_gate_failed"; detail="No pilot passed every episode-held-out internal criterion, so the V3 calibration split remained untouched."
    else:
        verdict=judge["decision"]
        detail=(f"The one-shot V3 calibration tournament selected `{judge['winner']}` at `{judge['operating_point']}`." if judge.get("winner") else "No internally advancing pilot survived the one-shot V3 calibration tournament.")
    lines=["# LeWM Adaptive Compute Discovery Report","", "## Decision","",f"**`{verdict}`**. {detail}","",
           "The existing V3 combined target cache was never opened. Discovery-fit and internal-validation were grouped subsets of the 420 V3 training episodes. V3 test targets were never extracted or loaded.","",
           "## Forensic conclusion","", "All seven suspected V3 defects were confirmed: critic gradients entered the solver, the monotonic term could reward worsening an earlier exit, normalization went stale, critic losses dominated raw MSE by about three orders of magnitude, epoch 0 was ineligible, pretrained/random parameters shared one optimizer, and stochastic depth was batch-level.","",
           "## Internal pilots","", "| family | selected seed | anchor | later exit | critic | matched Pareto | pass |","| --- | ---: | --- | --- | --- | --- | --- |"]
    for row in internal.get("candidates",[]):
        d=row["internal_decision"]
        lines.append(f"| {row['family']} | {row['selected_seed']} | {d['anchor_pass']} | {d['later_fixed_exit_pass']} | {d['critic_pass']} | {','.join(d['matched_pareto_points']) or 'none'} | {d['passed']} |")
    lines += ["", "Full machine-readable metrics, clustered intervals, gain quantiles, exact-call audits, checkpoints, and frozen hashes are in `metrics/`, `checkpoints/`, and `decision.json`.","",
              "No claim of a first adaptive world model is made; this study concerns adaptive computation in a visual latent physical world model.",""]
    (ROOT/"REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("fit","judge")); parser.add_argument("--device",default="auto")
    args=parser.parse_args(); device=choose_device(args.device)
    result=fit_phase(device) if args.phase=="fit" else judge_phase(device)
    log(f"complete phase={args.phase} decision={result.get('decision',result.get('discovery_internal_pass'))}")


if __name__=="__main__": main()
