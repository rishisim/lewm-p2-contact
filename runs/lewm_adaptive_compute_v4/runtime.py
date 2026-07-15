#!/usr/bin/env python3
"""Frozen V4 model I/O, gate scoring, and dense/sparse execution paths.

Nothing in this module fits or calibrates a scientific object.  It loads the
released LeWM, the sealed stagewise solver, the sealed linear student, and the
sealed discovery whitening transform by exact byte hash.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

import common


_DISCOVERY_MODULES: tuple[Any, Any, Any] | None = None
_MODEL_IO: Any | None = None


def choose_device(name: str = "auto") -> torch.device:
    value = name.lower()
    if value == "auto":
        value = "mps" if torch.backends.mps.is_available() else "cpu"
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_model_io() -> Any:
    global _MODEL_IO
    if _MODEL_IO is not None:
        return _MODEL_IO
    path = common.REPO / "runs/lewm_adaptive_compute_v2/model_io.py"
    spec = importlib.util.spec_from_file_location("v4_frozen_model_io", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen model I/O")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _MODEL_IO = module
    return module


def load_discovery_modules() -> tuple[Any, Any, Any]:
    """Load immutable discovery runner/models/policy without copying code."""
    global _DISCOVERY_MODULES
    if _DISCOVERY_MODULES is not None:
        return _DISCOVERY_MODULES
    discovery = common.DISCOVERY
    text = str(discovery)
    if text not in sys.path:
        sys.path.insert(0, text)
    # The discovery modules do not import a module named common.  Importing
    # these stable plain names therefore cannot shadow this V4 common module.
    for name in ("critics", "data_isolation", "models", "policy", "extract_isolated"):
        if name not in sys.modules:
            __import__(name)
    spec = importlib.util.spec_from_file_location(
        "v4_frozen_discovery_runner", discovery / "run_discovery.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load immutable discovery runner")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    _DISCOVERY_MODULES = (runner, sys.modules["models"], sys.modules["policy"])
    return _DISCOVERY_MODULES


def load_base_model(device: torch.device) -> tuple[nn.Module, Any, dict[str, Any]]:
    common.verify_frozen_objects()
    model_io = load_model_io()
    model, contract, provenance = model_io.load_frozen_lewm(
        common.BASE_CONFIG,
        common.BASE_WEIGHTS,
        device,
        expected_config_sha256=common.EXPECTED["base_config_sha256"],
        expected_weights_sha256=common.EXPECTED["base_weights_sha256"],
    )
    return model, contract, provenance


def load_solver(device: torch.device) -> tuple[nn.Module, nn.Module, Any]:
    common.verify_hash(
        common.SOLVER_CHECKPOINT,
        common.EXPECTED["solver_checkpoint_sha256"],
        "stagewise solver",
    )
    runner, models, _ = load_discovery_modules()
    compression_cfg = common.read_json(common.COMPRESSION / "config.json")
    discovery_cfg = common.read_json(common.DISCOVERY / "config.json")
    seed = int(compression_cfg["selected_solver_seed"])
    v1 = runner.load_v1_refiner(device)
    model = runner.build_solver("stagewise", seed, discovery_cfg, v1, device)
    payload = torch.load(common.SOLVER_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("family") != "stagewise" or int(payload.get("accepted_depth", -1)) != 4:
        raise RuntimeError("frozen stagewise checkpoint metadata mismatch")
    if payload.get("anchor_sha256") != compression_cfg["selected_solver_anchor_state_sha256"]:
        raise RuntimeError("frozen solver anchor mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.set_trainable_stage(None)
    model.to(device).eval().requires_grad_(False)
    v1.to(device).eval().requires_grad_(False)
    return model, v1, models


class FrozenLinearStudent(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        if tuple(weight.shape) != (1, common.FEATURE_DIM + 3) or tuple(bias.shape) != (1,):
            raise RuntimeError(f"unexpected frozen linear head shape {tuple(weight.shape)} {tuple(bias.shape)}")
        self.linear = nn.Linear(common.FEATURE_DIM + 3, 1)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value).squeeze(1)


@dataclass(frozen=True)
class GateBundle:
    model: FrozenLinearStudent
    feature_indices: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_center: float
    target_scale: float
    compute_price: float
    metadata: dict[str, Any]


def load_gate(device: torch.device) -> GateBundle:
    common.verify_hash(
        common.STUDENT_CHECKPOINT,
        common.EXPECTED["student_checkpoint_sha256"],
        "linear student",
    )
    payload = torch.load(common.STUDENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("family") != common.FAMILY or payload.get("operating_point") != common.OPERATING_POINT:
        raise RuntimeError("frozen student family/operating-point mismatch")
    if float(payload.get("compute_price")) != common.COMPUTE_PRICE:
        raise RuntimeError("frozen student compute-price mismatch")
    student = payload["student"]
    if student.get("family") != common.FAMILY or tuple(student.get("hidden_dims", ())) != ():
        raise RuntimeError("student is not the frozen shared full-feature linear gate")
    indices_np = np.asarray(student["feature_indices"], dtype=np.int64)
    if not np.array_equal(indices_np, np.arange(common.FEATURE_DIM, dtype=np.int64)):
        raise RuntimeError("frozen gate feature set changed")
    state = student["state_dict"]
    model = FrozenLinearStudent(
        state["network.0.weight"].detach().clone(),
        state["network.0.bias"].detach().clone(),
    ).to(device).eval().requires_grad_(False)
    mean_np = np.asarray(student["feature_mean"], dtype=np.float32)
    std_np = np.asarray(student["feature_std"], dtype=np.float32)
    if mean_np.shape != (common.FEATURE_DIM + 3,) or std_np.shape != mean_np.shape:
        raise RuntimeError("frozen gate normalization shape mismatch")
    if not np.isfinite(mean_np).all() or not np.isfinite(std_np).all() or np.any(std_np <= 0):
        raise RuntimeError("frozen gate normalization invalid")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameter_count != common.GATE_PARAMETERS:
        raise RuntimeError(f"gate parameter count changed: {parameter_count}")
    metadata = {
        "family": payload["family"],
        "operating_point": payload["operating_point"],
        "compute_price": float(payload["compute_price"]),
        "parameters": parameter_count,
        "input_dim": int(student["input_dim"]),
        "hidden_dims": list(student["hidden_dims"]),
        "feature_indices_sha256": common.array_sha256(indices_np),
        "feature_mean_sha256": common.array_sha256(mean_np),
        "feature_std_sha256": common.array_sha256(std_np),
        "target_center": float(student["target_center"]),
        "target_scale": float(student["target_scale"]),
        "all_gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
    }
    return GateBundle(
        model=model,
        feature_indices=torch.as_tensor(indices_np, dtype=torch.long, device=device),
        feature_mean=torch.as_tensor(mean_np, device=device),
        feature_std=torch.as_tensor(std_np, device=device),
        target_center=float(student["target_center"]),
        target_scale=float(student["target_scale"]),
        compute_price=float(payload["compute_price"]),
        metadata=metadata,
    )


def load_whitening() -> dict[str, np.ndarray]:
    common.verify_hash(
        common.WHITENING,
        common.EXPECTED["whitening_sha256"],
        "discovery whitening",
    )
    with np.load(common.WHITENING, allow_pickle=False) as stored:
        if set(stored.files) != {"mean", "matrix", "eigenvalues", "floor"}:
            raise RuntimeError("frozen whitening schema mismatch")
        result = {name: stored[name].copy() for name in stored.files}
    if result["matrix"].shape != (common.LATENT_DIM, common.LATENT_DIM):
        raise RuntimeError("frozen whitening matrix shape mismatch")
    return result


@torch.inference_mode()
def base_predict(base_model: nn.Module, history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return base_model.predict(history, base_model.action_encoder(actions))[:, -1]


@torch.inference_mode()
def gate_score(
    gate: GateBundle,
    model_module: Any,
    history: torch.Tensor,
    actions: torch.Tensor,
    current: torch.Tensor,
    update: torch.Tensor,
    stage: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stage not in (0, 1, 2):
        raise ValueError("gate stage must be 0, 1, or 2")
    causal = model_module.build_causal_features(history, actions, current, update)
    selected = causal.index_select(1, gate.feature_indices)
    depth = torch.zeros((len(history), 3), dtype=selected.dtype, device=selected.device)
    depth[:, stage] = 1.0
    encoded = torch.cat((selected, depth), dim=1)
    normalized = (encoded - gate.feature_mean) / gate.feature_std
    score = gate.model(normalized) * gate.target_scale + gate.target_center
    return score, causal


def sequential_calls_numpy(scores: np.ndarray, price: float = common.COMPUTE_PRICE) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("scores must be finite [N,3]")
    calls = np.ones(len(values), dtype=np.int64)
    active = np.ones(len(values), dtype=bool)
    for stage in range(3):
        active &= values[:, stage] > float(price)
        calls += active.astype(np.int64)
    return calls


@torch.inference_mode()
def dense_adaptive(
    solver: nn.Module,
    gate: GateBundle,
    model_module: Any,
    history: torch.Tensor,
    actions: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs, updates = solver(history, actions, base, max_depth=4, return_updates=True)
    score_parts = []
    feature_parts = []
    for stage, depth in enumerate((1, 2, 3)):
        score, feature = gate_score(
            gate, model_module, history, actions, outputs[depth], updates[depth], stage
        )
        score_parts.append(score)
        feature_parts.append(feature)
    scores = torch.stack(score_parts, dim=1)
    calls = torch.ones(len(history), dtype=torch.long, device=history.device)
    active = torch.ones(len(history), dtype=torch.bool, device=history.device)
    for stage in range(3):
        active &= scores[:, stage] > gate.compute_price
        calls += active.long()
    stacked = torch.stack([outputs[depth] for depth in (1, 2, 3, 4)], dim=1)
    selected = stacked[torch.arange(len(history), device=history.device), calls - 1]
    return selected, calls, scores, torch.stack(feature_parts, dim=1)


@torch.inference_mode()
def sparse_adaptive_reference(
    solver: nn.Module,
    gate: GateBundle,
    model_module: Any,
    history: torch.Tensor,
    actions: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchored = solver.anchor(history, actions, base)
    current = anchored[1]
    last_update = current - base
    calls = torch.ones(len(history), dtype=torch.long, device=history.device)
    active = torch.arange(len(history), device=history.device)
    for stage, adapter in enumerate(solver.adapters):
        if active.numel() == 0:
            break
        local_score, _ = gate_score(
            gate,
            model_module,
            history.index_select(0, active),
            actions.index_select(0, active),
            current.index_select(0, active),
            last_update.index_select(0, active),
            stage,
        )
        active = active[local_score > gate.compute_price]
        if active.numel() == 0:
            break
        preceding = current.index_select(0, active)
        update = adapter(
            history.index_select(0, active), actions.index_select(0, active), preceding
        )
        current = current.index_copy(0, active, preceding + update)
        last_update = last_update.index_copy(0, active, update)
        calls[active] += 1
    return current, calls


@torch.inference_mode()
def sparse_adaptive_optimized(
    solver: nn.Module,
    gate: GateBundle,
    model_module: Any,
    history: torch.Tensor,
    actions: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semantics-preserving compact path with contiguous active tensors.

    Scores and depth decisions are identical to the reference path.  The only
    engineering change is carrying compact active tensors between stages and
    scattering completed predictions once.
    """
    anchored = solver.anchor(history, actions, base)
    output = anchored[1]
    calls = torch.ones(len(history), dtype=torch.long, device=history.device)
    active_index = torch.arange(len(history), device=history.device)
    active_h = history
    active_a = actions
    active_current = output
    active_update = output - base
    for stage, adapter in enumerate(solver.adapters):
        if active_index.numel() == 0:
            break
        local_score, _ = gate_score(
            gate, model_module, active_h, active_a, active_current, active_update, stage
        )
        keep = local_score > gate.compute_price
        stopped = ~keep
        if stopped.any():
            stopped_index = active_index[stopped]
            output = output.index_copy(0, stopped_index, active_current[stopped])
        active_index = active_index[keep]
        if active_index.numel() == 0:
            break
        active_h = active_h[keep]
        active_a = active_a[keep]
        preceding = active_current[keep]
        active_update = adapter(active_h, active_a, preceding)
        active_current = preceding + active_update
        calls[active_index] += 1
    if active_index.numel():
        output = output.index_copy(0, active_index, active_current)
    return output, calls


def module_audit(*modules: nn.Module) -> dict[str, Any]:
    records = []
    for module in modules:
        records.append(
            {
                "class": module.__class__.__name__,
                "state_sha256": state_sha256(module),
                "parameters": int(sum(value.numel() for value in module.parameters())),
                "training": bool(module.training),
                "all_parameters_frozen": all(not value.requires_grad for value in module.parameters()),
                "all_gradients_absent": all(value.grad is None for value in module.parameters()),
            }
        )
    passed = all(
        not item["training"] and item["all_parameters_frozen"] and item["all_gradients_absent"]
        for item in records
    )
    return {"passed": passed, "modules": records}


__all__ = [
    "GateBundle",
    "base_predict",
    "choose_device",
    "dense_adaptive",
    "gate_score",
    "load_base_model",
    "load_discovery_modules",
    "load_gate",
    "load_model_io",
    "load_solver",
    "load_whitening",
    "module_audit",
    "sequential_calls_numpy",
    "sparse_adaptive_optimized",
    "sparse_adaptive_reference",
    "state_sha256",
    "synchronize",
]
