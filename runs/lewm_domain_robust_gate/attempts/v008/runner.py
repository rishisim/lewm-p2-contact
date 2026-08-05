#!/usr/bin/env python3
"""Frozen-stack development and robust-gate sparse/dense execution.

Production adaptive outputs come only from sequential reached-row sparse
execution.  The full-batch dense path is a numerical shadow and fixed-depth
source.  Every persisted gate primitive is causal and contact-free.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from compile_gate import COMPILED_KEYS, validate_compiled_arrays
from counted_features import (
    ACTION_DIM,
    FEATURE_DIM,
    HISTORY_LEN,
    LATENT_DIM,
    assert_frozen_feature_inputs,
    build_counted_causal_features,
)
from flops import exact_compute, gate_cost
from input_loader import array_sha256, load_model_gate_inputs
from inherited_authorization import (
    ACTIVE_ATTEMPT,
    INHERITANCE_SEAL_PATH,
    SCIENCE_ATTEMPT,
    SOURCE_PRE_SELECTION_SEAL_PATH,
    InheritedAuthorizationError,
    verify_inherited_pre_data_authorization,
)
from generator import (
    GenerationInputs,
    load_generation_inputs,
    materialization_lock,
    preflight_output_namespace,
    validate_raw_manifest_contract,
)
from runtime_contract import require_mps_device, verify_here
from study_common import (
    ATTEMPT_ROOT,
    REGIME_ORDER,
    REPO_ROOT,
    ROWS_PER_EPISODE,
    STUDY_ROOT,
    atomic_json,
    atomic_npz,
    read_json,
    read_verified_controller,
    relative_to_repo,
    resolve_repo_relative,
    sha256_file,
)


ATTEMPT_VERSION = ACTIVE_ATTEMPT
STAGES = 3
DEPTHS = 4
NUMERICAL_RTOL = 2e-6
NUMERICAL_ATOL = 2e-7
NUMERICAL_MAX_ABS = 4.76837158203125e-7
CONFIRMATION_EQUIVALENCE_CHECK_KEYS = frozenset(
    {
        "same_sparse_selected_exact",
        "same_sparse_calls_exact",
        "same_sparse_scores_exact",
        "same_sparse_head_scores_exact",
        "same_sparse_features_exact",
        "same_sparse_primitives_exact",
        "manual_sparse_forward_selected_exact",
        "sparse_dense_calls_exact",
        "sparse_dense_histogram_exact",
        "depth_one_selected_rows_exact",
        "cross_batch_allclose",
        "cross_batch_max_abs_ceiling",
        "sparse_primitive_feature_reconstruction_exact",
        "sparse_feature_score_reconstruction_exact",
        "dense_primitive_feature_reconstruction_exact",
        "dense_feature_score_reconstruction_exact",
        "unreached_sparse_values_are_nan",
        "target_loss_never_computed",
    }
)

DEVELOPMENT_ARRAY_KEYS = frozenset(
    {
        "episode_slot",
        "model_step",
        "target",
        "exits",
        "production_features",
        "history",
        "action_history",
        "stage_current",
        "stage_update",
    }
)
EXECUTION_ARRAY_KEYS_WITHOUT_TARGET = frozenset(
    {
        "episode_slot",
        "model_step",
        "exits",
        "selected",
        "calls",
        "scores",
        "head_scores",
        "production_features",
        "history",
        "action_history",
        "stage_current",
        "stage_update",
        "reached",
        "dense_scores",
        "dense_head_scores",
        "dense_stage_current",
        "dense_stage_update",
    }
)

V5_ROOT = (
    REPO_ROOT
    / "runs/lewm_v5_readiness_program/v5_package_versions/v004"
)
FROZEN_FACADE_SOURCES = {
    V5_ROOT / "runner.py": "ebcbe74b409867c538460b53b5ddb69a05c9163832ec4ab4eb2a3ea484ab1174",
    REPO_ROOT / "runs/lewm_adaptive_compute_v4/runtime.py": "801b160ad9c457f7eabb4b92395c90cb7eb2f9d1809a627bd0202df8a210bfdf",
    REPO_ROOT / "runs/lewm_adaptive_compute_v2/model_io.py": "5505e529ba17aeb0c641e7d4e8dc511d0ed61decdc13a596915ba8964b954874",
}


@dataclass(frozen=True)
class GateTensors:
    """Runtime-only stage-specific min-ensemble tensors."""

    architecture: str
    candidate_id: str
    head_names: tuple[str, ...]
    weights: Any
    biases: Any
    thresholds: Any
    source_path: Path | None = None
    source_sha256: str | None = None

    @property
    def head_count(self) -> int:
        return len(self.head_names)


@dataclass(frozen=True)
class ScientificReplayContext:
    """Frozen MPS objects required to reproduce an episode before adoption."""

    torch: Any
    runtime: Any
    model_io: Any
    device: Any
    base: Any
    contract: Any
    solver: Any
    distribution_common: Any
    module_before: Any


@dataclass(frozen=True)
class SparseExecution:
    selected: Any
    calls: Any
    scores: Any
    head_scores: Any
    features: Any
    stage_current: Any
    stage_update: Any
    reached: Any


@dataclass(frozen=True)
class DenseExecution:
    exits: Any
    calls: Any
    scores: Any
    head_scores: Any
    features: Any
    stage_current: Any
    stage_update: Any


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise RuntimeError(f"compiled {name} is not scalar")
    result = str(array.reshape(()).item())
    if not result:
        raise RuntimeError(f"compiled {name} is empty")
    return result


def load_gate_tensors(torch: Any, device: Any, path: Path) -> GateTensors:
    """Load only the selected compiled gate; fitting artifacts remain closed."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != COMPILED_KEYS:
            raise RuntimeError(
                "compiled gate key drift: "
                f"expected={sorted(COMPILED_KEYS)} observed={sorted(stored.files)}"
            )
        if int(np.asarray(stored["schema_version"]).reshape(())) != 1:
            raise RuntimeError("compiled gate schema drift")
        architecture = _scalar_text(stored["architecture"], "architecture")
        candidate_id = _scalar_text(stored["candidate_id"], "candidate_id")
        weights = stored["weights"].copy()
        biases = stored["biases"].copy()
        thresholds = stored["thresholds"].copy()
        names_array = stored["head_names"].copy()
    validation = validate_compiled_arrays(
        architecture=architecture,
        candidate_id=candidate_id,
        weights=weights,
        biases=biases,
        thresholds=thresholds,
        head_names=names_array,
        require_float32=True,
    )
    gate = GateTensors(
        architecture=architecture,
        candidate_id=candidate_id,
        head_names=tuple(validation["head_names"]),
        weights=torch.as_tensor(weights, dtype=torch.float32, device=device),
        biases=torch.as_tensor(biases, dtype=torch.float32, device=device),
        thresholds=torch.as_tensor(thresholds, dtype=torch.float32, device=device),
        source_path=path.resolve(),
        source_sha256=sha256_file(path),
    )
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (gate.weights, gate.biases, gate.thresholds)
    ):
        raise RuntimeError("compiled gate became nonfinite on runtime device")
    return gate


def gate_from_tensors(
    torch: Any,
    *,
    architecture: str,
    candidate_id: str,
    head_names: tuple[str, ...],
    weights: Any,
    biases: Any,
    thresholds: Any,
    device: Any | None = None,
) -> GateTensors:
    """Construct a validated runtime gate, principally for synthetic qualification."""

    weight_array = np.asarray(weights, dtype=np.float32)
    bias_array = np.asarray(biases, dtype=np.float32)
    threshold_array = np.asarray(thresholds, dtype=np.float32)
    validate_compiled_arrays(
        architecture=architecture,
        candidate_id=candidate_id,
        weights=weight_array,
        biases=bias_array,
        thresholds=threshold_array,
        head_names=head_names,
        require_float32=True,
    )
    return GateTensors(
        architecture=architecture,
        candidate_id=candidate_id,
        head_names=tuple(head_names),
        weights=torch.as_tensor(weight_array, dtype=torch.float32, device=device),
        biases=torch.as_tensor(bias_array, dtype=torch.float32, device=device),
        thresholds=torch.as_tensor(
            threshold_array, dtype=torch.float32, device=device
        ),
    )


def _validate_runtime_inputs(
    torch: Any,
    gate: GateTensors,
    history: Any,
    actions: Any,
    current: Any,
    update: Any,
) -> None:
    assert_frozen_feature_inputs(history, actions, current, update)
    if current.dtype != torch.float32:
        raise RuntimeError("frozen solver/gate arithmetic must be float32")
    if gate.weights.device != current.device or gate.weights.dtype != current.dtype:
        raise RuntimeError("gate and solver tensor device/dtype mismatch")


def score_gate(
    torch: Any,
    gate: GateTensors,
    history: Any,
    actions: Any,
    current: Any,
    update: Any,
    stage: int,
) -> tuple[Any, Any, Any]:
    """Return ``(minimum_score, all_head_scores, causal_features)``."""

    if stage not in range(STAGES):
        raise ValueError("gate stage must be zero, one, or two")
    _validate_runtime_inputs(torch, gate, history, actions, current, update)
    features = build_counted_causal_features(history, actions, current, update)
    if features.shape != (len(history), FEATURE_DIM):
        raise RuntimeError("counted causal feature width drift")
    head_scores = features @ gate.weights[stage].transpose(0, 1)
    head_scores = head_scores + gate.biases[stage]
    score = torch.amin(head_scores, dim=1)
    if not bool(torch.isfinite(score).all().item()):
        raise RuntimeError("active robust gate score is nonfinite")
    return score, head_scores, features


def manual_sparse(
    torch: Any,
    solver: Any,
    gate: GateTensors,
    history: Any,
    actions: Any,
    base_prediction: Any,
) -> SparseExecution:
    """Actual adaptive path: evaluate and refine reached rows only."""

    if len(solver.adapters) != STAGES:
        raise RuntimeError("frozen stagewise solver must expose exactly three adapters")
    if base_prediction.shape != (len(history), LATENT_DIM):
        raise RuntimeError("base prediction shape drift")
    with torch.inference_mode():
        anchored = solver.anchor(history, actions, base_prediction)
        current = anchored[1]
        last_update = current - base_prediction
        _validate_runtime_inputs(
            torch, gate, history, actions, current, last_update
        )
        batch = len(history)
        calls = torch.ones(batch, dtype=torch.long, device=history.device)
        scores = torch.full(
            (batch, STAGES), float("nan"), dtype=current.dtype, device=current.device
        )
        head_scores = torch.full(
            (batch, STAGES, gate.head_count),
            float("nan"),
            dtype=current.dtype,
            device=current.device,
        )
        features = torch.full(
            (batch, STAGES, FEATURE_DIM),
            float("nan"),
            dtype=current.dtype,
            device=current.device,
        )
        stage_current = torch.full(
            (batch, STAGES, LATENT_DIM),
            float("nan"),
            dtype=current.dtype,
            device=current.device,
        )
        stage_update = torch.full_like(stage_current, float("nan"))
        reached = torch.zeros(
            (batch, STAGES), dtype=torch.bool, device=current.device
        )
        active = torch.arange(batch, device=current.device)
        for stage, adapter in enumerate(solver.adapters):
            if not active.numel():
                break
            local_current = current.index_select(0, active)
            local_update = last_update.index_select(0, active)
            local_score, local_heads, local_features = score_gate(
                torch,
                gate,
                history.index_select(0, active),
                actions.index_select(0, active),
                local_current,
                local_update,
                stage,
            )
            reached[active, stage] = True
            stage_current[active, stage] = local_current
            stage_update[active, stage] = local_update
            scores[active, stage] = local_score
            head_scores[active, stage] = local_heads
            features[active, stage] = local_features
            active = active[local_score > gate.thresholds[stage]]
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
    return SparseExecution(
        selected=current,
        calls=calls,
        scores=scores,
        head_scores=head_scores,
        features=features,
        stage_current=stage_current,
        stage_update=stage_update,
        reached=reached,
    )


def dense_shadow(
    torch: Any,
    solver: Any,
    gate: GateTensors,
    history: Any,
    actions: Any,
    base_prediction: Any,
) -> DenseExecution:
    """Full-batch all-exit shadow; never supplies the adaptive result."""

    with torch.inference_mode():
        outputs, updates = solver(
            history, actions, base_prediction, max_depth=DEPTHS, return_updates=True
        )
        exits = torch.stack([outputs[depth] for depth in (1, 2, 3, 4)], dim=1)
        stage_current = torch.stack(
            [outputs[depth] for depth in (1, 2, 3)], dim=1
        )
        stage_update = torch.stack(
            [updates[depth] for depth in (1, 2, 3)], dim=1
        )
        score_parts = []
        head_parts = []
        feature_parts = []
        for stage in range(STAGES):
            score, heads, features = score_gate(
                torch,
                gate,
                history,
                actions,
                stage_current[:, stage],
                stage_update[:, stage],
                stage,
            )
            score_parts.append(score)
            head_parts.append(heads)
            feature_parts.append(features)
        scores = torch.stack(score_parts, dim=1)
        head_scores = torch.stack(head_parts, dim=1)
        features = torch.stack(feature_parts, dim=1)
        calls = sequential_calls_from_scores(torch, scores, gate.thresholds)
    return DenseExecution(
        exits=exits,
        calls=calls,
        scores=scores,
        head_scores=head_scores,
        features=features,
        stage_current=stage_current,
        stage_update=stage_update,
    )


def sequential_calls_from_scores(torch: Any, scores: Any, thresholds: Any) -> Any:
    """Reconstruct calls with the frozen strict-greater continuation rule."""

    if scores.ndim != 2 or scores.shape[1] != STAGES:
        raise ValueError("scores must have shape [rows,3]")
    if thresholds.shape != (STAGES,):
        raise ValueError("thresholds must have shape [3]")
    calls = torch.ones(len(scores), dtype=torch.long, device=scores.device)
    active = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
    for stage in range(STAGES):
        if not bool(torch.isfinite(scores[active, stage]).all().item()):
            raise RuntimeError("nonfinite reached score during call reconstruction")
        active = active & (scores[:, stage] > thresholds[stage])
        calls += active.long()
    return calls


def reconstruct_features_from_primitives(
    torch: Any,
    history: Any,
    actions: Any,
    stage_current: Any,
    stage_update: Any,
    reached: Any | None = None,
) -> Any:
    """Rebuild stage features using only the persisted causal primitives."""

    batch = len(history)
    if stage_current.shape != (batch, STAGES, LATENT_DIM):
        raise ValueError("stage_current shape drift")
    if stage_update.shape != stage_current.shape:
        raise ValueError("stage_update shape drift")
    if reached is None:
        reached = torch.ones(
            (batch, STAGES), dtype=torch.bool, device=history.device
        )
    if reached.shape != (batch, STAGES) or reached.dtype != torch.bool:
        raise ValueError("reached shape or dtype drift")
    if bool((reached[:, 1] & ~reached[:, 0]).any().item()) or bool(
        (reached[:, 2] & ~reached[:, 1]).any().item()
    ):
        raise RuntimeError("reached mask is not sequential")
    reconstructed = torch.full(
        (batch, STAGES, FEATURE_DIM),
        float("nan"),
        dtype=history.dtype,
        device=history.device,
    )
    for stage in range(STAGES):
        active = torch.nonzero(reached[:, stage], as_tuple=False).flatten()
        if not active.numel():
            continue
        local = build_counted_causal_features(
            history.index_select(0, active),
            actions.index_select(0, active),
            stage_current[active, stage],
            stage_update[active, stage],
        )
        reconstructed[active, stage] = local
    return reconstructed


def reconstruct_scores_from_features(
    torch: Any,
    gate: GateTensors,
    features: Any,
    reached: Any | None = None,
) -> tuple[Any, Any]:
    """Rebuild all affine heads and their minimum from persisted features."""

    batch = len(features)
    if features.shape != (batch, STAGES, FEATURE_DIM):
        raise ValueError("feature persistence shape drift")
    if reached is None:
        reached = torch.ones(
            (batch, STAGES), dtype=torch.bool, device=features.device
        )
    head_scores = torch.full(
        (batch, STAGES, gate.head_count),
        float("nan"),
        dtype=features.dtype,
        device=features.device,
    )
    scores = torch.full(
        (batch, STAGES),
        float("nan"),
        dtype=features.dtype,
        device=features.device,
    )
    for stage in range(STAGES):
        active = torch.nonzero(reached[:, stage], as_tuple=False).flatten()
        if not active.numel():
            continue
        local_heads = (
            features[active, stage] @ gate.weights[stage].transpose(0, 1)
            + gate.biases[stage]
        )
        head_scores[active, stage] = local_heads
        scores[active, stage] = torch.amin(local_heads, dim=1)
    return scores, head_scores


def score_feature_tensor(torch: Any, gate: GateTensors, features: Any) -> tuple[Any, Any]:
    """Score a dense ``[rows,3,1046]`` feature tensor using runtime arithmetic."""

    if features.ndim != 3 or tuple(features.shape[1:]) != (STAGES, FEATURE_DIM):
        raise ValueError("candidate qualification features must have shape [rows,3,1046]")
    if features.dtype != torch.float32 or features.device != gate.weights.device:
        raise ValueError("candidate qualification features must match gate float32/device")
    heads = torch.stack(
        [
            features[:, stage] @ gate.weights[stage].transpose(0, 1)
            + gate.biases[stage]
            for stage in range(STAGES)
        ],
        dim=1,
    )
    return torch.amin(heads, dim=2), heads


def candidate_runtime_equivalence(
    torch: Any,
    *,
    features: np.ndarray,
    fitted: Mapping[str, np.ndarray],
    device: Any = "cpu",
) -> dict[str, Any]:
    """Qualify exact NumPy-float32 fit calls against Torch runtime for all 24 objects.

    This helper consumes causal features only.  It neither accepts nor opens a
    target, contact, loss, reward, success, or other outcome channel.
    """

    source_features = np.asarray(features)
    if source_features.ndim != 3 or source_features.shape[1:] != (
        STAGES,
        FEATURE_DIM,
    ):
        raise ValueError("candidate runtime qualification feature shape drift")
    if not np.isfinite(source_features).all():
        raise ValueError("candidate runtime qualification features are nonfinite")
    if source_features.dtype != np.dtype(np.float32):
        raise TypeError("candidate runtime qualification features must be exactly float32")
    source_features = np.ascontiguousarray(source_features)
    required = {
        "compiled_weights",
        "compiled_biases",
        "thresholds",
        "head_count",
        "candidate_ids",
        "architectures",
        "head_names",
    }
    missing = required - set(fitted)
    if missing:
        raise ValueError(f"fitted family lacks runtime fields: {sorted(missing)}")
    candidate_count = len(np.asarray(fitted["candidate_ids"]))
    if candidate_count != 24:
        raise ValueError("runtime equivalence requires the complete 24-candidate family")
    runtime_features = torch.as_tensor(
        np.asarray(source_features, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )

    def numpy_calls(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        calls = np.ones(len(scores), dtype=np.int64)
        active = np.ones(len(scores), dtype=bool)
        for stage in range(STAGES):
            if not np.isfinite(scores[active, stage]).all():
                raise RuntimeError("nonfinite reached fit score")
            active &= scores[:, stage] > thresholds[stage]
            calls += active.astype(np.int64)
        return calls

    records = []
    for index in range(candidate_count):
        head_count = int(np.asarray(fitted["head_count"])[index])
        architecture = str(np.asarray(fitted["architectures"])[index])
        candidate_id = str(np.asarray(fitted["candidate_ids"])[index])
        source_weights = np.asarray(fitted["compiled_weights"])[
            index, :, :head_count
        ]
        source_biases = np.asarray(fitted["compiled_biases"])[
            index, :, :head_count
        ]
        source_thresholds = np.asarray(fitted["thresholds"])[index]
        if any(
            value.dtype != np.dtype(np.float32)
            for value in (source_weights, source_biases, source_thresholds)
        ):
            raise TypeError("all fitted runtime weights, biases, and thresholds must be float32")
        source_weights = np.ascontiguousarray(source_weights)
        source_biases = np.ascontiguousarray(source_biases)
        source_names = tuple(
            str(item)
            for item in np.asarray(fitted["head_names"])[index, :head_count]
        )
        fit_scores = np.empty((len(source_features), STAGES), dtype=np.float32)
        for stage in range(STAGES):
            reference_heads = (
                np.einsum(
                    "nd,hd->nh",
                    source_features[:, stage],
                    source_weights[stage],
                    optimize=False,
                )
                + source_biases[stage][None, :]
            )
            fit_scores[:, stage] = reference_heads.min(axis=1)
        fit_calls = numpy_calls(fit_scores, source_thresholds)
        gate = gate_from_tensors(
            torch,
            architecture=architecture,
            candidate_id=candidate_id,
            head_names=source_names,
            weights=source_weights,
            biases=source_biases,
            thresholds=source_thresholds,
            device=device,
        )
        runtime_scores, _ = score_feature_tensor(torch, gate, runtime_features)
        runtime_calls = sequential_calls_from_scores(
            torch, runtime_scores, gate.thresholds
        )
        runtime_scores_np = runtime_scores.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        runtime_calls_np = runtime_calls.detach().cpu().numpy().astype(np.int64)
        maximum = float(np.max(np.abs(fit_scores - runtime_scores_np), initial=0.0))
        record = {
            "candidate_index": index,
            "candidate_id": candidate_id,
            "architecture": architecture,
            "head_count": head_count,
            "numpy_float32_vs_torch_float32_scores_allclose": bool(
                np.allclose(fit_scores, runtime_scores_np, rtol=2e-6, atol=2e-7)
            ),
            "numpy_float32_vs_torch_float32_score_max_abs": maximum,
            "numpy_float32_vs_torch_float32_calls_exact": bool(
                np.array_equal(fit_calls, runtime_calls_np)
            ),
        }
        # Different BLAS reduction trees can shift low bits.  The scientific
        # routing contract is exact calls; score deltas remain an audited
        # numerical diagnostic rather than an outcome-dependent filter.
        record["passed"] = bool(
            record["numpy_float32_vs_torch_float32_calls_exact"]
        )
        records.append(record)
    return {
        "schema_version": 1,
        "candidate_count": candidate_count,
        "feature_rows": len(source_features),
        "strict_continue_operator": ">",
        "records": records,
        "all_24_candidates_passed": all(item["passed"] for item in records),
        "passed": all(item["passed"] for item in records),
        "target_or_forbidden_channel_accepted": False,
    }


def tensors_exact_with_nan(torch: Any, left: Any, right: Any) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if not torch.is_floating_point(left):
        return bool(torch.equal(left, right))
    left_nan = torch.isnan(left)
    right_nan = torch.isnan(right)
    return bool(
        torch.equal(left_nan, right_nan)
        and torch.equal(left[~left_nan], right[~right_nan])
    )


def frozen_facade_source_audit() -> dict[str, Any]:
    observed = {str(path): sha256_file(path) for path in FROZEN_FACADE_SOURCES}
    bad = {
        str(path): {"expected": expected, "observed": observed[str(path)]}
        for path, expected in FROZEN_FACADE_SOURCES.items()
        if observed[str(path)] != expected
    }
    if bad:
        raise RuntimeError(f"frozen V5 model/refiner/preprocessing facade drift: {bad}")
    return {"passed": True, "sources": observed}


_V5_FACADE: Any | None = None


def load_frozen_v5_facade() -> Any:
    """Load only V5's frozen model/refiner/preprocessing entry points."""

    global _V5_FACADE
    frozen_facade_source_audit()
    if _V5_FACADE is None:
        if str(V5_ROOT) not in sys.path:
            sys.path.insert(0, str(V5_ROOT))
        spec = importlib.util.spec_from_file_location(
            "lewm_domain_robust_frozen_v5_facade", V5_ROOT / "runner.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot import frozen V5 runner facade")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _V5_FACADE = module
    return _V5_FACADE


def load_frozen_scientific_stack(device_name: str) -> tuple[Any, ...]:
    return load_frozen_v5_facade().load_scientific_stack(device_name)


def prepare_frozen_episode_tensors(
    torch: Any,
    runtime: Any,
    model_io: Any,
    device: Any,
    base: Any,
    contract: Any,
    distribution_common: Any,
    loaded: Mapping[str, np.ndarray],
) -> tuple[Any, Any, Any, Any]:
    return load_frozen_v5_facade().prepare_episode_tensors(
        torch,
        runtime,
        model_io,
        device,
        base,
        contract,
        distribution_common,
        dict(loaded),
    )


def _state_authorization(role: str) -> dict[str, Any]:
    expected = {
        "fit": "FIT_COHORTS",
        "selection": "SELECTION_COHORTS",
        "smoke": "EXCLUDED_MECHANICAL_SMOKE",
        "confirmation": "CONFIRMATION_EXECUTION",
    }
    if role not in expected:
        raise ValueError(f"unknown execution role: {role}")
    state = read_verified_controller()
    if state.get("active_attempt") != ATTEMPT_VERSION:
        raise RuntimeError(f"{ATTEMPT_VERSION} is not the active attempt")
    if state.get("current_state") != expected[role]:
        raise RuntimeError(
            f"role {role} is unauthorized in state {state.get('current_state')}"
        )
    seal_spec = {
        "fit": (INHERITANCE_SEAL_PATH, "PRE_OUTCOME_SEAL"),
        "selection": (
            SOURCE_PRE_SELECTION_SEAL_PATH,
            "PRE_SELECTION_SEAL",
        ),
        "smoke": (
            ATTEMPT_ROOT / "audit/pre_confirmation_package_seal.json",
            "PRE_CONFIRMATION_PACKAGE_SEAL",
        ),
        "confirmation": (
            ATTEMPT_ROOT / "audit/pre_confirmation_package_seal.json",
            "PRE_CONFIRMATION_PACKAGE_SEAL",
        ),
    }[role]
    seal_path, checkpoint_state = seal_spec
    seal_relative = relative_to_repo(seal_path)
    lineage: dict[str, Any] | None = None
    if role in ("fit", "selection"):
        try:
            lineage = verify_inherited_pre_data_authorization(
                expected_current_state=expected[role],
                direct_selection_seal_path=(
                    seal_path if role == "selection" else None
                ),
            )
        except InheritedAuthorizationError as error:
            raise RuntimeError(
                f"controller does not verify inherited {role} authorization"
            ) from error
        if lineage.get("science_attempt") != SCIENCE_ATTEMPT:
            raise RuntimeError("inherited authorization science identity drift")
        if role == "fit":
            if lineage.get("seal_path") != seal_relative:
                raise RuntimeError("inherited fit authorization path drift")
            seal_sha256 = str(lineage["seal_sha256"])
        else:
            direct = lineage.get("direct_selection_seal")
            if (
                not isinstance(direct, Mapping)
                or direct.get("path") != seal_relative
                or direct.get("checkpoint_state") != "PRE_SELECTION_SEAL"
                or direct.get("pre_data_seal_sha256")
                != lineage.get("seal_sha256")
            ):
                raise RuntimeError(
                    "direct selection seal does not bind inherited raw authorization"
                )
            seal_sha256 = str(direct["sha256"])
    else:
        seal = read_json(seal_path)
        if (
            seal.get("passed") is not True
            or seal.get("attempt") != ATTEMPT_VERSION
            or seal.get("checkpoint_state") != checkpoint_state
        ):
            raise RuntimeError(f"authorization seal invalid for {role}: {seal_path}")
        seal_sha256 = sha256_file(seal_path)
        checkpoint_matches = any(
            item.get("evidence_path") == seal_relative
            and item.get("evidence_sha256") == seal_sha256
            and item.get("source_attempt") == ATTEMPT_VERSION
            for item in state.get("verified_checkpoints", [])
        )
        if not checkpoint_matches:
            raise RuntimeError(
                f"controller does not verify the {role} authorization seal"
            )
    result = {
        "state": state["current_state"],
        "active_attempt": state["active_attempt"],
        "state_sha256": (
            str(lineage["authenticated_state_sha256"])
            if lineage is not None
            else sha256_file(STUDY_ROOT / "STATE.json")
        ),
        "seal_path": seal_relative,
        "seal_sha256": seal_sha256,
        "seal_checkpoint_state": checkpoint_state,
    }
    if lineage is not None:
        result.update(
            {
                "science_attempt": SCIENCE_ATTEMPT,
                "raw_authorization_seal_path": lineage["seal_path"],
                "raw_authorization_seal_sha256": lineage["seal_sha256"],
                "exact_locked_snapshot_unchanged": lineage[
                    "exact_locked_snapshot_unchanged"
                ],
            }
        )
    return result


def _role_paths(role: str, regime: str) -> tuple[Path, Path, Path]:
    root = ATTEMPT_ROOT / "data" / role / regime
    return (
        root / "raw_manifest.json",
        root / "execution",
        root / "execution_manifest.json",
    )


def _enforce_execution_runtime(device_name: str) -> dict[str, Any]:
    """Fail before authorization or arrays unless this is sealed MPS execution."""

    require_mps_device(device_name)
    result = verify_here(
        "sparse_execution", require_mps=True, include_external_hashes=True
    )
    if result.get("passed") is not True or result.get("mps", {}).get(
        "available"
    ) is not True:
        raise RuntimeError("sparse_execution runtime/MPS preflight did not pass")
    return result


def _assert_regular_unlinked(path: Path, *, label: str) -> None:
    try:
        info = Path(path).lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"required {label} is absent: {path}") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
    ):
        raise RuntimeError(f"{label} is linked/aliased or non-regular: {path}")


def _preflight_execution_namespace(
    role: str,
    regime: str,
    raw_manifest: Mapping[str, Any],
) -> None:
    """Reject linked/non-directory execution ancestry before mkdir or resume."""

    _, output_directory, manifest_path = _role_paths(role, regime)
    expected_root = ATTEMPT_ROOT / "data" / role / regime
    if output_directory != expected_root / "execution":
        raise RuntimeError("execution output root is not canonical")
    current = ATTEMPT_ROOT
    for part in output_directory.relative_to(ATTEMPT_ROOT).parts:
        current /= part
        if not os.path.lexists(current):
            continue
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(
                f"execution output parent is linked or non-directory: {current}"
            )
    leaves = [manifest_path]
    for record in raw_manifest["episodes"]:
        episode_id = str(record["episode_id"])
        leaves.extend(
            (
                output_directory / f"{episode_id}.npz",
                output_directory / f"{episode_id}.json",
            )
        )
    identities: dict[tuple[int, int], Path] = {}
    for path in leaves:
        if path.parent not in {expected_root, output_directory}:
            raise RuntimeError(f"execution output leaf escapes canonical root: {path}")
        if not os.path.lexists(path):
            continue
        _assert_regular_unlinked(path, label="execution output leaf")
        info = path.lstat()
        identity = (int(info.st_dev), int(info.st_ino))
        previous = identities.get(identity)
        if previous is not None and previous != path:
            raise RuntimeError(
                f"distinct execution leaves alias one inode: {previous}, {path}"
            )
        identities[identity] = path


def _expected_role_count(role: str, raw_count: int) -> int:
    fixed = {"fit": 300, "selection": 500, "smoke": 6}
    if role in fixed:
        return fixed[role]
    if role != "confirmation":
        raise ValueError(role)
    state = read_verified_controller()
    frozen = state.get("expected_confirmation_episodes_per_regime")
    if frozen is None:
        raise RuntimeError("controller has not frozen confirmation episodes per regime")
    value = int(frozen)
    if value not in range(500, 4_501, 500) or raw_count != value:
        raise RuntimeError(
            "confirmation raw count differs from the frozen common per-regime size"
        )
    return value


def _load_raw_manifest(
    role: str,
    regime: str,
    path: Path,
    *,
    inputs: GenerationInputs | None = None,
) -> dict[str, Any]:
    """Use the generator's sole ledger-exact, path-safe raw validator."""

    if regime not in REGIME_ORDER:
        raise ValueError(f"unknown DGP regime: {regime}")
    generation_inputs = inputs or load_generation_inputs(role, regime)
    manifest = validate_raw_manifest_contract(
        role, regime, generation_inputs, path=path
    )
    if manifest is None:
        raise RuntimeError(f"raw manifest is absent: {path}")
    expected = _expected_role_count(role, len(manifest["episodes"]))
    if expected != len(generation_inputs.primary):
        raise RuntimeError("raw manifest count differs from the sealed cohort prefix")
    return manifest


def _verified_file_link(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"required execution binding is absent: {path}")
    return {"path": relative_to_repo(path), "sha256": sha256_file(path)}


def _execution_source_bindings(
    raw_manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    gate: GateTensors | None = None,
) -> dict[str, Any]:
    """Bind every transitive source needed by the later input seal."""

    dgp_path = resolve_repo_relative(str(raw_manifest["dgp_matrix_path"]))
    ledger_path = resolve_repo_relative(str(raw_manifest["cohort_seed_ledger_path"]))
    if sha256_file(dgp_path) != raw_manifest["dgp_matrix_sha256"]:
        raise RuntimeError("raw manifest DGP matrix hash drift")
    if sha256_file(ledger_path) != raw_manifest["cohort_seed_ledger_sha256"]:
        raise RuntimeError("raw manifest cohort ledger hash drift")
    authorization_path = resolve_repo_relative(str(authorization["seal_path"]))
    if sha256_file(authorization_path) != authorization["seal_sha256"]:
        raise RuntimeError("execution authorization seal hash drift")
    bindings: dict[str, Any] = {
        "dgp_matrix": _verified_file_link(dgp_path),
        "cohort_seed_ledger": _verified_file_link(ledger_path),
        "execution_authorization_seal": _verified_file_link(authorization_path),
        "counted_feature_source": _verified_file_link(ATTEMPT_ROOT / "counted_features.py"),
        "pricing_source": _verified_file_link(ATTEMPT_ROOT / "flops.py"),
        "runner_source": _verified_file_link(Path(__file__)),
    }
    raw_authorization_value = authorization.get("raw_authorization_seal_path")
    if raw_authorization_value is not None:
        raw_authorization_path = resolve_repo_relative(
            str(raw_authorization_value)
        )
        if (
            sha256_file(raw_authorization_path)
            != authorization.get("raw_authorization_seal_sha256")
            or raw_manifest.get("authorization_seal", {}).get("path")
            != str(raw_authorization_value)
            or raw_manifest.get("authorization_seal", {}).get("sha256")
            != authorization.get("raw_authorization_seal_sha256")
        ):
            raise RuntimeError(
                "raw manifest/inherited authorization seal binding drift"
            )
        bindings["raw_authorization_seal"] = _verified_file_link(
            raw_authorization_path
        )
    registry_value = raw_manifest.get("replacement_registry_path")
    if registry_value:
        bindings["replacement_registry"] = _verified_file_link(
            resolve_repo_relative(str(registry_value))
        )
    if gate is not None:
        if gate.source_path is None or gate.source_sha256 is None:
            raise RuntimeError("production gate has no immutable source link")
        freeze_root = gate.source_path.parent
        gate_fit = freeze_root / "gate_fit.npz"
        compiler_manifest_path = freeze_root / "compiled_gate_manifest.json"
        gate_freeze_path = freeze_root / "gate_freeze.json"
        compiler_manifest = read_json(compiler_manifest_path)
        if compiler_manifest.get("compiled_gate_sha256") != gate.source_sha256:
            raise RuntimeError("compiler manifest does not bind the runtime gate")
        gate_freeze = read_json(gate_freeze_path)
        if (
            gate_freeze.get("compiled_gate_sha256") != gate.source_sha256
            or gate_freeze.get("gate_fit_sha256") != sha256_file(gate_fit)
        ):
            raise RuntimeError("gate freeze does not bind selected/compiled gate")
        fixed_whitening = (
            V5_ROOT / "freeze/whitening.npz"
        )
        if sha256_file(fixed_whitening) != (
            "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796"
        ):
            raise RuntimeError("unchanged V5 fixed-whitening hash drift")
        bindings.update(
            {
                "gate_fit": _verified_file_link(gate_fit),
                "compiled_gate": _verified_file_link(gate.source_path),
                "compiled_gate_manifest": _verified_file_link(
                    compiler_manifest_path
                ),
                "gate_freeze": _verified_file_link(gate_freeze_path),
                "fixed_whitening": _verified_file_link(fixed_whitening),
            }
        )
    return bindings


def _expected_part_array_contract(
    role: str, *, head_count: int | None = None
) -> dict[str, tuple[tuple[int, ...], str]]:
    development = {
        "episode_slot": ((ROWS_PER_EPISODE,), "int32"),
        "model_step": ((ROWS_PER_EPISODE,), "int16"),
        "target": ((ROWS_PER_EPISODE, LATENT_DIM), "float32"),
        "exits": ((ROWS_PER_EPISODE, DEPTHS, LATENT_DIM), "float32"),
        "production_features": (
            (ROWS_PER_EPISODE, STAGES, FEATURE_DIM),
            "float32",
        ),
        "history": (
            (ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM),
            "float32",
        ),
        "action_history": (
            (ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM),
            "float32",
        ),
        "stage_current": (
            (ROWS_PER_EPISODE, STAGES, LATENT_DIM),
            "float32",
        ),
        "stage_update": (
            (ROWS_PER_EPISODE, STAGES, LATENT_DIM),
            "float32",
        ),
    }
    if role in ("fit", "selection"):
        return development
    if role not in ("smoke", "confirmation") or head_count not in (2, 8):
        raise ValueError("execution part contract requires a valid role/head count")
    contract = {
        "episode_slot": ((ROWS_PER_EPISODE,), "int32"),
        "model_step": ((ROWS_PER_EPISODE,), "int16"),
        "exits": ((ROWS_PER_EPISODE, DEPTHS, LATENT_DIM), "float32"),
        "selected": ((ROWS_PER_EPISODE, LATENT_DIM), "float32"),
        "calls": ((ROWS_PER_EPISODE,), "int8"),
        "scores": ((ROWS_PER_EPISODE, STAGES), "float32"),
        "head_scores": (
            (ROWS_PER_EPISODE, STAGES, head_count),
            "float32",
        ),
        "production_features": (
            (ROWS_PER_EPISODE, STAGES, FEATURE_DIM),
            "float32",
        ),
        "history": (
            (ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM),
            "float32",
        ),
        "action_history": (
            (ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM),
            "float32",
        ),
        "stage_current": (
            (ROWS_PER_EPISODE, STAGES, LATENT_DIM),
            "float32",
        ),
        "stage_update": (
            (ROWS_PER_EPISODE, STAGES, LATENT_DIM),
            "float32",
        ),
        "reached": ((ROWS_PER_EPISODE, STAGES), "bool"),
        "dense_scores": ((ROWS_PER_EPISODE, STAGES), "float32"),
        "dense_head_scores": (
            (ROWS_PER_EPISODE, STAGES, head_count),
            "float32",
        ),
        "dense_stage_current": (
            (ROWS_PER_EPISODE, STAGES, LATENT_DIM),
            "float32",
        ),
        "dense_stage_update": (
            (ROWS_PER_EPISODE, STAGES, LATENT_DIM),
            "float32",
        ),
    }
    if role == "confirmation":
        contract["target"] = ((ROWS_PER_EPISODE, LATENT_DIM), "float32")
    return contract


def _verify_persisted_gate_semantics(
    torch: Any,
    gate: GateTensors,
    arrays: Mapping[str, np.ndarray],
    equivalence: Any,
) -> None:
    """Recompute the persisted causal gate chain before adopting a part."""

    device = gate.weights.device

    def tensor(name: str, *, dtype: Any | None = None) -> Any:
        return torch.as_tensor(arrays[name], dtype=dtype, device=device)

    history = tensor("history", dtype=torch.float32)
    actions = tensor("action_history", dtype=torch.float32)
    reached = tensor("reached", dtype=torch.bool)
    calls = tensor("calls", dtype=torch.long)
    sparse_current = tensor("stage_current", dtype=torch.float32)
    sparse_update = tensor("stage_update", dtype=torch.float32)
    sparse_features = tensor("production_features", dtype=torch.float32)
    sparse_scores = tensor("scores", dtype=torch.float32)
    sparse_heads = tensor("head_scores", dtype=torch.float32)

    reconstructed_sparse_features = reconstruct_features_from_primitives(
        torch,
        history,
        actions,
        sparse_current,
        sparse_update,
        reached,
    )
    reconstructed_sparse_scores, reconstructed_sparse_heads = (
        reconstruct_scores_from_features(
            torch, gate, reconstructed_sparse_features, reached
        )
    )
    if not tensors_exact_with_nan(
        torch, sparse_features, reconstructed_sparse_features
    ):
        raise RuntimeError("persisted sparse primitive/feature reconstruction drift")
    if not (
        tensors_exact_with_nan(
            torch, sparse_scores, reconstructed_sparse_scores
        )
        and tensors_exact_with_nan(
            torch, sparse_heads, reconstructed_sparse_heads
        )
    ):
        raise RuntimeError("persisted sparse feature/score reconstruction drift")

    dense_current = tensor("dense_stage_current", dtype=torch.float32)
    dense_update = tensor("dense_stage_update", dtype=torch.float32)
    reconstructed_dense_features = reconstruct_features_from_primitives(
        torch, history, actions, dense_current, dense_update
    )
    reconstructed_dense_scores, reconstructed_dense_heads = (
        reconstruct_scores_from_features(torch, gate, reconstructed_dense_features)
    )
    dense_scores = tensor("dense_scores", dtype=torch.float32)
    dense_heads = tensor("dense_head_scores", dtype=torch.float32)
    if not (
        bool(torch.equal(dense_scores, reconstructed_dense_scores))
        and bool(torch.equal(dense_heads, reconstructed_dense_heads))
    ):
        raise RuntimeError("persisted dense primitive/gate reconstruction drift")

    calls_from_sparse = sequential_calls_from_scores(
        torch, reconstructed_sparse_scores, gate.thresholds
    )
    calls_from_dense = sequential_calls_from_scores(
        torch, reconstructed_dense_scores, gate.thresholds
    )
    if not (
        bool(torch.equal(calls, calls_from_sparse))
        and bool(torch.equal(calls, calls_from_dense))
    ):
        raise RuntimeError("persisted calls do not follow the frozen gate")
    reconstructed_reached = (
        torch.arange(1, STAGES + 1, dtype=torch.long, device=device)[None, :]
        <= calls[:, None]
    )
    if not bool(torch.equal(reached, reconstructed_reached)):
        raise RuntimeError("persisted reached mask does not follow frozen calls")

    unreached = ~reached
    if not bool(
        torch.isnan(sparse_scores[unreached]).all().item()
        and torch.isnan(sparse_heads[unreached]).all().item()
        and torch.isnan(sparse_features[unreached]).all().item()
        and torch.isnan(sparse_current[unreached]).all().item()
        and torch.isnan(sparse_update[unreached]).all().item()
    ):
        raise RuntimeError("persisted unreached sparse values are not NaN")

    selected = tensor("selected", dtype=torch.float32)
    exits = tensor("exits", dtype=torch.float32)
    rows = torch.arange(len(calls), device=device)
    dense_selected = exits[rows, calls - 1]
    stopped_before_four = calls < DEPTHS
    if bool(stopped_before_four.any().item()):
        sparse_selected_from_current = sparse_current[
            rows[stopped_before_four], calls[stopped_before_four] - 1
        ]
        if not bool(
            torch.equal(
                selected[stopped_before_four], sparse_selected_from_current
            )
        ):
            raise RuntimeError("persisted selected output/stage-current drift")
    depth_one = calls == 1
    delta = (selected - dense_selected).abs()
    cross_batch_max_abs = float(delta.max().item())
    cross_batch_mean_abs = float(delta.mean().item())
    cross_batch_rmse = float(torch.sqrt(torch.square(delta).mean()).item())
    if not (
        (not bool(depth_one.any().item()) or bool(
            torch.equal(selected[depth_one], dense_selected[depth_one])
        ))
        and bool(
            torch.allclose(
                selected,
                dense_selected,
                rtol=NUMERICAL_RTOL,
                atol=NUMERICAL_ATOL,
            )
        )
        and cross_batch_max_abs <= NUMERICAL_MAX_ABS
    ):
        raise RuntimeError("persisted selected output/dense shadow drift")

    if not isinstance(equivalence, Mapping) or set(equivalence) != {
        "checks",
        "cross_batch_max_abs",
        "cross_batch_mean_abs",
        "cross_batch_rmse",
        "passed",
    }:
        raise RuntimeError("persisted numerical-equivalence schema drift")
    checks = equivalence.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != CONFIRMATION_EQUIVALENCE_CHECK_KEYS
        or any(value is not True for value in checks.values())
        or equivalence.get("passed") is not True
    ):
        raise RuntimeError("persisted numerical-equivalence checks drift")
    expected_metrics = {
        "cross_batch_max_abs": cross_batch_max_abs,
        "cross_batch_mean_abs": cross_batch_mean_abs,
        "cross_batch_rmse": cross_batch_rmse,
    }
    if any(equivalence.get(name) != value for name, value in expected_metrics.items()):
        raise RuntimeError("persisted numerical-equivalence metric drift")


def _replay_context(value: tuple[Any, ...]) -> ScientificReplayContext:
    return ScientificReplayContext(
        torch=value[0],
        runtime=value[1],
        model_io=value[2],
        device=value[3],
        base=value[4],
        contract=value[5],
        solver=value[6],
        distribution_common=value[10],
        module_before=value[11],
    )


def _verify_replayed_part_semantics(
    *,
    role: str,
    raw_record: Mapping[str, Any],
    loaded_inputs: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    sidecar: Mapping[str, Any],
    replay: ScientificReplayContext,
    gate: GateTensors | None,
) -> None:
    """Rerun the frozen stack from raw inputs before adopting any cached part."""

    torch = replay.torch
    history, actions, target, base_prediction = prepare_frozen_episode_tensors(
        torch,
        replay.runtime,
        replay.model_io,
        replay.device,
        replay.base,
        replay.contract,
        replay.distribution_common,
        loaded_inputs,
    )

    expected: dict[str, np.ndarray] = {
        "episode_slot": np.full(
            ROWS_PER_EPISODE, int(raw_record["slot"]), dtype=np.int32
        ),
        "model_step": np.arange(3, 41, dtype=np.int16),
        "history": history.detach().cpu().numpy().astype(np.float32),
        "action_history": actions.detach().cpu().numpy().astype(np.float32),
    }
    if role in ("fit", "selection"):
        exits, features, stage_current, stage_update = _dense_development_trace(
            torch, replay.solver, history, actions, base_prediction
        )
        reconstructed = reconstruct_features_from_primitives(
            torch, history, actions, stage_current, stage_update
        )
        if not bool(torch.equal(features, reconstructed)):
            raise RuntimeError("replayed development primitive/feature drift")
        expected.update(
            {
                "target": target.detach().cpu().numpy().astype(np.float32),
                "exits": exits.detach().cpu().numpy().astype(np.float32),
                "production_features": features.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "stage_current": stage_current.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "stage_update": stage_update.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
            }
        )
        difference = expected["exits"].astype(np.float64) - expected[
            "target"
        ].astype(np.float64)[:, None, :]
        raw_endpoint = np.square(difference).mean(axis=2)
        raw_gain = raw_endpoint[:, :STAGES] - raw_endpoint[:, 1:]
        if (
            raw_endpoint.shape != (ROWS_PER_EPISODE, DEPTHS)
            or raw_gain.shape != (ROWS_PER_EPISODE, STAGES)
            or not np.isfinite(raw_endpoint).all()
            or not np.isfinite(raw_gain).all()
        ):
            raise RuntimeError("replayed development endpoint/gain drift")
    else:
        if gate is None:
            raise RuntimeError("robust replay lacks the frozen compiled gate")
        dense = dense_shadow(
            torch, replay.solver, gate, history, actions, base_prediction
        )
        sparse = manual_sparse(
            torch, replay.solver, gate, history, actions, base_prediction
        )
        repeat = manual_sparse(
            torch, replay.solver, gate, history, actions, base_prediction
        )
        equivalence = _confirmation_equivalence(
            torch,
            replay.solver,
            gate,
            history,
            actions,
            base_prediction,
            sparse,
            repeat,
            dense,
        )
        if equivalence.get("passed") is not True:
            raise RuntimeError("replayed sparse/dense numerical contract failed")
        expected.update(
            {
                "exits": dense.exits.detach().cpu().numpy().astype(np.float32),
                "selected": sparse.selected.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "calls": sparse.calls.detach().cpu().numpy().astype(np.int8),
                "scores": sparse.scores.detach().cpu().numpy().astype(np.float32),
                "head_scores": sparse.head_scores.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "production_features": sparse.features.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "stage_current": sparse.stage_current.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "stage_update": sparse.stage_update.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "reached": sparse.reached.detach().cpu().numpy().astype(np.bool_),
                "dense_scores": dense.scores.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "dense_head_scores": dense.head_scores.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "dense_stage_current": dense.stage_current.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
                "dense_stage_update": dense.stage_update.detach()
                .cpu()
                .numpy()
                .astype(np.float32),
            }
        )
        if role == "confirmation":
            expected["target"] = target.detach().cpu().numpy().astype(np.float32)
        if sidecar.get("equivalence") != equivalence:
            raise RuntimeError("persisted/replayed numerical-equivalence drift")

    if set(expected) != set(arrays):
        raise RuntimeError("replayed execution member-key drift")
    for name, expected_value in expected.items():
        observed = arrays[name]
        if not np.array_equal(observed, expected_value, equal_nan=True):
            raise RuntimeError(f"persisted/replayed execution array drift: {name}")


def _execution_record_if_valid(
    part_path: Path,
    sidecar_path: Path,
    *,
    role: str,
    regime: str,
    raw_record: Mapping[str, Any],
    head_count: int | None = None,
    compiled_gate_path: Path | None = None,
    compiled_gate_sha256: str | None = None,
    runtime_torch: Any | None = None,
    runtime_gate: GateTensors | None = None,
    scientific_replay: ScientificReplayContext | None = None,
) -> dict[str, Any] | None:
    if not part_path.exists() and not sidecar_path.exists():
        return None
    if not (part_path.is_file() and sidecar_path.is_file()):
        raise RuntimeError(f"partial execution artifact: {part_path}")
    _assert_regular_unlinked(part_path, label="execution part")
    _assert_regular_unlinked(sidecar_path, label="execution sidecar")
    sidecar_bytes = sidecar_path.read_bytes()
    try:
        sidecar = json.loads(sidecar_bytes)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"execution sidecar is invalid JSON: {sidecar_path}") from error
    if not isinstance(sidecar, dict) or sidecar_bytes != (
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"):
        raise RuntimeError("execution sidecar byte/object canonicalization drift")
    expected_contract = _expected_part_array_contract(
        role, head_count=head_count
    )
    common_sidecar_fields = {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "role",
        "regime",
        "slot",
        "episode_id",
        "raw_path",
        "raw_sha256",
        "raw_sidecar_path",
        "raw_sidecar_sha256",
        "input_loader_audit",
        "contact_or_privileged_materialized",
        "no_gradients",
        "complete",
        "path",
        "sha256",
        "bytes",
        "arrays",
    }
    if role in ("fit", "selection"):
        expected_sidecar_fields = common_sidecar_fields | {
            "evaluation",
            "target_use",
            "primitive_feature_reconstruction_exact",
        }
    else:
        expected_sidecar_fields = common_sidecar_fields | {
            "compiled_gate_path",
            "compiled_gate_sha256",
            "architecture",
            "candidate_id",
            "head_names",
            "actual_adaptive_output",
            "dense_role",
            "call_histogram",
            "exact_compute",
            "equivalence",
            "target_persisted",
            "target_loss_computed",
        }
    if set(sidecar) != expected_sidecar_fields:
        raise RuntimeError(
            "execution sidecar schema drift: "
            f"missing={sorted(expected_sidecar_fields - set(sidecar))} "
            f"extra={sorted(set(sidecar) - expected_sidecar_fields)}"
        )
    raw_path = resolve_repo_relative(str(raw_record["raw_path"]))
    raw_sidecar_path = raw_path.with_suffix(".json")
    _assert_regular_unlinked(raw_sidecar_path, label="raw episode sidecar")
    loaded_inputs, recomputed_loader_audit = load_model_gate_inputs(
        raw_path, expected_sha256=str(raw_record["raw_sha256"])
    )
    expected_raw_arrays = {
        name: {
            "shape": list(loaded_inputs[name].shape),
            "dtype": str(loaded_inputs[name].dtype),
            "sha256": recomputed_loader_audit["array_sha256"][name],
        }
        for name in sorted(loaded_inputs)
    }
    if (
        raw_record.get("input_loader_audit") != recomputed_loader_audit
        or raw_record.get("arrays") != expected_raw_arrays
    ):
        raise RuntimeError("raw record/input-loader audit cross-link drift")
    header_checks = {
        "schema": type(sidecar.get("schema_version")) is int
        and sidecar.get("schema_version") == 1,
        "timestamp": type(sidecar.get("created_unix_ns")) is int
        and int(sidecar["created_unix_ns"]) > 0,
        "complete": sidecar.get("complete") is True,
        "attempt": sidecar.get("attempt") == ATTEMPT_VERSION,
        "role": sidecar.get("role") == role,
        "regime": sidecar.get("regime") == regime,
        "slot": int(sidecar.get("slot", -1)) == int(raw_record["slot"]),
        "episode_id": str(sidecar.get("episode_id"))
        == str(raw_record["episode_id"]),
        "raw_path": sidecar.get("raw_path") == raw_record["raw_path"],
        "raw_sha256": sidecar.get("raw_sha256") == raw_record["raw_sha256"],
        "raw_sidecar_path": sidecar.get("raw_sidecar_path")
        == relative_to_repo(raw_sidecar_path),
        "raw_sidecar_sha256": sidecar.get("raw_sidecar_sha256")
        == sha256_file(raw_sidecar_path),
        "input_loader_audit": sidecar.get("input_loader_audit")
        == recomputed_loader_audit,
        "contact_free": sidecar.get("contact_or_privileged_materialized") is False,
        "no_gradients": sidecar.get("no_gradients") is True,
    }
    if role in ("fit", "selection"):
        header_checks.update(
            {
                "evaluation": sidecar.get("evaluation")
                == "four dense exits and all three frozen causal feature stages",
                "target_use": sidecar.get("target_use")
                == f"permitted only for isolated {role}",
                "primitive_reconstruction": sidecar.get(
                    "primitive_feature_reconstruction_exact"
                )
                is True,
            }
        )
    if role in ("smoke", "confirmation"):
        if compiled_gate_path is None:
            raise RuntimeError("execution sidecar lacks the expected compiled gate")
        gate_path = Path(compiled_gate_path).resolve()
        _assert_regular_unlinked(gate_path, label="compiled gate")
        if sha256_file(gate_path) != compiled_gate_sha256:
            raise RuntimeError("execution sidecar compiled-gate source drift")
        with np.load(gate_path, allow_pickle=False) as stored_gate:
            if set(stored_gate.files) != COMPILED_KEYS:
                raise RuntimeError("execution sidecar compiled-gate key drift")
            gate_architecture = _scalar_text(
                stored_gate["architecture"], "architecture"
            )
            gate_candidate_id = _scalar_text(
                stored_gate["candidate_id"], "candidate_id"
            )
            gate_head_names = [
                str(item) for item in stored_gate["head_names"].tolist()
            ]
            gate_weights = stored_gate["weights"].copy()
            gate_biases = stored_gate["biases"].copy()
            gate_thresholds = stored_gate["thresholds"].copy()
        if runtime_torch is None or runtime_gate is None:
            raise RuntimeError("robust execution resume lacks its runtime gate")
        runtime_gate_checks = {
            "source_path": runtime_gate.source_path is not None
            and Path(runtime_gate.source_path).resolve() == gate_path,
            "source_hash": runtime_gate.source_sha256 == compiled_gate_sha256,
            "architecture": runtime_gate.architecture == gate_architecture,
            "candidate_id": runtime_gate.candidate_id == gate_candidate_id,
            "head_names": list(runtime_gate.head_names) == gate_head_names,
            "head_count": runtime_gate.head_count == head_count,
            "weights": array_sha256(
                runtime_gate.weights.detach().cpu().numpy()
            )
            == array_sha256(gate_weights),
            "biases": array_sha256(
                runtime_gate.biases.detach().cpu().numpy()
            )
            == array_sha256(gate_biases),
            "thresholds": array_sha256(
                runtime_gate.thresholds.detach().cpu().numpy()
            )
            == array_sha256(gate_thresholds),
        }
        if not all(runtime_gate_checks.values()):
            raise RuntimeError(
                f"execution resume runtime-gate drift: {runtime_gate_checks}"
            )
        expected_gate_path = relative_to_repo(gate_path)
        header_checks.update(
            {
                "compiled_gate_path": sidecar.get("compiled_gate_path")
                == expected_gate_path,
                "compiled_gate_sha256": sidecar.get("compiled_gate_sha256")
                == compiled_gate_sha256,
                "architecture": sidecar.get("architecture")
                == gate_architecture,
                "candidate_id": sidecar.get("candidate_id")
                == gate_candidate_id,
                "head_names": sidecar.get("head_names") == gate_head_names
                and len(gate_head_names) == head_count,
                "actual_adaptive_output": sidecar.get("actual_adaptive_output")
                == "manual sequential sparse reached-row execution",
                "dense_role": sidecar.get("dense_role")
                == "numerical shadow and fixed-depth comparator only",
                "equivalence": isinstance(sidecar.get("equivalence"), Mapping)
                and sidecar["equivalence"].get("passed") is True,
                "target_persistence": sidecar.get("target_persisted")
                is (role == "confirmation"),
                "target_loss": sidecar.get("target_loss_computed") is False,
            }
        )
        histogram = sidecar.get("call_histogram")
        header_checks["call_histogram"] = (
            isinstance(histogram, list)
            and len(histogram) == DEPTHS
            and all(isinstance(value, int) and value >= 0 for value in histogram)
            and sum(histogram) == ROWS_PER_EPISODE
        )
    if not all(header_checks.values()):
        failed = sorted(name for name, passed in header_checks.items() if not passed)
        raise RuntimeError(f"execution sidecar resume contract failed: {failed}")
    if sidecar.get("path") != relative_to_repo(part_path):
        raise RuntimeError("execution sidecar path drift")
    if sidecar.get("sha256") != sha256_file(part_path):
        raise RuntimeError("execution part hash drift")
    if int(sidecar.get("bytes", -1)) != part_path.stat().st_size:
        raise RuntimeError("execution part byte count drift")
    array_metadata = sidecar.get("arrays")
    if not isinstance(array_metadata, Mapping) or set(array_metadata) != set(
        expected_contract
    ):
        raise RuntimeError("execution sidecar array-key contract drift")
    for name, (shape, dtype) in expected_contract.items():
        metadata = array_metadata[name]
        if (
            not isinstance(metadata, Mapping)
            or set(metadata) != {"shape", "dtype", "sha256"}
            or metadata.get("shape") != list(shape)
            or metadata.get("dtype") != dtype
            or not isinstance(metadata.get("sha256"), str)
            or len(metadata["sha256"]) != 64
        ):
            raise RuntimeError(f"execution sidecar array metadata drift: {name}")
    observed_arrays: dict[str, np.ndarray] = {}
    with np.load(part_path, allow_pickle=False) as stored:
        if set(stored.files) != set(expected_contract):
            raise RuntimeError("execution part member-key contract drift")
        for name, (shape, dtype) in expected_contract.items():
            observed = stored[name].copy()
            if observed.shape != shape or str(observed.dtype) != dtype:
                raise RuntimeError(f"execution part array shape/dtype drift: {name}")
            observed_hash = array_sha256(observed)
            if observed_hash != array_metadata[name]["sha256"]:
                raise RuntimeError(f"execution part array hash drift: {name}")
            observed_arrays[name] = observed
    if not np.array_equal(
        observed_arrays["episode_slot"],
        np.full(ROWS_PER_EPISODE, int(raw_record["slot"]), dtype=np.int32),
    ):
        raise RuntimeError("execution part episode-slot identity drift")
    if not np.array_equal(
        observed_arrays["model_step"], np.arange(3, 41, dtype=np.int16)
    ):
        raise RuntimeError("execution part model-step identity drift")
    if role in ("smoke", "confirmation"):
        calls = observed_arrays["calls"]
        try:
            observed_compute = exact_compute(calls, int(head_count))
        except (TypeError, ValueError) as error:
            raise RuntimeError("execution part call-depth drift") from error
        histogram = np.bincount(calls.astype(np.int64), minlength=5)[1:5].tolist()
        if sidecar.get("call_histogram") != histogram:
            raise RuntimeError("execution sidecar/calls histogram drift")
        if sidecar.get("exact_compute") != observed_compute:
            raise RuntimeError("execution sidecar/calls exact-compute drift")
        expected_reached = (
            np.arange(1, STAGES + 1, dtype=np.int8)[None, :]
            <= calls[:, None]
        )
        if not np.array_equal(observed_arrays["reached"], expected_reached):
            raise RuntimeError("execution part reached/calls drift")
        _verify_persisted_gate_semantics(
            runtime_torch,
            runtime_gate,
            observed_arrays,
            sidecar.get("equivalence"),
        )
    if scientific_replay is None:
        raise RuntimeError("execution resume lacks frozen scientific replay context")
    if scientific_replay.module_before.get("passed") is not True:
        raise RuntimeError("scientific replay module audit did not pass")
    _verify_replayed_part_semantics(
        role=role,
        raw_record=raw_record,
        loaded_inputs=loaded_inputs,
        arrays=observed_arrays,
        sidecar=sidecar,
        replay=scientific_replay,
        gate=runtime_gate,
    )
    return {
        "slot": int(sidecar["slot"]),
        "episode_id": str(sidecar["episode_id"]),
        "path": sidecar["path"],
        "sha256": sidecar["sha256"],
        "sidecar_path": relative_to_repo(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "source_raw_path": sidecar.get("raw_path"),
        "source_raw_sha256": sidecar.get("raw_sha256"),
        "source_raw_sidecar_path": sidecar.get("raw_sidecar_path"),
        "source_raw_sidecar_sha256": sidecar.get("raw_sidecar_sha256"),
        "call_histogram": sidecar.get("call_histogram"),
    }


def _array_contract(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_sha256(value),
        }
        for key, value in sorted(arrays.items())
    }


def _write_execution_part(
    part_path: Path,
    sidecar_path: Path,
    arrays: Mapping[str, np.ndarray],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    atomic_npz(part_path, arrays, exclusive=True)
    record = {
        **dict(sidecar),
        "complete": True,
        "path": relative_to_repo(part_path),
        "sha256": sha256_file(part_path),
        "bytes": part_path.stat().st_size,
        "arrays": _array_contract(arrays),
    }
    atomic_json(sidecar_path, record, exclusive=True)
    return {
        "slot": int(record["slot"]),
        "episode_id": str(record["episode_id"]),
        "path": record["path"],
        "sha256": record["sha256"],
        "sidecar_path": relative_to_repo(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "source_raw_path": record.get("raw_path"),
        "source_raw_sha256": record.get("raw_sha256"),
        "source_raw_sidecar_path": record.get("raw_sidecar_path"),
        "source_raw_sidecar_sha256": record.get("raw_sidecar_sha256"),
        "call_histogram": record.get("call_histogram"),
    }


def _dense_development_trace(
    torch: Any,
    solver: Any,
    history: Any,
    actions: Any,
    base_prediction: Any,
) -> tuple[Any, Any, Any, Any]:
    with torch.inference_mode():
        outputs, updates = solver(
            history, actions, base_prediction, max_depth=DEPTHS, return_updates=True
        )
        exits = torch.stack([outputs[depth] for depth in (1, 2, 3, 4)], dim=1)
        stage_current = torch.stack(
            [outputs[depth] for depth in (1, 2, 3)], dim=1
        )
        stage_update = torch.stack(
            [updates[depth] for depth in (1, 2, 3)], dim=1
        )
        features = torch.stack(
            [
                build_counted_causal_features(
                    history,
                    actions,
                    stage_current[:, stage],
                    stage_update[:, stage],
                )
                for stage in range(STAGES)
            ],
            dim=1,
        )
    return exits, features, stage_current, stage_update


def _write_or_adopt_exact_npz(
    output: Path,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Write once or adopt an interruption-complete NPZ after exact verification."""

    expected = {
        name: np.ascontiguousarray(value) for name, value in arrays.items()
    }

    def verify_existing(status: str) -> dict[str, Any]:
        with np.load(output, allow_pickle=False) as stored:
            if set(stored.files) != set(expected):
                raise RuntimeError("existing aggregate development key drift")
            observed_hashes: dict[str, str] = {}
            for name, expected_value in expected.items():
                observed = stored[name].copy()
                if (
                    observed.shape != expected_value.shape
                    or observed.dtype != expected_value.dtype
                    or not np.array_equal(observed, expected_value, equal_nan=True)
                ):
                    raise RuntimeError(
                        f"existing aggregate development content drift: {name}"
                    )
                observed_hash = array_sha256(observed)
                expected_hash = array_sha256(expected_value)
                if observed_hash != expected_hash:
                    raise RuntimeError(
                        f"existing aggregate development array hash drift: {name}"
                    )
                observed_hashes[name] = observed_hash
        return {
            "status": status,
            "exact_keys_verified": True,
            "exact_shapes_dtypes_content_verified": True,
            "array_sha256": dict(sorted(observed_hashes.items())),
            "file_sha256": sha256_file(output),
        }

    if output.exists():
        return verify_existing("existing_interruption_complete_aggregate_adopted")
    try:
        atomic_npz(output, expected, exclusive=True)
    except FileExistsError:
        # A concurrent identical writer is acceptable only after the same
        # complete content verification; a partial or divergent file fails.
        return verify_existing("concurrent_complete_aggregate_adopted")
    return verify_existing("new_complete_aggregate_written_and_verified")


def _write_development_aggregate(
    role: str,
    regime: str,
    records: list[Mapping[str, Any]],
) -> tuple[Path, str, dict[str, Any]]:
    """Create the five-key per-DGP archive consumed by isolated fit/selection."""

    output = ATTEMPT_ROOT / "data" / role / regime / "role.npz"
    required = (
        "episode_slot",
        "model_step",
        "target",
        "exits",
        "production_features",
    )
    parts: dict[str, list[np.ndarray]] = {key: [] for key in required}
    for record in records:
        part = resolve_repo_relative(str(record["path"]))
        if sha256_file(part) != record["sha256"]:
            raise RuntimeError("development part drift before aggregation")
        with np.load(part, allow_pickle=False) as stored:
            missing = set(required) - set(stored.files)
            if missing:
                raise RuntimeError(
                    f"development part lacks aggregate keys: {sorted(missing)}"
                )
            for key in required:
                parts[key].append(stored[key].copy())
    arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    expected_rows = len(records) * ROWS_PER_EPISODE
    if any(len(value) != expected_rows for value in arrays.values()):
        raise RuntimeError("aggregate development row count drift")
    adoption = _write_or_adopt_exact_npz(output, arrays)
    return output, sha256_file(output), adoption


def _validate_existing_execution_manifest(
    *,
    role: str,
    regime: str,
    path: Path,
    output_directory: Path,
    raw_manifest_path: Path,
    raw_manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    gate_path: Path | None = None,
    runtime_torch: Any | None = None,
    runtime_gate: GateTensors | None = None,
    scientific_replay: ScientificReplayContext | None = None,
) -> dict[str, Any] | None:
    """Fully validate a completed execution before an idempotent return."""

    if not os.path.lexists(path):
        return None
    _assert_regular_unlinked(path, label="execution manifest")
    value = read_json(path)
    common_fields = {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "role",
        "regime",
        "complete",
        "episode_count",
        "row_count",
        "episodes",
        "raw_manifest_path",
        "raw_manifest_sha256",
        "source_raw_manifest_sha256",
        "authorization",
        "source_bindings",
        "loaded_input_keys",
        "contact_or_privileged_materialized",
        "causal_primitive_contract",
        "module_before",
        "module_after",
        "base_provenance",
        "frozen_facade",
        "no_gradients",
    }
    if role in ("fit", "selection"):
        expected_fields = common_fields | {
            "aggregate_role_path",
            "aggregate_role_sha256",
            "aggregate_recovery",
            "aggregate_role_keys",
            "target_role_isolation",
        }
    elif role in ("smoke", "confirmation"):
        expected_fields = common_fields | {
            "compiled_gate_path",
            "compiled_gate_sha256",
            "architecture",
            "candidate_id",
            "head_names",
            "head_count",
            "gate_cost_per_reached_decision",
            "exact_compute",
            "actual_adaptive_output",
            "dense_role",
            "strict_continue_operator",
            "target_persisted",
            "target_loss_computed_during_execution",
            "numerical_contract",
            "all_equivalence_checks_passed",
        }
    else:
        raise ValueError(f"unsupported execution role: {role}")
    if set(value) != expected_fields:
        raise RuntimeError(
            "existing execution manifest schema drift: "
            f"missing={sorted(expected_fields - set(value))} "
            f"extra={sorted(set(value) - expected_fields)}"
        )
    episodes = value.get("episodes")
    module_before = value.get("module_before")
    module_after = value.get("module_after")
    frozen_facade = value.get("frozen_facade")
    checks = {
        "schema": type(value.get("schema_version")) is int
        and value.get("schema_version") == 1,
        "timestamp": type(value.get("created_unix_ns")) is int
        and int(value["created_unix_ns"]) > 0,
        "attempt": value.get("attempt") == ATTEMPT_VERSION,
        "role": value.get("role") == role,
        "regime": value.get("regime") == regime,
        "complete": value.get("complete") is True,
        "episode_count": value.get("episode_count")
        == len(raw_manifest["episodes"]),
        "row_count": value.get("row_count")
        == len(raw_manifest["episodes"]) * ROWS_PER_EPISODE,
        "episodes": isinstance(episodes, list),
        "raw_manifest_path": value.get("raw_manifest_path")
        == relative_to_repo(raw_manifest_path),
        "raw_manifest_sha256": value.get("raw_manifest_sha256")
        == sha256_file(raw_manifest_path),
        "source_raw_manifest_sha256": value.get("source_raw_manifest_sha256")
        == sha256_file(raw_manifest_path),
        "authorization": value.get("authorization") == dict(authorization),
        "loaded_input_keys": value.get("loaded_input_keys") == ["action", "pixels"],
        "contact_free": value.get("contact_or_privileged_materialized") is False,
        "no_gradients": value.get("no_gradients") is True,
        "module_immutable": isinstance(module_before, Mapping)
        and module_before == module_after
        and module_before.get("passed") is True,
        "module_replayed": scientific_replay is not None
        and module_before == scientific_replay.module_before,
        "base_provenance": isinstance(value.get("base_provenance"), Mapping),
        "frozen_facade": isinstance(frozen_facade, Mapping)
        and frozen_facade.get("passed") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"existing execution manifest header drift: {checks}")

    head_count: int | None = None
    compiled_sha256: str | None = None
    compiled_gate: GateTensors | None = None
    if role in ("smoke", "confirmation"):
        if gate_path is None:
            raise RuntimeError("robust-gate resume lacks the compiled gate path")
        gate_path = Path(gate_path).resolve()
        _assert_regular_unlinked(gate_path, label="compiled gate")
        compiled_sha256 = sha256_file(gate_path)
        with np.load(gate_path, allow_pickle=False) as stored:
            if set(stored.files) != COMPILED_KEYS:
                raise RuntimeError("resume compiled-gate member-key drift")
            if int(np.asarray(stored["schema_version"]).reshape(())) != 1:
                raise RuntimeError("resume compiled-gate schema drift")
            architecture = _scalar_text(stored["architecture"], "architecture")
            candidate_id = _scalar_text(stored["candidate_id"], "candidate_id")
            weights = stored["weights"].copy()
            biases = stored["biases"].copy()
            thresholds = stored["thresholds"].copy()
            head_names_array = stored["head_names"].copy()
        gate_validation = validate_compiled_arrays(
            architecture=architecture,
            candidate_id=candidate_id,
            weights=weights,
            biases=biases,
            thresholds=thresholds,
            head_names=head_names_array,
            require_float32=True,
        )
        head_count = int(gate_validation["head_count"])
        compiled_gate = GateTensors(
            architecture=architecture,
            candidate_id=candidate_id,
            head_names=tuple(gate_validation["head_names"]),
            weights=weights,
            biases=biases,
            thresholds=thresholds,
            source_path=gate_path,
            source_sha256=compiled_sha256,
        )
        gate_checks = {
            "gate_path": value.get("compiled_gate_path")
            == relative_to_repo(gate_path),
            "gate_hash": value.get("compiled_gate_sha256") == compiled_sha256,
            "architecture": value.get("architecture") == architecture,
            "candidate_id": value.get("candidate_id") == candidate_id,
            "head_names": value.get("head_names")
            == list(gate_validation["head_names"]),
            "head_count": value.get("head_count") == head_count,
            "gate_cost": value.get("gate_cost_per_reached_decision")
            == gate_cost(head_count),
            "actual_adaptive_output": value.get("actual_adaptive_output")
            == "manual sequential sparse reached-row execution",
            "dense_role": value.get("dense_role")
            == "numerical shadow and fixed-depth comparator only",
            "strict_continue": value.get("strict_continue_operator") == ">",
            "target_persistence": value.get("target_persisted")
            is (role == "confirmation"),
            "target_loss": value.get("target_loss_computed_during_execution")
            is False,
            "equivalence": value.get("all_equivalence_checks_passed") is True,
            "numerical_contract": value.get("numerical_contract")
            == {
                "rtol": NUMERICAL_RTOL,
                "atol": NUMERICAL_ATOL,
                "maximum_absolute_error_ceiling": NUMERICAL_MAX_ABS,
                "calls_and_same_path_sparse_values_exact": True,
            },
            "causal_contract": value.get("causal_primitive_contract")
            == {
                "history": [ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM],
                "action_history": [ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM],
                "stage_current": [ROWS_PER_EPISODE, STAGES, LATENT_DIM],
                "stage_update": [ROWS_PER_EPISODE, STAGES, LATENT_DIM],
                "reached": [ROWS_PER_EPISODE, STAGES],
                "unreached_sparse_values": "NaN",
                "dense_stage_current_and_update_also_persisted": True,
                "sufficient_for_independent_feature_and_score_reconstruction": True,
            },
        }
        if not all(gate_checks.values()):
            raise RuntimeError(f"existing compiled-gate binding drift: {gate_checks}")
    else:
        development_checks = {
            "aggregate_keys": value.get("aggregate_role_keys")
            == [
                "episode_slot",
                "model_step",
                "target",
                "exits",
                "production_features",
            ],
            "aggregate_recovery": isinstance(value.get("aggregate_recovery"), Mapping),
            "target_role_isolation": value.get("target_role_isolation") == role,
            "causal_contract": value.get("causal_primitive_contract")
            == {
                "history": [ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM],
                "action_history": [ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM],
                "stage_current": [ROWS_PER_EPISODE, STAGES, LATENT_DIM],
                "stage_update": [ROWS_PER_EPISODE, STAGES, LATENT_DIM],
                "sufficient_for_independent_feature_reconstruction": True,
            },
        }
        if not all(development_checks.values()):
            raise RuntimeError(
                f"existing development manifest invariant drift: {development_checks}"
            )

    expected_source_bindings = _execution_source_bindings(
        raw_manifest, authorization, compiled_gate
    )
    source_bindings = value.get("source_bindings")
    if source_bindings != expected_source_bindings:
        raise RuntimeError("existing execution source bindings are not exact")
    for name, link in expected_source_bindings.items():
        if not isinstance(link, Mapping) or set(link) != {"path", "sha256"}:
            raise RuntimeError(f"execution source binding schema drift: {name}")
        candidate = resolve_repo_relative(str(link["path"]))
        _assert_regular_unlinked(candidate, label=f"execution source {name}")
        if sha256_file(candidate) != link["sha256"]:
            raise RuntimeError(f"execution source binding hash drift: {name}")

    verified: list[dict[str, Any]] = []
    for manifest_record, raw_record in zip(
        episodes, raw_manifest["episodes"], strict=True
    ):
        episode_id = str(raw_record["episode_id"])
        expected = _execution_record_if_valid(
            output_directory / f"{episode_id}.npz",
            output_directory / f"{episode_id}.json",
            role=role,
            regime=regime,
            raw_record=raw_record,
            head_count=head_count,
            compiled_gate_path=gate_path,
            compiled_gate_sha256=compiled_sha256,
            runtime_torch=runtime_torch,
            runtime_gate=runtime_gate,
            scientific_replay=scientific_replay,
        )
        if expected is None or expected != manifest_record:
            raise RuntimeError(
                f"execution manifest/part/sidecar drift for {episode_id}"
            )
        verified.append(expected)

    if role in ("fit", "selection"):
        aggregate_path = ATTEMPT_ROOT / "data" / role / regime / "role.npz"
        _assert_regular_unlinked(aggregate_path, label="development aggregate")
        if (
            value.get("aggregate_role_path") != relative_to_repo(aggregate_path)
            or value.get("aggregate_role_sha256") != sha256_file(aggregate_path)
        ):
            raise RuntimeError("existing development aggregate binding drift")
        _, aggregate_sha256, _ = _write_development_aggregate(
            role, regime, verified
        )
        if aggregate_sha256 != value["aggregate_role_sha256"]:
            raise RuntimeError("existing development aggregate content drift")
    else:
        all_calls = np.concatenate(
            [_calls_from_histogram(record["call_histogram"]) for record in verified]
        )
        if value.get("exact_compute") != exact_compute(all_calls, int(head_count)):
            raise RuntimeError("existing robust-gate exact compute drift")
    return value


def _load_scientific_context(device_name: str) -> tuple[Any, ...]:
    torch, runtime, model_io, device, base, contract, stack = (
        load_frozen_scientific_stack(device_name)
    )
    solver, v1, models, provenance = stack
    distribution_common, _, _ = load_frozen_v5_facade().distribution_modules()
    before = runtime.module_audit(base, solver, v1)
    if not before.get("passed"):
        raise RuntimeError("frozen module audit failed before execution")
    return (
        torch,
        runtime,
        model_io,
        device,
        base,
        contract,
        solver,
        v1,
        models,
        provenance,
        distribution_common,
        before,
    )


def _verify_module_immutability(
    runtime: Any, device: Any, base: Any, solver: Any, v1: Any, before: Any
) -> Any:
    runtime.synchronize(device)
    after = runtime.module_audit(base, solver, v1)
    if before != after or not after.get("passed"):
        raise RuntimeError("frozen model/refiner or gradient state changed")
    return after


def _execute_development_locked(
    role: str,
    regime: str,
    device_name: str,
    authorization: Mapping[str, Any],
    inputs: GenerationInputs,
) -> dict[str, Any]:
    """Persist dense development artifacts while writer ownership is held."""

    raw_manifest_path, output_directory, manifest_path = _role_paths(role, regime)
    raw_manifest = _load_raw_manifest(
        role, regime, raw_manifest_path, inputs=inputs
    )
    _preflight_execution_namespace(role, regime, raw_manifest)
    output_directory.mkdir(parents=True, exist_ok=True)
    failure_path = ATTEMPT_ROOT / f"audit/{role}_{regime}_execution_failure.json"
    records: list[dict[str, Any]] = []
    try:
        context = _load_scientific_context(device_name)
        (
            torch,
            runtime,
            model_io,
            device,
            base,
            contract,
            solver,
            v1,
            _,
            provenance,
            distribution_common,
            before,
        ) = context
        replay = _replay_context(context)
        existing = _validate_existing_execution_manifest(
            role=role,
            regime=regime,
            path=manifest_path,
            output_directory=output_directory,
            raw_manifest_path=raw_manifest_path,
            raw_manifest=raw_manifest,
            authorization=authorization,
            scientific_replay=replay,
        )
        if existing is not None:
            return existing
        for index, raw_record in enumerate(raw_manifest["episodes"]):
            episode_id = str(raw_record["episode_id"])
            part_path = output_directory / f"{episode_id}.npz"
            sidecar_path = output_directory / f"{episode_id}.json"
            prior = _execution_record_if_valid(
                part_path,
                sidecar_path,
                role=role,
                regime=regime,
                raw_record=raw_record,
                scientific_replay=replay,
            )
            if prior is not None:
                records.append(prior)
                continue
            raw_path = resolve_repo_relative(str(raw_record["raw_path"]))
            raw_sidecar_path = raw_path.with_suffix(".json")
            if not raw_sidecar_path.is_file():
                raise RuntimeError(f"raw episode sidecar is missing: {raw_sidecar_path}")
            loaded, loader_audit = load_model_gate_inputs(
                raw_path, expected_sha256=str(raw_record["raw_sha256"])
            )
            history, actions, target, base_prediction = prepare_frozen_episode_tensors(
                torch,
                runtime,
                model_io,
                device,
                base,
                contract,
                distribution_common,
                loaded,
            )
            exits, features, stage_current, stage_update = _dense_development_trace(
                torch, solver, history, actions, base_prediction
            )
            reconstructed = reconstruct_features_from_primitives(
                torch, history, actions, stage_current, stage_update
            )
            if not torch.equal(features, reconstructed):
                raise RuntimeError("development primitive feature reconstruction is not exact")
            if exits.shape != (ROWS_PER_EPISODE, DEPTHS, LATENT_DIM) or features.shape != (
                ROWS_PER_EPISODE,
                STAGES,
                FEATURE_DIM,
            ):
                raise RuntimeError("development execution tensor shape drift")
            arrays = {
                "episode_slot": np.full(
                    ROWS_PER_EPISODE, int(raw_record["slot"]), dtype=np.int32
                ),
                "model_step": np.arange(3, 41, dtype=np.int16),
                "target": target.cpu().numpy().astype(np.float32),
                "exits": exits.cpu().numpy().astype(np.float32),
                "production_features": features.cpu().numpy().astype(np.float32),
                "history": history.cpu().numpy().astype(np.float32),
                "action_history": actions.cpu().numpy().astype(np.float32),
                "stage_current": stage_current.cpu().numpy().astype(np.float32),
                "stage_update": stage_update.cpu().numpy().astype(np.float32),
            }
            records.append(
                _write_execution_part(
                    part_path,
                    sidecar_path,
                    arrays,
                    {
                        "schema_version": 1,
                        "attempt": ATTEMPT_VERSION,
                        "created_unix_ns": time.time_ns(),
                        "role": role,
                        "regime": regime,
                        "slot": int(raw_record["slot"]),
                        "episode_id": episode_id,
                        "raw_path": raw_record["raw_path"],
                        "raw_sha256": raw_record["raw_sha256"],
                        "raw_sidecar_path": relative_to_repo(raw_sidecar_path),
                        "raw_sidecar_sha256": sha256_file(raw_sidecar_path),
                        "input_loader_audit": loader_audit,
                        "evaluation": "four dense exits and all three frozen causal feature stages",
                        "target_use": f"permitted only for isolated {role}",
                        "contact_or_privileged_materialized": False,
                        "primitive_feature_reconstruction_exact": True,
                        "no_gradients": True,
                    },
                )
            )
            print(
                f"evaluated {role} {regime} {index + 1}/{len(raw_manifest['episodes'])}",
                flush=True,
            )
        after = _verify_module_immutability(
            runtime, device, base, solver, v1, before
        )
        (
            aggregate_path,
            aggregate_sha256,
            aggregate_recovery,
        ) = _write_development_aggregate(role, regime, records)
        source_bindings = _execution_source_bindings(
            raw_manifest, authorization
        )
        manifest = {
            "schema_version": 1,
            "attempt": ATTEMPT_VERSION,
            "created_unix_ns": time.time_ns(),
            "role": role,
            "regime": regime,
            "complete": True,
            "episode_count": len(records),
            "row_count": len(records) * ROWS_PER_EPISODE,
            "episodes": records,
            "aggregate_role_path": relative_to_repo(aggregate_path),
            "aggregate_role_sha256": aggregate_sha256,
            "aggregate_recovery": aggregate_recovery,
            "aggregate_role_keys": [
                "episode_slot",
                "model_step",
                "target",
                "exits",
                "production_features",
            ],
            "raw_manifest_path": relative_to_repo(raw_manifest_path),
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
            "source_raw_manifest_sha256": sha256_file(raw_manifest_path),
            "authorization": authorization,
            "source_bindings": source_bindings,
            "loaded_input_keys": ["action", "pixels"],
            "contact_or_privileged_materialized": False,
            "target_role_isolation": role,
            "causal_primitive_contract": {
                "history": [ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM],
                "action_history": [ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM],
                "stage_current": [ROWS_PER_EPISODE, STAGES, LATENT_DIM],
                "stage_update": [ROWS_PER_EPISODE, STAGES, LATENT_DIM],
                "sufficient_for_independent_feature_reconstruction": True,
            },
            "module_before": before,
            "module_after": after,
            "base_provenance": provenance,
            "frozen_facade": frozen_facade_source_audit(),
            "no_gradients": True,
        }
        atomic_json(manifest_path, manifest, exclusive=True)
        return manifest
    except Exception as error:
        if not failure_path.exists():
            atomic_json(
                failure_path,
                {
                    "schema_version": 1,
                    "attempt": ATTEMPT_VERSION,
                    "created_unix_ns": time.time_ns(),
                    "role": role,
                    "regime": regime,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_execution_parts": len(records),
                    "contact_or_privileged_materialized": False,
                },
                exclusive=True,
            )
        raise


def execute_development(
    role: str,
    regime: str,
    device_name: str = "mps",
) -> dict[str, Any]:
    """Authorize, lock, and persist dense fit/selection execution."""

    if role not in ("fit", "selection"):
        raise ValueError("development role must be fit or selection")
    _enforce_execution_runtime(device_name)
    authorization = _state_authorization(role)
    inputs = load_generation_inputs(role, regime)
    preflight_output_namespace(role, regime, inputs.primary)
    with materialization_lock(role, regime):
        return _execute_development_locked(
            role, regime, device_name, authorization, inputs
        )


def _confirmation_equivalence(
    torch: Any,
    solver: Any,
    gate: GateTensors,
    history: Any,
    actions: Any,
    base_prediction: Any,
    sparse: SparseExecution,
    repeat: SparseExecution,
    dense: DenseExecution,
) -> dict[str, Any]:
    with torch.inference_mode():
        selected_api = solver.forward_selected(
            history, actions, base_prediction, sparse.calls
        )
    row_index = torch.arange(len(history), device=history.device)
    selected_dense = dense.exits[row_index, sparse.calls - 1]
    delta = (sparse.selected - selected_dense).abs()
    sparse_reconstructed_features = reconstruct_features_from_primitives(
        torch,
        history,
        actions,
        sparse.stage_current,
        sparse.stage_update,
        sparse.reached,
    )
    sparse_reconstructed_scores, sparse_reconstructed_heads = (
        reconstruct_scores_from_features(
            torch, gate, sparse_reconstructed_features, sparse.reached
        )
    )
    dense_reconstructed_features = reconstruct_features_from_primitives(
        torch, history, actions, dense.stage_current, dense.stage_update
    )
    dense_reconstructed_scores, dense_reconstructed_heads = (
        reconstruct_scores_from_features(
            torch, gate, dense_reconstructed_features
        )
    )
    depth_one = sparse.calls == 1
    checks = {
        "same_sparse_selected_exact": bool(
            torch.equal(sparse.selected, repeat.selected)
        ),
        "same_sparse_calls_exact": bool(torch.equal(sparse.calls, repeat.calls)),
        "same_sparse_scores_exact": tensors_exact_with_nan(
            torch, sparse.scores, repeat.scores
        ),
        "same_sparse_head_scores_exact": tensors_exact_with_nan(
            torch, sparse.head_scores, repeat.head_scores
        ),
        "same_sparse_features_exact": tensors_exact_with_nan(
            torch, sparse.features, repeat.features
        ),
        "same_sparse_primitives_exact": tensors_exact_with_nan(
            torch, sparse.stage_current, repeat.stage_current
        )
        and tensors_exact_with_nan(torch, sparse.stage_update, repeat.stage_update)
        and bool(torch.equal(sparse.reached, repeat.reached)),
        "manual_sparse_forward_selected_exact": bool(
            torch.equal(sparse.selected, selected_api)
        ),
        "sparse_dense_calls_exact": bool(torch.equal(sparse.calls, dense.calls)),
        "sparse_dense_histogram_exact": bool(
            torch.equal(
                torch.bincount(sparse.calls, minlength=5),
                torch.bincount(dense.calls, minlength=5),
            )
        ),
        "depth_one_selected_rows_exact": bool(
            not depth_one.any()
            or torch.equal(sparse.selected[depth_one], selected_dense[depth_one])
        ),
        "cross_batch_allclose": bool(
            torch.allclose(
                sparse.selected,
                selected_dense,
                rtol=NUMERICAL_RTOL,
                atol=NUMERICAL_ATOL,
            )
        ),
        "cross_batch_max_abs_ceiling": bool(
            float(delta.max().item()) <= NUMERICAL_MAX_ABS
        ),
        "sparse_primitive_feature_reconstruction_exact": tensors_exact_with_nan(
            torch, sparse.features, sparse_reconstructed_features
        ),
        "sparse_feature_score_reconstruction_exact": tensors_exact_with_nan(
            torch, sparse.scores, sparse_reconstructed_scores
        )
        and tensors_exact_with_nan(
            torch, sparse.head_scores, sparse_reconstructed_heads
        ),
        "dense_primitive_feature_reconstruction_exact": bool(
            torch.equal(dense.features, dense_reconstructed_features)
        ),
        "dense_feature_score_reconstruction_exact": bool(
            torch.equal(dense.scores, dense_reconstructed_scores)
            and torch.equal(dense.head_scores, dense_reconstructed_heads)
        ),
        "unreached_sparse_values_are_nan": bool(
            torch.isnan(sparse.scores[~sparse.reached]).all().item()
            and torch.isnan(sparse.head_scores[~sparse.reached]).all().item()
            and torch.isnan(sparse.features[~sparse.reached]).all().item()
            and torch.isnan(sparse.stage_current[~sparse.reached]).all().item()
            and torch.isnan(sparse.stage_update[~sparse.reached]).all().item()
        ),
        "target_loss_never_computed": True,
    }
    return {
        "checks": checks,
        "cross_batch_max_abs": float(delta.max().item()),
        "cross_batch_mean_abs": float(delta.mean().item()),
        "cross_batch_rmse": float(torch.sqrt(torch.square(delta).mean()).item()),
        "passed": all(checks.values()),
    }


def _calls_from_histogram(histogram: list[int]) -> np.ndarray:
    if len(histogram) != DEPTHS or any(int(value) < 0 for value in histogram):
        raise RuntimeError("call histogram drift")
    return np.concatenate(
        [np.full(int(count), depth, dtype=np.int8) for depth, count in enumerate(histogram, 1)]
    )


def _execute_confirmation_locked(
    role: str,
    regime: str,
    gate_path: Path,
    device_name: str,
    authorization: Mapping[str, Any],
    inputs: GenerationInputs,
) -> dict[str, Any]:
    """Run sparse routing while role-by-DGP writer ownership is held."""

    raw_manifest_path, output_directory, manifest_path = _role_paths(role, regime)
    raw_manifest = _load_raw_manifest(
        role, regime, raw_manifest_path, inputs=inputs
    )
    _preflight_execution_namespace(role, regime, raw_manifest)
    output_directory.mkdir(parents=True, exist_ok=True)
    failure_path = ATTEMPT_ROOT / f"audit/{role}_{regime}_execution_failure.json"
    records: list[dict[str, Any]] = []
    try:
        context = _load_scientific_context(device_name)
        (
            torch,
            runtime,
            model_io,
            device,
            base,
            contract,
            solver,
            v1,
            _,
            provenance,
            distribution_common,
            before,
        ) = context
        gate = load_gate_tensors(torch, device, gate_path)
        replay = _replay_context(context)
        existing = _validate_existing_execution_manifest(
            role=role,
            regime=regime,
            path=manifest_path,
            output_directory=output_directory,
            raw_manifest_path=raw_manifest_path,
            raw_manifest=raw_manifest,
            authorization=authorization,
            gate_path=gate_path,
            runtime_torch=torch,
            runtime_gate=gate,
            scientific_replay=replay,
        )
        if existing is not None:
            return existing
        for index, raw_record in enumerate(raw_manifest["episodes"]):
            episode_id = str(raw_record["episode_id"])
            part_path = output_directory / f"{episode_id}.npz"
            sidecar_path = output_directory / f"{episode_id}.json"
            prior = _execution_record_if_valid(
                part_path,
                sidecar_path,
                role=role,
                regime=regime,
                raw_record=raw_record,
                head_count=gate.head_count,
                compiled_gate_path=gate.source_path,
                compiled_gate_sha256=gate.source_sha256,
                runtime_torch=torch,
                runtime_gate=gate,
                scientific_replay=replay,
            )
            if prior is not None:
                records.append(prior)
                continue
            raw_path = resolve_repo_relative(str(raw_record["raw_path"]))
            raw_sidecar_path = raw_path.with_suffix(".json")
            if not raw_sidecar_path.is_file():
                raise RuntimeError(f"raw episode sidecar is missing: {raw_sidecar_path}")
            loaded, loader_audit = load_model_gate_inputs(
                raw_path, expected_sha256=str(raw_record["raw_sha256"])
            )
            history, actions, target, base_prediction = prepare_frozen_episode_tensors(
                torch,
                runtime,
                model_io,
                device,
                base,
                contract,
                distribution_common,
                loaded,
            )
            dense = dense_shadow(
                torch, solver, gate, history, actions, base_prediction
            )
            sparse = manual_sparse(
                torch, solver, gate, history, actions, base_prediction
            )
            repeat = manual_sparse(
                torch, solver, gate, history, actions, base_prediction
            )
            equivalence = _confirmation_equivalence(
                torch,
                solver,
                gate,
                history,
                actions,
                base_prediction,
                sparse,
                repeat,
                dense,
            )
            if not equivalence["passed"]:
                raise RuntimeError(
                    f"sparse/dense numerical contract failed for {episode_id}: {equivalence}"
                )
            calls_array = sparse.calls.cpu().numpy().astype(np.int8)
            call_histogram = np.bincount(calls_array, minlength=5)[1:].tolist()
            compute = exact_compute(calls_array, gate.head_count)
            arrays = {
                "episode_slot": np.full(
                    ROWS_PER_EPISODE, int(raw_record["slot"]), dtype=np.int32
                ),
                "model_step": np.arange(3, 41, dtype=np.int16),
                "exits": dense.exits.cpu().numpy().astype(np.float32),
                "selected": sparse.selected.cpu().numpy().astype(np.float32),
                "calls": calls_array,
                "scores": sparse.scores.cpu().numpy().astype(np.float32),
                "head_scores": sparse.head_scores.cpu().numpy().astype(np.float32),
                "production_features": sparse.features.cpu().numpy().astype(np.float32),
                "history": history.cpu().numpy().astype(np.float32),
                "action_history": actions.cpu().numpy().astype(np.float32),
                "stage_current": sparse.stage_current.cpu().numpy().astype(np.float32),
                "stage_update": sparse.stage_update.cpu().numpy().astype(np.float32),
                "reached": sparse.reached.cpu().numpy().astype(np.bool_),
                "dense_scores": dense.scores.cpu().numpy().astype(np.float32),
                "dense_head_scores": dense.head_scores.cpu().numpy().astype(np.float32),
                "dense_stage_current": dense.stage_current.cpu().numpy().astype(np.float32),
                "dense_stage_update": dense.stage_update.cpu().numpy().astype(np.float32),
            }
            if role == "confirmation":
                arrays["target"] = target.cpu().numpy().astype(np.float32)
            records.append(
                _write_execution_part(
                    part_path,
                    sidecar_path,
                    arrays,
                    {
                        "schema_version": 1,
                        "attempt": ATTEMPT_VERSION,
                        "created_unix_ns": time.time_ns(),
                        "role": role,
                        "regime": regime,
                        "slot": int(raw_record["slot"]),
                        "episode_id": episode_id,
                        "raw_path": raw_record["raw_path"],
                        "raw_sha256": raw_record["raw_sha256"],
                        "raw_sidecar_path": relative_to_repo(raw_sidecar_path),
                        "raw_sidecar_sha256": sha256_file(raw_sidecar_path),
                        "compiled_gate_path": relative_to_repo(gate.source_path),
                        "compiled_gate_sha256": gate.source_sha256,
                        "architecture": gate.architecture,
                        "candidate_id": gate.candidate_id,
                        "head_names": list(gate.head_names),
                        "input_loader_audit": loader_audit,
                        "actual_adaptive_output": "manual sequential sparse reached-row execution",
                        "dense_role": "numerical shadow and fixed-depth comparator only",
                        "call_histogram": call_histogram,
                        "exact_compute": compute,
                        "equivalence": equivalence,
                        "target_persisted": role == "confirmation",
                        "target_loss_computed": False,
                        "contact_or_privileged_materialized": False,
                        "no_gradients": True,
                    },
                )
            )
            print(
                f"executed {role} {regime} {index + 1}/{len(raw_manifest['episodes'])}",
                flush=True,
            )
        after = _verify_module_immutability(
            runtime, device, base, solver, v1, before
        )
        all_calls = np.concatenate(
            [_calls_from_histogram(record["call_histogram"]) for record in records]
        )
        aggregate_compute = exact_compute(all_calls, gate.head_count)
        source_bindings = _execution_source_bindings(
            raw_manifest, authorization, gate
        )
        manifest = {
            "schema_version": 1,
            "attempt": ATTEMPT_VERSION,
            "created_unix_ns": time.time_ns(),
            "role": role,
            "regime": regime,
            "complete": True,
            "episode_count": len(records),
            "row_count": len(records) * ROWS_PER_EPISODE,
            "episodes": records,
            "raw_manifest_path": relative_to_repo(raw_manifest_path),
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
            "source_raw_manifest_sha256": sha256_file(raw_manifest_path),
            "compiled_gate_path": relative_to_repo(gate.source_path),
            "compiled_gate_sha256": gate.source_sha256,
            "architecture": gate.architecture,
            "candidate_id": gate.candidate_id,
            "head_names": list(gate.head_names),
            "head_count": gate.head_count,
            "gate_cost_per_reached_decision": gate_cost(gate.head_count),
            "exact_compute": aggregate_compute,
            "authorization": authorization,
            "source_bindings": source_bindings,
            "actual_adaptive_output": "manual sequential sparse reached-row execution",
            "dense_role": "numerical shadow and fixed-depth comparator only",
            "strict_continue_operator": ">",
            "loaded_input_keys": ["action", "pixels"],
            "contact_or_privileged_materialized": False,
            "target_persisted": role == "confirmation",
            "target_loss_computed_during_execution": False,
            "causal_primitive_contract": {
                "history": [ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM],
                "action_history": [ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM],
                "stage_current": [ROWS_PER_EPISODE, STAGES, LATENT_DIM],
                "stage_update": [ROWS_PER_EPISODE, STAGES, LATENT_DIM],
                "reached": [ROWS_PER_EPISODE, STAGES],
                "unreached_sparse_values": "NaN",
                "dense_stage_current_and_update_also_persisted": True,
                "sufficient_for_independent_feature_and_score_reconstruction": True,
            },
            "numerical_contract": {
                "rtol": NUMERICAL_RTOL,
                "atol": NUMERICAL_ATOL,
                "maximum_absolute_error_ceiling": NUMERICAL_MAX_ABS,
                "calls_and_same_path_sparse_values_exact": True,
            },
            "all_equivalence_checks_passed": True,
            "module_before": before,
            "module_after": after,
            "base_provenance": provenance,
            "frozen_facade": frozen_facade_source_audit(),
            "no_gradients": True,
        }
        atomic_json(manifest_path, manifest, exclusive=True)
        return manifest
    except Exception as error:
        if not failure_path.exists():
            atomic_json(
                failure_path,
                {
                    "schema_version": 1,
                    "attempt": ATTEMPT_VERSION,
                    "created_unix_ns": time.time_ns(),
                    "role": role,
                    "regime": regime,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": traceback.format_exc(),
                    "completed_execution_parts": len(records),
                    "target_loss_computed": False,
                    "contact_or_privileged_materialized": False,
                },
                exclusive=True,
            )
        raise


def execute_confirmation(
    role: str,
    regime: str,
    gate_path: Path,
    device_name: str = "mps",
) -> dict[str, Any]:
    """Authorize, lock, and execute excluded smoke or confirmation."""

    if role not in ("smoke", "confirmation"):
        raise ValueError("confirmation execution role must be smoke or confirmation")
    _enforce_execution_runtime(device_name)
    authorization = _state_authorization(role)
    inputs = load_generation_inputs(role, regime)
    preflight_output_namespace(role, regime, inputs.primary)
    with materialization_lock(role, regime):
        return _execute_confirmation_locked(
            role,
            regime,
            Path(gate_path),
            device_name,
            authorization,
            inputs,
        )


class _FrozenLatencyAdapter:
    """Narrow post-analysis adapter consumed by ``latency.py``."""

    def __init__(self, device_name: str, gate_path: Path) -> None:
        controller = read_verified_controller()
        if controller.get("current_state") != "LATENCY_AND_RESOURCE_REPORTING":
            raise RuntimeError("latency inputs may open only in LATENCY_AND_RESOURCE_REPORTING")
        input_seal_path = ATTEMPT_ROOT / "audit/confirmation_input_seal.json"
        input_seal_sha256 = sha256_file(input_seal_path)
        input_seal_relative = relative_to_repo(input_seal_path)
        if not any(
            item.get("evidence_path") == input_seal_relative
            and item.get("evidence_sha256") == input_seal_sha256
            for item in controller.get("verified_checkpoints", [])
        ):
            raise RuntimeError("controller does not verify the confirmation input seal")
        context = _load_scientific_context(device_name)
        (
            self.torch,
            self.runtime,
            _,
            self.device,
            self.base,
            _,
            self.solver,
            self.v1,
            _,
            base_provenance,
            _,
            self._module_before,
        ) = context
        self.gate = load_gate_tensors(self.torch, self.device, gate_path)
        self._inputs: dict[str, tuple[Any, Any, Any]] = {}
        manifest_hashes: dict[str, str] = {}
        for regime in REGIME_ORDER:
            manifest_path = (
                ATTEMPT_ROOT
                / "data"
                / "confirmation"
                / regime
                / "execution_manifest.json"
            )
            manifest = read_json(manifest_path)
            if (
                manifest.get("complete") is not True
                or manifest.get("role") != "confirmation"
                or manifest.get("regime") != regime
                or manifest.get("compiled_gate_sha256") != self.gate.source_sha256
            ):
                raise RuntimeError(f"latency execution manifest drift: {regime}")
            history_parts: list[np.ndarray] = []
            action_parts: list[np.ndarray] = []
            call_parts: list[np.ndarray] = []
            rows = 0
            for record in manifest["episodes"]:
                part = resolve_repo_relative(record["path"])
                if sha256_file(part) != record["sha256"]:
                    raise RuntimeError("latency prefix execution hash drift")
                with np.load(part, allow_pickle=False) as stored:
                    # Deliberately materialize no target or endpoint array.
                    history_parts.append(stored["history"].copy())
                    action_parts.append(stored["action_history"].copy())
                    call_parts.append(stored["calls"].copy())
                rows += len(call_parts[-1])
                if rows >= 1_024:
                    break
            if rows < 1_024:
                raise RuntimeError("confirmation execution lacks 1,024 latency rows")
            history = np.concatenate(history_parts, axis=0)[:1_024].astype(
                np.float32, copy=False
            )
            actions = np.concatenate(action_parts, axis=0)[:1_024].astype(
                np.float32, copy=False
            )
            calls = np.concatenate(call_parts, axis=0)[:1_024].astype(
                np.int64, copy=False
            )
            self._inputs[regime] = (
                self.torch.as_tensor(history, device=self.device),
                self.torch.as_tensor(actions, device=self.device),
                self.torch.as_tensor(calls, dtype=self.torch.long, device=self.device),
            )
            manifest_hashes[regime] = sha256_file(manifest_path)
        self.provenance = {
            "runner": relative_to_repo(Path(__file__)),
            "runner_sha256": sha256_file(Path(__file__)),
            "compiled_gate_path": relative_to_repo(self.gate.source_path),
            "compiled_gate_sha256": self.gate.source_sha256,
            "candidate_id": self.gate.candidate_id,
            "architecture": self.gate.architecture,
            "confirmation_execution_manifest_sha256": manifest_hashes,
            "base_provenance": base_provenance,
            "latency_inputs_materialized": ["history", "action_history", "calls"],
            "target_or_contact_materialized": False,
            "confirmation_input_seal_path": input_seal_relative,
            "confirmation_input_seal_sha256": input_seal_sha256,
        }

    def synchronize(self) -> None:
        self.runtime.synchronize(self.device)

    def module_audit(self) -> Mapping[str, Any]:
        return self.runtime.module_audit(self.base, self.solver, self.v1)

    def _batch(self, regime: str, batch_size: int) -> tuple[Any, Any, Any]:
        if regime not in self._inputs or not 1 <= int(batch_size) <= 1_024:
            raise ValueError("invalid latency regime or batch size")
        history, actions, calls = self._inputs[regime]
        return history[:batch_size], actions[:batch_size], calls[:batch_size]

    def _base_prediction(self, history: Any, actions: Any) -> Any:
        with self.torch.inference_mode():
            return self.runtime.base_predict(self.base, history, actions)

    def expected_calls(self, regime: str, batch_size: int) -> Any:
        return self._batch(regime, batch_size)[2]

    def fixed_depth(
        self, regime: str, batch_size: int, depth: int
    ) -> tuple[Any, Any]:
        if depth not in (1, 2, 3, 4):
            raise ValueError("fixed depth must be one through four")
        history, actions, _ = self._batch(regime, batch_size)
        calls = self.torch.full(
            (batch_size,), depth, dtype=self.torch.long, device=self.device
        )
        with self.torch.inference_mode():
            base_prediction = self._base_prediction(history, actions)
            selected = self.solver.forward_selected(
                history, actions, base_prediction, calls
            )
        return selected, calls

    def adaptive_sparse(self, regime: str, batch_size: int) -> tuple[Any, Any]:
        history, actions, _ = self._batch(regime, batch_size)
        base_prediction = self._base_prediction(history, actions)
        result = manual_sparse(
            self.torch,
            self.solver,
            self.gate,
            history,
            actions,
            base_prediction,
        )
        return result.selected, result.calls

    def dense_shadow(self, regime: str, batch_size: int) -> tuple[Any, Any]:
        history, actions, _ = self._batch(regime, batch_size)
        base_prediction = self._base_prediction(history, actions)
        result = globals()["dense_shadow"](
            self.torch,
            self.solver,
            self.gate,
            history,
            actions,
            base_prediction,
        )
        row = self.torch.arange(batch_size, device=self.device)
        return result.exits[row, result.calls - 1], result.calls


def make_latency_adapter(
    device_name: str, gate_path: Path | None = None
) -> _FrozenLatencyAdapter:
    """Load the sealed gate/stack and confirmation-prefix latency inputs."""

    path = gate_path or ATTEMPT_ROOT / "freeze/compiled_gate.npz"
    return _FrozenLatencyAdapter(device_name, Path(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    development = subparsers.add_parser("development")
    development.add_argument("role", choices=("fit", "selection"))
    development.add_argument("regime", choices=REGIME_ORDER)
    development.add_argument("--device", default="mps")
    execute = subparsers.add_parser("execute")
    execute.add_argument("role", choices=("smoke", "confirmation"))
    execute.add_argument("regime", choices=REGIME_ORDER)
    execute.add_argument(
        "--gate", type=Path, default=ATTEMPT_ROOT / "freeze/compiled_gate.npz"
    )
    execute.add_argument("--device", default="mps")
    arguments = parser.parse_args()
    if arguments.command == "development":
        result = execute_development(
            arguments.role, arguments.regime, arguments.device
        )
    else:
        result = execute_confirmation(
            arguments.role, arguments.regime, arguments.gate, arguments.device
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
