#!/usr/bin/env python3
"""Independent, read-only scientific replay qualification for v008 artifacts.

This verifier deliberately does not import ``runner``, ``input_loader``,
``counted_features``, ``compile_gate``, or ``flops``.  It loads the frozen model
and solver directly, materializes only pixels and actions, independently
reconstructs every persisted scientific tensor, and never computes a loss or
effect.  Its JSON result is deterministic so the workflow can immutably adopt
it after a crash and then rerun it before role-count acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ATTEMPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ATTEMPT_ROOT.parents[3]
ATTEMPT = ATTEMPT_ROOT.name
if ATTEMPT != "v008":
    raise RuntimeError(f"scientific replay must run from v008, got {ATTEMPT!r}")
INHERITED_FIT_ATTEMPT = "v003"
INHERITED_FIT_ROOT = ATTEMPT_ROOT.parent / INHERITED_FIT_ATTEMPT

REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ROLES = ("fit", "selection", "smoke", "confirmation")
ROWS = 38
HISTORY_LEN = 3
LATENT_DIM = 192
ACTION_DIM = 25
FEATURE_DIM = 1046
STAGES = 3
DEPTHS = 4
PIXELS_SHAPE = (201, 224, 224, 3)
ACTION_SHAPE = (201, 5)
INPUT_KEYS = frozenset({"action", "pixels"})
AGGREGATE_KEYS = (
    "episode_slot",
    "model_step",
    "target",
    "exits",
    "production_features",
)
ARCHITECTURE_HEADS = {
    "balanced_pooled_dual": 2,
    "domain_envelope_eight": 8,
}

BASE_FLOPS = 70_529_190
DEPTH1_FLOPS = 669_184
ADAPTER_FLOPS = 264_960
FEATURE_FLOPS = 3_801
AFFINE_HEAD_FLOPS = 2_092
GATE_NONFLOPS = {2: 5, 8: 11}

EXPECTED_SOURCE_HASHES = {
    "runs/lewm_adaptive_compute_distribution_contract/common.py": (
        "52db77ee31ac44f5438270d818cd36832ec74c1a19764bf2df63b0570a4ea4c2"
    ),
    "runs/lewm_adaptive_compute_v4/runtime.py": (
        "801b160ad9c457f7eabb4b92395c90cb7eb2f9d1809a627bd0202df8a210bfdf"
    ),
    "runs/lewm_adaptive_compute_v2/model_io.py": (
        "5505e529ba17aeb0c641e7d4e8dc511d0ed61decdc13a596915ba8964b954874"
    ),
    "runs/lewm_domain_robust_gate/attempts/v008/counted_features.py": (
        "f745ff29039f7a00fc28b4de0923efeda23331fbbb8a2c654fb2c4c6afa642d4"
    ),
    "runs/lewm_domain_robust_gate/attempts/v008/input_loader.py": (
        "40989c2c22cb96bc26bff48c9f9bead9c3e1efd9ee355194800703b943737039"
    ),
    "runs/lewm_domain_robust_gate/attempts/v008/compile_gate.py": (
        "01a76055bf3fc70ee8a45853dd3260606e74f4693d77a444d22b83f923e2f126"
    ),
    "runs/lewm_domain_robust_gate/attempts/v008/flops.py": (
        "789375f939b125da9960625e0b502b8b6c28ebe0a75ba2c573d3c8473ce749a0"
    ),
    "runs/lewm_domain_robust_gate/attempts/v008/runtime_contract.py": (
        "3b3787d3a53fc790cc356c7f7e5c1cfb2cb62fa0c1d84ebe3defb59eaa63f4f7"
    ),
    "runs/lewm_domain_robust_gate/attempts/v008/runner.py": (
        "dd9b6028583bbe529b99181664e94b27cf470ba70a1f145df70f965016417c7a"
    ),
}
FROZEN_FACADE_SOURCE_HASHES = {
    str((REPO_ROOT / relative).resolve(strict=True)): digest
    for relative, digest in {
        "runs/lewm_adaptive_compute_v2/model_io.py": (
            "5505e529ba17aeb0c641e7d4e8dc511d0ed61decdc13a596915ba8964b954874"
        ),
        "runs/lewm_adaptive_compute_v4/runtime.py": (
            "801b160ad9c457f7eabb4b92395c90cb7eb2f9d1809a627bd0202df8a210bfdf"
        ),
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/runner.py": (
            "ebcbe74b409867c538460b53b5ddb69a05c9163832ec4ab4eb2a3ea484ab1174"
        ),
    }.items()
}

RAW_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "complete",
        "role",
        "regime",
        "regime_specification",
        "episode_count",
        "episodes",
        "replacements_used",
        "raw_archive_members",
        "raw_pixels_contract",
        "raw_action_contract",
        "role_isolation",
        "smoke_permanently_excluded",
        "retention_contract",
        "dgp_matrix_path",
        "dgp_matrix_sha256",
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "authorization_seal",
        "replacement_registry_path",
        "replacement_registry_scope",
        "orphan_policy",
    }
)
RAW_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "complete",
        "role",
        "regime",
        "slot",
        "episode_id",
        "seed_source_episode_id",
        "replacement_used",
        "replacement_claim_index",
        "replacement_claim_sha256",
        "env_seed",
        "policy_seed",
        "oracle_np_seed",
        "action_space_seed",
        "raw_path",
        "raw_sha256",
        "raw_bytes",
        "arrays",
        "initial_pixels_sha256",
        "generation_audit",
        "input_loader_audit",
        "persistence_intent_path",
        "persistence_intent_sha256",
        "authorization_seal",
        "raw_archive_members",
        "orphan_adoption_safe",
    }
)
EXECUTION_RECORD_KEYS = frozenset(
    {
        "slot",
        "episode_id",
        "path",
        "sha256",
        "sidecar_path",
        "sidecar_sha256",
        "source_raw_path",
        "source_raw_sha256",
        "source_raw_sidecar_path",
        "source_raw_sidecar_sha256",
        "call_histogram",
    }
)
COMMON_EXECUTION_MANIFEST_KEYS = frozenset(
    {
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
)
DEVELOPMENT_EXECUTION_MANIFEST_KEYS = COMMON_EXECUTION_MANIFEST_KEYS | {
    "aggregate_role_path",
    "aggregate_role_sha256",
    "aggregate_recovery",
    "aggregate_role_keys",
    "target_role_isolation",
}
ROBUST_EXECUTION_MANIFEST_KEYS = COMMON_EXECUTION_MANIFEST_KEYS | {
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
COMMON_EXECUTION_SIDECAR_KEYS = frozenset(
    {
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
)
DEVELOPMENT_SIDECAR_KEYS = COMMON_EXECUTION_SIDECAR_KEYS | {
    "evaluation",
    "target_use",
    "primitive_feature_reconstruction_exact",
}
ROBUST_SIDECAR_KEYS = COMMON_EXECUTION_SIDECAR_KEYS | {
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
INPUT_AUDIT_KEYS = frozenset(
    {
        "loaded_keys",
        "pixels_shape",
        "pixels_dtype",
        "action_shape",
        "action_dtype",
        "pixels_finite",
        "modeled_actions_finite",
        "terminal_action_nan_sentinel",
        "array_sha256",
        "archive_keys",
        "archive_sha256",
        "arrays_materialized",
        "non_input_arrays_materialized",
    }
)
COMPILED_GATE_KEYS = frozenset(
    {
        "schema_version",
        "architecture",
        "candidate_id",
        "weights",
        "biases",
        "thresholds",
        "head_names",
    }
)
COMMON_SOURCE_BINDING_KEYS = frozenset(
    {
        "dgp_matrix",
        "cohort_seed_ledger",
        "execution_authorization_seal",
        "counted_feature_source",
        "pricing_source",
        "runner_source",
        "replacement_registry",
    }
)
DEVELOPMENT_SOURCE_BINDING_KEYS = COMMON_SOURCE_BINDING_KEYS | {
    "raw_authorization_seal"
}
ROBUST_SOURCE_BINDING_KEYS = COMMON_SOURCE_BINDING_KEYS | {
    "gate_fit",
    "compiled_gate",
    "compiled_gate_manifest",
    "gate_freeze",
    "fixed_whitening",
}
DEVELOPMENT_AUTHORIZATION_KEYS = frozenset(
    {
        "state",
        "active_attempt",
        "state_sha256",
        "seal_path",
        "seal_sha256",
        "seal_checkpoint_state",
        "science_attempt",
        "raw_authorization_seal_path",
        "raw_authorization_seal_sha256",
        "exact_locked_snapshot_unchanged",
    }
)
ROBUST_AUTHORIZATION_KEYS = frozenset(
    {
        "state",
        "active_attempt",
        "state_sha256",
        "seal_path",
        "seal_sha256",
        "seal_checkpoint_state",
    }
)


class ReplayError(RuntimeError):
    """An artifact cannot be independently replay-qualified."""


@dataclass(frozen=True)
class Gate:
    architecture: str
    candidate_id: str
    head_names: tuple[str, ...]
    weights: Any
    biases: Any
    thresholds: Any
    path: Path
    sha256: str

    @property
    def head_count(self) -> int:
        return len(self.head_names)


@dataclass(frozen=True)
class Stack:
    torch: Any
    runtime: Any
    model_io: Any
    common: Any
    device: Any
    base: Any
    contract: Any
    solver: Any
    v1: Any
    provenance: Any
    module_before: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_json_equal(left: Any, right: Any) -> bool:
    """JSON equality that never treats booleans and integers as aliases."""

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_exact_json_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ReplayError(f"{label} must be an exact integer")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ReplayError(f"{label} must be an exact boolean")
    return value


def _exact_shape(value: Any, expected: Sequence[int], label: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != len(expected)
        or any(type(item) is not int for item in value)
        or value != list(expected)
    ):
        raise ReplayError(f"{label} must contain exact integer dimensions")


def _regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ReplayError(f"missing {label}: {path}") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or int(metadata.st_nlink) != 1
    ):
        raise ReplayError(f"{label} must be one unlinked regular file: {path}")
    return path


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ReplayError(f"path escapes repository: {path}") from error


def _resolve(raw: Any, label: str) -> Path:
    if not isinstance(raw, str):
        raise ReplayError(f"{label} path is not text")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
        raise ReplayError(f"{label} path is not canonical: {raw!r}")
    path = (REPO_ROOT / relative).resolve(strict=False)
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise ReplayError(f"{label} path escapes repository")
    return path


def _read_canonical_object(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayError(f"invalid {label} JSON: {path}") from error
    expected = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if not isinstance(value, dict) or raw != expected:
        raise ReplayError(f"noncanonical or non-object {label}: {path}")
    return value


def _link(path: Path) -> dict[str, str]:
    path = _regular_file(path, "bound artifact")
    return {"path": _relative(path), "sha256": sha256_file(path)}


def _source_audit() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        path = _regular_file(REPO_ROOT / relative, f"source {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise ReplayError(
                f"scientific replay source drift: {relative}: "
                f"expected={expected} observed={digest}"
            )
        observed[relative] = digest
    for local in (Path(__file__).resolve(), ATTEMPT_ROOT / "scientific_replay_launcher.py"):
        local = _regular_file(local, "scientific replay source")
        observed[_relative(local)] = sha256_file(local)
    return observed


def _directory_snapshot(path: Path) -> str:
    """Hash names and immutable stat identity without materializing file content."""

    root = Path(path)
    records: list[dict[str, Any]] = []
    if root.exists():
        for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            metadata = candidate.lstat()
            records.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "mode": int(metadata.st_mode),
                    "device": int(metadata.st_dev),
                    "inode": int(metadata.st_ino),
                    "links": int(metadata.st_nlink),
                    "size": int(metadata.st_size),
                    "mtime_ns": int(metadata.st_mtime_ns),
                }
            )
    return canonical_sha256(records)


def _assert_no_forbidden_keys(value: Any, location: str) -> None:
    forbidden = ("contact", "privileged", "qpos", "qvel", "motion", "phase", "reward", "success")
    allowed_negative = {"contact_or_privileged_materialized"}
    if isinstance(value, Mapping):
        if location.endswith(".frozen_facade.sources"):
            observed = {str(key): item for key, item in value.items()}
            if observed != FROZEN_FACADE_SOURCE_HASHES:
                raise ReplayError(f"frozen facade provenance map drift at {location}")
            for raw_path, expected_hash in observed.items():
                path = Path(raw_path)
                if (
                    not path.is_absolute()
                    or path.resolve(strict=True) != path
                    or not path.is_file()
                    or path.is_symlink()
                    or sha256_file(path) != expected_hash
                ):
                    raise ReplayError(
                        f"frozen facade provenance record drift at {location}"
                    )
            return
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in forbidden):
                if lowered not in allowed_negative or item is not False:
                    raise ReplayError(f"forbidden key materialized at {location}.{key}")
            _assert_no_forbidden_keys(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_forbidden_keys(item, f"{location}[{index}]")


def _arrays_exact(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left, right, equal_nan=True)
    )


def _array_contract(role: str, head_count: int | None) -> dict[str, tuple[tuple[int, ...], str]]:
    if role in ("fit", "selection"):
        return {
            "episode_slot": ((ROWS,), "int32"),
            "model_step": ((ROWS,), "int16"),
            "target": ((ROWS, LATENT_DIM), "float32"),
            "exits": ((ROWS, DEPTHS, LATENT_DIM), "float32"),
            "production_features": ((ROWS, STAGES, FEATURE_DIM), "float32"),
            "history": ((ROWS, HISTORY_LEN, LATENT_DIM), "float32"),
            "action_history": ((ROWS, HISTORY_LEN, ACTION_DIM), "float32"),
            "stage_current": ((ROWS, STAGES, LATENT_DIM), "float32"),
            "stage_update": ((ROWS, STAGES, LATENT_DIM), "float32"),
        }
    if role not in ("smoke", "confirmation") or head_count not in (2, 8):
        raise ReplayError("robust replay requires an exact compiled-gate head count")
    contract = {
        "episode_slot": ((ROWS,), "int32"),
        "model_step": ((ROWS,), "int16"),
        "exits": ((ROWS, DEPTHS, LATENT_DIM), "float32"),
        "selected": ((ROWS, LATENT_DIM), "float32"),
        "calls": ((ROWS,), "int8"),
        "scores": ((ROWS, STAGES), "float32"),
        "head_scores": ((ROWS, STAGES, head_count), "float32"),
        "production_features": ((ROWS, STAGES, FEATURE_DIM), "float32"),
        "history": ((ROWS, HISTORY_LEN, LATENT_DIM), "float32"),
        "action_history": ((ROWS, HISTORY_LEN, ACTION_DIM), "float32"),
        "stage_current": ((ROWS, STAGES, LATENT_DIM), "float32"),
        "stage_update": ((ROWS, STAGES, LATENT_DIM), "float32"),
        "reached": ((ROWS, STAGES), "bool"),
        "dense_scores": ((ROWS, STAGES), "float32"),
        "dense_head_scores": ((ROWS, STAGES, head_count), "float32"),
        "dense_stage_current": ((ROWS, STAGES, LATENT_DIM), "float32"),
        "dense_stage_update": ((ROWS, STAGES, LATENT_DIM), "float32"),
    }
    if role == "confirmation":
        contract["target"] = ((ROWS, LATENT_DIM), "float32")
    return contract


def _validate_input_audit(value: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != INPUT_AUDIT_KEYS:
        raise ReplayError(f"{label} input-loader audit schema drift")
    _exact_shape(value.get("pixels_shape"), PIXELS_SHAPE, f"{label}.pixels_shape")
    _exact_shape(value.get("action_shape"), ACTION_SHAPE, f"{label}.action_shape")
    for name in (
        "pixels_finite",
        "modeled_actions_finite",
        "terminal_action_nan_sentinel",
        "non_input_arrays_materialized",
    ):
        _exact_bool(value.get(name), f"{label}.{name}")
    if not _exact_json_equal(dict(value), dict(expected)):
        raise ReplayError(f"{label} input-loader audit disagrees with raw replay")


def _load_raw_inputs(path: Path, expected_sha256: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = _regular_file(path, "raw two-array archive")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ReplayError("raw two-array archive hash drift")
    with np.load(path, allow_pickle=False) as stored:
        archive_keys = tuple(stored.files)
        if len(archive_keys) != 2 or set(archive_keys) != INPUT_KEYS:
            raise ReplayError("raw archive does not contain exactly action and pixels")
        arrays = {name: np.asarray(stored[name]).copy() for name in sorted(INPUT_KEYS)}
    pixels = arrays["pixels"]
    actions = arrays["action"]
    if pixels.shape != PIXELS_SHAPE or pixels.dtype != np.dtype(np.uint8):
        raise ReplayError("raw pixels shape/dtype drift")
    if actions.shape != ACTION_SHAPE or actions.dtype != np.dtype(np.float32):
        raise ReplayError("raw action shape/dtype drift")
    if not np.isfinite(pixels).all():
        raise ReplayError("raw pixels contain nonfinite values")
    if not np.isfinite(actions[:200]).all():
        raise ReplayError("modeled actions contain nonfinite values")
    if not np.isnan(actions[200]).all():
        raise ReplayError("terminal action sentinel drift")
    audit = {
        "loaded_keys": ["action", "pixels"],
        "pixels_shape": [201, 224, 224, 3],
        "pixels_dtype": "uint8",
        "action_shape": [201, 5],
        "action_dtype": "float32",
        "pixels_finite": True,
        "modeled_actions_finite": True,
        "terminal_action_nan_sentinel": True,
        "array_sha256": {
            "pixels": array_sha256(pixels),
            "action": array_sha256(actions),
        },
        "archive_keys": sorted(archive_keys),
        "archive_sha256": observed_sha256,
        "arrays_materialized": ["action", "pixels"],
        "non_input_arrays_materialized": False,
    }
    return arrays, audit


def _load_stack() -> Stack:
    source_audit = _source_audit()
    del source_audit
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    distribution_root = REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract"
    common_path = distribution_root / "common.py"
    existing = sys.modules.get("common")
    if existing is not None and Path(str(getattr(existing, "__file__", ""))).resolve() != common_path.resolve():
        raise ReplayError("plain common module name is already occupied")
    if existing is None:
        specification = importlib.util.spec_from_file_location("common", common_path)
        if specification is None or specification.loader is None:
            raise ReplayError("cannot load exact distribution common source")
        common = importlib.util.module_from_spec(specification)
        sys.modules["common"] = common
        specification.loader.exec_module(common)
    else:
        common = existing
    runtime = common.load_runtime()
    model_io = common.load_model_io()
    device = runtime.choose_device("mps")
    if getattr(device, "type", None) != "mps":
        raise ReplayError("scientific replay did not receive the exact MPS device")
    base, contract, provenance = runtime.load_base_model(device)
    solver, v1, _models = runtime.load_solver(device)
    before = runtime.module_audit(base, solver, v1)
    if not isinstance(before, Mapping) or before.get("passed") is not True:
        raise ReplayError("frozen model/solver module audit failed before replay")
    return Stack(
        torch=torch,
        runtime=runtime,
        model_io=model_io,
        common=common,
        device=device,
        base=base,
        contract=contract,
        solver=solver,
        v1=v1,
        provenance=provenance,
        module_before=dict(before),
    )


def _prepare_episode(stack: Stack, loaded: Mapping[str, np.ndarray]) -> tuple[Any, Any, Any, Any]:
    torch = stack.torch
    pixels = np.asarray(loaded["pixels"])[::5]
    raw_actions = np.asarray(loaded["action"])[:200].astype(np.float32)
    if pixels.shape != (41, 224, 224, 3):
        raise ReplayError("independent frameskip did not produce 41 frames")
    latent_parts = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 64):
            transformed = stack.model_io.pixel_transform(
                pixels[start : start + 64], stack.contract.image_size, stack.device
            ).unsqueeze(0)
            latent_parts.append(
                stack.base.encode({"pixels": transformed})["emb"].squeeze(0).cpu()
            )
    latent = torch.cat(latent_parts).numpy().astype(np.float32)
    normalized = (
        raw_actions - stack.common.FROZEN_ACTION_MEAN
    ) / stack.common.FROZEN_ACTION_STD
    blocks = normalized.reshape(40, ACTION_DIM)
    history = np.stack([latent[index : index + HISTORY_LEN] for index in range(ROWS)])
    actions = np.stack([blocks[index : index + HISTORY_LEN] for index in range(ROWS)])
    target = latent[HISTORY_LEN:]
    history_tensor = torch.as_tensor(history, dtype=torch.float32, device=stack.device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=stack.device)
    with torch.inference_mode():
        base_prediction = stack.runtime.base_predict(
            stack.base, history_tensor, action_tensor
        )
    if (
        history_tensor.shape != (ROWS, HISTORY_LEN, LATENT_DIM)
        or action_tensor.shape != (ROWS, HISTORY_LEN, ACTION_DIM)
        or target.shape != (ROWS, LATENT_DIM)
        or base_prediction.shape != (ROWS, LATENT_DIM)
    ):
        raise ReplayError("independent frozen preprocessing shape drift")
    return (
        history_tensor,
        action_tensor,
        torch.as_tensor(target, dtype=torch.float32, device=stack.device),
        base_prediction,
    )


def _features(torch: Any, history: Any, actions: Any, current: Any, update: Any) -> Any:
    if (
        history.shape != (len(history), HISTORY_LEN, LATENT_DIM)
        or actions.shape != (len(history), HISTORY_LEN, ACTION_DIM)
        or current.shape != (len(history), LATENT_DIM)
        or update.shape != current.shape
    ):
        raise ReplayError("independent causal-feature input shape drift")
    values = (history, actions, current, update)
    if len({item.device for item in values}) != 1 or len({item.dtype for item in values}) != 1:
        raise ReplayError("independent causal-feature device/dtype drift")
    history, actions, current, update = (item.detach() for item in values)
    epsilon = torch.finfo(current.dtype).eps
    last_history = history[:, -1]
    gap = current - last_history
    current_norm = torch.linalg.vector_norm(current, dim=-1, keepdim=True)
    update_norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
    last_history_norm = torch.linalg.vector_norm(last_history, dim=-1, keepdim=True)
    gap_norm = torch.linalg.vector_norm(gap, dim=-1, keepdim=True)
    relative_update = update_norm / current_norm.clamp_min(epsilon)
    update_current_cosine = (update * current).sum(dim=-1, keepdim=True) / (
        update_norm * current_norm
    ).clamp_min(epsilon)
    update_gap_cosine = (update * gap).sum(dim=-1, keepdim=True) / (
        update_norm * gap_norm
    ).clamp_min(epsilon)
    history_change = history[:, 1:] - history[:, :-1]
    action_change = actions[:, 1:] - actions[:, :-1]
    result = torch.cat(
        (
            history.flatten(1),
            actions.flatten(1),
            current,
            update,
            current_norm,
            update_norm,
            relative_update,
            last_history_norm,
            gap_norm,
            update_current_cosine,
            update_gap_cosine,
            torch.linalg.vector_norm(history_change, dim=2),
            torch.linalg.vector_norm(action_change, dim=2),
        ),
        dim=1,
    )
    if result.shape != (len(history), FEATURE_DIM) or not bool(torch.isfinite(result).all().item()):
        raise ReplayError("independent causal-feature output drift")
    return result


def _load_gate(stack: Stack, path: Path) -> Gate:
    path = _regular_file(path, "compiled gate")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != COMPILED_GATE_KEYS:
            raise ReplayError("compiled-gate member schema drift")
        schema = np.asarray(stored["schema_version"])
        if schema.shape != () or schema.dtype != np.dtype(np.int16) or schema.item() != 1:
            raise ReplayError("compiled-gate schema_version must be scalar int16 one")
        architecture_array = np.asarray(stored["architecture"])
        candidate_array = np.asarray(stored["candidate_id"])
        if architecture_array.shape != () or candidate_array.shape != ():
            raise ReplayError("compiled-gate text identity must be scalar")
        architecture = str(architecture_array.item())
        candidate_id = str(candidate_array.item())
        weights = np.asarray(stored["weights"]).copy()
        biases = np.asarray(stored["biases"]).copy()
        thresholds = np.asarray(stored["thresholds"]).copy()
        head_names_array = np.asarray(stored["head_names"]).copy()
    if architecture not in ARCHITECTURE_HEADS or not candidate_id or candidate_id.strip() != candidate_id:
        raise ReplayError("compiled-gate identity drift")
    head_count = ARCHITECTURE_HEADS[architecture]
    head_names = tuple(str(item) for item in head_names_array.tolist())
    if (
        weights.shape != (STAGES, head_count, FEATURE_DIM)
        or biases.shape != (STAGES, head_count)
        or thresholds.shape != (STAGES,)
        or len(head_names) != head_count
        or len(set(head_names)) != head_count
        or any(not name or name.strip() != name for name in head_names)
    ):
        raise ReplayError("compiled-gate shape/head identity drift")
    for name, value in (("weights", weights), ("biases", biases), ("thresholds", thresholds)):
        if value.dtype != np.dtype(np.float32) or not np.isfinite(value).all():
            raise ReplayError(f"compiled-gate {name} must be finite float32")
    torch = stack.torch
    return Gate(
        architecture=architecture,
        candidate_id=candidate_id,
        head_names=head_names,
        weights=torch.as_tensor(weights, dtype=torch.float32, device=stack.device),
        biases=torch.as_tensor(biases, dtype=torch.float32, device=stack.device),
        thresholds=torch.as_tensor(thresholds, dtype=torch.float32, device=stack.device),
        path=path.resolve(),
        sha256=sha256_file(path),
    )


def _score(torch: Any, gate: Gate, history: Any, actions: Any, current: Any, update: Any, stage: int) -> tuple[Any, Any, Any]:
    features = _features(torch, history, actions, current, update)
    heads = features @ gate.weights[stage].transpose(0, 1) + gate.biases[stage]
    scores = torch.amin(heads, dim=1)
    if not bool(torch.isfinite(scores).all().item()):
        raise ReplayError("independent gate score became nonfinite")
    return scores, heads, features


def _dense(stack: Stack, gate: Gate | None, history: Any, actions: Any, base_prediction: Any) -> dict[str, Any]:
    torch = stack.torch
    with torch.inference_mode():
        outputs, updates = stack.solver(
            history, actions, base_prediction, max_depth=DEPTHS, return_updates=True
        )
        exits = torch.stack([outputs[index] for index in (1, 2, 3, 4)], dim=1)
        stage_current = torch.stack([outputs[index] for index in (1, 2, 3)], dim=1)
        stage_update = torch.stack([updates[index] for index in (1, 2, 3)], dim=1)
        feature_parts = [
            _features(torch, history, actions, stage_current[:, stage], stage_update[:, stage])
            for stage in range(STAGES)
        ]
        features = torch.stack(feature_parts, dim=1)
        result = {
            "exits": exits,
            "production_features": features,
            "stage_current": stage_current,
            "stage_update": stage_update,
        }
        if gate is not None:
            head_parts = []
            score_parts = []
            for stage in range(STAGES):
                heads = features[:, stage] @ gate.weights[stage].transpose(0, 1)
                heads = heads + gate.biases[stage]
                head_parts.append(heads)
                score_parts.append(torch.amin(heads, dim=1))
            result["dense_head_scores"] = torch.stack(head_parts, dim=1)
            result["dense_scores"] = torch.stack(score_parts, dim=1)
    return result


def _sparse(stack: Stack, gate: Gate, history: Any, actions: Any, base_prediction: Any) -> dict[str, Any]:
    torch = stack.torch
    if len(stack.solver.adapters) != STAGES:
        raise ReplayError("frozen solver adapter count drift")
    with torch.inference_mode():
        anchored = stack.solver.anchor(history, actions, base_prediction)
        current = anchored[1]
        last_update = current - base_prediction
        batch = len(history)
        calls = torch.ones(batch, dtype=torch.long, device=stack.device)
        scores = torch.full((batch, STAGES), float("nan"), dtype=torch.float32, device=stack.device)
        heads = torch.full((batch, STAGES, gate.head_count), float("nan"), dtype=torch.float32, device=stack.device)
        features = torch.full((batch, STAGES, FEATURE_DIM), float("nan"), dtype=torch.float32, device=stack.device)
        stage_current = torch.full((batch, STAGES, LATENT_DIM), float("nan"), dtype=torch.float32, device=stack.device)
        stage_update = torch.full_like(stage_current, float("nan"))
        reached = torch.zeros((batch, STAGES), dtype=torch.bool, device=stack.device)
        active = torch.arange(batch, device=stack.device)
        for stage, adapter in enumerate(stack.solver.adapters):
            if not active.numel():
                break
            local_current = current.index_select(0, active)
            local_update = last_update.index_select(0, active)
            local_score, local_heads, local_features = _score(
                torch,
                gate,
                history.index_select(0, active),
                actions.index_select(0, active),
                local_current,
                local_update,
                stage,
            )
            reached[active, stage] = True
            scores[active, stage] = local_score
            heads[active, stage] = local_heads
            features[active, stage] = local_features
            stage_current[active, stage] = local_current
            stage_update[active, stage] = local_update
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
    return {
        "selected": current,
        "calls": calls,
        "scores": scores,
        "head_scores": heads,
        "production_features": features,
        "stage_current": stage_current,
        "stage_update": stage_update,
        "reached": reached,
    }


def _tensor_array(value: Any, dtype: Any) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype)


def _expected_arrays(
    role: str,
    slot: int,
    loaded: Mapping[str, np.ndarray],
    stack: Stack,
    gate: Gate | None,
) -> dict[str, np.ndarray]:
    history, actions, target, base_prediction = _prepare_episode(stack, loaded)
    dense = _dense(stack, gate, history, actions, base_prediction)
    expected = {
        "episode_slot": np.full(ROWS, slot, dtype=np.int32),
        "model_step": np.arange(3, 41, dtype=np.int16),
        "history": _tensor_array(history, np.float32),
        "action_history": _tensor_array(actions, np.float32),
        "exits": _tensor_array(dense["exits"], np.float32),
    }
    if role in ("fit", "selection"):
        expected.update(
            {
                "target": _tensor_array(target, np.float32),
                "production_features": _tensor_array(
                    dense["production_features"], np.float32
                ),
                "stage_current": _tensor_array(dense["stage_current"], np.float32),
                "stage_update": _tensor_array(dense["stage_update"], np.float32),
            }
        )
        return expected
    if gate is None:
        raise ReplayError("robust expected-array replay lacks a compiled gate")
    sparse = _sparse(stack, gate, history, actions, base_prediction)
    expected.update(
        {
            "selected": _tensor_array(sparse["selected"], np.float32),
            "calls": _tensor_array(sparse["calls"], np.int8),
            "scores": _tensor_array(sparse["scores"], np.float32),
            "head_scores": _tensor_array(sparse["head_scores"], np.float32),
            "production_features": _tensor_array(
                sparse["production_features"], np.float32
            ),
            "stage_current": _tensor_array(sparse["stage_current"], np.float32),
            "stage_update": _tensor_array(sparse["stage_update"], np.float32),
            "reached": _tensor_array(sparse["reached"], np.bool_),
            "dense_scores": _tensor_array(dense["dense_scores"], np.float32),
            "dense_head_scores": _tensor_array(
                dense["dense_head_scores"], np.float32
            ),
            "dense_stage_current": _tensor_array(
                dense["stage_current"], np.float32
            ),
            "dense_stage_update": _tensor_array(dense["stage_update"], np.float32),
        }
    )
    if role == "confirmation":
        expected["target"] = _tensor_array(target, np.float32)
    return expected


ExpectedBuilder = Callable[
    [str, int, Mapping[str, np.ndarray], Stack, Gate | None],
    dict[str, np.ndarray],
]


def _exact_compute(calls: np.ndarray, head_count: int) -> dict[str, Any]:
    values = np.asarray(calls)
    if values.ndim != 1 or values.dtype != np.dtype(np.int8) or ((values < 1) | (values > 4)).any():
        raise ReplayError("call-depth vector drift")
    rows = len(values)
    solver_calls = int(values.astype(np.int64).sum())
    gate_evaluations = int(np.minimum(values, 3).astype(np.int64).sum())
    later = solver_calls - rows
    all_head_flops = head_count * AFFINE_HEAD_FLOPS
    gate_flops = gate_evaluations * (FEATURE_FLOPS + all_head_flops)
    equivalent = Fraction(gate_flops, ADAPTER_FLOPS)
    return {
        "row_count": rows,
        "base_calls": rows,
        "solver_calls": solver_calls,
        "mandatory_depth1_calls": rows,
        "later_adapter_calls": later,
        "gate_evaluations": gate_evaluations,
        "head_count": head_count,
        "base_flops": rows * BASE_FLOPS,
        "depth1_flops": rows * DEPTH1_FLOPS,
        "adapter_flops": later * ADAPTER_FLOPS,
        "gate_feature_flops": gate_evaluations * FEATURE_FLOPS,
        "gate_head_flops": gate_evaluations * all_head_flops,
        "gate_total_flops": gate_flops,
        "total_counted_flops": (
            rows * BASE_FLOPS
            + rows * DEPTH1_FLOPS
            + later * ADAPTER_FLOPS
            + gate_flops
        ),
        "gate_nonflop_operations": gate_evaluations * GATE_NONFLOPS[head_count],
        "exact_equivalent_transition_independent_solver_calls_numerator": (
            solver_calls * equivalent.denominator + equivalent.numerator
        ),
        "exact_equivalent_transition_independent_solver_calls_denominator": equivalent.denominator,
        "exact_equivalent_transition_independent_solver_calls": float(
            solver_calls + equivalent
        ),
        "historical_counting_convention": {
            "base_per_row": BASE_FLOPS,
            "mandatory_depth1_per_row": DEPTH1_FLOPS,
            "each_later_adapter": ADAPTER_FLOPS,
            "feature_per_reached_gate": FEATURE_FLOPS,
            "affine_head_per_reached_gate": AFFINE_HEAD_FLOPS,
            "activations_and_normalizations_inside_frozen_solver_omitted_as_in_V5": True,
        },
    }


def _validate_manifest_headers(
    role: str,
    regime: str,
    raw: Mapping[str, Any],
    execution: Mapping[str, Any],
    data_attempt: str = ATTEMPT,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if set(raw) != RAW_MANIFEST_KEYS:
        raise ReplayError("raw manifest closed schema drift")
    expected_execution_keys = (
        DEVELOPMENT_EXECUTION_MANIFEST_KEYS
        if role in ("fit", "selection")
        else ROBUST_EXECUTION_MANIFEST_KEYS
    )
    if set(execution) != expected_execution_keys:
        raise ReplayError("execution manifest closed schema drift")
    _assert_no_forbidden_keys(raw, "raw_manifest")
    _assert_no_forbidden_keys(execution, "execution_manifest")
    _exact_int(raw.get("schema_version"), "raw manifest schema_version")
    _exact_int(execution.get("schema_version"), "execution manifest schema_version")
    _exact_int(raw.get("created_unix_ns"), "raw manifest timestamp", minimum=1)
    _exact_int(
        execution.get("created_unix_ns"), "execution manifest timestamp", minimum=1
    )
    raw_count = _exact_int(raw.get("episode_count"), "raw episode_count", minimum=1)
    _exact_int(raw.get("replacements_used"), "raw replacements_used", minimum=0)
    execution_count = _exact_int(
        execution.get("episode_count"), "execution episode_count", minimum=1
    )
    row_count = _exact_int(execution.get("row_count"), "execution row_count", minimum=1)
    for name, value in (
        ("raw complete", raw.get("complete")),
        ("raw role isolation", raw.get("role_isolation")),
        ("raw smoke exclusion", raw.get("smoke_permanently_excluded")),
        ("execution complete", execution.get("complete")),
        (
            "execution contact materialization",
            execution.get("contact_or_privileged_materialized"),
        ),
        ("execution no gradients", execution.get("no_gradients")),
    ):
        _exact_bool(value, name)
    if (
        raw.get("schema_version") != 1
        or execution.get("schema_version") != 1
        or raw.get("attempt") != data_attempt
        or execution.get("attempt") != data_attempt
        or raw.get("role") != role
        or execution.get("role") != role
        or raw.get("regime") != regime
        or execution.get("regime") != regime
        or raw.get("complete") is not True
        or execution.get("complete") is not True
        or raw.get("role_isolation") is not True
        or raw.get("smoke_permanently_excluded") is not (role == "smoke")
        or execution.get("contact_or_privileged_materialized") is not False
        or execution.get("no_gradients") is not True
        or raw_count != execution_count
        or row_count != raw_count * ROWS
        or raw.get("raw_archive_members") != ["action", "pixels"]
        or execution.get("loaded_input_keys") != ["action", "pixels"]
    ):
        raise ReplayError("raw/execution manifest header drift")
    raw_records = raw.get("episodes")
    execution_records = execution.get("episodes")
    raw_authorization = raw.get("authorization_seal")
    if (
        not isinstance(raw_authorization, Mapping)
        or set(raw_authorization) != {"path", "sha256", "checkpoint_state"}
        or not isinstance(raw_authorization.get("path"), str)
        or not _is_sha256(raw_authorization.get("sha256"))
        or not isinstance(raw_authorization.get("checkpoint_state"), str)
    ):
        raise ReplayError("raw authorization-seal closed schema drift")
    if (
        not isinstance(raw_records, list)
        or not isinstance(execution_records, list)
        or len(raw_records) != raw_count
        or len(execution_records) != raw_count
    ):
        raise ReplayError("raw/execution episode list count drift")
    return raw_records, execution_records


def _validate_raw_record(
    role: str,
    regime: str,
    expected_slot: int,
    value: Any,
    raw_manifest: Mapping[str, Any],
    data_attempt: str = ATTEMPT,
) -> tuple[dict[str, Any], Path, dict[str, np.ndarray], dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != RAW_RECORD_KEYS:
        raise ReplayError("raw episode record closed schema drift")
    record = dict(value)
    _assert_no_forbidden_keys(record, "raw_record")
    _exact_int(record.get("schema_version"), "raw record schema_version")
    _exact_int(record.get("created_unix_ns"), "raw record timestamp", minimum=1)
    slot = _exact_int(record.get("slot"), "raw record slot", minimum=0)
    for name in ("env_seed", "policy_seed", "oracle_np_seed", "action_space_seed"):
        _exact_int(record.get(name), f"raw record {name}")
    _exact_int(record.get("raw_bytes"), "raw record byte count", minimum=1)
    for name in ("complete", "replacement_used", "orphan_adoption_safe"):
        _exact_bool(record.get(name), f"raw record {name}")
    if (
        record["schema_version"] != 1
        or record.get("attempt") != data_attempt
        or record.get("complete") is not True
        or record.get("role") != role
        or record.get("regime") != regime
        or slot != expected_slot
        or not isinstance(record.get("episode_id"), str)
        or not record["episode_id"]
        or record.get("raw_archive_members") != ["action", "pixels"]
        or record.get("orphan_adoption_safe") is not True
        or record.get("authorization_seal") != raw_manifest.get("authorization_seal")
    ):
        raise ReplayError("raw episode record identity drift")
    raw_path = _regular_file(_resolve(record["raw_path"], "raw episode"), "raw episode")
    raw_sidecar_path = raw_path.with_suffix(".json")
    raw_sidecar = _read_canonical_object(raw_sidecar_path, "raw episode sidecar")
    if raw_sidecar != record:
        raise ReplayError("raw manifest record and raw sidecar differ")
    if record["raw_bytes"] != raw_path.stat().st_size:
        raise ReplayError("raw episode byte count drift")
    loaded, audit = _load_raw_inputs(raw_path, str(record["raw_sha256"]))
    _validate_input_audit(record.get("input_loader_audit"), audit, "raw record")
    expected_metadata = {
        name: {
            "shape": list(loaded[name].shape),
            "dtype": str(loaded[name].dtype),
            "sha256": audit["array_sha256"][name],
        }
        for name in sorted(loaded)
    }
    if not _exact_json_equal(record.get("arrays"), expected_metadata):
        raise ReplayError("raw record array metadata disagrees with raw replay")
    return record, raw_sidecar_path, loaded, audit


def _validate_execution_record(
    role: str,
    raw_record: Mapping[str, Any],
    raw_sidecar_path: Path,
    value: Any,
) -> tuple[dict[str, Any], Path, Path]:
    if not isinstance(value, Mapping) or set(value) != EXECUTION_RECORD_KEYS:
        raise ReplayError("execution episode record closed schema drift")
    record = dict(value)
    _exact_int(record.get("slot"), "execution record slot", minimum=0)
    if (
        record.get("slot") != raw_record.get("slot")
        or record.get("episode_id") != raw_record.get("episode_id")
        or record.get("source_raw_path") != raw_record.get("raw_path")
        or record.get("source_raw_sha256") != raw_record.get("raw_sha256")
        or record.get("source_raw_sidecar_path") != _relative(raw_sidecar_path)
        or record.get("source_raw_sidecar_sha256") != sha256_file(raw_sidecar_path)
        or (role in ("fit", "selection") and record.get("call_histogram") is not None)
    ):
        raise ReplayError("execution record/raw identity drift")
    part_path = _regular_file(_resolve(record.get("path"), "execution part"), "execution part")
    sidecar_path = _regular_file(
        _resolve(record.get("sidecar_path"), "execution sidecar"),
        "execution sidecar",
    )
    if sha256_file(part_path) != record.get("sha256"):
        raise ReplayError("execution record part hash drift")
    if sha256_file(sidecar_path) != record.get("sidecar_sha256"):
        raise ReplayError("execution record sidecar hash drift")
    return record, part_path, sidecar_path


def _load_execution_arrays(
    role: str,
    regime: str,
    raw_record: Mapping[str, Any],
    execution_record: Mapping[str, Any],
    raw_sidecar_path: Path,
    part_path: Path,
    sidecar_path: Path,
    audit: Mapping[str, Any],
    gate: Gate | None,
    data_attempt: str = ATTEMPT,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sidecar = _read_canonical_object(sidecar_path, "execution sidecar")
    expected_keys = (
        DEVELOPMENT_SIDECAR_KEYS
        if role in ("fit", "selection")
        else ROBUST_SIDECAR_KEYS
    )
    if set(sidecar) != expected_keys:
        raise ReplayError("execution sidecar closed schema drift")
    _assert_no_forbidden_keys(sidecar, "execution_sidecar")
    _exact_int(sidecar.get("schema_version"), "execution sidecar schema_version")
    _exact_int(sidecar.get("created_unix_ns"), "execution sidecar timestamp", minimum=1)
    _exact_int(sidecar.get("slot"), "execution sidecar slot", minimum=0)
    _exact_int(sidecar.get("bytes"), "execution sidecar byte count", minimum=1)
    for name in ("complete", "contact_or_privileged_materialized", "no_gradients"):
        _exact_bool(sidecar.get(name), f"execution sidecar {name}")
    if (
        sidecar["schema_version"] != 1
        or sidecar.get("attempt") != data_attempt
        or sidecar.get("role") != role
        or sidecar.get("regime") != regime
        or sidecar.get("complete") is not True
        or sidecar.get("slot") != raw_record.get("slot")
        or sidecar.get("episode_id") != raw_record.get("episode_id")
        or sidecar.get("raw_path") != raw_record.get("raw_path")
        or sidecar.get("raw_sha256") != raw_record.get("raw_sha256")
        or sidecar.get("raw_sidecar_path") != _relative(raw_sidecar_path)
        or sidecar.get("raw_sidecar_sha256") != sha256_file(raw_sidecar_path)
        or sidecar.get("contact_or_privileged_materialized") is not False
        or sidecar.get("no_gradients") is not True
        or sidecar.get("path") != _relative(part_path)
        or sidecar.get("sha256") != sha256_file(part_path)
        or sidecar.get("bytes") != part_path.stat().st_size
    ):
        raise ReplayError("execution sidecar identity/header drift")
    _validate_input_audit(sidecar.get("input_loader_audit"), audit, "execution sidecar")
    head_count = None if gate is None else gate.head_count
    contract = _array_contract(role, head_count)
    metadata = sidecar.get("arrays")
    if not isinstance(metadata, Mapping) or set(metadata) != set(contract):
        raise ReplayError("execution sidecar array metadata schema drift")
    observed: dict[str, np.ndarray] = {}
    with np.load(part_path, allow_pickle=False) as stored:
        if set(stored.files) != set(contract):
            raise ReplayError("execution part member schema drift")
        for name, (shape, dtype) in contract.items():
            item = metadata[name]
            if not isinstance(item, Mapping) or set(item) != {"shape", "dtype", "sha256"}:
                raise ReplayError(f"execution array metadata schema drift: {name}")
            _exact_shape(item.get("shape"), shape, f"execution arrays.{name}.shape")
            if item.get("dtype") != dtype or not _is_sha256(item.get("sha256")):
                raise ReplayError(f"execution array metadata identity drift: {name}")
            array = np.asarray(stored[name]).copy()
            if array.shape != shape or str(array.dtype) != dtype:
                raise ReplayError(f"execution part shape/dtype drift: {name}")
            if array_sha256(array) != item["sha256"]:
                raise ReplayError(f"execution part array hash drift: {name}")
            observed[name] = array
    if role in ("fit", "selection"):
        for name in ("primitive_feature_reconstruction_exact",):
            _exact_bool(sidecar.get(name), f"execution sidecar {name}")
        if sidecar.get("primitive_feature_reconstruction_exact") is not True:
            raise ReplayError("development primitive reconstruction assertion failed")
    else:
        if gate is None:
            raise ReplayError("robust execution sidecar lacks compiled gate")
        for name in ("target_persisted", "target_loss_computed"):
            _exact_bool(sidecar.get(name), f"execution sidecar {name}")
        histogram = sidecar.get("call_histogram")
        if (
            not isinstance(histogram, list)
            or len(histogram) != DEPTHS
            or any(type(item) is not int or item < 0 for item in histogram)
            or sum(histogram) != ROWS
        ):
            raise ReplayError("execution sidecar call histogram type/count drift")
        calls = observed["calls"]
        recomputed_histogram = np.bincount(calls.astype(np.int64), minlength=5)[1:5].tolist()
        expected_reached = np.arange(1, STAGES + 1, dtype=np.int8)[None, :] <= calls[:, None]
        if (
            histogram != recomputed_histogram
            or execution_record.get("call_histogram") != histogram
            or not np.array_equal(observed["reached"], expected_reached)
            or not _exact_json_equal(
                sidecar.get("exact_compute"), _exact_compute(calls, gate.head_count)
            )
            or sidecar.get("compiled_gate_path") != _relative(gate.path)
            or sidecar.get("compiled_gate_sha256") != gate.sha256
            or sidecar.get("architecture") != gate.architecture
            or sidecar.get("candidate_id") != gate.candidate_id
            or sidecar.get("head_names") != list(gate.head_names)
            or sidecar.get("target_persisted") is not (role == "confirmation")
            or sidecar.get("target_loss_computed") is not False
        ):
            raise ReplayError("robust execution sidecar gate/call/compute drift")
    return observed, sidecar


def _validate_manifest_links(
    role: str,
    raw_path: Path,
    execution: Mapping[str, Any],
    gate: Gate | None,
    data_attempt: str = ATTEMPT,
) -> None:
    raw_hash = sha256_file(raw_path)
    if (
        execution.get("raw_manifest_path") != _relative(raw_path)
        or execution.get("raw_manifest_sha256") != raw_hash
        or execution.get("source_raw_manifest_sha256") != raw_hash
        or not _exact_json_equal(
            execution.get("module_before"), execution.get("module_after")
        )
        or not isinstance(execution.get("module_before"), Mapping)
        or execution["module_before"].get("passed") is not True
    ):
        raise ReplayError("execution manifest raw/module binding drift")
    source_bindings = execution.get("source_bindings")
    expected_source_keys = (
        DEVELOPMENT_SOURCE_BINDING_KEYS
        if role in ("fit", "selection")
        else ROBUST_SOURCE_BINDING_KEYS
    )
    if (
        not isinstance(source_bindings, Mapping)
        or set(source_bindings) != expected_source_keys
    ):
        raise ReplayError("execution manifest source-binding closed schema drift")
    for name, value in source_bindings.items():
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ReplayError(f"execution source binding schema drift: {name}")
        path = _regular_file(_resolve(value.get("path"), f"source binding {name}"), f"source binding {name}")
        if sha256_file(path) != value.get("sha256"):
            raise ReplayError(f"execution source binding hash drift: {name}")
    authorization = execution.get("authorization")
    expected_authorization_keys = (
        DEVELOPMENT_AUTHORIZATION_KEYS
        if role in ("fit", "selection")
        else ROBUST_AUTHORIZATION_KEYS
    )
    if (
        not isinstance(authorization, Mapping)
        or set(authorization) != expected_authorization_keys
        or authorization.get("active_attempt") != data_attempt
        or not _is_sha256(authorization.get("state_sha256"))
        or not _is_sha256(authorization.get("seal_sha256"))
    ):
        raise ReplayError("execution authorization closed schema drift")
    if role in ("fit", "selection"):
        _exact_bool(
            authorization.get("exact_locked_snapshot_unchanged"),
            "execution authorization exact_locked_snapshot_unchanged",
        )
    if role in ("smoke", "confirmation"):
        if gate is None:
            raise ReplayError("robust manifest lacks compiled gate")
        _exact_int(execution.get("head_count"), "execution manifest head_count")
        for name in (
            "target_persisted",
            "target_loss_computed_during_execution",
            "all_equivalence_checks_passed",
        ):
            _exact_bool(execution.get(name), f"execution manifest {name}")
        if (
            execution.get("compiled_gate_path") != _relative(gate.path)
            or execution.get("compiled_gate_sha256") != gate.sha256
            or execution.get("architecture") != gate.architecture
            or execution.get("candidate_id") != gate.candidate_id
            or execution.get("head_names") != list(gate.head_names)
            or execution.get("head_count") != gate.head_count
            or execution.get("strict_continue_operator") != ">"
            or execution.get("target_persisted") is not (role == "confirmation")
            or execution.get("target_loss_computed_during_execution") is not False
            or execution.get("all_equivalence_checks_passed") is not True
        ):
            raise ReplayError("robust execution manifest gate binding drift")


def _validate_aggregate(
    role: str,
    execution: Mapping[str, Any],
    expected_parts: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    if role not in ("fit", "selection"):
        return {"applicable": False, "path": None, "sha256": None, "arrays": None, "exact": None}
    if execution.get("aggregate_role_keys") != list(AGGREGATE_KEYS):
        raise ReplayError("development aggregate key-order drift")
    path = _regular_file(
        _resolve(execution.get("aggregate_role_path"), "development aggregate"),
        "development aggregate",
    )
    if sha256_file(path) != execution.get("aggregate_role_sha256"):
        raise ReplayError("development aggregate file hash drift")
    expected = {
        name: np.concatenate([part[name] for part in expected_parts], axis=0)
        for name in AGGREGATE_KEYS
    }
    hashes: dict[str, str] = {}
    with np.load(path, allow_pickle=False) as stored:
        if tuple(stored.files) != AGGREGATE_KEYS:
            raise ReplayError("development aggregate closed member/order drift")
        for name in AGGREGATE_KEYS:
            observed = np.asarray(stored[name]).copy()
            if not _arrays_exact(observed, expected[name]):
                raise ReplayError(f"development aggregate/replay drift: {name}")
            hashes[name] = array_sha256(observed)
    return {
        "applicable": True,
        "path": _relative(path),
        "sha256": sha256_file(path),
        "arrays": hashes,
        "exact": True,
    }


def qualify_regime(
    role: str,
    regime: str,
    *,
    runtime_audit: Mapping[str, Any] | None = None,
    stack_loader: Callable[[], Stack] = _load_stack,
    expected_builder: ExpectedBuilder = _expected_arrays,
    attempt_root: Path = ATTEMPT_ROOT,
) -> dict[str, Any]:
    """Recompute one complete role-by-DGP artifact set without writing it."""

    if role not in ROLES or regime not in REGIMES:
        raise ReplayError(f"unsupported scientific replay identity: {role}/{regime}")
    data_attempt = ATTEMPT
    data_root = Path(attempt_root)
    if role == "fit" and data_root.resolve() == ATTEMPT_ROOT.resolve():
        data_attempt = INHERITED_FIT_ATTEMPT
        data_root = INHERITED_FIT_ROOT
    root = data_root / "data" / role / regime
    before_snapshot = _directory_snapshot(root)
    source_hashes = _source_audit()
    raw_manifest_path = _regular_file(root / "raw_manifest.json", "raw manifest")
    execution_manifest_path = _regular_file(
        root / "execution_manifest.json", "execution manifest"
    )
    raw = _read_canonical_object(raw_manifest_path, "raw manifest")
    execution = _read_canonical_object(execution_manifest_path, "execution manifest")
    raw_records, execution_records = _validate_manifest_headers(
        role, regime, raw, execution, data_attempt
    )

    stack = stack_loader()
    if getattr(stack.device, "type", None) != "mps" and stack_loader is _load_stack:
        raise ReplayError("production scientific replay is not on MPS")
    if not _exact_json_equal(execution.get("module_before"), dict(stack.module_before)):
        raise ReplayError("execution manifest module audit differs from fresh replay")
    if not _exact_json_equal(execution.get("base_provenance"), stack.provenance):
        raise ReplayError("execution manifest base provenance differs from fresh replay")

    gate: Gate | None = None
    if role in ("smoke", "confirmation"):
        gate_path = _regular_file(
            _resolve(execution.get("compiled_gate_path"), "compiled gate"),
            "compiled gate",
        )
        gate = _load_gate(stack, gate_path)
    _validate_manifest_links(
        role, raw_manifest_path, execution, gate, data_attempt
    )

    episode_results: list[dict[str, Any]] = []
    expected_parts: list[dict[str, np.ndarray]] = []
    all_calls: list[np.ndarray] = []
    for index, (raw_value, execution_value) in enumerate(
        zip(raw_records, execution_records, strict=True)
    ):
        raw_record, raw_sidecar_path, loaded, loader_audit = _validate_raw_record(
            role, regime, index, raw_value, raw, data_attempt
        )
        execution_record, part_path, execution_sidecar_path = _validate_execution_record(
            role, raw_record, raw_sidecar_path, execution_value
        )
        observed_arrays, _sidecar = _load_execution_arrays(
            role,
            regime,
            raw_record,
            execution_record,
            raw_sidecar_path,
            part_path,
            execution_sidecar_path,
            loader_audit,
            gate,
            data_attempt,
        )
        expected_arrays = expected_builder(
            role, int(raw_record["slot"]), loaded, stack, gate
        )
        if set(expected_arrays) != set(observed_arrays):
            raise ReplayError("independent replay expected member schema drift")
        for name, expected in expected_arrays.items():
            if not _arrays_exact(observed_arrays[name], expected):
                raise ReplayError(
                    f"persisted tensor differs from independent replay: "
                    f"{raw_record['episode_id']}:{name}"
                )
        expected_parts.append(expected_arrays)
        if gate is not None:
            all_calls.append(expected_arrays["calls"])
        episode_results.append(
            {
                "slot": int(raw_record["slot"]),
                "episode_id": str(raw_record["episode_id"]),
                "raw": _link(_resolve(raw_record["raw_path"], "raw episode")),
                "raw_sidecar": _link(raw_sidecar_path),
                "execution_part": _link(part_path),
                "execution_sidecar": _link(execution_sidecar_path),
                "input_loader_audit_sha256": canonical_sha256(loader_audit),
                "array_sha256": {
                    name: array_sha256(expected_arrays[name])
                    for name in sorted(expected_arrays)
                },
                "every_persisted_tensor_exact": True,
            }
        )

    aggregate = _validate_aggregate(role, execution, expected_parts)
    if gate is not None:
        concatenated_calls = np.concatenate(all_calls)
        if not _exact_json_equal(
            execution.get("exact_compute"),
            _exact_compute(concatenated_calls, gate.head_count),
        ):
            raise ReplayError("robust execution manifest exact-compute drift")
    module_after = stack.runtime.module_audit(stack.base, stack.solver, stack.v1)
    if not _exact_json_equal(dict(module_after), dict(stack.module_before)):
        raise ReplayError("frozen model/solver changed during independent replay")
    parameters = list(stack.base.parameters()) + list(stack.solver.parameters())
    if any(getattr(parameter, "grad", None) is not None for parameter in parameters):
        raise ReplayError("gradient materialized during independent replay")
    after_snapshot = _directory_snapshot(root)
    if before_snapshot != after_snapshot:
        raise ReplayError("scientific artifact directory changed during read-only replay")
    if runtime_audit is not None:
        if (
            not isinstance(runtime_audit, Mapping)
            or runtime_audit.get("passed") is not True
            or runtime_audit.get("requested_role") != "independent_verification"
            or runtime_audit.get("mps", {}).get("required") is not True
            or runtime_audit.get("mps", {}).get("available") is not True
            or runtime_audit.get("read_only_preflight") is not True
        ):
            raise ReplayError("independent replay runtime/MPS audit drift")
        runtime_record: Mapping[str, Any] | None = dict(runtime_audit)
    else:
        runtime_record = None
    return {
        "schema_version": 1,
        "artifact_type": "v008_independent_scientific_replay_regime",
        "attempt": ATTEMPT,
        "role": role,
        "regime": regime,
        "raw_manifest": _link(raw_manifest_path),
        "execution_manifest": _link(execution_manifest_path),
        "compiled_gate": None if gate is None else _link(gate.path),
        "episode_count": len(episode_results),
        "row_count": len(episode_results) * ROWS,
        "episodes": episode_results,
        "aggregate": aggregate,
        "runtime_audit": runtime_record,
        "source_hashes": source_hashes,
        "module_before": dict(stack.module_before),
        "module_after": dict(module_after),
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "aggregate_exact": aggregate["exact"],
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_tree_unchanged": True,
        "read_only": True,
        "passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--regime", required=True, choices=REGIMES)
    arguments = parser.parse_args(argv)
    result = qualify_regime(arguments.role, arguments.regime)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
