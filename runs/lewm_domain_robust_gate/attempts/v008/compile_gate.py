#!/usr/bin/env python3
"""Compile a selected raw-affine robust gate into its immutable runtime form.

Fitting code folds every fit-only transform into raw-feature affine tensors.
This compiler performs no fitting and no recentering: it validates the selected
object, fixes float32 runtime arithmetic, preserves head order, and writes a
small self-describing NPZ plus a cryptographic manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from counted_features import FEATURE_DIM, FROZEN_FEATURE_NAMES
from flops import derive_exact_costs, gate_cost


STAGES = 3
ARCHITECTURE_HEADS = {
    "balanced_pooled_dual": 2,
    "domain_envelope_eight": 8,
}
COMPILED_KEYS = frozenset(
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


def _scalar_text(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar string")
    result = str(array.reshape(()).item())
    if not result or result.strip() != result:
        raise ValueError(f"{name} must be a nonempty canonical string")
    return result


def validate_compiled_arrays(
    *,
    architecture: str,
    candidate_id: str,
    weights: np.ndarray,
    biases: np.ndarray,
    thresholds: np.ndarray,
    head_names: Sequence[str] | np.ndarray,
    require_float32: bool,
) -> dict[str, Any]:
    """Validate shapes, dtypes, finiteness, and unambiguous head identity."""

    if architecture not in ARCHITECTURE_HEADS:
        raise ValueError(f"unknown architecture: {architecture}")
    if not candidate_id or candidate_id.strip() != candidate_id:
        raise ValueError("candidate_id must be a nonempty canonical string")
    head_count = ARCHITECTURE_HEADS[architecture]
    weight_array = np.asarray(weights)
    bias_array = np.asarray(biases)
    threshold_array = np.asarray(thresholds)
    names = tuple(str(item) for item in np.asarray(head_names).tolist())
    if weight_array.shape != (STAGES, head_count, FEATURE_DIM):
        raise ValueError(
            "compiled weight shape drift: "
            f"expected={(STAGES, head_count, FEATURE_DIM)} observed={weight_array.shape}"
        )
    if bias_array.shape != (STAGES, head_count):
        raise ValueError("compiled bias shape drift")
    if threshold_array.shape != (STAGES,):
        raise ValueError("compiled threshold shape drift")
    if len(names) != head_count or len(set(names)) != head_count:
        raise ValueError("head names must be unique and match head_count")
    if any(not name or name.strip() != name for name in names):
        raise ValueError("head names must be nonempty canonical strings")
    for name, value in (
        ("weights", weight_array),
        ("biases", bias_array),
        ("thresholds", threshold_array),
    ):
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{name} must be floating point")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains a nonfinite value")
        if require_float32 and value.dtype != np.dtype(np.float32):
            raise TypeError(f"runtime {name} must be exactly float32")
    return {
        "architecture": architecture,
        "candidate_id": candidate_id,
        "head_count": head_count,
        "head_names": list(names),
        "stage_count": STAGES,
        "feature_dim": FEATURE_DIM,
    }


def compile_arrays(
    *,
    architecture: str,
    candidate_id: str,
    weights: np.ndarray,
    biases: np.ndarray,
    thresholds: np.ndarray,
    head_names: Sequence[str] | np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Cast a validated selected raw-affine gate to frozen float32 tensors."""

    source_validation = validate_compiled_arrays(
        architecture=architecture,
        candidate_id=candidate_id,
        weights=weights,
        biases=biases,
        thresholds=thresholds,
        head_names=head_names,
        require_float32=False,
    )
    source_weights = np.asarray(weights)
    source_biases = np.asarray(biases)
    source_thresholds = np.asarray(thresholds)
    compiled = {
        "schema_version": np.asarray(1, dtype=np.int16),
        "architecture": np.asarray(architecture),
        "candidate_id": np.asarray(candidate_id),
        "weights": np.ascontiguousarray(source_weights, dtype=np.float32),
        "biases": np.ascontiguousarray(source_biases, dtype=np.float32),
        "thresholds": np.ascontiguousarray(source_thresholds, dtype=np.float32),
        "head_names": np.asarray(tuple(str(item) for item in head_names)),
    }
    validate_compiled_arrays(
        architecture=architecture,
        candidate_id=candidate_id,
        weights=compiled["weights"],
        biases=compiled["biases"],
        thresholds=compiled["thresholds"],
        head_names=compiled["head_names"],
        require_float32=True,
    )

    def maximum_cast_delta(source: np.ndarray, runtime: np.ndarray) -> float:
        source64 = np.asarray(source, dtype=np.float64)
        runtime64 = np.asarray(runtime, dtype=np.float64)
        return float(np.max(np.abs(source64 - runtime64), initial=0.0))

    metadata = {
        **source_validation,
        "runtime_dtype": "float32",
        "fit_only_transforms_already_folded_into_raw_affine_tensors": True,
        "compiler_performed_fitting_or_recentering": False,
        "float32_cast_max_abs": {
            "weights": maximum_cast_delta(source_weights, compiled["weights"]),
            "biases": maximum_cast_delta(source_biases, compiled["biases"]),
            "thresholds": maximum_cast_delta(
                source_thresholds, compiled["thresholds"]
            ),
        },
        "array_sha256": {
            key: array_sha256(value)
            for key, value in compiled.items()
        },
        "inference_score": "minimum_over_all_stage_specific_affine_heads",
        "continue_rule": "score_strictly_greater_than_stage_threshold",
        "gate_cost": gate_cost(source_validation["head_count"]),
    }
    return compiled, metadata


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive_temporary(temporary: Path, destination: Path) -> None:
    """Atomically create an immutable destination without replacement."""

    os.link(temporary, destination, follow_symlinks=False)
    _fsync_parent_directory(destination)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        _publish_exclusive_temporary(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_exclusive_temporary(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(path)


def compile_gate_file(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Compile an unpadded selected ``gate_fit.npz`` without outcome access."""

    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with np.load(source_path, allow_pickle=False) as stored:
        required = {
            "architecture",
            "candidate_id",
            "weights",
            "biases",
            "thresholds",
            "head_names",
        }
        missing = sorted(required - set(stored.files))
        if missing:
            raise RuntimeError(f"selected gate fit is missing keys: {missing}")
        architecture = _scalar_text(stored["architecture"], "architecture")
        candidate_id = _scalar_text(stored["candidate_id"], "candidate_id")
        arrays, metadata = compile_arrays(
            architecture=architecture,
            candidate_id=candidate_id,
            weights=stored["weights"].copy(),
            biases=stored["biases"].copy(),
            thresholds=stored["thresholds"].copy(),
            head_names=stored["head_names"].copy(),
        )
        observed_source_keys = sorted(stored.files)
    _atomic_npz(output_path, arrays)
    manifest = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "source_gate_fit_path": str(source_path),
        "source_gate_fit_sha256": sha256_file(source_path),
        "source_keys": observed_source_keys,
        "compiled_gate_path": str(output_path),
        "compiled_gate_sha256": sha256_file(output_path),
        "compiled_keys": sorted(arrays),
        "feature_names_sha256": hashlib.sha256(
            ("\n".join(FROZEN_FEATURE_NAMES) + "\n").encode("utf-8")
        ).hexdigest(),
        "feature_graph": "exact source-level V5 v004 carry-forward",
        "scientific_intervention": "gate/routing policy only",
        "target_or_contact_arrays_opened": False,
        "prior_outcome_arrays_opened": False,
        "metadata": metadata,
        "complete_compute_derivation": derive_exact_costs(),
        "passed": True,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    result = compile_gate_file(arguments.source, arguments.output, arguments.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
