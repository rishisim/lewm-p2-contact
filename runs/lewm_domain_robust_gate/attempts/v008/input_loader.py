#!/usr/bin/env python3
"""Allowlisted loader for a raw domain-robust-gate episode.

Raw rollout archives in this study contain exactly two arrays.  Keeping the
loader small and exact makes it impossible for model or gate code to
materialize any other environment channel through this interface.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


INPUT_ALLOWLIST = frozenset({"action", "pixels"})
PIXELS_SHAPE = (201, 224, 224, 3)
ACTION_SHAPE = (201, 5)
MODELED_ACTION_ROWS = 200


class InputContractError(RuntimeError):
    """A raw archive cannot satisfy the sealed two-array input contract."""


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


def validate_model_gate_inputs(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Validate only the two materialized causal input arrays."""
    if set(arrays) != INPUT_ALLOWLIST:
        raise InputContractError(
            f"raw input members must be exactly {sorted(INPUT_ALLOWLIST)}"
        )
    pixels = np.asarray(arrays["pixels"])
    action = np.asarray(arrays["action"])
    if pixels.shape != PIXELS_SHAPE or pixels.dtype != np.uint8:
        raise InputContractError(
            f"unexpected pixels contract: shape={pixels.shape} dtype={pixels.dtype}"
        )
    if action.shape != ACTION_SHAPE or action.dtype != np.float32:
        raise InputContractError(
            f"unexpected action contract: shape={action.shape} dtype={action.dtype}"
        )
    if not np.isfinite(pixels).all():
        raise InputContractError("pixels contain a nonfinite value")
    if not np.isfinite(action[:MODELED_ACTION_ROWS]).all():
        raise InputContractError("a modeled action row contains a nonfinite value")
    if not np.isnan(action[MODELED_ACTION_ROWS]).all():
        raise InputContractError("terminal action row is not the all-NaN sentinel")
    return {
        "loaded_keys": sorted(INPUT_ALLOWLIST),
        "pixels_shape": list(pixels.shape),
        "pixels_dtype": str(pixels.dtype),
        "action_shape": list(action.shape),
        "action_dtype": str(action.dtype),
        "pixels_finite": True,
        "modeled_actions_finite": True,
        "terminal_action_nan_sentinel": True,
        "array_sha256": {
            "pixels": array_sha256(pixels),
            "action": array_sha256(action),
        },
    }


def load_model_gate_inputs(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Copy exactly pixels/action from a lazy NPZ archive.

    ``NpzFile.files`` reads the ZIP member names, not the member arrays.  The
    only array subscripts below come from the immutable two-key allowlist.
    """
    path = Path(path)
    observed_file_sha256 = sha256_file(path)
    if expected_sha256 is not None and observed_file_sha256 != expected_sha256:
        raise InputContractError(f"raw archive hash drift: {path}")
    with np.load(path, allow_pickle=False) as stored:
        archive_keys = tuple(stored.files)
        if set(archive_keys) != INPUT_ALLOWLIST or len(archive_keys) != len(
            INPUT_ALLOWLIST
        ):
            raise InputContractError(
                f"raw archive members must be exactly {sorted(INPUT_ALLOWLIST)}"
            )
        loaded = {
            key: np.asarray(stored[key]).copy() for key in sorted(INPUT_ALLOWLIST)
        }
    audit = validate_model_gate_inputs(loaded)
    audit.update(
        {
            "archive_keys": sorted(archive_keys),
            "archive_sha256": observed_file_sha256,
            "arrays_materialized": sorted(loaded),
            "non_input_arrays_materialized": False,
        }
    )
    return loaded, audit


__all__ = [
    "ACTION_SHAPE",
    "INPUT_ALLOWLIST",
    "InputContractError",
    "MODELED_ACTION_ROWS",
    "PIXELS_SHAPE",
    "array_sha256",
    "load_model_gate_inputs",
    "sha256_file",
    "validate_model_gate_inputs",
]
