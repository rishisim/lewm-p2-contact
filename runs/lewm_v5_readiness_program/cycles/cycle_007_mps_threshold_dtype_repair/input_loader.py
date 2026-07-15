#!/usr/bin/env python3
"""Fail-closed raw episode loader for model/gate inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


INPUT_ALLOWLIST = frozenset({"pixels", "action"})
FORBIDDEN_MODEL_GATE_KEYS = frozenset(
    {
        "contact",
        "observation",
        "qpos",
        "qvel",
        "privileged_target_block_pos",
        "privileged_target_block_yaw",
        "reward",
        "terminated",
        "truncated",
    }
)


def load_model_gate_inputs(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Copy only pixels/action from a lazy NPZ archive.

    Merely listing archive members does not materialize their arrays.  The only
    two subscripts below are derived from the immutable allowlist.
    """
    with np.load(Path(path), allow_pickle=False) as stored:
        archive_keys = tuple(stored.files)
        if not INPUT_ALLOWLIST.issubset(archive_keys):
            raise RuntimeError("raw episode is missing an allowlisted input")
        loaded = {key: stored[key].copy() for key in sorted(INPUT_ALLOWLIST)}
    if set(loaded) != INPUT_ALLOWLIST:
        raise RuntimeError("input loader escaped its allowlist")
    if set(loaded) & FORBIDDEN_MODEL_GATE_KEYS:
        raise RuntimeError("privileged field entered the model/gate loader")
    pixels = loaded["pixels"]
    action = loaded["action"]
    if len(pixels) != 201 or action.shape != (201, 5):
        raise RuntimeError(f"unexpected raw shapes: {pixels.shape}, {action.shape}")
    if not np.isfinite(action[:200]).all():
        raise RuntimeError("nonfinite action in the modeled horizon")
    return loaded, {
        "loaded_keys": sorted(loaded),
        "archive_key_names_observed_without_array_load": sorted(archive_keys),
        "forbidden_arrays_materialized": False,
        "contact_or_privileged_loaded": False,
    }

