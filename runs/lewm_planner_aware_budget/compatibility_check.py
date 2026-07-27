#!/usr/bin/env python3
"""Fail-closed compatibility check for the Task A CEM and PushT refiner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = Path(
    "/Users/rishisim/Documents/research/lewm-p2-contact/"
    "runs/lewm_pusht_replication_pilot"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "29be92d39a2efba992c8e2dcfd289956d04f057d63d8ed1042de25c305641a6f"
)
EXPECTED_ARCHITECTURE = {
    "latent_dim": 192,
    "action_dim": 10,
    "history": 3,
    "hidden": 256,
    "iteration_dim": 16,
    "depths": [1, 2, 3, 4],
}
PLANNER_OBSERVATION_HISTORY = 1
PLANNER_HORIZON = 5
PREDICTOR_HISTORY_LIMIT = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def planner_history_lengths(
    observation_history: int = PLANNER_OBSERVATION_HISTORY,
    horizon: int = PLANNER_HORIZON,
    predictor_limit: int = PREDICTOR_HISTORY_LIMIT,
) -> list[int]:
    if min(observation_history, horizon, predictor_limit) <= 0:
        raise ValueError("history, horizon, and predictor limit must be positive")
    return [
        min(predictor_limit, observation_history + transition)
        for transition in range(horizon)
    ]


def inspect_artifact(root: Path) -> dict[str, Any]:
    checkpoint = root / "checkpoints/refiner.pt"
    fit_artifacts = root / "FIT_ARTIFACTS.npz"
    if not checkpoint.is_file() or not fit_artifacts.is_file():
        raise FileNotFoundError(
            f"PushT refiner export is incomplete under read-only root {root}"
        )
    observed_hash = sha256(checkpoint)
    if observed_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"refiner hash mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, "
            f"got {observed_hash}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"state_dict", "training_seed", "best_epoch", "architecture"}:
        raise RuntimeError(f"unexpected refiner checkpoint keys: {sorted(payload)}")
    architecture = payload["architecture"]
    if architecture != EXPECTED_ARCHITECTURE:
        raise RuntimeError(f"unexpected refiner architecture: {architecture}")
    first_weight = payload["state_dict"]["block.0.weight"]
    expected_width = (
        architecture["history"] * architecture["latent_dim"]
        + architecture["history"] * architecture["action_dim"]
        + architecture["latent_dim"]
        + architecture["iteration_dim"]
    )
    if tuple(first_weight.shape) != (architecture["hidden"], expected_width):
        raise RuntimeError(
            f"refiner first-layer shape mismatch: {tuple(first_weight.shape)}"
        )

    import numpy as np

    with np.load(fit_artifacts, allow_pickle=False) as stored:
        if "action_mean" not in stored or "action_scale" not in stored:
            raise RuntimeError("fit artifact lacks action normalization")
        action_mean = stored["action_mean"]
        action_scale = stored["action_scale"]
    if action_mean.shape != (architecture["action_dim"],) or action_scale.shape != (
        architecture["action_dim"],
    ):
        raise RuntimeError("refiner action normalization shape mismatch")

    lengths = planner_history_lengths()
    incompatible = [
        index for index, length in enumerate(lengths)
        if length != architecture["history"]
    ]
    return {
        "compatible": not incompatible,
        "decision": (
            "compatible"
            if not incompatible
            else "blocked_requires_prefix_history_training_export"
        ),
        "checkpoint_sha256": observed_hash,
        "architecture": architecture,
        "first_layer_shape": list(first_weight.shape),
        "planner": {
            "observation_history": PLANNER_OBSERVATION_HISTORY,
            "horizon": PLANNER_HORIZON,
            "predictor_history_limit": PREDICTOR_HISTORY_LIMIT,
            "transition_history_lengths": lengths,
        },
        "incompatible_transition_indices": incompatible,
        "action_normalization": {
            "source": "PushT WeakPolicy fit transitions",
            "coordinates": int(action_mean.size),
            "planner_mapping_exported": False,
        },
        "safe_to_implement_adapter": False if incompatible else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    args = parser.parse_args()
    print(json.dumps(inspect_artifact(args.artifact_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
