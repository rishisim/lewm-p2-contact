#!/usr/bin/env python3
"""Materialize the predeclared even/odd consumed-development split."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROGRAM_ROOT = ROOT.parents[1]
SOURCE = PROGRAM_ROOT / "cycles/cycle_004_threshold_recalibration/data/prospective_execution.npz"
PROTOCOL = ROOT / "development/stage_specific_threshold_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "frozen_before_any_candidate_specific_counterfactual":
        raise RuntimeError("threshold protocol is not frozen")
    with np.load(SOURCE, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    episode_id = arrays["episode_id"].astype(np.int64)
    if not np.array_equal(np.unique(episode_id), np.arange(300)):
        raise RuntimeError("source episode set drift")
    roles = {"selection": episode_id % 2 == 0, "validation": episode_id % 2 == 1}
    records = {}
    for role, mask in roles.items():
        path = ROOT / f"development/{role}_consumed.npz"
        subset = {name: value[mask] for name, value in arrays.items()}
        unique = np.unique(subset["episode_id"])
        if len(unique) != 150 or len(subset["target"]) != 5_700:
            raise RuntimeError(f"unexpected {role} split shape")
        expected_parity = 0 if role == "selection" else 1
        if np.any(unique % 2 != expected_parity):
            raise RuntimeError(f"{role} parity drift")
        atomic_npz(path, subset)
        records[role] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "episode_count": 150,
            "row_count": 5_700,
            "first_episode_id": int(unique[0]),
            "last_episode_id": int(unique[-1]),
            "episode_ids": unique.tolist(),
        }
    if set(records["selection"]["episode_ids"]) & set(records["validation"]["episode_ids"]):
        raise RuntimeError("development split overlap")
    manifest = {
        "schema_version": 1,
        "status": "materialized_before_candidate_specific_selection",
        "protocol_sha256": sha256(PROTOCOL),
        "source_execution_sha256": sha256(SOURCE),
        "split_rule": "even episode id selection; odd episode id validation",
        "roles": records,
        "candidate_counterfactuals_evaluated": 0,
        "validation_candidate_metrics_accessed": False,
        "cycle_004_consumed": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / "development/split_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
