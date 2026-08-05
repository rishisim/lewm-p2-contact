#!/usr/bin/env python3
"""Post-terminal contact, motion, phase, reward, and success interpretation."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from study_common import (
    ATTEMPT_ROOT,
    REGIMES,
    REPO_ROOT,
    append_ledger,
    assert_runtime_contract,
    atomic_json,
    raw_manifest_path,
    read_json,
    relative_to_repo,
    sha256_file,
)


def scalar_series(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        return array.astype(np.float64)
    return array.reshape(len(array), -1).mean(axis=1, dtype=np.float64)


def summarize_episode(path: Any) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        keys = set(stored.files)
        result: dict[str, Any] = {"available_keys": sorted(keys)}
        if "proprio_gripper_contact" in keys:
            contact = scalar_series(stored["proprio_gripper_contact"])
            result["contact_rate"] = float(np.mean(contact > 0))
            result["contact_mean"] = float(contact.mean())
            result["contact_phase"] = [
                float(np.mean(chunk > 0))
                for chunk in np.array_split(contact, 3)
            ]
        if "privileged_block_0_pos" in keys:
            position = np.asarray(
                stored["privileged_block_0_pos"], dtype=np.float64
            ).reshape(201, -1)
            motion = np.linalg.vector_norm(np.diff(position, axis=0), axis=1)
            result["block_motion_total"] = float(motion.sum())
            result["block_motion_mean_step"] = float(motion.mean())
            result["motion_phase"] = [
                float(chunk.sum()) for chunk in np.array_split(motion, 3)
            ]
        if "reward" in keys:
            reward = scalar_series(stored["reward"])
            result["reward_total"] = float(reward.sum())
            result["reward_phase"] = [
                float(chunk.sum()) for chunk in np.array_split(reward, 3)
            ]
        if "success" in keys:
            success = scalar_series(stored["success"])
            result["success_final"] = bool(success[-1] > 0)
            result["success_ever"] = bool(np.any(success > 0))
        return result


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            key
            for record in records
            for key in record
            if key != "available_keys" and not key.endswith("_phase")
        }
    )
    output: dict[str, Any] = {}
    for name in names:
        values = [record[name] for record in records if name in record]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        output[name] = {
            "available_episode_count": len(values),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "median": float(np.median(array)),
            "q05": float(np.quantile(array, 0.05)),
            "q95": float(np.quantile(array, 0.95)),
        }
    for name in ("contact_phase", "motion_phase", "reward_phase"):
        values = [record[name] for record in records if name in record]
        if values:
            array = np.asarray(values, dtype=np.float64)
            output[name] = {
                "available_episode_count": len(values),
                "phase_labels": ["early", "middle", "late"],
                "means": array.mean(axis=0).tolist(),
            }
    return output


def main() -> None:
    assert_runtime_contract("evaluation")
    output = ATTEMPT_ROOT / "metrics/posthoc_interpretation.json"
    if output.exists():
        print(json.dumps(read_json(output), sort_keys=True))
        return
    decision_path = ATTEMPT_ROOT / "decision.json"
    decision = read_json(decision_path)
    if not decision.get("confirmation_terminal"):
        raise RuntimeError("post-hoc interpretation requires terminal decision")
    decision_sha256_before = sha256_file(decision_path)
    regimes = {}
    for regime in REGIMES:
        manifest = read_json(raw_manifest_path("target", regime))
        episode_records = []
        for index, record in enumerate(manifest["episodes"]):
            path = REPO_ROOT / record["path"]
            if sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"post-hoc raw hash drift: {path}")
            episode_records.append(summarize_episode(path))
            if (index + 1) % 250 == 0:
                print(
                    f"posthoc {regime} {index + 1}/{len(manifest['episodes'])}",
                    flush=True,
                )
        regimes[regime] = {
            "summary": aggregate(episode_records),
            "episode_count": len(episode_records),
            "all_fields_interpretive_only": True,
        }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "decision_path": relative_to_repo(decision_path),
        "decision_sha256_before_posthoc": decision_sha256_before,
        "terminal_outcome_fixed_before_posthoc": decision["terminal_outcome"],
        "regimes": regimes,
        "fields": ["contact", "motion", "phase", "reward", "success"],
        "used_for_gate_fit_selection_whitening_exclusion_replacement_stopping_or_terminal_mapping": False,
        "decision_sha256_unchanged_after_posthoc": sha256_file(decision_path)
        == decision_sha256_before,
        "target_outcome_episodes": 9_000,
    }
    atomic_json(output, result, exclusive=True)
    append_ledger(
        "post_terminal_interpretation_complete",
        path=relative_to_repo(output),
        sha256=sha256_file(output),
        decision_sha256=decision_sha256_before,
        terminal_outcome=decision["terminal_outcome"],
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
