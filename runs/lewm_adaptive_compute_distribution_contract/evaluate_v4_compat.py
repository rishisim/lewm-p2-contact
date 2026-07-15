#!/usr/bin/env python3
"""Schema-only compatibility wrapper for the preserved V4 encoded manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common
import evaluate


COMPAT = common.STUDY_ROOT / "audit/v4_manifest_schema_compatibility.json"
SEAL = common.STUDY_ROOT / "audit/v4_manifest_schema_compatibility_seal.json"
V4_CACHE = common.V4_ROOT / "data/confirmation_encoded.npz"
V4_MANIFEST = common.V4_ROOT / "data/confirmation_encoded_manifest.json"


def seal() -> dict:
    common.assert_pre_generation_seal()
    output = common.STUDY_ROOT / "data/v4_markov_evaluation.npz"
    output_manifest = common.STUDY_ROOT / "data/v4_markov_evaluation_manifest.json"
    if output.exists() or output_manifest.exists() or COMPAT.exists() or SEAL.exists():
        raise RuntimeError("V4 compatibility artifacts or outputs already exist")
    manifest = common.study_json(V4_MANIFEST)
    if "encoded_sha256" in manifest or "cache_sha256" not in manifest:
        raise RuntimeError("V4 manifest no longer has the diagnosed schema")
    if common.sha256_file(V4_CACHE) != manifest["cache_sha256"]:
        raise RuntimeError("V4 encoded cache hash does not match its preserved manifest")
    payload = {
        "schema_version": 1,
        "status": "frozen_before_v4_diagnostic_recomputation",
        "diagnosis": (
            "the preserved V4 manifest names its encoded file hash cache_sha256, "
            "while the new evaluator expected encoded_sha256"
        ),
        "compatibility_action": (
            "verify cache_sha256, load the same NPZ, and rename episode_slot to "
            "episode_id in memory; no numerical array or outcome is changed"
        ),
        "v4_manifest_sha256": common.sha256_file(V4_MANIFEST),
        "v4_cache_sha256": manifest["cache_sha256"],
        "v4_used_for_thresholds_training_or_selection": False,
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    common.write_study_json(COMPAT, payload, exclusive=True)
    sealed = {
        "schema_version": 1,
        "status": "frozen_before_v4_diagnostic_recomputation",
        "sealed_files": {
            str(path.relative_to(common.REPO_ROOT)): common.sha256_file(path)
            for path in (Path(__file__).resolve(), COMPAT, V4_MANIFEST)
        },
        "v4_cache_sha256": manifest["cache_sha256"],
        "v3_test_targets_opened": False,
    }
    common.write_study_json(SEAL, sealed, exclusive=True)
    return sealed


def assert_seal() -> dict:
    payload = common.study_json(SEAL)
    for relative, expected in payload["sealed_files"].items():
        if common.sha256_file(common.REPO_ROOT / relative) != expected:
            raise RuntimeError(f"V4 compatibility seal drift: {relative}")
    if common.sha256_file(V4_CACHE) != payload["v4_cache_sha256"]:
        raise RuntimeError("V4 cache drift after compatibility seal")
    return payload


def load_v4(role: str):
    if role != "v4_markov":
        raise ValueError("compatibility loader is restricted to V4 diagnostic role")
    assert_seal()
    manifest = common.study_json(V4_MANIFEST)
    if common.sha256_file(V4_CACHE) != manifest["cache_sha256"]:
        raise RuntimeError("V4 encoded input drift")
    with np.load(V4_CACHE, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    arrays["episode_id"] = arrays.pop("episode_slot").astype(np.int64)
    return arrays, {
        **manifest,
        "role": "v4_markov_diagnostic_only",
        "schema_compatibility_sha256": common.sha256_file(COMPAT),
        "v3_test_targets_opened": False,
    }


def run(device: str) -> dict:
    assert_seal()
    evaluate._load_eval_inputs = load_v4
    return evaluate.evaluate_role("v4_markov", device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "run"))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    result = seal() if args.command == "seal" else run(args.device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
