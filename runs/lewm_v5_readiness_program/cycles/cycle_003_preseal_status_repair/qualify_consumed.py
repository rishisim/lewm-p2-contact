#!/usr/bin/env python3
"""Preseal equivalence on every accessible consumed PlanOracle feature row."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from counted_features import build_counted_causal_features, semantic_feature_names
from cycle_common import (
    ACTION_DIM,
    DISCOVERY_CODE,
    FEATURE_DIM,
    HISTORY_LEN,
    LATENT_DIM,
    NUMERICAL_ATOL,
    NUMERICAL_RTOL,
    ROOT,
    SOURCE_DISCOVERY,
    THRESHOLD,
    atomic_json,
    sequential_calls,
)


def expected_repository_names() -> tuple[str, ...]:
    names = [
        f"history_{time}_{dimension}"
        for time in range(HISTORY_LEN)
        for dimension in range(LATENT_DIM)
    ]
    names += [
        f"action_{time}_{dimension}"
        for time in range(HISTORY_LEN)
        for dimension in range(ACTION_DIM)
    ]
    names += [f"current_{dimension}" for dimension in range(LATENT_DIM)]
    names += [f"last_update_{dimension}" for dimension in range(LATENT_DIM)]
    names += [
        "current_norm",
        "last_update_norm",
        "relative_update_norm",
        "last_history_norm",
        "current_history_distance",
        "update_current_cosine",
        "update_history_distance_cosine",
        "history_change_0_1_norm",
        "history_change_1_2_norm",
        "action_change_0_1_norm",
        "action_change_1_2_norm",
    ]
    return tuple(names)


def calls(scores: np.ndarray) -> np.ndarray:
    return sequential_calls(scores, THRESHOLD)


def device_names(requested: str) -> list[str]:
    result = []
    for name in requested.split(","):
        name = name.strip()
        if not name:
            continue
        if name == "mps" and not torch.backends.mps.is_available():
            continue
        result.append(name)
    if not result:
        raise RuntimeError("no requested qualification device is available")
    return result


def qualify_dataset(
    role: str,
    path: Path,
    device_name: str,
    gate: dict[str, np.ndarray],
    contract: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, Any]:
    # Access only the already-consumed causal feature array.  The target member
    # in this archive is deliberately never subscripted.
    with np.load(path, allow_pickle=False) as stored:
        archive_keys = sorted(stored.files)
        source = stored["features"].astype(np.float32)
    if source.ndim != 3 or source.shape[1:] != (3, FEATURE_DIM):
        raise RuntimeError(f"unexpected consumed feature shape: {source.shape}")
    rows = source.shape[0]
    flat = source.reshape(-1, FEATURE_DIM)
    stages = np.tile(np.arange(3, dtype=np.int64), rows)
    candidate_scores = np.empty(len(flat), dtype=np.float32)
    reference_scores = np.empty(len(flat), dtype=np.float64)
    maximum_feature_delta = 0.0
    feature_allclose = True
    exact_feature_rows = 0
    repeated_scores_exact = True
    device = torch.device(device_name)
    a_raw = torch.as_tensor(gate["a_raw"], device=device)
    a_white = torch.as_tensor(gate["a_white"], device=device)
    b_raw = torch.as_tensor(gate["b_raw"], device=device)
    b_white = torch.as_tensor(gate["b_white"], device=device)

    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        block = flat[start:stop]
        tensor = torch.as_tensor(block, device=device)
        history = tensor[:, :576].reshape(-1, HISTORY_LEN, LATENT_DIM)
        actions = tensor[:, 576:651].reshape(-1, HISTORY_LEN, ACTION_DIM)
        current = tensor[:, 651:843]
        update = tensor[:, 843:1035]
        with torch.inference_mode():
            candidate = build_counted_causal_features(history, actions, current, update)
            depth = torch.zeros((len(candidate), 3), dtype=candidate.dtype, device=device)
            local_stages = torch.as_tensor(stages[start:stop], device=device)
            depth[torch.arange(len(candidate), device=device), local_stages] = 1
            encoded = torch.cat((candidate, depth), dim=1)
            score = torch.minimum(
                encoded @ a_raw + b_raw,
                encoded @ a_white + b_white,
            )
            score_repeat = torch.minimum(
                encoded @ a_raw + b_raw,
                encoded @ a_white + b_white,
            )
        repeated_scores_exact &= bool(torch.equal(score, score_repeat))
        candidate_cpu = candidate.cpu().numpy()
        delta = np.abs(candidate_cpu - block)
        maximum_feature_delta = max(maximum_feature_delta, float(delta.max(initial=0.0)))
        feature_allclose &= bool(
            np.allclose(candidate_cpu, block, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
        )
        exact_feature_rows += int(np.all(candidate_cpu == block, axis=1).sum())
        candidate_scores[start:stop] = score.cpu().numpy()

        encoded_reference = np.concatenate(
            (block.astype(np.float64), np.eye(3, dtype=np.float64)[stages[start:stop]]),
            axis=1,
        )
        normalized = (encoded_reference - contract["mean"]) / contract["std"]
        reference_scores[start:stop] = np.minimum(
            np.einsum("ni,i->n", normalized, weights["raw"], optimize=False),
            np.einsum("ni,i->n", normalized, weights["white"], optimize=False),
        )

    candidate_matrix = candidate_scores.reshape(rows, 3)
    reference_matrix = reference_scores.reshape(rows, 3)
    candidate_calls = calls(candidate_matrix)
    reference_calls = calls(reference_matrix)
    result = {
        "role": role,
        "path": str(path),
        "archive_keys_observed": archive_keys,
        "array_keys_loaded": ["features"],
        "target_array_loaded": False,
        "rows": rows,
        "reached_feature_rows": len(flat),
        "device": device_name,
        "feature_order_preserved": True,
        "feature_allclose_to_repository_execution": feature_allclose,
        "maximum_abs_feature_delta": maximum_feature_delta,
        "bitwise_exact_feature_rows": exact_feature_rows,
        "same_path_compiled_gate_scores_bitwise_exact_on_repeat": repeated_scores_exact,
        "stage_decisions_exact": bool(
            np.array_equal(candidate_matrix > THRESHOLD, reference_matrix > THRESHOLD)
        ),
        "calls_exact": bool(np.array_equal(candidate_calls, reference_calls)),
        "call_histogram_exact": bool(
            np.array_equal(
                np.bincount(candidate_calls, minlength=5),
                np.bincount(reference_calls, minlength=5),
            )
        ),
        "maximum_abs_score_delta": float(
            np.max(np.abs(candidate_matrix.astype(np.float64) - reference_matrix))
        ),
        "minimum_reference_threshold_margin": float(
            np.min(np.abs(reference_matrix - THRESHOLD))
        ),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "feature_order_preserved",
            "feature_allclose_to_repository_execution",
            "same_path_compiled_gate_scores_bitwise_exact_on_repeat",
            "stage_decisions_exact",
            "calls_exact",
            "call_histogram_exact",
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cpu,mps")
    parser.add_argument("--batch-size", type=int, default=2048)
    arguments = parser.parse_args()
    if str(DISCOVERY_CODE) not in sys.path:
        sys.path.insert(0, str(DISCOVERY_CODE))
    import models

    semantic = semantic_feature_names(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, history_len=HISTORY_LEN
    )
    repository = models.causal_feature_names(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, history_len=HISTORY_LEN
    )
    if repository != expected_repository_names() or len(semantic) != len(repository):
        raise RuntimeError("executable semantic feature order does not match repository order")

    with np.load(ROOT / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        gate = {name: stored[name].copy() for name in stored.files}
    with np.load(SOURCE_DISCOVERY / "freeze/gate_contract.npz", allow_pickle=False) as stored:
        contract = {
            "mean": stored["feature_mean"].astype(np.float64),
            "std": stored["feature_std"].astype(np.float64),
            "source_feature_order": stored["feature_order"].copy(),
        }
    with np.load(SOURCE_DISCOVERY / "freeze/gate_weights.npz", allow_pickle=False) as stored:
        weights = {
            "raw": stored["wr"].astype(np.float64),
            "white": stored["ww"].astype(np.float64),
        }
    expected_positional = np.asarray(
        [f"causal_{index}" for index in range(FEATURE_DIM)] + ["stage_1", "stage_2", "stage_3"]
    )
    if not np.array_equal(contract["source_feature_order"], expected_positional):
        raise RuntimeError("source gate positional order drift")

    paths = {
        role: SOURCE_DISCOVERY / f"data/{role}_evaluated.npz"
        for role in ("smoke", "fit", "selection", "prospective")
    }
    results = []
    for device_name in device_names(arguments.devices):
        for role, path in paths.items():
            result = qualify_dataset(
                role,
                path,
                device_name,
                gate,
                contract,
                weights,
                arguments.batch_size,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "device": device_name,
                        "role": role,
                        "rows": result["rows"],
                        "passed": result["passed"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    mapping = [
        {
            "index": index,
            "semantic_name": name,
            "repository_name": repository[index],
            "frozen_gate_positional_label": f"causal_{index}",
        }
        for index, name in enumerate(semantic)
    ]
    names_digest = hashlib.sha256(
        ("\n".join(semantic) + "\n").encode("utf-8")
    ).hexdigest()
    atomic_json(
        ROOT / "semantic_feature_names.json",
        {
            "schema_version": 1,
            "generated_by": "counted_features.semantic_feature_names",
            "feature_count": len(mapping),
            "semantic_names_sha256": names_digest,
            "position_mapping": mapping,
        },
        exclusive=True,
    )
    audit = {
        "schema_version": 1,
        "consumed_only": True,
        "devices": device_names(arguments.devices),
        "datasets": results,
        "total_source_rows": sum(item["rows"] for item in results if item["device"] == device_names(arguments.devices)[0]),
        "total_reached_feature_rows_per_device": sum(
            item["reached_feature_rows"] for item in results if item["device"] == device_names(arguments.devices)[0]
        ),
        "semantic_feature_names_sha256": names_digest,
        "identical_feature_order": True,
        "all_exact_gate_decisions_and_calls": all(
            item["stage_decisions_exact"] and item["calls_exact"] for item in results
        ),
        "passed": all(item["passed"] for item in results),
        "v3_test_targets_opened": False,
        "released_hdf5_opened": False,
        "v5_outcome_episodes": 0,
    }
    if not audit["passed"]:
        raise RuntimeError("consumed feature/gate qualification failed")
    atomic_json(ROOT / "audit/consumed_feature_gate_qualification.json", audit, exclusive=True)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
