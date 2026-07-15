#!/usr/bin/env python3
"""Qualify the frozen stage-specific gate on complete consumed fit/selection traces."""

from __future__ import annotations

import json

import numpy as np

from cycle_common import FEATURE_DIM, ROOT, atomic_json, read_json, sha256_file
from verify_gate_freeze import verify as verify_gate_freeze


def calls(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    result = np.ones(len(scores), dtype=np.int64)
    active = np.ones(len(scores), dtype=bool)
    for stage in range(3):
        if not np.isfinite(scores[active, stage]).all():
            raise RuntimeError("nonfinite active qualification score")
        active &= scores[:, stage] > thresholds[stage]
        result += active.astype(np.int64)
    return result


def compiled_scores(features: np.ndarray, compiled: dict[str, np.ndarray]) -> np.ndarray:
    values = np.empty((len(features), 3), dtype=np.float32)
    local = features.astype(np.float32)
    for stage in range(3):
        raw = local[:, stage] @ compiled["a_raw"][stage] + compiled["b_raw"][stage]
        white = local[:, stage] @ compiled["a_white"][stage] + compiled["b_white"][stage]
        values[:, stage] = np.minimum(raw, white)
    return values


def main() -> None:
    gate_seal = verify_gate_freeze()
    freeze = read_json(ROOT / "freeze/gate_freeze.json")
    selection = read_json(ROOT / "development/gate_selection_ledger.json")
    selected = next(
        item for item in selection["candidates"] if item["candidate_id"] == selection["selected_candidate_id"]
    )
    with np.load(ROOT / "freeze/gate_fit.npz", allow_pickle=False) as stored:
        fitted = {name: stored[name].copy() for name in stored.files}
    with np.load(ROOT / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        compiled = {name: stored[name].copy() for name in stored.files}
    if compiled["a_raw"].shape != (3, FEATURE_DIM):
        raise RuntimeError("compiled gate width drift")
    roles = {}
    cpu_score_cache = {}
    for role, expected in (("fit", 240), ("selection", 120)):
        with np.load(ROOT / f"data/{role}_evaluated.npz", allow_pickle=False) as stored:
            features = stored["features"].copy()
            episode_id = stored["episode_id"].copy()
        if len(np.unique(episode_id)) != expected:
            raise RuntimeError(f"{role} episode count drift")
        normalized_scores = np.empty((len(features), 3), dtype=np.float64)
        for stage in range(3):
            z = (features[:, stage].astype(np.float64) - fitted["feature_mean"][stage]) / fitted[
                "feature_std"
            ][stage]
            normalized_scores[:, stage] = np.minimum(
                z @ fitted["wr"][stage], z @ fitted["ww"][stage]
            )
        cpu_scores = compiled_scores(features, compiled)
        reference_calls = calls(normalized_scores, fitted["thresholds"])
        cpu_calls = calls(cpu_scores.astype(np.float64), compiled["thresholds"])
        cpu_score_cache[role] = (features, cpu_scores, cpu_calls)
        roles[role] = {
            "episodes": expected,
            "rows": len(features),
            "reference_compiled_calls_exact": bool(np.array_equal(reference_calls, cpu_calls)),
            "call_histogram": np.bincount(cpu_calls, minlength=5)[1:].tolist(),
            "maximum_reference_compiled_score_difference": float(
                np.max(np.abs(normalized_scores - cpu_scores.astype(np.float64)))
            ),
            "minimum_compiled_threshold_margin": float(
                min(
                    np.min(np.abs(cpu_scores[:, stage].astype(np.float64) - compiled["thresholds"][stage]))
                    for stage in range(3)
                )
            ),
        }
    roles["selection"]["selected_histogram_reproduced"] = (
        roles["selection"]["call_histogram"] == selected["call_histogram"]
    )

    device_checks = []
    try:
        import torch

        for device_name in ("cpu", "mps"):
            if device_name == "mps" and not torch.backends.mps.is_available():
                continue
            device = torch.device(device_name)
            features, cpu_scores, cpu_calls = cpu_score_cache["selection"]
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            a_raw = torch.as_tensor(compiled["a_raw"], device=device)
            b_raw = torch.as_tensor(compiled["b_raw"], device=device)
            a_white = torch.as_tensor(compiled["a_white"], device=device)
            b_white = torch.as_tensor(compiled["b_white"], device=device)
            score_parts = []
            for stage in range(3):
                score_parts.append(
                    torch.minimum(
                        tensor[:, stage] @ a_raw[stage] + b_raw[stage],
                        tensor[:, stage] @ a_white[stage] + b_white[stage],
                    )
                )
            scores_one = torch.stack(score_parts, dim=1)
            scores_two = torch.stack(
                [
                    torch.minimum(
                        tensor[:, stage] @ a_raw[stage] + b_raw[stage],
                        tensor[:, stage] @ a_white[stage] + b_white[stage],
                    )
                    for stage in range(3)
                ],
                dim=1,
            )
            if device_name == "mps":
                torch.mps.synchronize()
            device_scores = scores_one.cpu().numpy()
            device_calls = calls(device_scores.astype(np.float64), compiled["thresholds"])
            device_checks.append(
                {
                    "device": device_name,
                    "repeat_scores_bitwise_exact": bool(torch.equal(scores_one, scores_two)),
                    "calls_match_compiled_cpu": bool(np.array_equal(device_calls, cpu_calls)),
                    "maximum_score_difference_from_compiled_cpu": float(
                        np.max(np.abs(device_scores.astype(np.float64) - cpu_scores.astype(np.float64)))
                    ),
                }
            )
    except RuntimeError:
        raise

    checks = {
        "gate_freeze_seal": gate_seal["passed"],
        "fit_selection_isolation": freeze["fit_selection_isolation_passed"],
        "fit_calls_exact": roles["fit"]["reference_compiled_calls_exact"],
        "selection_calls_exact": roles["selection"]["reference_compiled_calls_exact"],
        "selection_histogram_reproduced": roles["selection"]["selected_histogram_reproduced"],
        "all_devices_repeat_and_calls_exact": bool(device_checks)
        and all(item["repeat_scores_bitwise_exact"] and item["calls_match_compiled_cpu"] for item in device_checks),
        "gate_artifact_hash": freeze["compiled_gate_sha256"]
        == sha256_file(ROOT / "freeze/compiled_gate.npz"),
        "zero_v5": freeze["v5_outcome_episodes"] == 0,
    }
    result = {
        "schema_version": 1,
        "checks": checks,
        "roles": roles,
        "device_checks": device_checks,
        "passed": all(checks.values()),
        "consumed_fit_selection_only": True,
        "prospective_episodes": 0,
        "v5_outcome_episodes": 0,
    }
    if not result["passed"]:
        raise RuntimeError(f"gate-fit qualification failed: {checks}")
    atomic_json(ROOT / "audit/gate_fit_qualification.json", result, exclusive=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
