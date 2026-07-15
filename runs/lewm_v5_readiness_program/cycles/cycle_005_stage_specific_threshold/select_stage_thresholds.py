#!/usr/bin/env python3
"""Evaluate the frozen 25-candidate grid on selection episodes only."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
DISCOVERY = REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_discovery"
PROTOCOL_PATH = ROOT / "development/stage_specific_threshold_protocol.json"
SPLIT_MANIFEST_PATH = ROOT / "development/split_manifest.json"
SELECTION_PATH = ROOT / "development/selection_consumed.npz"
LEDGER_PATH = ROOT / "development/candidate_selection_ledger.json"
LOCK_PATH = ROOT / "development/selected_stage_thresholds.json"

BASE_THRESHOLD = 0.2719204773217412
BASE_FLOPS = 70_529_190
V1_FLOPS = 669_184
ADAPTER_FLOPS = 264_960
GATE_FLOPS = 7_997
SEEDED_MIXTURE_SEED = 2_006_777_731
HISTOGRAM_SEED = 2_006_888_841


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
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


def strongest_mixture(losses: np.ndarray, mean_calls: float) -> dict[str, Any]:
    best = None
    for lower in range(1, 5):
        for upper in range(lower, 5):
            if not lower <= mean_calls <= upper:
                continue
            weight = 0.0 if lower == upper else (mean_calls - lower) / (upper - lower)
            values = (1.0 - weight) * losses[:, lower - 1] + weight * losses[:, upper - 1]
            candidate = (float(values.mean()), values, lower, upper, float(weight))
            if best is None or candidate[0] < best[0] or (
                candidate[0] == best[0] and candidate[2:4] < best[2:4]
            ):
                best = candidate
    if best is None:
        raise RuntimeError("no supported analytic mixture")
    return {
        "mean_loss": best[0],
        "loss": best[1],
        "depth_lower": best[2],
        "depth_upper": best[3],
        "weight_upper": best[4],
    }


def episode_mean(values: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    return np.asarray([values[episode_id == item].mean() for item in np.unique(episode_id)])


def sequential_calls(scores: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    calls = np.ones(len(scores), dtype=np.int64)
    active = np.ones(len(scores), dtype=bool)
    reached = []
    for stage in range(3):
        reached.append(active.copy())
        if not np.isfinite(scores[active, stage]).all():
            raise RuntimeError("candidate requires an unobserved score")
        active &= scores[:, stage] > thresholds[stage]
        calls += active.astype(np.int64)
    return calls, reached


def evaluate(
    arrays: dict[str, np.ndarray], whitening: np.ndarray, multipliers: tuple[float, float]
) -> dict[str, Any]:
    target = arrays["target"].astype(np.float64)
    exits = arrays["dense_exits"].astype(np.float64)
    scores = arrays["scores"].astype(np.float64)
    old_calls = arrays["calls"].astype(np.int64)
    episode_id = arrays["episode_id"].astype(np.int64)
    thresholds = np.asarray(
        [BASE_THRESHOLD * multipliers[0], BASE_THRESHOLD * multipliers[1], BASE_THRESHOLD],
        dtype=np.float64,
    )
    calls, reached = sequential_calls(scores, thresholds)
    counterfactual_complete = bool(np.all(calls <= old_calls))
    if not counterfactual_complete:
        raise RuntimeError("upward threshold candidate expanded an observed path")

    row_index = np.arange(len(calls))
    difference = exits - target[:, None, :]
    raw_losses = np.square(difference).mean(axis=2)
    white_difference = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    white_losses = np.square(white_difference).mean(axis=2)
    adaptive_raw = raw_losses[row_index, calls - 1]
    adaptive_white = white_losses[row_index, calls - 1]

    gate_evaluations = int(np.minimum(calls, 3).sum())
    refiner_calls = int(calls.sum())
    common_flops = len(calls) * (BASE_FLOPS + V1_FLOPS)
    adaptive_total_flops = int(
        common_flops
        + (refiner_calls - len(calls)) * ADAPTER_FLOPS
        + gate_evaluations * GATE_FLOPS
    )
    analytic_total_calls = refiner_calls + gate_evaluations * GATE_FLOPS / ADAPTER_FLOPS
    analytic_mean_calls = analytic_total_calls / len(calls)
    raw_analytic = strongest_mixture(raw_losses, analytic_mean_calls)
    white_analytic = strongest_mixture(white_losses, analytic_mean_calls)

    integer_target_calls = math.ceil(analytic_total_calls)
    integer_mean_calls = integer_target_calls / len(calls)
    seeded_shape = strongest_mixture(raw_losses, integer_mean_calls)
    lower = int(seeded_shape["depth_lower"])
    upper = int(seeded_shape["depth_upper"])
    number_upper = (
        0
        if lower == upper
        else math.ceil((integer_target_calls - lower * len(calls)) / (upper - lower))
    )
    seeded_calls = np.full(len(calls), lower, dtype=np.int64)
    seeded_rng = np.random.default_rng(SEEDED_MIXTURE_SEED)
    seeded_calls[seeded_rng.permutation(len(calls))[:number_upper]] = upper
    seeded_raw = raw_losses[row_index, seeded_calls - 1]
    seeded_total_flops = int(
        common_flops + (int(seeded_calls.sum()) - len(calls)) * ADAPTER_FLOPS
    )

    histogram_calls = np.empty_like(calls)
    histogram_rng = np.random.default_rng(HISTOGRAM_SEED)
    histograms_exact = True
    for episode in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == episode)
        histogram_calls[indices] = calls[indices][histogram_rng.permutation(len(indices))]
        histograms_exact &= bool(
            np.array_equal(
                np.bincount(calls[indices], minlength=5),
                np.bincount(histogram_calls[indices], minlength=5),
            )
        )
    histogram_raw = raw_losses[row_index, histogram_calls - 1]
    histogram_white = white_losses[row_index, histogram_calls - 1]

    contrasts = {
        "raw_vs_analytic": float((raw_analytic["loss"] - adaptive_raw).mean()),
        "raw_vs_seeded": float((seeded_raw - adaptive_raw).mean()),
        "raw_vs_fixed_d1": float((raw_losses[:, 0] - adaptive_raw).mean()),
        "raw_vs_within_episode_histogram": float((histogram_raw - adaptive_raw).mean()),
        "native_whitened_vs_analytic": float(
            (white_analytic["loss"] - adaptive_white).mean()
        ),
        "native_whitened_vs_within_episode_histogram": float(
            (histogram_white - adaptive_white).mean()
        ),
    }

    stagewise = []
    for stage in range(3):
        raw_gain = raw_losses[:, stage] - raw_losses[:, stage + 1]
        white_gain = white_losses[:, stage] - white_losses[:, stage + 1]
        combined = 0.5 * (
            raw_gain / (raw_gain.std() + 1e-12)
            + white_gain / (white_gain.std() + 1e-12)
        )
        rho = float(spearmanr(scores[reached[stage], stage], combined[reached[stage]]).statistic)
        stagewise.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached[stage].sum()),
                "rho": rho,
                "positive": bool(np.isfinite(rho) and rho > 0),
            }
        )

    histogram = np.bincount(calls, minlength=5)[1:].tolist()
    eligibility = {
        "counterfactual_complete": counterfactual_complete,
        "stage_2_reached_at_least_400": stagewise[1]["reached_rows"] >= 400,
        "stage_3_reached_at_least_120": stagewise[2]["reached_rows"] >= 120,
        "all_four_depths_at_least_20": all(value >= 20 for value in histogram),
        "all_six_point_contrasts_positive": all(value > 0 for value in contrasts.values()),
        "all_three_stage_rhos_positive": all(item["positive"] for item in stagewise),
        "seeded_comparator_weakly_more_compute": seeded_total_flops >= adaptive_total_flops,
        "within_episode_histograms_exact": histograms_exact,
    }
    return {
        "candidate_id": f"d1_{int(100*multipliers[0]):03d}_d2_{int(100*multipliers[1]):03d}",
        "multipliers": [multipliers[0], multipliers[1], 1.0],
        "thresholds": thresholds.tolist(),
        "eligible": all(eligibility.values()),
        "eligibility": eligibility,
        "point_contrasts": contrasts,
        "stagewise_rank": stagewise,
        "minimum_stagewise_rho": float(min(item["rho"] for item in stagewise)),
        "call_histogram": histogram,
        "mean_calls": float(calls.mean()),
        "gate_evaluations": gate_evaluations,
        "adaptive_total_flops": adaptive_total_flops,
        "analytic_equivalent_mean_calls": float(analytic_mean_calls),
        "seeded_total_flops": seeded_total_flops,
        "seeded_minus_adaptive_flops": seeded_total_flops - adaptive_total_flops,
        "episode_point_contrasts_computed": {
            name: len(episode_mean(values, episode_id))
            for name, values in {
                "raw_vs_analytic": raw_analytic["loss"] - adaptive_raw,
                "raw_vs_seeded": seeded_raw - adaptive_raw,
                "raw_vs_fixed_d1": raw_losses[:, 0] - adaptive_raw,
                "raw_vs_within_episode_histogram": histogram_raw - adaptive_raw,
                "native_whitened_vs_analytic": white_analytic["loss"] - adaptive_white,
                "native_whitened_vs_within_episode_histogram": histogram_white
                - adaptive_white,
            }.items()
        },
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    split = json.loads(SPLIT_MANIFEST_PATH.read_text())
    if protocol["status"] != "frozen_before_any_candidate_specific_counterfactual":
        raise RuntimeError("protocol not frozen")
    if split["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("protocol/split hash mismatch")
    if split["roles"]["selection"]["sha256"] != sha256(SELECTION_PATH):
        raise RuntimeError("selection split hash mismatch")
    with np.load(SELECTION_PATH, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(DISCOVERY / "freeze/gate_contract.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    if len(np.unique(arrays["episode_id"])) != 150 or np.any(arrays["episode_id"] % 2 != 0):
        raise RuntimeError("selection role drift")

    multipliers = [float(value) for value in protocol["single_change_family"]["decision_1_multipliers"]]
    candidates = [
        evaluate(arrays, whitening, (decision_1, decision_2))
        for decision_1 in multipliers
        for decision_2 in multipliers
    ]
    if len(candidates) != 25 or len({item["candidate_id"] for item in candidates}) != 25:
        raise RuntimeError("candidate grid drift")
    eligible = [item for item in candidates if item["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item["minimum_stagewise_rho"],
            item["multipliers"][0] + item["multipliers"][1],
            item["multipliers"][0],
            item["multipliers"][1],
        ),
    )
    selected = ranked[0] if ranked else None
    ledger = {
        "schema_version": 1,
        "status": "selection_complete_before_validation_candidate_metrics",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "split_manifest_sha256": sha256(SPLIT_MANIFEST_PATH),
        "selection_input_sha256": sha256(SELECTION_PATH),
        "validation_input_path_not_opened": True,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "candidates": candidates,
        "fixed_ranking_applied": True,
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "v5_outcome_episodes": 0,
    }
    atomic_json(LEDGER_PATH, ledger)
    lock = {
        "schema_version": 1,
        "status": "selected_and_locked_before_validation_candidate_metrics",
        "selection_ledger_sha256": sha256(LEDGER_PATH),
        "selection_input_sha256": sha256(SELECTION_PATH),
        "validation_input_sha256_declared_in_preselection_split": split["roles"]["validation"]["sha256"],
        "selected": selected,
        "eligible_candidate_exists": selected is not None,
        "validation_candidate_metrics_accessed": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(LOCK_PATH, lock)
    print(json.dumps(lock, sort_keys=True))


if __name__ == "__main__":
    main()
