#!/usr/bin/env python3
"""Diagnose the consumed cycle-004 miss without evaluating new candidates."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
PROGRAM_ROOT = ROOT.parents[1]
REPO_ROOT = ROOT.parents[3]
SOURCE = PROGRAM_ROOT / "cycles/cycle_004_threshold_recalibration"
DISCOVERY = REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_discovery"
OUTPUT = ROOT / "development/consumed_cycle_004_diagnosis.json"
THRESHOLD = 0.2719204773217412
ADAPTER_FLOPS = 264_960
GATE_FLOPS = 7_997


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise RuntimeError("nonfinite Spearman statistic")
    return value


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    names = ("minimum", "q10", "q25", "median", "q75", "q90", "q95", "q99", "maximum")
    return dict(zip(names, (float(x) for x in np.quantile(values, levels))))


def main() -> None:
    execution_path = SOURCE / "data/prospective_execution.npz"
    analysis_path = SOURCE / "analysis_result.json"
    episode_metrics_path = SOURCE / "metrics/prospective_episode_metrics.npz"
    whitening_path = DISCOVERY / "freeze/gate_contract.npz"
    with np.load(execution_path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(whitening_path, allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    with np.load(episode_metrics_path, allow_pickle=False) as stored:
        episode_metrics = {name: stored[name].astype(np.float64) for name in stored.files}
    analysis = json.loads(analysis_path.read_text())

    target = arrays["target"].astype(np.float64)
    exits = arrays["dense_exits"].astype(np.float64)
    scores = arrays["scores"].astype(np.float64)
    calls = arrays["calls"].astype(np.int64)
    episode_id = arrays["episode_id"].astype(np.int64)
    if len(target) != 11_400 or not np.array_equal(np.unique(episode_id), np.arange(300)):
        raise RuntimeError("cycle-004 cohort shape drift")
    difference = exits - target[:, None, :]
    raw_losses = np.square(difference).mean(axis=2)
    white_difference = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    white_losses = np.square(white_difference).mean(axis=2)

    stage_records = []
    for stage in range(3):
        reached = np.isfinite(scores[:, stage])
        raw_gain = raw_losses[:, stage] - raw_losses[:, stage + 1]
        white_gain = white_losses[:, stage] - white_losses[:, stage + 1]
        combined = 0.5 * (
            raw_gain / (raw_gain.std() + 1e-12)
            + white_gain / (white_gain.std() + 1e-12)
        )
        local_score = scores[reached, stage]
        local_gain = combined[reached]
        order = np.argsort(local_score, kind="stable")
        bins = []
        for rank, indices in enumerate(np.array_split(order, 5), start=1):
            bins.append(
                {
                    "score_quintile": rank,
                    "rows": int(len(indices)),
                    "mean_score": float(local_score[indices].mean()),
                    "mean_combined_gain": float(local_gain[indices].mean()),
                    "positive_combined_gain_fraction": float((local_gain[indices] > 0).mean()),
                }
            )
        margins = np.abs(local_score - THRESHOLD)
        stage_records.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "score_gain_spearman": finite_spearman(local_score, local_gain),
                "score_quantiles": quantiles(local_score),
                "margin_quantiles": quantiles(margins),
                "fractions_within_threshold_margin": {
                    "1e-6": float((margins <= 1e-6).mean()),
                    "1e-4": float((margins <= 1e-4).mean()),
                    "1e-3": float((margins <= 1e-3).mean()),
                    "1e-2": float((margins <= 1e-2).mean()),
                },
                "raw_gain_mean": float(raw_gain[reached].mean()),
                "raw_gain_median": float(np.median(raw_gain[reached])),
                "raw_gain_positive_fraction": float((raw_gain[reached] > 0).mean()),
                "native_whitened_gain_mean": float(white_gain[reached].mean()),
                "native_whitened_gain_median": float(np.median(white_gain[reached])),
                "native_whitened_gain_positive_fraction": float((white_gain[reached] > 0).mean()),
                "combined_gain_mean": float(local_gain.mean()),
                "combined_gain_median": float(np.median(local_gain)),
                "combined_gain_positive_fraction": float((local_gain > 0).mean()),
                "calibration_quintiles": bins,
            }
        )

    effect_names = (
        "raw_vs_analytic",
        "raw_vs_seeded",
        "raw_vs_fixed_d1",
        "raw_vs_within_episode_histogram",
        "native_whitened_vs_analytic",
        "native_whitened_vs_within_episode_histogram",
    )
    heterogeneity: dict[str, Any] = {}
    for name in effect_names:
        values = episode_metrics[name]
        blocks = [float(values[start : start + 50].mean()) for start in range(0, 300, 50)]
        heterogeneity[name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "episode_index_spearman": finite_spearman(np.arange(300), values),
            "six_contiguous_50_episode_block_means": blocks,
            "block_mean_range": [float(min(blocks)), float(max(blocks))],
        }

    checks = {
        "all_six_effect_lower_bounds_positive": all(
            analysis["criteria"][name]["lower"] > 0 for name in effect_names
        ),
        "both_simultaneous_lower_bounds_positive": all(
            item["lower"] > 0 for item in analysis["simultaneous_co_primary"].values()
        ),
        "only_failed_integrity_field_is_stagewise_positive": sorted(
            name for name, passed in analysis["integrity"].items() if not passed
        )
        == ["stagewise_positive"],
        "cycle_004_terminal_outcome_preserved": analysis["proposed_terminal_outcome"]
        == "cycle_execution_invalid",
        "zero_v5_outcomes": analysis["v5_outcome_episodes"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"unexpected cycle-004 diagnosis inputs: {checks}")

    output = {
        "schema_version": 1,
        "purpose": "consumed_cycle_004_diagnosis_before_any_cycle_005_candidate_counterfactual",
        "input_hashes": {
            "cycle_004_execution": sha256(execution_path),
            "cycle_004_analysis": sha256(analysis_path),
            "cycle_004_episode_metrics": sha256(episode_metrics_path),
            "frozen_whitening_contract": sha256(whitening_path),
        },
        "checks": checks,
        "heterogeneity_and_dgp_stability": heterogeneity,
        "calibration_gate_margins_and_stagewise_gains": stage_records,
        "compute_price": {
            "gate_flops_per_reached_decision": GATE_FLOPS,
            "adapter_flops_per_additional_call": ADAPTER_FLOPS,
            "gate_equivalent_adapter_calls_per_evaluation": GATE_FLOPS / ADAPTER_FLOPS,
            "cycle_004_gate_evaluations": int(np.minimum(calls, 3).sum()),
            "cycle_004_mean_calls": float(calls.mean()),
            "cycle_004_total_flops": int(analysis["compute"]["adaptive_total_flops"]),
        },
        "diagnosis": {
            "scientific_effect_status": "all frozen loss-contrast criteria passed",
            "failure_localization": "shared-threshold deeper-stage score/gain rank calibration",
            "stage_2_and_3_ranks_are_near_zero_not_large_adverse_effects": True,
            "no_numerical_semantic_dgp_isolation_or_hash_failure": True,
            "smallest_evidence_backed_change": "upward-only stage-specific thresholds at decisions one and two; retain decision-three threshold",
            "counterfactual_completeness_reason": "every proposed continuation set is a subset of cycle-004 continuations, so every required later-stage score and dense exit already exists",
            "unchanged": [
                "base model",
                "stagewise refiner",
                "dual affine gate weights",
                "whitening",
                "counted causal feature graph",
                "DGP",
                "statistics and all pass criteria",
                "sparse execution and numerical contract",
            ],
        },
        "candidate_counterfactuals_evaluated_by_this_script": 0,
        "cycle_004_consumed": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(OUTPUT, output)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
