#!/usr/bin/env python3
"""Finite preregistered threshold-only selection with one-shot validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
CYCLE_001 = REPO_ROOT / "runs/lewm_v5_readiness_program/cycles/cycle_001_audit_repair"
CYCLE_003 = REPO_ROOT / "runs/lewm_v5_readiness_program/cycles/cycle_003_preseal_status_repair"
SOURCE_DISCOVERY = REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_discovery"
OLD_THRESHOLD = 0.18128031821449414
CANDIDATE_MULTIPLIERS = (1.0, 1.25, 1.5, 1.75, 2.0)
CANDIDATES = tuple(OLD_THRESHOLD * value for value in CANDIDATE_MULTIPLIERS)
BOOTSTRAP_REPLICATES = 20_000
SELECTION_BOOTSTRAP_SEED = 2_094_001_001
VALIDATION_BOOTSTRAP_SEED = 2_094_003_001
SELECTION_HISTOGRAM_SEED = 2_094_001_002
VALIDATION_HISTOGRAM_SEED = 2_094_003_002
SELECTION_SEEDED_MIXTURE_SEED = 2_094_001_003
VALIDATION_SEEDED_MIXTURE_SEED = 2_094_003_003
BASE_FLOPS = 70_529_190
V1_FLOPS = 669_184
ADAPTER_FLOPS = 264_960
GATE_FLOPS = 7_997


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"immutable output already exists: {path}")
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise RuntimeError(f"immutable output appeared concurrently: {path}")
        os.replace(raw_temporary, path)
    except BaseException:
        try:
            os.unlink(raw_temporary)
        except FileNotFoundError:
            pass
        raise


def episode_mean(values: np.ndarray, episode: np.ndarray) -> np.ndarray:
    result = np.empty(300, dtype=np.float64)
    for index in range(300):
        selected = values[episode == index]
        if len(selected) != 38:
            raise RuntimeError("cohort is not exactly 38 rows per episode")
        result[index] = selected.mean(dtype=np.float64)
    return result


def calls_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    calls = np.ones(len(scores), dtype=np.int64)
    active = np.ones(len(scores), dtype=bool)
    for stage in range(3):
        if not np.isfinite(scores[active, stage]).all():
            raise RuntimeError("candidate requires an unobserved deeper gate score")
        active &= scores[:, stage] > threshold
        calls += active.astype(np.int64)
    return calls


def strongest_mixture(loss: np.ndarray, mean_calls: float) -> dict[str, Any]:
    candidates = []
    for lower in range(1, 5):
        for upper in range(lower, 5):
            if not lower <= mean_calls <= upper:
                continue
            weight = 0.0 if lower == upper else (mean_calls - lower) / (upper - lower)
            values = loss[:, lower - 1] + weight * (loss[:, upper - 1] - loss[:, lower - 1])
            candidates.append((float(values.mean()), lower, upper, float(weight), values))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    mean, lower, upper, weight, values = candidates[0]
    return {
        "mean_loss": mean,
        "depth_lower": lower,
        "depth_upper": upper,
        "weight_upper": weight,
        "values": values,
    }


def load_consumed(cycle: Path, whitening: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(cycle / "data/prospective_execution.npz", allow_pickle=False) as stored:
        target = stored["target"].astype(np.float64)
        dense = stored["dense_exits"].astype(np.float64)
        scores = stored["scores"].astype(np.float64)
        episode = stored["episode_id"].astype(np.int64)
    if dense.shape != (11_400, 4, 192) or not np.array_equal(np.unique(episode), np.arange(300)):
        raise RuntimeError("consumed cohort shape or episode set is invalid")
    error = dense - target[:, None, :]
    raw_loss = np.mean(error**2, axis=2, dtype=np.float64)
    white_error = np.einsum("nkd,df->nkf", error, whitening, optimize=False)
    white_loss = np.mean(white_error**2, axis=2, dtype=np.float64)
    return {
        "dense": dense,
        "scores": scores,
        "episode": episode,
        "raw_loss": raw_loss,
        "white_loss": white_loss,
    }


def evaluate(
    arrays: dict[str, np.ndarray],
    threshold: float,
    sampling: np.ndarray,
    histogram_seed: int,
    seeded_mixture_seed: int,
) -> dict[str, Any]:
    raw_loss = arrays["raw_loss"]
    white_loss = arrays["white_loss"]
    scores = arrays["scores"]
    episode = arrays["episode"]
    calls = calls_from_scores(scores, threshold)
    row = np.arange(len(calls))
    adaptive_raw = raw_loss[row, calls - 1]
    adaptive_white = white_loss[row, calls - 1]
    gate_evaluations = int(np.minimum(calls, 3).sum())
    solver_calls = int(calls.sum())
    common_flops = len(calls) * (BASE_FLOPS + V1_FLOPS)
    total_flops = int(
        common_flops
        + (solver_calls - len(calls)) * ADAPTER_FLOPS
        + gate_evaluations * GATE_FLOPS
    )
    fractional_calls = solver_calls + gate_evaluations * GATE_FLOPS / ADAPTER_FLOPS
    mean_calls = fractional_calls / len(calls)
    raw_analytic = strongest_mixture(raw_loss, mean_calls)
    white_analytic = strongest_mixture(white_loss, mean_calls)

    integer_calls = math.ceil(fractional_calls)
    seeded_shape = strongest_mixture(raw_loss, integer_calls / len(calls))
    lower = seeded_shape["depth_lower"]
    upper = seeded_shape["depth_upper"]
    number_upper = (
        0
        if lower == upper
        else math.ceil((integer_calls - lower * len(calls)) / (upper - lower))
    )
    seeded_calls = np.full(len(calls), lower, dtype=np.int64)
    seeded_order = np.random.default_rng(seeded_mixture_seed).permutation(len(calls))
    seeded_calls[seeded_order[:number_upper]] = upper
    seeded_raw = raw_loss[row, seeded_calls - 1]
    seeded_flops = int(
        common_flops + (int(seeded_calls.sum()) - len(calls)) * ADAPTER_FLOPS
    )

    randomized = np.empty_like(calls)
    histogram_rng = np.random.default_rng(histogram_seed)
    histograms_preserved = True
    for index in range(300):
        locations = np.flatnonzero(episode == index)
        randomized[locations] = calls[locations][histogram_rng.permutation(38)]
        histograms_preserved &= bool(
            np.array_equal(np.sort(randomized[locations]), np.sort(calls[locations]))
        )
    histogram_raw = raw_loss[row, randomized - 1]
    histogram_white = white_loss[row, randomized - 1]
    differences = {
        "raw_vs_analytic": episode_mean(raw_analytic["values"] - adaptive_raw, episode),
        "raw_vs_seeded": episode_mean(seeded_raw - adaptive_raw, episode),
        "raw_vs_fixed_d1": episode_mean(raw_loss[:, 0] - adaptive_raw, episode),
        "raw_vs_within_episode_histogram": episode_mean(
            histogram_raw - adaptive_raw, episode
        ),
        "native_whitened_vs_analytic": episode_mean(
            white_analytic["values"] - adaptive_white, episode
        ),
        "native_whitened_vs_within_episode_histogram": episode_mean(
            histogram_white - adaptive_white, episode
        ),
    }
    bootstrap = {
        name: values[sampling].mean(axis=1) for name, values in differences.items()
    }
    intervals = {
        name: {
            "estimate": float(values.mean()),
            "lower": float(np.quantile(bootstrap[name], 0.025)),
            "upper": float(np.quantile(bootstrap[name], 0.975)),
            "bootstrap_sd": float(bootstrap[name].std(ddof=1)),
            "standardized_margin": float(
                values.mean() / (bootstrap[name].std(ddof=1) + 1e-30)
            ),
        }
        for name, values in differences.items()
    }
    co_primary = ("raw_vs_analytic", "native_whitened_vs_analytic")
    simultaneous = {
        name: {
            "estimate": intervals[name]["estimate"],
            "lower": float(np.quantile(bootstrap[name], 0.025)),
        }
        for name in co_primary
    }
    stages = []
    for stage in range(3):
        reached = calls >= stage + 1
        raw_gain = raw_loss[:, stage] - raw_loss[:, stage + 1]
        white_gain = white_loss[:, stage] - white_loss[:, stage + 1]
        combined = 0.5 * (
            raw_gain / (raw_gain.std() + 1e-12)
            + white_gain / (white_gain.std() + 1e-12)
        )
        rho = float(spearmanr(scores[reached, stage], combined[reached]).statistic)
        stages.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "rho": rho,
                "positive_sign": bool(np.isfinite(rho) and rho > 0),
            }
        )
    all_interval_lowers_positive = all(value["lower"] > 0 for value in intervals.values())
    all_simultaneous_lowers_positive = all(
        value["lower"] > 0 for value in simultaneous.values()
    )
    all_stage_signs_positive = all(value["positive_sign"] for value in stages)
    eligible = (
        all_interval_lowers_positive
        and all_simultaneous_lowers_positive
        and all_stage_signs_positive
        and histograms_preserved
        and seeded_flops >= total_flops
    )
    block_stability = []
    episode_calls = episode_mean(calls.astype(np.float64), episode)
    for start in range(0, 300, 50):
        block_stability.append(
            {
                "episodes": [start, start + 49],
                "raw_vs_analytic": float(
                    differences["raw_vs_analytic"][start : start + 50].mean()
                ),
                "native_whitened_vs_analytic": float(
                    differences["native_whitened_vs_analytic"][start : start + 50].mean()
                ),
                "mean_calls": float(episode_calls[start : start + 50].mean()),
            }
        )
    return {
        "threshold": float(threshold),
        "call_histogram": np.bincount(calls, minlength=5)[1:].tolist(),
        "mean_actual_calls": float(calls.mean()),
        "gate_evaluations": gate_evaluations,
        "gate_equivalent_calls_per_row": float(
            gate_evaluations * GATE_FLOPS / ADAPTER_FLOPS / len(calls)
        ),
        "exact_total_compute_mean_calls": float(mean_calls),
        "total_flops": total_flops,
        "seeded_total_flops": seeded_flops,
        "raw_analytic_mapping": {
            key: raw_analytic[key] for key in ("depth_lower", "depth_upper", "weight_upper")
        },
        "native_whitened_analytic_mapping": {
            key: white_analytic[key]
            for key in ("depth_lower", "depth_upper", "weight_upper")
        },
        "criteria": intervals,
        "simultaneous": simultaneous,
        "stagewise": stages,
        "minimum_standardized_margin": float(
            min(value["standardized_margin"] for value in intervals.values())
        ),
        "all_interval_lowers_positive": all_interval_lowers_positive,
        "all_simultaneous_lowers_positive": all_simultaneous_lowers_positive,
        "all_stage_signs_positive": all_stage_signs_positive,
        "seeded_weakly_more_compute": seeded_flops >= total_flops,
        "histograms_preserved": histograms_preserved,
        "eligible": eligible,
        "episode_differences": {name: values.tolist() for name, values in differences.items()},
        "contiguous_50_episode_stability": block_stability,
    }


def diagnostic_summary(
    arrays: dict[str, np.ndarray], current: dict[str, Any], selected: dict[str, Any] | None
) -> dict[str, Any]:
    raw_loss = arrays["raw_loss"]
    white_loss = arrays["white_loss"]
    scores = arrays["scores"]
    calls = calls_from_scores(scores, OLD_THRESHOLD)
    calibration = []
    margins = []
    gains = []
    for stage in range(3):
        reached = calls >= stage + 1
        score = scores[reached, stage]
        raw_gain = raw_loss[reached, stage] - raw_loss[reached, stage + 1]
        white_gain = white_loss[reached, stage] - white_loss[reached, stage + 1]
        combined = 0.5 * (
            raw_gain / (np.std(raw_loss[:, stage] - raw_loss[:, stage + 1]) + 1e-12)
            + white_gain
            / (np.std(white_loss[:, stage] - white_loss[:, stage + 1]) + 1e-12)
        )
        order = np.argsort(score, kind="stable")
        bins = np.array_split(order, 5)
        calibration.append(
            {
                "stage": stage + 1,
                "rho": float(spearmanr(score, combined).statistic),
                "score_quintile_mean_gain": [
                    {
                        "mean_score": float(score[index].mean()),
                        "mean_combined_gain": float(combined[index].mean()),
                    }
                    for index in bins
                ],
            }
        )
        margin = score - OLD_THRESHOLD
        margins.append(
            {
                "stage": stage + 1,
                "minimum_absolute_margin": float(np.min(np.abs(margin))),
                "fraction_abs_margin_le_1e4": float(np.mean(np.abs(margin) <= 1e-4)),
                "quantiles": {
                    str(q): float(np.quantile(margin, q))
                    for q in (0.05, 0.25, 0.5, 0.75, 0.95)
                },
            }
        )
        gains.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "mean_raw_gain": float(raw_gain.mean()),
                "mean_native_whitened_gain": float(white_gain.mean()),
                "positive_raw_gain_fraction": float(np.mean(raw_gain > 0)),
                "positive_native_whitened_gain_fraction": float(np.mean(white_gain > 0)),
            }
        )
    heterogeneity = {}
    for name in ("raw_vs_analytic", "native_whitened_vs_analytic"):
        values = np.asarray(current["episode_differences"][name])
        heterogeneity[name] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "positive_fraction": float(np.mean(values > 0)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.5)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return {
        "heterogeneity_at_current_threshold": heterogeneity,
        "calibration": calibration,
        "compute_price_at_current_threshold": {
            key: current[key]
            for key in (
                "mean_actual_calls",
                "gate_evaluations",
                "gate_equivalent_calls_per_row",
                "exact_total_compute_mean_calls",
                "total_flops",
            )
        },
        "stagewise_gains": gains,
        "gate_margins": margins,
        "dgp_stability_at_current_threshold": current[
            "contiguous_50_episode_stability"
        ],
        "selected_threshold_summary": None
        if selected is None
        else {
            key: selected[key]
            for key in (
                "threshold",
                "call_histogram",
                "mean_actual_calls",
                "gate_equivalent_calls_per_row",
                "exact_total_compute_mean_calls",
                "total_flops",
                "criteria",
                "stagewise",
                "contiguous_50_episode_stability",
            )
        },
    }


def strip_episode_vectors(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "episode_differences"}


def main() -> None:
    development = ROOT / "development"
    protocol_path = development / "threshold_selection_protocol.json"
    result_path = development / "threshold_selection_result.json"
    diagnosis_path = development / "cycle_001_cycle_003_diagnosis.json"
    selected_path = development / "selected_threshold.json"
    if any(path.exists() for path in (protocol_path, result_path, diagnosis_path, selected_path)):
        raise RuntimeError("threshold selection is immutable and already started")
    expected_manifests = {
        "cycle_001": "41a1235355e5c167759d5d19a055fb48d2cde6372ea20c66a4d46f00f6056e0e",
        "cycle_003": "192ccfd0fa61381511f7f73062deec1804f31b9e123184c13fb4ddf7f87765cf",
    }
    actual_manifests = {
        "cycle_001": sha256_file(CYCLE_001 / "artifact_manifest.json"),
        "cycle_003": sha256_file(CYCLE_003 / "artifact_manifest.json"),
    }
    if actual_manifests != expected_manifests:
        raise RuntimeError("consumed-cycle final manifest drift")
    protocol = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "written_before_counterfactual_target_access": True,
        "script_sha256": sha256_file(Path(__file__)),
        "consumed_input_hashes": {
            "cycle_001_execution": sha256_file(
                CYCLE_001 / "data/prospective_execution.npz"
            ),
            "cycle_003_execution": sha256_file(
                CYCLE_003 / "data/prospective_execution.npz"
            ),
            "cycle_001_final_manifest": actual_manifests["cycle_001"],
            "cycle_003_final_manifest": actual_manifests["cycle_003"],
        },
        "highest_information_change": "threshold_only",
        "unchanged": [
            "base",
            "refiner",
            "dual-linear gate weights",
            "whitening",
            "feature order and operation graph",
            "DGP",
            "comparators",
            "all statistical criteria",
        ],
        "candidate_thresholds": list(CANDIDATES),
        "candidate_multipliers": list(CANDIDATE_MULTIPLIERS),
        "candidate_count": len(CANDIDATES),
        "only_upward_candidates_reason": "consumed traces contain deeper gate scores only for rows reached under the old threshold; upward candidates form path subsets and are therefore counterfactually complete",
        "selection_cohort": "cycle_001, 300 permanently consumed episodes",
        "validation_cohort": "cycle_003, 300 permanently consumed episodes; exactly one selected threshold is evaluated",
        "selection_rule": "among candidates passing every six-contrast lower-bound, two-co-primary Bonferroni, exact-compute, histogram, and stage-sign criterion on cycle 001, choose the largest minimum estimate/bootstrap-SD standardized margin; deterministic tie break chooses the higher threshold",
        "validation_rule": "freeze only if that one selected candidate independently passes every same criterion on cycle 003",
        "bootstrap_replicates_per_evaluation": BOOTSTRAP_REPLICATES,
        "selection_seeds": {
            "bootstrap": SELECTION_BOOTSTRAP_SEED,
            "histogram": SELECTION_HISTOGRAM_SEED,
            "seeded_mixture": SELECTION_SEEDED_MIXTURE_SEED,
        },
        "validation_seeds": {
            "bootstrap": VALIDATION_BOOTSTRAP_SEED,
            "histogram": VALIDATION_HISTOGRAM_SEED,
            "seeded_mixture": VALIDATION_SEEDED_MIXTURE_SEED,
        },
        "development_uses_dense_all_exit_counterfactuals": True,
        "new_prospective_must_use_actual_sparse_execution": True,
        "new_smoke_episodes": 12,
        "new_prospective_episodes": 300,
        "no_sequential_expansion": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(protocol_path, protocol)

    with np.load(
        SOURCE_DISCOVERY / "freeze/gate_contract.npz", allow_pickle=False
    ) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    selection_arrays = load_consumed(CYCLE_001, whitening)
    selection_sampling = np.random.default_rng(SELECTION_BOOTSTRAP_SEED).integers(
        0, 300, size=(BOOTSTRAP_REPLICATES, 300), dtype=np.int32
    )
    selection_results = []
    for threshold in CANDIDATES:
        selection_results.append(
            evaluate(
                selection_arrays,
                threshold,
                selection_sampling,
                SELECTION_HISTOGRAM_SEED,
                SELECTION_SEEDED_MIXTURE_SEED,
            )
        )
    eligible = [value for value in selection_results if value["eligible"]]
    selected = (
        None
        if not eligible
        else max(
            eligible,
            key=lambda value: (value["minimum_standardized_margin"], value["threshold"]),
        )
    )

    validation = None
    validation_arrays = None
    if selected is not None:
        validation_arrays = load_consumed(CYCLE_003, whitening)
        validation_sampling = np.random.default_rng(VALIDATION_BOOTSTRAP_SEED).integers(
            0, 300, size=(BOOTSTRAP_REPLICATES, 300), dtype=np.int32
        )
        validation = evaluate(
            validation_arrays,
            selected["threshold"],
            validation_sampling,
            VALIDATION_HISTOGRAM_SEED,
            VALIDATION_SEEDED_MIXTURE_SEED,
        )
    accepted = selected is not None and validation is not None and validation["eligible"]
    current_selection = next(
        value for value in selection_results if value["threshold"] == OLD_THRESHOLD
    )
    current_validation = None
    if validation_arrays is not None:
        # Current-threshold diagnosis is already consumed in cycle 003 and is
        # recomputed only to cover the required failure-diagnostic dimensions.
        current_validation = evaluate(
            validation_arrays,
            OLD_THRESHOLD,
            np.random.default_rng(VALIDATION_BOOTSTRAP_SEED).integers(
                0, 300, size=(BOOTSTRAP_REPLICATES, 300), dtype=np.int32
            ),
            VALIDATION_HISTOGRAM_SEED,
            VALIDATION_SEEDED_MIXTURE_SEED,
        )
    diagnosis = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "cycle_001_selection": diagnostic_summary(
            selection_arrays, current_selection, selected
        ),
        "cycle_003_one_shot_validation": None
        if validation_arrays is None or current_validation is None
        else diagnostic_summary(validation_arrays, current_validation, validation),
        "diagnostic_dimensions_complete": [
            "heterogeneity",
            "calibration",
            "compute_price",
            "stagewise_gains",
            "gate_margins",
            "DGP_stability",
        ],
        "all_inputs_permanently_consumed": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(diagnosis_path, diagnosis)
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "protocol_sha256": sha256_file(protocol_path),
        "selection_candidates": [strip_episode_vectors(value) for value in selection_results],
        "selected_on_cycle_001": None
        if selected is None
        else strip_episode_vectors(selected),
        "one_shot_cycle_003_validation": None
        if validation is None
        else strip_episode_vectors(validation),
        "validation_passed": bool(accepted),
        "threshold_frozen": bool(accepted),
        "new_prospective_eligible": bool(accepted),
        "v5_outcome_episodes": 0,
    }
    atomic_json(result_path, result)
    if accepted:
        frozen = {
            "schema_version": 1,
            "created_unix_ns": time.time_ns(),
            "threshold": float(selected["threshold"]),
            "old_threshold": OLD_THRESHOLD,
            "change": "threshold_only",
            "selection_protocol_sha256": sha256_file(protocol_path),
            "selection_result_sha256": sha256_file(result_path),
            "diagnosis_sha256": sha256_file(diagnosis_path),
            "selection_cohort": "cycle_001_consumed",
            "validation_cohort": "cycle_003_consumed_one_shot",
            "selection_passed": True,
            "validation_passed": True,
            "base_refiner_gate_whitening_unchanged": True,
            "requires_new_sparse_smoke": 12,
            "requires_new_prospective": 300,
            "v5_outcome_episodes": 0,
        }
        atomic_json(selected_path, frozen)
    print(
        json.dumps(
            {
                "accepted": accepted,
                "selected_threshold": None if selected is None else selected["threshold"],
                "selection_minimum_standardized_margin": None
                if selected is None
                else selected["minimum_standardized_margin"],
                "validation_eligible": None if validation is None else validation["eligible"],
                "selection_result_sha256": sha256_file(result_path),
                "diagnosis_sha256": sha256_file(diagnosis_path),
                "v5_outcome_episodes": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
