#!/usr/bin/env python3
"""Presealed prospective analysis and proposed terminal mapping."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from cycle_common import (
    ADAPTER_FLOPS,
    BASE_FLOPS,
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    REPO_ROOT,
    ROOT,
    SOURCE_DISCOVERY,
    THRESHOLD,
    V1_FLOPS,
    atomic_json,
    atomic_npz,
    read_json,
    sequential_calls,
    sha256_file,
)
from verify_pre_data import verify as verify_pre_data


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 1_975_999_991
HISTOGRAM_SEED = 1_975_888_881
SEEDED_MIXTURE_SEED = 1_975_777_771


def episode_mean(values: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    unique = np.unique(episode_ids)
    return np.asarray([values[episode_ids == episode].mean() for episode in unique])


def strongest_mixture(losses: np.ndarray, mean_calls: float) -> dict[str, Any]:
    best: tuple[float, np.ndarray, int, int, float] | None = None
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
        raise RuntimeError("no supported exact-compute analytic mixture")
    return {
        "mean_loss": best[0],
        "loss": best[1],
        "depth_lower": best[2],
        "depth_upper": best[3],
        "weight_upper": best[4],
    }


def input_seal() -> dict[str, Any]:
    seal_path = ROOT / "audit/prospective_input_seal.json"
    execution_path = ROOT / "data/prospective_execution.npz"
    execution_manifest_path = ROOT / "data/prospective_execution_manifest.json"
    raw_manifest_path = ROOT / "data/prospective_raw_manifest.json"
    execution_manifest = read_json(execution_manifest_path)
    raw_manifest = read_json(raw_manifest_path)
    if len(raw_manifest["episodes"]) != 300 or execution_manifest["episode_count"] != 300:
        raise RuntimeError("prospective input is not exactly 300 episodes")
    files = {
        str(execution_path.relative_to(REPO_ROOT)): sha256_file(execution_path),
        str(execution_manifest_path.relative_to(REPO_ROOT)): sha256_file(execution_manifest_path),
        str(raw_manifest_path.relative_to(REPO_ROOT)): sha256_file(raw_manifest_path),
    }
    raw_hashes = {record["path"]: record["sha256"] for record in raw_manifest["episodes"]}
    for relative, expected in raw_hashes.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"raw prospective hash drift: {relative}")
    payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "files": files,
        "raw_episode_hashes": raw_hashes,
        "exact_episode_count": 300,
        "all_300_raw_and_execution_traces_complete_before_target_array_open": True,
        "raw_manifest_precedes_execution": raw_manifest["created_unix_ns"] < execution_manifest["created_unix_ns"],
        "prospective_targets_opened_before_this_seal": False,
        "v5_outcome_episodes": 0,
    }
    if seal_path.exists():
        existing = read_json(seal_path)
        if existing["files"] != files or existing["raw_episode_hashes"] != raw_hashes:
            raise RuntimeError("existing prospective input seal mismatch")
        return existing
    atomic_json(seal_path, payload, exclusive=True)
    return payload


def main() -> None:
    if (ROOT / "analysis_result.json").exists():
        raise RuntimeError("analysis result is immutable and already exists")
    verify_pre_data()
    sealed_inputs = input_seal()
    execution_path = ROOT / "data/prospective_execution.npz"
    execution_manifest = read_json(ROOT / "data/prospective_execution_manifest.json")
    if sha256_file(execution_path) != execution_manifest["sha256"]:
        raise RuntimeError("prospective execution hash mismatch")

    # This is the first target-array access in cycle 002, after input_seal()
    # has durably recorded all 300 raw inputs and execution traces.
    with np.load(execution_path, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(SOURCE_DISCOVERY / "freeze/gate_contract.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)

    target = arrays["target"].astype(np.float64)
    dense = arrays["dense_exits"].astype(np.float64)
    sparse = arrays["sparse_selected"].astype(np.float64)
    calls = arrays["calls"].astype(np.int64)
    scores = arrays["scores"].astype(np.float64)
    episode_ids = arrays["episode_id"].astype(np.int64)
    if len(calls) != 300 * 38 or not np.array_equal(np.unique(episode_ids), np.arange(300)):
        raise RuntimeError("prospective rows/episode identifiers are not exact")
    row_index = np.arange(len(calls))
    selected_dense = dense[row_index, calls - 1]
    difference = dense - target[:, None, :]
    raw_losses = np.square(difference).mean(axis=2)
    white_difference = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    white_losses = np.square(white_difference).mean(axis=2)
    adaptive_raw = np.square(sparse - target).mean(axis=1)
    adaptive_white = np.square(
        np.einsum("nd,df->nf", sparse - target, whitening, optimize=True)
    ).mean(axis=1)

    operation = read_json(ROOT / "operation_ledger.json")
    feature_flops = int(operation["gate"]["feature_flops_per_reached_evaluation"])
    score_flops = int(operation["gate"]["dual_affine_score_flops_per_reached_evaluation"])
    gate_flops_per = int(operation["gate"]["total_flops_per_reached_evaluation"])
    nonflop_per = int(operation["gate"]["nonflop_comparison_min_per_reached_evaluation"])
    gate_evaluations = int(np.minimum(calls, 3).sum())
    refiner_calls = int(calls.sum())
    common_flops = len(calls) * (BASE_FLOPS + V1_FLOPS)
    adaptive_total_flops = int(
        common_flops
        + (refiner_calls - len(calls)) * ADAPTER_FLOPS
        + gate_evaluations * gate_flops_per
    )
    analytic_total_calls = refiner_calls + gate_evaluations * gate_flops_per / ADAPTER_FLOPS
    analytic_mean_calls = analytic_total_calls / len(calls)
    raw_analytic = strongest_mixture(raw_losses, analytic_mean_calls)
    white_analytic = strongest_mixture(white_losses, analytic_mean_calls)

    integer_target_calls = math.ceil(analytic_total_calls)
    integer_mean_calls = integer_target_calls / len(calls)
    seeded_shape = strongest_mixture(raw_losses, integer_mean_calls)
    lower = seeded_shape["depth_lower"]
    upper = seeded_shape["depth_upper"]
    if lower == upper:
        number_upper = 0
    else:
        number_upper = math.ceil((integer_target_calls - lower * len(calls)) / (upper - lower))
    seeded_calls = np.full(len(calls), lower, dtype=np.int64)
    seeded_rng = np.random.default_rng(SEEDED_MIXTURE_SEED)
    seeded_calls[seeded_rng.permutation(len(calls))[:number_upper]] = upper
    seeded_raw = raw_losses[row_index, seeded_calls - 1]
    seeded_total_flops = int(
        common_flops + (int(seeded_calls.sum()) - len(calls)) * ADAPTER_FLOPS
    )

    histogram_calls = np.empty_like(calls)
    histogram_records = []
    histogram_rng = np.random.default_rng(HISTOGRAM_SEED)
    for episode in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode)
        histogram_calls[indices] = calls[indices][histogram_rng.permutation(len(indices))]
        original_histogram = np.bincount(calls[indices], minlength=5)[1:]
        randomized_histogram = np.bincount(histogram_calls[indices], minlength=5)[1:]
        histogram_records.append(
            {
                "episode_id": int(episode),
                "original": original_histogram.tolist(),
                "randomized": randomized_histogram.tolist(),
                "preserved": bool(np.array_equal(original_histogram, randomized_histogram)),
            }
        )
    histogram_raw = raw_losses[row_index, histogram_calls - 1]
    histogram_white = white_losses[row_index, histogram_calls - 1]

    differences = {
        "raw_vs_analytic": episode_mean(raw_analytic["loss"] - adaptive_raw, episode_ids),
        "raw_vs_seeded": episode_mean(seeded_raw - adaptive_raw, episode_ids),
        "raw_vs_fixed_d1": episode_mean(raw_losses[:, 0] - adaptive_raw, episode_ids),
        "raw_vs_within_episode_histogram": episode_mean(histogram_raw - adaptive_raw, episode_ids),
        "native_whitened_vs_analytic": episode_mean(
            white_analytic["loss"] - adaptive_white, episode_ids
        ),
        "native_whitened_vs_within_episode_histogram": episode_mean(
            histogram_white - adaptive_white, episode_ids
        ),
    }
    if any(len(values) != 300 for values in differences.values()):
        raise RuntimeError("bootstrap unit is not exactly 300 episodes")
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampling = bootstrap_rng.integers(0, 300, size=(BOOTSTRAP_REPLICATES, 300), dtype=np.int32)
    bootstrap = {
        name: values[sampling].mean(axis=1) for name, values in differences.items()
    }
    individual = {
        name: {
            "estimate": float(values.mean()),
            "lower": float(np.quantile(bootstrap[name], 0.025)),
            "upper": float(np.quantile(bootstrap[name], 0.975)),
        }
        for name, values in differences.items()
    }
    co_primary = ("raw_vs_analytic", "native_whitened_vs_analytic")
    # The endpoints have different physical scales.  For the fixed family of
    # two co-primary one-sided claims, alpha/2 = .025 per endpoint gives at
    # least 95% familywise coverage by Bonferroni without common units.
    simultaneous = {
        name: {
            "estimate": float(differences[name].mean()),
            "lower": float(np.quantile(bootstrap[name], 0.025)),
            "method": "bonferroni_two_endpoint_one_sided_percentile",
            "familywise_alpha": 0.05,
            "per_endpoint_alpha": 0.025,
        }
        for name in co_primary
    }

    stages = []
    for stage in range(3):
        mask = np.isfinite(scores[:, stage])
        raw_gain = raw_losses[:, stage] - raw_losses[:, stage + 1]
        white_gain = white_losses[:, stage] - white_losses[:, stage + 1]
        combined_gain = 0.5 * (
            raw_gain / (raw_gain.std() + 1e-12)
            + white_gain / (white_gain.std() + 1e-12)
        )
        statistic = spearmanr(scores[mask, stage], combined_gain[mask]).statistic
        rho = float(statistic)
        stages.append(
            {
                "stage": stage + 1,
                "reached_rows": int(mask.sum()),
                "spearman_score_gain_rho": rho,
                "positive_sign": bool(np.isfinite(rho) and rho > 0),
            }
        )

    numerical_delta = np.abs(sparse - selected_dense)
    reconstructed_calls = sequential_calls(scores, THRESHOLD)
    seed_ledger = read_json(ROOT / "cohort_seed_ledger.json")
    graph_derivation = read_json(ROOT / "audit/flop_derivation_graph.json")
    symbolic_derivation = read_json(ROOT / "audit/flop_derivation_symbolic.json")
    integrity = {
        "pre_data_seal": True,
        "chronology": bool(
            sealed_inputs["all_300_raw_and_execution_traces_complete_before_target_array_open"]
            and sealed_inputs["raw_manifest_precedes_execution"]
        ),
        "exact_episode_count": len(np.unique(episode_ids)) == 300,
        "exact_row_count": len(calls) == 11_400,
        "finite_arrays": bool(
            all(np.isfinite(value).all() for value in (target, dense, sparse, adaptive_raw, adaptive_white))
        ),
        "hashes": True,
        "seed_isolation": not seed_ledger["overlap_with_prior_recorded_numeric_identifiers"]
        and not seed_ledger["overlap_with_prior_recorded_string_identifiers"],
        "input_key_allowlist": execution_manifest["loaded_input_keys"] == ["action", "pixels"],
        "contact_input_exclusion": not execution_manifest["contact_or_privileged_loaded"],
        "sparse_scores_reproduce_calls_exactly": bool(np.array_equal(reconstructed_calls, calls)),
        "sparse_dense_calls_and_histograms_exact": all(
            item["sparse_vs_dense_calls_exact"] and item["call_histogram_exact"]
            for item in execution_manifest["equivalence"]
        ),
        "manual_sparse_forward_selected_exact": all(
            item["manual_sparse_vs_forward_selected_bitwise_exact"]
            for item in execution_manifest["equivalence"]
        ),
        "same_path_repeat_exact": all(
            item["same_sparse_path_outputs_calls_scores_features_bitwise_exact"]
            for item in execution_manifest["equivalence"]
        ),
        "sparse_dense_numerical_contract": bool(
            np.allclose(sparse, selected_dense, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
            and numerical_delta.max() <= NUMERICAL_MAX_ABS
            and all(item["passed"] for item in execution_manifest["equivalence"])
        ),
        "frozen_modules": execution_manifest["module_before"] == execution_manifest["module_after"]
        and execution_manifest["module_after"]["passed"],
        "gradient_absence": execution_manifest["no_gradients"],
        "histograms_preserved": all(item["preserved"] for item in histogram_records),
        "exact_compute_two_derivations": graph_derivation["totals"] == symbolic_derivation["totals"]
        and gate_flops_per == 7997,
        "seeded_comparator_weakly_more_total_compute": seeded_total_flops >= adaptive_total_flops,
        "stagewise_positive": all(item["positive_sign"] for item in stages),
        "zero_v5_outcomes": execution_manifest["v5_outcome_episodes"] == 0,
    }
    statistical_pass = all(item["lower"] > 0 for item in individual.values()) and all(
        item["lower"] > 0 for item in simultaneous.values()
    )
    process_valid = all(integrity.values())
    proposed = (
        "cycle_prospective_discovery_passed"
        if process_valid and statistical_pass
        else (
            "cycle_prospective_discovery_failed" if process_valid else "cycle_execution_invalid"
        )
    )
    compute = {
        "rows": len(calls),
        "base_model_calls": len(calls),
        "refiner_model_calls": refiner_calls,
        "mean_refiner_calls": float(calls.mean()),
        "gate_evaluations": gate_evaluations,
        "base_flops": len(calls) * BASE_FLOPS,
        "mandatory_depth1_flops": len(calls) * V1_FLOPS,
        "additional_refiner_flops": (refiner_calls - len(calls)) * ADAPTER_FLOPS,
        "gate_feature_flops": gate_evaluations * feature_flops,
        "gate_dual_affine_score_flops": gate_evaluations * score_flops,
        "gate_total_flops": gate_evaluations * gate_flops_per,
        "adaptive_total_flops": adaptive_total_flops,
        "nonflop_comparison_min_operations": gate_evaluations * nonflop_per,
        "analytic_equivalent_mean_calls": float(analytic_mean_calls),
        "raw_analytic_mixture": {key: value for key, value in raw_analytic.items() if key != "loss"},
        "native_whitened_analytic_mixture": {
            key: value for key, value in white_analytic.items() if key != "loss"
        },
        "seeded_mixture": {
            "seed": SEEDED_MIXTURE_SEED,
            "depth_lower": lower,
            "depth_upper": upper,
            "number_upper": number_upper,
            "total_calls": int(seeded_calls.sum()),
            "total_flops": seeded_total_flops,
            "baseline_minus_adaptive_flops": seeded_total_flops - adaptive_total_flops,
        },
        "call_histogram": np.bincount(calls, minlength=5)[1:].tolist(),
    }
    analysis_result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "proposed_terminal_outcome": proposed,
        "statistical_pass": statistical_pass,
        "process_valid": process_valid,
        "criteria": individual,
        "simultaneous_co_primary": simultaneous,
        "integrity": integrity,
        "adaptive_raw_mse": float(adaptive_raw.mean()),
        "adaptive_planoracle_native_whitened_mse": float(adaptive_white.mean()),
        "stagewise_rank": stages,
        "compute": compute,
        "latency_excluded_from_terminal_mapping": True,
        "independent_verification_required_before_final_terminal_outcome": True,
        "v5_launched": False,
        "v5_outcome_episodes": 0,
    }
    atomic_npz(ROOT / "metrics/bootstrap_replicates.npz", bootstrap)
    atomic_json(
        ROOT / "metrics/bootstrap_summary.json",
        {
            "schema_version": 1,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "individual": individual,
            "simultaneous": simultaneous,
        },
        exclusive=True,
    )
    atomic_json(
        ROOT / "metrics/within_episode_histogram_ledger.json",
        {
            "schema_version": 1,
            "seed": HISTOGRAM_SEED,
            "episodes": histogram_records,
            "all_preserved": all(item["preserved"] for item in histogram_records),
        },
        exclusive=True,
    )
    atomic_json(ROOT / "metrics/compute_ledger_realized.json", compute, exclusive=True)
    atomic_json(ROOT / "metrics/stagewise_ranking.json", {"stages": stages}, exclusive=True)
    atomic_npz(
        ROOT / "metrics/prospective_episode_metrics.npz",
        {
            "episode_id": np.arange(300, dtype=np.int32),
            "adaptive_raw": episode_mean(adaptive_raw, episode_ids),
            "adaptive_native_whitened": episode_mean(adaptive_white, episode_ids),
            **differences,
        },
    )
    atomic_json(ROOT / "analysis_result.json", analysis_result, exclusive=True)
    print(json.dumps(analysis_result, sort_keys=True))


if __name__ == "__main__":
    main()
