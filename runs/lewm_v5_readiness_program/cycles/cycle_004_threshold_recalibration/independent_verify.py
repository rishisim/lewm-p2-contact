#!/usr/bin/env python3
"""Independent read-only recomputation of effects, intervals, calls, compute, and mapping."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from cycle_common import REPO_ROOT, ROOT, SOURCE_DISCOVERY, read_json, sha256_file
from verify_pre_data import verify as verify_pre_data


def per_episode(values: np.ndarray, identifiers: np.ndarray) -> np.ndarray:
    output = np.empty(300, dtype=np.float64)
    for episode in range(300):
        selected = values[identifiers == episode]
        if len(selected) != 38:
            raise RuntimeError("independent verifier found a non-38-row episode")
        output[episode] = selected.sum(dtype=np.float64) / 38.0
    return output


def optimal_linear_baseline(loss: np.ndarray, target_mean: float) -> tuple[np.ndarray, int, int, float]:
    candidates = []
    for left in (1, 2, 3, 4):
        for right in (1, 2, 3, 4):
            if right < left or target_mean < left or target_mean > right:
                continue
            probability = 0.0 if left == right else (target_mean - left) / (right - left)
            vector = loss[:, left - 1] + probability * (
                loss[:, right - 1] - loss[:, left - 1]
            )
            candidates.append((float(vector.sum() / len(vector)), left, right, probability, vector))
    if not candidates:
        raise RuntimeError("independent mixture enumeration failed")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, left, right, probability, vector = candidates[0]
    return vector, left, right, probability


def independent_calls(scores: np.ndarray, threshold: float) -> np.ndarray:
    result = np.ones(len(scores), dtype=np.int64)
    alive = np.arange(len(scores), dtype=np.int64)
    for column in range(3):
        local = scores[alive, column]
        if not np.isfinite(local).all():
            raise RuntimeError("independent call reconstruction found nonfinite active score")
        alive = alive[local > threshold]
        result[alive] += 1
    return result


def compare_float(left: float, right: float, tolerance: float = 2e-14) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def independent_linear_quantile(values: np.ndarray, probability: float) -> float:
    """Reproduce NumPy's default linear sample quantile without calling it."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    location = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def main() -> None:
    audit_path = ROOT / "audit/independent_verification.json"
    decision_path = ROOT / "decision.json"
    if audit_path.exists() or decision_path.exists():
        raise RuntimeError("independent audit/terminal decision is immutable and already exists")
    preseal = verify_pre_data()
    analysis = read_json(ROOT / "analysis_result.json")
    execution_manifest = read_json(ROOT / "data/prospective_execution_manifest.json")
    raw_manifest = read_json(ROOT / "data/prospective_raw_manifest.json")
    input_seal = read_json(ROOT / "audit/prospective_input_seal.json")
    latency = read_json(ROOT / "metrics/latency.json")
    execution_path = ROOT / "data/prospective_execution.npz"
    if sha256_file(execution_path) != execution_manifest["sha256"]:
        raise RuntimeError("independent verifier rejected the execution hash")
    for relative, expected in input_seal["files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"independent input seal hash mismatch: {relative}")
    for relative, expected in input_seal["raw_episode_hashes"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"independent raw hash mismatch: {relative}")

    with np.load(execution_path, allow_pickle=False) as stored:
        target = stored["target"].astype(np.float64)
        exits = stored["dense_exits"].astype(np.float64)
        sparse = stored["sparse_selected"].astype(np.float64)
        recorded_calls = stored["calls"].astype(np.int64)
        scores = stored["scores"].astype(np.float64)
        identifiers = stored["episode_id"].astype(np.int64)
    with np.load(SOURCE_DISCOVERY / "freeze/gate_contract.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    threshold = float(read_json(ROOT / "protocol.json")["threshold"])
    calls = independent_calls(scores, threshold)
    rows = len(calls)
    positions = np.arange(rows)
    selected_dense = exits[positions, calls - 1]
    raw = np.mean((exits - target[:, None, :]) ** 2, axis=2, dtype=np.float64)
    transformed = np.einsum(
        "nkd,df->nkf", exits - target[:, None, :], whitening, optimize=False
    )
    white = np.mean(transformed**2, axis=2, dtype=np.float64)
    adaptive_raw = np.mean((sparse - target) ** 2, axis=1, dtype=np.float64)
    adaptive_white = np.mean(
        np.einsum("nd,df->nf", sparse - target, whitening, optimize=False) ** 2,
        axis=1,
        dtype=np.float64,
    )

    # Independent closed-form compute, intentionally not imported from analyze.py.
    gate_evaluations = int(sum(min(int(value), 3) for value in calls))
    solver_calls = int(sum(int(value) for value in calls))
    gate_per = 7_997
    total_flops = (
        rows * (70_529_190 + 669_184)
        + (solver_calls - rows) * 264_960
        + gate_evaluations * gate_per
    )
    fractional_calls = solver_calls + gate_evaluations * gate_per / 264_960
    mean_calls = fractional_calls / rows
    raw_analytic, raw_left, raw_right, raw_probability = optimal_linear_baseline(raw, mean_calls)
    white_analytic, white_left, white_right, white_probability = optimal_linear_baseline(
        white, mean_calls
    )

    integer_target = math.ceil(fractional_calls)
    seeded_template, seeded_left, seeded_right, seeded_probability = optimal_linear_baseline(
        raw, integer_target / rows
    )
    del seeded_template, seeded_probability
    seeded = np.full(rows, seeded_left, dtype=np.int64)
    if seeded_left != seeded_right:
        count_high = math.ceil(
            (integer_target - seeded_left * rows) / (seeded_right - seeded_left)
        )
    else:
        count_high = 0
    seeded_rng = np.random.default_rng(2_005_777_771)
    order = seeded_rng.permutation(rows)
    seeded[order[:count_high]] = seeded_right
    seeded_loss = raw[positions, seeded - 1]
    seeded_flops = rows * (70_529_190 + 669_184) + (int(seeded.sum()) - rows) * 264_960

    randomized = np.empty_like(calls)
    histogram_rng = np.random.default_rng(2_005_888_881)
    histograms_preserved = True
    for episode in range(300):
        locations = np.flatnonzero(identifiers == episode)
        randomized[locations] = calls[locations][histogram_rng.permutation(38)]
        histograms_preserved &= bool(
            np.array_equal(np.sort(randomized[locations]), np.sort(calls[locations]))
        )
    randomized_raw = raw[positions, randomized - 1]
    randomized_white = white[positions, randomized - 1]
    contrasts = {
        "raw_vs_analytic": per_episode(raw_analytic - adaptive_raw, identifiers),
        "raw_vs_seeded": per_episode(seeded_loss - adaptive_raw, identifiers),
        "raw_vs_fixed_d1": per_episode(raw[:, 0] - adaptive_raw, identifiers),
        "raw_vs_within_episode_histogram": per_episode(
            randomized_raw - adaptive_raw, identifiers
        ),
        "native_whitened_vs_analytic": per_episode(
            white_analytic - adaptive_white, identifiers
        ),
        "native_whitened_vs_within_episode_histogram": per_episode(
            randomized_white - adaptive_white, identifiers
        ),
    }

    # Independent loop implementation of the 20,000 paired bootstrap.  The
    # analysis implementation is vectorized; both consume the same declared
    # int32 RNG stream.
    bootstrap = {name: np.empty(20_000, dtype=np.float64) for name in contrasts}
    bootstrap_rng = np.random.default_rng(2_005_999_991)
    for replicate in range(20_000):
        sample = bootstrap_rng.integers(0, 300, size=300, dtype=np.int32)
        for name, values in contrasts.items():
            bootstrap[name][replicate] = values[sample].sum(dtype=np.float64) / 300.0
    intervals = {
        name: {
            "estimate": float(values.sum(dtype=np.float64) / 300.0),
            "lower": float(np.quantile(bootstrap[name], 0.025)),
            "upper": float(np.quantile(bootstrap[name], 0.975)),
        }
        for name, values in contrasts.items()
    }
    co_primary = ("raw_vs_analytic", "native_whitened_vs_analytic")
    simultaneous = {
        name: {
            "estimate": intervals[name]["estimate"],
            "lower": independent_linear_quantile(bootstrap[name], 0.025),
            "method": "bonferroni_two_endpoint_one_sided_percentile",
            "familywise_alpha": 0.05,
            "per_endpoint_alpha": 0.025,
        }
        for name in co_primary
    }

    stages = []
    for stage in range(3):
        reached = np.isfinite(scores[:, stage])
        raw_gain = raw[:, stage] - raw[:, stage + 1]
        white_gain = white[:, stage] - white[:, stage + 1]
        gain = 0.5 * (
            raw_gain / (np.std(raw_gain) + 1e-12)
            + white_gain / (np.std(white_gain) + 1e-12)
        )
        score_ranks = rankdata(scores[reached, stage], method="average")
        gain_ranks = rankdata(gain[reached], method="average")
        rho = float(np.corrcoef(score_ranks, gain_ranks)[0, 1])
        stages.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "spearman_score_gain_rho": rho,
                "positive_sign": rho > 0,
            }
        )

    interval_matches = {
        name: all(
            compare_float(intervals[name][field], analysis["criteria"][name][field])
            for field in ("estimate", "lower", "upper")
        )
        for name in intervals
    }
    simultaneous_matches = {
        name: compare_float(
            simultaneous[name]["lower"], analysis["simultaneous_co_primary"][name]["lower"]
        )
        for name in simultaneous
    }
    stage_matches = all(
        left["reached_rows"] == right["reached_rows"]
        and compare_float(left["spearman_score_gain_rho"], right["spearman_score_gain_rho"])
        and left["positive_sign"] == right["positive_sign"]
        for left, right in zip(stages, analysis["stagewise_rank"])
    )
    graph = read_json(ROOT / "operation_graph.json")
    graph_feature = sum(
        int(node["count_per_reached_row"])
        for node in graph["nodes"]
        if node["section"] == "feature" and node["accounting"] == "flop"
    )
    graph_score = sum(
        int(node["count_per_reached_row"])
        for node in graph["nodes"]
        if node["section"] == "gate_score" and node["accounting"] == "flop"
    )
    graph_nonflop = sum(
        int(node["count_per_reached_row"])
        for node in graph["nodes"]
        if node["accounting"] == "non_flop"
    )

    raw_directory = ROOT / "data/prospective_raw"
    actual_raw_npz = {path.name for path in raw_directory.glob("*.npz")}
    actual_sidecars = {path.with_suffix(".npz").name for path in raw_directory.glob("*.json")}
    expected_raw = {Path(item["path"]).name for item in raw_manifest["episodes"]}
    seed_ledger = read_json(ROOT / "cohort_seed_ledger.json")
    required_paths = [
        ROOT / "audit/pre_data_seal.json",
        ROOT / "audit/prospective_input_seal.json",
        ROOT / "analysis_result.json",
        ROOT / "metrics/bootstrap_summary.json",
        ROOT / "metrics/compute_ledger_realized.json",
        ROOT / "metrics/latency.json",
        ROOT / "data/prospective_execution.npz",
        ROOT / "data/prospective_execution_manifest.json",
    ]
    chronology = (
        int(read_json(ROOT / "audit/pre_data_seal.json")["created_unix_ns"])
        < min(int(item["created_unix_ns"]) for item in raw_manifest["episodes"])
        <= int(raw_manifest["created_unix_ns"])
        < int(execution_manifest["created_unix_ns"])
        < int(input_seal["created_unix_ns"])
        < int(analysis["created_unix_ns"])
    )
    statistical_pass = all(item["lower"] > 0 for item in intervals.values()) and all(
        item["lower"] > 0 for item in simultaneous.values()
    )
    independently_proposed = (
        "cycle_prospective_discovery_passed"
        if all(analysis["integrity"].values()) and statistical_pass
        else (
            "cycle_prospective_discovery_failed"
            if all(analysis["integrity"].values())
            else "cycle_execution_invalid"
        )
    )
    checks = {
        "pre_data_seal": preseal["passed"],
        "exact_calls_reproduced_from_scores": bool(np.array_equal(calls, recorded_calls)),
        "exact_call_histogram_reproduced": bool(
            np.array_equal(
                np.bincount(calls, minlength=5)[1:],
                np.asarray(analysis["compute"]["call_histogram"]),
            )
        ),
        "effects_and_individual_intervals_reproduced": all(interval_matches.values()),
        "simultaneous_bounds_reproduced": all(simultaneous_matches.values()),
        "stagewise_ranks_reproduced": stage_matches,
        "adaptive_raw_mse_reproduced": compare_float(
            adaptive_raw.mean(), analysis["adaptive_raw_mse"]
        ),
        "adaptive_native_whitened_mse_reproduced": compare_float(
            adaptive_white.mean(), analysis["adaptive_planoracle_native_whitened_mse"]
        ),
        "exact_compute_reproduced": int(analysis["compute"]["adaptive_total_flops"])
        == int(total_flops)
        and int(analysis["compute"]["gate_evaluations"]) == gate_evaluations
        and int(analysis["compute"]["refiner_model_calls"]) == solver_calls
        and seeded_flops == int(analysis["compute"]["seeded_mixture"]["total_flops"]),
        "operation_graph_rederived": (graph_feature, graph_score, graph_nonflop)
        == (3801, 4196, 5),
        "analytic_mixture_mapping_reproduced": (
            raw_left,
            raw_right,
            raw_probability,
            white_left,
            white_right,
            white_probability,
        )
        == (
            analysis["compute"]["raw_analytic_mixture"]["depth_lower"],
            analysis["compute"]["raw_analytic_mixture"]["depth_upper"],
            analysis["compute"]["raw_analytic_mixture"]["weight_upper"],
            analysis["compute"]["native_whitened_analytic_mixture"]["depth_lower"],
            analysis["compute"]["native_whitened_analytic_mixture"]["depth_upper"],
            analysis["compute"]["native_whitened_analytic_mixture"]["weight_upper"],
        ),
        "seeded_baseline_weakly_more_compute": seeded_flops >= total_flops,
        "within_episode_histograms_reproduced": histograms_preserved,
        "numerical_equivalence_reproduced": bool(
            np.allclose(sparse, selected_dense, rtol=2e-6, atol=2e-7)
            and np.max(np.abs(sparse - selected_dense)) <= 4.76837158203125e-7
        ),
        "raw_path_set_complete": actual_raw_npz == expected_raw
        and actual_sidecars == expected_raw
        and len(expected_raw) == 300,
        "required_path_set_present": all(path.exists() for path in required_paths),
        "chronology": chronology,
        "seed_isolation": not seed_ledger["overlap_with_prior_recorded_numeric_identifiers"]
        and not seed_ledger["overlap_with_prior_recorded_string_identifiers"],
        "frozen_modules_and_gradients": execution_manifest["module_before"]
        == execution_manifest["module_after"]
        and execution_manifest["module_after"]["passed"]
        and execution_manifest["no_gradients"],
        "input_isolation": execution_manifest["loaded_input_keys"] == ["action", "pixels"]
        and not execution_manifest["contact_or_privileged_loaded"],
        "latency_separate_and_smoke_validated": latency["actual_latency_separate_from_fully_counted_flops"]
        and latency["statistical_and_flop_verdict_independent_of_latency"]
        and latency["passed"],
        "terminal_mapping_reproduced": independently_proposed
        == analysis["proposed_terminal_outcome"],
        "zero_v5_outcomes": analysis["v5_outcome_episodes"] == 0
        and execution_manifest["v5_outcome_episodes"] == 0
        and latency["v5_outcome_episodes"] == 0,
    }
    passed = all(checks.values())
    terminal_outcome = independently_proposed if passed else "cycle_execution_invalid"
    audit = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "implementation_independent_of_analyze_py": True,
        "read_only_scientific_inputs": True,
        "checks": checks,
        "interval_matches": interval_matches,
        "simultaneous_matches": simultaneous_matches,
        "recomputed": {
            "criteria": intervals,
            "simultaneous_co_primary": simultaneous,
            "adaptive_raw_mse": float(adaptive_raw.mean()),
            "adaptive_planoracle_native_whitened_mse": float(adaptive_white.mean()),
            "calls": solver_calls,
            "gate_evaluations": gate_evaluations,
            "adaptive_total_flops": int(total_flops),
            "seeded_total_flops": int(seeded_flops),
            "operation_graph": {
                "feature_flops": graph_feature,
                "score_flops": graph_score,
                "nonflop_operations": graph_nonflop,
            },
            "stagewise_rank": stages,
            "terminal_mapping": independently_proposed,
        },
        "passed": passed,
        "terminal_outcome_after_independent_audit": terminal_outcome,
        "v5_outcome_episodes": 0,
    }
    # These are audit/decision outputs only; all scientific inputs above remain read-only.
    from cycle_common import atomic_json

    atomic_json(audit_path, audit, exclusive=True)
    decision = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "terminal_outcome": terminal_outcome,
        "analysis_proposed_terminal_outcome": analysis["proposed_terminal_outcome"],
        "independent_audit_passed": passed,
        "statistical_pass": statistical_pass,
        "process_valid": bool(passed and all(analysis["integrity"].values())),
        "criteria": intervals,
        "simultaneous_co_primary": simultaneous,
        "integrity": checks,
        "compute": analysis["compute"],
        "latency_path": "metrics/latency.json",
        "candidate_eligible_for_v5_package": terminal_outcome
        == "cycle_prospective_discovery_passed",
        "v5_launched": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(decision_path, decision, exclusive=True)
    print(json.dumps({"audit": audit, "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
