#!/usr/bin/env python3
"""Independent recomputation for the fixed PushT binary confirmation."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PILOT_ROOT = ROOT.parent / "lewm_pusht_replication_pilot"
sys.path.insert(0, str(PILOT_ROOT))

from pusht_core import (  # noqa: E402
    DEPTHS,
    MODEL_CONFIG,
    MODEL_WEIGHTS,
    atomic_json,
    build_causal_features,
    causal_feature_names,
    git_snapshot,
    load_npz,
    normalize_actions,
    read_json,
    sha256_file,
)


THRESHOLD = 0.036598234837386154
COUNT = 240
SEED_FIRST = 15_910_000
SEED_LAST = 15_910_239
BOOTSTRAP_REPLICATES = 20_000
TEST_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")


def close(left: float, right: float, *, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, atol=atol, rtol=rtol))


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def episode_means(values: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    result = np.empty(COUNT, dtype=np.float64)
    for identifier in range(COUNT):
        rows = np.asarray(values)[episode_id == identifier]
        if not len(rows):
            raise RuntimeError("missing episode rows")
        result[identifier] = rows.mean()
    return result


def independent_losses(
    exits: np.ndarray, target: np.ndarray, whitening: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(exits, dtype=np.float64) - np.asarray(target, dtype=np.float64)[:, None]
    raw = np.sum(delta * delta, axis=2) / delta.shape[2]
    transformed = np.matmul(delta, whitening)
    white = np.sum(transformed * transformed, axis=2) / transformed.shape[2]
    return raw, white


def independent_stage1_score(
    arrays: Mapping[str, np.ndarray], fit: Mapping[str, np.ndarray], gate: Mapping[str, np.ndarray]
) -> np.ndarray:
    normalized = normalize_actions(arrays["actions"], fit["action_mean"], fit["action_scale"])
    chunks: list[np.ndarray] = []
    for start in range(0, len(normalized), 777):
        stop = min(start + 777, len(normalized))
        feature = build_causal_features(
            torch.from_numpy(np.ascontiguousarray(arrays["history"][start:stop])),
            torch.from_numpy(np.ascontiguousarray(normalized[start:stop])),
            torch.from_numpy(np.ascontiguousarray(arrays["dense_exits"][start:stop, 0])),
            torch.from_numpy(np.ascontiguousarray(arrays["stage1_update"][start:stop])),
        ).numpy()
        standardized = (feature.astype(np.float64) - gate["feature_mean"][0]) / gate[
            "feature_scale"
        ][0]
        with np.errstate(all="ignore"):
            two_heads = np.matmul(standardized, gate["weights"][0].T)
        chunks.append(np.minimum(two_heads[:, 0], two_heads[:, 1]))
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise RuntimeError("independent nonfinite gate score")
    return result


def independent_mixture(
    losses: np.ndarray, mean_depth: float, episode_id: np.ndarray
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    feasible = 0
    for low in DEPTHS:
        for high in DEPTHS:
            if high < low or mean_depth < low or mean_depth > high:
                continue
            feasible += 1
            p_high = 0.0 if low == high else (mean_depth - low) / (high - low)
            values = losses[:, low - 1] + p_high * (
                losses[:, high - 1] - losses[:, low - 1]
            )
            objective = float(episode_means(values, episode_id).mean())
            item = {
                "depth_lower": low,
                "depth_upper": high,
                "weight_upper": float(p_high),
                "mean_episode_loss": objective,
                "loss": values,
            }
            if best is None or (objective, low, high) < (
                best["mean_episode_loss"], best["depth_lower"], best["depth_upper"]
            ):
                best = item
    if best is None:
        raise RuntimeError("independent comparator search found no feasible pair")
    best["feasible_pair_count"] = feasible
    return best


def independent_bootstrap(values: Mapping[str, np.ndarray], seed: int) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, COUNT, size=(BOOTSTRAP_REPLICATES, COUNT), dtype=np.int32)
    result: dict[str, dict[str, Any]] = {}
    for name, vector in values.items():
        means = np.asarray(vector, dtype=np.float64)[indices].mean(1)
        result[name] = {
            "mean": float(np.asarray(vector).mean()),
            "exploratory_95_ci_low": float(np.quantile(means, 0.025)),
            "exploratory_95_ci_high": float(np.quantile(means, 0.975)),
        }
    return result


def independent_permutation(calls: np.ndarray, episode_id: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty_like(calls)
    for identifier in range(COUNT):
        rows = np.flatnonzero(episode_id == identifier)
        order = rng.permutation(len(rows))
        result[rows] = calls[rows][order]
    return result


def run_tests() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            str(TEST_PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider",
            str(ROOT / "tests/test_confirmation.py"),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "command": [str(TEST_PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_confirmation.py"],
        "output": result.stdout,
    }


def main() -> None:
    status_path = ROOT / "RUN_STATUS.json"
    atomic_json(
        status_path,
        {"schema_version": 1, "status": "running", "phase": "independent check", "updated_unix_ns": time.time_ns()},
    )
    config = read_json(ROOT / "CONFIG.json")
    decision = read_json(ROOT / "DECISION.json")
    arrays = load_npz(ROOT / "EVALUATION_ARRAYS.npz")
    fit = load_npz(PILOT_ROOT / "FIT_ARTIFACTS.npz")
    gate = load_npz(PILOT_ROOT / "FROZEN_GATE.npz")
    tests = run_tests()
    checks: dict[str, Any] = {"focused_tests": tests["passed"]}

    checks["evaluation_arrays_hash"] = decision["evaluation_arrays_sha256"] == sha256_file(
        ROOT / "EVALUATION_ARRAYS.npz"
    )
    correction = config["verification_source_correction"]
    checks["all_locked_source_hashes"] = all(
        (
            correction["original_sha256"] == expected
            and sha256_file(ROOT / relative) == correction["corrected_sha256"]
        )
        if relative == "verify_confirmation.py"
        else sha256_file(ROOT / relative) == expected
        for relative, expected in config["source_hashes_at_lock"].items()
    )
    checks["verifier_only_numerical_tolerance_correction"] = (
        correction["file"] == "verify_confirmation.py"
        and correction["scientific_analysis_or_arrays_rerun"] is False
        and correction["first_failed_check"] == "all_saved_episode_headline_arrays"
    )
    checks["all_frozen_hashes"] = all(
        item["matches"] and sha256_file(Path(item["path"])) == item["expected_sha256"]
        for item in config["frozen_contract"]["hashes"].values()
    )
    checks["cached_base_hashes"] = (
        sha256_file(MODEL_CONFIG)
        == config["frozen_contract"]["hashes"]["model_config"]["expected_sha256"]
        and sha256_file(MODEL_WEIGHTS)
        == config["frozen_contract"]["hashes"]["model_weights"]["expected_sha256"]
    )
    checks["frozen_array_digests"] = all(
        (
            array_digest(fit["action_mean"])
            == config["frozen_contract"]["frozen_array_digests"]["action_mean"],
            array_digest(fit["action_scale"])
            == config["frozen_contract"]["frozen_array_digests"]["action_scale"],
            array_digest(fit["whitening_matrix"])
            == config["frozen_contract"]["frozen_array_digests"]["whitening_matrix"],
            array_digest(gate["feature_mean"][0])
            == config["frozen_contract"]["frozen_array_digests"]["stage1_feature_mean"],
            array_digest(gate["feature_scale"][0])
            == config["frozen_contract"]["frozen_array_digests"]["stage1_feature_scale"],
            array_digest(gate["weights"][0])
            == config["frozen_contract"]["frozen_array_digests"]["stage1_weights"],
        )
    )
    checks["threshold_exact"] = float(gate["thresholds"][0]) == THRESHOLD
    checks["pilot_contract_reproduced_before_cohort"] = (
        config["frozen_contract"]["pilot_stage1_reproduction"]["scores_bitwise_exact"]
        and config["frozen_contract"]["pilot_stage1_reproduction"]["binary_calls_exact"]
        and config["protocol_locked_unix_ns"]
        < config["setup_smoke"]["started_unix_ns"]
        < config["fixed_cohort"]["generation_started_unix_ns"]
    )

    expected_seeds = np.arange(SEED_FIRST, SEED_LAST + 1, dtype=np.int64)
    expected_ids = np.asarray([f"pusht_binary_confirmation_{index:04d}" for index in range(COUNT)])
    checks["exact_240_episode_count"] = (
        len(arrays["episode_ordinal"]) == COUNT
        and np.array_equal(arrays["episode_ordinal"], np.arange(COUNT))
        and np.array_equal(np.unique(arrays["episode_id"]), np.arange(COUNT))
    )
    checks["seeds_and_identifiers_exact"] = (
        np.array_equal(arrays["episode_unique_seed"], expected_seeds)
        and np.array_equal(arrays["episode_identifier"], expected_ids)
        and np.array_equal(np.unique(arrays["episode_seed"]), expected_seeds)
    )
    pilot_seed_values: set[int] = set()
    pilot_config = read_json(PILOT_ROOT / "CONFIG.json")
    for role in ("fit", "selection", "evaluation"):
        pilot_seed_values.update(int(item["seed"]) for item in pilot_config["role_specs"][role])
    pilot_seed_values.update(int(item["seed"]) for item in pilot_config["smoke"]["episodes"])
    checks["seed_disjointness"] = set(map(int, expected_seeds)).isdisjoint(pilot_seed_values)
    checks["exact_two_excluded_smoke_episodes"] = (
        config["setup_smoke"]["exact_excluded_episode_count"] == 2
        and config["setup_smoke"]["no_replays"]
        and config["setup_smoke"]["not_in_target_cohort"]
    )
    checks["no_exclusions_or_replacements"] = (
        config["fixed_cohort"]["no_exclusions"]
        and config["fixed_cohort"]["no_replacements"]
        and bool(np.all(arrays["episode_transition_count"] > 0))
    )
    checks["fresh_simulator_no_hdf5"] = (
        config["fixed_cohort"]["fresh_simulator_only"]
        and not config["fixed_cohort"]["hdf5_or_h5_opened"]
    )
    checks["chronology"] = (
        config["protocol_locked_unix_ns"]
        < config["setup_smoke"]["started_unix_ns"]
        < config["setup_smoke"]["completed_unix_ns"]
        < config["fixed_cohort"]["generation_started_unix_ns"]
        < config["fixed_cohort"]["policy_decision_locked_unix_ns"]
        <= config["fixed_cohort"]["privileged_context_summarized_unix_ns"]
        < decision["analysis_started_unix_ns"]
        < decision["scientific_decision_locked_unix_ns"]
    )
    checks["frozen_models_unchanged"] = (
        config["fixed_cohort"]["base_parameter_digest_before"]
        == config["fixed_cohort"]["base_parameter_digest_after"]
        and config["fixed_cohort"]["refiner_parameter_digest_before"]
        == config["fixed_cohort"]["refiner_parameter_digest_after"]
    )
    checks["finite_lossless_arrays"] = all(
        np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number)
    )

    scores = independent_stage1_score(arrays, fit, gate)
    calls = (1 + (scores > THRESHOLD).astype(np.int8)).astype(np.int8)
    rows = np.arange(len(calls))
    selected = arrays["dense_exits"][rows, calls - 1]
    checks["stage1_scores_exact"] = np.array_equal(scores, arrays["stage1_score"])
    checks["binary_calls_exact"] = np.array_equal(calls, arrays["calls"])
    checks["selected_predictions_exact"] = np.array_equal(selected, arrays["selected"])
    checks["causal_feature_boundary"] = (
        list(inspect.signature(build_causal_features).parameters)
        == ["history", "actions", "current", "update"]
        and not any(
            forbidden in name.lower()
            for name in causal_feature_names()
            for forbidden in ("target", "future", "contact", "reward", "success", "simulator", "geometry", "state")
        )
        and config["fixed_cohort"]["gate_scores_computed"] == ["stage_1"]
        and not config["fixed_cohort"]["stage_2_or_stage_3_gate_scores_computed_or_consulted"]
        and not config["fixed_cohort"]["privileged_context_used_by_policy_eligibility_or_decision"]
    )

    raw, white = independent_losses(arrays["dense_exits"], arrays["target"], fit["whitening_matrix"])
    adaptive_raw = raw[rows, calls - 1]
    adaptive_white = white[rows, calls - 1]
    episode_id = arrays["episode_id"].astype(int)
    compute_config = config["frozen_contract"]["compute"]
    refiner_price = int(compute_config["refiner"]["total_flops_per_call"])
    gate_price = int(compute_config["gate"]["total_flops_per_reached_evaluation"])
    base_price = int(compute_config["base_predictor"]["total_flops_per_transition"])
    norm_price = int(compute_config["gate"]["action_normalization_flops_per_transition"])
    equivalent_depth = (int(calls.sum()) * refiner_price + len(calls) * gate_price) / (
        len(calls) * refiner_price
    )
    raw_comparator = independent_mixture(raw, equivalent_depth, episode_id)
    white_comparator = independent_mixture(white, equivalent_depth, episode_id)
    permutation = independent_permutation(calls, episode_id, int(config["protocol"]["analysis"]["permutation_seed"]))
    endpoint_values = {
        "raw_vs_exact_compute_analytic": episode_means(raw_comparator["loss"] - adaptive_raw, episode_id),
        "fit_whitened_vs_exact_compute_analytic": episode_means(
            white_comparator["loss"] - adaptive_white, episode_id
        ),
        "raw_vs_within_episode_call_randomization": episode_means(
            raw[rows, permutation - 1] - adaptive_raw, episode_id
        ),
        "fit_whitened_vs_within_episode_call_randomization": episode_means(
            white[rows, permutation - 1] - adaptive_white, episode_id
        ),
        "raw_vs_fixed_depth_1": episode_means(raw[:, 0] - adaptive_raw, episode_id),
        "fit_whitened_vs_fixed_depth_1": episode_means(white[:, 0] - adaptive_white, episode_id),
        "raw_vs_fixed_depth_2": episode_means(raw[:, 1] - adaptive_raw, episode_id),
        "fit_whitened_vs_fixed_depth_2": episode_means(white[:, 1] - adaptive_white, episode_id),
    }
    absolute_losses = {
        "adaptive_raw": episode_means(adaptive_raw, episode_id),
        "adaptive_fit_whitened": episode_means(adaptive_white, episode_id),
        "fixed_depth_1_raw": episode_means(raw[:, 0], episode_id),
        "fixed_depth_1_fit_whitened": episode_means(white[:, 0], episode_id),
        "fixed_depth_2_raw": episode_means(raw[:, 1], episode_id),
        "fixed_depth_2_fit_whitened": episode_means(white[:, 1], episode_id),
        "exact_compute_analytic_raw": episode_means(raw_comparator["loss"], episode_id),
        "exact_compute_analytic_fit_whitened": episode_means(white_comparator["loss"], episode_id),
    }
    interval = independent_bootstrap(
        endpoint_values, int(config["protocol"]["analysis"]["bootstrap_seed"])
    )
    loss_interval = independent_bootstrap(
        absolute_losses, int(config["protocol"]["analysis"]["bootstrap_seed"])
    )
    interval_checks = []
    array_checks = []
    for name, values in endpoint_values.items():
        recorded = decision["all_exploratory_episode_benefit_intervals"][name]
        interval_checks.extend(
            close(interval[name][key], recorded[key])
            for key in ("mean", "exploratory_95_ci_low", "exploratory_95_ci_high")
        )
        array_checks.append(
            np.allclose(values, arrays[f"episode_{name}"], rtol=2e-12, atol=2e-14)
        )
    for name, values in absolute_losses.items():
        recorded = decision["absolute_episode_loss_intervals"][name]
        interval_checks.extend(
            close(loss_interval[name][key], recorded[key])
            for key in ("mean", "exploratory_95_ci_low", "exploratory_95_ci_high")
        )
        array_checks.append(
            np.allclose(values, arrays[f"episode_loss_{name}"], rtol=2e-12, atol=2e-14)
        )
    checks["all_saved_episode_headline_arrays"] = all(array_checks)
    checks["all_bootstrap_outputs"] = all(interval_checks)
    checks["permutation_calls_exact"] = np.array_equal(permutation, arrays["permutation_calls"])
    checks["permutation_histograms_preserved"] = all(
        np.array_equal(
            np.bincount(calls[episode_id == identifier], minlength=3),
            np.bincount(permutation[episode_id == identifier], minlength=3),
        )
        for identifier in range(COUNT)
    )

    raw_gain = raw[:, 0] - raw[:, 1]
    white_gain = white[:, 0] - white[:, 1]
    combined = 0.5 * (
        raw_gain / float(gate["raw_gain_scale"][0])
        + white_gain / float(gate["white_gain_scale"][0])
    )
    ranks = {
        "combined_score_gain_spearman": float(spearmanr(scores, combined).statistic),
        "raw_score_gain_spearman": float(spearmanr(scores, raw_gain).statistic),
        "fit_whitened_score_gain_spearman": float(spearmanr(scores, white_gain).statistic),
    }
    checks["stage1_score_gain_ranks"] = all(
        close(value, decision["stage1_score_gain_rank"][name], atol=1e-11)
        for name, value in ranks.items()
    )
    total = (
        len(calls) * (base_price + norm_price)
        + int(calls.sum()) * refiner_price
        + len(calls) * gate_price
    )
    checks["exact_integer_compute"] = (
        total
        == decision["compute"]["adaptive_total_counted_flops"]
        == decision["compute"]["analytic_comparator_total_counted_flops"]
        and decision["compute"]["exact_integer_total_counted_compute_equality"]
    )
    checks["calls_and_histogram"] = (
        int(calls.sum()) == decision["compute"]["refiner_calls"]
        and len(calls) == decision["compute"]["stage1_gate_evaluations"]
        and np.bincount(calls, minlength=3)[1:].tolist()
        == decision["compute"]["call_histogram_depth_1_to_2"]
    )
    checks["strongest_pairwise_comparators"] = all(
        (
            raw_comparator[key] == decision["compute"]["raw_strongest_pairwise_mixture"][key]
            if key in ("depth_lower", "depth_upper", "feasible_pair_count")
            else close(raw_comparator[key], decision["compute"]["raw_strongest_pairwise_mixture"][key])
        )
        and (
            white_comparator[key] == decision["compute"]["fit_whitened_strongest_pairwise_mixture"][key]
            if key in ("depth_lower", "depth_upper", "feasible_pair_count")
            else close(white_comparator[key], decision["compute"]["fit_whitened_strongest_pairwise_mixture"][key])
        )
        for key in ("depth_lower", "depth_upper", "weight_upper", "mean_episode_loss", "feasible_pair_count")
    )

    raw_primary = endpoint_values["raw_vs_exact_compute_analytic"]
    white_primary = endpoint_values["fit_whitened_vs_exact_compute_analytic"]
    sign_relative = {
        "raw": {
            "positive_episode_count": int(np.sum(raw_primary > 0)),
            "zero_episode_count": int(np.sum(raw_primary == 0)),
            "negative_episode_count": int(np.sum(raw_primary < 0)),
            "relative_benefit_fraction_of_comparator_mse": float(
                raw_primary.mean() / absolute_losses["exact_compute_analytic_raw"].mean()
            ),
        },
        "fit_whitened": {
            "positive_episode_count": int(np.sum(white_primary > 0)),
            "zero_episode_count": int(np.sum(white_primary == 0)),
            "negative_episode_count": int(np.sum(white_primary < 0)),
            "relative_benefit_fraction_of_comparator_mse": float(
                white_primary.mean()
                / absolute_losses["exact_compute_analytic_fit_whitened"].mean()
            ),
        },
    }
    checks["sign_counts_and_relative_effects"] = all(
        sign_relative[metric][key] == decision["sign_counts_and_relative_effects"][metric][key]
        if "count" in key
        else close(
            sign_relative[metric][key],
            decision["sign_counts_and_relative_effects"][metric][key],
        )
        for metric in sign_relative
        for key in sign_relative[metric]
    )
    simultaneous_positive = bool(
        interval["raw_vs_exact_compute_analytic"]["exploratory_95_ci_low"] > 0
        and interval["fit_whitened_vs_exact_compute_analytic"]["exploratory_95_ci_low"] > 0
    )
    process_valid_recomputed = bool(
        checks["exact_integer_compute"]
        and checks["causal_feature_boundary"]
        and checks["finite_lossless_arrays"]
        and checks["exact_240_episode_count"]
        and checks["seeds_and_identifiers_exact"]
        and checks["no_exclusions_or_replacements"]
        and checks["all_frozen_hashes"]
    )
    supported_recomputed = bool(
        process_valid_recomputed and simultaneous_positive and ranks["combined_score_gain_spearman"] >= 0
    )
    label_recomputed = (
        "pusht_binary_confirmation_supported"
        if supported_recomputed
        else (
            "pusht_binary_confirmation_not_supported"
            if process_valid_recomputed
            else "pusht_binary_confirmation_execution_invalid"
        )
    )
    checks["decision_label"] = (
        decision["process_valid"] == process_valid_recomputed
        and decision["supported"] == supported_recomputed
        and decision["scientific_label"] == label_recomputed
    )

    current_git = git_snapshot()
    baseline = config["git_baseline"]
    checks["git_head_unchanged"] = current_git["head"] == baseline["head"]
    checks["git_tracked_state_unchanged"] = current_git["tracked_diff"] == baseline["tracked_diff"]
    checks["git_porcelain_scope_unchanged"] = current_git["status"] == baseline["status"]
    passed = all(bool(value) for value in checks.values())
    payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "passed": passed,
        "checks": checks,
        "focused_tests": tests,
        "headline_recomputation": {
            "scientific_label": label_recomputed,
            "raw": interval["raw_vs_exact_compute_analytic"],
            "fit_whitened": interval["fit_whitened_vs_exact_compute_analytic"],
            "simultaneous_raw_lower": interval["raw_vs_exact_compute_analytic"]["exploratory_95_ci_low"],
            "simultaneous_fit_whitened_lower": interval["fit_whitened_vs_exact_compute_analytic"]["exploratory_95_ci_low"],
            "stage1_score_gain_rank": ranks,
            "sign_counts_and_relative_effects": sign_relative,
            "compute_total_each": total,
            "call_histogram_depth_1_to_2": np.bincount(calls, minlength=3)[1:].tolist(),
        },
        "git": current_git,
        "method": "independent float64 recomputation from the saved lossless evaluation arrays",
    }
    atomic_json(ROOT / "INDEPENDENT_CHECK.json", payload)
    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "status": "complete" if passed else "failed",
            "phase": "independent check",
            "updated_unix_ns": time.time_ns(),
            "next_phase": "report" if passed else None,
        },
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "failed_checks": [name for name, value in checks.items() if not value],
                "headline": payload["headline_recomputation"],
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
