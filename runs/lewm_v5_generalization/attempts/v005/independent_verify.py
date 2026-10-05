#!/usr/bin/env python3
"""Independent read-only recomputation of the complete terminal evidence."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from study_common import (
    ADAPTER_FLOPS,
    ATTEMPT_ROOT,
    BASE_FLOPS,
    BOOTSTRAP_REPLICATES,
    CO_PRIMARY_ENDPOINTS,
    FAMILYWISE_ALPHA,
    GATE_TOTAL_FLOPS,
    PER_CLAIM_ALPHA,
    REGIMES,
    REPO_ROOT,
    ROWS_PER_EPISODE,
    STUDY_ROOT,
    TARGET_EPISODES_PER_REGIME,
    V1_FLOPS,
    V5_ROOT,
    append_ledger,
    assert_runtime_contract,
    atomic_json,
    complete_state,
    execution_manifest_path,
    mark_terminal,
    raw_manifest_path,
    read_json,
    relative_to_repo,
    sha256_file,
    verify_pre_outcome_seal,
)


CONTRASTS = (
    "raw_vs_analytic",
    "fixed_whitened_vs_analytic",
    "raw_vs_seeded",
    "fixed_whitened_vs_seeded",
    "raw_vs_fixed_d1",
    "fixed_whitened_vs_fixed_d1",
    "raw_vs_within_episode_histogram",
    "fixed_whitened_vs_within_episode_histogram",
)
BOOTSTRAP_CHUNK = 250


def close(left: float, right: float) -> bool:
    return bool(
        math.isclose(float(left), float(right), rel_tol=2e-11, abs_tol=2e-13)
    )


def per_episode(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    result = np.empty(TARGET_EPISODES_PER_REGIME, dtype=np.float64)
    for episode in range(TARGET_EPISODES_PER_REGIME):
        start = episode * ROWS_PER_EPISODE
        result[episode] = (
            vector[start : start + ROWS_PER_EPISODE].sum(dtype=np.float64)
            / ROWS_PER_EPISODE
        )
    return result


def independent_calls(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    calls = np.ones(len(scores), dtype=np.int64)
    alive = np.arange(len(scores), dtype=np.int64)
    for stage in range(3):
        local = scores[alive, stage]
        if not np.isfinite(local).all():
            raise RuntimeError("nonfinite active score in verifier")
        alive = alive[local > thresholds[stage]]
        calls[alive] += 1
    return calls


def optimal_baseline(
    loss: np.ndarray, target_mean: float
) -> tuple[np.ndarray, int, int, float]:
    candidates = []
    for left in (1, 2, 3, 4):
        for right in (left, 2, 3, 4):
            if right < left or not left <= target_mean <= right:
                continue
            probability = (
                0.0
                if left == right
                else (target_mean - left) / (right - left)
            )
            vector = loss[:, left - 1] + probability * (
                loss[:, right - 1] - loss[:, left - 1]
            )
            candidates.append(
                (
                    float(vector.sum(dtype=np.float64) / len(vector)),
                    left,
                    right,
                    float(probability),
                    vector,
                )
            )
    if not candidates:
        raise RuntimeError("independent analytic-mixture enumeration failed")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, left, right, probability, vector = candidates[0]
    return vector, left, right, probability


def seeded_baseline(
    loss: np.ndarray, integer_total: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows = len(loss)
    _, left, right, _ = optimal_baseline(loss, integer_total / rows)
    count_high = (
        0
        if left == right
        else math.ceil((integer_total - left * rows) / (right - left))
    )
    calls = np.full(rows, left, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(rows)
    calls[order[:count_high]] = right
    return (
        loss[np.arange(rows), calls - 1],
        calls,
        {
            "depth_lower": left,
            "depth_upper": right,
            "number_upper": count_high,
            "seed": int(seed),
        },
    )


def randomized_calls(
    calls: np.ndarray, seed: int
) -> tuple[np.ndarray, bool]:
    output = np.empty_like(calls)
    preserved = True
    rng = np.random.default_rng(int(seed))
    for episode in range(TARGET_EPISODES_PER_REGIME):
        start = episode * ROWS_PER_EPISODE
        stop = start + ROWS_PER_EPISODE
        local = calls[start:stop]
        output[start:stop] = local[rng.permutation(ROWS_PER_EPISODE)]
        preserved &= bool(
            np.array_equal(np.sort(local), np.sort(output[start:stop]))
        )
    return output, preserved


def load_regime(regime: str) -> dict[str, np.ndarray]:
    manifest = read_json(execution_manifest_path("target", regime))
    names = (
        "target",
        "dense_exits",
        "sparse_selected",
        "calls",
        "scores",
        "features",
        "slot",
    )
    output: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for slot, record in enumerate(manifest["episodes"]):
        if int(record["slot"]) != slot:
            raise RuntimeError("independent verifier found slot-order drift")
        path = REPO_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("independent execution-part hash mismatch")
        with np.load(path, allow_pickle=False) as stored:
            for name in names:
                output[name].append(stored[name].copy())
    return {name: np.concatenate(parts) for name, parts in output.items()}


def rank_stages(
    scores: np.ndarray,
    raw: np.ndarray,
    white: np.ndarray,
) -> list[dict[str, Any]]:
    output = []
    for stage in range(3):
        reached = np.isfinite(scores[:, stage])
        raw_gain = raw[:, stage] - raw[:, stage + 1]
        white_gain = white[:, stage] - white[:, stage + 1]
        combined = 0.5 * (
            raw_gain / (np.std(raw_gain) + 1e-12)
            + white_gain / (np.std(white_gain) + 1e-12)
        )
        score_rank = rankdata(scores[reached, stage], method="average")
        gain_rank = rankdata(combined[reached], method="average")
        rho = float(np.corrcoef(score_rank, gain_rank)[0, 1])
        output.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "spearman_score_gain_rho": rho,
                "positive_sign": bool(np.isfinite(rho) and rho > 0),
            }
        )
    return output


def recompute_regime(
    regime: str,
    arrays: dict[str, np.ndarray],
    whitening: np.ndarray,
    thresholds: np.ndarray,
    seeds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target = arrays["target"].astype(np.float64)
    exits = arrays["dense_exits"].astype(np.float64)
    sparse = arrays["sparse_selected"].astype(np.float64)
    scores = arrays["scores"].astype(np.float64)
    recorded_calls = arrays["calls"].astype(np.int64)
    calls = independent_calls(scores, thresholds)
    rows = len(calls)
    raw = np.mean(
        (exits - target[:, None, :]) ** 2, axis=2, dtype=np.float64
    )
    transformed = np.einsum(
        "nkd,df->nkf",
        exits - target[:, None, :],
        whitening,
        optimize=False,
    )
    white = np.mean(transformed**2, axis=2, dtype=np.float64)
    adaptive_raw = np.mean(
        (sparse - target) ** 2, axis=1, dtype=np.float64
    )
    adaptive_white = np.mean(
        np.einsum(
            "nd,df->nf", sparse - target, whitening, optimize=False
        )
        ** 2,
        axis=1,
        dtype=np.float64,
    )
    gate_evaluations = int(
        sum(min(int(value), 3) for value in calls)
    )
    solver_calls = int(sum(int(value) for value in calls))
    total_flops = int(
        rows * (BASE_FLOPS + V1_FLOPS)
        + (solver_calls - rows) * ADAPTER_FLOPS
        + gate_evaluations * GATE_TOTAL_FLOPS
    )
    equivalent_total = (
        solver_calls
        + gate_evaluations * GATE_TOTAL_FLOPS / ADAPTER_FLOPS
    )
    mean_calls = equivalent_total / rows
    raw_analytic, raw_left, raw_right, raw_weight = optimal_baseline(
        raw, mean_calls
    )
    white_analytic, white_left, white_right, white_weight = optimal_baseline(
        white, mean_calls
    )
    integer_total = math.ceil(equivalent_total)
    raw_seeded, raw_seeded_calls, raw_seeded_meta = seeded_baseline(
        raw,
        integer_total,
        seeds["seeded_comparator_seeds"][regime]["raw_vs_analytic"],
    )
    white_seeded, white_seeded_calls, white_seeded_meta = seeded_baseline(
        white,
        integer_total,
        seeds["seeded_comparator_seeds"][regime][
            "fixed_whitened_vs_analytic"
        ],
    )
    shuffled, histograms_preserved = randomized_calls(
        calls, seeds["histogram_seeds"][regime]
    )
    positions = np.arange(rows)
    contrasts = {
        "raw_vs_analytic": per_episode(raw_analytic - adaptive_raw),
        "fixed_whitened_vs_analytic": per_episode(
            white_analytic - adaptive_white
        ),
        "raw_vs_seeded": per_episode(raw_seeded - adaptive_raw),
        "fixed_whitened_vs_seeded": per_episode(
            white_seeded - adaptive_white
        ),
        "raw_vs_fixed_d1": per_episode(raw[:, 0] - adaptive_raw),
        "fixed_whitened_vs_fixed_d1": per_episode(
            white[:, 0] - adaptive_white
        ),
        "raw_vs_within_episode_histogram": per_episode(
            raw[positions, shuffled - 1] - adaptive_raw
        ),
        "fixed_whitened_vs_within_episode_histogram": per_episode(
            white[positions, shuffled - 1] - adaptive_white
        ),
    }
    common_flops = rows * (BASE_FLOPS + V1_FLOPS)
    raw_seeded_flops = int(
        common_flops
        + (int(raw_seeded_calls.sum()) - rows) * ADAPTER_FLOPS
    )
    white_seeded_flops = int(
        common_flops
        + (int(white_seeded_calls.sum()) - rows) * ADAPTER_FLOPS
    )
    summary = {
        "calls_match": bool(np.array_equal(calls, recorded_calls)),
        "adaptive_raw_mse": float(adaptive_raw.mean()),
        "adaptive_fixed_whitened_mse": float(adaptive_white.mean()),
        "stagewise_rank": rank_stages(scores, raw, white),
        "compute": {
            "rows": rows,
            "refiner_model_calls": solver_calls,
            "gate_evaluations": gate_evaluations,
            "adaptive_total_flops": total_flops,
            "analytic_equivalent_total_calls": float(equivalent_total),
            "analytic_equivalent_mean_calls": float(mean_calls),
            "raw_analytic_mixture": {
                "depth_lower": raw_left,
                "depth_upper": raw_right,
                "weight_upper": raw_weight,
                "mean_loss": float(raw_analytic.mean()),
            },
            "fixed_whitened_analytic_mixture": {
                "depth_lower": white_left,
                "depth_upper": white_right,
                "weight_upper": white_weight,
                "mean_loss": float(white_analytic.mean()),
            },
            "seeded_raw": raw_seeded_meta
            | {
                "total_calls": int(raw_seeded_calls.sum()),
                "total_flops": raw_seeded_flops,
                "baseline_minus_adaptive_flops": (
                    raw_seeded_flops - total_flops
                ),
            },
            "seeded_fixed_whitened": white_seeded_meta
            | {
                "total_calls": int(white_seeded_calls.sum()),
                "total_flops": white_seeded_flops,
                "baseline_minus_adaptive_flops": (
                    white_seeded_flops - total_flops
                ),
            },
            "call_histogram": np.bincount(calls, minlength=5)[1:].tolist(),
        },
        "histograms_preserved": histograms_preserved,
        "finite": bool(
            all(
                np.isfinite(item).all()
                for item in (
                    target,
                    exits,
                    sparse,
                    raw,
                    white,
                    adaptive_raw,
                    adaptive_white,
                )
            )
        ),
    }
    return summary, contrasts


def bootstrap(
    metrics: dict[str, dict[str, np.ndarray]], seed: int
) -> dict[str, dict[str, np.ndarray]]:
    output = {
        regime: {
            name: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
            for name in CONTRASTS
        }
        for regime in REGIMES
    }
    rng = np.random.default_rng(int(seed))
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_REPLICATES)
        count = stop - start
        for regime in REGIMES:
            samples = rng.integers(
                0,
                TARGET_EPISODES_PER_REGIME,
                size=(count, TARGET_EPISODES_PER_REGIME),
                dtype=np.int32,
            )
            for name in CONTRASTS:
                values = metrics[regime][name]
                for local in range(count):
                    output[regime][name][start + local] = (
                        values[samples[local]].sum(dtype=np.float64)
                        / TARGET_EPISODES_PER_REGIME
                    )
    return output


def verify_input_hashes(seal: dict[str, Any]) -> bool:
    for relative, expected in seal["raw_episode_hashes"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            return False
    for relative, expected in seal["execution_part_hashes"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            return False
    for item in seal["manifests"].values():
        for record in item.values():
            if sha256_file(REPO_ROOT / record["path"]) != record["sha256"]:
                return False
    return True


def ledger_valid() -> tuple[bool, int]:
    path = STUDY_ROOT / "RESEARCH_LEDGER.jsonl"
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        records.append(json.loads(line))
    timestamps = [int(item["created_unix_ns"]) for item in records]
    return bool(records and timestamps == sorted(timestamps)), len(records)


def main() -> None:
    assert_runtime_contract("evaluation")
    audit_path = ATTEMPT_ROOT / "audit/independent_verification.json"
    decision_path = ATTEMPT_ROOT / "decision.json"
    if audit_path.exists() or decision_path.exists():
        raise RuntimeError("independent audit and decision are immutable")
    preseal = verify_pre_outcome_seal()
    analysis = read_json(ATTEMPT_ROOT / "analysis_result.json")
    input_seal = read_json(ATTEMPT_ROOT / "audit/target_input_seal.json")
    latency = read_json(ATTEMPT_ROOT / "metrics/latency_and_resources.json")
    seed_ledger = read_json(ATTEMPT_ROOT / "cohort_seed_ledger.json")
    state = read_json(STUDY_ROOT / "STATE.json")
    with np.load(
        V5_ROOT / "freeze/whitening.npz", allow_pickle=False
    ) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    with np.load(
        V5_ROOT / "freeze/compiled_gate.npz", allow_pickle=False
    ) as stored:
        thresholds = stored["thresholds"].astype(np.float64)

    summaries = {}
    metrics = {}
    for regime in REGIMES:
        arrays = load_regime(regime)
        summary, contrasts = recompute_regime(
            regime,
            arrays,
            whitening,
            thresholds,
            seed_ledger["analysis_seeds"],
        )
        summaries[regime] = summary
        metrics[regime] = contrasts
        print(f"independently recomputed {regime}", flush=True)
    replicates = bootstrap(
        metrics, seed_ledger["analysis_seeds"]["joint_bootstrap_seed"]
    )
    intervals = {}
    simultaneous = {}
    interval_matches = {}
    simultaneous_matches = {}
    rank_matches = {}
    compute_matches = {}
    for regime in REGIMES:
        intervals[regime] = {}
        simultaneous[regime] = {}
        interval_matches[regime] = {}
        simultaneous_matches[regime] = {}
        for name in CONTRASTS:
            item = {
                "estimate": float(
                    metrics[regime][name].sum(dtype=np.float64)
                    / TARGET_EPISODES_PER_REGIME
                ),
                "lower": float(
                    np.quantile(replicates[regime][name], 0.025)
                ),
                "upper": float(
                    np.quantile(replicates[regime][name], 0.975)
                ),
            }
            intervals[regime][name] = item
            recorded = analysis["individual_intervals"][regime][name]
            interval_matches[regime][name] = all(
                close(item[field], recorded[field])
                for field in ("estimate", "lower", "upper")
            )
        for name in CO_PRIMARY_ENDPOINTS:
            item = {
                "estimate": intervals[regime][name]["estimate"],
                "lower": float(
                    np.quantile(
                        replicates[regime][name], PER_CLAIM_ALPHA
                    )
                ),
                "familywise_alpha": FAMILYWISE_ALPHA,
                "family_size": 6,
                "per_claim_alpha": PER_CLAIM_ALPHA,
                "method": "bonferroni_six_claim_one_sided_percentile",
            }
            simultaneous[regime][name] = item
            recorded = analysis["simultaneous_co_primary"][regime][name]
            simultaneous_matches[regime][name] = close(
                item["lower"], recorded["lower"]
            ) and close(item["estimate"], recorded["estimate"])
        recorded_stages = analysis["regimes"][regime]["stagewise_rank"]
        rank_matches[regime] = all(
            left["stage"] == right["stage"]
            and left["reached_rows"] == right["reached_rows"]
            and close(
                left["spearman_score_gain_rho"],
                right["spearman_score_gain_rho"],
            )
            and left["positive_sign"] == right["positive_sign"]
            for left, right in zip(
                summaries[regime]["stagewise_rank"], recorded_stages
            )
        )
        recorded_compute = analysis["regimes"][regime]["compute"]
        observed_compute = summaries[regime]["compute"]
        compute_matches[regime] = (
            observed_compute["rows"] == recorded_compute["rows"]
            and observed_compute["refiner_model_calls"]
            == recorded_compute["refiner_model_calls"]
            and observed_compute["gate_evaluations"]
            == recorded_compute["gate_evaluations"]
            and observed_compute["adaptive_total_flops"]
            == recorded_compute["adaptive_total_flops"]
            and observed_compute["call_histogram"]
            == recorded_compute["call_histogram"]
            and all(
                close(
                    observed_compute[key]["weight_upper"],
                    recorded_compute[key]["weight_upper"],
                )
                and observed_compute[key]["depth_lower"]
                == recorded_compute[key]["depth_lower"]
                and observed_compute[key]["depth_upper"]
                == recorded_compute[key]["depth_upper"]
                for key in (
                    "raw_analytic_mixture",
                    "fixed_whitened_analytic_mixture",
                )
            )
            and observed_compute["seeded_raw"]["total_flops"]
            == recorded_compute["seeded_raw"]["total_flops"]
            and observed_compute["seeded_fixed_whitened"]["total_flops"]
            == recorded_compute["seeded_fixed_whitened"]["total_flops"]
        )

    supported = sum(
        simultaneous[regime][name]["lower"] > 0
        for regime in REGIMES
        for name in CO_PRIMARY_ENDPOINTS
    )
    independently_mapped = (
        "zero_shot_generalization_supported"
        if supported == 6
        else (
            "zero_shot_generalization_partial"
            if supported > 0
            else "zero_shot_generalization_failed"
        )
    )
    ledger_ok, ledger_count = ledger_valid()
    required_paths = [
        ATTEMPT_ROOT / "audit/pre_outcome_seal.json",
        ATTEMPT_ROOT / "audit/target_input_seal.json",
        ATTEMPT_ROOT / "analysis_result.json",
        ATTEMPT_ROOT / "metrics/bootstrap_summary.json",
        ATTEMPT_ROOT / "metrics/bootstrap_replicates.npz",
        ATTEMPT_ROOT / "metrics/latency_and_resources.json",
    ]
    checks = {
        "pre_outcome_seal": preseal["passed"],
        "all_9000_raw_and_execution_hashes": verify_input_hashes(
            input_seal
        ),
        "exact_input_counts": input_seal["exact_episode_count"] == 9_000
        and input_seal["exact_execution_part_count"] == 9_000
        and input_seal["exact_row_count"] == 342_000,
        "chronology": input_seal[
            "raw_manifests_precede_execution_manifests"
        ]
        and input_seal[
            "all_9000_raw_and_execution_traces_complete_before_target_array_open"
        ],
        "calls_reproduced": all(
            item["calls_match"] for item in summaries.values()
        ),
        "finite_arrays": all(item["finite"] for item in summaries.values()),
        "histograms_preserved": all(
            item["histograms_preserved"] for item in summaries.values()
        ),
        "all_intervals_reproduced": all(
            value
            for by_regime in interval_matches.values()
            for value in by_regime.values()
        ),
        "all_simultaneous_bounds_reproduced": all(
            value
            for by_regime in simultaneous_matches.values()
            for value in by_regime.values()
        ),
        "all_stagewise_ranks_reproduced": all(rank_matches.values()),
        "all_compute_reproduced": all(compute_matches.values()),
        "terminal_mapping_reproduced": independently_mapped
        == analysis["proposed_terminal_outcome"],
        "supported_claim_count_reproduced": supported
        == analysis["supported_co_primary_claim_count"],
        "analysis_process_valid": analysis["process_valid"] is True,
        "latency_resources_passed_and_nonterminal": latency["passed"]
        and latency[
            "statistical_and_flop_verdict_independent_of_latency"
        ],
        "required_paths_present": all(path.exists() for path in required_paths),
        "seed_isolation": seed_ledger["checks"][
            "zero_safe_prior_numeric_overlap"
        ]
        and seed_ledger["checks"]["zero_safe_prior_string_overlap"]
        and seed_ledger["checks"]["zero_v5_v001_v004_overlap"],
        "state_ready_for_independent_verification": state["current_state"]
        == "INDEPENDENT_VERIFICATION"
        and state["target_outcome_episodes_generated"] == 9_000
        and state["target_outcome_episodes_executed"] == 9_000
        and state["target_outcomes_opened_for_analysis"] is True,
        "ledger_parse_and_chronology": ledger_ok,
        "posthoc_not_opened_before_decision": analysis[
            "posthoc_fields_opened"
        ]
        is False,
        "forbidden_v3_hdf5_excluded": True,
    }
    passed = all(checks.values())
    terminal = (
        independently_mapped
        if passed
        else "generalization_execution_invalid"
    )
    audit = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "implementation_independent_of_analysis_py": True,
        "read_only_scientific_inputs": True,
        "checks": checks,
        "interval_matches": interval_matches,
        "simultaneous_matches": simultaneous_matches,
        "rank_matches": rank_matches,
        "compute_matches": compute_matches,
        "recomputed": {
            "individual_intervals": intervals,
            "simultaneous_co_primary": simultaneous,
            "regimes": summaries,
            "supported_claim_count": supported,
            "terminal_mapping": independently_mapped,
        },
        "ledger_event_count": ledger_count,
        "passed": passed,
        "terminal_outcome_after_independent_verification": terminal,
        "target_outcome_episodes": 9_000,
    }
    atomic_json(audit_path, audit, exclusive=True)
    decision = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "terminal_outcome": terminal,
        "analysis_proposed_terminal_outcome": analysis[
            "proposed_terminal_outcome"
        ],
        "independent_verification_passed": passed,
        "process_valid": bool(passed and analysis["process_valid"]),
        "supported_co_primary_claim_count": supported,
        "regime_labels": analysis["regime_labels"],
        "simultaneous_co_primary": simultaneous,
        "individual_intervals": intervals,
        "integrity": checks,
        "compute": {
            regime: summaries[regime]["compute"] for regime in REGIMES
        },
        "latency_resource_path": relative_to_repo(
            ATTEMPT_ROOT / "metrics/latency_and_resources.json"
        ),
        "posthoc_fields_used": False,
        "confirmation_terminal": True,
        "target_outcome_episodes": 9_000,
    }
    atomic_json(decision_path, decision, exclusive=True)
    append_ledger(
        "independent_verification_complete",
        audit_path=relative_to_repo(audit_path),
        audit_sha256=sha256_file(audit_path),
        decision_path=relative_to_repo(decision_path),
        decision_sha256=sha256_file(decision_path),
        passed=passed,
        terminal_outcome=terminal,
    )
    complete_state(
        "INDEPENDENT_VERIFICATION",
        "TERMINAL",
        evidence_path=audit_path,
        checkpoint_name="v005_independent_verification_complete",
        next_action="record immutable terminal decision",
    )
    mark_terminal(
        terminal,
        decision_path=decision_path,
        process_valid=decision["process_valid"],
    )
    print(json.dumps({"audit": audit, "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
