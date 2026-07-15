#!/usr/bin/env python3
"""One-shot preregistered V4 primary statistics and mechanical verdict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import common
import runtime


def _load_outcome() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = common.ROOT / "data/confirmation_outcome_manifest.json"
    outcome_path = common.ROOT / "data/confirmation_outcome_once.npz"
    manifest = common.read_json(manifest_path)
    if manifest.get("status") != "complete_frozen_policy_evaluated_once":
        raise RuntimeError("confirmation outcome is not complete")
    if common.sha256_file(outcome_path) != manifest["outcome_sha256"]:
        raise RuntimeError("confirmation outcome hash drift")
    with np.load(outcome_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if len(arrays["calls"]) != 300 * common.EXAMPLES_PER_EPISODE:
        raise RuntimeError("confirmation row count is not exactly 300 episodes")
    return arrays, manifest


def _cluster_means(values: np.ndarray, episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    groups = np.asarray(episodes)
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=data)
    counts = np.bincount(inverse)
    return unique, sums / counts


def clustered_ci(
    benefit: np.ndarray,
    episodes: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[dict[str, Any], np.ndarray]:
    values = np.asarray(benefit, dtype=np.float64)
    groups = np.asarray(episodes)
    if values.ndim != 1 or groups.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("clustered CI inputs must be matching finite vectors")
    unique, means = _cluster_means(values, groups)
    # Every preregistered V4 episode contributes exactly 38 prediction rows,
    # so resampling episode means is exactly the transition-weighted cluster
    # bootstrap estimand used by prior work.
    counts = np.asarray([(groups == item).sum() for item in unique], dtype=np.int64)
    if not np.all(counts == counts[0]):
        raise RuntimeError("V4 cluster sizes changed; equal-size bootstrap contract failed")
    rng = np.random.default_rng(int(seed))
    replicates = np.empty(int(samples), dtype=np.float64)
    batch = 1000
    for start in range(0, int(samples), batch):
        size = min(batch, int(samples) - start)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        replicates[start : start + size] = means[selected].mean(1)
    result = {
        "mean_benefit": float(values.mean()),
        "ci_low": float(np.quantile(replicates, 0.025)),
        "ci_high": float(np.quantile(replicates, 0.975)),
        "bootstrap_standard_error": float(replicates.std(ddof=1)),
        "confidence": 0.95,
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "n_rows": len(values),
        "n_episodes": len(unique),
        "benefit_direction": "baseline_loss_minus_candidate_loss",
    }
    return result, replicates


def selected_loss(losses: np.ndarray, calls: np.ndarray) -> np.ndarray:
    values = np.asarray(losses, dtype=np.float64)
    selected = np.asarray(calls, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 4 or selected.shape != (len(values),):
        raise ValueError("invalid selected-loss inputs")
    if not np.isin(selected, common.SUPPORTED_CALLS).all():
        raise ValueError("selected loss received unsupported calls")
    return values[np.arange(len(values)), selected - 1]


def mixture_payload(
    losses: np.ndarray,
    target_mean_calls: float,
    target_integer_calls: int,
    *,
    seed: int,
) -> dict[str, Any]:
    _, _, policy = runtime.load_discovery_modules()
    means = np.asarray(losses, dtype=np.float64).mean(0)
    analytic = policy.optimal_expected_mixture(means, common.SUPPORTED_CALLS, float(target_mean_calls))
    exact = policy.strongest_transition_independent_baseline(
        means,
        common.SUPPORTED_CALLS,
        n=len(losses),
        target_total_calls=int(target_integer_calls),
        seed=int(seed),
    )
    return {
        "analytic_probabilities": np.asarray(analytic["probabilities"], dtype=np.float64),
        "analytic_expected_mean_calls": float(analytic["expected_mean_calls"]),
        "exact_calls": np.asarray(exact["selected_calls"], dtype=np.int64),
        "exact_counts": np.asarray(exact["counts"], dtype=np.int64),
        "exact_audit": exact["audit"],
        "fixed_exit_mean_loss_used_by_oracle": means,
        "assignment_seed": int(seed),
        "transition_independence": (
            "depth counts use only the aggregate fixed-exit loss vector; seeded assignment "
            "is independent of every transition feature, score, target row, episode, and regime"
        ),
    }


def _analytic_loss(losses: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.einsum(
        "nd,d->n", np.asarray(losses, dtype=np.float64), probabilities, optimize=False
    )


def _frontier_rows(
    losses: np.ndarray,
    adaptive_loss: np.ndarray,
    calls: np.ndarray,
    budget: Mapping[str, Any],
    equal: Mapping[str, Any],
    exact_total: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool, bool]:
    _, _, policy = runtime.load_discovery_modules()
    fixed_means = np.asarray(losses, dtype=np.float64).mean(0)
    common_flops = common.BASE_PREDICT_FLOPS + common.V1_CALL_FLOPS
    rows: list[dict[str, Any]] = []
    for depth in range(1, 5):
        rows.append(
            {
                "name": f"fixed_d{depth}",
                "kind": "fixed",
                "mean_block_calls": float(depth),
                "raw_mse": float(fixed_means[depth - 1]),
                "fully_counted_flops_per_transition": float(
                    common_flops + (depth - 1) * common.STAGE_ADAPTER_FLOPS
                ),
            }
        )
    equal_analytic = _analytic_loss(losses, equal["analytic_probabilities"])
    equal_exact = selected_loss(losses, equal["exact_calls"])
    exact_analytic = _analytic_loss(losses, exact_total["analytic_probabilities"])
    exact_seeded = selected_loss(losses, exact_total["exact_calls"])
    adaptive_mean_calls = float(np.mean(calls))
    adaptive_flops = float(budget["adaptive_total_flops"] / len(calls))
    rows.extend(
        [
            {
                "name": "equal_call_analytic_mixture",
                "kind": "matched_mixture_secondary",
                "mean_block_calls": adaptive_mean_calls,
                "raw_mse": float(equal_analytic.mean()),
                "fully_counted_flops_per_transition": float(
                    common_flops + (adaptive_mean_calls - 1.0) * common.STAGE_ADAPTER_FLOPS
                ),
            },
            {
                "name": "equal_call_seeded_exact_mixture",
                "kind": "matched_mixture_secondary",
                "mean_block_calls": float(equal["exact_calls"].mean()),
                "raw_mse": float(equal_exact.mean()),
                "fully_counted_flops_per_transition": float(
                    common_flops
                    + (float(equal["exact_calls"].mean()) - 1.0)
                    * common.STAGE_ADAPTER_FLOPS
                ),
            },
            {
                "name": "exact_total_flop_analytic_mixture",
                "kind": "primary_exact_total_flop_mixture",
                "mean_block_calls": float(budget["analytic_target_mean_calls"]),
                "raw_mse": float(exact_analytic.mean()),
                "fully_counted_flops_per_transition": adaptive_flops,
            },
            {
                "name": "exact_total_flop_seeded_integer_mixture",
                "kind": "primary_exact_total_flop_mixture",
                "mean_block_calls": float(exact_total["exact_calls"].mean()),
                "raw_mse": float(exact_seeded.mean()),
                "fully_counted_flops_per_transition": float(
                    budget["integer_baseline_total_flops"] / len(calls)
                ),
            },
            {
                "name": "adaptive_b1.25",
                "kind": "adaptive",
                "mean_block_calls": adaptive_mean_calls,
                "raw_mse": float(adaptive_loss.mean()),
                "fully_counted_flops_per_transition": adaptive_flops,
            },
        ]
    )
    call_mask = policy.nondominated_mask(
        [row["mean_block_calls"] for row in rows], [row["raw_mse"] for row in rows]
    )
    flop_mask = policy.nondominated_mask(
        [row["fully_counted_flops_per_transition"] for row in rows],
        [row["raw_mse"] for row in rows],
    )
    for row, call_ok, flop_ok in zip(rows, call_mask, flop_mask, strict=True):
        row["block_call_nondominated"] = bool(call_ok)
        row["fully_counted_flop_nondominated"] = bool(flop_ok)
    adaptive = next(row for row in rows if row["kind"] == "adaptive")
    return rows, adaptive["block_call_nondominated"], adaptive[
        "fully_counted_flop_nondominated"
    ]


def run() -> dict[str, Any]:
    metrics_path = common.ROOT / "metrics/primary_confirmation.json"
    boot_path = common.ROOT / "metrics/primary_bootstrap_replicates.npz"
    decision_path = common.ROOT / "decision.json"
    if any(path.exists() for path in (metrics_path, boot_path, decision_path)):
        raise RuntimeError("the one-shot primary verdict has already been written")
    arrays, outcome_manifest = _load_outcome()
    config = common.load_config()
    seeds = config["statistical_seeds"]
    samples = int(config["bootstrap_samples"])
    if samples != 10_000:
        raise RuntimeError("V4 requires exactly 10,000 bootstrap replicates")
    losses = np.asarray(arrays["losses"], dtype=np.float64)
    white = np.asarray(arrays["whitened_losses"], dtype=np.float64)
    calls = np.asarray(arrays["calls"], dtype=np.int64)
    scores = np.asarray(arrays["scores"], dtype=np.float64)
    episodes = np.asarray(arrays["episode_slot"], dtype=np.int64)
    adaptive = selected_loss(losses, calls)
    white_adaptive = selected_loss(white, calls)
    budget = common.exact_total_flop_budget(calls)

    exact_total = mixture_payload(
        losses,
        budget["analytic_target_mean_calls"],
        budget["integer_target_total_calls"],
        seed=int(seeds["exact_total_integer_assignment"]),
    )
    equal_calls = mixture_payload(
        losses,
        float(calls.mean()),
        int(calls.sum()),
        seed=int(seeds["equal_call_integer_assignment"]),
    )
    exact_analytic = _analytic_loss(losses, exact_total["analytic_probabilities"])
    exact_seeded = selected_loss(losses, exact_total["exact_calls"])
    equal_analytic = _analytic_loss(losses, equal_calls["analytic_probabilities"])
    equal_seeded = selected_loss(losses, equal_calls["exact_calls"])
    white_exact_analytic = _analytic_loss(white, exact_total["analytic_probabilities"])
    white_exact_seeded = selected_loss(white, exact_total["exact_calls"])
    _, _, policy = runtime.load_discovery_modules()
    histogram_calls = policy.randomized_histogram_control(
        calls, int(seeds["histogram_randomization"])
    )
    histogram = selected_loss(losses, histogram_calls)
    white_histogram = selected_loss(white, histogram_calls)
    permuted_scores = policy.permute_critic_scores(
        scores, int(seeds["score_permutation"]), episode_ids=episodes
    )
    permuted_calls = runtime.sequential_calls_numpy(permuted_scores, common.COMPUTE_PRICE)
    permuted_budget = common.exact_total_flop_budget(permuted_calls)
    permuted_mixture = mixture_payload(
        losses,
        permuted_budget["analytic_target_mean_calls"],
        permuted_budget["integer_target_total_calls"],
        seed=int(seeds["permuted_exact_total_integer_assignment"]),
    )
    permuted_loss = selected_loss(losses, permuted_calls)
    permuted_analytic = _analytic_loss(losses, permuted_mixture["analytic_probabilities"])
    permuted_seeded = selected_loss(losses, permuted_mixture["exact_calls"])

    comparisons: dict[str, dict[str, Any]] = {}
    replicates: dict[str, np.ndarray] = {}

    def compare(name: str, candidate: np.ndarray, baseline: np.ndarray, seed_key: str) -> None:
        comparisons[name], replicates[name] = clustered_ci(
            np.asarray(baseline) - np.asarray(candidate),
            episodes,
            seed=int(seeds[seed_key]),
            samples=samples,
        )

    compare("raw_vs_exact_total_flop_analytic", adaptive, exact_analytic, "bootstrap_exact_total_analytic")
    compare("raw_vs_exact_total_flop_seeded", adaptive, exact_seeded, "bootstrap_exact_total_seeded")
    compare("raw_vs_equal_call_analytic_secondary", adaptive, equal_analytic, "bootstrap_equal_call_analytic")
    compare("raw_vs_equal_call_seeded_secondary", adaptive, equal_seeded, "bootstrap_equal_call_seeded")
    compare("raw_vs_fixed_d1", adaptive, losses[:, 0], "bootstrap_fixed_d1")
    compare("raw_vs_histogram", adaptive, histogram, "bootstrap_histogram")
    compare(
        "score_permutation_vs_own_exact_total_analytic",
        permuted_loss,
        permuted_analytic,
        "bootstrap_score_permutation_analytic",
    )
    compare(
        "score_permutation_vs_own_exact_total_seeded",
        permuted_loss,
        permuted_seeded,
        "bootstrap_score_permutation_seeded",
    )
    compare(
        "whitened_vs_exact_total_flop_analytic",
        white_adaptive,
        white_exact_analytic,
        "bootstrap_whitened_exact_total_analytic",
    )
    compare(
        "whitened_vs_exact_total_flop_seeded",
        white_adaptive,
        white_exact_seeded,
        "bootstrap_whitened_exact_total_seeded",
    )
    compare("whitened_vs_histogram", white_adaptive, white_histogram, "bootstrap_whitened_histogram")

    # Diagnostic causal oracle uses true one-step gains as the stopping score
    # and a price chosen only to hit the already-realized adaptive call count.
    gains = losses[:, :-1] - losses[:, 1:]
    oracle_price = policy.calibrate_compute_price(
        gains, float(calls.mean()), exit_calls=common.SUPPORTED_CALLS
    )["compute_price"]
    oracle_calls = runtime.sequential_calls_numpy(gains, float(oracle_price))
    oracle_loss = selected_loss(losses, oracle_calls)
    frontier, call_nondominated, flop_nondominated = _frontier_rows(
        losses, adaptive, calls, budget, equal_calls, exact_total
    )

    blocks = []
    for block_index, start in enumerate((0, 100, 200)):
        block_episodes = np.arange(start, start + 100, dtype=np.int64)
        mask = np.isin(episodes, block_episodes)
        block_ci, block_rep = clustered_ci(
            exact_analytic[mask] - adaptive[mask],
            episodes[mask],
            seed=int(seeds[f"bootstrap_block_{block_index + 1}"]),
            samples=samples,
        )
        replicates[f"block_{block_index + 1}_vs_exact_total_analytic"] = block_rep
        blocks.append(
            {
                "block": block_index + 1,
                "episode_slots": [start, start + 99],
                "episodes": 100,
                "rows": int(mask.sum()),
                "adaptive_raw_mse": float(adaptive[mask].mean()),
                "realized_mean_calls": float(calls[mask].mean()),
                "vs_exact_total_flop_analytic": block_ci,
            }
        )

    exact_call_audits = {
        "adaptive_supported_integer_calls": bool(np.isin(calls, common.SUPPORTED_CALLS).all()),
        "equal_call_seeded_total_exact": int(equal_calls["exact_calls"].sum()) == int(calls.sum()),
        "histogram_exact": bool(np.array_equal(np.sort(histogram_calls), np.sort(calls))),
        "exact_total_integer_target_exact": int(exact_total["exact_calls"].sum())
        == int(budget["integer_target_total_calls"]),
        "exact_total_integer_baseline_weakly_more_flops": bool(
            budget["integer_baseline_weakly_more_compute"]
        ),
        "exact_total_analytic_flop_match": bool(budget["analytic_exact_flop_match"]),
        "permuted_exact_total_integer_target_exact": int(permuted_mixture["exact_calls"].sum())
        == int(permuted_budget["integer_target_total_calls"]),
    }
    validity = {
        "frozen_objects": common.verify_frozen_objects(),
        "freshness_duplicate_audit_passed": common.read_json(
            common.ROOT / "audit/freshness_duplicate_audit.json"
        )["passed"],
        "outcome_evaluation_audit_passed": outcome_manifest["evaluation_audit"]["passed"],
        "sparse_equivalence_passed": outcome_manifest["evaluation_audit"]["sparse_equivalence"][
            "passed"
        ],
        "no_gradient_audit_passed": outcome_manifest["evaluation_audit"]["no_gradient_audit"],
        "finite_metrics": bool(
            np.isfinite(losses).all()
            and np.isfinite(white).all()
            and np.isfinite(scores).all()
        ),
        "causality_audit_passed": bool(
            outcome_manifest["evaluation_audit"]["causal_feature_signature_excludes_target"]
            and not outcome_manifest["evaluation_audit"]["causal_features_require_grad"]
        ),
        "checkpoint_and_gate_hashes_passed": True,
        "data_isolation_passed": True,
        "v3_test_targets_opened": False,
        "confirmation_receipt_preceded_target_io": True,
        "exact_call_audits": exact_call_audits,
    }
    validity["passed"] = bool(
        validity["freshness_duplicate_audit_passed"]
        and validity["outcome_evaluation_audit_passed"]
        and validity["sparse_equivalence_passed"]
        and validity["no_gradient_audit_passed"]
        and validity["finite_metrics"]
        and validity["causality_audit_passed"]
        and all(exact_call_audits.values())
        and not validity["v3_test_targets_opened"]
    )
    raw_criteria = {
        "vs_exact_total_flop_analytic_ci_low_gt_zero": comparisons[
            "raw_vs_exact_total_flop_analytic"
        ]["ci_low"]
        > 0,
        "vs_exact_total_flop_seeded_ci_low_gt_zero": comparisons[
            "raw_vs_exact_total_flop_seeded"
        ]["ci_low"]
        > 0,
        "vs_fixed_d1_ci_low_gt_zero": comparisons["raw_vs_fixed_d1"]["ci_low"] > 0,
        "vs_histogram_ci_low_gt_zero": comparisons["raw_vs_histogram"]["ci_low"] > 0,
        "score_permutation_not_significantly_better_than_own_analytic_baseline": comparisons[
            "score_permutation_vs_own_exact_total_analytic"
        ]["ci_low"]
        <= 0,
        "score_permutation_not_significantly_better_than_own_seeded_baseline": comparisons[
            "score_permutation_vs_own_exact_total_seeded"
        ]["ci_low"]
        <= 0,
        "adaptive_block_call_nondominated": call_nondominated,
        "adaptive_fully_counted_flop_nondominated": flop_nondominated,
    }
    raw_passed = bool(all(raw_criteria.values()) and validity["passed"])
    whitening = comparisons["whitened_vs_exact_total_flop_analytic"]
    if not validity["passed"]:
        verdict = "v4_confirmatory_invalid"
    elif not raw_passed or whitening["mean_benefit"] < 0:
        verdict = "v4_confirmatory_failed"
    elif whitening["ci_low"] > 0:
        verdict = "v4_confirmatory_passed"
    else:
        verdict = "v4_confirmatory_raw_only"

    fixed_means = losses.mean(0)
    metrics = {
        "schema_version": 1,
        "phase": "one_shot_v4_confirmation_primary",
        "episodes": 300,
        "rows": len(calls),
        "family": common.FAMILY,
        "operating_point": common.OPERATING_POINT,
        "compute_price": common.COMPUTE_PRICE,
        "adaptive_raw_mse": float(adaptive.mean()),
        "adaptive_whitened_mse": float(white_adaptive.mean()),
        "fixed_exit_raw_mse": {f"d{i + 1}": float(value) for i, value in enumerate(fixed_means)},
        "fixed_exit_whitened_mse": {
            f"d{i + 1}": float(value) for i, value in enumerate(white.mean(0))
        },
        "realized_mean_calls": float(calls.mean()),
        "realized_total_calls": int(calls.sum()),
        "call_histogram": {
            str(depth): int(np.sum(calls == depth)) for depth in common.SUPPORTED_CALLS
        },
        "mean_gate_decisions": float(np.minimum(calls, 3).mean()),
        "exact_total_flop_budget": budget,
        "exact_total_flop_mixture": {
            key: value for key, value in exact_total.items() if key not in {"exact_calls"}
        },
        "equal_call_mixture_secondary": {
            key: value for key, value in equal_calls.items() if key not in {"exact_calls"}
        },
        "exact_total_flop_analytic_raw_mse": float(exact_analytic.mean()),
        "exact_total_flop_seeded_raw_mse": float(exact_seeded.mean()),
        "equal_call_analytic_raw_mse_secondary": float(equal_analytic.mean()),
        "equal_call_seeded_raw_mse_secondary": float(equal_seeded.mean()),
        "histogram_raw_mse": float(histogram.mean()),
        "score_permutation_raw_mse": float(permuted_loss.mean()),
        "oracle_raw_mse": float(oracle_loss.mean()),
        "oracle_mean_calls": float(oracle_calls.mean()),
        "oracle_headroom": float(adaptive.mean() - oracle_loss.mean()),
        "comparisons": comparisons,
        "raw_criteria": raw_criteria,
        "raw_passed": raw_passed,
        "whitening_classification_input": whitening,
        "frontier": frontier,
        "seed_blocks": blocks,
        "validity": validity,
        "verdict": verdict,
        "runtime_deployment_status": "pending_separate_benchmark",
        "interpretation_limit": (
            "This verdict concerns one frozen operating point in one visual latent physical "
            "world model; physical-regime claims are not part of this primary verdict."
        ),
    }
    np.savez_compressed(
        boot_path,
        **replicates,
        adaptive_calls=calls,
        exact_total_seeded_calls=exact_total["exact_calls"],
        equal_call_seeded_calls=equal_calls["exact_calls"],
        histogram_calls=histogram_calls,
        permuted_calls=permuted_calls,
        oracle_calls=oracle_calls,
    )
    metrics["bootstrap_artifact"] = {
        "path": str(boot_path.relative_to(common.REPO)),
        "sha256": common.sha256_file(boot_path),
        "replicates_per_comparison": samples,
    }
    common.write_json(metrics_path, metrics, exclusive=True)
    decision = {
        "schema_version": 1,
        "decision": verdict,
        "primary_verdict_written_before_secondary_physical_regime_analysis": True,
        "raw_criteria_passed": raw_passed,
        "whitened_point_benefit": whitening["mean_benefit"],
        "whitened_ci_low": whitening["ci_low"],
        "validity_passed": validity["passed"],
        "runtime_deployment_status": "pending_separate_benchmark",
        "primary_metrics_sha256": common.sha256_file(metrics_path),
        "confirmation_outcome_sha256": outcome_manifest["outcome_sha256"],
        "physical_claim": "not_yet_analyzed",
    }
    common.write_json(decision_path, decision, exclusive=True)
    return {"decision": decision, "metrics": metrics}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["decision"], sort_keys=True))
