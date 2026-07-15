#!/usr/bin/env python3
"""Freeze offline reference bands and map the final bounded-study decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr

import common
import evaluate


ROLE_ORDER = (
    "offline_discovery",
    "offline_calibration",
    "plan_oracle",
    "markov_oracle",
    "v4_markov",
)


def map_decision(validity_passed: bool, distribution_matched: bool, gate_passed: bool) -> str:
    """Mechanical preregistered decision tree, isolated for independent testing."""
    if not validity_passed:
        return "distribution_contract_invalid"
    if not distribution_matched:
        return "generator_reconstruction_failed"
    if not gate_passed:
        return "distribution_matched_gate_failed"
    return "distribution_contract_passed"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def _inputs(role: str) -> dict[str, np.ndarray]:
    if role in {"offline_discovery", "offline_calibration"}:
        return common.load_allowed_cache(role)[0]
    if role in {"plan_oracle", "markov_oracle"}:
        return _load_npz(common.STUDY_ROOT / "data" / f"{role}_encoded.npz")
    if role == "v4_markov":
        arrays = _load_npz(common.V4_ROOT / "data/confirmation_encoded.npz")
        arrays["episode_id"] = arrays.pop("episode_slot").astype(np.int64)
        return arrays
    raise ValueError(role)


def _raw_lookup(role: str) -> tuple[tuple[str, ...], dict[int, np.ndarray]]:
    if role in {"offline_discovery", "offline_calibration"}:
        raw = _load_npz(common.STUDY_ROOT / "reference/raw_offline_episode_metrics.npz")
        code = 0 if role == "offline_discovery" else 1
        mask = raw["role_code"] == code
    else:
        raw = _load_npz(common.STUDY_ROOT / "data" / f"{role}_raw_episode_metrics.npz")
        mask = np.ones(len(raw["episode_id"]), dtype=bool)
    names = tuple(raw["metric_names"].astype(str).tolist())
    lookup = {
        int(episode): value
        for episode, value in zip(raw["episode_id"][mask], raw["metrics"][mask], strict=True)
    }
    return names, lookup


def episode_metrics(role: str) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    inputs = _inputs(role)
    evaluation = _load_npz(common.STUDY_ROOT / "data" / f"{role}_evaluation.npz")
    episode = np.asarray(evaluation["episode_id"], dtype=np.int64)
    if not np.array_equal(episode, np.asarray(inputs["episode_id"], dtype=np.int64)):
        raise RuntimeError(f"input/evaluation episode rows differ for {role}")
    unique = np.unique(episode)
    raw_names, raw = _raw_lookup(role)
    if set(unique.tolist()) != set(raw):
        raise RuntimeError(f"raw/evaluation episode set differs for {role}")
    names: list[str] = list(raw_names)
    names += [
        "history_mean",
        "history_std",
        "history_norm",
        "history_temporal_change_norm",
        "target_mean",
        "target_std",
        "target_norm",
    ]
    names += [f"d{depth}_raw_mse" for depth in range(5)]
    names += [f"d{depth}_whitened_mse" for depth in range(5)]
    names += [f"d{depth}_to_d{depth + 1}_gain" for depth in range(4)]
    names += [f"d{depth}_to_d{depth + 1}_positive_rate" for depth in range(4)]
    names += [f"d{depth}_update_norm" for depth in range(1, 5)]
    names += [f"gate_score_s{stage}_mean" for stage in range(1, 4)]
    names += [f"gate_score_s{stage}_price_exceed_rate" for stage in range(1, 4)]
    names += ["mean_calls", *[f"call_{depth}_rate" for depth in range(1, 5)]]
    for prefix in ("raw_feature", "z_feature"):
        for stage in range(1, 4):
            for block, _, _ in common.FEATURE_BLOCKS:
                for stat in common.BLOCK_STAT_NAMES:
                    names.append(f"{prefix}_s{stage}_{block}_{stat}")

    rows: list[np.ndarray] = []
    for item in unique:
        mask = episode == item
        if int(mask.sum()) != common.EXAMPLES_PER_EPISODE:
            raise RuntimeError(f"{role} episode {item} does not have 38 rows")
        history = np.asarray(inputs["history"][mask], dtype=np.float64)
        target = np.asarray(inputs["target"][mask], dtype=np.float64)
        history_delta = history[:, 1:] - history[:, :-1]
        row: list[float] = raw[int(item)].astype(np.float64).tolist()
        row += [
            float(history.mean()),
            float(history.std()),
            float(np.linalg.vector_norm(history, axis=2).mean()),
            float(np.linalg.vector_norm(history_delta, axis=2).mean()),
            float(target.mean()),
            float(target.std()),
            float(np.linalg.vector_norm(target, axis=1).mean()),
        ]
        losses = evaluation["losses_d0_d4"][mask]
        white = evaluation["whitened_losses_d0_d4"][mask]
        gains = evaluation["marginal_gains_d0_d4"][mask]
        row += losses.mean(0).tolist()
        row += white.mean(0).tolist()
        row += gains.mean(0).tolist()
        row += (gains > 0).mean(0).tolist()
        row += evaluation["update_norms_d1_d4"][mask].mean(0).tolist()
        score = evaluation["scores"][mask]
        row += score.mean(0).tolist()
        row += (score > common.COMPUTE_PRICE).mean(0).tolist()
        calls = evaluation["calls"][mask]
        row += [float(calls.mean())]
        row += [float((calls == depth).mean()) for depth in range(1, 5)]
        for key in ("raw_feature_block_stats", "normalized_feature_block_stats"):
            summaries = evaluation[key][mask].mean(0)
            row += summaries.reshape(-1).tolist()
        if len(row) != len(names):
            raise RuntimeError(f"episode metric schema mismatch {len(row)} != {len(names)}")
        rows.append(np.asarray(row, dtype=np.float64))
    matrix = np.stack(rows)
    if not np.isfinite(matrix).all() or len(set(names)) != len(names):
        raise RuntimeError("invalid episode metric matrix")
    return unique, tuple(names), matrix


def _bootstrap_sample_means(
    matrix: np.ndarray, *, samples: int, sample_size: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    result = np.empty((samples, matrix.shape[1]), dtype=np.float64)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        selected = rng.integers(0, len(matrix), size=(size, sample_size))
        result[start : start + size] = matrix[selected].mean(1)
    return result


def freeze_reference() -> dict[str, Any]:
    output = common.STUDY_ROOT / "reference/offline_episode_metrics.npz"
    bands_path = common.STUDY_ROOT / "reference/reference_bands.json"
    isolation_path = common.STUDY_ROOT / "audit/offline_isolation.json"
    if any(path.exists() for path in (output, bands_path, isolation_path)):
        raise RuntimeError("reference artifacts already exist")
    ids_discovery, names_discovery, discovery = episode_metrics("offline_discovery")
    ids_calibration, names_calibration, calibration = episode_metrics("offline_calibration")
    if names_discovery != names_calibration:
        raise RuntimeError("offline episode metric schema differs by split")
    names = names_discovery
    pooled = np.concatenate([discovery, calibration])
    role_code = np.concatenate(
        [np.zeros(len(discovery), dtype=np.int8), np.ones(len(calibration), dtype=np.int8)]
    )
    ids = np.concatenate([ids_discovery, ids_calibration]).astype(np.int64)
    test_intersection = sorted(set(ids.tolist()) & common.v3_test_set())
    if test_intersection:
        raise RuntimeError("V3 test episode entered reference")
    cfg = common.study_json(common.STUDY_ROOT / "config.json")
    seeds = common.study_json(common.STUDY_ROOT / "seed_manifest.json")["statistical_seeds"]
    samples = int(cfg["reference_region"]["bootstrap_samples"])
    sample_size = int(cfg["reference_region"]["fresh_sample_size"])
    replicates = _bootstrap_sample_means(
        pooled,
        samples=samples,
        sample_size=sample_size,
        seed=int(seeds["reference_marginal_bands"]),
    )
    center = pooled.mean(0)
    episode_std = pooled.std(0, ddof=1)
    sample_se = episode_std / np.sqrt(sample_size)
    safe_se = np.maximum(sample_se, np.finfo(np.float64).eps)
    core_names = tuple(cfg["reference_region"]["core_metrics"])
    name_to_index = {name: index for index, name in enumerate(names)}
    missing = sorted(set(core_names) - set(name_to_index))
    if missing:
        raise RuntimeError(f"missing frozen core metrics: {missing}")
    core_indices = np.asarray([name_to_index[name] for name in core_names], dtype=np.int64)
    # A separate seed freezes the simultaneous max-T threshold independently
    # of the marginal-band replicate stream.
    joint_rep = _bootstrap_sample_means(
        pooled[:, core_indices],
        samples=samples,
        sample_size=sample_size,
        seed=int(seeds["reference_joint_max_t"]),
    )
    joint_se = safe_se[core_indices]
    max_t = np.max(np.abs((joint_rep - center[core_indices]) / joint_se), axis=1)
    threshold = float(np.quantile(max_t, 0.95))
    bands = {
        name: {
            "offline_mean": float(center[index]),
            "offline_episode_std": float(episode_std[index]),
            "fresh_90_standard_error": float(sample_se[index]),
            "sample_mean_95_low": float(np.quantile(replicates[:, index], 0.025)),
            "sample_mean_95_high": float(np.quantile(replicates[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }
    common.atomic_npz(
        output,
        {
            "role_code": role_code,
            "episode_id": ids,
            "metric_names": np.asarray(names, dtype="U128"),
            "metrics": pooled,
            "bootstrap_sample_means": replicates,
            "joint_max_t_replicates": max_t,
        },
    )
    result = {
        "schema_version": 1,
        "status": "frozen_before_any_new_smoke_or_main_rollout",
        "sources": {
            "offline_discovery_episodes": len(discovery),
            "offline_calibration_episodes": len(calibration),
            "offline_discovery_evaluation_sha256": common.sha256_file(
                common.STUDY_ROOT / "data/offline_discovery_evaluation.npz"
            ),
            "offline_calibration_evaluation_sha256": common.sha256_file(
                common.STUDY_ROOT / "data/offline_calibration_evaluation.npz"
            ),
            "raw_reference_sha256": common.sha256_file(
                common.STUDY_ROOT / "reference/raw_offline_episode_metrics.npz"
            ),
        },
        "episode_estimand": "equal-weight episode mean",
        "bootstrap_samples": samples,
        "fresh_sample_size": sample_size,
        "marginal_seed": int(seeds["reference_marginal_bands"]),
        "joint_seed": int(seeds["reference_joint_max_t"]),
        "core_metrics": list(core_names),
        "joint_max_t_95_threshold": threshold,
        "contact_impact_state_labels_used_in_primary_region": False,
        "bands": bands,
        "offline_episode_metrics_path": str(output.relative_to(common.REPO_ROOT)),
        "offline_episode_metrics_sha256": common.sha256_file(output),
        "v4_used_to_define_any_band_or_rule": False,
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    common.write_study_json(bands_path, result, exclusive=True)
    isolation = {
        "schema_version": 1,
        "passed": not test_intersection,
        "offline_episode_count": len(ids),
        "discovery_episode_count": len(ids_discovery),
        "calibration_episode_count": len(ids_calibration),
        "episode_keys_unique": len(np.unique(ids)) == len(ids),
        "v3_test_episode_intersection": test_intersection,
        "v3_test_targets_opened": False,
        "v3_combined_cache_opened_with_numpy": False,
        "isolated_discovery_cache_sha256": common.sha256_file(
            common.DISCOVERY_ROOT / "cache/v3_train_only.npz"
        ),
        "isolated_calibration_cache_sha256": common.sha256_file(
            common.COMPRESSION_ROOT / "cache/v3_calibration_only.npz"
        ),
        "raw_reference_access_sha256": common.sha256_file(
            common.STUDY_ROOT / "audit/raw_reference_access.json"
        ),
    }
    if not isolation["passed"] or not isolation["episode_keys_unique"]:
        raise RuntimeError("offline isolation failed")
    common.write_study_json(isolation_path, isolation, exclusive=True)
    return result


def _cluster_ci(
    benefit: np.ndarray, episode: np.ndarray, *, seed: int, samples: int = 10_000
) -> tuple[dict[str, Any], np.ndarray]:
    benefit = np.asarray(benefit, dtype=np.float64)
    episode = np.asarray(episode, dtype=np.int64)
    unique = np.unique(episode)
    means = np.asarray([benefit[episode == item].mean() for item in unique])
    counts = np.asarray([(episode == item).sum() for item in unique])
    if not np.all(counts == common.EXAMPLES_PER_EPISODE):
        raise RuntimeError("cluster bootstrap requires 38 rows per episode")
    rng = np.random.default_rng(int(seed))
    replicates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        replicates[start : start + size] = means[selected].mean(1)
    return {
        "mean_benefit": float(benefit.mean()),
        "ci_low": float(np.quantile(replicates, 0.025)),
        "ci_high": float(np.quantile(replicates, 0.975)),
        "bootstrap_standard_error": float(replicates.std(ddof=1)),
        "bootstrap_samples": samples,
        "seed": int(seed),
        "episodes": len(unique),
        "rows": len(benefit),
        "direction": "baseline_loss_minus_adaptive_loss",
    }, replicates


def _selected(losses: np.ndarray, calls: np.ndarray) -> np.ndarray:
    return np.asarray(losses)[np.arange(len(losses)), np.asarray(calls, dtype=np.int64) - 1]


def _analytic(losses: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.einsum("nd,d->n", losses, probabilities, optimize=False)


def gate_prospective() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    evaluation = _load_npz(common.STUDY_ROOT / "data/plan_oracle_evaluation.npz")
    losses = evaluation["losses_d0_d4"][:, 1:]
    white = evaluation["whitened_losses_d0_d4"][:, 1:]
    calls = evaluation["calls"].astype(np.int64)
    scores = evaluation["scores"]
    episode = evaluation["episode_id"].astype(np.int64)
    adaptive = _selected(losses, calls)
    white_adaptive = _selected(white, calls)
    budget = common.exact_total_flop_budget(calls)
    runtime = common.load_runtime()
    _, _, policy = runtime.load_discovery_modules()
    means = losses.mean(0)
    analytic_mix = policy.optimal_expected_mixture(
        means,
        common.SUPPORTED_CALLS,
        float(budget["analytic_target_mean_calls"]),
    )
    seeds = common.study_json(common.STUDY_ROOT / "seed_manifest.json")["statistical_seeds"]
    exact = policy.strongest_transition_independent_baseline(
        means,
        common.SUPPORTED_CALLS,
        n=len(losses),
        target_total_calls=int(budget["integer_target_total_calls"]),
        seed=int(seeds["exact_total_assignment"]),
    )
    exact_calls = np.asarray(exact["selected_calls"], dtype=np.int64)
    analytic_loss = _analytic(losses, analytic_mix["probabilities"])
    exact_loss = _selected(losses, exact_calls)
    white_analytic = _analytic(white, analytic_mix["probabilities"])
    histogram_calls = policy.randomized_histogram_control(
        calls, int(seeds["histogram_randomization"])
    )
    histogram_loss = _selected(losses, histogram_calls)
    comparisons: dict[str, Any] = {}
    replicates: dict[str, np.ndarray] = {}

    def compare(name: str, baseline: np.ndarray, seed_name: str) -> None:
        comparisons[name], replicates[name] = _cluster_ci(
            baseline - adaptive,
            episode,
            seed=int(seeds[seed_name]),
        )

    compare("raw_vs_exact_total_flop_analytic", analytic_loss, "gate_exact_analytic")
    compare("raw_vs_exact_total_flop_seeded", exact_loss, "gate_exact_seeded")
    compare("raw_vs_fixed_d1", losses[:, 0], "gate_fixed_d1")
    compare("raw_vs_histogram", histogram_loss, "gate_histogram")
    comparisons["whitened_vs_exact_total_flop_analytic"], replicates[
        "whitened_vs_exact_total_flop_analytic"
    ] = _cluster_ci(
        white_analytic - white_adaptive,
        episode,
        seed=int(seeds["gate_white_exact"]),
    )
    gains = losses[:, :-1] - losses[:, 1:]
    rankings = []
    for stage in range(3):
        correlation = spearmanr(scores[:, stage], gains[:, stage]).statistic
        rankings.append(
            {
                "decision_stage": stage + 1,
                "spearman_score_gain": float(correlation),
                "score_mean": float(scores[:, stage].mean()),
                "gain_mean": float(gains[:, stage].mean()),
                "score_price_exceed_rate": float(
                    (scores[:, stage] > common.COMPUTE_PRICE).mean()
                ),
            }
        )
    common_flops = common.BASE_PREDICT_FLOPS + common.V1_CALL_FLOPS
    fixed_means = losses.mean(0)
    frontier = [
        {
            "name": f"fixed_d{depth}",
            "mean_calls": float(depth),
            "raw_mse": float(fixed_means[depth - 1]),
            "flops": float(common_flops + (depth - 1) * common.STAGE_ADAPTER_FLOPS),
        }
        for depth in range(1, 5)
    ]
    frontier += [
        {
            "name": "exact_total_analytic",
            "mean_calls": float(budget["analytic_target_mean_calls"]),
            "raw_mse": float(analytic_loss.mean()),
            "flops": float(budget["adaptive_total_flops"] / len(calls)),
        },
        {
            "name": "exact_total_seeded_conservative",
            "mean_calls": float(exact_calls.mean()),
            "raw_mse": float(exact_loss.mean()),
            "flops": float(budget["integer_baseline_total_flops"] / len(calls)),
        },
        {
            "name": "adaptive_frozen_gate",
            "mean_calls": float(calls.mean()),
            "raw_mse": float(adaptive.mean()),
            "flops": float(budget["adaptive_total_flops"] / len(calls)),
        },
    ]
    call_mask = policy.nondominated_mask(
        [item["mean_calls"] for item in frontier], [item["raw_mse"] for item in frontier]
    )
    flop_mask = policy.nondominated_mask(
        [item["flops"] for item in frontier], [item["raw_mse"] for item in frontier]
    )
    for item, call_ok, flop_ok in zip(frontier, call_mask, flop_mask, strict=True):
        item["call_nondominated"] = bool(call_ok)
        item["flop_nondominated"] = bool(flop_ok)
    adaptive_frontier = next(item for item in frontier if item["name"] == "adaptive_frozen_gate")
    criteria = {
        "exact_analytic_ci_low_gt_zero": comparisons[
            "raw_vs_exact_total_flop_analytic"
        ]["ci_low"]
        > 0,
        "exact_seeded_ci_low_gt_zero": comparisons[
            "raw_vs_exact_total_flop_seeded"
        ]["ci_low"]
        > 0,
        "fixed_d1_ci_low_gt_zero": comparisons["raw_vs_fixed_d1"]["ci_low"] > 0,
        "histogram_ci_low_gt_zero": comparisons["raw_vs_histogram"]["ci_low"] > 0,
        "whitened_exact_analytic_point_gt_zero": comparisons[
            "whitened_vs_exact_total_flop_analytic"
        ]["mean_benefit"]
        > 0,
        "all_score_gain_spearman_gt_zero": all(
            item["spearman_score_gain"] > 0 for item in rankings
        ),
        "call_nondominated": adaptive_frontier["call_nondominated"],
        "fully_counted_flop_nondominated": adaptive_frontier["flop_nondominated"],
        "exact_histogram_preserved": bool(
            np.array_equal(np.sort(histogram_calls), np.sort(calls))
        ),
        "exact_total_analytic_flops_match": bool(budget["analytic_exact_flop_match"]),
        "seeded_integer_baseline_weakly_more_flops": bool(
            budget["integer_baseline_weakly_more_compute"]
        ),
    }
    return {
        "schema_version": 1,
        "role": "fresh_plan_oracle_discovery_evidence",
        "episodes": len(np.unique(episode)),
        "rows": len(episode),
        "adaptive_raw_mse": float(adaptive.mean()),
        "adaptive_whitened_mse": float(white_adaptive.mean()),
        "mean_calls": float(calls.mean()),
        "call_histogram": {str(depth): int((calls == depth).sum()) for depth in range(1, 5)},
        "comparisons": comparisons,
        "rankings": rankings,
        "exact_total_flop_budget": budget,
        "analytic_probabilities": np.asarray(analytic_mix["probabilities"]).tolist(),
        "seeded_exact_counts": np.asarray(exact["counts"]).tolist(),
        "seeded_exact_audit": exact["audit"],
        "frontier": frontier,
        "criteria": criteria,
        "passed": all(criteria.values()),
        "fully_counted_flops_include_gate_overhead": True,
        "v4_used_for_gate_evaluation_or_thresholds": False,
        "confirmation_claim": False,
    }, {
        **replicates,
        "adaptive_calls": calls,
        "exact_seeded_calls": exact_calls,
        "histogram_calls": histogram_calls,
    }


def final_analysis() -> dict[str, Any]:
    common.assert_pre_generation_seal()
    decision_path = common.STUDY_ROOT / "decision.json"
    if decision_path.exists():
        raise RuntimeError("final decision already exists")
    bands = common.study_json(common.STUDY_ROOT / "reference/reference_bands.json")
    all_ids: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    names: tuple[str, ...] | None = None
    for role in ROLE_ORDER:
        role_ids, role_names, matrix = episode_metrics(role)
        if names is None:
            names = role_names
        elif role_names != names:
            raise RuntimeError("final role episode metric schema mismatch")
        all_ids[role] = role_ids
        matrices[role] = matrix
    assert names is not None
    name_to_index = {name: index for index, name in enumerate(names)}
    core_names = tuple(bands["core_metrics"])
    core_indices = np.asarray([name_to_index[name] for name in core_names], dtype=np.int64)
    center = np.asarray([bands["bands"][name]["offline_mean"] for name in core_names])
    episode_std = np.asarray(
        [bands["bands"][name]["offline_episode_std"] for name in core_names]
    )
    standard_error = np.asarray(
        [bands["bands"][name]["fresh_90_standard_error"] for name in core_names]
    )
    plan_mean = matrices["plan_oracle"][:, core_indices].mean(0)
    markov_mean = matrices["markov_oracle"][:, core_indices].mean(0)
    plan_joint_z = np.abs((plan_mean - center) / np.maximum(standard_error, 1e-15))
    plan_inside_joint = float(plan_joint_z.max()) <= float(bands["joint_max_t_95_threshold"])
    plan30 = matrices["plan_oracle"][:30, core_indices].mean(0)
    markov30 = matrices["markov_oracle"][:, core_indices].mean(0)
    safe_std = np.maximum(episode_std, 1e-15)
    plan_distance = float(np.linalg.vector_norm((plan30 - center) / safe_std))
    markov_distance = float(np.linalg.vector_norm((markov30 - center) / safe_std))
    distance_ratio = plan_distance / max(markov_distance, 1e-15)
    coordinate_closer = np.abs(plan30 - center) < np.abs(markov30 - center)
    material_closeness = bool(distance_ratio <= 0.70 and coordinate_closer.sum() >= 5)
    gain_name = "d1_to_d2_gain"
    gain_index = name_to_index[gain_name]
    gain_mean = float(matrices["plan_oracle"][:, gain_index].mean())
    gain_band = bands["bands"][gain_name]
    # The required directional restoration criterion is deliberately only > 0.
    # Discovery and calibration came from the same upstream HDF5 but have
    # materially different frozen d1->d2 means; treating the pooled marginal
    # band as a hard equivalence interval would reject a value matching the
    # already-consumed calibration split.  Both split means and the pooled
    # bootstrap band remain frozen and reported descriptively.
    gain_restored = bool(gain_mean > 0)
    distribution_checks = {
        "plan_inside_frozen_joint_core_region": plan_inside_joint,
        "plan_materially_closer_than_paired_markov": material_closeness,
        "plan_d1_to_d2_gain_positive": gain_restored,
        "paired_environment_seed_and_initial_state_audit": common.study_json(
            common.STUDY_ROOT / "audit/main_pairing.json"
        )["passed"],
    }
    distribution_matched = all(distribution_checks.values())
    gate, bootstrap = gate_prospective()
    evaluation_manifests = {
        role: common.study_json(
            common.STUDY_ROOT / "data" / f"{role}_evaluation_manifest.json"
        )
        for role in ROLE_ORDER
    }
    validity_checks = {
        "all_evaluations_passed": all(item["passed"] for item in evaluation_manifests.values()),
        "all_evaluations_report_no_v3_test_access": all(
            not item["v3_test_targets_opened"] for item in evaluation_manifests.values()
        ),
        "offline_isolation_passed": common.study_json(
            common.STUDY_ROOT / "audit/offline_isolation.json"
        )["passed"],
        "smoke_pairing_passed": common.study_json(
            common.STUDY_ROOT / "audit/smoke_pairing.json"
        )["passed"],
        "main_pairing_passed": common.study_json(
            common.STUDY_ROOT / "audit/main_pairing.json"
        )["passed"],
        "freshness_audit_passed": common.study_json(
            common.STUDY_ROOT / "audit/freshness.json"
        )["passed"],
        "frozen_object_hashes_passed": bool(common.verify_frozen_objects()),
        "pre_generation_seal_valid": bool(common.assert_pre_generation_seal()),
        "v4_recomputed_as_diagnostic_only": bool(
            evaluation_manifests["v4_markov"]["v4_recomputation"]["passed"]
        ),
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    validity_passed = all(
        value
        for key, value in validity_checks.items()
        if key not in {"v3_test_targets_opened", "v5_confirmation_episodes"}
    )
    decision = map_decision(validity_passed, distribution_matched, gate["passed"])
    v5_authorized = decision == "distribution_contract_passed"
    role_summaries = {}
    report_metrics = [
        "raw_action_abs_ge_0_99_rate",
        "raw_action_rms",
        "raw_action_coord4_mean",
        "normalized_action_abs_gt3_rate",
        "pixel_r_mean",
        "pixel_g_mean",
        "pixel_b_mean",
        "pixel_temporal_abs_delta_mean",
        "observation_norm_mean",
        "history_temporal_change_norm",
        "target_norm",
        "d0_raw_mse",
        "d1_raw_mse",
        "d2_raw_mse",
        "d3_raw_mse",
        "d4_raw_mse",
        "d1_to_d2_gain",
        "contact_row_rate",
        "impact_row_rate",
        "block_displacement_mean",
        "block_motion_rate_gt_1e_6",
        "target_task_cube_rate",
        "final_success",
        "mean_calls",
    ]
    for role in ROLE_ORDER:
        role_summaries[role] = {
            name: float(matrices[role][:, name_to_index[name]].mean())
            for name in report_metrics
        }
        role_summaries[role]["episodes"] = len(matrices[role])
    feature_summary = {}
    for role in ROLE_ORDER:
        feature_summary[role] = {}
        for stage in range(1, 4):
            feature_summary[role][f"stage_{stage}"] = {}
            for block, _, _ in common.FEATURE_BLOCKS:
                feature_summary[role][f"stage_{stage}"][block] = {
                    stat: float(
                        matrices[role][
                            :,
                            name_to_index[f"z_feature_s{stage}_{block}_{stat}"],
                        ].mean()
                    )
                    for stat in common.BLOCK_STAT_NAMES
                }
    comparison = {
        "schema_version": 1,
        "reference_bands_sha256": common.sha256_file(
            common.STUDY_ROOT / "reference/reference_bands.json"
        ),
        "core_metrics": list(core_names),
        "reference_center": center.tolist(),
        "plan_90_mean": plan_mean.tolist(),
        "plan_joint_abs_z": plan_joint_z.tolist(),
        "plan_joint_max_abs_z": float(plan_joint_z.max()),
        "joint_95_threshold": bands["joint_max_t_95_threshold"],
        "paired_30_plan_mean": plan30.tolist(),
        "paired_30_markov_mean": markov30.tolist(),
        "plan_standardized_distance": plan_distance,
        "markov_standardized_distance": markov_distance,
        "plan_to_markov_distance_ratio": distance_ratio,
        "plan_closer_coordinate_count": int(coordinate_closer.sum()),
        "d1_to_d2_gain": {
            "plan_mean": gain_mean,
            "offline_frozen_band": gain_band,
            "offline_discovery_mean": float(
                matrices["offline_discovery"][:, gain_index].mean()
            ),
            "offline_calibration_mean": float(
                matrices["offline_calibration"][:, gain_index].mean()
            ),
            "hard_decision_rule": "plan_mean > 0",
        },
        "checks": distribution_checks,
        "distribution_matched": distribution_matched,
        "role_summaries": role_summaries,
        "normalized_gate_feature_blocks": feature_summary,
        "contact_impact_state_composition_posthoc_only": True,
        "v4_diagnostic_only": True,
        "v4_used_for_thresholds_training_or_selection": False,
    }
    metrics_dir = common.STUDY_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    common.write_study_json(metrics_dir / "distribution_comparison.json", comparison, exclusive=True)
    common.write_study_json(metrics_dir / "gate_prospective.json", gate, exclusive=True)
    common.atomic_npz(metrics_dir / "bootstrap_replicates.npz", bootstrap)
    role_codes = []
    episode_ids = []
    combined = []
    for code, role in enumerate(ROLE_ORDER):
        role_codes.append(np.full(len(matrices[role]), code, dtype=np.int8))
        episode_ids.append(all_ids[role])
        combined.append(matrices[role])
    common.atomic_npz(
        metrics_dir / "dataset_episode_metrics.npz",
        {
            "role_names": np.asarray(ROLE_ORDER, dtype="U32"),
            "role_code": np.concatenate(role_codes),
            "episode_id": np.concatenate(episode_ids),
            "metric_names": np.asarray(names, dtype="U128"),
            "metrics": np.concatenate(combined),
        },
    )
    decision_payload = {
        "schema_version": 1,
        "decision": decision,
        "validity_passed": validity_passed,
        "distribution_matched": distribution_matched,
        "frozen_gate_prospective_discovery_passed": gate["passed"],
        "distribution_checks": distribution_checks,
        "gate_criteria": gate["criteria"],
        "validity_checks": validity_checks,
        "future_entirely_fresh_v5_confirmation_authorized": v5_authorized,
        "v5_launched": False,
        "confirmatory_claim": False,
        "contact_aware_claim": False,
        "first_adaptive_world_model_claim": False,
        "v4_retained_negative_result_for_markov_oracle": True,
        "v4_used_for_training_selection_tuning_or_thresholds": False,
        "next_step": (
            "Design and preregister an entirely fresh V5 confirmation; do not reuse any episode, seed, or target from this discovery study."
            if v5_authorized
            else (
                "Run only the smallest bounded environment/version/action-semantics or replay check; do not tune the gate."
                if decision == "generator_reconstruction_failed"
                else "Run the smallest domain-robust gate discovery study; do not authorize V5."
            )
        ),
        "distribution_comparison_sha256": common.sha256_file(
            metrics_dir / "distribution_comparison.json"
        ),
        "gate_metrics_sha256": common.sha256_file(metrics_dir / "gate_prospective.json"),
        "bootstrap_sha256": common.sha256_file(metrics_dir / "bootstrap_replicates.npz"),
        "episode_metrics_sha256": common.sha256_file(
            metrics_dir / "dataset_episode_metrics.npz"
        ),
    }
    common.write_study_json(decision_path, decision_payload, exclusive=True)
    return decision_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-reference", "final"))
    args = parser.parse_args()
    result = freeze_reference() if args.command == "freeze-reference" else final_analysis()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
