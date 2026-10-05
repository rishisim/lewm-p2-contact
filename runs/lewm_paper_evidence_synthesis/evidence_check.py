#!/usr/bin/env python3
"""Read-only recomputation of the paper's headline evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]


def load_json(relative: str):
    return json.loads((REPO / relative).read_text())


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (REPO / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(observed, expected, atol=1e-12):
    if not np.allclose(observed, expected, rtol=1e-11, atol=atol):
        raise AssertionError(f"{observed!r} != {expected!r}")


def episode_means(values: np.ndarray, identifiers: np.ndarray) -> np.ndarray:
    return np.asarray(
        [values[identifiers == item].mean() for item in np.unique(identifiers)],
        dtype=np.float64,
    )


def losses(exits: np.ndarray, target: np.ndarray, whitening=None) -> np.ndarray:
    delta = exits.astype(np.float64) - target[:, None, :].astype(np.float64)
    if whitening is not None:
        with np.errstate(all="ignore"):
            delta = np.matmul(delta, whitening)
    return np.square(delta).mean(axis=2)


def strongest_mixture(
    per_transition: np.ndarray, mean_depth: float, identifiers: np.ndarray
):
    candidates = []
    for lower in range(1, 5):
        for upper in range(lower, 5):
            if not lower <= mean_depth <= upper:
                continue
            weight = 0.0 if lower == upper else (mean_depth - lower) / (upper - lower)
            vector = (1.0 - weight) * per_transition[:, lower - 1] + weight * per_transition[:, upper - 1]
            candidates.append(
                (episode_means(vector, identifiers).mean(), lower, upper, weight, vector)
            )
    return min(candidates, key=lambda item: item[:3])


def fit_whitening(target: np.ndarray, floor_ratio: float) -> np.ndarray:
    centered = target.astype(np.float64) - target.astype(np.float64).mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    floor = max(float(eigenvalues.max()) * floor_ratio, 1e-10)
    with np.errstate(all="ignore"):
        return (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T


def cube_confirmation():
    metrics_path = "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/v5_confirmation_episode_metrics.npz"
    reps_path = "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/bootstrap_replicates.npz"
    decision = load_json("runs/lewm_v5_readiness_program/v5_package_versions/v004/decision.json")
    analysis = load_json("runs/lewm_v5_readiness_program/v5_package_versions/v004/analysis_result.json")
    ledger = decision["compute"]
    with np.load(REPO / metrics_path, allow_pickle=False) as metrics, np.load(
        REPO / reps_path, allow_pickle=False
    ) as reps:
        result = {"episodes": int(len(metrics["episode_id"])), "endpoints": {}}
        for name in ("raw_vs_analytic", "native_whitened_vs_analytic"):
            estimate = float(metrics[name].mean())
            lower = float(np.quantile(reps[name], 0.025))
            upper = float(np.quantile(reps[name], 0.975))
            terminal = decision["simultaneous_co_primary"][name]
            close(estimate, terminal["estimate"])
            close(lower, terminal["lower"])
            result["endpoints"][name] = {
                "estimate": estimate,
                "simultaneous_one_sided_lower": lower,
                "individual_95_upper": upper,
            }
    close(
        analysis["compute"]["raw_analytic_mixture"]["mean_loss"]
        - analysis["adaptive_raw_mse"],
        result["endpoints"]["raw_vs_analytic"]["estimate"],
    )
    close(
        analysis["compute"]["native_whitened_analytic_mixture"]["mean_loss"]
        - analysis["adaptive_planoracle_native_whitened_mse"],
        result["endpoints"]["native_whitened_vs_analytic"]["estimate"],
    )
    recomputed_total = (
        ledger["base_flops"]
        + ledger["mandatory_depth1_flops"]
        + ledger["additional_refiner_flops"]
        + ledger["gate_total_flops"]
    )
    assert recomputed_total == ledger["adaptive_total_flops"]
    assert sum(ledger["call_histogram"]) == ledger["rows"]
    assert sum((i + 1) * n for i, n in enumerate(ledger["call_histogram"])) == ledger["refiner_model_calls"]
    result["compute"] = {
        "rows": ledger["rows"],
        "adaptive_total_flops": recomputed_total,
        "call_histogram_depth_1_to_4": ledger["call_histogram"],
        "analytic_equivalent_mean_depth": ledger["analytic_equivalent_mean_calls"],
        "analytic_comparator_kind": "expectation-level linear fixed-depth mixture",
        "seeded_runnable_comparator_total_flops": ledger["seeded_mixture"]["total_flops"],
    }
    return result


def cube_shifts():
    root = "runs/lewm_v5_generalization/attempts/v005"
    decision = load_json(f"{root}/decision.json")
    analysis = load_json(f"{root}/analysis_result.json")
    labels = decision["regime_labels"]
    output = {"episodes_total": 0, "regimes": {}}
    for regime in ("markov_oracle", "plan_action_noise_0p2", "plan_random_action_0p1"):
        metric_path = f"{root}/metrics/{regime}_episode_metrics.npz"
        with np.load(REPO / metric_path, allow_pickle=False) as metrics, np.load(
            REPO / f"{root}/metrics/bootstrap_replicates.npz", allow_pickle=False
        ) as reps:
            endpoints = {}
            for name in ("raw_vs_analytic", "fixed_whitened_vs_analytic"):
                estimate = float(metrics[name].mean())
                lower = float(np.quantile(reps[f"{regime}__{name}"], 0.05 / 6.0))
                terminal = decision["simultaneous_co_primary"][regime][name]
                close(estimate, terminal["estimate"])
                close(lower, terminal["lower"])
                endpoints[name] = {"estimate": estimate, "simultaneous_one_sided_lower": lower}
            count = int(len(metrics["episode"]))
            output["episodes_total"] += count
            output["regimes"][regime] = {
                "episodes": count,
                "label": labels[regime],
                "adaptive_raw_mse": float(metrics["adaptive_raw"].mean()),
                "adaptive_fixed_whitened_mse": float(metrics["adaptive_fixed_whitened"].mean()),
                "endpoints": endpoints,
            }
            compute = analysis["regimes"][regime]["compute"]
            recomputed_total = (
                compute["base_flops"]
                + compute["mandatory_depth1_flops"]
                + compute["additional_refiner_flops"]
                + compute["gate_total_flops"]
            )
            assert recomputed_total == compute["adaptive_total_flops"]
            assert sum(compute["call_histogram"]) == compute["rows"]
            output["regimes"][regime]["adaptive_total_counted_flops"] = recomputed_total
    return output


def pusht_pilot():
    root = "runs/lewm_pusht_replication_pilot"
    decision = load_json(f"{root}/PILOT_DECISION.json")
    config = load_json(f"{root}/CONFIG.json")
    with np.load(REPO / f"{root}/EVALUATION_ARRAYS.npz", allow_pickle=False) as arrays, np.load(
        REPO / f"{root}/FIT_ARTIFACTS.npz", allow_pickle=False
    ) as fit, np.load(REPO / f"{root}/FROZEN_GATE.npz", allow_pickle=False) as gate:
        raw = losses(arrays["dense_exits"], arrays["target"])
        white = losses(arrays["dense_exits"], arrays["target"], fit["whitening_matrix"])
        ids, calls = arrays["episode_id"], arrays["calls"]
        reconstructed_calls = np.ones(len(calls), dtype=np.int64)
        active = np.ones(len(calls), dtype=bool)
        reached = []
        for stage in range(3):
            reached.append(active.copy())
            active &= arrays["scores"][:, stage] > gate["thresholds"][stage]
            reconstructed_calls += active
        assert np.array_equal(reconstructed_calls, calls)
        total = decision["compute"]["adaptive_total_counted_flops"]
        base = decision["compute"]["base_predictor_flops"] + decision["compute"]["action_normalization_flops"]
        refiner_price = config["compute"]["refiner"]["total_flops_per_call"]
        mean_depth = (total - base) / (refiner_price * len(calls))
        raw_mix = strongest_mixture(raw, mean_depth, ids)
        white_mix = strongest_mixture(white, mean_depth, ids)
        selected_raw = raw[np.arange(len(calls)), calls - 1]
        selected_white = white[np.arange(len(calls)), calls - 1]
        raw_ep = episode_means(raw_mix[4] - selected_raw, ids)
        white_ep = episode_means(white_mix[4] - selected_white, ids)
        close(raw_ep, arrays["episode_raw_vs_primary_analytic"])
        close(white_ep, arrays["episode_fit_whitened_vs_primary_analytic"])
        rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
        indices = rng.integers(0, len(raw_ep), size=(10_000, len(raw_ep)), dtype=np.int32)
        raw_boot = raw_ep[indices].mean(axis=1)
        rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
        indices = rng.integers(0, len(white_ep), size=(10_000, len(white_ep)), dtype=np.int32)
        white_boot = white_ep[indices].mean(axis=1)
        close(raw_ep.mean(), decision["primary"]["raw"]["mean_benefit"])
        close(white_ep.mean(), decision["primary"]["fit_whitened"]["mean_benefit"])
        close(selected_raw.mean(), decision["adaptive_raw_mse"])
        close(selected_white.mean(), decision["adaptive_fit_whitened_mse"])
        close(raw_mix[0], decision["compute"]["raw_primary_analytic_mixture"]["mean_loss"])
        close(white_mix[0], decision["compute"]["fit_whitened_primary_analytic_mixture"]["mean_loss"])
        close(np.quantile(raw_boot, [0.025, 0.975]), [decision["primary"]["raw"]["ci_low"], decision["primary"]["raw"]["ci_high"]])
        close(np.quantile(white_boot, [0.025, 0.975]), [decision["primary"]["fit_whitened"]["ci_low"], decision["primary"]["fit_whitened"]["ci_high"]])
        histogram = np.bincount(calls, minlength=5)[1:].tolist()
        stagewise = []
        for stage in range(3):
            raw_gain = raw[:, stage] - raw[:, stage + 1]
            white_gain = white[:, stage] - white[:, stage + 1]
            combined = 0.5 * (
                raw_gain / float(gate["raw_gain_scale"][stage])
                + white_gain / float(gate["white_gain_scale"][stage])
            )
            rho = float(spearmanr(arrays["scores"][reached[stage], stage], combined[reached[stage]]).statistic)
            close(rho, decision["stagewise_rank"][stage]["score_vs_combined_gain_spearman"])
            stagewise.append(rho)
    comp = decision["compute"]
    recomputed_total = comp["base_predictor_flops"] + comp["action_normalization_flops"] + comp["refiner_flops"] + comp["gate_total_flops"]
    assert recomputed_total == comp["adaptive_total_counted_flops"] == comp["primary_analytic_comparator_total_counted_flops"]
    return {
        "episodes": len(raw_ep),
        "scientific_label": "pusht_replication_pilot_not_supported",
        "raw_benefit": float(raw_ep.mean()),
        "raw_95_interval": np.quantile(raw_boot, [0.025, 0.975]).tolist(),
        "fit_whitened_benefit": float(white_ep.mean()),
        "fit_whitened_95_interval": np.quantile(white_boot, [0.025, 0.975]).tolist(),
        "total_counted_flops_each": recomputed_total,
        "call_histogram_depth_1_to_4": histogram,
        "mean_depth_for_exact_compute_comparator": mean_depth,
        "stagewise_score_combined_gain_spearman": stagewise,
    }


def pusht_binary():
    root = "runs/lewm_pusht_binary_confirmation"
    decision = load_json(f"{root}/DECISION.json")
    config = load_json(f"{root}/CONFIG.json")
    with np.load(REPO / f"{root}/EVALUATION_ARRAYS.npz", allow_pickle=False) as arrays, np.load(
        REPO / "runs/lewm_pusht_replication_pilot/FIT_ARTIFACTS.npz", allow_pickle=False
    ) as fit, np.load(
        REPO / "runs/lewm_pusht_replication_pilot/FROZEN_GATE.npz", allow_pickle=False
    ) as gate:
        raw = losses(arrays["dense_exits"], arrays["target"])
        white = losses(arrays["dense_exits"], arrays["target"], fit["whitening_matrix"])
        ids, calls = arrays["episode_id"], arrays["calls"]
        reconstructed_calls = 1 + (
            arrays["stage1_score"] > config["protocol"]["policy"]["stage_1_threshold"]
        ).astype(np.int64)
        assert np.array_equal(reconstructed_calls, calls)
        mean_depth = decision["compute"]["analytic_equivalent_mean_depth"]
        raw_mix = strongest_mixture(raw, mean_depth, ids)
        white_mix = strongest_mixture(white, mean_depth, ids)
        adaptive_raw = raw[np.arange(len(calls)), calls - 1]
        adaptive_white = white[np.arange(len(calls)), calls - 1]
        raw_ep = episode_means(raw_mix[4] - adaptive_raw, ids)
        white_ep = episode_means(white_mix[4] - adaptive_white, ids)
        close(raw_ep, arrays["episode_raw_vs_exact_compute_analytic"])
        close(white_ep, arrays["episode_fit_whitened_vs_exact_compute_analytic"])
        close(
            episode_means(adaptive_raw, ids),
            arrays["episode_loss_adaptive_raw"],
        )
        close(
            episode_means(adaptive_white, ids),
            arrays["episode_loss_adaptive_fit_whitened"],
        )
        close(raw_mix[0], decision["compute"]["raw_strongest_pairwise_mixture"]["mean_episode_loss"])
        close(white_mix[0], decision["compute"]["fit_whitened_strongest_pairwise_mixture"]["mean_episode_loss"])
        rng = np.random.default_rng(config["protocol"]["analysis"]["bootstrap_seed"])
        indices = rng.integers(0, len(raw_ep), size=(20_000, len(raw_ep)), dtype=np.int32)
        raw_boot, white_boot = raw_ep[indices].mean(axis=1), white_ep[indices].mean(axis=1)
        combined_gain = 0.5 * (
            (raw[:, 0] - raw[:, 1]) / float(gate["raw_gain_scale"][0])
            + (white[:, 0] - white[:, 1]) / float(gate["white_gain_scale"][0])
        )
        rho = float(spearmanr(arrays["stage1_score"], combined_gain).statistic)
        sensitivity = {}
        for ratio in (1e-3, 3e-4, 1e-4):
            alternative = losses(arrays["dense_exits"], arrays["target"], fit_whitening(arrays["target"], ratio))
            alternative_mix = strongest_mixture(alternative, mean_depth, ids)
            alternative_adaptive = alternative[np.arange(len(calls)), calls - 1]
            sensitivity[f"{ratio:.0e}"] = float(episode_means(alternative_mix[4] - alternative_adaptive, ids).mean())
    comp = decision["compute"]
    recomputed_total = comp["base_predictor_flops"] + comp["action_normalization_flops"] + comp["refiner_flops"] + comp["gate_total_flops"]
    assert recomputed_total == comp["adaptive_total_counted_flops"] == comp["analytic_comparator_total_counted_flops"]
    close(raw_ep.mean(), decision["co_primary"]["raw"]["mean"])
    close(white_ep.mean(), decision["co_primary"]["fit_whitened"]["mean"])
    close(np.quantile(raw_boot, 0.025), decision["co_primary"]["simultaneous_one_sided_lower_bounds"]["raw_lower"])
    close(np.quantile(white_boot, 0.025), decision["co_primary"]["simultaneous_one_sided_lower_bounds"]["fit_whitened_lower"])
    close(rho, decision["stage1_score_gain_rank"]["combined_score_gain_spearman"])
    return {
        "episodes": len(raw_ep),
        "transitions": len(calls),
        "scientific_label": decision["scientific_label"],
        "raw_benefit": float(raw_ep.mean()),
        "raw_simultaneous_one_sided_lower": float(np.quantile(raw_boot, 0.025)),
        "fit_whitened_benefit": float(white_ep.mean()),
        "fit_whitened_simultaneous_one_sided_lower": float(np.quantile(white_boot, 0.025)),
        "total_counted_flops_each": recomputed_total,
        "call_histogram_depth_1_to_2": np.bincount(calls, minlength=3)[1:].tolist(),
        "stage1_score_combined_gain_spearman": rho,
        "posthoc_confirmation_target_covariance_floor_sensitivity": sensitivity,
    }


def latency_and_population_checks():
    cube = load_json(
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/v5_confirmation_latency.json"
    )
    cube_medians = {}
    for item in cube["timings"]:
        observed = float(np.median(np.asarray(item["all_seconds"], dtype=np.float64)))
        close(observed, item["median_seconds"])
        cube_medians[f"{item['path']}__batch_{item['batch_size']}"] = observed

    pilot = load_json("runs/lewm_pusht_replication_pilot/LATENCY.json")
    pilot_medians = {}
    for key in (
        "base_predictor_all_rows",
        "cached_base_to_adaptive_selected",
        "fixed_depth_1",
        "fixed_depth_4",
    ):
        observed = float(np.median(np.asarray(pilot[key]["seconds"], dtype=np.float64)))
        close(observed, pilot[key]["median_seconds"])
        pilot_medians[key] = observed

    binary_decision = load_json("runs/lewm_pusht_binary_confirmation/DECISION.json")
    binary = binary_decision["synchronized_latency"]
    binary_medians = {}
    for key in (
        "base_predictor_all_transitions",
        "cached_base_to_binary_adaptive_selected",
        "fixed_depth_1_cached_base",
        "fixed_depth_2_cached_base",
    ):
        observed = float(np.median(np.asarray(binary[key]["seconds"], dtype=np.float64)))
        close(observed, binary[key]["median_seconds"])
        binary_medians[key] = observed

    pilot_context = load_json("runs/lewm_pusht_replication_pilot/DESCRIPTIVE_CONTEXT.json")
    pilot_success = {
        role: float(values["success_fraction"])
        for role, values in pilot_context["roles"].items()
    }
    assert pilot_success == {"evaluation": 0.0, "fit": 0.0, "selection": 0.0}
    binary_config = load_json("runs/lewm_pusht_binary_confirmation/CONFIG.json")
    binary_success = float(
        binary_config["fixed_cohort"]["privileged_context_summary"]["success_fraction"]
    )
    assert binary_success == 0.0
    return {
        "cube_median_seconds": cube_medians,
        "pusht_pilot_median_seconds": pilot_medians,
        "pusht_binary_median_seconds": binary_medians,
        "pusht_pilot_success_fraction_by_role": pilot_success,
        "pusht_binary_success_fraction": binary_success,
    }


def planning_evidence():
    with np.load(REPO / "runs/lewm_frozen_gate_planning_bridge/phase_b_metrics.npz", allow_pickle=False) as arrays:
        names = arrays["condition"].tolist()
        adaptive, matched = names.index("adaptive"), names.index("matched")
        k5_raw = arrays["raw_mse"][adaptive, 4].mean() - arrays["raw_mse"][matched, 4].mean()
        k5_white = arrays["whitened_mse"][adaptive, 4].mean() - arrays["whitened_mse"][matched, 4].mean()
        cases, episodes = int(arrays["raw_mse"].shape[2]), int(len(arrays["episode_id"]))
    with np.load(REPO / "runs/lewm_planning_bridge_decomposition/decomposition_metrics.npz", allow_pickle=False) as arrays:
        raw, white = arrays["adaptive_minus_matched_start_effect_raw"], arrays["adaptive_minus_matched_start_effect_whitened"]
        rng = np.random.default_rng(27182818)
        indices = rng.integers(0, 20, size=(20_000, 20))
        raw_interval = np.quantile(raw[indices].mean(axis=1), [0.025, 0.975])
        white_interval = np.quantile(white[indices].mean(axis=1), [0.025, 0.975])
        informative = int(arrays["informative_start"].sum())
        defined = int(np.isfinite(arrays["actual_ceiling_spearman"][0]).sum())
        raw_ceiling = float(np.nanmean(arrays["actual_ceiling_spearman"][0]))
    return {
        "five_step_consumed_cube": {
            "episodes": episodes,
            "overlapping_starts": cases,
            "k5_adaptive_minus_matched_raw_mse": float(k5_raw),
            "k5_adaptive_minus_matched_whitened_mse": float(k5_white),
        },
        "planning_decomposition": {
            "independent_starts": 20,
            "adaptive_minus_matched_raw_mse": float(raw.mean()),
            "raw_95_cluster_interval": raw_interval.tolist(),
            "adaptive_minus_matched_whitened_mse": float(white.mean()),
            "whitened_95_cluster_interval": white_interval.tolist(),
            "informative_starts": informative,
            "rank_defined_starts": defined,
            "actual_latent_cost_vs_physical_error_raw_mean_spearman": raw_ceiling,
            "downstream_planning_supported": False,
        },
    }


def main():
    source_paths = [
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/PACKAGE_READY.json",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/DGP.json",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/analysis_result.json",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/decision.json",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/v5_confirmation_episode_metrics.npz",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/bootstrap_replicates.npz",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/compute_ledger_realized.json",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/stagewise_ranking.json",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/v5_confirmation_latency.json",
        "runs/lewm_v5_readiness_program/v5_package_versions/v004/audit/independent_verification.json",
        "runs/lewm_v5_generalization/attempts/v005/DGP_MATRIX.json",
        "runs/lewm_v5_generalization/attempts/v005/analysis_result.json",
        "runs/lewm_v5_generalization/attempts/v005/decision.json",
        "runs/lewm_v5_generalization/attempts/v005/metrics/markov_oracle_episode_metrics.npz",
        "runs/lewm_v5_generalization/attempts/v005/metrics/plan_action_noise_0p2_episode_metrics.npz",
        "runs/lewm_v5_generalization/attempts/v005/metrics/plan_random_action_0p1_episode_metrics.npz",
        "runs/lewm_v5_generalization/attempts/v005/metrics/bootstrap_replicates.npz",
        "runs/lewm_v5_generalization/attempts/v005/audit/independent_verification.json",
        "runs/lewm_frozen_gate_planning_bridge/RESULTS.json",
        "runs/lewm_frozen_gate_planning_bridge/phase_b_metrics.npz",
        "runs/lewm_frozen_gate_planning_bridge/phase_c_metrics.npz",
        "runs/lewm_planning_bridge_decomposition/RESULTS.json",
        "runs/lewm_planning_bridge_decomposition/decomposition_metrics.npz",
        "runs/lewm_pusht_replication_pilot/CONFIG.json",
        "runs/lewm_pusht_replication_pilot/PILOT_DECISION.json",
        "runs/lewm_pusht_replication_pilot/EVALUATION_ARRAYS.npz",
        "runs/lewm_pusht_replication_pilot/DESCRIPTIVE_CONTEXT.json",
        "runs/lewm_pusht_replication_pilot/FIT_ARTIFACTS.npz",
        "runs/lewm_pusht_replication_pilot/FROZEN_GATE.npz",
        "runs/lewm_pusht_replication_pilot/INDEPENDENT_VERIFICATION.json",
        "runs/lewm_pusht_replication_pilot/LATENCY.json",
        "runs/lewm_pusht_binary_confirmation/CONFIG.json",
        "runs/lewm_pusht_binary_confirmation/DECISION.json",
        "runs/lewm_pusht_binary_confirmation/EVALUATION_ARRAYS.npz",
        "runs/lewm_pusht_binary_confirmation/INDEPENDENT_CHECK.json",
    ]
    chronology_ns = {
        "cube_package_ready": load_json("runs/lewm_v5_readiness_program/v5_package_versions/v004/PACKAGE_READY.json")["created_unix_ns"],
        "cube_confirmation_decision": load_json("runs/lewm_v5_readiness_program/v5_package_versions/v004/decision.json")["created_unix_ns"],
        "pusht_pilot_decision": load_json("runs/lewm_pusht_replication_pilot/PILOT_DECISION.json")["created_unix_ns"],
        "pusht_binary_protocol_lock": load_json("runs/lewm_pusht_binary_confirmation/CONFIG.json")["protocol_locked_unix_ns"],
        "pusht_binary_decision": load_json("runs/lewm_pusht_binary_confirmation/DECISION.json")["scientific_decision_locked_unix_ns"],
    }
    output = {
        "all_checks_passed": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cube_confirmation": cube_confirmation(),
        "cube_zero_shot_shifts": cube_shifts(),
        "pusht_four_depth_pilot": pusht_pilot(),
        "pusht_binary_confirmation": pusht_binary(),
        "multistep_and_planning": planning_evidence(),
        "latency_and_population_checks": latency_and_population_checks(),
        "chronology_utc": {
            key: datetime.fromtimestamp(value / 1e9, tz=timezone.utc).isoformat()
            for key, value in chronology_ns.items()
        },
        "source_sha256": {path: sha256(path) for path in source_paths},
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
