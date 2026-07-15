#!/usr/bin/env python3
"""Conservative co-primary V5 power and heterogeneity sensitivity analysis."""

from __future__ import annotations

import json
import math
import time

import numpy as np
from scipy.stats import norm

from cycle_common import ROOT, SOURCE_CYCLE, V5_SAMPLE_SIZE, atomic_json, read_json, sha256_file


ENDPOINTS = ("raw_vs_analytic", "native_whitened_vs_analytic")
ALPHA_PER_ENDPOINT = 0.025
EFFECT_SHRINK = 0.80
PRIMARY_DESIGN_EFFECT = 1.50
STRESS_DESIGN_EFFECT = 2.00
PRIMARY_JOINT_POWER_TARGET = 0.95
STRESS_JOINT_POWER_TARGET = 0.90


def endpoint_power(n: int, effect: float, standard_deviation: float, design_effect: float) -> float:
    critical = float(norm.ppf(1.0 - ALPHA_PER_ENDPOINT))
    noncentral = effect * math.sqrt(n) / (standard_deviation * math.sqrt(design_effect))
    return float(norm.cdf(noncentral - critical))


def joint_union_lower(powers: dict[str, float]) -> float:
    return float(max(0.0, sum(powers.values()) - (len(powers) - 1)))


def powers_at(
    n: int,
    effects: dict[str, float],
    standard_deviations: dict[str, float],
    design_effect: float,
) -> dict[str, object]:
    marginal = {
        name: endpoint_power(n, effects[name], standard_deviations[name], design_effect)
        for name in ENDPOINTS
    }
    return {
        "sample_size": n,
        "design_effect": design_effect,
        "marginal_power": marginal,
        "joint_power_lower_by_union_bound": joint_union_lower(marginal),
    }


def main() -> None:
    output = ROOT / "power_analysis.json"
    if output.exists():
        raise RuntimeError("power analysis is immutable and already exists")
    decision_path = SOURCE_CYCLE / "decision.json"
    metrics_path = SOURCE_CYCLE / "metrics/prospective_episode_metrics.npz"
    decision = read_json(decision_path)
    if decision["terminal_outcome"] != "cycle_prospective_discovery_passed":
        raise RuntimeError("power source is not the accepted discovery pass")
    with np.load(metrics_path, allow_pickle=False) as stored:
        values = {name: stored[name].astype(np.float64) for name in ENDPOINTS}
    if any(len(array) != 300 or not np.isfinite(array).all() for array in values.values()):
        raise RuntimeError("discovery episode metrics are incomplete or nonfinite")

    standard_deviations = {
        name: float(np.std(array, ddof=1)) for name, array in values.items()
    }
    block_records = {}
    observed_design_effects = {}
    conservative_effects = {}
    for name, array in values.items():
        blocks = [part for part in np.array_split(array, 6)]
        block_means = np.asarray([part.mean() for part in blocks], dtype=np.float64)
        observed_de = float(
            np.var(block_means, ddof=1) / (np.var(array, ddof=1) / len(blocks[0]))
        )
        simultaneous_lower = float(decision["simultaneous_co_primary"][name]["lower"])
        minimum_block_mean = float(block_means.min())
        conservative_effect = EFFECT_SHRINK * min(simultaneous_lower, minimum_block_mean)
        if conservative_effect <= 0:
            raise RuntimeError("no positive conservative effect is available")
        block_records[name] = {
            "contiguous_block_size": 50,
            "block_means": block_means.tolist(),
            "minimum_block_mean": minimum_block_mean,
            "observed_block_design_effect": observed_de,
        }
        observed_design_effects[name] = observed_de
        conservative_effects[name] = conservative_effect

    selected = None
    search = []
    for n in range(300, 5001, 50):
        primary = powers_at(n, conservative_effects, standard_deviations, PRIMARY_DESIGN_EFFECT)
        stress = powers_at(n, conservative_effects, standard_deviations, STRESS_DESIGN_EFFECT)
        eligible = (
            primary["joint_power_lower_by_union_bound"] >= PRIMARY_JOINT_POWER_TARGET
            and stress["joint_power_lower_by_union_bound"] >= STRESS_JOINT_POWER_TARGET
        )
        search.append(
            {
                "sample_size": n,
                "primary_joint_power_lower": primary["joint_power_lower_by_union_bound"],
                "stress_joint_power_lower": stress["joint_power_lower_by_union_bound"],
                "eligible": eligible,
            }
        )
        if eligible and selected is None:
            selected = n
    if selected != V5_SAMPLE_SIZE:
        raise RuntimeError(f"fixed V5 sample size does not equal minimum eligible grid size: {selected}")

    primary = powers_at(selected, conservative_effects, standard_deviations, PRIMARY_DESIGN_EFFECT)
    stress = powers_at(selected, conservative_effects, standard_deviations, STRESS_DESIGN_EFFECT)
    sensitivity = []
    base_limits = {
        name: min(
            float(decision["simultaneous_co_primary"][name]["lower"]),
            float(block_records[name]["minimum_block_mean"]),
        )
        for name in ENDPOINTS
    }
    for shrink in (1.0, 0.8, 0.6):
        effects = {name: shrink * base_limits[name] for name in ENDPOINTS}
        for design_effect in (1.0, max(observed_design_effects.values()), 1.5, 2.0, 2.5):
            item = powers_at(selected, effects, standard_deviations, float(design_effect))
            sensitivity.append({"effect_fraction": shrink, **item})

    discovery_compute = decision["compute"]
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "source_cycle": str(SOURCE_CYCLE),
        "source_decision_sha256": sha256_file(decision_path),
        "source_episode_metrics_sha256": sha256_file(metrics_path),
        "source_episode_count": 300,
        "method": "one-sided normal paired-mean power with Bonferroni alpha .025 per fixed co-primary endpoint; joint power lower bounded by the union bound",
        "alpha_per_endpoint": ALPHA_PER_ENDPOINT,
        "effect_definition": "80 percent of the smaller of the independently reproduced simultaneous discovery lower bound and the minimum contiguous 50-episode block mean",
        "effect_shrink": EFFECT_SHRINK,
        "standard_deviations": standard_deviations,
        "block_heterogeneity": block_records,
        "maximum_observed_block_design_effect": max(observed_design_effects.values()),
        "conservative_effects": conservative_effects,
        "primary_design_effect": PRIMARY_DESIGN_EFFECT,
        "stress_design_effect": STRESS_DESIGN_EFFECT,
        "primary_joint_power_target": PRIMARY_JOINT_POWER_TARGET,
        "stress_joint_power_target": STRESS_JOINT_POWER_TARGET,
        "grid_step": 50,
        "minimum_eligible_sample_size": selected,
        "selected_v5_sample_size": V5_SAMPLE_SIZE,
        "primary_power": primary,
        "stress_power": stress,
        "sensitivity": sensitivity,
        "search_boundary": [item for item in search if selected - 100 <= item["sample_size"] <= selected + 100],
        "exact_compute_accounting": {
            "discovery_effect_already_compares_against_exact_total_compute_including_7985_gate_flops_per_reached_decision": True,
            "discovery_total_flops": discovery_compute["adaptive_total_flops"],
            "discovery_gate_total_flops": discovery_compute["gate_total_flops"],
            "discovery_nonflop_operations": discovery_compute["nonflop_comparison_min_operations"],
            "planning_scale_factor_v5_over_discovery": V5_SAMPLE_SIZE / 300.0,
            "latency_is_reported_separately_and_not_substituted_for_flops": True,
        },
        "passed": bool(
            selected == V5_SAMPLE_SIZE
            and primary["joint_power_lower_by_union_bound"] >= PRIMARY_JOINT_POWER_TARGET
            and stress["joint_power_lower_by_union_bound"] >= STRESS_JOINT_POWER_TARGET
        ),
        "remaining_uncertainty": "Power depends on future DGP stability, paired-episode variance, the conservative effect floor, and design-effect adequacy; no calculation guarantees a V5 pass.",
        "v5_outcome_episodes": 0,
    }
    if not result["passed"]:
        raise RuntimeError("power target was not met")
    atomic_json(output, result, exclusive=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
