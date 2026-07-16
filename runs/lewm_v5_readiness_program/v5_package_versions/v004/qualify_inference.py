#!/usr/bin/env python3
"""Preseal synthetic qualification of fixed V5 inference and terminal mapping."""

from __future__ import annotations

import json

import numpy as np

import analysis
import independent_verify
from cycle_common import (
    ROOT,
    V5_SAMPLE_SIZE,
    assert_runtime_contract,
    atomic_json,
    read_json,
)


def main() -> None:
    assert_runtime_contract("evaluation")
    output = ROOT / "audit/inference_qualification.json"
    if output.exists():
        raise RuntimeError("inference qualification is immutable and already exists")
    ledger = read_json(ROOT / "cohort_seed_ledger.json")
    mapping = read_json(ROOT / "outcome_mapping.json")
    rng = np.random.default_rng(ledger["inference_qualification_seed"])
    episode_ids = np.repeat(np.arange(V5_SAMPLE_SIZE, dtype=np.int32), 38)
    row_values = rng.normal(size=V5_SAMPLE_SIZE * 38)
    vectorized = analysis.episode_mean(row_values, episode_ids)
    looped = independent_verify.per_episode(row_values, episode_ids)
    losses = np.square(rng.normal(size=(257, 4)))
    mixture = analysis.strongest_mixture(losses, 1.375)
    independent, left, right, weight = independent_verify.optimal_linear_baseline(
        losses, 1.375
    )
    mixture_difference = np.abs(mixture["loss"] - independent)
    mixture_scale = max(
        1.0,
        float(np.max(np.abs(mixture["loss"]))),
        float(np.max(np.abs(independent))),
    )
    mixture_machine_precision_ceiling = float(
        16.0 * np.finfo(np.float64).eps * mixture_scale
    )
    sample = rng.normal(loc=0.25, scale=1.0, size=20001)
    independent_quantile = independent_verify.independent_linear_quantile(sample, 0.025)
    numpy_quantile = float(np.quantile(sample, 0.025))
    checks = {
        "sample_size_fixed_1600": V5_SAMPLE_SIZE == 1600,
        "bootstrap_replicates_fixed_20000": analysis.BOOTSTRAP_REPLICATES == 20_000,
        "analysis_seeds_match_frozen_ledger": (
            analysis.BOOTSTRAP_SEED == ledger["bootstrap_seed"]
            and analysis.HISTOGRAM_SEED == ledger["histogram_seed"]
            and analysis.SEEDED_MIXTURE_SEED == ledger["seeded_mixture_seed"]
        ),
        "episode_aggregation_two_implementations_exact": bool(
            np.array_equal(vectorized, looped)
        ),
        "mixture_two_implementations_machine_precision_equivalent": bool(
            mixture["depth_lower"] == left
            and mixture["depth_upper"] == right
            and mixture["weight_upper"] == weight
            and float(mixture_difference.max()) <= mixture_machine_precision_ceiling
        ),
        "linear_quantile_independent_implementation_exact": independent_quantile
        == numpy_quantile,
        "six_individual_criteria_frozen": len(
            mapping["individual_positive_lower_bound_criteria"]
        )
        == 6,
        "two_co_primary_criteria_frozen": len(
            mapping["simultaneous_co_primary_positive_lower_bound_criteria"]
        )
        == 2,
        "no_retry_frozen": mapping["no_retry"] is True,
        "terminal_labels_frozen": {
            mapping["integrity_failure_outcome"],
            mapping["process_valid_statistical_failure_outcome"],
            mapping["process_valid_statistical_pass_outcome"],
        }
        == {"v5_execution_invalid", "v5_confirmation_failed", "v5_confirmation_passed"},
    }
    result = {
        "schema_version": 1,
        "synthetic_inputs_only": True,
        "checks": checks,
        "passed": all(checks.values()),
        "mixture_qualification": {
            "depth_and_weight_must_be_exact": True,
            "maximum_absolute_difference": float(mixture_difference.max()),
            "mean_absolute_difference": float(mixture_difference.mean()),
            "float64_machine_precision_scale_multiplier": 16.0,
            "maximum_absolute_difference_ceiling": mixture_machine_precision_ceiling,
            "analysis_and_independent_implementations_unchanged": True,
        },
        "package_smoke_episodes": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"inference qualification failed: {result}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
