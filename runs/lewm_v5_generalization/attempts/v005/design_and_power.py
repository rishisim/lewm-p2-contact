#!/usr/bin/env python3
"""Freeze the complete bounded design, power rule, and fresh identifiers."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.stats import norm

from study_common import (
    ADAPTER_FLOPS,
    ATTEMPT_ROOT,
    ATTEMPT_VERSION,
    BASE_FLOPS,
    BOOTSTRAP_REPLICATES,
    CO_PRIMARY_ENDPOINTS,
    FAMILYWISE_ALPHA,
    FEATURE_DIM,
    GATE_FEATURE_FLOPS,
    GATE_HEAD_FLOPS,
    GATE_NONFLOP_OPS,
    GATE_TOTAL_FLOPS,
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    PER_CLAIM_ALPHA,
    REGIMES,
    REPLACEMENTS_PER_REGIME,
    REPO_ROOT,
    ROWS_PER_EPISODE,
    SIMULTANEOUS_FAMILY_SIZE,
    SMOKE_EPISODES_PER_REGIME,
    STUDY_ROOT,
    TARGET_EPISODES_PER_REGIME,
    TERMINAL_LABELS,
    V1_FLOPS,
    V5_ROOT,
    append_ledger,
    assert_runtime_contract,
    atomic_json,
    canonical_digest,
    complete_state,
    read_json,
    sha256_file,
)


SHORT = {
    "markov_oracle": "mk",
    "plan_action_noise_0p2": "n2",
    "plan_random_action_0p1": "ra",
}


def walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def safe_prior_identifiers() -> tuple[set[int], set[str], list[str]]:
    numbers: set[int] = set()
    strings: set[str] = set()
    scanned: list[str] = []
    runs = REPO_ROOT / "runs"
    forbidden_v3 = REPO_ROOT / "runs/lewm_adaptive_compute_v3"
    for path in sorted(runs.rglob("*.json")):
        if ATTEMPT_ROOT in path.parents or path == ATTEMPT_ROOT:
            continue
        if forbidden_v3 in path.parents or path == forbidden_v3:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for item in walk(value):
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                numbers.add(int(item))
            elif isinstance(item, str):
                strings.add(item)
        scanned.append(str(path))
    for path in runs.rglob("*"):
        if not path.is_file() or ATTEMPT_ROOT in path.parents:
            continue
        strings.update((path.name, path.stem))
    return numbers, strings, scanned


def records() -> tuple[dict[str, dict[str, list[dict[str, Any]]]], int]:
    roles: dict[str, dict[str, list[dict[str, Any]]]] = {}
    global_index = 0
    counts = {
        "smoke": SMOKE_EPISODES_PER_REGIME,
        "target": TARGET_EPISODES_PER_REGIME,
        "replacement": REPLACEMENTS_PER_REGIME,
    }
    for regime in REGIMES:
        roles[regime] = {}
        for role, count in counts.items():
            width = 4
            items = []
            for slot in range(count):
                item = {
                    "regime": regime,
                    "role": role,
                    "slot": slot,
                    "episode_id": (
                        f"g5v005-{SHORT[regime]}-{role}-{slot:0{width}d}"
                    ),
                    "env_seed": 3_460_000_000 + global_index,
                    "policy_seed": 3_470_000_000 + global_index,
                    "oracle_np_seed": 3_480_000_000 + global_index,
                    "action_space_seed": 3_490_000_000 + global_index,
                }
                if len(item["episode_id"]) > 32:
                    raise RuntimeError("episode identifier exceeds U32 storage")
                items.append(item)
                global_index += 1
            roles[regime][role] = items
    return roles, global_index


def power_analysis() -> dict[str, Any]:
    metrics_path = V5_ROOT / "metrics/v5_confirmation_episode_metrics.npz"
    if sha256_file(metrics_path) != (
        "4eb5cc464f2526a9ccd9ab0aaf84f3d789439120390dd95df0fef224d54e9391"
    ):
        # The metrics file is post-package and not one of the user-supplied
        # named hashes; its expected hash is independently sealed here.
        raise RuntimeError("consumed V5 episode-metrics hash drift")
    with np.load(metrics_path, allow_pickle=False) as stored:
        vectors = {
            "raw_vs_analytic": stored["raw_vs_analytic"].astype(np.float64),
            "fixed_whitened_vs_analytic": stored[
                "native_whitened_vs_analytic"
            ].astype(np.float64),
        }
    if any(len(value) != 1_600 for value in vectors.values()):
        raise RuntimeError("V5 power source is not exactly 1,600 episodes")
    source = {
        name: {
            "mean": float(values.mean()),
            "sd_ddof1": float(values.std(ddof=1)),
            "episode_count": len(values),
        }
        for name, values in vectors.items()
    }

    def marginal(n: int, retained_fraction: float, name: str) -> float:
        item = source[name]
        noncentral = (
            math.sqrt(n)
            * retained_fraction
            * item["mean"]
            / item["sd_ddof1"]
        )
        return float(norm.cdf(noncentral - norm.ppf(1.0 - PER_CLAIM_ALPHA)))

    def family_lower(n: int, retained_fraction: float) -> float:
        failure_sum = 0.0
        for name in CO_PRIMARY_ENDPOINTS:
            failure_sum += len(REGIMES) * (
                1.0 - marginal(n, retained_fraction, name)
            )
        return max(0.0, 1.0 - failure_sum)

    grid = []
    for n in range(500, 5_001, 500):
        grid.append(
            {
                "episodes_per_regime": n,
                "retained_effect_fraction": 0.35,
                "marginal_power": {
                    name: marginal(n, 0.35, name)
                    for name in CO_PRIMARY_ENDPOINTS
                },
                "union_bound_family_power_lower": family_lower(n, 0.35),
                "eligible": family_lower(n, 0.35) >= 0.95,
            }
        )
    selected = next(item for item in grid if item["eligible"])
    sensitivity = []
    for retained in (0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25):
        sensitivity.append(
            {
                "retained_effect_fraction": retained,
                "effect_shrink_fraction": 1.0 - retained,
                "episodes_per_regime": TARGET_EPISODES_PER_REGIME,
                "marginal_power": {
                    name: marginal(TARGET_EPISODES_PER_REGIME, retained, name)
                    for name in CO_PRIMARY_ENDPOINTS
                },
                "union_bound_family_power_lower": family_lower(
                    TARGET_EPISODES_PER_REGIME, retained
                ),
            }
        )
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "source": {
            "path": str(metrics_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(metrics_path),
            "consumed_v5_only": True,
            "source_outcome_episodes": 1_600,
        },
        "source_endpoint_moments": source,
        "method": (
            "one-sided normal paired-mean power using V5 episode-level mean "
            "and SD, Bonferroni alpha .05/6, and a union-bound lower bound "
            "over three regimes times two co-primary endpoints"
        ),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "family_size": SIMULTANEOUS_FAMILY_SIZE,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "primary_retained_effect_fraction": 0.35,
        "primary_effect_shrink_fraction": 0.65,
        "primary_family_power_target": 0.95,
        "grid_step": 500,
        "grid": grid,
        "selected": selected,
        "selected_episodes_per_regime": TARGET_EPISODES_PER_REGIME,
        "selection_rule": (
            "smallest 500-episode grid point with union-bound family-power "
            "lower bound >= .95 when only 35% of the V5 effect remains"
        ),
        "sensitivity": sensitivity,
        "passed": (
            selected["episodes_per_regime"] == TARGET_EPISODES_PER_REGIME
            and selected["union_bound_family_power_lower"] >= 0.95
        ),
        "sequential_expansion": False,
        "remaining_uncertainty": (
            "Shifted-DGP effects and variances may differ arbitrarily; power "
            "does not guarantee support and cannot justify expansion."
        ),
        "generalization_target_outcome_episodes": 0,
    }
    return result


def build_design() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "attempt": ATTEMPT_VERSION,
        "study": "LeWM V5 zero-shot one-factor generalization",
        "claim_boundary": (
            "external-validity envelope of the fully frozen V5 policy only; "
            "no universal, downstream-control, contact-aware, wall-clock, "
            "or first-method claim"
        ),
        "v5_package": str(V5_ROOT.relative_to(REPO_ROOT)),
        "candidate_id": "stage_dual_r0.01_q0.85",
        "thresholds_float64": [
            0.19465729885363534,
            0.1568665868361991,
            0.15153321811129283,
        ],
        "frozen_native_whitening": (
            "unchanged V5 PlanOracle-native matrix; termed fixed-whitened "
            "under shifted DGPs; no recalibration"
        ),
        "environment_common": {
            "environment": "swm/OGBCube-v0",
            "num_envs": 1,
            "max_episode_steps": 200,
            "image_shape": [224, 224],
            "env_type": "single",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
            "terminate_at_goal": False,
            "mode": "data_collection",
            "frameskip": 5,
            "history": 3,
            "rows_per_episode": ROWS_PER_EPISODE,
            "model_steps_inclusive": [3, 40],
            "deterministic_reset": (
                "the exact seed-forwarding correction consumed by V5"
            ),
        },
        "regime_order": list(REGIMES),
        "regimes": REGIMES,
        "one_factor_at_a_time": True,
        "target_episodes_per_regime": TARGET_EPISODES_PER_REGIME,
        "target_rows_per_regime": TARGET_EPISODES_PER_REGIME
        * ROWS_PER_EPISODE,
        "smoke_episodes_per_regime": SMOKE_EPISODES_PER_REGIME,
        "replacement_tuples_per_regime": REPLACEMENTS_PER_REGIME,
        "sequential_expansion": False,
        "model_gate_input_allowlist": ["action", "pixels"],
        "contact_free_gate": True,
        "replacement_and_exclusion_rule": (
            "assigned slots are retained; only a mechanical exception before "
            "artifact completion may consume the next replacement tuple; "
            "target, loss, contact, motion, phase, reward, and success are "
            "forbidden from replacement, exclusion, and stopping"
        ),
        "primary_comparator": (
            "strongest transition-independent analytic mixture of fixed "
            "depths 1..4 at exactly matched adaptive total counted FLOPs, "
            "including reached feature and dual-head overhead"
        ),
        "controls": [
            "seeded weakly-more-counted-compute transition-independent allocation",
            "fixed depth 1",
            "within-episode exact-call-histogram randomization",
        ],
        "co_primary_endpoints": list(CO_PRIMARY_ENDPOINTS),
        "bootstrap": {
            "unit": "episode",
            "replicates": BOOTSTRAP_REPLICATES,
            "family": (
                "three regimes x raw/fixed-whitened exact-compute analytic claims"
            ),
            "family_size": SIMULTANEOUS_FAMILY_SIZE,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "per_claim_alpha": PER_CLAIM_ALPHA,
            "simultaneous_method": (
                "Bonferroni one-sided percentile lower bounds"
            ),
            "individual_intervals": "two-sided 95% percentile",
        },
        "diagnostics": [
            "stagewise score/next-stage-gain Spearman signs and reached counts",
            "regime effect heterogeneity",
            "call-depth distribution shift",
            "gate-score and causal-feature shift",
            "solver-gain shift",
            "exact compute-price implications",
            "routing score-bin calibration",
        ],
        "resource_reporting": {
            "base_calls": True,
            "refiner_calls": True,
            "gate_evaluations": True,
            "feature_flops": True,
            "gate_head_flops": True,
            "total_flops": True,
            "nonflop_operations": True,
            "latency_separate": True,
            "energy_separate": True,
        },
        "posthoc_only_after_immutable_terminal_decision": [
            "contact",
            "motion",
            "phase",
            "reward",
            "success",
        ],
        "forbidden_evidence": {
            "v3_test_targets_opened": False,
            "combined_v3_cache_numpy_loaded": False,
            "released_hdf5_opened": False,
        },
        "scientific_components_changed": False,
        "generalization_target_outcome_episodes": 0,
    }


def build_mapping() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "terminal_labels": list(TERMINAL_LABELS),
        "integrity_precedence": (
            "any required integrity failure => generalization_execution_invalid"
        ),
        "claim_support": (
            "a regime x endpoint claim is supported exactly when its fixed "
            "Bonferroni one-sided episode-bootstrap lower bound is > 0"
        ),
        "regime_mapping": {
            "supported": "both co-primary claims supported",
            "mixed": "exactly one co-primary claim supported",
            "failed": "neither co-primary claim supported",
        },
        "overall_mapping": {
            "zero_shot_generalization_supported": (
                "all six regime x endpoint claims supported"
            ),
            "zero_shot_generalization_partial": (
                "one through five of the six claims supported"
            ),
            "zero_shot_generalization_failed": (
                "none of the six claims supported"
            ),
            "generalization_execution_invalid": (
                "any required integrity failure"
            ),
        },
        "auxiliary_controls_affect_terminal_mapping": False,
        "stagewise_ranks_affect_terminal_mapping": False,
        "heterogeneity_affects_terminal_mapping": False,
        "latency_energy_affect_terminal_mapping": False,
        "posthoc_fields_affect_terminal_mapping": False,
        "no_retry_after_process_valid_result": True,
        "family_size": SIMULTANEOUS_FAMILY_SIZE,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "generalization_target_outcome_episodes": 0,
    }


def build_compute_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_model_flops_per_row": BASE_FLOPS,
        "mandatory_depth1_flops_per_row": V1_FLOPS,
        "additional_adapter_flops_per_call": ADAPTER_FLOPS,
        "gate": {
            "feature_flops_per_reached_evaluation": GATE_FEATURE_FLOPS,
            "dual_affine_head_flops_per_reached_evaluation": GATE_HEAD_FLOPS,
            "total_flops_per_reached_evaluation": GATE_TOTAL_FLOPS,
            "nonflop_operations_per_reached_evaluation": GATE_NONFLOP_OPS,
            "feature_width": FEATURE_DIM,
        },
        "adaptive_total": (
            "rows*(base+mandatory_depth1) + "
            "(sum(calls)-rows)*adapter + "
            "sum(min(calls,3))*gate_total"
        ),
        "analytic_equivalent_calls": (
            "sum(calls) + gate_evaluations*gate_total/adapter"
        ),
        "seeded_control": (
            "ceil analytic-equivalent total calls; therefore counted FLOPs "
            "are weakly greater than adaptive"
        ),
        "latency_is_not_substituted_for_flops": True,
        "energy_is_not_inferred_from_flops": True,
        "generalization_target_outcome_episodes": 0,
    }


def build_numerical_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rtol": NUMERICAL_RTOL,
        "atol": NUMERICAL_ATOL,
        "maximum_absolute_error_ceiling": NUMERICAL_MAX_ABS,
        "manual_sparse_vs_forward_selected_bitwise_exact": True,
        "same_sparse_path_repeat_bitwise_exact": True,
        "sparse_dense_calls_and_histograms_exact": True,
        "depth_one_selected_rows_bitwise_exact": True,
        "cross_batch_selected_latents_within_contract": True,
        "calls_reproduced_from_float32_scores_and_frozen_thresholds": True,
        "generalization_target_outcome_episodes": 0,
    }


def preregistration_markdown(power: dict[str, Any]) -> str:
    selected_power = power["selected"]["union_bound_family_power_lower"]
    return f"""# LeWM V5 zero-shot one-factor generalization preregistration

Status: normative v005 text, frozen before every new regime smoke and target
episode.

This study consumes the process-valid V5 terminal package at
`runs/lewm_v5_readiness_program/v5_package_versions/v004`. It never reruns,
edits, overwrites, or regenerates V5. Candidate
`stage_dual_r0.01_q0.85`, thresholds
`0.19465729885363534`, `0.1568665868361991`,
`0.15153321811129283`, compiled heads, PlanOracle-native whitening, base
model, refiner, solver, causal features, gradient boundaries, and compute
prices are byte-frozen.

The bounded matrix contains exactly three one-factor shifts:

1. MarkovOracle replaces PlanOracle; action noise remains 0.1 and every other
   DGP setting remains fixed.
2. PlanOracle action noise changes only from 0.1 to 0.2.
3. PlanOracle random-action probability changes only from 0 to 0.1.

Each regime has six fresh excluded smoke episodes followed by exactly 3,000
fresh target episodes (114,000 modeled rows). There is no sequential
expansion. Episode IDs and environment, policy, oracle-NumPy, action-space,
bootstrap, comparator, histogram, and qualification seeds are preassigned,
mutually disjoint, and checked against prior recorded runs and V5 V001–V004.
Two hundred replacement tuples per regime are frozen. Replacement is allowed
only for a mechanical exception before artifact completion. Target, loss,
contact, motion, phase, reward, and success cannot affect replacement,
exclusion, retention, stopping, or terminal mapping.

Power uses only the consumed V5 episode-level co-primary contrasts. The family
is three regimes by two endpoints. Bonferroni one-sided alpha is 0.05/6. The
fixed rule selects the smallest 500-episode grid point whose union-bound
family-power lower bound is at least 0.95 when only 35% of the V5 effect
remains. That point is 3,000 episodes per regime, with calculated lower bound
{selected_power:.12f}. Sensitivities are immutable; power cannot guarantee
support under shifted DGPs.

Within each regime and endpoint, the primary comparator is the strongest
transition-independent analytic mixture over fixed depths 1–4 at exactly the
adaptive total counted FLOPs, including reached feature and dual-head gate
overhead. Controls are seeded weakly-more-compute allocation, fixed depth 1,
and within-episode exact-call-histogram randomization. Raw MSE and the unchanged
V5 PlanOracle-native-whitened endpoint are retained; the latter is called
fixed-whitened under shifts to emphasize that no whitening is recalibrated.

All 9,000 raw target episodes and all execution traces must be complete and
hash-sealed before target arrays are first opened. The episode is the
bootstrap unit. Exactly 20,000 fixed-seed replicates produce two-sided 95%
individual intervals and Bonferroni one-sided lower bounds at quantile 0.05/6.
Every regime is reported separately.

A regime-endpoint claim is supported iff its simultaneous lower bound is
strictly positive. All six supported maps to
`zero_shot_generalization_supported`; one through five maps to
`zero_shot_generalization_partial`; zero maps to
`zero_shot_generalization_failed`. Any required integrity failure has
precedence and maps to `generalization_execution_invalid`. Auxiliary controls,
rank signs, heterogeneity, latency, energy, and post-hoc fields cannot alter
the decision.

Per regime, reporting includes stagewise score/next-stage-gain rank signs and
reached counts; regime heterogeneity; call-depth, gate-score, causal-feature,
and solver-gain shifts; exact compute-price implications; routing calibration;
base and refiner calls; gate evaluations; feature, head, and total FLOPs;
non-FLOP operations; synchronized latency; and energy availability. Contact,
motion, phase, reward, and success may be opened only after the immutable
independent terminal decision and only for interpretation.

An independent implementation rehashes inputs and sealed sources and
recomputes calls, losses, exact compute, comparators, all 20,000 replicates,
simultaneous bounds, diagnostics, chronology, and mapping. A process-valid
partial or failed result is terminal and cannot be repaired or retried.
Version-forward repair is permitted only after a zero-target-outcome
procedural invalidity with exact source/AST/object/hash evidence and unchanged
science.
"""


def main() -> None:
    assert_runtime_contract("evaluation")
    design_path = ATTEMPT_ROOT / "DGP_MATRIX.json"
    power_path = ATTEMPT_ROOT / "power_analysis.json"
    ledger_path = ATTEMPT_ROOT / "cohort_seed_ledger.json"
    mapping_path = ATTEMPT_ROOT / "outcome_mapping.json"
    compute_path = ATTEMPT_ROOT / "operation_ledger.json"
    numerical_path = ATTEMPT_ROOT / "numerical_equivalence_contract.json"
    prereg_path = ATTEMPT_ROOT / "PREREGISTRATION.md"
    checkpoint_path = ATTEMPT_ROOT / "audit/design_and_power.json"
    outputs = (
        design_path,
        power_path,
        ledger_path,
        mapping_path,
        compute_path,
        numerical_path,
        prereg_path,
        checkpoint_path,
    )
    if all(path.exists() for path in outputs):
        complete_state(
            "DESIGN_AND_POWER",
            "IMPLEMENT_PACKAGE",
            evidence_path=checkpoint_path,
            checkpoint_name="v005_design_power_identifiers_reused",
            next_action="materialize complete sealed implementation",
        )
        print(json.dumps(read_json(checkpoint_path), sort_keys=True))
        return
    if any(path.exists() for path in outputs):
        raise RuntimeError("partial design-and-power artifact set")

    design = build_design()
    power = power_analysis()
    if not power["passed"]:
        raise RuntimeError("fixed 3,000-episode power rule did not pass")
    roles, record_count = records()
    analysis_seeds = {
        "joint_bootstrap_seed": 3_500_000_001,
        "inference_qualification_seed": 3_500_000_002,
        "torch_qualification_seed": 3_500_000_003,
        "latency_order_seed": 3_500_000_004,
        "histogram_seeds": {
            regime: 3_500_000_100 + index
            for index, regime in enumerate(REGIMES)
        },
        "seeded_comparator_seeds": {
            regime: {
                endpoint: 3_500_000_200
                + 10 * regime_index
                + endpoint_index
                for endpoint_index, endpoint in enumerate(CO_PRIMARY_ENDPOINTS)
            }
            for regime_index, regime in enumerate(REGIMES)
        },
    }
    comparator_ids = {
        regime: {
            "analytic": f"g5v005-{SHORT[regime]}-analytic-exact-flops",
            "fixed_depth_1": f"g5v005-{SHORT[regime]}-fixed-depth-1",
            "histogram": f"g5v005-{SHORT[regime]}-within-episode-hist",
            "seeded_raw": f"g5v005-{SHORT[regime]}-seeded-raw",
            "seeded_fixed_whitened": (
                f"g5v005-{SHORT[regime]}-seeded-fixed-white"
            ),
        }
        for regime in REGIMES
    }
    qualification_ids = [
        "g5v005-qualification-dgp-constructor",
        "g5v005-qualification-synthetic-inference",
        "g5v005-qualification-sparse-dense",
    ]
    all_records = [
        item
        for regime_roles in roles.values()
        for role_records in regime_roles.values()
        for item in role_records
    ]
    episode_ids = {item["episode_id"] for item in all_records}
    numeric = {
        int(item[key])
        for item in all_records
        for key in (
            "env_seed",
            "policy_seed",
            "oracle_np_seed",
            "action_space_seed",
        )
    }
    numeric.update(
        value
        for key, value in analysis_seeds.items()
        if isinstance(value, int)
    )
    numeric.update(analysis_seeds["histogram_seeds"].values())
    numeric.update(
        seed
        for by_endpoint in analysis_seeds["seeded_comparator_seeds"].values()
        for seed in by_endpoint.values()
    )
    prior_numbers, prior_strings, scanned = safe_prior_identifiers()
    v5_versions = {}
    v5_overlap = {}
    for version in ("v001", "v002", "v003", "v004"):
        path = V5_ROOT.parent / version / "cohort_seed_ledger.json"
        prior = read_json(path)
        prior_records = [
            item for values in prior["roles"].values() for item in values
        ]
        prior_ids = {str(item["episode_id"]) for item in prior_records}
        prior_tuples = {
            (
                int(item["env_seed"]),
                int(item["policy_seed"]),
                int(item["oracle_np_seed"]),
            )
            for item in prior_records
        }
        current_tuples = {
            (
                int(item["env_seed"]),
                int(item["policy_seed"]),
                int(item["oracle_np_seed"]),
            )
            for item in all_records
        }
        v5_versions[version] = {
            "record_count": len(prior_records),
            "ledger_sha256": sha256_file(path),
        }
        v5_overlap[version] = {
            "episode_ids": sorted(episode_ids & prior_ids),
            "seed_tuples": sorted(current_tuples & prior_tuples),
        }
    string_universe = (
        episode_ids
        | set(qualification_ids)
        | {
            value
            for by_regime in comparator_ids.values()
            for value in by_regime.values()
        }
        | {"g5v005-joint-bootstrap"}
    )
    checks = {
        "record_count_exact": record_count
        == len(REGIMES)
        * (
            SMOKE_EPISODES_PER_REGIME
            + TARGET_EPISODES_PER_REGIME
            + REPLACEMENTS_PER_REGIME
        ),
        "episode_ids_unique": len(episode_ids) == len(all_records),
        "numeric_identifiers_unique": len(numeric)
        == 4 * len(all_records) + 4 + len(REGIMES) + 2 * len(REGIMES),
        "zero_safe_prior_numeric_overlap": not (numeric & prior_numbers),
        "zero_safe_prior_string_overlap": not (
            string_universe & prior_strings
        ),
        "zero_v5_v001_v004_overlap": all(
            not item["episode_ids"] and not item["seed_tuples"]
            for item in v5_overlap.values()
        ),
        "ids_fit_uint32": max(numeric) < 2**32 and min(numeric) >= 0,
        "ids_fit_raw_u32_text": max(map(len, episode_ids)) <= 32,
        "power_passed": power["passed"],
        "sample_size_exact_3000": TARGET_EPISODES_PER_REGIME == 3_000,
        "no_sequential_expansion": design["sequential_expansion"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"design or identifier checks failed: {checks}")
    seed_ledger = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "attempt": ATTEMPT_VERSION,
        "roles": roles,
        "analysis_seeds": analysis_seeds,
        "bootstrap_id": "g5v005-joint-bootstrap",
        "comparator_ids": comparator_ids,
        "qualification_ids": qualification_ids,
        "replacement_rule": (
            "mechanical exception only; never target, loss, contact, motion, "
            "phase, reward, or success"
        ),
        "replacement_pool_regime_specific_and_single_use": True,
        "smoke_permanently_excluded": True,
        "target_identifiers_never_used_for_smoke": True,
        "checks": checks,
        "prior_snapshot": {
            "safe_json_file_count": len(scanned),
            "safe_numeric_count": len(prior_numbers),
            "safe_string_count": len(prior_strings),
            "safe_numeric_sha256": canonical_digest(prior_numbers),
            "safe_string_sha256": canonical_digest(prior_strings),
            "scanned_path_sha256": canonical_digest(scanned),
            "v3_subtree_content_excluded_to_obey_forbidden-target-boundary": True,
            "v3_filenames_included_in_string_scan": True,
        },
        "v5_versions": v5_versions,
        "v5_overlap": v5_overlap,
        "numeric_overlap": sorted(numeric & prior_numbers),
        "string_overlap": sorted(string_universe & prior_strings),
        "generalization_target_outcome_episodes": 0,
    }
    mapping = build_mapping()
    compute = build_compute_contract()
    numerical = build_numerical_contract()
    atomic_json(design_path, design, exclusive=True)
    atomic_json(power_path, power, exclusive=True)
    atomic_json(ledger_path, seed_ledger, exclusive=True)
    atomic_json(mapping_path, mapping, exclusive=True)
    atomic_json(compute_path, compute, exclusive=True)
    atomic_json(numerical_path, numerical, exclusive=True)
    prereg_path.write_text(preregistration_markdown(power), encoding="utf-8")
    checkpoint = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "passed": True,
        "checks": checks,
        "files": {
            str(path.relative_to(ATTEMPT_ROOT)): sha256_file(path)
            for path in (
                design_path,
                power_path,
                ledger_path,
                mapping_path,
                compute_path,
                numerical_path,
                prereg_path,
            )
        },
        "science_frozen_before_target_outcomes": True,
        "target_outcome_episodes": 0,
        "forbidden_evidence_access": {
            "v3_test_targets_opened": False,
            "combined_v3_cache_numpy_loaded": False,
            "released_hdf5_opened": False,
        },
    }
    atomic_json(checkpoint_path, checkpoint, exclusive=True)
    append_ledger(
        "design_power_identifiers_frozen",
        evidence_path=str(checkpoint_path.relative_to(REPO_ROOT)),
        evidence_sha256=sha256_file(checkpoint_path),
        target_episodes_per_regime=TARGET_EPISODES_PER_REGIME,
        target_outcomes=0,
    )
    complete_state(
        "DESIGN_AND_POWER",
        "IMPLEMENT_PACKAGE",
        evidence_path=checkpoint_path,
        checkpoint_name="v005_design_power_identifiers_frozen",
        next_action="materialize complete sealed implementation",
    )
    print(json.dumps(checkpoint, sort_keys=True))


if __name__ == "__main__":
    main()
