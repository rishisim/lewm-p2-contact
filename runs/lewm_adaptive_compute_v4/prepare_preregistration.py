#!/usr/bin/env python3
"""Materialize the complete V4 preregistration before any fresh V4 data."""

from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

import common
import flops


def _episode_specs(role: str, count: int, env_start: int, policy_start: int) -> list[dict[str, object]]:
    return [
        {
            "slot": slot,
            "trajectory_id": f"v4-{role}-{slot:03d}",
            "seed_block": chr(ord("A") + slot // 100) if role == "confirmation" else "smoke",
            "primary": {"env_seed": env_start + slot, "policy_seed": policy_start + slot},
        }
        for slot in range(count)
    ]


def _replacement_specs(count: int, env_start: int, policy_start: int) -> list[dict[str, int]]:
    return [
        {"env_seed": env_start + index, "policy_seed": policy_start + index}
        for index in range(count)
    ]


def seed_manifest() -> dict[str, object]:
    smoke = _episode_specs("smoke", 12, 740_410_000, 840_410_000)
    confirmation = _episode_specs("confirmation", 300, 740_420_000, 840_420_000)
    payload = {
        "schema_version": 1,
        "status": "frozen_before_any_v4_smoke_or_confirmation_generation",
        "selection_rule": (
            "contiguous integer blocks above every mechanically recorded prior numeric identifier; "
            "role-disjoint; no outcome-dependent selection"
        ),
        "roles": {
            "smoke": {
                "episode_count": 12,
                "forever_excluded_from_confirmation": True,
                "episodes": smoke,
                "replacement_pool": _replacement_specs(100, 740_411_000, 840_411_000),
            },
            "confirmation": {
                "episode_count": 300,
                "seed_blocks": {
                    "A": {"slots": [0, 99], "episodes": 100},
                    "B": {"slots": [100, 199], "episodes": 100},
                    "C": {"slots": [200, 299], "episodes": 100},
                },
                "episodes": confirmation,
                "replacement_pool": _replacement_specs(500, 740_421_000, 840_421_000),
            },
        },
        "failure_replacement_rule": (
            "Only an environment exception or mechanically malformed trajectory may consume the "
            "next unused role-specific replacement pair. Every failed pair and exception is logged. "
            "Success, loss, calls, motion, contact, or any performance quantity may never cause replacement."
        ),
        "statistical_seeds": {
            "exact_total_integer_assignment": 940_430_001,
            "equal_call_integer_assignment": 940_430_002,
            "histogram_randomization": 940_430_003,
            "score_permutation": 940_430_004,
            "permuted_exact_total_integer_assignment": 940_430_005,
            "bootstrap_exact_total_analytic": 940_431_001,
            "bootstrap_exact_total_seeded": 940_431_002,
            "bootstrap_equal_call_analytic": 940_431_003,
            "bootstrap_equal_call_seeded": 940_431_004,
            "bootstrap_fixed_d1": 940_431_005,
            "bootstrap_histogram": 940_431_006,
            "bootstrap_score_permutation_analytic": 940_431_007,
            "bootstrap_score_permutation_seeded": 940_431_008,
            "bootstrap_whitened_exact_total_analytic": 940_431_009,
            "bootstrap_whitened_exact_total_seeded": 940_431_010,
            "bootstrap_whitened_histogram": 940_431_011,
            "bootstrap_block_1": 940_431_101,
            "bootstrap_block_2": 940_431_102,
            "bootstrap_block_3": 940_431_103,
            "regime_bootstrap_root": 940_432_000,
            "contact_unmatched_calls": 940_432_101,
            "contact_unmatched_benefit": 940_432_102,
            "contact_unmatched_gain": 940_432_103,
            "contact_fine_matched_calls": 940_432_201,
            "contact_fine_matched_benefit": 940_432_202,
            "contact_fine_matched_gain": 940_432_203,
            "contact_coarse_matched_calls": 940_432_301,
            "contact_coarse_matched_benefit": 940_432_302,
            "contact_coarse_matched_gain": 940_432_303,
            "runtime_exact_mixture": 940_433_001,
        },
    }
    numeric = []
    for role in ("smoke", "confirmation"):
        for episode in payload["roles"][role]["episodes"]:
            numeric.extend(episode["primary"].values())
        for item in payload["roles"][role]["replacement_pool"]:
            numeric.extend(item.values())
    numeric.extend(payload["statistical_seeds"].values())
    if len(numeric) != len(set(numeric)):
        raise RuntimeError("V4 seed manifest contains duplicate numeric seeds")
    payload["all_seed_count"] = len(numeric)
    payload["all_seeds_sha256"] = common.array_sha256(np.asarray(sorted(numeric), dtype=np.int64))
    return payload


def sample_size() -> dict[str, object]:
    calibration_effect = 1.847239744601695e-5
    calibration_se = 5.188055129906612e-6
    calibration_n = 90
    n = 300
    target = calibration_effect / 2.0
    projected_se = calibration_se * math.sqrt(calibration_n / n)
    z_alpha = float(norm.ppf(0.975))
    noncentrality = target / projected_se
    power = float(norm.cdf(-z_alpha - noncentrality) + 1.0 - norm.cdf(z_alpha - noncentrality))
    return {
        "confirmation_episodes": n,
        "alpha_two_sided": 0.05,
        "calibration_episodes": calibration_n,
        "calibration_effect": calibration_effect,
        "calibration_episode_cluster_standard_error": calibration_se,
        "target_effect_half_calibration": target,
        "projected_standard_error": projected_se,
        "noncentrality": noncentrality,
        "normal_approximation_power": power,
        "formula": (
            "SE_300=SE_90*sqrt(90/300); lambda=(calibration_effect/2)/SE_300; "
            "power=Phi(-z_.975-lambda)+1-Phi(z_.975-lambda)"
        ),
        "interpretation": "approximately 90% power for a two-sided 5% test/95% interval",
        "sequential_expansion_forbidden": True,
    }


def decision_rule() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "frozen_before_any_v4_data_generation_or_target_access",
        "primary_endpoint": "raw latent mean squared error per transition",
        "estimand": "transition-weighted mean with episode-cluster resampling",
        "bootstrap": {"replicates": 10_000, "confidence": 0.95, "two_sided": True},
        "frozen_policy": {
            "family": common.FAMILY,
            "operating_point": common.OPERATING_POINT,
            "compute_price": common.COMPUTE_PRICE,
            "force_v4_mean_calls_to_1.25": False,
        },
        "primary_exact_total_flop_comparator": {
            "definition": (
                "Strongest transition-independent convex mixture from the V4 aggregate fixed-exit "
                "loss vector at adaptive solver calls plus adaptive gate feature/normalization/head/"
                "control FLOPs converted to fractional expected later-adapter calls."
            ),
            "analytic": "exact expected total-FLOP equality",
            "integer": (
                "MILP-optimal depth counts at ceil(fractional total calls), hash-seeded assignment "
                "independent of transition data; residual baseline-minus-adaptive FLOPs reported"
            ),
            "integer_rounding": "ceiling, conservatively granting the baseline weakly more compute",
            "raw_ci_low_must_exceed_zero_analytic": True,
            "raw_ci_low_must_exceed_zero_seeded_integer": True,
        },
        "equal_call_mixture": "secondary allocation-quality decomposition before gate cost",
        "raw_required": [
            "exact-total-FLOP analytic mixture CI lower > 0",
            "exact-total-FLOP hash-seeded conservative integer mixture CI lower > 0",
            "fixed d1 CI lower > 0",
            "exact histogram randomization CI lower > 0",
            "score permutation does not significantly beat its own exact-total analytic or seeded baseline",
            "adaptive is nondominated on realized block-call and fully counted FLOP frontiers",
            "all mechanical validity audits pass",
        ],
        "whitening": {
            "transform": "frozen discovery whitening; never refit",
            "full_pass": "analytic exact-total-FLOP whitened CI lower > 0",
            "raw_only": "all raw criteria pass and whitened point > 0 but CI includes 0",
            "failure": "whitened point < 0 or any raw/compute criterion fails",
        },
        "verdicts": {
            "v4_confirmatory_passed": "all raw/validity/compute criteria and whitened CI lower > 0",
            "v4_confirmatory_raw_only": "all raw/validity/compute criteria; whitened point positive, CI crosses zero",
            "v4_confirmatory_failed": "any primary raw/compute criterion fails, whitening is negative, or no replication",
            "v4_confirmatory_invalid": "freshness, leakage, preregistration, target-access, checkpoint, exact-call, or mechanical audit failure",
        },
        "runtime_status_separate": ["latency_improved", "latency_neutral", "latency_regressed"],
        "physical_claim_rule": (
            "Use the fine frozen coarsened-exact bins as primary physical matching. Call the policy "
            "contact-aware only if matched contact-minus-noncontact calls and exact-total-analytic "
            "prediction benefit are both positive with clustered 95% CI lower > 0."
        ),
    }


def config(seeds: dict[str, object]) -> dict[str, object]:
    commands = [
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/audit_prior.py",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/prepare_preregistration.py",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/run_tests.py phase0",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/seal.py phase0",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/freshness.py index",
        f"{common.GENERATION_PYTHON} runs/lewm_adaptive_compute_v4/generator.py smoke",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/evaluate.py smoke --device mps",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/run_tests.py preconfirmation",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/seal.py preconfirmation",
        f"{common.GENERATION_PYTHON} runs/lewm_adaptive_compute_v4/generator.py confirmation",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/receipt.py",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/freshness.py audit",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/evaluate.py confirmation --device mps",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/statistics.py",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/secondary.py",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/flops.py",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/benchmark_runtime.py",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/report.py",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/run_tests.py final",
        f"{common.EVALUATION_PYTHON} runs/lewm_adaptive_compute_v4/finalize.py",
    ]
    return {
        "schema_version": 1,
        "study": "lewm_adaptive_compute_v4_confirmatory",
        "family": common.FAMILY,
        "operating_point": common.OPERATING_POINT,
        "compute_price": common.COMPUTE_PRICE,
        "gate": {"parameters": common.GATE_PARAMETERS, "flops_per_decision": common.GATE_FLOPS_PER_DECISION},
        "sample": {"smoke_episodes": 12, "confirmation_episodes": 300, "examples_per_episode": 38},
        "bootstrap_samples": 10_000,
        "statistical_seeds": seeds["statistical_seeds"],
        "environment_generation": {
            "environment": "swm/OGBCube-v0",
            "num_envs": 1,
            "max_episode_steps": 200,
            "image_shape": [224, 224],
            "env_type": "single",
            "multiview": False,
            "terminate_at_goal": False,
            "mode": "data_collection",
            "policy": "markov_oracle",
            "policy_action_noise": 0.1,
            "policy_random_action_probability": 0.0,
            "policy_noise_smoothing": 0.5,
            "policy_min_norm": 0.4,
        },
        "evaluation": {"encode_batch_size": 64, "predict_batch_size": 1024, "solver_batch_size": 1024, "device": "mps"},
        "runtime_benchmark": {
            "batch_sizes": [1, 32, 256, 1024],
            "warmups": 2,
            "repetitions": 10,
            "improved_geomean_ratio_max": 0.98,
            "improved_any_batch_ratio_max": 1.05,
            "neutral_geomean_ratio_max": 1.02,
            "neutral_any_batch_ratio_max": 1.10,
            "status_compares": "optimized sparse versus frozen reference sparse",
        },
        "exact_commands": commands,
    }


def source_manifest() -> dict[str, object]:
    sources = [
        common.REPO / "runs/lewm_adaptive_compute_v2/model_io.py",
        common.DISCOVERY / "run_discovery.py",
        common.DISCOVERY / "models.py",
        common.DISCOVERY / "policy.py",
        common.DISCOVERY / "data_isolation.py",
        common.DISCOVERY / "extract_isolated.py",
        common.REPO / "runs/lewm_adaptive_compute_v1/refiner.py",
        common.REPO / "le-wm/jepa.py",
        common.REPO / "le-wm/module.py",
    ]
    return {
        "schema_version": 1,
        "immutable_dependency_sources": common.collect_file_hashes(sources, base=common.REPO),
        "v4_sources": common.collect_file_hashes(
            [path for path in common.ROOT.glob("*.py")], base=common.REPO
        ),
        "v3_test_targets_opened": False,
    }


def checkpoint_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "objects": common.verify_frozen_objects(),
        "immutable_contract": {
            "family": common.FAMILY,
            "operating_point": common.OPERATING_POINT,
            "compute_price": common.COMPUTE_PRICE,
            "gate_parameters": common.GATE_PARAMETERS,
            "gate_flops_per_decision": common.GATE_FLOPS_PER_DECISION,
            "normalization": "student checkpoint frozen mean/std",
            "whitening": "discovery whitening checkpoint",
        },
    }


def preregistration_markdown(power: dict[str, object]) -> str:
    return f"""# Preregistration: LeWM adaptive-computation V4 confirmation

Status: frozen before any V4 smoke or confirmation episode generation. This is confirmation of the already-frozen stagewise solver and shared full-feature linear gate, not discovery, tuning, or recalibration.

## Frozen scientific object

- Stagewise solver SHA-256: `{common.EXPECTED['solver_checkpoint_sha256']}`.
- Linear student SHA-256: `{common.EXPECTED['student_checkpoint_sha256']}`.
- Frozen tournament SHA-256: `{common.EXPECTED['tournament_sha256']}`.
- Family `{common.FAMILY}`, operating point `{common.OPERATING_POINT}`, compute price `{common.COMPUTE_PRICE!r}`.
- Gate: {common.GATE_PARAMETERS:,} parameters and {common.GATE_FLOPS_PER_DECISION:,} incremental FLOPs per evaluated decision.
- Student normalization/depth encoding and discovery whitening are loaded from the hashed checkpoints and never refit.

No V4 result may train, recalibrate, select, or alter a solver, student, normalizer, whitening transform, compute price, feature set, depth encoding, stopping rule, baseline seed, bin, or sample size. The policy's realized V4 call rate is accepted; it is not forced to 1.25.

## Authorization audit and sample size

The prior calibration result is historical authorization only: raw adaptive MSE 0.0031314349443701435, equal-call matched MSE 0.0031499073418161604, benefit 1.847239744601695e-5 with clustered 95% CI [8.881163526542912e-6, 2.9195376017610194e-5]. Whitening was positive but inconclusive; the policy was FLOP-nondominated and slower on the recorded MPS sparse path. These facts are audited, not retested as fresh confirmation.

N=300 is fixed. The calibration episode-cluster SE is {power['calibration_episode_cluster_standard_error']:.12g}. Detecting half the calibration effect ({power['target_effect_half_calibration']:.12g}) gives projected SE {power['projected_standard_error']:.12g}, noncentrality {power['noncentrality']:.6f}, and two-sided alpha=.05 normal-approximation power {power['normal_approximation_power']:.6f}. N will not change after outcomes.

Exactly 12 fresh pipeline-smoke episodes and exactly 300 confirmation episodes are preregistered. Smoke is physically separate, role-labelled, immutable, and forever excluded. Only environment exceptions or malformed generation may consume the next frozen replacement seed; every failure is retained. There are no performance exclusions and no sequential expansion.

## Primary comparator and decision

Raw latent MSE is primary. All intervals use exactly 10,000 hash-seeded episode-cluster bootstrap replicates. The primary transition-independent comparator is the strongest convex depth mixture at the adaptive policy's exact fully counted total FLOPs. The adaptive gate's feature construction, normalization, head, and control-flow FLOPs are converted to fractional expected later-adapter calls and granted to the baseline. Its analytic expected loss is exactly FLOP matched. Its integer realization uses MILP-optimal depth counts at the ceiling of that call budget, so it receives weakly more compute; the residual FLOPs are reported. Assignment is hash seeded and independent of transitions. Both raw CI lower bounds must exceed zero. Equal solver-call mixtures remain secondary allocation-quality decompositions.

A raw pass additionally requires CI lower >0 versus fixed d1 and exact histogram shuffle; score permutation may not significantly beat either of its own exact-total baselines; adaptive must be nondominated on block-call and fully counted FLOP frontiers; and every freshness, receipt, hash, exact-call, finite-value, causality, sparse-equivalence, and no-gradient audit must pass. The complete mechanical verdict mapping is in `decision_rule.json`.

Frozen discovery whitening is applied without refitting. Full pass requires whitened analytic exact-total benefit CI lower >0. Positive whitening with a crossing interval yields raw-only confirmation. A negative whitening point estimate yields failure and flags a likely motion/variance shortcut.

## Freshness, one-shot access, and isolation

All mechanically recorded prior identifiers were enumerated before selecting new disjoint seed blocks. Every accessible prior raw Cube episode outside the permanently untouched V3 test set is hashed for exact duplicate comparison. V3 test targets are never opened. The released checkpoint lacks its original pretraining episode manifest; therefore mechanical pretraining nonmembership cannot be claimed despite newly generated documented seeds and exact-duplicate checks.

An exclusive confirmation receipt is created after all 300 raw trajectories exist and before any evaluator opens confirmatory observations as target/next latents. The complete frozen policy is evaluated once. No partial confirmation metric is printed or inspected during evaluation, and no choice changes or rescue episodes are allowed.

## Secondary physical analysis

Contact, impact, other free motion, and static labels are prohibited from solver and gate inputs and attached only after the primary verdict. Frozen discovery-fit fine and coarse bins match contact to noncontact on trajectory phase, target block-motion magnitude, and five-action magnitude. Unmatched and coarsened-exact overlap-weighted contrasts, three fixed seed blocks, and offsets -2 through +3 around impact onset are reported. “Contact-aware” is allowed only when the fine-bin matched call and prediction-benefit contrasts are positive with clustered CI lower >0. A null physical contrast limits that claim without changing the primary verdict.

## Compute and deployment

FLOPs are recomputed from the common base, V1 call, later adapters, feature construction, normalization, linear head, and control flow. MPS benchmarks at batches 1, 32, 256, and 1024 include base prediction, solver, scoring, routing, indexing, synchronization, and compare fixed d1–d4, exact-total and equal-call mixtures, dense adaptive, frozen reference sparse, and optimized sparse. Optimized execution must select exactly the reference depths and produce numerically equivalent outputs. Deployment status compares optimized to reference sparse under the frozen thresholds in `config.json`; latency never substitutes for the FLOP/statistical endpoint. Energy is reported only if reliable.

## Interpretation

A pass supports adaptive computation for this frozen visual latent physical world model at one preregistered operating point. It is not a claim of the first adaptive world model; LoopWM remains relevant adaptive-depth world-model work in text environments. Overall compute-responsive allocation and the stronger contact-aware claim remain separate.
"""


def run() -> dict[str, object]:
    forbidden = [
        common.ROOT / "data/smoke_data_manifest.json",
        common.ROOT / "data/confirmation_data_manifest.json",
        common.ROOT / "audit/confirmation_access_receipt.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("cannot preregister after any V4 generated-data or access artifact exists")
    for name in (
        "config.json",
        "decision_rule.json",
        "seed_manifest.json",
        "source_manifest.json",
        "checkpoint_manifest.json",
        "provenance_manifest.json",
        "PREREGISTRATION.md",
    ):
        if (common.ROOT / name).exists():
            raise RuntimeError(f"preregistration artifact already exists: {name}")
    common.verify_frozen_objects()
    seeds = seed_manifest()
    inventory = common.read_json(common.ROOT / "audit/prior_identifier_inventory.json")
    prior_ids = set(map(int, inventory["unique_numeric_ids"]))
    all_seed_values = []
    for role in ("smoke", "confirmation"):
        for item in seeds["roles"][role]["episodes"]:
            all_seed_values.extend(item["primary"].values())
        for item in seeds["roles"][role]["replacement_pool"]:
            all_seed_values.extend(item.values())
    all_seed_values.extend(seeds["statistical_seeds"].values())
    overlap = sorted(set(all_seed_values) & prior_ids)
    if overlap:
        raise RuntimeError(f"new frozen seeds overlap recorded identifiers: {overlap[:20]}")
    seeds["prior_identifier_inventory_sha256"] = common.sha256_file(
        common.ROOT / "audit/prior_identifier_inventory.json"
    )
    seeds["recorded_identifier_count"] = len(prior_ids)
    seeds["recorded_identifier_maximum"] = max(prior_ids)
    seeds["overlap_with_recorded_identifiers"] = overlap
    power = sample_size()
    cfg = config(seeds)
    cfg["sample_size_calculation"] = power
    common.write_json(common.ROOT / "seed_manifest.json", seeds, exclusive=True)
    common.write_json(common.ROOT / "config.json", cfg, exclusive=True)
    common.write_json(common.ROOT / "decision_rule.json", decision_rule(), exclusive=True)
    common.write_json(common.ROOT / "source_manifest.json", source_manifest(), exclusive=True)
    common.write_json(common.ROOT / "checkpoint_manifest.json", checkpoint_manifest(), exclusive=True)
    provenance = {
        "schema_version": 1,
        "status": "preregistered_before_fresh_v4_generation",
        "environment": common.source_environment(),
        "evaluation_python": str(common.EVALUATION_PYTHON),
        "generation_python": str(common.GENERATION_PYTHON),
        "prior_evidence_audit_sha256": common.sha256_file(
            common.ROOT / "audit/prior_evidence_audit.json"
        ),
        "prior_identifier_inventory_sha256": common.sha256_file(
            common.ROOT / "audit/prior_identifier_inventory.json"
        ),
        "regime_matching_bins_sha256": common.sha256_file(
            common.ROOT / "audit/regime_matching_bins.json"
        ),
        "v3_test_targets_opened": False,
        "no_commit_or_push": True,
    }
    common.write_json(common.ROOT / "provenance_manifest.json", provenance, exclusive=True)
    (common.ROOT / "PREREGISTRATION.md").write_text(
        preregistration_markdown(power), encoding="utf-8"
    )
    flop_record = flops.recompute()
    if not flop_record["passed"]:
        raise RuntimeError("preregistered FLOP accounting failed")
    return {
        "status": "preregistered",
        "confirmation_episodes": 300,
        "smoke_episodes": 12,
        "power": power["normal_approximation_power"],
        "seed_overlap": overlap,
        "v3_test_targets_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
