#!/usr/bin/env python3
"""Materialize and seal the complete cycle-003 prospective protocol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from cycle_common import (
    ADAPTER_FLOPS,
    BASE_FLOPS,
    PROGRAM_ROOT,
    REPO_ROOT,
    ROOT,
    SOURCE_DISCOVERY,
    THRESHOLD,
    V1_FLOPS,
    atomic_json,
    read_json,
    sha256_file,
    verify_expected_sources,
)


def recursive_integers(value: Any) -> set[int]:
    result: set[int] = set()
    if isinstance(value, bool):
        return result
    if isinstance(value, int):
        result.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            result.update(recursive_integers(item))
    elif isinstance(value, list):
        for item in value:
            result.update(recursive_integers(item))
    return result


def recursive_strings(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, str):
        result.add(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            result.add(str(key))
            result.update(recursive_strings(item))
    elif isinstance(value, list):
        for item in value:
            result.update(recursive_strings(item))
    return result


def digest_values(values: set[Any]) -> str:
    return hashlib.sha256(
        ("\n".join(str(item) for item in sorted(values, key=lambda item: str(item))) + "\n").encode()
    ).hexdigest()


def prior_identifiers() -> tuple[set[int], set[str], dict[str, Any]]:
    integers: set[int] = set()
    strings: set[str] = set()
    scanned = []
    for path in sorted((REPO_ROOT / "runs").rglob("*.json")):
        if ROOT in path.parents or ".venv" in path.parts:
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        integers.update(recursive_integers(value))
        strings.update(recursive_strings(value))
        scanned.append(str(path.relative_to(REPO_ROOT)))
    for path in (REPO_ROOT / "runs").rglob("*"):
        if path.is_file() and ROOT not in path.parents and ".venv" not in path.parts:
            strings.add(path.stem)
            strings.add(path.name)
    snapshot = {
        "schema_version": 1,
        "scope": "all parseable prior run JSON and filenames outside the current unsealed cycle, including all earlier V5-readiness cycles; interpreter environments excluded",
        "json_file_count": len(scanned),
        "numeric_identifier_count": len(integers),
        "string_identifier_count": len(strings),
        "numeric_identifier_set_sha256": digest_values(integers),
        "string_identifier_set_sha256": digest_values(strings),
        "scanned_path_set_sha256": digest_values(set(scanned)),
    }
    return integers, strings, snapshot


def build_cohort() -> dict[str, Any]:
    prior_numbers, prior_strings, snapshot = prior_identifiers()
    atomic_json(ROOT / "audit/prior_identifier_snapshot.json", snapshot, exclusive=True)
    base = 1_885_000_000
    role_specs = ("smoke", 12, 0), ("prospective", 300, 10_000), ("replacement", 60, 50_000)
    roles = {}
    for role, count, offset in role_specs:
        roles[role] = [
            {
                "role": role,
                "slot": slot,
                "episode_id": f"c003-{role}-{slot:03d}",
                "env_seed": base + offset + slot,
                "policy_seed": base + 100_000 + offset + slot,
                "oracle_np_seed": base + 200_000 + offset + slot,
            }
            for slot in range(count)
        ]
    analysis_seeds = {
        "bootstrap_seed": 1_985_999_991,
        "histogram_seed": 1_985_888_881,
        "seeded_mixture_seed": 1_985_777_771,
    }
    new_numbers = {
        item[key]
        for values in roles.values()
        for item in values
        for key in ("env_seed", "policy_seed", "oracle_np_seed")
    } | set(analysis_seeds.values())
    new_strings = {item["episode_id"] for values in roles.values() for item in values}
    numeric_overlap = sorted(new_numbers & prior_numbers)
    string_overlap = sorted(new_strings & prior_strings)
    if numeric_overlap or string_overlap:
        raise RuntimeError(
            f"prospective identifier overlap: numeric={numeric_overlap[:5]}, strings={string_overlap[:5]}"
        )
    expected_number_count = 3 * sum(len(values) for values in roles.values()) + len(analysis_seeds)
    if len(new_numbers) != expected_number_count or len(new_strings) != sum(
        len(values) for values in roles.values()
    ):
        raise RuntimeError("new cohort identifiers are not internally unique")
    ledger = {
        "schema_version": 1,
        "roles": roles,
        **analysis_seeds,
        "all_new_numeric_identifiers_unique": True,
        "all_new_string_identifiers_unique": True,
        "prior_identifier_snapshot": snapshot,
        "overlap_with_prior_recorded_numeric_identifiers": numeric_overlap,
        "overlap_with_prior_recorded_string_identifiers": string_overlap,
        "replacement_rule": "mechanical generation exception or malformed rollout only; never outcome, loss, contact, reward, or success",
        "freshness": "all cycle-003 smoke and prospective episode identifiers and all three RNG streams are disjoint from every recorded prior run and prior V5-readiness-cycle identifier",
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / "cohort_seed_ledger.json", ledger, exclusive=True)
    return ledger


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "numpy",
        "scipy",
        "torch",
        "stable-worldmodel",
        "ogbench",
        "mujoco",
        "gymnasium",
        "transformers",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    expected = {
        "stable-worldmodel": "0.1.0",
        "ogbench": "1.2.1",
        "mujoco": "3.10.0",
        "gymnasium": "1.3.0",
    }
    if any(result[name] != version for name, version in expected.items()):
        raise RuntimeError(f"DGP package version mismatch: {result}")
    return result


def main() -> None:
    seal_path = ROOT / "audit/pre_data_seal.json"
    if seal_path.exists():
        raise RuntimeError("cycle 003 is already sealed")
    expected_sources = verify_expected_sources()
    required_qualification = {
        "audit/preseal_tests.json": "passed",
        "audit/consumed_feature_gate_qualification.json": "passed",
        "audit/runtime_qualification.json": "passed",
        "audit/flop_derivation_graph.json": "passed",
        "audit/flop_derivation_symbolic.json": "passed",
        "audit/simultaneous_inference_qualification.json": "passed",
        "numerical_equivalence_contract.json": "passed",
    }
    for relative, key in required_qualification.items():
        payload = read_json(ROOT / relative)
        if not payload.get(key):
            raise RuntimeError(f"preseal qualification did not pass: {relative}")
    graph = read_json(ROOT / "audit/flop_derivation_graph.json")
    symbolic = read_json(ROOT / "audit/flop_derivation_symbolic.json")
    if graph["totals"] != symbolic["totals"]:
        raise RuntimeError("independent FLOP derivations disagree")
    totals = graph["totals"]
    if totals["total_flops_per_reached_evaluation"] != 7997:
        raise RuntimeError("gate total is not the executable graph total")
    cohort = build_cohort()
    versions = package_versions()

    operation = {
        "schema_version": 1,
        "convention": "one scalar subtraction, multiplication, reduction add, sqrt, division, affine bias add is one FLOP; comparison/clamp/min is separately reported non-FLOP; detach/view/flatten/concatenate/index/scatter/one-hot assignment are zero FLOP",
        "base_flops_per_row": BASE_FLOPS,
        "mandatory_depth1_refiner_flops_per_row": V1_FLOPS,
        "additional_refiner_flops_per_call": ADAPTER_FLOPS,
        "feature_graph_path": "operation_graph.json",
        "semantic_feature_names_path": "semantic_feature_names.json",
        "gate": totals,
        "derivations": {
            "executable_graph": {
                "path": "audit/flop_derivation_graph.json",
                "sha256": sha256_file(ROOT / "audit/flop_derivation_graph.json"),
            },
            "independent_symbolic": {
                "path": "audit/flop_derivation_symbolic.json",
                "sha256": sha256_file(ROOT / "audit/flop_derivation_symbolic.json"),
            },
            "agree_exactly": True,
        },
        "feature_construction_counted_at_every_reached_gate_decision": True,
        "exact_total_compute_comparator": "adaptive refiner calls + reached_gate_evaluations*7997/264960 fractional adapter calls; analytic mixture exactly matches; seeded integer mixture rounds up",
    }
    # Stable aliases consumed by analysis/independent verifier.
    operation["gate"] = {
        "feature_flops_per_reached_evaluation": totals[
            "feature_flops_per_reached_evaluation"
        ],
        "dual_affine_score_flops_per_reached_evaluation": totals[
            "dual_affine_score_flops_per_reached_evaluation"
        ],
        "total_flops_per_reached_evaluation": totals[
            "total_flops_per_reached_evaluation"
        ],
        "nonflop_comparison_min_per_reached_evaluation": totals[
            "nonflop_comparison_min_per_reached_evaluation"
        ],
    }
    atomic_json(ROOT / "operation_ledger.json", operation, exclusive=True)

    protocol = {
        "schema_version": 1,
        "study": "cycle 003 PlanOracle scale-valid-simultaneous prospective discovery",
        "not_v5": True,
        "frozen_candidate": "minimax_dual_linear-r0.01-q0.85",
        "threshold": THRESHOLD,
        "sample_sizes": {"excluded_smoke": 12, "prospective": 300},
        "rows_per_episode": 38,
        "no_sequential_expansion": True,
        "dgp": {
            "stable-worldmodel": "0.1.0",
            "ogbench": "1.2.1",
            "mujoco": "3.10.0",
            "gymnasium": "1.3.0",
            "environment": "swm/OGBCube-v0",
            "agents": 1,
            "image": [224, 224],
            "environment_steps": 200,
            "frameskip": 5,
            "history": 3,
            "model_steps": [3, 40],
            "policy": "PlanOracle",
            "action_noise": 0.1,
            "p_random_action": 0,
            "noise_smoothing": 0.5,
            "min_norm": 0.4,
            "deterministic_reset_amendment": True,
        },
        "input_key_allowlist": ["pixels", "action"],
        "contact_and_privileged_fields": "never loaded into model, gate, stopping, whitening, exclusion, or decision statistics; contact may only be interpreted post hoc",
        "actual_adaptive_result": "manual sparse execution on active rows",
        "dense_execution": "separately accounted all-exit shadow for comparators and numerical equivalence only",
        "numerical_equivalence_contract": "numerical_equivalence_contract.json",
        "exact_requirements": [
            "call decisions",
            "stopping depths",
            "call histograms",
            "same-path repeated gate scores/features/outputs",
            "manual sparse versus StagewiseResidualCascade.forward_selected",
            "depth-one selected rows",
        ],
        "statistics": {
            "bootstrap_unit": "episode",
            "bootstrap_replicates": 20000,
            "bootstrap_seed": cohort["bootstrap_seed"],
            "individual_interval": "equal-tailed percentile 95%",
            "simultaneous_interval": "Bonferroni familywise-95% one-sided percentile lower bounds for the two fixed co-primary endpoints: each uses alpha/2 = 0.025; valid without commensurate units or dependence assumptions",
            "simultaneous_family_size": 2,
            "simultaneous_familywise_alpha": 0.05,
            "simultaneous_per_endpoint_alpha": 0.025,
            "strict_pass": "every individual lower bound >0 and both simultaneous lower bounds >0",
            "stagewise_rank": "Spearman score/gain rho >0 at every reached stage; scipy average ranks",
        },
        "required_individual_contrasts": [
            "raw_vs_analytic",
            "raw_vs_seeded",
            "raw_vs_fixed_d1",
            "raw_vs_within_episode_histogram",
            "native_whitened_vs_analytic",
            "native_whitened_vs_within_episode_histogram",
        ],
        "co_primary": ["raw_vs_analytic", "native_whitened_vs_analytic"],
        "comparators": {
            "analytic": "strongest transition-independent fixed-depth mixture at exact total adaptive FLOPs",
            "seeded": "preseeded integer fixed-depth mixture rounded to weakly more total FLOPs",
            "fixed_depth_1": "all rows at depth one",
            "histogram": "within-episode permutation preserving each exact call histogram",
        },
        "latency": "post-analysis synchronized latent/action-input through base prediction and full sparse path; reported separately and cannot affect statistical/FLOP verdict",
        "terminal_outcomes": [
            "cycle_prospective_discovery_passed",
            "cycle_prospective_discovery_failed",
            "cycle_execution_invalid",
        ],
        "unblinding": "only after exactly 300 raw episodes plus all sparse traces and dense shadows complete and their input seal is durable",
        "v3_test_targets_opened": False,
        "released_hdf5_opened": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / "protocol.json", protocol, exclusive=True)
    outcome = {
        "schema_version": 1,
        "pass": "cycle_prospective_discovery_passed",
        "failure": "cycle_prospective_discovery_failed",
        "invalid": "cycle_execution_invalid",
        "pass_requires": "all six individual lower bounds, both simultaneous co-primary lower bounds, every stagewise positive sign, every process/integrity criterion, and independent reproduction",
        "failure_when": "a procedurally valid sealed run misses any preregistered statistical criterion",
        "invalid_when": "any seal, DGP, chronology, path-set, isolation, sparse execution, numerical contract, hash, seed, frozen-module, finite-array, or exact-compute criterion fails",
        "latency_can_change_terminal_outcome": False,
    }
    atomic_json(ROOT / "outcome_mapping.json", outcome, exclusive=True)
    preregistration = f"""# Preregistration: LeWM V5-readiness cycle 003 scale-valid simultaneous inference

This is a prospective discovery cycle, not V5 confirmation. Cycle 001 is a permanently consumed, independently validated statistical failure. Its unstudentized shared absolute-error maximum combined raw and native-whitened contrasts whose bootstrap standard deviations differed by a factor of 73.84, so its shared absolute critical value had no common unit. This cycle makes one bounded design change: the same fixed family of two co-primary endpoints uses Bonferroni familywise-95% one-sided percentile lower bounds, each at alpha/2 = 0.025. The union bound guarantees at least 95% simultaneous coverage without common units or assumptions about dependence. The individual intervals and every substantive pass criterion remain unchanged. There is one candidate and no selection on the new cohort.

The base model, stagewise refiner, folded dual-linear gate, threshold `{THRESHOLD}`, whitening, feature order, and candidate claim are byte-identical to cycle 001. The executable causal feature graph reuses common norms, has 1,046 ordered features, and counts 3,801 feature FLOPs. Two 1,049-wide affine heads count 4,196 FLOPs. Every reached decision therefore costs 7,997 FLOPs plus five separately reported comparison/min operations. Two independent derivations must agree exactly.

Twelve fresh post-seal smoke episodes are excluded. Only if they pass will exactly 300 fresh PlanOracle episodes be generated. There is no sequential expansion. Generation uses stable-worldmodel 0.1.0, OGBench 1.2.1, MuJoCo 3.10.0, Gymnasium 1.3.0, `swm/OGBCube-v0`, one agent, 224x224 images, 200 environment steps, frameskip 5, history 3, model steps 3 through 40, PlanOracle action noise .1, random-action probability 0, smoothing .5, minimum norm .4, and the deterministic reset amendment.

The model/gate loader can materialize only `pixels` and `action`. Contact and privileged fields cannot enter execution, fitting, stopping, whitening, exclusion, statistics, or terminal mapping. The actual adaptive prediction comes from sparse active-row execution. Dense all-exit execution is a separately accounted comparator and numerical shadow.

Call decisions, stopping depths, histograms, same-path repeated scores/features/outputs, manual sparse versus `StagewiseResidualCascade.forward_selected`, and depth-one selected rows must remain bitwise exact. Cross-batch-shape selected latent tensors alone use the sealed `rtol=2e-6`, `atol=2e-7`, and independent maximum absolute ceiling `4.76837158203125e-7`; both allclose and the ceiling must pass.

All 300 raw episodes, sparse traces, and dense shadows must be complete and hash-sealed before target arrays are opened. The analysis uses 20,000 deterministic episode-level paired bootstrap replicates. The two fixed co-primary simultaneous lower bounds are the 0.025 percentile of their respective paired-bootstrap distributions; Bonferroni controls their familywise one-sided error at at most 0.05. All six individual 95% lower bounds and both simultaneous raw/native-whitened analytic lower bounds must be strictly positive. Score/gain Spearman sign must be positive at every stage. All integrity criteria and a second independent implementation must reproduce calls, effects, intervals, exact compute, and terminal mapping.

The immutable outcomes are `cycle_prospective_discovery_passed`, `cycle_prospective_discovery_failed`, and `cycle_execution_invalid`. No V5 outcome episode may be generated by this cycle.
"""
    (ROOT / "PREREGISTRATION.md").write_text(preregistration)
    (ROOT / "REPORT_TEMPLATE.md").write_text(
        "# Cycle report template\n\nTerminal outcome: {{terminal_outcome}}\n\nFacts, interpretation, process validity, limitations, exact compute, separate synchronized latency, independent audit, and zero-V5 statement are populated only by sealed `finalize.py`.\n"
    )
    atomic_json(
        ROOT / "expected_artifact_roles.json",
        {
            "schema_version": 1,
            "required_before_smoke": [
                "PREREGISTRATION.md",
                "protocol.json",
                "outcome_mapping.json",
                "cohort_seed_ledger.json",
                "operation_graph.json",
                "operation_ledger.json",
                "semantic_feature_names.json",
                "numerical_equivalence_contract.json",
                "freeze/compiled_gate.npz",
                "runner.py",
                "analyze.py",
                "latency.py",
                "independent_verify.py",
                "finalize.py",
                "verify_pre_data.py",
                "verify_final.py",
            ],
            "durable_path_set_excludes": ["__pycache__", "*.pyc", "dot-temporary files"],
            "zero_v5_outcomes": True,
        },
        exclusive=True,
    )

    local_dependencies = [
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/common.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/generator.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/generator_seedfix.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/config.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/protocol_amendment_01.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/audit/protocol_amendment_01_seal.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/source_manifest.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_v4/common.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_v4/runtime.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_v4/generator.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_v2/model_io.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_discovery/models.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_discovery/run_discovery.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_discovery/critics.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_discovery/policy.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_discovery/data_isolation.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_discovery/extract_isolated.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_discovery/config.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_v1/refiner.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_v1/checkpoints/refiner_seed_260713.pt",
        REPO_ROOT / "le-wm/jepa.py",
        REPO_ROOT / "le-wm/module.py",
        REPO_ROOT / "runs/lewm_adaptive_compute_critic_compression/config.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_critic_compression/checkpoints/final_student.pt",
        REPO_ROOT / "runs/lewm_adaptive_compute_critic_compression/audit/frozen_tournament.json",
        REPO_ROOT / "runs/lewm_adaptive_compute_critic_compression/checkpoints/discovery_whitening.npz",
        REPO_ROOT / "runs/lewm_adaptive_compute_discovery/checkpoints/stagewise_seed_261102.pt",
        REPO_ROOT / "runs/lewm_transfer/cube/cache/model/config.json",
        REPO_ROOT / "runs/lewm_transfer/cube/cache/model/weights.pt",
        SOURCE_DISCOVERY / "freeze/gate_weights.npz",
        SOURCE_DISCOVERY / "freeze/gate_contract.npz",
        SOURCE_DISCOVERY / "fit_only_normalization_whitening.npz",
    ]
    source_manifest = read_json(
        REPO_ROOT / "runs/lewm_adaptive_compute_distribution_contract/source_manifest.json"
    )
    external_dependencies = [
        Path(path) for path in source_manifest["installed_controlling_sources"]
    ]
    for path in local_dependencies + external_dependencies:
        if not path.is_file():
            raise RuntimeError(f"transitive dependency missing: {path}")
    external_hashes = {
        str(path): sha256_file(path) for path in external_dependencies
    }
    declared_external = {
        path: record["sha256"]
        for path, record in source_manifest["installed_controlling_sources"].items()
    }
    if external_hashes != declared_external:
        raise RuntimeError("installed DGP controlling source drift")
    transitive = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "generation_interpreter": str(Path(sys.executable).resolve()),
        "evaluation_interpreter": "/Users/rishisim/.cache/lewm-v2-venv/bin/python",
        "packages": versions,
        "local_and_frozen_files": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in local_dependencies
        },
        "installed_controlling_sources": external_hashes,
        "frozen_object_expected_hashes": expected_sources,
        "v3_combined_cache_loaded": False,
        "released_hdf5_opened": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / "transitive_source_package_manifest.json", transitive, exclusive=True)

    cycle_sources = sorted(ROOT.glob("*.py"))
    generated_normative = [
        ROOT / "PREREGISTRATION.md",
        ROOT / "REPORT_TEMPLATE.md",
        ROOT / "protocol.json",
        ROOT / "outcome_mapping.json",
        ROOT / "cohort_seed_ledger.json",
        ROOT / "operation_graph.json",
        ROOT / "operation_ledger.json",
        ROOT / "semantic_feature_names.json",
        ROOT / "numerical_equivalence_contract.json",
        ROOT / "expected_artifact_roles.json",
        ROOT / "transitive_source_package_manifest.json",
        ROOT / "freeze/compiled_gate.npz",
        ROOT / "audit/diagnosis_input.json",
        ROOT / "development/cycle_001_consumed_diagnosis.json",
        ROOT / "audit/preseal_tests.json",
        ROOT / "audit/consumed_feature_gate_qualification.json",
        ROOT / "audit/runtime_qualification.json",
        ROOT / "audit/flop_derivation_graph.json",
        ROOT / "audit/flop_derivation_symbolic.json",
        ROOT / "audit/simultaneous_inference_qualification.json",
        ROOT / "audit/prior_identifier_snapshot.json",
    ]
    sealed_paths = cycle_sources + generated_normative + local_dependencies + external_dependencies
    missing = [str(path) for path in sealed_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"preseal path missing: {missing}")
    data_root = ROOT / "data"
    raw_files = list(data_root.glob("**/*.npz")) if data_root.exists() else []
    if raw_files:
        raise RuntimeError(f"new raw/execution data existed before seal: {raw_files}")
    seal_files = {}
    for path in sealed_paths:
        key = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        seal_files[key] = sha256_file(path)
    seal = {
        "schema_version": 1,
        "status": "frozen_before_any_cycle_003_smoke_or_prospective_episode",
        "created_unix_ns": time.time_ns(),
        "files": seal_files,
        "zero_new_raw_episode_files_at_seal": True,
        "smoke_episodes_at_seal": 0,
        "prospective_episodes_at_seal": 0,
        "prospective_targets_opened": False,
        "v3_test_targets_opened": False,
        "released_hdf5_opened": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(seal_path, seal, exclusive=True)
    os.chmod(seal_path, 0o444)
    print(
        json.dumps(
            {
                "sealed_files": len(seal_files),
                "seal_sha256": sha256_file(seal_path),
                "gate_flops_per_reached_evaluation": 7997,
                "new_smoke_episodes": 0,
                "new_prospective_episodes": 0,
                "v5_outcome_episodes": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
