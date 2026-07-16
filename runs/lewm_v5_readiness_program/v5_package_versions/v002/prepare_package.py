#!/usr/bin/env python3
"""Materialize and seal complete V5 package v002 before excluded smoke."""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from counted_features import semantic_feature_names
from cycle_common import (
    ADAPTER_FLOPS,
    BASE_FLOPS,
    REPO_ROOT,
    ROOT,
    SOURCE_CYCLE,
    SOURCE_CYCLE_HASHES,
    V1_FLOPS,
    assert_runtime_contract,
    atomic_json,
    read_json,
    sha256_file,
    verify_expected_sources,
)


def package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
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
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    expected = {
        "stable-worldmodel": "0.1.0",
        "ogbench": "1.2.1",
        "mujoco": "3.10.0",
        "gymnasium": "1.3.0",
    }
    if any(values[name] != version for name, version in expected.items()):
        raise RuntimeError(f"V5 DGP package drift: {values}")
    return values


def verify_accepted_cycle() -> dict[str, str]:
    paths = {
        "decision": SOURCE_CYCLE / "decision.json",
        "artifact_manifest": SOURCE_CYCLE / "artifact_manifest.json",
        "independent_verification": SOURCE_CYCLE / "audit/independent_verification.json",
        "compiled_gate": SOURCE_CYCLE / "freeze/compiled_gate.npz",
        "gate_fit": SOURCE_CYCLE / "freeze/gate_fit.npz",
        "gate_freeze": SOURCE_CYCLE / "freeze/gate_freeze.json",
        "gate_freeze_seal": SOURCE_CYCLE / "freeze/gate_freeze_seal.json",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    if observed != SOURCE_CYCLE_HASHES:
        raise RuntimeError(f"accepted cycle source drift: {observed}")
    decision = read_json(paths["decision"])
    independent = read_json(paths["independent_verification"])
    if (
        decision["terminal_outcome"] != "cycle_prospective_discovery_passed"
        or not decision["independent_audit_passed"]
        or decision["v5_outcome_episodes"] != 0
        or not independent["passed"]
        or independent["v5_outcome_episodes"] != 0
    ):
        raise RuntimeError("accepted discovery status drift")
    manifest = read_json(paths["artifact_manifest"])
    bad = [
        relative
        for relative, record in manifest["files"].items()
        if not (REPO_ROOT / relative).exists()
        or sha256_file(REPO_ROOT / relative) != record["sha256"]
    ]
    if bad:
        raise RuntimeError(f"accepted final manifest hash drift: {bad[:5]}")
    return observed


def no_outcome_data() -> None:
    data = ROOT / "data"
    artifacts = [path for path in data.rglob("*") if path.is_file()]
    forbidden = [
        ROOT / "analysis_result.json",
        ROOT / "decision.json",
        ROOT / "PACKAGE_READY.json",
        ROOT / "artifact_manifest.json",
    ]
    if artifacts or any(path.exists() for path in forbidden):
        raise RuntimeError(f"data or terminal artifacts exist before pre-V5 seal: {artifacts}")


def write_semantic_and_operation_artifacts() -> None:
    names = semantic_feature_names(latent_dim=192, action_dim=25, history_len=3)
    if len(names) != 1046 or len(set(names)) != 1046:
        raise RuntimeError("semantic feature-name graph drift")
    atomic_json(
        ROOT / "semantic_feature_names.json",
        {
            "schema_version": 1,
            "generator": "counted_features.semantic_feature_names",
            "feature_count": len(names),
            "ordered_names": list(names),
            "executable_code_sha256": sha256_file(ROOT / "counted_features.py"),
        },
        exclusive=True,
    )
    graph = read_json(ROOT / "operation_graph.json")
    graph_derivation = read_json(ROOT / "audit/flop_derivation_graph.json")
    symbolic_derivation = read_json(ROOT / "audit/flop_derivation_symbolic.json")
    expected = {
        "feature_flops_per_reached_evaluation": 3801,
        "dual_affine_score_flops_per_reached_evaluation": 4184,
        "total_flops_per_reached_evaluation": 7985,
        "nonflop_comparison_min_per_reached_evaluation": 5,
    }
    if (
        graph["totals"] != expected
        or graph_derivation["totals"] != expected
        or symbolic_derivation["totals"] != expected
    ):
        raise RuntimeError("two exact operation derivations do not agree")
    atomic_json(
        ROOT / "operation_ledger.json",
        {
            "schema_version": 1,
            "executable_feature_function": "counted_features.build_counted_causal_features",
            "semantic_feature_names_path": "semantic_feature_names.json",
            "operation_graph_path": "operation_graph.json",
            "operation_graph_sha256": sha256_file(ROOT / "operation_graph.json"),
            "primitive_nodes": graph["nodes"],
            "gate": expected,
            "model": {
                "base_flops_per_row": BASE_FLOPS,
                "mandatory_depth1_refiner_flops_per_row": V1_FLOPS,
                "each_additional_adapter_flops_per_row": ADAPTER_FLOPS,
            },
            "accounting_rule": "feature and dual-affine gate FLOPs are counted at every reached decision; five comparison/min operations are reported separately",
            "two_independent_derivations_agree_exactly": True,
        },
        exclusive=True,
    )


def write_frozen_candidate(source_hashes: dict[str, str]) -> None:
    package_files = {
        relative: sha256_file(ROOT / relative)
        for relative in (
            "freeze/compiled_gate.npz",
            "freeze/gate_fit.npz",
            "freeze/gate_freeze.json",
            "freeze/gate_freeze_seal.json",
            "freeze/whitening.npz",
        )
    }
    expected_package = {
        "freeze/compiled_gate.npz": SOURCE_CYCLE_HASHES["compiled_gate"],
        "freeze/gate_fit.npz": SOURCE_CYCLE_HASHES["gate_fit"],
        "freeze/gate_freeze.json": SOURCE_CYCLE_HASHES["gate_freeze"],
        "freeze/gate_freeze_seal.json": SOURCE_CYCLE_HASHES["gate_freeze_seal"],
        "freeze/whitening.npz": "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796",
    }
    if package_files != expected_package:
        raise RuntimeError(f"frozen package object drift: {package_files}")
    with np.load(ROOT / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        thresholds = stored["thresholds"].astype(np.float64)
        shapes = {name: list(stored[name].shape) for name in stored.files}
    if shapes["a_raw"] != [3, 1046] or thresholds.shape != (3,):
        raise RuntimeError("compiled gate shape drift")
    external = verify_expected_sources()
    gate_freeze = read_json(ROOT / "freeze/gate_freeze.json")
    if (
        gate_freeze["selected"]["candidate_id"] != "stage_dual_r0.01_q0.85"
        or not gate_freeze["fit_selection_isolation_passed"]
    ):
        raise RuntimeError("frozen gate selection provenance drift")
    candidate = {
        "schema_version": 1,
        "candidate_id": "stage_dual_r0.01_q0.85",
        "claim": "lower episode-averaged raw and PlanOracle-native-whitened latent MSE than the strongest transition-independent analytic exact-total-compute mixture under the frozen PlanOracle DGP",
        "thresholds_float64_storage": thresholds.tolist(),
        "thresholds_float32_runtime": thresholds.astype(np.float32).tolist(),
        "compiled_shapes": shapes,
        "package_files": package_files,
        "external_model_sources": external,
        "accepted_discovery_source_hashes": source_hashes,
        "fit_selection_isolation_passed": True,
        "contact_free_gate": True,
        "input_allowlist": ["action", "pixels"],
        "all_hashes_verified": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / "freeze/frozen_candidate_manifest.json", candidate, exclusive=True)


def local_preseal_files() -> list[Path]:
    excluded = {
        ROOT / "transitive_source_package_manifest.json",
        ROOT / "audit/pre_v5_seal.json",
    }
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path not in excluded
        and "data" not in path.relative_to(ROOT).parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not path.name.startswith(".")
    )


def build_transitive() -> dict[str, Any]:
    source = read_json(SOURCE_CYCLE / "transitive_source_package_manifest.json")
    external = {
        relative: expected
        for relative, expected in source["files"].items()
        if "runs/lewm_v5_readiness_program/cycles/" not in relative
    }
    evidence_paths = [
        SOURCE_CYCLE / "decision.json",
        SOURCE_CYCLE / "artifact_manifest.json",
        SOURCE_CYCLE / "audit/independent_verification.json",
        SOURCE_CYCLE / "metrics/prospective_episode_metrics.npz",
        SOURCE_CYCLE / "metrics/latency.json",
    ]
    for path in evidence_paths:
        external[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    for relative, expected in external.items():
        path = Path(relative)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"external transitive source drift: {relative}")
    local = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in local_preseal_files()
    }
    transitive = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "files": dict(sorted({**external, **local}.items())),
        "local_normative_frozen_and_qualification_file_count": len(local),
        "external_transitive_file_count": len(external),
        "all_hashes_verified": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(
        ROOT / "transitive_source_package_manifest.json", transitive, exclusive=True
    )
    return transitive


def main() -> None:
    assert_runtime_contract("generation")
    seal_path = ROOT / "audit/pre_v5_seal.json"
    if seal_path.exists():
        raise RuntimeError("V5 package v002 is already sealed")
    no_outcome_data()
    source_hashes = verify_accepted_cycle()
    power = read_json(ROOT / "power_analysis.json")
    cohort = read_json(ROOT / "cohort_seed_ledger.json")
    dgp = read_json(ROOT / "DGP.json")
    if (
        not power["passed"]
        or power["selected_v5_sample_size"] != 1600
        or len(cohort["roles"]["package_smoke"]) != 12
        or len(cohort["roles"]["v5_confirmation"]) != 1600
        or cohort["v5_outcome_episodes"] != 0
        or dgp["v5_confirmation_episode_count"] != 1600
    ):
        raise RuntimeError("power, cohort, or DGP qualification drift")
    qualifications = [
        ROOT / "audit/preseal_tests.json",
        ROOT / "audit/runtime_qualification.json",
        ROOT / "audit/inference_qualification.json",
        ROOT / "audit/generation_runtime_qualification.json",
        ROOT / "audit/generation_negative_path_qualification.json",
        ROOT / "audit/identifier_freshness_verification.json",
        ROOT / "audit/package_source_qualification.json",
        ROOT / "audit/flop_derivation_graph.json",
        ROOT / "audit/flop_derivation_symbolic.json",
    ]
    for path in qualifications:
        if not path.exists() or not read_json(path)["passed"]:
            raise RuntimeError(f"missing or failed preseal qualification: {path}")
    required = set(read_json(ROOT / "expected_artifact_roles.json")["pre_v5_required"])
    missing = sorted(relative for relative in required if not (ROOT / relative).exists())
    if missing:
        raise RuntimeError(f"normative V5 components missing: {missing}")
    versions = package_versions()
    write_semantic_and_operation_artifacts()
    write_frozen_candidate(source_hashes)
    transitive = build_transitive()
    files = dict(transitive["files"])
    transitive_path = ROOT / "transitive_source_package_manifest.json"
    files[str(transitive_path.relative_to(REPO_ROOT))] = sha256_file(transitive_path)
    seal = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "status": "frozen_before_any_v002_package_smoke_or_v5_confirmation_episode",
        "package_version": "v002",
        "accepted_discovery_cycle": 7,
        "accepted_discovery_source_hashes": source_hashes,
        "selected_candidate_id": "stage_dual_r0.01_q0.85",
        "package_versions": versions,
        "package_smoke_episodes_at_seal": 0,
        "v5_confirmation_episode_count": 1600,
        "v5_outcome_episodes_at_seal": 0,
        "power_analysis_passed": True,
        "normative_code_complete": True,
        "dual_interpreter_contract_sha256": sha256_file(ROOT / "runtime_contract.json"),
        "carry_forward_manifest_sha256": sha256_file(
            ROOT / "carry_forward_manifest.json"
        ),
        "wrong_interpreter_negative_path_passed": True,
        "generation_global_preflight_qualified": True,
        "identifier_freshness_independently_verified": True,
        "post_seal_normative_patch_forbidden": True,
        "smoke_failure_requires_new_package_version": True,
        "files": dict(sorted(files.items())),
    }
    atomic_json(seal_path, seal, exclusive=True)
    print(json.dumps({"seal_sha256": sha256_file(seal_path), **seal}, sort_keys=True))


if __name__ == "__main__":
    main()
