#!/usr/bin/env python3
"""Materialize the post-selection package and freeze it before smoke/prospective data."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

from counted_features import semantic_feature_names
from cycle_common import (
    ADAPTER_FLOPS,
    BASE_FLOPS,
    PROGRAM_ROOT,
    REPO_ROOT,
    ROOT,
    V1_FLOPS,
    atomic_json,
    read_json,
    sha256_file,
    verify_expected_sources,
)
from verify_design import verify as verify_design
from verify_gate_freeze import verify as verify_gate_freeze


def package_versions() -> dict[str, str | None]:
    values = {}
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
        raise RuntimeError(f"DGP package version mismatch: {values}")
    return values


def main() -> None:
    seal_path = ROOT / "audit/pre_data_seal.json"
    if seal_path.exists():
        raise RuntimeError("cycle 007 prospective package is already sealed")
    design = verify_design()
    gate = verify_gate_freeze()
    expected_sources = verify_expected_sources()
    required = {
        "audit/preseal_tests.json": "passed",
        "audit/gate_fit_qualification.json": "passed",
        "audit/runtime_qualification.json": "passed",
        "audit/flop_derivation_graph.json": "passed",
        "audit/flop_derivation_symbolic.json": "passed",
        "audit/simultaneous_inference_qualification.json": "passed",
        "numerical_equivalence_contract.json": "passed",
    }
    for relative, field in required.items():
        if not read_json(ROOT / relative).get(field):
            raise RuntimeError(f"preseal qualification failed: {relative}")
    graph = read_json(ROOT / "audit/flop_derivation_graph.json")
    symbolic = read_json(ROOT / "audit/flop_derivation_symbolic.json")
    if graph["totals"] != symbolic["totals"]:
        raise RuntimeError("operation-count derivations disagree")
    totals = graph["totals"]
    if totals != {
        "feature_flops_per_reached_evaluation": 3801,
        "dual_affine_score_flops_per_reached_evaluation": 4184,
        "total_flops_per_reached_evaluation": 7985,
        "nonflop_comparison_min_per_reached_evaluation": 5,
    }:
        raise RuntimeError(f"unexpected operation total: {totals}")
    for role in ("smoke", "prospective"):
        if (ROOT / f"data/{role}_raw_manifest.json").exists() or any(
            (ROOT / f"data/{role}_raw").glob("*")
        ):
            raise RuntimeError(f"{role} data exists before pre-data seal")
    fit_manifest = read_json(ROOT / "data/fit_evaluated_manifest.json")
    selection_manifest = read_json(ROOT / "data/selection_evaluated_manifest.json")
    carry = read_json(ROOT / "carry_forward_manifest.json")
    if fit_manifest["episode_count"] != 240 or selection_manifest["episode_count"] != 120:
        raise RuntimeError("fit/selection cohort count drift")
    if fit_manifest["loaded_input_keys"] != ["action", "pixels"] or selection_manifest[
        "loaded_input_keys"
    ] != ["action", "pixels"]:
        raise RuntimeError("fit/selection input allowlist drift")
    if carry["new_fit_episodes"] != 0 or carry["new_selection_episodes"] != 0:
        raise RuntimeError("cycle-007 carry-forward isolation drift")
    runtime_qualification = read_json(ROOT / "audit/runtime_qualification.json")
    gate_qualification = read_json(ROOT / "audit/gate_fit_qualification.json")
    if gate_qualification.get("runtime_threshold_dtype") != "float32":
        raise RuntimeError("threshold dtype repair was not qualified")
    if sorted(runtime_qualification.get("tested_devices", [])) != ["cpu", "mps"]:
        raise RuntimeError("both CPU and MPS runtime qualification are required")

    names = semantic_feature_names(latent_dim=192, action_dim=25, history_len=3)
    atomic_json(
        ROOT / "semantic_feature_names.json",
        {
            "schema_version": 1,
            "implementation": "counted_features.semantic_feature_names",
            "feature_count": len(names),
            "names": list(names),
            "stage_specific_coefficient_selection_adds_no_feature": True,
        },
        exclusive=True,
    )
    operation = {
        "schema_version": 1,
        "convention": "subtraction multiplication reduction-add sqrt division and affine bias-add count as FLOPs; comparison clamp and minimum are separately reported non-FLOPs; detach view flatten concatenate index scatter and stage coefficient selection are zero FLOP",
        "base_flops_per_row": BASE_FLOPS,
        "mandatory_depth1_refiner_flops_per_row": V1_FLOPS,
        "additional_refiner_flops_per_call": ADAPTER_FLOPS,
        "feature_graph_path": "operation_graph.json",
        "semantic_feature_names_path": "semantic_feature_names.json",
        "gate": totals,
        "formula": "N*(base+mandatory_depth1) + (sum(calls)-N)*additional_refiner + sum(min(calls,3))*gate_total",
        "analytic_equivalent_calls": "sum(calls) + sum(min(calls,3))*7985/264960",
        "nonflop_operations_reported_separately": True,
    }
    atomic_json(ROOT / "operation_ledger.json", operation, exclusive=True)
    expected_roles = {
        "schema_version": 1,
        "fit": "240 consumed cycle-006 episodes carried only as immutable gate-fit provenance; no cycle-007 refit",
        "selection": "120 consumed cycle-006 episodes carried only as immutable one-shot selection provenance; no cycle-007 reselection or refit",
        "smoke": "12 fresh excluded implementation-validation episodes",
        "prospective": "exactly 300 fresh episodes used once for discovery inference",
        "dense_execution": "comparator and numerical shadow only",
        "actual_adaptive_execution": "manual sparse stage loop",
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / "expected_artifact_roles.json", expected_roles, exclusive=True)

    versions = package_versions()
    transitive = read_json(ROOT / "transitive_source_package_manifest.json")
    for relative, expected in transitive["files"].items():
        path = Path(relative)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"transitive source drift: {relative}")
    normative_names = {
        "analyze.py",
        "build_numerical_contract.py",
        "candidate_grid.json",
        "compile_gate.py",
        "counted_features.py",
        "cycle_common.py",
        "derive_flops_graph.py",
        "derive_flops_symbolic.py",
        "finalize.py",
        "independent_verify.py",
        "input_loader.py",
        "invalidate_cycle.py",
        "latency.py",
        "numerical_equivalence_contract.json",
        "outcome_mapping.json",
        "prepare.py",
        "prepare_design.py",
        "preseal_tests.py",
        "PREREGISTRATION.md",
        "protocol.json",
        "qualify_gate_fit.py",
        "qualify_runtime.py",
        "qualify_simultaneous.py",
        "REPORT_TEMPLATE.md",
        "runner.py",
        "verify_design.py",
        "verify_final.py",
        "verify_gate_freeze.py",
        "verify_pre_data.py",
        "carry_forward_manifest.json",
    }
    missing = sorted(name for name in normative_names if not (ROOT / name).exists())
    if missing:
        raise RuntimeError(f"normative source missing before seal: {missing}")
    direct_paths = [ROOT / name for name in sorted(normative_names)]
    direct_paths += [
        ROOT / "cohort_seed_ledger.json",
        ROOT / "carry_forward_manifest.json",
        ROOT / "operation_graph.json",
        ROOT / "operation_ledger.json",
        ROOT / "semantic_feature_names.json",
        ROOT / "expected_artifact_roles.json",
        ROOT / "transitive_source_package_manifest.json",
        ROOT / "audit/design_seal.json",
        ROOT / "freeze/gate_freeze_seal.json",
        ROOT / "freeze/gate_freeze.json",
        ROOT / "freeze/gate_fit.npz",
        ROOT / "freeze/compiled_gate.npz",
        ROOT / "development/fit_lock.json",
        ROOT / "development/gate_selection_ledger.json",
        ROOT / "audit/preseal_tests.json",
        ROOT / "audit/gate_fit_qualification.json",
        ROOT / "audit/runtime_qualification.json",
        ROOT / "audit/flop_derivation_graph.json",
        ROOT / "audit/flop_derivation_symbolic.json",
        ROOT / "audit/simultaneous_inference_qualification.json",
        ROOT / "data/fit_raw_manifest.json",
        ROOT / "data/fit_evaluated.npz",
        ROOT / "data/fit_evaluated_manifest.json",
        ROOT / "data/selection_raw_manifest.json",
        ROOT / "data/selection_evaluated.npz",
        ROOT / "data/selection_evaluated_manifest.json",
    ]
    files = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in direct_paths}
    files.update(transitive["files"])
    seal = {
        "schema_version": 1,
        "status": "frozen_before_any_cycle_007_smoke_or_prospective_episode",
        "created_unix_ns": time.time_ns(),
        "files": dict(sorted(files.items())),
        "sealed_file_count": len(files),
        "design_seal_sha256": design["seal_sha256"],
        "gate_freeze_seal_sha256": gate["seal_sha256"],
        "expected_source_hashes": expected_sources,
        "package_versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "new_fit_episodes_at_seal": 0,
        "new_selection_episodes_at_seal": 0,
        "carried_consumed_fit_episodes": 240,
        "carried_consumed_selection_episodes": 120,
        "smoke_episodes_at_seal": 0,
        "prospective_episodes_at_seal": 0,
        "v5_outcome_episodes": 0,
        "normative_code_complete_before_smoke": True,
        "actual_prospective_analysis_and_terminal_mapping_sealed": True,
        "no_postseal_normative_patch_allowed": True,
    }
    atomic_json(seal_path, seal, exclusive=True)
    print(
        json.dumps(
            {
                "seal_sha256": sha256_file(seal_path),
                "sealed_file_count": len(files),
                "status": seal["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
