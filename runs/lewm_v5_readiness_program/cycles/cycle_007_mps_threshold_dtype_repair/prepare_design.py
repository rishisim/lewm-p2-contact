#!/usr/bin/env python3
"""Seal the cycle-007 one-line MPS repair before any new episode."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from cycle_common import (
    REPO_ROOT,
    ROLE_BASE_SEEDS,
    ROLE_COUNTS,
    ROOT,
    SOURCE_CYCLE,
    SOURCE_CYCLE_HASHES,
    atomic_json,
    read_json,
    sha256_file,
    verify_expected_sources,
)


def recursive_integers(value: Any) -> set[int]:
    output: set[int] = set()
    if isinstance(value, bool):
        return output
    if isinstance(value, int):
        output.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            output.update(recursive_integers(item))
    elif isinstance(value, list):
        for item in value:
            output.update(recursive_integers(item))
    return output


def recursive_strings(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, str):
        output.add(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            output.add(str(key))
            output.update(recursive_strings(item))
    elif isinstance(value, list):
        for item in value:
            output.update(recursive_strings(item))
    return output


def digest(values: set[Any]) -> str:
    return hashlib.sha256(
        ("\n".join(str(item) for item in sorted(values, key=str)) + "\n").encode()
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
            strings.add(path.name)
            strings.add(path.stem)
    snapshot = {
        "schema_version": 1,
        "scope": "all parseable prior run JSON and filenames outside unsealed cycle 007",
        "json_file_count": len(scanned),
        "numeric_identifier_count": len(integers),
        "string_identifier_count": len(strings),
        "numeric_identifier_set_sha256": digest(integers),
        "string_identifier_set_sha256": digest(strings),
        "scanned_path_set_sha256": digest(set(scanned)),
    }
    return integers, strings, snapshot


def build_cohort() -> dict[str, Any]:
    prior_numbers, prior_strings, snapshot = prior_identifiers()
    atomic_json(ROOT / "audit/prior_identifier_snapshot.json", snapshot, exclusive=True)
    roles = {}
    for role, count in ROLE_COUNTS.items():
        base = ROLE_BASE_SEEDS[role]
        roles[role] = [
            {
                "role": role,
                "slot": slot,
                "episode_id": f"c007-{role}-{slot:03d}",
                "env_seed": base + slot,
                "policy_seed": base + 100_000 + slot,
                "oracle_np_seed": base + 200_000 + slot,
            }
            for slot in range(count)
        ]
    replacement_base = 1_917_090_000
    roles["replacement"] = [
        {
            "role": "replacement",
            "slot": slot,
            "episode_id": f"c007-replacement-{slot:03d}",
            "env_seed": replacement_base + slot,
            "policy_seed": replacement_base + 100_000 + slot,
            "oracle_np_seed": replacement_base + 200_000 + slot,
        }
        for slot in range(100)
    ]
    analysis_seeds = {
        "bootstrap_seed": 2_017_999_991,
        "histogram_seed": 2_017_888_881,
        "seeded_mixture_seed": 2_017_777_771,
    }
    numbers = {
        item[key]
        for records in roles.values()
        for item in records
        for key in ("env_seed", "policy_seed", "oracle_np_seed")
    } | set(analysis_seeds.values())
    strings = {item["episode_id"] for records in roles.values() for item in records}
    numeric_overlap = sorted(numbers & prior_numbers)
    string_overlap = sorted(strings & prior_strings)
    if numeric_overlap or string_overlap:
        raise RuntimeError(
            f"cycle-007 identifier overlap: numbers={numeric_overlap[:5]}, strings={string_overlap[:5]}"
        )
    expected_numbers = 3 * sum(len(records) for records in roles.values()) + len(
        analysis_seeds
    )
    if len(numbers) != expected_numbers or len(strings) != sum(
        len(records) for records in roles.values()
    ):
        raise RuntimeError("cycle-007 identifiers are not internally unique")
    ledger = {
        "schema_version": 1,
        "roles": roles,
        **analysis_seeds,
        "all_new_numeric_identifiers_unique": True,
        "all_new_string_identifiers_unique": True,
        "prior_identifier_snapshot": snapshot,
        "overlap_with_prior_recorded_numeric_identifiers": numeric_overlap,
        "overlap_with_prior_recorded_string_identifiers": string_overlap,
        "replacement_rule": "mechanical generation exception or malformed rollout only; never outcome loss contact reward or success",
        "freshness": "every cycle-007 smoke prospective and replacement ID and all three RNG streams are disjoint from every recorded prior run",
        "source_fit_selection_are_consumed_and_not_regenerated": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / "cohort_seed_ledger.json", ledger, exclusive=True)
    return ledger


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
    if any(values[name] != expected[name] for name in expected):
        raise RuntimeError(f"design environment package drift: {values}")
    return values


def verify_carry_forward() -> dict[str, Any]:
    source_paths = {
        "decision": SOURCE_CYCLE / "decision.json",
        "artifact_manifest": SOURCE_CYCLE / "artifact_manifest.json",
        "compiled_gate": SOURCE_CYCLE / "freeze/compiled_gate.npz",
        "gate_fit": SOURCE_CYCLE / "freeze/gate_fit.npz",
        "gate_freeze": SOURCE_CYCLE / "freeze/gate_freeze.json",
        "gate_freeze_seal": SOURCE_CYCLE / "freeze/gate_freeze_seal.json",
    }
    observed = {name: sha256_file(path) for name, path in source_paths.items()}
    if observed != SOURCE_CYCLE_HASHES:
        raise RuntimeError(f"cycle-006 source artifact drift: {observed}")
    decision = read_json(source_paths["decision"])
    if (
        decision["terminal_outcome"] != "cycle_execution_invalid"
        or decision["failure_stage"]
        != "post_selection_preseal_runtime_qualification_before_smoke"
        or decision["prospective_episodes_consumed"] != 0
        or decision["v5_outcome_episodes"] != 0
    ):
        raise RuntimeError("cycle-006 invalidation provenance drift")
    copied = {
        "data/fit_evaluated.npz": "ad326f419bc7ae584b5cb4605da589e6e3a426d5c057f23d37089e8b95c17906",
        "data/fit_evaluated_manifest.json": "0ce73bbd327fa2f7f7b7710e67593972dfabb46e8e0804d7f67f886e263c20d7",
        "data/fit_raw_manifest.json": "a02840cd927bc0cc71984ae3f6de9a5386293d683bb5ac6b0d8f6cc714b98ed2",
        "data/selection_evaluated.npz": "e025d591192d6541c960b48f3d5cf40ecfc46bdb9edce52367d53c4c7b217e34",
        "data/selection_evaluated_manifest.json": "3fce2abf238f6ee8f055bcdf91e2f551a9a3289e573b19f013a71254897af5b9",
        "data/selection_raw_manifest.json": "f52110c782e122b0eea9e824aca486f28564b4081207b8ba6850f332590dea79",
        "development/fit_lock.json": "e98aae792d97a3cd57c16df6f1cdc2bd74d533afe08ceb887a71e4bc4dfd42aa",
        "development/fitted_candidates.npz": "f4bffc40059f5890cf57b95891c62887d4b3b0ac7e58689e2a2d4aba3e75aab8",
        "development/gate_selection_ledger.json": "3bfb1ee62fce7dacce490132a2c6d32ea31ecf47be9eaec6fe7df1f46f67a520",
        "freeze/compiled_gate.npz": SOURCE_CYCLE_HASHES["compiled_gate"],
        "freeze/gate_fit.npz": SOURCE_CYCLE_HASHES["gate_fit"],
        "freeze/gate_freeze.json": SOURCE_CYCLE_HASHES["gate_freeze"],
        "freeze/gate_freeze_seal.json": SOURCE_CYCLE_HASHES["gate_freeze_seal"],
    }
    bad = [relative for relative, expected in copied.items() if sha256_file(ROOT / relative) != expected]
    if bad:
        raise RuntimeError(f"carried artifact is not an exact copy: {bad}")
    fit = read_json(ROOT / "data/fit_evaluated_manifest.json")
    selection = read_json(ROOT / "data/selection_evaluated_manifest.json")
    freeze = read_json(ROOT / "freeze/gate_freeze.json")
    if fit["episode_count"] != 240 or selection["episode_count"] != 120:
        raise RuntimeError("consumed fit/selection count drift")
    if fit["loaded_input_keys"] != ["action", "pixels"] or selection[
        "loaded_input_keys"
    ] != ["action", "pixels"]:
        raise RuntimeError("consumed fit/selection allowlist drift")
    if freeze["selected"]["candidate_id"] != "stage_dual_r0.01_q0.85":
        raise RuntimeError("selected candidate drift")
    with np.load(ROOT / "freeze/compiled_gate.npz", allow_pickle=False) as stored:
        float64_thresholds = stored["thresholds"].astype(np.float64)
    float32_thresholds = float64_thresholds.astype(np.float32)
    record = {
        "schema_version": 1,
        "status": "exact_frozen_candidate_carried_from_consumed_cycle_006",
        "source_cycle": str(SOURCE_CYCLE.relative_to(REPO_ROOT)),
        "source_hashes": observed,
        "copied_artifact_hashes": copied,
        "selected_candidate_id": "stage_dual_r0.01_q0.85",
        "numeric_thresholds_float64_storage": float64_thresholds.tolist(),
        "runtime_thresholds_float32": float32_thresholds.tolist(),
        "maximum_absolute_threshold_cast_change": float(
            np.max(np.abs(float64_thresholds - float32_thresholds.astype(np.float64)))
        ),
        "sole_executable_repair": "materialize threshold comparison tensor as float32 instead of unsupported MPS float64",
        "new_fit_episodes": 0,
        "new_selection_episodes": 0,
        "consumed_fit_episodes": 240,
        "consumed_selection_episodes": 120,
        "source_smoke_episodes": 0,
        "source_prospective_episodes": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(ROOT / "carry_forward_manifest.json", record, exclusive=True)
    return record


def main() -> None:
    seal_path = ROOT / "audit/design_seal.json"
    if seal_path.exists():
        raise RuntimeError("cycle-007 design already sealed")
    for role in ROLE_COUNTS:
        if (ROOT / f"data/{role}_raw_manifest.json").exists() or any(
            (ROOT / f"data/{role}_raw").glob("*")
        ):
            raise RuntimeError(f"new {role} data exists before design seal")
    expected_sources = verify_expected_sources()
    carry = verify_carry_forward()
    cohort = build_cohort()
    versions = package_versions()

    excluded = {
        "cohort_seed_ledger.json",
        "transitive_source_package_manifest.json",
        "artifact_manifest.json",
        "decision.json",
        "analysis_result.json",
        "REPORT.md",
    }
    current_files = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_file()
        and path.suffix in {".py", ".json", ".md"}
        and path.name not in excluded
    )
    required_names = {
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
    }
    missing = sorted(required_names - {path.name for path in current_files})
    if missing:
        raise RuntimeError(f"normative repair code missing: {missing}")

    source_transitive = read_json(SOURCE_CYCLE / "transitive_source_package_manifest.json")
    external = {
        relative: expected
        for relative, expected in source_transitive["files"].items()
        if "/cycles/cycle_006_stage_specific_gate_refit/" not in relative
    }
    external.update(
        {
            str((SOURCE_CYCLE / "decision.json").relative_to(REPO_ROOT)): SOURCE_CYCLE_HASHES[
                "decision"
            ],
            str(
                (SOURCE_CYCLE / "artifact_manifest.json").relative_to(REPO_ROOT)
            ): SOURCE_CYCLE_HASHES["artifact_manifest"],
        }
    )
    for relative, expected in external.items():
        path = Path(relative)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"external transitive source drift: {relative}")

    carried_paths = [
        ROOT / relative for relative in carry["copied_artifact_hashes"]
    ]
    local = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in [*current_files, *carried_paths]
    }
    transitive = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "files": dict(sorted({**external, **local}.items())),
        "local_normative_and_frozen_file_count": len(local),
        "external_transitive_file_count": len(external),
        "all_hashes_verified": True,
    }
    atomic_json(ROOT / "transitive_source_package_manifest.json", transitive, exclusive=True)
    seal_files = dict(transitive["files"])
    for path in (
        ROOT / "cohort_seed_ledger.json",
        ROOT / "transitive_source_package_manifest.json",
        ROOT / "audit/prior_identifier_snapshot.json",
    ):
        seal_files[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    seal = {
        "schema_version": 1,
        "status": "frozen_before_any_cycle_007_smoke_or_prospective_episode",
        "created_unix_ns": time.time_ns(),
        "files": dict(sorted(seal_files.items())),
        "sealed_file_count": len(seal_files),
        "expected_source_hashes": expected_sources,
        "package_versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "cohort_seed_ledger_sha256": sha256_file(ROOT / "cohort_seed_ledger.json"),
        "carry_forward_manifest_sha256": sha256_file(ROOT / "carry_forward_manifest.json"),
        "candidate_grid_sha256": sha256_file(ROOT / "candidate_grid.json"),
        "new_fit_episodes_at_seal": 0,
        "new_selection_episodes_at_seal": 0,
        "carried_consumed_fit_episodes": 240,
        "carried_consumed_selection_episodes": 120,
        "smoke_episodes_at_seal": 0,
        "prospective_episodes_at_seal": 0,
        "v5_outcome_episodes": 0,
        "all_generation_sparse_analysis_audit_and_manifest_code_materialized": True,
        "sole_repair": carry["sole_executable_repair"],
        "no_postseal_normative_code_patch_allowed": True,
        "cohort_roles": {role: len(records) for role, records in cohort["roles"].items()},
    }
    atomic_json(seal_path, seal, exclusive=True)
    print(
        json.dumps(
            {
                "status": seal["status"],
                "seal_sha256": sha256_file(seal_path),
                "sealed_file_count": len(seal_files),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
