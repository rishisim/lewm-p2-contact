#!/usr/bin/env python3
"""Seal every cycle-006 design/code path before fit or selection generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from cycle_common import (
    REPO_ROOT,
    ROLE_BASE_SEEDS,
    ROLE_COUNTS,
    ROOT,
    SOURCE_DISCOVERY,
    atomic_json,
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
        "scope": "all parseable prior run JSON and filenames outside unsealed cycle 006",
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
                "episode_id": f"c006-{role}-{slot:03d}",
                "env_seed": base + slot,
                "policy_seed": base + 100_000 + slot,
                "oracle_np_seed": base + 200_000 + slot,
            }
            for slot in range(count)
        ]
    replacement_base = 1_916_090_000
    roles["replacement"] = [
        {
            "role": "replacement",
            "slot": slot,
            "episode_id": f"c006-replacement-{slot:03d}",
            "env_seed": replacement_base + slot,
            "policy_seed": replacement_base + 100_000 + slot,
            "oracle_np_seed": replacement_base + 200_000 + slot,
        }
        for slot in range(100)
    ]
    analysis_seeds = {
        "bootstrap_seed": 2_007_999_991,
        "histogram_seed": 2_007_888_881,
        "seeded_mixture_seed": 2_007_777_771,
        "selection_histogram_seed": 2_007_188_881,
        "selection_seeded_mixture_seed": 2_007_177_771,
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
            f"cycle-006 identifier overlap: numbers={numeric_overlap[:5]}, strings={string_overlap[:5]}"
        )
    expected_numbers = 3 * sum(len(records) for records in roles.values()) + len(
        analysis_seeds
    )
    if len(numbers) != expected_numbers or len(strings) != sum(
        len(records) for records in roles.values()
    ):
        raise RuntimeError("cycle-006 identifiers are not internally unique")
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
        "freshness": "every fit selection smoke prospective and replacement ID and all three RNG streams are disjoint from every recorded prior run",
        "fit_selection_prospective_roles_disjoint": True,
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


def main() -> None:
    seal_path = ROOT / "audit/design_seal.json"
    if seal_path.exists():
        raise RuntimeError("cycle-006 design already sealed")
    for role in ROLE_COUNTS:
        if (ROOT / f"data/{role}_raw_manifest.json").exists() or any(
            (ROOT / f"data/{role}_raw").glob("*")
        ):
            raise RuntimeError(f"{role} data exists before design seal")
    if (ROOT / "freeze/compiled_gate.npz").exists():
        raise RuntimeError("gate exists before fit/selection")
    expected_sources = verify_expected_sources()
    cohort = build_cohort()
    versions = package_versions()

    current_files = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_file()
        and path.suffix in {".py", ".json", ".md"}
        and path.name
        not in {
            "cohort_seed_ledger.json",
            "transitive_source_package_manifest.json",
            "artifact_manifest.json",
            "decision.json",
            "analysis_result.json",
            "REPORT.md",
        }
    )
    required_names = {
        "analyze.py",
        "candidate_grid.json",
        "compile_gate.py",
        "counted_features.py",
        "cycle_common.py",
        "derive_flops_graph.py",
        "derive_flops_symbolic.py",
        "finalize.py",
        "fit_select_gate.py",
        "FIT_SELECTION_PREREGISTRATION.md",
        "independent_verify.py",
        "input_loader.py",
        "latency.py",
        "outcome_mapping.json",
        "prepare.py",
        "prepare_design.py",
        "preseal_tests.py",
        "PREREGISTRATION.md",
        "protocol.json",
        "qualify_gate_fit.py",
        "qualify_runtime.py",
        "qualify_simultaneous.py",
        "runner.py",
        "verify_design.py",
        "verify_final.py",
        "verify_gate_freeze.py",
        "verify_pre_data.py",
    }
    missing = sorted(required_names - {path.name for path in current_files})
    if missing:
        raise RuntimeError(f"normative design code missing: {missing}")
    prior_transitive = json.loads(
        (
            ROOT.parent
            / "cycle_004_threshold_recalibration/transitive_source_package_manifest.json"
        ).read_text()
    )
    external = {
        **prior_transitive["local_and_frozen_files"],
        **prior_transitive["installed_controlling_sources"],
    }
    external = {
        relative: expected
        for relative, expected in external.items()
        if relative
        not in {
            "runs/lewm_adaptive_compute_planoracle_native_discovery/freeze/gate_contract.npz",
            "runs/lewm_adaptive_compute_planoracle_native_discovery/freeze/gate_weights.npz",
        }
    }
    external.update(
        {
            str((SOURCE_DISCOVERY / "fit_only_normalization_whitening.npz").relative_to(REPO_ROOT)): expected_sources[
                "whitening"
            ]
        }
    )
    for relative, expected in external.items():
        path = Path(relative)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"external transitive source drift: {relative}")
    local = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in current_files}
    transitive = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "files": dict(sorted({**external, **local}.items())),
        "local_normative_file_count": len(local),
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
        "status": "frozen_before_any_cycle_006_fit_selection_smoke_or_prospective_episode",
        "created_unix_ns": time.time_ns(),
        "files": dict(sorted(seal_files.items())),
        "sealed_file_count": len(seal_files),
        "expected_source_hashes": expected_sources,
        "package_versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "cohort_seed_ledger_sha256": sha256_file(ROOT / "cohort_seed_ledger.json"),
        "candidate_grid_sha256": sha256_file(ROOT / "candidate_grid.json"),
        "fit_episodes_at_seal": 0,
        "selection_episodes_at_seal": 0,
        "smoke_episodes_at_seal": 0,
        "prospective_episodes_at_seal": 0,
        "v5_outcome_episodes": 0,
        "all_generation_evaluation_fit_selection_sparse_analysis_audit_and_manifest_code_materialized": True,
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
