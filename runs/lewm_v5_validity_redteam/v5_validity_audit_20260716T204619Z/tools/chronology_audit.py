#!/usr/bin/env python3
"""Reconstruct v001-v004 and one-shot recovery chronology."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import subprocess
import time
from typing import Any


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def function_digest(path: pathlib.Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"function {name} missing from {path}")
    rendered = ast.dump(matches[0], annotate_fields=True, include_attributes=False)
    return hashlib.sha256(rendered.encode()).hexdigest()


def git(repo: pathlib.Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--program", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    program = args.program.resolve()
    versions = program / "v5_package_versions"
    v = {name: versions / name for name in ("v001", "v002", "v003", "v004")}

    v001_summary = read(program / "v5_confirmation_recovery/v001_execution_invalid.json")
    v001_failures = read(v["v001"] / "audit/v5_confirmation_generation_failures.json")
    v002_failure = read(v["v002"] / "audit/preseal_qualification_failure.json")
    v003_failure = read(v["v003"] / "audit/preseal_qualification_failure.json")
    preseal = read(v["v004"] / "audit/pre_v5_seal.json")
    ready = read(v["v004"] / "PACKAGE_READY.json")
    package_manifest = read(v["v004"] / "artifact_manifest.json")
    raw_manifest = read(v["v004"] / "data/v5_confirmation_raw_manifest.json")
    execution_manifest = read(
        v["v004"] / "data/v5_confirmation_execution_manifest.json"
    )
    input_seal = read(v["v004"] / "audit/v5_confirmation_input_seal.json")
    analysis = read(v["v004"] / "analysis_result.json")
    latency = read(v["v004"] / "metrics/v5_confirmation_latency.json")
    independent = read(v["v004"] / "audit/independent_verification.json")
    decision = read(v["v004"] / "decision.json")
    recovery_state = read(program / "v5_confirmation_recovery/STATE.json")
    pointer = read(program / "v5_confirmation_recovery/CURRENT_POINTER.json")

    failures = v001_failures["failures"]
    version_records = {
        "v001": {
            "status": "pre-outcome execution invalid",
            "failed_attempts": len(failures),
            "raw_confirmation_npz": len(
                list((v["v001"] / "data/v5_confirmation_raw").glob("*.npz"))
            ),
            "modeled_rows": 0,
            "outcomes": 0,
            "analysis_exists": (v["v001"] / "analysis_result.json").exists(),
            "decision_exists": (v["v001"] / "decision.json").exists(),
            "failure_sha256": sha256(
                v["v001"] / "audit/v5_confirmation_generation_failures.json"
            ),
        },
        "v002": {
            "status": v002_failure["status"],
            "preseal_created": v002_failure["pre_v5_seal_created"],
            "raw_confirmation_npz": len(
                list((v["v002"] / "data/v5_confirmation_raw").glob("*.npz"))
            ),
            "modeled_rows": v002_failure["modeled_row_count"],
            "outcomes": v002_failure["confirmation_outcome_count"],
            "analysis_exists": (v["v002"] / "analysis_result.json").exists(),
            "decision_exists": (v["v002"] / "decision.json").exists(),
        },
        "v003": {
            "status": v003_failure["status"],
            "preseal_created": v003_failure["pre_v5_seal_created"],
            "raw_confirmation_npz": len(
                list((v["v003"] / "data/v5_confirmation_raw").glob("*.npz"))
            ),
            "modeled_rows": v003_failure["modeled_row_count"],
            "outcomes": v003_failure["confirmation_outcome_count"],
            "analysis_exists": (v["v003"] / "analysis_result.json").exists(),
            "decision_exists": (v["v003"] / "decision.json").exists(),
        },
        "v004": {
            "status": decision["terminal_outcome"],
            "raw_confirmation_npz": len(
                list((v["v004"] / "data/v5_confirmation_raw").glob("*.npz"))
            ),
            "raw_confirmation_sidecars": len(
                list((v["v004"] / "data/v5_confirmation_raw").glob("*.json"))
            ),
            "modeled_rows": execution_manifest["rows"],
            "outcomes": decision["v5_outcome_episodes"],
            "analysis_exists": True,
            "decision_exists": True,
        },
    }

    raw_records = raw_manifest["episodes"]
    timestamps = {
        "pre_v5_seal": int(preseal["created_unix_ns"]),
        "package_ready": int(ready["created_unix_ns"]),
        "final_package_manifest": int(package_manifest["created_unix_ns"]),
        "first_raw_episode": min(int(item["created_unix_ns"]) for item in raw_records),
        "last_raw_episode": max(int(item["created_unix_ns"]) for item in raw_records),
        "raw_manifest": int(raw_manifest["created_unix_ns"]),
        "execution_manifest": int(execution_manifest["created_unix_ns"]),
        "input_seal": int(input_seal["created_unix_ns"]),
        "analysis": int(analysis["created_unix_ns"]),
        "latency": int(latency["created_unix_ns"]),
        "independent_verification": int(independent["created_unix_ns"]),
        "decision": int(decision["created_unix_ns"]),
    }
    expected_order = [
        "pre_v5_seal",
        "package_ready",
        "final_package_manifest",
        "first_raw_episode",
        "last_raw_episode",
        "raw_manifest",
        "execution_manifest",
        "input_seal",
        "analysis",
        "latency",
        "independent_verification",
        "decision",
    ]
    chronology_monotonic = all(
        timestamps[left] <= timestamps[right]
        for left, right in zip(expected_order, expected_order[1:])
    )
    strict_boundaries = {
        "seal_before_any_raw": timestamps["pre_v5_seal"]
        < timestamps["first_raw_episode"],
        "ready_before_any_raw": timestamps["package_ready"]
        < timestamps["first_raw_episode"],
        "final_manifest_before_any_raw": timestamps["final_package_manifest"]
        < timestamps["first_raw_episode"],
        "all_raw_before_raw_manifest": timestamps["last_raw_episode"]
        < timestamps["raw_manifest"],
        "raw_manifest_before_execution": timestamps["raw_manifest"]
        < timestamps["execution_manifest"],
        "execution_before_input_seal": timestamps["execution_manifest"]
        < timestamps["input_seal"],
        "input_seal_before_target_analysis": timestamps["input_seal"]
        < timestamps["analysis"],
        "analysis_before_independent_audit": timestamps["analysis"]
        < timestamps["independent_verification"],
        "independent_audit_before_decision": timestamps[
            "independent_verification"
        ]
        < timestamps["decision"],
    }

    scientific_byte_files = [
        "DGP.json",
        "counted_features.py",
        "input_loader.py",
        "operation_graph.json",
        "outcome_mapping.json",
        "exclusions.json",
        "freeze/compiled_gate.npz",
        "freeze/gate_fit.npz",
        "freeze/gate_freeze.json",
        "freeze/gate_freeze_seal.json",
        "freeze/whitening.npz",
    ]
    scientific_hashes = {}
    for relative in scientific_byte_files:
        hashes = {name: sha256(path / relative) for name, path in v.items()}
        scientific_hashes[relative] = {
            "hashes": hashes,
            "all_versions_identical": len(set(hashes.values())) == 1,
        }
    runner_functions = [
        "prepare_episode_tensors",
        "load_gate_tensors",
        "score_gate",
        "manual_sparse",
        "dense_shadow",
        "tensors_exact_with_nan",
        "execute",
    ]
    runner_ast = {}
    for name in runner_functions:
        digests = {
            version: function_digest(root / "runner.py", name)
            for version, root in v.items()
        }
        runner_ast[name] = {
            "digests": digests,
            "all_versions_identical": len(set(digests.values())) == 1,
        }

    raw_ids = [item["episode_id"] for item in raw_records]
    raw_slots = [int(item["slot"]) for item in raw_records]
    raw_paths = [item["path"] for item in raw_records]
    raw_npz_names = {
        str(path.relative_to(repo))
        for path in (v["v004"] / "data/v5_confirmation_raw").glob("*.npz")
    }
    raw_json_npz_names = {
        str(path.with_suffix(".npz").relative_to(repo))
        for path in (v["v004"] / "data/v5_confirmation_raw").glob("*.json")
    }
    no_partial_hidden = not [
        str(path)
        for path in (v["v004"] / "data/v5_confirmation_raw").iterdir()
        if path.name.startswith(".") or path.suffix not in {".npz", ".json"}
    ]
    v004_failure_ledgers = list(
        (v["v004"] / "audit").glob("v5_confirmation_*failure*.json")
    )

    current_head = git(repo, "rev-parse", "HEAD")
    execution_head = recovery_state["expected_git_head"]
    git_relationship = {
        "execution_head": execution_head,
        "current_head": current_head,
        "merge_base": git(repo, "merge-base", execution_head, current_head),
        "execution_head_subject": git(repo, "show", "-s", "--format=%s", execution_head),
        "current_head_subject": git(repo, "show", "-s", "--format=%s", current_head),
        "same_head": execution_head == current_head,
        "package_sources_hash_bound_despite_later_commit": True,
    }

    seal_file_hashes = preseal["files"]
    bad_seal_hashes = []
    for raw_path, expected in seal_file_hashes.items():
        path = pathlib.Path(raw_path)
        if not path.is_absolute():
            path = repo / path
        if not path.exists() or sha256(path) != expected:
            bad_seal_hashes.append(raw_path)

    checks = {
        "v001_zero_outcome_operational_failure": (
            version_records["v001"]["failed_attempts"] == 101
            and version_records["v001"]["raw_confirmation_npz"] == 0
            and not version_records["v001"]["analysis_exists"]
            and not version_records["v001"]["decision_exists"]
            and v001_summary["confirmation_outcome_count"] == 0
        ),
        "v002_zero_outcome_preseal_failure": (
            version_records["v002"]["outcomes"] == 0
            and version_records["v002"]["raw_confirmation_npz"] == 0
            and not version_records["v002"]["preseal_created"]
        ),
        "v003_zero_outcome_preseal_failure": (
            version_records["v003"]["outcomes"] == 0
            and version_records["v003"]["raw_confirmation_npz"] == 0
            and not version_records["v003"]["preseal_created"]
        ),
        "only_v004_has_completed_confirmation": (
            version_records["v004"]["raw_confirmation_npz"] == 1600
            and version_records["v004"]["raw_confirmation_sidecars"] == 1600
            and version_records["v004"]["modeled_rows"] == 60800
        ),
        "chronology_monotonic": chronology_monotonic,
        "all_strict_chronology_boundaries": all(strict_boundaries.values()),
        "scientific_byte_files_unchanged_v001_v004": all(
            item["all_versions_identical"] for item in scientific_hashes.values()
        ),
        "runner_scientific_functions_unchanged_v001_v004": all(
            item["all_versions_identical"] for item in runner_ast.values()
        ),
        "v004_preseal_hashes_still_match": not bad_seal_hashes,
        "raw_ids_exact_unique": len(set(raw_ids)) == len(raw_ids) == 1600,
        "raw_slots_exact_order": raw_slots == list(range(1600)),
        "raw_paths_exact_unique": len(set(raw_paths)) == len(raw_paths) == 1600,
        "zero_replacements": all(
            item["replacement_used"] is False
            and item["episode_id"] == item["seed_source_episode_id"]
            for item in raw_records
        ),
        "raw_npz_and_sidecar_path_sets_exact": (
            set(raw_paths) == raw_npz_names == raw_json_npz_names
        ),
        "no_partial_or_hidden_raw_artifacts": no_partial_hidden,
        "no_v004_confirmation_failure_ledger": not v004_failure_ledgers,
        "recovery_terminal_matches_decision": (
            recovery_state["terminal_label"] == decision["terminal_outcome"]
            and pointer["terminal_evidence"]["sha256"]
            == sha256(v["v004"] / "decision.json")
        ),
        "decision_known_sha256": sha256(v["v004"] / "decision.json")
        == "69bb8af8d80d7e18aeabc5430963c57b1bc91be603745d5a0d5f1a855d61fbe5",
        "independent_audit_known_sha256": sha256(
            v["v004"] / "audit/independent_verification.json"
        )
        == "d30e58466184ac98c5643ea6242e1fb1d95cc2f820c001be16aee3556903c3f0",
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "version_records": version_records,
        "timeline_unix_ns": timestamps,
        "timeline_order": expected_order,
        "strict_boundaries": strict_boundaries,
        "scientific_byte_hash_comparisons": scientific_hashes,
        "runner_function_ast_comparisons": runner_ast,
        "v004_preseal": {
            "sealed_file_count": len(seal_file_hashes),
            "bad_hashes": bad_seal_hashes,
            "seal_sha256": sha256(v["v004"] / "audit/pre_v5_seal.json"),
        },
        "cohort_recovery": {
            "replacement_count": sum(
                bool(item["replacement_used"]) for item in raw_records
            ),
            "failure_ledgers": [str(path) for path in v004_failure_ledgers],
            "raw_npz_count": len(raw_npz_names),
            "raw_sidecar_count": len(raw_json_npz_names),
        },
        "git_provenance": git_relationship,
        "residual_chronology_limitations": [
            "Filesystem and JSON timestamps are local mutable records, not externally notarized.",
            "The recovery controller is outside the v004 pre-V5 package seal, but it only records checkpoints; the performance path is the sealed launcher/package.",
            "A narrow hash-then-open TOCTOU window exists for each raw file; the complete post-execution input seal rehashes all raw files but cannot exclude an adversarial change-and-restore within that window.",
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }
    atomic_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "checks": checks}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
