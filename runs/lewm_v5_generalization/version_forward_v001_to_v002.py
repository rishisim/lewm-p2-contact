#!/usr/bin/env python3
"""Preserve v001 and activate v002 after a zero-outcome external Git change."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = Path(__file__).resolve().parent
V001 = STUDY_ROOT / "attempts/v001"
V002 = STUDY_ROOT / "attempts/v002"
ORIGINAL = "d20ad79c4db7f564f5007513634bec4b9a785605"
OBSERVED = "86e4bb150de0d1546a3cf57f3ef301294eb04368"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise RuntimeError(f"immutable artifact already exists: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def command(*arguments: str) -> str:
    return subprocess.run(
        list(arguments),
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def assignments(path: Path, names: set[str]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    return found


def function_ast_hash(path: Path, names: set[str]) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    encoded = ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> None:
    output = V002 / "audit/version_forward_equivalence.json"
    if output.exists():
        print(output.read_text(encoding="utf-8"))
        return
    invalid = json.loads(
        (V001 / "audit/bootstrap_audit.json").read_text(encoding="utf-8")
    )
    state_path = STUDY_ROOT / "STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    observed = command("git", "rev-parse", "HEAD")
    changed = command("git", "diff", "--name-status", f"{ORIGINAL}..{observed}")
    changed_records = [
        {"status": line.split("\t", 1)[0], "path": line.split("\t", 1)[1]}
        for line in changed.splitlines()
        if line
    ]

    def operational_only(record: dict[str, str]) -> bool:
        path = record["path"]
        return (
            path == ".gitignore"
            or path == "docs/V5_NOVELTY_LITERATURE_REVIEW.md"
            or path.startswith("runs/lewm_v5_readiness_program/")
            or path
            == "runs/lewm_adaptive_compute_critic_compression/cache/stagewise_train_outputs.npz"
        )

    scientific_names = {
        "REGIMES",
        "SMOKE_EPISODES_PER_REGIME",
        "TARGET_EPISODES_PER_REGIME",
        "REPLACEMENTS_PER_REGIME",
        "ROWS_PER_EPISODE",
        "BOOTSTRAP_REPLICATES",
        "FAMILYWISE_ALPHA",
        "CO_PRIMARY_ENDPOINTS",
        "LATENT_DIM",
        "ACTION_DIM",
        "HISTORY_LEN",
        "FEATURE_DIM",
        "BASE_FLOPS",
        "V1_FLOPS",
        "ADAPTER_FLOPS",
        "GATE_FEATURE_FLOPS",
        "GATE_HEAD_FLOPS",
        "GATE_TOTAL_FLOPS",
        "GATE_NONFLOP_OPS",
        "NUMERICAL_RTOL",
        "NUMERICAL_ATOL",
        "NUMERICAL_MAX_ABS",
        "V5_REQUIRED_HASHES",
    }
    scientific_functions = {
        "array_sha256",
        "assert_runtime_contract",
        "load_v5_runner",
        "role_count",
        "sequential_calls",
    }
    v1_assignments = assignments(V001 / "study_common.py", scientific_names)
    v2_assignments = assignments(V002 / "study_common.py", scientific_names)
    v1_function_hash = function_ast_hash(
        V001 / "study_common.py", scientific_functions
    )
    v2_function_hash = function_ast_hash(
        V002 / "study_common.py", scientific_functions
    )
    v5_objects = {
        relative: sha256_file(
            REPO_ROOT
            / "runs/lewm_v5_readiness_program/v5_package_versions/v004"
            / relative
        )
        for relative in (
            "freeze/compiled_gate.npz",
            "freeze/gate_fit.npz",
            "freeze/whitening.npz",
            "freeze/gate_freeze.json",
            "freeze/gate_freeze_seal.json",
            "operation_ledger.json",
            "numerical_equivalence_contract.json",
        )
    }
    checks = {
        "v001_failure_zero_outcome": invalid.get("passed") is False
        and invalid["checks"]["generalization_target_outcomes_absent"],
        "state_reports_zero_target_generated": state.get(
            "target_outcome_episodes_generated"
        )
        == 0,
        "state_reports_zero_target_executed": state.get(
            "target_outcome_episodes_executed"
        )
        == 0,
        "target_arrays_never_opened": state.get(
            "target_outcomes_opened_for_analysis"
        )
        is False,
        "observed_head_exact": observed == OBSERVED,
        "all_git_delta_paths_operational": all(
            operational_only(record) for record in changed_records
        ),
        "v5_reverification_was_exact": all(
            invalid["checks"][name]
            for name in (
                "required_named_hashes_exact",
                "all_preseal_files_rehashed",
                "all_transitive_sources_rehashed",
                "all_package_manifest_files_rehashed",
                "all_external_runtime_files_rehashed",
                "all_1600_raw_episodes_rehashed",
            )
        ),
        "scientific_constant_objects_identical": v1_assignments
        == v2_assignments,
        "scientific_function_ast_identical": v1_function_hash
        == v2_function_hash,
        "v5_frozen_objects_still_exact": v5_objects
        == {
            "freeze/compiled_gate.npz": "1eb9253073dbc6aad52c1589ac6b0643fec2aa5388cb0c7fd474a5f0114c8d57",
            "freeze/gate_fit.npz": "d13cf89bc3d5f22d89612c30dd555bb50aa392bd465eba335e0917cbe860ff97",
            "freeze/whitening.npz": "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796",
            "freeze/gate_freeze.json": "11fedb55363539aea683d2960224b1e6b3d946f52027e6d18952a2ad4e480c3d",
            "freeze/gate_freeze_seal.json": "90b44528119b576d742d91572eab0f442dca748e5966b47a3a156cb02dd6bb36",
            "operation_ledger.json": "f1857aa43fb5d460dc493ea9aaf03b0cb192129ed3bbace7808e5d95d8806617",
            "numerical_equivalence_contract.json": "698b8adfba230562b405f1e57d618fe68434147a604820641177399f9934ddbd",
        },
    }
    payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "classification": "zero_outcome_procedural_invalidity",
        "repair_scope": "operational Git identity acceptance only",
        "original_required_head": ORIGINAL,
        "observed_operational_head": observed,
        "external_reflog_event": (
            "commit Add V5 readiness packages and literature review"
        ),
        "changed_paths": changed_records,
        "checks": checks,
        "passed": all(checks.values()),
        "source_comparison": {
            "v001_study_common_sha256": sha256_file(
                V001 / "study_common.py"
            ),
            "v002_study_common_sha256": sha256_file(
                V002 / "study_common.py"
            ),
            "scientific_assignments": v2_assignments,
            "v001_scientific_function_ast_sha256": v1_function_hash,
            "v002_scientific_function_ast_sha256": v2_function_hash,
            "intentional_non_scientific_changes": {
                "ATTEMPT_VERSION": ["v001", "v002"],
                "EXPECTED_GIT_HEAD": [ORIGINAL, observed],
                "active_attempt_guard": ["v001", "v002"],
            },
        },
        "frozen_object_hashes": v5_objects,
        "scientific_design_changed": False,
        "target_outcomes_consumed": 0,
    }
    atomic_json(output, payload, exclusive=True)
    if not payload["passed"]:
        raise RuntimeError(f"version-forward equivalence failed: {payload}")

    now = time.time_ns()
    for attempt in state["attempt_history"]:
        if attempt["version"] == "v001":
            attempt["status"] = "invalid_zero_outcome_external_git_change"
            attempt["invalidity_evidence_path"] = str(
                (V001 / "audit/bootstrap_audit.json").relative_to(REPO_ROOT)
            )
            attempt["invalidity_evidence_sha256"] = sha256_file(
                V001 / "audit/bootstrap_audit.json"
            )
    state["attempt_history"].append(
        {
            "version": "v002",
            "path": str(V002.relative_to(REPO_ROOT)),
            "status": "active_zero_outcome_version_forward",
            "created_unix_ns": now,
            "version_forward_evidence_path": str(output.relative_to(REPO_ROOT)),
            "version_forward_evidence_sha256": sha256_file(output),
        }
    )
    state.update(
        {
            "active_attempt": "v002",
            "active_attempt_path": str(V002.relative_to(REPO_ROOT)),
            "expected_git_head": observed,
            "current_state": "BOOTSTRAP_AUDIT",
            "completed_states": [],
            "last_verified_checkpoint": {
                "name": "v001_zero_outcome_invalid_v002_operationally_equivalent",
                "evidence_path": str(output.relative_to(REPO_ROOT)),
                "evidence_sha256": sha256_file(output),
                "created_unix_ns": now,
            },
            "next_action": "complete v002 bootstrap audit",
            "updated_unix_ns": now,
        }
    )
    atomic_json(state_path, state)
    ledger_path = STUDY_ROOT / "RESEARCH_LEDGER.jsonl"
    with ledger_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "created_unix_ns": now,
                    "attempt": "v002",
                    "event": "zero_outcome_version_forward",
                    "from_attempt": "v001",
                    "reason": "external Git HEAD changed during bootstrap",
                    "scientific_design_changed": False,
                    "evidence_path": str(output.relative_to(REPO_ROOT)),
                    "evidence_sha256": sha256_file(output),
                },
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
