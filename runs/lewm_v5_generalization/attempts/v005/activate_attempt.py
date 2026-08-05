#!/usr/bin/env python3
"""Version forward after the zero-target-outcome v004 smoke predicate bug."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from study_common import (
    ATTEMPT_ROOT,
    EXPECTED_GIT_HEAD,
    REPO_ROOT,
    STUDY_ROOT,
    append_ledger,
    atomic_json,
    read_json,
    relative_to_repo,
    sha256_file,
)


V004 = ATTEMPT_ROOT.parent / "v004"
ATTEMPT_TOKENS = ("v001", "v002", "v003", "v004", "v005")


def command(*parts: str) -> str:
    completed = subprocess.run(
        list(parts),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {parts}\n{completed.stdout}{completed.stderr}"
        )
    return completed.stdout.strip()


class OperationalNormalizer(ast.NodeTransformer):
    """Normalize attempt labels, fresh seeds, and the one repaired predicate."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            value = node.value
            for version in ATTEMPT_TOKENS:
                value = value.replace(version, "<ATTEMPT>")
            node.value = value
        elif isinstance(node.value, int) and not isinstance(node.value, bool):
            if 3_410_000_000 <= node.value <= 3_450_001_000:
                node.value = 3_000_000_000 + node.value % 10_000_000
            elif 3_460_000_000 <= node.value <= 3_500_001_000:
                node.value = 3_000_000_000 + node.value % 10_000_000
        return node

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        self.generic_visit(node)
        for index, (key, value) in enumerate(zip(node.keys, node.values)):
            old_count = (
                isinstance(key, ast.Constant)
                and key.value == "target_outcome_episodes"
                and isinstance(value, ast.Constant)
                and value.value == 0
            )
            repaired_boolean = (
                isinstance(key, ast.Constant)
                and key.value == "target_outcome_episodes_absent"
                and isinstance(value, ast.Constant)
                and value.value is True
            )
            if old_count or repaired_boolean:
                node.keys[index] = ast.Constant(
                    value="<NORMALIZED_ZERO_TARGET_ASSERTION>"
                )
                node.values[index] = ast.Constant(value=True)
        return node


def normalized_tree(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree = OperationalNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def load_source(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def assigned_dict(path: Path, function_name: str, variable_name: str) -> ast.Dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == variable_name
            and isinstance(node.value, ast.Dict)
        ):
            return node.value
    raise RuntimeError(
        f"missing {variable_name} dict in {function_name}: {path}"
    )


def dict_value(node: ast.Dict, key_name: str) -> ast.AST | None:
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == key_name:
            return value
    return None


def literal_is(node: ast.AST | None, expected: Any) -> bool:
    if node is None:
        return False
    try:
        return ast.literal_eval(node) == expected
    except (ValueError, TypeError):
        return False


def main() -> None:
    failure_path = (
        V004 / "audit/smoke_checkpoint_procedural_failure.json"
    )
    comparison_path = ATTEMPT_ROOT / "audit/version_forward_equivalence.json"
    state_path = STUDY_ROOT / "STATE.json"
    state = read_json(state_path)

    if state.get("active_attempt") == "v005":
        if not comparison_path.exists():
            raise RuntimeError("v005 active without version-forward evidence")
        print(json.dumps(read_json(comparison_path), sort_keys=True))
        return

    observed_head = command("git", "rev-parse", "HEAD")
    smoke_path = V004 / "audit/excluded_regime_smoke.json"
    smoke = read_json(smoke_path)
    smoke_checks = smoke["checks"]
    nonpredicate_smoke_checks = {
        key: value
        for key, value in smoke_checks.items()
        if key != "target_outcome_episodes"
    }
    zero_target_outcome = (
        state.get("active_attempt") == "v004"
        and state.get("current_state") == "EXCLUDED_REGIME_SMOKE"
        and state.get("target_outcome_episodes_generated") == 0
        and state.get("target_outcome_episodes_executed") == 0
        and state.get("target_outcomes_opened_for_analysis") is False
        and not (V004 / "data/target").exists()
        and not (V004 / "audit/target_generation_complete.json").exists()
        and not (V004 / "audit/target_execution_complete.json").exists()
        and not (V004 / "analysis_result.json").exists()
    )
    smoke_valid_except_predicate = (
        smoke.get("classification") == "fresh_excluded_regime_smoke"
        and smoke.get("passed") is False
        and smoke.get("smoke_episode_count") == 18
        and smoke.get("target_outcome_episodes") == 0
        and smoke_checks.get("target_outcome_episodes") == 0
        and all(value is True for value in nonpredicate_smoke_checks.values())
        and all(
            item.get("passed") is True
            for item in smoke.get("regimes", {}).values()
        )
    )
    if (
        observed_head != EXPECTED_GIT_HEAD
        or not zero_target_outcome
        or not smoke_valid_except_predicate
    ):
        raise RuntimeError(
            "v005 version-forward preconditions failed: "
            f"head={observed_head}, zero_target={zero_target_outcome}, "
            f"smoke_valid_except_predicate={smoke_valid_except_predicate}"
        )

    if failure_path.exists():
        failure = read_json(failure_path)
        if (
            failure.get("attempt") != "v004"
            or failure.get("version_forward_authorized") is not True
            or failure.get("target_outcome_episodes") != 0
        ):
            raise RuntimeError("existing v004 invalidity evidence is invalid")
    else:
        failure = {
            "schema_version": 1,
            "created_unix_ns": time.time_ns(),
            "attempt": "v004",
            "classification": "zero_target_outcome_procedural_invalidity",
            "failed_state": "EXCLUDED_REGIME_SMOKE",
            "exception_type": "RuntimeError",
            "exception_message": (
                "excluded smoke failed because all(checks.values()) treated "
                "the integer target-outcome count 0 as False"
            ),
            "cause": (
                "a count-bearing zero was placed inside a boolean check map; "
                "all 18 excluded smoke episodes, hashes, frozen-module checks, "
                "and sparse/dense numerical-equivalence checks passed"
            ),
            "repair": (
                "replace only the count-bearing check-map member with the "
                "boolean assertion target_outcome_episodes_absent=True; retain "
                "target_outcome_episodes=0 as a result field"
            ),
            "excluded_smoke_report_path": relative_to_repo(smoke_path),
            "excluded_smoke_report_sha256": sha256_file(smoke_path),
            "pre_outcome_seal_path": relative_to_repo(
                V004 / "audit/pre_outcome_seal.json"
            ),
            "pre_outcome_seal_sha256": sha256_file(
                V004 / "audit/pre_outcome_seal.json"
            ),
            "all_nonpredicate_smoke_checks_true": True,
            "all_three_regimes_passed": True,
            "smoke_episodes": 18,
            "smoke_outcomes_permanently_excluded": True,
            "target_outcome_episodes": 0,
            "target_outcomes_opened": False,
            "target_directories_absent": True,
            "scientific_result": None,
            "scientific_design_changed": False,
            "retry_same_attempt_forbidden": True,
            "fresh_v005_smoke_required": True,
            "version_forward_authorized": True,
        }
        atomic_json(failure_path, failure, exclusive=True)

    science_files = (
        "study_common.py",
        "state_init.py",
        "bootstrap_audit.py",
        "design_and_power.py",
        "generator.py",
        "runner.py",
        "checkpoints.py",
        "analysis.py",
        "latency.py",
        "independent_verify.py",
        "posthoc.py",
        "finalize.py",
        "launcher.py",
    )
    files = {}
    for name in science_files:
        old = V004 / name
        new = ATTEMPT_ROOT / name
        old_tree = normalized_tree(old)
        new_tree = normalized_tree(new)
        files[name] = {
            "v004_sha256": sha256_file(old),
            "v005_sha256": sha256_file(new),
            "normalized_ast_sha256_v004": hashlib.sha256(
                old_tree.encode()
            ).hexdigest(),
            "normalized_ast_sha256_v005": hashlib.sha256(
                new_tree.encode()
            ).hexdigest(),
            "normalized_ast_equal": old_tree == new_tree,
        }

    old_common = load_source(
        "generalization_v004_common_for_v005", V004 / "study_common.py"
    )
    new_common = load_source(
        "generalization_v005_common_for_v005",
        ATTEMPT_ROOT / "study_common.py",
    )
    scientific_names = (
        "REGIMES",
        "SMOKE_EPISODES_PER_REGIME",
        "TARGET_EPISODES_PER_REGIME",
        "REPLACEMENTS_PER_REGIME",
        "ROWS_PER_EPISODE",
        "BOOTSTRAP_REPLICATES",
        "FAMILYWISE_ALPHA",
        "CO_PRIMARY_ENDPOINTS",
        "SIMULTANEOUS_FAMILY_SIZE",
        "PER_CLAIM_ALPHA",
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
    )
    objects = {
        name: getattr(old_common, name) == getattr(new_common, name)
        for name in scientific_names
    }

    old_checks = assigned_dict(
        V004 / "checkpoints.py", "smoke_checkpoint", "checks"
    )
    new_checks = assigned_dict(
        ATTEMPT_ROOT / "checkpoints.py", "smoke_checkpoint", "checks"
    )
    old_result = assigned_dict(
        V004 / "checkpoints.py", "smoke_checkpoint", "result"
    )
    new_result = assigned_dict(
        ATTEMPT_ROOT / "checkpoints.py", "smoke_checkpoint", "result"
    )
    repair_source = {
        "v004_checkpoints_sha256": sha256_file(V004 / "checkpoints.py"),
        "v005_checkpoints_sha256": sha256_file(
            ATTEMPT_ROOT / "checkpoints.py"
        ),
        "v004_check_map_contains_integer_zero": literal_is(
            dict_value(old_checks, "target_outcome_episodes"), 0
        ),
        "v004_check_map_lacks_boolean_assertion": dict_value(
            old_checks, "target_outcome_episodes_absent"
        )
        is None,
        "v005_check_map_removes_integer_count": dict_value(
            new_checks, "target_outcome_episodes"
        )
        is None,
        "v005_check_map_contains_boolean_true": literal_is(
            dict_value(new_checks, "target_outcome_episodes_absent"), True
        ),
        "v004_result_retains_numeric_zero": literal_is(
            dict_value(old_result, "target_outcome_episodes"), 0
        ),
        "v005_result_retains_numeric_zero": literal_is(
            dict_value(new_result, "target_outcome_episodes"), 0
        ),
        "normalized_checkpoints_ast_equal": files["checkpoints.py"][
            "normalized_ast_equal"
        ],
    }
    comparison = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "source_attempt": "v004",
        "target_attempt": "v005",
        "classification": "zero_target_outcome_procedural_version_forward",
        "repair_scope": (
            "one smoke-checkpoint boolean predicate and fresh operational "
            "identifiers; scientific design and implementation unchanged"
        ),
        "invalidity_evidence_path": relative_to_repo(failure_path),
        "invalidity_evidence_sha256": sha256_file(failure_path),
        "excluded_smoke_report_sha256": sha256_file(smoke_path),
        "scientific_files": files,
        "scientific_constant_object_equality": objects,
        "repair_source": repair_source,
        "operational_identifier_ranges_changed": True,
        "fresh_v005_smoke_and_target_identifiers_required": True,
        "v004_smoke_data_reused_for_v005_inference": False,
        "target_outcomes_before_version_forward": 0,
        "target_outcomes_opened_before_version_forward": False,
        "scientific_result_before_version_forward": None,
        "scientific_design_changed": False,
        "passed": (
            all(item["normalized_ast_equal"] for item in files.values())
            and all(objects.values())
            and all(repair_source.values())
            and failure["version_forward_authorized"]
            and failure["target_outcome_episodes"] == 0
        ),
    }
    if not comparison["passed"]:
        raise RuntimeError(f"v004-v005 equivalence failed: {comparison}")
    if comparison_path.exists():
        existing = read_json(comparison_path)
        if existing.get("passed") is not True:
            raise RuntimeError("existing v005 comparison did not pass")
        comparison = existing
    else:
        atomic_json(comparison_path, comparison, exclusive=True)

    history = list(state["attempt_history"])
    if not history or history[-1].get("version") != "v004":
        raise RuntimeError("attempt history does not end at v004")
    history[-1]["status"] = (
        "invalid_zero_target_outcome_smoke_checkpoint_predicate"
    )
    history[-1]["invalidity_evidence_path"] = relative_to_repo(failure_path)
    history[-1]["invalidity_evidence_sha256"] = sha256_file(failure_path)
    history.append(
        {
            "version": "v005",
            "path": relative_to_repo(ATTEMPT_ROOT),
            "status": "active_zero_outcome_version_forward",
            "created_unix_ns": time.time_ns(),
            "version_forward_evidence_path": relative_to_repo(comparison_path),
            "version_forward_evidence_sha256": sha256_file(comparison_path),
        }
    )
    state.update(
        {
            "active_attempt": "v005",
            "active_attempt_path": relative_to_repo(ATTEMPT_ROOT),
            "attempt_history": history,
            "completed_states": ["BOOTSTRAP_AUDIT"],
            "current_state": "DESIGN_AND_POWER",
            "last_verified_checkpoint": {
                "name": "v004_smoke_predicate_versioned_to_v005",
                "evidence_path": relative_to_repo(comparison_path),
                "evidence_sha256": sha256_file(comparison_path),
                "created_unix_ns": time.time_ns(),
            },
            "next_action": "materialize v005 design with fresh identifiers",
            "target_outcome_episodes_generated": 0,
            "target_outcome_episodes_executed": 0,
            "target_outcomes_opened_for_analysis": False,
            "terminal_label": None,
            "process_valid": None,
            "updated_unix_ns": time.time_ns(),
        }
    )
    atomic_json(state_path, state)
    append_ledger(
        "zero_target_outcome_version_forward",
        from_attempt="v004",
        to_attempt="v005",
        reason=(
            "integer zero in excluded-smoke boolean check map despite all "
            "18 fresh excluded smoke and equivalence checks passing"
        ),
        evidence_path=relative_to_repo(comparison_path),
        evidence_sha256=sha256_file(comparison_path),
        scientific_design_changed=False,
        prior_smoke_episodes=18,
        prior_smoke_permanently_excluded=True,
        target_outcomes=0,
    )
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
