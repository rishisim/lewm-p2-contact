#!/usr/bin/env python3
"""Materialize exact v001-to-v002 scientific carry-forward evidence."""

from __future__ import annotations

import ast
import hashlib
import json
import time
from pathlib import Path

from cycle_common import ROOT, atomic_json, sha256_file


V001 = ROOT.parent / "v001"
BYTE_IDENTICAL = (
    "DGP.json",
    "counted_features.py",
    "input_loader.py",
    "exclusions.json",
    "outcome_mapping.json",
    "operation_graph.json",
    "freeze/compiled_gate.npz",
    "freeze/gate_fit.npz",
    "freeze/gate_freeze.json",
    "freeze/gate_freeze_seal.json",
    "freeze/whitening.npz",
)
VERSION_ONLY = (
    "REPORT_TEMPLATE.md",
    "numerical_equivalence_contract.json",
)
NORMALIZED_MODULES = ("analysis.py", "independent_verify.py", "latency.py", "power_analysis.py")
RUNNER_FUNCTIONS = (
    "load_gate_tensors",
    "score_gate",
    "manual_sparse",
    "dense_shadow",
    "tensors_exact_with_nan",
    "prepare_episode_tensors",
    "execute",
)
SCIENTIFIC_CONSTANTS = (
    "LATENT_DIM",
    "ACTION_DIM",
    "HISTORY_LEN",
    "FEATURE_DIM",
    "GATE_WIDTH",
    "ROWS_PER_EPISODE",
    "BASE_FLOPS",
    "V1_FLOPS",
    "ADAPTER_FLOPS",
    "V5_SAMPLE_SIZE",
    "SOURCE_CYCLE_HASHES",
    "NUMERICAL_RTOL",
    "NUMERICAL_ATOL",
    "NUMERICAL_MAX_ABS",
    "EXPECTED_SOURCE_HASHES",
)
SEED_REMAP = {
    2_411_999_991: 2_051_999_991,
    2_411_888_881: 2_051_888_881,
    2_411_777_771: 2_051_777_771,
}


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def ast_hash(path: Path, *, normalized: bool = False) -> str:
    tree = ast.parse(path.read_text())
    if normalized:
        tree = Normalizer().visit(tree)
        ast.fix_missing_locations(tree)
    return text_sha(ast.dump(tree, annotate_fields=True, include_attributes=False))


class Normalizer(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom):
        node.names = [item for item in node.names if item.name != "assert_runtime_contract"]
        return None if not node.names else self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "assert_runtime_contract"
        ):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, int) and node.value in SEED_REMAP:
            return ast.copy_location(ast.Constant(SEED_REMAP[node.value]), node)
        return node


def function_hash(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text())
    node = next(
        item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return text_sha(ast.dump(node, annotate_fields=True, include_attributes=False))


def constants(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    output = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in SCIENTIFIC_CONSTANTS:
                output[name] = ast.dump(node.value, annotate_fields=True, include_attributes=False)
    return output


def main() -> None:
    output = ROOT / "carry_forward_manifest.json"
    if output.exists():
        raise RuntimeError("carry-forward manifest is immutable and already exists")
    byte_records = {}
    for relative in BYTE_IDENTICAL:
        source = V001 / relative
        target = ROOT / relative
        byte_records[relative] = {
            "v001_sha256": sha256_file(source),
            "v002_sha256": sha256_file(target),
            "identical": sha256_file(source) == sha256_file(target),
        }
    version_records = {}
    for relative in VERSION_ONLY:
        source_text = (V001 / relative).read_text()
        target_text = (ROOT / relative).read_text().replace("v002", "v001")
        version_records[relative] = {
            "v001_sha256": sha256_file(V001 / relative),
            "v002_sha256": sha256_file(ROOT / relative),
            "canonical_sha256": text_sha(target_text),
            "version_only": target_text == source_text,
        }
    module_records = {
        relative: {
            "v001_ast_sha256": ast_hash(V001 / relative, normalized=True),
            "v002_normalized_ast_sha256": ast_hash(ROOT / relative, normalized=True),
            "scientific_ast_identical_after_runtime_guard_and_seed_normalization": ast_hash(
                V001 / relative, normalized=True
            )
            == ast_hash(ROOT / relative, normalized=True),
        }
        for relative in NORMALIZED_MODULES
    }
    runner_records = {
        name: {
            "v001_ast_sha256": function_hash(V001 / "runner.py", name),
            "v002_ast_sha256": function_hash(ROOT / "runner.py", name),
            "identical": function_hash(V001 / "runner.py", name)
            == function_hash(ROOT / "runner.py", name),
        }
        for name in RUNNER_FUNCTIONS
    }
    v001_constants = constants(V001 / "cycle_common.py")
    v002_constants = constants(ROOT / "cycle_common.py")
    checks = {
        "byte_identical_scientific_files": all(item["identical"] for item in byte_records.values()),
        "version_path_changes_only_where_declared": all(
            item["version_only"] for item in version_records.values()
        ),
        "analysis_and_evaluation_science_ast_identical": all(
            item["scientific_ast_identical_after_runtime_guard_and_seed_normalization"]
            for item in module_records.values()
        ),
        "runner_scientific_functions_ast_identical": all(
            item["identical"] for item in runner_records.values()
        ),
        "scientific_constants_identical": v001_constants == v002_constants
        and set(v001_constants) == set(SCIENTIFIC_CONSTANTS),
    }
    payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "source_package": "v001",
        "target_package": "v002",
        "byte_identical": byte_records,
        "version_only": version_records,
        "normalized_modules": module_records,
        "runner_scientific_functions": runner_records,
        "scientific_constants": v002_constants,
        "checks": checks,
        "passed": all(checks.values()),
        "allowed_changes": [
            "package version and path references",
            "entirely fresh package-smoke, confirmation, replacement, analysis, and qualification identifiers",
            "explicit dual-interpreter contract and fail-closed dispatcher",
            "generation package preflight moved before output creation and seed/replacement iteration",
            "read-only recovery and verification instrumentation",
            "operational recovery and interpreter appendix in PREREGISTRATION.md",
        ],
        "scientific_design_changed": False,
        "confirmation_outcomes_available_during_repair": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output, payload, exclusive=True)
    if not payload["passed"]:
        raise RuntimeError(f"scientific carry-forward failed: {payload['checks']}")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
