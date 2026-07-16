#!/usr/bin/env python3
"""Read-only independent verifier of frozen science and v001 carry-forward."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from cycle_common import ROOT, read_json, sha256_file
from verify_runtime_contract import verify as verify_runtime_contract


V001 = ROOT.parent / "v001"
SEED_REMAP = {
    2_411_999_991: 2_051_999_991,
    2_411_888_881: 2_051_888_881,
    2_411_777_771: 2_051_777_771,
}


def digest_ast(node: ast.AST) -> str:
    value = ast.dump(node, annotate_fields=True, include_attributes=False).encode()
    return hashlib.sha256(value).hexdigest()


def named_function(path: Path, name: str) -> ast.AST:
    module = ast.parse(path.read_text())
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one function {name} in {path}")
    return matches[0]


def canonical_module(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            node.value = SEED_REMAP.get(node.value, node.value)
        if isinstance(node, ast.ImportFrom):
            node.names = [alias for alias in node.names if alias.name != "assert_runtime_contract"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = [
                statement
                for statement in node.body
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == "assert_runtime_contract"
                )
            ]
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and not node.names)
    ]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def verify() -> dict:
    manifest = read_json(ROOT / "carry_forward_manifest.json")
    byte_checks = {
        relative: sha256_file(V001 / relative) == sha256_file(ROOT / relative)
        for relative in manifest["byte_identical"]
    }
    version_checks = {
        relative: (ROOT / relative).read_text().replace("v002", "v001")
        == (V001 / relative).read_text()
        for relative in manifest["version_only"]
    }
    function_checks = {
        name: digest_ast(named_function(V001 / "runner.py", name))
        == digest_ast(named_function(ROOT / "runner.py", name))
        for name in manifest["runner_scientific_functions"]
    }
    module_checks = {
        relative: canonical_module(V001 / relative) == canonical_module(ROOT / relative)
        for relative in manifest["normalized_modules"]
    }
    frozen_hashes = {
        "freeze/compiled_gate.npz": "1eb9253073dbc6aad52c1589ac6b0643fec2aa5388cb0c7fd474a5f0114c8d57",
        "freeze/gate_fit.npz": "d13cf89bc3d5f22d89612c30dd555bb50aa392bd465eba335e0917cbe860ff97",
        "freeze/whitening.npz": "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796",
    }
    frozen_checks = {
        relative: sha256_file(ROOT / relative) == expected
        for relative, expected in frozen_hashes.items()
    }
    analysis = (ROOT / "analysis.py").read_text()
    independent = (ROOT / "independent_verify.py").read_text()
    seed_checks = {
        "analysis_bootstrap": "BOOTSTRAP_SEED = 2_411_999_991" in analysis,
        "analysis_histogram": "HISTOGRAM_SEED = 2_411_888_881" in analysis,
        "analysis_mixture": "SEEDED_MIXTURE_SEED = 2_411_777_771" in analysis,
        "independent_bootstrap": "np.random.default_rng(2_411_999_991)" in independent,
        "independent_histogram": "np.random.default_rng(2_411_888_881)" in independent,
        "independent_mixture": "np.random.default_rng(2_411_777_771)" in independent,
    }
    runtime = verify_runtime_contract()
    checks = {
        "builder_manifest_passed": manifest["passed"] is True,
        "byte_carry_forward": all(byte_checks.values()),
        "version_only_carry_forward": all(version_checks.values()),
        "runner_scientific_functions": all(function_checks.values()),
        "analysis_and_evaluation_science": all(module_checks.values()),
        "frozen_object_hashes": all(frozen_checks.values()),
        "fresh_analysis_seeds_agree_in_two_implementations": all(seed_checks.values()),
        "runtime_contract": runtime["passed"],
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "byte_checks": byte_checks,
        "version_checks": version_checks,
        "function_checks": function_checks,
        "module_checks": module_checks,
        "frozen_checks": frozen_checks,
        "seed_checks": seed_checks,
        "carry_forward_manifest_sha256": sha256_file(ROOT / "carry_forward_manifest.json"),
        "runtime_contract_sha256": runtime["runtime_contract_sha256"],
        "read_only": True,
        "v5_outcome_episodes": 0,
    }
    if not result["passed"]:
        raise RuntimeError(f"package source verification failed: {result}")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
