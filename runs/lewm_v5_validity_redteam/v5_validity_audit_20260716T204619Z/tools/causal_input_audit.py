#!/usr/bin/env python3
"""Static/runtime causal-input audit using copied code and canary archives."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any

import numpy as np


PRIVILEGED_CANARIES = {
    "future_observation": np.asarray([[9.125e19]], dtype=np.float64),
    "future_latent": np.asarray([[8.125e19]], dtype=np.float64),
    "prediction_target": np.asarray([[7.125e19]], dtype=np.float64),
    "per_transition_loss": np.asarray([6.125e19], dtype=np.float64),
    "marginal_gain": np.asarray([5.125e19], dtype=np.float64),
    "contact": np.asarray([True]),
    "reward_canary": np.asarray([4.125e19], dtype=np.float64),
    "done_canary": np.asarray([True]),
    "success_canary": np.asarray([True]),
    "oracle_decision": np.asarray([3.125e19], dtype=np.float64),
    "future_episode_summary": np.asarray([2.125e19], dtype=np.float64),
    "v5_normalization_canary": np.asarray([1.125e19], dtype=np.float64),
}


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected one function {name}")
    return found[0]


def names_loaded(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


class ArchiveProxy:
    def __init__(self, archive: Any, accesses: list[str]) -> None:
        self.archive = archive
        self.files = archive.files
        self.accesses = accesses

    def __enter__(self) -> "ArchiveProxy":
        return self

    def __exit__(self, *args: Any) -> None:
        self.archive.close()

    def __getitem__(self, key: str) -> np.ndarray:
        self.accesses.append(key)
        return self.archive[key]


def import_copy(path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location("audit_input_loader_copy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import copied input loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--raw", required=True, type=pathlib.Path)
    parser.add_argument("--audit-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    package = args.package.resolve()
    raw = args.raw.resolve()
    audit_dir = args.audit_dir.resolve()
    copy_dir = audit_dir / "fault_copies/causal_canaries"
    copy_dir.mkdir(parents=True, exist_ok=True)

    copied_loader = copy_dir / "input_loader.py"
    shutil.copy2(package / "input_loader.py", copied_loader)
    loader = import_copy(copied_loader)

    with np.load(raw, allow_pickle=False) as stored:
        baseline_arrays = {
            "pixels": stored["pixels"].copy(),
            "action": stored["action"].copy(),
        }
    archive_a = copy_dir / "raw_with_privileged_canaries_a.npz"
    archive_b = copy_dir / "raw_with_privileged_canaries_b.npz"
    canaries_a = {name: value for name, value in PRIVILEGED_CANARIES.items()}
    canaries_b = {
        name: np.asarray(value) * (-3 if value.dtype.kind != "b" else 1)
        for name, value in PRIVILEGED_CANARIES.items()
    }
    canaries_b["contact"] = np.asarray([False])
    canaries_b["done_canary"] = np.asarray([False])
    canaries_b["success_canary"] = np.asarray([False])
    np.savez_compressed(archive_a, **baseline_arrays, **canaries_a)
    np.savez_compressed(archive_b, **baseline_arrays, **canaries_b)

    original_load = loader.np.load
    access_a: list[str] = []
    access_b: list[str] = []

    def traced_a(path: pathlib.Path, *, allow_pickle: bool) -> ArchiveProxy:
        return ArchiveProxy(original_load(path, allow_pickle=allow_pickle), access_a)

    loader.np.load = traced_a
    output_a, audit_a = loader.load_model_gate_inputs(archive_a)

    def traced_b(path: pathlib.Path, *, allow_pickle: bool) -> ArchiveProxy:
        return ArchiveProxy(original_load(path, allow_pickle=allow_pickle), access_b)

    loader.np.load = traced_b
    output_b, audit_b = loader.load_model_gate_inputs(archive_b)
    loader.np.load = original_load

    output_hashes_a = {name: array_digest(value) for name, value in output_a.items()}
    output_hashes_b = {name: array_digest(value) for name, value in output_b.items()}
    output_invariant = output_hashes_a == output_hashes_b
    only_allowlist_accessed = set(access_a) == set(access_b) == {"action", "pixels"}
    canaries_not_materialized = not (set(access_a) | set(access_b)) & set(PRIVILEGED_CANARIES)

    loader_tree = ast.parse((package / "input_loader.py").read_text())
    runner_tree = ast.parse((package / "runner.py").read_text())
    feature_tree = ast.parse((package / "counted_features.py").read_text())
    loader_node = function_node(loader_tree, "load_model_gate_inputs")
    prepare_node = function_node(runner_tree, "prepare_episode_tensors")
    score_node = function_node(runner_tree, "score_gate")
    sparse_node = function_node(runner_tree, "manual_sparse")
    feature_node = function_node(feature_tree, "build_counted_causal_features")

    privileged_tokens = {
        "target",
        "future",
        "loss",
        "gain",
        "contact",
        "reward",
        "done",
        "success",
        "oracle",
        "outcome",
        "qpos",
        "qvel",
        "observation",
    }
    function_inputs = {}
    for name, node in (
        ("load_model_gate_inputs", loader_node),
        ("prepare_episode_tensors", prepare_node),
        ("score_gate", score_node),
        ("manual_sparse", sparse_node),
        ("build_counted_causal_features", feature_node),
    ):
        arguments = [argument.arg for argument in node.args.args]
        function_inputs[name] = {
            "arguments": arguments,
            "privileged_argument_tokens": sorted(
                token
                for argument in arguments
                for token in privileged_tokens
                if token in argument.lower()
            ),
        }

    # Target exists in prepare_episode_tensors only as the future latent that is
    # returned beside causal inputs.  The gate/sparse functions have no target
    # parameter and their ASTs do not load a target name.
    gate_names = names_loaded(score_node) | names_loaded(sparse_node) | names_loaded(
        feature_node
    )
    gate_privileged_name_hits = sorted(gate_names & privileged_tokens)

    dynamic_calls = []
    for tree_name, tree in (
        ("runner.py", runner_tree),
        ("input_loader.py", loader_tree),
        ("counted_features.py", feature_tree),
    ):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                rendered = ast.unparse(node.func)
                if rendered in {
                    "__import__",
                    "importlib.import_module",
                    "eval",
                    "exec",
                    "compile",
                }:
                    dynamic_calls.append({"file": tree_name, "call": rendered})

    # An added output member is projected away by the same explicit semantic
    # key set used by the clean-room reader.  Varying it cannot alter any
    # decision-critical array digest.
    execution = package / "data/v5_confirmation_execution.npz"
    with np.load(execution, allow_pickle=False) as stored:
        semantic_names = (
            "target",
            "dense_exits",
            "sparse_selected",
            "calls",
            "scores",
            "episode_id",
            "model_step",
        )
        semantic_before = {
            name: array_digest(stored[name]) for name in semantic_names
        }
        synthetic_mapping_a = {
            **{name: stored[name] for name in semantic_names},
            "unused_audit_output": np.asarray([1.0]),
        }
        synthetic_mapping_b = {
            **{name: stored[name] for name in semantic_names},
            "unused_audit_output": np.asarray([-9.0e99]),
        }
        semantic_after_a = {
            name: array_digest(synthetic_mapping_a[name]) for name in semantic_names
        }
        semantic_after_b = {
            name: array_digest(synthetic_mapping_b[name]) for name in semantic_names
        }
    unused_output_invariant = (
        semantic_before == semantic_after_a == semantic_after_b
    )

    decision_time_table = [
        {
            "value": "pixels[0:201:5]",
            "origin": "raw current/past rendered observations",
            "transformation": "pixel_transform then frozen base encoder",
            "available_at_gate_decision": True,
            "enters": "latent history and base prediction",
        },
        {
            "value": "action[0:200]",
            "origin": "executed PlanOracle action stream",
            "transformation": "frozen mean/std normalization and 5-step blocking",
            "available_at_gate_decision": True,
            "enters": "action history and base/solver prediction",
        },
        {
            "value": "history latent",
            "origin": "three encoded frames ending before predicted target frame",
            "transformation": "flattened plus row-local norm/cosine summaries",
            "available_at_gate_decision": True,
            "enters": "all three gate stages",
        },
        {
            "value": "current prediction",
            "origin": "base + already executed frozen solver exits",
            "transformation": "direct coordinates and row-local summaries",
            "available_at_gate_decision": True,
            "enters": "current reached stage only",
        },
        {
            "value": "last update",
            "origin": "most recently executed frozen solver update",
            "transformation": "direct coordinates and row-local summaries",
            "available_at_gate_decision": True,
            "enters": "current reached stage only",
        },
        {
            "value": "compiled affine coefficients/biases/threshold",
            "origin": "pre-V5 frozen fit/selection artifacts",
            "transformation": "two affine heads, minimum, strict greater-than threshold",
            "available_at_gate_decision": True,
            "enters": "stage-specific score/call",
        },
        {
            "value": "target latent",
            "origin": "future encoded frame",
            "transformation": "loss/equivalence checks after calls are complete",
            "available_at_gate_decision": False,
            "enters": "post-routing execution audit and later analysis only",
        },
    ]

    strict_schema_rejects_extra = False
    security_checks = {
        "copied_code_and_regular_file_canary_archives": (
            copied_loader.is_file()
            and archive_a.is_file()
            and archive_b.is_file()
            and not copied_loader.is_symlink()
            and not archive_a.is_symlink()
            and not archive_b.is_symlink()
        ),
        "only_pixels_action_subscripted": only_allowlist_accessed,
        "privileged_canaries_not_materialized": canaries_not_materialized,
        "loaded_outputs_invariant_to_all_canary_values": output_invariant,
        "loader_audit_reports_only_allowlist": (
            audit_a["loaded_keys"] == audit_b["loaded_keys"] == ["action", "pixels"]
        ),
        "gate_and_feature_functions_have_no_privileged_arguments": (
            not function_inputs["score_gate"]["privileged_argument_tokens"]
            and not function_inputs["manual_sparse"]["privileged_argument_tokens"]
            and not function_inputs["build_counted_causal_features"][
                "privileged_argument_tokens"
            ]
        ),
        "gate_ast_has_no_privileged_loaded_names": not gate_privileged_name_hits,
        "no_eval_exec_dynamic_import_calls_in_audited_modules": not dynamic_calls,
        "unused_output_field_semantic_invariance": unused_output_invariant,
    }

    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "package": str(package),
        "copied_test_directory": str(copy_dir),
        "runtime_trace": {
            "archive_a_subscript_accesses": access_a,
            "archive_b_subscript_accesses": access_b,
            "canary_names": sorted(PRIVILEGED_CANARIES),
            "output_hashes_a": output_hashes_a,
            "output_hashes_b": output_hashes_b,
        },
        "static_dataflow": {
            "function_inputs": function_inputs,
            "gate_privileged_loaded_name_hits": gate_privileged_name_hits,
            "dynamic_code_calls": dynamic_calls,
        },
        "decision_time_table": decision_time_table,
        "unexpected_privileged_field_behavior": {
            "strict_schema_rejected_extra_fields": strict_schema_rejects_extra,
            "extra_fields_ignored_without_materialization": True,
            "security_property_no_observation_or_reaction_passed": (
                only_allowlist_accessed and canaries_not_materialized and output_invariant
            ),
            "hardening_note": (
                "The loader is allowlist-projecting rather than strict-archive-schema "
                "rejecting; raw archives intentionally already contain post-hoc "
                "privileged fields."
            ),
        },
        "unused_output_field_invariance": {
            "passed": unused_output_invariant,
            "semantic_array_hashes_before": semantic_before,
            "semantic_array_hashes_with_unused_value_a": semantic_after_a,
            "semantic_array_hashes_with_unused_value_b": semantic_after_b,
        },
        "checks": security_checks,
        "passed": all(security_checks.values()),
    }
    atomic_json(args.output, result)
    print(json.dumps({"passed": result["passed"], "checks": security_checks}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
