#!/usr/bin/env python3
"""Fail-closed implementation qualification and prospective stage sealing.

This module never generates an episode and never loads a prior or prospective
outcome archive during pre-data qualification.  Its numerical checks use only
deterministic synthetic arrays.  The three authorization seals are created
exclusively and only after their complete prerequisite set passes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import analysis
import build_manifest
import compile_gate
import counted_features
import fit_select
import flops
import generator
import input_loader
import power_analysis
import study_common
from inherited_authorization import SOURCE_PRE_SELECTION_SEAL_PATH
from runtime_contract import (
    ATTEMPT_ROOT,
    REPO_ROOT,
    RuntimeContractError,
    require_mps_device,
    sha256_file,
    verify_dual_interpreter_contract,
    verify_here,
)


STUDY_ROOT = ATTEMPT_ROOT.parents[1]
STATE_PATH = STUDY_ROOT / "STATE.json"
ACTIVE_ATTEMPT = ATTEMPT_ROOT.name
if ACTIVE_ATTEMPT != "v008":
    raise RuntimeError(f"preseal.py must run from v008, got {ACTIVE_ATTEMPT!r}")
SCIENCE_ATTEMPT = "v001"
SOURCE_MANIFEST_PATH = ATTEMPT_ROOT / "audit/pre_data_source_manifest.json"
IMPLEMENTATION_REPORT_PATH = ATTEMPT_ROOT / "audit/implementation_complete.json"
QUALIFICATION_REPORT_PATH = ATTEMPT_ROOT / "audit/preseal_qualification.json"
PRE_DATA_SEAL_PATH = ATTEMPT_ROOT / "audit/pre_data_inheritance_seal.json"
PRE_SELECTION_MANIFEST_PATH = ATTEMPT_ROOT / "audit/pre_selection_manifest.json"
PRE_SELECTION_SEAL_PATH = SOURCE_PRE_SELECTION_SEAL_PATH
PRE_CONFIRMATION_MANIFEST_PATH = ATTEMPT_ROOT / "audit/pre_confirmation_manifest.json"
PRE_CONFIRMATION_SEAL_PATH = (
    ATTEMPT_ROOT / "audit/pre_confirmation_package_seal.json"
)
VERIFIER_CONTRACT_PATHS = (
    ATTEMPT_ROOT / "verifier_contract.json",
    ATTEMPT_ROOT / "verifier_contract_no_candidate.json",
    ATTEMPT_ROOT / "verifier_contract_power_infeasible.json",
)

OUTCOME_COUNT_FIELDS = (
    "fit_outcome_episodes",
    "selection_outcome_episodes",
    "smoke_outcome_episodes",
    "confirmation_outcome_episodes_generated",
    "confirmation_outcome_episodes_executed",
)
REQUIRED_IMPLEMENTATION_FILES = (
    "study_common.py",
    "input_loader.py",
    "counted_features.py",
    "flops.py",
    "generator.py",
    "fit_select.py",
    "compile_gate.py",
    "runner.py",
    "analysis.py",
    "power_analysis.py",
    "build_seed_ledger.py",
    "verify_identifier_freshness.py",
    "runtime_contract.py",
    "build_manifest.py",
    "preseal.py",
    "launcher.py",
    "independent_verify.py",
    "capture_verifier.py",
    "terminal_workflow.py",
    "version_forward_transaction.py",
    "workflow.py",
    "tests/test_attempt_parameterization_v002.py",
    "tests/test_checkpoint_hardening.py",
    "tests/test_preseal.py",
    "tests/test_program_controller.py",
    "tests/test_runner_gate.py",
    "tests/test_terminal_workflow.py",
    "tests/test_version_forward_transaction.py",
    "tests/test_workflow.py",
)
RESOURCE_IMPLEMENTATION_ALTERNATIVES = (
    "resource_report.py",
    "latency.py",
    "runner.py",
)
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "contact",
        "contacts",
        "contact_state",
        "privileged",
        "privileged_state",
        "qpos",
        "qvel",
        "motion",
        "phase",
        "reward",
        "rewards",
        "success",
        "successes",
        "terminated",
        "truncated",
    }
)
EXPECTED_GIT_HEAD = "86e4bb150de0d1546a3cf57f3ef301294eb04368"
EXPECTED_VERIFIER_MODES = {
    "verifier_contract.json": "confirmation",
    "verifier_contract_no_candidate.json": "no_candidate",
    "verifier_contract_power_infeasible.json": "power_infeasible",
}


class PresealError(RuntimeError):
    """A prospective contract or chronology prerequisite failed closed."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PresealError(f"cannot read required JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise PresealError(f"required JSON is not an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Durably create one immutable JSON file with no overwrite route."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(REPO_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise PresealError(f"sealed path escapes repository: {path}") from exc


def _state() -> dict[str, Any]:
    try:
        state = study_common.read_verified_controller()
    except RuntimeError as exc:
        raise PresealError("controller state/ledger verification failed") from exc
    if state.get("active_attempt") != ACTIVE_ATTEMPT:
        raise PresealError(f"STATE active attempt is not {ACTIVE_ATTEMPT}")
    if state.get("confirmation_terminal") is not False:
        raise PresealError("cannot seal after a confirmation terminal state")
    return state


def validate_no_python_caches() -> dict[str, Any]:
    """Fail if interpreter caches could escape the immutable source/terminal sets."""

    artifacts = sorted(
        (
            path
            for path in STUDY_ROOT.rglob("*")
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        ),
        key=lambda path: str(path),
    )
    if artifacts:
        raise PresealError(
            "Python cache artifacts must be removed before sealing: "
            f"{[str(path) for path in artifacts[:20]]}"
        )
    return {
        "passed": True,
        "study_root": _repo_relative(STUDY_ROOT),
        "python_cache_artifact_count": 0,
    }


def outcome_count_snapshot(state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = _state() if state is None else dict(state)
    snapshot = {field: value.get(field) for field in OUTCOME_COUNT_FIELDS}
    snapshot["confirmation_outcomes_opened_for_analysis"] = value.get(
        "confirmation_outcomes_opened_for_analysis"
    )
    if any(
        not isinstance(snapshot[field], int) or isinstance(snapshot[field], bool)
        for field in OUTCOME_COUNT_FIELDS
    ):
        raise PresealError("STATE outcome counts are not exact integers")
    if snapshot["confirmation_outcomes_opened_for_analysis"] not in (True, False):
        raise PresealError("STATE confirmation analysis flag is not boolean")
    return snapshot


def validate_zero_outcome_state(
    state: Mapping[str, Any] | None = None,
    *,
    require_no_data_files: bool = True,
) -> dict[str, Any]:
    value = _state() if state is None else dict(state)
    snapshot = outcome_count_snapshot(value)
    files = []
    data_root = ATTEMPT_ROOT / "data"
    if data_root.exists():
        files = [
            path.relative_to(ATTEMPT_ROOT).as_posix()
            for path in data_root.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        ]
    checks = {
        "all_role_counts_zero": all(snapshot[field] == 0 for field in OUTCOME_COUNT_FIELDS),
        "confirmation_analysis_unopened": snapshot[
            "confirmation_outcomes_opened_for_analysis"
        ]
        is False,
        "no_role_data_files": not files or not require_no_data_files,
    }
    result = {"passed": all(checks.values()), "checks": checks, "files": files}
    if not result["passed"]:
        raise PresealError(f"zero-outcome invariant failed: {result}")
    return result


def _required_implementation_check() -> dict[str, Any]:
    missing = [
        relative
        for relative in REQUIRED_IMPLEMENTATION_FILES
        if not (ATTEMPT_ROOT / relative).is_file()
    ]
    resource = [
        relative
        for relative in RESOURCE_IMPLEMENTATION_ALTERNATIVES
        if (ATTEMPT_ROOT / relative).is_file()
    ]
    checks = {"all_required_files": not missing, "resource_implementation": bool(resource)}
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "missing": missing,
        "resource_implementations": resource,
    }
    if not result["passed"]:
        raise PresealError(f"implementation is incomplete: {result}")
    return result


def _literal_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower()
    return None


def _key_accesses(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        key: str | None = None
        if isinstance(node, ast.Subscript):
            key = _literal_key(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault"}
            and node.args
        ):
            key = _literal_key(node.args[0])
        elif isinstance(node, ast.Attribute):
            key = node.attr.lower()
        if key in FORBIDDEN_INPUT_KEYS:
            found.append((int(getattr(node, "lineno", -1)), str(key)))
    return found


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise PresealError(f"expected exactly one synchronous function {name}")
    return matches[0]


def validate_source_ast() -> dict[str, Any]:
    """Parse all sources and audit the causal gate/raw-capture boundaries."""

    sources = build_manifest.collect_pre_data_source_paths()
    python_sources = [path for path in sources if path.suffix == ".py"]
    trees: dict[str, ast.Module] = {}
    hashes: dict[str, str] = {}
    for path in python_sources:
        try:
            relative = path.relative_to(ATTEMPT_ROOT).as_posix()
        except ValueError:
            relative = _repo_relative(path)
        try:
            trees[relative] = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise PresealError(f"source does not parse: {relative}") from exc
        hashes[relative] = sha256_file(path)

    generator_accesses = _key_accesses(trees["generator.py"])
    counted_node = _function(trees["counted_features.py"], "build_counted_causal_features")
    counted_arguments = [argument.arg for argument in counted_node.args.args]
    counted_accesses = _key_accesses(counted_node)
    fit_score = _function(trees["fit_select.py"], "score_compiled_gate")
    fit_call = _function(trees["fit_select.py"], "calls_from_scores")
    score_arguments = [argument.arg for argument in fit_score.args.args]
    call_arguments = [argument.arg for argument in fit_call.args.args]
    compiled_accesses = _key_accesses(trees["compile_gate.py"])
    checks = {
        "every_python_source_ast_parses": len(trees) == len(python_sources),
        "generator_never_keys_forbidden_channels": not generator_accesses,
        "counted_feature_signature_exact": counted_arguments
        == ["history", "action_history", "current_prediction", "last_update"],
        "counted_feature_body_never_keys_forbidden_channels": not counted_accesses,
        "score_signature_uses_features_and_frozen_affines_only": score_arguments
        == ["features", "weights", "biases"],
        "routing_signature_uses_scores_and_thresholds_only": call_arguments
        == ["scores", "thresholds"],
        "compiled_gate_never_keys_forbidden_channels": not compiled_accesses,
        "raw_input_allowlist_exact": input_loader.INPUT_ALLOWLIST
        == frozenset({"pixels", "action"}),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "python_source_count": len(python_sources),
        "source_hashes": dict(sorted(hashes.items())),
        "forbidden_accesses": {
            "generator.py": generator_accesses,
            "counted_features.build_counted_causal_features": counted_accesses,
            "compile_gate.py": compiled_accesses,
        },
    }
    if not result["passed"]:
        raise PresealError(f"AST causal-input qualification failed: {result}")
    return result


def validate_seed_freshness_evidence() -> dict[str, Any]:
    ledger_path = ATTEMPT_ROOT / "cohort_seed_ledger.json"
    audit_path = ATTEMPT_ROOT / "audit/identifier_freshness_verification.json"
    ledger = _read_object(ledger_path)
    independent = _read_object(audit_path)
    checks = {
        "ledger_attempt": ledger.get("attempt") == SCIENCE_ATTEMPT,
        "builder_checks_all_pass": isinstance(ledger.get("checks"), Mapping)
        and bool(ledger["checks"])
        and all(value is True for value in ledger["checks"].values()),
        "builder_numeric_overlap_zero": ledger.get("numeric_overlap") == [],
        "builder_string_overlap_zero": ledger.get("string_overlap") == [],
        "builder_prior_outcomes_unopened": ledger.get("prior_outcome_arrays_opened") == 0,
        "builder_fresh_outcomes_unopened": ledger.get("fresh_outcome_episodes_opened") == 0,
        "independent_passed": independent.get("passed") is True,
        "independent_checks_all_pass": isinstance(independent.get("checks"), Mapping)
        and bool(independent["checks"])
        and all(value is True for value in independent["checks"].values()),
        "independent_numeric_overlap_zero": independent.get("numeric_overlap") == [],
        "independent_string_overlap_zero": independent.get("string_overlap") == [],
        "independent_outcomes_unopened": independent.get("prior_outcome_arrays_opened")
        == 0
        and independent.get("fresh_outcome_episodes_opened") == 0,
        "exact_role_counts": ledger.get("role_primary_counts_per_regime")
        == {"fit": 300, "selection": 500, "smoke": 6, "confirmation": 4500},
        "bootstrap_replicates": ledger.get("bootstrap_replicates") == 20_000,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "ledger_sha256": sha256_file(ledger_path),
        "independent_verification_sha256": sha256_file(audit_path),
    }
    if not result["passed"]:
        raise PresealError(f"identifier freshness evidence failed: {result}")
    return result


def validate_consumed_evidence_boundary() -> dict[str, Any]:
    path = ATTEMPT_ROOT / "audit/bootstrap_audit.json"
    audit = _read_object(path)
    expected = {
        "v005_decision": "5dc9f88f105be31dc47e50effd9fa44c1f7d2a7201351a675c3029529d755e38",
        "v005_independent_audit": "520f8f688c17419aa43018d19750d2ef8b52fcb5603da16ec1cc8d97c9ec9fd6",
        "v005_analysis": "e39d9642a406d3fb52f1f775f8a68d1d1e1dffedc5f0bd15b421230bb17484b8",
        "v005_robustness_map": "f3b19508936f235a811a483bcbd4364d691a01ad1e9c87a393c8f8ca533f8bb9",
    }
    consumed = audit.get("consumed_evidence", {})
    checks = {
        "bootstrap_audit_passed": audit.get("passed") is True,
        "v5_terminal": consumed.get("v5_recorded_terminal_label")
        == "v5_confirmation_passed",
        "v005_terminal": consumed.get("v005_recorded_terminal_label")
        == "zero_shot_generalization_partial",
        "v005_named_hashes": all(
            isinstance(consumed.get(name), Mapping)
            and consumed[name].get("sha256") == digest
            for name, digest in expected.items()
        ),
        "prior_not_used_for_new_roles": audit.get("outcome_access", {}).get(
            "v5_or_v005_used_for_new_fit_calibration_whitening_selection_smoke_or_confirmation"
        )
        is False,
        "forbidden_sources_unopened": audit.get("checks", {}).get(
            "forbidden_v3_targets_opened"
        )
        is False
        and audit.get("checks", {}).get("combined_v3_cache_numpy_loaded") is False
        and audit.get("checks", {}).get("released_hdf5_opened") is False,
    }
    result = {"passed": all(checks.values()), "checks": checks, "sha256": sha256_file(path)}
    if not result["passed"]:
        raise PresealError(f"consumed evidence boundary failed: {result}")
    return result


def qualify_exact_flops() -> dict[str, Any]:
    derived = flops.derive_exact_costs()
    pooled = flops.gate_cost(2)
    envelope = flops.gate_cost(8)
    checks = {
        "derivation_passed": derived.get("passed") is True,
        "feature_flops": derived["closed_form"]["feature_flops"] == 3_801,
        "affine_head_flops": derived["closed_form"]["affine_head_flops_each"]
        == 2_092,
        "pooled_total": pooled["total_gate_flops"] == 7_985,
        "envelope_total": envelope["total_gate_flops"] == 20_537,
        "pooled_nonflops": pooled["nonflop_operations"] == 5,
        "envelope_nonflops": envelope["nonflop_operations"] == 11,
        "feature_width": len(counted_features.FROZEN_FEATURE_NAMES) == 1_046,
    }
    result = {"passed": all(checks.values()), "checks": checks, "derivation": derived}
    if not result["passed"]:
        raise PresealError(f"exact gate-compute qualification failed: {result}")
    return result


def _synthetic_role(seed: int) -> fit_select.RoleData:
    rng = np.random.default_rng(seed)
    per_dgp: dict[str, dict[str, np.ndarray]] = {}
    rows = 32
    latent = 6
    for dgp_index, dgp in enumerate(fit_select.DGP_IDS):
        features = rng.normal(scale=0.3, size=(rows, 3, fit_select.FEATURE_DIM))
        signal = rng.normal(size=(rows, 3))
        features[:, :, 0] = signal
        features[:, :, 1] = dgp_index / 3.0
        target = rng.normal(size=(rows, latent))
        error = rng.normal(scale=0.4, size=(rows, latent))
        exits = [target + error]
        for stage in range(3):
            improvement = (0.05 + 0.015 * signal[:, stage, None]) * np.sign(error)
            error = error - improvement
            exits.append(target + error)
        per_dgp[dgp] = {
            "features": features,
            "exits": np.stack(exits, axis=1),
            "target": target,
            "episode_slot": np.repeat(np.arange(8, dtype=np.int64), 4),
            "model_step": np.tile(np.arange(4, dtype=np.int64), 8),
        }
    return fit_select.prepare_role(per_dgp)


def qualify_synthetic_fit_selection() -> dict[str, Any]:
    seeds = {
        dgp: {
            "raw": 4_100_124_000 + index * 3,
            "fixed_whitened": 4_100_124_001 + index * 3,
            "histogram": 4_100_124_002 + index * 3,
        }
        for index, dgp in enumerate(fit_select.DGP_IDS)
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = fit_select.fit_candidate_family(
            _synthetic_role(4_100_123_456), np.eye(6, dtype=np.float64)
        )
        selection = fit_select.evaluate_selection(
            fitted,
            _synthetic_role(4_100_123_457),
            comparator_seeds=seeds,
        )
    fit_select.validate_fitted_artifact(fitted)
    ranked = fit_select.rank_eligible_candidates(selection["candidates"])
    checks = {
        "complete_24_candidate_fit": len(fitted["candidate_ids"]) == 24,
        "both_architectures": set(fitted["head_count"].tolist()) == {2, 8},
        "all_compiled_objects_finite": bool(
            np.isfinite(fitted["compiled_weights"]).all()
            and np.isfinite(fitted["compiled_biases"]).all()
            and np.isfinite(fitted["thresholds"]).all()
        ),
        "selection_evaluated_all_24": selection["candidate_count"] == 24
        and len(selection["candidates"]) == 24,
        "mechanically_eligible_candidate_exists": selection["eligible_count"] > 0,
        "fixed_ranking_reproduced": bool(ranked)
        and selection["selected_candidate_id"] == ranked[0]["candidate_id"],
        "no_selected_head_refit": selection["selected_head_refit_after_selection"]
        is False,
        "separate_synthetic_fit_and_selection_seeds": True,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_count": 24,
        "eligible_count": selection["eligible_count"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "synthetic_only": True,
    }
    if not result["passed"]:
        raise PresealError(f"synthetic fit/selection qualification failed: {result}")
    return result


def qualify_synthetic_sparse_dense() -> dict[str, Any]:
    rng = np.random.default_rng(4_100_124_100)
    rows, heads, latent = 41, 8, 7
    features = rng.normal(scale=0.05, size=(rows, 3, 1_046)).astype(np.float32)
    weights = rng.normal(scale=0.01, size=(3, heads, 1_046)).astype(np.float32)
    biases = rng.normal(scale=0.01, size=(3, heads)).astype(np.float32)
    dense_scores = fit_select.score_compiled_gate(features, weights, biases)
    thresholds = (
        np.quantile(dense_scores, (0.45, 0.55, 0.65), axis=0)
        .diagonal()
        .astype(np.float32, copy=True)
    )
    # An exact stage-one boundary tie exercises the frozen strict-greater rule.
    thresholds[0] = dense_scores[0, 0]
    calls, reached = fit_select.calls_from_scores(dense_scores, thresholds)
    sparse_scores = np.where(reached, dense_scores, np.nan)
    reconstructed = analysis.sequential_calls_from_scores(sparse_scores, thresholds)

    manual_scores = np.full_like(dense_scores, np.nan)
    manual_calls = np.ones(rows, dtype=np.int64)
    active = np.ones(rows, dtype=bool)
    for stage in range(3):
        indices = np.flatnonzero(active)
        # Deliberately use an explicit reduction as the second score
        # implementation; this avoids sharing the producer's matmul kernel.
        heads_here = np.sum(
            features[indices, stage, None, :] * weights[stage, None, :, :],
            axis=2,
            dtype=np.float32,
        ) + biases[stage]
        manual_scores[indices, stage] = heads_here.min(axis=1)
        # Routing is checked against the frozen producer score. Numerical
        # score equivalence is assessed separately at machine precision.
        active[indices] = dense_scores[indices, stage] > thresholds[stage]
        manual_calls += active.astype(np.int64)
    exits = rng.normal(size=(rows, 4, latent))
    dense_selected = exits[np.arange(rows), calls - 1]
    sparse_selected = np.stack(
        [exits[index, manual_calls[index] - 1] for index in range(rows)]
    )

    compiled, compile_audit = compile_gate.compile_arrays(
        architecture="domain_envelope_eight",
        candidate_id="synthetic_preseal_gate",
        weights=weights,
        biases=biases,
        thresholds=thresholds,
        head_names=[f"synthetic_head_{index}" for index in range(heads)],
    )
    checks = {
        "fit_and_analysis_calls_exact": np.array_equal(calls, reconstructed),
        "manual_sparse_calls_exact": np.array_equal(calls, manual_calls),
        "manual_sparse_score_reach_mask_exact": np.array_equal(
            np.isnan(sparse_scores), np.isnan(manual_scores)
        ),
        "manual_sparse_scores_machine_precision_equivalent": bool(
            np.allclose(
                sparse_scores[reached],
                manual_scores[reached],
                rtol=2e-5,
                atol=2e-6,
            )
        ),
        "sparse_selected_equals_dense_selected": np.array_equal(
            sparse_selected, dense_selected
        ),
        "exact_boundary_tie_stops": int(calls[0]) == 1,
        "compiled_runtime_float32": all(
            compiled[name].dtype == np.dtype(np.float32)
            for name in ("weights", "biases", "thresholds")
        ),
        "compiled_head_count_eight": compile_audit["head_count"] == 8,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "call_histogram": np.bincount(calls, minlength=5)[1:].tolist(),
        "maximum_selected_difference": float(
            np.max(np.abs(sparse_selected - dense_selected))
        ),
        "synthetic_only": True,
    }
    if not result["passed"]:
        raise PresealError(f"synthetic sparse/dense qualification failed: {result}")
    return result


def _independent_bootstrap(
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    seed: int,
    replicates: int,
    chunk_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    output = {
        dgp: {
            name: np.empty(replicates, dtype=np.float64)
            for name in analysis.ALL_CONTRASTS
        }
        for dgp in analysis.DGP_ORDER
    }
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        for dgp in analysis.DGP_ORDER:
            count = len(metrics[dgp][analysis.ALL_CONTRASTS[0]])
            sampled = rng.integers(
                0, count, size=(stop - start, count), dtype=np.int32
            )
            for name in analysis.ALL_CONTRASTS:
                values = np.asarray(metrics[dgp][name], dtype=np.float64)
                output[dgp][name][start:stop] = values[sampled].mean(axis=1)
    return output


def qualify_synthetic_comparators_bootstrap() -> dict[str, Any]:
    rng = np.random.default_rng(4_100_124_200)
    calls = np.asarray([1, 2, 3, 4] * 8, dtype=np.int64)
    pricing = analysis.ComputePricing(
        gate_feature_flops=3_801,
        gate_head_flops=16_736,
        gate_nonflop_operations=11,
    )
    compute = analysis.adaptive_compute_account(calls, pricing)
    fixed_losses = np.square(rng.normal(size=(len(calls), 4)))
    allocation = analysis.strongest_transition_independent_allocation(
        fixed_losses, compute["adaptive_total_counted_flops"], pricing
    )
    seeded = analysis.seeded_weakly_more_compute_control(
        fixed_losses,
        allocation,
        compute["adaptive_total_counted_flops"],
        pricing,
        4_100_124_201,
    )
    slots = np.repeat(np.arange(8, dtype=np.int64), 4)
    randomized, histograms = analysis.within_episode_histogram_calls(
        calls, slots, 4_100_124_202
    )
    metrics = {
        dgp: {
            name: rng.normal(loc=0.01, scale=0.1, size=9)
            for name in analysis.ALL_CONTRASTS
        }
        for dgp in analysis.DGP_ORDER
    }
    produced = analysis.bootstrap_all(
        metrics,
        4_100_124_203,
        replicates=257,
        chunk_size=31,
    )
    independent = _independent_bootstrap(metrics, 4_100_124_203, 257, 31)
    bootstrap_exact = all(
        np.array_equal(produced[dgp][name], independent[dgp][name])
        for dgp in analysis.DGP_ORDER
        for name in analysis.ALL_CONTRASTS
    )
    checks = {
        "analytic_exact_total_compute": allocation["exact_total_compute_match"]
        is True,
        "analytic_transition_independent": allocation["transition_independent"]
        is True,
        "seeded_weakly_more_compute": seeded["weakly_more_compute"] is True
        and seeded["total_counted_flops"]
        >= compute["adaptive_total_counted_flops"],
        "within_episode_histograms_exact": all(item["preserved"] for item in histograms),
        "within_episode_global_calls_preserved": np.array_equal(
            np.sort(randomized), np.sort(calls)
        ),
        "bootstrap_fixed_contract": analysis.BOOTSTRAP_REPLICATES == 20_000
        and analysis.FAMILY_SIZE == 8
        and analysis.PER_CLAIM_ALPHA == 0.00625,
        "bootstrap_separate_implementation_elementwise_exact": bootstrap_exact,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "synthetic_replicates": 257,
        "synthetic_only": True,
    }
    if not result["passed"]:
        raise PresealError(f"synthetic comparator/bootstrap qualification failed: {result}")
    return result


def qualify_feature_lineage() -> dict[str, Any]:
    import torch

    old_path = (
        REPO_ROOT
        / "runs/lewm_v5_readiness_program/v5_package_versions/v004/counted_features.py"
    )
    specification = importlib.util.spec_from_file_location("_v5_v004_features", old_path)
    if specification is None or specification.loader is None:
        raise PresealError("cannot load frozen V5 counted-feature source")
    old = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(old)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4_100_124_300)
    tensors = (
        torch.randn((11, 3, 192), generator=generator),
        torch.randn((11, 3, 25), generator=generator),
        torch.randn((11, 192), generator=generator),
        torch.randn((11, 192), generator=generator),
    )
    current = counted_features.build_counted_causal_features(*tensors)
    inherited = old.build_counted_causal_features(*tensors)
    names = counted_features.semantic_feature_names(
        latent_dim=192, action_dim=25, history_len=3
    )
    checks = {
        "synthetic_feature_values_bitwise_v004_exact": bool(torch.equal(current, inherited)),
        "feature_shape": tuple(current.shape) == (11, 1_046),
        "feature_names_complete_unique": len(names) == 1_046 and len(set(names)) == 1_046,
        "gradient_boundary_detached": not current.requires_grad,
        "v004_source_present": old_path.is_file(),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "v004_counted_features_sha256": sha256_file(old_path),
        "v001_counted_features_sha256": sha256_file(ATTEMPT_ROOT / "counted_features.py"),
        "synthetic_only": True,
    }
    if not result["passed"]:
        raise PresealError(f"counted-feature lineage qualification failed: {result}")
    return result


def _write_or_verify_manifest() -> dict[str, Any]:
    if SOURCE_MANIFEST_PATH.exists():
        build_manifest.verify_manifest(SOURCE_MANIFEST_PATH)
        return _read_object(SOURCE_MANIFEST_PATH)
    manifest = build_manifest.build_pre_data_manifest()
    build_manifest.write_manifest_exclusive(SOURCE_MANIFEST_PATH, manifest)
    build_manifest.verify_manifest(SOURCE_MANIFEST_PATH)
    return manifest


def validate_verifier_contracts() -> dict[str, Any]:
    """Validate and hash all immutable terminal-mode contracts pre-outcome."""

    attempt_relative = ATTEMPT_ROOT.relative_to(REPO_ROOT).as_posix()
    study_relative = STUDY_ROOT.relative_to(REPO_ROOT).as_posix()
    terminal_manifest = f"{attempt_relative}/audit/terminal_input_manifest.json"
    exact_exclusions = {
        terminal_manifest,
        f"{attempt_relative}/audit/independent_verification.json",
    }
    hashes: dict[str, str] = {}
    modes: dict[str, str] = {}
    for path in VERIFIER_CONTRACT_PATHS:
        if not path.is_file() or path.is_symlink():
            raise PresealError(f"verifier contract is absent or linked: {path}")
        value = _read_object(path)
        expected_mode = EXPECTED_VERIFIER_MODES[path.name]
        checks = {
            "schema": value.get("schema_version") == 1,
            "attempt": value.get("attempt") == ACTIVE_ATTEMPT,
            "attempt_root": value.get("attempt_root") == attempt_relative,
            "study_root": value.get("study_root") == study_relative,
            "mode": value.get("mode") == expected_mode,
            "git_head": value.get("expected_git_head") == EXPECTED_GIT_HEAD,
            "fit_dgps": (
                isinstance(value.get("fit_inputs"), Mapping)
                and tuple(value["fit_inputs"]) == study_common.REGIME_ORDER
            ),
            "selection_dgps": (
                isinstance(value.get("selection_inputs"), Mapping)
                and tuple(value["selection_inputs"]) == study_common.REGIME_ORDER
            ),
            "development_roles": (
                isinstance(value.get("development_manifests"), Mapping)
                and set(value["development_manifests"]) == {"fit", "selection"}
            ),
            "terminal_manifest": (
                isinstance(value.get("paths"), Mapping)
                and value["paths"].get("terminal_manifest") == terminal_manifest
            ),
            "terminal_exclusions": (
                isinstance(value.get("terminal_manifest_exclusions"), list)
                and set(value["terminal_manifest_exclusions"]) == exact_exclusions
                and len(value["terminal_manifest_exclusions"]) == 2
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise PresealError(
                f"mode-specific verifier contract failed {path.name}: {failed}"
            )
        hashes[path.name] = sha256_file(path)
        modes[path.name] = expected_mode
    if len(set(hashes.values())) != len(hashes):
        raise PresealError("mode-specific verifier contract bytes are not distinct")
    return {
        "passed": True,
        "contract_count": len(hashes),
        "modes": modes,
        "sha256": hashes,
        "sealed_before_fresh_outcomes": True,
    }


def implementation_complete() -> dict[str, Any]:
    state = _state()
    if state.get("current_state") != "IMPLEMENTATION_COMPLETE":
        raise PresealError("implementation evidence may only be made in IMPLEMENTATION_COMPLETE")
    runtime = verify_here("sealing")
    required = _required_implementation_check()
    zero = validate_zero_outcome_state(state)
    contracts = study_common.validate_normative_contracts()
    ast_result = validate_source_ast()
    compute = qualify_exact_flops()
    consumed = validate_consumed_evidence_boundary()
    verifier_contracts = validate_verifier_contracts()
    cache_hygiene = validate_no_python_caches()
    manifest = _write_or_verify_manifest()
    checks = {
        "required_implementation": required["passed"],
        "zero_outcomes": zero["passed"],
        "normative_contracts": bool(contracts)
        and set(contracts)
        == {"DGP_MATRIX", "candidate_grid", "power_rule", "outcome_mapping"},
        "ast_causal_boundary": ast_result["passed"],
        "exact_compute": compute["passed"],
        "consumed_boundary": consumed["passed"],
        "mode_specific_verifier_contracts": verifier_contracts["passed"],
        "python_cache_hygiene": cache_hygiene["passed"],
        "manifest_verified": build_manifest.verify_manifest(SOURCE_MANIFEST_PATH)["passed"],
        "runtime_verified": runtime["passed"],
    }
    result = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "IMPLEMENTATION_COMPLETE",
        "created_unix_ns": time.time_ns(),
        "passed": all(checks.values()),
        "checks": checks,
        "source_manifest_path": _repo_relative(SOURCE_MANIFEST_PATH),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "source_ast": ast_result,
        "exact_compute": compute,
        "verifier_contracts": verifier_contracts,
        "python_cache_hygiene": cache_hygiene,
        "outcome_counts": outcome_count_snapshot(state),
        "prospective_outcome_arrays_opened": 0,
        "prior_outcome_arrays_opened": 0,
    }
    if not result["passed"]:
        raise PresealError(f"implementation qualification failed: {result}")
    _write_json_exclusive(IMPLEMENTATION_REPORT_PATH, result)
    return result


def _verified_checkpoint(path: Path, state: str) -> dict[str, Any]:
    value = _read_object(path)
    if (
        value.get("passed") is not True
        or value.get("attempt") != ACTIVE_ATTEMPT
        or value.get("checkpoint_state") != state
    ):
        raise PresealError(f"invalid checkpoint evidence: {path}")
    return value


def preseal_qualification() -> dict[str, Any]:
    state = _state()
    if state.get("current_state") != "PRESEAL_QUALIFICATION":
        raise PresealError("qualification may only run in PRESEAL_QUALIFICATION")
    runtime = verify_here("qualification")
    validate_no_python_caches()
    implementation = _verified_checkpoint(
        IMPLEMENTATION_REPORT_PATH, "IMPLEMENTATION_COMPLETE"
    )
    manifest_check = build_manifest.verify_manifest(SOURCE_MANIFEST_PATH)
    if implementation.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise PresealError("implementation/source-manifest cross-link drift")
    zero = validate_zero_outcome_state(state)
    qualifications = {
        "seed_freshness": validate_seed_freshness_evidence(),
        "feature_lineage": qualify_feature_lineage(),
        "fit_selection": qualify_synthetic_fit_selection(),
        "sparse_dense": qualify_synthetic_sparse_dense(),
        "comparators_bootstrap": qualify_synthetic_comparators_bootstrap(),
        "exact_flops": qualify_exact_flops(),
        "source_ast": validate_source_ast(),
    }
    checks = {
        "runtime": runtime["passed"],
        "implementation": implementation["passed"],
        "manifest": manifest_check["passed"],
        "zero_outcomes": zero["passed"],
        **{name: item["passed"] for name, item in qualifications.items()},
    }
    result = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "PRESEAL_QUALIFICATION",
        "created_unix_ns": time.time_ns(),
        "passed": all(checks.values()),
        "checks": checks,
        "qualifications": qualifications,
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "implementation_report_sha256": sha256_file(IMPLEMENTATION_REPORT_PATH),
        "synthetic_inputs_only": True,
        "fit_outcome_episodes_opened": 0,
        "selection_outcome_episodes_opened": 0,
        "smoke_outcome_episodes_opened": 0,
        "confirmation_outcome_episodes_opened": 0,
        "mps_required_for_real_execution": True,
        "cpu_fallback_permitted": False,
    }
    if not result["passed"]:
        raise PresealError(f"preseal qualification failed: {result}")
    _write_json_exclusive(QUALIFICATION_REPORT_PATH, result)
    return result


def _base_sealed_files() -> dict[str, str]:
    manifest = _read_object(SOURCE_MANIFEST_PATH)
    build_manifest.verify_manifest(manifest)
    files = build_manifest.repository_hash_map(manifest)
    for path in (
        SOURCE_MANIFEST_PATH,
        IMPLEMENTATION_REPORT_PATH,
        QUALIFICATION_REPORT_PATH,
    ):
        files[_repo_relative(path)] = sha256_file(path)
    return dict(sorted(files.items()))


def _validate_sealed_files(files: Mapping[str, str]) -> None:
    if not files:
        raise PresealError("sealed_files cannot be empty")
    bad: list[str] = []
    for raw_relative, expected in files.items():
        relative = Path(str(raw_relative))
        if relative.is_absolute() or ".." in relative.parts:
            bad.append(str(raw_relative))
            continue
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            bad.append(str(raw_relative))
    if bad:
        raise PresealError(f"sealed repository file failure: {bad[:20]}")


def seal_pre_data() -> dict[str, Any]:
    state = _state()
    if state.get("current_state") != "PRE_OUTCOME_SEAL":
        raise PresealError("pre-data seal may only be made in PRE_OUTCOME_SEAL")
    runtime = verify_here("sealing")
    cache_hygiene = validate_no_python_caches()
    dual = verify_dual_interpreter_contract()
    zero = validate_zero_outcome_state(state)
    implementation = _verified_checkpoint(
        IMPLEMENTATION_REPORT_PATH, "IMPLEMENTATION_COMPLETE"
    )
    qualification = _verified_checkpoint(
        QUALIFICATION_REPORT_PATH, "PRESEAL_QUALIFICATION"
    )
    manifest = build_manifest.verify_manifest(SOURCE_MANIFEST_PATH)
    verifier_contracts = validate_verifier_contracts()
    current_manifest_sha = sha256_file(SOURCE_MANIFEST_PATH)
    if {
        implementation.get("source_manifest_sha256"),
        qualification.get("source_manifest_sha256"),
    } != {current_manifest_sha}:
        raise PresealError("qualification reports do not bind the current source manifest")
    files = _base_sealed_files()
    _validate_sealed_files(files)
    result = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "PRE_OUTCOME_SEAL",
        "status": "frozen_before_any_fresh_fit_selection_smoke_or_confirmation_outcome",
        "created_unix_ns": time.time_ns(),
        "passed": all(
            (
                runtime["passed"],
                dual["passed"],
                zero["passed"],
                implementation["passed"],
                qualification["passed"],
                manifest["passed"],
                verifier_contracts["passed"],
            )
        ),
        "outcome_counts_at_seal": outcome_count_snapshot(state),
        "sealed_files": files,
        "source_manifest_sha256": current_manifest_sha,
        "implementation_report_sha256": sha256_file(IMPLEMENTATION_REPORT_PATH),
        "qualification_report_sha256": sha256_file(QUALIFICATION_REPORT_PATH),
        "dual_interpreter_contract": dual,
        "mode_specific_verifier_contracts": verifier_contracts,
        "python_cache_hygiene": cache_hygiene,
        "generation_preflight_required_before_seed_or_output": True,
        "manual_interpreter_selection_forbidden": True,
        "mps_required_for_execution": True,
        "cpu_fallback_permitted": False,
        "role_isolation_frozen": True,
        "post_seal_scientific_source_patch_forbidden": True,
    }
    if not result["passed"]:
        raise PresealError(f"pre-data seal prerequisites failed: {result}")
    _write_json_exclusive(PRE_DATA_SEAL_PATH, result)
    return result


def _files_under(*roots: Path) -> list[Path]:
    paths: list[Path] = []
    for supplied_root in roots:
        # Do not call is_dir/is_file until the lexical chain has been lstat'd:
        # both predicates follow a supplied root symlink and would let rglob
        # seal an unrelated directory under the symlink's trusted spelling.
        root = Path(os.path.abspath(os.fspath(supplied_root)))
        current = Path(root.anchor)
        for part in root.parts[1:]:
            current /= part
            if not os.path.lexists(current):
                continue
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise PresealError(
                    f"stage manifest lexical ancestor symlink forbidden: {current}"
                )
            if current != root and not stat.S_ISDIR(metadata.st_mode):
                raise PresealError(
                    f"stage manifest lexical ancestor is not a directory: {current}"
                )
        if not os.path.lexists(root):
            continue
        root_metadata = root.lstat()
        if stat.S_ISREG(root_metadata.st_mode):
            paths.append(root)
        elif stat.S_ISDIR(root_metadata.st_mode):
            candidates = list(root.rglob("*"))
            linked = [path for path in candidates if path.is_symlink()]
            if linked:
                raise PresealError(f"stage manifest symlinks forbidden: {linked[:20]}")
            paths.extend(
                path
                for path in candidates
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix not in {".pyc", ".lock"}
                and "__pycache__" not in path.parts
            )
        else:
            raise PresealError(
                f"stage manifest root is not a regular file or directory: {root}"
            )
    return sorted(set(paths), key=lambda path: str(path))


def _find_status(paths: Iterable[Path], status: str) -> Path:
    matches: list[Path] = []
    for path in paths:
        if path.suffix != ".json":
            continue
        try:
            value = _read_object(path)
        except PresealError:
            continue
        if value.get("status") == status:
            matches.append(path)
    if len(matches) != 1:
        raise PresealError(f"expected one {status!r} artifact, found {matches}")
    return matches[0]


def _write_stage_manifest(path: Path, label: str, paths: Sequence[Path]) -> dict[str, Any]:
    if path.exists():
        build_manifest.verify_manifest(path)
        return _read_object(path)
    manifest = build_manifest.build_addendum_manifest(label, paths)
    build_manifest.write_manifest_exclusive(path, manifest)
    build_manifest.verify_manifest(path)
    return manifest


def _stage_files(base: Mapping[str, str], manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    files = dict(base)
    files.update(build_manifest.repository_hash_map(manifest))
    files[_repo_relative(manifest_path)] = sha256_file(manifest_path)
    _validate_sealed_files(files)
    return dict(sorted(files.items()))


def _recorded_repo_input_matches(item: object, expected_path: Path) -> bool:
    """Verify one canonical repository-relative path/hash input independent of CWD."""

    if not isinstance(item, Mapping):
        return False
    raw = item.get("path")
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return False
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
        return False
    try:
        recorded_path = (REPO_ROOT / relative).resolve(strict=True)
        repository = REPO_ROOT.resolve(strict=True)
        expected = expected_path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    return (
        recorded_path.is_relative_to(repository)
        and recorded_path == expected
        and item.get("sha256") == sha256_file(expected_path)
    )


def _summary_claim_schema_exact(summary: object) -> bool:
    """Validate claim keys without relying on JSON dictionary insertion order."""

    if not isinstance(summary, Mapping):
        return False
    claims = summary.get("claims")
    if not isinstance(claims, Mapping) or set(claims) != set(
        study_common.REGIME_ORDER
    ):
        return False
    endpoints = {"raw_vs_analytic", "fixed_whitened_vs_analytic"}
    return all(
        isinstance(claims.get(regime), Mapping)
        and set(claims[regime]) == endpoints
        for regime in study_common.REGIME_ORDER
    )


def _validated_confirmation_count_map(value: object) -> dict[str, int]:
    """Return frozen-order counts after strict type/set/grid validation."""

    if not isinstance(value, Mapping) or set(value) != set(
        study_common.REGIME_ORDER
    ):
        raise PresealError("confirmation count map does not contain the exact DGP set")
    counts: dict[str, int] = {}
    for regime in study_common.REGIME_ORDER:
        count = value.get(regime)
        if not isinstance(count, int) or isinstance(count, bool):
            raise PresealError(f"confirmation count is not an integer for {regime}")
        if count not in range(500, 4501, 500):
            raise PresealError(f"confirmation count is off the frozen grid for {regime}")
        counts[regime] = count
    if len(set(counts.values())) != 1:
        raise PresealError("confirmation count is not common across all DGPs")
    return counts


def _canonical_object_sha256(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _power_provenance_checks(
    power: Mapping[str, Any],
    *,
    fit_summary_path: Path,
    selection_summary_path: Path,
    power_rule_path: Path,
    fit_lock_path: Path,
    selection_ledger_path: Path,
    gate_freeze_path: Path,
    selected_candidate_id: object,
    selected_candidate_object_sha256: object,
) -> dict[str, bool]:
    """Rehash every binding-power input and selected-gate identity link."""

    inputs = power.get("inputs")
    if not isinstance(inputs, Mapping):
        inputs = {}
    expected_input_names = {
        "fit_summary",
        "selection_summary",
        "power_rule",
        "fit_lock",
        "selection_ledger",
        "gate_freeze",
    }
    rule_record = inputs.get("power_rule")
    rule = _read_object(power_rule_path)
    moments = rule.get("consumed_v5_endpoint_moments")
    fit_lock_hash = sha256_file(fit_lock_path)
    selection_hash = sha256_file(selection_ledger_path)
    freeze_hash = sha256_file(gate_freeze_path)
    expected_identity = {
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_object_sha256": selected_candidate_object_sha256,
        "fit_lock_sha256": fit_lock_hash,
        "selection_ledger_sha256": selection_hash,
        "gate_freeze_sha256": freeze_hash,
    }
    selected_identity = power.get("selected_gate_identity")
    return {
        "input_names_exact": set(inputs) == expected_input_names,
        "fit_summary_record_exact": _recorded_repo_input_matches(
            inputs.get("fit_summary"), fit_summary_path
        ),
        "selection_summary_record_exact": _recorded_repo_input_matches(
            inputs.get("selection_summary"), selection_summary_path
        ),
        "power_rule_record_exact": _recorded_repo_input_matches(
            rule_record, power_rule_path
        ),
        "power_rule_expected_sha256_exact": (
            isinstance(rule_record, Mapping)
            and rule_record.get("expected_sha256")
            == power_analysis.POWER_RULE_SHA256
            and sha256_file(power_rule_path) == power_analysis.POWER_RULE_SHA256
        ),
        "power_rule_moments_object_sha256_exact": (
            isinstance(rule_record, Mapping)
            and isinstance(moments, Mapping)
            and rule_record.get("endpoint_moments_object_sha256")
            == power_analysis.POWER_RULE_MOMENTS_SHA256
            and _canonical_object_sha256(moments)
            == power_analysis.POWER_RULE_MOMENTS_SHA256
        ),
        "fit_lock_record_exact": _recorded_repo_input_matches(
            inputs.get("fit_lock"), fit_lock_path
        ),
        "selection_ledger_record_exact": _recorded_repo_input_matches(
            inputs.get("selection_ledger"), selection_ledger_path
        ),
        "gate_freeze_record_exact": _recorded_repo_input_matches(
            inputs.get("gate_freeze"), gate_freeze_path
        ),
        "selected_gate_identity_exact": (
            isinstance(selected_identity, Mapping)
            and set(selected_identity) == set(expected_identity)
            and all(
                selected_identity.get(key) == value
                for key, value in expected_identity.items()
            )
        ),
    }


def validate_power_input_coherence(paths: Sequence[Path]) -> dict[str, Any]:
    """Rebind power moments to one selected immutable candidate and its locks."""

    fit_summary_path = ATTEMPT_ROOT / "metrics/fit_power_summary.json"
    selection_summary_path = ATTEMPT_ROOT / "metrics/selection_power_summary.json"
    power_path = ATTEMPT_ROOT / "power_analysis.json"
    fit_summary = _read_object(fit_summary_path)
    selection_summary = _read_object(selection_summary_path)
    power = _read_object(power_path)
    fit_lock_path = _find_status(
        paths, "all_24_candidates_fit_compiled_and_locked_before_selection_open"
    )
    selection_ledger_path = _find_status(
        paths, "selection_complete_no_selected_head_refit"
    )
    gate_freeze_path = _find_status(
        paths, "selected_gate_frozen_before_smoke_or_confirmation"
    )
    fit_lock = _read_object(fit_lock_path)
    selection_ledger = _read_object(selection_ledger_path)
    gate_freeze = _read_object(gate_freeze_path)
    candidate_id = fit_summary.get("selected_candidate_id")
    object_hash = fit_summary.get("selected_candidate_object_sha256")
    selected_row = next(
        (
            row
            for row in selection_ledger.get("candidates", [])
            if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
        ),
        None,
    )
    compiled_gate_path = ATTEMPT_ROOT / "freeze/compiled_gate.npz"
    gate_fit_path = ATTEMPT_ROOT / "freeze/gate_fit.npz"
    power_rule_path = ATTEMPT_ROOT / "power_rule.json"
    power_provenance = _power_provenance_checks(
        power,
        fit_summary_path=fit_summary_path,
        selection_summary_path=selection_summary_path,
        power_rule_path=power_rule_path,
        fit_lock_path=fit_lock_path,
        selection_ledger_path=selection_ledger_path,
        gate_freeze_path=gate_freeze_path,
        selected_candidate_id=candidate_id,
        selected_candidate_object_sha256=object_hash,
    )
    expected_summary_identity = {
        "attempt": ACTIVE_ATTEMPT,
        "selected_candidate_id": gate_freeze.get("selected_candidate_id"),
        "selected_candidate_object_sha256": gate_freeze.get(
            "selected_candidate_object_sha256"
        ),
        "fit_lock_sha256": sha256_file(fit_lock_path),
        "selection_ledger_sha256": sha256_file(selection_ledger_path),
        "gate_freeze_sha256": sha256_file(gate_freeze_path),
    }
    fitted_path_candidates = [
        path
        for path in paths
        if path.suffix == ".npz"
        and path.is_relative_to(ATTEMPT_ROOT / "fit")
        and sha256_file(path) == str(fit_lock.get("fitted_candidates_sha256"))
    ]
    try:
        if len(fitted_path_candidates) != 1:
            raise PresealError(
                "binding-power recomputation requires one exact fitted artifact"
            )
        fitted = fit_select.load_fitted_candidates(fitted_path_candidates[0])
        expected_fit_summary, expected_selection_summary = (
            fit_select.selected_power_summaries(fitted, selection_ledger)
        )
        expected_fit_summary.update(expected_summary_identity)
        expected_selection_summary.update(expected_summary_identity)
        summary_recomputation_checks = {
            "fit": _canonical_object_sha256(fit_summary)
            == _canonical_object_sha256(expected_fit_summary),
            "selection": _canonical_object_sha256(selection_summary)
            == _canonical_object_sha256(expected_selection_summary),
        }

        frozen_moments = power_analysis.load_frozen_power_rule_moments(
            power_rule_path
        )
        expected_power = power_analysis.compute_binding_power(
            expected_fit_summary,
            expected_selection_summary,
            frozen_moments,
        )
        identity_artifacts = power_analysis.validate_identity_artifacts(
            expected_power["selected_gate_identity"],
            fit_lock_path=fit_lock_path,
            selection_ledger_path=selection_ledger_path,
            gate_freeze_path=gate_freeze_path,
        )
        expected_power["inputs"] = {
            "fit_summary": {
                "path": _repo_relative(fit_summary_path),
                "sha256": sha256_file(fit_summary_path),
            },
            "selection_summary": {
                "path": _repo_relative(selection_summary_path),
                "sha256": sha256_file(selection_summary_path),
            },
            "power_rule": {
                "path": _repo_relative(power_rule_path),
                "sha256": sha256_file(power_rule_path),
                "expected_sha256": power_analysis.POWER_RULE_SHA256,
                "endpoint_moments_object_sha256": (
                    power_analysis.POWER_RULE_MOMENTS_SHA256
                ),
            },
            **identity_artifacts,
        }
        created_unix_ns = power.get("created_unix_ns")
        freeze_created_unix_ns = gate_freeze.get("created_unix_ns")
        chronology_exact = (
            type(created_unix_ns) is int
            and type(freeze_created_unix_ns) is int
            and int(created_unix_ns) > int(freeze_created_unix_ns)
        )
        if chronology_exact:
            expected_power["created_unix_ns"] = int(created_unix_ns)
        power_recomputation_exact = chronology_exact and (
            _canonical_object_sha256(power)
            == _canonical_object_sha256(expected_power)
        )
    except PresealError:
        raise
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise PresealError(
            "independent pre-confirmation binding-power recomputation failed"
        ) from exc
    checks = {
        "summary_roles_exact": fit_summary.get("role") == "fit"
        and selection_summary.get("role") == "selection",
        "summaries_co_primary_only": fit_summary.get("co_primary_only") is True
        and selection_summary.get("co_primary_only") is True
        and fit_summary.get("auxiliary_robust_whitening_excluded") is True
        and selection_summary.get("auxiliary_robust_whitening_excluded") is True,
        "summaries_same_candidate": bool(candidate_id)
        and candidate_id == selection_summary.get("selected_candidate_id")
        and object_hash == selection_summary.get("selected_candidate_object_sha256"),
        "summary_claim_schema": all(
            _summary_claim_schema_exact(summary)
            for summary in (fit_summary, selection_summary)
        ),
        "summary_identity_bound_after_gate_freeze": all(
            all(summary.get(key) == value for key, value in expected_summary_identity.items())
            for summary in (fit_summary, selection_summary)
        ),
        "selection_ledger_same_candidate": selection_ledger.get(
            "selected_candidate_id"
        )
        == candidate_id
        and isinstance(selected_row, Mapping)
        and selected_row.get("candidate_object_sha256") == object_hash,
        "selection_ledger_binds_fit_lock": selection_ledger.get("fit_lock_sha256")
        == sha256_file(fit_lock_path),
        "fit_lock_binds_one_fitted_artifact": len(fitted_path_candidates) == 1,
        "selection_ledger_binds_fitted_artifact": len(fitted_path_candidates) == 1
        and selection_ledger.get("fitted_candidates_sha256")
        == sha256_file(fitted_path_candidates[0]),
        "fit_summary_exact_selected_gate_recomputation": (
            summary_recomputation_checks["fit"]
        ),
        "selection_summary_exact_selected_gate_recomputation": (
            summary_recomputation_checks["selection"]
        ),
        "binding_power_chronology_exact": chronology_exact,
        "binding_power_object_exact_recomputation": power_recomputation_exact,
        "gate_freeze_same_candidate_object": gate_freeze.get("selected_candidate_id")
        == candidate_id
        and gate_freeze.get("selected_candidate_object_sha256") == object_hash,
        "gate_freeze_binds_fit_lock": gate_freeze.get("fit_lock_sha256")
        == sha256_file(fit_lock_path),
        "gate_freeze_binds_selection_ledger": gate_freeze.get(
            "selection_ledger_sha256"
        )
        == sha256_file(selection_ledger_path),
        "gate_freeze_binds_gate_objects": compiled_gate_path.is_file()
        and gate_fit_path.is_file()
        and gate_freeze.get("compiled_gate_sha256") == sha256_file(compiled_gate_path)
        and gate_freeze.get("gate_fit_sha256") == sha256_file(gate_fit_path),
        "binding_power_is_feasible_and_fixed": power.get("binding") is True
        and power.get("passed") is True
        and power.get("feasible") is True
        and power.get("decision") == "confirmation_size_fixed",
        **{
            f"power_provenance_{name}": passed
            for name, passed in power_provenance.items()
        },
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "selected_candidate_id": candidate_id,
        "selected_candidate_object_sha256": object_hash,
        "fit_summary_sha256": sha256_file(fit_summary_path),
        "selection_summary_sha256": sha256_file(selection_summary_path),
        "fit_lock_sha256": sha256_file(fit_lock_path),
        "selection_ledger_sha256": sha256_file(selection_ledger_path),
        "gate_freeze_sha256": sha256_file(gate_freeze_path),
        "power_analysis_sha256": sha256_file(power_path),
        "recomputed_power_object_sha256": _canonical_object_sha256(expected_power),
        "power_provenance": power_provenance,
    }
    if not result["passed"]:
        raise PresealError(f"power/selection/gate coherence failed: {result}")
    return result


def validate_compiler_crosslink(paths: Sequence[Path]) -> dict[str, Any]:
    """Require one float32 selected object to pass through compiler unchanged."""

    gate_freeze_path = _find_status(
        paths, "selected_gate_frozen_before_smoke_or_confirmation"
    )
    gate_freeze = _read_object(gate_freeze_path)
    source_path = ATTEMPT_ROOT / "freeze/gate_fit.npz"
    compiled_path = ATTEMPT_ROOT / "freeze/compiled_gate.npz"
    manifest_path = ATTEMPT_ROOT / "freeze/compiled_gate_manifest.json"
    manifest = _read_object(manifest_path)
    try:
        manifest_source = Path(str(manifest["source_gate_fit_path"])).resolve(strict=True)
        manifest_compiled = Path(str(manifest["compiled_gate_path"])).resolve(
            strict=True
        )
    except (KeyError, OSError, RuntimeError) as exc:
        raise PresealError("compiler manifest paths are invalid") from exc
    with np.load(source_path, allow_pickle=False) as stored:
        source = {name: stored[name].copy() for name in stored.files}
    with np.load(compiled_path, allow_pickle=False) as stored:
        compiled = {name: stored[name].copy() for name in stored.files}
    identity_keys = ("architecture", "candidate_id", "head_names")
    float_keys = ("weights", "biases", "thresholds")
    metadata = manifest.get("metadata", {})
    cast_delta = metadata.get("float32_cast_max_abs", {})
    array_hashes = metadata.get("array_sha256", {})
    source_candidate = str(np.asarray(source.get("candidate_id")).item())
    source_architecture = str(np.asarray(source.get("architecture")).item())
    source_head_count = int(np.asarray(source.get("head_count")).item())
    checks = {
        "compiler_manifest_passed": manifest.get("passed") is True,
        "compiler_manifest_source_path_exact": manifest_source == source_path.resolve(),
        "compiler_manifest_output_path_exact": manifest_compiled
        == compiled_path.resolve(),
        "compiler_manifest_source_hash_exact": manifest.get("source_gate_fit_sha256")
        == sha256_file(source_path),
        "compiler_manifest_output_hash_exact": manifest.get("compiled_gate_sha256")
        == sha256_file(compiled_path),
        "gate_freeze_binds_compiled_gate": gate_freeze.get("compiled_gate_sha256")
        == sha256_file(compiled_path),
        "gate_freeze_binds_compiler_manifest": gate_freeze.get(
            "compiled_gate_manifest_sha256"
        )
        == sha256_file(manifest_path),
        "candidate_architecture_head_identity": gate_freeze.get(
            "selected_candidate_id"
        )
        == source_candidate
        and gate_freeze.get("architecture") == source_architecture
        and int(gate_freeze.get("head_count", -1)) == source_head_count,
        "identity_arrays_exact": all(
            key in source
            and key in compiled
            and np.array_equal(source[key], compiled[key])
            for key in identity_keys
        ),
        "selected_runtime_arrays_already_float32": all(
            key in source and source[key].dtype == np.dtype(np.float32)
            for key in float_keys
        ),
        "compiled_arrays_float32": all(
            key in compiled and compiled[key].dtype == np.dtype(np.float32)
            for key in float_keys
        ),
        "compiled_arrays_bitwise_source_exact": all(
            key in source
            and key in compiled
            and np.array_equal(source[key], compiled[key])
            for key in float_keys
        ),
        "compiler_cast_delta_zero": set(cast_delta) == set(float_keys)
        and all(float(cast_delta[key]) == 0.0 for key in float_keys),
        "compiler_array_hashes_exact": all(
            key in compiled
            and array_hashes.get(key) == compile_gate.array_sha256(compiled[key])
            for key in compiled
        ),
        "thresholds_and_heads_exact": np.array_equal(
            source.get("thresholds"), compiled.get("thresholds")
        )
        and np.array_equal(source.get("head_names"), compiled.get("head_names")),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_id": source_candidate,
        "architecture": source_architecture,
        "head_count": source_head_count,
        "source_gate_fit_sha256": sha256_file(source_path),
        "compiled_gate_sha256": sha256_file(compiled_path),
        "compiled_gate_manifest_sha256": sha256_file(manifest_path),
        "gate_freeze_sha256": sha256_file(gate_freeze_path),
    }
    if not result["passed"]:
        raise PresealError(f"selected/compiler/freeze cross-link failed: {result}")
    return result


def seal_pre_selection() -> dict[str, Any]:
    state = _state()
    if state.get("current_state") != "PRE_SELECTION_SEAL":
        raise PresealError("pre-selection seal may only be made in PRE_SELECTION_SEAL")
    validate_no_python_caches()
    verify_here("sealing")
    counts = outcome_count_snapshot(state)
    expected = {
        "fit_outcome_episodes": int(state.get("expected_fit_episode_count", -1)),
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    if counts != expected:
        raise PresealError(f"pre-selection role chronology failed: {counts} != {expected}")
    pre_data = _verified_checkpoint(PRE_DATA_SEAL_PATH, "PRE_OUTCOME_SEAL")
    replacement_prefix = generator.replacement_registry_prefix(
        permitted_existing_roles=("fit",)
    )
    paths = _files_under(
        ATTEMPT_ROOT / "data/fit",
        ATTEMPT_ROOT / "data/persistence_intents/fit",
        ATTEMPT_ROOT / "data/replacement_registry.json",
        ATTEMPT_ROOT / "data/replacement_claims",
        ATTEMPT_ROOT / "fit",
        ATTEMPT_ROOT / "metrics/fit_power_summary.json",
    )
    _find_status(paths, "all_24_candidates_fit_compiled_and_locked_before_selection_open")
    manifest = _write_stage_manifest(
        PRE_SELECTION_MANIFEST_PATH,
        f"{ACTIVE_ATTEMPT}_fit_role_and_all_candidate_objects_before_selection_open",
        paths,
    )
    files = _stage_files(pre_data["sealed_files"], PRE_SELECTION_MANIFEST_PATH, manifest)
    result = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "PRE_SELECTION_SEAL",
        "status": "all_fit_inputs_transforms_and_24_candidates_locked_before_selection",
        "created_unix_ns": time.time_ns(),
        "passed": True,
        "outcome_counts_at_seal": counts,
        "sealed_files": files,
        "pre_data_seal_sha256": sha256_file(PRE_DATA_SEAL_PATH),
        "fit_manifest_sha256": sha256_file(PRE_SELECTION_MANIFEST_PATH),
        "selection_arrays_opened_by_sealer": 0,
        "selected_head_refit_permitted": False,
        "replacement_registry_prefix": replacement_prefix,
        "authorized_later_replacement_roles": ["selection"],
    }
    _write_json_exclusive(PRE_SELECTION_SEAL_PATH, result)
    return result


def seal_pre_confirmation() -> dict[str, Any]:
    state = _state()
    if state.get("current_state") != "PRE_CONFIRMATION_PACKAGE_SEAL":
        raise PresealError(
            "pre-confirmation seal may only be made in PRE_CONFIRMATION_PACKAGE_SEAL"
        )
    validate_no_python_caches()
    require_mps_device("mps")
    verify_here("sealing", require_mps=True)
    dual = verify_dual_interpreter_contract()
    counts = outcome_count_snapshot(state)
    expected = {
        "fit_outcome_episodes": int(state.get("expected_fit_episode_count", -1)),
        "selection_outcome_episodes": int(
            state.get("expected_selection_episode_count", -1)
        ),
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    if counts != expected:
        raise PresealError(f"pre-confirmation chronology failed: {counts} != {expected}")
    missing_contracts = [
        _repo_relative(path) if path.exists() else str(path)
        for path in VERIFIER_CONTRACT_PATHS
        if not path.is_file() or path.is_symlink()
    ]
    if missing_contracts:
        raise PresealError(
            "all three immutable mode-specific verifier contracts are required: "
            f"{missing_contracts}"
        )
    verifier_contracts = validate_verifier_contracts()
    pre_selection = _verified_checkpoint(
        PRE_SELECTION_SEAL_PATH, "PRE_SELECTION_SEAL"
    )
    pre_selection_prefix = pre_selection.get("replacement_registry_prefix")
    replacement_prefix = generator.replacement_registry_prefix(
        permitted_existing_roles=("fit", "selection")
    )
    if not isinstance(pre_selection_prefix, Mapping):
        raise PresealError("pre-selection seal lacks replacement registry prefix")
    prior_count = pre_selection_prefix.get("claim_count")
    if (
        not isinstance(prior_count, int)
        or isinstance(prior_count, bool)
        or prior_count < 0
        or prior_count > replacement_prefix["claim_count"]
        or pre_selection_prefix.get("genesis") != replacement_prefix["genesis"]
        or pre_selection_prefix.get("claim_files")
        != replacement_prefix["claim_files"][:prior_count]
        or pre_selection_prefix.get("head_record_sha256")
        != (
            replacement_prefix["genesis"]["record_sha256"]
            if prior_count == 0
            else replacement_prefix["claim_files"][prior_count - 1][
                "record_sha256"
            ]
        )
    ):
        raise PresealError(
            "pre-confirmation replacement chain does not exactly extend the sealed fit prefix"
        )
    paths = _files_under(
        ATTEMPT_ROOT / "data/selection",
        ATTEMPT_ROOT / "data/persistence_intents/selection",
        ATTEMPT_ROOT / "data/replacement_registry.json",
        ATTEMPT_ROOT / "data/replacement_claims",
        ATTEMPT_ROOT / "fit",
        ATTEMPT_ROOT / "selection",
        ATTEMPT_ROOT / "freeze",
        ATTEMPT_ROOT / "metrics/fit_power_summary.json",
        ATTEMPT_ROOT / "metrics/selection_power_summary.json",
        ATTEMPT_ROOT / "power_analysis.json",
        ATTEMPT_ROOT / "power_rule.json",
        ATTEMPT_ROOT / "outcome_mapping.json",
        ATTEMPT_ROOT / "analysis.py",
        ATTEMPT_ROOT / "independent_verify.py",
        ATTEMPT_ROOT / "capture_verifier.py",
        *VERIFIER_CONTRACT_PATHS,
    )
    _find_status(paths, "selected_gate_frozen_before_smoke_or_confirmation")
    power_coherence = validate_power_input_coherence(paths)
    compiler_crosslink = validate_compiler_crosslink(paths)
    power = _read_object(ATTEMPT_ROOT / "power_analysis.json")
    confirmation_counts = _validated_confirmation_count_map(
        power.get("confirmation_episode_count_per_regime")
    )
    if (
        power.get("passed") is not True
        or power.get("feasible") is not True
    ):
        raise PresealError("binding power result does not authorize fixed confirmation")
    manifest = _write_stage_manifest(
        PRE_CONFIRMATION_MANIFEST_PATH,
        f"{ACTIVE_ATTEMPT}_selected_gate_power_inference_and_verifier_before_smoke_confirmation",
        paths,
    )
    files = _stage_files(
        pre_selection["sealed_files"], PRE_CONFIRMATION_MANIFEST_PATH, manifest
    )
    result = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "PRE_CONFIRMATION_PACKAGE_SEAL",
        "status": "selected_gate_and_complete_scientific_package_frozen_before_smoke_confirmation",
        "created_unix_ns": time.time_ns(),
        "passed": True,
        "outcome_counts_at_seal": counts,
        "confirmation_episode_count_per_regime": {
            name: confirmation_counts[name] for name in study_common.REGIME_ORDER
        },
        "sealed_files": files,
        "pre_selection_seal_sha256": sha256_file(PRE_SELECTION_SEAL_PATH),
        "pre_confirmation_manifest_sha256": sha256_file(
            PRE_CONFIRMATION_MANIFEST_PATH
        ),
        "dual_interpreter_contract": dual,
        "power_input_coherence": power_coherence,
        "compiler_crosslink": compiler_crosslink,
        "mode_specific_verifier_contracts": verifier_contracts,
        "mps_execution_available_and_required": True,
        "cpu_fallback_permitted": False,
        "smoke_permanently_excluded": True,
        "sequential_confirmation_expansion_permitted": False,
        "replacement_registry_prefix": replacement_prefix,
        "authorized_later_replacement_roles": ["smoke", "confirmation"],
        "postseal_replacement_contract": (
            "only new exclusive contiguous hash-chained claim segments bound to "
            "this PRE_CONFIRMATION_PACKAGE_SEAL, the exact sealed ledger tuple, "
            "and an authenticated mechanical rollout failure"
        ),
    }
    _write_json_exclusive(PRE_CONFIRMATION_SEAL_PATH, result)
    return result


def verify_seal(path: Path, expected_state: str) -> dict[str, Any]:
    value = _verified_checkpoint(path, expected_state)
    sealed_files = value.get("sealed_files")
    if not isinstance(sealed_files, Mapping):
        raise PresealError("seal lacks sealed_files")
    _validate_sealed_files(sealed_files)
    return {
        "passed": True,
        "path": _repo_relative(path),
        "sha256": sha256_file(path),
        "checkpoint_state": expected_state,
        "sealed_file_count": len(sealed_files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "implementation",
        "qualify",
        "seal-pre-data",
        "seal-pre-selection",
        "seal-pre-confirmation",
    ):
        subparsers.add_parser(name)
    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "stage", choices=("pre-data", "pre-selection", "pre-confirmation")
    )
    arguments = parser.parse_args(argv)
    # This is intentionally the first action after argument parsing and occurs
    # before any output directory, seed selection, or artifact creation.
    verify_here("sealing", include_external_hashes=True)
    if arguments.command == "implementation":
        result = implementation_complete()
    elif arguments.command == "qualify":
        result = preseal_qualification()
    elif arguments.command == "seal-pre-data":
        result = seal_pre_data()
    elif arguments.command == "seal-pre-selection":
        result = seal_pre_selection()
    elif arguments.command == "seal-pre-confirmation":
        result = seal_pre_confirmation()
    else:
        mapping = {
            "pre-data": (PRE_DATA_SEAL_PATH, "PRE_OUTCOME_SEAL"),
            "pre-selection": (PRE_SELECTION_SEAL_PATH, "PRE_SELECTION_SEAL"),
            "pre-confirmation": (
                PRE_CONFIRMATION_SEAL_PATH,
                "PRE_CONFIRMATION_PACKAGE_SEAL",
            ),
        }
        result = verify_seal(*mapping[arguments.stage])
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        _exit_code = main()
    except Exception as error:
        print(f"preseal failed closed: {error}", file=os.sys.stderr)
        raise
    raise SystemExit(_exit_code)
