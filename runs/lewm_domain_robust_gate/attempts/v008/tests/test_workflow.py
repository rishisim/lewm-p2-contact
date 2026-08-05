"""Synthetic-only tests for the fail-closed v008 orchestration workflow."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _module():
    specification = importlib.util.spec_from_file_location(
        "workflow_synthetic", ROOT / "workflow.py"
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


workflow = _module()


def _synthetic_replay_qualification(role: str) -> dict[str, Any]:
    return {
        "artifact_type": "synthetic_scientific_replay",
        "role": role,
        "passed": True,
    }


@pytest.fixture(autouse=True)
def _isolate_pre_replay_workflow_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy controller tests synthetic; replay has its own exact tests."""

    monkeypatch.setattr(
        workflow,
        "qualify_role_scientific_replay",
        lambda role, _state, _verification: _synthetic_replay_qualification(role),
    )
    monkeypatch.setattr(
        workflow,
        "_validate_role_replay_qualification",
        lambda _role, verification: dict(
            verification.get("scientific_replay_qualification", {"passed": True})
        ),
    )


STATE_SEQUENCE = (
    "IMPLEMENTATION_COMPLETE",
    "PRESEAL_QUALIFICATION",
    "PRE_OUTCOME_SEAL",
    "FIT_COHORTS",
    "FIT_LOCK",
    "PRE_SELECTION_SEAL",
    "SELECTION_COHORTS",
    "CANDIDATE_SELECTION",
    "GATE_FREEZE",
    "CONFIRMATION_POWER_AND_COHORT_FREEZE",
    "PRE_CONFIRMATION_PACKAGE_SEAL",
    "EXCLUDED_MECHANICAL_SMOKE",
    "CONFIRMATION_GENERATION",
    "CONFIRMATION_EXECUTION",
    "CONFIRMATION_INPUT_SEAL",
    "SEALED_ANALYSIS",
    "LATENCY_AND_RESOURCE_REPORTING",
    "INDEPENDENT_VERIFICATION",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _controller_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ledger_record(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["record_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def _relative(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _adapter_marker(
    *,
    repo: Path,
    adapter: Path,
    root_program: Path,
    event_count: int,
    head_sha256: str,
    operation_sha256: str,
    state_without_marker: Mapping[str, Any],
) -> dict[str, Any]:
    def ast_sha256(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode("utf-8")
        ).hexdigest()

    marker = {
        "schema_version": 2,
        "authorization_kind": "receipt_bound_v008_root_controller_adapter",
        "adapter_source_path": _relative(repo, adapter),
        "adapter_source_sha256": _sha(adapter),
        "adapter_source_ast_sha256": ast_sha256(adapter),
        "root_program_path": _relative(repo, root_program),
        "root_program_sha256": _sha(root_program),
        "root_program_ast_sha256": ast_sha256(root_program),
        "ledger_event_count": event_count,
        "ledger_head_sha256": head_sha256,
        "operation_sha256": operation_sha256,
    }
    marker["state_binding_sha256"] = workflow.canonical_sha256(
        {
            "marker_without_state_binding": marker,
            "state_without_marker": dict(state_without_marker),
        }
    )
    return marker


def _adapter_operation_sha256(
    *,
    base_state: Mapping[str, Any],
    base_ledger_payload: bytes,
    expected_suffix: bytes,
    intended_state_without_marker: Mapping[str, Any],
) -> str:
    return workflow.canonical_sha256(
        {
            "base_state_object_sha256": workflow.canonical_sha256(
                dict(base_state)
            ),
            "base_ledger_sha256": hashlib.sha256(
                base_ledger_payload
            ).hexdigest(),
            "expected_suffix_sha256": hashlib.sha256(
                expected_suffix
            ).hexdigest(),
            "intended_state_object_sha256": workflow.canonical_sha256(
                dict(intended_state_without_marker)
            ),
        }
    )


def test_frozen_facade_provenance_path_map_exception_is_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "contact-checkout" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    exact = {str(source.resolve()): _sha(source)}
    monkeypatch.setattr(workflow, "FROZEN_FACADE_SOURCE_HASHES", exact)

    workflow._assert_no_forbidden_keys(
        {"frozen_facade": {"sources": exact}}, location="execution.fit.native_plan"
    )

    for drift in (
        {str(source.resolve()): "0" * 64},
        {**exact, str(source.with_name("extra.py").resolve()): "1" * 64},
        {str(source.parent) + "/./source.py": _sha(source)},
    ):
        with pytest.raises(workflow.WorkflowError, match="provenance map drift"):
            workflow._assert_no_forbidden_keys(
                {"frozen_facade": {"sources": drift}},
                location="execution.fit.native_plan",
            )

    with pytest.raises(workflow.WorkflowError, match="forbidden preterminal key"):
        workflow._assert_no_forbidden_keys(
            {"contact_path": str(source)}, location="execution.fit.native_plan"
        )
    with pytest.raises(workflow.WorkflowError, match="forbidden preterminal key"):
        workflow._assert_no_forbidden_keys(
            {"sources": exact}, location="execution.fit.native_plan"
        )


class FakeController:
    def __init__(self, current: str, **updates: Any) -> None:
        self.value: dict[str, Any] = {
            "active_attempt": "v008",
            "current_state": current,
            "completed_states": list(STATE_SEQUENCE[: STATE_SEQUENCE.index(current)]),
            "expected_fit_episode_count": 1200,
            "expected_selection_episode_count": 2000,
            "fit_outcome_episodes": 0,
            "selection_outcome_episodes": 0,
            "smoke_outcome_episodes": 0,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
        }
        self.value.update(updates)
        self.count_calls: list[dict[str, Any]] = []
        self.advance_calls: list[tuple[str, Path]] = []
        self.early_calls: list[str] = []
        self.integrity_calls = 0

    def status(self) -> dict[str, Any]:
        return copy.deepcopy(self.value)

    def advance(
        self, completed_state: str, evidence: Path, _name: str, _next: str
    ) -> dict[str, Any]:
        assert self.value["current_state"] == completed_state
        value = json.loads(evidence.read_text(encoding="utf-8"))
        assert value["passed"] is True
        assert value["checkpoint_state"] == completed_state
        self.advance_calls.append((completed_state, evidence))
        self.value["completed_states"].append(completed_state)
        self.value["current_state"] = STATE_SEQUENCE[
            STATE_SEQUENCE.index(completed_state) + 1
        ]
        return self.status()

    def update_counts(self, **counts: Any) -> dict[str, Any]:
        self.count_calls.append(dict(counts))
        self.value.update(counts)
        return self.status()

    def stage_early_scientific_failure(self, mode: str) -> dict[str, Any]:
        self.early_calls.append(mode)
        self.value["current_state"] = "INDEPENDENT_VERIFICATION"
        self.value["early_scientific_failure"] = {"mode": mode}
        return self.status()

    def stage_postconfirmation_integrity_failure(self) -> dict[str, Any]:
        self.integrity_calls += 1
        self.value["current_state"] = "INDEPENDENT_VERIFICATION"
        self.value["postconfirmation_integrity_failure"] = {
            "source": "analysis_execution"
        }
        return self.status()


def _role_verification(role: str) -> dict[str, Any]:
    per_dgp = workflow.ROLE_EPISODES_PER_DGP[role]
    total = per_dgp * len(workflow.REGIMES)
    return {
        "role": role,
        "regime_order": list(workflow.REGIMES),
        "episodes_per_regime": per_dgp,
        "episode_count": total,
        "row_count": total * workflow.ROWS_PER_EPISODE,
        "regimes": {name: {"episode_count": per_dgp} for name in workflow.REGIMES},
        "aggregate_paths": {name: f"synthetic/{name}.npz" for name in workflow.REGIMES},
        "aggregate_hashes": {
            f"synthetic/{name}.npz": str(index) * 64
            for index, name in enumerate(workflow.REGIMES, start=1)
        },
        "development_part_arrays_opened": True,
        "aggregate_arrays_opened": False,
        "npz_arrays_opened_during_manifest_verification": True,
        "passed": True,
    }


def _audit_path(tmp_path: Path, state: str) -> dict[str, Path]:
    return {state: tmp_path / f"{state.lower()}.json"}


def test_role_counter_updates_only_after_all_four_exact_manifests(
    tmp_path: Path,
) -> None:
    controller = FakeController("FIT_COHORTS")
    calls = 0

    def verify(role: str, _state: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert role == "fit"
        return _role_verification(role)

    driver = workflow.Workflow(
        controller,
        object(),
        evidence_paths=_audit_path(tmp_path, "FIT_COHORTS"),
        manifest_verifier=verify,
        inputs_complete=lambda role: role == "fit",
    )
    result = driver.step()
    assert result["advanced"] is True
    assert calls == 1
    assert controller.count_calls == [{"fit_outcome_episodes": 1200}]
    assert controller.advance_calls[0][0] == "FIT_COHORTS"
    audit = json.loads(
        (tmp_path / "fit_cohorts.json").read_text(encoding="utf-8")
    )
    assert audit["checks"]["all_four_regimes_verified"] is True
    assert audit["evidence"]["role_verification"][
        "npz_arrays_opened_during_manifest_verification"
    ] is True
    assert audit["evidence"]["role_verification"][
        "aggregate_arrays_opened"
    ] is False


def test_role_verification_failure_never_updates_counter_or_checkpoint(
    tmp_path: Path,
) -> None:
    controller = FakeController("SELECTION_COHORTS")
    path = tmp_path / "selection.json"

    def fail(*_args: Any) -> dict[str, Any]:
        raise workflow.WorkflowError("synthetic manifest drift")

    driver = workflow.Workflow(
        controller,
        object(),
        evidence_paths={"SELECTION_COHORTS": path},
        manifest_verifier=fail,
        inputs_complete=lambda _role: True,
    )
    with pytest.raises(workflow.WorkflowError, match="synthetic manifest drift"):
        driver.step()
    assert controller.count_calls == []
    assert controller.advance_calls == []
    assert not path.exists()


def test_exact_role_count_without_authenticated_crash_record_fails_closed(
    tmp_path: Path,
) -> None:
    controller = FakeController("FIT_COHORTS", fit_outcome_episodes=1200)
    driver = workflow.Workflow(
        controller,
        object(),
        evidence_paths=_audit_path(tmp_path, "FIT_COHORTS"),
        manifest_verifier=lambda role, _state: _role_verification(role),
        inputs_complete=lambda _role: True,
    )
    with pytest.raises(workflow.WorkflowError, match="authenticated crash recovery"):
        driver.step()
    assert controller.count_calls == []
    assert controller.advance_calls == []


def _prior_selection_audit(path: Path) -> None:
    _json(
        path,
        {
            "schema_version": 1,
            "attempt": "v008",
            "checkpoint_state": "SELECTION_COHORTS",
            "passed": True,
            "evidence": {"role_verification": _role_verification("selection")},
        },
    )


def test_no_candidate_uses_mode_specific_early_branch_without_advance(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "selection_cohorts.json"
    _prior_selection_audit(prior)
    controller = FakeController("CANDIDATE_SELECTION")

    class Operations:
        @staticmethod
        def select(_verification: dict[str, Any]) -> dict[str, Any]:
            return {
                "status": "selection_complete_no_selected_head_refit",
                "candidate_count": 24,
                "eligible_count": 0,
                "selected_candidate_id": None,
                "selected_candidate_index": None,
                "selected_head_refit_after_selection": False,
                "prior_confirmation_outcome_episodes_used": 0,
            }

    driver = workflow.Workflow(
        controller,
        Operations(),
        evidence_paths={"SELECTION_COHORTS": prior},
    )
    result = driver.step()
    assert result["mode"] == "no_candidate"
    assert controller.early_calls == ["no_candidate"]
    assert controller.advance_calls == []


def _configure_power_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, feasible: bool = True
) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    attempt = repo / "runs/lewm_domain_robust_gate/attempts/v008"
    attempt.mkdir(parents=True)
    monkeypatch.setattr(workflow, "REPO_ROOT", repo)
    paths = {
        "FIT_LOCK_PATH": attempt / "fit/fit_lock.json",
        "FITTED_PATH": attempt / "fit/fitted_candidates.npz",
        "SELECTION_LEDGER_PATH": attempt / "selection/selection_ledger.json",
        "GATE_FREEZE_PATH": attempt / "freeze/gate_freeze.json",
        "FIT_POWER_SUMMARY_PATH": attempt / "metrics/fit_power_summary.json",
        "SELECTION_POWER_SUMMARY_PATH": attempt / "metrics/selection_power_summary.json",
        "POWER_PATH": attempt / "power_analysis.json",
        "POWER_RULE_PATH": attempt / "power_rule.json",
        "ANALYSIS_RESULT_PATH": attempt / "analysis_result.json",
        "ANALYSIS_INVALID_PATH": attempt / "audit/analysis_execution_invalid.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(workflow, name, path)
    monkeypatch.setattr(workflow.power_analysis, "REPO_ROOT", repo)
    selected = "gate-07"
    object_sha = "a" * 64
    paths["FITTED_PATH"].parent.mkdir(parents=True, exist_ok=True)
    paths["FITTED_PATH"].write_bytes(b"synthetic fitted candidate source\n")
    paths["POWER_RULE_PATH"].write_bytes(
        workflow.power_analysis.POWER_RULE_PATH.read_bytes()
    )
    _json(
        paths["FIT_LOCK_PATH"],
        {
            "status": "all_24_candidates_fit_compiled_and_locked_before_selection_open",
            "selection_input_opened_before_lock": False,
            "fitted_candidates_sha256": _sha(paths["FITTED_PATH"]),
        },
    )
    fit_lock_sha256 = _sha(paths["FIT_LOCK_PATH"])
    _json(
        paths["SELECTION_LEDGER_PATH"],
        {
            "status": "selection_complete_no_selected_head_refit",
            "selected_candidate_id": selected,
            "selected_candidate_index": 0,
            "selected_head_refit_after_selection": False,
            "fit_lock_sha256": fit_lock_sha256,
            "fitted_candidates_sha256": _sha(paths["FITTED_PATH"]),
            "candidates": [
                {
                    "candidate_id": selected,
                    "candidate_object_sha256": object_sha,
                    "eligible": True,
                }
            ],
        },
    )
    hashes = {
        "fit_lock_sha256": fit_lock_sha256,
        "selection_ledger_sha256": _sha(paths["SELECTION_LEDGER_PATH"]),
    }
    _json(
        paths["GATE_FREEZE_PATH"],
        {
            "status": "selected_gate_frozen_before_smoke_or_confirmation",
            "created_unix_ns": 100,
            "selected_candidate_id": selected,
            "selected_candidate_object_sha256": object_sha,
            "selected_head_refit_after_selection": False,
            "confirmation_episodes_at_freeze": 0,
            **hashes,
        },
    )
    hashes["gate_freeze_sha256"] = _sha(paths["GATE_FREEZE_PATH"])
    identity = {
        "selected_candidate_id": selected,
        "selected_candidate_object_sha256": object_sha,
        **hashes,
    }
    common = {
        "schema_version": 1,
        "selected_candidate_id": selected,
        "selected_candidate_object_sha256": object_sha,
        "co_primary_only": True,
        "auxiliary_robust_whitening_excluded": True,
    }
    fit_source = {
        **common,
        "role": "fit",
        "claims": {
            regime: {
                endpoint: {"episode_count": 300, "mean": 1.0, "sd_ddof1": 0.1}
                for endpoint in workflow.power_analysis.ENDPOINTS
            }
            for regime in workflow.REGIMES
        },
    }
    selection_source = {
        **common,
        "role": "selection",
        "claims": {
            regime: {
                endpoint: {
                    "episode_count": 500,
                    "mean": (
                        0.0
                        if (
                            not feasible
                            and regime == "markov_oracle"
                            and endpoint == "raw_vs_analytic"
                        )
                        else 0.8
                    ),
                    "sd_ddof1": 0.08,
                }
                for endpoint in workflow.power_analysis.ENDPOINTS
            }
            for regime in workflow.REGIMES
        },
    }
    stored_fit = copy.deepcopy(fit_source)
    stored_selection = copy.deepcopy(selection_source)
    stored_fit.update(identity)
    stored_selection.update(identity)
    stored_fit["attempt"] = "v008"
    stored_selection["attempt"] = "v008"
    _json(paths["FIT_POWER_SUMMARY_PATH"], stored_fit)
    _json(paths["SELECTION_POWER_SUMMARY_PATH"], stored_selection)
    monkeypatch.setattr(
        workflow.fit_select,
        "load_fitted_candidates",
        lambda _path: {"synthetic": True},
    )
    monkeypatch.setattr(
        workflow.fit_select,
        "selected_power_summaries",
        lambda _fitted, _ledger: (
            copy.deepcopy(fit_source),
            copy.deepcopy(selection_source),
        ),
    )
    return attempt, identity


def _power_result(
    attempt: Path, identity: dict[str, str], *, feasible: bool
) -> dict[str, Any]:
    fit_summary = json.loads(
        workflow.FIT_POWER_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    selection_summary = json.loads(
        workflow.SELECTION_POWER_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    result = workflow.power_analysis.compute_binding_power(
        fit_summary,
        selection_summary,
        workflow.power_analysis.load_frozen_power_rule_moments(
            workflow.POWER_RULE_PATH
        ),
    )
    assert result["selected_gate_identity"] == identity
    assert result["feasible"] is feasible
    identity_artifacts = workflow.power_analysis.validate_identity_artifacts(
        identity,
        fit_lock_path=workflow.FIT_LOCK_PATH,
        selection_ledger_path=workflow.SELECTION_LEDGER_PATH,
        gate_freeze_path=workflow.GATE_FREEZE_PATH,
    )
    result["created_unix_ns"] = 101
    result["inputs"] = {
        "fit_summary": {
            "path": workflow.relative_to_repo(workflow.FIT_POWER_SUMMARY_PATH),
            "sha256": _sha(workflow.FIT_POWER_SUMMARY_PATH),
        },
        "selection_summary": {
            "path": workflow.relative_to_repo(workflow.SELECTION_POWER_SUMMARY_PATH),
            "sha256": _sha(workflow.SELECTION_POWER_SUMMARY_PATH),
        },
        "power_rule": {
            "path": workflow.relative_to_repo(workflow.POWER_RULE_PATH),
            "sha256": _sha(workflow.POWER_RULE_PATH),
            "expected_sha256": workflow.power_analysis.POWER_RULE_SHA256,
            "endpoint_moments_object_sha256": (
                workflow.power_analysis.POWER_RULE_MOMENTS_SHA256
            ),
        },
        **identity_artifacts,
    }
    return result


def _tamper_power_result(result: dict[str, Any], mode: str) -> dict[str, Any]:
    tampered = copy.deepcopy(result)
    claim = tampered["claim_design_inputs"]["native_plan"]["raw_vs_analytic"]
    if mode == "coherent_n":
        selected = next(
            row for row in tampered["power_grid"]
            if row["episodes_per_regime"] == 1_000
        )
        tampered["selected_grid_record"] = copy.deepcopy(selected)
        tampered["selected_confirmation_episodes_per_regime"] = 1_000
        tampered["confirmation_episode_count_per_regime"] = {
            regime: 1_000 for regime in workflow.REGIMES
        }
        tampered["selected_confirmation_slots_per_regime"] = [0, 999]
    elif mode == "fit_mean":
        claim["fit"]["mean"] = float(claim["fit"]["mean"]) / 2.0
    elif mode == "selection_sd":
        claim["selection"]["sd_ddof1"] = (
            float(claim["selection"]["sd_ddof1"]) * 10.0
        )
    elif mode == "derived_ucl":
        claim["fit_sd_upper_95"] = float(claim["fit_sd_upper_95"]) + 1e-6
    elif mode == "alpha":
        tampered["per_claim_alpha"] = 0.01
    elif mode == "grid":
        first = tampered["power_grid"][0]
        first["union_bound_family_power_lower"] = (
            float(first["union_bound_family_power_lower"]) + 1e-6
        )
    elif mode == "coherent_infeasible":
        tampered.update(
            {
                "feasible": False,
                "passed": False,
                "decision": "power_infeasible_no_confirmation",
                "selected_grid_record": None,
                "selected_confirmation_episodes_per_regime": None,
                "confirmation_episode_count_per_regime": None,
                "selected_confirmation_slots_per_regime": None,
                "fixed_prefix_only": False,
                "terminal_label_if_infeasible": "domain_robust_gate_failed",
                "confirmation_generation_authorized_by_power": False,
            }
        )
    elif mode == "input_hash":
        tampered["inputs"]["fit_summary"]["sha256"] = "0" * 64
    elif mode == "extra_key":
        tampered["posthoc_override"] = True
    elif mode == "nested_extra_key":
        tampered["inputs"]["fit_summary"]["posthoc_override"] = True
    elif mode == "stale_timestamp":
        tampered["created_unix_ns"] = 100
    else:  # pragma: no cover - test fixture misuse
        raise AssertionError(mode)
    return tampered


@pytest.mark.parametrize("feasible", [True, False])
def test_power_identity_branch_and_controller_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, feasible: bool
) -> None:
    attempt, identity = _configure_power_paths(
        monkeypatch, tmp_path, feasible=feasible
    )
    result = _power_result(attempt, identity, feasible=feasible)
    _json(workflow.POWER_PATH, result)
    controller = FakeController("CONFIRMATION_POWER_AND_COHORT_FREEZE")

    class Operations:
        @staticmethod
        def power() -> dict[str, Any]:
            return result

    audit = attempt / "audit/power_checkpoint.json"
    driver = workflow.Workflow(
        controller,
        Operations(),
        evidence_paths={"CONFIRMATION_POWER_AND_COHORT_FREEZE": audit},
    )
    observed = driver.step()
    if feasible:
        checkpoint = json.loads(audit.read_text(encoding="utf-8"))
        assert checkpoint["fixed_confirmation_episode_count"] == 2000
        assert checkpoint["fixed_confirmation_episodes_per_regime"] == 500
        assert controller.advance_calls[0][0] == (
            "CONFIRMATION_POWER_AND_COHORT_FREEZE"
        )
        assert controller.early_calls == []
    else:
        assert observed["mode"] == "power_infeasible"
        assert controller.early_calls == ["power_infeasible"]
        assert controller.advance_calls == []
        assert not audit.exists()


@pytest.mark.parametrize(
    "tamper",
    (
        "coherent_n",
        "fit_mean",
        "selection_sd",
        "derived_ucl",
        "alpha",
        "grid",
        "coherent_infeasible",
        "input_hash",
        "extra_key",
        "nested_extra_key",
        "stale_timestamp",
    ),
)
def test_power_recomputation_rejects_stale_or_forged_existing_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    attempt, identity = _configure_power_paths(monkeypatch, tmp_path)
    forged = _tamper_power_result(
        _power_result(attempt, identity, feasible=True), tamper
    )
    _json(workflow.POWER_PATH, forged)
    controller = FakeController("CONFIRMATION_POWER_AND_COHORT_FREEZE")

    class Operations:
        @staticmethod
        def power() -> dict[str, Any]:
            return forged

    audit = attempt / "audit/power_checkpoint.json"
    driver = workflow.Workflow(
        controller,
        Operations(),
        evidence_paths={"CONFIRMATION_POWER_AND_COHORT_FREEZE": audit},
    )
    with pytest.raises(workflow.WorkflowError, match="binding-power"):
        driver.step()
    assert controller.advance_calls == []
    assert controller.early_calls == []
    assert not audit.exists()


def test_default_freeze_writes_identity_summaries_only_after_gate_freeze(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    attempt = repo / "runs/lewm_domain_robust_gate/attempts/v008"
    monkeypatch.setattr(workflow, "REPO_ROOT", repo)
    monkeypatch.setattr(workflow, "ATTEMPT_ROOT", attempt)
    names = (
        "FITTED_PATH",
        "FIT_LOCK_PATH",
        "SELECTION_LEDGER_PATH",
        "GATE_FIT_PATH",
        "COMPILED_GATE_PATH",
        "COMPILER_MANIFEST_PATH",
        "GATE_FREEZE_PATH",
        "FIT_POWER_SUMMARY_PATH",
        "SELECTION_POWER_SUMMARY_PATH",
    )
    relatives = (
        "fit/fitted_candidates.npz",
        "fit/fit_lock.json",
        "selection/selection_ledger.json",
        "freeze/gate_fit.npz",
        "freeze/compiled_gate.npz",
        "freeze/compiled_gate_manifest.json",
        "freeze/gate_freeze.json",
        "metrics/fit_power_summary.json",
        "metrics/selection_power_summary.json",
    )
    for name, relative in zip(names, relatives):
        monkeypatch.setattr(workflow, name, attempt / relative)
    workflow.FITTED_PATH.parent.mkdir(parents=True)
    workflow.SELECTION_LEDGER_PATH.parent.mkdir(parents=True)
    workflow.GATE_FIT_PATH.parent.mkdir(parents=True)
    workflow.FITTED_PATH.write_bytes(b"locked-fitted")
    workflow.GATE_FIT_PATH.write_bytes(b"selected-fit-only-gate")
    _json(workflow.FIT_LOCK_PATH, {"locked": True})
    _json(workflow.SELECTION_LEDGER_PATH, {"selected_candidate_id": "gate-07"})
    monkeypatch.setattr(workflow, "verify_here", lambda *_args, **_kwargs: None)

    def compile_file(_source: Path, output: Path, manifest: Path) -> None:
        output.write_bytes(b"compiled")
        _json(manifest, {"compiled": True})

    monkeypatch.setattr(workflow.compile_gate, "compile_gate_file", compile_file)

    def seal_gate_freeze(**_kwargs: Any) -> dict[str, Any]:
        value = {
            "status": "selected_gate_frozen_before_smoke_or_confirmation",
            "selected_candidate_id": "gate-07",
            "selected_candidate_object_sha256": "a" * 64,
            "fit_lock_sha256": _sha(workflow.FIT_LOCK_PATH),
            "selection_ledger_sha256": _sha(workflow.SELECTION_LEDGER_PATH),
            "gate_fit_sha256": _sha(workflow.GATE_FIT_PATH),
            "compiled_gate_sha256": _sha(workflow.COMPILED_GATE_PATH),
            "compiled_gate_manifest_sha256": _sha(workflow.COMPILER_MANIFEST_PATH),
            "compiled_runtime_arrays_bitwise_equal_selected_gate": True,
            "selected_head_refit_after_selection": False,
            "confirmation_episodes_at_freeze": 0,
        }
        _json(workflow.GATE_FREEZE_PATH, value)
        return value

    monkeypatch.setattr(workflow.fit_select, "seal_gate_freeze", seal_gate_freeze)
    monkeypatch.setattr(
        workflow.fit_select, "load_fitted_candidates", lambda _path: "locked-only"
    )
    call_order: list[str] = []

    def selected_power_summaries(
        fitted: Any, ledger: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert fitted == "locked-only"
        assert ledger["selected_candidate_id"] == "gate-07"
        assert workflow.GATE_FREEZE_PATH.is_file()
        assert not workflow.FIT_POWER_SUMMARY_PATH.exists()
        assert not workflow.SELECTION_POWER_SUMMARY_PATH.exists()
        call_order.append("summaries_after_freeze")
        common = {
            "schema_version": 1,
            "co_primary_only": True,
            "auxiliary_robust_whitening_excluded": True,
        }
        return ({**common, "role": "fit"}, {**common, "role": "selection"})

    monkeypatch.setattr(
        workflow.fit_select, "selected_power_summaries", selected_power_summaries
    )
    monkeypatch.setattr(
        workflow.fit_select,
        "select_after_fit_lock",
        lambda *_args, **_kwargs: pytest.fail("freeze must never reopen/refit selection"),
    )
    workflow.DefaultOperations().freeze()
    assert call_order == ["summaries_after_freeze"]
    for path in (
        workflow.FIT_POWER_SUMMARY_PATH,
        workflow.SELECTION_POWER_SUMMARY_PATH,
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["attempt"] == "v008"
        assert value["gate_freeze_sha256"] == _sha(workflow.GATE_FREEZE_PATH)
        assert value["selected_candidate_id"] == "gate-07"


def test_sealed_analysis_integrity_failure_is_staged_not_advanced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempt, _identity = _configure_power_paths(monkeypatch, tmp_path)
    invalid = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "SEALED_ANALYSIS",
        "passed": False,
        "process_valid": False,
        "integrity_passed": False,
        "execution_invalid": True,
        "confirmation_outcomes_opened_for_analysis": True,
        "analysis_result_present": True,
    }
    _json(workflow.ANALYSIS_RESULT_PATH, {"process_valid": False})
    _json(workflow.ANALYSIS_INVALID_PATH, invalid)
    controller = FakeController(
        "SEALED_ANALYSIS",
        confirmation_outcomes_opened_for_analysis=True,
    )
    observed = workflow.Workflow(controller, object()).step()
    assert observed["action"] == "postconfirmation_execution_invalid_staged"
    assert controller.integrity_calls == 1
    assert controller.advance_calls == []
    assert observed["source"]["path"] == _relative(
        attempt.parents[3], workflow.ANALYSIS_INVALID_PATH
    )


def _synthetic_role_manifests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    attempt = repo / "runs/lewm_domain_robust_gate/attempts/v008"
    study = attempt.parents[1]
    source = study / "attempts/v003"
    study.mkdir(parents=True)
    state_path = study / "STATE.json"
    controller_adapter_path = attempt / "version_forward_transaction.py"
    root_program_path = study / "program.py"
    controller_adapter_path.parent.mkdir(parents=True, exist_ok=True)
    controller_adapter_path.write_text("VALUE = 'synthetic adapter'\n", encoding="utf-8")
    root_program_path.write_text("VALUE = 'synthetic root'\n", encoding="utf-8")
    state = {
        "active_attempt": "v008",
        "active_attempt_path": _relative(repo, attempt),
        "current_state": "FIT_COHORTS",
        "expected_fit_episode_count": len(workflow.REGIMES),
        "fit_outcome_episodes": 0,
    }
    _controller_json(state_path, state)
    monkeypatch.setattr(workflow, "REPO_ROOT", repo)
    monkeypatch.setattr(workflow, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(workflow, "STATE_PATH", state_path)
    monkeypatch.setattr(
        workflow, "CONTROLLER_ADAPTER_PATH", controller_adapter_path
    )
    monkeypatch.setattr(workflow, "ROOT_PROGRAM_PATH", root_program_path)
    monkeypatch.setattr(workflow, "ROLE_EPISODES_PER_DGP", {"fit": 1, "selection": 1})
    dgp_path = source / "DGP_MATRIX.json"
    ledger_path = source / "cohort_seed_ledger.json"
    registry_path = source / "data/replacement_registry.json"
    _json(
        dgp_path,
        {
            "regimes": {
                regime: {"synthetic": True} for regime in workflow.REGIMES
            }
        },
    )
    _json(ledger_path, {"synthetic": "cohort ledger"})
    _json(registry_path, {"synthetic": "replacement registry"})
    monkeypatch.setattr(workflow, "DGP_MATRIX_PATH", dgp_path)
    monkeypatch.setattr(workflow, "COHORT_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(workflow, "REPLACEMENT_REGISTRY_PATH", registry_path)
    source_seal_path = source / "audit/pre_data_inheritance_seal.json"
    _json(
        source_seal_path,
        {
            "schema_version": 1,
            "attempt": "v003",
            "science_attempt": "v001",
            "checkpoint_state": "PRE_OUTCOME_SEAL",
            "passed": True,
        },
    )
    source_state_sha256 = "e" * 64
    source_invalidity_path = source / "audit/v003_procedural_invalidity.json"
    _json(
        source_invalidity_path,
        {
            "attempt": "v003",
            "frozen_evidence": {
                "controller_state_sha256_at_failure": source_state_sha256
            },
        },
    )
    inventory_path = attempt / "audit/inherited_fit_inventory.json"
    _json(inventory_path, {"attempt": "v008", "passed": True})
    expected_inventory_file_count = 1
    seal_path = attempt / "audit/pre_data_inheritance_seal.json"
    _json(
        seal_path,
        {
            "schema_version": 1,
            "attempt": "v008",
            "source_attempt": "v006",
            "target_attempt": "v008",
            "inherited_fit_role": {
                "source_attempt": "v003",
                "inventory_path": _relative(repo, inventory_path),
                "inventory_sha256": _sha(inventory_path),
                "file_count": expected_inventory_file_count,
                "episode_count": len(workflow.REGIMES),
                "outcome_arrays_opened": False,
                "passed": True,
            },
            "passed": True,
        },
    )
    monkeypatch.setattr(workflow, "INHERITANCE_SEAL_PATH", seal_path)
    monkeypatch.setattr(workflow.fit_inheritance, "SOURCE_ATTEMPT", "v003")
    monkeypatch.setattr(workflow.fit_inheritance, "SOURCE_ROOT", source)
    monkeypatch.setattr(
        workflow.fit_inheritance, "SOURCE_FIT_ROOT", source / "data/fit"
    )
    monkeypatch.setattr(
        workflow.fit_inheritance, "SOURCE_INVALIDITY_PATH", source_invalidity_path
    )
    monkeypatch.setattr(
        workflow.fit_inheritance, "SOURCE_SEAL_PATH", source_seal_path
    )
    monkeypatch.setattr(workflow.fit_inheritance, "INVENTORY_PATH", inventory_path)
    monkeypatch.setattr(
        workflow.fit_inheritance, "EXPECTED_FILE_COUNT", expected_inventory_file_count
    )
    monkeypatch.setattr(
        workflow.fit_inheritance,
        "verify_inventory",
        lambda: {
            "episode_count": len(workflow.REGIMES),
            "file_count": expected_inventory_file_count,
            "outcome_arrays_opened": False,
            "passed": True,
        },
    )
    monkeypatch.setattr(
        workflow,
        "ROLE_RAW_SEALS",
        {
            "fit": (source_seal_path, "PRE_OUTCOME_SEAL"),
            "selection": (seal_path, "PRE_OUTCOME_SEAL"),
        },
    )
    monkeypatch.setattr(
        workflow,
        "ROLE_EXECUTION_SEALS",
        {
            "fit": (source_seal_path, "PRE_OUTCOME_SEAL"),
            "selection": (seal_path, "PRE_SELECTION_SEAL"),
        },
    )
    seal = {
        "path": _relative(repo, source_seal_path),
        "sha256": _sha(source_seal_path),
        "checkpoint_state": "PRE_OUTCOME_SEAL",
    }
    receipt_path = attempt / "audit/version_forward_transaction_receipt.json"
    _json(receipt_path, {"passed": True})
    receipt = {
        "path": _relative(repo, receipt_path),
        "sha256": _sha(receipt_path),
        "passed": True,
    }
    def inherited_authorization(**kwargs: Any) -> dict[str, Any]:
        recovery_role = kwargs.get("authorized_role_count_recovery")
        recovery: dict[str, Any] | None = None
        if recovery_role is not None:
            controller = json.loads(workflow.STATE_PATH.read_text(encoding="utf-8"))
            counter = workflow.ROLE_COUNTER_FIELDS[recovery_role]
            expected_key = (
                "expected_fit_episode_count"
                if recovery_role == "fit"
                else "expected_selection_episode_count"
            )
            manifest_root = (
                workflow.fit_inheritance.SOURCE_ROOT
                if recovery_role == "fit"
                else attempt
            )
            manifest_path = (
                manifest_root
                / "data"
                / recovery_role
                / workflow.REGIMES[0]
                / "execution_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = [
                json.loads(line)
                for line in workflow.LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            ]
            audit_path = attempt / "audit" / f"{recovery_role}_cohorts.json"
            audit = (
                json.loads(audit_path.read_text(encoding="utf-8"))
                if audit_path.exists()
                else None
            )
            recovery = {
                "passed": True,
                "role": recovery_role,
                "state": workflow.ROLE_STATES[recovery_role],
                "counter_field": counter,
                "counter_value": controller[expected_key],
                "count_update_event_seq": len(records),
                "count_update_event_sha256": records[-1]["record_sha256"],
                "pre_count_state_sha256": manifest["authorization"][
                    "state_sha256"
                ],
                "prior_controller_adapter_marker_sha256": (
                    workflow.canonical_sha256(
                        records[-1]["prior_controller_adapter_marker"]
                    )
                ),
                "role_audit_sha256": _sha(audit_path)
                if audit_path.exists()
                else None,
                "role_audit_created_unix_ns": (
                    audit.get("created_unix_ns") if audit is not None else None
                ),
                "no_later_output_conflicts": True,
            }
        return {
            "seal_path": seal["path"],
            "seal_sha256": seal["sha256"],
            "authenticated_state_sha256": _sha(workflow.STATE_PATH),
            "version_forward_transaction_receipt": receipt,
            "role_count_recovery_authorization": recovery,
            "independent_recomputation": {
                "authorized_role_count_recovery": recovery_role,
            },
        }

    monkeypatch.setattr(
        workflow,
        "verify_inherited_pre_data_authorization",
        inherited_authorization,
    )
    source_bindings: dict[str, dict[str, str]] = {}
    canonical_sources = {
        "dgp_matrix": dgp_path,
        "cohort_seed_ledger": ledger_path,
        "execution_authorization_seal": source_seal_path,
        "raw_authorization_seal": source_seal_path,
        "counted_feature_source": source / "counted_features.py",
        "pricing_source": source / "flops.py",
        "runner_source": source / "runner.py",
        "replacement_registry": registry_path,
    }
    for name in workflow.SOURCE_BINDING_KEYS:
        path = canonical_sources[name]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# synthetic {name}\n", encoding="utf-8")
        source_bindings[name] = {
            "path": _relative(repo, path),
            "sha256": _sha(path),
        }
    for regime in workflow.REGIMES:
        root = source / "data/fit" / regime
        raw_dir = root / "raw"
        execution_dir = root / "execution"
        intent_dir = source / "data/persistence_intents/fit" / regime
        raw_dir.mkdir(parents=True)
        execution_dir.mkdir(parents=True)
        intent_dir.mkdir(parents=True)
        episode_id = f"synthetic-{regime}"
        raw_path = raw_dir / f"{episode_id}.npz"
        raw_path.write_bytes(b"synthetic raw bytes, never NumPy-loaded")
        intent_path = intent_dir / f"{episode_id}.json"
        _json(intent_path, {"intent": episode_id})
        raw_loader_audit = {
            "loaded_keys": ["action", "pixels"],
            "pixels_shape": [201, 224, 224, 3],
            "pixels_dtype": "uint8",
            "action_shape": [201, 5],
            "action_dtype": "float32",
            "pixels_finite": True,
            "modeled_actions_finite": True,
            "terminal_action_nan_sentinel": True,
            "array_sha256": {"action": "a" * 64, "pixels": "b" * 64},
            "archive_keys": ["action", "pixels"],
            "archive_sha256": _sha(raw_path),
            "arrays_materialized": ["action", "pixels"],
            "non_input_arrays_materialized": False,
        }
        raw_record = {
            "role": "fit",
            "regime": regime,
            "complete": True,
            "slot": 0,
            "episode_id": episode_id,
            "raw_archive_members": ["action", "pixels"],
            "authorization_seal": seal,
            "arrays": {
                "action": {
                    "shape": [201, 5],
                    "dtype": "float32",
                    "sha256": "a" * 64,
                },
                "pixels": {
                    "shape": [201, 224, 224, 3],
                    "dtype": "uint8",
                    "sha256": "b" * 64,
                },
            },
            "input_loader_audit": raw_loader_audit,
            "raw_path": _relative(repo, raw_path),
            "raw_sha256": _sha(raw_path),
            "persistence_intent_path": _relative(repo, intent_path),
            "persistence_intent_sha256": _sha(intent_path),
            "replacement_used": False,
        }
        raw_sidecar = raw_path.with_suffix(".json")
        _json(raw_sidecar, raw_record)
        raw_manifest_path = root / "raw_manifest.json"
        raw_manifest = {
            "schema_version": 1,
            "attempt": "v003",
            "created_unix_ns": 1,
            "complete": True,
            "role": "fit",
            "regime": regime,
            "regime_specification": {"synthetic": True},
            "episode_count": 1,
            "episodes": [raw_record],
            "replacements_used": 0,
            "raw_archive_members": ["action", "pixels"],
            "raw_pixels_contract": "uint8 [201,224,224,3]",
            "raw_action_contract": "float32 [201,5]; 200 finite plus terminal NaN",
            "role_isolation": True,
            "smoke_permanently_excluded": False,
            "retention_contract": "pixels_action_shapes_finiteness_and_local_step_count_only",
            "dgp_matrix_path": source_bindings["dgp_matrix"]["path"],
            "dgp_matrix_sha256": source_bindings["dgp_matrix"]["sha256"],
            "cohort_seed_ledger_path": source_bindings["cohort_seed_ledger"]["path"],
            "cohort_seed_ledger_sha256": source_bindings["cohort_seed_ledger"]["sha256"],
            "authorization_seal": seal,
            "replacement_registry_path": source_bindings["replacement_registry"]["path"],
            "replacement_registry_scope": "append_only_all_roles_and_all_dgps",
            "orphan_policy": (
                "adopt raw-only archive only after exact closed intent, current "
                "authorization, prospective ledger, canonical nonlink paths, and full "
                "two-array recomputation; otherwise stop"
            ),
        }
        _json(raw_manifest_path, raw_manifest)
        part_path = execution_dir / f"{episode_id}.npz"
        part_arrays = {
            name: np.zeros(shape, dtype=np.dtype(dtype))
            for name, (shape, dtype) in workflow.DEVELOPMENT_PART_CONTRACT.items()
        }
        part_arrays["model_step"][:] = np.arange(3, 41, dtype=np.int16)
        np.savez_compressed(part_path, **part_arrays)
        sidecar_path = execution_dir / f"{episode_id}.json"
        array_contract = {
            name: {
                "shape": shape,
                "dtype": dtype,
                "sha256": workflow.array_sha256(part_arrays[name]),
            }
            for name, (shape, dtype) in workflow.DEVELOPMENT_PART_CONTRACT.items()
        }
        execution_sidecar = {
            "schema_version": 1,
            "attempt": "v003",
            "created_unix_ns": 2,
            "role": "fit",
            "regime": regime,
            "slot": 0,
            "episode_id": episode_id,
            "path": _relative(repo, part_path),
            "sha256": _sha(part_path),
            "raw_path": _relative(repo, raw_path),
            "raw_sha256": _sha(raw_path),
            "raw_sidecar_path": _relative(repo, raw_sidecar),
            "raw_sidecar_sha256": _sha(raw_sidecar),
            "input_loader_audit": raw_loader_audit,
            "evaluation": "four dense exits and all three frozen causal feature stages",
            "target_use": "permitted only for isolated fit",
            "contact_or_privileged_materialized": False,
            "primitive_feature_reconstruction_exact": True,
            "no_gradients": True,
            "complete": True,
            "path": _relative(repo, part_path),
            "sha256": _sha(part_path),
            "bytes": part_path.stat().st_size,
            "arrays": array_contract,
        }
        _json(sidecar_path, execution_sidecar)
        execution_record = {
            "slot": 0,
            "episode_id": episode_id,
            "path": _relative(repo, part_path),
            "sha256": _sha(part_path),
            "sidecar_path": _relative(repo, sidecar_path),
            "sidecar_sha256": _sha(sidecar_path),
            "source_raw_path": _relative(repo, raw_path),
            "source_raw_sha256": _sha(raw_path),
            "source_raw_sidecar_path": _relative(repo, raw_sidecar),
            "source_raw_sidecar_sha256": _sha(raw_sidecar),
            "call_histogram": None,
        }
        aggregate_path = root / "role.npz"
        with zipfile.ZipFile(aggregate_path, "w") as archive:
            for key in workflow.AGGREGATE_KEYS:
                archive.writestr(f"{key}.npy", b"synthetic metadata only")
        execution_manifest = {
            "schema_version": 1,
            "attempt": "v003",
            "created_unix_ns": 2,
            "role": "fit",
            "regime": regime,
            "complete": True,
            "episode_count": 1,
            "row_count": workflow.ROWS_PER_EPISODE,
            "episodes": [execution_record],
            "aggregate_role_path": _relative(repo, aggregate_path),
            "aggregate_role_sha256": _sha(aggregate_path),
            "aggregate_recovery": {
                "status": "new_complete_aggregate_written_and_verified",
                "exact_keys_verified": True,
                "exact_shapes_dtypes_content_verified": True,
                "array_sha256": {
                    key: "1" * 64 for key in workflow.AGGREGATE_KEYS
                },
                "file_sha256": _sha(aggregate_path),
            },
            "aggregate_role_keys": list(workflow.AGGREGATE_KEYS),
            "raw_manifest_path": _relative(repo, raw_manifest_path),
            "raw_manifest_sha256": _sha(raw_manifest_path),
            "source_raw_manifest_sha256": _sha(raw_manifest_path),
            "authorization": {
                "state": "FIT_COHORTS",
                "active_attempt": "v003",
                "state_sha256": source_state_sha256,
                "seal_path": seal["path"],
                "seal_sha256": seal["sha256"],
                "seal_checkpoint_state": seal["checkpoint_state"],
                "science_attempt": "v001",
                "raw_authorization_seal_path": seal["path"],
                "raw_authorization_seal_sha256": seal["sha256"],
                "exact_locked_snapshot_unchanged": True,
            },
            "source_bindings": source_bindings,
            "loaded_input_keys": ["action", "pixels"],
            "contact_or_privileged_materialized": False,
            "target_role_isolation": "fit",
            "causal_primitive_contract": {},
            "module_before": {"passed": True},
            "module_after": {"passed": True},
            "base_provenance": {},
            "frozen_facade": {},
            "no_gradients": True,
        }
        _json(root / "execution_manifest.json", execution_manifest)
    return attempt, state


def _install_fit_count_crash(
    monkeypatch: pytest.MonkeyPatch, attempt: Path
) -> tuple[dict[str, Any], Path]:
    repo = attempt.parents[3]
    study = attempt.parents[1]
    state_path = study / "STATE.json"
    ledger_path = study / "RESEARCH_LEDGER.jsonl"
    monkeypatch.setattr(workflow, "LEDGER_PATH", ledger_path)
    previous = _ledger_record(
        {
            "event": "state_completed",
            "attempt": "v008",
            "completed_state": "PRE_OUTCOME_SEAL",
            "created_unix_ns": 100,
            "seq": 1,
            "prev_sha256": "legacy:" + "0" * 64,
        }
    )
    pre_count = {
        "active_attempt": "v008",
        "active_attempt_path": _relative(repo, attempt),
        "current_state": "FIT_COHORTS",
        "completed_states": ["PRE_OUTCOME_SEAL"],
        "verified_checkpoints": [],
        "expected_fit_episode_count": 4,
        "expected_selection_episode_count": 2000,
        "fit_outcome_episodes": 0,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
        "early_scientific_failure": None,
        "postconfirmation_integrity_failure": None,
        "terminal_label": None,
        "updated_unix_ns": 100,
        "ledger_event_count": 1,
        "ledger_head_sha256": previous["record_sha256"],
    }
    prior_adapter_marker = _adapter_marker(
        repo=repo,
        adapter=workflow.CONTROLLER_ADAPTER_PATH,
        root_program=workflow.ROOT_PROGRAM_PATH,
        event_count=1,
        head_sha256=previous["record_sha256"],
        operation_sha256="a" * 64,
        state_without_marker=pre_count,
    )
    pre_count["v008_durable_controller_adapter"] = prior_adapter_marker
    _controller_json(state_path, pre_count)
    count_event = _ledger_record(
        {
            "event": "outcome_counts_updated",
            "attempt": "v008",
            "fields": {"fit_outcome_episodes": 4},
            "prior_controller_adapter_marker": prior_adapter_marker,
            "created_unix_ns": 200,
            "seq": 2,
            "prev_sha256": previous["record_sha256"],
        }
    )
    previous_payload = (
        json.dumps(previous, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    count_event_payload = (
        json.dumps(count_event, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    ledger_path.write_bytes(previous_payload + count_event_payload)
    current_without_marker = dict(pre_count)
    current_without_marker.pop("v008_durable_controller_adapter", None)
    current_without_marker.update(
        {
            "fit_outcome_episodes": 4,
            "updated_unix_ns": 200,
            "ledger_event_count": 2,
            "ledger_head_sha256": count_event["record_sha256"],
        }
    )
    current_operation_sha256 = _adapter_operation_sha256(
        base_state=pre_count,
        base_ledger_payload=previous_payload,
        expected_suffix=count_event_payload,
        intended_state_without_marker=current_without_marker,
    )
    current = dict(current_without_marker)
    current["v008_durable_controller_adapter"] = _adapter_marker(
        repo=repo,
        adapter=workflow.CONTROLLER_ADAPTER_PATH,
        root_program=workflow.ROOT_PROGRAM_PATH,
        event_count=2,
        head_sha256=count_event["record_sha256"],
        operation_sha256=current_operation_sha256,
        state_without_marker=current_without_marker,
    )
    _controller_json(state_path, current)
    return current, ledger_path


def test_fit_count_commit_crash_reauthenticates_manifests_and_advances_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, _ledger = _install_fit_count_crash(monkeypatch, attempt)
    verification = workflow.verify_role_manifests("fit", current)
    recovery = verification["role_count_update_recovery"]
    assert recovery["passed"] is True
    assert recovery["counter_field"] == "fit_outcome_episodes"
    assert recovery["count_update_event_fields"] == {"fit_outcome_episodes": 4}
    assert recovery["count_operation_sha256"] == current[
        "v008_durable_controller_adapter"
    ]["operation_sha256"]
    assert recovery["current_state_binding_sha256"] == current[
        "v008_durable_controller_adapter"
    ]["state_binding_sha256"]
    assert recovery["prior_state_binding_sha256"] == json.loads(
        _ledger.read_text(encoding="utf-8").splitlines()[-1]
    )["prior_controller_adapter_marker"]["state_binding_sha256"]

    controller = FakeController(
        "FIT_COHORTS",
        expected_fit_episode_count=4,
        fit_outcome_episodes=4,
    )
    audit_path = tmp_path / "recovered_fit_cohorts.json"
    driver = workflow.Workflow(
        controller,
        object(),
        evidence_paths={"FIT_COHORTS": audit_path},
        manifest_verifier=lambda _role, _state: verification,
        inputs_complete=lambda _role: True,
    )
    result = driver.step()
    assert result["advanced"] is True
    assert controller.count_calls == []
    assert [item[0] for item in controller.advance_calls] == ["FIT_COHORTS"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["evidence"]["role_verification"][
        "role_count_update_recovery"
    ]["exactly_one_controller_event_after_authorized_state"] is True


def test_fit_count_crash_after_audit_publish_adopts_exact_audit_and_advances_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, _ledger = _install_fit_count_crash(monkeypatch, attempt)
    first_recovery = workflow.verify_role_manifests("fit", current)
    prior_verification = copy.deepcopy(first_recovery)
    prior_verification["role_count_update_recovery"] = None
    prior_verification["scientific_replay_qualification"] = (
        _synthetic_replay_qualification("fit")
    )
    audit_path = attempt / "audit/fit_cohorts.json"
    audit = workflow._checkpoint_object(
        "FIT_COHORTS",
        current,
        checks={
            "all_four_regimes_verified": True,
            "exact_episode_count": True,
            "exact_row_count": True,
            "controller_count_exact": True,
            "aggregate_arrays_unopened": True,
            "scientific_replay_qualified": True,
            "confirmation_unopened": True,
        },
        evidence={"role_verification": prior_verification},
    )
    _controller_json(audit_path, audit)
    audit_sha256 = _sha(audit_path)

    recovered = workflow.verify_role_manifests("fit", current)
    repeated = workflow.verify_role_manifests("fit", current)
    for verification in (recovered, repeated):
        recovery = verification["role_count_update_recovery"]
        assert recovery["role_audit_already_published"] is True
        assert recovery["existing_role_audit_sha256"] == audit_sha256
        assert recovery["no_role_checkpoint_before_recovery"] is True

    controller = FakeController("FIT_COHORTS")
    controller.value = copy.deepcopy(current)
    driver = workflow.Workflow(
        controller,
        object(),
        evidence_paths={"FIT_COHORTS": audit_path},
        manifest_verifier=lambda _role, _state: recovered,
        inputs_complete=lambda _role: True,
    )
    result = driver.step()
    assert result["advanced"] is True
    assert controller.count_calls == []
    assert [item[0] for item in controller.advance_calls] == ["FIT_COHORTS"]
    assert _sha(audit_path) == audit_sha256


def test_second_crash_after_recovery_audit_publish_adopts_exact_audit_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resume the distinct crash after a recovery-bearing audit was durable."""

    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, _ledger = _install_fit_count_crash(monkeypatch, attempt)
    first_recovery = workflow.verify_role_manifests("fit", current)
    first_recovery["scientific_replay_qualification"] = (
        _synthetic_replay_qualification("fit")
    )
    initial_proof = first_recovery["role_count_update_recovery"]
    assert initial_proof["role_audit_already_published"] is False
    assert initial_proof["no_role_audit_or_checkpoint_before_recovery"] is True

    audit_path = attempt / "audit/fit_cohorts.json"
    audit = workflow._checkpoint_object(
        "FIT_COHORTS",
        current,
        checks={
            "all_four_regimes_verified": True,
            "exact_episode_count": True,
            "exact_row_count": True,
            "controller_count_exact": True,
            "aggregate_arrays_unopened": True,
            "scientific_replay_qualified": True,
            "confirmation_unopened": True,
        },
        evidence={"role_verification": first_recovery},
    )
    _controller_json(audit_path, audit)
    audit_sha256 = _sha(audit_path)

    resumed = workflow.verify_role_manifests("fit", current)
    repeated = workflow.verify_role_manifests("fit", current)
    assert resumed == repeated
    resumed_proof = resumed["role_count_update_recovery"]
    assert resumed_proof["role_audit_already_published"] is True
    assert resumed_proof["existing_role_audit_sha256"] == audit_sha256
    assert (
        resumed_proof["existing_role_audit_verification_kind"]
        == "count_recovery_verification"
    )

    controller = FakeController("FIT_COHORTS")
    controller.value = copy.deepcopy(current)
    driver = workflow.Workflow(
        controller,
        object(),
        evidence_paths={"FIT_COHORTS": audit_path},
        manifest_verifier=lambda _role, _state: resumed,
        inputs_complete=lambda _role: True,
    )
    result = driver.step()
    assert result["advanced"] is True
    assert controller.count_calls == []
    assert [item[0] for item in controller.advance_calls] == ["FIT_COHORTS"]
    assert _sha(audit_path) == audit_sha256
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    assert persisted["evidence"]["role_verification"] == first_recovery


def test_count_recovery_rejects_state_only_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, _ledger = _install_fit_count_crash(monkeypatch, attempt)
    current["expected_selection_episode_count"] += 1
    _controller_json(workflow.STATE_PATH, current)

    with pytest.raises(workflow.WorkflowError, match="adapter marker drift"):
        workflow.verify_role_manifests("fit", current)


def test_count_recovery_rejects_valid_hex_operation_replacement_even_if_rebound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, _ledger = _install_fit_count_crash(monkeypatch, attempt)
    marker = dict(current["v008_durable_controller_adapter"])
    marker["operation_sha256"] = "d" * 64
    marker_without_state_binding = dict(marker)
    marker_without_state_binding.pop("state_binding_sha256")
    state_without_marker = dict(current)
    state_without_marker.pop("v008_durable_controller_adapter")
    marker["state_binding_sha256"] = workflow.canonical_sha256(
        {
            "marker_without_state_binding": marker_without_state_binding,
            "state_without_marker": state_without_marker,
        }
    )
    current["v008_durable_controller_adapter"] = marker
    _controller_json(workflow.STATE_PATH, current)

    with pytest.raises(workflow.WorkflowError, match="operation/context drift"):
        workflow.verify_role_manifests("fit", current)


@pytest.mark.parametrize("tamper", ("recovery_proof", "pre_count_timestamp"))
def test_recovery_bearing_published_audit_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, _ledger = _install_fit_count_crash(monkeypatch, attempt)
    first_recovery = workflow.verify_role_manifests("fit", current)
    audit = workflow._checkpoint_object(
        "FIT_COHORTS",
        current,
        checks={
            "all_four_regimes_verified": True,
            "exact_episode_count": True,
            "exact_row_count": True,
            "controller_count_exact": True,
            "aggregate_arrays_unopened": True,
            "scientific_replay_qualified": True,
            "confirmation_unopened": True,
        },
        evidence={"role_verification": first_recovery},
    )
    if tamper == "recovery_proof":
        audit["evidence"]["role_verification"]["role_count_update_recovery"][
            "count_update_event_sha256"
        ] = "f" * 64
    else:
        audit["created_unix_ns"] = 199
    _controller_json(attempt / "audit/fit_cohorts.json", audit)

    with pytest.raises(workflow.WorkflowError, match="existing role audit drift"):
        workflow.verify_role_manifests("fit", current)


@pytest.mark.parametrize(
    "tamper", ("controller", "prior_recovery", "checks", "pre_count_timestamp")
)
def test_count_crash_after_audit_publish_rejects_stale_or_ambiguous_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, _ledger = _install_fit_count_crash(monkeypatch, attempt)
    verification = workflow.verify_role_manifests("fit", current)
    prior_verification = copy.deepcopy(verification)
    prior_verification["role_count_update_recovery"] = None
    audit = workflow._checkpoint_object(
        "FIT_COHORTS",
        current,
        checks={
            "all_four_regimes_verified": True,
            "exact_episode_count": True,
            "exact_row_count": True,
            "controller_count_exact": True,
            "aggregate_arrays_unopened": True,
            "scientific_replay_qualified": True,
            "confirmation_unopened": True,
        },
        evidence={"role_verification": prior_verification},
    )
    if tamper == "controller":
        audit["controller_status_sha256"] = "f" * 64
    elif tamper == "prior_recovery":
        audit["evidence"]["role_verification"][
            "role_count_update_recovery"
        ] = {"passed": True}
    elif tamper == "checks":
        audit["checks"]["exact_row_count"] = False
    else:
        audit["created_unix_ns"] = 199
    _controller_json(attempt / "audit/fit_cohorts.json", audit)
    with pytest.raises(workflow.WorkflowError, match="existing role audit drift"):
        workflow.verify_role_manifests("fit", current)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("later_counter", "changed other/later counters"),
        ("existing_audit", "existing role audit"),
        ("event_fields", "ledger event drift"),
        ("split_authorization", "execution authorization drift"),
    ),
)
def test_count_crash_recovery_rejects_state_ledger_audit_and_manifest_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, ledger_path = _install_fit_count_crash(monkeypatch, attempt)
    if tamper == "later_counter":
        current["selection_outcome_episodes"] = 1
        _controller_json(workflow.STATE_PATH, current)
    elif tamper == "existing_audit":
        _json(attempt / "audit/fit_cohorts.json", {"partial": True})
    elif tamper == "event_fields":
        records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        payload = dict(records[-1])
        payload.pop("record_sha256")
        payload["fields"] = {
            "fit_outcome_episodes": 4,
            "selection_outcome_episodes": 0,
        }
        records[-1] = _ledger_record(payload)
        current["ledger_head_sha256"] = records[-1]["record_sha256"]
        _controller_json(workflow.STATE_PATH, current)
        ledger_path.write_text(
            "".join(
                json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                for item in records
            ),
            encoding="utf-8",
        )
    else:
        path = (
            workflow.fit_inheritance.SOURCE_FIT_ROOT
            / "native_plan/execution_manifest.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["authorization"]["state_sha256"] = "f" * 64
        _json(path, manifest)
    with pytest.raises(workflow.WorkflowError, match=message):
        workflow.verify_role_manifests("fit", current)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fit_outcome_episodes", False),
        ("confirmation_outcomes_opened_for_analysis", 0),
    ),
)
def test_count_crash_recovery_rejects_bool_int_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    attempt, _ = _synthetic_role_manifests(monkeypatch, tmp_path)
    current, _ledger_path = _install_fit_count_crash(monkeypatch, attempt)
    current[field] = value
    _controller_json(workflow.STATE_PATH, current)

    with pytest.raises(workflow.WorkflowError):
        workflow.verify_role_manifests("fit", current)


@pytest.mark.parametrize(
    ("mode", "state_name", "contract_name"),
    (
        (
            "no_candidate",
            "CANDIDATE_SELECTION",
            "verifier_contract_no_candidate.json",
        ),
        (
            "power_infeasible",
            "CONFIRMATION_POWER_AND_COHORT_FREEZE",
            "verifier_contract_power_infeasible.json",
        ),
    ),
)
def test_loaded_workflow_controller_adapts_both_inherited_early_stop_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    state_name: str,
    contract_name: str,
) -> None:
    repo = tmp_path / "repo"
    study = repo / "runs/lewm_domain_robust_gate"
    attempt = study / "attempts/v008"
    source = study / "attempts/v001"
    attempt.mkdir(parents=True)
    source.mkdir(parents=True)
    contract = attempt / contract_name
    _json(contract, {"schema_version": 1, "attempt": "v008", "mode": mode})
    source_seal = source / "audit/pre_data_seal.json"
    _json(source_seal, {"attempt": "v001", "passed": True})
    inheritance = attempt / "audit/pre_data_inheritance_seal.json"
    _json(
        inheritance,
        {
            "schema_version": 1,
            "passed": True,
            "attempt": "v008",
            "source_attempt": "v001",
            "science_attempt": "v001",
            "target_attempt": "v008",
            "checkpoint_state": "PRE_OUTCOME_SEAL",
            "authorization_kind": "zero_outcome_version_forward_inherited_pre_data",
            "source_pre_data_seal": {
                "path": _relative(repo, source_seal),
                "sha256": _sha(source_seal),
            },
            "sealed_files": {_relative(repo, contract): _sha(contract)},
        },
    )
    receipt_path = attempt / "audit/version_forward_transaction_receipt.json"
    _json(receipt_path, {"passed": True})
    receipt = {
        "path": _relative(repo, receipt_path),
        "sha256": _sha(receipt_path),
        "passed": True,
    }
    state = {
        "active_attempt": "v008",
        "active_attempt_path": _relative(repo, attempt),
        "current_state": state_name,
        "completed_states": ["PRE_OUTCOME_SEAL"],
        "verified_checkpoints": [
            {
                "source_attempt": "v001",
                "evidence_path": _relative(repo, source_seal),
                "evidence_sha256": _sha(source_seal),
                "verification_lineage": "direct_checkpoint",
            }
        ],
    }
    calls: list[tuple[str, str]] = []

    def inherited_authorization(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["state"] == state
        assert (
            kwargs["authorized_early_guard_token"]
            is workflow._EARLY_ROOT_GUARD_TOKEN
        )
        assert (
            Path(kwargs["authorized_early_verifier_contract_path"]).resolve()
            == contract.resolve()
        )
        calls.append((kwargs["expected_current_state"], mode))
        return {
            "passed": True,
            "active_attempt": "v008",
            "science_attempt": "v001",
            "state": state_name,
            "seal_path": _relative(repo, inheritance),
            "seal_sha256": _sha(inheritance),
            "seal_checkpoint_state": "PRE_OUTCOME_SEAL",
            "source_pre_data_seal_path": _relative(repo, source_seal),
            "source_pre_data_seal_sha256": _sha(source_seal),
            "version_forward_transaction_receipt": receipt,
        }

    monkeypatch.setattr(workflow, "REPO_ROOT", repo)
    monkeypatch.setattr(workflow, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(workflow, "INHERITANCE_SEAL_PATH", inheritance)
    monkeypatch.setattr(
        workflow,
        "verify_inherited_pre_data_authorization",
        inherited_authorization,
    )
    program = study / "synthetic_program.py"
    program.write_text(
        "from pathlib import Path\n"
        f"STATE = {state!r}\n"
        f"CONTRACTS = {{{mode!r}: {str(contract)!r}}}\n"
        "def _require_presealed_verifier_contract(state, contract):\n"
        "    raise RuntimeError('unadapted root guard')\n"
        "def stage_early_scientific_failure(mode):\n"
        "    _require_presealed_verifier_contract(STATE, Path(CONTRACTS[mode]))\n"
        "    return {'current_state': 'INDEPENDENT_VERIFICATION', 'mode': mode}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "CONTROLLER_ADAPTER_PATH", program)
    controller = workflow._load_controller()
    result = controller.stage_early_scientific_failure(mode)
    assert result == {"current_state": "INDEPENDENT_VERIFICATION", "mode": mode}
    assert calls == [(state_name, mode)]


def test_inherited_early_stop_adapter_rejects_unsealed_active_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    study = repo / "runs/lewm_domain_robust_gate"
    attempt = study / "attempts/v008"
    source = study / "attempts/v001"
    attempt.mkdir(parents=True)
    source.mkdir(parents=True)
    contract = attempt / "verifier_contract_no_candidate.json"
    _json(contract, {"attempt": "v008"})
    source_seal = source / "audit/pre_data_seal.json"
    _json(source_seal, {"attempt": "v001"})
    inheritance = attempt / "audit/pre_data_inheritance_seal.json"
    _json(
        inheritance,
        {
            "schema_version": 1,
            "passed": True,
            "attempt": "v008",
            "source_attempt": "v001",
            "science_attempt": "v001",
            "target_attempt": "v008",
            "checkpoint_state": "PRE_OUTCOME_SEAL",
            "authorization_kind": "zero_outcome_version_forward_inherited_pre_data",
            "source_pre_data_seal": {
                "path": _relative(repo, source_seal),
                "sha256": _sha(source_seal),
            },
            "sealed_files": {_relative(repo, contract): "0" * 64},
        },
    )
    receipt_path = attempt / "audit/version_forward_transaction_receipt.json"
    _json(receipt_path, {"passed": True})
    receipt = {
        "path": _relative(repo, receipt_path),
        "sha256": _sha(receipt_path),
        "passed": True,
    }
    state = {
        "active_attempt": "v008",
        "active_attempt_path": _relative(repo, attempt),
        "current_state": "CANDIDATE_SELECTION",
        "completed_states": ["PRE_OUTCOME_SEAL"],
        "verified_checkpoints": [
            {
                "source_attempt": "v001",
                "evidence_path": _relative(repo, source_seal),
                "evidence_sha256": _sha(source_seal),
                "verification_lineage": "direct_checkpoint",
            }
        ],
    }
    monkeypatch.setattr(workflow, "REPO_ROOT", repo)
    monkeypatch.setattr(workflow, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(workflow, "INHERITANCE_SEAL_PATH", inheritance)
    monkeypatch.setattr(
        workflow,
        "verify_inherited_pre_data_authorization",
        lambda **_kwargs: {
            "passed": True,
            "active_attempt": "v008",
            "science_attempt": "v001",
            "state": "CANDIDATE_SELECTION",
            "seal_path": _relative(repo, inheritance),
            "seal_sha256": _sha(inheritance),
            "seal_checkpoint_state": "PRE_OUTCOME_SEAL",
            "source_pre_data_seal_path": _relative(repo, source_seal),
            "source_pre_data_seal_sha256": _sha(source_seal),
            "version_forward_transaction_receipt": receipt,
        },
    )
    with pytest.raises(workflow.WorkflowError, match="inheritance seal drift"):
        workflow._require_inherited_presealed_verifier_contract(state, contract)


def test_real_manifest_verifier_authenticates_parts_and_rejects_closed_contract_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempt, state = _synthetic_role_manifests(monkeypatch, tmp_path)
    result = workflow.verify_role_manifests("fit", state)
    assert result["episode_count"] == 4
    assert result["row_count"] == 4 * workflow.ROWS_PER_EPISODE
    assert set(result["regimes"]) == set(workflow.REGIMES)
    assert result["npz_arrays_opened_during_manifest_verification"] is True
    assert result["development_part_arrays_opened"] is True
    assert result["aggregate_arrays_opened"] is False
    manifest_path = (
        workflow.fit_inheritance.SOURCE_FIT_ROOT
        / "native_plan/execution_manifest.json"
    )
    baseline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding_attack = copy.deepcopy(baseline_manifest)
    binding_attack["source_bindings"]["runner_source"] = binding_attack[
        "source_bindings"
    ]["dgp_matrix"]
    _json(manifest_path, binding_attack)
    with pytest.raises(workflow.WorkflowError, match="source binding path/hash drift"):
        workflow.verify_role_manifests("fit", state)
    _json(manifest_path, baseline_manifest)

    part_path = (
        workflow.fit_inheritance.SOURCE_FIT_ROOT
        / "native_plan/execution/synthetic-native_plan.npz"
    )
    sidecar_path = part_path.with_suffix(".json")
    original_part = part_path.read_bytes()
    baseline_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    def publish_attack(
        attacked_arrays: dict[str, np.ndarray] | None,
        attacked_sidecar: dict[str, Any],
    ) -> None:
        if attacked_arrays is not None:
            np.savez_compressed(part_path, **attacked_arrays)
            attacked_sidecar["sha256"] = _sha(part_path)
            attacked_sidecar["bytes"] = part_path.stat().st_size
            attacked_sidecar["arrays"] = {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": workflow.array_sha256(value),
                }
                for name, value in attacked_arrays.items()
            }
        _json(sidecar_path, attacked_sidecar)
        attacked_manifest = copy.deepcopy(baseline_manifest)
        attacked_manifest["episodes"][0]["sha256"] = _sha(part_path)
        attacked_manifest["episodes"][0]["sidecar_sha256"] = _sha(sidecar_path)
        _json(manifest_path, attacked_manifest)

    extra_sidecar = copy.deepcopy(baseline_sidecar)
    extra_sidecar["unexpected"] = True
    publish_attack(None, extra_sidecar)
    with pytest.raises(workflow.WorkflowError, match="sidecar schema drift"):
        workflow.verify_role_manifests("fit", state)
    sidecar_path.write_text(
        json.dumps(baseline_sidecar, sort_keys=True) + "\n", encoding="utf-8"
    )
    _json(manifest_path, baseline_manifest)

    with np.load(part_path, allow_pickle=False) as stored:
        baseline_arrays = {name: stored[name].copy() for name in stored.files}
    for attack_name in ("member_add", "member_drop", "dtype", "feature"):
        attacked_arrays = {
            name: value.copy() for name, value in baseline_arrays.items()
        }
        if attack_name == "member_add":
            attacked_arrays["calls"] = np.ones(38, dtype=np.int8)
        elif attack_name == "member_drop":
            attacked_arrays.pop("stage_update")
        elif attack_name == "dtype":
            attacked_arrays["target"] = attacked_arrays["target"].astype(np.float64)
        else:
            attacked_arrays["production_features"][0, 0, 0] = np.float32(1.0)
        publish_attack(attacked_arrays, copy.deepcopy(baseline_sidecar))
        with pytest.raises(workflow.WorkflowError):
            workflow.verify_role_manifests("fit", state)
        part_path.write_bytes(original_part)
        _json(sidecar_path, baseline_sidecar)
        _json(manifest_path, baseline_manifest)

    aggregate = workflow.fit_inheritance.SOURCE_FIT_ROOT / "native_plan/role.npz"
    with zipfile.ZipFile(aggregate, "w") as archive:
        archive.writestr("unexpected.npy", b"drift")
    with pytest.raises(workflow.WorkflowError, match="artifact hash drift"):
        workflow.verify_role_manifests("fit", state)


@pytest.mark.parametrize(
    ("artifact", "alias"),
    (
        ("raw_manifest", True),
        ("raw_manifest", 1.0),
        ("execution_manifest", True),
        ("execution_manifest", 1.0),
        ("execution_sidecar", True),
        ("execution_sidecar", 1.0),
    ),
)
def test_role_verifier_rejects_schema_bool_and_numeric_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact: str,
    alias: Any,
) -> None:
    attempt, state = _synthetic_role_manifests(monkeypatch, tmp_path)
    root = workflow.fit_inheritance.SOURCE_FIT_ROOT / "native_plan"
    raw_path = root / "raw_manifest.json"
    execution_path = root / "execution_manifest.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if artifact == "raw_manifest":
        value = json.loads(raw_path.read_text(encoding="utf-8"))
        value["schema_version"] = alias
        _json(raw_path, value)
        match = "raw manifest header failed"
    elif artifact == "execution_manifest":
        execution["schema_version"] = alias
        _json(execution_path, execution)
        match = "execution manifest header failed"
    else:
        sidecar_path = workflow.resolve_repo_relative(
            execution["episodes"][0]["sidecar_path"]
        )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["schema_version"] = alias
        _json(sidecar_path, sidecar)
        execution["episodes"][0]["sidecar_sha256"] = _sha(sidecar_path)
        _json(execution_path, execution)
        match = "sidecar contract"
    with pytest.raises(workflow.WorkflowError, match=match):
        workflow.verify_role_manifests("fit", state)


@pytest.mark.parametrize("alias", (True, 201.0))
def test_workflow_loader_shape_requires_exact_integer_elements(alias: Any) -> None:
    hashes = {"action": "a" * 64, "pixels": "b" * 64}
    raw_record = {
        "raw_sha256": "c" * 64,
        "arrays": {
            "action": {
                "shape": [201, 5],
                "dtype": "float32",
                "sha256": hashes["action"],
            },
            "pixels": {
                "shape": [201, 224, 224, 3],
                "dtype": "uint8",
                "sha256": hashes["pixels"],
            },
        },
    }
    audit = {
        "loaded_keys": ["action", "pixels"],
        "pixels_shape": [alias, 224, 224, 3],
        "pixels_dtype": "uint8",
        "action_shape": [201, 5],
        "action_dtype": "float32",
        "pixels_finite": True,
        "modeled_actions_finite": True,
        "terminal_action_nan_sentinel": True,
        "array_sha256": hashes,
        "archive_keys": ["action", "pixels"],
        "archive_sha256": "c" * 64,
        "arrays_materialized": ["action", "pixels"],
        "non_input_arrays_materialized": False,
    }
    with pytest.raises(workflow.WorkflowError, match="input-loader audit drift"):
        workflow._validate_input_loader_audit(audit, raw_record=raw_record)


def test_cli_surface_is_exact() -> None:
    assert workflow._parse_arguments(["status"]).command == "status"
    assert workflow._parse_arguments(["step"]).command == "step"
    parsed = workflow._parse_arguments(["resume", "--max-transitions", "7"])
    assert parsed.max_transitions == 7
    parsed = workflow._parse_arguments(["verify-role", "selection"])
    assert parsed.role == "selection"
    with pytest.raises(SystemExit):
        workflow._parse_arguments(["verify-role"])
