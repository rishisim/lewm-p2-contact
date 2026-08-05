from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    specification = importlib.util.spec_from_file_location(
        "terminal_workflow_synthetic", ROOT / "terminal_workflow.py"
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


tw = _module()


@pytest.fixture(autouse=True)
def _stub_terminal_scientific_replay(monkeypatch: pytest.MonkeyPatch):
    def qualify(
        *,
        state: dict[str, object],
        mode: str,
        attempt_root: Path,
        repo_root: Path,
        cell_runner: object | None = None,
    ) -> dict[str, object]:
        del state, repo_root, cell_runner
        value: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "synthetic_terminal_scientific_replay_qualification",
            "attempt": "v008",
            "mode": mode,
            "passed": True,
        }
        path = attempt_root / tw.SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = tw.canonical_json_bytes(value, pretty=True)
        if path.exists():
            assert path.read_bytes() == payload
        else:
            path.write_bytes(payload)
        return value

    monkeypatch.setattr(
        tw, "_ensure_terminal_scientific_replay_qualification", qualify
    )


def _attempt(tmp_path: Path) -> tuple[Path, Path]:
    attempt = tmp_path / "runs/lewm_domain_robust_gate/attempts/v008"
    attempt.mkdir(parents=True)
    return tmp_path, attempt


def _contract(repo: Path, attempt: Path, mode: str = "confirmation") -> Path:
    attempt_relative = attempt.relative_to(repo).as_posix()
    name = {
        "confirmation": "verifier_contract.json",
        "no_candidate": "verifier_contract_no_candidate.json",
        "power_infeasible": "verifier_contract_power_infeasible.json",
    }[mode]
    path = attempt / name
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt": "v008",
                "attempt_root": attempt_relative,
                "study_root": "runs/lewm_domain_robust_gate",
                "mode": mode,
                "paths": {
                    "terminal_manifest": f"{attempt_relative}/audit/terminal_input_manifest.json"
                },
                "terminal_manifest_exclusions": [
                    f"{attempt_relative}/audit/terminal_input_manifest.json",
                    f"{attempt_relative}/audit/independent_verification.json",
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _state(mode: str = "confirmation") -> dict[str, object]:
    state: dict[str, object] = {
        "active_attempt": "v008",
        "current_state": "INDEPENDENT_VERIFICATION",
        "smoke_outcome_episodes": 24 if mode == "confirmation" else 0,
        "confirmation_outcome_episodes_generated": 2000 if mode == "confirmation" else 0,
        "confirmation_outcome_episodes_executed": 2000 if mode == "confirmation" else 0,
        "confirmation_outcomes_opened_for_analysis": mode == "confirmation",
        "expected_confirmation_episode_count": 2000,
    }
    if mode != "confirmation":
        state["early_scientific_failure"] = {
            "mode": mode,
            "status": "awaiting_independent_verification",
            "trigger_evidence_path": f"runs/source/{mode}.json",
            "trigger_evidence_sha256": "1" * 64,
        }
    return state


def _terminal_replay_cell_case(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    repo, attempt = _attempt(tmp_path)
    role = "fit"
    regime = tw.DGP_ORDER[0]
    root = attempt / "data" / role / regime
    root.mkdir(parents=True)
    replay_source = attempt / tw.SCIENTIFIC_REPLAY_SOURCE_NAME
    replay_launcher = attempt / tw.SCIENTIFIC_REPLAY_LAUNCHER_NAME
    replay_source.write_text("REPLAY = 1\n", encoding="utf-8")
    replay_launcher.write_text("LAUNCHER = 1\n", encoding="utf-8")
    raw_path = root / "raw_000000.npz"
    raw_sidecar = root / "raw_000000.json"
    part_path = root / "part_000000.npz"
    part_sidecar = root / "part_000000.json"
    aggregate_path = root / "role.npz"
    raw_path.write_bytes(b"raw")
    raw_sidecar.write_bytes(b"raw-sidecar")
    part_path.write_bytes(b"part")
    part_sidecar.write_bytes(b"part-sidecar")
    aggregate_path.write_bytes(b"aggregate")
    relative = lambda path: path.relative_to(repo).as_posix()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    episode_id = "fit-native-000000"
    raw_manifest = {
        "attempt": "v008",
        "role": role,
        "regime": regime,
        "complete": True,
        "episode_count": 1,
        "episodes": [
            {
                "slot": 0,
                "episode_id": episode_id,
                "raw_path": relative(raw_path),
                "raw_sha256": sha(raw_path),
            }
        ],
    }
    execution_manifest = {
        "attempt": "v008",
        "role": role,
        "regime": regime,
        "complete": True,
        "episode_count": 1,
        "row_count": tw.REPLAY_ROWS_PER_EPISODE,
        "authorization": {"state_sha256": "a" * 64},
        "aggregate_role_path": relative(aggregate_path),
        "aggregate_role_sha256": sha(aggregate_path),
        "episodes": [
            {
                "slot": 0,
                "episode_id": episode_id,
                "path": relative(part_path),
                "sha256": sha(part_path),
                "sidecar_path": relative(part_sidecar),
                "sidecar_sha256": sha(part_sidecar),
                "source_raw_sidecar_sha256": sha(raw_sidecar),
            }
        ],
    }
    raw_manifest_path = root / "raw_manifest.json"
    execution_manifest_path = root / "execution_manifest.json"
    raw_manifest_path.write_bytes(
        tw.canonical_json_bytes(raw_manifest, pretty=True)
    )
    execution_manifest_path.write_bytes(
        tw.canonical_json_bytes(execution_manifest, pretty=True)
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "v008_independent_scientific_replay_regime",
        "attempt": "v008",
        "role": role,
        "regime": regime,
        "raw_manifest": {
            "path": relative(raw_manifest_path),
            "sha256": sha(raw_manifest_path),
        },
        "execution_manifest": {
            "path": relative(execution_manifest_path),
            "sha256": sha(execution_manifest_path),
        },
        "compiled_gate": None,
        "episode_count": 1,
        "row_count": tw.REPLAY_ROWS_PER_EPISODE,
        "episodes": [
            {
                "slot": 0,
                "episode_id": episode_id,
                "raw": {"path": relative(raw_path), "sha256": sha(raw_path)},
                "raw_sidecar": {
                    "path": relative(raw_sidecar),
                    "sha256": sha(raw_sidecar),
                },
                "execution_part": {
                    "path": relative(part_path),
                    "sha256": sha(part_path),
                },
                "execution_sidecar": {
                    "path": relative(part_sidecar),
                    "sha256": sha(part_sidecar),
                },
                "input_loader_audit_sha256": "b" * 64,
                "array_sha256": {
                    name: "c" * 64 for name in tw.REPLAY_DEVELOPMENT_ARRAY_KEYS
                },
                "every_persisted_tensor_exact": True,
            }
        ],
        "aggregate": {
            "applicable": True,
            "path": relative(aggregate_path),
            "sha256": sha(aggregate_path),
            "arrays": {
                name: "d" * 64
                for name in (
                    "episode_slot",
                    "model_step",
                    "target",
                    "exits",
                    "production_features",
                )
            },
            "exact": True,
        },
        "runtime_audit": {
            "schema_version": 1,
            "passed": True,
            "requested_role": "independent_verification",
            "runtime_role": "evaluation",
            "checks": {"runtime": True, "mps_available": True},
            "mps": {"required": True, "built": True, "available": True},
            "read_only_preflight": True,
            "seed_tuples_consumed": 0,
            "output_paths_created": 0,
        },
        "source_hashes": {
            relative(replay_source): sha(replay_source),
            relative(replay_launcher): sha(replay_launcher),
        },
        "module_before": {"passed": True},
        "module_after": {"passed": True},
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "aggregate_exact": True,
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_tree_unchanged": True,
        "read_only": True,
        "passed": True,
    }
    return repo, attempt, result


def test_terminal_replay_cell_accepts_exact_typed_positive_fixture(tmp_path: Path):
    repo, attempt, result = _terminal_replay_cell_case(tmp_path)

    evidence = tw._validate_terminal_replay_result(
        result,
        role="fit",
        regime=tw.DGP_ORDER[0],
        episodes_per_dgp=1,
        attempt_root=attempt,
        repo_root=repo,
    )

    assert evidence["episode_count"] == 1
    assert evidence["row_count"] == tw.REPLAY_ROWS_PER_EPISODE
    assert evidence["aggregate"]["exact"] is True


@pytest.mark.parametrize("tamper", ["bool_count", "mps_false", "exact_false"])
def test_terminal_replay_cell_rejects_type_runtime_or_exactness_drift(
    tmp_path: Path, tamper: str
):
    repo, attempt, result = _terminal_replay_cell_case(tmp_path)
    if tamper == "bool_count":
        result["episode_count"] = True
    elif tamper == "mps_false":
        result["runtime_audit"]["mps"]["available"] = False
    else:
        result["every_persisted_tensor_exact"] = False

    with pytest.raises(tw.TerminalWorkflowError, match="type|MPS|assertion"):
        tw._validate_terminal_replay_result(
            result,
            role="fit",
            regime=tw.DGP_ORDER[0],
            episodes_per_dgp=1,
            attempt_root=attempt,
            repo_root=repo,
        )


def test_terminal_manifest_is_exhaustive_idempotent_and_contract_exact(tmp_path: Path):
    repo, attempt = _attempt(tmp_path)
    contract = _contract(repo, attempt)
    (attempt / ".hidden").write_bytes(b"hidden")
    nested = attempt / "nested"
    nested.mkdir()
    (nested / "artifact.bin").write_bytes(b"sealed")
    result = tw.build_terminal_input_manifest(
        contract,
        state=_state(),
        attempt_root=attempt,
        repo_root=repo,
    )
    expected = {
        contract.relative_to(repo).as_posix(),
        (attempt / ".hidden").relative_to(repo).as_posix(),
        (nested / "artifact.bin").relative_to(repo).as_posix(),
        (
            attempt / tw.SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE
        ).relative_to(repo).as_posix(),
    }
    assert set(result["files"]) == expected
    assert result["file_count"] == 4
    assert result == tw.build_terminal_input_manifest(
        contract,
        state=_state(),
        attempt_root=attempt,
        repo_root=repo,
    )
    (nested / "late.bin").write_bytes(b"not in closure")
    with pytest.raises(tw.TerminalWorkflowError, match="manifest verification"):
        tw.verify_terminal_input_manifest(
            attempt / tw.MANIFEST_RELATIVE,
            contract,
            attempt_root=attempt,
            repo_root=repo,
        )


def test_terminal_manifest_reruns_replay_before_adopting_existing_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, attempt = _attempt(tmp_path)
    contract = _contract(repo, attempt)
    calls: list[str] = []

    def qualify(**arguments: object) -> dict[str, object]:
        calls.append(str(arguments["mode"]))
        path = attempt / tw.SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE
        value: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "synthetic_terminal_scientific_replay_qualification",
            "attempt": "v008",
            "mode": arguments["mode"],
            "passed": True,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = tw.canonical_json_bytes(value, pretty=True)
        if path.exists():
            assert path.read_bytes() == payload
        else:
            path.write_bytes(payload)
        return value

    monkeypatch.setattr(
        tw, "_ensure_terminal_scientific_replay_qualification", qualify
    )
    tw.build_terminal_input_manifest(
        contract, state=_state(), attempt_root=attempt, repo_root=repo
    )
    tw.build_terminal_input_manifest(
        contract, state=_state(), attempt_root=attempt, repo_root=repo
    )

    assert calls == ["confirmation", "confirmation"]


def test_replay_failure_cannot_create_terminal_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, attempt = _attempt(tmp_path)
    contract = _contract(repo, attempt)

    def fail(**_arguments: object) -> dict[str, object]:
        raise tw.TerminalWorkflowError("synthetic replay failure")

    monkeypatch.setattr(
        tw, "_ensure_terminal_scientific_replay_qualification", fail
    )
    with pytest.raises(tw.TerminalWorkflowError, match="synthetic replay failure"):
        tw.build_terminal_input_manifest(
            contract, state=_state(), attempt_root=attempt, repo_root=repo
        )

    assert not (attempt / tw.MANIFEST_RELATIVE).exists()


def test_terminal_manifest_rejects_hardlinks_and_symlink_directories(tmp_path: Path):
    repo, attempt = _attempt(tmp_path)
    contract = _contract(repo, attempt)
    source = attempt / "source.bin"
    source.write_bytes(b"same inode")
    os.link(source, attempt / "alias.bin")
    with pytest.raises(tw.TerminalWorkflowError, match="hard-link alias"):
        tw.build_terminal_input_manifest(
            contract,
            state=_state(),
            attempt_root=attempt,
            repo_root=repo,
        )

    repo_two, attempt_two = _attempt(tmp_path / "second")
    contract_two = _contract(repo_two, attempt_two)
    target = attempt_two / "target"
    target.mkdir()
    (target / "x").write_bytes(b"x")
    (attempt_two / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(tw.TerminalWorkflowError, match="linked or non-directory"):
        tw.build_terminal_input_manifest(
            contract_two,
            state=_state(),
            attempt_root=attempt_two,
            repo_root=repo_two,
        )


def _install_globals(monkeypatch: pytest.MonkeyPatch, repo: Path, attempt: Path) -> None:
    monkeypatch.setattr(tw, "REPO_ROOT", repo)
    monkeypatch.setattr(tw, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(tw, "STUDY_ROOT", attempt.parents[1])
    monkeypatch.setattr(tw, "CONTROLLER", attempt.parents[1] / "program.py")


def _captured_audit(
    repo: Path,
    attempt: Path,
    contract: Path,
    *,
    passed: bool,
    label: str | None,
    analysis_hash: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "INDEPENDENT_VERIFICATION",
        "mode": "confirmation",
        "passed": passed,
        "terminal_label": label,
        "read_only_verifier": True,
        "local_production_modules_imported": False,
        "stdout_json_only": True,
        "checks": {"synthetic_verification_complete": True},
        "verifier_contract": {
            "path": contract.relative_to(repo).as_posix(),
            "sha256": tw.sha256_file(contract),
        },
        "source_hashes": {},
        "capture": {
            "captured_exclusively": True,
            "wrapper_integrity_passed": True,
            "scientific_verifier_passed": passed,
            "verifier_returncode": 0 if passed else 1,
        },
    }
    if analysis_hash is not None:
        result["source_hashes"] = {"analysis_result": analysis_hash}
        analysis = json.loads((attempt / "analysis_result.json").read_text())
        result["evidence"] = {
            "confirmation": {
                "terminal_label": label,
                "supported_co_primary_claim_count": analysis[
                    "supported_co_primary_claim_count"
                ],
            }
        }
    if not passed:
        result["error_type"] = "VerificationError"
        result["error"] = "sealed input mismatch"
    audit = attempt / tw.AUDIT_RELATIVE
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _analysis(attempt: Path, label: str, supported: int) -> dict[str, object]:
    endpoint_supported = []
    for index in range(8):
        endpoint_supported.append(index < supported)
    cursor = 0
    simultaneous: dict[str, object] = {}
    for regime in tw.DGP_ORDER:
        simultaneous[regime] = {}
        for endpoint in tw.ENDPOINT_ORDER:
            is_supported = endpoint_supported[cursor]
            cursor += 1
            simultaneous[regime][endpoint] = {
                "contrast": f"{endpoint}_vs_analytic",
                "estimate": 0.2 if is_supported else -0.1,
                "lower": 0.01 if is_supported else -0.02,
                "supported": is_supported,
            }
    value: dict[str, object] = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "SEALED_ANALYSIS",
        "passed": True,
        "process_valid": True,
        "integrity_passed": True,
        "proposed_terminal_label": label,
        "supported_co_primary_claim_count": supported,
        "simultaneous_co_primary": simultaneous,
    }
    (attempt / "analysis_result.json").write_text(
        json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


def test_normal_partial_decision_and_controller_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, attempt = _attempt(tmp_path)
    _install_globals(monkeypatch, repo, attempt)
    contract = _contract(repo, attempt)
    _analysis(attempt, "domain_robust_gate_partial", 6)
    audit = _captured_audit(
        repo,
        attempt,
        contract,
        passed=True,
        label="domain_robust_gate_partial",
        analysis_hash=tw.sha256_file(attempt / "analysis_result.json"),
    )
    state = _state()
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(tw, "controller_status", lambda: dict(state))

    def controller(arguments):
        calls.append(tuple(arguments))
        if arguments[:2] == ("advance", "INDEPENDENT_VERIFICATION"):
            return dict(state) | {"current_state": "TERMINAL"}
        if arguments[0] == "terminal":
            return dict(state) | {
                "current_state": "TERMINAL",
                "terminal_label": "domain_robust_gate_partial",
            }
        if arguments[:2] == ("advance", "TERMINAL"):
            return dict(state) | {
                "current_state": "POST_TERMINAL_REPORTING",
                "terminal_label": "domain_robust_gate_partial",
                "process_valid": True,
            }
        raise AssertionError(arguments)

    monkeypatch.setattr(tw, "_controller", controller)
    result = tw.finalize_terminal()
    decision = json.loads((attempt / "decision.json").read_text())
    assert result["current_state"] == "POST_TERMINAL_REPORTING"
    assert decision["supported_co_primary_claim_count"] == 6
    robustness = tw._robustness_map(state, decision, audit)
    assert robustness["supported_co_primary_claim_count"] == 6
    assert sum(
        int(robustness["regimes"][regime][endpoint]["supported"])
        for regime in tw.DGP_ORDER
        for endpoint in tw.ENDPOINT_ORDER
    ) == 6
    assert [call[0] for call in calls] == ["advance", "terminal", "advance"]
    assert audit["terminal_label"] == decision["terminal_label"]


def test_confirmation_verifier_failure_maps_only_to_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, attempt = _attempt(tmp_path)
    _install_globals(monkeypatch, repo, attempt)
    contract = _contract(repo, attempt)
    _captured_audit(
        repo, attempt, contract, passed=False, label=None, analysis_hash=None
    )
    state = _state()
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(tw, "controller_status", lambda: dict(state))
    monkeypatch.setattr(
        tw,
        "_controller",
        lambda arguments: calls.append(tuple(arguments))
        or dict(state)
        | {
            "current_state": "POST_TERMINAL_REPORTING",
            "terminal_label": tw.INVALID_LABEL,
            "process_valid": False,
        },
    )
    result = tw.finalize_terminal()
    decision = json.loads((attempt / "decision.json").read_text())
    assert result["terminal_label"] == tw.INVALID_LABEL
    assert decision["process_valid"] is False
    assert decision["scientific_support_call_withheld"] is True
    assert calls == [("finalize-execution-invalid",)]


def test_authenticated_staged_analysis_failure_uses_same_invalid_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, attempt = _attempt(tmp_path)
    _install_globals(monkeypatch, repo, attempt)
    contract = _contract(repo, attempt)
    audit = _captured_audit(
        repo, attempt, contract, passed=True, label=tw.INVALID_LABEL
    )
    audit["source_hashes"] = {"analysis_execution_invalid": "a" * 64}
    (attempt / tw.AUDIT_RELATIVE).write_text(
        json.dumps(audit, sort_keys=True) + "\n"
    )
    state = _state() | {
        "postconfirmation_integrity_failure": {
            "status": "awaiting_independent_verification",
            "source": "analysis_execution",
            "source_sha256": "a" * 64,
        }
    }
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(tw, "controller_status", lambda: dict(state))
    monkeypatch.setattr(
        tw,
        "_controller",
        lambda arguments: calls.append(tuple(arguments))
        or dict(state)
        | {
            "current_state": "POST_TERMINAL_REPORTING",
            "terminal_label": tw.INVALID_LABEL,
            "process_valid": False,
        },
    )
    result = tw.finalize_terminal()
    assert result["terminal_label"] == tw.INVALID_LABEL
    assert calls == [("finalize-execution-invalid",)]
    assert json.loads((attempt / "decision.json").read_text())[
        "independent_integrity_failure_confirmed"
    ] is True


def test_result_present_analysis_integrity_failure_is_manifested_but_not_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, attempt = _attempt(tmp_path)
    _install_globals(monkeypatch, repo, attempt)
    contract = _contract(repo, attempt)
    result = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "SEALED_ANALYSIS",
        "passed": True,
        "integrity_passed": False,
        "process_valid": False,
        "proposed_terminal_label": tw.INVALID_LABEL,
    }
    (attempt / "analysis_result.json").write_text(
        json.dumps(result, sort_keys=True) + "\n"
    )
    invalid = attempt / "audit/analysis_execution_invalid.json"
    invalid.parent.mkdir(parents=True)
    invalid.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt": "v008",
                "checkpoint_state": "SEALED_ANALYSIS",
                "passed": False,
                "integrity_failure": True,
                "analysis_result_present": True,
                "analysis_result_sha256": tw.sha256_file(
                    attempt / "analysis_result.json"
                ),
            },
            sort_keys=True,
        )
        + "\n"
    )
    state = _state() | {
        "postconfirmation_integrity_failure": {
            "status": "awaiting_independent_verification",
            "source": "analysis_execution",
            "source_sha256": tw.sha256_file(invalid),
        }
    }
    manifest = tw.build_terminal_input_manifest(
        contract, state=state, attempt_root=attempt, repo_root=repo
    )
    assert (attempt / "analysis_result.json").relative_to(repo).as_posix() in manifest[
        "files"
    ]
    assert invalid.relative_to(repo).as_posix() in manifest["files"]

    audit = _captured_audit(
        repo, attempt, contract, passed=True, label=tw.INVALID_LABEL
    )
    audit["source_hashes"] = {
        "analysis_execution_invalid": tw.sha256_file(invalid)
    }
    (attempt / tw.AUDIT_RELATIVE).write_text(
        json.dumps(audit, sort_keys=True) + "\n"
    )
    decision = tw.build_terminal_decision(state, audit)
    robustness = tw._robustness_map(state, decision, audit)
    assert decision["terminal_label"] == tw.INVALID_LABEL
    assert robustness["supported_co_primary_claim_count"] is None
    assert robustness["scientific_support_call_withheld"] is True


def test_early_decision_has_zero_later_roles_and_reports_are_claim_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, attempt = _attempt(tmp_path)
    _install_globals(monkeypatch, repo, attempt)
    contract = _contract(repo, attempt, "no_candidate")
    audit: dict[str, object] = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "INDEPENDENT_VERIFICATION",
        "mode": "no_candidate",
        "passed": True,
        "terminal_label": "domain_robust_gate_failed",
        "read_only_verifier": True,
        "local_production_modules_imported": False,
        "stdout_json_only": True,
        "checks": {"synthetic_verification_complete": True},
        "verifier_contract": {
            "path": contract.relative_to(repo).as_posix(),
            "sha256": tw.sha256_file(contract),
        },
        "source_hashes": {"selection_ledger": "1" * 64},
        "capture": {
            "captured_exclusively": True,
            "wrapper_integrity_passed": True,
            "scientific_verifier_passed": True,
            "verifier_returncode": 0,
        },
    }
    audit_path = attempt / tw.AUDIT_RELATIVE
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n")
    state = _state("no_candidate")
    decision = tw.build_terminal_decision(state, audit)
    assert decision["confirmation_outcome_episodes_generated"] == 0
    assert decision["confirmation_outcomes_opened_for_analysis"] is False

    terminal_state = dict(state) | {
        "current_state": "POST_TERMINAL_REPORTING",
        "terminal_label": "domain_robust_gate_failed",
        "process_valid": True,
        "terminal_decision_path": (attempt / "decision.json")
        .relative_to(repo)
        .as_posix(),
        "terminal_decision_sha256": tw.sha256_file(attempt / "decision.json"),
    }
    checkpoint = tw.build_postterminal_reports(terminal_state, attempt_root=attempt)
    robustness = json.loads((attempt / "ROBUSTNESS_MAP.json").read_text())
    report = (attempt / "REPORT.md").read_text()
    limitations = (attempt / "LIMITATIONS.md").read_text()
    assert checkpoint["checkpoint_state"] == "POST_TERMINAL_REPORTING"
    assert robustness["supported_co_primary_claim_count"] is None
    assert "no confirmation claim was attempted" in report
    assert "does not evaluate a contact-aware router" in limitations
    assert all((attempt / name).is_file() for name in tw.REPORT_FILENAMES)


def test_report_artifact_mutation_is_not_repaired(tmp_path: Path, monkeypatch):
    repo, attempt = _attempt(tmp_path)
    _install_globals(monkeypatch, repo, attempt)
    contract = _contract(repo, attempt, "power_infeasible")
    audit = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "INDEPENDENT_VERIFICATION",
        "mode": "power_infeasible",
        "passed": True,
        "terminal_label": "domain_robust_gate_failed",
        "read_only_verifier": True,
        "local_production_modules_imported": False,
        "stdout_json_only": True,
        "checks": {"synthetic_verification_complete": True},
        "verifier_contract": {
            "path": contract.relative_to(repo).as_posix(),
            "sha256": tw.sha256_file(contract),
        },
        "source_hashes": {"power_freeze": hashlib.sha256(b"x").hexdigest()},
        "capture": {
            "captured_exclusively": True,
            "wrapper_integrity_passed": True,
            "scientific_verifier_passed": True,
            "verifier_returncode": 0,
        },
    }
    audit_path = attempt / tw.AUDIT_RELATIVE
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n")
    state = _state("power_infeasible")
    tw.build_terminal_decision(state, audit)
    terminal_state = dict(state) | {
        "current_state": "POST_TERMINAL_REPORTING",
        "terminal_label": "domain_robust_gate_failed",
        "process_valid": True,
        "terminal_decision_path": (attempt / "decision.json")
        .relative_to(repo)
        .as_posix(),
        "terminal_decision_sha256": tw.sha256_file(attempt / "decision.json"),
    }
    tw.build_postterminal_reports(terminal_state, attempt_root=attempt)
    (attempt / "REPORT.md").write_text("mutated\n")
    with pytest.raises(tw.TerminalWorkflowError, match="artifact drift"):
        tw.build_postterminal_reports(terminal_state, attempt_root=attempt)
