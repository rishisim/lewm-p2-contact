from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

import pytest


PROGRAM_PATH = Path(__file__).resolve().parents[3] / "program.py"
SPEC = importlib.util.spec_from_file_location("domain_robust_gate_program_test", PROGRAM_PATH)
assert SPEC is not None and SPEC.loader is not None
program = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(program)


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _repo_relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _setup_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_state: str,
    *,
    counters: dict[str, int | bool] | None = None,
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    study = repo / "runs/lewm_domain_robust_gate"
    attempt = study / "attempts/v001"
    attempt.mkdir(parents=True)

    monkeypatch.setattr(program, "ROOT", study)
    monkeypatch.setattr(program, "STUDY_ROOT", study)
    monkeypatch.setattr(program, "REPO_ROOT", repo)
    monkeypatch.setattr(program, "STATE_PATH", study / "STATE.json")
    monkeypatch.setattr(
        program, "STATE_TRANSACTION_PATH", study / "STATE_TRANSACTION.json"
    )
    monkeypatch.setattr(program, "LEDGER_PATH", study / "RESEARCH_LEDGER.jsonl")
    monkeypatch.setattr(
        program, "LEDGER_GENESIS_PATH", study / "LEDGER_CHAIN_GENESIS.json"
    )
    monkeypatch.setattr(program, "CONTROLLER_LOCK_PATH", study / ".program.lock")

    attempt_relative = "runs/lewm_domain_robust_gate/attempts/v001"
    contracts = {
        "no_candidate": attempt / "verifier_contract_no_candidate.json",
        "power_infeasible": attempt / "verifier_contract_power_infeasible.json",
        "confirmation": attempt / "verifier_contract.json",
    }
    for mode, path in contracts.items():
        source_key = {
            "no_candidate": "selection_ledger",
            "power_infeasible": "power_freeze",
            "confirmation": "analysis_result",
        }[mode]
        source_relative = {
            "no_candidate": f"{attempt_relative}/selection/selection_ledger.json",
            "power_infeasible": f"{attempt_relative}/power_analysis.json",
            "confirmation": f"{attempt_relative}/analysis_result.json",
        }[mode]
        _write_json(
            path,
            {
                "schema_version": 1,
                "attempt": "v001",
                "attempt_root": attempt_relative,
                "study_root": "runs/lewm_domain_robust_gate",
                "mode": mode,
                "paths": {source_key: source_relative},
            },
        )

    legacy = b'{"attempt":"v001","event":"legacy_initialization"}\n'
    program.LEDGER_PATH.write_bytes(legacy)
    legacy_hash = hashlib.sha256(legacy).hexdigest()
    _write_json(
        program.LEDGER_GENESIS_PATH,
        {
            "legacy_prefix_bytes": len(legacy),
            "legacy_prefix_sha256": legacy_hash,
            "legacy_prefix_event_count": 1,
        },
    )

    current_index = program.STATE_MACHINE.index(current_state)
    completed = list(program.STATE_MACHINE[:current_index])
    pre_data_seal = attempt / "audit/pre_data_seal.json"
    if "PRE_OUTCOME_SEAL" in completed:
        _write_json(
            pre_data_seal,
            {
                "schema_version": 1,
                "attempt": "v001",
                "checkpoint_state": "PRE_OUTCOME_SEAL",
                "passed": True,
                "sealed_files": {
                    _repo_relative(repo, path): program.sha256_file(path)
                    for path in contracts.values()
                },
            },
        )

    checkpoints: list[dict[str, Any]] = []
    for index, state_name in enumerate(completed):
        if state_name == "PRE_OUTCOME_SEAL":
            evidence = pre_data_seal
        else:
            evidence = _write_json(
                attempt / f"audit/checkpoint_{index:02d}_{state_name.lower()}.json",
                {
                    "schema_version": 1,
                    "attempt": "v001",
                    "checkpoint_state": state_name,
                    "passed": True,
                },
            )
        checkpoints.append(
            {
                "name": f"v001_{state_name.lower()}",
                "created_unix_ns": index + 1,
                "evidence_path": _repo_relative(repo, evidence),
                "evidence_sha256": program.sha256_file(evidence),
                "source_attempt": "v001",
                "verification_lineage": "synthetic_test_checkpoint",
            }
        )

    outcome_counts: dict[str, int | bool] = {
        "fit_outcome_episodes": 0,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    outcome_counts.update(counters or {})
    state: dict[str, Any] = {
        "schema_version": 1,
        "study": "lewm_domain_robust_gate",
        "state_machine": list(program.STATE_MACHINE),
        "current_state": current_state,
        "completed_states": completed,
        "verified_checkpoints": checkpoints,
        "last_verified_checkpoint": checkpoints[-1] if checkpoints else None,
        "active_attempt": "v001",
        "active_attempt_path": attempt_relative,
        "attempt_history": [
            {
                "version": "v001",
                "path": attempt_relative,
                "status": "active",
                "created_unix_ns": 1,
            }
        ],
        "skipped_states": [],
        "terminal_label": None,
        "process_valid": None,
        "scientific_terminal": False,
        "confirmation_terminal": False,
        "expected_fit_episode_count": 1200,
        "expected_selection_episode_count": 2000,
        "expected_smoke_episode_count": 24,
        "maximum_confirmation_episode_count": 18000,
        "next_action": "synthetic test action",
        "ledger_event_count": 1,
        "ledger_head_sha256": f"legacy:{legacy_hash}",
        **outcome_counts,
    }
    _write_json(program.STATE_PATH, state)
    assert program.status()["current_state"] == current_state
    return {
        "repo": repo,
        "study": study,
        "attempt": attempt,
        "contracts": contracts,
        "state": state,
    }


def _early_source(attempt: Path, mode: str) -> Path:
    if mode == "no_candidate":
        return _write_json(
            attempt / "selection/selection_ledger.json",
            {
                "schema_version": 1,
                "attempt": "v001",
                "status": "selection_complete_no_selected_head_refit",
                "candidate_count": 24,
                "eligible_count": 0,
                "selected_candidate_id": None,
                "selected_candidate_index": None,
                "selected_head_refit_after_selection": False,
                "prior_confirmation_outcome_episodes_used": 0,
            },
        )
    return _write_json(
        attempt / "power_analysis.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "status": "binding_post_selection_power_result",
            "decision": "power_infeasible_no_confirmation",
            "binding": True,
            "feasible": False,
            "passed": False,
            "confirmation_generation_authorized_by_power": False,
            "selected_confirmation_episodes_per_regime": None,
            "confirmation_episode_count_per_regime": None,
            "terminal_label_if_infeasible": "domain_robust_gate_failed",
            "fresh_confirmation_outcomes_opened": 0,
        },
    )


@pytest.mark.parametrize(
    ("mode", "trigger"),
    [
        ("no_candidate", "CANDIDATE_SELECTION"),
        ("power_infeasible", "CONFIRMATION_POWER_AND_COHORT_FREEZE"),
    ],
)
def test_process_valid_early_failure_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    trigger: str,
) -> None:
    env = _setup_controller(
        tmp_path,
        monkeypatch,
        trigger,
        counters={"fit_outcome_episodes": 1200, "selection_outcome_episodes": 2000},
    )
    attempt = env["attempt"]
    source = _early_source(attempt, mode)
    staged = program.stage_early_scientific_failure(mode)

    expected_prefix = list(program.STATE_MACHINE[: program.STATE_MACHINE.index(trigger)])
    assert staged["completed_states"] == expected_prefix
    assert len(staged["verified_checkpoints"]) == len(expected_prefix)
    assert staged["current_state"] == "INDEPENDENT_VERIFICATION"
    assert staged["early_scientific_failure"]["trigger_evidence_sha256"] == program.sha256_file(source)
    assert staged["smoke_outcome_episodes"] == 0
    assert staged["confirmation_outcome_episodes_generated"] == 0
    assert staged["confirmation_outcome_episodes_executed"] == 0
    assert staged["confirmation_outcomes_opened_for_analysis"] is False

    contract = env["contracts"][mode]
    audit = _write_json(
        attempt / "audit/independent_verification.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "mode": mode,
            "passed": True,
            "terminal_label": "domain_robust_gate_failed",
            "verifier_contract": {
                "path": _repo_relative(env["repo"], contract),
                "sha256": program.sha256_file(contract),
            },
            "source_hashes": {
                program.EARLY_FAILURE_SPECS[mode]["audit_hash_key"]: {
                    "path": _repo_relative(env["repo"], source),
                    "sha256": program.sha256_file(source),
                    "bytes": source.stat().st_size,
                }
            },
            "checks": {
                "normative_contracts": {
                    "file_count": 17,
                    "manifest_sha256": "a" * 64,
                },
                "ledger_state": {
                    "event_count": staged["ledger_event_count"],
                    "ledger_head_sha256": staged["ledger_head_sha256"],
                },
                "selection_recomputation": {
                    "eligible_count": 0 if mode == "no_candidate" else 1,
                    "selected_candidate_id": None if mode == "no_candidate" else "candidate_07",
                },
            },
            "read_only_verifier": True,
            "local_production_modules_imported": False,
            "stdout_json_only": True,
            "capture": {
                "captured_exclusively": True,
                "wrapper_integrity_passed": True,
                "scientific_verifier_passed": True,
                "verifier_returncode": 0,
            },
        },
    )
    decision = _write_json(
        attempt / "decision.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "passed": True,
            "checkpoint_state": "TERMINAL",
            "early_failure_mode": mode,
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
            "trigger_evidence_path": _repo_relative(env["repo"], source),
            "trigger_evidence_sha256": program.sha256_file(source),
            "independent_verification_path": _repo_relative(env["repo"], audit),
            "independent_verification_sha256": program.sha256_file(audit),
            "smoke_outcome_episodes": 0,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
        },
    )

    terminal = program.finalize_early_scientific_failure(mode)
    assert terminal["terminal_label"] == "domain_robust_gate_failed"
    assert terminal["process_valid"] is True
    assert terminal["confirmation_terminal"] is False
    assert terminal["current_state"] == "POST_TERMINAL_REPORTING"
    assert terminal["terminal_decision_sha256"] == program.sha256_file(decision)
    assert terminal["completed_states"][-2:] == ["INDEPENDENT_VERIFICATION", "TERMINAL"]
    assert all(
        record["reason"] == "preregistered_process_valid_early_scientific_failure"
        for record in terminal["skipped_states"]
    )

    report = _write_json(
        attempt / "audit/post_terminal_reporting.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "checkpoint_state": "POST_TERMINAL_REPORTING",
            "passed": True,
        },
    )
    final = program.advance(
        "POST_TERMINAL_REPORTING", report, "v001_post_terminal", "unused"
    )
    assert final["post_terminal_reporting_complete"] is True
    assert final["completed_states"][-1] == "POST_TERMINAL_REPORTING"
    assert program.status() == final
    with pytest.raises(RuntimeError, match="immutable"):
        program.finalize_early_scientific_failure(mode)
    with pytest.raises(RuntimeError, match="immutable|early"):
        program.update_counts(smoke_outcome_episodes=1)


def test_early_failure_rejects_later_role_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(
        tmp_path,
        monkeypatch,
        "CANDIDATE_SELECTION",
        counters={"fit_outcome_episodes": 1200, "selection_outcome_episodes": 2000},
    )
    _early_source(env["attempt"], "no_candidate")
    forbidden = env["attempt"] / "data/smoke/episode_000.npz"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"forbidden")
    with pytest.raises(RuntimeError, match="forbidden later-role artifacts"):
        program.stage_early_scientific_failure("no_candidate")


def test_normal_terminal_and_post_terminal_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(
        tmp_path,
        monkeypatch,
        "TERMINAL",
        counters={
            "fit_outcome_episodes": 1200,
            "selection_outcome_episodes": 2000,
            "smoke_outcome_episodes": 24,
            "confirmation_outcome_episodes_generated": 4000,
            "confirmation_outcome_episodes_executed": 4000,
            "confirmation_outcomes_opened_for_analysis": True,
        },
    )
    decision = _write_json(
        env["attempt"] / "decision.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "checkpoint_state": "TERMINAL",
            "passed": True,
            "terminal_label": "domain_robust_gate_partial",
            "process_valid": True,
        },
    )
    recorded = program.record_terminal(
        "domain_robust_gate_partial", decision, process_valid=True
    )
    assert recorded["terminal_label"] == "domain_robust_gate_partial"
    entered_reporting = program.advance(
        "TERMINAL", decision, "v001_terminal_decision", "write terminal reports"
    )
    assert entered_reporting["current_state"] == "POST_TERMINAL_REPORTING"
    report = _write_json(
        env["attempt"] / "audit/post_terminal_reporting.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "checkpoint_state": "POST_TERMINAL_REPORTING",
            "passed": True,
        },
    )
    final = program.advance(
        "POST_TERMINAL_REPORTING", report, "v001_post_terminal", "unused"
    )
    assert final["completed_states"] == list(program.STATE_MACHINE)
    assert final["next_action"] == "program_complete_no_further_scientific_or_reporting_transition"
    assert program.status() == final


def test_state_ledger_crash_window_reconciles_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(tmp_path, monkeypatch, "IMPLEMENTATION_COMPLETE")
    evidence = _write_json(
        env["attempt"] / "audit/implementation_complete.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "checkpoint_state": "IMPLEMENTATION_COMPLETE",
            "passed": True,
        },
    )
    original_atomic = program.atomic_json
    crashed = False

    def crash_after_ledger_fsync(
        path: Path, value: dict[str, Any], *, exclusive: bool = False
    ) -> None:
        nonlocal crashed
        if path == program.STATE_PATH and program.STATE_TRANSACTION_PATH.exists() and not crashed:
            crashed = True
            raise OSError("synthetic crash after ledger fsync")
        original_atomic(path, value, exclusive=exclusive)

    monkeypatch.setattr(program, "atomic_json", crash_after_ledger_fsync)
    with pytest.raises(OSError, match="synthetic crash"):
        program.advance(
            "IMPLEMENTATION_COMPLETE", evidence, "v001_implementation", "qualify"
        )
    assert crashed
    assert program.STATE_TRANSACTION_PATH.is_file()
    assert program.verify_ledger_chain()["event_count"] == 2
    assert json.loads(program.STATE_PATH.read_text(encoding="utf-8"))["ledger_event_count"] == 1

    monkeypatch.setattr(program, "atomic_json", original_atomic)
    recovered = program.status()
    assert recovered["current_state"] == "PRESEAL_QUALIFICATION"
    assert recovered["completed_states"][-1] == "IMPLEMENTATION_COMPLETE"
    assert recovered["ledger_event_count"] == 2
    assert not program.STATE_TRANSACTION_PATH.exists()
    assert program.status() == recovered


def test_global_lock_prevents_status_from_racing_live_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(tmp_path, monkeypatch, "IMPLEMENTATION_COMPLETE")
    evidence = _write_json(
        env["attempt"] / "audit/implementation_complete.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "checkpoint_state": "IMPLEMENTATION_COMPLETE",
            "passed": True,
        },
    )
    append_entered = threading.Event()
    release_append = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    errors: list[BaseException] = []
    results: list[dict[str, Any]] = []
    original_append = program.append_ledger

    def paused_append(event: dict[str, Any]) -> dict[str, Any]:
        assert getattr(program._LOCK_LOCAL, "held", False) is True
        append_entered.set()
        if not release_append.wait(timeout=2):
            raise TimeoutError("test did not release the live ledger writer")
        return original_append(event)

    monkeypatch.setattr(program, "append_ledger", paused_append)

    def writer() -> None:
        try:
            results.append(
                program.advance(
                    "IMPLEMENTATION_COMPLETE",
                    evidence,
                    "v001_implementation",
                    "qualify",
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    def reader() -> None:
        reader_started.set()
        try:
            results.append(program.status())
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)
        finally:
            reader_finished.set()

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    assert append_entered.wait(timeout=2)
    assert program.STATE_TRANSACTION_PATH.is_file()
    reader_thread.start()
    assert reader_started.wait(timeout=2)
    assert not reader_finished.wait(timeout=0.05)
    release_append.set()
    writer_thread.join(timeout=2)
    reader_thread.join(timeout=2)
    assert not writer_thread.is_alive() and not reader_thread.is_alive()
    assert not errors
    assert len(results) == 2
    assert all(result["current_state"] == "PRESEAL_QUALIFICATION" for result in results)
    assert program.verify_ledger_chain()["event_count"] == 2
    assert not program.STATE_TRANSACTION_PATH.exists()


def test_version_forward_preserves_source_attempt_lineage_and_resumes_v004(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(tmp_path, monkeypatch, "DIAGNOSTIC_ACCOUNT")
    old_checkpoint = env["state"]["verified_checkpoints"][0]
    v001_invalidity = _write_json(
        env["attempt"] / "audit/v001_procedural_invalidity.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "procedural_invalidity": True,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
        },
    )
    v002 = env["study"] / "attempts/v002"
    v002.mkdir()
    v002_equivalence = _write_json(
        v002 / "audit/v002_equivalence.json",
        {
            "schema_version": 1,
            "attempt": "v002",
            "source_hash_equivalent": True,
            "normalized_ast_equivalent": True,
            "scientific_object_hash_equivalent": True,
            "configuration_hash_equivalent": True,
            "scientific_changes": False,
            "resume_state": "DIAGNOSTIC_ACCOUNT",
            "attempt_parameterization_verified": True,
            "runtime_modules_accept_active_attempt": True,
            "independent_verifier_accepts_active_attempt": True,
            "verified_runtime_modules": [
                "program.py",
                "independent_verify.py",
                "capture_verifier.py",
                "workflow.py",
            ],
            "passed": True,
        },
    )
    forwarded_v002 = program.version_forward(
        "v002", v001_invalidity, v002_equivalence
    )
    assert forwarded_v002["active_attempt"] == "v002"

    v002_invalidity = _write_json(
        v002 / "audit/v002_procedural_invalidity.json",
        {
            "schema_version": 1,
            "attempt": "v002",
            "procedural_invalidity": True,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
        },
    )
    v003 = env["study"] / "attempts/v003"
    v003.mkdir()
    v003_equivalence = _write_json(
        v003 / "audit/v003_equivalence.json",
        {
            "schema_version": 1,
            "attempt": "v003",
            "source_hash_equivalent": True,
            "normalized_ast_equivalent": True,
            "scientific_object_hash_equivalent": True,
            "configuration_hash_equivalent": True,
            "scientific_changes": False,
            "resume_state": "DIAGNOSTIC_ACCOUNT",
            "attempt_parameterization_verified": True,
            "runtime_modules_accept_active_attempt": True,
            "independent_verifier_accepts_active_attempt": True,
            "verified_runtime_modules": [
                "program.py",
                "independent_verify.py",
                "capture_verifier.py",
                "workflow.py",
            ],
            "passed": True,
        },
    )
    forwarded_v003 = program.version_forward(
        "v003", v002_invalidity, v003_equivalence
    )
    assert forwarded_v003["active_attempt"] == "v003"
    v003_invalidity = _write_json(
        v003 / "audit/v003_procedural_invalidity.json",
        {
            "schema_version": 1,
            "attempt": "v003",
            "procedural_invalidity": True,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
        },
    )
    v004 = env["study"] / "attempts/v004"
    v004.mkdir()
    v004_equivalence = _write_json(
        v004 / "audit/v004_equivalence.json",
        {
            "schema_version": 1,
            "attempt": "v004",
            "source_hash_equivalent": True,
            "normalized_ast_equivalent": True,
            "scientific_object_hash_equivalent": True,
            "configuration_hash_equivalent": True,
            "scientific_changes": False,
            "resume_state": "DIAGNOSTIC_ACCOUNT",
            "attempt_parameterization_verified": True,
            "runtime_modules_accept_active_attempt": True,
            "independent_verifier_accepts_active_attempt": True,
            "verified_runtime_modules": [
                "program.py",
                "independent_verify.py",
                "capture_verifier.py",
                "workflow.py",
            ],
            "passed": True,
        },
    )
    forwarded = program.version_forward("v004", v003_invalidity, v004_equivalence)
    assert forwarded["active_attempt"] == "v004"
    assert forwarded["active_attempt_path"].endswith("/v004")
    inherited = forwarded["verified_checkpoints"][0]
    assert inherited["source_attempt"] == "v001"
    assert inherited["evidence_path"] == old_checkpoint["evidence_path"]
    assert [
        item["attempt"] for item in inherited["inherited_into_attempts"]
    ] == ["v002", "v003", "v004"]
    assert [
        (item["old_attempt"], item["new_attempt"])
        for item in forwarded["version_forward_lineage"]
    ] == [("v001", "v002"), ("v002", "v003"), ("v003", "v004")]
    assert all(
        item["resume_state"] == "DIAGNOSTIC_ACCOUNT"
        for item in forwarded["version_forward_lineage"]
    )
    assert program.status() == forwarded

    evidence = _write_json(
        v004 / "audit/diagnostic_account.json",
        {
            "schema_version": 1,
            "attempt": "v004",
            "checkpoint_state": "DIAGNOSTIC_ACCOUNT",
            "passed": True,
        },
    )
    advanced = program.advance(
        "DIAGNOSTIC_ACCOUNT", evidence, "v004_diagnostic", "preregister"
    )
    assert advanced["current_state"] == "PREREGISTRATION_AND_POWER"
    assert [item["source_attempt"] for item in advanced["verified_checkpoints"]] == [
        "v001",
        "v004",
    ]
    assert program.status() == advanced


def _confirmation_counters() -> dict[str, int | bool]:
    return {
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": 2000,
        "smoke_outcome_episodes": 24,
        "confirmation_outcome_episodes_generated": 4000,
        "confirmation_outcome_episodes_executed": 4000,
        "confirmation_outcomes_opened_for_analysis": True,
    }


def _install_confirmation_size(state_path: Path) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["expected_confirmation_episodes_per_regime"] = 1000
    state["expected_confirmation_episode_count"] = 4000
    _write_json(state_path, state)


def _execution_invalid_decision(env: dict[str, Any], audit: Path) -> Path:
    return _write_json(
        env["attempt"] / "decision.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "passed": True,
            "checkpoint_state": "TERMINAL",
            "terminal_basis": "postconfirmation_integrity_failure",
            "terminal_label": "domain_robust_gate_execution_invalid",
            "process_valid": False,
            "independent_verification_path": _repo_relative(env["repo"], audit),
            "independent_verification_sha256": program.sha256_file(audit),
            "confirmation_outcome_episodes_generated": 4000,
            "confirmation_outcome_episodes_executed": 4000,
            "confirmation_outcomes_opened_for_analysis": True,
        },
    )


def test_latency_integrity_failure_reaches_execution_invalid_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(
        tmp_path,
        monkeypatch,
        "LATENCY_AND_RESOURCE_REPORTING",
        counters=_confirmation_counters(),
    )
    _install_confirmation_size(program.STATE_PATH)
    latency = _write_json(
        env["attempt"] / "metrics/latency_and_resources.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "checkpoint_state": "LATENCY_AND_RESOURCE_REPORTING",
            "passed": False,
            "integrity_passed": False,
            "integrity_failure": True,
        },
    )
    staged = program.stage_postconfirmation_integrity_failure()
    assert staged["current_state"] == "INDEPENDENT_VERIFICATION"
    assert staged["postconfirmation_integrity_failure"]["source_sha256"] == program.sha256_file(latency)
    with pytest.raises(RuntimeError, match="narrow controller"):
        program.advance(
            "INDEPENDENT_VERIFICATION", latency, "bad_generic_advance", "terminal"
        )

    contract = env["contracts"]["confirmation"]
    audit = _write_json(
        env["attempt"] / "audit/independent_verification.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "mode": "confirmation",
            "passed": True,
            "terminal_label": "domain_robust_gate_execution_invalid",
            "read_only_verifier": True,
            "integrity_failure": True,
            "verifier_contract": {
                "path": _repo_relative(env["repo"], contract),
                "sha256": program.sha256_file(contract),
            },
            "source_hashes": {
                "latency_resource": {
                    "path": _repo_relative(env["repo"], latency),
                    "sha256": program.sha256_file(latency),
                }
            },
            "checks": {
                "latency_resource_integrity": {"integrity_valid": False},
                "confirmation_evidence_unchanged": True,
            },
            "capture": {
                "captured_exclusively": True,
                "wrapper_integrity_passed": True,
                "scientific_verifier_passed": True,
                "verifier_returncode": 0,
            },
        },
    )
    _execution_invalid_decision(env, audit)
    terminal = program.finalize_postconfirmation_execution_invalid()
    assert terminal["terminal_label"] == "domain_robust_gate_execution_invalid"
    assert terminal["process_valid"] is False
    assert terminal["confirmation_terminal"] is True
    assert terminal["postconfirmation_integrity_failure"]["status"] == "terminal_recorded"
    assert terminal["confirmation_outcome_episodes_generated"] == 4000
    with pytest.raises(RuntimeError, match="immutable"):
        program.update_counts(confirmation_outcome_episodes_generated=4001)


def test_analysis_exception_after_open_mark_cannot_strand_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(
        tmp_path,
        monkeypatch,
        "SEALED_ANALYSIS",
        counters=_confirmation_counters(),
    )
    _install_confirmation_size(program.STATE_PATH)
    failure = _write_json(
        env["attempt"] / "audit/analysis_execution_invalid.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "checkpoint_state": "SEALED_ANALYSIS",
            "passed": False,
            "execution_invalid": True,
            "process_valid": False,
            "error_type": "RuntimeError",
            "error": "synthetic analysis failure after durable outcome-open mark",
            "confirmation_outcome_episodes_generated": 4000,
            "confirmation_outcome_episodes_executed": 4000,
            "confirmation_outcomes_opened_for_analysis": True,
            "analysis_result_present": False,
            "scientific_objects_changed": False,
        },
    )
    staged = program.stage_postconfirmation_integrity_failure()
    assert staged["current_state"] == "INDEPENDENT_VERIFICATION"
    assert staged["completed_states"][-1] == "SEALED_ANALYSIS"
    assert "LATENCY_AND_RESOURCE_REPORTING" not in staged["completed_states"]
    assert staged["postconfirmation_integrity_failure"]["skipped_states"] == [
        "LATENCY_AND_RESOURCE_REPORTING"
    ]
    assert staged["skipped_states"][0]["reason"] == (
        "postconfirmation_integrity_failure_short_circuit"
    )

    contract = env["contracts"]["confirmation"]
    audit = _write_json(
        env["attempt"] / "audit/independent_verification.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "mode": "confirmation",
            "passed": True,
            "terminal_label": "domain_robust_gate_execution_invalid",
            "read_only_verifier": True,
            "integrity_failure": True,
            "verifier_contract": {
                "path": _repo_relative(env["repo"], contract),
                "sha256": program.sha256_file(contract),
            },
            "source_hashes": {
                "analysis_execution_invalid": {
                    "path": _repo_relative(env["repo"], failure),
                    "sha256": program.sha256_file(failure),
                }
            },
            "checks": {
                "outcome_open_mark_durable": True,
                "analysis_integrity": {"integrity_valid": False},
            },
            "capture": {
                "captured_exclusively": True,
                "wrapper_integrity_passed": True,
                "scientific_verifier_passed": True,
                "verifier_returncode": 0,
            },
        },
    )
    _execution_invalid_decision(env, audit)
    terminal = program.finalize_postconfirmation_execution_invalid()
    assert terminal["terminal_label"] == "domain_robust_gate_execution_invalid"
    assert terminal["process_valid"] is False
    assert "LATENCY_AND_RESOURCE_REPORTING" not in terminal["completed_states"]
    assert terminal["completed_states"][-2:] == ["INDEPENDENT_VERIFICATION", "TERMINAL"]
    assert program.status() == terminal


def test_independent_verifier_failure_reaches_execution_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(
        tmp_path,
        monkeypatch,
        "INDEPENDENT_VERIFICATION",
        counters=_confirmation_counters(),
    )
    _install_confirmation_size(program.STATE_PATH)
    contract = env["contracts"]["confirmation"]
    audit = _write_json(
        env["attempt"] / "audit/independent_verification.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "mode": "confirmation",
            "passed": False,
            "error_type": "VerificationError",
            "error": "sealed confirmation input hash drift",
            "read_only_verifier": True,
            "local_production_modules_imported": False,
            "verifier_contract": {
                "path": _repo_relative(env["repo"], contract),
                "sha256": program.sha256_file(contract),
            },
            "capture": {
                "captured_exclusively": True,
                "wrapper_integrity_passed": True,
                "scientific_verifier_passed": False,
                "verifier_returncode": 1,
            },
        },
    )
    _execution_invalid_decision(env, audit)
    terminal = program.finalize_postconfirmation_execution_invalid()
    assert terminal["terminal_label"] == "domain_robust_gate_execution_invalid"
    assert terminal["postconfirmation_integrity_failure"]["status"] == "terminal_recorded"
    assert "source_path" not in terminal["postconfirmation_integrity_failure"]
    assert program.status() == terminal


def test_generic_terminal_cannot_bypass_execution_invalid_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup_controller(tmp_path, monkeypatch, "TERMINAL")
    decision = _write_json(
        env["attempt"] / "decision.json",
        {
            "schema_version": 1,
            "attempt": "v001",
            "passed": True,
            "checkpoint_state": "TERMINAL",
            "terminal_label": "domain_robust_gate_execution_invalid",
            "process_valid": False,
        },
    )
    with pytest.raises(RuntimeError, match="integrity finalizer"):
        program.record_terminal(
            "domain_robust_gate_execution_invalid", decision, process_valid=False
        )
