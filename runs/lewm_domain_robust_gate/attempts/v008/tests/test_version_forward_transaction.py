from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest


ATTEMPT = Path(__file__).resolve().parents[1]


def _module():
    specification = importlib.util.spec_from_file_location(
        "version_forward_transaction_subject",
        ATTEMPT / "version_forward_transaction.py",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


transaction = _module()
_REAL_PREVIEW_CONTROLLER_PROPOSAL = (
    transaction._preview_controller_proposal_locked
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = tmp_path / "repo"
    study = repo / "runs/lewm_domain_robust_gate"
    source_study = ATTEMPT.parents[1]
    v001 = study / "attempts/v001"
    v002 = study / "attempts/v002"
    v006 = study / "attempts/v006"
    v008 = study / "attempts/v008"
    (v001 / "audit").mkdir(parents=True)
    (v002 / "audit").mkdir(parents=True)
    (v006 / "audit").mkdir(parents=True)
    (v008 / "audit").mkdir(parents=True)
    genesis_path = study / "LEDGER_CHAIN_GENESIS.json"
    ledger_path = study / "RESEARCH_LEDGER.jsonl"
    state_path = study / "STATE.json"
    program_path = study / "program.py"
    lock_path = study / ".program.lock"
    transaction_path = study / "STATE_TRANSACTION.json"
    seal_path = v008 / "audit/pre_data_inheritance_seal.json"
    invalidity_path = v006 / "audit/v006_procedural_invalidity.json"
    invalidity_draft_path = v006 / "audit/v006_procedural_invalidity_draft.json"
    prior_receipt_path = v006 / "audit/version_forward_transaction_receipt.json"
    receipt_path = v008 / "audit/version_forward_transaction_receipt.json"
    receipt_journal_path = study / "VERSION_FORWARD_TRANSACTION.json"
    source_path = v008 / "version_forward_transaction.py"
    source_v006 = v006 / "version_forward_transaction.py"
    shutil.copy2(source_study / "LEDGER_CHAIN_GENESIS.json", genesis_path)
    shutil.copy2(source_study / "RESEARCH_LEDGER.jsonl", ledger_path)
    shutil.copy2(source_study / "STATE.json", state_path)
    shutil.copy2(source_study / "program.py", program_path)
    shutil.copy2(
        source_study / "attempts/v006/version_forward_transaction.py",
        source_v006,
    )
    shutil.copy2(
        source_study / "attempts/v006/audit/v006_procedural_invalidity.json",
        invalidity_path,
    )
    shutil.copy2(
        source_study / "attempts/v006/audit/v006_procedural_invalidity_draft.json",
        invalidity_draft_path,
    )
    shutil.copy2(
        source_study
        / "attempts/v006/audit/version_forward_transaction_receipt.json",
        prior_receipt_path,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active_attempt"] == "v006"
    source_repository = source_study.parents[1]
    for checkpoint in state["verified_checkpoints"]:
        relative = checkpoint["evidence_path"]
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_repository / relative, destination)
    base_ledger = ledger_path.read_bytes()
    source_path.write_text("VALUE = 2\n", encoding="utf-8")
    invalidity_sha = hashlib.sha256(invalidity_path.read_bytes()).hexdigest()
    invalidity_draft_sha = hashlib.sha256(
        invalidity_draft_path.read_bytes()
    ).hexdigest()
    prior_receipt_sha = hashlib.sha256(prior_receipt_path.read_bytes()).hexdigest()
    program_sha = hashlib.sha256(program_path.read_bytes()).hexdigest()
    partition_members = {
        "exact_hash": ["exact.py"],
        "normalized_ast": ["normalized.py"],
        "canonical_contracts": ["contract.json"],
        "procedural_only": [],
        "new_lineage_support": ["fit_inheritance.py"],
        "lineage_support": ["version_forward_transaction.py"],
    }
    discovered_paths = sorted(
        relative
        for values in partition_members.values()
        for relative in values
    )
    source_paths = sorted(set(discovered_paths) - {"fit_inheritance.py"})
    _write_json(
        seal_path,
        {
            "attempt": "v008",
            "resume_state": "SELECTION_COHORTS",
            "source_hash_equivalent": True,
            "normalized_ast_equivalent": True,
            "scientific_object_hash_equivalent": True,
            "configuration_hash_equivalent": True,
            "scientific_changes": False,
            "attempt_parameterization_verified": True,
            "runtime_modules_accept_active_attempt": True,
            "independent_verifier_accepts_active_attempt": True,
            "passed": True,
            "root_controls": {"program.py": {"sha256": program_sha}},
            "invalidity_evidence": {"sha256": invalidity_sha},
            "superseded_invalidity_draft": {"sha256": invalidity_draft_sha},
            "source_version_forward_transaction_receipt": {
                "sha256": prior_receipt_sha
            },
            "sealed_files": {
                source_path.relative_to(repo).as_posix(): hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest()
            },
            "v008_manifest_closure": {"discovered_paths": discovered_paths},
            "source_partitions": {
                name: [{"relative_path": relative} for relative in values]
                for name, values in partition_members.items()
            },
            "complete_source_partition": {
                "source_paths": source_paths,
                "target_paths": discovered_paths,
                "partition_members": partition_members,
                "overlap": [],
                "unpartitioned_source": [],
                "unpartitioned_target": [],
                "passed": True,
            },
        },
    )
    values = {
        "REPOSITORY_ROOT": repo,
        "STUDY_ROOT": study,
        "ATTEMPT_ROOT": v008,
        "LOCK_PATH": lock_path,
        "STATE_PATH": state_path,
        "LEDGER_PATH": ledger_path,
        "GENESIS_PATH": genesis_path,
        "TRANSACTION_PATH": transaction_path,
        "PROGRAM_PATH": program_path,
        "SEAL_PATH": seal_path,
        "INVALIDITY_PATH": invalidity_path,
        "INVALIDITY_DRAFT_PATH": invalidity_draft_path,
        "PRIOR_RECEIPT_PATH": prior_receipt_path,
        "RECEIPT_PATH": receipt_path,
        "RECEIPT_JOURNAL_PATH": receipt_journal_path,
        "TRANSACTION_SOURCE_PATH": source_path,
    }
    for name, value in values.items():
        monkeypatch.setattr(transaction, name, value)
    # Staging paths are process-local mutable adapter residues and must follow
    # the configured synthetic study root too.
    monkeypatch.setattr(
        transaction,
        "PENDING_STAGING_PATH",
        study / ".STATE_TRANSACTION.json.v008-staging",
    )
    monkeypatch.setattr(
        transaction, "STATE_STAGING_PATH", study / ".STATE.json.v008-staging"
    )
    monkeypatch.setattr(
        transaction,
        "RECEIPT_JOURNAL_STAGING_PATH",
        study / ".VERSION_FORWARD_TRANSACTION.json.v008-staging",
    )
    monkeypatch.setattr(
        transaction,
        "RECEIPT_STAGING_PATH",
        v008 / "audit/.version_forward_transaction_receipt.json.v008-staging",
    )

    partition = {
        "source_path_count": 4,
        "target_path_count": 5,
        "partition_counts": {
            "exact_hash": 1,
            "normalized_ast": 1,
            "canonical_contracts": 1,
            "procedural_only": 0,
            "new_lineage_support": 1,
            "lineage_support": 1,
        },
    }

    def standalone(phase: str) -> dict[str, Any]:
        return {
            "passed": True,
            "phase": phase,
            "attempt": "v008",
            "active_attempt": "v008",
            "science_attempt": "v001",
            "seal_path": transaction._relative(seal_path),
            "seal_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(),
            "partition_file_count": 5,
            "sealed_file_count": 1,
            "authorized_early_verifier_state": None,
            "authorized_role_count_recovery": None,
            "outcome_arrays_opened": False,
            "output_paths_created": 0,
            "read_only": True,
        }

    def preverify() -> dict[str, Any]:
        return {
            "producer": {
                "passed": True,
                "phase": "pre-forward",
                "attempt": "v008",
                "seal_path": transaction._relative(seal_path),
                "seal_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(),
                "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                "outcome_arrays_opened": False,
                "read_only": True,
            },
            "standalone": standalone("pre-forward"),
            "independent_partitions": partition,
        }

    monkeypatch.setattr(transaction, "_preverify", preverify)
    monkeypatch.setattr(
        transaction,
        "_postverify",
        lambda: {
            "standalone": standalone("post-forward"),
            "independent_partitions": partition,
        },
    )

    def preview(
        base_state: dict[str, Any], *, created_unix_ns: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        proposed = json.loads(json.dumps(base_state))
        seal_sha = hashlib.sha256(seal_path.read_bytes()).hexdigest()
        history = proposed["attempt_history"]
        source_history = next(item for item in history if item["version"] == "v006")
        source_history.update(
            {
                "status": "invalid_zero_confirmation_outcome_procedural",
                "invalidity_evidence_path": transaction._relative(invalidity_path),
                "invalidity_evidence_sha256": invalidity_sha,
            }
        )
        history.append(
            {
                "version": "v008",
                "path": transaction._relative(v008),
                "status": "active_zero_confirmation_outcome_version_forward",
                "created_unix_ns": created_unix_ns,
                "version_forward_evidence_path": transaction._relative(seal_path),
                "version_forward_evidence_sha256": seal_sha,
                "attempt_parameterization_verified": True,
            }
        )
        edge = {
            "old_attempt": "v006",
            "new_attempt": "v008",
            "resume_state": "SELECTION_COHORTS",
            "equivalence_path": transaction._relative(seal_path),
            "equivalence_sha256": seal_sha,
            "invalidity_path": transaction._relative(invalidity_path),
            "invalidity_sha256": invalidity_sha,
            "inherited_verified_checkpoints": [],
            "attempt_parameterization_verified": True,
        }
        proposed.update(
            {
                "active_attempt": "v008",
                "active_attempt_path": transaction._relative(v008),
                "updated_unix_ns": created_unix_ns,
                "version_forward_lineage": [
                    *proposed["version_forward_lineage"], edge
                ],
            }
        )
        event = {
            "created_unix_ns": created_unix_ns,
            "event": "zero_confirmation_outcome_version_forward",
            "attempt": "v008",
            **edge,
        }
        return proposed, [event]

    monkeypatch.setattr(transaction, "_preview_controller_proposal_locked", preview)
    return {**values, "state": state, "base_ledger": base_ledger}


def _configure_real_production_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    """Create a complete isolated copy while leaving every verifier unmocked."""

    source_study = ATTEMPT.parents[1]
    source_repository = source_study.parents[1]
    repository = tmp_path / "repo"
    study = repository / "runs/lewm_domain_robust_gate"
    (repository / "runs").mkdir(parents=True)
    shutil.copytree(
        source_study,
        study,
        ignore=shutil.ignore_patterns(
            ".pytest_cache", "__pycache__", "*.pyc", "*.pyo"
        ),
    )

    # The v001 seal also authenticates a small set of repository-external
    # model/config inputs.  Copy their exact bytes so the real preparation and
    # independent implementations can rehash the complete closed world.
    source_seal = json.loads(
        (study / "attempts/v001/audit/pre_data_seal.json").read_text(
            encoding="utf-8"
        )
    )
    for relative in source_seal["sealed_files"]:
        destination = repository / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_repository / relative, destination)

    attempt = study / "attempts/v008"
    audit = attempt / "audit"
    seal = audit / "pre_data_inheritance_seal.json"
    assert not os.path.lexists(seal)
    specification = importlib.util.spec_from_file_location(
        "real_v008_prepare_copy", attempt / "version_forward_prepare.py"
    )
    assert specification is not None and specification.loader is not None
    prepare = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(prepare)
    value = prepare.build_inheritance_seal(study)
    assert value["passed"] is True
    prepare.write_exclusive(seal, value)

    paths = {
        "REPOSITORY_ROOT": repository,
        "STUDY_ROOT": study,
        "ATTEMPT_ROOT": attempt,
        "LOCK_PATH": study / ".program.lock",
        "STATE_PATH": study / "STATE.json",
        "LEDGER_PATH": study / "RESEARCH_LEDGER.jsonl",
        "GENESIS_PATH": study / "LEDGER_CHAIN_GENESIS.json",
        "TRANSACTION_PATH": study / "STATE_TRANSACTION.json",
        "PROGRAM_PATH": study / "program.py",
        "SEAL_PATH": seal,
        "INVALIDITY_PATH": (
            study / "attempts/v006/audit/v006_procedural_invalidity.json"
        ),
        "INVALIDITY_DRAFT_PATH": (
            study / "attempts/v006/audit/v006_procedural_invalidity_draft.json"
        ),
        "PRIOR_RECEIPT_PATH": (
            study
            / "attempts/v006/audit/version_forward_transaction_receipt.json"
        ),
        "RECEIPT_PATH": audit / "version_forward_transaction_receipt.json",
        "PENDING_STAGING_PATH": study / ".STATE_TRANSACTION.json.v008-staging",
        "STATE_STAGING_PATH": study / ".STATE.json.v008-staging",
        "RECEIPT_JOURNAL_PATH": study / "VERSION_FORWARD_TRANSACTION.json",
        "RECEIPT_JOURNAL_STAGING_PATH": (
            study / ".VERSION_FORWARD_TRANSACTION.json.v008-staging"
        ),
        "RECEIPT_STAGING_PATH": (
            audit / ".version_forward_transaction_receipt.json.v008-staging"
        ),
        "TRANSACTION_SOURCE_PATH": attempt / "version_forward_transaction.py",
    }
    for name, path in paths.items():
        monkeypatch.setattr(transaction, name, path)
    return paths


def _commit_controller(env: dict[str, Any]) -> dict[str, Any]:
    if os.path.lexists(env["RECEIPT_JOURNAL_PATH"]):
        journal = json.loads(env["RECEIPT_JOURNAL_PATH"].read_text(encoding="utf-8"))
        pending = journal["predicted_controller_transaction"]
        env["LEDGER_PATH"].write_bytes(
            env["base_ledger"] + pending["expected_suffix"].encode("ascii")
        )
        _write_json(env["STATE_PATH"], pending["intended_state"])
        return pending["intended_state"]
    genesis_payload = env["GENESIS_PATH"].read_bytes()
    chain = transaction._ledger_summary(env["base_ledger"], genesis_payload)
    event = {
        "created_unix_ns": 1,
        "event": "zero_confirmation_outcome_version_forward",
        "seq": chain["event_count"] + 1,
        "prev_sha256": chain["head_sha256"],
    }
    event["record_sha256"] = hashlib.sha256(_canonical(event)).hexdigest()
    env["LEDGER_PATH"].write_bytes(
        env["base_ledger"] + _canonical(event) + b"\n"
    )
    state = dict(env["state"])
    state.update(
        {
            "active_attempt": "v008",
            "ledger_event_count": chain["event_count"] + 1,
            "ledger_head_sha256": event["record_sha256"],
            "version_forward_lineage": [
                *state["version_forward_lineage"],
                {
                    "old_attempt": "v006",
                    "new_attempt": "v008",
                    "resume_state": "SELECTION_COHORTS",
                    "equivalence_path": transaction._relative(env["SEAL_PATH"]),
                    "invalidity_path": transaction._relative(env["INVALIDITY_PATH"]),
                }
            ],
        }
    )
    state[transaction.ADAPTER_STATE_KEY] = transaction._adapter_marker(
        event_count=chain["event_count"] + 1,
        head_sha256=event["record_sha256"],
        operation_sha256="a" * 64,
        state_without_marker=state,
        root_record=transaction._source_record(env["PROGRAM_PATH"]),
        adapter_record=transaction._source_record(
            transaction.TRANSACTION_SOURCE_PATH
        ),
    )
    _write_json(env["STATE_PATH"], state)
    return state


def _ensure_v008_base(env: dict[str, Any]) -> dict[str, Any]:
    current = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    if current.get("active_attempt") == "v006":
        current = _commit_controller(env)
    env["state"] = current
    return current


def _durable_target_state(env: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    intended = dict(_ensure_v008_base(env))
    intended["active_attempt"] = "v008"
    intended["selection_outcome_episodes"] += 1
    intended["updated_unix_ns"] = 7
    event = {
        "created_unix_ns": 7,
        "event": "outcome_counts_updated",
        "attempt": "v008",
        "fields": {
            "selection_outcome_episodes": intended[
                "selection_outcome_episodes"
            ]
        },
    }
    return intended, event


def _durable_commit(env: dict[str, Any]) -> dict[str, Any]:
    intended, event = _durable_target_state(env)
    with transaction._adapter_lock():
        return transaction._durable_commit_state_with_events(intended, [event])


def _commit_fit_count_descendant(
    env: dict[str, Any], *, created_unix_ns: int = 2
) -> dict[str, Any]:
    intended = json.loads(
        json.dumps(json.loads(env["STATE_PATH"].read_text(encoding="utf-8")))
    )
    intended["selection_outcome_episodes"] += 1
    intended["updated_unix_ns"] = created_unix_ns
    with transaction._adapter_lock():
        return transaction._durable_commit_state_with_events(
            intended,
            [
                {
                    "created_unix_ns": created_unix_ns,
                    "event": "outcome_counts_updated",
                    "attempt": "v008",
                    "fields": {
                        "selection_outcome_episodes": intended[
                            "selection_outcome_episodes"
                        ]
                    },
                }
            ],
        )


def _commit_ordinary_checkpoint(
    env: dict[str, Any], *, created_unix_ns: int
) -> dict[str, Any]:
    current = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    completed = current["current_state"]
    count_requirement = {
        "FIT_COHORTS": ("fit_outcome_episodes", "expected_fit_episode_count"),
        "SELECTION_COHORTS": (
            "selection_outcome_episodes", "expected_selection_episode_count"
        ),
    }.get(completed)
    if count_requirement is not None:
        field, expected_field = count_requirement
        if current[field] != current[expected_field]:
            intended_count = json.loads(json.dumps(current))
            intended_count[field] = current[expected_field]
            intended_count["updated_unix_ns"] = created_unix_ns
            with transaction._adapter_lock():
                transaction._durable_commit_state_with_events(
                    intended_count,
                    [
                        {
                            "event": "outcome_counts_updated",
                            "attempt": "v008",
                            "fields": {field: intended_count[field]},
                            "created_unix_ns": created_unix_ns,
                        }
                    ],
                )
            current = json.loads(
                env["STATE_PATH"].read_text(encoding="utf-8")
            )
    position = transaction.CONTROLLER_STATE_MACHINE.index(completed)
    next_state = transaction.CONTROLLER_STATE_MACHINE[position + 1]
    evidence = (
        env["ATTEMPT_ROOT"]
        / "audit"
        / f"synthetic_{created_unix_ns}_{completed.lower()}.json"
    )
    _write_json(
        evidence,
        {
            "attempt": "v008",
            "checkpoint_state": completed,
            "passed": True,
        },
    )
    evidence_relative = transaction._relative(evidence)
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    checkpoint = {
        "name": f"v008_synthetic_{completed.lower()}",
        "created_unix_ns": created_unix_ns,
        "evidence_path": evidence_relative,
        "evidence_sha256": evidence_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_checkpoint",
    }
    intended = json.loads(json.dumps(current))
    intended["completed_states"].append(completed)
    intended["current_state"] = next_state
    intended["last_verified_checkpoint"] = checkpoint
    intended["verified_checkpoints"].append(checkpoint)
    intended["next_action"] = f"continue synthetic transition to {next_state}"
    intended["updated_unix_ns"] = created_unix_ns
    with transaction._adapter_lock():
        return transaction._durable_commit_state_with_events(
            intended,
            [
                {
                    "event": "state_completed",
                    "attempt": "v008",
                    "completed_state": completed,
                    "next_state": next_state,
                    "checkpoint_name": checkpoint["name"],
                    "evidence_path": evidence_relative,
                    "evidence_sha256": evidence_sha256,
                    "created_unix_ns": created_unix_ns,
                }
            ],
        )


def _synthetic_no_candidate_transition(
    env: dict[str, Any], *, created_unix_ns: int
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    source = env["ATTEMPT_ROOT"] / "selection/selection_ledger.json"
    contract = env["ATTEMPT_ROOT"] / "verifier_contract_no_candidate.json"
    _write_json(
        source,
        {
            "attempt": "v008",
            "status": "selection_complete_no_selected_head_refit",
            "candidate_count": 24,
            "eligible_count": 0,
            "selected_candidate_id": None,
            "selected_candidate_index": None,
            "selected_head_refit_after_selection": False,
            "prior_confirmation_outcome_episodes_used": 0,
        },
    )
    source_relative = transaction._relative(source)
    _write_json(
        contract,
        {
            "schema_version": 1,
            "attempt": "v008",
            "attempt_root": transaction._relative(env["ATTEMPT_ROOT"]),
            "mode": "no_candidate",
            "paths": {"selection_ledger": source_relative},
        },
    )
    contract_relative = transaction._relative(contract)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    contract_sha256 = hashlib.sha256(contract.read_bytes()).hexdigest()
    skipped = list(
        transaction.CONTROLLER_STATE_MACHINE[
            transaction.CONTROLLER_STATE_MACHINE.index("CANDIDATE_SELECTION") + 1:
            transaction.CONTROLLER_STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
        ]
    )
    intended = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    intended["early_scientific_failure"] = {
        "mode": "no_candidate",
        "status": "awaiting_independent_verification",
        "trigger_state": "CANDIDATE_SELECTION",
        "trigger_evidence_path": source_relative,
        "trigger_evidence_sha256": source_sha256,
        "verifier_contract_path": contract_relative,
        "verifier_contract_sha256": contract_sha256,
        "skipped_states": skipped,
        "staged_unix_ns": created_unix_ns,
        "required_terminal_label": "domain_robust_gate_failed",
        "required_process_valid": True,
    }
    intended["skipped_states"] = [
        {
            "state": name,
            "mode": "no_candidate",
            "reason": "preregistered_process_valid_early_scientific_failure",
            "trigger_evidence_path": source_relative,
            "trigger_evidence_sha256": source_sha256,
            "recorded_unix_ns": created_unix_ns,
        }
        for name in skipped
    ]
    intended["current_state"] = "INDEPENDENT_VERIFICATION"
    intended["next_action"] = (
        "run the standalone read-only verifier in the exact staged mode and "
        "capture audit/independent_verification.json before finalizing"
    )
    intended["updated_unix_ns"] = created_unix_ns
    event = {
        "event": "preregistered_early_scientific_failure_staged",
        "attempt": "v008",
        "mode": "no_candidate",
        "trigger_state": "CANDIDATE_SELECTION",
        "trigger_evidence_path": source_relative,
        "trigger_evidence_sha256": source_sha256,
        "verifier_contract_path": contract_relative,
        "verifier_contract_sha256": contract_sha256,
        "skipped_states": skipped,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
        "created_unix_ns": created_unix_ns,
    }
    return intended, event, skipped


def _stage_synthetic_no_candidate(
    env: dict[str, Any], *, created_unix_ns: int
) -> list[str]:
    intended, event, skipped = _synthetic_no_candidate_transition(
        env, created_unix_ns=created_unix_ns
    )
    with transaction._adapter_lock():
        transaction._durable_commit_state_with_events(intended, [event])
    return skipped


def _synthetic_early_terminal_transition(
    env: dict[str, Any], *, skipped: list[str], created_unix_ns: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = env["ATTEMPT_ROOT"] / "audit/independent_verification.json"
    decision_file = env["ATTEMPT_ROOT"] / "decision.json"
    intended = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    early = intended["early_scientific_failure"]
    _write_json(
        audit,
        {
            "schema_version": 1,
            "attempt": "v008",
            "mode": "no_candidate",
            "passed": True,
            "terminal_label": "domain_robust_gate_failed",
            "read_only_verifier": True,
            "checks": {"synthetic_independent_replay": True},
            "verifier_contract": {
                "path": early["verifier_contract_path"],
                "sha256": early["verifier_contract_sha256"],
            },
            "source_hashes": {
                "selection_ledger": early["trigger_evidence_sha256"]
            },
        },
    )
    audit_relative = transaction._relative(audit)
    decision_relative = decision_file.relative_to(
        env["REPOSITORY_ROOT"]
    ).as_posix()
    audit_sha256 = hashlib.sha256(audit.read_bytes()).hexdigest()
    _write_json(
        decision_file,
        {
            "schema_version": 1,
            "attempt": "v008",
            "passed": True,
            "checkpoint_state": "TERMINAL",
            "early_failure_mode": "no_candidate",
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
            "trigger_evidence_path": early["trigger_evidence_path"],
            "trigger_evidence_sha256": early["trigger_evidence_sha256"],
            "independent_verification_path": audit_relative,
            "independent_verification_sha256": audit_sha256,
            "smoke_outcome_episodes": 0,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
        },
    )
    decision_sha256 = hashlib.sha256(decision_file.read_bytes()).hexdigest()
    audit_checkpoint = {
        "name": "v008_no_candidate_independent_verification_passed",
        "created_unix_ns": created_unix_ns,
        "evidence_path": audit_relative,
        "evidence_sha256": audit_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_early_scientific_failure_checkpoint",
    }
    decision_checkpoint = {
        "name": "v008_no_candidate_terminal_decision_recorded",
        "created_unix_ns": created_unix_ns,
        "evidence_path": decision_relative,
        "evidence_sha256": decision_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_early_scientific_failure_checkpoint",
    }
    intended["completed_states"].extend(("INDEPENDENT_VERIFICATION", "TERMINAL"))
    intended["verified_checkpoints"].extend(
        (audit_checkpoint, decision_checkpoint)
    )
    intended["last_verified_checkpoint"] = decision_checkpoint
    intended["current_state"] = "POST_TERMINAL_REPORTING"
    intended["terminal_label"] = "domain_robust_gate_failed"
    intended["process_valid"] = True
    intended["scientific_terminal"] = True
    intended["confirmation_terminal"] = False
    intended["terminal_basis"] = "preregistered_no_candidate"
    intended["terminal_decision_path"] = decision_relative
    intended["terminal_decision_sha256"] = decision_sha256
    intended["early_scientific_failure"].update(
        {
            "status": "terminal_recorded",
            "independent_verification_path": audit_relative,
            "independent_verification_sha256": audit_sha256,
            "decision_path": decision_relative,
            "decision_sha256": decision_sha256,
            "terminal_recorded_unix_ns": created_unix_ns,
        }
    )
    intended["next_action"] = (
        "write the concise terminal report, robustness map, audit, limitations, "
        "and next project-scoped task; then checkpoint POST_TERMINAL_REPORTING"
    )
    intended["updated_unix_ns"] = created_unix_ns
    events = [
        {
            "event": "state_completed",
            "attempt": "v008",
            "completed_state": "INDEPENDENT_VERIFICATION",
            "next_state": "TERMINAL",
            "checkpoint_name": audit_checkpoint["name"],
            "evidence_path": audit_relative,
            "evidence_sha256": audit_sha256,
            "early_failure_mode": "no_candidate",
            "created_unix_ns": created_unix_ns,
        },
        {
            "event": "scientific_terminal_decision_recorded",
            "attempt": "v008",
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
            "confirmation_terminal": False,
            "early_failure_mode": "no_candidate",
            "decision_path": decision_relative,
            "decision_sha256": decision_sha256,
            "created_unix_ns": created_unix_ns,
        },
        {
            "event": "state_completed",
            "attempt": "v008",
            "completed_state": "TERMINAL",
            "next_state": "POST_TERMINAL_REPORTING",
            "checkpoint_name": decision_checkpoint["name"],
            "evidence_path": decision_relative,
            "evidence_sha256": decision_sha256,
            "early_failure_mode": "no_candidate",
            "skipped_states": skipped,
            "created_unix_ns": created_unix_ns,
        },
    ]
    return intended, events


def _journal_bound_controller(env: dict[str, Any]) -> dict[str, Any]:
    journal = json.loads(
        env["RECEIPT_JOURNAL_PATH"].read_text(encoding="utf-8")
    )
    proposal = journal["controller_proposal"]
    state = json.loads(json.dumps(proposal["state"]))
    state[transaction.PRIOR_ADAPTER_STATE_KEY] = env["state"][
        transaction.PRIOR_ADAPTER_STATE_KEY
    ]
    events = json.loads(json.dumps(proposal["events"]))
    for event in events:
        event.pop(transaction.SOURCE_ADAPTER_MARKER_FIELD, None)
    return transaction._durable_commit_state_with_events(
        state, events
    )


def _install_synthetic_root_program(env: dict[str, Any]) -> None:
    values = {
        "state": str(env["STATE_PATH"]),
        "transaction": str(env["TRANSACTION_PATH"]),
        "ledger": str(env["LEDGER_PATH"]),
        "genesis": str(env["GENESIS_PATH"]),
        "lock": str(env["LOCK_PATH"]),
    }
    source = f'''\
import json
import threading
from pathlib import Path

STATE_PATH = Path({values["state"]!r})
STATE_TRANSACTION_PATH = Path({values["transaction"]!r})
LEDGER_PATH = Path({values["ledger"]!r})
LEDGER_GENESIS_PATH = Path({values["genesis"]!r})
CONTROLLER_LOCK_PATH = Path({values["lock"]!r})
_LOCK_LOCAL = threading.local()

def _status_locked():
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    chain = verify_ledger_chain()
    if state.get("ledger_event_count") != chain["event_count"]:
        raise RuntimeError("synthetic state/count drift")
    if state.get("ledger_head_sha256") != chain["head_sha256"]:
        raise RuntimeError("synthetic state/head drift")
    return state

def _raw_version_forward(new_attempt, invalidity, equivalence):
    state = _status_locked()
    state["active_attempt"] = new_attempt
    return _commit_state_with_events(state, [{{
        "created_unix_ns": 11,
        "event": "zero_confirmation_outcome_version_forward",
        "attempt": new_attempt,
    }}])

def version_forward(*args, **kwargs):
    raise RuntimeError("decorated wrapper must not be invoked")

version_forward.__wrapped__ = _raw_version_forward
'''
    env["PROGRAM_PATH"].write_text(source, encoding="utf-8")
    seal = json.loads(env["SEAL_PATH"].read_text(encoding="utf-8"))
    seal["root_controls"]["program.py"]["sha256"] = hashlib.sha256(
        env["PROGRAM_PATH"].read_bytes()
    ).hexdigest()
    _write_json(env["SEAL_PATH"], seal)


def test_atomic_transaction_writes_exact_no_replace_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    result = transaction.execute_version_forward()
    assert result["passed"] is True
    assert env["RECEIPT_PATH"].is_file()
    assert env["RECEIPT_PATH"].stat().st_nlink == 1
    assert transaction.verify_transaction_receipt()["receipt_sha256"] == result["receipt_sha256"]
    assert transaction.execute_version_forward() == result


@pytest.mark.parametrize(
    "boundary", ["receipt_staging_fsync", "receipt_link_parent_fsync"]
)
def test_real_production_verifiers_recover_receipt_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    paths = _configure_real_production_copy(tmp_path, monkeypatch)
    fired = False

    def crash(name: str) -> None:
        nonlocal fired
        if name == boundary and not fired:
            fired = True
            raise OSError(f"real verifier recovery crash at {name}")

    # No verifier, controller, or proposal function is patched in this test.
    # The first invocation reaches the real receipt publication boundary.
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="real verifier recovery crash"):
        transaction.execute_version_forward()
    assert fired is True
    assert paths["RECEIPT_JOURNAL_PATH"].is_file()

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    result = transaction.execute_version_forward()
    assert result["passed"] is True
    assert transaction.verify_transaction_receipt()["passed"] is True
    receipt = json.loads(paths["RECEIPT_PATH"].read_text(encoding="utf-8"))
    assert paths["RECEIPT_PATH"].stat().st_nlink == 1
    assert hashlib.sha256(transaction._pretty_bytes(receipt)).hexdigest() == result[
        "receipt_sha256"
    ]
    assert not any(
        os.path.lexists(path)
        for path in (
            paths["TRANSACTION_PATH"],
            paths["PENDING_STAGING_PATH"],
            paths["STATE_STAGING_PATH"],
            paths["RECEIPT_JOURNAL_PATH"],
            paths["RECEIPT_JOURNAL_STAGING_PATH"],
            paths["RECEIPT_STAGING_PATH"],
        )
    )

    # Exercise the separately written independent receipt implementation on
    # the recovered production transaction, not merely the adapter verifier.
    independent_specification = importlib.util.spec_from_file_location(
        "real_v008_independent_copy",
        paths["ATTEMPT_ROOT"] / "independent_verify.py",
    )
    assert (
        independent_specification is not None
        and independent_specification.loader is not None
    )
    independent = importlib.util.module_from_spec(independent_specification)
    independent_specification.loader.exec_module(independent)
    policy = independent.PathPolicy(
        paths["REPOSITORY_ROOT"],
        "runs/lewm_domain_robust_gate/attempts/v008",
        "runs/lewm_domain_robust_gate",
    )
    state = json.loads(paths["STATE_PATH"].read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in paths["LEDGER_PATH"].read_text(encoding="utf-8").splitlines()
    ]
    seal = json.loads(paths["SEAL_PATH"].read_text(encoding="utf-8"))
    independent_evidence = (
        independent._verify_version_forward_transaction_receipt(
            policy=policy,
            current_state=state,
            current_events=events,
            expected_seal_sha256=hashlib.sha256(
                paths["SEAL_PATH"].read_bytes()
            ).hexdigest(),
            expected_invalidity_sha256=hashlib.sha256(
                paths["INVALIDITY_PATH"].read_bytes()
            ).hexdigest(),
            expected_edge=state["version_forward_lineage"][-1],
            expected_source_marker=seal["source_controller_adapter_marker"],
                expected_equivalence=independent.verify_equivalence_partitions(
                    seal, policy
                ),
                expected_active_sealed_file_summary=independent.verify_file_map(
                    seal["sealed_files"], policy, scope="repository"
                ),
            )
        )
    assert independent_evidence["passed"] is True
    if boundary == "receipt_staging_fsync":
        standalone = transaction._load_module(
            "standalone_receipt_absence_subject",
            paths["ATTEMPT_ROOT"] / "verify_version_forward.py",
        )
        receipt_bytes = paths["RECEIPT_PATH"].read_bytes()
        paths["RECEIPT_PATH"].unlink()
        with pytest.raises(
            standalone.VerificationError,
            match="lacks transaction provenance",
        ):
            standalone.verify(
                paths["SEAL_PATH"],
                paths["STUDY_ROOT"],
                phase="post-forward",
            )
        paths["RECEIPT_PATH"].write_bytes(receipt_bytes)


def test_real_root_descendant_checkpoint_replays_in_all_three_verifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _configure_real_production_copy(tmp_path, monkeypatch)
    assert transaction.execute_version_forward()["passed"] is True
    counted = transaction.update_counts(selection_outcome_episodes=2000)
    assert counted["selection_outcome_episodes"] == 2000
    evidence = paths["ATTEMPT_ROOT"] / "audit/selection_cohorts.json"
    _write_json(
        evidence,
        {
            "attempt": "v008",
            "checkpoint_state": "SELECTION_COHORTS",
            "passed": True,
        },
    )
    advanced = transaction.advance(
        "SELECTION_COHORTS",
        evidence,
        "v008_real_root_selection_checkpoint",
        "apply the preregistered candidate-selection rule",
    )
    assert advanced["current_state"] == "CANDIDATE_SELECTION"
    assert transaction.verify_transaction_receipt()["passed"] is True

    standalone = transaction._load_module(
        "real_v008_descendant_standalone",
        paths["ATTEMPT_ROOT"] / "verify_version_forward.py",
    )
    receipt = json.loads(paths["RECEIPT_PATH"].read_text(encoding="utf-8"))
    standalone_state = json.loads(
        paths["STATE_PATH"].read_text(encoding="utf-8")
    )
    standalone_events = [
        json.loads(line)
        for line in paths["LEDGER_PATH"].read_text(encoding="utf-8").splitlines()
    ]
    standalone_groups = standalone._verify_descendant_adapter_transaction_chain(
        state=standalone_state,
        events=standalone_events,
        ledger_payload=paths["LEDGER_PATH"].read_bytes(),
        target=paths["ATTEMPT_ROOT"],
        repo=paths["REPOSITORY_ROOT"],
        seal_path=paths["SEAL_PATH"],
        source_state=receipt["pre_snapshot"]["state"]["object"],
        source_ledger=receipt["pre_snapshot"]["ledger"],
        authorized_transaction_context_sha256=None,
    )
    assert standalone_groups == 2

    independent = transaction._load_module(
        "real_v008_descendant_independent",
        paths["ATTEMPT_ROOT"] / "independent_verify.py",
    )
    policy = independent.PathPolicy(
        paths["REPOSITORY_ROOT"],
        "runs/lewm_domain_robust_gate/attempts/v008",
        "runs/lewm_domain_robust_gate",
    )
    state = json.loads(paths["STATE_PATH"].read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in paths["LEDGER_PATH"].read_text(encoding="utf-8").splitlines()
    ]
    seal = json.loads(paths["SEAL_PATH"].read_text(encoding="utf-8"))
    independent_result = independent._verify_version_forward_transaction_receipt(
        policy=policy,
        current_state=state,
        current_events=events,
        expected_seal_sha256=hashlib.sha256(
            paths["SEAL_PATH"].read_bytes()
        ).hexdigest(),
        expected_invalidity_sha256=hashlib.sha256(
            paths["INVALIDITY_PATH"].read_bytes()
        ).hexdigest(),
        expected_edge=state["version_forward_lineage"][-1],
        expected_source_marker=seal["source_controller_adapter_marker"],
        expected_equivalence=receipt["post_verifiers"]["independent_partitions"],
        expected_active_sealed_file_summary=independent.verify_file_map(
            seal["sealed_files"], policy, scope="repository"
        ),
    )
    assert independent_result["descendant_adapter_chain"] == {
        "group_count": 2,
        "event_count": 2,
        "passed": True,
    }


def test_authenticated_root_method_is_narrowly_patched_into_durable_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    source_event_count = env["state"]["ledger_event_count"]
    monkeypatch.setattr(
        transaction,
        "_preview_controller_proposal_locked",
        _REAL_PREVIEW_CONTROLLER_PROPOSAL,
    )

    with transaction._adapter_lock():
        journal = transaction._build_receipt_journal_locked(
            prospective_created_unix_ns=10
        )
        transaction._publish_receipt_journal(journal)
        transaction._ADAPTER_LOCAL.receipt_context_sha256 = journal[
            "receipt_context_sha256"
        ]
        transaction._ADAPTER_LOCAL.forward_created_unix_ns = 11
        try:
            result = transaction._invoke_root_locked(
                "version_forward",
                "v008",
                env["INVALIDITY_PATH"],
                env["SEAL_PATH"],
                require_receipt=False,
            )
            observed = transaction._invoke_root_locked(
                "status", require_receipt=False
            )
        finally:
            del transaction._ADAPTER_LOCAL.receipt_context_sha256
            del transaction._ADAPTER_LOCAL.forward_created_unix_ns

    assert observed == result
    assert result["active_attempt"] == "v008"
    assert result["ledger_event_count"] == source_event_count + 1
    assert result[transaction.ADAPTER_STATE_KEY]["root_program_sha256"] == hashlib.sha256(
        env["PROGRAM_PATH"].read_bytes()
    ).hexdigest()
    assert not os.path.lexists(env["TRANSACTION_PATH"])


def test_every_v008_controller_caller_routes_through_receipt_bound_adapter() -> None:
    required = {
        "study_common.py": 'CONTROLLER_PATH = ATTEMPT_ROOT / "version_forward_transaction.py"',
        "launcher.py": 'PROGRAM_SCRIPT = ATTEMPT_ROOT / "version_forward_transaction.py"',
        "terminal_workflow.py": 'CONTROLLER = ATTEMPT_ROOT / "version_forward_transaction.py"',
        "workflow.py": 'CONTROLLER_ADAPTER_PATH = ATTEMPT_ROOT / "version_forward_transaction.py"',
        "analysis.py": 'program = attempt_root / "version_forward_transaction.py"',
        "inherited_authorization.py": "CONTROLLER_ADAPTER_PATH = TRANSACTION_SOURCE_PATH",
    }
    for relative, fragment in required.items():
        source = (ATTEMPT / relative).read_text(encoding="utf-8")
        assert fragment in source, f"{relative} bypasses the durable v008 adapter"


def test_nonzero_pre_state_fails_before_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    state = dict(env["state"], selection_outcome_episodes=1)
    _write_json(env["STATE_PATH"], state)
    called = False

    def controller() -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(transaction, "_invoke_controller", controller)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="verified v006 selection boundary",
    ):
        transaction.execute_version_forward()
    assert called is False and not env["RECEIPT_PATH"].exists()


def test_stale_snapshot_during_preverification_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)

    def stale_preverify() -> dict[str, Any]:
        state = json.loads(env["STATE_PATH"].read_text())
        state["updated_unix_ns"] = 2
        _write_json(env["STATE_PATH"], state)
        return {"passed": True}

    monkeypatch.setattr(transaction, "_preverify", stale_preverify)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: pytest.fail("controller ran"))
    with pytest.raises(transaction.VersionForwardTransactionError, match="snapshot changed"):
        transaction.execute_version_forward()
    assert not env["RECEIPT_PATH"].exists()


def test_controller_failure_preserves_prestate_and_has_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    monkeypatch.setattr(
        transaction,
        "_invoke_controller",
        lambda: (_ for _ in ()).throw(RuntimeError("controller failure")),
    )
    with pytest.raises(RuntimeError, match="controller failure"):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger
    assert not env["RECEIPT_PATH"].exists()


def test_postcommit_pre_receipt_crash_resumes_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )
    original_postverify = transaction._postverify
    crashed = False

    def crash_once() -> dict[str, Any]:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("postverify crash")
        return original_postverify()

    monkeypatch.setattr(
        transaction,
        "_postverify",
        crash_once,
    )
    with pytest.raises(RuntimeError, match="postverify crash"):
        transaction.execute_version_forward()
    assert json.loads(env["STATE_PATH"].read_text())["active_attempt"] == "v008"
    assert not env["RECEIPT_PATH"].exists()
    assert env["RECEIPT_JOURNAL_PATH"].is_file()
    result = transaction.execute_version_forward()
    assert result["passed"] is True
    assert env["RECEIPT_PATH"].is_file()
    assert not env["RECEIPT_JOURNAL_PATH"].exists()


@pytest.mark.parametrize(
    "boundary",
    [
        "receipt_journal_staging_write",
        "receipt_journal_staging_fsync",
        "receipt_journal_link",
        "receipt_journal_link_parent_fsync",
        "receipt_journal_staging_unlink",
        "receipt_journal_staging_unlink_parent_fsync",
        "receipt_journal_publication_parent_fsync",
        "pending_staging_write",
        "pending_staging_fsync",
        "pending_rename",
        "pending_parent_fsync",
        "ledger_suffix_write",
        "ledger_suffix_fsync",
        "state_staging_write",
        "state_staging_fsync",
        "state_rename",
        "state_parent_fsync",
        "recovery_state_final_inode_fsync",
        "recovery_state_publication_parent_fsync",
        "pending_unlink",
        "pending_unlink_parent_fsync",
        "receipt_staging_write",
        "receipt_staging_fsync",
        "receipt_link",
        "receipt_link_parent_fsync",
        "receipt_staging_unlink",
        "receipt_staging_unlink_parent_fsync",
        "receipt_publication_recovery_parent_fsync",
        "receipt_journal_unlink",
        "receipt_journal_unlink_parent_fsync",
    ],
)
def test_receipt_journal_recovers_every_publication_boundary_bit_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )
    fired = False

    def crash(name: str) -> None:
        nonlocal fired
        if name == boundary and not fired:
            fired = True
            raise OSError(f"synthetic receipt-journal crash at {name}")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="synthetic receipt-journal crash"):
        transaction.execute_version_forward()
    assert fired is True

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    result = transaction.execute_version_forward()
    assert result["passed"] is True
    receipt = json.loads(env["RECEIPT_PATH"].read_text(encoding="utf-8"))
    assert env["RECEIPT_PATH"].stat().st_nlink == 1
    assert hashlib.sha256(transaction._pretty_bytes(receipt)).hexdigest() == result[
        "receipt_sha256"
    ]
    events = [
        json.loads(line) for line in env["LEDGER_PATH"].read_bytes().splitlines()
    ]
    forward = [
        event
        for event in events
        if event.get("event") == "zero_confirmation_outcome_version_forward"
    ]
    assert len(forward) == 6
    assert (forward[0]["old_attempt"], forward[0]["new_attempt"]) == (
        "v001",
        "v002",
    )
    assert (forward[1]["old_attempt"], forward[1]["new_attempt"]) == (
        "v002",
        "v003",
    )
    assert (forward[2]["old_attempt"], forward[2]["new_attempt"]) == (
        "v003",
        "v004",
    )
    assert (forward[3]["old_attempt"], forward[3]["new_attempt"]) == (
        "v004",
        "v005",
    )
    assert (forward[4]["old_attempt"], forward[4]["new_attempt"]) == (
        "v005",
        "v006",
    )
    assert (forward[5]["old_attempt"], forward[5]["new_attempt"]) == (
        "v006",
        "v008",
    )
    assert (
        forward[-1][transaction.RECEIPT_CONTEXT_EVENT_FIELD]
        == receipt["receipt_context_sha256"]
    )
    assert not any(
        os.path.lexists(path)
        for path in (
            env["TRANSACTION_PATH"],
            transaction.PENDING_STAGING_PATH,
            transaction.STATE_STAGING_PATH,
            transaction.RECEIPT_JOURNAL_PATH,
            transaction.RECEIPT_JOURNAL_STAGING_PATH,
            transaction.RECEIPT_STAGING_PATH,
        )
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "recovery_receipt_journal_inode_fsync",
        "recovery_receipt_journal_link",
        "recovery_receipt_journal_link_parent_fsync",
        "recovery_receipt_journal_staging_unlink",
        "recovery_receipt_journal_staging_unlink_parent_fsync",
        "recovery_receipt_journal_publication_parent_fsync",
    ],
)
def test_receipt_journal_recovers_its_own_staging_publication_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def first_crash(name: str) -> None:
        if name == "receipt_journal_staging_fsync":
            raise OSError("initial journal staging crash")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", first_crash)
    with pytest.raises(OSError, match="initial journal staging crash"):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_JOURNAL_STAGING_PATH.is_file()

    fired = False

    def recovery_crash(name: str) -> None:
        nonlocal fired
        if name == boundary and not fired:
            fired = True
            raise OSError(f"journal recovery crash at {name}")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", recovery_crash)
    with pytest.raises(OSError, match="journal recovery crash"):
        transaction.execute_version_forward()
    assert fired is True
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    assert transaction.execute_version_forward()["passed"] is True
    assert not transaction.RECEIPT_JOURNAL_PATH.exists()
    assert not transaction.RECEIPT_JOURNAL_STAGING_PATH.exists()


def test_unfsynced_journal_write_is_fsynced_before_recovery_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def write_crash(name: str) -> None:
        if name == "receipt_journal_staging_write":
            raise OSError("journal inode not yet fsynced")

    state_before = env["STATE_PATH"].read_bytes()
    ledger_before = env["LEDGER_PATH"].read_bytes()
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", write_crash)
    with pytest.raises(OSError, match="journal inode not yet fsynced"):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_JOURNAL_STAGING_PATH.is_file()

    def fsync_crash(name: str) -> None:
        if name == "recovery_receipt_journal_inode_fsync":
            raise OSError("journal recovery inode fsynced")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", fsync_crash)
    with pytest.raises(OSError, match="journal recovery inode fsynced"):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_JOURNAL_STAGING_PATH.is_file()
    assert env["STATE_PATH"].read_bytes() == state_before
    assert env["LEDGER_PATH"].read_bytes() == ledger_before

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    assert transaction.execute_version_forward()["passed"] is True


def test_unfsynced_receipt_write_is_fsynced_before_recovery_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def write_crash(name: str) -> None:
        if name == "receipt_staging_write":
            raise OSError("receipt inode not yet fsynced")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", write_crash)
    with pytest.raises(OSError, match="receipt inode not yet fsynced"):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_STAGING_PATH.is_file()
    assert not env["RECEIPT_PATH"].exists()
    assert transaction.RECEIPT_JOURNAL_PATH.is_file()
    post_state = env["STATE_PATH"].read_bytes()
    post_ledger = env["LEDGER_PATH"].read_bytes()

    def fsync_crash(name: str) -> None:
        if name == "receipt_recovery_inode_fsync":
            raise OSError("receipt recovery inode fsynced")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", fsync_crash)
    with pytest.raises(OSError, match="receipt recovery inode fsynced"):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_STAGING_PATH.is_file()
    assert not env["RECEIPT_PATH"].exists()
    assert env["STATE_PATH"].read_bytes() == post_state
    assert env["LEDGER_PATH"].read_bytes() == post_ledger

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    assert transaction.execute_version_forward()["passed"] is True


@pytest.mark.parametrize(
    ("write_boundary", "recovery_boundary", "label"),
    [
        (
            "pending_staging_write",
            "recovery_pending_inode_fsync",
            "pending",
        ),
        (
            "state_staging_write",
            "recovery_state_inode_fsync",
            "state",
        ),
    ],
)
def test_unfsynced_controller_staging_is_fsynced_before_recovery_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_boundary: str,
    recovery_boundary: str,
    label: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def write_crash(name: str) -> None:
        if name == write_boundary:
            raise OSError(f"{label} inode not yet fsynced")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", write_crash)
    with pytest.raises(OSError, match=f"{label} inode not yet fsynced"):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_JOURNAL_PATH.is_file()
    if label == "pending":
        assert transaction.PENDING_STAGING_PATH.is_file()
        assert not env["TRANSACTION_PATH"].exists()
    else:
        assert transaction.STATE_STAGING_PATH.is_file()
        assert env["TRANSACTION_PATH"].is_file()
    state_before = env["STATE_PATH"].read_bytes()
    ledger_before = env["LEDGER_PATH"].read_bytes()

    def fsync_crash(name: str) -> None:
        if name == recovery_boundary:
            raise OSError(f"{label} recovery inode fsynced")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", fsync_crash)
    with pytest.raises(OSError, match=f"{label} recovery inode fsynced"):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == state_before
    assert env["LEDGER_PATH"].read_bytes() == ledger_before

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    assert transaction.execute_version_forward()["passed"] is True


@pytest.mark.parametrize(
    ("rename_boundary", "recovery_boundary", "label"),
    [
        (
            "pending_rename",
            "recovery_pending_final_inode_fsync",
            "pending inode",
        ),
        (
            "pending_rename",
            "recovery_pending_publication_parent_fsync",
            "pending",
        ),
        (
            "state_rename",
            "recovery_state_publication_parent_fsync",
            "state",
        ),
    ],
)
def test_unfsynced_controller_rename_directory_is_barriered_before_recovery_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rename_boundary: str,
    recovery_boundary: str,
    label: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def rename_crash(name: str) -> None:
        if name == rename_boundary:
            raise OSError(f"{label} rename not yet directory-fsynced")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", rename_crash)
    with pytest.raises(
        OSError, match=f"{label} rename not yet directory-fsynced"
    ):
        transaction.execute_version_forward()
    state_before = env["STATE_PATH"].read_bytes()
    ledger_before = env["LEDGER_PATH"].read_bytes()

    def recovery_crash(name: str) -> None:
        if name == recovery_boundary:
            raise OSError(f"{label} recovery directory fsynced")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", recovery_crash)
    with pytest.raises(OSError, match=f"{label} recovery directory fsynced"):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == state_before
    assert env["LEDGER_PATH"].read_bytes() == ledger_before

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    assert transaction.execute_version_forward()["passed"] is True


def test_receipt_journal_rejects_rehashed_prediction_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def crash(name: str) -> None:
        if name == "receipt_journal_publication_parent_fsync":
            raise OSError("journal ready")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="journal ready"):
        transaction.execute_version_forward()
    journal = json.loads(
        transaction.RECEIPT_JOURNAL_PATH.read_text(encoding="utf-8")
    )
    journal["predicted_post_snapshot"]["state"]["sha256"] = "f" * 64
    journal["transaction_sha256"] = transaction._receipt_journal_digest(journal)
    _write_json(transaction.RECEIPT_JOURNAL_PATH, journal)
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="post-snapshot prediction drift",
    ):
        transaction.execute_version_forward()


def test_receipt_journal_rejects_stale_context_digest_before_state_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    with transaction._adapter_lock():
        journal = transaction._build_receipt_journal_locked(
            prospective_created_unix_ns=10
        )
        transaction._publish_receipt_journal(journal)
    changed = json.loads(
        transaction.RECEIPT_JOURNAL_PATH.read_text(encoding="utf-8")
    )
    changed["receipt_context"]["pre_verifiers"]["producer"][
        "state_sha256"
    ] = "f" * 64
    # Deliberately rehash only the outer journal.  The independently committed
    # receipt-context digest must still reject this otherwise valid shape.
    changed["transaction_sha256"] = transaction._receipt_journal_digest(changed)
    _write_json(transaction.RECEIPT_JOURNAL_PATH, changed)
    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="receipt context identity/timestamp drift",
    ):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger


def test_final_receipt_rejects_stale_context_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )
    assert transaction.execute_version_forward()["passed"] is True
    receipt = json.loads(env["RECEIPT_PATH"].read_text(encoding="utf-8"))
    receipt["receipt_context"]["pre_verifiers"]["producer"][
        "state_sha256"
    ] = "f" * 64
    _write_json(env["RECEIPT_PATH"], receipt)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="receipt header/type drift",
    ):
        transaction.verify_transaction_receipt()


def test_receipt_journal_atomic_link_never_overwrites_concurrent_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    publish_ready = threading.Event()
    competitor_created = threading.Event()
    errors: list[BaseException] = []
    competing = b"concurrent competing journal\n"

    def competitor() -> None:
        try:
            assert publish_ready.wait(timeout=2)
            descriptor = os.open(
                transaction.RECEIPT_JOURNAL_PATH,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                assert os.write(descriptor, competing) == len(competing)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory = os.open(transaction.RECEIPT_JOURNAL_PATH.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            competitor_created.set()
        except BaseException as error:
            errors.append(error)
            competitor_created.set()

    worker = threading.Thread(target=competitor)
    worker.start()

    def race(name: str) -> None:
        if name == "receipt_journal_staging_fsync":
            publish_ready.set()
            assert competitor_created.wait(timeout=2)

    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", race)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: pytest.fail("controller ran")
    )
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="competing version-forward receipt journal",
    ):
        transaction.execute_version_forward()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors == []
    assert transaction.RECEIPT_JOURNAL_PATH.read_bytes() == competing
    assert transaction.RECEIPT_JOURNAL_STAGING_PATH.is_file()
    assert transaction.RECEIPT_JOURNAL_STAGING_PATH.stat().st_nlink == 1
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger
    assert not env["TRANSACTION_PATH"].exists()
    assert not env["RECEIPT_PATH"].exists()


def test_receipt_journal_cleanup_preserves_swapped_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )
    original_verify = transaction.verify_transaction_receipt
    competing = b"competing cleanup journal\n"
    swapped = False

    def swap_before_cleanup(
        *, _allow_receipt_journal: bool = False
    ) -> dict[str, Any]:
        nonlocal swapped
        result = original_verify(
            _allow_receipt_journal=_allow_receipt_journal
        )
        if _allow_receipt_journal and not swapped:
            swapped = True
            transaction.RECEIPT_JOURNAL_PATH.unlink()
            transaction.RECEIPT_JOURNAL_PATH.write_bytes(competing)
        return result

    monkeypatch.setattr(
        transaction, "verify_transaction_receipt", swap_before_cleanup
    )
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="cleanup target identity drift",
    ):
        transaction.execute_version_forward()
    assert swapped is True
    assert transaction.RECEIPT_JOURNAL_PATH.read_bytes() == competing
    assert env["RECEIPT_PATH"].is_file()
    assert env["RECEIPT_PATH"].stat().st_nlink == 1


def test_live_controller_return_mismatch_is_rejected_after_exact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)

    def mismatched_return() -> dict[str, Any]:
        returned = json.loads(json.dumps(_journal_bound_controller(env)))
        returned["return_only_tamper"] = True
        return returned

    monkeypatch.setattr(transaction, "_invoke_controller", mismatched_return)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="live controller return differs",
    ):
        transaction.execute_version_forward()
    journal = json.loads(
        transaction.RECEIPT_JOURNAL_PATH.read_text(encoding="utf-8")
    )
    assert json.loads(env["STATE_PATH"].read_text(encoding="utf-8")) == journal[
        "predicted_post_snapshot"
    ]["state"]["object"]
    assert not env["RECEIPT_PATH"].exists()


def test_postverify_result_mismatch_is_rejected_before_receipt_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    original_postverify = transaction._postverify

    def mismatched_postverify() -> dict[str, Any]:
        value = json.loads(json.dumps(original_postverify()))
        value["standalone"]["output_paths_created"] = 1
        return value

    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )
    monkeypatch.setattr(transaction, "_postverify", mismatched_postverify)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="post-forward verifier result differs",
    ):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_JOURNAL_PATH.is_file()
    assert not env["RECEIPT_PATH"].exists()
    assert not transaction.RECEIPT_STAGING_PATH.exists()


def test_postverify_state_mutation_is_rejected_after_matching_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    original_postverify = transaction._postverify

    def mutating_postverify() -> dict[str, Any]:
        result = original_postverify()
        state = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
        state["mutated_during_postverify"] = True
        _write_json(env["STATE_PATH"], state)
        return result

    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )
    monkeypatch.setattr(transaction, "_postverify", mutating_postverify)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="post-forward state changed during receipt verification",
    ):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_JOURNAL_PATH.is_file()
    assert not env["RECEIPT_PATH"].exists()


def test_final_only_receipt_recovery_closes_before_postverify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def crash(name: str) -> None:
        if name == "receipt_staging_unlink_parent_fsync":
            raise OSError("final-only receipt")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="final-only receipt"):
        transaction.execute_version_forward()
    assert env["RECEIPT_PATH"].is_file()
    assert env["RECEIPT_PATH"].stat().st_nlink == 1
    assert not transaction.RECEIPT_STAGING_PATH.exists()
    assert transaction.RECEIPT_JOURNAL_PATH.is_file()

    original_postverify = transaction._postverify

    def inspect_closed_receipt() -> dict[str, Any]:
        assert env["RECEIPT_PATH"].stat().st_nlink == 1
        assert not transaction.RECEIPT_STAGING_PATH.exists()
        return original_postverify()

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    monkeypatch.setattr(transaction, "_postverify", inspect_closed_receipt)
    assert transaction.execute_version_forward()["passed"] is True


def test_receipt_staging_without_journal_fails_before_any_state_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    transaction.RECEIPT_STAGING_PATH.write_bytes(b"orphan receipt staging\n")
    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="receipt staging exists without its durable context journal",
    ):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger


def test_lost_journal_with_postcommit_receipt_staging_is_not_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def crash(name: str) -> None:
        if name == "receipt_staging_fsync":
            raise OSError("receipt staging durable")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="receipt staging durable"):
        transaction.execute_version_forward()
    assert transaction.RECEIPT_STAGING_PATH.is_file()
    assert not env["RECEIPT_PATH"].exists()
    transaction.RECEIPT_JOURNAL_PATH.unlink()
    state_before = env["STATE_PATH"].read_bytes()
    ledger_before = env["LEDGER_PATH"].read_bytes()
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="receipt staging exists without its durable context journal",
    ):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == state_before
    assert env["LEDGER_PATH"].read_bytes() == ledger_before


def test_receipt_recovery_rejects_hidden_third_hard_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def crash(name: str) -> None:
        if name == "receipt_link_parent_fsync":
            raise OSError("receipt pair durable")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="receipt pair durable"):
        transaction.execute_version_forward()
    hidden = tmp_path / "hidden-receipt-alias"
    os.link(env["RECEIPT_PATH"], hidden)
    assert env["RECEIPT_PATH"].stat().st_nlink == 3
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="linked, aliased, or non-regular",
    ):
        transaction.execute_version_forward()


def test_journal_staging_recovery_rejects_hidden_hard_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)

    def crash(name: str) -> None:
        if name == "receipt_journal_staging_fsync":
            raise OSError("journal staging durable")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="journal staging durable"):
        transaction.execute_version_forward()
    hidden = tmp_path / "hidden-journal-alias"
    os.link(transaction.RECEIPT_JOURNAL_STAGING_PATH, hidden)
    assert transaction.RECEIPT_JOURNAL_STAGING_PATH.stat().st_nlink == 2
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="hidden alias",
    ):
        transaction.execute_version_forward()
    assert json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))[
        "active_attempt"
    ] == "v006"


def test_receipt_journal_rejects_state_matching_neither_pre_nor_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    with transaction._adapter_lock():
        journal = transaction._build_receipt_journal_locked(
            prospective_created_unix_ns=10
        )
        transaction._publish_receipt_journal(journal)
    neither = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    neither["uncommitted_state_drift"] = True
    _write_json(env["STATE_PATH"], neither)
    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="neither its exact pre-state nor exact post-state",
    ):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger


def test_resume_rechecks_live_preverification_before_reconciling_persisted_initial_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    original_preverify = transaction._preverify
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def crash_with_durable_pending(name: str) -> None:
        if name == "pending_parent_fsync":
            raise OSError("initial pending durable")

    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash_with_durable_pending)
    with pytest.raises(OSError, match="initial pending durable"):
        transaction.execute_version_forward()
    assert env["TRANSACTION_PATH"].is_file()
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger

    def coherently_reprojected_preverify() -> dict[str, Any]:
        value = json.loads(json.dumps(original_preverify()))
        value["independent_partitions"]["source_path_count"] += 1
        return value

    # Recovery must reject a changed live implementation before completing the
    # already-durable pending transaction.
    monkeypatch.setattr(transaction, "_preverify", coherently_reprojected_preverify)
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="live preverification differs",
    ):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger
    assert env["TRANSACTION_PATH"].is_file()


def test_receipt_journal_rejects_coherently_wrong_sealed_count_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    original_preverify = transaction._preverify

    def wrong_count() -> dict[str, Any]:
        value = json.loads(json.dumps(original_preverify()))
        value["standalone"]["sealed_file_count"] += 1000
        return value

    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    monkeypatch.setattr(transaction, "_preverify", wrong_count)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="standalone pre-forward result drift",
    ):
        transaction.execute_version_forward()
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger
    assert not transaction.RECEIPT_JOURNAL_PATH.exists()
    assert not transaction.RECEIPT_JOURNAL_STAGING_PATH.exists()


@pytest.mark.parametrize("pending_location", ["final", "staging"])
def test_status_rejects_initial_forward_pending_without_journal_before_any_recovery_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending_location: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    with transaction._adapter_lock():
        journal = transaction._build_receipt_journal_locked(
            prospective_created_unix_ns=10
        )
        transaction._publish_receipt_journal(journal)
        pending = journal["predicted_controller_transaction"]
        if pending_location == "final":
            transaction._publish_pending(pending)
            pending_path = env["TRANSACTION_PATH"]
        else:
            transaction._write_staging_file(
                transaction.PENDING_STAGING_PATH,
                transaction._pretty_bytes(pending),
                prefix="setup_initial_pending",
            )
            pending_path = transaction.PENDING_STAGING_PATH
        transaction.RECEIPT_JOURNAL_PATH.unlink()

    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    before_pending = pending_path.read_bytes()
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="lacks its sole durable receipt journal",
    ):
        transaction._invoke_root_locked("status", require_receipt=False)
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger
    assert pending_path.read_bytes() == before_pending


def test_receipt_publication_parent_fsync_survives_double_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def first_crash(name: str) -> None:
        if name == "receipt_link_parent_fsync":
            raise OSError("receipt aliases durable")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", first_crash)
    with pytest.raises(OSError, match="receipt aliases durable"):
        transaction.execute_version_forward()
    assert env["RECEIPT_PATH"].is_file()
    assert transaction.RECEIPT_STAGING_PATH.is_file()

    def second_crash(name: str) -> None:
        if name == "receipt_publication_recovery_parent_fsync":
            raise OSError("recovery parent fsync interrupted")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", second_crash)
    with pytest.raises(OSError, match="recovery parent fsync interrupted"):
        transaction.execute_version_forward()
    assert env["RECEIPT_PATH"].is_file()
    assert env["RECEIPT_PATH"].stat().st_nlink == 1
    assert not transaction.RECEIPT_STAGING_PATH.exists()
    assert transaction.RECEIPT_JOURNAL_PATH.is_file()

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    assert transaction.execute_version_forward()["passed"] is True
    assert env["RECEIPT_PATH"].stat().st_nlink == 1
    assert not transaction.RECEIPT_JOURNAL_PATH.exists()


def test_receipt_journal_rejects_competing_receipt_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)

    def crash(name: str) -> None:
        if name == "receipt_journal_publication_parent_fsync":
            raise OSError("journal ready")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="journal ready"):
        transaction.execute_version_forward()
    _write_json(env["RECEIPT_PATH"], {"competing": True})
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="receipt exists before",
    ):
        transaction.execute_version_forward()


def test_postcommit_lost_receipt_journal_is_never_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def crash(name: str) -> None:
        if name == "state_parent_fsync":
            raise OSError("post-state crash")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="post-state crash"):
        transaction.execute_version_forward()
    transaction.RECEIPT_JOURNAL_PATH.unlink()
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="without a receipt context journal",
    ):
        transaction.execute_version_forward()


def test_receipt_recovery_rejects_same_bytes_on_competing_staging_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _journal_bound_controller(env)
    )

    def crash(name: str) -> None:
        if name == "receipt_link_parent_fsync":
            raise OSError("receipt linked")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match="receipt linked"):
        transaction.execute_version_forward()
    expected = transaction.RECEIPT_STAGING_PATH.read_bytes()
    transaction.RECEIPT_STAGING_PATH.unlink()
    transaction.RECEIPT_STAGING_PATH.write_bytes(expected)
    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="competing aliases",
    ):
        transaction.execute_version_forward()


def test_shared_lock_blocks_concurrent_controller_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    verifier_entered = threading.Event()
    release_verifier = threading.Event()
    writer_acquired = threading.Event()
    errors: list[BaseException] = []
    original_preverify = transaction._preverify

    def preverify() -> dict[str, Any]:
        verifier_entered.set()
        assert release_verifier.wait(timeout=2)
        return original_preverify()

    def writer() -> None:
        try:
            descriptor = os.open(env["LOCK_PATH"], os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                writer_acquired.set()
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
        except BaseException as error:
            errors.append(error)

    def run_transaction() -> None:
        try:
            transaction.execute_version_forward()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(transaction, "_preverify", preverify)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    thread = threading.Thread(target=writer)
    runner = threading.Thread(target=run_transaction)
    runner.start()
    assert verifier_entered.wait(timeout=2)
    thread.start()
    assert not writer_acquired.wait(timeout=0.05)
    release_verifier.set()
    runner.join(timeout=4)
    thread.join(timeout=4)
    assert not runner.is_alive() and not thread.is_alive()
    assert errors == []
    assert writer_acquired.is_set() and env["RECEIPT_PATH"].is_file()


def test_receipt_tamper_is_rejected_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    transaction.execute_version_forward()
    receipt = json.loads(env["RECEIPT_PATH"].read_text(encoding="utf-8"))
    receipt["post_snapshot"]["ledger"]["head_sha256"] = "0" * 64
    _write_json(env["RECEIPT_PATH"], receipt)
    with pytest.raises(transaction.VersionForwardTransactionError):
        transaction.verify_transaction_receipt()


def test_verified_receipt_remains_historically_checkable_after_later_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    result = transaction.execute_version_forward()
    intended = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    intended["selection_outcome_episodes"] += 1
    intended["updated_unix_ns"] = 2
    with transaction._adapter_lock():
        transaction._durable_commit_state_with_events(
            intended,
            [
                {
                    "created_unix_ns": 2,
                    "event": "outcome_counts_updated",
                    "attempt": "v008",
                    "fields": {
                        "selection_outcome_episodes": intended[
                            "selection_outcome_episodes"
                        ]
                    },
                }
            ],
        )
    assert transaction.verify_transaction_receipt()["receipt_sha256"] == result[
        "receipt_sha256"
    ]


def test_adapter_receipt_rejects_current_direct_root_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    transaction.execute_version_forward()
    state = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    event = {
        "created_unix_ns": 2,
        "event": "direct_root_cli_mutation",
        "attempt": "v008",
        "seq": state["ledger_event_count"] + 1,
        "prev_sha256": state["ledger_head_sha256"],
    }
    event["record_sha256"] = hashlib.sha256(_canonical(event)).hexdigest()
    with env["LEDGER_PATH"].open("ab") as handle:
        handle.write(_canonical(event) + b"\n")
    state["ledger_event_count"] = event["seq"]
    state["ledger_head_sha256"] = event["record_sha256"]
    _write_json(env["STATE_PATH"], state)

    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="authorization drift",
    ):
        transaction.verify_transaction_receipt()


def test_coherently_rehashed_direct_root_append_after_descendant_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _commit_controller(env)
    )
    transaction.execute_version_forward()
    _commit_fit_count_descendant(env)

    state = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    event = {
        "created_unix_ns": 3,
        "event": "direct_root_cli_mutation",
        "attempt": "v008",
        "seq": state["ledger_event_count"] + 1,
        "prev_sha256": state["ledger_head_sha256"],
    }
    event["record_sha256"] = hashlib.sha256(_canonical(event)).hexdigest()
    with env["LEDGER_PATH"].open("ab") as handle:
        handle.write(_canonical(event) + b"\n")
    state["ledger_event_count"] = event["seq"]
    state["ledger_head_sha256"] = event["record_sha256"]
    state["next_action"] = "coherently forged direct-root state"
    marker = state[transaction.ADAPTER_STATE_KEY]
    marker.update(
        {
            "ledger_event_count": event["seq"],
            "ledger_head_sha256": event["record_sha256"],
            "operation_sha256": "f" * 64,
        }
    )
    marker_core = json.loads(json.dumps(marker))
    marker_core.pop("state_binding_sha256")
    state_without_marker = json.loads(json.dumps(state))
    state_without_marker.pop(transaction.ADAPTER_STATE_KEY)
    marker["state_binding_sha256"] = transaction._canonical_sha256(
        {
            "marker_without_state_binding": marker_core,
            "state_without_marker": state_without_marker,
        }
    )
    _write_json(env["STATE_PATH"], state)

    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="descendant adapter transaction grouping gap",
    ):
        transaction.verify_transaction_receipt()
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="descendant adapter transaction grouping gap",
    ):
        transaction.status()


def test_descendant_target_state_rehash_cannot_escape_proposal_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _commit_controller(env)
    )
    transaction.execute_version_forward()
    _commit_fit_count_descendant(env)

    receipt = json.loads(env["RECEIPT_PATH"].read_text(encoding="utf-8"))
    post_count = receipt["post_snapshot"]["ledger"]["event_count"]
    lines = env["LEDGER_PATH"].read_bytes().splitlines(keepends=True)
    record = json.loads(lines[post_count])
    envelope = record[transaction.ADAPTER_TRANSACTION_ENVELOPE_FIELD]
    base_state = envelope["base_state"]
    state = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    state["next_action"] = "coherently rehashed altered target"
    state_without_marker = json.loads(json.dumps(state))
    state_without_marker.pop(transaction.ADAPTER_STATE_KEY)
    operation_sha256 = transaction._operation_sha256(
        base_state=base_state,
        base_ledger_sha256=hashlib.sha256(
            b"".join(lines[:post_count])
        ).hexdigest(),
        expected_suffix_sha256=hashlib.sha256(
            b"".join(lines[post_count:])
        ).hexdigest(),
        intended_without_marker=state_without_marker,
    )
    state[transaction.ADAPTER_STATE_KEY] = transaction._adapter_marker(
        event_count=state["ledger_event_count"],
        head_sha256=state["ledger_head_sha256"],
        operation_sha256=operation_sha256,
        state_without_marker=state_without_marker,
        root_record=transaction._source_record(env["PROGRAM_PATH"]),
        adapter_record=transaction._source_record(
            transaction.TRANSACTION_SOURCE_PATH
        ),
    )
    _write_json(env["STATE_PATH"], state)

    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="proposed-state commitment drift",
    ):
        transaction.verify_transaction_receipt()


def test_descendant_multi_event_transaction_group_replays_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _commit_controller(env)
    )
    transaction.execute_version_forward()
    _commit_ordinary_checkpoint(env, created_unix_ns=2)

    source = env["ATTEMPT_ROOT"] / "selection/selection_ledger.json"
    contract = env["ATTEMPT_ROOT"] / "verifier_contract_no_candidate.json"
    _write_json(
        source,
        {
            "attempt": "v008",
            "status": "selection_complete_no_selected_head_refit",
            "candidate_count": 24,
            "eligible_count": 0,
            "selected_candidate_id": None,
            "selected_candidate_index": None,
            "selected_head_refit_after_selection": False,
            "prior_confirmation_outcome_episodes_used": 0,
        },
    )
    source_relative = transaction._relative(source)
    _write_json(
        contract,
        {
            "schema_version": 1,
            "attempt": "v008",
            "attempt_root": transaction._relative(env["ATTEMPT_ROOT"]),
            "mode": "no_candidate",
            "paths": {"selection_ledger": source_relative},
        },
    )
    contract_relative = transaction._relative(contract)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    contract_sha256 = hashlib.sha256(contract.read_bytes()).hexdigest()
    skipped = list(
        transaction.CONTROLLER_STATE_MACHINE[
            transaction.CONTROLLER_STATE_MACHINE.index("CANDIDATE_SELECTION") + 1:
            transaction.CONTROLLER_STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
        ]
    )
    staged = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    staged["early_scientific_failure"] = {
        "mode": "no_candidate",
        "status": "awaiting_independent_verification",
        "trigger_state": "CANDIDATE_SELECTION",
        "trigger_evidence_path": source_relative,
        "trigger_evidence_sha256": source_sha256,
        "verifier_contract_path": contract_relative,
        "verifier_contract_sha256": contract_sha256,
        "skipped_states": skipped,
        "staged_unix_ns": 6,
        "required_terminal_label": "domain_robust_gate_failed",
        "required_process_valid": True,
    }
    staged["skipped_states"] = [
        {
            "state": name,
            "mode": "no_candidate",
            "reason": "preregistered_process_valid_early_scientific_failure",
            "trigger_evidence_path": source_relative,
            "trigger_evidence_sha256": source_sha256,
            "recorded_unix_ns": 6,
        }
        for name in skipped
    ]
    staged["current_state"] = "INDEPENDENT_VERIFICATION"
    staged["next_action"] = (
        "run the standalone read-only verifier in the exact staged mode and "
        "capture audit/independent_verification.json before finalizing"
    )
    staged["updated_unix_ns"] = 6
    with transaction._adapter_lock():
        transaction._durable_commit_state_with_events(
            staged,
            [
                {
                    "event": "preregistered_early_scientific_failure_staged",
                    "attempt": "v008",
                    "mode": "no_candidate",
                    "trigger_state": "CANDIDATE_SELECTION",
                    "trigger_evidence_path": source_relative,
                    "trigger_evidence_sha256": source_sha256,
                    "verifier_contract_path": contract_relative,
                    "verifier_contract_sha256": contract_sha256,
                    "skipped_states": skipped,
                    "smoke_outcome_episodes": 0,
                    "confirmation_outcome_episodes_generated": 0,
                    "confirmation_outcome_episodes_executed": 0,
                    "confirmation_outcomes_opened_for_analysis": False,
                    "created_unix_ns": 6,
                }
            ],
        )

    audit = env["ATTEMPT_ROOT"] / "audit/independent_verification.json"
    decision_file = env["ATTEMPT_ROOT"] / "decision.json"
    staged_state = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    early = staged_state["early_scientific_failure"]
    _write_json(
        audit,
        {
            "schema_version": 1,
            "attempt": "v008",
            "mode": "no_candidate",
            "passed": True,
            "terminal_label": "domain_robust_gate_failed",
            "read_only_verifier": True,
            "checks": {"synthetic_independent_replay": True},
            "verifier_contract": {
                "path": early["verifier_contract_path"],
                "sha256": early["verifier_contract_sha256"],
            },
            "source_hashes": {
                "selection_ledger": early["trigger_evidence_sha256"]
            },
        },
    )
    audit_relative = transaction._relative(audit)
    decision_relative = decision_file.relative_to(
        env["REPOSITORY_ROOT"]
    ).as_posix()
    audit_sha256 = hashlib.sha256(audit.read_bytes()).hexdigest()
    _write_json(
        decision_file,
        {
            "schema_version": 1,
            "attempt": "v008",
            "passed": True,
            "checkpoint_state": "TERMINAL",
            "early_failure_mode": "no_candidate",
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
            "trigger_evidence_path": early["trigger_evidence_path"],
            "trigger_evidence_sha256": early["trigger_evidence_sha256"],
            "independent_verification_path": audit_relative,
            "independent_verification_sha256": audit_sha256,
            "smoke_outcome_episodes": 0,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
        },
    )
    decision_sha256 = hashlib.sha256(decision_file.read_bytes()).hexdigest()
    intended = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    audit_checkpoint = {
        "name": "v008_no_candidate_independent_verification_passed",
        "created_unix_ns": 7,
        "evidence_path": audit_relative,
        "evidence_sha256": audit_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_early_scientific_failure_checkpoint",
    }
    decision_checkpoint = {
        "name": "v008_no_candidate_terminal_decision_recorded",
        "created_unix_ns": 7,
        "evidence_path": decision_relative,
        "evidence_sha256": decision_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_early_scientific_failure_checkpoint",
    }
    intended["completed_states"].extend(("INDEPENDENT_VERIFICATION", "TERMINAL"))
    intended["verified_checkpoints"].extend(
        (audit_checkpoint, decision_checkpoint)
    )
    intended["last_verified_checkpoint"] = decision_checkpoint
    intended["current_state"] = "POST_TERMINAL_REPORTING"
    intended["terminal_label"] = "domain_robust_gate_failed"
    intended["process_valid"] = True
    intended["scientific_terminal"] = True
    intended["confirmation_terminal"] = False
    intended["terminal_basis"] = "preregistered_no_candidate"
    intended["terminal_decision_path"] = decision_checkpoint["evidence_path"]
    intended["terminal_decision_sha256"] = decision_checkpoint["evidence_sha256"]
    intended["early_scientific_failure"].update(
        {
            "status": "terminal_recorded",
            "independent_verification_path": audit_relative,
            "independent_verification_sha256": audit_sha256,
            "decision_path": decision_relative,
            "decision_sha256": decision_sha256,
            "terminal_recorded_unix_ns": 7,
        }
    )
    intended["next_action"] = (
        "write the concise terminal report, robustness map, audit, limitations, "
        "and next project-scoped task; then checkpoint POST_TERMINAL_REPORTING"
    )
    intended["updated_unix_ns"] = 7
    events = [
        {
            "created_unix_ns": 7,
            "event": "state_completed",
            "attempt": "v008",
            "completed_state": "INDEPENDENT_VERIFICATION",
            "next_state": "TERMINAL",
            "checkpoint_name": audit_checkpoint["name"],
            "evidence_path": audit_checkpoint["evidence_path"],
            "evidence_sha256": audit_checkpoint["evidence_sha256"],
            "early_failure_mode": "no_candidate",
        },
        {
            "created_unix_ns": 7,
            "event": "scientific_terminal_decision_recorded",
            "attempt": "v008",
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
            "confirmation_terminal": False,
            "early_failure_mode": "no_candidate",
            "decision_path": decision_checkpoint["evidence_path"],
            "decision_sha256": decision_checkpoint["evidence_sha256"],
        },
        {
            "created_unix_ns": 7,
            "event": "state_completed",
            "attempt": "v008",
            "completed_state": "TERMINAL",
            "next_state": "POST_TERMINAL_REPORTING",
            "checkpoint_name": decision_checkpoint["name"],
            "evidence_path": decision_checkpoint["evidence_path"],
            "evidence_sha256": decision_checkpoint["evidence_sha256"],
            "early_failure_mode": "no_candidate",
            "skipped_states": skipped,
        },
    ]
    with transaction._adapter_lock():
        transaction._durable_commit_state_with_events(intended, events)
    assert transaction.verify_transaction_receipt()["passed"] is True
    lines = env["LEDGER_PATH"].read_bytes().splitlines()
    envelopes = [
        json.loads(line)[transaction.ADAPTER_TRANSACTION_ENVELOPE_FIELD]
        for line in lines[-3:]
    ]
    assert [envelope["event_index"] for envelope in envelopes] == [0, 1, 2]
    assert all(envelope["event_count"] == 3 for envelope in envelopes)
    assert "base_state" in envelopes[0]
    assert all("base_state" not in envelope for envelope in envelopes[1:])
    standalone = transaction._load_module(
        "standalone_valid_early_terminal_semantics",
        ATTEMPT / "verify_version_forward.py",
    )
    standalone_proposal = json.loads(json.dumps(intended))
    standalone_proposal.pop(transaction.ADAPTER_STATE_KEY)
    assert standalone._verify_descendant_controller_transition(
        base_state=staged_state,
        proposed_state=standalone_proposal,
        controller_events=events,
        repo=env["REPOSITORY_ROOT"],
    ) is None


def test_coherently_enveloped_non_controller_transition_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _commit_controller(env)
    )
    transaction.execute_version_forward()
    intended = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    intended["next_action"] = "fabricated but adapter-enveloped mutation"
    intended["updated_unix_ns"] = 3
    event = {
        "event": "state_completed",
        "attempt": "v008",
        "created_unix_ns": 3,
    }
    state_before = env["STATE_PATH"].read_bytes()
    ledger_before = env["LEDGER_PATH"].read_bytes()
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="replayed state-completed controller transition drift",
    ):
        transaction._durable_commit_state_with_events(intended, [event])
    assert env["STATE_PATH"].read_bytes() == state_before
    assert env["LEDGER_PATH"].read_bytes() == ledger_before
    assert not os.path.lexists(env["TRANSACTION_PATH"])
    assert not os.path.lexists(transaction.PENDING_STAGING_PATH)
    assert not os.path.lexists(transaction.STATE_STAGING_PATH)

    standalone = transaction._load_module(
        "standalone_integrity_semantic_guards",
        ATTEMPT / "verify_version_forward.py",
    )
    ordinary_base = json.loads(
        env["STATE_PATH"].read_text(encoding="utf-8")
    )
    cross_attempt_evidence = (
        env["STUDY_ROOT"] / "attempts/v002/audit/cross_attempt_fit.json"
    )
    _write_json(
        cross_attempt_evidence,
        {
            "attempt": "v008",
            "checkpoint_state": "FIT_COHORTS",
            "passed": True,
        },
    )
    cross_path = transaction._relative(cross_attempt_evidence)
    cross_sha256 = hashlib.sha256(
        cross_attempt_evidence.read_bytes()
    ).hexdigest()
    cross_checkpoint = {
        "name": "v008_cross_attempt_fit",
        "created_unix_ns": 6,
        "evidence_path": cross_path,
        "evidence_sha256": cross_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_checkpoint",
    }
    cross_proposal = json.loads(json.dumps(ordinary_base))
    cross_proposal.pop(transaction.ADAPTER_STATE_KEY)
    cross_proposal["completed_states"] = ordinary_base["completed_states"] + [
        "FIT_COHORTS"
    ]
    cross_proposal["current_state"] = "FIT_LOCK"
    cross_proposal["last_verified_checkpoint"] = cross_checkpoint
    cross_proposal["verified_checkpoints"] = ordinary_base[
        "verified_checkpoints"
    ] + [cross_checkpoint]
    cross_proposal["next_action"] = "continue from cross-attempt evidence"
    cross_proposal["updated_unix_ns"] = 6
    cross_event = {
        "event": "state_completed",
        "attempt": "v008",
        "completed_state": "FIT_COHORTS",
        "next_state": "FIT_LOCK",
        "checkpoint_name": cross_checkpoint["name"],
        "evidence_path": cross_path,
        "evidence_sha256": cross_sha256,
        "created_unix_ns": 6,
    }
    with pytest.raises(
        standalone.VerificationError,
        match="state-completed event closed schema drift",
    ):
        standalone._verify_descendant_controller_transition(
            base_state=ordinary_base,
            proposed_state=cross_proposal,
            controller_events=[cross_event],
            repo=env["REPOSITORY_ROOT"],
        )
    integrity_base = json.loads(
        env["STATE_PATH"].read_text(encoding="utf-8")
    )
    integrity_base["early_scientific_failure"] = None
    integrity_base["current_state"] = "SEALED_ANALYSIS"
    integrity_source = (
        env["ATTEMPT_ROOT"] / "audit/analysis_execution_invalid.json"
    )
    _write_json(
        integrity_source,
        {
            "attempt": "v008",
            "checkpoint_state": "SEALED_ANALYSIS",
            "passed": False,
            "execution_invalid": True,
        },
    )
    integrity_path = transaction._relative(integrity_source)
    integrity_sha256 = hashlib.sha256(
        integrity_source.read_bytes()
    ).hexdigest()
    integrity_event = {
        "event": "state_completed",
        "attempt": "v008",
        "completed_state": "SEALED_ANALYSIS",
        "next_state": "INDEPENDENT_VERIFICATION",
        "checkpoint_name": "v008_analysis_execution_integrity_failure",
        "evidence_path": integrity_path,
        "evidence_sha256": integrity_sha256,
        "postconfirmation_integrity_failure": True,
        "integrity_source": "analysis_execution",
        "skipped_states": ["LATENCY_AND_RESOURCE_REPORTING"],
        "created_unix_ns": 8,
    }
    integrity_proposal = json.loads(json.dumps(integrity_base))
    integrity_proposal.pop(transaction.ADAPTER_STATE_KEY)
    integrity_checkpoint = {
        "name": integrity_event["checkpoint_name"],
        "created_unix_ns": 8,
        "evidence_path": integrity_path,
        "evidence_sha256": integrity_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    integrity_proposal["completed_states"] = integrity_base[
        "completed_states"
    ] + ["SEALED_ANALYSIS"]
    integrity_proposal["verified_checkpoints"] = integrity_base[
        "verified_checkpoints"
    ] + [integrity_checkpoint]
    integrity_proposal["last_verified_checkpoint"] = integrity_checkpoint
    integrity_proposal["current_state"] = "INDEPENDENT_VERIFICATION"
    integrity_proposal["postconfirmation_integrity_failure"] = {
        "status": "awaiting_independent_verification",
        "source": "analysis_execution",
        "trigger_state": "SEALED_ANALYSIS",
        "source_path": integrity_path,
        "source_sha256": integrity_sha256,
        "audit_hash_keys": [
            "analysis_execution_invalid", "analysis_execution", "analysis_failure"
        ],
        "skipped_states": ["LATENCY_AND_RESOURCE_REPORTING"],
        "staged_unix_ns": 8,
    }
    integrity_proposal["skipped_states"] = [
        {
            "state": "LATENCY_AND_RESOURCE_REPORTING",
            "source": "analysis_execution",
            "reason": "postconfirmation_integrity_failure_short_circuit",
            "source_path": integrity_path,
            "source_sha256": integrity_sha256,
            "recorded_unix_ns": 8,
        }
    ]
    integrity_proposal["next_action"] = (
        "run the read-only confirmation-mode independent verifier and capture the "
        "execution-invalid audit; do not retry or alter confirmation evidence"
    )
    integrity_proposal["updated_unix_ns"] = 8
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="integrity fixed-open proof drift",
    ):
        transaction._validate_replayed_controller_transition(
            base_state=integrity_base,
            proposed_state=integrity_proposal,
            events=[integrity_event],
        )
    with pytest.raises(
        standalone.VerificationError,
        match="integrity fixed-open proof drift",
    ):
        standalone._verify_descendant_controller_transition(
            base_state=integrity_base,
            proposed_state=integrity_proposal,
            controller_events=[integrity_event],
            repo=env["REPOSITORY_ROOT"],
        )

    # Even a fully coherent three-event envelope cannot invent an
    # execution-invalid result from an audit lacking the integrity predicate.
    invalid_base = json.loads(json.dumps(integrity_base))
    invalid_base["current_state"] = "INDEPENDENT_VERIFICATION"
    invalid_base["expected_smoke_episode_count"] = 24
    invalid_base["smoke_outcome_episodes"] = 24
    invalid_base["expected_confirmation_episode_count"] = 2000
    invalid_base["confirmation_outcome_episodes_generated"] = 2000
    invalid_base["confirmation_outcome_episodes_executed"] = 2000
    invalid_base["confirmation_outcomes_opened_for_analysis"] = True
    invalid_base["postconfirmation_integrity_failure"] = None
    contract = env["ATTEMPT_ROOT"] / "verifier_contract.json"
    contract_path = contract.relative_to(env["REPOSITORY_ROOT"]).as_posix()
    _write_json(
        contract,
        {
            "schema_version": 1,
            "attempt": "v008",
            "attempt_root": transaction._relative(env["ATTEMPT_ROOT"]),
            "mode": "confirmation",
        },
    )
    contract_sha256 = hashlib.sha256(contract.read_bytes()).hexdigest()
    audit = env["ATTEMPT_ROOT"] / "audit/independent_verification.json"
    audit_path = audit.relative_to(env["REPOSITORY_ROOT"]).as_posix()
    _write_json(
        audit,
        {
            "schema_version": 1,
            "attempt": "v008",
            "mode": "confirmation",
            "passed": False,
            "terminal_label": "domain_robust_gate_execution_invalid",
            "read_only_verifier": True,
            "verifier_contract": {
                "path": contract_path,
                "sha256": contract_sha256,
            },
            "capture": {
                "captured_exclusively": True,
                "wrapper_integrity_passed": True,
                "scientific_verifier_passed": False,
                "verifier_returncode": 1,
            },
            "error_type": "SyntheticError",
            "error": "synthetic verifier failure without integrity predicate",
        },
    )
    audit_sha256 = hashlib.sha256(audit.read_bytes()).hexdigest()
    decision_file = env["ATTEMPT_ROOT"] / "decision.json"
    decision_path = decision_file.relative_to(
        env["REPOSITORY_ROOT"]
    ).as_posix()
    _write_json(
        decision_file,
        {
            "schema_version": 1,
            "attempt": "v008",
            "passed": True,
            "checkpoint_state": "TERMINAL",
            "terminal_basis": "postconfirmation_integrity_failure",
            "terminal_label": "domain_robust_gate_execution_invalid",
            "process_valid": False,
            "independent_verification_path": audit_path,
            "independent_verification_sha256": audit_sha256,
            "confirmation_outcome_episodes_generated": 2000,
            "confirmation_outcome_episodes_executed": 2000,
            "confirmation_outcomes_opened_for_analysis": True,
        },
    )
    decision_sha256 = hashlib.sha256(decision_file.read_bytes()).hexdigest()
    invalid_events = [
        {
            "event": "state_completed",
            "attempt": "v008",
            "completed_state": "INDEPENDENT_VERIFICATION",
            "next_state": "TERMINAL",
            "checkpoint_name": "v008_execution_invalid_independent_audit",
            "evidence_path": audit_path,
            "evidence_sha256": audit_sha256,
            "postconfirmation_integrity_failure": True,
            "created_unix_ns": 9,
        },
        {
            "event": "scientific_terminal_decision_recorded",
            "attempt": "v008",
            "terminal_label": "domain_robust_gate_execution_invalid",
            "process_valid": False,
            "decision_path": decision_path,
            "decision_sha256": decision_sha256,
            "created_unix_ns": 9,
        },
        {
            "event": "state_completed",
            "attempt": "v008",
            "completed_state": "TERMINAL",
            "next_state": "POST_TERMINAL_REPORTING",
            "checkpoint_name": "v008_execution_invalid_terminal_decision",
            "evidence_path": decision_path,
            "evidence_sha256": decision_sha256,
            "postconfirmation_integrity_failure": True,
            "created_unix_ns": 9,
        },
    ]
    invalid_proposal = json.loads(json.dumps(invalid_base))
    invalid_proposal.pop(transaction.ADAPTER_STATE_KEY)
    invalid_audit_checkpoint = {
        "name": invalid_events[0]["checkpoint_name"],
        "created_unix_ns": 9,
        "evidence_path": audit_path,
        "evidence_sha256": audit_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    invalid_decision_checkpoint = {
        "name": invalid_events[2]["checkpoint_name"],
        "created_unix_ns": 9,
        "evidence_path": decision_path,
        "evidence_sha256": decision_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    invalid_proposal["completed_states"] = invalid_base["completed_states"] + [
        "INDEPENDENT_VERIFICATION", "TERMINAL"
    ]
    invalid_proposal["verified_checkpoints"] = invalid_base[
        "verified_checkpoints"
    ] + [invalid_audit_checkpoint, invalid_decision_checkpoint]
    invalid_proposal["last_verified_checkpoint"] = invalid_decision_checkpoint
    invalid_proposal["current_state"] = "POST_TERMINAL_REPORTING"
    invalid_proposal["terminal_label"] = (
        "domain_robust_gate_execution_invalid"
    )
    invalid_proposal["process_valid"] = False
    invalid_proposal["scientific_terminal"] = True
    invalid_proposal["confirmation_terminal"] = True
    invalid_proposal["terminal_basis"] = "postconfirmation_integrity_failure"
    invalid_proposal["terminal_decision_path"] = decision_path
    invalid_proposal["terminal_decision_sha256"] = decision_sha256
    invalid_proposal["postconfirmation_integrity_failure"] = {
        "source": "independent_verifier",
        "trigger_state": "INDEPENDENT_VERIFICATION",
        "skipped_states": [],
        "status": "terminal_recorded",
        "independent_verification_path": audit_path,
        "independent_verification_sha256": audit_sha256,
        "decision_path": decision_path,
        "decision_sha256": decision_sha256,
        "terminal_recorded_unix_ns": 9,
    }
    invalid_proposal["skipped_states"] = []
    invalid_proposal["next_action"] = (
        "report the immutable execution-invalid result and integrity diagnosis; "
        "do not retry or reinterpret it as a scientific partial/failure"
    )
    invalid_proposal["updated_unix_ns"] = 9
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="execution-invalid base/evidence drift",
    ):
        transaction._validate_replayed_controller_transition(
            base_state=invalid_base,
            proposed_state=invalid_proposal,
            events=invalid_events,
        )
    with pytest.raises(
        standalone.VerificationError,
        match="execution-invalid base/evidence drift",
    ):
        standalone._verify_descendant_controller_transition(
            base_state=invalid_base,
            proposed_state=invalid_proposal,
            controller_events=invalid_events,
            repo=env["REPOSITORY_ROOT"],
        )

    # The ordinary checkpoint route must never absorb an execution-invalid
    # verifier result; only the narrow three-event finalizer may do that.
    base = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    base["current_state"] = "INDEPENDENT_VERIFICATION"
    evidence = env["ATTEMPT_ROOT"] / "audit/ordinary_invalid.json"
    _write_json(
        evidence,
        {
            "attempt": "v008",
            "checkpoint_state": "INDEPENDENT_VERIFICATION",
            "passed": True,
            "terminal_label": "domain_robust_gate_execution_invalid",
        },
    )
    evidence_path = transaction._relative(evidence)
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    checkpoint = {
        "name": "v008_illegal_ordinary_execution_invalid",
        "created_unix_ns": 4,
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_checkpoint",
    }
    proposal = json.loads(json.dumps(base))
    proposal.pop(transaction.ADAPTER_STATE_KEY)
    proposal["completed_states"] = base["completed_states"] + [
        "INDEPENDENT_VERIFICATION"
    ]
    proposal["current_state"] = "TERMINAL"
    proposal["last_verified_checkpoint"] = checkpoint
    proposal["verified_checkpoints"] = base["verified_checkpoints"] + [checkpoint]
    proposal["next_action"] = "illegally bypass the narrow finalizer"
    proposal["updated_unix_ns"] = 4
    event = {
        "event": "state_completed",
        "attempt": "v008",
        "completed_state": "INDEPENDENT_VERIFICATION",
        "next_state": "TERMINAL",
        "checkpoint_name": checkpoint["name"],
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "created_unix_ns": 4,
    }
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="ordinary state-completed authorization drift",
    ):
        transaction._validate_replayed_controller_transition(
            base_state=base,
            proposed_state=proposal,
            events=[event],
        )
    standalone = transaction._load_module(
        "standalone_ordinary_execution_invalid_guard",
        ATTEMPT / "verify_version_forward.py",
    )
    with pytest.raises(
        standalone.VerificationError,
        match="ordinary state-completed authorization drift",
    ):
        standalone._verify_descendant_controller_transition(
            base_state=base,
            proposed_state=proposal,
            controller_events=[event],
            repo=env["REPOSITORY_ROOT"],
        )

    # A coherent terminal event/state projection is still unauthorized when
    # the immutable decision JSON does not carry the same scientific call.
    terminal_base = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    terminal_base["current_state"] = "TERMINAL"
    decision_file = env["ATTEMPT_ROOT"] / "decision.json"
    _write_json(
        decision_file,
        {
            "attempt": "v008",
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
        },
    )
    decision_path = transaction._relative(decision_file)
    decision_sha256 = hashlib.sha256(decision_file.read_bytes()).hexdigest()
    terminal_event = {
        "event": "scientific_terminal_decision_recorded",
        "attempt": "v008",
        "terminal_label": "domain_robust_gate_partial",
        "process_valid": True,
        "decision_path": decision_path,
        "decision_sha256": decision_sha256,
        "created_unix_ns": 5,
    }
    terminal_proposal = json.loads(json.dumps(terminal_base))
    terminal_proposal.pop(transaction.ADAPTER_STATE_KEY)
    terminal_proposal.update(
        {
            "terminal_label": "domain_robust_gate_partial",
            "process_valid": True,
            "scientific_terminal": True,
            "confirmation_terminal": True,
            "terminal_decision_path": decision_path,
            "terminal_decision_sha256": decision_sha256,
            "updated_unix_ns": 5,
        }
    )
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="terminal decision evidence drift",
    ):
        transaction._validate_replayed_controller_transition(
            base_state=terminal_base,
            proposed_state=terminal_proposal,
            events=[terminal_event],
        )
    with pytest.raises(
        standalone.VerificationError,
        match="terminal decision evidence drift",
    ):
        standalone._verify_descendant_controller_transition(
            base_state=terminal_base,
            proposed_state=terminal_proposal,
            controller_events=[terminal_event],
            repo=env["REPOSITORY_ROOT"],
        )


def test_schema_complete_checkpoint_cannot_publish_unrelated_state_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _commit_controller(env)
    )
    transaction.execute_version_forward()
    base = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    evidence = env["ATTEMPT_ROOT"] / "audit/schema_complete_selection.json"
    _write_json(
        evidence,
        {
            "attempt": "v008",
            "checkpoint_state": "SELECTION_COHORTS",
            "passed": True,
        },
    )
    evidence_path = transaction._relative(evidence)
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    checkpoint = {
        "name": "v008_schema_complete_selection",
        "created_unix_ns": 3,
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "source_attempt": "v008",
        "verification_lineage": "direct_checkpoint",
    }
    intended = json.loads(json.dumps(base))
    intended["completed_states"].append("SELECTION_COHORTS")
    intended["current_state"] = "CANDIDATE_SELECTION"
    intended["last_verified_checkpoint"] = checkpoint
    intended["verified_checkpoints"].append(checkpoint)
    intended["next_action"] = "continue after a schema-complete checkpoint"
    intended["updated_unix_ns"] = 3
    intended["fit_outcome_episodes"] = 999999
    event = {
        "event": "state_completed",
        "attempt": "v008",
        "completed_state": "SELECTION_COHORTS",
        "next_state": "CANDIDATE_SELECTION",
        "checkpoint_name": checkpoint["name"],
        "evidence_path": evidence_path,
        "evidence_sha256": evidence_sha256,
        "created_unix_ns": 3,
    }
    state_before = env["STATE_PATH"].read_bytes()
    ledger_before = env["LEDGER_PATH"].read_bytes()
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="replayed state-completed unrelated STATE drift",
    ):
        transaction._durable_commit_state_with_events(intended, [event])
    assert env["STATE_PATH"].read_bytes() == state_before
    assert env["LEDGER_PATH"].read_bytes() == ledger_before
    assert not os.path.lexists(env["TRANSACTION_PATH"])
    assert not os.path.lexists(transaction.PENDING_STAGING_PATH)
    assert not os.path.lexists(transaction.STATE_STAGING_PATH)


def test_schema_complete_early_failure_cannot_publish_unrelated_state_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _commit_controller(env)
    )
    transaction.execute_version_forward()
    _commit_ordinary_checkpoint(env, created_unix_ns=2)
    intended, event, _skipped = _synthetic_no_candidate_transition(
        env, created_unix_ns=6
    )
    intended["fit_outcome_episodes"] = 999999
    state_before = env["STATE_PATH"].read_bytes()
    ledger_before = env["LEDGER_PATH"].read_bytes()
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="replayed early-failure unrelated STATE drift",
    ):
        transaction._durable_commit_state_with_events(intended, [event])
    assert env["STATE_PATH"].read_bytes() == state_before
    assert env["LEDGER_PATH"].read_bytes() == ledger_before
    assert not os.path.lexists(env["TRANSACTION_PATH"])
    assert not os.path.lexists(transaction.PENDING_STAGING_PATH)
    assert not os.path.lexists(transaction.STATE_STAGING_PATH)


def test_schema_complete_terminal_group_cannot_publish_unrelated_state_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        transaction, "_invoke_controller", lambda: _commit_controller(env)
    )
    transaction.execute_version_forward()
    _commit_ordinary_checkpoint(env, created_unix_ns=2)
    skipped = _stage_synthetic_no_candidate(env, created_unix_ns=6)
    intended, events = _synthetic_early_terminal_transition(
        env, skipped=skipped, created_unix_ns=7
    )
    intended["fit_outcome_episodes"] = 999999
    state_before = env["STATE_PATH"].read_bytes()
    ledger_before = env["LEDGER_PATH"].read_bytes()
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="replayed three-event terminal unrelated STATE drift",
    ):
        transaction._durable_commit_state_with_events(intended, events)
    assert env["STATE_PATH"].read_bytes() == state_before
    assert env["LEDGER_PATH"].read_bytes() == ledger_before
    assert not os.path.lexists(env["TRANSACTION_PATH"])
    assert not os.path.lexists(transaction.PENDING_STAGING_PATH)
    assert not os.path.lexists(transaction.STATE_STAGING_PATH)


@pytest.mark.parametrize("tamper", ["state_only", "lowercase_operation"])
def test_adapter_marker_v2_rejects_current_state_and_operation_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    transaction.execute_version_forward()
    state = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    if tamper == "state_only":
        state["current_state"] = "TAMPERED_STATE_ONLY"
    else:
        state[transaction.ADAPTER_STATE_KEY]["operation_sha256"] = "b" * 64
    _write_json(env["STATE_PATH"], state)

    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="authorization drift",
    ):
        transaction.verify_transaction_receipt()
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="authorization drift",
    ):
        transaction._recover_durable_transaction_locked()


@pytest.mark.parametrize("tamper", ["state_only", "lowercase_operation"])
def test_adapter_marker_v2_rejects_tamper_after_legitimate_descendant_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    transaction.execute_version_forward()
    _commit_fit_count_descendant(env)
    state = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    if tamper == "state_only":
        state["expected_fit_episode_count"] = 999999
    else:
        state[transaction.ADAPTER_STATE_KEY]["operation_sha256"] = "b" * 64
    _write_json(env["STATE_PATH"], state)

    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="authorization drift",
    ):
        transaction.verify_transaction_receipt()
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="authorization drift",
    ):
        transaction._recover_durable_transaction_locked()


@pytest.mark.parametrize(
    "boundary",
    [
        "pending_staging_write",
        "pending_staging_fsync",
        "pending_rename",
        "pending_parent_fsync",
        "ledger_suffix_write",
        "ledger_suffix_fsync",
        "state_staging_write",
        "state_staging_fsync",
        "state_rename",
        "state_parent_fsync",
        "pending_unlink",
        "pending_unlink_parent_fsync",
    ],
)
def test_durable_adapter_recovers_every_commit_boundary_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    fired = False

    def crash(name: str) -> None:
        nonlocal fired
        if name == boundary and not fired:
            fired = True
            raise OSError(f"synthetic crash at {name}")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with pytest.raises(OSError, match=boundary):
        _durable_commit(env)
    assert fired is True

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with transaction._adapter_lock():
        state, chain = transaction._recover_durable_transaction_locked()
    assert state["active_attempt"] == "v008"
    assert state[transaction.ADAPTER_STATE_KEY]["ledger_event_count"] == (
        env["state"]["ledger_event_count"] + 1
    )
    assert state[transaction.ADAPTER_STATE_KEY]["ledger_head_sha256"] == chain[
        "head_sha256"
    ]
    assert not any(
        os.path.lexists(path)
        for path in (
            env["TRANSACTION_PATH"],
            transaction.PENDING_STAGING_PATH,
            transaction.STATE_STAGING_PATH,
        )
    )
    with transaction._adapter_lock():
        repeated_state, repeated_chain = transaction._recover_durable_transaction_locked()
    assert repeated_state == state and repeated_chain == chain


def test_authenticated_torn_ledger_suffix_is_truncated_and_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    intended, event = _durable_target_state(env)
    with transaction._adapter_lock():
        pending = transaction._build_pending(intended, [event])
        transaction._publish_pending(pending)
    suffix = pending["expected_suffix"].encode("ascii")
    base_ledger = env["LEDGER_PATH"].read_bytes()
    torn = suffix[: max(1, len(suffix) // 2)]
    assert not torn.endswith(b"\n")
    env["LEDGER_PATH"].write_bytes(base_ledger + torn)

    with transaction._adapter_lock():
        state, chain = transaction._recover_durable_transaction_locked()

    assert state["ledger_event_count"] == chain["event_count"] == (
        env["state"]["ledger_event_count"] + 1
    )
    assert env["LEDGER_PATH"].read_bytes() == base_ledger + suffix


@pytest.mark.parametrize(
    "boundary",
    [
        "recovery_pending_rename",
        "recovery_pending_parent_fsync",
        "ledger_partial_truncate",
        "ledger_partial_truncate_fsync",
        "ledger_complete_fsync",
        "recovery_state_rename",
        "recovery_state_parent_fsync",
    ],
)
def test_durable_adapter_recovers_every_reconciliation_boundary_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    intended, event = _durable_target_state(env)
    with transaction._adapter_lock():
        pending = transaction._build_pending(intended, [event])
        if boundary.startswith("recovery_pending"):
            transaction._write_staging_file(
                transaction.PENDING_STAGING_PATH,
                transaction._pretty_bytes(pending),
                prefix="setup_pending",
            )
        else:
            transaction._publish_pending(pending)
    suffix = pending["expected_suffix"].encode("ascii")
    base_ledger = env["LEDGER_PATH"].read_bytes()
    if boundary.startswith("ledger_partial"):
        env["LEDGER_PATH"].write_bytes(
            base_ledger + suffix[: max(1, len(suffix) // 2)]
        )
    elif boundary in {
        "ledger_complete_fsync",
        "recovery_state_rename",
        "recovery_state_parent_fsync",
    }:
        env["LEDGER_PATH"].write_bytes(base_ledger + suffix)
    if boundary.startswith("recovery_state"):
        transaction._write_staging_file(
            transaction.STATE_STAGING_PATH,
            transaction._pretty_bytes(pending["intended_state"]),
            prefix="setup_state",
        )

    fired = False

    def crash(name: str) -> None:
        nonlocal fired
        if name == boundary and not fired:
            fired = True
            raise OSError(f"synthetic recovery crash at {name}")

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
    with transaction._adapter_lock(), pytest.raises(OSError, match=boundary):
        transaction._recover_durable_transaction_locked()
    assert fired is True

    monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", lambda _name: None)
    with transaction._adapter_lock():
        state, chain = transaction._recover_durable_transaction_locked()
    assert state["active_attempt"] == "v008"
    assert state["ledger_event_count"] == chain["event_count"] == (
        env["state"]["ledger_event_count"] + 1
    )
    assert not any(
        os.path.lexists(path)
        for path in (
            env["TRANSACTION_PATH"],
            transaction.PENDING_STAGING_PATH,
            transaction.STATE_STAGING_PATH,
        )
    )


def test_nonprefix_torn_suffix_is_rejected_without_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    intended, event = _durable_target_state(env)
    with transaction._adapter_lock():
        pending = transaction._build_pending(intended, [event])
        transaction._publish_pending(pending)
    corrupted = env["LEDGER_PATH"].read_bytes() + b"x"
    env["LEDGER_PATH"].write_bytes(corrupted)

    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError, match="exact prefix"
    ):
        transaction._recover_durable_transaction_locked()

    assert env["LEDGER_PATH"].read_bytes() == corrupted


def test_lost_pending_and_unframed_ledger_are_never_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    intended, event = _durable_target_state(env)
    with transaction._adapter_lock():
        pending = transaction._build_pending(intended, [event])
        transaction._publish_pending(pending)
    suffix = pending["expected_suffix"].encode("ascii")
    base_ledger = env["LEDGER_PATH"].read_bytes()
    env["LEDGER_PATH"].write_bytes(base_ledger + suffix)
    env["TRANSACTION_PATH"].unlink()

    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError, match="diverged"
    ):
        transaction._recover_durable_transaction_locked()

    env["LEDGER_PATH"].write_bytes(base_ledger + suffix[:-1])
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError, match="newline"
    ):
        transaction._recover_durable_transaction_locked()


def test_direct_root_style_mutation_cannot_authorize_downstream_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    state = _durable_commit(env)
    direct = {
        "created_unix_ns": 8,
        "event": "direct_root_cli_mutation",
        "attempt": "v008",
        "seq": state["ledger_event_count"] + 1,
        "prev_sha256": state["ledger_head_sha256"],
    }
    direct["record_sha256"] = hashlib.sha256(_canonical(direct)).hexdigest()
    with env["LEDGER_PATH"].open("ab") as handle:
        handle.write(_canonical(direct) + b"\n")
    state["ledger_event_count"] = direct["seq"]
    state["ledger_head_sha256"] = direct["record_sha256"]
    _write_json(env["STATE_PATH"], state)

    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError, match="authorization drift"
    ):
        transaction._recover_durable_transaction_locked()


def test_two_crashed_count_commits_bind_and_restore_exact_prior_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    current = _durable_commit(env)

    for counter, value, created, crash_boundary in (
        ("selection_outcome_episodes", 4, 8, "ledger_suffix_fsync"),
        ("selection_outcome_episodes", 5, 9, "state_parent_fsync"),
    ):
        prior = json.loads(json.dumps(current))
        intended = json.loads(json.dumps(current))
        intended[counter] = value
        intended["updated_unix_ns"] = created
        event = {
            "event": "outcome_counts_updated",
            "attempt": "v008",
            "fields": {counter: value},
            "created_unix_ns": created,
        }
        fired = False

        def crash(name: str) -> None:
            nonlocal fired
            if name == crash_boundary and not fired:
                fired = True
                raise OSError(f"synthetic count crash at {name}")

        monkeypatch.setattr(transaction, "_DURABILITY_TEST_HOOK", crash)
        with transaction._adapter_lock(), pytest.raises(
            OSError, match="synthetic count crash"
        ):
            transaction._durable_commit_state_with_events(intended, [event])
        assert fired is True

        monkeypatch.setattr(
            transaction, "_DURABILITY_TEST_HOOK", lambda _name: None
        )
        with transaction._adapter_lock():
            current, chain = transaction._recover_durable_transaction_locked()

        records = [
            json.loads(line)
            for line in env["LEDGER_PATH"].read_bytes().splitlines()
        ]
        count_record = records[-1]
        prior_marker = count_record[transaction.PRIOR_ADAPTER_MARKER_FIELD]
        assert prior_marker == prior[transaction.ADAPTER_STATE_KEY]
        assert transaction.PRIOR_ADAPTER_MARKER_FIELD not in prior_marker

        reconstructed = json.loads(json.dumps(current))
        reconstructed[counter] = prior[counter]
        reconstructed["updated_unix_ns"] = prior["updated_unix_ns"]
        reconstructed["ledger_event_count"] = prior["ledger_event_count"]
        reconstructed["ledger_head_sha256"] = prior["ledger_head_sha256"]
        reconstructed[transaction.ADAPTER_STATE_KEY] = prior_marker
        assert reconstructed == prior
        assert current["ledger_event_count"] == chain["event_count"]
        assert current["ledger_head_sha256"] == chain["head_sha256"]


def test_pending_count_event_rederives_prior_marker_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    current = _durable_commit(env)
    intended = json.loads(json.dumps(current))
    intended["selection_outcome_episodes"] = 4
    intended["updated_unix_ns"] = 8
    event = {
        "event": "outcome_counts_updated",
        "attempt": "v008",
        "fields": {"selection_outcome_episodes": 4},
        "created_unix_ns": 8,
    }
    with transaction._adapter_lock():
        pending = transaction._build_pending(intended, [event])
    assert (
        pending["events"][0][transaction.PRIOR_ADAPTER_MARKER_FIELD]
        == current[transaction.ADAPTER_STATE_KEY]
    )

    tampered = json.loads(json.dumps(pending))
    tampered["events"][0][transaction.PRIOR_ADAPTER_MARKER_FIELD][
        "operation_sha256"
    ] = "f" * 64
    tampered["transaction_sha256"] = transaction._pending_digest_v2(tampered)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="envelope content drift",
    ):
        transaction._validate_pending(tampered)

    state_tampered = json.loads(json.dumps(pending))
    state_tampered["intended_state"]["current_state"] = "TAMPERED"
    state_tampered["transaction_sha256"] = transaction._pending_digest_v2(
        state_tampered
    )
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="transaction envelope drift",
    ):
        transaction._validate_pending(state_tampered)


@pytest.mark.parametrize("mutation", ["decrease", "wrong_state"])
def test_count_transition_rejects_backward_or_wrong_state_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    current = _durable_commit(env)
    intended = json.loads(json.dumps(current))
    if mutation == "decrease":
        field = "selection_outcome_episodes"
        value = current[field] - 1
    else:
        field = "fit_outcome_episodes"
        value = current[field] + 1
    intended[field] = value
    intended["updated_unix_ns"] = 9
    event = {
        "event": "outcome_counts_updated",
        "attempt": "v008",
        "fields": {field: value},
        "created_unix_ns": 9,
    }
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="moved backward|forbidden state",
    ):
        transaction._build_pending(intended, [event])


def test_controller_cannot_prepopulate_adapter_owned_prior_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    current = _durable_commit(env)
    intended = json.loads(json.dumps(current))
    intended["updated_unix_ns"] = 8
    event = {
        "event": "outcome_counts_updated",
        "attempt": "v008",
        "fields": {"selection_outcome_episodes": 4},
        "created_unix_ns": 8,
        transaction.PRIOR_ADAPTER_MARKER_FIELD: current[
            transaction.ADAPTER_STATE_KEY
        ],
    }
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="pre-populates adapter-owned prior marker",
    ):
        transaction._build_pending(intended, [event])


@pytest.mark.parametrize(
    ("assignments", "match"),
    [
        ({"fit_outcome_episodes": False}, "type drift"),
        ({"fit_outcome_episodes": True}, "type drift"),
        ({"confirmation_outcomes_opened_for_analysis": 0}, "type drift"),
        ({"confirmation_outcomes_opened_for_analysis": 1}, "type drift"),
        ({"fit_outcome_episodes": -1}, "type drift"),
    ],
)
def test_adapter_rejects_bool_int_aliases_before_controller_invocation(
    monkeypatch: pytest.MonkeyPatch,
    assignments: dict[str, int | bool],
    match: str,
) -> None:
    invoked = False

    def forbidden(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal invoked
        invoked = True
        raise AssertionError("root controller must not be invoked")

    monkeypatch.setattr(transaction, "_adapter_operation", forbidden)
    with pytest.raises(transaction.VersionForwardTransactionError, match=match):
        transaction.update_counts(**assignments)
    assert invoked is False


@pytest.mark.parametrize(
    "assignment",
    [
        "fit_outcome_episodes=false",
        "confirmation_outcomes_opened_for_analysis=0",
    ],
)
def test_adapter_cli_rejects_bool_int_aliases_before_io(assignment: str) -> None:
    with pytest.raises(
        transaction.VersionForwardTransactionError, match="type drift"
    ):
        transaction.main(["counts", assignment])


def test_pending_state_rejects_bool_int_outcome_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    current = _durable_commit(env)
    intended = json.loads(json.dumps(current))
    intended["selection_outcome_episodes"] = False
    intended["updated_unix_ns"] = 8
    event = {
        "event": "outcome_counts_updated",
        "attempt": "v008",
        "fields": {"selection_outcome_episodes": False},
        "created_unix_ns": 8,
    }
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="outcome-counter type drift",
    ):
        transaction._build_pending(intended, [event])
    assert not os.path.lexists(env["TRANSACTION_PATH"])


@pytest.mark.parametrize("alias", [True, 1.0])
def test_current_state_rejects_structural_ledger_count_aliases_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: bool | float,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    state = json.loads(env["STATE_PATH"].read_text(encoding="utf-8"))
    state["ledger_event_count"] = alias
    _write_json(env["STATE_PATH"], state)
    before_state = env["STATE_PATH"].read_bytes()
    before_ledger = env["LEDGER_PATH"].read_bytes()
    with transaction._adapter_lock(), pytest.raises(
        transaction.VersionForwardTransactionError,
        match="ledger structural type drift",
    ):
        transaction._recover_durable_transaction_locked()
    assert env["STATE_PATH"].read_bytes() == before_state
    assert env["LEDGER_PATH"].read_bytes() == before_ledger


def test_pending_rejects_float_target_event_count_with_recomputed_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    intended, event = _durable_target_state(env)
    with transaction._adapter_lock():
        pending = transaction._build_pending(intended, [event])
    pending["target_event_count"] = float(pending["target_event_count"])
    pending["transaction_sha256"] = transaction._pending_digest_v2(pending)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="base binding drift",
    ):
        transaction._validate_pending(pending)


@pytest.mark.parametrize(
    ("record_name", "field", "alias_kind"),
    [
        *[
            (record_name, field, "float")
            for record_name in ("root_program", "adapter_source", "ledger_genesis")
            for field in ("bytes", "device", "inode", "nlink")
        ],
        *[
            (record_name, "nlink", "bool")
            for record_name in ("root_program", "adapter_source", "ledger_genesis")
        ],
    ],
)
def test_pending_rejects_source_and_genesis_metadata_numeric_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_name: str,
    field: str,
    alias_kind: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    with transaction._adapter_lock():
        journal = transaction._build_receipt_journal_locked(
            prospective_created_unix_ns=100
        )
    pending = json.loads(
        json.dumps(journal["predicted_controller_transaction"])
    )
    original = pending[record_name][field]
    pending[record_name][field] = (
        True if alias_kind == "bool" else float(original)
    )
    pending["transaction_sha256"] = transaction._pending_digest_v2(pending)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="source-record metadata type drift|ledger-genesis schema/type drift",
    ):
        transaction._validate_pending(pending)


@pytest.mark.parametrize(
    ("record_name", "mutation"),
    [
        (record_name, mutation)
        for record_name in ("root_program", "adapter_source", "ledger_genesis")
        for mutation in ("extra", "missing")
    ],
)
def test_pending_rejects_source_and_genesis_closed_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_name: str,
    mutation: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    with transaction._adapter_lock():
        journal = transaction._build_receipt_journal_locked(
            prospective_created_unix_ns=100
        )
    pending = json.loads(
        json.dumps(journal["predicted_controller_transaction"])
    )
    if mutation == "extra":
        pending[record_name]["unexpected"] = 1
    else:
        pending[record_name].pop("device")
    pending["transaction_sha256"] = transaction._pending_digest_v2(pending)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="source-record schema drift|ledger-genesis schema/type drift",
    ):
        transaction._validate_pending(pending)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seq", "float"),
        ("created_unix_ns", "float"),
        ("attempt_parameterization_verified", "int"),
    ],
)
def test_pending_rejects_expected_record_numeric_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    with transaction._adapter_lock():
        journal = transaction._build_receipt_journal_locked(
            prospective_created_unix_ns=100
        )
    pending = json.loads(
        json.dumps(journal["predicted_controller_transaction"])
    )
    original = pending["expected_records"][0][field]
    pending["expected_records"][0][field] = (
        1 if replacement == "int" else float(original)
    )
    pending["transaction_sha256"] = transaction._pending_digest_v2(pending)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="pending expected ledger/state drift",
    ):
        transaction._validate_pending(pending)


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("schema_version", True),
        ("event_index", 0.0),
        ("event_count", 1.0),
    ],
)
def test_pending_rejects_descendant_envelope_numeric_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    alias: bool | float,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    intended, event = _durable_target_state(env)
    with transaction._adapter_lock():
        pending = transaction._build_pending(intended, [event])
    pending["events"][0][transaction.ADAPTER_TRANSACTION_ENVELOPE_FIELD][
        field
    ] = alias
    pending["transaction_sha256"] = transaction._pending_digest_v2(pending)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="transaction envelope drift",
    ):
        transaction._validate_pending(pending)


def test_receipt_context_rejects_bool_schema_alias_with_recomputed_context_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    transaction.execute_version_forward()
    receipt = json.loads(env["RECEIPT_PATH"].read_text(encoding="utf-8"))
    receipt["receipt_context"]["schema_version"] = True
    receipt["receipt_context_sha256"] = transaction._canonical_sha256(
        receipt["receipt_context"]
    )
    _write_json(env["RECEIPT_PATH"], receipt)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="precommit context cross-link drift",
    ):
        transaction.verify_transaction_receipt()


@pytest.mark.parametrize("mutation", ["operation", "post_state"])
def test_adapter_receipt_rejects_coherently_rehashed_operation_and_post_state_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    transaction.execute_version_forward()
    receipt = json.loads(env["RECEIPT_PATH"].read_text(encoding="utf-8"))
    controller_return = json.loads(json.dumps(receipt["controller_return"]))
    if mutation == "operation":
        controller_return[transaction.ADAPTER_STATE_KEY]["operation_sha256"] = "f" * 64
    else:
        controller_return["current_state"] = "TAMPERED_POST_STATE"
    marker_without_binding = json.loads(
        json.dumps(controller_return[transaction.ADAPTER_STATE_KEY])
    )
    marker_without_binding.pop("state_binding_sha256")
    state_without_marker = json.loads(json.dumps(controller_return))
    state_without_marker.pop(transaction.ADAPTER_STATE_KEY)
    controller_return[transaction.ADAPTER_STATE_KEY][
        "state_binding_sha256"
    ] = transaction._canonical_sha256(
        {
            "marker_without_state_binding": marker_without_binding,
            "state_without_marker": state_without_marker,
        }
    )
    post_snapshot = json.loads(json.dumps(receipt["post_snapshot"]))
    encoded_state = transaction._pretty_bytes(controller_return)
    post_snapshot["state"].update(
        {
            "sha256": hashlib.sha256(encoded_state).hexdigest(),
            "bytes": len(encoded_state),
            "object_sha256": transaction._canonical_sha256(controller_return),
            "object": controller_return,
        }
    )
    tampered = transaction._build_receipt_value(
        context=receipt["receipt_context"],
        receipt_context_sha256=receipt["receipt_context_sha256"],
        controller_return=controller_return,
        post_snapshot=post_snapshot,
        post_verifiers=receipt["post_verifiers"],
    )
    _write_json(env["RECEIPT_PATH"], tampered)
    with pytest.raises(
        transaction.VersionForwardTransactionError,
        match="post-state differs from its ledger-bound proposal",
    ):
        transaction.verify_transaction_receipt()


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_top",
        "missing_top",
        "wrong_top_type",
        "extra_snapshot",
        "missing_snapshot",
        "wrong_nested_type",
        "extra_verifier_wrapper",
        "missing_standalone_key",
        "extra_partition_count",
    ],
)
def test_receipt_closed_schema_rejects_extra_missing_and_type_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    env = _configure(tmp_path, monkeypatch)
    monkeypatch.setattr(transaction, "_invoke_controller", lambda: _commit_controller(env))
    transaction.execute_version_forward()
    receipt = json.loads(env["RECEIPT_PATH"].read_text(encoding="utf-8"))
    if mutation == "extra_top":
        receipt["extra"] = None
    elif mutation == "missing_top":
        receipt.pop("passed")
    elif mutation == "wrong_top_type":
        receipt["created_unix_ns"] = True
    elif mutation == "extra_snapshot":
        receipt["pre_snapshot"]["state"]["extra"] = None
    elif mutation == "missing_snapshot":
        receipt["post_snapshot"]["ledger"].pop("event_count")
    elif mutation == "wrong_nested_type":
        receipt["pre_snapshot"]["ledger"]["event_count"] = True
    elif mutation == "extra_verifier_wrapper":
        receipt["pre_verifiers"]["extra"] = None
    elif mutation == "missing_standalone_key":
        receipt["post_verifiers"]["standalone"].pop("read_only")
    else:
        receipt["pre_verifiers"]["independent_partitions"]["partition_counts"][
            "extra"
        ] = 0
    _write_json(env["RECEIPT_PATH"], receipt)

    with pytest.raises(transaction.VersionForwardTransactionError):
        transaction.verify_transaction_receipt()
