"""Synthetic provenance checks for the transitive v006 -> v008 lineage edge."""

from __future__ import annotations

import copy
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import checkpoints  # noqa: E402
import generator  # noqa: E402
import inherited_authorization as inherited  # noqa: E402
import independent_verify as independent  # noqa: E402
import runner  # noqa: E402
import study_common  # noqa: E402
import verify_version_forward as version_forward  # noqa: E402
import workflow  # noqa: E402

REAL_RECOMPUTATION = inherited._verify_post_forward_recomputation


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _bind_adapter_marker(
    state: Mapping[str, Any], marker: Mapping[str, Any]
) -> dict[str, Any]:
    core = copy.deepcopy(dict(marker))
    core.pop("state_binding_sha256", None)
    state_without_marker = copy.deepcopy(dict(state))
    state_without_marker.pop("v008_durable_controller_adapter", None)
    bound = copy.deepcopy(core)
    bound["state_binding_sha256"] = hashlib.sha256(
        _canonical(
            {
                "marker_without_state_binding": core,
                "state_without_marker": state_without_marker,
            }
        )
    ).hexdigest()
    return bound


def _lineage_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Build a v006 selection-boundary forward, then verify it as v008."""

    specification = importlib.util.spec_from_file_location(
        "v008_transaction_fixture_for_inherited",
        ROOT / "tests/test_version_forward_transaction.py",
    )
    assert specification is not None and specification.loader is not None
    transaction_fixture = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(transaction_fixture)
    env = transaction_fixture._configure(tmp_path, monkeypatch)
    transaction = transaction_fixture.transaction

    repo = env["REPOSITORY_ROOT"]
    study = env["STUDY_ROOT"]
    v001 = study / "attempts/v001"
    v002 = study / "attempts/v002"
    v003 = study / "attempts/v003"
    v004 = study / "attempts/v004"
    v005 = study / "attempts/v005"
    v006 = study / "attempts/v006"
    v008 = env["ATTEMPT_ROOT"]
    live_study = ROOT.parents[1]
    required_runtime = (
        "inherited_authorization.py",
        "generator.py",
        "runner.py",
        "workflow.py",
        "checkpoints.py",
        "verify_version_forward.py",
        "DGP_MATRIX.json",
        "cohort_seed_ledger.json",
    )
    for name in required_runtime:
        path = v008 / name
        if not path.exists():
            path.write_text(f"sealed:{name}\n", encoding="utf-8")

    source_seal_path = v006 / "audit/pre_data_inheritance_seal.json"
    source_preselection_path = v006 / "audit/pre_selection_seal.json"
    intermediate_seal_path = v004 / "audit/pre_data_inheritance_seal.json"
    fit_seal_path = v003 / "audit/pre_data_inheritance_seal.json"
    prior_seal_path = v002 / "audit/pre_data_inheritance_seal.json"
    fit_invalidity_path = v003 / "audit/v003_procedural_invalidity.json"
    fit_receipt_path = v003 / "audit/version_forward_transaction_receipt.json"
    intermediate_invalidity_path = (
        v004 / "audit/v004_procedural_invalidity.json"
    )
    intermediate_receipt_path = (
        v004 / "audit/version_forward_transaction_receipt.json"
    )
    prior_invalidity_path = (
        v002 / "audit/v002_procedural_invalidity_v2.json"
    )
    first_receipt_path = v002 / "audit/version_forward_transaction_receipt.json"
    original_invalidity_path = v001 / "audit/v001_procedural_invalidity.json"
    for live, destination in (
        (
            live_study / "attempts/v002/audit/pre_data_inheritance_seal.json",
            prior_seal_path,
        ),
        (
            live_study / "attempts/v002/audit/v002_procedural_invalidity_v2.json",
            prior_invalidity_path,
        ),
        (
            live_study / "attempts/v002/audit/version_forward_transaction_receipt.json",
            first_receipt_path,
        ),
        (
            live_study / "attempts/v003/audit/pre_data_inheritance_seal.json",
            fit_seal_path,
        ),
        (
            live_study / "attempts/v003/audit/v003_procedural_invalidity.json",
            fit_invalidity_path,
        ),
        (
            live_study / "attempts/v003/audit/version_forward_transaction_receipt.json",
            fit_receipt_path,
        ),
        (
            live_study / "attempts/v004/audit/pre_data_inheritance_seal.json",
            intermediate_seal_path,
        ),
        (
            live_study / "attempts/v004/audit/v004_procedural_invalidity.json",
            intermediate_invalidity_path,
        ),
        (
            live_study / "attempts/v004/audit/version_forward_transaction_receipt.json",
            intermediate_receipt_path,
        ),
            (
                live_study / "attempts/v006/audit/pre_data_inheritance_seal.json",
                source_seal_path,
            ),
            (
                live_study / "attempts/v006/audit/pre_selection_seal.json",
                source_preselection_path,
            ),
        (
            live_study / "attempts/v001/audit/v001_procedural_invalidity.json",
            original_invalidity_path,
        ),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, destination)

    seal_path = env["SEAL_PATH"]
    invalidity_path = env["INVALIDITY_PATH"]
    invalidity_draft_path = env["INVALIDITY_DRAFT_PATH"]
    prior_receipt_path = env["PRIOR_RECEIPT_PATH"]
    source_transaction_path = v006 / "version_forward_transaction.py"
    source_state = copy.deepcopy(env["state"])
    projection = [
        inherited._checkpoint_projection(item)
        for item in source_state["verified_checkpoints"]
    ]
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal.update(
        {
            "schema_version": 1,
            "attempt": "v008",
                "source_attempt": "v006",
            "target_attempt": "v008",
            "science_attempt": "v001",
            "checkpoint_state": "PRE_OUTCOME_SEAL",
                "resume_state": "SELECTION_COHORTS",
                "authorization_kind": (
                    "zero_confirmation_outcome_version_forward_inherited_pre_selection"
                ),
            "procedural_invalidity_confirmed": True,
            "zero_confirmation_outcomes_at_version_forward": True,
            "outcome_counts_at_seal": dict(inherited.ZERO_COUNTS),
            "invalidity_evidence": {
                "path": _relative(repo, invalidity_path),
                "sha256": _sha(invalidity_path),
            },
            "superseded_invalidity_draft": {
                "path": _relative(repo, invalidity_draft_path),
                "sha256": _sha(invalidity_draft_path),
                "authoritative": False,
            },
                "source_pre_data_seal": {
                    "path": _relative(repo, source_seal_path),
                    "sha256": _sha(source_seal_path),
                },
                "source_pre_selection_seal": {
                    "path": _relative(repo, source_preselection_path),
                    "sha256": _sha(source_preselection_path),
                },
            "source_version_forward_transaction_receipt": {
                "path": _relative(repo, prior_receipt_path),
                "sha256": _sha(prior_receipt_path),
            },
                "source_controller_adapter_marker": copy.deepcopy(
                    source_state["v006_durable_controller_adapter"]
                ),
            "prior_version_forward_lineage": copy.deepcopy(
                source_state["version_forward_lineage"]
            ),
            "inherited_verified_checkpoints": projection,
            "sealed_files": {
                _relative(repo, v008 / name): _sha(v008 / name)
                for name in required_runtime
            },
            **dict(inherited.REQUIRED_EQUIVALENCE),
        }
    )
    _json(seal_path, seal)
    seal_hash = _sha(seal_path)
    seal_relative = _relative(repo, seal_path)
    partition_result = {
        "source_path_count": len(
            seal["complete_source_partition"]["source_paths"]
        ),
        "target_path_count": len(
            seal["complete_source_partition"]["target_paths"]
        ),
        "partition_counts": {
            name: len(records)
            for name, records in seal["source_partitions"].items()
        },
    }

    def standalone_result(phase: str) -> dict[str, Any]:
        return {
            "passed": True,
            "phase": phase,
            "attempt": "v008",
            "active_attempt": "v008",
            "science_attempt": "v001",
            "seal_path": seal_relative,
            "seal_sha256": seal_hash,
            "partition_file_count": partition_result["target_path_count"],
            "sealed_file_count": len(seal["sealed_files"]),
            "authorized_early_verifier_state": None,
            "authorized_role_count_recovery": None,
            "outcome_arrays_opened": False,
            "output_paths_created": 0,
            "read_only": True,
        }

    monkeypatch.setattr(
        transaction,
        "_preverify",
        lambda: {
            "producer": {
                "passed": True,
                "phase": "pre-forward",
                "attempt": "v008",
                "seal_path": seal_relative,
                "seal_sha256": seal_hash,
                "state_sha256": _sha(env["STATE_PATH"]),
                "ledger_sha256": _sha(env["LEDGER_PATH"]),
                "outcome_arrays_opened": False,
                "read_only": True,
            },
            "standalone": standalone_result("pre-forward"),
            "independent_partitions": partition_result,
        },
    )
    monkeypatch.setattr(
        transaction,
        "_postverify",
        lambda: {
            "standalone": standalone_result("post-forward"),
            "independent_partitions": partition_result,
        },
    )

    base_preview = transaction._preview_controller_proposal_locked

    def transitive_preview(
        base_state: dict[str, Any], *, created_unix_ns: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        proposed, events = base_preview(
            base_state, created_unix_ns=created_unix_ns
        )
        annotation = {
            "attempt": "v008",
            "equivalence_evidence_path": seal_relative,
            "equivalence_evidence_sha256": seal_hash,
        }
        checkpoints_value = copy.deepcopy(proposed["verified_checkpoints"])
        for checkpoint in checkpoints_value:
            inherited_into = list(checkpoint.get("inherited_into_attempts", []))
            checkpoint["inherited_into_attempts"] = [
                *inherited_into,
                copy.deepcopy(annotation),
            ]
        proposed["verified_checkpoints"] = checkpoints_value
        proposed["last_verified_checkpoint"] = copy.deepcopy(
            checkpoints_value[-1]
        )
        current_projection = [
            inherited._checkpoint_projection(item)
            for item in checkpoints_value
        ]
        proposed["version_forward_lineage"][-1][
            "inherited_verified_checkpoints"
        ] = copy.deepcopy(current_projection)
        events[0]["inherited_verified_checkpoints"] = copy.deepcopy(
            current_projection
        )
        return proposed, events

    monkeypatch.setattr(
        transaction, "_preview_controller_proposal_locked", transitive_preview
    )
    monkeypatch.setattr(
        transaction,
        "_invoke_controller",
        lambda: transaction_fixture._journal_bound_controller(env),
    )
    receipt = transaction.execute_version_forward()
    assert receipt["passed"] is True
    state_path = env["STATE_PATH"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    ledger = env["LEDGER_PATH"]
    genesis = env["GENESIS_PATH"]
    receipt_path = env["RECEIPT_PATH"]
    lock_path = env["LOCK_PATH"]
    program_path = env["PROGRAM_PATH"]
    transaction_source_path = env["TRANSACTION_SOURCE_PATH"]

    values = {
        "REPO_ROOT": repo,
        "STUDY_ROOT": study,
        "ATTEMPT_ROOT": v008,
        "STATE_PATH": state_path,
        "LEDGER_PATH": ledger,
        "LEDGER_GENESIS_PATH": genesis,
        "PROGRAM_LOCK_PATH": lock_path,
        "INHERITANCE_SEAL_PATH": seal_path,
        "SOURCE_PRE_DATA_SEAL_PATH": source_seal_path,
        "SOURCE_PRE_SELECTION_SEAL_PATH": source_preselection_path,
        "INTERMEDIATE_PRE_DATA_SEAL_PATH": intermediate_seal_path,
        "FIT_PRE_DATA_SEAL_PATH": fit_seal_path,
        "PRIOR_PRE_DATA_SEAL_PATH": prior_seal_path,
        "INVALIDITY_PATH": invalidity_path,
        "INVALIDITY_DRAFT_PATH": invalidity_draft_path,
        "INTERMEDIATE_INVALIDITY_PATH": intermediate_invalidity_path,
        "FIT_INVALIDITY_PATH": fit_invalidity_path,
        "PRIOR_INVALIDITY_PATH": prior_invalidity_path,
        "FIT_TRANSACTION_RECEIPT_PATH": fit_receipt_path,
        "INTERMEDIATE_TRANSACTION_RECEIPT_PATH": intermediate_receipt_path,
        "PRIOR_TRANSACTION_RECEIPT_PATH": first_receipt_path,
        "SOURCE_TRANSACTION_RECEIPT_PATH": prior_receipt_path,
        "SOURCE_CONTROLLER_ADAPTER_PATH": source_transaction_path,
        "TRANSACTION_RECEIPT_PATH": receipt_path,
        "TRANSACTION_SOURCE_PATH": transaction_source_path,
        "CONTROLLER_ADAPTER_PATH": transaction_source_path,
        "PROGRAM_PATH": program_path,
        "STATE_TRANSACTION_PATH": env["TRANSACTION_PATH"],
        "PENDING_STAGING_PATH": study / ".STATE_TRANSACTION.json.v008-staging",
        "STATE_STAGING_PATH": study / ".STATE.json.v008-staging",
        "RECEIPT_JOURNAL_PATH": env["RECEIPT_JOURNAL_PATH"],
        "RECEIPT_JOURNAL_STAGING_PATH": (
            study / ".VERSION_FORWARD_TRANSACTION.json.v008-staging"
        ),
        "RECEIPT_STAGING_PATH": (
            v008 / "audit/.version_forward_transaction_receipt.json.v008-staging"
        ),
        "INDEPENDENT_VERIFIER_PATH": v008 / "verify_version_forward.py",
    }
    for name, value in values.items():
        monkeypatch.setattr(inherited, name, value)

    partition_file_count = len(
        seal["v008_manifest_closure"]["discovered_paths"]
    )
    sealed_file_count = len(seal["sealed_files"])
    monkeypatch.setattr(
        inherited,
        "_verify_post_forward_recomputation",
        lambda path, **kwargs: {
            "passed": True,
            "phase": "post-forward",
            "attempt": "v008",
            "active_attempt": "v008",
            "science_attempt": "v001",
            "seal_path": _relative(repo, path),
            "seal_sha256": _sha(path),
            "partition_file_count": partition_file_count,
            "sealed_file_count": sealed_file_count,
            "outcome_arrays_opened": False,
            "output_paths_created": 0,
            "read_only": True,
            "authorized_early_verifier_state": kwargs.get(
                "authorized_early_state"
            ),
            "authorized_role_count_recovery": kwargs.get(
                "authorized_role_count_recovery"
            ),
        },
    )
    return state, {
        "repo": repo,
        "study": study,
        "v008": v008,
        "seal": seal_path,
        "ledger": ledger,
        "genesis": genesis,
        "state": state_path,
        "lock": lock_path,
        "receipt": receipt_path,
    }


def _verify(
    state: dict[str, Any], paths: dict[str, Path], *, expected: str = "SELECTION_COHORTS"
) -> dict[str, Any]:
    return inherited.verify_inherited_pre_data_authorization(
        expected_current_state=expected,
        state=state,
        seal_path=paths["seal"],
        ledger_path=paths["ledger"],
        genesis_path=paths["genesis"],
    )


def test_exact_inherited_lineage_authorizes_fit_without_replaying_prior_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    result = _verify(state, paths)
    assert result["passed"] is True
    assert result["active_attempt"] == "v008"
    assert result["science_attempt"] == "v001"
    assert result["inherited_checkpoint_count"] == 9
    assert state["completed_states"] == list(inherited.INHERITED_STATES)


@pytest.mark.parametrize("count_kind", ["sealed", "partition"])
def test_inherited_receipt_rejects_coherently_wrong_active_seal_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count_kind: str,
) -> None:
    _state, paths = _lineage_fixture(tmp_path, monkeypatch)
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    sealed_files = inherited._verify_sealed_files(seal["sealed_files"])
    expected = inherited._active_seal_verifier_projection(seal, sealed_files)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    result = copy.deepcopy(receipt["pre_verifiers"])
    if count_kind == "sealed":
        result["standalone"]["sealed_file_count"] += 1000
    else:
        result["standalone"]["partition_file_count"] += 1000
        result["independent_partitions"]["source_path_count"] += 1000
        result["independent_partitions"]["target_path_count"] += 1000
        result["independent_partitions"]["partition_counts"][
            "exact_hash"
        ] += 1000
    with pytest.raises(inherited.InheritedAuthorizationError):
        inherited._verify_receipt_verifier_result(
            result,
            phase="pre-forward",
            seal_relative=_relative(paths["repo"], paths["seal"]),
            seal_sha256=_sha(paths["seal"]),
            expected_projection=expected,
        )


@pytest.mark.parametrize("tamper", ["state_only", "lowercase_operation"])
def test_inherited_marker_v2_rejects_current_state_and_operation_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    if tamper == "state_only":
        state["expected_fit_episode_count"] = 999999
    else:
        state["v008_durable_controller_adapter"]["operation_sha256"] = "d" * 64
    _json(paths["state"], state)
    with pytest.raises(
        inherited.InheritedAuthorizationError,
        match="drift",
    ):
        _verify(state, paths)


@pytest.mark.parametrize(
    "tamper",
    (
        "schema",
        "outcome_type_alias",
        "stable_snapshot_inode",
        "noncanonical_bytes",
        "stale_context_digest",
        "rehashed_context_without_ledger_commitment",
    ),
)
def test_transaction_receipt_v2_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if tamper == "schema":
        receipt["schema_version"] = 1
    elif tamper == "outcome_type_alias":
        receipt["outcome_counts"]["fit_outcome_episodes"] = False
    elif tamper == "stable_snapshot_inode":
        receipt["pre_snapshot"]["state"]["inode"] = 0
    elif tamper == "noncanonical_bytes":
        paths["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
    elif tamper == "stale_context_digest":
        receipt["receipt_context"]["controller_proposal"]["events"][0][
            "resume_state"
        ] = "FIT_LOCK"
    else:
        receipt["receipt_context"]["controller_proposal"]["events"][0][
            "resume_state"
        ] = "FIT_LOCK"
        receipt["receipt_context_sha256"] = hashlib.sha256(
            _canonical(receipt["receipt_context"])
        ).hexdigest()
    if tamper != "noncanonical_bytes":
        _json(paths["receipt"], receipt)
    with pytest.raises(inherited.InheritedAuthorizationError):
        _verify(state, paths)


@pytest.mark.parametrize(
    "residue_name",
    (
        "STATE_TRANSACTION.json",
        ".STATE_TRANSACTION.json.v008-staging",
        ".STATE.json.v008-staging",
        "VERSION_FORWARD_TRANSACTION.json",
        ".VERSION_FORWARD_TRANSACTION.json.v008-staging",
    ),
)
def test_transaction_receipt_rejects_controller_or_journal_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    residue_name: str,
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    (paths["study"] / residue_name).write_text("residue\n", encoding="utf-8")
    with pytest.raises(inherited.InheritedAuthorizationError, match="header drift"):
        _verify(state, paths)


def test_transaction_receipt_rejects_receipt_staging_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    staging = (
        paths["v008"]
        / "audit/.version_forward_transaction_receipt.json.v008-staging"
    )
    staging.write_text("residue\n", encoding="utf-8")
    with pytest.raises(inherited.InheritedAuthorizationError, match="header drift"):
        _verify(state, paths)


@pytest.mark.parametrize("tamper", ["edge_hash", "annotation", "first_direct"])
def test_inherited_lineage_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    state = copy.deepcopy(state)
    if tamper == "edge_hash":
        state["version_forward_lineage"][0]["equivalence_sha256"] = "0" * 64
    elif tamper == "annotation":
        state["verified_checkpoints"][0]["inherited_into_attempts"] = []
    else:
        evidence = paths["v008"] / "audit/wrong_first.json"
        _json(evidence, {"passed": True})
        state["completed_states"].append("FIT_LOCK")
        state["verified_checkpoints"].append(
            {
                "name": "wrong_first",
                "evidence_path": _relative(paths["repo"], evidence),
                "evidence_sha256": _sha(evidence),
                "source_attempt": "v008",
            }
        )
    with pytest.raises(inherited.InheritedAuthorizationError):
        _verify(state, paths)


def test_ledger_edge_tamper_fails_independent_chain_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    raw = paths["ledger"].read_bytes()
    paths["ledger"].write_bytes(raw.replace(b'"resume_state":"FIT_COHORTS"', b'"resume_state":"FIT_LOCK"'))
    with pytest.raises(inherited.InheritedAuthorizationError, match="ledger"):
        _verify(state, paths)


def test_incomplete_handcrafted_seal_cannot_bypass_independent_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)

    def reject(_path: Path, **_kwargs: Any) -> dict[str, Any]:
        raise inherited.InheritedAuthorizationError(
            "independent verifier found incomplete source partitions"
        )

    monkeypatch.setattr(inherited, "_verify_post_forward_recomputation", reject)
    with pytest.raises(inherited.InheritedAuthorizationError, match="incomplete"):
        _verify(state, paths)


def test_independent_verifier_result_requires_science_identity_and_read_only_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _state, paths = _lineage_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        inherited.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "passed": True,
                    "phase": "post-forward",
                    "attempt": "v008",
                    "active_attempt": "v008",
                    # science_attempt deliberately absent
                    "seal_path": _relative(paths["repo"], paths["seal"]),
                    "seal_sha256": _sha(paths["seal"]),
                    "outcome_arrays_opened": False,
                    "output_paths_created": 0,
                    "read_only": True,
                }
            )
            + "\n",
            stderr="",
        ),
    )
    with pytest.raises(inherited.InheritedAuthorizationError, match="completeness"):
        REAL_RECOMPUTATION(paths["seal"])


@pytest.mark.parametrize("alias_field", ["partition_file_count", "sealed_file_count"])
def test_post_forward_recomputation_rejects_float_count_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_field: str,
) -> None:
    _state, paths = _lineage_fixture(tmp_path, monkeypatch)
    result: dict[str, Any] = {
        "passed": True,
        "phase": "post-forward",
        "attempt": "v008",
        "active_attempt": "v008",
        "science_attempt": "v001",
        "seal_path": _relative(paths["repo"], paths["seal"]),
        "seal_sha256": _sha(paths["seal"]),
        "partition_file_count": 5,
        "sealed_file_count": 8,
        "authorized_early_verifier_state": None,
        "authorized_role_count_recovery": None,
        "outcome_arrays_opened": False,
        "output_paths_created": 0,
        "read_only": True,
    }
    result[alias_field] = float(result[alias_field])
    monkeypatch.setattr(
        inherited.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(result) + "\n",
            stderr="",
        ),
    )
    with pytest.raises(inherited.InheritedAuthorizationError, match="completeness"):
        REAL_RECOMPUTATION(
            paths["seal"],
            expected_partition_file_count=5,
            expected_sealed_file_count=8,
        )


def test_post_forward_recomputation_rejects_extra_result_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _state, paths = _lineage_fixture(tmp_path, monkeypatch)
    result: dict[str, Any] = {
        "passed": True,
        "phase": "post-forward",
        "attempt": "v008",
        "active_attempt": "v008",
        "science_attempt": "v001",
        "seal_path": _relative(paths["repo"], paths["seal"]),
        "seal_sha256": _sha(paths["seal"]),
        "partition_file_count": 5,
        "sealed_file_count": 8,
        "authorized_early_verifier_state": None,
        "authorized_role_count_recovery": None,
        "outcome_arrays_opened": False,
        "output_paths_created": 0,
        "read_only": True,
        "unrecognized_projection": "must-fail-closed",
    }
    monkeypatch.setattr(
        inherited.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(result) + "\n",
            stderr="",
        ),
    )
    with pytest.raises(inherited.InheritedAuthorizationError, match="completeness"):
        REAL_RECOMPUTATION(
            paths["seal"],
            expected_partition_file_count=5,
            expected_sealed_file_count=8,
        )


def test_generator_checks_inherited_lineage_before_missing_seal_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []

    def reject(*, expected_current_state: str) -> dict[str, Any]:
        called.append(expected_current_state)
        raise inherited.InheritedAuthorizationError("synthetic lineage rejection")

    monkeypatch.setattr(generator, "verify_inherited_pre_data_authorization", reject)
    monkeypatch.setattr(generator, "PRE_DATA_SEAL_PATH", tmp_path / "absent.json")
    with pytest.raises(generator.GenerationPreflightError, match="before seed or output"):
        generator.verify_authorization_seal("fit")
    assert called == ["FIT_COHORTS"]
    assert list(tmp_path.iterdir()) == []


def test_traversal_episode_id_is_rejected_before_path_construction(
    tmp_path: Path
) -> None:
    record = {
        "slot": 0,
        "episode_id": "../../drgv001-np-ft-p-000",
        "env_seed": 1,
        "policy_seed": 2,
        "oracle_np_seed": 3,
        "action_space_seed": 4,
    }
    with pytest.raises(generator.GenerationContractError, match="unsafe episode_id"):
        generator._validate_seed_record(record, label="synthetic/traversal")
    with pytest.raises(generator.GenerationContractError):
        generator.episode_paths("fit", "native_plan", record["episode_id"])
    assert list(tmp_path.iterdir()) == []


def test_symlinked_output_parent_fails_before_any_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempts/v008"
    outside = tmp_path / "outside"
    attempt.mkdir(parents=True)
    outside.mkdir()
    (attempt / "data").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(generator, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(
        generator, "REPLACEMENT_REGISTRY_PATH", attempt / "data/replacement_registry.json"
    )
    monkeypatch.setattr(
        generator,
        "REPLACEMENT_REGISTRY_LOCK_PATH",
        attempt / "data/replacement_registry.lock",
    )
    primary = ({"episode_id": "drgv001-np-ft-p-000"},)
    with pytest.raises(generator.GenerationPreflightError, match="linked"):
        generator.preflight_output_namespace("fit", "native_plan", primary)
    assert list(outside.iterdir()) == []


def test_selection_workflow_verifies_inherited_raw_and_direct_execution_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    attempt = repo / "runs/lewm_domain_robust_gate/attempts/v008"
    state_path = attempt.parents[1] / "STATE.json"
    raw_seal = attempt / "audit/pre_data_inheritance_seal.json"
    execution_seal = attempt / "audit/pre_selection_seal.json"
    _json(state_path, {"state": "selection"})
    _json(
        raw_seal,
        {
            "schema_version": 1,
            "attempt": "v008",
            "science_attempt": "v001",
            "checkpoint_state": "PRE_OUTCOME_SEAL",
            "passed": True,
        },
    )
    _json(
        execution_seal,
        {
            "schema_version": 1,
            "attempt": "v008",
            "checkpoint_state": "PRE_SELECTION_SEAL",
            "passed": True,
            "pre_data_seal_sha256": _sha(raw_seal),
        },
    )
    raw_link = {
        "path": _relative(repo, raw_seal),
        "sha256": _sha(raw_seal),
        "checkpoint_state": "PRE_OUTCOME_SEAL",
    }
    execution_link = {
        "path": _relative(repo, execution_seal),
        "sha256": _sha(execution_seal),
        "checkpoint_state": "PRE_SELECTION_SEAL",
    }
    receipt_path = attempt / "audit/version_forward_transaction_receipt.json"
    _json(receipt_path, {"passed": True})
    receipt = {
        "path": _relative(repo, receipt_path),
        "sha256": _sha(receipt_path),
        "passed": True,
    }
    monkeypatch.setattr(workflow, "REPO_ROOT", repo)
    monkeypatch.setattr(workflow, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(workflow, "STATE_PATH", state_path)
    monkeypatch.setattr(workflow, "ROLE_EPISODES_PER_DGP", {"fit": 1, "selection": 1})
    monkeypatch.setattr(
        workflow,
        "ROLE_RAW_SEALS",
        {"fit": (raw_seal, "PRE_OUTCOME_SEAL"), "selection": (raw_seal, "PRE_OUTCOME_SEAL")},
    )
    monkeypatch.setattr(
        workflow,
        "ROLE_EXECUTION_SEALS",
        {
            "fit": (raw_seal, "PRE_OUTCOME_SEAL"),
            "selection": (execution_seal, "PRE_SELECTION_SEAL"),
        },
    )
    monkeypatch.setattr(
        workflow,
        "verify_inherited_pre_data_authorization",
        lambda **_: {
            "seal_path": raw_link["path"],
            "seal_sha256": raw_link["sha256"],
                "version_forward_transaction_receipt": receipt,
                "authenticated_state_sha256": _sha(state_path),
                "independent_recomputation": {
                    "authorized_role_count_recovery": None,
                },
        },
    )
    captured: list[tuple[dict[str, str], dict[str, str]]] = []

    def verify_regime(role: str, regime: str, **kwargs: Any) -> dict[str, Any]:
        captured.append(
            (dict(kwargs["expected_raw_seal"]), dict(kwargs["expected_execution_seal"]))
        )
        return {
            "episode_count": 1,
            "row_count": workflow.ROWS_PER_EPISODE,
            "aggregate": {
                "path": f"aggregate/{regime}.npz",
                "sha256": hashlib.sha256(regime.encode()).hexdigest(),
            },
        }

    monkeypatch.setattr(workflow, "_verify_role_regime", verify_regime)
    state = {
        "active_attempt": "v008",
        "current_state": "SELECTION_COHORTS",
        "expected_selection_episode_count": len(workflow.REGIMES),
        "verified_checkpoints": [
            {
                "source_attempt": "v008",
                "evidence_path": execution_link["path"],
                "evidence_sha256": execution_link["sha256"],
            }
        ],
    }
    _json(state_path, state)
    result = workflow.verify_role_manifests("selection", state)
    assert result["raw_authorization_seal"] == raw_link
    assert result["execution_authorization_seal"] == execution_link
    assert captured == [(raw_link, execution_link)] * len(workflow.REGIMES)


def test_active_and_science_attempts_are_split_in_owned_runtime_modules() -> None:
    assert generator.ATTEMPT == runner.ATTEMPT_VERSION == workflow.ATTEMPT == "v008"
    assert generator.SCIENCE_ATTEMPT == runner.SCIENCE_ATTEMPT == "v001"
    assert checkpoints.ATTEMPT == "v008"
    assert checkpoints.SCIENCE_ATTEMPT == "v001"
def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hardening_relative(repo: Path, path: Path) -> str:
    return path.absolute().relative_to(repo.absolute()).as_posix()


def _selection_authorization_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    study = repo / "runs/lewm_domain_robust_gate"
    attempt = study / "attempts/v008"
    inheritance = attempt / "audit/pre_data_inheritance_seal.json"
    preselection = attempt / "audit/pre_selection_seal.json"
    state_path = study / "STATE.json"
    _write_json(
        inheritance,
        {
            "schema_version": 1,
            "attempt": "v008",
            "science_attempt": "v001",
            "checkpoint_state": "PRE_OUTCOME_SEAL",
            "passed": True,
        },
    )
    _write_json(
        preselection,
        {
            "schema_version": 1,
            "attempt": "v008",
            "checkpoint_state": "PRE_SELECTION_SEAL",
            "passed": True,
            "pre_data_seal_sha256": _sha256(inheritance),
        },
    )
    state = {
        "active_attempt": "v008",
        "current_state": "SELECTION_COHORTS",
        "verified_checkpoints": [
            {
                "source_attempt": "v008",
                "evidence_path": _hardening_relative(repo, preselection),
                "evidence_sha256": _sha256(preselection),
            }
        ],
    }
    _write_json(state_path, state)
    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    monkeypatch.setattr(runner, "STUDY_ROOT", study)
    monkeypatch.setattr(runner, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(runner, "INHERITANCE_SEAL_PATH", inheritance)
    monkeypatch.setattr(runner, "SOURCE_PRE_SELECTION_SEAL_PATH", preselection)
    monkeypatch.setattr(runner, "read_verified_controller", lambda: copy.deepcopy(state))
    monkeypatch.setattr(
        runner,
        "relative_to_repo",
        lambda path: _hardening_relative(repo, Path(path)),
    )
    monkeypatch.setattr(
        runner,
        "verify_here",
        lambda *args, **kwargs: {"passed": True, "mps": {"available": True}},
        raising=False,
    )
    monkeypatch.setattr(
        runner, "require_mps_device", lambda value: str(value), raising=False
    )
    return {
        "repo": repo,
        "study": study,
        "attempt": attempt,
        "inheritance": inheritance,
        "preselection": preselection,
        "state": state,
    }


def test_selection_execution_authenticates_inheritance_before_raw_manifest_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _selection_authorization_fixture(tmp_path, monkeypatch)
    raw_opened = False

    def reject_inheritance(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["expected_current_state"] == "SELECTION_COHORTS"
        assert kwargs["direct_selection_seal_path"] == env["preselection"]
        raise inherited.InheritedAuthorizationError("synthetic inherited-lineage drift")

    class RawManifestOpened(RuntimeError):
        pass

    def open_raw(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal raw_opened
        raw_opened = True
        raise RawManifestOpened("raw manifest opened before inherited authorization")

    monkeypatch.setattr(runner, "verify_inherited_pre_data_authorization", reject_inheritance)
    monkeypatch.setattr(runner, "_load_raw_manifest", open_raw)
    with pytest.raises(RuntimeError) as caught:
        runner.execute_development("selection", "native_plan", "mps")
    assert not isinstance(caught.value, RawManifestOpened)
    assert raw_opened is False
    assert env["preselection"].is_file()


def test_direct_preselection_authorization_must_bind_authenticated_inheritance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _selection_authorization_fixture(tmp_path, monkeypatch)
    inheritance_link = {
        "seal_path": _hardening_relative(env["repo"], env["inheritance"]),
        "seal_sha256": _sha256(env["inheritance"]),
        "science_attempt": "v001",
    }
    monkeypatch.setattr(
        runner,
        "verify_inherited_pre_data_authorization",
        lambda **kwargs: dict(inheritance_link),
    )
    seal = json.loads(env["preselection"].read_text(encoding="utf-8"))
    seal["pre_data_seal_sha256"] = "0" * 64
    _write_json(env["preselection"], seal)
    env["state"]["verified_checkpoints"][0]["evidence_sha256"] = _sha256(
        env["preselection"]
    )
    with pytest.raises(RuntimeError):
        runner._state_authorization("selection")


def _install_strict_raw_manifest_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    repo = tmp_path / "repo"
    study = repo / "runs/lewm_domain_robust_gate"
    attempt = study / "attempts/v008"
    raw_root = attempt / "data/fit/native_plan/raw"
    intent_root = attempt / "data/persistence_intents/fit/native_plan"
    manifest_path = attempt / "data/fit/native_plan/raw_manifest.json"
    ledger_path = attempt / "cohort_seed_ledger.json"
    dgp_path = attempt / "DGP_MATRIX.json"
    seal_path = attempt / "audit/pre_data_inheritance_seal.json"
    registry_path = attempt / "data/replacement_registry.json"
    raw_root.mkdir(parents=True)
    intent_root.mkdir(parents=True)
    shutil.copyfile(ROOT / "cohort_seed_ledger.json", ledger_path)
    shutil.copyfile(ROOT / "DGP_MATRIX.json", dgp_path)
    _write_json(seal_path, {"passed": True, "attempt": "v008"})
    _write_json(registry_path, {"schema_version": 1, "claims": []})
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    dgp = json.loads(dgp_path.read_text(encoding="utf-8"))
    assignments = ledger["regimes"]["native_plan"]["roles"]["fit"]["primary"]
    seal_link = {
        "path": _hardening_relative(repo, seal_path),
        "sha256": _sha256(seal_path),
        "checkpoint_state": "PRE_OUTCOME_SEAL",
    }
    seal = generator.SealLink(
        path=seal_path,
        relative_path=seal_link["path"],
        sha256=seal_link["sha256"],
        checkpoint_state="PRE_OUTCOME_SEAL",
        payload={},
    )
    monkeypatch.setattr(generator, "REPO_ROOT", repo)
    monkeypatch.setattr(generator, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(generator, "DGP_MATRIX_PATH", dgp_path)
    monkeypatch.setattr(generator, "SEED_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(generator, "REPLACEMENT_REGISTRY_PATH", registry_path)
    monkeypatch.setattr(
        generator,
        "REPLACEMENT_CLAIMS_ROOT",
        attempt / "data/replacement_claims",
    )
    monkeypatch.setattr(
        generator,
        "REPLACEMENT_REGISTRY_LOCK_PATH",
        attempt / "data/replacement_registry.lock",
    )
    raw_audit = {"array_sha256": {}, "fixture_without_array_open": True}
    monkeypatch.setattr(
        generator,
        "_verify_raw_against_intent",
        lambda raw_path, intent: ({}, copy.deepcopy(raw_audit)),
    )
    # This fixture attacks raw-manifest path/ledger/link closure only.  The
    # schema-v2 persistence-intent contract has its own dedicated tests.
    monkeypatch.setattr(
        generator,
        "_validate_persistence_intent",
        lambda intent, **_kwargs: dict(intent),
    )
    # Retained-failure/claim closure is exercised by its dedicated schema-v2
    # tests.  This fixture isolates raw-manifest path, ledger, and link closure.
    monkeypatch.setattr(
        generator,
        "_validate_retained_failure_partition",
        lambda *_args, **_kwargs: None,
    )
    records: list[dict[str, Any]] = []
    for assignment in assignments:
        episode_id = str(assignment["episode_id"])
        paths_for_episode = generator.episode_paths("fit", "native_plan", episode_id)
        raw_path = paths_for_episode.raw
        intent_path = paths_for_episode.intent
        raw_path.write_bytes(f"sealed raw fixture {episode_id}\n".encode())
        intent = {
            "schema_version": 1,
            "attempt": "v008",
            "created_unix_ns": 900_000 + int(assignment["slot"]),
            "status": "valid_in_memory_episode_ready_for_persistence",
            "role": "fit",
            "regime": "native_plan",
            **assignment,
            "seed_source_episode_id": episode_id,
            "replacement_used": False,
            "replacement_claim_index": None,
            "replacement_claim_sha256": None,
            "arrays": {},
            "initial_pixels_sha256": "1" * 64,
            "generation_audit": {"fixture": True},
            "authorization_seal": seal_link,
            "orphan_adoption_rule": "fixture exact-intent binding",
        }
        _write_json(intent_path, intent)
        record = generator._record_from_intent(paths_for_episode, intent, raw_audit)
        record["created_unix_ns"] = 1_000_000 + int(assignment["slot"])
        _write_json(paths_for_episode.sidecar, record)
        records.append(record)
    replacements = tuple(
        ledger["regimes"]["native_plan"]["roles"]["fit"]["replacements"]
    )
    inputs = generator.GenerationInputs(
        dgp_matrix=dgp,
        seed_ledger=ledger,
        seal=seal,
        regime_specification=dgp["regimes"]["native_plan"],
        primary=tuple(assignments),
        replacements=replacements,
    )
    manifest = {
        "schema_version": 1,
        "attempt": "v008",
        "created_unix_ns": 2_000_000,
        "complete": True,
        "role": "fit",
        "regime": "native_plan",
        "regime_specification": dgp["regimes"]["native_plan"],
        "episode_count": len(records),
        "episodes": records,
        "replacements_used": 0,
        "raw_archive_members": ["action", "pixels"],
        "raw_pixels_contract": "uint8 [201,224,224,3]",
        "raw_action_contract": "float32 [201,5]; 200 finite plus terminal NaN",
        "role_isolation": True,
        "smoke_permanently_excluded": False,
        "retention_contract": "pixels_action_shapes_finiteness_and_local_step_count_only",
        "dgp_matrix_path": _hardening_relative(repo, dgp_path),
        "dgp_matrix_sha256": _sha256(dgp_path),
        "cohort_seed_ledger_path": _hardening_relative(repo, ledger_path),
        "cohort_seed_ledger_sha256": _sha256(ledger_path),
        "authorization_seal": seal_link,
        "replacement_registry_path": _hardening_relative(repo, registry_path),
        "replacement_registry_scope": "append_only_all_roles_and_all_dgps",
        "orphan_policy": generator.RAW_MANIFEST_ORPHAN_POLICY,
    }
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    monkeypatch.setattr(runner, "STUDY_ROOT", study)
    monkeypatch.setattr(runner, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(runner, "INHERITANCE_SEAL_PATH", seal_path)
    monkeypatch.setattr(study_common, "REPO_ROOT", repo)
    monkeypatch.setattr(study_common, "STUDY_ROOT", study)
    monkeypatch.setattr(study_common, "ATTEMPT_ROOT", attempt)
    return {
        "repo": repo,
        "study": study,
        "attempt": attempt,
        "raw_root": raw_root,
        "intent_root": intent_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "inputs": inputs,
    }


def test_raw_manifest_is_exactly_ledger_bound_canonical_and_link_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each named attack must independently fail before any raw array is opened."""

    env = _install_strict_raw_manifest_fixture(tmp_path, monkeypatch)
    manifest_path: Path = env["manifest_path"]
    baseline = copy.deepcopy(env["manifest"])
    assert runner._load_raw_manifest(
        "fit", "native_plan", manifest_path, inputs=env["inputs"]
    )[
        "episode_count"
    ] == 300

    failures: list[str] = []

    def run_attack(name: str, mutate: Callable[[], Callable[[], None]]) -> None:
        cleanup = mutate()
        try:
            try:
                runner._load_raw_manifest(
                    "fit", "native_plan", manifest_path, inputs=env["inputs"]
                )
            except RuntimeError:
                return
            failures.append(name)
        finally:
            cleanup()
            _write_json(manifest_path, baseline)
            # Cleanup must restore a genuinely accepted baseline, not merely a
            # shape-compatible object for the next attack.
            assert runner._load_raw_manifest(
                "fit", "native_plan", manifest_path, inputs=env["inputs"]
            )[
                "episode_count"
            ] == 300

    def reorder() -> Callable[[], None]:
        value = copy.deepcopy(baseline)
        value["episodes"][0], value["episodes"][1] = (
            value["episodes"][1],
            value["episodes"][0],
        )
        _write_json(manifest_path, value)
        return lambda: None

    def wrong_ledger_id() -> Callable[[], None]:
        value = copy.deepcopy(baseline)
        old = value["episodes"][0]
        new_id = "drgv001-np-ft-p-999"
        new_raw = env["raw_root"] / f"{new_id}.npz"
        new_intent = env["intent_root"] / f"{new_id}.json"
        new_sidecar = new_raw.with_suffix(".json")
        shutil.copyfile(env["repo"] / old["raw_path"], new_raw)
        shutil.copyfile(env["repo"] / old["persistence_intent_path"], new_intent)
        changed = dict(old)
        changed.update(
            {
                "episode_id": new_id,
                "seed_source_episode_id": new_id,
                "raw_path": _hardening_relative(env["repo"], new_raw),
                "raw_sha256": _sha256(new_raw),
                "persistence_intent_path": _hardening_relative(env["repo"], new_intent),
                "persistence_intent_sha256": _sha256(new_intent),
            }
        )
        value["episodes"][0] = changed
        _write_json(new_sidecar, changed)
        _write_json(manifest_path, value)

        def cleanup() -> None:
            for path in (new_raw, new_intent, new_sidecar):
                path.unlink(missing_ok=True)

        return cleanup

    def wrong_seed() -> Callable[[], None]:
        value = copy.deepcopy(baseline)
        changed = dict(value["episodes"][0])
        changed["env_seed"] += 7
        value["episodes"][0] = changed
        sidecar = env["repo"] / changed["raw_path"]
        sidecar = sidecar.with_suffix(".json")
        original = json.loads(sidecar.read_text(encoding="utf-8"))
        _write_json(sidecar, changed)
        _write_json(manifest_path, value)
        return lambda: _write_json(sidecar, original)

    def noncanonical_raw_path() -> Callable[[], None]:
        value = copy.deepcopy(baseline)
        changed = dict(value["episodes"][0])
        alias = env["raw_root"].parent / "raw-alias.npz"
        shutil.copyfile(env["repo"] / changed["raw_path"], alias)
        changed["raw_path"] = _hardening_relative(env["repo"], alias)
        changed["raw_sha256"] = _sha256(alias)
        changed["raw_bytes"] = alias.stat().st_size
        value["episodes"][0] = changed
        sidecar = (env["repo"] / baseline["episodes"][0]["raw_path"]).with_suffix(
            ".json"
        )
        original = json.loads(sidecar.read_text(encoding="utf-8"))
        _write_json(sidecar, changed)
        _write_json(manifest_path, value)

        def cleanup() -> None:
            _write_json(sidecar, original)
            alias.unlink(missing_ok=True)

        return cleanup

    def noncanonical_intent_path() -> Callable[[], None]:
        value = copy.deepcopy(baseline)
        changed = dict(value["episodes"][0])
        source = env["repo"] / changed["persistence_intent_path"]
        alias = env["intent_root"].parent / "intent-alias.json"
        shutil.copyfile(source, alias)
        changed["persistence_intent_path"] = _hardening_relative(env["repo"], alias)
        changed["persistence_intent_sha256"] = _sha256(alias)
        value["episodes"][0] = changed
        sidecar = (env["repo"] / changed["raw_path"]).with_suffix(".json")
        original = json.loads(sidecar.read_text(encoding="utf-8"))
        _write_json(sidecar, changed)
        _write_json(manifest_path, value)

        def cleanup() -> None:
            _write_json(sidecar, original)
            alias.unlink(missing_ok=True)

        return cleanup

    def sidecar_drift() -> Callable[[], None]:
        record = baseline["episodes"][0]
        sidecar = (env["repo"] / record["raw_path"]).with_suffix(".json")
        original = json.loads(sidecar.read_text(encoding="utf-8"))
        changed = dict(original)
        changed["raw_bytes"] += 1
        _write_json(sidecar, changed)
        return lambda: _write_json(sidecar, original)

    def hardlinked_raw() -> Callable[[], None]:
        raw = env["repo"] / baseline["episodes"][0]["raw_path"]
        alias = raw.with_name(f".{raw.name}.hardlink")
        os.link(raw, alias)
        return lambda: alias.unlink(missing_ok=True)

    def symlinked_raw_parent() -> Callable[[], None]:
        raw_root: Path = env["raw_root"]
        real = raw_root.with_name("raw-real")
        raw_root.rename(real)
        raw_root.symlink_to(real.name, target_is_directory=True)

        def cleanup() -> None:
            raw_root.unlink(missing_ok=True)
            real.rename(raw_root)

        return cleanup

    def symlinked_manifest() -> Callable[[], None]:
        real = manifest_path.with_name("raw_manifest.real.json")
        manifest_path.rename(real)
        manifest_path.symlink_to(real.name)

        def cleanup() -> None:
            manifest_path.unlink(missing_ok=True)
            real.rename(manifest_path)

        return cleanup

    def manifest_claim_drift(field: str) -> Callable[[], Callable[[], None]]:
        def mutate() -> Callable[[], None]:
            value = copy.deepcopy(baseline)
            value[field] = "tampered-but-schema-compatible"
            _write_json(manifest_path, value)
            return lambda: None

        return mutate

    for name, attack in (
        ("ledger_order", reorder),
        ("ledger_episode_id", wrong_ledger_id),
        ("ledger_seed_tuple", wrong_seed),
        ("canonical_raw_path", noncanonical_raw_path),
        ("canonical_intent_path", noncanonical_intent_path),
        ("sidecar_exact_equality", sidecar_drift),
        ("raw_hardlink", hardlinked_raw),
        ("symlinked_raw_parent", symlinked_raw_parent),
        ("symlinked_manifest", symlinked_manifest),
        ("raw_pixels_contract", manifest_claim_drift("raw_pixels_contract")),
        ("raw_action_contract", manifest_claim_drift("raw_action_contract")),
        ("retention_contract", manifest_claim_drift("retention_contract")),
        ("orphan_policy", manifest_claim_drift("orphan_policy")),
    ):
        run_attack(name, attack)
    assert failures == [], f"raw manifest attacks accepted: {failures}"


def _record_two_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    Path,
    list[dict[str, Any]],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    generator.SealLink,
]:
    path = tmp_path / "rollout_failures.jsonl"
    monkeypatch.setattr(generator, "failure_log_path", lambda role, regime: path)
    destination = {
        "slot": 0,
        "episode_id": "drgv001-np-ft-p-000",
        "env_seed": 1,
        "policy_seed": 2,
        "oracle_np_seed": 3,
        "action_space_seed": 4,
    }
    replacements = tuple(
        {
            **destination,
            "episode_id": f"drgv001-np-ft-r-{index:03d}",
            "env_seed": 10 + index,
            "policy_seed": 20 + index,
            "oracle_np_seed": 30 + index,
            "action_space_seed": 40 + index,
            "replacement_claim_index": index,
            "replacement_claim_sha256": f"{index + 1:064x}",
        }
        for index in range(2)
    )
    primary = (destination,)
    seal = generator.SealLink(
        path=tmp_path / "seal.json",
        relative_path="runs/study/attempts/v008/audit/seal.json",
        sha256="a" * 64,
        checkpoint_state="PRE_OUTCOME_SEAL",
        payload={},
    )
    for index, candidate in enumerate(replacements):
        try:
            raise generator.EpisodeRolloutError(f"synthetic rollout {index}")
        except generator.EpisodeRolloutError as error:
            generator.record_rollout_failure(
                role="fit",
                regime="native_plan",
                destination=destination,
                candidate=candidate,
                replacement_used=True,
                error=error,
                primary=primary,
                replacements=replacements,
                seal=seal,
            )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return path, records, primary, replacements, seal


def test_rollout_failure_log_is_canonical_hash_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, records, _, _, _ = _record_two_failures(tmp_path, monkeypatch)
    assert [record["seq"] for record in records] == [1, 2]
    assert isinstance(records[0]["prev_sha256"], str)
    assert records[1]["prev_sha256"] == records[0]["record_sha256"]
    for record in records:
        payload = dict(record)
        observed = payload.pop("record_sha256")
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert observed == expected


def test_rollout_failure_log_rejects_in_place_chain_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, records, primary, replacements, seal = _record_two_failures(
        tmp_path, monkeypatch
    )
    records[0]["exception_message"] = "tampered after append"
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    with pytest.raises(generator.EpisodePersistenceError):
        generator.read_rollout_failures(
            "fit",
            "native_plan",
            primary=primary,
            replacements=replacements,
            seal=seal,
        )


def test_generator_direct_entrypoint_verifies_runtime_before_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    class RuntimeRejected(RuntimeError):
        pass

    class InputsTouched(RuntimeError):
        pass

    def verify(role: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(("verify", (role, kwargs)))
        raise RuntimeRejected("synthetic generation-runtime mismatch")

    def load(role: str, regime: str) -> generator.GenerationInputs:
        calls.append(("inputs", (role, regime)))
        raise InputsTouched("sealed inputs touched before runtime verification")

    monkeypatch.setattr(generator, "verify_here", verify, raising=False)
    monkeypatch.setattr(generator, "load_generation_inputs", load)
    with pytest.raises(RuntimeRejected):
        generator.generate("fit", "native_plan")
    assert calls == [
        ("verify", ("generation", {"include_external_hashes": True}))
    ]


@pytest.mark.parametrize("entrypoint", ["development", "confirmation"])
def test_runner_direct_sparse_entrypoint_rejects_non_mps_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    calls: list[tuple[str, Any]] = []

    class MPSRejected(RuntimeError):
        pass

    class AuthorizationTouched(RuntimeError):
        pass

    monkeypatch.setattr(runner, "ATTEMPT_ROOT", tmp_path / "attempt")
    monkeypatch.setattr(
        runner,
        "verify_here",
        lambda *args, **kwargs: calls.append(("verify", (args, kwargs))) or {},
        raising=False,
    )

    def require(device: str) -> str:
        calls.append(("require_mps", device))
        raise MPSRejected("synthetic non-MPS request")

    monkeypatch.setattr(runner, "require_mps_device", require, raising=False)
    monkeypatch.setattr(
        runner,
        "_state_authorization",
        lambda role: (_ for _ in ()).throw(
            AuthorizationTouched("authorization touched before MPS check")
        ),
    )
    with pytest.raises(MPSRejected):
        if entrypoint == "development":
            runner.execute_development("fit", "native_plan", "cpu")
        else:
            runner.execute_confirmation(
                "smoke", "native_plan", tmp_path / "gate.npz", "cpu"
            )
    assert ("require_mps", "cpu") in calls


@pytest.mark.parametrize("entrypoint", ["development", "confirmation"])
def test_runner_direct_sparse_entrypoint_runs_mps_runtime_preflight_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    calls: list[tuple[str, Any]] = []

    class RuntimeRejected(RuntimeError):
        pass

    class AuthorizationTouched(RuntimeError):
        pass

    monkeypatch.setattr(runner, "ATTEMPT_ROOT", tmp_path / "attempt")
    monkeypatch.setattr(
        runner,
        "require_mps_device",
        lambda value: calls.append(("require_mps", value)) or "mps",
        raising=False,
    )

    def verify(role: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(("verify", (role, kwargs)))
        raise RuntimeRejected("synthetic sparse-runtime mismatch")

    monkeypatch.setattr(runner, "verify_here", verify, raising=False)
    monkeypatch.setattr(
        runner,
        "_state_authorization",
        lambda role: (_ for _ in ()).throw(
            AuthorizationTouched("authorization touched before runtime preflight")
        ),
    )
    with pytest.raises(RuntimeRejected):
        if entrypoint == "development":
            runner.execute_development("fit", "native_plan", "mps")
        else:
            runner.execute_confirmation(
                "smoke", "native_plan", tmp_path / "gate.npz", "mps"
            )
    assert ("require_mps", "mps") in calls
    assert (
        "verify",
        (
            "sparse_execution",
            {"require_mps": True, "include_external_hashes": True},
        ),
    ) in calls


def test_generator_role_regime_lock_is_nonblocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    calls_second = threading.Event()
    first_inside = threading.Event()
    release_first = threading.Event()
    first_errors: list[BaseException] = []

    class StopFirst(RuntimeError):
        pass

    class SecondReachedProtectedRegion(RuntimeError):
        pass

    inputs = generator.GenerationInputs(
        dgp_matrix={},
        seed_ledger={},
        seal=generator.SealLink(
            path=attempt / "seal.json",
            relative_path="seal.json",
            sha256="0" * 64,
            checkpoint_state="PRE_OUTCOME_SEAL",
            payload={},
        ),
        regime_specification={},
        primary=(),
        replacements=(),
    )
    monkeypatch.setattr(generator, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(generator, "REPLACEMENT_REGISTRY_PATH", attempt / "registry.json")
    monkeypatch.setattr(
        generator, "REPLACEMENT_REGISTRY_LOCK_PATH", attempt / "registry.lock"
    )
    monkeypatch.setattr(
        generator,
        "verify_here",
        lambda *args, **kwargs: {"passed": True},
        raising=False,
    )
    monkeypatch.setattr(generator, "load_generation_inputs", lambda *args: inputs)
    monkeypatch.setattr(generator, "preflight_output_namespace", lambda *args: {})

    def validate(*args: Any) -> None:
        if threading.current_thread().name == "first-generator":
            first_inside.set()
            assert release_first.wait(5)
            raise StopFirst("release first generator")
        calls_second.set()
        raise SecondReachedProtectedRegion("second generator passed the lock boundary")

    monkeypatch.setattr(generator, "validate_existing_manifest", validate)

    def first() -> None:
        try:
            generator.generate("fit", "native_plan")
        except BaseException as error:  # recorded and asserted in the owner thread
            first_errors.append(error)

    thread = threading.Thread(target=first, name="first-generator", daemon=True)
    thread.start()
    assert first_inside.wait(5)
    started = time.monotonic()
    second_error: BaseException | None = None
    try:
        generator.generate("fit", "native_plan")
    except BaseException as error:
        second_error = error
    finally:
        release_first.set()
        thread.join(5)
    assert time.monotonic() - started < 1.0
    assert calls_second.is_set() is False
    assert isinstance(second_error, RuntimeError)
    assert any(token in str(second_error).lower() for token in ("lock", "busy", "concurrent"))
    assert len(first_errors) == 1 and isinstance(first_errors[0], StopFirst)
    assert not thread.is_alive()


@pytest.mark.parametrize("entrypoint", ["development", "confirmation"])
def test_runner_role_regime_lock_is_nonblocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    first_inside = threading.Event()
    release_first = threading.Event()
    second_inside = threading.Event()
    first_errors: list[BaseException] = []

    class StopFirst(RuntimeError):
        pass

    class SecondReachedProtectedRegion(RuntimeError):
        pass

    monkeypatch.setattr(runner, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(generator, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(
        runner,
        "verify_here",
        lambda *args, **kwargs: {"passed": True, "mps": {"available": True}},
        raising=False,
    )
    monkeypatch.setattr(
        runner, "require_mps_device", lambda value: "mps", raising=False
    )
    monkeypatch.setattr(runner, "_state_authorization", lambda role: {"role": role})
    inputs = generator.GenerationInputs(
        dgp_matrix={},
        seed_ledger={},
        seal=generator.SealLink(
            path=attempt / "seal.json",
            relative_path="seal.json",
            sha256="0" * 64,
            checkpoint_state="PRE_OUTCOME_SEAL",
            payload={},
        ),
        regime_specification={},
        primary=(),
        replacements=(),
    )
    monkeypatch.setattr(runner, "load_generation_inputs", lambda *args: inputs)
    monkeypatch.setattr(runner, "preflight_output_namespace", lambda *args: {})

    def protected(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if threading.current_thread().name == "first-runner":
            first_inside.set()
            assert release_first.wait(5)
            raise StopFirst("release first runner")
        second_inside.set()
        raise SecondReachedProtectedRegion("second runner passed the lock boundary")

    if entrypoint == "development":
        monkeypatch.setattr(runner, "_execute_development_locked", protected)
    else:
        monkeypatch.setattr(runner, "_execute_confirmation_locked", protected)

    def invoke() -> dict[str, Any]:
        if entrypoint == "development":
            return runner.execute_development("fit", "native_plan", "mps")
        return runner.execute_confirmation(
            "smoke", "native_plan", attempt / "gate.npz", "mps"
        )

    def first() -> None:
        try:
            invoke()
        except BaseException as error:
            first_errors.append(error)

    thread = threading.Thread(target=first, name="first-runner", daemon=True)
    thread.start()
    assert first_inside.wait(5)
    started = time.monotonic()
    second_error: BaseException | None = None
    try:
        invoke()
    except BaseException as error:
        second_error = error
    finally:
        release_first.set()
        thread.join(5)
    assert time.monotonic() - started < 1.0
    assert second_inside.is_set() is False
    assert isinstance(second_error, RuntimeError)
    assert any(token in str(second_error).lower() for token in ("lock", "busy", "concurrent"))
    assert len(first_errors) == 1 and isinstance(first_errors[0], StopFirst)
    assert not thread.is_alive()


def _install_controller_snapshot(
    state: dict[str, Any],
    paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    state_path = paths["study"] / "STATE.json"
    lock_path = paths["study"] / ".program.lock"
    _write_json(state_path, state)
    lock_path.touch()
    monkeypatch.setattr(inherited, "STATE_PATH", state_path)
    monkeypatch.setattr(inherited, "PROGRAM_LOCK_PATH", lock_path, raising=False)
    monkeypatch.setattr(inherited, "CONTROLLER_LOCK_PATH", lock_path, raising=False)
    monkeypatch.setattr(
        study_common, "read_verified_controller", lambda: copy.deepcopy(state)
    )
    monkeypatch.setattr(
        inherited,
        "_read_verified_controller_via_adapter",
        lambda: copy.deepcopy(state),
    )
    return state_path, lock_path


def test_inherited_authorization_rejects_controller_snapshot_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    state_path, _ = _install_controller_snapshot(state, paths, monkeypatch)
    drifted = copy.deepcopy(state)
    drifted["current_state"] = "CANDIDATE_SELECTION"
    _write_json(state_path, drifted)
    with pytest.raises(inherited.InheritedAuthorizationError):
        inherited.verify_inherited_pre_data_authorization(
            expected_current_state="SELECTION_COHORTS",
            seal_path=paths["seal"],
            ledger_path=paths["ledger"],
            genesis_path=paths["genesis"],
        )


def test_inherited_authorization_holds_program_lock_across_snapshot_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    _, lock_path = _install_controller_snapshot(state, paths, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    original = inherited._verify_sealed_files

    def pause(value: Any) -> dict[str, str]:
        entered.set()
        assert release.wait(5)
        return original(value)

    monkeypatch.setattr(inherited, "_verify_sealed_files", pause)

    def verify() -> None:
        try:
            inherited.verify_inherited_pre_data_authorization(
                expected_current_state="SELECTION_COHORTS",
                seal_path=paths["seal"],
                ledger_path=paths["ledger"],
                genesis_path=paths["genesis"],
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=verify, name="inherited-verifier", daemon=True)
    thread.start()
    assert entered.wait(5)
    descriptor = os.open(lock_path, os.O_RDWR)
    writer_blocked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            writer_blocked = True
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
        release.set()
        thread.join(5)
    assert writer_blocked is True
    assert errors == []
    assert not thread.is_alive()


def test_injected_authorization_state_must_equal_locked_disk_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    injected = copy.deepcopy(state)
    injected["unpersisted_injected_field"] = True
    with pytest.raises(
        inherited.InheritedAuthorizationError,
        match="current locked STATE.json differ",
    ):
        _verify(injected, paths)


@pytest.mark.parametrize(
    ("state_name", "contract_name"),
    (
        ("CANDIDATE_SELECTION", "verifier_contract_no_candidate.json"),
        (
            "CONFIRMATION_POWER_AND_COHORT_FREEZE",
            "verifier_contract_power_infeasible.json",
        ),
    ),
)
def test_authorized_early_guard_does_not_reacquire_root_exclusive_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_name: str,
    contract_name: str,
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    state.update(
        {
            "current_state": state_name,
            "expected_fit_episode_count": 1200,
            "expected_selection_episode_count": 2000,
            "fit_outcome_episodes": 1200,
            "selection_outcome_episodes": 2000,
        }
    )
    state["v008_durable_controller_adapter"] = _bind_adapter_marker(
        state, state["v008_durable_controller_adapter"]
    )
    _json(paths["state"], state)
    contract = paths["v008"] / contract_name
    _json(contract, {"schema_version": 1, "attempt": "v008"})
    errors: list[BaseException] = []
    results: list[dict[str, Any]] = []

    def verify() -> None:
        try:
            results.append(
                inherited.verify_inherited_pre_data_authorization(
                    expected_current_state=state_name,
                    state=state,
                    seal_path=paths["seal"],
                    ledger_path=paths["ledger"],
                    genesis_path=paths["genesis"],
                    authorized_early_guard_token=inherited._EARLY_ROOT_GUARD_TOKEN,
                    authorized_early_verifier_contract_path=contract,
                )
            )
        except BaseException as error:
            errors.append(error)

    descriptor = os.open(paths["lock"], os.O_RDWR)
    thread = threading.Thread(target=verify, daemon=True)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        thread.start()
        thread.join(3)
        completed_while_exclusive_held = not thread.is_alive()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    thread.join(3)
    assert completed_while_exclusive_held is True
    assert errors == []
    assert results[0]["state"] == state_name
    assert not thread.is_alive()


def test_workflow_rejects_selection_seal_without_inheritance_crosslink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    attempt = repo / "runs/lewm_domain_robust_gate/attempts/v008"
    inheritance = attempt / "audit/pre_data_inheritance_seal.json"
    selection = attempt / "audit/pre_selection_seal.json"
    _json(
        inheritance,
        {
            "schema_version": 1,
            "attempt": "v008",
            "science_attempt": "v001",
            "checkpoint_state": "PRE_OUTCOME_SEAL",
            "passed": True,
        },
    )
    _json(
        selection,
        {
            "schema_version": 1,
            "attempt": "v008",
            "checkpoint_state": "PRE_SELECTION_SEAL",
            "passed": True,
            "pre_data_seal_sha256": "0" * 64,
        },
    )
    inheritance_relative = _relative(repo, inheritance)
    receipt_path = attempt / "audit/version_forward_transaction_receipt.json"
    _json(receipt_path, {"passed": True})
    receipt = {
        "path": _relative(repo, receipt_path),
        "sha256": _sha(receipt_path),
        "passed": True,
    }
    monkeypatch.setattr(workflow, "REPO_ROOT", repo)
    monkeypatch.setattr(workflow, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(
        workflow,
        "ROLE_RAW_SEALS",
        {"fit": (inheritance, "PRE_OUTCOME_SEAL"), "selection": (inheritance, "PRE_OUTCOME_SEAL")},
    )
    monkeypatch.setattr(
        workflow,
        "ROLE_EXECUTION_SEALS",
        {"fit": (inheritance, "PRE_OUTCOME_SEAL"), "selection": (selection, "PRE_SELECTION_SEAL")},
    )
    monkeypatch.setattr(
        workflow,
        "verify_inherited_pre_data_authorization",
        lambda **_kwargs: {
            "seal_path": inheritance_relative,
            "seal_sha256": _sha(inheritance),
            "version_forward_transaction_receipt": receipt,
            "independent_recomputation": {
                "authorized_role_count_recovery": None,
            },
        },
    )
    with pytest.raises(workflow.WorkflowError, match="execution authorization seal"):
        workflow.verify_role_manifests(
            "selection",
            {
                "active_attempt": "v008",
                "current_state": "SELECTION_COHORTS",
                "expected_selection_episode_count": 2000,
            },
        )


@pytest.mark.parametrize(
    "state_name",
    (
        "CANDIDATE_SELECTION",
        "CONFIRMATION_POWER_AND_COHORT_FREEZE",
    ),
)
def test_independent_post_forward_early_state_requires_exact_guard_and_prefix(
    state_name: str,
) -> None:
    state_machine = (
        "BOOTSTRAP_AUDIT",
        "DIAGNOSTIC_ACCOUNT",
        "PREREGISTRATION_AND_POWER",
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
        "TERMINAL",
        "POST_TERMINAL_REPORTING",
    )
    index = state_machine.index(state_name)
    state = {
        "active_attempt": "v008",
        "current_state": state_name,
        "state_machine": list(state_machine),
        "completed_states": list(state_machine[:index]),
        "expected_fit_episode_count": 1200,
        "expected_selection_episode_count": 2000,
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": 2000,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
        "early_scientific_failure": None,
        "skipped_states": [],
    }

    version_forward._validate_post_forward_raw_authorization_state(
        state,
        authorized_early_verifier_state=state_name,
    )
    with pytest.raises(
        version_forward.VerificationError,
        match="not selection raw",
    ):
        version_forward._validate_post_forward_raw_authorization_state(state)
    other = (
        "CONFIRMATION_POWER_AND_COHORT_FREEZE"
        if state_name == "CANDIDATE_SELECTION"
        else "CANDIDATE_SELECTION"
    )
    with pytest.raises(
        version_forward.VerificationError,
        match="authorized early-verifier state drift",
    ):
        version_forward._validate_post_forward_raw_authorization_state(
            state,
            authorized_early_verifier_state=other,
        )
    drifted = copy.deepcopy(state)
    drifted["completed_states"] = drifted["completed_states"][:-1]
    with pytest.raises(
        version_forward.VerificationError,
        match="completed-state prefix drift",
    ):
        version_forward._validate_post_forward_raw_authorization_state(
            drifted,
            authorized_early_verifier_state=state_name,
        )


def test_independent_verifier_rejects_selection_seal_crosslink_tamper(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "runs/study/attempts/v008"
    attempt.mkdir(parents=True)
    policy = independent.PathPolicy(
        tmp_path,
        "runs/study/attempts/v008",
        "runs/study",
    )
    sealed = attempt / "sealed.txt"
    sealed.write_bytes(b"sealed\n")
    sealed_relative = sealed.relative_to(tmp_path).as_posix()
    seal_path = attempt / "pre_selection_seal.json"
    seal = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "PRE_SELECTION_SEAL",
        "passed": True,
        "created_unix_ns": 1,
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
        "pre_data_seal_sha256": "0" * 64,
        "sealed_files": {sealed_relative: _sha(sealed)},
    }
    _json(seal_path, seal)
    counts = {
        "fit": 1200,
        "selection": 0,
        "smoke": 0,
        "confirmation_generated": 0,
        "confirmation_executed": 0,
        "confirmation_opened": False,
    }
    expected = "a" * 64
    with pytest.raises(
        independent.VerificationError,
        match="direct inheritance cross-link drift",
    ):
        independent.verify_seal(
            seal_path,
            policy,
            checkpoint_state="PRE_SELECTION_SEAL",
            expected_counts=counts,
            expected_pre_data_seal_sha256=expected,
        )

    seal["pre_data_seal_sha256"] = expected
    _json(seal_path, seal)
    verified = independent.verify_seal(
        seal_path,
        policy,
        checkpoint_state="PRE_SELECTION_SEAL",
        expected_counts=counts,
        expected_pre_data_seal_sha256=expected,
    )
    assert verified["object"]["pre_data_seal_sha256"] == expected


def _install_inherited_selection_count_crash(
    state: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    """Install an exact selection count commit after the inherited prefix."""

    pre_count = copy.deepcopy(state)
    pre_count.update(
        {
            "state_machine": list(independent.STATE_MACHINE),
            "current_state": "SELECTION_COHORTS",
            "completed_states": list(inherited.INHERITED_STATES),
            "expected_fit_episode_count": 1200,
            "expected_selection_episode_count": 2000,
            "fit_outcome_episodes": 1200,
            "selection_outcome_episodes": 0,
            "smoke_outcome_episodes": 0,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
            "early_scientific_failure": None,
            "postconfirmation_integrity_failure": None,
            "terminal_label": None,
            "confirmation_terminal": False,
            "skipped_states": [],
        }
    )

    anchor = {
        "event": "synthetic_recovery_anchor",
        "attempt": "v008",
        "created_unix_ns": 100,
        "seq": int(pre_count["ledger_event_count"]) + 1,
        "prev_sha256": pre_count["ledger_head_sha256"],
    }
    anchor["record_sha256"] = hashlib.sha256(_canonical(anchor)).hexdigest()
    pre_count.update(
        {
            "updated_unix_ns": 100,
            "ledger_event_count": anchor["seq"],
            "ledger_head_sha256": anchor["record_sha256"],
        }
    )
    pre_count_marker = dict(pre_count["v008_durable_controller_adapter"])
    pre_count_marker.update(
        {
            "ledger_event_count": anchor["seq"],
            "ledger_head_sha256": anchor["record_sha256"],
            "operation_sha256": "a" * 64,
        }
    )
    pre_count["v008_durable_controller_adapter"] = _bind_adapter_marker(
        pre_count, pre_count_marker
    )
    pre_count_marker = copy.deepcopy(
        pre_count["v008_durable_controller_adapter"]
    )
    paths["ledger"].write_bytes(
        paths["ledger"].read_bytes() + _canonical(anchor) + b"\n"
    )
    _json(paths["state"], pre_count)
    pre_count_sha256 = _sha(paths["state"])

    for regime in (
        "native_plan",
        "markov_oracle",
        "plan_action_noise_0p2",
        "plan_random_action_0p1",
    ):
        _json(
            paths["v008"] / "data/selection" / regime / "execution_manifest.json",
            {
                "attempt": "v008",
                "role": "selection",
                "regime": regime,
                "complete": True,
                "episode_count": 500,
                "authorization": {
                    "active_attempt": "v008",
                    "state": "SELECTION_COHORTS",
                    "state_sha256": pre_count_sha256,
                },
            },
        )

    count = {
        "event": "outcome_counts_updated",
        "attempt": "v008",
        "fields": {"selection_outcome_episodes": 2000},
        "prior_controller_adapter_marker": pre_count_marker,
        "created_unix_ns": 200,
        "seq": int(anchor["seq"]) + 1,
        "prev_sha256": anchor["record_sha256"],
    }
    controller_event = {
        key: copy.deepcopy(value)
        for key, value in count.items()
        if key not in {"seq", "prev_sha256"}
    }
    proposed_state = copy.deepcopy(pre_count)
    proposed_state.pop("v008_durable_controller_adapter")
    proposed_state["selection_outcome_episodes"] = 2000
    proposed_state["updated_unix_ns"] = 200
    proposed_state_sha256 = hashlib.sha256(
        _canonical(proposed_state)
    ).hexdigest()
    transaction_projection = {
        "schema_version": 1,
        "authorization_kind": (
            "receipt_anchored_v008_descendant_adapter_transaction"
        ),
        "base_state_sha256": hashlib.sha256(_canonical(pre_count)).hexdigest(),
        "proposed_state_sha256": proposed_state_sha256,
        "events": [controller_event],
    }
    count["v008_durable_adapter_transaction"] = {
        "schema_version": 1,
        "authorization_kind": (
            "receipt_anchored_v008_descendant_adapter_transaction"
        ),
        "transaction_id": hashlib.sha256(
            _canonical(transaction_projection)
        ).hexdigest(),
        "event_index": 0,
        "event_count": 1,
        "base_state_sha256": transaction_projection["base_state_sha256"],
        "proposed_state_sha256": proposed_state_sha256,
        "base_state": copy.deepcopy(pre_count),
    }
    count["record_sha256"] = hashlib.sha256(_canonical(count)).hexdigest()
    base_ledger_payload = paths["ledger"].read_bytes()
    count_event_suffix = _canonical(count) + b"\n"
    paths["ledger"].write_bytes(base_ledger_payload + count_event_suffix)
    current = copy.deepcopy(pre_count)
    current.update(
        {
            "selection_outcome_episodes": 2000,
            "updated_unix_ns": 200,
            "ledger_event_count": count["seq"],
            "ledger_head_sha256": count["record_sha256"],
        }
    )
    current_marker = dict(current["v008_durable_controller_adapter"])
    current_marker.update(
        {
            "ledger_event_count": count["seq"],
            "ledger_head_sha256": count["record_sha256"],
        }
    )
    current_without_marker = copy.deepcopy(current)
    current_without_marker.pop("v008_durable_controller_adapter")
    current_marker["operation_sha256"] = hashlib.sha256(
        _canonical(
            {
                "base_state_object_sha256": hashlib.sha256(
                    _canonical(pre_count)
                ).hexdigest(),
                "base_ledger_sha256": hashlib.sha256(
                    base_ledger_payload
                ).hexdigest(),
                "expected_suffix_sha256": hashlib.sha256(
                    count_event_suffix
                ).hexdigest(),
                "intended_state_object_sha256": hashlib.sha256(
                    _canonical(current_without_marker)
                ).hexdigest(),
            }
        )
    ).hexdigest()
    current["v008_durable_controller_adapter"] = _bind_adapter_marker(
        current, current_marker
    )
    _json(paths["state"], current)
    return current


def _verify_inherited_selection_count_recovery(
    state: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    role: str = "selection",
    guard: object = inherited._ROLE_COUNT_RECOVERY_GUARD_TOKEN,
) -> dict[str, Any]:
    return inherited.verify_inherited_pre_data_authorization(
        expected_current_state="SELECTION_COHORTS",
        state=state,
        seal_path=paths["seal"],
        ledger_path=paths["ledger"],
        genesis_path=paths["genesis"],
        authorized_role_count_recovery_guard_token=guard,
        authorized_role_count_recovery=role,
    )


def test_private_role_count_guard_authorizes_exact_post_count_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        inherited,
        "CONTROLLER_ADAPTER_PATH",
        inherited.TRANSACTION_SOURCE_PATH,
        raising=False,
    )
    current = _install_inherited_selection_count_crash(state, paths)

    events = [
        json.loads(line)
        for line in paths["ledger"].read_text(encoding="utf-8").splitlines()
    ]
    independent_proof = version_forward._verify_role_count_recovery_authorization(
        current,
        events,
        role="selection",
        target=paths["v008"],
        repo=paths["repo"],
    )
    result = _verify_inherited_selection_count_recovery(current, paths)
    proof = result["role_count_recovery_authorization"]
    assert proof["passed"] is True
    assert proof["role"] == "selection"
    assert proof["state"] == "SELECTION_COHORTS"
    assert proof["counter_field"] == "selection_outcome_episodes"
    assert proof["counter_value"] == 2000
    assert len(proof["execution_manifests"]) == 4
    assert proof["role_audit_sha256"] is None
    assert proof["no_later_output_conflicts"] is True
    assert independent_proof["pre_count_state_sha256"] == proof[
        "pre_count_state_sha256"
    ]
    assert independent_proof["count_update_event_sha256"] == proof[
        "count_update_event_sha256"
    ]
    assert independent_proof[
        "prior_controller_adapter_marker_sha256"
    ] == proof["prior_controller_adapter_marker_sha256"]
    assert result["independent_recomputation"][
        "authorized_role_count_recovery"
    ] == "selection"


@pytest.mark.parametrize("tamper", ("state_only", "valid_hex_operation"))
def test_independent_role_count_recovery_rejects_state_and_operation_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    current = _install_inherited_selection_count_crash(state, paths)
    if tamper == "state_only":
        current["expected_selection_episode_count"] += 1
    else:
        marker = dict(current["v008_durable_controller_adapter"])
        marker["operation_sha256"] = "d" * 64
        current["v008_durable_controller_adapter"] = _bind_adapter_marker(
            current, marker
        )
    _json(paths["state"], current)
    events = [
        json.loads(line)
        for line in paths["ledger"].read_text(encoding="utf-8").splitlines()
    ]

    with pytest.raises(version_forward.VerificationError):
        version_forward._verify_role_count_recovery_authorization(
            current,
            events,
            role="selection",
            target=paths["v008"],
            repo=paths["repo"],
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_guard",
        "wrong_guard",
        "wrong_role",
        "count_event",
        "prior_marker",
        "execution_manifest",
        "later_output",
        "bool_count",
        "numeric_open_flag",
    ),
)
def test_private_role_count_guard_rejects_ambiguous_or_later_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        inherited,
        "CONTROLLER_ADAPTER_PATH",
        inherited.TRANSACTION_SOURCE_PATH,
        raising=False,
    )
    current = _install_inherited_selection_count_crash(state, paths)
    role = "selection"
    guard: object = inherited._ROLE_COUNT_RECOVERY_GUARD_TOKEN

    if tamper == "missing_guard":
        with pytest.raises(inherited.InheritedAuthorizationError):
            _verify(current, paths)
        return
    if tamper == "wrong_guard":
        guard = object()
    elif tamper == "wrong_role":
        role = "fit"
    elif tamper == "count_event":
        records = [
            json.loads(line)
            for line in paths["ledger"].read_text(encoding="utf-8").splitlines()
        ]
        payload = dict(records[-1])
        payload.pop("record_sha256")
        payload["fields"] = {"selection_outcome_episodes": 1999}
        records[-1] = {
            **payload,
            "record_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        }
        paths["ledger"].write_bytes(
            b"".join(_canonical(record) + b"\n" for record in records)
        )
        current["ledger_head_sha256"] = records[-1]["record_sha256"]
        _json(paths["state"], current)
    elif tamper == "prior_marker":
        records = [
            json.loads(line)
            for line in paths["ledger"].read_text(encoding="utf-8").splitlines()
        ]
        payload = dict(records[-1])
        payload.pop("record_sha256")
        payload["prior_controller_adapter_marker"] = dict(
            payload["prior_controller_adapter_marker"]
        )
        payload["prior_controller_adapter_marker"]["operation_sha256"] = "d" * 64
        records[-1] = {
            **payload,
            "record_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        }
        paths["ledger"].write_bytes(
            b"".join(_canonical(record) + b"\n" for record in records)
        )
        current["ledger_head_sha256"] = records[-1]["record_sha256"]
        current_marker = dict(current["v008_durable_controller_adapter"])
        current_marker["ledger_head_sha256"] = records[-1]["record_sha256"]
        current["v008_durable_controller_adapter"] = current_marker
        _json(paths["state"], current)
    elif tamper == "execution_manifest":
        manifest_path = (
            paths["v008"]
            / "data/selection/native_plan/execution_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["authorization"]["state_sha256"] = "f" * 64
        _json(manifest_path, manifest)
    elif tamper == "later_output":
        _json(
            paths["v008"] / "audit/candidate_selection.json",
            {"passed": True},
        )
    elif tamper == "bool_count":
        current["selection_outcome_episodes"] = False
        _json(paths["state"], current)
    else:
        current["confirmation_outcomes_opened_for_analysis"] = 0
        _json(paths["state"], current)

    if tamper in {
        "wrong_role",
        "count_event",
        "prior_marker",
        "execution_manifest",
        "later_output",
        "bool_count",
        "numeric_open_flag",
    }:
        events = [
            json.loads(line)
            for line in paths["ledger"].read_text(encoding="utf-8").splitlines()
        ]
        with pytest.raises(version_forward.VerificationError):
            version_forward._verify_role_count_recovery_authorization(
                current,
                events,
                role=role,
                target=paths["v008"],
                repo=paths["repo"],
            )

    with pytest.raises(inherited.InheritedAuthorizationError):
        _verify_inherited_selection_count_recovery(
            current,
            paths,
            role=role,
            guard=guard,
        )


def test_ordinary_zero_count_selection_rejects_preexisting_role_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, paths = _lineage_fixture(tmp_path, monkeypatch)
    _json(
        paths["v008"] / "audit/selection_cohorts.json",
        {
            "schema_version": 1,
            "attempt": "v008",
            "checkpoint_state": "SELECTION_COHORTS",
            "passed": True,
        },
    )

    with pytest.raises(
        inherited.InheritedAuthorizationError,
        match="ordinary selection authorization found a pre-count role audit",
    ):
        _verify(state, paths)
