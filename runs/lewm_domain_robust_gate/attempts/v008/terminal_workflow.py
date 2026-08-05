#!/usr/bin/env python3
"""Fail-closed terminalization and claim-bounded reporting for v008.

This module is deliberately non-scientific.  It never opens an NPZ or any
other outcome array.  It authenticates the immutable JSON products written by
the producer and standalone verifier, builds the verifier's exhaustive input
path-set manifest, applies only the preregistered terminal mapping already
reproduced by the verifier, and writes deterministic post-terminal reports.

The normal confirmation path is::

    manifest -> capture_verifier.py -> finalize -> reports

``complete`` is resume-safe shorthand for ``finalize`` followed by ``reports``
and the final controller checkpoint.  The preregistered early no-candidate and
power-infeasible paths use the controller's narrow early-failure finalizers.
Any failure of the confirmation-mode independent verifier after the fixed
confirmation has been opened maps to execution-invalid, never to a repaired or
retried scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


ATTEMPT_ROOT = Path(__file__).resolve().parent
ACTIVE_ATTEMPT = ATTEMPT_ROOT.name
if ACTIVE_ATTEMPT != "v008":
    raise RuntimeError(
        f"terminal_workflow.py must run from v008, got {ACTIVE_ATTEMPT!r}"
    )
STUDY_ROOT = ATTEMPT_ROOT.parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
CONTROLLER = ATTEMPT_ROOT / "version_forward_transaction.py"

DGP_ORDER = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ENDPOINT_ORDER = ("raw", "fixed_whitened")
SCIENTIFIC_LABELS = (
    "domain_robust_gate_confirmed",
    "domain_robust_gate_partial",
    "domain_robust_gate_failed",
)
INVALID_LABEL = "domain_robust_gate_execution_invalid"
ALL_LABELS = (*SCIENTIFIC_LABELS, INVALID_LABEL)
EARLY_MODES = ("no_candidate", "power_infeasible")

MANIFEST_RELATIVE = "audit/terminal_input_manifest.json"
AUDIT_RELATIVE = "audit/independent_verification.json"
DECISION_RELATIVE = "decision.json"
REPORT_FILENAMES = (
    "REPORT.md",
    "ROBUSTNESS_MAP.json",
    "INDEPENDENT_AUDIT.md",
    "LIMITATIONS.md",
    "FOLLOW_ON_TASK.md",
)
REPORT_CHECKPOINT_RELATIVE = "audit/post_terminal_reporting.json"
SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE = (
    "audit/scientific_replay_qualification.json"
)
SCIENTIFIC_REPLAY_SOURCE_NAME = "scientific_replay.py"
SCIENTIFIC_REPLAY_LAUNCHER_NAME = "scientific_replay_launcher.py"
REPLAY_ROLE_ORDER = ("fit", "selection", "smoke", "confirmation")
REPLAY_ROLE_STATES = {
    "fit": "FIT_COHORTS",
    "selection": "SELECTION_COHORTS",
    "smoke": "EXCLUDED_MECHANICAL_SMOKE",
    "confirmation": "CONFIRMATION_EXECUTION",
}
REPLAY_FIXED_EPISODES_PER_DGP = {"fit": 300, "selection": 500, "smoke": 6}
REPLAY_ROWS_PER_EPISODE = 38
REPLAY_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "attempt",
        "role",
        "regime",
        "raw_manifest",
        "execution_manifest",
        "compiled_gate",
        "episode_count",
        "row_count",
        "episodes",
        "aggregate",
        "runtime_audit",
        "source_hashes",
        "module_before",
        "module_after",
        "input_loader_agreement_exact",
        "every_persisted_tensor_exact",
        "aggregate_exact",
        "target_loss_computed",
        "loss_or_effect_used_for_acceptance",
        "contact_or_privileged_materialized",
        "no_gradients",
        "artifact_tree_unchanged",
        "read_only",
        "passed",
    }
)
REPLAY_EPISODE_KEYS = frozenset(
    {
        "slot",
        "episode_id",
        "raw",
        "raw_sidecar",
        "execution_part",
        "execution_sidecar",
        "input_loader_audit_sha256",
        "array_sha256",
        "every_persisted_tensor_exact",
    }
)
REPLAY_ROLE_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "attempt",
        "role",
        "state",
        "authorization_state_sha256",
        "manifest_verification_sha256",
        "regime_order",
        "episodes_per_regime",
        "episode_count",
        "row_count",
        "regimes",
        "source_hashes",
        "input_loader_agreement_exact",
        "every_persisted_tensor_exact",
        "all_development_aggregates_exact",
        "all_runtime_mps_exact",
        "target_loss_computed",
        "loss_or_effect_used_for_acceptance",
        "contact_or_privileged_materialized",
        "no_gradients",
        "artifact_trees_unchanged",
        "read_only",
        "passed",
    }
)
REPLAY_ROLE_REGIME_KEYS = frozenset(
    {
        "result",
        "raw_manifest",
        "execution_manifest",
        "aggregate",
        "episode_count",
        "row_count",
    }
)
REPLAY_DEVELOPMENT_ARRAY_KEYS = frozenset(
    {
        "episode_slot",
        "model_step",
        "target",
        "exits",
        "production_features",
        "history",
        "action_history",
        "stage_current",
        "stage_update",
    }
)
REPLAY_ROBUST_ARRAY_KEYS = frozenset(
    {
        "episode_slot",
        "model_step",
        "exits",
        "selected",
        "calls",
        "scores",
        "head_scores",
        "production_features",
        "history",
        "action_history",
        "stage_current",
        "stage_update",
        "reached",
        "dense_scores",
        "dense_head_scores",
        "dense_stage_current",
        "dense_stage_update",
    }
)
REPLAY_OUTCOME_COUNT_FIELDS = (
    "fit_outcome_episodes",
    "selection_outcome_episodes",
    "smoke_outcome_episodes",
    "confirmation_outcome_episodes_generated",
    "confirmation_outcome_episodes_executed",
    "confirmation_outcomes_opened_for_analysis",
)


class TerminalWorkflowError(RuntimeError):
    """An immutable input, chronology link, or terminal mapping is invalid."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalWorkflowError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise TerminalWorkflowError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    if pretty:
        encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False)
    else:
        encoded = json.dumps(
            dict(value), separators=(",", ":"), sort_keys=True, allow_nan=False
        )
    return encoded.encode("utf-8") + b"\n"


def _canonical_relative(raw: Any) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise TerminalWorkflowError("path is not a canonical nonempty POSIX path")
    value = PurePosixPath(raw)
    if (
        value.is_absolute()
        or "." in value.parts
        or ".." in value.parts
        or value.as_posix() != raw
        or raw.endswith("/")
    ):
        raise TerminalWorkflowError(f"noncanonical relative path: {raw}")
    return raw


def _repo_relative(path: Path, *, repo_root: Path | None = None) -> str:
    root = REPO_ROOT if repo_root is None else Path(repo_root)
    try:
        return Path(path).resolve(strict=False).relative_to(
            root.resolve(strict=True)
        ).as_posix()
    except ValueError as error:
        raise TerminalWorkflowError(f"path escapes repository: {path}") from error


def _attempt_relative(attempt_root: Path, repo_root: Path) -> str:
    return _repo_relative(attempt_root, repo_root=repo_root)


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Durably publish immutable bytes using a no-replace hard-link operation."""

    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable terminal artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _write_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise TerminalWorkflowError(f"immutable terminal artifact drift: {path}")
        return
    _write_exclusive(path, payload)
    if path.read_bytes() != payload:
        raise TerminalWorkflowError(f"terminal artifact publication failed: {path}")


def _stable_file_record(path: Path) -> dict[str, Any]:
    """Hash one regular non-link file and reject mutation during the read."""

    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TerminalWorkflowError(f"terminal input is not a regular file: {path}")
    digest = sha256_file(path)
    after = path.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise TerminalWorkflowError(f"terminal input changed while hashing: {path}")
    return {"sha256": digest, "bytes": int(after.st_size)}


def _strict_sha256(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise TerminalWorkflowError(f"{label} is not a lowercase SHA-256")
    return value


def _canonical_object_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TerminalWorkflowError("replay evidence is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _replay_repo_path(raw: Any, *, repo_root: Path, label: str) -> Path:
    relative = _canonical_relative(raw)
    root = Path(repo_root).resolve(strict=True)
    path = (root / relative).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise TerminalWorkflowError(f"{label} escapes repository") from error
    return path


def _verified_replay_link(
    value: Any,
    *,
    repo_root: Path,
    label: str,
    expected_path: str | None = None,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise TerminalWorkflowError(f"{label} link schema drift")
    raw = _canonical_relative(value.get("path"))
    if expected_path is not None and raw != expected_path:
        raise TerminalWorkflowError(f"{label} path drift")
    expected_hash = _strict_sha256(value.get("sha256"), f"{label} hash")
    path = _replay_repo_path(raw, repo_root=repo_root, label=label)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise TerminalWorkflowError(f"missing {label}: {path}") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or int(metadata.st_nlink) != 1
        or sha256_file(path) != expected_hash
    ):
        raise TerminalWorkflowError(f"{label} live identity drift")
    return {"path": raw, "sha256": expected_hash}


def _read_canonical_replay_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path)
    if path.read_bytes() != canonical_json_bytes(value, pretty=True):
        raise TerminalWorkflowError(f"{label} is not canonical pretty JSON")
    return value


def _expected_replay_roles(mode: str) -> tuple[str, ...]:
    if mode == "confirmation":
        return REPLAY_ROLE_ORDER
    if mode in EARLY_MODES:
        return ("fit", "selection")
    raise TerminalWorkflowError(f"unsupported replay mode: {mode}")


def _expected_replay_episodes_per_dgp(
    role: str, state: Mapping[str, Any]
) -> int:
    if role in REPLAY_FIXED_EPISODES_PER_DGP:
        value = REPLAY_FIXED_EPISODES_PER_DGP[role]
    elif role == "confirmation":
        total = state.get("expected_confirmation_episode_count")
        if type(total) is not int or total <= 0 or total % len(DGP_ORDER):
            raise TerminalWorkflowError(
                "confirmation replay count is not one fixed four-regime total"
            )
        value = total // len(DGP_ORDER)
    else:
        raise TerminalWorkflowError(f"unknown replay role: {role}")
    if type(value) is not int or value <= 0:
        raise TerminalWorkflowError(f"invalid replay count for {role}")
    return value


def _controller_replay_snapshot(
    state: Mapping[str, Any], *, attempt_root: Path, repo_root: Path
) -> dict[str, Any]:
    study_root = Path(attempt_root).parents[1]
    state_path = study_root / "STATE.json"
    ledger_path = study_root / "RESEARCH_LEDGER.jsonl"
    state_record = _stable_file_record(state_path)
    ledger_record = _stable_file_record(ledger_path)
    for path, label in ((state_path, "STATE"), (ledger_path, "research ledger")):
        metadata = path.lstat()
        if path.is_symlink() or int(metadata.st_nlink) != 1:
            raise TerminalWorkflowError(f"{label} is linked during scientific replay")
    observed_state = read_json(state_path)
    if observed_state != dict(state):
        raise TerminalWorkflowError(
            "controller status differs from STATE during scientific replay"
        )
    event_count = state.get("ledger_event_count")
    head = state.get("ledger_head_sha256")
    if type(event_count) is not int or event_count <= 0 or not isinstance(head, str) or not head:
        raise TerminalWorkflowError("scientific replay controller ledger identity drift")
    counts: dict[str, Any] = {}
    for field in REPLAY_OUTCOME_COUNT_FIELDS:
        value = state.get(field)
        if field == "confirmation_outcomes_opened_for_analysis":
            if type(value) is not bool:
                raise TerminalWorkflowError(f"controller count type drift: {field}")
        elif type(value) is not int or value < 0:
            raise TerminalWorkflowError(f"controller count type drift: {field}")
        counts[field] = value
    return {
        "active_attempt": state.get("active_attempt"),
        "current_state": state.get("current_state"),
        "state": {
            "path": _repo_relative(state_path, repo_root=repo_root),
            **state_record,
            "object_sha256": _canonical_object_sha256(observed_state),
        },
        "ledger": {
            "path": _repo_relative(ledger_path, repo_root=repo_root),
            **ledger_record,
            "event_count": event_count,
            "head_sha256": head,
        },
        "outcome_counts": counts,
    }


def _run_terminal_scientific_replay_cell(
    role: str,
    regime: str,
    *,
    attempt_root: Path = ATTEMPT_ROOT,
) -> dict[str, Any]:
    launcher = Path(attempt_root) / SCIENTIFIC_REPLAY_LAUNCHER_NAME
    if launcher.is_symlink() or not launcher.is_file():
        raise TerminalWorkflowError("scientific replay launcher is absent or linked")
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(launcher),
            "--role",
            role,
            "--regime",
            regime,
        ],
        cwd=attempt_root,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise TerminalWorkflowError(
            f"scientific replay failed for {role}/{regime}: {detail}"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise TerminalWorkflowError(
            f"scientific replay emitted non-single JSON: {role}/{regime}"
        )
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise TerminalWorkflowError(
            f"scientific replay emitted invalid JSON: {role}/{regime}"
        ) from error
    if not isinstance(value, dict):
        raise TerminalWorkflowError("scientific replay result is not an object")
    return value


def _expected_replay_array_keys(role: str) -> frozenset[str]:
    if role in ("fit", "selection"):
        return REPLAY_DEVELOPMENT_ARRAY_KEYS
    if role == "smoke":
        return REPLAY_ROBUST_ARRAY_KEYS
    if role == "confirmation":
        return REPLAY_ROBUST_ARRAY_KEYS | {"target"}
    raise TerminalWorkflowError(f"unknown replay role: {role}")


def _validate_terminal_replay_result(
    result: Mapping[str, Any],
    *,
    role: str,
    regime: str,
    episodes_per_dgp: int,
    attempt_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if set(result) != REPLAY_RESULT_KEYS:
        raise TerminalWorkflowError(f"replay result schema drift: {role}/{regime}")
    for field in ("schema_version", "episode_count", "row_count"):
        if type(result.get(field)) is not int:
            raise TerminalWorkflowError(
                f"replay result integer type drift: {role}/{regime}/{field}"
            )
    exact_bools = {
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_tree_unchanged": True,
        "read_only": True,
        "passed": True,
    }
    for field, expected in exact_bools.items():
        if type(result.get(field)) is not bool or result.get(field) is not expected:
            raise TerminalWorkflowError(
                f"replay result assertion drift: {role}/{regime}/{field}"
            )
    if (
        result.get("schema_version") != 1
        or result.get("artifact_type")
        != "v008_independent_scientific_replay_regime"
        or result.get("attempt") != ACTIVE_ATTEMPT
        or result.get("role") != role
        or result.get("regime") != regime
        or result.get("episode_count") != episodes_per_dgp
        or result.get("row_count") != episodes_per_dgp * REPLAY_ROWS_PER_EPISODE
    ):
        raise TerminalWorkflowError(f"replay result identity/count drift: {role}/{regime}")

    attempt_relative = _attempt_relative(attempt_root, repo_root)
    raw_relative = f"{attempt_relative}/data/{role}/{regime}/raw_manifest.json"
    execution_relative = (
        f"{attempt_relative}/data/{role}/{regime}/execution_manifest.json"
    )
    raw_link = _verified_replay_link(
        result.get("raw_manifest"),
        repo_root=repo_root,
        label=f"{role}/{regime} raw manifest",
        expected_path=raw_relative,
    )
    execution_link = _verified_replay_link(
        result.get("execution_manifest"),
        repo_root=repo_root,
        label=f"{role}/{regime} execution manifest",
        expected_path=execution_relative,
    )
    raw_manifest = _read_canonical_replay_object(
        _replay_repo_path(raw_relative, repo_root=repo_root, label="raw manifest"),
        f"{role}/{regime} raw manifest",
    )
    execution_manifest = _read_canonical_replay_object(
        _replay_repo_path(
            execution_relative, repo_root=repo_root, label="execution manifest"
        ),
        f"{role}/{regime} execution manifest",
    )
    for manifest, label in (
        (raw_manifest, "raw"),
        (execution_manifest, "execution"),
    ):
        if (
            manifest.get("attempt") != ACTIVE_ATTEMPT
            or manifest.get("role") != role
            or manifest.get("regime") != regime
            or manifest.get("complete") is not True
            or type(manifest.get("episode_count")) is not int
            or manifest.get("episode_count") != episodes_per_dgp
        ):
            raise TerminalWorkflowError(
                f"{role}/{regime} {label} manifest identity/count drift"
            )
    if (
        type(execution_manifest.get("row_count")) is not int
        or execution_manifest.get("row_count")
        != episodes_per_dgp * REPLAY_ROWS_PER_EPISODE
    ):
        raise TerminalWorkflowError(
            f"{role}/{regime} execution manifest row-count drift"
        )
    authorization = execution_manifest.get("authorization")
    if not isinstance(authorization, Mapping):
        raise TerminalWorkflowError(
            f"{role}/{regime} execution authorization is absent"
        )
    authorization_state_sha256 = _strict_sha256(
        authorization.get("state_sha256"),
        f"{role}/{regime} authorization-state hash",
    )

    runtime = result.get("runtime_audit")
    runtime_checks = runtime.get("checks") if isinstance(runtime, Mapping) else None
    mps = runtime.get("mps") if isinstance(runtime, Mapping) else None
    if not (
        isinstance(runtime, Mapping)
        and type(runtime.get("schema_version")) is int
        and runtime.get("schema_version") == 1
        and runtime.get("passed") is True
        and runtime.get("requested_role") == "independent_verification"
        and runtime.get("runtime_role") == "evaluation"
        and isinstance(runtime_checks, Mapping)
        and bool(runtime_checks)
        and all(type(value) is bool and value for value in runtime_checks.values())
        and isinstance(mps, Mapping)
        and set(mps) == {"required", "built", "available"}
        and all(type(mps[name]) is bool and mps[name] for name in mps)
        and runtime.get("read_only_preflight") is True
        and type(runtime.get("seed_tuples_consumed")) is int
        and runtime.get("seed_tuples_consumed") == 0
        and type(runtime.get("output_paths_created")) is int
        and runtime.get("output_paths_created") == 0
    ):
        raise TerminalWorkflowError(f"replay runtime/MPS drift: {role}/{regime}")
    module_before = result.get("module_before")
    if not (
        isinstance(module_before, Mapping)
        and module_before.get("passed") is True
        and result.get("module_after") == module_before
    ):
        raise TerminalWorkflowError(f"replay module audit drift: {role}/{regime}")

    sources = result.get("source_hashes")
    if not isinstance(sources, Mapping) or not sources:
        raise TerminalWorkflowError(f"replay source hashes absent: {role}/{regime}")
    verified_sources: dict[str, str] = {}
    for raw_path, digest in sources.items():
        relative = _canonical_relative(raw_path)
        expected_hash = _strict_sha256(
            digest, f"{role}/{regime} replay source {relative}"
        )
        source_path = _replay_repo_path(
            relative, repo_root=repo_root, label="replay source"
        )
        metadata = source_path.lstat()
        if (
            source_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or int(metadata.st_nlink) != 1
            or sha256_file(source_path) != expected_hash
        ):
            raise TerminalWorkflowError(
                f"replay source live identity drift: {relative}"
            )
        verified_sources[relative] = expected_hash
    required_sources = {
        f"{attempt_relative}/{SCIENTIFIC_REPLAY_SOURCE_NAME}",
        f"{attempt_relative}/{SCIENTIFIC_REPLAY_LAUNCHER_NAME}",
    }
    if not required_sources <= set(verified_sources):
        raise TerminalWorkflowError("replay source audit omits replay implementation")

    raw_episodes = raw_manifest.get("episodes")
    execution_episodes = execution_manifest.get("episodes")
    replay_episodes = result.get("episodes")
    if not (
        type(raw_episodes) is list
        and type(execution_episodes) is list
        and type(replay_episodes) is list
        and len(raw_episodes) == len(execution_episodes) == len(replay_episodes)
        == episodes_per_dgp
    ):
        raise TerminalWorkflowError(f"replay episode-list drift: {role}/{regime}")
    array_keys = _expected_replay_array_keys(role)
    episode_ids: list[str] = []
    file_index: list[dict[str, Any]] = []
    for slot, (raw_record, execution_record, replay_record) in enumerate(
        zip(raw_episodes, execution_episodes, replay_episodes, strict=True)
    ):
        if not (
            isinstance(raw_record, Mapping)
            and isinstance(execution_record, Mapping)
            and isinstance(replay_record, Mapping)
            and set(replay_record) == REPLAY_EPISODE_KEYS
            and type(replay_record.get("slot")) is int
            and replay_record.get("slot") == slot
            and type(raw_record.get("slot")) is int
            and raw_record.get("slot") == slot
            and type(execution_record.get("slot")) is int
            and execution_record.get("slot") == slot
            and isinstance(replay_record.get("episode_id"), str)
            and bool(replay_record["episode_id"])
            and replay_record.get("episode_id") == raw_record.get("episode_id")
            == execution_record.get("episode_id")
            and replay_record.get("every_persisted_tensor_exact") is True
        ):
            raise TerminalWorkflowError(
                f"replay episode identity drift: {role}/{regime}/{slot}"
            )
        _strict_sha256(
            replay_record.get("input_loader_audit_sha256"),
            f"{role}/{regime}/{slot} loader-audit hash",
        )
        hashes = replay_record.get("array_sha256")
        if not (
            isinstance(hashes, Mapping)
            and set(hashes) == array_keys
            and all(
                _strict_sha256(value, f"{role}/{regime}/{slot}/{name}")
                for name, value in hashes.items()
            )
        ):
            raise TerminalWorkflowError(
                f"replay array-digest schema drift: {role}/{regime}/{slot}"
            )
        raw_path = _canonical_relative(raw_record.get("raw_path"))
        raw_sidecar_path = PurePosixPath(raw_path).with_suffix(".json").as_posix()
        expected_links = {
            "raw": {"path": raw_path, "sha256": raw_record.get("raw_sha256")},
            "raw_sidecar": {
                "path": raw_sidecar_path,
                "sha256": execution_record.get("source_raw_sidecar_sha256"),
            },
            "execution_part": {
                "path": execution_record.get("path"),
                "sha256": execution_record.get("sha256"),
            },
            "execution_sidecar": {
                "path": execution_record.get("sidecar_path"),
                "sha256": execution_record.get("sidecar_sha256"),
            },
        }
        verified_links: dict[str, dict[str, str]] = {}
        for name, expected in expected_links.items():
            if replay_record.get(name) != expected:
                raise TerminalWorkflowError(
                    f"replay episode link drift: {role}/{regime}/{slot}/{name}"
                )
            verified_links[name] = _verified_replay_link(
                replay_record[name],
                repo_root=repo_root,
                label=f"{role}/{regime}/{slot} {name}",
            )
        episode_id = str(replay_record["episode_id"])
        episode_ids.append(episode_id)
        file_index.append({"episode_id": episode_id, **verified_links})

    aggregate = result.get("aggregate")
    if not isinstance(aggregate, Mapping) or set(aggregate) != {
        "applicable",
        "path",
        "sha256",
        "arrays",
        "exact",
    }:
        raise TerminalWorkflowError(f"replay aggregate schema drift: {role}/{regime}")
    if role in ("fit", "selection"):
        aggregate_arrays = aggregate.get("arrays")
        expected_aggregate_path = execution_manifest.get("aggregate_role_path")
        expected_aggregate_hash = execution_manifest.get("aggregate_role_sha256")
        if not (
            aggregate.get("applicable") is True
            and aggregate.get("path") == expected_aggregate_path
            and aggregate.get("sha256") == expected_aggregate_hash
            and aggregate.get("exact") is True
            and result.get("aggregate_exact") is True
            and isinstance(aggregate_arrays, Mapping)
            and set(aggregate_arrays)
            == {
                "episode_slot",
                "model_step",
                "target",
                "exits",
                "production_features",
            }
            and all(
                _strict_sha256(value, f"{role}/{regime} aggregate {name}")
                for name, value in aggregate_arrays.items()
            )
        ):
            raise TerminalWorkflowError(
                f"replay development aggregate drift: {role}/{regime}"
            )
        _verified_replay_link(
            {"path": aggregate["path"], "sha256": aggregate["sha256"]},
            repo_root=repo_root,
            label=f"{role}/{regime} aggregate",
        )
        if result.get("compiled_gate") is not None:
            raise TerminalWorkflowError(
                f"development replay unexpectedly binds a gate: {role}/{regime}"
            )
    else:
        if (
            dict(aggregate)
            != {
                "applicable": False,
                "path": None,
                "sha256": None,
                "arrays": None,
                "exact": None,
            }
            or result.get("aggregate_exact") is not None
        ):
            raise TerminalWorkflowError(
                f"robust replay aggregate must be inapplicable: {role}/{regime}"
            )
        gate = _verified_replay_link(
            result.get("compiled_gate"),
            repo_root=repo_root,
            label=f"{role}/{regime} compiled gate",
        )
        if (
            execution_manifest.get("compiled_gate_path") != gate["path"]
            or execution_manifest.get("compiled_gate_sha256") != gate["sha256"]
        ):
            raise TerminalWorkflowError(
                f"robust replay gate/manifest binding drift: {role}/{regime}"
            )

    return {
        "raw_manifest": raw_link,
        "execution_manifest": execution_link,
        "aggregate": dict(aggregate),
        "episode_count": episodes_per_dgp,
        "row_count": episodes_per_dgp * REPLAY_ROWS_PER_EPISODE,
        "authorization_state_sha256": authorization_state_sha256,
        "episode_ids_sha256": _canonical_object_sha256(episode_ids),
        "verified_file_index_sha256": _canonical_object_sha256(file_index),
        "source_hashes": dict(sorted(verified_sources.items())),
    }


def _development_replay_manifest_binding(
    role: str,
    *,
    attempt_root: Path,
) -> dict[str, Any]:
    checkpoint_name = {
        "fit": "fit_cohorts.json",
        "selection": "selection_cohorts.json",
    }.get(role)
    if checkpoint_name is None:
        raise TerminalWorkflowError(f"no development checkpoint for {role}")
    checkpoint_path = Path(attempt_root) / "audit" / checkpoint_name
    checkpoint = _read_canonical_replay_object(
        checkpoint_path, f"{role} role checkpoint"
    )
    evidence = checkpoint.get("evidence")
    verification = (
        evidence.get("role_verification")
        if isinstance(evidence, Mapping)
        else None
    )
    if not isinstance(verification, Mapping) or verification.get("role") != role:
        raise TerminalWorkflowError(f"{role} checkpoint role verification is absent")
    regimes = verification.get("regimes")
    if not isinstance(regimes, Mapping) or set(regimes) != set(DGP_ORDER):
        raise TerminalWorkflowError(f"{role} checkpoint regime verification drift")
    stable_regimes: dict[str, Any] = {}
    for regime in DGP_ORDER:
        value = regimes[regime]
        if not isinstance(value, Mapping):
            raise TerminalWorkflowError(
                f"{role}/{regime} checkpoint verification is not an object"
            )
        raw = value.get("raw_manifest")
        execution = value.get("execution_manifest")
        aggregate = value.get("aggregate")
        if not (
            isinstance(raw, Mapping)
            and set(raw) == {"path", "sha256"}
            and isinstance(execution, Mapping)
            and set(execution) == {"path", "sha256"}
            and isinstance(aggregate, Mapping)
            and set(aggregate) == {"path", "sha256", "bytes"}
            and type(aggregate.get("bytes")) is int
            and aggregate["bytes"] > 0
            and type(value.get("episode_count")) is int
            and type(value.get("row_count")) is int
        ):
            raise TerminalWorkflowError(
                f"{role}/{regime} stable manifest binding drift"
            )
        for link_name, link in (("raw", raw), ("execution", execution)):
            _strict_sha256(link.get("sha256"), f"{role}/{regime} {link_name}")
            _canonical_relative(link.get("path"))
        _strict_sha256(aggregate.get("sha256"), f"{role}/{regime} aggregate")
        episode_digest = _strict_sha256(
            value.get("episode_ids_sha256"),
            f"{role}/{regime} episode-ID digest",
        )
        index_digest = _strict_sha256(
            value.get("verified_file_index_sha256"),
            f"{role}/{regime} verified-file digest",
        )
        stable_regimes[regime] = {
            "raw_manifest": dict(raw),
            "execution_manifest": dict(execution),
            "aggregate": dict(aggregate),
            "episode_count": value["episode_count"],
            "row_count": value["row_count"],
            "episode_ids_sha256": episode_digest,
            "verified_file_index_sha256": index_digest,
        }
    authorization = _strict_sha256(
        verification.get("authorization_state_sha256"),
        f"{role} authorization-state hash",
    )
    stable = {
        "role": role,
        "state": REPLAY_ROLE_STATES[role],
        "regime_order": list(DGP_ORDER),
        "episodes_per_regime": verification.get("episodes_per_regime"),
        "episode_count": verification.get("episode_count"),
        "row_count": verification.get("row_count"),
        "authorization_state_sha256": authorization,
        "regimes": stable_regimes,
    }
    qualification = verification.get("scientific_replay_qualification")
    if not (
        isinstance(qualification, Mapping)
        and qualification.get("passed") is True
        and qualification.get("role") == role
        and qualification.get("state") == REPLAY_ROLE_STATES[role]
        and qualification.get("manifest_verification_sha256")
        == _canonical_object_sha256(stable)
    ):
        raise TerminalWorkflowError(
            f"{role} checkpoint does not bind the stable scientific replay"
        )
    return stable


def _robust_replay_manifest_binding(
    role: str,
    *,
    episodes_per_dgp: int,
    authorization_state_sha256: str,
    cell_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    for regime in DGP_ORDER:
        item = cell_metadata[regime]
        regimes[regime] = {
            "raw_manifest": item["raw_manifest"],
            "execution_manifest": item["execution_manifest"],
            "aggregate": item["aggregate"],
            "episode_count": item["episode_count"],
            "row_count": item["row_count"],
            "episode_ids_sha256": item["episode_ids_sha256"],
            "verified_file_index_sha256": item[
                "verified_file_index_sha256"
            ],
        }
    return {
        "role": role,
        "state": REPLAY_ROLE_STATES[role],
        "regime_order": list(DGP_ORDER),
        "episodes_per_regime": episodes_per_dgp,
        "episode_count": episodes_per_dgp * len(DGP_ORDER),
        "row_count": episodes_per_dgp
        * len(DGP_ORDER)
        * REPLAY_ROWS_PER_EPISODE,
        "authorization_state_sha256": authorization_state_sha256,
        "regimes": regimes,
    }


def _persist_or_verify_replay_object(
    path: Path,
    value: Mapping[str, Any],
    *,
    label: str,
    require_existing: bool,
) -> dict[str, Any]:
    payload = canonical_json_bytes(value, pretty=True)
    if path.exists() or path.is_symlink():
        observed = _read_canonical_replay_object(path, label)
        if observed != dict(value) or path.read_bytes() != payload:
            raise TerminalWorkflowError(f"immutable {label} drift")
        return observed
    if require_existing:
        raise TerminalWorkflowError(f"required pre-count {label} is absent")
    _write_exclusive(path, payload)
    return _read_canonical_replay_object(path, label)


def _qualify_terminal_replay_role(
    role: str,
    *,
    state: Mapping[str, Any],
    attempt_root: Path,
    repo_root: Path,
    cell_runner: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    episodes_per_dgp = _expected_replay_episodes_per_dgp(role, state)
    cell_metadata: dict[str, dict[str, Any]] = {}
    cell_results: dict[str, dict[str, Any]] = {}
    common_sources: dict[str, str] | None = None
    authorization_hash: str | None = None
    regime_records: dict[str, Any] = {}
    for regime in DGP_ORDER:
        fresh = cell_runner(role, regime)
        if not isinstance(fresh, Mapping):
            raise TerminalWorkflowError(
                f"scientific replay returned no object: {role}/{regime}"
            )
        result = dict(fresh)
        metadata = _validate_terminal_replay_result(
            result,
            role=role,
            regime=regime,
            episodes_per_dgp=episodes_per_dgp,
            attempt_root=attempt_root,
            repo_root=repo_root,
        )
        result_path = (
            Path(attempt_root)
            / "audit"
            / "scientific_replay"
            / f"{role}_{regime}.json"
        )
        _persist_or_verify_replay_object(
            result_path,
            result,
            label=f"{role}/{regime} scientific replay result",
            require_existing=role in ("fit", "selection"),
        )
        sources = metadata["source_hashes"]
        if common_sources is None:
            common_sources = dict(sources)
        elif common_sources != sources:
            raise TerminalWorkflowError(
                f"scientific replay source hashes differ within {role}"
            )
        observed_authorization = metadata["authorization_state_sha256"]
        if authorization_hash is None:
            authorization_hash = observed_authorization
        elif authorization_hash != observed_authorization:
            raise TerminalWorkflowError(
                f"scientific replay authorization differs within {role}"
            )
        cell_results[regime] = result
        cell_metadata[regime] = metadata
        result_link = {
            "path": _repo_relative(result_path, repo_root=repo_root),
            "sha256": sha256_file(result_path),
        }
        regime_records[regime] = {
            "result": result_link,
            "raw_manifest": metadata["raw_manifest"],
            "execution_manifest": metadata["execution_manifest"],
            "aggregate": metadata["aggregate"],
            "episode_count": metadata["episode_count"],
            "row_count": metadata["row_count"],
        }
    if common_sources is None or authorization_hash is None:
        raise TerminalWorkflowError(f"empty scientific replay role: {role}")
    if role in ("fit", "selection"):
        stable = _development_replay_manifest_binding(
            role, attempt_root=attempt_root
        )
        if (
            stable["authorization_state_sha256"] != authorization_hash
            or stable["episodes_per_regime"] != episodes_per_dgp
            or stable["episode_count"] != episodes_per_dgp * len(DGP_ORDER)
            or stable["row_count"]
            != episodes_per_dgp * len(DGP_ORDER) * REPLAY_ROWS_PER_EPISODE
        ):
            raise TerminalWorkflowError(
                f"{role} fresh replay differs from pre-count manifest binding"
            )
        for regime in DGP_ORDER:
            expected = stable["regimes"][regime]
            observed = cell_metadata[regime]
            if (
                expected["raw_manifest"] != observed["raw_manifest"]
                or expected["execution_manifest"]
                != observed["execution_manifest"]
                or expected["aggregate"]["path"]
                != observed["aggregate"]["path"]
                or expected["aggregate"]["sha256"]
                != observed["aggregate"]["sha256"]
                or expected["episode_count"] != observed["episode_count"]
                or expected["row_count"] != observed["row_count"]
            ):
                raise TerminalWorkflowError(
                    f"{role}/{regime} replay/checkpoint binding drift"
                )
    else:
        stable = _robust_replay_manifest_binding(
            role,
            episodes_per_dgp=episodes_per_dgp,
            authorization_state_sha256=authorization_hash,
            cell_metadata=cell_metadata,
        )
    total_episodes = episodes_per_dgp * len(DGP_ORDER)
    total_rows = total_episodes * REPLAY_ROWS_PER_EPISODE
    role_audit = {
        "schema_version": 1,
        "artifact_type": "v008_independent_scientific_replay_role",
        "attempt": ACTIVE_ATTEMPT,
        "role": role,
        "state": REPLAY_ROLE_STATES[role],
        "authorization_state_sha256": authorization_hash,
        "manifest_verification_sha256": _canonical_object_sha256(stable),
        "regime_order": list(DGP_ORDER),
        "episodes_per_regime": episodes_per_dgp,
        "episode_count": total_episodes,
        "row_count": total_rows,
        "regimes": regime_records,
        "source_hashes": common_sources,
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "all_development_aggregates_exact": True,
        "all_runtime_mps_exact": True,
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_trees_unchanged": True,
        "read_only": True,
        "passed": True,
    }
    if set(role_audit) != REPLAY_ROLE_AUDIT_KEYS:
        raise TerminalWorkflowError(f"internal replay role-audit schema drift: {role}")
    role_path = Path(attempt_root) / "audit" / f"{role}_scientific_replay.json"
    persisted = _persist_or_verify_replay_object(
        role_path,
        role_audit,
        label=f"{role} scientific replay role audit",
        require_existing=role in ("fit", "selection"),
    )
    return {
        "path": _repo_relative(role_path, repo_root=repo_root),
        "sha256": sha256_file(role_path),
        "role": role,
        "state": REPLAY_ROLE_STATES[role],
        "authorization_state_sha256": authorization_hash,
        "manifest_verification_sha256": role_audit[
            "manifest_verification_sha256"
        ],
        "episode_count": total_episodes,
        "row_count": total_rows,
        "passed": persisted.get("passed") is True,
    }, common_sources


def _ensure_terminal_scientific_replay_qualification(
    *,
    state: Mapping[str, Any],
    mode: str,
    attempt_root: Path,
    repo_root: Path,
    cell_runner: Any | None = None,
) -> dict[str, Any]:
    roles = _expected_replay_roles(mode)
    if mode in EARLY_MODES:
        for role in ("smoke", "confirmation"):
            data_root = Path(attempt_root) / "data" / role
            role_audit = (
                Path(attempt_root) / "audit" / f"{role}_scientific_replay.json"
            )
            cell_root = Path(attempt_root) / "audit" / "scientific_replay"
            unexpected_cells = (
                list(cell_root.glob(f"{role}_*.json")) if cell_root.exists() else []
            )
            if (
                (data_root.exists() and any(data_root.rglob("*")))
                or role_audit.exists()
                or role_audit.is_symlink()
                or unexpected_cells
            ):
                raise TerminalWorkflowError(
                    f"early replay mode contains unavailable later role: {role}"
                )

    expected_totals = {
        role: _expected_replay_episodes_per_dgp(role, state) * len(DGP_ORDER)
        for role in roles
    }
    expected_state_counts = {
        "fit": state.get("fit_outcome_episodes"),
        "selection": state.get("selection_outcome_episodes"),
        "smoke": state.get("smoke_outcome_episodes"),
        "confirmation": state.get("confirmation_outcome_episodes_executed"),
    }
    if any(
        type(expected_state_counts[role]) is not int
        or expected_state_counts[role] != expected_totals[role]
        for role in roles
    ):
        raise TerminalWorkflowError(
            "controller role counts do not match scientific replay scope"
        )
    if mode == "confirmation":
        if (
            state.get("confirmation_outcome_episodes_generated")
            != expected_totals["confirmation"]
            or state.get("confirmation_outcomes_opened_for_analysis") is not True
        ):
            raise TerminalWorkflowError(
                "confirmation replay lacks the one fixed opened cohort"
            )

    before = _controller_replay_snapshot(
        state, attempt_root=attempt_root, repo_root=repo_root
    )
    if cell_runner is None:
        cell_runner = lambda role, regime: _run_terminal_scientific_replay_cell(
            role, regime, attempt_root=attempt_root
        )
    role_records: dict[str, Any] = {}
    common_sources: dict[str, str] | None = None
    total_episodes = 0
    total_rows = 0
    for role in roles:
        record, sources = _qualify_terminal_replay_role(
            role,
            state=state,
            attempt_root=attempt_root,
            repo_root=repo_root,
            cell_runner=cell_runner,
        )
        if common_sources is None:
            common_sources = sources
        elif common_sources != sources:
            raise TerminalWorkflowError(
                "scientific replay source hashes differ across roles"
            )
        role_records[role] = record
        total_episodes += int(record["episode_count"])
        total_rows += int(record["row_count"])
    if common_sources is None:
        raise TerminalWorkflowError("scientific replay role scope is empty")
    after = _controller_replay_snapshot(
        state, attempt_root=attempt_root, repo_root=repo_root
    )
    if before != after:
        raise TerminalWorkflowError(
            "controller state or ledger changed during scientific replay"
        )
    audit = {
        "schema_version": 1,
        "artifact_type": "v008_terminal_scientific_replay_qualification",
        "attempt": ACTIVE_ATTEMPT,
        "mode": mode,
        "checkpoint_state": "INDEPENDENT_VERIFICATION",
        "role_order": list(roles),
        "regime_order": list(DGP_ORDER),
        "role_count": len(roles),
        "qualification_count": len(roles) * len(DGP_ORDER),
        "episode_count": total_episodes,
        "row_count": total_rows,
        "role_audits": role_records,
        "source_hashes": common_sources,
        "controller_snapshot": before,
        "controller_unchanged_during_replay": True,
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "all_applicable_development_aggregates_exact": True,
        "all_runtime_mps_exact": True,
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_trees_unchanged": True,
        "read_only": True,
        "terminal_manifest_created_only_after_replay_qualification": True,
        "passed": True,
    }
    expected_keys = {
        "schema_version",
        "artifact_type",
        "attempt",
        "mode",
        "checkpoint_state",
        "role_order",
        "regime_order",
        "role_count",
        "qualification_count",
        "episode_count",
        "row_count",
        "role_audits",
        "source_hashes",
        "controller_snapshot",
        "controller_unchanged_during_replay",
        "input_loader_agreement_exact",
        "every_persisted_tensor_exact",
        "all_applicable_development_aggregates_exact",
        "all_runtime_mps_exact",
        "target_loss_computed",
        "loss_or_effect_used_for_acceptance",
        "contact_or_privileged_materialized",
        "no_gradients",
        "artifact_trees_unchanged",
        "read_only",
        "terminal_manifest_created_only_after_replay_qualification",
        "passed",
    }
    if set(audit) != expected_keys:
        raise TerminalWorkflowError("internal terminal replay-audit schema drift")
    destination = Path(attempt_root) / SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE
    return _persist_or_verify_replay_object(
        destination,
        audit,
        label="terminal scientific replay qualification",
        require_existing=False,
    )


def _walk_attempt_files(
    *, attempt_root: Path, repo_root: Path, exclusions: set[str]
) -> dict[str, dict[str, Any]]:
    attempt_root = attempt_root.resolve(strict=True)
    repo_root = repo_root.resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    inodes: dict[tuple[int, int], str] = {}
    for raw_root, directories, filenames in os.walk(
        attempt_root, topdown=True, followlinks=False
    ):
        root = Path(raw_root)
        directories[:] = sorted(directories)
        for directory in directories:
            candidate = root / directory
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise TerminalWorkflowError(
                    f"linked or non-directory terminal path component: {candidate}"
                )
        for filename in sorted(filenames):
            candidate = root / filename
            relative = candidate.relative_to(repo_root).as_posix()
            if relative in exclusions:
                metadata = candidate.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise TerminalWorkflowError(
                        f"excluded terminal path is linked or nonregular: {candidate}"
                    )
                continue
            record = _stable_file_record(candidate)
            metadata = candidate.stat()
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in inodes:
                raise TerminalWorkflowError(
                    f"hard-link alias in terminal path set: {relative}/{inodes[identity]}"
                )
            inodes[identity] = relative
            if relative in files:
                raise TerminalWorkflowError(f"duplicate terminal path: {relative}")
            files[relative] = record
    if not files:
        raise TerminalWorkflowError("terminal input manifest would be empty")
    return dict(sorted(files.items()))


def _manifest_contract(
    contract_path: Path, *, attempt_root: Path, repo_root: Path
) -> tuple[dict[str, Any], set[str]]:
    if contract_path.is_symlink() or not contract_path.is_file():
        raise TerminalWorkflowError("verifier contract is absent, linked, or nonregular")
    contract = read_json(contract_path)
    attempt_relative = _attempt_relative(attempt_root, repo_root)
    expected_exclusions = {
        f"{attempt_relative}/{MANIFEST_RELATIVE}",
        f"{attempt_relative}/{AUDIT_RELATIVE}",
    }
    exclusions_raw = contract.get("terminal_manifest_exclusions")
    if not isinstance(exclusions_raw, list):
        raise TerminalWorkflowError("verifier contract lacks terminal exclusions")
    exclusions = {_canonical_relative(item) for item in exclusions_raw}
    paths = contract.get("paths")
    checks = {
        "schema": contract.get("schema_version") == 1,
        "attempt": contract.get("attempt") == ACTIVE_ATTEMPT,
        "attempt_root": contract.get("attempt_root") == attempt_relative,
        "mode": contract.get("mode") in ("confirmation", *EARLY_MODES),
        "contract_in_attempt": contract_path.resolve(strict=True).parent
        == attempt_root.resolve(strict=True),
        "path_map": isinstance(paths, Mapping),
        "exclusions_exact": exclusions == expected_exclusions,
    }
    if isinstance(paths, Mapping):
        checks["manifest_path_exact"] = paths.get("terminal_manifest") == (
            f"{attempt_relative}/{MANIFEST_RELATIVE}"
        )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise TerminalWorkflowError(f"terminal verifier contract drift: {failed}")
    return contract, exclusions


def _validate_manifest_state(state: Mapping[str, Any], mode: str) -> None:
    if state.get("active_attempt") != ACTIVE_ATTEMPT:
        raise TerminalWorkflowError("terminal manifest attempt is not active")
    if state.get("current_state") != "INDEPENDENT_VERIFICATION":
        raise TerminalWorkflowError(
            "terminal manifest may only close inputs at independent verification"
        )
    early = state.get("early_scientific_failure")
    if mode in EARLY_MODES:
        if (
            not isinstance(early, Mapping)
            or early.get("mode") != mode
            or early.get("status") != "awaiting_independent_verification"
        ):
            raise TerminalWorkflowError("early verifier mode/controller state mismatch")
        zero = (
            state.get("smoke_outcome_episodes") == 0
            and state.get("confirmation_outcome_episodes_generated") == 0
            and state.get("confirmation_outcome_episodes_executed") == 0
            and state.get("confirmation_outcomes_opened_for_analysis") is False
        )
        if not zero:
            raise TerminalWorkflowError("early terminal path contains later-role outcomes")
    elif early is not None:
        raise TerminalWorkflowError("confirmation verifier selected on an early path")


def verify_terminal_input_manifest(
    manifest_path: Path,
    contract_path: Path,
    *,
    attempt_root: Path = ATTEMPT_ROOT,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    contract, exclusions = _manifest_contract(
        contract_path, attempt_root=attempt_root, repo_root=repo_root
    )
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise TerminalWorkflowError("terminal manifest file map is absent")
    observed = _walk_attempt_files(
        attempt_root=attempt_root, repo_root=repo_root, exclusions=exclusions
    )
    checks = {
        "schema": manifest.get("schema_version") == 1,
        "attempt": manifest.get("attempt") == ACTIVE_ATTEMPT,
        "passed": manifest.get("passed") is True,
        "mode": manifest.get("mode") == contract["mode"],
        "exclusions": manifest.get("excluded_paths") == sorted(exclusions),
        "files": dict(files) == observed,
        "file_count": manifest.get("file_count") == len(observed),
        "total_bytes": manifest.get("total_bytes")
        == sum(int(item["bytes"]) for item in observed.values()),
    }
    replay_relative = (
        f"{_attempt_relative(attempt_root, repo_root)}/"
        f"{SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE}"
    )
    checks["scientific_replay_qualification_included"] = (
        replay_relative in files
        and replay_relative in observed
        and files.get(replay_relative) == observed.get(replay_relative)
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise TerminalWorkflowError(f"terminal manifest verification failed: {failed}")
    return manifest


def build_terminal_input_manifest(
    contract_path: Path,
    *,
    state: Mapping[str, Any],
    attempt_root: Path = ATTEMPT_ROOT,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    contract_path = Path(contract_path)
    contract, exclusions = _manifest_contract(
        contract_path, attempt_root=attempt_root, repo_root=repo_root
    )
    _validate_manifest_state(state, str(contract["mode"]))
    _ensure_terminal_scientific_replay_qualification(
        state=state,
        mode=str(contract["mode"]),
        attempt_root=attempt_root,
        repo_root=repo_root,
    )
    destination = attempt_root / MANIFEST_RELATIVE
    if destination.exists():
        return verify_terminal_input_manifest(
            destination,
            contract_path,
            attempt_root=attempt_root,
            repo_root=repo_root,
        )
    if (attempt_root / AUDIT_RELATIVE).exists():
        raise TerminalWorkflowError(
            "independent audit exists before terminal input closure"
        )
    files = _walk_attempt_files(
        attempt_root=attempt_root, repo_root=repo_root, exclusions=exclusions
    )
    payload = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "mode": contract["mode"],
        "created_unix_ns": time.time_ns(),
        "passed": True,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files.values()),
        "excluded_paths": sorted(exclusions),
        "outcome_arrays_opened_by_manifest_builder": False,
        "path_set_complete_at_independent_verification": True,
    }
    _write_exclusive(destination, canonical_json_bytes(payload, pretty=True))
    return verify_terminal_input_manifest(
        destination,
        contract_path,
        attempt_root=attempt_root,
        repo_root=repo_root,
    )


def _controller(arguments: Sequence[str], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(CONTROLLER), *arguments],
        cwd=repo_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise TerminalWorkflowError(
            f"controller command failed ({' '.join(arguments)}): {detail}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise TerminalWorkflowError("controller emitted non-JSON output") from error
    if not isinstance(value, dict):
        raise TerminalWorkflowError("controller output is not an object")
    return value


def controller_status() -> dict[str, Any]:
    return _controller(("status",))


def _audit_common(
    audit: Mapping[str, Any], state: Mapping[str, Any], *, require_pass: bool | None
) -> None:
    capture = audit.get("capture")
    contract = audit.get("verifier_contract")
    checks = {
        "schema": audit.get("schema_version") == 1,
        "attempt": audit.get("attempt") == ACTIVE_ATTEMPT,
        "mode": audit.get("mode") in ("confirmation", *EARLY_MODES),
        "read_only": audit.get("read_only_verifier") is True,
        "checkpoint": audit.get("checkpoint_state")
        == "INDEPENDENT_VERIFICATION",
        "contract": isinstance(contract, Mapping),
        "capture": isinstance(capture, Mapping),
    }
    if require_pass is not None:
        checks["passed"] = audit.get("passed") is require_pass
    if isinstance(capture, Mapping):
        checks["capture_exclusive"] = capture.get("captured_exclusively") is True
        checks["capture_integrity"] = capture.get("wrapper_integrity_passed") is True
        checks["capture_pass_agrees"] = (
            capture.get("scientific_verifier_passed")
            is (audit.get("passed") is True)
        )
        if audit.get("passed") is True:
            checks["capture_returncode"] = capture.get("verifier_returncode") == 0
    if isinstance(contract, Mapping):
        try:
            contract_relative = _canonical_relative(contract.get("path"))
        except TerminalWorkflowError:
            contract_relative = ""
        contract_path = REPO_ROOT / contract_relative
        checks["contract_path"] = (
            bool(contract_relative)
            and contract_path.is_file()
            and not contract_path.is_symlink()
            and contract_path.resolve().parent == ATTEMPT_ROOT.resolve()
        )
        checks["contract_hash"] = (
            contract_path.is_file()
            and contract.get("sha256") == sha256_file(contract_path)
        )
    checks["controller_state"] = state.get("current_state") in (
        "INDEPENDENT_VERIFICATION",
        "TERMINAL",
        "POST_TERMINAL_REPORTING",
    )
    if audit.get("passed") is True:
        verifier_checks = audit.get("checks")
        checks["verifier_checks"] = (
            isinstance(verifier_checks, Mapping)
            and bool(verifier_checks)
            and all(value is True for value in verifier_checks.values())
        )
        checks["production_modules_not_imported"] = (
            audit.get("local_production_modules_imported") is False
        )
        checks["stdout_json_only"] = audit.get("stdout_json_only") is True
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise TerminalWorkflowError(f"captured independent audit mismatch: {failed}")


def _analysis_for_decision(audit: Mapping[str, Any]) -> dict[str, Any]:
    path = ATTEMPT_ROOT / "analysis_result.json"
    if not path.is_file():
        raise TerminalWorkflowError("verified scientific terminal lacks analysis_result.json")
    result = read_json(path)
    source_hashes = audit.get("source_hashes")
    checks = {
        "schema": result.get("schema_version") == 1,
        "attempt": result.get("attempt") == ACTIVE_ATTEMPT,
        "checkpoint": result.get("checkpoint_state") == "SEALED_ANALYSIS",
        "analysis_passed": result.get("passed") is True,
        "integrity": result.get("process_valid") is True
        and result.get("integrity_passed") is True,
        "source_hashes": isinstance(source_hashes, Mapping),
    }
    if isinstance(source_hashes, Mapping):
        checks["analysis_hash"] = source_hashes.get("analysis_result") == sha256_file(
            path
        )
    label = audit.get("terminal_label")
    checks["label"] = label in SCIENTIFIC_LABELS
    checks["producer_label"] = result.get("proposed_terminal_label") == label
    supported = result.get("supported_co_primary_claim_count")
    checks["support_count"] = isinstance(supported, int) and supported in range(9)
    verifier_evidence = audit.get("evidence")
    verifier_confirmation = (
        verifier_evidence.get("confirmation")
        if isinstance(verifier_evidence, Mapping)
        else None
    )
    checks["verifier_confirmation"] = isinstance(verifier_confirmation, Mapping)
    if isinstance(verifier_confirmation, Mapping):
        checks["verifier_support_count"] = (
            verifier_confirmation.get("supported_co_primary_claim_count") == supported
        )
        checks["verifier_terminal_label"] = (
            verifier_confirmation.get("terminal_label") == label
        )
    expected = (
        "domain_robust_gate_confirmed"
        if supported == 8
        else "domain_robust_gate_partial"
        if supported
        else "domain_robust_gate_failed"
    )
    checks["mapping"] = label == expected
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise TerminalWorkflowError(f"verified analysis/terminal mapping drift: {failed}")
    return result


def _decision_existing_or_write(payload: Mapping[str, Any]) -> dict[str, Any]:
    path = ATTEMPT_ROOT / DECISION_RELATIVE
    if path.exists():
        existing = read_json(path)
        # Timestamps may differ across a resumed builder call; every scientific
        # and integrity-bearing field must remain exact.
        comparable = dict(existing)
        comparable.pop("created_unix_ns", None)
        expected = dict(payload)
        expected.pop("created_unix_ns", None)
        if comparable != expected:
            raise TerminalWorkflowError("immutable terminal decision drift")
        return existing
    _write_exclusive(path, canonical_json_bytes(payload, pretty=True))
    return read_json(path)


def build_terminal_decision(
    state: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    _audit_common(audit, state, require_pass=None)
    audit_path = ATTEMPT_ROOT / AUDIT_RELATIVE
    if not audit_path.is_file():
        raise TerminalWorkflowError("captured independent audit is absent")
    common: dict[str, Any] = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "TERMINAL",
        "created_unix_ns": time.time_ns(),
        "passed": True,
        "independent_verification_path": _repo_relative(audit_path),
        "independent_verification_sha256": sha256_file(audit_path),
        "no_confirmation_retry_or_sequential_expansion": True,
        "claim_scope": (
            "causal contact-free latent-prediction adaptive compute at exact "
            "counted compute within the four predeclared DGPs"
        ),
    }
    early = state.get("early_scientific_failure")
    if isinstance(early, Mapping):
        mode = str(early.get("mode"))
        _audit_common(audit, state, require_pass=True)
        if mode not in EARLY_MODES or audit.get("mode") != mode:
            raise TerminalWorkflowError("early terminal audit mode drift")
        if audit.get("terminal_label") != "domain_robust_gate_failed":
            raise TerminalWorkflowError("early terminal label is not failed")
        payload = common | {
            "terminal_basis": f"preregistered_{mode}",
            "early_failure_mode": mode,
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
            "confirmation_terminal": False,
            "trigger_evidence_path": early.get("trigger_evidence_path"),
            "trigger_evidence_sha256": early.get("trigger_evidence_sha256"),
            "smoke_outcome_episodes": 0,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcomes_opened_for_analysis": False,
            "supported_co_primary_claim_count": None,
            "supported_count_interpretation": "no confirmation claim was attempted",
        }
        return _decision_existing_or_write(payload)

    if audit.get("mode") != "confirmation":
        raise TerminalWorkflowError("ordinary terminal path used a non-confirmation audit")
    audit_passed = audit.get("passed") is True
    label = audit.get("terminal_label")
    if audit_passed and label in SCIENTIFIC_LABELS:
        analysis = _analysis_for_decision(audit)
        expected = int(state.get("expected_confirmation_episode_count", -1))
        if not (
            expected > 0
            and state.get("confirmation_outcome_episodes_generated") == expected
            and state.get("confirmation_outcome_episodes_executed") == expected
            and state.get("confirmation_outcomes_opened_for_analysis") is True
        ):
            raise TerminalWorkflowError("scientific decision lacks the fixed opened cohort")
        payload = common | {
            "terminal_basis": "independently_verified_fixed_confirmation",
            "terminal_label": label,
            "process_valid": True,
            "confirmation_terminal": True,
            "supported_co_primary_claim_count": analysis[
                "supported_co_primary_claim_count"
            ],
            "required_co_primary_claim_count": 8,
            "analysis_result_path": _repo_relative(
                ATTEMPT_ROOT / "analysis_result.json"
            ),
            "analysis_result_sha256": sha256_file(
                ATTEMPT_ROOT / "analysis_result.json"
            ),
            "confirmation_outcome_episodes_generated": expected,
            "confirmation_outcome_episodes_executed": expected,
            "confirmation_outcomes_opened_for_analysis": True,
        }
        return _decision_existing_or_write(payload)

    # Once the fixed confirmation has been opened, an independent verifier
    # failure or an independently corroborated integrity failure has precedence
    # over every scientific support count.
    capture = audit.get("capture")
    verifier_failure = (
        audit.get("passed") is False
        and isinstance(capture, Mapping)
        and capture.get("scientific_verifier_passed") is False
        and isinstance(capture.get("verifier_returncode"), int)
        and int(capture["verifier_returncode"]) != 0
        and isinstance(audit.get("error_type"), str)
        and bool(audit.get("error_type"))
        and isinstance(audit.get("error"), str)
        and bool(audit.get("error"))
    )
    independently_invalid = audit_passed and label == INVALID_LABEL
    if not (verifier_failure or independently_invalid):
        raise TerminalWorkflowError(
            "confirmation audit is neither a verified result nor a valid integrity failure"
        )
    expected = int(state.get("expected_confirmation_episode_count", -1))
    if not (
        expected > 0
        and state.get("confirmation_outcome_episodes_generated") == expected
        and state.get("confirmation_outcome_episodes_executed") == expected
        and state.get("confirmation_outcomes_opened_for_analysis") is True
    ):
        raise TerminalWorkflowError("execution-invalid mapping lacks fixed opened cohort")
    payload = common | {
        "terminal_basis": "postconfirmation_integrity_failure",
        "terminal_label": INVALID_LABEL,
        "process_valid": False,
        "confirmation_terminal": True,
        "confirmation_outcome_episodes_generated": expected,
        "confirmation_outcome_episodes_executed": expected,
        "confirmation_outcomes_opened_for_analysis": True,
        "scientific_support_call_withheld": True,
        "independent_verifier_process_failed": verifier_failure,
        "independent_integrity_failure_confirmed": independently_invalid,
    }
    return _decision_existing_or_write(payload)


def _decision_path_argument() -> str:
    return str((ATTEMPT_ROOT / DECISION_RELATIVE).resolve())


def finalize_terminal() -> dict[str, Any]:
    """Resume and complete the mode-correct controller terminal transitions."""

    state = controller_status()
    if state.get("post_terminal_reporting_complete") is True:
        return state
    current = state.get("current_state")
    if current == "POST_TERMINAL_REPORTING":
        return state
    audit_path = ATTEMPT_ROOT / AUDIT_RELATIVE
    if not audit_path.is_file():
        raise TerminalWorkflowError("capture the independent audit before finalizing")
    audit = read_json(audit_path)
    decision = build_terminal_decision(state, audit)
    label = str(decision["terminal_label"])

    if current == "INDEPENDENT_VERIFICATION":
        early = state.get("early_scientific_failure")
        if isinstance(early, Mapping):
            if audit.get("passed") is not True:
                raise TerminalWorkflowError(
                    "early-path verifier failure is a zero-confirmation procedural "
                    "invalidity, not a scientific terminal decision"
                )
            return _controller(("finalize-early-failure", str(early["mode"])))
        if label == INVALID_LABEL:
            return _controller(("finalize-execution-invalid",))
        state = _controller(
            (
                "advance",
                "INDEPENDENT_VERIFICATION",
                str(audit_path.resolve()),
                f"{ACTIVE_ATTEMPT}_independent_verification_passed",
                "record the immutable independently verified terminal decision",
            )
        )
        current = state.get("current_state")

    if current != "TERMINAL":
        raise TerminalWorkflowError(f"terminal finalizer reached illegal state: {current}")
    if state.get("terminal_label") is None:
        state = _controller(("terminal", label, _decision_path_argument()))
    elif state.get("terminal_label") != label:
        raise TerminalWorkflowError("controller terminal label differs from decision")
    return _controller(
        (
            "advance",
            "TERMINAL",
            _decision_path_argument(),
            f"{ACTIVE_ATTEMPT}_terminal_decision",
            "write and checkpoint the claim-bounded post-terminal reports",
        )
    )


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.9g}"
    except (TypeError, ValueError):
        return "n/a"


def _robustness_map(
    state: Mapping[str, Any], decision: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, Any]:
    label = str(decision["terminal_label"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "terminal_label": label,
        "process_valid": bool(decision["process_valid"]),
        "claim_scope": decision["claim_scope"],
        "dgp_order": list(DGP_ORDER),
        "co_primary_endpoints": list(ENDPOINT_ORDER),
        "simultaneous_family_size": 8,
        "familywise_alpha": 0.05,
        "exact_total_counted_compute_comparator": True,
        "confirmation_retried": False,
        "regimes": {},
    }
    analysis_path = ATTEMPT_ROOT / "analysis_result.json"
    if label in SCIENTIFIC_LABELS and decision.get("confirmation_terminal") is True:
        analysis = read_json(analysis_path)
        simultaneous = analysis.get("simultaneous_co_primary")
        # Producer JSON is canonically key-sorted on disk; scientific ordering
        # is imposed explicitly here rather than inferred from object order.
        if not isinstance(simultaneous, Mapping) or set(simultaneous) != set(DGP_ORDER):
            raise TerminalWorkflowError("analysis robustness map DGP order drift")
        supported = 0
        for regime in DGP_ORDER:
            endpoint_map = simultaneous.get(regime)
            if not isinstance(endpoint_map, Mapping):
                raise TerminalWorkflowError("analysis endpoint map is absent")
            record: dict[str, Any] = {}
            for endpoint in ENDPOINT_ORDER:
                item = endpoint_map.get(endpoint)
                if not isinstance(item, Mapping):
                    raise TerminalWorkflowError("analysis co-primary record is absent")
                endpoint_supported = bool(item.get("supported"))
                if endpoint_supported != (float(item.get("lower")) > 0.0):
                    raise TerminalWorkflowError("analysis support/lower-bound mismatch")
                supported += int(endpoint_supported)
                record[endpoint] = {
                    "contrast": item.get("contrast"),
                    "estimate": item.get("estimate"),
                    "simultaneous_lower_bound": item.get("lower"),
                    "supported": endpoint_supported,
                }
            record["regime_status"] = (
                "supported"
                if all(record[name]["supported"] for name in ENDPOINT_ORDER)
                else "mixed"
                if any(record[name]["supported"] for name in ENDPOINT_ORDER)
                else "failed"
            )
            result["regimes"][regime] = record
        if supported != decision.get("supported_co_primary_claim_count"):
            raise TerminalWorkflowError("decision/robustness support count drift")
        result["supported_co_primary_claim_count"] = supported
        result["required_co_primary_claim_count"] = 8
    else:
        reason = (
            "execution_invalid_no_scientific_call"
            if label == INVALID_LABEL
            else "confirmation_not_attempted"
        )
        result["supported_co_primary_claim_count"] = None
        result["required_co_primary_claim_count"] = 8
        result["scientific_support_call_withheld"] = label == INVALID_LABEL
        result["regimes"] = {
            regime: {
                "raw": None,
                "fixed_whitened": None,
                "regime_status": reason,
            }
            for regime in DGP_ORDER
        }
    result["independent_verification"] = {
        "passed": audit.get("passed") is True,
        "mode": audit.get("mode"),
        "path": _repo_relative(ATTEMPT_ROOT / AUDIT_RELATIVE),
        "sha256": sha256_file(ATTEMPT_ROOT / AUDIT_RELATIVE),
    }
    result["not_claimed"] = [
        "downstream control improvement",
        "contact-aware routing",
        "universal robustness",
        "wall-clock acceleration",
        "energy efficiency",
        "priority",
    ]
    return result


def _report_markdown(
    state: Mapping[str, Any], decision: Mapping[str, Any], robustness: Mapping[str, Any]
) -> str:
    label = str(decision["terminal_label"])
    lines = [
        "# LeWM domain-robust gate terminal report",
        "",
        f"Terminal label: `{label}`.",
        "",
        f"Claim boundary: {decision['claim_scope']}.",
        "",
    ]
    if decision.get("confirmation_terminal") is True and label in SCIENTIFIC_LABELS:
        supported = int(decision["supported_co_primary_claim_count"])
        lines.extend(
            [
                f"The entirely fresh fixed confirmation supported {supported}/8 "
                "simultaneously corrected DGP-by-endpoint claims. The strongest "
                "transition-independent analytic allocation was matched at exact "
                "total counted compute, including causal feature/head overhead.",
                "",
                "| DGP | Raw estimate | Raw simultaneous LB | Raw | Fixed-whitened estimate | Fixed-whitened simultaneous LB | Fixed-whitened |",
                "|---|---:|---:|:---:|---:|---:|:---:|",
            ]
        )
        for regime in DGP_ORDER:
            row = robustness["regimes"][regime]
            raw = row["raw"]
            fixed = row["fixed_whitened"]
            lines.append(
                f"| `{regime}` | {_fmt(raw['estimate'])} | "
                f"{_fmt(raw['simultaneous_lower_bound'])} | "
                f"{'yes' if raw['supported'] else 'no'} | "
                f"{_fmt(fixed['estimate'])} | "
                f"{_fmt(fixed['simultaneous_lower_bound'])} | "
                f"{'yes' if fixed['supported'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                "Inference used the preregistered 20,000-replicate episode bootstrap "
                "and the immutable family-eight one-sided support rule. Latency and "
                "energy availability were reported separately and did not alter this call.",
            ]
        )
    elif label == INVALID_LABEL:
        lines.extend(
            [
                "The fixed confirmation was opened, but a required integrity or "
                "independent-verification condition failed. Integrity has precedence, "
                "so no scientific support count is reported and this confirmation will "
                "not be repaired, expanded, or retried.",
            ]
        )
    else:
        mode = decision.get("early_failure_mode")
        lines.extend(
            [
                f"The preregistered `{mode}` condition was independently verified "
                "before smoke or confirmation. This is a process-valid failed study: "
                "no confirmation claim was attempted and no confirmation outcome was opened.",
            ]
        )
    lines.extend(
        [
            "",
            "The base world model, stagewise refiner, V5 evidence, and v005 terminal "
            "evidence remain unchanged. This result does not establish downstream "
            "control improvement, contact-aware routing, universal robustness, "
            "wall-clock acceleration, energy efficiency, or priority.",
            "",
        ]
    )
    return "\n".join(lines)


def _independent_audit_markdown(audit: Mapping[str, Any]) -> str:
    contract = audit.get("verifier_contract", {})
    capture = audit.get("capture", {})
    source_hashes = audit.get("source_hashes", {})
    lines = [
        "# Independent audit",
        "",
        f"Mode: `{audit.get('mode')}`.",
        f"Verifier passed: `{audit.get('passed') is True}`.",
        f"Read-only verifier: `{audit.get('read_only_verifier') is True}`.",
        f"Captured exclusively: `{capture.get('captured_exclusively') is True}`.",
        f"Contract: `{contract.get('path')}` (`{contract.get('sha256')}`).",
    ]
    if audit.get("passed") is True:
        lines.extend(
            [
                "",
                "The standalone implementation imported no production scientific module. "
                "It independently checked the applicable chronology, role isolation, hashes, "
                "candidate selection, exact compute, comparators, fixed inference, and terminal mapping.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The standalone verifier did not complete its required closure. The captured "
                "failure is therefore integrity evidence for an execution-invalid result, not "
                "a successful independent audit or a scientific endpoint call.",
            ]
        )
    lines.extend(["", "## Bound source hashes", ""])
    if isinstance(source_hashes, Mapping) and source_hashes:
        lines.extend(
            f"- `{name}`: `"
            f"{value if isinstance(value, str) else value.get('sha256') if isinstance(value, Mapping) else None}`"
            for name, value in sorted(source_hashes.items())
        )
    else:
        lines.append("- The verifier failed before producing a complete source-hash map; the captured failure is immutable.")
    lines.append("")
    return "\n".join(lines)


def _limitations_markdown(decision: Mapping[str, Any]) -> str:
    terminal_specific = {
        "domain_robust_gate_confirmed": (
            "Confirmation is limited to the four sealed one-factor DGP regimes and "
            "the two latent-prediction endpoints; robustness outside that envelope is unknown."
        ),
        "domain_robust_gate_partial": (
            "One or more predeclared DGP-by-endpoint claims were unsupported; the "
            "supported cells must not be generalized to the failed cells."
        ),
        "domain_robust_gate_failed": (
            "This study did not establish the preregistered domain-robust claim. "
            "An early failed result does not provide confirmation endpoint estimates."
        ),
        INVALID_LABEL: (
            "The opened confirmation is execution-invalid; its scientific calls are "
            "withheld and it cannot be repaired or retried within this study."
        ),
    }[str(decision["terminal_label"])]
    return "\n".join(
        [
            "# Limitations",
            "",
            f"- {terminal_specific}",
            "",
            "- The gate is causal and contact-free; this study does not evaluate a contact-aware router.",
            "",
            "- Latent-prediction gain does not imply downstream control or task-success improvement.",
            "",
            "- Exact counted FLOPs include the gate feature/head overhead, but do not by themselves imply wall-clock or energy savings.",
            "",
            "- The inference does not establish universal robustness, deployment safety, or priority over untested methods.",
            "",
        ]
    )


def _follow_on_markdown(decision: Mapping[str, Any]) -> str:
    label = str(decision["terminal_label"])
    if label == "domain_robust_gate_confirmed":
        task = (
            "Synthesize the sealed V5, v005, and present confirmation evidence into "
            "a paper-ready claim table, and preregister a separate deployment-efficiency "
            "study for wall-clock and energy behavior. Do not broaden the scientific claim."
        )
    elif label == "domain_robust_gate_partial":
        task = (
            "Diagnose the unsupported DGP-by-endpoint cells using only the sealed fit, "
            "selection, and confirmation evidence. Any revised gate must be a new, "
            "independently preregistered project with fresh confirmation, never a retry."
        )
    elif label == "domain_robust_gate_failed":
        mode = decision.get("early_failure_mode")
        task = (
            "If the result was early, use the sealed development evidence to determine "
            f"why `{mode}` occurred; otherwise diagnose all unsupported cells. Design a "
            "new bounded gate-only study only if the diagnosis motivates one. Preserve "
            "this failed result and do not reuse its held-out roles."
        )
    else:
        task = (
            "Perform a forensic, read-only integrity diagnosis of the immutable failure. "
            "Because confirmation was opened, do not repair, expand, or rerun this study. "
            "Any future experiment must be a separately scoped and preregistered project."
        )
    return "\n".join(["# Follow-on task", "", task, ""])


def build_postterminal_reports(
    state: Mapping[str, Any], *, attempt_root: Path = ATTEMPT_ROOT
) -> dict[str, Any]:
    if state.get("current_state") != "POST_TERMINAL_REPORTING":
        raise TerminalWorkflowError("reports require an immutable terminal decision")
    label = state.get("terminal_label")
    if label not in ALL_LABELS or not isinstance(state.get("process_valid"), bool):
        raise TerminalWorkflowError("controller terminal fields are incomplete")
    decision_path = attempt_root / DECISION_RELATIVE
    audit_path = attempt_root / AUDIT_RELATIVE
    decision = read_json(decision_path)
    audit = read_json(audit_path)
    checks = {
        "decision_label": decision.get("terminal_label") == label,
        "decision_process": decision.get("process_valid") == state.get("process_valid"),
        "decision_link": state.get("terminal_decision_path")
        == _repo_relative(decision_path),
        "decision_hash": state.get("terminal_decision_sha256")
        == sha256_file(decision_path),
        "audit_link": decision.get("independent_verification_sha256")
        == sha256_file(audit_path),
    }
    if not all(checks.values()):
        raise TerminalWorkflowError(
            f"postterminal input drift: {[name for name, passed in checks.items() if not passed]}"
        )
    robustness = _robustness_map(state, decision, audit)
    payloads = {
        "REPORT.md": _report_markdown(state, decision, robustness).encode("utf-8"),
        "ROBUSTNESS_MAP.json": canonical_json_bytes(robustness, pretty=True),
        "INDEPENDENT_AUDIT.md": _independent_audit_markdown(audit).encode("utf-8"),
        "LIMITATIONS.md": _limitations_markdown(decision).encode("utf-8"),
        "FOLLOW_ON_TASK.md": _follow_on_markdown(decision).encode("utf-8"),
    }
    for name, payload in payloads.items():
        _write_or_verify(attempt_root / name, payload)
    artifact_hashes = {
        _repo_relative(attempt_root / name): {
            "sha256": sha256_file(attempt_root / name),
            "bytes": (attempt_root / name).stat().st_size,
        }
        for name in REPORT_FILENAMES
    }
    artifact_hashes[_repo_relative(decision_path)] = {
        "sha256": sha256_file(decision_path),
        "bytes": decision_path.stat().st_size,
    }
    artifact_hashes[_repo_relative(audit_path)] = {
        "sha256": sha256_file(audit_path),
        "bytes": audit_path.stat().st_size,
    }
    checkpoint = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "POST_TERMINAL_REPORTING",
        "created_unix_ns": time.time_ns(),
        "passed": True,
        "terminal_label": label,
        "process_valid": state["process_valid"],
        "terminal_decision_path": _repo_relative(decision_path),
        "terminal_decision_sha256": sha256_file(decision_path),
        "independent_verification_path": _repo_relative(audit_path),
        "independent_verification_sha256": sha256_file(audit_path),
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "claim_boundary_preserved": True,
        "no_postterminal_scientific_remapping": True,
        "reports_opened_no_additional_outcome_arrays": True,
    }
    checkpoint_path = attempt_root / REPORT_CHECKPOINT_RELATIVE
    if checkpoint_path.exists():
        existing = read_json(checkpoint_path)
        comparable = dict(existing)
        comparable.pop("created_unix_ns", None)
        expected = dict(checkpoint)
        expected.pop("created_unix_ns", None)
        if comparable != expected:
            raise TerminalWorkflowError("postterminal reporting checkpoint drift")
        return existing
    _write_exclusive(checkpoint_path, canonical_json_bytes(checkpoint, pretty=True))
    return read_json(checkpoint_path)


def complete_terminal_workflow() -> dict[str, Any]:
    state = controller_status()
    if state.get("current_state") != "POST_TERMINAL_REPORTING":
        state = finalize_terminal()
    if state.get("post_terminal_reporting_complete") is True:
        # The controller authenticates the reporting checkpoint itself; this
        # additional pass also rehashes every report named by that checkpoint.
        build_postterminal_reports(state)
        return controller_status()
    checkpoint = build_postterminal_reports(state)
    return _controller(
        (
            "advance",
            "POST_TERMINAL_REPORTING",
            str((ATTEMPT_ROOT / REPORT_CHECKPOINT_RELATIVE).resolve()),
            f"{ACTIVE_ATTEMPT}_post_terminal_reporting_complete",
            "program complete; preserve immutable evidence",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--contract", type=Path, required=True)
    subparsers.add_parser("finalize")
    subparsers.add_parser("reports")
    subparsers.add_parser("complete")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "manifest":
            result = build_terminal_input_manifest(
                arguments.contract.resolve(), state=controller_status()
            )
        elif arguments.command == "finalize":
            result = finalize_terminal()
        elif arguments.command == "reports":
            result = build_postterminal_reports(controller_status())
        else:
            result = complete_terminal_workflow()
        print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except (OSError, ValueError, TerminalWorkflowError) as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
