"""Synthetic negative checks for fail-closed checkpoint evidence."""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest


ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
if str(ATTEMPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTEMPT_ROOT))

import checkpoints  # noqa: E402
import analysis  # noqa: E402
import compile_gate  # noqa: E402
import latency  # noqa: E402
import study_common  # noqa: E402


JSON_EXCLUSIVE_WRITERS = (
    (
        "checkpoints",
        lambda path, value: checkpoints.atomic_json(path, value, exclusive=True),
    ),
    (
        "study_common",
        lambda path, value: study_common.atomic_json(path, value, exclusive=True),
    ),
    (
        "analysis",
        lambda path, value: analysis.atomic_json(path, value, exclusive=True),
    ),
    ("compile_gate", compile_gate._atomic_json),
    (
        "latency",
        lambda path, value: latency.atomic_json(path, value, exclusive=True),
    ),
)

NPZ_EXCLUSIVE_WRITERS = (
    (
        "study_common",
        lambda path, value: study_common.atomic_npz(path, value, exclusive=True),
    ),
    (
        "analysis",
        lambda path, value: analysis.atomic_npz(path, value, exclusive=True),
    ),
    ("compile_gate", compile_gate._atomic_npz),
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_raw_manifest_rejects_phase_as_role_alias(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoints, "REPO_ROOT", tmp_path)
    root = tmp_path / "runs/study/attempts/v001/data/confirmation/native_plan"
    array_path = root / "episode.bin"
    array_path.parent.mkdir(parents=True, exist_ok=True)
    array_path.write_bytes(b"pixels-and-actions-only")
    manifest_path = root / "raw_manifest.json"
    _write_json(
        manifest_path,
        {
            "complete": True,
            "phase": "confirmation",
            "regime": "native_plan",
            "episode_count": 1,
            "episodes": [
                {
                    "slot": 0,
                    "episode_id": "fresh-episode",
                    "raw_path": str(array_path.relative_to(tmp_path)),
                    "raw_sha256": checkpoints.sha256_file(array_path),
                }
            ],
            "raw_archive_members": ["action", "pixels"],
            "role_isolation": True,
            "retention_contract": (
                "pixels_action_shapes_finiteness_and_local_step_count_only"
            ),
        },
    )
    with pytest.raises(RuntimeError, match="noncanonical confirmation"):
        checkpoints.verify_raw_manifest(
            manifest_path,
            role="confirmation",
            regime="native_plan",
            expected_episodes=1,
        )


def test_execution_manifest_requires_explicit_unopened_fields(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoints, "REPO_ROOT", tmp_path)
    root = tmp_path / "runs/study/attempts/v001/data/confirmation/native_plan"
    part_path = root / "execution.bin"
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.write_bytes(b"execution")
    manifest_path = root / "execution_manifest.json"
    _write_json(
        manifest_path,
        {
            "complete": True,
            "role": "confirmation",
            "regime": "native_plan",
            "episode_count": 1,
            "row_count": checkpoints.ROWS_PER_EPISODE,
            "episodes": [
                {
                    "slot": 0,
                    "episode_id": "fresh-episode",
                    "path": str(part_path.relative_to(tmp_path)),
                    "sha256": checkpoints.sha256_file(part_path),
                }
            ],
            "loaded_input_keys": ["action", "pixels"],
            "all_equivalence_checks_passed": True,
            "module_before": {"passed": True},
            "module_after": {"passed": True},
            "no_gradients": True,
            # Deliberately omit the two exact negative assertions.
        },
    )
    with pytest.raises(RuntimeError, match="noncanonical confirmation"):
        checkpoints.verify_execution_manifest(
            manifest_path,
            role="confirmation",
            regime="native_plan",
            expected_episodes=1,
            raw={"episode_ids": ["fresh-episode"], "manifest": {"sha256": "unused"}},
        )


def test_controller_opened_flag_cannot_be_replaced_by_literal(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoints, "REPO_ROOT", tmp_path)
    attempt = tmp_path / "runs/lewm_domain_robust_gate/attempts/v001"
    prior_path = attempt / "audit/confirmation_generation_complete.json"
    _write_json(
        prior_path,
        {
            "passed": True,
            "checkpoint_state": "CONFIRMATION_GENERATION",
            "created_unix_ns": 10,
        },
    )
    prior = {
        "name": "generation",
        "created_unix_ns": 20,
        "evidence_path": str(prior_path.relative_to(tmp_path)),
        "evidence_sha256": checkpoints.sha256_file(prior_path),
    }
    state_path = attempt.parent.parent / "STATE.json"
    state = {
        "active_attempt": "v001",
        "active_attempt_path": str(attempt.relative_to(tmp_path)),
        "current_state": "CONFIRMATION_EXECUTION",
        "completed_states": ["CONFIRMATION_GENERATION"],
        "verified_checkpoints": [prior],
        "last_verified_checkpoint": prior,
        "confirmation_outcome_episodes_generated": 2000,
        "confirmation_outcome_episodes_executed": 2000,
        "confirmation_outcomes_opened_for_analysis": True,
        "terminal_label": None,
        "updated_unix_ns": 30,
        "ledger_event_count": 1,
        "ledger_head_sha256": "head",
    }
    _write_json(state_path, state)
    monkeypatch.setattr(checkpoints, "read_verified_controller", lambda: state)
    evidence = checkpoints._controller_chronology_evidence(
        attempt,
        current_state="CONFIRMATION_EXECUTION",
        prior_state="CONFIRMATION_GENERATION",
        expected_counts={
            "confirmation_outcome_episodes_generated": 2000,
            "confirmation_outcome_episodes_executed": 2000,
        },
    )
    assert evidence["checks"]["confirmation_outcomes_unopened"] is False
    assert evidence["passed"] is False


def test_execution_source_raw_manifest_hash_is_mandatory(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoints, "REPO_ROOT", tmp_path)
    raw_manifest = tmp_path / "runs/study/attempts/v008/data/confirmation/native_plan/raw_manifest.json"
    _write_json(raw_manifest, {"schema_version": 1})
    relative = raw_manifest.relative_to(tmp_path).as_posix()
    digest = checkpoints.sha256_file(raw_manifest)
    with pytest.raises(RuntimeError, match="mandatory raw binding"):
        checkpoints._validate_source_raw_manifest_binding(
            {
                "raw_manifest_path": relative,
                "raw_manifest_sha256": digest,
            },
            raw_manifest,
        )


def test_confirmation_namespace_rejects_unrelated_file(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoints, "REPO_ROOT", tmp_path)
    attempt = tmp_path / "runs/study/attempts/v008"
    expected = attempt / "data/confirmation/native_plan/raw_manifest.json"
    unrelated = attempt / "data/confirmation/native_plan/fabricated.json"
    _write_json(expected, {"expected": True})
    _write_json(unrelated, {"fabricated": True})
    expected_map = {
        expected.relative_to(tmp_path).as_posix(): checkpoints.sha256_file(expected)
    }
    with pytest.raises(RuntimeError, match="unrelated"):
        checkpoints._assert_exact_confirmation_namespace(attempt, expected_map)


def test_confirmation_file_link_rejects_symlink_and_inode_alias(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoints, "REPO_ROOT", tmp_path)
    real = tmp_path / "runs/study/attempts/v008/data/confirmation/native_plan/raw/episode.npz"
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_bytes(b"raw")
    symlink = real.with_name("linked.npz")
    symlink.symlink_to(real)
    with pytest.raises(RuntimeError, match="symlink"):
        checkpoints._strict_file_link(
            symlink.relative_to(tmp_path).as_posix(),
            checkpoints.sha256_file(real),
            expected_path=symlink,
            identities={},
        )

    symlink.unlink()
    alias = real.with_name("alias.npz")
    os.link(real, alias)
    with pytest.raises(RuntimeError, match="inode alias"):
        checkpoints._strict_file_link(
            real.relative_to(tmp_path).as_posix(),
            checkpoints.sha256_file(real),
            expected_path=real,
            identities={},
        )


def test_confirmation_file_link_rejects_valid_hash_at_wrong_path(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoints, "REPO_ROOT", tmp_path)
    expected = tmp_path / "runs/study/attempts/v008/data/confirmation/native_plan/raw/expected.npz"
    unrelated = expected.with_name("unrelated.npz")
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"same")
    unrelated.write_bytes(b"same")
    with pytest.raises(RuntimeError, match="noncanonical file link"):
        checkpoints._strict_file_link(
            unrelated.relative_to(tmp_path).as_posix(),
            checkpoints.sha256_file(unrelated),
            expected_path=expected,
            identities={},
        )


@pytest.mark.parametrize(
    "writer_name,writer",
    JSON_EXCLUSIVE_WRITERS,
    ids=[name for name, _ in JSON_EXCLUSIVE_WRITERS],
)
def test_exclusive_json_concurrent_writers_never_replace(
    writer_name, writer, tmp_path: Path
) -> None:
    destination = tmp_path / f"{writer_name}.json"
    barrier = threading.Barrier(2)

    def attempt(index: int) -> tuple[str, int]:
        barrier.wait()
        try:
            writer(destination, {"writer": index})
        except FileExistsError:
            return "exists", index
        return "created", index

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (1, 2)))
    assert [status for status, _ in results].count("created") == 1
    assert [status for status, _ in results].count("exists") == 1
    winner = next(index for status, index in results if status == "created")
    assert json.loads(destination.read_text(encoding="utf-8")) == {"writer": winner}
    assert list(tmp_path.glob(f".{destination.name}.*")) == []


@pytest.mark.parametrize(
    "writer_name,writer",
    NPZ_EXCLUSIVE_WRITERS,
    ids=[name for name, _ in NPZ_EXCLUSIVE_WRITERS],
)
def test_exclusive_npz_concurrent_writers_never_replace(
    writer_name, writer, tmp_path: Path
) -> None:
    destination = tmp_path / f"{writer_name}.npz"
    barrier = threading.Barrier(2)

    def attempt(index: int) -> tuple[str, int]:
        barrier.wait()
        try:
            writer(destination, {"writer": np.asarray([index], dtype=np.int64)})
        except FileExistsError:
            return "exists", index
        return "created", index

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (1, 2)))
    assert [status for status, _ in results].count("created") == 1
    assert [status for status, _ in results].count("exists") == 1
    winner = next(index for status, index in results if status == "created")
    with np.load(destination, allow_pickle=False) as stored:
        assert stored.files == ["writer"]
        assert stored["writer"].tolist() == [winner]
    assert list(tmp_path.glob(f".{destination.name}.*")) == []


@pytest.mark.parametrize(
    "writer_name,writer",
    JSON_EXCLUSIVE_WRITERS,
    ids=[name for name, _ in JSON_EXCLUSIVE_WRITERS],
)
def test_exclusive_json_preserves_preexisting_destination(
    writer_name, writer, tmp_path: Path
) -> None:
    destination = tmp_path / f"{writer_name}.json"
    destination.write_bytes(b"preexisting-immutable-bytes")
    with pytest.raises(FileExistsError):
        writer(destination, {"replacement": True})
    assert destination.read_bytes() == b"preexisting-immutable-bytes"
    assert list(tmp_path.glob(f".{destination.name}.*")) == []


@pytest.mark.parametrize(
    "writer_name,writer",
    NPZ_EXCLUSIVE_WRITERS,
    ids=[name for name, _ in NPZ_EXCLUSIVE_WRITERS],
)
def test_exclusive_npz_preserves_preexisting_destination(
    writer_name, writer, tmp_path: Path
) -> None:
    destination = tmp_path / f"{writer_name}.npz"
    destination.write_bytes(b"preexisting-immutable-bytes")
    with pytest.raises(FileExistsError):
        writer(destination, {"replacement": np.asarray([1], dtype=np.int8)})
    assert destination.read_bytes() == b"preexisting-immutable-bytes"
    assert list(tmp_path.glob(f".{destination.name}.*")) == []


@pytest.mark.parametrize(
    "writer",
    (
        lambda path, value: study_common.atomic_json(path, value, exclusive=False),
        lambda path, value: analysis.atomic_json(path, value, exclusive=False),
        lambda path, value: latency.atomic_json(path, value, exclusive=False),
    ),
)
def test_nonexclusive_json_intentionally_retains_replace_semantics(
    writer, tmp_path: Path
) -> None:
    destination = tmp_path / "mutable.json"
    writer(destination, {"version": 1})
    writer(destination, {"version": 2})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 2}


@pytest.mark.parametrize(
    "writer",
    (
        lambda path, value: study_common.atomic_npz(path, value, exclusive=False),
        lambda path, value: analysis.atomic_npz(path, value, exclusive=False),
    ),
)
def test_nonexclusive_npz_intentionally_retains_replace_semantics(
    writer, tmp_path: Path
) -> None:
    destination = tmp_path / "mutable.npz"
    writer(destination, {"version": np.asarray([1], dtype=np.int8)})
    writer(destination, {"version": np.asarray([2], dtype=np.int8)})
    with np.load(destination, allow_pickle=False) as stored:
        assert stored["version"].tolist() == [2]
