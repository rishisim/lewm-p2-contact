#!/usr/bin/env python3
"""Authenticate and inherit the immutable v003 fit role without decoding arrays.

The source role is a completed, fixed dataset role whose controller count was
not committed because provenance validation stopped on a path/schema
classification defect.  This module inventories every retained fit-role byte,
checks the closed namespace against the source manifests, and provides a
read-only path map for v008.  It never creates, replaces, or decodes an outcome
array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ATTEMPT_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ATTEMPT_ROOT.parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
SOURCE_ATTEMPT = "v003"
INTERMEDIATE_ATTEMPT = "v004"
FORWARD_SOURCE_ATTEMPT = "v005"
CURRENT_SOURCE_ATTEMPT = "v006"
ACTIVATION_SOURCE_ATTEMPT = "v007"
TARGET_ATTEMPT = "v008"
SOURCE_ROOT = STUDY_ROOT / "attempts" / SOURCE_ATTEMPT
SOURCE_FIT_ROOT = SOURCE_ROOT / "data/fit"
SOURCE_INTENT_ROOT = SOURCE_ROOT / "data/persistence_intents/fit"
SOURCE_INVALIDITY_PATH = SOURCE_ROOT / "audit/v003_procedural_invalidity.json"
SOURCE_SEAL_PATH = SOURCE_ROOT / "audit/pre_data_inheritance_seal.json"
SOURCE_RECEIPT_PATH = SOURCE_ROOT / "audit/version_forward_transaction_receipt.json"
SOURCE_LEDGER_PATH = SOURCE_ROOT / "cohort_seed_ledger.json"
SOURCE_REGISTRY_PATH = SOURCE_ROOT / "data/replacement_registry.json"
INVENTORY_PATH = ATTEMPT_ROOT / "audit/inherited_fit_inventory.json"
STATE_PATH = STUDY_ROOT / "STATE.json"
RESEARCH_LEDGER_PATH = STUDY_ROOT / "RESEARCH_LEDGER.jsonl"

REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
EPISODES_PER_REGIME = 300
EXPECTED_FILE_COUNT = 6013
EXPECTED_TOTAL_EPISODES = 1200
EXPECTED_SOURCE_LEDGER_SHA256 = (
    "1b7f524ed11cf787e67e4887590d985cfa4eb928e333e7adb7667e48a0b8785d"
)
EXPECTED_SOURCE_INVALIDITY_SHA256 = (
    "25be74578dc22eb2c1b4a3cde6b0dbd7e5a788d1bcb146f251b76fa40bc95165"
)
EXPECTED_SOURCE_SEAL_SHA256 = (
    "74018164e9e208babd7a3b420053a58e665e98694e4116d5a4f1cf70f143d943"
)
EXPECTED_SOURCE_RECEIPT_SHA256 = (
    "8cdd1a24aacdaecf4eda85fb5b9119536d88bec92e0cc64db6a3bcc7d9ecb262"
)
EXPECTED_SOURCE_REGISTRY_SHA256 = (
    "7aa18ee9ab56e45a3f3aac8ff95ba75978debb2e72368e5ee9e78a1f27c2b871"
)


class FitInheritanceError(RuntimeError):
    """The immutable source fit role or its dataset-role lineage drifted."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FitInheritanceError(f"cannot read immutable JSON record: {path}") from error
    if not isinstance(value, dict):
        raise FitInheritanceError(f"immutable JSON record is not an object: {path}")
    return value


def _relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(REPO_ROOT.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise FitInheritanceError(f"fit inheritance path escapes repository: {path}") from error


def _regular_record(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FitInheritanceError(f"fit inheritance file is absent: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FitInheritanceError(f"fit inheritance file is not a regular file: {path}")
    if metadata.st_nlink != 1:
        raise FitInheritanceError(f"fit inheritance file has an inode alias: {path}")
    return {
        "path": _relative(path),
        "bytes": int(metadata.st_size),
        "sha256": _sha256_file(path),
    }


def _source_header() -> tuple[dict[str, Any], dict[str, Any]]:
    if ATTEMPT_ROOT.name != TARGET_ATTEMPT:
        raise FitInheritanceError("fit inheritance implementation is outside v008")
    invalidity = _read_object(SOURCE_INVALIDITY_PATH)
    state = _read_object(STATE_PATH)
    counts = invalidity.get("controller_outcome_counts")
    if (
        invalidity.get("attempt") != SOURCE_ATTEMPT
        or invalidity.get("procedural_invalidity") is not True
        or invalidity.get("authoritative") is not True
        or invalidity.get("current_state") != "FIT_COHORTS"
        or not isinstance(counts, Mapping)
        or counts
        != {
            "confirmation_outcome_episodes_executed": 0,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcomes_opened_for_analysis": False,
            "fit_outcome_episodes": 0,
            "selection_outcome_episodes": 0,
            "smoke_outcome_episodes": 0,
        }
    ):
        raise FitInheritanceError("source invalidity/controller checkpoint drift")
    frozen = invalidity.get("frozen_evidence")
    if not isinstance(frozen, Mapping):
        raise FitInheritanceError("source invalidity frozen evidence is absent")
    active_attempt = state.get("active_attempt")
    if active_attempt == SOURCE_ATTEMPT:
        if (
            state.get("current_state") != "FIT_COHORTS"
            or any(state.get(key) != value for key, value in counts.items())
            or _sha256_file(STATE_PATH)
            != frozen.get("controller_state_sha256_at_failure")
            or _sha256_file(RESEARCH_LEDGER_PATH)
            != frozen.get("research_ledger_sha256_at_failure")
        ):
            raise FitInheritanceError("live source controller checkpoint drift")
    elif active_attempt in {
        INTERMEDIATE_ATTEMPT,
        FORWARD_SOURCE_ATTEMPT,
        CURRENT_SOURCE_ATTEMPT,
        ACTIVATION_SOURCE_ATTEMPT,
        TARGET_ATTEMPT,
    }:
        lineage = state.get("version_forward_lineage")
        source_edges = [
            edge
            for edge in lineage
            if isinstance(edge, Mapping)
            and edge.get("old_attempt") == SOURCE_ATTEMPT
            and edge.get("new_attempt") == INTERMEDIATE_ATTEMPT
            and edge.get("resume_state") == "FIT_COHORTS"
        ] if isinstance(lineage, list) else []
        expected_tip = {
            INTERMEDIATE_ATTEMPT: (SOURCE_ATTEMPT, INTERMEDIATE_ATTEMPT),
            FORWARD_SOURCE_ATTEMPT: (
                INTERMEDIATE_ATTEMPT,
                FORWARD_SOURCE_ATTEMPT,
            ),
            CURRENT_SOURCE_ATTEMPT: (
                FORWARD_SOURCE_ATTEMPT,
                CURRENT_SOURCE_ATTEMPT,
            ),
            ACTIVATION_SOURCE_ATTEMPT: (
                CURRENT_SOURCE_ATTEMPT,
                ACTIVATION_SOURCE_ATTEMPT,
            ),
            TARGET_ATTEMPT: (ACTIVATION_SOURCE_ATTEMPT, TARGET_ATTEMPT),
        }[active_attempt]
        expected_resume = (
            "SELECTION_COHORTS"
            if active_attempt in {ACTIVATION_SOURCE_ATTEMPT, TARGET_ATTEMPT}
            else "FIT_COHORTS"
        )
        expected_state = (
            "SELECTION_COHORTS"
            if active_attempt
            in {
                CURRENT_SOURCE_ATTEMPT,
                ACTIVATION_SOURCE_ATTEMPT,
                TARGET_ATTEMPT,
            }
            else "FIT_COHORTS"
        )
        expected_counts = dict(counts)
        if active_attempt in {
            CURRENT_SOURCE_ATTEMPT,
            ACTIVATION_SOURCE_ATTEMPT,
            TARGET_ATTEMPT,
        }:
            expected_counts["fit_outcome_episodes"] = EXPECTED_TOTAL_EPISODES
        if (
            not isinstance(lineage, list)
            or not lineage
            or not isinstance(lineage[-1], Mapping)
            or len(source_edges) != 1
            or lineage[-1].get("old_attempt") != expected_tip[0]
            or lineage[-1].get("new_attempt") != expected_tip[1]
            or lineage[-1].get("resume_state") != expected_resume
            or state.get("current_state") != expected_state
            or any(
                state.get(key) != value
                for key, value in expected_counts.items()
            )
        ):
            raise FitInheritanceError(
                "active descendant lacks the immutable source fit-role lineage"
            )
    else:
        raise FitInheritanceError("controller attempt is outside fit-role inheritance lineage")
    fixed = {
        SOURCE_INVALIDITY_PATH: EXPECTED_SOURCE_INVALIDITY_SHA256,
        SOURCE_SEAL_PATH: EXPECTED_SOURCE_SEAL_SHA256,
        SOURCE_RECEIPT_PATH: EXPECTED_SOURCE_RECEIPT_SHA256,
        SOURCE_LEDGER_PATH: EXPECTED_SOURCE_LEDGER_SHA256,
        SOURCE_REGISTRY_PATH: EXPECTED_SOURCE_REGISTRY_SHA256,
    }
    for path, expected in fixed.items():
        if _sha256_file(path) != expected:
            raise FitInheritanceError(f"source lineage hash drift: {path}")
    later_roots = (
        SOURCE_ROOT / "data/selection",
        SOURCE_ROOT / "data/smoke",
        SOURCE_ROOT / "data/confirmation",
        SOURCE_ROOT / "fit",
        SOURCE_ROOT / "selection",
        SOURCE_ROOT / "freeze",
        SOURCE_ROOT / "metrics",
    )
    later_files = sorted(
        _relative(path)
        for root in later_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if later_files:
        raise FitInheritanceError(f"source attempt has later-role artifacts: {later_files}")
    return invalidity, state


def _manifest_paths() -> tuple[set[Path], dict[str, dict[str, Any]]]:
    expected: set[Path] = {SOURCE_REGISTRY_PATH}
    regimes: dict[str, dict[str, Any]] = {}
    invalidity = _read_object(SOURCE_INVALIDITY_PATH)
    invalidity_regimes = invalidity.get("fit_evidence", {}).get("regimes")
    if not isinstance(invalidity_regimes, Mapping) or set(invalidity_regimes) != set(REGIMES):
        raise FitInheritanceError("source invalidity fit-regime census drift")
    for regime in REGIMES:
        root = SOURCE_FIT_ROOT / regime
        raw_path = root / "raw_manifest.json"
        execution_path = root / "execution_manifest.json"
        aggregate_path = root / "role.npz"
        raw = _read_object(raw_path)
        execution = _read_object(execution_path)
        declared = invalidity_regimes[regime]
        if not isinstance(declared, Mapping):
            raise FitInheritanceError(f"source invalidity regime is not an object: {regime}")
        fixed_hashes = {
            raw_path: declared.get("raw_manifest_sha256"),
            execution_path: declared.get("execution_manifest_sha256"),
            aggregate_path: declared.get("role_npz_sha256"),
        }
        if any(_sha256_file(path) != expected_hash for path, expected_hash in fixed_hashes.items()):
            raise FitInheritanceError(f"source fit top-level hash drift: {regime}")
        raw_records = raw.get("episodes")
        execution_records = execution.get("episodes")
        if (
            raw.get("attempt") != SOURCE_ATTEMPT
            or raw.get("role") != "fit"
            or raw.get("regime") != regime
            or raw.get("complete") is not True
            or raw.get("episode_count") != EPISODES_PER_REGIME
            or not isinstance(raw_records, list)
            or len(raw_records) != EPISODES_PER_REGIME
            or execution.get("attempt") != SOURCE_ATTEMPT
            or execution.get("role") != "fit"
            or execution.get("regime") != regime
            or execution.get("complete") is not True
            or execution.get("episode_count") != EPISODES_PER_REGIME
            or not isinstance(execution_records, list)
            or len(execution_records) != EPISODES_PER_REGIME
        ):
            raise FitInheritanceError(f"source fit manifest header drift: {regime}")
        raw_ids = [record.get("episode_id") for record in raw_records if isinstance(record, Mapping)]
        execution_ids = [
            record.get("episode_id") for record in execution_records if isinstance(record, Mapping)
        ]
        if (
            len(raw_ids) != EPISODES_PER_REGIME
            or len(set(raw_ids)) != EPISODES_PER_REGIME
            or raw_ids != execution_ids
        ):
            raise FitInheritanceError(f"source fit episode-order drift: {regime}")
        expected.update((raw_path, execution_path, aggregate_path))
        for slot, (raw_record, execution_record) in enumerate(
            zip(raw_records, execution_records, strict=True)
        ):
            if not isinstance(raw_record, Mapping) or not isinstance(execution_record, Mapping):
                raise FitInheritanceError(f"source fit episode record drift: {regime}/{slot}")
            episode_id = raw_ids[slot]
            if (
                not isinstance(episode_id, str)
                or raw_record.get("slot") != slot
                or execution_record.get("slot") != slot
            ):
                raise FitInheritanceError(f"source fit episode slot drift: {regime}/{slot}")
            raw_npz = REPO_ROOT / str(raw_record.get("raw_path"))
            raw_sidecar = root / "raw" / f"{episode_id}.json"
            execution_npz = REPO_ROOT / str(execution_record.get("path"))
            execution_sidecar = REPO_ROOT / str(execution_record.get("sidecar_path"))
            intent = REPO_ROOT / str(raw_record.get("persistence_intent_path"))
            required = (raw_npz, raw_sidecar, execution_npz, execution_sidecar, intent)
            expected.update(required)
            hash_links = (
                (raw_npz, raw_record.get("raw_sha256")),
                (raw_sidecar, execution_record.get("source_raw_sidecar_sha256")),
                (execution_npz, execution_record.get("sha256")),
                (execution_sidecar, execution_record.get("sidecar_sha256")),
                (intent, raw_record.get("persistence_intent_sha256")),
            )
            if (
                execution_record.get("source_raw_path") != raw_record.get("raw_path")
                or execution_record.get("source_raw_sha256") != raw_record.get("raw_sha256")
                or execution_record.get("source_raw_sidecar_path") != _relative(raw_sidecar)
            ):
                raise FitInheritanceError(
                    f"source fit raw/execution provenance drift: {regime}/{slot}"
                )
            if any(_sha256_file(path) != expected_hash for path, expected_hash in hash_links):
                raise FitInheritanceError(f"source fit episode hash drift: {regime}/{slot}")
            if any(not path.resolve().is_relative_to(SOURCE_ROOT.resolve()) for path in required):
                raise FitInheritanceError(f"source fit episode path escapes source attempt: {regime}/{slot}")
        regimes[regime] = {
            "episode_count": EPISODES_PER_REGIME,
            "raw_manifest": _regular_record(raw_path),
            "execution_manifest": _regular_record(execution_path),
            "aggregate": _regular_record(aggregate_path),
            "episode_ids_sha256": hashlib.sha256(_canonical_bytes(raw_ids)).hexdigest(),
        }
    return expected, regimes


def build_inventory(*, created_unix_ns: int | None = None) -> dict[str, Any]:
    """Recompute the complete source-role byte inventory without array decoding."""

    invalidity, state = _source_header()
    expected, regimes = _manifest_paths()
    observed = {
        path
        for root in (SOURCE_FIT_ROOT, SOURCE_INTENT_ROOT)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    } | {SOURCE_REGISTRY_PATH}
    if observed != expected or len(observed) != EXPECTED_FILE_COUNT:
        raise FitInheritanceError(
            "source fit namespace census drift: "
            f"missing={sorted(map(str, expected-observed))}, "
            f"unexpected={sorted(map(str, observed-expected))}, count={len(observed)}"
        )
    records = [_regular_record(path) for path in sorted(observed, key=lambda item: str(item))]
    identities: dict[tuple[int, int], str] = {}
    for path in sorted(observed, key=lambda item: str(item)):
        metadata = path.stat()
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in identities:
            raise FitInheritanceError(
                f"source fit namespace has inode aliases: {identities[identity]} and {path}"
            )
        identities[identity] = str(path)
    inventory_digest = hashlib.sha256(_canonical_bytes(records)).hexdigest()
    return {
        "schema_version": 1,
        "artifact_type": "immutable_fit_role_inheritance_inventory",
        "attempt": TARGET_ATTEMPT,
        "source_attempt": SOURCE_ATTEMPT,
        "source_state": "FIT_COHORTS",
        "created_unix_ns": time.time_ns() if created_unix_ns is None else created_unix_ns,
        "episode_count": EXPECTED_TOTAL_EPISODES,
        "episodes_per_regime": EPISODES_PER_REGIME,
        "regime_order": list(REGIMES),
        "regimes": regimes,
        "file_count": len(records),
        "files": records,
        "files_canonical_sha256": inventory_digest,
        "source_invalidity": _regular_record(SOURCE_INVALIDITY_PATH),
        "source_pre_data_seal": _regular_record(SOURCE_SEAL_PATH),
        "source_version_forward_receipt": _regular_record(SOURCE_RECEIPT_PATH),
        "source_cohort_seed_ledger": _regular_record(SOURCE_LEDGER_PATH),
        "source_replacement_registry": _regular_record(SOURCE_REGISTRY_PATH),
        "source_controller_state_sha256": invalidity["frozen_evidence"][
            "controller_state_sha256_at_failure"
        ],
        "source_research_ledger_sha256": invalidity["frozen_evidence"][
            "research_ledger_sha256_at_failure"
        ],
        "controller_fit_outcome_episodes_at_inventory": invalidity[
            "controller_outcome_counts"
        ]["fit_outcome_episodes"],
        "later_role_artifact_file_count": 0,
        "replacement_record_count": 0,
        "dataset_role_separation_verified": True,
        "source_bytes_rehashed": True,
        "outcome_arrays_opened": False,
        "passed": True,
    }


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_inventory_exclusive(path: Path = INVENTORY_PATH) -> dict[str, Any]:
    value = build_inventory()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        raise FileExistsError(f"immutable fit inheritance inventory already exists: {output}")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_pretty_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return value


def verify_inventory(path: Path = INVENTORY_PATH) -> dict[str, Any]:
    observed = _read_object(path)
    created = observed.get("created_unix_ns")
    if type(created) is not int or created <= 0:
        raise FitInheritanceError("fit inheritance inventory timestamp drift")
    expected = build_inventory(created_unix_ns=created)
    if path.read_bytes() != _pretty_bytes(expected) or observed != expected:
        raise FitInheritanceError("fit inheritance inventory recomputation drift")
    return {
        "passed": True,
        "attempt": TARGET_ATTEMPT,
        "source_attempt": SOURCE_ATTEMPT,
        "inventory_path": _relative(path),
        "inventory_sha256": _sha256_file(path),
        "file_count": expected["file_count"],
        "episode_count": expected["episode_count"],
        "outcome_arrays_opened": False,
        "read_only": True,
    }


def aggregate_paths() -> dict[str, Path]:
    """Return authenticated read-only aggregate paths for the inherited fit role."""

    verify_inventory()
    return {regime: SOURCE_FIT_ROOT / regime / "role.npz" for regime in REGIMES}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "verify"))
    arguments = parser.parse_args(argv)
    if arguments.command == "write":
        value = write_inventory_exclusive()
        result = {
            "passed": True,
            "attempt": TARGET_ATTEMPT,
            "source_attempt": SOURCE_ATTEMPT,
            "inventory_path": _relative(INVENTORY_PATH),
            "inventory_sha256": _sha256_file(INVENTORY_PATH),
            "file_count": value["file_count"],
            "episode_count": value["episode_count"],
            "outcome_arrays_opened": False,
        }
    else:
        result = verify_inventory()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
