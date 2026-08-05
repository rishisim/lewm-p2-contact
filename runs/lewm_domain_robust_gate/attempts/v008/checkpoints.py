#!/usr/bin/env python3
"""Fail-closed, outcome-blind evidence checkpoints for smoke and confirmation.

This module never NumPy-loads a rollout or execution artifact.  It reads only
JSON manifests/sidecars, file metadata, and raw bytes for hashing.  In
particular, :func:`confirmation_input_seal` hashes every confirmation input
before :mod:`analysis` is allowed to open a confirmation target array.

The root state controller remains the sole owner of ``STATE.json`` and the
research ledger.  These functions create immutable evidence objects; callers
advance the state only after inspecting a passing object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from inherited_authorization import ACTIVE_ATTEMPT, SCIENCE_ATTEMPT
from study_common import read_verified_controller


ATTEMPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ATTEMPT_ROOT.parents[3]
ATTEMPT = ACTIVE_ATTEMPT
DGP_ORDER = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ROWS_PER_EPISODE = 38
SMOKE_EPISODES_PER_DGP = 6
CONFIRMATION_MAX_EPISODES_PER_DGP = 4_500
CONFIRMATION_REPLACEMENTS_PER_DGP = 200
SEED_FIELDS = (
    "env_seed",
    "policy_seed",
    "oracle_np_seed",
    "action_space_seed",
)
ROLE_ORDER = ("fit", "selection", "smoke", "confirmation")
ROLE_PRIMARY_COUNTS = {
    "fit": 300,
    "selection": 500,
    "smoke": 6,
    "confirmation": CONFIRMATION_MAX_EPISODES_PER_DGP,
}
REGIME_SLUGS = {
    "native_plan": "np",
    "markov_oracle": "mo",
    "plan_action_noise_0p2": "n2",
    "plan_random_action_0p1": "r1",
}
V5_FIXED_WHITENING = (
    REPO_ROOT
    / "runs/lewm_v5_readiness_program/v5_package_versions/v004/freeze/whitening.npz"
)

FORBIDDEN_EVIDENCE_NAMES = (
    "contact",
    "privileged",
    "qpos",
    "qvel",
    "motion",
    "phase",
    "reward",
    "success",
)
PERSISTENCE_INTENT_STATUS = "valid_in_memory_episode_closed_for_raw_persistence"
ORPHAN_ADOPTION_RULE = (
    "adopt only after exact closed-intent, current authorization, prospective "
    "ledger, canonical path, nonlink identity, and fully recomputed two-array "
    "verification"
)
PERSISTENCE_INTENT_KEYS = {
    "schema_version",
    "attempt",
    "science_attempt",
    "created_unix_ns",
    "status",
    "role",
    "regime",
    "slot",
    "episode_id",
    "raw_path",
    "raw_sidecar_path",
    "persistence_intent_path",
    "dgp_matrix_path",
    "dgp_matrix_sha256",
    "cohort_seed_ledger_path",
    "cohort_seed_ledger_sha256",
    "destination_seed_record",
    "seed_source_pool",
    "seed_source_slot",
    "seed_source_record",
    "seed_source_episode_id",
    "replacement_used",
    "replacement_claim_index",
    "replacement_claim_sha256",
    *SEED_FIELDS,
    "arrays",
    "initial_pixels_sha256",
    "generation_audit",
    "authorization_seal",
    "orphan_adoption_rule",
}


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_repo(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))


def resolve_repo_relative(raw: str | Path) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"unsafe repository-relative path: {raw}")
    resolved = (REPO_ROOT / candidate).resolve(strict=False)
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise RuntimeError(f"path escapes repository: {raw}")
    return resolved


def atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = True) -> None:
    """Durably publish JSON, with an atomic no-replace commit when exclusive.

    A preflight ``exists`` check followed by ``os.replace`` is racy: two
    writers can both pass the check and the second silently overwrites the
    first immutable checkpoint.  The temporary file is therefore committed
    with ``link(2)`` for the exclusive case.  Creating the destination hard
    link is one atomic no-replace operation on the same filesystem.  The
    temporary name is then removed and the parent directory is fsynced so the
    directory entry is durable as well as the file contents.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"immutable evidence already exists: {path}"
                ) from error
            temporary.unlink()
        else:
            os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = manifest.get("episodes")
    if not isinstance(records, list) or any(not isinstance(item, Mapping) for item in records):
        raise RuntimeError("manifest episodes must be a list of objects")
    return records


def _record_path(record: Mapping[str, Any], *, sidecar: bool = False) -> tuple[str, str]:
    path_names = (
        ("sidecar_path", "sidecar_sha256"),
        ("metadata_path", "metadata_sha256"),
    ) if sidecar else (
        ("path", "sha256"),
        ("raw_path", "raw_sha256"),
        ("execution_path", "execution_sha256"),
    )
    found = [(record[name], record[hash_name]) for name, hash_name in path_names if name in record and hash_name in record]
    if len(found) != 1:
        kind = "sidecar" if sidecar else "array"
        raise RuntimeError(f"episode record must identify exactly one {kind} path/hash pair")
    return str(found[0][0]), str(found[0][1])


def _check_file(path_text: str, expected_sha256: str, *, required_root: Path) -> dict[str, Any]:
    path = resolve_repo_relative(path_text)
    lexical = REPO_ROOT / Path(path_text)
    if lexical.is_symlink() or any(parent.is_symlink() for parent in lexical.parents if parent != REPO_ROOT.parent):
        raise RuntimeError(f"symlink is forbidden in evidence path: {path_text}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.is_relative_to(required_root.resolve()):
        raise RuntimeError(f"artifact outside required role/DGP directory: {path_text}")
    observed = sha256_file(path)
    return {
        "path": path_text,
        "sha256": observed,
        "expected_sha256": expected_sha256,
        "bytes": path.stat().st_size,
        "hash_matches": observed == expected_sha256,
        "regular_file": path.is_file(),
        "symlink": False,
    }


def _boolean_assertion(
    manifest: Mapping[str, Any],
    positive_names: Sequence[str],
    *,
    default: bool = False,
) -> bool:
    found = [manifest[name] for name in positive_names if name in manifest]
    return bool(found[0]) if len(found) == 1 else default


def _negative_assertions(manifest: Mapping[str, Any]) -> bool:
    """Require every present forbidden-use/open assertion to be false."""

    for key, value in manifest.items():
        lowered = str(key).lower()
        if any(token in lowered for token in FORBIDDEN_EVIDENCE_NAMES):
            if lowered.endswith(
                ("_opened", "_loaded", "_used", "_inspected", "_materialized")
            ) and value is not False:
                return False
        if any(token in lowered for token in ("v3_test", "combined_v3", "released_hdf5")):
            if value is not False:
                return False
    return True


def verify_raw_manifest(
    path: Path,
    *,
    role: str,
    regime: str,
    expected_episodes: int,
) -> dict[str, Any]:
    """Rehash one pixels/action-only raw cohort without opening its NPZs."""

    if role == "confirmation":
        return _verify_confirmation_raw_manifest(
            path,
            regime=regime,
            expected_episodes=expected_episodes,
        )

    manifest = read_json(path)
    records = _records(manifest)
    # The caller supplies the canonical role/DGP manifest path.  Constraining
    # every referenced artifact to that manifest's directory also keeps this
    # verifier usable in isolated synthetic test roots.
    required_root = path.parent
    file_records: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    slots: list[int] = []
    sidecar_count = 0
    for record in records:
        path_text, expected_hash = _record_path(record)
        file_records.append(
            _check_file(path_text, expected_hash, required_root=required_root)
        )
        if "sidecar_path" in record or "metadata_path" in record:
            sidecar_path, sidecar_hash = _record_path(record, sidecar=True)
            file_records.append(
                _check_file(sidecar_path, sidecar_hash, required_root=required_root)
            )
            sidecar_count += 1
        episode_ids.append(str(record["episode_id"]))
        slots.append(int(record["slot"]))
    # ``role`` is the sole accepted role discriminator.  In particular, never
    # inspect a simulator ``phase`` field as an alias: phase is scientifically
    # forbidden before terminal interpretation.
    role_value = manifest.get("role")
    input_key_fields = [
        manifest[name]
        for name in (
            "materialized_array_keys",
            "raw_array_keys",
            "raw_archive_members",
        )
        if name in manifest
    ]
    input_keys = input_key_fields[0] if len(input_key_fields) == 1 else ()
    replacement_count = int(
        manifest.get("replacement_count", manifest.get("replacements_used", 0))
    )
    checks = {
        "complete": _boolean_assertion(manifest, ("complete", "passed")),
        "role": role_value == role,
        "regime": manifest.get("regime") == regime,
        "episode_count": int(manifest.get("episode_count", -1)) == expected_episodes,
        "record_count": len(records) == expected_episodes,
        "slots_exact": slots == list(range(expected_episodes)),
        "episode_ids_unique": len(set(episode_ids)) == expected_episodes,
        "all_files_hash_match": all(item["hash_matches"] for item in file_records),
        "sidecars_complete_if_declared": sidecar_count in (0, expected_episodes),
        "pixels_action_only": len(input_key_fields) == 1
        and isinstance(input_keys, list)
        and sorted(input_keys) == ["action", "pixels"],
        "role_isolation": manifest.get("role_isolation") is True,
        "smoke_permanently_excluded_if_applicable": role != "smoke"
        or manifest.get("smoke_permanently_excluded") is True,
        "replacement_count_bounded": 0 <= replacement_count <= 200,
        "outcome_blind_retention": manifest.get(
            "contact_motion_phase_reward_success_used_for_retention", False
        )
        is False
        and manifest.get("retention_contract")
        == "pixels_action_shapes_finiteness_and_local_step_count_only",
        "forbidden_sources_excluded": _negative_assertions(manifest),
    }
    return {
        "manifest": {
            "path": relative_to_repo(path),
            "sha256": sha256_file(path),
            "created_unix_ns": int(manifest.get("created_unix_ns", 0)),
        },
        "role": role,
        "regime": regime,
        "episode_ids": episode_ids,
        "replacement_count": replacement_count,
        "files": file_records,
        "checks": checks,
        "passed": all(checks.values()),
    }


def verify_execution_manifest(
    path: Path,
    *,
    role: str,
    regime: str,
    expected_episodes: int,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash execution parts without NumPy-loading any target array."""

    if role == "confirmation":
        return _verify_confirmation_execution_manifest(
            path,
            regime=regime,
            expected_episodes=expected_episodes,
            raw=raw,
        )

    manifest = read_json(path)
    records = _records(manifest)
    required_root = path.parent
    files: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    slots: list[int] = []
    for record in records:
        path_text, expected_hash = _record_path(record)
        files.append(_check_file(path_text, expected_hash, required_root=required_root))
        if "sidecar_path" in record or "metadata_path" in record:
            sidecar_path, sidecar_hash = _record_path(record, sidecar=True)
            sidecars.append(
                _check_file(sidecar_path, sidecar_hash, required_root=required_root)
            )
        episode_ids.append(str(record["episode_id"]))
        slots.append(int(record["slot"]))
    role_value = manifest.get("role")
    module_before = manifest.get("module_before")
    module_after = manifest.get("module_after")
    loaded_keys = manifest.get("loaded_input_keys")
    expected_rows = expected_episodes * ROWS_PER_EPISODE
    checks = {
        "complete": _boolean_assertion(manifest, ("complete", "passed")),
        "role": role_value == role,
        "regime": manifest.get("regime") == regime,
        "episode_count": int(manifest.get("episode_count", -1)) == expected_episodes,
        "row_count": int(manifest.get("row_count", -1)) == expected_rows,
        "record_count": len(records) == expected_episodes,
        "slots_exact": slots == list(range(expected_episodes)),
        "episode_ids_unique": len(set(episode_ids)) == expected_episodes,
        "raw_execution_episode_ids_exact": episode_ids == raw["episode_ids"],
        "all_files_hash_match": all(item["hash_matches"] for item in files),
        "sidecars_complete_if_declared": len(sidecars) in (0, expected_episodes),
        "all_sidecar_hashes_match": all(
            item["hash_matches"] for item in sidecars
        ),
        "sparse_dense_equivalence": manifest.get("all_equivalence_checks_passed") is True,
        "module_frozen": module_before == module_after
        and isinstance(module_after, Mapping)
        and module_after.get("passed") is True,
        "no_gradients": manifest.get("no_gradients") is True,
        "pixels_action_input_only": isinstance(loaded_keys, list)
        and sorted(loaded_keys) == ["action", "pixels"],
        "contact_or_privileged_not_materialized": manifest.get(
            "contact_or_privileged_materialized"
        )
        is False,
        "target_loss_not_computed_during_execution": manifest.get(
            "target_loss_computed_during_execution"
        )
        is False,
        "forbidden_sources_excluded": _negative_assertions(manifest),
    }
    if "source_raw_manifest_sha256" in manifest:
        checks["source_raw_manifest_hash"] = (
            manifest["source_raw_manifest_sha256"] == raw["manifest"]["sha256"]
        )
    return {
        "manifest": {
            "path": relative_to_repo(path),
            "sha256": sha256_file(path),
            "created_unix_ns": int(manifest.get("created_unix_ns", 0)),
        },
        "role": role,
        "regime": regime,
        "episode_ids": episode_ids,
        "files": files,
        "sidecars": sidecars,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _raw_path(root: Path, role: str, regime: str) -> Path:
    return root / "data" / role / regime / "raw_manifest.json"


def _execution_path(root: Path, role: str, regime: str) -> Path:
    return root / "data" / role / regime / "execution_manifest.json"


def _all_unique(values: Iterable[str]) -> bool:
    items = list(values)
    return len(items) == len(set(items))


def _validated_sealed_files(files: Mapping[str, Any]) -> dict[str, str]:
    """Rehash a repository-relative map and reject links or inode aliases."""

    if not files:
        raise RuntimeError("sealed_files is empty")
    normalized: dict[str, str] = {}
    identities: dict[tuple[int, int], str] = {}
    for raw_path, raw_hash in sorted(files.items()):
        relative = Path(str(raw_path))
        digest = str(raw_hash)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != str(raw_path)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"noncanonical sealed file record: {raw_path}")
        path = REPO_ROOT / relative
        current = REPO_ROOT
        for part in relative.parts:
            current = current / part
            try:
                metadata = current.lstat()
            except FileNotFoundError as exc:
                raise RuntimeError(f"sealed path missing: {raw_path}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"sealed symlink forbidden: {raw_path}")
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"sealed path is not a regular file: {raw_path}")
        observed = sha256_file(path)
        if observed != digest:
            raise RuntimeError(f"sealed file hash drift: {raw_path}")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in identities:
            raise RuntimeError(
                f"sealed file inode alias: {identities[identity]} and {raw_path}"
            )
        identities[identity] = str(raw_path)
        normalized[str(raw_path)] = observed
    return normalized


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{label} schema drift: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _attempt_root_from_manifest(
    path: Path, *, role: str, regime: str, filename: str
) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = path.absolute()
    if len(path.parents) < 4:
        raise RuntimeError(f"manifest path is too shallow: {path}")
    attempt_root = path.parents[3]
    expected = attempt_root / "data" / role / regime / filename
    if path != expected or attempt_root.name != ATTEMPT:
        raise RuntimeError(
            f"noncanonical {role}/{regime} manifest path for {ATTEMPT}: {path}"
        )
    if regime not in DGP_ORDER:
        raise RuntimeError(f"unsealed confirmation DGP: {regime}")
    return attempt_root


def _canonical_relative(path: Path) -> str:
    try:
        return Path(path).relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise RuntimeError(f"path escapes repository: {path}") from error


def _strict_file_link(
    raw_path: Any,
    raw_sha256: Any,
    *,
    expected_path: Path,
    identities: dict[tuple[int, int], str],
) -> dict[str, Any]:
    expected_relative = _canonical_relative(expected_path)
    if not isinstance(raw_path, str):
        raise RuntimeError(f"file link path is not a string: {raw_path!r}")
    relative = Path(raw_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "." in relative.parts
        or "\\" in raw_path
        or relative.as_posix() != raw_path
        or raw_path != expected_relative
    ):
        raise RuntimeError(
            f"noncanonical file link: observed={raw_path!r} expected={expected_relative!r}"
        )
    expected_hash = _strict_sha256(raw_sha256, f"hash for {raw_path}")
    current = REPO_ROOT
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"linked file is absent: {raw_path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink is forbidden in confirmation path: {raw_path}")
    metadata = expected_path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"confirmation artifact is not regular: {raw_path}")
    if int(metadata.st_nlink) != 1:
        raise RuntimeError(f"confirmation artifact has an inode alias: {raw_path}")
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    previous = identities.get(identity)
    if previous is not None and previous != raw_path:
        raise RuntimeError(f"confirmation inode alias: {previous} and {raw_path}")
    identities[identity] = raw_path
    observed = sha256_file(expected_path)
    if observed != expected_hash:
        raise RuntimeError(f"confirmation artifact hash drift: {raw_path}")
    return {
        "path": raw_path,
        "sha256": observed,
        "expected_sha256": expected_hash,
        "bytes": int(metadata.st_size),
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "hash_matches": True,
        "regular_file": True,
        "symlink": False,
        "inode_alias": False,
    }


def _observed_file_link(
    path: Path, *, identities: dict[tuple[int, int], str]
) -> dict[str, Any]:
    return _strict_file_link(
        _canonical_relative(path),
        sha256_file(path),
        expected_path=path,
        identities=identities,
    )


def _expected_episode_id(regime: str, pool: str, index: int) -> str:
    if pool not in {"primary", "replacements"}:
        raise RuntimeError(f"unknown seed-ledger pool: {pool}")
    kind = "p" if pool == "primary" else "r"
    width = 4 if pool == "primary" else 3
    return f"drgv001-{REGIME_SLUGS[regime]}-cf-{kind}-{index:0{width}d}"


def _validate_seed_record(
    value: Any, *, regime: str, pool: str, index: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"seed-ledger record is not an object: {regime}/{pool}/{index}")
    expected_keys = {"slot", "episode_id", *SEED_FIELDS}
    _exact_keys(value, expected_keys, f"seed-ledger {regime}/{pool}/{index}")
    expected_id = _expected_episode_id(regime, pool, index)
    if value.get("episode_id") != expected_id:
        raise RuntimeError(f"seed-ledger episode ID drift: {regime}/{pool}/{index}")
    slot = _strict_int(value.get("slot"), "seed-ledger slot")
    if slot != index:
        raise RuntimeError(f"seed-ledger slot order drift: {regime}/{pool}/{index}")
    result = {"slot": slot, "episode_id": expected_id}
    for field in SEED_FIELDS:
        result[field] = _strict_int(value.get(field), f"seed-ledger {field}")
    return result


def _exact_seal_link(path: Path, checkpoint_state: str) -> dict[str, str]:
    return {
        "path": _canonical_relative(path),
        "sha256": sha256_file(path),
        "checkpoint_state": checkpoint_state,
    }


def _canonical_record_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes repository: {path}") from error
    current = REPO_ROOT
    for part in relative.parts:
        current = current / part
        try:
            ancestor_info = current.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"missing {label}: {path}") from error
        if stat.S_ISLNK(ancestor_info.st_mode):
            raise RuntimeError(f"{label} has a symlink path component: {path}")
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"missing {label}: {path}") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
    ):
        raise RuntimeError(f"{label} is linked, aliased, or non-regular: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {label}: {path}") from error
    if not isinstance(value, dict) or raw != (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"):
        raise RuntimeError(f"noncanonical {label}: {path}")
    return value


def _failure_chain_v2(
    path: Path,
    *,
    role: str,
    regime: str,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
    registry_claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
    ):
        raise RuntimeError("rollout failure log is linked, aliased, or non-regular")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise RuntimeError("rollout failure chain is empty or partial")
    primary_by_key = {
        (str(item["episode_id"]), int(item["slot"])): item for item in primary
    }
    replacement_by_id = {
        str(item["episode_id"]): item for item in replacements
    }
    previous = f"authorization:{authorization['sha256']}"
    seen_attempts: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    expected_keys = {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "classification",
        "role",
        "regime",
        "slot",
        "episode_id",
        "attempted_seed_source_episode_id",
        "replacement_used",
        "replacement_claim_index",
        "replacement_claim_sha256",
        *SEED_FIELDS,
        "exception_type",
        "exception_message",
        "traceback",
        "replacement_permitted",
        "retention_contract",
        "authorization_seal",
        "seq",
        "prev_sha256",
        "record_sha256",
    }
    for sequence, encoded in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid failure record {sequence}") from error
        if not isinstance(value, Mapping) or encoded != json.dumps(
            value, sort_keys=True, allow_nan=False
        ).encode("utf-8"):
            raise RuntimeError(f"noncanonical failure record {sequence}")
        _exact_keys(value, expected_keys, f"failure record {sequence}")
        body = dict(value)
        observed_hash = body.pop("record_sha256")
        destination = primary_by_key.get(
            (str(value.get("episode_id")), value.get("slot"))
        )
        source_id = str(value.get("attempted_seed_source_episode_id"))
        replacement_used = value.get("replacement_used")
        candidate = (
            replacement_by_id.get(source_id)
            if replacement_used is True
            else destination
        )
        checks = {
            "schema": type(value.get("schema_version")) is int
            and value.get("schema_version") == 2,
            "attempt": value.get("attempt") == ATTEMPT,
            "timestamp": type(value.get("created_unix_ns")) is int
            and int(value["created_unix_ns"]) > 0,
            "classification": value.get("classification")
            == "mechanical_rollout_exception",
            "role": value.get("role") == role,
            "regime": value.get("regime") == regime,
            "slot": type(value.get("slot")) is int,
            "destination": destination is not None,
            "replacement_flag": type(replacement_used) is bool,
            "candidate": candidate is not None,
            "authorization": value.get("authorization_seal")
            == dict(authorization),
            "sequence": type(value.get("seq")) is int
            and value.get("seq") == sequence,
            "previous": value.get("prev_sha256") == previous,
            "record_hash": observed_hash == _canonical_record_sha256(body),
            "exception_type": isinstance(value.get("exception_type"), str)
            and bool(value.get("exception_type")),
            "exception_message": isinstance(value.get("exception_message"), str),
            "traceback": isinstance(value.get("traceback"), str),
            "replacement_permitted": value.get("replacement_permitted") is True,
            "retention": value.get("retention_contract")
            == "pixels_action_shapes_finiteness_and_local_step_count_only",
        }
        if candidate is not None:
            checks["candidate_seeds"] = _seed_tuple(value) == _seed_tuple(candidate)
        if replacement_used is False:
            checks["primary_source"] = source_id == value.get("episode_id")
            checks["primary_claim"] = value.get("replacement_claim_index") is None
            checks["primary_claim_hash"] = (
                value.get("replacement_claim_sha256") is None
            )
        elif replacement_used is True:
            claim_index = value.get("replacement_claim_index")
            checks["claim_index"] = (
                type(claim_index) is int
                and 0 <= int(claim_index) < len(registry_claims)
            )
            claim = (
                registry_claims[int(claim_index)]
                if checks["claim_index"]
                else {}
            )
            checks["claim_hash"] = value.get("replacement_claim_sha256") == claim.get(
                "record_sha256"
            )
            checks["claim_destination"] = (
                claim.get("role") == role
                and claim.get("regime") == regime
                and claim.get("slot") == value.get("slot")
                and claim.get("episode_id") == value.get("episode_id")
            )
            checks["claim_source"] = (
                claim.get("replacement_episode_id") == source_id
                and _seed_tuple(claim) == _seed_tuple(value)
            )
        attempt_key = (str(value.get("episode_id")), source_id)
        checks["attempt_unique"] = attempt_key not in seen_attempts
        if not all(checks.values()):
            raise RuntimeError(
                f"rollout failure v2 authentication drift at line {sequence}: {checks}"
            )
        seen_attempts.add(attempt_key)
        previous = str(observed_hash)
        result.append(dict(value))
    return result


def _validate_replacement_registry_chain(
    attempt_root: Path,
    *,
    ledger: Mapping[str, Any],
    dgp: Mapping[str, Any],
    pre_confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    genesis_path = attempt_root / "data/replacement_registry.json"
    claims_root = attempt_root / "data/replacement_claims"
    genesis = _canonical_json_object(genesis_path, "replacement registry genesis")
    genesis_keys = {
        "schema_version",
        "record_type",
        "attempt",
        "science_attempt",
        "scope",
        "created_unix_ns",
        "dgp_matrix_path",
        "dgp_matrix_sha256",
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "regime_order",
        "role_order",
        "seed_field_order",
        "replacement_count_per_regime_per_role",
        "replacement_rule",
        "claims_directory",
        "claim_filename_format",
        "authorization_policy",
        "claim_contract",
        "record_sha256",
    }
    _exact_keys(genesis, genesis_keys, "replacement registry genesis")
    genesis_body = dict(genesis)
    genesis_record_hash = genesis_body.pop("record_sha256")
    inheritance_path = attempt_root / "audit/pre_data_inheritance_seal.json"
    pre_confirmation_path = attempt_root / "audit/pre_confirmation_package_seal.json"
    authorization_policy = {
        role: {
            "path": _canonical_relative(
                inheritance_path
                if role in ("fit", "selection")
                else pre_confirmation_path
            ),
            "checkpoint_state": (
                "PRE_OUTCOME_SEAL"
                if role in ("fit", "selection")
                else "PRE_CONFIRMATION_PACKAGE_SEAL"
            ),
        }
        for role in ROLE_ORDER
    }
    if not (
        genesis.get("schema_version") == 2
        and genesis.get("record_type") == "replacement_registry_genesis"
        and genesis.get("attempt") == ATTEMPT
        and genesis.get("science_attempt") == SCIENCE_ATTEMPT
        and genesis.get("scope") == "append_only_all_roles_and_all_dgps"
        and type(genesis.get("created_unix_ns")) is int
        and int(genesis["created_unix_ns"]) > 0
        and genesis.get("dgp_matrix_path")
        == _canonical_relative(attempt_root / "DGP_MATRIX.json")
        and genesis.get("dgp_matrix_sha256")
        == sha256_file(attempt_root / "DGP_MATRIX.json")
        and genesis.get("cohort_seed_ledger_path")
        == _canonical_relative(attempt_root / "cohort_seed_ledger.json")
        and genesis.get("cohort_seed_ledger_sha256")
        == sha256_file(attempt_root / "cohort_seed_ledger.json")
        and genesis.get("regime_order") == list(DGP_ORDER)
        and genesis.get("role_order") == list(ROLE_ORDER)
        and genesis.get("seed_field_order") == list(SEED_FIELDS)
        and genesis.get("replacement_count_per_regime_per_role")
        == CONFIRMATION_REPLACEMENTS_PER_DGP
        and genesis.get("replacement_rule") == ledger.get("replacement_rule")
        and genesis.get("claims_directory") == _canonical_relative(claims_root)
        and genesis.get("claim_filename_format") == "{claim_index:06d}.json"
        and genesis.get("authorization_policy") == authorization_policy
        and genesis.get("claim_contract")
        == (
            "exclusive immutable canonical JSON segments; contiguous global indexes; "
            "SHA-256 predecessor chain; exact sealed-ledger tuple; exact role seal; "
            "authenticated mechanical-failure trigger; global source single-use"
        )
        and genesis_record_hash == _canonical_record_sha256(genesis_body)
    ):
        raise RuntimeError("replacement registry genesis contract drift")

    if claims_root.exists():
        root_info = claims_root.lstat()
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise RuntimeError("replacement claims root is linked or non-directory")
        entries = sorted(claims_root.iterdir(), key=lambda item: item.name)
    else:
        entries = []
    expected_names = [f"{index:06d}.json" for index in range(len(entries))]
    if [entry.name for entry in entries] != expected_names:
        raise RuntimeError("replacement claim segment prefix has a gap or extra entry")

    claim_keys = {
        "schema_version",
        "record_type",
        "attempt",
        "science_attempt",
        "claim_index",
        "claimed_unix_ns",
        "prev_sha256",
        "registry_genesis_path",
        "registry_genesis_file_sha256",
        "registry_genesis_record_sha256",
        "role",
        "regime",
        "slot",
        "episode_id",
        "replacement_slot",
        "replacement_episode_id",
        *SEED_FIELDS,
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "authorization_seal",
        "trigger_failure_log_path",
        "trigger_failure_seq",
        "trigger_failure_record_sha256",
        "failed_seed_source_episode_id",
        "record_sha256",
    }
    previous = str(genesis_record_hash)
    claims: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    source_seeds: set[tuple[int, int, int, int]] = set()
    trigger_hashes: set[str] = set()
    claim_files: list[dict[str, Any]] = []
    identities: dict[tuple[int, int], str] = {}
    genesis_file = _observed_file_link(genesis_path, identities=identities)
    ledger_regimes = ledger.get("regimes")
    if not isinstance(ledger_regimes, Mapping):
        raise RuntimeError("replacement registry cannot resolve ledger regimes")
    for index, claim_path in enumerate(entries):
        claim = _canonical_json_object(claim_path, f"replacement claim {index}")
        _exact_keys(claim, claim_keys, f"replacement claim {index}")
        body = dict(claim)
        observed_hash = body.pop("record_sha256")
        role = claim.get("role")
        regime = claim.get("regime")
        slot = claim.get("slot")
        replacement_slot = claim.get("replacement_slot")
        role_payload = (
            ledger_regimes.get(regime, {}).get("roles", {}).get(role, {})
            if isinstance(ledger_regimes.get(regime), Mapping)
            else {}
        )
        primary = role_payload.get("primary") if isinstance(role_payload, Mapping) else None
        replacements = (
            role_payload.get("replacements") if isinstance(role_payload, Mapping) else None
        )
        destination = (
            primary[int(slot)]
            if isinstance(primary, list)
            and type(slot) is int
            and 0 <= int(slot) < len(primary)
            else None
        )
        replacement = (
            replacements[int(replacement_slot)]
            if isinstance(replacements, list)
            and type(replacement_slot) is int
            and 0 <= int(replacement_slot) < len(replacements)
            else None
        )
        authorization_path = (
            inheritance_path
            if role in ("fit", "selection")
            else pre_confirmation_path
        )
        authorization_state = (
            "PRE_OUTCOME_SEAL"
            if role in ("fit", "selection")
            else "PRE_CONFIRMATION_PACKAGE_SEAL"
        )
        expected_authorization = _exact_seal_link(
            authorization_path, authorization_state
        )
        checks = {
            "schema": claim.get("schema_version") == 2,
            "record_type": claim.get("record_type") == "replacement_claim",
            "attempt": claim.get("attempt") == ATTEMPT,
            "science_attempt": claim.get("science_attempt") == SCIENCE_ATTEMPT,
            "index": claim.get("claim_index") == index
            and type(claim.get("claim_index")) is int,
            "timestamp": type(claim.get("claimed_unix_ns")) is int
            and int(claim["claimed_unix_ns"]) > 0,
            "previous": claim.get("prev_sha256") == previous,
            "genesis_path": claim.get("registry_genesis_path")
            == genesis_file["path"],
            "genesis_file_hash": claim.get("registry_genesis_file_sha256")
            == genesis_file["sha256"],
            "genesis_record_hash": claim.get("registry_genesis_record_sha256")
            == genesis_record_hash,
            "role": role in ROLE_ORDER,
            "regime": regime in DGP_ORDER,
            "destination": destination is not None
            and claim.get("episode_id") == destination.get("episode_id"),
            "replacement": replacement is not None
            and claim.get("replacement_episode_id")
            == replacement.get("episode_id"),
            "seeds": replacement is not None
            and _seed_tuple(claim) == _seed_tuple(replacement),
            "ledger_path": claim.get("cohort_seed_ledger_path")
            == genesis.get("cohort_seed_ledger_path"),
            "ledger_hash": claim.get("cohort_seed_ledger_sha256")
            == genesis.get("cohort_seed_ledger_sha256"),
            "authorization": claim.get("authorization_seal")
            == expected_authorization,
            "failure_path": role in ROLE_ORDER
            and regime in DGP_ORDER
            and claim.get("trigger_failure_log_path")
            == _canonical_relative(
                attempt_root / "data" / str(role) / str(regime) / "rollout_failures.jsonl"
            ),
            "failure_seq": type(claim.get("trigger_failure_seq")) is int
            and int(claim["trigger_failure_seq"]) > 0,
            "failure_hash": isinstance(
                claim.get("trigger_failure_record_sha256"), str
            )
            and len(str(claim["trigger_failure_record_sha256"])) == 64,
            "failed_source": isinstance(
                claim.get("failed_seed_source_episode_id"), str
            )
            and bool(claim.get("failed_seed_source_episode_id")),
            "record_hash": observed_hash == _canonical_record_sha256(body),
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"replacement claim ledger/authentication drift at {index}: {checks}"
            )
        source_id = str(claim["replacement_episode_id"])
        seeds = _seed_tuple(claim)
        trigger = str(claim["trigger_failure_record_sha256"])
        if source_id in source_ids or seeds in source_seeds or trigger in trigger_hashes:
            raise RuntimeError("replacement claim reuses source, seeds, or failure trigger")
        source_ids.add(source_id)
        source_seeds.add(seeds)
        trigger_hashes.add(trigger)
        previous = str(observed_hash)
        claims.append(claim)
        claim_file = _observed_file_link(claim_path, identities=identities)
        claim_files.append(
            {
                "path": claim_file["path"],
                "sha256": claim_file["sha256"],
                "record_sha256": observed_hash,
            }
        )

    prefix = pre_confirmation.get("replacement_registry_prefix")
    expected_prefix_keys = {
        "schema_version",
        "storage",
        "genesis",
        "claims_directory",
        "claim_count",
        "head_record_sha256",
        "claim_files",
        "permitted_existing_roles",
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "dgp_matrix_path",
        "dgp_matrix_sha256",
    }
    if not isinstance(prefix, Mapping):
        raise RuntimeError("pre-confirmation seal lacks replacement registry prefix")
    _exact_keys(prefix, expected_prefix_keys, "pre-confirmation registry prefix")
    prefix_count = prefix.get("claim_count")
    if not (
        prefix.get("schema_version") == 1
        and prefix.get("storage")
        == "immutable_genesis_and_exclusive_hash_chained_claim_segments"
        and prefix.get("genesis")
        == {
            "path": genesis_file["path"],
            "sha256": genesis_file["sha256"],
            "record_sha256": genesis_record_hash,
        }
        and prefix.get("claims_directory") == _canonical_relative(claims_root)
        and type(prefix_count) is int
        and 0 <= int(prefix_count) <= len(claims)
        and prefix.get("claim_files") == claim_files[: int(prefix_count)]
        and prefix.get("head_record_sha256")
        == (
            genesis_record_hash
            if int(prefix_count) == 0
            else claims[int(prefix_count) - 1]["record_sha256"]
        )
        and prefix.get("permitted_existing_roles") == ["fit", "selection"]
        and prefix.get("cohort_seed_ledger_path")
        == genesis.get("cohort_seed_ledger_path")
        and prefix.get("cohort_seed_ledger_sha256")
        == genesis.get("cohort_seed_ledger_sha256")
        and prefix.get("dgp_matrix_path") == genesis.get("dgp_matrix_path")
        and prefix.get("dgp_matrix_sha256") == genesis.get("dgp_matrix_sha256")
        and pre_confirmation.get("authorized_later_replacement_roles")
        == ["smoke", "confirmation"]
        and pre_confirmation.get("postseal_replacement_contract")
        == (
            "only new exclusive contiguous hash-chained claim segments bound to "
            "this PRE_CONFIRMATION_PACKAGE_SEAL, the exact sealed ledger tuple, "
            "and an authenticated mechanical rollout failure"
        )
    ):
        raise RuntimeError("pre-confirmation replacement prefix contract drift")
    sealed_files = pre_confirmation.get("sealed_files")
    if not isinstance(sealed_files, Mapping):
        raise RuntimeError("pre-confirmation sealed-files map is malformed")
    for item in (prefix["genesis"], *prefix["claim_files"]):
        if sealed_files.get(item["path"]) != item["sha256"]:
            raise RuntimeError("pre-confirmation seal omits a registry prefix file")
    if any(claim["role"] not in ("fit", "selection") for claim in claims[: int(prefix_count)]):
        raise RuntimeError("pre-confirmation prefix contains a postseal role")
    if any(claim["role"] not in ("smoke", "confirmation") for claim in claims[int(prefix_count) :]):
        raise RuntimeError("post-preconfirmation registry extension has an unauthorized role")

    failure_chains: dict[tuple[str, str], list[dict[str, Any]]] = {}
    last_source: dict[tuple[str, str, str], tuple[str, str | None]] = {}
    for claim in claims:
        role = str(claim["role"])
        regime = str(claim["regime"])
        role_payload = ledger_regimes[regime]["roles"][role]
        authorization_path = (
            inheritance_path
            if role in ("fit", "selection")
            else pre_confirmation_path
        )
        authorization_state = (
            "PRE_OUTCOME_SEAL"
            if role in ("fit", "selection")
            else "PRE_CONFIRMATION_PACKAGE_SEAL"
        )
        key = (role, regime)
        if key not in failure_chains:
            failure_chains[key] = _failure_chain_v2(
                attempt_root / "data" / role / regime / "rollout_failures.jsonl",
                role=role,
                regime=regime,
                primary=role_payload["primary"],
                replacements=role_payload["replacements"],
                authorization=_exact_seal_link(
                    authorization_path, authorization_state
                ),
                registry_claims=claims,
            )
        matches = [
            item
            for item in failure_chains[key]
            if item["record_sha256"] == claim["trigger_failure_record_sha256"]
        ]
        destination_key = (role, regime, str(claim["episode_id"]))
        expected_failed_source, expected_failed_claim_hash = last_source.get(
            destination_key,
            (str(claim["episode_id"]), None),
        )
        if not (
            len(matches) == 1
            and matches[0]["seq"] == claim["trigger_failure_seq"]
            and matches[0]["episode_id"] == claim["episode_id"]
            and matches[0]["slot"] == claim["slot"]
            and matches[0]["attempted_seed_source_episode_id"]
            == claim["failed_seed_source_episode_id"]
            == expected_failed_source
            and matches[0]["replacement_claim_sha256"]
            == expected_failed_claim_hash
            and claim["claimed_unix_ns"] >= matches[0]["created_unix_ns"]
        ):
            raise RuntimeError("replacement claim is not justified by the next authenticated failure")
        last_source[destination_key] = (
            str(claim["replacement_episode_id"]),
            str(claim["record_sha256"]),
        )

    failure_files: list[dict[str, Any]] = []
    for role, regime in sorted(failure_chains):
        failure_path = (
            attempt_root / "data" / role / regime / "rollout_failures.jsonl"
        )
        failure_files.append(
            _observed_file_link(failure_path, identities=identities)
        )

    return {
        "genesis": genesis,
        "claims": claims,
        "head_record_sha256": previous,
        "prefix_count": int(prefix_count),
        "files": [genesis_file, *[
            _strict_file_link(
                item["path"],
                item["sha256"],
                expected_path=REPO_ROOT / item["path"],
                identities={},
            )
            for item in claim_files
        ]],
        "claim_files": claim_files,
        "failure_chains": failure_chains,
        "failure_files": failure_files,
    }


def _confirmation_contract(
    attempt_root: Path, *, regime: str, expected_episodes: int
) -> dict[str, Any]:
    if expected_episodes not in range(500, CONFIRMATION_MAX_EPISODES_PER_DGP + 1, 500):
        raise RuntimeError("confirmation count is not one frozen 500-episode grid value")
    if attempt_root.name != ATTEMPT or regime not in DGP_ORDER:
        raise RuntimeError("active confirmation attempt/DGP identity drift")

    dgp_path = attempt_root / "DGP_MATRIX.json"
    ledger_path = attempt_root / "cohort_seed_ledger.json"
    seal_path = attempt_root / "audit/pre_confirmation_package_seal.json"
    registry_path = attempt_root / "data/replacement_registry.json"
    for required in (dgp_path, ledger_path, seal_path, registry_path):
        if not required.is_file() or required.is_symlink():
            raise RuntimeError(f"required confirmation authority is absent or linked: {required}")

    dgp = read_json(dgp_path)
    if not (
        dgp.get("schema_version") == 1
        and not isinstance(dgp.get("schema_version"), bool)
        and dgp.get("attempt") == SCIENCE_ATTEMPT
        and tuple(dgp.get("regime_order", ())) == DGP_ORDER
        and isinstance(dgp.get("regimes"), Mapping)
        and tuple(dgp["regimes"]) == DGP_ORDER
    ):
        raise RuntimeError("sealed DGP matrix identity/order drift")

    ledger = read_json(ledger_path)
    if not (
        ledger.get("schema_version") == 1
        and not isinstance(ledger.get("schema_version"), bool)
        and ledger.get("attempt") == SCIENCE_ATTEMPT
        and tuple(ledger.get("regime_order", ())) == DGP_ORDER
        and tuple(ledger.get("role_order", ())) == ROLE_ORDER
        and ledger.get("role_primary_counts_per_regime") == ROLE_PRIMARY_COUNTS
        and ledger.get("replacement_count_per_regime_per_role")
        == CONFIRMATION_REPLACEMENTS_PER_DGP
        and ledger.get("confirmation_max_slots_per_regime")
        == CONFIRMATION_MAX_EPISODES_PER_DGP
        and ledger.get("confirmation_prefix_rule")
        == (
            "the fixed post-selection N uses exactly primary confirmation slots "
            "0..N-1 separately in every regime; later slots are never substituted"
        )
    ):
        raise RuntimeError("cohort seed ledger header/count contract drift")
    ledger_checks = ledger.get("checks")
    if not isinstance(ledger_checks, Mapping) or not ledger_checks or any(
        value is not True for value in ledger_checks.values()
    ):
        raise RuntimeError("cohort seed ledger freshness/uniqueness checks do not pass")
    regimes = ledger.get("regimes")
    if not isinstance(regimes, Mapping) or tuple(regimes) != DGP_ORDER:
        raise RuntimeError("cohort seed ledger DGP mapping drift")
    regime_payload = regimes.get(regime)
    roles = regime_payload.get("roles") if isinstance(regime_payload, Mapping) else None
    confirmation = roles.get("confirmation") if isinstance(roles, Mapping) else None
    if (
        not isinstance(roles, Mapping)
        or set(roles) != set(ROLE_ORDER)
        or not isinstance(confirmation, Mapping)
        or set(confirmation) != {"primary", "replacements"}
        or not isinstance(confirmation.get("primary"), list)
        or not isinstance(confirmation.get("replacements"), list)
        or len(confirmation["primary"]) != CONFIRMATION_MAX_EPISODES_PER_DGP
        or len(confirmation["replacements"]) != CONFIRMATION_REPLACEMENTS_PER_DGP
    ):
        raise RuntimeError("confirmation seed-ledger role/pool schema drift")
    primary_all = [
        _validate_seed_record(item, regime=regime, pool="primary", index=index)
        for index, item in enumerate(confirmation["primary"])
    ]
    replacements = [
        _validate_seed_record(item, regime=regime, pool="replacements", index=index)
        for index, item in enumerate(confirmation["replacements"])
    ]
    identifiers = [item["episode_id"] for item in (*primary_all, *replacements)]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("confirmation ledger episode IDs are not unique")
    for field in SEED_FIELDS:
        values = [item[field] for item in (*primary_all, *replacements)]
        if len(values) != len(set(values)):
            raise RuntimeError(f"confirmation ledger {field} identifiers are not unique")

    seal = read_json(seal_path)
    counts = seal.get("outcome_counts_at_seal")
    confirmation_counts = seal.get("confirmation_episode_count_per_regime")
    expected_counts = {
        "fit_outcome_episodes": len(DGP_ORDER) * ROLE_PRIMARY_COUNTS["fit"],
        "selection_outcome_episodes": len(DGP_ORDER) * ROLE_PRIMARY_COUNTS["selection"],
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    if not (
        seal.get("schema_version") == 1
        and not isinstance(seal.get("schema_version"), bool)
        and seal.get("attempt") == ATTEMPT
        and seal.get("checkpoint_state") == "PRE_CONFIRMATION_PACKAGE_SEAL"
        and seal.get("passed") is True
        and counts == expected_counts
        and isinstance(confirmation_counts, Mapping)
        and tuple(confirmation_counts) == DGP_ORDER
        and all(
            isinstance(confirmation_counts[name], int)
            and not isinstance(confirmation_counts[name], bool)
            and confirmation_counts[name] == expected_episodes
            for name in DGP_ORDER
        )
        and seal.get("smoke_permanently_excluded") is True
        and seal.get("sequential_confirmation_expansion_permitted") is False
    ):
        raise RuntimeError("pre-confirmation authorization/count contract drift")
    sealed_files = seal.get("sealed_files")
    if not isinstance(sealed_files, Mapping) or not sealed_files:
        raise RuntimeError("pre-confirmation authorization lacks sealed files")
    for source in (dgp_path, ledger_path, registry_path):
        relative = _canonical_relative(source)
        if sealed_files.get(relative) != sha256_file(source):
            raise RuntimeError(f"pre-confirmation seal does not bind {relative}")

    registry = _validate_replacement_registry_chain(
        attempt_root,
        ledger=ledger,
        dgp=dgp,
        pre_confirmation=seal,
    )

    return {
        "attempt_root": attempt_root,
        "dgp_path": dgp_path,
        "dgp": dgp,
        "ledger_path": ledger_path,
        "ledger": ledger,
        "seal_path": seal_path,
        "seal": seal,
        "authorization": _exact_seal_link(
            seal_path, "PRE_CONFIRMATION_PACKAGE_SEAL"
        ),
        "registry_path": registry_path,
        "registry": registry,
        "primary": primary_all[:expected_episodes],
        "replacements": replacements,
    }


def _validate_raw_array_metadata(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"action", "pixels"}:
        raise RuntimeError(f"{label} must describe exactly action and pixels")
    expected = {
        "action": ([201, 5], "float32"),
        "pixels": ([201, 224, 224, 3], "uint8"),
    }
    normalized: dict[str, Any] = {}
    for name, (shape, dtype) in expected.items():
        metadata = value[name]
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"{label}/{name} metadata is not an object")
        _exact_keys(metadata, {"shape", "dtype", "sha256"}, f"{label}/{name}")
        if metadata.get("shape") != shape or metadata.get("dtype") != dtype:
            raise RuntimeError(f"{label}/{name} shape or dtype drift")
        _strict_sha256(metadata.get("sha256"), f"{label}/{name} array hash")
        normalized[name] = dict(metadata)
    return normalized


def _validate_generation_audit_contract(
    value: Any,
    *,
    arrays: Mapping[str, Any],
    initial_pixels_sha256: Any,
    source: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    keys = {
        "rows",
        "rollout_steps_expected",
        "rollout_steps_completed",
        "pixel_frames_captured",
        "initial_pixels_sha256",
        "input_validation",
        "reset_metadata",
        "action_space_seed",
        "action_spaces_seeded",
        "rng_activation_order",
        "only_pixels_and_actions_captured",
        "retention_uses_only_input_contract_and_local_step_bookkeeping",
    }
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not an object")
    _exact_keys(value, keys, label)
    _strict_sha256(initial_pixels_sha256, f"{label} initial pixels hash")
    expected_counts = {
        "rows": 201,
        "rollout_steps_expected": 200,
        "rollout_steps_completed": 200,
        "pixel_frames_captured": 201,
        "action_space_seed": _strict_int(
            source.get("action_space_seed"), f"{label} source action-space seed"
        ),
    }
    if any(
        type(value.get(key)) is not int or value.get(key) != expected
        for key, expected in expected_counts.items()
    ):
        raise RuntimeError(f"{label} exact generation counts/seeds drift")
    if not (
        value.get("initial_pixels_sha256") == initial_pixels_sha256
        and value.get("rng_activation_order")
        == (
            "policy.set_seed; deterministic V5 seed-forwarded reset; "
            "action-space seed; numpy oracle seed; first policy action"
        )
        and value.get("only_pixels_and_actions_captured") is True
        and value.get(
            "retention_uses_only_input_contract_and_local_step_bookkeeping"
        )
        is True
    ):
        raise RuntimeError(f"{label} causal generation contract drift")
    seeded = value.get("action_spaces_seeded")
    allowed_seeded = ["vector_action_space", "unwrapped_action_space"]
    if not (
        isinstance(seeded, list)
        and seeded
        and len(seeded) == len(set(seeded))
        and all(name in allowed_seeded for name in seeded)
        and seeded == [name for name in allowed_seeded if name in seeded]
    ):
        raise RuntimeError(f"{label} action-space seed list drift")

    validation = value.get("input_validation")
    validation_keys = {
        "loaded_keys",
        "pixels_shape",
        "pixels_dtype",
        "action_shape",
        "action_dtype",
        "pixels_finite",
        "modeled_actions_finite",
        "terminal_action_nan_sentinel",
        "array_sha256",
    }
    if not isinstance(validation, Mapping):
        raise RuntimeError(f"{label} input validation is not an object")
    _exact_keys(validation, validation_keys, f"{label} input validation")
    array_hashes = validation.get("array_sha256")
    if not isinstance(array_hashes, Mapping):
        raise RuntimeError(f"{label} input-validation hashes are not an object")
    _exact_keys(array_hashes, {"action", "pixels"}, f"{label} input hashes")
    if not (
        validation.get("loaded_keys") == ["action", "pixels"]
        and validation.get("pixels_shape") == [201, 224, 224, 3]
        and validation.get("pixels_dtype") == "uint8"
        and validation.get("action_shape") == [201, 5]
        and validation.get("action_dtype") == "float32"
        and validation.get("pixels_finite") is True
        and validation.get("modeled_actions_finite") is True
        and validation.get("terminal_action_nan_sentinel") is True
        and array_hashes
        == {
            "action": arrays["action"]["sha256"],
            "pixels": arrays["pixels"]["sha256"],
        }
    ):
        raise RuntimeError(f"{label} exact input-validation contract drift")

    reset = value.get("reset_metadata")
    reset_keys = {
        "environment_seed",
        "variation_seed",
        "physical_state_seed",
        "variation_values_sha256",
        "reset_reference_path",
        "reset_reference_sha256",
        "reset_values_persisted",
    }
    if not isinstance(reset, Mapping):
        raise RuntimeError(f"{label} reset metadata is not an object")
    _exact_keys(reset, reset_keys, f"{label} reset metadata")
    env_seed = _strict_int(source.get("env_seed"), f"{label} source env seed")
    for key in ("environment_seed", "variation_seed", "physical_state_seed"):
        if type(reset.get(key)) is not int or reset.get(key) != env_seed:
            raise RuntimeError(f"{label} reset seed drift")
    _strict_sha256(reset.get("variation_values_sha256"), f"{label} variation hash")
    reset_reference = (
        REPO_ROOT
        / "runs/lewm_adaptive_compute_distribution_contract/generator_seedfix.py"
    )
    if not (
        reset.get("reset_reference_path") == _canonical_relative(reset_reference)
        and reset.get("reset_reference_sha256") == sha256_file(reset_reference)
        and reset.get("reset_values_persisted") is False
    ):
        raise RuntimeError(f"{label} reset-reference contract drift")
    return dict(value)


def _validate_closed_persistence_intent_contract(
    intent: Any,
    *,
    record: Mapping[str, Any],
    destination: Mapping[str, Any],
    source: Mapping[str, Any],
    source_pool: str,
    source_slot: int,
    arrays: Mapping[str, Any],
    raw_path: Path,
    sidecar_path: Path,
    intent_path: Path,
    dgp_path: Path,
    ledger_path: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(intent, Mapping):
        raise RuntimeError(f"{label} is not an object")
    _exact_keys(intent, PERSISTENCE_INTENT_KEYS, label)
    shared = {
        "role",
        "regime",
        "slot",
        "episode_id",
        "seed_source_episode_id",
        "replacement_used",
        "replacement_claim_index",
        "replacement_claim_sha256",
        *SEED_FIELDS,
        "arrays",
        "initial_pixels_sha256",
        "generation_audit",
        "authorization_seal",
    }
    checks = {
        "schema": type(intent.get("schema_version")) is int
        and intent.get("schema_version") == 2,
        "attempt": intent.get("attempt") == ATTEMPT,
        "science_attempt": intent.get("science_attempt") == SCIENCE_ATTEMPT,
        "timestamp": type(intent.get("created_unix_ns")) is int
        and int(intent["created_unix_ns"]) >= 1,
        "status": intent.get("status") == PERSISTENCE_INTENT_STATUS,
        "shared": all(intent.get(key) == record.get(key) for key in shared),
        "orphan_rule": intent.get("orphan_adoption_rule") == ORPHAN_ADOPTION_RULE,
        "arrays": intent.get("arrays") == arrays,
        "raw_path": intent.get("raw_path") == _canonical_relative(raw_path),
        "sidecar_path": intent.get("raw_sidecar_path")
        == _canonical_relative(sidecar_path),
        "intent_path": intent.get("persistence_intent_path")
        == _canonical_relative(intent_path),
        "dgp_path": intent.get("dgp_matrix_path") == _canonical_relative(dgp_path),
        "dgp_sha256": intent.get("dgp_matrix_sha256") == sha256_file(dgp_path),
        "ledger_path": intent.get("cohort_seed_ledger_path")
        == _canonical_relative(ledger_path),
        "ledger_sha256": intent.get("cohort_seed_ledger_sha256")
        == sha256_file(ledger_path),
        "destination": intent.get("destination_seed_record") == destination,
        "source_pool": intent.get("seed_source_pool") == source_pool,
        "source_slot": type(intent.get("seed_source_slot")) is int
        and intent.get("seed_source_slot") == source_slot,
        "source": intent.get("seed_source_record") == source,
    }
    if not all(checks.values()):
        raise RuntimeError(f"{label} closed-intent contract drift: {checks}")
    return dict(intent)


def _seed_tuple(value: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(_strict_int(value.get(name), name) for name in SEED_FIELDS)  # type: ignore[return-value]


def _validate_rollout_failure_log(
    path: Path,
    *,
    regime: str,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    final_records: Mapping[str, Mapping[str, Any]],
    authorization: Mapping[str, Any],
    registry_claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = _failure_chain_v2(
        path,
        role="confirmation",
        regime=regime,
        primary=primary,
        replacements=replacements,
        authorization=authorization,
        registry_claims=registry_claims,
    )
    if not normalized:
        raise RuntimeError("empty confirmation rollout failure log is forbidden")
    failures_by_destination: dict[str, list[dict[str, Any]]] = {}
    for value in normalized:
        episode_id = str(value["episode_id"])
        final = final_records.get(episode_id)
        if final is None:
            raise RuntimeError(
                "failure chain references a destination outside the retained cohort"
            )
        failures_by_destination.setdefault(episode_id, []).append(value)
    claims_by_destination: dict[str, list[Mapping[str, Any]]] = {}
    for claim in registry_claims:
        if claim.get("role") != "confirmation" or claim.get("regime") != regime:
            continue
        episode_id = str(claim.get("episode_id"))
        if episode_id not in final_records:
            raise RuntimeError("confirmation claim targets an unretained destination")
        claims_by_destination.setdefault(episode_id, []).append(claim)
    for claims in claims_by_destination.values():
        claims.sort(key=lambda item: int(item["claim_index"]))

    for episode_id, final in final_records.items():
        if type(final.get("replacement_used")) is not bool:
            raise RuntimeError("retained replacement_used is not exact boolean")
        failures = failures_by_destination.get(episode_id, [])
        claims = claims_by_destination.get(episode_id, [])
        retained_source = str(final.get("seed_source_episode_id"))
        failed_sources = [
            str(item["attempted_seed_source_episode_id"]) for item in failures
        ]
        if retained_source in failed_sources:
            raise RuntimeError(
                "retained source has an authenticated mechanical failure"
            )
        if not failures:
            if claims or final.get("replacement_used") is not False:
                raise RuntimeError(
                    "failure-free destination has a claim or retained replacement"
                )
            continue
        if final.get("replacement_used") is not True:
            raise RuntimeError("failed destination did not retain a replacement")
        if len(claims) != len(failures):
            raise RuntimeError(
                "each authenticated failure must have exactly one next claim"
            )
        expected_failed_source = episode_id
        for failure, claim in zip(failures, claims, strict=True):
            if not (
                failure.get("attempted_seed_source_episode_id")
                == expected_failed_source
                and claim.get("trigger_failure_record_sha256")
                == failure.get("record_sha256")
                and claim.get("trigger_failure_seq") == failure.get("seq")
                and claim.get("failed_seed_source_episode_id")
                == expected_failed_source
            ):
                raise RuntimeError(
                    "failure/replacement claim progression is not exact"
                )
            expected_failed_source = str(claim.get("replacement_episode_id"))
        terminal_claim = claims[-1]
        claim_index = final.get("replacement_claim_index")
        if not (
            type(claim_index) is int
            and int(claim_index) == terminal_claim.get("claim_index")
            and retained_source == expected_failed_source
            and final.get("replacement_claim_sha256")
            == terminal_claim.get("record_sha256")
            and _seed_tuple(final) == _seed_tuple(terminal_claim)
        ):
            raise RuntimeError("final replacement is not the failure-free terminal claim")
    return normalized


def _verify_confirmation_raw_manifest(
    path: Path, *, regime: str, expected_episodes: int
) -> dict[str, Any]:
    attempt_root = _attempt_root_from_manifest(
        path,
        role="confirmation",
        regime=regime,
        filename="raw_manifest.json",
    )
    contract = _confirmation_contract(
        attempt_root, regime=regime, expected_episodes=expected_episodes
    )
    manifest = read_json(path)
    manifest_keys = {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "complete",
        "role",
        "regime",
        "regime_specification",
        "episode_count",
        "episodes",
        "replacements_used",
        "raw_archive_members",
        "raw_pixels_contract",
        "raw_action_contract",
        "role_isolation",
        "smoke_permanently_excluded",
        "retention_contract",
        "dgp_matrix_path",
        "dgp_matrix_sha256",
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "authorization_seal",
        "replacement_registry_path",
        "replacement_registry_scope",
        "orphan_policy",
    }
    _exact_keys(manifest, manifest_keys, "confirmation raw manifest")
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise RuntimeError("confirmation raw manifest episodes is not a list")
    if not (
        manifest.get("schema_version") == 1
        and not isinstance(manifest.get("schema_version"), bool)
        and manifest.get("attempt") == ATTEMPT
        and _strict_int(manifest.get("created_unix_ns"), "raw manifest timestamp", minimum=1)
        and manifest.get("complete") is True
        and manifest.get("role") == "confirmation"
        and manifest.get("regime") == regime
        and manifest.get("regime_specification")
        == contract["dgp"]["regimes"][regime]
        and manifest.get("episode_count") == expected_episodes
        and not isinstance(manifest.get("episode_count"), bool)
        and len(records) == expected_episodes
        and manifest.get("raw_archive_members") == ["action", "pixels"]
        and manifest.get("raw_pixels_contract") == "uint8 [201,224,224,3]"
        and manifest.get("raw_action_contract")
        == "float32 [201,5]; 200 finite plus terminal NaN"
        and manifest.get("role_isolation") is True
        and manifest.get("smoke_permanently_excluded") is False
        and manifest.get("retention_contract")
        == "pixels_action_shapes_finiteness_and_local_step_count_only"
        and manifest.get("dgp_matrix_path")
        == _canonical_relative(contract["dgp_path"])
        and manifest.get("dgp_matrix_sha256")
        == sha256_file(contract["dgp_path"])
        and manifest.get("cohort_seed_ledger_path")
        == _canonical_relative(contract["ledger_path"])
        and manifest.get("cohort_seed_ledger_sha256")
        == sha256_file(contract["ledger_path"])
        and manifest.get("authorization_seal") == contract["authorization"]
        and manifest.get("replacement_registry_path")
        == _canonical_relative(contract["registry_path"])
        and manifest.get("replacement_registry_scope")
        == "append_only_all_roles_and_all_dgps"
        and manifest.get("orphan_policy")
        == (
            "adopt raw-only archive only after exact closed intent, current "
            "authorization, prospective ledger, canonical nonlink paths, and full "
            "two-array recomputation; otherwise stop"
        )
    ):
        raise RuntimeError("confirmation raw manifest header/source contract drift")

    identities: dict[tuple[int, int], str] = {}
    manifest_file = _observed_file_link(path, identities=identities)
    raw_files: list[dict[str, Any]] = []
    sidecar_files: list[dict[str, Any]] = []
    intent_files: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    normalized_records: dict[str, Mapping[str, Any]] = {}
    used_replacement_ids: set[str] = set()
    record_keys = {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "complete",
        "role",
        "regime",
        "slot",
        "episode_id",
        "seed_source_episode_id",
        "replacement_used",
        "replacement_claim_index",
        "replacement_claim_sha256",
        *SEED_FIELDS,
        "raw_path",
        "raw_sha256",
        "raw_bytes",
        "arrays",
        "initial_pixels_sha256",
        "generation_audit",
        "input_loader_audit",
        "persistence_intent_path",
        "persistence_intent_sha256",
        "authorization_seal",
        "raw_archive_members",
        "orphan_adoption_safe",
    }
    replacement_by_id = {
        str(item["episode_id"]): item for item in contract["replacements"]
    }
    registry_claims = contract["registry"]["claims"]
    for index, (record, destination) in enumerate(
        zip(records, contract["primary"], strict=True)
    ):
        if not isinstance(record, Mapping):
            raise RuntimeError("confirmation raw episode record is not an object")
        _exact_keys(record, record_keys, f"confirmation raw record {index}")
        episode_id = str(destination["episode_id"])
        if not (
            record.get("schema_version") == 1
            and not isinstance(record.get("schema_version"), bool)
            and record.get("attempt") == ATTEMPT
            and _strict_int(record.get("created_unix_ns"), "raw sidecar timestamp", minimum=1)
            and record.get("complete") is True
            and record.get("role") == "confirmation"
            and record.get("regime") == regime
            and record.get("slot") == index == destination["slot"]
            and not isinstance(record.get("slot"), bool)
            and record.get("episode_id") == episode_id
            and record.get("authorization_seal") == contract["authorization"]
            and record.get("raw_archive_members") == ["action", "pixels"]
            and record.get("orphan_adoption_safe") is True
        ):
            raise RuntimeError(f"confirmation raw record identity drift at slot {index}")
        arrays = _validate_raw_array_metadata(
            record.get("arrays"), f"confirmation raw record {index}"
        )
        _strict_sha256(
            record.get("initial_pixels_sha256"), "initial pixels SHA-256"
        )
        if not isinstance(record.get("input_loader_audit"), Mapping):
            raise RuntimeError("confirmation raw record audit schema drift")

        replacement_used = record.get("replacement_used")
        source_record: Mapping[str, Any] = destination
        source_pool = "primary"
        source_slot = index
        if replacement_used is False:
            if not (
                record.get("seed_source_episode_id") == episode_id
                and record.get("replacement_claim_index") is None
                and record.get("replacement_claim_sha256") is None
                and _seed_tuple(record) == _seed_tuple(destination)
            ):
                raise RuntimeError("primary confirmation seed assignment drift")
        elif replacement_used is True:
            source_id = str(record.get("seed_source_episode_id"))
            source = replacement_by_id.get(source_id)
            claim_index = record.get("replacement_claim_index")
            if (
                source is None
                or not isinstance(claim_index, int)
                or isinstance(claim_index, bool)
                or claim_index < 0
                or claim_index >= len(registry_claims)
                or source_id in used_replacement_ids
                or _seed_tuple(record) != _seed_tuple(source)
            ):
                raise RuntimeError("replacement confirmation seed assignment drift")
            source_record = source
            source_pool = "replacements"
            source_slot = int(source["slot"])
            claim = registry_claims[claim_index]
            if not isinstance(claim, Mapping) or not (
                claim.get("role") == "confirmation"
                and claim.get("regime") == regime
                and claim.get("slot") == index
                and claim.get("episode_id") == episode_id
                and claim.get("replacement_episode_id") == source_id
                and _seed_tuple(claim) == _seed_tuple(source)
                and record.get("replacement_claim_sha256")
                == claim.get("record_sha256")
            ):
                raise RuntimeError("replacement registry/sidecar binding drift")
            used_replacement_ids.add(source_id)
        else:
            raise RuntimeError("replacement_used must be an exact boolean")
        _validate_generation_audit_contract(
            record.get("generation_audit"),
            arrays=arrays,
            initial_pixels_sha256=record.get("initial_pixels_sha256"),
            source=source_record,
            label=f"confirmation raw record {index} generation audit",
        )

        raw_path = attempt_root / "data/confirmation" / regime / "raw" / f"{episode_id}.npz"
        sidecar_path = raw_path.with_suffix(".json")
        intent_path = (
            attempt_root
            / "data/persistence_intents/confirmation"
            / regime
            / f"{episode_id}.json"
        )
        raw_file = _strict_file_link(
            record.get("raw_path"),
            record.get("raw_sha256"),
            expected_path=raw_path,
            identities=identities,
        )
        if record.get("raw_bytes") != raw_file["bytes"] or isinstance(
            record.get("raw_bytes"), bool
        ):
            raise RuntimeError("confirmation raw byte-count drift")
        sidecar = read_json(sidecar_path)
        if sidecar != dict(record):
            raise RuntimeError("raw manifest record/raw sidecar equality drift")
        sidecar_file = _observed_file_link(sidecar_path, identities=identities)
        if record.get("persistence_intent_path") != _canonical_relative(intent_path):
            raise RuntimeError("confirmation persistence-intent path drift")
        intent_file = _strict_file_link(
            record.get("persistence_intent_path"),
            record.get("persistence_intent_sha256"),
            expected_path=intent_path,
            identities=identities,
        )
        intent = read_json(intent_path)
        _validate_closed_persistence_intent_contract(
            intent,
            record=record,
            destination=destination,
            source=source_record,
            source_pool=source_pool,
            source_slot=source_slot,
            arrays=arrays,
            raw_path=raw_path,
            sidecar_path=sidecar_path,
            intent_path=intent_path,
            dgp_path=contract["dgp_path"],
            ledger_path=contract["ledger_path"],
            label=f"confirmation intent {index}",
        )
        raw_files.append(raw_file)
        sidecar_files.append(sidecar_file)
        intent_files.append(intent_file)
        episode_ids.append(episode_id)
        normalized_records[episode_id] = record

    replacement_count = sum(
        record.get("replacement_used") is True for record in records
    )
    if (
        manifest.get("replacements_used") != replacement_count
        or isinstance(manifest.get("replacements_used"), bool)
        or replacement_count > CONFIRMATION_REPLACEMENTS_PER_DGP
    ):
        raise RuntimeError("confirmation replacement count drift")

    failure_path = attempt_root / "data/confirmation" / regime / "rollout_failures.jsonl"
    supporting_files: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    regime_claims = [
        claim
        for claim in registry_claims
        if claim.get("role") == "confirmation" and claim.get("regime") == regime
    ]
    if failure_path.exists():
        failure_records = _validate_rollout_failure_log(
            failure_path,
            regime=regime,
            primary=contract["primary"],
            replacements=contract["replacements"],
            final_records=normalized_records,
            authorization=contract["authorization"],
            registry_claims=registry_claims,
        )
        supporting_files.append(
            _observed_file_link(failure_path, identities=identities)
        )
    elif replacement_count or regime_claims:
        raise RuntimeError("replacement claim/retention lacks its failure log")

    confirmation_files = {
        item["path"]: item["sha256"]
        for item in (
            manifest_file,
            *raw_files,
            *sidecar_files,
            *intent_files,
            *supporting_files,
        )
    }
    checks = {
        "exact_v008_schema_attempt_role_dgp_count": True,
        "exact_dgp_and_cohort_ledger_bindings": True,
        "exact_preconfirmation_raw_authorization": True,
        "exact_ledger_ids_slots_order_and_seeds": True,
        "canonical_nonlink_raw_sidecar_intent_paths": True,
        "raw_manifest_sidecar_and_intent_equality": True,
        "replacement_registry_and_failure_chain_exact": True,
        "no_symlink_or_inode_aliases": True,
        "pixels_action_only": True,
        "role_isolation": True,
        "outcome_blind_retention": True,
        "forbidden_sources_excluded": _negative_assertions(manifest),
        "smoke_permanently_excluded_if_applicable": True,
        "replacement_count_bounded": replacement_count
        <= CONFIRMATION_REPLACEMENTS_PER_DGP,
    }
    return {
        "manifest": {
            "path": manifest_file["path"],
            "sha256": manifest_file["sha256"],
            "created_unix_ns": manifest["created_unix_ns"],
        },
        "role": "confirmation",
        "regime": regime,
        "episode_ids": episode_ids,
        "replacement_count": replacement_count,
        "files": raw_files,
        "sidecars": sidecar_files,
        "intents": intent_files,
        "supporting_files": supporting_files,
        "failure_records": failure_records,
        "used_replacement_claim_indexes": sorted(
            int(record["replacement_claim_index"])
            for record in records
            if record.get("replacement_used") is True
        ),
        "replacement_registry": contract["registry"],
        "confirmation_files": confirmation_files,
        "records": {key: dict(value) for key, value in normalized_records.items()},
        "authorization": contract["authorization"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_final_replacement_claim_references(
    attempt_root: Path,
    raw_by_regime: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove every post-seal claim ends as failed or uniquely retained.

    Intermediate claims are legitimate only when their source has its own
    authenticated failure and the next claim continues that exact chain.  The
    sole terminal claim for a failed destination must be the retained source
    and must have no failure record.  Zero failures require no log and primary
    retention.
    """

    if tuple(raw_by_regime) != DGP_ORDER:
        raise RuntimeError("replacement closure requires all four DGPs in order")
    baseline = raw_by_regime[DGP_ORDER[0]].get("replacement_registry")
    if not isinstance(baseline, Mapping):
        raise RuntimeError("confirmation raw evidence lacks replacement registry")
    for regime in DGP_ORDER[1:]:
        observed = raw_by_regime[regime].get("replacement_registry")
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != baseline.get(key)
            for key in (
                "genesis",
                "claims",
                "head_record_sha256",
                "prefix_count",
                "claim_files",
            )
        ):
            raise RuntimeError("replacement registry changed across DGP verification")

    claims = baseline.get("claims")
    prefix_count = baseline.get("prefix_count")
    if not isinstance(claims, list) or type(prefix_count) is not int:
        raise RuntimeError("replacement registry closure schema drift")
    dispositions: dict[int, tuple[str, str, str, str]] = {}
    support_files: dict[str, str] = {}
    support_identities: dict[tuple[int, int], str] = {}

    for item in (*baseline.get("files", ()), *baseline.get("failure_files", ())):
        if not isinstance(item, Mapping):
            raise RuntimeError("replacement registry support-file schema drift")
        relative = str(item.get("path"))
        digest = _strict_sha256(
            item.get("sha256"), f"replacement support hash {relative}"
        )
        if relative in support_files and support_files[relative] != digest:
            raise RuntimeError("replacement support-file hash collision")
        support_files[relative] = digest

    ledger = read_json(attempt_root / "cohort_seed_ledger.json")
    ledger_regimes = ledger.get("regimes")
    if not isinstance(ledger_regimes, Mapping):
        raise RuntimeError("replacement closure cannot resolve seed ledger")
    authorization = _exact_seal_link(
        attempt_root / "audit/pre_confirmation_package_seal.json",
        "PRE_CONFIRMATION_PACKAGE_SEAL",
    )

    for role in ("smoke", "confirmation"):
        for regime in DGP_ORDER:
            if role == "confirmation":
                records = list(raw_by_regime[regime].get("records", {}).values())
            else:
                manifest_path = (
                    attempt_root / "data" / role / regime / "raw_manifest.json"
                )
                manifest = _canonical_json_object(
                    manifest_path, f"{role}/{regime} raw manifest"
                )
                if not (
                    manifest.get("attempt") == ATTEMPT
                    and manifest.get("role") == role
                    and manifest.get("regime") == regime
                    and manifest.get("complete") is True
                    and isinstance(manifest.get("episodes"), list)
                ):
                    raise RuntimeError("post-seal raw manifest identity drift")
                records = list(manifest["episodes"])
                manifest_file = _observed_file_link(
                    manifest_path, identities=support_identities
                )
                support_files[manifest_file["path"]] = manifest_file["sha256"]
            role_payload = ledger_regimes[regime]["roles"][role]
            primary = list(role_payload["primary"][: len(records)])
            replacements = list(role_payload["replacements"])
            failure_path = (
                attempt_root / "data" / role / regime / "rollout_failures.jsonl"
            )
            if failure_path.exists():
                failure_records = _failure_chain_v2(
                    failure_path,
                    role=role,
                    regime=regime,
                    primary=primary,
                    replacements=replacements,
                    authorization=authorization,
                    registry_claims=claims,
                )
                if not failure_records:
                    raise RuntimeError("present post-seal failure log is empty")
                failure_file = _observed_file_link(
                    failure_path, identities=support_identities
                )
                support_files[failure_file["path"]] = failure_file["sha256"]
            else:
                failure_records = []
            failures_by_destination: dict[str, list[Mapping[str, Any]]] = {}
            for failure in failure_records:
                failures_by_destination.setdefault(
                    str(failure["episode_id"]), []
                ).append(failure)
            claims_by_destination: dict[str, list[Mapping[str, Any]]] = {}
            for claim in claims[int(prefix_count) :]:
                if claim.get("role") == role and claim.get("regime") == regime:
                    claims_by_destination.setdefault(
                        str(claim["episode_id"]), []
                    ).append(claim)
            for destination_claims in claims_by_destination.values():
                destination_claims.sort(key=lambda item: int(item["claim_index"]))

            replacement_count = 0
            record_ids: set[str] = set()
            for record in records:
                if not isinstance(record, Mapping):
                    raise RuntimeError("post-seal raw record is not an object")
                episode_id = str(record.get("episode_id"))
                if episode_id in record_ids:
                    raise RuntimeError("post-seal retained destination is duplicated")
                record_ids.add(episode_id)
                replacement_used = record.get("replacement_used")
                claim_index = record.get("replacement_claim_index")
                claim_hash = record.get("replacement_claim_sha256")
                destination = next(
                    (
                        item
                        for item in primary
                        if item.get("episode_id") == episode_id
                        and item.get("slot") == record.get("slot")
                    ),
                    None,
                )
                if destination is None:
                    raise RuntimeError("post-seal retained destination is unsealed")
                destination_failures = failures_by_destination.get(episode_id, [])
                destination_claims = claims_by_destination.get(episode_id, [])
                retained_source = str(record.get("seed_source_episode_id"))
                failed_sources = [
                    str(item["attempted_seed_source_episode_id"])
                    for item in destination_failures
                ]
                if retained_source in failed_sources:
                    raise RuntimeError(
                        "retained post-seal source has an authenticated failure"
                    )
                if type(replacement_used) is not bool:
                    raise RuntimeError("post-seal replacement flag is not exact boolean")
                if not destination_failures:
                    if destination_claims:
                        raise RuntimeError("post-seal claim exists without a failure")
                    if not (
                        replacement_used is False
                        and claim_index is None
                        and claim_hash is None
                        and retained_source == episode_id
                        and _seed_tuple(record) == _seed_tuple(destination)
                    ):
                        raise RuntimeError(
                            "failure-free post-seal destination did not retain primary"
                        )
                else:
                    if replacement_used is not True:
                        raise RuntimeError(
                            "failed post-seal destination did not retain replacement"
                        )
                    replacement_count += 1
                    if len(destination_claims) != len(destination_failures):
                        raise RuntimeError(
                            "post-seal failure chain lacks an exact next claim"
                        )
                    expected_failed_source = episode_id
                    for failure, claim in zip(
                        destination_failures, destination_claims, strict=True
                    ):
                        index = int(claim["claim_index"])
                        if index in dispositions:
                            raise RuntimeError("post-seal claim is reused")
                        if not (
                            failure.get("attempted_seed_source_episode_id")
                            == expected_failed_source
                            and claim.get("failed_seed_source_episode_id")
                            == expected_failed_source
                            and claim.get("trigger_failure_record_sha256")
                            == failure.get("record_sha256")
                            and claim.get("trigger_failure_seq")
                            == failure.get("seq")
                        ):
                            raise RuntimeError(
                                "post-seal failure/claim progression drift"
                            )
                        expected_failed_source = str(
                            claim["replacement_episode_id"]
                        )
                        disposition = (
                            "failed"
                            if expected_failed_source in failed_sources
                            else "retained"
                        )
                        dispositions[index] = (
                            role,
                            regime,
                            episode_id,
                            disposition,
                        )
                    terminal_claim = destination_claims[-1]
                    if not (
                        type(claim_index) is int
                        and int(claim_index) == terminal_claim.get("claim_index")
                        and terminal_claim.get("replacement_episode_id")
                        == retained_source
                        and terminal_claim.get("record_sha256") == claim_hash
                        and _seed_tuple(terminal_claim) == _seed_tuple(record)
                        and dispositions[int(claim_index)][3] == "retained"
                    ):
                        raise RuntimeError(
                            "terminal post-seal claim is not the failure-free retained source"
                        )

                if role == "smoke":
                    sidecar_path = (
                        attempt_root
                        / "data"
                        / role
                        / regime
                        / "raw"
                        / f"{record.get('episode_id')}.json"
                    )
                    sidecar = _canonical_json_object(
                        sidecar_path, f"{role}/{regime} raw sidecar"
                    )
                    if sidecar != dict(record):
                        raise RuntimeError("smoke manifest/sidecar equality drift")
                    sidecar_file = _observed_file_link(
                        sidecar_path, identities=support_identities
                    )
                    support_files[sidecar_file["path"]] = sidecar_file["sha256"]
            if role == "smoke":
                if manifest.get("replacements_used") != replacement_count:
                    raise RuntimeError("smoke manifest replacement count drift")
            elif raw_by_regime[regime].get("replacement_count") != replacement_count:
                raise RuntimeError("confirmation replacement count drift in closure")
            if set(failures_by_destination) - record_ids:
                raise RuntimeError("post-seal failure targets an unretained destination")
            if set(claims_by_destination) - record_ids:
                raise RuntimeError("post-seal claim targets an unretained destination")

    expected = list(range(int(prefix_count), len(claims)))
    if sorted(dispositions) != expected:
        raise RuntimeError(
            "post-seal claims are not the exact failed-or-retained closure"
        )
    return {
        "genesis": baseline["genesis"],
        "claim_count": len(claims),
        "prefix_count": int(prefix_count),
        "head_record_sha256": baseline["head_record_sha256"],
        "postseal_claim_indexes": expected,
        "references": {
            str(index): list(dispositions[index]) for index in sorted(dispositions)
        },
        "support_files": support_files,
    }


def _exact_compute_from_histograms(
    histograms: Sequence[Sequence[int]], head_count: int
) -> dict[str, Any]:
    if head_count not in (2, 8):
        raise RuntimeError("compiled gate head count must be exactly 2 or 8")
    if not histograms:
        raise RuntimeError("cannot price an empty confirmation execution")
    rows = 0
    solver_calls = 0
    gate_evaluations = 0
    for histogram in histograms:
        if (
            not isinstance(histogram, (list, tuple))
            or len(histogram) != 4
            or any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in histogram
            )
            or sum(histogram) != ROWS_PER_EPISODE
        ):
            raise RuntimeError("confirmation call histogram contract drift")
        rows += sum(histogram)
        solver_calls += sum((depth + 1) * count for depth, count in enumerate(histogram))
        gate_evaluations += sum(min(depth + 1, 3) * count for depth, count in enumerate(histogram))
    later_adapter_calls = solver_calls - rows
    feature_per_gate = 3_801
    head_per_gate = 4_184 if head_count == 2 else 16_736
    nonflops_per_gate = 5 if head_count == 2 else 11
    base_flops = rows * 70_529_190
    depth1_flops = rows * 669_184
    adapter_flops = later_adapter_calls * 264_960
    feature_flops = gate_evaluations * feature_per_gate
    head_flops = gate_evaluations * head_per_gate
    gate_flops = feature_flops + head_flops
    numerator = solver_calls * 264_960 + gate_flops
    denominator = 264_960
    divisor = math.gcd(numerator, denominator)
    numerator //= divisor
    denominator //= divisor
    return {
        "row_count": rows,
        "base_calls": rows,
        "solver_calls": solver_calls,
        "mandatory_depth1_calls": rows,
        "later_adapter_calls": later_adapter_calls,
        "gate_evaluations": gate_evaluations,
        "head_count": head_count,
        "base_flops": base_flops,
        "depth1_flops": depth1_flops,
        "adapter_flops": adapter_flops,
        "gate_feature_flops": feature_flops,
        "gate_head_flops": head_flops,
        "gate_total_flops": gate_flops,
        "total_counted_flops": base_flops
        + depth1_flops
        + adapter_flops
        + gate_flops,
        "gate_nonflop_operations": gate_evaluations * nonflops_per_gate,
        "exact_equivalent_transition_independent_solver_calls_numerator": numerator,
        "exact_equivalent_transition_independent_solver_calls_denominator": denominator,
        "exact_equivalent_transition_independent_solver_calls": numerator / denominator,
        "historical_counting_convention": {
            "base_per_row": 70_529_190,
            "mandatory_depth1_per_row": 669_184,
            "each_later_adapter": 264_960,
            "feature_per_reached_gate": 3_801,
            "affine_head_per_reached_gate": 2_092,
            "activations_and_normalizations_inside_frozen_solver_omitted_as_in_V5": True,
        },
    }


def _validate_source_binding(
    value: Any,
    *,
    expected_path: Path,
    identities: dict[tuple[int, int], str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"source binding is not an object: {expected_path}")
    _exact_keys(value, {"path", "sha256"}, f"source binding {expected_path.name}")
    return _strict_file_link(
        value.get("path"),
        value.get("sha256"),
        expected_path=expected_path,
        identities=identities,
    )


def _validate_source_raw_manifest_binding(
    manifest: Mapping[str, Any], raw_manifest_path: Path
) -> str:
    """Require all three exact raw-manifest links; no optional legacy alias."""

    required = (
        "raw_manifest_path",
        "raw_manifest_sha256",
        "source_raw_manifest_sha256",
    )
    missing = [name for name in required if name not in manifest]
    if missing:
        raise RuntimeError(f"execution manifest lacks mandatory raw binding: {missing}")
    observed = sha256_file(raw_manifest_path)
    if not (
        manifest.get("raw_manifest_path") == _canonical_relative(raw_manifest_path)
        and manifest.get("raw_manifest_sha256") == observed
        and manifest.get("source_raw_manifest_sha256") == observed
    ):
        raise RuntimeError("execution source raw-manifest binding drift")
    return observed


def _verify_confirmation_execution_manifest(
    path: Path,
    *,
    regime: str,
    expected_episodes: int,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_root = _attempt_root_from_manifest(
        path,
        role="confirmation",
        regime=regime,
        filename="execution_manifest.json",
    )
    if raw.get("passed") is not True or raw.get("role") != "confirmation":
        raise RuntimeError("execution verifier received an unauthenticated raw cohort")
    if raw.get("regime") != regime or len(raw.get("episode_ids", ())) != expected_episodes:
        raise RuntimeError("execution/raw DGP or count binding drift")
    contract = _confirmation_contract(
        attempt_root, regime=regime, expected_episodes=expected_episodes
    )
    raw_manifest_path = attempt_root / "data/confirmation" / regime / "raw_manifest.json"
    if raw.get("manifest", {}).get("path") != _canonical_relative(raw_manifest_path):
        raise RuntimeError("execution verifier raw-manifest path binding drift")

    manifest = read_json(path)
    manifest_keys = {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "role",
        "regime",
        "complete",
        "episode_count",
        "row_count",
        "episodes",
        "raw_manifest_path",
        "raw_manifest_sha256",
        "source_raw_manifest_sha256",
        "compiled_gate_path",
        "compiled_gate_sha256",
        "architecture",
        "candidate_id",
        "head_names",
        "head_count",
        "gate_cost_per_reached_decision",
        "exact_compute",
        "authorization",
        "source_bindings",
        "actual_adaptive_output",
        "dense_role",
        "strict_continue_operator",
        "loaded_input_keys",
        "contact_or_privileged_materialized",
        "target_persisted",
        "target_loss_computed_during_execution",
        "causal_primitive_contract",
        "numerical_contract",
        "all_equivalence_checks_passed",
        "module_before",
        "module_after",
        "base_provenance",
        "frozen_facade",
        "no_gradients",
    }
    _exact_keys(manifest, manifest_keys, "confirmation execution manifest")
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise RuntimeError("confirmation execution manifest episodes is not a list")
    source_raw_hash = _validate_source_raw_manifest_binding(
        manifest, raw_manifest_path
    )
    head_count = _strict_int(manifest.get("head_count"), "compiled head count", minimum=1)
    if not (
        manifest.get("schema_version") == 1
        and not isinstance(manifest.get("schema_version"), bool)
        and manifest.get("attempt") == ATTEMPT
        and _strict_int(
            manifest.get("created_unix_ns"), "execution manifest timestamp", minimum=1
        )
        and manifest.get("role") == "confirmation"
        and manifest.get("regime") == regime
        and manifest.get("complete") is True
        and manifest.get("episode_count") == expected_episodes
        and not isinstance(manifest.get("episode_count"), bool)
        and manifest.get("row_count") == expected_episodes * ROWS_PER_EPISODE
        and not isinstance(manifest.get("row_count"), bool)
        and len(records) == expected_episodes
        and manifest.get("raw_manifest_path") == _canonical_relative(raw_manifest_path)
        and manifest.get("raw_manifest_sha256") == source_raw_hash
        and manifest.get("source_raw_manifest_sha256") == source_raw_hash
        and manifest.get("actual_adaptive_output")
        == "manual sequential sparse reached-row execution"
        and manifest.get("dense_role")
        == "numerical shadow and fixed-depth comparator only"
        and manifest.get("strict_continue_operator") == ">"
        and manifest.get("loaded_input_keys") == ["action", "pixels"]
        and manifest.get("contact_or_privileged_materialized") is False
        and manifest.get("target_persisted") is True
        and manifest.get("target_loss_computed_during_execution") is False
        and manifest.get("all_equivalence_checks_passed") is True
        and manifest.get("no_gradients") is True
        and isinstance(manifest.get("module_before"), Mapping)
        and manifest.get("module_before") == manifest.get("module_after")
        and manifest["module_after"].get("passed") is True
    ):
        raise RuntimeError("confirmation execution manifest header/state contract drift")

    authorization = manifest.get("authorization")
    if not isinstance(authorization, Mapping):
        raise RuntimeError("confirmation execution authorization is not an object")
    _exact_keys(
        authorization,
        {
            "state",
            "active_attempt",
            "state_sha256",
            "seal_path",
            "seal_sha256",
            "seal_checkpoint_state",
        },
        "confirmation execution authorization",
    )
    if not (
        authorization.get("state") == "CONFIRMATION_EXECUTION"
        and authorization.get("active_attempt") == ATTEMPT
        and _strict_sha256(
            authorization.get("state_sha256"), "execution authorization state hash"
        )
        and authorization.get("seal_path") == contract["authorization"]["path"]
        and authorization.get("seal_sha256") == contract["authorization"]["sha256"]
        and authorization.get("seal_checkpoint_state")
        == "PRE_CONFIRMATION_PACKAGE_SEAL"
    ):
        raise RuntimeError("confirmation execution authorization/state drift")

    identities: dict[tuple[int, int], str] = {}
    manifest_file = _observed_file_link(path, identities=identities)
    source_bindings = manifest.get("source_bindings")
    if not isinstance(source_bindings, Mapping):
        raise RuntimeError("confirmation execution source_bindings is not an object")
    expected_binding_paths = {
        "dgp_matrix": contract["dgp_path"],
        "cohort_seed_ledger": contract["ledger_path"],
        "execution_authorization_seal": contract["seal_path"],
        "counted_feature_source": attempt_root / "counted_features.py",
        "pricing_source": attempt_root / "flops.py",
        "runner_source": attempt_root / "runner.py",
        "replacement_registry": contract["registry_path"],
        "gate_fit": attempt_root / "freeze/gate_fit.npz",
        "compiled_gate": attempt_root / "freeze/compiled_gate.npz",
        "compiled_gate_manifest": attempt_root / "freeze/compiled_gate_manifest.json",
        "gate_freeze": attempt_root / "freeze/gate_freeze.json",
        "fixed_whitening": V5_FIXED_WHITENING,
    }
    _exact_keys(
        source_bindings,
        set(expected_binding_paths),
        "confirmation execution source_bindings",
    )
    verified_bindings = {
        name: _validate_source_binding(
            source_bindings[name],
            expected_path=expected_path,
            identities=identities,
        )
        for name, expected_path in expected_binding_paths.items()
    }
    if source_bindings["execution_authorization_seal"] != {
        "path": contract["authorization"]["path"],
        "sha256": contract["authorization"]["sha256"],
    }:
        raise RuntimeError("execution source binding does not equal authorization seal")

    compiled_path = expected_binding_paths["compiled_gate"]
    compiled_manifest_path = expected_binding_paths["compiled_gate_manifest"]
    gate_freeze_path = expected_binding_paths["gate_freeze"]
    gate_fit_path = expected_binding_paths["gate_fit"]
    compiled_manifest = read_json(compiled_manifest_path)
    gate_freeze = read_json(gate_freeze_path)
    compiled_hash = sha256_file(compiled_path)
    if not (
        manifest.get("compiled_gate_path") == _canonical_relative(compiled_path)
        and manifest.get("compiled_gate_sha256") == compiled_hash
        and Path(str(compiled_manifest.get("compiled_gate_path"))).resolve()
        == compiled_path.resolve()
        and Path(str(compiled_manifest.get("source_gate_fit_path"))).resolve()
        == gate_fit_path.resolve()
        and compiled_manifest.get("compiled_gate_sha256") == compiled_hash
        and compiled_manifest.get("source_gate_fit_sha256") == sha256_file(gate_fit_path)
        and compiled_manifest.get("passed") is True
        and compiled_manifest.get("target_or_contact_arrays_opened") is False
        and compiled_manifest.get("prior_outcome_arrays_opened") is False
        and gate_freeze.get("compiled_gate_sha256") == compiled_hash
        and gate_freeze.get("gate_fit_sha256") == sha256_file(gate_fit_path)
        and gate_freeze.get("compiled_gate_manifest_sha256")
        == sha256_file(compiled_manifest_path)
        and gate_freeze.get("selected_candidate_id") == manifest.get("candidate_id")
        and gate_freeze.get("architecture") == manifest.get("architecture")
        and gate_freeze.get("head_count") == head_count
        and gate_freeze.get("selected_head_refit_after_selection") is False
        and gate_freeze.get("contact_or_privileged_gate_inputs") is False
        and gate_freeze.get("confirmation_episodes_at_freeze") == 0
    ):
        raise RuntimeError("compiled gate/compiler/gate-freeze binding drift")
    gate_cost = manifest.get("gate_cost_per_reached_decision")
    expected_gate_cost = {
        "head_count": head_count,
        "feature_flops": 3_801,
        "affine_head_flops_each": 2_092,
        "all_head_flops": 4_184 if head_count == 2 else 16_736,
        "total_gate_flops": 7_985 if head_count == 2 else 20_537,
        "nonflop_operations": 5 if head_count == 2 else 11,
    }
    if gate_cost != expected_gate_cost:
        raise RuntimeError("execution manifest gate pricing drift")

    raw_records = raw.get("records")
    if not isinstance(raw_records, Mapping):
        raise RuntimeError("authenticated raw verifier result lacks record bindings")
    execution_files: list[dict[str, Any]] = []
    sidecar_files: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    histograms: list[list[int]] = []
    record_keys = {
        "slot",
        "episode_id",
        "path",
        "sha256",
        "sidecar_path",
        "sidecar_sha256",
        "source_raw_path",
        "source_raw_sha256",
        "source_raw_sidecar_path",
        "source_raw_sidecar_sha256",
        "call_histogram",
    }
    sidecar_keys = {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "role",
        "regime",
        "slot",
        "episode_id",
        "raw_path",
        "raw_sha256",
        "raw_sidecar_path",
        "raw_sidecar_sha256",
        "compiled_gate_path",
        "compiled_gate_sha256",
        "architecture",
        "candidate_id",
        "head_names",
        "input_loader_audit",
        "actual_adaptive_output",
        "dense_role",
        "call_histogram",
        "exact_compute",
        "equivalence",
        "target_persisted",
        "target_loss_computed",
        "contact_or_privileged_materialized",
        "no_gradients",
        "complete",
        "path",
        "sha256",
        "bytes",
        "arrays",
    }
    expected_array_keys = {
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
        "target",
    }
    for index, (record, episode_id) in enumerate(
        zip(records, raw["episode_ids"], strict=True)
    ):
        if not isinstance(record, Mapping):
            raise RuntimeError("confirmation execution record is not an object")
        _exact_keys(record, record_keys, f"confirmation execution record {index}")
        raw_record = raw_records.get(episode_id)
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("execution record has no authenticated raw record")
        part_path = (
            attempt_root
            / "data/confirmation"
            / regime
            / "execution"
            / f"{episode_id}.npz"
        )
        sidecar_path = part_path.with_suffix(".json")
        raw_path = REPO_ROOT / Path(str(raw_record["raw_path"]))
        raw_sidecar_path = raw_path.with_suffix(".json")
        if not (
            record.get("slot") == index == raw_record.get("slot")
            and not isinstance(record.get("slot"), bool)
            and record.get("episode_id") == episode_id
            and record.get("source_raw_path") == raw_record.get("raw_path")
            and record.get("source_raw_sha256") == raw_record.get("raw_sha256")
            and record.get("source_raw_sidecar_path")
            == _canonical_relative(raw_sidecar_path)
            and record.get("source_raw_sidecar_sha256")
            == sha256_file(raw_sidecar_path)
        ):
            raise RuntimeError("execution record/raw-sidecar binding drift")
        execution_file = _strict_file_link(
            record.get("path"),
            record.get("sha256"),
            expected_path=part_path,
            identities=identities,
        )
        sidecar_file = _strict_file_link(
            record.get("sidecar_path"),
            record.get("sidecar_sha256"),
            expected_path=sidecar_path,
            identities=identities,
        )
        sidecar = read_json(sidecar_path)
        _exact_keys(sidecar, sidecar_keys, f"confirmation execution sidecar {index}")
        arrays = sidecar.get("arrays")
        if not isinstance(arrays, Mapping) or set(arrays) != expected_array_keys:
            raise RuntimeError("confirmation execution sidecar array-key drift")
        for name, metadata in arrays.items():
            if not isinstance(metadata, Mapping):
                raise RuntimeError(f"execution array metadata is not an object: {name}")
            _exact_keys(metadata, {"shape", "dtype", "sha256"}, f"execution array {name}")
            if (
                not isinstance(metadata.get("shape"), list)
                or any(
                    not isinstance(dimension, int)
                    or isinstance(dimension, bool)
                    or dimension < 0
                    for dimension in metadata["shape"]
                )
                or not isinstance(metadata.get("dtype"), str)
            ):
                raise RuntimeError(f"execution array shape/dtype drift: {name}")
            _strict_sha256(metadata.get("sha256"), f"execution array hash {name}")
        histogram = sidecar.get("call_histogram")
        expected_episode_compute = _exact_compute_from_histograms([histogram], head_count)
        summary = {
            "slot": sidecar.get("slot"),
            "episode_id": sidecar.get("episode_id"),
            "path": sidecar.get("path"),
            "sha256": sidecar.get("sha256"),
            "sidecar_path": _canonical_relative(sidecar_path),
            "sidecar_sha256": sha256_file(sidecar_path),
            "source_raw_path": sidecar.get("raw_path"),
            "source_raw_sha256": sidecar.get("raw_sha256"),
            "source_raw_sidecar_path": sidecar.get("raw_sidecar_path"),
            "source_raw_sidecar_sha256": sidecar.get("raw_sidecar_sha256"),
            "call_histogram": histogram,
        }
        if dict(record) != summary:
            raise RuntimeError("execution manifest record/sidecar equality drift")
        if not (
            sidecar.get("schema_version") == 1
            and not isinstance(sidecar.get("schema_version"), bool)
            and sidecar.get("attempt") == ATTEMPT
            and _strict_int(
                sidecar.get("created_unix_ns"), "execution sidecar timestamp", minimum=1
            )
            and sidecar.get("role") == "confirmation"
            and sidecar.get("regime") == regime
            and sidecar.get("slot") == index
            and sidecar.get("episode_id") == episode_id
            and sidecar.get("raw_path") == raw_record.get("raw_path")
            and sidecar.get("raw_sha256") == raw_record.get("raw_sha256")
            and sidecar.get("raw_sidecar_path") == _canonical_relative(raw_sidecar_path)
            and sidecar.get("raw_sidecar_sha256") == sha256_file(raw_sidecar_path)
            and sidecar.get("compiled_gate_path") == _canonical_relative(compiled_path)
            and sidecar.get("compiled_gate_sha256") == compiled_hash
            and sidecar.get("architecture") == manifest.get("architecture")
            and sidecar.get("candidate_id") == manifest.get("candidate_id")
            and sidecar.get("head_names") == manifest.get("head_names")
            and sidecar.get("actual_adaptive_output")
            == "manual sequential sparse reached-row execution"
            and sidecar.get("dense_role")
            == "numerical shadow and fixed-depth comparator only"
            and sidecar.get("exact_compute") == expected_episode_compute
            and isinstance(sidecar.get("equivalence"), Mapping)
            and sidecar["equivalence"].get("passed") is True
            and sidecar.get("target_persisted") is True
            and sidecar.get("target_loss_computed") is False
            and sidecar.get("contact_or_privileged_materialized") is False
            and sidecar.get("no_gradients") is True
            and sidecar.get("complete") is True
            and sidecar.get("path") == _canonical_relative(part_path)
            and sidecar.get("sha256") == sha256_file(part_path)
            and sidecar.get("bytes") == execution_file["bytes"]
            and not isinstance(sidecar.get("bytes"), bool)
        ):
            raise RuntimeError("confirmation execution sidecar contract drift")
        execution_files.append(execution_file)
        sidecar_files.append(sidecar_file)
        episode_ids.append(str(episode_id))
        histograms.append(list(histogram))

    expected_compute = _exact_compute_from_histograms(histograms, head_count)
    if manifest.get("exact_compute") != expected_compute:
        raise RuntimeError("aggregate confirmation exact-compute drift")
    if manifest.get("compiled_gate_sha256") != source_bindings["compiled_gate"]["sha256"]:
        raise RuntimeError("execution compiled-gate/source-binding hash drift")
    confirmation_files = {
        item["path"]: item["sha256"]
        for item in (manifest_file, *execution_files, *sidecar_files)
    }
    checks = {
        "exact_v008_schema_attempt_role_dgp_counts": True,
        "source_raw_manifest_hash_mandatory_and_exact": True,
        "exact_execution_authorization_and_state": True,
        "exact_source_dgp_ledger_gate_freeze_bindings": True,
        "compiled_gate_compiler_freeze_crosslinks_exact": True,
        "canonical_nonlink_execution_and_sidecar_paths": True,
        "execution_manifest_sidecar_equality": True,
        "raw_execution_episode_ids_exact": episode_ids == raw["episode_ids"],
        "exact_compute_recomputed_from_call_histograms": True,
        "sparse_dense_equivalence": True,
        "module_frozen": True,
        "no_gradients": True,
        "pixels_action_input_only": True,
        "contact_or_privileged_not_materialized": True,
        "target_loss_not_computed_during_execution": True,
        "forbidden_sources_excluded": _negative_assertions(manifest),
        "no_symlink_or_inode_aliases": True,
    }
    return {
        "manifest": {
            "path": manifest_file["path"],
            "sha256": manifest_file["sha256"],
            "created_unix_ns": manifest["created_unix_ns"],
        },
        "role": "confirmation",
        "regime": regime,
        "episode_ids": episode_ids,
        "files": execution_files,
        "sidecars": sidecar_files,
        "source_bindings": verified_bindings,
        "confirmation_files": confirmation_files,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _confirmation_files(attempt_root: Path) -> list[Path]:
    roots = (
        attempt_root / "data/confirmation",
        attempt_root / "data/persistence_intents/confirmation",
    )
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"confirmation root is linked or non-directory: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"confirmation symlink forbidden: {path}")
            if path.is_file():
                files.append(path)
    return sorted(set(files), key=lambda item: str(item))


def _assert_exact_confirmation_namespace(
    attempt_root: Path, expected_files: Mapping[str, str]
) -> list[Path]:
    actual = _confirmation_files(attempt_root)
    actual_map = {_canonical_relative(path): sha256_file(path) for path in actual}
    normalized_expected = {
        str(path): _strict_sha256(digest, f"expected confirmation hash {path}")
        for path, digest in expected_files.items()
    }
    if actual_map != normalized_expected:
        raise RuntimeError(
            "confirmation namespace is not the exact authenticated closure: "
            f"missing={sorted(set(normalized_expected) - set(actual_map))} "
            f"unrelated={sorted(set(actual_map) - set(normalized_expected))} "
            f"hash_drift={sorted(path for path in set(actual_map) & set(normalized_expected) if actual_map[path] != normalized_expected[path])}"
        )
    return actual


def _controller_chronology_evidence(
    attempt_root: Path,
    *,
    current_state: str,
    prior_state: str,
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Bind a checkpoint claim to the controller and its last passed evidence.

    The controller will independently revalidate its ledger when advancing the
    state.  This local check prevents checkpoint construction from replacing
    controller facts with literals: it rehashes the current state and the last
    controller-recorded passing checkpoint, and checks the monotone outcome
    counters before emitting evidence.
    """

    attempt_root = Path(attempt_root).resolve()
    study_root = attempt_root.parent.parent
    state_path = study_root / "STATE.json"
    state = read_verified_controller()
    checkpoints = state.get("verified_checkpoints")
    completed = state.get("completed_states")
    last = state.get("last_verified_checkpoint")
    if not isinstance(checkpoints, list) or not isinstance(completed, list):
        raise RuntimeError("controller checkpoint history is malformed")
    if not isinstance(last, Mapping):
        raise RuntimeError("controller lacks a last verified checkpoint")

    evidence_path = resolve_repo_relative(str(last.get("evidence_path", "")))
    if not evidence_path.is_relative_to(attempt_root):
        raise RuntimeError("controller prior evidence is outside the active attempt")
    evidence = read_json(evidence_path)
    observed_counts = {
        key: int(state.get(key, -1)) for key in expected_counts
    }
    expected_attempt_path = relative_to_repo(attempt_root)
    evidence_created = int(evidence.get("created_unix_ns", 0))
    checkpoint_created = int(last.get("created_unix_ns", 0))
    state_updated = int(state.get("updated_unix_ns", 0))
    checks = {
        "active_attempt_exact": state.get("active_attempt") == attempt_root.name,
        "active_attempt_path_exact": state.get("active_attempt_path")
        == expected_attempt_path,
        "current_state_exact": state.get("current_state") == current_state,
        "prior_state_completed_last": bool(completed)
        and completed[-1] == prior_state,
        "last_checkpoint_pointer_exact": bool(checkpoints)
        and checkpoints[-1] == last,
        "prior_evidence_exists_and_rehashes": evidence_path.is_file()
        and sha256_file(evidence_path) == last.get("evidence_sha256"),
        "prior_evidence_passed": evidence.get("passed") is True,
        "prior_evidence_state_exact": evidence.get("checkpoint_state")
        == prior_state,
        "outcome_counts_exact": observed_counts
        == {key: int(value) for key, value in expected_counts.items()},
        "confirmation_outcomes_unopened": state.get(
            "confirmation_outcomes_opened_for_analysis"
        )
        is False,
        "no_terminal_decision": state.get("terminal_label") is None,
        "chronology_monotone": 0 < evidence_created <= checkpoint_created
        <= state_updated,
    }
    return {
        "state": {
            "path": relative_to_repo(state_path),
            "sha256": sha256_file(state_path),
            "current_state": state.get("current_state"),
            "updated_unix_ns": state_updated,
            "ledger_event_count": state.get("ledger_event_count"),
            "ledger_head_sha256": state.get("ledger_head_sha256"),
        },
        "prior_checkpoint": {
            "path": relative_to_repo(evidence_path),
            "sha256": sha256_file(evidence_path),
            "checkpoint_state": evidence.get("checkpoint_state"),
            "created_unix_ns": evidence_created,
        },
        "outcome_counts": observed_counts,
        "confirmation_outcomes_opened_for_analysis": state.get(
            "confirmation_outcomes_opened_for_analysis"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def smoke_checkpoint(
    *,
    attempt_root: Path = ATTEMPT_ROOT,
    write: bool = False,
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    all_ids: list[str] = []
    for regime in DGP_ORDER:
        raw = verify_raw_manifest(
            _raw_path(attempt_root, "smoke", regime),
            role="smoke",
            regime=regime,
            expected_episodes=SMOKE_EPISODES_PER_DGP,
        )
        execution = verify_execution_manifest(
            _execution_path(attempt_root, "smoke", regime),
            role="smoke",
            regime=regime,
            expected_episodes=SMOKE_EPISODES_PER_DGP,
            raw=raw,
        )
        all_ids.extend(raw["episode_ids"])
        regimes[regime] = {
            "raw": raw,
            "execution": execution,
            "passed": raw["passed"] and execution["passed"],
        }
    controller = _controller_chronology_evidence(
        attempt_root,
        current_state="EXCLUDED_MECHANICAL_SMOKE",
        prior_state="PRE_CONFIRMATION_PACKAGE_SEAL",
        expected_counts={
            "smoke_outcome_episodes": 4 * SMOKE_EPISODES_PER_DGP,
            "confirmation_outcome_episodes_generated": 0,
            "confirmation_outcome_episodes_executed": 0,
        },
    )
    manifest_forbidden_evidence = all(
        item["raw"]["checks"]["pixels_action_only"]
        and item["raw"]["checks"]["forbidden_sources_excluded"]
        and item["execution"]["checks"]["pixels_action_input_only"]
        and item["execution"]["checks"]["contact_or_privileged_not_materialized"]
        for item in regimes.values()
    )
    checks = {
        "all_four_dgps_present": tuple(regimes) == DGP_ORDER,
        "exactly_six_per_dgp": len(all_ids) == 4 * SMOKE_EPISODES_PER_DGP,
        "all_smoke_ids_unique": _all_unique(all_ids),
        "all_regimes_mechanically_passed": all(item["passed"] for item in regimes.values()),
        "smoke_permanently_excluded_from_fit_selection_inference": all(
            item["raw"]["checks"]["smoke_permanently_excluded_if_applicable"]
            for item in regimes.values()
        ),
        "smoke_losses_not_analyzed": all(
            item["execution"]["checks"][
                "target_loss_not_computed_during_execution"
            ]
            for item in regimes.values()
        ),
        "confirmation_directory_absent": not (attempt_root / "data/confirmation").exists(),
        "controller_chronology_passed": controller["passed"],
        "forbidden_scientific_inputs_excluded_by_manifests": manifest_forbidden_evidence,
    }
    result = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "checkpoint_state": "EXCLUDED_MECHANICAL_SMOKE",
        "created_unix_ns": time.time_ns(),
        "classification": "permanently_excluded_mechanical_smoke",
        "regimes": regimes,
        "controller_chronology": controller,
        "checks": checks,
        "passed": all(checks.values()),
        "smoke_episode_count": len(all_ids),
        "confirmation_outcome_episodes": 0,
        "contact_motion_phase_reward_success_opened": not manifest_forbidden_evidence,
    }
    if write:
        atomic_json(
            attempt_root / "audit/excluded_mechanical_smoke.json", result, exclusive=True
        )
    return result


def confirmation_generation_checkpoint(
    expected_per_dgp: int,
    *,
    attempt_root: Path = ATTEMPT_ROOT,
    write: bool = False,
) -> dict[str, Any]:
    if expected_per_dgp not in range(500, 4501, 500):
        raise ValueError("confirmation size must be one frozen power-grid value")
    regimes: dict[str, Any] = {}
    all_ids: list[str] = []
    authenticated_files: dict[str, str] = {}
    for regime in DGP_ORDER:
        raw = verify_raw_manifest(
            _raw_path(attempt_root, "confirmation", regime),
            role="confirmation",
            regime=regime,
            expected_episodes=expected_per_dgp,
        )
        regimes[regime] = raw
        all_ids.extend(raw["episode_ids"])
        for relative, digest in raw["confirmation_files"].items():
            if relative in authenticated_files and authenticated_files[relative] != digest:
                raise RuntimeError(f"confirmation generation file collision: {relative}")
            authenticated_files[relative] = digest
    confirmation_files = _assert_exact_confirmation_namespace(
        attempt_root, authenticated_files
    )
    replacement_closure = _validate_final_replacement_claim_references(
        attempt_root, regimes
    )
    expected_total = len(DGP_ORDER) * expected_per_dgp
    controller = _controller_chronology_evidence(
        attempt_root,
        current_state="CONFIRMATION_GENERATION",
        prior_state="EXCLUDED_MECHANICAL_SMOKE",
        expected_counts={
            "confirmation_outcome_episodes_generated": expected_total,
            "confirmation_outcome_episodes_executed": 0,
        },
    )
    manifest_forbidden_evidence = all(
        item["checks"]["pixels_action_only"]
        and item["checks"]["forbidden_sources_excluded"]
        for item in regimes.values()
    )
    checks = {
        "all_four_dgps_present": tuple(regimes) == DGP_ORDER,
        "fixed_common_size": all(
            len(item["episode_ids"]) == expected_per_dgp for item in regimes.values()
        ),
        "exact_total_episode_count": len(all_ids) == expected_total,
        "all_confirmation_ids_unique": _all_unique(all_ids),
        "all_raw_manifests_passed": all(item["passed"] for item in regimes.values()),
        "no_sequential_expansion": controller["outcome_counts"][
            "confirmation_outcome_episodes_generated"
        ]
        == expected_total,
        "outcome_blind_retention": all(
            item["checks"]["outcome_blind_retention"] for item in regimes.values()
        ),
        "confirmation_outcomes_not_opened_for_analysis": controller[
            "confirmation_outcomes_opened_for_analysis"
        ]
        is False,
        "controller_chronology_passed": controller["passed"],
        "forbidden_scientific_inputs_excluded_by_manifests": manifest_forbidden_evidence,
        "confirmation_namespace_exact_no_unrelated_files": len(confirmation_files)
        == len(authenticated_files),
        "replacement_registry_final_claim_reference_closure_exact": True,
    }
    result = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "checkpoint_state": "CONFIRMATION_GENERATION",
        "created_unix_ns": time.time_ns(),
        "expected_per_dgp": expected_per_dgp,
        "regimes": regimes,
        "replacement_registry_closure": replacement_closure,
        "controller_chronology": controller,
        "checks": checks,
        "passed": all(checks.values()),
        "confirmation_outcome_episodes_generated": len(all_ids),
        "confirmation_outcomes_opened_for_analysis": controller[
            "confirmation_outcomes_opened_for_analysis"
        ],
        "contact_motion_phase_reward_success_opened": not manifest_forbidden_evidence,
    }
    if write:
        atomic_json(
            attempt_root / "audit/confirmation_generation_complete.json",
            result,
            exclusive=True,
        )
    return result


def confirmation_execution_checkpoint(
    expected_per_dgp: int,
    *,
    attempt_root: Path = ATTEMPT_ROOT,
    write: bool = False,
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    raw_by_regime: dict[str, Any] = {}
    all_ids: list[str] = []
    authenticated_files: dict[str, str] = {}
    for regime in DGP_ORDER:
        raw = verify_raw_manifest(
            _raw_path(attempt_root, "confirmation", regime),
            role="confirmation",
            regime=regime,
            expected_episodes=expected_per_dgp,
        )
        execution = verify_execution_manifest(
            _execution_path(attempt_root, "confirmation", regime),
            role="confirmation",
            regime=regime,
            expected_episodes=expected_per_dgp,
            raw=raw,
        )
        regimes[regime] = {"raw": raw, "execution": execution, "passed": raw["passed"] and execution["passed"]}
        raw_by_regime[regime] = raw
        all_ids.extend(execution["episode_ids"])
        for source in (raw, execution):
            for relative, digest in source["confirmation_files"].items():
                if relative in authenticated_files and authenticated_files[relative] != digest:
                    raise RuntimeError(f"confirmation execution file collision: {relative}")
                authenticated_files[relative] = digest
    confirmation_files = _assert_exact_confirmation_namespace(
        attempt_root, authenticated_files
    )
    replacement_closure = _validate_final_replacement_claim_references(
        attempt_root, raw_by_regime
    )
    expected_total = len(DGP_ORDER) * expected_per_dgp
    controller = _controller_chronology_evidence(
        attempt_root,
        current_state="CONFIRMATION_EXECUTION",
        prior_state="CONFIRMATION_GENERATION",
        expected_counts={
            "confirmation_outcome_episodes_generated": expected_total,
            "confirmation_outcome_episodes_executed": expected_total,
        },
    )
    manifest_forbidden_evidence = all(
        item["raw"]["checks"]["pixels_action_only"]
        and item["raw"]["checks"]["forbidden_sources_excluded"]
        and item["execution"]["checks"]["pixels_action_input_only"]
        and item["execution"]["checks"]["contact_or_privileged_not_materialized"]
        and item["execution"]["checks"][
            "target_loss_not_computed_during_execution"
        ]
        for item in regimes.values()
    )
    checks = {
        "all_four_dgps_present": tuple(regimes) == DGP_ORDER,
        "exact_total_episode_count": len(all_ids) == expected_total,
        "exact_total_row_count": len(all_ids) * ROWS_PER_EPISODE
        == expected_total * ROWS_PER_EPISODE,
        "all_confirmation_ids_unique": _all_unique(all_ids),
        "all_raw_and_execution_manifests_passed": all(
            item["passed"] for item in regimes.values()
        ),
        "all_sparse_dense_equivalence_passed": all(
            item["execution"]["checks"]["sparse_dense_equivalence"]
            for item in regimes.values()
        ),
        "confirmation_outcomes_not_opened_for_analysis": controller[
            "confirmation_outcomes_opened_for_analysis"
        ]
        is False,
        "controller_chronology_passed": controller["passed"],
        "forbidden_scientific_inputs_excluded_by_manifests": manifest_forbidden_evidence,
        "confirmation_namespace_exact_no_unrelated_files": len(confirmation_files)
        == len(authenticated_files),
        "replacement_registry_final_claim_reference_closure_exact": True,
    }
    result = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "checkpoint_state": "CONFIRMATION_EXECUTION",
        "created_unix_ns": time.time_ns(),
        "expected_per_dgp": expected_per_dgp,
        "regimes": regimes,
        "replacement_registry_closure": replacement_closure,
        "controller_chronology": controller,
        "checks": checks,
        "passed": all(checks.values()),
        "confirmation_outcome_episodes_executed": len(all_ids),
        "confirmation_rows": len(all_ids) * ROWS_PER_EPISODE,
        "confirmation_outcomes_opened_for_analysis": controller[
            "confirmation_outcomes_opened_for_analysis"
        ],
        "contact_motion_phase_reward_success_opened": not manifest_forbidden_evidence,
    }
    if write:
        atomic_json(
            attempt_root / "audit/confirmation_execution_complete.json",
            result,
            exclusive=True,
        )
    return result


def confirmation_input_seal(
    expected_per_dgp: int,
    *,
    attempt_root: Path = ATTEMPT_ROOT,
    write: bool = False,
) -> dict[str, Any]:
    """Hash every raw/execution confirmation input before first array open."""

    regimes: dict[str, Any] = {}
    raw_hashes: dict[str, str] = {}
    raw_sidecar_hashes: dict[str, str] = {}
    persistence_intent_hashes: dict[str, str] = {}
    execution_hashes: dict[str, str] = {}
    execution_sidecar_hashes: dict[str, str] = {}
    authenticated_confirmation_files: dict[str, str] = {}
    raw_by_regime: dict[str, Any] = {}
    all_ids: list[str] = []
    chronology = True
    manifest_forbidden_evidence = True
    total_raw_bytes = 0
    total_execution_bytes = 0
    expected_total = len(DGP_ORDER) * expected_per_dgp
    controller = _controller_chronology_evidence(
        attempt_root,
        current_state="CONFIRMATION_INPUT_SEAL",
        prior_state="CONFIRMATION_EXECUTION",
        expected_counts={
            "confirmation_outcome_episodes_generated": expected_total,
            "confirmation_outcome_episodes_executed": expected_total,
        },
    )
    pre_confirmation_path = (
        attempt_root / "audit/pre_confirmation_package_seal.json"
    )
    pre_confirmation = read_json(pre_confirmation_path)
    if not (
        pre_confirmation.get("passed") is True
        and pre_confirmation.get("attempt") == ATTEMPT
        and pre_confirmation.get("checkpoint_state")
        == "PRE_CONFIRMATION_PACKAGE_SEAL"
    ):
        raise RuntimeError("pre-confirmation package seal header drift")
    frozen_counts = pre_confirmation.get(
        "confirmation_episode_count_per_regime"
    )
    if (
        not isinstance(frozen_counts, Mapping)
        or set(frozen_counts) != set(DGP_ORDER)
        or any(int(frozen_counts[name]) != expected_per_dgp for name in DGP_ORDER)
    ):
        raise RuntimeError("input-seal size differs from pre-confirmation freeze")
    pre_confirmation_files = _validated_sealed_files(
        pre_confirmation.get("sealed_files", {})
    )
    for regime in DGP_ORDER:
        raw = verify_raw_manifest(
            _raw_path(attempt_root, "confirmation", regime),
            role="confirmation",
            regime=regime,
            expected_episodes=expected_per_dgp,
        )
        execution = verify_execution_manifest(
            _execution_path(attempt_root, "confirmation", regime),
            role="confirmation",
            regime=regime,
            expected_episodes=expected_per_dgp,
            raw=raw,
        )
        manifest_forbidden_evidence &= (
            raw["checks"]["pixels_action_only"]
            and raw["checks"]["forbidden_sources_excluded"]
            and execution["checks"]["pixels_action_input_only"]
            and execution["checks"]["contact_or_privileged_not_materialized"]
            and execution["checks"][
                "target_loss_not_computed_during_execution"
            ]
        )
        chronology &= (
            raw["manifest"]["created_unix_ns"] > 0
            and raw["manifest"]["created_unix_ns"]
            < execution["manifest"]["created_unix_ns"]
        )
        for record in raw["files"]:
            raw_hashes[record["path"]] = record["sha256"]
            total_raw_bytes += int(record["bytes"])
        for record in raw["sidecars"]:
            raw_sidecar_hashes[record["path"]] = record["sha256"]
            total_raw_bytes += int(record["bytes"])
        for record in raw["intents"]:
            persistence_intent_hashes[record["path"]] = record["sha256"]
            total_raw_bytes += int(record["bytes"])
        for record in execution["files"]:
            execution_hashes[record["path"]] = record["sha256"]
            total_execution_bytes += int(record["bytes"])
        for record in execution["sidecars"]:
            execution_sidecar_hashes[record["path"]] = record["sha256"]
            total_execution_bytes += int(record["bytes"])
        for source in (raw, execution):
            for relative, digest in source["confirmation_files"].items():
                if (
                    relative in authenticated_confirmation_files
                    and authenticated_confirmation_files[relative] != digest
                ):
                    raise RuntimeError(
                        f"confirmation input file collision: {relative}"
                    )
                authenticated_confirmation_files[relative] = digest
        regimes[regime] = {
            "raw_manifest": raw["manifest"],
            "execution_manifest": execution["manifest"],
            "episode_count": len(raw["episode_ids"]),
            "passed": raw["passed"] and execution["passed"],
        }
        raw_by_regime[regime] = raw
        all_ids.extend(raw["episode_ids"])
    manifest_forbidden_evidence = bool(manifest_forbidden_evidence)
    replacement_closure = _validate_final_replacement_claim_references(
        attempt_root, raw_by_regime
    )
    sealed_files = dict(pre_confirmation_files)
    pre_confirmation_relative = relative_to_repo(pre_confirmation_path)
    sealed_files[pre_confirmation_relative] = sha256_file(pre_confirmation_path)
    confirmation_files = _assert_exact_confirmation_namespace(
        attempt_root, authenticated_confirmation_files
    )
    for relative, digest in authenticated_confirmation_files.items():
        if relative in sealed_files and sealed_files[relative] != digest:
            raise RuntimeError(f"confirmation file conflicts with frozen path: {relative}")
        sealed_files[relative] = digest
    for relative, digest in replacement_closure["support_files"].items():
        if relative in sealed_files and sealed_files[relative] != digest:
            raise RuntimeError(
                f"replacement support file conflicts with frozen path: {relative}"
            )
        sealed_files[relative] = digest
    sealed_files = _validated_sealed_files(sealed_files)
    checks = {
        "all_four_dgps_present": tuple(regimes) == DGP_ORDER,
        "all_manifests_and_files_rehashed": all(
            item["passed"] for item in regimes.values()
        ),
        "exact_episode_count": len(all_ids) == expected_total,
        "exact_execution_part_count": len(execution_hashes) == expected_total,
        "exact_raw_array_count": len(raw_hashes) == expected_total,
        "exact_raw_sidecar_count": len(raw_sidecar_hashes) == expected_total,
        "exact_persistence_intent_count": len(persistence_intent_hashes)
        == expected_total,
        "exact_execution_sidecar_count": len(execution_sidecar_hashes)
        == expected_total,
        "exact_row_count": len(all_ids) * ROWS_PER_EPISODE
        == expected_total * ROWS_PER_EPISODE,
        "all_episode_ids_unique": _all_unique(all_ids),
        "raw_manifests_precede_execution_manifests": chronology,
        "confirmation_arrays_not_opened_before_seal": controller[
            "confirmation_outcomes_opened_for_analysis"
        ]
        is False,
        "contact_motion_phase_reward_success_not_opened": manifest_forbidden_evidence,
        "controller_chronology_passed": controller["passed"],
        "pre_confirmation_seal_rehashed": sealed_files[
            pre_confirmation_relative
        ]
        == sha256_file(pre_confirmation_path),
        "pre_confirmation_sealed_files_preserved_transitively": all(
            sealed_files.get(path) == digest
            for path, digest in pre_confirmation_files.items()
        ),
        "every_confirmation_file_listed": set(authenticated_confirmation_files)
        == {_canonical_relative(path) for path in confirmation_files},
        "no_unrelated_confirmation_file_accepted": len(confirmation_files)
        == len(authenticated_confirmation_files),
        "sealed_files_have_no_links_or_inode_aliases": True,
        "replacement_registry_and_failure_chain_frozen_at_final_head": all(
            sealed_files.get(path) == digest
            for path, digest in replacement_closure["support_files"].items()
        ),
    }
    result = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "checkpoint_state": "CONFIRMATION_INPUT_SEAL",
        "created_unix_ns": time.time_ns(),
        "expected_per_dgp": expected_per_dgp,
        "regimes": regimes,
        "replacement_registry_closure": replacement_closure,
        "controller_chronology": controller,
        "raw_and_sidecar_hashes": raw_hashes,
        "raw_sidecar_hashes": raw_sidecar_hashes,
        "persistence_intent_hashes": persistence_intent_hashes,
        "execution_part_hashes": execution_hashes,
        "execution_sidecar_hashes": execution_sidecar_hashes,
        "sealed_files": sealed_files,
        "pre_confirmation_package_seal_path": pre_confirmation_relative,
        "pre_confirmation_package_seal_sha256": sha256_file(
            pre_confirmation_path
        ),
        "pre_confirmation_sealed_file_count": len(pre_confirmation_files),
        "confirmation_file_count": len(confirmation_files),
        "total_raw_and_sidecar_bytes": total_raw_bytes,
        "total_execution_bytes": total_execution_bytes,
        "exact_episode_count": len(all_ids),
        "exact_execution_part_count": len(execution_hashes),
        "exact_row_count": len(all_ids) * ROWS_PER_EPISODE,
        "checks": checks,
        "passed": all(checks.values()),
        # This exact field is consumed by analysis.py before any np.load call.
        "confirmation_arrays_opened_before_seal": controller[
            "confirmation_outcomes_opened_for_analysis"
        ],
        "contact_motion_phase_reward_success_opened": not manifest_forbidden_evidence,
    }
    if write:
        atomic_json(
            attempt_root / "audit/confirmation_input_seal.json",
            result,
            exclusive=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke")
    for name in ("generation", "execution", "seal"):
        child = subparsers.add_parser(name)
        child.add_argument("--expected-per-dgp", required=True, type=int)
    arguments = parser.parse_args()
    if arguments.command == "smoke":
        result = smoke_checkpoint(write=True)
    elif arguments.command == "generation":
        result = confirmation_generation_checkpoint(
            arguments.expected_per_dgp, write=True
        )
    elif arguments.command == "execution":
        result = confirmation_execution_checkpoint(
            arguments.expected_per_dgp, write=True
        )
    else:
        result = confirmation_input_seal(arguments.expected_per_dgp, write=True)
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
