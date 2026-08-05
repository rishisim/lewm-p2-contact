#!/usr/bin/env python3
"""Build and verify immutable path/hash/byte manifests without opening arrays.

Only byte streams and filesystem metadata are inspected.  NPZ/HDF5 contents
are never decoded here.  Local v004 source/configuration paths must be regular,
non-symlink files with unique inodes; the two inherited v004 interpreter
symlinks are recorded as explicit, immutable exceptions because those exact
paths are part of the consumed terminal contract.  Scientific objects remain
the byte-identical v001 objects: v004 changes only procedural authorization and
source-closure machinery.
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
from typing import Any, Iterable, Mapping, Sequence

from runtime_contract import (
    ATTEMPT_ROOT,
    EVALUATION_PYTHON,
    GENERATION_PYTHON,
    REPO_ROOT,
    V5_RUNTIME_CONTRACT_PATH,
    V5_RUNTIME_CONTRACT_SHA256,
    load_v5_runtime_contract,
    sha256_file,
)

ACTIVE_ATTEMPT = ATTEMPT_ROOT.name
EXPECTED_ACTIVE_ATTEMPT = "v008"
SCIENCE_ATTEMPT = "v001"
if ACTIVE_ATTEMPT != EXPECTED_ACTIVE_ATTEMPT:
    raise RuntimeError(
        "v004 manifest implementation loaded from the wrong attempt: "
        f"{ACTIVE_ATTEMPT!r}"
    )

V5_FIXED_WHITENING_PATH = (
    REPO_ROOT
    / "runs/lewm_v5_readiness_program/v5_package_versions/v004/freeze/whitening.npz"
)
V5_FIXED_WHITENING_SHA256 = (
    "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796"
)


PRE_DATA_MANIFEST_PATH = ATTEMPT_ROOT / "audit/pre_data_source_manifest.json"
ALLOWED_SOURCE_SUFFIXES = frozenset({".py", ".json", ".md"})
PRE_DATA_STATIC_RELATIVE_PATHS = frozenset(
    {
        "DGP_MATRIX.json",
        "DIAGNOSTIC_ACCOUNT.md",
        "PREREGISTRATION.md",
        "analysis.py",
        "build_manifest.py",
        "build_seed_ledger.py",
        "candidate_grid.json",
        "capture_verifier.py",
        "checkpoints.py",
        "cohort_seed_ledger.json",
        "compile_gate.py",
        "counted_features.py",
        "fit_select.py",
        "flops.py",
        "fit_inheritance.py",
        "generator.py",
        "inherited_authorization.py",
        "independent_verify.py",
        "input_loader.py",
        "latency.py",
        "launcher.py",
        "outcome_mapping.json",
        "power_analysis.py",
        "power_baseline.json",
        "power_rule.json",
        "preseal.py",
        "runner.py",
        "runtime_contract.py",
        "scientific_replay.py",
        "scientific_replay_launcher.py",
        "study_common.py",
        "tests/test_analysis.py",
        "tests/test_attempt_parameterization_v002.py",
        "tests/test_checkpoint_hardening.py",
        "tests/test_fit_select.py",
        "tests/test_generator_contract.py",
        "tests/test_independent_verify.py",
        "tests/test_inherited_authorization_v002.py",
        "tests/test_manifest_closure_v002.py",
        "tests/test_preseal.py",
        "tests/test_program_controller.py",
        "tests/test_runner_gate.py",
        "tests/test_scientific_replay.py",
        "tests/test_seed_power.py",
        "tests/test_terminal_workflow.py",
        "tests/test_workflow.py",
        "tests/test_version_forward_v002.py",
        "tests/test_version_forward_transaction.py",
        "terminal_workflow.py",
        "verifier_contract.json",
        "verifier_contract_no_candidate.json",
        "verifier_contract_power_infeasible.json",
        "verify_identifier_freshness.py",
        "verify_version_forward.py",
        "version_forward_prepare.py",
        "version_forward_transaction.py",
        "workflow.py",
    }
)
PRE_DATA_REPOSITORY_RELATIVE_PATHS = frozenset(
    {
        "runs/lewm_domain_robust_gate/LEDGER_CHAIN_GENESIS.json",
        "runs/lewm_domain_robust_gate/README.md",
        "runs/lewm_domain_robust_gate/program.py",
    }
)
PRE_DATA_AUDIT_INPUTS = frozenset(
    {
        "audit/bootstrap_audit.json",
        "audit/diagnostic_account.json",
        "audit/preregistration_and_power.json",
        "audit/identifier_freshness_verification.json",
    }
)
GENERATED_AUDIT_NAMES = frozenset(
    {
        "audit/analysis_execution_invalid.json",
        "audit/candidate_selection.json",
        "audit/confirmation_execution_complete.json",
        "audit/confirmation_generation_complete.json",
        "audit/confirmation_input_seal.json",
        "audit/confirmation_power_and_cohort_freeze.json",
        "audit/confirmation_scientific_replay.json",
        "audit/excluded_mechanical_smoke.json",
        "audit/fit_cohorts.json",
        "audit/fit_lock_checkpoint.json",
        "audit/fit_scientific_replay.json",
        "audit/gate_freeze_checkpoint.json",
        "audit/implementation_complete.json",
        "audit/inherited_fit_inventory.json",
        "audit/independent_verification.json",
        "audit/post_terminal_reporting.json",
        "audit/pre_data_inheritance_seal.json",
        "audit/version_forward_transaction_receipt.json",
        "audit/preseal_qualification.json",
        "audit/pre_data_source_manifest.json",
        "audit/pre_data_seal.json",
        "audit/pre_selection_manifest.json",
        "audit/pre_selection_seal.json",
        "audit/pre_confirmation_manifest.json",
        "audit/pre_confirmation_seal.json",
        "audit/pre_confirmation_package_seal.json",
        "audit/scientific_replay/confirmation_markov_oracle.json",
        "audit/scientific_replay/confirmation_native_plan.json",
        "audit/scientific_replay/confirmation_plan_action_noise_0p2.json",
        "audit/scientific_replay/confirmation_plan_random_action_0p1.json",
        "audit/scientific_replay/fit_markov_oracle.json",
        "audit/scientific_replay/fit_native_plan.json",
        "audit/scientific_replay/fit_plan_action_noise_0p2.json",
        "audit/scientific_replay/fit_plan_random_action_0p1.json",
        "audit/scientific_replay/selection_markov_oracle.json",
        "audit/scientific_replay/selection_native_plan.json",
        "audit/scientific_replay/selection_plan_action_noise_0p2.json",
        "audit/scientific_replay/selection_plan_random_action_0p1.json",
        "audit/scientific_replay/smoke_markov_oracle.json",
        "audit/scientific_replay/smoke_native_plan.json",
        "audit/scientific_replay/smoke_plan_action_noise_0p2.json",
        "audit/scientific_replay/smoke_plan_random_action_0p1.json",
        "audit/scientific_replay_qualification.json",
        "audit/selection_cohorts.json",
        "audit/selection_scientific_replay.json",
        "audit/smoke_scientific_replay.json",
        "audit/terminal_input_manifest.json",
    }
)
GENERATED_ATTEMPT_PRODUCTS = frozenset(
    {
        "FOLLOW_ON_TASK.md",
        "INDEPENDENT_AUDIT.md",
        "LIMITATIONS.md",
        "REPORT.md",
        "ROBUSTNESS_MAP.json",
        "analysis_result.json",
        "decision.json",
        "fit/fit_lock.json",
        "freeze/compiled_gate_manifest.json",
        "freeze/gate_freeze.json",
        "metrics/bootstrap_summary.json",
        "metrics/compute_ledger.json",
        "metrics/fit_power_summary.json",
        "metrics/latency_and_resources.json",
        "metrics/selection_power_summary.json",
        "power_analysis.json",
        "selection/selection_ledger.json",
    }
)
GENERATED_ROLES = ("fit", "selection", "smoke", "confirmation")
GENERATED_REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
GENERATED_EXECUTION_FAILURE_NAMES = frozenset(
    f"audit/{role}_{regime}_execution_failure.json"
    for role in GENERATED_ROLES
    for regime in GENERATED_REGIMES
)
MUTABLE_STUDY_ROOT_RELATIVE_PATHS = frozenset(
    {
        "runs/lewm_domain_robust_gate/STATE.json",
        "runs/lewm_domain_robust_gate/STATE_TRANSACTION.json",
        "runs/lewm_domain_robust_gate/VERSION_FORWARD_TRANSACTION.json",
    }
)
PRE_DATA_LABEL = (
    f"{ACTIVE_ATTEMPT}_pre_data_normative_source_and_inherited_runtime"
)
INHERITED_RUNTIME_SYMLINKS = frozenset(
    {str(EVALUATION_PYTHON), str(GENERATION_PYTHON)}
)


class ManifestError(RuntimeError):
    """A path is missing, aliased, linked, mutable, or hash-drifted."""


def _atomic_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable manifest already exists: {path}")
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


def write_manifest_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Public exclusive writer used by the fail-closed seal orchestrator."""

    _atomic_json_exclusive(path, value)


def _relative_to_repo(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(REPO_ROOT.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ManifestError(f"path is not repository-contained: {path}") from exc


def _record(path: Path, *, local_source: bool) -> dict[str, Any]:
    raw = Path(path)
    try:
        link_stat = raw.lstat()
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest input is missing: {raw}") from exc
    is_symlink = stat.S_ISLNK(link_stat.st_mode)
    if local_source and is_symlink:
        raise ManifestError(f"local source symlink forbidden: {raw}")
    if is_symlink and str(raw) not in INHERITED_RUNTIME_SYMLINKS:
        raise ManifestError(f"unsealed external symlink forbidden: {raw}")
    if not raw.is_file():
        raise ManifestError(f"manifest input is not a file: {raw}")
    resolved = raw.resolve(strict=True)
    target_stat = resolved.stat()
    if not stat.S_ISREG(target_stat.st_mode):
        raise ManifestError(f"manifest target is not regular: {raw}")
    return {
        "sha256": sha256_file(raw),
        "bytes": int(target_stat.st_size),
        "symlink": is_symlink,
        "symlink_target": os.readlink(raw) if is_symlink else None,
        "resolved_path": str(resolved),
        "device": int(target_stat.st_dev),
        "inode": int(target_stat.st_ino),
        "regular_file": True,
        "contents_decoded": False,
    }


def _assert_unique_local_inodes(records: Mapping[str, Mapping[str, Any]]) -> None:
    seen: dict[tuple[int, int], str] = {}
    aliases: list[tuple[str, str]] = []
    for path, record in sorted(records.items()):
        identity = (int(record["device"]), int(record["inode"]))
        if identity in seen:
            aliases.append((seen[identity], path))
        else:
            seen[identity] = path
    if aliases:
        raise ManifestError(f"local manifest contains inode aliases: {aliases}")


def _study_root() -> Path:
    """Return the study root from the active attempt, including in tests."""

    return ATTEMPT_ROOT.parents[1]


def _allowed_suffix(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_SOURCE_SUFFIXES


def _walk_attempt_candidates() -> tuple[dict[str, Path], list[str]]:
    """Discover every allowed-suffix attempt entry without following links."""

    found: dict[str, Path] = {}
    linked_directories: list[str] = []
    if not ATTEMPT_ROOT.is_dir():
        return found, linked_directories
    for directory, directory_names, file_names in os.walk(
        ATTEMPT_ROOT, topdown=True, followlinks=False
    ):
        base = Path(directory)
        retained: list[str] = []
        for name in directory_names:
            path = base / name
            if path.is_symlink():
                linked_directories.append(
                    path.relative_to(ATTEMPT_ROOT).as_posix()
                )
            else:
                retained.append(name)
        directory_names[:] = retained
        for name in file_names:
            path = base / name
            if not _allowed_suffix(path):
                continue
            relative = path.relative_to(ATTEMPT_ROOT).as_posix()
            if relative in found:
                raise ManifestError(f"duplicate discovered attempt path: {relative}")
            found[relative] = path
    return found, sorted(linked_directories)


def _walk_study_root_candidates() -> tuple[dict[str, Path], list[str]]:
    """Discover allowed-suffix immediate files at the shared study root."""

    found: dict[str, Path] = {}
    linked_directories: list[str] = []
    root = _study_root()
    if not root.is_dir():
        return found, linked_directories
    for path in root.iterdir():
        if path.is_symlink() and path.is_dir():
            linked_directories.append(path.name)
            continue
        if not _allowed_suffix(path):
            continue
        try:
            relative = path.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise ManifestError(
                f"study-root candidate is outside repository: {path}"
            ) from exc
        if relative in found:
            raise ManifestError(f"duplicate discovered study-root path: {relative}")
        found[relative] = path
    return found, sorted(linked_directories)


def _assert_discovered_integrity(
    attempt: Mapping[str, Path],
    repository: Mapping[str, Path],
    linked_directories: Sequence[str],
) -> None:
    """Reject all local links, non-files, and inode aliases in the closure."""

    if linked_directories:
        raise ManifestError(
            "local source-closure directory symlinks forbidden: "
            f"{sorted(linked_directories)}"
        )
    seen: dict[tuple[int, int], str] = {}
    aliases: list[tuple[str, str]] = []
    for relative, path in sorted(
        [*attempt.items(), *repository.items()], key=lambda item: item[0]
    ):
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise ManifestError(
                f"discovered source-closure entry vanished: {path}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ManifestError(f"local source symlink forbidden: {path}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ManifestError(f"local source candidate is not regular: {path}")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in seen:
            aliases.append((seen[identity], relative))
        else:
            seen[identity] = relative
    if aliases:
        raise ManifestError(f"source-closure contains inode aliases: {aliases}")


def _prospective_episode_ids() -> dict[tuple[str, str], frozenset[str]]:
    """Read the frozen ledger and return every prospectively assigned ID."""

    try:
        ledger = json.loads(
            (ATTEMPT_ROOT / "cohort_seed_ledger.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("cannot read the prospective cohort seed ledger") from exc
    regimes = ledger.get("regimes") if isinstance(ledger, Mapping) else None
    if not isinstance(regimes, Mapping):
        raise ManifestError("prospective cohort seed ledger regimes are absent")
    result: dict[tuple[str, str], frozenset[str]] = {}
    for regime in GENERATED_REGIMES:
        regime_value = regimes.get(regime)
        roles = regime_value.get("roles") if isinstance(regime_value, Mapping) else None
        if not isinstance(roles, Mapping):
            raise ManifestError(f"prospective roles are absent: {regime}")
        for role in GENERATED_ROLES:
            role_value = roles.get(role)
            if not isinstance(role_value, Mapping):
                raise ManifestError(f"prospective role is absent: {role}/{regime}")
            records = [
                *(role_value.get("primary") or ()),
                *(role_value.get("replacements") or ()),
            ]
            identifiers = {
                str(record.get("episode_id"))
                for record in records
                if isinstance(record, Mapping)
            }
            if not identifiers or len(identifiers) != len(records):
                raise ManifestError(
                    f"prospective episode IDs are invalid: {role}/{regime}"
                )
            result[(role, regime)] = frozenset(identifiers)
    return result


def _is_generated_data_json(
    relative: str,
    episode_ids: Mapping[tuple[str, str], frozenset[str]] | None = None,
) -> bool:
    """Recognize only predeclared high-cardinality data JSON namespaces."""

    path = Path(relative)
    parts = path.parts
    if path.suffix.lower() != ".json" or not parts or parts[0] != "data":
        return False
    if parts == ("data", "replacement_registry.json"):
        return True
    if len(parts) == 3 and parts[:2] == ("data", "replacement_claims"):
        stem = Path(parts[2]).stem
        return (
            Path(parts[2]).suffix == ".json"
            and len(stem) == 6
            and all(character in "0123456789" for character in stem)
        )
    if len(parts) == 4 and parts[1] in GENERATED_ROLES:
        return (
            parts[2] in GENERATED_REGIMES
            and parts[3] in {"raw_manifest.json", "execution_manifest.json"}
        )
    if len(parts) == 5 and parts[1] in GENERATED_ROLES:
        return (
            parts[2] in GENERATED_REGIMES
            and parts[3] in {"raw", "execution"}
            and episode_ids is not None
            and Path(parts[4]).stem
            in episode_ids.get((parts[1], parts[2]), frozenset())
        )
    if len(parts) == 5 and parts[1] == "persistence_intents":
        return (
            parts[2] in GENERATED_ROLES
            and parts[3] in GENERATED_REGIMES
            and episode_ids is not None
            and Path(parts[4]).stem
            in episode_ids.get((parts[2], parts[3]), frozenset())
        )
    return False


def is_generated_attempt_product(
    relative: str,
    episode_ids: Mapping[tuple[str, str], frozenset[str]] | None = None,
) -> bool:
    """Return whether a candidate is an exact, predeclared later product."""

    return (
        relative in GENERATED_AUDIT_NAMES
        or relative in GENERATED_ATTEMPT_PRODUCTS
        or relative in GENERATED_EXECUTION_FAILURE_NAMES
        or _is_generated_data_json(relative, episode_ids)
    )


def classify_pre_data_source_paths() -> dict[str, list[str]]:
    """Classify the complete allowed-suffix closure, failing on local aliases."""

    attempt, linked_attempt = _walk_attempt_candidates()
    repository, linked_repository = _walk_study_root_candidates()
    _assert_discovered_integrity(
        attempt,
        repository,
        [
            *(f"attempt:{item}" for item in linked_attempt),
            *(f"study_root:{item}" for item in linked_repository),
        ],
    )
    required_attempt = PRE_DATA_STATIC_RELATIVE_PATHS | PRE_DATA_AUDIT_INPUTS
    episode_ids = _prospective_episode_ids()
    generated_attempt = {
        relative
        for relative in attempt
        if relative not in required_attempt
        and is_generated_attempt_product(relative, episode_ids)
    }
    unexpected_attempt = set(attempt) - required_attempt - generated_attempt
    expected_repository = set(PRE_DATA_REPOSITORY_RELATIVE_PATHS)
    mutable_repository = set(MUTABLE_STUDY_ROOT_RELATIVE_PATHS)
    generated_repository = set(repository) & mutable_repository
    unexpected_repository = (
        set(repository) - expected_repository - generated_repository
    )
    return {
        "required_attempt_present": sorted(set(attempt) & required_attempt),
        "required_repository_present": sorted(
            set(repository) & expected_repository
        ),
        "generated_attempt": sorted(generated_attempt),
        "generated_repository": sorted(generated_repository),
        "unexpected_attempt": sorted(unexpected_attempt),
        "unexpected_repository": sorted(unexpected_repository),
    }


def collect_pre_data_source_paths() -> list[Path]:
    """Return only the explicit normative allowlist, never later products."""

    relative_paths = PRE_DATA_STATIC_RELATIVE_PATHS | PRE_DATA_AUDIT_INPUTS
    attempt_paths = [
        ATTEMPT_ROOT / relative
        for relative in sorted(relative_paths)
        if (ATTEMPT_ROOT / relative).is_file()
    ]
    repository_paths = [
        REPO_ROOT / relative
        for relative in sorted(PRE_DATA_REPOSITORY_RELATIVE_PATHS)
        if (REPO_ROOT / relative).is_file()
    ]
    return attempt_paths + repository_paths


def missing_pre_data_source_paths() -> list[str]:
    required = PRE_DATA_STATIC_RELATIVE_PATHS | PRE_DATA_AUDIT_INPUTS
    missing_attempt = {
        relative
        for relative in required
        if not (ATTEMPT_ROOT / relative).is_file()
    }
    missing_repository = {
        relative
        for relative in PRE_DATA_REPOSITORY_RELATIVE_PATHS
        if not (REPO_ROOT / relative).is_file()
    }
    return sorted(missing_attempt | missing_repository)


def discovered_python_source_paths() -> set[str]:
    """Compatibility view of all discovered attempt-local Python sources."""

    attempt, linked = _walk_attempt_candidates()
    repository, linked_repository = _walk_study_root_candidates()
    _assert_discovered_integrity(
        attempt,
        repository,
        [
            *(f"attempt:{item}" for item in linked),
            *(f"study_root:{item}" for item in linked_repository),
        ],
    )
    return {
        relative
        for relative in attempt
        if Path(relative).suffix.lower() == ".py"
    }


def unexpected_python_source_paths() -> list[str]:
    classification = classify_pre_data_source_paths()
    return sorted(
        relative
        for relative in (
            classification["unexpected_attempt"]
            + classification["unexpected_repository"]
        )
        if Path(relative).suffix.lower() == ".py"
    )


def unexpected_pre_data_source_paths() -> list[str]:
    """Return every unclassified allowed-suffix attempt/root path."""

    classification = classify_pre_data_source_paths()
    return sorted(
        classification["unexpected_attempt"]
        + classification["unexpected_repository"]
    )


def source_closure_policy() -> dict[str, Any]:
    """Return the canonical prospective required/generated classification."""

    return {
        "allowed_suffixes": sorted(ALLOWED_SOURCE_SUFFIXES),
        "required_attempt_paths": sorted(
            PRE_DATA_STATIC_RELATIVE_PATHS | PRE_DATA_AUDIT_INPUTS
        ),
        "required_repository_paths": sorted(
            PRE_DATA_REPOSITORY_RELATIVE_PATHS
        ),
        "generated_exact_attempt_paths": sorted(
            GENERATED_AUDIT_NAMES
            | GENERATED_ATTEMPT_PRODUCTS
            | GENERATED_EXECUTION_FAILURE_NAMES
        ),
        "generated_data_json_shapes": [
            "data/replacement_registry.json",
            "data/replacement_claims/{claim_index:06d}.json",
            "data/{role}/{regime}/{raw_manifest.json|execution_manifest.json}",
            "data/{role}/{regime}/{raw|execution}/{episode_id}.json",
            "data/persistence_intents/{role}/{regime}/{episode_id}.json",
        ],
        "generated_roles": list(GENERATED_ROLES),
        "generated_regimes": list(GENERATED_REGIMES),
        "generated_episode_jsons_prospective_ledger_bound": True,
        "replacement_claim_index_exact_ascii_six_digit": True,
        "mutable_repository_paths": sorted(
            MUTABLE_STUDY_ROOT_RELATIVE_PATHS
        ),
        "attempt_discovery_recursive": True,
        "study_root_discovery_immediate_only": True,
        "directory_symlinks_forbidden": True,
        "file_symlinks_forbidden": True,
        "inode_aliases_forbidden": True,
        "classification_precedence": ["required", "generated", "unexpected"],
    }


def _manifest_for_paths(
    local_paths: Iterable[Path],
    *,
    label: str,
    include_inherited_runtime: bool,
) -> dict[str, Any]:
    local_records: dict[str, dict[str, Any]] = {}
    for path in sorted({Path(item) for item in local_paths}, key=lambda item: str(item)):
        relative = _relative_to_repo(path)
        if relative in local_records:
            raise ManifestError(f"duplicate normalized local path: {relative}")
        local_records[relative] = _record(path, local_source=True)
    if not local_records:
        raise ManifestError("manifest cannot be empty")
    _assert_unique_local_inodes(local_records)

    external_records: dict[str, dict[str, Any]] = {}
    inherited_aliases: list[dict[str, str]] = []
    if include_inherited_runtime:
        contract = load_v5_runtime_contract()
        for raw_path, expected in sorted(contract["external_file_hashes"].items()):
            path = Path(raw_path)
            record = _record(path, local_source=False)
            if record["sha256"] != expected:
                raise ManifestError(f"inherited external hash drift: {path}")
            external_records[str(path)] = record
        contract_record = _record(V5_RUNTIME_CONTRACT_PATH, local_source=False)
        if contract_record["sha256"] != V5_RUNTIME_CONTRACT_SHA256:
            raise ManifestError("V5 runtime-contract hash drift")
        external_records[str(V5_RUNTIME_CONTRACT_PATH)] = contract_record
        whitening_record = _record(V5_FIXED_WHITENING_PATH, local_source=False)
        if whitening_record["sha256"] != V5_FIXED_WHITENING_SHA256:
            raise ManifestError("V5 fixed-whitening hash drift")
        external_records[str(V5_FIXED_WHITENING_PATH)] = whitening_record
        # The exact inherited venv executables are two distinct symlinks to the
        # same Homebrew interpreter.  This is the sole accepted alias and is
        # explicitly represented rather than silently normalized away.
        inherited_aliases.append(
            {
                "left": str(EVALUATION_PYTHON),
                "right": str(GENERATION_PYTHON),
                "reason": "exact terminal-v004 interpreter paths share frozen target",
            }
        )

    return {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "science_attempt": SCIENCE_ATTEMPT,
        "label": str(label),
        "created_unix_ns": time.time_ns(),
        "local_files": dict(sorted(local_records.items())),
        "external_files": dict(sorted(external_records.items())),
        "local_file_count": len(local_records),
        "external_file_count": len(external_records),
        "local_total_bytes": sum(item["bytes"] for item in local_records.values()),
        "external_total_bytes": sum(
            item["bytes"] for item in external_records.values()
        ),
        "local_symlink_count": 0,
        "local_inode_alias_count": 0,
        "inherited_runtime_symlink_exceptions": sorted(INHERITED_RUNTIME_SYMLINKS),
        "inherited_runtime_aliases": inherited_aliases,
        "outcome_array_contents_opened": False,
        "hdf5_contents_opened": False,
        "v3_targets_opened": False,
        "all_hashes_verified": True,
    }


def build_pre_data_manifest() -> dict[str, Any]:
    missing = missing_pre_data_source_paths()
    unexpected = unexpected_pre_data_source_paths()
    unexpected_python = [
        path for path in unexpected if Path(path).suffix.lower() == ".py"
    ]
    if missing or unexpected:
        raise ManifestError(
            "explicit pre-data source/config set mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"unexpected_python={unexpected_python}"
        )
    manifest = _manifest_for_paths(
        collect_pre_data_source_paths(),
        label=PRE_DATA_LABEL,
        include_inherited_runtime=True,
    )
    manifest["source_closure_policy"] = source_closure_policy()
    return manifest


def build_addendum_manifest(label: str, paths: Iterable[Path]) -> dict[str, Any]:
    """Build an in-memory manifest for later fit/selection/freeze artifacts."""

    return _manifest_for_paths(
        paths,
        label=label,
        include_inherited_runtime=False,
    )


def verify_manifest(value_or_path: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(value_or_path, Mapping):
        manifest = dict(value_or_path)
        manifest_path = None
    else:
        manifest_path = Path(value_or_path)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ManifestError("manifest JSON is not an object")
        manifest = raw
    local = manifest.get("local_files")
    external = manifest.get("external_files")
    if not isinstance(local, Mapping) or not isinstance(external, Mapping):
        raise ManifestError("manifest file maps are missing")
    observed_local: dict[str, dict[str, Any]] = {}
    drifted: list[str] = []
    for relative, expected in sorted(local.items()):
        path = REPO_ROOT / str(relative)
        record = _record(path, local_source=True)
        observed_local[str(relative)] = record
        if record != expected:
            drifted.append(str(relative))
    _assert_unique_local_inodes(observed_local)
    for raw_path, expected in sorted(external.items()):
        record = _record(Path(str(raw_path)), local_source=False)
        if record != expected:
            drifted.append(str(raw_path))
    checks = {
        "schema": manifest.get("schema_version") == 1,
        "attempt": manifest.get("attempt") == ACTIVE_ATTEMPT,
        "science_attempt": manifest.get("science_attempt") == SCIENCE_ATTEMPT,
        "counts": (
            manifest.get("local_file_count") == len(local)
            and manifest.get("external_file_count") == len(external)
        ),
        "total_bytes": (
            manifest.get("local_total_bytes")
            == sum(int(item["bytes"]) for item in local.values())
            and manifest.get("external_total_bytes")
            == sum(int(item["bytes"]) for item in external.values())
        ),
        "no_local_symlinks": manifest.get("local_symlink_count") == 0,
        "no_local_aliases": manifest.get("local_inode_alias_count") == 0,
        "no_drift": not drifted,
        "no_arrays_decoded": manifest.get("outcome_array_contents_opened") is False,
    }
    if manifest.get("label") == PRE_DATA_LABEL:
        missing = missing_pre_data_source_paths()
        unexpected = unexpected_pre_data_source_paths()
        current = {
            _relative_to_repo(path) for path in collect_pre_data_source_paths()
        }
        checks["complete_current_pre_data_source_set"] = (
            not missing and not unexpected and set(local) == current
        )
        checks["source_closure_policy"] = (
            manifest.get("source_closure_policy") == source_closure_policy()
        )
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "drifted": drifted,
        "manifest_sha256": (
            sha256_file(manifest_path) if manifest_path is not None else None
        ),
    }
    if not result["passed"]:
        raise ManifestError(f"manifest verification failed: {result}")
    return result


def repository_hash_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return repository-relative hashes suitable for authorization seals."""

    result = {
        str(relative): str(record["sha256"])
        for relative, record in manifest["local_files"].items()
    }
    repository = REPO_ROOT.resolve(strict=True)
    for raw_path, record in manifest["external_files"].items():
        path = Path(raw_path).resolve(strict=True)
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError:
            continue
        result[relative] = str(record["sha256"])
    return dict(sorted(result.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-pre-data")
    build.add_argument("--output", type=Path, default=PRE_DATA_MANIFEST_PATH)
    verify = subparsers.add_parser("verify")
    verify.add_argument("path", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "verify":
        result = verify_manifest(arguments.path)
    else:
        result = build_pre_data_manifest()
        _atomic_json_exclusive(arguments.output, result)
        result = {
            "passed": True,
            "output": str(arguments.output),
            "sha256": sha256_file(arguments.output),
            "local_file_count": result["local_file_count"],
            "external_file_count": result["external_file_count"],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
