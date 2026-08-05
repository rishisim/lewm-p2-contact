#!/usr/bin/env python3
"""Sealed four-DGP analysis for the domain-robust adaptive gate study.

The module deliberately separates pure numerical functions from the filesystem
wrapper.  The pure functions are used by the independent verifier and by the
synthetic tests; :func:`main` is the only entry point that opens confirmation
arrays.  Confirmation arrays must already be covered by the immutable
``audit/confirmation_input_seal.json`` before ``main`` will open them.

Positive contrasts mean that the adaptive gate has lower latent-prediction MSE
than the named comparator.  Only ``*_vs_analytic`` for the two frozen
co-primary endpoints enter the eight-claim terminal family.  All controls and
heterogeneity intervals are descriptive and cannot change the terminal label.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from study_common import read_verified_controller


ATTEMPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ATTEMPT_ROOT.parents[3]
ACTIVE_ATTEMPT = ATTEMPT_ROOT.name
if ACTIVE_ATTEMPT != "v008":
    raise RuntimeError(f"analysis.py must run from v008, got {ACTIVE_ATTEMPT!r}")
V5_FIXED_WHITENING_RELATIVE = (
    "runs/lewm_v5_readiness_program/v5_package_versions/v004/freeze/whitening.npz"
)
V5_FIXED_WHITENING_SHA256 = (
    "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796"
)

DGP_ORDER = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ENDPOINTS = ("raw", "fixed_whitened")
COMPARATORS = (
    "analytic",
    "seeded_weakly_more_compute",
    "fixed_depth_1",
    "within_episode_histogram",
)
PRIMARY_CONTRASTS = tuple(
    f"{endpoint}_vs_analytic" for endpoint in ENDPOINTS
)
ALL_CONTRASTS = tuple(
    f"{endpoint}_vs_{comparator}"
    for endpoint in ENDPOINTS
    for comparator in COMPARATORS
)

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_CHUNK = 250
FAMILYWISE_ALPHA = 0.05
FAMILY_SIZE = 8
PER_CLAIM_ALPHA = 0.00625
INDIVIDUAL_INTERVAL_ALPHA = 0.05
QUANTILE_METHOD = "linear"

BASE_FLOPS = 70_529_190
MANDATORY_DEPTH1_FLOPS = 669_184
ADDITIONAL_REFINER_FLOPS = 264_960
ROWS_PER_EPISODE = 38
MAX_DEPTH = 4
MAX_GATE_EVALUATIONS = 3

TERMINAL_CONFIRMED = "domain_robust_gate_confirmed"
TERMINAL_PARTIAL = "domain_robust_gate_partial"
TERMINAL_FAILED = "domain_robust_gate_failed"
TERMINAL_INVALID = "domain_robust_gate_execution_invalid"

FORBIDDEN_ARRAY_TOKENS = (
    "contact",
    "privileged",
    "motion",
    "phase",
    "reward",
    "success",
)
EXECUTION_ARRAY_KEYS = frozenset(
    {
        "episode_slot",
        "model_step",
        "target",
        "exits",
        "selected",
        "calls",
        "scores",
        "production_features",
        "history",
        "action_history",
        "stage_current",
        "stage_update",
        "head_scores",
        "reached",
        "dense_scores",
        "dense_head_scores",
        "dense_stage_current",
        "dense_stage_update",
    }
)
ANALYSIS_ARRAY_KEYS = (
    "episode_slot",
    "model_step",
    "target",
    "exits",
    "selected",
    "calls",
    "scores",
    "production_features",
)


class AnalysisIntegrityFailure(RuntimeError):
    """A completed sealed analysis whose immutable integrity checks failed."""


@dataclasses.dataclass(frozen=True)
class ComputePricing:
    """Frozen counted-compute prices for one prediction row.

    ``gate_feature_flops`` and ``gate_head_flops`` are charged once for every
    reached gate evaluation.  Routing comparisons/indexing are recorded as
    non-FLOP operations and do not silently enter the FLOP denominator.
    """

    base_flops: int = BASE_FLOPS
    mandatory_depth1_flops: int = MANDATORY_DEPTH1_FLOPS
    additional_refiner_flops: int = ADDITIONAL_REFINER_FLOPS
    gate_feature_flops: int = 0
    gate_head_flops: int = 0
    gate_nonflop_operations: int = 0
    max_depth: int = MAX_DEPTH

    def __post_init__(self) -> None:
        integer_fields = (
            self.base_flops,
            self.mandatory_depth1_flops,
            self.additional_refiner_flops,
            self.gate_feature_flops,
            self.gate_head_flops,
            self.gate_nonflop_operations,
            self.max_depth,
        )
        if any(int(value) != value or value < 0 for value in integer_fields):
            raise ValueError("compute prices must be nonnegative integers")
        if self.additional_refiner_flops <= 0:
            raise ValueError("additional-refiner FLOPs must be positive")
        if self.max_depth < 1:
            raise ValueError("max depth must be positive")

    @property
    def gate_total_flops(self) -> int:
        return self.gate_feature_flops + self.gate_head_flops

    def fixed_depth_flops_per_row(self, depth: int) -> int:
        if not 1 <= depth <= self.max_depth:
            raise ValueError(f"depth outside 1..{self.max_depth}: {depth}")
        return (
            self.base_flops
            + self.mandatory_depth1_flops
            + (depth - 1) * self.additional_refiner_flops
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_json_authenticated(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f"authenticated JSON hash drift: {path}")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_path_text(path: Path) -> str:
    """Use repository-relative paths in study artifacts and absolute paths in tests."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _canonical_repo_relative(path: Path, repo_root: Path) -> str:
    repo = repo_root.resolve()
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(repo)
    except ValueError as exc:
        raise RuntimeError(f"sealed path is outside repository: {path}") from exc
    if not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"noncanonical sealed path: {path}")
    return relative.as_posix()


def _verify_sealed_file_map(
    raw_files: Any,
    *,
    repo_root: Path,
) -> dict[str, str]:
    """Rehash a complete seal map without decoding any NPZ contents."""

    if not isinstance(raw_files, Mapping) or not raw_files:
        raise RuntimeError("sealed_files must be a nonempty mapping")
    repo = repo_root.resolve()
    observed: dict[str, str] = {}
    identities: dict[tuple[int, int], str] = {}
    for raw_relative, raw_digest in sorted(raw_files.items()):
        relative = Path(str(raw_relative))
        digest = str(raw_digest)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != str(raw_relative)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"noncanonical sealed file record: {raw_relative}")
        path = repo / relative
        current = repo
        for part in relative.parts:
            current = current / part
            try:
                metadata = current.lstat()
            except FileNotFoundError as exc:
                raise RuntimeError(f"sealed file missing: {raw_relative}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"sealed symlink forbidden: {raw_relative}")
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"sealed path is not a regular file: {raw_relative}")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in identities:
            raise RuntimeError(
                f"sealed inode alias: {identities[identity]} and {raw_relative}"
            )
        identities[identity] = str(raw_relative)
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"sealed file hash drift: {raw_relative}")
        observed[str(raw_relative)] = actual
    return observed


def _verify_replacement_registry_closure(
    input_seal: Mapping[str, Any],
    input_files: Mapping[str, str],
    *,
    attempt_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Reject any post-input-seal append, removal, reorder, or head drift."""

    closure = input_seal.get("replacement_registry_closure")
    expected_keys = {
        "genesis",
        "claim_count",
        "prefix_count",
        "head_record_sha256",
        "postseal_claim_indexes",
        "references",
        "support_files",
    }
    if not isinstance(closure, Mapping) or set(closure) != expected_keys:
        raise RuntimeError("replacement registry closure schema drift")
    claim_count = closure.get("claim_count")
    prefix_count = closure.get("prefix_count")
    if (
        type(claim_count) is not int
        or type(prefix_count) is not int
        or not 0 <= int(prefix_count) <= int(claim_count)
        or closure.get("postseal_claim_indexes")
        != list(range(int(prefix_count), int(claim_count)))
    ):
        raise RuntimeError("replacement registry closure count/prefix drift")
    support = closure.get("support_files")
    if not isinstance(support, Mapping) or not support:
        raise RuntimeError("replacement registry closure lacks support files")
    normalized_support = {str(path): str(digest) for path, digest in support.items()}
    if any(input_files.get(path) != digest for path, digest in normalized_support.items()):
        raise RuntimeError("replacement support files are not frozen by the input seal")

    claims_root = attempt_root / "data/replacement_claims"
    claims_relative = _canonical_repo_relative(claims_root, repo_root)
    try:
        root_info = claims_root.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("replacement claims directory is absent") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise RuntimeError("replacement claims path is linked or non-directory")
    entries = sorted(claims_root.iterdir(), key=lambda item: item.name)
    expected_names = [f"{index:06d}.json" for index in range(int(claim_count))]
    if [entry.name for entry in entries] != expected_names:
        raise RuntimeError("live replacement claim directory differs from sealed prefix")
    expected_claim_paths = {
        f"{claims_relative}/{name}" for name in expected_names
    }
    sealed_claim_paths = {
        path
        for path in normalized_support
        if Path(path).parent.as_posix() == claims_relative
    }
    if sealed_claim_paths != expected_claim_paths:
        raise RuntimeError("input seal does not list the exact live claim prefix")

    genesis_path = attempt_root / "data/replacement_registry.json"
    genesis_relative = _canonical_repo_relative(genesis_path, repo_root)
    if normalized_support.get(genesis_relative) != input_files.get(genesis_relative):
        raise RuntimeError("replacement registry genesis is absent from final support")
    genesis_raw = genesis_path.read_bytes()
    genesis = json.loads(genesis_raw)
    if not isinstance(genesis, dict) or genesis_raw != (
        json.dumps(genesis, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"):
        raise RuntimeError("replacement registry genesis is noncanonical")
    if closure.get("genesis") != genesis:
        raise RuntimeError("replacement registry closure/genesis object drift")
    genesis_body = dict(genesis)
    previous = genesis_body.pop("record_sha256", None)
    genesis_hash = hashlib.sha256(
        json.dumps(
            genesis_body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if previous != genesis_hash:
        raise RuntimeError("replacement registry genesis record hash drift")
    for index, path in enumerate(entries):
        raw = path.read_bytes()
        claim = json.loads(raw)
        if not isinstance(claim, dict) or raw != (
            json.dumps(claim, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"):
            raise RuntimeError("replacement claim segment is noncanonical")
        body = dict(claim)
        observed_hash = body.pop("record_sha256", None)
        calculated_hash = hashlib.sha256(
            json.dumps(
                body, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        if not (
            claim.get("claim_index") == index
            and type(claim.get("claim_index")) is int
            and claim.get("prev_sha256") == previous
            and observed_hash == calculated_hash
        ):
            raise RuntimeError("replacement claim chain drift after input seal")
        previous = observed_hash
    if closure.get("head_record_sha256") != previous:
        raise RuntimeError("live replacement claim head differs from input seal")
    references = closure.get("references")
    if not isinstance(references, Mapping) or set(references) != {
        str(index) for index in range(int(prefix_count), int(claim_count))
    }:
        raise RuntimeError("replacement claim reference closure drift")
    return {
        "claim_count": int(claim_count),
        "prefix_count": int(prefix_count),
        "head_record_sha256": str(previous),
        "support_file_count": len(normalized_support),
        "live_directory_exact": True,
    }


def _seal_requires(
    files: Mapping[str, str],
    path: Path,
    *,
    repo_root: Path,
    expected_sha256: str | None = None,
) -> str:
    relative = _canonical_repo_relative(path, repo_root)
    if relative not in files:
        raise RuntimeError(f"required file is absent from seal: {relative}")
    if expected_sha256 is not None and files[relative] != expected_sha256:
        raise RuntimeError(f"required sealed hash mismatch: {relative}")
    return relative


def _manifest_declared_path(raw: Any, repo_root: Path) -> Path:
    candidate = Path(str(raw))
    return candidate if candidate.is_absolute() else repo_root / candidate


def _default_scientific_paths(
    *,
    attempt_root: Path,
    whitening_path: Path,
    pricing_path: Path,
    seed_path: Path,
    thresholds_path: Path,
    analysis_source_path: Path,
) -> tuple[Path, ...]:
    return (
        whitening_path,
        pricing_path,
        seed_path,
        thresholds_path,
        analysis_source_path,
        attempt_root / "DGP_MATRIX.json",
        attempt_root / "candidate_grid.json",
        attempt_root / "power_rule.json",
        attempt_root / "power_analysis.json",
        attempt_root / "outcome_mapping.json",
        attempt_root / "flops.py",
        attempt_root / "independent_verify.py",
        attempt_root / "verifier_contract.json",
    )


def validate_sealed_analysis_authorization(
    *,
    execution_root: Path,
    whitening_path: Path,
    pricing_path: Path,
    seed_path: Path,
    thresholds_path: Path,
    output_root: Path = ATTEMPT_ROOT,
    repo_root: Path = REPO_ROOT,
    pre_confirmation_seal_path: Path | None = None,
    analysis_source_path: Path | None = None,
    required_scientific_paths: Sequence[Path] | None = None,
    expected_whitening_relative: str | None = V5_FIXED_WHITENING_RELATIVE,
    expected_whitening_sha256: str = V5_FIXED_WHITENING_SHA256,
    require_state_binding: bool = True,
) -> dict[str, Any]:
    """Authenticate the complete analysis trust chain using bytes and JSON only.

    This function intentionally contains no ``np.load`` call.  It validates the
    pre-confirmation package seal, its exact extension by the confirmation-input
    seal, every confirmation execution manifest and part, and every scientific
    object used below before the caller may open an array.
    """

    attempt_root = output_root
    input_seal_path = attempt_root / "audit/confirmation_input_seal.json"
    pre_confirmation_path = pre_confirmation_seal_path or (
        attempt_root / "audit/pre_confirmation_package_seal.json"
    )
    source_path = analysis_source_path or Path(__file__)
    input_seal = read_json(input_seal_path)
    pre_confirmation = read_json(pre_confirmation_path)
    input_seal_sha256 = sha256_file(input_seal_path)
    if not (
        input_seal.get("schema_version") == 1
        and input_seal.get("attempt") == ACTIVE_ATTEMPT
        and input_seal.get("checkpoint_state") == "CONFIRMATION_INPUT_SEAL"
        and input_seal.get("passed") is True
        and input_seal.get("confirmation_arrays_opened_before_seal") is False
        and input_seal.get("contact_motion_phase_reward_success_opened") is False
    ):
        raise RuntimeError("confirmation input seal header or exclusion contract drift")
    if not (
        pre_confirmation.get("schema_version") == 1
        and pre_confirmation.get("attempt") == ACTIVE_ATTEMPT
        and pre_confirmation.get("checkpoint_state")
        == "PRE_CONFIRMATION_PACKAGE_SEAL"
        and pre_confirmation.get("passed") is True
    ):
        raise RuntimeError("pre-confirmation package seal header drift")
    if require_state_binding:
        state = read_verified_controller()
        checkpoint = state.get("last_verified_checkpoint")
        if not (
            state.get("active_attempt") == ACTIVE_ATTEMPT
            and state.get("current_state") == "SEALED_ANALYSIS"
            and isinstance(checkpoint, Mapping)
            and checkpoint.get("evidence_path")
            == _canonical_repo_relative(input_seal_path, repo_root)
            and checkpoint.get("evidence_sha256") == input_seal_sha256
        ):
            raise RuntimeError("controller state does not authenticate the input seal")
    pre_confirmation_sha256 = sha256_file(pre_confirmation_path)
    if input_seal.get("pre_confirmation_package_seal_sha256") != pre_confirmation_sha256:
        raise RuntimeError("confirmation input seal does not bind the pre-confirmation seal")
    input_files = _verify_sealed_file_map(
        input_seal.get("sealed_files"), repo_root=repo_root
    )
    replacement_closure = _verify_replacement_registry_closure(
        input_seal,
        input_files,
        attempt_root=attempt_root,
        repo_root=repo_root,
    )
    pre_files_raw = pre_confirmation.get("sealed_files")
    if not isinstance(pre_files_raw, Mapping) or not pre_files_raw:
        raise RuntimeError("pre-confirmation seal lacks sealed_files")
    pre_files = {str(path): str(digest) for path, digest in pre_files_raw.items()}
    if any(input_files.get(path) != digest for path, digest in pre_files.items()):
        raise RuntimeError("confirmation input seal does not preserve the frozen package")
    pre_confirmation_relative = _seal_requires(
        input_files,
        pre_confirmation_path,
        repo_root=repo_root,
        expected_sha256=pre_confirmation_sha256,
    )
    if input_seal.get("pre_confirmation_package_seal_path") not in (
        None,
        pre_confirmation_relative,
    ):
        raise RuntimeError("pre-confirmation seal path cross-link drift")

    scientific_paths = tuple(required_scientific_paths or _default_scientific_paths(
        attempt_root=attempt_root,
        whitening_path=whitening_path,
        pricing_path=pricing_path,
        seed_path=seed_path,
        thresholds_path=thresholds_path,
        analysis_source_path=source_path,
    ))
    scientific_hashes: dict[str, str] = {}
    for path in scientific_paths:
        relative = _seal_requires(pre_files, path, repo_root=repo_root)
        _seal_requires(
            input_files,
            path,
            repo_root=repo_root,
            expected_sha256=pre_files[relative],
        )
        scientific_hashes[relative] = pre_files[relative]
    whitening_relative = _seal_requires(
        pre_files,
        whitening_path,
        repo_root=repo_root,
        expected_sha256=expected_whitening_sha256,
    )
    if expected_whitening_relative is not None and whitening_relative != expected_whitening_relative:
        raise RuntimeError("fixed whitening is not the unchanged V5 v004 object")

    mapping_path = attempt_root / "outcome_mapping.json"
    mapping = read_json_authenticated(
        mapping_path,
        input_files[_canonical_repo_relative(mapping_path, repo_root)],
    )
    if not (
        mapping.get("family_size") == FAMILY_SIZE
        and mapping.get("familywise_alpha") == FAMILYWISE_ALPHA
        and mapping.get("per_claim_alpha") == PER_CLAIM_ALPHA
        and tuple(mapping.get("co_primary_endpoints", ())) == PRIMARY_CONTRASTS
        and tuple(mapping.get("regime_order", ())) == DGP_ORDER
        and mapping.get("bootstrap", {}).get("paired_replicates")
        == BOOTSTRAP_REPLICATES
        and mapping.get("bootstrap", {}).get("quantile_method") == "NumPy linear"
    ):
        raise RuntimeError("sealed outcome mapping drift")
    dgp_path = attempt_root / "DGP_MATRIX.json"
    dgp_matrix = read_json_authenticated(
        dgp_path, input_files[_canonical_repo_relative(dgp_path, repo_root)]
    )
    if tuple(dgp_matrix.get("regime_order", ())) != DGP_ORDER:
        raise RuntimeError("sealed DGP order drift")

    seed_ledger = read_json_authenticated(
        seed_path, input_files[_canonical_repo_relative(seed_path, repo_root)]
    )
    analysis_seeds = seed_ledger.get("analysis_seeds")
    rng_ids = seed_ledger.get("analysis_rng_ids")
    if not isinstance(analysis_seeds, Mapping) or not isinstance(rng_ids, Mapping):
        raise RuntimeError("sealed analysis RNG identifiers absent")
    bootstrap_contract = rng_ids.get("bootstrap", {})
    bootstrap_records = bootstrap_contract.get("rng_ids", [])
    if not (
        analysis_seeds.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES
        and bootstrap_contract.get("replicate_count") == BOOTSTRAP_REPLICATES
        and bootstrap_contract.get("chunk_size_defining_rng_consumption")
        == BOOTSTRAP_CHUNK
        and tuple(bootstrap_contract.get("regime_order", ())) == DGP_ORDER
        and isinstance(bootstrap_records, list)
        and len(bootstrap_records) == 1
        and int(analysis_seeds.get("joint_bootstrap_seed", -1))
        == int(bootstrap_records[0].get("rng_id", -2))
        and set(analysis_seeds.get("seeded_comparator_seeds", {})) == set(DGP_ORDER)
        and set(analysis_seeds.get("histogram_seeds", {})) == set(DGP_ORDER)
    ):
        raise RuntimeError("sealed bootstrap/comparator RNG contract drift")

    compiled_manifest = read_json_authenticated(
        pricing_path,
        input_files[_canonical_repo_relative(pricing_path, repo_root)],
    )
    compiled_path = _manifest_declared_path(
        compiled_manifest.get("compiled_gate_path"), repo_root
    )
    if _canonical_repo_relative(compiled_path, repo_root) != _canonical_repo_relative(
        thresholds_path, repo_root
    ):
        raise RuntimeError("compiled-gate manifest path drift")
    if not (
        compiled_manifest.get("passed") is True
        and compiled_manifest.get("compiled_gate_sha256")
        == input_files[_canonical_repo_relative(thresholds_path, repo_root)]
        and compiled_manifest.get("target_or_contact_arrays_opened") is False
        and compiled_manifest.get("prior_outcome_arrays_opened") is False
    ):
        raise RuntimeError("compiled-gate manifest integrity drift")
    gate_cost = compiled_manifest.get("metadata", {}).get("gate_cost", {})
    compute = compiled_manifest.get("complete_compute_derivation", {})
    if not (
        gate_cost.get("total_gate_flops") in (7_985, 20_537)
        and gate_cost.get("total_gate_flops")
        == gate_cost.get("feature_flops", -1) + gate_cost.get("all_head_flops", -2)
        and compute.get("base_flops_per_row") == BASE_FLOPS
        and compute.get("depth1_flops_per_row") == MANDATORY_DEPTH1_FLOPS
        and compute.get("later_adapter_flops_per_call")
        == ADDITIONAL_REFINER_FLOPS
    ):
        raise RuntimeError("compiled gate pricing/compute derivation drift")

    gate_freezes: list[tuple[str, dict[str, Any]]] = []
    attempt_relative = _canonical_repo_relative(attempt_root, repo_root)
    freeze_prefix = f"{attempt_relative}/freeze/"
    for relative in pre_files:
        if relative.startswith(freeze_prefix) and relative.endswith(".json"):
            value = read_json_authenticated(repo_root / relative, input_files[relative])
            if value.get("status") == "selected_gate_frozen_before_smoke_or_confirmation":
                gate_freezes.append((relative, value))
    if len(gate_freezes) != 1:
        raise RuntimeError("expected exactly one sealed selected-gate freeze")
    gate_freeze_relative, gate_freeze = gate_freezes[0]
    if not (
        gate_freeze.get("compiled_gate_sha256")
        == compiled_manifest.get("compiled_gate_sha256")
        and gate_freeze.get("selected_head_refit_after_selection") is False
        and gate_freeze.get("contact_or_privileged_gate_inputs") is False
        and gate_freeze.get("confirmation_episodes_at_freeze") == 0
    ):
        raise RuntimeError("selected-gate freeze cross-link drift")

    frozen_counts = pre_confirmation.get("confirmation_episode_count_per_regime")
    if (
        not isinstance(frozen_counts, Mapping)
        or set(frozen_counts) != set(DGP_ORDER)
        or len({int(value) for value in frozen_counts.values()}) != 1
    ):
        raise RuntimeError("pre-confirmation fixed cohort size drift")
    expected_per_dgp = int(next(iter(frozen_counts.values())))
    if input_seal.get("expected_per_dgp") != expected_per_dgp:
        raise RuntimeError("input seal confirmation size differs from frozen power result")
    input_regimes = input_seal.get("regimes")
    if not isinstance(input_regimes, Mapping) or set(input_regimes) != set(DGP_ORDER):
        raise RuntimeError("input-seal DGP order drift")
    declared_parts = input_seal.get("execution_part_hashes")
    declared_sidecars = input_seal.get("execution_sidecar_hashes", {})
    if not isinstance(declared_parts, Mapping) or not isinstance(declared_sidecars, Mapping):
        raise RuntimeError("input seal lacks execution part maps")
    expected_parts: dict[str, str] = {}
    expected_sidecars: dict[str, str] = {}
    for regime in DGP_ORDER:
        manifest_path = execution_root / regime / "execution_manifest.json"
        relative = _seal_requires(input_files, manifest_path, repo_root=repo_root)
        recorded = input_regimes[regime].get("execution_manifest")
        if not isinstance(recorded, Mapping) or recorded.get("path") != relative or recorded.get("sha256") != input_files[relative]:
            raise RuntimeError(f"input-seal execution manifest cross-link drift: {regime}")
        manifest = read_json_authenticated(manifest_path, input_files[relative])
        records = _manifest_records(manifest)
        if len(records) != expected_per_dgp:
            raise RuntimeError(f"execution manifest count drift: {regime}")
        for slot, record in enumerate(records):
            if int(record.get("slot", -1)) != slot:
                raise RuntimeError(f"execution manifest slot drift: {regime}")
            part_path = _manifest_declared_path(record.get("path"), repo_root)
            part_relative = _seal_requires(
                input_files,
                part_path,
                repo_root=repo_root,
                expected_sha256=str(record.get("sha256")),
            )
            expected_parts[part_relative] = str(record["sha256"])
            if "sidecar_path" in record or "metadata_path" in record:
                sidecar_raw = record.get("sidecar_path", record.get("metadata_path"))
                sidecar_digest = record.get(
                    "sidecar_sha256", record.get("metadata_sha256")
                )
                sidecar_path = _manifest_declared_path(sidecar_raw, repo_root)
                sidecar_relative = _seal_requires(
                    input_files,
                    sidecar_path,
                    repo_root=repo_root,
                    expected_sha256=str(sidecar_digest),
                )
                expected_sidecars[sidecar_relative] = str(sidecar_digest)
    if dict(declared_parts) != expected_parts or dict(declared_sidecars) != expected_sidecars:
        raise RuntimeError("input-seal execution part/sidecar map is not exact")
    for confirmation_root in (
        execution_root,
        attempt_root / "data/persistence_intents/confirmation",
    ):
        if not confirmation_root.exists():
            continue
        for path in confirmation_root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"unsealed confirmation symlink: {path}")
            if (
                path.is_file()
                and _canonical_repo_relative(path, repo_root) not in input_files
            ):
                raise RuntimeError(f"unlisted confirmation file: {path}")

    return {
        "confirmation_input_seal_path": _canonical_repo_relative(
            input_seal_path, repo_root
        ),
        "confirmation_input_seal_sha256": input_seal_sha256,
        "pre_confirmation_package_seal_path": pre_confirmation_relative,
        "pre_confirmation_package_seal_sha256": pre_confirmation_sha256,
        "sealed_file_count": len(input_files),
        "pre_confirmation_file_count": len(pre_files),
        "scientific_object_hashes": dict(sorted(scientific_hashes.items())),
        "gate_freeze_path": gate_freeze_relative,
        "gate_freeze_sha256": input_files[gate_freeze_relative],
        "fixed_whitening_path": whitening_relative,
        "fixed_whitening_sha256": input_files[whitening_relative],
        "fixed_whitening_matches_known_v5_v004_content_hash": True,
        "bootstrap_seed": int(analysis_seeds["joint_bootstrap_seed"]),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "expected_confirmation_episodes_per_dgp": expected_per_dgp,
        "execution_manifest_and_part_hashes_transitively_verified": True,
        "replacement_registry_closure": replacement_closure,
        "no_symlinks_or_inode_aliases": True,
        "npz_arrays_opened_during_authorization": False,
        # Private runtime-only bindings are removed before JSON persistence.
        "_sealed_files": input_files,
        "_analysis_seeds": dict(analysis_seeds),
    }


def mark_confirmation_outcomes_opened(
    *,
    repo_root: Path = REPO_ROOT,
    attempt_root: Path = ATTEMPT_ROOT,
) -> None:
    """Use the sole state controller to mark analysis-open before first np.load."""

    state = read_verified_controller()
    if state.get("current_state") != "SEALED_ANALYSIS":
        raise RuntimeError("confirmation analysis may open only in SEALED_ANALYSIS")
    if state.get("confirmation_outcomes_opened_for_analysis") is not True:
        program = attempt_root / "version_forward_transaction.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(program),
                "counts",
                "confirmation_outcomes_opened_for_analysis=true",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "state controller could not mark confirmation analysis open: "
                + completed.stderr.strip()
            )
        state = read_verified_controller()
    if state.get("confirmation_outcomes_opened_for_analysis") is not True:
        raise RuntimeError("confirmation-open state marker did not persist")


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes, independent of NPZ metadata."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temporary(
    temporary: Path, destination: Path, *, exclusive: bool
) -> None:
    """Publish a complete inode, using link(2) for true no-replace."""

    if exclusive:
        os.link(temporary, destination, follow_symlinks=False)
    else:
        os.replace(temporary, destination)
    _fsync_parent_directory(destination)


def atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(temporary, path, exclusive=exclusive)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray], *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(temporary, path, exclusive=exclusive)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(path)


def _verify_exact_npz(
    path: Path, expected_arrays: Mapping[str, np.ndarray]
) -> str:
    """Verify an immutable derived NPZ against freshly recomputed arrays."""

    candidate = Path(path)
    metadata = candidate.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"derived NPZ is linked or nonregular: {candidate}")
    expected = {str(name): np.asarray(value) for name, value in expected_arrays.items()}
    with candidate.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        observed_sha256 = digest.hexdigest()
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as stored:
            if set(stored.files) != set(expected):
                raise RuntimeError(f"derived NPZ array schema drift: {candidate}")
            for name, recomputed in expected.items():
                observed = stored[name]
                if (
                    observed.dtype != recomputed.dtype
                    or observed.shape != recomputed.shape
                    or array_sha256(observed) != array_sha256(recomputed)
                ):
                    raise RuntimeError(
                        f"derived NPZ differs from exact recomputation: {candidate}/{name}"
                    )
    if sha256_file(candidate) != observed_sha256:
        raise RuntimeError(f"derived NPZ changed during adoption: {candidate}")
    return observed_sha256


def persist_or_verify_npz(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> str:
    """Create an immutable NPZ or adopt it only after exact recomputation."""

    candidate = Path(path)
    try:
        candidate.lstat()
    except FileNotFoundError:
        atomic_npz(candidate, arrays, exclusive=True)
    return _verify_exact_npz(candidate, arrays)


def _reject_nonfinite_json_constant(token: str) -> Any:
    raise RuntimeError(f"nonfinite JSON constant in derived artifact: {token}")


def _json_semantic_value(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        separators=(",", ":"),
        sort_keys=True,
    )
    decoded = json.loads(encoded, parse_constant=_reject_nonfinite_json_constant)
    if not isinstance(decoded, dict):
        raise RuntimeError("derived JSON semantic value is not an object")
    return decoded


def _verify_exact_json(path: Path, expected: Mapping[str, Any]) -> str:
    """Verify immutable derived JSON by exact decoded semantic equality."""

    candidate = Path(path)
    metadata = candidate.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"derived JSON is linked or nonregular: {candidate}")
    payload = candidate.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        observed = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"derived JSON is not strict UTF-8 JSON: {candidate}") from exc
    if observed != _json_semantic_value(expected):
        raise RuntimeError(
            f"derived JSON differs from exact semantic recomputation: {candidate}"
        )
    if sha256_file(candidate) != observed_sha256:
        raise RuntimeError(f"derived JSON changed during adoption: {candidate}")
    return observed_sha256


def persist_or_verify_json(path: Path, value: Mapping[str, Any]) -> str:
    """Create immutable JSON or adopt it only after semantic recomputation."""

    candidate = Path(path)
    try:
        candidate.lstat()
    except FileNotFoundError:
        atomic_json(candidate, value, exclusive=True)
    return _verify_exact_json(candidate, value)


def validate_no_forbidden_names(names: Iterable[str]) -> None:
    lowered = tuple(str(name).lower() for name in names)
    forbidden = sorted(
        name
        for name in lowered
        if any(token in name for token in FORBIDDEN_ARRAY_TOKENS)
    )
    if forbidden:
        raise RuntimeError(f"forbidden outcome/posthoc array fields present: {forbidden}")


def validate_execution_array_keys(names: Iterable[str]) -> None:
    observed = frozenset(str(name) for name in names)
    validate_no_forbidden_names(observed)
    missing = EXECUTION_ARRAY_KEYS - observed
    extra = observed - EXECUTION_ARRAY_KEYS
    if missing or extra:
        raise RuntimeError(
            f"confirmation execution array schema drift; missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )


def adaptive_compute_account(
    calls: np.ndarray,
    pricing: ComputePricing,
    *,
    observed_gate_evaluations: int | None = None,
) -> dict[str, Any]:
    """Return exact counted compute for the adaptive policy."""

    call_vector = np.asarray(calls)
    if call_vector.ndim != 1 or call_vector.size == 0:
        raise ValueError("calls must be a nonempty vector")
    if not np.issubdtype(call_vector.dtype, np.integer):
        if not np.equal(call_vector, np.floor(call_vector)).all():
            raise ValueError("calls must be integer-valued")
        call_vector = call_vector.astype(np.int64)
    call_vector = call_vector.astype(np.int64, copy=False)
    if not np.isin(call_vector, np.arange(1, pricing.max_depth + 1)).all():
        raise ValueError("calls outside frozen depth range")
    rows = int(call_vector.size)
    refiner_calls = int(call_vector.sum(dtype=np.int64))
    derived_gate_evaluations = int(
        np.minimum(call_vector, pricing.max_depth - 1).sum(dtype=np.int64)
    )
    if observed_gate_evaluations is not None and int(observed_gate_evaluations) != derived_gate_evaluations:
        raise RuntimeError("observed gate-evaluation count differs from call-derived count")
    gate_evaluations = derived_gate_evaluations
    base = rows * pricing.base_flops
    mandatory = rows * pricing.mandatory_depth1_flops
    additional = (refiner_calls - rows) * pricing.additional_refiner_flops
    feature = gate_evaluations * pricing.gate_feature_flops
    head = gate_evaluations * pricing.gate_head_flops
    total = base + mandatory + additional + feature + head
    equivalent_total_refiner_calls = rows + (
        additional + feature + head
    ) / pricing.additional_refiner_flops
    return {
        "rows": rows,
        "base_model_calls": rows,
        "refiner_model_calls": refiner_calls,
        "mean_refiner_calls": float(refiner_calls / rows),
        "gate_evaluations": gate_evaluations,
        "base_flops": int(base),
        "mandatory_depth1_flops": int(mandatory),
        "additional_refiner_flops": int(additional),
        "gate_feature_flops": int(feature),
        "gate_head_flops": int(head),
        "gate_total_flops": int(feature + head),
        "gate_nonflop_operations": int(
            gate_evaluations * pricing.gate_nonflop_operations
        ),
        "adaptive_total_counted_flops": int(total),
        "analytic_equivalent_total_refiner_calls": float(
            equivalent_total_refiner_calls
        ),
        "analytic_equivalent_mean_depth": float(
            equivalent_total_refiner_calls / rows
        ),
        "pricing": dataclasses.asdict(pricing) | {
            "gate_total_flops_per_evaluation": pricing.gate_total_flops
        },
        "call_histogram": np.bincount(
            call_vector, minlength=pricing.max_depth + 1
        )[1:].astype(int).tolist(),
    }


def strongest_transition_independent_allocation(
    fixed_depth_losses: np.ndarray,
    adaptive_total_counted_flops: int,
    pricing: ComputePricing,
) -> dict[str, Any]:
    """Find the strongest fixed-depth mixture at exactly adaptive FLOPs.

    The allocation is transition independent: a single probability vector over
    depths is applied to every row.  With a simplex constraint plus one compute
    equality, an optimum has support on at most two depths, so enumerating all
    depth pairs solves the finite linear program exactly (up to IEEE arithmetic).
    """

    losses = np.asarray(fixed_depth_losses, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] == 0:
        raise ValueError("fixed-depth losses must have shape [rows, depths]")
    rows, depths = losses.shape
    if depths != pricing.max_depth:
        raise ValueError("loss depth count differs from frozen maximum depth")
    if not np.isfinite(losses).all():
        raise ValueError("fixed-depth losses contain nonfinite values")
    budget = int(adaptive_total_counted_flops)
    per_depth = tuple(
        pricing.fixed_depth_flops_per_row(depth) for depth in range(1, depths + 1)
    )
    if budget < rows * per_depth[0] or budget > rows * per_depth[-1]:
        raise RuntimeError(
            "adaptive counted-compute budget is outside the analytic fixed-depth envelope"
        )
    means = losses.mean(axis=0, dtype=np.float64)
    candidates: list[tuple[float, int, int, Fraction, np.ndarray]] = []
    for lower_index in range(depths):
        for upper_index in range(lower_index, depths):
            lower_cost = per_depth[lower_index]
            upper_cost = per_depth[upper_index]
            if lower_index == upper_index:
                if budget != rows * lower_cost:
                    continue
                weight_fraction = Fraction(0, 1)
            else:
                numerator = budget - rows * lower_cost
                denominator = rows * (upper_cost - lower_cost)
                if numerator < 0 or numerator > denominator:
                    continue
                weight_fraction = Fraction(numerator, denominator)
            weight_upper = float(weight_fraction)
            expected_loss = losses[:, lower_index] + weight_upper * (
                losses[:, upper_index] - losses[:, lower_index]
            )
            objective = float(
                means[lower_index]
                + weight_upper * (means[upper_index] - means[lower_index])
            )
            candidates.append(
                (
                    objective,
                    lower_index + 1,
                    upper_index + 1,
                    weight_fraction,
                    expected_loss,
                )
            )
    if not candidates:
        raise RuntimeError("no exact-compute transition-independent allocation exists")
    objective, lower, upper, weight_fraction, expected_loss = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    weight_upper = float(weight_fraction)
    weights = np.zeros(depths, dtype=np.float64)
    if lower == upper:
        weights[lower - 1] = 1.0
    else:
        weights[lower - 1] = 1.0 - weight_upper
        weights[upper - 1] = weight_upper
    allocation_numerator = rows * (
        per_depth[lower - 1] * weight_fraction.denominator
        + weight_fraction.numerator
        * (per_depth[upper - 1] - per_depth[lower - 1])
    )
    allocation_denominator = weight_fraction.denominator
    if allocation_numerator != budget * allocation_denominator:
        raise RuntimeError("analytic allocation failed exact-compute equality")
    return {
        "loss": expected_loss,
        "mean_loss": objective,
        "depth_lower": int(lower),
        "depth_upper": int(upper),
        "weight_upper": float(weight_upper),
        "weight_upper_numerator": int(weight_fraction.numerator),
        "weight_upper_denominator": int(weight_fraction.denominator),
        "depth_probabilities": weights,
        "adaptive_total_counted_flops": budget,
        "allocation_total_counted_flops": budget,
        "allocation_compute_numerator": int(allocation_numerator),
        "allocation_compute_denominator": int(allocation_denominator),
        "integer_cross_multiplication_equality": bool(
            allocation_numerator == budget * allocation_denominator
        ),
        "exact_total_compute_match": True,
        "transition_independent": True,
        "optimization": "enumerated_two_support_solution_of_finite_linear_program",
    }


def seeded_weakly_more_compute_control(
    fixed_depth_losses: np.ndarray,
    allocation: Mapping[str, Any],
    adaptive_total_counted_flops: int,
    pricing: ComputePricing,
    seed: int,
) -> dict[str, Any]:
    """Realize the analytic mixture with a seeded, weakly-more-compute draw."""

    losses = np.asarray(fixed_depth_losses, dtype=np.float64)
    rows, depths = losses.shape
    if depths != pricing.max_depth:
        raise ValueError("loss depth count differs from pricing")
    lower = int(allocation["depth_lower"])
    upper = int(allocation["depth_upper"])
    lower_total = rows * pricing.fixed_depth_flops_per_row(lower)
    increment = pricing.fixed_depth_flops_per_row(upper) - pricing.fixed_depth_flops_per_row(lower)
    if lower == upper:
        if lower_total < adaptive_total_counted_flops:
            raise RuntimeError("degenerate allocation cannot weakly cover adaptive compute")
        number_upper = 0
    else:
        number_upper = int(
            math.ceil((adaptive_total_counted_flops - lower_total) / increment)
        )
        number_upper = min(rows, max(0, number_upper))
    calls = np.full(rows, lower, dtype=np.int64)
    if number_upper:
        rng = np.random.default_rng(int(seed))
        calls[rng.permutation(rows)[:number_upper]] = upper
    selected = losses[np.arange(rows), calls - 1]
    total_flops = int(
        sum(pricing.fixed_depth_flops_per_row(int(depth)) for depth in calls)
    )
    if total_flops < adaptive_total_counted_flops:
        raise RuntimeError("seeded comparator did not use weakly more compute")
    return {
        "loss": selected,
        "calls": calls,
        "seed": int(seed),
        "depth_lower": lower,
        "depth_upper": upper,
        "number_upper": number_upper,
        "total_counted_flops": total_flops,
        "adaptive_total_counted_flops": int(adaptive_total_counted_flops),
        "baseline_minus_adaptive_flops": int(
            total_flops - adaptive_total_counted_flops
        ),
        "weakly_more_compute": True,
        "transition_independent": True,
    }


def within_episode_histogram_calls(
    calls: np.ndarray,
    episode_slots: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Randomize routing within episode while exactly preserving each histogram."""

    call_vector = np.asarray(calls, dtype=np.int64)
    slots = np.asarray(episode_slots, dtype=np.int64)
    if call_vector.ndim != 1 or slots.shape != call_vector.shape:
        raise ValueError("calls and episode slots must be aligned vectors")
    randomized = np.empty_like(call_vector)
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    for slot in np.unique(slots):
        indices = np.flatnonzero(slots == slot)
        randomized[indices] = call_vector[indices][rng.permutation(len(indices))]
        width = int(max(call_vector.max(initial=1), randomized.max(initial=1))) + 1
        original_histogram = np.bincount(call_vector[indices], minlength=width)[1:]
        randomized_histogram = np.bincount(randomized[indices], minlength=width)[1:]
        records.append(
            {
                "episode_slot": int(slot),
                "rows": int(len(indices)),
                "original": original_histogram.astype(int).tolist(),
                "randomized": randomized_histogram.astype(int).tolist(),
                "preserved": bool(
                    np.array_equal(original_histogram, randomized_histogram)
                ),
            }
        )
    if not all(record["preserved"] for record in records):
        raise RuntimeError("within-episode histogram randomization drift")
    return randomized, records


def fixed_whitened_mse(difference: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    delta = np.asarray(difference, dtype=np.float64)
    matrix = np.asarray(whitening, dtype=np.float64)
    if delta.ndim < 2 or matrix.ndim != 2 or delta.shape[-1] != matrix.shape[0]:
        raise ValueError("difference/whitening shape mismatch")
    # Freeze the literal contraction order so the standalone verifier can
    # reproduce every episode metric and bootstrap value elementwise.
    transformed = np.einsum("...d,df->...f", delta, matrix, optimize=False)
    return np.mean(np.square(transformed), axis=-1, dtype=np.float64)


def endpoint_losses(
    target: np.ndarray,
    exits: np.ndarray,
    selected: np.ndarray,
    whitening: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    targets = np.asarray(target, dtype=np.float64)
    dense = np.asarray(exits, dtype=np.float64)
    sparse = np.asarray(selected, dtype=np.float64)
    if targets.ndim != 2 or dense.ndim != 3 or sparse.shape != targets.shape:
        raise ValueError("target/exits/selected shape mismatch")
    if dense.shape[0] != targets.shape[0] or dense.shape[2] != targets.shape[1]:
        raise ValueError("target/exits shape mismatch")
    if not all(np.isfinite(value).all() for value in (targets, dense, sparse)):
        raise ValueError("nonfinite prediction endpoint input")
    dense_difference = dense - targets[:, None, :]
    adaptive_difference = sparse - targets
    fixed = {
        "raw": np.mean(np.square(dense_difference), axis=-1, dtype=np.float64),
        "fixed_whitened": fixed_whitened_mse(dense_difference, whitening),
    }
    adaptive = {
        "raw": np.mean(np.square(adaptive_difference), axis=-1, dtype=np.float64),
        "fixed_whitened": fixed_whitened_mse(adaptive_difference, whitening),
    }
    return fixed, adaptive


def episode_means(
    values: np.ndarray,
    episode_slots: np.ndarray,
    *,
    expected_rows_per_episode: int | None = ROWS_PER_EPISODE,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    slots = np.asarray(episode_slots, dtype=np.int64)
    if vector.ndim != 1 or slots.shape != vector.shape:
        raise ValueError("episode aggregation requires aligned vectors")
    unique = np.unique(slots)
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise RuntimeError("episode slots must be contiguous and zero based")
    output = np.empty(len(unique), dtype=np.float64)
    for index, slot in enumerate(unique):
        selected = vector[slots == slot]
        if expected_rows_per_episode is not None and len(selected) != expected_rows_per_episode:
            raise RuntimeError(
                f"episode {slot} has {len(selected)} rows, expected {expected_rows_per_episode}"
            )
        output[index] = selected.sum(dtype=np.float64) / len(selected)
    return output


def _average_ranks(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    order = np.argsort(vector, kind="mergesort")
    ranks = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and vector[order[stop]] == vector[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denominator) if denominator else float("nan")


def stagewise_rank_and_calibration(
    scores: np.ndarray,
    calls: np.ndarray,
    fixed_losses: Mapping[str, np.ndarray],
    *,
    bins: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    score_matrix = np.asarray(scores, dtype=np.float64)
    call_vector = np.asarray(calls, dtype=np.int64)
    if score_matrix.ndim != 2 or score_matrix.shape[0] != len(call_vector):
        raise ValueError("scores/calls shape mismatch")
    if score_matrix.shape[1] != MAX_GATE_EVALUATIONS:
        raise ValueError("scores must contain exactly three stage columns")
    rank_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    integrity: dict[str, bool] = {}
    for stage_index in range(MAX_GATE_EVALUATIONS):
        expected_reached = call_vector >= stage_index + 1
        finite = np.isfinite(score_matrix[:, stage_index])
        integrity[f"stage_{stage_index + 1}_score_reached_mask"] = bool(
            np.array_equal(finite, expected_reached)
        )
        reached_indices = np.flatnonzero(finite)
        endpoint_gains: dict[str, np.ndarray] = {}
        rank_record: dict[str, Any] = {
            "stage": stage_index + 1,
            "reached_rows": int(len(reached_indices)),
            "continued_rows": int(
                np.sum(call_vector[reached_indices] > stage_index + 1)
            ),
            "continuation_rate_among_reached": float(
                np.mean(call_vector[reached_indices] > stage_index + 1)
            )
            if len(reached_indices)
            else None,
            "endpoints": {},
        }
        for endpoint in ENDPOINTS:
            losses = np.asarray(fixed_losses[endpoint], dtype=np.float64)
            gain = losses[:, stage_index] - losses[:, stage_index + 1]
            endpoint_gains[endpoint] = gain
            rho = spearman_rho(
                score_matrix[reached_indices, stage_index], gain[reached_indices]
            )
            rho_is_finite = bool(np.isfinite(rho))
            rank_record["endpoints"][endpoint] = {
                "spearman_score_next_stage_gain_rho": (
                    float(rho) if rho_is_finite else None
                ),
                "rank_sign": (
                    "positive"
                    if rho_is_finite and rho > 0
                    else "negative"
                    if rho_is_finite and rho < 0
                    else "zero_or_undefined"
                ),
                "positive_sign": bool(rho_is_finite and rho > 0),
                "mean_next_stage_gain": float(gain[reached_indices].mean())
                if len(reached_indices)
                else None,
            }
        rank_rows.append(rank_record)
        order = reached_indices[
            np.argsort(score_matrix[reached_indices, stage_index], kind="mergesort")
        ]
        bin_records: list[dict[str, Any]] = []
        for bin_index, indices in enumerate(np.array_split(order, bins)):
            if not len(indices):
                continue
            record: dict[str, Any] = {
                "bin": bin_index + 1,
                "rows": int(len(indices)),
                "score_mean": float(score_matrix[indices, stage_index].mean()),
                "continuation_rate": float(
                    np.mean(call_vector[indices] > stage_index + 1)
                ),
            }
            for endpoint in ENDPOINTS:
                record[f"{endpoint}_next_stage_gain_mean"] = float(
                    endpoint_gains[endpoint][indices].mean()
                )
                record[f"{endpoint}_score_minus_gain_mean"] = float(
                    (
                        score_matrix[indices, stage_index]
                        - endpoint_gains[endpoint][indices]
                    ).mean()
                )
            bin_records.append(record)
        calibration_rows.append(
            {
                "stage": stage_index + 1,
                "reached_rows": int(len(reached_indices)),
                "equal_count_score_bins": bin_records,
                "descriptive_only": True,
            }
        )
    return rank_rows, calibration_rows, integrity


def sequential_calls_from_scores(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    threshold = np.asarray(thresholds, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != MAX_GATE_EVALUATIONS:
        raise ValueError("score matrix shape drift")
    if threshold.shape != (MAX_GATE_EVALUATIONS,):
        raise ValueError("threshold shape drift")
    calls = np.ones(len(matrix), dtype=np.int64)
    active = np.ones(len(matrix), dtype=bool)
    for stage in range(MAX_GATE_EVALUATIONS):
        reached = active.copy()
        if not np.isfinite(matrix[reached, stage]).all():
            raise RuntimeError("nonfinite reached gate score")
        if not np.isnan(matrix[~reached, stage]).all():
            raise RuntimeError("unreached gate score must be NaN")
        # The frozen gate contract uses a strict continuation inequality.  An
        # exact threshold tie therefore stops at the current depth.
        active = reached & (matrix[:, stage] > threshold[stage])
        calls[active] += 1
    return calls


def _strict_int64_vector(name: str, value: np.ndarray) -> np.ndarray:
    """Accept only losslessly representable integer vectors before casting."""

    array = np.asarray(value)
    if array.ndim != 1:
        raise RuntimeError(f"{name} must be a one-dimensional integer array")
    if array.dtype.kind not in ("i", "u"):
        raise RuntimeError(f"{name} must have an integer dtype")
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max())
        limits = np.iinfo(np.int64)
        if minimum < limits.min or maximum > limits.max:
            raise RuntimeError(f"{name} is not losslessly representable as int64")
    converted = array.astype(np.int64, copy=False)
    if not np.array_equal(converted.astype(array.dtype, copy=False), array):
        raise RuntimeError(f"{name} changed during int64 conversion")
    return converted


def feature_and_gain_shift(
    current_features: np.ndarray,
    current_scores: np.ndarray,
    current_calls: np.ndarray,
    current_losses: Mapping[str, np.ndarray],
    reference_features: np.ndarray,
    reference_scores: np.ndarray,
    reference_calls: np.ndarray,
    reference_losses: Mapping[str, np.ndarray],
    *,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe causal production-feature, score, gain, and depth shifts."""

    features = np.asarray(current_features, dtype=np.float64)
    scores = np.asarray(current_scores, dtype=np.float64)
    calls = np.asarray(current_calls, dtype=np.int64)
    ref_features = np.asarray(reference_features, dtype=np.float64)
    ref_scores = np.asarray(reference_scores, dtype=np.float64)
    ref_calls = np.asarray(reference_calls, dtype=np.int64)
    if features.ndim != 3 or ref_features.ndim != 3:
        raise ValueError("production features must have [rows, stages, features]")
    if features.shape[1:] != ref_features.shape[1:]:
        raise ValueError("current/reference production feature shape drift")
    names = list(feature_names or [f"feature_{index}" for index in range(features.shape[2])])
    if len(names) != features.shape[2]:
        raise ValueError("feature-name count drift")
    validate_no_forbidden_names(names)
    current_hist = np.bincount(calls, minlength=MAX_DEPTH + 1)[1:].astype(np.float64)
    reference_hist = np.bincount(ref_calls, minlength=MAX_DEPTH + 1)[1:].astype(np.float64)
    current_probability = current_hist / current_hist.sum()
    reference_probability = reference_hist / reference_hist.sum()
    stages: list[dict[str, Any]] = []
    for stage in range(MAX_GATE_EVALUATIONS):
        current_reached = np.isfinite(scores[:, stage])
        reference_reached = np.isfinite(ref_scores[:, stage])
        current_stage_features = features[current_reached, stage]
        reference_stage_features = ref_features[reference_reached, stage]
        both_feature_samples_available = bool(
            len(current_stage_features) and len(reference_stage_features)
        )
        standardized: np.ndarray | None = None
        if both_feature_samples_available:
            current_mean = current_stage_features.mean(axis=0, dtype=np.float64)
            reference_mean = reference_stage_features.mean(
                axis=0, dtype=np.float64
            )
            pooled_scale = np.sqrt(
                0.5
                * (
                    current_stage_features.var(axis=0, dtype=np.float64)
                    + reference_stage_features.var(axis=0, dtype=np.float64)
                )
                + 1e-12
            )
            standardized = (current_mean - reference_mean) / pooled_scale
            top = np.argsort(np.abs(standardized), kind="stable")[-10:][::-1]
        else:
            top = np.empty(0, dtype=np.int64)
        gain_shift: dict[str, Any] = {}
        for endpoint in ENDPOINTS:
            gain = current_losses[endpoint][:, stage] - current_losses[endpoint][:, stage + 1]
            reference_gain = (
                reference_losses[endpoint][:, stage]
                - reference_losses[endpoint][:, stage + 1]
            )
            current_gain_mean = (
                float(gain[current_reached].mean())
                if current_reached.any()
                else None
            )
            reference_gain_mean = (
                float(reference_gain[reference_reached].mean())
                if reference_reached.any()
                else None
            )
            gain_shift[endpoint] = {
                "mean": current_gain_mean,
                "reference_mean": reference_gain_mean,
                "mean_shift": (
                    current_gain_mean - reference_gain_mean
                    if current_gain_mean is not None
                    and reference_gain_mean is not None
                    else None
                ),
                "available": current_gain_mean is not None
                and reference_gain_mean is not None,
            }
        current_stage_scores = scores[current_reached, stage]
        reference_stage_scores = ref_scores[reference_reached, stage]
        current_score_available = bool(len(current_stage_scores))
        reference_score_available = bool(len(reference_stage_scores))
        both_score_samples_available = (
            current_score_available and reference_score_available
        )
        stages.append(
            {
                "stage": stage + 1,
                "reached_rows": int(current_reached.sum()),
                "reference_reached_rows": int(reference_reached.sum()),
                "score": {
                    "mean": (
                        float(current_stage_scores.mean())
                        if current_score_available
                        else None
                    ),
                    "reference_mean": (
                        float(reference_stage_scores.mean())
                        if reference_score_available
                        else None
                    ),
                    "mean_shift": (
                        float(
                            current_stage_scores.mean()
                            - reference_stage_scores.mean()
                        )
                        if both_score_samples_available
                        else None
                    ),
                    "quantiles": (
                        np.quantile(
                            current_stage_scores,
                            [0.05, 0.25, 0.5, 0.75, 0.95],
                            method=QUANTILE_METHOD,
                        ).tolist()
                        if current_score_available
                        else None
                    ),
                    "reference_quantiles": (
                        np.quantile(
                            reference_stage_scores,
                            [0.05, 0.25, 0.5, 0.75, 0.95],
                            method=QUANTILE_METHOD,
                        ).tolist()
                        if reference_score_available
                        else None
                    ),
                    "shift_available": both_score_samples_available,
                },
                "features": {
                    "rms_standardized_mean_shift": (
                        float(np.sqrt(np.mean(np.square(standardized))))
                        if standardized is not None
                        else None
                    ),
                    "maximum_absolute_standardized_mean_shift": (
                        float(np.max(np.abs(standardized)))
                        if standardized is not None
                        else None
                    ),
                    "top_absolute_standardized_mean_shifts": [
                        {
                            "index": int(index),
                            "name": names[int(index)],
                            "standardized_mean_shift": float(
                                standardized[int(index)]
                            ),
                        }
                        for index in top
                    ],
                    "shift_available": both_feature_samples_available,
                },
                "solver_next_stage_gain": gain_shift,
            }
        )
    return {
        "call_depth": {
            "histogram": current_hist.astype(int).tolist(),
            "probability": current_probability.tolist(),
            "mean_calls": float(calls.mean()),
            "reference_histogram": reference_hist.astype(int).tolist(),
            "reference_probability": reference_probability.tolist(),
            "reference_mean_calls": float(ref_calls.mean()),
            "mean_call_shift": float(calls.mean() - ref_calls.mean()),
            "total_variation": float(
                0.5 * np.abs(current_probability - reference_probability).sum()
            ),
        },
        "stages": stages,
        "causal_production_features_only": True,
        "contact_motion_phase_reward_success_excluded": True,
    }


def analyze_regime(
    arrays: Mapping[str, np.ndarray],
    whitening: np.ndarray,
    pricing: ComputePricing,
    *,
    seeded_seeds: Mapping[str, int],
    histogram_seed: int,
    thresholds: Sequence[float] | None = None,
    expected_rows_per_episode: int | None = ROWS_PER_EPISODE,
    expected_model_step_start: int = 3,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute all frozen endpoint, comparator, diagnostic, and compute objects."""

    required = frozenset(ANALYSIS_ARRAY_KEYS)
    observed = frozenset(arrays)
    validate_no_forbidden_names(observed)
    if not required.issubset(observed):
        raise RuntimeError(f"analysis input missing fields: {sorted(required - observed)}")
    slots = _strict_int64_vector("episode_slot", arrays["episode_slot"])
    steps = _strict_int64_vector("model_step", arrays["model_step"])
    calls = _strict_int64_vector("calls", arrays["calls"])
    scores = np.asarray(arrays["scores"], dtype=np.float64)
    features = np.asarray(arrays["production_features"], dtype=np.float64)
    rows = len(calls)
    if not all(len(np.asarray(arrays[name])) == rows for name in required):
        raise RuntimeError("analysis arrays have inconsistent row counts")
    if scores.shape != (rows, MAX_GATE_EVALUATIONS):
        raise RuntimeError("gate-score shape drift")
    if features.ndim != 3 or features.shape[:2] != scores.shape:
        raise RuntimeError("production-feature shape drift")
    if not np.isin(calls, np.arange(1, MAX_DEPTH + 1)).all():
        raise RuntimeError("calls are outside the frozen range")
    unique_slots = np.unique(slots)
    if not np.array_equal(unique_slots, np.arange(len(unique_slots))):
        raise RuntimeError("episode slots are not contiguous and zero based")
    if expected_rows_per_episode is not None:
        expected_steps = np.tile(
            np.arange(
                expected_model_step_start,
                expected_model_step_start + expected_rows_per_episode,
                dtype=np.int64,
            ),
            len(unique_slots),
        )
        if not np.array_equal(steps, expected_steps):
            raise RuntimeError("model-step order drift")
    fixed_losses, adaptive_losses = endpoint_losses(
        arrays["target"], arrays["exits"], arrays["selected"], whitening
    )
    selected_from_exits = np.asarray(arrays["exits"])[
        np.arange(rows), calls - 1
    ]
    selected_equivalence = bool(
        np.allclose(
            selected_from_exits,
            np.asarray(arrays["selected"]),
            rtol=1e-6,
            atol=1e-7,
        )
    )
    observed_gate_evaluations = int(np.isfinite(scores).sum())
    expected_reached = np.column_stack(
        [calls >= stage + 1 for stage in range(MAX_GATE_EVALUATIONS)]
    )
    if not np.isfinite(scores[expected_reached]).all():
        raise RuntimeError("reached gate scores must be finite")
    if not np.isnan(scores[~expected_reached]).all():
        raise RuntimeError("unreached gate scores must be exactly NaN")
    if not np.isfinite(features[expected_reached]).all():
        raise RuntimeError("reached production-feature rows must be finite")
    if not np.isnan(features[~expected_reached]).all():
        raise RuntimeError("unreached production-feature rows must be entirely NaN")
    compute = adaptive_compute_account(
        calls, pricing, observed_gate_evaluations=observed_gate_evaluations
    )
    histogram_calls, histogram_records = within_episode_histogram_calls(
        calls, slots, histogram_seed
    )
    positions = np.arange(rows)
    contrasts: dict[str, np.ndarray] = {}
    analytic_allocations: dict[str, dict[str, Any]] = {}
    seeded_controls: dict[str, dict[str, Any]] = {}
    control_compute: dict[str, Any] = {
        "fixed_depth_1_total_counted_flops": int(
            rows * pricing.fixed_depth_flops_per_row(1)
        ),
        "within_episode_histogram_total_counted_flops": int(
            compute["adaptive_total_counted_flops"]
        ),
        "within_episode_histogram_exact_adaptive_compute_match": True,
    }
    for endpoint in ENDPOINTS:
        allocation = strongest_transition_independent_allocation(
            fixed_losses[endpoint],
            compute["adaptive_total_counted_flops"],
            pricing,
        )
        seeded = seeded_weakly_more_compute_control(
            fixed_losses[endpoint],
            allocation,
            compute["adaptive_total_counted_flops"],
            pricing,
            int(seeded_seeds[endpoint]),
        )
        histogram_loss = fixed_losses[endpoint][positions, histogram_calls - 1]
        row_contrasts = {
            f"{endpoint}_vs_analytic": allocation["loss"] - adaptive_losses[endpoint],
            f"{endpoint}_vs_seeded_weakly_more_compute": seeded["loss"]
            - adaptive_losses[endpoint],
            f"{endpoint}_vs_fixed_depth_1": fixed_losses[endpoint][:, 0]
            - adaptive_losses[endpoint],
            f"{endpoint}_vs_within_episode_histogram": histogram_loss
            - adaptive_losses[endpoint],
        }
        for name, vector in row_contrasts.items():
            contrasts[name] = episode_means(
                vector,
                slots,
                expected_rows_per_episode=expected_rows_per_episode,
            )
        analytic_allocations[endpoint] = {
            key: value
            for key, value in allocation.items()
            if key not in ("loss", "depth_probabilities")
        } | {"depth_probabilities": allocation["depth_probabilities"].tolist()}
        seeded_controls[endpoint] = {
            key: value for key, value in seeded.items() if key not in ("loss", "calls")
        } | {
            "call_histogram": np.bincount(
                seeded["calls"], minlength=pricing.max_depth + 1
            )[1:].astype(int).tolist()
        }
        control_compute[f"seeded_{endpoint}"] = seeded_controls[endpoint]
    stagewise, calibration, score_integrity = stagewise_rank_and_calibration(
        scores, calls, fixed_losses
    )
    reconstructed = (
        sequential_calls_from_scores(scores, thresholds)
        if thresholds is not None
        else None
    )
    integrity = {
        "all_inputs_finite_except_unreached_scores_and_features": bool(
            np.isfinite(np.asarray(arrays["target"])).all()
            and np.isfinite(np.asarray(arrays["exits"])).all()
            and np.isfinite(np.asarray(arrays["selected"])).all()
        ),
        "calls_in_frozen_range": True,
        "selected_matches_dense_exit": selected_equivalence,
        "gate_evaluation_count_matches_calls": observed_gate_evaluations
        == int(np.minimum(calls, MAX_GATE_EVALUATIONS).sum()),
        "production_features_finite_exactly_when_stage_reached": True,
        "unreached_scores_exactly_nan": True,
        "within_episode_histograms_preserved": all(
            record["preserved"] for record in histogram_records
        ),
        "analytic_exact_total_compute": all(
            record["exact_total_compute_match"]
            for record in analytic_allocations.values()
        ),
        "seeded_controls_weakly_more_compute": all(
            record["weakly_more_compute"] for record in seeded_controls.values()
        ),
        "calls_reproduced_from_frozen_scores_and_thresholds": bool(
            np.array_equal(reconstructed, calls)
        )
        if reconstructed is not None
        else True,
        **score_integrity,
    }
    episode_metrics = {
        "episode_slot": np.arange(len(unique_slots), dtype=np.int32),
        "adaptive_raw": episode_means(
            adaptive_losses["raw"], slots, expected_rows_per_episode=expected_rows_per_episode
        ),
        "adaptive_fixed_whitened": episode_means(
            adaptive_losses["fixed_whitened"],
            slots,
            expected_rows_per_episode=expected_rows_per_episode,
        ),
        **contrasts,
    }
    summary = {
        "episode_count": int(len(unique_slots)),
        "row_count": rows,
        "adaptive_mse": {
            endpoint: float(adaptive_losses[endpoint].mean())
            for endpoint in ENDPOINTS
        },
        "contrast_point_estimates": {
            name: float(value.mean()) for name, value in contrasts.items()
        },
        "analytic_allocations": analytic_allocations,
        "controls": {
            "seeded_weakly_more_compute": seeded_controls,
            "fixed_depth_1": {"depth": 1},
            "within_episode_histogram": {
                "seed": int(histogram_seed),
                "episodes": histogram_records,
            },
        },
        "stagewise_gate_score_next_stage_gain_rank": stagewise,
        "routing_calibration": calibration,
        "compute": compute | {"controls": control_compute},
        "integrity": integrity,
        "process_valid_prebootstrap": all(integrity.values()),
        "contact_motion_phase_reward_success_opened": False,
    }
    diagnostic_arrays = {
        "production_features": features,
        "scores": scores,
        "calls": calls,
        **{f"{endpoint}_fixed_losses": fixed_losses[endpoint] for endpoint in ENDPOINTS},
    }
    return summary, episode_metrics, diagnostic_arrays


def bootstrap_all(
    episode_metrics: Mapping[str, Mapping[str, np.ndarray]],
    seed: int,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = BOOTSTRAP_CHUNK,
    dgp_order: Sequence[str] = DGP_ORDER,
) -> dict[str, dict[str, np.ndarray]]:
    """Joint fixed-order episode bootstrap with a frozen RNG-consumption rule."""

    if replicates <= 0 or chunk_size <= 0:
        raise ValueError("bootstrap counts must be positive")
    if tuple(dgp_order) != tuple(episode_metrics):
        raise RuntimeError("bootstrap DGP order differs from sealed order")
    output: dict[str, dict[str, np.ndarray]] = {}
    for regime in dgp_order:
        names = [name for name in ALL_CONTRASTS if name in episode_metrics[regime]]
        if tuple(names) != ALL_CONTRASTS:
            raise RuntimeError(f"bootstrap contrast schema drift for {regime}")
        episode_count = len(np.asarray(episode_metrics[regime][names[0]]))
        if any(len(np.asarray(episode_metrics[regime][name])) != episode_count for name in names):
            raise RuntimeError("episode metric length drift")
        output[regime] = {
            name: np.empty(replicates, dtype=np.float64) for name in names
        }
    rng = np.random.default_rng(int(seed))
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        size = stop - start
        for regime in dgp_order:
            first = next(iter(output[regime]))
            episode_count = len(episode_metrics[regime][first])
            sampled = rng.integers(
                0,
                episode_count,
                size=(size, episode_count),
                dtype=np.int32,
            )
            for name in output[regime]:
                values = np.asarray(episode_metrics[regime][name], dtype=np.float64)
                output[regime][name][start:stop] = values[sampled].sum(
                    axis=1, dtype=np.float64
                ) / episode_count
    return output


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method=QUANTILE_METHOD))


def summarize_bootstrap(
    episode_metrics: Mapping[str, Mapping[str, np.ndarray]],
    bootstrap: Mapping[str, Mapping[str, np.ndarray]],
    *,
    dgp_order: Sequence[str] = DGP_ORDER,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    individual: dict[str, Any] = {}
    simultaneous: dict[str, Any] = {}
    for regime in dgp_order:
        individual[regime] = {}
        simultaneous[regime] = {}
        for name in ALL_CONTRASTS:
            values = np.asarray(episode_metrics[regime][name], dtype=np.float64)
            replicates = np.asarray(bootstrap[regime][name], dtype=np.float64)
            individual[regime][name] = {
                "estimate": float(values.mean()),
                "lower": quantile(replicates, INDIVIDUAL_INTERVAL_ALPHA / 2),
                "upper": quantile(replicates, 1 - INDIVIDUAL_INTERVAL_ALPHA / 2),
                "confidence": 0.95,
                "sidedness": "two_sided",
                "method": "episode_bootstrap_percentile_numpy_linear_quantile",
                "terminal": name in PRIMARY_CONTRASTS,
            }
        for endpoint, name in zip(ENDPOINTS, PRIMARY_CONTRASTS, strict=True):
            replicates = np.asarray(bootstrap[regime][name], dtype=np.float64)
            lower = quantile(replicates, PER_CLAIM_ALPHA)
            simultaneous[regime][endpoint] = {
                "contrast": name,
                "estimate": float(np.mean(episode_metrics[regime][name])),
                "lower": lower,
                "supported": bool(lower > 0.0),
                "familywise_alpha": FAMILYWISE_ALPHA,
                "family_size": FAMILY_SIZE,
                "per_claim_one_sided_alpha": PER_CLAIM_ALPHA,
                "method": "bonferroni_one_sided_episode_bootstrap_percentile_numpy_linear_quantile",
            }
    heterogeneity: dict[str, Any] = {"pairwise": {}, "range": {}}
    for endpoint, contrast in zip(ENDPOINTS, PRIMARY_CONTRASTS, strict=True):
        estimates = {
            regime: float(np.mean(episode_metrics[regime][contrast]))
            for regime in dgp_order
        }
        heterogeneity["range"][endpoint] = {
            "minimum_regime": min(estimates, key=estimates.get),
            "maximum_regime": max(estimates, key=estimates.get),
            "max_minus_min": max(estimates.values()) - min(estimates.values()),
            "descriptive_not_terminal": True,
        }
        for left_index, left in enumerate(dgp_order):
            for right in dgp_order[left_index + 1 :]:
                difference = bootstrap[left][contrast] - bootstrap[right][contrast]
                key = f"{left}_minus_{right}"
                heterogeneity["pairwise"].setdefault(key, {})[endpoint] = {
                    "estimate": estimates[left] - estimates[right],
                    "lower": quantile(difference, 0.025),
                    "upper": quantile(difference, 0.975),
                    "descriptive_not_terminal": True,
                }
    return individual, simultaneous, heterogeneity


def terminal_mapping(
    simultaneous: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    integrity_valid: bool,
    dgp_order: Sequence[str] = DGP_ORDER,
) -> tuple[str, int]:
    """Apply the immutable 8/1--7/0 mapping with invalidity precedence."""

    supported = sum(
        bool(simultaneous[regime][endpoint]["supported"])
        for regime in dgp_order
        for endpoint in ENDPOINTS
    )
    if not integrity_valid:
        return TERMINAL_INVALID, supported
    if supported == FAMILY_SIZE:
        return TERMINAL_CONFIRMED, supported
    if supported:
        return TERMINAL_PARTIAL, supported
    return TERMINAL_FAILED, supported


def persist_bootstrap_replicates(
    path: Path,
    bootstrap: Mapping[str, Mapping[str, np.ndarray]],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = BOOTSTRAP_CHUNK,
) -> dict[str, Any]:
    arrays = {
        f"{regime}__{contrast}": np.asarray(values, dtype=np.float64)
        for regime in DGP_ORDER
        for contrast, values in bootstrap[regime].items()
    }
    if any(value.shape != (replicates,) for value in arrays.values()):
        raise RuntimeError("bootstrap replicate array length drift")
    hashes = {
        name: {
            "sha256": array_sha256(value),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
        for name, value in arrays.items()
    }
    artifact_sha256 = persist_or_verify_npz(path, arrays)
    return {
        "path": artifact_path_text(path),
        "sha256": artifact_sha256,
        "array_hash_definition": "sha256(dtype.str + NUL + compact_shape_json + NUL + C_order_bytes)",
        "arrays": hashes,
        "array_count": len(arrays),
        "replicates_per_array": replicates,
        "seed": int(seed),
        "chunk_size_defining_rng_consumption": int(chunk_size),
        "dgp_order_defining_rng_consumption": list(DGP_ORDER),
    }


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _manifest_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise RuntimeError("execution manifest lacks episode records")
    return records


def load_execution_manifest_arrays(
    path: Path,
    *,
    authorized_files: Mapping[str, str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one sealed confirmation manifest, rejecting any noncausal field."""

    manifest_relative = _canonical_repo_relative(path, repo_root)
    if authorized_files is None or manifest_relative not in authorized_files:
        raise RuntimeError("execution manifest is not analysis-authorized")
    manifest = read_json_authenticated(path, authorized_files[manifest_relative])
    records = _manifest_records(manifest)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in ANALYSIS_ARRAY_KEYS}
    episode_ids: list[str] = []
    for expected_slot, record in enumerate(records):
        if int(record["slot"]) != expected_slot:
            raise RuntimeError("execution manifest slot order drift")
        candidate = _manifest_declared_path(record["path"], repo_root)
        relative = _canonical_repo_relative(candidate, repo_root)
        if (
            relative not in authorized_files
            or authorized_files[relative] != record["sha256"]
        ):
            raise RuntimeError(f"execution part is not sealed: {candidate}")
        # Hash and decode the same open file description, closing the usual
        # path-based time-of-check/time-of-use gap.
        with candidate.open("rb") as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != authorized_files[relative]:
                raise RuntimeError(f"execution part hash drift: {candidate}")
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as stored:
                validate_execution_array_keys(stored.files)
                for name in ANALYSIS_ARRAY_KEYS:
                    parts[name].append(stored[name].copy())
        episode_ids.append(str(record["episode_id"]))
    arrays = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    expected_slots = np.repeat(
        np.arange(len(records), dtype=np.int64),
        np.bincount(arrays["episode_slot"].astype(np.int64), minlength=len(records)),
    )
    if not np.array_equal(arrays["episode_slot"].astype(np.int64), expected_slots):
        raise RuntimeError("execution episode-slot ordering drift")
    return arrays, {
        "manifest_path": manifest_relative,
        "manifest_sha256": authorized_files[manifest_relative],
        "episode_ids": episode_ids,
    }


def _load_npz_array_authenticated(
    path: Path,
    key: str,
    expected_sha256: str,
) -> np.ndarray:
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"authenticated NPZ hash drift: {path}")
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as stored:
            if key not in stored.files:
                raise RuntimeError(f"authenticated NPZ key absent: {key}")
            return stored[key].copy()


def _load_whitening(
    path: Path,
    expected_sha256: str,
    key: str = "whitening_matrix",
) -> np.ndarray:
    matrix = _load_npz_array_authenticated(path, key, expected_sha256).astype(
        np.float64
    )
    if matrix.shape != (192, 192) or not np.isfinite(matrix).all():
        raise RuntimeError("fixed V5 whitening matrix invalid")
    return matrix


def pricing_from_json(
    path: Path, *, expected_sha256: str | None = None
) -> ComputePricing:
    root = (
        read_json_authenticated(path, expected_sha256)
        if expected_sha256 is not None
        else read_json(path)
    )
    value: Mapping[str, Any] = root
    # The gate compiler records executable-graph pricing here.  Keep support
    # for a flattened verifier contract as well, but never infer a head count
    # from an architecture label.
    metadata = root.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("gate_cost"), Mapping):
        gate_cost = metadata["gate_cost"]
        derivation = root.get("complete_compute_derivation", {})
        value = {
            "base_flops": derivation.get("base_flops_per_row"),
            "mandatory_depth1_flops": derivation.get("depth1_flops_per_row"),
            "additional_refiner_flops": derivation.get("later_adapter_flops_per_call"),
            "gate_feature_flops": gate_cost.get("feature_flops"),
            "gate_head_flops": gate_cost.get("all_head_flops"),
            "gate_nonflop_operations": gate_cost.get("nonflop_operations"),
        }
    for key in ("compute_pricing", "pricing", "selected_gate_compute"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            value = nested
            break
    aliases = {
        "base_flops": ("base_flops",),
        "mandatory_depth1_flops": ("mandatory_depth1_flops", "v1_flops"),
        "additional_refiner_flops": ("additional_refiner_flops", "adapter_flops"),
        "gate_feature_flops": ("gate_feature_flops", "feature_flops_per_evaluation"),
        "gate_head_flops": ("gate_head_flops", "head_flops_per_evaluation"),
        "gate_nonflop_operations": ("gate_nonflop_operations", "nonflop_operations_per_evaluation"),
    }
    fields: dict[str, int] = {}
    for field, names in aliases.items():
        for name in names:
            if name in value and value[name] is not None:
                fields[field] = int(value[name])
                break
        if field not in fields:
            raise RuntimeError(f"frozen compute price absent: {field}")
    pricing = ComputePricing(**fields)
    if (
        pricing.base_flops != BASE_FLOPS
        or pricing.mandatory_depth1_flops != MANDATORY_DEPTH1_FLOPS
        or pricing.additional_refiner_flops != ADDITIONAL_REFINER_FLOPS
    ):
        raise RuntimeError("frozen base/refiner compute prices drift from V5")
    return pricing


def _seed_map(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[int, dict[str, dict[str, int]], dict[str, int]]:
    value = (
        read_json_authenticated(path, expected_sha256)
        if expected_sha256 is not None
        else read_json(path)
    )
    analysis = value.get("analysis_seeds", value)
    bootstrap_seed = int(analysis["joint_bootstrap_seed"])
    seeded_raw = analysis["seeded_comparator_seeds"]
    histogram_raw = analysis["histogram_seeds"]
    seeded: dict[str, dict[str, int]] = {}
    histogram: dict[str, int] = {}
    for regime in DGP_ORDER:
        item = seeded_raw[regime]
        seeded[regime] = {
            "raw": int(item.get("raw", item.get("raw_vs_analytic"))),
            "fixed_whitened": int(
                item.get("fixed_whitened", item.get("fixed_whitened_vs_analytic"))
            ),
        }
        histogram[regime] = int(histogram_raw[regime])
    return bootstrap_seed, seeded, histogram


def run_sealed_analysis(
    *,
    execution_root: Path,
    whitening_path: Path,
    pricing_path: Path,
    seed_path: Path,
    thresholds_path: Path | None = None,
    output_root: Path = ATTEMPT_ROOT,
    mark_opened: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Open sealed confirmation arrays once and materialize immutable analysis."""

    output_path = output_root / "analysis_result.json"
    if output_path.exists():
        return read_json(output_path)
    if thresholds_path is None:
        raise RuntimeError("the sealed compiled gate is required for analysis")
    authorization = validate_sealed_analysis_authorization(
        execution_root=execution_root,
        whitening_path=whitening_path,
        pricing_path=pricing_path,
        seed_path=seed_path,
        thresholds_path=thresholds_path,
        output_root=output_root,
    )
    sealed_files = authorization.pop("_sealed_files")
    authorization.pop("_analysis_seeds")
    # No NPZ has been opened above this line. Marking through the sole state
    # controller is the last prerequisite before any confirmation/scientific
    # array is decoded.
    (mark_opened or (
        lambda: mark_confirmation_outcomes_opened(attempt_root=output_root)
    ))()
    whitening_relative = _canonical_repo_relative(whitening_path, REPO_ROOT)
    pricing_relative = _canonical_repo_relative(pricing_path, REPO_ROOT)
    seed_relative = _canonical_repo_relative(seed_path, REPO_ROOT)
    thresholds_relative = _canonical_repo_relative(thresholds_path, REPO_ROOT)
    whitening = _load_whitening(
        whitening_path, sealed_files[whitening_relative]
    )
    pricing = pricing_from_json(
        pricing_path, expected_sha256=sealed_files[pricing_relative]
    )
    bootstrap_seed, seeded_seeds, histogram_seeds = _seed_map(
        seed_path, expected_sha256=sealed_files[seed_relative]
    )
    thresholds = _load_npz_array_authenticated(
        thresholds_path,
        "thresholds",
        sealed_files[thresholds_relative],
    ).astype(np.float64)
    if thresholds.shape != (MAX_GATE_EVALUATIONS,) or not np.isfinite(thresholds).all():
        raise RuntimeError("compiled gate threshold array drift")
    summaries: dict[str, Any] = {}
    episode_metrics: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    input_records: dict[str, Any] = {}
    for regime in DGP_ORDER:
        manifest_path = execution_root / regime / "execution_manifest.json"
        arrays, record = load_execution_manifest_arrays(
            manifest_path, authorized_files=sealed_files
        )
        summary, metrics, diagnostic = analyze_regime(
            arrays,
            whitening,
            pricing,
            seeded_seeds=seeded_seeds[regime],
            histogram_seed=histogram_seeds[regime],
            thresholds=thresholds,
        )
        metric_path = output_root / f"metrics/{regime}_episode_metrics.npz"
        metric_sha256 = persist_or_verify_npz(metric_path, metrics)
        summary["episode_metrics"] = {
            "path": str(metric_path.resolve().relative_to(REPO_ROOT.resolve())),
            "sha256": metric_sha256,
            "arrays": {name: array_sha256(value) for name, value in metrics.items()},
        }
        summaries[regime] = summary
        episode_metrics[regime] = metrics
        diagnostics[regime] = diagnostic
        input_records[regime] = record
    native = diagnostics[DGP_ORDER[0]]
    native_losses = {
        endpoint: native[f"{endpoint}_fixed_losses"] for endpoint in ENDPOINTS
    }
    for regime in DGP_ORDER:
        current = diagnostics[regime]
        summaries[regime]["feature_score_gain_depth_shift_vs_native_confirmation"] = feature_and_gain_shift(
            current["production_features"],
            current["scores"],
            current["calls"],
            {endpoint: current[f"{endpoint}_fixed_losses"] for endpoint in ENDPOINTS},
            native["production_features"],
            native["scores"],
            native["calls"],
            native_losses,
        )
    bootstrap = bootstrap_all(episode_metrics, bootstrap_seed)
    bootstrap_path = output_root / "metrics/bootstrap_replicates.npz"
    bootstrap_manifest = persist_bootstrap_replicates(
        bootstrap_path, bootstrap, seed=bootstrap_seed
    )
    individual, simultaneous, heterogeneity = summarize_bootstrap(
        episode_metrics, bootstrap
    )
    global_integrity = {
        "confirmation_input_seal_passed": True,
        "transitive_analysis_authorization_passed": authorization[
            "execution_manifest_and_part_hashes_transitively_verified"
        ],
        "all_four_dgps_present_in_fixed_order": tuple(summaries) == DGP_ORDER,
        "all_regime_integrity": all(
            all(summary["integrity"].values()) for summary in summaries.values()
        ),
        "bootstrap_replicates_exactly_20000": all(
            len(array) == BOOTSTRAP_REPLICATES
            for regime in bootstrap.values()
            for array in regime.values()
        ),
        "bootstrap_family_exactly_eight_coprimary_claims": sum(
            len(items) for items in simultaneous.values()
        )
        == FAMILY_SIZE,
        "unchanged_v5_fixed_whitening_used": authorization[
            "fixed_whitening_matches_known_v5_v004_content_hash"
        ],
        "contact_motion_phase_reward_success_excluded": True,
        "latency_excluded_from_terminal_mapping": True,
    }
    process_valid = all(global_integrity.values())
    terminal_label, supported = terminal_mapping(
        simultaneous, integrity_valid=process_valid
    )
    compute_ledger = {
        "schema_version": 1,
        "pricing_source": {
            "path": pricing_relative,
            "sha256": sealed_files[pricing_relative],
        },
        "pricing": dataclasses.asdict(pricing),
        "regimes": {regime: summaries[regime]["compute"] for regime in DGP_ORDER},
        "exact_compute_used_for_terminal_analytic_comparators": True,
    }
    compute_path = output_root / "metrics/compute_ledger.json"
    compute_sha256 = persist_or_verify_json(compute_path, compute_ledger)
    bootstrap_summary = {
        "schema_version": 1,
        "individual": individual,
        "simultaneous_co_primary": simultaneous,
        "heterogeneity": heterogeneity,
        "bootstrap_artifact": bootstrap_manifest,
        "bootstrap": {
            "unit": "episode",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": bootstrap_seed,
            "chunk_size_defining_rng_consumption": BOOTSTRAP_CHUNK,
            "dgp_order_defining_rng_consumption": list(DGP_ORDER),
            "individual_interval": "two_sided_95_percent_percentile",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "family_size": FAMILY_SIZE,
            "per_claim_one_sided_alpha": PER_CLAIM_ALPHA,
            "quantile_method": QUANTILE_METHOD,
        },
    }
    bootstrap_summary_path = output_root / "metrics/bootstrap_summary.json"
    persist_or_verify_json(bootstrap_summary_path, bootstrap_summary)
    result = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "SEALED_ANALYSIS",
        "created_unix_ns": time.time_ns(),
        # ``passed`` means the sealed analysis checkpoint completed and may be
        # advanced to independent verification.  Scientific/process validity
        # is a separate immutable field so an execution-invalid result cannot
        # strand the state machine or be mistaken for a repairable exception.
        "passed": True,
        "integrity_passed": process_valid,
        "process_valid": process_valid,
        "proposed_terminal_label": terminal_label,
        "supported_co_primary_claim_count": supported,
        "required_co_primary_claim_count": FAMILY_SIZE,
        "simultaneous_co_primary": simultaneous,
        "individual_intervals": individual,
        "heterogeneity": heterogeneity,
        "regimes": summaries,
        "inputs": input_records,
        "integrity": global_integrity,
        "bootstrap": bootstrap_summary["bootstrap"] | {
            "artifact": bootstrap_manifest
        },
        "compute_ledger": {
            "path": str(compute_path.resolve().relative_to(REPO_ROOT.resolve())),
            "sha256": compute_sha256,
        },
        "fixed_whitening": {
            "path": whitening_relative,
            "sha256": authorization["fixed_whitening_sha256"],
            "unchanged_v5_endpoint": authorization[
                "fixed_whitening_matches_known_v5_v004_content_hash"
            ],
        },
        "sealed_analysis_authorization": authorization,
        "independent_verification_required": True,
        "posthoc_contact_fields_opened": False,
    }
    atomic_json(output_path, result, exclusive=True)
    return result


def capture_post_open_analysis_failure(
    error: Exception,
    *,
    execution_root: Path,
    whitening_path: Path,
    pricing_path: Path,
    seed_path: Path,
    thresholds_path: Path,
    output_root: Path = ATTEMPT_ROOT,
) -> Path | None:
    """Persist one outcome-free failure audit after irreversible array open.

    The controller is queried first, so a pre-open authorization error cannot
    be mislabeled as an irreversible confirmation failure.  This routine does
    not open or summarize an NPZ; the confirmation input-seal hash transitively
    binds the already authenticated execution inputs.
    """

    state = read_verified_controller()
    if state.get("confirmation_outcomes_opened_for_analysis") is not True:
        return None
    if not (
        state.get("active_attempt") == ACTIVE_ATTEMPT
        and state.get("current_state") == "SEALED_ANALYSIS"
    ):
        raise RuntimeError(
            "post-open analysis failure occurred outside the bound analysis state"
        )
    study_root = output_root.parents[1]
    state_path = study_root / "STATE.json"
    ledger_path = study_root / "RESEARCH_LEDGER.jsonl"
    input_paths = {
        "confirmation_input_seal": output_root
        / "audit/confirmation_input_seal.json",
        "pre_confirmation_package_seal": output_root
        / "audit/pre_confirmation_package_seal.json",
        "analysis_source": Path(__file__),
        "fixed_whitening": whitening_path,
        "compiled_gate_manifest": pricing_path,
        "cohort_seed_ledger": seed_path,
        "compiled_gate": thresholds_path,
    }
    input_hashes: dict[str, dict[str, Any]] = {}
    for name, path in input_paths.items():
        candidate = Path(path)
        if not candidate.is_file() or candidate.is_symlink():
            raise RuntimeError(
                f"post-open failure audit input missing or linked: {name}"
            )
        input_hashes[name] = {
            "path": artifact_path_text(candidate),
            "sha256": sha256_file(candidate),
            "bytes": int(candidate.stat().st_size),
        }
    analysis_result_path = output_root / "analysis_result.json"
    analysis_result_record: dict[str, Any] | None = None
    if analysis_result_path.exists() or analysis_result_path.is_symlink():
        if analysis_result_path.is_symlink() or not analysis_result_path.is_file():
            raise RuntimeError("present analysis result is linked or nonregular")
        analysis_result_record = {
            "path": artifact_path_text(analysis_result_path),
            "sha256": sha256_file(analysis_result_path),
            "bytes": int(analysis_result_path.stat().st_size),
        }
    partial_paths = tuple(
        output_root / f"metrics/{regime}_episode_metrics.npz"
        for regime in DGP_ORDER
    ) + (
        output_root / "metrics/bootstrap_replicates.npz",
        output_root / "metrics/compute_ledger.json",
        output_root / "metrics/bootstrap_summary.json",
    )
    partial_derived_outputs: dict[str, dict[str, Any]] = {}
    for candidate in partial_paths:
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"partial derived output is linked or nonregular: {candidate}")
        relative = artifact_path_text(candidate)
        partial_derived_outputs[relative] = {
            "sha256": sha256_file(candidate),
            "bytes": int(candidate.stat().st_size),
        }
    if not state_path.is_file() or not ledger_path.is_file():
        raise RuntimeError("controller chronology files are absent after array open")
    formatted_traceback = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    result = {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "SEALED_ANALYSIS",
        "status": "irreversible_post_open_analysis_execution_invalid",
        "created_unix_ns": time.time_ns(),
        "passed": False,
        "integrity_failure": True,
        "integrity_passed": False,
        "process_valid": False,
        "execution_invalid": True,
        "proposed_terminal_label": TERMINAL_INVALID,
        "confirmation_opened": True,
        "confirmation_outcomes_opened_for_analysis": True,
        "confirmation_outcome_episodes_generated": state.get(
            "confirmation_outcome_episodes_generated"
        ),
        "confirmation_outcome_episodes_executed": state.get(
            "confirmation_outcome_episodes_executed"
        ),
        "analysis_result_present": analysis_result_record is not None,
        "analysis_result": analysis_result_record,
        "error_type": type(error).__qualname__,
        "error": str(error),
        "scientific_objects_changed": False,
        "failure": {
            "exception_module": type(error).__module__,
            "exception_type": type(error).__qualname__,
            "exception_message": str(error),
            "traceback": formatted_traceback,
            "traceback_includes_locals": False,
        },
        "controller_chronology": {
            "state_path": artifact_path_text(state_path),
            "state_sha256": sha256_file(state_path),
            "research_ledger_path": artifact_path_text(ledger_path),
            "research_ledger_sha256": sha256_file(ledger_path),
            "ledger_event_count": state.get("ledger_event_count"),
            "ledger_head_sha256": state.get("ledger_head_sha256"),
            "last_verified_checkpoint": state.get("last_verified_checkpoint"),
            "controller_status_verified": True,
        },
        "inputs": input_hashes,
        "partial_derived_outputs": partial_derived_outputs,
        "confirmation_input_seal_sha256": input_hashes[
            "confirmation_input_seal"
        ]["sha256"],
        "execution_root": artifact_path_text(execution_root),
        "failure_capture_opened_no_arrays": True,
        "outcome_values_recorded": False,
        "contact_motion_phase_reward_success_opened": False,
        "scientific_objects_changed_after_open": False,
        "retry_permitted": False,
        "independent_verification_required": True,
    }
    path = output_root / "audit/analysis_execution_invalid.json"
    atomic_json(path, result, exclusive=True)
    return path


def execute_analysis_cli(arguments: argparse.Namespace) -> dict[str, Any]:
    """Execute the CLI body while distinguishing interruption from failure."""

    try:
        result = run_sealed_analysis(
            execution_root=arguments.execution_root,
            whitening_path=arguments.whitening,
            pricing_path=arguments.pricing,
            seed_path=arguments.seeds,
            thresholds_path=arguments.thresholds,
        )
        if not (
            result.get("passed") is True
            and result.get("process_valid") is True
            and result.get("integrity_passed") is True
        ):
            raise AnalysisIntegrityFailure(
                "sealed analysis completed with an immutable integrity-false result"
            )
        print(json.dumps(result, sort_keys=True, default=_json_default))
        return result
    except Exception as error:
        try:
            capture_post_open_analysis_failure(
                error,
                execution_root=arguments.execution_root,
                whitening_path=arguments.whitening,
                pricing_path=arguments.pricing,
                seed_path=arguments.seeds,
                thresholds_path=arguments.thresholds,
            )
        except Exception as capture_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    "post-open failure capture also failed without overwriting "
                    f"evidence: {type(capture_error).__name__}: {capture_error}"
                )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-root",
        type=Path,
        default=ATTEMPT_ROOT / "data/confirmation",
    )
    parser.add_argument(
        "--whitening",
        type=Path,
        default=REPO_ROOT
        / "runs/lewm_v5_readiness_program/v5_package_versions/v004/freeze/whitening.npz",
    )
    parser.add_argument(
        "--pricing",
        type=Path,
        default=ATTEMPT_ROOT / "freeze/compiled_gate_manifest.json",
    )
    parser.add_argument(
        "--seeds", type=Path, default=ATTEMPT_ROOT / "cohort_seed_ledger.json"
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ATTEMPT_ROOT / "freeze/compiled_gate.npz",
    )
    execute_analysis_cli(parser.parse_args())


if __name__ == "__main__":
    main()
