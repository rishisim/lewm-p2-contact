#!/usr/bin/env python3
"""Build the immutable, globally fresh v001 cohort and RNG ledger.

This module reads metadata only.  It never opens rollout arrays, V3 targets,
the combined V3 cache, or an HDF5 corpus.  The fixed namespace is deliberately
below 2**32 and is rejected if any allowed prior metadata already contains one
of its identifiers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ATTEMPT_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ATTEMPT_ROOT.parents[1]
RUNS_ROOT = ATTEMPT_ROOT.parents[2]

REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
REGIME_SHORT = {
    "native_plan": "np",
    "markov_oracle": "mo",
    "plan_action_noise_0p2": "n2",
    "plan_random_action_0p1": "r1",
}
ROLES = ("fit", "selection", "smoke", "confirmation")
ROLE_SHORT = {
    "fit": "ft",
    "selection": "sl",
    "smoke": "sm",
    "confirmation": "cf",
}
PRIMARY_COUNTS = {
    "fit": 300,
    "selection": 500,
    "smoke": 6,
    "confirmation": 4_500,
}
REPLACEMENTS_PER_ROLE = 200
BOOTSTRAP_REPLICATES = 20_000

# 24,424 records fit within each 30,000-wide seed-stream block.  The fifth
# block holds all process RNG identifiers.
NUMERIC_NAMESPACE_BASE = 4_100_000_000
SEED_STREAM_STRIDE = 30_000
RNG_NAMESPACE_BASE = NUMERIC_NAMESPACE_BASE + 4 * SEED_STREAM_STRIDE
UINT32_LIMIT = 2**32

FORBIDDEN_TREE_NAMES = frozenset({"lewm_adaptive_compute_v3"})
METADATA_SUFFIXES = frozenset({".json", ".jsonl"})
REQUIRED_PRIOR_METADATA = (
    "lewm_v5_readiness_program/v5_package_versions/v001/cohort_seed_ledger.json",
    "lewm_v5_readiness_program/v5_package_versions/v002/cohort_seed_ledger.json",
    "lewm_v5_readiness_program/v5_package_versions/v004/cohort_seed_ledger.json",
    "lewm_v5_readiness_program/v5_package_versions/v004/cohort_seed_ledger.json",
    "lewm_v5_generalization/attempts/v005/cohort_seed_ledger.json",
    "lewm_v5_readiness_program/v5_package_versions/v004/audit/prior_identifier_snapshot.json",
)
V004_PRIOR_SNAPSHOT_RELATIVE = REQUIRED_PRIOR_METADATA[-1]
SAFE_V3_RELATIVE_PATHS = (
    "lewm_adaptive_compute_v3/run_experiment.py",
    "lewm_adaptive_compute_v3/model.py",
    "lewm_adaptive_compute_v3/policy.py",
    "lewm_adaptive_compute_v3/full_config.json",
    "lewm_adaptive_compute_v3/smoke_config.json",
    "lewm_adaptive_compute_v3/preregistration.json",
    "lewm_adaptive_compute_v3/PLAN.md",
)


def _walk_json(value: Any) -> Iterator[Any]:
    """Yield JSON leaves and mapping keys without treating bools as ints."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)
    else:
        yield value


def _digest(values: Iterable[Any]) -> str:
    payload = "\n".join(sorted((str(value) for value in values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _forbidden(path: Path) -> bool:
    return any(part in FORBIDDEN_TREE_NAMES for part in path.parts)


def _safe_v3_identifiers(path: Path, payload: bytes) -> tuple[set[int], set[str]]:
    """Extract literals from the explicit safe V3 source/config/plan allowlist."""

    text = payload.decode("utf-8")
    numbers: set[int] = set()
    strings: set[str] = {path.name, path.stem}
    if path.suffix == ".json":
        value = json.loads(text)
        for item in _walk_json(value):
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                numbers.add(item)
            elif isinstance(item, str):
                strings.add(item)
    elif path.suffix == ".py":
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if type(node.value) is int:
                    numbers.add(node.value)
                elif isinstance(node.value, str):
                    strings.add(node.value)
            elif isinstance(node, ast.Name):
                strings.add(node.id)
    else:
        numbers.update(
            int(token.replace("_", ""))
            for token in re.findall(r"(?<![\w.])-?\d[\d_]*(?![\w.])", text)
        )
        strings.update(re.findall(r"[A-Za-z][A-Za-z0-9_.:/-]{2,}", text))
    return numbers, strings


def scan_prior_metadata(
    runs_root: Path = RUNS_ROOT,
    *,
    excluded_roots: Sequence[Path] | None = None,
    required_relative_paths: Sequence[str] = REQUIRED_PRIOR_METADATA,
    require_authoritative: bool = True,
) -> tuple[set[int], set[str], dict[str, Any]]:
    """Scan allowed prior JSON/JSONL and filenames for identifier collisions.

    Binary contents are never opened.  V3 is restricted to an explicit safe
    source/config/plan allowlist; its tests, targets, caches, splits, metrics,
    and outcome artifacts remain unopened.  The current study root is excluded
    so concurrent construction of v001 cannot change the prior snapshot.
    """

    runs_root = runs_root.resolve()
    if excluded_roots is None:
        excluded_roots = (STUDY_ROOT,)
    excluded = tuple(path.resolve() for path in excluded_roots)

    numbers: set[int] = set()
    strings: set[str] = set()
    metadata_paths: list[str] = []
    filename_paths: list[str] = []
    file_hashes: dict[str, str] = {}
    required = set(required_relative_paths)

    for path in sorted(runs_root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if _forbidden(resolved) or any(_under(resolved, root) for root in excluded):
            continue
        relative = path.relative_to(runs_root).as_posix()
        filename_paths.append(relative)
        # Filename identifiers matter, but file contents are opened only for
        # explicitly allowed textual metadata suffixes.
        strings.update((path.name, path.stem))
        if path.suffix.lower() not in METADATA_SUFFIXES:
            continue

        payload = path.read_bytes()
        file_hashes[relative] = _sha256_bytes(payload)
        try:
            if path.suffix.lower() == ".json":
                documents = (json.loads(payload.decode("utf-8")),)
            else:
                documents = tuple(
                    json.loads(line)
                    for line in payload.decode("utf-8").splitlines()
                    if line.strip()
                )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unparseable allowed prior metadata: {relative}") from exc

        for document in documents:
            for item in _walk_json(document):
                if isinstance(item, bool):
                    continue
                if isinstance(item, int):
                    numbers.add(item)
                elif isinstance(item, str):
                    strings.add(item)
        metadata_paths.append(relative)

    safe_v3_hashes: dict[str, str] = {}
    safe_v3_numbers: set[int] = set()
    safe_v3_strings: set[str] = set()
    for relative in SAFE_V3_RELATIVE_PATHS:
        path = runs_root / relative
        if not path.is_file() or path.is_symlink():
            if require_authoritative:
                raise RuntimeError(f"required safe V3 audit text missing: {relative}")
            continue
        payload = path.read_bytes()
        try:
            local_numbers, local_strings = _safe_v3_identifiers(path, payload)
        except (UnicodeDecodeError, json.JSONDecodeError, SyntaxError) as exc:
            raise RuntimeError(f"cannot audit safe V3 text: {relative}") from exc
        safe_v3_numbers.update(local_numbers)
        safe_v3_strings.update(local_strings)
        safe_v3_hashes[relative] = _sha256_bytes(payload)
        numbers.update(local_numbers)
        strings.update(local_strings)
        filename_paths.append(relative)
        metadata_paths.append(relative)

    v004_snapshot_path = runs_root / V004_PRIOR_SNAPSHOT_RELATIVE
    v004_snapshot_cross_reference: dict[str, Any] | None = None
    if v004_snapshot_path.is_file():
        v004_value = json.loads(v004_snapshot_path.read_text(encoding="utf-8"))
        v004_snapshot_cross_reference = {
            "path": V004_PRIOR_SNAPSHOT_RELATIVE,
            "sha256": _sha256_bytes(v004_snapshot_path.read_bytes()),
            "declared_scope": v004_value.get("scope"),
            "declared_json_file_count": v004_value.get("json_file_count"),
            "declared_numeric_identifier_count": v004_value.get(
                "numeric_identifier_count"
            ),
            "declared_string_identifier_count": v004_value.get(
                "string_identifier_count"
            ),
            "declared_numeric_identifier_set_sha256": v004_value.get(
                "numeric_identifier_set_sha256"
            ),
            "declared_string_identifier_set_sha256": v004_value.get(
                "string_identifier_set_sha256"
            ),
        }

    missing = sorted(required - set(metadata_paths))
    if require_authoritative and missing:
        raise RuntimeError(f"required consumed identifier metadata missing: {missing}")

    required_hashes = {
        relative: file_hashes[relative]
        for relative in required_relative_paths
        if relative in file_hashes
    }
    snapshot = {
        "schema_version": 1,
        "scope": (
            "all non-V3 parseable JSON/JSONL and allowed filenames under runs, "
            "plus the explicit safe V3 source/config/plan allowlist; excludes "
            "the current study and every forbidden V3 artifact class"
        ),
        "metadata_file_count": len(metadata_paths),
        "filename_count": len(filename_paths),
        "numeric_identifier_count": len(numbers),
        "string_identifier_count": len(strings),
        "numeric_identifier_set_sha256": _digest(numbers),
        "string_identifier_set_sha256": _digest(strings),
        "metadata_path_set_sha256": _digest(metadata_paths),
        "filename_path_set_sha256": _digest(filename_paths),
        "required_consumed_metadata": required_hashes,
        "required_consumed_metadata_complete": not missing,
        "missing_required_consumed_metadata": missing,
        "safe_v3_source_config_plan_files": safe_v3_hashes,
        "safe_v3_numeric_literal_count": len(safe_v3_numbers),
        "safe_v3_numeric_literal_set_sha256": _digest(safe_v3_numbers),
        "safe_v3_string_identifier_count": len(safe_v3_strings),
        "safe_v3_string_identifier_set_sha256": _digest(safe_v3_strings),
        "v004_authoritative_prior_snapshot": v004_snapshot_cross_reference,
        "v004_authoritative_prior_snapshot_cross_referenced": (
            v004_snapshot_cross_reference is not None
        ),
        "v3_safe_source_config_plan_opened": bool(safe_v3_hashes),
        "v3_test_target_cache_split_metrics_artifacts_opened": False,
        "binary_contents_opened": False,
        "outcome_array_contents_opened": False,
        "hdf5_contents_opened": False,
    }
    return numbers, strings, snapshot


def _record(global_index: int, regime: str, role: str, pool: str, slot: int) -> dict[str, Any]:
    pool_short = "p" if pool == "primary" else "r"
    width = 4 if pool == "primary" and role == "confirmation" else 3
    episode_id = (
        f"drgv001-{REGIME_SHORT[regime]}-{ROLE_SHORT[role]}-"
        f"{pool_short}-{slot:0{width}d}"
    )
    return {
        "slot": slot,
        "episode_id": episode_id,
        "env_seed": NUMERIC_NAMESPACE_BASE + global_index,
        "policy_seed": NUMERIC_NAMESPACE_BASE + SEED_STREAM_STRIDE + global_index,
        "oracle_np_seed": NUMERIC_NAMESPACE_BASE + 2 * SEED_STREAM_STRIDE + global_index,
        "action_space_seed": NUMERIC_NAMESPACE_BASE + 3 * SEED_STREAM_STRIDE + global_index,
    }


def _rng_record(counter: list[int], group: str, purpose: str, regime: str | None = None) -> dict[str, Any]:
    rng_id = RNG_NAMESPACE_BASE + counter[0]
    counter[0] += 1
    parts = ["drgv001", "rng", group]
    if regime is not None:
        parts.append(REGIME_SHORT[regime])
    parts.append(purpose.replace("_", "-"))
    item: dict[str, Any] = {
        "purpose": purpose,
        "label": "-".join(parts),
        "rng_id": rng_id,
    }
    if regime is not None:
        item["regime"] = regime
    return item


def build_analysis_rng_ids() -> dict[str, Any]:
    """Assign traceable RNG IDs even to frozen deterministic procedures."""

    counter = [0]
    fit = [
        _rng_record(counter, "fit", purpose)
        for purpose in (
            "pooled_candidate_fit",
            "pooled_normalization_fit",
            "pooled_robust_whitening_fit",
        )
    ]
    for regime in REGIMES:
        fit.extend(
            _rng_record(counter, "fit", purpose, regime)
            for purpose in (
                "candidate_fit",
                "normalization_fit",
                "robust_whitening_fit",
            )
        )

    selection = [
        _rng_record(counter, "selection", "worst_dgp_candidate_scoring"),
        _rng_record(counter, "selection", "lexicographic_tie_break"),
    ]
    for regime in REGIMES:
        selection.extend(
            _rng_record(counter, "selection", purpose, regime)
            for purpose in ("candidate_metric_evaluation", "routing_compute_constraint")
        )

    bootstrap = {
        "rng_ids": [_rng_record(counter, "bootstrap", "joint_master")],
        "replicate_count": BOOTSTRAP_REPLICATES,
        "chunk_size_defining_rng_consumption": 250,
        "regime_order": list(REGIMES),
        "endpoint_order": ["raw_vs_analytic", "fixed_whitened_vs_analytic"],
        "replicate_derivation": (
            "instantiate numpy default_rng(joint_master) once; iterate replicate "
            "chunks of 250 in ascending order, then regimes in frozen regime_order; "
            "draw one int32 episode-index matrix of shape (chunk_size, fixed_N) "
            "per regime/chunk and reuse that paired matrix across every endpoint "
            "and control for the regime"
        ),
    }

    comparator: list[dict[str, Any]] = []
    for regime in REGIMES:
        comparator.extend(
            _rng_record(counter, "comparator", purpose, regime)
            for purpose in (
                "seeded_weak_more_raw",
                "seeded_weak_more_fixed_whitened",
                "within_episode_call_histogram",
            )
        )

    qualification = [
        _rng_record(counter, "qualification", purpose)
        for purpose in (
            "seed_ledger",
            "dgp_constructor",
            "candidate_family",
            "causal_input_isolation",
            "sparse_dense_equivalence",
            "synthetic_inference",
            "terminal_mapping",
            "independent_verifier",
        )
    ]

    latency = [_rng_record(counter, "latency", "global_order")]
    latency.extend(
        _rng_record(counter, "latency", "regime_order", regime)
        for regime in REGIMES
    )
    return {
        "fit": {
            "rng_ids": fit,
            "deterministic_procedures_still_have_trace_ids": True,
        },
        "selection": {
            "rng_ids": selection,
            "deterministic_procedures_still_have_trace_ids": True,
        },
        "bootstrap": bootstrap,
        "comparator": {"rng_ids": comparator},
        "qualification": {"rng_ids": qualification},
        "latency": {"rng_ids": latency},
    }


def iter_rng_records(analysis_rng_ids: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for group in analysis_rng_ids.values():
        if not isinstance(group, Mapping):
            continue
        records = group.get("rng_ids", [])
        if not isinstance(records, list):
            raise TypeError("analysis RNG group rng_ids must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("analysis RNG record must be a mapping")
            yield record


def canonical_analysis_seeds(analysis_rng_ids: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the analysis-facing aliases without allocating any new IDs."""

    bootstrap_records = analysis_rng_ids["bootstrap"]["rng_ids"]
    comparator_records = analysis_rng_ids["comparator"]["rng_ids"]
    comparator_lookup = {
        (str(item["regime"]), str(item["purpose"])): int(item["rng_id"])
        for item in comparator_records
    }
    return {
        "joint_bootstrap_seed": int(bootstrap_records[0]["rng_id"]),
        "bootstrap_replicates": int(analysis_rng_ids["bootstrap"]["replicate_count"]),
        "seeded_comparator_seeds": {
            regime: {
                "raw": comparator_lookup[(regime, "seeded_weak_more_raw")],
                "fixed_whitened": comparator_lookup[
                    (regime, "seeded_weak_more_fixed_whitened")
                ],
            }
            for regime in REGIMES
        },
        "histogram_seeds": {
            regime: comparator_lookup[(regime, "within_episode_call_histogram")]
            for regime in REGIMES
        },
        "aliases_only_no_additional_rng_ids": True,
    }


def build_seed_ledger(
    prior_numbers: set[int],
    prior_strings: set[str],
    prior_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    global_index = 0
    all_records: list[dict[str, Any]] = []
    for regime in REGIMES:
        role_map: dict[str, Any] = {}
        for role in ROLES:
            primary = []
            for slot in range(PRIMARY_COUNTS[role]):
                item = _record(global_index, regime, role, "primary", slot)
                primary.append(item)
                all_records.append(item)
                global_index += 1
            replacements = []
            for slot in range(REPLACEMENTS_PER_ROLE):
                item = _record(global_index, regime, role, "replacements", slot)
                replacements.append(item)
                all_records.append(item)
                global_index += 1
            role_map[role] = {
                "primary": primary,
                "replacements": replacements,
            }
        regimes[regime] = {"roles": role_map}

    analysis_rng_ids = build_analysis_rng_ids()
    analysis_seeds = canonical_analysis_seeds(analysis_rng_ids)
    rng_records = list(iter_rng_records(analysis_rng_ids))
    episode_ids = [str(item["episode_id"]) for item in all_records]
    seed_numbers = [
        int(item[key])
        for item in all_records
        for key in ("env_seed", "policy_seed", "oracle_np_seed", "action_space_seed")
    ]
    rng_numbers = [int(item["rng_id"]) for item in rng_records]
    rng_labels = [str(item["label"]) for item in rng_records]
    new_numbers = set(seed_numbers + rng_numbers)
    new_strings = set(episode_ids + rng_labels)

    expected_record_count = len(REGIMES) * (
        sum(PRIMARY_COUNTS.values()) + len(ROLES) * REPLACEMENTS_PER_ROLE
    )
    expected_number_count = 4 * expected_record_count + len(rng_records)
    numeric_overlap = sorted(new_numbers & prior_numbers)
    string_overlap = sorted(new_strings & prior_strings)
    checks = {
        "exact_record_count": len(all_records) == expected_record_count,
        "exact_primary_counts": all(
            len(regimes[regime]["roles"][role]["primary"]) == PRIMARY_COUNTS[role]
            for regime in REGIMES
            for role in ROLES
        ),
        "exact_200_replacements_per_dgp_per_role": all(
            len(regimes[regime]["roles"][role]["replacements"])
            == REPLACEMENTS_PER_ROLE
            for regime in REGIMES
            for role in ROLES
        ),
        "episode_ids_globally_unique": len(set(episode_ids)) == len(episode_ids),
        "numeric_identifiers_globally_unique": len(new_numbers) == expected_number_count,
        "rng_labels_globally_unique": len(set(rng_labels)) == len(rng_labels),
        "numeric_namespace_uint32": bool(new_numbers)
        and min(new_numbers) >= 0
        and max(new_numbers) < UINT32_LIMIT,
        "numeric_namespace_fixed": bool(new_numbers)
        and min(new_numbers) >= NUMERIC_NAMESPACE_BASE
        and max(new_numbers) < RNG_NAMESPACE_BASE + SEED_STREAM_STRIDE,
        "zero_prior_numeric_overlap": not numeric_overlap,
        "zero_prior_string_overlap": not string_overlap,
        "required_consumed_metadata_complete": bool(
            prior_snapshot.get("required_consumed_metadata_complete", False)
        ),
        "v004_authoritative_all_prior_snapshot_cross_referenced": bool(
            prior_snapshot.get(
                "v004_authoritative_prior_snapshot_cross_referenced", False
            )
        ),
        "safe_v3_source_config_plan_audited": bool(
            prior_snapshot.get("safe_v3_source_config_plan_files")
        ),
        "forbidden_binary_and_v3_contents_unopened": (
            prior_snapshot.get(
                "v3_test_target_cache_split_metrics_artifacts_opened"
            )
            is False
            and prior_snapshot.get("binary_contents_opened") is False
            and prior_snapshot.get("hdf5_contents_opened") is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "identifier ledger failed closed: "
            + json.dumps(
                {
                    "failed_checks": [name for name, passed in checks.items() if not passed],
                    "numeric_overlap": numeric_overlap[:20],
                    "string_overlap": string_overlap[:20],
                },
                sort_keys=True,
            )
        )

    return {
        "schema_version": 1,
        "attempt": "v001",
        "study": "lewm_domain_robust_gate",
        "regime_order": list(REGIMES),
        "role_order": list(ROLES),
        "role_primary_counts_per_regime": dict(PRIMARY_COUNTS),
        "replacement_count_per_regime_per_role": REPLACEMENTS_PER_ROLE,
        "regimes": regimes,
        "analysis_rng_ids": analysis_rng_ids,
        "analysis_seeds": analysis_seeds,
        "numeric_namespace": {
            "lower_inclusive": min(new_numbers),
            "upper_inclusive": max(new_numbers),
            "uint32_upper_exclusive": UINT32_LIMIT,
            "fixed_base": NUMERIC_NAMESPACE_BASE,
            "seed_stream_stride": SEED_STREAM_STRIDE,
        },
        "confirmation_prefix_rule": (
            "the fixed post-selection N uses exactly primary confirmation slots "
            "0..N-1 separately in every regime; later slots are never substituted"
        ),
        "confirmation_max_slots_per_regime": PRIMARY_COUNTS["confirmation"],
        "fit_selection_smoke_confirmation_roles_disjoint": True,
        "smoke_permanently_excluded": True,
        "replacement_pool_regime_and_role_specific_single_use": True,
        "replacement_rule": (
            "mechanical failure before artifact completion only; target, loss, "
            "contact, privileged state, motion, phase, reward, success, and all "
            "outcomes are forbidden from replacement, exclusion, and stopping"
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "prior_identifier_snapshot": dict(prior_snapshot),
        "numeric_overlap": numeric_overlap,
        "string_overlap": string_overlap,
        "checks": checks,
        "fresh_outcome_episodes_opened": 0,
        "prior_outcome_arrays_opened": 0,
        "v3_targets_opened": False,
        "combined_v3_cache_numpy_loaded": False,
        "released_hdf5_opened": False,
    }


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Durably create JSON without an overwrite path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ATTEMPT_ROOT / "cohort_seed_ledger.json",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RuntimeError(f"immutable cohort seed ledger already exists: {args.output}")
    numbers, strings, snapshot = scan_prior_metadata(args.runs_root)
    ledger = build_seed_ledger(numbers, strings, snapshot)
    write_json_exclusive(args.output, ledger)
    print(json.dumps({"output": str(args.output), "checks": ledger["checks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
