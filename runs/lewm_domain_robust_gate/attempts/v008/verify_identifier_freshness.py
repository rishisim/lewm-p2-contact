#!/usr/bin/env python3
"""Independent read-only verification of the v001 identifier ledger."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
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
ROLES = ("fit", "selection", "smoke", "confirmation")
PRIMARY_COUNTS = {"fit": 300, "selection": 500, "smoke": 6, "confirmation": 4_500}
REPLACEMENT_COUNT = 200
BOOTSTRAP_REPLICATES = 20_000
NAMESPACE_BASE = 4_100_000_000
RNG_NAMESPACE_BASE = 4_100_120_000
UINT32_LIMIT = 2**32
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
SEED_KEYS = ("env_seed", "policy_seed", "oracle_np_seed", "action_space_seed")
RECORD_KEYS = frozenset(("slot", "episode_id", *SEED_KEYS))


def _leaves(value: Any) -> Iterator[Any]:
    """A separately written JSON traversal used only by the verifier."""

    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                yield str(key)
                pending.append(child)
        elif isinstance(current, list):
            pending.extend(reversed(current))
        else:
            yield current


def _set_hash(items: Iterable[Any]) -> str:
    ordered = sorted(str(item) for item in items)
    return hashlib.sha256(("\n".join(ordered) + "\n").encode("utf-8")).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _audit_safe_v3_text(path: Path) -> tuple[set[int], set[str], str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    numeric: set[int] = set()
    textual: set[str] = {path.name, path.stem}
    if path.suffix == ".json":
        for leaf in _leaves(json.loads(text)):
            if type(leaf) is int:
                numeric.add(leaf)
            elif isinstance(leaf, str):
                textual.add(leaf)
    elif path.suffix == ".py":
        parsed = ast.parse(text, filename=str(path))
        constants = [node for node in ast.walk(parsed) if isinstance(node, ast.Constant)]
        for node in constants:
            if type(node.value) is int:
                numeric.add(node.value)
            elif isinstance(node.value, str):
                textual.add(node.value)
        textual.update(node.id for node in ast.walk(parsed) if isinstance(node, ast.Name))
    else:
        for token in re.findall(r"(?<![\w.])-?\d[\d_]*(?![\w.])", text):
            numeric.add(int(token.replace("_", "")))
        textual.update(re.findall(r"[A-Za-z][A-Za-z0-9_.:/-]{2,}", text))
    return numeric, textual, hashlib.sha256(raw).hexdigest()


def independently_scan_prior(
    runs_root: Path = RUNS_ROOT,
    *,
    excluded_roots: Sequence[Path] | None = None,
    required_relative_paths: Sequence[str] = REQUIRED_PRIOR_METADATA,
    require_authoritative: bool = True,
) -> tuple[set[int], set[str], dict[str, Any]]:
    """Rebuild the prior universe without importing the ledger builder."""

    runs_root = runs_root.resolve()
    if excluded_roots is None:
        excluded_roots = (STUDY_ROOT,)
    exclusions = tuple(root.resolve() for root in excluded_roots)
    numeric: set[int] = set()
    textual: set[str] = set()
    metadata_paths: list[str] = []
    filename_paths: list[str] = []
    hashes: dict[str, str] = {}

    for candidate in sorted(runs_root.rglob("*")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        absolute = candidate.resolve()
        if "lewm_adaptive_compute_v3" in absolute.parts:
            continue
        if any(_inside(absolute, excluded) for excluded in exclusions):
            continue
        relative = candidate.relative_to(runs_root).as_posix()
        filename_paths.append(relative)
        textual.add(candidate.name)
        textual.add(candidate.stem)
        suffix = candidate.suffix.casefold()
        if suffix not in {".json", ".jsonl"}:
            continue

        raw = candidate.read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        try:
            decoded = raw.decode("utf-8")
            if suffix == ".json":
                documents = [json.loads(decoded)]
            else:
                documents = [json.loads(line) for line in decoded.splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"independent scan cannot parse {relative}") from exc
        for document in documents:
            for leaf in _leaves(document):
                if type(leaf) is int:
                    numeric.add(leaf)
                elif isinstance(leaf, str):
                    textual.add(leaf)
        metadata_paths.append(relative)

    safe_v3_hashes: dict[str, str] = {}
    safe_v3_numbers: set[int] = set()
    safe_v3_strings: set[str] = set()
    for relative in SAFE_V3_RELATIVE_PATHS:
        path = runs_root / relative
        if not path.is_file() or path.is_symlink():
            if require_authoritative:
                raise RuntimeError(f"safe V3 audit input is absent: {relative}")
            continue
        try:
            local_numbers, local_strings, digest = _audit_safe_v3_text(path)
        except (UnicodeDecodeError, json.JSONDecodeError, SyntaxError) as exc:
            raise RuntimeError(f"independent safe V3 audit failed: {relative}") from exc
        safe_v3_numbers.update(local_numbers)
        safe_v3_strings.update(local_strings)
        numeric.update(local_numbers)
        textual.update(local_strings)
        safe_v3_hashes[relative] = digest
        filename_paths.append(relative)
        metadata_paths.append(relative)

    required = set(required_relative_paths)
    missing = sorted(required - set(metadata_paths))
    if require_authoritative and missing:
        raise RuntimeError(f"required prior ledgers absent during independent scan: {missing}")
    v004_snapshot_cross_reference: dict[str, Any] | None = None
    v004_path = runs_root / V004_PRIOR_SNAPSHOT_RELATIVE
    if v004_path.is_file():
        raw = v004_path.read_bytes()
        declared = json.loads(raw.decode("utf-8"))
        v004_snapshot_cross_reference = {
            "path": V004_PRIOR_SNAPSHOT_RELATIVE,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "declared_scope": declared.get("scope"),
            "declared_json_file_count": declared.get("json_file_count"),
            "declared_numeric_identifier_count": declared.get(
                "numeric_identifier_count"
            ),
            "declared_string_identifier_count": declared.get(
                "string_identifier_count"
            ),
            "declared_numeric_identifier_set_sha256": declared.get(
                "numeric_identifier_set_sha256"
            ),
            "declared_string_identifier_set_sha256": declared.get(
                "string_identifier_set_sha256"
            ),
        }
    summary = {
        "metadata_file_count": len(metadata_paths),
        "filename_count": len(filename_paths),
        "numeric_identifier_count": len(numeric),
        "string_identifier_count": len(textual),
        "numeric_identifier_set_sha256": _set_hash(numeric),
        "string_identifier_set_sha256": _set_hash(textual),
        "metadata_path_set_sha256": _set_hash(metadata_paths),
        "filename_path_set_sha256": _set_hash(filename_paths),
        "required_consumed_metadata": {
            relative: hashes[relative]
            for relative in required_relative_paths
            if relative in hashes
        },
        "required_consumed_metadata_complete": not missing,
        "missing_required_consumed_metadata": missing,
        "safe_v3_source_config_plan_files": safe_v3_hashes,
        "safe_v3_numeric_literal_count": len(safe_v3_numbers),
        "safe_v3_numeric_literal_set_sha256": _set_hash(safe_v3_numbers),
        "safe_v3_string_identifier_count": len(safe_v3_strings),
        "safe_v3_string_identifier_set_sha256": _set_hash(safe_v3_strings),
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
    return numeric, textual, summary


def _rng_records(ledger: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    groups = ledger.get("analysis_rng_ids")
    if not isinstance(groups, Mapping):
        return result
    for group in groups.values():
        if not isinstance(group, Mapping):
            continue
        records = group.get("rng_ids")
        if isinstance(records, list):
            result.extend(item for item in records if isinstance(item, Mapping))
    return result


def _required_direct_overlaps(
    runs_root: Path,
    current_numbers: set[int],
    current_strings: set[str],
    required_relative_paths: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in required_relative_paths:
        path = runs_root / relative
        if not path.is_file():
            result[relative] = {"missing": True}
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        numbers = {leaf for leaf in _leaves(document) if type(leaf) is int}
        strings = {leaf for leaf in _leaves(document) if isinstance(leaf, str)}
        result[relative] = {
            "missing": False,
            "numeric_overlap": sorted(current_numbers & numbers),
            "string_overlap": sorted(current_strings & strings),
        }
    return result


def verify_identifier_freshness(
    ledger: Mapping[str, Any],
    prior_numbers: set[int],
    prior_strings: set[str],
    independent_snapshot: Mapping[str, Any],
    *,
    direct_required_overlap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    all_records: list[Mapping[str, Any]] = []
    role_episode_sets: dict[tuple[str, str], set[str]] = {}
    exact_schema = True
    exact_slots = True
    exact_counts = True
    try:
        regimes = ledger["regimes"]
        for regime in REGIMES:
            role_map = regimes[regime]["roles"]
            for role in ROLES:
                pools = role_map[role]
                primary = pools["primary"]
                replacements = pools["replacements"]
                exact_counts = exact_counts and (
                    len(primary) == PRIMARY_COUNTS[role]
                    and len(replacements) == REPLACEMENT_COUNT
                )
                exact_slots = exact_slots and [item.get("slot") for item in primary] == list(
                    range(PRIMARY_COUNTS[role])
                )
                exact_slots = exact_slots and [
                    item.get("slot") for item in replacements
                ] == list(range(REPLACEMENT_COUNT))
                for item in [*primary, *replacements]:
                    exact_schema = exact_schema and set(item) == RECORD_KEYS
                    all_records.append(item)
                role_episode_sets[(regime, role)] = {
                    str(item["episode_id"]) for item in [*primary, *replacements]
                }
    except (KeyError, TypeError):
        exact_counts = exact_slots = exact_schema = False

    episode_ids = [str(item.get("episode_id")) for item in all_records]
    tuples = [tuple(int(item[key]) for key in SEED_KEYS) for item in all_records]
    record_numbers = [number for seed_tuple in tuples for number in seed_tuple]
    rng_records = _rng_records(ledger)
    rng_ids = [int(item["rng_id"]) for item in rng_records if "rng_id" in item]
    rng_labels = [str(item["label"]) for item in rng_records if "label" in item]
    current_numbers = set(record_numbers + rng_ids)
    current_strings = set(episode_ids + rng_labels)
    numeric_overlap = sorted(current_numbers & prior_numbers)
    string_overlap = sorted(current_strings & prior_strings)
    expected_records = len(REGIMES) * (
        sum(PRIMARY_COUNTS.values()) + len(ROLES) * REPLACEMENT_COUNT
    )

    builder_snapshot = ledger.get("prior_identifier_snapshot", {})
    snapshot_fields = (
        "metadata_file_count",
        "filename_count",
        "numeric_identifier_count",
        "string_identifier_count",
        "numeric_identifier_set_sha256",
        "string_identifier_set_sha256",
        "metadata_path_set_sha256",
        "filename_path_set_sha256",
        "required_consumed_metadata",
        "required_consumed_metadata_complete",
        "missing_required_consumed_metadata",
        "safe_v3_source_config_plan_files",
        "safe_v3_numeric_literal_count",
        "safe_v3_numeric_literal_set_sha256",
        "safe_v3_string_identifier_count",
        "safe_v3_string_identifier_set_sha256",
        "v004_authoritative_prior_snapshot",
        "v004_authoritative_prior_snapshot_cross_referenced",
    )
    snapshot_matches = all(
        builder_snapshot.get(field) == independent_snapshot.get(field)
        for field in snapshot_fields
    )

    direct_required_overlap = dict(direct_required_overlap or {})
    direct_required_clean = all(
        not item.get("missing", False)
        and not item.get("numeric_overlap", [])
        and not item.get("string_overlap", [])
        for item in direct_required_overlap.values()
    ) if direct_required_overlap else True

    analysis_seeds = ledger.get("analysis_seeds", {})
    rng_by_purpose = {
        (str(item.get("regime", "")), str(item.get("purpose"))): int(item["rng_id"])
        for item in rng_records
        if "rng_id" in item
    }
    expected_seed_aliases = {
        "joint_bootstrap_seed": rng_by_purpose.get(("", "joint_master")),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seeded_comparator_seeds": {
            regime: {
                "raw": rng_by_purpose.get((regime, "seeded_weak_more_raw")),
                "fixed_whitened": rng_by_purpose.get(
                    (regime, "seeded_weak_more_fixed_whitened")
                ),
            }
            for regime in REGIMES
        },
        "histogram_seeds": {
            regime: rng_by_purpose.get((regime, "within_episode_call_histogram"))
            for regime in REGIMES
        },
        "aliases_only_no_additional_rng_ids": True,
    }

    role_sets = list(role_episode_sets.values())
    checks = {
        "exact_regime_order": ledger.get("regime_order") == list(REGIMES),
        "exact_role_order": ledger.get("role_order") == list(ROLES),
        "exact_nested_record_schema": exact_schema,
        "exact_primary_and_replacement_counts": exact_counts,
        "all_pool_slots_ordered_and_complete": exact_slots,
        "exact_total_record_count_24424": len(all_records) == expected_records == 24_424,
        "all_episode_ids_unique": len(set(episode_ids)) == len(episode_ids),
        "all_seed_tuples_unique": len(set(tuples)) == len(tuples),
        "every_record_seed_and_rng_id_unique": (
            len(current_numbers) == len(record_numbers) + len(rng_ids)
        ),
        "all_rng_labels_unique": len(set(rng_labels)) == len(rng_labels),
        "all_dgp_role_episode_sets_disjoint": all(
            role_sets[left].isdisjoint(role_sets[right])
            for left in range(len(role_sets))
            for right in range(left + 1, len(role_sets))
        ),
        "numeric_namespace_is_fixed_uint32": bool(current_numbers)
        and min(current_numbers) >= NAMESPACE_BASE
        and max(current_numbers) < UINT32_LIMIT,
        "rng_namespace_is_separate": bool(rng_ids)
        and min(rng_ids) >= RNG_NAMESPACE_BASE
        and max(record_numbers, default=-1) < RNG_NAMESPACE_BASE,
        "zero_all_prior_numeric_overlap": not numeric_overlap,
        "zero_all_prior_string_overlap": not string_overlap,
        "zero_direct_v5_v001_v004_and_v005_overlap": direct_required_clean,
        "independent_prior_snapshot_matches_builder": snapshot_matches,
        "all_required_consumed_ledgers_scanned": bool(
            independent_snapshot.get("required_consumed_metadata_complete", False)
        ),
        "v004_authoritative_all_prior_snapshot_cross_referenced": bool(
            independent_snapshot.get(
                "v004_authoritative_prior_snapshot_cross_referenced", False
            )
        ),
        "safe_v3_source_config_plan_identifiers_audited": bool(
            independent_snapshot.get("safe_v3_source_config_plan_files")
        ),
        "confirmation_has_all_4500_ordered_slots_per_regime": all(
            len(ledger["regimes"][regime]["roles"]["confirmation"]["primary"])
            == 4_500
            for regime in REGIMES
        ) if isinstance(ledger.get("regimes"), Mapping) else False,
        "bootstrap_master_and_20000_replicates_preassigned": (
            ledger.get("bootstrap_replicates") == BOOTSTRAP_REPLICATES
            and ledger.get("analysis_rng_ids", {})
            .get("bootstrap", {})
            .get("replicate_count")
            == BOOTSTRAP_REPLICATES
            and len(
                ledger.get("analysis_rng_ids", {})
                .get("bootstrap", {})
                .get("rng_ids", [])
            )
            == 1
        ),
        "canonical_analysis_seed_aliases_exact": analysis_seeds
        == expected_seed_aliases,
        "paired_bootstrap_rng_contract_exact": (
            ledger.get("analysis_rng_ids", {})
            .get("bootstrap", {})
            .get("chunk_size_defining_rng_consumption")
            == 250
            and ledger.get("analysis_rng_ids", {})
            .get("bootstrap", {})
            .get("regime_order")
            == list(REGIMES)
            and "default_rng(joint_master)"
            in ledger.get("analysis_rng_ids", {})
            .get("bootstrap", {})
            .get("replicate_derivation", "")
        ),
        "builder_reported_zero_overlap": (
            ledger.get("numeric_overlap") == [] and ledger.get("string_overlap") == []
        ),
        "forbidden_sources_unopened": (
            ledger.get("v3_targets_opened") is False
            and ledger.get("combined_v3_cache_numpy_loaded") is False
            and ledger.get("released_hdf5_opened") is False
            and independent_snapshot.get(
                "v3_test_target_cache_split_metrics_artifacts_opened"
            )
            is False
            and independent_snapshot.get("binary_contents_opened") is False
        ),
    }
    return {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "implementation_independent_of_build_seed_ledger": True,
        "checks": checks,
        "passed": all(checks.values()),
        "record_count": len(all_records),
        "rng_identifier_count": len(rng_ids),
        "numeric_identifier_count": len(current_numbers),
        "string_identifier_count": len(current_strings),
        "current_numeric_set_sha256": _set_hash(current_numbers),
        "current_string_set_sha256": _set_hash(current_strings),
        "numeric_overlap": numeric_overlap,
        "string_overlap": string_overlap,
        "direct_required_overlap": direct_required_overlap,
        "independent_prior_snapshot": dict(independent_snapshot),
        "fresh_outcome_episodes_opened": 0,
        "prior_outcome_arrays_opened": 0,
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
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
        "--ledger", type=Path, default=ATTEMPT_ROOT / "cohort_seed_ledger.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ATTEMPT_ROOT / "audit/identifier_freshness_verification.json",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise RuntimeError(f"immutable verification already exists: {args.output}")
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    numbers, strings, snapshot = independently_scan_prior(args.runs_root)
    # Recompute the current identifier sets for the direct authoritative checks.
    records = [
        item
        for regime in REGIMES
        for role in ROLES
        for pool in ("primary", "replacements")
        for item in ledger["regimes"][regime]["roles"][role][pool]
    ]
    rng_records = _rng_records(ledger)
    current_numbers = {
        int(item[key]) for item in records for key in SEED_KEYS
    } | {int(item["rng_id"]) for item in rng_records}
    current_strings = {str(item["episode_id"]) for item in records} | {
        str(item["label"]) for item in rng_records
    }
    direct = _required_direct_overlaps(
        args.runs_root,
        current_numbers,
        current_strings,
        REQUIRED_PRIOR_METADATA,
    )
    result = verify_identifier_freshness(
        ledger,
        numbers,
        strings,
        snapshot,
        direct_required_overlap=direct,
    )
    _write_exclusive(args.output, result)
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
