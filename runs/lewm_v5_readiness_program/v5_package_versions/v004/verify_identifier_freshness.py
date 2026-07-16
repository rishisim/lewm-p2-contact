#!/usr/bin/env python3
"""Independently recompute v004 identifier freshness against all prior records."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

from cycle_common import REPO_ROOT, ROOT, assert_runtime_contract, atomic_json, read_json


def walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def digest(values: set[Any]) -> str:
    encoded = "\n".join(map(str, sorted(values, key=str))) + "\n"
    return hashlib.sha256(encoded.encode()).hexdigest()


def external_identifiers() -> tuple[set[int], set[str], list[str]]:
    runs = REPO_ROOT / "runs"
    numbers: set[int] = set()
    strings: set[str] = set()
    scanned: list[str] = []
    for path in sorted(runs.rglob("*.json")):
        if ROOT in path.parents or ".venv" in path.parts:
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for item in walk(value):
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                numbers.add(item)
            elif isinstance(item, str):
                strings.add(item)
        scanned.append(str(path))
    for path in runs.rglob("*"):
        if path.is_file() and ROOT not in path.parents and ".venv" not in path.parts:
            strings.update((path.name, path.stem))
    return numbers, strings, scanned


def main() -> None:
    assert_runtime_contract("evaluation")
    output = ROOT / "audit/identifier_freshness_verification.json"
    if output.exists():
        raise RuntimeError("identifier freshness verification is already immutable")
    ledger = read_json(ROOT / "cohort_seed_ledger.json")
    cohorts = ledger["roles"]
    all_records = [item for role in cohorts.values() for item in role]
    ids = [str(item["episode_id"]) for item in all_records]
    tuples = [
        (int(item["env_seed"]), int(item["policy_seed"]), int(item["oracle_np_seed"]))
        for item in all_records
    ]
    numbers = {value for triple in tuples for value in triple}
    seed_names = (
        "bootstrap_seed",
        "histogram_seed",
        "seeded_mixture_seed",
        "inference_qualification_seed",
        "preseal_torch_seed",
    )
    numbers.update(int(ledger[name]) for name in seed_names)
    prior_numbers, prior_strings, scanned = external_identifiers()
    v001 = read_json(
        ROOT.parent / "v001/cohort_seed_ledger.json"
    )
    v001_records = [item for role in v001["roles"].values() for item in role]
    v001_ids = {str(item["episode_id"]) for item in v001_records}
    v001_tuples = {
        (int(item["env_seed"]), int(item["policy_seed"]), int(item["oracle_np_seed"]))
        for item in v001_records
    }
    v002 = read_json(ROOT.parent / "v002/cohort_seed_ledger.json")
    v002_records = [item for role in v002["roles"].values() for item in role]
    v002_ids = {str(item["episode_id"]) for item in v002_records}
    v002_tuples = {
        (int(item["env_seed"]), int(item["policy_seed"]), int(item["oracle_np_seed"]))
        for item in v002_records
    }
    v003 = read_json(ROOT.parent / "v003/cohort_seed_ledger.json")
    v003_records = [item for role in v003["roles"].values() for item in role]
    v003_ids = {str(item["episode_id"]) for item in v003_records}
    v003_tuples = {
        (int(item["env_seed"]), int(item["policy_seed"]), int(item["oracle_np_seed"]))
        for item in v003_records
    }
    role_sets = {
        role: {str(item["episode_id"]) for item in records}
        for role, records in cohorts.items()
    }
    role_numeric_sets = {
        role: {
            int(item[key])
            for item in records
            for key in ("env_seed", "policy_seed", "oracle_np_seed")
        }
        for role, records in cohorts.items()
    }
    checks = {
        "exact_role_counts": {role: len(records) for role, records in cohorts.items()}
        == {"package_smoke": 12, "v5_confirmation": 1600, "replacement": 100},
        "exact_total_tuple_count_1712": len(all_records) == 1712,
        "all_episode_ids_unique": len(set(ids)) == len(ids),
        "all_seed_tuples_unique": len(set(tuples)) == len(tuples),
        "every_numeric_seed_unique": len(numbers) == 3 * len(tuples) + len(seed_names),
        "string_roles_disjoint": all(
            role_sets[left].isdisjoint(role_sets[right])
            for index, left in enumerate(role_sets)
            for right in list(role_sets)[index + 1 :]
        ),
        "numeric_roles_disjoint": all(
            role_numeric_sets[left].isdisjoint(role_numeric_sets[right])
            for index, left in enumerate(role_numeric_sets)
            for right in list(role_numeric_sets)[index + 1 :]
        ),
        "zero_overlap_all_prior_numeric_records": not (numbers & prior_numbers),
        "zero_overlap_all_prior_string_records": not (set(ids) & prior_strings),
        "v001_exactly_1712_consumed_tuples_seen": len(v001_records) == 1712
        and len(v001_tuples) == 1712,
        "zero_overlap_v001_episode_ids": not (set(ids) & v001_ids),
        "zero_overlap_v001_seed_tuples": not (set(tuples) & v001_tuples),
        "v002_exactly_1712_consumed_tuples_seen": len(v002_records) == 1712
        and len(v002_tuples) == 1712,
        "zero_overlap_v002_episode_ids": not (set(ids) & v002_ids),
        "zero_overlap_v002_seed_tuples": not (set(tuples) & v002_tuples),
        "v003_exactly_1712_consumed_tuples_seen": len(v003_records) == 1712
        and len(v003_tuples) == 1712,
        "zero_overlap_v003_episode_ids": not (set(ids) & v003_ids),
        "zero_overlap_v003_seed_tuples": not (set(tuples) & v003_tuples),
        "confirmation_never_smoke": role_sets["v5_confirmation"].isdisjoint(
            role_sets["package_smoke"] | role_sets["replacement"]
        ),
        "builder_reported_zero_overlap": not ledger[
            "overlap_with_prior_recorded_numeric_identifiers"
        ]
        and not ledger["overlap_with_prior_recorded_string_identifiers"],
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "implementation_independent_of_build_seed_ledger": True,
        "checks": checks,
        "passed": all(checks.values()),
        "prior_json_file_count": len(scanned),
        "prior_numeric_set_sha256": digest(prior_numbers),
        "prior_string_set_sha256": digest(prior_strings),
        "v004_numeric_set_sha256": digest(numbers),
        "v004_string_set_sha256": digest(set(ids)),
        "numeric_overlap": sorted(numbers & prior_numbers),
        "string_overlap": sorted(set(ids) & prior_strings),
        "v5_outcome_episodes": 0,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"identifier freshness verification failed: {result}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
