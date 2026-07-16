#!/usr/bin/env python3
"""Freeze fresh excluded-smoke and untouched V5 confirmation identifiers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cycle_common import ROLE_BASE_SEEDS, ROLE_COUNTS, ROOT, atomic_json


def recursive_integers(value: Any) -> set[int]:
    output: set[int] = set()
    if isinstance(value, bool):
        return output
    if isinstance(value, int):
        output.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            output.update(recursive_integers(item))
    elif isinstance(value, list):
        for item in value:
            output.update(recursive_integers(item))
    return output


def recursive_strings(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, str):
        output.add(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            output.add(str(key))
            output.update(recursive_strings(item))
    elif isinstance(value, list):
        for item in value:
            output.update(recursive_strings(item))
    return output


def digest(values: set[Any]) -> str:
    payload = "\n".join(str(item) for item in sorted(values, key=str)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def prior_identifiers() -> tuple[set[int], set[str], dict[str, Any]]:
    runs = ROOT.parents[2]
    integers: set[int] = set()
    strings: set[str] = set()
    scanned: list[str] = []
    for path in sorted(runs.rglob("*.json")):
        if ROOT in path.parents or ".venv" in path.parts:
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        integers.update(recursive_integers(value))
        strings.update(recursive_strings(value))
        scanned.append(str(path))
    for path in runs.rglob("*"):
        if path.is_file() and ROOT not in path.parents and ".venv" not in path.parts:
            strings.add(path.name)
            strings.add(path.stem)
    snapshot = {
        "schema_version": 1,
        "scope": "all parseable prior run JSON and filenames outside V5 package v003",
        "json_file_count": len(scanned),
        "numeric_identifier_count": len(integers),
        "string_identifier_count": len(strings),
        "numeric_identifier_set_sha256": digest(integers),
        "string_identifier_set_sha256": digest(strings),
        "scanned_path_set_sha256": digest(set(scanned)),
    }
    return integers, strings, snapshot


def records(role: str, count: int, base: int) -> list[dict[str, Any]]:
    width = 4 if role == "v5_confirmation" else 3
    label = "confirmation" if role == "v5_confirmation" else "package-smoke"
    return [
        {
            "role": role,
            "slot": slot,
            "episode_id": f"v5v003-{label}-{slot:0{width}d}",
            "env_seed": base + slot,
            "policy_seed": base + 100_000 + slot,
            "oracle_np_seed": base + 200_000 + slot,
        }
        for slot in range(count)
    ]


def main() -> None:
    output = ROOT / "cohort_seed_ledger.json"
    snapshot_path = ROOT / "audit/prior_identifier_snapshot.json"
    if output.exists() or snapshot_path.exists():
        raise RuntimeError("V5 cohort ledger is immutable and already exists")
    prior_numbers, prior_strings, snapshot = prior_identifiers()
    roles = {
        role: records(role, count, ROLE_BASE_SEEDS[role])
        for role, count in ROLE_COUNTS.items()
    }
    roles["replacement"] = records("replacement", 100, 2_520_900_000)
    for slot, item in enumerate(roles["replacement"]):
        item["episode_id"] = f"v5v003-replacement-{slot:03d}"
    analysis_seeds = {
        "bootstrap_seed": 2_621_999_991,
        "histogram_seed": 2_621_888_881,
        "seeded_mixture_seed": 2_621_777_771,
        "inference_qualification_seed": 2_621_666_661,
        "preseal_torch_seed": 2_621_555_551,
    }
    numbers = {
        int(item[key])
        for cohort in roles.values()
        for item in cohort
        for key in ("env_seed", "policy_seed", "oracle_np_seed")
    } | set(analysis_seeds.values())
    strings = {item["episode_id"] for cohort in roles.values() for item in cohort}
    numeric_overlap = sorted(numbers & prior_numbers)
    string_overlap = sorted(strings & prior_strings)
    expected_numbers = 3 * sum(len(cohort) for cohort in roles.values()) + len(
        analysis_seeds
    )
    expected_strings = sum(len(cohort) for cohort in roles.values())
    if len(numbers) != expected_numbers or len(strings) != expected_strings:
        raise RuntimeError("V5 identifiers are not internally unique")
    if numeric_overlap or string_overlap:
        raise RuntimeError(
            f"V5 identifiers overlap prior records: numbers={numeric_overlap[:5]}, "
            f"strings={string_overlap[:5]}"
        )
    atomic_json(snapshot_path, snapshot, exclusive=True)
    ledger = {
        "schema_version": 1,
        "package_version": "v003",
        "roles": roles,
        **analysis_seeds,
        "package_smoke_episode_count": 12,
        "v5_confirmation_episode_count": 1600,
        "confirmation_status": "preassigned_unopened_unconsumed",
        "all_new_numeric_identifiers_unique": True,
        "all_new_string_identifiers_unique": True,
        "prior_identifier_snapshot": snapshot,
        "overlap_with_prior_recorded_numeric_identifiers": numeric_overlap,
        "overlap_with_prior_recorded_string_identifiers": string_overlap,
        "replacement_rule": "mechanical generation exception or malformed rollout only; never outcome loss contact reward or success",
        "replacement_pool_is_shared_but_each_seed_tuple_is_single_use": True,
        "freshness": "all package-smoke, confirmation, replacement, analysis, and synthetic-qualification identifiers are disjoint from every recorded prior run identifier",
        "confirmation_identifiers_must_never_be_used_for_smoke": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output, ledger, exclusive=True)
    print(json.dumps(ledger, sort_keys=True))


if __name__ == "__main__":
    main()
