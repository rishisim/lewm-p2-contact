#!/usr/bin/env python3
"""Exact record-level comparison of initial and final evidence snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from typing import Any


FIELDS = (
    "type",
    "mode_octal",
    "size",
    "mtime_ns",
    "ctime_ns",
    "uid",
    "gid",
    "inode",
    "device",
    "nlink",
    "sha256",
    "symlink_target",
    "sha256_symlink_target_utf8",
)


def read(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def file_sha(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    before = {item["path"]: item for item in left["records"]}
    after = {item["path"]: item for item in right["records"]}
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changes = []
    for path in sorted(set(before) & set(after)):
        changed = {
            field: {"before": before[path].get(field), "after": after[path].get(field)}
            for field in FIELDS
            if before[path].get(field) != after[path].get(field)
        }
        if changed:
            changes.append({"path": path, "fields": changed})
    content_changes = [
        item
        for item in changes
        if "sha256" in item["fields"]
        or "sha256_symlink_target_utf8" in item["fields"]
    ]
    stat_changes = [
        item
        for item in changes
        if any(
            field
            in {
                "type",
                "mode_octal",
                "size",
                "mtime_ns",
                "ctime_ns",
                "uid",
                "gid",
                "inode",
                "device",
                "nlink",
                "symlink_target",
            }
            for field in item["fields"]
        )
    ]
    return {
        "initial_record_count": len(before),
        "final_record_count": len(after),
        "added_paths": added,
        "removed_paths": removed,
        "changed_records": changes,
        "content_changed_records": content_changes,
        "stat_changed_records": stat_changes,
        "path_set_exact": not added and not removed,
        "content_exact": not content_changes,
        "stat_exact": not stat_changes,
        "all_record_fields_exact": not changes and not added and not removed,
        "initial_aggregate_sha256": left["summary"]["aggregate_sha256"],
        "final_aggregate_sha256": right["summary"]["aggregate_sha256"],
        "aggregate_exact": left["summary"]["aggregate_sha256"]
        == right["summary"]["aggregate_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True, type=pathlib.Path)
    parser.add_argument("--final", required=True, type=pathlib.Path)
    parser.add_argument("--initial-supplement", required=True, type=pathlib.Path)
    parser.add_argument("--final-supplement", required=True, type=pathlib.Path)
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    initial_path = args.initial.resolve()
    final_path = args.final.resolve()
    initial_supplement_path = args.initial_supplement.resolve()
    final_supplement_path = args.final_supplement.resolve()
    package = args.package.resolve()

    initial_snapshot = read(initial_path)
    final_snapshot = read(final_path)
    initial_supplement = read(initial_supplement_path)
    final_supplement = read(final_supplement_path)
    primary = compare(initial_snapshot, final_snapshot)
    supplemental = compare(initial_supplement, final_supplement)
    repo = package.parents[3]
    raw_manifest = read(package / "data/v5_confirmation_raw_manifest.json")
    raw_records = raw_manifest["episodes"]
    raw_paths = [item["path"] for item in raw_records]
    raw_candidates = [repo / relative for relative in raw_paths]
    raw_confined = True
    for candidate in raw_candidates:
        try:
            candidate.resolve(strict=True).relative_to(repo.resolve(strict=True))
        except (ValueError, FileNotFoundError):
            raw_confined = False
            break
    primary_types = {
        item["type"] for item in final_snapshot["records"]
    }
    supplement_types = {
        item["type"] for item in final_supplement["records"]
    }
    canonical_security = {
        "v001_v004_record_types": sorted(primary_types),
        "full_program_record_types": sorted(supplement_types),
        "v001_v004_symlink_count": sum(
            item["type"] == "symlink" for item in final_snapshot["records"]
        ),
        "full_program_symlink_count": sum(
            item["type"] == "symlink"
            for item in final_supplement["records"]
        ),
        "snapshot_paths_relative_and_parent_free": all(
            not pathlib.PurePath(item["path"]).is_absolute()
            and ".." not in pathlib.PurePath(item["path"]).parts
            for snapshot in (final_snapshot, final_supplement)
            for item in snapshot["records"]
        ),
        "v5_raw_paths_relative_and_parent_free": all(
            not pathlib.PurePath(relative).is_absolute()
            and ".." not in pathlib.PurePath(relative).parts
            for relative in raw_paths
        ),
        "v5_raw_paths_resolve_within_repo": raw_confined,
        "v5_raw_files_regular_not_symlink": all(
            candidate.is_file() and not candidate.is_symlink()
            for candidate in raw_candidates
        ),
        "v5_raw_paths_unique": len(set(raw_paths)) == len(raw_paths) == 1600,
        "v5_raw_identifiers_unique": len(
            {item["episode_id"] for item in raw_records}
        )
        == len(raw_records)
        == 1600,
        "v5_raw_hashes_unique": len({item["sha256"] for item in raw_records})
        == len(raw_records)
        == 1600,
    }
    decision_sha = file_sha(package / "decision.json")
    independent_sha = file_sha(package / "audit/independent_verification.json")
    checks = {
        "v001_v004_path_set_exact": primary["path_set_exact"],
        "v001_v004_content_exact": primary["content_exact"],
        "v001_v004_stat_exact": primary["stat_exact"],
        "v001_v004_aggregate_exact": primary["aggregate_exact"],
        "full_program_path_set_exact": supplemental["path_set_exact"],
        "full_program_content_exact": supplemental["content_exact"],
        "full_program_stat_exact": supplemental["stat_exact"],
        "full_program_aggregate_exact": supplemental["aggregate_exact"],
        "decision_known_sha256": decision_sha
        == "69bb8af8d80d7e18aeabc5430963c57b1bc91be603745d5a0d5f1a855d61fbe5",
        "independent_audit_known_sha256": independent_sha
        == "d30e58466184ac98c5643ea6242e1fb1d95cc2f820c001be16aee3556903c3f0",
        "canonical_snapshot_contains_only_directories_and_regular_files": primary_types
        <= {"directory", "regular_file"}
        and supplement_types <= {"directory", "regular_file"},
        "canonical_paths_relative_confined_regular_unique": all(
            value
            for key, value in canonical_security.items()
            if key
            not in {
                "v001_v004_record_types",
                "full_program_record_types",
                "v001_v004_symlink_count",
                "full_program_symlink_count",
            }
        )
        and canonical_security["v001_v004_symlink_count"] == 0
        and canonical_security["full_program_symlink_count"] == 0,
    }
    output = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "checks": checks,
        "passed": all(checks.values()),
        "v001_v004": primary,
        "full_readiness_program_supplement": supplemental,
        "snapshot_files": {
            "initial": {
                "path": str(initial_path),
                "sha256": file_sha(initial_path),
            },
            "final": {"path": str(final_path), "sha256": file_sha(final_path)},
            "initial_supplement": {
                "path": str(initial_supplement_path),
                "sha256": file_sha(initial_supplement_path),
            },
            "final_supplement": {
                "path": str(final_supplement_path),
                "sha256": file_sha(final_supplement_path),
            },
        },
        "known_terminal_hashes": {
            "decision_sha256": decision_sha,
            "independent_verification_sha256": independent_sha,
        },
        "canonical_security": canonical_security,
        "canonical_evidence_changed_during_audit": not all(
            (
                primary["all_record_fields_exact"],
                supplemental["all_record_fields_exact"],
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "passed": output["passed"],
                "canonical_evidence_changed_during_audit": output[
                    "canonical_evidence_changed_during_audit"
                ],
                "v001_v004_changes": len(primary["changed_records"]),
                "full_program_changes": len(supplemental["changed_records"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
