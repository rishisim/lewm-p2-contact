#!/usr/bin/env python3
"""Hash and stat immutable V5 evidence without following symlinks."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import stat
import subprocess
import sys
from typing import Any


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stat_record(path: pathlib.Path, common_root: pathlib.Path) -> dict[str, Any]:
    info = path.lstat()
    relative = path.relative_to(common_root).as_posix()
    record: dict[str, Any] = {
        "path": relative,
        "type": (
            "regular_file"
            if stat.S_ISREG(info.st_mode)
            else "directory"
            if stat.S_ISDIR(info.st_mode)
            else "symlink"
            if stat.S_ISLNK(info.st_mode)
            else "other"
        ),
        "mode_octal": oct(stat.S_IMODE(info.st_mode)),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "inode": info.st_ino,
        "device": info.st_dev,
        "nlink": info.st_nlink,
    }
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(path)
        record["symlink_target"] = target
        record["sha256_symlink_target_utf8"] = hashlib.sha256(
            target.encode("utf-8", errors="surrogateescape")
        ).hexdigest()
    return record


def git_value(repo: pathlib.Path, args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode:
        return None
    return completed.stdout.rstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions-root", type=pathlib.Path, required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    versions_root = args.versions_root.resolve()
    package_roots = [versions_root / f"v{number:03d}" for number in range(1, 5)]
    missing = [str(path) for path in package_roots if not path.is_dir()]
    if missing:
        raise SystemExit(f"missing evidence roots: {missing}")

    paths: list[pathlib.Path] = []
    for root in package_roots:
        paths.append(root)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = pathlib.Path(dirpath)
            dirnames.sort()
            filenames.sort()
            paths.extend(base / name for name in dirnames)
            paths.extend(base / name for name in filenames)
    paths = sorted(set(paths), key=lambda item: item.relative_to(versions_root).as_posix())

    records = [stat_record(path, versions_root) for path in paths]
    regular_indices = [
        index for index, record in enumerate(records) if record["type"] == "regular_file"
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(sha256_file, paths[index]): index for index in regular_indices
        }
        for future in concurrent.futures.as_completed(futures):
            records[futures[future]]["sha256"] = future.result()

    canonical_lines = []
    for record in records:
        digest = record.get("sha256", record.get("sha256_symlink_target_utf8", "-"))
        canonical_lines.append(
            "\t".join(
                [
                    record["path"],
                    record["type"],
                    str(record["size"]),
                    str(record["mtime_ns"]),
                    record["mode_octal"],
                    digest,
                ]
            )
        )
    aggregate_sha256 = hashlib.sha256(
        ("\n".join(canonical_lines) + "\n").encode("utf-8")
    ).hexdigest()

    type_counts: dict[str, int] = {}
    total_regular_bytes = 0
    for record in records:
        type_counts[record["type"]] = type_counts.get(record["type"], 0) + 1
        if record["type"] == "regular_file":
            total_regular_bytes += record["size"]

    snapshot = {
        "schema_version": 1,
        "snapshot_role": "pre_audit_source_snapshot",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "stat_semantics": "lstat; symlinks not followed",
        "versions_root": str(versions_root),
        "package_roots": [str(path) for path in package_roots],
        "source_repo": {
            "path": str(args.repo_root.resolve()),
            "head": git_value(args.repo_root, ["rev-parse", "HEAD"]),
            "branch": git_value(args.repo_root, ["branch", "--show-current"]),
            "status_porcelain_v1": git_value(
                args.repo_root, ["status", "--porcelain=v1", "--untracked-files=no"]
            ),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "summary": {
            "record_count": len(records),
            "type_counts": type_counts,
            "total_regular_bytes": total_regular_bytes,
            "aggregate_sha256": aggregate_sha256,
            "aggregate_definition": (
                "sha256 of UTF-8 sorted lines "
                "path<TAB>type<TAB>size<TAB>mtime_ns<TAB>mode_octal<TAB>content_or_link_sha256"
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
