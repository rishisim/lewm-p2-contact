#!/usr/bin/env python3
"""Hash/stat one evidence tree without following symlinks."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import pathlib
import stat


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()

    paths = [root]
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        base = pathlib.Path(dirpath)
        paths.extend(base / name for name in dirnames)
        paths.extend(base / name for name in filenames)
    paths = sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())

    records = []
    regular_jobs = []
    for index, path in enumerate(paths):
        info = path.lstat()
        kind = (
            "regular_file"
            if stat.S_ISREG(info.st_mode)
            else "directory"
            if stat.S_ISDIR(info.st_mode)
            else "symlink"
            if stat.S_ISLNK(info.st_mode)
            else "other"
        )
        record = {
            "path": "." if path == root else path.relative_to(root).as_posix(),
            "type": kind,
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
        if kind == "symlink":
            target = os.readlink(path)
            record["symlink_target"] = target
            record["sha256_symlink_target_utf8"] = hashlib.sha256(
                target.encode("utf-8", errors="surrogateescape")
            ).hexdigest()
        if kind == "regular_file":
            regular_jobs.append((index, path))
        records.append(record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(digest, path): index for index, path in regular_jobs}
        for future in concurrent.futures.as_completed(futures):
            records[futures[future]]["sha256"] = future.result()

    lines = []
    counts: dict[str, int] = {}
    total_bytes = 0
    for record in records:
        counts[record["type"]] = counts.get(record["type"], 0) + 1
        if record["type"] == "regular_file":
            total_bytes += record["size"]
        item_digest = record.get(
            "sha256", record.get("sha256_symlink_target_utf8", "-")
        )
        lines.append(
            "\t".join(
                [
                    record["path"],
                    record["type"],
                    str(record["size"]),
                    str(record["mtime_ns"]),
                    record["mode_octal"],
                    item_digest,
                ]
            )
        )
    aggregate = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    result = {
        "schema_version": 1,
        "snapshot_role": args.role,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root),
        "hash_algorithm": "sha256",
        "stat_semantics": "lstat; symlinks not followed",
        "summary": {
            "record_count": len(records),
            "type_counts": counts,
            "total_regular_bytes": total_bytes,
            "aggregate_sha256": aggregate,
            "aggregate_definition": (
                "sha256 of UTF-8 sorted lines "
                "path<TAB>type<TAB>size<TAB>mtime_ns<TAB>mode_octal<TAB>content_or_link_sha256"
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
