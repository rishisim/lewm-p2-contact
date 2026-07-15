#!/usr/bin/env python3
"""Small fail-closed helpers for the persistent V5-readiness state machine."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "PROGRAM_STATE.json"
POINTER_PATH = ROOT / "CURRENT_POINTER.json"
LEDGER_PATH = ROOT / "RESEARCH_LEDGER.md"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON by fsync + same-directory replace, then fsync the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_ledger(entry: str) -> None:
    if not LEDGER_PATH.exists():
        raise RuntimeError("research ledger is missing")
    prior = LEDGER_PATH.read_text()
    if not prior.endswith("\n"):
        raise RuntimeError("research ledger is malformed")
    with LEDGER_PATH.open("a") as stream:
        stream.write(entry.rstrip() + "\n\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    show = subparsers.add_parser("show")
    show.add_argument("which", choices=("state", "pointer"))
    write = subparsers.add_parser("write")
    write.add_argument("which", choices=("state", "pointer"))
    write.add_argument("source", type=Path)
    append = subparsers.add_parser("append-ledger")
    append.add_argument("source", type=Path)
    arguments = parser.parse_args()

    if arguments.command == "show":
        path = STATE_PATH if arguments.which == "state" else POINTER_PATH
        print(json.dumps(load_json(path), indent=2, sort_keys=True))
        return
    if arguments.command == "append-ledger":
        append_ledger(arguments.source.read_text())
        return
    value = load_json(arguments.source)
    path = STATE_PATH if arguments.which == "state" else POINTER_PATH
    atomic_write_json(path, value)


if __name__ == "__main__":
    main()
