#!/usr/bin/env python3
"""Exclusively capture one read-only independent-verifier result.

This wrapper performs no scientific computation and imports no project code.
It is the only verifier component permitted to write, and it can write only
``audit/independent_verification.json`` beneath the contract's attempt root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence


sys.dont_write_bytecode = True


def _canonical_relative(raw: object) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise RuntimeError("contract attempt_root is not canonical")
    value = PurePosixPath(raw)
    if value.is_absolute() or "." in value.parts or ".." in value.parts or value.as_posix() != raw:
        raise RuntimeError("contract attempt_root is not canonical")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_links(root: Path, path: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            raise RuntimeError(f"symlink component forbidden: {current}")


def _write_exclusive(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"immutable audit already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def capture(contract_path: Path) -> tuple[dict[str, object], Path]:
    contract_path = Path(contract_path)
    if contract_path.is_symlink() or not contract_path.is_file() or not stat.S_ISREG(contract_path.lstat().st_mode):
        raise RuntimeError("verifier contract must be a regular non-symlink file")
    contract_path = contract_path.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    attempt_relative = _canonical_relative(contract.get("attempt_root"))
    attempt_parts = PurePosixPath(attempt_relative).parts
    if tuple(contract_path.parent.parts[-len(attempt_parts) :]) != attempt_parts:
        raise RuntimeError("contract is not in its declared attempt root")
    repository_root = contract_path.parent
    for _ in attempt_parts:
        repository_root = repository_root.parent
    attempt_root = repository_root / attempt_relative
    output = attempt_root / "audit/independent_verification.json"
    verifier = Path(__file__).resolve().with_name("independent_verify.py")
    if not verifier.is_file() or verifier.is_symlink():
        raise RuntimeError("standalone verifier source is absent or aliased")
    _reject_links(repository_root, contract_path)
    _reject_links(repository_root, verifier)
    _reject_links(repository_root, output.parent)
    if output.exists():
        raise FileExistsError(f"immutable audit already exists: {output}")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", str(verifier), str(contract_path)],
        cwd=repository_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    expected_contract_hash = _sha256(contract_path)
    stdout = completed.stdout.decode("utf-8", errors="replace")
    lines = stdout.splitlines()
    parsed: object = None
    if len(lines) == 1:
        try:
            parsed = json.loads(lines[0])
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict):
        result: dict[str, object] = dict(parsed)
    else:
        result = {
            "schema_version": 1,
            "attempt": contract.get("attempt"),
            "mode": contract.get("mode"),
            "checkpoint_state": "INDEPENDENT_VERIFICATION",
            "passed": False,
            "error_type": "VerifierProcessOutputError",
            "error": "standalone verifier did not emit exactly one valid JSON object line",
            "read_only_verifier": True,
        }
    expected_contract_path = contract_path.relative_to(repository_root).as_posix()
    if result.get("passed") is True:
        success_contract = result.get("verifier_contract")
        success_checks = result.get("checks")
        integrity = {
            "returncode_zero": completed.returncode == 0,
            "one_json_line": len(lines) == 1 and isinstance(parsed, dict),
            "schema": result.get("schema_version") == 1,
            "attempt": result.get("attempt") == contract.get("attempt"),
            "mode": result.get("mode") == contract.get("mode"),
            "checkpoint": result.get("checkpoint_state") == "INDEPENDENT_VERIFICATION",
            "read_only": result.get("read_only_verifier") is True,
            "no_local_production_import": result.get("local_production_modules_imported") is False,
            "stdout_contract": result.get("stdout_json_only") is True,
            "checks_boolean_only": isinstance(success_checks, dict)
            and bool(success_checks)
            and all(type(value) is bool and value for value in success_checks.values()),
            "contract_record": isinstance(success_contract, dict)
            and success_contract.get("path") == expected_contract_path
            and success_contract.get("sha256") == expected_contract_hash,
        }
        if not all(integrity.values()):
            result = {
                "schema_version": 1,
                "attempt": contract.get("attempt"),
                "mode": contract.get("mode"),
                "checkpoint_state": "INDEPENDENT_VERIFICATION",
                "passed": False,
                "error_type": "VerifierSuccessIntegrityError",
                "error": "standalone verifier success record failed wrapper integrity checks: "
                + ",".join(name for name, passed in integrity.items() if not passed),
                "read_only_verifier": True,
            }
    contract_record = result.get("verifier_contract")
    if isinstance(contract_record, dict) and contract_record.get("sha256") not in (None, expected_contract_hash):
        raise RuntimeError("verifier result binds a different contract")
    result["attempt"] = contract.get("attempt")
    result["mode"] = contract.get("mode")
    result["checkpoint_state"] = "INDEPENDENT_VERIFICATION"
    result.setdefault("schema_version", 1)
    result.setdefault("read_only_verifier", True)
    result["verifier_contract"] = {
        "path": expected_contract_path,
        "sha256": expected_contract_hash,
    }
    source_hashes = result.get("source_hashes")
    if not isinstance(source_hashes, dict):
        source_hashes = {}
    source_hashes = dict(source_hashes)
    source_hashes["independent_verify"] = _sha256(verifier)
    source_hashes["capture_verifier"] = _sha256(Path(__file__).resolve())
    result["source_hashes"] = source_hashes
    result["capture"] = {
        "schema_version": 1,
        "captured_exclusively": True,
        "wrapper_integrity_passed": True,
        "verifier_returncode": int(completed.returncode),
        "verifier_stdout_line_count": len(lines),
        "verifier_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "verifier_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "verifier_stderr_bytes": len(completed.stderr),
        "scientific_verifier_passed": result.get("passed") is True,
        "independent_verify_source_sha256": _sha256(verifier),
        "capture_verifier_source_sha256": _sha256(Path(__file__).resolve()),
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    _write_exclusive(output, payload)
    return result, output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args(argv)
    try:
        result, _ = capture(args.contract)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True, separators=(",", ":")), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
