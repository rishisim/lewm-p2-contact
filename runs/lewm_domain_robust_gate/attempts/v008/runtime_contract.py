#!/usr/bin/env python3
"""Read-only dual-interpreter and frozen-source runtime contract.

The v001 program inherits the two exact Python environments and the frozen
model/source objects recorded by the terminal V5 v004 package.  This module is
deliberately read-only: it is safe to invoke as the first subprocess before a
seed tuple is selected or an output path is created.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ATTEMPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ATTEMPT_ROOT.parents[3]
V5_ROOT = REPO_ROOT / "runs/lewm_v5_readiness_program/v5_package_versions/v004"
V5_RUNTIME_CONTRACT_PATH = V5_ROOT / "runtime_contract.json"
V5_RUNTIME_CONTRACT_SHA256 = (
    "7d7c9d9a9a014f1b4d0ef6ee505438bde0f53295f26646c3893fb5b9c9feb4e4"
)

EVALUATION_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")
GENERATION_PYTHON = Path(
    "/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python"
)
PYTHON_VERSION = "3.10.20"

ROLE_RUNTIME = {
    "generation": "generation",
    "fit": "evaluation",
    "selection": "evaluation",
    "sparse_execution": "evaluation",
    "dense_shadow": "evaluation",
    "analysis": "evaluation",
    "latency": "evaluation",
    "resource_reporting": "evaluation",
    "qualification": "evaluation",
    "sealing": "evaluation",
    "independent_verification": "evaluation",
    "evaluation": "evaluation",
}
RUNTIME_PATHS = {
    "evaluation": EVALUATION_PYTHON,
    "generation": GENERATION_PYTHON,
}


class RuntimeContractError(RuntimeError):
    """The inherited runtime, frozen source, role, or device has drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"cannot read runtime contract: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeContractError(f"runtime contract is not an object: {path}")
    return value


def load_v5_runtime_contract() -> dict[str, Any]:
    """Load v004 only after authenticating the complete JSON byte stream."""

    if not V5_RUNTIME_CONTRACT_PATH.is_file():
        raise RuntimeContractError("terminal V5 v004 runtime contract is missing")
    observed = sha256_file(V5_RUNTIME_CONTRACT_PATH)
    if observed != V5_RUNTIME_CONTRACT_SHA256:
        raise RuntimeContractError(
            "terminal V5 v004 runtime contract hash drift: "
            f"expected {V5_RUNTIME_CONTRACT_SHA256}, got {observed}"
        )
    contract = _read_object(V5_RUNTIME_CONTRACT_PATH)
    expected_header = {
        "package_version": "v004",
        "contract": "fail_closed_dual_interpreter_dispatch",
        "dispatcher": "launcher.py",
        "manual_interpreter_selection_forbidden": True,
        "global_generation_preflight_before_seed_or_output": True,
        "wrong_interpreter_consumes_zero_tuples": True,
        "v5_outcome_episodes": 0,
    }
    bad = {
        key: {"expected": expected, "observed": contract.get(key)}
        for key, expected in expected_header.items()
        if contract.get(key) != expected
    }
    if bad:
        raise RuntimeContractError(f"V5 runtime contract header drift: {bad}")
    if set(contract.get("runtimes", {})) != set(RUNTIME_PATHS):
        raise RuntimeContractError("V5 runtime role membership drift")
    for role, expected_path in RUNTIME_PATHS.items():
        recorded = contract["runtimes"][role]
        if (
            recorded.get("sys_executable") != str(expected_path)
            or recorded.get("python_version") != PYTHON_VERSION
            or recorded.get("python_version_info") != [3, 10, 20]
        ):
            raise RuntimeContractError(f"V5 {role} interpreter record drift")
    return contract


def runtime_for_role(role: str) -> str:
    try:
        return ROLE_RUNTIME[str(role)]
    except KeyError as exc:
        raise RuntimeContractError(f"unknown sealed runtime role: {role}") from exc


def interpreter_for_role(role: str) -> Path:
    return RUNTIME_PATHS[runtime_for_role(role)]


def _package_probe(package_names: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    versions: dict[str, Any] = {}
    origins: dict[str, Any] = {}
    for package in package_names:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        specification = importlib.util.find_spec(package.replace("-", "_"))
        origins[package] = None if specification is None else specification.origin
    return versions, origins


def verify_external_hashes(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rehash every external object inherited in the exact V5 contract."""

    value = load_v5_runtime_contract() if contract is None else dict(contract)
    expected = value.get("external_file_hashes")
    if not isinstance(expected, Mapping) or not expected:
        raise RuntimeContractError("V5 runtime contract lacks external hashes")
    missing: list[str] = []
    drifted: list[dict[str, str]] = []
    total_bytes = 0
    for raw_path, raw_hash in sorted(expected.items()):
        path = Path(str(raw_path))
        if not path.is_file():
            missing.append(str(path))
            continue
        observed = sha256_file(path)
        total_bytes += path.stat().st_size
        if observed != str(raw_hash):
            drifted.append(
                {"path": str(path), "expected": str(raw_hash), "observed": observed}
            )
    result = {
        "passed": not missing and not drifted,
        "file_count": len(expected),
        "total_bytes_hashed": total_bytes,
        "missing": missing,
        "drifted": drifted,
        "v5_runtime_contract_sha256": V5_RUNTIME_CONTRACT_SHA256,
    }
    if not result["passed"]:
        raise RuntimeContractError(f"frozen external source hash failure: {result}")
    return result


def verify_here(
    role: str,
    *,
    require_mps: bool = False,
    include_external_hashes: bool = True,
) -> dict[str, Any]:
    """Verify the current process is the exact interpreter for ``role``."""

    runtime_role = runtime_for_role(role)
    contract = load_v5_runtime_contract()
    recorded = contract["runtimes"][runtime_role]
    package_names = tuple(recorded["packages"])
    versions, origins = _package_probe(package_names)
    current_executable = str(Path(sys.executable))
    checks = {
        "runtime_role": runtime_role == ROLE_RUNTIME[str(role)],
        "sys_executable": current_executable == recorded["sys_executable"],
        "python_version": platform.python_version() == recorded["python_version"],
        "python_version_info": list(sys.version_info[:3])
        == recorded["python_version_info"],
        "packages": versions == recorded["packages"],
        "module_origins": origins == recorded["module_origins"],
    }
    mps = {
        "required": bool(require_mps),
        "built": None,
        "available": None,
    }
    if require_mps:
        if runtime_role != "evaluation":
            raise RuntimeContractError("MPS may only be required under evaluation runtime")
        import torch

        mps["built"] = bool(torch.backends.mps.is_built())
        mps["available"] = bool(torch.backends.mps.is_available())
        checks["mps_built"] = bool(mps["built"])
        checks["mps_available"] = bool(mps["available"])
    external = (
        verify_external_hashes(contract)
        if include_external_hashes
        else {"passed": True, "skipped": True}
    )
    passed = all(checks.values()) and bool(external["passed"])
    result = {
        "schema_version": 1,
        "passed": passed,
        "requested_role": str(role),
        "runtime_role": runtime_role,
        "checks": checks,
        "mps": mps,
        "external_hashes": external,
        "v5_runtime_contract_path": str(V5_RUNTIME_CONTRACT_PATH),
        "v5_runtime_contract_sha256": V5_RUNTIME_CONTRACT_SHA256,
        "read_only_preflight": True,
        "seed_tuples_consumed": 0,
        "output_paths_created": 0,
    }
    if not passed:
        raise RuntimeContractError(f"runtime preflight failed closed: {result}")
    return result


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def probe_role_subprocess(
    role: str,
    *,
    require_mps: bool = False,
    include_external_hashes: bool = False,
) -> dict[str, Any]:
    """Run the read-only verifier under the interpreter assigned to ``role``."""

    command = [
        str(interpreter_for_role(role)),
        str(Path(__file__).resolve()),
        "verify-here",
        "--role",
        str(role),
    ]
    if require_mps:
        command.append("--require-mps")
    if not include_external_hashes:
        command.append("--skip-external-hashes")
    completed = subprocess.run(
        command,
        cwd=ATTEMPT_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeContractError(
            f"{role} subprocess preflight failed: {completed.stderr.strip()}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeContractError(
            f"{role} subprocess preflight emitted non-JSON stdout"
        ) from exc
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise RuntimeContractError(f"{role} subprocess preflight did not pass")
    return result


def verify_dual_interpreter_contract() -> dict[str, Any]:
    """Probe both exact runtimes and independently rehash inherited objects."""

    external = verify_external_hashes()
    probes = {
        "evaluation": probe_role_subprocess("evaluation"),
        "generation": probe_role_subprocess("generation"),
    }
    return {
        "passed": bool(external["passed"])
        and all(item.get("passed") is True for item in probes.values()),
        "runtime_probes": probes,
        "external_hashes": external,
        "manual_interpreter_selection_forbidden": True,
        "generation_preflight_before_seed_or_output": True,
        "mps_required_for_sparse_execution": True,
        "cpu_fallback_permitted": False,
    }


def require_mps_device(device: str) -> str:
    """Reject every execution-device spelling except the sealed MPS device."""

    if str(device) != "mps":
        raise RuntimeContractError(
            f"sparse/dense execution requires device='mps'; got {device!r}"
        )
    return "mps"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-here")
    verify.add_argument("--role", required=True, choices=tuple(ROLE_RUNTIME))
    verify.add_argument("--require-mps", action="store_true")
    verify.add_argument("--skip-external-hashes", action="store_true")
    subparsers.add_parser("verify-dual")
    arguments = parser.parse_args(argv)
    if arguments.command == "verify-dual":
        result = verify_dual_interpreter_contract()
    else:
        result = verify_here(
            arguments.role,
            require_mps=arguments.require_mps,
            include_external_hashes=not arguments.skip_external_hashes,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        _exit_code = main()
    except Exception as error:
        print(f"runtime contract failed closed: {error}", file=sys.stderr)
        raise
    raise SystemExit(_exit_code)
