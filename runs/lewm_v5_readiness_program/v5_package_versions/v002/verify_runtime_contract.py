#!/usr/bin/env python3
"""Read-only independent verifier for the v002 dual-runtime contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from cycle_common import REPO_ROOT, ROOT, read_json, sha256_file


def probe(executable: str, names: list[str]) -> dict:
    code = (
        "import importlib.metadata as m,json,platform,sys;"
        "names=json.loads(sys.argv[1]);versions={};"
        "exec(\"for n in names:\\n try: versions[n]=m.version(n)\\n "
        "except m.PackageNotFoundError: versions[n]=None\");"
        "print(json.dumps({'sys_executable':sys.executable,"
        "'python_version':platform.python_version(),'packages':versions},sort_keys=True))"
    )
    completed = subprocess.run(
        [executable, "-c", code, json.dumps(names)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"runtime probe failed for {executable}: {completed.stderr}")
    return json.loads(completed.stdout)


def verify() -> dict:
    contract = read_json(ROOT / "runtime_contract.json")
    names = list(contract["runtimes"]["evaluation"]["packages"])
    bad_external = [
        raw_path
        for raw_path, expected in contract["external_file_hashes"].items()
        if not Path(raw_path).exists() or sha256_file(Path(raw_path)) != expected
    ]
    runtime_checks = {}
    for role in ("evaluation", "generation"):
        recorded = contract["runtimes"][role]
        observed = probe(recorded["sys_executable"], names)
        runtime_checks[role] = {
            "sys_executable": observed["sys_executable"] == recorded["sys_executable"],
            "python_version": observed["python_version"] == recorded["python_version"],
            "packages": observed["packages"] == recorded["packages"],
        }
    checks = {
        "package_version": contract["package_version"] == "v002",
        "dispatcher": contract["dispatcher"] == "launcher.py",
        "manual_selection_forbidden": contract["manual_interpreter_selection_forbidden"]
        is True,
        "role_assignment": contract["role_assignment"]
        == {
            "generation": "generation",
            "sparse_execution": "evaluation",
            "dense_shadow": "evaluation",
            "analysis": "evaluation",
            "latency": "evaluation",
            "independent_verification": "evaluation",
        },
        "external_hashes": not bad_external,
        "runtime_probes": all(
            all(values.values()) for values in runtime_checks.values()
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "runtime_checks": runtime_checks,
        "bad_external_hashes": bad_external,
        "runtime_contract_sha256": sha256_file(ROOT / "runtime_contract.json"),
        "v5_outcome_episodes": 0,
    }
    if not result["passed"]:
        raise RuntimeError(f"runtime contract verification failed: {result}")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
