#!/usr/bin/env python3
"""Build the explicit dual-interpreter contract before package sealing."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from cycle_common import (
    EVALUATION_PYTHON,
    EXPECTED_RUNTIME_PACKAGES,
    GENERATION_PYTHON,
    PYTHON_VERSION,
    REPO_ROOT,
    ROOT,
    RUNTIME_PACKAGES,
    atomic_json,
    sha256_file,
)


PROBE = r"""
import importlib.metadata
import importlib.util
import json
import platform
import sys

names = json.loads(sys.argv[1])
versions = {}
origins = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
    module_name = name.replace('-', '_')
    spec = importlib.util.find_spec(module_name)
    origins[name] = None if spec is None else spec.origin
print(json.dumps({
    'sys_executable': sys.executable,
    'python_version': platform.python_version(),
    'python_version_info': list(sys.version_info[:3]),
    'packages': versions,
    'module_origins': origins,
}, sort_keys=True))
"""


def probe(path: Path, role: str) -> dict:
    completed = subprocess.run(
        [str(path), "-c", PROBE, json.dumps(list(RUNTIME_PACKAGES))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{role} runtime probe failed: {completed.stderr}")
    value = json.loads(completed.stdout)
    if (
        value["sys_executable"] != str(path)
        or value["python_version"] != PYTHON_VERSION
        or value["packages"] != EXPECTED_RUNTIME_PACKAGES[role]
    ):
        raise RuntimeError(f"{role} runtime contract drift: {value}")
    return value


def external_files(runtimes: dict[str, dict]) -> dict[str, str]:
    paths = {
        "evaluation_python": EVALUATION_PYTHON,
        "evaluation_pyvenv_cfg": EVALUATION_PYTHON.parent.parent / "pyvenv.cfg",
        "generation_python": GENERATION_PYTHON,
        "generation_pyvenv_cfg": GENERATION_PYTHON.parent.parent / "pyvenv.cfg",
        "distribution_common": REPO_ROOT
        / "runs/lewm_adaptive_compute_distribution_contract/common.py",
        "distribution_generator": REPO_ROOT
        / "runs/lewm_adaptive_compute_distribution_contract/generator.py",
        "distribution_generator_seedfix": REPO_ROOT
        / "runs/lewm_adaptive_compute_distribution_contract/generator_seedfix.py",
        "base_model_config": REPO_ROOT
        / "runs/lewm_transfer/cube/cache/model/config.json",
        "base_model_weights": REPO_ROOT
        / "runs/lewm_transfer/cube/cache/model/weights.pt",
        "stagewise_refiner": REPO_ROOT
        / "runs/lewm_adaptive_compute_discovery/checkpoints/stagewise_seed_261102.pt",
    }
    for role, runtime in runtimes.items():
        for package, raw_path in runtime["module_origins"].items():
            if raw_path is None:
                continue
            path = Path(raw_path)
            if path.exists() and path.is_file():
                paths[f"{role}_module_{package}"] = path
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"runtime contract external files missing: {missing}")
    return {str(path): sha256_file(path) for path in sorted(set(paths.values()))}


def main() -> None:
    output = ROOT / "runtime_contract.json"
    if output.exists():
        raise RuntimeError("runtime contract is immutable and already exists")
    runtimes = {
        "evaluation": probe(EVALUATION_PYTHON, "evaluation"),
        "generation": probe(GENERATION_PYTHON, "generation"),
    }
    payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "package_version": "v002",
        "contract": "fail_closed_dual_interpreter_dispatch",
        "runtimes": runtimes,
        "role_assignment": {
            "generation": "generation",
            "sparse_execution": "evaluation",
            "dense_shadow": "evaluation",
            "analysis": "evaluation",
            "latency": "evaluation",
            "independent_verification": "evaluation",
        },
        "dispatcher": "launcher.py",
        "manual_interpreter_selection_forbidden": True,
        "global_generation_preflight_before_seed_or_output": True,
        "wrong_interpreter_consumes_zero_tuples": True,
        "external_file_hashes": external_files(runtimes),
        "v5_outcome_episodes": 0,
    }
    atomic_json(output, payload, exclusive=True)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
