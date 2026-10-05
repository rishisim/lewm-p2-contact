#!/usr/bin/env python3
"""Fail-closed dual-runtime dispatcher for the complete v005 workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from study_common import (
    ATTEMPT_ROOT,
    EVALUATION_PYTHON,
    GENERATION_PYTHON,
    REGIMES,
    STUDY_ROOT,
    read_json,
)


def run(role: str, script: str, *arguments: str) -> None:
    interpreter = (
        EVALUATION_PYTHON if role == "evaluation" else GENERATION_PYTHON
    )
    command = [str(interpreter), str(ATTEMPT_ROOT / script), *arguments]
    completed = subprocess.run(command, cwd=ATTEMPT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"dispatcher command failed: {command}")


def preseal() -> None:
    run("evaluation", "preseal.py", "implementation")
    run("evaluation", "preseal.py", "qualify")
    run("evaluation", "preseal.py", "seal")
    run("generation", "preseal.py", "verify-seal")


def smoke(device: str) -> None:
    run("evaluation", "preseal.py", "verify-seal")
    run("generation", "preseal.py", "verify-seal")
    for regime in REGIMES:
        run("generation", "generator.py", "smoke", regime)
        run("evaluation", "runner.py", "smoke", regime, "--device", device)
    run("evaluation", "checkpoints.py", "smoke")


def target_generation() -> None:
    for regime in REGIMES:
        run("generation", "generator.py", "target", regime)
    run("evaluation", "checkpoints.py", "generation")


def target_execution(device: str) -> None:
    for regime in REGIMES:
        run("evaluation", "runner.py", "target", regime, "--device", device)
    run("evaluation", "checkpoints.py", "execution")


def finish(device: str) -> None:
    run("evaluation", "analysis.py")
    run("evaluation", "latency.py", "--device", device)
    if not (ATTEMPT_ROOT / "decision.json").exists():
        run("evaluation", "independent_verify.py")
    run("evaluation", "posthoc.py")
    run("evaluation", "finalize.py")


def workflow(device: str) -> None:
    state = read_json(STUDY_ROOT / "STATE.json")
    current = state["current_state"]
    order = [
        "IMPLEMENT_PACKAGE",
        "PRESEAL_QUALIFY",
        "PRE_OUTCOME_SEAL",
        "EXCLUDED_REGIME_SMOKE",
        "FRESH_COHORT_GENERATION",
        "SPARSE_DENSE_EXECUTION",
        "SEALED_ANALYSIS",
        "LATENCY_AND_RESOURCE_REPORTING",
        "INDEPENDENT_VERIFICATION",
        "TERMINAL",
    ]
    if current in order[:3]:
        preseal()
        current = "EXCLUDED_REGIME_SMOKE"
    if current == "EXCLUDED_REGIME_SMOKE":
        smoke(device)
        current = "FRESH_COHORT_GENERATION"
    if current == "FRESH_COHORT_GENERATION":
        target_generation()
        current = "SPARSE_DENSE_EXECUTION"
    if current == "SPARSE_DENSE_EXECUTION":
        target_execution(device)
        current = "SEALED_ANALYSIS"
    if current in (
        "SEALED_ANALYSIS",
        "LATENCY_AND_RESOURCE_REPORTING",
        "INDEPENDENT_VERIFICATION",
        "TERMINAL",
    ):
        finish(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "preseal",
            "smoke",
            "generate-target",
            "execute-target",
            "finish",
            "workflow",
            "status",
        ),
    )
    parser.add_argument("--device", default="mps")
    arguments = parser.parse_args()
    if arguments.command == "preseal":
        preseal()
    elif arguments.command == "smoke":
        smoke(arguments.device)
    elif arguments.command == "generate-target":
        target_generation()
    elif arguments.command == "execute-target":
        target_execution(arguments.device)
    elif arguments.command == "finish":
        finish(arguments.device)
    elif arguments.command == "workflow":
        workflow(arguments.device)
    else:
        print(json.dumps(read_json(STUDY_ROOT / "STATE.json"), sort_keys=True))


if __name__ == "__main__":
    main()
