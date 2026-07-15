#!/usr/bin/env python3
"""Run and retain focused V4 tests at each frozen protocol boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import common


def run(phase: str) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(common.ROOT / "tests"),
    ]
    completed = subprocess.run(
        command,
        cwd=common.REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    result = {
        "schema_version": 1,
        "phase": phase,
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output": completed.stdout,
        "frozen_objects": common.verify_frozen_objects(),
        "v3_test_targets_opened": False,
    }
    output = common.ROOT / "logs" / f"{phase}_tests.json"
    common.write_json(output, result)
    print(completed.stdout, end="")
    if not result["passed"]:
        raise SystemExit(completed.returncode)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase0", "preconfirmation", "final"))
    args = parser.parse_args()
    result = run(args.phase)
    print(json.dumps({"passed": result["passed"], "phase": args.phase}, sort_keys=True))
