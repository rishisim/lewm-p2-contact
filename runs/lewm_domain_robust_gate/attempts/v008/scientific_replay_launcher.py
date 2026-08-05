#!/usr/bin/env python3
"""Exact-runtime launcher for the read-only v008 scientific replay qualifier."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from runtime_contract import EVALUATION_PYTHON, verify_here


ATTEMPT_ROOT = Path(__file__).resolve().parent
SCRIPT_PATH = Path(__file__).resolve()
REPLAY_PATH = ATTEMPT_ROOT / "scientific_replay.py"
ROLES = ("fit", "selection", "smoke", "confirmation")
REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)


class ReplayLaunchError(RuntimeError):
    """The exact replay runtime or its single JSON result failed closed."""


def _environment() -> dict[str, str]:
    value = dict(os.environ)
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    value["PYTHONNOUSERSITE"] = "1"
    return value


def _parse_single_json(stdout: str, label: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    if len(lines) != 1:
        raise ReplayLaunchError(f"{label} did not emit exactly one JSON line")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ReplayLaunchError(f"{label} emitted invalid JSON") from error
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise ReplayLaunchError(f"{label} did not return a passing object")
    return value


def _run_exact_subprocess(role: str, regime: str) -> dict[str, Any]:
    command = [
        str(EVALUATION_PYTHON),
        "-B",
        str(SCRIPT_PATH),
        "--role",
        role,
        "--regime",
        regime,
        "--inside-exact-runtime",
    ]
    completed = subprocess.run(
        command,
        cwd=ATTEMPT_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ReplayLaunchError(
            "scientific replay exact-runtime subprocess failed: "
            f"{completed.stderr.strip()}"
        )
    return _parse_single_json(completed.stdout, "scientific replay subprocess")


def run_qualification(
    role: str,
    regime: str,
    *,
    inside_exact_runtime: bool = False,
) -> dict[str, Any]:
    """Return one deterministic result; dispatch first when called elsewhere."""

    if role not in ROLES or regime not in REGIMES:
        raise ReplayLaunchError(f"invalid replay identity: {role}/{regime}")
    if Path(sys.executable).resolve() != EVALUATION_PYTHON.resolve():
        if inside_exact_runtime:
            raise ReplayLaunchError("inside-runtime replay used the wrong interpreter")
        return _run_exact_subprocess(role, regime)
    if not inside_exact_runtime:
        # Always isolate the model replay in a child, even when the caller happens
        # to use the evaluation interpreter.  This prevents module/cache state in
        # the workflow process from influencing qualification.
        return _run_exact_subprocess(role, regime)
    if not REPLAY_PATH.is_file() or REPLAY_PATH.is_symlink():
        raise ReplayLaunchError("scientific replay source is absent or linked")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        runtime = verify_here(
            "independent_verification",
            require_mps=True,
            include_external_hashes=True,
        )
        import scientific_replay

        result = scientific_replay.qualify_regime(
            role, regime, runtime_audit=runtime
        )
    if captured.getvalue():
        raise ReplayLaunchError("scientific replay dependency emitted unexpected stdout")
    if not isinstance(result, Mapping) or result.get("passed") is not True:
        raise ReplayLaunchError("scientific replay did not pass")
    return dict(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--regime", required=True, choices=REGIMES)
    parser.add_argument("--inside-exact-runtime", action="store_true")
    arguments = parser.parse_args(argv)
    result = run_qualification(
        arguments.role,
        arguments.regime,
        inside_exact_runtime=arguments.inside_exact_runtime,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
