#!/usr/bin/env python3
"""Fail-closed dispatcher for the complete sealed v003 workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVALUATION_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")
GENERATION_PYTHON = Path(
    "/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python"
)


def interpreter(role: str) -> Path:
    if role == "evaluation":
        return EVALUATION_PYTHON
    if role == "generation":
        return GENERATION_PYTHON
    raise RuntimeError(f"unknown dispatcher role: {role}")


def run(role: str, script: str, *arguments: str) -> None:
    command = [str(interpreter(role)), str(ROOT / script), *arguments]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"sealed dispatcher command failed under {role} runtime: {command}"
        )


def run_if_missing(role: str, script: str, output: str, *arguments: str) -> None:
    if (ROOT / output).exists():
        print(json.dumps({"already_complete": output, "role": role}), flush=True)
        return
    run(role, script, *arguments)


def verify_preseal_both() -> None:
    run("evaluation", "verify_runtime_contract.py")
    run("generation", "verify_runtime_contract.py")
    run("evaluation", "verify_package_sources.py")
    run("generation", "verify_package_sources.py")


def verify_seal_both() -> None:
    run("evaluation", "verify_pre_v5.py")
    run("generation", "verify_pre_v5.py")


def verify_seal_for(role: str) -> None:
    run(role, "verify_pre_v5.py")


def preseal_qualify() -> None:
    run_if_missing("evaluation", "build_runtime_contract.py", "runtime_contract.json")
    run_if_missing("evaluation", "build_seed_ledger.py", "cohort_seed_ledger.json")
    run_if_missing(
        "evaluation",
        "verify_identifier_freshness.py",
        "audit/identifier_freshness_verification.json",
    )
    run_if_missing("evaluation", "power_analysis.py", "power_analysis.json")
    run_if_missing("evaluation", "preseal_tests.py", "audit/preseal_tests.json")
    run_if_missing(
        "evaluation",
        "derive_flops_graph.py",
        "audit/flop_derivation_graph.json",
    )
    run_if_missing(
        "evaluation",
        "derive_flops_symbolic.py",
        "audit/flop_derivation_symbolic.json",
    )
    run_if_missing(
        "evaluation", "qualify_runtime.py", "audit/runtime_qualification.json"
    )
    run_if_missing(
        "evaluation", "qualify_inference.py", "audit/inference_qualification.json"
    )
    run_if_missing(
        "generation",
        "qualify_generation_runtime.py",
        "audit/generation_runtime_qualification.json",
    )
    run_if_missing(
        "evaluation",
        "test_generation_preflight.py",
        "audit/generation_negative_path_qualification.json",
    )
    run_if_missing(
        "evaluation", "build_carry_forward_manifest.py", "carry_forward_manifest.json"
    )
    run_if_missing(
        "evaluation",
        "record_package_source_qualification.py",
        "audit/package_source_qualification.json",
    )
    verify_preseal_both()


def seal() -> None:
    if not (ROOT / "audit/pre_v5_seal.json").exists():
        run("generation", "prepare_package.py")
    verify_seal_both()


def workflow_smoke(device: str) -> None:
    verify_seal_both()
    run("generation", "runner.py", "generate", "package_smoke")
    run("evaluation", "runner.py", "execute", "package_smoke", "--device", device)
    run_if_missing(
        "evaluation",
        "latency.py",
        "metrics/package_smoke_latency.json",
        "package_smoke",
        "--device",
        device,
    )
    run_if_missing(
        "evaluation",
        "verify_package_smoke.py",
        "audit/package_smoke_verification.json",
    )
    run_if_missing("evaluation", "finalize_package.py", "PACKAGE_READY.json")
    run("evaluation", "verify_final_package.py")


def workflow_confirmation(device: str) -> None:
    verify_seal_both()
    run("generation", "runner.py", "generate", "v5_confirmation")
    run("evaluation", "runner.py", "execute", "v5_confirmation", "--device", device)
    run_if_missing("evaluation", "analysis.py", "analysis_result.json")
    run_if_missing(
        "evaluation",
        "latency.py",
        "metrics/v5_confirmation_latency.json",
        "v5_confirmation",
        "--device",
        device,
    )
    run_if_missing(
        "evaluation",
        "independent_verify.py",
        "decision.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preseal-qualify")
    subparsers.add_parser("verify-preseal")
    subparsers.add_parser("seal")
    subparsers.add_parser("verify-seal")
    smoke = subparsers.add_parser("workflow-smoke")
    smoke.add_argument("--device", default="mps")
    confirmation = subparsers.add_parser("workflow-confirmation")
    confirmation.add_argument("--device", default="mps")
    generate = subparsers.add_parser("generate")
    generate.add_argument("role", choices=("package_smoke", "v5_confirmation"))
    execute = subparsers.add_parser("execute")
    execute.add_argument("role", choices=("package_smoke", "v5_confirmation"))
    execute.add_argument("--device", default="mps")
    latency = subparsers.add_parser("latency")
    latency.add_argument("role", choices=("package_smoke", "v5_confirmation"))
    latency.add_argument("--device", default="mps")
    subparsers.add_parser("analyze")
    subparsers.add_parser("independent-audit")
    args = parser.parse_args()

    if args.command == "preseal-qualify":
        preseal_qualify()
    elif args.command == "verify-preseal":
        verify_preseal_both()
    elif args.command == "seal":
        seal()
    elif args.command == "verify-seal":
        verify_seal_both()
    elif args.command == "workflow-smoke":
        workflow_smoke(args.device)
    elif args.command == "workflow-confirmation":
        workflow_confirmation(args.device)
    elif args.command == "generate":
        verify_seal_for("generation")
        run("generation", "runner.py", "generate", args.role)
    elif args.command == "execute":
        verify_seal_for("evaluation")
        run("evaluation", "runner.py", "execute", args.role, "--device", args.device)
    elif args.command == "latency":
        verify_seal_for("evaluation")
        run("evaluation", "latency.py", args.role, "--device", args.device)
    elif args.command == "analyze":
        verify_seal_for("evaluation")
        run("evaluation", "analysis.py")
    else:
        verify_seal_for("evaluation")
        run("evaluation", "independent_verify.py")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"launcher failed closed: {error}", file=sys.stderr)
        raise
