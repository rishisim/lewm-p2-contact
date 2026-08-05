#!/usr/bin/env python3
"""Single fail-closed dispatcher for every v008 interpreter role.

No scientific entry point is invoked directly by this dispatcher.  It first
runs the read-only runtime/source preflight under the exact interpreter for the
requested role.  Real model execution is MPS-only; a CPU or automatic fallback
spelling is rejected before the target process starts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from runtime_contract import (
    ATTEMPT_ROOT,
    REPO_ROOT,
    RuntimeContractError,
    interpreter_for_role,
    require_mps_device,
    sha256_file,
    verify_here,
)


RUNTIME_CONTRACT_SCRIPT = ATTEMPT_ROOT / "runtime_contract.py"
PROGRAM_SCRIPT = ATTEMPT_ROOT / "version_forward_transaction.py"
ACTIVE_ATTEMPT = ATTEMPT_ROOT.name
if ACTIVE_ATTEMPT != "v008":
    raise RuntimeError(f"launcher.py must run from v008, got {ACTIVE_ATTEMPT!r}")
ALLOWED_SCRIPTS = frozenset(
    {
        "preseal.py",
        "generator.py",
        "runner.py",
        "analysis.py",
        "latency.py",
        "independent_verify.py",
        "capture_verifier.py",
        "checkpoints.py",
        "compile_gate.py",
        "power_analysis.py",
        "launcher.py",
        "terminal_workflow.py",
        "workflow.py",
    }
)
REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ROLES = ("fit", "selection", "smoke", "confirmation")
V5_FIXED_WHITENING_PATH = (
    REPO_ROOT
    / "runs/lewm_v5_readiness_program/v5_package_versions/v004/freeze/whitening.npz"
)
COHORT_LEDGER_PATH = ATTEMPT_ROOT / "cohort_seed_ledger.json"
VERIFIER_CONTRACT_NAMES = frozenset(
    {
        "verifier_contract.json",
        "verifier_contract_no_candidate.json",
        "verifier_contract_power_infeasible.json",
    }
)


class DispatchError(RuntimeError):
    """A role, script, device, preflight, or child command failed closed."""


def _environment() -> dict[str, str]:
    value = dict(os.environ)
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    value["PYTHONNOUSERSITE"] = "1"
    return value


def _script(name: str) -> Path:
    if name not in ALLOWED_SCRIPTS:
        raise DispatchError(f"script is not in the sealed dispatcher allowlist: {name}")
    path = ATTEMPT_ROOT / name
    if not path.is_file() or path.is_symlink():
        raise DispatchError(f"allowed script is absent or linked: {path}")
    return path


def preflight_command(role: str, *, require_mps: bool) -> list[str]:
    command = [
        str(interpreter_for_role(role)),
        str(RUNTIME_CONTRACT_SCRIPT),
        "verify-here",
        "--role",
        role,
    ]
    if require_mps:
        command.append("--require-mps")
    return command


def target_command(role: str, script: str, arguments: Sequence[str]) -> list[str]:
    return [str(interpreter_for_role(role)), str(_script(script)), *map(str, arguments)]


def _run_checked(command: Sequence[str], *, label: str, capture: bool = False) -> str:
    completed = subprocess.run(
        list(command),
        cwd=ATTEMPT_ROOT,
        env=_environment(),
        capture_output=capture,
        text=capture,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() if capture and completed.stderr else ""
        raise DispatchError(
            f"{label} failed with code {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout if capture else ""


def run_dispatched(
    role: str,
    script: str,
    arguments: Sequence[str] = (),
    *,
    require_mps: bool = False,
) -> None:
    """Preflight globally, then and only then invoke the assigned target."""

    if require_mps:
        if role not in {
            "sparse_execution",
            "dense_shadow",
            "latency",
            "resource_reporting",
        }:
            raise DispatchError(f"MPS execution flag is invalid for role {role}")
    _run_checked(
        preflight_command(role, require_mps=require_mps),
        label=f"{role} global preflight",
    )
    _run_checked(
        target_command(role, script, arguments),
        label=f"{role} {script}",
    )


def run_controller(arguments: Sequence[str]) -> None:
    """Mutate STATE/ledger only after the exact evaluation preflight passes."""

    if not PROGRAM_SCRIPT.is_file() or PROGRAM_SCRIPT.is_symlink():
        raise DispatchError("durable program controller is absent or linked")
    _run_checked(
        preflight_command("evaluation", require_mps=False),
        label="controller global preflight",
    )
    _run_checked(
        [str(interpreter_for_role("evaluation")), str(PROGRAM_SCRIPT), *arguments],
        label="durable program controller",
    )


def run_all_regimes(
    role: str,
    script: str,
    argument_factory: object,
    *,
    require_mps: bool,
) -> None:
    if not callable(argument_factory):
        raise TypeError("argument_factory must be callable")
    for regime in REGIMES:
        run_dispatched(
            role,
            script,
            argument_factory(regime),
            require_mps=require_mps,
        )


def _mps_arguments(arguments: Sequence[str], device: str) -> list[str]:
    return [*arguments, "--device", require_mps_device(device)]


def _role_aggregate_paths(role: str) -> dict[str, Path]:
    if role not in {"fit", "selection"}:
        raise DispatchError(f"aggregate paths are not defined for role {role}")
    if role == "fit":
        import fit_inheritance

        try:
            paths = fit_inheritance.aggregate_paths()
        except fit_inheritance.FitInheritanceError as error:
            raise DispatchError("inherited fit aggregate provenance failed") from error
    else:
        paths = {
            regime: ATTEMPT_ROOT / "data" / role / regime / "role.npz"
            for regime in REGIMES
        }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise DispatchError(f"{role} aggregate inputs are incomplete: {missing}")
    if any(path.is_symlink() for path in paths.values()):
        raise DispatchError(f"{role} aggregate inputs may not be symlinks")
    return paths


def _repo_hashes(paths: Sequence[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    repository = REPO_ROOT.resolve(strict=True)
    for path in sorted(paths, key=lambda item: str(item)):
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(repository).as_posix()
        except ValueError as error:
            raise DispatchError(f"input escapes repository: {path}") from error
        if relative in result:
            raise DispatchError(f"duplicate normalized input path: {relative}")
        result[relative] = sha256_file(path)
    return result


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise DispatchError(f"required immutable JSON input is absent or linked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DispatchError(f"cannot read required JSON object: {path}") from error
    if not isinstance(value, dict):
        raise DispatchError(f"required JSON input is not an object: {path}")
    return value


def _require_controller_state(expected: str) -> dict[str, object]:
    """Authenticate controller/ledger chronology before scientific mutation."""

    import study_common

    try:
        state = study_common.read_verified_controller()
    except RuntimeError as error:
        raise DispatchError("controller state/ledger verification failed") from error
    if (
        state.get("active_attempt") != ACTIVE_ATTEMPT
        or state.get("current_state") != expected
        or state.get("confirmation_terminal") is not False
    ):
        raise DispatchError(
            f"operation requires controller state {expected}, got "
            f"{state.get('current_state')}"
        )
    return state


def _attempt_path(raw: str) -> Path:
    value = Path(raw)
    if value.is_absolute() or ".." in value.parts or value.as_posix() != raw:
        raise DispatchError("worker path must be canonical and attempt-relative")
    path = (ATTEMPT_ROOT / value).resolve(strict=False)
    if not path.is_relative_to(ATTEMPT_ROOT.resolve(strict=True)):
        raise DispatchError("worker path escapes attempt")
    return path


def _worker_fit_lock() -> dict[str, object]:
    """Fit and immutably lock all 24 candidates under the exact eval runtime."""

    verify_here("fit", include_external_hashes=True)
    _require_controller_state("FIT_LOCK")
    import fit_select

    paths = _role_aggregate_paths("fit")
    whitening = fit_select.load_fixed_whitening(V5_FIXED_WHITENING_PATH)
    return fit_select.fit_and_seal(
        fit_select.load_per_dgp_npz(paths),
        whitening,
        fitted_path=ATTEMPT_ROOT / "fit/fitted_candidates.npz",
        lock_path=ATTEMPT_ROOT / "fit/fit_lock.json",
        fit_input_hashes=_repo_hashes(list(paths.values())),
        fixed_whitening_sha256=fit_select.V5_FIXED_WHITENING_SHA256,
    )


def _worker_select_gate() -> dict[str, object]:
    """Open selection only through fit_select's post-lock loader callback."""

    verify_here("selection", include_external_hashes=True)
    _require_controller_state("CANDIDATE_SELECTION")
    import fit_select

    paths = _role_aggregate_paths("selection")
    cohort_ledger = _read_json_object(COHORT_LEDGER_PATH)
    return fit_select.select_after_fit_lock(
        fitted_path=ATTEMPT_ROOT / "fit/fitted_candidates.npz",
        fit_lock_path=ATTEMPT_ROOT / "fit/fit_lock.json",
        selection_loader=lambda: fit_select.load_per_dgp_npz(paths),
        comparator_seeds=fit_select.comparator_seeds_from_ledger(cohort_ledger),
        selection_input_hashes=_repo_hashes(list(paths.values())),
        selection_ledger_path=ATTEMPT_ROOT / "selection/selection_ledger.json",
        gate_fit_path=ATTEMPT_ROOT / "freeze/gate_fit.npz",
        # Summary creation is deliberately deferred until the selected gate,
        # compiler output, and gate-freeze hash all exist.
        fit_power_summary_path=None,
        selection_power_summary_path=None,
    )


def _seal_enriched_power_summaries(fit_select_module: object) -> dict[str, str]:
    """Create or verify both identity-bound power summaries after gate freeze."""

    fitted_path = ATTEMPT_ROOT / "fit/fitted_candidates.npz"
    fit_lock_path = ATTEMPT_ROOT / "fit/fit_lock.json"
    selection_path = ATTEMPT_ROOT / "selection/selection_ledger.json"
    freeze_path = ATTEMPT_ROOT / "freeze/gate_freeze.json"
    fit_summary_path = ATTEMPT_ROOT / "metrics/fit_power_summary.json"
    selection_summary_path = ATTEMPT_ROOT / "metrics/selection_power_summary.json"
    freeze = _read_json_object(freeze_path)
    identity = {
        "attempt": ACTIVE_ATTEMPT,
        "selected_candidate_id": str(freeze["selected_candidate_id"]),
        "selected_candidate_object_sha256": str(
            freeze["selected_candidate_object_sha256"]
        ),
        "fit_lock_sha256": sha256_file(fit_lock_path),
        "selection_ledger_sha256": sha256_file(selection_path),
        "gate_freeze_sha256": sha256_file(freeze_path),
    }
    existence = (fit_summary_path.exists(), selection_summary_path.exists())
    if existence[0] != existence[1]:
        raise DispatchError("partial immutable power-summary output")
    if not existence[0]:
        fitted = fit_select_module.load_fitted_candidates(fitted_path)
        ledger = _read_json_object(selection_path)
        fit_summary, selection_summary = fit_select_module.selected_power_summaries(
            fitted, ledger
        )
        fit_summary.update(identity)
        selection_summary.update(identity)
        fit_select_module._atomic_json_exclusive(fit_summary_path, fit_summary)
        fit_select_module._atomic_json_exclusive(
            selection_summary_path, selection_summary
        )
    for path, role in (
        (fit_summary_path, "fit"),
        (selection_summary_path, "selection"),
    ):
        summary = _read_json_object(path)
        if summary.get("role") != role or any(
            summary.get(key) != value for key, value in identity.items()
        ):
            raise DispatchError(f"identity-bound power-summary drift: {path}")
    return {
        "fit_power_summary_sha256": sha256_file(fit_summary_path),
        "selection_power_summary_sha256": sha256_file(selection_summary_path),
    }


def _worker_freeze_gate() -> dict[str, object]:
    """Cross-link selection, compiler output, and the immutable gate freeze."""

    verify_here("selection", include_external_hashes=True)
    _require_controller_state("GATE_FREEZE")
    import fit_select

    freeze = fit_select.seal_gate_freeze(
        fitted_path=ATTEMPT_ROOT / "fit/fitted_candidates.npz",
        fit_lock_path=ATTEMPT_ROOT / "fit/fit_lock.json",
        selection_ledger_path=ATTEMPT_ROOT / "selection/selection_ledger.json",
        gate_fit_path=ATTEMPT_ROOT / "freeze/gate_fit.npz",
        compiled_gate_path=ATTEMPT_ROOT / "freeze/compiled_gate.npz",
        compiled_gate_manifest_path=(
            ATTEMPT_ROOT / "freeze/compiled_gate_manifest.json"
        ),
        gate_freeze_path=ATTEMPT_ROOT / "freeze/gate_freeze.json",
    )
    freeze["identity_bound_power_summaries"] = _seal_enriched_power_summaries(
        fit_select
    )
    return freeze


def _worker_compile_gate(source: str, output: str, manifest: str) -> dict[str, object]:
    verify_here("evaluation", include_external_hashes=True)
    _require_controller_state("GATE_FREEZE")
    import compile_gate

    return compile_gate.compile_gate_file(
        _attempt_path(source),
        _attempt_path(output),
        _attempt_path(manifest),
    )


def _worker_power() -> dict[str, object]:
    verify_here("evaluation", include_external_hashes=True)
    _require_controller_state("CONFIRMATION_POWER_AND_COHORT_FREEZE")
    import power_analysis

    if power_analysis.main([]) != 0:
        raise DispatchError("binding power analysis returned a nonzero status")
    return {"passed": True, "state": "CONFIRMATION_POWER_AND_COHORT_FREEZE"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument(
        "role",
        choices=(
            "generation",
            "evaluation",
            "fit",
            "selection",
            "sparse_execution",
            "analysis",
            "latency",
            "independent_verification",
        ),
    )
    preflight.add_argument("--require-mps", action="store_true")
    subparsers.add_parser("verify-dual-runtime")

    preseal = subparsers.add_parser("preseal")
    preseal.add_argument(
        "stage",
        choices=(
            "implementation",
            "qualify",
            "seal-pre-data",
            "seal-pre-selection",
            "seal-pre-confirmation",
        ),
    )
    seal_verify = subparsers.add_parser("verify-seal")
    seal_verify.add_argument(
        "stage", choices=("pre-data", "pre-selection", "pre-confirmation")
    )

    generate = subparsers.add_parser("generate")
    generate.add_argument("role", choices=ROLES)
    generate.add_argument("regime", choices=REGIMES)
    generate_role = subparsers.add_parser("generate-role")
    generate_role.add_argument("role", choices=ROLES)

    development = subparsers.add_parser("development")
    development.add_argument("role", choices=("fit", "selection"))
    development.add_argument("regime", choices=REGIMES)
    development.add_argument("--device", default="mps")
    development_role = subparsers.add_parser("development-role")
    development_role.add_argument("role", choices=("fit", "selection"))
    development_role.add_argument("--device", default="mps")

    execute = subparsers.add_parser("execute")
    execute.add_argument("role", choices=("smoke", "confirmation"))
    execute.add_argument("regime", choices=REGIMES)
    execute.add_argument("--gate", default="freeze/compiled_gate.npz")
    execute.add_argument("--device", default="mps")
    execute_role = subparsers.add_parser("execute-role")
    execute_role.add_argument("role", choices=("smoke", "confirmation"))
    execute_role.add_argument("--gate", default="freeze/compiled_gate.npz")
    execute_role.add_argument("--device", default="mps")

    compile_parser = subparsers.add_parser("compile-gate")
    compile_parser.add_argument("--source", default="freeze/gate_fit.npz")
    compile_parser.add_argument("--output", default="freeze/compiled_gate.npz")
    compile_parser.add_argument(
        "--manifest", default="freeze/compiled_gate_manifest.json"
    )
    subparsers.add_parser("fit-lock")
    subparsers.add_parser("select-gate")
    subparsers.add_parser("freeze-gate")
    # Private child commands are invoked only after the public command has
    # dispatched this same sealed source under the exact evaluation runtime.
    subparsers.add_parser("_worker-fit-lock")
    subparsers.add_parser("_worker-select-gate")
    subparsers.add_parser("_worker-freeze-gate")
    worker_compile = subparsers.add_parser("_worker-compile-gate")
    worker_compile.add_argument("--source", required=True)
    worker_compile.add_argument("--output", required=True)
    worker_compile.add_argument("--manifest", required=True)
    subparsers.add_parser("_worker-power")
    subparsers.add_parser("power")
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument(
        "stage", choices=("smoke", "generation", "execution", "seal")
    )
    checkpoint.add_argument("--expected-per-dgp", type=int)

    controller = subparsers.add_parser("controller")
    controller.add_argument("controller_arguments", nargs=argparse.REMAINDER)

    subparsers.add_parser("analyze")
    latency = subparsers.add_parser("latency")
    latency.add_argument("--device", default="mps")
    latency.add_argument("--warmups", type=int, default=2)
    latency.add_argument("--repetitions", type=int, default=7)

    verify_readonly = subparsers.add_parser("independent-verify-readonly")
    verify_readonly.add_argument("contract", type=Path)
    capture = subparsers.add_parser("capture-independent-verification")
    capture.add_argument("contract", type=Path)
    capture.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=ATTEMPT_ROOT / "audit/independent_verification.json",
    )
    terminal_manifest = subparsers.add_parser("terminal-manifest")
    terminal_manifest.add_argument("--contract", default="verifier_contract.json")
    subparsers.add_parser("terminal-finalize")
    subparsers.add_parser("terminal-reports")
    subparsers.add_parser("terminal-complete")
    workflow = subparsers.add_parser("workflow")
    workflow.add_argument(
        "action", choices=("status", "step", "resume", "verify-role")
    )
    workflow.add_argument("role", nargs="?", choices=("fit", "selection"))
    workflow.add_argument("--max-transitions", type=int)
    arguments = parser.parse_args(argv)

    if arguments.command == "preflight":
        if arguments.require_mps:
            require_mps_device("mps")
        _run_checked(
            preflight_command(arguments.role, require_mps=arguments.require_mps),
            label=f"{arguments.role} global preflight",
        )
    elif arguments.command == "verify-dual-runtime":
        _run_checked(
            [
                str(interpreter_for_role("evaluation")),
                str(RUNTIME_CONTRACT_SCRIPT),
                "verify-dual",
            ],
            label="dual-runtime verification",
        )
    elif arguments.command == "preseal":
        run_dispatched("sealing", "preseal.py", [arguments.stage])
    elif arguments.command == "verify-seal":
        run_dispatched("sealing", "preseal.py", ["verify", arguments.stage])
    elif arguments.command == "generate":
        if arguments.role == "fit":
            raise DispatchError(
                "v004 fit generation is forbidden because the fixed role is inherited read-only"
            )
        run_dispatched(
            "generation", "generator.py", [arguments.role, arguments.regime]
        )
    elif arguments.command == "generate-role":
        if arguments.role == "fit":
            raise DispatchError(
                "v004 fit generation is forbidden because the fixed role is inherited read-only"
            )
        run_all_regimes(
            "generation",
            "generator.py",
            lambda regime: [arguments.role, regime],
            require_mps=False,
        )
    elif arguments.command == "development":
        if arguments.role == "fit":
            raise DispatchError(
                "v004 fit execution is forbidden because the fixed role is inherited read-only"
            )
        run_dispatched(
            "sparse_execution",
            "runner.py",
            _mps_arguments(
                ["development", arguments.role, arguments.regime], arguments.device
            ),
            require_mps=True,
        )
    elif arguments.command == "development-role":
        if arguments.role == "fit":
            raise DispatchError(
                "v004 fit execution is forbidden because the fixed role is inherited read-only"
            )
        device = require_mps_device(arguments.device)
        run_all_regimes(
            "sparse_execution",
            "runner.py",
            lambda regime: [
                "development",
                arguments.role,
                regime,
                "--device",
                device,
            ],
            require_mps=True,
        )
    elif arguments.command == "execute":
        gate = Path(arguments.gate)
        if gate.is_absolute() or ".." in gate.parts:
            raise DispatchError("gate path must be canonical and attempt-relative")
        run_dispatched(
            "sparse_execution",
            "runner.py",
            _mps_arguments(
                [
                    "execute",
                    arguments.role,
                    arguments.regime,
                    "--gate",
                    gate.as_posix(),
                ],
                arguments.device,
            ),
            require_mps=True,
        )
    elif arguments.command == "execute-role":
        gate = Path(arguments.gate)
        if gate.is_absolute() or ".." in gate.parts:
            raise DispatchError("gate path must be canonical and attempt-relative")
        device = require_mps_device(arguments.device)
        run_all_regimes(
            "sparse_execution",
            "runner.py",
            lambda regime: [
                "execute",
                arguments.role,
                regime,
                "--gate",
                gate.as_posix(),
                "--device",
                device,
            ],
            require_mps=True,
        )
    elif arguments.command == "compile-gate":
        values = [Path(arguments.source), Path(arguments.output), Path(arguments.manifest)]
        if any(path.is_absolute() or ".." in path.parts for path in values):
            raise DispatchError("compiler paths must be canonical and attempt-relative")
        run_dispatched(
            "evaluation",
            "launcher.py",
            [
                "_worker-compile-gate",
                "--source",
                values[0].as_posix(),
                "--output",
                values[1].as_posix(),
                "--manifest",
                values[2].as_posix(),
            ],
        )
    elif arguments.command == "fit-lock":
        run_dispatched("fit", "launcher.py", ["_worker-fit-lock"])
    elif arguments.command == "select-gate":
        run_dispatched("selection", "launcher.py", ["_worker-select-gate"])
    elif arguments.command == "freeze-gate":
        run_dispatched("selection", "launcher.py", ["_worker-freeze-gate"])
    elif arguments.command == "_worker-fit-lock":
        print(json.dumps(_worker_fit_lock(), sort_keys=True))
    elif arguments.command == "_worker-select-gate":
        print(json.dumps(_worker_select_gate(), sort_keys=True))
    elif arguments.command == "_worker-freeze-gate":
        print(json.dumps(_worker_freeze_gate(), sort_keys=True))
    elif arguments.command == "_worker-compile-gate":
        print(
            json.dumps(
                _worker_compile_gate(
                    arguments.source, arguments.output, arguments.manifest
                ),
                sort_keys=True,
            )
        )
    elif arguments.command == "power":
        run_dispatched("evaluation", "launcher.py", ["_worker-power"])
    elif arguments.command == "_worker-power":
        print(json.dumps(_worker_power(), sort_keys=True))
    elif arguments.command == "checkpoint":
        checkpoint_arguments = [arguments.stage]
        if arguments.stage != "smoke":
            if arguments.expected_per_dgp is None:
                raise DispatchError("confirmation checkpoint requires --expected-per-dgp")
            checkpoint_arguments.extend(
                ["--expected-per-dgp", str(arguments.expected_per_dgp)]
            )
        elif arguments.expected_per_dgp is not None:
            raise DispatchError("smoke checkpoint has a frozen size and accepts no override")
        run_dispatched("evaluation", "checkpoints.py", checkpoint_arguments)
    elif arguments.command == "controller":
        if not arguments.controller_arguments:
            raise DispatchError("controller requires an exact controller subcommand")
        run_controller(arguments.controller_arguments)
    elif arguments.command == "analyze":
        run_dispatched("analysis", "analysis.py")
    elif arguments.command == "latency":
        run_dispatched(
            "latency",
            "latency.py",
            _mps_arguments(
                [
                    "--warmups",
                    str(arguments.warmups),
                    "--repetitions",
                    str(arguments.repetitions),
                ],
                arguments.device,
            ),
            require_mps=True,
        )
    elif arguments.command == "independent-verify-readonly":
        run_dispatched(
            "independent_verification",
            "independent_verify.py",
            [str(arguments.contract)],
        )
    elif arguments.command == "terminal-manifest":
        contract = Path(arguments.contract)
        if (
            contract.parts != (contract.name,)
            or contract.name not in VERIFIER_CONTRACT_NAMES
        ):
            raise DispatchError(
                "terminal manifest contract must be one of the three sealed "
                "attempt-root contract filenames"
            )
        run_dispatched(
            "sealing",
            "terminal_workflow.py",
            ["manifest", "--contract", contract.name],
        )
    elif arguments.command in {
        "terminal-finalize",
        "terminal-reports",
        "terminal-complete",
    }:
        action = arguments.command.removeprefix("terminal-")
        run_dispatched("sealing", "terminal_workflow.py", [action])
    elif arguments.command == "workflow":
        if arguments.action == "verify-role":
            if arguments.role is None or arguments.max_transitions is not None:
                raise DispatchError(
                    "workflow verify-role requires exactly one fit/selection role"
                )
            workflow_arguments = [arguments.action, arguments.role]
        elif arguments.action == "resume":
            if arguments.role is not None:
                raise DispatchError("workflow resume accepts no role")
            maximum = (
                32 if arguments.max_transitions is None else arguments.max_transitions
            )
            if maximum <= 0:
                raise DispatchError("workflow --max-transitions must be positive")
            workflow_arguments = [
                arguments.action,
                "--max-transitions",
                str(maximum),
            ]
        else:
            if arguments.role is not None or arguments.max_transitions is not None:
                raise DispatchError(
                    f"workflow {arguments.action} accepts no trailing arguments"
                )
            workflow_arguments = [arguments.action]
        run_dispatched("evaluation", "workflow.py", workflow_arguments)
    else:
        run_dispatched(
            "independent_verification",
            "capture_verifier.py",
            [str(arguments.contract), str(arguments.output)],
        )
    print(json.dumps({"passed": True, "command": arguments.command}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        _exit_code = main()
    except Exception as error:
        print(f"launcher failed closed: {error}", file=sys.stderr)
        raise
    raise SystemExit(_exit_code)
