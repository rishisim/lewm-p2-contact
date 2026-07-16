#!/usr/bin/env python3
"""Isolated negative test: wrong runtime fails once before tuple or raw output."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from cycle_common import EVALUATION_PYTHON, ROOT, assert_runtime_contract, atomic_json


def main() -> None:
    assert_runtime_contract("evaluation")
    output = ROOT / "audit/generation_negative_path_qualification.json"
    if output.exists():
        raise RuntimeError("generation negative-path qualification is already immutable")
    with tempfile.TemporaryDirectory(prefix="lewm-v5v002-wrong-runtime-") as raw:
        temporary = Path(raw)
        for name in ("runner.py", "cycle_common.py", "counted_features.py", "input_loader.py"):
            shutil.copy2(ROOT / name, temporary / name)
        completed = subprocess.run(
            [str(EVALUATION_PYTHON), str(temporary / "runner.py"), "generate", "package_smoke"],
            cwd=temporary,
            capture_output=True,
            text=True,
            check=False,
        )
        package_failures = sorted(
            (temporary / "audit").glob("package_smoke_generation_preflight_failure.json")
        ) if (temporary / "audit").exists() else []
        failure = json.loads(package_failures[0].read_text()) if len(package_failures) == 1 else {}
        checks = {
            "wrong_interpreter_rejected": completed.returncode != 0,
            "contract_violation_reported": "generation interpreter contract violation"
            in completed.stderr,
            "exactly_one_package_level_error": len(package_failures) == 1,
            "failure_scope_package_level": failure.get("failure_scope")
            == "package_level_generation_preflight",
            "zero_attempted_seed_tuples": failure.get("attempted_seed_tuple_count") == 0,
            "zero_consumed_identifiers": failure.get(
                "primary_or_replacement_identifiers_consumed"
            )
            == 0,
            "no_role_raw_directory": not (temporary / "data/package_smoke_raw").exists(),
            "no_per_seed_failure_ledger": not (
                temporary / "audit/package_smoke_generation_failures.json"
            ).exists(),
            "no_cohort_ledger_copied_or_read": not (
                temporary / "cohort_seed_ledger.json"
            ).exists(),
            "no_outcome_inspection": failure.get(
                "outcome_loss_contact_reward_or_success_inspected"
            )
            is False,
        }
        result = {
            "schema_version": 1,
            "created_unix_ns": time.time_ns(),
            "isolated_temporary_harness": True,
            "invoked_interpreter": str(EVALUATION_PYTHON),
            "invoked_candidate_runner_sha256_recorded_at_seal": True,
            "returncode": completed.returncode,
            "checks": checks,
            "passed": all(checks.values()),
            "package_smoke_episodes": 0,
            "v5_outcome_episodes": 0,
        }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"generation negative-path qualification failed: {result}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
