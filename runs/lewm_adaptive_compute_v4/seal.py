#!/usr/bin/env python3
"""Hash-seal preregistration and final one-shot evaluation sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common


IMMUTABLE_PREREG = (
    "PREREGISTRATION.md",
    "config.json",
    "decision_rule.json",
    "seed_manifest.json",
    "source_manifest.json",
    "checkpoint_manifest.json",
    "provenance_manifest.json",
    "audit/prior_evidence_audit.json",
    "audit/prior_identifier_inventory.json",
    "audit/regime_matching_bins.json",
)


def _scientific_sources() -> list[Path]:
    paths = list(common.ROOT.glob("*.py")) + list((common.ROOT / "tests").glob("*.py"))
    return sorted(path for path in paths if path.is_file())


def _relative_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(common.REPO)): common.sha256_file(path)
        for path in sorted(paths)
    }


def phase0() -> dict[str, object]:
    output = common.ROOT / "audit/phase0_seal.json"
    if output.exists():
        raise RuntimeError("phase-0 seal already exists")
    forbidden = [
        common.ROOT / "data/smoke_data_manifest.json",
        common.ROOT / "data/confirmation_data_manifest.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("phase-0 seal must precede all V4 episode generation")
    prereg_paths = [common.ROOT / name for name in IMMUTABLE_PREREG]
    for path in prereg_paths:
        if not path.exists():
            raise RuntimeError(f"missing preregistration artifact: {path}")
    source_paths = _scientific_sources()
    test_log = common.ROOT / "logs/phase0_tests.json"
    if not test_log.exists() or not common.read_json(test_log)["passed"]:
        raise RuntimeError("focused phase-0 tests must pass before sealing")
    paths = prereg_paths + source_paths + [test_log]
    payload = {
        "schema_version": 1,
        "status": "sealed_before_any_v4_episode_generation",
        "files": _relative_hashes(paths),
        "immutable_preregistration_files": [
            str(path.relative_to(common.REPO)) for path in prereg_paths
        ],
        "evaluation_runtime_source_files": [
            str(path.relative_to(common.REPO)) for path in source_paths
        ],
        "frozen_objects": common.verify_frozen_objects(),
        "v3_test_targets_opened": False,
    }
    common.write_json(output, payload, exclusive=True)
    common.file_mode_read_only(output)
    return payload


def preconfirmation() -> dict[str, object]:
    output = common.ROOT / "audit/pre_confirmation_seal.json"
    if output.exists():
        raise RuntimeError("pre-confirmation seal already exists")
    phase0_path = common.ROOT / "audit/phase0_seal.json"
    phase0_payload = common.read_json(phase0_path)
    # Preregistration and source/checkpoint provenance can never change.
    for relative in phase0_payload["immutable_preregistration_files"]:
        path = common.REPO / relative
        if common.sha256_file(path) != phase0_payload["files"][relative]:
            raise RuntimeError(f"immutable preregistration changed after smoke: {relative}")
    smoke = common.ROOT / "audit/smoke_completion.json"
    smoke_manifest = common.ROOT / "data/smoke_data_manifest.json"
    smoke_outcome = common.ROOT / "data/smoke_outcome_manifest.json"
    test_log = common.ROOT / "logs/preconfirmation_tests.json"
    for path in (smoke, smoke_manifest, smoke_outcome, test_log):
        if not path.exists():
            raise RuntimeError(f"missing smoke/preconfirmation artifact: {path}")
    if not common.read_json(smoke)["evaluation_passed"] or not common.read_json(test_log)["passed"]:
        raise RuntimeError("smoke or preconfirmation tests did not pass")
    source_paths = _scientific_sources()
    current_sources = _relative_hashes(source_paths)
    initial_sources = {
        key: value
        for key, value in phase0_payload["files"].items()
        if key in phase0_payload["evaluation_runtime_source_files"]
    }
    changed = sorted(
        key for key, value in current_sources.items() if initial_sources.get(key) != value
    )
    removed = sorted(set(initial_sources) - set(current_sources))
    if removed:
        raise RuntimeError(f"sealed source files were removed after smoke: {removed}")
    allowed_runtime_changes = {
        str((common.ROOT / name).relative_to(common.REPO))
        for name in ("runtime.py", "evaluate.py", "benchmark_runtime.py")
    }
    if not set(changed) <= allowed_runtime_changes:
        raise RuntimeError(f"non-runtime scientific source changed after smoke: {changed}")
    paths = [
        phase0_path,
        *[common.ROOT / name for name in IMMUTABLE_PREREG],
        *source_paths,
        smoke,
        smoke_manifest,
        smoke_outcome,
        test_log,
    ]
    payload = {
        "schema_version": 1,
        "status": "final_runtime_and_evaluation_frozen_before_confirmation_receipt_or_target_io",
        "phase0_seal_sha256": common.sha256_file(phase0_path),
        "files": _relative_hashes(paths),
        "source_changes_since_phase0": changed,
        "source_change_policy": (
            "Only semantics-preserving runtime implementation files may change after smoke; "
            "all final paths passed exact-depth and numerical-equivalence smoke tests."
        ),
        "smoke_episodes": 12,
        "smoke_forever_excluded": True,
        "frozen_objects": common.verify_frozen_objects(),
        "v3_test_targets_opened": False,
    }
    common.write_json(output, payload, exclusive=True)
    common.file_mode_read_only(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase0", "preconfirmation"))
    args = parser.parse_args()
    result = phase0() if args.phase == "phase0" else preconfirmation()
    print(json.dumps({"status": result["status"], "files": len(result["files"])}, sort_keys=True))


if __name__ == "__main__":
    main()
