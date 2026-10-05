#!/usr/bin/env python3
"""Copy-only, fail-closed mutation campaign for the frozen V5 package.

The canonical package is never imported from a mutable test process and is
never written.  The package-ready artifact-manifest files are copied as
regular files into a repository-shaped fixture so the package's own verifiers
compute their normal ROOT/REPO_ROOT paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def atomic_output(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def run(
    command: list[str],
    *,
    cwd: pathlib.Path,
    timeout: int = 180,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "accepted": completed.returncode == 0,
    }


def flip_one_byte(path: pathlib.Path) -> dict[str, Any]:
    before = sha256(path)
    size = path.stat().st_size
    offset = max(0, size // 2)
    with path.open("r+b") as handle:
        handle.seek(offset)
        old = handle.read(1)
        if not old:
            raise RuntimeError(f"cannot flip byte in empty file: {path}")
        handle.seek(offset)
        handle.write(bytes([old[0] ^ 0x01]))
        handle.flush()
        os.fsync(handle.fileno())
    after = sha256(path)
    return {
        "offset": offset,
        "old_hex": old.hex(),
        "new_hex": bytes([old[0] ^ 0x01]).hex(),
        "sha256_before": before,
        "sha256_after": after,
        "exactly_one_byte_changed": before != after,
    }


def save_npz(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".mutating.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_npz(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def make_regular_backup(
    source: pathlib.Path, backup_root: pathlib.Path, name: str
) -> pathlib.Path:
    destination = backup_root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError("fault backup is not a regular file")
    return destination


def restore(backup: pathlib.Path, destination: pathlib.Path) -> None:
    destination.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup, destination, follow_symlinks=True)
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError("restored fixture file is not regular")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-repo", required=True, type=pathlib.Path)
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--audit-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--evaluation-python",
        default="/Users/rishisim/.cache/lewm-v2-venv/bin/python",
    )
    args = parser.parse_args()

    canonical_repo = args.canonical_repo.resolve()
    canonical_package = args.package.resolve()
    audit_dir = args.audit_dir.resolve()
    evaluation_python = pathlib.Path(args.evaluation_python)
    package_relative = canonical_package.relative_to(canonical_repo)
    copies = audit_dir / "fault_copies"
    fixture_parent = copies / "package_ready_fixture"
    fixture_repo = fixture_parent / "repo"
    fixture_package = fixture_repo / package_relative
    backups = copies / "backups"

    # This deletion is confined to a generated audit fixture under audit_dir.
    if fixture_parent.exists():
        shutil.rmtree(fixture_parent)
    if backups.exists():
        shutil.rmtree(backups)
    fixture_package.mkdir(parents=True, exist_ok=True)
    backups.mkdir(parents=True, exist_ok=True)

    canonical_manifest_path = canonical_package / "artifact_manifest.json"
    manifest = read_json(canonical_manifest_path)
    copied = []
    for relative, record in manifest["files"].items():
        source = canonical_repo / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"refusing non-regular canonical fixture source: {source}")
        destination = fixture_repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)
        observed = sha256(destination)
        if observed != record["sha256"]:
            raise RuntimeError(f"copy hash mismatch: {relative}")
        copied.append(
            {
                "path": relative,
                "bytes": destination.stat().st_size,
                "sha256": observed,
            }
        )
    shutil.copy2(
        canonical_manifest_path,
        fixture_package / "artifact_manifest.json",
        follow_symlinks=True,
    )

    # The pre-V5 seal also binds selected repository inputs outside the package
    # root.  Copy those specific regular files so the copied verifier has the
    # same repository-relative surface.  Absolute external-runtime records stay
    # read-only references to their canonical locations.
    preseal_fixture_files = []
    canonical_preseal = read_json(canonical_package / "audit/pre_v5_seal.json")
    for raw_path, expected in canonical_preseal["files"].items():
        relative = pathlib.Path(raw_path)
        if relative.is_absolute():
            continue
        source = canonical_repo / relative
        destination = fixture_repo / relative
        if not destination.exists():
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(
                    f"refusing non-regular canonical preseal fixture source: {source}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=True)
        observed = sha256(destination)
        if observed != expected:
            raise RuntimeError(f"preseal fixture hash mismatch: {relative}")
        preseal_fixture_files.append(
            {
                "path": str(relative),
                "bytes": destination.stat().st_size,
                "sha256": observed,
            }
        )

    final_verifier = fixture_package / "verify_final_package.py"
    preseal_verifier = fixture_package / "verify_pre_v5.py"
    smoke_verifier = fixture_package / "verify_package_smoke.py"
    final_command = [str(evaluation_python), "-B", str(final_verifier)]
    preseal_command = [str(evaluation_python), "-B", str(preseal_verifier)]
    smoke_command = [str(evaluation_python), "-B", str(smoke_verifier)]

    baseline_final = run(final_command, cwd=fixture_package)
    baseline_preseal = run(preseal_command, cwd=fixture_package)
    if not baseline_final["accepted"] or not baseline_preseal["accepted"]:
        raise RuntimeError(
            "copied package did not pass baseline verifiers: "
            f"final={baseline_final}, preseal={baseline_preseal}"
        )

    cases: list[dict[str, Any]] = []

    def add_case(
        *,
        case_id: str,
        domain: str,
        mutation: str,
        expected: str,
        observations: dict[str, Any],
        passed: bool,
        package_fail_open: bool = False,
        causal_note: str,
        pre_or_post_seal: str,
    ) -> None:
        cases.append(
            {
                "case_id": case_id,
                "domain": domain,
                "mutation": mutation,
                "expected": expected,
                "observations": observations,
                "passed": bool(passed),
                "package_fail_open": bool(package_fail_open),
                "causal_note": causal_note,
                "mutation_timing_model": pre_or_post_seal,
            }
        )

    # FLT-01: literal one-byte corruption of the preserved execution archive.
    execution = fixture_package / "data/package_smoke_execution.npz"
    execution_backup = make_regular_backup(execution, backups, "execution.npz")
    byte_change = flip_one_byte(execution)
    corrupted_execution_final = run(final_command, cwd=fixture_package)
    add_case(
        case_id="FLT-01A",
        domain="FLT-01",
        mutation="one-byte execution/outcome archive corruption",
        expected="final package hash verifier rejects",
        observations={
            "byte_change": byte_change,
            "final_verifier": corrupted_execution_final,
        },
        passed=not corrupted_execution_final["accepted"],
        causal_note="The package-ready SHA-256 binds the complete execution archive.",
        pre_or_post_seal="post-final-manifest",
    )
    restore(execution_backup, execution)

    # FLT-01: gate bytes are bound by both pre-V5 and final package manifests.
    gate = fixture_package / "freeze/compiled_gate.npz"
    gate_backup = make_regular_backup(gate, backups, "compiled_gate.npz")
    gate_change = flip_one_byte(gate)
    corrupted_gate_preseal = run(preseal_command, cwd=fixture_package)
    corrupted_gate_final = run(final_command, cwd=fixture_package)
    add_case(
        case_id="FLT-01B",
        domain="FLT-01",
        mutation="one-byte compiled-gate corruption",
        expected="pre-V5 seal and final package verifier reject",
        observations={
            "byte_change": gate_change,
            "preseal_verifier": corrupted_gate_preseal,
            "final_verifier": corrupted_gate_final,
        },
        passed=(
            not corrupted_gate_preseal["accepted"]
            and not corrupted_gate_final["accepted"]
        ),
        causal_note="The frozen gate is independently bound in two preserved manifests.",
        pre_or_post_seal="post-pre-V5-seal",
    )
    restore(gate_backup, gate)

    raw_manifest_path = fixture_package / "data/package_smoke_raw_manifest.json"
    raw_manifest_backup = make_regular_backup(
        raw_manifest_path, backups, "package_smoke_raw_manifest.json"
    )
    raw_manifest = read_json(raw_manifest_path)
    first_raw_relative = raw_manifest["episodes"][0]["path"]
    first_raw = fixture_repo / first_raw_relative
    first_raw_backup = make_regular_backup(first_raw, backups, "first_raw.npz")

    # FLT-02: a missing expected raw file is rejected.
    first_raw.unlink()
    missing_final = run(final_command, cwd=fixture_package)
    add_case(
        case_id="FLT-02A",
        domain="FLT-02",
        mutation="delete one expected raw episode",
        expected="final package path/hash verifier rejects",
        observations={"final_verifier": missing_final},
        passed=not missing_final["accepted"],
        causal_note="Missing preserved inputs cannot authenticate.",
        pre_or_post_seal="post-final-manifest",
    )
    restore(first_raw_backup, first_raw)

    # FLT-02: an extra durable raw file is rejected by exact path-set checking.
    extra_raw = first_raw.parent / "unexpected-extra.npz"
    shutil.copy2(first_raw, extra_raw, follow_symlinks=True)
    extra_final = run(final_command, cwd=fixture_package)
    add_case(
        case_id="FLT-02B",
        domain="FLT-02",
        mutation="add one extra raw episode file",
        expected="final package exact path-set verifier rejects",
        observations={"final_verifier": extra_final},
        passed=not extra_final["accepted"],
        causal_note="The package-ready path set rejects unmanifested durable files.",
        pre_or_post_seal="post-final-manifest",
    )
    extra_raw.unlink()

    # FLT-02: manifest hash mismatch is rejected by the runtime raw-manifest
    # preflight (not merely by the package-ready outer hash).
    manifest_mismatch = read_json(raw_manifest_path)
    manifest_mismatch["episodes"][0]["sha256"] = "0" * 64
    write_json(raw_manifest_path, manifest_mismatch)
    raw_validator_code = (
        "import json,runner;"
        "runner._validate_existing_raw_manifest('package_smoke',12);"
        "print(json.dumps({'accepted':True}))"
    )
    manifest_preflight = run(
        [str(evaluation_python), "-B", "-c", raw_validator_code],
        cwd=fixture_package,
    )
    add_case(
        case_id="FLT-02C",
        domain="FLT-02",
        mutation="raw-manifest record hash mismatch",
        expected="raw-manifest execution preflight rejects",
        observations={"raw_manifest_preflight": manifest_preflight},
        passed=not manifest_preflight["accepted"],
        causal_note="Every listed raw file is rehashed before execution.",
        pre_or_post_seal="before-execution",
    )
    restore(raw_manifest_backup, raw_manifest_path)

    # FLT-02: the package's existing-manifest preflight does not validate
    # identifier uniqueness.  The excluded-smoke verifier does, indirectly via
    # the frozen seed ledger, but the V5 final verifier has no equivalent
    # episode-string-to-ledger comparison.
    duplicate_manifest = read_json(raw_manifest_path)
    duplicate_manifest["episodes"][1]["episode_id"] = duplicate_manifest["episodes"][0][
        "episode_id"
    ]
    write_json(raw_manifest_path, duplicate_manifest)
    duplicate_preflight = run(
        [str(evaluation_python), "-B", "-c", raw_validator_code],
        cwd=fixture_package,
    )
    output_smoke = fixture_package / "audit/package_smoke_verification.json"
    smoke_output_backup = make_regular_backup(
        output_smoke, backups, "package_smoke_verification.json"
    )
    output_smoke.unlink()
    duplicate_smoke_final = run(smoke_command, cwd=fixture_package)
    restore(smoke_output_backup, output_smoke)
    canonical_v5_manifest = read_json(
        canonical_package / "data/v5_confirmation_raw_manifest.json"
    )
    v5_duplicate_probe = json.loads(json.dumps(canonical_v5_manifest))
    v5_duplicate_probe["episodes"][1]["episode_id"] = v5_duplicate_probe["episodes"][0][
        "episode_id"
    ]
    expected_names = {
        pathlib.Path(item["path"]).name for item in v5_duplicate_probe["episodes"]
    }
    actual_names = {
        path.name
        for path in (canonical_package / "data/v5_confirmation_raw").glob("*.npz")
    }
    v5_final_visible_predicates = {
        "manifest_list_length_1600": len(v5_duplicate_probe["episodes"]) == 1600,
        "raw_path_set_still_exact": expected_names == actual_names,
        "episode_string_uniqueness_explicitly_checked": False,
        "execution_numeric_episode_ids_unchanged": True,
    }
    add_case(
        case_id="FLT-02D",
        domain="FLT-02",
        mutation="duplicate one manifest episode identifier while retaining unique paths",
        expected="every applicable preflight/final verifier rejects",
        observations={
            "existing_manifest_preflight": duplicate_preflight,
            "excluded_smoke_final_verifier": duplicate_smoke_final,
            "v5_final_verifier_predicate_probe": v5_final_visible_predicates,
            "canonical_audit_uniqueness_check": "separate chronology/cohort audit",
        },
        passed=(
            not duplicate_preflight["accepted"]
            and not duplicate_smoke_final["accepted"]
        ),
        package_fail_open=duplicate_preflight["accepted"],
        causal_note=(
            "The generation/execution preflight accepts duplicate identifier metadata; "
            "the smoke verifier later rejects it, but the V5 independent verifier does "
            "not explicitly bind manifest episode strings to the seed ledger. The "
            "canonical cohort is separately proven unique."
        ),
        pre_or_post_seal="malformed manifest before execution/input seal",
    )
    restore(raw_manifest_backup, raw_manifest_path)

    # FLT-02: consistently reorder all serialized execution rows, rebind the
    # execution-manifest hash, and run the package smoke verifier.  This probes
    # semantic order enforcement after integrity hashes are legitimately
    # regenerated (a post-seal byte mutation would already be rejected).
    execution_arrays = load_npz(execution)
    rows = len(execution_arrays["episode_id"])
    permutation = np.concatenate(
        [np.arange(start, start + 38) for start in range(rows - 38, -1, -38)]
    )
    if len(permutation) != rows:
        raise RuntimeError("row permutation fixture length mismatch")
    for name, value in list(execution_arrays.items()):
        if value.ndim >= 1 and len(value) == rows:
            execution_arrays[name] = value[permutation]
    save_npz(execution, execution_arrays)
    execution_manifest_path = (
        fixture_package / "data/package_smoke_execution_manifest.json"
    )
    execution_manifest_backup = make_regular_backup(
        execution_manifest_path, backups, "package_smoke_execution_manifest.json"
    )
    execution_manifest = read_json(execution_manifest_path)
    execution_manifest["sha256"] = sha256(execution)
    write_json(execution_manifest_path, execution_manifest)
    output_smoke.unlink()
    reordered_smoke_final = run(smoke_command, cwd=fixture_package)
    restore(smoke_output_backup, output_smoke)
    add_case(
        case_id="FLT-02E",
        domain="FLT-02",
        mutation="reverse complete episode blocks in every execution array and rebind manifest hash",
        expected="semantic verifier rejects noncanonical row order",
        observations={
            "row_count": rows,
            "first_reordered_episode_id": int(
                execution_arrays["episode_id"][0]
            ),
            "package_smoke_verifier": reordered_smoke_final,
        },
        passed=not reordered_smoke_final["accepted"],
        package_fail_open=reordered_smoke_final["accepted"],
        causal_note=(
            "The verifier checks membership/counts but not canonical serialized row "
            "order. A paired permutation is metric-invariant, so this is hardening and "
            "provenance weakness rather than an observed decision error."
        ),
        pre_or_post_seal="hypothetical regenerated manifest before analysis",
    )
    restore(execution_backup, execution)
    restore(execution_manifest_backup, execution_manifest_path)

    # FLT-03: path traversal in a pre-execution raw manifest.
    traversal_target = copies / "traversal_target.npz"
    shutil.copy2(first_raw, traversal_target, follow_symlinks=True)
    traversal_relative = os.path.relpath(traversal_target, fixture_repo)
    if ".." not in pathlib.PurePath(traversal_relative).parts:
        raise RuntimeError("traversal fixture did not escape the synthetic repo root")
    traversal_manifest = {
        "complete": True,
        "episode_count": 1,
        "episodes": [
            {
                "episode_id": "traversal-probe",
                "path": traversal_relative,
                "sha256": sha256(traversal_target),
            }
        ],
    }
    traversal_manifest_path = fixture_package / "data/traversal_probe_raw_manifest.json"
    write_json(traversal_manifest_path, traversal_manifest)
    traversal_code = (
        "import json,runner;"
        "value=runner._validate_existing_raw_manifest('traversal_probe',1);"
        "print(json.dumps({'accepted':value is not None,"
        "'path':value['episodes'][0]['path']}))"
    )
    traversal_preflight = run(
        [str(evaluation_python), "-B", "-c", traversal_code],
        cwd=fixture_package,
    )
    traversal_manifest_path.unlink()
    add_case(
        case_id="FLT-03A",
        domain="FLT-03",
        mutation="raw-manifest path escapes REPO_ROOT with .. and carries a valid external hash",
        expected="raw preflight rejects root escape before opening external file",
        observations={
            "manifest_path": traversal_relative,
            "raw_manifest_preflight": traversal_preflight,
        },
        passed=not traversal_preflight["accepted"],
        package_fail_open=traversal_preflight["accepted"],
        causal_note=(
            "The preflight joins paths without resolve/root-confinement and therefore "
            "authenticates an out-of-root file if its hash is declared."
        ),
        pre_or_post_seal="malformed manifest before execution/input seal",
    )

    # FLT-03: replace an expected regular package artifact with a symlink to an
    # identical external regular copy.  is_file and sha256 both follow it.
    symlink_target = copies / "compiled_gate_external_identical.npz"
    shutil.copy2(gate, symlink_target, follow_symlinks=True)
    gate.unlink()
    gate.symlink_to(symlink_target)
    symlink_final = run(final_command, cwd=fixture_package)
    add_case(
        case_id="FLT-03B",
        domain="FLT-03",
        mutation="replace compiled gate with symlink to byte-identical external file",
        expected="final verifier rejects non-regular/symlink artifact",
        observations={
            "fixture_is_symlink": gate.is_symlink(),
            "target": str(symlink_target),
            "final_verifier": symlink_final,
        },
        passed=not symlink_final["accepted"],
        package_fail_open=symlink_final["accepted"],
        causal_note=(
            "The final verifier follows symlinks and authenticates content only; the "
            "canonical snapshot independently shows regular files, so preserved bytes "
            "are not presently substituted."
        ),
        pre_or_post_seal="filesystem substitution after final manifest",
    )
    restore(gate_backup, gate)

    # FLT-04: change an actual threshold value while retaining the old seal.
    gate_arrays = load_npz(gate)
    old_threshold = float(gate_arrays["thresholds"][0])
    gate_arrays["thresholds"][0] = np.nextafter(
        gate_arrays["thresholds"][0],
        np.asarray(np.inf, dtype=gate_arrays["thresholds"].dtype),
    )
    new_threshold = float(gate_arrays["thresholds"][0])
    save_npz(gate, gate_arrays)
    threshold_preseal = run(preseal_command, cwd=fixture_package)
    add_case(
        case_id="FLT-04A",
        domain="FLT-04",
        mutation="change first compiled score threshold by one representable float",
        expected="pre-V5 seal rejects",
        observations={
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "preseal_verifier": threshold_preseal,
        },
        passed=not threshold_preseal["accepted"],
        causal_note="Threshold bytes are directly included in the pre-V5 seal.",
        pre_or_post_seal="post-pre-V5-seal",
    )
    restore(gate_backup, gate)

    # FLT-04: change the declared adapter compute price in sealed source.
    common_path = fixture_package / "cycle_common.py"
    common_backup = make_regular_backup(common_path, backups, "cycle_common.py")
    common_source = common_path.read_text()
    changed_source = common_source.replace(
        "ADAPTER_FLOPS = 264_960", "ADAPTER_FLOPS = 264_961", 1
    )
    if changed_source == common_source:
        raise RuntimeError("compute-price mutation did not apply")
    common_path.write_text(changed_source)
    compute_preseal = run(preseal_command, cwd=fixture_package)
    add_case(
        case_id="FLT-04B",
        domain="FLT-04",
        mutation="change sealed adapter compute price from 264960 to 264961",
        expected="pre-V5 seal rejects source drift",
        observations={"preseal_verifier": compute_preseal},
        passed=not compute_preseal["accepted"],
        causal_note="The compute constants' source file is hash-bound before outcomes.",
        pre_or_post_seal="post-pre-V5-seal",
    )
    restore(common_backup, common_path)

    # FLT-04: invoke the exact runtime assertion from an unapproved interpreter.
    wrong_python_candidates = [
        pathlib.Path("/usr/bin/python3"),
        pathlib.Path(shutil.which("python3") or ""),
    ]
    wrong_python = next(
        (
            candidate
            for candidate in wrong_python_candidates
            if candidate.exists()
            and candidate.resolve() != evaluation_python.resolve()
        ),
        None,
    )
    if wrong_python is None:
        wrong_runtime = {
            "accepted": False,
            "returncode": None,
            "unavailable": True,
        }
        wrong_runtime_passed = False
    else:
        runtime_code = (
            "from cycle_common import assert_runtime_contract;"
            "assert_runtime_contract('evaluation')"
        )
        wrong_runtime = run(
            [str(wrong_python), "-B", "-c", runtime_code],
            cwd=fixture_package,
        )
        wrong_runtime_passed = not wrong_runtime["accepted"]
    add_case(
        case_id="FLT-04C",
        domain="FLT-04",
        mutation="invoke evaluation runtime assertion under an unapproved interpreter",
        expected="runtime assertion rejects before inference",
        observations={
            "wrong_interpreter": str(wrong_python) if wrong_python else None,
            "runtime_assertion": wrong_runtime,
        },
        passed=wrong_runtime_passed,
        causal_note="The execution/analysis entry points call the strict runtime assertion.",
        pre_or_post_seal="runtime preflight",
    )

    # FLT-05: inject several unexpected privileged canaries into one copied raw
    # NPZ.  First retain the old manifest to test post-manifest rejection; then
    # rebind the manifest to probe loader safe-ignore and the final chain.
    original_arrays = load_npz(first_raw)
    canary_arrays = dict(original_arrays)
    canary_arrays.update(
        {
            "future_target_canary": np.full((201, 192), 12345.5, dtype=np.float32),
            "oracle_gain_canary": np.full((200,), -9876.25, dtype=np.float32),
            "unexpected_privileged_contact": np.ones((201,), dtype=np.int8),
        }
    )
    baseline_action_sha = hashlib.sha256(
        np.ascontiguousarray(original_arrays["action"]).tobytes()
    ).hexdigest()
    baseline_pixels_sha = hashlib.sha256(
        np.ascontiguousarray(original_arrays["pixels"]).tobytes()
    ).hexdigest()
    save_npz(first_raw, canary_arrays)
    canary_sha = sha256(first_raw)
    post_manifest_canary_preflight = run(
        [str(evaluation_python), "-B", "-c", raw_validator_code],
        cwd=fixture_package,
    )
    updated_manifest = read_json(raw_manifest_path)
    updated_manifest["episodes"][0]["sha256"] = canary_sha
    updated_manifest["episodes"][0]["bytes"] = first_raw.stat().st_size
    write_json(raw_manifest_path, updated_manifest)
    rebound_canary_preflight = run(
        [str(evaluation_python), "-B", "-c", raw_validator_code],
        cwd=fixture_package,
    )
    loader_code = (
        "import hashlib,json,pathlib;"
        "from input_loader import load_model_gate_inputs;"
        f"loaded,audit=load_model_gate_inputs(pathlib.Path({str(first_raw)!r}));"
        "out={'loaded_keys':sorted(loaded),"
        "'archive_keys':audit['archive_key_names_observed_without_array_load'],"
        "'forbidden_arrays_materialized':audit['forbidden_arrays_materialized'],"
        "'contact_or_privileged_loaded':audit['contact_or_privileged_loaded'],"
        "'action_sha256':hashlib.sha256(loaded['action'].tobytes()).hexdigest(),"
        "'pixels_sha256':hashlib.sha256(loaded['pixels'].tobytes()).hexdigest()};"
        "print(json.dumps(out,sort_keys=True))"
    )
    canary_loader = run(
        [str(evaluation_python), "-B", "-c", loader_code],
        cwd=fixture_package,
    )
    try:
        loader_payload = json.loads(canary_loader["stdout_tail"].strip())
    except Exception:
        loader_payload = {}
    output_smoke.unlink()
    rebound_canary_smoke = run(smoke_command, cwd=fixture_package)
    restore(smoke_output_backup, output_smoke)
    final_with_old_outer_manifest = run(final_command, cwd=fixture_package)

    outer_manifest_path = fixture_package / "artifact_manifest.json"
    outer_manifest_backup = make_regular_backup(
        outer_manifest_path, backups, "artifact_manifest.json"
    )
    outer_manifest = read_json(outer_manifest_path)
    for path in (first_raw, raw_manifest_path):
        relative = str(path.relative_to(fixture_repo))
        outer_manifest["files"][relative]["sha256"] = sha256(path)
        outer_manifest["files"][relative]["bytes"] = path.stat().st_size
    write_json(outer_manifest_path, outer_manifest)
    final_after_rebinding = run(final_command, cwd=fixture_package)
    canary_safe_ignore = bool(
        canary_loader["accepted"]
        and loader_payload.get("loaded_keys") == ["action", "pixels"]
        and loader_payload.get("forbidden_arrays_materialized") is False
        and loader_payload.get("contact_or_privileged_loaded") is False
        and loader_payload.get("action_sha256") == baseline_action_sha
        and loader_payload.get("pixels_sha256") == baseline_pixels_sha
    )
    add_case(
        case_id="FLT-05A",
        domain="FLT-05",
        mutation="add future-target, oracle-gain, and privileged-contact canary arrays",
        expected=(
            "post-manifest injection is rejected; if present before sealing, loader "
            "must not materialize it or change model/gate inputs"
        ),
        observations={
            "canary_keys": [
                "future_target_canary",
                "oracle_gain_canary",
                "unexpected_privileged_contact",
            ],
            "post_manifest_raw_preflight": post_manifest_canary_preflight,
            "rebound_manifest_raw_preflight": rebound_canary_preflight,
            "copied_loader": canary_loader,
            "loader_payload": loader_payload,
            "rebound_manifest_smoke_verifier": rebound_canary_smoke,
            "old_outer_manifest_final_verifier": final_with_old_outer_manifest,
            "rebound_outer_manifest_final_verifier": final_after_rebinding,
            "strict_unknown_field_rejection": False,
            "safe_ignore_invariance": canary_safe_ignore,
        },
        passed=(
            not post_manifest_canary_preflight["accepted"]
            and rebound_canary_preflight["accepted"]
            and canary_safe_ignore
            and rebound_canary_smoke["accepted"]
            and not final_with_old_outer_manifest["accepted"]
            and final_after_rebinding["accepted"]
        ),
        package_fail_open=False,
        causal_note=(
            "Unknown arrays are not schema-rejected when present before the relevant "
            "manifest, but the allowlist loader only materializes action/pixels and "
            "is invariant. Post-manifest insertion is hash-rejected. This is safe "
            "ignore, not privileged consumption."
        ),
        pre_or_post_seal="both post-manifest and pre-manifest models",
    )
    restore(first_raw_backup, first_raw)
    restore(raw_manifest_backup, raw_manifest_path)
    restore(outer_manifest_backup, outer_manifest_path)

    # Restore-check the reusable fixture after every destructive case.
    final_after_restore = run(final_command, cwd=fixture_package)
    preseal_after_restore = run(preseal_command, cwd=fixture_package)

    domains: dict[str, Any] = {}
    for domain in ("FLT-01", "FLT-02", "FLT-03", "FLT-04", "FLT-05"):
        selected = [case for case in cases if case["domain"] == domain]
        domains[domain] = {
            "case_count": len(selected),
            "passed_case_count": sum(bool(case["passed"]) for case in selected),
            "failed_case_count": sum(not bool(case["passed"]) for case in selected),
            "status": "pass" if all(case["passed"] for case in selected) else "fail",
            "package_fail_open_case_ids": [
                case["case_id"] for case in selected if case["package_fail_open"]
            ],
        }

    output = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "canonical_evidence_mutated": False,
        "fixture": {
            "canonical_package": str(canonical_package),
            "fixture_repo": str(fixture_repo),
            "fixture_package": str(fixture_package),
            "copied_regular_file_count": len(copied),
            "copied_total_bytes": sum(item["bytes"] for item in copied),
            "preseal_repository_file_count": len(preseal_fixture_files),
            "preseal_repository_total_bytes": sum(
                item["bytes"] for item in preseal_fixture_files
            ),
            "copy_manifest_sha256": hashlib.sha256(
                json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "baseline_final_verifier": baseline_final,
            "baseline_preseal_verifier": baseline_preseal,
            "final_after_restore": final_after_restore,
            "preseal_after_restore": preseal_after_restore,
        },
        "cases": cases,
        "domains": domains,
        "summary": {
            "case_count": len(cases),
            "passed_case_count": sum(bool(case["passed"]) for case in cases),
            "failed_case_count": sum(not bool(case["passed"]) for case in cases),
            "domain_pass_count": sum(
                item["status"] == "pass" for item in domains.values()
            ),
            "domain_fail_count": sum(
                item["status"] == "fail" for item in domains.values()
            ),
            "package_fail_open_case_ids": [
                case["case_id"] for case in cases if case["package_fail_open"]
            ],
            "all_post_seal_hash_mutations_rejected": all(
                case["passed"]
                for case in cases
                if case["mutation_timing_model"].startswith("post-")
                and case["case_id"] not in {"FLT-03B"}
            ),
            "strict_privileged_schema_rejection": False,
            "privileged_canary_safe_ignore": next(
                case["passed"] for case in cases if case["case_id"] == "FLT-05A"
            ),
        },
        "interpretation": {
            "validated_fail_open_surfaces": [
                "pre-execution raw-manifest paths are not root-confined",
                "final durable-file verifier follows symlinks",
                "existing raw-manifest preflight does not enforce identifier uniqueness",
                "semantic verifiers do not enforce canonical serialized row order",
            ],
            "canonical_state_cross_check_required": [
                "canonical paths confined and regular",
                "canonical identifiers, paths, and raw hashes unique",
                "canonical execution row order matches raw manifest",
                "initial/final canonical snapshots unchanged",
            ],
            "no_fault_was_applied_to_canonical_evidence": True,
        },
    }
    atomic_output(args.output.resolve(), output)
    print(
        json.dumps(
            {
                "domains": domains,
                "summary": output["summary"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
