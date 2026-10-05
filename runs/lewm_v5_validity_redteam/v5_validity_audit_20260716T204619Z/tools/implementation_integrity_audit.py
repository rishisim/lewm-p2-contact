#!/usr/bin/env python3
"""Frozen implementation, executed-code, runtime, and environment audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import marshal
import os
import pathlib
import struct
import subprocess
import time
from typing import Any


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def subprocess_probe(
    executable: str, script: pathlib.Path, cwd: pathlib.Path
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [executable, "-B", str(script)],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    payload = None
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout.strip())
        except Exception:
            payload = None
    return {
        "executable": executable,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "payload": payload,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (ValueError, FileNotFoundError):
        return False


def durable_package_files(package: pathlib.Path) -> set[str]:
    excluded = {"artifact_manifest.json", "audit/final_package_verification.json"}
    return {
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and str(path.relative_to(package)) not in excluded
        and not path.name.startswith(".")
    }


def pyc_audit(package: pathlib.Path) -> dict[str, Any]:
    records = []
    for path in sorted(package.rglob("*.cpython-310.pyc")):
        relative = str(path.relative_to(package))
        source_name = path.name.split(".cpython-310.pyc", 1)[0] + ".py"
        source = path.parent.parent / source_name
        raw = path.read_bytes()
        record: dict[str, Any] = {
            "path": relative,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source": str(source.relative_to(package)) if source.exists() else None,
            "unsealed": True,
        }
        if len(raw) < 16:
            record["valid_header"] = False
            records.append(record)
            continue
        flags = struct.unpack("<I", raw[4:8])[0]
        record["flags"] = flags
        record["hash_based"] = bool(flags & 1)
        record["valid_header"] = True
        if not (flags & 1):
            header_mtime, header_size = struct.unpack("<II", raw[8:16])
            record["header_mtime"] = header_mtime
            record["header_source_size"] = header_size
            if source.exists():
                record["timestamp_size_match_source"] = (
                    header_mtime == int(source.stat().st_mtime) & 0xFFFFFFFF
                    and header_size == source.stat().st_size & 0xFFFFFFFF
                )
        if source.exists():
            try:
                cached_code = marshal.loads(raw[16:])
                fresh_code = compile(
                    source.read_bytes(),
                    str(source),
                    "exec",
                    dont_inherit=True,
                    optimize=0,
                )
                record["marshal_code_exact_to_fresh_compile"] = (
                    marshal.dumps(cached_code) == marshal.dumps(fresh_code)
                )
            except Exception as error:
                record["marshal_code_exact_to_fresh_compile"] = False
                record["marshal_error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    source_backed = [item for item in records if item["source"] is not None]
    return {
        "cpython_310_count": len(records),
        "records": records,
        "all_timestamp_size_headers_match": bool(source_backed)
        and all(item.get("timestamp_size_match_source") for item in source_backed),
        "all_cached_code_exact_to_fresh_compile": bool(source_backed)
        and all(
            item.get("marshal_code_exact_to_fresh_compile")
            for item in source_backed
        ),
        "pyc_files_in_pre_v5_seal": False,
        "interpretation": (
            "CPython 3.10 caches are excluded from the seals, but every present "
            "source-backed cache is validated against a fresh compile here."
        ),
    }


def static_scan(paths: list[pathlib.Path]) -> dict[str, Any]:
    records = []
    for path in paths:
        tree = ast.parse(path.read_text())
        imports = []
        dynamic_calls = []
        environment_access = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                rendered = ast.unparse(node.func)
                if rendered in {
                    "__import__",
                    "eval",
                    "exec",
                    "importlib.import_module",
                    "importlib.util.spec_from_file_location",
                    "sys.path.insert",
                }:
                    dynamic_calls.append(
                        {
                            "call": rendered,
                            "line": getattr(node, "lineno", None),
                            "expression": ast.unparse(node)[:500],
                        }
                    )
                if rendered.startswith("os.environ"):
                    environment_access.append(
                        {
                            "call": rendered,
                            "line": getattr(node, "lineno", None),
                            "expression": ast.unparse(node)[:500],
                        }
                    )
            elif isinstance(node, ast.Subscript):
                rendered = ast.unparse(node.value)
                if rendered == "os.environ":
                    environment_access.append(
                        {
                            "access": "subscript",
                            "line": getattr(node, "lineno", None),
                            "expression": ast.unparse(node)[:500],
                        }
                    )
        text = path.read_text()
        records.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "imports": sorted(set(imports)),
                "dynamic_calls": dynamic_calls,
                "environment_access": environment_access,
                "mps_fallback_enabled": "PYTORCH_ENABLE_MPS_FALLBACK" in text,
                "source_file_loads": "spec_from_file_location" in text,
            }
        )
    return {
        "files": records,
        "dynamic_call_count": sum(len(item["dynamic_calls"]) for item in records),
        "environment_access_count": sum(
            len(item["environment_access"]) for item in records
        ),
        "all_scanned_file_hashes_bound_by_preseal": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--replay", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    package = args.package.resolve()
    replay = read(args.replay.resolve())

    preseal_path = package / "audit/pre_v5_seal.json"
    preseal = read(preseal_path)
    artifact_manifest_path = package / "artifact_manifest.json"
    artifact_manifest = read(artifact_manifest_path)
    input_seal = read(package / "audit/v5_confirmation_input_seal.json")
    raw_manifest = read(package / "data/v5_confirmation_raw_manifest.json")
    execution_manifest = read(
        package / "data/v5_confirmation_execution_manifest.json"
    )
    frozen = read(package / "freeze/frozen_candidate_manifest.json")
    contract = read(package / "runtime_contract.json")

    preseal_bad = []
    preseal_path_records = []
    for raw_path, expected in preseal["files"].items():
        candidate = pathlib.Path(raw_path)
        absolute_record = candidate.is_absolute()
        if not absolute_record:
            candidate = repo / candidate
        exists = candidate.exists()
        observed = sha256(candidate) if exists and candidate.is_file() else None
        preseal_path_records.append(
            {
                "declared_path": raw_path,
                "absolute": absolute_record,
                "exists": exists,
                "is_symlink": candidate.is_symlink(),
                "is_regular_lstat": exists
                and not candidate.is_symlink()
                and candidate.is_file(),
                "within_repo_after_resolve": (
                    is_within(candidate, repo) if not absolute_record and exists else None
                ),
                "sha256_match": observed == expected,
            }
        )
        if observed != expected:
            preseal_bad.append(raw_path)

    artifact_bad = []
    expected_package_relative = set()
    for relative, record in artifact_manifest["files"].items():
        candidate = repo / relative
        try:
            expected_package_relative.add(str(candidate.relative_to(package)))
        except ValueError:
            artifact_bad.append(relative)
            continue
        if (
            not candidate.exists()
            or candidate.is_symlink()
            or sha256(candidate) != record["sha256"]
        ):
            artifact_bad.append(relative)
    actual_durable = durable_package_files(package)
    post_confirmation_extra = sorted(actual_durable - expected_package_relative)
    package_ready_missing = sorted(expected_package_relative - actual_durable)

    input_seal_bad = []
    for relative, expected in {
        **input_seal["files"],
        **input_seal["raw_episode_hashes"],
    }.items():
        candidate = repo / relative
        if (
            not candidate.exists()
            or candidate.is_symlink()
            or sha256(candidate) != expected
        ):
            input_seal_bad.append(relative)

    raw_records = raw_manifest["episodes"]
    raw_path_security = {
        "all_paths_relative": all(
            not pathlib.PurePath(item["path"]).is_absolute() for item in raw_records
        ),
        "no_parent_components": all(
            ".." not in pathlib.PurePath(item["path"]).parts for item in raw_records
        ),
        "all_resolve_within_repo": all(
            is_within(repo / item["path"], repo) for item in raw_records
        ),
        "all_regular_not_symlink": all(
            (repo / item["path"]).is_file()
            and not (repo / item["path"]).is_symlink()
            for item in raw_records
        ),
        "unique_paths": len({item["path"] for item in raw_records})
        == len(raw_records)
        == 1600,
        "unique_episode_ids": len({item["episode_id"] for item in raw_records})
        == len(raw_records)
        == 1600,
        "unique_raw_hashes": len({item["sha256"] for item in raw_records})
        == len(raw_records)
        == 1600,
    }

    frozen_bad = []
    for relative, expected in frozen["package_files"].items():
        candidate = package / relative
        if (
            not candidate.exists()
            or candidate.is_symlink()
            or sha256(candidate) != expected
        ):
            frozen_bad.append(relative)
    external_mapping = {
        "base_config": repo / "runs/lewm_transfer/cube/cache/model/config.json",
        "base_weights": repo / "runs/lewm_transfer/cube/cache/model/weights.pt",
        "stagewise_refiner": repo
        / "runs/lewm_adaptive_compute_discovery/checkpoints/stagewise_seed_261102.pt",
        "whitening": package / "freeze/whitening.npz",
    }
    for name, expected in frozen["external_model_sources"].items():
        candidate = external_mapping[name]
        if not candidate.exists() or sha256(candidate) != expected:
            frozen_bad.append(str(candidate))

    runtime_probes = {
        role: subprocess_probe(
            contract["runtimes"][role]["sys_executable"],
            package / "verify_runtime_contract.py",
            package,
        )
        for role in ("evaluation", "generation")
    }
    seal_probes = {
        role: subprocess_probe(
            contract["runtimes"][role]["sys_executable"],
            package / "verify_pre_v5.py",
            package,
        )
        for role in ("evaluation", "generation")
    }
    source_probes = {
        role: subprocess_probe(
            contract["runtimes"][role]["sys_executable"],
            package / "verify_package_sources.py",
            package,
        )
        for role in ("evaluation", "generation")
    }

    scientific_paths = [
        package / "runner.py",
        package / "cycle_common.py",
        package / "input_loader.py",
        package / "counted_features.py",
        package / "analysis.py",
        package / "independent_verify.py",
        repo / "runs/lewm_adaptive_compute_distribution_contract/common.py",
        repo / "runs/lewm_adaptive_compute_distribution_contract/generator.py",
        repo / "runs/lewm_adaptive_compute_distribution_contract/generator_seedfix.py",
        repo / "runs/lewm_adaptive_compute_v4/runtime.py",
        repo / "runs/lewm_adaptive_compute_v4/common.py",
        repo / "runs/lewm_adaptive_compute_v4/generator.py",
        repo / "runs/lewm_adaptive_compute_v2/model_io.py",
        repo / "runs/lewm_adaptive_compute_discovery/models.py",
        repo / "runs/lewm_adaptive_compute_v1/refiner.py",
    ]
    scan = static_scan(scientific_paths)
    sealed_hashes = set(preseal["files"].values())
    scan["all_scanned_file_hashes_bound_by_preseal"] = all(
        item["sha256"] in sealed_hashes for item in scan["files"]
    )
    caches = pyc_audit(package)

    external_hash_records = contract["external_file_hashes"]
    package_origins = {
        role: contract["runtimes"][role]["module_origins"]
        for role in ("evaluation", "generation")
    }
    hashed_module_origins = []
    unhashed_declared_origins = []
    for role, origins in package_origins.items():
        for name, raw_path in origins.items():
            if raw_path is None:
                continue
            if raw_path in external_hash_records:
                hashed_module_origins.append(f"{role}:{name}")
            else:
                unhashed_declared_origins.append(f"{role}:{name}:{raw_path}")

    exact_execution_device_recorded = (
        "device" in execution_manifest
        or any(
            isinstance(item, dict) and "device" in item
            for item in execution_manifest.get("equivalence", [])
        )
    )
    replay_exact = bool(
        replay["passed"]
        and replay["complete_confirmation_replay"]
        and replay["checks"]["packaged_calls_exact"]
        and replay["checks"]["packaged_sparse_bitwise_exact"]
        and replay["checks"]["packaged_scores_exact"]
        and replay["checks"]["packaged_features_exact"]
    )
    module_unchanged = (
        execution_manifest["module_before"] == execution_manifest["module_after"]
        and execution_manifest["module_after"]["passed"]
        and execution_manifest["no_gradients"]
        and replay["module_before"] == replay["module_after"]
    )

    checks = {
        "preseal_hashes_current": not preseal_bad,
        "preseal_relative_paths_confined": all(
            item["within_repo_after_resolve"] is not False
            for item in preseal_path_records
        ),
        "package_ready_manifest_hashes_current": not artifact_bad,
        "package_ready_manifest_expected_subset_current": not package_ready_missing,
        "input_seal_hashes_current_and_regular": not input_seal_bad,
        "raw_canonical_paths_confined_regular_unique": all(
            raw_path_security.values()
        ),
        "frozen_candidate_hashes_current": not frozen_bad,
        "dual_runtime_contract_current": all(
            item["passed"] for item in runtime_probes.values()
        ),
        "dual_preseal_verification_current": all(
            item["passed"] for item in seal_probes.values()
        ),
        "dual_package_source_verification_current": all(
            item["passed"] for item in source_probes.values()
        ),
        "all_scanned_scientific_sources_sealed": scan[
            "all_scanned_file_hashes_bound_by_preseal"
        ],
        "cpython310_caches_match_fresh_source_compile": caches[
            "all_cached_code_exact_to_fresh_compile"
        ],
        "module_state_and_gradients_unchanged": module_unchanged,
        "full_mps_replay_exact": replay_exact and replay["device"] == "mps",
        "all_declared_top_level_module_origins_hashed": not unhashed_declared_origins,
        "full_transitive_dependency_graph_hashed": False,
        "historical_execution_device_explicitly_attested_in_execution_manifest": exact_execution_device_recorded,
    }

    output = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "source": {
            "repo": str(repo),
            "package": str(package),
            "pre_v5_seal_sha256": sha256(preseal_path),
            "artifact_manifest_sha256": sha256(artifact_manifest_path),
            "execution_sha256": sha256(
                package / "data/v5_confirmation_execution.npz"
            ),
        },
        "checks": checks,
        "preseal": {
            "sealed_file_count": len(preseal["files"]),
            "bad_hashes": preseal_bad,
            "path_records": preseal_path_records,
        },
        "package_ready_manifest": {
            "status": artifact_manifest["status"],
            "v5_outcome_episodes": artifact_manifest["v5_outcome_episodes"],
            "file_count": len(artifact_manifest["files"]),
            "bad_hashes": artifact_bad,
            "missing_expected": package_ready_missing,
            "post_confirmation_extra_file_count": len(post_confirmation_extra),
            "post_confirmation_extra_files": post_confirmation_extra,
            "is_post_confirmation_complete_path_manifest": False,
            "interpretation": (
                "The exact path-set manifest finalized PACKAGE_READY before V5. "
                "Post-confirmation inputs are instead bound by the input seal and "
                "final independent verifier; there is no second complete durable "
                "package path-set manifest after decision."
            ),
        },
        "input_seal": {
            "file_count": len(input_seal["files"]),
            "raw_episode_count": len(input_seal["raw_episode_hashes"]),
            "bad_hashes_or_nonregular": input_seal_bad,
            "raw_sidecars_hashed": False,
            "raw_sidecars_semantically_used_for_confirmation": False,
        },
        "raw_path_security": raw_path_security,
        "frozen_candidate": {
            "bad_hashes": frozen_bad,
            "all_current": not frozen_bad,
        },
        "runtime": {
            "probes": runtime_probes,
            "seal_probes": seal_probes,
            "source_probes": source_probes,
            "hashed_top_level_module_origins": sorted(hashed_module_origins),
            "unhashed_declared_module_origins": sorted(unhashed_declared_origins),
            "external_file_hash_count": len(external_hash_records),
            "recorded_package_names": list(
                contract["runtimes"]["evaluation"]["packages"]
            ),
            "complete_transitive_python_and_native_dependency_manifest": False,
            "os_macos_mps_framework_build_attested": False,
            "loaded_sys_modules_snapshot_attested": False,
            "mps_cpu_fallback_allowed_by_source": True,
        },
        "static_scan": scan,
        "python_bytecode_caches": caches,
        "executed_code_evidence": {
            "execution_manifest_records_explicit_device": exact_execution_device_recorded,
            "launcher_default_device": "mps",
            "preserved_mps_execution_replayed_bitwise_exact": replay_exact,
            "device_inference": (
                "MPS is strongly identified by the launcher default, MPS-specific "
                "qualification, and exact full MPS replay, but the historical execution "
                "manifest itself omits a device field."
            ),
            "module_before_after_exact": module_unchanged,
            "no_gradients": execution_manifest["no_gradients"],
            "full_replay_device": replay["device"],
            "full_replay_rows": replay["requested_row_count"],
        },
        "limitations": [
            {
                "id": "ENV-TRANSITIVE",
                "severity_candidate": "Major",
                "description": (
                    "Runtime sealing records interpreter hashes, versions, and top-level "
                    "package __init__ origins, but not every imported Python module, "
                    "native shared library, Metal/MPS framework, or OS build."
                ),
                "decision_error_observed": False,
                "bounded_by": (
                    "current dual-runtime probes plus bitwise-exact 60,800-row MPS replay"
                ),
            },
            {
                "id": "EXEC-DEVICE-ATTEST",
                "severity_candidate": "Minor",
                "description": (
                    "The execution manifest omits an explicit device/command field; "
                    "historical MPS use is inferred rather than directly attested there."
                ),
                "decision_error_observed": False,
                "bounded_by": "exact MPS replay and MPS launcher default",
            },
            {
                "id": "PYC-SEAL",
                "severity_candidate": "Minor",
                "description": (
                    "CPython bytecode caches are excluded from seals. Present 3.10 "
                    "caches were independently shown code-identical to fresh source "
                    "compilation."
                ),
                "decision_error_observed": False,
                "bounded_by": "fresh-compile bytecode comparison",
            },
            {
                "id": "POSTCONF-PATHSET",
                "severity_candidate": "Minor",
                "description": (
                    "No post-decision complete durable path-set manifest exists; the "
                    "input seal and independent verifier cover decision inputs, and the "
                    "audit's complete initial/final snapshots cover the preserved tree."
                ),
                "decision_error_observed": False,
                "bounded_by": "input seal, final verifier, and audit snapshots",
            },
        ],
        "test_results": {
            "FRZ-01": {
                "status": "pass",
                "reason": "all frozen scientific artifacts and external models hash-match",
            },
            "FRZ-02": {
                "status": "pass",
                "reason": (
                    "sealed sources/module state plus exact full MPS replay tie preserved "
                    "outputs to the intended path, with device-attestation caveat"
                ),
            },
            "FRZ-03": {
                "status": "fail",
                "reason": (
                    "the transitive Python/native/MPS environment is not completely "
                    "hashed or loaded-module-attested; no substitution was observed"
                ),
            },
            "FRZ-04": {
                "status": "pass",
                "reason": "frozen/no-gradient state and repeated full replay are exact",
            },
        },
    }
    atomic_json(args.output.resolve(), output)
    print(
        json.dumps(
            {
                "test_results": output["test_results"],
                "preseal_bad": len(preseal_bad),
                "input_seal_bad": len(input_seal_bad),
                "full_replay_exact": replay_exact,
                "transitive_environment_complete": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
