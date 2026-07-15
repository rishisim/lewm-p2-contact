#!/usr/bin/env python3
"""Audit all predecessor evidence and enumerate recorded identifiers.

The V3 combined target cache and the isolated V3 calibration target payload are
never opened.  Their bytes are checked opaquely.  Every other file in the two
required evidence trees is read according to its format and hashed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import hdf5plugin  # noqa: F401
import numpy as np
import torch
from PIL import Image

import common


REQUIRED_TREES = (common.DISCOVERY, common.COMPRESSION)
OPAQUE_PAYLOADS = {
    common.COMPRESSION / "cache/v3_calibration_only.npz",
    common.V3 / "cache/cube_inputs.npz",
}
IDENTIFIER_ROOTS = (
    common.REPO / "runs/lewm_adaptive_compute_v1",
    common.REPO / "runs/lewm_adaptive_compute_v2",
    common.REPO / "runs/lewm_adaptive_compute_v3",
    common.DISCOVERY,
    common.COMPRESSION,
    common.REPO / "le-wm/diagnostics",
)
IDENTIFIER_TOKEN = re.compile(
    r"(?:^|_)(?:seed|episode|episode_id|episode_idx|episode_ordinal|ep_idx|"
    r"trajectory|identifier|id)(?:$|_)", re.IGNORECASE
)
TEXT_SUFFIXES = {".md", ".py", ".txt", ".json", ".csv"}


def _tensor_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all().item())
            if torch.is_floating_point(value)
            else True,
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.isfinite(value).all())
            if value.dtype.kind in "fc"
            else True,
        }
    if isinstance(value, Mapping):
        return {str(key): _tensor_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 25:
            return {
                "kind": type(value).__name__,
                "length": len(value),
                "head": [_tensor_summary(item) for item in value[:3]],
            }
        return [_tensor_summary(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__, "repr": repr(value)[:200]}


def inspect_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    record: dict[str, Any] = {
        "path": str(path.relative_to(common.REPO)),
        "bytes": int(path.stat().st_size),
        "sha256": common.sha256_file(path),
    }
    if path in OPAQUE_PAYLOADS:
        record.update(format="opaque_forbidden_target_container", opened=False)
        return record
    if suffix in {".md", ".py", ".txt"}:
        text = path.read_text(encoding="utf-8")
        record.update(format="text", characters=len(text), lines=text.count("\n") + 1)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        record.update(
            format="json",
            top_level_type=type(payload).__name__,
            top_level_keys=sorted(payload) if isinstance(payload, dict) else None,
        )
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = 0
            for _ in reader:
                rows += 1
        record.update(format="csv", columns=reader.fieldnames, rows=rows)
    elif suffix == ".png":
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record.update(format="png", width=image.width, height=image.height, mode=image.mode)
    elif suffix == ".npz":
        arrays = {}
        with np.load(path, allow_pickle=False) as stored:
            for name in stored.files:
                value = stored[name]
                arrays[name] = {
                    **_tensor_summary(value),
                    "array_sha256": common.array_sha256(value),
                }
        record.update(format="npz", arrays=arrays)
    elif suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        record.update(format="torch", payload=_tensor_summary(payload))
    else:
        # Complete-byte hashing above is the appropriate read for unknown data.
        record.update(format="opaque_unknown", opened=True)
    return record


def verify_artifact_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in payload["files"].items():
        path = root / relative
        observed = common.sha256_file(path) if path.exists() else None
        if observed != expected:
            failures.append({"path": relative, "expected": expected, "observed": observed})
    if failures:
        raise RuntimeError(f"artifact manifest failed for {root}: {failures[:3]}")
    return {
        "path": str(manifest_path.relative_to(common.REPO)),
        "sha256": common.sha256_file(manifest_path),
        "entries": len(payload["files"]),
        "all_match": True,
    }


def audit_claims() -> dict[str, Any]:
    calibration = common.read_json(common.COMPRESSION / "metrics/calibration_once.json")
    decision = common.read_json(common.COMPRESSION / "decision.json")
    adversarial = common.read_json(
        common.COMPRESSION / "metrics/adversarial_verification.json"
    )
    runtime = common.read_json(common.COMPRESSION / "metrics/runtime.json")
    runtime_1024 = {
        row["path"]: row
        for row in runtime["timings"]
        if int(row["batch_size"]) == 1024
    }
    expected = {
        "adaptive_raw_mse": 0.0031314349443701435,
        "matched_raw_mse": 0.0031499073418161604,
        "benefit": 0.00001847239744601695,
        "ci_low": 0.000008881163526542912,
        "ci_high": 0.000029195376017610194,
    }
    observed = {
        "adaptive_raw_mse": calibration["raw_mse"],
        "matched_raw_mse": calibration["matched_raw_mse"],
        "benefit": calibration["vs_matched_randomized"]["mean_benefit"],
        "ci_low": calibration["vs_matched_randomized"]["ci_low"],
        "ci_high": calibration["vs_matched_randomized"]["ci_high"],
    }
    exact = all(float(observed[key]) == float(value) for key, value in expected.items())
    checks = {
        "calibration_values_exact": exact,
        "analytic_ci_positive": calibration["vs_analytic_mixture"]["ci_low"] > 0,
        "fixed_d1_ci_positive": calibration["vs_fixed_d1"]["ci_low"] > 0,
        "histogram_ci_positive": calibration["vs_histogram_null"]["ci_low"] > 0,
        "whitened_positive_inconclusive": calibration["whitened_vs_matched"]["mean_benefit"]
        > 0
        and calibration["whitened_vs_matched"]["ci_low"] <= 0,
        "call_nondominated": calibration["call_nondominated"],
        "flop_nondominated": calibration["flop_nondominated"],
        "mps_sparse_slower_than_dense": runtime_1024["adaptive_realistic_sparse"][
            "median_seconds"
        ]
        > runtime_1024["adaptive_dense_all_exits"]["median_seconds"],
        "discovery_cv_two_repeats": common.read_json(common.COMPRESSION / "config.json")[
            "outer_repeats"
        ]
        == 2,
        "calibration_consumed_once": decision["calibration_consumed_once"],
        "v3_test_targets_unconsumed": not decision["v3_test_targets_consumed"],
        "adversarial_verification_passed": adversarial["passed"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"V4 authorization audit failed: {checks}")
    return {
        "classification": "historical_calibration_evidence_only_not_v4_confirmation",
        "expected": expected,
        "observed": observed,
        "checks": checks,
    }


def _record_identifier(
    records: list[dict[str, Any]], values: set[int], *, path: Path, key: str, value: Any
) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
        values.add(parsed)
        records.append(
            {"path": str(path.relative_to(common.REPO)), "field": key, "value": parsed}
        )
    elif isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        parsed = int(value)
        values.add(parsed)
        records.append(
            {"path": str(path.relative_to(common.REPO)), "field": key, "value": parsed}
        )


def _walk_json_identifiers(
    payload: Any,
    path: Path,
    records: list[dict[str, Any]],
    values: set[int],
    context: tuple[str, ...] = (),
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            next_context = (*context, str(key))
            if any(IDENTIFIER_TOKEN.search(part) for part in next_context):
                if isinstance(value, list):
                    for item in value:
                        _record_identifier(
                            records, values, path=path, key=".".join(next_context), value=item
                        )
                else:
                    _record_identifier(
                        records, values, path=path, key=".".join(next_context), value=value
                    )
            _walk_json_identifiers(value, path, records, values, next_context)
    elif isinstance(payload, list):
        for item in payload:
            _walk_json_identifiers(item, path, records, values, context)


def enumerate_identifiers() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    values: set[int] = set()
    files_scanned = 0
    for root in IDENTIFIER_ROOTS:
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            if "__pycache__" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            files_scanned += 1
            suffix = path.suffix.lower()
            try:
                if suffix == ".json":
                    _walk_json_identifiers(
                        json.loads(path.read_text(encoding="utf-8")), path, records, values
                    )
                elif suffix == ".csv":
                    with path.open(newline="", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        relevant = [
                            name for name in (reader.fieldnames or []) if IDENTIFIER_TOKEN.search(name)
                        ]
                        for row in reader:
                            for name in relevant:
                                _record_identifier(
                                    records, values, path=path, key=name, value=row[name]
                                )
                else:
                    text = path.read_text(encoding="utf-8", errors="strict")
                    for line_number, line in enumerate(text.splitlines(), start=1):
                        if re.search(r"seed|episode[_ -]?(?:id|idx|ordinal)?", line, re.I):
                            for match in re.finditer(r"(?<![A-Za-z0-9.])-?\d+(?![A-Za-z0-9.])", line):
                                _record_identifier(
                                    records,
                                    values,
                                    path=path,
                                    key=f"text_line_{line_number}",
                                    value=match.group(),
                                )
                    for match in re.finditer(r"seed[_-](\d+)", path.name, re.I):
                        _record_identifier(
                            records, values, path=path, key="filename_seed", value=match.group(1)
                        )
            except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
                raise RuntimeError(f"identifier scan failed for {path}") from exc

    # Load only checkpoint metadata explicitly carrying a seed.  No V3 target
    # container is opened.
    for root in IDENTIFIER_ROOTS[:-1]:
        for path in sorted(root.rglob("*.pt")):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if isinstance(payload, Mapping):
                for key in ("seed", "selected_seed", "training_seed"):
                    if key in payload:
                        _record_identifier(records, values, path=path, key=key, value=payload[key])

    v3_manifest = common.V3 / "cache/split_manifest.json"
    if common.sha256_file(v3_manifest) != common.EXPECTED["v3_split_manifest_sha256"]:
        raise RuntimeError("V3 split manifest drift during identifier enumeration")
    v3_payload = json.loads(v3_manifest.read_text(encoding="utf-8"))
    v3_sets = v3_payload["selection"]["split_episode_ordinals"]
    v3_test = sorted(int(value) for value in v3_sets["test"])
    return {
        "schema_version": 1,
        "scope": [str(path.relative_to(common.REPO)) for path in IDENTIFIER_ROOTS],
        "files_scanned": files_scanned,
        "records": records,
        "unique_numeric_ids": sorted(values),
        "unique_numeric_id_count": len(values),
        "v3_episode_sets": {key: sorted(map(int, value)) for key, value in v3_sets.items()},
        "v3_test_episode_ids": v3_test,
        "v3_test_targets_opened": False,
        "opaque_v3_combined_cache_sha256": common.sha256_file(
            common.V3 / "cache/cube_inputs.npz"
        ),
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze_matching_bins() -> dict[str, Any]:
    """Derive numeric bin edges from discovery-fit episodes only."""
    isolation = _load_module(common.DISCOVERY / "data_isolation.py", "v4_data_isolation")
    model_io = _load_module(
        common.REPO / "runs/lewm_adaptive_compute_v2/model_io.py", "v4_old_model_io"
    )
    with np.load(common.DISCOVERY / "cache/v3_train_only.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    split = isolation.grouped_discovery_split(84, 261001)
    fit_episodes = set(map(int, split["discovery_fit"]))
    fit = np.asarray([int(value) in fit_episodes for value in arrays["episode_id"]], dtype=bool)
    labels = model_io.extract_post_prediction_labels(
        common.SOURCE_H5, arrays["episode_id"], arrays["model_step"]
    )
    normalized = np.asarray(arrays["action"], dtype=np.float32)
    raw = normalized * common.FROZEN_ACTION_STD.reshape(1, 1, 1, 5) + common.FROZEN_ACTION_MEAN.reshape(
        1, 1, 1, 5
    ) if normalized.ndim == 4 else None
    # Stored action histories are [N,3,25]; the last 25-vector is the five
    # raw actions immediately preceding the target transition.
    last_block = normalized[:, -1].reshape(-1, common.FRAMESKIP, common.RAW_ACTION_DIM)
    raw_last = last_block * common.FROZEN_ACTION_STD.reshape(1, 1, -1) + common.FROZEN_ACTION_MEAN.reshape(
        1, 1, -1
    )
    action_magnitude = np.sqrt(np.square(raw_last.astype(np.float64)).sum(2).mean(1))
    motion = np.asarray(labels["block_disp"], dtype=np.float64)

    def quantiles(values: np.ndarray, probabilities: Iterable[float]) -> list[float]:
        result = []
        for value in np.quantile(values[fit], list(probabilities)):
            parsed = float(value)
            if not result or parsed > result[-1] + 1e-12:
                result.append(parsed)
        return result

    fine_motion = quantiles(motion, (0.25, 0.5, 0.625, 0.75, 0.875, 0.95))
    coarse_motion = quantiles(motion, (0.5, 0.75, 0.9))
    fine_action = quantiles(action_magnitude, (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875))
    coarse_action = quantiles(action_magnitude, (0.25, 0.5, 0.75))
    return {
        "schema_version": 1,
        "derivation": "discovery_fit episodes only; frozen before any V4 episode generation",
        "discovery_train_cache_sha256": common.sha256_file(
            common.DISCOVERY / "cache/v3_train_only.npz"
        ),
        "fit_episode_count": len(fit_episodes),
        "fit_row_count": int(fit.sum()),
        "fine": {
            "phase_interior_edges": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
            "target_motion_interior_edges": fine_motion,
            "action_magnitude_interior_edges": fine_action,
        },
        "coarse": {
            "phase_interior_edges": [0.25, 0.5, 0.75],
            "target_motion_interior_edges": coarse_motion,
            "action_magnitude_interior_edges": coarse_action,
        },
        "regime_definitions": {
            "impact": "current five-step contact window positive and preceding window negative",
            "contact": "current five-step contact window positive and not impact",
            "static": "no current contact and block displacement <= 0.0",
            "other_free_motion": "no current contact and block displacement > 0.0",
            "high_transport_threshold_discovery_q75": 0.06765174865722656,
        },
        "action_magnitude": "sqrt(mean over five raw actions of sum over five action coordinates of action^2))",
        "target_motion": "Euclidean block-position displacement over the target five-step interval",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=common.ROOT / "audit/prior_evidence_audit.json")
    args = parser.parse_args()
    common.verify_frozen_objects()
    manifests = [verify_artifact_manifest(root) for root in REQUIRED_TREES]
    files = []
    for root in REQUIRED_TREES:
        files.extend(inspect_file(path) for path in sorted(item for item in root.rglob("*") if item.is_file()))
    audit = {
        "schema_version": 1,
        "required_trees": [str(path.relative_to(common.REPO)) for path in REQUIRED_TREES],
        "artifact_manifests": manifests,
        "files": files,
        "file_count": len(files),
        "all_files_hashed": True,
        "all_nonforbidden_structured_files_opened": True,
        "forbidden_target_payloads_opened": False,
        "authorization_claims": audit_claims(),
        "frozen_objects": common.verify_frozen_objects(),
    }
    common.write_json(args.output, audit)
    common.write_json(common.ROOT / "audit/prior_identifier_inventory.json", enumerate_identifiers())
    common.write_json(common.ROOT / "audit/regime_matching_bins.json", freeze_matching_bins())
    print(
        json.dumps(
            {
                "audit": str(args.output),
                "files": len(files),
                "identifiers": str(common.ROOT / "audit/prior_identifier_inventory.json"),
                "v3_test_targets_opened": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
