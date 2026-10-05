#!/usr/bin/env python3
"""Independent exact and near-duplicate cohort isolation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from collections import defaultdict
from typing import Any

import numpy as np


PRUNED_DIRECTORY_NAMES = {
    "lewm_v5_generalization",
    "lewm_adaptive_compute_v3",
    "__pycache__",
    ".git",
}
SEED_KEYS = {
    "env_seed",
    "policy_seed",
    "oracle_np_seed",
    "seed",
    "torch_seed",
    "numpy_seed",
    "bootstrap_seed",
    "histogram_seed",
}
IDENTIFIER_KEYS = {
    "episode_id",
    "trajectory_id",
    "seed_source_episode_id",
    "identifier",
}


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def walk_json(
    runs_root: pathlib.Path, excluded_package: pathlib.Path
) -> list[pathlib.Path]:
    output = []
    for dirpath, dirnames, filenames in os.walk(runs_root):
        base = pathlib.Path(dirpath)
        dirnames[:] = [
            name for name in dirnames if name not in PRUNED_DIRECTORY_NAMES
        ]
        if base == excluded_package or excluded_package in base.parents:
            dirnames[:] = []
            continue
        for name in filenames:
            if name.endswith(".json"):
                output.append(base / name)
    return sorted(output)


def collect_scalar_keys(
    value: Any,
    path: pathlib.Path,
    seeds: dict[int, list[str]],
    identifiers: dict[str, list[str]],
    key: str | None = None,
) -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            collect_scalar_keys(child, path, seeds, identifiers, child_key)
    elif isinstance(value, list):
        for child in value:
            collect_scalar_keys(child, path, seeds, identifiers, key)
    elif key in SEED_KEYS and isinstance(value, int) and not isinstance(value, bool):
        seeds[int(value)].append(str(path))
    elif key in IDENTIFIER_KEYS and isinstance(value, str):
        identifiers[value].append(str(path))


def raw_npz_paths(
    runs_root: pathlib.Path, v5_raw: pathlib.Path
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    prior = []
    v5 = sorted(v5_raw.glob("*.npz"))
    for dirpath, dirnames, filenames in os.walk(runs_root):
        base = pathlib.Path(dirpath)
        dirnames[:] = [
            name for name in dirnames if name not in PRUNED_DIRECTORY_NAMES
        ]
        if base == v5_raw or v5_raw in base.parents:
            dirnames[:] = []
            continue
        relative_parts = [part.lower() for part in base.relative_to(runs_root).parts]
        if not any("raw" in part for part in relative_parts):
            continue
        for name in filenames:
            if name.endswith(".npz"):
                prior.append(base / name)
    return sorted(prior), v5


def extract_raw_fingerprint(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=False) as stored:
            keys = set(stored.files)
            vector_parts = []
            for key in (
                "qpos",
                "qvel",
                "privileged_block_0_pos",
                "privileged_block_0_quat",
                "privileged_target_block_pos",
                "privileged_target_block_yaw",
            ):
                if key in keys:
                    array = np.asarray(stored[key])
                    if len(array):
                        vector_parts.append(np.asarray(array[0], dtype=np.float64).ravel())
            vector = (
                np.concatenate(vector_parts) if vector_parts else np.empty(0, dtype=np.float64)
            )
            hashes = {}
            sketches = {}
            for key in ("action", "observation"):
                if key in keys:
                    array = np.asarray(stored[key])
                    hashes[key] = array_hash(array)
                    numeric = np.asarray(array, dtype=np.float64)
                    finite_rows = numeric[
                        np.isfinite(numeric).reshape(len(numeric), -1).all(axis=1)
                    ]
                    if len(finite_rows):
                        prefix = finite_rows[: min(len(finite_rows), 20)]
                        sketches[key] = np.concatenate(
                            [
                                prefix.ravel(),
                                finite_rows.mean(axis=0).ravel(),
                                finite_rows.std(axis=0).ravel(),
                            ]
                        )
            if "pixels" in keys:
                frame = np.asarray(stored["pixels"][0])
                hashes["pixels_initial"] = array_hash(frame)
                if frame.shape == (224, 224, 3):
                    grid = frame.reshape(14, 16, 14, 16, 3).mean(axis=(1, 3))
                    sketches["pixels_initial"] = np.concatenate(
                        [
                            grid.astype(np.float64).ravel(),
                            frame.mean(axis=(0, 1), dtype=np.float64),
                            frame.std(axis=(0, 1), dtype=np.float64),
                        ]
                    )
            return {
                "path": str(path),
                "keys": sorted(keys),
                "initial_vector": vector,
                "array_hashes": hashes,
                "sketches": sketches,
            }
    except (ValueError, OSError, KeyError):
        return None


def nearest_cross(
    left: np.ndarray, right: np.ndarray, *, chunk: int = 128
) -> np.ndarray:
    output = np.full(len(left), np.inf, dtype=np.float64)
    for start in range(0, len(left), chunk):
        stop = min(start + chunk, len(left))
        difference = left[start:stop, None, :] - right[None, :, :]
        output[start:stop] = np.sqrt(
            np.min(np.sum(difference * difference, axis=2), axis=1)
        )
    return output


def nearest_within(values: np.ndarray, *, chunk: int = 128) -> np.ndarray:
    output = np.full(len(values), np.inf, dtype=np.float64)
    for start in range(0, len(values), chunk):
        stop = min(start + chunk, len(values))
        difference = values[start:stop, None, :] - values[None, :, :]
        distance = np.sum(difference * difference, axis=2)
        rows = np.arange(start, stop)
        distance[np.arange(stop - start), rows] = np.inf
        output[start:stop] = np.sqrt(np.min(distance, axis=1))
    return output


def distance_summary(
    prior_records: list[dict[str, Any]],
    v5_records: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    by_dimension_prior: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
    by_dimension_v5: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for record in prior_records:
        value = record[field]
        if isinstance(value, dict):
            continue
        if len(value) and np.isfinite(value).all():
            by_dimension_prior[len(value)].append((record["path"], value))
    for record in v5_records:
        value = record[field]
        if isinstance(value, dict):
            continue
        if len(value) and np.isfinite(value).all():
            by_dimension_v5[len(value)].append((record["path"], value))

    groups = []
    suspicious = []
    for dimension in sorted(set(by_dimension_prior) & set(by_dimension_v5)):
        prior_items = by_dimension_prior[dimension]
        v5_items = by_dimension_v5[dimension]
        if len(prior_items) < 1 or len(v5_items) < 2:
            continue
        prior = np.stack([item[1] for item in prior_items])
        v5 = np.stack([item[1] for item in v5_items])
        combined = np.concatenate([prior, v5])
        scale = combined.std(axis=0)
        scale[scale < 1e-12] = 1.0
        center = combined.mean(axis=0)
        prior_scaled = (prior - center) / scale
        v5_scaled = (v5 - center) / scale
        cross = nearest_cross(v5_scaled, prior_scaled)
        within_v5 = nearest_within(v5_scaled)
        threshold = max(
            1e-8,
            0.01 * float(np.median(within_v5[np.isfinite(within_v5)])),
        )
        indices = np.flatnonzero(cross <= threshold)
        for index in indices[:20]:
            nearest_index = int(
                np.argmin(
                    np.sum(
                        (prior_scaled - v5_scaled[index][None, :]) ** 2, axis=1
                    )
                )
            )
            suspicious.append(
                {
                    "v5": v5_items[index][0],
                    "prior": prior_items[nearest_index][0],
                    "standardized_distance": float(cross[index]),
                    "threshold": threshold,
                }
            )
        groups.append(
            {
                "dimension": dimension,
                "prior_count": len(prior),
                "v5_count": len(v5),
                "cross_minimum": float(cross.min()),
                "cross_quantiles": {
                    "0.01": float(np.quantile(cross, 0.01)),
                    "0.05": float(np.quantile(cross, 0.05)),
                    "0.50": float(np.quantile(cross, 0.50)),
                },
                "within_v5_minimum": float(within_v5.min()),
                "within_v5_median": float(np.median(within_v5)),
                "near_duplicate_threshold": threshold,
                "cross_below_threshold": int(len(indices)),
            }
        )
    return {
        "groups": groups,
        "suspicious_pairs": suspicious,
        "passed": not suspicious,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    started = time.time()
    repo = args.repo.resolve()
    package = args.package.resolve()
    runs_root = repo / "runs"
    v5_raw = package / "data/v5_confirmation_raw"

    ledger = json.loads((package / "cohort_seed_ledger.json").read_text())
    confirmation = ledger["roles"]["v5_confirmation"]
    v5_ids = {item["episode_id"] for item in confirmation}
    v5_seed_tuples = {
        (
            int(item["env_seed"]),
            int(item["policy_seed"]),
            int(item["oracle_np_seed"]),
        )
        for item in confirmation
    }
    v5_seeds = {value for item in v5_seed_tuples for value in item}

    prior_seeds: dict[int, list[str]] = defaultdict(list)
    prior_ids: dict[str, list[str]] = defaultdict(list)
    json_paths = walk_json(runs_root, package)
    json_paths.extend(sorted((package / "data/package_smoke_raw").glob("*.json")))
    json_paths = sorted(set(json_paths))
    parse_failures = []
    for path in json_paths:
        try:
            payload = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            parse_failures.append(str(path))
            continue
        collect_scalar_keys(payload, path, prior_seeds, prior_ids)
    for dirpath, dirnames, filenames in os.walk(runs_root):
        base = pathlib.Path(dirpath)
        dirnames[:] = [
            name for name in dirnames if name not in PRUNED_DIRECTORY_NAMES
        ]
        if base == package or package in base.parents:
            dirnames[:] = []
            continue
        for name in filenames:
            stem = pathlib.Path(name).stem
            if stem:
                prior_ids[stem].append(str(base / name))
    for path in sorted((package / "data/package_smoke_raw").glob("*")):
        if path.is_file():
            prior_ids[path.stem].append(str(path))

    exact_seed_overlap = sorted(v5_seeds & set(prior_seeds))
    exact_id_overlap = sorted(v5_ids & set(prior_ids))
    low32_overlap = sorted(
        seed
        for seed in v5_seeds
        if any((prior & 0xFFFFFFFF) == (seed & 0xFFFFFFFF) for prior in prior_seeds)
    )
    # SeedSequence state is a deterministic transformed fingerprint.  Compare
    # four uint32 words for every structured prior seed.
    prior_states = {
        tuple(np.random.SeedSequence(seed).generate_state(4).tolist())
        for seed in prior_seeds
        if 0 <= seed < 2**63
    }
    transformed_overlap = []
    for seed in sorted(v5_seeds):
        state = tuple(np.random.SeedSequence(seed).generate_state(4).tolist())
        if state in prior_states:
            transformed_overlap.append(seed)

    prior_paths, v5_paths = raw_npz_paths(runs_root, v5_raw)
    prior_records = []
    v5_records = []
    for index, path in enumerate(prior_paths):
        record = extract_raw_fingerprint(path)
        if record is not None:
            prior_records.append(record)
        if (index + 1) % 250 == 0:
            print(
                json.dumps(
                    {
                        "phase": "prior_raw_fingerprints",
                        "processed": index + 1,
                        "total": len(prior_paths),
                    }
                ),
                flush=True,
            )
    for index, path in enumerate(v5_paths):
        record = extract_raw_fingerprint(path)
        if record is not None:
            v5_records.append(record)
        if (index + 1) % 250 == 0:
            print(
                json.dumps(
                    {
                        "phase": "v5_raw_fingerprints",
                        "processed": index + 1,
                        "total": len(v5_paths),
                    }
                ),
                flush=True,
            )

    exact_hash_overlap = {}
    for key in ("action", "observation", "pixels_initial"):
        prior_hashes: dict[str, list[str]] = defaultdict(list)
        for record in prior_records:
            if key in record["array_hashes"]:
                prior_hashes[record["array_hashes"][key]].append(record["path"])
        collisions = []
        for record in v5_records:
            value = record["array_hashes"].get(key)
            if value in prior_hashes:
                collisions.append(
                    {
                        "v5": record["path"],
                        "prior": prior_hashes[value],
                        "sha256": value,
                    }
                )
        exact_hash_overlap[key] = collisions

    initial_vectors = distance_summary(
        prior_records, v5_records, "initial_vector"
    )
    # For action/observation sketches, group each key explicitly.
    sketch_results = {}
    for key in ("action", "observation", "pixels_initial"):
        prior_sketch = [
            {"path": item["path"], "value": item["sketches"][key]}
            for item in prior_records
            if key in item["sketches"]
        ]
        v5_sketch = [
            {"path": item["path"], "value": item["sketches"][key]}
            for item in v5_records
            if key in item["sketches"]
        ]
        # Reuse distance helper under a common field name.
        sketch_results[key] = distance_summary(
            [
                {"path": item["path"], "initial_vector": item["value"]}
                for item in prior_sketch
            ],
            [
                {"path": item["path"], "initial_vector": item["value"]}
                for item in v5_sketch
            ],
            "initial_vector",
        )

    sidecar_initial_hashes_prior: dict[str, list[str]] = defaultdict(list)
    sidecar_initial_hashes_v5: dict[str, list[str]] = defaultdict(list)
    for path in json_paths:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        value = payload.get("initial_state_sha256") if isinstance(payload, dict) else None
        if isinstance(value, str):
            sidecar_initial_hashes_prior[value].append(str(path))
    for path in sorted(v5_raw.glob("*.json")):
        payload = json.loads(path.read_text())
        value = payload.get("initial_state_sha256")
        if isinstance(value, str):
            sidecar_initial_hashes_v5[value].append(str(path))
    initial_hash_collisions = [
        {
            "sha256": value,
            "v5": sidecar_initial_hashes_v5[value],
            "prior": sidecar_initial_hashes_prior[value],
        }
        for value in sorted(set(sidecar_initial_hashes_v5) & set(sidecar_initial_hashes_prior))
    ]
    within_v5_initial_duplicates = [
        {"sha256": value, "paths": paths}
        for value, paths in sidecar_initial_hashes_v5.items()
        if len(paths) > 1
    ]

    checks = {
        "exact_episode_identifier_overlap_absent": not exact_id_overlap,
        "exact_structured_seed_overlap_absent": not exact_seed_overlap,
        "low32_seed_overlap_absent": not low32_overlap,
        "seedsequence_transformed_overlap_absent": not transformed_overlap,
        "exact_initial_state_hash_overlap_absent": not initial_hash_collisions,
        "within_v5_initial_state_hashes_unique": not within_v5_initial_duplicates,
        "exact_action_array_overlap_absent": not exact_hash_overlap["action"],
        "exact_observation_array_overlap_absent": not exact_hash_overlap["observation"],
        "exact_initial_pixel_overlap_absent": not exact_hash_overlap[
            "pixels_initial"
        ],
        "initial_state_near_duplicate_search_passed": initial_vectors["passed"],
        "action_sketch_near_duplicate_search_passed": sketch_results["action"]["passed"],
        "observation_sketch_near_duplicate_search_passed": sketch_results[
            "observation"
        ]["passed"],
        "initial_pixel_sketch_near_duplicate_search_passed": sketch_results[
            "pixels_initial"
        ]["passed"],
        "all_1600_v5_raw_archives_fingerprinted": len(v5_records) == 1600,
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "scope": {
            "runs_root": str(runs_root),
            "explicitly_pruned": sorted(PRUNED_DIRECTORY_NAMES),
            "v3_test_target_directory_not_opened": True,
            "v5_generalization_not_used": True,
            "prior_json_paths_scanned": len(json_paths),
            "prior_raw_npz_candidates": len(prior_paths),
            "prior_raw_npz_fingerprinted": len(prior_records),
            "v5_raw_npz_fingerprinted": len(v5_records),
            "json_parse_failures": parse_failures,
            "base_model_training_episode_level_raw_corpus_accessible": False,
            "base_model_training_surface_note": (
                "The repository preserves base weights/config and many prior "
                "adaptive-compute cohorts, but not a complete episode-level raw "
                "corpus for the original base-model training. Observation-level "
                "absence from that entire historical corpus is therefore not "
                "proven by this scan."
            ),
        },
        "exact_identifier_seed_tests": {
            "v5_confirmation_identifier_count": len(v5_ids),
            "v5_confirmation_seed_tuple_count": len(v5_seed_tuples),
            "prior_structured_seed_count": len(prior_seeds),
            "prior_structured_identifier_count": len(prior_ids),
            "exact_seed_overlap": exact_seed_overlap,
            "exact_identifier_overlap": exact_id_overlap,
            "low32_seed_overlap": low32_overlap,
            "seedsequence_transformed_overlap": transformed_overlap,
        },
        "exact_raw_fingerprints": {
            "initial_state_hash_collisions": initial_hash_collisions,
            "within_v5_initial_state_hash_duplicates": within_v5_initial_duplicates,
            "action_hash_collisions": exact_hash_overlap["action"],
            "observation_hash_collisions": exact_hash_overlap["observation"],
            "initial_pixel_hash_collisions": exact_hash_overlap[
                "pixels_initial"
            ],
        },
        "near_duplicate_tests": {
            "initial_state_numeric_vectors": initial_vectors,
            "action_prefix_mean_std_sketches": sketch_results["action"],
            "observation_prefix_mean_std_sketches": sketch_results["observation"],
            "initial_pixel_14x14_grid_sketches": sketch_results[
                "pixels_initial"
            ],
            "threshold_definition": (
                "standardized Euclidean distance <= max(1e-8, 1% of the "
                "within-V5 median nearest-neighbor distance), grouped by exact "
                "fingerprint dimension"
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "runtime_seconds": time.time() - started,
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "prior_raw": len(prior_records),
                "v5_raw": len(v5_records),
                "runtime_seconds": result["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
