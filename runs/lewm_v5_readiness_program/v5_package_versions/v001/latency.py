#!/usr/bin/env python3
"""Synchronized package-smoke or V5 latency, separate from FLOP inference."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np

from cycle_common import (
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    REPO_ROOT,
    ROLE_COUNTS,
    ROOT,
    atomic_json,
    read_json,
    sha256_file,
)
from runner import dense_shadow, load_gate_tensors, load_scientific_stack, manual_sparse
from verify_pre_v5 import verify as verify_pre_v5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("package_smoke", "v5_confirmation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    arguments = parser.parse_args()
    output_path = ROOT / f"metrics/{arguments.role}_latency.json"
    if output_path.exists():
        raise RuntimeError("latency artifact is immutable and already exists")
    verify_pre_v5()
    if arguments.role == "v5_confirmation" and not (ROOT / "analysis_result.json").exists():
        raise RuntimeError("confirmation latency is post-analysis")
    manifest = read_json(ROOT / f"data/{arguments.role}_execution_manifest.json")
    input_path = REPO_ROOT / manifest["latency_input_path"]
    if sha256_file(input_path) != manifest["latency_input_sha256"]:
        raise RuntimeError("latency input hash drift")
    with np.load(input_path, allow_pickle=False) as stored:
        history_np = stored["history"].copy()
        actions_np = stored["actions"].copy()
        expected_calls_np = stored["expected_calls"].astype(np.int64)

    torch, runtime, model_io, device, base, contract, stack = load_scientific_stack(arguments.device)
    solver, v1, models, provenance = stack
    gate = load_gate_tensors(torch, device)
    before = runtime.module_audit(base, solver, v1)
    timings = []
    equivalence = []

    for batch_size in (1, 32, 256, 1024):
        if batch_size > len(history_np):
            continue
        history = torch.as_tensor(history_np[:batch_size], device=device)
        actions = torch.as_tensor(actions_np[:batch_size], device=device)
        expected_calls = torch.as_tensor(expected_calls_np[:batch_size], device=device)

        def fixed_depth_one() -> tuple[Any, Any]:
            base_prediction = runtime.base_predict(base, history, actions)
            selected = solver.forward_selected(
                history, actions, base_prediction, torch.ones_like(expected_calls)
            )
            return selected, torch.ones_like(expected_calls)

        def adaptive_sparse() -> tuple[Any, Any]:
            base_prediction = runtime.base_predict(base, history, actions)
            selected, calls, _, _ = manual_sparse(
                torch, solver, gate, history, actions, base_prediction
            )
            return selected, calls

        def adaptive_dense_shadow() -> tuple[Any, Any]:
            base_prediction = runtime.base_predict(base, history, actions)
            exits, calls, _, _ = dense_shadow(
                torch, solver, gate, history, actions, base_prediction
            )
            selected = exits[torch.arange(len(history), device=device), calls - 1]
            return selected, calls

        paths: list[tuple[str, Callable[[], tuple[Any, Any]]]] = [
            ("fixed_depth_1", fixed_depth_one),
            ("actual_adaptive_sparse", adaptive_sparse),
            ("dense_all_exit_shadow", adaptive_dense_shadow),
        ]
        path_outputs = {}
        for name, function in paths:
            for _ in range(arguments.warmups):
                function()
                runtime.synchronize(device)
            measurements = []
            output = calls = None
            for _ in range(arguments.repetitions):
                runtime.synchronize(device)
                started = time.perf_counter_ns()
                output, calls = function()
                runtime.synchronize(device)
                measurements.append((time.perf_counter_ns() - started) / 1e9)
            assert output is not None and calls is not None
            path_outputs[name] = (output, calls)
            timings.append(
                {
                    "path": name,
                    "batch_size": batch_size,
                    "warmups": arguments.warmups,
                    "repetitions": arguments.repetitions,
                    "all_seconds": measurements,
                    "median_seconds": float(np.median(measurements)),
                    "p05_seconds": float(np.quantile(measurements, 0.05)),
                    "p95_seconds": float(np.quantile(measurements, 0.95)),
                    "throughput_rows_per_second_at_median": float(
                        batch_size / np.median(measurements)
                    ),
                }
            )
        sparse_output, sparse_calls = path_outputs["actual_adaptive_sparse"]
        dense_output, dense_calls = path_outputs["dense_all_exit_shadow"]
        equivalence.append(
            {
                "batch_size": batch_size,
                "sparse_calls_match_sealed_execution_prefix": bool(
                    torch.equal(sparse_calls, expected_calls)
                ),
                "dense_calls_match_sealed_execution_prefix": bool(
                    torch.equal(dense_calls, expected_calls)
                ),
                "sparse_dense_calls_exact": bool(torch.equal(sparse_calls, dense_calls)),
                "sparse_dense_outputs_within_sealed_contract": bool(
                    torch.allclose(
                        sparse_output,
                        dense_output,
                        rtol=NUMERICAL_RTOL,
                        atol=NUMERICAL_ATOL,
                    )
                ),
                "sparse_dense_max_abs": float(
                    (sparse_output - dense_output).abs().max().item()
                ),
            }
        )

    after = runtime.module_audit(base, solver, v1)
    audit = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "device": str(device),
        "role": arguments.role,
        "measurement": "synchronized latent/action-input through base prediction and solver/gate path",
        "includes": [
            "frozen base prediction",
            "solver blocks",
            "counted causal feature construction",
            "two compiled affine gate heads",
            "routing/indexing",
            "device synchronization",
        ],
        "excludes": ["simulator rollout", "pixel encoder", "FLOP-based terminal inference"],
        "actual_latency_separate_from_fully_counted_flops": True,
        "statistical_and_flop_verdict_independent_of_latency": True,
        "base_provenance": provenance,
        "module_before": before,
        "module_after": after,
        "equivalence": equivalence,
        "timings": timings,
        "passed": bool(
            before == after
            and before["passed"]
            and all(
                item["sparse_calls_match_sealed_execution_prefix"]
                and item["dense_calls_match_sealed_execution_prefix"]
                and item["sparse_dense_calls_exact"]
                and item["sparse_dense_outputs_within_sealed_contract"]
                and item["sparse_dense_max_abs"] <= NUMERICAL_MAX_ABS
                for item in equivalence
            )
        ),
        "energy_measured": False,
        "energy_reason": "no reliable per-path energy measurement interface available",
        "v5_outcome_episodes": ROLE_COUNTS[arguments.role]
        if arguments.role == "v5_confirmation"
        else 0,
    }
    atomic_json(output_path, audit, exclusive=True)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
