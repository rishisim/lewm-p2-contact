#!/usr/bin/env python3
"""Synchronized latency and resource reporting, separate from FLOP inference."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np

from study_common import (
    ATTEMPT_ROOT,
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    REGIMES,
    REPO_ROOT,
    append_ledger,
    assert_runtime_contract,
    atomic_json,
    complete_state,
    latency_input_path,
    load_v5_runner,
    read_json,
    relative_to_repo,
    sha256_file,
    verify_pre_outcome_seal,
)


def mps_memory(torch: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "current_allocated_bytes": None,
        "driver_allocated_bytes": None,
    }
    if not torch.backends.mps.is_available():
        return result
    for name, function_name in (
        ("current_allocated_bytes", "current_allocated_memory"),
        ("driver_allocated_bytes", "driver_allocated_memory"),
    ):
        function = getattr(torch.mps, function_name, None)
        if callable(function):
            try:
                result[name] = int(function())
            except RuntimeError:
                result[name] = None
    return result


def benchmark_path(
    runtime: Any,
    device: Any,
    function: Callable[[], tuple[Any, Any]],
    *,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    for _ in range(warmups):
        function()
        runtime.synchronize(device)
    measurements = []
    output = calls = None
    for _ in range(repetitions):
        runtime.synchronize(device)
        started = time.perf_counter_ns()
        output, calls = function()
        runtime.synchronize(device)
        measurements.append((time.perf_counter_ns() - started) / 1e9)
    assert output is not None and calls is not None
    return (
        {
            "warmups": warmups,
            "repetitions": repetitions,
            "all_seconds": measurements,
            "median_seconds": float(np.median(measurements)),
            "p05_seconds": float(np.quantile(measurements, 0.05)),
            "p95_seconds": float(np.quantile(measurements, 0.95)),
        },
        (output, calls),
    )


def main() -> None:
    assert_runtime_contract("evaluation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    arguments = parser.parse_args()
    output = ATTEMPT_ROOT / "metrics/latency_and_resources.json"
    if output.exists():
        complete_state(
            "LATENCY_AND_RESOURCE_REPORTING",
            "INDEPENDENT_VERIFICATION",
            evidence_path=output,
            checkpoint_name="v005_latency_resources_reused",
            next_action="run independent read-only verification",
        )
        print(json.dumps(read_json(output), sort_keys=True))
        return
    verify_pre_outcome_seal()
    analysis = read_json(ATTEMPT_ROOT / "analysis_result.json")
    v5 = load_v5_runner()
    torch, runtime, _, device, base, _, stack = v5.load_scientific_stack(
        arguments.device
    )
    solver, v1, _, provenance = stack
    gate = v5.load_gate_tensors(torch, device)
    before = runtime.module_audit(base, solver, v1)
    if not before["passed"]:
        raise RuntimeError("latency module audit failed before measurement")
    memory_before = mps_memory(torch)
    regime_results = {}
    for regime in REGIMES:
        input_path = latency_input_path(regime)
        manifest = read_json(
            ATTEMPT_ROOT / f"data/target/{regime}/execution_manifest.json"
        )
        if (
            manifest["latency_input_path"] != relative_to_repo(input_path)
            or manifest["latency_input_sha256"] != sha256_file(input_path)
        ):
            raise RuntimeError(f"latency input seal drift for {regime}")
        with np.load(input_path, allow_pickle=False) as stored:
            history_np = stored["history"].copy()
            actions_np = stored["actions"].copy()
            expected_np = stored["expected_calls"].astype(np.int64)
        timings = []
        equivalence = []
        for batch_size in (1, 32, 256, 1024):
            history = torch.as_tensor(
                history_np[:batch_size], device=device
            )
            actions = torch.as_tensor(
                actions_np[:batch_size], device=device
            )
            expected_calls = torch.as_tensor(
                expected_np[:batch_size], device=device
            )

            def fixed_depth(depth: int) -> Callable[[], tuple[Any, Any]]:
                def function() -> tuple[Any, Any]:
                    base_prediction = runtime.base_predict(
                        base, history, actions
                    )
                    calls = torch.full_like(expected_calls, depth)
                    selected = solver.forward_selected(
                        history, actions, base_prediction, calls
                    )
                    return selected, calls

                return function

            def adaptive() -> tuple[Any, Any]:
                base_prediction = runtime.base_predict(base, history, actions)
                selected, calls, _, _ = v5.manual_sparse(
                    torch,
                    solver,
                    gate,
                    history,
                    actions,
                    base_prediction,
                )
                return selected, calls

            def dense_shadow() -> tuple[Any, Any]:
                base_prediction = runtime.base_predict(base, history, actions)
                exits, calls, _, _ = v5.dense_shadow(
                    torch,
                    solver,
                    gate,
                    history,
                    actions,
                    base_prediction,
                )
                selected = exits[
                    torch.arange(len(history), device=device), calls - 1
                ]
                return selected, calls

            functions: list[
                tuple[str, Callable[[], tuple[Any, Any]]]
            ] = [
                (f"fixed_depth_{depth}", fixed_depth(depth))
                for depth in range(1, 5)
            ] + [
                ("actual_adaptive_sparse", adaptive),
                ("dense_all_exit_shadow", dense_shadow),
            ]
            path_outputs = {}
            for name, function in functions:
                measurement, observed = benchmark_path(
                    runtime,
                    device,
                    function,
                    warmups=arguments.warmups,
                    repetitions=arguments.repetitions,
                )
                measurement.update(
                    {
                        "path": name,
                        "batch_size": batch_size,
                        "throughput_rows_per_second_at_median": float(
                            batch_size / measurement["median_seconds"]
                        ),
                    }
                )
                timings.append(measurement)
                path_outputs[name] = observed
            sparse_output, sparse_calls = path_outputs[
                "actual_adaptive_sparse"
            ]
            dense_output, dense_calls = path_outputs[
                "dense_all_exit_shadow"
            ]
            delta = (sparse_output - dense_output).abs()
            equivalence.append(
                {
                    "batch_size": batch_size,
                    "sparse_calls_match_execution_prefix": bool(
                        torch.equal(sparse_calls, expected_calls)
                    ),
                    "dense_calls_match_execution_prefix": bool(
                        torch.equal(dense_calls, expected_calls)
                    ),
                    "sparse_dense_calls_exact": bool(
                        torch.equal(sparse_calls, dense_calls)
                    ),
                    "sparse_dense_output_allclose": bool(
                        torch.allclose(
                            sparse_output,
                            dense_output,
                            rtol=NUMERICAL_RTOL,
                            atol=NUMERICAL_ATOL,
                        )
                    ),
                    "sparse_dense_max_abs": float(delta.max().item()),
                }
            )
        expected_mixtures = {}
        for endpoint, mixture_name in (
            ("raw", "raw_analytic_mixture"),
            ("fixed_whitened", "fixed_whitened_analytic_mixture"),
        ):
            mixture = analysis["regimes"][regime]["compute"][mixture_name]
            rows = []
            for batch_size in (1, 32, 256, 1024):
                lower = next(
                    item
                    for item in timings
                    if item["path"]
                    == f"fixed_depth_{mixture['depth_lower']}"
                    and item["batch_size"] == batch_size
                )
                upper = next(
                    item
                    for item in timings
                    if item["path"]
                    == f"fixed_depth_{mixture['depth_upper']}"
                    and item["batch_size"] == batch_size
                )
                weight = float(mixture["weight_upper"])
                expected_seconds = (
                    (1.0 - weight) * lower["median_seconds"]
                    + weight * upper["median_seconds"]
                )
                rows.append(
                    {
                        "batch_size": batch_size,
                        "depth_lower": mixture["depth_lower"],
                        "depth_upper": mixture["depth_upper"],
                        "weight_upper": weight,
                        "expected_median_seconds": expected_seconds,
                        "expected_throughput_rows_per_second": (
                            batch_size / expected_seconds
                        ),
                        "interpretation": (
                            "transition-independent expected latency; "
                            "not a FLOP substitution"
                        ),
                    }
                )
            expected_mixtures[endpoint] = rows
        regime_results[regime] = {
            "timings": timings,
            "equivalence": equivalence,
            "analytic_expected_latency": expected_mixtures,
            "compute_resources": analysis["regimes"][regime]["compute"],
            "passed": all(
                item["sparse_calls_match_execution_prefix"]
                and item["dense_calls_match_execution_prefix"]
                and item["sparse_dense_calls_exact"]
                and item["sparse_dense_output_allclose"]
                and item["sparse_dense_max_abs"] <= NUMERICAL_MAX_ABS
                for item in equivalence
            ),
        }
        print(f"measured latency {regime}", flush=True)
    runtime.synchronize(device)
    after = runtime.module_audit(base, solver, v1)
    memory_after = mps_memory(torch)
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "device": str(device),
        "measurement": (
            "synchronized latent/action-input through frozen base prediction "
            "and solver/gate path"
        ),
        "includes": [
            "frozen base prediction",
            "solver blocks",
            "counted causal features",
            "two compiled affine gate heads",
            "routing and indexing",
            "device synchronization",
        ],
        "excludes": [
            "simulator rollout",
            "pixel encoder",
            "FLOP-based terminal inference",
        ],
        "regimes": regime_results,
        "module_before": before,
        "module_after": after,
        "base_provenance": provenance,
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "interpretation": (
                "observed MPS allocation around the combined benchmark; "
                "not per-path peak memory"
            ),
        },
        "energy": {
            "measured": False,
            "reason": (
                "no reliable resettable per-path energy interface is "
                "available in this runtime"
            ),
        },
        "actual_latency_separate_from_fully_counted_flops": True,
        "statistical_and_flop_verdict_independent_of_latency": True,
        "passed": before == after
        and before["passed"]
        and all(item["passed"] for item in regime_results.values()),
        "target_outcome_episodes": 9_000,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError("latency/resource reporting failed integrity checks")
    append_ledger(
        "latency_and_resources_complete",
        path=relative_to_repo(output),
        sha256=sha256_file(output),
        energy_measured=False,
    )
    complete_state(
        "LATENCY_AND_RESOURCE_REPORTING",
        "INDEPENDENT_VERIFICATION",
        evidence_path=output,
        checkpoint_name="v005_latency_resources_complete",
        next_action="run independent read-only verification",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
