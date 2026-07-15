#!/usr/bin/env python3
"""Preregistered end-to-end MPS latency and deployment audit."""

from __future__ import annotations

import json
import math
import statistics as py_statistics
import time
from typing import Any, Callable

import numpy as np
import torch

import common
import runtime


@torch.inference_mode()
def _fixed(base_model, solver, history, actions, depth: int):
    base = runtime.base_predict(base_model, history, actions)
    return solver(history, actions, base, max_depth=int(depth))[int(depth)]


@torch.inference_mode()
def _mixture(base_model, solver, history, actions, calls: torch.Tensor):
    base = runtime.base_predict(base_model, history, actions)
    return solver.forward_selected(history, actions, base, calls)


@torch.inference_mode()
def _dense(base_model, solver, gate, models, history, actions):
    base = runtime.base_predict(base_model, history, actions)
    return runtime.dense_adaptive(solver, gate, models, history, actions, base)[:2]


@torch.inference_mode()
def _reference(base_model, solver, gate, models, history, actions):
    base = runtime.base_predict(base_model, history, actions)
    return runtime.sparse_adaptive_reference(solver, gate, models, history, actions, base)


@torch.inference_mode()
def _optimized(base_model, solver, gate, models, history, actions):
    base = runtime.base_predict(base_model, history, actions)
    return runtime.sparse_adaptive_optimized(solver, gate, models, history, actions, base)


def _memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "mps":
        return {"current_allocated_bytes": None, "driver_allocated_bytes": None}
    current = int(torch.mps.current_allocated_memory()) if hasattr(torch.mps, "current_allocated_memory") else None
    driver = int(torch.mps.driver_allocated_memory()) if hasattr(torch.mps, "driver_allocated_memory") else None
    return {"current_allocated_bytes": current, "driver_allocated_bytes": driver}


def benchmark(
    function: Callable[[], Any],
    device: torch.device,
    *,
    batch_size: int,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        function()
        runtime.synchronize(device)
    elapsed = []
    memory = []
    for _ in range(repetitions):
        before = _memory(device)
        start = time.perf_counter_ns()
        function()
        runtime.synchronize(device)
        value = (time.perf_counter_ns() - start) / 1e9
        elapsed.append(value)
        after = _memory(device)
        memory.append({"before": before, "after": after})
    values = np.asarray(elapsed, dtype=np.float64)
    median = float(np.median(values))
    observed_driver = [
        item[side]["driver_allocated_bytes"]
        for item in memory
        for side in ("before", "after")
        if item[side]["driver_allocated_bytes"] is not None
    ]
    return {
        "batch_size": int(batch_size),
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "all_seconds": elapsed,
        "minimum_seconds": float(values.min()),
        "median_seconds": median,
        "p05_seconds": float(np.quantile(values, 0.05)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "maximum_seconds": float(values.max()),
        "mean_seconds": float(values.mean()),
        "throughput_rows_per_second_at_median": float(batch_size / median),
        "observed_max_driver_allocated_bytes": max(observed_driver) if observed_driver else None,
        "peak_memory_semantics": (
            "observed MPS driver allocation around synchronized calls; a resettable reliable per-path "
            "peak-memory counter was unavailable"
            if device.type == "mps"
            else "unavailable on non-MPS device"
        ),
    }


def run(device: torch.device) -> dict[str, Any]:
    output = common.ROOT / "metrics/runtime.json"
    if output.exists():
        raise RuntimeError("runtime benchmark already exists")
    if device.type != "mps":
        raise RuntimeError("the preregistered deployment benchmark requires MPS")
    config = common.load_config()["runtime_benchmark"]
    batch_sizes = [int(value) for value in config["batch_sizes"]]
    if batch_sizes != [1, 32, 256, 1024]:
        raise RuntimeError("runtime batch sizes changed after preregistration")
    encoded_path = common.ROOT / "data/confirmation_encoded.npz"
    with np.load(encoded_path, allow_pickle=False) as stored:
        history_all = stored["history"]
        action_all = stored["action"]
    with np.load(common.ROOT / "metrics/primary_bootstrap_replicates.npz", allow_pickle=False) as stored:
        adaptive_calls = stored["adaptive_calls"]
        exact_total_calls = stored["exact_total_seeded_calls"]
        equal_call_calls = stored["equal_call_seeded_calls"]
    primary = common.read_json(common.ROOT / "metrics/primary_confirmation.json")
    base_model, _, base_provenance = runtime.load_base_model(device)
    solver, v1, models = runtime.load_solver(device)
    gate = runtime.load_gate(device)
    before = runtime.module_audit(base_model, solver, v1, gate.model)
    rows = []
    equivalence = []
    for batch_size in batch_sizes:
        history = torch.as_tensor(np.ascontiguousarray(history_all[:batch_size]), device=device)
        actions = torch.as_tensor(np.ascontiguousarray(action_all[:batch_size]), device=device)
        exact_calls_tensor = torch.as_tensor(
            np.ascontiguousarray(exact_total_calls[:batch_size]), dtype=torch.long, device=device
        )
        equal_calls_tensor = torch.as_tensor(
            np.ascontiguousarray(equal_call_calls[:batch_size]), dtype=torch.long, device=device
        )
        base = runtime.base_predict(base_model, history, actions)
        dense_out, dense_calls, _, _ = runtime.dense_adaptive(
            solver, gate, models, history, actions, base
        )
        reference_out, reference_calls = runtime.sparse_adaptive_reference(
            solver, gate, models, history, actions, base
        )
        optimized_out, optimized_calls = runtime.sparse_adaptive_optimized(
            solver, gate, models, history, actions, base
        )
        equivalence.append(
            {
                "batch_size": batch_size,
                "dense_reference_calls_exact": bool(torch.equal(dense_calls, reference_calls)),
                "dense_optimized_calls_exact": bool(torch.equal(dense_calls, optimized_calls)),
                "calls_match_frozen_confirmation_prefix": bool(
                    np.array_equal(dense_calls.cpu().numpy(), adaptive_calls[:batch_size])
                ),
                "dense_reference_max_abs_output": float(
                    (dense_out - reference_out).abs().max().cpu()
                ),
                "dense_optimized_max_abs_output": float(
                    (dense_out - optimized_out).abs().max().cpu()
                ),
                "dense_reference_outputs_equivalent": bool(
                    torch.allclose(dense_out, reference_out, rtol=2e-5, atol=2e-6)
                ),
                "dense_optimized_outputs_equivalent": bool(
                    torch.allclose(dense_out, optimized_out, rtol=2e-5, atol=2e-6)
                ),
            }
        )
        functions: list[tuple[str, Callable[[], Any]]] = []
        for depth in (1, 2, 3, 4):
            functions.append(
                (
                    f"fixed_d{depth}",
                    lambda depth=depth: _fixed(base_model, solver, history, actions, depth),
                )
            )
        functions.extend(
            [
                (
                    "exact_total_flop_seeded_integer_mixture",
                    lambda: _mixture(base_model, solver, history, actions, exact_calls_tensor),
                ),
                (
                    "equal_call_seeded_mixture_secondary",
                    lambda: _mixture(base_model, solver, history, actions, equal_calls_tensor),
                ),
                (
                    "adaptive_dense_all_exits",
                    lambda: _dense(base_model, solver, gate, models, history, actions),
                ),
                (
                    "adaptive_reference_sparse",
                    lambda: _reference(base_model, solver, gate, models, history, actions),
                ),
                (
                    "adaptive_optimized_sparse",
                    lambda: _optimized(base_model, solver, gate, models, history, actions),
                ),
            ]
        )
        timing_by_name = {}
        for name, function in functions:
            timing = benchmark(
                function,
                device,
                batch_size=batch_size,
                warmups=int(config["warmups"]),
                repetitions=int(config["repetitions"]),
            )
            timing_by_name[name] = timing
            rows.append({"path": name, **timing})
        fixed_medians = np.asarray(
            [timing_by_name[f"fixed_d{depth}"]["median_seconds"] for depth in (1, 2, 3, 4)]
        )
        exact_probabilities = np.asarray(
            primary["exact_total_flop_mixture"]["analytic_probabilities"], dtype=np.float64
        )
        equal_probabilities = np.asarray(
            primary["equal_call_mixture_secondary"]["analytic_probabilities"], dtype=np.float64
        )
        rows.append(
            {
                "path": "exact_total_flop_analytic_mixture_expected_latency",
                "batch_size": batch_size,
                "measurement": "derived_from_preregistered_probability_weighted_fixed_exit_medians",
                "median_seconds": float(exact_probabilities @ fixed_medians),
                "throughput_rows_per_second_at_median": float(
                    batch_size / (exact_probabilities @ fixed_medians)
                ),
            }
        )
        rows.append(
            {
                "path": "equal_call_analytic_mixture_expected_latency_secondary",
                "batch_size": batch_size,
                "measurement": "derived_from_preregistered_probability_weighted_fixed_exit_medians",
                "median_seconds": float(equal_probabilities @ fixed_medians),
                "throughput_rows_per_second_at_median": float(
                    batch_size / (equal_probabilities @ fixed_medians)
                ),
            }
        )
    after = runtime.module_audit(base_model, solver, v1, gate.model)
    ratios = []
    for batch_size in batch_sizes:
        reference = next(
            row
            for row in rows
            if row["path"] == "adaptive_reference_sparse" and row["batch_size"] == batch_size
        )
        optimized = next(
            row
            for row in rows
            if row["path"] == "adaptive_optimized_sparse" and row["batch_size"] == batch_size
        )
        ratios.append(float(optimized["median_seconds"] / reference["median_seconds"]))
    geometric_ratio = float(math.exp(np.mean(np.log(ratios))))
    if geometric_ratio <= float(config["improved_geomean_ratio_max"]) and max(ratios) <= float(
        config["improved_any_batch_ratio_max"]
    ):
        status = "latency_improved"
    elif geometric_ratio <= float(config["neutral_geomean_ratio_max"]) and max(ratios) <= float(
        config["neutral_any_batch_ratio_max"]
    ):
        status = "latency_neutral"
    else:
        status = "latency_regressed"
    passed = bool(
        before["passed"]
        and after["passed"]
        and before == after
        and all(
            item["dense_reference_calls_exact"]
            and item["dense_optimized_calls_exact"]
            and item["calls_match_frozen_confirmation_prefix"]
            and item["dense_reference_outputs_equivalent"]
            and item["dense_optimized_outputs_equivalent"]
            for item in equivalence
        )
    )
    result = {
        "schema_version": 1,
        "device": str(device),
        "batch_sizes": batch_sizes,
        "warmups": int(config["warmups"]),
        "repetitions": int(config["repetitions"]),
        "includes": [
            "common frozen base prediction",
            "solver blocks",
            "feature construction",
            "frozen normalization and linear head",
            "device synchronization",
            "routing control flow and indexing",
        ],
        "base_model_provenance": base_provenance,
        "module_audit_before": before,
        "module_audit_after": after,
        "sparse_equivalence": equivalence,
        "timings": rows,
        "optimized_vs_reference_median_ratios": dict(zip(map(str, batch_sizes), ratios)),
        "optimized_vs_reference_geometric_mean_ratio": geometric_ratio,
        "deployment_status": status,
        "energy": {
            "measured": False,
            "reason": "no reliable per-path energy measurement interface was available",
        },
        "passed": passed,
        "statistical_and_flop_verdict_independent_of_latency": True,
    }
    if not passed:
        raise RuntimeError(f"runtime equivalence/deployment audit failed: {result}")
    common.write_json(output, result, exclusive=True)
    decision_path = common.ROOT / "decision.json"
    decision = common.read_json(decision_path)
    decision["runtime_deployment_status"] = status
    decision["runtime_metrics_sha256"] = common.sha256_file(output)
    common.write_json(decision_path, decision)
    return result


if __name__ == "__main__":
    result = run(runtime.choose_device("mps"))
    print(json.dumps({"deployment_status": result["deployment_status"], "passed": result["passed"]}, sort_keys=True))
