#!/usr/bin/env python3
"""Synchronized, end-to-end latency benchmark kept separate from FLOP claims."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import numpy as np
import torch

import common


@torch.inference_mode()
def _fixed(runtime, base_model, solver, history, action, depth: int):
    base = runtime.base_predict(base_model, history, action)
    return solver(history, action, base, max_depth=depth)[depth]


@torch.inference_mode()
def _mixture(runtime, base_model, solver, history, action, calls):
    base = runtime.base_predict(base_model, history, action)
    return solver.forward_selected(history, action, base, calls)


@torch.inference_mode()
def _dense(runtime, base_model, solver, gate, models, history, action):
    base = runtime.base_predict(base_model, history, action)
    return runtime.dense_adaptive(solver, gate, models, history, action, base)[:2]


@torch.inference_mode()
def _reference(runtime, base_model, solver, gate, models, history, action):
    base = runtime.base_predict(base_model, history, action)
    return runtime.sparse_adaptive_reference(solver, gate, models, history, action, base)


@torch.inference_mode()
def _optimized(runtime, base_model, solver, gate, models, history, action):
    base = runtime.base_predict(base_model, history, action)
    return runtime.sparse_adaptive_optimized(solver, gate, models, history, action, base)


def _benchmark(
    function: Callable[[], Any],
    runtime: Any,
    device: torch.device,
    batch_size: int,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        function()
        runtime.synchronize(device)
    elapsed = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        function()
        runtime.synchronize(device)
        elapsed.append((time.perf_counter_ns() - started) / 1e9)
    values = np.asarray(elapsed, dtype=np.float64)
    median = float(np.median(values))
    return {
        "batch_size": batch_size,
        "warmups": warmups,
        "repetitions": repetitions,
        "all_seconds": elapsed,
        "median_seconds": median,
        "p05_seconds": float(np.quantile(values, 0.05)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "throughput_rows_per_second_at_median": float(batch_size / median),
    }


def run() -> dict[str, Any]:
    common.assert_pre_generation_seal()
    output = common.STUDY_ROOT / "metrics/latency.json"
    if output.exists():
        prior = common.study_json(output)
        if not prior.get("passed"):
            raise RuntimeError("existing latency audit failed")
        return prior
    cfg = common.study_json(common.STUDY_ROOT / "config.json")["runtime_benchmark"]
    runtime = common.load_runtime()
    device = runtime.choose_device(cfg["device"])
    if device.type != "mps":
        raise RuntimeError("frozen latency benchmark requires MPS")
    with np.load(common.STUDY_ROOT / "data/plan_oracle_encoded.npz", allow_pickle=False) as stored:
        history_all = stored["history"]
        action_all = stored["action"]
    with np.load(common.STUDY_ROOT / "metrics/bootstrap_replicates.npz", allow_pickle=False) as stored:
        adaptive_calls = stored["adaptive_calls"]
        exact_calls = stored["exact_seeded_calls"]
    base_model, _, base_provenance = runtime.load_base_model(device)
    solver, v1, models = runtime.load_solver(device)
    gate = runtime.load_gate(device)
    before = runtime.module_audit(base_model, solver, v1, gate.model)
    seeds = common.study_json(common.STUDY_ROOT / "seed_manifest.json")["statistical_seeds"]
    order = np.random.default_rng(int(seeds["latency_order"])).permutation(len(history_all))
    rows = []
    equivalence = []
    for batch_size in cfg["batch_sizes"]:
        selected = order[: int(batch_size)]
        history = torch.as_tensor(np.ascontiguousarray(history_all[selected]), device=device)
        action = torch.as_tensor(np.ascontiguousarray(action_all[selected]), device=device)
        exact_tensor = torch.as_tensor(
            np.ascontiguousarray(exact_calls[selected]), dtype=torch.long, device=device
        )
        base = runtime.base_predict(base_model, history, action)
        dense_out, dense_calls, _, _ = runtime.dense_adaptive(
            solver, gate, models, history, action, base
        )
        reference_out, reference_calls = runtime.sparse_adaptive_reference(
            solver, gate, models, history, action, base
        )
        optimized_out, optimized_calls = runtime.sparse_adaptive_optimized(
            solver, gate, models, history, action, base
        )
        equivalence.append(
            {
                "batch_size": int(batch_size),
                "dense_reference_calls_exact": bool(torch.equal(dense_calls, reference_calls)),
                "dense_optimized_calls_exact": bool(torch.equal(dense_calls, optimized_calls)),
                "calls_match_frozen_plan_evaluation": bool(
                    np.array_equal(dense_calls.cpu().numpy(), adaptive_calls[selected])
                ),
                "dense_reference_max_abs": float((dense_out - reference_out).abs().max().cpu()),
                "dense_optimized_max_abs": float((dense_out - optimized_out).abs().max().cpu()),
                "dense_reference_equivalent": bool(
                    torch.allclose(dense_out, reference_out, rtol=2e-5, atol=2e-6)
                ),
                "dense_optimized_equivalent": bool(
                    torch.allclose(dense_out, optimized_out, rtol=2e-5, atol=2e-6)
                ),
            }
        )
        functions = [
            ("fixed_d1", lambda: _fixed(runtime, base_model, solver, history, action, 1)),
            ("fixed_d2", lambda: _fixed(runtime, base_model, solver, history, action, 2)),
            ("fixed_d4", lambda: _fixed(runtime, base_model, solver, history, action, 4)),
            (
                "exact_total_seeded_mixture",
                lambda: _mixture(runtime, base_model, solver, history, action, exact_tensor),
            ),
            (
                "adaptive_dense_all_exits",
                lambda: _dense(runtime, base_model, solver, gate, models, history, action),
            ),
            (
                "adaptive_reference_sparse",
                lambda: _reference(runtime, base_model, solver, gate, models, history, action),
            ),
            (
                "adaptive_optimized_sparse",
                lambda: _optimized(runtime, base_model, solver, gate, models, history, action),
            ),
        ]
        for name, function in functions:
            timing = _benchmark(
                function,
                runtime,
                device,
                int(batch_size),
                int(cfg["warmups"]),
                int(cfg["repetitions"]),
            )
            rows.append({"path": name, **timing})
            print(f"latency {name} batch={batch_size}", flush=True)
    after = runtime.module_audit(base_model, solver, v1, gate.model)
    passed = bool(
        before["passed"]
        and before == after
        and all(
            item["dense_reference_calls_exact"]
            and item["dense_optimized_calls_exact"]
            and item["calls_match_frozen_plan_evaluation"]
            and item["dense_reference_equivalent"]
            and item["dense_optimized_equivalent"]
            for item in equivalence
        )
    )
    result = {
        "schema_version": 1,
        "device": str(device),
        "measurement": cfg["measurement"],
        "includes": [
            "frozen base prediction",
            "solver blocks",
            "causal feature construction",
            "frozen normalization and linear head",
            "routing and indexing",
            "device synchronization",
        ],
        "timings": rows,
        "equivalence": equivalence,
        "module_before": before,
        "module_after": after,
        "base_provenance": base_provenance,
        "passed": passed,
        "actual_latency_separate_from_fully_counted_flops": True,
        "statistical_and_flop_verdict_independent_of_latency": True,
        "energy_measured": False,
        "energy_reason": "no reliable per-path energy measurement interface available",
    }
    if not passed:
        raise RuntimeError("latency validity audit failed")
    common.write_study_json(output, result, exclusive=True)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"passed": result["passed"]}, sort_keys=True))

