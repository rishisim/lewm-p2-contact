#!/usr/bin/env python3
"""Post-decision end-to-end dense/sparse runtime and equivalence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import common
import judge_runner
import protocol
import students


def sync(device: torch.device) -> None:
    common.sync(device)


def load_everything(device: torch.device):
    tournament = judge_runner.validate_frozen_tournament()
    arrays = judge_runner.load_calibration_arrays()
    model_io = judge_runner.load_model_io()
    base_model, _, base_provenance = model_io.load_frozen_lewm(
        judge_runner.BASE_CONFIG,
        judge_runner.BASE_WEIGHTS,
        device,
        expected_config_sha256="4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999",
        expected_weights_sha256="2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89",
    )
    solver, _, _, model_module = common.load_frozen_solver(device)
    student = judge_runner.load_student(tournament, device)
    student_model = student.build(device)
    return tournament, arrays, base_model, base_provenance, solver, model_module, student, student_model


@torch.inference_mode()
def base_predict(base_model: torch.nn.Module, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return base_model.predict(h, base_model.action_encoder(a))[:, -1]


def scorer(student: students.FrozenStudent, student_model: torch.nn.Module, model_module: Any, device: torch.device):
    indices = torch.as_tensor(student.feature_indices, dtype=torch.long, device=device)
    mean = torch.as_tensor(student.feature_mean, device=device)
    std = torch.as_tensor(student.feature_std, device=device)
    def score(h: torch.Tensor, a: torch.Tensor, current: torch.Tensor, update: torch.Tensor, stage: int) -> torch.Tensor:
        causal = model_module.build_causal_features(h, a, current, update).index_select(1, indices)
        depth = torch.zeros((len(h), 3), dtype=causal.dtype, device=device)
        depth[:, int(stage)] = 1.0
        encoded = torch.cat((causal, depth), 1)
        normalized = (encoded - mean) / std
        return student_model(normalized) * student.target_scale + student.target_center
    return score


@torch.inference_mode()
def fixed_inference(base_model, solver, h, a, depth: int):
    base = base_predict(base_model, h, a)
    return solver(h, a, base, max_depth=int(depth))[int(depth)]


@torch.inference_mode()
def selected_inference(base_model, solver, h, a, calls: torch.Tensor):
    base = base_predict(base_model, h, a)
    return solver.forward_selected(h, a, base, calls)


@torch.inference_mode()
def dense_adaptive(base_model, solver, score, h, a, price: float):
    base = base_predict(base_model, h, a)
    output, updates = solver(h, a, base, max_depth=4, return_updates=True)
    scores = torch.stack([score(h, a, output[d], updates[d], d - 1) for d in (1, 2, 3)], 1)
    calls = torch.ones(len(h), dtype=torch.long, device=h.device)
    active = torch.ones(len(h), dtype=torch.bool, device=h.device)
    for stage in range(3):
        active = active & (scores[:, stage] > float(price))
        calls += active.long()
    stacked = torch.stack([output[d] for d in (1, 2, 3, 4)], 1)
    selected = stacked[torch.arange(len(h), device=h.device), calls - 1]
    return selected, calls


@torch.inference_mode()
def sparse_adaptive(base_model, solver, score, h, a, price: float):
    base = base_predict(base_model, h, a)
    anchored = solver.anchor(h, a, base)
    current = anchored[1]
    update_all = current - base
    calls = torch.ones(len(h), dtype=torch.long, device=h.device)
    active = torch.arange(len(h), device=h.device)
    for stage, adapter in enumerate(solver.adapters):
        if not active.numel():
            break
        local_score = score(
            h.index_select(0, active),
            a.index_select(0, active),
            current.index_select(0, active),
            update_all.index_select(0, active),
            stage,
        )
        continuing = local_score > float(price)
        active = active[continuing]
        if not active.numel():
            break
        preceding = current.index_select(0, active)
        update = adapter(h.index_select(0, active), a.index_select(0, active), preceding)
        current = current.index_copy(0, active, preceding + update)
        update_all = update_all.index_copy(0, active, update)
        calls[active] += 1
    return current, calls


def benchmark(fn: Callable[[], Any], device: torch.device, warmups: int = 2, repetitions: int = 7) -> dict[str, Any]:
    for _ in range(warmups):
        fn(); sync(device)
    elapsed = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn(); sync(device)
        elapsed.append(time.perf_counter() - start)
    return {
        "repetitions": repetitions,
        "median_seconds": float(statistics.median(elapsed)),
        "minimum_seconds": float(min(elapsed)),
        "maximum_seconds": float(max(elapsed)),
        "all_seconds": elapsed,
    }


def run(device: torch.device) -> dict[str, Any]:
    tournament, arrays, base_model, base_provenance, solver, model_module, student, student_model = load_everything(device)
    score = scorer(student, student_model, model_module, device)
    h_all = np.ascontiguousarray(arrays["history"])
    a_all = np.ascontiguousarray(arrays["action"])
    cached_base = np.ascontiguousarray(arrays["base_pred"])
    # Frozen calibration calls are recomputed from the frozen score and price,
    # without target access or threshold adjustment.
    _, prepared_features, _ = judge_runner.solver_outputs(arrays, device)
    frozen_scores = students.predict_student(student, prepared_features, device)
    adaptive_calls = protocol.sequential_calls(frozen_scores, float(tournament["compute_price"]))
    policy = protocol.prior_policy()
    mixture = policy.strongest_transition_independent_baseline(
        np.asarray(tournament["discovery_fixed_exit_raw_mse"]), protocol.SUPPORTED_CALLS,
        n=len(adaptive_calls), target_total_calls=int(adaptive_calls.sum()),
        seed=int(tournament["baseline_seeds"]["matched"]),
    )["selected_calls"]
    rows = []
    equivalence = []
    base_checks = []
    for batch_size in (256, 1024):
        # Time the same full calibration set for each path and batch size.
        indices = [np.arange(start, min(start + batch_size, len(h_all))) for start in range(0, len(h_all), batch_size)]
        def batches_call(callback):
            outputs = []
            calls = []
            for part in indices:
                h = torch.as_tensor(h_all[part], device=device)
                a = torch.as_tensor(a_all[part], device=device)
                value = callback(part, h, a)
                if isinstance(value, tuple):
                    outputs.append(value[0].cpu().numpy()); calls.append(value[1].cpu().numpy())
                else:
                    outputs.append(value.cpu().numpy())
            return (np.concatenate(outputs), np.concatenate(calls)) if calls else np.concatenate(outputs)
        # Verify common base prediction is genuinely executed and consistent.
        recomputed = batches_call(lambda part, h, a: base_predict(base_model, h, a))
        base_checks.append({
            "batch_size": batch_size,
            "max_abs_difference_vs_cached_base": float(np.max(np.abs(recomputed - cached_base))),
            "close": bool(np.allclose(recomputed, cached_base, rtol=2e-5, atol=2e-6)),
        })
        for depth in (1, 2, 3, 4):
            fn = lambda depth=depth: batches_call(lambda part, h, a: fixed_inference(base_model, solver, h, a, depth))
            timing = benchmark(fn, device)
            rows.append({"path": f"fixed_d{depth}", "batch_size": batch_size, "rows": len(h_all), **timing})
        mixture_fn = lambda: batches_call(
            lambda part, h, a: selected_inference(
                base_model, solver, h, a,
                torch.as_tensor(mixture[part], dtype=torch.long, device=device),
            )
        )
        rows.append({"path": "matched_transition_independent_mixture", "batch_size": batch_size, "rows": len(h_all), **benchmark(mixture_fn, device)})
        dense_fn = lambda: batches_call(lambda part, h, a: dense_adaptive(base_model, solver, score, h, a, float(tournament["compute_price"])))
        sparse_fn = lambda: batches_call(lambda part, h, a: sparse_adaptive(base_model, solver, score, h, a, float(tournament["compute_price"])))
        dense_output, dense_calls = dense_fn()
        sparse_output, sparse_calls = sparse_fn()
        equivalence.append({
            "batch_size": batch_size,
            "calls_exact": bool(np.array_equal(dense_calls, sparse_calls)),
            "calls_match_frozen_policy": bool(np.array_equal(sparse_calls, adaptive_calls)),
            "max_abs_output_difference": float(np.max(np.abs(dense_output - sparse_output))),
            "outputs_close": bool(np.allclose(dense_output, sparse_output, rtol=2e-5, atol=2e-6)),
            "processed_solver_rows": int(sparse_calls.sum()),
            "expected_solver_rows": int(adaptive_calls.sum()),
        })
        rows.append({"path": "adaptive_dense_all_exits", "batch_size": batch_size, "rows": len(h_all), **benchmark(dense_fn, device)})
        rows.append({"path": "adaptive_realistic_sparse", "batch_size": batch_size, "rows": len(h_all), **benchmark(sparse_fn, device)})
    passed = bool(
        all(item["close"] for item in base_checks)
        and all(item["calls_exact"] and item["calls_match_frozen_policy"] and item["outputs_close"] and item["processed_solver_rows"] == item["expected_solver_rows"] for item in equivalence)
    )
    result = {
        "schema_version": 1,
        "device": str(device),
        "rows": int(len(h_all)),
        "batch_sizes": [256, 1024],
        "includes": [
            "common frozen base prediction", "feature construction", "student normalization/scoring",
            "dynamic routing", "solver adapters", "synchronization", "control flow",
        ],
        "base_model_provenance": base_provenance,
        "base_prediction_checks": base_checks,
        "sparse_dense_equivalence": equivalence,
        "timings": rows,
        "gate_cost": tournament["gate_cost"],
        "passed": passed,
        "interpretation": "Analytic FLOPs, latency, and energy are different notions; energy was not measured.",
    }
    if not passed:
        raise RuntimeError(f"runtime/equivalence audit failed: {result}")
    common.write_json(ROOT / "metrics/runtime.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run(common.choose_device(args.device))
    print(json.dumps({"passed": result["passed"], "rows": result["rows"]}, sort_keys=True))
