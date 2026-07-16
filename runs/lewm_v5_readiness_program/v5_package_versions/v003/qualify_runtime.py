#!/usr/bin/env python3
"""CPU/MPS determinism and sparse/dense qualification on consumed smoke only."""

from __future__ import annotations

import argparse
import json

import numpy as np

from cycle_common import (
    NUMERICAL_ATOL,
    NUMERICAL_MAX_ABS,
    NUMERICAL_RTOL,
    PRIOR_INVALID,
    REPO_ROOT,
    ROOT,
    assert_runtime_contract,
    atomic_json,
    read_json,
    sha256_file,
)
from input_loader import load_model_gate_inputs
from runner import (
    dense_shadow,
    distribution_modules,
    load_gate_tensors,
    load_scientific_stack,
    manual_sparse,
    prepare_episode_tensors,
    tensors_exact_with_nan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cpu,mps")
    arguments = parser.parse_args()
    runtime_contract = assert_runtime_contract("evaluation")
    prior_manifest = read_json(PRIOR_INVALID / "data/smoke_raw_manifest.json")
    record = prior_manifest["episodes"][0]
    raw_path = REPO_ROOT / record["path"]
    if sha256_file(raw_path) != record["sha256"]:
        raise RuntimeError("consumed smoke input drift")
    loaded, loader_audit = load_model_gate_inputs(raw_path)
    results = []

    requested = [name.strip() for name in arguments.devices.split(",") if name.strip()]
    for device_name in requested:
        try:
            torch, runtime, model_io, device, base, contract, stack = load_scientific_stack(device_name)
        except RuntimeError as error:
            if device_name == "mps" and "unavailable" in str(error).lower():
                continue
            raise
        solver, v1, models, provenance = stack
        common, _, _ = distribution_modules()
        gate = load_gate_tensors(torch, device)
        before = runtime.module_audit(base, solver, v1)
        history, actions, target, base_prediction = prepare_episode_tensors(
            torch, runtime, model_io, device, base, contract, common, loaded
        )
        dense_one = dense_shadow(torch, solver, gate, history, actions, base_prediction)
        dense_two = dense_shadow(torch, solver, gate, history, actions, base_prediction)
        sparse_one = manual_sparse(torch, solver, gate, history, actions, base_prediction)
        sparse_two = manual_sparse(torch, solver, gate, history, actions, base_prediction)
        dense, dense_calls, dense_scores, dense_features = dense_one
        sparse, calls, scores, features = sparse_one
        with torch.inference_mode():
            selected_api = solver.forward_selected(history, actions, base_prediction, calls)
        selected_dense = dense[torch.arange(len(history), device=device), calls - 1]
        delta = sparse - selected_dense
        max_abs = float(delta.abs().max().item())
        depth_one = calls == 1
        raw_sparse = torch.square(sparse - target).mean(1)
        raw_dense = torch.square(selected_dense - target).mean(1)
        checks = {
            "module_before_passed": bool(before["passed"]),
            "dense_repeat_exits_exact": bool(torch.equal(dense_one[0], dense_two[0])),
            "dense_repeat_calls_exact": bool(torch.equal(dense_one[1], dense_two[1])),
            "dense_repeat_scores_exact": bool(torch.equal(dense_one[2], dense_two[2])),
            "dense_repeat_features_exact": bool(torch.equal(dense_one[3], dense_two[3])),
            "sparse_repeat_output_exact": bool(torch.equal(sparse_one[0], sparse_two[0])),
            "sparse_repeat_calls_exact": bool(torch.equal(sparse_one[1], sparse_two[1])),
            "sparse_repeat_scores_exact": tensors_exact_with_nan(torch, sparse_one[2], sparse_two[2]),
            "sparse_repeat_features_exact": tensors_exact_with_nan(torch, sparse_one[3], sparse_two[3]),
            "manual_sparse_vs_forward_selected_exact": bool(torch.equal(sparse, selected_api)),
            "sparse_dense_calls_exact": bool(torch.equal(calls, dense_calls)),
            "sparse_dense_histogram_exact": bool(
                torch.equal(torch.bincount(calls, minlength=5), torch.bincount(dense_calls, minlength=5))
            ),
            "depth_one_selected_rows_exact": bool(
                not depth_one.any() or torch.equal(sparse[depth_one], selected_dense[depth_one])
            ),
            "cross_batch_allclose": bool(
                torch.allclose(sparse, selected_dense, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL)
            ),
            "cross_batch_max_abs_ceiling": max_abs <= NUMERICAL_MAX_ABS,
        }
        runtime.synchronize(device)
        after = runtime.module_audit(base, solver, v1)
        checks["module_after_identical"] = before == after
        result = {
            "device": device_name,
            "consumed_episode_id": record["episode_id"],
            "consumed_episode_sha256": record["sha256"],
            "checks": checks,
            "call_histogram": torch.bincount(calls, minlength=5)[1:].cpu().tolist(),
            "cross_batch_max_abs": max_abs,
            "cross_batch_mean_abs": float(delta.abs().mean().item()),
            "cross_batch_rmse": float(torch.sqrt(torch.square(delta).mean()).item()),
            "maximum_observed_per_row_raw_mse_impact": float(
                (raw_sparse - raw_dense).abs().max().item()
            ),
            "mean_observed_raw_mse_impact": float((raw_sparse - raw_dense).mean().item()),
            "base_provenance": provenance,
            "passed": all(checks.values()),
        }
        results.append(result)
        print(json.dumps({"device": device_name, "passed": result["passed"]}), flush=True)
        del base, solver, v1, history, actions, target, base_prediction
        if device_name == "mps":
            torch.mps.empty_cache()

    audit = {
        "schema_version": 1,
        "consumed_only": True,
        "input_loader": loader_audit,
        "requested_devices": requested,
        "tested_devices": [item["device"] for item in results],
        "devices": results,
        "fixed_contract": {
            "rtol": NUMERICAL_RTOL,
            "atol": NUMERICAL_ATOL,
            "maximum_absolute_error_ceiling": NUMERICAL_MAX_ABS,
        },
        "passed": bool(results) and all(item["passed"] for item in results),
        "prospective_episodes": 0,
        "v5_outcome_episodes": 0,
        "runtime_contract": runtime_contract,
    }
    if not audit["passed"]:
        raise RuntimeError("runtime qualification failed")
    atomic_json(ROOT / "audit/runtime_qualification.json", audit, exclusive=True)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
