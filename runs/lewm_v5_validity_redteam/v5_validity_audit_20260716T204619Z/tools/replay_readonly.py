#!/usr/bin/env python3
"""Read-only full raw replay with packaged and independent sparse strategies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
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


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def local_features(
    torch: Any,
    history: Any,
    action_history: Any,
    current: Any,
    update: Any,
) -> Any:
    """Independent transcription of the frozen row-local 1,046 features."""
    history = history.detach()
    action_history = action_history.detach()
    current = current.detach()
    update = update.detach()
    epsilon = torch.finfo(current.dtype).eps
    previous = history[:, 2]
    gap = current - previous
    current_norm = torch.sqrt(torch.sum(current * current, dim=1, keepdim=True))
    update_norm = torch.sqrt(torch.sum(update * update, dim=1, keepdim=True))
    previous_norm = torch.sqrt(torch.sum(previous * previous, dim=1, keepdim=True))
    gap_norm = torch.sqrt(torch.sum(gap * gap, dim=1, keepdim=True))
    relative = update_norm / torch.clamp(current_norm, min=epsilon)
    update_current = torch.sum(update * current, dim=1, keepdim=True) / torch.clamp(
        update_norm * current_norm, min=epsilon
    )
    update_gap = torch.sum(update * gap, dim=1, keepdim=True) / torch.clamp(
        update_norm * gap_norm, min=epsilon
    )
    history_change = torch.sqrt(
        torch.sum(
            (history[:, 1:] - history[:, :-1])
            * (history[:, 1:] - history[:, :-1]),
            dim=2,
        )
    )
    action_change = torch.sqrt(
        torch.sum(
            (action_history[:, 1:] - action_history[:, :-1])
            * (action_history[:, 1:] - action_history[:, :-1]),
            dim=2,
        )
    )
    result = torch.cat(
        [
            history.reshape(len(history), -1),
            action_history.reshape(len(history), -1),
            current,
            update,
            current_norm,
            update_norm,
            relative,
            previous_norm,
            gap_norm,
            update_current,
            update_gap,
            history_change,
            action_change,
        ],
        dim=1,
    )
    if result.shape != (len(history), 1046):
        raise RuntimeError(f"independent feature shape mismatch: {result.shape}")
    return result


def local_features_order_matched(
    torch: Any,
    history: Any,
    action_history: Any,
    current: Any,
    update: Any,
) -> Any:
    """Second transcription preserving the runtime vector-norm primitive order."""
    history = history.detach()
    action_history = action_history.detach()
    current = current.detach()
    update = update.detach()
    epsilon = torch.finfo(current.dtype).eps
    previous = history[:, -1]
    gap = current - previous
    current_norm = torch.linalg.vector_norm(current, dim=-1, keepdim=True)
    update_norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
    previous_norm = torch.linalg.vector_norm(previous, dim=-1, keepdim=True)
    gap_norm = torch.linalg.vector_norm(gap, dim=-1, keepdim=True)
    relative = update_norm / current_norm.clamp_min(epsilon)
    update_current = (update * current).sum(dim=-1, keepdim=True) / (
        update_norm * current_norm
    ).clamp_min(epsilon)
    update_gap = (update * gap).sum(dim=-1, keepdim=True) / (
        update_norm * gap_norm
    ).clamp_min(epsilon)
    history_change = torch.linalg.vector_norm(
        history[:, 1:] - history[:, :-1], dim=2
    )
    action_change = torch.linalg.vector_norm(
        action_history[:, 1:] - action_history[:, :-1], dim=2
    )
    result = torch.cat(
        (
            history.flatten(1),
            action_history.flatten(1),
            current,
            update,
            current_norm,
            update_norm,
            relative,
            previous_norm,
            gap_norm,
            update_current,
            update_gap,
            history_change,
            action_change,
        ),
        dim=1,
    )
    if result.shape != (len(history), 1046):
        raise RuntimeError("order-matched independent feature shape mismatch")
    return result


def local_sparse(
    torch: Any,
    solver: Any,
    gate: dict[str, Any],
    history: Any,
    actions: Any,
    base_prediction: Any,
    feature_builder: Any = local_features,
) -> tuple[Any, Any, Any, Any]:
    with torch.inference_mode():
        anchored = solver.anchor(history, actions, base_prediction)
        current = anchored[1]
        update = current - base_prediction
        calls = torch.ones(
            len(history), dtype=torch.long, device=history.device
        )
        scores = torch.full(
            (len(history), 3), float("nan"), dtype=current.dtype, device=history.device
        )
        features = torch.full(
            (len(history), 3, 1046),
            float("nan"),
            dtype=current.dtype,
            device=history.device,
        )
        active = torch.arange(len(history), device=history.device)
        for stage in range(3):
            if active.numel() == 0:
                break
            local = feature_builder(
                torch,
                history.index_select(0, active),
                actions.index_select(0, active),
                current.index_select(0, active),
                update.index_select(0, active),
            )
            raw = local @ gate["a_raw"][stage] + gate["b_raw"][stage]
            white = local @ gate["a_white"][stage] + gate["b_white"][stage]
            score = torch.minimum(raw, white)
            scores[active, stage] = score
            features[active, stage] = local
            active = active[score > gate["thresholds"][stage]]
            if active.numel() == 0:
                break
            preceding = current.index_select(0, active)
            next_update = solver.adapters[stage](
                history.index_select(0, active),
                actions.index_select(0, active),
                preceding,
            )
            current = current.index_copy(0, active, preceding + next_update)
            update = update.index_copy(0, active, next_update)
            calls[active] += 1
    return current, calls, scores, features


def exact_nan(torch: Any, left: Any, right: Any) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_nan = torch.isnan(left)
    right_nan = torch.isnan(right)
    return bool(
        torch.equal(left_nan, right_nan)
        and torch.equal(left[~left_nan], right[~right_nan])
    )


def synthetic_feature_validation(torch: Any, device: Any, packaged_builder: Any) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(730001)
    history = torch.randn(7, 3, 192, generator=generator, dtype=torch.float32).to(device)
    actions = torch.randn(7, 3, 25, generator=generator, dtype=torch.float32).to(device)
    current = torch.randn(7, 192, generator=generator, dtype=torch.float32).to(device)
    update = torch.randn(7, 192, generator=generator, dtype=torch.float32).to(device)
    packaged = packaged_builder(history, actions, current, update)
    manual = local_features(torch, history, actions, current, update)
    ordered = local_features_order_matched(torch, history, actions, current, update)
    manual_delta = float((manual - packaged).abs().max().item())
    ordered_delta = float((ordered - packaged).abs().max().item())
    checks = {
        "finite": bool(
            torch.isfinite(packaged).all()
            and torch.isfinite(manual).all()
            and torch.isfinite(ordered).all()
        ),
        "shape_exact": packaged.shape == manual.shape == ordered.shape == (7, 1046),
        "order_matched_second_method_exact": bool(torch.equal(ordered, packaged)),
        "manual_algebra_within_predeclared_tolerance": bool(
            torch.allclose(manual, packaged, rtol=2e-5, atol=2e-6)
        ),
    }
    return {
        "seed": 730001,
        "device": str(device),
        "manual_max_abs": manual_delta,
        "order_matched_max_abs": ordered_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    package = args.package.resolve()
    sys.path.insert(0, str(package))
    import runner  # type: ignore

    manifest = json.loads(
        (package / "data/v5_confirmation_raw_manifest.json").read_text()
    )
    records = manifest["episodes"]
    if args.limit is not None:
        records = records[: args.limit]
    count = len(records)

    execution_path = package / "data/v5_confirmation_execution.npz"
    with np.load(execution_path, allow_pickle=False) as stored:
        expected = {name: stored[name].copy() for name in stored.files}
    torch, runtime, model_io, device, base, contract, stack = (
        runner.load_scientific_stack(args.device)
    )
    solver, v1, _, _ = stack
    common, _, _ = runner.distribution_modules()
    gate = runner.load_gate_tensors(torch, device)
    from counted_features import build_counted_causal_features as packaged_feature_builder  # type: ignore

    synthetic = synthetic_feature_validation(
        torch, device, packaged_feature_builder
    )
    module_before = runtime.module_audit(base, solver, v1)
    semantic_names = json.loads(
        (package / "semantic_feature_names.json").read_text()
    )["ordered_names"]

    metrics = {
        "episodes": 0,
        "rows": 0,
        "raw_hashes_verified": 0,
        "target_exact_rows": 0,
        "dense_exact_rows": 0,
        "packaged_sparse_exact_rows": 0,
        "independent_sparse_exact_rows": 0,
        "packaged_calls_exact_rows": 0,
        "independent_calls_exact_rows": 0,
        "packaged_scores_exact_episodes": 0,
        "independent_scores_exact_episodes": 0,
        "independent_scores_allclose_episodes": 0,
        "packaged_features_exact_episodes": 0,
        "independent_features_exact_episodes": 0,
        "independent_features_allclose_episodes": 0,
        "strategy_outputs_exact_episodes": 0,
        "strategy_calls_exact_episodes": 0,
        "strategy_scores_exact_episodes": 0,
        "strategy_scores_allclose_episodes": 0,
        "strategy_features_exact_episodes": 0,
        "strategy_features_allclose_episodes": 0,
        "packaged_sparse_max_abs": 0.0,
        "independent_sparse_max_abs": 0.0,
        "independent_feature_max_abs": 0.0,
        "independent_score_max_abs": 0.0,
        "order_matched_sparse_exact_rows": 0,
        "order_matched_calls_exact_rows": 0,
        "order_matched_scores_exact_episodes": 0,
        "order_matched_features_exact_episodes": 0,
        "manual_scores_sealed_allclose_episodes": 0,
        "manual_features_sealed_allclose_episodes": 0,
        "worst_independent_feature": None,
        "worst_independent_score": None,
    }
    failures = []
    started = time.time()
    for episode_index, record in enumerate(records):
        raw_path = runner.REPO_ROOT / record["path"]
        if sha256(raw_path) != record["sha256"]:
            failures.append(
                {"episode": episode_index, "kind": "raw_hash_mismatch"}
            )
            break
        metrics["raw_hashes_verified"] += 1
        loaded, loader_audit = runner.load_model_gate_inputs(raw_path)
        if loader_audit["loaded_keys"] != ["action", "pixels"]:
            failures.append(
                {"episode": episode_index, "kind": "input_allowlist"}
            )
            break
        history, actions, target, base_prediction = runner.prepare_episode_tensors(
            torch, runtime, model_io, device, base, contract, common, loaded
        )
        dense, dense_calls, _, _ = runner.dense_shadow(
            torch, solver, gate, history, actions, base_prediction
        )
        packaged = runner.manual_sparse(
            torch, solver, gate, history, actions, base_prediction
        )
        independent = local_sparse(
            torch, solver, gate, history, actions, base_prediction
        )
        order_matched = local_sparse(
            torch,
            solver,
            gate,
            history,
            actions,
            base_prediction,
            feature_builder=local_features_order_matched,
        )
        packaged_output, packaged_calls, packaged_scores, packaged_features = packaged
        local_output, local_calls, local_scores, local_features_value = independent
        (
            ordered_output,
            ordered_calls,
            ordered_scores,
            ordered_features,
        ) = order_matched
        start = episode_index * 38
        stop = start + 38

        expected_target = torch.as_tensor(expected["target"][start:stop], device=device)
        expected_dense = torch.as_tensor(
            expected["dense_exits"][start:stop], device=device
        )
        expected_sparse = torch.as_tensor(
            expected["sparse_selected"][start:stop], device=device
        )
        expected_calls = torch.as_tensor(
            expected["calls"][start:stop].astype(np.int64), device=device
        )
        expected_scores = torch.as_tensor(
            expected["scores"][start:stop], device=device
        )
        expected_features = torch.as_tensor(
            expected["features"][start:stop], device=device
        )

        target_exact = torch.all(target == expected_target, dim=1)
        dense_exact = torch.all(dense == expected_dense, dim=(1, 2))
        packaged_exact = torch.all(packaged_output == expected_sparse, dim=1)
        local_exact = torch.all(local_output == expected_sparse, dim=1)
        metrics["target_exact_rows"] += int(target_exact.sum().item())
        metrics["dense_exact_rows"] += int(dense_exact.sum().item())
        metrics["packaged_sparse_exact_rows"] += int(packaged_exact.sum().item())
        metrics["independent_sparse_exact_rows"] += int(local_exact.sum().item())
        metrics["order_matched_sparse_exact_rows"] += int(
            torch.all(ordered_output == expected_sparse, dim=1).sum().item()
        )
        metrics["packaged_calls_exact_rows"] += int(
            (packaged_calls == expected_calls).sum().item()
        )
        metrics["independent_calls_exact_rows"] += int(
            (local_calls == expected_calls).sum().item()
        )
        metrics["order_matched_calls_exact_rows"] += int(
            (ordered_calls == expected_calls).sum().item()
        )
        metrics["packaged_scores_exact_episodes"] += int(
            exact_nan(torch, packaged_scores, expected_scores)
        )
        metrics["independent_scores_exact_episodes"] += int(
            exact_nan(torch, local_scores, expected_scores)
        )
        metrics["order_matched_scores_exact_episodes"] += int(
            exact_nan(torch, ordered_scores, expected_scores)
        )
        metrics["packaged_features_exact_episodes"] += int(
            exact_nan(torch, packaged_features, expected_features)
        )
        metrics["independent_features_exact_episodes"] += int(
            exact_nan(torch, local_features_value, expected_features)
        )
        metrics["order_matched_features_exact_episodes"] += int(
            exact_nan(torch, ordered_features, expected_features)
        )
        reached_features = torch.isfinite(expected_features)
        metrics["independent_features_allclose_episodes"] += int(
            torch.allclose(
                local_features_value[reached_features],
                expected_features[reached_features],
                rtol=2e-5,
                atol=2e-6,
            )
        )
        metrics["manual_features_sealed_allclose_episodes"] += int(
            torch.allclose(
                local_features_value[reached_features],
                expected_features[reached_features],
                rtol=2e-6,
                atol=2e-7,
            )
        )
        metrics["strategy_outputs_exact_episodes"] += int(
            torch.equal(packaged_output, local_output)
        )
        metrics["strategy_calls_exact_episodes"] += int(
            torch.equal(packaged_calls, local_calls)
        )
        metrics["strategy_scores_exact_episodes"] += int(
            exact_nan(torch, packaged_scores, local_scores)
        )
        reached_scores = torch.isfinite(expected_scores)
        metrics["independent_scores_allclose_episodes"] += int(
            torch.allclose(
                local_scores[reached_scores],
                expected_scores[reached_scores],
                rtol=2e-5,
                atol=2e-6,
            )
        )
        metrics["manual_scores_sealed_allclose_episodes"] += int(
            torch.allclose(
                local_scores[reached_scores],
                expected_scores[reached_scores],
                rtol=2e-6,
                atol=2e-7,
            )
        )
        metrics["strategy_scores_allclose_episodes"] += int(
            torch.allclose(
                packaged_scores[reached_scores],
                local_scores[reached_scores],
                rtol=2e-5,
                atol=2e-6,
            )
        )
        metrics["strategy_features_exact_episodes"] += int(
            exact_nan(torch, packaged_features, local_features_value)
        )
        metrics["strategy_features_allclose_episodes"] += int(
            torch.allclose(
                packaged_features[reached_features],
                local_features_value[reached_features],
                rtol=2e-5,
                atol=2e-6,
            )
        )
        metrics["packaged_sparse_max_abs"] = max(
            metrics["packaged_sparse_max_abs"],
            float((packaged_output - expected_sparse).abs().max().item()),
        )
        metrics["independent_sparse_max_abs"] = max(
            metrics["independent_sparse_max_abs"],
            float((local_output - expected_sparse).abs().max().item()),
        )
        reached = reached_features
        feature_delta = (local_features_value - expected_features).abs()
        feature_delta[~reached] = -1
        feature_max = float(feature_delta.max().item())
        if feature_max > metrics["independent_feature_max_abs"]:
            flat = int(feature_delta.argmax().item())
            local_row, stage, feature = np.unravel_index(
                flat, tuple(feature_delta.shape)
            )
            expected_value = float(
                expected_features[local_row, stage, feature].item()
            )
            observed_value = float(
                local_features_value[local_row, stage, feature].item()
            )
            metrics["independent_feature_max_abs"] = feature_max
            metrics["worst_independent_feature"] = {
                "episode_index": episode_index,
                "global_row": start + int(local_row),
                "local_row": int(local_row),
                "model_step": int(expected["model_step"][start + local_row]),
                "stage": int(stage + 1),
                "feature_index": int(feature),
                "feature_name": semantic_names[feature],
                "expected": expected_value,
                "observed": observed_value,
                "absolute_delta": feature_max,
                "sealed_allclose_allowance": 2e-7
                + 2e-6 * abs(expected_value),
                "audit_predeclared_allowance": 2e-6
                + 2e-5 * abs(expected_value),
            }
        score_delta = (local_scores - expected_scores).abs()
        score_delta[~reached_scores] = -1
        score_max = float(score_delta.max().item())
        if score_max > metrics["independent_score_max_abs"]:
            flat = int(score_delta.argmax().item())
            local_row, stage = np.unravel_index(flat, tuple(score_delta.shape))
            expected_value = float(expected_scores[local_row, stage].item())
            observed_value = float(local_scores[local_row, stage].item())
            metrics["independent_score_max_abs"] = score_max
            metrics["worst_independent_score"] = {
                "episode_index": episode_index,
                "global_row": start + int(local_row),
                "local_row": int(local_row),
                "model_step": int(expected["model_step"][start + local_row]),
                "stage": int(stage + 1),
                "expected": expected_value,
                "observed": observed_value,
                "absolute_delta": score_max,
                "sealed_allclose_allowance": 2e-7
                + 2e-6 * abs(expected_value),
                "audit_predeclared_allowance": 2e-6
                + 2e-5 * abs(expected_value),
                "threshold": float(gate["thresholds"][stage].item()),
                "expected_threshold_margin": abs(
                    expected_value - float(gate["thresholds"][stage].item())
                ),
            }
        metrics["episodes"] += 1
        metrics["rows"] += 38
        if (episode_index + 1) % 25 == 0 or episode_index + 1 == count:
            print(
                json.dumps(
                    {
                        "replayed_episodes": episode_index + 1,
                        "total": count,
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )

    runtime.synchronize(device)
    module_after = runtime.module_audit(base, solver, v1)
    expected_rows = count * 38
    checks = {
        "all_requested_episodes_replayed": metrics["episodes"] == count,
        "all_raw_hashes_verified": metrics["raw_hashes_verified"] == count,
        "all_rows_replayed": metrics["rows"] == expected_rows,
        "targets_bitwise_exact": metrics["target_exact_rows"] == expected_rows,
        "dense_exits_bitwise_exact": metrics["dense_exact_rows"] == expected_rows,
        "packaged_sparse_bitwise_exact": metrics["packaged_sparse_exact_rows"]
        == expected_rows,
        "independent_sparse_bitwise_exact": metrics[
            "independent_sparse_exact_rows"
        ]
        == expected_rows,
        "order_matched_sparse_bitwise_exact": metrics[
            "order_matched_sparse_exact_rows"
        ]
        == expected_rows,
        "packaged_calls_exact": metrics["packaged_calls_exact_rows"]
        == expected_rows,
        "independent_calls_exact": metrics["independent_calls_exact_rows"]
        == expected_rows,
        "order_matched_calls_exact": metrics[
            "order_matched_calls_exact_rows"
        ]
        == expected_rows,
        "packaged_scores_exact": metrics["packaged_scores_exact_episodes"]
        == count,
        "independent_scores_within_predeclared_float32_tolerance": metrics[
            "independent_scores_allclose_episodes"
        ]
        == count,
        "order_matched_scores_exact": metrics[
            "order_matched_scores_exact_episodes"
        ]
        == count,
        "packaged_features_exact": metrics["packaged_features_exact_episodes"]
        == count,
        "independent_features_within_predeclared_float32_tolerance": metrics[
            "independent_features_allclose_episodes"
        ]
        == count,
        "order_matched_features_exact": metrics[
            "order_matched_features_exact_episodes"
        ]
        == count,
        "two_strategy_outputs_exact": metrics["strategy_outputs_exact_episodes"]
        == count,
        "two_strategy_calls_exact": metrics["strategy_calls_exact_episodes"]
        == count,
        "two_strategy_scores_within_predeclared_float32_tolerance": metrics[
            "strategy_scores_allclose_episodes"
        ]
        == count,
        "two_strategy_features_within_predeclared_float32_tolerance": metrics[
            "strategy_features_allclose_episodes"
        ]
        == count,
        "modules_and_gradients_unchanged": module_before == module_after
        and module_after["passed"],
        "synthetic_independent_feature_harness_passed": synthetic["passed"],
        "no_failures": not failures,
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "device": str(device),
        "requested_episode_count": count,
        "requested_row_count": expected_rows,
        "complete_confirmation_replay": count == 1600,
        "strategies": [
            "packaged manual_sparse/dense_shadow path",
            "independently transcribed direct-module sparse loop with explicit sqrt/sum feature algebra",
            "second independent direct-module sparse loop preserving the vector_norm primitive order",
        ],
        "synthetic_feature_harness": synthetic,
        "metrics": metrics,
        "checks": checks,
        "failures": failures,
        "module_before": module_before,
        "module_after": module_after,
        "runtime_seconds": time.time() - started,
        "initial_failure_diagnosis": {
            "initial_artifact": "REPLAY_RESULTS_INITIAL_FAILURE.json",
            "initial_failure_was_audit_harness_acceptance_logic": True,
            "cause": (
                "The first harness compared a different but algebraically "
                "equivalent explicit sqrt/sum reduction against an absolute "
                "2e-6 cap and omitted the declared relative term. The resulting "
                "float32 reduction-order deltas did not alter a single call or "
                "output. A second order-matched transcription is required to be "
                "exact; the explicit algebra is evaluated under the predeclared "
                "atol/rtol without changing the sealed V5 contract."
            ),
            "sealed_cross_batch_tolerance_not_relaxed": True,
            "manual_explicit_algebra_sealed_allclose_episode_counts": {
                "features": metrics[
                    "manual_features_sealed_allclose_episodes"
                ],
                "scores": metrics["manual_scores_sealed_allclose_episodes"],
                "total": count,
            },
        },
        "passed": all(checks.values()),
    }
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "complete": result["complete_confirmation_replay"],
                "episodes": metrics["episodes"],
                "rows": metrics["rows"],
                "runtime_seconds": result["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
