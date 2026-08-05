#!/usr/bin/env python3
"""Evaluate frozen V5 conditions on the fixed common-candidate Cube pilot."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr

import bridge
import phase_c_generate


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CONDITIONS = ("adaptive", "matched", "fixed_d1", "fixed_d4")
START_COUNT = phase_c_generate.START_COUNT
CANDIDATE_COUNT = phase_c_generate.CANDIDATE_COUNT
HORIZONS = 5
TOP_K = 5


def encode_pixels(stack: Mapping[str, Any], pixels: np.ndarray) -> np.ndarray:
    torch = stack["torch"]
    parts = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 64):
            transformed = stack["model_io"].pixel_transform(
                pixels[start : start + 64], stack["contract"].image_size, stack["device"]
            ).unsqueeze(0)
            parts.append(stack["base"].encode({"pixels": transformed})["emb"].squeeze(0).cpu())
    return torch.cat(parts).numpy().astype(np.float32)


def matched_permutation(start_index: int, horizon: int) -> np.ndarray:
    seed = phase_c_generate.MATCHED_SEED + start_index * HORIZONS + horizon
    return np.random.default_rng(seed).permutation(CANDIDATE_COUNT).astype(np.int8)


def rank_metrics(cost: np.ndarray, real_error: np.ndarray) -> dict[str, Any]:
    if np.unique(cost).size < 2 or np.unique(real_error).size < 2:
        rho = None
    else:
        value = spearmanr(cost, real_error).statistic
        rho = float(value) if np.isfinite(value) else None
    selected = int(np.argmin(cost))
    real_order = np.argsort(real_error, kind="stable")
    predicted_top = np.argsort(cost, kind="stable")[:TOP_K]
    real_top = real_order[:TOP_K]
    return {
        "spearman": rho,
        "selected_index": selected,
        "selected_real_error": float(real_error[selected]),
        "real_regret": float(real_error[selected] - real_error[real_order[0]]),
        "top5_overlap": float(len(set(predicted_top.tolist()) & set(real_top.tolist())) / TOP_K),
        "selected_in_real_top5": bool(selected in set(real_top.tolist())),
        "oracle_best_index": int(real_order[0]),
        "oracle_best_real_error": float(real_error[real_order[0]]),
    }


def summarize(per_start: list[dict[str, Any]]) -> dict[str, Any]:
    rhos = [item["spearman"] for item in per_start if item["spearman"] is not None]
    return {
        "defined_spearman_starts": len(rhos),
        "mean_spearman": float(np.mean(rhos)) if rhos else None,
        "mean_real_regret": float(np.mean([item["real_regret"] for item in per_start])),
        "median_real_regret": float(np.median([item["real_regret"] for item in per_start])),
        "mean_selected_real_error": float(
            np.mean([item["selected_real_error"] for item in per_start])
        ),
        "mean_top5_overlap": float(np.mean([item["top5_overlap"] for item in per_start])),
        "selected_in_real_top5_frequency": float(
            np.mean([item["selected_in_real_top5"] for item in per_start])
        ),
    }


def run(device_name: str = "auto") -> dict[str, Any]:
    config = bridge.read_json(ROOT / "PHASE_C_PILOT_CONFIG.json")
    simulator_manifest = bridge.read_json(ROOT / "PHASE_C_SIMULATOR.json")
    simulator_path = ROOT / "phase_c_simulator.npz"
    if bridge.sha256_file(simulator_path) != simulator_manifest["sha256"]:
        raise RuntimeError("Phase-C simulator artifact hash drift")
    if (ROOT / "phase_c_metrics.npz").exists():
        raise RuntimeError("Phase-C model metrics already exist")
    with np.load(simulator_path, allow_pickle=False) as stored:
        simulation = {name: stored[name].copy() for name in stored.files}
    stack = bridge.load_stack(device_name)
    torch = stack["torch"]
    audit_before = stack["runtime"].module_audit(stack["base"], stack["solver"], stack["v1"])
    if not audit_before["passed"]:
        raise RuntimeError("frozen module audit failed before Phase C")
    initial_latents = encode_pixels(
        stack, simulation["initial_pixels"].reshape(-1, 224, 224, 3)
    ).reshape(START_COUNT, 3, 192)
    goal_latents = encode_pixels(stack, simulation["goal_pixels"])
    mean = stack["distribution"].FROZEN_ACTION_MEAN
    std = stack["distribution"].FROZEN_ACTION_STD
    past_blocks = ((simulation["past_actions"] - mean) / std).reshape(
        START_COUNT, 2, 25
    ).astype(np.float32)
    candidate_blocks = ((simulation["candidate_actions"] - mean) / std).reshape(
        START_COUNT, CANDIDATE_COUNT, 5, 25
    ).astype(np.float32)
    costs = np.empty((len(CONDITIONS), START_COUNT, CANDIDATE_COUNT), dtype=np.float64)
    adaptive_calls = np.empty((START_COUNT, HORIZONS, CANDIDATE_COUNT), dtype=np.int8)
    matched_calls = np.empty_like(adaptive_calls)
    permutations = np.empty_like(adaptive_calls)
    latency_seconds = np.zeros((len(CONDITIONS), START_COUNT), dtype=np.float64)
    condition_index = {name: index for index, name in enumerate(CONDITIONS)}
    exact_compute = True
    finite = True
    for start_index in range(START_COUNT):
        initial = torch.as_tensor(
            np.repeat(initial_latents[start_index][None], CANDIDATE_COUNT, axis=0),
            device=stack["device"],
        )
        histories = {name: initial.clone() for name in CONDITIONS}
        all_blocks = np.concatenate(
            (
                np.repeat(past_blocks[start_index][None], CANDIDATE_COUNT, axis=0),
                candidate_blocks[start_index],
            ),
            axis=1,
        )
        for horizon in range(HORIZONS):
            action = torch.as_tensor(
                np.ascontiguousarray(all_blocks[:, horizon : horizon + 3]),
                device=stack["device"],
            )
            started = time.perf_counter()
            adaptive_base = stack["runtime"].base_predict(
                stack["base"], histories["adaptive"], action
            )
            adaptive_output, calls, _ = bridge.adaptive_sparse(
                stack, histories["adaptive"], action, adaptive_base
            )
            stack["runtime"].synchronize(stack["device"])
            latency_seconds[condition_index["adaptive"], start_index] += (
                time.perf_counter() - started
            )
            calls_np = calls.cpu().numpy().astype(np.int8)
            adaptive_calls[start_index, horizon] = calls_np
            histories["adaptive"] = torch.cat(
                (histories["adaptive"][:, 1:], adaptive_output[:, None, :]), dim=1
            )

            permutation = matched_permutation(start_index, horizon)
            permutations[start_index, horizon] = permutation
            assigned_np = calls_np[permutation]
            matched_calls[start_index, horizon] = assigned_np
            exact_compute &= bool(
                np.array_equal(
                    np.bincount(calls_np, minlength=5),
                    np.bincount(assigned_np, minlength=5),
                )
                and int(np.minimum(calls_np, 3).sum())
                == int(np.minimum(assigned_np, 3).sum())
            )
            assigned = torch.as_tensor(
                assigned_np, dtype=torch.long, device=stack["device"]
            )
            started = time.perf_counter()
            matched_base = stack["runtime"].base_predict(
                stack["base"], histories["matched"], action
            )
            matched_output = bridge.forced_sparse(
                stack, histories["matched"], action, matched_base, assigned
            )
            stack["runtime"].synchronize(stack["device"])
            latency_seconds[condition_index["matched"], start_index] += (
                time.perf_counter() - started
            )
            histories["matched"] = torch.cat(
                (histories["matched"][:, 1:], matched_output[:, None, :]), dim=1
            )

            for name, depth in (("fixed_d1", 1), ("fixed_d4", 4)):
                started = time.perf_counter()
                base_prediction = stack["runtime"].base_predict(
                    stack["base"], histories[name], action
                )
                selected = torch.full(
                    (CANDIDATE_COUNT,), depth, dtype=torch.long, device=stack["device"]
                )
                with torch.inference_mode():
                    output = stack["solver"].forward_selected(
                        histories[name], action, base_prediction, selected
                    )
                stack["runtime"].synchronize(stack["device"])
                latency_seconds[condition_index[name], start_index] += (
                    time.perf_counter() - started
                )
                histories[name] = torch.cat((histories[name][:, 1:], output[:, None, :]), dim=1)
        goal = goal_latents[start_index].astype(np.float64)
        for name in CONDITIONS:
            terminal = histories[name][:, -1].detach().cpu().numpy().astype(np.float64)
            costs[condition_index[name], start_index] = np.square(terminal - goal).mean(axis=1)
        finite &= bool(np.isfinite(costs[:, start_index]).all())
        print(f"Phase C model ranking {start_index + 1}/{START_COUNT}", flush=True)
    audit_after = stack["runtime"].module_audit(stack["base"], stack["solver"], stack["v1"])
    modules_unchanged = audit_before == audit_after and audit_after["passed"]
    real_error = simulation["real_terminal_error"].astype(np.float64)
    per_start: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITIONS}
    for name in CONDITIONS:
        ci = condition_index[name]
        for start_index in range(START_COUNT):
            item = rank_metrics(costs[ci, start_index], real_error[start_index])
            item["start_id"] = str(simulation["start_id"][start_index])
            per_start[name].append(item)
    aggregate = {name: summarize(per_start[name]) for name in CONDITIONS}
    selected = {
        name: np.asarray([item["selected_index"] for item in per_start[name]], dtype=np.int16)
        for name in CONDITIONS
    }
    disagreement = {}
    for left_index, left in enumerate(CONDITIONS):
        for right in CONDITIONS[left_index + 1 :]:
            disagreement[f"{left}_vs_{right}"] = float(np.mean(selected[left] != selected[right]))
    adaptive = aggregate["adaptive"]
    matched = aggregate["matched"]
    criterion = {
        "adaptive_mean_spearman_strictly_greater_than_matched": bool(
            adaptive["mean_spearman"] is not None
            and matched["mean_spearman"] is not None
            and adaptive["mean_spearman"] > matched["mean_spearman"]
        ),
        "adaptive_mean_real_regret_strictly_less_than_matched": bool(
            adaptive["mean_real_regret"] < matched["mean_real_regret"]
        ),
        "adaptive_mean_top5_overlap_at_least_matched": bool(
            adaptive["mean_top5_overlap"] >= matched["mean_top5_overlap"]
        ),
        "finite_and_exact_compute": bool(finite and exact_compute and modules_unchanged),
    }
    promising = all(criterion.values())
    label = "planner_ranking_promising" if promising else "planner_ranking_not_supported"
    compute = []
    for start_index in range(START_COUNT):
        for horizon in range(HORIZONS):
            compute.append(
                {
                    "start_id": str(simulation["start_id"][start_index]),
                    "horizon": horizon + 1,
                    "depth_histogram": np.bincount(
                        adaptive_calls[start_index, horizon], minlength=5
                    )[1:].tolist(),
                    "adaptive_refiner_calls": int(
                        adaptive_calls[start_index, horizon].sum()
                    ),
                    "matched_refiner_calls": int(matched_calls[start_index, horizon].sum()),
                    "adaptive_reached_gate_evaluations": int(
                        np.minimum(adaptive_calls[start_index, horizon], 3).sum()
                    ),
                    "matched_reached_gate_evaluations": int(
                        np.minimum(matched_calls[start_index, horizon], 3).sum()
                    ),
                    "exact": True,
                }
            )
    operation_ledger = bridge.read_json(bridge.V5 / "operation_ledger.json")
    model_ledger = operation_ledger["model"]
    base_flops = int(model_ledger["base_flops_per_row"])
    depth1_flops = int(model_ledger["mandatory_depth1_refiner_flops_per_row"])
    adapter_flops = int(model_ledger["each_additional_adapter_flops_per_row"])
    gate_flops = int(operation_ledger["gate"]["total_flops_per_reached_evaluation"])
    candidate_transitions = START_COUNT * CANDIDATE_COUNT * HORIZONS
    adaptive_refiner_calls = int(adaptive_calls.sum())
    matched_refiner_calls = int(matched_calls.sum())
    adaptive_gate_evaluations = int(np.minimum(adaptive_calls, 3).sum())
    matched_gate_evaluations = int(np.minimum(matched_calls, 3).sum())

    def counted_flops(refiner_calls: int, gate_evaluations: int) -> dict[str, int]:
        components = {
            "base_flops": candidate_transitions * base_flops,
            "mandatory_depth1_flops": candidate_transitions * depth1_flops,
            "additional_refiner_flops": (
                refiner_calls - candidate_transitions
            ) * adapter_flops,
            "gate_flops": gate_evaluations * gate_flops,
        }
        components["total_counted_flops"] = sum(components.values())
        return components

    aggregate_compute = {
        "candidate_transitions_per_condition": candidate_transitions,
        "adaptive": {
            "depth_histogram": np.bincount(adaptive_calls.ravel(), minlength=5)[1:].tolist(),
            "refiner_calls": adaptive_refiner_calls,
            "reached_gate_evaluations": adaptive_gate_evaluations,
            **counted_flops(adaptive_refiner_calls, adaptive_gate_evaluations),
        },
        "matched": {
            "depth_histogram": np.bincount(matched_calls.ravel(), minlength=5)[1:].tolist(),
            "refiner_calls": matched_refiner_calls,
            "reached_gate_evaluations": matched_gate_evaluations,
            **counted_flops(matched_refiner_calls, matched_gate_evaluations),
        },
        "fixed_d1": {
            "depth_histogram": [candidate_transitions, 0, 0, 0],
            "refiner_calls": candidate_transitions,
            "reached_gate_evaluations": 0,
            **counted_flops(candidate_transitions, 0),
        },
        "fixed_d4": {
            "depth_histogram": [0, 0, 0, candidate_transitions],
            "refiner_calls": 4 * candidate_transitions,
            "reached_gate_evaluations": 0,
            **counted_flops(4 * candidate_transitions, 0),
        },
    }
    outcome_ranges = np.ptp(real_error, axis=1)
    outcome_diagnostics = {
        "exactly_constant_real_outcome_starts": int(np.sum(outcome_ranges == 0.0)),
        "real_outcome_range_at_most_1e-12_starts": int(np.sum(outcome_ranges <= 1e-12)),
        "median_real_outcome_range_m": float(np.median(outcome_ranges)),
        "maximum_real_outcome_range_m": float(np.max(outcome_ranges)),
        "defined_spearman_starts": int(aggregate["adaptive"]["defined_spearman_starts"]),
        "adaptive_matched_selected_disagreement_count": int(
            np.sum(selected["adaptive"] != selected["matched"])
        ),
        "adaptive_minus_matched_mean_spearman": float(
            aggregate["adaptive"]["mean_spearman"] - aggregate["matched"]["mean_spearman"]
        ),
        "adaptive_minus_matched_mean_real_regret_m": float(
            aggregate["adaptive"]["mean_real_regret"]
            - aggregate["matched"]["mean_real_regret"]
        ),
        "adaptive_minus_matched_mean_top5_overlap": float(
            aggregate["adaptive"]["mean_top5_overlap"]
            - aggregate["matched"]["mean_top5_overlap"]
        ),
        "chance_expected_top5_overlap": TOP_K / CANDIDATE_COUNT,
        "interpretation": (
            "The fixed pilot rule passed, but ranking correlations were near zero and candidate "
            "outcomes were tie-heavy; this is weak directional evidence, not a practically "
            "established planning bridge."
        ),
    }
    bridge.write_npz(
        ROOT / "phase_c_metrics.npz",
        {
            "condition": np.asarray(CONDITIONS, dtype="U12"),
            "start_id": simulation["start_id"],
            "real_terminal_error": real_error,
            "predicted_terminal_cost": costs,
            "adaptive_calls": adaptive_calls,
            "matched_calls": matched_calls,
            "matched_permutation": permutations,
            "selected_candidate_index": np.stack([selected[name] for name in CONDITIONS]),
            "real_regret": np.stack(
                [
                    np.asarray([item["real_regret"] for item in per_start[name]], dtype=np.float64)
                    for name in CONDITIONS
                ]
            ),
            "spearman": np.stack(
                [
                    np.asarray(
                        [np.nan if item["spearman"] is None else item["spearman"] for item in per_start[name]],
                        dtype=np.float64,
                    )
                    for name in CONDITIONS
                ]
            ),
            "top5_overlap": np.stack(
                [
                    np.asarray([item["top5_overlap"] for item in per_start[name]], dtype=np.float64)
                    for name in CONDITIONS
                ]
            ),
            "latency_seconds": latency_seconds,
        },
    )
    phase_c = {
        "tested": True,
        "scientific_role": "pilot_only",
        "terminal_label": label,
        "start_count": START_COUNT,
        "candidate_count_per_start": CANDIDATE_COUNT,
        "candidate_outcomes": START_COUNT * CANDIDATE_COUNT,
        "consumed_start_ids": simulation["start_id"].tolist(),
        "consumed_seed_tuples": config["starts"],
        "predicted_cost": config["goal"]["predicted_cost"],
        "real_terminal_error": config["goal"]["real_terminal_error"],
        "aggregate": aggregate,
        "per_start": per_start,
        "selected_action_disagreement_frequency": disagreement,
        "exact_compute": compute,
        "aggregate_compute": aggregate_compute,
        "outcome_diagnostics": outcome_diagnostics,
        "criterion": criterion,
        "descriptive_latency": {
            "interpretation": "synchronized five-step batched model inference; descriptive only",
            "mean_seconds_per_start": {
                name: float(latency_seconds[condition_index[name]].mean()) for name in CONDITIONS
            },
        },
        "finite": finite,
        "frozen_modules_and_gradients_unchanged": modules_unchanged,
        "phase_c_config_sha256": bridge.sha256_file(ROOT / "PHASE_C_PILOT_CONFIG.json"),
        "simulator_artifact_sha256": simulator_manifest["sha256"],
        "metrics_sha256": bridge.sha256_file(ROOT / "phase_c_metrics.npz"),
        "caveats": [
            "This is a 20-start pilot, not a closed-loop control confirmation.",
            "Candidates are PlanOracle-centered perturbations at a fixed mid-trajectory warm start.",
            "The latent goal image is simulator-rendered and the real endpoint is position error only.",
            "Candidate outcomes were tie-heavy and aggregate rank correlations were near zero.",
            "No claim of universal planning improvement or wall-clock acceleration follows.",
        ],
    }
    results = bridge.read_json(ROOT / "RESULTS.json")
    results["phase_c"] = phase_c
    results["terminal_label"] = label
    results["claim_scope"] = "five-step composition plus small fixed-candidate ranking pilot"
    bridge.write_json(ROOT / "RESULTS.json", results)
    bridge.write_report(results)
    bridge.update_status(
        label,
        list(bridge.PHASE_A_IDS),
        list(bridge.PHASE_B_IDS),
        (
            "bounded study complete; do not automatically launch follow-on work"
            if promising
            else "stop: no larger control task is warranted for this frozen mechanism"
        ),
    )
    status = bridge.read_json(ROOT / "RUN_STATUS.json")
    status["completed_case_ids"]["phase_c"] = simulation["start_id"].tolist()
    bridge.write_json(ROOT / "RUN_STATUS.json", status)
    bridge.append_note(
        f"Phase C completed with terminal label {label}; frozen planner criterion components "
        f"were {criterion}. No follow-on task was launched."
    )
    return results


if __name__ == "__main__":
    print(json.dumps(bridge.jsonable(run()), sort_keys=True))
