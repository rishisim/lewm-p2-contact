#!/usr/bin/env python3
"""Execute the frozen Task C 12-cell mechanics and latency grid."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from gymnasium.spaces import Box

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = ROOT / "work/task_c"
sys.path.insert(0, str(REPO / "le-wm"))
sys.path.insert(0, str(ROOT))

import stable_worldmodel as swm
from omegaconf import OmegaConf
import eval as lewm_eval
from pusht_cem_adapter import PushTRefinedCostModel
from run_task_b import load_base
from task_c_population import (
    BASE_COUNTED_FLOPS_BY_HISTORY,
    ENCODER_COUNTED_FLOPS_PER_ROW,
    PrefixComparableCEMSolver,
    PopulationConfig,
    counted_flops_per_call,
)


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("frozen Task C hardware requires MPS")
    return torch.device("mps")


def synchronize() -> None:
    torch.mps.synchronize()


def ledger_snapshot(model: PushTRefinedCostModel) -> dict:
    ledger = model.ledger
    return {
        "base_calls": ledger.base_calls,
        "base_rows": ledger.base_rows,
        "refiner_stage_calls": list(ledger.stage_calls),
        "refiner_stage_rows": list(ledger.stage_rows),
        "image_encoder_calls": ledger.image_encoder_calls,
        "image_encoder_rows": ledger.image_encoder_rows,
        "goal_encoder_calls": ledger.goal_encoder_calls,
        "goal_encoder_rows": ledger.goal_encoder_rows,
        "terminal_cost_calls": ledger.terminal_cost_calls,
        "terminal_cost_rows": ledger.terminal_cost_rows,
    }


def subtract(after: dict, before: dict) -> dict:
    result = {}
    for key, value in after.items():
        if isinstance(value, list):
            result[key] = [a - b for a, b in zip(value, before[key])]
        else:
            result[key] = value - before[key]
    return result


def expected_model_ledger(population: int, depth: int, replans: int = 1) -> dict:
    calls = replans * 20 * 5
    rows = replans * population * 20 * 5
    encodes = replans * 20
    return {
        "base_calls": calls,
        "base_rows": rows,
        "refiner_stage_calls": [calls if stage < depth else 0 for stage in range(4)],
        "refiner_stage_rows": [rows if stage < depth else 0 for stage in range(4)],
        "image_encoder_calls": encodes,
        "image_encoder_rows": encodes,
        "goal_encoder_calls": encodes,
        "goal_encoder_rows": encodes,
        "terminal_cost_calls": encodes,
        "terminal_cost_rows": replans * population * 20,
    }


def make_info(target: torch.device) -> tuple[dict, dict]:
    cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    dataset = lewm_eval.get_dataset(cfg, "pusht_expert_train.lance")
    transform = lewm_eval.img_transform(cfg)
    initial, goal = dataset[0], dataset[25]
    info = {
        "pixels": transform(initial["pixels"]).unsqueeze(0).to(target),
        "goal": transform(goal["pixels"]).unsqueeze(0).to(target),
        "action": torch.zeros(1, 1, 10, device=target),
    }
    identity = {
        "start_dataset_row": 0,
        "goal_dataset_row": 25,
        "initial_state": np.asarray(initial["state"]).tolist(),
        "goal_state": np.asarray(goal["state"]).tolist(),
    }
    return info, identity


def expanded_info(info: dict, samples: int) -> dict:
    return {
        key: value.unsqueeze(1).expand(1, samples, *value.shape[1:])
        for key, value in info.items()
    }


def load_models(target: torch.device, config: dict) -> dict[int, PushTRefinedCostModel]:
    checkpoint = ROOT / "pusht_refiner_checkpoint.pt"
    if sha256(checkpoint) != config["checkpoint_sha256"]["refiner"]:
        raise RuntimeError("Task C refiner checkpoint mismatch")
    models = {}
    for depth in config["depths"]:
        base = load_base(target)
        if depth == 0:
            model = PushTRefinedCostModel(base, None, 0)
        else:
            model = PushTRefinedCostModel.from_checkpoint(
                base,
                checkpoint,
                expected_sha256=config["checkpoint_sha256"]["refiner"],
                refinement_depth=depth,
            )
        models[depth] = model.to(target).eval().requires_grad_(False)
    return models


def anchor_check(model: PushTRefinedCostModel, info: dict) -> dict:
    model_device = next(model.parameters()).device
    generator = torch.Generator(device=model_device).manual_seed(26072610)
    candidates = torch.randn(
        1, 128, 5, 10, generator=generator, device=model_device
    )
    native_info = expanded_info(info, 128)
    boundary_info = expanded_info(info, 128)
    with torch.inference_mode():
        native = model.base.get_cost(native_info, candidates.clone())
        boundary = model.get_cost(boundary_info, candidates.clone())
    difference = torch.max(torch.abs(native - boundary)).item()
    return {
        "population": 128,
        "depth": 0,
        "same_candidates": True,
        "torch_equal": bool(torch.equal(native, boundary)),
        "maximum_absolute_cost_difference": float(difference),
        "passed": bool(torch.equal(native, boundary)),
        "scope": "native Task A get_cost versus depth-zero project boundary",
    }


def effect_checks(models: dict, info: dict, target: torch.device) -> dict:
    effects = {}
    for population in (64, 128, 300):
        generator = torch.Generator(device=target).manual_seed(26072610)
        candidates = torch.randn(
            1, population, 5, 10, generator=generator, device=target
        )
        costs = {}
        for depth, model in models.items():
            with torch.inference_mode():
                value = model.get_cost(
                    expanded_info(info, population), candidates.clone()
                )
            costs[depth] = value.detach().cpu().numpy()
        base = costs[0]
        effects[str(population)] = {}
        for depth in (1, 2, 4):
            effects[str(population)][str(depth)] = {
                "costs_changed": bool(np.any(costs[depth] != base)),
                "ranking_changed": bool(
                    np.any(np.argsort(costs[depth], axis=1) != np.argsort(base, axis=1))
                ),
                "finite": bool(np.isfinite(costs[depth]).all()),
            }
            if not (
                effects[str(population)][str(depth)]["costs_changed"]
                and effects[str(population)][str(depth)]["ranking_changed"]
            ):
                raise RuntimeError("positive depth did not change costs and rankings")
    return effects


def cell_label(depth: int, population: int) -> str:
    return f"d{depth}_p{population:03d}"


def main() -> None:
    config = json.loads((ROOT / "task_c_config.json").read_text())
    target = target_device()
    info, start_identity = make_info(target)
    models = load_models(target, config)
    anchor = anchor_check(models[0], info)
    if not anchor["passed"]:
        raise RuntimeError("Task A depth-zero native-equivalence anchor failed")
    effects = effect_checks(models, info, target)

    solvers = {}
    cells = []
    for depth in config["depths"]:
        for spec in config["populations"].values():
            population = PopulationConfig.create(
                spec["population"], elites=spec["elites"]
            )
            solver = PrefixComparableCEMSolver(
                model=models[depth],
                population=population,
                device=target,
                candidate_seed=config["planner"]["candidate_seed"],
            )
            solver.configure(
                action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
                n_envs=1,
                config=swm.PlanConfig(
                    horizon=5, receding_horizon=5, action_block=5
                ),
            )
            label = cell_label(depth, population.population)
            solvers[label] = (solver, models[depth], depth, population)
            cells.append(label)

    raw = {label: {"warmup_ns": [], "timed_ns": [], "order": []} for label in cells}
    smoke = {}
    call_counter = {label: 0 for label in cells}
    warmups = config["measurement"]["warmups_per_cell"]
    for label in cells:
        solver, model, depth, population = solvers[label]
        for warmup in range(warmups):
            before = ledger_snapshot(model)
            synchronize()
            started = time.perf_counter_ns()
            output = solver.solve(
                info,
                start_id=0,
                replan_index=call_counter[label],
            )
            synchronize()
            elapsed = time.perf_counter_ns() - started
            observed = subtract(ledger_snapshot(model), before)
            expected = expected_model_ledger(population.population, depth)
            if observed != expected:
                raise RuntimeError(f"{label} model ledger mismatch: {observed}")
            raw[label]["warmup_ns"].append(elapsed)
            call_counter[label] += 1
            if warmup == 0:
                actions = output["actions"].numpy()
                smoke[label] = {
                    "population": population.population,
                    "elite_count": population.elites,
                    "iterations": 20,
                    "horizon": 5,
                    "replans": 1,
                    "selected_depth": depth,
                    "finite_actions": bool(np.isfinite(actions).all()),
                    "finite_costs": bool(np.isfinite(output["costs"]).all()),
                    "selected_action_min": float(actions.min()),
                    "selected_action_max": float(actions.max()),
                    "selected_action_within_environment_bounds": bool(
                        np.all((actions >= -1) & (actions <= 1))
                    ),
                    "selected_action_shape": list(actions.shape),
                    "final_elite_cost": output["costs"],
                    "model_ledger": observed,
                    "cem_ledger_per_call": {
                        "candidate_sequences_evaluated": population.population * 20,
                        "predicted_transition_rows": population.population * 20 * 5,
                        "sampling_invocations": 20,
                        "update_invocations": 20,
                        "topk_invocations": 20,
                        "innovation_tensor_shape": [1, 300, 5, 10],
                        "candidate_tensor_shape": [1, population.population, 5, 10],
                        "topk_tensor_shape": [1, population.elites, 5, 10],
                    },
                }
                if (
                    not smoke[label]["finite_actions"]
                    or not smoke[label]["finite_costs"]
                ):
                    raise RuntimeError(f"{label} nonfinite smoke")

    order_rng = np.random.default_rng(config["measurement"]["order_seed"])
    run_order = []
    repetitions = config["measurement"]["timed_repetitions_per_cell"]
    for repetition in range(repetitions):
        order = [cells[index] for index in order_rng.permutation(len(cells))]
        for position, label in enumerate(order):
            solver, model, depth, population = solvers[label]
            before = ledger_snapshot(model)
            synchronize()
            started = time.perf_counter_ns()
            solver.solve(
                info,
                start_id=0,
                replan_index=call_counter[label],
            )
            synchronize()
            elapsed = time.perf_counter_ns() - started
            observed = subtract(ledger_snapshot(model), before)
            if observed != expected_model_ledger(population.population, depth):
                raise RuntimeError(f"{label} timed model ledger mismatch")
            raw[label]["timed_ns"].append(elapsed)
            raw[label]["order"].append(
                {"repetition": repetition, "position": position}
            )
            run_order.append(
                {
                    "repetition": repetition,
                    "position": position,
                    "cell": label,
                    "replan_index": call_counter[label],
                }
            )
            call_counter[label] += 1

    table = []
    for label in cells:
        solver, _, depth, population = solvers[label]
        values = np.asarray(raw[label]["timed_ns"], dtype=np.int64)
        solver.ledger.validate()
        table.append(
            {
                "cell": label,
                "depth": depth,
                "population": population.population,
                "elite_count": population.elites,
                "planner_call": smoke[label],
                "counted_flops": counted_flops_per_call(
                    population.population, population.elites, depth
                ),
                "latency": {
                    "unit": "ns",
                    "warmups_excluded": warmups,
                    "repetitions": int(len(values)),
                    "median_ns": int(np.median(values)),
                    "p95_ns": int(np.quantile(values, 0.95, method="linear")),
                    "boundary": "synchronize; solve; synchronize",
                    "constant_work": True,
                },
            }
        )

    aggregate = {
        "schema_version": 1,
        "decision": "task_c_mechanics_complete_no_control_effect_claim",
        "protocol_status": config["status"],
        "grid_complete": len(table) == 12,
        "start": start_identity,
        "anchor": anchor,
        "positive_depth_effect_checks": effects,
        "counted_flop_constants": {
            "encoder_per_row": ENCODER_COUNTED_FLOPS_PER_ROW,
            "base_by_history_length_per_row": BASE_COUNTED_FLOPS_BY_HISTORY,
            "method": (
                "hash-pinned module Linear/Conv executed-shape hooks plus explicit "
                "QK^T and AV matmul formulas"
            ),
            "linear_convention": (
                "one multiply-accumulate is two FLOPs; dense/conv totals use "
                "2 * reduction_width * output_elements and do not separately "
                "count bias or fused-kernel implementation details"
            ),
            "coverage": (
                "dense/conv and attention matmuls, refiner dense/residual, terminal "
                "squared error, CEM scale/shift and elite mean/std arithmetic"
            ),
            "exclusions": [
                "activation functions",
                "normalization",
                "attention softmax/scaling/masking",
                "other elementwise image/base-model operations",
                "random-number generation",
                "top-k comparisons",
                "indexing and gather",
                "tensor movement/allocation",
                "Python/controller overhead",
                "MPS synchronization and kernel implementation overhead"
            ],
            "complete_measured_flops": False,
            "smallest_unblocker": (
                "operator-complete device profiler counters for MPS SDPA, "
                "normalization, activation, RNG, and top-k kernels"
            ),
        },
        "measurement": {
            **config["measurement"],
            "device": str(target),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "run_order": run_order,
        },
        "cells": table,
    }
    dump(WORK / "raw_latency.json", raw)
    dump(ROOT / "task_c_results.json", aggregate)
    print(json.dumps({"cells": len(table), "anchor": anchor, "output": str(ROOT / "task_c_results.json")}, indent=2))


if __name__ == "__main__":
    main()
