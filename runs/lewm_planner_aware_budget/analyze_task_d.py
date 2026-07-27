#!/usr/bin/env python3
"""Seal compact Task D records and run the frozen opportunity analysis."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/task_d/sealed"
sys.path.insert(0, str(ROOT))

from task_d import (
    CELLS,
    canonical_bytes,
    cluster_bootstrap_mean_difference,
    file_sha256,
    hard_budget_allocate,
    sha256_bytes,
    sign_permutation_pvalue,
)


def parse(label):
    a, b = label.split("_")
    return int(a[1:]), int(b[1:])


def ci(values, seed, draws=9999):
    result = cluster_bootstrap_mean_difference(np.asarray(values), draws, seed)
    return {"mean": result["estimate"], "ci95": result["ci95"]}


def package_costs(records, mode):
    by_cell = {row["key"]["cell"]: row for row in records}
    if mode == "work":
        return np.asarray(
            [
                by_cell[cell]["counted_flops_per_call"]["counted_total"] * 10
                for cell in CELLS
            ],
            dtype=np.int64,
        )
    task_c = json.loads((ROOT / "task_c_results.json").read_text())
    task_c_cells = {row["cell"]: row for row in task_c["cells"]}
    return np.asarray(
        [
            round(task_c_cells[cell]["latency"]["median_ns"] * 10)
            for cell in CELLS
        ],
        dtype=np.int64,
    )


def executable_mix(train_mean, held, costs, budget, rng):
    n = held.shape[0]
    quality = np.broadcast_to(train_mean, (n, len(CELLS)))
    allocation = hard_budget_allocate(quality, costs, budget)
    choices = np.asarray(allocation["choices"])
    rng.shuffle(choices)
    return {
        **allocation,
        "choices": choices.tolist(),
        "quality": float(held[np.arange(n), choices].mean()),
        "counts": dict(sorted(Counter(CELLS[i] for i in choices).items())),
    }


def convex_envelope_reference(mean_quality, costs, per_start_budget):
    best = -math.inf
    recipe = None
    for a in range(len(CELLS)):
        if costs[a] <= per_start_budget and mean_quality[a] > best:
            best, recipe = float(mean_quality[a]), {CELLS[a]: 1.0}
        for b in range(a + 1, len(CELLS)):
            low, high = (a, b) if costs[a] <= costs[b] else (b, a)
            if costs[low] <= per_start_budget <= costs[high] and costs[high] > costs[low]:
                high_weight = (per_start_budget - costs[low]) / (costs[high] - costs[low])
                value = (1 - high_weight) * mean_quality[low] + high_weight * mean_quality[high]
                if value > best:
                    best = float(value)
                    recipe = {CELLS[low]: float(1 - high_weight), CELLS[high]: float(high_weight)}
    return {
        "quality": best,
        "weights": recipe,
        "executable": False,
        "label": "analytic convex-envelope reference",
    }


def constrained_oracle(train, held, costs, budget, allowed=None):
    if allowed is None:
        allowed = np.arange(len(CELLS))
    result = hard_budget_allocate(train[:, allowed], costs[allowed], budget)
    choices = allowed[np.asarray(result["choices"])]
    return {
        **result,
        "choices": choices.tolist(),
        "quality": float(held[np.arange(len(held)), choices].mean()),
        "counts": dict(sorted(Counter(CELLS[i] for i in choices).items())),
    }


def best_one_axis(train, held, costs, budget, axis):
    candidates = []
    values = (0, 1, 2, 4) if axis == "population_only" else (64, 128, 300)
    for fixed in values:
        allowed = np.asarray(
            [
                i
                for i, cell in enumerate(CELLS)
                if (parse(cell)[0] == fixed if axis == "population_only" else parse(cell)[1] == fixed)
            ]
        )
        try:
            candidate = constrained_oracle(train, held, costs, budget, allowed)
            candidate["fixed_depth" if axis == "population_only" else "fixed_population"] = fixed
            candidate["training_quality"] = float(
                train[np.arange(len(train)), candidate["choices"]].mean()
            )
            candidates.append(candidate)
        except ValueError:
            pass
    return max(candidates, key=lambda row: row["training_quality"])


def main():
    config = json.loads((ROOT / "task_d_config.json").read_text())
    cohort = json.loads((ROOT / "task_d_cohort.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted(WORK.glob("*.json"))]
    expected = 24 * 2 * 12
    if len(records) != expected:
        raise RuntimeError(f"incomplete sealed grid: {len(records)} != {expected}")
    keys = [tuple(row["key"][key] for key in ("row_id", "candidate_seed", "cell")) for row in records]
    if len(set(keys)) != expected:
        raise RuntimeError("duplicate sealed key")
    if any(row["failure"] is not None or row["normalized_return"] is None for row in records):
        raise RuntimeError("sealed mechanical/nonfinite failure present")
    rows = [entry["row_id"] for entry in cohort["sealed"]]
    seeds = config["execution"]["sealed_candidate_seeds"]
    lookup = {(r["key"]["row_id"], r["key"]["candidate_seed"], r["key"]["cell"]): r for r in records}
    quality = np.asarray([[[lookup[row, seed, cell]["normalized_return"] for cell in CELLS] for seed in seeds] for row in rows])
    success = np.asarray([[[lookup[row, seed, cell]["success"] for cell in CELLS] for seed in seeds] for row in rows], float)
    cumulative = np.asarray([[[lookup[row, seed, cell]["cumulative_task_cost"] for cell in CELLS] for seed in seeds] for row in rows])

    cells = {}
    for index, cell in enumerate(CELLS):
        subset = [lookup[row, seed, cell] for row in rows for seed in seeds]
        cells[cell] = {
            "depth": parse(cell)[0],
            "population": parse(cell)[1],
            "success_rate": float(success[:, :, index].mean()),
            "normalized_return": ci(quality[:, :, index].mean(1), 2026072691 + index),
            "cumulative_cost": ci(cumulative[:, :, index].mean(1), 2026072791 + index),
            "final_cost_mean": float(np.mean([r["final_task_cost"] for r in subset])),
            "episode_length_mean": float(np.mean([r["episode_length"] for r in subset])),
            "planner_calls_mean": float(np.mean([r["work"]["replans"] for r in subset])),
            "counted_flops_per_call": subset[0]["counted_flops_per_call"]["counted_total"],
            "counted_flops_executed_mean": float(np.mean([r["counted_flops_executed"] for r in subset])),
            "planner_latency_ms_median": float(np.median([r["planner_latency_ns"] for r in subset]) / 1e6),
            "episode_latency_ms_median": float(np.median([r["episode_latency_ns"] for r in subset]) / 1e6),
        }

    contrasts = {}
    families = []
    for population in (64, 128, 300):
        indices = [CELLS.index(f"d{d}_p{population:03d}") for d in (0, 1, 2, 4)]
        families += [(indices[a], indices[b]) for a in range(4) for b in range(a + 1, 4)]
    for depth in (0, 1, 2, 4):
        indices = [CELLS.index(f"d{depth}_p{p:03d}") for p in (64, 128, 300)]
        families += [(indices[a], indices[b]) for a in range(3) for b in range(a + 1, 3)]
    for a, b in families:
        diff = (quality[:, :, b] - quality[:, :, a]).mean(1)
        label = f"{CELLS[b]}-minus-{CELLS[a]}"
        contrasts[label] = {
            "return": cluster_bootstrap_mean_difference(diff, 9999, config["analysis"]["bootstrap_seed"] + a * 20 + b),
            "permutation": sign_permutation_pvalue(diff, 9999, config["analysis"]["permutation_seed"] + a * 20 + b),
            "success_risk_difference": float((success[:, :, b] - success[:, :, a]).mean()),
        }
    depth_main = np.mean(
        quality[:, :, [CELLS.index(f"d4_p{p:03d}") for p in (64, 128, 300)]]
        - quality[:, :, [CELLS.index(f"d0_p{p:03d}") for p in (64, 128, 300)]],
        axis=(1, 2),
    )
    population_main = np.mean(
        quality[:, :, [CELLS.index(f"d{d}_p300") for d in (0, 1, 2, 4)]]
        - quality[:, :, [CELLS.index(f"d{d}_p064") for d in (0, 1, 2, 4)]],
        axis=(1, 2),
    )
    interaction = np.mean(
        (quality[:, :, [CELLS.index(f"d4_p{p:03d}") for p in (64, 128, 300)]]
         - quality[:, :, [CELLS.index(f"d0_p{p:03d}") for p in (64, 128, 300)]])[:, :, -1]
        - (quality[:, :, [CELLS.index(f"d4_p{p:03d}") for p in (64, 128, 300)]]
           - quality[:, :, [CELLS.index(f"d0_p{p:03d}") for p in (64, 128, 300)]])[:, :, 0],
        axis=1,
    )
    effects = {
        "depth_d4_minus_d0_average_population": cluster_bootstrap_mean_difference(depth_main, 9999, 2026072901),
        "population_p300_minus_p64_average_depth": cluster_bootstrap_mean_difference(population_main, 9999, 2026072902),
        "extreme_difference_in_differences": cluster_bootstrap_mean_difference(interaction, 9999, 2026072903),
    }

    winner_a = np.argmax(quality[:, 0], axis=1)
    winner_b = np.argmax(quality[:, 1], axis=1)
    held_regret_ab = quality[:, 1].max(1) - quality[np.arange(24), 1, winner_a]
    held_regret_ba = quality[:, 0].max(1) - quality[np.arange(24), 0, winner_b]
    fixed_a = int(np.argmax(quality[:, 0].mean(0)))
    fixed_b = int(np.argmax(quality[:, 1].mean(0)))
    reproducible = np.concatenate(
        [
            (held_regret_ab <= 0.03) & (quality[np.arange(24), 1, winner_a] - quality[:, 1, fixed_a] >= 0.03),
            (held_regret_ba <= 0.03) & (quality[np.arange(24), 0, winner_b] - quality[:, 0, fixed_b] >= 0.03),
        ]
    )
    counts = Counter(np.concatenate([winner_a, winner_b]).tolist())
    probabilities = np.asarray(list(counts.values())) / 48
    taus = [kendalltau(quality[i, 0], quality[i, 1]).statistic for i in range(24)]
    pair_agreements = []
    for a in range(12):
        for b in range(a + 1, 12):
            pair_agreements.append(np.mean(np.sign(quality[:, 0, a] - quality[:, 0, b]) == np.sign(quality[:, 1, a] - quality[:, 1, b])))
    action_l2 = []
    for row in rows:
        for cell in CELLS:
            ca = lookup[row, seeds[0], cell]["planner_calls"]
            cb = lookup[row, seeds[1], cell]["planner_calls"]
            for a, b in zip(ca, cb):
                action_l2.append(float(np.linalg.norm(np.asarray(a["selected_action_block"]) - np.asarray(b["selected_action_block"]))))
    centered = quality - quality.mean(2, keepdims=True)
    heterogeneity = {
        "winner_entropy_nats": float(-np.sum(probabilities * np.log(probabilities))),
        "winner_counts": {CELLS[k]: v for k, v in sorted(counts.items())},
        "exact_cross_seed_winner_agreement": float(np.mean(winner_a == winner_b)),
        "kendall_tau_mean": float(np.nanmean(taus)),
        "pairwise_sign_agreement_mean": float(np.mean(pair_agreements)),
        "selected_action_block_l2_median": float(np.median(action_l2)),
        "between_start_variance": float(np.var(centered.mean(1), ddof=1)),
        "within_start_across_seed_variance": float(np.mean(np.var(centered, axis=1, ddof=1))),
        "reproducible_preference_fraction": float(reproducible.mean()),
        "tolerance": 0.03,
    }

    opportunity = {}
    for mode, raw_budgets in (
        ("work", config["budgets"]["counted_flops_per_allowed_call"]),
        ("latency", [round(x * 1e9) for x in config["budgets"]["synchronized_latency_seconds_per_allowed_call"]]),
    ):
        costs = package_costs(records, mode)
        opportunity[mode] = {}
        for budget_per_call in raw_budgets:
            budget = int(budget_per_call * 10 * 24)
            directions = []
            for train_seed, held_seed in ((0, 1), (1, 0)):
                train, held = quality[:, train_seed], quality[:, held_seed]
                feasible = np.flatnonzero(costs <= int(budget_per_call * 10))
                fixed = int(feasible[np.argmax(train[:, feasible].mean(0))])
                fixed_quality = float(held[:, fixed].mean())
                mix = executable_mix(train.mean(0), held, costs, budget, np.random.default_rng(2026072693 + train_seed))
                envelope = convex_envelope_reference(
                    train.mean(0), costs, int(budget_per_call * 10)
                )
                joint = constrained_oracle(train, held, costs, budget)
                depth_only = best_one_axis(train, held, costs, budget, "depth_only")
                population_only = best_one_axis(train, held, costs, budget, "population_only")
                choices = np.asarray(joint["choices"])
                null = []
                rng = np.random.default_rng(2026072693 + train_seed + int(budget_per_call % 10000))
                for _ in range(9999):
                    shuffled = rng.permutation(choices)
                    null.append(float(held[np.arange(24), shuffled].mean()))
                comparison_cis = {}
                baselines = {
                    "fixed": np.full(24, fixed),
                    "mixture": np.asarray(mix["choices"]),
                    "depth_only": np.asarray(depth_only["choices"]),
                    "population_only": np.asarray(population_only["choices"]),
                }
                for offset, (name, baseline_choices) in enumerate(baselines.items()):
                    paired = (
                        held[np.arange(24), choices]
                        - held[np.arange(24), baseline_choices]
                    )
                    comparison_cis[name] = cluster_bootstrap_mean_difference(
                        paired, 9999, 2026073100 + train_seed * 100 + offset
                    )
                directions.append(
                    {
                        "train_seed": seeds[train_seed],
                        "held_seed": seeds[held_seed],
                        "fixed_package": CELLS[fixed],
                        "fixed_quality": fixed_quality,
                        "mixture": mix,
                        "analytic_convex_envelope": envelope,
                        "joint": joint,
                        "depth_only": depth_only,
                        "population_only": population_only,
                        "joint_minus_fixed": joint["quality"] - fixed_quality,
                        "joint_minus_mixture": joint["quality"] - mix["quality"],
                        "joint_minus_depth_only": joint["quality"] - depth_only["quality"],
                        "joint_minus_population_only": joint["quality"] - population_only["quality"],
                        "comparison_uncertainty": comparison_cis,
                        "histogram_randomization_mean": float(np.mean(null)),
                        "histogram_assignment_pvalue": float((1 + np.count_nonzero(np.asarray(null) >= joint["quality"])) / 10000),
                    }
                )
            opportunity[mode][str(budget_per_call)] = directions

    best_cell = max(cells, key=lambda cell: cells[cell]["normalized_return"]["mean"])
    max_cross_seed_joint = max(
        direction["joint_minus_fixed"]
        for table in opportunity.values()
        for directions in table.values()
        for direction in directions
    )
    decision = {
        "task_e": "stop_two_axis_dispatcher",
        "supported": False,
        "reason": "preregistered conjunction not satisfied",
        "best_fixed_package": best_cell,
        "maximum_cross_seed_joint_minus_fixed_point_estimate": max_cross_seed_joint,
        "mapping": "inspect one-axis margins; if neither is robust, improve refinement/value targets before dispatcher work",
    }
    ledger = {
        "schema_version": 1,
        "status": "complete",
        "records": records,
        "record_count": len(records),
        "record_key_sha256": sha256_bytes(canonical_bytes(sorted(keys))),
        "cohort_sha256": file_sha256(ROOT / "task_d_cohort.json"),
    }
    (ROOT / "task_d_ledger.json").write_bytes(canonical_bytes(ledger))
    result = {
        "schema_version": 1,
        "grid_complete": True,
        "record_count": len(records),
        "starts": 24,
        "candidate_seeds": seeds,
        "failures": 0,
        "cells": cells,
        "contrasts": contrasts,
        "factorial_effects": effects,
        "heterogeneity": heterogeneity,
        "opportunity": opportunity,
        "decision": decision,
        "limitations": [
            "success and interaction precision are limited at 24 start clusters",
            "counted FLOPs exclude material MPS operators and are not complete FLOPs",
            "episode latency is outcome-dependent; Task C constant-work latency indexes budgets",
            "two candidate seeds estimate but do not exhaust CEM sampling variation",
            "oracle allocations are diagnostic ceilings, not adaptive-control results",
        ],
    }
    (ROOT / "task_d_results.json").write_bytes(canonical_bytes(result))
    print(json.dumps({"ledger": len(records), "best_cell": best_cell, "decision": decision["task_e"]}))


if __name__ == "__main__":
    main()
