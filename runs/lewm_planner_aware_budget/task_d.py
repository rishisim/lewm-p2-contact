"""Task D frozen-cohort, scheduling, inference, and integrity utilities."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


DEPTHS = (0, 1, 2, 4)
POPULATIONS = (64, 128, 300)
ELITES = {64: 8, 128: 16, 300: 38}
CELLS = tuple(f"d{d}_p{p:03d}" for d in DEPTHS for p in POPULATIONS)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(base: int, *parts: int) -> int:
    payload = ":".join(map(str, (base, *parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def cell_order(order_seed: int, row_id: int, candidate_seed: int) -> list[str]:
    rng = np.random.default_rng(derived_seed(order_seed, row_id, candidate_seed))
    return [CELLS[index] for index in rng.permutation(len(CELLS))]


def task_a_rows(dataset, get_index_columns, get_episode_index, config: dict) -> dict:
    _, episodes, steps = get_index_columns(dataset)
    _, inverse, lengths = get_episode_index(episodes, steps)
    offset = config["dataset"]["goal_offset_steps"]
    valid = np.flatnonzero(steps <= (lengths - offset - 1)[inverse])
    used: set[int] = set()
    result = {}
    for name in ("smoke", "tuning", "heldout"):
        spec = config["cohorts"][name]
        available = np.asarray([row for row in valid if int(row) not in used])
        chosen = np.sort(
            np.random.default_rng(spec["seed"]).choice(
                available, size=spec["starts"], replace=False
            )
        )
        used.update(map(int, chosen))
        result[name] = chosen
    return result


def task_b_expert_episodes(
    episodes: np.ndarray, task_a_episode_union: set[int]
) -> dict[str, list[int]]:
    unique, counts = np.unique(episodes, return_counts=True)
    eligible = np.asarray(
        [
            int(identifier)
            for identifier, count in zip(unique, counts)
            if count >= 101 and int(identifier) not in task_a_episode_union
        ]
    )
    result = {}
    for index, (name, count) in enumerate(
        (("fit", 20), ("selection", 10), ("evaluation", 10))
    ):
        chosen = np.random.default_rng(26072622 + index).choice(
            eligible, count, replace=False
        )
        result[name] = sorted(map(int, chosen))
    return result


def freeze_cohort(dataset, eval_module, config: dict) -> dict:
    _, episodes, steps = eval_module.get_index_columns(dataset)
    _, inverse, lengths = eval_module.get_episode_index(episodes, steps)
    task_a_cfg = config["task_a"]
    a_rows = task_a_rows(
        dataset, eval_module.get_index_columns, eval_module.get_episode_index, task_a_cfg
    )
    a_episodes = {
        int(episodes[row]) for rows in a_rows.values() for row in rows.tolist()
    }
    b_by_role = task_b_expert_episodes(np.asarray(episodes), a_episodes)
    c_episode = int(episodes[0])
    excluded = a_episodes | {
        episode for values in b_by_role.values() for episode in values
    } | {c_episode}

    offset = task_a_cfg["dataset"]["goal_offset_steps"]
    valid = np.flatnonzero(steps <= (lengths - offset - 1)[inverse])
    by_episode: dict[int, list[int]] = {}
    for row in valid:
        episode = int(episodes[row])
        if episode not in excluded:
            by_episode.setdefault(episode, []).append(int(row))
    rng = np.random.default_rng(config["cohort_seed"])
    selected_episodes = rng.choice(
        np.asarray(sorted(by_episode)), config["pilot_starts"] + config["sealed_starts"],
        replace=False,
    )
    entries = []
    for episode in selected_episodes:
        candidates = by_episode[int(episode)]
        row = int(candidates[int(rng.integers(len(candidates)))])
        initial, goal = dataset[row], dataset[row + offset]
        entries.append(
            {
                "row_id": row,
                "episode_id": int(episode),
                "start_step": int(steps[row]),
                "goal_row_id": row + offset,
                "goal_step": int(steps[row] + offset),
                "environment_seed": int(episode),
                "initial_state": np.asarray(initial["state"]).tolist(),
                "goal_state": np.asarray(goal["state"]).tolist(),
            }
        )
    pilot_n = config["pilot_starts"]
    payload = {
        "schema_version": 1,
        "status": "frozen_before_task_d_outcomes",
        "selection": {
            "unit": "unique_source_episode",
            "cohort_seed": config["cohort_seed"],
            "goal_offset_steps": offset,
            "no_outcome_replacement": True,
        },
        "exclusions": {
            "task_a_episodes": sorted(a_episodes),
            "task_a_rows_by_role": {
                name: list(map(int, rows)) for name, rows in a_rows.items()
            },
            "task_b_expert_episodes_by_role": b_by_role,
            "task_c_smoke_episode": c_episode,
            "task_b_offpolicy_seed_ranges": [
                [7100000, 7100019], [7110000, 7110009], [7120000, 7120009]
            ],
        },
        "pilot": entries[:pilot_n],
        "sealed": entries[pilot_n:],
    }
    payload["content_sha256"] = sha256_bytes(canonical_bytes(payload))
    return payload


def validate_cohort(payload: dict, pilot_n: int = 2, sealed_n: int = 24) -> None:
    if len(payload["pilot"]) != pilot_n or len(payload["sealed"]) != sealed_n:
        raise ValueError("cohort size mismatch")
    entries = payload["pilot"] + payload["sealed"]
    episodes = [row["episode_id"] for row in entries]
    if len(set(episodes)) != len(episodes):
        raise ValueError("source episodes are not disjoint")
    excluded = set(payload["exclusions"]["task_a_episodes"])
    excluded |= {
        episode
        for values in payload["exclusions"]["task_b_expert_episodes_by_role"].values()
        for episode in values
    }
    excluded.add(payload["exclusions"]["task_c_smoke_episode"])
    if excluded.intersection(episodes):
        raise ValueError("cohort overlaps prior source episodes")
    for row in entries:
        if row["goal_row_id"] - row["row_id"] != 25:
            raise ValueError("goal-row offset mismatch")


def expected_work(population: int, depth: int, replans: int) -> dict:
    if population not in POPULATIONS or depth not in DEPTHS:
        raise ValueError("unsupported cell")
    return {
        "replans": replans,
        "candidate_sequences_evaluated": replans * population * 20,
        "predicted_transition_rows": replans * population * 20 * 5,
        "base_calls": replans * 100,
        "base_rows": replans * population * 100,
        "refiner_stage_calls": [
            replans * 100 if stage < depth else 0 for stage in range(4)
        ],
        "refiner_stage_rows": [
            replans * population * 100 if stage < depth else 0 for stage in range(4)
        ],
        "encoder_calls_each": replans * 20,
        "terminal_cost_calls": replans * 20,
        "terminal_cost_rows": replans * population * 20,
    }


def hard_budget_allocate(quality: np.ndarray, costs: np.ndarray, budget: int) -> dict:
    """Exact multiple-choice allocation by Pareto-pruned dynamic programming."""
    quality = np.asarray(quality, float)
    costs = np.asarray(costs, np.int64)
    if quality.ndim != 2 or costs.shape != (quality.shape[1],):
        raise ValueError("invalid allocation shapes")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for row in quality:
        next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
        for spent, (value, choices) in states.items():
            for package, cost in enumerate(costs):
                total = spent + int(cost)
                if total > budget:
                    continue
                candidate = (value + float(row[package]), choices + (package,))
                if total not in next_states or candidate[0] > next_states[total][0]:
                    next_states[total] = candidate
        best = -math.inf
        pruned = {}
        for spent in sorted(next_states):
            if next_states[spent][0] > best:
                pruned[spent] = next_states[spent]
                best = next_states[spent][0]
        states = pruned
    if not states:
        raise ValueError("budget infeasible")
    spent, (value, choices) = max(
        states.items(), key=lambda item: (item[1][0], -item[0])
    )
    return {"choices": list(choices), "spent": spent, "slack": budget - spent, "value": value}


def brute_force_allocate(quality: np.ndarray, costs: np.ndarray, budget: int) -> dict:
    best = None
    for choices in itertools.product(range(quality.shape[1]), repeat=quality.shape[0]):
        spent = int(sum(costs[index] for index in choices))
        value = float(sum(quality[row, index] for row, index in enumerate(choices)))
        if spent <= budget and (best is None or value > best["value"]):
            best = {"choices": list(choices), "spent": spent, "slack": budget - spent, "value": value}
    if best is None:
        raise ValueError("budget infeasible")
    return best


def cluster_bootstrap_mean_difference(
    differences: np.ndarray, draws: int, seed: int
) -> dict:
    values = np.asarray(differences, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("invalid paired differences")
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return {
        "estimate": float(values.mean()),
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "draws_sha256": sha256_bytes(sampled.astype("<f8").tobytes()),
    }


def sign_permutation_pvalue(differences: np.ndarray, draws: int, seed: int) -> dict:
    values = np.asarray(differences, float)
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null = np.abs((signs * values).mean(1))
    return {
        "observed": observed,
        "pvalue": float((1 + np.count_nonzero(null >= observed)) / (draws + 1)),
        "draws_sha256": sha256_bytes(null.astype("<f8").tobytes()),
    }
