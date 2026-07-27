#!/usr/bin/env python3
"""Restartable bounded Task D closed-loop fixed-grid runner."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = ROOT / "work/task_d"
sys.path.insert(0, str(REPO / "le-wm"))
sys.path.insert(0, str(ROOT))

import stable_worldmodel as swm
from omegaconf import OmegaConf
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal
import eval as lewm_eval

from pusht_cem_adapter import PushTRefinedCostModel
from qualify import processors
from run_task_b import load_base
from run_task_c import expected_model_ledger, ledger_snapshot, subtract
from task_c_population import (
    PrefixComparableCEMSolver,
    PopulationConfig,
    counted_flops_per_call,
)
from task_d import CELLS, cell_order, expected_work, file_sha256, validate_cohort


def atomic_dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def synchronize() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


class RecordedSolver:
    """Bind Task C seed ownership to the policy's implicit replan calls."""

    def __init__(self, solver, model, start_id: int):
        self.solver = solver
        self.model = model
        self.start_id = start_id
        self.replan_index = 0
        self.calls: list[dict] = []

    def configure(self, **kwargs):
        result = self.solver.configure(**kwargs)
        self._n_envs = kwargs["n_envs"]
        return result

    @property
    def action_dim(self):
        return self.solver.action_dim

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def horizon(self):
        return self.solver.horizon

    def __call__(self, info, init_action=None):
        return self.solve(info, init_action)

    def solve(self, info, init_action=None):
        info = {
            key: np.asarray(value) if isinstance(value, list) else value
            for key, value in info.items()
        }
        before = ledger_snapshot(self.model)
        synchronize()
        started = time.perf_counter_ns()
        output = self.solver.solve(
            info,
            init_action=init_action,
            start_id=self.start_id,
            replan_index=self.replan_index,
        )
        synchronize()
        elapsed = time.perf_counter_ns() - started
        observed = subtract(ledger_snapshot(self.model), before)
        actions = np.asarray(output["actions"])
        self.calls.append(
            {
                "replan_index": self.replan_index,
                "planner_latency_ns": elapsed,
                "selected_action_block": actions.reshape(-1).tolist(),
                "selected_action_min": float(actions.min()),
                "selected_action_max": float(actions.max()),
                "elite_cost_mean": float(output["costs"][0]),
                "model_ledger": observed,
            }
        )
        self.replan_index += 1
        return output


def load_models(device: torch.device, config: dict) -> dict[int, PushTRefinedCostModel]:
    checkpoint = ROOT / "pusht_refiner_checkpoint.pt"
    if file_sha256(checkpoint) != config["hashes"]["refiner_checkpoint"]:
        raise RuntimeError("Task D refiner checkpoint mismatch")
    result = {}
    for depth in config["depths"]:
        base = load_base(device)
        result[depth] = (
            PushTRefinedCostModel(base, None, 0)
            if depth == 0
            else PushTRefinedCostModel.from_checkpoint(
                base,
                checkpoint,
                expected_sha256=config["hashes"]["refiner_checkpoint"],
                refinement_depth=depth,
            )
        ).to(device).eval().requires_grad_(False)
    return result


def parse_cell(label: str) -> tuple[int, int]:
    depth, population = label.split("_")
    return int(depth[1:]), int(population[1:])


def evaluate(
    dataset,
    eval_cfg,
    process,
    model,
    entry: dict,
    label: str,
    candidate_seed: int,
    config: dict,
) -> dict:
    depth, population = parse_cell(label)
    solver = PrefixComparableCEMSolver(
        model=model,
        population=PopulationConfig.create(
            population, elites=config["populations"][str(population)]["elites"]
        ),
        device=torch.device("mps"),
        candidate_seed=candidate_seed,
    )
    recorded = RecordedSolver(solver, model, entry["row_id"])
    plan = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    policy = swm.policy.WorldModelPolicy(
        solver=recorded,
        config=plan,
        process=process,
        transform={
            "pixels": lewm_eval.img_transform(eval_cfg),
            "goal": lewm_eval.img_transform(eval_cfg),
        },
    )
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=2 * config["execution"]["max_environment_steps"],
        image_shape=(224, 224),
    )
    world.set_policy(policy)
    init, goal, _ = _extract_init_goal(
        dataset, [entry["episode_id"]], [entry["start_step"]], 25
    )
    callables = OmegaConf.to_container(eval_cfg.eval.callables, resolve=True)
    rewards: list[float] = []
    distances: list[float] = []
    successes = np.zeros(1, dtype=bool)
    synchronize()
    episode_started = time.perf_counter_ns()
    world.reset(seed=init.get("seed"))
    merged = {**init, **goal}
    _apply_callables(
        world.envs.envs[0].unwrapped,
        callables,
        {key: value[0] for key, value in merged.items()},
    )
    shape_prefix = world.infos["pixels"].shape[:2]
    for source in (init, goal):
        for key, value in source.items():
            if key in world.infos or key in goal:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()
    goal_snapshot = {key: world.infos[key].copy() for key in goal}
    initial_distance = float(
        np.linalg.norm(np.asarray(goal["goal_state"][0]) - np.asarray(init["state"][0]))
    )

    def on_step(current):
        current.infos.update({key: value.copy() for key, value in goal_snapshot.items()})
        successes[:] |= current.terminateds
        rewards.append(float(current.rewards[0]))
        distances.append(-float(current.rewards[0]))

    failure = None
    try:
        world._run(
            max_steps=config["execution"]["max_environment_steps"],
            mode="wait",
            on_step=on_step,
        )
        synchronize()
    except Exception as error:
        synchronize()
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    episode_latency = time.perf_counter_ns() - episode_started
    world.envs.close()
    replans = len(recorded.calls)
    for call in recorded.calls:
        if call["model_ledger"] != expected_model_ledger(population, depth):
            raise RuntimeError(f"{label} model ledger mismatch")
    task_cost = float(sum(distances)) if distances else None
    normalized_return = (
        float(1 - task_cost / (initial_distance * 50)) if task_cost is not None else None
    )
    finite = all(
        value is not None and math.isfinite(value)
        for value in (task_cost, normalized_return)
    )
    work = expected_work(population, depth, replans)
    flops = counted_flops_per_call(
        population, config["populations"][str(population)]["elites"], depth
    )
    return {
        "schema_version": 1,
        "key": {
            "row_id": entry["row_id"],
            "candidate_seed": candidate_seed,
            "cell": label,
        },
        "start": entry,
        "depth": depth,
        "population": population,
        "elites": config["populations"][str(population)]["elites"],
        "success": bool(successes[0]) if failure is None else False,
        "normalized_return": normalized_return if finite else None,
        "cumulative_task_cost": task_cost if finite else None,
        "final_task_cost": float(distances[-1]) if distances else None,
        "episode_length": len(rewards),
        "failure": failure,
        "planner_calls": recorded.calls,
        "planner_latency_ns": int(sum(row["planner_latency_ns"] for row in recorded.calls)),
        "episode_latency_ns": int(episode_latency),
        "latency_outcome_dependent": True,
        "work": work,
        "counted_flops_per_call": flops,
        "counted_flops_executed": replans * flops["counted_total"],
        "counted_flops_complete": False,
        "unclipped_mean_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "sealed"), required=True)
    parser.add_argument("--tranche", type=int, choices=(1, 2))
    parser.add_argument("--limit", type=int, help="mechanical debugging only")
    args = parser.parse_args()
    if args.phase == "sealed" and args.limit is not None:
        parser.error("--limit is forbidden for sealed execution")
    config = json.loads((ROOT / "task_d_config.json").read_text())
    cohort = json.loads((ROOT / "task_d_cohort.json").read_text())
    validate_cohort(
        cohort,
        config["execution"]["pilot_starts"],
        config["execution"]["sealed_starts"],
    )
    entries = cohort[args.phase]
    if args.phase == "sealed" and args.tranche:
        size = config["execution"]["tranche_starts"]
        entries = entries[(args.tranche - 1) * size : args.tranche * size]
    seeds = config["execution"][f"{args.phase}_candidate_seeds"]
    jobs = []
    for seed in seeds:
        orders = {row["row_id"]: cell_order(config["execution"]["order_seed"], row["row_id"], seed) for row in entries}
        for position in range(len(CELLS)):
            for row in entries:
                jobs.append((row, seed, orders[row["row_id"]][position], position))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    eval_cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    dataset = lewm_eval.get_dataset(eval_cfg, "pusht_expert_train.lance")
    process = processors(dataset, list(eval_cfg.dataset.keys_to_cache))
    models = load_models(torch.device("mps"), config)
    completed = 0
    for entry, seed, label, position in jobs:
        output = WORK / args.phase / f"row{entry['row_id']}_seed{seed}_{label}.json"
        if output.exists():
            existing = json.loads(output.read_text())
            if existing["key"] != {"row_id": entry["row_id"], "candidate_seed": seed, "cell": label}:
                raise RuntimeError("resume key mismatch")
            continue
        record = evaluate(dataset, eval_cfg, process, models[parse_cell(label)[0]], entry, label, seed, config)
        record["execution_position"] = position
        record["phase"] = args.phase
        atomic_dump(output, record)
        completed += 1
    print(json.dumps({"phase": args.phase, "scheduled": len(jobs), "newly_completed": completed}))


if __name__ == "__main__":
    main()
