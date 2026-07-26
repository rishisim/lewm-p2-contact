#!/usr/bin/env python3
"""Bounded, fixed-CEM PushT qualification harness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn import preprocessing
import stable_worldmodel as swm
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal
import torch


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CONFIG = ROOT / "config.json"
EVAL_PATH = REPO / "le-wm" / "eval.py"
WORK = ROOT / "work"

# The released object checkpoint was serialized with the tracked top-level
# modules `jepa` and `module`; preserve that canonical import contract.
sys.path.insert(0, str(REPO / "le-wm"))


def load_eval_module():
    spec = importlib.util.spec_from_file_location("lewm_eval", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qualification_config() -> dict:
    return json.loads(CONFIG.read_text())["planner_qualification"]


def valid_rows(dataset, goal_offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ev = load_eval_module()
    _, episodes, steps = ev.get_index_columns(dataset)
    _, inverse, lengths = ev.get_episode_index(episodes, steps)
    mask = steps <= (lengths - goal_offset - 1)[inverse]
    return np.nonzero(mask)[0], np.asarray(episodes), np.asarray(steps)


def cohort_ledgers(dataset, cfg: dict) -> dict[str, list[dict[str, int]]]:
    valid, episodes, steps = valid_rows(dataset, cfg["dataset"]["goal_offset_steps"])
    used: set[int] = set()
    ledgers = {}
    for name in ("smoke", "tuning", "heldout"):
        spec = cfg["cohorts"][name]
        rng = np.random.default_rng(spec["seed"])
        available = np.asarray([row for row in valid if int(row) not in used])
        chosen = np.sort(rng.choice(available, size=spec["starts"], replace=False))
        used.update(map(int, chosen))
        ledgers[name] = [
            {
                "row_id": int(row),
                "episode_id": int(episodes[row]),
                "start_step": int(steps[row]),
                "goal_step": int(steps[row] + cfg["dataset"]["goal_offset_steps"]),
            }
            for row in chosen
        ]
    return ledgers


def processors(dataset, keys: list[str]) -> dict:
    result = {}
    for col in keys:
        if col == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        values = dataset.get_col_data(col)
        values = values[~np.isnan(values).any(axis=1)]
        scaler.fit(values)
        result[col] = scaler
        if col != "action":
            result[f"goal_{col}"] = scaler
    return result


def make_policy(model, process, eval_cfg, planner: dict, seed: int):
    solver_cfg = OmegaConf.create(
        {
            "_target_": "stable_worldmodel.solver.CEMSolver",
            "model": "???",
            "batch_size": 1,
            "num_samples": planner["population"],
            "var_scale": 1.0,
            "n_steps": planner["cem_iterations"],
            "topk": planner["elites"],
            "device": "mps",
            "seed": seed,
        }
    )
    solver = hydra.utils.instantiate(solver_cfg, model=model)
    plan_cfg = swm.PlanConfig(
        horizon=planner["horizon"],
        receding_horizon=planner["receding_horizon"],
        action_block=5,
    )
    ev = load_eval_module()
    transforms = {"pixels": ev.img_transform(eval_cfg), "goal": ev.img_transform(eval_cfg)}
    return swm.policy.WorldModelPolicy(
        solver=solver, config=plan_cfg, process=process, transform=transforms
    )


def evaluate_start(dataset, model, process, eval_cfg, planner, entry, fixed: dict) -> dict:
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=2 * fixed["execution"]["eval_budget_env_steps"],
        image_shape=(224, 224),
    )
    policy = make_policy(model, process, eval_cfg, planner, fixed["execution"]["candidate_seed"])
    world.set_policy(policy)

    init, goal, _ = _extract_init_goal(
        dataset,
        [entry["episode_id"]],
        [entry["start_step"]],
        fixed["dataset"]["goal_offset_steps"],
    )
    callables = OmegaConf.to_container(eval_cfg.eval.callables, resolve=True)
    distances = []
    rewards = []
    successes = np.zeros(1, dtype=bool)

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    started = time.perf_counter_ns()
    world.reset(seed=init.get("seed"))
    merged = {**init, **goal}
    env_init = {key: value[0] for key, value in merged.items()}
    _apply_callables(world.envs.envs[0].unwrapped, callables, env_init)

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

    world._run(
        max_steps=fixed["execution"]["eval_budget_env_steps"],
        mode="wait",
        on_step=on_step,
    )
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    latency_ns = time.perf_counter_ns() - started
    world.envs.close()

    task_cost = float(sum(distances))
    denominator = initial_distance * fixed["execution"]["eval_budget_env_steps"]
    return {
        **entry,
        "success": bool(successes[0]),
        "steps_executed": len(rewards),
        "initial_task_cost": initial_distance,
        "final_task_cost": float(distances[-1]),
        "task_cost": task_cost,
        "normalized_return": float(1.0 - task_cost / denominator),
        "latency_ns": int(latency_ns),
    }


def summarize(records: list[dict]) -> dict:
    success = np.asarray([row["success"] for row in records], dtype=float)
    costs = np.asarray([row["task_cost"] for row in records], dtype=float)
    returns = np.asarray([row["normalized_return"] for row in records], dtype=float)
    latency = np.asarray([row["latency_ns"] for row in records], dtype=float)
    return {
        "starts": len(records),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "task_cost_mean": float(costs.mean()),
        "task_cost_median": float(np.median(costs)),
        "task_cost_std": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "normalized_return_mean": float(returns.mean()),
        "normalized_return_median": float(np.median(returns)),
        "latency_ms_median": float(np.median(latency) / 1e6),
        "latency_ms_p95": float(np.quantile(latency, 0.95) / 1e6),
        "distinct_task_costs_1e6": len(set(np.round(costs, 6))),
    }


def validate_records(records: list[dict]) -> None:
    if not records:
        raise RuntimeError("no records")
    numeric = ("initial_task_cost", "final_task_cost", "task_cost", "normalized_return")
    if not all(math.isfinite(row[key]) for row in records for key in numeric):
        raise RuntimeError("nonfinite metric")
    if not all(row["steps_executed"] > 0 and row["latency_ns"] > 0 for row in records):
        raise RuntimeError("mechanically invalid episode")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("ledger", "smoke", "tuning", "heldout"), required=True)
    parser.add_argument("--planner-index", type=int)
    parser.add_argument("--planner-file", type=Path)
    args = parser.parse_args()

    fixed = qualification_config()
    checkpoint = Path(swm.data.utils.get_cache_dir()) / "pusht/lewm_object.ckpt"
    actual_hash = sha256(checkpoint)
    if actual_hash != fixed["checkpoint"]["object_checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {actual_hash}")

    eval_cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    eval_cfg.eval.dataset_name = fixed["dataset"]["name"]
    eval_cfg.eval.img_size = 224
    dataset = load_eval_module().get_dataset(eval_cfg, fixed["dataset"]["name"])
    ledgers = cohort_ledgers(dataset, fixed)
    WORK.mkdir(parents=True, exist_ok=True)
    ledger_path = WORK / "start_ledgers.json"
    ledger_path.write_text(json.dumps(ledgers, indent=2) + "\n")
    if args.cohort == "ledger":
        print(json.dumps({"ledger": str(ledger_path), "counts": {k: len(v) for k, v in ledgers.items()}}))
        return

    if args.cohort == "smoke":
        planner = fixed["smoke_configuration"]
        tag = "smoke"
    elif args.cohort == "tuning":
        if args.planner_index is None:
            parser.error("--planner-index is required for tuning")
        planner = fixed["bounded_tuning_grid"][args.planner_index]
        tag = f"tuning_{args.planner_index}"
    else:
        if args.planner_file is None:
            parser.error("--planner-file is required for heldout")
        planner = json.loads(args.planner_file.read_text())
        tag = "heldout"

    device = swm.resolve_torch_device("mps") if hasattr(swm, "resolve_torch_device") else torch.device("mps")
    model = swm.policy.AutoCostModel("pusht/lewm", cache_dir=swm.data.utils.get_cache_dir())
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    process = processors(dataset, list(eval_cfg.dataset.keys_to_cache))

    records = [
        evaluate_start(dataset, model, process, eval_cfg, planner, row, fixed)
        for row in ledgers[args.cohort]
    ]
    validate_records(records)
    payload = {
        "cohort": args.cohort,
        "planner": planner,
        "summary": summarize(records),
        "records": records,
        "provenance": {
            "checkpoint_sha256": actual_hash,
            "dataset": str(Path(swm.data.utils.get_cache_dir()) / "datasets" / fixed["dataset"]["name"]),
            "python": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
    }
    output = WORK / f"{tag}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "summary": payload["summary"]}))


if __name__ == "__main__":
    main()
