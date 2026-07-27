#!/usr/bin/env python3
"""Restartable Task E fixed-bank generation, scoring, and simulator labelling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = ROOT / "work/task_e"
sys.path.insert(0, str(REPO / "le-wm"))
sys.path.insert(0, str(ROOT))

import stable_worldmodel as swm
from omegaconf import OmegaConf
import eval as lewm_eval

from pusht_cem_adapter import PushTRefinedCostModel
from qualify import processors
from run_task_b import load_base
from task_c_population import PrefixComparableCEMSolver, PopulationConfig
from task_e import (
    ScheduledPushTRefinedCostModel,
    array_sha256,
    candidate_hashes,
    score_candidates,
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def target_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("Task E inherits the verified MPS execution stack")
    return torch.device("mps")


def make_info(dataset, transform, entry: dict, device: torch.device) -> dict:
    initial, goal = dataset[entry["row_id"]], dataset[entry["goal_row_id"]]
    return {
        "pixels": transform(initial["pixels"]).unsqueeze(0).to(device),
        "goal": transform(goal["pixels"]).unsqueeze(0).to(device),
        "action": torch.zeros(1, 1, 10, device=device),
    }


class CaptureModel:
    def __init__(self, model):
        self.model = model
        self.calls: list[np.ndarray] = []

    def parameters(self):
        return self.model.parameters()

    def get_cost(self, info, candidates):
        self.calls.append(candidates.detach().cpu().numpy().astype(np.float32))
        return self.model.get_cost(info, candidates)


def configure_solver(solver) -> None:
    plan = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
    solver.configure(
        action_space=gym.spaces.Box(
            low=-np.ones((1, 2), np.float32),
            high=np.ones((1, 2), np.float32),
        ),
        n_envs=1,
        config=plan,
    )


def generate_bank(info: dict, entry: dict, seed: int, device: torch.device) -> dict:
    base_model = PushTRefinedCostModel(load_base(device), None, 0).to(device).eval()
    capture = CaptureModel(base_model)
    solver = PrefixComparableCEMSolver(
        model=capture,
        population=PopulationConfig.create(300, elites=38),
        device=device,
        candidate_seed=seed,
    )
    configure_solver(solver)
    solver.solve(info, start_id=entry["row_id"], replan_index=0)
    candidates = capture.calls[-1][0]
    proposal = capture.calls[0][0, :64]
    return {
        "candidates": candidates,
        "proposal": proposal,
        "candidate_hashes": candidate_hashes(candidates),
        "tensor_sha256": array_sha256(candidates),
        "proposal_sha256": array_sha256(proposal),
        "unclipped_min": float(candidates.min()),
        "unclipped_max": float(candidates.max()),
        "mean_row_preserved": bool(np.array_equal(candidates[0], capture.calls[-1][0, 0])),
    }


def restore_env(env, entry: dict) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=entry["environment_seed"])
    unwrapped = env.unwrapped
    state = np.asarray(entry["initial_state"], np.float32).reshape(-1)
    goal_state = np.asarray(entry["goal_state"], np.float32).reshape(-1)
    unwrapped._set_state(state=state)
    unwrapped._set_goal_state(goal_state=goal_state)
    initial = np.asarray(unwrapped._get_obs(), dtype=np.float64).reshape(-1)
    goal = np.asarray(entry["goal_state"], dtype=np.float64).reshape(-1)
    return initial, goal


def label_candidate(env, entry: dict, candidate: np.ndarray) -> dict:
    initial, goal = restore_env(env, entry)
    model_visible = np.asarray(candidate, dtype=np.float32).reshape(25, 2)
    executed, states, costs, rewards = [], [], [], []
    terminated = truncated = False
    for action in model_visible:
        observation, reward, terminated, truncated, info = env.step(action.copy())
        latest = np.asarray(env.unwrapped.latest_action, np.float32).reshape(2)
        executed.append(latest)
        state = np.asarray(observation["state"], dtype=np.float64).reshape(-1)
        states.append(state)
        costs.append(float(np.linalg.norm(goal - state)))
        rewards.append(float(reward))
    executed_array = np.asarray(executed, np.float32)
    state_array = np.asarray(states, np.float64)
    return {
        "model_visible_sha256": array_sha256(model_visible),
        "reconstructed_initial_state": initial,
        "executed_sha256": array_sha256(executed_array),
        "executed_matches_model_visible": bool(np.array_equal(model_visible, executed_array)),
        "executed_actions": executed_array,
        "states": state_array,
        "costs": np.asarray(costs, np.float64),
        "rewards": np.asarray(rewards, np.float64),
        "cumulative_cost": float(np.sum(costs)),
        "terminal_cost": float(costs[-1]),
        "success": bool(terminated),
        "truncated": bool(truncated),
        "valid": bool(
            len(states) == 25
            and np.isfinite(state_array).all()
            and np.isfinite(costs).all()
            and np.array_equal(model_visible, executed_array)
        ),
    }


def load_refiner(device: torch.device):
    checkpoint = ROOT / "pusht_refiner_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    from pusht_refiner import MaskedStagewiseRefiner

    refiner = MaskedStagewiseRefiner(verified_export=True)
    refiner.load_state_dict(payload["state_dict"], strict=True)
    return refiner.to(device).eval().requires_grad_(False)


def score_bank(info: dict, candidates: np.ndarray, device: torch.device) -> dict:
    tensor = torch.from_numpy(candidates[None]).to(device)
    output = {}
    schedules = {f"d{depth}_all": (depth,) * 5 for depth in (0, 1, 2, 4)}
    for depth in (1, 2, 4):
        for transition in range(5):
            schedule = [0] * 5
            schedule[transition] = depth
            schedules[f"d{depth}_only_t{transition + 1}"] = tuple(schedule)
        for prefix in range(1, 6):
            schedules[f"d{depth}_prefix_{prefix}"] = (depth,) * prefix + (0,) * (5 - prefix)
    for history in (1, 2, 3):
        for label, schedule in schedules.items():
            if history != 3 and not label.endswith("_all"):
                continue
            base = load_base(device)
            refiner = None if max(schedule) == 0 else load_refiner(device)
            model = ScheduledPushTRefinedCostModel(
                base, refiner, schedule, history_limit=history
            ).to(device).eval().requires_grad_(False)
            costs = score_candidates(model, info, tensor)
            trace = model.last_trace
            output[f"{label}_h{history}"] = {
                "costs": costs,
                "stage_calls": trace.stage_calls if trace is not None else [0] * 4,
                "residual_norm_mean": (
                    trace.residual_norm_mean if trace is not None else [0.0] * 5
                ),
                "prediction_sha256": hashlib_sha(costs),
            }
    return output


def hashlib_sha(value: np.ndarray) -> str:
    return __import__("hashlib").sha256(
        np.ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def bank_paths(phase: str, row_id: int, seed: int) -> tuple[Path, Path]:
    stem = WORK / phase / f"row{row_id}_seed{seed}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def run_bank(
    dataset,
    transform,
    entry: dict,
    seed: int,
    phase: str,
    device: torch.device,
    *,
    label_secondary: bool,
) -> None:
    npz_path, json_path = bank_paths(phase, entry["row_id"], seed)
    if npz_path.exists() and json_path.exists():
        return
    started = time.perf_counter()
    info = make_info(dataset, transform, entry, device)
    bank = generate_bank(info, entry, seed, device)
    scores = score_bank(info, bank["candidates"], device)
    env = gym.make("swm/PushT-v1", max_episode_steps=100)
    labels = []
    proposal_labels = []
    replay_checks = []
    try:
        reset_a, _ = restore_env(env, entry)
        reset_b, _ = restore_env(env, entry)
        if not np.array_equal(reset_a, reset_b):
            raise RuntimeError("reset/state reconstruction is not exact")
        for candidate in bank["candidates"]:
            labels.append(label_candidate(env, entry, candidate))
        if label_secondary:
            for candidate in bank["proposal"]:
                proposal_labels.append(label_candidate(env, entry, candidate))
        replay_rng = np.random.default_rng(
            json.loads((ROOT / "task_e_config.json").read_text())["execution"][
                "replay_subset_seed"
            ]
            + entry["row_id"]
            + seed
        )
        replay_indices = np.sort(
            replay_rng.choice(len(bank["candidates"]), 12, replace=False)
        )
        for index in replay_indices:
            repeated = label_candidate(env, entry, bank["candidates"][index])
            original = labels[int(index)]
            replay_checks.append(
                {
                    "index": int(index),
                    "actions_exact": repeated["executed_sha256"]
                    == original["executed_sha256"],
                    "states_exact": bool(
                        np.array_equal(repeated["states"], original["states"])
                    ),
                    "costs_exact": bool(
                        np.array_equal(repeated["costs"], original["costs"])
                    ),
                }
            )
    finally:
        env.close()
    if not all(
        row["actions_exact"] and row["states_exact"] and row["costs_exact"]
        for row in replay_checks
    ):
        raise RuntimeError("deterministic replay subset mismatch")
    arrays = {
        "candidates": bank["candidates"],
        "proposal": bank["proposal"],
        "simulator_cost": np.asarray([row["cumulative_cost"] for row in labels]),
        "terminal_cost": np.asarray([row["terminal_cost"] for row in labels]),
        "valid": np.asarray([row["valid"] for row in labels], np.bool_),
        "executed_actions": np.asarray([row["executed_actions"] for row in labels]),
        "states": np.asarray([row["states"] for row in labels]),
    }
    if label_secondary:
        proposal_scores = score_bank(info, bank["proposal"], device)
        arrays["proposal_simulator_cost"] = np.asarray(
            [row["cumulative_cost"] for row in proposal_labels]
        )
        for label in ("d0_all_h3", "d1_all_h3", "d2_all_h3", "d4_all_h3"):
            arrays[f"proposal_score__{label}"] = proposal_scores[label]["costs"]
    for label, record in scores.items():
        arrays[f"score__{label}"] = record["costs"]
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(npz_path)
    metadata = {
        "schema_version": 1,
        "phase": phase,
        "row_id": entry["row_id"],
        "episode_id": entry["episode_id"],
        "candidate_seed": seed,
        "candidate_tensor_sha256": bank["tensor_sha256"],
        "proposal_sha256": bank["proposal_sha256"],
        "candidate_hashes_sha256": __import__("hashlib").sha256(
            "".join(bank["candidate_hashes"]).encode()
        ).hexdigest(),
        "candidate_count": len(bank["candidates"]),
        "valid_count": int(sum(row["valid"] for row in labels)),
        "model_visible_equals_executed_count": int(
            sum(row["executed_matches_model_visible"] for row in labels)
        ),
        "reset_reconstruction_exact": True,
        "replay_checks": replay_checks,
        "secondary_bank_labelled": label_secondary,
        "secondary_valid_count": int(
            sum(row["valid"] for row in proposal_labels)
        ),
        "unclipped_min": bank["unclipped_min"],
        "unclipped_max": bank["unclipped_max"],
        "score_records": {
            label: {key: value for key, value in record.items() if key != "costs"}
            for label, record in scores.items()
        },
        "npz_sha256": __import__("hashlib").sha256(npz_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(json_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "sealed"), required=True)
    parser.add_argument("--limit-banks", type=int)
    args = parser.parse_args()
    if args.phase == "sealed" and args.limit_banks is not None:
        parser.error("--limit-banks is forbidden for sealed execution")
    config = json.loads((ROOT / "task_e_config.json").read_text())
    manifest = json.loads((ROOT / "task_e_manifest.json").read_text())
    entries = manifest[args.phase]
    seeds = config["execution"][f"{args.phase}_candidate_seeds"]
    jobs = [
        (
            entry,
            seed,
            args.phase == "pilot"
            or entries.index(entry)
            < config["secondary_bank"]["sealed_confirmation_starts"],
        )
        for entry in entries
        for seed in seeds
    ]
    if args.limit_banks is not None:
        jobs = jobs[: args.limit_banks]
    cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    dataset = lewm_eval.get_dataset(cfg, "pusht_expert_train.lance")
    transform = lewm_eval.img_transform(cfg)
    device = target_device()
    for entry, seed, label_secondary in jobs:
        run_bank(
            dataset,
            transform,
            entry,
            seed,
            args.phase,
            device,
            label_secondary=label_secondary,
        )


if __name__ == "__main__":
    main()
