"""Wheel CEM with trajectory-aware simulator dynamics and process workers."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing as mp

import numpy as np
import torch
from stable_worldmodel.solver.callbacks import Callback

from ..envs.pusht_peg import PushTPeg
from .conditions import role_cost

_WORKER_ENV = None


def _simulate(task):
    global _WORKER_ENV
    snapshot, condition, actions, prior_max, approach_weight = task
    if _WORKER_ENV is None:
        _WORKER_ENV = PushTPeg(with_target=False)
        _WORKER_ENV.reset(seed=0)
    env = _WORKER_ENV
    # Rebuild the physics space: body bias velocities/contact solver state are
    # not represented in snapshots and can survive repeated candidate rewinds.
    env._setup()
    env.restore_snapshot(snapshot)
    trajectory = [env._get_obs().copy()]
    for action in actions:
        trajectory.append(env.step(action)[0]["state"].copy())
    cost = role_cost(trajectory, condition)
    if condition.scoring_spec["peg"] == "preserve":
        observed = max(np.linalg.norm(s[7:9] - condition.scoring_spec["peg_start"])
                       for s in trajectory)
        cost += 2 * (max(prior_max, observed) - observed)
    xy = trajectory[-1][7:9] if condition.scoring_spec["peg"] == "target" else trajectory[-1][2:4]
    cost += approach_weight * np.linalg.norm(trajectory[-1][:2] - xy)
    return cost


class SimulatorCost:
    """One-condition cost; snapshots are replaced at each planning call.

    Candidates are (env=1, population, blocks=5, block_dim=10). All
    candidates start from the same snapshot, including velocities.
    """

    def __init__(self, condition, normalization, workers=1, approach_weight=0):
        if workers < 1:
            raise ValueError("workers must be positive")
        if approach_weight < 0:
            raise ValueError("approach weight must be nonnegative")
        self.approach_weight = approach_weight
        self.condition = condition
        self.normalization = normalization
        self.snapshot = condition.start_snapshot
        self.prior_max = 0.0
        self.workers = workers
        self.pool = (ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))
                     if workers > 1 else None)

    def get_cost(self, info_dict, action_candidates):
        if action_candidates.ndim != 4 or action_candidates.shape[0] != 1 or tuple(action_candidates.shape[2:]) != (5, 10):
            raise ValueError("expected candidate shape (1, population, 5, 10)")
        spec = self.normalization["columns"]["action"]
        actions = action_candidates.detach().cpu().numpy().reshape(-1, 25, 2)
        actions = actions * np.asarray(spec["std"]) + np.asarray(spec["mean"])
        # Physics never consumes rendered pixels; omit them from process IPC.
        snapshot = {**self.snapshot, "goal": None}
        condition = replace(self.condition, goal_image=None, start_snapshot=None)
        tasks = [(snapshot, condition, a, self.prior_max, self.approach_weight) for a in actions]
        if self.pool:
            costs = list(self.pool.map(_simulate, tasks, chunksize=max(1, len(tasks) // (self.workers * 4))))
        else:
            costs = list(map(_simulate, tasks))
        return torch.as_tensor(costs, dtype=action_candidates.dtype,
                               device=action_candidates.device).reshape(1, -1)

    criterion = get_cost

    def close(self):
        if self.pool:
            self.pool.shutdown()


class PopulationRecorder(Callback):
    """Expose the last evaluated population and elites via wheel callbacks."""

    def reset(self):
        super().reset()
        self.population = None

    def compute(self, **state):
        self.population = {k: state[k].detach().cpu().clone()
                           for k in ("candidates", "costs", "topk_candidates", "topk_vals")}
        return float(state["topk_vals"].min())
