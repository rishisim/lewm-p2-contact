"""PushT simulator and pretrained planning benchmarks."""

import time

import gymnasium as gym
import numpy as np

import stable_worldmodel as swm  # Register PushT with Gymnasium.

from .device import resolve_device
from .lewm import load_lewm
from .paths import checkpoint_dir
from .planning import config_dict, make_solver, pusht_config
from .gates import synthetic_info
from .runs import create_run, write_metrics


def bench_sim(requested_device: str = "auto") -> dict:
    """Time 2,000 PushT steps and one upstream-config CEM call."""
    device = resolve_device(requested_device)
    cfg = pusht_config(requested_device=requested_device)
    run_dir = create_run("bench-sim", config_dict(cfg))
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    rng = np.random.default_rng(cfg.seed)
    try:
        env.reset(seed=cfg.seed)
        started = time.perf_counter()
        for _ in range(2000):
            _, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
            if terminated or truncated:
                env.reset()
        elapsed = time.perf_counter() - started
    finally:
        env.close()
    model = load_lewm(checkpoint_dir("lewm-pusht"), device)
    solver = make_solver(model, cfg)
    started = time.perf_counter()
    output = solver.solve(synthetic_info(device))
    planning_s = time.perf_counter() - started
    assert output["actions"].shape == (1, 5, 10)
    result = {
        "device": str(device),
        "env_steps": 2000,
        "env_elapsed_s": round(elapsed, 4),
        "env_steps_per_s": round(2000 / elapsed, 2),
        "planning_elapsed_s": round(planning_s, 4),
        "cem_samples": int(cfg.solver.num_samples),
        "cem_iterations": int(cfg.solver.n_steps),
        "run_dir": str(run_dir),
    }
    write_metrics(run_dir, result)
    return result
