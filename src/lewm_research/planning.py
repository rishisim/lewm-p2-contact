"""Compose the pinned upstream PushT planning configuration."""

from pathlib import Path

import hydra
from gymnasium.spaces import Box
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import stable_worldmodel as swm

from .device import resolve_device


def pusht_config(num_eval: int = 50, requested_device: str = "auto") -> DictConfig:
    """Compose upstream evaluation YAML with local device and HDF5 overrides."""
    config_dir = Path(__file__).resolve().parents[2] / "third_party/le-wm/config/eval"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="pusht", overrides=[f"eval.num_eval={num_eval}"])
    cfg.policy = "lewm-pusht"
    cfg.eval.dataset_name = "pusht_expert_train.h5"
    cfg.solver.device = str(resolve_device(requested_device))
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    return cfg


def make_solver(model: torch.nn.Module, cfg: DictConfig, n_envs: int = 1) -> swm.solver.CEMSolver:
    """Instantiate upstream CEM settings for the 0.1.1 API."""
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    action_space = Box(-1.0, 1.0, shape=(n_envs, 2), dtype=np.float32)
    solver.configure(action_space=action_space, n_envs=n_envs, config=swm.PlanConfig(**cfg.plan_config))
    return solver


def config_dict(cfg: DictConfig) -> dict:
    """Return a JSON-compatible resolved configuration."""
    return OmegaConf.to_container(cfg, resolve=True)
