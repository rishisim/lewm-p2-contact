"""Upstream-style dataset-driven PushT smoke evaluation."""

import time

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms

from .lewm import load_lewm
from .paths import checkpoint_dir, dataset_path
from .planning import config_dict, pusht_config
from .runs import create_run, write_metrics


def smoke_eval(num_eval: int, video: bool, device: str) -> dict:
    """Evaluate LeWM from HDF5 starts and goals using upstream CEM settings."""
    path = dataset_path("pusht_expert_train.h5")
    if not path.is_file():
        raise FileNotFoundError(f"PushT dataset missing: {path}. Finish downloading and decompressing it first.")
    cfg = pusht_config(num_eval, device)
    if cfg.plan_config.horizon * cfg.plan_config.action_block > cfg.eval.eval_budget:
        raise ValueError("Planning horizon exceeds evaluation budget")
    dataset = swm.data.HDF5Dataset(path=path, keys_to_cache=list(cfg.dataset.keys_to_cache))
    index_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(index_name)
    step_idx = dataset.get_col_data("step_idx")
    episodes, inverse = np.unique(episode_idx, return_inverse=True)
    lengths = np.zeros(len(episodes), dtype=np.int64)
    np.maximum.at(lengths, inverse, step_idx + 1)
    valid = np.flatnonzero(step_idx <= (lengths - cfg.eval.goal_offset_steps - 1)[inverse])
    if len(valid) < num_eval:
        raise ValueError(f"Only {len(valid)} valid PushT starts; need {num_eval}")
    chosen = np.sort(np.random.default_rng(cfg.seed).choice(valid, size=num_eval, replace=False))
    rows = dataset.get_row_data(chosen)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        values = dataset.get_col_data(col)
        scaler = StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    transform = transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])
    model = load_lewm(checkpoint_dir("lewm-pusht"), torch.device(cfg.solver.device))
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process,
        transform={"pixels": transform, "goal": transform},
    )
    world = swm.World(**cfg.world, image_shape=(224, 224))
    world.set_policy(policy)
    run_dir = create_run("smoke-eval", config_dict(cfg))
    started = time.perf_counter()
    results = world.evaluate(
        dataset=dataset, start_steps=rows["step_idx"].tolist(),
        goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
        episodes_idx=rows[index_name].tolist(),
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        video=run_dir / "video" if video else None,
    )
    metrics = {
        "success_rate": float(results["success_rate"]),
        "episode_successes": np.asarray(results["episode_successes"]).tolist(),
        "elapsed_s": time.perf_counter() - started,
        "run_dir": str(run_dir),
    }
    write_metrics(run_dir, metrics)
    return metrics
