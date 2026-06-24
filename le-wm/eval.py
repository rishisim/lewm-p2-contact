import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def resolve_torch_device(requested="auto"):
    requested = str(requested or "auto").lower()
    if requested == "auto":
        return torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_index_columns(dataset):
    errors = {}
    for col_name in ("episode_idx", "ep_idx"):
        try:
            episode_idx = dataset.get_col_data(col_name)
            break
        except (KeyError, ValueError) as exc:
            errors[col_name] = exc
    else:
        raise KeyError(f"Dataset has no episode index column; tried {list(errors)}")

    step_idx = dataset.get_col_data("step_idx")
    return col_name, episode_idx, step_idx


def get_episode_index(episode_idx, step_idx):
    episodes, inverse = np.unique(episode_idx, return_inverse=True)
    lengths = np.zeros(len(episodes), dtype=np.asarray(step_idx).dtype)
    np.maximum.at(lengths, inverse, step_idx + 1)
    return episodes, inverse, lengths


def get_dataset(cfg, dataset_name):
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_cfg.pop("stats", None)
    dataset_cfg.setdefault("frameskip", 1)
    dataset_cfg.setdefault("num_steps", 1)
    dataset_cfg = {k: v for k, v in dataset_cfg.items() if v is not None}
    return swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get("cache_dir"),
        **dataset_cfg,
    )

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name, episode_idx, step_idx = get_index_columns(dataset)
    ep_indices, episode_inverse, episode_len = get_episode_index(
        episode_idx, step_idx
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        device = resolve_torch_device(cfg.solver.get("device", "auto"))
        cfg.solver.device = str(device)
        print(f"Using torch device: {device}")
        model = swm.policy.AutoCostModel(
            cfg.policy, cache_dir=swm.data.utils.get_cache_dir()
        )
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    # Map each dataset row's episode id to its max valid start step.
    max_start_per_row = max_start_idx[episode_inverse]

    # Remove rows whose goal offset would run beyond the episode.
    valid_mask = step_idx <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # Keep deterministic ascending row access across dataset backends.
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = episode_idx[random_episode_indices]
    eval_start_idx = step_idx[random_episode_indices]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
