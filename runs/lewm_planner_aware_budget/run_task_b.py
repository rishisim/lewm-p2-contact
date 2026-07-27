#!/usr/bin/env python3
"""Bounded Task B data, training, and offline validation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import shutil

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = ROOT / "work/task_b"
OLD = Path(
    "/Users/rishisim/Documents/research/lewm-p2-contact/"
    "runs/lewm_pusht_replication_pilot"
)
sys.path.insert(0, str(REPO / "le-wm"))
sys.path.insert(0, str(ROOT))

import stable_worldmodel as swm
from omegaconf import OmegaConf
import eval as lewm_eval
from pusht_refiner import (
    ACTION_DIM,
    LATENT_DIM,
    MAX_HISTORY,
    MaskedStagewiseRefiner,
    prefix_mask,
)
from pusht_cem_adapter import PushTRefinedCostModel

SOURCE = {"weak": 0, "expert": 1, "offpolicy": 2}
ROLE_NAMES = {value: key for key, value in SOURCE.items()}
SPLITS = {
    "fit": {"expert": 20, "offpolicy": 20},
    "selection": {"expert": 10, "offpolicy": 10},
    "evaluation": {"expert": 10, "offpolicy": 10},
}


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_base(target: torch.device) -> nn.Module:
    checkpoint = Path(swm.data.utils.get_cache_dir()) / "pusht/lewm_object.ckpt"
    if sha256(checkpoint) != (
        "0be0611227823a00b2f9acac299375a013bb23cd2a33b8ffa94a3bdb5e19b737"
    ):
        raise RuntimeError("Task A base checkpoint hash mismatch")
    model = swm.policy.AutoCostModel(
        "pusht/lewm", cache_dir=swm.data.utils.get_cache_dir()
    )
    return model.to(target).eval().requires_grad_(False)


def encode(model: nn.Module, pixels: torch.Tensor, target: torch.device) -> np.ndarray:
    cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    transform = lewm_eval.img_transform(cfg)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 64):
            value = transform(pixels[start : start + 64]).to(target)
            chunks.append(
                model.encode({"pixels": value.unsqueeze(0)})["emb"]
                .squeeze(0)
                .detach()
                .cpu()
            )
    result = torch.cat(chunks).numpy().astype(np.float32)
    if result.shape[1] != LATENT_DIM or not np.isfinite(result).all():
        raise RuntimeError("invalid frozen-base encoding")
    return result


def predict_base(
    model: nn.Module,
    history: np.ndarray,
    actions: np.ndarray,
    lengths: np.ndarray,
    target: torch.device,
) -> np.ndarray:
    output = np.empty((len(history), LATENT_DIM), dtype=np.float32)
    with torch.inference_mode():
        for length in (1, 2, 3):
            selected = np.flatnonzero(lengths == length)
            for start in range(0, len(selected), 512):
                rows = selected[start : start + 512]
                h = torch.from_numpy(history[rows, -length:]).to(target)
                a = torch.from_numpy(actions[rows, -length:]).to(target)
                predicted = model.predict(h, model.action_encoder(a))[:, -1]
                output[rows] = predicted.detach().cpu().numpy()
    return output


def examples_from_latents(
    model: nn.Module,
    latents: list[np.ndarray],
    action_blocks: list[np.ndarray],
    *,
    source: int,
    episode_offset: int,
    target: torch.device,
) -> dict[str, np.ndarray]:
    histories, actions, masks, bases, targets = [], [], [], [], []
    episode_ids, lengths, sources = [], [], []
    for episode, (embedding, blocks) in enumerate(zip(latents, action_blocks)):
        usable = min(len(blocks), len(embedding) - 1)
        for target_step in range(1, usable + 1):
            for length in range(1, min(MAX_HISTORY, target_step) + 1):
                history = np.zeros((MAX_HISTORY, LATENT_DIM), dtype=np.float32)
                action = np.zeros((MAX_HISTORY, ACTION_DIM), dtype=np.float32)
                history[-length:] = embedding[target_step - length : target_step]
                action[-length:] = blocks[target_step - length : target_step]
                histories.append(history)
                actions.append(action)
                masks.append([False] * (MAX_HISTORY - length) + [True] * length)
                targets.append(embedding[target_step])
                episode_ids.append(episode_offset + episode)
                lengths.append(length)
                sources.append(source)
    result = {
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "source": np.asarray(sources, dtype=np.int8),
        "length": np.asarray(lengths, dtype=np.int8),
        "history": np.asarray(histories, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "mask": np.asarray(masks, dtype=np.bool_),
        "target": np.asarray(targets, dtype=np.float32),
    }
    in_range = np.all(
        np.where(result["mask"][..., None], np.abs(result["actions"]) <= 1.0, True),
        axis=(1, 2),
    )
    result = {name: value[in_range] for name, value in result.items()}
    result["base"] = predict_base(
        model, result["history"], result["actions"], result["length"], target
    )
    return result


def weak_examples(model: nn.Module, split: str, target: torch.device) -> dict[str, np.ndarray]:
    with np.load(OLD / f"data/{split}.npz", allow_pickle=False) as stored:
        old = {name: stored[name].copy() for name in stored.files}
    histories, actions, masks, targets, episode_ids, lengths = [], [], [], [], [], []
    for row in range(len(old["target"])):
        for length in (1, 2, 3):
            history = np.zeros((3, LATENT_DIM), np.float32)
            action = np.zeros((3, ACTION_DIM), np.float32)
            history[-length:] = old["history"][row, -length:]
            action[-length:] = old["actions"][row, -length:]
            histories.append(history)
            actions.append(action)
            masks.append([False] * (3 - length) + [True] * length)
            targets.append(old["target"][row])
            episode_ids.append(int(old["episode_id"][row]) + 1_000_000)
            lengths.append(length)
    result = {
        "episode_id": np.asarray(episode_ids, np.int32),
        "source": np.full(len(targets), SOURCE["weak"], np.int8),
        "length": np.asarray(lengths, np.int8),
        "history": np.asarray(histories, np.float32),
        "actions": np.asarray(actions, np.float32),
        "mask": np.asarray(masks, np.bool_),
        "target": np.asarray(targets, np.float32),
    }
    result["base"] = predict_base(
        model, result["history"], result["actions"], result["length"], target
    )
    return result


def expert_episodes(split: str, count: int) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    dataset = lewm_eval.get_dataset(cfg, "pusht_expert_train.lance")
    _, episode, step = lewm_eval.get_index_columns(dataset)
    unique, counts = np.unique(episode, return_counts=True)
    eligible = unique[counts >= 101]
    task_a_episodes = set()
    fixed = json.loads((ROOT / "config.json").read_text())["planner_qualification"]
    valid = np.flatnonzero(step <= 100 - fixed["dataset"]["goal_offset_steps"] - 1)
    used: set[int] = set()
    for cohort in ("smoke", "tuning", "heldout"):
        spec = fixed["cohorts"][cohort]
        rng_a = np.random.default_rng(spec["seed"])
        available = np.asarray([row for row in valid if int(row) not in used])
        chosen_rows = rng_a.choice(available, size=spec["starts"], replace=False)
        used.update(map(int, chosen_rows))
        task_a_episodes.update(map(int, episode[chosen_rows]))
    eligible = np.asarray(
        [identifier for identifier in eligible if int(identifier) not in task_a_episodes]
    )
    rng = np.random.default_rng(26072622 + ("fit", "selection", "evaluation").index(split))
    chosen = rng.choice(eligible, count, replace=False)
    pixels, blocks = [], []
    for identifier in chosen:
        indices = np.flatnonzero(episode == identifier)[:101]
        items = dataset.__getitems__(indices.tolist())
        frames = torch.cat([items[index]["pixels"] for index in range(0, 101, 5)])
        raw = torch.cat([item["action"] for item in items[:100]]).numpy()
        pixels.append(frames)
        blocks.append(raw.reshape(20, ACTION_DIM).astype(np.float32))
    return pixels, blocks


def offpolicy_episodes(
    split: str, count: int
) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    seed0 = 7_100_000 + 10_000 * ("fit", "selection", "evaluation").index(split)
    pixels, blocks = [], []
    for episode in range(count):
        seed = seed0 + episode
        rng = np.random.default_rng(seed)
        world = swm.World(
            "swm/PushT-v1",
            num_envs=1,
            image_shape=(224, 224),
            max_episode_steps=101,
            render_mode="rgb_array",
        )
        world.reset(seed=[int(seed)])
        frames = [torch.from_numpy(np.asarray(world.infos["pixels"][0, 0])).permute(2, 0, 1)]
        raw = []
        try:
            for step in range(100):
                if step % 10 < 5:
                    center = rng.uniform(-0.6, 0.6, size=2)
                    action = np.clip(center + rng.normal(0, 0.35, 2), -1, 1)
                else:
                    action = rng.uniform(-1, 1, 2)
                batch = np.asarray(action, dtype=np.float32)[None]
                _, _, _, _, info = world.envs.step(batch)
                world.infos = info
                raw.append(batch[0])
                if (step + 1) % 5 == 0:
                    frames.append(
                        torch.from_numpy(np.asarray(info["pixels"][0, 0])).permute(2, 0, 1)
                    )
        finally:
            world.envs.close()
        pixels.append(torch.stack(frames))
        blocks.append(np.asarray(raw, np.float32).reshape(20, ACTION_DIM))
    return pixels, blocks


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def validate_data(data: dict[str, np.ndarray], split: str) -> dict:
    expected_masks = prefix_mask(torch.from_numpy(data["length"].astype(np.int64))).numpy()
    if not np.array_equal(data["mask"], expected_masks):
        raise RuntimeError("mask/length disagreement")
    if not all(np.isfinite(value).all() for name, value in data.items() if name != "mask"):
        raise RuntimeError("nonfinite data")
    valid_actions = data["actions"][data["mask"]]
    if valid_actions.min() < -1 or valid_actions.max() > 1:
        raise RuntimeError("canonical action range failure")
    cells = {}
    for source, name in ROLE_NAMES.items():
        for length in (1, 2, 3):
            cells[f"{name}_length_{length}"] = int(
                np.sum((data["source"] == source) & (data["length"] == length))
            )
    if min(cells.values()) < 100:
        raise RuntimeError(f"{split} readiness cell below 100: {cells}")
    return {"rows": len(data["target"]), "cells": cells}


def prepare() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    target = device()
    model = load_base(target)
    manifest = {"base_device": str(target), "splits": {}}
    for split in ("fit", "selection", "evaluation"):
        output = WORK / f"{split}.npz"
        if output.exists():
            continue
        weak = weak_examples(model, split, target)
        expert_pixels, expert_actions = expert_episodes(split, SPLITS[split]["expert"])
        expert_latents = [
            encode(model, frames, target) for frames in expert_pixels
        ]
        expert = examples_from_latents(
            model, expert_latents, expert_actions, source=SOURCE["expert"],
            episode_offset=2_000_000, target=target,
        )
        random_pixels, random_actions = offpolicy_episodes(
            split, SPLITS[split]["offpolicy"]
        )
        random_latents = [
            encode(model, frames, target) for frames in random_pixels
        ]
        offpolicy = examples_from_latents(
            model, random_latents, random_actions, source=SOURCE["offpolicy"],
            episode_offset=3_000_000, target=target,
        )
        combined = concatenate([weak, expert, offpolicy])
        report = validate_data(combined, split)
        np.savez_compressed(output, **combined)
        manifest["splits"][split] = {
            **report, "sha256": sha256(output),
            "episodes_by_source": {
                "weak": int(len(np.unique(weak["episode_id"]))),
                "expert": SPLITS[split]["expert"],
                "offpolicy": SPLITS[split]["offpolicy"],
            },
        }
    dump(WORK / "data_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def load_split(split: str) -> dict[str, np.ndarray]:
    with np.load(WORK / f"{split}.npz", allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def dense(
    model: MaskedStagewiseRefiner, data: dict[str, np.ndarray], target: torch.device
) -> np.ndarray:
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["target"]), 512):
            args = [
                torch.from_numpy(data[name][start : start + 512]).to(target)
                for name in ("history", "actions", "mask", "base")
            ]
            exits = []
            for depth in (1, 2, 4):
                exits.append(model(*args[:3], args[-1], depth))
            rows.append(torch.stack(exits, 1).cpu().numpy())
    return np.concatenate(rows)


def fit_whitening(targets: np.ndarray) -> np.ndarray:
    centered = targets.astype(np.float64) - targets.mean(0)
    covariance = np.einsum("ni,nj->ij", centered, centered, optimize=True)
    covariance /= max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, max(values.max() * 1e-3, 1e-10))
    return np.einsum(
        "di,ji->dj", vectors * (1 / np.sqrt(values)), vectors, optimize=True
    )


def train() -> None:
    fit = load_split("fit")
    selection = load_split("selection")
    target = device()
    torch.manual_seed(26072621)
    model = MaskedStagewiseRefiner(verified_export=True).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    whitening = fit_whitening(fit["target"])
    whitening_tensor = torch.from_numpy(whitening.astype(np.float32)).to(target)
    rng = np.random.default_rng(26072623)
    best, best_epoch, state, stale = math.inf, -1, None, 0
    history = []
    cell = fit["source"].astype(np.int64) * 3 + fit["length"].astype(np.int64) - 1
    counts = np.bincount(cell, minlength=9)
    row_weight = (len(fit["target"]) / (9 * counts[cell])).astype(np.float32)
    for epoch in range(200):
        model.train()
        order = rng.permutation(len(fit["target"]))
        for start in range(0, len(order), 256):
            rows = order[start : start + 256]
            values = [
                torch.from_numpy(fit[name][rows]).to(target)
                for name in ("history", "actions", "mask", "base", "target")
            ]
            weights = torch.from_numpy(row_weight[rows]).to(target)
            losses = []
            predictions = []
            for depth in (1, 2, 4):
                prediction = model(
                    values[0], values[1], values[2], values[3], depth
                )
                predictions.append(prediction)
                error = prediction - values[4]
                raw_per_row = error.square().mean(1)
                white_per_row = (error @ whitening_tensor).square().mean(1)
                losses.append(((raw_per_row + 0.1 * white_per_row) * weights).mean())
            depth2_error = predictions[1] - values[4]
            depth4_error = predictions[2] - values[4]
            depth2_raw = depth2_error.square().mean(1)
            depth4_raw = depth4_error.square().mean(1)
            depth2_white = (depth2_error @ whitening_tensor).square().mean(1)
            depth4_white = (depth4_error @ whitening_tensor).square().mean(1)
            monotonic = (
                torch.relu(depth4_raw - depth2_raw)
                + 0.1 * torch.relu(depth4_white - depth2_white)
            )
            loss = torch.stack(losses).mean() + 5.0 * (monotonic * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        outputs = dense(model, selection, target)
        base_loss = np.square(selection["base"] - selection["target"]).mean(1)
        exit_loss = np.square(outputs - selection["target"][:, None]).mean(2)
        selection_white = np.square(
            np.einsum(
                "nkd,df->nkf",
                outputs.astype(np.float64) - selection["target"][:, None],
                whitening,
                optimize=True,
            )
        ).mean(2)
        cell_scores, feasible = [], True
        for source in SOURCE.values():
            for length in (1, 2, 3):
                selected = (
                    (selection["source"] == source)
                    & (selection["length"] == length)
                )
                denominator = float(base_loss[selected].mean())
                cell_scores.append(float(exit_loss[selected, 2].mean()))
                feasible &= all(
                    float(exit_loss[selected, column].mean()) / denominator - 1
                    <= 0.01
                    for column in (0, 1)
                )
                feasible &= (
                    float(exit_loss[selected, 2].mean())
                    <= 1.01 * float(exit_loss[selected, 1].mean())
                    and float(selection_white[selected, 2].mean())
                    <= 1.01 * float(selection_white[selected, 1].mean())
                )
        score = float(np.mean(cell_scores))
        history.append({
            "epoch": epoch + 1,
            "selection_depth4_cell_balanced_raw_mse": score,
            "depth1_depth2_cell_noninferiority": bool(feasible),
        })
        if feasible and score < best - 1e-9:
            best, best_epoch = score, epoch + 1
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (epoch + 1) % 10 == 0 or stale >= 35:
            print(
                f"epoch={epoch+1} selection={score:.8g} feasible={feasible} "
                f"best={best:.8g}",
                flush=True,
            )
        if stale >= 35:
            break
    if state is None:
        raise RuntimeError("no finite selected state")
    checkpoint = WORK / "refiner.pt"
    torch.save(
        {
            "state_dict": state,
            "architecture": {
                "latent_dim": 192, "action_dim": 10, "history": 3,
                "hidden": 256, "iteration_dim": 16, "depths": [1, 2, 4],
                "mask": "left_padding_valid_suffix",
                "canonical_action": "PushT environment identity [-1,1]",
            },
            "training_seed": 26072621,
            "best_epoch": best_epoch,
        },
        checkpoint,
    )
    np.savez_compressed(WORK / "fit_constants.npz", whitening=whitening)
    dump(
        WORK / "training.json",
        {
            "best_epoch": best_epoch, "selection_depth4_raw_mse": best,
            "checkpoint_sha256": sha256(checkpoint), "history": history,
        },
    )


def metrics_for(data: dict[str, np.ndarray], exits: np.ndarray, whitening: np.ndarray) -> dict:
    all_predictions = np.concatenate([data["base"][:, None], exits], axis=1)
    difference = all_predictions.astype(np.float64) - data["target"][:, None]
    raw = np.square(difference).mean(2)
    white = np.square(
        np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    ).mean(2)
    cells = {}
    for source, name in ROLE_NAMES.items():
        for length in (1, 2, 3):
            mask = (data["source"] == source) & (data["length"] == length)
            cells[f"{name}_length_{length}"] = {
                "rows": int(mask.sum()),
                "raw_mse": raw[mask].mean(0).tolist(),
                "whitened_mse": white[mask].mean(0).tolist(),
                "depth4_raw_relative": float(raw[mask, 3].mean() / raw[mask, 0].mean() - 1),
                "depth4_white_relative": float(white[mask, 3].mean() / white[mask, 0].mean() - 1),
            }
    return {"cells": cells}


def evaluate() -> None:
    checkpoint = torch.load(WORK / "refiner.pt", map_location="cpu", weights_only=True)
    target = device()
    model = MaskedStagewiseRefiner(verified_export=True).to(target)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    data = load_split("evaluation")
    exits = dense(model, data, target)
    with np.load(WORK / "fit_constants.npz", allow_pickle=False) as stored:
        whitening = stored["whitening"]
    result = metrics_for(data, exits, whitening)
    passing = sum(
        cell["depth4_raw_relative"] <= 0.01
        and cell["depth4_white_relative"] <= 0.01
        for cell in result["cells"].values()
    )
    improved = sum(
        cell["depth4_raw_relative"] <= -0.01
        and cell["depth4_white_relative"] <= -0.01
        for cell in result["cells"].values()
    )
    result["gate"] = {
        "noninferior_cells": passing,
        "improved_cells": improved,
        "required_cells": 9,
        "required_improved_cells": 5,
        "point_estimate_screen_passed": passing == 9 and improved >= 5,
        "bootstrap_upper_bounds_pending": True,
    }
    np.savez_compressed(
        WORK / "evaluation_outputs.npz",
        episode_id=data["episode_id"], source=data["source"], length=data["length"],
        target=data["target"], base=data["base"], exits=exits,
    )
    dump(WORK / "offline_results.json", result)
    print(json.dumps(result["gate"], indent=2))


def export() -> None:
    results = json.loads((WORK / "offline_results.json").read_text())
    if not results["gate"]["point_estimate_screen_passed"]:
        raise RuntimeError("offline point-estimate gate did not pass")
    data = load_split("evaluation")
    with np.load(WORK / "evaluation_outputs.npz", allow_pickle=False) as stored:
        outputs = {name: stored[name].copy() for name in stored.files}
    checkpoint_source = WORK / "refiner.pt"
    checkpoint = ROOT / "pusht_refiner_checkpoint.pt"
    shutil.copyfile(checkpoint_source, checkpoint)
    checkpoint_hash = sha256(checkpoint)

    rng = np.random.default_rng(26072625)
    bootstrap = {}
    with np.load(WORK / "fit_constants.npz", allow_pickle=False) as stored:
        whitening = stored["whitening"]
    prediction = np.concatenate([outputs["base"][:, None], outputs["exits"]], axis=1)
    difference = prediction.astype(np.float64) - outputs["target"][:, None]
    raw = np.square(difference).mean(2)
    white = np.square(
        np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    ).mean(2)
    for source, name in ROLE_NAMES.items():
        for length in (1, 2, 3):
            selected = (outputs["source"] == source) & (outputs["length"] == length)
            episodes = np.unique(outputs["episode_id"][selected])
            estimates_raw, estimates_white = [], []
            for _ in range(2000):
                chosen = rng.choice(episodes, len(episodes), replace=True)
                rows = np.concatenate(
                    [np.flatnonzero(selected & (outputs["episode_id"] == item)) for item in chosen]
                )
                estimates_raw.append(raw[rows, 3].mean() / raw[rows, 0].mean() - 1)
                estimates_white.append(white[rows, 3].mean() / white[rows, 0].mean() - 1)
            bootstrap[f"{name}_length_{length}"] = {
                "raw_relative_upper_95": float(np.quantile(estimates_raw, 0.95)),
                "white_relative_upper_95": float(np.quantile(estimates_white, 0.95)),
            }
    bootstrap_pass = all(
        cell["raw_relative_upper_95"] <= 0.01
        and cell["white_relative_upper_95"] <= 0.01
        for cell in bootstrap.values()
    )

    target = device()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MaskedStagewiseRefiner(verified_export=True).to(target)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    reference_rows = []
    for source in SOURCE.values():
        for length in (1, 2, 3):
            reference_rows.append(
                int(np.flatnonzero((data["source"] == source) & (data["length"] == length))[0])
            )
    indices = np.asarray(reference_rows)
    reference_exits = dense(
        model, {name: value[indices] for name, value in data.items()}, target
    )
    np.savez_compressed(
        ROOT / "pusht_refiner_references.npz",
        history=data["history"][indices],
        actions=data["actions"][indices],
        mask=data["mask"][indices],
        base=data["base"][indices],
        depth_1=reference_exits[:, 0],
        depth_2=reference_exits[:, 1],
        depth_4=reference_exits[:, 2],
        source=data["source"][indices],
        length=data["length"][indices],
    )

    latency = {}
    batch = {name: value[:512] for name, value in data.items()}
    for depth in (0, 1, 2, 4):
        timings = []
        for _ in range(30):
            arguments = [
                torch.from_numpy(batch[name]).to(target)
                for name in ("history", "actions", "mask", "base")
            ]
            if target.type == "mps":
                torch.mps.synchronize()
            started = time.perf_counter_ns()
            with torch.inference_mode():
                model(*arguments, depth)
            if target.type == "mps":
                torch.mps.synchronize()
            timings.append(time.perf_counter_ns() - started)
        latency[str(depth)] = {
            "rows": 512,
            "median_ns": int(np.median(timings)),
            "p95_ns": int(np.quantile(timings, 0.95)),
        }
    manifest = {
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "reference_sha256": sha256(ROOT / "pusht_refiner_references.npz"),
        "cross_device_reference_atol": 3e-7,
        "cross_device_reference_rtol": 2e-5,
        "supported_depths": [0, 1, 2, 4],
        "bootstrap": bootstrap,
        "bootstrap_gate_passed": bootstrap_pass,
        "latency": latency,
        "operation_ledger": {
            "one_transition_batch_at_depth_d": {
                "base_calls": 1,
                "base_rows": "batch_rows",
                "stage_calls": "one for each stage 1..d",
                "stage_rows": "batch_rows for each stage 1..d",
            }
        },
    }
    results["gate"]["bootstrap_upper_bounds_pending"] = False
    results["gate"]["bootstrap_gate_passed"] = bootstrap_pass
    dump(ROOT / "pusht_refiner_manifest.json", manifest)
    dump(
        ROOT / "task_b_results.json",
        {
            "decision": "b3_passed_b4_integration_pending",
            "data": json.loads((WORK / "data_manifest.json").read_text()),
            "training": json.loads((WORK / "training.json").read_text()),
            "offline": results,
            "export": manifest,
        },
    )
    if not bootstrap_pass:
        raise RuntimeError("episode-bootstrap noninferiority gate failed")
    print(json.dumps(manifest, indent=2))


def smoke() -> None:
    from gymnasium.spaces import Box
    from stable_worldmodel.solver import CEMSolver

    manifest = json.loads((ROOT / "pusht_refiner_manifest.json").read_text())
    target = device()
    cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    dataset = lewm_eval.get_dataset(cfg, "pusht_expert_train.lance")
    transform = lewm_eval.img_transform(cfg)
    initial = dataset[0]
    goal = dataset[25]
    info = {
        "pixels": transform(initial["pixels"]).unsqueeze(0),
        "goal": transform(goal["pixels"]).unsqueeze(0),
        "action": torch.zeros(1, 1, ACTION_DIM),
    }
    records = {}
    for depth in (0, 1, 2, 4):
        torch.manual_seed(26072610)
        base = load_base(target)
        if depth == 0:
            model = PushTRefinedCostModel(base, None, 0).to(target)
        else:
            model = PushTRefinedCostModel.from_checkpoint(
                base,
                ROOT / "pusht_refiner_checkpoint.pt",
                expected_sha256=manifest["checkpoint_sha256"],
                refinement_depth=depth,
            ).to(target)
        model.eval().requires_grad_(False)
        solver = CEMSolver(
            model=model, batch_size=1, num_samples=32, n_steps=3, topk=4,
            device=target, seed=26072610,
        )
        solver.configure(
            action_space=Box(-1.0, 1.0, shape=(1, 2), dtype=np.float32),
            n_envs=1,
            config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        )
        generator = torch.Generator(device=target).manual_seed(26072610)
        probe_candidates = torch.randn(
            1, 32, 5, ACTION_DIM, generator=generator, device=target
        )
        probe_info = {
            key: value.unsqueeze(1).expand(1, 32, *value.shape[1:])
            for key, value in info.items()
        }
        with torch.inference_mode():
            probe_costs = model.get_cost(probe_info, probe_candidates)
        started = time.perf_counter_ns()
        solved = solver.solve(info)
        elapsed = time.perf_counter_ns() - started
        records[str(depth)] = {
            "selected_action": solved["actions"].detach().cpu().numpy().tolist(),
            "final_elite_cost": solved["costs"],
            "probe_costs": probe_costs.detach().cpu().numpy().tolist(),
            "latency_ns": elapsed,
            "ledger": {
                "base_calls": model.ledger.base_calls,
                "base_rows": model.ledger.base_rows,
                "stage_calls": model.ledger.stage_calls,
                "stage_rows": model.ledger.stage_rows,
            },
        }
    base_cost = np.asarray(records["0"]["probe_costs"])
    effects = {}
    for depth in ("1", "2", "4"):
        cost = np.asarray(records[depth]["probe_costs"])
        effects[depth] = {
            "costs_changed": bool(np.any(cost != base_cost)),
            "selected_action_changed": records[depth]["selected_action"] != records["0"]["selected_action"],
            "ranking_changed": bool(np.any(np.argsort(cost) != np.argsort(base_cost))),
        }
    smoke_result = {"records": records, "effects": effects}
    dump(ROOT / "task_b_cem_smoke.json", smoke_result)
    if not any(value["costs_changed"] or value["ranking_changed"] for value in effects.values()):
        raise RuntimeError("refinement had no measurable CEM smoke effect")
    result_path = ROOT / "task_b_results.json"
    result = json.loads(result_path.read_text())
    result["decision"] = "task_b_complete_integration_valid_no_control_effect_claim"
    result["cem_smoke"] = smoke_result
    dump(result_path, result)
    print(json.dumps(effects, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "train", "evaluate", "export", "smoke"))
    phase = parser.parse_args().phase
    globals()[phase]()


if __name__ == "__main__":
    main()
