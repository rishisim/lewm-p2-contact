"""Plain PyTorch fine-tuning with the upstream LeJEPA objective."""

import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from stable_worldmodel.wm.loss import SIGReg
from torch.utils.data import DataLoader, RandomSampler, Subset

from ..device import resolve_device
from ..lewm import load_lewm
from ..normalization import load_normalization, normalize_column
from ..paths import checkpoint_dir, dataset_path
from ..runs import create_run, write_metrics

HISTORY_SIZE = 3
NUM_PREDS = 1
FRAMESKIP = 5
LR = 5e-5


class Preprocess:
    """The upstream image transform and persisted column z-scores."""

    def __init__(self, stats: dict):
        image_stats = spt.data.dataset_stats.ImageNet
        self.images = spt.data.transforms.Compose(
            spt.data.transforms.ToImage(**image_stats, source="pixels", target="pixels"),
            spt.data.transforms.Resize(224, source="pixels", target="pixels"),
        )
        self.stats = stats

    def __call__(self, batch: dict) -> dict:
        batch = self.images(batch)
        for column in ("action", "proprio"):
            batch[column] = normalize_column(batch[column], self.stats, column)
        return batch


def episode_split(dataset, seed: int) -> tuple[Subset, Subset, list[int], list[int]]:
    """Split episodes before selecting windows; no episode can cross sets."""
    count = len(dataset.lengths)
    if count < 2:
        raise ValueError("At least two episodes are required for train/validation")
    shuffled = np.random.default_rng(seed).permutation(count)
    n_train = min(count - 1, max(1, round(0.9 * count)))
    train_eps = sorted(int(i) for i in shuffled[:n_train])
    val_eps = sorted(int(i) for i in shuffled[n_train:])
    train_members = set(train_eps)
    train_indices = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in train_members]
    val_indices = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep not in train_members]
    if not train_indices or not val_indices:
        raise ValueError("Train and validation must each contain at least one window")
    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_eps, val_eps


def load_windows(name: str, stats: dict, seed: int):
    path = dataset_path(name if name.endswith(".h5") else f"{name}.h5")
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = swm.data.load_dataset(
        str(path), num_steps=HISTORY_SIZE + NUM_PREDS, frameskip=FRAMESKIP,
        keys_to_load=["pixels", "action", "proprio"], transform=Preprocess(stats),
    )
    return episode_split(dataset, seed)


def lejepa_loss(model, sigreg: SIGReg, batch: dict) -> dict[str, torch.Tensor]:
    """Match third_party/le-wm/train.py:lejepa_forward exactly."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]
    pred = model.predict(emb[:, :HISTORY_SIZE], output["act_emb"][:, :HISTORY_SIZE])
    pred_loss = (pred - emb[:, NUM_PREDS:]).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    return {"pred_loss": pred_loss, "sigreg_loss": sigreg_loss,
            "loss": pred_loss + 0.09 * sigreg_loss}


def _loader(dataset, batch_size: int, workers: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=workers, generator=generator, worker_init_fn=seed_worker,
                      persistent_workers=workers > 0)


def _lr(step: int, max_steps: int) -> float:
    warmup = max(1, round(0.05 * max_steps))
    if step <= warmup:
        return LR * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))


def _save(model, optimizer, output: Path, step: int, epoch: int, batch_index: int) -> None:
    # Keep only weights.pt in the .pt format expected by load_pretrained.
    weights_tmp = output / "weights.tmp"
    torch.save(model.state_dict(), weights_tmp)
    weights_tmp.replace(output / "weights.pt")
    state_tmp = output / "trainer_state.tmp"
    torch.save({"step": step, "epoch": epoch, "batch_index": batch_index,
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, state_tmp)
    state_tmp.replace(output / "trainer_state.pth")


def _validation(model, sigreg, loader, device) -> dict:
    model.eval()
    totals = {"pred_loss": 0.0, "sigreg_loss": 0.0, "loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses = lejepa_loss(model, sigreg, batch)
            size = len(batch["pixels"])
            for key in totals:
                totals[key] += float(losses[key]) * size
            count += size
    model.train()
    return {f"val_{key}": value / count for key, value in totals.items()}


def finetune(dataset: str, name: str, init: str = "lewm-pusht", max_steps: int = 0,
             batch_size: int = 32, seed: int = 0, device: str = "auto",
             log_interval: int = 1, val_interval: int = 100,
             checkpoint_interval: int = 100, workers: int = 4) -> dict:
    if max_steps < 0 or batch_size < 1 or workers < 0 or min(log_interval, val_interval, checkpoint_interval) < 1:
        raise ValueError("Invalid training count, batch size, workers, or interval")
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Output checkpoint name must be a single directory name")
    source = checkpoint_dir(init)
    output = checkpoint_dir(name)
    if output == source:
        raise ValueError("Output checkpoint must differ from initialization")
    stats = load_normalization(source)
    selected_device = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = load_lewm(output if (output / "trainer_state.pth").exists() else source, selected_device)
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    state_file = output / "trainer_state.pth"
    if state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, batch_index = state["step"], state["epoch"], state["batch_index"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "config.json", output / "config.json")
        shutil.copyfile(source / "normalization.json", output / "normalization.json")
        step = epoch = batch_index = 0
    (output / "train_log.jsonl").touch(exist_ok=True)
    config = {"dataset": dataset, "name": name, "init": init, "max_steps": max_steps,
              "batch_size": batch_size, "seed": seed, "device": str(selected_device),
              "log_interval": log_interval, "val_interval": val_interval,
              "checkpoint_interval": checkpoint_interval, "workers": workers}
    run_dir = create_run("finetune", config)
    if max_steps:
        train, val, train_eps, val_eps = load_windows(dataset, stats, seed)
        val_loader = _loader(val, batch_size, workers, seed, False)
        sigreg = SIGReg(knots=17, num_proj=1024).to(selected_device)
        model.train()
        while step < max_steps:
            loader = _loader(train, batch_size, workers, seed + epoch, True)
            for index, batch in enumerate(loader):
                if index < batch_index:
                    continue
                start = time.perf_counter()
                batch = {k: v.to(selected_device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                losses = lejepa_loss(model, sigreg, batch)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                step += 1
                lr = _lr(step, max_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                record = {"step": step, "train_pred_loss": float(losses["pred_loss"].detach()),
                          "train_sigreg_loss": float(losses["sigreg_loss"].detach()),
                          "train_total_loss": float(losses["loss"].detach()),
                          "lr": lr, "step_seconds": time.perf_counter() - start}
                if step % val_interval == 0 or step == max_steps:
                    record.update(_validation(model, sigreg, val_loader, selected_device))
                if step % log_interval == 0 or step % val_interval == 0 or step == max_steps:
                    with (output / "train_log.jsonl").open("a") as file:
                        file.write(json.dumps(record) + "\n")
                if step % checkpoint_interval == 0 or step == max_steps:
                    _save(model, optimizer, output, step, epoch, index + 1)
                if step == max_steps:
                    break
            else:
                epoch += 1
                batch_index = 0
                continue
            break
        split = {"train_episodes": train_eps, "val_episodes": val_eps,
                 "train_windows": len(train), "val_windows": len(val)}
    else:
        _save(model, optimizer, output, step, epoch, batch_index)
        split = {}
    summary = {**config, **split, "completed_steps": step, "checkpoint": str(output),
               "run_dir": str(run_dir)}
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_metrics(run_dir, summary)
    return summary


def bench_train(dataset: str, device: str = "auto", steps: int = 20,
                workers: int = 4, batch_sizes: tuple[int, ...] = (32, 64, 128)) -> dict:
    """Measure complete optimizer steps on real windows without saving weights."""
    selected_device = resolve_device(device)
    stats = load_normalization(checkpoint_dir("lewm-pusht"))
    train, _, _, _ = load_windows(dataset, stats, 0)
    run_dir = create_run("bench-train", {"dataset": dataset, "device": str(selected_device),
                                         "steps": steps, "batch_sizes": batch_sizes})
    results = []
    for size in batch_sizes:
        if (selected_device.type == "mps" and results
                and results[-1].get("peak_memory_bytes")
                and hasattr(torch.mps, "recommended_max_memory")):
            projected = results[-1]["peak_memory_bytes"] * size / results[-1]["batch_size"]
            limit = torch.mps.recommended_max_memory()
            if projected > limit:
                results.append({"batch_size": size, "steps": 0, "status": "skipped_memory_limit",
                                "projected_memory_bytes": int(projected),
                                "recommended_max_memory_bytes": limit})
                continue
        torch.manual_seed(0)
        model = load_lewm(checkpoint_dir("lewm-pusht"), selected_device).train().requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
        sigreg = SIGReg(knots=17, num_proj=1024).to(selected_device)
        # Replacement keeps the measured batch size exact even for a tiny pilot.
        sampler = RandomSampler(train, replacement=True, num_samples=steps * size,
                                generator=torch.Generator().manual_seed(0))
        loader = DataLoader(train, batch_size=size, sampler=sampler, num_workers=workers,
                            persistent_workers=workers > 0)
        iterator = iter(loader)
        times = []
        peak_memory = 0
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(selected_device) for k, v in batch.items()}
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = lejepa_loss(model, sigreg, batch)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if selected_device.type == "mps":
                torch.mps.synchronize()
            elif selected_device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if selected_device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
                peak_memory = max(peak_memory, torch.mps.driver_allocated_memory())
            elif selected_device.type == "cuda":
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        memory = peak_memory if selected_device.type != "cpu" else None
        results.append({"batch_size": size, "steps": steps, "status": "completed",
                        "mean_step_seconds": sum(times) / len(times),
                        "peak_memory_bytes": memory, "last_loss": float(loss.detach())})
        write_metrics(run_dir, {"dataset": dataset, "device": str(selected_device),
                                "results": results, "status": "in_progress"})
        del model, optimizer, sigreg, loader, iterator
        if selected_device.type == "mps":
            torch.mps.empty_cache()
    result = {"dataset": dataset, "device": str(selected_device), "results": results,
              "run_dir": str(run_dir)}
    write_metrics(run_dir, result)
    return result
