"""Plain PyTorch fine-tuning with the upstream LeJEPA objective."""

import contextlib
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra.utils import instantiate
from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
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


def seed_worker(worker_id: int) -> None:
    """Picklable worker initializer for macOS spawn."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(dataset, batch_size: int, workers: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                        num_workers=workers, generator=generator, worker_init_fn=seed_worker,
                        persistent_workers=workers > 0)
    if shuffle and not len(loader):
        raise ValueError("Zero training batches: reduce batch_size or supply more windows")
    return loader


def _scheduler(optimizer, schedule_steps: int):
    if not schedule_steps:
        return None
    return LinearWarmupCosineAnnealingLR(
        optimizer, warmup_steps=max(1, int(0.01 * schedule_steps)),
        max_steps=schedule_steps, warmup_start_lr=0.0, eta_min=0.0,
    )


def _fingerprint(path: Path, cache: bool = False) -> dict:
    """Hash file content in bounded chunks; dataset caches are local hints only."""
    stat = path.stat()
    key = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache_path = path.with_name(path.name + ".sha256")
    if cache:
        try:
            saved = json.loads(cache_path.read_text())
            if not isinstance(saved, dict):
                raise ValueError("Invalid fingerprint cache")
            digest = saved["sha256"]
            if (all(saved.get(k) == v for k, v in key.items())
                    and isinstance(digest, str) and len(digest) == 64
                    and all(c in "0123456789abcdef" for c in digest)):
                return {"size": stat.st_size, "sha256": digest}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError(f"Input changed while fingerprinting: {path}")
    digest = hasher.hexdigest()
    if cache:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                             prefix=cache_path.name + ".", delete=False) as file:
                temporary = Path(file.name)
                json.dump({**key, "sha256": digest}, file)
            temporary.replace(cache_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"size": stat.st_size, "sha256": digest}


def _validate_resume(state: dict, metadata: dict) -> None:
    saved = state.get("metadata")
    if (state.get("version") != 2 or not isinstance(saved, dict)
            or any(not isinstance(saved.get(key), dict)
                   or not {"name", "size", "sha256"} <= saved[key].keys()
                   for key in ("dataset", "init"))):
        raise ValueError("Legacy/incomplete trainer checkpoint: portable resume requires "
                         "version 2 content fingerprints; start a new run")
    mismatches = []
    for key, value in metadata.items():
        if key in {"dataset", "init"}:
            # Names describe the inputs; only content determines identity.
            matches = isinstance(value, dict) and all(
                saved[key].get(field) == value.get(field) for field in ("size", "sha256"))
        elif key == "from_scratch":
            matches = saved.get(key, False) == value
        else:
            matches = saved.get(key) == value
        if not matches:
            mismatches.append(key)
    if mismatches:
        raise ValueError("Resume metadata mismatch: " + ", ".join(mismatches))


def _rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
             "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _atomic_save(state: dict, target: Path) -> None:
    temporary = target.with_suffix(".tmp")
    try:
        with temporary.open("wb") as file:
            torch.save(state, file)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _save(model, optimizer, scheduler, output: Path, step: int, epoch: int,
          batch_index: int, metadata: dict, model_config: dict) -> None:
    # Trainer state is the sole resume source; weights.pt is only an export.
    _atomic_save({"version": 2, "model": model.state_dict(), "model_config": model_config,
                  "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict() if scheduler else None,
                  "step": step, "epoch": epoch, "batch_index": batch_index,
                  "metadata": metadata, "rng": _rng_state()}, output / "trainer_state.pth")
    _atomic_save(model.state_dict(), output / "weights.pt")


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


def _autocast(device: torch.device, precision: str):
    """bf16 autocast for forward+loss on CUDA (upstream trains in bf16); fp32 otherwise."""
    if precision not in {"auto", "fp32", "bf16"}:
        raise ValueError("precision must be auto, fp32, or bf16")
    use_bf16 = precision == "bf16" or (precision == "auto" and device.type == "cuda")
    if use_bf16 and device.type != "cuda":
        raise ValueError("bf16 precision is only supported on CUDA")
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), "bf16"
    return contextlib.nullcontext(), "fp32"


def _random_model(config: dict):
    # vit_hf(pretrained=True) calls HF from_pretrained; disable weight loading
    # without changing the architecture or the checkpoint's copied config.
    if config.get("encoder", {}).get("_target_") == "stable_pretraining.backbone.utils.vit_hf":
        return instantiate(config, encoder={"pretrained": False})
    return instantiate(config)


def finetune(dataset: str, name: str, init: str = "lewm-pusht", max_steps: int = 0,
             batch_size: int = 32, seed: int = 0, device: str = "auto",
             log_interval: int = 1, val_interval: int = 100,
             checkpoint_interval: int = 100, workers: int = 4,
             schedule_steps: int | None = None, precision: str = "auto",
             from_scratch: bool = False) -> dict:
    """Train to max_steps; schedule_steps is an immutable total LR budget."""
    budget = max_steps if schedule_steps is None else schedule_steps
    if (max_steps < 0 or batch_size < 1 or workers < 0
            or min(log_interval, val_interval, checkpoint_interval) < 1
            or budget < max_steps or budget < 0 or budget == 1):
        raise ValueError("Invalid training counts/intervals; schedule budget must be 0 or >=2 and >= max_steps")
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Output checkpoint name must be a single directory name")
    dataset_name = Path(dataset if dataset.endswith(".h5") else f"{dataset}.h5")
    init_name = Path(init)
    if any(path.is_absolute() or ".." in path.parts for path in (dataset_name, init_name)):
        raise ValueError("Dataset and init names must be relative to their storage roots")
    source = checkpoint_dir(init)
    output = checkpoint_dir(name)
    if output == source:
        raise ValueError("Output checkpoint must differ from initialization")
    state_file = output / "trainer_state.pth"
    state = torch.load(state_file, map_location="cpu", weights_only=False) if state_file.exists() else None
    if state is not None:
        _validate_resume(state, {})
    if state is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    stats = load_normalization(output if state is not None else source)
    metadata = {"dataset": {"name": dataset_name.as_posix(),
                            **_fingerprint(dataset_path(str(dataset_name)), cache=True)},
                "init": {"name": init_name.as_posix(),
                         **_fingerprint(source / "weights.pt")},
                "seed": seed, "batch_size": batch_size, "schedule_steps": budget,
                "from_scratch": from_scratch}
    if state is not None:
        # Reject changed content before trying to parse the dataset.
        _validate_resume(state, metadata)
    train_eps, val_eps = [], []
    split = {}
    if budget:
        train, val, train_eps, val_eps = load_windows(dataset, stats, seed)
        # Validate before creating output files or a model.
        _loader(train, batch_size, workers, seed, True)
        split = {"train_episodes": train_eps, "val_episodes": val_eps,
                 "train_windows": len(train), "val_windows": len(val)}
    metadata.update(train_episodes=train_eps, val_episodes=val_eps)
    if state is not None:
        _validate_resume(state, metadata)
        if max_steps < state["step"]:
            raise ValueError("max_steps is below the resumed step")
    selected_device = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if state is not None:
        model_config = state["model_config"]
        model = instantiate(model_config)
        model.load_state_dict(state["model"])
        model.to(selected_device)
        model.interpolate_pos_encoding = True
    else:
        model_config = json.loads((source / "config.json").read_text())
        if from_scratch:
            model = _random_model(model_config)
            model.to(selected_device).eval()
            model.interpolate_pos_encoding = True
        else:
            model = load_lewm(source, selected_device)
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "config.json", output / "config.json")
        shutil.copyfile(source / "normalization.json", output / "normalization.json")
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = _scheduler(optimizer, budget)
    step = epoch = batch_index = 0
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        step, epoch, batch_index = state["step"], state["epoch"], state["batch_index"]
    config = {"dataset": dataset, "name": name, "init": init, "max_steps": max_steps,
              "from_scratch": from_scratch,
              "schedule_steps": budget, "batch_size": batch_size, "seed": seed,
              "device": str(selected_device), "log_interval": log_interval,
              "val_interval": val_interval, "checkpoint_interval": checkpoint_interval,
              "workers": workers, "precision": _autocast(selected_device, precision)[1],
              "batch_size_deviation": "Default 32 instead of upstream 128: 128 exceeds this Mac's memory; configurable."}
    run_dir = create_run("finetune", config)
    start_step = step
    wall_start = time.perf_counter()
    if budget:
        val_loader = _loader(val, batch_size, workers, seed, False)
        sigreg = SIGReg(knots=17, num_proj=1024).to(selected_device)
        model.train()
        # Model/SIGReg construction must not consume the resumed RNG stream.
        if state is not None:
            _restore_rng(state["rng"])
        while step < max_steps:
            loader = _loader(train, batch_size, workers, seed + epoch, True)
            for index, batch in enumerate(loader):
                if index < batch_index:
                    continue
                start = time.perf_counter()
                batch = {k: v.to(selected_device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with _autocast(selected_device, precision)[0]:
                    losses = lejepa_loss(model, sigreg, batch)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                step += 1
                if selected_device.type == "mps":
                    torch.mps.synchronize()
                elif selected_device.type == "cuda":
                    torch.cuda.synchronize()
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
                    _save(model, optimizer, scheduler, output, step, epoch, index + 1,
                          metadata, model_config)
                if step == max_steps:
                    break
            else:
                epoch += 1
                batch_index = 0
                continue
            break
    if step == start_step:
        if state is not None:
            _restore_rng(state["rng"])
        _save(model, optimizer, scheduler, output, step, epoch, batch_index, metadata, model_config)
    summary = {**config, **split, "completed_steps": step, "start_step": start_step,
               "wall_seconds": time.perf_counter() - wall_start,
               "checkpoint": str(output), "run_dir": str(run_dir)}
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
