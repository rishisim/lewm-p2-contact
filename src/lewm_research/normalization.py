"""Persisted pretrained PushT action/proprio normalization."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .paths import checkpoint_dir, dataset_path


def fit_normalization(dataset: Path, chunk_size: int = 100_000) -> dict:
    """Match upstream get_column_normalizer, preserving HDF5 dtype.

    Full-column torch reductions are intentional: chunked/float64 reductions
    change rounding relative to upstream. ``chunk_size`` remains API-compatible.
    """
    result = {"source": dataset.name, "method": "torch_sample_std", "columns": {}}
    with h5py.File(dataset, "r", swmr=True) as file:
        for column in ("action", "proprio"):
            rows = torch.from_numpy(np.array(file[column]))
            rows = rows[~torch.isnan(rows).any(dim=1)]
            if len(rows) < 2:
                raise ValueError(f"At least two valid rows required for {column}")
            result["columns"][column] = {
                "mean": rows.mean(0, keepdim=True).clone().flatten().tolist(),
                "std": rows.std(0, keepdim=True).clone().flatten().tolist(),
                "valid_rows": len(rows),
            }
    return result


def save_normalization(stats: dict, checkpoint: Path) -> Path:
    checkpoint.mkdir(parents=True, exist_ok=True)
    target = checkpoint / "normalization.json"
    target.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return target


def load_normalization(checkpoint: Path) -> dict:
    path = checkpoint / "normalization.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def normalize_column(values: torch.Tensor, stats: dict, column: str) -> torch.Tensor:
    spec = stats["columns"][column]
    mean = torch.as_tensor(spec["mean"], dtype=values.dtype, device=values.device)
    std = torch.as_tensor(spec["std"], dtype=values.dtype, device=values.device)
    return ((values - mean) / std).float()


def create_pretrained_normalization() -> dict:
    source = dataset_path("pusht_expert_train.h5")
    stats = fit_normalization(source)
    target = save_normalization(stats, checkpoint_dir("lewm-pusht"))
    return {"path": str(target), **stats}
