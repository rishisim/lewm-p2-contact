"""Persisted pretrained PushT action/proprio normalization."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .paths import checkpoint_dir, dataset_path


def fit_normalization(dataset: Path, chunk_size: int = 100_000) -> dict:
    """Fit population z-scores on valid rows, as in sklearn StandardScaler."""
    result = {"source": dataset.name, "method": "standard_scaler", "columns": {}}
    with h5py.File(dataset, "r", swmr=True) as file:
        for column in ("action", "proprio"):
            scaler = StandardScaler()
            count = 0
            data = file[column]
            for start in range(0, len(data), chunk_size):
                rows = np.asarray(data[start:start + chunk_size]).reshape(-1, data.shape[-1])
                rows = rows[~np.isnan(rows).any(axis=1)]
                if len(rows):
                    scaler.partial_fit(rows)
                    count += len(rows)
            if not count:
                raise ValueError(f"No valid rows for {column}")
            result["columns"][column] = {
                "mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist(), "valid_rows": count,
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
