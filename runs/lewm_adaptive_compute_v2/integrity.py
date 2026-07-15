"""Strict artifact and provenance checks for LeWM adaptive-compute V2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def state_tensor_hash(module: torch.nn.Module) -> str:
    """Hash ordered tensor names, dtypes, shapes, and exact CPU bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def assert_frozen(module: torch.nn.Module, name: str) -> None:
    if module.training:
        raise RuntimeError(f"{name} must be in evaluation mode")
    trainable = [key for key, parameter in module.named_parameters() if parameter.requires_grad]
    gradients = [key for key, parameter in module.named_parameters() if parameter.grad is not None]
    if trainable:
        raise RuntimeError(f"{name} has trainable parameters: {trainable[:5]}")
    if gradients:
        raise RuntimeError(f"{name} has parameter gradients: {gradients[:5]}")


def _finite_json_value(value: Any, path: str = "root") -> Any:
    if isinstance(value, Mapping):
        return {str(key): _finite_json_value(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        return _finite_json_value(value.tolist(), path)
    if isinstance(value, np.generic):
        return _finite_json_value(value.item(), path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite JSON value at {path}: {value!r}")
        return value
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def strict_json_dump(path: Path, value: Any) -> None:
    clean = _finite_json_value(value)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit_npz_finite(path: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as arrays:
        for name in arrays.files:
            value = arrays[name]
            if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                raise ValueError(f"{path}:{name} contains nonfinite values")
            summary[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return summary


def audit_png(path: Path, *, min_width: int = 400, min_height: int = 300) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if image.format != "PNG":
            raise ValueError(f"{path} is not PNG")
        if width < min_width or height < min_height:
            raise ValueError(f"{path} is unexpectedly small: {width}x{height}")
        return {"width": width, "height": height, "format": image.format, "sha256": sha256_file(path)}
