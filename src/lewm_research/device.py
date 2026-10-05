"""Select a usable PyTorch device."""

import os

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA, then MPS, then CPU; fall back when unavailable."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    requested = requested.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device: {requested}")
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
