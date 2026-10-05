"""Thin helpers for the pinned LeWM checkpoint."""

from pathlib import Path

import torch
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.utils import load_pretrained


def load_lewm(checkpoint_dir: Path, device: torch.device) -> LeWM:
    """Load local weights and config using stable-worldmodel 0.1.1 semantics."""
    for filename in ("weights.pt", "config.json"):
        if not (checkpoint_dir / filename).is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_dir / filename}")
    model = load_pretrained(str(checkpoint_dir))
    model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def encode(model: LeWM, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CLS and projected embeddings for BCHW pixels."""
    cls = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    return cls, model.projector(cls)
