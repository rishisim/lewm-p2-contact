"""Stable planner-independent PushT refiner interface.

This module freezes the Task B boundary.  It intentionally contains no
checkpoint or training routine: positive depths remain unavailable until the
preregistered data-readiness gate is satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import torch
from torch import Tensor, nn


LATENT_DIM: Final = 192
RAW_ACTION_DIM: Final = 2
ACTION_BLOCK: Final = 5
ACTION_DIM: Final = RAW_ACTION_DIM * ACTION_BLOCK
MAX_HISTORY: Final = 3
MAX_DEPTH: Final = 4
SUPPORTED_DEPTHS: Final = (0, 1, 2, 4)
ACTION_LOW: Final = -1.0
ACTION_HIGH: Final = 1.0


def canonicalize_action(raw_action: Tensor, *, check_range: bool = True) -> Tensor:
    """Map physical PushT actions to the canonical representation.

    PushT's environment action is already dimensionless in ``[-1, 1]^2``.
    The canonical transform is therefore identity, independently for every raw
    action, with five chronological actions flattened into each planner block.
    """
    if not raw_action.is_floating_point():
        raise TypeError("PushT actions must be floating point")
    if raw_action.shape[-1] not in (RAW_ACTION_DIM, ACTION_DIM):
        raise ValueError("expected raw [...,2] actions or [...,10] action blocks")
    if not bool(torch.isfinite(raw_action).all()):
        raise ValueError("nonfinite PushT action")
    if check_range and not bool(
        ((raw_action >= ACTION_LOW) & (raw_action <= ACTION_HIGH)).all()
    ):
        raise ValueError("PushT action outside the environment range [-1, 1]")
    return raw_action


def physical_action(canonical_action: Tensor, *, check_range: bool = True) -> Tensor:
    """Invert :func:`canonicalize_action` (also identity)."""
    return canonicalize_action(canonical_action, check_range=check_range)


def prefix_mask(lengths: Tensor) -> Tensor:
    """Return the sole supported history mask: left padding, valid suffix.

    A length of one maps to ``[0, 0, 1]``; two maps to ``[0, 1, 1]``; and
    three maps to ``[1, 1, 1]``.  Valid rows are chronological.
    """
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise TypeError("history lengths must use an integer dtype")
    if lengths.ndim != 1:
        raise ValueError("history lengths must have shape [batch]")
    if not bool(((lengths >= 1) & (lengths <= MAX_HISTORY)).all()):
        raise ValueError("history lengths must be in [1, 3]")
    positions = torch.arange(MAX_HISTORY, device=lengths.device)
    return positions.unsqueeze(0) >= (MAX_HISTORY - lengths).unsqueeze(1)


@dataclass
class OperationLedger:
    base_calls: int = 0
    base_rows: int = 0
    stage_calls: list[int] = field(default_factory=lambda: [0] * MAX_DEPTH)
    stage_rows: list[int] = field(default_factory=lambda: [0] * MAX_DEPTH)

    def record_base(self, rows: int) -> None:
        self.base_calls += 1
        self.base_rows += int(rows)

    def record_stage(self, stage: int, rows: int) -> None:
        self.stage_calls[stage] += 1
        self.stage_rows[stage] += int(rows)


class MaskedStagewiseRefiner(nn.Module):
    """Small stagewise residual design implementing the frozen B1 contract."""

    def __init__(
        self,
        *,
        hidden: int = 256,
        iteration_dim: int = 16,
        verified_export: bool = False,
    ) -> None:
        super().__init__()
        self.verified_export = bool(verified_export)
        self.iteration_embedding = nn.Embedding(MAX_DEPTH, iteration_dim)
        width = (
            MAX_HISTORY * LATENT_DIM
            + MAX_HISTORY * ACTION_DIM
            + MAX_HISTORY
            + LATENT_DIM
            + iteration_dim
        )
        self.block = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, LATENT_DIM),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    @staticmethod
    def _validate(
        history: Tensor, actions: Tensor, mask: Tensor, base: Tensor, depth: int
    ) -> Tensor:
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise TypeError("refinement depth must be an integer")
        if depth not in SUPPORTED_DEPTHS:
            raise ValueError(f"unsupported refinement depth {depth}")
        batch = base.shape[0]
        if base.shape != (batch, LATENT_DIM):
            raise ValueError("base prediction must have shape [batch,192]")
        if history.shape != (batch, MAX_HISTORY, LATENT_DIM):
            raise ValueError("history must have shape [batch,3,192]")
        if actions.shape != (batch, MAX_HISTORY, ACTION_DIM):
            raise ValueError("actions must have shape [batch,3,10]")
        if mask.shape != (batch, MAX_HISTORY) or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [batch,3]")
        lengths = mask.sum(1)
        if not torch.equal(mask, prefix_mask(lengths)):
            raise ValueError("mask must be a nonempty valid suffix (left padding only)")
        if not all(bool(torch.isfinite(value).all()) for value in (history, actions, base)):
            raise ValueError("nonfinite refiner input")
        canonicalize_action(actions[mask], check_range=False)
        return lengths

    def forward(
        self,
        history: Tensor,
        actions: Tensor,
        mask: Tensor,
        base: Tensor,
        depth: int,
        *,
        ledger: OperationLedger | None = None,
    ) -> Tensor:
        self._validate(history, actions, mask, base, depth)
        if ledger is not None:
            ledger.record_base(len(base))
        if depth == 0:
            return base
        if not self.verified_export:
            raise RuntimeError(
                "positive refinement depth requires a B3-verified checkpoint export"
            )

        expanded_mask = mask.unsqueeze(-1)
        masked_history = history * expanded_mask
        masked_actions = actions * expanded_mask
        current = base
        for stage in range(depth):
            ids = torch.full(
                (len(base),), stage, dtype=torch.long, device=base.device
            )
            features = torch.cat(
                (
                    masked_history.flatten(1),
                    masked_actions.flatten(1),
                    mask.to(base.dtype),
                    current,
                    self.iteration_embedding(ids),
                ),
                dim=1,
            )
            current = current + self.block(features)
            if ledger is not None:
                ledger.record_stage(stage, len(base))
        return current
