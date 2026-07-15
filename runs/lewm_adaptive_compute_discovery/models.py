"""Leakage-resistant solver architectures for adaptive-compute discovery.

This module deliberately contains no data or checkpoint loading.  Callers own
checkpoint provenance and pass an already-loaded V1 ``SharedResidualRefiner``
to the wrappers below.  The frozen anchor is always evaluated through V1's
public ``forward(..., depths=(0, 1))`` path, which makes depth one an exact
architectural constraint rather than a distillation target.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import re
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SolverComputeStats:
    """Exact per-row solver-block accounting for one forward call."""

    processed_rows: int
    block_invocations: int

    @property
    def block_calls(self) -> int:
        return self.processed_rows

    def as_dict(self) -> dict[str, int]:
        return {
            "processed_rows": self.processed_rows,
            "block_calls": self.block_calls,
            "block_invocations": self.block_invocations,
        }


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_inputs(
    history: Tensor,
    action_history: Tensor,
    base_prediction: Tensor,
    *,
    latent_dim: int,
    action_dim: int,
    history_len: int,
) -> int:
    for name, value in (
        ("history", history),
        ("action_history", action_history),
        ("base_prediction", base_prediction),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.is_floating_point(value):
            raise ValueError(f"{name} must have a floating dtype")
    if history.ndim != 3 or tuple(history.shape[1:]) != (history_len, latent_dim):
        raise ValueError(
            f"history must have shape [batch, {history_len}, {latent_dim}], "
            f"got {tuple(history.shape)}"
        )
    if action_history.ndim != 3 or tuple(action_history.shape[1:]) != (
        history_len,
        action_dim,
    ):
        raise ValueError(
            f"action_history must have shape [batch, {history_len}, {action_dim}], "
            f"got {tuple(action_history.shape)}"
        )
    if base_prediction.ndim != 2 or base_prediction.shape[1] != latent_dim:
        raise ValueError(
            f"base_prediction must have shape [batch, {latent_dim}], "
            f"got {tuple(base_prediction.shape)}"
        )
    batch = history.shape[0]
    if batch <= 0:
        raise ValueError("batch must be nonempty")
    if action_history.shape[0] != batch or base_prediction.shape[0] != batch:
        raise ValueError("solver input batch sizes differ")
    if not (
        history.device == action_history.device == base_prediction.device
        and history.dtype == action_history.dtype == base_prediction.dtype
    ):
        raise ValueError("solver inputs must share device and dtype")
    return batch


def _selected_depths(
    depths: Tensor | Sequence[int], *, batch: int, maximum: int, device: torch.device
) -> Tensor:
    if isinstance(depths, Tensor):
        if depths.dtype == torch.bool or torch.is_floating_point(depths):
            raise ValueError("selected depths must use an integer dtype")
        result = depths.to(device=device, dtype=torch.long)
    else:
        values = tuple(depths)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError("selected depths must be integers")
        result = torch.as_tensor(values, device=device, dtype=torch.long)
    if result.shape != (batch,):
        raise ValueError(f"selected depths must have shape [{batch}]")
    if bool(((result < 0) | (result > maximum)).any().item()):
        raise ValueError(f"selected depths must lie in 0..{maximum}")
    return result


def _extract_v1_outputs(value: object) -> Mapping[int, Tensor]:
    # V1 optionally returns (outputs, stats); accepting the tuple keeps this
    # wrapper compatible with instrumented callers without weakening checks.
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, Mapping) or 0 not in value or 1 not in value:
        raise TypeError("V1 refiner must return a mapping containing exits 0 and 1")
    if not isinstance(value[0], Tensor) or not isinstance(value[1], Tensor):
        raise TypeError("V1 exits must be tensors")
    return value


class FrozenV1Anchor(nn.Module):
    """Own an immutable V1 refiner copy and expose its exact depth-1 exit.

    ``copy_refiner=True`` is intentional: freezing this wrapper never mutates a
    caller's model.  Set it false only when transferring ownership explicitly.
    ``train()`` cannot put the anchor into training mode.
    """

    def __init__(self, v1_refiner: nn.Module, *, copy_refiner: bool = True) -> None:
        super().__init__()
        if not isinstance(v1_refiner, nn.Module):
            raise TypeError("v1_refiner must be an nn.Module")
        for name in ("latent_dim", "action_dim", "history_len"):
            if not hasattr(v1_refiner, name):
                raise TypeError(f"V1 refiner is missing required attribute {name!r}")
        self.latent_dim = int(v1_refiner.latent_dim)
        self.action_dim = int(v1_refiner.action_dim)
        self.history_len = int(v1_refiner.history_len)
        self.refiner = copy.deepcopy(v1_refiner) if copy_refiner else v1_refiner
        self.refiner.eval().requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "FrozenV1Anchor":
        # Keep the child immutable/eval even when a parent cascade is trained.
        super().train(False)
        self.refiner.eval()
        return self

    def forward(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
        *,
        return_stats: bool = False,
    ) -> dict[int, Tensor] | tuple[dict[int, Tensor], SolverComputeStats]:
        batch = _validate_inputs(
            history,
            action_history,
            base_prediction,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            history_len=self.history_len,
        )
        with torch.no_grad():
            raw = self.refiner(
                history, action_history, base_prediction, depths=(0, 1)
            )
        v1 = _extract_v1_outputs(raw)
        if v1[0] is not base_prediction or not torch.equal(v1[0], base_prediction):
            raise RuntimeError("V1 depth zero did not preserve the base tensor exactly")
        if v1[1].shape != base_prediction.shape:
            raise RuntimeError("V1 depth-one shape differs from the base prediction")
        outputs = {0: base_prediction, 1: v1[1]}
        if return_stats:
            return outputs, SolverComputeStats(batch, 1)
        return outputs


class StageResidualAdapter(nn.Module):
    """One depth-specific, zero-initialized residual solver stage."""

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        history_len: int,
        hidden_dim: int = 128,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.action_dim = _positive_int("action_dim", action_dim)
        self.history_len = _positive_int("history_len", history_len)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha must lie in (0, 1]")
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        input_dim = history_len * latent_dim + history_len * action_dim + latent_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, history: Tensor, action_history: Tensor, preceding_exit: Tensor
    ) -> Tensor:
        _validate_inputs(
            history,
            action_history,
            preceding_exit,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            history_len=self.history_len,
        )
        features = torch.cat(
            (history.flatten(1), action_history.flatten(1), preceding_exit), dim=1
        )
        return self.alpha.to(dtype=preceding_exit.dtype) * self.network(features)


class StagewiseResidualCascade(nn.Module):
    """Frozen V1 depth one followed by independently trainable residual stages.

    Depth ``d >= 2`` is ``detach(z[d-1]) + adapter[d](...)``.  Calling
    :meth:`set_trainable_stage` freezes every other solver parameter, enforcing
    stagewise training mechanically rather than by optimizer convention.
    """

    def __init__(
        self,
        v1_refiner: nn.Module | FrozenV1Anchor,
        *,
        later_stages: int = 3,
        hidden_dim: int = 128,
        alphas: float | Sequence[float] = 1.0,
    ) -> None:
        super().__init__()
        self.anchor = (
            v1_refiner
            if isinstance(v1_refiner, FrozenV1Anchor)
            else FrozenV1Anchor(v1_refiner)
        )
        self.latent_dim = self.anchor.latent_dim
        self.action_dim = self.anchor.action_dim
        self.history_len = self.anchor.history_len
        later_stages = _positive_int("later_stages", later_stages)
        if isinstance(alphas, (int, float)) and not isinstance(alphas, bool):
            alpha_values = (float(alphas),) * later_stages
        else:
            alpha_values = tuple(float(value) for value in alphas)
        if len(alpha_values) != later_stages:
            raise ValueError("alphas must contain one value per later stage")
        self.adapters = nn.ModuleList(
            StageResidualAdapter(
                latent_dim=self.latent_dim,
                action_dim=self.action_dim,
                history_len=self.history_len,
                hidden_dim=hidden_dim,
                alpha=alpha_values[index],
            )
            for index in range(later_stages)
        )
        self.max_depth = 1 + later_stages
        self._trainable_depth: int | None = None
        self.set_trainable_stage(2)
        self.train(True)

    @property
    def trainable_depth(self) -> int | None:
        return self._trainable_depth

    def set_trainable_stage(self, depth: int | None) -> None:
        if depth is not None and (
            isinstance(depth, bool) or not isinstance(depth, int) or depth < 2 or depth > self.max_depth
        ):
            raise ValueError(f"trainable stage must be None or lie in 2..{self.max_depth}")
        self.anchor.requires_grad_(False)
        for stage_depth, adapter in enumerate(self.adapters, start=2):
            adapter.requires_grad_(stage_depth == depth)
        self._trainable_depth = depth

    def train(self, mode: bool = True) -> "StagewiseResidualCascade":
        super().train(mode)
        self.anchor.eval()
        for stage_depth, adapter in enumerate(self.adapters, start=2):
            adapter.train(mode and stage_depth == self._trainable_depth)
        return self

    def _dense(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
        maximum: int,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor], SolverComputeStats]:
        anchors = self.anchor(history, action_history, base_prediction)
        assert isinstance(anchors, dict)
        outputs = dict(anchors)
        updates: dict[int, Tensor] = {1: anchors[1] - base_prediction}
        current = anchors[1]
        for depth in range(2, maximum + 1):
            # All upstream solver tensors are detached at every stage boundary.
            preceding = current.detach()
            update = self.adapters[depth - 2](
                history.detach(), action_history.detach(), preceding
            )
            current = preceding + update
            outputs[depth] = current
            updates[depth] = update
        batch = len(base_prediction)
        return outputs, updates, SolverComputeStats(batch * maximum, maximum)

    def forward(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
        *,
        max_depth: int | None = None,
        return_updates: bool = False,
        return_stats: bool = False,
    ) -> object:
        maximum = self.max_depth if max_depth is None else max_depth
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= self.max_depth:
            raise ValueError(f"max_depth must lie in 1..{self.max_depth}")
        outputs, updates, stats = self._dense(
            history, action_history, base_prediction, maximum
        )
        if return_updates and return_stats:
            return outputs, updates, stats
        if return_updates:
            return outputs, updates
        if return_stats:
            return outputs, stats
        return outputs

    def forward_selected(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
        selected_depths: Tensor | Sequence[int],
        *,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SolverComputeStats]:
        batch = _validate_inputs(
            history,
            action_history,
            base_prediction,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            history_len=self.history_len,
        )
        selected = _selected_depths(
            selected_depths, batch=batch, maximum=self.max_depth, device=base_prediction.device
        )
        current = base_prediction
        invocations = 0
        active = torch.nonzero(selected >= 1, as_tuple=False).flatten()
        if active.numel():
            anchored = self.anchor(
                history.index_select(0, active),
                action_history.index_select(0, active),
                base_prediction.index_select(0, active),
            )
            assert isinstance(anchored, dict)
            current = current.index_copy(0, active, anchored[1])
            invocations += 1
        for depth, adapter in enumerate(self.adapters, start=2):
            active = torch.nonzero(selected >= depth, as_tuple=False).flatten()
            if not active.numel():
                continue
            preceding = current.index_select(0, active).detach()
            update = adapter(
                history.index_select(0, active).detach(),
                action_history.index_select(0, active).detach(),
                preceding,
            )
            current = current.index_copy(0, active, preceding + update)
            invocations += 1
        stats = SolverComputeStats(int(selected.sum().item()), invocations)
        return (current, stats) if return_stats else current


class V3RepairControl(nn.Module):
    """Minimal V3 control: immutable V1 call 0, trainable later-call copy.

    The later block and iteration embeddings are copied from V1, but iteration
    row zero is deliberately omitted.  Consequently every trainable parameter
    is used only after the protected V1 depth-one exit.  At construction, dense
    outputs reproduce V1's recurrent later calls up to floating-point identity.
    """

    def __init__(self, v1_refiner: nn.Module) -> None:
        super().__init__()
        for name in ("block", "iteration_embedding", "max_depth"):
            if not hasattr(v1_refiner, name):
                raise TypeError(f"V1 refiner is missing required attribute {name!r}")
        self.anchor = FrozenV1Anchor(v1_refiner)
        self.latent_dim = self.anchor.latent_dim
        self.action_dim = self.anchor.action_dim
        self.history_len = self.anchor.history_len
        self.max_depth = int(v1_refiner.max_depth)
        if self.max_depth < 2:
            raise ValueError("V3 repair requires at least one later depth")
        self.later_block = copy.deepcopy(v1_refiner.block)
        source_embeddings = v1_refiner.iteration_embedding.weight.detach()
        if source_embeddings.shape[0] < self.max_depth:
            raise ValueError("V1 iteration embedding is shorter than max_depth")
        self.later_iteration_embedding = nn.Embedding(
            self.max_depth - 1,
            source_embeddings.shape[1],
            device=source_embeddings.device,
            dtype=source_embeddings.dtype,
        )
        with torch.no_grad():
            self.later_iteration_embedding.weight.copy_(
                source_embeddings[1 : self.max_depth]
            )
        self.anchor.requires_grad_(False)

    def train(self, mode: bool = True) -> "V3RepairControl":
        super().train(mode)
        self.anchor.eval()
        return self

    def _later_update(
        self,
        history: Tensor,
        action_history: Tensor,
        current: Tensor,
        depth: int,
    ) -> Tensor:
        if depth < 2 or depth > self.max_depth:
            raise ValueError("later depth outside repair-control range")
        ids = torch.full(
            (len(current),), depth - 2, dtype=torch.long, device=current.device
        )
        iteration = self.later_iteration_embedding(ids)
        values = torch.cat(
            (history.flatten(1), action_history.flatten(1), current, iteration), dim=1
        )
        update = self.later_block(values)
        if update.shape != current.shape:
            raise RuntimeError("later block returned an invalid update shape")
        return update

    def forward(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
        *,
        max_depth: int | None = None,
        return_stats: bool = False,
    ) -> dict[int, Tensor] | tuple[dict[int, Tensor], SolverComputeStats]:
        maximum = self.max_depth if max_depth is None else max_depth
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= self.max_depth:
            raise ValueError(f"max_depth must lie in 1..{self.max_depth}")
        anchored = self.anchor(history, action_history, base_prediction)
        assert isinstance(anchored, dict)
        outputs = dict(anchored)
        current = anchored[1].detach()
        for depth in range(2, maximum + 1):
            current = current + self._later_update(
                history.detach(), action_history.detach(), current, depth
            )
            outputs[depth] = current
        stats = SolverComputeStats(len(base_prediction) * maximum, maximum)
        return (outputs, stats) if return_stats else outputs

    def forward_selected(
        self, history: Tensor, action_history: Tensor, base_prediction: Tensor,
        selected_depths: Tensor | Sequence[int], *, return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SolverComputeStats]:
        batch = _validate_inputs(history, action_history, base_prediction,
            latent_dim=self.latent_dim, action_dim=self.action_dim, history_len=self.history_len)
        selected = _selected_depths(selected_depths, batch=batch, maximum=self.max_depth, device=base_prediction.device)
        current = base_prediction; invocations = 0
        active = torch.nonzero(selected >= 1, as_tuple=False).flatten()
        if active.numel():
            anchored = self.anchor(history.index_select(0, active), action_history.index_select(0, active), base_prediction.index_select(0, active))
            assert isinstance(anchored, dict)
            current = current.index_copy(0, active, anchored[1]); invocations += 1
        for depth in range(2, self.max_depth + 1):
            active = torch.nonzero(selected >= depth, as_tuple=False).flatten()
            if not active.numel(): continue
            preceding = current.index_select(0, active)
            update = self._later_update(history.index_select(0, active).detach(), action_history.index_select(0, active).detach(), preceding, depth)
            current = current.index_copy(0, active, preceding + update); invocations += 1
        stats = SolverComputeStats(int(selected.sum().item()), invocations)
        return (current, stats) if return_stats else current


class _EquilibriumAdapter(nn.Module):
    def __init__(
        self, *, latent_dim: int, action_dim: int, history_len: int, hidden_dim: int
    ) -> None:
        super().__init__()
        input_dim = history_len * latent_dim + history_len * action_dim + latent_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, history: Tensor, actions: Tensor, anchor: Tensor) -> Tensor:
        return self.network(
            torch.cat((history.flatten(1), actions.flatten(1), anchor), dim=1)
        )


class ContractiveSharedResidualCascade(nn.Module):
    """Shared contractive relaxation toward a causal learned equilibrium.

    For frozen context and ``0 < relaxation <= 1``, each later call applies
    ``T(z) = z + relaxation * (equilibrium - z)``.  Thus T is a contraction in
    z with exact factor ``1 - relaxation``.  The equilibrium offset is
    zero-initialized, making every epoch-zero exit equal to protected V1 z1.
    """

    def __init__(
        self,
        v1_refiner: nn.Module | FrozenV1Anchor,
        *,
        max_depth: int = 4,
        hidden_dim: int = 128,
        relaxation: float = 0.5,
    ) -> None:
        super().__init__()
        self.anchor = (
            v1_refiner
            if isinstance(v1_refiner, FrozenV1Anchor)
            else FrozenV1Anchor(v1_refiner)
        )
        self.latent_dim = self.anchor.latent_dim
        self.action_dim = self.anchor.action_dim
        self.history_len = self.anchor.history_len
        self.max_depth = _positive_int("max_depth", max_depth)
        if self.max_depth < 2:
            raise ValueError("max_depth must be at least two")
        _positive_int("hidden_dim", hidden_dim)
        if not 0.0 < float(relaxation) <= 1.0:
            raise ValueError("relaxation must lie in (0, 1]")
        self.register_buffer("relaxation", torch.tensor(float(relaxation)))
        self.shared_adapter = _EquilibriumAdapter(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            history_len=self.history_len,
            hidden_dim=hidden_dim,
        )
        self.anchor.requires_grad_(False)

    @property
    def contraction_factor(self) -> float:
        return 1.0 - float(self.relaxation.item())

    def train(self, mode: bool = True) -> "ContractiveSharedResidualCascade":
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
        *,
        max_depth: int | None = None,
        return_stats: bool = False,
    ) -> dict[int, Tensor] | tuple[dict[int, Tensor], SolverComputeStats]:
        maximum = self.max_depth if max_depth is None else max_depth
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= self.max_depth:
            raise ValueError(f"max_depth must lie in 1..{self.max_depth}")
        anchored = self.anchor(history, action_history, base_prediction)
        assert isinstance(anchored, dict)
        outputs = dict(anchored)
        z1 = anchored[1].detach()
        current = z1
        rho = self.relaxation.to(dtype=current.dtype)
        for depth in range(2, maximum + 1):
            # Recompute through the shared block on every counted solver call.
            equilibrium = z1 + self.shared_adapter(
                history.detach(), action_history.detach(), z1
            )
            current = current + rho * (equilibrium - current)
            outputs[depth] = current
        stats = SolverComputeStats(len(base_prediction) * maximum, maximum)
        return (outputs, stats) if return_stats else outputs

    def stability_losses(self, outputs: Mapping[int, Tensor]) -> dict[str, Tensor]:
        """Target-free shortcut/stability diagnostics for consecutive exits."""
        depths = sorted(depth for depth in outputs if depth >= 1)
        if len(depths) < 3 or any(b != a + 1 for a, b in zip(depths, depths[1:])):
            raise ValueError("stability losses require at least three consecutive exits")
        factor = self.contraction_factor
        shortcut_terms = []
        violation_terms = []
        for d0, d1, d2 in zip(depths, depths[1:], depths[2:]):
            previous_step = outputs[d1] - outputs[d0]
            next_step = outputs[d2] - outputs[d1]
            shortcut_terms.append((next_step - factor * previous_step).square().mean())
            previous_norm = torch.linalg.vector_norm(previous_step, dim=1)
            next_norm = torch.linalg.vector_norm(next_step, dim=1)
            violation_terms.append(torch.relu(next_norm - factor * previous_norm).mean())
        return {
            "shortcut_consistency": torch.stack(shortcut_terms).mean(),
            "contraction_violation": torch.stack(violation_terms).mean(),
        }

    def forward_selected(
        self, history: Tensor, action_history: Tensor, base_prediction: Tensor,
        selected_depths: Tensor | Sequence[int], *, return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SolverComputeStats]:
        batch = _validate_inputs(history, action_history, base_prediction,
            latent_dim=self.latent_dim, action_dim=self.action_dim, history_len=self.history_len)
        selected = _selected_depths(selected_depths, batch=batch, maximum=self.max_depth, device=base_prediction.device)
        current = base_prediction; z1 = base_prediction; invocations = 0
        active = torch.nonzero(selected >= 1, as_tuple=False).flatten()
        if active.numel():
            anchored = self.anchor(history.index_select(0, active), action_history.index_select(0, active), base_prediction.index_select(0, active))
            assert isinstance(anchored, dict)
            current = current.index_copy(0, active, anchored[1]); z1 = z1.index_copy(0, active, anchored[1]); invocations += 1
        rho = self.relaxation.to(dtype=current.dtype)
        for depth in range(2, self.max_depth + 1):
            active = torch.nonzero(selected >= depth, as_tuple=False).flatten()
            if not active.numel(): continue
            anchored_active = z1.index_select(0, active)
            equilibrium = anchored_active + self.shared_adapter(history.index_select(0, active).detach(), action_history.index_select(0, active).detach(), anchored_active)
            preceding = current.index_select(0, active)
            current = current.index_copy(0, active, preceding + rho * (equilibrium - preceding)); invocations += 1
        stats = SolverComputeStats(int(selected.sum().item()), invocations)
        return (current, stats) if return_stats else current


_FORBIDDEN_FEATURE_TOKENS = frozenset(
    {
        "contact",
        "episode",
        "future",
        "gain",
        "global",
        "groundtruth",
        "label",
        "loss",
        "oracle",
        "outcome",
        "regime",
        "target",
        "truth",
    }
)


def validate_causal_feature_names(names: Iterable[str]) -> tuple[str, ...]:
    values = tuple(names)
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise ValueError("causal feature names must be nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError("causal feature names must be unique")
    rejected = []
    for name in values:
        tokens = re.findall(r"[a-z0-9]+", name.lower())
        if any(token in _FORBIDDEN_FEATURE_TOKENS for token in tokens):
            rejected.append(name)
    if rejected:
        raise ValueError("forbidden critic feature names: " + ", ".join(rejected))
    return values


def causal_feature_names(
    *, latent_dim: int, action_dim: int, history_len: int
) -> tuple[str, ...]:
    _positive_int("latent_dim", latent_dim)
    _positive_int("action_dim", action_dim)
    _positive_int("history_len", history_len)
    names = [
        f"history_{time}_{dimension}"
        for time in range(history_len)
        for dimension in range(latent_dim)
    ]
    names += [
        f"action_{time}_{dimension}"
        for time in range(history_len)
        for dimension in range(action_dim)
    ]
    names += [f"current_{dimension}" for dimension in range(latent_dim)]
    names += [f"last_update_{dimension}" for dimension in range(latent_dim)]
    names += [
        "current_norm",
        "last_update_norm",
        "relative_update_norm",
        "last_history_norm",
        "current_history_distance",
        "update_current_cosine",
        "update_history_distance_cosine",
    ]
    names += [f"history_change_{time}_{time + 1}_norm" for time in range(history_len - 1)]
    names += [f"action_change_{time}_{time + 1}_norm" for time in range(history_len - 1)]
    return validate_causal_feature_names(names)


def build_causal_features(
    history: Tensor,
    action_history: Tensor,
    current_prediction: Tensor,
    last_update: Tensor,
    *,
    return_names: bool = False,
) -> Tensor | tuple[Tensor, tuple[str, ...]]:
    """Build row-local critic features and sever every solver gradient path.

    The function has no target/future/label argument and performs no batch or
    episode reduction.  All inputs are detached before feature construction,
    so a critic loss cannot reach solver parameters even if the solver itself
    remains trainable during an audit.
    """
    if history.ndim != 3 or action_history.ndim != 3 or current_prediction.ndim != 2:
        raise ValueError("invalid causal feature input rank")
    batch, history_len, latent_dim = history.shape
    if batch <= 0 or history_len <= 0 or latent_dim <= 0:
        raise ValueError("causal feature dimensions must be nonzero")
    if action_history.shape[:2] != (batch, history_len):
        raise ValueError("history and action batch/history dimensions differ")
    action_dim = action_history.shape[2]
    if current_prediction.shape != (batch, latent_dim):
        raise ValueError("current prediction shape does not match history")
    if last_update.shape != current_prediction.shape:
        raise ValueError("last update shape does not match current prediction")
    if not (
        history.device == action_history.device == current_prediction.device == last_update.device
        and history.dtype == action_history.dtype == current_prediction.dtype == last_update.dtype
    ):
        raise ValueError("causal feature inputs must share device and dtype")
    if not all(torch.is_floating_point(value) for value in (history, action_history, current_prediction, last_update)):
        raise ValueError("causal feature inputs must have floating dtypes")

    history = history.detach()
    action_history = action_history.detach()
    current_prediction = current_prediction.detach()
    last_update = last_update.detach()
    eps = torch.finfo(current_prediction.dtype).eps
    gap = current_prediction - history[:, -1]

    def norm(value: Tensor) -> Tensor:
        return torch.linalg.vector_norm(value, dim=-1, keepdim=True)

    def cosine(left: Tensor, right: Tensor) -> Tensor:
        denominator = (norm(left) * norm(right)).clamp_min(eps)
        return (left * right).sum(dim=-1, keepdim=True) / denominator

    summaries = [
        norm(current_prediction),
        norm(last_update),
        norm(last_update) / norm(current_prediction).clamp_min(eps),
        norm(history[:, -1]),
        norm(gap),
        cosine(last_update, current_prediction),
        cosine(last_update, gap),
    ]
    if history_len > 1:
        summaries.append(torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=2))
        summaries.append(
            torch.linalg.vector_norm(action_history[:, 1:] - action_history[:, :-1], dim=2)
        )
    features = torch.cat(
        (
            history.flatten(1),
            action_history.flatten(1),
            current_prediction,
            last_update,
            *summaries,
        ),
        dim=1,
    )
    names = causal_feature_names(
        latent_dim=latent_dim, action_dim=action_dim, history_len=history_len
    )
    if features.shape != (batch, len(names)):
        raise RuntimeError("causal feature/name width mismatch")
    if not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("causal features contain nonfinite values")
    return (features, names) if return_names else features


@dataclass(frozen=True)
class GradientBoundaryAudit:
    solver_trainable_parameters: int
    solver_connected_tensors: int
    solver_nonzero_tensors: int
    critic_trainable_parameters: int
    critic_connected_tensors: int
    critic_nonzero_tensors: int

    @property
    def passed(self) -> bool:
        return (
            self.solver_connected_tensors == 0
            and self.solver_nonzero_tensors == 0
            and self.critic_nonzero_tensors > 0
        )

    def as_dict(self) -> dict[str, int | bool]:
        return {**asdict(self), "passed": self.passed}


def audit_critic_gradient_boundary(
    critic_scores: Tensor,
    *,
    solver: nn.Module,
    critic: nn.Module,
    retain_graph: bool = True,
) -> GradientBoundaryAudit:
    """Probe critic-only autograd connectivity without accumulating ``.grad``."""
    if not isinstance(critic_scores, Tensor) or not critic_scores.requires_grad:
        raise ValueError("critic_scores must be a differentiable tensor")
    solver_parameters = [parameter for parameter in solver.parameters() if parameter.requires_grad]
    critic_parameters = [parameter for parameter in critic.parameters() if parameter.requires_grad]
    if not critic_parameters:
        raise ValueError("critic has no trainable parameters")
    parameters = solver_parameters + critic_parameters
    gradients = torch.autograd.grad(
        critic_scores.sum(),
        parameters,
        allow_unused=True,
        retain_graph=retain_graph,
    )
    solver_gradients = gradients[: len(solver_parameters)]
    critic_gradients = gradients[len(solver_parameters) :]

    def nonzero(gradient: Tensor | None) -> bool:
        return gradient is not None and bool(torch.count_nonzero(gradient).item())

    return GradientBoundaryAudit(
        solver_trainable_parameters=sum(parameter.numel() for parameter in solver_parameters),
        solver_connected_tensors=sum(gradient is not None for gradient in solver_gradients),
        solver_nonzero_tensors=sum(nonzero(gradient) for gradient in solver_gradients),
        critic_trainable_parameters=sum(parameter.numel() for parameter in critic_parameters),
        critic_connected_tensors=sum(gradient is not None for gradient in critic_gradients),
        critic_nonzero_tensors=sum(nonzero(gradient) for gradient in critic_gradients),
    )


def assert_critic_gradient_boundary(
    critic_scores: Tensor,
    *,
    solver: nn.Module,
    critic: nn.Module,
    retain_graph: bool = True,
) -> GradientBoundaryAudit:
    audit = audit_critic_gradient_boundary(
        critic_scores, solver=solver, critic=critic, retain_graph=retain_graph
    )
    if not audit.passed:
        raise RuntimeError(f"critic gradient-boundary audit failed: {audit.as_dict()}")
    return audit


@dataclass(frozen=True)
class EpochZeroAnchorAudit:
    depth_zero_identity: bool
    depth_zero_bitwise: bool
    depth_one_bitwise: bool
    later_exits_preserve_predecessor: bool

    @property
    def passed(self) -> bool:
        return all(asdict(self).values())

    def as_dict(self) -> dict[str, bool]:
        return {**asdict(self), "passed": self.passed}


def audit_epoch_zero_anchor(
    outputs: Mapping[int, Tensor],
    *,
    base_prediction: Tensor,
    expected_v1_depth_one: Tensor,
) -> EpochZeroAnchorAudit:
    """Mechanically check anchor identity and zero-init eligibility."""
    if 0 not in outputs or 1 not in outputs:
        raise ValueError("outputs must contain depths zero and one")
    later_depths = sorted(depth for depth in outputs if depth >= 2)
    prior = outputs[1]
    later_preserved = True
    for depth in later_depths:
        later_preserved &= torch.equal(outputs[depth], prior)
        prior = outputs[depth]
    return EpochZeroAnchorAudit(
        depth_zero_identity=outputs[0] is base_prediction,
        depth_zero_bitwise=torch.equal(outputs[0], base_prediction),
        depth_one_bitwise=torch.equal(outputs[1], expected_v1_depth_one),
        later_exits_preserve_predecessor=later_preserved,
    )


def trainable_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Return a stable parameter-scope audit for configs and reports."""
    return tuple(name for name, parameter in module.named_parameters() if parameter.requires_grad)


__all__ = [
    "ContractiveSharedResidualCascade",
    "EpochZeroAnchorAudit",
    "FrozenV1Anchor",
    "GradientBoundaryAudit",
    "SolverComputeStats",
    "StageResidualAdapter",
    "StagewiseResidualCascade",
    "V3RepairControl",
    "assert_critic_gradient_boundary",
    "audit_critic_gradient_boundary",
    "audit_epoch_zero_anchor",
    "build_causal_features",
    "causal_feature_names",
    "trainable_parameter_names",
    "validate_causal_feature_names",
]
