"""Shared variable-depth latent refiner and causal gate features.

The module deliberately has no dependency on the released LeWM implementation.
It accepts the already-computed frozen LeWM prediction as ``base_prediction``;
therefore exit zero is the exact input tensor and costs no refiner calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


EXITS = (0, 1, 2, 4)
"""The preregistered V1 exits. Depth 8 is intentionally excluded."""


@dataclass(frozen=True)
class RefinerComputeStats:
    """Compute performed by one high-level refiner call.

    ``processed_rows`` is the controlled compute quantity in the experiment:
    applying the shared block once to one sample costs one block call.
    ``block_invocations`` records how many batched Python/module invocations
    were needed to perform those calls.
    """

    block_invocations: int = 0
    processed_rows: int = 0

    @property
    def block_calls(self) -> int:
        """Return per-sample shared-block calls (an alias for rows)."""

        return self.processed_rows

    @property
    def row_calls(self) -> int:
        return self.processed_rows

    def as_dict(self) -> dict[str, int]:
        return {
            "block_invocations": self.block_invocations,
            "processed_rows": self.processed_rows,
            "block_calls": self.block_calls,
        }


def _normalise_requested_depths(depths: Iterable[int] | int) -> tuple[int, ...]:
    if isinstance(depths, bool):
        raise ValueError("depth must be an integer in {0, 1, 2, 4}")
    if isinstance(depths, int):
        raw = (depths,)
    else:
        raw = tuple(depths)
    if not raw:
        raise ValueError("at least one exit depth must be requested")
    if any(isinstance(depth, bool) or not isinstance(depth, int) for depth in raw):
        raise ValueError("exit depths must be integers")
    invalid = sorted(set(raw) - set(EXITS))
    if invalid:
        raise ValueError(f"unsupported exit depths {invalid}; allowed exits are {EXITS}")
    return tuple(sorted(set(raw)))


def _selected_depth_tensor(
    selected_depths: Tensor | Sequence[int], *, batch_size: int, device: torch.device
) -> Tensor:
    if isinstance(selected_depths, Tensor):
        if selected_depths.dtype == torch.bool or torch.is_floating_point(selected_depths):
            raise ValueError("selected depths must have an integer dtype")
        selected = selected_depths.to(device=device, dtype=torch.long)
    else:
        values = tuple(selected_depths)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("selected depths must be integers")
        selected = torch.as_tensor(values, dtype=torch.long, device=device)
    if selected.ndim != 1 or selected.shape[0] != batch_size:
        raise ValueError(
            "selected_depths must have shape [batch]; "
            f"got {tuple(selected.shape)} for batch {batch_size}"
        )
    valid = torch.zeros_like(selected, dtype=torch.bool)
    for depth in EXITS:
        valid |= selected == depth
    if not bool(torch.all(valid).item()):
        invalid = torch.unique(selected[~valid]).detach().cpu().tolist()
        raise ValueError(f"unsupported selected depths {invalid}; allowed exits are {EXITS}")
    return selected


def gather_depth_outputs(
    outputs: Mapping[int, Tensor], selected_depths: Tensor | Sequence[int]
) -> Tensor:
    """Gather one dense exit per sample without recomputing the refiner.

    This helper is useful for diagnostics that already require every permitted
    prefix exit. For actual adaptive inference use
    :meth:`SharedResidualRefiner.forward_selected`, which skips inactive rows.
    """

    if not outputs:
        raise ValueError("outputs must contain at least one depth")
    invalid_keys = set(outputs) - set(EXITS)
    if invalid_keys:
        raise ValueError(f"outputs contains unsupported depths {sorted(invalid_keys)}")
    first_depth = next(iter(outputs))
    first = outputs[first_depth]
    if not isinstance(first, Tensor) or first.ndim < 1:
        raise ValueError("each exit output must be a tensor with a batch dimension")
    batch_size = first.shape[0]
    if batch_size <= 0:
        raise ValueError("exit outputs must contain at least one sample")
    selected = _selected_depth_tensor(
        selected_depths, batch_size=batch_size, device=first.device
    )
    required = {depth for depth in EXITS if bool(torch.any(selected == depth).item())}
    missing = required - set(outputs)
    if missing:
        raise ValueError(f"outputs is missing selected depths {sorted(missing)}")
    for depth, value in outputs.items():
        if not isinstance(value, Tensor):
            raise ValueError(f"output at depth {depth} is not a tensor")
        if value.shape != first.shape:
            raise ValueError(
                f"all exit outputs must have shape {tuple(first.shape)}; "
                f"depth {depth} has {tuple(value.shape)}"
            )
        if value.device != first.device or value.dtype != first.dtype:
            raise ValueError("all exit outputs must share device and dtype")

    if len(required) == 1:
        # Preserve object identity as well as bits for an all-depth-zero gather.
        return outputs[next(iter(required))]
    gathered = torch.empty_like(first)
    for depth in EXITS:
        if depth not in required:
            continue
        indices = torch.nonzero(selected == depth, as_tuple=False).flatten()
        gathered = gathered.index_copy(0, indices, outputs[depth].index_select(0, indices))
    return gathered


class SharedResidualRefiner(nn.Module):
    """Action-conditioned refiner with one block shared across all four calls.

    Defaults implement the preregistered Cube architecture: three 192-D latent
    history tokens, three raw 25-D blocked actions, two 256-unit GELU hidden
    layers, and a 192-D residual update. ``hidden_dim`` and other dimensions
    remain configurable to permit small, exact unit tests.
    """

    exits = EXITS

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        action_dim: int = 25,
        history_len: int = 3,
        hidden_dim: int = 256,
        iteration_dim: int = 16,
    ) -> None:
        super().__init__()
        dimensions = {
            "latent_dim": latent_dim,
            "action_dim": action_dim,
            "history_len": history_len,
            "hidden_dim": hidden_dim,
            "iteration_dim": iteration_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.iteration_dim = iteration_dim
        self.max_depth = max(EXITS)

        self.iteration_embedding = nn.Embedding(self.max_depth, iteration_dim)
        block_input_dim = (
            history_len * latent_dim
            + history_len * action_dim
            + latent_dim
            + iteration_dim
        )
        # There is exactly one residual block. It is called recurrently rather
        # than copied into a ModuleList, so every exit uses identical weights.
        self.block = nn.Sequential(
            nn.Linear(block_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Residual-safe initialization: before training, every additional exit
        # exactly preserves the frozen LeWM prediction. This mirrors the
        # zero-initialized residual gating already used by LeWM's AdaLN blocks.
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

        self._last_block_invocations = 0
        self._last_processed_rows = 0
        self._total_block_invocations = 0
        self._total_processed_rows = 0

    @property
    def residual_block(self) -> nn.Module:
        """Alias that makes the shared nature of ``block`` explicit."""

        return self.block

    @property
    def last_compute(self) -> RefinerComputeStats:
        return RefinerComputeStats(
            block_invocations=self._last_block_invocations,
            processed_rows=self._last_processed_rows,
        )

    @property
    def total_compute(self) -> RefinerComputeStats:
        return RefinerComputeStats(
            block_invocations=self._total_block_invocations,
            processed_rows=self._total_processed_rows,
        )

    @property
    def last_refiner_calls(self) -> int:
        """Number of per-sample block calls made by the most recent forward."""

        return self._last_processed_rows

    def reset_compute_counters(self) -> None:
        self._last_block_invocations = 0
        self._last_processed_rows = 0
        self._total_block_invocations = 0
        self._total_processed_rows = 0

    def _begin_call(self) -> None:
        self._last_block_invocations = 0
        self._last_processed_rows = 0

    def _record_block_call(self, rows: int) -> None:
        if rows == 0:
            return
        self._last_block_invocations += 1
        self._last_processed_rows += rows
        self._total_block_invocations += 1
        self._total_processed_rows += rows

    def _validate_inputs(
        self, history: Tensor, action_history: Tensor, base_prediction: Tensor
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
        expected_history_tail = (self.history_len, self.latent_dim)
        expected_action_tail = (self.history_len, self.action_dim)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected_history_tail:
            raise ValueError(
                "history must have shape "
                f"[batch, {self.history_len}, {self.latent_dim}], got {tuple(history.shape)}"
            )
        if action_history.ndim != 3 or tuple(action_history.shape[1:]) != expected_action_tail:
            raise ValueError(
                "action_history must have shape "
                f"[batch, {self.history_len}, {self.action_dim}], "
                f"got {tuple(action_history.shape)}"
            )
        if base_prediction.ndim != 2 or base_prediction.shape[1] != self.latent_dim:
            raise ValueError(
                f"base_prediction must have shape [batch, {self.latent_dim}], "
                f"got {tuple(base_prediction.shape)}"
            )
        batch_size = history.shape[0]
        if batch_size <= 0:
            raise ValueError("batch must contain at least one sample")
        if action_history.shape[0] != batch_size or base_prediction.shape[0] != batch_size:
            raise ValueError("history, action_history, and base_prediction batch sizes differ")
        if not (
            history.device == action_history.device == base_prediction.device
            and history.dtype == action_history.dtype == base_prediction.dtype
        ):
            raise ValueError("history, action_history, and base_prediction must share device and dtype")
        return batch_size

    def _residual_update(
        self,
        history: Tensor,
        action_history: Tensor,
        current_prediction: Tensor,
        iteration: int,
    ) -> Tensor:
        batch_size = current_prediction.shape[0]
        iteration_ids = torch.full(
            (batch_size,), iteration, dtype=torch.long, device=current_prediction.device
        )
        iteration_features = self.iteration_embedding(iteration_ids)
        block_input = torch.cat(
            (
                history.flatten(start_dim=1),
                action_history.flatten(start_dim=1),
                current_prediction,
                iteration_features,
            ),
            dim=-1,
        )
        update = self.block(block_input)
        expected = (batch_size, self.latent_dim)
        if tuple(update.shape) != expected:
            raise RuntimeError(
                f"residual block returned shape {tuple(update.shape)}, expected {expected}"
            )
        self._record_block_call(batch_size)
        return update

    def forward(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
        depths: Iterable[int] | int = EXITS,
        *,
        return_stats: bool = False,
    ) -> dict[int, Tensor] | tuple[dict[int, Tensor], RefinerComputeStats]:
        """Compute dense prefix exits up to the deepest requested depth.

        The value stored at key zero is ``base_prediction`` itself: no clone,
        cast, arithmetic, or refiner call occurs before that exit is recorded.
        """

        self._begin_call()
        batch_size = self._validate_inputs(history, action_history, base_prediction)
        requested = _normalise_requested_depths(depths)
        outputs: dict[int, Tensor] = {}
        if 0 in requested:
            outputs[0] = base_prediction

        deepest = requested[-1]
        current = base_prediction
        for iteration in range(deepest):
            current = current + self._residual_update(
                history, action_history, current, iteration
            )
            depth = iteration + 1
            if depth in requested:
                outputs[depth] = current

        # The variable is deliberately used in the assertion: it guards any
        # future change that could accidentally count something other than rows.
        assert self._last_processed_rows == batch_size * deepest
        if return_stats:
            return outputs, self.last_compute
        return outputs

    def forward_selected(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
        selected_depths: Tensor | Sequence[int],
        *,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, RefinerComputeStats]:
        """Run a per-sample depth allocation while skipping inactive samples.

        At iteration ``k`` only rows whose requested depth is greater than
        ``k`` enter the block. Thus ``processed_rows == selected_depths.sum()``
        exactly, including mixed allocations containing depth zero.
        """

        self._begin_call()
        batch_size = self._validate_inputs(history, action_history, base_prediction)
        selected = _selected_depth_tensor(
            selected_depths, batch_size=batch_size, device=base_prediction.device
        )
        current = base_prediction
        for iteration in range(self.max_depth):
            active_indices = torch.nonzero(
                selected > iteration, as_tuple=False
            ).flatten()
            if active_indices.numel() == 0:
                break
            active_history = history.index_select(0, active_indices)
            active_actions = action_history.index_select(0, active_indices)
            active_current = current.index_select(0, active_indices)
            refined = active_current + self._residual_update(
                active_history, active_actions, active_current, iteration
            )
            current = current.index_copy(0, active_indices, refined)

        expected_calls = int(selected.sum().item())
        if self._last_processed_rows != expected_calls:
            raise RuntimeError(
                "adaptive compute accounting mismatch: "
                f"recorded {self._last_processed_rows}, expected {expected_calls}"
            )
        if return_stats:
            return current, self.last_compute
        return current

    def forward_with_updates(
        self,
        history: Tensor,
        action_history: Tensor,
        base_prediction: Tensor,
    ) -> tuple[dict[int, Tensor], list[Tensor]]:
        """Return all preregistered exits and each of the four residual updates.

        This is the training path for the per-call update penalty. The third
        recurrent state is used to reach depth four, but is intentionally not
        exposed as an evaluation exit.
        """

        self._begin_call()
        batch_size = self._validate_inputs(history, action_history, base_prediction)
        outputs = {0: base_prediction}
        updates: list[Tensor] = []
        current = base_prediction
        for iteration in range(self.max_depth):
            update = self._residual_update(
                history, action_history, current, iteration
            )
            updates.append(update)
            current = current + update
            depth = iteration + 1
            if depth in EXITS:
                outputs[depth] = current

        if tuple(outputs) != EXITS:
            raise RuntimeError(f"internal exit mismatch: got {tuple(outputs)}, expected {EXITS}")
        if self._last_processed_rows != batch_size * self.max_depth:
            raise RuntimeError("dense training compute accounting mismatch")
        return outputs, updates

    @staticmethod
    def gather(
        outputs: Mapping[int, Tensor], selected_depths: Tensor | Sequence[int]
    ) -> Tensor:
        return gather_depth_outputs(outputs, selected_depths)


_FORBIDDEN_FEATURE_ROOTS = frozenset(
    {
        "benefit",
        "contact",
        "error",
        "future",
        "gain",
        "groundtruth",
        "improvement",
        "kinematic",
        "label",
        "loss",
        "next",
        "oracle",
        "outcome",
        "phase",
        "regime",
        "residual",
        "target",
        "truth",
        "y",
    }
)


def _feature_tokens(name: str) -> tuple[str, ...]:
    # Split snake/kebab/space names as well as common camelCase names.
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return tuple(token.lower() for token in re.findall(r"[A-Za-z0-9]+", expanded))


def validate_causal_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Reject names that indicate privileged or target-derived gate inputs.

    The check is intentionally conservative. Target-derived quantities such as
    loss or gain labels are valid ridge *targets*, but never valid gate feature
    columns. Returning the validated tuple makes this easy to enforce at data
    boundaries rather than relying on a comment or naming convention.
    """

    names = tuple(feature_names)
    if not names:
        raise ValueError("feature_names must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("every feature name must be a non-empty string")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    rejected: list[str] = []
    for name in names:
        compact = re.sub(r"[^a-z0-9]", "", name.lower())
        tokens = _feature_tokens(name)
        forbidden = any(
            token == root or token.startswith(root)
            for token in tokens
            for root in _FORBIDDEN_FEATURE_ROOTS
        )
        forbidden |= any(root in compact for root in ("groundtruth", "targetderived"))
        if forbidden:
            rejected.append(name)
    if rejected:
        raise ValueError(
            "forbidden non-causal or target-derived gate feature names: "
            + ", ".join(rejected)
        )
    return names


def causal_gate_feature_names(history_len: int) -> tuple[str, ...]:
    """Return names, in column order, for compact causal summaries."""

    if isinstance(history_len, bool) or not isinstance(history_len, int) or history_len <= 0:
        raise ValueError("history_len must be a positive integer")
    names = [f"history_token_{index}_norm" for index in range(history_len)]
    names.extend(
        f"history_change_{index - 1}_{index}_norm" for index in range(1, history_len)
    )
    names.extend(f"action_token_{index}_norm" for index in range(history_len))
    names.extend(
        f"action_change_{index - 1}_{index}_norm" for index in range(1, history_len)
    )
    names.extend(("base_prediction_norm", "base_prediction_to_last_history_norm"))
    return validate_causal_feature_names(names)


def build_causal_gate_features(
    history: Tensor,
    action_history: Tensor,
    base_prediction: Tensor,
    *,
    return_names: bool = False,
) -> Tensor | tuple[Tensor, tuple[str, ...]]:
    """Build compact prediction-time-only features for the Stage B ridge gate.

    Columns are per-token L2 norms, consecutive history/action change norms,
    the base-prediction norm, and its distance to the last history token. No
    target, future state, refinement result, or post-hoc physical label is an
    argument to this function.
    """

    for name, value in (
        ("history", history),
        ("action_history", action_history),
        ("base_prediction", base_prediction),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not torch.is_floating_point(value):
            raise ValueError(f"{name} must have a floating dtype")
    if history.ndim != 3:
        raise ValueError(f"history must have shape [batch, history, latent], got {tuple(history.shape)}")
    if action_history.ndim != 3:
        raise ValueError(
            "action_history must have shape [batch, history, action], "
            f"got {tuple(action_history.shape)}"
        )
    batch_size, history_len, latent_dim = history.shape
    if batch_size <= 0 or history_len <= 0 or latent_dim <= 0:
        raise ValueError("history dimensions must be nonzero")
    if action_history.shape[0] != batch_size or action_history.shape[1] != history_len:
        raise ValueError("history and action_history batch/history dimensions differ")
    if base_prediction.ndim != 2 or tuple(base_prediction.shape) != (
        batch_size,
        latent_dim,
    ):
        raise ValueError(
            f"base_prediction must have shape [{batch_size}, {latent_dim}], "
            f"got {tuple(base_prediction.shape)}"
        )
    if not (
        history.device == action_history.device == base_prediction.device
        and history.dtype == action_history.dtype == base_prediction.dtype
    ):
        raise ValueError("history, action_history, and base_prediction must share device and dtype")

    history_norms = torch.linalg.vector_norm(history, dim=-1)
    if history_len > 1:
        history_changes = torch.linalg.vector_norm(
            history[:, 1:] - history[:, :-1], dim=-1
        )
        action_changes = torch.linalg.vector_norm(
            action_history[:, 1:] - action_history[:, :-1], dim=-1
        )
    else:
        history_changes = history.new_empty((batch_size, 0))
        action_changes = action_history.new_empty((batch_size, 0))
    action_norms = torch.linalg.vector_norm(action_history, dim=-1)
    prediction_norm = torch.linalg.vector_norm(base_prediction, dim=-1, keepdim=True)
    prediction_history_gap = torch.linalg.vector_norm(
        base_prediction - history[:, -1], dim=-1, keepdim=True
    )
    features = torch.cat(
        (
            history_norms,
            history_changes,
            action_norms,
            action_changes,
            prediction_norm,
            prediction_history_gap,
        ),
        dim=-1,
    )
    names = causal_gate_feature_names(history_len)
    if features.shape[1] != len(names):
        raise RuntimeError("causal feature/name width mismatch")
    if return_names:
        return features, names
    return features


# A concise alias for callers that do not need the longer causal name. The
# implementation still validates and constructs only the frozen causal inputs.
build_gate_features = build_causal_gate_features


__all__ = [
    "EXITS",
    "RefinerComputeStats",
    "SharedResidualRefiner",
    "build_causal_gate_features",
    "build_gate_features",
    "causal_gate_feature_names",
    "gather_depth_outputs",
    "validate_causal_feature_names",
]
