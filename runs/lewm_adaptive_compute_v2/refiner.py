"""Frozen V1 shared refiner and preregistered V2 post-call features.

The refiner architecture intentionally retains the exact V1 state-dict key
layout.  V2 never trains it: :func:`load_frozen_v1_refiner` verifies the
checkpoint hash and exact parameter keys, loads strictly, and freezes every
parameter.  Sequential inference is split into the universal first call and a
continuation call that starts at recurrent iteration one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn


DENSE_EXITS = (0, 1, 2, 4)
FINAL_DEPTHS = (1, 2, 4)
EXPECTED_V1_REFINER_SHA256 = (
    "388a82fc30c96921083bfa4f2578cb296e6c3544510953d17578cf766cdf10f1"
)
EXPECTED_V1_SEED = 260713
EXPECTED_PARAMETER_COUNT = 335_360


@dataclass(frozen=True)
class RefinerComputeStats:
    """Exact shared-block compute performed by one public inference call."""

    block_invocations: int = 0
    processed_rows: int = 0

    @property
    def block_calls(self) -> int:
        return self.processed_rows

    def as_dict(self) -> dict[str, int]:
        return {
            "block_invocations": self.block_invocations,
            "processed_rows": self.processed_rows,
            "block_calls": self.block_calls,
        }


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def hash_state_tensors(state: Mapping[str, Tensor]) -> str:
    """Hash ordered names, tensor metadata, and exact contiguous CPU bytes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"state value {name!r} is not a tensor")
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(repr(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _normalise_dense_depths(depths: Iterable[int] | int) -> tuple[int, ...]:
    if isinstance(depths, bool):
        raise ValueError("depth must be an integer")
    raw = (depths,) if isinstance(depths, int) else tuple(depths)
    if not raw:
        raise ValueError("at least one depth is required")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError("depths must be integers")
    invalid = sorted(set(raw) - set(DENSE_EXITS))
    if invalid:
        raise ValueError(f"unsupported depths {invalid}; allowed={DENSE_EXITS}")
    return tuple(sorted(set(raw)))


def _final_depth_tensor(
    final_depths: Tensor | Sequence[int], *, batch_size: int, device: torch.device
) -> Tensor:
    if isinstance(final_depths, Tensor):
        if final_depths.dtype == torch.bool or torch.is_floating_point(final_depths):
            raise ValueError("final depths must have integer dtype")
        selected = final_depths.to(device=device, dtype=torch.long)
    else:
        values = tuple(final_depths)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("final depths must be integers")
        selected = torch.as_tensor(values, dtype=torch.long, device=device)
    if selected.ndim != 1 or selected.shape[0] != batch_size:
        raise ValueError(
            f"final_depths must have shape [{batch_size}], got {tuple(selected.shape)}"
        )
    valid = (selected == 1) | (selected == 2) | (selected == 4)
    if not bool(torch.all(valid).item()):
        invalid = torch.unique(selected[~valid]).detach().cpu().tolist()
        raise ValueError(f"unsupported final depths {invalid}; allowed={FINAL_DEPTHS}")
    return selected


class SharedResidualRefiner(nn.Module):
    """The exact V1 shared recurrent residual-block architecture."""

    exits = DENSE_EXITS
    final_depths = FINAL_DEPTHS

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
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dimensions.values()):
            raise ValueError(f"all dimensions must be positive integers: {dimensions}")
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.iteration_dim = iteration_dim
        self.max_depth = 4

        # Names and ordering must remain byte-for-byte checkpoint-compatible.
        self.iteration_embedding = nn.Embedding(self.max_depth, iteration_dim)
        input_dim = history_len * latent_dim + history_len * action_dim + latent_dim + iteration_dim
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._last_block_invocations = 0
        self._last_processed_rows = 0
        self._total_block_invocations = 0
        self._total_processed_rows = 0

    @property
    def last_compute(self) -> RefinerComputeStats:
        return RefinerComputeStats(self._last_block_invocations, self._last_processed_rows)

    @property
    def total_compute(self) -> RefinerComputeStats:
        return RefinerComputeStats(self._total_block_invocations, self._total_processed_rows)

    def reset_compute_counters(self) -> None:
        self._last_block_invocations = 0
        self._last_processed_rows = 0
        self._total_block_invocations = 0
        self._total_processed_rows = 0

    def _begin_call(self) -> None:
        self._last_block_invocations = 0
        self._last_processed_rows = 0

    def _record(self, rows: int) -> None:
        if rows <= 0:
            return
        self._last_block_invocations += 1
        self._last_processed_rows += rows
        self._total_block_invocations += 1
        self._total_processed_rows += rows

    def _validate_inputs(self, history: Tensor, action_history: Tensor, prediction: Tensor) -> int:
        for name, value in (
            ("history", history),
            ("action_history", action_history),
            ("prediction", prediction),
        ):
            if not isinstance(value, Tensor) or not torch.is_floating_point(value):
                raise ValueError(f"{name} must be a floating torch.Tensor")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains nonfinite values")
        expected_history = (self.history_len, self.latent_dim)
        expected_action = (self.history_len, self.action_dim)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected_history:
            raise ValueError(f"history must have shape [batch, {expected_history[0]}, {expected_history[1]}]")
        if action_history.ndim != 3 or tuple(action_history.shape[1:]) != expected_action:
            raise ValueError(f"action_history must have shape [batch, {expected_action[0]}, {expected_action[1]}]")
        if prediction.ndim != 2 or prediction.shape[1] != self.latent_dim:
            raise ValueError(f"prediction must have shape [batch, {self.latent_dim}]")
        batch = history.shape[0]
        if batch <= 0 or action_history.shape[0] != batch or prediction.shape[0] != batch:
            raise ValueError("inputs must have the same nonzero batch size")
        if not (
            history.device == action_history.device == prediction.device
            and history.dtype == action_history.dtype == prediction.dtype
        ):
            raise ValueError("inputs must share device and dtype")
        return batch

    def _residual_update(
        self, history: Tensor, action_history: Tensor, current: Tensor, iteration: int
    ) -> Tensor:
        if iteration not in range(self.max_depth):
            raise ValueError(f"iteration must be in [0, {self.max_depth})")
        iteration_ids = torch.full(
            (current.shape[0],), iteration, dtype=torch.long, device=current.device
        )
        block_input = torch.cat(
            (
                history.flatten(start_dim=1),
                action_history.flatten(start_dim=1),
                current,
                self.iteration_embedding(iteration_ids),
            ),
            dim=1,
        )
        update = self.block(block_input)
        if update.shape != current.shape:
            raise RuntimeError("refiner update shape mismatch")
        self._record(current.shape[0])
        return update

    def forward(
        self,
        history: Tensor,
        action_history: Tensor,
        z0: Tensor,
        depths: Iterable[int] | int = DENSE_EXITS,
        *,
        return_stats: bool = False,
    ) -> dict[int, Tensor] | tuple[dict[int, Tensor], RefinerComputeStats]:
        """Compute dense prefix exits from z0, preserving depth-zero identity."""

        self._begin_call()
        batch = self._validate_inputs(history, action_history, z0)
        requested = _normalise_dense_depths(depths)
        outputs: dict[int, Tensor] = {0: z0} if 0 in requested else {}
        current = z0
        for iteration in range(requested[-1]):
            current = current + self._residual_update(history, action_history, current, iteration)
            if iteration + 1 in requested:
                outputs[iteration + 1] = current
        if self._last_processed_rows != batch * requested[-1]:
            raise RuntimeError("dense compute accounting mismatch")
        return (outputs, self.last_compute) if return_stats else outputs

    def forward_first(
        self,
        history: Tensor,
        action_history: Tensor,
        z0: Tensor,
        *,
        return_update: bool = False,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, RefinerComputeStats] | tuple[Tensor, Tensor, RefinerComputeStats]:
        """Execute the mandatory first call for every row exactly once."""

        self._begin_call()
        batch = self._validate_inputs(history, action_history, z0)
        update = self._residual_update(history, action_history, z0, iteration=0)
        z1 = z0 + update
        if self._last_processed_rows != batch or self._last_block_invocations != 1:
            raise RuntimeError("first-call accounting mismatch")
        if return_update and return_stats:
            return z1, update, self.last_compute
        if return_update:
            return z1, update
        if return_stats:
            return z1, self.last_compute
        return z1

    def continue_selected(
        self,
        history: Tensor,
        action_history: Tensor,
        z1: Tensor,
        final_depths: Tensor | Sequence[int],
        *,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, RefinerComputeStats]:
        """Continue from z1 at iteration one, pruning rows at their final depth."""

        self._begin_call()
        batch = self._validate_inputs(history, action_history, z1)
        selected = _final_depth_tensor(final_depths, batch_size=batch, device=z1.device)
        current = z1
        # iteration 1 reaches depth 2; iterations 2 and 3 reach depth 3 and 4.
        for iteration in range(1, self.max_depth):
            active = torch.nonzero(selected > iteration, as_tuple=False).flatten()
            if active.numel() == 0:
                break
            active_current = current.index_select(0, active)
            refined = active_current + self._residual_update(
                history.index_select(0, active),
                action_history.index_select(0, active),
                active_current,
                iteration,
            )
            current = current.index_copy(0, active, refined)
        expected = int((selected - 1).sum().item())
        if self._last_processed_rows != expected:
            raise RuntimeError(
                f"continuation accounting mismatch: got {self._last_processed_rows}, expected {expected}"
            )
        return (current, self.last_compute) if return_stats else current


def validate_frozen_refiner(model: SharedResidualRefiner) -> None:
    if model.training:
        raise RuntimeError("frozen refiner is not in evaluation mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen refiner has trainable parameters")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen refiner has gradients")


def load_frozen_v1_refiner(
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
    expected_sha256: str = EXPECTED_V1_REFINER_SHA256,
) -> tuple[SharedResidualRefiner, dict[str, object]]:
    """Strictly load and freeze the preregistered V1-selected refiner."""

    path = Path(checkpoint_path)
    observed_hash = sha256_file(path)
    if observed_hash != expected_sha256:
        raise RuntimeError(
            f"V1 refiner checkpoint hash mismatch: expected {expected_sha256}, got {observed_hash}"
        )
    try:
        stored = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older local torch
        stored = torch.load(path, map_location="cpu")
    if not isinstance(stored, dict) or set(stored) != {"state_dict", "seed"}:
        raise RuntimeError("V1 refiner checkpoint must contain exactly state_dict and seed")
    if int(stored["seed"]) != EXPECTED_V1_SEED:
        raise RuntimeError(f"unexpected V1 refiner seed {stored['seed']!r}")
    state = stored["state_dict"]
    if not isinstance(state, Mapping) or not all(isinstance(value, Tensor) for value in state.values()):
        raise RuntimeError("V1 refiner state_dict is invalid")

    model = SharedResidualRefiner()
    expected_keys = set(model.state_dict())
    observed_keys = set(state)
    if observed_keys != expected_keys:
        raise RuntimeError(
            "V1 refiner strict key mismatch: "
            f"missing={sorted(expected_keys - observed_keys)}, unexpected={sorted(observed_keys - expected_keys)}"
        )
    if not all(bool(torch.isfinite(value).all().item()) for value in state.values()):
        raise RuntimeError("V1 refiner checkpoint contains nonfinite tensors")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("V1 refiner strict load unexpectedly reported key differences")
    model.requires_grad_(False).eval().to(device)
    validate_frozen_refiner(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"unexpected refiner parameter count {parameter_count}")
    provenance: dict[str, object] = {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": observed_hash,
        "state_tensor_sha256": hash_state_tensors(model.state_dict()),
        "seed": EXPECTED_V1_SEED,
        "parameter_count": parameter_count,
        "all_parameters_frozen": True,
        "evaluation_mode": True,
        "strict_key_count": len(expected_keys),
    }
    return model, provenance


def post_call_feature_names(
    history_len: int = 3, latent_dim: int = 192, action_dim: int = 25
) -> tuple[str, ...]:
    """Return the frozen V2 whitelist in exact feature-column order."""

    if (history_len, latent_dim, action_dim) != (3, 192, 25):
        raise ValueError("V2 feature contract is frozen to history=3, latent=192, action=25")
    names: list[str] = []
    names.extend(
        f"history_token_{token}_component_{component}"
        for token in range(history_len)
        for component in range(latent_dim)
    )
    names.extend(
        f"action_block_{token}_component_{component}"
        for token in range(history_len)
        for component in range(action_dim)
    )
    names.extend(f"z0_component_{component}" for component in range(latent_dim))
    names.extend(f"z1_component_{component}" for component in range(latent_dim))
    names.extend(f"first_update_component_{component}" for component in range(latent_dim))
    names.extend(f"history_token_{token}_norm" for token in range(history_len))
    names.extend(f"history_difference_{token - 1}_{token}_norm" for token in range(1, history_len))
    names.extend(f"action_block_{token}_norm" for token in range(history_len))
    names.extend(f"action_difference_{token - 1}_{token}_norm" for token in range(1, history_len))
    names.extend(
        (
            "z0_norm",
            "z1_norm",
            "first_update_norm",
            "first_update_relative_to_z0",
            "z0_z1_cosine_alignment",
            "z0_first_update_cosine_alignment",
            "z1_first_update_cosine_alignment",
            "first_update_last_history_gap_cosine_alignment",
            "z0_last_history_distance",
            "z1_last_history_distance",
            "last_history_distance_relative_reduction",
        )
    )
    result = tuple(names)
    if len(result) != 1_248 or len(set(result)) != len(result):
        raise RuntimeError("internal frozen feature whitelist mismatch")
    return result


def validate_causal_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Accept only the complete frozen V2 whitelist, in exact order."""

    names = tuple(feature_names)
    expected = post_call_feature_names()
    if names != expected:
        raise ValueError("feature names do not exactly match the frozen V2 causal whitelist and order")
    return names


def _cosine(a: Tensor, b: Tensor, epsilon: float = 1e-6) -> Tensor:
    numerator = torch.sum(a * b, dim=1, keepdim=True)
    denominator = (
        torch.linalg.vector_norm(a, dim=1, keepdim=True)
        * torch.linalg.vector_norm(b, dim=1, keepdim=True)
    ).clamp_min(epsilon)
    return numerator / denominator


def build_post_call_features(
    history: Tensor,
    action_history: Tensor,
    z0: Tensor,
    z1: Tensor,
    *,
    return_names: bool = False,
) -> Tensor | tuple[Tensor, tuple[str, ...]]:
    """Build the exact target-free feature vector after the mandatory first call.

    Deliberately, the signature has no target, loss, future state, label,
    physical regime, phase, or split-outcome input.
    """

    for name, value in (("history", history), ("action_history", action_history), ("z0", z0), ("z1", z1)):
        if not isinstance(value, Tensor) or not torch.is_floating_point(value):
            raise ValueError(f"{name} must be a floating torch.Tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains nonfinite values")
    if history.ndim != 3 or tuple(history.shape[1:]) != (3, 192):
        raise ValueError("history must have frozen shape [batch, 3, 192]")
    batch = history.shape[0]
    if action_history.ndim != 3 or tuple(action_history.shape) != (batch, 3, 25):
        raise ValueError("action_history must have frozen shape [batch, 3, 25]")
    if z0.shape != (batch, 192) or z1.shape != (batch, 192):
        raise ValueError("z0 and z1 must have frozen shape [batch, 192]")
    if batch <= 0:
        raise ValueError("feature batch must be nonempty")
    if not (
        history.device == action_history.device == z0.device == z1.device
        and history.dtype == action_history.dtype == z0.dtype == z1.dtype
    ):
        raise ValueError("feature inputs must share device and dtype")

    epsilon = 1e-6
    delta1 = z1 - z0
    history_norms = torch.linalg.vector_norm(history, dim=2)
    history_differences = torch.linalg.vector_norm(history[:, 1:] - history[:, :-1], dim=2)
    action_norms = torch.linalg.vector_norm(action_history, dim=2)
    action_differences = torch.linalg.vector_norm(
        action_history[:, 1:] - action_history[:, :-1], dim=2
    )
    z0_norm = torch.linalg.vector_norm(z0, dim=1, keepdim=True)
    z1_norm = torch.linalg.vector_norm(z1, dim=1, keepdim=True)
    update_norm = torch.linalg.vector_norm(delta1, dim=1, keepdim=True)
    update_relative = update_norm / (z0_norm + epsilon)
    last_history_gap = z0 - history[:, -1]
    dist0 = torch.linalg.vector_norm(last_history_gap, dim=1, keepdim=True)
    dist1 = torch.linalg.vector_norm(z1 - history[:, -1], dim=1, keepdim=True)
    distance_reduction = (dist0 - dist1) / (dist0 + epsilon)

    features = torch.cat(
        (
            history.flatten(start_dim=1),
            action_history.flatten(start_dim=1),
            z0,
            z1,
            delta1,
            history_norms,
            history_differences,
            action_norms,
            action_differences,
            z0_norm,
            z1_norm,
            update_norm,
            update_relative,
            _cosine(z0, z1),
            _cosine(z0, delta1),
            _cosine(z1, delta1),
            _cosine(delta1, last_history_gap),
            dist0,
            dist1,
            distance_reduction,
        ),
        dim=1,
    )
    names = validate_causal_feature_names(post_call_feature_names())
    if features.shape != (batch, len(names)) or not bool(torch.isfinite(features).all().item()):
        raise RuntimeError("post-call features violate frozen shape/finite contract")
    return (features, names) if return_names else features


# Concise aliases for experiment-driver call sites.
build_causal_post_call_features = build_post_call_features
load_frozen_refiner = load_frozen_v1_refiner


__all__ = [
    "DENSE_EXITS",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_V1_REFINER_SHA256",
    "FINAL_DEPTHS",
    "RefinerComputeStats",
    "SharedResidualRefiner",
    "build_causal_post_call_features",
    "build_post_call_features",
    "hash_state_tensors",
    "load_frozen_refiner",
    "load_frozen_v1_refiner",
    "post_call_feature_names",
    "sha256_file",
    "validate_causal_feature_names",
    "validate_frozen_refiner",
]
