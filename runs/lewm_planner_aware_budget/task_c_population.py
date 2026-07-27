"""Frozen Task C population boundary and prefix-comparable CEM solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Final

import numpy as np
import torch
from gymnasium.spaces import Box
from torch import Tensor


SUPPORTED_POPULATIONS: Final = (64, 128, 300)
SUPPORTED_DEPTHS: Final = (0, 1, 2, 4)
MAX_POPULATION: Final = 300
ELITE_DENOMINATOR: Final = 8
# Dense/conv hooks on the hash-pinned Task A checkpoint, plus explicit SDPA
# QK^T and AV matmul formulas. Activation, normalization, and softmax FLOPs are
# intentionally excluded and named in the report.
ENCODER_COUNTED_FLOPS_PER_ROW: Final = 3_396_609_024
BASE_COUNTED_FLOPS_BY_HISTORY: Final = {
    1: 23_436_488,
    2: 46_922_128,
    3: 70_456_920,
}


def elite_count(population: int) -> int:
    """Task A-preserving 1/8 fraction with integer round-half-up."""
    validate_population(population)
    result = (population * 2 + ELITE_DENOMINATOR) // (2 * ELITE_DENOMINATOR)
    if not 0 < result < population:
        raise ValueError("invalid elite count")
    return result


def validate_population(population: int) -> int:
    if isinstance(population, bool) or not isinstance(population, int):
        raise TypeError("population must be a plain integer")
    if population <= 0:
        raise ValueError("population must be positive")
    if population not in SUPPORTED_POPULATIONS:
        raise ValueError(f"unsupported population {population}")
    return population


@dataclass(frozen=True)
class PopulationConfig:
    population: int
    elites: int
    label: str

    @classmethod
    def create(
        cls, population: int, *, elites: int | None = None, label: str | None = None
    ) -> "PopulationConfig":
        population = validate_population(population)
        expected_elites = elite_count(population)
        if elites is not None:
            if isinstance(elites, bool) or not isinstance(elites, int):
                raise TypeError("elite count must be a plain integer")
            if elites != expected_elites:
                raise ValueError(
                    f"population {population} requires {expected_elites} elites"
                )
        expected_label = f"p{population:03d}"
        if label is not None and label != expected_label:
            raise ValueError(f"population {population} requires label {expected_label}")
        return cls(population, expected_elites, expected_label)

    def assert_solver(self, *, num_samples: int, topk: int) -> None:
        if num_samples != self.population or topk != self.elites:
            raise ValueError("population/top-k inconsistency")


def call_seed(candidate_seed: int, start_id: int, replan_index: int) -> int:
    for name, value in (
        ("candidate_seed", candidate_seed),
        ("start_id", start_id),
        ("replan_index", replan_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative plain integer")
    # Explicit, stable counter ownership. Keep the result in signed int64 range.
    return int(
        (
            candidate_seed * 6_364_136_223_846_793_005
            + start_id * 1_442_695_040_888_963_407
            + replan_index * 22_695_477
        )
        % (2**63 - 1)
    )


def innovation_stream(
    *,
    seed: int,
    iterations: int,
    batch: int,
    horizon: int,
    action_dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Draw the frozen maximum-population stream without reading global RNG."""
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (iterations, batch, horizon, action_dim)
    ):
        raise ValueError("innovation dimensions must be positive plain integers")
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        iterations,
        batch,
        MAX_POPULATION,
        horizon,
        action_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )


@dataclass
class CEMWorkLedger:
    population: int
    elite_count: int
    iterations: int
    horizon: int
    replans: int = 0
    candidate_sequences_evaluated: int = 0
    predicted_transition_rows: int = 0
    sampling_invocations: int = 0
    update_invocations: int = 0
    topk_invocations: int = 0
    innovation_tensor_shapes: list[list[int]] = field(default_factory=list)
    candidate_tensor_shapes: list[list[int]] = field(default_factory=list)
    topk_tensor_shapes: list[list[int]] = field(default_factory=list)

    def record_iteration(self, batch: int, action_dim: int) -> None:
        self.candidate_sequences_evaluated += batch * self.population
        self.predicted_transition_rows += batch * self.population * self.horizon
        self.sampling_invocations += 1
        self.update_invocations += 1
        self.topk_invocations += 1
        self.innovation_tensor_shapes.append(
            [batch, MAX_POPULATION, self.horizon, action_dim]
        )
        self.candidate_tensor_shapes.append(
            [batch, self.population, self.horizon, action_dim]
        )
        self.topk_tensor_shapes.append(
            [batch, self.elite_count, self.horizon, action_dim]
        )

    def validate(self, *, batch: int = 1) -> None:
        expected_candidates = self.replans * self.iterations * batch * self.population
        expected_rows = expected_candidates * self.horizon
        expected_invocations = self.replans * self.iterations
        if self.candidate_sequences_evaluated != expected_candidates:
            raise RuntimeError("candidate sequence ledger mismatch")
        if self.predicted_transition_rows != expected_rows:
            raise RuntimeError("predicted transition row ledger mismatch")
        if not all(
            value == expected_invocations
            for value in (
                self.sampling_invocations,
                self.update_invocations,
                self.topk_invocations,
                len(self.innovation_tensor_shapes),
                len(self.candidate_tensor_shapes),
                len(self.topk_tensor_shapes),
            )
        ):
            raise RuntimeError("CEM invocation ledger mismatch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrefixComparableCEMSolver:
    """Task A CEM mechanics with a maximum-population prefix innovation stream."""

    def __init__(
        self,
        *,
        model: Any,
        population: PopulationConfig,
        batch_size: int = 1,
        n_steps: int = 20,
        device: torch.device | str = "cpu",
        candidate_seed: int = 26072610,
    ) -> None:
        if batch_size != 1:
            raise ValueError("Task C freezes batch size 1")
        if n_steps != 20:
            raise ValueError("Task C freezes 20 CEM iterations")
        population.assert_solver(
            num_samples=population.population, topk=population.elites
        )
        self.model = model
        self.population_config = population
        self.num_samples = population.population
        self.topk = population.elites
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.device = torch.device(device)
        self.candidate_seed = candidate_seed
        self._dtype = next(model.parameters()).dtype
        self.ledger = CEMWorkLedger(
            population=self.num_samples,
            elite_count=self.topk,
            iterations=self.n_steps,
            horizon=5,
        )
        self.last_innovations: list[Tensor] = []

    def configure(self, *, action_space: Box, n_envs: int, config: Any) -> None:
        if not isinstance(action_space, Box):
            raise TypeError("Task C requires a continuous Box action space")
        if n_envs != 1:
            raise ValueError("Task C freezes one environment")
        if (
            config.horizon != 5
            or config.receding_horizon != 5
            or config.action_block != 5
        ):
            raise ValueError("Task C planner horizon/receding-horizon/action-block mismatch")
        if action_space.shape != (1, 2):
            raise ValueError("Task C requires PushT action shape (1,2)")
        if not (
            np.all(np.asarray(action_space.low) == -1)
            and np.all(np.asarray(action_space.high) == 1)
        ):
            raise ValueError("Task C requires PushT action bounds [-1,1]")
        self.horizon = 5
        self.action_dim = 10
        self._configured = True

    @torch.inference_mode()
    def solve(
        self,
        info_dict: dict,
        init_action: Tensor | None = None,
        *,
        start_id: int,
        replan_index: int,
    ) -> dict:
        if not getattr(self, "_configured", False):
            raise RuntimeError("solver must be configured")
        total_envs = len(next(iter(info_dict.values())))
        if total_envs != 1:
            raise ValueError("Task C solver accepts one environment")
        if init_action is None:
            mean = torch.zeros(
                total_envs, self.horizon, self.action_dim, dtype=self._dtype
            )
        else:
            if init_action.ndim != 3 or init_action.shape[0] != total_envs:
                raise ValueError("invalid warm-start action shape")
            mean = init_action
            remaining = self.horizon - mean.shape[1]
            if remaining < 0:
                raise ValueError("warm start exceeds horizon")
            if remaining:
                mean = torch.cat(
                    [
                        mean,
                        torch.zeros(
                            total_envs,
                            remaining,
                            self.action_dim,
                            dtype=self._dtype,
                            device=mean.device,
                        ),
                    ],
                    dim=1,
                )
        mean = mean.to(self.device)
        variance = torch.ones_like(mean)
        expanded = {
            key: (
                value.to(
                    device=self.device,
                    dtype=self._dtype if value.is_floating_point() else None,
                )
                .unsqueeze(1)
                .expand(1, self.num_samples, *value.shape[1:])
                if torch.is_tensor(value)
                else np.repeat(value[:, None, ...], self.num_samples, axis=1)
            )
            for key, value in info_dict.items()
        }
        stream = innovation_stream(
            seed=call_seed(self.candidate_seed, start_id, replan_index),
            iterations=self.n_steps,
            batch=1,
            horizon=self.horizon,
            action_dim=self.action_dim,
            device=self.device,
            dtype=self._dtype,
        )
        self.last_innovations = []
        final_cost = None
        for iteration in range(self.n_steps):
            innovation = stream[iteration, :, : self.num_samples]
            self.last_innovations.append(innovation.detach().cpu())
            candidates = innovation * variance.unsqueeze(1) + mean.unsqueeze(1)
            candidates[:, 0] = mean
            costs = self.model.get_cost(expanded, candidates)
            if costs.shape != (1, self.num_samples) or not bool(
                torch.isfinite(costs).all()
            ):
                raise RuntimeError("invalid CEM candidate costs")
            topk_values, topk_indices = torch.topk(
                costs, k=self.topk, dim=1, largest=False
            )
            batch_indices = torch.arange(1, device=self.device).unsqueeze(1).expand(
                -1, self.topk
            )
            elites = candidates[batch_indices, topk_indices]
            mean = elites.mean(dim=1)
            variance = elites.std(dim=1)
            if not bool(torch.isfinite(mean).all() & torch.isfinite(variance).all()):
                raise RuntimeError("nonfinite CEM distribution")
            final_cost = topk_values.mean(dim=1)
            self.ledger.record_iteration(1, self.action_dim)
        self.ledger.replans += 1
        self.ledger.validate()
        return {
            "actions": mean.detach().cpu(),
            "costs": final_cost.detach().cpu().tolist(),
            "mean": [mean.detach().cpu()],
            "var": [variance.detach().cpu()],
        }


def refiner_stage_counted_flops_per_row() -> int:
    """Dense linear multiply/add plus the explicit residual addition."""
    width, hidden, latent = 817, 256, 192
    return 2 * (width * hidden + hidden * hidden + hidden * latent) + latent


def cem_counted_flops_per_call(population: int, elites: int) -> dict[str, int]:
    """Exact tensor arithmetic counted by Task C; exclusions remain explicit."""
    validate_population(population)
    if elites != elite_count(population):
        raise ValueError("elite mismatch")
    iterations, horizon, action_dim = 20, 5, 10
    candidate_scalars = population * iterations * horizon * action_dim
    elite_scalars = elites * iterations * horizon * action_dim
    # candidate scale + shift: multiply and add
    sampling_transform = 2 * candidate_scalars
    # mean: K-1 adds + one division; unbiased std: subtract, square, K-1
    # adds, division, sqrt. Both are applied per output scalar.
    output_scalars = iterations * horizon * action_dim
    mean = elites * output_scalars
    std = (3 * elites + 1) * output_scalars
    return {
        "sampling_scale_shift": sampling_transform,
        "elite_mean": mean,
        "elite_std": std,
        "cem_arithmetic_total": sampling_transform + mean + std,
        "terminal_squared_error": population * iterations * (192 * 3 - 1),
        "topk_comparisons": 0,
    }


def counted_flops_per_call(population: int, elites: int, depth: int) -> dict[str, int]:
    validate_population(population)
    if depth not in SUPPORTED_DEPTHS or isinstance(depth, bool):
        raise ValueError("unsupported fixed refinement depth")
    if elites != elite_count(population):
        raise ValueError("elite mismatch")
    iterations = 20
    encoding = 2 * iterations * ENCODER_COUNTED_FLOPS_PER_ROW
    per_candidate_rollout = (
        BASE_COUNTED_FLOPS_BY_HISTORY[1]
        + BASE_COUNTED_FLOPS_BY_HISTORY[2]
        + 3 * BASE_COUNTED_FLOPS_BY_HISTORY[3]
    )
    base_prediction = population * iterations * per_candidate_rollout
    stage_rows = population * iterations * 5
    refiner_stages = [
        stage_rows * refiner_stage_counted_flops_per_row()
        if stage < depth
        else 0
        for stage in range(4)
    ]
    cem = cem_counted_flops_per_call(population, elites)
    components = {
        "image_encoder": iterations * ENCODER_COUNTED_FLOPS_PER_ROW,
        "goal_encoder": iterations * ENCODER_COUNTED_FLOPS_PER_ROW,
        "base_prediction": base_prediction,
        "refiner_stage_1": refiner_stages[0],
        "refiner_stage_2": refiner_stages[1],
        "refiner_stage_3": refiner_stages[2],
        "refiner_stage_4": refiner_stages[3],
        "terminal_squared_error": cem["terminal_squared_error"],
        "cem_sampling_scale_shift": cem["sampling_scale_shift"],
        "cem_elite_mean": cem["elite_mean"],
        "cem_elite_std": cem["elite_std"],
    }
    components["counted_total"] = sum(components.values())
    return components
