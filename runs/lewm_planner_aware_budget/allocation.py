"""Exact hard-budget scheduling for joint model/planner compute actions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class ComputeAction:
    refinement_depth: int
    planner_population: int

    def __post_init__(self) -> None:
        if isinstance(self.refinement_depth, bool) or self.refinement_depth < 1:
            raise ValueError("refinement_depth must be a positive integer")
        if int(self.refinement_depth) != self.refinement_depth:
            raise ValueError("refinement_depth must be a positive integer")
        if isinstance(self.planner_population, bool) or self.planner_population < 1:
            raise ValueError("planner_population must be a positive integer")
        if int(self.planner_population) != self.planner_population:
            raise ValueError("planner_population must be a positive integer")


@dataclass(frozen=True)
class AllocationResult:
    action_indices: np.ndarray
    total_cost: int
    budget: int
    total_value: float
    abstained: np.ndarray

    @property
    def budget_slack(self) -> int:
        return self.budget - self.total_cost


def action_grid(
    refinement_depths: Sequence[int], planner_populations: Sequence[int]
) -> tuple[ComputeAction, ...]:
    """Return every fixed depth × planner-population baseline."""

    depths = tuple(int(value) for value in refinement_depths)
    populations = tuple(int(value) for value in planner_populations)
    if not depths or not populations:
        raise ValueError("both action axes must be nonempty")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("refinement_depths must be unique and increasing")
    if tuple(sorted(set(populations))) != populations:
        raise ValueError("planner_populations must be unique and increasing")
    return tuple(ComputeAction(depth, population) for depth, population in product(depths, populations))


def _positive_integer_vector(values: Sequence[int], length: int, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.shape != (length,) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain one positive integer per action")
    result = result.astype(np.int64, copy=False)
    if np.any(result <= 0):
        raise ValueError(f"{name} must contain one positive integer per action")
    return result


def conservative_values(
    predicted_value: np.ndarray,
    uncertainty: np.ndarray,
    *,
    uncertainty_penalty: float,
) -> np.ndarray:
    """Convert value/uncertainty estimates to frozen lower-confidence values."""

    values = np.asarray(predicted_value, dtype=np.float64)
    errors = np.asarray(uncertainty, dtype=np.float64)
    penalty = float(uncertainty_penalty)
    if values.ndim != 2 or errors.shape != values.shape:
        raise ValueError("predicted_value and uncertainty must share shape [decisions, actions]")
    if not np.isfinite(values).all() or not np.isfinite(errors).all():
        raise ValueError("value inputs must be finite")
    if np.any(errors < 0) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("uncertainty and uncertainty_penalty must be nonnegative")
    return values - penalty * errors


def trust_mask(
    uncertainty: np.ndarray,
    disagreement: np.ndarray,
    *,
    max_uncertainty: float,
    max_disagreement: float,
    fallback_index: int,
) -> np.ndarray:
    """Fail closed to a fallback when no optional compute action is trusted."""

    errors = np.asarray(uncertainty, dtype=np.float64)
    disagreement_values = np.asarray(disagreement, dtype=np.float64)
    if errors.ndim != 2 or disagreement_values.shape != errors.shape:
        raise ValueError("trust inputs must share shape [decisions, actions]")
    if not np.isfinite(errors).all() or not np.isfinite(disagreement_values).all():
        raise ValueError("trust inputs must be finite")
    if np.any(errors < 0) or np.any(disagreement_values < 0):
        raise ValueError("trust inputs must be nonnegative")
    if fallback_index < 0 or fallback_index >= errors.shape[1]:
        raise ValueError("fallback_index is out of range")
    mask = (errors <= float(max_uncertainty)) & (
        disagreement_values <= float(max_disagreement)
    )
    mask[:, fallback_index] = True
    return mask


def allocate_hard_budget(
    conservative_value: np.ndarray,
    action_costs: Sequence[int],
    *,
    budget: int,
    eligible: np.ndarray | None = None,
    fallback_index: int = 0,
) -> AllocationResult:
    """Solve a multiple-choice knapsack with one joint action per decision.

    Ties prefer lower action indices at earlier decisions, making schedules
    deterministic.  The budget is an upper bound; the returned slack is
    explicit.
    """

    values = np.asarray(conservative_value, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("conservative_value must be a finite [decisions, actions] matrix")
    n_decisions, n_actions = values.shape
    costs = _positive_integer_vector(action_costs, n_actions, "action_costs")
    if isinstance(budget, bool) or int(budget) != budget or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    budget = int(budget)
    if fallback_index < 0 or fallback_index >= n_actions:
        raise ValueError("fallback_index is out of range")

    if eligible is None:
        allowed = np.ones(values.shape, dtype=bool)
    else:
        allowed = np.asarray(eligible, dtype=bool)
        if allowed.shape != values.shape:
            raise ValueError("eligible must match conservative_value shape")
    allowed[:, fallback_index] = True

    minimum = n_decisions * int(costs[fallback_index])
    if minimum > budget:
        raise ValueError("budget cannot fund the mandatory fallback schedule")

    # Suffix DP permits deterministic forward reconstruction.
    suffix = np.full((n_decisions + 1, budget + 1), -np.inf, dtype=np.float64)
    suffix[n_decisions, :] = 0.0
    choices = np.full((n_decisions, budget + 1), -1, dtype=np.int64)
    for row in range(n_decisions - 1, -1, -1):
        for capacity in range(budget + 1):
            best_value = -np.inf
            best_action = -1
            for action in range(n_actions):
                cost = int(costs[action])
                if not allowed[row, action] or cost > capacity:
                    continue
                candidate = values[row, action] + suffix[row + 1, capacity - cost]
                if candidate > best_value:
                    best_value = candidate
                    best_action = action
            suffix[row, capacity] = best_value
            choices[row, capacity] = best_action

    if not np.isfinite(suffix[0, budget]):
        raise ValueError("no eligible schedule fits the hard budget")
    selected = np.empty(n_decisions, dtype=np.int64)
    remaining = budget
    for row in range(n_decisions):
        action = int(choices[row, remaining])
        if action < 0:
            raise RuntimeError("hard-budget backtracking failed")
        selected[row] = action
        remaining -= int(costs[action])

    total_cost = int(costs[selected].sum(dtype=np.int64)) if n_decisions else 0
    total_value = float(values[np.arange(n_decisions), selected].sum())
    abstained = selected == fallback_index
    return AllocationResult(selected, total_cost, budget, total_value, abstained)


def fixed_pair_cost_table(
    actions: Sequence[ComputeAction],
    counted_flops: Sequence[int],
    wall_clock_ns: Sequence[int],
) -> list[dict[str, int]]:
    """Build the required complete, measured fixed-pair baseline table."""

    action_tuple = tuple(actions)
    flop_costs = _positive_integer_vector(counted_flops, len(action_tuple), "counted_flops")
    latency_costs = _positive_integer_vector(wall_clock_ns, len(action_tuple), "wall_clock_ns")
    return [
        {
            "refinement_depth": action.refinement_depth,
            "planner_population": action.planner_population,
            "counted_flops": int(flop_costs[index]),
            "wall_clock_ns": int(latency_costs[index]),
        }
        for index, action in enumerate(action_tuple)
    ]
