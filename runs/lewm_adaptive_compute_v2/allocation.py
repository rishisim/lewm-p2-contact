"""Sequential exact-budget allocation and statistics for LeWM V2.

Every transition has already executed one frozen-refiner call.  The available
terminal depths are therefore ``(1, 2, 4)``, with additional continuation
costs ``(0, 1, 3)``.  The confirmatory budget is exactly ``N`` continuation
calls, or ``2N`` total block calls, for ``N`` transitions.

The primary optimizer accepts only predicted marginal terminal utilities.  A
separately named oracle accepts observed losses and is diagnostic only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


SEQUENTIAL_DEPTHS = (1, 2, 4)
CONTINUATION_COSTS = (0, 1, 3)
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = 260_818
DEFAULT_RANDOM_SEED = 260_816
DEFAULT_PERMUTATION_SEED = 260_817


@dataclass(frozen=True)
class WhiteningTransform:
    """A calibration-fit symmetric whitening transform."""

    mean: np.ndarray
    matrix: np.ndarray
    eigenvalues: np.ndarray
    used_eigenvalues: np.ndarray
    floor: float
    n_calibration: int


@dataclass(frozen=True)
class PairedBootstrapCI:
    """Episode-clustered percentile CI for a mean paired benefit."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_bootstrap: int
    n_samples: int
    n_episodes: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_benefit": self.estimate,
            "ci_low": self.lower,
            "ci_high": self.upper,
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
            "n_samples": self.n_samples,
            "n_episodes": self.n_episodes,
        }


def _validated_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result != value or result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _validated_depths(depths: Sequence[int] = SEQUENTIAL_DEPTHS) -> np.ndarray:
    values = np.asarray(depths)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("available depths must be a nonempty one-dimensional sequence")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("available depths must be integers")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 1) or np.any(np.diff(values) <= 0):
        raise ValueError("sequential depths must be unique, increasing, and at least one")
    if not np.array_equal(values, np.asarray(SEQUENTIAL_DEPTHS, dtype=np.int64)):
        raise ValueError("V2 sequential depths are frozen to (1, 2, 4)")
    return values


def _validated_allocations(
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = SEQUENTIAL_DEPTHS,
) -> np.ndarray:
    depths = _validated_depths(available_depths)
    selected = np.asarray(allocations)
    if selected.ndim != 1:
        raise ValueError("allocations must be one-dimensional")
    if not np.issubdtype(selected.dtype, np.integer):
        if not np.all(np.isfinite(selected)) or not np.all(selected == np.floor(selected)):
            raise ValueError("allocations must contain integer depths")
    selected = selected.astype(np.int64, copy=False)
    valid = np.isin(selected, depths)
    if not np.all(valid):
        raise ValueError(f"allocation contains unavailable depths: {np.unique(selected[~valid]).tolist()}")
    return selected


def _validated_depth_matrix(
    values_by_depth: np.ndarray,
    *,
    name: str,
    available_depths: Sequence[int] = SEQUENTIAL_DEPTHS,
) -> tuple[np.ndarray, np.ndarray]:
    depths = _validated_depths(available_depths)
    values = np.asarray(values_by_depth)
    if values.ndim < 2:
        raise ValueError(f"{name} must have sample and depth axes")
    if values.shape[1] != len(depths):
        raise ValueError(f"{name} depth axis must have {len(depths)} columns for depths {tuple(depths)}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values")
    return values, depths


def gather_depths(
    values_by_depth: np.ndarray,
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = SEQUENTIAL_DEPTHS,
) -> np.ndarray:
    """Gather one terminal value per sample by actual depth label."""

    values, depths = _validated_depth_matrix(
        values_by_depth, name="values_by_depth", available_depths=available_depths
    )
    selected = _validated_allocations(allocations, depths)
    if len(selected) != values.shape[0]:
        raise ValueError("allocation length must equal the number of samples")
    columns = np.searchsorted(depths, selected)
    return values[np.arange(values.shape[0]), columns]


gather_by_depth = gather_depths


def total_block_calls(
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = SEQUENTIAL_DEPTHS,
) -> int:
    """Count all calls, including the mandatory first call for every row."""

    selected = _validated_allocations(allocations, available_depths)
    return int(selected.sum(dtype=np.int64))


count_refiner_calls = total_block_calls


def continuation_block_calls(
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = SEQUENTIAL_DEPTHS,
) -> int:
    """Count calls made after the mandatory first call."""

    selected = _validated_allocations(allocations, available_depths)
    return int((selected - 1).sum(dtype=np.int64))


def validate_exact_budget(
    allocations: Iterable[int] | np.ndarray,
    total_budget: int,
    available_depths: Sequence[int] = SEQUENTIAL_DEPTHS,
) -> np.ndarray:
    """Validate an allocation against an explicit total-call budget."""

    budget = _validated_nonnegative_integer(total_budget, "total_budget")
    selected = _validated_allocations(allocations, available_depths)
    actual = total_block_calls(selected, available_depths)
    if actual != budget:
        raise ValueError(f"budget mismatch: expected {budget} total calls, got {actual}")
    return selected


def validate_sequential_exact_budget(
    allocations: Iterable[int] | np.ndarray,
    n_samples: int | None = None,
) -> np.ndarray:
    """Enforce the V2 contract: one first call plus exactly ``N`` more calls."""

    selected = _validated_allocations(allocations)
    n = len(selected) if n_samples is None else _validated_nonnegative_integer(n_samples, "n_samples")
    if len(selected) != n:
        raise ValueError(f"sample-count mismatch: expected {n}, got {len(selected)}")
    if continuation_block_calls(selected) != n or total_block_calls(selected) != 2 * n:
        raise ValueError(
            "sequential budget mismatch: expected N continuation calls and 2N total calls"
        )
    return selected


def _terminal_utility_dp(utilities: np.ndarray, continuation_budget: int) -> np.ndarray:
    """Maximize terminal utility under an exact continuation-call budget."""

    values = np.asarray(utilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("terminal utilities must have shape (N, 3) for depths (1, 2, 4)")
    if not np.all(np.isfinite(values)):
        raise ValueError("terminal utilities contain nonfinite values")
    budget = _validated_nonnegative_integer(continuation_budget, "continuation_budget")
    n = len(values)
    if n == 0:
        if budget == 0:
            return np.empty(0, dtype=np.int64)
        raise ValueError("nonzero continuation budget is infeasible for zero samples")
    costs = np.asarray(CONTINUATION_COSTS, dtype=np.int64)
    if budget > n * int(costs[-1]):
        raise ValueError("exact continuation budget is infeasible")

    # Solve suffixes so strict improvement plus increasing-depth iteration
    # gives lexicographically smaller final depths at the earliest sample when
    # total utility ties. This is the PLAN's fixed smaller-depth tie rule.
    choices = np.full((n, budget + 1), -1, dtype=np.int8)
    suffix = np.full(budget + 1, -np.inf, dtype=np.float64)
    suffix[0] = 0.0
    for row in range(n - 1, -1, -1):
        current = np.full(budget + 1, -np.inf, dtype=np.float64)
        row_choices = choices[row]
        for column, cost in enumerate(costs):
            cost_int = int(cost)
            if cost_int > budget:
                continue
            candidate = suffix[: budget + 1 - cost_int] + values[row, column]
            destination = current[cost_int:]
            improve = candidate > destination
            destination[improve] = candidate[improve]
            row_choices[cost_int:][improve] = column
        suffix = current
    if not np.isfinite(suffix[budget]):
        raise ValueError("exact continuation budget is infeasible")

    depths = np.asarray(SEQUENTIAL_DEPTHS, dtype=np.int64)
    allocation = np.empty(n, dtype=np.int64)
    remaining = budget
    for row in range(n):
        column = int(choices[row, remaining])
        if column < 0:
            raise RuntimeError("internal exact-budget backtracking failure")
        allocation[row] = depths[column]
        remaining -= int(costs[column])
    if remaining != 0:
        raise RuntimeError("internal continuation-call accounting failure")
    return allocation


def sequential_exact_budget_allocation(
    predicted_marginal_utilities: np.ndarray,
    n_samples: int | None = None,
) -> np.ndarray:
    """Allocate depths from target-free predictions at exactly ``2N`` calls.

    The two columns are predicted ``(L1-L2, L1-L4)``.  A fixed zero utility
    for stopping at depth 1 is inserted internally.  This primary interface
    intentionally has no target, loss, label, or tunable-budget argument.
    """

    predicted = np.asarray(predicted_marginal_utilities, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted marginal utilities must have shape (N, 2) for b12 and b14")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("predicted marginal utilities contain nonfinite values")
    n = len(predicted) if n_samples is None else _validated_nonnegative_integer(n_samples, "n_samples")
    if len(predicted) != n:
        raise ValueError(f"sample-count mismatch: expected {n}, got {len(predicted)}")
    utilities = np.column_stack([np.zeros(n, dtype=np.float64), predicted])
    allocation = _terminal_utility_dp(utilities, continuation_budget=n)
    return validate_sequential_exact_budget(allocation, n)


adaptive_sequential_allocation = sequential_exact_budget_allocation
predicted_utility_exact_budget_allocation = sequential_exact_budget_allocation
sequential_gate_allocation = sequential_exact_budget_allocation


def marginal_continuation_benefits(losses_by_depth: np.ndarray) -> np.ndarray:
    """Construct target-derived train labels ``(L1-L2, L1-L4)``."""

    losses, _ = _validated_depth_matrix(losses_by_depth, name="losses_by_depth")
    losses = np.asarray(losses, dtype=np.float64)
    return np.column_stack([losses[:, 0] - losses[:, 1], losses[:, 0] - losses[:, 2]])


def sequential_oracle_exact_budget_allocation(losses_by_depth: np.ndarray) -> np.ndarray:
    """Target-informed exact-budget oracle; diagnostic only."""

    losses, _ = _validated_depth_matrix(losses_by_depth, name="losses_by_depth")
    benefits = marginal_continuation_benefits(np.asarray(losses, dtype=np.float64))
    utilities = np.column_stack([np.zeros(len(losses), dtype=np.float64), benefits])
    allocation = _terminal_utility_dp(utilities, continuation_budget=len(losses))
    return validate_sequential_exact_budget(allocation, len(losses))


sequential_exact_budget_oracle = sequential_oracle_exact_budget_allocation
oracle_exact_budget_allocation = sequential_oracle_exact_budget_allocation


def random_sequential_exact_budget_allocation(
    n_samples: int,
    rng: np.random.Generator | int | None = DEFAULT_RANDOM_SEED,
) -> np.ndarray:
    """Generate a target- and prediction-free random exact-budget baseline."""

    n = _validated_nonnegative_integer(n_samples, "n_samples")
    if n == 0:
        return np.empty(0, dtype=np.int64)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    costs = np.asarray(CONTINUATION_COSTS, dtype=np.int64)
    budget = n

    # reachable[r,b] indicates whether b calls can be assigned to r rows.
    reachable = np.zeros((n + 1, budget + 1), dtype=bool)
    reachable[0, 0] = True
    for rows in range(1, n + 1):
        for cost in costs:
            cost_int = int(cost)
            if cost_int <= budget:
                reachable[rows, cost_int:] |= reachable[rows - 1, : budget + 1 - cost_int]
    if not reachable[n, budget]:
        raise RuntimeError("frozen V2 continuation budget unexpectedly infeasible")

    allocation = np.empty(n, dtype=np.int64)
    depths = np.asarray(SEQUENTIAL_DEPTHS, dtype=np.int64)
    row_order = generator.permutation(n)
    remaining = budget
    for position in range(n):
        rows_after = n - position - 1
        feasible_columns = np.asarray(
            [
                column
                for column, cost in enumerate(costs)
                if int(cost) <= remaining and reachable[rows_after, remaining - int(cost)]
            ],
            dtype=np.int64,
        )
        column = int(feasible_columns[int(generator.integers(len(feasible_columns)))])
        allocation[row_order[position]] = depths[column]
        remaining -= int(costs[column])
    return validate_sequential_exact_budget(allocation, n)


random_exact_budget_allocation = random_sequential_exact_budget_allocation


def permuted_allocation(
    allocations: Iterable[int] | np.ndarray,
    rng: np.random.Generator | int | None = DEFAULT_PERMUTATION_SEED,
) -> np.ndarray:
    """Permute row assignments while preserving depth histogram and calls."""

    selected = _validated_allocations(allocations)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    result = generator.permutation(selected)
    if depth_histogram(result) != depth_histogram(selected):
        raise RuntimeError("internal permutation histogram failure")
    return result


permute_allocation = permuted_allocation


def uniform_allocation(n_samples: int, depth: int) -> np.ndarray:
    """Return a fixed permitted final depth."""

    n = _validated_nonnegative_integer(n_samples, "n_samples")
    return _validated_allocations(np.full(n, depth))


def depth_histogram(allocations: Iterable[int] | np.ndarray) -> dict[int, int]:
    """Stable histogram including unused frozen sequential depths."""

    selected = _validated_allocations(allocations)
    return {depth: int(np.count_nonzero(selected == depth)) for depth in SEQUENTIAL_DEPTHS}


def fit_calibration_whitening(
    calibration_targets: np.ndarray,
    *,
    floor_fraction: float = 1e-6,
) -> WhiteningTransform:
    """Fit the frozen symmetric whitening transform on calibration targets."""

    targets = np.asarray(calibration_targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[0] < 2 or targets.shape[1] < 1:
        raise ValueError("calibration_targets must have shape (at least 2, latent_dim)")
    if not np.all(np.isfinite(targets)):
        raise ValueError("calibration_targets contains nonfinite values")
    if not np.isfinite(floor_fraction) or floor_fraction <= 0:
        raise ValueError("floor_fraction must be positive and finite")
    mean = targets.mean(axis=0)
    covariance = np.atleast_2d(np.cov(targets, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[-1])
    floor = max(float(floor_fraction) * largest, np.finfo(np.float64).eps)
    used = np.maximum(eigenvalues, floor)
    matrix = np.einsum(
        "ij,kj,j->ik", eigenvectors, eigenvectors, 1.0 / np.sqrt(used), optimize=False
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("calibration whitening produced a nonfinite transform")
    return WhiteningTransform(mean, matrix, eigenvalues, used, floor, int(len(targets)))


fit_whitening = fit_calibration_whitening


def latent_mse_by_depth(
    targets: np.ndarray,
    exits_by_depth: np.ndarray,
    *,
    whitening: WhiteningTransform | None = None,
) -> np.ndarray:
    """Return raw or calibration-whitened per-sample terminal MSE."""

    target = np.asarray(targets, dtype=np.float64)
    exits, _ = _validated_depth_matrix(exits_by_depth, name="exits_by_depth")
    exits = np.asarray(exits, dtype=np.float64)
    if target.ndim != 2 or exits.ndim != 3:
        raise ValueError("targets and exits must have shapes (N,D) and (N,3,D)")
    if exits.shape[0] != target.shape[0] or exits.shape[2] != target.shape[1]:
        raise ValueError("target and exit sample/latent dimensions must match")
    if not np.all(np.isfinite(target)):
        raise ValueError("targets contains nonfinite values")
    delta = exits - target[:, None, :]
    if whitening is not None:
        if whitening.matrix.shape != (target.shape[1], target.shape[1]):
            raise ValueError("whitening transform does not match latent dimension")
        delta = np.einsum("nkd,df->nkf", delta, whitening.matrix, optimize=False)
    losses = np.mean(delta**2, axis=2)
    if not np.all(np.isfinite(losses)):
        raise ValueError("latent MSE contains nonfinite values")
    return losses


depth_losses = latent_mse_by_depth


def whitened_mse(
    predictions: np.ndarray,
    targets: np.ndarray,
    transform: WhiteningTransform,
) -> np.ndarray:
    """Score prediction errors with a frozen calibration transform."""

    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("predictions and targets must have matching (N,D) shapes")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("predictions and targets must be finite")
    if transform.matrix.shape != (prediction.shape[1], prediction.shape[1]):
        raise ValueError("whitening transform does not match latent dimension")
    delta = np.einsum("nd,df->nf", prediction - target, transform.matrix, optimize=False)
    losses = np.mean(delta**2, axis=1)
    if not np.all(np.isfinite(losses)):
        raise ValueError("whitened MSE contains nonfinite values")
    return losses


def episode_clustered_paired_bootstrap_ci(
    left_values: Iterable[float] | np.ndarray,
    right_values: Iterable[float] | np.ndarray,
    episode_ids: Iterable[object] | np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedBootstrapCI:
    """Percentile CI for paired ``left-right``, resampling episodes."""

    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    episodes = np.asarray(episode_ids)
    if left.ndim != 1 or right.ndim != 1 or episodes.ndim != 1:
        raise ValueError("paired values and episode_ids must be one-dimensional")
    if not (len(left) == len(right) == len(episodes)) or len(left) == 0:
        raise ValueError("paired values and episode_ids must have equal nonzero length")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired values must be finite")
    replicates = _validated_nonnegative_integer(n_bootstrap, "n_bootstrap")
    if replicates < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    cluster_map: dict[object, list[int]] = {}
    for index, episode in enumerate(episodes.tolist()):
        try:
            cluster_map.setdefault(episode, []).append(index)
        except TypeError as exc:
            raise ValueError("episode IDs must be hashable") from exc
    clusters = [np.asarray(rows, dtype=np.int64) for rows in cluster_map.values()]
    differences = left - right
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = generator.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[int(index)] for index in sampled])
        bootstrap[replicate] = float(differences[rows].mean())
    alpha = 1.0 - float(confidence)
    lower, upper = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedBootstrapCI(
        float(differences.mean()),
        float(lower),
        float(upper),
        float(confidence),
        replicates,
        int(len(differences)),
        int(len(clusters)),
    )


episode_clustered_paired_bootstrap = episode_clustered_paired_bootstrap_ci


def paired_cluster_bootstrap(
    delta: Iterable[float] | np.ndarray,
    episode_ids: Iterable[object] | np.ndarray,
    **kwargs: object,
) -> PairedBootstrapCI:
    """Clustered CI when paired differences are already computed."""

    differences = np.asarray(delta, dtype=np.float64)
    return episode_clustered_paired_bootstrap_ci(
        differences, np.zeros_like(differences), episode_ids, **kwargs
    )


def summarize_allocation(losses_by_depth: np.ndarray, allocations: Iterable[int]) -> dict[str, object]:
    """Summarize selected loss and exact sequential call accounting."""

    losses, _ = _validated_depth_matrix(losses_by_depth, name="losses_by_depth")
    selected = _validated_allocations(allocations)
    if len(selected) != len(losses):
        raise ValueError("allocation length must equal the number of samples")
    chosen = gather_depths(losses, selected)
    return {
        "n_samples": int(len(selected)),
        "first_calls": int(len(selected)),
        "continuation_calls": continuation_block_calls(selected),
        "total_calls": total_block_calls(selected),
        "mean_calls": float(selected.mean()) if len(selected) else 0.0,
        "mean_loss": float(chosen.mean()) if len(chosen) else float("nan"),
        "mean_benefit_vs_depth1": float((losses[:, 0] - chosen).mean())
        if len(chosen)
        else float("nan"),
        "depth_histogram": depth_histogram(selected),
    }


def validate_episode_split_isolation(
    train_episode_ids: Iterable[object],
    calibration_episode_ids: Iterable[object],
    test_episode_ids: Iterable[object],
) -> None:
    """Reject any episode overlap across train/calibration/test."""

    try:
        train, calibration, test = map(set, (train_episode_ids, calibration_episode_ids, test_episode_ids))
    except TypeError as exc:
        raise ValueError("episode IDs must be hashable") from exc
    overlaps = {
        "train/calibration": train & calibration,
        "train/test": train & test,
        "calibration/test": calibration & test,
    }
    present = {name: sorted(values, key=repr) for name, values in overlaps.items() if values}
    if present:
        raise ValueError(f"episode split overlap: {present}")


def validate_split_isolation(split_episode_mapping: Mapping[str, Iterable[object]]) -> None:
    """Validate a mapping with exactly the three preregistered splits."""

    required = {"train", "calibration", "test"}
    missing, extra = required - set(split_episode_mapping), set(split_episode_mapping) - required
    if missing or extra:
        raise ValueError(
            f"split mapping must contain exactly {sorted(required)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    validate_episode_split_isolation(
        split_episode_mapping["train"],
        split_episode_mapping["calibration"],
        split_episode_mapping["test"],
    )


assert_split_isolation = validate_episode_split_isolation


__all__ = [
    "CONTINUATION_COSTS",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_PERMUTATION_SEED",
    "DEFAULT_RANDOM_SEED",
    "PairedBootstrapCI",
    "SEQUENTIAL_DEPTHS",
    "WhiteningTransform",
    "adaptive_sequential_allocation",
    "assert_split_isolation",
    "continuation_block_calls",
    "count_refiner_calls",
    "depth_histogram",
    "depth_losses",
    "episode_clustered_paired_bootstrap",
    "episode_clustered_paired_bootstrap_ci",
    "fit_calibration_whitening",
    "fit_whitening",
    "gather_by_depth",
    "gather_depths",
    "latent_mse_by_depth",
    "marginal_continuation_benefits",
    "oracle_exact_budget_allocation",
    "paired_cluster_bootstrap",
    "permute_allocation",
    "permuted_allocation",
    "predicted_utility_exact_budget_allocation",
    "random_exact_budget_allocation",
    "random_sequential_exact_budget_allocation",
    "sequential_exact_budget_allocation",
    "sequential_exact_budget_oracle",
    "sequential_gate_allocation",
    "sequential_oracle_exact_budget_allocation",
    "summarize_allocation",
    "total_block_calls",
    "uniform_allocation",
    "validate_episode_split_isolation",
    "validate_exact_budget",
    "validate_sequential_exact_budget",
    "validate_split_isolation",
    "whitened_mse",
]
