"""Exact-budget allocation and statistics for the adaptive-compute pilot.

Depth is the number of shared residual-block calls, not the column index.  All
helpers therefore carry the available depths explicitly and default to the
preregistered exits ``(0, 1, 2, 4)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


DEFAULT_DEPTHS = (0, 1, 2, 4)
DEFAULT_BOOTSTRAP_REPLICATES = 1_000
DEFAULT_BOOTSTRAP_SEED = 260_713


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
    """Episode-clustered CI for the mean paired benefit ``left - right``."""

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


def _validated_available_depths(depths: Sequence[int] = DEFAULT_DEPTHS) -> np.ndarray:
    arr = np.asarray(depths)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("available depths must be a nonempty one-dimensional sequence")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
            raise ValueError("available depths must be integers")
    arr = arr.astype(np.int64, copy=False)
    if np.any(arr < 0):
        raise ValueError("available depths must be nonnegative")
    if len(np.unique(arr)) != len(arr):
        raise ValueError("available depths must be unique")
    if np.any(np.diff(arr) <= 0):
        raise ValueError("available depths must be in increasing order")
    if arr[0] != 0:
        raise ValueError("available depths must include depth 0 first")
    return arr


def _validated_allocations(
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    depths = _validated_available_depths(available_depths)
    selected = np.asarray(allocations)
    if selected.ndim != 1:
        raise ValueError("allocations must be one-dimensional")
    if not np.issubdtype(selected.dtype, np.integer):
        if not np.all(np.isfinite(selected)) or not np.all(selected == np.floor(selected)):
            raise ValueError("allocations must contain integer depths")
    selected = selected.astype(np.int64, copy=False)
    if not np.all(np.isin(selected, depths)):
        invalid = np.unique(selected[~np.isin(selected, depths)]).tolist()
        raise ValueError(f"allocation contains unavailable depths: {invalid}")
    return selected


def _validated_depth_matrix(
    values_by_depth: np.ndarray,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
    *,
    name: str = "values_by_depth",
    finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    depths = _validated_available_depths(available_depths)
    values = np.asarray(values_by_depth)
    if values.ndim < 2:
        raise ValueError(f"{name} must have sample and depth axes")
    if values.shape[1] != len(depths):
        raise ValueError(
            f"{name} depth axis has length {values.shape[1]}, expected {len(depths)}"
        )
    if finite and not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values")
    return values, depths


def gather_depths(
    values_by_depth: np.ndarray,
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """Gather one value/exit per sample using actual depth labels.

    ``values_by_depth`` has shape ``(sample, exit, ...)``.  In particular,
    depth 4 maps to exit column 3 for the default exits; it never maps to
    column 4.
    """

    values, depths = _validated_depth_matrix(values_by_depth, available_depths)
    selected = _validated_allocations(allocations, depths)
    if len(selected) != values.shape[0]:
        raise ValueError("allocation length must equal the number of samples")
    # Depths are strictly increasing, so searchsorted is an exact label lookup.
    columns = np.searchsorted(depths, selected)
    return values[np.arange(values.shape[0]), columns]


# Singular spelling is convenient for call sites and remains exact-label based.
gather_by_depth = gather_depths


def total_block_calls(
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> int:
    """Return exact refiner calls; the cost of selected depth ``k`` is ``k``."""

    selected = _validated_allocations(allocations, available_depths)
    return int(selected.sum(dtype=np.int64))


count_refiner_calls = total_block_calls


def validate_exact_budget(
    allocations: Iterable[int] | np.ndarray,
    total_budget: int,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """Validate and return an allocation whose actual depths sum to budget."""

    budget = _validated_budget(total_budget)
    selected = _validated_allocations(allocations, available_depths)
    actual = total_block_calls(selected, available_depths)
    if actual != budget:
        raise ValueError(f"budget mismatch: expected {budget} calls, got {actual}")
    return selected


def _validated_budget(total_budget: int) -> int:
    if isinstance(total_budget, (bool, np.bool_)):
        raise ValueError("total_budget must be a nonnegative integer")
    try:
        budget = int(total_budget)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("total_budget must be a nonnegative integer") from exc
    if budget != total_budget or budget < 0:
        raise ValueError("total_budget must be a nonnegative integer")
    return budget


def exact_budget_minimize(
    costs_by_depth: np.ndarray,
    total_budget: int,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """Solve the exact-budget multiple-choice problem by dynamic programming.

    One depth is selected for every row, minimizing the sum of the supplied
    costs subject to ``sum(selected_depths) == total_budget``.  The transition
    costs are actual depths, so the default exit columns cost 0, 1, 2, and 4.
    Ties are deterministic in increasing-depth order.
    """

    costs, depths = _validated_depth_matrix(
        np.asarray(costs_by_depth, dtype=np.float64),
        available_depths,
        name="costs_by_depth",
    )
    budget = _validated_budget(total_budget)
    n_samples = costs.shape[0]
    if budget > n_samples * int(depths[-1]):
        raise ValueError("exact budget is infeasible for the sample count and depths")
    if n_samples == 0:
        if budget == 0:
            return np.empty(0, dtype=np.int64)
        raise ValueError("nonzero budget is infeasible for zero samples")

    # choice[i, b] stores the exit-column selected for sample i on an optimal
    # path having spent b calls after that row. Previous spend is b-depth[j].
    choice_dtype = np.int16 if len(depths) <= np.iinfo(np.int16).max else np.int32
    choices = np.full((n_samples, budget + 1), -1, dtype=choice_dtype)
    previous = np.full(budget + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0

    for row in range(n_samples):
        current = np.full(budget + 1, np.inf, dtype=np.float64)
        row_choices = choices[row]
        for column, depth in enumerate(depths):
            depth_int = int(depth)
            if depth_int > budget:
                break
            candidate = previous[: budget + 1 - depth_int] + costs[row, column]
            destination = current[depth_int:]
            improve = candidate < destination
            destination[improve] = candidate[improve]
            row_choices[depth_int:][improve] = column
        previous = current

    if not np.isfinite(previous[budget]):
        raise ValueError("exact budget is infeasible for the sample count and depths")

    allocation = np.empty(n_samples, dtype=np.int64)
    remaining = budget
    for row in range(n_samples - 1, -1, -1):
        column = int(choices[row, remaining])
        if column < 0:
            raise RuntimeError("internal exact-budget backtracking failure")
        allocation[row] = depths[column]
        remaining -= int(depths[column])
    if remaining != 0:
        raise RuntimeError("internal exact-budget accounting failure")
    return validate_exact_budget(allocation, budget, depths)


def oracle_exact_budget_allocation(
    losses_by_depth: np.ndarray,
    total_budget: int,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """Target-informed diagnostic allocation minimizing observed loss."""

    return exact_budget_minimize(losses_by_depth, total_budget, available_depths)


# Short aliases used in experiment scripts.
exact_budget_oracle = oracle_exact_budget_allocation
oracle_allocation = oracle_exact_budget_allocation


def exact_budget_optimal_allocation(
    loss_matrix: np.ndarray,
    depths: Sequence[int],
    total_budget: int,
) -> np.ndarray:
    """Driver-facing exact-budget oracle with explicit positional depths."""

    return oracle_exact_budget_allocation(loss_matrix, total_budget, depths)


def predicted_gain_exact_budget_allocation(
    predicted_gains: np.ndarray,
    total_budget: int,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """Allocate from causal predicted gains without observing evaluation loss.

    ``predicted_gains`` may contain columns for every available depth, including
    a depth-0 column, or only the positive depths.  In the latter case a fixed
    zero-gain depth-0 column is inserted.  No target or observed-loss argument
    is accepted by this interface.
    """

    depths = _validated_available_depths(available_depths)
    gains = np.asarray(predicted_gains, dtype=np.float64)
    if gains.ndim != 2:
        raise ValueError("predicted_gains must be a two-dimensional array")
    if gains.shape[1] == len(depths) - 1:
        gains = np.concatenate([np.zeros((len(gains), 1), dtype=np.float64), gains], axis=1)
    elif gains.shape[1] != len(depths):
        raise ValueError("predicted_gains must have one column per depth or per positive depth")
    if not np.all(np.isfinite(gains)):
        raise ValueError("predicted_gains contains nonfinite values")
    # Maximizing predicted total gain is equivalent to minimizing its negative.
    return exact_budget_minimize(-gains, total_budget, depths)


predicted_gain_allocation = predicted_gain_exact_budget_allocation


def random_exact_budget_allocation(
    n_samples: int,
    depths_or_budget: Sequence[int] | int | None = None,
    total_budget: int | None = None,
    rng: np.random.Generator | int | None = None,
    *,
    available_depths: Sequence[int] | None = None,
) -> np.ndarray:
    """Draw a target- and prediction-free feasible exact-budget allocation.

    At each row, a depth is sampled uniformly from choices that admit at least
    one completion.  Feasibility is computed only from sample count, budget,
    available depths, and RNG state.
    """

    if isinstance(n_samples, (bool, np.bool_)):
        raise ValueError("n_samples must be a nonnegative integer")
    try:
        n = int(n_samples)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_samples must be a nonnegative integer") from exc
    if n != n_samples or n < 0:
        raise ValueError("n_samples must be a nonnegative integer")
    if depths_or_budget is None:
        if total_budget is None:
            raise ValueError("total_budget is required")
        budget = _validated_budget(total_budget)
        depths = _validated_available_depths(
            DEFAULT_DEPTHS if available_depths is None else available_depths
        )
    elif np.isscalar(depths_or_budget):
        if total_budget is not None:
            raise ValueError(
                "pass either (n, budget) or (n, depths, budget), not two budget values"
            )
        budget = _validated_budget(depths_or_budget)  # type: ignore[arg-type]
        depths = _validated_available_depths(
            DEFAULT_DEPTHS if available_depths is None else available_depths
        )
    else:
        if total_budget is None:
            raise ValueError("total_budget is required when explicit depths are supplied")
        if available_depths is not None:
            raise ValueError("depths were supplied twice")
        budget = _validated_budget(total_budget)
        depths = _validated_available_depths(depths_or_budget)
    if budget > n * int(depths[-1]):
        raise ValueError("exact budget is infeasible for the sample count and depths")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # reachable[r, b] says b calls can be assigned to exactly r remaining rows.
    reachable = np.zeros((n + 1, budget + 1), dtype=bool)
    reachable[0, 0] = True
    for rows in range(1, n + 1):
        for depth in depths:
            depth_int = int(depth)
            if depth_int <= budget:
                reachable[rows, depth_int:] |= reachable[rows - 1, : budget + 1 - depth_int]
    if not reachable[n, budget]:
        raise ValueError("exact budget is infeasible for the sample count and depths")

    allocation = np.empty(n, dtype=np.int64)
    # Randomizing which sample occupies each sequential feasibility position
    # makes the baseline exchangeable over rows (for example, it cannot favor
    # early transitions merely because the final rows are more constrained).
    row_order = generator.permutation(n)
    remaining = budget
    for row in range(n):
        rows_after = n - row - 1
        feasible = np.asarray(
            [
                depth
                for depth in depths
                if int(depth) <= remaining
                and reachable[rows_after, remaining - int(depth)]
            ],
            dtype=np.int64,
        )
        selected = int(feasible[int(generator.integers(len(feasible)))])
        allocation[row_order[row]] = selected
        remaining -= selected
    return validate_exact_budget(allocation, budget, depths)


random_allocation = random_exact_budget_allocation


def permuted_allocation(
    allocations: Iterable[int] | np.ndarray,
    rng: np.random.Generator | int | None = None,
    *,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """Permute sample assignments while preserving histogram and exact calls."""

    selected = _validated_allocations(allocations, available_depths)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return generator.permutation(selected)


permute_allocation = permuted_allocation


def uniform_allocation(
    n_samples: int,
    depth: int,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """Return a fixed-depth allocation, validating depth by label."""

    if isinstance(n_samples, (bool, np.bool_)) or int(n_samples) != n_samples or int(n_samples) < 0:
        raise ValueError("n_samples must be a nonnegative integer")
    selected = np.full(int(n_samples), depth)
    return _validated_allocations(selected, available_depths)


def fit_calibration_whitening(
    calibration_targets: np.ndarray,
    *,
    floor_fraction: float = 1e-6,
) -> WhiteningTransform:
    """Fit whitening on calibration target latents only.

    The covariance eigenspectrum is floored at ``floor_fraction * lambda_max``
    as preregistered.  A machine-epsilon fallback handles a fully constant
    calibration set while keeping the transform finite.
    """

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
    inv_sqrt = 1.0 / np.sqrt(used)
    # NumPy 2.x linked against Apple Accelerate can leave spurious floating
    # status flags on otherwise finite BLAS matmul results. This explicit
    # contraction is algebraically identical and warning-free on that runtime.
    matrix = np.einsum(
        "ij,kj,j->ik", eigenvectors, eigenvectors, inv_sqrt, optimize=False
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("calibration whitening produced a nonfinite transform")
    return WhiteningTransform(
        mean=mean,
        matrix=matrix,
        eigenvalues=eigenvalues,
        used_eigenvalues=used,
        floor=floor,
        n_calibration=int(targets.shape[0]),
    )


fit_whitening = fit_calibration_whitening


def latent_mse_by_depth(
    targets: np.ndarray,
    exits_by_depth: np.ndarray,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
    *,
    whitening: WhiteningTransform | None = None,
) -> np.ndarray:
    """Return raw or calibration-whitened per-sample MSE at every exit."""

    target = np.asarray(targets, dtype=np.float64)
    exits, _ = _validated_depth_matrix(
        np.asarray(exits_by_depth, dtype=np.float64),
        available_depths,
        name="exits_by_depth",
    )
    if target.ndim != 2 or exits.ndim != 3:
        raise ValueError("targets and exits must have shapes (N,D) and (N,K,D)")
    if exits.shape[0] != target.shape[0] or exits.shape[2] != target.shape[1]:
        raise ValueError("target and exit sample/latent dimensions must match")
    if not np.all(np.isfinite(target)):
        raise ValueError("targets contains nonfinite values")
    delta = exits - target[:, None, :]
    if whitening is not None:
        if whitening.matrix.ndim != 2 or whitening.matrix.shape[0] != target.shape[1]:
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
    """Apply a frozen calibration transform to prediction errors and score MSE."""

    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("predictions and targets must have matching (N,D) shapes")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("predictions and targets must be finite")
    if transform.matrix.shape[0] != prediction.shape[1]:
        raise ValueError("whitening transform does not match latent dimension")
    whitened_delta = np.einsum(
        "nd,df->nf", prediction - target, transform.matrix, optimize=False
    )
    losses = np.mean(whitened_delta**2, axis=1)
    if not np.all(np.isfinite(losses)):
        raise ValueError("whitened MSE contains nonfinite values")
    return losses


def robust_relative_gain(
    losses_by_depth: np.ndarray,
    calibration_depth0_losses: Iterable[float] | np.ndarray,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> np.ndarray:
    """Compute preregistered robust relative gain for every sample and depth."""

    losses, depths = _validated_depth_matrix(
        np.asarray(losses_by_depth, dtype=np.float64),
        available_depths,
        name="losses_by_depth",
    )
    calibration = np.asarray(calibration_depth0_losses, dtype=np.float64)
    if calibration.ndim != 1 or calibration.size == 0 or not np.all(np.isfinite(calibration)):
        raise ValueError("calibration_depth0_losses must be a nonempty finite vector")
    depth0_column = int(np.flatnonzero(depths == 0)[0])
    baseline = losses[:, depth0_column]
    q10 = float(np.quantile(calibration, 0.10))
    denominator = np.maximum.reduce(
        [baseline, np.full_like(baseline, q10), np.full_like(baseline, 1e-8)]
    )
    return (baseline[:, None] - losses) / denominator[:, None]


def episode_clustered_paired_bootstrap_ci(
    left_values: Iterable[float] | np.ndarray,
    right_values: Iterable[float] | np.ndarray,
    episode_ids: Iterable[object] | np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedBootstrapCI:
    """Percentile CI for mean paired ``left - right``, resampling episodes."""

    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    episodes = np.asarray(episode_ids)
    if left.ndim != 1 or right.ndim != 1 or episodes.ndim != 1:
        raise ValueError("paired values and episode_ids must be one-dimensional")
    if not (len(left) == len(right) == len(episodes)) or len(left) == 0:
        raise ValueError("paired values and episode_ids must have equal nonzero length")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired values must be finite")
    if isinstance(n_bootstrap, (bool, np.bool_)) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    # np.unique cannot reliably sort arbitrary mixed object identifiers. This
    # insertion-ordered grouping supports any hashable episode ID.
    cluster_map: dict[object, list[int]] = {}
    for index, episode in enumerate(episodes.tolist()):
        try:
            cluster_map.setdefault(episode, []).append(index)
        except TypeError as exc:
            raise ValueError("episode IDs must be hashable") from exc
    clusters = [np.asarray(indices, dtype=np.int64) for indices in cluster_map.values()]
    differences = left - right
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(int(n_bootstrap), dtype=np.float64)
    for replicate in range(int(n_bootstrap)):
        selected = generator.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[int(index)] for index in selected])
        bootstrap[replicate] = float(differences[rows].mean())
    alpha = 1.0 - float(confidence)
    lower, upper = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedBootstrapCI(
        estimate=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        n_bootstrap=int(n_bootstrap),
        n_samples=int(len(differences)),
        n_episodes=int(len(clusters)),
    )


episode_clustered_paired_bootstrap = episode_clustered_paired_bootstrap_ci


def paired_cluster_bootstrap(
    delta: Iterable[float] | np.ndarray,
    episode_ids: Iterable[object] | np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> PairedBootstrapCI:
    """Driver-facing clustered CI when paired differences are precomputed."""

    differences = np.asarray(delta, dtype=np.float64)
    return episode_clustered_paired_bootstrap_ci(
        differences,
        np.zeros_like(differences),
        episode_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )


def depth_histogram(
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> dict[int, int]:
    """Return a stable histogram including unused permitted depths."""

    depths = _validated_available_depths(available_depths)
    selected = _validated_allocations(allocations, depths)
    return {int(depth): int(np.count_nonzero(selected == depth)) for depth in depths}


def summarize_allocation(
    losses_by_depth: np.ndarray,
    allocations: Iterable[int] | np.ndarray,
    available_depths: Sequence[int] = DEFAULT_DEPTHS,
) -> dict[str, object]:
    """Return concise loss, benefit, call, and depth-distribution metrics."""

    losses, depths = _validated_depth_matrix(
        np.asarray(losses_by_depth, dtype=np.float64),
        available_depths,
        name="losses_by_depth",
    )
    selected = _validated_allocations(allocations, depths)
    selected_loss = gather_depths(losses, selected, depths)
    baseline = losses[:, int(np.flatnonzero(depths == 0)[0])]
    return {
        "n_samples": int(len(selected)),
        "total_calls": total_block_calls(selected, depths),
        "mean_calls": float(selected.mean()) if len(selected) else 0.0,
        "mean_loss": float(selected_loss.mean()) if len(selected_loss) else float("nan"),
        "mean_gain_vs_depth0": float((baseline - selected_loss).mean())
        if len(selected_loss)
        else float("nan"),
        "depth_histogram": depth_histogram(selected, depths),
    }


def validate_episode_split_isolation(
    train_episode_ids: Iterable[object],
    calibration_episode_ids: Iterable[object],
    test_episode_ids: Iterable[object],
) -> None:
    """Assert pairwise-disjoint episode-level train/calibration/test splits."""

    try:
        train = set(train_episode_ids)
        calibration = set(calibration_episode_ids)
        test = set(test_episode_ids)
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


assert_split_isolation = validate_episode_split_isolation


def validate_split_isolation(
    split_episode_mapping: Mapping[str, Iterable[object]],
) -> None:
    """Validate a mapping containing train, calibration, and test episodes."""

    required = {"train", "calibration", "test"}
    missing = required - set(split_episode_mapping)
    extra = set(split_episode_mapping) - required
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


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_DEPTHS",
    "PairedBootstrapCI",
    "WhiteningTransform",
    "assert_split_isolation",
    "count_refiner_calls",
    "depth_histogram",
    "depth_losses",
    "episode_clustered_paired_bootstrap",
    "episode_clustered_paired_bootstrap_ci",
    "exact_budget_minimize",
    "exact_budget_oracle",
    "exact_budget_optimal_allocation",
    "fit_calibration_whitening",
    "fit_whitening",
    "gather_by_depth",
    "gather_depths",
    "latent_mse_by_depth",
    "oracle_allocation",
    "oracle_exact_budget_allocation",
    "paired_cluster_bootstrap",
    "permute_allocation",
    "permuted_allocation",
    "predicted_gain_allocation",
    "predicted_gain_exact_budget_allocation",
    "random_allocation",
    "random_exact_budget_allocation",
    "robust_relative_gain",
    "summarize_allocation",
    "total_block_calls",
    "uniform_allocation",
    "validate_episode_split_isolation",
    "validate_exact_budget",
    "validate_split_isolation",
    "whitened_mse",
]
