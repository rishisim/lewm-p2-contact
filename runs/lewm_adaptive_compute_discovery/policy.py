"""Causal adaptive-compute policies and evaluation statistics.

This module deliberately has no data-loading or model dependencies.  A caller
must provide already-frozen, out-of-fold critic scores and realized losses.  In
particular, none of the policy functions can inspect a target or a future exit
while making a stopping decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.stats import rankdata


ArrayLike = Sequence[float] | np.ndarray


def _finite_1d(values: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite 1-D array")
    return result


def _finite_2d(values: ArrayLike, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite 2-D array")
    return result


def _stage_vector(value: float | ArrayLike, stages: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(stages, float(result), dtype=np.float64)
    if result.shape != (stages,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite scalar or have shape [{stages}]")
    return result


def _exit_calls(exit_calls: Sequence[int] | None, stages: int) -> np.ndarray:
    if exit_calls is None:
        result = np.arange(1, stages + 2, dtype=np.int64)
    else:
        result = np.asarray(exit_calls)
        if result.ndim != 1 or len(result) != stages + 1:
            raise ValueError(f"exit_calls must have shape [{stages + 1}]")
        if not np.issubdtype(result.dtype, np.integer):
            if not np.equal(result, np.round(result)).all():
                raise ValueError("exit_calls must be integers")
        result = result.astype(np.int64)
    if np.any(result <= 0) or np.any(np.diff(result) <= 0):
        raise ValueError("exit_calls must be positive and strictly increasing")
    return result


def causal_sequential_stopping(
    scores: ArrayLike,
    *,
    thresholds: float | ArrayLike = 0.0,
    compute_price: float | ArrayLike = 0.0,
    exit_calls: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply a transition-local sequential stopping rule.

    ``scores[i, k]`` is the frozen critic's predicted marginal loss reduction
    available at exit ``k`` for transition ``i``.  The next stage is evaluated
    iff::

        score > threshold + compute_price * additional_block_calls

    A scalar price is a price per block call; a vector supplies a distinct
    per-block price at each decision.  Once a row stops, later score columns
    cannot affect its selected exit.  The returned values are block-call counts
    from ``exit_calls``, not zero-based column indices.
    """

    score_array = _finite_2d(scores, "scores")
    stages = score_array.shape[1]
    calls = _exit_calls(exit_calls, stages)
    threshold_array = _stage_vector(thresholds, stages, "thresholds")
    price_array = _stage_vector(compute_price, stages, "compute_price")
    if np.any(price_array < 0):
        raise ValueError("compute_price must be nonnegative")

    selected = np.full(score_array.shape[0], calls[0], dtype=np.int64)
    active = np.ones(score_array.shape[0], dtype=bool)
    marginal_costs = price_array * np.diff(calls)
    for stage in range(stages):
        advance = active & (
            score_array[:, stage] > threshold_array[stage] + marginal_costs[stage]
        )
        selected[advance] = calls[stage + 1]
        # Prefix halting is the causality constraint: stopped rows never rejoin.
        active &= advance
    return selected


# Short alias used by experiment scripts.
sequential_stopping = causal_sequential_stopping


def calibrate_compute_price(
    scores: ArrayLike,
    target_mean_calls: float,
    *,
    thresholds: float | ArrayLike = 0.0,
    exit_calls: Sequence[int] | None = None,
    calibration_losses: ArrayLike | None = None,
) -> dict[str, Any]:
    """Select a scalar compute price nearest a requested calibration budget.

    The search considers only score breakpoints and uses binary search because
    realized calls are monotone in price.  If two prices are equally close to
    the target, supplied calibration loss breaks the tie; the final tie-break
    is the more conservative (higher) price.  The returned selected calls make
    the realized calibration budget fully auditable.
    """

    score_array = _finite_2d(scores, "scores")
    stages = score_array.shape[1]
    calls = _exit_calls(exit_calls, stages)
    threshold_array = _stage_vector(thresholds, stages, "thresholds")
    target = float(target_mean_calls)
    if not np.isfinite(target) or target < calls[0] or target > calls[-1]:
        raise ValueError("target_mean_calls is outside the supported range")
    losses = None
    if calibration_losses is not None:
        losses = _finite_2d(calibration_losses, "calibration_losses")
        if losses.shape != (len(score_array), len(calls)):
            raise ValueError("calibration_losses must have shape [N, exits]")

    breakpoints = (
        (score_array - threshold_array[None, :]) / np.diff(calls)[None, :]
    ).ravel()
    breakpoints = np.unique(breakpoints[breakpoints >= 0.0])
    candidates = np.concatenate(
        (
            np.asarray([0.0]),
            breakpoints,
            np.asarray(
                [np.nextafter(breakpoints[-1], np.inf) if len(breakpoints) else 1.0]
            ),
        )
    )
    candidates = np.unique(candidates)

    cache: dict[int, tuple[np.ndarray, float, float | None]] = {}

    def evaluate(index: int) -> tuple[np.ndarray, float, float | None]:
        if index not in cache:
            selected = causal_sequential_stopping(
                score_array,
                thresholds=threshold_array,
                compute_price=float(candidates[index]),
                exit_calls=calls,
            )
            mean_calls = float(selected.mean())
            mean_loss = (
                float(gather_exit_values(losses, selected, calls).mean())
                if losses is not None
                else None
            )
            cache[index] = selected, mean_calls, mean_loss
        return cache[index]

    # Find the first price whose realized budget is at or below the target.
    low, high = 0, len(candidates) - 1
    while low < high:
        middle = (low + high) // 2
        if evaluate(middle)[1] <= target:
            high = middle
        else:
            low = middle + 1
    crossing = low
    # Adjacent breakpoints suffice for call mismatch; a small neighborhood also
    # gives loss tie-breaking stable behavior across large equal-score plateaus.
    probe = range(max(0, crossing - 3), min(len(candidates), crossing + 4))
    best: tuple[tuple[float, float, float], int] | None = None
    for index in probe:
        _, mean_calls, mean_loss = evaluate(index)
        key = (
            abs(mean_calls - target),
            float(mean_loss) if mean_loss is not None else 0.0,
            -float(candidates[index]),
        )
        if best is None or key < best[0]:
            best = key, index
    assert best is not None
    selected, realized, mean_loss = evaluate(best[1])
    result: dict[str, Any] = {
        "compute_price": float(candidates[best[1]]),
        "target_mean_calls": target,
        "realized_mean_calls": realized,
        "mean_call_error": float(realized - target),
        "selected_calls": selected,
        "audit": call_audit(selected, supported_calls=calls),
    }
    if mean_loss is not None:
        result["calibration_mean_loss"] = mean_loss
    return result


def fixed_exit(n: int, calls: int) -> np.ndarray:
    """Return a fixed-exit policy for ``n`` transitions."""

    if int(n) != n or n < 0 or int(calls) != calls or calls <= 0:
        raise ValueError("n must be nonnegative and calls must be positive integers")
    return np.full(int(n), int(calls), dtype=np.int64)


def gather_exit_values(
    values: ArrayLike, selected_calls: Sequence[int], exit_calls: Sequence[int]
) -> np.ndarray:
    """Gather one value per row using explicit call-count-to-column mapping."""

    matrix = _finite_2d(values, "values")
    selected = np.asarray(selected_calls)
    calls = np.asarray(exit_calls, dtype=np.int64)
    if calls.shape != (matrix.shape[1],) or len(selected) != matrix.shape[0]:
        raise ValueError("values, selected_calls, and exit_calls shapes disagree")
    if len(np.unique(calls)) != len(calls):
        raise ValueError("exit_calls must be unique")
    mapping = {int(call): column for column, call in enumerate(calls)}
    try:
        columns = np.asarray([mapping[int(call)] for call in selected], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"selected unsupported call count {exc.args[0]}") from None
    return matrix[np.arange(len(matrix)), columns]


def call_audit(
    selected_calls: Sequence[int],
    *,
    supported_calls: Sequence[int] | None = None,
    target_total_calls: int | None = None,
    expected_mean_calls: float | None = None,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Return exact realized and optional expected/target compute accounting."""

    selected = np.asarray(selected_calls)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("selected_calls must be a nonempty 1-D array")
    if not np.equal(selected, np.round(selected)).all() or np.any(selected <= 0):
        raise ValueError("selected_calls must contain positive integers")
    selected = selected.astype(np.int64)
    if supported_calls is not None:
        supported = np.asarray(supported_calls, dtype=np.int64)
        if supported.ndim != 1 or supported.size == 0:
            raise ValueError("supported_calls must be nonempty and 1-D")
        if not np.isin(selected, supported).all():
            raise ValueError("selected_calls contains an unsupported exit")

    unique, counts = np.unique(selected, return_counts=True)
    total = int(selected.sum(dtype=np.int64))
    mean = float(total / len(selected))
    result: dict[str, Any] = {
        "n": int(len(selected)),
        "total_calls": total,
        "realized_mean_calls": mean,
        "histogram": {str(int(k)): int(v) for k, v in zip(unique, counts, strict=True)},
    }
    if target_total_calls is not None:
        if int(target_total_calls) != target_total_calls:
            raise ValueError("target_total_calls must be an integer")
        target = int(target_total_calls)
        result.update(
            target_total_calls=target,
            total_call_error=int(total - target),
            exact_total_match=bool(total == target),
        )
    if expected_mean_calls is not None:
        expected = float(expected_mean_calls)
        if not np.isfinite(expected):
            raise ValueError("expected_mean_calls must be finite")
        result.update(
            expected_mean_calls=expected,
            realized_minus_expected=float(mean - expected),
            expected_match=bool(abs(mean - expected) <= atol),
        )
    return result


def optimal_expected_mixture(
    calibration_losses: ArrayLike,
    supported_calls: Sequence[int],
    target_mean_calls: float,
) -> dict[str, Any]:
    """Find the strongest transition-independent mixture in expectation.

    Strength is defined *only* by the supplied calibration losses.  The same
    probabilities apply to every transition, so this baseline has no local
    allocation signal.
    """

    calls = np.asarray(supported_calls, dtype=np.int64)
    if calls.ndim != 1 or calls.size == 0 or len(np.unique(calls)) != len(calls):
        raise ValueError("supported_calls must be a nonempty vector of unique calls")
    losses = np.asarray(calibration_losses, dtype=np.float64)
    if losses.ndim == 2:
        if losses.shape[1] != len(calls) or losses.shape[0] == 0:
            raise ValueError("calibration_losses columns must match supported_calls")
        means = losses.mean(axis=0)
    elif losses.shape == calls.shape:
        means = losses
    else:
        raise ValueError("calibration_losses must have shape [N,D] or [D]")
    if not np.isfinite(means).all():
        raise ValueError("calibration_losses must be finite")
    target = float(target_mean_calls)
    if not np.isfinite(target) or target < calls.min() or target > calls.max():
        raise ValueError("target_mean_calls is outside the supported range")

    result = linprog(
        means,
        A_eq=np.vstack((np.ones(len(calls)), calls)),
        b_eq=np.asarray([1.0, target]),
        bounds=[(0.0, 1.0)] * len(calls),
        method="highs",
    )
    if not result.success:
        raise ValueError(f"no feasible expected mixture: {result.message}")
    probabilities = np.clip(result.x, 0.0, 1.0)
    probabilities /= probabilities.sum()
    if not np.isclose(probabilities @ calls, target, atol=1e-8):
        raise RuntimeError("expected mixture compute accounting failed")
    return {
        "supported_calls": calls.copy(),
        "probabilities": probabilities,
        "expected_mean_calls": float(probabilities @ calls),
        "calibration_mean_loss": float(probabilities @ means),
        "calibration_exit_means": means.copy(),
    }


def strongest_transition_independent_baseline(
    calibration_losses: ArrayLike,
    supported_calls: Sequence[int],
    *,
    n: int,
    target_total_calls: int,
    seed: int,
) -> dict[str, Any]:
    """Construct the best exact-call randomized mixture on calibration loss.

    Integer depth counts minimize calibration mean loss subject to exactly
    ``n`` assignments and exactly ``target_total_calls``.  Their assignment to
    transitions is a seeded shuffle independent of all transition features,
    scores, targets, and episode labels.
    """

    if int(n) != n or n <= 0 or int(target_total_calls) != target_total_calls:
        raise ValueError("n must be positive and target_total_calls an integer")
    n = int(n)
    target_total_calls = int(target_total_calls)
    calls = np.asarray(supported_calls, dtype=np.int64)
    if (
        calls.ndim != 1
        or calls.size == 0
        or np.any(calls <= 0)
        or len(np.unique(calls)) != len(calls)
    ):
        raise ValueError("supported_calls must be unique positive integers")
    losses = np.asarray(calibration_losses, dtype=np.float64)
    if losses.ndim == 2:
        if losses.shape[0] == 0 or losses.shape[1] != len(calls):
            raise ValueError("calibration_losses columns must match supported_calls")
        means = losses.mean(axis=0)
    elif losses.shape == calls.shape:
        means = losses
    else:
        raise ValueError("calibration_losses must have shape [N,D] or [D]")
    if not np.isfinite(means).all():
        raise ValueError("calibration_losses must be finite")
    if target_total_calls < n * calls.min() or target_total_calls > n * calls.max():
        raise ValueError("target_total_calls is outside the feasible range")

    constraints = LinearConstraint(
        np.vstack((np.ones(len(calls)), calls)),
        lb=np.asarray([n, target_total_calls], dtype=np.float64),
        ub=np.asarray([n, target_total_calls], dtype=np.float64),
    )
    solution = milp(
        c=means,
        integrality=np.ones(len(calls), dtype=np.int8),
        bounds=Bounds(np.zeros(len(calls)), np.full(len(calls), n)),
        constraints=constraints,
        options={"presolve": True},
    )
    if not solution.success or solution.x is None:
        raise ValueError("no exact transition-independent mixture exists")
    counts = np.rint(solution.x).astype(np.int64)
    if counts.sum() != n or int(counts @ calls) != target_total_calls:
        raise RuntimeError("integer mixture compute accounting failed")

    selected = np.repeat(calls, counts)
    np.random.default_rng(seed).shuffle(selected)
    probabilities = counts.astype(np.float64) / n
    return {
        "selected_calls": selected,
        "supported_calls": calls.copy(),
        "counts": counts,
        "probabilities": probabilities,
        "calibration_exit_means": means.copy(),
        "calibration_mean_loss": float(probabilities @ means),
        "audit": call_audit(
            selected,
            supported_calls=calls,
            target_total_calls=target_total_calls,
            expected_mean_calls=target_total_calls / n,
        ),
    }


def randomized_histogram_control(selected_calls: Sequence[int], seed: int) -> np.ndarray:
    """Remove local allocation information while preserving calls exactly."""

    result = np.asarray(selected_calls).copy()
    if result.ndim != 1 or result.size == 0:
        raise ValueError("selected_calls must be nonempty and 1-D")
    np.random.default_rng(seed).shuffle(result)
    return result


def permute_critic_scores(
    scores: ArrayLike,
    seed: int,
    *,
    episode_ids: Sequence[Any] | None = None,
) -> np.ndarray:
    """Label-free null: independently permute each critic-depth score column.

    With ``episode_ids``, scores are permuted within episode, retaining any
    episode-level score distribution while destroying transition-local ranking.
    Without them, each column is globally permuted.  Realized gains are neither
    accepted nor inspected by this function.
    """

    score_array = _finite_2d(scores, "scores")
    rng = np.random.default_rng(seed)
    result = np.empty_like(score_array)
    if episode_ids is None:
        for column in range(score_array.shape[1]):
            result[:, column] = score_array[rng.permutation(len(score_array)), column]
        return result

    episodes = np.asarray(episode_ids)
    if episodes.ndim != 1 or len(episodes) != len(score_array):
        raise ValueError("episode_ids must match scores rows")
    result[:] = score_array
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        for column in range(score_array.shape[1]):
            result[indices, column] = score_array[rng.permutation(indices), column]
    return result


def clustered_bootstrap_ci(
    values: ArrayLike,
    episode_ids: Sequence[Any],
    *,
    samples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Percentile CI for a transition mean, resampling whole episodes.

    Episodes are sampled with replacement; every transition from each sampled
    episode is retained.  Thus long episodes keep their observed transition
    weight, matching the primary transition-level MSE estimand.
    """

    value_array = _finite_1d(values, "values")
    episodes = np.asarray(episode_ids)
    if episodes.ndim != 1 or len(episodes) != len(value_array):
        raise ValueError("episode_ids must be 1-D and match values")
    if int(samples) != samples or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    unique, inverse = np.unique(episodes, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("clustered bootstrap requires at least two episodes")
    group_sums = np.bincount(inverse, weights=value_array)
    group_counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    replicates = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        selected = rng.integers(0, len(unique), size=len(unique))
        replicates[index] = group_sums[selected].sum() / group_counts[selected].sum()
    alpha = (1.0 - confidence) / 2.0
    estimate = float(value_array.mean())
    return {
        "estimate": estimate,
        # Compatibility name makes the direction explicit for benefit arrays.
        "mean_benefit": estimate,
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
        "bootstrap_standard_error": float(replicates.std(ddof=1)) if samples > 1 else 0.0,
        "confidence": float(confidence),
        "n": int(len(value_array)),
        "n_episodes": int(len(unique)),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def clustered_paired_loss_ci(
    candidate_losses: ArrayLike,
    baseline_losses: ArrayLike,
    episode_ids: Sequence[Any],
    **bootstrap_kwargs: Any,
) -> dict[str, Any]:
    """Clustered CI for benefit = baseline loss minus candidate loss."""

    candidate = _finite_1d(candidate_losses, "candidate_losses")
    baseline = _finite_1d(baseline_losses, "baseline_losses")
    if candidate.shape != baseline.shape:
        raise ValueError("candidate_losses and baseline_losses must have equal shape")
    return clustered_bootstrap_ci(
        baseline - candidate, episode_ids, **bootstrap_kwargs
    )


def nondominated_mask(
    mean_calls: ArrayLike, errors: ArrayLike, *, atol: float = 0.0
) -> np.ndarray:
    """Return the minimization Pareto mask for compute and error."""

    calls = _finite_1d(mean_calls, "mean_calls")
    error = _finite_1d(errors, "errors")
    if calls.shape != error.shape or atol < 0:
        raise ValueError("mean_calls/errors must match and atol must be nonnegative")
    keep = np.ones(len(calls), dtype=bool)
    for index in range(len(calls)):
        weakly_better = (calls <= calls[index] + atol) & (error <= error[index] + atol)
        strictly_better = (calls < calls[index] - atol) | (error < error[index] - atol)
        weakly_better[index] = False
        keep[index] = not np.any(weakly_better & strictly_better)
    return keep


def pareto_frontier(
    mean_calls: ArrayLike, errors: ArrayLike, *, atol: float = 0.0
) -> np.ndarray:
    """Indices of nondominated points, sorted by increasing compute then error."""

    calls = _finite_1d(mean_calls, "mean_calls")
    error = _finite_1d(errors, "errors")
    mask = nondominated_mask(calls, error, atol=atol)
    indices = np.flatnonzero(mask)
    return indices[np.lexsort((error[indices], calls[indices]))]


def _optional_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def gain_ranking_calibration(
    predicted_gain: ArrayLike,
    realized_gain: ArrayLike,
    *,
    bins: int = 5,
) -> dict[str, Any]:
    """Ranking, calibration, and ordered gain-quantile diagnostics.

    Bins are equal-count score quantiles in ascending predicted-gain order.
    ``roc_auc_positive_gain`` ranks examples whose realized gain is strictly
    positive.  Undefined correlations/AUCs are returned as ``None`` so the
    result remains strict-JSON serializable.
    """

    predicted = _finite_1d(predicted_gain, "predicted_gain")
    realized = _finite_1d(realized_gain, "realized_gain")
    if predicted.shape != realized.shape:
        raise ValueError("predicted_gain and realized_gain must have equal shape")
    if int(bins) != bins or bins < 2 or bins > len(predicted):
        raise ValueError("bins must be an integer in [2, n]")
    bins = int(bins)

    predicted_constant = np.ptp(predicted) == 0
    realized_constant = np.ptp(realized) == 0
    if predicted_constant or realized_constant:
        spearman = None
    else:
        spearman = _optional_float(
            np.corrcoef(rankdata(predicted), rankdata(realized))[0, 1]
        )

    positive = realized > 0
    positives = int(positive.sum())
    negatives = int(len(realized) - positives)
    if positives and negatives:
        ranks = rankdata(predicted, method="average")
        auc = float(
            (ranks[positive].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
    else:
        auc = None

    order = np.argsort(predicted, kind="stable")
    chunks = np.array_split(order, bins)
    quantiles: list[dict[str, Any]] = []
    mean_scores = np.empty(bins, dtype=np.float64)
    mean_gains = np.empty(bins, dtype=np.float64)
    ece = 0.0
    for number, indices in enumerate(chunks):
        score_mean = float(predicted[indices].mean())
        gain_mean = float(realized[indices].mean())
        mean_scores[number] = score_mean
        mean_gains[number] = gain_mean
        ece += len(indices) / len(predicted) * abs(score_mean - gain_mean)
        quantiles.append(
            {
                "quantile": int(number),
                "n": int(len(indices)),
                "score_min": float(predicted[indices].min()),
                "score_max": float(predicted[indices].max()),
                "mean_predicted_gain": score_mean,
                "mean_realized_gain": gain_mean,
                "positive_gain_rate": float(positive[indices].mean()),
            }
        )

    differences = np.diff(mean_gains)
    if predicted_constant:
        slope = intercept = None
    else:
        design = np.column_stack((predicted, np.ones(len(predicted))))
        fitted, _, _, _ = np.linalg.lstsq(design, realized, rcond=None)
        slope, intercept = map(float, fitted)

    residual = predicted - realized
    return {
        "n": int(len(predicted)),
        "positive_gain_rate": float(positive.mean()),
        "spearman_rho": spearman,
        "roc_auc_positive_gain": auc,
        "calibration_bias_predicted_minus_realized": float(residual.mean()),
        "calibration_mae": float(np.abs(residual).mean()),
        "calibration_rmse": float(np.sqrt(np.mean(residual**2))),
        "calibration_ece": float(ece),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "top_bottom_realized_gain_gap": float(mean_gains[-1] - mean_gains[0]),
        "ordered_quantiles": bool(np.all(differences >= 0)),
        "ordered_adjacent_fraction": float(np.mean(differences >= 0)),
        "quantiles": quantiles,
    }


def per_depth_gain_diagnostics(
    predicted_gains: ArrayLike,
    realized_gains: ArrayLike,
    *,
    bins: int = 5,
) -> list[dict[str, Any]]:
    """Run gain diagnostics independently at every sequential decision."""

    predicted = _finite_2d(predicted_gains, "predicted_gains")
    realized = _finite_2d(realized_gains, "realized_gains")
    if predicted.shape != realized.shape:
        raise ValueError("predicted_gains and realized_gains must have equal shape")
    result = []
    for depth in range(predicted.shape[1]):
        diagnostics = gain_ranking_calibration(
            predicted[:, depth], realized[:, depth], bins=bins
        )
        diagnostics["decision_index"] = int(depth)
        result.append(diagnostics)
    return result


# Compatibility wrappers used by the preceding V3 scripts.
def local_depths(scores: ArrayLike, threshold: float = 0.0) -> np.ndarray:
    return causal_sequential_stopping(scores, thresholds=threshold)


def clustered_ci(
    values: ArrayLike, episode_ids: Sequence[Any], samples: int, seed: int
) -> dict[str, Any]:
    return clustered_bootstrap_ci(
        values, episode_ids, samples=samples, seed=seed
    )


histogram_randomized = randomized_histogram_control


__all__ = [
    "call_audit",
    "calibrate_compute_price",
    "causal_sequential_stopping",
    "clustered_bootstrap_ci",
    "clustered_ci",
    "clustered_paired_loss_ci",
    "fixed_exit",
    "gain_ranking_calibration",
    "gather_exit_values",
    "histogram_randomized",
    "local_depths",
    "nondominated_mask",
    "optimal_expected_mixture",
    "pareto_frontier",
    "per_depth_gain_diagnostics",
    "permute_critic_scores",
    "randomized_histogram_control",
    "sequential_stopping",
    "strongest_transition_independent_baseline",
]
