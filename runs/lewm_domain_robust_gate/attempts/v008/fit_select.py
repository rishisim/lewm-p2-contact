#!/usr/bin/env python3
"""Fit and select the preregistered domain-robust causal gate family.

This module deliberately has no world-model or episode-generation dependency.
Callers provide already materialized, role-isolated arrays.  The fitter reads
only the fit role, compiles every fitted head to an affine function of the raw
1,046 causal features, and seals all 24 candidates before a selection loader is
ever invoked.  Selection uses outcome values only through the fixed mechanical
eligibility checks and the preregistered lexicographic ranking rule.

The two co-primary endpoints are raw latent MSE and the unchanged V5
fixed-whitened latent MSE.  The whitening matrix derived from the new fit role
is retained as an auxiliary endpoint only and is never a gate head or a
selection-ranking endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


FEATURE_DIM = 1_046
STAGE_COUNT = 3
EXIT_COUNT = 4
ENDPOINT_NAMES = ("raw", "fixed_whitened")
DGP_IDS = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ARCHITECTURES = ("balanced_pooled_dual", "domain_envelope_eight")
RIDGES = (0.01, 1.0, 100.0)
FIT_QUANTILES = (0.55, 0.65, 0.75, 0.85)
CANDIDATE_COUNT = len(ARCHITECTURES) * len(RIDGES) * len(FIT_QUANTILES)
MAX_HEADS = 8

BASE_FLOPS_PER_ROW = 70_529_190
DEPTH1_FLOPS_PER_ROW = 669_184
ADAPTER_FLOPS_PER_ADDITIONAL_CALL = 264_960
ARCHITECTURE_GATE_FLOPS = {
    "balanced_pooled_dual": 7_985,
    "domain_envelope_eight": 20_537,
}
ARCHITECTURE_GATE_NONFLOPS = {
    "balanced_pooled_dual": 5,
    "domain_envelope_eight": 11,
}
ARCHITECTURE_HEADS = {
    "balanced_pooled_dual": 2,
    "domain_envelope_eight": 8,
}

V5_FIXED_WHITENING_SHA256 = (
    "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796"
)
_FORBIDDEN_ARRAY_NAMES = {
    "contact",
    "contacts",
    "contact_state",
    "privileged",
    "privileged_state",
    "motion",
    "phase",
    "reward",
    "rewards",
    "success",
    "successes",
}


@dataclass(frozen=True)
class RoleData:
    """Validated arrays for one isolated development role."""

    features: np.ndarray
    exits: np.ndarray
    target: np.ndarray
    episode_slot: np.ndarray
    model_step: np.ndarray
    dgp_index: np.ndarray
    dgp_ids: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.target.shape[1])


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def candidate_object_sha256(
    *,
    candidate_id: str,
    architecture: str,
    ridge: float,
    quantile: float,
    weights: np.ndarray,
    biases: np.ndarray,
    thresholds: np.ndarray,
    head_names: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    metadata = json.dumps(
        {
            "candidate_id": candidate_id,
            "architecture": architecture,
            "ridge": float(ridge),
            "fit_quantile": float(quantile),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(metadata)
    for value in (weights, biases, thresholds, head_names):
        digest.update(array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _atomic_npz_exclusive(path: Path | str, arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"immutable artifact already exists: {destination}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json_exclusive(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"immutable artifact already exists: {destination}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_fixed_whitening(
    path: Path | str,
    *,
    expected_sha256: str | None = V5_FIXED_WHITENING_SHA256,
) -> np.ndarray:
    """Load the unchanged V5 matrix, refusing an unexpected file or key."""

    source = Path(path)
    observed = sha256_file(source)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(
            f"fixed V5 whitening hash drift: expected {expected_sha256}, got {observed}"
        )
    with np.load(source, allow_pickle=False) as stored:
        if "whitening_matrix" not in stored.files:
            raise RuntimeError("fixed whitening archive lacks whitening_matrix")
        matrix = stored["whitening_matrix"].astype(np.float64, copy=True)
    _validate_whitening(matrix)
    return matrix


def load_per_dgp_npz(paths: Mapping[str, Path | str]) -> dict[str, dict[str, np.ndarray]]:
    """Read a per-DGP role without requesting any forbidden field."""

    result: dict[str, dict[str, np.ndarray]] = {}
    for dgp_id in DGP_IDS:
        if dgp_id not in paths:
            raise ValueError(f"missing DGP role path: {dgp_id}")
        with np.load(Path(paths[dgp_id]), allow_pickle=False) as stored:
            forbidden = _FORBIDDEN_ARRAY_NAMES.intersection(name.lower() for name in stored.files)
            if forbidden:
                raise RuntimeError(
                    f"forbidden arrays present in development role {dgp_id}: {sorted(forbidden)}"
                )
            required = {
                "episode_slot", "model_step", "target", "exits", "production_features"
            }
            missing = required.difference(stored.files)
            if missing:
                raise RuntimeError(f"{dgp_id} role archive missing {sorted(missing)}")
            result[dgp_id] = {name: stored[name].copy() for name in required}
    return result


def prepare_role(
    per_dgp: Mapping[str, Mapping[str, np.ndarray]],
    *,
    dgp_ids: Sequence[str] = DGP_IDS,
) -> RoleData:
    """Validate and concatenate the four DGP-specific role mappings.

    The only gate input read here is ``production_features`` (``features`` is
    accepted as a synthetic-test alias).  Targets and dense exits are used for
    fit/selection evaluation, never for a routing decision.
    """

    order = tuple(str(value) for value in dgp_ids)
    if len(order) != 4 or len(set(order)) != 4:
        raise ValueError("exactly four unique DGP identifiers are required")
    feature_parts: list[np.ndarray] = []
    exit_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    dgp_parts: list[np.ndarray] = []
    latent_dim: int | None = None
    for dgp_index, dgp_id in enumerate(order):
        if dgp_id not in per_dgp:
            raise ValueError(f"missing DGP arrays: {dgp_id}")
        arrays = per_dgp[dgp_id]
        forbidden = _FORBIDDEN_ARRAY_NAMES.intersection(str(key).lower() for key in arrays)
        if forbidden:
            raise ValueError(f"forbidden arrays supplied for {dgp_id}: {sorted(forbidden)}")
        feature_key = "production_features" if "production_features" in arrays else "features"
        required = {feature_key, "exits", "target", "episode_slot", "model_step"}
        missing = required.difference(arrays)
        if missing:
            raise ValueError(f"{dgp_id} arrays missing {sorted(missing)}")
        features = np.asarray(arrays[feature_key], dtype=np.float64)
        exits = np.asarray(arrays["exits"], dtype=np.float64)
        target = np.asarray(arrays["target"], dtype=np.float64)
        episode = np.asarray(arrays["episode_slot"])
        model_step = np.asarray(arrays["model_step"])
        n = len(features)
        if features.shape != (n, STAGE_COUNT, FEATURE_DIM):
            raise ValueError(
                f"{dgp_id} production_features must have shape [N,3,{FEATURE_DIM}]"
            )
        if target.ndim != 2 or target.shape[0] != n or target.shape[1] <= 0:
            raise ValueError(f"{dgp_id} target must have shape [N,latent_dim]")
        if exits.shape != (n, EXIT_COUNT, target.shape[1]):
            raise ValueError(f"{dgp_id} exits must have shape [N,4,latent_dim]")
        if latent_dim is None:
            latent_dim = int(target.shape[1])
        elif target.shape[1] != latent_dim:
            raise ValueError("latent dimension differs across DGPs")
        if episode.shape != (n,) or model_step.shape != (n,):
            raise ValueError(f"{dgp_id} identifiers must be one-dimensional")
        if not np.issubdtype(episode.dtype, np.integer):
            if not np.equal(episode, np.round(episode)).all():
                raise ValueError(f"{dgp_id} episode_slot must be integral")
        if not np.issubdtype(model_step.dtype, np.integer):
            if not np.equal(model_step, np.round(model_step)).all():
                raise ValueError(f"{dgp_id} model_step must be integral")
        episode = episode.astype(np.int64)
        model_step = model_step.astype(np.int64)
        if np.any(episode < 0) or np.any(model_step < 0):
            raise ValueError(f"{dgp_id} identifiers must be nonnegative")
        pairs = np.column_stack((episode, model_step))
        if len(np.unique(pairs, axis=0)) != n:
            raise ValueError(f"{dgp_id} has duplicate episode_slot/model_step rows")
        if n == 0 or not all(
            np.isfinite(value).all() for value in (features, exits, target)
        ):
            raise ValueError(f"{dgp_id} role arrays must be nonempty and finite")
        feature_parts.append(features)
        exit_parts.append(exits)
        target_parts.append(target)
        episode_parts.append(episode)
        step_parts.append(model_step)
        dgp_parts.append(np.full(n, dgp_index, dtype=np.int64))
    return RoleData(
        features=np.concatenate(feature_parts, axis=0),
        exits=np.concatenate(exit_parts, axis=0),
        target=np.concatenate(target_parts, axis=0),
        episode_slot=np.concatenate(episode_parts, axis=0),
        model_step=np.concatenate(step_parts, axis=0),
        dgp_index=np.concatenate(dgp_parts, axis=0),
        dgp_ids=order,
    )


def validate_role_cardinality(
    role: RoleData,
    *,
    episodes_per_dgp: int,
    rows_per_episode: int = 38,
    first_model_step: int = 3,
) -> None:
    """Enforce the sealed role cardinality and row ordering for each DGP."""

    for dgp, dgp_id in enumerate(role.dgp_ids):
        mask = role.dgp_index == dgp
        episodes = role.episode_slot[mask]
        steps = role.model_step[mask]
        expected_episodes = np.arange(episodes_per_dgp, dtype=np.int64)
        if not np.array_equal(np.unique(episodes), expected_episodes):
            raise RuntimeError(f"{dgp_id} episode slots violate sealed cardinality")
        expected_episode_vector = np.repeat(expected_episodes, rows_per_episode)
        expected_steps = np.tile(
            np.arange(
                first_model_step,
                first_model_step + rows_per_episode,
                dtype=np.int64,
            ),
            episodes_per_dgp,
        )
        if not np.array_equal(episodes, expected_episode_vector):
            raise RuntimeError(f"{dgp_id} episode row order drift")
        if not np.array_equal(steps, expected_steps):
            raise RuntimeError(f"{dgp_id} model-step row order drift")


def _validate_whitening(matrix: np.ndarray, latent_dim: int | None = None) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("whitening matrix must be square")
    if latent_dim is not None and matrix.shape != (latent_dim, latent_dim):
        raise ValueError("whitening matrix latent dimension mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("whitening matrix must be finite")


def _balanced_mean_std(
    values: np.ndarray, dgp_index: np.ndarray, dgp_count: int, *, floor: float
) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([values[dgp_index == d].mean(axis=0) for d in range(dgp_count)])
    mean = means.mean(axis=0)
    second = np.stack(
        [np.square(values[dgp_index == d] - mean).mean(axis=0) for d in range(dgp_count)]
    ).mean(axis=0)
    std = np.sqrt(np.maximum(second, 0.0))
    std = np.maximum(std, floor)
    return mean, std


def _domain_mean_std(values: np.ndarray, *, floor: float) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.maximum(std, floor)
    return mean, std


def fit_robust_whitening(role: RoleData) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit the preregistered pooled fit-only whitening auxiliary."""

    # Fit cohorts have identical cardinality in every DGP, so ordinary pooled
    # moments give each DGP equal mass while retaining between-DGP variation.
    mean = role.target.mean(axis=0)
    centered = role.target - mean
    covariance = np.einsum(
        "ni,nj->ij", centered, centered, optimize=False
    ) / max(len(centered) - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(eigenvalues[-1]), 0.0)
    floor = max(largest * 1e-6, 1e-12)
    clipped = np.maximum(eigenvalues, floor)
    whitening = np.einsum(
        "ik,k,jk->ij",
        eigenvectors,
        np.reciprocal(np.sqrt(clipped)),
        eigenvectors,
        optimize=False,
    )
    if not np.isfinite(whitening).all():
        raise RuntimeError("fit-derived robust whitening is nonfinite")
    return whitening, mean, floor


def endpoint_losses(
    role: RoleData, whitening: np.ndarray
) -> np.ndarray:
    """Return raw and fixed-whitened losses with shape [N,4,2]."""

    _validate_whitening(whitening, role.latent_dim)
    difference = role.exits - role.target[:, None, :]
    raw = np.square(difference).mean(axis=2)
    transformed = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    fixed = np.square(transformed).mean(axis=2)
    result = np.stack((raw, fixed), axis=2)
    if not np.isfinite(result).all():
        raise RuntimeError("endpoint losses are nonfinite")
    return result


def _balanced_row_weights(dgp_index: np.ndarray, dgp_count: int) -> np.ndarray:
    n = len(dgp_index)
    weights = np.empty(n, dtype=np.float64)
    for dgp in range(dgp_count):
        count = int(np.sum(dgp_index == dgp))
        if count == 0:
            raise ValueError("each DGP must contain fit rows")
        weights[dgp_index == dgp] = n / (dgp_count * count)
    return weights


def _ridge_path(
    features: np.ndarray,
    targets: np.ndarray,
    ridges: Sequence[float],
    *,
    row_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit zero-intercept ridge heads, using the smaller primal/dual system."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("ridge inputs must be aligned nonempty matrices")
    if row_weights is None:
        root_weight = np.ones(len(x), dtype=np.float64)
    else:
        weights = np.asarray(row_weights, dtype=np.float64)
        if weights.shape != (len(x),) or np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("ridge row weights must be positive and finite")
        root_weight = np.sqrt(weights)
    xw = x * root_weight[:, None]
    yw = y * root_weight[:, None]
    result = np.empty((len(ridges), x.shape[1], y.shape[1]), dtype=np.float64)
    if len(x) <= x.shape[1]:
        gram = np.einsum("ik,jk->ij", xw, xw, optimize=False)
        identity = np.eye(len(x), dtype=np.float64)
        for index, ridge in enumerate(ridges):
            dual = np.linalg.solve(gram + float(ridge) * identity, yw)
            result[index] = np.einsum(
                "nd,nt->dt", xw, dual, optimize=False
            )
    else:
        gram = np.einsum("nd,ne->de", xw, xw, optimize=False)
        cross = np.einsum("nd,nt->dt", xw, yw, optimize=False)
        identity = np.eye(x.shape[1], dtype=np.float64)
        for index, ridge in enumerate(ridges):
            result[index] = np.linalg.solve(gram + float(ridge) * identity, cross)
    if not np.isfinite(result).all():
        raise RuntimeError("ridge fit produced nonfinite coefficients")
    return result


def compile_affine_head(
    normalized_weight: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    normalized_bias: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Fold a fit-only feature transform into a raw-feature affine head."""

    weight = np.asarray(normalized_weight, dtype=np.float64)
    mean = np.asarray(feature_mean, dtype=np.float64)
    std = np.asarray(feature_std, dtype=np.float64)
    if weight.shape != (FEATURE_DIM,) or mean.shape != weight.shape or std.shape != weight.shape:
        raise ValueError(f"affine compilation expects width {FEATURE_DIM}")
    if np.any(std <= 0) or not all(np.isfinite(v).all() for v in (weight, mean, std)):
        raise ValueError("invalid affine compilation inputs")
    raw_weight = weight / std
    raw_bias = float(normalized_bias - np.dot(mean / std, weight))
    if not np.isfinite(raw_weight).all() or not np.isfinite(raw_bias):
        raise RuntimeError("compiled affine head is nonfinite")
    return raw_weight, raw_bias


def score_compiled_gate(
    features: np.ndarray, weights: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    """Evaluate the frozen runtime gate using contiguous float32 arithmetic."""

    x = np.ascontiguousarray(features, dtype=np.float32)
    w = np.asarray(weights)
    b = np.asarray(biases)
    if x.ndim != 3 or x.shape[1:] != (STAGE_COUNT, FEATURE_DIM):
        raise ValueError("features must have shape [N,3,1046]")
    if w.ndim != 3 or w.shape[0] != STAGE_COUNT or w.shape[2] != FEATURE_DIM:
        raise ValueError("weights must have shape [3,H,1046]")
    if b.shape != w.shape[:2] or w.shape[1] not in (2, 8):
        raise ValueError("biases/head count mismatch")
    if w.dtype != np.dtype(np.float32) or b.dtype != np.dtype(np.float32):
        raise TypeError("frozen runtime weights and biases must be exactly float32")
    w = np.ascontiguousarray(w)
    b = np.ascontiguousarray(b)
    scores = np.empty((len(x), STAGE_COUNT), dtype=np.float32)
    for stage in range(STAGE_COUNT):
        heads = (
            np.einsum(
                "nd,hd->nh", x[:, stage], w[stage], optimize=False
            )
            + b[stage][None, :]
        )
        scores[:, stage] = heads.min(axis=1)
    if scores.dtype != np.dtype(np.float32) or not np.isfinite(scores).all():
        raise RuntimeError("compiled gate scores are nonfinite")
    return scores


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.ndim != 1 or weights.shape != values.shape or len(values) == 0:
        raise ValueError("weighted quantile inputs must be aligned vectors")
    if not 0.0 <= quantile <= 1.0 or np.any(weights <= 0):
        raise ValueError("invalid weighted quantile")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def sequential_fit_thresholds(
    scores: np.ndarray,
    dgp_index: np.ndarray,
    quantile: float,
    *,
    dgp_count: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ordinary NumPy-linear quantiles sequentially on reached fit rows."""

    values = np.asarray(scores)
    domains = np.asarray(dgp_index, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != STAGE_COUNT or domains.shape != (len(values),):
        raise ValueError("sequential threshold shapes disagree")
    if set(domains.tolist()) != set(range(dgp_count)):
        raise ValueError("sequential threshold DGP membership is incomplete")
    if not np.isfinite(values).all():
        raise ValueError("sequential threshold scores must be finite")
    if values.dtype != np.dtype(np.float32):
        raise TypeError("fit gate scores must be exactly float32")
    thresholds = np.empty(STAGE_COUNT, dtype=np.float32)
    calls = np.ones(len(values), dtype=np.int64)
    active = np.ones(len(values), dtype=bool)
    reached = np.empty((len(values), STAGE_COUNT), dtype=bool)
    for stage in range(STAGE_COUNT):
        reached[:, stage] = active
        if not bool(active.any()):
            raise RuntimeError(f"no fit rows reach threshold stage {stage + 1}")
        thresholds[stage] = np.float32(
            np.quantile(values[active, stage], float(quantile), method="linear")
        )
        active &= values[:, stage] > thresholds[stage]
        calls += active.astype(np.int64)
    return thresholds, calls, reached


def calls_from_scores(
    scores: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores)
    cuts = np.asarray(thresholds)
    if values.ndim != 2 or values.shape[1] != STAGE_COUNT or cuts.shape != (STAGE_COUNT,):
        raise ValueError("score/threshold shape mismatch")
    if not np.isfinite(values).all() or not np.isfinite(cuts).all():
        raise ValueError("scores and thresholds must be finite")
    if values.dtype != np.dtype(np.float32) or cuts.dtype != np.dtype(np.float32):
        raise TypeError("runtime scores and thresholds must be exactly float32")
    calls = np.ones(len(values), dtype=np.int64)
    active = np.ones(len(values), dtype=bool)
    reached = np.empty((len(values), STAGE_COUNT), dtype=bool)
    for stage in range(STAGE_COUNT):
        reached[:, stage] = active
        active &= values[:, stage] > cuts[stage]
        calls += active.astype(np.int64)
    return calls, reached


def sparse_runtime_scores(
    dense_scores: np.ndarray, reached: np.ndarray
) -> np.ndarray:
    """Materialize the runtime trace contract: NaN exactly where unreached."""

    values = np.asarray(dense_scores)
    mask = np.asarray(reached, dtype=bool)
    if values.dtype != np.dtype(np.float32) or values.shape != mask.shape:
        raise ValueError("dense score/reached trace contract mismatch")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    result[mask] = values[mask]
    return result


def calls_from_sparse_runtime_scores(
    scores: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct calls from the exact sparse float32 runtime trace."""

    values = np.asarray(scores)
    cuts = np.asarray(thresholds)
    if values.ndim != 2 or values.shape[1] != STAGE_COUNT:
        raise ValueError("runtime sparse scores must have shape [N,3]")
    if cuts.shape != (STAGE_COUNT,):
        raise ValueError("runtime thresholds must have shape [3]")
    if values.dtype != np.dtype(np.float32) or cuts.dtype != np.dtype(np.float32):
        raise TypeError("runtime sparse scores and thresholds must be float32")
    calls = np.ones(len(values), dtype=np.int64)
    active = np.ones(len(values), dtype=bool)
    reached = np.empty(values.shape, dtype=bool)
    for stage in range(STAGE_COUNT):
        reached[:, stage] = active
        if not np.isfinite(values[active, stage]).all():
            raise RuntimeError("a reached runtime score is nonfinite")
        if np.isfinite(values[~active, stage]).any():
            raise RuntimeError("an unreached runtime score is finite")
        active &= values[:, stage] > cuts[stage]
        calls += active.astype(np.int64)
    return calls, reached


def audit_dense_sparse_call_equivalence(
    dense_scores: np.ndarray, thresholds: np.ndarray
) -> dict[str, Any]:
    """Prove dense candidate routing equals the sparse runtime trace."""

    dense = np.asarray(dense_scores)
    dense_calls, dense_reached = calls_from_scores(dense, thresholds)
    sparse = sparse_runtime_scores(dense, dense_reached)
    sparse_calls, sparse_reached = calls_from_sparse_runtime_scores(
        sparse, thresholds
    )
    calls_equal = bool(np.array_equal(dense_calls, sparse_calls))
    reached_equal = bool(np.array_equal(dense_reached, sparse_reached))
    if not calls_equal or not reached_equal:
        raise RuntimeError("dense and sparse runtime call reconstruction differ")
    return {
        "scores": dense,
        "sparse_scores": sparse,
        "calls": dense_calls,
        "reached": dense_reached,
        "dense_sparse_calls_exact": calls_equal,
        "dense_sparse_reached_exact": reached_equal,
        "runtime_dtype": "float32",
        "continue_operator": "strict_greater_than",
        "threshold_ties_stop": True,
    }


def audit_runtime_call_equivalence(
    features: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    """Score raw features and prove complete sparse-runtime call equivalence."""

    return audit_dense_sparse_call_equivalence(
        score_compiled_gate(features, weights, biases), thresholds
    )


def _episode_means(values: np.ndarray, episode_slot: np.ndarray) -> np.ndarray:
    episodes = np.unique(episode_slot)
    return np.asarray([values[episode_slot == episode].mean() for episode in episodes])


def _strongest_analytic_mixture(
    losses: np.ndarray, total_counted_flops: int
) -> dict[str, Any]:
    """Return the best transition-independent allocation at an exact budget.

    The allocation weight is derived as a :class:`Fraction` from integer
    counted FLOPs.  Floating point is used only to mix losses; compute
    equality is proved by integer cross multiplication.
    """

    if losses.ndim != 2 or losses.shape[1] != EXIT_COUNT or not np.isfinite(losses).all():
        raise ValueError("analytic losses must have shape [N,4]")
    rows = int(losses.shape[0])
    budget = int(total_counted_flops)
    fixed_cost = tuple(
        BASE_FLOPS_PER_ROW
        + DEPTH1_FLOPS_PER_ROW
        + (depth - 1) * ADAPTER_FLOPS_PER_ADDITIONAL_CALL
        for depth in range(1, EXIT_COUNT + 1)
    )
    if not rows * fixed_cost[0] <= budget <= rows * fixed_cost[-1]:
        raise ValueError("analytic counted-compute budget is outside [depth 1,4]")
    best: tuple[float, int, int, Fraction, np.ndarray] | None = None
    for lower in range(1, EXIT_COUNT + 1):
        for upper in range(lower, EXIT_COUNT + 1):
            lower_total = rows * fixed_cost[lower - 1]
            if lower == upper:
                if budget != lower_total:
                    continue
                weight_upper = Fraction(0, 1)
            else:
                denominator = rows * (fixed_cost[upper - 1] - fixed_cost[lower - 1])
                numerator = budget - lower_total
                if numerator < 0 or numerator > denominator:
                    continue
                weight_upper = Fraction(numerator, denominator)
            weight_float = float(weight_upper)
            mixture = (
                (1.0 - weight_float) * losses[:, lower - 1]
                + weight_float * losses[:, upper - 1]
            )
            candidate = (float(mixture.mean()), lower, upper, weight_upper, mixture)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise RuntimeError("no exact-compute analytic allocation")
    objective, lower, upper, weight_upper, mixture = best
    lower_total = rows * fixed_cost[lower - 1]
    delta_total = rows * (fixed_cost[upper - 1] - fixed_cost[lower - 1])
    allocation_numerator = (
        lower_total * weight_upper.denominator
        + delta_total * weight_upper.numerator
    )
    allocation_denominator = weight_upper.denominator
    if allocation_numerator != budget * allocation_denominator:
        raise RuntimeError("analytic allocation failed exact integer compute identity")
    return {
        "mean_loss": objective,
        "depth_lower": lower,
        "depth_upper": upper,
        "weight_upper": float(weight_upper),
        "weight_upper_numerator": int(weight_upper.numerator),
        "weight_upper_denominator": int(weight_upper.denominator),
        "adaptive_total_counted_flops": budget,
        "allocation_total_counted_flops": budget,
        "allocation_total_counted_flops_numerator": int(allocation_numerator),
        "allocation_total_counted_flops_denominator": int(allocation_denominator),
        "exact_integer_cross_product_identity": True,
        "exact_total_compute_match": True,
        "loss": mixture,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        return float("nan")
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(np.dot(rx, ry) / denominator) if denominator > 0 else float("nan")


def _json_float(value: float) -> float | None:
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def _head_names(architecture: str, dgp_ids: Sequence[str]) -> tuple[str, ...]:
    if architecture == "balanced_pooled_dual":
        return ENDPOINT_NAMES
    if architecture == "domain_envelope_eight":
        return tuple(f"{dgp}/{endpoint}" for dgp in dgp_ids for endpoint in ENDPOINT_NAMES)
    raise ValueError(f"unknown architecture: {architecture}")


def _candidate_id(architecture: str, ridge: float, quantile: float) -> str:
    ridge_label = f"{ridge:g}".replace(".", "p")
    quantile_label = f"{quantile:.2f}".replace(".", "p")
    return f"{architecture}__ridge_{ridge_label}__q_{quantile_label}"


def _candidate_grid() -> list[tuple[str, float, float, str]]:
    return [
        (architecture, float(ridge), float(quantile), _candidate_id(architecture, ridge, quantile))
        for architecture in ARCHITECTURES
        for ridge in RIDGES
        for quantile in FIT_QUANTILES
    ]


def _compute_accounting(calls: np.ndarray, architecture: str) -> dict[str, Any]:
    if calls.ndim != 1 or len(calls) == 0 or not np.isin(calls, (1, 2, 3, 4)).all():
        raise ValueError("calls must be a nonempty vector in {1,2,3,4}")
    rows = len(calls)
    refiner_calls = int(calls.sum(dtype=np.int64))
    gate_evaluations = int(np.minimum(calls, STAGE_COUNT).sum(dtype=np.int64))
    gate_flops = int(ARCHITECTURE_GATE_FLOPS[architecture])
    gate_nonflops = int(ARCHITECTURE_GATE_NONFLOPS[architecture])
    common = rows * (BASE_FLOPS_PER_ROW + DEPTH1_FLOPS_PER_ROW)
    total = int(
        common
        + (refiner_calls - rows) * ADAPTER_FLOPS_PER_ADDITIONAL_CALL
        + gate_evaluations * gate_flops
    )
    equivalent_total_calls = Fraction(
        refiner_calls * ADAPTER_FLOPS_PER_ADDITIONAL_CALL
        + gate_evaluations * gate_flops,
        ADAPTER_FLOPS_PER_ADDITIONAL_CALL,
    )
    reconstructed_total_numerator = (
        common * equivalent_total_calls.denominator
        + (
            equivalent_total_calls.numerator
            - rows * equivalent_total_calls.denominator
        )
        * ADAPTER_FLOPS_PER_ADDITIONAL_CALL
    )
    if reconstructed_total_numerator != total * equivalent_total_calls.denominator:
        raise RuntimeError("equivalent-call fraction failed exact compute identity")
    return {
        "rows": rows,
        "refiner_calls": refiner_calls,
        "gate_evaluations": gate_evaluations,
        "gate_flops_per_evaluation": gate_flops,
        "gate_nonflops_per_evaluation": gate_nonflops,
        "gate_flops": gate_evaluations * gate_flops,
        "gate_nonflops": gate_evaluations * gate_nonflops,
        "total_flops": total,
        "flops_per_row": float(total / rows),
        "analytic_equivalent_total_calls": float(equivalent_total_calls),
        "analytic_equivalent_total_calls_numerator": int(
            equivalent_total_calls.numerator
        ),
        "analytic_equivalent_total_calls_denominator": int(
            equivalent_total_calls.denominator
        ),
        "analytic_equivalent_mean_calls": float(equivalent_total_calls / rows),
        "equivalent_call_exact_integer_identity": True,
    }


def comparator_seeds_from_ledger(
    cohort_ledger: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Extract the twelve prospectively assigned comparator RNG identifiers."""

    analysis_rng = cohort_ledger.get("analysis_rng_ids")
    if not isinstance(analysis_rng, Mapping):
        raise ValueError("cohort ledger lacks analysis_rng_ids")
    comparator = analysis_rng.get("comparator")
    if not isinstance(comparator, Mapping) or not isinstance(
        comparator.get("rng_ids"), list
    ):
        raise ValueError("cohort ledger lacks comparator rng_ids")
    purpose_to_key = {
        "seeded_weak_more_raw": "raw",
        "seeded_weak_more_fixed_whitened": "fixed_whitened",
        "within_episode_call_histogram": "histogram",
    }
    result = {dgp: {} for dgp in DGP_IDS}
    for record in comparator["rng_ids"]:
        if not isinstance(record, Mapping):
            raise ValueError("invalid comparator RNG record")
        purpose = str(record.get("purpose"))
        dgp = str(record.get("regime"))
        if purpose not in purpose_to_key or dgp not in result:
            raise ValueError("unexpected comparator RNG assignment")
        key = purpose_to_key[purpose]
        if key in result[dgp]:
            raise ValueError("duplicate comparator RNG assignment")
        seed = int(record["rng_id"])
        if not 0 <= seed < 2**64:
            raise ValueError("comparator RNG identifier is outside uint64")
        result[dgp][key] = seed
    expected = set(purpose_to_key.values())
    if any(set(values) != expected for values in result.values()):
        raise ValueError("incomplete comparator RNG assignments")
    flattened = [seed for values in result.values() for seed in values.values()]
    if len(set(flattened)) != 12:
        raise ValueError("comparator RNG identifiers are not unique")
    return result


def _validate_comparator_seeds(
    seeds: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    if set(seeds) != set(DGP_IDS):
        raise ValueError("comparator seeds must contain exactly four DGPs")
    result: dict[str, dict[str, int]] = {}
    expected = {"raw", "fixed_whitened", "histogram"}
    for dgp in DGP_IDS:
        values = seeds[dgp]
        if set(values) != expected:
            raise ValueError(f"comparator seeds incomplete for {dgp}")
        result[dgp] = {key: int(values[key]) for key in sorted(expected)}
        if any(not 0 <= value < 2**64 for value in result[dgp].values()):
            raise ValueError("comparator seed is outside uint64")
    flattened = [value for values in result.values() for value in values.values()]
    if len(set(flattened)) != 12:
        raise ValueError("comparator seeds must be prospectively unique")
    return result


def _evaluate_candidate(
    role: RoleData,
    *,
    architecture: str,
    candidate_id: str,
    weights: np.ndarray,
    biases: np.ndarray,
    thresholds: np.ndarray,
    fixed_whitening: np.ndarray,
    robust_whitening: np.ndarray,
    gain_mean: np.ndarray,
    gain_std: np.ndarray,
    fit_contrast_sd: np.ndarray | None,
    comparator_seeds: Mapping[str, Mapping[str, int]],
    precomputed_scores: np.ndarray | None = None,
    precomputed_primary_losses: np.ndarray | None = None,
    precomputed_auxiliary_losses: np.ndarray | None = None,
) -> dict[str, Any]:
    assigned_seeds = _validate_comparator_seeds(comparator_seeds)
    scores = (
        score_compiled_gate(role.features, weights, biases)
        if precomputed_scores is None
        else np.asarray(precomputed_scores)
    )
    if (
        scores.shape != (role.row_count, STAGE_COUNT)
        or scores.dtype != np.dtype(np.float32)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("precomputed candidate scores are invalid")
    call_audit = audit_dense_sparse_call_equivalence(scores, thresholds)
    calls = call_audit["calls"]
    reached = call_audit["reached"]
    primary_losses = (
        endpoint_losses(role, fixed_whitening)
        if precomputed_primary_losses is None
        else np.asarray(precomputed_primary_losses, dtype=np.float64)
    )
    auxiliary_losses = (
        endpoint_losses(role, robust_whitening)[:, :, 1]
        if precomputed_auxiliary_losses is None
        else np.asarray(precomputed_auxiliary_losses, dtype=np.float64)
    )
    if primary_losses.shape != (role.row_count, EXIT_COUNT, 2):
        raise ValueError("precomputed primary loss shape mismatch")
    if auxiliary_losses.shape != (role.row_count, EXIT_COUNT):
        raise ValueError("precomputed auxiliary loss shape mismatch")
    result_dgps: list[dict[str, Any]] = []
    standardized_values: list[float] = []
    rank_values: list[float] = []
    all_valid = True
    for dgp, dgp_id in enumerate(role.dgp_ids):
        mask = role.dgp_index == dgp
        local_calls = calls[mask]
        local_reached = reached[mask]
        local_scores = scores[mask]
        local_episode = role.episode_slot[mask]
        local_primary = primary_losses[mask]
        local_auxiliary = auxiliary_losses[mask]
        n = len(local_calls)
        local_index = np.arange(n)
        adaptive_primary = local_primary[local_index, local_calls - 1, :]
        adaptive_auxiliary = local_auxiliary[local_index, local_calls - 1]
        compute = _compute_accounting(local_calls, architecture)
        analytic_budget = int(compute["total_flops"])
        depth1_total = n * (
            BASE_FLOPS_PER_ROW + DEPTH1_FLOPS_PER_ROW
        )
        depth4_total = depth1_total + n * 3 * ADAPTER_FLOPS_PER_ADDITIONAL_CALL
        analytic_supported = depth1_total <= analytic_budget <= depth4_total
        analytic = (
            [
                _strongest_analytic_mixture(
                    local_primary[:, :, endpoint], analytic_budget
                )
                for endpoint in range(2)
            ]
            if analytic_supported
            else []
        )
        auxiliary_analytic = (
            _strongest_analytic_mixture(local_auxiliary, analytic_budget)
            if analytic_supported
            else None
        )
        contrast_episode = []
        contrast_means = []
        for endpoint in range(2):
            episode_values = (
                _episode_means(
                    analytic[endpoint]["loss"] - adaptive_primary[:, endpoint],
                    local_episode,
                )
                if analytic_supported
                else np.asarray([float("nan")])
            )
            contrast_episode.append(episode_values)
            contrast_means.append(float(episode_values.mean()))
            if fit_contrast_sd is not None:
                denominator = float(fit_contrast_sd[dgp, endpoint])
                standardized = (
                    float(episode_values.mean() / max(denominator, 1e-12))
                    if np.isfinite(denominator) and denominator >= 0
                    else float("nan")
                )
                standardized_values.append(standardized)
        auxiliary_values = (
            auxiliary_analytic["loss"] - adaptive_auxiliary
            if auxiliary_analytic is not None
            else np.full(n, float("nan"), dtype=np.float64)
        )

        # Weakly-more-compute seeded transition-independent control.
        numerator = int(compute["analytic_equivalent_total_calls_numerator"])
        denominator = int(compute["analytic_equivalent_total_calls_denominator"])
        integer_target = (numerator + denominator - 1) // denominator
        seeded_valid = bool(integer_target <= EXIT_COUNT * n and analytic_supported)
        seeded_total_flops: list[int] = [-1, -1]
        seeded_contrasts: list[float] = [float("nan"), float("nan")]
        if seeded_valid:
            for endpoint, endpoint_name in enumerate(ENDPOINT_NAMES):
                lower = int(analytic[endpoint]["depth_lower"])
                upper = int(analytic[endpoint]["depth_upper"])
                lower_total = (
                    n * (BASE_FLOPS_PER_ROW + DEPTH1_FLOPS_PER_ROW)
                    + n * (lower - 1) * ADAPTER_FLOPS_PER_ADDITIONAL_CALL
                )
                increment = (upper - lower) * ADAPTER_FLOPS_PER_ADDITIONAL_CALL
                if lower == upper:
                    number_upper = 0
                    endpoint_valid = lower_total >= int(compute["total_flops"])
                else:
                    delta = int(compute["total_flops"]) - lower_total
                    number_upper = int((delta + increment - 1) // increment)
                    number_upper = min(n, max(0, number_upper))
                    endpoint_valid = True
                seeded_calls = np.full(n, lower, dtype=np.int64)
                rng = np.random.default_rng(
                    int(assigned_seeds[dgp_id][endpoint_name])
                )
                seeded_calls[rng.permutation(n)[:number_upper]] = upper
                seeded_total_flops[endpoint] = int(
                    n * (BASE_FLOPS_PER_ROW + DEPTH1_FLOPS_PER_ROW)
                    + (int(seeded_calls.sum()) - n)
                    * ADAPTER_FLOPS_PER_ADDITIONAL_CALL
                )
                endpoint_valid &= (
                    seeded_total_flops[endpoint] >= int(compute["total_flops"])
                )
                seeded_valid &= endpoint_valid
                seeded_loss = local_primary[
                    local_index, seeded_calls - 1, endpoint
                ]
                seeded_contrasts[endpoint] = float(
                    _episode_means(
                        seeded_loss - adaptive_primary[:, endpoint], local_episode
                    ).mean()
                )

        # Within-episode randomization preserves every call histogram exactly.
        randomized_calls = np.empty_like(local_calls)
        histogram_rng = np.random.default_rng(int(assigned_seeds[dgp_id]["histogram"]))
        histogram_exact = True
        for episode in np.unique(local_episode):
            indices = np.flatnonzero(local_episode == episode)
            randomized_calls[indices] = local_calls[indices][
                histogram_rng.permutation(len(indices))
            ]
            histogram_exact &= bool(
                np.array_equal(
                    np.bincount(local_calls[indices], minlength=5),
                    np.bincount(randomized_calls[indices], minlength=5),
                )
            )
        histogram_contrasts = []
        fixed_d1_contrasts = []
        for endpoint in range(2):
            randomized_loss = local_primary[local_index, randomized_calls - 1, endpoint]
            histogram_contrasts.append(
                float(
                    _episode_means(
                        randomized_loss - adaptive_primary[:, endpoint], local_episode
                    ).mean()
                )
            )
            fixed_d1_contrasts.append(
                float(
                    _episode_means(
                        local_primary[:, 0, endpoint] - adaptive_primary[:, endpoint],
                        local_episode,
                    ).mean()
                )
            )

        stage_records = []
        for stage in range(STAGE_COUNT):
            gain = local_primary[:, stage, :] - local_primary[:, stage + 1, :]
            standardized_gain = (
                gain - gain_mean[dgp, stage][None, :]
            ) / gain_std[dgp, stage][None, :]
            combined_gain = standardized_gain.mean(axis=1)
            stage_mask = local_reached[:, stage]
            rho = _spearman(local_scores[stage_mask, stage], combined_gain[stage_mask])
            rank_values.append(rho)
            stage_records.append(
                {
                    "stage": stage + 1,
                    "reached_rows": int(stage_mask.sum()),
                    "reached_fraction": float(stage_mask.mean()),
                    "spearman": _json_float(rho),
                }
            )

        analytic_exact = bool(
            analytic_supported
            and all(
                item["exact_total_compute_match"] is True
                and item["exact_integer_cross_product_identity"] is True
                and int(item["allocation_total_counted_flops"])
                == analytic_budget
                and int(item["allocation_total_counted_flops_numerator"])
                == analytic_budget
                * int(item["allocation_total_counted_flops_denominator"])
                for item in analytic
            )
        )
        finite_exact = bool(
            np.isin(local_calls, (1, 2, 3, 4)).all()
            and all(
                np.isfinite(value)
                for value in (
                    *contrast_means,
                    *rank_values[-STAGE_COUNT:],
                    float(compute["flops_per_row"]),
                )
            )
        )
        comparator_valid = bool(analytic_exact and seeded_valid and histogram_exact)
        mean_calls_valid = 1.05 <= float(local_calls.mean()) <= 2.5
        reach_valid = (
            stage_records[1]["reached_fraction"] >= 0.05
            and stage_records[2]["reached_fraction"] >= 0.01
        )
        local_valid = bool(finite_exact and comparator_valid and mean_calls_valid and reach_valid)
        all_valid &= local_valid
        result_dgps.append(
            {
                "dgp_id": dgp_id,
                "row_count": n,
                "episode_count": int(len(np.unique(local_episode))),
                "mean_calls": float(local_calls.mean()),
                "call_histogram": np.bincount(local_calls, minlength=5)[1:].tolist(),
                "stagewise_rank": stage_records,
                "compute": compute,
                "exact_compute_contrast": {
                    ENDPOINT_NAMES[endpoint]: _json_float(contrast_means[endpoint])
                    for endpoint in range(2)
                },
                "exact_compute_episode_summary": {
                    f"{ENDPOINT_NAMES[endpoint]}_vs_analytic": {
                        "episode_count": int(len(contrast_episode[endpoint])),
                        "mean": _json_float(contrast_means[endpoint]),
                        "sd_ddof1": _json_float(
                            contrast_episode[endpoint].std(ddof=1)
                            if len(contrast_episode[endpoint]) > 1
                            else float("nan")
                        ),
                    }
                    for endpoint in range(2)
                },
                "exact_compute_analytic": {
                    ENDPOINT_NAMES[endpoint]: (
                        {
                            key: value
                            for key, value in analytic[endpoint].items()
                            if key != "loss"
                        }
                        if analytic_supported
                        else None
                    )
                    for endpoint in range(2)
                },
                "auxiliary_robust_whitened_exact_compute_contrast": _json_float(
                    _episode_means(auxiliary_values, local_episode).mean()
                ),
                "seeded_weakly_more_compute_contrast": {
                    ENDPOINT_NAMES[endpoint]: _json_float(seeded_contrasts[endpoint])
                    for endpoint in range(2)
                },
                "seeded_total_flops": {
                    ENDPOINT_NAMES[endpoint]: seeded_total_flops[endpoint]
                    for endpoint in range(2)
                },
                "comparator_rng_ids": dict(assigned_seeds[dgp_id]),
                "fixed_depth1_contrast": {
                    ENDPOINT_NAMES[endpoint]: fixed_d1_contrasts[endpoint]
                    for endpoint in range(2)
                },
                "within_episode_histogram_contrast": {
                    ENDPOINT_NAMES[endpoint]: histogram_contrasts[endpoint]
                    for endpoint in range(2)
                },
                "mechanical_checks": {
                    "finite_and_exact": finite_exact,
                    "mean_calls_in_closed_1p05_2p5": mean_calls_valid,
                    "stage2_reached_at_least_5_percent": (
                        stage_records[1]["reached_fraction"] >= 0.05
                    ),
                    "stage3_reached_at_least_1_percent": (
                        stage_records[2]["reached_fraction"] >= 0.01
                    ),
                    "analytic_compute_exact": analytic_exact,
                    "seeded_comparator_weakly_more_compute": seeded_valid,
                    "within_episode_histograms_exact": histogram_exact,
                    "comparators_valid": comparator_valid,
                },
                "mechanically_valid": local_valid,
            }
        )
    fit_sd_valid = fit_contrast_sd is None or bool(
        np.isfinite(fit_contrast_sd).all() and np.all(fit_contrast_sd >= 0)
    )
    all_valid &= fit_sd_valid
    return {
        "candidate_id": candidate_id,
        "architecture": architecture,
        "head_count": int(weights.shape[1]),
        "eligible": bool(all_valid),
        "eligibility_is_mechanical_only": True,
        "fit_episode_sd_valid": fit_sd_valid,
        "runtime_call_equivalence": {
            "dense_sparse_calls_exact": True,
            "dense_sparse_reached_exact": True,
            "runtime_dtype": "float32",
            "continue_operator": "strict_greater_than",
            "threshold_ties_stop": True,
        },
        "dgps": result_dgps,
        "worst_standardized_exact_compute_contrast": (
            _json_float(min(standardized_values)) if standardized_values else None
        ),
        "worst_dgp_stage_spearman": (
            _json_float(min(rank_values)) if rank_values else None
        ),
        "worst_flops_per_row": float(
            max(record["compute"]["flops_per_row"] for record in result_dgps)
        ),
    }


def fit_candidate_family(
    fit_role: RoleData,
    fixed_whitening: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit, compile, and fit-evaluate the complete bounded 24-candidate family."""

    if fit_role.dgp_ids != DGP_IDS:
        raise ValueError(f"fit DGP order must be exactly {DGP_IDS}")
    _validate_whitening(fixed_whitening, fit_role.latent_dim)
    dgp_count = len(fit_role.dgp_ids)
    primary_loss = endpoint_losses(fit_role, fixed_whitening)
    gains = primary_loss[:, :STAGE_COUNT, :] - primary_loss[:, 1:, :]
    robust_whitening, robust_target_means, robust_floor = fit_robust_whitening(fit_role)

    pooled_mean = np.empty((STAGE_COUNT, FEATURE_DIM), dtype=np.float64)
    pooled_std = np.empty_like(pooled_mean)
    domain_mean = np.empty((dgp_count, STAGE_COUNT, FEATURE_DIM), dtype=np.float64)
    domain_std = np.empty_like(domain_mean)
    gain_mean = np.empty((dgp_count, STAGE_COUNT, 2), dtype=np.float64)
    gain_std = np.empty_like(gain_mean)
    standardized_gain = np.empty_like(gains)
    for stage in range(STAGE_COUNT):
        pooled_mean[stage], pooled_std[stage] = _balanced_mean_std(
            fit_role.features[:, stage], fit_role.dgp_index, dgp_count, floor=1e-6
        )
        for dgp in range(dgp_count):
            mask = fit_role.dgp_index == dgp
            domain_mean[dgp, stage], domain_std[dgp, stage] = _domain_mean_std(
                fit_role.features[mask, stage], floor=1e-6
            )
            gain_mean[dgp, stage], gain_std[dgp, stage] = _domain_mean_std(
                gains[mask, stage], floor=1e-12
            )
            standardized_gain[mask, stage] = (
                gains[mask, stage] - gain_mean[dgp, stage]
            ) / gain_std[dgp, stage]

    # Fit six ridge values per stage/architecture once; quantiles only alter
    # the sequential fit thresholds and never refit a head.
    pooled_compiled_w = np.empty((len(RIDGES), STAGE_COUNT, 2, FEATURE_DIM))
    pooled_compiled_b = np.empty((len(RIDGES), STAGE_COUNT, 2))
    balanced_weights = _balanced_row_weights(fit_role.dgp_index, dgp_count)
    for stage in range(STAGE_COUNT):
        normalized = (
            fit_role.features[:, stage] - pooled_mean[stage]
        ) / pooled_std[stage]
        path = _ridge_path(
            normalized, standardized_gain[:, stage], RIDGES,
            row_weights=balanced_weights,
        )
        for ridge_index in range(len(RIDGES)):
            for endpoint in range(2):
                raw_w, raw_b = compile_affine_head(
                    path[ridge_index, :, endpoint], pooled_mean[stage], pooled_std[stage]
                )
                pooled_compiled_w[ridge_index, stage, endpoint] = raw_w
                pooled_compiled_b[ridge_index, stage, endpoint] = raw_b

    envelope_compiled_w = np.empty((len(RIDGES), STAGE_COUNT, MAX_HEADS, FEATURE_DIM))
    envelope_compiled_b = np.empty((len(RIDGES), STAGE_COUNT, MAX_HEADS))
    for stage in range(STAGE_COUNT):
        for dgp in range(dgp_count):
            mask = fit_role.dgp_index == dgp
            normalized = (
                fit_role.features[mask, stage] - domain_mean[dgp, stage]
            ) / domain_std[dgp, stage]
            path = _ridge_path(normalized, standardized_gain[mask, stage], RIDGES)
            for ridge_index in range(len(RIDGES)):
                for endpoint in range(2):
                    head = 2 * dgp + endpoint
                    raw_w, raw_b = compile_affine_head(
                        path[ridge_index, :, endpoint],
                        domain_mean[dgp, stage],
                        domain_std[dgp, stage],
                    )
                    envelope_compiled_w[ridge_index, stage, head] = raw_w
                    envelope_compiled_b[ridge_index, stage, head] = raw_b

    grid = _candidate_grid()
    candidate_ids = np.asarray([item[3] for item in grid])
    architectures = np.asarray([item[0] for item in grid])
    ridge_values = np.asarray([item[1] for item in grid], dtype=np.float64)
    quantiles = np.asarray([item[2] for item in grid], dtype=np.float64)
    head_count = np.asarray(
        [ARCHITECTURE_HEADS[item[0]] for item in grid], dtype=np.int64
    )
    head_names = np.full((CANDIDATE_COUNT, MAX_HEADS), "", dtype="<U64")
    compiled_weights = np.zeros(
        (CANDIDATE_COUNT, STAGE_COUNT, MAX_HEADS, FEATURE_DIM), dtype=np.float32
    )
    compiled_biases = np.zeros(
        (CANDIDATE_COUNT, STAGE_COUNT, MAX_HEADS), dtype=np.float32
    )
    thresholds = np.empty((CANDIDATE_COUNT, STAGE_COUNT), dtype=np.float32)
    fit_contrast_episode_sd = np.empty(
        (CANDIDATE_COUNT, dgp_count, 2), dtype=np.float64
    )
    fit_contrast_episode_mean = np.empty_like(fit_contrast_episode_sd)
    fit_contrast_episode_count = np.empty(
        (CANDIDATE_COUNT, dgp_count, 2), dtype=np.int64
    )
    fit_mean_calls = np.empty((CANDIDATE_COUNT, dgp_count), dtype=np.float64)
    fit_reached_fraction = np.empty(
        (CANDIDATE_COUNT, dgp_count, STAGE_COUNT), dtype=np.float64
    )
    fit_runtime_call_equivalence = np.zeros(CANDIDATE_COUNT, dtype=bool)
    score_cache: dict[tuple[str, float], np.ndarray] = {}

    for candidate_index, (architecture, ridge, quantile, candidate_id) in enumerate(grid):
        ridge_index = RIDGES.index(ridge)
        heads = ARCHITECTURE_HEADS[architecture]
        names = _head_names(architecture, fit_role.dgp_ids)
        head_names[candidate_index, :heads] = np.asarray(names)
        if architecture == "balanced_pooled_dual":
            weights = pooled_compiled_w[ridge_index]
            biases = pooled_compiled_b[ridge_index]
        else:
            weights = envelope_compiled_w[ridge_index]
            biases = envelope_compiled_b[ridge_index]
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        biases = np.ascontiguousarray(biases, dtype=np.float32)
        compiled_weights[candidate_index, :, :heads] = weights
        compiled_biases[candidate_index, :, :heads] = biases
        cache_key = (architecture, ridge)
        if cache_key not in score_cache:
            score_cache[cache_key] = score_compiled_gate(
                fit_role.features, weights, biases
            )
        scores = score_cache[cache_key]
        candidate_thresholds, calls, reached = sequential_fit_thresholds(
            scores, fit_role.dgp_index, quantile, dgp_count=dgp_count
        )
        runtime_audit = audit_dense_sparse_call_equivalence(
            scores, candidate_thresholds
        )
        if not np.array_equal(calls, runtime_audit["calls"]) or not np.array_equal(
            reached, runtime_audit["reached"]
        ):
            raise RuntimeError("fit quantile calls differ from runtime reconstruction")
        fit_runtime_call_equivalence[candidate_index] = True
        thresholds[candidate_index] = candidate_thresholds
        # Fit-role episode SD is the sole scale used to standardize selection
        # exact-compute mean contrasts.  It is frozen before selection opens.
        for dgp in range(dgp_count):
            mask = fit_role.dgp_index == dgp
            local_calls = calls[mask]
            local_loss = primary_loss[mask]
            local_episode = fit_role.episode_slot[mask]
            local_index = np.arange(len(local_calls))
            compute = _compute_accounting(local_calls, architecture)
            for endpoint in range(2):
                analytic = _strongest_analytic_mixture(
                    local_loss[:, :, endpoint], int(compute["total_flops"])
                )
                adaptive = local_loss[local_index, local_calls - 1, endpoint]
                episode_contrast = _episode_means(
                    analytic["loss"] - adaptive, local_episode
                )
                fit_contrast_episode_mean[candidate_index, dgp, endpoint] = float(
                    episode_contrast.mean()
                )
                fit_contrast_episode_sd[candidate_index, dgp, endpoint] = (
                    float(episode_contrast.std(ddof=1))
                    if len(episode_contrast) > 1
                    else float("nan")
                )
                fit_contrast_episode_count[candidate_index, dgp, endpoint] = len(
                    episode_contrast
                )
            fit_mean_calls[candidate_index, dgp] = float(local_calls.mean())
            fit_reached_fraction[candidate_index, dgp] = reached[mask].mean(axis=0)

    candidate_object_hashes = np.empty(CANDIDATE_COUNT, dtype="<U64")
    for candidate_index in range(CANDIDATE_COUNT):
        heads = int(head_count[candidate_index])
        candidate_object_hashes[candidate_index] = candidate_object_sha256(
            candidate_id=str(candidate_ids[candidate_index]),
            architecture=str(architectures[candidate_index]),
            ridge=float(ridge_values[candidate_index]),
            quantile=float(quantiles[candidate_index]),
            weights=compiled_weights[candidate_index, :, :heads],
            biases=compiled_biases[candidate_index, :, :heads],
            thresholds=thresholds[candidate_index],
            head_names=head_names[candidate_index, :heads],
        )

    finite_arrays = (
        pooled_mean, pooled_std, domain_mean, domain_std, gain_mean, gain_std,
        robust_whitening, compiled_weights, compiled_biases, thresholds,
        fit_contrast_episode_mean, fit_mean_calls, fit_reached_fraction,
    )
    if not all(np.isfinite(value).all() for value in finite_arrays):
        raise RuntimeError("nonfinite fitted candidate artifact")
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "dgp_ids": np.asarray(fit_role.dgp_ids),
        "endpoint_names": np.asarray(ENDPOINT_NAMES),
        "candidate_ids": candidate_ids,
        "candidate_object_sha256": candidate_object_hashes,
        "architectures": architectures,
        "ridges": ridge_values,
        "fit_quantiles": quantiles,
        "head_count": head_count,
        "head_names": head_names,
        "compiled_weights": compiled_weights,
        "compiled_biases": compiled_biases,
        "thresholds": thresholds,
        "fit_contrast_episode_sd": fit_contrast_episode_sd,
        "fit_contrast_episode_mean": fit_contrast_episode_mean,
        "fit_contrast_episode_count": fit_contrast_episode_count,
        "fit_mean_calls": fit_mean_calls,
        "fit_reached_fraction": fit_reached_fraction,
        "fit_runtime_call_equivalence": fit_runtime_call_equivalence,
        "pooled_feature_mean": pooled_mean,
        "pooled_feature_std": pooled_std,
        "per_dgp_feature_mean": domain_mean,
        "per_dgp_feature_std": domain_std,
        "per_dgp_stage_endpoint_gain_mean": gain_mean,
        "per_dgp_stage_endpoint_gain_std": gain_std,
        "fixed_whitening_matrix": np.asarray(fixed_whitening, dtype=np.float64),
        "robust_fit_whitening_matrix": robust_whitening,
        "robust_fit_target_mean": robust_target_means,
        "robust_fit_eigenvalue_floor": np.asarray(robust_floor, dtype=np.float64),
        "feature_dim": np.asarray(FEATURE_DIM, dtype=np.int64),
        "stage_count": np.asarray(STAGE_COUNT, dtype=np.int64),
        "fit_row_count": np.asarray(fit_role.row_count, dtype=np.int64),
    }


def validate_fitted_artifact(fitted: Mapping[str, np.ndarray]) -> None:
    required = {
        "dgp_ids", "candidate_ids", "candidate_object_sha256", "architectures", "ridges", "fit_quantiles",
        "head_count", "head_names", "compiled_weights", "compiled_biases",
        "thresholds", "fit_contrast_episode_sd", "per_dgp_stage_endpoint_gain_mean",
        "fit_contrast_episode_mean", "fit_contrast_episode_count",
        "fit_runtime_call_equivalence",
        "per_dgp_stage_endpoint_gain_std", "fixed_whitening_matrix",
        "robust_fit_whitening_matrix",
    }
    missing = required.difference(fitted)
    if missing:
        raise RuntimeError(f"fitted artifact missing {sorted(missing)}")
    if tuple(fitted["dgp_ids"].astype(str).tolist()) != DGP_IDS:
        raise RuntimeError("fitted DGP order drift")
    if len(fitted["candidate_ids"]) != CANDIDATE_COUNT:
        raise RuntimeError("fitted candidate count is not 24")
    if fitted["candidate_object_sha256"].shape != (CANDIDATE_COUNT,):
        raise RuntimeError("candidate object hash shape drift")
    if fitted["compiled_weights"].shape != (
        CANDIDATE_COUNT, STAGE_COUNT, MAX_HEADS, FEATURE_DIM
    ):
        raise RuntimeError("fitted compiled weight shape drift")
    if fitted["compiled_biases"].shape != (CANDIDATE_COUNT, STAGE_COUNT, MAX_HEADS):
        raise RuntimeError("fitted compiled bias shape drift")
    if fitted["thresholds"].shape != (CANDIDATE_COUNT, STAGE_COUNT):
        raise RuntimeError("fitted threshold shape drift")
    if (
        fitted["compiled_weights"].dtype != np.dtype(np.float32)
        or fitted["compiled_biases"].dtype != np.dtype(np.float32)
        or fitted["thresholds"].dtype != np.dtype(np.float32)
    ):
        raise RuntimeError("fitted runtime gate objects must be exactly float32")
    if not all(
        np.asarray(fitted[name]).flags.c_contiguous
        for name in ("compiled_weights", "compiled_biases", "thresholds")
    ):
        raise RuntimeError("fitted runtime gate objects must be contiguous")
    if (
        fitted["fit_runtime_call_equivalence"].shape != (CANDIDATE_COUNT,)
        or not fitted["fit_runtime_call_equivalence"].astype(bool).all()
    ):
        raise RuntimeError("fit runtime call-equivalence contract failed")
    for name in (
        "fit_contrast_episode_mean", "fit_contrast_episode_sd",
        "fit_contrast_episode_count",
    ):
        if fitted[name].shape != (CANDIDATE_COUNT, 4, 2):
            raise RuntimeError(f"{name} shape drift")
    if not np.isin(fitted["head_count"], (2, 8)).all():
        raise RuntimeError("invalid fitted head count")
    if np.any(fitted["fit_contrast_episode_count"] < 2):
        raise RuntimeError("fit contrast summaries require at least two episodes")
    if (
        not np.isfinite(fitted["fit_contrast_episode_sd"]).all()
        or np.any(fitted["fit_contrast_episode_sd"] < 0)
    ):
        raise RuntimeError("fit contrast episode SD is invalid")
    always_finite = (
        fitted["compiled_weights"], fitted["compiled_biases"], fitted["thresholds"],
        fitted["fit_contrast_episode_mean"],
        fitted["per_dgp_stage_endpoint_gain_mean"],
        fitted["per_dgp_stage_endpoint_gain_std"],
        fitted["fixed_whitening_matrix"], fitted["robust_fit_whitening_matrix"],
    )
    if not all(np.isfinite(value).all() for value in always_finite):
        raise RuntimeError("fitted artifact contains nonfinite scientific objects")
    for candidate_index in range(CANDIDATE_COUNT):
        heads = int(fitted["head_count"][candidate_index])
        observed = candidate_object_sha256(
            candidate_id=str(fitted["candidate_ids"][candidate_index]),
            architecture=str(fitted["architectures"][candidate_index]),
            ridge=float(fitted["ridges"][candidate_index]),
            quantile=float(fitted["fit_quantiles"][candidate_index]),
            weights=fitted["compiled_weights"][candidate_index, :, :heads],
            biases=fitted["compiled_biases"][candidate_index, :, :heads],
            thresholds=fitted["thresholds"][candidate_index],
            head_names=fitted["head_names"][candidate_index, :heads],
        )
        if str(fitted["candidate_object_sha256"][candidate_index]) != observed:
            raise RuntimeError("candidate object hash drift")


def save_fitted_candidates(path: Path | str, fitted: Mapping[str, np.ndarray]) -> None:
    validate_fitted_artifact(fitted)
    _atomic_npz_exclusive(path, fitted)


def load_fitted_candidates(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as stored:
        fitted = {name: stored[name].copy() for name in stored.files}
    validate_fitted_artifact(fitted)
    return fitted


def seal_fit_lock(
    fitted_path: Path | str,
    lock_path: Path | str,
    *,
    fit_input_hashes: Mapping[str, str],
    fixed_whitening_sha256: str,
    scientific_source_hashes: Mapping[str, str] | None = None,
    selection_input_opened: bool = False,
) -> dict[str, Any]:
    """Seal the complete family; fail closed if selection was already opened."""

    if selection_input_opened:
        raise RuntimeError("cannot seal fit after any selection input was opened")
    fitted = load_fitted_candidates(fitted_path)
    payload = {
        "schema_version": 1,
        "status": "all_24_candidates_fit_compiled_and_locked_before_selection_open",
        "created_unix_ns": time.time_ns(),
        "candidate_count": CANDIDATE_COUNT,
        "architectures": list(ARCHITECTURES),
        "ridges": list(RIDGES),
        "sequential_fit_quantiles": list(FIT_QUANTILES),
        "dgp_ids": list(DGP_IDS),
        "feature_dim": FEATURE_DIM,
        "fit_input_hashes": dict(sorted(fit_input_hashes.items())),
        "scientific_source_hashes": dict(
            sorted((scientific_source_hashes or {}).items())
        ),
        "fixed_whitening_sha256": str(fixed_whitening_sha256),
        "fitted_candidates_sha256": sha256_file(fitted_path),
        "all_candidates_have_raw_feature_affine_heads": True,
        "all_candidate_runtime_objects_contiguous_float32": True,
        "all_24_fit_call_traces_reconstruct_exactly": bool(
            fitted["fit_runtime_call_equivalence"].all()
        ),
        "runtime_continue_operator": "strict_greater_than",
        "runtime_threshold_ties_stop": True,
        "fixed_whitened_endpoint_uses_unchanged_v5_matrix": True,
        "robust_fit_whitening_is_auxiliary_only": True,
        "selection_input_opened_before_lock": False,
        "selection_outcomes_used_for_fit": False,
        "selected_head_refit_permitted": False,
        "fit_row_count": int(np.asarray(fitted["fit_row_count"]).item()),
        "prior_confirmation_outcome_episodes_used": 0,
    }
    _atomic_json_exclusive(lock_path, payload)
    return payload


def fit_and_seal(
    fit_per_dgp: Mapping[str, Mapping[str, np.ndarray]],
    fixed_whitening: np.ndarray,
    *,
    fitted_path: Path | str,
    lock_path: Path | str,
    fit_input_hashes: Mapping[str, str],
    fixed_whitening_sha256: str,
) -> dict[str, Any]:
    """Convenience entry point that has no selection argument by construction."""

    role = prepare_role(fit_per_dgp)
    validate_role_cardinality(role, episodes_per_dgp=300)
    if fixed_whitening_sha256 != V5_FIXED_WHITENING_SHA256:
        raise RuntimeError("fit did not receive the unchanged sealed V5 whitening file")
    attempt_root = Path(__file__).resolve().parent
    scientific_sources = {
        name: sha256_file(attempt_root / name)
        for name in (
            "PREREGISTRATION.md",
            "candidate_grid.json",
            "fit_select.py",
            "compile_gate.py",
            "counted_features.py",
            "flops.py",
        )
    }
    fitted = fit_candidate_family(role, fixed_whitening)
    save_fitted_candidates(fitted_path, fitted)
    return seal_fit_lock(
        fitted_path,
        lock_path,
        fit_input_hashes=fit_input_hashes,
        fixed_whitening_sha256=fixed_whitening_sha256,
        scientific_source_hashes=scientific_sources,
        selection_input_opened=False,
    )


def verify_fit_lock(fitted_path: Path | str, lock_path: Path | str) -> dict[str, Any]:
    lock = _read_json(lock_path)
    if lock.get("status") != "all_24_candidates_fit_compiled_and_locked_before_selection_open":
        raise RuntimeError("fit lock status is not selection-ready")
    if lock.get("candidate_count") != CANDIDATE_COUNT:
        raise RuntimeError("fit lock candidate count drift")
    if lock.get("selection_input_opened_before_lock") is not False:
        raise RuntimeError("fit/selection chronology invalid")
    observed = sha256_file(fitted_path)
    if lock.get("fitted_candidates_sha256") != observed:
        raise RuntimeError("fitted candidate hash drift after fit lock")
    return lock


def rank_eligible_candidates(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Apply the immutable lexicographic ranking to mechanically eligible rows."""

    eligible = [row for row in rows if bool(row.get("eligible"))]
    for row in eligible:
        required = (
            "worst_standardized_exact_compute_contrast",
            "worst_dgp_stage_spearman",
            "worst_flops_per_row",
            "head_count",
            "ridge",
            "fit_quantile",
            "candidate_id",
        )
        if any(key not in row for key in required):
            raise ValueError("eligible candidate lacks a ranking field")
        numeric = [
            row["worst_standardized_exact_compute_contrast"],
            row["worst_dgp_stage_spearman"], row["worst_flops_per_row"],
            row["head_count"], row["ridge"], row["fit_quantile"],
        ]
        if not all(np.isfinite(float(value)) for value in numeric):
            raise ValueError("eligible candidate has nonfinite ranking values")
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["worst_standardized_exact_compute_contrast"]),
            -float(row["worst_dgp_stage_spearman"]),
            float(row["worst_flops_per_row"]),
            int(row["head_count"]),
            float(row["ridge"]),
            float(row["fit_quantile"]),
            str(row["candidate_id"]),
        ),
    )


def evaluate_selection(
    fitted: Mapping[str, np.ndarray],
    selection_role: RoleData,
    *,
    comparator_seeds: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Evaluate all 24 candidates once and apply the fixed robust ranking."""

    validate_fitted_artifact(fitted)
    assigned_seeds = _validate_comparator_seeds(comparator_seeds)
    if selection_role.dgp_ids != DGP_IDS:
        raise ValueError(f"selection DGP order must be exactly {DGP_IDS}")
    fixed_whitening = np.asarray(fitted["fixed_whitening_matrix"], dtype=np.float64)
    robust_whitening = np.asarray(
        fitted["robust_fit_whitening_matrix"], dtype=np.float64
    )
    gain_mean = np.asarray(
        fitted["per_dgp_stage_endpoint_gain_mean"], dtype=np.float64
    )
    gain_std = np.asarray(
        fitted["per_dgp_stage_endpoint_gain_std"], dtype=np.float64
    )
    primary_losses = endpoint_losses(selection_role, fixed_whitening)
    auxiliary_losses = endpoint_losses(selection_role, robust_whitening)[:, :, 1]
    score_cache: dict[tuple[str, float], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for candidate_index in range(CANDIDATE_COUNT):
        heads = int(fitted["head_count"][candidate_index])
        architecture = str(fitted["architectures"][candidate_index])
        candidate_id = str(fitted["candidate_ids"][candidate_index])
        ridge = float(fitted["ridges"][candidate_index])
        cache_key = (architecture, ridge)
        if cache_key not in score_cache:
            score_cache[cache_key] = score_compiled_gate(
                selection_role.features,
                fitted["compiled_weights"][candidate_index, :, :heads],
                fitted["compiled_biases"][candidate_index, :, :heads],
            )
        row = _evaluate_candidate(
            selection_role,
            architecture=architecture,
            candidate_id=candidate_id,
            weights=fitted["compiled_weights"][candidate_index, :, :heads],
            biases=fitted["compiled_biases"][candidate_index, :, :heads],
            thresholds=fitted["thresholds"][candidate_index],
            fixed_whitening=fixed_whitening,
            robust_whitening=robust_whitening,
            gain_mean=gain_mean,
            gain_std=gain_std,
            fit_contrast_sd=fitted["fit_contrast_episode_sd"][candidate_index],
            comparator_seeds=assigned_seeds,
            precomputed_scores=score_cache[cache_key],
            precomputed_primary_losses=primary_losses,
            precomputed_auxiliary_losses=auxiliary_losses,
        )
        row.update(
            {
                "candidate_index": candidate_index,
                "candidate_object_sha256": str(
                    fitted["candidate_object_sha256"][candidate_index]
                ),
                "ridge": ridge,
                "fit_quantile": float(fitted["fit_quantiles"][candidate_index]),
                "thresholds": fitted["thresholds"][candidate_index].tolist(),
            }
        )
        rows.append(row)
    ranked = rank_eligible_candidates(rows)
    selected = ranked[0] if ranked else None
    return {
        "schema_version": 1,
        "status": "selection_complete_no_selected_head_refit",
        "created_unix_ns": time.time_ns(),
        "candidate_count": CANDIDATE_COUNT,
        "eligible_count": len(ranked),
        "eligibility": (
            "mechanical finite/exact compute and comparator validity; every DGP mean calls "
            "in [1.05,2.5], stage-2 reached >=5%, and stage-3 reached >=1%"
        ),
        "fixed_ranking": [
            "maximize worst of eight DGP-by-co-primary exact-compute mean contrasts standardized by corresponding fit episode SD",
            "maximize worst of twelve DGP-by-stage Spearman coefficients",
            "minimize worst DGP FLOPs per row",
            "fewer heads",
            "lower ridge",
            "lower sequential fit quantile",
            "lexical candidate id",
        ],
        "candidates": rows,
        "ranked_eligible_candidate_ids": [str(row["candidate_id"]) for row in ranked],
        "selected_candidate_id": None if selected is None else str(selected["candidate_id"]),
        "selected_candidate_index": None if selected is None else int(selected["candidate_index"]),
        "selected_head_refit_after_selection": False,
        "all_24_selection_call_traces_reconstruct_exactly": bool(
            len(rows) == CANDIDATE_COUNT
            and all(
                row["runtime_call_equivalence"]["dense_sparse_calls_exact"] is True
                and row["runtime_call_equivalence"]["dense_sparse_reached_exact"] is True
                for row in rows
            )
        ),
        "selection_runtime_dtype": "float32",
        "selection_runtime_continue_operator": "strict_greater_than",
        "selection_runtime_threshold_ties_stop": True,
        "robust_fit_whitening_used_for_selection": False,
        "comparator_rng_ids": assigned_seeds,
        "comparator_rng_derivation": (
            "use each prospectively assigned cohort-ledger rng_id directly; "
            "no candidate-specific or hash-derived RNG identifiers"
        ),
        "prior_confirmation_outcome_episodes_used": 0,
    }


def selected_gate_arrays(
    fitted: Mapping[str, np.ndarray], selection_ledger: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Copy the already-fitted selected heads without any refit or recalibration."""

    selected_index = selection_ledger.get("selected_candidate_index")
    if selected_index is None:
        raise RuntimeError("no mechanically eligible gate was selected")
    index = int(selected_index)
    if not 0 <= index < CANDIDATE_COUNT:
        raise RuntimeError("selected candidate index out of range")
    expected_id = str(fitted["candidate_ids"][index])
    if selection_ledger.get("selected_candidate_id") != expected_id:
        raise RuntimeError("selected candidate identity mismatch")
    heads = int(fitted["head_count"][index])
    weights = np.ascontiguousarray(
        fitted["compiled_weights"][index, :, :heads]
    )
    biases = np.ascontiguousarray(fitted["compiled_biases"][index, :, :heads])
    thresholds = np.ascontiguousarray(fitted["thresholds"][index])
    if any(
        value.dtype != np.dtype(np.float32)
        for value in (weights, biases, thresholds)
    ):
        raise RuntimeError("selected gate is not the frozen float32 runtime object")
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "candidate_index": np.asarray(index, dtype=np.int64),
        "candidate_id": np.asarray(expected_id),
        "source_candidate_object_sha256": np.asarray(
            str(fitted["candidate_object_sha256"][index])
        ),
        "architecture": np.asarray(str(fitted["architectures"][index])),
        "ridge": np.asarray(float(fitted["ridges"][index]), dtype=np.float64),
        "fit_quantile": np.asarray(
            float(fitted["fit_quantiles"][index]), dtype=np.float64
        ),
        "weights": weights,
        "biases": biases,
        "thresholds": thresholds,
        "head_names": fitted["head_names"][index, :heads].copy(),
        "head_count": np.asarray(heads, dtype=np.int64),
        "feature_dim": np.asarray(FEATURE_DIM, dtype=np.int64),
        "runtime_dtype": np.asarray("float32"),
        "continue_operator": np.asarray("strict_greater_than"),
        "threshold_ties_stop": np.asarray(True),
        "gate_flops_per_evaluation": np.asarray(
            ARCHITECTURE_GATE_FLOPS[str(fitted["architectures"][index])],
            dtype=np.int64,
        ),
        "gate_nonflops_per_evaluation": np.asarray(
            ARCHITECTURE_GATE_NONFLOPS[str(fitted["architectures"][index])],
            dtype=np.int64,
        ),
    }


def selected_power_summaries(
    fitted: Mapping[str, np.ndarray], selection_ledger: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact two summary documents consumed by the sealed power rule."""

    selected_index = selection_ledger.get("selected_candidate_index")
    if selected_index is None:
        raise RuntimeError("power summaries require a mechanically eligible gate")
    index = int(selected_index)
    selected_id = str(fitted["candidate_ids"][index])
    if selection_ledger.get("selected_candidate_id") != selected_id:
        raise RuntimeError("power-summary selected identity mismatch")
    selected_row = next(
        (
            row for row in selection_ledger["candidates"]
            if row.get("candidate_id") == selected_id
        ),
        None,
    )
    if not isinstance(selected_row, Mapping) or not selected_row.get("eligible"):
        raise RuntimeError("power-summary candidate is not mechanically eligible")
    selected_dgps = {
        str(record["dgp_id"]): record for record in selected_row["dgps"]
    }
    endpoint_keys = ("raw_vs_analytic", "fixed_whitened_vs_analytic")
    fit_claims: dict[str, dict[str, dict[str, float | int]]] = {}
    selection_claims: dict[str, dict[str, dict[str, float | int]]] = {}
    for dgp_index, dgp_id in enumerate(DGP_IDS):
        fit_claims[dgp_id] = {}
        selection_claims[dgp_id] = {}
        for endpoint, endpoint_key in enumerate(endpoint_keys):
            fit_claims[dgp_id][endpoint_key] = {
                "episode_count": int(
                    fitted["fit_contrast_episode_count"][index, dgp_index, endpoint]
                ),
                "mean": float(
                    fitted["fit_contrast_episode_mean"][index, dgp_index, endpoint]
                ),
                "sd_ddof1": float(
                    fitted["fit_contrast_episode_sd"][index, dgp_index, endpoint]
                ),
            }
            summary = selected_dgps[dgp_id]["exact_compute_episode_summary"][
                endpoint_key
            ]
            if any(summary[key] is None for key in ("mean", "sd_ddof1")):
                raise RuntimeError("selected power summary is nonfinite")
            selection_claims[dgp_id][endpoint_key] = {
                "episode_count": int(summary["episode_count"]),
                "mean": float(summary["mean"]),
                "sd_ddof1": float(summary["sd_ddof1"]),
            }
    common = {
        "schema_version": 1,
        "selected_candidate_id": selected_id,
        "selected_candidate_object_sha256": str(
            fitted["candidate_object_sha256"][index]
        ),
        "co_primary_only": True,
        "auxiliary_robust_whitening_excluded": True,
    }
    return (
        common | {"role": "fit", "claims": fit_claims},
        common | {"role": "selection", "claims": selection_claims},
    )


def select_after_fit_lock(
    *,
    fitted_path: Path | str,
    fit_lock_path: Path | str,
    selection_loader: Callable[[], Mapping[str, Mapping[str, np.ndarray]]],
    comparator_seeds: Mapping[str, Mapping[str, int]],
    selection_input_hashes: Mapping[str, str] | None = None,
    selection_ledger_path: Path | str,
    gate_fit_path: Path | str,
    fit_power_summary_path: Path | str | None = None,
    selection_power_summary_path: Path | str | None = None,
) -> dict[str, Any]:
    """Open selection only after verifying the immutable fit lock and its hash."""

    if (fit_power_summary_path is None) != (selection_power_summary_path is None):
        raise ValueError("both power summary paths must be supplied together")
    lock = verify_fit_lock(fitted_path, fit_lock_path)
    fitted = load_fitted_candidates(fitted_path)
    # The callback invocation is intentionally below both lock/hash checks.
    selection_role = prepare_role(selection_loader())
    validate_role_cardinality(selection_role, episodes_per_dgp=500)
    ledger = evaluate_selection(
        fitted, selection_role, comparator_seeds=comparator_seeds
    )
    ledger["fit_lock_sha256"] = sha256_file(fit_lock_path)
    ledger["fitted_candidates_sha256"] = lock["fitted_candidates_sha256"]
    ledger["selection_input_hashes"] = dict(
        sorted((selection_input_hashes or {}).items())
    )
    _atomic_json_exclusive(selection_ledger_path, ledger)
    if ledger["selected_candidate_index"] is not None:
        gate = selected_gate_arrays(fitted, ledger)
        _atomic_npz_exclusive(gate_fit_path, gate)
        if fit_power_summary_path is not None:
            fit_summary, selection_summary = selected_power_summaries(fitted, ledger)
            _atomic_json_exclusive(fit_power_summary_path, fit_summary)
            _atomic_json_exclusive(selection_power_summary_path, selection_summary)
    return ledger


def seal_gate_freeze(
    *,
    fitted_path: Path | str,
    fit_lock_path: Path | str,
    selection_ledger_path: Path | str,
    gate_fit_path: Path | str,
    compiled_gate_path: Path | str,
    compiled_gate_manifest_path: Path | str,
    gate_freeze_path: Path | str,
) -> dict[str, Any]:
    """Seal the selected raw-affine gate after the separate compiler succeeds."""

    fit_lock = verify_fit_lock(fitted_path, fit_lock_path)
    ledger = _read_json(selection_ledger_path)
    if ledger.get("selected_candidate_id") is None:
        raise RuntimeError("cannot freeze because selection found no eligible gate")
    if ledger.get("selected_head_refit_after_selection") is not False:
        raise RuntimeError("post-selection head refit detected")
    expected_fit_lock_hash = sha256_file(fit_lock_path)
    if ledger.get("fit_lock_sha256") != expected_fit_lock_hash:
        raise RuntimeError("selection ledger is not linked to the verified fit lock")
    if (
        ledger.get("fitted_candidates_sha256")
        != fit_lock.get("fitted_candidates_sha256")
    ):
        raise RuntimeError(
            "selection ledger is not linked to the verified fitted candidates"
        )
    if ledger.get("all_24_selection_call_traces_reconstruct_exactly") is not True:
        raise RuntimeError("selection runtime call-equivalence contract is incomplete")
    with np.load(gate_fit_path, allow_pickle=False) as stored:
        candidate_id = str(np.asarray(stored["candidate_id"]).item())
        architecture = str(np.asarray(stored["architecture"]).item())
        head_count = int(np.asarray(stored["head_count"]).item())
        source_object_hash = str(
            np.asarray(stored["source_candidate_object_sha256"]).item()
        )
        source_runtime = {
            name: np.ascontiguousarray(stored[name])
            for name in ("weights", "biases", "thresholds", "head_names")
        }
    if candidate_id != ledger["selected_candidate_id"]:
        raise RuntimeError("gate fit does not match the selection ledger")
    selected_row = next(
        (
            row for row in ledger["candidates"]
            if row.get("candidate_id") == candidate_id
        ),
        None,
    )
    if (
        not isinstance(selected_row, Mapping)
        or selected_row.get("candidate_object_sha256") != source_object_hash
    ):
        raise RuntimeError("selected gate object hash does not match the ledger")
    compiled_hash = sha256_file(compiled_gate_path)
    gate_fit_hash = sha256_file(gate_fit_path)
    compiler_manifest = _read_json(compiled_gate_manifest_path)
    metadata = compiler_manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("compiler manifest lacks semantic metadata")
    cast_delta = metadata.get("float32_cast_max_abs")
    if not isinstance(cast_delta, Mapping) or any(
        float(cast_delta.get(name, float("nan"))) != 0.0
        for name in ("weights", "biases", "thresholds")
    ):
        raise RuntimeError("compiler changed the already-frozen float32 gate")
    semantic_checks = {
        "compiler_passed": compiler_manifest.get("passed") is True,
        "source_gate_fit_hash_exact": (
            compiler_manifest.get("source_gate_fit_sha256") == gate_fit_hash
        ),
        "compiled_gate_hash_exact": (
            compiler_manifest.get("compiled_gate_sha256") == compiled_hash
        ),
        "candidate_id_exact": metadata.get("candidate_id") == candidate_id,
        "architecture_exact": metadata.get("architecture") == architecture,
        "head_count_exact": int(metadata.get("head_count", -1)) == head_count,
        "runtime_dtype_float32": metadata.get("runtime_dtype") == "float32",
        "compiler_did_not_fit_or_recenter": (
            metadata.get("compiler_performed_fitting_or_recentering") is False
        ),
        "compiler_opened_no_target_or_contact_arrays": (
            compiler_manifest.get("target_or_contact_arrays_opened") is False
        ),
        "compiler_opened_no_prior_outcome_arrays": (
            compiler_manifest.get("prior_outcome_arrays_opened") is False
        ),
        "inference_score_exact": (
            metadata.get("inference_score")
            == "minimum_over_all_stage_specific_affine_heads"
        ),
        "continue_rule_exact": (
            metadata.get("continue_rule")
            == "score_strictly_greater_than_stage_threshold"
        ),
        "compute_derivation_passed": bool(
            isinstance(compiler_manifest.get("complete_compute_derivation"), Mapping)
            and compiler_manifest["complete_compute_derivation"].get("passed") is True
        ),
        "gate_cost_exact": (
            isinstance(metadata.get("gate_cost"), Mapping)
            and int(metadata["gate_cost"].get("total_gate_flops", -1))
            == ARCHITECTURE_GATE_FLOPS[architecture]
            and int(metadata["gate_cost"].get("nonflop_operations", -1))
            == ARCHITECTURE_GATE_NONFLOPS[architecture]
        ),
    }
    if not all(semantic_checks.values()):
        raise RuntimeError(
            "compiled gate manifest semantic cross-link failed: "
            + ", ".join(name for name, passed in semantic_checks.items() if not passed)
        )
    with np.load(compiled_gate_path, allow_pickle=False) as compiled:
        for name in ("weights", "biases", "thresholds"):
            value = np.ascontiguousarray(compiled[name])
            if value.dtype != np.dtype(np.float32) or not np.array_equal(
                value, source_runtime[name]
            ):
                raise RuntimeError(f"compiled {name} differs from selected float32 object")
        if not np.array_equal(
            np.asarray(compiled["head_names"]), source_runtime["head_names"]
        ):
            raise RuntimeError("compiled head order differs from selected object")
        if str(np.asarray(compiled["candidate_id"]).item()) != candidate_id:
            raise RuntimeError("compiled candidate identity drift")
        if str(np.asarray(compiled["architecture"]).item()) != architecture:
            raise RuntimeError("compiled architecture identity drift")
    payload = {
        "schema_version": 1,
        "status": "selected_gate_frozen_before_smoke_or_confirmation",
        "created_unix_ns": time.time_ns(),
        "selected_candidate_id": candidate_id,
        "architecture": architecture,
        "head_count": head_count,
        "selected_candidate_object_sha256": source_object_hash,
        "fitted_candidates_sha256": sha256_file(fitted_path),
        "fit_lock_sha256": expected_fit_lock_hash,
        "selection_ledger_sha256": sha256_file(selection_ledger_path),
        "gate_fit_sha256": gate_fit_hash,
        "compiled_gate_sha256": compiled_hash,
        "compiled_gate_manifest_sha256": sha256_file(
            compiled_gate_manifest_path
        ),
        "compiled_gate_manifest_semantic_checks": semantic_checks,
        "compiler_float32_cast_max_abs": dict(cast_delta),
        "compiled_runtime_arrays_bitwise_equal_selected_gate": True,
        "selected_head_refit_after_selection": False,
        "heads_are_raw_causal_feature_affines": True,
        "contact_or_privileged_gate_inputs": False,
        "smoke_episodes_at_freeze": 0,
        "confirmation_episodes_at_freeze": 0,
        "prior_confirmation_outcome_episodes_used": 0,
    }
    _atomic_json_exclusive(gate_freeze_path, payload)
    return payload


__all__ = [
    "ADAPTER_FLOPS_PER_ADDITIONAL_CALL",
    "ARCHITECTURES",
    "ARCHITECTURE_GATE_FLOPS",
    "ARCHITECTURE_GATE_NONFLOPS",
    "CANDIDATE_COUNT",
    "DGP_IDS",
    "FEATURE_DIM",
    "FIT_QUANTILES",
    "RIDGES",
    "RoleData",
    "audit_dense_sparse_call_equivalence",
    "audit_runtime_call_equivalence",
    "calls_from_sparse_runtime_scores",
    "calls_from_scores",
    "candidate_object_sha256",
    "comparator_seeds_from_ledger",
    "compile_affine_head",
    "endpoint_losses",
    "evaluate_selection",
    "fit_and_seal",
    "fit_candidate_family",
    "fit_robust_whitening",
    "load_fixed_whitening",
    "load_fitted_candidates",
    "load_per_dgp_npz",
    "prepare_role",
    "rank_eligible_candidates",
    "save_fitted_candidates",
    "score_compiled_gate",
    "seal_fit_lock",
    "seal_gate_freeze",
    "select_after_fit_lock",
    "selected_gate_arrays",
    "selected_power_summaries",
    "sequential_fit_thresholds",
    "sha256_file",
    "sparse_runtime_scores",
    "validate_fitted_artifact",
    "validate_role_cardinality",
    "verify_fit_lock",
]
