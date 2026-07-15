"""Preregistered lightweight sequential gate for LeWM adaptive compute V2.

This module is deliberately independent of episode extraction and physical
labels.  It accepts only already-built causal feature matrices and the two
train-only marginal-benefit labels ``(L1-L2, L1-L4)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


OUTPUT_NAMES = ("b12", "b14")
HIDDEN_DIMS = (128, 64)
OUTPUT_DIM = 2
DROPOUT = 0.05
TRANSFORM_FLOOR = 1e-6
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class GateTrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 150
    gradient_clip: float = 1.0
    smooth_l1_beta: float = 0.5
    ranking_weight: float = 0.25
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-5
    calibration_pair_seed: int = 260813

    def validate(self) -> "GateTrainingConfig":
        positive = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "gradient_clip": self.gradient_clip,
            "smooth_l1_beta": self.smooth_l1_beta,
            "early_stopping_patience": self.early_stopping_patience,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative = {
            "weight_decay": self.weight_decay,
            "ranking_weight": self.ranking_weight,
            "early_stopping_min_delta": self.early_stopping_min_delta,
        }
        for name, value in nonnegative.items():
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        for name in ("batch_size", "max_epochs", "early_stopping_patience"):
            value = getattr(self, name)
            if int(value) != value:
                raise ValueError(f"{name} must be an integer")
        return self


@dataclass(frozen=True)
class InputStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    floor: float = TRANSFORM_FLOOR

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = _finite_matrix(features, "features", columns=len(self.mean))
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class RobustTargetTransform:
    center: np.ndarray
    scale: np.ndarray
    floor: float = TRANSFORM_FLOOR

    def transform(self, targets: np.ndarray) -> np.ndarray:
        values = _finite_matrix(targets, "targets", columns=OUTPUT_DIM)
        return (values - self.center) / self.scale

    def inverse(self, standardized: np.ndarray) -> np.ndarray:
        values = _finite_matrix(
            standardized, "standardized predictions", columns=OUTPUT_DIM
        )
        return values * self.scale + self.center


class GateMLP(nn.Module):
    """The sole frozen V2 architecture: ``D -> 128 -> 64 -> 2``."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIMS[0]),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.GELU(),
            nn.Linear(HIDDEN_DIMS[1], OUTPUT_DIM),
        )

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor) or not torch.is_floating_point(features):
            raise TypeError("features must be a floating torch.Tensor")
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch, {self.input_dim}], got {tuple(features.shape)}"
            )
        return self.network(features)


@dataclass
class GateSeedResult:
    seed: int
    model: GateMLP
    input_transform: InputStandardizer
    target_transform: RobustTargetTransform
    best_epoch: int
    best_calibration_objective: float
    epochs_completed: int
    history: list[dict[str, float | int | bool]]
    trainable_parameters: int


@dataclass(frozen=True)
class LoadedGateCheckpoint:
    model: GateMLP
    input_transform: InputStandardizer
    target_transform: RobustTargetTransform
    seed: int
    best_epoch: int
    best_calibration_objective: float
    feature_names: tuple[str, ...]
    state_tensor_hash: str
    file_sha256: str


def _finite_matrix(
    values: np.ndarray, name: str, *, columns: int | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns, got {array.shape[1]}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def fit_input_standardizer(
    train_features: np.ndarray, floor: float = TRANSFORM_FLOOR
) -> InputStandardizer:
    values = _finite_matrix(train_features, "train_features")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be positive and finite")
    mean = values.mean(axis=0, dtype=np.float64)
    # Population standard deviation is the fixed feature standardization.
    scale = np.maximum(values.std(axis=0, ddof=0, dtype=np.float64), floor)
    return InputStandardizer(mean=mean, scale=scale, floor=float(floor))


def fit_robust_target_transform(
    train_targets: np.ndarray, floor: float = TRANSFORM_FLOOR
) -> RobustTargetTransform:
    values = _finite_matrix(train_targets, "train_targets", columns=OUTPUT_DIM)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be positive and finite")
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, floor)
    return RobustTargetTransform(center=center, scale=scale, floor=float(floor))


def _validated_pair_orders(
    pair_orders: Sequence[np.ndarray | Tensor], rows: int, outputs: int
) -> tuple[Tensor, ...]:
    if len(pair_orders) != outputs:
        raise ValueError("one pair order is required for each output")
    validated: list[Tensor] = []
    expected = np.arange(rows, dtype=np.int64)
    for order in pair_orders:
        raw = order.detach().cpu().numpy() if isinstance(order, Tensor) else np.asarray(order)
        if raw.ndim != 1 or len(raw) != rows or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("each pair order must be an integer permutation of all rows")
        raw = raw.astype(np.int64, copy=False)
        if not np.array_equal(np.sort(raw), expected):
            raise ValueError("each pair order must be an integer permutation of all rows")
        validated.append(torch.from_numpy(raw.copy()))
    return tuple(validated)


def shuffled_pair_orders(
    rows: int, *, outputs: int = OUTPUT_DIM, rng: int | np.random.Generator
) -> tuple[np.ndarray, ...]:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    if isinstance(outputs, bool) or not isinstance(outputs, int) or outputs <= 0:
        raise ValueError("outputs must be a positive integer")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return tuple(generator.permutation(rows).astype(np.int64) for _ in range(outputs))


def adjacent_pair_ranking_loss(
    predictions: Tensor,
    targets: Tensor,
    pair_orders: Sequence[np.ndarray | Tensor],
) -> Tensor:
    """Equal-output pairwise logistic loss on shuffled adjacent row pairs."""

    if predictions.ndim != 2 or targets.shape != predictions.shape:
        raise ValueError("predictions and targets must have the same [rows, outputs] shape")
    if predictions.shape[0] == 0 or predictions.shape[1] == 0:
        raise ValueError("ranking inputs must be nonempty")
    orders = _validated_pair_orders(pair_orders, predictions.shape[0], predictions.shape[1])
    output_losses: list[Tensor] = []
    for output, cpu_order in enumerate(orders):
        usable = len(cpu_order) - len(cpu_order) % 2
        if usable == 0:
            output_losses.append(predictions[:, output].sum() * 0.0)
            continue
        order = cpu_order[:usable].to(device=predictions.device)
        left, right = order[0::2], order[1::2]
        target_difference = targets[left, output] - targets[right, output]
        nonzero = target_difference != 0
        if not bool(torch.any(nonzero).item()):
            output_losses.append(predictions[:, output].sum() * 0.0)
            continue
        signed_prediction_difference = torch.sign(target_difference[nonzero]) * (
            predictions[left[nonzero], output] - predictions[right[nonzero], output]
        )
        output_losses.append(F.softplus(-signed_prediction_difference).mean())
    return torch.stack(output_losses).mean()


def gate_objective(
    predictions: Tensor,
    targets: Tensor,
    pair_orders: Sequence[np.ndarray | Tensor],
    *,
    beta: float = 0.5,
    ranking_weight: float = 0.25,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, Smooth-L1, and ranking components in standardized units."""

    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if not np.isfinite(ranking_weight) or ranking_weight < 0:
        raise ValueError("ranking_weight must be nonnegative and finite")
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("predictions and targets must share a two-dimensional shape")
    smooth = F.smooth_l1_loss(predictions, targets, reduction="mean", beta=float(beta))
    ranking = adjacent_pair_ranking_loss(predictions, targets, pair_orders)
    return smooth + float(ranking_weight) * ranking, smooth, ranking


def _objective_numpy(
    model: GateMLP,
    features: np.ndarray,
    targets: np.ndarray,
    config: GateTrainingConfig,
    device: torch.device,
    pair_orders: Sequence[np.ndarray],
) -> tuple[float, float, float]:
    model.eval()
    x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
    with torch.inference_mode():
        total, smooth, ranking = gate_objective(
            model(x),
            y,
            pair_orders,
            beta=config.smooth_l1_beta,
            ranking_weight=config.ranking_weight,
        )
    return float(total.item()), float(smooth.item()), float(ranking.item())


def train_gate_seed(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    *,
    seed: int,
    config: GateTrainingConfig | None = None,
    device: str | torch.device = "cpu",
) -> GateSeedResult:
    """Fit one preregistered seed with train-only transforms and early stopping."""

    cfg = (config or GateTrainingConfig()).validate()
    train_x = _finite_matrix(train_features, "train_features")
    train_y = _finite_matrix(train_targets, "train_targets", columns=OUTPUT_DIM)
    calibration_x = _finite_matrix(
        calibration_features, "calibration_features", columns=train_x.shape[1]
    )
    calibration_y = _finite_matrix(
        calibration_targets, "calibration_targets", columns=OUTPUT_DIM
    )
    if len(train_x) != len(train_y) or len(calibration_x) != len(calibration_y):
        raise ValueError("feature and target row counts differ")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    input_transform = fit_input_standardizer(train_x)
    target_transform = fit_robust_target_transform(train_y)
    train_x_std = input_transform.transform(train_x).astype(np.float32)
    train_y_std = target_transform.transform(train_y).astype(np.float32)
    calibration_x_std = input_transform.transform(calibration_x).astype(np.float32)
    calibration_y_std = target_transform.transform(calibration_y).astype(np.float32)

    target_device = torch.device(device)
    torch.manual_seed(seed)
    model = GateMLP(train_x.shape[1]).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    shuffle_rng = np.random.default_rng(seed)
    pair_rng = np.random.default_rng(seed)
    calibration_orders = shuffled_pair_orders(
        len(calibration_x_std), rng=cfg.calibration_pair_seed
    )

    best_objective = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int | bool]] = []
    batch_size = int(cfg.batch_size)

    for epoch in range(int(cfg.max_epochs)):
        model.train()
        order = shuffle_rng.permutation(len(train_x_std))
        total_weighted = 0.0
        smooth_weighted = 0.0
        ranking_weighted = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = torch.from_numpy(train_x_std[indices]).to(target_device)
            y = torch.from_numpy(train_y_std[indices]).to(target_device)
            pair_orders = shuffled_pair_orders(len(indices), rng=pair_rng)
            prediction = model(x)
            objective, smooth, ranking = gate_objective(
                prediction,
                y,
                pair_orders,
                beta=cfg.smooth_l1_beta,
                ranking_weight=cfg.ranking_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            rows = len(indices)
            total_weighted += float(objective.detach().item()) * rows
            smooth_weighted += float(smooth.detach().item()) * rows
            ranking_weighted += float(ranking.detach().item()) * rows
            seen += rows

        calibration_objective, calibration_smooth, calibration_ranking = _objective_numpy(
            model,
            calibration_x_std,
            calibration_y_std,
            cfg,
            target_device,
            calibration_orders,
        )
        improved = calibration_objective < best_objective - cfg.early_stopping_min_delta
        if improved:
            best_objective = calibration_objective
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_objective": total_weighted / seen,
                "train_smooth_l1": smooth_weighted / seen,
                "train_ranking": ranking_weighted / seen,
                "calibration_objective": calibration_objective,
                "calibration_smooth_l1": calibration_smooth,
                "calibration_ranking": calibration_ranking,
                "best_calibration_objective": best_objective,
                "improved": improved,
            }
        )
        if stale >= int(cfg.early_stopping_patience):
            break

    if best_state is None or best_epoch < 0 or not np.isfinite(best_objective):
        raise RuntimeError("gate training did not produce a finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return GateSeedResult(
        seed=seed,
        model=model,
        input_transform=input_transform,
        target_transform=target_transform,
        best_epoch=best_epoch,
        best_calibration_objective=best_objective,
        epochs_completed=len(history),
        history=history,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def predict_gate(
    result_or_model: GateSeedResult | GateMLP,
    features: np.ndarray,
    input_transform: InputStandardizer | None = None,
    target_transform: RobustTargetTransform | None = None,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """Predict ``b12`` and ``b14`` in original raw-MSE units."""

    if isinstance(result_or_model, GateSeedResult):
        model = result_or_model.model
        input_transform = result_or_model.input_transform
        target_transform = result_or_model.target_transform
    else:
        model = result_or_model
    if input_transform is None or target_transform is None:
        raise ValueError("input_transform and target_transform are required")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    values = input_transform.transform(features).astype(np.float32)
    if values.shape[1] != model.input_dim:
        raise ValueError("feature width does not match gate input dimension")
    if device is None:
        target_device = next(model.parameters()).device
    else:
        target_device = torch.device(device)
        model = model.to(target_device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(values[start : start + batch_size]).to(target_device)
            chunks.append(model(x).detach().cpu().numpy().astype(np.float64))
    standardized = np.concatenate(chunks, axis=0)
    predicted = target_transform.inverse(standardized)
    if not np.all(np.isfinite(predicted)):
        raise RuntimeError("gate produced nonfinite predictions")
    return predicted


def _average_ranks(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    order = np.argsort(raw, kind="mergesort")
    ranks = np.empty(len(raw), dtype=np.float64)
    start = 0
    while start < len(raw):
        end = start + 1
        while end < len(raw) and raw[order[end]] == raw[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    result = float(np.corrcoef(x, y)[0, 1])
    return result if np.isfinite(result) else 0.0


def calibration_diagnostics(
    result: GateSeedResult,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
) -> dict[str, Any]:
    actual = _finite_matrix(
        calibration_targets, "calibration_targets", columns=OUTPUT_DIM
    )
    predicted = predict_gate(result, calibration_features)
    if len(predicted) != len(actual):
        raise ValueError("calibration feature and target row counts differ")
    rows = []
    spearman = []
    for column, name in enumerate(OUTPUT_NAMES):
        error = predicted[:, column] - actual[:, column]
        pearson = _safe_correlation(predicted[:, column], actual[:, column])
        rho = _safe_correlation(
            _average_ranks(predicted[:, column]), _average_ranks(actual[:, column])
        )
        spearman.append(rho)
        rows.append(
            {
                "benefit": name,
                "pearson": pearson,
                "spearman": rho,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "predicted_mean": float(predicted[:, column].mean()),
                "actual_mean": float(actual[:, column].mean()),
            }
        )
    return {
        "seed": result.seed,
        "best_epoch": result.best_epoch,
        "calibration_objective": result.best_calibration_objective,
        "mean_spearman": float(np.mean(spearman)),
        "outputs": rows,
    }


def select_seed_from_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]], seed_order: Sequence[int]
) -> int:
    if not diagnostics:
        raise ValueError("diagnostics must not be empty")
    order = tuple(int(seed) for seed in seed_order)
    if len(set(order)) != len(order):
        raise ValueError("seed_order must be unique")
    by_seed = {int(row["seed"]): row for row in diagnostics}
    if set(by_seed) != set(order) or len(by_seed) != len(diagnostics):
        raise ValueError("diagnostics seeds must exactly match seed_order")
    selected = order[0]
    for seed in order[1:]:
        candidate = by_seed[seed]
        current = by_seed[selected]
        candidate_rho = float(candidate["mean_spearman"])
        current_rho = float(current["mean_spearman"])
        candidate_objective = float(candidate["calibration_objective"])
        current_objective = float(current["calibration_objective"])
        if not all(
            np.isfinite(value)
            for value in (candidate_rho, current_rho, candidate_objective, current_objective)
        ):
            raise ValueError("seed-selection diagnostics must be finite")
        if candidate_rho > current_rho + 1e-12:
            selected = seed
        elif abs(candidate_rho - current_rho) <= 1e-12:
            if candidate_objective < current_objective:
                selected = seed
            # Exact objective ties retain the earlier preregistered seed.
    return selected


def select_gate_seed(
    results: Sequence[GateSeedResult],
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    *,
    seed_order: Sequence[int] = (260813, 260814, 260815),
) -> tuple[GateSeedResult, list[dict[str, Any]]]:
    if len({result.seed for result in results}) != len(results):
        raise ValueError("gate seed results must be unique")
    diagnostics = [
        calibration_diagnostics(result, calibration_features, calibration_targets)
        for result in results
    ]
    selected_seed = select_seed_from_diagnostics(diagnostics, seed_order)
    return next(result for result in results if result.seed == selected_seed), diagnostics


def benefit_quantile_rows(
    predicted_benefits: np.ndarray,
    actual_benefits: np.ndarray,
    *,
    quantiles: int = 5,
) -> list[dict[str, float | int | str]]:
    predicted = _finite_matrix(
        predicted_benefits, "predicted_benefits", columns=OUTPUT_DIM
    )
    actual = _finite_matrix(actual_benefits, "actual_benefits", columns=OUTPUT_DIM)
    if predicted.shape != actual.shape:
        raise ValueError("predicted and actual benefit shapes differ")
    if isinstance(quantiles, bool) or not isinstance(quantiles, int) or quantiles <= 0:
        raise ValueError("quantiles must be a positive integer")
    rows: list[dict[str, float | int | str]] = []
    for column, name in enumerate(OUTPUT_NAMES):
        order = np.argsort(predicted[:, column], kind="mergesort")
        for quantile, indices in enumerate(np.array_split(order, quantiles)):
            if len(indices) == 0:
                continue
            rows.append(
                {
                    "benefit": name,
                    "predicted_score_quantile": quantile,
                    "n": int(len(indices)),
                    "mean_predicted_benefit": float(predicted[indices, column].mean()),
                    "mean_actual_benefit": float(actual[indices, column].mean()),
                }
            )
    return rows


def gate_parameter_count(input_dim: int) -> int:
    return int(sum(parameter.numel() for parameter in GateMLP(input_dim).parameters()))


def gate_dense_linear_flops(input_dim: int) -> int:
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer")
    return 2 * (
        input_dim * HIDDEN_DIMS[0]
        + HIDDEN_DIMS[0] * HIDDEN_DIMS[1]
        + HIDDEN_DIMS[1] * OUTPUT_DIM
    )


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def state_tensor_hash(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, Tensor):
            raise TypeError("state dictionaries may contain only tensors")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


_CHECKPOINT_KEYS = frozenset(
    {
        "format_version",
        "architecture",
        "seed",
        "best_epoch",
        "best_calibration_objective",
        "feature_names",
        "state_dict",
        "input_mean",
        "input_scale",
        "target_center",
        "target_scale",
        "state_tensor_hash",
    }
)


def save_gate_checkpoint(
    path: str | Path,
    result: GateSeedResult,
    *,
    feature_names: Iterable[str],
) -> dict[str, str]:
    names = tuple(feature_names)
    if len(names) != result.model.input_dim or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match gate input width")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature names must be nonempty strings")
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in result.model.state_dict().items()
    }
    tensor_hash = state_tensor_hash(state)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": {
            "input_dim": result.model.input_dim,
            "hidden_dims": list(HIDDEN_DIMS),
            "output_dim": OUTPUT_DIM,
            "dropout": DROPOUT,
        },
        "seed": result.seed,
        "best_epoch": result.best_epoch,
        "best_calibration_objective": result.best_calibration_objective,
        "feature_names": list(names),
        "state_dict": state,
        "input_mean": torch.from_numpy(result.input_transform.mean.copy()),
        "input_scale": torch.from_numpy(result.input_transform.scale.copy()),
        "target_center": torch.from_numpy(result.target_transform.center.copy()),
        "target_scale": torch.from_numpy(result.target_transform.scale.copy()),
        "state_tensor_hash": tensor_hash,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {"file_sha256": sha256_file(destination), "state_tensor_hash": tensor_hash}


def load_gate_checkpoint(
    path: str | Path,
    *,
    expected_input_dim: int,
    expected_feature_names: Iterable[str] | None = None,
    expected_seed: int | None = None,
    expected_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> LoadedGateCheckpoint:
    source = Path(path)
    file_hash = sha256_file(source)
    if expected_sha256 is not None and file_hash != expected_sha256:
        raise RuntimeError(
            f"gate checkpoint SHA-256 mismatch: expected {expected_sha256}, got {file_hash}"
        )
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        observed = set(payload) if isinstance(payload, dict) else set()
        raise RuntimeError(
            f"strict gate checkpoint keys mismatch: missing={sorted(_CHECKPOINT_KEYS-observed)}, "
            f"unexpected={sorted(observed-_CHECKPOINT_KEYS)}"
        )
    expected_architecture = {
        "input_dim": expected_input_dim,
        "hidden_dims": list(HIDDEN_DIMS),
        "output_dim": OUTPUT_DIM,
        "dropout": DROPOUT,
    }
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError("unsupported gate checkpoint format version")
    if payload["architecture"] != expected_architecture:
        raise RuntimeError("gate checkpoint architecture mismatch")
    seed = int(payload["seed"])
    if expected_seed is not None and seed != expected_seed:
        raise RuntimeError(f"gate checkpoint seed mismatch: expected {expected_seed}, got {seed}")
    names = tuple(payload["feature_names"])
    if len(names) != expected_input_dim or len(set(names)) != len(names):
        raise RuntimeError("gate checkpoint feature-name contract is invalid")
    if expected_feature_names is not None and names != tuple(expected_feature_names):
        raise RuntimeError("gate checkpoint feature names mismatch")

    model = GateMLP(expected_input_dim)
    expected_state_keys = set(model.state_dict())
    state = payload["state_dict"]
    if not isinstance(state, dict) or set(state) != expected_state_keys:
        observed = set(state) if isinstance(state, dict) else set()
        raise RuntimeError(
            f"strict gate state keys mismatch: missing={sorted(expected_state_keys-observed)}, "
            f"unexpected={sorted(observed-expected_state_keys)}"
        )
    tensor_hash = state_tensor_hash(state)
    if tensor_hash != payload["state_tensor_hash"]:
        raise RuntimeError("gate checkpoint state tensor hash mismatch")
    if any(not torch.isfinite(tensor).all().item() for tensor in state.values()):
        raise RuntimeError("gate checkpoint contains nonfinite parameters")
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device)).eval()

    input_mean = payload["input_mean"].detach().cpu().numpy().astype(np.float64)
    input_scale = payload["input_scale"].detach().cpu().numpy().astype(np.float64)
    target_center = payload["target_center"].detach().cpu().numpy().astype(np.float64)
    target_scale = payload["target_scale"].detach().cpu().numpy().astype(np.float64)
    if input_mean.shape != (expected_input_dim,) or input_scale.shape != (expected_input_dim,):
        raise RuntimeError("gate checkpoint input-transform shape mismatch")
    if target_center.shape != (OUTPUT_DIM,) or target_scale.shape != (OUTPUT_DIM,):
        raise RuntimeError("gate checkpoint target-transform shape mismatch")
    if not all(
        np.all(np.isfinite(value))
        for value in (input_mean, input_scale, target_center, target_scale)
    ) or np.any(input_scale < TRANSFORM_FLOOR) or np.any(target_scale < TRANSFORM_FLOOR):
        raise RuntimeError("gate checkpoint contains an invalid transform")
    objective = float(payload["best_calibration_objective"])
    if not np.isfinite(objective):
        raise RuntimeError("gate checkpoint calibration objective is nonfinite")
    return LoadedGateCheckpoint(
        model=model,
        input_transform=InputStandardizer(input_mean, input_scale),
        target_transform=RobustTargetTransform(target_center, target_scale),
        seed=seed,
        best_epoch=int(payload["best_epoch"]),
        best_calibration_objective=objective,
        feature_names=names,
        state_tensor_hash=tensor_hash,
        file_sha256=file_hash,
    )


__all__ = [
    "DROPOUT",
    "GateMLP",
    "GateSeedResult",
    "GateTrainingConfig",
    "HIDDEN_DIMS",
    "InputStandardizer",
    "LoadedGateCheckpoint",
    "OUTPUT_NAMES",
    "RobustTargetTransform",
    "adjacent_pair_ranking_loss",
    "benefit_quantile_rows",
    "calibration_diagnostics",
    "fit_input_standardizer",
    "fit_robust_target_transform",
    "gate_dense_linear_flops",
    "gate_objective",
    "gate_parameter_count",
    "load_gate_checkpoint",
    "predict_gate",
    "save_gate_checkpoint",
    "select_gate_seed",
    "select_seed_from_diagnostics",
    "sha256_file",
    "shuffled_pair_orders",
    "state_tensor_hash",
    "train_gate_seed",
]
