"""Task E fixed-bank identities, schedule boundary, metrics, and inference."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy import stats
from torch import Tensor

from pusht_cem_adapter import PushTRefinedCostModel


DEPTHS = (0, 1, 2, 4)
TRANSITIONS = 5


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def candidate_hashes(candidates: np.ndarray) -> list[str]:
    value = np.asarray(candidates, dtype="<f4")
    if value.ndim != 3 or value.shape[1:] != (5, 10):
        raise ValueError("candidates must have shape [population,5,10]")
    return [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in value]


def validate_schedule(schedule: Iterable[int]) -> tuple[int, ...]:
    result = tuple(schedule)
    if len(result) != TRANSITIONS:
        raise ValueError("depth schedule must contain five transitions")
    if any(isinstance(depth, bool) or depth not in DEPTHS for depth in result):
        raise ValueError("schedule contains an unsupported depth")
    return result


@dataclass
class ScheduleTrace:
    schedule: tuple[int, ...]
    history_limit: int
    stage_calls: list[int]
    residual_norm_mean: list[float]


class ScheduledPushTRefinedCostModel(PushTRefinedCostModel):
    """Offline-only fixed transition schedule; never used as a controller."""

    def __init__(self, base, refiner, schedule: Iterable[int], history_limit: int = 3):
        schedule = validate_schedule(schedule)
        if history_limit not in (1, 2, 3):
            raise ValueError("history_limit must be 1, 2, or 3")
        super().__init__(base, refiner, max(schedule))
        self.schedule = schedule
        self.history_limit = history_limit
        self.last_trace: ScheduleTrace | None = None
        self.last_predictions: list[Tensor] = []
        self.last_base_predictions: list[Tensor] = []

    def rollout(self, info: dict, action_sequence: Tensor) -> dict:
        observation_history = info["pixels"].size(2)
        batch, samples, horizon = action_sequence.shape[:3]
        if horizon != TRANSITIONS:
            raise ValueError("Task E freezes five predicted transitions")
        initial_actions, future_actions = torch.split(
            action_sequence,
            [observation_history, horizon - observation_history],
            dim=2,
        )
        initial = {
            key: value[:, 0] for key, value in info.items() if torch.is_tensor(value)
        }
        initial["action"] = initial_actions[:, 0]
        initial = self.base.encode(initial)
        embedding = initial["emb"].unsqueeze(1).expand(batch, samples, -1, -1)
        embedding = rearrange(embedding, "b s ... -> (b s) ...").clone()
        actions = rearrange(initial_actions, "b s ... -> (b s) ...")
        future_actions = rearrange(future_actions, "b s ... -> (b s) ...")
        stage_calls = [0] * 4
        residual_means: list[float] = []
        self.last_predictions = []
        self.last_base_predictions = []

        for transition, depth in enumerate(self.schedule):
            latent_history = embedding[:, -self.history_limit :]
            action_history = actions[:, -self.history_limit :]
            encoded_action = self.base.action_encoder(action_history)
            base_prediction = self.base.predict(latent_history, encoded_action)[:, -1]
            self.last_base_predictions.append(base_prediction.detach().cpu())
            if depth == 0:
                self.ledger.record_base(len(base_prediction))
                prediction = base_prediction
            else:
                prediction = self._refine(
                    latent_history, action_history, base_prediction
                )
                for stage in range(depth):
                    stage_calls[stage] += 1
            residual_means.append(
                float(torch.linalg.vector_norm(prediction - base_prediction, dim=1).mean())
            )
            self.last_predictions.append(prediction.detach().cpu())
            embedding = torch.cat([embedding, prediction[:, None]], dim=1)
            if transition < horizon - observation_history:
                actions = torch.cat(
                    [actions, future_actions[:, transition : transition + 1]], dim=1
                )

        self.last_trace = ScheduleTrace(
            self.schedule, self.history_limit, stage_calls, residual_means
        )
        info["predicted_emb"] = rearrange(
            embedding, "(b s) ... -> b s ...", b=batch, s=samples
        )
        return info


def score_candidates(model: ScheduledPushTRefinedCostModel, info: dict, candidates: Tensor) -> np.ndarray:
    samples = int(candidates.shape[1])
    expanded = {
        key: value.unsqueeze(1).expand(1, samples, *value.shape[1:])
        for key, value in info.items()
    }
    with torch.inference_mode():
        result = model.get_cost(expanded, candidates.clone())
    return result.detach().cpu().numpy()[0].astype(np.float64)


def _tie_groups(values: np.ndarray, tolerance: float) -> list[np.ndarray]:
    order = np.argsort(values, kind="stable")
    groups: list[list[int]] = []
    for index in order:
        if not groups or abs(values[index] - values[groups[-1][0]]) > tolerance:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return [np.asarray(group, dtype=np.int64) for group in groups]


def top_set(values: np.ndarray, k: int, tolerance: float) -> np.ndarray:
    if not 0 < k <= len(values):
        raise ValueError("invalid top-k")
    groups = _tie_groups(np.asarray(values, float), tolerance)
    selected: list[int] = []
    for group in groups:
        selected.extend(group.tolist())
        if len(selected) >= k:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def pairwise_concordance(
    prediction_cost: np.ndarray,
    simulator_cost: np.ndarray,
    prediction_tolerance: float,
    simulator_tolerance: float,
) -> dict:
    prediction = np.asarray(prediction_cost, float)
    reference = np.asarray(simulator_cost, float)
    if prediction.shape != reference.shape or prediction.ndim != 1:
        raise ValueError("ranking vectors must be matching one-dimensional arrays")
    concordant = discordant = prediction_ties = simulator_ties = 0
    for left in range(len(prediction) - 1):
        pd = prediction[left] - prediction[left + 1 :]
        sd = reference[left] - reference[left + 1 :]
        valid = np.abs(sd) > simulator_tolerance
        simulator_ties += int((~valid).sum())
        prediction_ties += int((valid & (np.abs(pd) <= prediction_tolerance)).sum())
        signed = pd[valid] * sd[valid]
        concordant += int((signed > prediction_tolerance * simulator_tolerance).sum())
        discordant += int((signed < -prediction_tolerance * simulator_tolerance).sum())
    denominator = concordant + discordant + prediction_ties
    return {
        "concordance": (
            float((concordant + 0.5 * prediction_ties) / denominator)
            if denominator
            else None
        ),
        "concordant": concordant,
        "discordant": discordant,
        "prediction_ties": prediction_ties,
        "simulator_ties": simulator_ties,
    }


def ranking_metrics(
    prediction_cost: np.ndarray,
    simulator_cost: np.ndarray,
    *,
    prediction_tolerance: float = 1e-8,
    simulator_tolerance: float = 1e-7,
    top_ks: tuple[int, ...] = (10, 30),
) -> dict:
    prediction = np.asarray(prediction_cost, float)
    reference = np.asarray(simulator_cost, float)
    valid = np.isfinite(prediction) & np.isfinite(reference)
    prediction, reference = prediction[valid], reference[valid]
    if len(prediction) == 0:
        raise ValueError("no valid candidates")
    selected = int(np.argmin(prediction))
    oracle = float(np.min(reference))
    result = {
        "valid_candidates": int(len(prediction)),
        "selected_index_within_valid": selected,
        "selected_simulator_cost": float(reference[selected]),
        "oracle_simulator_cost": oracle,
        "top_choice_regret": float(reference[selected] - oracle),
        "spearman": None,
        "kendall_tau_b": None,
    }
    if np.ptp(prediction) > prediction_tolerance and np.ptp(reference) > simulator_tolerance:
        result["spearman"] = float(stats.spearmanr(prediction, reference).statistic)
        result["kendall_tau_b"] = float(
            stats.kendalltau(prediction, reference, variant="b").statistic
        )
    result.update(
        pairwise_concordance(
            prediction, reference, prediction_tolerance, simulator_tolerance
        )
    )
    for k in top_ks:
        predicted_top = top_set(prediction, k, prediction_tolerance)
        simulator_top = top_set(reference, k, simulator_tolerance)
        overlap = np.intersect1d(predicted_top, simulator_top)
        union = np.union1d(predicted_top, simulator_top)
        result[f"top_{k}_predicted_size"] = int(len(predicted_top))
        result[f"top_{k}_simulator_size"] = int(len(simulator_top))
        result[f"top_{k}_recall"] = float(len(overlap) / len(simulator_top))
        result[f"top_{k}_jaccard"] = float(len(overlap) / len(union))
    return result


def selected_change(
    base_prediction: np.ndarray,
    depth_prediction: np.ndarray,
    simulator_cost: np.ndarray,
    candidates: np.ndarray,
    tie_tolerance: float,
) -> dict:
    base_index = int(np.argmin(base_prediction))
    depth_index = int(np.argmin(depth_prediction))
    difference = np.asarray(candidates[depth_index] - candidates[base_index], float)
    outcome_delta = float(simulator_cost[depth_index] - simulator_cost[base_index])
    return {
        "base_index": base_index,
        "depth_index": depth_index,
        "changed": depth_index != base_index,
        "action_l2": float(np.linalg.vector_norm(difference)),
        "action_max_absolute": float(np.max(np.abs(difference))),
        "simulator_cost_delta": outcome_delta,
        "direction": (
            "improves"
            if outcome_delta < -tie_tolerance
            else "harms"
            if outcome_delta > tie_tolerance
            else "ties"
        ),
    }
