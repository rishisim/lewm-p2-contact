#!/usr/bin/env python3
"""Fit on 240 episodes, then select once on a disjoint 120-episode role."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from cycle_common import (
    ADAPTER_FLOPS,
    BASE_FLOPS,
    FEATURE_DIM,
    REPO_ROOT,
    ROOT,
    SOURCE_DISCOVERY,
    V1_FLOPS,
    atomic_json,
    atomic_npz,
    read_json,
    sha256_file,
)
from compile_gate import main as compile_gate
from verify_design import verify as verify_design


GRID_PATH = ROOT / "candidate_grid.json"
FIT_PATH = ROOT / "data/fit_evaluated.npz"
SELECTION_PATH = ROOT / "data/selection_evaluated.npz"
FITTED_PATH = ROOT / "development/fitted_candidates.npz"
FIT_LOCK_PATH = ROOT / "development/fit_lock.json"
SELECTION_LEDGER_PATH = ROOT / "development/gate_selection_ledger.json"
GATE_FIT_PATH = ROOT / "freeze/gate_fit.npz"
GATE_FREEZE_PATH = ROOT / "freeze/gate_freeze.json"
GATE_SEAL_PATH = ROOT / "freeze/gate_freeze_seal.json"
GATE_FLOPS = 7_985
SELECTION_HISTOGRAM_SEED = 2_007_188_881
SELECTION_SEEDED_MIXTURE_SEED = 2_007_177_771


def strongest_mixture(losses: np.ndarray, mean_calls: float) -> dict[str, Any]:
    best = None
    for lower in range(1, 5):
        for upper in range(lower, 5):
            if not lower <= mean_calls <= upper:
                continue
            weight = 0.0 if lower == upper else (mean_calls - lower) / (upper - lower)
            values = (1.0 - weight) * losses[:, lower - 1] + weight * losses[:, upper - 1]
            candidate = (float(values.mean()), values, lower, upper, float(weight))
            if best is None or candidate[0] < best[0] or (
                candidate[0] == best[0] and candidate[2:4] < best[2:4]
            ):
                best = candidate
    if best is None:
        raise RuntimeError("no exact-compute analytic mixture")
    return {
        "mean_loss": best[0],
        "loss": best[1],
        "depth_lower": best[2],
        "depth_upper": best[3],
        "weight_upper": best[4],
    }


def episode_mean(values: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    return np.asarray([values[episode_id == item].mean() for item in np.unique(episode_id)])


def losses(arrays: dict[str, np.ndarray], whitening: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    difference = arrays["exits"].astype(np.float64) - arrays["target"].astype(np.float64)[:, None, :]
    raw = np.square(difference).mean(axis=2)
    transformed = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    white = np.square(transformed).mean(axis=2)
    return raw, white


def candidate_id(regularization: float, quantile: float) -> str:
    return f"stage_dual_r{regularization:g}_q{quantile:.2f}"


def load_role(role: str) -> dict[str, np.ndarray]:
    path = ROOT / f"data/{role}_evaluated.npz"
    manifest = read_json(ROOT / f"data/{role}_evaluated_manifest.json")
    if sha256_file(path) != manifest["sha256"]:
        raise RuntimeError(f"{role} evaluated hash drift")
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def fit() -> None:
    verify_design()
    if FITTED_PATH.exists() or FIT_LOCK_PATH.exists():
        raise RuntimeError("fit artifacts are immutable and already exist")
    if SELECTION_PATH.exists():
        # Selection may be generated/evaluated in parallel operationally, but the
        # fitter is prohibited from opening it. Existence is recorded, not read.
        selection_existed = True
    else:
        selection_existed = False
    grid = read_json(GRID_PATH)
    fit_arrays = load_role("fit")
    if len(np.unique(fit_arrays["episode_id"])) != 240:
        raise RuntimeError("fit role episode count drift")
    with np.load(
        SOURCE_DISCOVERY / "fit_only_normalization_whitening.npz", allow_pickle=False
    ) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    raw, white = losses(fit_arrays, whitening)
    raw_gain = raw[:, :3] - raw[:, 1:]
    white_gain = white[:, :3] - white[:, 1:]
    features = fit_arrays["features"].astype(np.float64)
    if features.shape != (240 * 38, 3, FEATURE_DIM):
        raise RuntimeError("fit feature shape drift")
    regularizations = np.asarray(grid["regularizations"], dtype=np.float64)
    quantiles = np.asarray(grid["fit_score_quantiles"], dtype=np.float64)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    raw_mean = raw_gain.mean(axis=0)
    white_mean = white_gain.mean(axis=0)
    raw_std = raw_gain.std(axis=0) + 1e-12
    white_std = white_gain.std(axis=0) + 1e-12
    weights = np.empty((len(regularizations), 3, 2, FEATURE_DIM), dtype=np.float64)
    thresholds = np.empty((len(regularizations), len(quantiles), 3), dtype=np.float64)
    fit_rhos = np.empty((len(regularizations), 3), dtype=np.float64)

    import torch

    for stage in range(3):
        z = (features[:, stage] - feature_mean[stage]) / feature_std[stage]
        y = np.column_stack(
            (
                (raw_gain[:, stage] - raw_mean[stage]) / raw_std[stage],
                (white_gain[:, stage] - white_mean[stage]) / white_std[stage],
            )
        )
        z_tensor = torch.as_tensor(z, dtype=torch.float64)
        y_tensor = torch.as_tensor(y, dtype=torch.float64)
        gram = z_tensor.T @ z_tensor
        cross = z_tensor.T @ y_tensor
        identity = torch.eye(FEATURE_DIM, dtype=torch.float64)
        combined_gain = 0.5 * (y[:, 0] + y[:, 1])
        for reg_index, regularization in enumerate(regularizations):
            solution = torch.linalg.solve(gram + float(regularization) * identity, cross)
            local_weights = solution.numpy().T
            weights[reg_index, stage] = local_weights
            scores = np.minimum(z @ local_weights[0], z @ local_weights[1])
            fit_rhos[reg_index, stage] = float(spearmanr(scores, combined_gain).statistic)
            for quantile_index, quantile in enumerate(quantiles):
                thresholds[reg_index, quantile_index, stage] = float(
                    np.quantile(scores, float(quantile))
                )
    if not all(
        np.isfinite(value).all()
        for value in (
            feature_mean,
            feature_std,
            raw_mean,
            raw_std,
            white_mean,
            white_std,
            weights,
            thresholds,
            fit_rhos,
        )
    ):
        raise RuntimeError("nonfinite fit artifact")
    atomic_npz(
        FITTED_PATH,
        {
            "regularizations": regularizations,
            "quantiles": quantiles,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "raw_target_mean": raw_mean,
            "raw_target_std": raw_std,
            "white_target_mean": white_mean,
            "white_target_std": white_std,
            "weights": weights,
            "thresholds": thresholds,
            "fit_score_gain_rhos": fit_rhos,
        },
    )
    lock = {
        "schema_version": 1,
        "status": "all_nine_candidates_fit_and_locked_before_selection_open",
        "created_unix_ns": time.time_ns(),
        "grid_sha256": sha256_file(GRID_PATH),
        "fit_input_sha256": sha256_file(FIT_PATH),
        "fit_manifest_sha256": sha256_file(ROOT / "data/fit_evaluated_manifest.json"),
        "fitted_candidates_sha256": sha256_file(FITTED_PATH),
        "candidate_count": 9,
        "selection_input_existed_but_was_not_opened": selection_existed,
        "selection_target_or_feature_accessed": False,
        "fit_episode_count": 240,
        "native_whitening_unchanged_sha256": sha256_file(
            SOURCE_DISCOVERY / "fit_only_normalization_whitening.npz"
        ),
        "v5_outcome_episodes": 0,
    }
    atomic_json(FIT_LOCK_PATH, lock, exclusive=True)
    print(json.dumps(lock, sort_keys=True))


def calls_from_scores(scores: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    calls = np.ones(len(scores), dtype=np.int64)
    active = np.ones(len(scores), dtype=bool)
    reached = []
    for stage in range(3):
        reached.append(active.copy())
        if not np.isfinite(scores[active, stage]).all():
            raise RuntimeError("nonfinite active selection score")
        active &= scores[:, stage] > thresholds[stage]
        calls += active.astype(np.int64)
    return calls, reached


def evaluate_selection_candidate(
    arrays: dict[str, np.ndarray],
    whitening: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    regularization: float,
    quantile: float,
) -> dict[str, Any]:
    raw, white = losses(arrays, whitening)
    calls, reached = calls_from_scores(scores, thresholds)
    row_index = np.arange(len(calls))
    adaptive_raw = raw[row_index, calls - 1]
    adaptive_white = white[row_index, calls - 1]
    episode_id = arrays["episode_id"].astype(np.int64)
    gate_evaluations = int(np.minimum(calls, 3).sum())
    total_calls = int(calls.sum())
    common = len(calls) * (BASE_FLOPS + V1_FLOPS)
    total_flops = int(
        common
        + (total_calls - len(calls)) * ADAPTER_FLOPS
        + gate_evaluations * GATE_FLOPS
    )
    analytic_total_calls = total_calls + gate_evaluations * GATE_FLOPS / ADAPTER_FLOPS
    analytic_mean_calls = analytic_total_calls / len(calls)
    raw_analytic = strongest_mixture(raw, analytic_mean_calls)
    white_analytic = strongest_mixture(white, analytic_mean_calls)
    integer_target = math.ceil(analytic_total_calls)
    seeded_shape = strongest_mixture(raw, integer_target / len(calls))
    lower = int(seeded_shape["depth_lower"])
    upper = int(seeded_shape["depth_upper"])
    number_upper = (
        0
        if lower == upper
        else math.ceil((integer_target - lower * len(calls)) / (upper - lower))
    )
    seeded_calls = np.full(len(calls), lower, dtype=np.int64)
    seeded_rng = np.random.default_rng(SELECTION_SEEDED_MIXTURE_SEED)
    seeded_calls[seeded_rng.permutation(len(calls))[:number_upper]] = upper
    seeded_raw = raw[row_index, seeded_calls - 1]
    seeded_flops = int(common + (int(seeded_calls.sum()) - len(calls)) * ADAPTER_FLOPS)
    histogram_calls = np.empty_like(calls)
    histogram_rng = np.random.default_rng(SELECTION_HISTOGRAM_SEED)
    histogram_exact = True
    for episode in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == episode)
        histogram_calls[indices] = calls[indices][histogram_rng.permutation(len(indices))]
        histogram_exact &= bool(
            np.array_equal(
                np.bincount(calls[indices], minlength=5),
                np.bincount(histogram_calls[indices], minlength=5),
            )
        )
    histogram_raw = raw[row_index, histogram_calls - 1]
    histogram_white = white[row_index, histogram_calls - 1]
    episode_contrasts = {
        "raw_vs_analytic": episode_mean(raw_analytic["loss"] - adaptive_raw, episode_id),
        "raw_vs_seeded": episode_mean(seeded_raw - adaptive_raw, episode_id),
        "raw_vs_fixed_d1": episode_mean(raw[:, 0] - adaptive_raw, episode_id),
        "raw_vs_within_episode_histogram": episode_mean(
            histogram_raw - adaptive_raw, episode_id
        ),
        "native_whitened_vs_analytic": episode_mean(
            white_analytic["loss"] - adaptive_white, episode_id
        ),
        "native_whitened_vs_within_episode_histogram": episode_mean(
            histogram_white - adaptive_white, episode_id
        ),
    }
    point = {name: float(values.mean()) for name, values in episode_contrasts.items()}
    standardized_co_primary = {
        name: float(values.mean() / (values.std(ddof=1) + 1e-18))
        for name, values in episode_contrasts.items()
        if name in ("raw_vs_analytic", "native_whitened_vs_analytic")
    }
    stagewise = []
    for stage in range(3):
        raw_gain = raw[:, stage] - raw[:, stage + 1]
        white_gain = white[:, stage] - white[:, stage + 1]
        combined = 0.5 * (
            raw_gain / (raw_gain.std() + 1e-12)
            + white_gain / (white_gain.std() + 1e-12)
        )
        rho = float(spearmanr(scores[reached[stage], stage], combined[reached[stage]]).statistic)
        stagewise.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached[stage].sum()),
                "rho": rho,
                "positive": bool(np.isfinite(rho) and rho > 0),
            }
        )
    histogram = np.bincount(calls, minlength=5)[1:].tolist()
    eligibility = {
        "all_six_point_contrasts_positive": all(value > 0 for value in point.values()),
        "all_three_out_of_fit_rhos_positive": all(item["positive"] for item in stagewise),
        "stage_2_reached_at_least_400": stagewise[1]["reached_rows"] >= 400,
        "stage_3_reached_at_least_100": stagewise[2]["reached_rows"] >= 100,
        "all_four_depths_at_least_15": all(value >= 15 for value in histogram),
        "seeded_comparator_weakly_more_compute": seeded_flops >= total_flops,
        "within_episode_histograms_exact": histogram_exact,
    }
    return {
        "candidate_id": candidate_id(regularization, quantile),
        "regularization": regularization,
        "fit_score_quantile": quantile,
        "thresholds": thresholds.tolist(),
        "eligible": all(eligibility.values()),
        "eligibility": eligibility,
        "point_contrasts": point,
        "standardized_co_primary": standardized_co_primary,
        "minimum_standardized_co_primary": min(standardized_co_primary.values()),
        "stagewise_rank": stagewise,
        "minimum_stagewise_rho": min(item["rho"] for item in stagewise),
        "call_histogram": histogram,
        "mean_calls": float(calls.mean()),
        "gate_evaluations": gate_evaluations,
        "adaptive_total_flops": total_flops,
        "analytic_equivalent_mean_calls": float(analytic_mean_calls),
        "seeded_total_flops": seeded_flops,
        "seeded_minus_adaptive_flops": seeded_flops - total_flops,
    }


def select() -> None:
    verify_design()
    if not FIT_LOCK_PATH.exists() or not FITTED_PATH.exists():
        raise RuntimeError("fit lock required before selection")
    if any(path.exists() for path in (SELECTION_LEDGER_PATH, GATE_FIT_PATH, GATE_FREEZE_PATH, GATE_SEAL_PATH)):
        raise RuntimeError("selection/freeze artifacts are immutable and already exist")
    fit_lock = read_json(FIT_LOCK_PATH)
    if fit_lock["fitted_candidates_sha256"] != sha256_file(FITTED_PATH):
        raise RuntimeError("fitted candidate hash drift")
    selection_arrays = load_role("selection")
    if len(np.unique(selection_arrays["episode_id"])) != 120:
        raise RuntimeError("selection role episode count drift")
    with np.load(
        SOURCE_DISCOVERY / "fit_only_normalization_whitening.npz", allow_pickle=False
    ) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    with np.load(FITTED_PATH, allow_pickle=False) as stored:
        fitted = {name: stored[name].copy() for name in stored.files}
    features = selection_arrays["features"].astype(np.float64)
    candidates = []
    score_cache = {}
    for reg_index, regularization in enumerate(fitted["regularizations"]):
        scores = np.empty((len(features), 3), dtype=np.float64)
        for stage in range(3):
            z = (features[:, stage] - fitted["feature_mean"][stage]) / fitted["feature_std"][stage]
            wr = fitted["weights"][reg_index, stage, 0]
            ww = fitted["weights"][reg_index, stage, 1]
            scores[:, stage] = np.minimum(z @ wr, z @ ww)
        score_cache[reg_index] = scores
        for quantile_index, quantile in enumerate(fitted["quantiles"]):
            candidates.append(
                evaluate_selection_candidate(
                    selection_arrays,
                    whitening,
                    scores,
                    fitted["thresholds"][reg_index, quantile_index],
                    float(regularization),
                    float(quantile),
                )
            )
    eligible = [item for item in candidates if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            -item["minimum_stagewise_rho"],
            -item["minimum_standardized_co_primary"],
            item["adaptive_total_flops"],
            item["regularization"],
            item["fit_score_quantile"],
            item["candidate_id"],
        )
    )
    selected = eligible[0] if eligible else None
    ledger = {
        "schema_version": 1,
        "status": "selection_complete_no_refit",
        "created_unix_ns": time.time_ns(),
        "fit_lock_sha256": sha256_file(FIT_LOCK_PATH),
        "selection_input_sha256": sha256_file(SELECTION_PATH),
        "selection_manifest_sha256": sha256_file(
            ROOT / "data/selection_evaluated_manifest.json"
        ),
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "candidates": candidates,
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "fixed_ranking_applied": True,
        "selected_head_refit_after_selection": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(SELECTION_LEDGER_PATH, ledger, exclusive=True)
    if selected is None:
        failure = {
            "schema_version": 1,
            "status": "no_eligible_stage_specific_gate",
            "selection_ledger_sha256": sha256_file(SELECTION_LEDGER_PATH),
            "smoke_episodes": 0,
            "prospective_episodes": 0,
            "v5_outcome_episodes": 0,
        }
        atomic_json(ROOT / "development/no_gate_selected.json", failure, exclusive=True)
        print(json.dumps(failure, sort_keys=True))
        return
    reg_index = int(
        np.flatnonzero(np.isclose(fitted["regularizations"], selected["regularization"]))[0]
    )
    quantile_index = int(
        np.flatnonzero(np.isclose(fitted["quantiles"], selected["fit_score_quantile"]))[0]
    )
    atomic_npz(
        GATE_FIT_PATH,
        {
            "feature_mean": fitted["feature_mean"],
            "feature_std": fitted["feature_std"],
            "raw_target_mean": fitted["raw_target_mean"],
            "raw_target_std": fitted["raw_target_std"],
            "white_target_mean": fitted["white_target_mean"],
            "white_target_std": fitted["white_target_std"],
            "wr": fitted["weights"][reg_index, :, 0],
            "ww": fitted["weights"][reg_index, :, 1],
            "thresholds": fitted["thresholds"][reg_index, quantile_index],
            "regularization": np.asarray(selected["regularization"], dtype=np.float64),
            "fit_score_quantile": np.asarray(
                selected["fit_score_quantile"], dtype=np.float64
            ),
        },
    )
    compile_gate()
    gate_freeze = {
        "schema_version": 1,
        "status": "frozen_after_fit_selection_before_any_smoke_or_prospective_episode",
        "created_unix_ns": time.time_ns(),
        "selected": selected,
        "fit_selection_isolation_passed": True,
        "fit_episode_count": 240,
        "selection_episode_count": 120,
        "selected_head_refit_after_selection": False,
        "candidate_grid_sha256": sha256_file(GRID_PATH),
        "fit_lock_sha256": sha256_file(FIT_LOCK_PATH),
        "selection_ledger_sha256": sha256_file(SELECTION_LEDGER_PATH),
        "gate_fit_sha256": sha256_file(GATE_FIT_PATH),
        "compiled_gate_sha256": sha256_file(ROOT / "freeze/compiled_gate.npz"),
        "native_whitening_sha256": sha256_file(
            SOURCE_DISCOVERY / "fit_only_normalization_whitening.npz"
        ),
        "causal_feature_width": FEATURE_DIM,
        "contact_or_privileged_inputs": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(GATE_FREEZE_PATH, gate_freeze, exclusive=True)
    seal_inputs = [
        GRID_PATH,
        ROOT / "FIT_SELECTION_PREREGISTRATION.md",
        ROOT / "fit_select_gate.py",
        ROOT / "compile_gate.py",
        ROOT / "data/fit_raw_manifest.json",
        ROOT / "data/fit_evaluated.npz",
        ROOT / "data/fit_evaluated_manifest.json",
        ROOT / "data/selection_raw_manifest.json",
        ROOT / "data/selection_evaluated.npz",
        ROOT / "data/selection_evaluated_manifest.json",
        FITTED_PATH,
        FIT_LOCK_PATH,
        SELECTION_LEDGER_PATH,
        GATE_FIT_PATH,
        ROOT / "freeze/compiled_gate.npz",
        GATE_FREEZE_PATH,
    ]
    seal = {
        "schema_version": 1,
        "status": "immutable_gate_freeze_before_smoke_and_prospective",
        "created_unix_ns": time.time_ns(),
        "files": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in seal_inputs
        },
        "fit_selection_isolation_passed": True,
        "smoke_episodes_at_seal": 0,
        "prospective_episodes_at_seal": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(GATE_SEAL_PATH, seal, exclusive=True)
    print(json.dumps(gate_freeze, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "select"))
    arguments = parser.parse_args()
    fit() if arguments.command == "fit" else select()


if __name__ == "__main__":
    main()
