#!/usr/bin/env python3
"""Sealed multi-regime analysis and proposed immutable terminal mapping."""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from study_common import (
    ADAPTER_FLOPS,
    ATTEMPT_ROOT,
    BASE_FLOPS,
    BOOTSTRAP_REPLICATES,
    CO_PRIMARY_ENDPOINTS,
    FAMILYWISE_ALPHA,
    GATE_FEATURE_FLOPS,
    GATE_HEAD_FLOPS,
    GATE_NONFLOP_OPS,
    GATE_TOTAL_FLOPS,
    PER_CLAIM_ALPHA,
    REGIMES,
    REPO_ROOT,
    ROWS_PER_EPISODE,
    TARGET_EPISODES_PER_REGIME,
    V1_FLOPS,
    V5_ROOT,
    append_ledger,
    assert_runtime_contract,
    atomic_json,
    atomic_npz,
    complete_state,
    execution_manifest_path,
    raw_manifest_path,
    read_json,
    relative_to_repo,
    sequential_calls,
    sha256_file,
    update_state_fields,
    verify_pre_outcome_seal,
)


CONTRASTS = (
    "raw_vs_analytic",
    "fixed_whitened_vs_analytic",
    "raw_vs_seeded",
    "fixed_whitened_vs_seeded",
    "raw_vs_fixed_d1",
    "fixed_whitened_vs_fixed_d1",
    "raw_vs_within_episode_histogram",
    "fixed_whitened_vs_within_episode_histogram",
)
FEATURE_BLOCKS = {
    "history_latents": (0, 576),
    "normalized_actions": (576, 651),
    "current_prediction": (651, 843),
    "last_update": (843, 1035),
    "scalar_summaries": (1035, 1046),
}
BOOTSTRAP_CHUNK = 250


def strongest_mixture(losses: np.ndarray, mean_calls: float) -> dict[str, Any]:
    best: tuple[float, np.ndarray, int, int, float] | None = None
    for lower in range(1, 5):
        for upper in range(lower, 5):
            if not lower <= mean_calls <= upper:
                continue
            weight = (
                0.0
                if lower == upper
                else (mean_calls - lower) / (upper - lower)
            )
            vector = (
                (1.0 - weight) * losses[:, lower - 1]
                + weight * losses[:, upper - 1]
            )
            candidate = (
                float(vector.mean()),
                vector,
                lower,
                upper,
                float(weight),
            )
            if best is None or candidate[0] < best[0] or (
                candidate[0] == best[0]
                and candidate[2:4] < best[2:4]
            ):
                best = candidate
    if best is None:
        raise RuntimeError(
            f"no exact-compute analytic mixture for mean calls {mean_calls}"
        )
    return {
        "mean_loss": best[0],
        "loss": best[1],
        "depth_lower": best[2],
        "depth_upper": best[3],
        "weight_upper": best[4],
    }


def episode_mean(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    expected = TARGET_EPISODES_PER_REGIME * ROWS_PER_EPISODE
    if vector.shape != (expected,):
        raise RuntimeError(f"episode aggregation shape drift: {vector.shape}")
    return vector.reshape(TARGET_EPISODES_PER_REGIME, ROWS_PER_EPISODE).mean(
        axis=1
    )


def load_execution(regime: str) -> dict[str, np.ndarray]:
    manifest = read_json(execution_manifest_path("target", regime))
    names = (
        "target",
        "dense_exits",
        "sparse_selected",
        "calls",
        "scores",
        "features",
        "model_step",
        "slot",
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for expected_slot, record in enumerate(manifest["episodes"]):
        if int(record["slot"]) != expected_slot:
            raise RuntimeError("execution manifest slot order drift")
        path = REPO_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"execution part hash drift: {path}")
        with np.load(path, allow_pickle=False) as stored:
            for name in names:
                parts[name].append(stored[name].copy())
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    expected_rows = TARGET_EPISODES_PER_REGIME * ROWS_PER_EPISODE
    if any(len(value) != expected_rows for value in arrays.values()):
        raise RuntimeError(f"execution row count drift for {regime}")
    if not np.array_equal(
        arrays["slot"],
        np.repeat(
            np.arange(TARGET_EPISODES_PER_REGIME, dtype=np.int32),
            ROWS_PER_EPISODE,
        ),
    ):
        raise RuntimeError("execution slot array drift")
    return arrays


def target_input_seal() -> dict[str, Any]:
    output = ATTEMPT_ROOT / "audit/target_input_seal.json"
    manifests: dict[str, Any] = {}
    raw_hashes: dict[str, str] = {}
    execution_hashes: dict[str, str] = {}
    total_raw_bytes = 0
    total_execution_bytes = 0
    chronology = True
    for regime in REGIMES:
        raw_path = raw_manifest_path("target", regime)
        execution_path = execution_manifest_path("target", regime)
        raw = read_json(raw_path)
        execution = read_json(execution_path)
        if (
            raw["episode_count"] != TARGET_EPISODES_PER_REGIME
            or execution["episode_count"] != TARGET_EPISODES_PER_REGIME
        ):
            raise RuntimeError("input seal cohort count drift")
        chronology &= raw["created_unix_ns"] < execution["created_unix_ns"]
        manifests[regime] = {
            "raw_manifest": {
                "path": relative_to_repo(raw_path),
                "sha256": sha256_file(raw_path),
            },
            "execution_manifest": {
                "path": relative_to_repo(execution_path),
                "sha256": sha256_file(execution_path),
            },
        }
        for record in raw["episodes"]:
            path = REPO_ROOT / record["path"]
            observed = sha256_file(path)
            if observed != record["sha256"]:
                raise RuntimeError(f"raw target hash drift: {path}")
            raw_hashes[record["path"]] = observed
            total_raw_bytes += path.stat().st_size
        for record in execution["episodes"]:
            path = REPO_ROOT / record["path"]
            observed = sha256_file(path)
            if observed != record["sha256"]:
                raise RuntimeError(f"execution target hash drift: {path}")
            execution_hashes[record["path"]] = observed
            total_execution_bytes += path.stat().st_size
    payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "manifests": manifests,
        "raw_episode_hashes": raw_hashes,
        "execution_part_hashes": execution_hashes,
        "exact_regime_count": len(REGIMES),
        "exact_episode_count": len(raw_hashes),
        "exact_execution_part_count": len(execution_hashes),
        "exact_row_count": len(execution_hashes) * ROWS_PER_EPISODE,
        "total_raw_bytes": total_raw_bytes,
        "total_execution_bytes": total_execution_bytes,
        "all_9000_raw_and_execution_traces_complete_before_target_array_open": (
            len(raw_hashes) == 9_000 and len(execution_hashes) == 9_000
        ),
        "raw_manifests_precede_execution_manifests": chronology,
        "target_arrays_opened_before_seal": False,
        "contact_motion_phase_reward_success_opened": False,
    }
    if output.exists():
        prior = read_json(output)
        stable_keys = (
            "manifests",
            "raw_episode_hashes",
            "execution_part_hashes",
            "exact_episode_count",
            "exact_execution_part_count",
            "exact_row_count",
        )
        if any(prior[key] != payload[key] for key in stable_keys):
            raise RuntimeError("existing target input seal mismatch")
        return prior
    atomic_json(output, payload, exclusive=True)
    append_ledger(
        "target_inputs_sealed_before_analysis",
        path=relative_to_repo(output),
        sha256=sha256_file(output),
        episodes=9_000,
        rows=342_000,
    )
    return payload


def seeded_control(
    losses: np.ndarray,
    integer_total_calls: int,
    seed: int,
) -> dict[str, Any]:
    rows = len(losses)
    template = strongest_mixture(losses, integer_total_calls / rows)
    lower = int(template["depth_lower"])
    upper = int(template["depth_upper"])
    if lower == upper:
        number_upper = 0
    else:
        number_upper = math.ceil(
            (integer_total_calls - lower * rows) / (upper - lower)
        )
    calls = np.full(rows, lower, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    calls[rng.permutation(rows)[:number_upper]] = upper
    selected = losses[np.arange(rows), calls - 1]
    return {
        "loss": selected,
        "calls": calls,
        "depth_lower": lower,
        "depth_upper": upper,
        "number_upper": number_upper,
        "seed": int(seed),
    }


def histogram_control(
    calls: np.ndarray,
    episode_slots: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    randomized = np.empty_like(calls)
    records = []
    rng = np.random.default_rng(int(seed))
    for episode in range(TARGET_EPISODES_PER_REGIME):
        indices = np.flatnonzero(episode_slots == episode)
        if len(indices) != ROWS_PER_EPISODE:
            raise RuntimeError("histogram control found non-38-row episode")
        randomized[indices] = calls[indices][rng.permutation(len(indices))]
        original = np.bincount(calls[indices], minlength=5)[1:]
        shuffled = np.bincount(randomized[indices], minlength=5)[1:]
        records.append(
            {
                "episode": episode,
                "original": original.tolist(),
                "randomized": shuffled.tolist(),
                "preserved": bool(np.array_equal(original, shuffled)),
            }
        )
    return randomized, records


def rank_and_calibration(
    scores: np.ndarray,
    raw_losses: np.ndarray,
    white_losses: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stages = []
    calibration = []
    for stage in range(3):
        reached = np.isfinite(scores[:, stage])
        raw_gain = raw_losses[:, stage] - raw_losses[:, stage + 1]
        white_gain = white_losses[:, stage] - white_losses[:, stage + 1]
        combined = 0.5 * (
            raw_gain / (raw_gain.std() + 1e-12)
            + white_gain / (white_gain.std() + 1e-12)
        )
        statistic = spearmanr(
            scores[reached, stage], combined[reached]
        ).statistic
        stages.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "spearman_score_gain_rho": float(statistic),
                "positive_sign": bool(
                    np.isfinite(statistic) and statistic > 0
                ),
            }
        )
        order = np.argsort(scores[reached, stage], kind="mergesort")
        reached_indices = np.flatnonzero(reached)[order]
        bins = []
        for bin_index, indices in enumerate(
            np.array_split(reached_indices, 10)
        ):
            bins.append(
                {
                    "bin": bin_index + 1,
                    "rows": len(indices),
                    "score_mean": float(scores[indices, stage].mean()),
                    "combined_gain_mean": float(combined[indices].mean()),
                    "raw_gain_mean": float(raw_gain[indices].mean()),
                    "fixed_whitened_gain_mean": float(
                        white_gain[indices].mean()
                    ),
                }
            )
        calibration.append(
            {
                "stage": stage + 1,
                "equal_count_score_bins": bins,
                "rank_monotonicity": float(statistic),
                "continuation_rate_among_reached": None,
            }
        )
    return stages, calibration


def distribution_shift(
    regime: str,
    arrays: dict[str, np.ndarray],
    raw_losses: np.ndarray,
    white_losses: np.ndarray,
    v5: dict[str, np.ndarray],
    v5_raw_losses: np.ndarray,
    v5_white_losses: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    calls = arrays["calls"].astype(np.int64)
    reference_calls = v5["calls"].astype(np.int64)
    current_hist = np.bincount(calls, minlength=5)[1:].astype(np.float64)
    reference_hist = np.bincount(
        reference_calls, minlength=5
    )[1:].astype(np.float64)
    current_probability = current_hist / current_hist.sum()
    reference_probability = reference_hist / reference_hist.sum()
    midpoint = 0.5 * (current_probability + reference_probability)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        selected = left > 0
        return float(np.sum(left[selected] * np.log(left[selected] / right[selected])))

    stages = []
    for stage in range(3):
        current_reached = np.isfinite(arrays["scores"][:, stage])
        reference_reached = np.isfinite(v5["scores"][:, stage])
        current_score = arrays["scores"][current_reached, stage].astype(
            np.float64
        )
        reference_score = v5["scores"][
            reference_reached, stage
        ].astype(np.float64)
        current_features = arrays["features"][current_reached, stage]
        reference_features = v5["features"][reference_reached, stage]
        mean_current = current_features.mean(axis=0, dtype=np.float64)
        mean_reference = reference_features.mean(axis=0, dtype=np.float64)
        var_current = current_features.var(axis=0, dtype=np.float64)
        var_reference = reference_features.var(axis=0, dtype=np.float64)
        standardized = (mean_current - mean_reference) / np.sqrt(
            0.5 * (var_current + var_reference) + 1e-12
        )
        top = np.argsort(np.abs(standardized))[-10:][::-1]
        blocks = {}
        for block, (start, stop) in FEATURE_BLOCKS.items():
            values = standardized[start:stop]
            blocks[block] = {
                "width": stop - start,
                "rms_standardized_mean_shift": float(
                    np.sqrt(np.mean(np.square(values)))
                ),
                "maximum_absolute_standardized_mean_shift": float(
                    np.max(np.abs(values))
                ),
            }
        raw_gain = raw_losses[:, stage] - raw_losses[:, stage + 1]
        white_gain = white_losses[:, stage] - white_losses[:, stage + 1]
        ref_raw_gain = (
            v5_raw_losses[:, stage] - v5_raw_losses[:, stage + 1]
        )
        ref_white_gain = (
            v5_white_losses[:, stage] - v5_white_losses[:, stage + 1]
        )
        stages.append(
            {
                "stage": stage + 1,
                "reached_rows": int(current_reached.sum()),
                "v5_reached_rows": int(reference_reached.sum()),
                "score": {
                    "mean": float(current_score.mean()),
                    "std": float(current_score.std()),
                    "quantiles": np.quantile(
                        current_score, [0.05, 0.25, 0.5, 0.75, 0.95]
                    ).tolist(),
                    "v5_mean": float(reference_score.mean()),
                    "v5_std": float(reference_score.std()),
                    "v5_quantiles": np.quantile(
                        reference_score, [0.05, 0.25, 0.5, 0.75, 0.95]
                    ).tolist(),
                    "standardized_mean_shift": float(
                        (current_score.mean() - reference_score.mean())
                        / math.sqrt(
                            0.5
                            * (
                                current_score.var()
                                + reference_score.var()
                            )
                            + 1e-12
                        )
                    ),
                },
                "feature_blocks": blocks,
                "top_absolute_feature_shifts": [
                    {
                        "index": int(index),
                        "name": names[int(index)],
                        "standardized_mean_shift": float(
                            standardized[int(index)]
                        ),
                    }
                    for index in top
                ],
                "solver_gain": {
                    "raw_mean": float(raw_gain.mean()),
                    "v5_raw_mean": float(ref_raw_gain.mean()),
                    "raw_mean_shift": float(
                        raw_gain.mean() - ref_raw_gain.mean()
                    ),
                    "fixed_whitened_mean": float(white_gain.mean()),
                    "v5_fixed_whitened_mean": float(
                        ref_white_gain.mean()
                    ),
                    "fixed_whitened_mean_shift": float(
                        white_gain.mean() - ref_white_gain.mean()
                    ),
                },
            }
        )
    return {
        "regime": regime,
        "call_depth": {
            "histogram": current_hist.astype(int).tolist(),
            "probability": current_probability.tolist(),
            "mean_calls": float(calls.mean()),
            "v5_histogram": reference_hist.astype(int).tolist(),
            "v5_probability": reference_probability.tolist(),
            "v5_mean_calls": float(reference_calls.mean()),
            "mean_call_shift": float(
                calls.mean() - reference_calls.mean()
            ),
            "total_variation": float(
                0.5
                * np.abs(
                    current_probability - reference_probability
                ).sum()
            ),
            "jensen_shannon_nats": float(
                0.5 * kl(current_probability, midpoint)
                + 0.5 * kl(reference_probability, midpoint)
            ),
        },
        "stages": stages,
    }


def compute_regime(
    regime: str,
    arrays: dict[str, np.ndarray],
    whitening: np.ndarray,
    thresholds: np.ndarray,
    seed_ledger: dict[str, Any],
    v5_reference: dict[str, np.ndarray],
    v5_raw_losses: np.ndarray,
    v5_white_losses: np.ndarray,
    feature_names: list[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target = arrays["target"].astype(np.float64)
    dense = arrays["dense_exits"].astype(np.float64)
    sparse = arrays["sparse_selected"].astype(np.float64)
    calls = arrays["calls"].astype(np.int64)
    scores = arrays["scores"].astype(np.float64)
    slots = arrays["slot"].astype(np.int64)
    rows = len(calls)
    positions = np.arange(rows)
    raw_losses = np.mean(
        np.square(dense - target[:, None, :]), axis=2, dtype=np.float64
    )
    white_difference = np.einsum(
        "nkd,df->nkf",
        dense - target[:, None, :],
        whitening,
        optimize=True,
    )
    white_losses = np.mean(
        np.square(white_difference), axis=2, dtype=np.float64
    )
    adaptive_raw = np.mean(
        np.square(sparse - target), axis=1, dtype=np.float64
    )
    adaptive_white = np.mean(
        np.square(
            np.einsum(
                "nd,df->nf", sparse - target, whitening, optimize=True
            )
        ),
        axis=1,
        dtype=np.float64,
    )
    gate_evaluations = int(np.minimum(calls, 3).sum())
    refiner_calls = int(calls.sum())
    common_flops = rows * (BASE_FLOPS + V1_FLOPS)
    adaptive_total_flops = int(
        common_flops
        + (refiner_calls - rows) * ADAPTER_FLOPS
        + gate_evaluations * GATE_TOTAL_FLOPS
    )
    equivalent_total_calls = (
        refiner_calls
        + gate_evaluations * GATE_TOTAL_FLOPS / ADAPTER_FLOPS
    )
    equivalent_mean_calls = equivalent_total_calls / rows
    raw_analytic = strongest_mixture(raw_losses, equivalent_mean_calls)
    white_analytic = strongest_mixture(
        white_losses, equivalent_mean_calls
    )
    integer_total_calls = math.ceil(equivalent_total_calls)
    seeds = seed_ledger["analysis_seeds"]
    raw_seeded = seeded_control(
        raw_losses,
        integer_total_calls,
        seeds["seeded_comparator_seeds"][regime]["raw_vs_analytic"],
    )
    white_seeded = seeded_control(
        white_losses,
        integer_total_calls,
        seeds["seeded_comparator_seeds"][regime][
            "fixed_whitened_vs_analytic"
        ],
    )
    histogram_calls, histogram_records = histogram_control(
        calls, slots, seeds["histogram_seeds"][regime]
    )
    histogram_raw = raw_losses[positions, histogram_calls - 1]
    histogram_white = white_losses[positions, histogram_calls - 1]
    contrasts = {
        "raw_vs_analytic": episode_mean(
            raw_analytic["loss"] - adaptive_raw
        ),
        "fixed_whitened_vs_analytic": episode_mean(
            white_analytic["loss"] - adaptive_white
        ),
        "raw_vs_seeded": episode_mean(
            raw_seeded["loss"] - adaptive_raw
        ),
        "fixed_whitened_vs_seeded": episode_mean(
            white_seeded["loss"] - adaptive_white
        ),
        "raw_vs_fixed_d1": episode_mean(
            raw_losses[:, 0] - adaptive_raw
        ),
        "fixed_whitened_vs_fixed_d1": episode_mean(
            white_losses[:, 0] - adaptive_white
        ),
        "raw_vs_within_episode_histogram": episode_mean(
            histogram_raw - adaptive_raw
        ),
        "fixed_whitened_vs_within_episode_histogram": episode_mean(
            histogram_white - adaptive_white
        ),
    }
    stagewise, calibration = rank_and_calibration(
        scores, raw_losses, white_losses
    )
    for stage, item in enumerate(calibration):
        reached = np.isfinite(scores[:, stage])
        item["continuation_rate_among_reached"] = float(
            np.mean(calls[reached] > stage + 1)
        )
    reconstructed = sequential_calls(scores, thresholds)
    execution_manifest = read_json(
        execution_manifest_path("target", regime)
    )
    raw_seeded_total_flops = int(
        common_flops
        + (int(raw_seeded["calls"].sum()) - rows) * ADAPTER_FLOPS
    )
    white_seeded_total_flops = int(
        common_flops
        + (int(white_seeded["calls"].sum()) - rows) * ADAPTER_FLOPS
    )
    integrity = {
        "exact_rows": rows
        == TARGET_EPISODES_PER_REGIME * ROWS_PER_EPISODE,
        "slot_order": bool(
            np.array_equal(
                slots,
                np.repeat(
                    np.arange(
                        TARGET_EPISODES_PER_REGIME, dtype=np.int64
                    ),
                    ROWS_PER_EPISODE,
                ),
            )
        ),
        "finite_target_dense_sparse_losses": bool(
            all(
                np.isfinite(value).all()
                for value in (
                    target,
                    dense,
                    sparse,
                    raw_losses,
                    white_losses,
                    adaptive_raw,
                    adaptive_white,
                )
            )
        ),
        "calls_range": bool(np.isin(calls, [1, 2, 3, 4]).all()),
        "calls_reproduced_from_scores": bool(
            np.array_equal(reconstructed, calls)
        ),
        "execution_equivalence": execution_manifest[
            "all_equivalence_checks_passed"
        ],
        "modules_frozen": execution_manifest["module_before"]
        == execution_manifest["module_after"]
        and execution_manifest["module_after"]["passed"],
        "no_gradients": execution_manifest["no_gradients"],
        "input_isolation": execution_manifest["loaded_input_keys"]
        == ["action", "pixels"]
        and not execution_manifest["contact_or_privileged_loaded"],
        "histograms_preserved": all(
            item["preserved"] for item in histogram_records
        ),
        "seeded_raw_weakly_more_compute": raw_seeded_total_flops
        >= adaptive_total_flops,
        "seeded_fixed_whitened_weakly_more_compute": (
            white_seeded_total_flops >= adaptive_total_flops
        ),
        "analytic_mean_calls_supported": 1 <= equivalent_mean_calls <= 4,
    }
    compute = {
        "rows": rows,
        "base_model_calls": rows,
        "refiner_model_calls": refiner_calls,
        "mean_refiner_calls": float(calls.mean()),
        "gate_evaluations": gate_evaluations,
        "base_flops": rows * BASE_FLOPS,
        "mandatory_depth1_flops": rows * V1_FLOPS,
        "additional_refiner_flops": (refiner_calls - rows)
        * ADAPTER_FLOPS,
        "gate_feature_flops": gate_evaluations * GATE_FEATURE_FLOPS,
        "gate_dual_affine_head_flops": gate_evaluations
        * GATE_HEAD_FLOPS,
        "gate_total_flops": gate_evaluations * GATE_TOTAL_FLOPS,
        "nonflop_operations": gate_evaluations * GATE_NONFLOP_OPS,
        "adaptive_total_flops": adaptive_total_flops,
        "analytic_equivalent_total_calls": float(
            equivalent_total_calls
        ),
        "analytic_equivalent_mean_calls": float(
            equivalent_mean_calls
        ),
        "raw_analytic_mixture": {
            key: value
            for key, value in raw_analytic.items()
            if key != "loss"
        },
        "fixed_whitened_analytic_mixture": {
            key: value
            for key, value in white_analytic.items()
            if key != "loss"
        },
        "seeded_raw": {
            key: value
            for key, value in raw_seeded.items()
            if key not in ("loss", "calls")
        }
        | {
            "total_calls": int(raw_seeded["calls"].sum()),
            "total_flops": raw_seeded_total_flops,
            "baseline_minus_adaptive_flops": (
                raw_seeded_total_flops - adaptive_total_flops
            ),
        },
        "seeded_fixed_whitened": {
            key: value
            for key, value in white_seeded.items()
            if key not in ("loss", "calls")
        }
        | {
            "total_calls": int(white_seeded["calls"].sum()),
            "total_flops": white_seeded_total_flops,
            "baseline_minus_adaptive_flops": (
                white_seeded_total_flops - adaptive_total_flops
            ),
        },
        "call_histogram": np.bincount(calls, minlength=5)[1:].tolist(),
    }
    shift = distribution_shift(
        regime,
        arrays,
        raw_losses,
        white_losses,
        v5_reference,
        v5_raw_losses,
        v5_white_losses,
        feature_names,
    )
    summary = {
        "adaptive_raw_mse": float(adaptive_raw.mean()),
        "adaptive_fixed_whitened_mse": float(adaptive_white.mean()),
        "contrast_point_estimates": {
            name: float(value.mean()) for name, value in contrasts.items()
        },
        "stagewise_rank": stagewise,
        "routing_calibration": calibration,
        "distribution_shift_vs_v5": shift,
        "compute": compute,
        "integrity": integrity,
        "process_valid_prebootstrap": all(integrity.values()),
    }
    episode_metrics = {
        "episode": np.arange(
            TARGET_EPISODES_PER_REGIME, dtype=np.int32
        ),
        "adaptive_raw": episode_mean(adaptive_raw),
        "adaptive_fixed_whitened": episode_mean(adaptive_white),
        **contrasts,
    }
    return summary, episode_metrics


def bootstrap_all(
    metrics: dict[str, dict[str, np.ndarray]], seed: int
) -> dict[str, dict[str, np.ndarray]]:
    output = {
        regime: {
            name: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
            for name in CONTRASTS
        }
        for regime in REGIMES
    }
    rng = np.random.default_rng(int(seed))
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_REPLICATES)
        size = stop - start
        for regime in REGIMES:
            sample = rng.integers(
                0,
                TARGET_EPISODES_PER_REGIME,
                size=(size, TARGET_EPISODES_PER_REGIME),
                dtype=np.int32,
            )
            for name in CONTRASTS:
                output[regime][name][start:stop] = metrics[regime][name][
                    sample
                ].mean(axis=1)
    return output


def terminal_mapping(simultaneous: dict[str, Any]) -> tuple[str, int]:
    supported = sum(
        item["lower"] > 0
        for regime in REGIMES
        for item in simultaneous[regime].values()
    )
    if supported == 6:
        return "zero_shot_generalization_supported", supported
    if supported > 0:
        return "zero_shot_generalization_partial", supported
    return "zero_shot_generalization_failed", supported


def main() -> None:
    assert_runtime_contract("evaluation")
    output = ATTEMPT_ROOT / "analysis_result.json"
    if output.exists():
        complete_state(
            "SEALED_ANALYSIS",
            "LATENCY_AND_RESOURCE_REPORTING",
            evidence_path=output,
            checkpoint_name="v005_sealed_analysis_reused",
            next_action="measure synchronized latency and report resources",
            extra_fields={"target_outcomes_opened_for_analysis": True},
        )
        print(json.dumps(read_json(output), sort_keys=True))
        return
    verify_pre_outcome_seal()
    sealed_inputs = target_input_seal()
    if not (
        sealed_inputs[
            "all_9000_raw_and_execution_traces_complete_before_target_array_open"
        ]
        and sealed_inputs["raw_manifests_precede_execution_manifests"]
    ):
        raise RuntimeError("target input seal is not complete")
    update_state_fields(target_outcomes_opened_for_analysis=True)

    seed_ledger = read_json(ATTEMPT_ROOT / "cohort_seed_ledger.json")
    with np.load(
        V5_ROOT / "freeze/whitening.npz", allow_pickle=False
    ) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    with np.load(
        V5_ROOT / "freeze/compiled_gate.npz", allow_pickle=False
    ) as stored:
        thresholds = stored["thresholds"].astype(np.float64)
    with np.load(
        V5_ROOT / "data/v5_confirmation_execution.npz",
        allow_pickle=False,
    ) as stored:
        v5_reference = {
            name: stored[name].copy()
            for name in (
                "target",
                "dense_exits",
                "calls",
                "scores",
                "features",
            )
        }
    v5_target = v5_reference["target"].astype(np.float64)
    v5_dense = v5_reference["dense_exits"].astype(np.float64)
    v5_raw_losses = np.mean(
        np.square(v5_dense - v5_target[:, None, :]),
        axis=2,
        dtype=np.float64,
    )
    v5_white_losses = np.mean(
        np.square(
            np.einsum(
                "nkd,df->nkf",
                v5_dense - v5_target[:, None, :],
                whitening,
                optimize=True,
            )
        ),
        axis=2,
        dtype=np.float64,
    )
    del v5_target, v5_dense
    feature_names = read_json(
        V5_ROOT / "semantic_feature_names.json"
    )["ordered_names"]

    summaries: dict[str, Any] = {}
    episode_metrics: dict[str, dict[str, np.ndarray]] = {}
    for regime in REGIMES:
        arrays = load_execution(regime)
        summary, metrics = compute_regime(
            regime,
            arrays,
            whitening,
            thresholds,
            seed_ledger,
            v5_reference,
            v5_raw_losses,
            v5_white_losses,
            feature_names,
        )
        summaries[regime] = summary
        episode_metrics[regime] = metrics
        metric_path = ATTEMPT_ROOT / f"metrics/{regime}_episode_metrics.npz"
        atomic_npz(metric_path, metrics, exclusive=True)
        del arrays
        gc.collect()
        print(f"analyzed regime {regime}", flush=True)

    bootstrap = bootstrap_all(
        episode_metrics,
        seed_ledger["analysis_seeds"]["joint_bootstrap_seed"],
    )
    individual: dict[str, Any] = {}
    simultaneous: dict[str, Any] = {}
    bootstrap_arrays = {}
    for regime in REGIMES:
        individual[regime] = {}
        simultaneous[regime] = {}
        for name in CONTRASTS:
            values = episode_metrics[regime][name]
            replicates = bootstrap[regime][name]
            individual[regime][name] = {
                "estimate": float(values.mean()),
                "lower": float(np.quantile(replicates, 0.025)),
                "upper": float(np.quantile(replicates, 0.975)),
            }
            bootstrap_arrays[f"{regime}__{name}"] = replicates
        for name in CO_PRIMARY_ENDPOINTS:
            simultaneous[regime][name] = {
                "estimate": float(episode_metrics[regime][name].mean()),
                "lower": float(
                    np.quantile(
                        bootstrap[regime][name], PER_CLAIM_ALPHA
                    )
                ),
                "familywise_alpha": FAMILYWISE_ALPHA,
                "family_size": 6,
                "per_claim_alpha": PER_CLAIM_ALPHA,
                "method": (
                    "bonferroni_six_claim_one_sided_percentile"
                ),
            }
    atomic_npz(
        ATTEMPT_ROOT / "metrics/bootstrap_replicates.npz",
        bootstrap_arrays,
        exclusive=True,
    )

    heterogeneity = {"pairwise": {}, "range": {}}
    regime_names = list(REGIMES)
    for endpoint in CO_PRIMARY_ENDPOINTS:
        estimates = {
            regime: individual[regime][endpoint]["estimate"]
            for regime in REGIMES
        }
        heterogeneity["range"][endpoint] = {
            "minimum_regime": min(estimates, key=estimates.get),
            "maximum_regime": max(estimates, key=estimates.get),
            "max_minus_min": max(estimates.values())
            - min(estimates.values()),
        }
        for left_index, left in enumerate(regime_names):
            for right in regime_names[left_index + 1 :]:
                difference = (
                    bootstrap[left][endpoint]
                    - bootstrap[right][endpoint]
                )
                key = f"{left}_minus_{right}"
                heterogeneity["pairwise"].setdefault(key, {})[
                    endpoint
                ] = {
                    "estimate": estimates[left] - estimates[right],
                    "lower": float(np.quantile(difference, 0.025)),
                    "upper": float(np.quantile(difference, 0.975)),
                    "descriptive_not_terminal": True,
                }

    proposed, supported_claims = terminal_mapping(simultaneous)
    regime_labels = {}
    for regime in REGIMES:
        count = sum(
            item["lower"] > 0
            for item in simultaneous[regime].values()
        )
        regime_labels[regime] = (
            "supported" if count == 2 else ("mixed" if count == 1 else "failed")
        )
    global_integrity = {
        "pre_outcome_seal": True,
        "input_seal_complete": sealed_inputs[
            "all_9000_raw_and_execution_traces_complete_before_target_array_open"
        ],
        "chronology": sealed_inputs[
            "raw_manifests_precede_execution_manifests"
        ],
        "exact_episode_count": sealed_inputs["exact_episode_count"] == 9_000,
        "exact_execution_count": sealed_inputs[
            "exact_execution_part_count"
        ]
        == 9_000,
        "exact_row_count": sealed_inputs["exact_row_count"] == 342_000,
        "all_regime_integrity": all(
            all(summary["integrity"].values())
            for summary in summaries.values()
        ),
        "seed_isolation": read_json(
            ATTEMPT_ROOT / "cohort_seed_ledger.json"
        )["checks"]["zero_safe_prior_numeric_overlap"]
        and read_json(ATTEMPT_ROOT / "cohort_seed_ledger.json")[
            "checks"
        ]["zero_v5_v001_v004_overlap"],
        "bootstrap_replicates_exact": all(
            len(values) == BOOTSTRAP_REPLICATES
            for by_regime in bootstrap.values()
            for values in by_regime.values()
        ),
        "contact_motion_phase_reward_success_excluded": True,
        "forbidden_v3_hdf5_excluded": True,
    }
    process_valid = all(global_integrity.values())
    if not process_valid:
        proposed = "generalization_execution_invalid"
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "proposed_terminal_outcome": proposed,
        "independent_verification_required": True,
        "process_valid": process_valid,
        "supported_co_primary_claim_count": supported_claims,
        "regime_labels": regime_labels,
        "simultaneous_co_primary": simultaneous,
        "individual_intervals": individual,
        "regimes": summaries,
        "heterogeneity": heterogeneity,
        "integrity": global_integrity,
        "bootstrap": {
            "unit": "episode",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": seed_ledger["analysis_seeds"]["joint_bootstrap_seed"],
            "chunk_size_defining_rng_consumption": BOOTSTRAP_CHUNK,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "family_size": 6,
            "per_claim_alpha": PER_CLAIM_ALPHA,
        },
        "latency_excluded_from_terminal_mapping": True,
        "posthoc_fields_opened": False,
        "target_outcome_episodes": 9_000,
        "target_rows": 342_000,
    }
    atomic_json(
        ATTEMPT_ROOT / "metrics/bootstrap_summary.json",
        {
            "schema_version": 1,
            "individual": individual,
            "simultaneous_co_primary": simultaneous,
            "heterogeneity": heterogeneity,
            "bootstrap": result["bootstrap"],
        },
        exclusive=True,
    )
    atomic_json(output, result, exclusive=True)
    append_ledger(
        "sealed_analysis_complete",
        path=relative_to_repo(output),
        sha256=sha256_file(output),
        proposed_terminal_outcome=proposed,
        process_valid=process_valid,
        supported_claims=supported_claims,
    )
    complete_state(
        "SEALED_ANALYSIS",
        "LATENCY_AND_RESOURCE_REPORTING",
        evidence_path=output,
        checkpoint_name="v005_sealed_analysis_complete",
        next_action="measure synchronized latency and report resources",
        extra_fields={"target_outcomes_opened_for_analysis": True},
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
