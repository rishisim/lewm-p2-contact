#!/usr/bin/env python3
"""Read-only diagnosis of the permanently consumed cycle 001 cohort."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
SOURCE = REPO_ROOT / "runs/lewm_v5_readiness_program/cycles/cycle_001_audit_repair"
OUTPUT = ROOT / "development/cycle_001_consumed_diagnosis.json"


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }


def episode_mean(values: np.ndarray, episode: np.ndarray) -> np.ndarray:
    return np.asarray([values[episode == index].mean() for index in range(300)])


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("consumed-cohort diagnosis is immutable and already exists")
    decision = json.loads((SOURCE / "decision.json").read_text())
    analysis = json.loads((SOURCE / "analysis_result.json").read_text())
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    with np.load(SOURCE / "metrics/prospective_episode_metrics.npz", allow_pickle=False) as stored:
        episode_metrics = {name: stored[name].astype(np.float64) for name in stored.files}
    with np.load(SOURCE / "metrics/bootstrap_replicates.npz", allow_pickle=False) as stored:
        bootstrap = {name: stored[name].astype(np.float64) for name in stored.files}
    with np.load(SOURCE / "data/prospective_execution.npz", allow_pickle=False) as stored:
        target = stored["target"].astype(np.float64)
        exits = stored["dense_exits"].astype(np.float64)
        calls = stored["calls"].astype(np.int64)
        scores = stored["scores"].astype(np.float64)
        episode = stored["episode_id"].astype(np.int64)
    with np.load(
        REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_discovery/freeze/gate_contract.npz",
        allow_pickle=False,
    ) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)

    raw_loss = np.mean((exits - target[:, None, :]) ** 2, axis=2, dtype=np.float64)
    white_error = np.einsum(
        "nkd,df->nkf", exits - target[:, None, :], whitening, optimize=False
    )
    white_loss = np.mean(white_error**2, axis=2, dtype=np.float64)
    threshold = float(protocol["threshold"])
    co_primary = ("raw_vs_analytic", "native_whitened_vs_analytic")

    heterogeneity = {}
    for name in co_primary:
        values = episode_metrics[name]
        heterogeneity[name] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "standard_error": float(values.std(ddof=1) / math.sqrt(len(values))),
            "positive_episode_fraction": float(np.mean(values > 0)),
            "quantiles": quantiles(values),
            "worst_episode_index": int(np.argmin(values)),
            "best_episode_index": int(np.argmax(values)),
        }
    heterogeneity["cross_endpoint_episode_spearman"] = float(
        spearmanr(episode_metrics[co_primary[0]], episode_metrics[co_primary[1]]).statistic
    )

    calibration = []
    stagewise_gains = []
    gate_margins = []
    for stage in range(3):
        reached = np.isfinite(scores[:, stage])
        score = scores[reached, stage]
        raw_gain = raw_loss[reached, stage] - raw_loss[reached, stage + 1]
        white_gain = white_loss[reached, stage] - white_loss[reached, stage + 1]
        combined = 0.5 * (
            raw_gain / (np.std(raw_loss[:, stage] - raw_loss[:, stage + 1]) + 1e-12)
            + white_gain / (np.std(white_loss[:, stage] - white_loss[:, stage + 1]) + 1e-12)
        )
        ranks = rankdata(score, method="average")
        order = np.argsort(ranks, kind="stable")
        bins = np.array_split(order, 5)
        summaries = []
        for index, locations in enumerate(bins):
            summaries.append(
                {
                    "quintile": index + 1,
                    "rows": int(len(locations)),
                    "mean_score": float(score[locations].mean()),
                    "mean_combined_gain": float(combined[locations].mean()),
                    "mean_raw_gain": float(raw_gain[locations].mean()),
                    "mean_native_whitened_gain": float(white_gain[locations].mean()),
                    "positive_combined_gain_fraction": float(np.mean(combined[locations] > 0)),
                }
            )
        continue_decision = score > threshold
        beneficial = combined > 0
        calibration.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "score_gain_spearman": float(spearmanr(score, combined).statistic),
                "continued_fraction": float(continue_decision.mean()),
                "beneficial_fraction": float(beneficial.mean()),
                "continued_precision_for_positive_gain": float(
                    beneficial[continue_decision].mean() if continue_decision.any() else math.nan
                ),
                "positive_gain_recall": float(
                    continue_decision[beneficial].mean() if beneficial.any() else math.nan
                ),
                "score_quintiles": summaries,
            }
        )
        stagewise_gains.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached.sum()),
                "mean_raw_gain_reached": float(raw_gain.mean()),
                "mean_native_whitened_gain_reached": float(white_gain.mean()),
                "positive_raw_gain_fraction": float(np.mean(raw_gain > 0)),
                "positive_native_whitened_gain_fraction": float(np.mean(white_gain > 0)),
                "mean_raw_gain_if_continued": float(raw_gain[continue_decision].mean()),
                "mean_native_whitened_gain_if_continued": float(white_gain[continue_decision].mean()),
            }
        )
        margin = score - threshold
        gate_margins.append(
            {
                "stage": stage + 1,
                "threshold": threshold,
                "margin_quantiles": quantiles(margin),
                "fraction_abs_margin_le_1e-6": float(np.mean(np.abs(margin) <= 1e-6)),
                "fraction_abs_margin_le_1e-4": float(np.mean(np.abs(margin) <= 1e-4)),
                "minimum_absolute_margin": float(np.min(np.abs(margin))),
            }
        )

    rows = len(calls)
    gate_evaluations = int(np.minimum(calls, 3).sum())
    gate_flops = int(analysis["compute"]["gate_total_flops"])
    adapter_flops = 264_960
    compute_price = {
        "rows": rows,
        "call_histogram": np.bincount(calls, minlength=5)[1:].tolist(),
        "mean_actual_refiner_calls": float(calls.mean()),
        "gate_evaluations": gate_evaluations,
        "gate_flops": gate_flops,
        "gate_equivalent_refiner_calls": float(gate_flops / adapter_flops),
        "gate_equivalent_calls_per_row": float(gate_flops / adapter_flops / rows),
        "analytic_exact_total_compute_mean_calls": float(
            analysis["compute"]["analytic_equivalent_mean_calls"]
        ),
        "adaptive_total_flops": int(analysis["compute"]["adaptive_total_flops"]),
        "seeded_comparator_excess_flops": int(
            analysis["compute"]["seeded_mixture"]["baseline_minus_adaptive_flops"]
        ),
        "nonflop_comparison_min_operations": int(
            analysis["compute"]["nonflop_comparison_min_operations"]
        ),
    }

    block_size = 50
    dgp_blocks = []
    episode_call_mean = episode_mean(calls.astype(np.float64), episode)
    for start in range(0, 300, block_size):
        stop = start + block_size
        dgp_blocks.append(
            {
                "episodes": [start, stop - 1],
                "raw_effect_mean": float(episode_metrics[co_primary[0]][start:stop].mean()),
                "native_whitened_effect_mean": float(
                    episode_metrics[co_primary[1]][start:stop].mean()
                ),
                "mean_calls": float(episode_call_mean[start:stop].mean()),
                "adaptive_raw_mse": float(episode_metrics["adaptive_raw"][start:stop].mean()),
                "adaptive_native_whitened_mse": float(
                    episode_metrics["adaptive_native_whitened"][start:stop].mean()
                ),
            }
        )
    episode_index = np.arange(300, dtype=np.float64)
    dgp_stability = {
        "contiguous_50_episode_blocks": dgp_blocks,
        "episode_index_spearman": {
            name: float(spearmanr(episode_index, episode_metrics[name]).statistic)
            for name in co_primary
        },
        "first_half_minus_second_half": {
            name: float(episode_metrics[name][:150].mean() - episode_metrics[name][150:].mean())
            for name in co_primary
        },
        "call_mean_episode_index_spearman": float(
            spearmanr(episode_index, episode_call_mean).statistic
        ),
    }

    estimates = {name: float(episode_metrics[name].mean()) for name in co_primary}
    centered = {
        name: estimates[name] - bootstrap[name]
        for name in co_primary
    }
    unstudentized_winner = np.argmax(
        np.column_stack([centered[name] for name in co_primary]), axis=1
    )
    simultaneous_diagnosis = {
        "cycle_001_method": "unstudentized shared absolute max centered error across dimensionally different endpoints",
        "endpoint_units_are_not_commensurate": True,
        "bootstrap_sd": {name: float(bootstrap[name].std(ddof=1)) for name in co_primary},
        "bootstrap_sd_ratio_native_whitened_over_raw": float(
            bootstrap[co_primary[1]].std(ddof=1) / bootstrap[co_primary[0]].std(ddof=1)
        ),
        "fraction_shared_max_selected_native_whitened_endpoint": float(
            np.mean(unstudentized_winner == 1)
        ),
        "cycle_001_shared_absolute_error_quantile": float(
            analysis["simultaneous_co_primary"][co_primary[0]][
                "shared_centered_max_error_quantile_95"
            ]
        ),
        "bonferroni_familywise_95_lower_bounds_diagnostic_only": {
            name: float(np.quantile(bootstrap[name], 0.025)) for name in co_primary
        },
        "bonferroni_rule": "two fixed co-primary one-sided percentile bounds at alpha/2 = 0.025; union bound gives familywise coverage at least 0.95 without requiring common units or dependence assumptions",
    }

    output = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "source_cycle": "cycle_001_audit_repair",
        "source_terminal_outcome": decision["terminal_outcome"],
        "source_cohort_is_permanently_consumed": True,
        "read_only_source_inputs": True,
        "heterogeneity": heterogeneity,
        "calibration": calibration,
        "compute_price": compute_price,
        "stagewise_gains": stagewise_gains,
        "gate_margins": gate_margins,
        "dgp_stability": dgp_stability,
        "simultaneous_inference_diagnosis": simultaneous_diagnosis,
        "bounded_change": {
            "candidate_count": 1,
            "candidate_policy_changed": False,
            "base_changed": False,
            "refiner_changed": False,
            "gate_changed": False,
            "threshold_changed": False,
            "whitening_changed": False,
            "dgp_changed": False,
            "only_change": "replace the dimensionally invalid shared-absolute-error simultaneous construction with fixed two-endpoint Bonferroni one-sided percentile lower bounds at 0.025 each",
            "familywise_confidence_target": 0.95,
            "individual_criteria_unchanged": True,
            "new_excluded_smoke_episodes": 12,
            "new_prospective_episodes": 300,
            "no_sequential_expansion": True,
            "selection_on_new_cohort": False,
        },
        "v5_outcome_episodes": 0,
    }
    atomic_json(OUTPUT, output)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
