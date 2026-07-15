#!/usr/bin/env python3
"""One-shot validation of the locked stage-threshold candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
DISCOVERY = REPO_ROOT / "runs/lewm_adaptive_compute_planoracle_native_discovery"
PROTOCOL_PATH = ROOT / "development/stage_specific_threshold_protocol.json"
SPLIT_MANIFEST_PATH = ROOT / "development/split_manifest.json"
SELECTION_LEDGER_PATH = ROOT / "development/candidate_selection_ledger.json"
LOCK_PATH = ROOT / "development/selected_stage_thresholds.json"
VALIDATION_PATH = ROOT / "development/validation_consumed.npz"
RESULT_PATH = ROOT / "development/one_shot_validation_result.json"
BOOTSTRAP_PATH = ROOT / "development/one_shot_validation_bootstrap.npz"
VALIDATED_PATH = ROOT / "development/validated_stage_thresholds.json"

BASE_FLOPS = 70_529_190
V1_FLOPS = 669_184
ADAPTER_FLOPS = 264_960
GATE_FLOPS = 7_997
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_006_999_951
HISTOGRAM_SEED = 2_006_888_841
SEEDED_MIXTURE_SEED = 2_006_777_731


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
        raise RuntimeError("no supported analytic mixture")
    return {
        "mean_loss": best[0],
        "loss": best[1],
        "depth_lower": best[2],
        "depth_upper": best[3],
        "weight_upper": best[4],
    }


def episode_mean(values: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    return np.asarray([values[episode_id == item].mean() for item in np.unique(episode_id)])


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    split = json.loads(SPLIT_MANIFEST_PATH.read_text())
    selection_ledger = json.loads(SELECTION_LEDGER_PATH.read_text())
    lock = json.loads(LOCK_PATH.read_text())
    if lock["status"] != "selected_and_locked_before_validation_candidate_metrics":
        raise RuntimeError("selection lock status drift")
    if lock["selection_ledger_sha256"] != sha256(SELECTION_LEDGER_PATH):
        raise RuntimeError("selection ledger hash drift")
    if selection_ledger["selected_candidate_id"] != lock["selected"]["candidate_id"]:
        raise RuntimeError("selected candidate mismatch")
    if lock["validation_input_sha256_declared_in_preselection_split"] != sha256(VALIDATION_PATH):
        raise RuntimeError("validation split hash drift")
    if split["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("protocol hash drift")
    declared = protocol["one_shot_validation"]
    if (
        declared["bootstrap_replicates"] != BOOTSTRAP_REPLICATES
        or declared["bootstrap_seed"] != BOOTSTRAP_SEED
        or declared["histogram_seed"] != HISTOGRAM_SEED
        or declared["seeded_mixture_seed"] != SEEDED_MIXTURE_SEED
    ):
        raise RuntimeError("validation seed contract drift")

    with np.load(VALIDATION_PATH, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    with np.load(DISCOVERY / "freeze/gate_contract.npz", allow_pickle=False) as stored:
        whitening = stored["whitening_matrix"].astype(np.float64)
    episode_id = arrays["episode_id"].astype(np.int64)
    if len(np.unique(episode_id)) != 150 or np.any(episode_id % 2 != 1):
        raise RuntimeError("validation role drift")

    thresholds = np.asarray(lock["selected"]["thresholds"], dtype=np.float64)
    target = arrays["target"].astype(np.float64)
    exits = arrays["dense_exits"].astype(np.float64)
    scores = arrays["scores"].astype(np.float64)
    old_calls = arrays["calls"].astype(np.int64)
    calls = np.ones(len(scores), dtype=np.int64)
    active = np.ones(len(scores), dtype=bool)
    reached = []
    finite_required_scores = True
    for stage in range(3):
        reached.append(active.copy())
        finite_required_scores &= bool(np.isfinite(scores[active, stage]).all())
        active &= scores[:, stage] > thresholds[stage]
        calls += active.astype(np.int64)
    counterfactual_complete = finite_required_scores and bool(np.all(calls <= old_calls))

    row_index = np.arange(len(calls))
    difference = exits - target[:, None, :]
    raw_losses = np.square(difference).mean(axis=2)
    white_difference = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    white_losses = np.square(white_difference).mean(axis=2)
    adaptive_raw = raw_losses[row_index, calls - 1]
    adaptive_white = white_losses[row_index, calls - 1]

    gate_evaluations = int(np.minimum(calls, 3).sum())
    refiner_calls = int(calls.sum())
    common_flops = len(calls) * (BASE_FLOPS + V1_FLOPS)
    adaptive_total_flops = int(
        common_flops
        + (refiner_calls - len(calls)) * ADAPTER_FLOPS
        + gate_evaluations * GATE_FLOPS
    )
    analytic_total_calls = refiner_calls + gate_evaluations * GATE_FLOPS / ADAPTER_FLOPS
    analytic_mean_calls = analytic_total_calls / len(calls)
    raw_analytic = strongest_mixture(raw_losses, analytic_mean_calls)
    white_analytic = strongest_mixture(white_losses, analytic_mean_calls)

    integer_target_calls = math.ceil(analytic_total_calls)
    integer_mean_calls = integer_target_calls / len(calls)
    seeded_shape = strongest_mixture(raw_losses, integer_mean_calls)
    lower = int(seeded_shape["depth_lower"])
    upper = int(seeded_shape["depth_upper"])
    number_upper = (
        0
        if lower == upper
        else math.ceil((integer_target_calls - lower * len(calls)) / (upper - lower))
    )
    seeded_calls = np.full(len(calls), lower, dtype=np.int64)
    seeded_rng = np.random.default_rng(SEEDED_MIXTURE_SEED)
    seeded_calls[seeded_rng.permutation(len(calls))[:number_upper]] = upper
    seeded_raw = raw_losses[row_index, seeded_calls - 1]
    seeded_total_flops = int(
        common_flops + (int(seeded_calls.sum()) - len(calls)) * ADAPTER_FLOPS
    )

    histogram_calls = np.empty_like(calls)
    histogram_rng = np.random.default_rng(HISTOGRAM_SEED)
    histograms_exact = True
    for episode in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == episode)
        histogram_calls[indices] = calls[indices][histogram_rng.permutation(len(indices))]
        histograms_exact &= bool(
            np.array_equal(
                np.bincount(calls[indices], minlength=5),
                np.bincount(histogram_calls[indices], minlength=5),
            )
        )
    histogram_raw = raw_losses[row_index, histogram_calls - 1]
    histogram_white = white_losses[row_index, histogram_calls - 1]

    differences = {
        "raw_vs_analytic": episode_mean(raw_analytic["loss"] - adaptive_raw, episode_id),
        "raw_vs_seeded": episode_mean(seeded_raw - adaptive_raw, episode_id),
        "raw_vs_fixed_d1": episode_mean(raw_losses[:, 0] - adaptive_raw, episode_id),
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
    if any(len(values) != 150 for values in differences.values()):
        raise RuntimeError("validation bootstrap unit drift")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampling = rng.integers(0, 150, size=(BOOTSTRAP_REPLICATES, 150), dtype=np.int32)
    bootstrap = {name: values[sampling].mean(axis=1) for name, values in differences.items()}
    individual = {
        name: {
            "estimate": float(values.mean()),
            "lower": float(np.quantile(bootstrap[name], 0.025)),
            "upper": float(np.quantile(bootstrap[name], 0.975)),
        }
        for name, values in differences.items()
    }
    simultaneous = {
        name: {
            "estimate": float(differences[name].mean()),
            "lower": float(np.quantile(bootstrap[name], 0.025)),
            "method": "bonferroni_two_endpoint_one_sided_percentile",
            "familywise_alpha": 0.05,
            "per_endpoint_alpha": 0.025,
        }
        for name in ("raw_vs_analytic", "native_whitened_vs_analytic")
    }

    stages = []
    for stage in range(3):
        raw_gain = raw_losses[:, stage] - raw_losses[:, stage + 1]
        white_gain = white_losses[:, stage] - white_losses[:, stage + 1]
        combined = 0.5 * (
            raw_gain / (raw_gain.std() + 1e-12)
            + white_gain / (white_gain.std() + 1e-12)
        )
        rho = float(spearmanr(scores[reached[stage], stage], combined[reached[stage]]).statistic)
        stages.append(
            {
                "stage": stage + 1,
                "reached_rows": int(reached[stage].sum()),
                "rho": rho,
                "positive": bool(np.isfinite(rho) and rho > 0),
            }
        )

    histogram = np.bincount(calls, minlength=5)[1:].tolist()
    checks = {
        "exact_150_episode_validation_role": len(np.unique(episode_id)) == 150,
        "counterfactual_complete": counterfactual_complete,
        "stage_2_reached_at_least_400": stages[1]["reached_rows"] >= 400,
        "stage_3_reached_at_least_120": stages[2]["reached_rows"] >= 120,
        "all_four_depths_at_least_20": all(value >= 20 for value in histogram),
        "all_six_individual_lowers_positive": all(item["lower"] > 0 for item in individual.values()),
        "both_simultaneous_lowers_positive": all(item["lower"] > 0 for item in simultaneous.values()),
        "all_three_stage_rhos_positive": all(item["positive"] for item in stages),
        "within_episode_histograms_exact": histograms_exact,
        "seeded_comparator_weakly_more_compute": seeded_total_flops >= adaptive_total_flops,
        "selection_lock_preceded_validation": lock["validation_candidate_metrics_accessed"] is False,
        "zero_v5_outcomes": True,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "status": "one_shot_validation_passed" if passed else "one_shot_validation_failed",
        "selected_candidate_id": lock["selected"]["candidate_id"],
        "thresholds": thresholds.tolist(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "selection_lock_sha256": sha256(LOCK_PATH),
        "validation_input_sha256": sha256(VALIDATION_PATH),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "criteria": individual,
        "simultaneous_co_primary": simultaneous,
        "stagewise_rank": stages,
        "checks": checks,
        "call_histogram": histogram,
        "mean_calls": float(calls.mean()),
        "gate_evaluations": gate_evaluations,
        "adaptive_total_flops": adaptive_total_flops,
        "analytic_equivalent_mean_calls": float(analytic_mean_calls),
        "seeded_total_flops": seeded_total_flops,
        "seeded_minus_adaptive_flops": seeded_total_flops - adaptive_total_flops,
        "candidate_eligible_for_fresh_cycle_005": passed,
        "development_only_not_prospective_evidence": True,
        "v5_outcome_episodes": 0,
    }
    atomic_npz(BOOTSTRAP_PATH, bootstrap)
    atomic_json(RESULT_PATH, result)
    if passed:
        validated = {
            "schema_version": 1,
            "status": "validated_on_disjoint_consumed_development_role",
            "candidate_id": lock["selected"]["candidate_id"],
            "thresholds": thresholds.tolist(),
            "selection_lock_sha256": sha256(LOCK_PATH),
            "validation_result_sha256": sha256(RESULT_PATH),
            "validation_bootstrap_sha256": sha256(BOOTSTRAP_PATH),
            "requires_wholly_fresh_smoke_and_prospective_cohort": True,
            "v5_outcome_episodes": 0,
        }
        atomic_json(VALIDATED_PATH, validated)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
