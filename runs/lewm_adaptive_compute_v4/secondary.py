#!/usr/bin/env python3
"""Post-verdict preregistered physical-regime and matching analyses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

import common


REGIME_NAMES = {0: "impact", 1: "contact", 2: "static", 3: "other_free_motion"}


def _load() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    decision = common.read_json(common.ROOT / "decision.json")
    if not decision.get("primary_verdict_written_before_secondary_physical_regime_analysis"):
        raise RuntimeError("primary verdict was not frozen before secondary analysis")
    outcome_path = common.ROOT / "data/confirmation_outcome_once.npz"
    with np.load(outcome_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    metrics = common.read_json(common.ROOT / "metrics/primary_confirmation.json")
    return arrays, metrics


def _conditional_cluster_ci(
    values: np.ndarray,
    mask: np.ndarray,
    episodes: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    episodes = np.asarray(episodes, dtype=np.int64)
    unique, inverse = np.unique(episodes, return_inverse=True)
    sums = np.bincount(inverse, weights=np.where(mask, values, 0.0), minlength=len(unique))
    counts = np.bincount(inverse, weights=mask.astype(np.float64), minlength=len(unique))
    if counts.sum() == 0:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_samples": samples,
            "seed": seed,
            "rows": 0,
            "episodes_with_rows": 0,
        }
    rng = np.random.default_rng(int(seed))
    reps = np.empty(samples, dtype=np.float64)
    cursor = 0
    while cursor < samples:
        size = min(1000, samples - cursor)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        numerator = sums[selected].sum(1)
        denominator = counts[selected].sum(1)
        reps[cursor : cursor + size] = np.divide(
            numerator,
            denominator,
            out=np.full(size, np.nan),
            where=denominator > 0,
        )
        cursor += size
    finite = reps[np.isfinite(reps)]
    return {
        "estimate": float(values[mask].mean()),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "bootstrap_samples": samples,
        "finite_bootstrap_replicates": len(finite),
        "seed": int(seed),
        "rows": int(mask.sum()),
        "episodes_with_rows": int(np.sum(counts > 0)),
    }


def _unmatched_contrast_ci(
    values: np.ndarray,
    contact: np.ndarray,
    episodes: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    contact = np.asarray(contact, dtype=bool)
    episodes = np.asarray(episodes, dtype=np.int64)
    unique, inverse = np.unique(episodes, return_inverse=True)
    t_sum = np.bincount(inverse, weights=np.where(contact, values, 0.0), minlength=len(unique))
    t_n = np.bincount(inverse, weights=contact, minlength=len(unique))
    c_sum = np.bincount(inverse, weights=np.where(~contact, values, 0.0), minlength=len(unique))
    c_n = np.bincount(inverse, weights=~contact, minlength=len(unique))
    estimate = float(values[contact].mean() - values[~contact].mean())
    rng = np.random.default_rng(int(seed))
    reps = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        treated = t_sum[selected].sum(1) / t_n[selected].sum(1)
        control = c_sum[selected].sum(1) / c_n[selected].sum(1)
        reps[start : start + size] = treated - control
    return {
        "estimate_contact_minus_noncontact": estimate,
        "ci_low": float(np.quantile(reps, 0.025)),
        "ci_high": float(np.quantile(reps, 0.975)),
        "bootstrap_samples": samples,
        "seed": int(seed),
        "contact_rows": int(contact.sum()),
        "noncontact_rows": int((~contact).sum()),
    }


def _strata(values: Mapping[str, np.ndarray], bins: Mapping[str, Any]) -> np.ndarray:
    phase = np.digitize(values["normalized_phase"], bins["phase_interior_edges"])
    motion = np.digitize(values["block_disp"], bins["target_motion_interior_edges"])
    action = np.digitize(values["action_magnitude"], bins["action_magnitude_interior_edges"])
    motion_width = len(bins["target_motion_interior_edges"]) + 1
    action_width = len(bins["action_magnitude_interior_edges"]) + 1
    return (phase * motion_width * action_width + motion * action_width + action).astype(np.int64)


def _matching_weights(strata: np.ndarray, contact: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.zeros(len(strata), dtype=np.float64)
    matched_strata = 0
    overlap_per_arm = 0.0
    dropped_t = dropped_c = 0
    for value in np.unique(strata):
        rows = strata == value
        treated = rows & contact
        control = rows & ~contact
        nt = int(treated.sum())
        nc = int(control.sum())
        if not nt or not nc:
            dropped_t += nt
            dropped_c += nc
            continue
        matched_strata += 1
        overlap = min(nt, nc)
        weights[treated] = overlap / nt
        weights[control] = overlap / nc
        overlap_per_arm += overlap
    if overlap_per_arm <= 0:
        raise RuntimeError("frozen matching bins produced no contact/noncontact overlap")
    return weights, {
        "matched_strata": matched_strata,
        "overlap_effective_rows_per_arm": float(overlap_per_arm),
        "unmatched_contact_rows": dropped_t,
        "unmatched_noncontact_rows": dropped_c,
        "positive_weight_contact_rows": int(np.sum((weights > 0) & contact)),
        "positive_weight_noncontact_rows": int(np.sum((weights > 0) & ~contact)),
    }


def _weighted_contrast_ci(
    values: np.ndarray,
    contact: np.ndarray,
    weights: np.ndarray,
    episodes: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    contact = np.asarray(contact, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    episodes = np.asarray(episodes, dtype=np.int64)
    unique, inverse = np.unique(episodes, return_inverse=True)
    tw = weights * contact
    cw = weights * (~contact)
    ty = np.bincount(inverse, weights=tw * values, minlength=len(unique))
    tn = np.bincount(inverse, weights=tw, minlength=len(unique))
    cy = np.bincount(inverse, weights=cw * values, minlength=len(unique))
    cn = np.bincount(inverse, weights=cw, minlength=len(unique))
    estimate = float((tw * values).sum() / tw.sum() - (cw * values).sum() / cw.sum())
    rng = np.random.default_rng(int(seed))
    reps = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        size = min(500, samples - start)
        selected = rng.integers(0, len(unique), size=(size, len(unique)))
        treated = ty[selected].sum(1) / tn[selected].sum(1)
        control = cy[selected].sum(1) / cn[selected].sum(1)
        reps[start : start + size] = treated - control
    return {
        "estimate_contact_minus_noncontact": estimate,
        "ci_low": float(np.quantile(reps, 0.025)),
        "ci_high": float(np.quantile(reps, 0.975)),
        "bootstrap_samples": samples,
        "seed": int(seed),
        "weighting": "fixed full-sample coarsened-exact overlap weights; episode-cluster resampling",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    output = common.ROOT / "metrics/physical_regimes.json"
    if output.exists():
        raise RuntimeError("physical-regime analysis already exists")
    arrays, primary = _load()
    config = common.load_config()
    samples = int(config["bootstrap_samples"])
    seeds = config["statistical_seeds"]
    episodes = arrays["episode_slot"].astype(np.int64)
    calls = arrays["calls"].astype(np.int64)
    losses = arrays["losses"].astype(np.float64)
    adaptive = losses[np.arange(len(calls)), calls - 1]
    probabilities = np.asarray(
        primary["exact_total_flop_mixture"]["analytic_probabilities"], dtype=np.float64
    )
    analytic = np.einsum("nd,d->n", losses, probabilities, optimize=False)
    benefit = analytic - adaptive
    cumulative_gain = losses[:, 0] - adaptive
    prior_depth = np.maximum(calls - 2, 0)
    last_gain = np.where(calls > 1, losses[np.arange(len(calls)), prior_depth] - adaptive, 0.0)
    positive = (benefit > 0).astype(np.float64)
    regime_code = arrays["regime_code"].astype(np.int64)
    regime_rows = []
    regime_metrics = []
    for code, name in REGIME_NAMES.items():
        mask = regime_code == code
        base_seed = int(seeds["regime_bootstrap_root"]) + code * 101
        row = {
            "regime": name,
            "rows": int(mask.sum()),
            "episodes_with_rows": int(len(np.unique(episodes[mask]))),
            "mean_calls": float(calls[mask].mean()) if mask.any() else None,
            "mean_cumulative_gain_vs_d1": float(cumulative_gain[mask].mean()) if mask.any() else None,
            "mean_last_accepted_marginal_gain": float(last_gain[mask].mean()) if mask.any() else None,
            "mean_prediction_benefit_vs_exact_total_analytic": float(benefit[mask].mean())
            if mask.any()
            else None,
            "positive_prediction_benefit_rate": float(positive[mask].mean()) if mask.any() else None,
        }
        regime_rows.append(row)
        regime_metrics.append(
            {
                **row,
                "calls_ci": _conditional_cluster_ci(
                    calls, mask, episodes, seed=base_seed, samples=samples
                ),
                "cumulative_gain_ci": _conditional_cluster_ci(
                    cumulative_gain, mask, episodes, seed=base_seed + 1, samples=samples
                ),
                "last_marginal_gain_ci": _conditional_cluster_ci(
                    last_gain, mask, episodes, seed=base_seed + 2, samples=samples
                ),
                "prediction_benefit_ci": _conditional_cluster_ci(
                    benefit, mask, episodes, seed=base_seed + 3, samples=samples
                ),
                "positive_gain_rate_ci": _conditional_cluster_ci(
                    positive, mask, episodes, seed=base_seed + 4, samples=samples
                ),
            }
        )
    contact = arrays["interaction"].astype(bool)
    unmatched = {
        "calls": _unmatched_contrast_ci(
            calls, contact, episodes, seed=int(seeds["contact_unmatched_calls"]), samples=samples
        ),
        "prediction_benefit": _unmatched_contrast_ci(
            benefit,
            contact,
            episodes,
            seed=int(seeds["contact_unmatched_benefit"]),
            samples=samples,
        ),
        "cumulative_gain": _unmatched_contrast_ci(
            cumulative_gain,
            contact,
            episodes,
            seed=int(seeds["contact_unmatched_gain"]),
            samples=samples,
        ),
    }
    frozen_bins = common.read_json(common.ROOT / "audit/regime_matching_bins.json")
    matching = {}
    matching_rows = []
    for index, level in enumerate(("fine", "coarse")):
        strata = _strata(arrays, frozen_bins[level])
        weights, overlap = _matching_weights(strata, contact)
        calls_ci = _weighted_contrast_ci(
            calls,
            contact,
            weights,
            episodes,
            seed=int(seeds[f"contact_{level}_matched_calls"]),
            samples=samples,
        )
        benefit_ci = _weighted_contrast_ci(
            benefit,
            contact,
            weights,
            episodes,
            seed=int(seeds[f"contact_{level}_matched_benefit"]),
            samples=samples,
        )
        gain_ci = _weighted_contrast_ci(
            cumulative_gain,
            contact,
            weights,
            episodes,
            seed=int(seeds[f"contact_{level}_matched_gain"]),
            samples=samples,
        )
        matching[level] = {
            "bins": frozen_bins[level],
            "overlap": overlap,
            "calls": calls_ci,
            "prediction_benefit": benefit_ci,
            "cumulative_gain": gain_ci,
        }
        for outcome, value in (
            ("calls", calls_ci),
            ("prediction_benefit", benefit_ci),
            ("cumulative_gain", gain_ci),
        ):
            matching_rows.append({"matching": level, "outcome": outcome, **value, **overlap})

    block_consistency = []
    for block, start in enumerate((0, 100, 200), start=1):
        mask = (episodes >= start) & (episodes < start + 100)
        for outcome, values in (("calls", calls), ("prediction_benefit", benefit)):
            estimate = float(values[mask & contact].mean() - values[mask & ~contact].mean())
            block_consistency.append(
                {
                    "block": block,
                    "episode_slots": f"{start}-{start + 99}",
                    "outcome": outcome,
                    "unmatched_contact_minus_noncontact": estimate,
                }
            )

    onset_rows = []
    for offset in range(-2, 4):
        selected_rows = []
        for episode in np.unique(episodes):
            ep_rows = np.flatnonzero(episodes == episode)
            local_impact = np.flatnonzero(arrays["impact"][ep_rows])
            for onset in local_impact:
                candidate = onset + offset
                if 0 <= candidate < len(ep_rows):
                    selected_rows.append(ep_rows[candidate])
        index = np.asarray(selected_rows, dtype=np.int64)
        onset_rows.append(
            {
                "model_step_offset_from_impact_onset": offset,
                "rows": len(index),
                "mean_calls": float(calls[index].mean()) if len(index) else None,
                "mean_prediction_benefit_vs_exact_total_analytic": float(benefit[index].mean())
                if len(index)
                else None,
                "mean_cumulative_gain_vs_d1": float(cumulative_gain[index].mean())
                if len(index)
                else None,
            }
        )

    fine = matching["fine"]
    contact_aware = bool(
        fine["calls"]["estimate_contact_minus_noncontact"] > 0
        and fine["calls"]["ci_low"] > 0
        and fine["prediction_benefit"]["estimate_contact_minus_noncontact"] > 0
        and fine["prediction_benefit"]["ci_low"] > 0
    )
    result = {
        "schema_version": 1,
        "status": "executed_after_primary_verdict",
        "primary_decision": common.read_json(common.ROOT / "decision.json")["decision"],
        "contact_and_regime_labels_forbidden_to_solver_and_gate": True,
        "analytic_comparator": "primary exact-total-FLOP transition-independent mixture",
        "regime_definitions": frozen_bins["regime_definitions"],
        "regimes": regime_metrics,
        "contact_vs_noncontact_unmatched": unmatched,
        "contact_vs_noncontact_matched": matching,
        "seed_block_consistency": block_consistency,
        "onset_localized": onset_rows,
        "contact_aware_claim_supported": contact_aware,
        "claim": "motion_phase_controlled_contact_contrast_positive_and_significant"
        if contact_aware
        else "contact_aware_claim_not_supported",
        "interpretation": (
            "A null physical contrast limits the contact-specific motivation but does not alter "
            "the already-written primary adaptive-compute verdict."
        ),
    }
    common.write_json(output, result, exclusive=True)
    _write_csv(common.ROOT / "metrics/regime_table.csv", regime_rows)
    _write_csv(common.ROOT / "metrics/contact_matching_table.csv", matching_rows)
    _write_csv(common.ROOT / "metrics/contact_seed_blocks.csv", block_consistency)
    _write_csv(common.ROOT / "metrics/contact_onset.csv", onset_rows)
    decision_path = common.ROOT / "decision.json"
    original_sha = common.sha256_file(decision_path)
    common.write_json(
        common.ROOT / "audit/primary_decision_pre_secondary.json",
        {
            "decision_sha256_before_secondary": original_sha,
            "physical_metrics_sha256": common.sha256_file(output),
            "primary_verdict_was_written_first": True,
        },
        exclusive=True,
    )
    decision = common.read_json(decision_path)
    decision["physical_claim"] = result["claim"]
    decision["contact_aware_claim_supported"] = contact_aware
    decision["physical_regime_metrics_sha256"] = common.sha256_file(output)
    common.write_json(decision_path, decision)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"claim": result["claim"], "status": result["status"]}, sort_keys=True))
