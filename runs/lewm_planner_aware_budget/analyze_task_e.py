#!/usr/bin/env python3
"""Aggregate Task E fixed-bank results without treating candidates as IID."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/task_e/sealed"
sys.path.insert(0, str(ROOT))

from task_e import ranking_metrics, selected_change


DEPTH_LABELS = {0: "d0_all_h3", 1: "d1_all_h3", 2: "d2_all_h3", 4: "d4_all_h3"}
PRIMARY_METRICS = ("top_choice_regret", "concordance", "top_30_recall", "kendall_tau_b")


def bootstrap(values: np.ndarray, seed: int, draws: int = 9999) -> dict:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    return {
        "estimate": float(values.mean()),
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
    }


def sign_flip(values: np.ndarray, seed: int, draws: int = 9999) -> float:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    exceed = 0
    for _ in range(draws):
        exceed += abs(float(np.mean(values * rng.choice((-1, 1), len(values))))) >= observed
    return float((exceed + 1) / (draws + 1))


def main() -> None:
    config = json.loads((ROOT / "task_e_config.json").read_text())
    manifest = json.loads((ROOT / "task_e_manifest.json").read_text())
    entries = {row["row_id"]: row for row in manifest["sealed"]}
    records = []
    schedule_records = []
    secondary_records = []
    proximity_records = []
    magnitude_records = []
    bank_manifest = []
    for metadata_path in sorted(WORK.glob("*.json")):
        metadata = json.loads(metadata_path.read_text())
        bank_manifest.append(
            {
                key: metadata[key]
                for key in (
                    "phase",
                    "row_id",
                    "episode_id",
                    "candidate_seed",
                    "candidate_tensor_sha256",
                    "proposal_sha256",
                    "candidate_hashes_sha256",
                    "candidate_count",
                    "valid_count",
                    "model_visible_equals_executed_count",
                    "reset_reconstruction_exact",
                    "replay_checks",
                    "secondary_bank_labelled",
                    "secondary_valid_count",
                    "npz_sha256",
                )
            }
        )
        row_id, seed = metadata["row_id"], metadata["candidate_seed"]
        npz_path = metadata_path.with_suffix(".npz")
        with np.load(npz_path, allow_pickle=False) as data:
            candidates = data["candidates"]
            simulator = data["simulator_cost"]
            scores = {
                depth: data[f"score__{label}"] for depth, label in DEPTH_LABELS.items()
            }
            cost_range = float(np.ptp(simulator))
            block_states = data["states"][:, [4, 9, 14, 19, 24]]
            proximity = np.min(
                np.linalg.norm(block_states[..., :2] - block_states[..., 2:4], axis=-1),
                axis=1,
            )
            action_magnitude = np.linalg.vector_norm(candidates.reshape(len(candidates), -1), axis=1)
            for stratum_name, values in (
                ("proximity_near", proximity),
                ("action_magnitude_high", action_magnitude),
            ):
                boundary = float(np.median(values))
                masks = {
                    "low": values <= boundary,
                    "high": values > boundary,
                }
                if stratum_name == "proximity_near":
                    masks = {"near": values <= boundary, "free": values > boundary}
                for group, mask in masks.items():
                    target = proximity_records if stratum_name == "proximity_near" else magnitude_records
                    for depth, prediction in scores.items():
                        metrics = ranking_metrics(prediction[mask], simulator[mask], top_ks=(10, 30))
                        target.append(
                            {
                                "row_id": row_id,
                                "candidate_seed": seed,
                                "depth": depth,
                                "group": group,
                                "concordance": metrics["concordance"],
                                "kendall_tau_b": metrics["kendall_tau_b"],
                            }
                        )
            for depth, prediction in scores.items():
                metrics = ranking_metrics(prediction, simulator)
                metrics.update(
                    {
                        "row_id": row_id,
                        "episode_id": entries[row_id]["episode_id"],
                        "candidate_seed": seed,
                        "depth": depth,
                        "normalized_top_choice_regret": (
                            metrics["top_choice_regret"] / cost_range if cost_range else None
                        ),
                    }
                )
                if depth:
                    metrics.update(
                        selected_change(
                            scores[0], prediction, simulator, candidates, 1e-7
                        )
                    )
                records.append(metrics)
            for key in data.files:
                if not key.startswith("score__d") or not (
                    "_only_t" in key or "_prefix_" in key
                ):
                    continue
                label = key[len("score__") :]
                metrics = ranking_metrics(data[key], simulator)
                schedule_records.append(
                    {
                        "row_id": row_id,
                        "candidate_seed": seed,
                        "schedule": label,
                        **{name: metrics[name] for name in PRIMARY_METRICS},
                    }
                )
            if metadata["secondary_bank_labelled"]:
                proposal_simulator = data["proposal_simulator_cost"]
                for depth, label in DEPTH_LABELS.items():
                    metrics = ranking_metrics(
                        data[f"proposal_score__{label}"], proposal_simulator, top_ks=(10, 30)
                    )
                    secondary_records.append(
                        {
                            "row_id": row_id,
                            "candidate_seed": seed,
                            "depth": depth,
                            **{name: metrics[name] for name in PRIMARY_METRICS},
                        }
                    )

    analysis = config["analysis"]
    by_key = {(row["row_id"], row["candidate_seed"], row["depth"]): row for row in records}
    contrasts = {}
    for depth in (1, 2, 4):
        for metric in PRIMARY_METRICS:
            differences = []
            for row_id in entries:
                seed_differences = []
                for seed in config["execution"]["sealed_candidate_seeds"]:
                    base = by_key[(row_id, seed, 0)][metric]
                    positive = by_key[(row_id, seed, depth)][metric]
                    value = positive - base
                    if metric == "top_choice_regret":
                        value = -value
                    seed_differences.append(value)
                differences.append(np.mean(seed_differences))
            label = f"d{depth}_minus_d0__{metric}__benefit"
            contrasts[label] = bootstrap(
                np.asarray(differences), analysis["bootstrap_seed"] + depth
            ) | {
                "sign_flip_pvalue": sign_flip(
                    np.asarray(differences), analysis["permutation_seed"] + depth
                ),
                "start_cluster_count": len(differences),
            }

    seed_sign = {}
    for depth in (1, 2, 4):
        signs = []
        for row_id in entries:
            values = []
            for seed in config["execution"]["sealed_candidate_seeds"]:
                base = by_key[(row_id, seed, 0)]["top_choice_regret"]
                positive = by_key[(row_id, seed, depth)]["top_choice_regret"]
                values.append(np.sign(base - positive))
            signs.append(values)
        signs = np.asarray(signs)
        nonzero = (signs[:, 0] != 0) & (signs[:, 1] != 0)
        seed_sign[f"d{depth}"] = {
            "same_nonzero_sign": int(np.sum(nonzero & (signs[:, 0] == signs[:, 1]))),
            "opposite_sign": int(np.sum(nonzero & (signs[:, 0] == -signs[:, 1]))),
            "starts": len(signs),
            "seed_mean_benefit": [
                float(
                    np.mean(
                        [
                            by_key[(row_id, seed, 0)]["top_choice_regret"]
                            - by_key[(row_id, seed, depth)]["top_choice_regret"]
                            for row_id in entries
                        ]
                    )
                )
                for seed in config["execution"]["sealed_candidate_seeds"]
            ],
        }

    oracle = {}
    for depth in (1, 2, 4):
        per_bank = []
        base_regret = []
        depth_regret = []
        for row in records:
            if row["depth"] != 0:
                continue
            other = by_key[(row["row_id"], row["candidate_seed"], depth)]
            base_regret.append(row["top_choice_regret"])
            depth_regret.append(other["top_choice_regret"])
            per_bank.append(min(row["top_choice_regret"], other["top_choice_regret"]))
        oracle[f"base_vs_d{depth}"] = {
            "base_mean_regret": float(np.mean(base_regret)),
            "depth_mean_regret": float(np.mean(depth_regret)),
            "oracle_mean_regret": float(np.mean(per_bank)),
            "oracle_gain_over_base": float(np.mean(base_regret) - np.mean(per_bank)),
            "positive_depth_selected_banks": int(
                np.sum(np.asarray(depth_regret) < np.asarray(base_regret))
            ),
            "banks": len(per_bank),
        }
        selected_positive = np.asarray(depth_regret) < np.asarray(base_regret)
        observed_gain = float(np.mean(base_regret) - np.mean(per_bank))
        rng = np.random.default_rng(analysis["oracle_randomization_seed"] + depth)
        controls = np.empty(analysis["permutation_draws"])
        base_regret_array = np.asarray(base_regret)
        depth_regret_array = np.asarray(depth_regret)
        for draw in range(len(controls)):
            assigned = np.zeros(len(selected_positive), dtype=bool)
            for seed_index in range(2):
                positions = np.arange(seed_index, len(selected_positive), 2)
                count = int(selected_positive[positions].sum())
                assigned[rng.choice(positions, count, replace=False)] = True
            randomized = np.where(assigned, depth_regret_array, base_regret_array)
            controls[draw] = np.mean(base_regret_array) - np.mean(randomized)
        oracle[f"base_vs_d{depth}"].update(
            {
                "identity_control_gain": 0.0,
                "histogram_randomization_mean_gain": float(controls.mean()),
                "histogram_randomization_pvalue": float(
                    (np.sum(controls >= observed_gain) + 1) / (len(controls) + 1)
                ),
            }
        )

    primary_rows = []
    for depth in (0, 1, 2, 4):
        selected = [row for row in records if row["depth"] == depth]
        primary_rows.append(
            {
                "depth": depth,
                **{
                    f"mean_{metric}": float(
                        np.mean([row[metric] for row in selected if row[metric] is not None])
                    )
                    for metric in PRIMARY_METRICS
                },
                "mean_normalized_top_choice_regret": float(
                    np.mean([row["normalized_top_choice_regret"] for row in selected])
                ),
                "selected_changes_vs_base": int(
                    sum(bool(row.get("changed")) for row in selected)
                ),
                "selected_improves": int(
                    sum(row.get("direction") == "improves" for row in selected)
                ),
                "selected_harms": int(
                    sum(row.get("direction") == "harms" for row in selected)
                ),
            }
        )

    schedule_summary = {}
    for label in sorted({row["schedule"] for row in schedule_records}):
        selected = [row for row in schedule_records if row["schedule"] == label]
        schedule_summary[label] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in PRIMARY_METRICS
        }
        base_lookup = {
            (row["row_id"], row["candidate_seed"]): by_key[
                (row["row_id"], row["candidate_seed"], 0)
            ]
            for row in selected
        }
        regret_benefit = np.asarray(
            [
                base_lookup[(row["row_id"], row["candidate_seed"])]["top_choice_regret"]
                - row["top_choice_regret"]
                for row in selected
            ]
        )
        schedule_summary[label]["top_choice_regret_benefit"] = bootstrap(
            regret_benefit, analysis["bootstrap_seed"] + sum(map(ord, label))
        )
    secondary_summary = {}
    for depth in (0, 1, 2, 4):
        selected = [row for row in secondary_records if row["depth"] == depth]
        secondary_summary[f"d{depth}"] = {
            metric: float(np.mean([row[metric] for row in selected]))
            for metric in PRIMARY_METRICS
        }

    def stratum_summary(rows):
        output = {}
        for group in sorted({row["group"] for row in rows}):
            for depth in (1, 2, 4):
                differences = []
                keys = sorted(
                    {
                        (row["row_id"], row["candidate_seed"])
                        for row in rows
                        if row["group"] == group
                    }
                )
                lookup = {
                    (row["row_id"], row["candidate_seed"], row["depth"], row["group"]): row
                    for row in rows
                }
                for row_id, seed in keys:
                    differences.append(
                        lookup[(row_id, seed, depth, group)]["concordance"]
                        - lookup[(row_id, seed, 0, group)]["concordance"]
                    )
                output[f"{group}_d{depth}_concordance_benefit"] = bootstrap(
                    np.asarray(differences),
                    analysis["bootstrap_seed"] + depth + sum(map(ord, group)),
                )
        return output

    variance = {}
    for depth in (1, 2, 4):
        matrix = np.asarray(
            [
                [
                    by_key[(row_id, seed, 0)]["top_choice_regret"]
                    - by_key[(row_id, seed, depth)]["top_choice_regret"]
                    for seed in config["execution"]["sealed_candidate_seeds"]
                ]
                for row_id in entries
            ]
        )
        start_means = matrix.mean(1)
        within = matrix - start_means[:, None]
        variance[f"d{depth}"] = {
            "between_start_variance_of_seed_means": float(np.var(start_means, ddof=1)),
            "between_bank_within_start_variance": float(np.var(within, ddof=1)),
            "depth_benefit_variance_total": float(np.var(matrix, ddof=1)),
        }

    result = {
        "schema_version": 1,
        "banks": 16,
        "starts": 8,
        "candidate_seeds": config["execution"]["sealed_candidate_seeds"],
        "candidates_per_primary_bank": 300,
        "valid_primary_labels": 4800,
        "primary_summary": primary_rows,
        "primary_contrasts": contrasts,
        "seed_sign_reproduction": seed_sign,
        "optimistic_oracle": oracle,
        "schedule_summary": schedule_summary,
        "secondary_proposal_summary": secondary_summary,
        "proximity_strata": stratum_summary(proximity_records),
        "action_magnitude_strata": stratum_summary(magnitude_records),
        "variance_partition_descriptive": variance,
    }
    (ROOT / "task_e_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    run_manifest = {
        "schema_version": 1,
        "status": "sealed_complete",
        "start_manifest": "task_e_manifest.json",
        "raw_root": "work/task_e/sealed (ignored)",
        "protocol": {
            "initial_frozen_sha256": manifest["provenance"]["task_e_protocol_sha256"],
            "final_sha256": hashlib.sha256(
                (ROOT / "TASK_E_PROTOCOL.md").read_bytes()
            ).hexdigest(),
            "mechanical_amendment": (
                "Clarified inherited _set_state one-tick reconstruction after the "
                "outcome-blind pilot; no bank, endpoint, seed, or analysis changed."
            ),
        },
        "banks": bank_manifest,
        "integrity": {
            "bank_count": len(bank_manifest),
            "candidate_labels": int(sum(row["valid_count"] for row in bank_manifest)),
            "model_visible_equals_executed": int(
                sum(row["model_visible_equals_executed_count"] for row in bank_manifest)
            ),
            "replay_checks": int(
                sum(len(row["replay_checks"]) for row in bank_manifest)
            ),
            "all_replays_exact": all(
                check["actions_exact"] and check["states_exact"] and check["costs_exact"]
                for row in bank_manifest
                for check in row["replay_checks"]
            ),
        },
    }
    (ROOT / "task_e_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    fields = sorted({key for row in records for key in row})
    with (ROOT / "task_e_bank_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"results": str(ROOT / "task_e_results.json"), "banks": 16}))


if __name__ == "__main__":
    main()
