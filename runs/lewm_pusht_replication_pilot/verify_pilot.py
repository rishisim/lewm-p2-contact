#!/usr/bin/env python3
"""Independent, lossless-array recomputation for the PushT replication pilot."""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pusht_core import (
    MODEL_CONFIG, MODEL_WEIGHTS, REPO_ROOT, ROOT, atomic_json, bootstrap_interval,
    build_causal_features, causal_feature_names, episode_means, feature_arrays,
    gate_scores, git_snapshot, load_npz, raw_white_losses, read_json, safe_spearman,
    sequential_calls, sha256_file, strongest_analytic_mixture,
)


TEST_PYTHON = Path("/Users/rishisim/.cache/lewm-v2-venv/bin/python")


def close(left: float, right: float, *, atol: float = 1e-12, rtol: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, atol=atol, rtol=rtol))


def run_tests() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [str(TEST_PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests/test_pilot.py")],
        cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    payload = {
        "command": [str(TEST_PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_pilot.py"],
        "exit_code": result.returncode, "passed": result.returncode == 0,
        "output": result.stdout, "created_unix_ns": time.time_ns(),
    }
    atomic_json(ROOT / "TEST_RESULTS.json", payload)
    return payload


def verify_roles(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    checks: dict[str, Any] = {}
    roles: dict[str, dict[str, np.ndarray]] = {}
    all_seeds: dict[str, set[int]] = {}
    all_identifiers: dict[str, set[str]] = {}
    for role, count in config["episode_counts"].items():
        path = ROOT / f"data/{role}.npz"
        manifest_path = ROOT / f"data/{role}_manifest.json"
        if not path.exists():
            if role == "evaluation" and (ROOT / "EARLY_TERMINAL.json").exists():
                continue
            checks[f"{role}_exists"] = False
            continue
        arrays = load_npz(path)
        manifest = read_json(manifest_path)
        roles[role] = arrays
        expected_specs = config["role_specs"][role]
        expected_seeds = {int(item["seed"]) for item in expected_specs}
        expected_identifiers = {str(item["identifier"]) for item in expected_specs}
        all_seeds[role] = expected_seeds
        all_identifiers[role] = expected_identifiers
        checks[f"{role}_episode_count"] = (
            len(expected_specs) == count
            and manifest["episode_count"] == count
            and np.array_equal(np.unique(arrays["episode_id"]), np.arange(count))
        )
        checks[f"{role}_seeds_exact"] = set(map(int, np.unique(arrays["episode_seed"]))) == expected_seeds
        checks[f"{role}_hash"] = manifest["data_sha256"] == sha256_file(path)
        checks[f"{role}_no_replacement"] = manifest["no_exclusions"] and manifest["no_replacements"]
        checks[f"{role}_fresh_no_hdf5"] = manifest["fresh_simulator_generated"] and not manifest["hdf5_or_h5_opened"]
        checks[f"{role}_pixels_not_retained"] = not manifest["pixels_retained"]
        checks[f"{role}_base_frozen"] = (
            manifest["base_frozen"]
            and manifest["base_parameter_digest_before"] == manifest["base_parameter_digest_after"]
        )
        checks[f"{role}_finite"] = all(np.isfinite(value).all() for value in arrays.values())
        checks[f"{role}_every_episode_has_rows"] = all(
            np.sum(arrays["episode_id"] == identifier) > 0 for identifier in range(count)
        )
    names = list(all_seeds)
    checks["role_seed_disjointness"] = all(
        all_seeds[names[i]].isdisjoint(all_seeds[names[j]])
        for i in range(len(names)) for j in range(i + 1, len(names))
    )
    checks["role_identifier_disjointness"] = all(
        all_identifiers[names[i]].isdisjoint(all_identifiers[names[j]])
        for i in range(len(names)) for j in range(i + 1, len(names))
    )
    smoke_seeds = {int(item["seed"]) for item in config["smoke"]["episodes"]}
    checks["smoke_seeds_excluded"] = all(smoke_seeds.isdisjoint(value) for value in all_seeds.values())
    checks["exact_two_smoke_episode_specs"] = config["smoke"]["compatibility"]["exact_unique_smoke_episode_count"] == 2
    return checks, roles


def verify_fit(config: dict[str, Any], fit: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics = read_json(ROOT / "REFINER_FIT_TABLE.json")
    outputs = load_npz(ROOT / "development/fit_refiner_outputs.npz")
    artifacts = load_npz(ROOT / "FIT_ARTIFACTS.npz")
    raw, white = raw_white_losses(outputs["exits"], fit["target"], artifacts["whitening_matrix"])
    base_difference = fit["base"].astype(np.float64) - fit["target"].astype(np.float64)
    base_raw = np.square(base_difference).mean(axis=1)
    base_white = np.square(base_difference @ artifacts["whitening_matrix"]).mean(axis=1)
    table_exact = close(metrics["base_raw_mse"], float(base_raw.mean())) and close(
        metrics["base_fit_whitened_mse"], float(base_white.mean())
    )
    previous_raw, previous_white = base_raw, base_white
    for column, row in enumerate(metrics["depth_table"]):
        raw_gain = previous_raw - raw[:, column]
        white_gain = previous_white - white[:, column]
        table_exact &= all(
            (
                close(row["raw_mse"], float(raw[:, column].mean())),
                close(row["fit_whitened_mse"], float(white[:, column].mean())),
                close(row["raw_marginal_gain"], float(raw_gain.mean())),
                close(row["fit_whitened_marginal_gain"], float(white_gain.mean())),
                close(row["fraction_raw_benefiting"], float(np.mean(raw_gain > 0))),
            )
        )
        previous_raw, previous_white = raw[:, column], white[:, column]
    manifest = read_json(ROOT / "checkpoints/refiner_manifest.json")
    return {
        "fit_losses_recomputed_from_lossless_exits": bool(
            np.allclose(raw, outputs["raw_losses"], rtol=1e-11, atol=1e-13)
            and np.allclose(white, outputs["white_losses"], rtol=1e-11, atol=1e-13)
        ),
        "refiner_fit_table_recomputed": bool(table_exact),
        "checkpoint_hash": manifest["checkpoint_sha256"] == sha256_file(ROOT / "checkpoints/refiner.pt"),
        "fit_artifacts_hash": manifest["fit_artifacts_sha256"] == sha256_file(ROOT / "FIT_ARTIFACTS.npz"),
        "base_not_loaded_or_trainable_during_refiner_fit": not manifest[
            "base_model_trainable_or_loaded_during_refiner_training"
        ],
        "finite_fit_outputs": bool(np.isfinite(raw).all() and np.isfinite(white).all()),
    }


def verify_evaluation(
    config: dict[str, Any], evaluation: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = read_json(ROOT / "PILOT_DECISION.json")
    stored = load_npz(ROOT / "EVALUATION_ARRAYS.npz")
    artifacts = load_npz(ROOT / "FIT_ARTIFACTS.npz")
    gate = load_npz(ROOT / "FROZEN_GATE.npz")
    freeze = read_json(ROOT / "GATE_FREEZE.json")
    raw, white = raw_white_losses(stored["dense_exits"], stored["target"], artifacts["whitening_matrix"])
    features = feature_arrays(
        evaluation, stored["dense_exits"], stored["dense_updates"],
        artifacts["action_mean"], artifacts["action_scale"],
    )
    fitted = {
        "feature_mean": gate["feature_mean"], "feature_scale": gate["feature_scale"],
        "weights": gate["weights"][None],
    }
    scores = gate_scores(features, fitted, 0)
    calls, reached = sequential_calls(scores, gate["thresholds"])
    row = np.arange(len(calls))
    selected = stored["dense_exits"][row, calls - 1]
    episode_id = stored["episode_id"].astype(int)
    episode_count = int(config["episode_counts"]["evaluation"])
    refiner_price = int(config["compute"]["refiner"]["total_flops_per_call"])
    gate_price = int(config["compute"]["gate"]["total_flops_per_reached_evaluation"])
    gate_evaluations = int(np.minimum(calls, 3).sum())
    equivalent_depth = (
        int(calls.sum()) * refiner_price + gate_evaluations * gate_price
    ) / (len(calls) * refiner_price)
    raw_analytic = strongest_analytic_mixture(raw, equivalent_depth, episode_id, episode_count)
    white_analytic = strongest_analytic_mixture(white, equivalent_depth, episode_id, episode_count)
    adaptive_raw = raw[row, calls - 1]
    adaptive_white = white[row, calls - 1]
    permutation = stored["permutation_calls"].astype(int)
    endpoint_values = {
        "raw_vs_primary_analytic": episode_means(raw_analytic["loss"] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_primary_analytic": episode_means(
            white_analytic["loss"] - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_within_episode_permutation": episode_means(
            raw[row, permutation - 1] - adaptive_raw, episode_id, episode_count
        ),
        "fit_whitened_vs_within_episode_permutation": episode_means(
            white[row, permutation - 1] - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_fixed_depth_1": episode_means(raw[:, 0] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_fixed_depth_1": episode_means(
            white[:, 0] - adaptive_white, episode_id, episode_count
        ),
        "raw_vs_fixed_depth_4": episode_means(raw[:, 3] - adaptive_raw, episode_id, episode_count),
        "fit_whitened_vs_fixed_depth_4": episode_means(
            white[:, 3] - adaptive_white, episode_id, episode_count
        ),
    }
    interval_exact = True
    array_exact = True
    recomputed_intervals: dict[str, Any] = {}
    for name, values in endpoint_values.items():
        array_exact &= np.array_equal(values, stored[f"episode_{name}"])
        interval = bootstrap_interval(
            values, seed=int(config["evaluation"]["bootstrap_seed"]),
            replicates=int(config["evaluation"]["bootstrap_replicates"]), confidence=0.95,
        )
        recomputed_intervals[name] = interval
        recorded = decision["all_endpoints"][name]
        interval_exact &= all(
            close(interval[key], recorded[key]) for key in ("mean_benefit", "ci_low", "ci_high")
        )
    stage_ranks = []
    for stage in range(3):
        raw_gain = raw[:, stage] - raw[:, stage + 1]
        white_gain = white[:, stage] - white[:, stage + 1]
        combined = 0.5 * (
            raw_gain / gate["raw_gain_scale"][stage]
            + white_gain / gate["white_gain_scale"][stage]
        )
        mask = reached[stage]
        stage_ranks.append(safe_spearman(scores[mask, stage], combined[mask]))
    recorded_ranks = [item["score_vs_combined_gain_spearman"] for item in decision["stagewise_rank"]]
    base_price = int(config["compute"]["base_predictor"]["total_flops_per_transition"])
    norm_price = int(config["compute"]["gate"]["action_normalization_flops_per_transition"])
    total = (
        len(calls) * (base_price + norm_price)
        + int(calls.sum()) * refiner_price + gate_evaluations * gate_price
    )
    forbidden = ("target", "future", "contact", "reward", "success", "simulator", "geometry", "state")
    checks = {
        "evaluation_arrays_hash": decision["evaluation_arrays_sha256"] == sha256_file(ROOT / "EVALUATION_ARRAYS.npz"),
        "target_matches_fixed_role": np.array_equal(stored["target"], evaluation["target"]),
        "episode_identifiers_match_fixed_role": np.array_equal(stored["episode_id"], evaluation["episode_id"]),
        "scores_recomputed_from_causal_features": np.allclose(scores, stored["scores"], rtol=2e-11, atol=2e-12),
        "calls_recomputed": np.array_equal(calls, stored["calls"]),
        "selected_outputs_recomputed": np.array_equal(selected, stored["selected"]),
        "headline_episode_arrays_recomputed": bool(array_exact),
        "all_intervals_recomputed": bool(interval_exact),
        "stagewise_ranks_recomputed": all(close(a, b, atol=1e-11) for a, b in zip(stage_ranks, recorded_ranks)),
        "exact_compute_recomputed": (
            total == decision["compute"]["adaptive_total_counted_flops"]
            == decision["compute"]["primary_analytic_comparator_total_counted_flops"]
        ),
        "call_histogram_recomputed": np.bincount(calls, minlength=5)[1:].tolist()
        == decision["compute"]["call_histogram_depth_1_to_4"],
        "permutation_histograms_exact": all(
            np.array_equal(
                np.bincount(calls[episode_id == identifier], minlength=5),
                np.bincount(permutation[episode_id == identifier], minlength=5),
            )
            for identifier in range(episode_count)
        ),
        "finite_evaluation_arrays": all(
            np.isfinite(value).all()
            for value in (stored["dense_exits"], stored["dense_updates"], scores, raw, white, selected)
        ),
        "causal_signature_exact": list(inspect.signature(build_causal_features).parameters)
        == ["history", "actions", "current", "update"],
        "causal_feature_names_exclude_forbidden_terms": not any(
            term in name.lower() for name in causal_feature_names() for term in forbidden
        ),
        "gate_freeze_precedes_decision": freeze["created_unix_ns"] < decision["created_unix_ns"],
        "evaluation_absent_at_gate_freeze": not freeze["evaluation_data_exists_at_freeze"],
        "no_privileged_gate_inputs": not freeze[
            "target_future_contact_reward_success_simulator_state_or_geometry_inputs"
        ],
    }
    headline = {
        "raw": recomputed_intervals["raw_vs_primary_analytic"],
        "fit_whitened": recomputed_intervals["fit_whitened_vs_primary_analytic"],
        "stagewise_combined_spearman": stage_ranks,
        "total_counted_flops_each": total,
    }
    return checks, headline


def main() -> None:
    config = read_json(ROOT / "CONFIG.json")
    tests = run_tests()
    role_checks, roles = verify_roles(config)
    checks: dict[str, Any] = {
        "focused_tests": tests["passed"],
        "cached_config_hash": config["read_only_inputs"]["model_config_sha256"] == sha256_file(MODEL_CONFIG),
        "cached_weights_hash": config["read_only_inputs"]["model_weights_sha256"] == sha256_file(MODEL_WEIGHTS),
        **role_checks,
    }
    checks.update({f"fit_{key}": value for key, value in verify_fit(config, roles["fit"]).items()})
    headline: dict[str, Any] = {}
    if (ROOT / "EARLY_TERMINAL.json").exists():
        checks["early_terminal_no_gate_or_evaluation"] = not any(
            path.exists() for path in (ROOT / "GATE_FREEZE.json", ROOT / "data/evaluation.npz", ROOT / "PILOT_DECISION.json")
        )
    else:
        evaluation_checks, headline = verify_evaluation(config, roles["evaluation"])
        checks.update({f"evaluation_{key}": value for key, value in evaluation_checks.items()})
        selection = read_json(ROOT / "SELECTION_LEDGER.json")
        freeze = read_json(ROOT / "GATE_FREEZE.json")
        checks["selection_once_no_refit"] = (
            selection["status"] == "single_selection_complete_no_refit"
            and not selection["refiner_or_features_refit_after_selection"]
            and not freeze["refit_after_selection"]
        )
        checks["candidate_count_at_most_eight"] = selection["candidate_count"] == 6
        checks["gate_checkpoint_hashes"] = (
            freeze["frozen_gate_sha256"] == sha256_file(ROOT / "FROZEN_GATE.npz")
            and freeze["refiner_checkpoint_sha256"] == sha256_file(ROOT / "checkpoints/refiner.pt")
            and freeze["fit_artifacts_sha256"] == sha256_file(ROOT / "FIT_ARTIFACTS.npz")
        )
    current_git = git_snapshot()
    baseline = config["git_baseline"]
    checks["git_head_unchanged"] = current_git["head"] == baseline["head"]
    checks["git_tracked_state_unchanged"] = current_git["tracked_diff"] == baseline["tracked_diff"]
    checks["git_porcelain_scope_unchanged"] = current_git["status"] == baseline["status"]
    checks["all_source_hashes_still_locked"] = all(
        sha256_file(ROOT / relative) == expected
        for relative, expected in config["source_hashes_at_lock"].items()
    )
    passed = all(bool(value) for value in checks.values())
    payload = {
        "schema_version": 1, "created_unix_ns": time.time_ns(), "passed": passed,
        "checks": checks, "headline_recomputation": headline,
        "git": current_git,
        "verification_method": "independent reload and float64 recomputation from lossless NPZ arrays",
    }
    atomic_json(ROOT / "INDEPENDENT_VERIFICATION.json", payload)
    print(json.dumps({"passed": passed, "failed_checks": [key for key, value in checks.items() if not value], "headline": headline}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
