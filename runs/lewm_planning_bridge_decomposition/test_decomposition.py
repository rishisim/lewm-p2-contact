from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import decompose


def test_configuration_and_minimal_status_are_sealed() -> None:
    config = json.loads((ROOT / "CONFIG.json").read_text(encoding="utf-8"))
    status = json.loads((ROOT / "RUN_STATUS.json").read_text(encoding="utf-8"))
    assert config["consumed_pilot"]["start_count"] == 20
    assert config["consumed_pilot"]["candidate_count_per_start"] == 64
    assert config["scope"] == {
        "fit_anything": False,
        "model_gate_or_threshold_change": False,
        "new_candidates_or_starts": False,
        "new_scientific_cohort": False,
        "prior_runs_read_only": True,
    }
    assert set(status) == {
        "configured",
        "simulator_verified",
        "model_verified",
        "analyzed",
        "complete",
    }


def test_all_configured_frozen_and_prior_hashes_verify() -> None:
    observed = decompose.verify_source_hashes()
    config = decompose.read_json(ROOT / "CONFIG.json")
    assert len(observed) == len(config["frozen_source_hashes_sha256"]) + len(
        config["prior_bridge_hashes_sha256"]
    )


def test_tolerance_groups_use_non_chaining_complete_link_rule() -> None:
    values = np.asarray([0.00018, 0.0, 0.00009, 0.00028], dtype=np.float64)
    groups = decompose.tolerance_groups(values, 0.0001)
    assert groups.tolist() == [1, 0, 0, 1]


def test_stable_top_k_breaks_exact_ties_by_candidate_index() -> None:
    values = np.asarray([1.0, 0.0, 0.0, 2.0, 0.0, 3.0])
    assert decompose.stable_top_k(values, 3).tolist() == [1, 2, 4]
    groups = np.asarray([1, 0, 0, 2, 0, 3])
    assert decompose.stable_group_top_k(groups, 3).tolist() == [1, 2, 4]


def test_physical_rank_is_undefined_for_one_meaningful_group() -> None:
    cost = np.arange(64, dtype=np.float64)
    physical = np.zeros(64, dtype=np.float64)
    groups = np.zeros(64, dtype=np.int16)
    result = decompose.physical_rank_metrics(cost, physical, groups)
    assert result["spearman"] is None
    assert result["selected_physical_regret_m"] == 0.0


def test_cluster_bootstrap_is_deterministic_for_fixed_resamples() -> None:
    values = np.asarray([-2.0, -1.0, 1.0, 2.0])
    rng = np.random.default_rng(7)
    resamples = rng.integers(0, 4, size=(100, 4))
    first = decompose.bootstrap_effect(values, resamples)
    second = decompose.bootstrap_effect(values, resamples.copy())
    assert first == second
    assert first["mean"] == 0.0
    assert first["starts_favoring_adaptive"] == 2


def test_per_start_step_compute_matching_is_exact() -> None:
    adaptive = np.ones((20, 5, 64), dtype=np.int8)
    adaptive[:, :, :8] = 2
    adaptive[:, :, :2] = 4
    matched = np.roll(adaptive, shift=11, axis=2)
    summary, arrays = decompose.compute_accounting(adaptive, matched)
    assert summary["adaptive_matched_exact_at_all_100_start_steps"] is True
    assert np.array_equal(arrays["depth_histogram"][0], arrays["depth_histogram"][1])
    assert np.array_equal(arrays["counted_flops"][0], arrays["counted_flops"][1])


def test_final_compact_artifacts_and_decision_table_verify() -> None:
    verification = decompose.verify_final()
    assert verification["source_hashes_verified"] is True
    assert verification["arrays_verified"] is True
    assert verification["status"]["configured"] is True
    assert verification["status"]["model_verified"] is True
    assert verification["status"]["simulator_verified"] is True
    assert verification["status"]["analyzed"] is True
    assert set(verification["decision_table"]) == {
        "candidate_rollout_transfer",
        "goal_cost_alignment",
        "candidate_informativeness",
        "downstream_planning_evidence",
    }
