import json
from pathlib import Path

import numpy as np

from task_d import (
    CELLS,
    brute_force_allocate,
    cell_order,
    cluster_bootstrap_mean_difference,
    expected_work,
    hard_budget_allocate,
    sign_permutation_pvalue,
)


ROOT = Path(__file__).parents[1]


def test_frozen_config_grid_and_protocol_status():
    cfg = json.loads((ROOT / "task_d_config.json").read_text())
    assert cfg["status"] == "frozen_before_task_d_outcomes"
    assert len(CELLS) == 12
    assert cfg["execution"]["sealed_starts"] == 24
    assert len(cfg["execution"]["sealed_candidate_seeds"]) == 2


def test_order_is_deterministic_complete_and_seeded():
    a = cell_order(2026072611, 123, 2026072601)
    assert a == cell_order(2026072611, 123, 2026072601)
    assert set(a) == set(CELLS)
    assert a != cell_order(2026072611, 123, 2026072602)


def test_work_identities():
    work = expected_work(300, 4, 7)
    assert work["candidate_sequences_evaluated"] == 7 * 300 * 20
    assert work["base_calls"] == 700
    assert work["refiner_stage_calls"] == [700] * 4
    assert work["terminal_cost_rows"] == 7 * 300 * 20


def test_bootstrap_and_permutation_are_deterministic():
    values = np.asarray([1.0, -2.0, 4.0, 3.0])
    assert cluster_bootstrap_mean_difference(values, 99, 7) == cluster_bootstrap_mean_difference(values, 99, 7)
    assert sign_permutation_pvalue(values, 99, 8) == sign_permutation_pvalue(values, 99, 8)


def test_hard_budget_matches_brute_force_on_bounded_cases():
    rng = np.random.default_rng(11)
    for _ in range(12):
        quality = rng.normal(size=(4, 3))
        costs = np.asarray([2, 3, 5])
        budget = 15
        exact = hard_budget_allocate(quality, costs, budget)
        brute = brute_force_allocate(quality, costs, budget)
        assert np.isclose(exact["value"], brute["value"])
        assert exact["spent"] <= budget
