import json
from pathlib import Path

import numpy as np

from task_d import CELLS, expected_work


ROOT = Path(__file__).parents[1]


def test_complete_result_schema_and_sealed_ledger():
    result = json.loads((ROOT / "task_d_results.json").read_text())
    ledger = json.loads((ROOT / "task_d_ledger.json").read_text())
    assert result["grid_complete"]
    assert result["record_count"] == 24 * 2 * 12
    assert set(result["cells"]) == set(CELLS)
    assert ledger["record_count"] == result["record_count"]
    keys = [
        (row["key"]["row_id"], row["key"]["candidate_seed"], row["key"]["cell"])
        for row in ledger["records"]
    ]
    assert len(keys) == len(set(keys))


def test_every_record_has_exact_work_and_finite_outcome():
    ledger = json.loads((ROOT / "task_d_ledger.json").read_text())
    for row in ledger["records"]:
        assert row["failure"] is None
        assert np.isfinite(row["normalized_return"])
        assert np.isfinite(row["cumulative_task_cost"])
        expected = expected_work(row["population"], row["depth"], row["work"]["replans"])
        assert row["work"] == expected
        assert len(row["planner_calls"]) == row["work"]["replans"]
        assert row["counted_flops_complete"] is False


def test_seed_and_cell_pairing_is_complete():
    ledger = json.loads((ROOT / "task_d_ledger.json").read_text())
    grouped = {}
    for row in ledger["records"]:
        grouped.setdefault(row["key"]["row_id"], set()).add(
            (row["key"]["candidate_seed"], row["key"]["cell"])
        )
    expected = {
        (seed, cell)
        for seed in (2026072601, 2026072602)
        for cell in CELLS
    }
    assert len(grouped) == 24
    assert all(values == expected for values in grouped.values())


def test_decision_is_negative_and_oracles_are_budget_feasible():
    result = json.loads((ROOT / "task_d_results.json").read_text())
    assert result["decision"]["supported"] is False
    for mode in result["opportunity"].values():
        for directions in mode.values():
            for row in directions:
                for name in ("mixture", "joint", "depth_only", "population_only"):
                    assert row[name]["slack"] >= 0
