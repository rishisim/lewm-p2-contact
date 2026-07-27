from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = json.loads((ROOT / "task_c_results.json").read_text())


def test_complete_grid_and_latency_schema():
    cells = RESULT["cells"]
    assert RESULT["grid_complete"] is True
    assert len(cells) == 12
    assert {
        (cell["depth"], cell["population"]) for cell in cells
    } == {
        (depth, population)
        for depth in (0, 1, 2, 4)
        for population in (64, 128, 300)
    }
    for cell in cells:
        latency = cell["latency"]
        assert latency["unit"] == "ns"
        assert latency["repetitions"] == 7
        assert latency["warmups_excluded"] == 2
        assert latency["constant_work"] is True
        assert 0 < latency["median_ns"] <= latency["p95_ns"]


def test_exact_call_ledgers_for_every_cell():
    for cell in RESULT["cells"]:
        population, depth = cell["population"], cell["depth"]
        smoke = cell["planner_call"]
        model = smoke["model_ledger"]
        cem = smoke["cem_ledger_per_call"]
        assert smoke["replans"] == 1
        assert smoke["iterations"] == 20 and smoke["horizon"] == 5
        assert smoke["finite_actions"] and smoke["finite_costs"]
        assert model["base_calls"] == 100
        assert model["base_rows"] == population * 100
        assert model["image_encoder_calls"] == model["goal_encoder_calls"] == 20
        assert model["terminal_cost_rows"] == population * 20
        assert model["refiner_stage_calls"] == [
            100 if stage < depth else 0 for stage in range(4)
        ]
        assert model["refiner_stage_rows"] == [
            population * 100 if stage < depth else 0 for stage in range(4)
        ]
        assert cem["candidate_sequences_evaluated"] == population * 20
        assert cem["predicted_transition_rows"] == population * 100
        assert cem["sampling_invocations"] == 20
        assert cem["update_invocations"] == 20
        assert cem["topk_invocations"] == 20
        assert cem["innovation_tensor_shape"] == [1, 300, 5, 10]


def test_anchor_effects_flop_coverage_and_order():
    assert RESULT["anchor"]["passed"] is True
    assert RESULT["anchor"]["torch_equal"] is True
    assert RESULT["anchor"]["maximum_absolute_cost_difference"] == 0
    for population in ("64", "128", "300"):
        for depth in ("1", "2", "4"):
            effect = RESULT["positive_depth_effect_checks"][population][depth]
            assert effect["finite"]
            assert effect["costs_changed"]
            assert effect["ranking_changed"]
    constants = RESULT["counted_flop_constants"]
    assert constants["complete_measured_flops"] is False
    assert constants["exclusions"]
    assert constants["smallest_unblocker"]
    assert len(RESULT["measurement"]["run_order"]) == 7 * 12
    for cell in RESULT["cells"]:
        assert cell["counted_flops"]["counted_total"] > 0
