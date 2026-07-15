from __future__ import annotations

import numpy as np

import common
import flops


def test_frozen_flops_recompute_from_first_principles() -> None:
    result = flops.recompute()
    assert result["passed"]
    assert result["later_adapter"]["flops"] == 264_960
    assert result["gate_per_evaluated_decision"]["total_flops"] == 8_196
    assert result["gate_per_evaluated_decision"]["parameters"] == 1_050


def test_exact_total_flop_budget_rounds_baseline_up_conservatively() -> None:
    calls = np.asarray([1, 1, 2, 3, 4, 1, 2, 1], dtype=np.int64)
    result = common.exact_total_flop_budget(calls)
    assert result["analytic_exact_flop_match"]
    assert result["integer_baseline_weakly_more_compute"]
    assert 0 <= result["integer_baseline_minus_adaptive_flops"] < common.STAGE_ADAPTER_FLOPS
    assert result["integer_target_total_calls"] == int(
        np.ceil(result["analytic_target_total_calls"])
    )


def test_gate_overhead_is_added_to_baseline_not_removed_from_adaptive() -> None:
    calls = np.ones(100, dtype=np.int64)
    result = common.exact_total_flop_budget(calls)
    assert result["analytic_target_total_calls"] > calls.sum()
    assert result["adaptive_gate_total_flops"] == 100 * common.GATE_FLOPS_PER_DECISION
