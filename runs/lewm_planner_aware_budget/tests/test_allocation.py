from __future__ import annotations

import itertools
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from allocation import (  # noqa: E402
    action_grid,
    allocate_hard_budget,
    conservative_values,
    fixed_pair_cost_table,
    trust_mask,
)


class ActionGridTests(unittest.TestCase):
    def test_grid_enumerates_every_fixed_pair(self) -> None:
        actions = action_grid([1, 2, 4], [64, 128, 300])
        self.assertEqual(len(actions), 9)
        self.assertEqual((actions[0].refinement_depth, actions[0].planner_population), (1, 64))
        self.assertEqual((actions[-1].refinement_depth, actions[-1].planner_population), (4, 300))
        table = fixed_pair_cost_table(actions, range(1, 10), range(11, 20))
        self.assertEqual(len(table), 9)
        self.assertEqual(table[4]["counted_flops"], 5)
        self.assertEqual(table[4]["wall_clock_ns"], 15)


class HardBudgetTests(unittest.TestCase):
    def test_allocator_matches_bruteforce_joint_optimum(self) -> None:
        values = np.asarray(
            [[0.0, 1.0, 4.0], [0.0, 3.0, 3.2], [0.0, 2.0, 5.0]]
        )
        costs = np.asarray([1, 2, 4])
        result = allocate_hard_budget(values, costs, budget=7)
        feasible = [
            candidate
            for candidate in itertools.product(range(3), repeat=3)
            if sum(costs[list(candidate)]) <= 7
        ]
        optimum = max(sum(values[row, action] for row, action in enumerate(candidate)) for candidate in feasible)
        self.assertAlmostEqual(result.total_value, optimum)
        self.assertLessEqual(result.total_cost, 7)
        self.assertEqual(result.budget_slack, 7 - result.total_cost)

    def test_trust_failure_abstains_and_reserves_fallback(self) -> None:
        predicted = np.asarray([[0.0, 10.0], [0.0, 10.0]])
        uncertainty = np.asarray([[0.0, 2.0], [0.0, 0.1]])
        disagreement = np.asarray([[0.0, 0.1], [0.0, 3.0]])
        mask = trust_mask(
            uncertainty,
            disagreement,
            max_uncertainty=1.0,
            max_disagreement=1.0,
            fallback_index=0,
        )
        result = allocate_hard_budget(
            conservative_values(predicted, uncertainty, uncertainty_penalty=1.0),
            [1, 2],
            budget=4,
            eligible=mask,
            fallback_index=0,
        )
        np.testing.assert_array_equal(result.action_indices, [0, 0])
        np.testing.assert_array_equal(result.abstained, [True, True])

    def test_infeasible_fallback_budget_and_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "mandatory fallback"):
            allocate_hard_budget(np.zeros((3, 2)), [2, 3], budget=5)
        with self.assertRaisesRegex(ValueError, "finite"):
            allocate_hard_budget(np.asarray([[0.0, np.nan]]), [1, 2], budget=2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            conservative_values([[1.0]], [[-1.0]], uncertainty_penalty=1.0)

    def test_ties_are_deterministic_and_choose_lower_indices(self) -> None:
        values = np.zeros((3, 3))
        first = allocate_hard_budget(values, [1, 2, 3], budget=9)
        second = allocate_hard_budget(values.copy(), [1, 2, 3], budget=9)
        np.testing.assert_array_equal(first.action_indices, [0, 0, 0])
        np.testing.assert_array_equal(first.action_indices, second.action_indices)


if __name__ == "__main__":
    unittest.main()
