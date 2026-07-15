from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy import (  # noqa: E402
    call_audit,
    calibrate_compute_price,
    causal_sequential_stopping,
    clustered_bootstrap_ci,
    clustered_paired_loss_ci,
    fixed_exit,
    gain_ranking_calibration,
    gather_exit_values,
    nondominated_mask,
    optimal_expected_mixture,
    pareto_frontier,
    per_depth_gain_diagnostics,
    permute_critic_scores,
    randomized_histogram_control,
    strongest_transition_independent_baseline,
)


class SequentialPolicyTests(unittest.TestCase):
    def test_prefix_halting_and_future_score_causality(self):
        scores = np.asarray(
            [[2.0, 2.0, 2.0], [2.0, -1.0, 1e9], [-1.0, 1e9, 1e9], [2.0, 2.0, -1.0]]
        )
        selected = causal_sequential_stopping(scores)
        np.testing.assert_array_equal(selected, [4, 2, 1, 3])

        # Arbitrarily changing every score after the first failed decision has
        # no effect: a stopped transition cannot re-enter the cascade.
        changed = scores.copy()
        changed[1, 2] = -1e12
        changed[2, 1:] = -1e12
        np.testing.assert_array_equal(causal_sequential_stopping(changed), selected)

    def test_threshold_price_and_nonconsecutive_call_cost(self):
        scores = np.asarray([[0.6, 1.1], [0.3, 100.0], [0.6, 0.8]])
        selected = causal_sequential_stopping(
            scores,
            thresholds=[0.1, 0.1],
            compute_price=0.25,
            exit_calls=[1, 2, 5],
        )
        # Required gains are .35 then .85 because the second jump costs 3 calls.
        np.testing.assert_array_equal(selected, [5, 1, 2])

    def test_price_calibration_hits_nearest_budget_without_external_data(self):
        scores = np.asarray(
            [[0.9, 0.9], [0.8, 0.8], [0.7, -1.0], [0.6, -1.0], [-1.0, 9.0]]
        )
        losses = np.asarray(
            [[0.4, 0.3, 0.2], [0.4, 0.3, 0.2], [0.4, 0.3, 0.4],
             [0.4, 0.3, 0.4], [0.4, 0.5, 0.1]]
        )
        result = calibrate_compute_price(
            scores, 1.8, calibration_losses=losses
        )
        self.assertAlmostEqual(result["realized_mean_calls"], 1.8)
        self.assertEqual(int(result["selected_calls"].sum()), 9)
        self.assertAlmostEqual(result["calibration_mean_loss"], 0.32)

    def test_fixed_exit_gather_and_audit(self):
        selected = fixed_exit(4, 2)
        np.testing.assert_array_equal(selected, [2, 2, 2, 2])
        values = np.asarray([[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33]])
        np.testing.assert_array_equal(
            gather_exit_values(values, [1, 3, 2, 3], [1, 2, 3]), [10, 31, 22, 33]
        )
        audit = call_audit(
            [1, 2, 2, 3],
            supported_calls=[1, 2, 3],
            target_total_calls=8,
            expected_mean_calls=2.0,
        )
        self.assertEqual(audit["histogram"], {"1": 1, "2": 2, "3": 1})
        self.assertTrue(audit["exact_total_match"])
        self.assertTrue(audit["expected_match"])


class MatchedBaselineTests(unittest.TestCase):
    def test_expected_mixture_uses_strongest_feasible_depths(self):
        # At mean 2 calls, 50/50 exits 1 and 3 has loss .25, beating fixed d2.
        result = optimal_expected_mixture([0.4, 0.3, 0.1], [1, 2, 3], 2.0)
        self.assertAlmostEqual(result["expected_mean_calls"], 2.0)
        self.assertAlmostEqual(result["calibration_mean_loss"], 0.25)
        np.testing.assert_allclose(result["probabilities"], [0.5, 0.0, 0.5], atol=1e-7)

    def test_exact_strongest_randomized_baseline_and_determinism(self):
        losses = np.tile([0.4, 0.3, 0.1], (20, 1))
        first = strongest_transition_independent_baseline(
            losses, [1, 2, 3], n=20, target_total_calls=40, seed=7
        )
        second = strongest_transition_independent_baseline(
            losses, [1, 2, 3], n=20, target_total_calls=40, seed=7
        )
        np.testing.assert_array_equal(first["selected_calls"], second["selected_calls"])
        np.testing.assert_array_equal(first["counts"], [10, 0, 10])
        self.assertEqual(int(first["selected_calls"].sum()), 40)
        self.assertTrue(first["audit"]["exact_total_match"])
        self.assertTrue(first["audit"]["expected_match"])

        shuffled = randomized_histogram_control(first["selected_calls"], seed=12)
        np.testing.assert_array_equal(np.sort(shuffled), np.sort(first["selected_calls"]))

    def test_infeasible_integer_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            strongest_transition_independent_baseline(
                [1.0, 0.5], [1, 3], n=3, target_total_calls=4, seed=0
            )


class StatisticalTests(unittest.TestCase):
    def test_clustered_bootstrap_is_deterministic_and_paired_direction_is_benefit(self):
        episodes = np.repeat(np.arange(5), 4)
        values = np.linspace(-0.2, 0.4, 20)
        one = clustered_bootstrap_ci(values, episodes, samples=200, seed=11)
        two = clustered_bootstrap_ci(values, episodes, samples=200, seed=11)
        self.assertEqual(one, two)
        self.assertEqual(one["n_episodes"], 5)

        candidate = np.full(20, 0.2)
        baseline = np.full(20, 0.3)
        benefit = clustered_paired_loss_ci(
            candidate, baseline, episodes, samples=100, seed=2
        )
        self.assertAlmostEqual(benefit["estimate"], 0.1)
        self.assertGreater(benefit["ci_low"], 0.0)

    def test_pareto_frontier(self):
        calls = [1.0, 1.5, 2.0, 2.0, 2.5]
        errors = [0.40, 0.30, 0.29, 0.35, 0.28]
        np.testing.assert_array_equal(
            nondominated_mask(calls, errors), [True, True, True, False, True]
        )
        np.testing.assert_array_equal(pareto_frontier(calls, errors), [0, 1, 2, 4])

    def test_permutation_null_is_label_free_and_preserves_marginals(self):
        scores = np.arange(36, dtype=float).reshape(12, 3)
        permuted = permute_critic_scores(scores, seed=5)
        for column in range(3):
            np.testing.assert_array_equal(
                np.sort(permuted[:, column]), np.sort(scores[:, column])
            )
        np.testing.assert_array_equal(permuted, permute_critic_scores(scores, seed=5))

        episodes = np.repeat(np.arange(3), 4)
        within = permute_critic_scores(scores, seed=6, episode_ids=episodes)
        for episode in range(3):
            index = episodes == episode
            for column in range(3):
                np.testing.assert_array_equal(
                    np.sort(within[index, column]), np.sort(scores[index, column])
                )

    def test_ordered_gain_ranking_and_calibration_metrics(self):
        predicted = np.linspace(-1.0, 1.0, 100)
        realized = predicted + 0.01 * np.sin(np.arange(100))
        result = gain_ranking_calibration(predicted, realized, bins=5)
        self.assertGreater(result["spearman_rho"], 0.99)
        self.assertGreater(result["roc_auc_positive_gain"], 0.99)
        self.assertTrue(result["ordered_quantiles"])
        self.assertGreater(result["top_bottom_realized_gain_gap"], 1.0)
        self.assertLess(result["calibration_rmse"], 0.02)
        self.assertEqual(sum(item["n"] for item in result["quantiles"]), 100)
        # Diagnostics are strict-JSON friendly, including undefined metrics.
        json.dumps(result, allow_nan=False)

        per_depth = per_depth_gain_diagnostics(
            np.column_stack((predicted, -predicted)),
            np.column_stack((realized, -realized)),
            bins=4,
        )
        self.assertEqual([item["decision_index"] for item in per_depth], [0, 1])

    def test_constant_scores_return_json_null_not_nan(self):
        result = gain_ranking_calibration(np.ones(8), np.arange(8), bins=2)
        self.assertIsNone(result["spearman_rho"])
        self.assertIsNone(result["calibration_slope"])
        json.dumps(result, allow_nan=False)


class ValidationTests(unittest.TestCase):
    def test_nonfinite_or_shape_errors_fail_closed(self):
        with self.assertRaises(ValueError):
            causal_sequential_stopping([[1.0, np.nan]])
        with self.assertRaises(ValueError):
            call_audit([1, 2.5])
        with self.assertRaises(ValueError):
            clustered_bootstrap_ci([1.0, 2.0], [0, 0])
        with self.assertRaises(ValueError):
            gain_ranking_calibration([1, 2], [1], bins=2)


if __name__ == "__main__":
    unittest.main()
