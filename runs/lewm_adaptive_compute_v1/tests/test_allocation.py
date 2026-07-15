from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from allocation import (  # noqa: E402
    depth_histogram,
    episode_clustered_paired_bootstrap_ci,
    exact_budget_optimal_allocation,
    fit_calibration_whitening,
    gather_depths,
    latent_mse_by_depth,
    oracle_exact_budget_allocation,
    permuted_allocation,
    predicted_gain_exact_budget_allocation,
    random_exact_budget_allocation,
    robust_relative_gain,
    summarize_allocation,
    total_block_calls,
    uniform_allocation,
    validate_episode_split_isolation,
    validate_split_isolation,
    validate_exact_budget,
    whitened_mse,
)


class AllocationTest(unittest.TestCase):
    def test_gather_uses_depth_cost_not_exit_index(self) -> None:
        values = np.asarray(
            [
                [10.0, 11.0, 12.0, 14.0],
                [20.0, 21.0, 22.0, 24.0],
                [30.0, 31.0, 32.0, 34.0],
                [40.0, 41.0, 42.0, 44.0],
            ]
        )
        allocation = np.asarray([4, 0, 2, 1])
        np.testing.assert_array_equal(gather_depths(values, allocation), [14, 20, 32, 41])
        # Column indices sum to six; actual block-call depths sum to seven.
        self.assertEqual(total_block_calls(allocation), 7)
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            validate_exact_budget(allocation, 6)
        np.testing.assert_array_equal(validate_exact_budget(allocation, 7), allocation)

    def test_numpy_exit_gather_preserves_trailing_dimensions(self) -> None:
        exits = np.arange(3 * 4 * 2, dtype=np.float64).reshape(3, 4, 2)
        selected = gather_depths(exits, [4, 1, 0])
        expected = np.stack([exits[0, 3], exits[1, 1], exits[2, 0]])
        np.testing.assert_array_equal(selected, expected)
        with self.assertRaisesRegex(ValueError, "allocation length"):
            gather_depths(exits, [0, 1])
        with self.assertRaisesRegex(ValueError, "unavailable"):
            gather_depths(exits, [0, 3, 1])

    def test_oracle_matches_exact_budget_and_global_bruteforce_optimum(self) -> None:
        rng = np.random.default_rng(12)
        losses = rng.normal(size=(5, 4))
        allocation = oracle_exact_budget_allocation(losses, total_budget=7)
        self.assertEqual(total_block_calls(allocation), 7)

        choices = np.asarray([0, 1, 2, 4])
        brute_objectives = []
        brute_allocations = []
        for flat_index in range(4 ** len(losses)):
            digits = []
            remainder = flat_index
            for _ in range(len(losses)):
                digits.append(remainder % 4)
                remainder //= 4
            candidate = choices[np.asarray(digits)]
            if int(candidate.sum()) == 7:
                brute_allocations.append(candidate)
                brute_objectives.append(float(gather_depths(losses, candidate).sum()))
        self.assertTrue(brute_objectives)
        oracle_objective = float(gather_depths(losses, allocation).sum())
        self.assertAlmostEqual(oracle_objective, min(brute_objectives), places=12)
        positional = exact_budget_optimal_allocation(losses, [0, 1, 2, 4], 7)
        self.assertAlmostEqual(float(gather_depths(losses, positional).sum()), min(brute_objectives))

    def test_oracle_beats_or_equals_uniform_on_heterogeneous_synthetic_case(self) -> None:
        # Both allocations cost N=4 calls. Uniform depth 1 cannot target the one
        # sample with a large depth-4 benefit, while the oracle can spend 4+0+0+0.
        losses = np.asarray(
            [
                [10.0, 9.9, 9.8, 0.0],
                [1.0, 0.9, 0.8, 0.7],
                [1.0, 0.9, 0.8, 0.7],
                [1.0, 0.9, 0.8, 0.7],
            ]
        )
        uniform = uniform_allocation(len(losses), 1)
        oracle = oracle_exact_budget_allocation(losses, total_budget=len(losses))
        self.assertEqual(total_block_calls(oracle), total_block_calls(uniform))
        self.assertLessEqual(
            float(gather_depths(losses, oracle).sum()),
            float(gather_depths(losses, uniform).sum()),
        )
        self.assertLess(
            float(gather_depths(losses, oracle).sum()),
            float(gather_depths(losses, uniform).sum()),
        )

    def test_predicted_gain_optimizer_is_exact_and_uses_positive_depth_columns(self) -> None:
        # Input deliberately has only depth-1/2/4 predicted gains. The optimizer
        # inserts a zero depth-0 gain and spends four calls on sample 0.
        predicted = np.asarray(
            [
                [1.0, 2.0, 10.0],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
            ]
        )
        allocation = predicted_gain_exact_budget_allocation(predicted, 4)
        np.testing.assert_array_equal(allocation, [4, 0, 0, 0])
        self.assertEqual(total_block_calls(allocation), 4)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            predicted_gain_exact_budget_allocation(np.asarray([[1.0, np.nan, 2.0]]), 1)

    def test_random_allocation_exact_budget_and_target_free_signature(self) -> None:
        first = random_exact_budget_allocation(40, 40, rng=260713)
        second = random_exact_budget_allocation(40, 40, rng=260713)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(total_block_calls(first), 40)
        self.assertEqual(len(first), 40)
        # Check several nonuniform and boundary budgets, including actual depth 4.
        for budget in [0, 1, 7, 39, 40, 80, 157, 160]:
            allocation = random_exact_budget_allocation(40, budget, rng=budget + 1)
            self.assertEqual(total_block_calls(allocation), budget)
        explicit = random_exact_budget_allocation(40, [0, 1, 2, 4], 40, 260713)
        self.assertEqual(total_block_calls(explicit), 40)
        # For one sample, three calls are impossible with exits {0,1,2,4}.
        with self.assertRaisesRegex(ValueError, "infeasible"):
            random_exact_budget_allocation(1, 3, rng=0)

    def test_permutation_preserves_histogram_and_calls(self) -> None:
        allocation = np.asarray([0, 0, 1, 1, 1, 2, 2, 4])
        permuted = permuted_allocation(allocation, rng=21)
        self.assertEqual(depth_histogram(permuted), depth_histogram(allocation))
        self.assertEqual(total_block_calls(permuted), total_block_calls(allocation))
        self.assertEqual(len(permuted), len(allocation))

    def test_split_isolation_rejects_any_episode_overlap(self) -> None:
        validate_episode_split_isolation(range(30, 70), range(70, 80), range(80, 90))
        validate_split_isolation(
            {"train": range(30, 70), "calibration": range(70, 80), "test": range(80, 90)}
        )
        with self.assertRaisesRegex(ValueError, "train/calibration"):
            validate_episode_split_isolation([30, 31], [31, 70], [80])
        with self.assertRaisesRegex(ValueError, "calibration/test"):
            validate_episode_split_isolation([30], [70, 80], [80])

    def test_whitening_is_fit_from_calibration_and_frozen_on_evaluation(self) -> None:
        calibration = np.asarray(
            [
                [-2.0, -0.02],
                [-1.0, 0.01],
                [1.0, -0.01],
                [2.0, 0.02],
            ]
        )
        evaluation_targets = np.asarray([[1000.0, 1000.0], [-1000.0, -1000.0]])
        exits = np.stack(
            [
                evaluation_targets + np.asarray([1.0, 0.0]),
                evaluation_targets + np.asarray([0.0, 1.0]),
                evaluation_targets,
                evaluation_targets + np.asarray([1.0, 1.0]),
            ],
            axis=1,
        )
        transform = fit_calibration_whitening(calibration)
        frozen_matrix = transform.matrix.copy()
        losses = latent_mse_by_depth(evaluation_targets, exits, whitening=transform)
        manual = np.mean(
            ((exits - evaluation_targets[:, None, :]) @ frozen_matrix) ** 2,
            axis=2,
        )
        np.testing.assert_allclose(losses, manual)
        np.testing.assert_allclose(
            whitened_mse(exits[:, 0], evaluation_targets, transform),
            manual[:, 0],
        )
        np.testing.assert_array_equal(transform.matrix, frozen_matrix)
        np.testing.assert_allclose(transform.mean, calibration.mean(axis=0))
        self.assertEqual(transform.n_calibration, len(calibration))

        # Refitting (incorrectly) on evaluation targets gives another transform;
        # the evaluation helper never does this implicitly.
        leaked = fit_calibration_whitening(evaluation_targets)
        self.assertFalse(np.allclose(transform.matrix, leaked.matrix))

    def test_whitening_floor_and_robust_relative_gain(self) -> None:
        calibration_targets = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        transform = fit_calibration_whitening(calibration_targets)
        self.assertAlmostEqual(
            transform.floor,
            float(transform.eigenvalues.max()) * 1e-6,
        )
        self.assertTrue(np.isfinite(transform.matrix).all())

        losses = np.asarray([[0.0, 0.0, 0.0, 0.0], [2.0, 1.0, 3.0, 0.0]])
        calibration_loss0 = np.asarray([1.0, 2.0, 3.0, 4.0])
        relative = robust_relative_gain(losses, calibration_loss0)
        q10 = float(np.quantile(calibration_loss0, 0.1))
        np.testing.assert_allclose(relative[0], np.zeros(4))
        np.testing.assert_allclose(relative[1], [0.0, 0.5, -0.5, 1.0])
        self.assertGreater(q10, 1e-8)  # exercises the calibration quantile floor

    def test_episode_clustered_paired_bootstrap_is_finite_and_clustered(self) -> None:
        # Every paired benefit is positive, so every episode-resampled mean and
        # both percentile bounds must be positive.
        left = np.asarray([2.0, 2.5, 4.0, 4.5, 7.0, 7.5])
        right = np.asarray([1.0, 2.0, 2.0, 3.0, 4.0, 4.5])
        episodes = np.repeat([80, 81, 82], 2)
        ci = episode_clustered_paired_bootstrap_ci(
            left,
            right,
            episodes,
            n_bootstrap=200,
            seed=260713,
        )
        self.assertAlmostEqual(ci.estimate, float(np.mean(left - right)))
        self.assertGreater(ci.lower, 0.0)
        self.assertGreaterEqual(ci.upper, ci.lower)
        self.assertEqual(ci.n_episodes, 3)
        self.assertEqual(ci.n_samples, 6)

    def test_concise_summary_uses_gathered_loss_and_actual_calls(self) -> None:
        losses = np.asarray([[4.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0]])
        summary = summarize_allocation(losses, [4, 0])
        self.assertEqual(summary["total_calls"], 4)
        self.assertEqual(summary["depth_histogram"], {0: 1, 1: 0, 2: 0, 4: 1})
        self.assertAlmostEqual(summary["mean_loss"], 4.5)
        self.assertAlmostEqual(summary["mean_gain_vs_depth0"], 1.5)


if __name__ == "__main__":
    unittest.main()
