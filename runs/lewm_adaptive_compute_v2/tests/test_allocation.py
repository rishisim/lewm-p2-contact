from __future__ import annotations

import inspect
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from allocation import (  # noqa: E402
    CONTINUATION_COSTS,
    SEQUENTIAL_DEPTHS,
    continuation_block_calls,
    depth_histogram,
    episode_clustered_paired_bootstrap_ci,
    fit_calibration_whitening,
    gather_depths,
    latent_mse_by_depth,
    marginal_continuation_benefits,
    paired_cluster_bootstrap,
    permuted_allocation,
    random_sequential_exact_budget_allocation,
    sequential_exact_budget_allocation,
    sequential_oracle_exact_budget_allocation,
    summarize_allocation,
    total_block_calls,
    uniform_allocation,
    validate_episode_split_isolation,
    validate_exact_budget,
    validate_sequential_exact_budget,
    whitened_mse,
)


class SequentialAllocationTest(unittest.TestCase):
    def test_frozen_depths_costs_and_actual_depth_gather(self) -> None:
        self.assertEqual(SEQUENTIAL_DEPTHS, (1, 2, 4))
        self.assertEqual(CONTINUATION_COSTS, (0, 1, 3))
        values = np.asarray(
            [[11.0, 12.0, 14.0], [21.0, 22.0, 24.0], [31.0, 32.0, 34.0]]
        )
        np.testing.assert_array_equal(gather_depths(values, [4, 1, 2]), [14.0, 21.0, 32.0])
        self.assertEqual(total_block_calls([4, 1, 2]), 7)
        self.assertEqual(continuation_block_calls([4, 1, 2]), 4)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            gather_depths(values, [3, 1, 2])

    def test_primary_optimizer_signature_is_target_free_and_budget_fixed(self) -> None:
        parameters = set(inspect.signature(sequential_exact_budget_allocation).parameters)
        self.assertEqual(parameters, {"predicted_marginal_utilities", "n_samples"})
        self.assertFalse(parameters & {"target", "targets", "loss", "losses", "budget"})

        predicted = np.asarray(
            [
                [0.1, 10.0],
                [0.3, 0.2],
                [0.2, 0.1],
                [-1.0, -1.0],
            ]
        )
        allocation = sequential_exact_budget_allocation(predicted, n_samples=4)
        validate_sequential_exact_budget(allocation, 4)
        self.assertEqual(total_block_calls(allocation), 8)
        self.assertEqual(continuation_block_calls(allocation), 4)
        self.assertEqual(set(allocation.tolist()) - set(SEQUENTIAL_DEPTHS), set())

    def test_primary_optimizer_matches_global_bruteforce_optimum(self) -> None:
        predicted = np.asarray(
            [
                [0.2, 1.5],
                [0.8, -0.4],
                [-0.1, 0.5],
                [0.6, 0.7],
                [0.0, -1.0],
            ]
        )
        allocation = sequential_exact_budget_allocation(predicted)
        costs = dict(zip(SEQUENTIAL_DEPTHS, CONTINUATION_COSTS, strict=True))

        def objective(candidate: tuple[int, ...]) -> float:
            result = 0.0
            for row, depth in enumerate(candidate):
                if depth == 2:
                    result += predicted[row, 0]
                elif depth == 4:
                    result += predicted[row, 1]
            return result

        feasible = [
            candidate
            for candidate in itertools.product(SEQUENTIAL_DEPTHS, repeat=len(predicted))
            if sum(costs[depth] for depth in candidate) == len(predicted)
        ]
        self.assertTrue(feasible)
        optimum = max(objective(candidate) for candidate in feasible)
        self.assertAlmostEqual(objective(tuple(allocation.tolist())), optimum, places=12)

    def test_optimizer_terminal_b14_is_not_added_to_b12(self) -> None:
        # A depth-4 utility is L1-L4 directly. It is not b12+b14.
        predicted = np.asarray([[100.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
        allocation = sequential_exact_budget_allocation(predicted)
        self.assertEqual(total_block_calls(allocation), 6)
        self.assertEqual(continuation_block_calls(allocation), 3)
        # Spending all continuation calls on either depth-4 row has utility 2,
        # whereas three depth-2 choices have utility 100.
        np.testing.assert_array_equal(allocation, [2, 2, 2])

    def test_ties_are_deterministic_and_edge_cases_are_exact(self) -> None:
        zeros = np.zeros((6, 2), dtype=np.float64)
        first = sequential_exact_budget_allocation(zeros)
        second = sequential_exact_budget_allocation(zeros.copy())
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            sequential_exact_budget_allocation(np.zeros((3, 2))),
            [1, 1, 4],
        )
        validate_sequential_exact_budget(first, 6)

        np.testing.assert_array_equal(sequential_exact_budget_allocation(np.zeros((1, 2))), [2])
        empty = sequential_exact_budget_allocation(np.empty((0, 2)), n_samples=0)
        self.assertEqual(empty.shape, (0,))
        with self.assertRaisesRegex(ValueError, "sample-count mismatch"):
            sequential_exact_budget_allocation(np.zeros((2, 2)), n_samples=3)
        with self.assertRaisesRegex(ValueError, "shape"):
            sequential_exact_budget_allocation(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            sequential_exact_budget_allocation(np.asarray([[0.0, np.nan]]))

    def test_oracle_is_sequentially_feasible_and_globally_optimal(self) -> None:
        losses = np.asarray(
            [
                [4.0, 3.0, 0.0],
                [2.0, 1.9, 1.8],
                [3.0, 2.0, 2.5],
                [1.0, 1.1, 1.2],
            ]
        )
        oracle = sequential_oracle_exact_budget_allocation(losses)
        validate_sequential_exact_budget(oracle, len(losses))
        oracle_loss = float(gather_depths(losses, oracle).sum())
        costs = dict(zip(SEQUENTIAL_DEPTHS, CONTINUATION_COSTS, strict=True))
        feasible_losses = []
        for candidate in itertools.product(SEQUENTIAL_DEPTHS, repeat=len(losses)):
            if sum(costs[depth] for depth in candidate) == len(losses):
                feasible_losses.append(float(gather_depths(losses, candidate).sum()))
        self.assertAlmostEqual(oracle_loss, min(feasible_losses), places=12)
        uniform_loss = float(gather_depths(losses, uniform_allocation(len(losses), 2)).sum())
        self.assertLessEqual(oracle_loss, uniform_loss)

    def test_marginal_labels_are_l1_minus_l2_and_l1_minus_l4(self) -> None:
        losses = np.asarray([[5.0, 3.0, 4.0], [2.0, 4.0, 1.0]])
        np.testing.assert_array_equal(
            marginal_continuation_benefits(losses),
            [[2.0, 1.0], [-2.0, 1.0]],
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            marginal_continuation_benefits(np.asarray([[1.0, np.inf, 0.0]]))

    def test_random_baseline_is_seeded_target_free_and_exact(self) -> None:
        parameters = set(inspect.signature(random_sequential_exact_budget_allocation).parameters)
        self.assertEqual(parameters, {"n_samples", "rng"})
        first = random_sequential_exact_budget_allocation(100, rng=260816)
        second = random_sequential_exact_budget_allocation(100, rng=260816)
        np.testing.assert_array_equal(first, second)
        validate_sequential_exact_budget(first, 100)
        self.assertEqual(total_block_calls(first), 200)
        self.assertEqual(sum(depth_histogram(first).values()), 100)
        for n in [0, 1, 2, 3, 10, 31]:
            allocation = random_sequential_exact_budget_allocation(n, rng=n + 1)
            validate_sequential_exact_budget(allocation, n)

    def test_permutation_preserves_adaptive_histogram_and_exact_budget(self) -> None:
        adaptive = np.asarray([1, 1, 1, 2, 2, 2, 4, 4])
        # continuation calls = 3*0 + 3*1 + 2*3 = 9, so this deliberately
        # tests histogram/call preservation independently of the 2N contract.
        permuted = permuted_allocation(adaptive, rng=260817)
        self.assertEqual(depth_histogram(permuted), depth_histogram(adaptive))
        self.assertEqual(total_block_calls(permuted), total_block_calls(adaptive))
        np.testing.assert_array_equal(permuted, permuted_allocation(adaptive, rng=260817))

        exact = sequential_exact_budget_allocation(np.arange(20, dtype=float).reshape(10, 2))
        exact_permutation = permuted_allocation(exact, rng=260817)
        validate_sequential_exact_budget(exact_permutation, 10)

    def test_uniform_and_budget_validators_count_actual_calls(self) -> None:
        fixed1 = uniform_allocation(7, 1)
        uniform2 = uniform_allocation(7, 2)
        self.assertEqual(total_block_calls(fixed1), 7)
        self.assertEqual(total_block_calls(uniform2), 14)
        validate_exact_budget(uniform2, 14)
        validate_sequential_exact_budget(uniform2, 7)
        with self.assertRaisesRegex(ValueError, "budget mismatch"):
            validate_exact_budget(uniform2, 7)
        with self.assertRaisesRegex(ValueError, "sequential budget mismatch"):
            validate_sequential_exact_budget(fixed1, 7)

    def test_whitening_is_calibration_fit_symmetric_and_frozen(self) -> None:
        calibration = np.asarray(
            [[-2.0, -0.02], [-1.0, 0.01], [1.0, -0.01], [2.0, 0.02]]
        )
        targets = np.asarray([[1000.0, 1000.0], [-1000.0, -1000.0]])
        exits = np.stack(
            [
                targets + [1.0, 0.0],
                targets + [0.0, 1.0],
                targets + [1.0, 1.0],
            ],
            axis=1,
        )
        transform = fit_calibration_whitening(calibration)
        np.testing.assert_allclose(transform.matrix, transform.matrix.T)
        frozen = transform.matrix.copy()
        losses = latent_mse_by_depth(targets, exits, whitening=transform)
        manual = np.mean(((exits - targets[:, None, :]) @ frozen) ** 2, axis=2)
        np.testing.assert_allclose(losses, manual)
        np.testing.assert_allclose(whitened_mse(exits[:, 0], targets, transform), manual[:, 0])
        np.testing.assert_array_equal(transform.matrix, frozen)
        self.assertEqual(transform.n_calibration, 4)
        self.assertAlmostEqual(
            transform.floor,
            float(transform.eigenvalues.max()) * 1e-6,
        )

    def test_clustered_bootstrap_is_finite_seeded_and_positive(self) -> None:
        left = np.asarray([2.0, 2.5, 4.0, 4.5, 7.0, 7.5])
        right = np.asarray([1.0, 2.0, 2.0, 3.0, 4.0, 4.5])
        episodes = np.repeat([80, 81, 82], 2)
        first = episode_clustered_paired_bootstrap_ci(
            left, right, episodes, n_bootstrap=200, seed=260818
        )
        second = paired_cluster_bootstrap(
            left - right, episodes, n_bootstrap=200, seed=260818
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.estimate, float(np.mean(left - right)))
        self.assertGreater(first.lower, 0.0)
        self.assertGreaterEqual(first.upper, first.lower)
        self.assertEqual(first.n_episodes, 3)
        self.assertEqual(first.n_samples, 6)
        self.assertTrue(all(np.isfinite(value) for value in first.as_dict().values()))

    def test_summary_reports_first_continuation_and_total_calls(self) -> None:
        losses = np.asarray([[4.0, 3.0, 1.0], [8.0, 7.0, 5.0]])
        summary = summarize_allocation(losses, [4, 1])
        self.assertEqual(summary["first_calls"], 2)
        self.assertEqual(summary["continuation_calls"], 3)
        self.assertEqual(summary["total_calls"], 5)
        self.assertEqual(summary["depth_histogram"], {1: 1, 2: 0, 4: 1})
        self.assertAlmostEqual(summary["mean_loss"], 4.5)
        self.assertAlmostEqual(summary["mean_benefit_vs_depth1"], 1.5)

    def test_split_isolation_rejects_overlap(self) -> None:
        validate_episode_split_isolation(range(100), range(100, 120), range(120, 140))
        with self.assertRaisesRegex(ValueError, "train/calibration"):
            validate_episode_split_isolation([1, 2], [2, 3], [4])


if __name__ == "__main__":
    unittest.main()
