#!/usr/bin/env python3
"""Tests for Fetch experiment utility helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_utils import add_shifted_input_fullstate, bias_variance_components, probability_superiority
from run_recurrent_refiner import (
    assign_depths_from_thresholds,
    calibrate_depth_thresholds,
    uniform_curve_at_budget,
)


class ExperimentUtilsTest(unittest.TestCase):
    def test_shifted_input_fullstate_uses_previous_row(self) -> None:
        df = pd.DataFrame(
            [
                {"env_id": "env", "episode_id": 0, "step_idx": 0, "qpos_0": 10.0, "qvel_0": 1.0},
                {"env_id": "env", "episode_id": 0, "step_idx": 1, "qpos_0": 20.0, "qvel_0": 2.0},
                {"env_id": "env", "episode_id": 0, "step_idx": 2, "qpos_0": 30.0, "qvel_0": 3.0},
            ]
        )
        shifted, cols = add_shifted_input_fullstate(df)
        self.assertEqual(cols, ["input_qpos_0", "input_qvel_0"])
        self.assertTrue(pd.isna(shifted.loc[0, "input_qpos_0"]))
        self.assertEqual(float(shifted.loc[1, "input_qpos_0"]), 10.0)
        self.assertEqual(float(shifted.loc[2, "input_qpos_0"]), 20.0)
        self.assertEqual(float(shifted.loc[2, "input_qvel_0"]), 2.0)

    def test_probability_superiority_counts_ties_as_half(self) -> None:
        self.assertAlmostEqual(probability_superiority([2.0, 3.0], [1.0, 3.0]), 0.625)

    def test_bias_variance_components_decompose_mean_member_error(self) -> None:
        components = bias_variance_components(
            predictions=[[[1.0], [2.0]], [[3.0], [2.0]]],
            targets=[[0.0], [1.0]],
        )
        self.assertAlmostEqual(float(components.loc[0, "bias2_mse"]), 4.0)
        self.assertAlmostEqual(float(components.loc[0, "variance_mse"]), 1.0)
        self.assertAlmostEqual(float(components.loc[0, "heldout_error_mse"]), 5.0)
        self.assertAlmostEqual(float(components.loc[0, "decomposition_residual_mse"]), 0.0)
        self.assertAlmostEqual(float(components.loc[1, "bias2_mse"]), 1.0)
        self.assertAlmostEqual(float(components.loc[1, "variance_mse"]), 0.0)
        self.assertAlmostEqual(float(components.loc[1, "heldout_error_mse"]), 1.0)

    def test_depth_thresholds_assign_more_compute_to_higher_scores(self) -> None:
        scores = pd.Series(range(12), dtype=float).to_numpy()
        config = calibrate_depth_thresholds(scores, {1: 0.25, 2: 0.25, 4: 0.25, 8: 0.25})
        depths = assign_depths_from_thresholds(scores, config["thresholds"])
        self.assertLessEqual(depths[0], depths[-1])
        self.assertEqual(set(depths), {1, 2, 4, 8})

    def test_uniform_curve_at_budget_interpolates_adjacent_depths(self) -> None:
        mse = pd.DataFrame(
            {
                "1": [1.0, 3.0],
                "2": [2.0, 4.0],
                "4": [4.0, 8.0],
                "8": [8.0, 16.0],
            }
        )
        interpolated = uniform_curve_at_budget(mse, 3.0)
        self.assertEqual(interpolated.tolist(), [3.0, 6.0])


if __name__ == "__main__":
    unittest.main()
