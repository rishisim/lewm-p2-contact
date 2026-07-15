from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy import (calibrate_threshold, clustered_ci, histogram,
                    histogram_permutation, histogram_randomized, local_depths,
                    oracle_exact, strongest_mixture)


class PolicyTests(unittest.TestCase):
    def test_local_prefix_halting(self):
        scores = np.asarray([[2, 2, 2], [2, -1, 2], [-1, 2, 2], [2, 2, -1]], float)
        np.testing.assert_array_equal(local_depths(scores, 0), [4, 2, 1, 3])
        changed = scores.copy(); changed[1:] += 100
        self.assertEqual(local_depths(changed, 0)[0], local_depths(scores, 0)[0])

    def test_threshold_calibration_uses_only_supplied_calibration_arrays(self):
        scores = np.linspace(-1, 1, 30).reshape(10, 3)
        losses = np.ones((10, 5)); losses[:, 1:] -= np.arange(4) * 0.01
        result = calibrate_threshold(scores, losses, 2.0, 31)
        self.assertIn("threshold", result)
        self.assertTrue(math.isfinite(float(result["calibration_raw_mse"])))

    def test_exact_histograms_and_calls(self):
        adaptive = np.asarray([1, 2, 3, 4] * 10)
        for fn, seed in ((histogram_randomized, 2),):
            other = fn(adaptive, seed)
            self.assertEqual(histogram(other), histogram(adaptive))
            self.assertEqual(int(other.sum()), int(adaptive.sum()))
        episodes = np.repeat(np.arange(10), 4)
        perm = histogram_permutation(adaptive, episodes, 3)
        self.assertEqual(histogram(perm), histogram(adaptive))
        means = np.asarray([1.0, .9, .85, .82, .8])
        mix = strongest_mixture(40, int(adaptive.sum()), means, 4)
        self.assertEqual(int(mix.sum()), int(adaptive.sum()))

    def test_oracle_budget_and_clustered_ci_determinism(self):
        rng = np.random.default_rng(5)
        losses = rng.random((20, 5))
        depths = oracle_exact(losses, 41)
        self.assertEqual(int(depths.sum()), 41)
        values = rng.normal(size=20); episodes = np.repeat(np.arange(5), 4)
        self.assertEqual(clustered_ci(values, episodes, 20, 6), clustered_ci(values, episodes, 20, 6))

    def test_configs_are_strict_finite_json(self):
        for name in ("full_config.json", "smoke_config.json", "preregistration.json"):
            value = json.loads((ROOT / name).read_text())
            self.assertNotIn("NaN", json.dumps(value, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
