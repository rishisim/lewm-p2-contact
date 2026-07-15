import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metric_hardening_utils import (
    clustered_contrast_summary,
    cross_episode_neighbors,
    local_sensitivity_frame,
    per_dim_std,
    probability_superiority,
    scaled_mse,
)


class MetricHardeningUtilsTest(unittest.TestCase):
    def test_local_expansion_contracting_map(self) -> None:
        x = np.arange(20, dtype=np.float64).reshape(-1, 1)
        y = 0.5 * x
        residuals = np.zeros_like(x)
        episodes = np.arange(20)
        regimes = ["interaction" if i % 2 else "transport_free" for i in range(20)]
        frame = local_sensitivity_frame(
            x=x,
            y=y,
            residuals=residuals,
            episode_keys=episodes,
            regimes=regimes,
            row_ids=np.arange(20),
            k=4,
            k_min=2,
        )
        valid = frame.loc[frame["valid_neighbors"], "local_expansion"].to_numpy()
        self.assertGreater(len(valid), 0)
        self.assertLess(float(np.median(valid)), 1.0)

    def test_local_expansion_branching_map(self) -> None:
        x = np.zeros((12, 1), dtype=np.float64)
        y = np.asarray([[0.0] if i % 2 else [10.0] for i in range(12)], dtype=np.float64)
        residuals = np.zeros_like(x)
        episodes = np.arange(12)
        regimes = ["interaction"] * 12
        frame = local_sensitivity_frame(
            x=x,
            y=y,
            residuals=residuals,
            episode_keys=episodes,
            regimes=regimes,
            row_ids=np.arange(12),
            k=6,
            k_min=3,
        )
        valid = frame.loc[frame["valid_neighbors"], "local_expansion"].to_numpy()
        self.assertGreater(len(valid), 0)
        self.assertGreater(float(np.median(valid)), 1.0)

    def test_per_dim_target_std_normalization(self) -> None:
        calibration = np.asarray([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]])
        scale = per_dim_std(calibration)
        self.assertAlmostEqual(float(scale[1]), 1.0)
        delta = np.asarray([[2.0, 4.0]])
        expected = np.mean((delta / scale) ** 2, axis=1)
        actual = scaled_mse(delta, scale)
        self.assertTrue(np.allclose(actual, expected))

    def test_probability_superiority_bounds_and_ties(self) -> None:
        value = probability_superiority([2.0, 3.0], [1.0, 3.0])
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(value, 0.625)

    def test_cross_episode_neighbors_exclude_anchor_episode(self) -> None:
        x = np.arange(12, dtype=np.float64).reshape(-1, 1)
        episodes = np.repeat(np.arange(4), 3)
        result = cross_episode_neighbors(x, episodes, k=3, k_min=2)
        self.assertTrue(result.valid.all())
        for anchor in range(len(x)):
            idx = result.indices[anchor]
            idx = idx[idx >= 0]
            self.assertTrue(np.all(episodes[idx] != episodes[anchor]))

    def test_clustered_contrast_smoke(self) -> None:
        df = pd.DataFrame(
            {
                "episode_ordinal": [0, 0, 1, 1, 2, 2],
                "regime": ["interaction", "transport_free"] * 3,
                "metric": [2.0, 1.0, 3.0, 1.0, 4.0, 2.0],
            }
        )
        row = clustered_contrast_summary(
            df,
            left_mask=df["regime"].eq("interaction"),
            right_mask=df["regime"].eq("transport_free"),
            value_col="metric",
            n_bootstrap=50,
            seed=123,
        )
        self.assertTrue(np.isfinite(row["delta_trimmed_mean_ci_low"]))
        self.assertTrue(np.isfinite(row["delta_trimmed_mean_ci_high"]))
        self.assertGreaterEqual(row["p_superiority"], 0.0)
        self.assertLessEqual(row["p_superiority"], 1.0)


if __name__ == "__main__":
    unittest.main()
