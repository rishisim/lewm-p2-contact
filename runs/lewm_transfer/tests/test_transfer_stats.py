import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm_transfer_plots import write_all_figures
from lewm_transfer_stats import mean_dim_mse, probability_superiority


class TransferStatsTest(unittest.TestCase):
    def test_persistence_baseline_toy_case(self) -> None:
        latents = np.asarray(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 4.0],
            ]
        )
        persistence = mean_dim_mse(latents[1:], latents[:-1])
        self.assertTrue(np.allclose(persistence, [2.0, 8.0]))

    def test_probability_superiority_bounds_and_ties(self) -> None:
        value = probability_superiority([2.0, 3.0], [1.0, 3.0])
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(value, 0.625)

    def test_figures_generate(self) -> None:
        records = pd.DataFrame(
            [
                {
                    "row_id": i,
                    "episode_ordinal": i // 3,
                    "regime": ["interaction", "support", "free"][i % 3],
                    "normalized_position": (i % 10 + 0.5) / 10.0,
                    "phase_decile": i % 10,
                    "mse_model": 0.2 + 0.01 * i,
                    "mse_persistence": 0.1 + 0.005 * i,
                    "excess": 0.1 + 0.005 * i,
                }
                for i in range(60)
            ]
        )
        summary_rows = []
        for regime in ["interaction", "support", "free"]:
            for metric in ["mse_model", "mse_persistence", "excess"]:
                values = records.loc[records["regime"] == regime, metric]
                mean = float(values.mean())
                summary_rows.append(
                    {
                        "regime": regime,
                        "metric": metric,
                        "n": int(len(values)),
                        "n_episodes": 20,
                        "mean": mean,
                        "median": float(values.median()),
                        "trimmed_mean": mean,
                        "trimmed_mean_ci_low": mean - 0.01,
                        "trimmed_mean_ci_high": mean + 0.01,
                    }
                )
        regime_summary = pd.DataFrame(summary_rows)
        regime_contrasts = pd.DataFrame(
            [
                {
                    "contrast": "interaction_vs_free",
                    "metric": "excess",
                    "delta_trimmed_mean": 0.02,
                    "delta_trimmed_mean_ci_low": 0.01,
                    "delta_trimmed_mean_ci_high": 0.03,
                    "p_superiority": 0.6,
                }
            ]
        )
        pair_rows = pd.DataFrame(
            [
                {
                    "episode_ordinal": i,
                    "metric": "mse_model",
                    "left_median": 0.2 + i * 0.01,
                    "right_median": 0.15 + i * 0.01,
                    "delta": 0.05,
                }
                for i in range(8)
            ]
        )
        pair_summary = pd.DataFrame(
            [
                {
                    "metric": "mse_model",
                    "eligible_episodes": 8,
                    "fraction_episode_positive": 1.0,
                    "median_delta": 0.05,
                    "wilcoxon_p_greater": 0.01,
                }
            ]
        )
        residual_structure = pd.DataFrame(
            [
                {
                    "regime": regime,
                    "n": 20,
                    "directional_consistency": 0.1 + j * 0.02,
                    "directional_consistency_ci_low": 0.08,
                    "directional_consistency_ci_high": 0.14,
                    "effective_rank": 5.0 + j,
                    "effective_rank_ci_low": 4.0 + j,
                    "effective_rank_ci_high": 6.0 + j,
                }
                for j, regime in enumerate(["interaction", "support", "free"])
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            outputs = write_all_figures(
                records=records,
                regime_summary=regime_summary,
                regime_contrasts=regime_contrasts,
                within_episode_pairs=pair_rows,
                within_episode_summary=pair_summary,
                residual_structure=residual_structure,
                figure_dir=Path(tmp),
            )
            self.assertEqual(len(outputs), 6)
            for path in outputs.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

