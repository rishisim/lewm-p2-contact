from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_experiment as driver


class DriverProtocolTests(unittest.TestCase):
    def test_procedural_oracle_bootstrap_reoptimizes_and_is_seeded(self) -> None:
        raw = np.asarray(
            [
                [0.30, 0.20, 0.19],
                [0.30, 0.29, 0.10],
                [0.40, 0.20, 0.19],
                [0.40, 0.39, 0.11],
            ],
            dtype=np.float64,
        )
        # Deliberately make the whitened-optimal choices differ from raw. The
        # procedural sensitivity must still use each replicate's raw oracle.
        white = np.asarray(
            [
                [0.8, 0.7, 0.1],
                [0.8, 0.2, 0.7],
                [0.9, 0.8, 0.1],
                [0.9, 0.2, 0.8],
            ],
            dtype=np.float64,
        )
        episodes = np.asarray([10, 10, 20, 20])
        adaptive = np.asarray([2, 2, 2, 2])
        adaptive_raw = driver.allocation_lib.gather_depths(raw, adaptive)
        adaptive_white = driver.allocation_lib.gather_depths(white, adaptive)
        cfg = {"bootstrap_samples": 12, "bootstrap_seed": 260818}

        original = driver.allocation_lib.sequential_oracle_exact_budget_allocation
        calls: list[int] = []

        def audited(losses: np.ndarray) -> np.ndarray:
            allocation = original(losses)
            driver.allocation_lib.validate_sequential_exact_budget(allocation, len(losses))
            calls.append(len(losses))
            return allocation

        with mock.patch.object(
            driver.allocation_lib,
            "sequential_oracle_exact_budget_allocation",
            side_effect=audited,
        ):
            first = driver.procedural_oracle_comparison_rows(
                raw_losses=raw,
                whitened_losses=white,
                adaptive_raw=adaptive_raw,
                adaptive_white=adaptive_white,
                episode_ids=episodes,
                cfg=cfg,
            )
        self.assertEqual(calls, [4] * 13)  # 12 replicates plus full-test point.
        second = driver.procedural_oracle_comparison_rows(
            raw_losses=raw,
            whitened_losses=white,
            adaptive_raw=adaptive_raw,
            adaptive_white=adaptive_white,
            episode_ids=episodes,
            cfg=cfg,
        )
        self.assertEqual(first, second)

        by_key = {(row["comparison"], row["metric"]): row for row in first}
        full_oracle = original(raw)
        expected_white = np.mean(white[:, 1] - driver.allocation_lib.gather_depths(white, full_oracle))
        self.assertAlmostEqual(
            by_key[("oracle_advantage_vs_uniform_d2", "whitened")]["mean_benefit"],
            expected_white,
        )
        self.assertEqual(
            by_key[("oracle_advantage_vs_uniform_d2", "raw")]["bootstrap_procedure"],
            "episode_cluster_resample_then_reoptimize_raw_oracle",
        )

    def test_exact_configuration_schema_rejects_unknown_field(self) -> None:
        import json

        cfg = json.loads((ROOT / "full_config.json").read_text(encoding="utf-8"))
        driver.validate_configuration(cfg)
        cfg["posthoc_tuning"] = True
        with self.assertRaisesRegex(RuntimeError, "frozen exact schema"):
            driver.validate_configuration(cfg)


if __name__ == "__main__":
    unittest.main()
