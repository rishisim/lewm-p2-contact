import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qualification", ROOT / "qualify.py")
QUAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QUAL)


class QualificationTests(unittest.TestCase):
    def test_released_checkpoint_module_path_is_available(self):
        self.assertEqual(Path(sys.path[0]).resolve(), (QUAL.REPO / "le-wm").resolve())

    def test_valid_rows_support_noncontiguous_episode_ids(self):
        class Dataset:
            values = {
                "episode_idx": np.array([10, 10, 10, 30, 30]),
                "step_idx": np.array([0, 1, 2, 0, 1]),
            }

            def get_col_data(self, name):
                return self.values[name]

        rows, _, _ = QUAL.valid_rows(Dataset(), goal_offset=1)
        np.testing.assert_array_equal(rows, np.array([0, 1, 3]))

    def test_summary_reports_latency_and_variation(self):
        records = [
            {
                "success": False,
                "task_cost": 10.0,
                "normalized_return": 0.1,
                "latency_ns": 1_000_000,
            },
            {
                "success": True,
                "task_cost": 5.0,
                "normalized_return": 0.5,
                "latency_ns": 3_000_000,
            },
        ]
        result = QUAL.summarize(records)
        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["latency_ms_median"], 2.0)
        self.assertEqual(result["distinct_task_costs_1e6"], 2)


if __name__ == "__main__":
    unittest.main()
