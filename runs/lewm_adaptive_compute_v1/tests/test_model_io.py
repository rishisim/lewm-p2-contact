from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from model_io import deterministic_episode_split  # noqa: E402


class ModelIOTest(unittest.TestCase):
    def test_preregistered_selection_is_reproducible_disjoint_and_excludes_prior(self) -> None:
        kwargs = dict(
            total_episodes=10_000,
            excluded=range(30),
            seed=260713,
            train_count=420,
            calibration_count=90,
            test_count=90,
        )
        first = deterministic_episode_split(**kwargs)
        second = deterministic_episode_split(**kwargs)
        for name, count in (("train", 420), ("calibration", 90), ("test", 90)):
            self.assertEqual(len(first[name]), count)
            np.testing.assert_array_equal(first[name], second[name])
            self.assertFalse(set(first[name].tolist()) & set(range(30)))
        self.assertFalse(set(first["train"]) & set(first["calibration"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["calibration"]) & set(first["test"]))

    def test_selection_rejects_oversubscription(self) -> None:
        with self.assertRaisesRegex(ValueError, "Requested"):
            deterministic_episode_split(5, [0], 1, 3, 1, 1)


if __name__ == "__main__":
    unittest.main()

