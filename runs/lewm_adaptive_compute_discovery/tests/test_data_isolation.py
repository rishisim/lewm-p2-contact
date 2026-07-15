import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_isolation as iso


class IsolationTests(unittest.TestCase):
    def test_grouped_split_and_allowlists_exclude_test(self):
        pinned = iso.load_pinned_v3_episode_sets()
        split = iso.grouped_discovery_split(84, 261013)
        self.assertFalse(set(split["discovery_fit"]) & set(split["internal_validation"]))
        self.assertEqual(set(np.concatenate(tuple(split.values()))), set(pinned["train"]))
        for role in ("train", "calibration_once"):
            chosen = set(np.concatenate(tuple(iso.extraction_splits(role).values())))
            self.assertFalse(chosen & set(pinned["test"]))
        with self.assertRaises(ValueError):
            iso.extraction_splits("test")

    def test_calibration_receipt_is_one_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            iso.create_calibration_receipt(path, "abc")
            self.assertEqual(json.loads(path.read_text())["tournament_sha256"], "abc")
            with self.assertRaises(RuntimeError):
                iso.create_calibration_receipt(path, "abc")

    def test_cache_role_rejects_test_episode(self):
        pinned = iso.load_pinned_v3_episode_sets()
        arrays = {
            "episode_id": np.asarray([pinned["test"][0]]), "model_step": np.asarray([3]),
            "split": np.asarray([2]), "history": np.zeros((1,3,192)),
            "action": np.zeros((1,3,25)), "base_pred": np.zeros((1,192)),
            "target": np.zeros((1,192)),
        }
        with self.assertRaises(RuntimeError):
            iso.assert_isolated_cache(arrays, "train")


if __name__ == "__main__":
    unittest.main()
