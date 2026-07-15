from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrity import assert_frozen, audit_npz_finite, state_tensor_hash, strict_json_dump


class IntegrityTests(unittest.TestCase):
    def test_strict_json_rejects_nonfinite_nested_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            with self.assertRaisesRegex(ValueError, "root.a\[1\]"):
                strict_json_dump(path, {"a": [1.0, float("nan")]})
            self.assertFalse(path.exists())

    def test_strict_json_round_trip_numpy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ok.json"
            strict_json_dump(path, {"x": np.asarray([1, 2]), "v": np.float64(0.5)})
            self.assertEqual(json.loads(path.read_text()), {"v": 0.5, "x": [1, 2]})

    def test_state_hash_changes_and_frozen_audit(self) -> None:
        module = torch.nn.Linear(2, 1).eval().requires_grad_(False)
        before = state_tensor_hash(module)
        assert_frozen(module, "module")
        with torch.no_grad():
            module.bias.add_(1)
        self.assertNotEqual(before, state_tensor_hash(module))
        module.train()
        with self.assertRaisesRegex(RuntimeError, "evaluation"):
            assert_frozen(module, "module")

    def test_npz_finite_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.npz"
            np.savez_compressed(path, x=np.asarray([1.0, 2.0]))
            self.assertEqual(audit_npz_finite(path)["x"]["shape"], [2])
            np.savez_compressed(path, x=np.asarray([np.inf]))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                audit_npz_finite(path)


if __name__ == "__main__":
    unittest.main()
