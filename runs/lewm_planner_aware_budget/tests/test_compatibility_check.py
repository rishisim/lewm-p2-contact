import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cem_refiner_compatibility", ROOT / "compatibility_check.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CHECK
SPEC.loader.exec_module(CHECK)


class CompatibilityCheckTests(unittest.TestCase):
    def test_task_a_rollout_has_short_prefix_histories(self):
        self.assertEqual(CHECK.planner_history_lengths(), [1, 2, 3, 3, 3])

    def test_invalid_history_contract_fails(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            CHECK.planner_history_lengths(observation_history=0)

    @unittest.skipUnless(
        CHECK.DEFAULT_ARTIFACT_ROOT.is_dir(), "read-only PushT artifact unavailable"
    )
    def test_read_only_artifact_proves_adapter_blocker(self):
        result = CHECK.inspect_artifact(CHECK.DEFAULT_ARTIFACT_ROOT)
        self.assertFalse(result["compatible"])
        self.assertFalse(result["safe_to_implement_adapter"])
        self.assertEqual(result["incompatible_transition_indices"], [0, 1])
        self.assertEqual(result["first_layer_shape"], [256, 814])
        self.assertEqual(
            result["checkpoint_sha256"], CHECK.EXPECTED_CHECKPOINT_SHA256
        )
        self.assertFalse(
            result["action_normalization"]["planner_mapping_exported"]
        )


if __name__ == "__main__":
    unittest.main()
