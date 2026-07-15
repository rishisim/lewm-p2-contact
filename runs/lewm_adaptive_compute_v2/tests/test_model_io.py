from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_DIR.parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from model_io import (  # noqa: E402
    EXPECTED_V1_FULL_MANIFEST_SHA256,
    EXPECTED_V1_SMOKE_MANIFEST_SHA256,
    FROZEN_ACTION_MEAN,
    FROZEN_ACTION_STATS_ROWS,
    FROZEN_ACTION_STD,
    ModelContract,
    assert_frozen_module,
    deterministic_episode_split,
    extract_post_prediction_labels,
    load_frozen_refiner,
    load_prior_episode_exclusions,
    module_state_sha256,
    sha256_file,
    validate_cache_arrays,
    verify_file_hash,
)


class ModelIOTest(unittest.TestCase):
    def test_action_normalizer_is_frozen_not_fit_from_v2(self) -> None:
        self.assertEqual(FROZEN_ACTION_STATS_ROWS, 2_000_000)
        np.testing.assert_array_equal(
            FROZEN_ACTION_MEAN,
            np.asarray(
                [0.010884696617722511, -0.003141433000564575,
                 0.002646582666784525, 0.00042392866453155875,
                 0.1592525690793991],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            FROZEN_ACTION_STD,
            np.asarray(
                [0.28941991925239563, 0.39371708035469055,
                 0.6431366801261902, 0.3928017318248749,
                 0.25030744075775146],
                dtype=np.float32,
            ),
        )

    def test_frozen_prior_manifests_parse_to_exact_636_exclusions(self) -> None:
        full = REPO_ROOT / "runs/lewm_adaptive_compute_v1/cache/split_manifest.json"
        smoke = REPO_ROOT / "runs/lewm_adaptive_compute_v1/smoke_run/cache/split_manifest.json"
        exclusions, provenance = load_prior_episode_exclusions(full, smoke)
        self.assertEqual(sha256_file(full), EXPECTED_V1_FULL_MANIFEST_SHA256)
        self.assertEqual(sha256_file(smoke), EXPECTED_V1_SMOKE_MANIFEST_SHA256)
        self.assertEqual(len(exclusions), 636)
        self.assertEqual(len(np.unique(exclusions)), 636)
        np.testing.assert_array_equal(exclusions[:36], np.arange(36))
        self.assertFalse(set(range(36, 42)) & set(exclusions.tolist()))
        self.assertEqual(provenance["union_count"], 636)
        self.assertEqual(provenance["full_manifest"]["episode_count"], 600)
        self.assertEqual(provenance["smoke_manifest"]["episode_count"], 6)

    def test_manifest_hash_mismatch_is_rejected_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_file_hash(path, "0" * 64, label="test manifest")

    def test_small_selection_is_reproducible_disjoint_and_reserves_smoke(self) -> None:
        # Deliberately small synthetic universe: this test never generates the
        # preregistered 600-episode V2 split.
        kwargs = dict(
            total_episodes=30,
            excluded=range(5),
            reserved=range(5, 8),
            seed=260813,
            train_count=8,
            calibration_count=3,
            test_count=3,
        )
        first = deterministic_episode_split(**kwargs)
        second = deterministic_episode_split(**kwargs)
        selected: set[int] = set()
        for name, expected_count in (("train", 8), ("calibration", 3), ("test", 3)):
            self.assertEqual(len(first[name]), expected_count)
            np.testing.assert_array_equal(first[name], second[name])
            self.assertFalse(selected & set(first[name].tolist()))
            selected |= set(first[name].tolist())
        self.assertFalse(selected & set(range(8)))

    def test_selection_rejects_oversubscription_and_out_of_range_exclusion(self) -> None:
        with self.assertRaisesRegex(ValueError, "Requested"):
            deterministic_episode_split(5, [0], 1, 3, 1, 1)
        with self.assertRaisesRegex(ValueError, "outside source"):
            deterministic_episode_split(5, [5], 1, 1, 1, 1)

    @staticmethod
    def _valid_small_cache() -> tuple[dict[str, np.ndarray], dict[str, list[int]], ModelContract]:
        contract = ModelContract(
            latent_dim=2,
            history_size=2,
            frameskip=5,
            raw_action_dim=1,
            blocked_action_dim=3,
            image_size=8,
        )
        split_episodes = {"train": [2], "calibration": [4], "test": [5]}
        episode_id = np.repeat(np.asarray([2, 4, 5], dtype=np.int64), 3)
        model_step = np.tile(np.asarray([2, 3, 4], dtype=np.int64), 3)
        split = np.repeat(np.asarray([0, 1, 2], dtype=np.int8), 3)
        arrays = {
            "history": np.zeros((9, 2, 2), dtype=np.float32),
            "action": np.zeros((9, 2, 3), dtype=np.float32),
            "base_pred": np.zeros((9, 2), dtype=np.float32),
            "target": np.ones((9, 2), dtype=np.float32),
            "episode_id": episode_id,
            "model_step": model_step,
            "split": split,
        }
        return arrays, split_episodes, contract

    def test_cache_audit_enforces_shapes_keys_temporal_order_and_no_labels(self) -> None:
        arrays, splits, contract = self._valid_small_cache()
        audit = validate_cache_arrays(
            arrays,
            split_episodes=splits,
            contract=contract,
            expected_examples_per_episode=3,
        )
        self.assertEqual(audit["examples"], 9)
        self.assertEqual(audit["split_examples"], {"train": 3, "calibration": 3, "test": 3})

        with_label = dict(arrays, interaction=np.zeros(9, dtype=bool))
        with self.assertRaisesRegex(RuntimeError, "must be exactly"):
            validate_cache_arrays(
                with_label,
                split_episodes=splits,
                contract=contract,
                expected_examples_per_episode=3,
            )

        bad_order = dict(arrays, model_step=arrays["model_step"].copy())
        bad_order["model_step"][:3] = [3, 2, 4]
        with self.assertRaisesRegex(RuntimeError, "temporal order"):
            validate_cache_arrays(
                bad_order,
                split_episodes=splits,
                contract=contract,
                expected_examples_per_episode=3,
            )

    def test_cache_audit_rejects_nonfinite_targets_and_wrong_episode_set(self) -> None:
        arrays, splits, contract = self._valid_small_cache()
        nonfinite = dict(arrays, target=arrays["target"].copy())
        nonfinite["target"][0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            validate_cache_arrays(
                nonfinite,
                split_episodes=splits,
                contract=contract,
                expected_examples_per_episode=3,
            )
        wrong_splits = {"train": [2], "calibration": [4], "test": [6]}
        with self.assertRaisesRegex(RuntimeError, "planned split"):
            validate_cache_arrays(
                arrays,
                split_episodes=wrong_splits,
                contract=contract,
                expected_examples_per_episode=3,
            )

    def test_frozen_module_audit_detects_state_mutation_and_gradients(self) -> None:
        module = torch.nn.Linear(3, 2).eval().requires_grad_(False)
        before = assert_frozen_module(module, label="toy")
        self.assertEqual(before.state_sha256, module_state_sha256(module))
        with torch.no_grad():
            module.weight[0, 0].add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "audit changed"):
            assert_frozen_module(module, expected=before, label="toy")
        module.weight.grad = torch.zeros_like(module.weight)
        with self.assertRaisesRegex(RuntimeError, "gradients"):
            assert_frozen_module(module, label="toy")

    def test_refiner_load_is_strict_hash_checked_and_frozen(self) -> None:
        source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.GELU(), torch.nn.Linear(4, 2))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "refiner.pt"
            torch.save({"state_dict": source.state_dict(), "seed": 7}, checkpoint)
            target = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.GELU(), torch.nn.Linear(4, 2))
            loaded, provenance = load_frozen_refiner(
                target,
                checkpoint,
                torch.device("cpu"),
                expected_sha256=sha256_file(checkpoint),
                expected_seed=7,
                expected_parameter_count=None,
            )
            self.assertFalse(loaded.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
            self.assertTrue(provenance["strict_key_match"])
            for key, value in source.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)

            wrong = torch.nn.Linear(3, 2)
            with self.assertRaisesRegex(RuntimeError, "Strict refiner checkpoint mismatch"):
                load_frozen_refiner(
                    wrong,
                    checkpoint,
                    torch.device("cpu"),
                    expected_sha256=sha256_file(checkpoint),
                    expected_seed=7,
                    expected_parameter_count=None,
                )

    def test_physical_signals_are_extracted_only_by_post_prediction_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "cube.h5"
            with h5py.File(source, "w") as h5:
                h5["ep_offset"] = np.asarray([0, 11], dtype=np.int64)
                h5["ep_len"] = np.asarray([11, 11], dtype=np.int64)
                h5["ep_idx"] = np.repeat(np.asarray([0, 1], dtype=np.int64), 11)
                contact = np.zeros((22, 1), dtype=np.float32)
                contact[4:6, 0] = 1.0
                h5["proprio_gripper_contact"] = contact
                effector = np.zeros((22, 3), dtype=np.float32)
                block = np.zeros((22, 3), dtype=np.float32)
                for episode, offset in enumerate((0, 11)):
                    effector[offset : offset + 11, 0] = np.arange(11, dtype=np.float32)
                    block[offset : offset + 11, 1] = 2 * np.arange(11, dtype=np.float32)
                h5["proprio_effector_pos"] = effector
                h5["privileged_block_0_pos"] = block

            labels = extract_post_prediction_labels(
                source,
                episode_id=np.asarray([1, 0], dtype=np.int64),
                model_step=np.asarray([2, 1], dtype=np.int64),
                frameskip=5,
            )
            self.assertEqual(set(labels), {
                "interaction", "impact", "effector_disp", "block_disp", "normalized_phase"
            })
            np.testing.assert_array_equal(labels["interaction"], [False, True])
            np.testing.assert_array_equal(labels["impact"], [False, True])
            np.testing.assert_allclose(labels["effector_disp"], [5.0, 5.0])
            np.testing.assert_allclose(labels["block_disp"], [10.0, 10.0])
            np.testing.assert_allclose(labels["normalized_phase"], [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
