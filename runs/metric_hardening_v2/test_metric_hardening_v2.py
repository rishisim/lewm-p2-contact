import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metric_hardening_v2_utils import (
    apply_delta_transform,
    balanced_joint_features,
    contrast_values,
    cube_augmented_kinematics_from_arrays,
    cross_episode_neighbors,
    cube_action_blocks_from_arrays,
    fit_whitening_transform,
    k_sweep_pass,
    local_sensitivity_frame,
    per_dim_std,
    probability_superiority,
)


SOURCE_H5 = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")


def synthetic_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [0, 1],
            "episode_ordinal": [0, 0],
            "episode_id": [7, 7],
            "model_step": [3, 4],
            "raw_step": [15, 20],
            "transition_block": [2, 3],
        }
    )


class MetricHardeningV2Test(unittest.TestCase):
    def test_cube_action_join_correct_blocks(self) -> None:
        actions = np.arange(40 * 5, dtype=np.float64).reshape(40, 5)
        offsets = np.asarray([0], dtype=np.int64)
        lengths = np.asarray([40], dtype=np.int64)
        episode_ids = np.asarray([7], dtype=np.int64)
        blocks = cube_action_blocks_from_arrays(
            synthetic_records(),
            actions=actions,
            offsets=offsets,
            lengths=lengths,
            episode_ids=episode_ids,
        )
        self.assertEqual(blocks.shape, (2, 25))
        self.assertTrue(np.array_equal(blocks[0], actions[10:15].reshape(-1)))
        self.assertTrue(np.array_equal(blocks[1], actions[15:20].reshape(-1)))

    def test_cube_action_join_rejects_bad_keys(self) -> None:
        actions = np.zeros((40, 5), dtype=np.float64)
        offsets = np.asarray([0], dtype=np.int64)
        lengths = np.asarray([40], dtype=np.int64)
        episode_ids = np.asarray([7], dtype=np.int64)
        duplicate = pd.concat([synthetic_records().iloc[[0]], synthetic_records().iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            cube_action_blocks_from_arrays(duplicate, actions=actions, offsets=offsets, lengths=lengths, episode_ids=episode_ids)
        off_by_one = synthetic_records().copy()
        off_by_one.loc[0, "raw_step"] = 16
        with self.assertRaises(ValueError):
            cube_action_blocks_from_arrays(off_by_one, actions=actions, offsets=offsets, lengths=lengths, episode_ids=episode_ids)
        bad_episode = synthetic_records().copy()
        bad_episode.loc[0, "episode_id"] = 99
        with self.assertRaises(ValueError):
            cube_action_blocks_from_arrays(bad_episode, actions=actions, offsets=offsets, lengths=lengths, episode_ids=episode_ids)

    def test_augmented_kinematics_uses_input_time_and_finite_difference(self) -> None:
        records = synthetic_records().iloc[[0]].copy()
        n = 40
        eff = np.zeros((n, 3), dtype=np.float64)
        block = np.zeros((n, 3), dtype=np.float64)
        quat = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (n, 1))
        eff[10] = [1.0, 2.0, 3.0]
        eff[5] = [0.5, 1.0, 1.5]
        eff[15] = [9.0, 9.0, 9.0]
        block[10] = [4.0, 5.0, 6.0]
        block[5] = [3.0, 3.0, 3.0]
        qvel = np.ones((n, 2), dtype=np.float64)
        grip = np.arange(n, dtype=np.float64).reshape(n, 1)
        values, provenance = cube_augmented_kinematics_from_arrays(
            records,
            effector_pos=eff,
            block_pos=block,
            block_quat=quat,
            offsets=np.asarray([0], dtype=np.int64),
            lengths=np.asarray([n], dtype=np.int64),
            native_velocity_arrays={"qvel": qvel},
            gripper_arrays={"opening": grip},
        )
        self.assertEqual(values.shape[0], 1)
        self.assertTrue(np.allclose(values[0, :3], eff[10]))
        self.assertTrue(np.allclose(values[0, 10:13], eff[10] - eff[5]))
        self.assertFalse(np.allclose(values[0, :3], eff[15]))
        self.assertEqual(provenance["input_time"], "transition_block * 5")

    def test_real_cube_action_join_smoke(self) -> None:
        try:
            import h5py
        except Exception:
            self.skipTest("h5py unavailable")
        if not SOURCE_H5.exists():
            self.skipTest("verified Cube H5 unavailable")
        records_path = Path("runs/lewm_transfer/cube/relabel_motion/step_records_relabel_motion.csv")
        records = pd.read_csv(records_path).sort_values("row_id").head(4)
        with h5py.File(SOURCE_H5, "r") as h5:
            lengths = np.asarray(h5["ep_len"][:1], dtype=np.int64)
            offsets = np.asarray(h5["ep_offset"][:1], dtype=np.int64)
            episode_ids = np.asarray([int(h5["ep_idx"][int(offsets[0])])], dtype=np.int64)
            actions = np.asarray(h5["action"][: int(lengths[0])], dtype=np.float64)
        blocks = cube_action_blocks_from_arrays(records, actions=actions, offsets=offsets, lengths=lengths, episode_ids=episode_ids)
        self.assertEqual(blocks.shape[1], 25)
        for pos, row in enumerate(records.itertuples(index=False)):
            expected = actions[int(row.transition_block) * 5 : int(row.transition_block) * 5 + 5].reshape(-1)
            self.assertTrue(np.allclose(blocks[pos], expected))

    def test_shrinkage_whitening_math(self) -> None:
        values = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float64)
        transform = fit_whitening_transform(values, name="floor", family="shrinkage", floor_fraction=1e-3)
        expected_floor = float(transform.eigvals.max()) * 1e-3
        self.assertAlmostEqual(transform.floor_value, expected_floor)
        self.assertTrue(np.isfinite(transform.transform).all())
        delta = np.asarray([[0.0, 1.0]], dtype=np.float64)
        whitened = apply_delta_transform(delta, transform)
        self.assertAlmostEqual(float(np.sum(whitened**2)), 1.0 / expected_floor)

    def test_pca_truncation_math(self) -> None:
        values = np.asarray([[4.0, 1.0], [4.0, -1.0], [-4.0, 1.0], [-4.0, -1.0]], dtype=np.float64)
        transform90 = fit_whitening_transform(values, name="pca90", family="pca", pca_variance=0.90)
        transform99 = fit_whitening_transform(values, name="pca99", family="pca", pca_variance=0.99)
        self.assertEqual(transform90.retained_dim, 1)
        self.assertEqual(transform99.retained_dim, 2)

    def test_cross_episode_neighbors_exclude_anchor_episode(self) -> None:
        x = np.arange(12, dtype=np.float64).reshape(-1, 1)
        episodes = np.repeat(np.arange(4), 3)
        result = cross_episode_neighbors(x, episodes, k=3, k_min=2)
        self.assertTrue(result.valid.all())
        for anchor in range(len(x)):
            idx = result.indices[anchor]
            idx = idx[idx >= 0]
            self.assertTrue(np.all(episodes[idx] != episodes[anchor]))

    def test_local_expansion_contracting_and_branching(self) -> None:
        x = np.arange(20, dtype=np.float64).reshape(-1, 1)
        y = 0.5 * x
        frame = local_sensitivity_frame(
            x=x,
            y=y,
            residuals=np.zeros_like(x),
            episode_keys=np.arange(20),
            regimes=["interaction"] * 20,
            row_ids=np.arange(20),
            k=4,
            k_min=2,
        )
        self.assertLess(float(np.median(frame.loc[frame["valid_neighbors"], "local_expansion"])), 1.0)

        x_branch = np.zeros((12, 1), dtype=np.float64)
        y_branch = np.asarray([[0.0] if i % 2 else [10.0] for i in range(12)], dtype=np.float64)
        frame_branch = local_sensitivity_frame(
            x=x_branch,
            y=y_branch,
            residuals=np.zeros_like(x_branch),
            episode_keys=np.arange(12),
            regimes=["interaction"] * 12,
            row_ids=np.arange(12),
            k=6,
            k_min=3,
        )
        self.assertGreater(float(np.median(frame_branch.loc[frame_branch["valid_neighbors"], "local_expansion"])), 1.0)

    def test_probability_superiority_bounds_and_ties(self) -> None:
        value = probability_superiority([2.0, 3.0], [1.0, 3.0])
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(value, 0.625)

    def test_bootstrap_contrast_values_are_finite(self) -> None:
        row = contrast_values(np.asarray([2.0, 3.0, 4.0]), np.asarray([1.0, 1.0, 2.0]))
        self.assertTrue(np.isfinite(row["delta_trimmed_mean"]))
        self.assertGreater(row["p_superiority"], 0.0)

    def test_balanced_joint_features_shape(self) -> None:
        state = np.ones((3, 6), dtype=np.float64)
        action = np.ones((3, 2), dtype=np.float64)
        joint = balanced_joint_features(state, action)
        self.assertEqual(joint.shape, (3, 8))
        self.assertAlmostEqual(float(np.linalg.norm(joint[0, :6])), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(joint[0, 6:])), 1.0)

    def test_per_dim_std_floor(self) -> None:
        scale = per_dim_std(np.asarray([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]]))
        self.assertAlmostEqual(float(scale[1]), 1.0)

    def test_k_sweep_pass_rule(self) -> None:
        rows = pd.DataFrame(
            {
                "metric": ["local_expansion"] * 4,
                "delta_trimmed_mean": [0.1, 0.2, 0.3, 0.4],
                "delta_trimmed_mean_ci_low": [0.01, 0.02, 0.03, -0.01],
            }
        )
        self.assertTrue(k_sweep_pass(rows, metric="local_expansion"))


if __name__ == "__main__":
    unittest.main()
