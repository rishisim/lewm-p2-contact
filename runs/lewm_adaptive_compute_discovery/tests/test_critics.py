from pathlib import Path
import unittest
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import critics


class CriticTests(unittest.TestCase):
    def test_cross_fit_and_episode_permutation(self):
        rng = np.random.default_rng(4)
        episodes = np.repeat(np.arange(10), 4)
        x = rng.normal(size=(40, 5)).astype(np.float32)
        y = (0.1 * x[:, 0] + rng.normal(scale=.01, size=40)).astype(np.float32)
        models, oof = critics.fit_cross_fitted_critics(
            x, y, episodes, folds=2, seeds=[1,2], hidden_dims=[8], epochs=2,
            batch_size=16, lr=1e-3, weight_decay=0, device=torch.device("cpu"))
        self.assertEqual(len(models), 2)
        self.assertTrue(np.isfinite(oof).all())
        perm = critics.permute_gain_by_episode(y, episodes, 9)
        self.assertCountEqual(perm.tolist(), y.tolist())

    def test_torch_inputs_rejected_to_enforce_gradient_boundary(self):
        with self.assertRaises(TypeError):
            critics.fit_cross_fitted_critics(
                torch.zeros(4,2), np.zeros(4), np.arange(4), folds=2,
                seeds=[1], hidden_dims=[2], epochs=1, batch_size=2, lr=1e-3,
                weight_decay=0, device=torch.device("cpu"))


if __name__ == "__main__": unittest.main()
