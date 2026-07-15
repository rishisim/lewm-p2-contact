from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import (AdaptiveLeWM, LocalGainGate, SharedResidualRefiner,
                   build_causal_features, causal_feature_names)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.h = torch.randn(5, 3, 7)
        self.a = torch.randn(5, 3, 4)
        self.z = torch.randn(5, 7)
        self.r = SharedResidualRefiner(latent_dim=7, action_dim=4, history_len=3,
                                       hidden_dim=11, iteration_dim=3)

    def test_shared_refiner_exact_calls_and_selected_execution(self):
        exits, _, stats = self.r.forward_all(self.h, self.a, self.z)
        self.assertEqual(stats.processed_rows, 20)
        depths = torch.tensor([0, 1, 2, 3, 4])
        selected, selected_stats = self.r.forward_selected(self.h, self.a, self.z, depths)
        expected = torch.stack([exits[int(d)][i] for i, d in enumerate(depths)])
        torch.testing.assert_close(selected, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(selected_stats.processed_rows, 10)

    def test_causal_features_are_row_local(self):
        # Production dimensions are used because feature names deliberately
        # encode the frozen 192/25 contract.
        h = torch.randn(4, 3, 192)
        a = torch.randn(4, 3, 25)
        z = torch.randn(4, 192)
        delta = torch.randn(4, 192)
        first = build_causal_features(h, a, z, delta)
        h2, a2, z2, d2 = h.clone(), a.clone(), z.clone(), delta.clone()
        h2[1:] += 100; a2[1:] -= 50; z2[1:] *= 3; d2[1:] *= -2
        second = build_causal_features(h2, a2, z2, d2)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first.shape[1], len(causal_feature_names()))

    def test_target_mutation_cannot_enter_feature_api(self):
        import inspect
        parameters = set(inspect.signature(build_causal_features).parameters)
        forbidden = {"target", "future", "loss", "contact", "label", "regime", "episode"}
        self.assertTrue(parameters.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
