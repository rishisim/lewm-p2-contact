"""Unit tests for the preregistered shared LeWM latent refiner."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from refiner import (  # noqa: E402
    EXITS,
    SharedResidualRefiner,
    build_causal_gate_features,
    gather_depth_outputs,
    validate_causal_feature_names,
)


class SharedResidualRefinerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(260713)
        self.batch = 4
        self.history_len = 3
        self.latent_dim = 5
        self.action_dim = 2
        self.history = torch.randn(self.batch, self.history_len, self.latent_dim)
        self.actions = torch.randn(self.batch, self.history_len, self.action_dim)
        self.base = torch.randn(self.batch, self.latent_dim)
        self.model = SharedResidualRefiner(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            history_len=self.history_len,
            hidden_dim=11,
            iteration_dim=3,
        )

    def _make_constant_update(self, amount: float = 1.0) -> None:
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.block[-1].bias.fill_(amount)

    def test_depth_zero_is_exact_identity_and_zero_compute(self) -> None:
        outputs, stats = self.model(
            self.history, self.actions, self.base, depths=(0,), return_stats=True
        )
        self.assertIs(outputs[0], self.base)
        self.assertTrue(torch.equal(outputs[0], self.base))
        self.assertEqual(stats.processed_rows, 0)
        self.assertEqual(stats.block_invocations, 0)

    def test_residual_safe_initialization_preserves_all_exits(self) -> None:
        outputs = self.model(self.history, self.actions, self.base, depths=EXITS)
        for depth in EXITS:
            self.assertTrue(torch.equal(outputs[depth], self.base))

    def test_one_shared_block_is_used_recurrently(self) -> None:
        self._make_constant_update(0.25)
        seen_batch_sizes: list[int] = []
        hook = self.model.block.register_forward_hook(
            lambda _module, inputs, _output: seen_batch_sizes.append(inputs[0].shape[0])
        )
        try:
            outputs, stats = self.model(
                self.history,
                self.actions,
                self.base,
                depths=EXITS,
                return_stats=True,
            )
        finally:
            hook.remove()

        self.assertIs(self.model.block, self.model.residual_block)
        self.assertEqual(sum(1 for _ in self.model.modules() if _ is self.model.block), 1)
        self.assertEqual(seen_batch_sizes, [self.batch] * 4)
        self.assertEqual(stats.block_invocations, 4)
        self.assertEqual(stats.processed_rows, self.batch * 4)
        for depth in EXITS:
            torch.testing.assert_close(
                outputs[depth], self.base + 0.25 * depth, rtol=0.0, atol=0.0
            )

    def test_gather_and_selected_execution_match_with_actual_call_counts(self) -> None:
        self._make_constant_update(0.5)
        dense = self.model(self.history, self.actions, self.base, depths=EXITS)
        selected_depths = torch.tensor([0, 1, 2, 4])
        gathered = gather_depth_outputs(dense, selected_depths)

        seen_batch_sizes: list[int] = []
        hook = self.model.block.register_forward_hook(
            lambda _module, inputs, _output: seen_batch_sizes.append(inputs[0].shape[0])
        )
        try:
            adaptive, stats = self.model.forward_selected(
                self.history,
                self.actions,
                self.base,
                selected_depths,
                return_stats=True,
            )
        finally:
            hook.remove()

        torch.testing.assert_close(adaptive, gathered, rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(adaptive[0], self.base[0]))
        self.assertEqual(seen_batch_sizes, [3, 2, 1, 1])
        self.assertEqual(stats.processed_rows, int(selected_depths.sum()))
        self.assertEqual(stats.block_calls, 7)
        self.assertEqual(stats.block_invocations, 4)

    def test_training_updates_reconstruct_depth_four_without_depth_three_exit(self) -> None:
        outputs, updates = self.model.forward_with_updates(
            self.history, self.actions, self.base
        )
        self.assertEqual(tuple(outputs), EXITS)
        self.assertNotIn(3, outputs)
        self.assertEqual(len(updates), 4)
        reconstructed = self.base
        for update in updates:
            self.assertEqual(update.shape, self.base.shape)
            reconstructed = reconstructed + update
        torch.testing.assert_close(reconstructed, outputs[4], rtol=0.0, atol=0.0)
        self.assertEqual(self.model.last_compute.processed_rows, self.batch * 4)
        self.assertEqual(self.model.last_compute.block_invocations, 4)

    def test_shapes_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "history must have shape"):
            self.model(self.history[:, :2], self.actions, self.base)
        with self.assertRaisesRegex(ValueError, "action_history must have shape"):
            self.model(self.history, self.actions[..., :1], self.base)
        with self.assertRaisesRegex(ValueError, "batch sizes differ"):
            self.model(self.history, self.actions, self.base[:2])
        with self.assertRaisesRegex(ValueError, "selected_depths must have shape"):
            self.model.forward_selected(
                self.history, self.actions, self.base, torch.tensor([[0, 1, 2, 4]])
            )
        with self.assertRaisesRegex(ValueError, "unsupported selected depths"):
            self.model.forward_selected(
                self.history, self.actions, self.base, torch.tensor([0, 1, 2, 3])
            )


class CausalGateFeatureTests(unittest.TestCase):
    def test_compact_features_have_expected_values_and_names(self) -> None:
        history = torch.tensor([[[3.0, 4.0], [0.0, 0.0], [0.0, 4.0]]])
        actions = torch.tensor([[[0.0], [2.0], [5.0]]])
        prediction = torch.tensor([[3.0, 0.0]])
        features, names = build_causal_gate_features(
            history, actions, prediction, return_names=True
        )

        expected = torch.tensor(
            [[5.0, 0.0, 4.0, 5.0, 4.0, 0.0, 2.0, 5.0, 2.0, 3.0, 3.0, 5.0]]
        )
        torch.testing.assert_close(features, expected)
        self.assertEqual(features.shape, (1, 12))
        self.assertEqual(len(names), features.shape[1])
        self.assertEqual(validate_causal_feature_names(names), names)

    def test_privileged_and_target_derived_feature_names_are_rejected(self) -> None:
        forbidden = (
            "target_norm",
            "futureStateVelocity",
            "contact_flag",
            "object_kinematics",
            "trajectory_phase",
            "motion_regime",
            "depth_4_gain",
            "oracle_choice",
        )
        for name in forbidden:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    validate_causal_feature_names(("history_token_0_norm", name))

    def test_gate_feature_shapes_are_validated(self) -> None:
        history = torch.zeros(2, 3, 5)
        actions = torch.zeros(2, 2, 4)
        prediction = torch.zeros(2, 5)
        with self.assertRaisesRegex(ValueError, "batch/history dimensions differ"):
            build_causal_gate_features(history, actions, prediction)
        with self.assertRaisesRegex(ValueError, "base_prediction must have shape"):
            build_causal_gate_features(history, torch.zeros(2, 3, 4), prediction[:, :4])


if __name__ == "__main__":
    unittest.main()
