from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch


RUN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RUN_DIR.parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from refiner import (  # noqa: E402
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_V1_REFINER_SHA256,
    SharedResidualRefiner,
    build_post_call_features,
    hash_state_tensors,
    load_frozen_v1_refiner,
    post_call_feature_names,
    validate_causal_feature_names,
    validate_frozen_refiner,
)


V1_CHECKPOINT = REPO_ROOT / "runs" / "lewm_adaptive_compute_v1" / "checkpoints" / "refiner_seed_260713.pt"


class SequentialRefinerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(260813)
        self.batch = 5
        self.history = torch.randn(self.batch, 3, 7)
        self.action = torch.randn(self.batch, 3, 4)
        self.z0 = torch.randn(self.batch, 7)
        self.model = SharedResidualRefiner(
            latent_dim=7, action_dim=4, history_len=3, hidden_dim=13, iteration_dim=5
        ).eval()

    def _constant_updates(self, amount: float) -> None:
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
            self.model.block[-1].bias.fill_(amount)

    def test_first_call_processes_every_row_exactly_once(self) -> None:
        self._constant_updates(0.25)
        z1, update, stats = self.model.forward_first(
            self.history, self.action, self.z0, return_update=True, return_stats=True
        )
        torch.testing.assert_close(update, torch.full_like(update, 0.25), rtol=0, atol=0)
        torch.testing.assert_close(z1, self.z0 + 0.25, rtol=0, atol=0)
        self.assertEqual(stats.processed_rows, self.batch)
        self.assertEqual(stats.block_invocations, 1)

    def test_staged_dense_outputs_are_bitwise_identical_on_identical_rows(self) -> None:
        dense = self.model(self.history, self.action, self.z0, depths=(0, 1, 2, 4))
        z1 = self.model.forward_first(self.history, self.action, self.z0)
        self.assertTrue(torch.equal(z1, dense[1]))
        for depth in (1, 2, 4):
            staged, stats = self.model.continue_selected(
                self.history,
                self.action,
                z1,
                torch.full((self.batch,), depth, dtype=torch.long),
                return_stats=True,
            )
            self.assertTrue(torch.equal(staged, dense[depth]))
            self.assertEqual(stats.processed_rows, self.batch * (depth - 1))

    def test_mixed_continuation_prunes_rows_and_accounts_exact_calls(self) -> None:
        self._constant_updates(0.5)
        z1 = self.model.forward_first(self.history, self.action, self.z0)
        allocation = torch.tensor([1, 2, 4, 4, 1])
        output, stats = self.model.continue_selected(
            self.history, self.action, z1, allocation, return_stats=True
        )
        expected = self.z0 + 0.5 * allocation[:, None]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(stats.processed_rows, int((allocation - 1).sum()))
        self.assertEqual(stats.block_invocations, 3)
        self.assertEqual(self.batch + stats.processed_rows, int(allocation.sum()))

    def test_invalid_final_depths_are_rejected(self) -> None:
        z1 = self.model.forward_first(self.history, self.action, self.z0)
        with self.assertRaisesRegex(ValueError, "unsupported final depths"):
            self.model.continue_selected(
                self.history, self.action, z1, torch.tensor([1, 2, 3, 4, 1])
            )

    def test_repeated_fixed_checkpoint_outputs_are_bitwise_identical(self) -> None:
        model, _ = load_frozen_v1_refiner(V1_CHECKPOINT)
        torch.manual_seed(99)
        history = torch.randn(3, 3, 192)
        action = torch.randn(3, 3, 25)
        z0 = torch.randn(3, 192)
        with torch.inference_mode():
            first = model(history, action, z0, depths=(1, 2, 4))
            second = model(history, action, z0, depths=(1, 2, 4))
        for depth in (1, 2, 4):
            self.assertTrue(torch.equal(first[depth], second[depth]))


class FrozenCheckpointTests(unittest.TestCase):
    def test_strict_checkpoint_load_hash_keys_freeze_and_state_hash(self) -> None:
        model, provenance = load_frozen_v1_refiner(V1_CHECKPOINT)
        self.assertEqual(provenance["checkpoint_sha256"], EXPECTED_V1_REFINER_SHA256)
        self.assertEqual(provenance["parameter_count"], EXPECTED_PARAMETER_COUNT)
        self.assertEqual(provenance["strict_key_count"], 7)
        self.assertEqual(provenance["state_tensor_sha256"], hash_state_tensors(model.state_dict()))
        validate_frozen_refiner(model)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_wrong_checkpoint_hash_is_rejected_before_load(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            load_frozen_v1_refiner(V1_CHECKPOINT, expected_sha256="0" * 64)


class PostCallFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(260813)
        self.history = torch.randn(2, 3, 192)
        self.action = torch.randn(2, 3, 25)
        self.z0 = torch.randn(2, 192)
        self.z1 = self.z0 + 0.1 * torch.randn(2, 192)

    def test_feature_shape_order_values_and_finiteness(self) -> None:
        features, names = build_post_call_features(
            self.history, self.action, self.z0, self.z1, return_names=True
        )
        self.assertEqual(features.shape, (2, 1248))
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(names, post_call_feature_names())
        self.assertEqual(names[0], "history_token_0_component_0")
        self.assertEqual(names[575], "history_token_2_component_191")
        self.assertEqual(names[576], "action_block_0_component_0")
        self.assertEqual(names[650], "action_block_2_component_24")
        self.assertEqual(names[651], "z0_component_0")
        self.assertEqual(names[843], "z1_component_0")
        self.assertEqual(names[1035], "first_update_component_0")
        torch.testing.assert_close(features[:, :576], self.history.flatten(1), rtol=0, atol=0)
        torch.testing.assert_close(features[:, 576:651], self.action.flatten(1), rtol=0, atol=0)
        torch.testing.assert_close(features[:, 651:843], self.z0, rtol=0, atol=0)
        torch.testing.assert_close(features[:, 843:1035], self.z1, rtol=0, atol=0)
        torch.testing.assert_close(features[:, 1035:1227], self.z1 - self.z0, rtol=0, atol=0)

    def test_zero_vectors_remain_finite_with_floored_denominators(self) -> None:
        zero_history = torch.zeros(1, 3, 192)
        features = build_post_call_features(
            zero_history, torch.zeros(1, 3, 25), torch.zeros(1, 192), torch.zeros(1, 192)
        )
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.equal(features, torch.zeros_like(features)))

    def test_whitelist_rejects_missing_reordered_or_privileged_names(self) -> None:
        names = post_call_feature_names()
        self.assertEqual(validate_causal_feature_names(names), names)
        with self.assertRaisesRegex(ValueError, "whitelist"):
            validate_causal_feature_names(names[:-1])
        with self.assertRaisesRegex(ValueError, "whitelist"):
            validate_causal_feature_names((names[1], names[0], *names[2:]))
        with self.assertRaisesRegex(ValueError, "whitelist"):
            validate_causal_feature_names((*names[:-1], "target_loss"))

    def test_builder_has_no_target_or_label_input_and_external_mutation_is_irrelevant(self) -> None:
        parameters = set(inspect.signature(build_post_call_features).parameters)
        forbidden = {"target", "future", "loss", "contact", "regime", "phase", "label", "split"}
        self.assertFalse(parameters & forbidden)
        first = build_post_call_features(self.history, self.action, self.z0, self.z1)
        target = torch.randn(2, 192)
        labels = torch.tensor([0, 1])
        target.mul_(1e6)
        labels.logical_not_()
        second = build_post_call_features(self.history, self.action, self.z0, self.z1)
        self.assertTrue(torch.equal(first, second))
        with self.assertRaises(TypeError):
            build_post_call_features(
                self.history, self.action, self.z0, self.z1, target=target  # type: ignore[call-arg]
            )


if __name__ == "__main__":
    unittest.main()
