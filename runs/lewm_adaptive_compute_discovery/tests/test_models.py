from __future__ import annotations

import inspect
import sys
from pathlib import Path
import unittest

import torch
from torch import nn


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from models import (  # noqa: E402
    ContractiveSharedResidualCascade,
    FrozenV1Anchor,
    StagewiseResidualCascade,
    V3RepairControl,
    assert_critic_gradient_boundary,
    audit_epoch_zero_anchor,
    build_causal_features,
    trainable_parameter_names,
    validate_causal_feature_names,
)


class ToyV1Refiner(nn.Module):
    """Small exact implementation of the V1 public/model-state contract."""

    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = 5
        self.action_dim = 2
        self.history_len = 3
        self.hidden_dim = 11
        self.iteration_dim = 3
        self.max_depth = 4
        width = (
            self.history_len * self.latent_dim
            + self.history_len * self.action_dim
            + self.latent_dim
            + self.iteration_dim
        )
        self.iteration_embedding = nn.Embedding(self.max_depth, self.iteration_dim)
        self.block = nn.Sequential(
            nn.Linear(width, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

    def forward(self, history, actions, base, depths=(0, 1, 2, 4)):
        requested = tuple(sorted(set(depths)))
        outputs = {0: base} if 0 in requested else {}
        current = base
        for iteration in range(max(requested)):
            emb = self.iteration_embedding(
                torch.full(
                    (len(base),), iteration, dtype=torch.long, device=base.device
                )
            )
            values = torch.cat(
                (history.flatten(1), actions.flatten(1), current, emb), dim=1
            )
            current = current + self.block(values)
            if iteration + 1 in requested:
                outputs[iteration + 1] = current
        return outputs


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260713)
        self.history = torch.randn(6, 3, 5)
        self.actions = torch.randn(6, 3, 2)
        self.base = torch.randn(6, 5)

    def test_frozen_anchor_is_bitwise_v1_and_cannot_train(self) -> None:
        v1 = ToyV1Refiner().eval()
        expected = v1(self.history, self.actions, self.base, depths=(0, 1))
        anchor = FrozenV1Anchor(v1).train()
        actual, stats = anchor(
            self.history, self.actions, self.base, return_stats=True
        )

        self.assertIs(actual[0], self.base)
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertFalse(anchor.training)
        self.assertFalse(anchor.refiner.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in anchor.parameters()))
        self.assertEqual(stats.processed_rows, len(self.base))
        self.assertEqual(stats.block_invocations, 1)

    def test_stagewise_epoch_zero_and_only_one_stage_trainable(self) -> None:
        v1 = ToyV1Refiner()
        expected_z1 = v1(self.history, self.actions, self.base, depths=(1,))[1]
        cascade = StagewiseResidualCascade(v1, later_stages=3, hidden_dim=9)
        outputs, stats = cascade(
            self.history, self.actions, self.base, return_stats=True
        )
        audit = audit_epoch_zero_anchor(
            outputs, base_prediction=self.base, expected_v1_depth_one=expected_z1
        )

        self.assertTrue(audit.passed, audit.as_dict())
        self.assertEqual(stats.processed_rows, len(self.base) * 4)
        self.assertTrue(
            all(name.startswith("adapters.0.") for name in trainable_parameter_names(cascade))
        )
        cascade.set_trainable_stage(3)
        self.assertTrue(
            all(name.startswith("adapters.1.") for name in trainable_parameter_names(cascade))
        )

    def test_stagewise_detaches_preceding_exit_and_counts_calls(self) -> None:
        cascade = StagewiseResidualCascade(
            ToyV1Refiner(), later_stages=3, hidden_dim=9
        )
        with torch.no_grad():
            cascade.adapters[0].network[-1].bias.fill_(0.2)
        # Deliberately make adjacent stages trainable: the architecture, rather
        # than the normal optimizer scope, must block the earlier gradient.
        cascade.adapters[0].requires_grad_(True)
        cascade.adapters[1].requires_grad_(True)
        outputs = cascade(self.history, self.actions, self.base, max_depth=3)
        outputs[3].square().mean().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in cascade.adapters[0].parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in cascade.adapters[1].parameters())
        )

        selected = torch.tensor([0, 1, 2, 3, 4, 2])
        _, stats = cascade.forward_selected(
            self.history,
            self.actions,
            self.base,
            selected,
            return_stats=True,
        )
        self.assertEqual(stats.processed_rows, int(selected.sum()))
        self.assertEqual(stats.block_invocations, 4)

    def test_v3_repair_reproduces_v1_and_excludes_call_zero_params(self) -> None:
        v1 = ToyV1Refiner().eval()
        expected = v1(
            self.history, self.actions, self.base, depths=(0, 1, 2, 3, 4)
        )
        repair = V3RepairControl(v1)
        actual = repair(self.history, self.actions, self.base)

        for depth in range(5):
            self.assertTrue(torch.equal(actual[depth], expected[depth]), depth)
        names = trainable_parameter_names(repair)
        self.assertTrue(names)
        self.assertTrue(
            all(
                name.startswith(("later_block.", "later_iteration_embedding."))
                for name in names
            )
        )
        self.assertEqual(
            repair.later_iteration_embedding.num_embeddings, v1.max_depth - 1
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in repair.anchor.parameters())
        )
        selected = torch.tensor([0, 1, 2, 3, 4, 2])
        sparse, stats = repair.forward_selected(
            self.history, self.actions, self.base, selected, return_stats=True
        )
        for row, depth in enumerate(selected.tolist()):
            torch.testing.assert_close(sparse[row], actual[depth][row], rtol=2e-6, atol=2e-7)
        self.assertEqual(stats.processed_rows, int(selected.sum()))

    def test_contractive_shared_cascade_factor_and_epoch_zero(self) -> None:
        v1 = ToyV1Refiner()
        expected_z1 = v1(self.history, self.actions, self.base, depths=(1,))[1]
        cascade = ContractiveSharedResidualCascade(
            v1, max_depth=4, hidden_dim=9, relaxation=0.4
        )
        initial = cascade(self.history, self.actions, self.base)
        self.assertTrue(
            audit_epoch_zero_anchor(
                initial,
                base_prediction=self.base,
                expected_v1_depth_one=expected_z1,
            ).passed
        )
        self.assertAlmostEqual(cascade.contraction_factor, 0.6, places=6)

        with torch.no_grad():
            cascade.shared_adapter.network[-1].bias.copy_(
                torch.tensor([0.3, -0.2, 0.4, 0.1, -0.1])
            )
        outputs = cascade(self.history, self.actions, self.base)
        first = outputs[2] - outputs[1]
        second = outputs[3] - outputs[2]
        torch.testing.assert_close(
            second,
            cascade.contraction_factor * first,
            atol=2e-7,
            rtol=2e-6,
        )
        losses = cascade.stability_losses(outputs)
        self.assertLess(losses["shortcut_consistency"].item(), 1e-12)
        self.assertLess(losses["contraction_violation"].item(), 1e-6)
        selected = torch.tensor([0, 1, 2, 3, 4, 2])
        sparse, stats = cascade.forward_selected(
            self.history, self.actions, self.base, selected, return_stats=True
        )
        for row, depth in enumerate(selected.tolist()):
            torch.testing.assert_close(sparse[row], outputs[depth][row], rtol=2e-6, atol=2e-7)
        self.assertEqual(stats.processed_rows, int(selected.sum()))

    def test_causal_features_are_row_local_target_free_and_detached(self) -> None:
        base = self.base.requires_grad_()
        update = torch.randn_like(base, requires_grad=True)
        features, names = build_causal_features(
            self.history, self.actions, base, update, return_names=True
        )
        self.assertFalse(features.requires_grad)
        validate_causal_feature_names(names)

        changed = [
            value.clone()
            for value in (
                self.history,
                self.actions,
                base.detach(),
                update.detach(),
            )
        ]
        for value in changed:
            value[1:] += 100
        second = build_causal_features(*changed)
        self.assertTrue(torch.equal(features[0], second[0]))

        forbidden = {"target", "future", "contact", "label", "episode", "regime"}
        self.assertTrue(
            forbidden.isdisjoint(inspect.signature(build_causal_features).parameters)
        )

    def test_gradient_boundary_audit_proves_detached_critic_path(self) -> None:
        solver = StagewiseResidualCascade(
            ToyV1Refiner(), later_stages=1, hidden_dim=9
        )
        outputs, updates = solver(
            self.history, self.actions, self.base, return_updates=True
        )
        features = build_causal_features(
            self.history, self.actions, outputs[2], updates[2]
        )
        critic = nn.Sequential(
            nn.Linear(features.shape[1], 7), nn.GELU(), nn.Linear(7, 1)
        )
        scores = critic(features)
        audit = assert_critic_gradient_boundary(scores, solver=solver, critic=critic)

        self.assertTrue(audit.passed)
        self.assertGreater(audit.solver_trainable_parameters, 0)
        self.assertEqual(audit.solver_connected_tensors, 0)
        self.assertGreater(audit.critic_nonzero_tensors, 0)

    def test_causal_name_validator_rejects_privileged_columns(self) -> None:
        for name in (
            "target_mse",
            "future_latent",
            "contact_flag",
            "episode_global",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    validate_causal_feature_names(("current_0", name))


if __name__ == "__main__":
    unittest.main()
