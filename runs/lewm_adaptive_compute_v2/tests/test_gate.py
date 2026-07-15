from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from gate import (  # noqa: E402
    GateMLP,
    GateTrainingConfig,
    adjacent_pair_ranking_loss,
    benefit_quantile_rows,
    calibration_diagnostics,
    fit_input_standardizer,
    fit_robust_target_transform,
    gate_dense_linear_flops,
    gate_objective,
    gate_parameter_count,
    load_gate_checkpoint,
    predict_gate,
    save_gate_checkpoint,
    select_seed_from_diagnostics,
    sha256_file,
    shuffled_pair_orders,
    state_tensor_hash,
    train_gate_seed,
)


class TransformAndArchitectureTests(unittest.TestCase):
    def test_frozen_architecture_and_compute_formula(self) -> None:
        model = GateMLP(7)
        linears = [module for module in model.network if isinstance(module, torch.nn.Linear)]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linears],
            [(7, 128), (128, 64), (64, 2)],
        )
        self.assertEqual(gate_parameter_count(7), sum(p.numel() for p in model.parameters()))
        self.assertEqual(gate_dense_linear_flops(7), 2 * (7 * 128 + 128 * 64 + 64 * 2))
        self.assertEqual(tuple(model(torch.zeros(3, 7)).shape), (3, 2))

    def test_train_only_input_and_robust_target_transforms(self) -> None:
        train_x = np.asarray([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
        transform = fit_input_standardizer(train_x)
        np.testing.assert_allclose(transform.mean, [3.0, 5.0])
        self.assertAlmostEqual(transform.scale[0], np.std([1.0, 3.0, 5.0]))
        self.assertEqual(transform.scale[1], 1e-6)
        # A wildly shifted calibration matrix cannot alter a fitted transform.
        frozen_mean = transform.mean.copy()
        transformed = transform.transform(np.asarray([[1000.0, -1000.0]]))
        np.testing.assert_array_equal(transform.mean, frozen_mean)
        self.assertTrue(np.isfinite(transformed).all())

        targets = np.asarray([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [100.0, 2.0]])
        robust = fit_robust_target_transform(targets)
        np.testing.assert_allclose(robust.center, [1.5, 2.0])
        self.assertEqual(robust.scale[1], 1e-6)
        standardized = robust.transform(targets)
        np.testing.assert_allclose(robust.inverse(standardized), targets)


class ObjectiveTests(unittest.TestCase):
    def test_ranking_loss_rewards_correct_order_and_ignores_ties(self) -> None:
        target = torch.tensor([[2.0, 0.0], [0.0, 0.0], [3.0, 1.0], [1.0, 1.0]])
        orders = (np.asarray([0, 1, 2, 3]), np.asarray([0, 1, 2, 3]))
        correct = target * 4.0
        reversed_prediction = -correct
        self.assertLess(
            adjacent_pair_ranking_loss(correct, target, orders).item(),
            adjacent_pair_ranking_loss(reversed_prediction, target, orders).item(),
        )
        # Output 1 contains only equal-target adjacent pairs and contributes a
        # differentiable zero, while output 0 remains finite.
        self.assertTrue(torch.isfinite(adjacent_pair_ranking_loss(correct, target, orders)))

    def test_full_objective_matches_frozen_weights(self) -> None:
        prediction = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        orders = (np.asarray([0, 1]), np.asarray([0, 1]))
        total, smooth, ranking = gate_objective(prediction, target, orders)
        torch.testing.assert_close(total, smooth + 0.25 * ranking)
        expected_smooth = torch.nn.functional.smooth_l1_loss(
            prediction, target, beta=0.5
        )
        torch.testing.assert_close(smooth, expected_smooth)
        total.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_pair_orders_are_seeded_and_validated(self) -> None:
        first = shuffled_pair_orders(9, rng=260813)
        second = shuffled_pair_orders(9, rng=260813)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        with self.assertRaisesRegex(ValueError, "permutation"):
            adjacent_pair_ranking_loss(
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                (np.asarray([0, 0, 1]), np.asarray([0, 1, 2])),
            )


class TrainingSelectionAndDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(12)
        x = rng.normal(size=(96, 6))
        weights = np.asarray(
            [
                [0.8, -0.4],
                [-0.2, 0.6],
                [0.5, 0.3],
                [0.1, -0.7],
                [0.0, 0.2],
                [-0.3, 0.1],
            ]
        )
        y = x @ weights + 0.03 * rng.normal(size=(96, 2))
        return x[:72], y[:72], x[72:], y[72:]

    def test_seeded_training_is_reproducible_and_predicts_raw_units(self) -> None:
        train_x, train_y, calibration_x, calibration_y = self.synthetic_data()
        cfg = GateTrainingConfig(
            batch_size=24,
            max_epochs=5,
            early_stopping_patience=3,
            early_stopping_min_delta=0.0,
        )
        first = train_gate_seed(
            train_x,
            train_y,
            calibration_x,
            calibration_y,
            seed=260813,
            config=cfg,
        )
        second = train_gate_seed(
            train_x,
            train_y,
            calibration_x,
            calibration_y,
            seed=260813,
            config=cfg,
        )
        self.assertEqual(first.best_epoch, second.best_epoch)
        self.assertEqual(first.best_calibration_objective, second.best_calibration_objective)
        self.assertEqual(state_tensor_hash(first.model.state_dict()), state_tensor_hash(second.model.state_dict()))
        predicted = predict_gate(first, calibration_x, batch_size=7)
        self.assertEqual(predicted.shape, calibration_y.shape)
        self.assertTrue(np.isfinite(predicted).all())
        diagnostic = calibration_diagnostics(first, calibration_x, calibration_y)
        self.assertEqual(diagnostic["seed"], 260813)
        self.assertEqual({row["benefit"] for row in diagnostic["outputs"]}, {"b12", "b14"})
        self.assertTrue(np.isfinite(diagnostic["mean_spearman"]))

    def test_seed_selection_uses_spearman_then_objective_then_order(self) -> None:
        order = (260813, 260814, 260815)
        diagnostics = [
            {"seed": 260813, "mean_spearman": 0.4, "calibration_objective": 1.0},
            # Within 1e-12, so the smaller objective wins.
            {"seed": 260814, "mean_spearman": 0.4 + 5e-13, "calibration_objective": 0.9},
            {"seed": 260815, "mean_spearman": 0.3, "calibration_objective": 0.1},
        ]
        self.assertEqual(select_seed_from_diagnostics(diagnostics, order), 260814)
        diagnostics[0]["calibration_objective"] = 0.9
        diagnostics[1]["calibration_objective"] = 0.9
        self.assertEqual(select_seed_from_diagnostics(diagnostics, order), 260813)

    def test_equal_count_quantiles_are_separate_and_complete(self) -> None:
        predicted = np.column_stack([np.arange(23), -np.arange(23)]).astype(float)
        actual = predicted * 0.5
        rows = benefit_quantile_rows(predicted, actual)
        self.assertEqual(len(rows), 10)
        for benefit in ("b12", "b14"):
            subset = [row for row in rows if row["benefit"] == benefit]
            self.assertEqual(sum(int(row["n"]) for row in subset), 23)
            self.assertLessEqual(max(int(row["n"]) for row in subset) - min(int(row["n"]) for row in subset), 1)
            means = [float(row["mean_predicted_benefit"]) for row in subset]
            self.assertEqual(means, sorted(means))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_hash_and_strict_contract(self) -> None:
        train_x, train_y, calibration_x, calibration_y = (
            TrainingSelectionAndDiagnosticsTests.synthetic_data()
        )
        result = train_gate_seed(
            train_x,
            train_y,
            calibration_x,
            calibration_y,
            seed=260813,
            config=GateTrainingConfig(
                batch_size=36,
                max_epochs=2,
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
            ),
        )
        names = tuple(f"feature_{index}" for index in range(train_x.shape[1]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.pt"
            hashes = save_gate_checkpoint(path, result, feature_names=names)
            self.assertEqual(hashes["file_sha256"], sha256_file(path))
            loaded = load_gate_checkpoint(
                path,
                expected_input_dim=train_x.shape[1],
                expected_feature_names=names,
                expected_seed=260813,
                expected_sha256=hashes["file_sha256"],
            )
            self.assertEqual(loaded.state_tensor_hash, hashes["state_tensor_hash"])
            expected = predict_gate(result, calibration_x)
            observed = predict_gate(
                loaded.model,
                calibration_x,
                loaded.input_transform,
                loaded.target_transform,
            )
            np.testing.assert_array_equal(expected, observed)
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                load_gate_checkpoint(
                    path,
                    expected_input_dim=train_x.shape[1],
                    expected_sha256="0" * 64,
                )

    def test_checkpoint_rejects_missing_state_key(self) -> None:
        train_x, train_y, calibration_x, calibration_y = (
            TrainingSelectionAndDiagnosticsTests.synthetic_data()
        )
        result = train_gate_seed(
            train_x,
            train_y,
            calibration_x,
            calibration_y,
            seed=260813,
            config=GateTrainingConfig(max_epochs=1, early_stopping_patience=1),
        )
        with tempfile.TemporaryDirectory() as temporary:
            valid = Path(temporary) / "valid.pt"
            malformed = Path(temporary) / "malformed.pt"
            save_gate_checkpoint(
                valid,
                result,
                feature_names=tuple(f"f{i}" for i in range(train_x.shape[1])),
            )
            payload = torch.load(valid, map_location="cpu", weights_only=True)
            bad = copy.deepcopy(payload)
            bad["state_dict"].pop(next(iter(bad["state_dict"])))
            torch.save(bad, malformed)
            with self.assertRaisesRegex(RuntimeError, "strict gate state keys mismatch"):
                load_gate_checkpoint(malformed, expected_input_dim=train_x.shape[1])


if __name__ == "__main__":
    unittest.main()
