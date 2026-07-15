from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common
import protocol
import students


class GroupIsolationTests(unittest.TestCase):
    def test_nested_episode_assignment_never_splits_episode(self) -> None:
        episodes = np.repeat(np.arange(30), 4)
        outer = common.episode_assignment(episodes, 3, 11)
        for episode in np.unique(episodes):
            self.assertEqual(len(np.unique(outer[episodes == episode])), 1)
        train = outer != 0
        inner = common.episode_assignment(episodes[train], 3, 17)
        self.assertFalse(set(episodes[train][inner == 0]) & set(episodes[train][inner == 1]))

    def test_fold_local_sparse_selection_ignores_held_rows(self) -> None:
        rng = np.random.default_rng(4)
        episodes = np.repeat(np.arange(12), 2)
        features = rng.normal(size=(len(episodes), 3, 1046)).astype(np.float32)
        target = rng.normal(size=(len(episodes), 3))
        assignment = common.episode_assignment(episodes, 3, 9)
        train = assignment != 0
        first, _ = students.stable_feature_selection(features[train], target[train], episodes[train], k=8, folds=2, seed=3)
        changed = features.copy()
        changed[~train] *= 1e6
        second, _ = students.stable_feature_selection(changed[train], target[train], episodes[train], k=8, folds=2, seed=3)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first < 1035))

    def test_teacher_scores_are_strictly_cross_fitted_by_episode(self) -> None:
        rng = np.random.default_rng(14)
        episodes = np.repeat(np.arange(9), 2)
        features = rng.normal(size=(len(episodes), 3, 1046)).astype(np.float32)
        gains = rng.normal(scale=1e-3, size=(len(episodes), 3)).astype(np.float32)
        scores, provenance = students.crossfit_teacher(
            features, gains, episodes,
            folds=3, seed=19, teacher_seeds=(21, 22, 23),
            device=torch.device("cpu"), hidden_dims=(2,), epochs=1,
            batch_size=32, learning_rate=1e-3,
        )
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores.shape, gains.shape)
        for fold in provenance:
            self.assertFalse(set(fold["train_episodes"]) & set(fold["held_episodes"]))


class CausalityAndBoundaryTests(unittest.TestCase):
    def test_feature_allowlist_and_detachment(self) -> None:
        audit = common.causal_feature_audit()
        self.assertTrue(audit["passed"])
        _, models, _, _ = common.load_prior_modules()
        parameter = torch.nn.Parameter(torch.randn(5, 192))
        history = torch.randn(5, 3, 192, requires_grad=True)
        action = torch.randn(5, 3, 25, requires_grad=True)
        update = torch.randn(5, 192, requires_grad=True)
        features = models.build_causal_features(history, action, parameter, update)
        self.assertFalse(features.requires_grad)
        self.assertEqual(features.shape, (5, 1046))
        self.assertIsNone(parameter.grad)

    def test_sequential_stopping_does_not_consult_unreached_future_score(self) -> None:
        scores = np.asarray([[-2.0, 99.0, 99.0], [2.0, -2.0, 99.0]])
        original = protocol.sequential_calls(scores, 0.0)
        changed = scores.copy()
        changed[0, 1:] = -9999.0
        changed[1, 2] = -9999.0
        np.testing.assert_array_equal(original, protocol.sequential_calls(changed, 0.0))
        np.testing.assert_array_equal(original, np.asarray([1, 2]))

    def test_threshold_is_fit_without_evaluation_scores(self) -> None:
        train = np.linspace(-1, 1, 90).reshape(30, 3)
        before = protocol.calibrate_price(train, 1.5)["compute_price"]
        evaluation = np.full((12, 3), 1e9)
        evaluation[:] = -1e9
        after = protocol.calibrate_price(train, 1.5)["compute_price"]
        self.assertEqual(before, after)


class AccountingTests(unittest.TestCase):
    def test_all_declared_students_fit_hard_flop_budget(self) -> None:
        audit = students.audit_candidate_costs(common.load_config())
        for family in audit.values():
            for candidate in family:
                self.assertTrue(candidate["cost"]["within_budget"])
                self.assertLessEqual(candidate["cost"]["total_incremental_gate_flops_per_evaluated_decision"], 66240)
        low_rank_16 = students.gate_cost("low_rank", 1046, (16,))
        self.assertEqual(low_rank_16["total_incremental_gate_flops_per_evaluated_decision"], 39874)

    def test_exact_call_baseline_and_histogram(self) -> None:
        rng = np.random.default_rng(2)
        losses = rng.uniform(size=(40, 4))
        calls = np.asarray([1] * 10 + [2] * 12 + [3] * 8 + [4] * 10)
        policy = protocol.prior_policy()
        baseline = policy.strongest_transition_independent_baseline(
            losses, protocol.SUPPORTED_CALLS, n=len(calls),
            target_total_calls=int(calls.sum()), seed=8,
        )
        self.assertTrue(baseline["audit"]["exact_total_match"])
        self.assertEqual(int(baseline["selected_calls"].sum()), int(calls.sum()))
        shuffled = policy.randomized_histogram_control(calls, 9)
        np.testing.assert_array_equal(np.sort(shuffled), np.sort(calls))

    def test_sparse_student_prediction_matches_manual_selected_path(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(10, 3, 1046)).astype(np.float32)
        gains = rng.normal(scale=1e-3, size=(10, 3)).astype(np.float32)
        selected = np.asarray([1, 3, 651, 843, 1000], dtype=np.int64)
        model = students.fit_student(
            features, gains, gains,
            family="stable_sparse",
            config={"hidden_dims": [], "weight_decay": 0.0, "objective": "realized", "sparse_k": 5},
            seed=12, device=torch.device("cpu"), feature_selection=selected,
            epochs=1, batch_size=16, learning_rate=1e-3,
        )
        predicted = students.predict_student(model, features, torch.device("cpu"))
        x = students._depth_encoded(features, selected)
        built = model.build(torch.device("cpu"))
        with torch.inference_mode():
            manual = built(torch.from_numpy((x - model.feature_mean) / model.feature_std)).numpy()
        manual = manual.reshape(10, 3) * model.target_scale + model.target_center
        np.testing.assert_allclose(predicted, manual, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
