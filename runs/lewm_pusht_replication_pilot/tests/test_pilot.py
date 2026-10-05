from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pusht_core import (  # noqa: E402
    ACTION_DIM, HISTORY, LATENT_DIM, StagewiseRefiner, bootstrap_interval,
    build_causal_features, causal_feature_names, episode_means, gate_operation_ledger,
    raw_white_losses, refiner_flops, sequential_calls, strongest_analytic_mixture,
)


def test_stagewise_refiner_has_one_shared_block_and_depths_one_to_four() -> None:
    torch.manual_seed(3)
    model = StagewiseRefiner(hidden=16, iteration_dim=4)
    history = torch.randn(7, HISTORY, LATENT_DIM)
    actions = torch.randn(7, HISTORY, ACTION_DIM)
    base = torch.randn(7, LATENT_DIM)
    exits, updates = model.dense(history, actions, base)
    assert exits.shape == (7, 4, LATENT_DIM)
    assert updates.shape == exits.shape
    assert sum(1 for _ in model.modules() if isinstance(_, torch.nn.Sequential)) == 1
    assert torch.equal(exits[:, 0], base)  # zero-initialized final layer


def test_causal_feature_boundary_and_operation_ledger() -> None:
    signature = list(inspect.signature(build_causal_features).parameters)
    assert signature == ["history", "actions", "current", "update"]
    forbidden = ("target", "future", "contact", "reward", "success", "simulator", "geometry", "state")
    names = causal_feature_names()
    assert len(names) == 1001
    assert not any(term in name.lower() for name in names for term in forbidden)
    features = build_causal_features(
        torch.randn(5, HISTORY, LATENT_DIM), torch.randn(5, HISTORY, ACTION_DIM),
        torch.randn(5, LATENT_DIM), torch.randn(5, LATENT_DIM),
    )
    assert features.shape == (5, 1001)
    ledger = gate_operation_ledger()
    assert ledger["feature_flops_per_reached_evaluation"] == 3711
    assert ledger["dual_affine_score_flops_per_reached_evaluation"] == 4004
    assert ledger["total_flops_per_reached_evaluation"] == 7715
    assert refiner_flops()["total_flops_per_call"] > ledger["total_flops_per_reached_evaluation"]


def test_sequential_calls_and_exact_analytic_mixture() -> None:
    scores = np.asarray([[2, 2, 2], [2, -1, 9], [-1, 9, 9], [2, 2, -1]], dtype=float)
    calls, reached = sequential_calls(scores, np.zeros(3))
    np.testing.assert_array_equal(calls, [4, 2, 1, 3])
    np.testing.assert_array_equal([mask.sum() for mask in reached], [4, 3, 2])
    losses = np.asarray(
        [[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 2.5, 2.7]],
        dtype=float,
    )
    episode_id = np.arange(3)
    mixture = strongest_analytic_mixture(losses, 2.25, episode_id, 3)
    assert mixture["depth_lower"] <= 2.25 <= mixture["depth_upper"]
    assert np.isclose(
        (1 - mixture["weight_upper"]) * mixture["depth_lower"]
        + mixture["weight_upper"] * mixture["depth_upper"],
        2.25,
    )


def test_loss_and_episode_cluster_bootstrap_are_reproducible() -> None:
    rng = np.random.default_rng(44)
    exits = rng.normal(size=(8, 4, LATENT_DIM)).astype(np.float32)
    target = rng.normal(size=(8, LATENT_DIM)).astype(np.float32)
    raw, white = raw_white_losses(exits, target, np.eye(LATENT_DIM))
    np.testing.assert_allclose(raw, white, rtol=0, atol=1e-12)
    ids = np.repeat(np.arange(4), 2)
    values = episode_means(raw[:, 0] - raw[:, 1], ids, 4)
    first = bootstrap_interval(values, seed=17, replicates=300)
    second = bootstrap_interval(values, seed=17, replicates=300)
    assert first == second
