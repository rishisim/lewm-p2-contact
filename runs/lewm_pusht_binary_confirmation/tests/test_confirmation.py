from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT.parent / "lewm_pusht_replication_pilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PILOT_ROOT))

from pusht_core import (  # noqa: E402
    ACTION_DIM, HISTORY, LATENT_DIM, build_causal_features, causal_feature_names,
    gate_operation_ledger, refiner_flops,
)
from run_confirmation import (  # noqa: E402
    BOOTSTRAP_REPLICATES, THRESHOLD, analytic_mixture, binary_calls,
    bootstrap_summaries, episode_means,
)


def test_binary_policy_uses_strict_single_threshold() -> None:
    scores = np.asarray([THRESHOLD - 1e-12, THRESHOLD, THRESHOLD + 1e-12])
    np.testing.assert_array_equal(binary_calls(scores), [1, 1, 2])


def test_causal_boundary_and_frozen_operation_prices() -> None:
    assert list(inspect.signature(build_causal_features).parameters) == [
        "history", "actions", "current", "update"
    ]
    forbidden = ("target", "future", "contact", "reward", "success", "simulator", "geometry", "state")
    assert not any(term in name.lower() for name in causal_feature_names() for term in forbidden)
    feature = build_causal_features(
        torch.randn(3, HISTORY, LATENT_DIM), torch.randn(3, HISTORY, ACTION_DIM),
        torch.randn(3, LATENT_DIM), torch.randn(3, LATENT_DIM),
    )
    assert feature.shape == (3, 1001)
    assert gate_operation_ledger()["total_flops_per_reached_evaluation"] == 7715
    assert refiner_flops()["total_flops_per_call"] == 650432


def test_pairwise_comparator_search_is_episode_averaged_and_exact_depth() -> None:
    losses = np.asarray(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 1.2, 3.0, 4.0],
            [2.0, 1.5, 1.4, 1.6],
            [5.0, 3.0, 2.5, 2.0],
        ]
    )
    episode_id = np.asarray([0, 0, 1, 1])
    result = analytic_mixture(losses, 1.25, episode_id, 2)
    reconstructed = (
        (1 - result["weight_upper"]) * result["depth_lower"]
        + result["weight_upper"] * result["depth_upper"]
    )
    assert np.isclose(reconstructed, 1.25)
    assert result["feasible_pair_count"] == 3


def test_episode_cluster_bootstrap_is_fixed_and_reproducible() -> None:
    values = {"x": np.linspace(-0.2, 0.5, 240)}
    first = bootstrap_summaries(values, seed=19)
    second = bootstrap_summaries(values, seed=19)
    assert first == second
    assert first["x"]["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    ids = np.repeat(np.arange(240), 2)
    rows = np.arange(480, dtype=float)
    means = episode_means(rows, ids)
    np.testing.assert_allclose(means, rows.reshape(240, 2).mean(1))
