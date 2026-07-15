from __future__ import annotations

import numpy as np
import torch

import common
import runtime


@torch.inference_mode()
def test_dense_reference_and_compact_sparse_paths_on_old_discovery_data() -> None:
    with np.load(common.COMPRESSION / "cache/stagewise_train_outputs.npz", allow_pickle=False) as stored:
        # This cache lacks input tensors, so use the immutable discovery input cache.
        assert stored["features"].shape[2] == 1046
    with np.load(common.DISCOVERY / "cache/v3_train_only.npz", allow_pickle=False) as stored:
        history = torch.from_numpy(np.ascontiguousarray(stored["history"][:8]))
        action = torch.from_numpy(np.ascontiguousarray(stored["action"][:8]))
        base = torch.from_numpy(np.ascontiguousarray(stored["base_pred"][:8]))
    solver, _, models = runtime.load_solver(torch.device("cpu"))
    gate = runtime.load_gate(torch.device("cpu"))
    dense, dense_calls, scores, features = runtime.dense_adaptive(
        solver, gate, models, history, action, base
    )
    reference, reference_calls = runtime.sparse_adaptive_reference(
        solver, gate, models, history, action, base
    )
    optimized, optimized_calls = runtime.sparse_adaptive_optimized(
        solver, gate, models, history, action, base
    )
    assert torch.equal(dense_calls, reference_calls)
    assert torch.equal(dense_calls, optimized_calls)
    assert np.array_equal(
        dense_calls.numpy(), runtime.sequential_calls_numpy(scores.numpy(), common.COMPUTE_PRICE)
    )
    assert torch.allclose(dense, reference, rtol=2e-5, atol=2e-6)
    assert torch.allclose(dense, optimized, rtol=2e-5, atol=2e-6)
    assert features.shape == (8, 3, 1046)
    assert not features.requires_grad
