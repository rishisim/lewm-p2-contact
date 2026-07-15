from __future__ import annotations

import numpy as np
import torch

import common
import runtime


def test_all_frozen_byte_hashes_and_tournament_contract() -> None:
    result = common.verify_frozen_objects()
    assert all(result["contract"].values())
    assert result["solver_checkpoint"]["sha256"] == common.EXPECTED["solver_checkpoint_sha256"]
    assert result["student_checkpoint"]["sha256"] == common.EXPECTED["student_checkpoint_sha256"]


def test_linear_student_normalizer_feature_set_and_price_are_frozen() -> None:
    gate = runtime.load_gate(torch.device("cpu"))
    assert gate.compute_price == common.COMPUTE_PRICE
    assert gate.metadata["parameters"] == common.GATE_PARAMETERS
    assert gate.feature_mean.shape == (1049,)
    assert gate.feature_std.shape == (1049,)
    assert torch.equal(gate.feature_indices.cpu(), torch.arange(1046))
    assert all(parameter.grad is None for parameter in gate.model.parameters())


def test_discovery_whitening_is_loaded_not_refit() -> None:
    result = runtime.load_whitening()
    assert result["matrix"].shape == (192, 192)
    assert np.isfinite(result["matrix"]).all()
    assert common.sha256_file(common.WHITENING) == common.EXPECTED["whitening_sha256"]
