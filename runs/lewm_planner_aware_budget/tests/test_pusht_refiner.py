import pytest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pusht_refiner import (
    MaskedStagewiseRefiner,
    OperationLedger,
    canonicalize_action,
    physical_action,
    prefix_mask,
)


def inputs(length: int = 1):
    generator = torch.Generator().manual_seed(26072621)
    history = torch.randn(2, 3, 192, generator=generator)
    actions = torch.rand(2, 3, 10, generator=generator) * 2 - 1
    base = torch.randn(2, 192, generator=generator)
    mask = prefix_mask(torch.full((2,), length, dtype=torch.int64))
    return history, actions, mask, base


def test_canonical_action_is_source_independent_invertible_identity():
    action = torch.tensor([[-1.0, -0.25], [0.5, 1.0]], dtype=torch.float32)
    canonical = canonicalize_action(action)
    assert canonical.data_ptr() == action.data_ptr()
    assert torch.equal(physical_action(canonical), action)
    with pytest.raises(ValueError, match="outside"):
        canonicalize_action(torch.tensor([[1.01, 0.0]]))


@pytest.mark.parametrize(
    ("length", "expected"),
    [(1, [False, False, True]), (2, [False, True, True]), (3, [True, True, True])],
)
def test_prefix_mask_is_explicit_valid_suffix(length, expected):
    observed = prefix_mask(torch.tensor([length], dtype=torch.int64))
    assert observed.tolist() == [expected]


@pytest.mark.parametrize("length", [1, 2, 3])
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_masked_padding_values_cannot_affect_outputs(length, depth):
    model = MaskedStagewiseRefiner(verified_export=True).eval()
    history, actions, mask, base = inputs(length)
    changed_history = history.clone()
    changed_actions = actions.clone()
    changed_history[~mask] = 1e6
    changed_actions[~mask] = -1e6
    with torch.inference_mode():
        expected = model(history, actions, mask, base, depth)
        observed = model(changed_history, changed_actions, mask, base, depth)
    assert torch.equal(observed, expected)


def test_depth_zero_is_exact_base_and_has_no_stage_calls():
    model = MaskedStagewiseRefiner().eval()
    history, actions, mask, base = inputs(1)
    ledger = OperationLedger()
    output = model(history, actions, mask, base, 0, ledger=ledger)
    assert output is base
    assert ledger.base_calls == 1 and ledger.base_rows == 2
    assert ledger.stage_calls == [0, 0, 0, 0]
    assert ledger.stage_rows == [0, 0, 0, 0]


def test_positive_depth_fails_closed_without_verified_export():
    model = MaskedStagewiseRefiner().eval()
    history, actions, mask, base = inputs(1)
    with pytest.raises(RuntimeError, match="B3-verified"):
        model(history, actions, mask, base, 1)


def test_invalid_or_ambiguous_masks_fail_closed():
    model = MaskedStagewiseRefiner()
    history, actions, _, base = inputs(2)
    for invalid in (
        torch.tensor([[True, False, True], [True, False, True]]),
        torch.zeros(2, 3, dtype=torch.bool),
    ):
        with pytest.raises(ValueError, match="suffix|length"):
            model(history, actions, invalid, base, 1)


def test_positive_depth_stage_ledger_is_auditable():
    model = MaskedStagewiseRefiner(verified_export=True).eval()
    history, actions, mask, base = inputs(3)
    ledger = OperationLedger()
    with torch.inference_mode():
        model(history, actions, mask, base, 4, ledger=ledger)
    assert ledger.base_calls == 1 and ledger.base_rows == 2
    assert ledger.stage_calls == [1, 1, 1, 1]
    assert ledger.stage_rows == [2, 2, 2, 2]
