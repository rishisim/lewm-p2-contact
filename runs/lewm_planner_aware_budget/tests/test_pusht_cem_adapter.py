import sys
from pathlib import Path

import torch
from torch import nn
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pusht_cem_adapter import PushTRefinedCostModel
from pusht_refiner import MaskedStagewiseRefiner


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.action_encoder = nn.Identity()

    def encode(self, info):
        pixels = info["pixels"]
        info["emb"] = pixels[..., 0, 0, :192]
        return info

    def predict(self, history, actions):
        return history + 0.01 * actions.mean(-1, keepdim=True)


def fake_info(batch=1, samples=4):
    pixels = torch.zeros(batch, samples, 1, 1, 1, 192)
    return {
        "pixels": pixels,
        "goal": pixels.clone(),
        "action": torch.zeros(batch, samples, 1, 10),
    }


def test_minimal_cem_loop_invokes_requested_fixed_depth_and_preserves_shape():
    refiner = MaskedStagewiseRefiner(verified_export=True).eval().requires_grad_(False)
    model = PushTRefinedCostModel(FakeBase(), refiner, 2)
    candidates = torch.zeros(1, 4, 5, 10)
    for _ in range(3):
        costs = model.get_cost(fake_info(), candidates)
        assert costs.shape == (1, 4)
    assert model.ledger.base_calls == 15
    assert model.ledger.base_rows == 60
    assert model.ledger.stage_calls == [15, 15, 0, 0]
    assert model.ledger.stage_rows == [60, 60, 0, 0]


def test_depth_zero_delegates_exactly_to_base():
    class DelegatingBase(FakeBase):
        def get_cost(self, info, candidates):
            return candidates.sum(dim=(2, 3))

    base = DelegatingBase()
    model = PushTRefinedCostModel(base, None, 0)
    candidates = torch.randn(1, 4, 5, 10)
    assert torch.equal(model.get_cost({}, candidates), base.get_cost({}, candidates))
    assert model.ledger.base_calls == 5
    assert model.ledger.base_rows == 20


def test_hash_pinned_export_matches_dense_references_and_is_frozen():
    checkpoint = ROOT / "pusht_refiner_checkpoint.pt"
    references = ROOT / "pusht_refiner_references.npz"
    manifest_path = ROOT / "pusht_refiner_manifest.json"
    if not all(path.exists() for path in (checkpoint, references, manifest_path)):
        return
    manifest = json.loads(manifest_path.read_text())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MaskedStagewiseRefiner(verified_export=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    import numpy as np

    with np.load(references, allow_pickle=False) as stored:
        history = torch.from_numpy(stored["history"])
        actions = torch.from_numpy(stored["actions"])
        mask = torch.from_numpy(stored["mask"])
        base = torch.from_numpy(stored["base"])
        with torch.inference_mode():
            for depth in (1, 2, 4):
                observed = model(history, actions, mask, base, depth)
                torch.testing.assert_close(
                    observed,
                    torch.from_numpy(stored[f"depth_{depth}"]),
                    rtol=manifest["cross_device_reference_rtol"],
                    atol=manifest["cross_device_reference_atol"],
                )
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert len(manifest["checkpoint_sha256"]) == 64
