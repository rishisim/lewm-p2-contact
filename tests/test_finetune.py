"""Fast checks for W3 normalization, episode isolation, and exact objective."""

import json
import shutil

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from stable_worldmodel.wm.loss import SIGReg

from lewm_research.normalization import (fit_normalization, load_normalization,
                                         normalize_column, save_normalization)
from lewm_research.train.finetune import episode_split, lejepa_loss


def test_normalization_roundtrip(tmp_path):
    import h5py

    rows = np.array([[1., 2.], [3., 4.], [np.nan, 5.]], dtype="f4")
    source = tmp_path / "tiny.h5"
    with h5py.File(source, "w") as file:
        file["action"] = rows
        file["proprio"] = rows
    stats = fit_normalization(source, chunk_size=2)
    save_normalization(stats, tmp_path / "checkpoint")
    loaded = load_normalization(tmp_path / "checkpoint")
    assert loaded == stats
    expected = StandardScaler().fit(rows[:2])
    assert loaded["columns"]["action"]["mean"] == pytest.approx(expected.mean_)
    assert loaded["columns"]["action"]["std"] == pytest.approx(expected.scale_)
    torch.testing.assert_close(normalize_column(torch.tensor(rows[:2]), loaded, "action"),
                               torch.tensor(expected.transform(rows[:2])))


def test_episode_split_disjoint():
    class Dataset:
        lengths = np.array([40] * 6)
        clip_indices = [(ep, start) for ep in range(6) for start in range(21)]

    train, val, train_eps, val_eps = episode_split(Dataset(), 17)
    assert not set(train_eps) & set(val_eps)
    assert {Dataset.clip_indices[i][0] for i in train.indices} == set(train_eps)
    assert {Dataset.clip_indices[i][0] for i in val.indices} == set(val_eps)


def test_loss_matches_upstream_expression():
    class SmallModel:
        def encode(self, batch):
            return {"emb": batch["pixels"], "act_emb": batch["action"]}

        def predict(self, emb, action):
            return emb + action

    model = SmallModel()
    sigreg = SIGReg(knots=17, num_proj=1024)
    pixels = torch.randn(2, 4, 8)
    actions = torch.randn(2, 4, 8)
    actions[0, 0, 0] = float("nan")
    torch.manual_seed(123)
    got = lejepa_loss(model, sigreg, {"pixels": pixels, "action": actions.clone()})
    torch.manual_seed(123)
    act = torch.nan_to_num(actions, 0.0)
    emb = pixels
    pred = model.predict(emb[:, :3], act[:, :3])
    pred_loss = (pred - emb[:, 1:]).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))
    torch.testing.assert_close(got["pred_loss"], pred_loss)
    torch.testing.assert_close(got["sigreg_loss"], sigreg_loss)
    torch.testing.assert_close(got["loss"], pred_loss + 0.09 * sigreg_loss)


def test_zero_step_checkpoint_parity(tmp_path, monkeypatch):
    from lewm_research import paths
    from lewm_research.lewm import load_lewm
    from lewm_research.train import finetune as module

    pretrained = paths.checkpoint_dir("lewm-pusht")
    if not (pretrained / "normalization.json").exists():
        pytest.skip("Pretrained normalization has not been generated")
    monkeypatch.setattr(module, "checkpoint_dir",
                        lambda name: pretrained if name == "lewm-pusht" else tmp_path / name)
    monkeypatch.setattr(module, "create_run", lambda stage, config: tmp_path)
    summary = module.finetune("unused", "zero", max_steps=0, device="cpu", workers=0)
    assert summary["completed_steps"] == 0
    left = torch.load(pretrained / "weights.pt", map_location="cpu", weights_only=True)
    right = torch.load(tmp_path / "zero" / "weights.pt", map_location="cpu", weights_only=True)
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)
    assert json.loads((tmp_path / "zero" / "normalization.json").read_text()) == load_normalization(pretrained)
    first = load_lewm(pretrained, torch.device("cpu"))
    second = load_lewm(tmp_path / "zero", torch.device("cpu"))
    pixels = torch.randn(1, 4, 3, 224, 224)
    actions = torch.randn(1, 4, 10)
    with torch.inference_mode():
        a = first.encode({"pixels": pixels, "action": actions})
        b = second.encode({"pixels": pixels, "action": actions})
        torch.testing.assert_close(a["emb"], b["emb"], rtol=0, atol=0)
        torch.testing.assert_close(first.predict(a["emb"][:, :3], a["act_emb"][:, :3]),
                                   second.predict(b["emb"][:, :3], b["act_emb"][:, :3]),
                                   rtol=0, atol=0)

    # Reload a changed state too, so parity is not limited to the pretrained copy.
    with torch.no_grad():
        next(second.parameters()).add_(0.001)
    changed = tmp_path / "changed"
    changed.mkdir()
    shutil.copyfile(pretrained / "config.json", changed / "config.json")
    torch.save(second.state_dict(), changed / "weights.pt")
    reloaded = load_lewm(changed, torch.device("cpu"))
    with torch.inference_mode():
        b = second.encode({"pixels": pixels, "action": actions})
        c = reloaded.encode({"pixels": pixels, "action": actions})
        torch.testing.assert_close(b["emb"], c["emb"], rtol=0, atol=0)
        torch.testing.assert_close(second.predict(b["emb"][:, :3], b["act_emb"][:, :3]),
                                   reloaded.predict(c["emb"][:, :3], c["act_emb"][:, :3]),
                                   rtol=0, atol=0)
