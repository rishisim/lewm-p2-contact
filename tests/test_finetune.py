"""Fast checks for W3 normalization, episode isolation, and exact objective."""

import hashlib
import json
import os
import shutil

import numpy as np
import pytest
import torch
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
    valid = torch.from_numpy(rows[:2])
    assert loaded["columns"]["action"]["mean"] == valid.mean(0).tolist()
    assert loaded["columns"]["action"]["std"] == valid.std(0).tolist()
    torch.testing.assert_close(normalize_column(valid, loaded, "action"),
                               (valid - valid.mean(0)) / valid.std(0), rtol=0, atol=0)


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


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def encode(self, batch):
        return {"emb": self.linear(batch["pixels"]), "act_emb": batch["action"]}

    def predict(self, emb, action):
        return emb + action


@pytest.fixture
def tiny_training(tmp_path, monkeypatch):
    from lewm_research.train import finetune as module
    source = tmp_path / "init"
    source.mkdir()
    (source / "config.json").write_text("{}")
    stats = {"source": "tiny.h5", "method": "torch_sample_std", "columns": {}}
    save_normalization(stats, source)
    torch.manual_seed(9)
    initial = TinyModel().state_dict()
    torch.save(initial, source / "weights.pt")
    (tmp_path / "tiny.h5").write_bytes(b"tiny dataset")
    (tmp_path / "unused.h5").write_bytes(b"unused dataset")
    def load_model(*args):
        model = TinyModel()
        model.load_state_dict(initial)
        return model
    monkeypatch.setattr(module, "load_lewm", load_model)
    monkeypatch.setattr(module, "instantiate", lambda config: TinyModel())
    monkeypatch.setattr(module, "checkpoint_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(module, "dataset_path", lambda name: tmp_path / name)
    monkeypatch.setattr(module, "create_run", lambda *args: tmp_path)
    samples = [{"pixels": torch.full((4, 2), i / 10),
                "action": torch.ones(4, 2) * 0.1,
                "state": torch.full((4, 9), 128.)} for i in range(9)]
    seen = []
    def windows(dataset, normalization, seed, pegsup=False):
        seen.append(normalization)
        return samples, samples[:3], [seed, 10], [20]
    monkeypatch.setattr(module, "load_windows", windows)
    return module, tmp_path, initial, seen


def test_zero_step_checkpoint_parity(tiny_training):
    module, root, initial, _ = tiny_training
    summary = module.finetune("unused", "zero", init="init", device="cpu", workers=0)
    assert summary["completed_steps"] == 0
    state = torch.load(root / "zero" / "trainer_state.pth", weights_only=False)
    exported = torch.load(root / "zero" / "weights.pt", weights_only=True)
    for key in initial:
        torch.testing.assert_close(state["model"][key], initial[key], rtol=0, atol=0)
        torch.testing.assert_close(exported[key], initial[key], rtol=0, atol=0)
    assert "Default 32" in summary["batch_size_deviation"]


def test_loader_spawn_trains_four_workers():
    from lewm_research.train.finetune import _loader
    # macOS uses spawn by default; explicitly exercise spawn on other platforms.
    from torch.utils.data import DataLoader, TensorDataset
    from lewm_research.train.finetune import seed_worker
    loader = DataLoader(TensorDataset(torch.arange(32.).reshape(16, 2)),
                        batch_size=4, num_workers=4, multiprocessing_context="spawn",
                        worker_init_fn=seed_worker)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for (batch,) in loader:
        optimizer.zero_grad()
        model(batch).square().mean().backward()
        optimizer.step()
    assert torch.isfinite(model.weight).all()


def test_training_drops_tail_validation_keeps_it():
    from lewm_research.train.finetune import _loader
    train = _loader(list(range(9)), 4, 0, 0, True)
    val = _loader(list(range(9)), 4, 0, 0, False)
    assert [len(batch) for batch in train] == [4, 4]
    assert [len(batch) for batch in val] == [4, 4, 1]
    with pytest.raises(ValueError, match="Zero training batches"):
        _loader([1], 2, 0, 0, True)


def test_upstream_scheduler_trace_and_resume():
    from lewm_research.train.finetune import _scheduler, LR
    from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
    def setup(factory):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=LR)
        return parameter, optimizer, factory(optimizer)
    a, opt, sched = setup(lambda opt: _scheduler(opt, 250))
    b, ref_opt, ref = setup(lambda opt: LinearWarmupCosineAnnealingLR(
        opt, warmup_steps=2, max_steps=250))
    assert sched.warmup_steps == 2
    assert opt.param_groups[0]["lr"] == 0
    for _ in range(250):
        assert opt.param_groups[0]["lr"] == ref_opt.param_groups[0]["lr"]
        a.grad = b.grad = torch.ones(1)
        opt.step(); sched.step()
        ref_opt.step(); ref.step()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert opt.param_groups[0]["lr"] == 0


@pytest.mark.parametrize("key", ["dataset", "seed", "batch_size", "init",
                                 "schedule_steps", "train_episodes", "val_episodes"])
def test_resume_metadata_refuses_mismatch(key):
    from lewm_research.train.finetune import _validate_resume
    metadata = {"dataset": {"name": "a.h5", "size": 1, "sha256": "a" * 64},
                "seed": 0, "batch_size": 32,
                "init": {"name": "init", "size": 1, "sha256": "b" * 64},
                "schedule_steps": 40, "train_episodes": [0], "val_episodes": [1]}
    state = {"version": 2, "metadata": metadata}
    _validate_resume(state, metadata)
    _validate_resume(state, {**metadata,
                            "dataset": {**metadata["dataset"], "name": "renamed.h5"},
                            "init": {**metadata["init"], "name": "renamed-init"}})
    with pytest.raises(ValueError, match=key):
        _validate_resume(state, {**metadata, key: "different"})
    with pytest.raises(ValueError, match="Legacy"):
        _validate_resume({}, metadata)
    with pytest.raises(ValueError, match="version 2 content fingerprints"):
        _validate_resume({"version": 1, "metadata": metadata}, metadata)


def test_dataset_fingerprint_cache(tmp_path, monkeypatch):
    from lewm_research.train.finetune import _fingerprint
    path = tmp_path / "tiny.h5"
    data = b"abc" * 1_000_000  # More than one streaming chunk.
    path.write_bytes(data)
    expected = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert _fingerprint(path, cache=True) == expected
    cache = path.with_name("tiny.h5.sha256")
    saved = json.loads(cache.read_text())
    assert saved == {**expected, "mtime_ns": path.stat().st_mtime_ns}
    real_open = type(path).open
    def no_rehash(self, *args, **kwargs):
        if self == path:
            raise AssertionError("Valid cache must avoid reading the dataset")
        return real_open(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(type(path), "open", no_rehash)
        assert _fingerprint(path, cache=True) == expected
    path.write_bytes(b"z" + data[1:])
    os.utime(path, ns=(saved["mtime_ns"] + 1, saved["mtime_ns"] + 1))
    changed = _fingerprint(path, cache=True)
    assert changed["size"] == expected["size"]
    assert changed["sha256"] != expected["sha256"]
    cache.write_text("broken cache")
    assert _fingerprint(path, cache=True) == changed
    path.write_bytes(data + b"extra")
    os.utime(path, ns=(saved["mtime_ns"] + 1, saved["mtime_ns"] + 1))
    assert _fingerprint(path, cache=True)["size"] == len(data) + 5


def test_resume_portable_work_root_and_content_checks(tiny_training, monkeypatch):
    import h5py
    from lewm_research import paths
    module, root, _, _ = tiny_training
    a, b = root / "machineA", root / "machineB"
    monkeypatch.setenv("LEWM_WORK_ROOT", str(a))
    monkeypatch.setattr(module, "checkpoint_dir", paths.checkpoint_dir)
    monkeypatch.setattr(module, "dataset_path", paths.dataset_path)
    source = paths.checkpoint_dir("init")
    shutil.copytree(root / "init", source)
    dataset = paths.dataset_path("tiny.h5")
    dataset.parent.mkdir(parents=True)
    with h5py.File(dataset, "w") as file:
        file["pixels"] = np.arange(9 * 4 * 2, dtype="f4").reshape(9, 4, 2) / 100
        file["action"] = np.full((9, 4, 2), 0.1, dtype="f4")
    def windows(name, stats, seed):
        with h5py.File(paths.dataset_path(name if name.endswith(".h5") else name + ".h5"), "r") as file:
            samples = [{key: torch.from_numpy(file[key][i]) for key in file}
                       for i in range(9)]
        return samples, samples[:3], [0], [1]
    monkeypatch.setattr(module, "load_windows", windows)
    kwargs = dict(dataset="tiny", name="ported", init="init", batch_size=4,
                  workers=0, device="cpu", schedule_steps=4,
                  val_interval=1, checkpoint_interval=1)
    module.finetune(max_steps=2, **{**kwargs, "dataset": "tiny.h5"})
    shutil.copytree(a, b)
    monkeypatch.setenv("LEWM_WORK_ROOT", str(b))
    # Force cache invalidation as copying to a new machine can change mtimes.
    dataset_b = paths.dataset_path("tiny.h5")
    stamp = dataset_b.stat().st_mtime_ns + 1_000_000
    os.utime(dataset_b, ns=(stamp, stamp))
    summary = module.finetune(max_steps=4, **kwargs)
    assert (summary["start_step"], summary["completed_steps"]) == (2, 4)
    checkpoint = paths.checkpoint_dir("ported")
    state = torch.load(checkpoint / "trainer_state.pth", weights_only=False)
    assert state["metadata"]["dataset"]["name"] == "tiny.h5"
    assert state["metadata"]["init"]["name"] == "init"
    assert str(a) not in json.dumps(state["metadata"])
    assert str(b) not in json.dumps(state["metadata"])
    assert [json.loads(row)["step"] for row in
            (checkpoint / "train_log.jsonl").read_text().splitlines()] == [1, 2, 3, 4]
    with dataset_b.open("r+b") as file:
        byte = file.read(1)
        file.seek(0)
        file.write(bytes([byte[0] ^ 1]))
    os.utime(dataset_b, ns=(stamp + 1, stamp + 1))
    with pytest.raises(ValueError, match="Resume metadata mismatch: dataset"):
        module.finetune(max_steps=4, **kwargs)
    shutil.copyfile(dataset, dataset_b)
    weights = paths.checkpoint_dir("init") / "weights.pt"
    with weights.open("r+b") as file:
        byte = file.read(1)
        file.seek(0)
        file.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="Resume metadata mismatch: init"):
        module.finetune(max_steps=4, **kwargs)


def test_resume_atomic_state_and_rng_parity(tiny_training):
    module, root, _, seen = tiny_training
    kwargs = dict(dataset="tiny", init="init", batch_size=4, workers=0,
                  device="cpu", schedule_steps=4, val_interval=1, checkpoint_interval=1)
    module.finetune(name="full", max_steps=4, **kwargs)
    module.finetune(name="split", max_steps=2, **kwargs)
    # Resume must neither read initialization normalization nor exported weights.
    (root / "init" / "normalization.json").unlink()
    (root / "split" / "weights.pt").write_bytes(b"broken inference export")
    summary = module.finetune(name="split", max_steps=4, **kwargs)
    assert summary["start_step"] == 2
    assert seen[-1] == load_normalization(root / "split")
    full = torch.load(root / "full" / "trainer_state.pth", weights_only=False)
    split = torch.load(root / "split" / "trainer_state.pth", weights_only=False)
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], split["model"][key], rtol=0, atol=0)
    assert full["scheduler"] == split["scheduler"]
    assert (full["step"], full["epoch"], full["batch_index"]) == (4, 1, 2)
    torch.testing.assert_close(full["rng"]["torch"], split["rng"]["torch"], rtol=0, atol=0)
    for param, values in full["optimizer"]["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, split["optimizer"]["state"][param][key], rtol=0, atol=0)
    assert not list((root / "split").glob("*.tmp"))


def test_atomic_checkpoint_survives_export_failure(tiny_training, monkeypatch):
    module, root, _, _ = tiny_training
    module.finetune("tiny", "split", init="init", max_steps=0, workers=0, device="cpu")
    state = torch.load(root / "split" / "trainer_state.pth", weights_only=False)
    model = TinyModel()
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters())
    real_save = module._atomic_save
    def fail_export(value, target):
        if target.name == "weights.pt":
            raise OSError("interrupted export")
        real_save(value, target)
    monkeypatch.setattr(module, "_atomic_save", fail_export)
    with pytest.raises(OSError, match="interrupted"):
        module._save(model, optimizer, None, root / "split", 3, 1, 2,
                     state["metadata"], {})
    saved = torch.load(root / "split" / "trainer_state.pth", weights_only=False)
    assert saved["step"] == 3
    assert all(key in saved for key in ("model", "optimizer", "scheduler", "rng", "metadata"))


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_normalization_exact_upstream_dtype_and_sample_std(tmp_path, dtype):
    import h5py
    rows = np.random.default_rng(8).normal(size=(101, 4)).astype(dtype)
    rows[3, 1] = np.nan
    path = tmp_path / "normalization.h5"
    with h5py.File(path, "w") as file:
        file["action"] = rows
        file["proprio"] = rows
    stats = fit_normalization(path, chunk_size=7)
    data = torch.from_numpy(np.array(rows))
    data = data[~torch.isnan(data).any(dim=1)]
    for column in ("action", "proprio"):
        assert stats["columns"][column]["mean"] == data.mean(0, keepdim=True).flatten().tolist()
        assert stats["columns"][column]["std"] == data.std(0, keepdim=True).flatten().tolist()
        assert stats["columns"][column]["valid_rows"] == 100


def test_precision_selection():
    from lewm_research.train.finetune import _autocast

    assert _autocast(torch.device("cpu"), "auto")[1] == "fp32"
    assert _autocast(torch.device("cpu"), "fp32")[1] == "fp32"
    with pytest.raises(ValueError):
        _autocast(torch.device("cpu"), "bf16")
    with pytest.raises(ValueError):
        _autocast(torch.device("cpu"), "fp16")


def test_zero_auxiliary_loss_and_rng_match():
    from lewm_research.train.finetune import _aux_heads
    model = TinyModel()
    sigreg = SIGReg(knots=17, num_proj=32)
    batch = {"pixels": torch.randn(2, 4, 2), "action": torch.randn(2, 4, 2)}
    torch.manual_seed(123)
    expected = lejepa_loss(model, sigreg, {k: v.clone() for k, v in batch.items()})
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)
    heads = _aux_heads(2, 2, 9, "cpu")
    assert not heads
    got = lejepa_loss(model, sigreg, {k: v.clone() for k, v in batch.items()}, heads,
                      idm_enc_weight=0, idm_pred_weight=0, pegsup_weight=0)
    assert got.keys() == expected.keys()
    assert all(torch.equal(got[k], expected[k]) for k in got)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    before = torch.get_rng_state()
    heads = _aux_heads(2, 2, 9, "cpu", idm=True, pegsup=True)
    assert torch.equal(before, torch.get_rng_state())
    again = _aux_heads(2, 2, 9, "cpu", idm=True, pegsup=True)
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in heads.state_dict().items())
    assert heads["idm"][0].in_features == 4
    assert heads["idm"][-1].out_features == 2


def test_idm_values_and_detached_context():
    class Difference(torch.nn.Module):
        def forward(self, pair):
            left, right = pair.chunk(2, -1)
            return right - left
    z = torch.tensor([[[0., 0.], [1., 2.], [3., 5.], [6., 9.]]], requires_grad=True)
    pred = torch.tensor([[[2., 3.], [4., 6.], [8., 12.]]], requires_grad=True)
    class Model:
        def encode(self, batch):
            return {"emb": z, "act_emb": batch["action"]}
        def predict(self, emb, action):
            return pred
    batch = {"action": torch.tensor([[[0., 1.], [1., 1.], [2., 2.], [99., 99.]]])}
    losses = lejepa_loss(Model(), lambda emb: emb.sum()*0, batch,
                         {"idm": Difference()}, idm_enc_weight=1, idm_pred_weight=1)
    # Encoder residual squared sums are 2, 5, 5; predictor residual is (3, 5).
    assert losses["idm_enc_loss"].item() == 4
    assert losses["idm_pred_loss"].item() == 34
    losses["idm_pred_loss"].backward(retain_graph=True)
    assert z.grad is None
    assert torch.equal(pred.grad, torch.tensor([[[0., 0.], [0., 0.], [6., 10.]]]))
    losses["idm_enc_loss"].backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_pegsup_values_all_frames():
    class Model:
        def encode(self, batch):
            return {"emb": batch["pixels"], "act_emb": batch["action"]}
        def predict(self, emb, action):
            return emb
    state = torch.full((1, 4, 9), -999.)
    state[..., [7, 8, 2, 3]] = 256.
    state[:, 0, [7, 8, 2, 3]] = torch.tensor([0., 256., 512., 768.])
    batch = {"pixels": torch.zeros(1, 4, 4), "action": torch.zeros(1, 4, 4), "state": state}
    losses = lejepa_loss(Model(), lambda emb: emb.sum()*0, batch,
                         {"pegsup": torch.nn.Identity()}, pegsup_weight=2)
    assert losses["pegsup_loss"].item() == 0.375
    assert losses["loss"].item() == 0.75


def test_zero_auxiliary_training_parity(tiny_training):
    module, root, _, _ = tiny_training
    kwargs = dict(init="init", max_steps=2, schedule_steps=4, batch_size=4,
                  workers=0, device="cpu", val_interval=2)
    module.finetune("tiny", "default", **kwargs)
    module.finetune("tiny", "explicit_zero", **kwargs,
                    idm_enc_weight=0, idm_pred_weight=0, pegsup_weight=0)
    def state(name):
        return torch.load(root / name / "trainer_state.pth", weights_only=False)
    a, b = state("default"), state("explicit_zero")
    assert a["metadata"] == b["metadata"] and "aux" not in a["metadata"]
    assert "aux_heads" not in a and "aux_heads" not in b
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    for key, value in torch.load(root / "default/weights.pt", weights_only=True).items():
        assert torch.equal(value, torch.load(root / "explicit_zero/weights.pt", weights_only=True)[key])
    def logs(name):
        return [{k: v for k, v in json.loads(line).items() if "loss" in k}
                for line in (root / name / "train_log.jsonl").read_text().splitlines()]
    assert logs("default") == logs("explicit_zero")


@pytest.mark.parametrize("weights", [dict(idm_enc_weight=1., idm_pred_weight=1.),
                                     dict(pegsup_weight=1.),
                                     dict(idm_enc_weight=1., idm_pred_weight=1., pegsup_weight=1.)])
def test_auxiliary_export_resume_and_mismatch(tiny_training, weights):
    module, root, _, _ = tiny_training
    kwargs = dict(init="init", schedule_steps=4, batch_size=4, workers=0,
                  device="cpu", val_interval=2, **weights)
    module.finetune("tiny", "full_aux", max_steps=4, **kwargs)
    module.finetune("tiny", "split_aux", max_steps=2, **kwargs)
    state = torch.load(root / "split_aux/trainer_state.pth", weights_only=False)
    assert state["aux_heads"]
    initialized = module._aux_heads(2, 2, 0, "cpu",
        idm=weights.get("idm_enc_weight", 0) > 0 or weights.get("idm_pred_weight", 0) > 0,
        pegsup=weights.get("pegsup_weight", 0) > 0).state_dict()
    assert any(not torch.equal(v, initialized[k]) for k, v in state["aux_heads"].items())
    exported = torch.load(root / "split_aux/weights.pt", weights_only=True)
    assert exported.keys() == TinyModel().state_dict().keys()
    with pytest.raises(ValueError, match="aux"):
        module.finetune("tiny", "split_aux", max_steps=4, **{**kwargs, next(iter(weights)): 0.5})
    with pytest.raises(ValueError, match="aux"):
        module.finetune("tiny", "split_aux", init="init", max_steps=4, schedule_steps=4,
                        batch_size=4, workers=0, device="cpu")
    module.finetune("tiny", "split_aux", max_steps=4, **kwargs)
    a = torch.load(root / "full_aux/trainer_state.pth", weights_only=False)
    b = torch.load(root / "split_aux/trainer_state.pth", weights_only=False)
    for section in ("model", "aux_heads"):
        assert all(torch.equal(v, b[section][k]) for k, v in a[section].items())
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    log = json.loads((root / "split_aux/train_log.jsonl").read_text().splitlines()[-1])
    for key, weight in weights.items():
        if weight:
            term = key.replace("_weight", "_loss")
            assert f"train_{term}" in log and f"val_{term}" in log


def test_state_preprocess_alignment_and_conditional_loading(tmp_path, monkeypatch):
    import h5py
    from lewm_research.train import finetune as module
    path = tmp_path / "aligned.h5"
    with h5py.File(path, "w") as file:
        file["episode_idx"] = np.repeat(np.arange(2), 30)
        file["ep_len"] = np.array([30, 30])
        file["ep_offset"] = np.array([0, 30])
        file["pixels"] = np.zeros((60, 16, 16, 3), dtype="u1")
        file["state"] = np.repeat(np.arange(60, dtype="f4")[:, None], 9, axis=1)
        file["action"] = np.repeat(np.arange(60, dtype="f4")[:, None], 2, axis=1)
        file["proprio"] = np.zeros((60, 2), dtype="f4")
    stats = {"columns": {key: {"mean": [0., 0.], "std": [1., 1.]}
                         for key in ("action", "proprio")}}
    monkeypatch.setattr(module, "dataset_path", lambda name: path)
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "cache"))
    train, _, _, _ = module.load_windows("aligned", stats, 0, pegsup=True)
    sample = train[0]
    start = sample["state"][0, 0].item()
    assert sample["pixels"].shape[0] == sample["state"].shape[0] == 4
    assert torch.equal(sample["state"][:, 0], torch.arange(start, start+20, 5))
    assert torch.equal(sample["action"], torch.arange(start, start+20).repeat_interleave(2).reshape(4, 10))
    train, _, _, _ = module.load_windows("aligned", stats, 0)
    assert "state" not in train[0]


def test_validation_keeps_original_metrics_with_auxiliary_heads():
    from lewm_research.train.finetune import _aux_heads, _validation
    model = TinyModel()
    sigreg = SIGReg(knots=17, num_proj=32)
    batch = {"pixels": torch.randn(2, 4, 2), "action": torch.randn(2, 4, 2),
             "state": torch.zeros(2, 4, 9)}
    heads = _aux_heads(2, 2, 0, "cpu", idm=True, pegsup=True)
    torch.manual_seed(4)
    expected = _validation(model, sigreg, [batch], torch.device("cpu"))
    rng = torch.get_rng_state()
    torch.manual_seed(4)
    got = _validation(model, sigreg, [batch], torch.device("cpu"), heads,
                      dict(idm_enc_weight=1, idm_pred_weight=1, pegsup_weight=1))
    assert all(got[key] == value for key, value in expected.items())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(key in got for key in ("val_idm_enc_loss", "val_idm_pred_loss", "val_pegsup_loss"))


def test_idm_predictor_gradients_reach_action_encoder_and_head():
    z = torch.randn(2, 4, 2, requires_grad=True)
    action_encoder = torch.nn.Linear(2, 2)
    predictor = torch.nn.Linear(2, 2)
    head = torch.nn.Linear(4, 2)
    class Model:
        def encode(self, batch):
            return {"emb": z, "act_emb": action_encoder(batch["action"])}
        def predict(self, emb, action):
            # Isolate the detached head input from any encoder path via predict.
            return predictor(action)
    losses = lejepa_loss(Model(), lambda emb: emb.sum()*0,
        {"action": torch.randn(2, 4, 2)}, {"idm": head}, idm_pred_weight=1)
    losses["idm_pred_loss"].backward()
    assert z.grad is None
    for module in (action_encoder, predictor, head):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
