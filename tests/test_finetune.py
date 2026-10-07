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
                "action": torch.ones(4, 2) * 0.1} for i in range(9)]
    seen = []
    def windows(dataset, normalization, seed):
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


def test_scratch_zero_step_loadable_checkpoint(tiny_training, monkeypatch):
    from hydra.utils import instantiate
    from lewm_research.lewm import load_lewm

    module, root, initial, _ = tiny_training
    config = {"_target_": f"{__name__}.TinyModel"}
    (root / "init" / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(module, "instantiate", instantiate)
    monkeypatch.setenv("STABLEWM_HOME", str(root))
    def refuse_weights(*args):
        raise AssertionError("scratch must not load init weights")
    monkeypatch.setattr(module, "load_lewm", refuse_weights)
    summary = module.finetune("unused", "scratch", init="init", from_scratch=True,
                              seed=42, device="cpu", workers=0)
    exported = load_lewm(root / "scratch", torch.device("cpu"))
    assert any(not torch.equal(exported.state_dict()[key], value)
               for key, value in initial.items())
    for filename in ("config.json", "normalization.json"):
        assert (root / "scratch" / filename).read_bytes() == (root / "init" / filename).read_bytes()
    assert exported.interpolate_pos_encoding is True
    assert summary["from_scratch"] is True
    assert summary["completed_steps"] == 0
    state = torch.load(root / "scratch" / "trainer_state.pth", weights_only=False)
    assert state["metadata"]["from_scratch"] is True


def test_scratch_deterministic_seed(tiny_training):
    module, root, _, _ = tiny_training
    states = []
    for name, seed in (("a", 42), ("b", 42), ("c", 43)):
        module.finetune("unused", name, init="init", from_scratch=True,
                        seed=seed, device="cpu", workers=0)
        states.append(torch.load(root / name / "weights.pt", weights_only=True))
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    assert any(not torch.equal(states[0][key], states[2][key]) for key in states[0])


def test_scratch_resume_flag_mismatch(tiny_training):
    module, _, _, _ = tiny_training
    kwargs = dict(dataset="unused", name="scratch", init="init", device="cpu", workers=0)
    module.finetune(**kwargs, from_scratch=True)
    with pytest.raises(ValueError, match="Resume metadata mismatch: from_scratch"):
        module.finetune(**kwargs)
    assert module.finetune(**kwargs, from_scratch=True)["start_step"] == 0
    module.finetune(**{**kwargs, "name": "loaded"})
    with pytest.raises(ValueError, match="Resume metadata mismatch: from_scratch"):
        module.finetune(**{**kwargs, "name": "loaded"}, from_scratch=True)


def test_non_scratch_resume_accepts_old_metadata(tiny_training):
    module, root, _, _ = tiny_training
    kwargs = dict(dataset="unused", name="old", init="init", device="cpu", workers=0)
    module.finetune(**kwargs)
    path = root / "old" / "trainer_state.pth"
    state = torch.load(path, weights_only=False)
    del state["metadata"]["from_scratch"]
    torch.save(state, path)
    assert module.finetune(**kwargs)["start_step"] == 0


def test_scratch_disables_hf_pretrained(monkeypatch):
    from lewm_research.train import finetune as module
    config = {"encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf",
                          "pretrained": True}}
    calls = []
    monkeypatch.setattr(module, "instantiate", lambda *args, **kwargs: calls.append((args, kwargs)))
    module._random_model(config)
    assert calls == [((config,), {"encoder": {"pretrained": False}})]
    assert config["encoder"]["pretrained"] is True


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
