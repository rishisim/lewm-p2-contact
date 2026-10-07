"""Persisted-pool integrity and paired pin-fix analysis on small inputs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lewm_research.probe import analysis, readout


@pytest.fixture
def persisted_pool(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    rng = np.random.default_rng(4)
    states = rng.normal(size=(18, 9))
    groups = np.repeat([f"tiny.h5:{i}" for i in range(9)], 2)
    identity = hashlib.sha256(states.tobytes()+json.dumps(groups.tolist()).encode()).hexdigest()
    np.savez(source / "pool.npz", states=states, groups=groups)
    (source / "pool.json").write_text(json.dumps({"rows": [{"dataset": "tiny.h5", "row": i}
        for i in range(18)], "datasets": {"tiny.h5": {"selected": 18}}}))
    np.savez(source / "pixels_1.npz", pixels=np.zeros((18, 2, 2, 3), dtype="u1"))
    config = {"seed": 42, "pool_identity": identity, "batch_size": 2, "outer_folds": 3,
              "inner_folds": 2, "bootstrap_samples": 31, "dataset_sha256": {"tiny.h5": "abc"}}
    (source / "config.json").write_text(json.dumps(config))
    (source / "readout.json").write_text(json.dumps({"pool_identity": identity}))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.pt").write_bytes(b"test weights")
    metadata = {"dataset": {"name": "tiny.h5", "sha256": "abc"}, "train_episodes": [99]}
    torch.save({"metadata": metadata}, checkpoint / "trainer_state.pth")
    monkeypatch.setattr(readout, "checkpoint_dir", lambda name: checkpoint)
    monkeypatch.setattr(analysis, "prepare_output_root", lambda path: Path(path))
    closed = []
    class Planner:
        with_target = True
        def __init__(self, name, device):
            pass
        def close(self):
            closed.append(True)
    monkeypatch.setattr(readout, "Planner", Planner)
    features = {"cls": states[:, :3], "projected": states[:, 2:6]}
    def encode(planner, pixels, batch_size):
        assert len(pixels) == 18 and batch_size == 2
        return features
    monkeypatch.setattr(readout, "encode_pool", encode)
    return source, tmp_path / "output", checkpoint, states, groups, features, closed


def test_pool_readout_matches_main_procedure_and_preserves_source(persisted_pool):
    source, output, _, states, groups, features, closed = persisted_pool
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = readout.run_pool_readout(source, ["pin3_idm"], output, device="cpu")
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert (output / "pool.npz").read_bytes() == before["pool.npz"]
    assert (output / "pool.json").read_bytes() == before["pool.json"]
    assert (output / "pixels_1.npz").read_bytes() == before["pixels_1.npz"]
    assert not (output / "pixels_0.npz").exists()
    assert closed == [True]
    for kind, x in features.items():
        expected = readout.nested_readout(x, states[:, [7, 8, 2, 3]], groups,
                                         42, 3, 2, samples=31)
        got = result["checkpoints"]["pin3_idm"]["features"][kind]
        assert got["ridge"] == expected["ridge"]
        assert got["mean"] == expected["mean"]
        with np.load(output / f"pin3_idm_{kind}_oof.npz") as oof:
            for key in ("predictions", "errors", "folds"):
                assert np.array_equal(oof[key], expected[key])
    again = readout.run_pool_readout(source, ["pin3_idm"], output, device="cpu")
    assert again["checkpoints"] == result["checkpoints"]


def test_pool_readout_rejects_leakage_identity_and_seed(persisted_pool):
    source, output, checkpoint, states, groups, _, _ = persisted_pool
    with pytest.raises(ValueError, match="seed"):
        readout.run_pool_readout(source, ["pin3_idm"], output, seed=1)
    metadata = {"dataset": {"name": "tiny.h5", "sha256": "abc"}, "train_episodes": [1]}
    torch.save({"metadata": metadata}, checkpoint / "trainer_state.pth")
    with pytest.raises(ValueError, match="overlaps training"):
        readout.run_pool_readout(source, ["pin3_idm"], output)
    assert not output.exists()
    states[0, 7] += 1
    np.savez(source / "pool.npz", states=states, groups=groups)
    with pytest.raises(ValueError, match="pool identity"):
        readout.run_pool_readout(source, ["pin3_idm"], output)
    with pytest.raises(ValueError, match="separate"):
        readout.run_pool_readout(source, ["pin3_idm"], source)


def _analysis_module():
    path = Path(__file__).resolve().parents[1] / "experiments/role_swap/pin_fixes_analysis.py"
    spec = importlib.util.spec_from_file_location("pin_fixes_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pin_analysis_order_feasibility_pairing_and_seed(tmp_path):
    module = _analysis_module()
    main = tmp_path / "main"
    (main / "construction").mkdir(parents=True)
    (main / "construction/bases.json").write_text(json.dumps([{"id": b} for b in [3, 1, 2]]))
    def rows(scores):
        return [{"base_id": b, "condition": c, "score": {"success": scores[b][i]},
                 "seed": 100+b*2+i} for b in [3, 1, 2] for i, c in enumerate((module.HARD, module.CONTROL))]
    def save(root, data):
        root.mkdir(parents=True)
        (root / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in data))
    baseline = rows({3: [0, 1], 1: [0, 1], 2: [1, 1]})
    save(main / "ft_mixed_s0", baseline)
    save(main / "reference-F", rows({3: [1, 1], 1: [1, 0], 2: [1, 1]}))
    save(main / "reference-E", rows({3: [1, 1], 1: [1, 0], 2: [1, 1]}))
    pin = tmp_path / "pin"
    new = rows({3: [1, 1], 1: [0, 1], 2: [1, 1]})
    save(pin, new)
    result = module.analyze(main, {"pin3_idm": pin, "pin3_pegsup": tmp_path / "missing"}, tmp_path, n=2)
    assert result["base_ids"] == [3, 1]
    assert result["missing_arms"] == ["pin3_pegsup"]
    assert result["feasibility"]["included_base_ids"] == [3]
    arm = result["arms"]["pin3_idm"]
    assert arm["common_feasible"]["difference_of_gaps_vs_ft_mixed_s0"]["difference"] == -1
    assert arm["unconditional"]["difference_of_gaps_vs_ft_mixed_s0"]["difference"] == -.5
    assert arm["common_feasible"]["reference_E"]["gap_control_minus_hard"]["difference"] == 0
    new[0]["seed"] += 1
    (pin / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in new))
    with pytest.raises(ValueError, match="seed differs"):
        module.analyze(main, {"pin3_idm": pin}, tmp_path, n=2)
