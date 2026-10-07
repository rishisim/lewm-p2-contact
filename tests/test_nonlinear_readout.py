"""Fast planted-signal checks and frozen pin-rule/cache guards."""

import json

import numpy as np
import pytest

from lewm_research.probe import nonlinear_readout as nr
from lewm_research.probe.readout import grouped_folds, nested_readout


def test_mlp_recovers_nonlinear_signal():
    rng = np.random.default_rng(9)
    x = rng.uniform(-2, 2, (800, 2))
    signal = np.column_stack((abs(x[:,0]), abs(x[:,1]), abs(x[:,0])+abs(x[:,1]), abs(x[:,0])-abs(x[:,1])))
    y = signal+rng.normal(0, .025, signal.shape)
    groups = np.repeat(np.arange(40), 20)
    result = nr.mlp_readout(x, y, groups, seed=9, outer_folds=3, hidden=(32,32),
                           max_epochs=150, patience=15, batch_size=128)
    ridge = nested_readout(x, y, groups, seed=9, outer_folds=3, samples=100)
    assert result["errors"].shape == (800,2)
    assert result["predictions"].shape == y.shape
    assert result["errors"].mean() < .08  # noise-only Euclidean mean is about .031
    assert result["errors"].mean() < .15*ridge["errors"].mean()
    np.testing.assert_array_equal(result["folds"], ridge["folds"])
    assert all(1 <= r["peg_best_epoch"] <= r["peg_epochs"] <= 150 for r in result["early_stopping"])


def test_outer_and_inner_episode_disjointness():
    groups = np.repeat(np.arange(25), np.arange(1,26))
    seen = np.zeros(len(groups), int)
    for fold, (train, test) in enumerate(grouped_folds(groups, 5, 42)):
        itrain, ival = nr.inner_split(groups[train], seed=42+fold+1)
        a, b = train[itrain], train[ival]
        assert not set(groups[a]) & set(groups[b])
        assert not set(groups[train]) & set(groups[test])
        assert len(np.unique(groups[b])) == 4
        np.testing.assert_array_equal(np.sort(np.r_[a,b]), train)
        again = nr.inner_split(groups[train], seed=42+fold+1)
        np.testing.assert_array_equal(ival, again[1])
        seen[test] += 1
    np.testing.assert_array_equal(seen, 1)
    with pytest.raises(ValueError, match="two episodes"):
        nr.inner_split(np.zeros(5))


def test_knn_recovers_smooth_target_and_mean():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1,1,(600,1))
    y = np.column_stack((np.sin(x[:,0]), np.cos(x[:,0]), x[:,0], x[:,0]**2))
    groups = np.repeat(np.arange(30),20)
    result = nr.knn_readout(x, y, groups, seed=42)
    assert result["errors"].mean() < .025
    expected_mean = np.empty_like(y)
    for fold, (train, test) in enumerate(grouped_folds(groups,5,42)):
        assert np.all(result["folds"][test] == fold)
        expected_mean[test] = y[train].mean(0)
    errors = np.linalg.norm((expected_mean-y).reshape(-1,2,2),axis=-1)
    assert result["mean"]["peg"]["mean"] == pytest.approx(errors[:,0].mean())
    # Constant columns and changes of units must not alter standardized distances.
    scaled = nr.knn_readout(np.c_[x*1000+30, np.ones(len(x))],y,groups,seed=42)
    np.testing.assert_allclose(result["predictions"], scaled["predictions"])


@pytest.mark.parametrize("values,label,conflict", [
    ((30,15,20,100,90), "present", False),  # inclusive present boundaries
    ((90,10,20,100,90), "absent", False),
    ((80,10,20,100,90), "partial", False),  # strict absent boundary
    ((90,10,20,100,80), "partial", False),
    ((50,10,20,100,50), "partial", False),
    ((90,50,60,100,90), "present", True),
])
def test_pin_label(values, label, conflict):
    assert nr.pin_label(*values) == {"label":label,"conflict":conflict}


def _cache(root, names, states, groups):
    root.mkdir(parents=True)
    targets = states[:,[7,8,2,3]]
    folds = np.empty(len(groups), int)
    for i, (_, test) in enumerate(grouped_folds(groups,5,42)):
        folds[test] = i
    nr.write_npz(root/"pool.npz", states=states, groups=groups)
    nr.write_json(root/"config.json", {"seed":42,"outer_folds":5})
    metric = {"mean":10.,"ci":[9.,11.]}
    linear = {"ridge":{k:metric for k in nr.TARGETS}, "mean":{k:{"mean":100.,"ci":[90.,110.]} for k in nr.TARGETS}}
    for stem in ["pixel_r45_0", *(f"{n}_r45_{k}" for n in names for k in ("cls","projected"))]:
        nr.write_json(root/f"{stem}.json",linear)
        nr.write_npz(root/f"{stem}_oof.npz", targets=targets, groups=groups, folds=folds)
    nr.write_npz(root/"pixels_r45_0.npz", pixels=np.zeros((len(groups),32,32,3),np.uint8))
    for name in names:
        nr.write_json(root/f"{name}_r45.json",{"with_target":False})
        nr.write_npz(root/f"{name}_r45_features.npz", cls=targets, projected=targets)


def test_runner_resume_pixel_sharing_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT",str(tmp_path))
    a, b = tmp_path/"phaseA", tmp_path/"phaseB"
    states = np.arange(90,dtype=float).reshape(10,9)
    groups = np.repeat(np.arange(5),2)
    _cache(a,["ft_block_s0","ft_block_s1"],states,groups)
    _cache(b,["ft45_block_s0"],states,groups)
    jobs = []
    # Run persistence/report integration without fitting 30 full networks.
    class Immediate:
        def __init__(self, value): self.value = value
        def result(self): return self.value
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, job):
            jobs.append(job)
            return Immediate(fn(job))
    monkeypatch.setattr(nr,"ProcessPoolExecutor",Pool)
    monkeypatch.setattr(nr,"as_completed",lambda futures:iter(futures))
    def readout(kind):
        def run(x,y,g,**kwargs):
            folds = np.empty(len(g),int)
            for i,(_,test) in enumerate(grouped_folds(g,5,42)): folds[test] = i
            return nr._result(kind,y,y,y,g,folds,42)
        return run
    monkeypatch.setattr(nr,"mlp_readout",readout("mlp"))
    monkeypatch.setattr(nr,"knn_readout",readout("knn"))
    root = tmp_path/"runs/pin-mlp"
    options = dict(out_dir=root,checkpoints=("ft_block_s0","ft_block_s1","ft45_block_s0"),
                   radii=(45,),phase_a=a,phase_b=b)
    result = nr.run_pin_mlp(**options)
    assert len(jobs) == 7 and len(result["completed"]) == 7
    assert sum(j["kind"] == "pixel" for j in jobs) == 1
    report = nr.pin_mlp_report(root)
    assert len(report["rows"]) == 6 and len(report["pixels"]) == 1
    assert not report["complete"] and all(r["label"] == "present" for r in report["rows"])
    assert "Pixel-MLP T" in (root/"summary.md").read_text()
    resumed = nr.run_pin_mlp(**options)
    assert resumed["skipped"] == 7 and len(jobs) == 7
    # OOF cache guards run even on a fully cached resume.
    bad = a/"ft_block_s0_r45_cls_oof.npz"
    nr.write_npz(bad, groups=groups, targets=states[:,[7,8,2,3]], folds=np.zeros(len(groups),int))
    with pytest.raises(AssertionError,match="cached ridge folds differ"):
        nr.run_pin_mlp(**options)
    with pytest.raises(ValueError,match="under runs/pin-mlp"):
        nr.run_pin_mlp(out_dir=tmp_path/"runs/size-control")
