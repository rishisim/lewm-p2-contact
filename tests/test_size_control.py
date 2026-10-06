"""Paired size-control pool and frozen reading gates."""

from copy import deepcopy
import json

import h5py
import numpy as np
import pytest

from lewm_research.envs.pusht_peg import PushTPeg
from lewm_research.probe import size_control as sc


def test_pool_sampling_and_shared_nonoverlapping_pegs(tmp_path, monkeypatch):
    state = np.array([100,100,250,250,0,0,0,400,400], dtype=float)
    source = np.tile(state, (20,1))
    source[:,4] = np.linspace(0, 2*np.pi, 20)
    episodes = np.repeat(np.arange(10), 2)
    for name in ("a","b"):
        with h5py.File(tmp_path/name, "w") as file:
            file["state"] = source
            file["episode_idx"] = episodes
    monkeypatch.setattr(sc, "dataset_path", lambda name:tmp_path/name)
    heldout = {"b":list(range(5)), "a":list(range(5))}
    states, groups, manifest = sc.build_pool(heldout, 8)
    again, groups2, _ = sc.build_pool(heldout, 8)
    np.testing.assert_array_equal(states, again)
    np.testing.assert_array_equal(groups, groups2)
    assert len(states) == 16 and len(manifest["rows"]) == 16
    rng = np.random.default_rng(42)
    for offset, name in ((0,"a"),(8,"b")):
        selected = np.sort(rng.choice(np.arange(10), 8, replace=False))
        np.testing.assert_array_equal(states[offset:offset+8,:7], source[selected,:7])
        assert groups[offset:offset+8].tolist() == [f"{name}:{ep}" for ep in episodes[selected]]
    assert np.all((states[:,7:9] >= 75) & (states[:,7:9] <= 437))
    original = states.copy()
    for radius in sc.RADII:
        env = PushTPeg(peg_radius=radius, with_target=False)
        try:
            env.reset(seed=0)
            for row in states:
                env._set_state(row)
                assert not env.peg_overlaps(row[7:9])
                np.testing.assert_array_equal(env._get_obs()[7:9], row[7:9])
        finally:
            env.close()
        pixels = sc.render_pool(states, radius, False)
        assert pixels.shape == (16,224,224,3)
        np.testing.assert_array_equal(states, original)


def _metric(mean, ci=None):
    return {"mean":mean, "ci":ci or [mean*.99,mean*1.01]}


def _cell(peg=30, t=20, pixel=25, mean=100, reduction=50, reduction_ci=(40,60)):
    return {"peg":_metric(peg), "T":_metric(t), "pixel_peg":_metric(pixel), "mean_peg":_metric(mean),
            "peg_over_T":_metric(peg/t), "peg_over_pixel":_metric(peg/pixel),
            "peg_over_mean":_metric(peg/mean),
            "peg_reduction_15_to_45":_metric(reduction,list(reduction_ci))}


def _reading_input(cell):
    return {name:{"15":{k:_cell(peg=90) for k in ("cls","projected")},
                  "45":{k:deepcopy(cell) for k in ("cls","projected")}} for name in sc.DEFAULT_CHECKPOINTS[1:]}


@pytest.mark.parametrize("cell,label", [
    (_cell(), "size-explained"),
    (_cell(peg=90,t=100,pixel=100), "size-unexplained"),
    (_cell(peg=80,t=40,pixel=80/1.5), "size-explained"),
    (_cell(peg=41), "unlabeled"),
    (_cell(pixel=19), "unlabeled"),
    (_cell(reduction_ci=(0,60)), "unlabeled"),
    (_cell(reduction=-20,reduction_ci=(-30,-10)), "unlabeled"),
])
def test_ordered_reading(cell, label):
    data = _reading_input(cell)
    data["lewm-pusht"] = {"45":{"cls":_cell(peg=1000)}}
    before = deepcopy(data)
    result = sc.apply_reading(data)
    assert len(result["cells"]) == 8
    assert all(c["label"] == label for c in result["cells"])
    expected = {"size-explained":"size substantially explains the readout deficit",
                "size-unexplained":"size alone does not explain it"}.get(label,"partial/ambiguous")
    assert result["overall"] == expected and result["complete"]
    assert result["radius_15_sanity_passed"] and not result["pool_only"]
    assert data == before


def test_reading_sanity_threshold_flags_and_missing_cells():
    data = _reading_input(_cell())
    cell = data["ft_block_s0"]["45"]["cls"]
    cell["peg_over_mean"]["ci"] = [.7,.9]
    cell["peg_over_T"]["ci"] = [1,3]
    cell["peg_over_pixel"]["ci"] = [1,2]
    cell["peg_reduction_15_to_45"]["ci"] = [-1,60]
    data["ft_block_s0"]["15"]["cls"] = _cell(peg=80)
    result = sc.apply_reading(data)
    assert result["pool_only"] and not result["radius_15_sanity_passed"]
    assert len(result["cells"][0]["flags"]) == 5
    assert result["overall"] == "partial/ambiguous"
    del data["ft_mixed_s1"]
    result = sc.apply_reading(data)
    assert not result["complete"] and result["cells"][-1]["label"] == "unavailable"


def _episodes(values):
    return {f"ep:{i}":{"peg":{"sum":v*n,"n":n},"T":{"sum":10*n,"n":n}}
            for i,(v,n) in enumerate(zip(values,(1,3,2,4,1)))}


def _arm(values):
    return {"features":{k:{"episode_errors":_episodes(values if k!="pixel" else [20]*5),
                           "mean_episode_errors":_episodes([100]*5)} for k in ("cls","projected","pixel")}}


def test_summary_paired_draws_and_ratios():
    small = _arm([40,50,60,70,80])
    large = _arm([30,40,50,60,70])
    result = sc.summarize({"ft_block_s0":{"15":small,"45":large}}, samples=100)
    cell = result["ft_block_s0"]["45"]["cls"]
    assert cell["peg_reduction_15_to_45"]["mean"] == pytest.approx(10)
    np.testing.assert_allclose(cell["peg_reduction_15_to_45"]["ci"], [10,10])
    assert cell["peg_over_mean"]["mean"] == pytest.approx(cell["peg"]["mean"]/100)
    assert cell["peg_over_pixel"]["mean"] == pytest.approx(cell["peg"]["mean"]/20)
    assert cell["peg_over_T"]["mean"] == pytest.approx(cell["peg"]["mean"]/10)
    del small["features"]["cls"]["episode_errors"]["ep:0"]
    with pytest.raises(ValueError, match="identical episode"):
        sc.summarize({"ft_block_s0":{"15":small,"45":large}}, samples=10)


def test_runner_caches_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    states = sc.resample_pegs(np.tile([100,100,250,250,0,0,0,400,400], (10,1)))
    groups = np.array([f"episode:{i}" for i in range(10)])
    manifest = {"rows":[{"dataset":"tiny","row":i} for i in range(10)], "datasets":{}}
    monkeypatch.setattr(sc,"validation_episodes",lambda:({"tiny":list(range(10))},
        {"ft_block_s0":{"dataset":{"name":"tiny","sha256":"hash"}}}))
    monkeypatch.setattr(sc,"fingerprint",lambda path:"hash")
    monkeypatch.setattr(sc,"build_pool",lambda *args:(states,groups,manifest))
    planners, encodes, renders, readouts = [], [], [], []
    class FakePlanner:
        def __init__(self, name, device):
            planners.append(name)
            self.with_target = False
        def close(self): pass
    monkeypatch.setattr(sc,"Planner",FakePlanner)
    def encode(planner,pixels):
        encodes.append(len(pixels))
        return {"cls":states[:,7:9],"projected":states[:,7:9]**2}
    monkeypatch.setattr(sc,"encode_pool",encode)
    render = sc.render_pool
    def render_count(*args):
        renders.append(args[1:])
        return render(*args)
    monkeypatch.setattr(sc,"render_pool",render_count)
    monkeypatch.setattr(sc,"pixel_features",lambda pixels:pixels.mean(axis=(1,2)))
    nested = sc.nested_readout
    def readout_count(*args,**kwargs):
        readouts.append(1)
        return nested(*args,**kwargs)
    monkeypatch.setattr(sc,"nested_readout",readout_count)
    options = {"checkpoints":["ft_block_s0","ft_block_s1"], "radii":[15,45],
               "frames_per_dataset":10, "run_dir":tmp_path/"runs/size-control"}
    result = sc.run_size_control(**options)
    assert len(planners) == 2 and len(encodes) == 4 and len(renders) == 2 and len(readouts) == 10
    assert not result["reading"]["complete"]
    resumed = sc.run_size_control(**options)
    assert result["checkpoints"] == resumed["checkpoints"]
    assert len(planners) == 2 and len(encodes) == 4 and len(readouts) == 10
    root = options["run_dir"]
    assert json.loads((root/"summary.json").read_text())["pool_identity"] == result["pool_identity"]
    assert "15→45 reduction" in (root/"summary.md").read_text()
    with pytest.raises(ValueError,match="resume configuration differs"):
        sc.run_size_control(**{**options,"seed":43})
