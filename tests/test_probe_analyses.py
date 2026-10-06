"""Fast synthetic checks for Step-4 analysis and frozen-bar boundaries."""

import json

import numpy as np
import pytest

from lewm_research.probe.abc import abc_metrics, bank_candidates, bank_exclusions, load_bank
from lewm_research.probe.analysis import contrasts
from lewm_research.probe.readout import grouped_folds, nested_readout, pixel_features
from lewm_research.probe.report import (
    compute_report, coverage_label, gap_label, localization_labels, markdown_report,
)


def test_grouped_nested_splits_are_disjoint():
    groups = np.repeat(np.arange(15), 3)
    held = []
    for train, test in grouped_folds(groups, 5, 6):
        assert not set(groups[train]) & set(groups[test])
        held.extend(test)
        for inner_train, inner_test in grouped_folds(groups[train],3,7):
            assert not set(groups[train[inner_train]]) & set(groups[train[inner_test]])
            assert not set(groups[test]) & set(groups[train[inner_test]])
    assert sorted(held)==list(range(len(groups)))
    assert all(np.array_equal(a,b) for pair1,pair2 in zip(grouped_folds(groups,5,6),grouped_folds(groups,5,6))
               for a,b in zip(pair1,pair2))


def test_readout_recovers_planted_linear_target():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(120,8))
    targets = features@rng.normal(size=(8,4))*20 + 256
    result = nested_readout(features,targets,np.repeat(np.arange(20),6),outer_folds=4,
                            alphas=(.001,.1,10),samples=50)
    assert result["ridge"]["peg"]["mean"]<.1
    assert result["ridge"]["T"]["mean"]<.1
    assert result["mean"]["peg"]["mean"]>20
    assert len(result["episode_errors"])==20


def test_pixel_baseline_works():
    rng = np.random.default_rng(5)
    colors = rng.integers(0,256,(60,3),dtype=np.uint8)
    pixels = np.broadcast_to(colors[:,None,None],(60,40,40,3))
    features = pixel_features(pixels)
    assert features.shape==(60,32*32*3)
    targets = np.column_stack([colors[:,0],colors[:,1],colors[:,1],colors[:,2]]).astype(float)
    result = nested_readout(features,targets,np.repeat(np.arange(15),4),outer_folds=3,
                            alphas=(.001,.1),samples=25)
    assert result["ridge"]["peg"]["mean"]<.01


def test_abc_known_rankings_regret_and_ties():
    result = abc_metrics([4,3,2,1],[1,2,3,4],[0,10,20,30])
    assert result==pytest.approx({"AC":-1.,"BC":1.,"AB":-1.,"A_regret":30.,"B_regret":0.})
    tied = abc_metrics([1,1,2,3],[0,10,20,30],[0,10,20,30])
    assert tied["AC"]==pytest.approx(.9486832980505139)
    assert tied["A_regret"]==0
    assert abc_metrics([1]*4,[0,1,2,3],[0,1,2,3])["AC"] is None


def test_physical_bank_deterministic_and_reused(tmp_path):
    rng = np.random.default_rng(4)
    population = rng.normal(size=(32,5,10)).astype(np.float32)
    population[1] = population[0]
    normalization = {"columns":{"action":{"mean":[0,0],"std":[.2,.3]}}}
    bank = bank_candidates(population,np.arange(32),normalization,6)
    assert bank.shape==(64,25,2)
    assert len(np.unique(bank[:16].reshape(16,-1),axis=0))==16
    assert np.all(bank[16:]>=-1) and np.all(bank[16:]<=1)
    assert np.array_equal(bank,bank_candidates(population,np.arange(32),normalization,6))
    path = tmp_path/"bank.npz"
    np.savez(path,actions=bank)
    first = load_bank(path); second = load_bank(path)
    first["actions"][0]=0
    assert np.array_equal(second["actions"],bank)
    assert np.array_equal(load_bank(path)["actions"],bank)
    assert not bank_exclusions({"success":[True,False],"C":[0,21]})
    assert "C_range_not_above_20" in bank_exclusions({"success":[True,False],"C":[0,20]})
    assert "no_successful_candidate" in bank_exclusions({"success":[False,False],"C":[0,21]})


@pytest.mark.parametrize("difference,ci,reference,complete,label",[
    (.15,[.01,.3],.04,True,"gap present"),(.149,[.01,.3],.04,True,"gap absent"),
    (.2,[0,.3],.04,True,"gap absent"),(.2,[.01,.3],.05,True,"gap absent"),
    (.2,[.01,.3],-.1,True,"gap absent"),(.2,[.01,.3],.01,False,"unavailable"),
])
def test_gap_clauses(difference,ci,reference,complete,label):
    assert gap_label({"difference":difference,"ci":ci},reference,complete)==label


def test_localization_each_clause():
    good = {"BC":.8,"AB":.9,"AC":.8}
    assert localization_labels(30,15,20,good,good)["labels"]==["linear-readout deficit"]
    assert not localization_labels(29.9,15,20,good,good)["labels"]
    mismatch = {"BC":.5,"AB":.9,"AC":.8}
    assert localization_labels(20,20,20,mismatch,good)["labels"]==["observed-state cost mismatch"]
    assert not localization_labels(25,20,20,mismatch,good)["labels"]
    degraded = {"BC":.5,"AB":.7,"AC":.6}
    labels = localization_labels(20,20,20,degraded,good)["labels"]
    assert "prediction-associated degradation" in labels
    assert "prediction-associated degradation" not in localization_labels(20,20,20,{**degraded,"BC":.499},good)["labels"]
    assert len(localization_labels(None,None,None,{}, {})["unavailable_clauses"])==3


@pytest.mark.parametrize("gain,ci,control,block,mixed,label",[
    (.1,[.01,.2],-.04,.3,.15,"sufficient remedy"),
    (.1,[0,.2],-.04,.3,.15,"reduced disparity"),
    (.099,[.01,.2],-.04,.3,.15,"reduced disparity"),
    (.1,[.01,.2],-.05,.3,.15,"reduced disparity"),
    (.1,[.01,.2],0,.3,.16,"reduced disparity"),
    (.1,[.01,.2],0,.3,.3,"none"),
    (.1,[.01,.2],0,-.1,-.2,"none"),
])
def test_coverage_clauses(gain,ci,control,block,mixed,label):
    assert coverage_label({"difference":gain,"ci":ci},control,block,mixed)==label


def _episode(arm,base,condition,success,seed=42):
    return {"arm":arm,"base_id":str(base),"condition":condition,"seed":seed,
            "score":{"success":success,"t_success":success}}


def test_report_feasibility_pairing_and_seed_pooling():
    hard,control = contrasts()["G"]
    episodes,feasibility,reference = [],[],[]
    for base in range(20):
        for condition in (hard,control):
            feasibility.append(_episode("reference",base,condition,base!=19,seed=17))
            reference.append(_episode("reference",base,condition,True,seed=18))
            for s in (0,1):
                episodes.append(_episode(f"ft_block_s{s}",base,condition,condition==control))
                episodes.append(_episode(f"ft_mixed_s{s}",base,condition,True))
    # Unconditional learned success on an F-infeasible base cannot change the bar.
    for row in episodes:
        if row["base_id"]=="19":
            row["score"]["success"]=True
    result = compute_report(episodes,feasibility,reference,samples=50)
    assert len(result["feasibility"]["G"]["included_base_ids"])==19
    arm = result["checkpoints"]["ft_block_s0"]["contrasts"]["G"]
    assert arm["gap_label"]=="gap present"
    assert arm["unconditional"]["learned"]["gap_control_minus_hard"]["difference"]==.95
    pooled = result["checkpoints"]["ft_block:pooled"]["contrasts"]["G"]
    assert pooled["common_feasible"]["learned"]["gap_control_minus_hard"]["bases"]==19
    assert result["coverage"]["ft_block:pooled->ft_mixed:pooled"]["G"]["label"]=="sufficient remedy"
    assert result["feasibility"]["D"]["complete"] is False
    assert arm["localization"]["projected"]["unavailable_clauses"]
    assert all(line.startswith("|") or not line for line in markdown_report(result).splitlines())
    json.dumps(result,allow_nan=False)


def test_missing_reference_does_not_assert_gap_absent():
    hard,control = contrasts()["G"]
    episodes = [_episode("ft_block_s0",0,hard,False),_episode("ft_block_s0",0,control,True)]
    f = [_episode("reference",0,hard,True,17),_episode("reference",0,control,True,17)]
    result = compute_report(episodes,f,[],samples=10)
    assert result["checkpoints"]["ft_block_s0"]["contrasts"]["G"]["gap_label"]=="unavailable"


def test_ridge_path_matches_sklearn_train_only_scaling():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from lewm_research.probe.readout import _ridge_path
    rng = np.random.default_rng(7)
    for rows,columns in ((20,5),(5,20)):
        x = rng.normal(size=(rows,columns)); x[:,0]=1
        y = rng.normal(size=(rows,4)); test = rng.normal(size=(6,columns));test[:,0]=1
        actual = _ridge_path(x,y,test,[.1,10])
        scaler = StandardScaler().fit(x)
        for i,alpha in enumerate((.1,10)):
            expected = Ridge(alpha=alpha).fit(scaler.transform(x),y).predict(scaler.transform(test))
            np.testing.assert_allclose(actual[i],expected,rtol=1e-8,atol=1e-8)


def test_validation_pool_uses_intersection_and_namespaces_episodes(tmp_path,monkeypatch):
    import h5py
    import torch
    from lewm_research.probe import readout
    root = tmp_path/'storage';root.mkdir()
    for name,dataset,val in (("block0","block.h5",[0,1]),("block1","block.h5",[1,2]),
                             ("mixed0","mixed.h5",[0,1]),("mixed1","mixed.h5",[1,2])):
        folder=root/name;folder.mkdir()
        torch.save({"metadata":{"dataset":{"name":dataset},"seed":0,"val_episodes":val,
                    "train_episodes":list(set(range(3))-set(val))}},folder/'trainer_state.pth')
    monkeypatch.setattr(readout,"checkpoint_dir",lambda name:root/name)
    held,_ = readout.validation_episodes(["block0","block1","mixed0","mixed1"])
    assert held=={"block.h5":[1],"mixed.h5":[1]}
    for name in held:
        with h5py.File(root/name,'w') as f:
            f['episode_idx']=np.repeat(np.arange(3),5)
            f['state']=np.zeros((15,9))
    monkeypatch.setattr(readout,"dataset_path",lambda name:root/name)
    monkeypatch.setattr(readout,"load_bases",lambda path:[])
    states,groups,manifest=readout.build_pool('unused',20,9,held)
    assert len(states)==10
    assert set(groups)=={"block.h5:1","mixed.h5:1"}
    assert all(r['row'] in range(5,10) for r in manifest['rows'])


def test_run_abc_reuses_banks_across_checkpoints_and_resume(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    from lewm_research.probe import abc
    monkeypatch.setenv('LEWM_WORK_ROOT',str(tmp_path))
    storage = tmp_path/'checkpoints';storage.mkdir()
    for name in ('lewm-pusht','first','second'):
        folder=storage/name;folder.mkdir()
        (folder/'weights.pt').write_bytes(name.encode())
        (folder/'normalization.json').write_text('{}')
    monkeypatch.setattr(abc,'checkpoint_dir',lambda name:storage/name)
    bases=tmp_path/'bases.json';bases.write_text('[]')
    base=SimpleNamespace(id='base')
    condition=SimpleNamespace(name=contrasts()['G'][0])
    monkeypatch.setattr(abc,'load_bases',lambda path:[base])
    monkeypatch.setattr(abc,'make_conditions',lambda *args,**kwargs:{condition.name:condition})
    normalization={"columns":{"action":{"mean":[0,0],"std":[1,1]}}}
    population=torch.arange(32*50,dtype=torch.float32).reshape(1,32,5,10)/10000
    planner_calls=[]
    class FakePlanner:
        def __init__(self,arm,**kwargs):
            self.arm=arm;self.stats=normalization;self.with_target=False
            self.recorder=SimpleNamespace(population={'candidates':population,'costs':torch.arange(32)[None]})
        def close(self): pass
    monkeypatch.setattr(abc,'Planner',FakePlanner)
    def run_episode(*args,**kwargs):
        planner_calls.append(1)
        return {'planning_times_s':[1.]}
    monkeypatch.setattr(abc,'run_episode',run_episode)
    def execute(actions,condition):
        return {'actions':actions,'trajectories':np.zeros((64,26,9)),
                'C':np.arange(64)*10,'C_end':np.arange(64)*10,'success':np.arange(64)%2==0}
    monkeypatch.setattr(abc,'execute_bank',execute)
    seen=[]
    def costs(planner,bank,*args):
        seen.append((planner.arm,bank['actions'].copy()))
        return np.arange(64),np.arange(64)
    monkeypatch.setattr(abc,'model_costs',costs)
    options=dict(checkpoints=['first','second'],conditions=condition.name,n=1,run_dir=tmp_path/'runs/abc',
                 population=32,topk=4,iterations=1,samples=20)
    first=abc.run_abc(bases,**options)
    assert len(planner_calls)==1 and len(seen)==2
    assert np.array_equal(seen[0][1],seen[1][1])
    assert first['records'][0]['bank_sha256']==first['records'][1]['bank_sha256']
    second=abc.run_abc(bases,**options)
    assert len(planner_calls)==1 and len(seen)==2
    assert first['records']==second['records']
    with pytest.raises(ValueError,match='resume configuration differs'):
        abc.run_abc(bases,**{**options,'seed':999})


def test_abc_normalizes_physical_actions_and_encodes_real_history():
    from types import SimpleNamespace
    import torch
    from lewm_research.probe.abc import model_costs
    seen=[]
    class Model:
        def get_cost(self,info,action):
            assert info['pixels'].shape[2]==1  # wheel planning history_len=1
            assert info['action'].shape[2:]==(1,10)
            seen.append(action.cpu().numpy())
            info['goal_emb']=torch.zeros(1,1,2)
            return action[...,0].sum(-1).square()
        def encode(self,info):
            assert info['pixels'].shape[1]==3  # actual frames 15,20,25
            count=len(info['pixels'])
            emb=torch.zeros(count,3,2);emb[:,-1]=1
            return {'emb':emb}
        def criterion(self,info):
            return (info['predicted_emb'][...,-1,:]-info['goal_emb'][...,-1,:]).square().sum(-1)
    planner=SimpleNamespace(device=torch.device('cpu'),with_target=False,model=Model(),
        stats={'columns':{'action':{'mean':[.1,-.1],'std':[.2,.4]}}},transform=lambda x:x.float()/255)
    state=np.array([50,50,250,250,0,0,0,400,400])
    bank={'trajectories':np.broadcast_to(state,(4,26,9)).copy(),'actions':np.zeros((4,25,2),np.float32)}
    a,b=model_costs(planner,bank,SimpleNamespace(goal_state=state),batch_size=2)
    assert np.all(a==6.25) and np.all(b==2)
    assert len(seen)==2
    np.testing.assert_allclose(seen[0][...,::2],-.5)
    np.testing.assert_allclose(seen[0][...,1::2],.25)


def test_report_io_global_seeds_and_conflicting_protocol(tmp_path,monkeypatch):
    from lewm_research.probe.analysis import write_json
    from lewm_research.probe.report import run_report
    monkeypatch.setenv('LEWM_WORK_ROOT',str(tmp_path))
    hard,control=contrasts()['G']
    roots=[]
    for name,arm,seed,success in [('F','reference',17,True),('E','reference',18,True),('learned','ft_block_s0',18,False)]:
        root=tmp_path/'runs'/name;root.mkdir(parents=True);roots.append(root)
        config={'arm':arm,'seed':seed,'bases_sha256':'same','normalization_sha256':'same','budget':50,
                'displacement_range':[40,100],'population':300,'iterations':30,'topk':30,'approach_weight':.1}
        write_json(root/'config.json',config)
        rows=[_episode(arm,b,c,success or c==control,seed=1000+b) for b in range(5) for c in (hard,control)]
        (root/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=run_report(roots[1:],run_dir=tmp_path/'runs/report',feasibility_run=roots[0],samples=20)
    assert result['reference_E_seeds']==[18]
    arm=result['checkpoints']['ft_block_s0']
    assert arm['evaluation_seeds']==[18] and arm['training_seeds']==[0]
    assert arm['contrasts']['G']['gap_label']=='gap present'
    assert (tmp_path/'runs/report/RESULTS.md').is_file()
    assert (tmp_path/'runs/report/results.json').is_file()
    config=json.loads((roots[-1]/'config.json').read_text());config['budget']=100
    write_json(roots[-1]/'config.json',config)
    with pytest.raises(ValueError,match='evaluation runs disagree'):
        run_report(roots[1:],feasibility_run=roots[0],samples=20)


def test_report_missing_seed_on_feasible_base_prevents_pooled_claim():
    hard,control=contrasts()['G']
    rows=[_episode(f'ft_block_s{s}',b,c,c==control) for s in (0,1) for b in range(5) for c in (hard,control)
          if not (s==1 and b==4)]
    f=[_episode('reference',b,c,True,17) for b in range(5) for c in (hard,control)]
    e=[_episode('reference',b,c,True,18) for b in range(5) for c in (hard,control)]
    result=compute_report(rows,f,e,samples=20)
    assert result['checkpoints']['ft_block_s0']['contrasts']['G']['gap_label']=='gap present'
    assert result['checkpoints']['ft_block:pooled']['contrasts']['G']['gap_label']=='unavailable'
