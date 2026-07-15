import json
from pathlib import Path

P = Path(__file__).resolve().parent

def load(name): return json.loads((P / name).read_text())

def test_frozen_sample_exact():
    p = load("protocol.json")
    assert p["sample"]["episode_ids"] == [3316,5355,6405,2420,6763,8240,9767,8510]
    assert p["sample"]["transition_rows"] == [0,1,2,5,10,25,50,100]
    assert p["sample"]["frame_rows"] == [0,50,200]

def test_no_candidate_and_fail_closed():
    inv, dec, met = load("candidate_inventory.json"), load("decision.json"), load("metrics.json")
    assert inv["eligible_candidates"] == []
    assert dec["outcome"] == "coherent_generator_candidate_not_identified"
    assert not dec["scientific_replay_executed"] and not dec["hdf5_opened"]
    assert met["hdf5_rows_read"] == met["transition_count_executed"] == met["frame_count_executed"] == 0

def test_candidate_failure_reasons():
    cs = load("candidate_inventory.json")["candidates"]
    assert not cs[0]["checks"]["version_coherent"] and not cs[0]["checks"]["required_files_exist"] and not cs[0]["checks"]["collector_contract_matches"]
    assert not cs[1]["checks"]["version_coherent"] and not cs[1]["checks"]["base_requirements_coherent"]
    assert cs[1]["checks"]["required_files_exist"] and cs[1]["checks"]["collector_contract_matches"]

def test_zero_prohibited_work():
    d = load("decision.json")
    assert d["new_policy_trajectories"] == d["model_or_gate_rows"] == d["v5_episodes"] == 0
    assert not d["v3_test_targets_opened"] and not d["combined_v3_cache_numpy_loaded"]
