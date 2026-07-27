import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from task_e import (
    ScheduledPushTRefinedCostModel,
    candidate_hashes,
    pairwise_concordance,
    ranking_metrics,
    selected_change,
    top_set,
    validate_schedule,
)
from task_c_population import call_seed, innovation_stream


def test_identical_candidate_identity_across_depths():
    seed = call_seed(2026072701, 1234, 0)
    candidates = innovation_stream(
        seed=seed, iterations=20, batch=1, horizon=5, action_dim=10
    )[-1, 0].numpy()
    hashes = candidate_hashes(candidates)
    for _depth in (0, 1, 2, 4):
        assert candidate_hashes(candidates.copy()) == hashes


def test_schedule_validation_and_expected_stage_calls():
    assert validate_schedule((0, 1, 2, 4, 0)) == (0, 1, 2, 4, 0)
    with pytest.raises(ValueError):
        validate_schedule((0, 1))
    with pytest.raises(ValueError):
        validate_schedule((0, 1, 2, 3, 4))

    # The boundary's ledger contract is the sum of requested stage prefixes.
    schedule = (0, 1, 2, 4, 0)
    expected = [sum(depth > stage for depth in schedule) for stage in range(4)]
    assert expected == [3, 2, 1, 1]


class DummyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def encode(self, info):
        pixels = info["pixels"]
        return {"emb": torch.zeros(*pixels.shape[:2], 192, device=pixels.device)}

    def action_encoder(self, actions):
        return actions

    def predict(self, history, _actions):
        return torch.zeros_like(history)

    def get_cost(self, _info, candidates):
        return candidates.square().sum(dim=(2, 3))


def test_schedule_executes_requested_stages_and_zero_reference():
    from pusht_refiner import MaskedStagewiseRefiner

    info = {
        "pixels": torch.zeros(1, 2, 1, 1),
        "goal": torch.zeros(1, 2, 1, 1),
        "action": torch.zeros(1, 2, 1, 10),
    }
    candidates = torch.randn(1, 2, 5, 10)
    base = DummyBase()
    zero = ScheduledPushTRefinedCostModel(base, None, (0, 0, 0, 0, 0))
    assert torch.equal(zero.get_cost(info.copy(), candidates), base.get_cost(info, candidates))
    refiner = MaskedStagewiseRefiner(verified_export=True)
    scheduled = ScheduledPushTRefinedCostModel(base, refiner, (0, 1, 2, 4, 0))
    scheduled.get_cost(info.copy(), candidates)
    assert scheduled.last_trace is not None
    assert scheduled.last_trace.stage_calls == [3, 2, 1, 1]


def test_ties_are_not_broken_for_top_k_or_concordance():
    values = np.array([0.0, 1.0, 1.0, 2.0])
    assert np.array_equal(top_set(values, 2, 1e-8), np.array([0, 1, 2]))
    record = pairwise_concordance(values, values, 1e-8, 1e-8)
    assert record["concordance"] == 1.0
    assert record["simulator_ties"] == 1


def test_ranking_metrics_direction_and_selected_change():
    prediction = np.array([3.0, 2.0, 1.0, 0.0])
    simulator = np.array([3.0, 2.0, 1.0, 0.0])
    metrics = ranking_metrics(prediction, simulator, top_ks=(2,))
    assert metrics["top_choice_regret"] == 0
    assert metrics["concordance"] == 1
    assert metrics["spearman"] == 1
    candidates = np.arange(4 * 50, dtype=float).reshape(4, 5, 10)
    changed = selected_change(
        np.array([0, 1, 2, 3.0]),
        np.array([3, 2, 1, 0.0]),
        simulator,
        candidates,
        1e-7,
    )
    assert changed["changed"]
    assert changed["direction"] == "improves"


def test_constant_rank_vector_is_explicitly_missing():
    metrics = ranking_metrics(
        np.ones(40), np.arange(40, dtype=float), top_ks=(10, 30)
    )
    assert metrics["spearman"] is None
    assert metrics["kendall_tau_b"] is None


def test_manifest_split_isolation_and_prior_exclusions():
    manifest = json.loads((ROOT / "task_e_manifest.json").read_text())
    pilot = {row["episode_id"] for row in manifest["pilot"]}
    sealed = {row["episode_id"] for row in manifest["sealed"]}
    exclusions = manifest["exclusions"]
    prior = set(exclusions["task_a_episodes"]) | set(exclusions["task_d_episodes"])
    prior |= {
        episode
        for values in exclusions["task_b_expert_episodes_by_role"].values()
        for episode in values
    }
    assert pilot.isdisjoint(sealed)
    assert (pilot | sealed).isdisjoint(prior)
    assert len(pilot) == 2
    assert len(sealed) == 8


def test_sealed_reset_replay_manifest_is_exact():
    run_manifest = json.loads((ROOT / "task_e_run_manifest.json").read_text())
    assert run_manifest["integrity"] == {
        "all_replays_exact": True,
        "bank_count": 16,
        "candidate_labels": 4800,
        "model_visible_equals_executed": 4800,
        "replay_checks": 192,
    }
    assert all(
        check["actions_exact"] and check["states_exact"] and check["costs_exact"]
        for bank in run_manifest["banks"]
        for check in bank["replay_checks"]
    )
