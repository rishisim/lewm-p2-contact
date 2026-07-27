from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task_c_population import (
    MAX_POPULATION,
    PrefixComparableCEMSolver,
    PopulationConfig,
    call_seed,
    counted_flops_per_call,
    cem_counted_flops_per_call,
    elite_count,
    innovation_stream,
    refiner_stage_counted_flops_per_row,
    validate_population,
)


class QuadraticCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0
        self.rows = 0

    def get_cost(self, info, candidates):
        self.calls += 1
        self.rows += candidates.shape[0] * candidates.shape[1] * candidates.shape[2]
        return candidates.square().sum(dim=(2, 3))


class Plan:
    horizon = 5
    receding_horizon = 5
    action_block = 5


def solver(population: int, seed: int = 11):
    model = QuadraticCost()
    result = PrefixComparableCEMSolver(
        model=model,
        population=PopulationConfig.create(population),
        candidate_seed=seed,
    )
    result.configure(
        action_space=Box(-1, 1, shape=(1, 2), dtype=np.float32),
        n_envs=1,
        config=Plan(),
    )
    return result, model


@pytest.mark.parametrize(
    ("population", "elites", "label"),
    [(64, 8, "p064"), (128, 16, "p128"), (300, 38, "p300")],
)
def test_population_and_elite_boundary(population, elites, label):
    config = PopulationConfig.create(population, elites=elites, label=label)
    assert config.elites == elite_count(population)
    config.assert_solver(num_samples=population, topk=elites)


@pytest.mark.parametrize("value", [True, False, 64.0, 0, -1, 65, 301])
def test_population_rejects_ambiguous_or_unsupported_values(value):
    with pytest.raises((TypeError, ValueError)):
        validate_population(value)


def test_elites_labels_and_solver_consistency_fail_closed():
    with pytest.raises(TypeError):
        PopulationConfig.create(64, elites=True)
    with pytest.raises(ValueError):
        PopulationConfig.create(64, elites=7)
    with pytest.raises(ValueError):
        PopulationConfig.create(64, label="64")
    with pytest.raises(ValueError):
        PopulationConfig.create(64).assert_solver(num_samples=64, topk=16)


def test_maximum_stream_prefixes_repeatability_seed_independence_and_global_rng():
    arguments = dict(iterations=20, batch=1, horizon=5, action_dim=10)
    torch.manual_seed(999)
    first = innovation_stream(seed=17, **arguments)
    torch.randn(1000)
    repeated = innovation_stream(seed=17, **arguments)
    other = innovation_stream(seed=18, **arguments)
    assert first.shape == (20, 1, MAX_POPULATION, 5, 10)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)
    for iteration in range(20):
        assert torch.equal(first[iteration, :, :64], repeated[iteration, :, :128,][:, :64])
        assert torch.equal(first[iteration, :, :128], repeated[iteration, :, :300,][:, :128])
    assert call_seed(1, 2, 3) == call_seed(1, 2, 3)
    assert len({call_seed(1, 2, 3), call_seed(2, 2, 3), call_seed(1, 3, 3), call_seed(1, 2, 4)}) == 4


def test_all_populations_use_prefix_innovations_across_iterations_and_replans():
    info = {"pixels": torch.zeros(1, 1), "goal": torch.zeros(1, 1)}
    solved = {}
    for population in (64, 128, 300):
        instance, _ = solver(population)
        instance.solve(copy.deepcopy(info), start_id=7, replan_index=3)
        solved[population] = instance
    for iteration in range(20):
        assert torch.equal(
            solved[64].last_innovations[iteration],
            solved[128].last_innovations[iteration][:, :64],
        )
        assert torch.equal(
            solved[128].last_innovations[iteration],
            solved[300].last_innovations[iteration][:, :128],
        )
    first_replan = solved[300].last_innovations[0].clone()
    solved[300].solve(copy.deepcopy(info), start_id=7, replan_index=4)
    assert not torch.equal(first_replan, solved[300].last_innovations[0])


@pytest.mark.parametrize("population", [64, 128, 300])
def test_exact_work_ledger_and_deterministic_replay(population):
    info = {"pixels": torch.zeros(1, 1), "goal": torch.zeros(1, 1)}
    first, first_model = solver(population)
    second, second_model = solver(population)
    one = first.solve(copy.deepcopy(info), start_id=2, replan_index=0)
    two = second.solve(copy.deepcopy(info), start_id=2, replan_index=0)
    assert torch.equal(one["actions"], two["actions"])
    assert one["costs"] == two["costs"]
    assert first_model.calls == second_model.calls == 20
    assert first_model.rows == second_model.rows == population * 20 * 5
    ledger = first.ledger
    assert ledger.replans == 1
    assert ledger.candidate_sequences_evaluated == population * 20
    assert ledger.predicted_transition_rows == population * 20 * 5
    assert ledger.sampling_invocations == ledger.update_invocations == 20
    assert ledger.topk_invocations == 20
    assert all(shape == [1, 300, 5, 10] for shape in ledger.innovation_tensor_shapes)


def test_iteration_zero_transformed_candidates_are_prefixes():
    stream = innovation_stream(
        seed=3, iterations=1, batch=1, horizon=5, action_dim=10
    )[0]
    transformed = {}
    for population in (64, 128, 300):
        value = stream[:, :population].clone()
        value[:, 0] = 0
        transformed[population] = value
    assert torch.equal(transformed[64], transformed[128][:, :64])
    assert torch.equal(transformed[128], transformed[300][:, :128])


def test_action_space_contract_and_flop_formula_schema():
    instance, _ = solver(64)
    with pytest.raises(ValueError):
        instance.configure(
            action_space=Box(-2, 2, shape=(1, 2), dtype=np.float32),
            n_envs=1,
            config=Plan(),
        )
    values = cem_counted_flops_per_call(64, 8)
    assert values["cem_arithmetic_total"] > 0
    assert values["terminal_squared_error"] == 64 * 20 * (192 * 3 - 1)
    assert values["topk_comparisons"] == 0
    assert refiner_stage_counted_flops_per_row() == 647_872
    totals = {
        (depth, population): counted_flops_per_call(
            population, elite_count(population), depth
        )
        for depth in (0, 1, 2, 4)
        for population in (64, 128, 300)
    }
    assert len(totals) == 12
    assert all(value["counted_total"] > 0 for value in totals.values())
    assert totals[(0, 128)]["refiner_stage_1"] == 0
    assert totals[(4, 300)]["refiner_stage_4"] > 0
