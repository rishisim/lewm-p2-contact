from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ATTEMPT_ROOT.parents[3]
if str(ATTEMPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTEMPT_ROOT))

import compile_gate
import counted_features
import flops
import runner
import workflow


def _zero_confirmation_equivalence() -> dict[str, object]:
    return {
        "checks": {
            name: True for name in runner.CONFIRMATION_EQUIVALENCE_CHECK_KEYS
        },
        "cross_batch_max_abs": 0.0,
        "cross_batch_mean_abs": 0.0,
        "cross_batch_rmse": 0.0,
        "passed": True,
    }


def _raw_fixture(
    path: Path, *, slot: int, episode_id: str
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    pixels = np.zeros((201, 224, 224, 3), dtype=np.uint8)
    action = np.zeros((201, 5), dtype=np.float32)
    action[-1] = np.nan
    np.savez_compressed(path, action=action, pixels=pixels)
    loaded, audit = runner.load_model_gate_inputs(
        path, expected_sha256=runner.sha256_file(path)
    )
    record: dict[str, object] = {
        "slot": slot,
        "episode_id": episode_id,
        "raw_path": str(path.resolve()),
        "raw_sha256": runner.sha256_file(path),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": audit["array_sha256"][name],
            }
            for name, value in sorted(loaded.items())
        },
        "input_loader_audit": audit,
    }
    path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    return record, loaded


def _synthetic_replay_context(
    monkeypatch: pytest.MonkeyPatch, *, seed: int
) -> tuple[
    runner.ScientificReplayContext,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(seed)
    history = torch.randn(
        runner.ROWS_PER_EPISODE,
        runner.HISTORY_LEN,
        runner.LATENT_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    actions = torch.randn(
        runner.ROWS_PER_EPISODE,
        runner.HISTORY_LEN,
        runner.ACTION_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    target = torch.randn(
        runner.ROWS_PER_EPISODE,
        runner.LATENT_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    base_prediction = torch.randn(
        runner.ROWS_PER_EPISODE,
        runner.LATENT_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    solver = _SyntheticSolver()
    monkeypatch.setattr(
        runner,
        "prepare_frozen_episode_tensors",
        lambda *_args, **_kwargs: (history, actions, target, base_prediction),
    )
    context = runner.ScientificReplayContext(
        torch=torch,
        runtime=object(),
        model_io=object(),
        device=torch.device("cpu"),
        base=object(),
        contract=object(),
        solver=solver,
        distribution_common=object(),
        module_before={"passed": True},
    )
    return context, history, actions, target, base_prediction


def _load_v5_counted_features():
    path = (
        REPO_ROOT
        / "runs/lewm_v5_readiness_program/v5_package_versions/v004/counted_features.py"
    )
    spec = importlib.util.spec_from_file_location("synthetic_v5_counted_features", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_feature_graph_is_exact_v5_source_carry_forward(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(31)
    history = torch.randn(7, 3, 192, generator=generator, dtype=dtype, requires_grad=True)
    actions = torch.randn(7, 3, 25, generator=generator, dtype=dtype, requires_grad=True)
    current = torch.randn(7, 192, generator=generator, dtype=dtype, requires_grad=True)
    update = torch.randn(7, 192, generator=generator, dtype=dtype, requires_grad=True)
    v5 = _load_v5_counted_features()
    observed, names = counted_features.build_counted_causal_features(
        history, actions, current, update, return_names=True
    )
    expected, expected_names = v5.build_counted_causal_features(
        history, actions, current, update, return_names=True
    )
    assert observed.shape == (7, 1046)
    assert torch.equal(observed, expected)
    assert names == expected_names == counted_features.FROZEN_FEATURE_NAMES
    assert observed.requires_grad is False


def test_gate_information_boundary_has_no_outcome_or_privileged_argument() -> None:
    forbidden = {
        "target",
        "contact",
        "state",
        "motion",
        "phase",
        "reward",
        "success",
        "dgp",
        "regime",
    }
    for function in (
        counted_features.build_counted_causal_features,
        runner.score_gate,
        runner.manual_sparse,
        runner.dense_shadow,
        runner.candidate_runtime_equivalence,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_exact_operation_and_execution_costs() -> None:
    pooled = flops.gate_cost(2)
    envelope = flops.gate_cost(8)
    assert pooled == {
        "head_count": 2,
        "feature_flops": 3801,
        "affine_head_flops_each": 2092,
        "all_head_flops": 4184,
        "total_gate_flops": 7985,
        "nonflop_operations": 5,
    }
    assert envelope == {
        "head_count": 8,
        "feature_flops": 3801,
        "affine_head_flops_each": 2092,
        "all_head_flops": 16736,
        "total_gate_flops": 20537,
        "nonflop_operations": 11,
    }
    calls = np.asarray([1, 2, 3, 4], dtype=np.int8)
    priced = flops.exact_compute(calls, 8)
    assert priced["base_flops"] == 4 * 70_529_190
    assert priced["depth1_flops"] == 4 * 669_184
    assert priced["adapter_flops"] == 6 * 264_960
    assert priced["gate_evaluations"] == 9
    assert priced["gate_total_flops"] == 9 * 20_537
    assert priced["gate_nonflop_operations"] == 9 * 11
    assert flops.derive_exact_costs()["passed"] is True


@pytest.mark.parametrize(
    ("architecture", "head_count"),
    [("balanced_pooled_dual", 2), ("domain_envelope_eight", 8)],
)
def test_compiler_preserves_selected_raw_affines(
    architecture: str, head_count: int
) -> None:
    rng = np.random.default_rng(800 + head_count)
    weights = rng.normal(size=(3, head_count, 1046)).astype(np.float64)
    biases = rng.normal(size=(3, head_count)).astype(np.float64)
    thresholds = rng.normal(size=3).astype(np.float64)
    names = np.asarray([f"head_{index}" for index in range(head_count)])
    arrays, metadata = compile_gate.compile_arrays(
        architecture=architecture,
        candidate_id=f"synthetic_{head_count}",
        weights=weights,
        biases=biases,
        thresholds=thresholds,
        head_names=names,
    )
    assert set(arrays) == compile_gate.COMPILED_KEYS
    assert arrays["weights"].dtype == np.float32
    assert arrays["biases"].dtype == np.float32
    assert arrays["thresholds"].dtype == np.float32
    assert arrays["weights"].shape == (3, head_count, 1046)
    assert np.array_equal(arrays["weights"], weights.astype(np.float32))
    assert np.array_equal(arrays["biases"], biases.astype(np.float32))
    assert metadata["compiler_performed_fitting_or_recentering"] is False
    assert metadata["gate_cost"]["total_gate_flops"] in (7985, 20537)


class _Adapter:
    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.batch_sizes: list[int] = []

    def __call__(
        self, history: torch.Tensor, actions: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        self.batch_sizes.append(len(current))
        action_term = actions.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return 0.01 * torch.tanh(
            self.scale * current + 0.05 * history[:, -1] + action_term
        )


class _SyntheticSolver:
    def __init__(self) -> None:
        self.adapters = [_Adapter(0.2), _Adapter(0.3), _Adapter(0.4)]

    def anchor(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
        base_prediction: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        action_term = actions.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        update = 0.02 * torch.tanh(
            base_prediction + 0.03 * history[:, -1] + action_term
        )
        return {0: base_prediction, 1: base_prediction + update}

    def __call__(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
        base_prediction: torch.Tensor,
        *,
        max_depth: int,
        return_updates: bool,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        assert max_depth == 4 and return_updates
        outputs = self.anchor(history, actions, base_prediction)
        updates = {1: outputs[1] - base_prediction}
        current = outputs[1]
        for stage, adapter in enumerate(self.adapters, start=2):
            update = adapter(history, actions, current)
            current = current + update
            outputs[stage] = current
            updates[stage] = update
        return outputs, updates

    def forward_selected(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
        base_prediction: torch.Tensor,
        calls: torch.Tensor,
    ) -> torch.Tensor:
        current = self.anchor(history, actions, base_prediction)[1]
        for adapter_index, adapter in enumerate(self.adapters):
            active = torch.nonzero(calls > adapter_index + 1, as_tuple=False).flatten()
            if not active.numel():
                break
            preceding = current.index_select(0, active)
            update = adapter(
                history.index_select(0, active),
                actions.index_select(0, active),
                preceding,
            )
            current = current.index_copy(0, active, preceding + update)
        return current


def _synthetic_gate(head_count: int) -> runner.GateTensors:
    architecture = (
        "balanced_pooled_dual" if head_count == 2 else "domain_envelope_eight"
    )
    weights = np.zeros((3, head_count, 1046), dtype=np.float32)
    # First current-prediction component: history 576 + actions 75.
    weights[:, :, 651] = 1.0
    biases = np.broadcast_to(
        np.arange(head_count, dtype=np.float32)[None, :] * 0.1,
        (3, head_count),
    ).copy()
    thresholds = np.asarray([-0.35, 0.0, 0.35], dtype=np.float32)
    return runner.gate_from_tensors(
        torch,
        architecture=architecture,
        candidate_id=f"synthetic_h{head_count}",
        head_names=tuple(f"head_{index}" for index in range(head_count)),
        weights=weights,
        biases=biases,
        thresholds=thresholds,
        device="cpu",
    )


@pytest.mark.parametrize("head_count", [2, 8])
def test_sparse_dense_contract_and_primitive_reconstruction(head_count: int) -> None:
    generator = torch.Generator().manual_seed(99 + head_count)
    batch = 13
    history = torch.randn(batch, 3, 192, generator=generator, dtype=torch.float32)
    actions = torch.randn(batch, 3, 25, generator=generator, dtype=torch.float32)
    base = torch.randn(batch, 192, generator=generator, dtype=torch.float32) * 0.05
    base[:, 0] = torch.linspace(-1.0, 1.0, batch)
    gate = _synthetic_gate(head_count)
    sparse_solver = _SyntheticSolver()
    sparse = runner.manual_sparse(
        torch, sparse_solver, gate, history, actions, base
    )
    repeat = runner.manual_sparse(
        torch, _SyntheticSolver(), gate, history, actions, base
    )
    dense = runner.dense_shadow(
        torch, _SyntheticSolver(), gate, history, actions, base
    )
    assert torch.equal(sparse.calls, dense.calls)
    assert runner.tensors_exact_with_nan(torch, sparse.scores, repeat.scores)
    assert runner.tensors_exact_with_nan(
        torch, sparse.head_scores, repeat.head_scores
    )
    assert torch.equal(
        sparse.selected,
        _SyntheticSolver().forward_selected(history, actions, base, sparse.calls),
    )
    selected_dense = dense.exits[torch.arange(batch), sparse.calls - 1]
    assert torch.allclose(sparse.selected, selected_dense, rtol=2e-6, atol=2e-7)
    assert sparse.features.shape == (batch, 3, 1046)
    assert sparse.head_scores.shape == (batch, 3, head_count)
    assert torch.isnan(sparse.features[~sparse.reached]).all()
    assert torch.isnan(sparse.scores[~sparse.reached]).all()
    reconstructed = runner.reconstruct_features_from_primitives(
        torch,
        history,
        actions,
        sparse.stage_current,
        sparse.stage_update,
        sparse.reached,
    )
    assert runner.tensors_exact_with_nan(torch, sparse.features, reconstructed)
    scores, heads = runner.reconstruct_scores_from_features(
        torch, gate, reconstructed, sparse.reached
    )
    assert runner.tensors_exact_with_nan(torch, scores, sparse.scores)
    assert runner.tensors_exact_with_nan(torch, heads, sparse.head_scores)
    assert torch.equal(sparse.calls, runner.sequential_calls_from_scores(torch, dense.scores, gate.thresholds))
    for index, adapter in enumerate(sparse_solver.adapters):
        expected_reached_adapter = int((sparse.calls > index + 1).sum().item())
        assert adapter.batch_sizes == ([expected_reached_adapter] if expected_reached_adapter else [])


def _synthetic_fitted_family() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2201)
    weights = np.zeros((24, 3, 8, 1046), dtype=np.float32)
    biases = np.zeros((24, 3, 8), dtype=np.float32)
    thresholds = np.zeros((24, 3), dtype=np.float32)
    head_count = np.empty(24, dtype=np.int16)
    architectures = np.empty(24, dtype="<U32")
    names = np.full((24, 8), "padding", dtype="<U32")
    for index in range(24):
        heads = 2 if index % 2 == 0 else 8
        head_count[index] = heads
        architectures[index] = (
            "balanced_pooled_dual" if heads == 2 else "domain_envelope_eight"
        )
        weights[index, :, :heads] = rng.normal(
            scale=1e-3, size=(3, heads, 1046)
        ).astype(np.float32)
        # Keep thresholds far from scores so BLAS low-bit differences cannot
        # manufacture a synthetic boundary mismatch.
        thresholds[index] = np.asarray([-10.0, -10.0, -10.0], dtype=np.float32)
        names[index, :heads] = [f"c{index}_h{head}" for head in range(heads)]
    return {
        "compiled_weights": weights,
        "compiled_biases": biases,
        "thresholds": thresholds,
        "head_count": head_count,
        "candidate_ids": np.asarray([f"candidate_{index:02d}" for index in range(24)]),
        "architectures": architectures,
        "head_names": names,
    }


def test_all_24_candidate_numpy_runtime_calls_are_exact_on_two_roles() -> None:
    fitted = _synthetic_fitted_family()
    for seed, rows in ((710, 19), (711, 23)):
        features = np.random.default_rng(seed).normal(
            size=(rows, 3, 1046)
        ).astype(np.float32)
        qualification = runner.candidate_runtime_equivalence(
            torch, features=features, fitted=fitted, device="cpu"
        )
        assert qualification["candidate_count"] == 24
        assert qualification["all_24_candidates_passed"] is True
        assert all(
            row["numpy_float32_vs_torch_float32_calls_exact"]
            for row in qualification["records"]
        )


def test_frozen_scientific_facade_sources_are_unchanged_without_loading_data() -> None:
    audit = runner.frozen_facade_source_audit()
    assert audit["passed"] is True
    assert len(audit["sources"]) == 3


def test_interruption_complete_aggregate_is_adopted_only_after_exact_content_check(
    tmp_path: Path,
) -> None:
    output = tmp_path / "role.npz"
    arrays = {
        "episode_slot": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "target": np.arange(12, dtype=np.float32).reshape(4, 3),
    }
    first = runner._write_or_adopt_exact_npz(output, arrays)
    assert first["status"] == "new_complete_aggregate_written_and_verified"
    second = runner._write_or_adopt_exact_npz(output, arrays)
    assert second["status"] == "existing_interruption_complete_aggregate_adopted"
    assert second["file_sha256"] == first["file_sha256"]
    assert second["array_sha256"] == first["array_sha256"]

    divergent = dict(arrays)
    divergent["target"] = arrays["target"].copy()
    divergent["target"][0, 0] += np.float32(1.0)
    with pytest.raises(RuntimeError, match="content drift: target"):
        runner._write_or_adopt_exact_npz(output, divergent)


def test_complete_confirmation_part_is_idempotently_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner, "relative_to_repo", lambda path: str(Path(path).resolve())
    )
    monkeypatch.setattr(runner, "resolve_repo_relative", lambda path: Path(path))
    part = tmp_path / "synthetic-confirmation-0000.npz"
    sidecar = part.with_suffix(".json")
    raw_path = tmp_path / "synthetic-raw.npz"
    raw_record, _ = _raw_fixture(
        raw_path, slot=0, episode_id="synthetic-confirmation-0000"
    )
    raw_sidecar = raw_path.with_suffix(".json")
    replay, history, actions, target, base_prediction = _synthetic_replay_context(
        monkeypatch, seed=73
    )
    gate_path = tmp_path / "compiled_gate.npz"
    gate_arrays, _ = compile_gate.compile_arrays(
        architecture="domain_envelope_eight",
        candidate_id="synthetic-candidate",
        weights=np.zeros((3, 8, runner.FEATURE_DIM), dtype=np.float32),
        biases=np.zeros((3, 8), dtype=np.float32),
        thresholds=np.zeros(3, dtype=np.float32),
        head_names=[f"head-{index}" for index in range(8)],
    )
    np.savez_compressed(gate_path, **gate_arrays)
    gate_sha256 = runner.sha256_file(gate_path)
    runtime_gate = runner.load_gate_tensors(torch, torch.device("cpu"), gate_path)
    dense = runner.dense_shadow(
        torch, replay.solver, runtime_gate, history, actions, base_prediction
    )
    sparse = runner.manual_sparse(
        torch, replay.solver, runtime_gate, history, actions, base_prediction
    )
    repeat = runner.manual_sparse(
        torch, replay.solver, runtime_gate, history, actions, base_prediction
    )
    equivalence = runner._confirmation_equivalence(
        torch,
        replay.solver,
        runtime_gate,
        history,
        actions,
        base_prediction,
        sparse,
        repeat,
        dense,
    )
    assert equivalence["passed"] is True
    arrays = {
        "episode_slot": np.zeros(runner.ROWS_PER_EPISODE, dtype=np.int32),
        "model_step": np.arange(3, 41, dtype=np.int16),
        "target": target.numpy().astype(np.float32),
        "exits": dense.exits.numpy().astype(np.float32),
        "selected": sparse.selected.numpy().astype(np.float32),
        "calls": sparse.calls.numpy().astype(np.int8),
        "scores": sparse.scores.numpy().astype(np.float32),
        "head_scores": sparse.head_scores.numpy().astype(np.float32),
        "production_features": sparse.features.numpy().astype(np.float32),
        "history": history.numpy().astype(np.float32),
        "action_history": actions.numpy().astype(np.float32),
        "stage_current": sparse.stage_current.numpy().astype(np.float32),
        "stage_update": sparse.stage_update.numpy().astype(np.float32),
        "reached": sparse.reached.numpy().astype(np.bool_),
        "dense_scores": dense.scores.numpy().astype(np.float32),
        "dense_head_scores": dense.head_scores.numpy().astype(np.float32),
        "dense_stage_current": dense.stage_current.numpy().astype(np.float32),
        "dense_stage_update": dense.stage_update.numpy().astype(np.float32),
    }
    calls = arrays["calls"]
    histogram = np.bincount(calls.astype(np.int64), minlength=5)[1:].tolist()
    runner._write_execution_part(
        part,
        sidecar,
        arrays,
        {
            "schema_version": 1,
            "attempt": "v008",
            "created_unix_ns": 1,
            "role": "confirmation",
            "regime": "native_plan",
            "slot": 0,
            "episode_id": raw_record["episode_id"],
            "raw_path": raw_record["raw_path"],
            "raw_sha256": raw_record["raw_sha256"],
            "raw_sidecar_path": str(raw_sidecar.resolve()),
            "raw_sidecar_sha256": runner.sha256_file(raw_sidecar),
            "input_loader_audit": raw_record["input_loader_audit"],
            "compiled_gate_path": str(gate_path.resolve()),
            "compiled_gate_sha256": gate_sha256,
            "architecture": "domain_envelope_eight",
            "candidate_id": "synthetic-candidate",
            "head_names": [f"head-{index}" for index in range(8)],
            "actual_adaptive_output": "manual sequential sparse reached-row execution",
            "dense_role": "numerical shadow and fixed-depth comparator only",
            "call_histogram": histogram,
            "exact_compute": runner.exact_compute(calls, 8),
            "equivalence": equivalence,
            "target_persisted": True,
            "target_loss_computed": False,
            "contact_or_privileged_materialized": False,
            "no_gradients": True,
        },
    )
    resumed = runner._execution_record_if_valid(
        part,
        sidecar,
        role="confirmation",
        regime="native_plan",
        raw_record=raw_record,
        head_count=8,
        compiled_gate_path=gate_path,
        compiled_gate_sha256=gate_sha256,
        runtime_torch=torch,
        runtime_gate=runtime_gate,
        scientific_replay=replay,
    )
    assert resumed is not None
    assert resumed["episode_id"] == raw_record["episode_id"]
    assert resumed["call_histogram"] == histogram
    assert resumed["sha256"] == runner.sha256_file(part)

    baseline = json.loads(sidecar.read_text(encoding="utf-8"))

    def write_sidecar(value: dict[str, object]) -> None:
        sidecar.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    attacks = []
    extra = copy.deepcopy(baseline)
    extra["unexpected"] = True
    attacks.append(extra)
    privileged = copy.deepcopy(baseline)
    privileged["contact_or_privileged_materialized"] = True
    attacks.append(privileged)
    gradients = copy.deepcopy(baseline)
    gradients["no_gradients"] = False
    attacks.append(gradients)
    fake_calls = copy.deepcopy(baseline)
    fake_calls["call_histogram"] = [0, 38, 0, 0]
    fake_calls["exact_compute"] = runner.exact_compute(
        np.full(38, 2, dtype=np.int8), 8
    )
    attacks.append(fake_calls)
    fake_array_hash = copy.deepcopy(baseline)
    fake_array_hash["arrays"]["selected"]["sha256"] = "c" * 64
    attacks.append(fake_array_hash)
    loader_extra = copy.deepcopy(baseline)
    loader_extra["input_loader_audit"]["unexpected"] = True
    attacks.append(loader_extra)
    loader_type = copy.deepcopy(baseline)
    loader_type["input_loader_audit"] = []
    attacks.append(loader_type)
    loader_fabricated = copy.deepcopy(baseline)
    loader_fabricated["input_loader_audit"]["archive_sha256"] = "f" * 64
    attacks.append(loader_fabricated)
    for attack in attacks:
        write_sidecar(attack)
        with pytest.raises(RuntimeError):
            runner._execution_record_if_valid(
                part,
                sidecar,
                role="confirmation",
                regime="native_plan",
                raw_record=raw_record,
                head_count=8,
                compiled_gate_path=gate_path,
                compiled_gate_sha256=gate_sha256,
                runtime_torch=torch,
                runtime_gate=runtime_gate,
                scientific_replay=replay,
            )
    write_sidecar(baseline)

    original_part_bytes = part.read_bytes()
    for attack_name in (
        "gate_calls",
        "primitive_features",
        "selected",
        "target",
        "dense_exit",
    ):
        attacked_arrays = {name: value.copy() for name, value in arrays.items()}
        attacked_sidecar = copy.deepcopy(baseline)
        if attack_name == "gate_calls":
            attacked_arrays["calls"][:] = 2
            attacked_arrays["reached"][:, 1] = True
            for name in (
                "scores",
                "head_scores",
                "production_features",
                "stage_current",
                "stage_update",
            ):
                attacked_arrays[name][:, 1] = 0.0
            attacked_sidecar["call_histogram"] = [0, 38, 0, 0]
            attacked_sidecar["exact_compute"] = runner.exact_compute(
                attacked_arrays["calls"], 8
            )
        elif attack_name == "primitive_features":
            attacked_arrays["production_features"][:, 0, 0] = 1.0
        elif attack_name == "selected":
            attacked_arrays["selected"][:, 0] = np.float32(1e-4)
        elif attack_name == "target":
            attacked_arrays["target"][:, 0] += np.float32(1.0)
        else:
            attacked_arrays["exits"][:, 3, 0] += np.float32(1.0)
        np.savez_compressed(part, **attacked_arrays)
        attacked_sidecar["sha256"] = runner.sha256_file(part)
        attacked_sidecar["bytes"] = part.stat().st_size
        attacked_sidecar["arrays"] = runner._array_contract(attacked_arrays)
        write_sidecar(attacked_sidecar)
        with pytest.raises(RuntimeError):
            runner._execution_record_if_valid(
                part,
                sidecar,
                role="confirmation",
                regime="native_plan",
                raw_record=raw_record,
                head_count=8,
                compiled_gate_path=gate_path,
                compiled_gate_sha256=gate_sha256,
                runtime_torch=torch,
                runtime_gate=runtime_gate,
                scientific_replay=replay,
            )
        part.write_bytes(original_part_bytes)
        write_sidecar(baseline)


@pytest.mark.parametrize("role", ("fit", "selection"))
def test_existing_development_manifest_resume_is_closed_and_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    monkeypatch.setattr(runner, "ATTEMPT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "relative_to_repo", lambda path: str(Path(path).resolve())
    )
    monkeypatch.setattr(runner, "resolve_repo_relative", lambda path: Path(path))
    raw_path = tmp_path / f"raw-{role}.npz"
    raw_record, _ = _raw_fixture(
        raw_path, slot=0, episode_id=f"synthetic-{role}-0000"
    )
    raw_sidecar = raw_path.with_suffix(".json")
    replay, history, actions, target, base_prediction = _synthetic_replay_context(
        monkeypatch, seed=101 if role == "fit" else 103
    )
    raw_manifest_path = tmp_path / "raw_manifest.json"
    raw_manifest_path.write_text("{}\n", encoding="utf-8")
    raw_manifest = {"episodes": [raw_record]}
    output_directory = tmp_path / f"data/{role}/native_plan/execution"
    output_directory.mkdir(parents=True)
    part = output_directory / f"{raw_record['episode_id']}.npz"
    sidecar = part.with_suffix(".json")
    exits, features, stage_current, stage_update = runner._dense_development_trace(
        torch, replay.solver, history, actions, base_prediction
    )
    contract = runner._expected_part_array_contract(role)
    arrays = {
        "episode_slot": np.zeros(runner.ROWS_PER_EPISODE, dtype=np.int32),
        "model_step": np.arange(3, 41, dtype=np.int16),
        "target": target.numpy().astype(np.float32),
        "exits": exits.numpy().astype(np.float32),
        "production_features": features.numpy().astype(np.float32),
        "history": history.numpy().astype(np.float32),
        "action_history": actions.numpy().astype(np.float32),
        "stage_current": stage_current.numpy().astype(np.float32),
        "stage_update": stage_update.numpy().astype(np.float32),
    }
    assert set(arrays) == set(contract)
    record = runner._write_execution_part(
        part,
        sidecar,
        arrays,
        {
            "schema_version": 1,
            "attempt": "v008",
            "created_unix_ns": 1,
            "role": role,
            "regime": "native_plan",
            "slot": 0,
            "episode_id": raw_record["episode_id"],
            "raw_path": raw_record["raw_path"],
            "raw_sha256": raw_record["raw_sha256"],
            "raw_sidecar_path": str(raw_sidecar.resolve()),
            "raw_sidecar_sha256": runner.sha256_file(raw_sidecar),
            "input_loader_audit": raw_record["input_loader_audit"],
            "evaluation": "four dense exits and all three frozen causal feature stages",
            "target_use": f"permitted only for isolated {role}",
            "contact_or_privileged_materialized": False,
            "primitive_feature_reconstruction_exact": True,
            "no_gradients": True,
        },
    )
    aggregate_path, aggregate_sha256, aggregate_recovery = (
        runner._write_development_aggregate(role, "native_plan", [record])
    )
    source = tmp_path / "source.py"
    source.write_text("# sealed source\n", encoding="utf-8")
    source_bindings = {
        "runner_source": {
            "path": str(source.resolve()),
            "sha256": runner.sha256_file(source),
        }
    }
    monkeypatch.setattr(
        runner,
        "_execution_source_bindings",
        lambda raw, authorization, gate=None: copy.deepcopy(source_bindings),
    )
    authorization = {
        "state": "FIT_COHORTS" if role == "fit" else "SELECTION_COHORTS",
        "active_attempt": "v008",
    }
    module_audit = replay.module_before
    manifest = {
        "schema_version": 1,
        "attempt": "v008",
        "created_unix_ns": 1,
        "role": role,
        "regime": "native_plan",
        "complete": True,
        "episode_count": 1,
        "row_count": 38,
        "episodes": [record],
        "aggregate_role_path": str(aggregate_path.resolve()),
        "aggregate_role_sha256": aggregate_sha256,
        "aggregate_recovery": aggregate_recovery,
        "aggregate_role_keys": [
            "episode_slot",
            "model_step",
            "target",
            "exits",
            "production_features",
        ],
        "raw_manifest_path": str(raw_manifest_path.resolve()),
        "raw_manifest_sha256": runner.sha256_file(raw_manifest_path),
        "source_raw_manifest_sha256": runner.sha256_file(raw_manifest_path),
        "authorization": authorization,
        "source_bindings": source_bindings,
        "loaded_input_keys": ["action", "pixels"],
        "contact_or_privileged_materialized": False,
        "target_role_isolation": role,
        "causal_primitive_contract": {
            "history": [38, runner.HISTORY_LEN, runner.LATENT_DIM],
            "action_history": [38, runner.HISTORY_LEN, runner.ACTION_DIM],
            "stage_current": [38, runner.STAGES, runner.LATENT_DIM],
            "stage_update": [38, runner.STAGES, runner.LATENT_DIM],
            "sufficient_for_independent_feature_reconstruction": True,
        },
        "module_before": module_audit,
        "module_after": module_audit,
        "base_provenance": {},
        "frozen_facade": {"passed": True},
        "no_gradients": True,
    }
    manifest_path = tmp_path / "execution_manifest.json"

    def write_manifest(value: dict[str, object]) -> None:
        manifest_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_manifest(manifest)
    assert runner._validate_existing_execution_manifest(
        role=role,
        regime="native_plan",
        path=manifest_path,
        output_directory=output_directory,
        raw_manifest_path=raw_manifest_path,
        raw_manifest=raw_manifest,
        authorization=authorization,
        scientific_replay=replay,
    ) == manifest

    for field, value in (
        ("contact_or_privileged_materialized", True),
        ("no_gradients", False),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("unexpected", True),
    ):
        attack = copy.deepcopy(manifest)
        attack[field] = value
        write_manifest(attack)
        with pytest.raises(RuntimeError):
            runner._validate_existing_execution_manifest(
                role=role,
                regime="native_plan",
                path=manifest_path,
                output_directory=output_directory,
                raw_manifest_path=raw_manifest_path,
                raw_manifest=raw_manifest,
                authorization=authorization,
                scientific_replay=replay,
            )
    subset = copy.deepcopy(manifest)
    subset["source_bindings"] = {}
    write_manifest(subset)
    with pytest.raises(RuntimeError, match="source bindings are not exact"):
        runner._validate_existing_execution_manifest(
            role=role,
            regime="native_plan",
            path=manifest_path,
            output_directory=output_directory,
            raw_manifest_path=raw_manifest_path,
            raw_manifest=raw_manifest,
            authorization=authorization,
            scientific_replay=replay,
        )

    baseline_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    original_part = part.read_bytes()

    def write_sidecar(value: dict[str, object]) -> None:
        sidecar.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    for attack_name in (
        "feature",
        "primitive",
        "target",
        "exit",
        "gain_member",
        "call_member",
        "drop_member",
        "dtype",
    ):
        attacked = {name: value.copy() for name, value in arrays.items()}
        if attack_name == "feature":
            attacked["production_features"][0, 0, 0] += np.float32(1.0)
        elif attack_name == "primitive":
            attacked["history"][0, 0, 0] += np.float32(1.0)
        elif attack_name == "target":
            attacked["target"][0, 0] += np.float32(1.0)
        elif attack_name == "exit":
            attacked["exits"][0, 3, 0] += np.float32(1.0)
        elif attack_name == "gain_member":
            attacked["gain"] = np.zeros((38, 3), dtype=np.float32)
        elif attack_name == "call_member":
            attacked["calls"] = np.ones(38, dtype=np.int8)
        elif attack_name == "drop_member":
            attacked.pop("stage_update")
        else:
            attacked["target"] = attacked["target"].astype(np.float64)
        np.savez_compressed(part, **attacked)
        attacked_sidecar = copy.deepcopy(baseline_sidecar)
        attacked_sidecar["sha256"] = runner.sha256_file(part)
        attacked_sidecar["bytes"] = part.stat().st_size
        attacked_sidecar["arrays"] = runner._array_contract(attacked)
        write_sidecar(attacked_sidecar)
        with pytest.raises(RuntimeError):
            runner._execution_record_if_valid(
                part,
                sidecar,
                role=role,
                regime="native_plan",
                raw_record=raw_record,
                scientific_replay=replay,
            )
        part.write_bytes(original_part)
        write_sidecar(baseline_sidecar)

    for audit_attack in ("extra", "type", "fabricated"):
        attacked_sidecar = copy.deepcopy(baseline_sidecar)
        if audit_attack == "extra":
            attacked_sidecar["input_loader_audit"]["unexpected"] = True
        elif audit_attack == "type":
            attacked_sidecar["input_loader_audit"] = []
        else:
            attacked_sidecar["input_loader_audit"]["archive_sha256"] = "f" * 64
        write_sidecar(attacked_sidecar)
        with pytest.raises(RuntimeError):
            runner._execution_record_if_valid(
                part,
                sidecar,
                role=role,
                regime="native_plan",
                raw_record=raw_record,
                scientific_replay=replay,
            )
        write_sidecar(baseline_sidecar)

    corrupted_raw_record = copy.deepcopy(raw_record)
    corrupted_raw_record["input_loader_audit"] = {"fabricated": True}
    with pytest.raises(RuntimeError, match="raw record/input-loader audit"):
        runner._execution_record_if_valid(
            part,
            sidecar,
            role=role,
            regime="native_plan",
            raw_record=corrupted_raw_record,
            scientific_replay=replay,
        )


def test_development_binding_producer_matches_workflow_consumer_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "relative_to_repo", lambda path: str(Path(path)))
    monkeypatch.setattr(runner, "resolve_repo_relative", lambda path: Path(path))

    def sealed(name: str) -> dict[str, str]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        return {"path": str(path), "sha256": runner.sha256_file(path)}

    dgp = sealed("dgp")
    ledger = sealed("ledger")
    direct = sealed("direct_authorization")
    raw_authorization = sealed("raw_authorization")
    registry = sealed("replacement_registry")
    raw_manifest = {
        "dgp_matrix_path": dgp["path"],
        "dgp_matrix_sha256": dgp["sha256"],
        "cohort_seed_ledger_path": ledger["path"],
        "cohort_seed_ledger_sha256": ledger["sha256"],
        "authorization_seal": raw_authorization,
        "replacement_registry_path": registry["path"],
    }
    authorization = {
        "seal_path": direct["path"],
        "seal_sha256": direct["sha256"],
        "raw_authorization_seal_path": raw_authorization["path"],
        "raw_authorization_seal_sha256": raw_authorization["sha256"],
    }

    produced = runner._execution_source_bindings(raw_manifest, authorization)
    assert set(produced) == set(workflow.SOURCE_BINDING_KEYS)
    assert produced["raw_authorization_seal"] == raw_authorization


def test_execution_handlers_do_not_capture_process_interruptions() -> None:
    for function in (
        runner._execute_development_locked,
        runner._execute_confirmation_locked,
    ):
        tree = ast.parse(inspect.getsource(function))
        handlers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler)
        ]
        assert len(handlers) == 1
        assert isinstance(handlers[0].type, ast.Name)
        assert handlers[0].type.id == "Exception"
