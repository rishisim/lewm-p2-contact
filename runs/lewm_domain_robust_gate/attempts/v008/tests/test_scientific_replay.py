"""Synthetic qualification for the separately written scientific replay."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


replay = _load("scientific_replay_synthetic", ROOT / "scientific_replay.py")
workflow = _load("scientific_replay_workflow_synthetic", ROOT / "workflow.py")


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


class _FakeModule:
    def parameters(self) -> list[Any]:
        return []


class _FakeRuntime:
    def __init__(self, module_audit: Mapping[str, Any]) -> None:
        self.module_audit_value = dict(module_audit)

    def module_audit(self, _base: Any, _solver: Any, _v1: Any) -> dict[str, Any]:
        return dict(self.module_audit_value)


def _fake_stack() -> Any:
    module_audit = {"passed": True, "fixture_module_sha256": "1" * 64}
    return replay.Stack(
        torch=torch,
        runtime=_FakeRuntime(module_audit),
        model_io=None,
        common=None,
        device=torch.device("cpu"),
        base=_FakeModule(),
        contract=None,
        solver=_FakeModule(),
        v1=object(),
        provenance={"fixture": "frozen"},
        module_before=module_audit,
    )


def _expected_arrays(role: str, slot: int, head_count: int = 8) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "episode_slot": np.full(replay.ROWS, slot, dtype=np.int32),
        "model_step": np.arange(3, 41, dtype=np.int16),
        "exits": np.zeros(
            (replay.ROWS, replay.DEPTHS, replay.LATENT_DIM), dtype=np.float32
        ),
        "history": np.zeros(
            (replay.ROWS, replay.HISTORY_LEN, replay.LATENT_DIM), dtype=np.float32
        ),
        "action_history": np.zeros(
            (replay.ROWS, replay.HISTORY_LEN, replay.ACTION_DIM), dtype=np.float32
        ),
    }
    if role in ("fit", "selection"):
        arrays.update(
            {
                "target": np.zeros(
                    (replay.ROWS, replay.LATENT_DIM), dtype=np.float32
                ),
                "production_features": np.zeros(
                    (replay.ROWS, replay.STAGES, replay.FEATURE_DIM),
                    dtype=np.float32,
                ),
                "stage_current": np.zeros(
                    (replay.ROWS, replay.STAGES, replay.LATENT_DIM),
                    dtype=np.float32,
                ),
                "stage_update": np.zeros(
                    (replay.ROWS, replay.STAGES, replay.LATENT_DIM),
                    dtype=np.float32,
                ),
            }
        )
        return arrays
    reached = np.zeros((replay.ROWS, replay.STAGES), dtype=np.bool_)
    reached[:, 0] = True
    sparse_scores = np.full(
        (replay.ROWS, replay.STAGES), np.nan, dtype=np.float32
    )
    sparse_scores[:, 0] = 0.0
    sparse_heads = np.full(
        (replay.ROWS, replay.STAGES, head_count), np.nan, dtype=np.float32
    )
    sparse_heads[:, 0] = 0.0
    sparse_features = np.full(
        (replay.ROWS, replay.STAGES, replay.FEATURE_DIM),
        np.nan,
        dtype=np.float32,
    )
    sparse_features[:, 0] = 0.0
    sparse_primitives = np.full(
        (replay.ROWS, replay.STAGES, replay.LATENT_DIM),
        np.nan,
        dtype=np.float32,
    )
    sparse_primitives[:, 0] = 0.0
    arrays.update(
        {
            "selected": np.zeros(
                (replay.ROWS, replay.LATENT_DIM), dtype=np.float32
            ),
            "calls": np.ones(replay.ROWS, dtype=np.int8),
            "scores": sparse_scores,
            "head_scores": sparse_heads,
            "production_features": sparse_features,
            "stage_current": sparse_primitives.copy(),
            "stage_update": sparse_primitives.copy(),
            "reached": reached,
            "dense_scores": np.zeros(
                (replay.ROWS, replay.STAGES), dtype=np.float32
            ),
            "dense_head_scores": np.zeros(
                (replay.ROWS, replay.STAGES, head_count), dtype=np.float32
            ),
            "dense_stage_current": np.zeros(
                (replay.ROWS, replay.STAGES, replay.LATENT_DIM),
                dtype=np.float32,
            ),
            "dense_stage_update": np.zeros(
                (replay.ROWS, replay.STAGES, replay.LATENT_DIM),
                dtype=np.float32,
            ),
        }
    )
    if role == "confirmation":
        arrays["target"] = np.zeros(
            (replay.ROWS, replay.LATENT_DIM), dtype=np.float32
        )
    return arrays


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": replay.array_sha256(value),
        }
        for name, value in arrays.items()
    }


def test_replay_frozen_facade_provenance_path_map_exception_is_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "contact-checkout" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    exact = {str(source.resolve()): _sha(source)}
    monkeypatch.setattr(replay, "FROZEN_FACADE_SOURCE_HASHES", exact)

    replay._assert_no_forbidden_keys(
        {"frozen_facade": {"sources": exact}}, "execution_manifest"
    )
    for drift in (
        {str(source.resolve()): "0" * 64},
        {**exact, str(source.with_name("extra.py").resolve()): "1" * 64},
        {str(source.parent) + "/./source.py": _sha(source)},
    ):
        with pytest.raises(replay.ReplayError, match="provenance map drift"):
            replay._assert_no_forbidden_keys(
                {"frozen_facade": {"sources": drift}}, "execution_manifest"
            )
    with pytest.raises(replay.ReplayError, match="forbidden key materialized"):
        replay._assert_no_forbidden_keys(
            {"contact_path": str(source)}, "execution_manifest"
        )
    with pytest.raises(replay.ReplayError, match="forbidden key materialized"):
        replay._assert_no_forbidden_keys({"sources": exact}, "execution_manifest")


def _make_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> dict[str, Any]:
    repo = tmp_path
    attempt = repo / "runs/lewm_domain_robust_gate/attempts/v008"
    root = attempt / "data" / role / "native_plan"
    raw_directory = root / "raw"
    execution_directory = root / "execution"
    raw_directory.mkdir(parents=True)
    execution_directory.mkdir(parents=True)
    monkeypatch.setattr(replay, "REPO_ROOT", repo)
    monkeypatch.setattr(replay, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(replay, "INHERITED_FIT_ROOT", attempt)
    monkeypatch.setattr(replay, "INHERITED_FIT_ATTEMPT", "v008")
    monkeypatch.setattr(replay, "_source_audit", lambda: {"fixture.py": "f" * 64})
    monkeypatch.setattr(
        replay, "DEVELOPMENT_SOURCE_BINDING_KEYS", frozenset({"fixture_source"})
    )
    monkeypatch.setattr(
        replay, "ROBUST_SOURCE_BINDING_KEYS", frozenset({"fixture_source"})
    )
    monkeypatch.setattr(
        replay,
        "DEVELOPMENT_AUTHORIZATION_KEYS",
        frozenset(
            {
                "state_sha256",
                "active_attempt",
                "seal_sha256",
                "exact_locked_snapshot_unchanged",
            }
        ),
    )
    monkeypatch.setattr(
        replay,
        "ROBUST_AUTHORIZATION_KEYS",
        frozenset({"state_sha256", "active_attempt", "seal_sha256"}),
    )

    episode_id = f"fixture-{role}-000"
    raw_path = raw_directory / f"{episode_id}.npz"
    action = np.zeros(replay.ACTION_SHAPE, dtype=np.float32)
    action[200] = np.nan
    pixels = np.zeros(replay.PIXELS_SHAPE, dtype=np.uint8)
    np.savez_compressed(raw_path, action=action, pixels=pixels)
    loaded, loader_audit = replay._load_raw_inputs(raw_path, _sha(raw_path))
    raw_arrays = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": loader_audit["array_sha256"][name],
        }
        for name, value in sorted(loaded.items())
    }
    authorization = {
        "path": "fixture/pre_data_seal.json",
        "sha256": "a" * 64,
        "checkpoint_state": "PRE_OUTCOME_SEAL",
    }
    raw_record = {
        "schema_version": 1,
        "attempt": "v008",
        "created_unix_ns": 1,
        "complete": True,
        "role": role,
        "regime": "native_plan",
        "slot": 0,
        "episode_id": episode_id,
        "seed_source_episode_id": episode_id,
        "replacement_used": False,
        "replacement_claim_index": None,
        "replacement_claim_sha256": None,
        "env_seed": 11,
        "policy_seed": 12,
        "oracle_np_seed": 13,
        "action_space_seed": 14,
        "raw_path": _relative(repo, raw_path),
        "raw_sha256": _sha(raw_path),
        "raw_bytes": raw_path.stat().st_size,
        "arrays": raw_arrays,
        "initial_pixels_sha256": replay.array_sha256(pixels[0]),
        "generation_audit": {},
        "input_loader_audit": loader_audit,
        "persistence_intent_path": "fixture/intent.json",
        "persistence_intent_sha256": "b" * 64,
        "authorization_seal": authorization,
        "raw_archive_members": ["action", "pixels"],
        "orphan_adoption_safe": True,
    }
    raw_sidecar_path = raw_path.with_suffix(".json")
    _json(raw_sidecar_path, raw_record)
    raw_manifest = {
        "schema_version": 1,
        "attempt": "v008",
        "created_unix_ns": 2,
        "complete": True,
        "role": role,
        "regime": "native_plan",
        "regime_specification": {},
        "episode_count": 1,
        "episodes": [raw_record],
        "replacements_used": 0,
        "raw_archive_members": ["action", "pixels"],
        "raw_pixels_contract": "uint8 [201,224,224,3]",
        "raw_action_contract": "float32 [201,5]; 200 finite plus terminal NaN",
        "role_isolation": True,
        "smoke_permanently_excluded": role == "smoke",
        "retention_contract": "pixels_action_shapes_finiteness_and_local_step_count_only",
        "dgp_matrix_path": "fixture/DGP_MATRIX.json",
        "dgp_matrix_sha256": "c" * 64,
        "cohort_seed_ledger_path": "fixture/cohort_seed_ledger.json",
        "cohort_seed_ledger_sha256": "d" * 64,
        "authorization_seal": authorization,
        "replacement_registry_path": "fixture/replacement_registry.json",
        "replacement_registry_scope": "append_only_all_roles_and_all_dgps",
        "orphan_policy": "fixture exact orphan policy",
    }
    raw_manifest_path = root / "raw_manifest.json"
    _json(raw_manifest_path, raw_manifest)

    stack = _fake_stack()
    gate_path: Path | None = None
    head_names = [f"head_{index}" for index in range(8)]
    if role in ("smoke", "confirmation"):
        gate_path = attempt / "freeze/compiled_gate.npz"
        gate_path.parent.mkdir(parents=True)
        np.savez_compressed(
            gate_path,
            schema_version=np.asarray(1, dtype=np.int16),
            architecture=np.asarray("domain_envelope_eight"),
            candidate_id=np.asarray("fixture_candidate"),
            weights=np.zeros((3, 8, replay.FEATURE_DIM), dtype=np.float32),
            biases=np.zeros((3, 8), dtype=np.float32),
            thresholds=np.ones(3, dtype=np.float32),
            head_names=np.asarray(head_names),
        )

    expected = _expected_arrays(role, 0)
    part_path = execution_directory / f"{episode_id}.npz"
    np.savez_compressed(part_path, **expected)
    sidecar = {
        "schema_version": 1,
        "attempt": "v008",
        "created_unix_ns": 3,
        "role": role,
        "regime": "native_plan",
        "slot": 0,
        "episode_id": episode_id,
        "raw_path": _relative(repo, raw_path),
        "raw_sha256": _sha(raw_path),
        "raw_sidecar_path": _relative(repo, raw_sidecar_path),
        "raw_sidecar_sha256": _sha(raw_sidecar_path),
        "input_loader_audit": loader_audit,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "complete": True,
        "path": _relative(repo, part_path),
        "sha256": _sha(part_path),
        "bytes": part_path.stat().st_size,
        "arrays": _array_metadata(expected),
    }
    if role in ("fit", "selection"):
        sidecar.update(
            {
                "evaluation": "four dense exits and all three frozen causal feature stages",
                "target_use": f"permitted only for isolated {role}",
                "primitive_feature_reconstruction_exact": True,
            }
        )
    else:
        assert gate_path is not None
        compute = replay._exact_compute(expected["calls"], 8)
        sidecar.update(
            {
                "compiled_gate_path": _relative(repo, gate_path),
                "compiled_gate_sha256": _sha(gate_path),
                "architecture": "domain_envelope_eight",
                "candidate_id": "fixture_candidate",
                "head_names": head_names,
                "actual_adaptive_output": "manual sequential sparse reached-row execution",
                "dense_role": "numerical shadow and fixed-depth comparator only",
                "call_histogram": [replay.ROWS, 0, 0, 0],
                "exact_compute": compute,
                "equivalence": {"passed": True},
                "target_persisted": role == "confirmation",
                "target_loss_computed": False,
            }
        )
    sidecar_path = execution_directory / f"{episode_id}.json"
    _json(sidecar_path, sidecar)
    execution_record = {
        "slot": 0,
        "episode_id": episode_id,
        "path": _relative(repo, part_path),
        "sha256": _sha(part_path),
        "sidecar_path": _relative(repo, sidecar_path),
        "sidecar_sha256": _sha(sidecar_path),
        "source_raw_path": _relative(repo, raw_path),
        "source_raw_sha256": _sha(raw_path),
        "source_raw_sidecar_path": _relative(repo, raw_sidecar_path),
        "source_raw_sidecar_sha256": _sha(raw_sidecar_path),
        "call_histogram": None
        if role in ("fit", "selection")
        else [replay.ROWS, 0, 0, 0],
    }
    source_path = attempt / "fixture_source.txt"
    source_path.write_text("fixture\n", encoding="utf-8")
    frozen_facade_sources = {str(source_path.resolve()): _sha(source_path)}
    monkeypatch.setattr(
        replay, "FROZEN_FACADE_SOURCE_HASHES", frozen_facade_sources
    )
    module_audit = dict(stack.module_before)
    execution_manifest = {
        "schema_version": 1,
        "attempt": "v008",
        "created_unix_ns": 4,
        "role": role,
        "regime": "native_plan",
        "complete": True,
        "episode_count": 1,
        "row_count": replay.ROWS,
        "episodes": [execution_record],
        "raw_manifest_path": _relative(repo, raw_manifest_path),
        "raw_manifest_sha256": _sha(raw_manifest_path),
        "source_raw_manifest_sha256": _sha(raw_manifest_path),
        "authorization": {
            "state_sha256": "e" * 64,
            "active_attempt": "v008",
            "seal_sha256": "9" * 64,
            **(
                {"exact_locked_snapshot_unchanged": True}
                if role in ("fit", "selection")
                else {}
            ),
        },
        "source_bindings": {
            "fixture_source": {
                "path": _relative(repo, source_path),
                "sha256": _sha(source_path),
            }
        },
        "loaded_input_keys": ["action", "pixels"],
        "contact_or_privileged_materialized": False,
        "causal_primitive_contract": {},
        "module_before": module_audit,
        "module_after": module_audit,
        "base_provenance": {"fixture": "frozen"},
        "frozen_facade": {
            "sources": frozen_facade_sources,
            "passed": True,
        },
        "no_gradients": True,
    }
    aggregate_path: Path | None = None
    if role in ("fit", "selection"):
        aggregate_path = root / "role.npz"
        np.savez_compressed(
            aggregate_path, **{name: expected[name] for name in replay.AGGREGATE_KEYS}
        )
        execution_manifest.update(
            {
                "aggregate_role_path": _relative(repo, aggregate_path),
                "aggregate_role_sha256": _sha(aggregate_path),
                "aggregate_recovery": {},
                "aggregate_role_keys": list(replay.AGGREGATE_KEYS),
                "target_role_isolation": role,
            }
        )
    else:
        assert gate_path is not None
        execution_manifest.update(
            {
                "compiled_gate_path": _relative(repo, gate_path),
                "compiled_gate_sha256": _sha(gate_path),
                "architecture": "domain_envelope_eight",
                "candidate_id": "fixture_candidate",
                "head_names": head_names,
                "head_count": 8,
                "gate_cost_per_reached_decision": {},
                "exact_compute": replay._exact_compute(expected["calls"], 8),
                "actual_adaptive_output": "manual sequential sparse reached-row execution",
                "dense_role": "numerical shadow and fixed-depth comparator only",
                "strict_continue_operator": ">",
                "target_persisted": role == "confirmation",
                "target_loss_computed_during_execution": False,
                "numerical_contract": {},
                "all_equivalence_checks_passed": True,
            }
        )
    execution_manifest_path = root / "execution_manifest.json"
    _json(execution_manifest_path, execution_manifest)
    return {
        "repo": repo,
        "attempt": attempt,
        "root": root,
        "role": role,
        "stack": stack,
        "expected": expected,
        "part": part_path,
        "sidecar": sidecar_path,
        "raw_manifest": raw_manifest_path,
        "execution_manifest": execution_manifest_path,
        "aggregate": aggregate_path,
    }


def _builder(fixture: Mapping[str, Any]) -> Any:
    expected = fixture["expected"]

    def build(
        role: str,
        slot: int,
        _loaded: Mapping[str, np.ndarray],
        _stack: Any,
        _gate: Any,
    ) -> dict[str, np.ndarray]:
        assert role == fixture["role"] and slot == 0
        return {name: value.copy() for name, value in expected.items()}

    return build


@pytest.mark.parametrize("role", ("fit", "selection", "smoke", "confirmation"))
def test_legitimate_all_role_replays_are_exact_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch, role)
    result = replay.qualify_regime(
        role,
        "native_plan",
        stack_loader=lambda: fixture["stack"],
        expected_builder=_builder(fixture),
        attempt_root=fixture["attempt"],
    )
    assert result["passed"] is True
    assert result["every_persisted_tensor_exact"] is True
    assert result["input_loader_agreement_exact"] is True
    assert result["aggregate_exact"] is (True if role in ("fit", "selection") else None)
    assert result["target_loss_computed"] is False
    assert result["loss_or_effect_used_for_acceptance"] is False


def _coherently_refresh_execution(fixture: Mapping[str, Any], name: str) -> None:
    part_path = fixture["part"]
    with np.load(part_path, allow_pickle=False) as stored:
        arrays = {key: np.asarray(stored[key]).copy() for key in stored.files}
    arrays[name].flat[0] = np.float32(7.0)
    np.savez_compressed(part_path, **arrays)
    sidecar = json.loads(fixture["sidecar"].read_text(encoding="utf-8"))
    sidecar["sha256"] = _sha(part_path)
    sidecar["bytes"] = part_path.stat().st_size
    sidecar["arrays"] = _array_metadata(arrays)
    _json(fixture["sidecar"], sidecar)
    execution = json.loads(
        fixture["execution_manifest"].read_text(encoding="utf-8")
    )
    execution["episodes"][0]["sha256"] = _sha(part_path)
    execution["episodes"][0]["sidecar_sha256"] = _sha(fixture["sidecar"])
    aggregate = fixture["aggregate"]
    if aggregate is not None:
        np.savez_compressed(
            aggregate, **{key: arrays[key] for key in replay.AGGREGATE_KEYS}
        )
        execution["aggregate_role_sha256"] = _sha(aggregate)
    _json(fixture["execution_manifest"], execution)


@pytest.mark.parametrize(
    ("role", "member"), (("fit", "target"), ("confirmation", "exits"))
)
def test_coherent_target_exit_and_aggregate_refresh_cannot_pass_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    member: str,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch, role)
    _coherently_refresh_execution(fixture, member)
    with pytest.raises(replay.ReplayError, match="persisted tensor differs"):
        replay.qualify_regime(
            role,
            "native_plan",
            stack_loader=lambda: fixture["stack"],
            expected_builder=_builder(fixture),
            attempt_root=fixture["attempt"],
        )


@pytest.mark.parametrize("alias", (True, 1.0))
def test_json_schema_version_rejects_bool_and_float_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: Any
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch, "fit")
    raw = json.loads(fixture["raw_manifest"].read_text(encoding="utf-8"))
    raw["schema_version"] = alias
    _json(fixture["raw_manifest"], raw)
    with pytest.raises(replay.ReplayError, match="exact integer"):
        replay.qualify_regime(
            "fit",
            "native_plan",
            stack_loader=lambda: fixture["stack"],
            expected_builder=_builder(fixture),
            attempt_root=fixture["attempt"],
        )


@pytest.mark.parametrize("alias", (True, 201.0))
def test_input_loader_shape_alias_or_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: Any
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch, "selection")
    sidecar = json.loads(fixture["sidecar"].read_text(encoding="utf-8"))
    sidecar["input_loader_audit"]["pixels_shape"][0] = alias
    _json(fixture["sidecar"], sidecar)
    execution = json.loads(
        fixture["execution_manifest"].read_text(encoding="utf-8")
    )
    execution["episodes"][0]["sidecar_sha256"] = _sha(fixture["sidecar"])
    _json(fixture["execution_manifest"], execution)
    with pytest.raises(replay.ReplayError, match="exact integer dimensions"):
        replay.qualify_regime(
            "selection",
            "native_plan",
            stack_loader=lambda: fixture["stack"],
            expected_builder=_builder(fixture),
            attempt_root=fixture["attempt"],
        )


def _synthetic_manifest_verification() -> dict[str, Any]:
    regimes = {}
    for index, regime in enumerate(workflow.REGIMES):
        regimes[regime] = {
            "episode_count": 1,
            "row_count": workflow.ROWS_PER_EPISODE,
            "raw_manifest": {
                "path": f"fixture/{regime}/raw_manifest.json",
                "sha256": f"{index + 1:x}" * 64,
            },
            "execution_manifest": {
                "path": f"fixture/{regime}/execution_manifest.json",
                "sha256": f"{index + 5:x}" * 64,
            },
            "aggregate": {
                "path": f"fixture/{regime}/role.npz",
                "sha256": f"{index + 9:x}" * 64,
                "bytes": 1,
            },
            "episode_ids_sha256": "d" * 64,
            "verified_file_index_sha256": "e" * 64,
        }
    return {
        "role": "fit",
        "regime_order": list(workflow.REGIMES),
        "episodes_per_regime": 1,
        "episode_count": 4,
        "row_count": 4 * workflow.ROWS_PER_EPISODE,
        "authorization_state_sha256": "a" * 64,
        "regimes": regimes,
    }


def _synthetic_replay_result(
    verification: Mapping[str, Any], regime: str
) -> dict[str, Any]:
    item = verification["regimes"][regime]
    return {
        "schema_version": 1,
        "artifact_type": "v008_independent_scientific_replay_regime",
        "attempt": "v008",
        "role": "fit",
        "regime": regime,
        "raw_manifest": item["raw_manifest"],
        "execution_manifest": item["execution_manifest"],
        "compiled_gate": None,
        "episode_count": 1,
        "row_count": workflow.ROWS_PER_EPISODE,
        "episodes": [
            {
                "slot": 0,
                "episode_id": f"fixture-{regime}",
                "raw": {"path": "fixture/raw.npz", "sha256": "1" * 64},
                "raw_sidecar": {"path": "fixture/raw.json", "sha256": "2" * 64},
                "execution_part": {"path": "fixture/part.npz", "sha256": "3" * 64},
                "execution_sidecar": {"path": "fixture/part.json", "sha256": "4" * 64},
                "input_loader_audit_sha256": "5" * 64,
                "array_sha256": {"target": "6" * 64},
                "every_persisted_tensor_exact": True,
            }
        ],
        "aggregate": {
            "applicable": True,
            "path": item["aggregate"]["path"],
            "sha256": item["aggregate"]["sha256"],
            "arrays": {"target": "7" * 64},
            "exact": True,
        },
        "runtime_audit": {
            "passed": True,
            "requested_role": "independent_verification",
            "mps": {"required": True, "built": True, "available": True},
        },
        "source_hashes": {"fixture.py": "8" * 64},
        "module_before": {"passed": True},
        "module_after": {"passed": True},
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "aggregate_exact": True,
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_tree_unchanged": True,
        "read_only": True,
        "passed": True,
    }


def test_role_audit_hash_is_bound_and_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "runs/lewm_domain_robust_gate/attempts/v008"
    monkeypatch.setattr(workflow, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(workflow, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(
        workflow,
        "SCIENTIFIC_REPLAY_AUDIT_ROOT",
        attempt / "audit/scientific_replay",
    )
    verification = _synthetic_manifest_verification()
    qualification = workflow.qualify_role_scientific_replay(
        "fit",
        {"current_state": "FIT_COHORTS"},
        verification,
        launcher=lambda _role, regime: _synthetic_replay_result(
            verification, regime
        ),
    )
    qualified = dict(verification)
    qualified["scientific_replay_qualification"] = qualification
    assert workflow._validate_role_replay_qualification("fit", qualified) == qualification
    role_path = workflow.resolve_repo_relative(qualification["path"])
    audit = json.loads(role_path.read_text(encoding="utf-8"))
    result_path = workflow.resolve_repo_relative(
        audit["regimes"]["native_plan"]["result"]["path"]
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    attacked_result = copy.deepcopy(result)
    attacked_result["passed"] = False
    _json(result_path, attacked_result)
    with pytest.raises(workflow.WorkflowError, match="qualification failed"):
        workflow._validate_role_replay_qualification("fit", qualified)
    _json(result_path, result)
    audit["passed"] = False
    _json(role_path, audit)
    with pytest.raises(workflow.WorkflowError, match="qualification failed"):
        workflow._validate_role_replay_qualification("fit", qualified)


def test_replay_source_never_imports_production_runner() -> None:
    tree = (ROOT / "scientific_replay.py").read_text(encoding="utf-8")
    assert "import runner" not in tree
    assert "from runner" not in tree
