from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


iv = _module("independent_verify_synthetic", "independent_verify.py")
capture = _module("capture_verifier_synthetic", "capture_verifier.py")


def _primitives(rows: int = 4):
    history = np.zeros((rows, 3, 192), dtype=np.float32)
    history[:, 0, 0] = np.arange(rows, dtype=np.float32) - 1.0
    action = np.zeros((rows, 3, 25), dtype=np.float32)
    current = np.zeros((rows, 3, 192), dtype=np.float32)
    update = np.zeros_like(current)
    return history, action, current, update


def test_causal_feature_width_and_strict_greater_routing():
    history, action, dense_current, dense_update = _primitives()
    rows = len(history)
    weights = np.zeros((3, 2, 1046), dtype=np.float32)
    weights[:, :, 0] = 1.0
    biases = np.zeros((3, 2), dtype=np.float32)
    thresholds = np.asarray([0.0, 0.5, 1.5], dtype=np.float32)
    dense_features = np.stack(
        [
            iv.build_causal_features_numpy(history, action, dense_current[:, stage], dense_update[:, stage])
            for stage in range(3)
        ],
        axis=1,
    )
    assert dense_features.shape == (rows, 3, 1046)
    dense_scores = dense_features[:, :, 0]
    calls = np.ones(rows, dtype=np.int64)
    reached = np.ones((rows, 3), dtype=bool)
    active = np.ones(rows, dtype=bool)
    for stage in range(3):
        reached[:, stage] = active
        active &= dense_scores[:, stage] > thresholds[stage]
        calls += active
    assert calls.tolist() == [1, 1, 3, 4]  # x==0 stops: comparison is strict.
    sparse_features = np.full_like(dense_features, np.nan)
    sparse_scores = np.full((rows, 3), np.nan)
    sparse_current = np.full_like(dense_current, np.nan)
    sparse_update = np.full_like(dense_update, np.nan)
    for stage in range(3):
        sparse_features[reached[:, stage], stage] = dense_features[reached[:, stage], stage]
        sparse_scores[reached[:, stage], stage] = dense_scores[reached[:, stage], stage]
        sparse_current[reached[:, stage], stage] = dense_current[reached[:, stage], stage]
        sparse_update[reached[:, stage], stage] = dense_update[reached[:, stage], stage]
    result = iv.reconstruct_scores_calls(
        {
            "calls": calls,
            "scores": sparse_scores,
            "production_features": sparse_features,
            "history": history,
            "action_history": action,
            "stage_current": sparse_current,
            "stage_update": sparse_update,
        },
        weights,
        biases,
        thresholds,
    )
    assert np.array_equal(result["calls"], calls)
    assert result["reached_counts"] == [4, 2, 2]


def test_exact_compute_analytic_is_rational_and_rejects_one_flop_outside():
    losses = np.asarray(
        [[4.0, 3.0, 2.0, 1.0], [1.0, 1.2, 1.4, 1.6], [2.0, 1.8, 1.7, 1.5]],
        dtype=np.float64,
    )
    compute = iv.exact_compute(np.asarray([1, 2, 4]), 2)
    allocation = iv.strongest_analytic(losses, compute["adaptive_total_counted_flops"])
    assert allocation["integer_cross_multiplication_equality"] is True
    assert (
        allocation["allocation_compute_numerator"]
        == compute["adaptive_total_counted_flops"] * allocation["allocation_compute_denominator"]
    )
    minimum = len(losses) * (iv.BASE_FLOPS + iv.MANDATORY_DEPTH1_FLOPS)
    maximum = minimum + len(losses) * 3 * iv.ADDITIONAL_REFINER_FLOPS
    with pytest.raises(iv.VerificationError, match="no exact-compute"):
        iv.strongest_analytic(losses, minimum - 1)
    with pytest.raises(iv.VerificationError, match="no exact-compute"):
        iv.strongest_analytic(losses, maximum + 1)


def test_bootstrap_is_paired_fixed_order_and_terminal_mapping():
    metrics = {
        dgp: {
            name: np.asarray([index, index + 1.0, index + 2.0], dtype=np.float64)
            for index, name in enumerate(iv.ALL_CONTRASTS)
        }
        for dgp in iv.DGP_ORDER
    }
    first = iv.bootstrap_all(metrics, 1234, replicates=17, chunk_size=5)
    second = iv.bootstrap_all(metrics, 1234, replicates=17, chunk_size=5)
    assert all(
        np.array_equal(first[dgp][name], second[dgp][name])
        for dgp in iv.DGP_ORDER
        for name in iv.ALL_CONTRASTS
    )
    positive = {dgp: {endpoint: 0.1 for endpoint in iv.ENDPOINTS} for dgp in iv.DGP_ORDER}
    assert iv.terminal_mapping(positive, True) == (iv.TERMINAL_CONFIRMED, 8)
    positive[iv.DGP_ORDER[-1]][iv.ENDPOINTS[-1]] = -1e-9
    assert iv.terminal_mapping(positive, True) == (iv.TERMINAL_PARTIAL, 7)
    assert iv.terminal_mapping(positive, False)[0] == iv.TERMINAL_INVALID


def test_stdlib_chi_square_quantile_reproduces_frozen_power_bound():
    # Reference values are independently frozen numerical constants (the
    # verifier cannot import SciPy).
    assert iv._chi_square_quantile(0.05, 299) == pytest.approx(259.94538288343006, abs=2e-12)
    assert iv._chi_square_quantile(0.05, 499) == pytest.approx(448.19882158627, abs=2e-12)


def test_power_provenance_is_local_identity_bound_and_tamper_evident(tmp_path: Path):
    policy, attempt = _policy(tmp_path)
    paths = {
        "power_rule": attempt / "power_rule.json",
        "fit_lock": attempt / "fit/fit_lock.json",
        "selection_ledger": attempt / "selection/selection_ledger.json",
        "gate_freeze": attempt / "freeze/gate_freeze.json",
        "fit_summary": attempt / "metrics/fit_power_summary.json",
        "selection_summary": attempt / "metrics/selection_power_summary.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["power_rule"].write_bytes((ROOT / "power_rule.json").read_bytes())
    candidate_id = "candidate_007"
    object_hash = "a" * 64
    fit_lock = {
        "status": "all_24_candidates_fit_compiled_and_locked_before_selection_open",
        "selection_input_opened_before_lock": False,
    }
    paths["fit_lock"].write_text(json.dumps(fit_lock))
    fit_lock_hash = hashlib.sha256(paths["fit_lock"].read_bytes()).hexdigest()
    selection = {
        "status": "selection_complete_no_selected_head_refit",
        "selected_candidate_id": candidate_id,
        "selected_head_refit_after_selection": False,
        "fit_lock_sha256": fit_lock_hash,
        "candidates": [
            {
                "candidate_id": candidate_id,
                "candidate_object_sha256": object_hash,
                "eligible": True,
            }
        ],
    }
    paths["selection_ledger"].write_text(json.dumps(selection))
    selection_hash = hashlib.sha256(paths["selection_ledger"].read_bytes()).hexdigest()
    freeze = {
        "status": "selected_gate_frozen_before_smoke_or_confirmation",
        "selected_candidate_id": candidate_id,
        "selected_candidate_object_sha256": object_hash,
        "fit_lock_sha256": fit_lock_hash,
        "selection_ledger_sha256": selection_hash,
        "selected_head_refit_after_selection": False,
        "confirmation_episodes_at_freeze": 0,
    }
    paths["gate_freeze"].write_text(json.dumps(freeze))
    freeze_hash = hashlib.sha256(paths["gate_freeze"].read_bytes()).hexdigest()
    identity = {
        "selected_candidate_id": candidate_id,
        "selected_candidate_object_sha256": object_hash,
        "fit_lock_sha256": fit_lock_hash,
        "selection_ledger_sha256": selection_hash,
        "gate_freeze_sha256": freeze_hash,
    }
    claims = {
        dgp: {
            endpoint: {"episode_count": 300, "mean": 0.1, "sd_ddof1": 0.2}
            for endpoint in iv.PRIMARY_CONTRASTS
        }
        for dgp in iv.DGP_ORDER
    }
    for role, path in (
        ("fit", paths["fit_summary"]),
        ("selection", paths["selection_summary"]),
    ):
        document = {
            "schema_version": 1,
            "attempt": "v008",
            "role": role,
            "co_primary_only": True,
            "auxiliary_robust_whitening_excluded": True,
            "claims": claims,
            **identity,
        }
        path.write_text(json.dumps(document))
    relative = lambda path: path.relative_to(tmp_path).as_posix()
    inputs = {
        name: {
            "path": relative(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
        if name != "power_rule"
    }
    inputs["power_rule"] = {
        "path": relative(paths["power_rule"]),
        "sha256": hashlib.sha256(paths["power_rule"].read_bytes()).hexdigest(),
        "expected_sha256": iv.POWER_RULE_SHA256,
        "endpoint_moments_object_sha256": iv.POWER_RULE_MOMENTS_SHA256,
    }
    power = {"inputs": inputs, "selected_gate_identity": identity}
    contract = {
        "attempt_root": "runs/study/attempts/v008",
        "paths": {
            "power_rule": relative(paths["power_rule"]),
            "fit_lock": relative(paths["fit_lock"]),
            "selection_ledger": relative(paths["selection_ledger"]),
            "gate_freeze": relative(paths["gate_freeze"]),
        },
    }

    provenance = iv.verify_power_input_provenance(power, contract, policy)

    assert provenance["local_power_rule_only"] is True
    assert all(provenance["identity_crosslinks"].values())
    source = (ROOT / "independent_verify.py").read_text(encoding="utf-8")
    assert "lewm_v5_generalization" not in source
    assert "consumed_v5_power" not in source

    fit_document = json.loads(paths["fit_summary"].read_text())
    fit_document["auxiliary_robust_whitening_excluded"] = False
    paths["fit_summary"].write_text(json.dumps(fit_document))
    power["inputs"]["fit_summary"]["sha256"] = hashlib.sha256(paths["fit_summary"].read_bytes()).hexdigest()
    with pytest.raises(iv.VerificationError, match="auxiliary"):
        iv.verify_power_input_provenance(power, contract, policy)
    fit_document["auxiliary_robust_whitening_excluded"] = True
    fit_document["selected_candidate_object_sha256"] = "b" * 64
    paths["fit_summary"].write_text(json.dumps(fit_document))
    power["inputs"]["fit_summary"]["sha256"] = hashlib.sha256(paths["fit_summary"].read_bytes()).hexdigest()
    with pytest.raises(iv.VerificationError, match="selected identity"):
        iv.verify_power_input_provenance(power, contract, policy)


def test_independent_rank_calibration_serializes_undefined_spearman_as_null():
    scores = np.full((3, 3), np.nan, dtype=np.float64)
    calls = np.zeros(3, dtype=np.int64)
    losses = {
        endpoint: np.zeros((3, 5), dtype=np.float64)
        for endpoint in iv.ENDPOINTS
    }

    ranks, calibration = iv.independent_rank_calibration(scores, calls, losses)

    assert len(ranks) == 3
    assert len(calibration) == 3
    for stage in ranks:
        for endpoint in iv.ENDPOINTS:
            diagnostic = stage["endpoints"][endpoint]
            assert diagnostic["spearman_score_next_stage_gain_rho"] is None
            assert diagnostic["rank_sign"] == "zero_or_undefined"
            assert diagnostic["positive_sign"] is False
            assert diagnostic["mean_next_stage_gain"] is None
    json.dumps(ranks, allow_nan=False)


def _policy(tmp_path: Path):
    attempt = tmp_path / "runs/study/attempts/v008"
    attempt.mkdir(parents=True)
    return iv.PathPolicy(tmp_path, "runs/study/attempts/v008", "runs/study"), attempt


def _development_loader_case(
    tmp_path: Path, *, array_tamper: str | None = None
) -> dict[str, Any]:
    policy, attempt = _policy(tmp_path)
    raw_paths: dict[str, str] = {}
    manifest_paths: dict[str, str] = {}
    replay_regimes: dict[str, Any] = {}
    rows = iv.ROWS_PER_EPISODE
    history = np.zeros((rows, iv.HISTORY_LEN, iv.LATENT_DIM), dtype=np.float32)
    action = np.zeros((rows, iv.HISTORY_LEN, iv.ACTION_DIM), dtype=np.float32)
    stage_current = np.zeros(
        (rows, iv.STAGE_COUNT, iv.LATENT_DIM), dtype=np.float32
    )
    stage_update = np.zeros_like(stage_current)
    exits = np.zeros((rows, iv.EXIT_COUNT, iv.LATENT_DIM), dtype=np.float32)
    target = np.full((rows, iv.LATENT_DIM), 0.25, dtype=np.float32)
    features = np.stack(
        [
            iv.build_causal_features_numpy(
                history, action, stage_current[:, stage], stage_update[:, stage]
            )
            for stage in range(iv.STAGE_COUNT)
        ],
        axis=1,
    )
    base_part: dict[str, np.ndarray] = {
        "episode_slot": np.zeros(rows, dtype=np.int32),
        "model_step": np.arange(3, 3 + rows, dtype=np.int16),
        "target": target,
        "exits": exits,
        "production_features": features,
        "history": history,
        "action_history": action,
        "stage_current": stage_current,
        "stage_update": stage_update,
    }
    if array_tamper == "episode_slot_dtype":
        base_part["episode_slot"] = base_part["episode_slot"].astype(np.float64)
    elif array_tamper == "target_dtype":
        base_part["target"] = base_part["target"].astype(np.float64)
    elif array_tamper == "exits_dtype":
        base_part["exits"] = base_part["exits"].astype(np.float64)
    elif array_tamper == "exits_stage_coherence":
        base_part["exits"] = base_part["exits"].copy()
        base_part["exits"][:, 0, 0] = 1.0
    elif array_tamper is not None:
        raise AssertionError(array_tamper)

    for dgp in iv.DGP_ORDER:
        root = attempt / "synthetic" / "fit" / dgp
        root.mkdir(parents=True)
        part_path = root / "000000.npz"
        aggregate_path = root / "role.npz"
        sidecar_path = root / "000000.json"
        np.savez(part_path, **base_part)
        aggregate = {
            name: base_part[name] for name in iv.DEVELOPMENT_AGGREGATE_KEYS
        }
        np.savez(aggregate_path, **aggregate)
        sidecar_path.write_text("{}\n", encoding="utf-8")

        relative = lambda path: path.relative_to(tmp_path).as_posix()
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        record = {
            "slot": 0,
            "episode_id": f"fit-{dgp}-000000",
            "path": relative(part_path),
            "sha256": sha(part_path),
            "sidecar_path": relative(sidecar_path),
            "sidecar_sha256": sha(sidecar_path),
            "source_raw_path": f"runs/study/attempts/v008/raw/{dgp}/000000.npz",
            "source_raw_sha256": "a" * 64,
            "source_raw_sidecar_path": f"runs/study/attempts/v008/raw/{dgp}/000000.json",
            "source_raw_sidecar_sha256": "b" * 64,
            "call_histogram": None,
        }
        manifest = {
            "schema_version": 1,
            "attempt": "v008",
            "created_unix_ns": 10,
            "role": "fit",
            "regime": dgp,
            "complete": True,
            "episode_count": 1,
            "row_count": rows,
            "episodes": [record],
            "aggregate_role_path": relative(aggregate_path),
            "aggregate_role_sha256": sha(aggregate_path),
            "aggregate_recovery": {"passed": True},
            "aggregate_role_keys": [
                "episode_slot",
                "model_step",
                "target",
                "exits",
                "production_features",
            ],
            "raw_manifest_path": f"runs/study/attempts/v008/raw/{dgp}/manifest.json",
            "raw_manifest_sha256": "c" * 64,
            "source_raw_manifest_sha256": "c" * 64,
            "authorization": {"passed": True},
            "source_bindings": {"sealed": True},
            "loaded_input_keys": ["action", "pixels"],
            "contact_or_privileged_materialized": False,
            "target_role_isolation": "fit",
            "causal_primitive_contract": {
                "history": [rows, iv.HISTORY_LEN, iv.LATENT_DIM],
                "action_history": [rows, iv.HISTORY_LEN, iv.ACTION_DIM],
                "stage_current": [rows, iv.STAGE_COUNT, iv.LATENT_DIM],
                "stage_update": [rows, iv.STAGE_COUNT, iv.LATENT_DIM],
                "sufficient_for_independent_feature_reconstruction": True,
            },
            "module_before": {"passed": True},
            "module_after": {"passed": True},
            "base_provenance": {"passed": True},
            "frozen_facade": {"passed": True},
            "no_gradients": True,
        }
        manifest_path = root / "execution_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_paths[dgp] = relative(aggregate_path)
        manifest_paths[dgp] = relative(manifest_path)
        replay_regimes[dgp] = {
            "result": {
                "passed": True,
                "role": "fit",
                "regime": dgp,
                "episode_count": 1,
                "aggregate": {
                    "applicable": True,
                    "path": relative(aggregate_path),
                    "sha256": sha(aggregate_path),
                    "arrays": {
                        name: iv.replay_array_sha256(value)
                        for name, value in aggregate.items()
                    },
                    "exact": True,
                },
                "episodes": [
                    {
                        "slot": 0,
                        "episode_id": record["episode_id"],
                        "execution_part": {
                            "path": relative(part_path),
                            "sha256": sha(part_path),
                        },
                        "execution_sidecar": {
                            "path": relative(sidecar_path),
                            "sha256": sha(sidecar_path),
                        },
                        "array_sha256": {
                            name: iv.replay_array_sha256(value)
                            for name, value in base_part.items()
                        },
                        "every_persisted_tensor_exact": True,
                    }
                ],
            }
        }
    return {
        "policy": policy,
        "attempt": attempt,
        "raw_paths": raw_paths,
        "manifest_paths": manifest_paths,
        "replay_role": {
            "role_audit": {"passed": True},
            "regimes": replay_regimes,
        },
    }


def test_independent_development_loader_accepts_exact_typed_coherent_parts(
    tmp_path: Path,
) -> None:
    case = _development_loader_case(tmp_path)

    role, evidence = iv.load_development_role(
        case["raw_paths"],
        case["manifest_paths"],
        case["policy"],
        episodes_per_dgp=1,
        role="fit",
        replay_role=case["replay_role"],
    )

    assert role["target"].shape == (4 * iv.ROWS_PER_EPISODE, iv.LATENT_DIM)
    assert role["exits"].shape == (
        4 * iv.ROWS_PER_EPISODE,
        iv.EXIT_COUNT,
        iv.LATENT_DIM,
    )
    assert np.all(role["target"] == 0.25)
    assert np.all(role["exits"] == 0.0)
    assert evidence["maximum_feature_absolute_delta"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("created_unix_ns", True),
        ("episode_count", True),
        ("row_count", float(iv.ROWS_PER_EPISODE)),
        ("loaded_input_keys", "action,pixels"),
        ("episodes", {"slot": 0}),
        ("causal_history_float", None),
        ("record_slot_bool", None),
    ],
)
def test_independent_development_loader_rejects_json_type_aliases(
    tmp_path: Path, field: str, value: Any
) -> None:
    case = _development_loader_case(tmp_path)
    manifest_path = tmp_path / case["manifest_paths"][iv.DGP_ORDER[0]]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "causal_history_float":
        manifest["causal_primitive_contract"]["history"][-1] = float(iv.LATENT_DIM)
    elif field == "record_slot_bool":
        manifest["episodes"][0]["slot"] = False
    else:
        manifest[field] = value
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(iv.VerificationError, match="type|list|count|header|identity"):
        iv.load_development_role(
            case["raw_paths"],
            case["manifest_paths"],
            case["policy"],
            episodes_per_dgp=1,
            role="fit",
            replay_role=case["replay_role"],
        )


@pytest.mark.parametrize(
    "array_tamper",
    ["episode_slot_dtype", "target_dtype", "exits_dtype", "exits_stage_coherence"],
)
def test_independent_development_loader_rejects_coherent_array_type_or_exit_tamper(
    tmp_path: Path, array_tamper: str
) -> None:
    case = _development_loader_case(tmp_path, array_tamper=array_tamper)

    with pytest.raises(iv.VerificationError, match="dtype|exits"):
        iv.load_development_role(
            case["raw_paths"],
            case["manifest_paths"],
            case["policy"],
            episodes_per_dgp=1,
            role="fit",
            replay_role=case["replay_role"],
        )


@pytest.mark.parametrize("name", ["target", "fourth_exit"])
def test_development_loader_rejects_coherent_rewrite_before_feature_arithmetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    case = _development_loader_case(tmp_path)
    dgp = iv.DGP_ORDER[0]
    manifest_path = tmp_path / case["manifest_paths"][dgp]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    part_path = tmp_path / manifest["episodes"][0]["path"]
    aggregate_path = tmp_path / manifest["aggregate_role_path"]
    with np.load(part_path, allow_pickle=False) as stored:
        part = {key: stored[key].copy() for key in stored.files}
    if name == "target":
        part["target"][:, 0] = 9.0
    else:
        part["exits"][:, 3, 0] = 9.0
    np.savez(part_path, **part)
    aggregate = {
        key: part[key] for key in iv.DEVELOPMENT_AGGREGATE_KEYS
    }
    np.savez(aggregate_path, **aggregate)
    manifest["episodes"][0]["sha256"] = hashlib.sha256(
        part_path.read_bytes()
    ).hexdigest()
    manifest["aggregate_role_sha256"] = hashlib.sha256(
        aggregate_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def arithmetic_must_not_run(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("feature arithmetic ran before replay authentication")

    monkeypatch.setattr(iv, "_verify_dense_causal_features", arithmetic_must_not_run)
    with pytest.raises(iv.VerificationError, match="replay aggregate"):
        iv.load_development_role(
            case["raw_paths"],
            case["manifest_paths"],
            case["policy"],
            episodes_per_dgp=1,
            role="fit",
            replay_role=case["replay_role"],
        )


def _scientific_replay_qualification_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    policy, attempt = _policy(tmp_path)
    study = attempt.parents[1]
    state_path = study / "STATE.json"
    ledger_path = study / "RESEARCH_LEDGER.jsonl"
    state = {
        "active_attempt": "v008",
        "current_state": "INDEPENDENT_VERIFICATION",
        "ledger_event_count": 9,
        "ledger_head_sha256": "e" * 64,
        "fit_outcome_episodes": 4,
        "selection_outcome_episodes": 4,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
        "expected_confirmation_episode_count": 0,
    }

    def write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def relative(path: Path) -> str:
        return path.relative_to(tmp_path).as_posix()

    def link(path: Path) -> dict[str, str]:
        return {
            "path": relative(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    write_json(state_path, state)
    ledger_path.write_bytes(b"synthetic-ledger\n")
    source_path = attempt / "scientific_replay.py"
    source_path.write_text("REPLAY = 1\n", encoding="utf-8")
    source_hashes = {relative(source_path): hashlib.sha256(source_path.read_bytes()).hexdigest()}
    metadata: dict[str, dict[str, Any]] = {}
    stable_bindings: dict[str, dict[str, Any]] = {}
    role_records: dict[str, Any] = {}
    terminal_paths: list[Path] = [source_path]
    for role in ("fit", "selection"):
        regimes: dict[str, Any] = {}
        role_regimes: dict[str, Any] = {}
        for dgp in iv.DGP_ORDER:
            result_path = attempt / "audit/scientific_replay" / f"{role}_{dgp}.json"
            write_json(result_path, {})
            terminal_paths.append(result_path)
            raw_link = {
                "path": f"runs/study/attempts/v008/data/{role}/{dgp}/raw_manifest.json",
                "sha256": "1" * 64,
            }
            execution_link = {
                "path": f"runs/study/attempts/v008/data/{role}/{dgp}/execution_manifest.json",
                "sha256": "2" * 64,
            }
            aggregate = {
                "applicable": True,
                "path": f"runs/study/attempts/v008/data/{role}/{dgp}/role.npz",
                "sha256": "3" * 64,
                "arrays": {name: "4" * 64 for name in iv.DEVELOPMENT_AGGREGATE_KEYS},
                "exact": True,
            }
            observed = {
                "result": {},
                "raw_manifest": raw_link,
                "execution_manifest": execution_link,
                "aggregate": aggregate,
                "episode_count": 1,
                "row_count": iv.ROWS_PER_EPISODE,
                "authorization_state_sha256": "5" * 64,
                "episode_ids_sha256": "6" * 64,
                "verified_file_index_sha256": "7" * 64,
                "source_hashes": source_hashes,
            }
            metadata[f"{role}/{dgp}"] = observed
            result_link = link(result_path)
            role_regimes[dgp] = {
                "result": result_link,
                "raw_manifest": raw_link,
                "execution_manifest": execution_link,
                "aggregate": aggregate,
                "episode_count": 1,
                "row_count": iv.ROWS_PER_EPISODE,
            }
            regimes[dgp] = {
                "raw_manifest": raw_link,
                "execution_manifest": execution_link,
                "aggregate": {
                    "path": aggregate["path"],
                    "sha256": aggregate["sha256"],
                    "bytes": 1,
                },
                "episode_count": 1,
                "row_count": iv.ROWS_PER_EPISODE,
                "episode_ids_sha256": "6" * 64,
                "verified_file_index_sha256": "7" * 64,
            }
        stable = {
            "role": role,
            "state": iv.REPLAY_ROLE_STATES[role],
            "regime_order": list(iv.DGP_ORDER),
            "episodes_per_regime": 1,
            "episode_count": 4,
            "row_count": 4 * iv.ROWS_PER_EPISODE,
            "authorization_state_sha256": "5" * 64,
            "regimes": regimes,
        }
        stable_bindings[role] = stable
        role_audit = {
            "schema_version": 1,
            "artifact_type": "v008_independent_scientific_replay_role",
            "attempt": "v008",
            "role": role,
            "state": iv.REPLAY_ROLE_STATES[role],
            "authorization_state_sha256": "5" * 64,
            "manifest_verification_sha256": iv.canonical_object_sha256(stable),
            "regime_order": list(iv.DGP_ORDER),
            "episodes_per_regime": 1,
            "episode_count": 4,
            "row_count": 4 * iv.ROWS_PER_EPISODE,
            "regimes": role_regimes,
            "source_hashes": source_hashes,
            "input_loader_agreement_exact": True,
            "every_persisted_tensor_exact": True,
            "all_development_aggregates_exact": True,
            "all_runtime_mps_exact": True,
            "target_loss_computed": False,
            "loss_or_effect_used_for_acceptance": False,
            "contact_or_privileged_materialized": False,
            "no_gradients": True,
            "artifact_trees_unchanged": True,
            "read_only": True,
            "passed": True,
        }
        role_path = attempt / "audit" / f"{role}_scientific_replay.json"
        write_json(role_path, role_audit)
        terminal_paths.append(role_path)
        role_records[role] = {
            **link(role_path),
            "role": role,
            "state": iv.REPLAY_ROLE_STATES[role],
            "authorization_state_sha256": "5" * 64,
            "manifest_verification_sha256": iv.canonical_object_sha256(stable),
            "episode_count": 4,
            "row_count": 4 * iv.ROWS_PER_EPISODE,
            "passed": True,
        }
    snapshot = {
        "active_attempt": "v008",
        "current_state": "INDEPENDENT_VERIFICATION",
        "state": {
            **link(state_path),
            "bytes": state_path.stat().st_size,
            "object_sha256": iv.canonical_object_sha256(state),
        },
        "ledger": {
            **link(ledger_path),
            "bytes": ledger_path.stat().st_size,
            "event_count": 9,
            "head_sha256": "e" * 64,
        },
        "outcome_counts": {
            name: state[name]
            for name in (
                "fit_outcome_episodes",
                "selection_outcome_episodes",
                "smoke_outcome_episodes",
                "confirmation_outcome_episodes_generated",
                "confirmation_outcome_episodes_executed",
                "confirmation_outcomes_opened_for_analysis",
            )
        },
    }
    audit = {
        "schema_version": 1,
        "artifact_type": "v008_terminal_scientific_replay_qualification",
        "attempt": "v008",
        "mode": "no_candidate",
        "checkpoint_state": "INDEPENDENT_VERIFICATION",
        "role_order": ["fit", "selection"],
        "regime_order": list(iv.DGP_ORDER),
        "role_count": 2,
        "qualification_count": 8,
        "episode_count": 8,
        "row_count": 8 * iv.ROWS_PER_EPISODE,
        "role_audits": role_records,
        "source_hashes": source_hashes,
        "controller_snapshot": snapshot,
        "controller_unchanged_during_replay": True,
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "all_applicable_development_aggregates_exact": True,
        "all_runtime_mps_exact": True,
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_trees_unchanged": True,
        "read_only": True,
        "terminal_manifest_created_only_after_replay_qualification": True,
        "passed": True,
    }
    audit_path = attempt / iv.SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE
    write_json(audit_path, audit)
    terminal_paths.append(audit_path)
    terminal_path = attempt / "audit/terminal_input_manifest.json"
    terminal = {
        "files": {
            relative(path): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in terminal_paths
        }
    }
    write_json(terminal_path, terminal)
    contract = {
        "mode": "no_candidate",
        "paths": {
            "terminal_manifest": relative(terminal_path),
            "state": relative(state_path),
            "ledger": relative(ledger_path),
        },
    }
    monkeypatch.setattr(
        iv,
        "REPLAY_FIXED_EPISODES_PER_DGP",
        {"fit": 1, "selection": 1, "smoke": 1},
    )
    monkeypatch.setattr(
        iv,
        "_verify_replay_regime_result",
        lambda _result, *, role, dgp, **_kwargs: metadata[f"{role}/{dgp}"],
    )
    monkeypatch.setattr(
        iv,
        "_independent_development_replay_binding",
        lambda role, _policy: stable_bindings[role],
    )
    return {
        "policy": policy,
        "state": state,
        "contract": contract,
        "audit": audit,
        "audit_path": audit_path,
        "terminal_path": terminal_path,
    }


def test_independent_replay_qualification_accepts_exact_cross_role_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _scientific_replay_qualification_case(tmp_path, monkeypatch)

    result = iv.verify_scientific_replay_qualification(
        case["contract"],
        case["policy"],
        case["state"],
        confirmation_n=0,
    )

    assert result["passed"] is True
    assert result["qualification_count"] == 8
    assert result["role_order"] == ["fit", "selection"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("role_count", 2.0), ("all_runtime_mps_exact", False)],
)
def test_independent_replay_qualification_rejects_type_or_exactness_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    case = _scientific_replay_qualification_case(tmp_path, monkeypatch)
    audit = dict(case["audit"])
    audit[field] = value
    case["audit_path"].write_text(
        json.dumps(audit, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terminal = json.loads(case["terminal_path"].read_text(encoding="utf-8"))
    relative = case["audit_path"].relative_to(tmp_path).as_posix()
    terminal["files"][relative] = {
        "sha256": hashlib.sha256(case["audit_path"].read_bytes()).hexdigest(),
        "bytes": case["audit_path"].stat().st_size,
    }
    case["terminal_path"].write_text(
        json.dumps(terminal, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(iv.VerificationError, match="type|assertion"):
        iv.verify_scientific_replay_qualification(
            case["contract"],
            case["policy"],
            case["state"],
            confirmation_n=0,
        )


def _durable_controller_marker_case(
    tmp_path: Path,
) -> tuple[Any, dict[str, Any], int, str, Path, Path]:
    policy, attempt = _policy(tmp_path)
    study = attempt.parents[1]
    adapter = attempt / "version_forward_transaction.py"
    root_program = study / "program.py"
    adapter.write_text("ADAPTER_VERSION = 2\n", encoding="utf-8")
    root_program.write_text("PROGRAM_VERSION = 2\n", encoding="utf-8")

    def ast_sha256(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode("utf-8")
        ).hexdigest()

    event_count = 17
    head = "a" * 64
    state_without_marker = {
        "ledger_event_count": event_count,
        "ledger_head_sha256": head,
    }
    marker = {
        "schema_version": 2,
        "authorization_kind": "receipt_bound_v008_root_controller_adapter",
        "adapter_source_path": adapter.relative_to(tmp_path).as_posix(),
        "adapter_source_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "adapter_source_ast_sha256": ast_sha256(adapter),
        "root_program_path": root_program.relative_to(tmp_path).as_posix(),
        "root_program_sha256": hashlib.sha256(root_program.read_bytes()).hexdigest(),
        "root_program_ast_sha256": ast_sha256(root_program),
        "ledger_event_count": event_count,
        "ledger_head_sha256": head,
        "operation_sha256": "b" * 64,
    }
    marker["state_binding_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "marker_without_state_binding": marker,
                "state_without_marker": state_without_marker,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    state = {**state_without_marker, "v008_durable_controller_adapter": marker}
    return policy, state, event_count, head, adapter, root_program


def _version_forward_receipt_v2_case_legacy(
    tmp_path: Path,
    *,
    proposal_mutator: Any = None,
    fresh_sealed_file_count: int = 121,
    fresh_partition_target_count: int = 57,
    stored_sealed_file_count: int | None = None,
    stored_partition_target_count: int | None = None,
) -> dict[str, Any]:
    """Construct a transitive v005 -> v008 receipt from declarative equations."""

    if stored_sealed_file_count is None:
        stored_sealed_file_count = fresh_sealed_file_count
    if stored_partition_target_count is None:
        stored_partition_target_count = fresh_partition_target_count

    study_relative = iv.STUDY_RELATIVE
    v001_relative = iv.ATTEMPT_ROOTS[iv.SCIENCE_ATTEMPT]
    v002_relative = iv.ATTEMPT_ROOTS[iv.PRIOR_ATTEMPT]
    v003_relative = iv.ATTEMPT_ROOTS[iv.FIT_SOURCE_ATTEMPT]
    v004_relative = iv.ATTEMPT_ROOTS[iv.INTERMEDIATE_ATTEMPT]
    v005_relative = iv.ATTEMPT_ROOTS[iv.SOURCE_ATTEMPT]
    v008_relative = iv.ATTEMPT_ROOTS[iv.ACTIVE_ATTEMPT]
    attempt = tmp_path / v008_relative
    attempt.mkdir(parents=True)
    policy = iv.PathPolicy(tmp_path, v008_relative, study_relative)
    study = tmp_path / study_relative
    v001_audit = tmp_path / v001_relative / "audit"
    v002_audit = tmp_path / v002_relative / "audit"
    v003_audit = tmp_path / v003_relative / "audit"
    v004 = tmp_path / v004_relative
    v004_audit = v004 / "audit"
    v005 = tmp_path / v005_relative
    v005_audit = v005 / "audit"
    v008_audit = attempt / "audit"
    for directory in (
        v001_audit,
        v002_audit,
        v003_audit,
        v004_audit,
        v005_audit,
        v008_audit,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    def compact(value: Any) -> bytes:
        return json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def pretty(value: Any) -> bytes:
        return (
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    def object_hash(value: Any) -> str:
        return hashlib.sha256(compact(value)).hexdigest()

    def clone(value: Any) -> Any:
        return json.loads(json.dumps(value, allow_nan=False))

    def ast_hash(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode("utf-8")
        ).hexdigest()

    def rich_record(path: Path, *, source: bool = False) -> dict[str, Any]:
        metadata = path.lstat()
        payload = path.read_bytes()
        record = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "nlink": int(metadata.st_nlink),
        }
        if source:
            record["ast_sha256"] = ast_hash(path)
        return record

    def state_record(state: dict[str, Any]) -> dict[str, Any]:
        payload = pretty(state)
        return {
            "path": f"{study_relative}/STATE.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "object_sha256": object_hash(state),
            "object": clone(state),
        }

    root_program = study / "program.py"
    source_adapter = v005 / "version_forward_transaction.py"
    adapter = attempt / "version_forward_transaction.py"
    seal_path = v008_audit / "pre_data_inheritance_seal.json"
    prior_seal_path = v002_audit / "pre_data_inheritance_seal.json"
    fit_seal_path = v003_audit / "pre_data_inheritance_seal.json"
    intermediate_seal_path = v004_audit / "pre_data_inheritance_seal.json"
    source_seal_path = v005_audit / "pre_data_inheritance_seal.json"
    invalidity_path = tmp_path / iv.INVALIDITY_RELATIVE
    invalidity_draft_path = tmp_path / iv.INVALIDITY_DRAFT_RELATIVE
    prior_receipt_path = tmp_path / iv.SOURCE_RECEIPT_RELATIVE
    science_invalidity_path = tmp_path / iv.SCIENCE_INVALIDITY_RELATIVE
    prior_invalidity_path = v002_audit / "v002_procedural_invalidity_v2.json"
    fit_invalidity_path = v003_audit / "v003_procedural_invalidity.json"
    intermediate_invalidity_path = v004_audit / "v004_procedural_invalidity.json"
    checkpoint_path = v001_audit / "bootstrap_audit.json"
    root_program.write_text("PROGRAM_VERSION = 3\n", encoding="utf-8")
    source_adapter.write_text("ADAPTER_VERSION = 2\n", encoding="utf-8")
    adapter.write_text("ADAPTER_VERSION = 3\n", encoding="utf-8")
    seal_path.write_bytes(pretty({"artifact": "inheritance_v008", "passed": True}))
    prior_seal_path.write_bytes(
        pretty({"artifact": "inheritance_v002", "passed": True})
    )
    fit_seal_path.write_bytes(
        pretty({"artifact": "inheritance_v003", "passed": True})
    )
    intermediate_seal_path.write_bytes(
        pretty({"artifact": "inheritance_v004", "passed": True})
    )
    source_seal_path.write_bytes(
        pretty({"artifact": "inheritance_v005", "passed": True})
    )
    invalidity_path.write_bytes(
        pretty({"artifact": "v005_procedural_invalidity", "passed": True})
    )
    invalidity_draft_path.write_bytes(
        pretty({"artifact": "v005_procedural_invalidity_draft", "passed": False})
    )
    prior_invalidity_path.write_bytes(
        pretty({"artifact": "v002_procedural_invalidity_v2", "passed": True})
    )
    fit_invalidity_path.write_bytes(
        pretty({"artifact": "v003_procedural_invalidity", "passed": True})
    )
    intermediate_invalidity_path.write_bytes(
        pretty({"artifact": "v004_procedural_invalidity", "passed": True})
    )
    science_invalidity_path.write_bytes(
        pretty({"artifact": "v001_procedural_invalidity", "passed": True})
    )
    checkpoint_path.write_bytes(pretty({"attempt": "v001", "passed": True}))

    seal_hash = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    prior_seal_hash = hashlib.sha256(prior_seal_path.read_bytes()).hexdigest()
    fit_seal_hash = hashlib.sha256(fit_seal_path.read_bytes()).hexdigest()
    intermediate_seal_hash = hashlib.sha256(
        intermediate_seal_path.read_bytes()
    ).hexdigest()
    source_seal_hash = hashlib.sha256(source_seal_path.read_bytes()).hexdigest()
    invalidity_hash = hashlib.sha256(invalidity_path.read_bytes()).hexdigest()
    science_invalidity_hash = hashlib.sha256(
        science_invalidity_path.read_bytes()
    ).hexdigest()
    prior_invalidity_hash = hashlib.sha256(
        prior_invalidity_path.read_bytes()
    ).hexdigest()
    fit_invalidity_hash = hashlib.sha256(
        fit_invalidity_path.read_bytes()
    ).hexdigest()
    intermediate_invalidity_hash = hashlib.sha256(
        intermediate_invalidity_path.read_bytes()
    ).hexdigest()
    checkpoint = {
        "created_unix_ns": 2,
        "evidence_path": checkpoint_path.relative_to(tmp_path).as_posix(),
        "evidence_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "name": "v001_bootstrap_audit_passed",
        "source_attempt": "v001",
        "verification_lineage": "direct_checkpoint",
        "inherited_into_attempts": [
            {
                "attempt": "v002",
                "equivalence_evidence_path": prior_seal_path.relative_to(
                    tmp_path
                ).as_posix(),
                "equivalence_evidence_sha256": prior_seal_hash,
            },
            {
                "attempt": "v003",
                "equivalence_evidence_path": fit_seal_path.relative_to(
                    tmp_path
                ).as_posix(),
                "equivalence_evidence_sha256": fit_seal_hash,
            },
            {
                "attempt": "v004",
                "equivalence_evidence_path": intermediate_seal_path.relative_to(
                    tmp_path
                ).as_posix(),
                "equivalence_evidence_sha256": intermediate_seal_hash,
            },
            {
                "attempt": "v005",
                "equivalence_evidence_path": source_seal_path.relative_to(
                    tmp_path
                ).as_posix(),
                "equivalence_evidence_sha256": source_seal_hash,
            },
        ],
    }
    projection = [
        {
            "checkpoint_name": checkpoint["name"],
            "source_attempt": "v001",
            "evidence_path": checkpoint["evidence_path"],
            "evidence_sha256": checkpoint["evidence_sha256"],
        }
    ]
    first_edge = {
        "old_attempt": "v001",
        "new_attempt": "v002",
        "resume_state": "FIT_COHORTS",
        "invalidity_path": science_invalidity_path.relative_to(tmp_path).as_posix(),
        "invalidity_sha256": science_invalidity_hash,
        "equivalence_path": prior_seal_path.relative_to(tmp_path).as_posix(),
        "equivalence_sha256": prior_seal_hash,
        "inherited_verified_checkpoints": clone(projection),
        "attempt_parameterization_verified": True,
    }
    fit_edge = {
        "old_attempt": "v002",
        "new_attempt": "v003",
        "resume_state": "FIT_COHORTS",
        "invalidity_path": prior_invalidity_path.relative_to(tmp_path).as_posix(),
        "invalidity_sha256": prior_invalidity_hash,
        "equivalence_path": fit_seal_path.relative_to(tmp_path).as_posix(),
        "equivalence_sha256": fit_seal_hash,
        "inherited_verified_checkpoints": clone(projection),
        "attempt_parameterization_verified": True,
    }
    intermediate_edge = {
        "old_attempt": "v003",
        "new_attempt": "v004",
        "resume_state": "FIT_COHORTS",
        "invalidity_path": fit_invalidity_path.relative_to(tmp_path).as_posix(),
        "invalidity_sha256": fit_invalidity_hash,
        "equivalence_path": intermediate_seal_path.relative_to(tmp_path).as_posix(),
        "equivalence_sha256": intermediate_seal_hash,
        "inherited_verified_checkpoints": clone(projection),
        "attempt_parameterization_verified": True,
    }
    source_edge = {
        "old_attempt": "v004",
        "new_attempt": "v005",
        "resume_state": "FIT_COHORTS",
        "invalidity_path": intermediate_invalidity_path.relative_to(
            tmp_path
        ).as_posix(),
        "invalidity_sha256": intermediate_invalidity_hash,
        "equivalence_path": source_seal_path.relative_to(tmp_path).as_posix(),
        "equivalence_sha256": source_seal_hash,
        "inherited_verified_checkpoints": clone(projection),
        "attempt_parameterization_verified": True,
    }

    legacy_event = {
        "event": "program_initialized",
        "attempt": "v001",
        "created_unix_ns": 3,
    }
    legacy_prefix = compact(legacy_event) + b"\n"
    legacy_sha256 = hashlib.sha256(legacy_prefix).hexdigest()
    legacy_head = f"legacy:{legacy_sha256}"
    first_unhashed = {
        "event": "zero_confirmation_outcome_version_forward",
        "attempt": "v002",
        **clone(first_edge),
        "created_unix_ns": 5,
        "version_forward_receipt_context_sha256": "a" * 64,
        "seq": 2,
        "prev_sha256": legacy_head,
    }
    first_record_hash = object_hash(first_unhashed)
    first_record = {**first_unhashed, "record_sha256": first_record_hash}
    fit_unhashed = {
        "event": "zero_confirmation_outcome_version_forward",
        "attempt": "v003",
        **clone(fit_edge),
        "created_unix_ns": 7,
        "version_forward_receipt_context_sha256": "b" * 64,
        "seq": 3,
        "prev_sha256": first_record_hash,
    }
    fit_record_hash = object_hash(fit_unhashed)
    fit_record = {**fit_unhashed, "record_sha256": fit_record_hash}
    intermediate_unhashed = {
        "event": "zero_confirmation_outcome_version_forward",
        "attempt": "v004",
        **clone(intermediate_edge),
        "created_unix_ns": 9,
        "version_forward_receipt_context_sha256": "c" * 64,
        "seq": 4,
        "prev_sha256": fit_record_hash,
    }
    intermediate_record_hash = object_hash(intermediate_unhashed)
    intermediate_record = {
        **intermediate_unhashed,
        "record_sha256": intermediate_record_hash,
    }
    source_unhashed = {
        "event": "zero_confirmation_outcome_version_forward",
        "attempt": "v005",
        **clone(source_edge),
        "created_unix_ns": 11,
        "version_forward_receipt_context_sha256": "d" * 64,
        "seq": 5,
        "prev_sha256": intermediate_record_hash,
    }
    source_record_hash = object_hash(source_unhashed)
    source_record = {**source_unhashed, "record_sha256": source_record_hash}
    pre_ledger = (
        legacy_prefix + compact(first_record) + b"\n"
        + compact(fit_record) + b"\n"
        + compact(intermediate_record) + b"\n"
        + compact(source_record) + b"\n"
    )
    pre_head = source_record_hash
    genesis_path = study / "LEDGER_CHAIN_GENESIS.json"
    genesis = {
        "schema_version": 1,
        "legacy_prefix_bytes": len(legacy_prefix),
        "legacy_prefix_event_count": 1,
        "legacy_prefix_sha256": legacy_sha256,
    }
    genesis_path.write_bytes(pretty(genesis))
    genesis_payload = genesis_path.read_bytes()
    genesis_record = {
        "path": f"{study_relative}/LEDGER_CHAIN_GENESIS.json",
        "sha256": hashlib.sha256(genesis_payload).hexdigest(),
        "bytes": len(genesis_payload),
    }
    zero = {
        "fit_outcome_episodes": 0,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    pre_without_marker = {
        "schema_version": 1,
        "active_attempt": "v005",
        "active_attempt_path": v005_relative,
        "current_state": "FIT_COHORTS",
        "completed_states": list(
            iv.STATE_MACHINE[: iv.STATE_MACHINE.index("FIT_COHORTS")]
        ),
        "next_action": "materialize the fixed fit role only",
        "process_valid": None,
        "terminal_label": None,
        "confirmation_terminal": False,
        "early_scientific_failure": None,
        "created_unix_ns": 1,
        "updated_unix_ns": 11,
        "ledger_event_count": 5,
        "ledger_head_sha256": pre_head,
        "attempt_history": [
            {
                "version": "v001",
                "path": v001_relative,
                "status": "invalid_zero_confirmation_outcome_procedural",
                "created_unix_ns": 1,
                "invalidity_evidence_path": first_edge["invalidity_path"],
                "invalidity_evidence_sha256": science_invalidity_hash,
            },
            {
                "version": "v002",
                "path": v002_relative,
                "status": "invalid_zero_confirmation_outcome_procedural",
                "created_unix_ns": 5,
                "version_forward_evidence_path": first_edge["equivalence_path"],
                "version_forward_evidence_sha256": prior_seal_hash,
                "invalidity_evidence_path": fit_edge["invalidity_path"],
                "invalidity_evidence_sha256": prior_invalidity_hash,
                "attempt_parameterization_verified": True,
            },
            {
                "version": "v003",
                "path": v003_relative,
                "status": "invalid_zero_confirmation_outcome_procedural",
                "created_unix_ns": 7,
                "version_forward_evidence_path": fit_edge["equivalence_path"],
                "version_forward_evidence_sha256": fit_seal_hash,
                "invalidity_evidence_path": intermediate_edge["invalidity_path"],
                "invalidity_evidence_sha256": fit_invalidity_hash,
                "attempt_parameterization_verified": True,
            },
            {
                "version": "v004",
                "path": v004_relative,
                "status": "invalid_zero_confirmation_outcome_procedural",
                "created_unix_ns": 9,
                "version_forward_evidence_path": intermediate_edge[
                    "equivalence_path"
                ],
                "version_forward_evidence_sha256": intermediate_seal_hash,
                "invalidity_evidence_path": source_edge["invalidity_path"],
                "invalidity_evidence_sha256": intermediate_invalidity_hash,
                "attempt_parameterization_verified": True,
            },
            {
                "version": "v005",
                "path": v005_relative,
                "status": "active_zero_confirmation_outcome_version_forward",
                "created_unix_ns": 11,
                "version_forward_evidence_path": source_edge["equivalence_path"],
                "version_forward_evidence_sha256": source_seal_hash,
                "attempt_parameterization_verified": True,
            },
        ],
        "version_forward_lineage": [
            clone(first_edge),
            clone(fit_edge),
            clone(intermediate_edge),
            clone(source_edge),
        ],
        "verified_checkpoints": [clone(checkpoint)],
        "last_verified_checkpoint": clone(checkpoint),
        "expected_fit_episode_count": 1200,
        "expected_selection_episode_count": 2000,
        "expected_smoke_episode_count": 24,
        "maximum_confirmation_episode_count": 18000,
        **zero,
    }
    source_adapter_record = rich_record(source_adapter, source=True)
    root_record = rich_record(root_program, source=True)
    source_marker = {
        "schema_version": 2,
        "authorization_kind": "receipt_bound_v005_root_controller_adapter",
        "adapter_source_path": source_adapter_record["path"],
        "adapter_source_sha256": source_adapter_record["sha256"],
        "adapter_source_ast_sha256": source_adapter_record["ast_sha256"],
        "root_program_path": root_record["path"],
        "root_program_sha256": root_record["sha256"],
        "root_program_ast_sha256": root_record["ast_sha256"],
        "ledger_event_count": 5,
        "ledger_head_sha256": pre_head,
        "operation_sha256": "b" * 64,
    }
    source_marker["state_binding_sha256"] = object_hash(
        {
            "marker_without_state_binding": source_marker,
            "state_without_marker": pre_without_marker,
        }
    )
    pre_state = {
        **clone(pre_without_marker),
        "v005_durable_controller_adapter": source_marker,
    }
    prior_receipt = {
        "source_attempt": "v004",
        "target_attempt": "v005",
        "passed": True,
        "post_snapshot": {
            "state": {
                "object": clone(pre_state),
                "object_sha256": object_hash(pre_state),
            }
        },
        "controller_return": clone(pre_state),
        "controller_return_sha256": object_hash(pre_state),
        "transaction_source": clone(source_adapter_record),
    }
    prior_receipt_path.write_bytes(pretty(prior_receipt))
    pre_state_snapshot = state_record(pre_state)
    pre_ledger_snapshot = {
        "path": f"{study_relative}/RESEARCH_LEDGER.jsonl",
        "sha256": hashlib.sha256(pre_ledger).hexdigest(),
        "bytes": len(pre_ledger),
        "event_count": 5,
        "head_sha256": pre_head,
        "ledger_sha256": hashlib.sha256(pre_ledger).hexdigest(),
    }
    pre_snapshot = {
        "state": pre_state_snapshot,
        "ledger": pre_ledger_snapshot,
        "genesis": clone(genesis_record),
    }

    current_edge = {
        "old_attempt": "v005",
        "new_attempt": "v008",
        "resume_state": "FIT_COHORTS",
        "invalidity_path": invalidity_path.relative_to(tmp_path).as_posix(),
        "invalidity_sha256": invalidity_hash,
        "equivalence_path": seal_path.relative_to(tmp_path).as_posix(),
        "equivalence_sha256": seal_hash,
        "inherited_verified_checkpoints": clone(projection),
        "attempt_parameterization_verified": True,
    }
    journal_created = 100
    forward_created = 101
    receipt_created = 102
    proposal_state = clone(pre_state)
    proposal_state.pop("v005_durable_controller_adapter")
    proposal_checkpoint = proposal_state["verified_checkpoints"][0]
    proposal_checkpoint["inherited_into_attempts"].append(
        {
            "attempt": "v008",
            "equivalence_evidence_path": current_edge["equivalence_path"],
            "equivalence_evidence_sha256": seal_hash,
        }
    )
    proposal_state["last_verified_checkpoint"] = clone(proposal_checkpoint)
    proposal_state["attempt_history"][4].update(
        {
            "status": "invalid_zero_confirmation_outcome_procedural",
            "invalidity_evidence_path": current_edge["invalidity_path"],
            "invalidity_evidence_sha256": invalidity_hash,
        }
    )
    proposal_state["attempt_history"].append(
        {
            "version": "v008",
            "path": v008_relative,
            "status": "active_zero_confirmation_outcome_version_forward",
            "created_unix_ns": forward_created,
            "version_forward_evidence_path": current_edge["equivalence_path"],
            "version_forward_evidence_sha256": seal_hash,
            "attempt_parameterization_verified": True,
        }
    )
    proposal_state["active_attempt"] = "v008"
    proposal_state["active_attempt_path"] = v008_relative
    proposal_state["version_forward_lineage"] = [
        clone(first_edge),
        clone(fit_edge),
        clone(intermediate_edge),
        clone(source_edge),
        clone(current_edge),
    ]
    proposal_state["updated_unix_ns"] = forward_created
    proposal_event = {
        "event": "zero_confirmation_outcome_version_forward",
        "attempt": "v008",
        **clone(current_edge),
        "source_controller_adapter_marker": clone(source_marker),
        "created_unix_ns": forward_created,
    }
    proposal = {"state": proposal_state, "events": [proposal_event]}
    if proposal_mutator is not None:
        proposal_mutator(proposal)

    def equivalence_projection(target_count: int) -> dict[str, Any]:
        return {
            "source_path_count": target_count - 1,
            "target_path_count": target_count,
            "partition_counts": {
                "exact_hash": target_count - 4,
                "normalized_ast": 1,
                "canonical_contracts": 1,
                "procedural_only": 0,
                "new_lineage_support": 1,
                "lineage_support": 1,
            },
        }

    equivalence = equivalence_projection(fresh_partition_target_count)
    stored_equivalence = equivalence_projection(stored_partition_target_count)
    standalone = {
        "passed": True,
        "phase": "pre-forward",
        "attempt": "v008",
        "active_attempt": "v008",
        "science_attempt": "v001",
        "seal_path": seal_path.relative_to(tmp_path).as_posix(),
        "seal_sha256": seal_hash,
        "partition_file_count": stored_partition_target_count,
        "sealed_file_count": stored_sealed_file_count,
        "authorized_early_verifier_state": None,
        "authorized_role_count_recovery": None,
        "outcome_arrays_opened": False,
        "output_paths_created": 0,
        "read_only": True,
    }
    pre_verifiers = {
        "producer": {
            "passed": True,
            "phase": "pre-forward",
            "attempt": "v008",
            "seal_path": seal_path.relative_to(tmp_path).as_posix(),
            "seal_sha256": seal_hash,
            "state_sha256": pre_state_snapshot["sha256"],
            "ledger_sha256": pre_ledger_snapshot["sha256"],
            "outcome_arrays_opened": False,
            "read_only": True,
        },
        "standalone": standalone,
        "independent_partitions": clone(stored_equivalence),
    }
    post_standalone = clone(standalone)
    post_standalone["phase"] = "post-forward"
    post_verifiers = {
        "standalone": post_standalone,
        "independent_partitions": clone(stored_equivalence),
    }
    fixed_inputs = {
        "seal": rich_record(seal_path),
        "invalidity": rich_record(invalidity_path),
        "invalidity_draft": rich_record(invalidity_draft_path),
        "prior_receipt": rich_record(prior_receipt_path),
        "root_program": clone(root_record),
        "transaction_source": rich_record(adapter, source=True),
    }
    context = {
        "schema_version": 1,
        "source_attempt": "v005",
        "target_attempt": "v008",
        "resume_state": "FIT_COHORTS",
        "journal_created_unix_ns": journal_created,
        "forward_created_unix_ns": forward_created,
        "receipt_created_unix_ns": receipt_created,
        "fixed_inputs": clone(fixed_inputs),
        "pre_snapshot": clone(pre_snapshot),
        "pre_verifiers": clone(pre_verifiers),
        "controller_proposal": clone(proposal),
        "predicted_post_verifiers": clone(post_verifiers),
    }
    context_hash = object_hash(context)
    record_without_hash = clone(proposal["events"][0])
    record_without_hash["version_forward_receipt_context_sha256"] = context_hash
    record_without_hash["seq"] = 6
    record_without_hash["prev_sha256"] = pre_head
    record_hash = object_hash(record_without_hash)
    forward_record = {**record_without_hash, "record_sha256": record_hash}
    suffix = compact(forward_record) + b"\n"
    post_ledger = pre_ledger + suffix
    intended_without_marker = clone(proposal["state"])
    intended_without_marker["ledger_event_count"] = 6
    intended_without_marker["ledger_head_sha256"] = record_hash
    operation_hash = object_hash(
        {
            "base_state_object_sha256": object_hash(pre_state),
            "base_ledger_sha256": hashlib.sha256(pre_ledger).hexdigest(),
            "expected_suffix_sha256": hashlib.sha256(suffix).hexdigest(),
            "intended_state_object_sha256": object_hash(intended_without_marker),
        }
    )
    marker = {
        "schema_version": 2,
        "authorization_kind": "receipt_bound_v008_root_controller_adapter",
        "adapter_source_path": fixed_inputs["transaction_source"]["path"],
        "adapter_source_sha256": fixed_inputs["transaction_source"]["sha256"],
        "adapter_source_ast_sha256": fixed_inputs["transaction_source"]["ast_sha256"],
        "root_program_path": fixed_inputs["root_program"]["path"],
        "root_program_sha256": fixed_inputs["root_program"]["sha256"],
        "root_program_ast_sha256": fixed_inputs["root_program"]["ast_sha256"],
        "ledger_event_count": 6,
        "ledger_head_sha256": record_hash,
        "operation_sha256": operation_hash,
    }
    marker["state_binding_sha256"] = object_hash(
        {
            "marker_without_state_binding": marker,
            "state_without_marker": intended_without_marker,
        }
    )
    post_state = {
        **clone(intended_without_marker),
        "v008_durable_controller_adapter": marker,
    }
    post_snapshot = {
        "state": state_record(post_state),
        "ledger": {
            "path": f"{study_relative}/RESEARCH_LEDGER.jsonl",
            "sha256": hashlib.sha256(post_ledger).hexdigest(),
            "bytes": len(post_ledger),
            "event_count": 6,
            "head_sha256": record_hash,
            "ledger_sha256": hashlib.sha256(post_ledger).hexdigest(),
        },
        "genesis": clone(genesis_record),
    }
    receipt = {
        "schema_version": 2,
        "artifact_type": "atomic_zero_outcome_version_forward_transaction_receipt",
        "authorization_kind": "locked_v005_to_v008_inherited_pre_data_forward",
        "source_attempt": "v005",
        "target_attempt": "v008",
        "resume_state": "FIT_COHORTS",
        "journal_created_unix_ns": journal_created,
        "forward_created_unix_ns": forward_created,
        "created_unix_ns": receipt_created,
        "receipt_context_sha256": context_hash,
        "receipt_context": clone(context),
        "seal": clone(fixed_inputs["seal"]),
        "invalidity": clone(fixed_inputs["invalidity"]),
        "invalidity_draft": clone(fixed_inputs["invalidity_draft"]),
        "prior_receipt": clone(fixed_inputs["prior_receipt"]),
        "root_program": clone(fixed_inputs["root_program"]),
        "transaction_source": clone(fixed_inputs["transaction_source"]),
        "pre_snapshot": clone(pre_snapshot),
        "pre_verifiers": clone(pre_verifiers),
        "controller_return": clone(post_state),
        "controller_return_sha256": object_hash(post_state),
        "post_snapshot": clone(post_snapshot),
        "post_verifiers": clone(post_verifiers),
        "outcome_counts": clone(zero),
        "state_transaction_absent": True,
        "receipt_journal_absent": True,
        "passed": True,
    }
    ledger_path = study / "RESEARCH_LEDGER.jsonl"
    state_path = study / "STATE.json"
    receipt_path = v008_audit / "version_forward_transaction_receipt.json"
    ledger_path.write_bytes(post_ledger)
    state_path.write_bytes(pretty(post_state))
    receipt_path.write_bytes(pretty(receipt))

    def write_receipt(value: dict[str, Any], *, canonical: bool = True) -> None:
        receipt_path.write_bytes(pretty(value) if canonical else compact(value))

    current_events = [
        legacy_event,
        first_record,
        fit_record,
        intermediate_record,
        source_record,
        forward_record,
    ]
    return {
        "policy": policy,
        "state": post_state,
        "events": current_events,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "study": study,
        "attempt": attempt,
        "seal_path": seal_path,
        "invalidity_path": invalidity_path,
        "invalidity_draft_path": invalidity_draft_path,
        "prior_receipt_path": prior_receipt_path,
        "root_program": root_program,
        "source_adapter": source_adapter,
        "adapter": adapter,
        "source_marker": source_marker,
        "context_hash": context_hash,
        "operation_hash": operation_hash,
        "record_hash": record_hash,
        "equivalence": equivalence,
        "active_sealed_summary": {
            "file_count": fresh_sealed_file_count,
            "total_bytes": 1,
        },
        "lineage_edge": current_edge,
        "write_receipt": write_receipt,
        "verify_kwargs": {
            "policy": policy,
            "current_state": post_state,
            "current_events": current_events,
            "expected_seal_sha256": seal_hash,
            "expected_invalidity_sha256": invalidity_hash,
            "expected_edge": current_edge,
            "expected_source_marker": source_marker,
            "expected_equivalence": equivalence,
            "expected_active_sealed_file_summary": {
                "file_count": fresh_sealed_file_count,
                "total_bytes": 1,
            },
        },
    }


def _version_forward_receipt_v2_case(
    tmp_path: Path,
    *,
    proposal_mutator: Any = None,
    fresh_sealed_file_count: int = 6475,
    fresh_partition_target_count: int = 61,
    stored_sealed_file_count: int | None = None,
    stored_partition_target_count: int | None = None,
) -> dict[str, Any]:
    """Construct a v006 -> v008 receipt from the immutable live v006 prefix."""

    if stored_sealed_file_count is None:
        stored_sealed_file_count = fresh_sealed_file_count
    if stored_partition_target_count is None:
        stored_partition_target_count = fresh_partition_target_count

    live_study = ROOT.parents[1]
    study_relative = iv.STUDY_RELATIVE
    source_relative = iv.ATTEMPT_ROOTS[iv.SOURCE_ATTEMPT]
    target_relative = iv.ATTEMPT_ROOTS[iv.ACTIVE_ATTEMPT]
    study = tmp_path / study_relative
    source = tmp_path / source_relative
    attempt = tmp_path / target_relative
    audit = attempt / "audit"
    audit.mkdir(parents=True)
    source.mkdir(parents=True)
    policy = iv.PathPolicy(tmp_path, target_relative, study_relative)

    def compact(value: Any) -> bytes:
        return json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def pretty(value: Any) -> bytes:
        return (
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

    def clone(value: Any) -> Any:
        return json.loads(json.dumps(value, allow_nan=False))

    def object_hash(value: Any) -> str:
        return hashlib.sha256(compact(value)).hexdigest()

    def ast_hash(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode("utf-8")
        ).hexdigest()

    def copy_live(relative: str) -> Path:
        source_path = live_study.parents[1] / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())
        return destination

    def rich_record(path: Path, *, ast_bound: bool = False) -> dict[str, Any]:
        metadata = path.lstat()
        payload = path.read_bytes()
        record = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "device": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "nlink": int(metadata.st_nlink),
        }
        if ast_bound:
            record["ast_sha256"] = ast_hash(path)
        return record

    def state_record(state: dict[str, Any]) -> dict[str, Any]:
        payload = pretty(state)
        return {
            "path": f"{study_relative}/STATE.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "object_sha256": object_hash(state),
            "object": clone(state),
        }

    root_program = copy_live(f"{study_relative}/program.py")
    genesis_path = copy_live(f"{study_relative}/LEDGER_CHAIN_GENESIS.json")
    source_adapter = copy_live(
        f"{source_relative}/version_forward_transaction.py"
    )
    invalidity_path = copy_live(iv.INVALIDITY_RELATIVE)
    invalidity_draft_path = copy_live(iv.INVALIDITY_DRAFT_RELATIVE)
    prior_receipt_path = copy_live(iv.SOURCE_RECEIPT_RELATIVE)
    adapter = attempt / "version_forward_transaction.py"
    adapter.write_bytes((ROOT / "version_forward_transaction.py").read_bytes())
    seal_path = audit / "pre_data_inheritance_seal.json"
    seal_path.write_bytes(
        pretty(
            {
                "artifact_type": "v008_synthetic_inheritance_seal",
                "attempt": iv.ACTIVE_ATTEMPT,
                "passed": True,
            }
        )
    )

    pre_state_payload = (live_study / "STATE.json").read_bytes()
    pre_state = json.loads(pre_state_payload)
    pre_ledger = (live_study / "RESEARCH_LEDGER.jsonl").read_bytes()
    pre_events = [json.loads(line) for line in pre_ledger.splitlines()]
    assert hashlib.sha256(pre_state_payload).hexdigest() == iv.EXPECTED_SOURCE_STATE_SHA256
    assert hashlib.sha256(pre_ledger).hexdigest() == iv.EXPECTED_SOURCE_LEDGER_SHA256
    assert hashlib.sha256(prior_receipt_path.read_bytes()).hexdigest() == iv.EXPECTED_SOURCE_RECEIPT_SHA256
    assert pre_state["active_attempt"] == iv.SOURCE_ATTEMPT
    assert pre_state["current_state"] == iv.VERSION_FORWARD_RESUME_STATE
    assert pre_state["ledger_event_count"] == len(pre_events)

    seal_hash = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    invalidity_hash = hashlib.sha256(invalidity_path.read_bytes()).hexdigest()
    source_marker = clone(pre_state["v006_durable_controller_adapter"])
    projection = [
        {
            "checkpoint_name": checkpoint["name"],
            "source_attempt": checkpoint["source_attempt"],
            "evidence_path": checkpoint["evidence_path"],
            "evidence_sha256": checkpoint["evidence_sha256"],
        }
        for checkpoint in pre_state["verified_checkpoints"]
    ]
    current_edge = {
        "old_attempt": iv.SOURCE_ATTEMPT,
        "new_attempt": iv.ACTIVE_ATTEMPT,
        "resume_state": iv.VERSION_FORWARD_RESUME_STATE,
        "invalidity_path": iv.INVALIDITY_RELATIVE,
        "invalidity_sha256": invalidity_hash,
        "equivalence_path": seal_path.relative_to(tmp_path).as_posix(),
        "equivalence_sha256": seal_hash,
        "inherited_verified_checkpoints": clone(projection),
        "attempt_parameterization_verified": True,
    }

    journal_created = int(pre_events[-1]["created_unix_ns"]) + 100
    forward_created = journal_created + 1
    receipt_created = journal_created + 2
    proposal_state = clone(pre_state)
    proposal_state.pop("v006_durable_controller_adapter")
    annotation = {
        "attempt": iv.ACTIVE_ATTEMPT,
        "equivalence_evidence_path": current_edge["equivalence_path"],
        "equivalence_evidence_sha256": seal_hash,
    }
    for checkpoint in proposal_state["verified_checkpoints"]:
        inherited = checkpoint.setdefault("inherited_into_attempts", [])
        if annotation not in inherited:
            inherited.append(clone(annotation))
    proposal_state["last_verified_checkpoint"] = clone(
        proposal_state["verified_checkpoints"][-1]
    )
    proposal_state["attempt_history"][-1].update(
        {
            "status": "invalid_zero_confirmation_outcome_procedural",
            "invalidity_evidence_path": iv.INVALIDITY_RELATIVE,
            "invalidity_evidence_sha256": invalidity_hash,
        }
    )
    proposal_state["attempt_history"].append(
        {
            "version": iv.ACTIVE_ATTEMPT,
            "path": target_relative,
            "status": "active_zero_confirmation_outcome_version_forward",
            "created_unix_ns": forward_created,
            "version_forward_evidence_path": current_edge["equivalence_path"],
            "version_forward_evidence_sha256": seal_hash,
            "attempt_parameterization_verified": True,
        }
    )
    proposal_state["active_attempt"] = iv.ACTIVE_ATTEMPT
    proposal_state["active_attempt_path"] = target_relative
    proposal_state["version_forward_lineage"] = [
        *proposal_state["version_forward_lineage"],
        clone(current_edge),
    ]
    proposal_state["updated_unix_ns"] = forward_created
    proposal_event = {
        "event": "zero_confirmation_outcome_version_forward",
        "attempt": iv.ACTIVE_ATTEMPT,
        **clone(current_edge),
        "source_controller_adapter_marker": clone(source_marker),
        "created_unix_ns": forward_created,
    }
    proposal = {"state": proposal_state, "events": [proposal_event]}
    if proposal_mutator is not None:
        proposal_mutator(proposal)

    def equivalence_projection(target_count: int) -> dict[str, Any]:
        return {
            "source_path_count": target_count,
            "target_path_count": target_count,
            "partition_counts": {
                "exact_hash": target_count - 35,
                "normalized_ast": 12,
                "canonical_contracts": 3,
                "procedural_only": 0,
                "new_lineage_support": 0,
                "lineage_support": 20,
            },
        }

    equivalence = equivalence_projection(fresh_partition_target_count)
    stored_equivalence = equivalence_projection(stored_partition_target_count)
    pre_count = len(pre_events)
    pre_head = pre_state["ledger_head_sha256"]
    pre_state_snapshot = state_record(pre_state)
    pre_ledger_snapshot = {
        "path": f"{study_relative}/RESEARCH_LEDGER.jsonl",
        "sha256": hashlib.sha256(pre_ledger).hexdigest(),
        "bytes": len(pre_ledger),
        "event_count": pre_count,
        "head_sha256": pre_head,
        "ledger_sha256": hashlib.sha256(pre_ledger).hexdigest(),
    }
    genesis_payload = genesis_path.read_bytes()
    genesis_record = {
        "path": f"{study_relative}/LEDGER_CHAIN_GENESIS.json",
        "sha256": hashlib.sha256(genesis_payload).hexdigest(),
        "bytes": len(genesis_payload),
    }
    pre_snapshot = {
        "state": pre_state_snapshot,
        "ledger": pre_ledger_snapshot,
        "genesis": clone(genesis_record),
    }
    standalone = {
        "passed": True,
        "phase": "pre-forward",
        "attempt": iv.ACTIVE_ATTEMPT,
        "active_attempt": iv.ACTIVE_ATTEMPT,
        "science_attempt": iv.SCIENCE_ATTEMPT,
        "seal_path": current_edge["equivalence_path"],
        "seal_sha256": seal_hash,
        "partition_file_count": stored_partition_target_count,
        "sealed_file_count": stored_sealed_file_count,
        "authorized_early_verifier_state": None,
        "authorized_role_count_recovery": None,
        "outcome_arrays_opened": False,
        "output_paths_created": 0,
        "read_only": True,
    }
    pre_verifiers = {
        "producer": {
            "passed": True,
            "phase": "pre-forward",
            "attempt": iv.ACTIVE_ATTEMPT,
            "seal_path": current_edge["equivalence_path"],
            "seal_sha256": seal_hash,
            "state_sha256": pre_state_snapshot["sha256"],
            "ledger_sha256": pre_ledger_snapshot["sha256"],
            "outcome_arrays_opened": False,
            "read_only": True,
        },
        "standalone": standalone,
        "independent_partitions": clone(stored_equivalence),
    }
    post_standalone = clone(standalone)
    post_standalone["phase"] = "post-forward"
    post_verifiers = {
        "standalone": post_standalone,
        "independent_partitions": clone(stored_equivalence),
    }
    fixed_inputs = {
        "seal": rich_record(seal_path),
        "invalidity": rich_record(invalidity_path),
        "invalidity_draft": rich_record(invalidity_draft_path),
        "prior_receipt": rich_record(prior_receipt_path),
        "root_program": rich_record(root_program, ast_bound=True),
        "transaction_source": rich_record(adapter, ast_bound=True),
    }
    context = {
        "schema_version": 1,
        "source_attempt": iv.SOURCE_ATTEMPT,
        "target_attempt": iv.ACTIVE_ATTEMPT,
        "resume_state": iv.VERSION_FORWARD_RESUME_STATE,
        "journal_created_unix_ns": journal_created,
        "forward_created_unix_ns": forward_created,
        "receipt_created_unix_ns": receipt_created,
        "fixed_inputs": clone(fixed_inputs),
        "pre_snapshot": clone(pre_snapshot),
        "pre_verifiers": clone(pre_verifiers),
        "controller_proposal": clone(proposal),
        "predicted_post_verifiers": clone(post_verifiers),
    }
    context_hash = object_hash(context)
    record_without_hash = clone(proposal["events"][0])
    record_without_hash["version_forward_receipt_context_sha256"] = context_hash
    record_without_hash["seq"] = pre_count + 1
    record_without_hash["prev_sha256"] = pre_head
    record_hash = object_hash(record_without_hash)
    forward_record = {**record_without_hash, "record_sha256": record_hash}
    suffix = compact(forward_record) + b"\n"
    post_ledger = pre_ledger + suffix
    intended_without_marker = clone(proposal["state"])
    intended_without_marker["ledger_event_count"] = pre_count + 1
    intended_without_marker["ledger_head_sha256"] = record_hash
    operation_hash = object_hash(
        {
            "base_state_object_sha256": object_hash(pre_state),
            "base_ledger_sha256": hashlib.sha256(pre_ledger).hexdigest(),
            "expected_suffix_sha256": hashlib.sha256(suffix).hexdigest(),
            "intended_state_object_sha256": object_hash(intended_without_marker),
        }
    )
    marker = {
        "schema_version": 2,
        "authorization_kind": "receipt_bound_v008_root_controller_adapter",
        "adapter_source_path": fixed_inputs["transaction_source"]["path"],
        "adapter_source_sha256": fixed_inputs["transaction_source"]["sha256"],
        "adapter_source_ast_sha256": fixed_inputs["transaction_source"]["ast_sha256"],
        "root_program_path": fixed_inputs["root_program"]["path"],
        "root_program_sha256": fixed_inputs["root_program"]["sha256"],
        "root_program_ast_sha256": fixed_inputs["root_program"]["ast_sha256"],
        "ledger_event_count": pre_count + 1,
        "ledger_head_sha256": record_hash,
        "operation_sha256": operation_hash,
    }
    marker["state_binding_sha256"] = object_hash(
        {
            "marker_without_state_binding": marker,
            "state_without_marker": intended_without_marker,
        }
    )
    post_state = {
        **clone(intended_without_marker),
        "v008_durable_controller_adapter": marker,
    }
    post_snapshot = {
        "state": state_record(post_state),
        "ledger": {
            "path": f"{study_relative}/RESEARCH_LEDGER.jsonl",
            "sha256": hashlib.sha256(post_ledger).hexdigest(),
            "bytes": len(post_ledger),
            "event_count": pre_count + 1,
            "head_sha256": record_hash,
            "ledger_sha256": hashlib.sha256(post_ledger).hexdigest(),
        },
        "genesis": clone(genesis_record),
    }
    outcomes = {
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    receipt = {
        "schema_version": 2,
        "artifact_type": "atomic_zero_confirmation_outcome_version_forward_transaction_receipt",
        "authorization_kind": "locked_v006_to_v008_inherited_pre_selection_forward",
        "source_attempt": iv.SOURCE_ATTEMPT,
        "target_attempt": iv.ACTIVE_ATTEMPT,
        "resume_state": iv.VERSION_FORWARD_RESUME_STATE,
        "journal_created_unix_ns": journal_created,
        "forward_created_unix_ns": forward_created,
        "created_unix_ns": receipt_created,
        "receipt_context_sha256": context_hash,
        "receipt_context": clone(context),
        "seal": clone(fixed_inputs["seal"]),
        "invalidity": clone(fixed_inputs["invalidity"]),
        "invalidity_draft": clone(fixed_inputs["invalidity_draft"]),
        "prior_receipt": clone(fixed_inputs["prior_receipt"]),
        "root_program": clone(fixed_inputs["root_program"]),
        "transaction_source": clone(fixed_inputs["transaction_source"]),
        "pre_snapshot": clone(pre_snapshot),
        "pre_verifiers": clone(pre_verifiers),
        "controller_return": clone(post_state),
        "controller_return_sha256": object_hash(post_state),
        "post_snapshot": clone(post_snapshot),
        "post_verifiers": clone(post_verifiers),
        "outcome_counts": clone(outcomes),
        "state_transaction_absent": True,
        "receipt_journal_absent": True,
        "passed": True,
    }
    ledger_path = study / "RESEARCH_LEDGER.jsonl"
    state_path = study / "STATE.json"
    receipt_path = audit / "version_forward_transaction_receipt.json"
    ledger_path.write_bytes(post_ledger)
    state_path.write_bytes(pretty(post_state))
    receipt_path.write_bytes(pretty(receipt))

    def write_receipt(value: dict[str, Any], *, canonical: bool = True) -> None:
        receipt_path.write_bytes(pretty(value) if canonical else compact(value))

    current_events = [*pre_events, forward_record]
    active_summary = {
        "file_count": fresh_sealed_file_count,
        "total_bytes": 1,
    }
    return {
        "policy": policy,
        "state": post_state,
        "events": current_events,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "study": study,
        "attempt": attempt,
        "seal_path": seal_path,
        "invalidity_path": invalidity_path,
        "invalidity_draft_path": invalidity_draft_path,
        "prior_receipt_path": prior_receipt_path,
        "root_program": root_program,
        "source_adapter": source_adapter,
        "adapter": adapter,
        "source_marker": source_marker,
        "context_hash": context_hash,
        "operation_hash": operation_hash,
        "record_hash": record_hash,
        "equivalence": equivalence,
        "active_sealed_summary": active_summary,
        "lineage_edge": current_edge,
        "write_receipt": write_receipt,
        "verify_kwargs": {
            "policy": policy,
            "current_state": post_state,
            "current_events": current_events,
            "expected_seal_sha256": seal_hash,
            "expected_invalidity_sha256": invalidity_hash,
            "expected_edge": current_edge,
            "expected_source_marker": source_marker,
            "expected_equivalence": equivalence,
            "expected_active_sealed_file_summary": active_summary,
        },
    }


def _descendant_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _descendant_compact(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _descendant_object_sha256(value: Any) -> str:
    return hashlib.sha256(_descendant_compact(value)).hexdigest()


def _append_descendant_group(
    case: dict[str, Any],
    proposed_state: dict[str, Any],
    controller_events: list[dict[str, Any]],
    *, prepared_event_mutator: Any = None, envelope_mutator: Any = None,
) -> dict[str, Any]:
    """Append one deterministic adapter group to the synthetic receipt case."""

    marker_key = "v008_durable_controller_adapter"
    base = _descendant_clone(case["verify_kwargs"]["current_state"])
    base_marker = _descendant_clone(base[marker_key])
    proposal = _descendant_clone(proposed_state)
    proposal.pop(marker_key, None)
    assert proposal["ledger_event_count"] == base["ledger_event_count"]
    assert proposal["ledger_head_sha256"] == base["ledger_head_sha256"]

    events = _descendant_clone(controller_events)
    if [event["event"] for event in events] == ["outcome_counts_updated"]:
        events[0]["prior_controller_adapter_marker"] = _descendant_clone(
            base_marker
        )
    if prepared_event_mutator is not None:
        prepared_event_mutator(events)
    projection = {
        "schema_version": 1,
        "authorization_kind": (
            "receipt_anchored_v008_descendant_adapter_transaction"
        ),
        "base_state_sha256": _descendant_object_sha256(base),
        "proposed_state_sha256": _descendant_object_sha256(proposal),
        "events": _descendant_clone(events),
    }
    transaction_id = _descendant_object_sha256(projection)
    envelope_common = {
        "schema_version": 1,
        "authorization_kind": (
            "receipt_anchored_v008_descendant_adapter_transaction"
        ),
        "transaction_id": transaction_id,
        "event_count": len(events),
        "base_state_sha256": projection["base_state_sha256"],
        "proposed_state_sha256": projection["proposed_state_sha256"],
    }
    records: list[dict[str, Any]] = []
    previous = base["ledger_head_sha256"]
    for index, event in enumerate(events):
        record = _descendant_clone(event)
        envelope = {**envelope_common, "event_index": index}
        if index == 0:
            envelope["base_state"] = _descendant_clone(base)
        if envelope_mutator is not None:
            envelope_mutator(envelope, index)
        record["v008_durable_adapter_transaction"] = envelope
        record["seq"] = base["ledger_event_count"] + index + 1
        record["prev_sha256"] = previous
        record["record_sha256"] = _descendant_object_sha256(record)
        previous = record["record_sha256"]
        records.append(record)

    suffix = b"".join(_descendant_compact(record) + b"\n" for record in records)
    ledger_path = case["study"] / "RESEARCH_LEDGER.jsonl"
    base_ledger = ledger_path.read_bytes()
    ledger_path.write_bytes(base_ledger + suffix)

    intended = _descendant_clone(proposal)
    intended["ledger_event_count"] = base["ledger_event_count"] + len(records)
    intended["ledger_head_sha256"] = previous
    operation_sha256 = _descendant_object_sha256(
        {
            "base_state_object_sha256": _descendant_object_sha256(base),
            "base_ledger_sha256": hashlib.sha256(base_ledger).hexdigest(),
            "expected_suffix_sha256": hashlib.sha256(suffix).hexdigest(),
            "intended_state_object_sha256": _descendant_object_sha256(intended),
        }
    )
    marker = _descendant_clone(base_marker)
    marker.update(
        {
            "ledger_event_count": intended["ledger_event_count"],
            "ledger_head_sha256": intended["ledger_head_sha256"],
            "operation_sha256": operation_sha256,
        }
    )
    marker.pop("state_binding_sha256", None)
    marker["state_binding_sha256"] = _descendant_object_sha256(
        {
            "marker_without_state_binding": marker,
            "state_without_marker": intended,
        }
    )
    current = {**intended, marker_key: marker}
    (case["study"] / "STATE.json").write_text(
        json.dumps(current, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    all_events = case["verify_kwargs"]["current_events"] + records
    case["verify_kwargs"]["current_state"] = current
    case["verify_kwargs"]["current_events"] = all_events
    case["state"] = current
    case["events"] = all_events
    return current


def _append_ordinary_descendant_checkpoint(
    case: dict[str, Any], *, created_unix_ns: int,
    event_mutator: Any = None, proposal_mutator: Any = None,
    envelope_mutator: Any = None,
) -> dict[str, Any]:
    base = _descendant_clone(case["verify_kwargs"]["current_state"])
    completed = base["current_state"]
    position = iv.STATE_MACHINE.index(completed)
    following = iv.STATE_MACHINE[position + 1]
    evidence = case["attempt"] / "audit" / (
        f"descendant_{created_unix_ns}_{completed.lower()}.json"
    )
    evidence_value: dict[str, Any] = {
        "attempt": iv.ACTIVE_ATTEMPT,
        "checkpoint_state": completed,
        "passed": True,
    }
    if completed == "CONFIRMATION_POWER_AND_COHORT_FREEZE":
        evidence_value.update(
            {
                "fixed_confirmation_episodes_per_regime": 500,
                "fixed_confirmation_episode_count": 2000,
            }
        )
    evidence.write_text(
        json.dumps(evidence_value, sort_keys=True) + "\n", encoding="utf-8"
    )
    relative = evidence.relative_to(case["policy"].repository_root).as_posix()
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    checkpoint = {
        "name": f"v008_fixture_{completed.lower()}_{created_unix_ns}",
        "created_unix_ns": created_unix_ns,
        "evidence_path": relative,
        "evidence_sha256": digest,
        "source_attempt": iv.ACTIVE_ATTEMPT,
        "verification_lineage": "direct_checkpoint",
    }
    proposal = _descendant_clone(base)
    proposal.pop("v008_durable_controller_adapter")
    proposal["completed_states"] = base["completed_states"] + [completed]
    proposal["current_state"] = following
    proposal["last_verified_checkpoint"] = checkpoint
    proposal["verified_checkpoints"] = base["verified_checkpoints"] + [checkpoint]
    proposal["next_action"] = f"continue fixture workflow to {following}"
    if completed == "CONFIRMATION_POWER_AND_COHORT_FREEZE":
        proposal["expected_confirmation_episodes_per_regime"] = 500
        proposal["expected_confirmation_episode_count"] = 2000
    proposal["updated_unix_ns"] = created_unix_ns
    event = {
        "event": "state_completed",
        "attempt": iv.ACTIVE_ATTEMPT,
        "completed_state": completed,
        "next_state": following,
        "checkpoint_name": checkpoint["name"],
        "evidence_path": relative,
        "evidence_sha256": digest,
        "created_unix_ns": created_unix_ns,
    }
    if event_mutator is not None:
        event_mutator(event)
    if proposal_mutator is not None:
        proposal_mutator(proposal)
    return _append_descendant_group(
        case, proposal, [event], envelope_mutator=envelope_mutator
    )


def _append_descendant_count(
    case: dict[str, Any], *, field: str, value: int | bool,
    created_unix_ns: int, prepared_event_mutator: Any = None,
) -> dict[str, Any]:
    base = _descendant_clone(case["verify_kwargs"]["current_state"])
    proposal = _descendant_clone(base)
    proposal.pop("v008_durable_controller_adapter")
    proposal[field] = value
    proposal["updated_unix_ns"] = created_unix_ns
    event = {
        "event": "outcome_counts_updated",
        "attempt": iv.ACTIVE_ATTEMPT,
        "fields": {field: value},
        "created_unix_ns": created_unix_ns,
    }
    return _append_descendant_group(
        case,
        proposal,
        [event],
        prepared_event_mutator=prepared_event_mutator,
    )


def _advance_descendant_state(
    case: dict[str, Any], target_state: str, *, first_unix_ns: int,
    complete_counts: bool = True,
) -> int:
    created = first_unix_ns
    while case["verify_kwargs"]["current_state"]["current_state"] != target_state:
        current = case["verify_kwargs"]["current_state"]
        count_at_state: dict[str, tuple[str, int | bool]] = {
            "FIT_COHORTS": ("fit_outcome_episodes", 1200),
            "SELECTION_COHORTS": ("selection_outcome_episodes", 2000),
            "EXCLUDED_MECHANICAL_SMOKE": ("smoke_outcome_episodes", 24),
            "CONFIRMATION_GENERATION": (
                "confirmation_outcome_episodes_generated", 2000
            ),
            "CONFIRMATION_EXECUTION": (
                "confirmation_outcome_episodes_executed", 2000
            ),
        }
        assignment = count_at_state.get(current["current_state"])
        if (
            complete_counts
            and assignment is not None
            and current[assignment[0]] != assignment[1]
        ):
            _append_descendant_count(
                case,
                field=assignment[0],
                value=assignment[1],
                created_unix_ns=created,
            )
            created += 1
        _append_ordinary_descendant_checkpoint(case, created_unix_ns=created)
        created += 1
    return created


def _append_no_candidate_staging(
    case: dict[str, Any], *, created_unix_ns: int,
    source_mutator: Any = None, contract_mutator: Any = None,
) -> list[str]:
    base = _descendant_clone(case["verify_kwargs"]["current_state"])
    assert base["current_state"] == "CANDIDATE_SELECTION"
    selection = case["attempt"] / "selection/selection_ledger.json"
    contract = case["attempt"] / "verifier_contract_no_candidate.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection_value = {
        "attempt": iv.ACTIVE_ATTEMPT,
        "checkpoint_state": "CANDIDATE_SELECTION",
        "passed": True,
        "status": "selection_complete_no_selected_head_refit",
        "candidate_count": 24,
        "eligible_count": 0,
        "selected_candidate": None,
        "selected_candidate_id": None,
        "selected_candidate_index": None,
        "selected_head_refit_after_selection": False,
        "prior_confirmation_outcome_episodes_used": 0,
    }
    if source_mutator is not None:
        source_mutator(selection_value)
    selection.write_text(
        json.dumps(selection_value, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    contract_value = {
        "schema_version": 1,
        "attempt": iv.ACTIVE_ATTEMPT,
        "attempt_root": iv.ATTEMPT_ROOTS[iv.ACTIVE_ATTEMPT],
        "mode": "no_candidate",
        "paths": {
            "selection_ledger": selection.relative_to(
                case["policy"].repository_root
            ).as_posix()
        },
    }
    if contract_mutator is not None:
        contract_mutator(contract_value)
    contract.write_text(
        json.dumps(contract_value, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    selection_relative = selection.relative_to(
        case["policy"].repository_root
    ).as_posix()
    contract_relative = contract.relative_to(
        case["policy"].repository_root
    ).as_posix()
    selection_sha256 = hashlib.sha256(selection.read_bytes()).hexdigest()
    contract_sha256 = hashlib.sha256(contract.read_bytes()).hexdigest()
    skipped = list(
        iv.STATE_MACHINE[
            iv.STATE_MACHINE.index("CANDIDATE_SELECTION") + 1:
            iv.STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
        ]
    )
    failure = {
        "mode": "no_candidate",
        "status": "awaiting_independent_verification",
        "trigger_state": "CANDIDATE_SELECTION",
        "trigger_evidence_path": selection_relative,
        "trigger_evidence_sha256": selection_sha256,
        "verifier_contract_path": contract_relative,
        "verifier_contract_sha256": contract_sha256,
        "skipped_states": skipped,
        "staged_unix_ns": created_unix_ns,
        "required_terminal_label": "domain_robust_gate_failed",
        "required_process_valid": True,
    }
    proposal = _descendant_clone(base)
    proposal.pop("v008_durable_controller_adapter")
    proposal["early_scientific_failure"] = failure
    proposal["skipped_states"] = [
        {
            "state": state,
            "mode": "no_candidate",
            "reason": "preregistered_process_valid_early_scientific_failure",
            "trigger_evidence_path": selection_relative,
            "trigger_evidence_sha256": selection_sha256,
            "recorded_unix_ns": created_unix_ns,
        }
        for state in skipped
    ]
    proposal["current_state"] = "INDEPENDENT_VERIFICATION"
    proposal["next_action"] = (
        "run the standalone read-only verifier in the exact staged mode and "
        "capture audit/independent_verification.json before finalizing"
    )
    proposal["updated_unix_ns"] = created_unix_ns
    event = {
        "event": "preregistered_early_scientific_failure_staged",
        "attempt": iv.ACTIVE_ATTEMPT,
        "mode": "no_candidate",
        "trigger_state": "CANDIDATE_SELECTION",
        "trigger_evidence_path": selection_relative,
        "trigger_evidence_sha256": selection_sha256,
        "verifier_contract_path": contract_relative,
        "verifier_contract_sha256": contract_sha256,
        "skipped_states": skipped,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
        "created_unix_ns": created_unix_ns,
    }
    _append_descendant_group(case, proposal, [event])
    return skipped


def _append_single_terminal_decision(
    case: dict[str, Any], *, created_unix_ns: int
) -> dict[str, Any]:
    base = _descendant_clone(case["verify_kwargs"]["current_state"])
    assert base["current_state"] == "TERMINAL"
    assert base["terminal_label"] is None
    decision = case["attempt"] / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "attempt": iv.ACTIVE_ATTEMPT,
                "terminal_label": "domain_robust_gate_partial",
                "process_valid": True,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    relative = decision.relative_to(case["policy"].repository_root).as_posix()
    digest = hashlib.sha256(decision.read_bytes()).hexdigest()
    proposal = _descendant_clone(base)
    proposal.pop("v008_durable_controller_adapter")
    proposal.update(
        {
            "terminal_label": "domain_robust_gate_partial",
            "process_valid": True,
            "scientific_terminal": True,
            "confirmation_terminal": True,
            "terminal_decision_path": relative,
            "terminal_decision_sha256": digest,
            "updated_unix_ns": created_unix_ns,
        }
    )
    event = {
        "event": "scientific_terminal_decision_recorded",
        "attempt": iv.ACTIVE_ATTEMPT,
        "terminal_label": "domain_robust_gate_partial",
        "process_valid": True,
        "decision_path": relative,
        "decision_sha256": digest,
        "created_unix_ns": created_unix_ns,
    }
    return _append_descendant_group(case, proposal, [event])


def _append_early_terminal_group(
    case: dict[str, Any], *, created_unix_ns: int,
    audit_mutator: Any = None, decision_mutator: Any = None,
    events_mutator: Any = None, envelope_mutator: Any = None,
    audit_relative_name: str = "audit/independent_verification.json",
) -> dict[str, Any]:
    base = _descendant_clone(case["verify_kwargs"]["current_state"])
    failure = base["early_scientific_failure"]
    assert base["current_state"] == "INDEPENDENT_VERIFICATION"
    assert failure["mode"] == "no_candidate"
    audit = case["attempt"] / audit_relative_name
    audit.parent.mkdir(parents=True, exist_ok=True)
    decision = case["attempt"] / "decision.json"
    contract = case["policy"].repository_root / failure["verifier_contract_path"]
    contract_relative = failure["verifier_contract_path"]
    audit_value = {
        "schema_version": 1,
        "attempt": iv.ACTIVE_ATTEMPT,
        "mode": "no_candidate",
        "passed": True,
        "terminal_label": "domain_robust_gate_failed",
        "read_only_verifier": True,
        "checks": {"fixture_reconstruction": True},
        "verifier_contract": {
            "path": contract_relative,
            "sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        },
        "source_hashes": {
            "selection_ledger": failure["trigger_evidence_sha256"]
        },
    }
    if audit_mutator is not None:
        audit_mutator(audit_value)
    audit.write_text(
        json.dumps(audit_value, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    decision_value = {
        "schema_version": 1,
        "attempt": iv.ACTIVE_ATTEMPT,
        "passed": True,
        "checkpoint_state": "TERMINAL",
        "early_failure_mode": "no_candidate",
        "terminal_label": "domain_robust_gate_failed",
        "process_valid": True,
        "trigger_evidence_path": failure["trigger_evidence_path"],
        "trigger_evidence_sha256": failure["trigger_evidence_sha256"],
        "independent_verification_path": audit.relative_to(
            case["policy"].repository_root
        ).as_posix(),
        "independent_verification_sha256": hashlib.sha256(
            audit.read_bytes()
        ).hexdigest(),
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    if decision_mutator is not None:
        decision_mutator(decision_value)
    decision.write_text(
        json.dumps(decision_value, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    audit_relative = audit.relative_to(
        case["policy"].repository_root
    ).as_posix()
    decision_relative = decision.relative_to(
        case["policy"].repository_root
    ).as_posix()
    audit_sha256 = hashlib.sha256(audit.read_bytes()).hexdigest()
    decision_sha256 = hashlib.sha256(decision.read_bytes()).hexdigest()
    audit_checkpoint = {
        "name": "v008_no_candidate_independent_verification_passed",
        "created_unix_ns": created_unix_ns,
        "evidence_path": audit_relative,
        "evidence_sha256": audit_sha256,
        "source_attempt": iv.ACTIVE_ATTEMPT,
        "verification_lineage": "direct_early_scientific_failure_checkpoint",
    }
    decision_checkpoint = {
        "name": "v008_no_candidate_terminal_decision_recorded",
        "created_unix_ns": created_unix_ns,
        "evidence_path": decision_relative,
        "evidence_sha256": decision_sha256,
        "source_attempt": iv.ACTIVE_ATTEMPT,
        "verification_lineage": "direct_early_scientific_failure_checkpoint",
    }
    proposal = _descendant_clone(base)
    proposal.pop("v008_durable_controller_adapter")
    proposal["completed_states"] = base["completed_states"] + [
        "INDEPENDENT_VERIFICATION", "TERMINAL"
    ]
    proposal["verified_checkpoints"] = base["verified_checkpoints"] + [
        audit_checkpoint, decision_checkpoint
    ]
    proposal["last_verified_checkpoint"] = decision_checkpoint
    proposal["current_state"] = "POST_TERMINAL_REPORTING"
    proposal["terminal_label"] = "domain_robust_gate_failed"
    proposal["process_valid"] = True
    proposal["scientific_terminal"] = True
    proposal["confirmation_terminal"] = False
    proposal["terminal_basis"] = "preregistered_no_candidate"
    proposal["terminal_decision_path"] = decision_relative
    proposal["terminal_decision_sha256"] = decision_sha256
    proposal["early_scientific_failure"].update(
        {
            "status": "terminal_recorded",
            "independent_verification_path": audit_relative,
            "independent_verification_sha256": audit_sha256,
            "decision_path": decision_relative,
            "decision_sha256": decision_sha256,
            "terminal_recorded_unix_ns": created_unix_ns,
        }
    )
    proposal["next_action"] = (
        "write the concise terminal report, robustness map, audit, limitations, "
        "and next project-scoped task; then checkpoint POST_TERMINAL_REPORTING"
    )
    proposal["updated_unix_ns"] = created_unix_ns
    events = [
        {
            "event": "state_completed",
            "attempt": iv.ACTIVE_ATTEMPT,
            "completed_state": "INDEPENDENT_VERIFICATION",
            "next_state": "TERMINAL",
            "checkpoint_name": audit_checkpoint["name"],
            "evidence_path": audit_relative,
            "evidence_sha256": audit_sha256,
            "early_failure_mode": "no_candidate",
            "created_unix_ns": created_unix_ns,
        },
        {
            "event": "scientific_terminal_decision_recorded",
            "attempt": iv.ACTIVE_ATTEMPT,
            "terminal_label": "domain_robust_gate_failed",
            "process_valid": True,
            "confirmation_terminal": False,
            "early_failure_mode": "no_candidate",
            "decision_path": decision_relative,
            "decision_sha256": decision_sha256,
            "created_unix_ns": created_unix_ns,
        },
        {
            "event": "state_completed",
            "attempt": iv.ACTIVE_ATTEMPT,
            "completed_state": "TERMINAL",
            "next_state": "POST_TERMINAL_REPORTING",
            "checkpoint_name": decision_checkpoint["name"],
            "evidence_path": decision_relative,
            "evidence_sha256": decision_sha256,
            "early_failure_mode": "no_candidate",
            "skipped_states": failure["skipped_states"],
            "created_unix_ns": created_unix_ns,
        },
    ]
    if events_mutator is not None:
        events_mutator(events)
    return _append_descendant_group(
        case, proposal, events, envelope_mutator=envelope_mutator
    )


def _append_analysis_integrity_staging(
    case: dict[str, Any], *, created_unix_ns: int,
    source_mutator: Any = None,
) -> dict[str, Any]:
    base = _descendant_clone(case["verify_kwargs"]["current_state"])
    assert base["current_state"] == "SEALED_ANALYSIS"
    evidence = case["attempt"] / "audit/analysis_execution_invalid.json"
    evidence_value = {
        "attempt": iv.ACTIVE_ATTEMPT,
        "checkpoint_state": "SEALED_ANALYSIS",
        "passed": False,
        "execution_invalid": True,
        "process_valid": False,
        "confirmation_outcome_episodes_generated": 2000,
        "confirmation_outcome_episodes_executed": 2000,
        "confirmation_outcomes_opened_for_analysis": True,
        "analysis_result_present": False,
        "error_type": "RuntimeError",
        "error": "fixture analysis execution invalidity",
        "scientific_objects_changed": False,
    }
    if source_mutator is not None:
        source_mutator(evidence_value)
    evidence.write_text(
        json.dumps(evidence_value, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    relative = evidence.relative_to(case["policy"].repository_root).as_posix()
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    skipped = ["LATENCY_AND_RESOURCE_REPORTING"]
    checkpoint = {
        "name": "v008_analysis_execution_integrity_failure",
        "created_unix_ns": created_unix_ns,
        "evidence_path": relative,
        "evidence_sha256": digest,
        "source_attempt": iv.ACTIVE_ATTEMPT,
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    proposal = _descendant_clone(base)
    proposal.pop("v008_durable_controller_adapter")
    proposal["completed_states"] = base["completed_states"] + ["SEALED_ANALYSIS"]
    proposal["verified_checkpoints"] = base["verified_checkpoints"] + [checkpoint]
    proposal["last_verified_checkpoint"] = checkpoint
    proposal["current_state"] = "INDEPENDENT_VERIFICATION"
    proposal["postconfirmation_integrity_failure"] = {
        "status": "awaiting_independent_verification",
        "source": "analysis_execution",
        "trigger_state": "SEALED_ANALYSIS",
        "source_path": relative,
        "source_sha256": digest,
        "audit_hash_keys": [
            "analysis_execution_invalid", "analysis_execution", "analysis_failure"
        ],
        "skipped_states": skipped,
        "staged_unix_ns": created_unix_ns,
    }
    proposal["skipped_states"] = [
        {
            "state": "LATENCY_AND_RESOURCE_REPORTING",
            "source": "analysis_execution",
            "reason": "postconfirmation_integrity_failure_short_circuit",
            "source_path": relative,
            "source_sha256": digest,
            "recorded_unix_ns": created_unix_ns,
        }
    ]
    proposal["next_action"] = (
        "run the read-only confirmation-mode independent verifier and capture "
        "the execution-invalid audit; do not retry or alter confirmation evidence"
    )
    proposal["updated_unix_ns"] = created_unix_ns
    event = {
        "event": "state_completed",
        "attempt": iv.ACTIVE_ATTEMPT,
        "completed_state": "SEALED_ANALYSIS",
        "next_state": "INDEPENDENT_VERIFICATION",
        "checkpoint_name": checkpoint["name"],
        "evidence_path": relative,
        "evidence_sha256": digest,
        "postconfirmation_integrity_failure": True,
        "integrity_source": "analysis_execution",
        "skipped_states": skipped,
        "created_unix_ns": created_unix_ns,
    }
    return _append_descendant_group(case, proposal, [event])


def _append_execution_invalid_terminal_group(
    case: dict[str, Any], *, created_unix_ns: int,
    audit_mutator: Any = None, decision_mutator: Any = None,
    events_mutator: Any = None, envelope_mutator: Any = None,
    audit_relative_name: str = "audit/independent_verification.json",
) -> dict[str, Any]:
    base = _descendant_clone(case["verify_kwargs"]["current_state"])
    assert base["current_state"] == "INDEPENDENT_VERIFICATION"
    assert isinstance(base.get("postconfirmation_integrity_failure"), dict)
    audit = case["attempt"] / audit_relative_name
    audit.parent.mkdir(parents=True, exist_ok=True)
    decision = case["attempt"] / "decision.json"
    contract = case["attempt"] / "verifier_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt": iv.ACTIVE_ATTEMPT,
                "attempt_root": iv.ATTEMPT_ROOTS[iv.ACTIVE_ATTEMPT],
                "mode": "confirmation",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    contract_relative = contract.relative_to(
        case["policy"].repository_root
    ).as_posix()
    contract_sha256 = hashlib.sha256(contract.read_bytes()).hexdigest()
    audit_value = {
        "schema_version": 1,
        "attempt": iv.ACTIVE_ATTEMPT,
        "mode": "confirmation",
        "read_only_verifier": True,
        "passed": True,
        "execution_invalid": True,
        "integrity_failure": True,
        "terminal_label": "domain_robust_gate_execution_invalid",
        "verifier_contract": {
            "path": contract_relative,
            "sha256": contract_sha256,
        },
        "capture": {
            "captured_exclusively": True,
            "wrapper_integrity_passed": True,
        },
        "source_hashes": {
            "analysis_execution": base[
                "postconfirmation_integrity_failure"
            ]["source_sha256"]
        },
    }
    if audit_mutator is not None:
        audit_mutator(audit_value)
    audit.write_text(
        json.dumps(audit_value, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    decision_value = {
        "schema_version": 1,
        "attempt": iv.ACTIVE_ATTEMPT,
        "passed": True,
        "checkpoint_state": "TERMINAL",
        "terminal_basis": "postconfirmation_integrity_failure",
        "terminal_label": "domain_robust_gate_execution_invalid",
        "process_valid": False,
        "independent_verification_path": audit.relative_to(
            case["policy"].repository_root
        ).as_posix(),
        "independent_verification_sha256": hashlib.sha256(
            audit.read_bytes()
        ).hexdigest(),
        "confirmation_outcome_episodes_generated": 2000,
        "confirmation_outcome_episodes_executed": 2000,
        "confirmation_outcomes_opened_for_analysis": True,
    }
    if decision_mutator is not None:
        decision_mutator(decision_value)
    decision.write_text(
        json.dumps(decision_value, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    audit_relative = audit.relative_to(
        case["policy"].repository_root
    ).as_posix()
    decision_relative = decision.relative_to(
        case["policy"].repository_root
    ).as_posix()
    audit_sha256 = hashlib.sha256(audit.read_bytes()).hexdigest()
    decision_sha256 = hashlib.sha256(decision.read_bytes()).hexdigest()
    audit_checkpoint = {
        "name": "v008_execution_invalid_independent_audit",
        "created_unix_ns": created_unix_ns,
        "evidence_path": audit_relative,
        "evidence_sha256": audit_sha256,
        "source_attempt": iv.ACTIVE_ATTEMPT,
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    decision_checkpoint = {
        "name": "v008_execution_invalid_terminal_decision",
        "created_unix_ns": created_unix_ns,
        "evidence_path": decision_relative,
        "evidence_sha256": decision_sha256,
        "source_attempt": iv.ACTIVE_ATTEMPT,
        "verification_lineage": "direct_postconfirmation_integrity_checkpoint",
    }
    proposal = _descendant_clone(base)
    proposal.pop("v008_durable_controller_adapter")
    proposal["completed_states"] = base["completed_states"] + [
        "INDEPENDENT_VERIFICATION", "TERMINAL"
    ]
    proposal["verified_checkpoints"] = base["verified_checkpoints"] + [
        audit_checkpoint, decision_checkpoint
    ]
    proposal["last_verified_checkpoint"] = decision_checkpoint
    proposal["current_state"] = "POST_TERMINAL_REPORTING"
    proposal["terminal_label"] = "domain_robust_gate_execution_invalid"
    proposal["process_valid"] = False
    proposal["scientific_terminal"] = True
    proposal["confirmation_terminal"] = True
    proposal["terminal_basis"] = "postconfirmation_integrity_failure"
    proposal["terminal_decision_path"] = decision_relative
    proposal["terminal_decision_sha256"] = decision_sha256
    proposal["postconfirmation_integrity_failure"].update(
        {
            "status": "terminal_recorded",
            "independent_verification_path": audit_relative,
            "independent_verification_sha256": audit_sha256,
            "decision_path": decision_relative,
            "decision_sha256": decision_sha256,
            "terminal_recorded_unix_ns": created_unix_ns,
        }
    )
    proposal["next_action"] = (
        "report the immutable execution-invalid result and integrity diagnosis; "
        "do not retry or reinterpret it as a scientific partial/failure"
    )
    proposal["updated_unix_ns"] = created_unix_ns
    events = [
        {
            "event": "state_completed",
            "attempt": iv.ACTIVE_ATTEMPT,
            "completed_state": "INDEPENDENT_VERIFICATION",
            "next_state": "TERMINAL",
            "checkpoint_name": audit_checkpoint["name"],
            "evidence_path": audit_relative,
            "evidence_sha256": audit_sha256,
            "postconfirmation_integrity_failure": True,
            "created_unix_ns": created_unix_ns,
        },
        {
            "event": "scientific_terminal_decision_recorded",
            "attempt": iv.ACTIVE_ATTEMPT,
            "terminal_label": "domain_robust_gate_execution_invalid",
            "process_valid": False,
            "decision_path": decision_relative,
            "decision_sha256": decision_sha256,
            "created_unix_ns": created_unix_ns,
        },
        {
            "event": "state_completed",
            "attempt": iv.ACTIVE_ATTEMPT,
            "completed_state": "TERMINAL",
            "next_state": "POST_TERMINAL_REPORTING",
            "checkpoint_name": decision_checkpoint["name"],
            "evidence_path": decision_relative,
            "evidence_sha256": decision_sha256,
            "postconfirmation_integrity_failure": True,
            "created_unix_ns": created_unix_ns,
        },
    ]
    if events_mutator is not None:
        events_mutator(events)
    return _append_descendant_group(
        case, proposal, events, envelope_mutator=envelope_mutator
    )



def test_independent_version_forward_receipt_v2_recomputes_closed_transaction(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)

    evidence = iv._verify_version_forward_transaction_receipt(
        **case["verify_kwargs"]
    )

    assert evidence["passed"] is True
    assert evidence["schema_version"] == 2
    assert evidence["receipt_context_sha256"] == case["context_hash"]
    assert evidence["operation_sha256"] == case["operation_hash"]
    assert evidence["forward_record_sha256"] == case["record_hash"]
    assert evidence["pre_event_count"] == 16
    assert evidence["post_event_count"] == 17
    assert evidence["journal_resolved"] is True
    assert evidence["receipt_nlink"] == 1
    assert evidence["active_sealed_file_count"] == 6475
    assert evidence["partition_target_path_count"] == 61


def test_independent_version_forward_receipt_uses_fresh_dynamic_projections(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(
        tmp_path,
        fresh_sealed_file_count=127,
        fresh_partition_target_count=59,
    )

    evidence = iv._verify_version_forward_transaction_receipt(
        **case["verify_kwargs"]
    )

    assert evidence["active_sealed_file_count"] == 127
    assert evidence["partition_target_path_count"] == 59


def test_independent_version_forward_receipt_replays_descendant_transaction(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    clone = lambda value: json.loads(json.dumps(value))
    compact = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    object_hash = lambda value: hashlib.sha256(compact(value)).hexdigest()
    base = clone(case["state"])
    event = {
        "event": "outcome_counts_updated",
        "attempt": "v008",
        "fields": {"selection_outcome_episodes": 1},
        "prior_controller_adapter_marker": clone(
            base["v008_durable_controller_adapter"]
        ),
        "created_unix_ns": 500,
    }
    proposal = clone(base)
    proposal.pop("v008_durable_controller_adapter")
    proposal["selection_outcome_episodes"] = 1
    proposal["updated_unix_ns"] = 500
    proposal_hash = object_hash(proposal)
    transaction_projection = {
        "schema_version": 1,
        "authorization_kind": (
            "receipt_anchored_v008_descendant_adapter_transaction"
        ),
        "base_state_sha256": object_hash(base),
        "proposed_state_sha256": proposal_hash,
        "events": [clone(event)],
    }
    event["v008_durable_adapter_transaction"] = {
        "schema_version": 1,
        "authorization_kind": (
            "receipt_anchored_v008_descendant_adapter_transaction"
        ),
        "transaction_id": object_hash(transaction_projection),
        "event_index": 0,
        "event_count": 1,
        "base_state_sha256": object_hash(base),
        "proposed_state_sha256": proposal_hash,
        "base_state": clone(base),
    }
    event["seq"] = base["ledger_event_count"] + 1
    event["prev_sha256"] = base["ledger_head_sha256"]
    event["record_sha256"] = object_hash(event)
    suffix = compact(event) + b"\n"
    ledger_path = case["study"] / "RESEARCH_LEDGER.jsonl"
    base_ledger = ledger_path.read_bytes()
    ledger_path.write_bytes(base_ledger + suffix)
    intended = clone(proposal)
    intended["ledger_event_count"] = base["ledger_event_count"] + 1
    intended["ledger_head_sha256"] = event["record_sha256"]
    marker = clone(base["v008_durable_controller_adapter"])
    marker["ledger_event_count"] = intended["ledger_event_count"]
    marker["ledger_head_sha256"] = event["record_sha256"]
    marker["operation_sha256"] = object_hash(
        {
            "base_state_object_sha256": object_hash(base),
            "base_ledger_sha256": hashlib.sha256(base_ledger).hexdigest(),
            "expected_suffix_sha256": hashlib.sha256(suffix).hexdigest(),
            "intended_state_object_sha256": object_hash(intended),
        }
    )
    marker_without_binding = clone(marker)
    marker_without_binding.pop("state_binding_sha256", None)
    marker["state_binding_sha256"] = object_hash(
        {
            "marker_without_state_binding": marker_without_binding,
            "state_without_marker": intended,
        }
    )
    current = {**intended, "v008_durable_controller_adapter": marker}
    state_path = case["study"] / "STATE.json"
    state_path.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    case["verify_kwargs"]["current_state"] = current
    case["verify_kwargs"]["current_events"] = case["events"] + [event]

    evidence = iv._verify_version_forward_transaction_receipt(
        **case["verify_kwargs"]
    )

    assert evidence["descendant_adapter_chain"] == {
        "group_count": 1,
        "event_count": 1,
        "passed": True,
    }


def test_independent_descendant_ordinary_checkpoint_reconstructs_exact_state(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    current = _append_ordinary_descendant_checkpoint(
        case, created_unix_ns=500
    )

    evidence = iv._verify_version_forward_transaction_receipt(
        **case["verify_kwargs"]
    )

    assert evidence["descendant_adapter_chain"] == {
        "group_count": 1,
        "event_count": 1,
        "passed": True,
    }
    assert current["current_state"] == "CANDIDATE_SELECTION"
    assert current["completed_states"][-1] == "SELECTION_COHORTS"
    assert current["last_verified_checkpoint"] == current["verified_checkpoints"][-1]


def test_independent_descendant_early_failure_staging_reconstructs_exact_state(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case, "CANDIDATE_SELECTION", first_unix_ns=500
    )
    skipped = _append_no_candidate_staging(
        case, created_unix_ns=created
    )

    evidence = iv._verify_version_forward_transaction_receipt(
        **case["verify_kwargs"]
    )

    assert evidence["descendant_adapter_chain"] == {
        "group_count": 3,
        "event_count": 3,
        "passed": True,
    }
    current = case["verify_kwargs"]["current_state"]
    assert current["current_state"] == "INDEPENDENT_VERIFICATION"
    assert current["early_scientific_failure"]["skipped_states"] == skipped


@pytest.mark.parametrize(
    "defect",
    ["source_eligibility", "role_counts", "contract_link", "later_artifact"],
)
def test_independent_descendant_early_failure_rejects_ineligible_provenance(
    tmp_path: Path, defect: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case,
        "CANDIDATE_SELECTION",
        first_unix_ns=500,
        complete_counts=defect != "role_counts",
    )

    def mutate_source(source: dict[str, Any]) -> None:
        if defect == "source_eligibility":
            source["eligible_count"] = 1

    def mutate_contract(contract: dict[str, Any]) -> None:
        if defect == "contract_link":
            contract["paths"]["selection_ledger"] = (
                "runs/lewm_domain_robust_gate/attempts/v008/selection/other.json"
            )

    if defect == "later_artifact":
        forbidden = case["attempt"] / "data/smoke/unregistered.json"
        forbidden.parent.mkdir(parents=True)
        forbidden.write_text("{}\n", encoding="utf-8")
    _append_no_candidate_staging(
        case,
        created_unix_ns=created,
        source_mutator=mutate_source,
        contract_mutator=mutate_contract,
    )

    with pytest.raises(
        iv.VerificationError,
        match="early-failure (source eligibility|role completeness|evidence|later-role)",
    ):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


def test_independent_descendant_single_terminal_decision_reconstructs_exact_state(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case, "TERMINAL", first_unix_ns=500
    )
    current = _append_single_terminal_decision(
        case, created_unix_ns=created
    )

    evidence = iv._verify_version_forward_transaction_receipt(
        **case["verify_kwargs"]
    )

    assert evidence["descendant_adapter_chain"] == {
        "group_count": 17,
        "event_count": 17,
        "passed": True,
    }
    assert current["terminal_label"] == "domain_robust_gate_partial"
    assert current["terminal_decision_path"].endswith("/decision.json")


def test_independent_descendant_early_three_event_terminal_reconstructs_exact_state(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case, "CANDIDATE_SELECTION", first_unix_ns=500
    )
    _append_no_candidate_staging(case, created_unix_ns=created)
    current = _append_early_terminal_group(
        case, created_unix_ns=created + 1
    )

    evidence = iv._verify_version_forward_transaction_receipt(
        **case["verify_kwargs"]
    )

    assert evidence["descendant_adapter_chain"] == {
        "group_count": 4,
        "event_count": 6,
        "passed": True,
    }
    assert current["terminal_label"] == "domain_robust_gate_failed"
    assert current["confirmation_terminal"] is False
    assert current["early_scientific_failure"]["status"] == "terminal_recorded"


def test_independent_descendant_execution_invalid_terminal_reconstructs_exact_state(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case, "SEALED_ANALYSIS", first_unix_ns=500
    )
    _append_descendant_count(
        case,
        field="confirmation_outcomes_opened_for_analysis",
        value=True,
        created_unix_ns=created,
    )
    _append_analysis_integrity_staging(
        case, created_unix_ns=created + 1
    )
    current = _append_execution_invalid_terminal_group(
        case, created_unix_ns=created + 2
    )

    evidence = iv._verify_version_forward_transaction_receipt(
        **case["verify_kwargs"]
    )

    assert evidence["descendant_adapter_chain"] == {
        "group_count": 16,
        "event_count": 18,
        "passed": True,
    }
    assert current["terminal_label"] == "domain_robust_gate_execution_invalid"
    assert current["process_valid"] is False
    assert current["postconfirmation_integrity_failure"]["status"] == (
        "terminal_recorded"
    )


@pytest.mark.parametrize(
    "defect",
    ["generated", "opened_type", "failure_proof", "result_presence", "science"],
)
def test_independent_descendant_analysis_integrity_rejects_invalid_source_proof(
    tmp_path: Path, defect: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case, "SEALED_ANALYSIS", first_unix_ns=500
    )
    _append_descendant_count(
        case,
        field="confirmation_outcomes_opened_for_analysis",
        value=True,
        created_unix_ns=created,
    )

    def mutate_source(source: dict[str, Any]) -> None:
        if defect == "generated":
            source["confirmation_outcome_episodes_generated"] = 1999
        elif defect == "opened_type":
            source["confirmation_outcomes_opened_for_analysis"] = 1
        elif defect == "failure_proof":
            source["execution_invalid"] = False
            source["process_valid"] = True
        elif defect == "result_presence":
            source["analysis_result_present"] = True
        else:
            source["scientific_objects_changed"] = True

    _append_analysis_integrity_staging(
        case,
        created_unix_ns=created + 1,
        source_mutator=mutate_source,
    )

    with pytest.raises(
        iv.VerificationError,
        match="integrity fixed-open proof|analysis-invalid evidence",
    ):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    "defect",
    ["audit_path", "audit_source", "decision_counts", "timestamp"],
)
def test_independent_descendant_early_terminal_rejects_invalid_provenance(
    tmp_path: Path, defect: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case, "CANDIDATE_SELECTION", first_unix_ns=500
    )
    _append_no_candidate_staging(case, created_unix_ns=created)

    def mutate_audit(audit: dict[str, Any]) -> None:
        if defect == "audit_source":
            audit["source_hashes"]["selection_ledger"] = "0" * 64

    def mutate_decision(decision: dict[str, Any]) -> None:
        if defect == "decision_counts":
            decision["confirmation_outcomes_opened_for_analysis"] = True

    def mutate_events(events: list[dict[str, Any]]) -> None:
        if defect == "timestamp":
            events[1]["created_unix_ns"] += 1

    _append_early_terminal_group(
        case,
        created_unix_ns=created + 1,
        audit_mutator=mutate_audit,
        decision_mutator=mutate_decision,
        events_mutator=mutate_events,
        audit_relative_name=(
            "audit/other.json"
            if defect == "audit_path"
            else "audit/independent_verification.json"
        ),
    )

    with pytest.raises(
        iv.VerificationError, match="three-event|early terminal|timestamp/type"
    ):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    "defect",
    ["audit_path", "audit_source", "decision_count", "timestamp"],
)
def test_independent_descendant_execution_terminal_rejects_invalid_provenance(
    tmp_path: Path, defect: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case, "SEALED_ANALYSIS", first_unix_ns=500
    )
    _append_descendant_count(
        case,
        field="confirmation_outcomes_opened_for_analysis",
        value=True,
        created_unix_ns=created,
    )
    _append_analysis_integrity_staging(
        case, created_unix_ns=created + 1
    )

    def mutate_audit(audit: dict[str, Any]) -> None:
        if defect == "audit_source":
            audit["source_hashes"]["analysis_execution"] = "0" * 64

    def mutate_decision(decision: dict[str, Any]) -> None:
        if defect == "decision_count":
            decision["confirmation_outcome_episodes_executed"] = 1999

    def mutate_events(events: list[dict[str, Any]]) -> None:
        if defect == "timestamp":
            events[2]["created_unix_ns"] += 1

    _append_execution_invalid_terminal_group(
        case,
        created_unix_ns=created + 2,
        audit_mutator=mutate_audit,
        decision_mutator=mutate_decision,
        events_mutator=mutate_events,
        audit_relative_name=(
            "audit/other.json"
            if defect == "audit_path"
            else "audit/independent_verification.json"
        ),
    )

    with pytest.raises(
        iv.VerificationError,
        match="three-event|execution-invalid",
    ):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize("mutation", ["extra_event_key", "float_timestamp"])
def test_independent_descendant_ordinary_checkpoint_rejects_schema_or_type_drift(
    tmp_path: Path, mutation: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)

    def mutate_event(event: dict[str, Any]) -> None:
        if mutation == "extra_event_key":
            event["unexpected"] = "not_declared"
        else:
            event["created_unix_ns"] = 500.0

    def mutate_proposal(proposal: dict[str, Any]) -> None:
        if mutation == "float_timestamp":
            proposal["updated_unix_ns"] = 500.0

    _append_ordinary_descendant_checkpoint(
        case,
        created_unix_ns=500,
        event_mutator=mutate_event,
        proposal_mutator=mutate_proposal,
    )

    with pytest.raises(
        iv.VerificationError, match="descendant|timestamp/type"
    ):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


def test_independent_descendant_ordinary_checkpoint_rehashes_evidence(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    current = _append_ordinary_descendant_checkpoint(
        case, created_unix_ns=500
    )
    evidence_path = (
        case["policy"].repository_root
        / current["last_verified_checkpoint"]["evidence_path"]
    )
    evidence_path.write_text(
        json.dumps(
            {
                "attempt": iv.ACTIVE_ATTEMPT,
                "checkpoint_state": "FIT_COHORTS",
                "passed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(iv.VerificationError, match="evidence drift"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


def test_independent_descendant_count_rejects_typed_prior_marker_alias(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)

    def mutate_marker(events: list[dict[str, Any]]) -> None:
        events[0]["prior_controller_adapter_marker"][
            "ledger_event_count"
        ] = 3.0

    _append_descendant_count(
        case,
        field="fit_outcome_episodes",
        value=1,
        created_unix_ns=500,
        prepared_event_mutator=mutate_marker,
    )

    with pytest.raises(iv.VerificationError, match="count event schema drift"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    "defect", ["index", "count", "missing_base", "transaction_id"]
)
def test_independent_descendant_singleton_rejects_envelope_provenance_drift(
    tmp_path: Path, defect: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)

    def mutate_envelope(envelope: dict[str, Any], index: int) -> None:
        assert index == 0
        if defect == "index":
            envelope["event_index"] = 1
        elif defect == "count":
            envelope["event_count"] = 2
        elif defect == "missing_base":
            envelope.pop("base_state")
        else:
            envelope["transaction_id"] = "0" * 64

    _append_ordinary_descendant_checkpoint(
        case,
        created_unix_ns=500,
        envelope_mutator=mutate_envelope,
    )

    with pytest.raises(iv.VerificationError, match="envelope|transaction|group"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    "defect",
    [
        "later_base", "later_transaction", "later_index", "later_count",
        "under_size", "over_size",
    ],
)
def test_independent_descendant_terminal_rejects_grouping_provenance_drift(
    tmp_path: Path, defect: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    created = _advance_descendant_state(
        case, "CANDIDATE_SELECTION", first_unix_ns=500
    )
    _append_no_candidate_staging(case, created_unix_ns=created)
    base = _descendant_clone(case["verify_kwargs"]["current_state"])

    def mutate_events(events: list[dict[str, Any]]) -> None:
        if defect == "under_size":
            events.pop()
        elif defect == "over_size":
            events.append(_descendant_clone(events[-1]))

    def mutate_envelope(envelope: dict[str, Any], index: int) -> None:
        if defect == "later_base" and index == 1:
            envelope["base_state"] = _descendant_clone(base)
        elif defect == "later_transaction" and index == 1:
            envelope["transaction_id"] = "0" * 64
        elif defect == "later_index" and index == 2:
            envelope["event_index"] = 1
        elif defect == "later_count" and index == 2:
            envelope["event_count"] -= 1

    _append_early_terminal_group(
        case,
        created_unix_ns=created + 1,
        events_mutator=mutate_events,
        envelope_mutator=mutate_envelope,
    )

    with pytest.raises(
        iv.VerificationError, match="envelope|event sequence|group|transaction"
    ):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize("duplicate", ["supplied_state", "controller_return"])
def test_independent_receipt_rejects_typed_duplicate_state_alias(
    tmp_path: Path, duplicate: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    if duplicate == "supplied_state":
        supplied = _descendant_clone(case["verify_kwargs"]["current_state"])
        supplied["fit_outcome_episodes"] = 0.0
        case["verify_kwargs"]["current_state"] = supplied
    else:
        case["receipt"]["controller_return"]["fit_outcome_episodes"] = 0.0
        case["write_receipt"](case["receipt"])

    with pytest.raises(iv.VerificationError, match="STATE|controller return"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])





@pytest.mark.parametrize(
    "fixture_kwargs",
    [
        {"stored_sealed_file_count": 122},
        {"stored_partition_target_count": 58},
    ],
)
def test_independent_version_forward_receipt_rejects_coherent_stored_projection_drift(
    tmp_path: Path, fixture_kwargs: dict[str, int]
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path, **fixture_kwargs)

    with pytest.raises(iv.VerificationError, match="verifier|equivalence"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    "tamper",
    [
        "schema_v1",
        "top_extra",
        "top_missing",
        "context_extra",
        "context_missing",
        "snapshot_extra",
        "verifier_extra",
        "bool_timestamp",
        "bool_ledger_count",
        "partition_extra",
        "noncanonical_receipt",
    ],
)
def test_independent_version_forward_receipt_rejects_closed_schema_and_type_drift(
    tmp_path: Path, tamper: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    receipt = case["receipt"]
    canonical = True
    if tamper == "schema_v1":
        receipt["schema_version"] = 1
    elif tamper == "top_extra":
        receipt["unexpected"] = None
    elif tamper == "top_missing":
        receipt.pop("passed")
    elif tamper == "context_extra":
        receipt["receipt_context"]["unexpected"] = None
    elif tamper == "context_missing":
        receipt["receipt_context"].pop("controller_proposal")
    elif tamper == "snapshot_extra":
        receipt["pre_snapshot"]["ledger"]["unexpected"] = None
    elif tamper == "verifier_extra":
        receipt["pre_verifiers"]["standalone"]["unexpected"] = None
    elif tamper == "bool_timestamp":
        receipt["journal_created_unix_ns"] = True
    elif tamper == "bool_ledger_count":
        receipt["pre_snapshot"]["ledger"]["event_count"] = True
    elif tamper == "partition_extra":
        receipt["pre_verifiers"]["independent_partitions"][
            "partition_counts"
        ]["unexpected"] = 0
    elif tamper == "noncanonical_receipt":
        canonical = False
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(tamper)
    case["write_receipt"](receipt, canonical=canonical)

    with pytest.raises(iv.VerificationError):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    "tamper",
    [
        "journal_nonpositive",
        "forward_gap",
        "receipt_gap",
        "proposal_event_timestamp",
        "context_projection",
        "context_digest",
        "controller_digest",
    ],
)
def test_independent_version_forward_receipt_rejects_timestamp_context_and_digest_tamper(
    tmp_path: Path, tamper: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    receipt = case["receipt"]

    def canonical_hash(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    if tamper == "journal_nonpositive":
        receipt["journal_created_unix_ns"] = 0
    elif tamper == "forward_gap":
        receipt["forward_created_unix_ns"] += 1
    elif tamper == "receipt_gap":
        receipt["created_unix_ns"] += 1
    elif tamper == "proposal_event_timestamp":
        receipt["receipt_context"]["controller_proposal"]["events"][0][
            "created_unix_ns"
        ] += 1
        receipt["receipt_context_sha256"] = canonical_hash(
            receipt["receipt_context"]
        )
    elif tamper == "context_projection":
        receipt["receipt_context"]["predicted_post_verifiers"]["standalone"][
            "phase"
        ] = "pre-forward"
        receipt["receipt_context_sha256"] = canonical_hash(
            receipt["receipt_context"]
        )
    elif tamper == "context_digest":
        receipt["receipt_context_sha256"] = "f" * 64
    elif tamper == "controller_digest":
        receipt["controller_return_sha256"] = "f" * 64
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(tamper)
    case["write_receipt"](receipt)

    with pytest.raises(iv.VerificationError):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    ("record_name", "field", "tampered"),
    [
        ("seal", "path", "runs/study/attempts/v001/audit/wrong.json"),
        ("invalidity", "sha256", "f" * 64),
        ("root_program", "bytes", 1),
        ("root_program", "device", 0),
        ("root_program", "inode", 1),
        ("root_program", "nlink", 2),
        ("transaction_source", "ast_sha256", "f" * 64),
    ],
)
def test_independent_version_forward_receipt_rejects_rehashed_live_identity_tamper(
    tmp_path: Path, record_name: str, field: str, tampered: Any
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    receipt = case["receipt"]
    receipt[record_name][field] = tampered
    receipt["receipt_context"]["fixed_inputs"][record_name] = json.loads(
        json.dumps(receipt[record_name])
    )
    receipt["receipt_context_sha256"] = hashlib.sha256(
        json.dumps(
            receipt["receipt_context"],
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    case["write_receipt"](receipt)

    with pytest.raises(iv.VerificationError, match="identity|metadata|path|hash|link"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


def test_independent_version_forward_receipt_rejects_live_source_drift(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    case["adapter"].write_text("ADAPTER_VERSION = 99\n", encoding="utf-8")

    with pytest.raises(iv.VerificationError, match="live identity"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    "tamper",
    [
        "pre_ledger_sha256",
        "pre_ledger_bytes",
        "pre_ledger_head",
        "post_ledger_sha256",
        "post_ledger_head",
        "post_state_object_sha256",
    ],
)
def test_independent_version_forward_receipt_rejects_ledger_and_state_digest_tamper(
    tmp_path: Path, tamper: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    receipt = case["receipt"]
    if tamper == "pre_ledger_sha256":
        receipt["pre_snapshot"]["ledger"]["sha256"] = "f" * 64
        receipt["pre_snapshot"]["ledger"]["ledger_sha256"] = "f" * 64
    elif tamper == "pre_ledger_bytes":
        receipt["pre_snapshot"]["ledger"]["bytes"] += 1
    elif tamper == "pre_ledger_head":
        receipt["pre_snapshot"]["ledger"]["head_sha256"] = "f" * 64
    elif tamper == "post_ledger_sha256":
        receipt["post_snapshot"]["ledger"]["sha256"] = "f" * 64
        receipt["post_snapshot"]["ledger"]["ledger_sha256"] = "f" * 64
    elif tamper == "post_ledger_head":
        receipt["post_snapshot"]["ledger"]["head_sha256"] = "f" * 64
    elif tamper == "post_state_object_sha256":
        receipt["post_snapshot"]["state"]["object_sha256"] = "f" * 64
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(tamper)
    case["write_receipt"](receipt)

    with pytest.raises(iv.VerificationError, match="digest|prefix|binding"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


def test_independent_version_forward_receipt_rejects_coherently_rehashed_operation_tamper(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    receipt = case["receipt"]
    bad_operation = "f" * 64
    receipt["controller_return"]["v008_durable_controller_adapter"][
        "operation_sha256"
    ] = bad_operation
    post_object = receipt["post_snapshot"]["state"]["object"]
    post_object["v008_durable_controller_adapter"][
        "operation_sha256"
    ] = bad_operation

    def rebind(state: dict[str, Any]) -> None:
        marker = state["v008_durable_controller_adapter"]
        marker_without_binding = dict(marker)
        marker_without_binding.pop("state_binding_sha256")
        state_without_marker = json.loads(json.dumps(state))
        state_without_marker.pop("v008_durable_controller_adapter")
        marker["state_binding_sha256"] = hashlib.sha256(
            json.dumps(
                {
                    "marker_without_state_binding": marker_without_binding,
                    "state_without_marker": state_without_marker,
                },
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    rebind(receipt["controller_return"])
    rebind(post_object)
    state_payload = (
        json.dumps(post_object, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    receipt["post_snapshot"]["state"].update(
        {
            "sha256": hashlib.sha256(state_payload).hexdigest(),
            "bytes": len(state_payload),
            "object_sha256": hashlib.sha256(
                json.dumps(
                    post_object,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
    )
    receipt["controller_return_sha256"] = hashlib.sha256(
        json.dumps(
            receipt["controller_return"],
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    case["write_receipt"](receipt)

    with pytest.raises(iv.VerificationError, match="operation linkage"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize(
    "relative",
    [
        f"{iv.STUDY_RELATIVE}/STATE_TRANSACTION.json",
        f"{iv.STUDY_RELATIVE}/.STATE_TRANSACTION.json.v008-staging",
        f"{iv.STUDY_RELATIVE}/.STATE.json.v008-staging",
        f"{iv.STUDY_RELATIVE}/VERSION_FORWARD_TRANSACTION.json",
        f"{iv.STUDY_RELATIVE}/.VERSION_FORWARD_TRANSACTION.json.v008-staging",
        f"{iv.ATTEMPT_ROOTS[iv.ACTIVE_ATTEMPT]}/audit/.version_forward_transaction_receipt.json.v008-staging",
    ],
)
def test_independent_version_forward_receipt_rejects_every_journal_and_staging_residue(
    tmp_path: Path, relative: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    residue = tmp_path / relative
    residue.parent.mkdir(parents=True, exist_ok=True)
    if relative.endswith(".v008-staging") and "STATE.json" in relative:
        residue.symlink_to(residue.parent / "missing-target")
    else:
        residue.write_text("residue", encoding="utf-8")

    with pytest.raises(iv.VerificationError, match="residue"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


@pytest.mark.parametrize("alias_kind", ["receipt_hardlink", "fixed_hardlink", "receipt_symlink"])
def test_independent_version_forward_receipt_rejects_replace_or_alias_identity(
    tmp_path: Path, alias_kind: str
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    if alias_kind == "receipt_hardlink":
        os.link(case["receipt_path"], case["receipt_path"].with_suffix(".alias"))
    elif alias_kind == "fixed_hardlink":
        os.link(case["seal_path"], case["seal_path"].with_suffix(".alias"))
    elif alias_kind == "receipt_symlink":
        payload = case["receipt_path"].read_bytes()
        target = case["receipt_path"].with_suffix(".target")
        target.write_bytes(payload)
        case["receipt_path"].unlink()
        case["receipt_path"].symlink_to(target)
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(alias_kind)

    with pytest.raises(iv.VerificationError, match="linked|symlink|identity"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


def test_independent_version_forward_receipt_rejects_fully_rehashed_direct_root_proposal(
    tmp_path: Path,
) -> None:
    def remove_inheritance_binding(proposal: dict[str, Any]) -> None:
        checkpoint = proposal["state"]["verified_checkpoints"][0]
        checkpoint.pop("inherited_into_attempts")
        proposal["state"]["last_verified_checkpoint"] = json.loads(
            json.dumps(checkpoint)
        )

    case = _version_forward_receipt_v2_case(
        tmp_path, proposal_mutator=remove_inheritance_binding
    )

    with pytest.raises(iv.VerificationError, match="declarative root transition"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


def test_independent_version_forward_receipt_rejects_current_direct_root_state(
    tmp_path: Path,
) -> None:
    case = _version_forward_receipt_v2_case(tmp_path)
    current = json.loads(json.dumps(case["state"]))
    current.pop("v008_durable_controller_adapter")
    (case["study"] / "STATE.json").write_bytes(
        (
            json.dumps(current, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )
    case["verify_kwargs"]["current_state"] = current

    with pytest.raises(iv.VerificationError, match="adapter marker drift"):
        iv._verify_version_forward_transaction_receipt(**case["verify_kwargs"])


def test_version_forward_lineage_invokes_independent_schema_v2_receipt_verifier() -> None:
    tree = ast.parse((ROOT / "independent_verify.py").read_text(encoding="utf-8"))
    lineage = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "verify_version_forward_lineage"
    )
    calls = {
        node.func.id
        for node in ast.walk(lineage)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    returned_keys = {
        key.value
        for node in ast.walk(lineage)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        for key in node.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "_verify_version_forward_transaction_receipt" in calls
    assert "transaction_receipt" in returned_keys


def test_independent_durable_controller_marker_binds_schema_sources_and_operation(
    tmp_path: Path,
) -> None:
    policy, state, event_count, head, adapter, _ = _durable_controller_marker_case(
        tmp_path
    )

    evidence = iv._verify_durable_controller_adapter_marker(
        state,
        ledger_event_count=event_count,
        ledger_head_sha256=head,
        policy=policy,
    )

    assert evidence["passed"] is True
    assert evidence["ledger_event_count"] == event_count
    assert evidence["ledger_head_sha256"] == head
    assert evidence["operation_sha256"] == "b" * 64
    assert evidence["state_binding_sha256"] == state[
        "v008_durable_controller_adapter"
    ]["state_binding_sha256"]

    adapter.write_text("ADAPTER_VERSION = 3\n", encoding="utf-8")
    with pytest.raises(iv.VerificationError, match="adapter marker drift"):
        iv._verify_durable_controller_adapter_marker(
            state,
            ledger_event_count=event_count,
            ledger_head_sha256=head,
            policy=policy,
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("schema_version", 1),
        ("schema_version", True),
        ("authorization_kind", "direct_root_controller"),
        ("adapter_source_path", "runs/study/program.py"),
        ("adapter_source_sha256", "c" * 64),
        ("adapter_source_ast_sha256", "c" * 64),
        ("root_program_path", "runs/study/attempts/v008/program.py"),
        ("root_program_sha256", "c" * 64),
        ("root_program_ast_sha256", "c" * 64),
        ("operation_sha256", "C" * 64),
        ("state_binding_sha256", "c" * 64),
    ],
)
def test_independent_durable_controller_marker_rejects_field_tamper(
    tmp_path: Path, field: str, tampered: Any
) -> None:
    policy, state, event_count, head, _, _ = _durable_controller_marker_case(
        tmp_path
    )
    state["v008_durable_controller_adapter"][field] = tampered

    with pytest.raises(iv.VerificationError, match="adapter marker drift"):
        iv._verify_durable_controller_adapter_marker(
            state,
            ledger_event_count=event_count,
            ledger_head_sha256=head,
            policy=policy,
        )


@pytest.mark.parametrize("mutation", ["missing_state_binding", "extra_field"])
def test_independent_durable_controller_marker_requires_exact_v2_schema(
    tmp_path: Path, mutation: str
) -> None:
    policy, state, event_count, head, _, _ = _durable_controller_marker_case(
        tmp_path
    )
    marker = state["v008_durable_controller_adapter"]
    if mutation == "missing_state_binding":
        marker.pop("state_binding_sha256")
    elif mutation == "extra_field":
        marker["unexpected"] = None
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(mutation)

    with pytest.raises(iv.VerificationError, match="adapter marker drift"):
        iv._verify_durable_controller_adapter_marker(
            state,
            ledger_event_count=event_count,
            ledger_head_sha256=head,
            policy=policy,
        )


def test_independent_durable_controller_marker_rejects_direct_root_advance(
    tmp_path: Path,
) -> None:
    policy, state, event_count, head, _, _ = _durable_controller_marker_case(
        tmp_path
    )
    advanced_head = "d" * 64
    state["ledger_event_count"] = event_count + 1
    state["ledger_head_sha256"] = advanced_head

    with pytest.raises(iv.VerificationError, match="adapter marker drift"):
        iv._verify_durable_controller_adapter_marker(
            state,
            ledger_event_count=event_count + 1,
            ledger_head_sha256=advanced_head,
            policy=policy,
        )


def test_independent_durable_controller_marker_rejects_lowercase_operation_replacement(
    tmp_path: Path,
) -> None:
    policy, state, event_count, head, _, _ = _durable_controller_marker_case(
        tmp_path
    )
    state["v008_durable_controller_adapter"]["operation_sha256"] = "c" * 64

    with pytest.raises(iv.VerificationError, match="adapter marker drift"):
        iv._verify_durable_controller_adapter_marker(
            state,
            ledger_event_count=event_count,
            ledger_head_sha256=head,
            policy=policy,
        )


def test_independent_durable_controller_marker_rejects_state_only_mutation(
    tmp_path: Path,
) -> None:
    policy, state, event_count, head, _, _ = _durable_controller_marker_case(
        tmp_path
    )
    state["arbitrary_uncommitted_state"] = {"same_ledger": True}

    with pytest.raises(iv.VerificationError, match="adapter marker drift"):
        iv._verify_durable_controller_adapter_marker(
            state,
            ledger_event_count=event_count,
            ledger_head_sha256=head,
            policy=policy,
        )


def test_manifest_hash_mutation_and_symlink_fail_closed(tmp_path: Path):
    policy, attempt = _policy(tmp_path)
    artifact = attempt / "artifact.bin"
    artifact.write_bytes(b"sealed")
    relative = artifact.relative_to(tmp_path).as_posix()
    files = {relative: hashlib.sha256(b"sealed").hexdigest()}
    assert iv.verify_file_map(files, policy, scope="attempt")["file_count"] == 1
    artifact.write_bytes(b"mutated")
    with pytest.raises(iv.VerificationError, match="hash drift"):
        iv.verify_file_map(files, policy, scope="attempt")
    target = attempt / "target.bin"
    target.write_bytes(b"x")
    alias = attempt / "alias.bin"
    alias.symlink_to(target)
    with pytest.raises(iv.VerificationError, match="symlink"):
        policy.path(alias.relative_to(tmp_path).as_posix(), scope="attempt")


def test_legacy_prefix_byte_mutation_is_rejected_before_state_use(tmp_path: Path):
    policy, attempt = _policy(tmp_path)
    study = attempt.parents[1]
    ledger = study / "RESEARCH_LEDGER.jsonl"
    prefix = b'{"event":"program_initialized","attempt":"v001","created_unix_ns":1}\n'
    ledger.write_bytes(prefix[:-2] + b"2}\n")
    genesis = study / "LEDGER_CHAIN_GENESIS.json"
    genesis.write_text(
        json.dumps(
            {
                "legacy_prefix_bytes": len(prefix),
                "legacy_prefix_event_count": 1,
                "legacy_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            }
        )
    )
    state = study / "STATE.json"
    state.write_text("{}")
    contract = {
        "paths": {
            "ledger": ledger.relative_to(tmp_path).as_posix(),
            "ledger_genesis": genesis.relative_to(tmp_path).as_posix(),
            "state": state.relative_to(tmp_path).as_posix(),
        }
    }
    with pytest.raises(iv.VerificationError, match="legacy ledger prefix drift"):
        iv.verify_ledger_and_state(contract, policy)


def test_capture_audit_is_exclusive(tmp_path: Path):
    destination = tmp_path / "audit.json"
    capture._write_exclusive(destination, b'{"passed":true}\n')
    assert destination.read_bytes() == b'{"passed":true}\n'
    with pytest.raises(FileExistsError):
        capture._write_exclusive(destination, b'{"passed":false}\n')


def test_capture_persists_one_failed_verifier_result_for_invalid_mapping(tmp_path: Path, monkeypatch):
    attempt = tmp_path / "runs/study/attempts/v001"
    attempt.mkdir(parents=True)
    contract_path = attempt / "verifier_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "attempt": "v001",
                "mode": "confirmation",
                "attempt_root": "runs/study/attempts/v001",
            }
        )
    )
    failed = b'{"schema_version":1,"attempt":"v001","passed":false,"error":"mutated input"}\n'
    monkeypatch.setattr(capture, "_reject_links", lambda *_: None)
    monkeypatch.setattr(
        capture.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stdout=failed, stderr=b"diagnostic"),
    )
    result, output = capture.capture(contract_path)
    assert result["passed"] is False
    assert result["capture"]["wrapper_integrity_passed"] is True
    assert result["capture"]["verifier_returncode"] == 1
    assert json.loads(output.read_text())["error"] == "mutated input"
    with pytest.raises(FileExistsError):
        capture.capture(contract_path)


def test_capture_analysis_invalid_success_is_accepted_by_controller_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    attempt = tmp_path / "runs/study/attempts/v001"
    (attempt / "audit").mkdir(parents=True)
    contract_path = attempt / "verifier_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt": "v001",
                "mode": "confirmation",
                "attempt_root": "runs/study/attempts/v001",
            }
        )
    )
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    failure = attempt / "audit/analysis_execution_invalid.json"
    failure.write_text('{"execution_invalid":true}\n')
    failure_hash = hashlib.sha256(failure.read_bytes()).hexdigest()
    success = {
        "schema_version": 1,
        "attempt": "v001",
        "mode": "confirmation",
        "checkpoint_state": "INDEPENDENT_VERIFICATION",
        "passed": True,
        "terminal_label": iv.TERMINAL_INVALID,
        "integrity_failure": True,
        "read_only_verifier": True,
        "local_production_modules_imported": False,
        "stdout_json_only": True,
        "checks": {"analysis_integrity_failure_authenticated": True},
        "verifier_contract": {
            "path": contract_path.relative_to(tmp_path).as_posix(),
            "sha256": contract_hash,
        },
        "source_hashes": {
            "analysis_execution_invalid": {
                "path": failure.relative_to(tmp_path).as_posix(),
                "sha256": failure_hash,
            }
        },
    }
    encoded = json.dumps(success, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    monkeypatch.setattr(capture, "_reject_links", lambda *_: None)
    monkeypatch.setattr(
        capture.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=encoded, stderr=b""),
    )

    captured, audit_path = capture.capture(contract_path)

    assert captured["passed"] is True
    assert captured["capture"]["scientific_verifier_passed"] is True
    assert captured["capture"]["verifier_returncode"] == 0
    specification = importlib.util.spec_from_file_location(
        "domain_gate_program_capture_integration", ROOT.parents[1] / "program.py"
    )
    program = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(program)
    monkeypatch.setattr(program, "REPO_ROOT", tmp_path)
    state = {
        "active_attempt": "v001",
        "active_attempt_path": "runs/study/attempts/v001",
        "postconfirmation_integrity_failure": {
            "source_sha256": failure_hash,
            "audit_hash_keys": ["analysis_execution_invalid"],
        },
    }
    assert program._validate_execution_invalid_audit(state, audit_path)["passed"] is True


def test_capture_downgrades_malformed_success_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    attempt = tmp_path / "runs/study/attempts/v001"
    attempt.mkdir(parents=True)
    contract_path = attempt / "verifier_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt": "v001",
                "mode": "confirmation",
                "attempt_root": "runs/study/attempts/v001",
            }
        )
    )
    malformed = b'{"schema_version":1,"attempt":"v001","mode":"confirmation","passed":true}\n'
    monkeypatch.setattr(capture, "_reject_links", lambda *_: None)
    monkeypatch.setattr(
        capture.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=malformed, stderr=b""),
    )

    result, _ = capture.capture(contract_path)

    assert result["passed"] is False
    assert result["error_type"] == "VerifierSuccessIntegrityError"


@pytest.mark.parametrize("result_present", [False, True])
def test_analysis_execution_invalid_branch_is_preload_and_binds_result_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result_present: bool
):
    policy, attempt = _policy(tmp_path)
    contract_path = attempt / "verifier_contract.json"
    paths = {
        "state": "runs/study/STATE.json",
        "ledger": "runs/study/RESEARCH_LEDGER.jsonl",
        "pre_data_seal": "runs/study/attempts/v008/audit/pre_data.json",
        "pre_selection_seal": "runs/study/attempts/v008/audit/pre_selection.json",
        "pre_confirmation_seal": "runs/study/attempts/v008/audit/pre_confirmation.json",
        "confirmation_input_seal": "runs/study/attempts/v008/audit/confirmation_input.json",
        "power_freeze": "runs/study/attempts/v008/power_analysis.json",
        "power_rule": "runs/study/attempts/v008/power_rule.json",
        "fit_lock": "runs/study/attempts/v008/fit/fit_lock.json",
        "selection_ledger": "runs/study/attempts/v008/selection/selection_ledger.json",
        "gate_freeze": "runs/study/attempts/v008/freeze/gate_freeze.json",
        "compiled_gate": "runs/study/attempts/v008/freeze/compiled_gate.npz",
        "compiled_gate_manifest": "runs/study/attempts/v008/freeze/compiled_gate_manifest.json",
        "cohort_seed_ledger": "runs/study/attempts/v008/cohort_seed_ledger.json",
        "fixed_whitening": "runs/v5/whitening.npz",
        "analysis_result": "runs/study/attempts/v008/analysis_result.json",
    }
    required_files = set(paths.values()) - {paths["analysis_result"]}
    required_files.add("runs/study/attempts/v008/analysis.py")
    for raw in required_files:
        path = tmp_path / raw
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
    contract_path.write_text("{}\n")
    analysis_path = tmp_path / paths["analysis_result"]
    if result_present:
        analysis_path.write_text('{"partial":true}\n')
    expected_inputs = {
        "confirmation_input_seal": paths["confirmation_input_seal"],
        "pre_confirmation_package_seal": paths["pre_confirmation_seal"],
        "analysis_source": "runs/study/attempts/v008/analysis.py",
        "fixed_whitening": paths["fixed_whitening"],
        "compiled_gate_manifest": paths["compiled_gate_manifest"],
        "cohort_seed_ledger": paths["cohort_seed_ledger"],
        "compiled_gate": paths["compiled_gate"],
    }
    input_records = {}
    for name, raw in expected_inputs.items():
        path = tmp_path / raw
        input_records[name] = {
            "path": raw,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    prior_checkpoint = {"evidence_path": paths["confirmation_input_seal"], "evidence_sha256": "prior"}
    failure = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "SEALED_ANALYSIS",
        "status": "irreversible_post_open_analysis_execution_invalid",
        "created_unix_ns": 5,
        "passed": False,
        "integrity_failure": True,
        "integrity_passed": False,
        "process_valid": False,
        "execution_invalid": True,
        "proposed_terminal_label": iv.TERMINAL_INVALID,
        "confirmation_opened": True,
        "confirmation_outcomes_opened_for_analysis": True,
        "confirmation_outcome_episodes_generated": 2000,
        "confirmation_outcome_episodes_executed": 2000,
        "analysis_result_present": result_present,
        "analysis_result": (
            {
                "path": paths["analysis_result"],
                "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
                "bytes": analysis_path.stat().st_size,
            }
            if result_present
            else None
        ),
        "error_type": "RuntimeError",
        "error": "synthetic post-open failure",
        "scientific_objects_changed": False,
        "failure": {
            "exception_type": "RuntimeError",
            "exception_message": "synthetic post-open failure",
            "traceback_includes_locals": False,
        },
        "controller_chronology": {
            "state_path": paths["state"],
            "research_ledger_path": paths["ledger"],
            "ledger_event_count": 1,
            "ledger_head_sha256": "oldhead",
            "last_verified_checkpoint": prior_checkpoint,
            "controller_status_verified": True,
        },
        "inputs": input_records,
        "partial_derived_outputs": {},
        "confirmation_input_seal_sha256": input_records["confirmation_input_seal"]["sha256"],
        "execution_root": "runs/study/attempts/v008/data/confirmation",
        "failure_capture_opened_no_arrays": True,
        "outcome_values_recorded": False,
        "contact_motion_phase_reward_success_opened": False,
        "scientific_objects_changed_after_open": False,
        "retry_permitted": False,
        "independent_verification_required": True,
    }
    failure_path = attempt / "audit/analysis_execution_invalid.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps(failure))
    failure_hash = hashlib.sha256(failure_path.read_bytes()).hexdigest()
    state = {
        "postconfirmation_integrity_failure": {
            "source_sha256": failure_hash,
            "staged_unix_ns": 6,
        },
        "smoke_outcome_episodes": 24,
        "confirmation_outcome_episodes_generated": 2000,
        "confirmation_outcome_episodes_executed": 2000,
        "confirmation_outcomes_opened_for_analysis": True,
        "expected_confirmation_episode_count": 2000,
        "terminal_label": None,
        "verified_checkpoints": [
            prior_checkpoint,
            {"evidence_path": failure_path.relative_to(tmp_path).as_posix()},
        ],
    }
    events = [
        {"event": "program_initialized"},
        {
            "event": "state_completed",
            "completed_state": "SEALED_ANALYSIS",
            "postconfirmation_integrity_failure": True,
            "prev_sha256": "oldhead",
        },
    ]
    contract = {
        "mode": "confirmation",
        "attempt_root": "runs/study/attempts/v008",
        "paths": paths,
        "_analysis_execution_invalid_relative": failure_path.relative_to(tmp_path).as_posix(),
        "_loaded_contract_relative": contract_path.relative_to(tmp_path).as_posix(),
    }
    monkeypatch.setattr(iv, "verify_terminal_manifest", lambda *_: {"path_set_complete": True})
    monkeypatch.setattr(iv, "verify_normative_contracts", lambda *_: {"frozen": "hash"})
    monkeypatch.setattr(
        iv,
        "verify_ledger_and_state",
        lambda *_: {
            "state": state,
            "events": events,
            "event_count": len(events),
            "legacy_event_count": 1,
            "ledger_head_sha256": "newhead",
            "completed_state_count": 19,
        },
    )
    monkeypatch.setattr(
        iv,
        "verify_version_forward_lineage",
        lambda *_: {"version_forward_edges": 0, "seal_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        iv,
        "verify_identifier_ledger",
        lambda *_: {"episode_identifier_count": 1, "numeric_identifier_count": 2, "rng_identifier_count": 3},
    )
    seal_times = {"PRE_OUTCOME_SEAL": 1, "PRE_SELECTION_SEAL": 2, "PRE_CONFIRMATION_PACKAGE_SEAL": 3}
    monkeypatch.setattr(
        iv,
        "verify_seal",
        lambda *args, **kwargs: {"created_unix_ns": seal_times[kwargs["checkpoint_state"]]},
    )
    monkeypatch.setattr(iv, "verify_power_result", lambda *_args, **_kwargs: 500)
    monkeypatch.setattr(
        iv,
        "verify_power_input_provenance",
        lambda *_: {"identity": {}, "input_hashes": {}, "identity_crosslinks": {}, "local_power_rule_only": True},
    )
    monkeypatch.setattr(iv, "verify_confirmation_input_bindings", lambda *_: {"created_unix_ns": 4})
    monkeypatch.setattr(
        iv,
        "verify_scientific_replay_qualification",
        lambda *_args, **_kwargs: {
            "passed": True,
            "role_order": list(iv.REPLAY_ROLE_ORDER),
            "outcome_arrays_loaded": False,
        },
    )
    monkeypatch.setattr(iv, "_load_npz", lambda *_args, **_kwargs: pytest.fail("failure branch loaded an NPZ"))

    result = iv.verify_analysis_execution_invalid_branch(contract, policy)

    assert result["passed"] is True
    assert result["terminal_label"] == iv.TERMINAL_INVALID
    assert result["integrity_failure"] is True
    assert result["outcome_arrays_loaded"] is False
    assert all(type(value) is bool and value for value in result["checks"].values())


def test_verifier_imports_only_stdlib_and_numpy_and_failure_stdout_is_one_json_line(tmp_path: Path):
    tree = ast.parse((ROOT / "independent_verify.py").read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__", "argparse", "ast", "difflib", "hashlib", "json", "math", "os", "re", "stat", "sys",
        "fractions", "pathlib", "statistics", "typing", "numpy",
    }
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "independent_verify.py"), str(tmp_path / "absent.json")],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    lines = completed.stdout.splitlines()
    assert completed.returncode == 1 and len(lines) == 1
    assert json.loads(lines[0])["passed"] is False


def test_execution_allowlist_matches_frozen_runner_schema():
    assert iv.EXECUTION_KEYS == {
        "episode_slot", "model_step", "target", "exits", "selected", "calls", "scores",
        "production_features", "history", "action_history", "stage_current", "stage_update",
        "head_scores", "reached", "dense_scores", "dense_head_scores", "dense_stage_current",
        "dense_stage_update",
    }


def test_v008_lineage_normalization_and_contract_canonicalization_are_tamper_evident(tmp_path: Path):
    source = tmp_path / "source.py"
    target = tmp_path / "target.py"
    source.write_text('ATTEMPT = "v002"\nVALUE = "v002_pre_data"\n')
    target.write_text(
        'ACTIVE_ATTEMPT = ATTEMPT_ROOT.name\n'
        'if ACTIVE_ATTEMPT != "v008":\n    raise RuntimeError("wrong v008")\n'
        'VALUE = "v008_pre_data"\n'
    )
    assert (
        iv.normalized_administrative_ast_sha256(source)
        == iv.normalized_administrative_ast_sha256(target)
    )
    target.write_text(target.read_text() + 'SCIENTIFIC_DELTA = 1\n')
    assert (
        iv.normalized_administrative_ast_sha256(source)
        != iv.normalized_administrative_ast_sha256(target)
    )

    source = ROOT.parent / "v006"
    for name in iv.CANONICAL_CONTRACT_PATHS:
        assert iv.canonical_contract_sha256(source / name) == iv.canonical_contract_sha256(ROOT / name)


def test_v008_equivalence_partitions_are_independently_recomputed_and_tamper_evident():
    prepare = _module("version_forward_prepare_for_verifier_test", "version_forward_prepare.py")
    seal = prepare.build_inheritance_seal(ROOT.parents[1])
    repository = ROOT.parents[3]
    policy = iv.PathPolicy(
        repository,
        "runs/lewm_domain_robust_gate/attempts/v008",
        "runs/lewm_domain_robust_gate",
    )
    result = iv.verify_equivalence_partitions(seal, policy)
    assert result["source_path_count"] > 0
    assert result["target_path_count"] == sum(result["partition_counts"].values())

    tampered = json.loads(json.dumps(seal))
    tampered["source_partitions"]["exact_hash"][0]["sha256"] = "0" * 64
    with pytest.raises(iv.VerificationError, match="exact-file equivalence drift"):
        iv.verify_equivalence_partitions(tampered, policy)


def test_v008_equivalence_partitions_reject_lineage_provenance_drift():
    prepare = _module("version_forward_prepare_for_wildcard_test", "version_forward_prepare.py")
    seal = prepare.build_inheritance_seal(ROOT.parents[1])
    repository = ROOT.parents[3]
    policy = iv.PathPolicy(
        repository,
        "runs/lewm_domain_robust_gate/attempts/v008",
        "runs/lewm_domain_robust_gate",
    )
    tampered = json.loads(json.dumps(seal))
    assert tampered["source_partitions"]["new_lineage_support"] == []
    record = next(
        item
        for item in tampered["source_partitions"]["lineage_support"]
        if item["relative_path"] == "fit_inheritance.py"
    )
    record["purpose"] = "unrecorded_lineage_change"
    with pytest.raises(
        iv.VerificationError,
        match="lineage-support raw binding drift",
    ):
        iv.verify_equivalence_partitions(tampered, policy)


def test_v008_equivalence_partitions_reject_omitted_lineage_unit():
    prepare = _module("version_forward_prepare_for_omission_test", "version_forward_prepare.py")
    seal = prepare.build_inheritance_seal(ROOT.parents[1])
    repository = ROOT.parents[3]
    policy = iv.PathPolicy(
        repository,
        "runs/lewm_domain_robust_gate/attempts/v008",
        "runs/lewm_domain_robust_gate",
    )
    tampered = json.loads(json.dumps(seal))
    record = tampered["source_partitions"]["lineage_support"][0]
    record["required_changed_top_level_units"] = record[
        "required_changed_top_level_units"
    ][:-1]
    with pytest.raises(
        iv.VerificationError,
        match="lineage-support unit scope record drift",
    ):
        iv.verify_equivalence_partitions(tampered, policy)
