from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
if str(ATTEMPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTEMPT_ROOT))

import analysis  # noqa: E402
import latency  # noqa: E402


def tiny_pricing() -> analysis.ComputePricing:
    return analysis.ComputePricing(
        base_flops=100,
        mandatory_depth1_flops=10,
        additional_refiner_flops=20,
        gate_feature_flops=2,
        gate_head_flops=3,
        gate_nonflop_operations=5,
    )


def _synthetic_regime_arrays() -> dict[str, np.ndarray]:
    calls = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int64)
    slots = np.repeat(np.arange(2, dtype=np.int64), 4)
    steps = np.tile(np.arange(3, 7, dtype=np.int64), 2)
    scores = np.full((8, 3), np.nan, dtype=np.float64)
    features = np.full((8, 3, 5), np.nan, dtype=np.float64)
    score_template = {
        1: [-1.0],
        2: [1.0, -1.0],
        3: [1.0, 1.0, -1.0],
        4: [1.0, 1.0, 1.0],
    }
    for row, depth in enumerate(calls):
        values = score_template[int(depth)]
        scores[row, : len(values)] = values
        features[row, : len(values)] = row + 1
    target = np.zeros((8, 2), dtype=np.float64)
    exits = np.empty((8, 4, 2), dtype=np.float64)
    for depth in range(4):
        exits[:, depth] = (4 - depth) * 0.1 + np.arange(8)[:, None] * 0.001
    return {
        "episode_slot": slots,
        "model_step": steps,
        "target": target,
        "exits": exits,
        "selected": exits[np.arange(8), calls - 1],
        "calls": calls,
        "scores": scores,
        "production_features": features,
    }


def _analyze_synthetic(arrays: dict[str, np.ndarray]) -> None:
    analysis.analyze_regime(
        arrays,
        np.eye(2),
        tiny_pricing(),
        seeded_seeds={"raw": 10, "fixed_whitened": 11},
        histogram_seed=12,
        thresholds=[0.0, 0.0, 0.0],
        expected_rows_per_episode=4,
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization_fixture(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    attempt = repo / "runs/study/attempts/v008"
    execution_root = attempt / "data/confirmation"
    freeze = attempt / "freeze"
    audit = attempt / "audit"
    freeze.mkdir(parents=True)
    audit.mkdir(parents=True)

    whitening = repo / "fixed_whitening.npz"
    thresholds = freeze / "compiled_gate.npz"
    source = attempt / "analysis_frozen.py"
    whitening.write_bytes(b"known-v5-whitening-content")
    thresholds.write_bytes(b"compiled-gate-content")
    source.write_text("# frozen analysis source\n", encoding="utf-8")
    claims_root = attempt / "data/replacement_claims"
    claims_root.mkdir(parents=True)
    registry_path = attempt / "data/replacement_registry.json"
    registry = {"record_type": "replacement_registry_genesis"}
    registry["record_sha256"] = hashlib.sha256(
        json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(registry_path, registry)

    mapping_path = attempt / "outcome_mapping.json"
    _write_json(
        mapping_path,
        {
            "family_size": 8,
            "familywise_alpha": 0.05,
            "per_claim_alpha": 0.00625,
            "co_primary_endpoints": list(analysis.PRIMARY_CONTRASTS),
            "regime_order": list(analysis.DGP_ORDER),
            "bootstrap": {
                "paired_replicates": 20_000,
                "quantile_method": "NumPy linear",
            },
        },
    )
    dgp_path = attempt / "DGP_MATRIX.json"
    _write_json(dgp_path, {"regime_order": list(analysis.DGP_ORDER)})
    seed_path = attempt / "cohort_seed_ledger.json"
    seeded = {
        regime: {"raw": 100 + index * 3, "fixed_whitened": 101 + index * 3}
        for index, regime in enumerate(analysis.DGP_ORDER)
    }
    histograms = {
        regime: 102 + index * 3
        for index, regime in enumerate(analysis.DGP_ORDER)
    }
    _write_json(
        seed_path,
        {
            "analysis_seeds": {
                "joint_bootstrap_seed": 99,
                "bootstrap_replicates": 20_000,
                "seeded_comparator_seeds": seeded,
                "histogram_seeds": histograms,
            },
            "analysis_rng_ids": {
                "bootstrap": {
                    "rng_ids": [{"rng_id": 99}],
                    "replicate_count": 20_000,
                    "chunk_size_defining_rng_consumption": 250,
                    "regime_order": list(analysis.DGP_ORDER),
                }
            },
        },
    )
    pricing_path = freeze / "compiled_gate_manifest.json"
    _write_json(
        pricing_path,
        {
            "passed": True,
            "compiled_gate_path": thresholds.relative_to(repo).as_posix(),
            "compiled_gate_sha256": _sha(thresholds),
            "target_or_contact_arrays_opened": False,
            "prior_outcome_arrays_opened": False,
            "metadata": {
                "gate_cost": {
                    "feature_flops": 3_801,
                    "all_head_flops": 4_184,
                    "total_gate_flops": 7_985,
                }
            },
            "complete_compute_derivation": {
                "base_flops_per_row": analysis.BASE_FLOPS,
                "depth1_flops_per_row": analysis.MANDATORY_DEPTH1_FLOPS,
                "later_adapter_flops_per_call": analysis.ADDITIONAL_REFINER_FLOPS,
            },
        },
    )
    gate_freeze = freeze / "gate_freeze.json"
    _write_json(
        gate_freeze,
        {
            "status": "selected_gate_frozen_before_smoke_or_confirmation",
            "compiled_gate_sha256": _sha(thresholds),
            "selected_head_refit_after_selection": False,
            "contact_or_privileged_gate_inputs": False,
            "confirmation_episodes_at_freeze": 0,
        },
    )
    scientific = (
        whitening,
        pricing_path,
        seed_path,
        thresholds,
        source,
        mapping_path,
        dgp_path,
    )
    pre_files = {
        path.relative_to(repo).as_posix(): _sha(path)
        for path in (*scientific, gate_freeze)
    }
    registry_relative = registry_path.relative_to(repo).as_posix()
    pre_files[registry_relative] = _sha(registry_path)
    preseal_path = audit / "pre_confirmation_package_seal.json"
    _write_json(
        preseal_path,
        {
            "schema_version": 1,
            "attempt": "v008",
            "checkpoint_state": "PRE_CONFIRMATION_PACKAGE_SEAL",
            "passed": True,
            "confirmation_episode_count_per_regime": {
                regime: 1 for regime in analysis.DGP_ORDER
            },
            "sealed_files": pre_files,
        },
    )

    input_files = dict(pre_files)
    input_files[preseal_path.relative_to(repo).as_posix()] = _sha(preseal_path)
    regimes = {}
    parts = {}
    for index, regime in enumerate(analysis.DGP_ORDER):
        directory = execution_root / regime
        directory.mkdir(parents=True)
        part = directory / "episode.npz"
        part.write_bytes(f"sealed-part-{regime}".encode())
        part_relative = part.relative_to(repo).as_posix()
        part_hash = _sha(part)
        parts[part_relative] = part_hash
        manifest_path = directory / "execution_manifest.json"
        _write_json(
            manifest_path,
            {
                "episodes": [
                    {
                        "slot": 0,
                        "episode_id": f"episode-{index}",
                        "path": part_relative,
                        "sha256": part_hash,
                    }
                ]
            },
        )
        manifest_relative = manifest_path.relative_to(repo).as_posix()
        input_files[part_relative] = part_hash
        input_files[manifest_relative] = _sha(manifest_path)
        regimes[regime] = {
            "execution_manifest": {
                "path": manifest_relative,
                "sha256": _sha(manifest_path),
            }
        }
    input_seal_path = audit / "confirmation_input_seal.json"
    _write_json(
        input_seal_path,
        {
            "schema_version": 1,
            "attempt": "v008",
            "checkpoint_state": "CONFIRMATION_INPUT_SEAL",
            "passed": True,
            "confirmation_arrays_opened_before_seal": False,
            "contact_motion_phase_reward_success_opened": False,
            "pre_confirmation_package_seal_path": preseal_path.relative_to(repo).as_posix(),
            "pre_confirmation_package_seal_sha256": _sha(preseal_path),
            "expected_per_dgp": 1,
            "regimes": regimes,
            "execution_part_hashes": parts,
            "execution_sidecar_hashes": {},
            "replacement_registry_closure": {
                "genesis": registry,
                "claim_count": 0,
                "prefix_count": 0,
                "head_record_sha256": registry["record_sha256"],
                "postseal_claim_indexes": [],
                "references": {},
                "support_files": {
                    registry_relative: _sha(registry_path),
                },
            },
            "sealed_files": input_files,
        },
    )
    return {
        "execution_root": execution_root,
        "whitening_path": whitening,
        "pricing_path": pricing_path,
        "seed_path": seed_path,
        "thresholds_path": thresholds,
        "output_root": attempt,
        "repo_root": repo,
        "pre_confirmation_seal_path": preseal_path,
        "analysis_source_path": source,
        "required_scientific_paths": scientific,
        "expected_whitening_relative": None,
        "expected_whitening_sha256": _sha(whitening),
        "require_state_binding": False,
        "input_seal_path": input_seal_path,
    }


def test_compute_account_charges_every_selected_gate_overhead() -> None:
    calls = np.array([1, 2, 3, 4], dtype=np.int64)
    account = analysis.adaptive_compute_account(calls, tiny_pricing())
    assert account["gate_evaluations"] == 9
    assert account["refiner_model_calls"] == 10
    assert account["gate_feature_flops"] == 18
    assert account["gate_head_flops"] == 27
    assert account["gate_nonflop_operations"] == 45
    expected = 4 * 100 + 4 * 10 + 6 * 20 + 9 * 5
    assert account["adaptive_total_counted_flops"] == expected


def test_strongest_allocation_is_endpoint_specific_and_exact_compute() -> None:
    pricing = tiny_pricing()
    calls = np.array([2, 2, 2, 2], dtype=np.int64)
    budget = analysis.adaptive_compute_account(calls, pricing)[
        "adaptive_total_counted_flops"
    ]
    # Gate overhead raises the analytic equivalent depth from 2.0 to 2.5.
    raw = np.tile([8.0, 7.0, 6.0, 0.0], (4, 1))
    white = np.tile([100.0, 0.0, 0.0, 100.0], (4, 1))
    raw_allocation = analysis.strongest_transition_independent_allocation(
        raw, budget, pricing
    )
    white_allocation = analysis.strongest_transition_independent_allocation(
        white, budget, pricing
    )
    assert raw_allocation["depth_lower"] == 1
    assert raw_allocation["depth_upper"] == 4
    assert raw_allocation["allocation_total_counted_flops"] == pytest.approx(budget)
    assert white_allocation["depth_lower"] == 2
    assert white_allocation["depth_upper"] == 3
    assert white_allocation["allocation_total_counted_flops"] == pytest.approx(budget)
    assert not np.array_equal(
        raw_allocation["depth_probabilities"],
        white_allocation["depth_probabilities"],
    )
    assert raw_allocation["integer_cross_multiplication_equality"]
    assert (
        raw_allocation["allocation_compute_numerator"]
        == budget * raw_allocation["allocation_compute_denominator"]
    )


def test_strongest_allocation_rejects_large_budget_one_flop_outside_envelope() -> None:
    # A float tolerance scaled by this large base price would incorrectly admit
    # many outside-envelope FLOPs. Exact integer arithmetic must reject +1.
    pricing = analysis.ComputePricing(
        base_flops=10**18,
        mandatory_depth1_flops=10**17,
        additional_refiner_flops=1,
        gate_feature_flops=0,
        gate_head_flops=0,
        gate_nonflop_operations=0,
    )
    losses = np.ones((1, 4), dtype=np.float64)
    outside = pricing.fixed_depth_flops_per_row(4) + 1
    with pytest.raises(RuntimeError, match="outside"):
        analysis.strongest_transition_independent_allocation(
            losses, outside, pricing
        )


def test_seeded_control_is_deterministic_and_weakly_more_compute() -> None:
    pricing = tiny_pricing()
    calls = np.array([2, 2, 2, 2, 2], dtype=np.int64)
    budget = analysis.adaptive_compute_account(calls, pricing)[
        "adaptive_total_counted_flops"
    ]
    losses = np.arange(20, dtype=np.float64).reshape(5, 4)
    allocation = analysis.strongest_transition_independent_allocation(
        losses, budget, pricing
    )
    first = analysis.seeded_weakly_more_compute_control(
        losses, allocation, budget, pricing, 1234
    )
    second = analysis.seeded_weakly_more_compute_control(
        losses, allocation, budget, pricing, 1234
    )
    assert np.array_equal(first["calls"], second["calls"])
    assert np.array_equal(first["loss"], second["loss"])
    assert first["total_counted_flops"] >= budget
    assert first["baseline_minus_adaptive_flops"] < (
        pricing.fixed_depth_flops_per_row(first["depth_upper"])
        - pricing.fixed_depth_flops_per_row(first["depth_lower"])
    )


def test_histogram_randomization_preserves_every_episode() -> None:
    calls = np.array([1, 2, 2, 4, 1, 1, 3, 4], dtype=np.int64)
    slots = np.repeat(np.arange(2), 4)
    randomized, records = analysis.within_episode_histogram_calls(calls, slots, 55)
    assert all(record["preserved"] for record in records)
    for slot in range(2):
        original = np.bincount(calls[slots == slot], minlength=5)
        observed = np.bincount(randomized[slots == slot], minlength=5)
        assert np.array_equal(original, observed)
    repeated, _ = analysis.within_episode_histogram_calls(calls, slots, 55)
    assert np.array_equal(randomized, repeated)


def test_fixed_whitening_is_a_distinct_frozen_endpoint() -> None:
    target = np.zeros((2, 2), dtype=np.float64)
    exits = np.array(
        [
            [[1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
            [[2.0, 1.0], [1.0, 2.0], [1.0, 0.0], [1.0, 1.0]],
        ]
    )
    selected = exits[:, 0]
    whitening = np.diag([2.0, 0.5])
    fixed, adaptive = analysis.endpoint_losses(target, exits, selected, whitening)
    assert fixed["raw"].shape == (2, 4)
    assert fixed["fixed_whitened"].shape == (2, 4)
    assert not np.allclose(fixed["raw"], fixed["fixed_whitened"])
    assert np.array_equal(adaptive["raw"], fixed["raw"][:, 0])


def test_strict_threshold_tie_stops() -> None:
    scores = np.array(
        [
            [0.0, np.nan, np.nan],
            [0.1, 0.0, np.nan],
            [0.1, 0.1, 0.0],
            [0.1, 0.1, 0.1],
        ]
    )
    calls = analysis.sequential_calls_from_scores(scores, [0.0, 0.0, 0.0])
    assert calls.tolist() == [1, 2, 3, 4]


def test_stagewise_rank_reports_both_endpoints_and_reached_counts() -> None:
    calls = np.array([1, 2, 3, 4], dtype=np.int64)
    scores = np.array(
        [
            [-2.0, np.nan, np.nan],
            [-1.0, -1.0, np.nan],
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 1.0],
        ]
    )
    base = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 2.0, 1.0, -1.0],
        ]
    )
    ranks, calibration, integrity = analysis.stagewise_rank_and_calibration(
        scores, calls, {"raw": base, "fixed_whitened": base * 2}, bins=2
    )
    assert [item["reached_rows"] for item in ranks] == [4, 3, 2]
    assert set(ranks[0]["endpoints"]) == set(analysis.ENDPOINTS)
    assert len(calibration) == 3
    assert all(integrity.values())


def test_undefined_diagnostics_are_null_and_zero_reached_shifts_are_available_safe() -> None:
    rows = 6
    calls = np.ones(rows, dtype=np.int64)
    scores = np.full((rows, 3), np.nan, dtype=np.float64)
    scores[:, 0] = 0.5
    features = np.full((rows, 3, 2), np.nan, dtype=np.float64)
    features[:, 0] = np.arange(rows, dtype=np.float64)[:, None]
    losses = np.tile(np.arange(4, dtype=np.float64), (rows, 1))
    ranks, _, integrity = analysis.stagewise_rank_and_calibration(
        scores,
        calls,
        {"raw": losses, "fixed_whitened": losses},
    )
    assert ranks[0]["endpoints"]["raw"]["spearman_score_next_stage_gain_rho"] is None
    assert ranks[1]["reached_rows"] == 0
    assert all(integrity.values())

    shifts = analysis.feature_and_gain_shift(
        features,
        scores,
        calls,
        {"raw": losses, "fixed_whitened": losses},
        features,
        scores,
        calls,
        {"raw": losses, "fixed_whitened": losses},
    )
    assert shifts["stages"][1]["features"]["shift_available"] is False
    assert shifts["stages"][1]["score"]["mean"] is None
    assert shifts["stages"][1]["solver_next_stage_gain"]["raw"]["mean"] is None
    json.dumps({"ranks": ranks, "shifts": shifts}, allow_nan=False)


def test_analyze_regime_end_to_end_on_outcome_free_synthetic_arrays() -> None:
    calls = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int64)
    slots = np.repeat(np.arange(2), 4)
    steps = np.tile(np.arange(3, 7), 2)
    scores = np.full((8, 3), np.nan, dtype=np.float64)
    features = np.full((8, 3, 5), np.nan, dtype=np.float64)
    score_template = {
        1: [-1.0],
        2: [1.0, -1.0],
        3: [1.0, 1.0, -1.0],
        4: [1.0, 1.0, 1.0],
    }
    for row, depth in enumerate(calls):
        values = score_template[int(depth)]
        scores[row, : len(values)] = values
        features[row, : len(values)] = row + 1
    target = np.zeros((8, 2), dtype=np.float64)
    exits = np.empty((8, 4, 2), dtype=np.float64)
    for depth in range(4):
        exits[:, depth] = (4 - depth) * 0.1 + np.arange(8)[:, None] * 0.001
    selected = exits[np.arange(8), calls - 1]
    arrays = {
        "episode_slot": slots,
        "model_step": steps,
        "target": target,
        "exits": exits,
        "selected": selected,
        "calls": calls,
        "scores": scores,
        "production_features": features,
    }
    summary, metrics, diagnostic = analysis.analyze_regime(
        arrays,
        np.eye(2),
        tiny_pricing(),
        seeded_seeds={"raw": 10, "fixed_whitened": 11},
        histogram_seed=12,
        thresholds=[0.0, 0.0, 0.0],
        expected_rows_per_episode=4,
    )
    assert summary["process_valid_prebootstrap"]
    assert set(analysis.ALL_CONTRASTS).issubset(metrics)
    assert all(metrics[name].shape == (2,) for name in analysis.ALL_CONTRASTS)
    assert diagnostic["production_features"].shape == (8, 3, 5)


@pytest.mark.parametrize("field", ["episode_slot", "model_step", "calls"])
def test_analysis_rejects_noninteger_execution_indices_before_cast(
    field: str,
) -> None:
    arrays = _synthetic_regime_arrays()
    arrays[field] = arrays[field].astype(np.float64)
    arrays[field][0] += 0.25
    with pytest.raises(RuntimeError, match="integer dtype"):
        _analyze_synthetic(arrays)


def test_analysis_rejects_integer_calls_outside_frozen_range() -> None:
    arrays = _synthetic_regime_arrays()
    arrays["calls"][0] = 0
    with pytest.raises(RuntimeError, match="outside the frozen range"):
        _analyze_synthetic(arrays)


def test_analysis_rejects_non_nan_unreached_score() -> None:
    arrays = _synthetic_regime_arrays()
    arrays["scores"][0, 1] = np.inf
    with pytest.raises(RuntimeError, match="unreached gate scores must be exactly NaN"):
        _analyze_synthetic(arrays)


def test_analysis_rejects_partially_populated_unreached_feature_row() -> None:
    arrays = _synthetic_regime_arrays()
    arrays["production_features"][0, 1, 0] = 0.0
    with pytest.raises(RuntimeError, match="must be entirely NaN"):
        _analyze_synthetic(arrays)


def test_latency_equivalence_and_energy_unavailability() -> None:
    sparse = (np.array([[1.0, 2.0]]), np.array([2]))
    dense = (np.array([[1.0, 2.0 + 1e-8]]), np.array([2]))
    result = latency.path_equivalence(sparse, dense, np.array([2]))
    assert result["passed"]

    class Adapter:
        pass

    energy = latency.energy_availability(Adapter())
    assert energy["available"] is False

    nonfinite = latency.path_equivalence(
        (np.array([[np.nan]]), np.array([1])),
        (np.array([[np.nan]]), np.array([1])),
        np.array([1]),
    )
    assert nonfinite["passed"] is False
    assert nonfinite["sparse_dense_max_abs"] is None
    json.dumps(nonfinite, allow_nan=False)
    assert energy["measured"] is False


def _latency_binding_and_result() -> tuple[dict, dict]:
    binding = {
        "analysis_result_path": "analysis_result.json",
        "analysis_result_sha256": "a" * 64,
        "analysis_result_bytes": 123,
        "runner_module": "runner",
        "runner_source_path": "runner.py",
        "runner_source_sha256": "b" * 64,
        "runner_provenance": {"frozen": True},
        "requested_device": "mps",
        "resolved_device": "mps:0",
        "warmups": 2,
        "repetitions": 7,
    }
    regimes = {
        regime: {
            "passed": True,
            "equivalence": [
                {"batch_size": batch_size, "passed": True}
                for batch_size in latency.BATCH_SIZES
            ],
        }
        for regime in latency.DGP_ORDER
    }
    result = {
        "schema_version": 1,
        "attempt": "v008",
        "checkpoint_state": "LATENCY_AND_RESOURCE_REPORTING",
        "passed": True,
        "integrity_passed": True,
        "input_binding": binding,
        "input_binding_sha256": latency._canonical_sha256(binding),
        "device": "mps:0",
        "regimes": regimes,
        "module_before": {"passed": True},
        "module_after": {"passed": True},
        "runner_provenance": {"frozen": True},
        "contact_motion_phase_reward_success_opened": False,
    }
    return binding, result


def test_latency_pass_cannot_mask_failed_integrity() -> None:
    binding, result = _latency_binding_and_result()
    result["integrity_passed"] = False
    checks = latency.validate_latency_result(result, binding)
    assert checks["passed_implies_integrity"] is False
    assert checks["integrity_exact"] is False


def test_existing_latency_output_rejects_input_binding_drift(tmp_path: Path) -> None:
    binding, result = _latency_binding_and_result()
    path = tmp_path / "latency.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    changed = dict(binding)
    changed["analysis_result_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="unbound or internally invalid"):
        latency.load_existing_result(path, changed)


def _metric_fixture(offset: float) -> dict[str, np.ndarray]:
    return {
        name: np.linspace(-0.2, 0.4, 12, dtype=np.float64) + offset
        for name in analysis.ALL_CONTRASTS
    }


def test_joint_bootstrap_reuses_dgp_sample_over_all_contrasts() -> None:
    metrics = {
        regime: _metric_fixture(index)
        for index, regime in enumerate(analysis.DGP_ORDER)
    }
    # Within a DGP every fixture contrast is identical, so sharing the frozen
    # sample matrix must produce elementwise-identical replicate arrays.
    bootstrap = analysis.bootstrap_all(
        metrics, 987, replicates=500, chunk_size=50
    )
    for regime in analysis.DGP_ORDER:
        first = bootstrap[regime][analysis.ALL_CONTRASTS[0]]
        assert all(
            np.array_equal(first, bootstrap[regime][name])
            for name in analysis.ALL_CONTRASTS
        )


def test_linear_quantile_and_eight_claim_terminal_mapping() -> None:
    assert analysis.quantile(np.array([0.0, 10.0]), 0.25) == 2.5

    def family(supported: int) -> dict[str, dict[str, dict[str, bool]]]:
        remaining = supported
        output = {}
        for regime in analysis.DGP_ORDER:
            output[regime] = {}
            for endpoint in analysis.ENDPOINTS:
                output[regime][endpoint] = {"supported": remaining > 0}
                remaining -= 1
        return output

    assert analysis.terminal_mapping(family(8), integrity_valid=True) == (
        analysis.TERMINAL_CONFIRMED,
        8,
    )
    assert analysis.terminal_mapping(family(7), integrity_valid=True) == (
        analysis.TERMINAL_PARTIAL,
        7,
    )
    assert analysis.terminal_mapping(family(0), integrity_valid=True) == (
        analysis.TERMINAL_FAILED,
        0,
    )
    assert analysis.terminal_mapping(family(8), integrity_valid=False) == (
        analysis.TERMINAL_INVALID,
        8,
    )


def test_replicate_bundle_persists_every_array_and_hash(tmp_path: Path) -> None:
    bootstrap = {
        regime: {
            name: np.arange(20, dtype=np.float64) + index
            for index, name in enumerate(analysis.ALL_CONTRASTS)
        }
        for regime in analysis.DGP_ORDER
    }
    path = tmp_path / "bootstrap_replicates.npz"
    manifest = analysis.persist_bootstrap_replicates(
        path, bootstrap, seed=44, replicates=20, chunk_size=5
    )
    assert manifest["array_count"] == 32
    assert len(manifest["arrays"]) == 32
    assert manifest["sha256"] == analysis.sha256_file(path)
    with np.load(path, allow_pickle=False) as stored:
        assert set(stored.files) == set(manifest["arrays"])
        for name in stored.files:
            assert analysis.array_sha256(stored[name]) == manifest["arrays"][name]["sha256"]


def test_confirmation_schema_rejects_contact_and_extra_fields() -> None:
    good = set(analysis.EXECUTION_ARRAY_KEYS)
    analysis.validate_execution_array_keys(good)
    with pytest.raises(RuntimeError, match="forbidden"):
        analysis.validate_execution_array_keys(good | {"contact_force"})
    with pytest.raises(RuntimeError, match="schema drift"):
        analysis.validate_execution_array_keys(good | {"innocent_extra"})


def test_transitive_authorization_opens_no_npz(monkeypatch, tmp_path: Path) -> None:
    fixture = _authorization_fixture(tmp_path)
    fixture.pop("input_seal_path")

    def forbidden_np_load(*args, **kwargs):
        raise AssertionError("np.load was reached during byte-only authorization")

    monkeypatch.setattr(analysis.np, "load", forbidden_np_load)
    result = analysis.validate_sealed_analysis_authorization(**fixture)
    assert result["npz_arrays_opened_during_authorization"] is False
    assert result["execution_manifest_and_part_hashes_transitively_verified"]


def test_post_input_seal_replacement_append_fails_before_npz(
    monkeypatch, tmp_path: Path
) -> None:
    fixture = _authorization_fixture(tmp_path)
    fixture.pop("input_seal_path")
    claims_root = fixture["output_root"] / "data/replacement_claims"
    _write_json(claims_root / "000000.json", {})

    def forbidden_np_load(*args, **kwargs):
        raise AssertionError("np.load was reached after a post-seal claim append")

    monkeypatch.setattr(analysis.np, "load", forbidden_np_load)
    with pytest.raises(RuntimeError, match="live replacement claim directory"):
        analysis.validate_sealed_analysis_authorization(**fixture)


def test_substituted_scientific_file_fails_before_np_load(
    monkeypatch, tmp_path: Path
) -> None:
    fixture = _authorization_fixture(tmp_path)
    fixture.pop("input_seal_path")
    fixture["thresholds_path"].write_bytes(b"substituted-compiled-gate")
    calls = 0

    def forbidden_np_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("np.load must remain unreachable")

    monkeypatch.setattr(analysis.np, "load", forbidden_np_load)
    with pytest.raises(RuntimeError, match="hash drift"):
        analysis.validate_sealed_analysis_authorization(**fixture)
    assert calls == 0


def test_substituted_sealed_hash_fails_before_np_load(
    monkeypatch, tmp_path: Path
) -> None:
    fixture = _authorization_fixture(tmp_path)
    input_seal_path = fixture.pop("input_seal_path")
    seal = json.loads(input_seal_path.read_text(encoding="utf-8"))
    part_path = next(iter(seal["execution_part_hashes"]))
    seal["sealed_files"][part_path] = "0" * 64
    _write_json(input_seal_path, seal)
    calls = 0

    def forbidden_np_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("np.load must remain unreachable")

    monkeypatch.setattr(analysis.np, "load", forbidden_np_load)
    with pytest.raises(RuntimeError, match="hash drift"):
        analysis.validate_sealed_analysis_authorization(**fixture)
    assert calls == 0


def test_post_open_failure_capture_is_outcome_free_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "study"
    attempt = study / "attempts/v008"
    audit = attempt / "audit"
    freeze = attempt / "freeze"
    execution = attempt / "data/confirmation"
    audit.mkdir(parents=True)
    freeze.mkdir(parents=True)
    execution.mkdir(parents=True)
    state_path = study / "STATE.json"
    ledger_path = study / "RESEARCH_LEDGER.jsonl"
    state_path.write_text("{}\n", encoding="utf-8")
    ledger_path.write_text("{}\n", encoding="utf-8")
    whitening = tmp_path / "whitening.npz"
    pricing = freeze / "compiled_gate_manifest.json"
    seeds = attempt / "cohort_seed_ledger.json"
    thresholds = freeze / "compiled_gate.npz"
    for path in (
        audit / "confirmation_input_seal.json",
        audit / "pre_confirmation_package_seal.json",
        whitening,
        pricing,
        seeds,
        thresholds,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed-{path.name}".encode())
    analysis_result = attempt / "analysis_result.json"
    analysis_result.write_text(
        json.dumps({"process_valid": False}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        analysis,
        "read_verified_controller",
        lambda: {
            "active_attempt": "v008",
            "current_state": "SEALED_ANALYSIS",
            "confirmation_outcomes_opened_for_analysis": True,
            "confirmation_outcome_episodes_generated": 2_000,
            "confirmation_outcome_episodes_executed": 2_000,
            "ledger_event_count": 12,
            "ledger_head_sha256": "a" * 64,
            "last_verified_checkpoint": {"name": "input_seal"},
        },
    )
    path = analysis.capture_post_open_analysis_failure(
        RuntimeError("synthetic post-open failure"),
        execution_root=execution,
        whitening_path=whitening,
        pricing_path=pricing,
        seed_path=seeds,
        thresholds_path=thresholds,
        output_root=attempt,
    )
    assert path == audit / "analysis_execution_invalid.json"
    captured = json.loads(path.read_text(encoding="utf-8"))
    assert captured["confirmation_opened"] is True
    assert captured["outcome_values_recorded"] is False
    assert captured["integrity_failure"] is True
    assert captured["failure"]["exception_type"] == "RuntimeError"
    assert captured["error_type"] == "RuntimeError"
    assert captured["scientific_objects_changed"] is False
    assert captured["analysis_result_present"] is True
    assert captured["analysis_result"]["sha256"] == analysis.sha256_file(
        analysis_result
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        analysis.capture_post_open_analysis_failure(
            RuntimeError("second failure"),
            execution_root=execution,
            whitening_path=whitening,
            pricing_path=pricing,
            seed_path=seeds,
            thresholds_path=thresholds,
            output_root=attempt,
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_cli_interruption_never_creates_execution_invalid_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    arguments = SimpleNamespace(
        execution_root=tmp_path / "execution",
        whitening=tmp_path / "whitening",
        pricing=tmp_path / "pricing",
        seeds=tmp_path / "seeds",
        thresholds=tmp_path / "thresholds",
    )
    capture_calls: list[object] = []

    def interrupted(**kwargs):
        raise interruption

    monkeypatch.setattr(analysis, "run_sealed_analysis", interrupted)
    monkeypatch.setattr(
        analysis,
        "capture_post_open_analysis_failure",
        lambda *args, **kwargs: capture_calls.append((args, kwargs)),
    )
    with pytest.raises(type(interruption)):
        analysis.execute_analysis_cli(arguments)
    assert capture_calls == []


def test_integrity_false_cli_result_creates_failure_audit_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = SimpleNamespace(
        execution_root=tmp_path / "execution",
        whitening=tmp_path / "whitening",
        pricing=tmp_path / "pricing",
        seeds=tmp_path / "seeds",
        thresholds=tmp_path / "thresholds",
    )
    captured: list[Exception] = []
    monkeypatch.setattr(
        analysis,
        "run_sealed_analysis",
        lambda **kwargs: {
            "passed": True,
            "process_valid": False,
            "integrity_passed": False,
        },
    )
    monkeypatch.setattr(
        analysis,
        "capture_post_open_analysis_failure",
        lambda error, **kwargs: captured.append(error),
    )
    with pytest.raises(analysis.AnalysisIntegrityFailure):
        analysis.execute_analysis_cli(arguments)
    assert len(captured) == 1
    assert isinstance(captured[0], analysis.AnalysisIntegrityFailure)


def test_partial_derived_outputs_are_adopted_only_after_exact_recomputation(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "episode_metrics.npz"
    arrays = {
        "episode_slot": np.arange(3, dtype=np.int32),
        "contrast": np.array([0.1, 0.2, 0.3], dtype=np.float64),
    }
    first_npz_sha = analysis.persist_or_verify_npz(npz_path, arrays)
    original_npz = npz_path.read_bytes()
    assert analysis.persist_or_verify_npz(npz_path, arrays) == first_npz_sha
    assert npz_path.read_bytes() == original_npz
    changed_arrays = dict(arrays)
    changed_arrays["contrast"] = arrays["contrast"].copy()
    changed_arrays["contrast"][1] += 1e-12
    with pytest.raises(RuntimeError, match="exact recomputation"):
        analysis.persist_or_verify_npz(npz_path, changed_arrays)
    assert npz_path.read_bytes() == original_npz

    json_path = tmp_path / "compute_ledger.json"
    value = {"schema_version": 1, "exact_compute": {"flops": 123}}
    first_json_sha = analysis.persist_or_verify_json(json_path, value)
    original_json = json_path.read_bytes()
    assert analysis.persist_or_verify_json(json_path, value) == first_json_sha
    assert json_path.read_bytes() == original_json
    with pytest.raises(RuntimeError, match="semantic recomputation"):
        analysis.persist_or_verify_json(
            json_path,
            {"schema_version": 1, "exact_compute": {"flops": 124}},
        )
    assert json_path.read_bytes() == original_json
