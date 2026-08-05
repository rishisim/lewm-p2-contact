from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed_builder = _load("build_seed_ledger")
freshness = _load("verify_identifier_freshness")
power = _load("power_analysis")


def _empty_snapshot() -> dict:
    empty_digest = seed_builder._digest(set())
    return {
        "metadata_file_count": 0,
        "filename_count": 0,
        "numeric_identifier_count": 0,
        "string_identifier_count": 0,
        "numeric_identifier_set_sha256": empty_digest,
        "string_identifier_set_sha256": empty_digest,
        "metadata_path_set_sha256": empty_digest,
        "filename_path_set_sha256": empty_digest,
        "required_consumed_metadata": {},
        "required_consumed_metadata_complete": True,
        "missing_required_consumed_metadata": [],
        "safe_v3_source_config_plan_files": {"synthetic-safe-v3.py": "0" * 64},
        "safe_v3_numeric_literal_count": 1,
        "safe_v3_numeric_literal_set_sha256": "1" * 64,
        "safe_v3_string_identifier_count": 1,
        "safe_v3_string_identifier_set_sha256": "2" * 64,
        "v004_authoritative_prior_snapshot": {"sha256": "3" * 64},
        "v004_authoritative_prior_snapshot_cross_referenced": True,
        "v3_safe_source_config_plan_opened": True,
        "v3_test_target_cache_split_metrics_artifacts_opened": False,
        "binary_contents_opened": False,
        "outcome_array_contents_opened": False,
        "hdf5_contents_opened": False,
    }


@pytest.fixture(scope="module")
def ledger() -> dict:
    return seed_builder.build_seed_ledger(set(), set(), _empty_snapshot())


def test_ledger_has_exact_role_structure_and_prefix(ledger: dict) -> None:
    assert ledger["regime_order"] == list(seed_builder.REGIMES)
    assert ledger["role_order"] == list(seed_builder.ROLES)
    records = []
    for regime in seed_builder.REGIMES:
        roles = ledger["regimes"][regime]["roles"]
        for role in seed_builder.ROLES:
            assert len(roles[role]["primary"]) == seed_builder.PRIMARY_COUNTS[role]
            assert len(roles[role]["replacements"]) == 200
            assert [item["slot"] for item in roles[role]["primary"]] == list(
                range(seed_builder.PRIMARY_COUNTS[role])
            )
            assert [item["slot"] for item in roles[role]["replacements"]] == list(
                range(200)
            )
            records.extend(roles[role]["primary"])
            records.extend(roles[role]["replacements"])
        confirmation = roles["confirmation"]["primary"]
        short = seed_builder.REGIME_SHORT[regime]
        assert confirmation[0]["episode_id"] == f"drgv001-{short}-cf-p-0000"
        assert confirmation[-1]["episode_id"] == f"drgv001-{short}-cf-p-4499"

    assert len(records) == 24_424
    ids = [item["episode_id"] for item in records]
    numbers = [
        item[key]
        for item in records
        for key in ("env_seed", "policy_seed", "oracle_np_seed", "action_space_seed")
    ]
    rng_records = list(seed_builder.iter_rng_records(ledger["analysis_rng_ids"]))
    rng_ids = [item["rng_id"] for item in rng_records]
    assert len(ids) == len(set(ids))
    assert len(numbers) + len(rng_ids) == len(set(numbers + rng_ids))
    assert min(numbers + rng_ids) >= 4_100_000_000
    assert max(numbers + rng_ids) < 2**32


def test_analysis_seed_aliases_reuse_assigned_rng_ids(ledger: dict) -> None:
    aliases = ledger["analysis_seeds"]
    assigned = {
        item["rng_id"]
        for item in seed_builder.iter_rng_records(ledger["analysis_rng_ids"])
    }
    alias_ids = {aliases["joint_bootstrap_seed"]}
    alias_ids.update(aliases["histogram_seeds"].values())
    alias_ids.update(
        value
        for by_endpoint in aliases["seeded_comparator_seeds"].values()
        for value in by_endpoint.values()
    )
    assert alias_ids <= assigned
    assert aliases["joint_bootstrap_seed"] == 4_100_120_025
    bootstrap = ledger["analysis_rng_ids"]["bootstrap"]
    assert bootstrap["replicate_count"] == 20_000
    assert bootstrap["chunk_size_defining_rng_consumption"] == 250
    assert "default_rng(joint_master)" in bootstrap["replicate_derivation"]


def test_builder_rejects_any_prior_collision() -> None:
    with pytest.raises(RuntimeError, match="failed closed"):
        seed_builder.build_seed_ledger(
            {seed_builder.NUMERIC_NAMESPACE_BASE}, set(), _empty_snapshot()
        )
    with pytest.raises(RuntimeError, match="failed closed"):
        seed_builder.build_seed_ledger(
            set(), {"drgv001-np-ft-p-000"}, _empty_snapshot()
        )


def test_metadata_scanner_never_parses_binary_or_v3(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    allowed = runs / "allowed"
    allowed.mkdir(parents=True)
    (allowed / "meta.json").write_text(
        '{"seed": 123, "episode_id": "old-episode"}\n', encoding="utf-8"
    )
    # Invalid UTF-8 would fail immediately if either binary were opened as JSON.
    (allowed / "outcome.npz").write_bytes(b"\xff\xfe\x00not-json")
    (allowed / "released.hdf5").write_bytes(b"\xff\xfe\x00not-json")
    forbidden = runs / "lewm_adaptive_compute_v3"
    forbidden.mkdir()
    (forbidden / "targets.json").write_text(
        '{"forbidden_seed": 999}\n', encoding="utf-8"
    )
    (forbidden / "run_experiment.py").write_text(
        "SAFE_SOURCE_SEED = 777\n", encoding="utf-8"
    )

    numbers, strings, snapshot = seed_builder.scan_prior_metadata(
        runs,
        excluded_roots=(),
        required_relative_paths=(),
        require_authoritative=False,
    )
    assert 123 in numbers
    assert 777 in numbers
    assert 999 not in numbers
    assert "old-episode" in strings
    assert snapshot["binary_contents_opened"] is False
    assert snapshot["hdf5_contents_opened"] is False
    assert snapshot["v3_test_target_cache_split_metrics_artifacts_opened"] is False
    assert list(snapshot["safe_v3_source_config_plan_files"]) == [
        "lewm_adaptive_compute_v3/run_experiment.py"
    ]


def test_independent_verifier_recomputes_all_invariants(ledger: dict) -> None:
    result = freshness.verify_identifier_freshness(
        ledger,
        set(),
        set(),
        _empty_snapshot(),
    )
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["record_count"] == 24_424


def test_inherited_family8_yardstick_reproduces_3500() -> None:
    moments = power.load_frozen_power_rule_moments()
    yardstick = power.inherited_family8_yardstick(moments)
    assert yardstick["family_size"] == 8
    assert yardstick["per_claim_alpha"] == pytest.approx(0.05 / 8)
    assert yardstick["first_qualifying_episodes_per_regime"] == 3_500
    assert yardstick["reproduces_family8_inherited_yardstick"] is True
    assert yardstick["binding"] is False


IDENTITY = {
    "selected_candidate_id": "candidate_07",
    "selected_candidate_object_sha256": "a" * 64,
    "fit_lock_sha256": "b" * 64,
    "selection_ledger_sha256": "c" * 64,
    "gate_freeze_sha256": "d" * 64,
}


def _claim_summary(count: int, *, role: str, mean: float, sd: float) -> dict:
    return {
        "schema_version": 1,
        "attempt": "v008",
        "role": role,
        "co_primary_only": True,
        "auxiliary_robust_whitening_excluded": True,
        **IDENTITY,
        "claims": {
            regime: {
                endpoint: {
                    "episode_count": count,
                    "mean": mean,
                    "sd_ddof1": sd,
                }
                for endpoint in power.ENDPOINTS
            }
            for regime in power.REGIMES
        }
    }


def test_binding_rule_uses_min_positive_mean_and_max_upper_sd() -> None:
    fit = _claim_summary(300, role="fit", mean=1.0, sd=0.10)
    selection = _claim_summary(500, role="selection", mean=0.80, sd=0.08)
    moments = power.load_frozen_power_rule_moments()
    result = power.compute_binding_power(fit, selection, moments)
    claim = result["claim_design_inputs"]["native_plan"]["raw_vs_analytic"]
    assert claim["delta"] == pytest.approx(0.35 * 0.80)
    assert claim["sigma"] == pytest.approx(
        max(
            power.upper_95_sd(0.10, 300),
            power.upper_95_sd(0.08, 500),
            moments["raw_vs_analytic"]["sd_ddof1"],
        )
    )
    assert result["feasible"] is True
    assert result["selected_confirmation_episodes_per_regime"] == 500
    assert result["selected_confirmation_slots_per_regime"] == [0, 499]
    assert result["confirmation_episode_count_per_regime"] == {
        regime: 500 for regime in power.REGIMES
    }


def test_nonpositive_claim_produces_explicit_infeasible_no_outcome() -> None:
    fit = _claim_summary(300, role="fit", mean=1.0, sd=0.10)
    selection = _claim_summary(500, role="selection", mean=0.80, sd=0.08)
    selection["claims"]["markov_oracle"]["raw_vs_analytic"]["mean"] = 0.0
    result = power.compute_binding_power(
        fit, selection, power.load_frozen_power_rule_moments()
    )
    assert result["feasible"] is False
    assert result["passed"] is False
    assert result["decision"] == "power_infeasible_no_confirmation"
    assert result["confirmation_generation_authorized_by_power"] is False
    assert result["terminal_label_if_infeasible"] == "domain_robust_gate_failed"
    assert result["selected_confirmation_episodes_per_regime"] is None
    assert result["fresh_confirmation_outcomes_opened"] == 0


def test_positive_claims_with_no_qualifying_grid_are_explicitly_infeasible() -> None:
    fit = _claim_summary(300, role="fit", mean=1e-12, sd=1.0)
    selection = _claim_summary(500, role="selection", mean=1e-12, sd=1.0)
    result = power.compute_binding_power(
        fit, selection, power.load_frozen_power_rule_moments()
    )
    assert result["all_eight_claims_have_positive_fit_and_selection_means"] is True
    assert not any(record["eligible"] for record in result["power_grid"])
    assert result["selected_grid_record"] is None
    assert result["feasible"] is False
    assert result["passed"] is False
    assert result["decision"] == "power_infeasible_no_confirmation"
    assert result["confirmation_generation_authorized_by_power"] is False
    assert result["selected_confirmation_episodes_per_regime"] is None


def test_summary_role_counts_are_fail_closed() -> None:
    fit = _claim_summary(299, role="fit", mean=1.0, sd=0.1)
    selection = _claim_summary(500, role="selection", mean=1.0, sd=0.1)
    with pytest.raises(ValueError, match="count 299"):
        power.compute_binding_power(
            fit, selection, power.load_frozen_power_rule_moments()
        )


@pytest.mark.parametrize("bad_count", [300.0, 300.9, True, "300"])
def test_episode_count_requires_exact_non_bool_json_integer(bad_count: object) -> None:
    fit = _claim_summary(300, role="fit", mean=1.0, sd=0.1)
    selection = _claim_summary(500, role="selection", mean=1.0, sd=0.1)
    fit["claims"]["native_plan"]["raw_vs_analytic"]["episode_count"] = bad_count
    with pytest.raises(ValueError, match="exact JSON integer"):
        power.compute_binding_power(
            fit, selection, power.load_frozen_power_rule_moments()
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("mean", True), ("mean", "1.0"), ("sd_ddof1", False), ("sd_ddof1", "0.1")],
)
def test_moments_require_non_bool_json_numbers(field: str, bad_value: object) -> None:
    fit = _claim_summary(300, role="fit", mean=1.0, sd=0.1)
    selection = _claim_summary(500, role="selection", mean=1.0, sd=0.1)
    fit["claims"]["native_plan"]["raw_vs_analytic"][field] = bad_value
    with pytest.raises(ValueError, match="non-bool JSON numbers"):
        power.compute_binding_power(
            fit, selection, power.load_frozen_power_rule_moments()
        )


def test_upper_sd_bound_is_conservative_and_finite() -> None:
    bound = power.upper_95_sd(2.0, 300)
    assert math.isfinite(bound)
    assert bound > 2.0


def test_power_runtime_has_no_prior_attempt_path_and_rule_is_hash_pinned(
    tmp_path: Path,
) -> None:
    source = (ROOT / "power_analysis.py").read_text(encoding="utf-8").lower()
    assert "runs/lewm_v5_generalization/attempts/v005" not in source
    assert "lewm_v5_generalization" not in source
    copied = tmp_path / "power_rule.json"
    shutil.copyfile(power.POWER_RULE_PATH, copied)
    assert power.load_frozen_power_rule_moments(copied) == (
        power.load_frozen_power_rule_moments()
    )
    document = json.loads(copied.read_text(encoding="utf-8"))
    document["status"] = "tampered"
    copied.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="power-rule hash drift"):
        power.load_frozen_power_rule_moments(copied)


def test_power_summary_identity_is_role_and_object_fail_closed() -> None:
    fit = _claim_summary(300, role="fit", mean=1.0, sd=0.1)
    selection = _claim_summary(500, role="selection", mean=1.0, sd=0.1)
    selection["selected_candidate_object_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="identity drift"):
        power.compute_binding_power(
            fit, selection, power.load_frozen_power_rule_moments()
        )
    selection["selected_candidate_object_sha256"] = IDENTITY[
        "selected_candidate_object_sha256"
    ]
    selection["auxiliary_robust_whitening_excluded"] = False
    with pytest.raises(ValueError, match="identity contract"):
        power.compute_binding_power(
            fit, selection, power.load_frozen_power_rule_moments()
        )


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_power_cli_binds_gate_locks_and_persists_only_repo_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    attempt = repo / "runs/lewm_domain_robust_gate/attempts/v008"
    rule = attempt / "power_rule.json"
    rule.parent.mkdir(parents=True)
    shutil.copyfile(power.POWER_RULE_PATH, rule)
    monkeypatch.setattr(power, "REPO_ROOT", repo)

    candidate_id = "candidate_07"
    object_sha256 = "1" * 64
    fit_lock = _write_json(
        attempt / "fit/fit_lock.json",
        {
            "schema_version": 1,
            "status": "all_24_candidates_fit_compiled_and_locked_before_selection_open",
            "selection_input_opened_before_lock": False,
        },
    )
    fit_lock_sha256 = power._sha256(fit_lock)
    selection_ledger = _write_json(
        attempt / "selection/selection_ledger.json",
        {
            "schema_version": 1,
            "status": "selection_complete_no_selected_head_refit",
            "selected_candidate_id": candidate_id,
            "selected_candidate_index": 7,
            "selected_head_refit_after_selection": False,
            "fit_lock_sha256": fit_lock_sha256,
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "candidate_object_sha256": object_sha256,
                    "eligible": True,
                }
            ],
        },
    )
    selection_sha256 = power._sha256(selection_ledger)
    gate_freeze = _write_json(
        attempt / "freeze/gate_freeze.json",
        {
            "schema_version": 1,
            "status": "selected_gate_frozen_before_smoke_or_confirmation",
            "created_unix_ns": 123,
            "selected_candidate_id": candidate_id,
            "selected_candidate_object_sha256": object_sha256,
            "fit_lock_sha256": fit_lock_sha256,
            "selection_ledger_sha256": selection_sha256,
            "selected_head_refit_after_selection": False,
            "confirmation_episodes_at_freeze": 0,
        },
    )
    identity = {
        "selected_candidate_id": candidate_id,
        "selected_candidate_object_sha256": object_sha256,
        "fit_lock_sha256": fit_lock_sha256,
        "selection_ledger_sha256": selection_sha256,
        "gate_freeze_sha256": power._sha256(gate_freeze),
    }
    fit_summary = _claim_summary(300, role="fit", mean=1.0, sd=0.1)
    selection_summary = _claim_summary(
        500, role="selection", mean=0.8, sd=0.08
    )
    fit_summary.update(identity)
    selection_summary.update(identity)
    fit_summary_path = _write_json(
        attempt / "metrics/fit_power_summary.json", fit_summary
    )
    selection_summary_path = _write_json(
        attempt / "metrics/selection_power_summary.json", selection_summary
    )
    output = attempt / "power_analysis.json"

    api_output = attempt / "power_analysis_api.json"
    api_result = power.run_binding_power(
        fit_summary_path=fit_summary_path,
        selection_summary_path=selection_summary_path,
        fit_lock_path=fit_lock,
        selection_ledger_path=selection_ledger,
        gate_freeze_path=gate_freeze,
        power_rule_path=rule,
        output_path=api_output,
    )
    assert json.loads(api_output.read_text(encoding="utf-8")) == api_result
    assert api_result["created_unix_ns"] > 123
    with pytest.raises(RuntimeError, match="already exists"):
        power.run_binding_power(
            fit_summary_path=fit_summary_path,
            selection_summary_path=selection_summary_path,
            fit_lock_path=fit_lock,
            selection_ledger_path=selection_ledger,
            gate_freeze_path=gate_freeze,
            power_rule_path=rule,
            output_path=api_output,
        )

    assert (
        power.main(
            [
                "--fit-summary",
                str(fit_summary_path),
                "--selection-summary",
                str(selection_summary_path),
                "--power-rule",
                str(rule),
                "--fit-lock",
                str(fit_lock),
                "--selection-ledger",
                str(selection_ledger),
                "--gate-freeze",
                str(gate_freeze),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["output"] == output.relative_to(repo).as_posix()
    assert result["selected_gate_identity"] == identity
    assert result["created_unix_ns"] > 123
    assert set(result["inputs"]) == {
        "fit_summary",
        "selection_summary",
        "power_rule",
        "fit_lock",
        "selection_ledger",
        "gate_freeze",
    }
    for record in result["inputs"].values():
        assert not Path(record["path"]).is_absolute()
        assert (repo / record["path"]).is_file()
    assert result["inputs"]["power_rule"]["expected_sha256"] == (
        power.POWER_RULE_SHA256
    )

    stale = dict(identity)
    stale["gate_freeze_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="artifact hash drift"):
        power.validate_identity_artifacts(
            stale,
            fit_lock_path=fit_lock,
            selection_ledger_path=selection_ledger,
            gate_freeze_path=gate_freeze,
        )
