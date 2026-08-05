"""Synthetic-only tests for the v008 dispatcher, contracts, and preseals."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ATTEMPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTEMPT_ROOT))

import build_manifest
import fit_select
import launcher
import power_analysis
import preseal
import runtime_contract


def _zero_state() -> dict[str, object]:
    return {
        "fit_outcome_episodes": 0,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }


def test_exact_terminal_v004_runtime_contract_is_authenticated() -> None:
    contract = runtime_contract.load_v5_runtime_contract()
    assert contract["package_version"] == "v004"
    assert contract["runtimes"]["evaluation"]["sys_executable"] == str(
        runtime_contract.EVALUATION_PYTHON
    )
    assert contract["runtimes"]["generation"]["sys_executable"] == str(
        runtime_contract.GENERATION_PYTHON
    )
    observed = runtime_contract.verify_here(
        "qualification", include_external_hashes=False
    )
    assert observed["passed"] is True
    assert observed["seed_tuples_consumed"] == 0
    assert observed["output_paths_created"] == 0


def test_dispatcher_role_and_mps_contract_are_fail_closed() -> None:
    command = launcher.preflight_command("generation", require_mps=False)
    assert command[0] == str(runtime_contract.GENERATION_PYTHON)
    assert command[-2:] == ["--role", "generation"]
    execute = launcher.target_command("analysis", "analysis.py", [])
    assert execute[0] == str(runtime_contract.EVALUATION_PYTHON)
    with pytest.raises(runtime_contract.RuntimeContractError):
        runtime_contract.require_mps_device("cpu")
    with pytest.raises(runtime_contract.RuntimeContractError):
        runtime_contract.require_mps_device("auto")
    with pytest.raises(runtime_contract.RuntimeContractError):
        runtime_contract.runtime_for_role("manual")
    with pytest.raises(launcher.DispatchError):
        launcher.target_command("analysis", "arbitrary.py", [])


def test_fit_selection_freeze_public_commands_dispatch_exact_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def record(role: str, script: str, arguments=(), **_kwargs) -> None:
        calls.append((role, script, tuple(arguments)))

    monkeypatch.setattr(launcher, "run_dispatched", record)
    assert launcher.main(["fit-lock"]) == 0
    assert launcher.main(["select-gate"]) == 0
    assert launcher.main(["freeze-gate"]) == 0
    assert calls == [
        ("fit", "launcher.py", ("_worker-fit-lock",)),
        ("selection", "launcher.py", ("_worker-select-gate",)),
        ("selection", "launcher.py", ("_worker-freeze-gate",)),
    ]


def test_compile_and_power_are_state_gated_inside_exact_eval_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        launcher,
        "run_dispatched",
        lambda role, script, arguments=(), **_kwargs: calls.append(
            (role, script, tuple(arguments))
        ),
    )
    assert launcher.main(["compile-gate"]) == 0
    assert launcher.main(["power"]) == 0
    assert calls == [
        (
            "evaluation",
            "launcher.py",
            (
                "_worker-compile-gate",
                "--source",
                "freeze/gate_fit.npz",
                "--output",
                "freeze/compiled_gate.npz",
                "--manifest",
                "freeze/compiled_gate_manifest.json",
            ),
        ),
        ("evaluation", "launcher.py", ("_worker-power",)),
    ]

    monkeypatch.setattr(
        preseal.study_common,
        "read_verified_controller",
        lambda: {
            "active_attempt": "v008",
            "current_state": "GATE_FREEZE",
            "confirmation_terminal": False,
        },
    )
    assert launcher._require_controller_state("GATE_FREEZE")["active_attempt"] == "v008"
    with pytest.raises(launcher.DispatchError, match="requires controller state"):
        launcher._require_controller_state("FIT_LOCK")


def test_terminal_workflow_commands_dispatch_exact_sealed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        launcher,
        "run_dispatched",
        lambda role, script, arguments=(), **_kwargs: calls.append(
            (role, script, tuple(arguments))
        ),
    )
    assert launcher.main(
        ["terminal-manifest", "--contract", "verifier_contract_no_candidate.json"]
    ) == 0
    assert launcher.main(["terminal-finalize"]) == 0
    assert launcher.main(["terminal-reports"]) == 0
    assert launcher.main(["terminal-complete"]) == 0
    assert calls == [
        (
            "sealing",
            "terminal_workflow.py",
            ("manifest", "--contract", "verifier_contract_no_candidate.json"),
        ),
        ("sealing", "terminal_workflow.py", ("finalize",)),
        ("sealing", "terminal_workflow.py", ("reports",)),
        ("sealing", "terminal_workflow.py", ("complete",)),
    ]
    with pytest.raises(launcher.DispatchError):
        launcher.main(["terminal-manifest", "--contract", "../contract.json"])


def test_resumable_workflow_commands_dispatch_exact_evaluation_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        launcher,
        "run_dispatched",
        lambda role, script, arguments=(), **_kwargs: calls.append(
            (role, script, tuple(arguments))
        ),
    )
    assert launcher.main(["workflow", "status"]) == 0
    assert launcher.main(["workflow", "step"]) == 0
    assert launcher.main(["workflow", "verify-role", "selection"]) == 0
    assert launcher.main(["workflow", "resume"]) == 0
    assert launcher.main(
        ["workflow", "resume", "--max-transitions", "7"]
    ) == 0
    assert calls == [
        ("evaluation", "workflow.py", ("status",)),
        ("evaluation", "workflow.py", ("step",)),
        ("evaluation", "workflow.py", ("verify-role", "selection")),
        (
            "evaluation",
            "workflow.py",
            ("resume", "--max-transitions", "32"),
        ),
        (
            "evaluation",
            "workflow.py",
            ("resume", "--max-transitions", "7"),
        ),
    ]
    with pytest.raises(launcher.DispatchError):
        launcher.main(["workflow", "verify-role"])
    with pytest.raises(launcher.DispatchError):
        launcher.main(["workflow", "resume", "--max-transitions", "0"])


def test_all_mode_specific_verifier_contracts_are_prospectively_required() -> None:
    names = {
        "verifier_contract.json",
        "verifier_contract_no_candidate.json",
        "verifier_contract_power_infeasible.json",
    }
    assert names <= build_manifest.PRE_DATA_STATIC_RELATIVE_PATHS
    assert {path.name for path in preseal.VERIFIER_CONTRACT_PATHS} == names
    validated = preseal.validate_verifier_contracts()
    assert validated["passed"] is True
    assert set(validated["modes"].values()) == {
        "confirmation",
        "no_candidate",
        "power_infeasible",
    }


def test_manifest_records_bytes_hashes_and_rejects_local_symlinks(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "source.py"
    regular.write_text("value = 1\n", encoding="utf-8")
    record = build_manifest._record(regular, local_source=True)
    assert record["bytes"] == len(b"value = 1\n")
    assert record["regular_file"] is True
    assert record["contents_decoded"] is False
    linked = tmp_path / "linked.py"
    linked.symlink_to(regular)
    with pytest.raises(build_manifest.ManifestError):
        build_manifest._record(linked, local_source=True)


def test_stage_file_collection_rejects_supplied_directory_symlink_before_rglob(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-stage"
    real_root.mkdir()
    (real_root / "apparently-valid.json").write_text("{}\n", encoding="utf-8")
    linked_root = tmp_path / "linked-stage"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(
        preseal.PresealError, match="lexical ancestor symlink forbidden"
    ):
        preseal._files_under(linked_root)

    real_parent = tmp_path / "real-parent"
    nested = real_parent / "nested-stage"
    nested.mkdir(parents=True)
    (nested / "also-apparently-valid.json").write_text("{}\n", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(
        preseal.PresealError, match="lexical ancestor symlink forbidden"
    ):
        preseal._files_under(linked_parent / "nested-stage")


def test_study_root_controller_is_an_explicit_local_predata_source() -> None:
    relative = "runs/lewm_domain_robust_gate/program.py"
    assert relative in build_manifest.PRE_DATA_REPOSITORY_RELATIVE_PATHS
    controller = build_manifest.REPO_ROOT / relative
    assert controller in build_manifest.collect_pre_data_source_paths()
    record = build_manifest._record(controller, local_source=True)
    assert record["symlink"] is False
    assert record["sha256"] == runtime_contract.sha256_file(controller)


def test_pre_data_manifest_ignores_later_stage_top_level_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempts/v008"
    attempt.mkdir(parents=True)
    source = attempt / "analysis.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(build_manifest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build_manifest, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(
        build_manifest,
        "PRE_DATA_STATIC_RELATIVE_PATHS",
        frozenset({"analysis.py"}),
    )
    monkeypatch.setattr(build_manifest, "PRE_DATA_AUDIT_INPUTS", frozenset())
    monkeypatch.setattr(
        build_manifest, "PRE_DATA_REPOSITORY_RELATIVE_PATHS", frozenset()
    )
    monkeypatch.setattr(build_manifest, "_prospective_episode_ids", lambda: {})
    manifest = build_manifest._manifest_for_paths(
        build_manifest.collect_pre_data_source_paths(),
        label="v008_pre_data_normative_source_and_inherited_runtime",
        include_inherited_runtime=False,
    )
    manifest["source_closure_policy"] = build_manifest.source_closure_policy()
    (attempt / "power_analysis.json").write_text("{}\n", encoding="utf-8")
    (attempt / "decision.json").write_text("{}\n", encoding="utf-8")
    assert build_manifest.verify_manifest(manifest)["passed"] is True
    assert set(manifest["local_files"]) == {
        "attempts/v008/analysis.py"
    }


def test_pre_data_manifest_rejects_unlisted_python_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempts/v008"
    attempt.mkdir(parents=True)
    (attempt / "analysis.py").write_text("VALUE = 1\n", encoding="utf-8")
    (attempt / "omitted.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(build_manifest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build_manifest, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(
        build_manifest,
        "PRE_DATA_STATIC_RELATIVE_PATHS",
        frozenset({"analysis.py"}),
    )
    monkeypatch.setattr(build_manifest, "PRE_DATA_AUDIT_INPUTS", frozenset())
    monkeypatch.setattr(
        build_manifest, "PRE_DATA_REPOSITORY_RELATIVE_PATHS", frozenset()
    )
    monkeypatch.setattr(build_manifest, "_prospective_episode_ids", lambda: {})
    assert build_manifest.unexpected_python_source_paths() == ["omitted.py"]
    with pytest.raises(build_manifest.ManifestError, match="unexpected_python"):
        build_manifest.build_pre_data_manifest()


def test_forbidden_ast_key_access_is_detected_but_guard_literals_are_not() -> None:
    accessed = ast.parse("value = info['contact']\n")
    guarded = ast.parse("FORBIDDEN = {'contact', 'reward'}\n")
    assert preseal._key_accesses(accessed) == [(1, "contact")]
    assert preseal._key_accesses(guarded) == []
    source = preseal.validate_source_ast()
    assert source["passed"] is True
    assert source["checks"]["raw_input_allowlist_exact"] is True


def test_zero_outcome_state_rejects_any_role_count() -> None:
    assert preseal.validate_zero_outcome_state(
        _zero_state(), require_no_data_files=False
    )["passed"]
    bad = _zero_state()
    bad["selection_outcome_episodes"] = 1
    with pytest.raises(preseal.PresealError):
        preseal.validate_zero_outcome_state(bad, require_no_data_files=False)


def test_preseal_state_uses_verified_controller_not_direct_state_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _zero_state() | {
        "active_attempt": "v008",
        "confirmation_terminal": False,
    }
    calls = 0

    def status() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return dict(verified)

    monkeypatch.setattr(preseal.study_common, "read_verified_controller", status)
    assert preseal._state() == verified
    assert calls == 1
    monkeypatch.setattr(
        preseal.study_common,
        "read_verified_controller",
        lambda: (_ for _ in ()).throw(RuntimeError("ledger drift")),
    )
    with pytest.raises(preseal.PresealError, match="controller state/ledger"):
        preseal._state()


def test_python_cache_hygiene_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study = tmp_path / "runs/study"
    study.mkdir(parents=True)
    monkeypatch.setattr(preseal, "STUDY_ROOT", study)
    monkeypatch.setattr(preseal, "REPO_ROOT", tmp_path)
    assert preseal.validate_no_python_caches()["passed"] is True
    cache = study / "attempts/v008/__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.pyc").write_bytes(b"cache")
    with pytest.raises(preseal.PresealError, match="Python cache artifacts"):
        preseal.validate_no_python_caches()


def test_power_input_paths_resolve_from_repository_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    expected = repository / "runs/study/attempts/v008/metrics/fit_power_summary.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(preseal, "REPO_ROOT", repository)
    monkeypatch.chdir(tmp_path)
    relative = expected.relative_to(repository).as_posix()
    item = {"path": relative, "sha256": runtime_contract.sha256_file(expected)}
    assert preseal._recorded_repo_input_matches(item, expected) is True
    assert preseal._recorded_repo_input_matches(
        {"path": str(expected), "sha256": item["sha256"]}, expected
    ) is False
    assert preseal._recorded_repo_input_matches(
        {"path": "../escape.json", "sha256": item["sha256"]}, expected
    ) is False


def test_sorted_json_claim_and_confirmation_maps_preserve_exact_schema() -> None:
    claims = {
        regime: {
            "raw_vs_analytic": {"episode_count": 300},
            "fixed_whitened_vs_analytic": {"episode_count": 300},
        }
        for regime in preseal.study_common.REGIME_ORDER
    }
    summary = json.loads(json.dumps({"claims": claims}, sort_keys=True))
    assert tuple(summary["claims"]) != preseal.study_common.REGIME_ORDER
    assert preseal._summary_claim_schema_exact(summary) is True
    del summary["claims"]["markov_oracle"]
    assert preseal._summary_claim_schema_exact(summary) is False

    counts = json.loads(
        json.dumps(
            {regime: 1500 for regime in preseal.study_common.REGIME_ORDER},
            sort_keys=True,
        )
    )
    assert tuple(counts) != preseal.study_common.REGIME_ORDER
    assert preseal._validated_confirmation_count_map(counts) == {
        regime: 1500 for regime in preseal.study_common.REGIME_ORDER
    }
    bad_bool = dict(counts)
    bad_bool["native_plan"] = True
    with pytest.raises(preseal.PresealError, match="not an integer"):
        preseal._validated_confirmation_count_map(bad_bool)
    bad_grid = dict(counts)
    bad_grid["native_plan"] = 1501
    with pytest.raises(preseal.PresealError, match="off the frozen grid"):
        preseal._validated_confirmation_count_map(bad_grid)
    bad_common = dict(counts)
    bad_common["native_plan"] = 2000
    with pytest.raises(preseal.PresealError, match="not common"):
        preseal._validated_confirmation_count_map(bad_common)


def test_power_provenance_binds_rule_locks_freeze_and_selected_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    attempt = repository / "runs/study/attempts/v008"
    paths = {
        "fit_summary": attempt / "metrics/fit_power_summary.json",
        "selection_summary": attempt / "metrics/selection_power_summary.json",
        "power_rule": attempt / "power_rule.json",
        "fit_lock": attempt / "fit/fit_lock.json",
        "selection_ledger": attempt / "selection/selection_ledger.json",
        "gate_freeze": attempt / "freeze/gate_freeze.json",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "power_rule":
            path.write_bytes(power_analysis.POWER_RULE_PATH.read_bytes())
        else:
            path.write_text(json.dumps({"artifact": name}) + "\n", encoding="utf-8")
    monkeypatch.setattr(preseal, "REPO_ROOT", repository)

    def record(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(repository).as_posix(),
            "sha256": runtime_contract.sha256_file(path),
        }

    candidate_id = "candidate-01"
    object_hash = "a" * 64
    fit_lock_hash = runtime_contract.sha256_file(paths["fit_lock"])
    selection_hash = runtime_contract.sha256_file(paths["selection_ledger"])
    freeze_hash = runtime_contract.sha256_file(paths["gate_freeze"])
    inputs = {name: record(path) for name, path in paths.items()}
    inputs["power_rule"].update(
        {
            "expected_sha256": power_analysis.POWER_RULE_SHA256,
            "endpoint_moments_object_sha256": (
                power_analysis.POWER_RULE_MOMENTS_SHA256
            ),
        }
    )
    power = {
        "inputs": inputs,
        "selected_gate_identity": {
            "selected_candidate_id": candidate_id,
            "selected_candidate_object_sha256": object_hash,
            "fit_lock_sha256": fit_lock_hash,
            "selection_ledger_sha256": selection_hash,
            "gate_freeze_sha256": freeze_hash,
        },
    }

    def checks(value: dict[str, object]) -> dict[str, bool]:
        return preseal._power_provenance_checks(
            value,
            fit_summary_path=paths["fit_summary"],
            selection_summary_path=paths["selection_summary"],
            power_rule_path=paths["power_rule"],
            fit_lock_path=paths["fit_lock"],
            selection_ledger_path=paths["selection_ledger"],
            gate_freeze_path=paths["gate_freeze"],
            selected_candidate_id=candidate_id,
            selected_candidate_object_sha256=object_hash,
        )

    assert all(checks(power).values())
    tampered_rule = json.loads(json.dumps(power))
    tampered_rule["inputs"]["power_rule"][
        "endpoint_moments_object_sha256"
    ] = "0" * 64
    assert checks(tampered_rule)["power_rule_moments_object_sha256_exact"] is False
    tampered_lock = json.loads(json.dumps(power))
    tampered_lock["inputs"]["fit_lock"]["sha256"] = "0" * 64
    assert checks(tampered_lock)["fit_lock_record_exact"] is False
    tampered_identity = json.loads(json.dumps(power))
    tampered_identity["selected_gate_identity"]["gate_freeze_sha256"] = "0" * 64
    assert checks(tampered_identity)["selected_gate_identity_exact"] is False


def _preseal_power_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    repository = tmp_path / "repo"
    attempt = repository / "runs/lewm_domain_robust_gate/attempts/v008"
    monkeypatch.setattr(preseal, "REPO_ROOT", repository)
    monkeypatch.setattr(preseal, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(power_analysis, "REPO_ROOT", repository)

    fitted = attempt / "fit/fitted_candidates.npz"
    fit_lock = attempt / "fit/fit_lock.json"
    selection_ledger = attempt / "selection/selection_ledger.json"
    gate_fit = attempt / "freeze/gate_fit.npz"
    compiled_gate = attempt / "freeze/compiled_gate.npz"
    gate_freeze = attempt / "freeze/gate_freeze.json"
    fit_summary_path = attempt / "metrics/fit_power_summary.json"
    selection_summary_path = attempt / "metrics/selection_power_summary.json"
    power_rule = attempt / "power_rule.json"
    power_path = attempt / "power_analysis.json"

    def write(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    fitted.parent.mkdir(parents=True, exist_ok=True)
    fitted.write_bytes(b"synthetic fitted candidate source\n")
    gate_fit.parent.mkdir(parents=True, exist_ok=True)
    gate_fit.write_bytes(b"synthetic gate fit\n")
    compiled_gate.write_bytes(b"synthetic compiled gate\n")
    power_rule.parent.mkdir(parents=True, exist_ok=True)
    power_rule.write_bytes(power_analysis.POWER_RULE_PATH.read_bytes())

    candidate_id = "candidate-07"
    candidate_object = "a" * 64
    write(
        fit_lock,
        {
            "status": "all_24_candidates_fit_compiled_and_locked_before_selection_open",
            "selection_input_opened_before_lock": False,
            "fitted_candidates_sha256": runtime_contract.sha256_file(fitted),
        },
    )
    fit_lock_hash = runtime_contract.sha256_file(fit_lock)
    write(
        selection_ledger,
        {
            "status": "selection_complete_no_selected_head_refit",
            "selected_candidate_id": candidate_id,
            "selected_candidate_index": 0,
            "selected_head_refit_after_selection": False,
            "fit_lock_sha256": fit_lock_hash,
            "fitted_candidates_sha256": runtime_contract.sha256_file(fitted),
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "candidate_object_sha256": candidate_object,
                    "eligible": True,
                }
            ],
        },
    )
    selection_hash = runtime_contract.sha256_file(selection_ledger)
    write(
        gate_freeze,
        {
            "status": "selected_gate_frozen_before_smoke_or_confirmation",
            "created_unix_ns": 100,
            "selected_candidate_id": candidate_id,
            "selected_candidate_object_sha256": candidate_object,
            "fit_lock_sha256": fit_lock_hash,
            "selection_ledger_sha256": selection_hash,
            "gate_fit_sha256": runtime_contract.sha256_file(gate_fit),
            "compiled_gate_sha256": runtime_contract.sha256_file(compiled_gate),
            "selected_head_refit_after_selection": False,
            "confirmation_episodes_at_freeze": 0,
        },
    )
    identity = {
        "selected_candidate_id": candidate_id,
        "selected_candidate_object_sha256": candidate_object,
        "fit_lock_sha256": fit_lock_hash,
        "selection_ledger_sha256": selection_hash,
        "gate_freeze_sha256": runtime_contract.sha256_file(gate_freeze),
    }
    common = {
        "schema_version": 1,
        "selected_candidate_id": candidate_id,
        "selected_candidate_object_sha256": candidate_object,
        "co_primary_only": True,
        "auxiliary_robust_whitening_excluded": True,
    }
    fit_source = {
        **common,
        "role": "fit",
        "claims": {
            regime: {
                endpoint: {"episode_count": 300, "mean": 1.0, "sd_ddof1": 0.1}
                for endpoint in power_analysis.ENDPOINTS
            }
            for regime in power_analysis.REGIMES
        },
    }
    selection_source = {
        **common,
        "role": "selection",
        "claims": {
            regime: {
                endpoint: {"episode_count": 500, "mean": 0.8, "sd_ddof1": 0.08}
                for endpoint in power_analysis.ENDPOINTS
            }
            for regime in power_analysis.REGIMES
        },
    }
    fit_summary = copy.deepcopy(fit_source)
    selection_summary = copy.deepcopy(selection_source)
    fit_summary.update({"attempt": "v008", **identity})
    selection_summary.update({"attempt": "v008", **identity})
    write(fit_summary_path, fit_summary)
    write(selection_summary_path, selection_summary)
    monkeypatch.setattr(
        preseal.fit_select,
        "load_fitted_candidates",
        lambda _path: {"synthetic": True},
    )
    monkeypatch.setattr(
        preseal.fit_select,
        "selected_power_summaries",
        lambda _fitted, _ledger: (
            copy.deepcopy(fit_source),
            copy.deepcopy(selection_source),
        ),
    )

    power = power_analysis.compute_binding_power(
        fit_summary,
        selection_summary,
        power_analysis.load_frozen_power_rule_moments(power_rule),
    )
    identity_artifacts = power_analysis.validate_identity_artifacts(
        identity,
        fit_lock_path=fit_lock,
        selection_ledger_path=selection_ledger,
        gate_freeze_path=gate_freeze,
    )

    def record(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(repository).as_posix(),
            "sha256": runtime_contract.sha256_file(path),
        }

    power["created_unix_ns"] = 101
    power["inputs"] = {
        "fit_summary": record(fit_summary_path),
        "selection_summary": record(selection_summary_path),
        "power_rule": {
            **record(power_rule),
            "expected_sha256": power_analysis.POWER_RULE_SHA256,
            "endpoint_moments_object_sha256": (
                power_analysis.POWER_RULE_MOMENTS_SHA256
            ),
        },
        **identity_artifacts,
    }
    write(power_path, power)
    return {
        "paths": [fitted, fit_lock, selection_ledger, gate_freeze],
        "power": power,
        "power_path": power_path,
        "fit_summary_path": fit_summary_path,
        "selection_summary_path": selection_summary_path,
        "pre_confirmation_seal": attempt / "audit/pre_confirmation_package_seal.json",
        "write": write,
    }


def _tamper_preseal_power(tree: dict[str, object], mode: str) -> None:
    power = copy.deepcopy(tree["power"])
    power_path = tree["power_path"]
    write = tree["write"]
    assert isinstance(power, dict) and isinstance(power_path, Path) and callable(write)
    claim = power["claim_design_inputs"]["native_plan"]["raw_vs_analytic"]
    if mode == "n":
        power["selected_confirmation_episodes_per_regime"] = 1_000
        power["confirmation_episode_count_per_regime"] = {
            regime: 1_000 for regime in power_analysis.REGIMES
        }
        power["selected_confirmation_slots_per_regime"] = [0, 999]
        power["selected_grid_record"] = copy.deepcopy(power["power_grid"][1])
    elif mode == "stored_fit_mean":
        path = tree["fit_summary_path"]
        assert isinstance(path, Path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["claims"]["native_plan"]["raw_vs_analytic"]["mean"] = 0.5
        write(path, summary)
        power["inputs"]["fit_summary"]["sha256"] = runtime_contract.sha256_file(path)
    elif mode == "stored_selection_sd":
        path = tree["selection_summary_path"]
        assert isinstance(path, Path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["claims"]["native_plan"]["raw_vs_analytic"]["sd_ddof1"] = 0.8
        write(path, summary)
        power["inputs"]["selection_summary"]["sha256"] = (
            runtime_contract.sha256_file(path)
        )
    elif mode == "ucl":
        claim["fit_sd_upper_95"] = float(claim["fit_sd_upper_95"]) + 1e-6
    elif mode == "alpha":
        power["per_claim_alpha"] = 0.01
    elif mode == "grid":
        power["power_grid"][0]["union_bound_family_power_lower"] += 1e-6
    elif mode == "feasibility":
        power.update(
            {
                "feasible": False,
                "passed": False,
                "decision": "power_infeasible_no_confirmation",
                "selected_grid_record": None,
                "selected_confirmation_episodes_per_regime": None,
                "confirmation_episode_count_per_regime": None,
                "selected_confirmation_slots_per_regime": None,
                "fixed_prefix_only": False,
                "terminal_label_if_infeasible": "domain_robust_gate_failed",
                "confirmation_generation_authorized_by_power": False,
            }
        )
    elif mode == "input_hash":
        power["inputs"]["fit_summary"]["sha256"] = "0" * 64
    elif mode == "extra_key":
        power["posthoc_override"] = True
    elif mode == "nested_extra_key":
        power["inputs"]["fit_summary"]["posthoc_override"] = True
    else:  # pragma: no cover - test fixture misuse
        raise AssertionError(mode)
    write(power_path, power)


def test_preseal_independently_recomputes_exact_binding_power(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = _preseal_power_tree(tmp_path, monkeypatch)
    result = preseal.validate_power_input_coherence(tree["paths"])
    assert result["passed"] is True
    assert result["checks"]["binding_power_object_exact_recomputation"] is True


@pytest.mark.parametrize(
    "tamper",
    (
        "n",
        "stored_fit_mean",
        "stored_selection_sd",
        "ucl",
        "alpha",
        "grid",
        "feasibility",
        "input_hash",
        "extra_key",
        "nested_extra_key",
    ),
)
def test_preseal_rejects_tampered_power_before_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    tree = _preseal_power_tree(tmp_path, monkeypatch)
    _tamper_preseal_power(tree, tamper)
    with pytest.raises(preseal.PresealError, match="power|summary"):
        preseal.validate_power_input_coherence(tree["paths"])
    assert not tree["pre_confirmation_seal"].exists()


def test_direct_selection_defers_power_summaries_until_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "cohort_seed_ledger.json"
    ledger.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "COHORT_LEDGER_PATH", ledger)
    monkeypatch.setattr(launcher, "verify_here", lambda *args, **kwargs: {})
    monkeypatch.setattr(launcher, "_require_controller_state", lambda *_: {})
    monkeypatch.setattr(
        launcher,
        "_role_aggregate_paths",
        lambda _role: {regime: tmp_path / f"{regime}.npz" for regime in launcher.REGIMES},
    )
    monkeypatch.setattr(launcher, "_repo_hashes", lambda _paths: {"input": "0" * 64})
    monkeypatch.setattr(fit_select, "comparator_seeds_from_ledger", lambda _: {})
    monkeypatch.setattr(fit_select, "load_per_dgp_npz", lambda _: {})
    observed: dict[str, object] = {}

    def select(**kwargs):
        observed.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(fit_select, "select_after_fit_lock", select)
    assert launcher._worker_select_gate()["passed"] is True
    assert observed["fit_power_summary_path"] is None
    assert observed["selection_power_summary_path"] is None


def test_direct_freeze_summaries_are_identity_bound_before_power(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launcher, "ATTEMPT_ROOT", tmp_path)
    fit_lock = tmp_path / "fit/fit_lock.json"
    selection = tmp_path / "selection/selection_ledger.json"
    freeze = tmp_path / "freeze/gate_freeze.json"
    for path, value in (
        (fit_lock, {"status": "locked"}),
        (selection, {"selected_candidate_id": "candidate-01"}),
        (
            freeze,
            {
                "selected_candidate_id": "candidate-01",
                "selected_candidate_object_sha256": "a" * 64,
            },
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def write(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    fake = SimpleNamespace(
        load_fitted_candidates=lambda _path: {"fitted": True},
        selected_power_summaries=lambda _fitted, _ledger: (
            {
                "schema_version": 1,
                "role": "fit",
                "co_primary_only": True,
                "auxiliary_robust_whitening_excluded": True,
            },
            {
                "schema_version": 1,
                "role": "selection",
                "co_primary_only": True,
                "auxiliary_robust_whitening_excluded": True,
            },
        ),
        _atomic_json_exclusive=write,
    )
    result = launcher._seal_enriched_power_summaries(fake)
    assert set(result) == {
        "fit_power_summary_sha256",
        "selection_power_summary_sha256",
    }
    fit_summary = json.loads(
        (tmp_path / "metrics/fit_power_summary.json").read_text(encoding="utf-8")
    )
    selection_summary = json.loads(
        (tmp_path / "metrics/selection_power_summary.json").read_text(
            encoding="utf-8"
        )
    )
    identity = power_analysis.validate_summary_identity(
        fit_summary, selection_summary
    )
    assert identity["gate_freeze_sha256"] == runtime_contract.sha256_file(freeze)


def test_exact_flops_and_synthetic_scientific_qualifications() -> None:
    assert preseal.qualify_exact_flops()["passed"]
    assert preseal.qualify_synthetic_fit_selection()["passed"]
    sparse = preseal.qualify_synthetic_sparse_dense()
    assert sparse["passed"]
    assert sparse["checks"]["exact_boundary_tie_stops"]
    comparator = preseal.qualify_synthetic_comparators_bootstrap()
    assert comparator["passed"]
    assert comparator["checks"][
        "bootstrap_separate_implementation_elementwise_exact"
    ]


def test_frozen_v004_feature_lineage_is_bitwise_exact() -> None:
    result = preseal.qualify_feature_lineage()
    assert result["passed"]
    assert result["checks"]["synthetic_feature_values_bitwise_v004_exact"]


def test_exclusive_seal_writer_has_no_overwrite_route(tmp_path: Path) -> None:
    destination = tmp_path / "seal.json"
    preseal._write_json_exclusive(destination, {"passed": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"passed": True}
    with pytest.raises(FileExistsError):
        preseal._write_json_exclusive(destination, {"passed": False})
