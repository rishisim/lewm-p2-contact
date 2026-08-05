from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
if str(ATTEMPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTEMPT_ROOT))

import build_manifest


STUDY_RELATIVE = Path("runs/lewm_domain_robust_gate")


def _write(path: Path, value: str = "{}\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _configure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    required_attempt: frozenset[str] = frozenset({"analysis.py"}),
    audit_inputs: frozenset[str] = frozenset(),
    required_repository: frozenset[str] = frozenset(),
) -> tuple[Path, Path]:
    study = tmp_path / STUDY_RELATIVE
    attempt = study / "attempts/v008"
    attempt.mkdir(parents=True)
    monkeypatch.setattr(build_manifest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build_manifest, "ATTEMPT_ROOT", attempt)
    monkeypatch.setattr(
        build_manifest, "PRE_DATA_STATIC_RELATIVE_PATHS", required_attempt
    )
    monkeypatch.setattr(build_manifest, "PRE_DATA_AUDIT_INPUTS", audit_inputs)
    monkeypatch.setattr(
        build_manifest,
        "PRE_DATA_REPOSITORY_RELATIVE_PATHS",
        required_repository,
    )
    monkeypatch.setattr(
        build_manifest,
        "_prospective_episode_ids",
        lambda: {
            (role, regime): frozenset({"episode-1"})
            for role in build_manifest.GENERATED_ROLES
            for regime in build_manifest.GENERATED_REGIMES
        },
    )
    for relative in sorted(required_attempt | audit_inputs):
        _write(attempt / relative, "VALUE = 1\n" if relative.endswith(".py") else "{}\n")
    for relative in sorted(required_repository):
        _write(
            tmp_path / relative,
            "VALUE = 1\n" if relative.endswith(".py") else "{}\n",
        )
    return study, attempt


def _synthetic_predata_manifest() -> dict[str, object]:
    manifest = build_manifest._manifest_for_paths(
        build_manifest.collect_pre_data_source_paths(),
        label=build_manifest.PRE_DATA_LABEL,
        include_inherited_runtime=False,
    )
    manifest["source_closure_policy"] = build_manifest.source_closure_policy()
    return manifest


def test_v008_identity_and_all_anticipated_sources_are_required() -> None:
    assert build_manifest.ACTIVE_ATTEMPT == "v008"
    assert build_manifest.SCIENCE_ATTEMPT == "v001"
    assert {
        "inherited_authorization.py",
        "scientific_replay.py",
        "scientific_replay_launcher.py",
        "version_forward_prepare.py",
        "version_forward_transaction.py",
        "verify_version_forward.py",
        "tests/test_manifest_closure_v002.py",
        "tests/test_attempt_parameterization_v002.py",
        "tests/test_inherited_authorization_v002.py",
        "tests/test_scientific_replay.py",
        "tests/test_version_forward_v002.py",
        "tests/test_version_forward_transaction.py",
    } <= build_manifest.PRE_DATA_STATIC_RELATIVE_PATHS


def test_scientific_replay_outputs_are_exact_prospective_generated_artifacts() -> None:
    roles = ("fit", "selection", "smoke", "confirmation")
    regimes = (
        "native_plan",
        "markov_oracle",
        "plan_action_noise_0p2",
        "plan_random_action_0p1",
    )
    expected = {
        "audit/scientific_replay_qualification.json",
        *(f"audit/{role}_scientific_replay.json" for role in roles),
        *(
            f"audit/scientific_replay/{role}_{regime}.json"
            for role in roles
            for regime in regimes
        ),
    }
    assert len(expected) == 21
    assert expected <= build_manifest.GENERATED_AUDIT_NAMES
    assert expected.isdisjoint(build_manifest.PRE_DATA_STATIC_RELATIVE_PATHS)
    assert all(build_manifest.is_generated_attempt_product(path) for path in expected)


def test_immutable_study_root_inputs_and_mutable_exceptions_are_exact() -> None:
    assert build_manifest.PRE_DATA_REPOSITORY_RELATIVE_PATHS == frozenset(
        {
            "runs/lewm_domain_robust_gate/LEDGER_CHAIN_GENESIS.json",
            "runs/lewm_domain_robust_gate/README.md",
            "runs/lewm_domain_robust_gate/program.py",
        }
    )
    assert build_manifest.MUTABLE_STUDY_ROOT_RELATIVE_PATHS == frozenset(
        {
            "runs/lewm_domain_robust_gate/STATE.json",
            "runs/lewm_domain_robust_gate/STATE_TRANSACTION.json",
            "runs/lewm_domain_robust_gate/VERSION_FORWARD_TRANSACTION.json",
        }
    )


@pytest.mark.parametrize(
    "relative",
    [
        "omitted.py",
        "shadow_config.json",
        "notes.md",
        "nested/shadow.JSON",
        "tests/undeclared.md",
    ],
)
def test_build_rejects_every_unlisted_allowed_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    _, attempt = _configure(tmp_path, monkeypatch)
    _write(attempt / relative, "VALUE = 2\n")
    with pytest.raises(build_manifest.ManifestError, match="unexpected"):
        build_manifest.build_pre_data_manifest()


@pytest.mark.parametrize(
    "relative",
    ["required.json", "docs/required.md", "required.py"],
)
def test_build_rejects_missing_required_allowed_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    _, attempt = _configure(
        tmp_path,
        monkeypatch,
        required_attempt=frozenset({"analysis.py", relative}),
    )
    (attempt / relative).unlink()
    with pytest.raises(build_manifest.ManifestError, match="missing"):
        build_manifest.build_pre_data_manifest()


def test_verification_recomputes_closure_and_rejects_late_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, attempt = _configure(tmp_path, monkeypatch)
    manifest = _synthetic_predata_manifest()
    assert build_manifest.verify_manifest(manifest)["passed"] is True
    _write(attempt / "late_config.json")
    with pytest.raises(build_manifest.ManifestError, match="verification failed"):
        build_manifest.verify_manifest(manifest)


def test_exact_generated_products_do_not_destabilize_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, attempt = _configure(tmp_path, monkeypatch)
    manifest = _synthetic_predata_manifest()
    products = (
        "audit/pre_data_source_manifest.json",
        "audit/pre_data_inheritance_seal.json",
        "audit/fit_cohorts.json",
        "audit/fit_native_plan_execution_failure.json",
        "data/replacement_registry.json",
        "data/replacement_claims/000001.json",
        "data/fit/native_plan/raw_manifest.json",
        "data/fit/native_plan/execution_manifest.json",
        "data/fit/native_plan/raw/episode-1.json",
        "data/fit/native_plan/execution/episode-1.json",
        "data/persistence_intents/fit/native_plan/episode-1.json",
        "fit/fit_lock.json",
        "selection/selection_ledger.json",
        "freeze/compiled_gate_manifest.json",
        "freeze/gate_freeze.json",
        "metrics/fit_power_summary.json",
        "metrics/selection_power_summary.json",
        "metrics/compute_ledger.json",
        "metrics/bootstrap_summary.json",
        "metrics/latency_and_resources.json",
        "power_analysis.json",
        "analysis_result.json",
        "decision.json",
        "REPORT.md",
        "ROBUSTNESS_MAP.json",
        "INDEPENDENT_AUDIT.md",
        "LIMITATIONS.md",
        "FOLLOW_ON_TASK.md",
    )
    for relative in products:
        _write(attempt / relative)
    _write(study / "STATE.json")
    _write(study / "STATE_TRANSACTION.json")
    _write(study / "VERSION_FORWARD_TRANSACTION.json")
    result = build_manifest.verify_manifest(manifest)
    assert result["passed"] is True
    classification = build_manifest.classify_pre_data_source_paths()
    assert set(products) <= set(classification["generated_attempt"])
    assert set(classification["generated_repository"]) == {
        "runs/lewm_domain_robust_gate/STATE.json",
        "runs/lewm_domain_robust_gate/STATE_TRANSACTION.json",
        "runs/lewm_domain_robust_gate/VERSION_FORWARD_TRANSACTION.json",
    }


@pytest.mark.parametrize(
    "relative",
    [
        "audit/omitted.json",
        "audit/fit_native_plan_execution_failure_extra.json",
        "fit/shadow.json",
        "selection/shadow.json",
        "freeze/shadow.json",
        "metrics/shadow.json",
        "data/fit/not_a_regime/raw/episode.json",
        "data/not_a_role/native_plan/raw/episode.json",
        "data/fit/native_plan/shadow/episode.json",
        "data/fit/native_plan/raw/evil.py",
        "data/fit/native_plan/raw/not-prospective.json",
        "data/replacement_claims/1.json",
        "data/replacement_claims/０００００１.json",
        "audit/evil.md",
    ],
)
def test_near_miss_generated_paths_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    _, attempt = _configure(tmp_path, monkeypatch)
    manifest = _synthetic_predata_manifest()
    _write(attempt / relative, "VALUE = 2\n")
    with pytest.raises(build_manifest.ManifestError, match="verification failed"):
        build_manifest.verify_manifest(manifest)


def test_root_required_inputs_are_recorded_but_mutable_state_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = build_manifest.PRE_DATA_REPOSITORY_RELATIVE_PATHS
    study, _ = _configure(
        tmp_path,
        monkeypatch,
        required_repository=required,
    )
    _write(study / "STATE.json")
    _write(study / "STATE_TRANSACTION.json")
    _write(study / "VERSION_FORWARD_TRANSACTION.json")
    manifest = _synthetic_predata_manifest()
    assert required <= set(manifest["local_files"])
    assert not (
        build_manifest.MUTABLE_STUDY_ROOT_RELATIVE_PATHS
        & set(manifest["local_files"])
    )
    assert build_manifest.verify_manifest(manifest)["passed"] is True
    _write(study / "shadow.md", "unsealed\n")
    with pytest.raises(build_manifest.ManifestError, match="verification failed"):
        build_manifest.verify_manifest(manifest)


def test_required_file_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, attempt = _configure(tmp_path, monkeypatch)
    source = attempt / "analysis.py"
    source.unlink()
    target = _write(attempt / "target.txt", "VALUE = 1\n")
    source.symlink_to(target)
    with pytest.raises(build_manifest.ManifestError, match="symlink forbidden"):
        build_manifest.build_pre_data_manifest()


def test_generated_file_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, attempt = _configure(tmp_path, monkeypatch)
    manifest = _synthetic_predata_manifest()
    target = _write(attempt / "target.txt")
    linked = attempt / "audit/fit_cohorts.json"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(target)
    with pytest.raises(build_manifest.ManifestError, match="symlink forbidden"):
        build_manifest.verify_manifest(manifest)


def test_directory_symlink_is_rejected_without_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, attempt = _configure(tmp_path, monkeypatch)
    manifest = _synthetic_predata_manifest()
    target = tmp_path / "outside"
    target.mkdir()
    _write(target / "hidden.py", "VALUE = 2\n")
    (attempt / "linked_sources").symlink_to(target, target_is_directory=True)
    with pytest.raises(build_manifest.ManifestError, match="directory symlinks"):
        build_manifest.verify_manifest(manifest)


def test_required_hardlink_inode_alias_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, attempt = _configure(
        tmp_path,
        monkeypatch,
        required_attempt=frozenset({"analysis.py", "runner.py"}),
    )
    (attempt / "runner.py").unlink()
    os.link(attempt / "analysis.py", attempt / "runner.py")
    with pytest.raises(build_manifest.ManifestError, match="inode aliases"):
        build_manifest.build_pre_data_manifest()


def test_generated_hardlink_inode_alias_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, attempt = _configure(tmp_path, monkeypatch)
    manifest = _synthetic_predata_manifest()
    left = _write(attempt / "audit/fit_cohorts.json")
    right = attempt / "audit/selection_cohorts.json"
    os.link(left, right)
    with pytest.raises(build_manifest.ManifestError, match="inode aliases"):
        build_manifest.verify_manifest(manifest)


def test_manifest_cannot_drop_required_or_inject_generated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, attempt = _configure(tmp_path, monkeypatch)
    manifest = _synthetic_predata_manifest()
    required_key = next(iter(manifest["local_files"]))
    dropped = copy.deepcopy(manifest)
    dropped["local_files"].pop(required_key)
    dropped["local_file_count"] -= 1
    dropped["local_total_bytes"] = sum(
        item["bytes"] for item in dropped["local_files"].values()
    )
    with pytest.raises(build_manifest.ManifestError, match="verification failed"):
        build_manifest.verify_manifest(dropped)

    generated = _write(attempt / "audit/fit_cohorts.json")
    injected = copy.deepcopy(manifest)
    injected["local_files"][
        generated.relative_to(tmp_path).as_posix()
    ] = build_manifest._record(generated, local_source=True)
    injected["local_file_count"] += 1
    injected["local_total_bytes"] += generated.stat().st_size
    with pytest.raises(build_manifest.ManifestError, match="verification failed"):
        build_manifest.verify_manifest(injected)


def test_manifest_binds_exact_closure_policy_and_science_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(tmp_path, monkeypatch)
    manifest = _synthetic_predata_manifest()
    assert manifest["attempt"] == "v008"
    assert manifest["science_attempt"] == "v001"
    assert (
        "data/replacement_claims/{claim_index:06d}.json"
        in manifest["source_closure_policy"]["generated_data_json_shapes"]
    )
    assert manifest["source_closure_policy"][
        "generated_episode_jsons_prospective_ledger_bound"
    ] is True
    assert build_manifest.verify_manifest(manifest)["passed"] is True
    changed = copy.deepcopy(manifest)
    changed["source_closure_policy"]["allowed_suffixes"] = [".py"]
    with pytest.raises(build_manifest.ManifestError, match="verification failed"):
        build_manifest.verify_manifest(changed)
