from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ATTEMPT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


prepare = _load("version_forward_prepare_subject", ATTEMPT / "version_forward_prepare.py")
verify = _load("verify_version_forward_subject", ATTEMPT / "verify_version_forward.py")
independent = _load(
    "independent_verify_version_forward_subject", ATTEMPT / "independent_verify.py"
)
inherited = _load(
    "inherited_authorization_version_forward_subject",
    ATTEMPT / "inherited_authorization.py",
)


def _clone_fixture_file(source: str, destination: str) -> str:
    """Create an inode-distinct copy-on-write fixture with preserved metadata."""

    subprocess.run(
        ["/bin/cp", "-c", "-p", source, destination],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return destination


def _copy_live_study(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    study = repo / "runs/lewm_domain_robust_gate"
    shutil.copytree(ATTEMPT.parents[1], study, copy_function=_clone_fixture_file)
    live_repo = ATTEMPT.parents[3]
    science_seal = json.loads(
        (study / "attempts/v001/audit/pre_data_seal.json").read_text(
            encoding="utf-8"
        )
    )
    for raw in science_seal["sealed_files"]:
        if raw.startswith("runs/lewm_domain_robust_gate/"):
            continue
        destination = repo / raw
        destination.parent.mkdir(parents=True, exist_ok=True)
        _clone_fixture_file(str(live_repo / raw), str(destination))
    return repo, study, study / "attempts/v006", study / "attempts/v008"


def _closure_calls(
    repo: Path, study: Path, source: Path, target: Path
) -> tuple[object, object, object]:
    source_seal = json.loads(
        (source / "audit/pre_data_inheritance_seal.json").read_text(encoding="utf-8")
    )
    state = json.loads((study / "STATE.json").read_text(encoding="utf-8"))
    policy = independent.PathPolicy(
        repo,
        "runs/lewm_domain_robust_gate/attempts/v008",
        "runs/lewm_domain_robust_gate",
    )
    return (
        lambda: prepare._manifest_closure(source, target, repo, source_seal),
        lambda: verify._closure(
            source, target, repo, source_seal, state, phase="pre-forward"
        ),
        lambda: independent._independent_manifest_closure(policy),
    )


def _assert_all_closures_reject(
    repo: Path, study: Path, source: Path, target: Path
) -> None:
    for call in _closure_calls(repo, study, source, target):
        with pytest.raises(
            (
                prepare.PreparationError,
                verify.VerificationError,
                independent.VerificationError,
            )
        ):
            call()


def _lineage_pair(
    tmp_path: Path, relative: str
) -> tuple[Path, Path]:
    source = tmp_path / "source" / relative
    target = tmp_path / "target" / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ATTEMPT.parent / "v006" / relative, source)
    shutil.copyfile(ATTEMPT / relative, target)
    return source, target


def _assert_all_lineage_scopes_reject(
    source: Path, target: Path, relative: str
) -> None:
    calls = (
        lambda: prepare._lineage_support_scope(source, target, relative),
        lambda: verify._lineage_scope(source, target, relative),
        lambda: independent._independent_lineage_support_scope(
            source, target, relative
        ),
    )
    for call in calls:
        with pytest.raises(
            (
                prepare.PreparationError,
                verify.VerificationError,
                independent.VerificationError,
            )
        ):
            call()


def _literal_policy_copy(module: object) -> dict[str, dict[str, object]]:
    return {
        relative: dict(value)
        for relative, value in module.LINEAGE_SUPPORT_UNIT_POLICY.items()
    }


def test_import_is_side_effect_free_and_real_artifact_is_absent() -> None:
    assert not (ATTEMPT / "audit/pre_data_inheritance_seal.json").exists()


def test_exclusive_writer_never_replaces_temp_artifact(tmp_path: Path) -> None:
    output = tmp_path / "attempts/v008/audit/pre_data_inheritance_seal.json"
    value = {"schema_version": 1, "passed": True}
    prepare.write_exclusive(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        prepare.write_exclusive(output, {"passed": False})
    assert output.read_bytes() == original


def test_existing_seal_recomputation_requires_exact_canonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "repo/runs/lewm_domain_robust_gate"
    seal_path = study / "attempts/v008/audit/pre_data_inheritance_seal.json"
    seal_path.parent.mkdir(parents=True)
    (study / "STATE.json").write_text("{}\n", encoding="utf-8")
    (study / "RESEARCH_LEDGER.jsonl").write_text("{}\n", encoding="utf-8")
    expected = {
        "schema_version": 1,
        "created_unix_ns": 17,
        "root_controls": {"program.py": {"bytes": 9}},
        "passed": True,
    }
    monkeypatch.setattr(
        prepare,
        "build_inheritance_seal",
        lambda _study, _existing_created_unix_ns: json.loads(json.dumps(expected)),
    )
    seal_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = prepare.verify_existing_inheritance_seal(seal_path, study)
    assert result["passed"] is True

    aliased = json.loads(json.dumps(expected))
    aliased["root_controls"]["program.py"]["bytes"] = 9.0
    seal_path.write_text(
        json.dumps(aliased, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(prepare.PreparationError, match="producer recomputation"):
        prepare.verify_existing_inheritance_seal(seal_path, study)

    schema_aliased = json.loads(json.dumps(expected))
    schema_aliased["schema_version"] = True
    seal_path.write_text(
        json.dumps(schema_aliased, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(prepare.PreparationError, match="schema type"):
        prepare.verify_existing_inheritance_seal(seal_path, study)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("schema_version",), True),
        (("root_controls", "program.py", "bytes"), 9.0),
        (("confirmation_artifact_scan", "file_count"), False),
        (("passed",), 1),
    ),
)
def test_all_seal_consumers_reject_numeric_scalar_aliases(
    path: tuple[str, ...], value: object
) -> None:
    clean = {
        "schema_version": 1,
        "root_controls": {
            "program.py": {"bytes": 9, "source_seal_bound": True}
        },
        "confirmation_artifact_scan": {"file_count": 0, "paths": []},
        "passed": True,
    }
    validators = (
        (
            verify._validate_inheritance_seal_scalar_types,
            verify.VerificationError,
        ),
        (
            inherited._validate_inheritance_seal_scalar_types,
            inherited.InheritedAuthorizationError,
        ),
        (
            independent._validate_inheritance_seal_scalar_types,
            independent.VerificationError,
        ),
    )
    for validator, _error in validators:
        validator(clean)
    mutated = json.loads(json.dumps(clean))
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    for validator, error in validators:
        with pytest.raises(error):
            validator(mutated)


def test_independently_transcribed_ast_normalizers_agree_in_temp_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v001.py"
    target = tmp_path / "v008.py"
    source.write_text(
        'ATTEMPT_ROOT = object()\nATTEMPT = "v001"\nRESULT = {"attempt": ATTEMPT}\n',
        encoding="utf-8",
    )
    target.write_text(
        'ATTEMPT_ROOT = object()\nACTIVE_ATTEMPT = ATTEMPT_ROOT.name\n'
        'if ACTIVE_ATTEMPT != "v008":\n    raise RuntimeError("wrong v008")\n'
        'RESULT = {"attempt": ACTIVE_ATTEMPT}\n',
        encoding="utf-8",
    )
    prepare_source = prepare._normalized_ast(source, target=False)
    prepare_target = prepare._normalized_ast(target, target=True)
    verify_source = verify._normalized(source, False)
    verify_target = verify._normalized(target, True)
    assert prepare_source == prepare_target
    assert verify_source == verify_target
    assert prepare_source == verify_source


def test_declared_live_normalized_paths_agree_across_all_three_implementations() -> None:
    source_root = ATTEMPT.parent / "v006"
    assert prepare.NORMALIZED_AST_PATHS == verify.NORMALIZED
    assert prepare.NORMALIZED_AST_PATHS == independent.NORMALIZED_AST_PATHS
    assert len(prepare.NORMALIZED_AST_PATHS) == 12
    for relative in sorted(prepare.NORMALIZED_AST_PATHS):
        source = source_root / relative
        target = ATTEMPT / relative
        prepare_source = prepare._normalized_ast(source, target=False)
        prepare_target = prepare._normalized_ast(target, target=True)
        verify_source = verify._normalized(source, False)
        verify_target = verify._normalized(target, True)
        independent_source = independent.normalized_administrative_ast_sha256(
            source
        )
        independent_target = independent.normalized_administrative_ast_sha256(
            target
        )
        assert prepare_source == prepare_target
        assert verify_source == verify_target
        assert prepare_source == verify_source
        assert prepare_source[1] == independent_source == independent_target


def test_independently_transcribed_contract_canonicalizers_agree() -> None:
    source = json.loads(
        (ATTEMPT.parent / "v006/verifier_contract.json").read_text(encoding="utf-8")
    )
    target = json.loads(
        (ATTEMPT / "verifier_contract.json").read_text(encoding="utf-8")
    )
    assert prepare._normalize_contract(source) == prepare._normalize_contract(target)
    assert verify._contract(source) == verify._contract(target)
    assert prepare._normalize_contract(target) == verify._contract(target)


def test_verifier_rejects_nonexact_seal_path_in_temp_root(tmp_path: Path) -> None:
    study = tmp_path / "repo/runs/lewm_domain_robust_gate"
    (study / "attempts/v001").mkdir(parents=True)
    (study / "attempts/v008/audit").mkdir(parents=True)
    wrong = study / "attempts/v008/audit/equivalence.json"
    wrong.write_text("{}\n", encoding="utf-8")
    with pytest.raises(verify.VerificationError, match="path is not exact"):
        verify.verify(wrong, study, phase="pre-forward")


def test_procedural_partition_is_explicitly_empty() -> None:
    assert prepare.PROCEDURAL_EXISTING_PATHS == frozenset()
    assert verify.PROCEDURAL == frozenset()
    assert independent.PROCEDURAL_EXISTING_PATHS == frozenset()


def test_no_procedural_policy_contains_wildcard() -> None:
    assert prepare.PROCEDURAL_ALLOWED_SYMBOLS == prepare.PROCEDURAL_EXPECTED_CHANGED_SYMBOLS
    assert verify.PROCEDURAL_ALLOWED_SYMBOLS == verify.PROCEDURAL_EXPECTED_CHANGED_SYMBOLS
    assert independent.PROCEDURAL_ALLOWED_SYMBOLS == independent.PROCEDURAL_EXPECTED_CHANGED_SYMBOLS
    assert prepare.PROCEDURAL_ALLOWED_SYMBOLS == verify.PROCEDURAL_ALLOWED_SYMBOLS
    assert prepare.PROCEDURAL_ALLOWED_SYMBOLS == independent.PROCEDURAL_ALLOWED_SYMBOLS
    assert prepare.PROCEDURAL_EXISTING_PATHS == verify.PROCEDURAL
    assert prepare.PROCEDURAL_EXISTING_PATHS == independent.PROCEDURAL_EXISTING_PATHS
    assert all(
        "*" not in allowed
        for allowed in prepare.PROCEDURAL_ALLOWED_SYMBOLS.values()
    )
    assert all(
        "*" not in allowed
        for allowed in verify.PROCEDURAL_ALLOWED_SYMBOLS.values()
    )
    assert all(
        "*" not in allowed
        for allowed in independent.PROCEDURAL_ALLOWED_SYMBOLS.values()
    )


def test_lineage_unit_policies_are_exact_literal_transcriptions() -> None:
    policy = prepare.LINEAGE_SUPPORT_UNIT_POLICY
    assert policy == verify.LINEAGE_SUPPORT_UNIT_POLICY
    assert policy == independent.LINEAGE_SUPPORT_UNIT_POLICY
    assert set(policy) == prepare.LINEAGE_SUPPORT_PATHS
    assert set(policy) == verify.LINEAGE_SUPPORT
    assert set(policy) == independent.LINEAGE_SUPPORT_PATHS
    for value in policy.values():
        assert value["required_changed_top_level_units"] == value[
            "allowed_changed_top_level_units"
        ]
        assert all(
            "*" not in label
            for label in value["allowed_changed_top_level_units"]
        )


def test_lineage_unit_census_covers_every_ordered_module_body_node() -> None:
    for relative in sorted(prepare.LINEAGE_SUPPORT_UNIT_POLICY):
        for root in (ATTEMPT.parent / "v005", ATTEMPT):
            path = root / relative
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            first = prepare._top_level_symbols(path)
            second = verify._symbols(path)
            third = independent._top_level_symbol_hashes(path)
            assert first == second == third
            assert len(first) == len(tree.body)
            assert list(first.values()) == [
                hashlib.sha256(
                    ast.dump(
                        node,
                        annotate_fields=True,
                        include_attributes=False,
                    ).encode()
                ).hexdigest()
                for node in tree.body
            ]
            assert all(
                len(label) >= 4
                and label[-4] == "#"
                and label[-3:].isdigit()
                for label in first
            )


def test_lineage_scope_rejects_unauthorized_changed_shared_unit(
    tmp_path: Path,
) -> None:
    relative = "tests/test_workflow.py"
    source, target = _lineage_pair(tmp_path, relative)
    text = target.read_text(encoding="utf-8")
    marker = "import json\n"
    assert marker in text
    target.write_text(
        text.replace(marker, "import json as json_module\n", 1),
        encoding="utf-8",
    )
    _assert_all_lineage_scopes_reject(source, target, relative)


@pytest.mark.parametrize(
    ("snippet", "label_prefix"),
    (
        ("\nif True:\n    pass\n", "if#"),
        ("\ntry:\n    pass\nexcept RuntimeError:\n    pass\n", "try#"),
        ("\n0\n", "expr#"),
    ),
)
def test_lineage_scope_rejects_inserted_unnamed_module_units(
    tmp_path: Path, snippet: str, label_prefix: str
) -> None:
    relative = "tests/test_workflow.py"
    source, target = _lineage_pair(tmp_path, relative)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(snippet)
    assert any(
        label.startswith(label_prefix)
        for label in prepare._top_level_symbols(target)
    )
    _assert_all_lineage_scopes_reject(source, target, relative)


def test_lineage_scope_rejects_duplicate_binding_occurrence(
    tmp_path: Path,
) -> None:
    relative = "independent_verify.py"
    source, target = _lineage_pair(tmp_path, relative)
    with target.open("a", encoding="utf-8") as handle:
        handle.write('\nACTIVE_ATTEMPT = "tamper"\n')
    assert "assign:ACTIVE_ATTEMPT#002" in prepare._top_level_symbols(target)
    _assert_all_lineage_scopes_reject(source, target, relative)


def test_lineage_scope_rejects_deleted_required_unit(tmp_path: Path) -> None:
    relative = "version_forward_transaction.py"
    source, target = _lineage_pair(tmp_path, relative)
    text = target.read_text(encoding="utf-8")
    marker = 'SCIENCE_ATTEMPT = "v001"\n'
    assert text.count(marker) == 1
    target.write_text(text.replace(marker, "", 1), encoding="utf-8")
    _assert_all_lineage_scopes_reject(source, target, relative)


def test_lineage_scope_rejects_label_sequence_reorder(tmp_path: Path) -> None:
    relative = "tests/test_workflow.py"
    source, target = _lineage_pair(tmp_path, relative)
    text = target.read_text(encoding="utf-8")
    marker = "import importlib.util\nimport json\n"
    assert marker in text
    target.write_text(
        text.replace(marker, "import json\nimport importlib.util\n", 1),
        encoding="utf-8",
    )
    _assert_all_lineage_scopes_reject(source, target, relative)


@pytest.mark.parametrize(
    "mutation",
    (
        "omitted_path",
        "extra_path",
        "omitted_required",
        "extra_allowed",
        "wildcard_allowed",
        "inserted_list",
        "deleted_list",
    ),
)
def test_lineage_scope_rejects_static_policy_tamper(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    relative = "tests/test_workflow.py"
    source = ATTEMPT.parent / "v005" / relative
    target = ATTEMPT / relative
    implementations = (
        (prepare, prepare._lineage_support_scope),
        (verify, verify._lineage_scope),
        (independent, independent._independent_lineage_support_scope),
    )
    for module, scope in implementations:
        policy = _literal_policy_copy(module)
        if mutation == "omitted_path":
            del policy[relative]
        elif mutation == "extra_path":
            policy["extra.py"] = dict(policy[relative])
        elif mutation == "omitted_required":
            policy[relative]["required_changed_top_level_units"] = list(
                policy[relative]["required_changed_top_level_units"]
            )[1:]
        elif mutation == "extra_allowed":
            policy[relative]["allowed_changed_top_level_units"] = sorted(
                [
                    *policy[relative]["allowed_changed_top_level_units"],
                        "function:unrecorded_unit#999",
                ]
            )
        elif mutation == "wildcard_allowed":
            policy[relative]["allowed_changed_top_level_units"] = sorted(
                [*policy[relative]["allowed_changed_top_level_units"], "*#001"]
            )
        elif mutation == "inserted_list":
            policy[relative]["inserted_top_level_units"] = ["if#999"]
        else:
            policy[relative]["deleted_top_level_units"] = ["try#999"]
        monkeypatch.setattr(module, "LINEAGE_SUPPORT_UNIT_POLICY", policy)
        with pytest.raises(
            (
                prepare.PreparationError,
                verify.VerificationError,
                independent.VerificationError,
            )
        ):
            scope(source, target, relative)


def test_exact_generator_cannot_change_without_lineage_authorization(
    tmp_path: Path,
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    generator = target / "generator.py"
    generator.write_text(
        generator.read_text(encoding="utf-8")
        + "\ndef unrecorded_generator_delta():\n    return 1\n",
        encoding="utf-8",
    )
    source_seal = json.loads(
        (source / "audit/pre_data_inheritance_seal.json").read_text(
            encoding="utf-8"
        )
    )
    closure = prepare._manifest_closure(source, target, repo, source_seal)
    with pytest.raises(prepare.PreparationError, match="exact-hash object differs"):
        prepare._partition_sources(source, target, repo, closure)


def test_post_forward_selection_raw_progression_is_exact_and_tamper_evident() -> None:
    selection = {
        "active_attempt": "v008",
        "current_state": "SELECTION_COHORTS",
        "expected_fit_episode_count": 1200,
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    verify._validate_post_forward_raw_authorization_state(selection)
    wrong_lineage = dict(selection, active_attempt="v002")
    with pytest.raises(verify.VerificationError, match="active attempt drift"):
        verify._validate_post_forward_raw_authorization_state(wrong_lineage)
    wrong_count = dict(selection, fit_outcome_episodes=1199)
    with pytest.raises(verify.VerificationError, match="counters drift"):
        verify._validate_post_forward_raw_authorization_state(wrong_count)
    wrong_state = dict(selection, current_state="GATE_FREEZE")
    with pytest.raises(verify.VerificationError, match="not selection raw"):
        verify._validate_post_forward_raw_authorization_state(wrong_state)


def test_post_forward_source_closure_recognizes_only_predeclared_products() -> None:
    constants: dict[str, set[str]] = {}
    episode_ids = {
        ("fit", "native_plan"): {"fit-episode"},
        ("selection", "markov_oracle"): {"selection-episode"},
    }
    state = {"current_state": "SELECTION_COHORTS"}
    assert verify._generated_source_product(
        "data/fit/native_plan/raw_manifest.json",
        constants,
        episode_ids,
        state,
        "post-forward",
    )
    assert verify._generated_source_product(
        "data/selection/markov_oracle/execution/selection-episode.json",
        constants,
        episode_ids,
        state,
        "post-forward",
    )
    assert verify._generated_source_product(
        "data/replacement_claims/000001.json",
        constants,
        episode_ids,
        state,
        "post-forward",
    )
    assert verify._generated_source_product(
        "audit/fit_cohorts.json", constants, episode_ids, state, "post-forward"
    )
    assert not verify._generated_source_product(
        "data/selection/markov_oracle/execution/arbitrary.json",
        constants,
        episode_ids,
        state,
        "post-forward",
    )
    assert not verify._generated_source_product(
        "data/replacement_claims/1.json",
        constants,
        episode_ids,
        state,
        "post-forward",
    )
    assert not verify._generated_source_product(
        "data/fit/native_plan/raw_manifest.json",
        constants,
        episode_ids,
        state,
        "pre-forward",
    )
    assert not verify._generated_source_product(
        "new_scientific_source.py",
        constants,
        episode_ids,
        state,
        "post-forward",
    )


def test_all_three_live_closure_walks_accept_exact_clean_preforward_tree(
    tmp_path: Path,
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    results = [call() for call in _closure_calls(repo, study, source, target)]
    assert all(result["passed"] is True for result in results)
    assert results[0] == results[1] == results[2]


def test_all_three_fit_inventory_reconstructions_reject_coherent_hash_tamper(
    tmp_path: Path,
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    inventory_path = target / "audit/inherited_fit_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["files"][0]["sha256"] = "0" * 64
    inventory["files_canonical_sha256"] = prepare._canonical_json_sha(
        inventory["files"]
    )
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _assert_all_closures_reject(repo, study, source, target)


def test_lineage_policy_one_owner_literal_drift_is_rejected(
    tmp_path: Path,
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    owner = target / "verify_version_forward.py"
    text = owner.read_text(encoding="utf-8")
    marker = "LINEAGE_SUPPORT_UNIT_POLICY: dict[str, dict[str, Any]] = {"
    assert text.count(marker) == 1
    owner.write_text(
        text.replace(marker, marker + "'drift.py': {}, ", 1),
        encoding="utf-8",
    )
    source_seal = json.loads(
        (source / "audit/pre_data_inheritance_seal.json").read_text(
            encoding="utf-8"
        )
    )
    closure = prepare._manifest_closure(source, target, repo, source_seal)
    with pytest.raises(prepare.PreparationError, match="transcription drift"):
        prepare._partition_sources(source, target, repo, closure)


def test_coherently_rehashed_lineage_record_cannot_self_authorize_unit_delta(
    tmp_path: Path,
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    seal = prepare.build_inheritance_seal(study)
    closure = seal["v008_manifest_closure"]
    relative = "tests/test_workflow.py"
    target_path = target / relative
    text = target_path.read_text(encoding="utf-8")
    marker = "import json\n"
    assert marker in text
    target_path.write_text(
        text.replace(marker, "import json as json_module\n", 1),
        encoding="utf-8",
    )

    record = next(
        item
        for item in seal["source_partitions"]["lineage_support"]
        if item["relative_path"] == relative
    )
    left = prepare._top_level_symbols(source / relative)
    right = prepare._top_level_symbols(target_path)
    changed = sorted(
        label
        for label in set(left) | set(right)
        if left.get(label) != right.get(label)
    )
    unchanged = [
        [label, digest]
        for label, digest in left.items()
        if label in right and digest == right[label]
    ]
    record.update(
        {
            "target_sha256": prepare.sha256_file(target_path),
            "source_top_level_unit_count": len(left),
            "target_top_level_unit_count": len(right),
            "source_top_level_unit_sequence_sha256": prepare._canonical_json_sha(
                list(left)
            ),
            "target_top_level_unit_sequence_sha256": prepare._canonical_json_sha(
                list(right)
            ),
            "required_changed_top_level_units": changed,
            "allowed_changed_top_level_units": changed,
            "observed_changed_top_level_units": changed,
            "inserted_top_level_units": sorted(set(right) - set(left)),
            "deleted_top_level_units": sorted(set(left) - set(right)),
            "unchanged_top_level_unit_count": len(unchanged),
            "unchanged_top_level_units_sha256": prepare._canonical_json_sha(
                unchanged
            ),
        }
    )

    with pytest.raises(prepare.PreparationError, match="unit policy drift"):
        prepare._lineage_support_scope(source / relative, target_path, relative)
    with pytest.raises(verify.VerificationError, match="unit policy drift"):
        verify._verify_records(seal, source, target, repo, closure)
    policy = independent.PathPolicy(
        repo,
        "runs/lewm_domain_robust_gate/attempts/v008",
        "runs/lewm_domain_robust_gate",
    )
    with pytest.raises(independent.VerificationError, match="unit policy drift"):
        independent.verify_equivalence_partitions(seal, policy)


@pytest.mark.parametrize(
    "mutation",
    (
        "cache_directory",
        "cache_file",
        "root_shadow",
        "source_addition",
        "inherited_fit_addition",
        "arbitrary_generated_json",
        "wrong_phase_generated_json",
        "empty_directory",
        "fifo",
        "cross_attempt_hardlink",
        "valid_seal_symlink",
        "broken_seal_symlink",
        "seal_hardlink",
    ),
)
def test_all_three_closures_reject_complete_namespace_attacks(
    tmp_path: Path, mutation: str
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    if mutation == "cache_directory":
        (target / "__pycache__").mkdir()
    elif mutation == "cache_file":
        (target / "runner.py.bak").write_text("shadow\n", encoding="utf-8")
    elif mutation == "root_shadow":
        (study / "shadow.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "source_addition":
        (source / "shadow.md").write_text("shadow\n", encoding="utf-8")
    elif mutation == "inherited_fit_addition":
        path = study / "attempts/v003/data/fit/native_plan/unrecorded.json"
        path.write_text("{}\n", encoding="utf-8")
    elif mutation == "arbitrary_generated_json":
        path = target / "data/fit/native_plan/raw/arbitrary.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
    elif mutation == "wrong_phase_generated_json":
        path = target / "data/selection/markov_oracle/raw_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
    elif mutation == "empty_directory":
        (target / "empty").mkdir()
    elif mutation == "fifo":
        os.mkfifo(target / "unexpected.fifo")
    elif mutation == "cross_attempt_hardlink":
        path = target / "analysis.py"
        path.unlink()
        os.link(source / "analysis.py", path)
    elif mutation in {"valid_seal_symlink", "broken_seal_symlink"}:
        path = target / "audit/pre_data_inheritance_seal.json"
        path.symlink_to(
            target / "analysis.py"
            if mutation == "valid_seal_symlink"
            else target / "absent.json"
        )
    else:
        path = target / "audit/pre_data_inheritance_seal.json"
        os.link(target / "audit/bootstrap_audit.json", path)
    _assert_all_closures_reject(repo, study, source, target)


def test_target_manifest_cannot_self_approve_new_static_path(
    tmp_path: Path,
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    manifest_source = target / "build_manifest.py"
    text = manifest_source.read_text(encoding="utf-8")
    marker = '        "analysis.py",\n'
    assert marker in text
    manifest_source.write_text(
        text.replace(marker, marker + '        "self_approved.json",\n', 1),
        encoding="utf-8",
    )
    (target / "self_approved.json").write_text("{}\n", encoding="utf-8")
    _assert_all_closures_reject(repo, study, source, target)


def test_target_manifest_cannot_swap_static_and_audit_memberships(
    tmp_path: Path,
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    manifest_source = target / "build_manifest.py"
    text = manifest_source.read_text(encoding="utf-8")
    static_marker = '        "analysis.py",\n'
    audit_marker = "PRE_DATA_AUDIT_INPUTS = frozenset(\n    {\n"
    assert static_marker in text and audit_marker in text
    text = text.replace(static_marker, "", 1)
    text = text.replace(audit_marker, audit_marker + static_marker, 1)
    manifest_source.write_text(text, encoding="utf-8")
    _assert_all_closures_reject(repo, study, source, target)


def test_target_only_fit_inheritance_cannot_be_removed(
    tmp_path: Path,
) -> None:
    repo, study, source, target = _copy_live_study(tmp_path)
    (target / "fit_inheritance.py").unlink()
    _assert_all_closures_reject(repo, study, source, target)
