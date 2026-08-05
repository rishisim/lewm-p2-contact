#!/usr/bin/env python3
"""Standalone, read-only verifier for the domain-robust gate study.

The verifier deliberately imports no project module.  Its only non-stdlib
dependency is NumPy.  A pre-confirmation ``verifier_contract.json`` supplies
repository-relative artifact paths; scientific invariants are independently
transcribed here from the frozen v001 preregistration.

The process writes nothing.  Its sole normal output is one JSON object on
stdout.  ``capture_verifier.py`` is the separate non-scientific wrapper that
may exclusively persist that object.

Contract schema (schema_version 1)
----------------------------------
Required scalar fields are ``attempt``, ``attempt_root``, ``study_root``,
``mode`` (``confirmation``, ``no_candidate``, or ``power_infeasible``), and
``expected_git_head``.  ``paths`` contains the mode-specific names in
``MODE_PATH_NAMES`` as canonical repository-relative paths.  Every mode also
supplies:

``fit_inputs`` / ``selection_inputs`` and ``development_manifests``
    Four-DGP mappings to role-isolated five-array aggregates and the sealed
    per-episode primitive manifests used to reconstruct those aggregates.
``role_manifests``
    ``role -> DGP -> manifest`` mappings for fit, selection, smoke, and
    confirmation episodes.
``confirmation_manifests``
    Four execution-manifest paths, one per DGP.
``terminal_manifest_exclusions``
    Exact repository-relative files omitted from the terminal path-set
    manifest (normally that manifest itself and the not-yet-created captured
    audit).  Patterns and directory exclusions are not accepted.

Confirmation mode additionally supplies ``confirmation_manifests`` and
``confirmation_artifact_roots``.  An optional ``expected_hashes`` mapping adds direct hash pins.  All other hash
links are recovered from the sealed manifests, locks, state, and ledger.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import math
import os
import re
import stat
import sys
from fractions import Fraction
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True

SCHEMA_VERSION = 1
ACTIVE_ATTEMPT = "v008"
SCIENCE_ATTEMPT = "v001"
PRIOR_ATTEMPT = "v002"
FIT_SOURCE_ATTEMPT = "v003"
INTERMEDIATE_ATTEMPT = "v004"
SOURCE_PARENT_ATTEMPT = "v005"
SOURCE_ATTEMPT = "v006"
ACTIVATION_SOURCE_ATTEMPT = "v007"
ALLOWED_ATTEMPTS = (
    SCIENCE_ATTEMPT,
    PRIOR_ATTEMPT,
    FIT_SOURCE_ATTEMPT,
    INTERMEDIATE_ATTEMPT,
    SOURCE_PARENT_ATTEMPT,
    SOURCE_ATTEMPT,
    ACTIVATION_SOURCE_ATTEMPT,
    ACTIVE_ATTEMPT,
)
# ``ATTEMPT`` is retained as the local shorthand used throughout the
# independently transcribed verifier.  Unlike the v001 verifier it is not
# contract-selected: this source can authenticate exactly one immutable
# lineage tip.
ATTEMPT = ACTIVE_ATTEMPT

STUDY_RELATIVE = "runs/lewm_domain_robust_gate"
ATTEMPT_ROOTS = {
    SCIENCE_ATTEMPT: f"{STUDY_RELATIVE}/attempts/{SCIENCE_ATTEMPT}",
    PRIOR_ATTEMPT: f"{STUDY_RELATIVE}/attempts/{PRIOR_ATTEMPT}",
    FIT_SOURCE_ATTEMPT: f"{STUDY_RELATIVE}/attempts/{FIT_SOURCE_ATTEMPT}",
    INTERMEDIATE_ATTEMPT: f"{STUDY_RELATIVE}/attempts/{INTERMEDIATE_ATTEMPT}",
    SOURCE_PARENT_ATTEMPT: f"{STUDY_RELATIVE}/attempts/{SOURCE_PARENT_ATTEMPT}",
    SOURCE_ATTEMPT: f"{STUDY_RELATIVE}/attempts/{SOURCE_ATTEMPT}",
    ACTIVATION_SOURCE_ATTEMPT: f"{STUDY_RELATIVE}/attempts/{ACTIVATION_SOURCE_ATTEMPT}",
    ACTIVE_ATTEMPT: f"{STUDY_RELATIVE}/attempts/{ACTIVE_ATTEMPT}",
}
INVALIDITY_RELATIVE = (
    f"{ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]}/audit/v007_procedural_invalidity.json"
)
INVALIDITY_DRAFT_RELATIVE = (
    f"{ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]}/audit/v007_procedural_invalidity_draft.json"
)
SOURCE_PRE_DATA_RELATIVE = (
    f"{ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]}/audit/pre_data_inheritance_seal.json"
)
SOURCE_RECEIPT_RELATIVE = (
    f"{ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]}/audit/version_forward_transaction_receipt.json"
)
SOURCE_PRE_SELECTION_RELATIVE = (
    f"{ATTEMPT_ROOTS[SOURCE_ATTEMPT]}/audit/pre_selection_seal.json"
)
SOURCE_PARENT_INVALIDITY_RELATIVE = (
    f"{ATTEMPT_ROOTS[SOURCE_PARENT_ATTEMPT]}/audit/v005_procedural_invalidity.json"
)
SOURCE_PARENT_PRE_DATA_RELATIVE = (
    f"{ATTEMPT_ROOTS[SOURCE_PARENT_ATTEMPT]}/audit/pre_data_inheritance_seal.json"
)
SOURCE_PARENT_RECEIPT_RELATIVE = (
    f"{ATTEMPT_ROOTS[SOURCE_PARENT_ATTEMPT]}/audit/version_forward_transaction_receipt.json"
)
INTERMEDIATE_INVALIDITY_RELATIVE = (
    f"{ATTEMPT_ROOTS[INTERMEDIATE_ATTEMPT]}/audit/v004_procedural_invalidity.json"
)
INTERMEDIATE_PRE_DATA_RELATIVE = (
    f"{ATTEMPT_ROOTS[INTERMEDIATE_ATTEMPT]}/audit/pre_data_inheritance_seal.json"
)
INTERMEDIATE_RECEIPT_RELATIVE = (
    f"{ATTEMPT_ROOTS[INTERMEDIATE_ATTEMPT]}/audit/version_forward_transaction_receipt.json"
)
FIT_INVALIDITY_RELATIVE = (
    f"{ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]}/audit/v003_procedural_invalidity.json"
)
FIT_INVALIDITY_DRAFT_RELATIVE = (
    f"{ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]}/audit/v003_procedural_invalidity_draft.json"
)
FIT_PRE_DATA_RELATIVE = (
    f"{ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]}/audit/pre_data_inheritance_seal.json"
)
FIT_RECEIPT_RELATIVE = (
    f"{ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]}/audit/version_forward_transaction_receipt.json"
)
PRIOR_INVALIDITY_RELATIVE = (
    f"{ATTEMPT_ROOTS[PRIOR_ATTEMPT]}/audit/v002_procedural_invalidity_v2.json"
)
PRIOR_PRE_DATA_RELATIVE = (
    f"{ATTEMPT_ROOTS[PRIOR_ATTEMPT]}/audit/pre_data_inheritance_seal.json"
)
PRIOR_RECEIPT_RELATIVE = (
    f"{ATTEMPT_ROOTS[PRIOR_ATTEMPT]}/audit/version_forward_transaction_receipt.json"
)
SCIENCE_INVALIDITY_RELATIVE = (
    f"{ATTEMPT_ROOTS[SCIENCE_ATTEMPT]}/audit/v001_procedural_invalidity.json"
)
SCIENCE_PRE_DATA_RELATIVE = (
    f"{ATTEMPT_ROOTS[SCIENCE_ATTEMPT]}/audit/pre_data_seal.json"
)
INHERITANCE_SEAL_RELATIVE = (
    f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/audit/pre_data_inheritance_seal.json"
)
FIT_INVENTORY_RELATIVE = (
    f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/audit/inherited_fit_inventory.json"
)
EXPECTED_FIT_INVENTORY_FILE_COUNT = 6013
EXPECTED_FIT_EPISODE_COUNT = 1200
EXPECTED_TRANSITIVE_SEALED_FILE_COUNT = 6541
EXPECTED_SOURCE_STATE_SHA256 = (
    "0493b0f1bcb4c2d1b91fdc51548d64af20d568709ee3aa33611c1d234dd030c5"
)
EXPECTED_SOURCE_LEDGER_SHA256 = (
    "a6a9374dddddfc20254d8fd532252c045cec8a7c4ab31a06ea7a703de542d0db"
)
EXPECTED_SOURCE_INVALIDITY_SHA256 = (
    "adddaa7ebf470866365da414d74f6094f4aefaeb02480e81c5b75961204d584e"
)
EXPECTED_SOURCE_SEAL_SHA256 = (
    "8868038420c049f649b2e37c4b17dd87dbbbc4d8042b43fb003b4efa8e288190"
)
EXPECTED_SOURCE_RECEIPT_SHA256 = (
    "f038c25974c3ddd1bf7ea4ae9e9be68d1230a359653d6ba3c90558e9034b0818"
)
EXPECTED_SOURCE_INVALIDITY_DRAFT_SHA256 = (
    "83014917878d8fc226add312284f6c34795ee88d6f49dab80079f04bacea75b5"
)
EXPECTED_SOURCE_ACTIVATION_STATE_SHA256 = (
    "3548d4cbd0643cfb01b508e6af80c948d31d9a54cec482c7549c4f6e08657475"
)
EXPECTED_SOURCE_ACTIVATION_LEDGER_SHA256 = (
    "c12771fad7dd94fb00a153130834434c47255d3ec4d56a0be78e5d100ddac7dc"
)
EXPECTED_SOURCE_PRE_SELECTION_SHA256 = (
    "3ebff0d56f515b399a4000df2d20f39314cd2e2aadb31bd271ab75f42e58327c"
)
EXPECTED_SELECTION_SOURCE_SEAL_SHA256 = (
    "8f5c24d0903317bacbd3722fdd13abc8697a4ca98d1f6e9456541d620609b499"
)
EXPECTED_FIT_SOURCE_STATE_SHA256 = (
    "53788a9e7c1633f95c66af682d69b32bfcc73baee4db4e4c3176e0fb9456d821"
)
EXPECTED_FIT_SOURCE_LEDGER_SHA256 = (
    "611be25f22716de7b593dea523904fc871b854962a8fe9e936029e4ee0dd237b"
)
EXPECTED_FIT_SOURCE_INVALIDITY_SHA256 = (
    "25be74578dc22eb2c1b4a3cde6b0dbd7e5a788d1bcb146f251b76fa40bc95165"
)
EXPECTED_FIT_SOURCE_SEAL_SHA256 = (
    "74018164e9e208babd7a3b420053a58e665e98694e4116d5a4f1cf70f143d943"
)
EXPECTED_FIT_SOURCE_RECEIPT_SHA256 = (
    "8cdd1a24aacdaecf4eda85fb5b9119536d88bec92e0cc64db6a3bcc7d9ecb262"
)
EXPECTED_SOURCE_COHORT_LEDGER_SHA256 = (
    "1b7f524ed11cf787e67e4887590d985cfa4eb928e333e7adb7667e48a0b8785d"
)
EXPECTED_SOURCE_REPLACEMENT_REGISTRY_SHA256 = (
    "f50e060aa616d6ad8616fb6d28e768e2e5cc10999097d95928af7926d50fab46"
)
EXPECTED_FIT_SOURCE_REPLACEMENT_REGISTRY_SHA256 = (
    "7aa18ee9ab56e45a3f3aac8ff95ba75978debb2e72368e5ee9e78a1f27c2b871"
)
SOURCE_OPERATIONAL_LOCKS = frozenset()
SOURCE_SELECTION_BOUNDARY_PRODUCTS = frozenset()
EXPECTED_ALLOWED_SOURCE_SUFFIXES = frozenset({".py", ".json", ".md"})
EXPECTED_PRE_DATA_STATIC_PATHS = frozenset({
    "DGP_MATRIX.json", "DIAGNOSTIC_ACCOUNT.md", "PREREGISTRATION.md", "analysis.py",
    "build_manifest.py", "build_seed_ledger.py", "candidate_grid.json", "capture_verifier.py",
    "checkpoints.py", "cohort_seed_ledger.json", "compile_gate.py", "counted_features.py",
    "fit_inheritance.py", "fit_select.py", "flops.py", "generator.py", "inherited_authorization.py",
    "independent_verify.py", "input_loader.py", "latency.py", "launcher.py",
    "outcome_mapping.json", "power_analysis.py", "power_baseline.json", "power_rule.json",
    "preseal.py", "runner.py", "runtime_contract.py", "scientific_replay.py",
    "scientific_replay_launcher.py", "study_common.py",
    "terminal_workflow.py", "tests/test_analysis.py",
    "tests/test_attempt_parameterization_v002.py", "tests/test_checkpoint_hardening.py",
    "tests/test_fit_select.py", "tests/test_generator_contract.py",
    "tests/test_independent_verify.py", "tests/test_inherited_authorization_v002.py",
    "tests/test_manifest_closure_v002.py", "tests/test_preseal.py",
    "tests/test_program_controller.py", "tests/test_runner_gate.py",
    "tests/test_scientific_replay.py", "tests/test_seed_power.py",
    "tests/test_terminal_workflow.py", "tests/test_version_forward_v002.py",
    "tests/test_version_forward_transaction.py",
    "tests/test_workflow.py", "verifier_contract.json",
    "verifier_contract_no_candidate.json", "verifier_contract_power_infeasible.json",
    "verify_identifier_freshness.py", "verify_version_forward.py",
    "version_forward_prepare.py", "version_forward_transaction.py", "workflow.py",
})
EXPECTED_PRE_DATA_AUDIT_INPUTS = frozenset({
    "audit/bootstrap_audit.json", "audit/diagnostic_account.json",
    "audit/identifier_freshness_verification.json", "audit/preregistration_and_power.json",
})
EXPECTED_REPOSITORY_PATHS = frozenset({
    "runs/lewm_domain_robust_gate/LEDGER_CHAIN_GENESIS.json",
    "runs/lewm_domain_robust_gate/README.md", "runs/lewm_domain_robust_gate/program.py",
})
EXPECTED_MUTABLE_REPOSITORY_PATHS = frozenset({
    "runs/lewm_domain_robust_gate/STATE.json",
    "runs/lewm_domain_robust_gate/STATE_TRANSACTION.json",
    "runs/lewm_domain_robust_gate/VERSION_FORWARD_TRANSACTION.json",
})
EXPECTED_GENERATED_AUDIT_PATHS = frozenset({
    "audit/analysis_execution_invalid.json", "audit/candidate_selection.json",
    "audit/confirmation_execution_complete.json", "audit/confirmation_generation_complete.json",
    "audit/confirmation_input_seal.json", "audit/confirmation_power_and_cohort_freeze.json",
    "audit/confirmation_scientific_replay.json",
    "audit/excluded_mechanical_smoke.json", "audit/fit_cohorts.json",
    "audit/fit_lock_checkpoint.json", "audit/fit_scientific_replay.json",
    "audit/gate_freeze_checkpoint.json",
    "audit/implementation_complete.json", "audit/inherited_fit_inventory.json",
    "audit/independent_verification.json",
    "audit/post_terminal_reporting.json", "audit/pre_confirmation_manifest.json",
    "audit/pre_confirmation_package_seal.json", "audit/pre_confirmation_seal.json",
    "audit/pre_data_inheritance_seal.json", "audit/pre_data_seal.json",
    "audit/version_forward_transaction_receipt.json",
    "audit/pre_data_source_manifest.json", "audit/pre_selection_manifest.json",
    "audit/pre_selection_seal.json", "audit/preseal_qualification.json",
    "audit/scientific_replay/confirmation_markov_oracle.json",
    "audit/scientific_replay/confirmation_native_plan.json",
    "audit/scientific_replay/confirmation_plan_action_noise_0p2.json",
    "audit/scientific_replay/confirmation_plan_random_action_0p1.json",
    "audit/scientific_replay/fit_markov_oracle.json",
    "audit/scientific_replay/fit_native_plan.json",
    "audit/scientific_replay/fit_plan_action_noise_0p2.json",
    "audit/scientific_replay/fit_plan_random_action_0p1.json",
    "audit/scientific_replay/selection_markov_oracle.json",
    "audit/scientific_replay/selection_native_plan.json",
    "audit/scientific_replay/selection_plan_action_noise_0p2.json",
    "audit/scientific_replay/selection_plan_random_action_0p1.json",
    "audit/scientific_replay/smoke_markov_oracle.json",
    "audit/scientific_replay/smoke_native_plan.json",
    "audit/scientific_replay/smoke_plan_action_noise_0p2.json",
    "audit/scientific_replay/smoke_plan_random_action_0p1.json",
    "audit/scientific_replay_qualification.json",
    "audit/selection_cohorts.json", "audit/selection_scientific_replay.json",
    "audit/smoke_scientific_replay.json", "audit/terminal_input_manifest.json",
})
EXPECTED_GENERATED_ATTEMPT_PRODUCTS = frozenset({
    "FOLLOW_ON_TASK.md", "INDEPENDENT_AUDIT.md", "LIMITATIONS.md", "REPORT.md",
    "ROBUSTNESS_MAP.json", "analysis_result.json", "decision.json", "fit/fit_lock.json",
    "freeze/compiled_gate_manifest.json", "freeze/gate_freeze.json",
    "metrics/bootstrap_summary.json", "metrics/compute_ledger.json",
    "metrics/fit_power_summary.json", "metrics/latency_and_resources.json",
    "metrics/selection_power_summary.json", "power_analysis.json",
    "selection/selection_ledger.json",
})
EXPECTED_GENERATED_ROLES = frozenset({"fit", "selection", "smoke", "confirmation"})
EXPECTED_GENERATED_REGIMES = frozenset({
    "native_plan", "markov_oracle", "plan_action_noise_0p2", "plan_random_action_0p1",
})
EXPECTED_STUDY_ROOT_OPERATIONAL_NAMES = frozenset({".program.lock", "RESEARCH_LEDGER.jsonl"})
EXPECTED_STUDY_ROOT_DIRECTORY_NAMES = frozenset({"attempts"})

NORMALIZED_AST_PATHS = frozenset(
    {
        'analysis.py',
        'checkpoints.py',
        'latency.py',
        'power_analysis.py',
        'scientific_replay_launcher.py',
        'terminal_workflow.py',
        'tests/test_analysis.py',
        'tests/test_checkpoint_hardening.py',
        'tests/test_preseal.py',
        'tests/test_runner_gate.py',
        'tests/test_seed_power.py',
        'tests/test_terminal_workflow.py',
    }
)
CANONICAL_CONTRACT_PATHS = frozenset(
    {
        'verifier_contract.json',
        'verifier_contract_no_candidate.json',
        'verifier_contract_power_infeasible.json',
    }
)
PROCEDURAL_EXISTING_PATHS = frozenset({
    "analysis.py", "build_manifest.py", "capture_verifier.py", "checkpoints.py",
    "compile_gate.py", "generator.py", "independent_verify.py", "latency.py", "launcher.py",
    "preseal.py", "runner.py", "study_common.py", "tests/test_analysis.py",
    "tests/test_checkpoint_hardening.py", "tests/test_generator_contract.py",
    "tests/test_independent_verify.py", "tests/test_preseal.py",
    "tests/test_runner_gate.py", "tests/test_seed_power.py",
    "tests/test_terminal_workflow.py", "tests/test_workflow.py", "terminal_workflow.py", "workflow.py",
})
PROCEDURAL_ALLOWED_SYMBOLS = {
    "analysis.py": frozenset({
        "ACTIVE_ATTEMPT", "ATTEMPT", "_fsync_parent_directory", "_publish_temporary",
        "atomic_json", "atomic_npz", "capture_post_open_analysis_failure",
        "run_sealed_analysis", "validate_sealed_analysis_authorization",
    }),
    "build_manifest.py": frozenset({
        "<docstring>", "ACTIVE_ATTEMPT", "EXPECTED_ACTIVE_ATTEMPT",
        "SCIENCE_ATTEMPT", "PRE_DATA_STATIC_RELATIVE_PATHS",
        "PRE_DATA_REPOSITORY_RELATIVE_PATHS", "GENERATED_AUDIT_NAMES",
        "GENERATED_ATTEMPT_PRODUCTS", "GENERATED_ROLES", "GENERATED_REGIMES",
        "GENERATED_EXECUTION_FAILURE_NAMES", "MUTABLE_STUDY_ROOT_RELATIVE_PATHS",
        "PRE_DATA_LABEL", "_study_root", "_allowed_suffix", "_walk_attempt_candidates",
        "_walk_study_root_candidates", "_assert_discovered_integrity",
        "_is_generated_data_json", "classify_pre_data_source_paths",
        "collect_pre_data_source_paths", "is_generated_attempt_product",
        "source_closure_policy",
        "discovered_python_source_paths", "unexpected_pre_data_source_paths",
        "unexpected_python_source_paths", "build_pre_data_manifest", "verify_manifest",
        "_manifest_for_paths",
    }),
    "capture_verifier.py": frozenset(),
    "checkpoints.py": frozenset({
        "<imports>", "ATTEMPT", "CONFIRMATION_MAX_EPISODES_PER_DGP",
        "CONFIRMATION_REPLACEMENTS_PER_DGP", "REGIME_SLUGS", "ROLE_ORDER",
        "ROLE_PRIMARY_COUNTS", "SEED_FIELDS", "V5_FIXED_WHITENING",
        "_assert_exact_confirmation_namespace", "_attempt_root_from_manifest",
        "_canonical_relative", "_confirmation_contract", "_confirmation_files",
        "_exact_compute_from_histograms", "_exact_keys", "_exact_seal_link",
        "_expected_episode_id", "_observed_file_link", "_seed_tuple",
        "_strict_file_link", "_strict_int", "_strict_sha256",
        "_validate_raw_array_metadata", "_validate_rollout_failure_log",
        "_validate_seed_record", "_validate_source_binding",
        "_validate_source_raw_manifest_binding", "_verify_confirmation_execution_manifest",
        "_verify_confirmation_raw_manifest", "atomic_json",
        "confirmation_execution_checkpoint", "confirmation_generation_checkpoint",
        "confirmation_input_seal", "verify_execution_manifest", "verify_raw_manifest",
    }),
    "compile_gate.py": frozenset({
        "_atomic_json", "_atomic_npz", "_fsync_parent_directory",
        "_publish_exclusive_temporary",
    }),
    "generator.py": frozenset({
        "<imports>", "ATTEMPT", "EPISODE_ID_PATTERN", "PRE_DATA_SEAL_PATH",
        "REGIME_SLUGS", "ROLE_SLUGS", "_append_jsonl", "_assert_safe_directory_chain",
        "_assert_safe_output_leaf", "_canonical_output_file", "_expected_episode_id",
        "_failure_descriptor", "_failure_genesis", "_failure_hash", "_generate_locked",
        "_lexical_attempt_relative", "_strict_failure_records",
        "_validate_destination_episode_id", "_validate_episode_id", "_validate_seed_record",
        "episode_paths", "generate", "load_generation_inputs", "materialization_lock",
        "materialization_lock_path", "preflight_output_namespace", "raw_directory",
        "read_rollout_failures", "record_rollout_failure", "validate_dgp_matrix",
        "validate_existing_manifest", "validate_raw_manifest_contract", "validate_seed_ledger",
        "verify_authorization_seal", "verify_episode_record",
    }),
    "independent_verify.py": frozenset({
        "<imports>", "ACTIVE_ATTEMPT", "ALLOWED_ATTEMPTS", "ATTEMPT", "ATTEMPT_ROOTS",
        "CANONICAL_CONTRACT_PATHS", "INHERITANCE_SEAL_RELATIVE",
        "INHERITED_COMPLETED_STATES", "INVALIDITY_RELATIVE", "NEW_LINEAGE_SUPPORT_PATHS",
        "NORMALIZED_AST_PATHS", "PROCEDURAL_ALLOWED_SYMBOLS",
        "PROCEDURAL_EXPECTED_CHANGED_SYMBOLS", "PROCEDURAL_EXISTING_PATHS",
        "SCIENTIFIC_OBJECT_PATHS", "SOURCE_PRE_DATA_RELATIVE", "STUDY_RELATIVE",
        "VERSION_FORWARD_RESUME_STATE", "_AdministrativeAstNormalizer",
        "_assignment_names", "_canonicalize_contract", "_is_active_attempt_guard",
        "_lineage_file_link", "_normalize_normative_sources",
        "_source_manifest_relative_paths", "_top_level_symbol_hashes",
        "canonical_contract_sha256", "load_contract", "normalized_administrative_ast_sha256",
        "verify", "verify_analysis_execution_invalid_branch", "verify_confirmation_analysis",
        "verify_equivalence_partitions", "verify_ledger_and_state", "verify_role_manifests",
        "verify_seal", "verify_version_forward_lineage",
    }),
    "latency.py": frozenset({
        "ACTIVE_ATTEMPT", "_fsync_parent_directory", "_publish_temporary", "atomic_json",
        "run_latency_suite", "validate_latency_result",
    }),
    "preseal.py": frozenset({
        "<imports>", "ACTIVE_ATTEMPT", "PRE_DATA_SEAL_PATH", "REQUIRED_IMPLEMENTATION_FILES",
        "SCIENCE_ATTEMPT", "_files_under", "_state", "_verified_checkpoint",
        "implementation_complete", "preseal_qualification", "seal_pre_confirmation",
        "seal_pre_data", "seal_pre_selection", "validate_power_input_coherence",
        "validate_seed_freshness_evidence", "validate_verifier_contracts",
    }),
    "runner.py": frozenset({
        "<imports>", "ATTEMPT_VERSION", "_assert_regular_unlinked",
        "_enforce_execution_runtime", "_execute_confirmation_locked", "_execute_development_locked",
        "_execution_record_if_valid", "_execution_source_bindings", "_load_raw_manifest",
        "_preflight_execution_namespace", "_state_authorization",
        "_validate_existing_execution_manifest", "execute_confirmation", "execute_development",
    }),
    "study_common.py": frozenset({
        "_fsync_parent_directory", "_publish_temporary", "atomic_json", "atomic_npz",
    }),
    "tests/test_analysis.py": frozenset({
        "_authorization_fixture", "_latency_binding_and_result",
        "test_post_open_failure_capture_is_outcome_free_and_never_overwrites",
    }),
    "tests/test_checkpoint_hardening.py": frozenset({
        "<imports>", "JSON_EXCLUSIVE_WRITERS", "NPZ_EXCLUSIVE_WRITERS",
        "test_confirmation_file_link_rejects_symlink_and_inode_alias",
        "test_confirmation_file_link_rejects_valid_hash_at_wrong_path",
        "test_confirmation_namespace_rejects_unrelated_file",
        "test_exclusive_json_concurrent_writers_never_replace",
        "test_exclusive_json_preserves_preexisting_destination",
        "test_exclusive_npz_concurrent_writers_never_replace",
        "test_exclusive_npz_preserves_preexisting_destination",
        "test_execution_manifest_requires_explicit_unopened_fields",
        "test_execution_source_raw_manifest_hash_is_mandatory",
        "test_nonexclusive_json_intentionally_retains_replace_semantics",
        "test_nonexclusive_npz_intentionally_retains_replace_semantics",
        "test_raw_manifest_rejects_phase_as_role_alias",
    }),
    "tests/test_generator_contract.py": frozenset({
        "test_intent_only_partial_artifact_stops_without_reroll",
        "test_paths_are_role_and_regime_isolated",
        "test_replacement_registry_is_global_across_roles_and_dgps",
    }),
    "tests/test_independent_verify.py": frozenset({
        "_policy", "test_analysis_execution_invalid_branch_is_preload_and_binds_result_presence",
        "test_power_provenance_is_local_identity_bound_and_tamper_evident",
        "test_v008_equivalence_partitions_are_independently_recomputed_and_tamper_evident",
        "test_v008_equivalence_partitions_reject_omitted_required_symbol",
        "test_v008_equivalence_partitions_reject_wildcard_procedural_scope",
        "test_v008_lineage_normalization_and_contract_canonicalization_are_tamper_evident",
        "test_verifier_imports_only_stdlib_and_numpy_and_failure_stdout_is_one_json_line",
        "test_zero_confirmation_version_forward_requires_explicit_equivalence",
    }),
    "tests/test_preseal.py": frozenset({
        "<docstring>", "test_compile_and_power_are_state_gated_inside_exact_eval_worker",
        "test_power_input_paths_resolve_from_repository_not_cwd",
        "test_power_provenance_binds_rule_locks_freeze_and_selected_identity",
        "test_pre_data_manifest_ignores_later_stage_top_level_products",
        "test_pre_data_manifest_rejects_unlisted_python_source",
        "test_preseal_state_uses_verified_controller_not_direct_state_read",
        "test_python_cache_hygiene_fails_closed",
        "test_stage_file_collection_rejects_supplied_directory_symlink_before_rglob",
    }),
    "tests/test_runner_gate.py": frozenset({
        "test_complete_confirmation_part_is_idempotently_resumed",
        "test_execution_handlers_do_not_capture_process_interruptions",
    }),
    "tests/test_seed_power.py": frozenset({
        "_claim_summary", "test_power_cli_binds_gate_locks_and_persists_only_repo_relative_paths",
    }),
    "tests/test_terminal_workflow.py": frozenset({
        "_analysis", "_attempt", "_captured_audit", "_contract", "_state",
        "test_early_decision_has_zero_later_roles_and_reports_are_claim_bounded",
        "test_report_artifact_mutation_is_not_repaired",
        "test_result_present_analysis_integrity_failure_is_manifested_but_not_called",
    }),
    "tests/test_workflow.py": frozenset({
        "<docstring>", "FakeController", "_configure_power_paths", "_controller_json",
        "_install_fit_count_crash", "_ledger_record", "_power_result",
        "_prior_selection_audit", "_synthetic_role_manifests",
        "test_count_crash_recovery_rejects_state_ledger_audit_and_manifest_tamper",
        "test_default_freeze_writes_identity_summaries_only_after_gate_freeze",
        "test_exact_role_count_is_resume_idempotent",
        "test_exact_role_count_without_authenticated_crash_record_fails_closed",
        "test_fit_count_commit_crash_reauthenticates_manifests_and_advances_once",
        "test_inherited_early_stop_adapter_rejects_unsealed_active_contract",
        "test_loaded_workflow_controller_adapts_both_inherited_early_stop_contracts",
        "test_sealed_analysis_integrity_failure_is_staged_not_advanced",
    }),
    "workflow.py": frozenset({
        "<imports>", "ATTEMPT", "DEFAULT_EVIDENCE_PATHS", "LEDGER_PATH",
        "ROLE_COUNTER_FIELDS", "ROLE_EXECUTION_SEALS", "ROLE_RAW_SEALS", "ROLE_SEALS",
        "Workflow", "_common_role_authorization_state_sha256",
        "_controller_state_object_sha256", "_load_controller",
        "_require_inherited_presealed_verifier_contract", "_verify_role_count_update_recovery",
        "_verify_role_regime", "verify_role_manifests",
    }),
}
PROCEDURAL_EXPECTED_CHANGED_SYMBOLS = {
    "analysis.py": frozenset({"ACTIVE_ATTEMPT", "ATTEMPT", "_fsync_parent_directory", "_publish_temporary", "atomic_json", "atomic_npz", "capture_post_open_analysis_failure", "run_sealed_analysis", "validate_sealed_analysis_authorization"}),
    "build_manifest.py": frozenset({"<docstring>", "ACTIVE_ATTEMPT", "EXPECTED_ACTIVE_ATTEMPT", "GENERATED_ATTEMPT_PRODUCTS", "GENERATED_AUDIT_NAMES", "GENERATED_EXECUTION_FAILURE_NAMES", "GENERATED_REGIMES", "GENERATED_ROLES", "MUTABLE_STUDY_ROOT_RELATIVE_PATHS", "PRE_DATA_LABEL", "PRE_DATA_REPOSITORY_RELATIVE_PATHS", "PRE_DATA_STATIC_RELATIVE_PATHS", "SCIENCE_ATTEMPT", "_allowed_suffix", "_assert_discovered_integrity", "_is_generated_data_json", "_manifest_for_paths", "_study_root", "_walk_attempt_candidates", "_walk_study_root_candidates", "build_pre_data_manifest", "classify_pre_data_source_paths", "collect_pre_data_source_paths", "discovered_python_source_paths", "is_generated_attempt_product", "source_closure_policy", "unexpected_pre_data_source_paths", "unexpected_python_source_paths", "verify_manifest"}),
    "capture_verifier.py": frozenset(),
    "checkpoints.py": frozenset({"<imports>", "ATTEMPT", "CONFIRMATION_MAX_EPISODES_PER_DGP", "CONFIRMATION_REPLACEMENTS_PER_DGP", "REGIME_SLUGS", "ROLE_ORDER", "ROLE_PRIMARY_COUNTS", "SEED_FIELDS", "V5_FIXED_WHITENING", "_assert_exact_confirmation_namespace", "_attempt_root_from_manifest", "_canonical_relative", "_confirmation_contract", "_confirmation_files", "_exact_compute_from_histograms", "_exact_keys", "_exact_seal_link", "_expected_episode_id", "_observed_file_link", "_seed_tuple", "_strict_file_link", "_strict_int", "_strict_sha256", "_validate_raw_array_metadata", "_validate_rollout_failure_log", "_validate_seed_record", "_validate_source_binding", "_validate_source_raw_manifest_binding", "_verify_confirmation_execution_manifest", "_verify_confirmation_raw_manifest", "atomic_json", "confirmation_execution_checkpoint", "confirmation_generation_checkpoint", "confirmation_input_seal", "verify_execution_manifest", "verify_raw_manifest"}),
    "compile_gate.py": frozenset({"_atomic_json", "_atomic_npz", "_fsync_parent_directory", "_publish_exclusive_temporary"}),
    "generator.py": frozenset({"<imports>", "ATTEMPT", "EPISODE_ID_PATTERN", "PRE_DATA_SEAL_PATH", "REGIME_SLUGS", "ROLE_SLUGS", "_append_jsonl", "_assert_safe_directory_chain", "_assert_safe_output_leaf", "_canonical_output_file", "_expected_episode_id", "_failure_descriptor", "_failure_genesis", "_failure_hash", "_generate_locked", "_lexical_attempt_relative", "_strict_failure_records", "_validate_destination_episode_id", "_validate_episode_id", "_validate_seed_record", "episode_paths", "generate", "load_generation_inputs", "materialization_lock", "materialization_lock_path", "preflight_output_namespace", "raw_directory", "read_rollout_failures", "record_rollout_failure", "validate_dgp_matrix", "validate_existing_manifest", "validate_raw_manifest_contract", "validate_seed_ledger", "verify_authorization_seal", "verify_episode_record"}),
    "independent_verify.py": frozenset({"<imports>", "ACTIVE_ATTEMPT", "ALLOWED_ATTEMPTS", "ATTEMPT", "ATTEMPT_ROOTS", "CANONICAL_CONTRACT_PATHS", "INHERITANCE_SEAL_RELATIVE", "INHERITED_COMPLETED_STATES", "INVALIDITY_RELATIVE", "NEW_LINEAGE_SUPPORT_PATHS", "NORMALIZED_AST_PATHS", "PROCEDURAL_ALLOWED_SYMBOLS", "PROCEDURAL_EXPECTED_CHANGED_SYMBOLS", "PROCEDURAL_EXISTING_PATHS", "SCIENTIFIC_OBJECT_PATHS", "SOURCE_PRE_DATA_RELATIVE", "STUDY_RELATIVE", "VERSION_FORWARD_RESUME_STATE", "_AdministrativeAstNormalizer", "_assignment_names", "_canonicalize_contract", "_is_active_attempt_guard", "_lineage_file_link", "_normalize_normative_sources", "_source_manifest_relative_paths", "_top_level_symbol_hashes", "canonical_contract_sha256", "load_contract", "normalized_administrative_ast_sha256", "verify", "verify_analysis_execution_invalid_branch", "verify_confirmation_analysis", "verify_equivalence_partitions", "verify_ledger_and_state", "verify_role_manifests", "verify_seal", "verify_version_forward_lineage"}),
    "latency.py": frozenset({"ACTIVE_ATTEMPT", "_fsync_parent_directory", "_publish_temporary", "atomic_json", "run_latency_suite", "validate_latency_result"}),
    "launcher.py": frozenset({"<docstring>", "ACTIVE_ATTEMPT", "PROGRAM_SCRIPT", "_require_controller_state", "_seal_enriched_power_summaries"}),
    "preseal.py": frozenset({"<imports>", "ACTIVE_ATTEMPT", "PRE_DATA_SEAL_PATH", "REQUIRED_IMPLEMENTATION_FILES", "SCIENCE_ATTEMPT", "_files_under", "_state", "_verified_checkpoint", "implementation_complete", "preseal_qualification", "seal_pre_confirmation", "seal_pre_data", "seal_pre_selection", "validate_power_input_coherence", "validate_seed_freshness_evidence", "validate_verifier_contracts"}),
    "runner.py": frozenset({"<imports>", "ATTEMPT_VERSION", "_assert_regular_unlinked", "_enforce_execution_runtime", "_execute_confirmation_locked", "_execute_development_locked", "_execution_record_if_valid", "_execution_source_bindings", "_load_raw_manifest", "_preflight_execution_namespace", "_state_authorization", "_validate_existing_execution_manifest", "execute_confirmation", "execute_development"}),
    "study_common.py": frozenset({"_fsync_parent_directory", "_publish_temporary", "atomic_json", "atomic_npz"}),
    "tests/test_analysis.py": frozenset({"_authorization_fixture", "_latency_binding_and_result", "test_post_open_failure_capture_is_outcome_free_and_never_overwrites"}),
    "tests/test_checkpoint_hardening.py": frozenset({"<imports>", "JSON_EXCLUSIVE_WRITERS", "NPZ_EXCLUSIVE_WRITERS", "test_confirmation_file_link_rejects_symlink_and_inode_alias", "test_confirmation_file_link_rejects_valid_hash_at_wrong_path", "test_confirmation_namespace_rejects_unrelated_file", "test_exclusive_json_concurrent_writers_never_replace", "test_exclusive_json_preserves_preexisting_destination", "test_exclusive_npz_concurrent_writers_never_replace", "test_exclusive_npz_preserves_preexisting_destination", "test_execution_manifest_requires_explicit_unopened_fields", "test_execution_source_raw_manifest_hash_is_mandatory", "test_nonexclusive_json_intentionally_retains_replace_semantics", "test_nonexclusive_npz_intentionally_retains_replace_semantics", "test_raw_manifest_rejects_phase_as_role_alias"}),
    "tests/test_generator_contract.py": frozenset({"test_intent_only_partial_artifact_stops_without_reroll", "test_paths_are_role_and_regime_isolated", "test_replacement_registry_is_global_across_roles_and_dgps"}),
    "tests/test_independent_verify.py": frozenset({"_policy", "test_analysis_execution_invalid_branch_is_preload_and_binds_result_presence", "test_power_provenance_is_local_identity_bound_and_tamper_evident", "test_v008_equivalence_partitions_are_independently_recomputed_and_tamper_evident", "test_v008_equivalence_partitions_reject_omitted_required_symbol", "test_v008_equivalence_partitions_reject_wildcard_procedural_scope", "test_v008_lineage_normalization_and_contract_canonicalization_are_tamper_evident", "test_verifier_imports_only_stdlib_and_numpy_and_failure_stdout_is_one_json_line", "test_zero_confirmation_version_forward_requires_explicit_equivalence"}),
    "tests/test_preseal.py": frozenset({"<docstring>", "test_compile_and_power_are_state_gated_inside_exact_eval_worker", "test_power_input_paths_resolve_from_repository_not_cwd", "test_power_provenance_binds_rule_locks_freeze_and_selected_identity", "test_pre_data_manifest_ignores_later_stage_top_level_products", "test_pre_data_manifest_rejects_unlisted_python_source", "test_preseal_state_uses_verified_controller_not_direct_state_read", "test_python_cache_hygiene_fails_closed", "test_stage_file_collection_rejects_supplied_directory_symlink_before_rglob"}),
    "tests/test_runner_gate.py": frozenset({"test_complete_confirmation_part_is_idempotently_resumed", "test_execution_handlers_do_not_capture_process_interruptions"}),
    "tests/test_seed_power.py": frozenset({"_claim_summary", "test_power_cli_binds_gate_locks_and_persists_only_repo_relative_paths"}),
    "tests/test_terminal_workflow.py": frozenset({"_analysis", "_attempt", "_captured_audit", "_contract", "_state", "test_early_decision_has_zero_later_roles_and_reports_are_claim_bounded", "test_report_artifact_mutation_is_not_repaired", "test_result_present_analysis_integrity_failure_is_manifested_but_not_called"}),
    "tests/test_workflow.py": frozenset({"<docstring>", "FakeController", "_configure_power_paths", "_controller_json", "_install_fit_count_crash", "_ledger_record", "_power_result", "_prior_selection_audit", "_synthetic_role_manifests", "test_count_crash_recovery_rejects_state_ledger_audit_and_manifest_tamper", "test_default_freeze_writes_identity_summaries_only_after_gate_freeze", "test_exact_role_count_is_resume_idempotent", "test_exact_role_count_without_authenticated_crash_record_fails_closed", "test_fit_count_commit_crash_reauthenticates_manifests_and_advances_once", "test_inherited_early_stop_adapter_rejects_unsealed_active_contract", "test_loaded_workflow_controller_adapts_both_inherited_early_stop_contracts", "test_sealed_analysis_integrity_failure_is_staged_not_advanced"}),
    "terminal_workflow.py": frozenset({"<docstring>", "ACTIVE_ATTEMPT", "ATTEMPT", "CONTROLLER", "_analysis_for_decision", "_audit_common", "_manifest_contract", "_robustness_map", "_validate_manifest_state", "build_postterminal_reports", "build_terminal_decision", "build_terminal_input_manifest", "complete_terminal_workflow", "finalize_terminal", "verify_terminal_input_manifest"}),
    "workflow.py": frozenset({"<imports>", "ATTEMPT", "DEFAULT_EVIDENCE_PATHS", "LEDGER_PATH", "ROLE_COUNTER_FIELDS", "ROLE_EXECUTION_SEALS", "ROLE_RAW_SEALS", "ROLE_SEALS", "Workflow", "_common_role_authorization_state_sha256", "_controller_state_object_sha256", "_load_controller", "_require_inherited_presealed_verifier_contract", "_verify_role_count_update_recovery", "_verify_role_regime", "verify_role_manifests"}),
}
# Final, independently transcribed additions from the frozen v008 implementation
# lanes. These remain exact required deltas; observed symbols must equal them.
_PROCEDURAL_REQUIRED_ADDITIONS: dict[str, frozenset[str]] = {
    "analysis.py": frozenset({
        "_verify_replacement_registry_closure", "mark_confirmation_outcomes_opened",
    }),
    "build_manifest.py": frozenset({"_prospective_episode_ids"}),
    "checkpoints.py": frozenset({
        "ORPHAN_ADOPTION_RULE", "PERSISTENCE_INTENT_KEYS",
        "PERSISTENCE_INTENT_STATUS", "_canonical_json_object",
        "_canonical_record_sha256", "_failure_chain_v2",
        "_validate_closed_persistence_intent_contract",
        "_validate_final_replacement_claim_references",
        "_validate_generation_audit_contract",
        "_validate_replacement_registry_chain",
    }),
    "generator.py": frozenset({
        "GENERATION_AUDIT_KEYS", "ORPHAN_ADOPTION_RULE", "PERSISTENCE_INTENT_KEYS",
        "PERSISTENCE_INTENT_SCHEMA_VERSION", "PERSISTENCE_INTENT_STATUS",
        "RAW_MANIFEST_ORPHAN_POLICY", "REPLACEMENT_CLAIMS_ROOT",
        "RESET_METADATA_KEYS", "RNG_ACTIVATION_ORDER",
        "_PERSISTENCE_SOURCE_HASH_CACHE", "_assert_repo_file_chain",
        "_candidate_from_claim", "_canonical_json_file",
        "_canonical_seed_record_for_persistence", "_empty_registry",
        "_exact_positive_int", "_exact_sha256", "_expected_array_metadata",
        "_expected_episode_paths", "_is_sha256", "_load_registry",
        "_prospective_assignment", "_read_canonical_json_object",
        "_record_from_intent", "_regular_file_snapshot",
        "_registry_authorization_policy", "_registry_genesis",
        "_stable_artifact_sha256", "_stable_source_sha256",
        "_validate_current_crosslinks", "_validate_generation_audit",
        "_validate_persistence_intent", "_validate_registry_claim",
        "_validate_registry_genesis", "_validate_retained_failure_partition",
        "_verify_raw_against_intent", "_write_or_verify_intent",
        "_write_or_verify_sidecar", "adopt_orphan", "build_persistence_intent",
        "claim_next_replacement",
        "ensure_replacement_registry", "existing_replacement_claims",
        "generate_episode", "next_candidate", "persist_episode",
        "replacement_registry_prefix", "resume_episode",
    }),
    "independent_verify.py": frozenset({
        "EXPECTED_ALLOWED_SOURCE_SUFFIXES", "EXPECTED_GENERATED_ATTEMPT_PRODUCTS",
        "EXPECTED_GENERATED_AUDIT_PATHS", "EXPECTED_GENERATED_REGIMES",
        "EXPECTED_GENERATED_ROLES", "EXPECTED_MUTABLE_REPOSITORY_PATHS",
        "EXPECTED_PRE_DATA_AUDIT_INPUTS", "EXPECTED_PRE_DATA_STATIC_PATHS",
        "EXPECTED_REPOSITORY_PATHS", "EXPECTED_STUDY_ROOT_DIRECTORY_NAMES",
        "EXPECTED_STUDY_ROOT_OPERATIONAL_NAMES", "REPLAY_DEVELOPMENT_ARRAY_KEYS",
        "REPLAY_EPISODE_KEYS", "REPLAY_FIXED_EPISODES_PER_DGP",
        "REPLAY_RESULT_KEYS", "REPLAY_ROBUST_ARRAY_KEYS",
        "REPLAY_ROLE_AUDIT_KEYS", "REPLAY_ROLE_ORDER",
        "REPLAY_ROLE_REGIME_KEYS", "REPLAY_ROLE_STATES",
        "SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE", "SEED_FIELDS",
        "_PROCEDURAL_REQUIRED_ADDITIONS", "_closure_literal_set",
        "_independent_cache_or_near_miss", "_independent_canonical_json_record",
        "_independent_data_json", "_independent_data_path", "_independent_episode_ids",
        "_independent_development_replay_binding",
        "_independent_failure_contract", "_independent_generated_path",
        "_independent_generated_policy", "_independent_manifest_closure",
        "_independent_manifest_policy", "_independent_registry_contract",
        "_independent_robust_replay_binding",
        "_independent_root_namespace", "_independent_scope_audit",
        "_independent_seed_tuple", "_independent_sha256",
        "_read_canonical_replay_json", "_replay_array_keys",
        "_replay_episodes_per_dgp", "_replay_exact_link", "_replay_roles",
        "_replay_strict_sha256",
        "_verify_durable_controller_adapter_marker",
        "_verify_replay_regime_result",
        "_verify_replacement_registry_and_failures",
        "_verify_version_forward_transaction_receipt", "load_confirmation_regime",
        "load_development_role", "replay_array_sha256",
        "verify_scientific_replay_qualification",
    }),
    "runner.py": frozenset({
        "CONFIRMATION_EQUIVALENCE_CHECK_KEYS", "ScientificReplayContext",
        "_replay_context", "_verify_persisted_gate_semantics",
        "_verify_replayed_part_semantics",
    }),
    "study_common.py": frozenset({"CONTROLLER_PATH"}),
    "tests/test_analysis.py": frozenset({
        "test_post_input_seal_replacement_append_fails_before_npz",
    }),
    "tests/test_generator_contract.py": frozenset({
        "<imports>", "_adopt_fixture", "_canonical_json", "_fixture_seed_base",
        "_independent_policy", "_issue_claim", "_ledger_tuple",
        "_multi_failure_case", "_raw_orphan_fixture", "_registry_fixture",
        "generation_audit",
        "test_checkpoint_failure_chain_v2_rejects_tamper_reorder_and_reuse",
        "test_confirmation_checkpoint_requires_failure_free_terminal_replacement",
        "test_confirmation_consumer_accepts_only_the_closed_v008_intent_contract",
        "test_independent_verifier_authenticates_multi_failure_partition_and_adversaries",
        "test_persistence_error_is_not_a_rollout_error_and_does_not_claim_replacement",
        "test_present_empty_or_malformed_failure_log_stops_before_generation_outputs",
        "test_raw_orphan_intent_adversaries_stop_before_sidecar_or_replacement",
        "test_raw_orphan_link_and_alias_attacks_stop_before_adoption",
        "test_raw_orphan_malformed_or_partial_intent_never_writes_sidecar",
        "test_raw_orphan_recomputes_raw_arrays_and_rejects_post_intent_change",
        "test_raw_orphan_rejects_live_crosslink_change_after_intent",
        "test_raw_orphan_without_intent_stops_instead_of_replacing",
        "test_replacement_raw_orphan_resume_binds_claim_failure_and_ledger",
        "test_replacement_registry_rejects_extra_field_and_reorder",
        "test_replacement_registry_rejects_source_reuse_and_segment_gap",
        "test_retained_partition_accepts_zero_and_multiple_failures_but_rejects_failed_retained",
        "test_safe_raw_orphan_is_adopted_only_through_closed_matching_intent",
        "test_safe_raw_orphan_is_adopted_only_through_matching_intent",
    }),
    "tests/test_independent_verify.py": frozenset({
        "<imports>", "_development_loader_case", "_durable_controller_marker_case",
        "_scientific_replay_qualification_case", "_version_forward_receipt_v2_case",
        "test_development_loader_rejects_coherent_rewrite_before_feature_arithmetic",
        "test_independent_development_loader_accepts_exact_typed_coherent_parts",
        "test_independent_development_loader_rejects_coherent_array_type_or_exit_tamper",
        "test_independent_development_loader_rejects_json_type_aliases",
        "test_independent_durable_controller_marker_binds_schema_sources_and_operation",
        "test_independent_durable_controller_marker_rejects_direct_root_advance",
        "test_independent_durable_controller_marker_rejects_field_tamper",
        "test_independent_durable_controller_marker_rejects_lowercase_operation_replacement",
        "test_independent_durable_controller_marker_rejects_state_only_mutation",
        "test_independent_durable_controller_marker_requires_exact_v2_schema",
        "test_independent_replay_qualification_accepts_exact_cross_role_fixture",
        "test_independent_replay_qualification_rejects_type_or_exactness_alias",
        "test_independent_version_forward_receipt_rejects_closed_schema_and_type_drift",
        "test_independent_version_forward_receipt_rejects_coherent_stored_projection_drift",
        "test_independent_version_forward_receipt_rejects_coherently_rehashed_operation_tamper",
        "test_independent_version_forward_receipt_rejects_current_direct_root_state",
        "test_independent_version_forward_receipt_rejects_every_journal_and_staging_residue",
        "test_independent_version_forward_receipt_rejects_fully_rehashed_direct_root_proposal",
        "test_independent_version_forward_receipt_rejects_ledger_and_state_digest_tamper",
        "test_independent_version_forward_receipt_rejects_live_source_drift",
        "test_independent_version_forward_receipt_rejects_rehashed_live_identity_tamper",
        "test_independent_version_forward_receipt_rejects_replace_or_alias_identity",
        "test_independent_version_forward_receipt_rejects_timestamp_context_and_digest_tamper",
        "test_independent_version_forward_receipt_uses_fresh_dynamic_projections",
        "test_independent_version_forward_receipt_v2_recomputes_closed_transaction",
        "test_version_forward_lineage_invokes_independent_schema_v2_receipt_verifier",
    }),
    "tests/test_preseal.py": frozenset({
        "<imports>", "_preseal_power_tree", "_tamper_preseal_power",
        "test_preseal_independently_recomputes_exact_binding_power",
        "test_preseal_rejects_tampered_power_before_authorization",
    }),
    "tests/test_runner_gate.py": frozenset({
        "<imports>", "_raw_fixture", "_synthetic_replay_context",
        "_zero_confirmation_equivalence",
        "test_development_binding_producer_matches_workflow_consumer_exactly",
        "test_existing_development_manifest_resume_is_closed_and_recomputed",
    }),
    "tests/test_workflow.py": frozenset({
        "<imports>", "_adapter_marker", "_adapter_operation_sha256",
        "_isolate_pre_replay_workflow_tests", "_role_verification",
        "_synthetic_replay_qualification", "_tamper_power_result",
        "test_count_crash_after_audit_publish_rejects_stale_or_ambiguous_audit",
        "test_count_crash_recovery_rejects_bool_int_aliases",
        "test_count_recovery_rejects_state_only_mutation",
        "test_count_recovery_rejects_valid_hex_operation_replacement_even_if_rebound",
        "test_fit_count_crash_after_audit_publish_adopts_exact_audit_and_advances_once",
        "test_power_identity_branch_and_controller_mapping",
        "test_power_recomputation_rejects_stale_or_forged_existing_result",
        "test_real_manifest_verifier_authenticates_four_synthetic_regimes_without_numpy",
        "test_real_manifest_verifier_authenticates_parts_and_rejects_closed_contract_drift",
        "test_recovery_bearing_published_audit_tamper_fails_closed",
        "test_role_verifier_rejects_schema_bool_and_numeric_aliases",
        "test_role_counter_updates_only_after_all_four_exact_manifests",
        "test_second_crash_after_recovery_audit_publish_adopts_exact_audit_once",
        "test_workflow_loader_shape_requires_exact_integer_elements",
    }),
    "tests/test_terminal_workflow.py": frozenset({
        "_stub_terminal_scientific_replay", "_terminal_replay_cell_case",
        "test_replay_failure_cannot_create_terminal_manifest",
        "test_terminal_manifest_is_exhaustive_idempotent_and_contract_exact",
        "test_terminal_manifest_reruns_replay_before_adopting_existing_closure",
        "test_terminal_replay_cell_accepts_exact_typed_positive_fixture",
        "test_terminal_replay_cell_rejects_type_runtime_or_exactness_drift",
    }),
    "terminal_workflow.py": frozenset({
        "REPLAY_DEVELOPMENT_ARRAY_KEYS", "REPLAY_EPISODE_KEYS",
        "REPLAY_FIXED_EPISODES_PER_DGP", "REPLAY_OUTCOME_COUNT_FIELDS",
        "REPLAY_RESULT_KEYS", "REPLAY_ROBUST_ARRAY_KEYS", "REPLAY_ROLE_AUDIT_KEYS",
        "REPLAY_ROLE_ORDER", "REPLAY_ROLE_REGIME_KEYS", "REPLAY_ROLE_STATES",
        "REPLAY_ROWS_PER_EPISODE", "SCIENTIFIC_REPLAY_LAUNCHER_NAME",
        "SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE", "SCIENTIFIC_REPLAY_SOURCE_NAME",
        "_canonical_object_sha256", "_controller_replay_snapshot",
        "_development_replay_manifest_binding",
        "_ensure_terminal_scientific_replay_qualification",
        "_expected_replay_array_keys", "_expected_replay_episodes_per_dgp",
        "_expected_replay_roles", "_persist_or_verify_replay_object",
        "_qualify_terminal_replay_role", "_read_canonical_replay_object",
        "_replay_repo_path", "_robust_replay_manifest_binding",
        "_run_terminal_scientific_replay_cell", "_strict_sha256",
        "_validate_terminal_replay_result", "_verified_replay_link",
    }),
    "workflow.py": frozenset({
        "CONTROLLER_ADAPTER_PATH", "DEVELOPMENT_SIDECAR_KEYS", "DGP_MATRIX_PATH",
        "INPUT_LOADER_AUDIT_KEYS", "PROGRAM_PATH", "REPLACEMENT_REGISTRY_PATH",
        "ROOT_PROGRAM_PATH", "SCIENTIFIC_REPLAY_AUDIT_ROOT",
        "SCIENTIFIC_REPLAY_EPISODE_KEYS", "SCIENTIFIC_REPLAY_LAUNCHER_PATH",
        "SCIENTIFIC_REPLAY_QUALIFICATION_KEYS", "SCIENTIFIC_REPLAY_RESULT_KEYS",
        "SCIENTIFIC_REPLAY_ROLE_AUDIT_KEYS", "SOURCE_BINDING_KEYS",
        "_numpy_causal_features", "_run_scientific_replay_launcher",
        "_scientific_replay_manifest_binding", "_source_ast_sha256",
        "_validate_input_loader_audit", "_validate_role_replay_qualification",
        "_validate_scientific_replay_result",
        "_verify_development_part_arrays", "_verify_recovery_adapter_marker",
        "_verify_source_bindings", "qualify_role_scientific_replay",
    }),
}
PROCEDURAL_EXPECTED_CHANGED_SYMBOLS = {
    relative: frozenset(
        symbols | _PROCEDURAL_REQUIRED_ADDITIONS.get(relative, frozenset())
    )
    for relative, symbols in PROCEDURAL_EXPECTED_CHANGED_SYMBOLS.items()
}
PROCEDURAL_ALLOWED_SYMBOLS = dict(PROCEDURAL_EXPECTED_CHANGED_SYMBOLS)


NEW_LINEAGE_SUPPORT_PATHS = frozenset()
LINEAGE_SUPPORT_PATHS = frozenset(
    {
        'build_manifest.py',
        'fit_inheritance.py',
        'independent_verify.py',
        'inherited_authorization.py',
        'launcher.py',
        'preseal.py',
        'runner.py',
        'scientific_replay.py',
        'tests/test_attempt_parameterization_v002.py',
        'tests/test_independent_verify.py',
        'tests/test_inherited_authorization_v002.py',
        'tests/test_manifest_closure_v002.py',
        'tests/test_scientific_replay.py',
        'tests/test_version_forward_transaction.py',
        'tests/test_version_forward_v002.py',
        'tests/test_workflow.py',
        'verify_version_forward.py',
        'version_forward_prepare.py',
        'version_forward_transaction.py',
        'workflow.py',
    }
)
PROCEDURAL_EXISTING_PATHS = frozenset()
PROCEDURAL_EXPECTED_CHANGED_SYMBOLS: dict[str, frozenset[str]] = {}
PROCEDURAL_ALLOWED_SYMBOLS = dict(PROCEDURAL_EXPECTED_CHANGED_SYMBOLS)
LINEAGE_SUPPORT_UNIT_POLICY: dict[str, dict[str, Any]] = {'build_manifest.py': {'allowed_changed_top_level_units': ['assign:EXPECTED_ACTIVE_ATTEMPT#001'],
                       'deleted_top_level_units': [],
                       'inserted_top_level_units': [],
                       'required_changed_top_level_units': ['assign:EXPECTED_ACTIVE_ATTEMPT#001'],
                       'source_top_level_unit_count': 59,
                       'source_top_level_unit_sequence_sha256': '2f5a7ce7d519ae84691a3fff43ecc13c12fcf086904cbb3636f79db100c1e56f',
                       'target_top_level_unit_count': 59,
                       'target_top_level_unit_sequence_sha256': '2f5a7ce7d519ae84691a3fff43ecc13c12fcf086904cbb3636f79db100c1e56f'},
 'fit_inheritance.py': {'allowed_changed_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                            'assign:TARGET_ATTEMPT#001',
                                                            'docstring#001',
                                                            'function:_source_header#001'],
                        'deleted_top_level_units': [],
                        'inserted_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001'],
                        'required_changed_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                             'assign:TARGET_ATTEMPT#001',
                                                             'docstring#001',
                                                             'function:_source_header#001'],
                        'source_top_level_unit_count': 54,
                        'source_top_level_unit_sequence_sha256': 'bcaa7a9e47964ff4b8e1f2cdb11aad10e353b2a7b63e68e8c0d334f6ed52d40f',
                        'target_top_level_unit_count': 55,
                        'target_top_level_unit_sequence_sha256': '0844fe37cb9a5fcb5670dcc73187f021da8cf1bc60e6862d7aec7595a247b3fd'},
 'independent_verify.py': {'allowed_changed_top_level_units': ['annassign:LINEAGE_SUPPORT_UNIT_POLICY#001',
                                                               'annassign:_PROCEDURAL_REQUIRED_ADDITIONS#001',
                                                               'assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                               'assign:ACTIVE_ATTEMPT#001',
                                                               'assign:ALLOWED_ATTEMPTS#001',
                                                               'assign:ATTEMPT_ROOTS#001',
                                                               'assign:EXPECTED_SELECTION_SOURCE_SEAL_SHA256#001',
                                                               'assign:EXPECTED_SOURCE_ACTIVATION_LEDGER_SHA256#001',
                                                               'assign:EXPECTED_SOURCE_ACTIVATION_STATE_SHA256#001',
                                                               'assign:EXPECTED_SOURCE_INVALIDITY_DRAFT_SHA256#001',
                                                               'assign:EXPECTED_SOURCE_INVALIDITY_SHA256#001',
                                                               'assign:EXPECTED_SOURCE_LEDGER_SHA256#001',
                                                               'assign:EXPECTED_SOURCE_RECEIPT_SHA256#001',
                                                               'assign:EXPECTED_SOURCE_SEAL_SHA256#001',
                                                               'assign:EXPECTED_SOURCE_STATE_SHA256#001',
                                                               'assign:EXPECTED_TRANSITIVE_SEALED_FILE_COUNT#001',
                                                               'assign:INVALIDITY_DRAFT_RELATIVE#001',
                                                               'assign:INVALIDITY_RELATIVE#001',
                                                               'assign:PROCEDURAL_ALLOWED_SYMBOLS#001',
                                                               'assign:PROCEDURAL_EXPECTED_CHANGED_SYMBOLS#001',
                                                               'assign:SOURCE_OPERATIONAL_LOCKS#001',
                                                               'assign:SOURCE_PRE_DATA_RELATIVE#001',
                                                               'assign:SOURCE_RECEIPT_RELATIVE#001',
                                                               'assign:SOURCE_SELECTION_BOUNDARY_PRODUCTS#001',
                                                               'class:_AdministrativeAstNormalizer#001',
                                                               'function:_canonicalize_contract#001',
                                                               'function:_independent_manifest_closure#001',
                                                               'function:_is_active_attempt_guard#001',
                                                               'function:_source_manifest_relative_paths#001',
                                                               'function:_validate_inheritance_seal_scalar_types#001',
                                                               'function:_verify_descendant_adapter_transaction_chain#001',
                                                               'function:_verify_durable_controller_adapter_marker#001',
                                                               'function:_verify_replay_regime_result#001',
                                                               'function:_verify_version_forward_lineage_legacy#001',
                                                               'function:_verify_version_forward_transaction_receipt#001',
                                                               'function:_verify_version_forward_transaction_receipt#002',
                                                               'function:load_contract#001',
                                                               'function:verify_equivalence_partitions#001',
                                                               'function:verify_ledger_and_state#001',
                                                               'function:verify_scientific_replay_qualification#001',
                                                               'function:verify_version_forward_lineage#001'],
                           'deleted_top_level_units': [],
                           'inserted_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                        'assign:EXPECTED_SELECTION_SOURCE_SEAL_SHA256#001'],
                           'required_changed_top_level_units': ['annassign:LINEAGE_SUPPORT_UNIT_POLICY#001',
                                                                'annassign:_PROCEDURAL_REQUIRED_ADDITIONS#001',
                                                                'assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                                'assign:ACTIVE_ATTEMPT#001',
                                                                'assign:ALLOWED_ATTEMPTS#001',
                                                                'assign:ATTEMPT_ROOTS#001',
                                                                'assign:EXPECTED_SELECTION_SOURCE_SEAL_SHA256#001',
                                                                'assign:EXPECTED_SOURCE_ACTIVATION_LEDGER_SHA256#001',
                                                                'assign:EXPECTED_SOURCE_ACTIVATION_STATE_SHA256#001',
                                                                'assign:EXPECTED_SOURCE_INVALIDITY_DRAFT_SHA256#001',
                                                                'assign:EXPECTED_SOURCE_INVALIDITY_SHA256#001',
                                                                'assign:EXPECTED_SOURCE_LEDGER_SHA256#001',
                                                                'assign:EXPECTED_SOURCE_RECEIPT_SHA256#001',
                                                                'assign:EXPECTED_SOURCE_SEAL_SHA256#001',
                                                                'assign:EXPECTED_SOURCE_STATE_SHA256#001',
                                                                'assign:EXPECTED_TRANSITIVE_SEALED_FILE_COUNT#001',
                                                                'assign:INVALIDITY_DRAFT_RELATIVE#001',
                                                                'assign:INVALIDITY_RELATIVE#001',
                                                                'assign:PROCEDURAL_ALLOWED_SYMBOLS#001',
                                                                'assign:PROCEDURAL_EXPECTED_CHANGED_SYMBOLS#001',
                                                                'assign:SOURCE_OPERATIONAL_LOCKS#001',
                                                                'assign:SOURCE_PRE_DATA_RELATIVE#001',
                                                                'assign:SOURCE_RECEIPT_RELATIVE#001',
                                                                'assign:SOURCE_SELECTION_BOUNDARY_PRODUCTS#001',
                                                                'class:_AdministrativeAstNormalizer#001',
                                                                'function:_canonicalize_contract#001',
                                                                'function:_independent_manifest_closure#001',
                                                                'function:_is_active_attempt_guard#001',
                                                                'function:_source_manifest_relative_paths#001',
                                                                'function:_validate_inheritance_seal_scalar_types#001',
                                                                'function:_verify_descendant_adapter_transaction_chain#001',
                                                                'function:_verify_durable_controller_adapter_marker#001',
                                                                'function:_verify_replay_regime_result#001',
                                                                'function:_verify_version_forward_lineage_legacy#001',
                                                                'function:_verify_version_forward_transaction_receipt#001',
                                                                'function:_verify_version_forward_transaction_receipt#002',
                                                                'function:load_contract#001',
                                                                'function:verify_equivalence_partitions#001',
                                                                'function:verify_ledger_and_state#001',
                                                                'function:verify_scientific_replay_qualification#001',
                                                                'function:verify_version_forward_lineage#001'],
                           'source_top_level_unit_count': 286,
                           'source_top_level_unit_sequence_sha256': 'daedc1e5e5accf91cfb0a386a8cde1479548083a718575d54df77d1e40c78b88',
                           'target_top_level_unit_count': 288,
                           'target_top_level_unit_sequence_sha256': 'a9115c158b7e7154f6a3056958b56f0c9aad954ad06b5f51b4a0b4ee8cc5bc9e'},
 'inherited_authorization.py': {'allowed_changed_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                                    'assign:ACTIVE_ATTEMPT#001',
                                                                    'assign:INVALIDITY_DRAFT_PATH#001',
                                                                    'assign:INVALIDITY_PATH#001',
                                                                    'assign:PENDING_STAGING_PATH#001',
                                                                    'assign:RECEIPT_JOURNAL_STAGING_PATH#001',
                                                                    'assign:RECEIPT_STAGING_PATH#001',
                                                                    'assign:SELECTION_CONTROLLER_ADAPTER_PATH#001',
                                                                    'assign:SELECTION_PRE_DATA_SEAL_PATH#001',
                                                                    'assign:SOURCE_ADAPTER_STATE_KEY#001',
                                                                    'assign:SOURCE_CONTROLLER_ADAPTER_PATH#001',
                                                                    'assign:SOURCE_PRE_DATA_SEAL_PATH#001',
                                                                    'assign:SOURCE_TRANSACTION_RECEIPT_PATH#001',
                                                                    'assign:STATE_STAGING_PATH#001',
                                                                    'class:InheritedAuthorizationError#001',
                                                                    'docstring#001',
                                                                    'function:_active_seal_verifier_projection#001',
                                                                    'function:_read_verified_controller_via_adapter#001',
                                                                    'function:_validate_inheritance_seal_scalar_types#001',
                                                                    'function:_verify_adapter_count_event_bindings#001',
                                                                    'function:_verify_controller_adapter_marker#001',
                                                                    'function:_verify_controller_adapter_marker_value#001',
                                                                    'function:_verify_inherited_pre_data_authorization_locked#001',
                                                                    'function:_verify_inherited_pre_data_authorization_locked#002',
                                                                    'function:_verify_role_count_recovery_authorization#001',
                                                                    'function:_verify_sealed_files#001',
                                                                    'function:_verify_source_controller_adapter_marker_value#001',
                                                                    'function:_verify_version_forward_transaction_receipt#001',
                                                                    'function:verify_inherited_pre_data_authorization#001'],
                                'deleted_top_level_units': [],
                                'inserted_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                             'assign:SELECTION_CONTROLLER_ADAPTER_PATH#001',
                                                             'assign:SELECTION_PRE_DATA_SEAL_PATH#001'],
                                'required_changed_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                                     'assign:ACTIVE_ATTEMPT#001',
                                                                     'assign:INVALIDITY_DRAFT_PATH#001',
                                                                     'assign:INVALIDITY_PATH#001',
                                                                     'assign:PENDING_STAGING_PATH#001',
                                                                     'assign:RECEIPT_JOURNAL_STAGING_PATH#001',
                                                                     'assign:RECEIPT_STAGING_PATH#001',
                                                                     'assign:SELECTION_CONTROLLER_ADAPTER_PATH#001',
                                                                     'assign:SELECTION_PRE_DATA_SEAL_PATH#001',
                                                                     'assign:SOURCE_ADAPTER_STATE_KEY#001',
                                                                     'assign:SOURCE_CONTROLLER_ADAPTER_PATH#001',
                                                                     'assign:SOURCE_PRE_DATA_SEAL_PATH#001',
                                                                     'assign:SOURCE_TRANSACTION_RECEIPT_PATH#001',
                                                                     'assign:STATE_STAGING_PATH#001',
                                                                     'class:InheritedAuthorizationError#001',
                                                                     'docstring#001',
                                                                     'function:_active_seal_verifier_projection#001',
                                                                     'function:_read_verified_controller_via_adapter#001',
                                                                     'function:_validate_inheritance_seal_scalar_types#001',
                                                                     'function:_verify_adapter_count_event_bindings#001',
                                                                     'function:_verify_controller_adapter_marker#001',
                                                                     'function:_verify_controller_adapter_marker_value#001',
                                                                     'function:_verify_inherited_pre_data_authorization_locked#001',
                                                                     'function:_verify_inherited_pre_data_authorization_locked#002',
                                                                     'function:_verify_role_count_recovery_authorization#001',
                                                                     'function:_verify_sealed_files#001',
                                                                     'function:_verify_source_controller_adapter_marker_value#001',
                                                                     'function:_verify_version_forward_transaction_receipt#001',
                                                                     'function:verify_inherited_pre_data_authorization#001'],
                                'source_top_level_unit_count': 103,
                                'source_top_level_unit_sequence_sha256': '1f1f649c84d21361d78512509d7c2c6fb2b527c3b447b228dff7d9c4af7b194b',
                                'target_top_level_unit_count': 106,
                                'target_top_level_unit_sequence_sha256': '5904d59842c5d87f1f306921015696151878d437bda66002e3ca679bf09b9003'},
 'launcher.py': {'allowed_changed_top_level_units': ['docstring#001', 'if#001'],
                 'deleted_top_level_units': [],
                 'inserted_top_level_units': [],
                 'required_changed_top_level_units': ['docstring#001', 'if#001'],
                 'source_top_level_unit_count': 43,
                 'source_top_level_unit_sequence_sha256': '037b786c53c288bade17e74ed9151fb29be851b2c6843328720683bd2abc0e03',
                 'target_top_level_unit_count': 43,
                 'target_top_level_unit_sequence_sha256': '037b786c53c288bade17e74ed9151fb29be851b2c6843328720683bd2abc0e03'},
 'preseal.py': {'allowed_changed_top_level_units': ['if#001'],
                'deleted_top_level_units': [],
                'inserted_top_level_units': [],
                'required_changed_top_level_units': ['if#001'],
                'source_top_level_unit_count': 93,
                'source_top_level_unit_sequence_sha256': '0cdc99cd4a4c0d1bda8e46b729e55fa48ece3b6f4f7d2620a8fe90bb53dc3c5f',
                'target_top_level_unit_count': 93,
                'target_top_level_unit_sequence_sha256': '0cdc99cd4a4c0d1bda8e46b729e55fa48ece3b6f4f7d2620a8fe90bb53dc3c5f'},
 'scientific_replay.py': {'allowed_changed_top_level_units': ['assign:EXPECTED_SOURCE_HASHES#001',
                                                              'docstring#001',
                                                              'function:qualify_regime#001',
                                                              'if#001'],
                          'deleted_top_level_units': [],
                          'inserted_top_level_units': [],
                          'required_changed_top_level_units': ['assign:EXPECTED_SOURCE_HASHES#001',
                                                               'docstring#001',
                                                               'function:qualify_regime#001',
                                                               'if#001'],
                          'source_top_level_unit_count': 101,
                          'source_top_level_unit_sequence_sha256': 'e51e5c76e60e60f2c961654ae8fa3065fb528e23af4a300af636848a5fda4b40',
                          'target_top_level_unit_count': 101,
                          'target_top_level_unit_sequence_sha256': 'e51e5c76e60e60f2c961654ae8fa3065fb528e23af4a300af636848a5fda4b40'},
 'tests/test_attempt_parameterization_v002.py': {'allowed_changed_top_level_units': ['docstring#001',
                                                                                     'function:test_every_active_module_derives_and_guards_v007#001',
                                                                                     'function:test_every_active_module_derives_and_guards_v008#001'],
                                                 'deleted_top_level_units': ['function:test_every_active_module_derives_and_guards_v007#001'],
                                                 'inserted_top_level_units': ['function:test_every_active_module_derives_and_guards_v008#001'],
                                                 'required_changed_top_level_units': ['docstring#001',
                                                                                      'function:test_every_active_module_derives_and_guards_v007#001',
                                                                                      'function:test_every_active_module_derives_and_guards_v008#001'],
                                                 'source_top_level_unit_count': 27,
                                                 'source_top_level_unit_sequence_sha256': 'a18385e74efdd3402177b6dcc9894a746816b85114dd5e7bc03b4bc8c4ae5745',
                                                 'target_top_level_unit_count': 27,
                                                 'target_top_level_unit_sequence_sha256': 'ea603a52c2f2058b193e57d618e5bb47a87648defc93c5a9fc3651ab893ea757'},
 'tests/test_independent_verify.py': {'allowed_changed_top_level_units': ['function:_append_analysis_integrity_staging#001',
                                                                          'function:_append_descendant_count#001',
                                                                          'function:_append_descendant_group#001',
                                                                          'function:_append_early_terminal_group#001',
                                                                          'function:_append_execution_invalid_terminal_group#001',
                                                                          'function:_append_no_candidate_staging#001',
                                                                          'function:_append_ordinary_descendant_checkpoint#001',
                                                                          'function:_append_single_terminal_decision#001',
                                                                          'function:_development_loader_case#001',
                                                                          'function:_durable_controller_marker_case#001',
                                                                          'function:_policy#001',
                                                                          'function:_scientific_replay_qualification_case#001',
                                                                          'function:_version_forward_receipt_v2_case#001',
                                                                          'function:_version_forward_receipt_v2_case_legacy#001',
                                                                          'function:test_analysis_execution_invalid_branch_is_preload_and_binds_result_presence#001',
                                                                          'function:test_independent_descendant_early_failure_rejects_ineligible_provenance#001',
                                                                          'function:test_independent_durable_controller_marker_binds_schema_sources_and_operation#001',
                                                                          'function:test_independent_durable_controller_marker_rejects_field_tamper#001',
                                                                          'function:test_independent_durable_controller_marker_rejects_lowercase_operation_replacement#001',
                                                                          'function:test_independent_durable_controller_marker_requires_exact_v2_schema#001',
                                                                          'function:test_independent_version_forward_receipt_rejects_coherently_rehashed_operation_tamper#001',
                                                                          'function:test_independent_version_forward_receipt_rejects_current_direct_root_state#001',
                                                                          'function:test_independent_version_forward_receipt_rejects_every_journal_and_staging_residue#001',
                                                                          'function:test_independent_version_forward_receipt_replays_descendant_transaction#001',
                                                                          'function:test_power_provenance_is_local_identity_bound_and_tamper_evident#001',
                                                                          'function:test_v007_equivalence_partitions_are_independently_recomputed_and_tamper_evident#001',
                                                                          'function:test_v007_equivalence_partitions_reject_lineage_provenance_drift#001',
                                                                          'function:test_v007_equivalence_partitions_reject_omitted_lineage_unit#001',
                                                                          'function:test_v007_lineage_normalization_and_contract_canonicalization_are_tamper_evident#001',
                                                                          'function:test_v008_equivalence_partitions_are_independently_recomputed_and_tamper_evident#001',
                                                                          'function:test_v008_equivalence_partitions_reject_lineage_provenance_drift#001',
                                                                          'function:test_v008_equivalence_partitions_reject_omitted_lineage_unit#001',
                                                                          'function:test_v008_lineage_normalization_and_contract_canonicalization_are_tamper_evident#001'],
                                      'deleted_top_level_units': ['function:test_v007_equivalence_partitions_are_independently_recomputed_and_tamper_evident#001',
                                                                  'function:test_v007_equivalence_partitions_reject_lineage_provenance_drift#001',
                                                                  'function:test_v007_equivalence_partitions_reject_omitted_lineage_unit#001',
                                                                  'function:test_v007_lineage_normalization_and_contract_canonicalization_are_tamper_evident#001'],
                                      'inserted_top_level_units': ['function:test_v008_equivalence_partitions_are_independently_recomputed_and_tamper_evident#001',
                                                                   'function:test_v008_equivalence_partitions_reject_lineage_provenance_drift#001',
                                                                   'function:test_v008_equivalence_partitions_reject_omitted_lineage_unit#001',
                                                                   'function:test_v008_lineage_normalization_and_contract_canonicalization_are_tamper_evident#001'],
                                      'required_changed_top_level_units': ['function:_append_analysis_integrity_staging#001',
                                                                           'function:_append_descendant_count#001',
                                                                           'function:_append_descendant_group#001',
                                                                           'function:_append_early_terminal_group#001',
                                                                           'function:_append_execution_invalid_terminal_group#001',
                                                                           'function:_append_no_candidate_staging#001',
                                                                           'function:_append_ordinary_descendant_checkpoint#001',
                                                                           'function:_append_single_terminal_decision#001',
                                                                           'function:_development_loader_case#001',
                                                                           'function:_durable_controller_marker_case#001',
                                                                           'function:_policy#001',
                                                                           'function:_scientific_replay_qualification_case#001',
                                                                           'function:_version_forward_receipt_v2_case#001',
                                                                           'function:_version_forward_receipt_v2_case_legacy#001',
                                                                           'function:test_analysis_execution_invalid_branch_is_preload_and_binds_result_presence#001',
                                                                           'function:test_independent_descendant_early_failure_rejects_ineligible_provenance#001',
                                                                           'function:test_independent_durable_controller_marker_binds_schema_sources_and_operation#001',
                                                                           'function:test_independent_durable_controller_marker_rejects_field_tamper#001',
                                                                           'function:test_independent_durable_controller_marker_rejects_lowercase_operation_replacement#001',
                                                                           'function:test_independent_durable_controller_marker_requires_exact_v2_schema#001',
                                                                           'function:test_independent_version_forward_receipt_rejects_coherently_rehashed_operation_tamper#001',
                                                                           'function:test_independent_version_forward_receipt_rejects_current_direct_root_state#001',
                                                                           'function:test_independent_version_forward_receipt_rejects_every_journal_and_staging_residue#001',
                                                                           'function:test_independent_version_forward_receipt_replays_descendant_transaction#001',
                                                                           'function:test_power_provenance_is_local_identity_bound_and_tamper_evident#001',
                                                                           'function:test_v007_equivalence_partitions_are_independently_recomputed_and_tamper_evident#001',
                                                                           'function:test_v007_equivalence_partitions_reject_lineage_provenance_drift#001',
                                                                           'function:test_v007_equivalence_partitions_reject_omitted_lineage_unit#001',
                                                                           'function:test_v007_lineage_normalization_and_contract_canonicalization_are_tamper_evident#001',
                                                                           'function:test_v008_equivalence_partitions_are_independently_recomputed_and_tamper_evident#001',
                                                                           'function:test_v008_equivalence_partitions_reject_lineage_provenance_drift#001',
                                                                           'function:test_v008_equivalence_partitions_reject_omitted_lineage_unit#001',
                                                                           'function:test_v008_lineage_normalization_and_contract_canonicalization_are_tamper_evident#001'],
                                      'source_top_level_unit_count': 96,
                                      'source_top_level_unit_sequence_sha256': '852c570120f3db9e4e14e5f9b6aa9a24e9d58c743b5dff8cf18fac21493cc8da',
                                      'target_top_level_unit_count': 96,
                                      'target_top_level_unit_sequence_sha256': '7ee1838d1c553aa8676c8a5b523be0797a62f8d297bf0293c61adfee7c3a64e8'},
 'tests/test_inherited_authorization_v002.py': {'allowed_changed_top_level_units': ['docstring#001',
                                                                                    'function:_bind_adapter_marker#001',
                                                                                    'function:_install_inherited_selection_count_crash#001',
                                                                                    'function:_install_strict_raw_manifest_fixture#001',
                                                                                    'function:_lineage_fixture#001',
                                                                                    'function:_record_two_failures#001',
                                                                                    'function:_selection_authorization_fixture#001',
                                                                                    'function:test_active_and_science_attempts_are_split_in_owned_runtime_modules#001',
                                                                                    'function:test_authorized_early_guard_does_not_reacquire_root_exclusive_lock#001',
                                                                                    'function:test_exact_inherited_lineage_authorizes_fit_without_replaying_prior_states#001',
                                                                                    'function:test_independent_post_forward_early_state_requires_exact_guard_and_prefix#001',
                                                                                    'function:test_independent_role_count_recovery_rejects_state_and_operation_rebinding#001',
                                                                                    'function:test_independent_verifier_rejects_selection_seal_crosslink_tamper#001',
                                                                                    'function:test_independent_verifier_result_requires_science_identity_and_read_only_proof#001',
                                                                                    'function:test_inherited_lineage_tampering_fails_closed#001',
                                                                                    'function:test_inherited_marker_v2_rejects_current_state_and_operation_tamper#001',
                                                                                    'function:test_ordinary_zero_count_selection_rejects_preexisting_role_audit#001',
                                                                                    'function:test_post_forward_recomputation_rejects_extra_result_key#001',
                                                                                    'function:test_post_forward_recomputation_rejects_float_count_aliases#001',
                                                                                    'function:test_private_role_count_guard_authorizes_exact_post_count_snapshot#001',
                                                                                    'function:test_private_role_count_guard_rejects_ambiguous_or_later_state#001',
                                                                                    'function:test_selection_workflow_verifies_inherited_raw_and_direct_execution_seals#001',
                                                                                    'function:test_symlinked_output_parent_fails_before_any_output_write#001',
                                                                                    'function:test_transaction_receipt_rejects_controller_or_journal_residue#001',
                                                                                    'function:test_transaction_receipt_rejects_receipt_staging_residue#001',
                                                                                    'function:test_workflow_rejects_selection_seal_without_inheritance_crosslink#001'],
                                                'deleted_top_level_units': [],
                                                'inserted_top_level_units': [],
                                                'required_changed_top_level_units': ['docstring#001',
                                                                                     'function:_bind_adapter_marker#001',
                                                                                     'function:_install_inherited_selection_count_crash#001',
                                                                                     'function:_install_strict_raw_manifest_fixture#001',
                                                                                     'function:_lineage_fixture#001',
                                                                                     'function:_record_two_failures#001',
                                                                                     'function:_selection_authorization_fixture#001',
                                                                                     'function:test_active_and_science_attempts_are_split_in_owned_runtime_modules#001',
                                                                                     'function:test_authorized_early_guard_does_not_reacquire_root_exclusive_lock#001',
                                                                                     'function:test_exact_inherited_lineage_authorizes_fit_without_replaying_prior_states#001',
                                                                                     'function:test_independent_post_forward_early_state_requires_exact_guard_and_prefix#001',
                                                                                     'function:test_independent_role_count_recovery_rejects_state_and_operation_rebinding#001',
                                                                                     'function:test_independent_verifier_rejects_selection_seal_crosslink_tamper#001',
                                                                                     'function:test_independent_verifier_result_requires_science_identity_and_read_only_proof#001',
                                                                                     'function:test_inherited_lineage_tampering_fails_closed#001',
                                                                                     'function:test_inherited_marker_v2_rejects_current_state_and_operation_tamper#001',
                                                                                     'function:test_ordinary_zero_count_selection_rejects_preexisting_role_audit#001',
                                                                                     'function:test_post_forward_recomputation_rejects_extra_result_key#001',
                                                                                     'function:test_post_forward_recomputation_rejects_float_count_aliases#001',
                                                                                     'function:test_private_role_count_guard_authorizes_exact_post_count_snapshot#001',
                                                                                     'function:test_private_role_count_guard_rejects_ambiguous_or_later_state#001',
                                                                                     'function:test_selection_workflow_verifies_inherited_raw_and_direct_execution_seals#001',
                                                                                     'function:test_symlinked_output_parent_fails_before_any_output_write#001',
                                                                                     'function:test_transaction_receipt_rejects_controller_or_journal_residue#001',
                                                                                     'function:test_transaction_receipt_rejects_receipt_staging_residue#001',
                                                                                     'function:test_workflow_rejects_selection_seal_without_inheritance_crosslink#001'],
                                                'source_top_level_unit_count': 82,
                                                'source_top_level_unit_sequence_sha256': '1cc9628ed0fd41d3f8402cbdf62dbae6f08af1d6bfbe72434093a926d9806d81',
                                                'target_top_level_unit_count': 82,
                                                'target_top_level_unit_sequence_sha256': '1cc9628ed0fd41d3f8402cbdf62dbae6f08af1d6bfbe72434093a926d9806d81'},
 'tests/test_manifest_closure_v002.py': {'allowed_changed_top_level_units': ['function:_configure#001',
                                                                             'function:test_manifest_binds_exact_closure_policy_and_science_lineage#001',
                                                                             'function:test_v007_identity_and_all_anticipated_sources_are_required#001',
                                                                             'function:test_v008_identity_and_all_anticipated_sources_are_required#001'],
                                         'deleted_top_level_units': ['function:test_v007_identity_and_all_anticipated_sources_are_required#001'],
                                         'inserted_top_level_units': ['function:test_v008_identity_and_all_anticipated_sources_are_required#001'],
                                         'required_changed_top_level_units': ['function:_configure#001',
                                                                              'function:test_manifest_binds_exact_closure_policy_and_science_lineage#001',
                                                                              'function:test_v007_identity_and_all_anticipated_sources_are_required#001',
                                                                              'function:test_v008_identity_and_all_anticipated_sources_are_required#001'],
                                         'source_top_level_unit_count': 29,
                                         'source_top_level_unit_sequence_sha256': 'ae7673468e83700dc0414430a6396ca117b3e8bfbe50dd9994341fa5edf95bd7',
                                         'target_top_level_unit_count': 29,
                                         'target_top_level_unit_sequence_sha256': 'bca0636e757e6e4f580efdbc1e8b7cde62cb28b8f255151ae391903aa4161c4a'},
 'tests/test_scientific_replay.py': {'allowed_changed_top_level_units': ['function:_make_fixture#001',
                                                                         'function:_synthetic_replay_result#001',
                                                                         'function:test_role_audit_hash_is_bound_and_tamper_fails_closed#001'],
                                     'deleted_top_level_units': [],
                                     'inserted_top_level_units': [],
                                     'required_changed_top_level_units': ['function:_make_fixture#001',
                                                                          'function:_synthetic_replay_result#001',
                                                                          'function:test_role_audit_hash_is_bound_and_tamper_fails_closed#001'],
                                     'source_top_level_unit_count': 37,
                                     'source_top_level_unit_sequence_sha256': '2a2db5adf1714bc8d9186ba29ec6edec10ebc06f9bd1bbfe6d69deee8a91f7b7',
                                     'target_top_level_unit_count': 37,
                                     'target_top_level_unit_sequence_sha256': '2a2db5adf1714bc8d9186ba29ec6edec10ebc06f9bd1bbfe6d69deee8a91f7b7'},
 'tests/test_version_forward_transaction.py': {'allowed_changed_top_level_units': ['function:_commit_controller#001',
                                                                                   'function:_commit_fit_count_descendant#001',
                                                                                   'function:_commit_ordinary_checkpoint#001',
                                                                                   'function:_configure#001',
                                                                                   'function:_configure_real_production_copy#001',
                                                                                   'function:_durable_target_state#001',
                                                                                   'function:_ensure_v007_base#001',
                                                                                   'function:_ensure_v008_base#001',
                                                                                   'function:_synthetic_early_terminal_transition#001',
                                                                                   'function:_synthetic_no_candidate_transition#001',
                                                                                   'function:test_adapter_receipt_rejects_current_direct_root_state#001',
                                                                                   'function:test_authenticated_root_method_is_narrowly_patched_into_durable_adapter#001',
                                                                                   'function:test_coherently_enveloped_non_controller_transition_is_rejected#001',
                                                                                   'function:test_coherently_rehashed_direct_root_append_after_descendant_is_rejected#001',
                                                                                   'function:test_controller_cannot_prepopulate_adapter_owned_prior_marker#001',
                                                                                   'function:test_count_transition_rejects_backward_or_wrong_state_updates#001',
                                                                                   'function:test_descendant_multi_event_transaction_group_replays_exactly#001',
                                                                                   'function:test_direct_root_style_mutation_cannot_authorize_downstream_status#001',
                                                                                   'function:test_durable_adapter_recovers_every_commit_boundary_once#001',
                                                                                   'function:test_durable_adapter_recovers_every_reconciliation_boundary_once#001',
                                                                                   'function:test_every_v007_controller_caller_routes_through_receipt_bound_adapter#001',
                                                                                   'function:test_every_v008_controller_caller_routes_through_receipt_bound_adapter#001',
                                                                                   'function:test_pending_count_event_rederives_prior_marker_and_rejects_tamper#001',
                                                                                   'function:test_pending_state_rejects_bool_int_outcome_aliases#001',
                                                                                   'function:test_postcommit_pre_receipt_crash_resumes_exact_receipt#001',
                                                                                   'function:test_real_production_verifiers_recover_receipt_publication#001',
                                                                                   'function:test_real_root_descendant_checkpoint_replays_in_all_three_verifiers#001',
                                                                                   'function:test_receipt_journal_recovers_every_publication_boundary_bit_exact#001',
                                                                                   'function:test_schema_complete_checkpoint_cannot_publish_unrelated_state_drift#001',
                                                                                   'function:test_two_crashed_count_commits_bind_and_restore_exact_prior_markers#001',
                                                                                   'function:test_verified_receipt_remains_historically_checkable_after_later_state#001'],
                                               'deleted_top_level_units': ['function:_ensure_v007_base#001',
                                                                           'function:test_every_v007_controller_caller_routes_through_receipt_bound_adapter#001'],
                                               'inserted_top_level_units': ['function:_ensure_v008_base#001',
                                                                            'function:test_every_v008_controller_caller_routes_through_receipt_bound_adapter#001'],
                                               'required_changed_top_level_units': ['function:_commit_controller#001',
                                                                                    'function:_commit_fit_count_descendant#001',
                                                                                    'function:_commit_ordinary_checkpoint#001',
                                                                                    'function:_configure#001',
                                                                                    'function:_configure_real_production_copy#001',
                                                                                    'function:_durable_target_state#001',
                                                                                    'function:_ensure_v007_base#001',
                                                                                    'function:_ensure_v008_base#001',
                                                                                    'function:_synthetic_early_terminal_transition#001',
                                                                                    'function:_synthetic_no_candidate_transition#001',
                                                                                    'function:test_adapter_receipt_rejects_current_direct_root_state#001',
                                                                                    'function:test_authenticated_root_method_is_narrowly_patched_into_durable_adapter#001',
                                                                                    'function:test_coherently_enveloped_non_controller_transition_is_rejected#001',
                                                                                    'function:test_coherently_rehashed_direct_root_append_after_descendant_is_rejected#001',
                                                                                    'function:test_controller_cannot_prepopulate_adapter_owned_prior_marker#001',
                                                                                    'function:test_count_transition_rejects_backward_or_wrong_state_updates#001',
                                                                                    'function:test_descendant_multi_event_transaction_group_replays_exactly#001',
                                                                                    'function:test_direct_root_style_mutation_cannot_authorize_downstream_status#001',
                                                                                    'function:test_durable_adapter_recovers_every_commit_boundary_once#001',
                                                                                    'function:test_durable_adapter_recovers_every_reconciliation_boundary_once#001',
                                                                                    'function:test_every_v007_controller_caller_routes_through_receipt_bound_adapter#001',
                                                                                    'function:test_every_v008_controller_caller_routes_through_receipt_bound_adapter#001',
                                                                                    'function:test_pending_count_event_rederives_prior_marker_and_rejects_tamper#001',
                                                                                    'function:test_pending_state_rejects_bool_int_outcome_aliases#001',
                                                                                    'function:test_postcommit_pre_receipt_crash_resumes_exact_receipt#001',
                                                                                    'function:test_real_production_verifiers_recover_receipt_publication#001',
                                                                                    'function:test_real_root_descendant_checkpoint_replays_in_all_three_verifiers#001',
                                                                                    'function:test_receipt_journal_recovers_every_publication_boundary_bit_exact#001',
                                                                                    'function:test_schema_complete_checkpoint_cannot_publish_unrelated_state_drift#001',
                                                                                    'function:test_two_crashed_count_commits_bind_and_restore_exact_prior_markers#001',
                                                                                    'function:test_verified_receipt_remains_historically_checkable_after_later_state#001'],
                                               'source_top_level_unit_count': 101,
                                               'source_top_level_unit_sequence_sha256': '5fdcc0edbd5b148705883abdc16175fa79fa2384054202fbc7763f70e520787e',
                                               'target_top_level_unit_count': 101,
                                               'target_top_level_unit_sequence_sha256': 'c46870eeac35c1a91ee35aba28b7469ad2fb7d8d6dddf2465d4c6174cb34f3f0'},
 'tests/test_version_forward_v002.py': {'allowed_changed_top_level_units': ['function:_closure_calls#001',
                                                                            'function:_copy_live_study#001',
                                                                            'function:test_coherently_rehashed_lineage_record_cannot_self_authorize_unit_delta#001',
                                                                            'function:test_exclusive_writer_never_replaces_temp_artifact#001',
                                                                            'function:test_existing_seal_recomputation_requires_exact_canonical_bytes#001',
                                                                            'function:test_independently_transcribed_ast_normalizers_agree_in_temp_root#001',
                                                                            'function:test_post_forward_selection_raw_progression_is_exact_and_tamper_evident#001',
                                                                            'function:test_verifier_rejects_nonexact_seal_path_in_temp_root#001'],
                                        'deleted_top_level_units': [],
                                        'inserted_top_level_units': [],
                                        'required_changed_top_level_units': ['function:_closure_calls#001',
                                                                             'function:_copy_live_study#001',
                                                                             'function:test_coherently_rehashed_lineage_record_cannot_self_authorize_unit_delta#001',
                                                                             'function:test_exclusive_writer_never_replaces_temp_artifact#001',
                                                                             'function:test_existing_seal_recomputation_requires_exact_canonical_bytes#001',
                                                                             'function:test_independently_transcribed_ast_normalizers_agree_in_temp_root#001',
                                                                             'function:test_post_forward_selection_raw_progression_is_exact_and_tamper_evident#001',
                                                                             'function:test_verifier_rejects_nonexact_seal_path_in_temp_root#001'],
                                        'source_top_level_unit_count': 52,
                                        'source_top_level_unit_sequence_sha256': 'f8a57092c5b5eb9903aed1f46f78ef495e5258f52ef1fc8e7207f337010d972c',
                                        'target_top_level_unit_count': 52,
                                        'target_top_level_unit_sequence_sha256': 'f8a57092c5b5eb9903aed1f46f78ef495e5258f52ef1fc8e7207f337010d972c'},
 'tests/test_workflow.py': {'allowed_changed_top_level_units': ['class:FakeController#001',
                                                                'docstring#001',
                                                                'function:_adapter_marker#001',
                                                                'function:_configure_power_paths#001',
                                                                'function:_install_fit_count_crash#001',
                                                                'function:_prior_selection_audit#001',
                                                                'function:_synthetic_role_manifests#001',
                                                                'function:test_count_recovery_rejects_valid_hex_operation_replacement_even_if_rebound#001',
                                                                'function:test_default_freeze_writes_identity_summaries_only_after_gate_freeze#001',
                                                                'function:test_fit_count_commit_crash_reauthenticates_manifests_and_advances_once#001',
                                                                'function:test_inherited_early_stop_adapter_rejects_unsealed_active_contract#001',
                                                                'function:test_loaded_workflow_controller_adapts_both_inherited_early_stop_contracts#001',
                                                                'function:test_sealed_analysis_integrity_failure_is_staged_not_advanced#001'],
                            'deleted_top_level_units': [],
                            'inserted_top_level_units': [],
                            'required_changed_top_level_units': ['class:FakeController#001',
                                                                 'docstring#001',
                                                                 'function:_adapter_marker#001',
                                                                 'function:_configure_power_paths#001',
                                                                 'function:_install_fit_count_crash#001',
                                                                 'function:_prior_selection_audit#001',
                                                                 'function:_synthetic_role_manifests#001',
                                                                 'function:test_count_recovery_rejects_valid_hex_operation_replacement_even_if_rebound#001',
                                                                 'function:test_default_freeze_writes_identity_summaries_only_after_gate_freeze#001',
                                                                 'function:test_fit_count_commit_crash_reauthenticates_manifests_and_advances_once#001',
                                                                 'function:test_inherited_early_stop_adapter_rejects_unsealed_active_contract#001',
                                                                 'function:test_loaded_workflow_controller_adapts_both_inherited_early_stop_contracts#001',
                                                                 'function:test_sealed_analysis_integrity_failure_is_staged_not_advanced#001'],
                            'source_top_level_unit_count': 61,
                            'source_top_level_unit_sequence_sha256': '2a8ddf94ea891161dff81c06568af9814a060fe0dd2e6cbd4f127e265352a924',
                            'target_top_level_unit_count': 61,
                            'target_top_level_unit_sequence_sha256': '2a8ddf94ea891161dff81c06568af9814a060fe0dd2e6cbd4f127e265352a924'},
 'verify_version_forward.py': {'allowed_changed_top_level_units': ['annassign:LINEAGE_SUPPORT_UNIT_POLICY#001',
                                                                   'assign:EXACT#001',
                                                                   'assign:EXPECTED_INVALIDITY_DRAFT_SHA256#001',
                                                                   'assign:EXPECTED_INVALIDITY_SHA256#001',
                                                                   'assign:EXPECTED_SELECTION_SOURCE_ADAPTER_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_ADAPTER_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_FIT_INVENTORY_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_LAUNCHER_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_LEDGER_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_PRE_FORWARD_STATE_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_RECEIPT_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_SEAL_SHA256#001',
                                                                   'assign:EXPECTED_SOURCE_STATE_SHA256#001',
                                                                   'assign:EXPECTED_TRANSITIVE_SEALED_FILE_COUNT#001',
                                                                   'assign:INVALIDITY#001',
                                                                   'assign:INVALIDITY_DRAFT#001',
                                                                   'assign:LINEAGE_SUPPORT#001',
                                                                   'assign:SELECTION_SOURCE#001',
                                                                   'assign:SOURCE#001',
                                                                   'assign:SOURCE_ACTIVATION_COUNTS#001',
                                                                   'assign:SOURCE_OPERATIONAL_LOCKS#001',
                                                                   'assign:SOURCE_SELECTION_BOUNDARY_PRODUCTS#001',
                                                                   'assign:TARGET#001',
                                                                   'class:_Normalize#001',
                                                                   'docstring#001',
                                                                   'function:_audit_source_attempt_namespace#001',
                                                                   'function:_audit_study_root_namespace#001',
                                                                   'function:_authenticate_immediate_source#001',
                                                                   'function:_authenticate_immediate_source_legacy#001',
                                                                   'function:_closure#001',
                                                                   'function:_contract#001',
                                                                   'function:_load_active_transaction_receipt#001',
                                                                   'function:_roots#001',
                                                                   'function:_source_adapter_marker#001',
                                                                   'function:_validate_inheritance_seal_scalar_types#001',
                                                                   'function:_verify_adapter_count_event_bindings#001',
                                                                   'function:_verify_descendant_adapter_transaction_chain#001',
                                                                   'function:_verify_descendant_controller_transition#001',
                                                                   'function:_verify_initial_transaction_receipt_anchor#001',
                                                                   'function:_verify_records#001',
                                                                   'function:_verify_recovery_adapter_marker#001',
                                                                   'function:_verify_role_count_recovery_authorization#001',
                                                                   'function:verify#001'],
                               'deleted_top_level_units': [],
                               'inserted_top_level_units': ['assign:EXPECTED_SELECTION_SOURCE_ADAPTER_SHA256#001',
                                                            'assign:SELECTION_SOURCE#001'],
                               'required_changed_top_level_units': ['annassign:LINEAGE_SUPPORT_UNIT_POLICY#001',
                                                                    'assign:EXACT#001',
                                                                    'assign:EXPECTED_INVALIDITY_DRAFT_SHA256#001',
                                                                    'assign:EXPECTED_INVALIDITY_SHA256#001',
                                                                    'assign:EXPECTED_SELECTION_SOURCE_ADAPTER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_ADAPTER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_FIT_INVENTORY_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_LAUNCHER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_LEDGER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_PRE_FORWARD_STATE_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_RECEIPT_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_SEAL_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_STATE_SHA256#001',
                                                                    'assign:EXPECTED_TRANSITIVE_SEALED_FILE_COUNT#001',
                                                                    'assign:INVALIDITY#001',
                                                                    'assign:INVALIDITY_DRAFT#001',
                                                                    'assign:LINEAGE_SUPPORT#001',
                                                                    'assign:SELECTION_SOURCE#001',
                                                                    'assign:SOURCE#001',
                                                                    'assign:SOURCE_ACTIVATION_COUNTS#001',
                                                                    'assign:SOURCE_OPERATIONAL_LOCKS#001',
                                                                    'assign:SOURCE_SELECTION_BOUNDARY_PRODUCTS#001',
                                                                    'assign:TARGET#001',
                                                                    'class:_Normalize#001',
                                                                    'docstring#001',
                                                                    'function:_audit_source_attempt_namespace#001',
                                                                    'function:_audit_study_root_namespace#001',
                                                                    'function:_authenticate_immediate_source#001',
                                                                    'function:_authenticate_immediate_source_legacy#001',
                                                                    'function:_closure#001',
                                                                    'function:_contract#001',
                                                                    'function:_load_active_transaction_receipt#001',
                                                                    'function:_roots#001',
                                                                    'function:_source_adapter_marker#001',
                                                                    'function:_validate_inheritance_seal_scalar_types#001',
                                                                    'function:_verify_adapter_count_event_bindings#001',
                                                                    'function:_verify_descendant_adapter_transaction_chain#001',
                                                                    'function:_verify_descendant_controller_transition#001',
                                                                    'function:_verify_initial_transaction_receipt_anchor#001',
                                                                    'function:_verify_records#001',
                                                                    'function:_verify_recovery_adapter_marker#001',
                                                                    'function:_verify_role_count_recovery_authorization#001',
                                                                    'function:verify#001'],
                               'source_top_level_unit_count': 136,
                               'source_top_level_unit_sequence_sha256': 'a9c0378e1740a24e6e2a9eccce6e6e816f733c9ff92ee56607aff452ffdf803d',
                               'target_top_level_unit_count': 138,
                               'target_top_level_unit_sequence_sha256': '041831dbb5e01db0aa6d63170db51a7dc6f669cb552551c1f3ba1137c502512c'},
 'version_forward_prepare.py': {'allowed_changed_top_level_units': ['annassign:LINEAGE_SUPPORT_UNIT_POLICY#001',
                                                                    'annassign:SOURCE_ACTIVATION_OUTCOMES#001',
                                                                    'assign:EXACT_HASH_PATHS#001',
                                                                    'assign:EXPECTED_INVALIDITY_DRAFT_SHA256#001',
                                                                    'assign:EXPECTED_INVALIDITY_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_ADAPTER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_FIT_INVENTORY_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_LAUNCHER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_LEDGER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_PRE_FORWARD_STATE_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_RECEIPT_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_SEAL_SHA256#001',
                                                                    'assign:EXPECTED_SOURCE_STATE_SHA256#001',
                                                                    'assign:EXPECTED_TRANSITIVE_SEALED_FILE_COUNT#001',
                                                                    'assign:INVALIDITY_DRAFT_RELATIVE#001',
                                                                    'assign:INVALIDITY_RELATIVE#001',
                                                                    'assign:LINEAGE_SUPPORT_PATHS#001',
                                                                    'assign:SELECTION_SOURCE_ATTEMPT#001',
                                                                    'assign:SOURCE_ATTEMPT#001',
                                                                    'assign:SOURCE_OPERATIONAL_LOCKS#001',
                                                                    'assign:SOURCE_SELECTION_BOUNDARY_PRODUCTS#001',
                                                                    'assign:TARGET_ATTEMPT#001',
                                                                    'class:_AttemptNormalizer#001',
                                                                    'docstring#001',
                                                                    'function:_attempt_roots#001',
                                                                    'function:_audit_source_attempt_namespace#001',
                                                                    'function:_audit_study_root_namespace#001',
                                                                    'function:_manifest_closure#001',
                                                                    'function:_normalize_contract#001',
                                                                    'function:_partition_sources#001',
                                                                    'function:_state_and_lineage#001',
                                                                    'function:build_inheritance_seal#001'],
                                'deleted_top_level_units': [],
                                'inserted_top_level_units': ['assign:SELECTION_SOURCE_ATTEMPT#001'],
                                'required_changed_top_level_units': ['annassign:LINEAGE_SUPPORT_UNIT_POLICY#001',
                                                                     'annassign:SOURCE_ACTIVATION_OUTCOMES#001',
                                                                     'assign:EXACT_HASH_PATHS#001',
                                                                     'assign:EXPECTED_INVALIDITY_DRAFT_SHA256#001',
                                                                     'assign:EXPECTED_INVALIDITY_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_ADAPTER_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_FIT_INVENTORY_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_LAUNCHER_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_LEDGER_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_PRE_FORWARD_STATE_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_RECEIPT_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_SEAL_SHA256#001',
                                                                     'assign:EXPECTED_SOURCE_STATE_SHA256#001',
                                                                     'assign:EXPECTED_TRANSITIVE_SEALED_FILE_COUNT#001',
                                                                     'assign:INVALIDITY_DRAFT_RELATIVE#001',
                                                                     'assign:INVALIDITY_RELATIVE#001',
                                                                     'assign:LINEAGE_SUPPORT_PATHS#001',
                                                                     'assign:SELECTION_SOURCE_ATTEMPT#001',
                                                                     'assign:SOURCE_ATTEMPT#001',
                                                                     'assign:SOURCE_OPERATIONAL_LOCKS#001',
                                                                     'assign:SOURCE_SELECTION_BOUNDARY_PRODUCTS#001',
                                                                     'assign:TARGET_ATTEMPT#001',
                                                                     'class:_AttemptNormalizer#001',
                                                                     'docstring#001',
                                                                     'function:_attempt_roots#001',
                                                                     'function:_audit_source_attempt_namespace#001',
                                                                     'function:_audit_study_root_namespace#001',
                                                                     'function:_manifest_closure#001',
                                                                     'function:_normalize_contract#001',
                                                                     'function:_partition_sources#001',
                                                                     'function:_state_and_lineage#001',
                                                                     'function:build_inheritance_seal#001'],
                                'source_top_level_unit_count': 127,
                                'source_top_level_unit_sequence_sha256': '57dcf66653e1040bb89f3d64c1fb916606ffe46c292ef406c7eeed8a4dcf37f9',
                                'target_top_level_unit_count': 128,
                                'target_top_level_unit_sequence_sha256': '6b107e69a84f01c59f35f3fcd719c6fd064c2a85e71ae7c8c606c1adaf70f439'},
 'version_forward_transaction.py': {'allowed_changed_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                                        'assign:ADAPTER_ARTIFACT_TYPE#001',
                                                                        'assign:ADAPTER_AUTHORIZATION_KIND#001',
                                                                        'assign:ADAPTER_STATE_KEY#001',
                                                                        'assign:ADAPTER_TRANSACTION_ENVELOPE_FIELD#001',
                                                                        'assign:ADAPTER_TRANSACTION_ENVELOPE_KIND#001',
                                                                        'assign:INVALIDITY_DRAFT_PATH#001',
                                                                        'assign:INVALIDITY_PATH#001',
                                                                        'assign:PENDING_STAGING_PATH#001',
                                                                        'assign:PRIOR_ADAPTER_STATE_KEY#001',
                                                                        'assign:PRIOR_RECEIPT_PATH#001',
                                                                        'assign:RECEIPT_JOURNAL_ARTIFACT_TYPE#001',
                                                                        'assign:RECEIPT_JOURNAL_STAGING_PATH#001',
                                                                        'assign:RECEIPT_STAGING_PATH#001',
                                                                        'assign:SOURCE_ATTEMPT#001',
                                                                        'assign:STATE_STAGING_PATH#001',
                                                                        'assign:TARGET_ATTEMPT#001',
                                                                        'docstring#001',
                                                                        'function:_authenticated_active_seal_verifier_projection#001',
                                                                        'function:_bind_prior_marker_to_count_events#001',
                                                                        'function:_build_receipt_value#001',
                                                                        'function:_invoke_root_locked#001',
                                                                        'function:_postverify#001',
                                                                        'function:_preverify#001',
                                                                        'function:_require_presealed_verifier_contract#001',
                                                                        'function:_rollover_source_adapter_marker#001',
                                                                        'function:_validate_adapter_marker#001',
                                                                        'function:_validate_post_state#001',
                                                                        'function:_validate_pre_state#001',
                                                                        'function:_validate_receipt_closed_schema#001',
                                                                        'function:_validate_source_adapter_marker#001',
                                                                        'function:verify_transaction_receipt#001',
                                                                        'function:version_forward#001'],
                                    'deleted_top_level_units': [],
                                    'inserted_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001'],
                                    'required_changed_top_level_units': ['assign:ACTIVATION_SOURCE_ATTEMPT#001',
                                                                         'assign:ADAPTER_ARTIFACT_TYPE#001',
                                                                         'assign:ADAPTER_AUTHORIZATION_KIND#001',
                                                                         'assign:ADAPTER_STATE_KEY#001',
                                                                         'assign:ADAPTER_TRANSACTION_ENVELOPE_FIELD#001',
                                                                         'assign:ADAPTER_TRANSACTION_ENVELOPE_KIND#001',
                                                                         'assign:INVALIDITY_DRAFT_PATH#001',
                                                                         'assign:INVALIDITY_PATH#001',
                                                                         'assign:PENDING_STAGING_PATH#001',
                                                                         'assign:PRIOR_ADAPTER_STATE_KEY#001',
                                                                         'assign:PRIOR_RECEIPT_PATH#001',
                                                                         'assign:RECEIPT_JOURNAL_ARTIFACT_TYPE#001',
                                                                         'assign:RECEIPT_JOURNAL_STAGING_PATH#001',
                                                                         'assign:RECEIPT_STAGING_PATH#001',
                                                                         'assign:SOURCE_ATTEMPT#001',
                                                                         'assign:STATE_STAGING_PATH#001',
                                                                         'assign:TARGET_ATTEMPT#001',
                                                                         'docstring#001',
                                                                         'function:_authenticated_active_seal_verifier_projection#001',
                                                                         'function:_bind_prior_marker_to_count_events#001',
                                                                         'function:_build_receipt_value#001',
                                                                         'function:_invoke_root_locked#001',
                                                                         'function:_postverify#001',
                                                                         'function:_preverify#001',
                                                                         'function:_require_presealed_verifier_contract#001',
                                                                         'function:_rollover_source_adapter_marker#001',
                                                                         'function:_validate_adapter_marker#001',
                                                                         'function:_validate_post_state#001',
                                                                         'function:_validate_pre_state#001',
                                                                         'function:_validate_receipt_closed_schema#001',
                                                                         'function:_validate_source_adapter_marker#001',
                                                                         'function:verify_transaction_receipt#001',
                                                                         'function:version_forward#001'],
                                    'source_top_level_unit_count': 162,
                                    'source_top_level_unit_sequence_sha256': '22047e77b4f01b3014b26e62de2624fe92267696aa675bb35e7debcb9cb66547',
                                    'target_top_level_unit_count': 163,
                                    'target_top_level_unit_sequence_sha256': 'a93ba926c7dbad8bbe5f49aa3f59b17f995d71b8daa11c2ed42fb1fd573733bd'},
 'workflow.py': {'allowed_changed_top_level_units': ['function:_load_controller#001',
                                                     'function:_require_inherited_presealed_verifier_contract#001',
                                                     'function:_validate_role_replay_qualification#001',
                                                     'function:_validate_scientific_replay_result#001',
                                                     'function:_verify_inherited_fit_manifests#001',
                                                     'function:_verify_recovery_adapter_marker#001',
                                                     'function:_verify_role_count_update_recovery#001',
                                                     'function:qualify_role_scientific_replay#001',
                                                     'function:verify_role_manifests#001'],
                 'deleted_top_level_units': [],
                 'inserted_top_level_units': [],
                 'required_changed_top_level_units': ['function:_load_controller#001',
                                                      'function:_require_inherited_presealed_verifier_contract#001',
                                                      'function:_validate_role_replay_qualification#001',
                                                      'function:_validate_scientific_replay_result#001',
                                                      'function:_verify_inherited_fit_manifests#001',
                                                      'function:_verify_recovery_adapter_marker#001',
                                                      'function:_verify_role_count_update_recovery#001',
                                                      'function:qualify_role_scientific_replay#001',
                                                      'function:verify_role_manifests#001'],
                 'source_top_level_unit_count': 122,
                 'source_top_level_unit_sequence_sha256': '36ff65351b11d1210e2c34f840feed4191ed3bbf1ece4b3e3aee769476246d35',
                 'target_top_level_unit_count': 122,
                 'target_top_level_unit_sequence_sha256': '36ff65351b11d1210e2c34f840feed4191ed3bbf1ece4b3e3aee769476246d35'}}
SCIENTIFIC_OBJECT_PATHS = frozenset({
    "DGP_MATRIX.json", "DIAGNOSTIC_ACCOUNT.md", "PREREGISTRATION.md",
    "analysis.py", "audit/bootstrap_audit.json", "audit/diagnostic_account.json",
    "audit/identifier_freshness_verification.json", "audit/preregistration_and_power.json",
    "candidate_grid.json", "cohort_seed_ledger.json", "compile_gate.py",
    "counted_features.py", "fit_select.py", "flops.py", "input_loader.py",
    "outcome_mapping.json", "power_analysis.py", "power_baseline.json",
    "power_rule.json",
})

DGP_ORDER = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ROLE_ORDER = ("fit", "selection", "smoke", "confirmation")
ROLE_COUNTS = {"fit": 300, "selection": 500, "smoke": 6}
SEED_FIELDS = (
    "env_seed",
    "policy_seed",
    "oracle_np_seed",
    "action_space_seed",
)
ENDPOINTS = ("raw", "fixed_whitened")
COMPARATORS = (
    "analytic",
    "seeded_weakly_more_compute",
    "fixed_depth_1",
    "within_episode_histogram",
)
ALL_CONTRASTS = tuple(
    f"{endpoint}_vs_{comparator}"
    for endpoint in ENDPOINTS
    for comparator in COMPARATORS
)
PRIMARY_CONTRASTS = ("raw_vs_analytic", "fixed_whitened_vs_analytic")

ARCHITECTURES = ("balanced_pooled_dual", "domain_envelope_eight")
ARCHITECTURE_HEADS = {"balanced_pooled_dual": 2, "domain_envelope_eight": 8}
RIDGES = (0.01, 1.0, 100.0)
FIT_QUANTILES = (0.55, 0.65, 0.75, 0.85)
CANDIDATE_COUNT = 24

LATENT_DIM = 192
ACTION_DIM = 25
HISTORY_LEN = 3
FEATURE_DIM = 1046
STAGE_COUNT = 3
EXIT_COUNT = 4
ROWS_PER_EPISODE = 38
MAX_HEADS = 8

BASE_FLOPS = 70_529_190
MANDATORY_DEPTH1_FLOPS = 669_184
ADDITIONAL_REFINER_FLOPS = 264_960
FEATURE_FLOPS = 3_801
AFFINE_HEAD_FLOPS = 2_092
GATE_FLOPS = {2: 7_985, 8: 20_537}
GATE_NONFLOPS = {2: 5, 8: 11}

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_CHUNK = 250
FAMILY_SIZE = 8
FAMILYWISE_ALPHA = 0.05
PER_CLAIM_ALPHA = 0.00625
V5_WHITENING_SHA256 = (
    "515d31ea8df1afa6c11368236c507eecf1853abfaa9239189c17855445ccf796"
)
POWER_RULE_SHA256 = "de64698680b3060181120ce40a07f447f2746af4c3f9c6aab882965a639e467a"
POWER_RULE_MOMENTS_SHA256 = "5b32f9a48dc6d2df07f19d452d9031359eb5bb9a30778eb26f47709f4d20e36f"

TERMINAL_CONFIRMED = "domain_robust_gate_confirmed"
TERMINAL_PARTIAL = "domain_robust_gate_partial"
TERMINAL_FAILED = "domain_robust_gate_failed"
TERMINAL_INVALID = "domain_robust_gate_execution_invalid"

SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE = (
    "audit/scientific_replay_qualification.json"
)
REPLAY_ROLE_ORDER = ("fit", "selection", "smoke", "confirmation")
REPLAY_ROLE_STATES = {
    "fit": "FIT_COHORTS",
    "selection": "SELECTION_COHORTS",
    "smoke": "EXCLUDED_MECHANICAL_SMOKE",
    "confirmation": "CONFIRMATION_EXECUTION",
}
REPLAY_FIXED_EPISODES_PER_DGP = {"fit": 300, "selection": 500, "smoke": 6}
REPLAY_RESULT_KEYS = frozenset({
    "schema_version", "artifact_type", "attempt", "role", "regime",
    "raw_manifest", "execution_manifest", "compiled_gate", "episode_count",
    "row_count", "episodes", "aggregate", "runtime_audit", "source_hashes",
    "module_before", "module_after", "input_loader_agreement_exact",
    "every_persisted_tensor_exact", "aggregate_exact", "target_loss_computed",
    "loss_or_effect_used_for_acceptance", "contact_or_privileged_materialized",
    "no_gradients", "artifact_tree_unchanged", "read_only", "passed",
})
REPLAY_EPISODE_KEYS = frozenset({
    "slot", "episode_id", "raw", "raw_sidecar", "execution_part",
    "execution_sidecar", "input_loader_audit_sha256", "array_sha256",
    "every_persisted_tensor_exact",
})
REPLAY_ROLE_AUDIT_KEYS = frozenset({
    "schema_version", "artifact_type", "attempt", "role", "state",
    "authorization_state_sha256", "manifest_verification_sha256",
    "regime_order", "episodes_per_regime", "episode_count", "row_count",
    "regimes", "source_hashes", "input_loader_agreement_exact",
    "every_persisted_tensor_exact", "all_development_aggregates_exact",
    "all_runtime_mps_exact", "target_loss_computed",
    "loss_or_effect_used_for_acceptance", "contact_or_privileged_materialized",
    "no_gradients", "artifact_trees_unchanged", "read_only", "passed",
})
REPLAY_ROLE_REGIME_KEYS = frozenset({
    "result", "raw_manifest", "execution_manifest", "aggregate",
    "episode_count", "row_count",
})
REPLAY_DEVELOPMENT_ARRAY_KEYS = frozenset({
    "episode_slot", "model_step", "target", "exits", "production_features",
    "history", "action_history", "stage_current", "stage_update",
})
REPLAY_ROBUST_ARRAY_KEYS = frozenset({
    "episode_slot", "model_step", "exits", "selected", "calls", "scores",
    "head_scores", "production_features", "history", "action_history",
    "stage_current", "stage_update", "reached", "dense_scores",
    "dense_head_scores", "dense_stage_current", "dense_stage_update",
})

STATE_MACHINE = (
    "BOOTSTRAP_AUDIT",
    "DIAGNOSTIC_ACCOUNT",
    "PREREGISTRATION_AND_POWER",
    "IMPLEMENTATION_COMPLETE",
    "PRESEAL_QUALIFICATION",
    "PRE_OUTCOME_SEAL",
    "FIT_COHORTS",
    "FIT_LOCK",
    "PRE_SELECTION_SEAL",
    "SELECTION_COHORTS",
    "CANDIDATE_SELECTION",
    "GATE_FREEZE",
    "CONFIRMATION_POWER_AND_COHORT_FREEZE",
    "PRE_CONFIRMATION_PACKAGE_SEAL",
    "EXCLUDED_MECHANICAL_SMOKE",
    "CONFIRMATION_GENERATION",
    "CONFIRMATION_EXECUTION",
    "CONFIRMATION_INPUT_SEAL",
    "SEALED_ANALYSIS",
    "LATENCY_AND_RESOURCE_REPORTING",
    "INDEPENDENT_VERIFICATION",
    "TERMINAL",
    "POST_TERMINAL_REPORTING",
)
INHERITED_COMPLETED_STATES = STATE_MACHINE[: STATE_MACHINE.index("SELECTION_COHORTS")]
VERSION_FORWARD_RESUME_STATE = "SELECTION_COHORTS"

COMMON_PATH_NAMES = (
    "state",
    "ledger",
    "ledger_genesis",
    "dgp_matrix",
    "candidate_grid",
    "power_rule",
    "outcome_mapping",
    "cohort_seed_ledger",
    "pre_data_seal",
    "pre_selection_seal",
    "terminal_manifest",
    "fitted_candidates",
    "fit_lock",
    "selection_ledger",
    "fixed_whitening",
)
MODE_PATH_NAMES = {
    "no_candidate": COMMON_PATH_NAMES,
    "power_infeasible": COMMON_PATH_NAMES + (
        "gate_fit", "compiled_gate", "compiled_gate_manifest", "gate_freeze", "power_freeze",
    ),
    "confirmation": COMMON_PATH_NAMES + (
        "pre_confirmation_seal", "gate_fit", "compiled_gate", "compiled_gate_manifest", "gate_freeze",
        "power_freeze", "confirmation_input_seal", "analysis_result", "bootstrap_replicates", "bootstrap_summary",
    ),
}

EXECUTION_KEYS = frozenset(
    {
        "episode_slot",
        "model_step",
        "target",
        "exits",
        "selected",
        "calls",
        "scores",
        "production_features",
        "history",
        "action_history",
        "stage_current",
        "stage_update",
        "head_scores",
        "reached",
        "dense_scores",
        "dense_head_scores",
        "dense_stage_current",
        "dense_stage_update",
    }
)
DEVELOPMENT_KEYS = frozenset(
    {
        "episode_slot",
        "model_step",
        "target",
        "exits",
        "production_features",
        "history",
        "action_history",
        "stage_current",
        "stage_update",
    }
)
DEVELOPMENT_AGGREGATE_KEYS = frozenset(
    {"episode_slot", "model_step", "target", "exits", "production_features"}
)
FORBIDDEN_TOKENS = (
    "contact",
    "privileged",
    "qpos",
    "qvel",
    "motion",
    "phase",
    "reward",
    "success",
)
SAFE_V3_RELATIVE_PATHS = (
    "lewm_adaptive_compute_v3/run_experiment.py",
    "lewm_adaptive_compute_v3/model.py",
    "lewm_adaptive_compute_v3/policy.py",
    "lewm_adaptive_compute_v3/full_config.json",
    "lewm_adaptive_compute_v3/smoke_config.json",
    "lewm_adaptive_compute_v3/preregistration.json",
    "lewm_adaptive_compute_v3/PLAN.md",
)


class VerificationError(RuntimeError):
    """A sealed integrity or scientific-contract check failed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_object_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise VerificationError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Producer-compatible dtype/shape/C-byte array digest."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def replay_array_sha256(array: np.ndarray) -> str:
    """Digest independently transcribed from the sealed replay qualifier."""

    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _set_digest(values: Iterable[Any]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _validate_inheritance_seal_scalar_types(value: Any) -> None:
    """Reject every JSON numeric alias in the inheritance-seal schema."""

    integer_fields = {
        "bytes",
        "confirmation_outcome_episodes_executed",
        "confirmation_outcome_episodes_generated",
        "created_unix_ns",
        "event_count",
        "episode_count",
        "file_count",
        "fit_outcome_episodes",
        "ledger_event_count",
        "schema_version",
        "science_attempt_exact_path_count",
        "selection_outcome_episodes",
        "smoke_outcome_episodes",
        "source_attempt_allowed_suffix_path_count",
        "source_top_level_unit_count",
        "target_top_level_unit_count",
        "unchanged_top_level_symbol_count",
        "unchanged_top_level_unit_count",
    }
    boolean_fields = {
        "all_inherited_checkpoints_authenticated",
        "all_rehashed",
        "ast_equivalent",
        "attempt_parameterization_verified",
        "authoritative",
        "complete_scope_live_walk",
        "complete_source_partition",
        "configuration_hash_equivalent",
        "confirmation_outcomes_opened_for_analysis",
        "contracts_canonically_equivalent",
        "expected_boundary_outcome_counts",
        "hashes_equivalent",
        "hdf5_contents_opened",
        "independent_verifier_accepts_active_attempt",
        "invalidity_authenticated",
        "no_confirmation_artifacts",
        "normalized_ast_equivalent",
        "outcome_arrays_opened",
        "passed",
        "procedural_invalidity_confirmed",
        "root_controls_authenticated",
        "runtime_modules_accept_active_attempt",
        "scientific_changes",
        "scientific_object_hash_equivalent",
        "scientific_objects_equivalent",
        "source_equivalent",
        "source_hash_equivalent",
        "source_pre_data_seal_authenticated",
        "source_pre_selection_seal_authenticated",
        "source_seal_bound",
        "source_sealed_files_rehashed",
        "state_transaction_present",
        "study_root_namespace_complete",
        "v008_manifest_closure",
        "v3_targets_opened",
        "zero_confirmation_outcomes_at_version_forward",
        "zero_outcome_counters",
    }
    stack: list[tuple[str | None, Any]] = [(None, value)]
    while stack:
        key, item = stack.pop()
        if isinstance(item, Mapping):
            _require(
                all(isinstance(child_key, str) for child_key in item),
                "inheritance seal contains a non-string object key",
            )
            stack.extend((str(child_key), child) for child_key, child in item.items())
            continue
        if isinstance(item, list):
            stack.extend((None, child) for child in item)
            continue
        if key in integer_fields:
            _require(type(item) is int, f"inheritance seal integer type drift: {key}")
        elif key in boolean_fields:
            _require(type(item) is bool, f"inheritance seal boolean type drift: {key}")
        else:
            _require(
                type(item) not in (bool, int, float),
                f"inheritance seal unexpected numeric scalar: {key}",
            )


def _canonical_relative(raw: Any) -> str:
    _require(isinstance(raw, str) and bool(raw), "path must be a nonempty string")
    _require("\\" not in raw, f"path uses a non-POSIX separator: {raw}")
    pure = PurePosixPath(raw)
    _require(not pure.is_absolute(), f"absolute path forbidden: {raw}")
    _require(".." not in pure.parts and "." not in pure.parts, f"path alias forbidden: {raw}")
    canonical = pure.as_posix()
    _require(canonical == raw and not raw.endswith("/"), f"noncanonical path: {raw}")
    return canonical


class PathPolicy:
    """Canonical containment and no-link policy for every consumed artifact."""

    def __init__(self, repository_root: Path, attempt_root: str, study_root: str):
        self.repository_root = repository_root.resolve(strict=True)
        self.attempt_relative = _canonical_relative(attempt_root)
        self.study_relative = _canonical_relative(study_root)
        self.attempt_root = self.repository_root / self.attempt_relative
        self.study_root = self.repository_root / self.study_relative
        _require(self.attempt_root.is_dir(), "attempt root is absent")
        _require(self.study_root.is_dir(), "study root is absent")
        self._reject_link_components(self.attempt_root)
        self._reject_link_components(self.study_root)

    def _reject_link_components(self, path: Path) -> None:
        relative = path.relative_to(self.repository_root)
        current = self.repository_root
        for part in relative.parts:
            current = current / part
            mode = current.lstat().st_mode
            _require(not stat.S_ISLNK(mode), f"symlink component forbidden: {current}")

    def path(
        self,
        raw: Any,
        *,
        scope: str = "repository",
        must_exist: bool = True,
        file_only: bool = True,
    ) -> Path:
        relative = _canonical_relative(raw)
        candidate = self.repository_root / relative
        parent = candidate.parent
        if must_exist:
            _require(candidate.exists(), f"required path absent: {relative}")
            self._reject_link_components(candidate)
        elif parent.exists():
            self._reject_link_components(parent)
        if scope == "attempt":
            _require(candidate.is_relative_to(self.attempt_root), f"path outside attempt: {relative}")
        elif scope == "study":
            _require(candidate.is_relative_to(self.study_root), f"path outside study: {relative}")
        elif scope != "repository":
            raise VerificationError(f"unknown path scope: {scope}")
        if must_exist and file_only:
            _require(candidate.is_file(), f"expected regular file: {relative}")
            mode = candidate.lstat().st_mode
            _require(stat.S_ISREG(mode), f"non-regular artifact forbidden: {relative}")
        return candidate


def load_contract(contract_path: Path) -> tuple[dict[str, Any], PathPolicy]:
    _require(contract_path.is_file(), f"contract absent: {contract_path}")
    _require(not contract_path.is_symlink(), "verifier contract may not be a symlink")
    contract = read_json(contract_path)
    _require(contract.get("schema_version") == SCHEMA_VERSION, "contract schema drift")
    active_attempt = contract.get("attempt")
    _require(active_attempt == ACTIVE_ATTEMPT, "contract active attempt is not v008")
    lineage = contract.get("lineage")
    _require(isinstance(lineage, Mapping), "lineage contract must be an object")
    expected_lineage_keys = {
        "science_attempt",
        "active_attempt",
        "attempts",
        "source_roots",
        "invalidity_evidence",
        "source_pre_data_seal",
        "pre_data_inheritance_seal",
        "prior_pre_data_inheritance_seal",
        "source_invalidity_evidence",
        "superseded_source_invalidity_draft",
        "prior_version_forward_receipt",
        "version_forward_receipt",
    }
    _require(set(lineage) == expected_lineage_keys, "lineage contract key set drift")
    _require(lineage.get("science_attempt") == SCIENCE_ATTEMPT, "science attempt drift")
    _require(lineage.get("active_attempt") == ACTIVE_ATTEMPT, "lineage active attempt drift")
    _require(tuple(lineage.get("attempts", ())) == ALLOWED_ATTEMPTS, "allowed lineage is not exactly v001->v002->v008")
    _require(lineage.get("source_roots") == ATTEMPT_ROOTS, "lineage source-root mapping drift")
    _require(lineage.get("invalidity_evidence") == SCIENCE_INVALIDITY_RELATIVE, "science invalidity lineage path drift")
    _require(lineage.get("source_pre_data_seal") == SCIENCE_PRE_DATA_RELATIVE, "science pre-data lineage path drift")
    _require(lineage.get("pre_data_inheritance_seal") == INHERITANCE_SEAL_RELATIVE, "inheritance-seal lineage path drift")
    _require(lineage.get("prior_pre_data_inheritance_seal") == SOURCE_PRE_DATA_RELATIVE, "prior inheritance-seal lineage path drift")
    _require(lineage.get("source_invalidity_evidence") == INVALIDITY_RELATIVE, "source invalidity lineage path drift")
    _require(lineage.get("superseded_source_invalidity_draft") == INVALIDITY_DRAFT_RELATIVE, "source invalidity draft path drift")
    _require(lineage.get("prior_version_forward_receipt") == SOURCE_RECEIPT_RELATIVE, "prior receipt path drift")
    _require(lineage.get("version_forward_receipt") == f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/audit/version_forward_transaction_receipt.json", "active receipt path drift")
    attempt_relative = _canonical_relative(contract.get("attempt_root"))
    _require(attempt_relative == ATTEMPT_ROOTS[ACTIVE_ATTEMPT], "attempt root/version drift")
    _require(contract.get("study_root") == STUDY_RELATIVE, "study root drift")
    attempt_parts = PurePosixPath(attempt_relative).parts
    parent = contract_path.resolve().parent
    _require(tuple(parent.parts[-len(attempt_parts) :]) == attempt_parts, "contract not in declared attempt")
    repository_root = parent
    for _ in attempt_parts:
        repository_root = repository_root.parent
    policy = PathPolicy(repository_root, attempt_relative, contract.get("study_root"))
    _require(contract_path.resolve() == policy.attempt_root / contract_path.name, "contract path alias")
    contract["_loaded_contract_relative"] = contract_path.resolve().relative_to(policy.repository_root).as_posix()
    _require(contract.get("mode") in {"confirmation", "no_candidate", "power_infeasible"}, "unknown mode")
    paths = contract.get("paths")
    _require(isinstance(paths, dict), "contract paths must be an object")
    missing = [name for name in MODE_PATH_NAMES[contract["mode"]] if name not in paths]
    _require(not missing, f"contract missing paths: {missing}")
    science_paths = {
        "dgp_matrix": f"{ATTEMPT_ROOTS[SCIENCE_ATTEMPT]}/DGP_MATRIX.json",
        "candidate_grid": f"{ATTEMPT_ROOTS[SCIENCE_ATTEMPT]}/candidate_grid.json",
        "power_rule": f"{ATTEMPT_ROOTS[SCIENCE_ATTEMPT]}/power_rule.json",
        "outcome_mapping": f"{ATTEMPT_ROOTS[SCIENCE_ATTEMPT]}/outcome_mapping.json",
        "cohort_seed_ledger": f"{ATTEMPT_ROOTS[SCIENCE_ATTEMPT]}/cohort_seed_ledger.json",
        "pre_data_seal": SCIENCE_PRE_DATA_RELATIVE,
    }
    for name, expected in science_paths.items():
        _require(paths.get(name) == expected, f"science identity path drift: {name}")
    study_paths = {
        "state": f"{STUDY_RELATIVE}/STATE.json",
        "ledger": f"{STUDY_RELATIVE}/RESEARCH_LEDGER.jsonl",
        "ledger_genesis": f"{STUDY_RELATIVE}/LEDGER_CHAIN_GENESIS.json",
    }
    for name, expected in study_paths.items():
        _require(paths.get(name) == expected, f"study controller path drift: {name}")
    inherited_paths = {
        "pre_selection_seal": SOURCE_PRE_SELECTION_RELATIVE,
        "fitted_candidates": (
            f"{ATTEMPT_ROOTS[SOURCE_ATTEMPT]}/fit/fitted_candidates.npz"
        ),
        "fit_lock": f"{ATTEMPT_ROOTS[SOURCE_ATTEMPT]}/fit/fit_lock.json",
    }
    for name, expected in inherited_paths.items():
        if name in MODE_PATH_NAMES[contract["mode"]]:
            _require(
                paths.get(name) == expected,
                f"inherited selection-boundary path drift: {name}",
            )
    active_output_names = (
        set(MODE_PATH_NAMES[contract["mode"]])
        - set(science_paths)
        - set(study_paths)
        - set(inherited_paths)
        - {"fixed_whitening"}
    )
    active_prefix = f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/"
    for name in active_output_names:
        raw = _canonical_relative(paths[name])
        _require(raw.startswith(active_prefix), f"active output is not rooted in v008: {name}")
    fit_data_prefix = f"{ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]}/data/fit/"
    active_role_prefixes = {
        "selection_inputs": f"{active_prefix}data/selection/",
        "confirmation_manifests": f"{active_prefix}data/confirmation/",
    }
    fit_inputs = contract.get("fit_inputs", {})
    _require(isinstance(fit_inputs, Mapping), "fit_inputs must be an object")
    for raw in fit_inputs.values():
        _require(
            _canonical_relative(raw).startswith(fit_data_prefix),
            "fit_inputs path is not rooted in the immutable v003 fit role",
        )
    for mapping_name, expected_prefix in active_role_prefixes.items():
        mapping = contract.get(mapping_name, {})
        _require(isinstance(mapping, Mapping), f"{mapping_name} must be an object")
        for raw in mapping.values():
            _require(
                _canonical_relative(raw).startswith(expected_prefix),
                f"{mapping_name} path is not rooted in its v008 dataset role",
            )
    for mapping_name in ("development_manifests", "role_manifests"):
        outer = contract.get(mapping_name, {})
        _require(isinstance(outer, Mapping), f"{mapping_name} must be an object")
        for role, inner in outer.items():
            _require(isinstance(inner, Mapping), f"{mapping_name} nested value must be an object")
            expected_prefix = (
                fit_data_prefix
                if role == "fit"
                else f"{active_prefix}data/{role}/"
            )
            for raw in inner.values():
                _require(
                    _canonical_relative(raw).startswith(expected_prefix),
                    f"{mapping_name} path is not rooted in its isolated dataset role",
                )
    for name in ("confirmation_artifact_roots", "zero_confirmation_paths", "producer_sources"):
        values = contract.get(name, [])
        _require(isinstance(values, list), f"{name} must be a list")
        for raw in values:
            _require(_canonical_relative(raw).startswith(active_prefix), f"{name} path is not rooted in v008")
    invalid_relative = f"{attempt_relative}/audit/analysis_execution_invalid.json"
    invalid_path = policy.repository_root / invalid_relative
    analysis_raw = paths.get("analysis_result")
    analysis_path = (
        policy.repository_root / _canonical_relative(analysis_raw)
        if isinstance(analysis_raw, str)
        else None
    )
    analysis_execution_invalid = bool(
        contract["mode"] == "confirmation"
        and invalid_path.is_file()
    )
    if invalid_path.exists() or invalid_path.is_symlink():
        policy.path(invalid_relative, scope="attempt")
    contract["_analysis_execution_invalid_branch"] = analysis_execution_invalid
    contract["_analysis_execution_invalid_relative"] = invalid_relative
    failure_optional = {"analysis_result", "bootstrap_replicates", "bootstrap_summary"}
    for name, raw in paths.items():
        scope = "study" if name in {"state", "ledger", "ledger_genesis"} else "repository"
        must_exist = not (
            name in failure_optional
            and (contract["mode"] != "confirmation" or analysis_execution_invalid)
        )
        policy.path(raw, scope=scope, must_exist=must_exist)
    return contract, policy


def _record_hash(record: Any) -> str:
    if isinstance(record, str):
        return record
    _require(isinstance(record, Mapping), "hash record must be a string or object")
    value = record.get("sha256")
    _require(isinstance(value, str) and len(value) == 64, "manifest SHA-256 absent")
    return value


def verify_file_map(
    files: Mapping[str, Any],
    policy: PathPolicy,
    *,
    scope: str,
) -> dict[str, Any]:
    _require(isinstance(files, Mapping) and bool(files), "manifest file map is empty")
    seen_inodes: dict[tuple[int, int], str] = {}
    total_bytes = 0
    for raw, record in files.items():
        path = policy.path(raw, scope=scope)
        observed = sha256_file(path)
        _require(observed == _record_hash(record), f"artifact hash drift: {raw}")
        size = path.stat().st_size
        if isinstance(record, Mapping) and "bytes" in record:
            _require(int(record["bytes"]) == size, f"artifact byte-size drift: {raw}")
        inode = (path.stat().st_dev, path.stat().st_ino)
        _require(inode not in seen_inodes, f"hard-link alias: {raw} and {seen_inodes.get(inode)}")
        seen_inodes[inode] = str(raw)
        total_bytes += size
    return {"file_count": len(files), "total_bytes": total_bytes}


def _manifest_files(value: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("files", "sealed_files", "artifact_hashes"):
        files = value.get(key)
        if isinstance(files, Mapping):
            return files
    raise VerificationError("seal/manifest lacks a file-hash mapping")


def verify_terminal_manifest(
    contract: Mapping[str, Any], policy: PathPolicy
) -> dict[str, Any]:
    manifest_path = policy.path(contract["paths"]["terminal_manifest"], scope="attempt")
    manifest = read_json(manifest_path)
    _require(manifest.get("schema_version") == 1, "terminal manifest schema drift")
    _require(manifest.get("attempt") in (None, ATTEMPT), "terminal manifest attempt drift")
    _require(manifest.get("passed", True) is True, "terminal manifest did not pass")
    files = _manifest_files(manifest)
    summary = verify_file_map(files, policy, scope="attempt")
    exclusions_raw = contract.get("terminal_manifest_exclusions", [])
    _require(isinstance(exclusions_raw, list), "terminal manifest exclusions must be a list")
    exclusions = {_canonical_relative(value) for value in exclusions_raw}
    required_exclusions = {
        contract["paths"]["terminal_manifest"],
        f"{contract['attempt_root']}/audit/independent_verification.json",
    }
    _require(exclusions == required_exclusions, "terminal manifest exclusions are not the exact two frozen paths")
    for excluded in exclusions:
        policy.path(excluded, scope="attempt", must_exist=False)
    actual: set[str] = set()
    for root, directories, filenames in os.walk(policy.attempt_root, followlinks=False):
        root_path = Path(root)
        for directory in tuple(directories):
            candidate = root_path / directory
            _require(not candidate.is_symlink(), f"symlink directory in attempt: {candidate}")
        for filename in filenames:
            candidate = root_path / filename
            _require(not candidate.is_symlink(), f"symlink file in attempt: {candidate}")
            relative = candidate.relative_to(policy.repository_root).as_posix()
            if relative not in exclusions:
                actual.add(relative)
    expected = set(files)
    _require(actual == expected, f"terminal path set drift: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    return summary | {"path_set_complete": True, "excluded_paths": sorted(exclusions)}


def _outcome_counts(value: Mapping[str, Any]) -> dict[str, Any]:
    nested = value.get("outcome_counts", value.get("outcome_counts_at_seal"))
    controller = value.get("controller_chronology")
    if not isinstance(nested, Mapping) and isinstance(controller, Mapping):
        nested = controller.get("outcome_counts")
    source = nested if isinstance(nested, Mapping) else value
    aliases = {
        "fit": ("fit", "fit_outcome_episodes"),
        "selection": ("selection", "selection_outcome_episodes"),
        "smoke": ("smoke", "smoke_outcome_episodes"),
        "confirmation_generated": (
            "confirmation_generated",
            "confirmation_outcome_episodes_generated",
            "confirmation_outcome_episodes",
        ),
        "confirmation_executed": (
            "confirmation_executed",
            "confirmation_outcome_episodes_executed",
        ),
        "confirmation_opened": (
            "confirmation_opened",
            "confirmation_outcomes_opened_for_analysis",
        ),
    }
    result: dict[str, Any] = {}
    for name, names in aliases.items():
        for alias in names:
            if alias in source:
                result[name] = source[alias]
                break
    if "confirmation_opened" not in result and isinstance(controller, Mapping):
        if "confirmation_outcomes_opened_for_analysis" in controller:
            result["confirmation_opened"] = controller["confirmation_outcomes_opened_for_analysis"]
    return result


def verify_seal(
    path: Path,
    policy: PathPolicy,
    *,
    checkpoint_state: str,
    expected_counts: Mapping[str, Any],
    expected_attempt: str = ACTIVE_ATTEMPT,
    expected_pre_data_seal_sha256: str | None = None,
) -> dict[str, Any]:
    seal = read_json(path)
    _require(seal.get("schema_version") == 1, f"{checkpoint_state} seal schema drift")
    _require(seal.get("attempt") == expected_attempt, f"{checkpoint_state} seal attempt drift")
    _require(seal.get("checkpoint_state") == checkpoint_state, f"{checkpoint_state} seal state drift")
    _require(seal.get("passed") is True, f"{checkpoint_state} seal did not pass")
    created = int(seal.get("created_unix_ns", 0))
    _require(created > 0, f"{checkpoint_state} seal lacks chronology timestamp")
    observed = _outcome_counts(seal)
    for key, expected in expected_counts.items():
        _require(key in observed, f"{checkpoint_state} seal lacks count {key}")
        _require(observed[key] == expected, f"{checkpoint_state} illegal count {key}")
    if checkpoint_state == "PRE_SELECTION_SEAL":
        _require(
            isinstance(expected_pre_data_seal_sha256, str)
            and len(expected_pre_data_seal_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in expected_pre_data_seal_sha256
            ),
            "PRE_SELECTION_SEAL expected inheritance hash is invalid",
        )
        _require(
            seal.get("pre_data_seal_sha256")
            == expected_pre_data_seal_sha256,
            "PRE_SELECTION_SEAL direct inheritance cross-link drift",
        )
    else:
        _require(
            expected_pre_data_seal_sha256 is None,
            f"{checkpoint_state} received an inapplicable inheritance hash",
        )
    files = _manifest_files(seal)
    file_summary = verify_file_map(files, policy, scope="repository")
    return {"created_unix_ns": created, "counts": observed, **file_summary, "object": seal}


def verify_normative_contracts(contract: Mapping[str, Any], policy: PathPolicy) -> dict[str, Any]:
    paths = contract["paths"]
    dgp = read_json(policy.path(paths["dgp_matrix"]))
    grid = read_json(policy.path(paths["candidate_grid"]))
    power = read_json(policy.path(paths["power_rule"]))
    mapping = read_json(policy.path(paths["outcome_mapping"]))
    _require(tuple(dgp.get("regime_order", ())) == DGP_ORDER, "DGP order drift")
    _require(dgp.get("additional_dgps_permitted") is False, "additional DGPs became legal")
    _require(dgp.get("sequential_confirmation_expansion") is False, "sequential expansion became legal")
    roles = dgp.get("roles_per_regime", {})
    _require(int(roles["fit"]["episodes"]) == 300, "fit size drift")
    _require(int(roles["selection"]["episodes"]) == 500, "selection size drift")
    _require(int(roles["smoke"]["episodes"]) == 6, "smoke size drift")
    _require(int(roles["confirmation_maximum_assigned"]["episodes"]) == 4500, "confirmation maximum drift")
    _require(grid.get("architectures") is not None, "candidate architectures absent")
    architecture_records = {item["id"]: item for item in grid["architectures"]}
    _require(tuple(architecture_records) == ARCHITECTURES, "architecture order drift")
    for architecture, heads in ARCHITECTURE_HEADS.items():
        record = architecture_records[architecture]
        _require(int(record["head_count_per_stage"]) == heads, "candidate head count drift")
        _require(int(record["gate_flops_per_reached_decision"]) == GATE_FLOPS[heads], "gate FLOP drift")
        _require(int(record["nonflop_operations_per_reached_decision"]) == GATE_NONFLOPS[heads], "gate non-FLOP drift")
    _require(tuple(float(value) for value in grid["ridge_penalties"]) == RIDGES, "ridge grid drift")
    _require(tuple(float(value) for value in grid["sequential_fit_score_quantiles"]) == FIT_QUANTILES, "quantile grid drift")
    _require(int(grid["candidate_count"]) == CANDIDATE_COUNT, "candidate count drift")
    _require(int(grid["feature_width"]) == FEATURE_DIM, "feature width drift")
    _require(grid.get("fit_lock_before_selection_open") is True, "fit lock rule drift")
    _require(grid.get("selected_head_refit") is False, "refit rule drift")
    _require(int(power["family_size"]) == FAMILY_SIZE, "power family drift")
    _require(float(power["per_claim_alpha"]) == PER_CLAIM_ALPHA, "power alpha drift")
    _require(tuple(power["common_episode_grid_per_regime"]) == tuple(range(500, 4501, 500)), "power grid drift")
    _require(power.get("sequential_expansion") is False, "power expansion drift")
    _require(tuple(mapping["regime_order"]) == DGP_ORDER, "mapping DGP order drift")
    _require(tuple(mapping["co_primary_endpoints"]) == PRIMARY_CONTRASTS, "mapping endpoints drift")
    _require(int(mapping["family_size"]) == FAMILY_SIZE, "mapping family drift")
    _require(mapping["bootstrap"]["paired_replicates"] == BOOTSTRAP_REPLICATES, "bootstrap replicate drift")
    _require(mapping["bootstrap"]["independent_elementwise_reproduction_required"] is True, "bootstrap reproduction weakened")
    _require(mapping.get("no_retry_after_confirmation_open") is True, "no-retry rule drift")
    hashes = {
        name: sha256_file(policy.path(paths[name]))
        for name in ("dgp_matrix", "candidate_grid", "power_rule", "outcome_mapping")
    }
    for raw, expected in contract.get("expected_hashes", {}).items():
        observed = sha256_file(policy.path(raw))
        _require(observed == expected, f"direct contract hash drift: {raw}")
    return hashes


def _verify_durable_controller_adapter_marker(
    state: Mapping[str, Any],
    *,
    ledger_event_count: int,
    ledger_head_sha256: str,
    policy: PathPolicy,
) -> dict[str, Any]:
    """Independently bind STATE to the receipt-bound v008 controller adapter."""

    marker = state.get("v008_durable_controller_adapter")
    expected_keys = {
        "schema_version",
        "authorization_kind",
        "adapter_source_path",
        "adapter_source_sha256",
        "adapter_source_ast_sha256",
        "root_program_path",
        "root_program_sha256",
        "root_program_ast_sha256",
        "ledger_event_count",
        "ledger_head_sha256",
        "operation_sha256",
        "state_binding_sha256",
    }

    def source_identity(relative: str, label: str) -> dict[str, str]:
        path = policy.path(relative, scope="study")
        metadata = path.lstat()
        _require(
            stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and int(metadata.st_nlink) == 1,
            f"{label} is linked, aliased, or non-regular",
        )
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            raise VerificationError(f"cannot parse {label}") from error
        ast_sha256 = hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode("utf-8")
        ).hexdigest()
        return {
            "path": relative,
            "sha256": sha256_file(path),
            "ast_sha256": ast_sha256,
        }

    adapter = source_identity(
        f"{policy.attempt_relative}/version_forward_transaction.py",
        "v008 durable controller adapter source",
    )
    root_program = source_identity(
        f"{policy.study_relative}/program.py",
        "root durable controller source",
    )
    head_is_sha256 = (
        isinstance(ledger_head_sha256, str)
        and len(ledger_head_sha256) == 64
        and all(character in "0123456789abcdef" for character in ledger_head_sha256)
    )
    marker_without_binding = dict(marker) if isinstance(marker, Mapping) else {}
    state_binding = marker_without_binding.pop("state_binding_sha256", None)
    state_without_marker = dict(state)
    state_without_marker.pop("v008_durable_controller_adapter", None)
    operation = marker.get("operation_sha256") if isinstance(marker, Mapping) else None
    operation_is_sha256 = (
        isinstance(operation, str)
        and len(operation) == 64
        and all(character in "0123456789abcdef" for character in operation)
    )
    checks = {
        "schema": isinstance(marker, Mapping) and set(marker) == expected_keys,
        "version": isinstance(marker, Mapping)
        and type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 2,
        "authorization": isinstance(marker, Mapping)
        and marker.get("authorization_kind")
        == "receipt_bound_v008_root_controller_adapter",
        "adapter_path": isinstance(marker, Mapping)
        and marker.get("adapter_source_path") == adapter["path"],
        "adapter_hash": isinstance(marker, Mapping)
        and marker.get("adapter_source_sha256") == adapter["sha256"],
        "adapter_ast": isinstance(marker, Mapping)
        and marker.get("adapter_source_ast_sha256") == adapter["ast_sha256"],
        "root_path": isinstance(marker, Mapping)
        and marker.get("root_program_path") == root_program["path"],
        "root_hash": isinstance(marker, Mapping)
        and marker.get("root_program_sha256") == root_program["sha256"],
        "root_ast": isinstance(marker, Mapping)
        and marker.get("root_program_ast_sha256") == root_program["ast_sha256"],
        "ledger_count": isinstance(marker, Mapping)
        and type(ledger_event_count) is int
        and ledger_event_count > 0
        and type(marker.get("ledger_event_count")) is int
        and type(state.get("ledger_event_count")) is int
        and marker.get("ledger_event_count")
        == state.get("ledger_event_count")
        == ledger_event_count,
        "ledger_head": isinstance(marker, Mapping)
        and head_is_sha256
        and marker.get("ledger_head_sha256")
        == state.get("ledger_head_sha256")
        == ledger_head_sha256,
        "operation": operation_is_sha256,
        "state_binding": isinstance(state_binding, str)
        and len(state_binding) == 64
        and all(character in "0123456789abcdef" for character in state_binding)
        and state_binding
        == canonical_object_sha256(
            {
                "marker_without_state_binding": marker_without_binding,
                "state_without_marker": state_without_marker,
            }
        ),
    }
    _require(
        all(checks.values()),
        f"v008 durable-controller adapter marker drift: {checks}",
    )
    return {
        "authorization_kind": marker["authorization_kind"],
        "adapter_source_path": adapter["path"],
        "adapter_source_sha256": adapter["sha256"],
        "adapter_source_ast_sha256": adapter["ast_sha256"],
        "root_program_path": root_program["path"],
        "root_program_sha256": root_program["sha256"],
        "root_program_ast_sha256": root_program["ast_sha256"],
        "ledger_event_count": ledger_event_count,
        "ledger_head_sha256": ledger_head_sha256,
        "operation_sha256": operation,
        "state_binding_sha256": state_binding,
        "passed": True,
    }


def _verify_descendant_adapter_transaction_chain(
    *,
    post_state: Mapping[str, Any],
    post_ledger: Mapping[str, Any],
    current_state: Mapping[str, Any],
    current_events: Sequence[Mapping[str, Any]],
    ledger_payload: bytes,
    policy: PathPolicy,
) -> dict[str, Any]:
    """Separately replay every post-receipt v008 adapter transaction group."""

    def deep_copy(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise VerificationError(
                "independent descendant transaction contains non-JSON data"
            ) from error

    def compact_bytes(value: Any) -> bytes:
        try:
            return json.dumps(
                value, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise VerificationError(
                "independent descendant transaction is noncanonical"
            ) from error

    def is_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    early_routing = {
        "no_candidate": (
            "CANDIDATE_SELECTION", "selection/selection_ledger.json",
            "verifier_contract_no_candidate.json",
        ),
        "power_infeasible": (
            "CONFIRMATION_POWER_AND_COHORT_FREEZE", "power_analysis.json",
            "verifier_contract_power_infeasible.json",
        ),
    }
    failure_routing = {
        "SEALED_ANALYSIS": (
            "analysis_execution", "audit/analysis_execution_invalid.json",
            ["analysis_execution_invalid", "analysis_execution", "analysis_failure"],
        ),
        "LATENCY_AND_RESOURCE_REPORTING": (
            "latency_resource", "metrics/latency_and_resources.json",
            ["latency_resource", "latency_and_resources", "latency_report"],
        ),
    }
    active_root = (
        policy.repository_root
        / "runs/lewm_domain_robust_gate/attempts"
        / ACTIVE_ATTEMPT
    ).resolve()

    def active_relative_file(relative: Any, label: str) -> Path:
        _require(
            isinstance(relative, str) and bool(relative),
            f"independent descendant {label} path type drift",
        )
        try:
            resolved = (policy.repository_root / relative).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise VerificationError(
                f"independent descendant {label} path is unreadable"
            ) from error
        _require(
            resolved.is_relative_to(active_root)
            and resolved.relative_to(policy.repository_root.resolve()).as_posix()
            == relative,
            f"independent descendant {label} path drift",
        )
        return resolved

    def active_json_object(relative: Any, label: str) -> tuple[Path, dict[str, Any]]:
        path = active_relative_file(relative, label)
        try:
            value = read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError(
                f"independent descendant {label} is unreadable"
            ) from error
        return path, value

    def proves_integrity_failure(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered in {
                    "execution_invalid", "integrity_failure", "integrity_failed"
                } and child is True:
                    return True
                if lowered in {
                    "integrity_valid", "integrity_passed", "process_valid",
                    "all_integrity_checks_passed",
                } and child is False:
                    return True
                if proves_integrity_failure(child):
                    return True
        elif isinstance(value, list):
            return any(proves_integrity_failure(item) for item in value)
        return False

    def linked_sha256(value: Any) -> Any:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return value.get("sha256")
        return None

    def verify_controller_transition(
        base_state: Mapping[str, Any],
        proposed_state: Mapping[str, Any],
        controller_events: Sequence[Mapping[str, Any]],
    ) -> None:
        base = deep_copy(base_state)
        base_marker = base.pop("v008_durable_controller_adapter", None)
        proposed = deep_copy(proposed_state)
        sequence = [event.get("event") for event in controller_events]
        _require(
            controller_events
            and all(
                type(event.get("created_unix_ns")) is int
                and event.get("created_unix_ns") > 0
                and event.get("attempt") == ACTIVE_ATTEMPT
                for event in controller_events
            ),
            "independent descendant controller event identity/type drift",
        )
        if sequence == ["outcome_counts_updated"]:
            event = controller_events[0]
            fields = event.get("fields")
            allowed = {
                "fit_outcome_episodes", "selection_outcome_episodes",
                "smoke_outcome_episodes",
                "confirmation_outcome_episodes_generated",
                "confirmation_outcome_episodes_executed",
                "confirmation_outcomes_opened_for_analysis",
            }
            _require(
                set(event)
                == {
                    "event", "attempt", "fields",
                    "prior_controller_adapter_marker", "created_unix_ns",
                }
                and isinstance(fields, Mapping)
                and len(fields) == 1
                and set(fields).issubset(allowed)
                and compact_bytes(event.get("prior_controller_adapter_marker"))
                == compact_bytes(base_marker),
                "independent descendant count event schema drift",
            )
            field, value = next(iter(fields.items()))
            _require(
                (
                    type(value) is bool
                    if field == "confirmation_outcomes_opened_for_analysis"
                    else type(value) is int and value >= 0
                ),
                "independent descendant count value type drift",
            )
            allowed_states = {
                "fit_outcome_episodes": {"FIT_COHORTS"},
                "selection_outcome_episodes": {"SELECTION_COHORTS"},
                "smoke_outcome_episodes": {"EXCLUDED_MECHANICAL_SMOKE"},
                "confirmation_outcome_episodes_generated": {
                    "CONFIRMATION_GENERATION", "CONFIRMATION_EXECUTION"
                },
                "confirmation_outcome_episodes_executed": {
                    "CONFIRMATION_EXECUTION", "CONFIRMATION_INPUT_SEAL"
                },
                "confirmation_outcomes_opened_for_analysis": {"SEALED_ANALYSIS"},
            }
            expected_keys = {
                "fit_outcome_episodes": "expected_fit_episode_count",
                "selection_outcome_episodes": "expected_selection_episode_count",
                "smoke_outcome_episodes": "expected_smoke_episode_count",
                "confirmation_outcome_episodes_generated": (
                    "expected_confirmation_episode_count"
                ),
                "confirmation_outcome_episodes_executed": (
                    "expected_confirmation_episode_count"
                ),
            }
            old = base.get(field)
            expected_key = expected_keys.get(field)
            _require(
                base.get("early_scientific_failure") is None
                and base.get("terminal_label") is None
                and base.get("current_state") in allowed_states[field]
                and (
                    (
                        type(value) is bool
                        and type(old) is bool
                        and not (old and not value)
                    )
                    or (type(value) is int and type(old) is int and value >= old)
                )
                and (
                    expected_key is None
                    or expected_key not in base
                    or (
                        type(base.get(expected_key)) is int
                        and type(value) is int
                        and value <= base[expected_key]
                    )
                ),
                "independent descendant count monotonicity/state/maximum drift",
            )
            expected = deep_copy(base)
            expected[field] = value
            expected["updated_unix_ns"] = event["created_unix_ns"]
            _require(
                compact_bytes(proposed) == compact_bytes(expected),
                "independent descendant count transition drift",
            )
            return
        if sequence == ["scientific_terminal_decision_recorded"]:
            event = controller_events[0]
            _require(
                set(event)
                == {
                    "event", "attempt", "terminal_label", "process_valid",
                    "decision_path", "decision_sha256", "created_unix_ns",
                }
                and base.get("current_state") == "TERMINAL"
                and base.get("terminal_label") is None
                and event.get("terminal_label")
                in {
                    "domain_robust_gate_confirmed",
                    "domain_robust_gate_partial", "domain_robust_gate_failed",
                }
                and type(event.get("process_valid")) is bool
                and isinstance(event.get("decision_path"), str)
                and is_sha256(event.get("decision_sha256")),
                "independent descendant terminal event schema drift",
            )
            decision, decision_value = active_json_object(
                event["decision_path"], "terminal decision"
            )
            _require(
                sha256_file(decision) == event["decision_sha256"]
                and decision_value.get("attempt") == ACTIVE_ATTEMPT
                and decision_value.get("terminal_label")
                == event["terminal_label"]
                and type(decision_value.get("process_valid")) is bool
                and decision_value.get("process_valid")
                is event["process_valid"],
                "independent descendant terminal evidence drift",
            )
            expected = deep_copy(base)
            expected.update(
                {
                    "terminal_label": event["terminal_label"],
                    "process_valid": event["process_valid"],
                    "scientific_terminal": True,
                    "confirmation_terminal": True,
                    "terminal_decision_path": event["decision_path"],
                    "terminal_decision_sha256": event["decision_sha256"],
                    "updated_unix_ns": event["created_unix_ns"],
                }
            )
            _require(
                compact_bytes(proposed) == compact_bytes(expected),
                "independent descendant terminal transition drift",
            )
            return
        if sequence == ["state_completed"]:
            event = controller_events[0]
            ordinary = {
                "event", "attempt", "completed_state", "next_state",
                "checkpoint_name", "evidence_path", "evidence_sha256",
                "created_unix_ns",
            }
            integrity = ordinary | {
                "postconfirmation_integrity_failure", "integrity_source",
                "skipped_states",
            }
            _require(
                set(event) in (ordinary, integrity)
                and event.get("completed_state") == base.get("current_state")
                and isinstance(event.get("checkpoint_name"), str)
                and isinstance(event.get("evidence_path"), str)
                and is_sha256(event.get("evidence_sha256")),
                "independent descendant state-completed schema drift",
            )
            evidence_file = active_relative_file(
                event["evidence_path"], "state-completed evidence"
            )
            _require(
                sha256_file(evidence_file) == event["evidence_sha256"],
                "independent descendant state-completed evidence drift",
            )
            try:
                evidence_value = json.loads(evidence_file.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise VerificationError(
                    "independent descendant state-completed evidence is unreadable"
                ) from error
            _require(
                isinstance(evidence_value, dict),
                "independent descendant state-completed evidence is not an object",
            )
            before_completed = base.get("completed_states")
            before_checkpoints = base.get("verified_checkpoints")
            _require(
                isinstance(before_completed, list)
                and isinstance(before_checkpoints, list),
                "independent descendant state-completed history drift",
            )
            completed = str(event["completed_state"])
            timestamp = int(event["created_unix_ns"])
            exact = deep_copy(base)
            if set(event) == ordinary:
                _require(
                    completed in STATE_MACHINE,
                    "independent descendant ordinary state-machine drift",
                )
                position = STATE_MACHINE.index(completed)
                is_final = completed == "POST_TERMINAL_REPORTING"
                following = None if is_final else STATE_MACHINE[position + 1]
                _require(
                    event.get("next_state") == following
                    and evidence_value.get("passed") is True
                    and evidence_value.get("attempt") == ACTIVE_ATTEMPT
                    and evidence_value.get("checkpoint_state") == completed
                    and not (
                        completed == "INDEPENDENT_VERIFICATION"
                        and evidence_value.get("terminal_label")
                        == "domain_robust_gate_execution_invalid"
                    )
                    and isinstance(proposed.get("next_action"), str)
                    and not (
                        base.get("early_scientific_failure") is not None
                        and not is_final
                    )
                    and not (
                        base.get("postconfirmation_integrity_failure") is not None
                        and not is_final
                    )
                    and not (
                        base.get("terminal_label") is not None
                        and completed not in {"TERMINAL", "POST_TERMINAL_REPORTING"}
                    ),
                    "independent descendant ordinary state authorization drift",
                )
                checkpoint_value = {
                    "name": event["checkpoint_name"],
                    "created_unix_ns": timestamp,
                    "evidence_path": event["evidence_path"],
                    "evidence_sha256": event["evidence_sha256"],
                    "source_attempt": ACTIVE_ATTEMPT,
                    "verification_lineage": "direct_checkpoint",
                }
                exact["completed_states"] = before_completed + [completed]
                exact["current_state"] = (
                    "POST_TERMINAL_REPORTING" if is_final else following
                )
                exact["last_verified_checkpoint"] = checkpoint_value
                exact["verified_checkpoints"] = before_checkpoints + [
                    checkpoint_value
                ]
                exact["next_action"] = proposed["next_action"]
                if is_final:
                    _require(
                        base.get("terminal_label") is not None
                        and base.get("process_valid") is not None
                        and base.get("post_terminal_reporting_complete") is not True
                        and proposed.get("next_action")
                        == "program_complete_no_further_scientific_or_reporting_transition",
                        "independent descendant post-terminal completion drift",
                    )
                    exact["post_terminal_reporting_complete"] = True
                if completed == "PREREGISTRATION_AND_POWER":
                    declared = evidence_value.get("registered_counts")
                    required_counts = {
                        "expected_fit_episode_count": 1200,
                        "expected_selection_episode_count": 2000,
                        "expected_smoke_episode_count": 24,
                        "maximum_confirmation_episode_count": 18000,
                    }
                    _require(
                        declared == required_counts
                        and all(
                            type(declared[name]) is int for name in required_counts
                        ),
                        "independent descendant preregistered counts drift",
                    )
                    exact.update(required_counts)
                if completed == "CONFIRMATION_POWER_AND_COHORT_FREEZE":
                    each = evidence_value.get(
                        "fixed_confirmation_episodes_per_regime"
                    )
                    total = evidence_value.get("fixed_confirmation_episode_count")
                    _require(
                        type(each) is int
                        and each in range(500, 4501, 500)
                        and type(total) is int
                        and total == 4 * each,
                        "independent descendant confirmation-size drift",
                    )
                    exact["expected_confirmation_episodes_per_regime"] = each
                    exact["expected_confirmation_episode_count"] = total
            else:
                route = failure_routing.get(completed)
                source, suffix, audit_keys = (
                    route if route is not None else (None, None, None)
                )
                omitted = (
                    list(
                        STATE_MACHINE[
                            STATE_MACHINE.index(completed) + 1:
                            STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
                        ]
                    )
                    if route is not None
                    else None
                )
                exact_evidence = (
                    f"runs/lewm_domain_robust_gate/attempts/{ACTIVE_ATTEMPT}/{suffix}"
                    if suffix is not None
                    else None
                )
                _require(
                    route is not None
                    and event.get("next_state") == "INDEPENDENT_VERIFICATION"
                    and event.get("postconfirmation_integrity_failure") is True
                    and event.get("integrity_source") == source
                    and event.get("skipped_states") == omitted
                    and event.get("evidence_path") == exact_evidence
                    and event.get("checkpoint_name")
                    == f"{ACTIVE_ATTEMPT}_{source}_integrity_failure"
                    and evidence_value.get("attempt") == ACTIVE_ATTEMPT
                    and evidence_value.get("checkpoint_state") == completed
                    and evidence_value.get("passed") is False
                    and base.get("terminal_label") is None,
                    "independent descendant integrity staging drift",
                )
                expected_confirmation = base.get(
                    "expected_confirmation_episode_count"
                )
                _require(
                    type(expected_confirmation) is int
                    and expected_confirmation > 0
                    and type(base.get("expected_smoke_episode_count")) is int
                    and type(base.get("smoke_outcome_episodes")) is int
                    and base.get("smoke_outcome_episodes")
                    == base.get("expected_smoke_episode_count")
                    and type(
                        base.get("confirmation_outcome_episodes_generated")
                    )
                    is int
                    and base.get("confirmation_outcome_episodes_generated")
                    == expected_confirmation
                    and type(
                        base.get("confirmation_outcome_episodes_executed")
                    )
                    is int
                    and base.get("confirmation_outcome_episodes_executed")
                    == expected_confirmation
                    and base.get("confirmation_outcomes_opened_for_analysis")
                    is True
                    and proves_integrity_failure(evidence_value),
                    "independent descendant integrity fixed-open proof drift",
                )
                if completed == "SEALED_ANALYSIS":
                    analysis_result = active_root / "analysis_result.json"
                    _require(
                        type(
                            evidence_value.get(
                                "confirmation_outcome_episodes_generated"
                            )
                        )
                        is int
                        and evidence_value.get(
                            "confirmation_outcome_episodes_generated"
                        )
                        == expected_confirmation
                        and type(
                            evidence_value.get(
                                "confirmation_outcome_episodes_executed"
                            )
                        )
                        is int
                        and evidence_value.get(
                            "confirmation_outcome_episodes_executed"
                        )
                        == expected_confirmation
                        and evidence_value.get(
                            "confirmation_outcomes_opened_for_analysis"
                        )
                        is True
                        and type(evidence_value.get("analysis_result_present"))
                        is bool
                        and evidence_value.get("analysis_result_present")
                        is analysis_result.is_file()
                        and isinstance(evidence_value.get("error_type"), str)
                        and bool(evidence_value["error_type"])
                        and isinstance(evidence_value.get("error"), str)
                        and bool(evidence_value["error"])
                        and evidence_value.get("scientific_objects_changed")
                        is False,
                        "independent descendant analysis-invalid evidence drift",
                    )
                checkpoint_value = {
                    "name": event["checkpoint_name"],
                    "created_unix_ns": timestamp,
                    "evidence_path": event["evidence_path"],
                    "evidence_sha256": event["evidence_sha256"],
                    "source_attempt": ACTIVE_ATTEMPT,
                    "verification_lineage": (
                        "direct_postconfirmation_integrity_checkpoint"
                    ),
                }
                exact["completed_states"] = before_completed + [completed]
                exact["verified_checkpoints"] = before_checkpoints + [
                    checkpoint_value
                ]
                exact["last_verified_checkpoint"] = checkpoint_value
                exact["current_state"] = "INDEPENDENT_VERIFICATION"
                exact["postconfirmation_integrity_failure"] = {
                    "status": "awaiting_independent_verification",
                    "source": source,
                    "trigger_state": completed,
                    "source_path": event["evidence_path"],
                    "source_sha256": event["evidence_sha256"],
                    "audit_hash_keys": audit_keys,
                    "skipped_states": omitted,
                    "staged_unix_ns": timestamp,
                }
                exact["skipped_states"] = [
                    {
                        "state": item,
                        "source": source,
                        "reason": "postconfirmation_integrity_failure_short_circuit",
                        "source_path": event["evidence_path"],
                        "source_sha256": event["evidence_sha256"],
                        "recorded_unix_ns": timestamp,
                    }
                    for item in omitted
                ]
                exact["next_action"] = (
                    "run the read-only confirmation-mode independent verifier and "
                    "capture the execution-invalid audit; do not retry or alter "
                    "confirmation evidence"
                )
            exact["updated_unix_ns"] = timestamp
            _require(
                compact_bytes(proposed) == compact_bytes(exact),
                "independent descendant state-completed unrelated STATE drift",
            )
            return
        if sequence == ["preregistered_early_scientific_failure_staged"]:
            event = controller_events[0]
            early_keys = {
                "event", "attempt", "mode", "trigger_state",
                "trigger_evidence_path", "trigger_evidence_sha256",
                "verifier_contract_path", "verifier_contract_sha256",
                "skipped_states", "smoke_outcome_episodes",
                "confirmation_outcome_episodes_generated",
                "confirmation_outcome_episodes_executed",
                "confirmation_outcomes_opened_for_analysis",
                "created_unix_ns",
            }
            route = early_routing.get(event.get("mode"))
            trigger, source_suffix, contract_suffix = (
                route if route is not None else (None, None, None)
            )
            omitted = (
                list(
                    STATE_MACHINE[
                        STATE_MACHINE.index(str(trigger)) + 1:
                        STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
                    ]
                )
                if trigger in STATE_MACHINE
                else None
            )
            source_relative = (
                f"runs/lewm_domain_robust_gate/attempts/{ACTIVE_ATTEMPT}/{source_suffix}"
                if source_suffix is not None
                else None
            )
            contract_relative = (
                f"runs/lewm_domain_robust_gate/attempts/{ACTIVE_ATTEMPT}/{contract_suffix}"
                if contract_suffix is not None
                else None
            )
            _require(
                set(event) == early_keys
                and route is not None
                and event.get("trigger_state") == trigger == base.get("current_state")
                and event.get("trigger_evidence_path") == source_relative
                and event.get("verifier_contract_path") == contract_relative
                and event.get("skipped_states") == omitted
                and is_sha256(event.get("trigger_evidence_sha256"))
                and is_sha256(event.get("verifier_contract_sha256"))
                and type(event.get("smoke_outcome_episodes")) is int
                and event.get("smoke_outcome_episodes") == 0
                and type(event.get("confirmation_outcome_episodes_generated")) is int
                and event.get("confirmation_outcome_episodes_generated") == 0
                and type(event.get("confirmation_outcome_episodes_executed")) is int
                and event.get("confirmation_outcome_episodes_executed") == 0
                and event.get("confirmation_outcomes_opened_for_analysis") is False
                and base.get("early_scientific_failure") is None
                and base.get("terminal_label") is None
                and type(base.get("smoke_outcome_episodes")) is int
                and base.get("smoke_outcome_episodes") == 0
                and type(
                    base.get("confirmation_outcome_episodes_generated")
                )
                is int
                and base.get("confirmation_outcome_episodes_generated") == 0
                and type(
                    base.get("confirmation_outcome_episodes_executed")
                )
                is int
                and base.get("confirmation_outcome_episodes_executed") == 0
                and base.get("confirmation_outcomes_opened_for_analysis") is False,
                "independent descendant early-failure transition drift",
            )
            source_file, source_value = active_json_object(
                source_relative, "early-failure source"
            )
            contract_file, contract_value = active_json_object(
                contract_relative, "early-failure verifier contract"
            )
            contract_path_key = (
                "selection_ledger" if event["mode"] == "no_candidate"
                else "power_freeze"
            )
            _require(
                sha256_file(source_file) == event["trigger_evidence_sha256"]
                and sha256_file(contract_file)
                == event["verifier_contract_sha256"]
                and type(contract_value.get("schema_version")) is int
                and contract_value.get("schema_version") == 1
                and contract_value.get("attempt") == ACTIVE_ATTEMPT
                and contract_value.get("attempt_root")
                == ATTEMPT_ROOTS[ACTIVE_ATTEMPT]
                and contract_value.get("mode") == event["mode"]
                and isinstance(contract_value.get("paths"), Mapping)
                and contract_value["paths"].get(contract_path_key)
                == source_relative,
                "independent descendant early-failure evidence drift",
            )
            if event["mode"] == "no_candidate":
                source_semantics = (
                    source_value.get("status")
                    == "selection_complete_no_selected_head_refit"
                    and type(source_value.get("candidate_count")) is int
                    and source_value.get("candidate_count") == 24
                    and type(source_value.get("eligible_count")) is int
                    and source_value.get("eligible_count") == 0
                    and source_value.get("selected_candidate_id") is None
                    and source_value.get("selected_candidate_index") is None
                    and source_value.get("selected_head_refit_after_selection")
                    is False
                    and type(
                        source_value.get(
                            "prior_confirmation_outcome_episodes_used", 0
                        )
                    )
                    is int
                    and source_value.get(
                        "prior_confirmation_outcome_episodes_used", 0
                    )
                    == 0
                )
            else:
                source_semantics = (
                    source_value.get("status")
                    == "binding_post_selection_power_result"
                    and source_value.get("decision")
                    == "power_infeasible_no_confirmation"
                    and source_value.get("feasible") is False
                    and source_value.get("passed") is False
                    and source_value.get(
                        "confirmation_generation_authorized_by_power"
                    )
                    is False
                    and source_value.get(
                        "selected_confirmation_episodes_per_regime"
                    )
                    is None
                    and source_value.get("confirmation_episode_count_per_regime")
                    is None
                    and source_value.get("terminal_label_if_infeasible")
                    == "domain_robust_gate_failed"
                    and type(
                        source_value.get("fresh_confirmation_outcomes_opened")
                    )
                    is int
                    and source_value.get("fresh_confirmation_outcomes_opened")
                    == 0
                )
            _require(
                source_semantics,
                "independent descendant early-failure source eligibility drift",
            )
            _require(
                type(base.get("expected_fit_episode_count")) is int
                and type(base.get("fit_outcome_episodes")) is int
                and base.get("fit_outcome_episodes")
                == base.get("expected_fit_episode_count")
                and type(base.get("expected_selection_episode_count")) is int
                and type(base.get("selection_outcome_episodes")) is int
                and base.get("selection_outcome_episodes")
                == base.get("expected_selection_episode_count"),
                "independent descendant early-failure role completeness drift",
            )
            forbidden_roots = (
                "data/smoke", "data/confirmation", "execution/smoke",
                "execution/confirmation", "metrics/smoke",
                "metrics/confirmation",
            )
            forbidden_files = (
                "audit/pre_confirmation_manifest.json",
                "audit/pre_confirmation_package_seal.json",
                "audit/excluded_mechanical_smoke.json",
                "audit/confirmation_generation.json",
                "audit/confirmation_execution.json",
                "audit/confirmation_input_seal.json", "analysis_result.json",
                "bootstrap_replicates.npz", "bootstrap_summary.json",
            )
            _require(
                all(
                    not root.exists()
                    or not any(path.is_file() for path in root.rglob("*"))
                    for root in (active_root / relative for relative in forbidden_roots)
                )
                and all(
                    not (active_root / relative).exists()
                    for relative in forbidden_files
                ),
                "independent descendant early-failure later-role artifact drift",
            )
            timestamp = int(event["created_unix_ns"])
            exact = deep_copy(base)
            exact["early_scientific_failure"] = {
                "mode": event["mode"],
                "status": "awaiting_independent_verification",
                "trigger_state": trigger,
                "trigger_evidence_path": source_relative,
                "trigger_evidence_sha256": event["trigger_evidence_sha256"],
                "verifier_contract_path": contract_relative,
                "verifier_contract_sha256": event["verifier_contract_sha256"],
                "skipped_states": omitted,
                "staged_unix_ns": timestamp,
                "required_terminal_label": "domain_robust_gate_failed",
                "required_process_valid": True,
            }
            exact["skipped_states"] = [
                {
                    "state": item,
                    "mode": event["mode"],
                    "reason": "preregistered_process_valid_early_scientific_failure",
                    "trigger_evidence_path": source_relative,
                    "trigger_evidence_sha256": event["trigger_evidence_sha256"],
                    "recorded_unix_ns": timestamp,
                }
                for item in omitted
            ]
            exact["current_state"] = "INDEPENDENT_VERIFICATION"
            exact["next_action"] = (
                "run the standalone read-only verifier in the exact staged mode and "
                "capture audit/independent_verification.json before finalizing"
            )
            exact["updated_unix_ns"] = timestamp
            _require(
                compact_bytes(proposed) == compact_bytes(exact),
                "independent descendant early-failure unrelated STATE drift",
            )
            return
        if sequence == [
            "state_completed", "scientific_terminal_decision_recorded",
            "state_completed",
        ]:
            first, decision, last = controller_events
            state_keys = {
                "event", "attempt", "completed_state", "next_state",
                "checkpoint_name", "evidence_path", "evidence_sha256",
                "created_unix_ns",
            }
            decision_keys = {
                "event", "attempt", "terminal_label", "process_valid",
                "decision_path", "decision_sha256", "created_unix_ns",
            }
            early = decision.get("terminal_label") == "domain_robust_gate_failed"
            first_keys = state_keys | (
                {"early_failure_mode"}
                if early
                else {"postconfirmation_integrity_failure"}
            )
            middle_keys = decision_keys | (
                {"confirmation_terminal", "early_failure_mode"}
                if early
                else set()
            )
            last_keys = state_keys | (
                {"early_failure_mode", "skipped_states"}
                if early
                else {"postconfirmation_integrity_failure"}
            )
            timestamp = first.get("created_unix_ns")
            _require(
                set(first) == first_keys
                and set(decision) == middle_keys
                and set(last) == last_keys
                and first.get("completed_state") == "INDEPENDENT_VERIFICATION"
                and first.get("next_state") == "TERMINAL"
                and last.get("completed_state") == "TERMINAL"
                and last.get("next_state") == "POST_TERMINAL_REPORTING"
                and timestamp == decision.get("created_unix_ns")
                == last.get("created_unix_ns")
                and base.get("current_state") == "INDEPENDENT_VERIFICATION"
                and base.get("terminal_label") is None
                and isinstance(first.get("checkpoint_name"), str)
                and isinstance(first.get("evidence_path"), str)
                and isinstance(last.get("checkpoint_name"), str)
                and isinstance(last.get("evidence_path"), str)
                and isinstance(decision.get("decision_path"), str)
                and is_sha256(first.get("evidence_sha256"))
                and is_sha256(last.get("evidence_sha256"))
                and decision.get("decision_path") == last.get("evidence_path")
                and decision.get("decision_sha256") == last.get("evidence_sha256")
                and (
                    (
                        early
                        and decision.get("process_valid") is True
                        and decision.get("confirmation_terminal") is False
                        and first.get("early_failure_mode")
                        == decision.get("early_failure_mode")
                        == last.get("early_failure_mode")
                    )
                    or (
                        not early
                        and decision.get("terminal_label")
                        == "domain_robust_gate_execution_invalid"
                        and decision.get("process_valid") is False
                        and first.get("postconfirmation_integrity_failure") is True
                        and last.get("postconfirmation_integrity_failure") is True
                    )
                ),
                "independent descendant three-event transition drift",
            )
            exact_audit_relative = (
                f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/audit/independent_verification.json"
            )
            exact_decision_relative = (
                f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/decision.json"
            )
            _require(
                first["evidence_path"] == exact_audit_relative
                and last["evidence_path"] == exact_decision_relative,
                "independent descendant three-event evidence path drift",
            )
            audit_file, audit_value = active_json_object(
                first["evidence_path"], "three-event independent audit"
            )
            decision_file, decision_value = active_json_object(
                last["evidence_path"], "three-event terminal decision"
            )
            _require(
                sha256_file(audit_file) == first["evidence_sha256"]
                and sha256_file(decision_file) == last["evidence_sha256"]
                and audit_value.get("attempt") == ACTIVE_ATTEMPT
                and decision_value.get("attempt") == ACTIVE_ATTEMPT
                and decision_value.get("terminal_label")
                == decision["terminal_label"]
                and type(decision_value.get("process_valid")) is bool
                and decision_value.get("process_valid")
                is decision["process_valid"],
                "independent descendant three-event evidence drift",
            )
            before_completed = base.get("completed_states")
            before_checkpoints = base.get("verified_checkpoints")
            _require(
                isinstance(before_completed, list)
                and isinstance(before_checkpoints, list),
                "independent descendant three-event history drift",
            )
            if early:
                mode = decision["early_failure_mode"]
                staged = base.get("early_scientific_failure")
                contract_file, contract_value = active_json_object(
                    staged.get("verifier_contract_path")
                    if isinstance(staged, Mapping)
                    else None,
                    "early-terminal verifier contract",
                )
                contract_record = audit_value.get("verifier_contract")
                source_hashes = audit_value.get("source_hashes")
                _require(
                    mode in early_routing
                    and isinstance(staged, Mapping)
                    and staged.get("mode") == mode
                    and staged.get("status") == "awaiting_independent_verification"
                    and base.get("postconfirmation_integrity_failure") is None
                    and audit_value.get("mode") == mode
                    and audit_value.get("passed") is True
                    and audit_value.get("terminal_label")
                    == "domain_robust_gate_failed"
                    and type(audit_value.get("schema_version")) is int
                    and audit_value.get("schema_version") == 1
                    and audit_value.get("read_only_verifier") is True
                    and isinstance(audit_value.get("checks"), Mapping)
                    and bool(audit_value["checks"])
                    and isinstance(contract_record, Mapping)
                    and contract_record.get("path")
                    == staged.get("verifier_contract_path")
                    and contract_record.get("sha256")
                    == sha256_file(contract_file)
                    and type(contract_value.get("schema_version")) is int
                    and contract_value.get("schema_version") == 1
                    and contract_value.get("attempt") == ACTIVE_ATTEMPT
                    and contract_value.get("mode") == mode
                    and isinstance(source_hashes, Mapping)
                    and linked_sha256(
                        source_hashes.get(
                            "selection_ledger"
                            if mode == "no_candidate"
                            else "power_freeze"
                        )
                    )
                    == staged.get("trigger_evidence_sha256")
                    and type(decision_value.get("schema_version")) is int
                    and decision_value.get("schema_version") == 1
                    and decision_value.get("passed") is True
                    and decision_value.get("checkpoint_state") == "TERMINAL"
                    and decision_value.get("early_failure_mode") == mode
                    and decision_value.get("trigger_evidence_path")
                    == staged.get("trigger_evidence_path")
                    and decision_value.get("trigger_evidence_sha256")
                    == staged.get("trigger_evidence_sha256")
                    and decision_value.get("independent_verification_path")
                    == first.get("evidence_path")
                    and decision_value.get("independent_verification_sha256")
                    == first.get("evidence_sha256")
                    and type(decision_value.get("smoke_outcome_episodes")) is int
                    and decision_value.get("smoke_outcome_episodes") == 0
                    and type(
                        decision_value.get(
                            "confirmation_outcome_episodes_generated"
                        )
                    )
                    is int
                    and decision_value.get(
                        "confirmation_outcome_episodes_generated"
                    )
                    == 0
                    and type(
                        decision_value.get(
                            "confirmation_outcome_episodes_executed"
                        )
                    )
                    is int
                    and decision_value.get(
                        "confirmation_outcome_episodes_executed"
                    )
                    == 0
                    and decision_value.get(
                        "confirmation_outcomes_opened_for_analysis"
                    )
                    is False
                    and last.get("skipped_states") == staged.get("skipped_states")
                    and first.get("checkpoint_name")
                    == f"{ACTIVE_ATTEMPT}_{mode}_independent_verification_passed"
                    and last.get("checkpoint_name")
                    == f"{ACTIVE_ATTEMPT}_{mode}_terminal_decision_recorded",
                    "independent descendant early terminal authorization drift",
                )
                lineage = "direct_early_scientific_failure_checkpoint"
            else:
                expected_confirmation = base.get(
                    "expected_confirmation_episode_count"
                )
                staged_integrity = base.get(
                    "postconfirmation_integrity_failure"
                )
                contract_file, contract_value = active_json_object(
                    f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/verifier_contract.json",
                    "execution-invalid verifier contract",
                )
                contract_record = audit_value.get("verifier_contract")
                capture_record = audit_value.get("capture")
                source_hashes = audit_value.get("source_hashes")
                _require(
                    first.get("checkpoint_name")
                    == f"{ACTIVE_ATTEMPT}_execution_invalid_independent_audit"
                    and last.get("checkpoint_name")
                    == f"{ACTIVE_ATTEMPT}_execution_invalid_terminal_decision",
                    "independent descendant execution-invalid checkpoint drift",
                )
                _require(
                    base.get("early_scientific_failure") is None
                    and type(expected_confirmation) is int
                    and expected_confirmation > 0
                    and type(base.get("expected_smoke_episode_count")) is int
                    and type(base.get("smoke_outcome_episodes")) is int
                    and base.get("smoke_outcome_episodes")
                    == base.get("expected_smoke_episode_count")
                    and type(
                        base.get("confirmation_outcome_episodes_generated")
                    )
                    is int
                    and base.get("confirmation_outcome_episodes_generated")
                    == expected_confirmation
                    and type(
                        base.get("confirmation_outcome_episodes_executed")
                    )
                    is int
                    and base.get("confirmation_outcome_episodes_executed")
                    == expected_confirmation
                    and base.get("confirmation_outcomes_opened_for_analysis")
                    is True
                    and type(audit_value.get("schema_version")) is int
                    and audit_value.get("schema_version") == 1
                    and audit_value.get("mode") == "confirmation"
                    and audit_value.get("read_only_verifier") is True
                    and audit_value.get("execution_invalid") is True
                    and audit_value.get("terminal_label")
                    == "domain_robust_gate_execution_invalid"
                    and proves_integrity_failure(audit_value)
                    and isinstance(contract_record, Mapping)
                    and contract_record.get("path")
                    == f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/verifier_contract.json"
                    and contract_record.get("sha256")
                    == sha256_file(contract_file)
                    and type(contract_value.get("schema_version")) is int
                    and contract_value.get("schema_version") == 1
                    and contract_value.get("attempt") == ACTIVE_ATTEMPT
                    and contract_value.get("attempt_root")
                    == ATTEMPT_ROOTS[ACTIVE_ATTEMPT]
                    and contract_value.get("mode") == "confirmation"
                    and isinstance(capture_record, Mapping)
                    and capture_record.get("captured_exclusively") is True
                    and capture_record.get("wrapper_integrity_passed") is True
                    and type(decision_value.get("schema_version")) is int
                    and decision_value.get("schema_version") == 1
                    and decision_value.get("passed") is True
                    and decision_value.get("checkpoint_state") == "TERMINAL"
                    and decision_value.get("terminal_basis")
                    == "postconfirmation_integrity_failure"
                    and decision_value.get("independent_verification_path")
                    == first.get("evidence_path")
                    and decision_value.get("independent_verification_sha256")
                    == first.get("evidence_sha256")
                    and type(
                        decision_value.get(
                            "confirmation_outcome_episodes_generated"
                        )
                    )
                    is int
                    and decision_value.get(
                        "confirmation_outcome_episodes_generated"
                    )
                    == expected_confirmation
                    and type(
                        decision_value.get(
                            "confirmation_outcome_episodes_executed"
                        )
                    )
                    is int
                    and decision_value.get(
                        "confirmation_outcome_episodes_executed"
                    )
                    == expected_confirmation
                    and decision_value.get(
                        "confirmation_outcomes_opened_for_analysis"
                    )
                    is True,
                    "independent descendant execution-invalid base/evidence drift",
                )
                if isinstance(staged_integrity, Mapping):
                    candidates = staged_integrity.get("audit_hash_keys")
                    _require(
                        audit_value.get("passed") is True
                        and isinstance(source_hashes, Mapping)
                        and isinstance(candidates, list)
                        and bool(candidates)
                        and any(
                            linked_sha256(source_hashes.get(name))
                            == staged_integrity.get("source_sha256")
                            for name in candidates
                        ),
                        "independent descendant execution-invalid staged-source drift",
                    )
                else:
                    _require(
                        staged_integrity is None
                        and audit_value.get("passed") is False
                        and capture_record.get("scientific_verifier_passed")
                        is False
                        and type(capture_record.get("verifier_returncode")) is int
                        and capture_record.get("verifier_returncode") != 0
                        and isinstance(audit_value.get("error_type"), str)
                        and bool(audit_value["error_type"])
                        and isinstance(audit_value.get("error"), str)
                        and bool(audit_value["error"]),
                        "independent descendant execution-invalid verifier-failure drift",
                    )
                lineage = "direct_postconfirmation_integrity_checkpoint"
            audit_checkpoint = {
                "name": first["checkpoint_name"],
                "created_unix_ns": timestamp,
                "evidence_path": first["evidence_path"],
                "evidence_sha256": first["evidence_sha256"],
                "source_attempt": ACTIVE_ATTEMPT,
                "verification_lineage": lineage,
            }
            decision_checkpoint = {
                "name": last["checkpoint_name"],
                "created_unix_ns": timestamp,
                "evidence_path": last["evidence_path"],
                "evidence_sha256": last["evidence_sha256"],
                "source_attempt": ACTIVE_ATTEMPT,
                "verification_lineage": lineage,
            }
            exact = deep_copy(base)
            exact["completed_states"] = before_completed + [
                "INDEPENDENT_VERIFICATION", "TERMINAL"
            ]
            exact["verified_checkpoints"] = before_checkpoints + [
                audit_checkpoint, decision_checkpoint
            ]
            exact["last_verified_checkpoint"] = decision_checkpoint
            exact["current_state"] = "POST_TERMINAL_REPORTING"
            exact["terminal_label"] = decision["terminal_label"]
            exact["process_valid"] = decision["process_valid"]
            exact["scientific_terminal"] = True
            exact["confirmation_terminal"] = not early
            exact["terminal_decision_path"] = decision["decision_path"]
            exact["terminal_decision_sha256"] = decision["decision_sha256"]
            if early:
                exact["terminal_basis"] = f"preregistered_{mode}"
                exact_early = exact["early_scientific_failure"]
                exact_early.update(
                    {
                        "status": "terminal_recorded",
                        "independent_verification_path": first["evidence_path"],
                        "independent_verification_sha256": first["evidence_sha256"],
                        "decision_path": last["evidence_path"],
                        "decision_sha256": last["evidence_sha256"],
                        "terminal_recorded_unix_ns": timestamp,
                    }
                )
                exact["next_action"] = (
                    "write the concise terminal report, robustness map, audit, "
                    "limitations, and next project-scoped task; then checkpoint "
                    "POST_TERMINAL_REPORTING"
                )
            else:
                exact["terminal_basis"] = "postconfirmation_integrity_failure"
                exact_integrity = exact.get("postconfirmation_integrity_failure")
                if not isinstance(exact_integrity, dict):
                    exact_integrity = {
                        "source": "independent_verifier",
                        "trigger_state": "INDEPENDENT_VERIFICATION",
                        "skipped_states": [],
                    }
                    exact["postconfirmation_integrity_failure"] = exact_integrity
                    exact["skipped_states"] = []
                exact_integrity.update(
                    {
                        "status": "terminal_recorded",
                        "independent_verification_path": first["evidence_path"],
                        "independent_verification_sha256": first["evidence_sha256"],
                        "decision_path": last["evidence_path"],
                        "decision_sha256": last["evidence_sha256"],
                        "terminal_recorded_unix_ns": timestamp,
                    }
                )
                exact["next_action"] = (
                    "report the immutable execution-invalid result and integrity "
                    "diagnosis; do not retry or reinterpret it as a scientific "
                    "partial/failure"
                )
            exact["updated_unix_ns"] = timestamp
            _require(
                compact_bytes(proposed) == compact_bytes(exact),
                "independent descendant three-event unrelated STATE drift",
            )
            return
        raise VerificationError(
            "independent descendant adapter event sequence drift"
        )

    post_count = post_ledger.get("event_count")
    post_bytes = post_ledger.get("bytes")
    lines = ledger_payload.splitlines(keepends=True)
    active_edges = [
        index
        for index, event in enumerate(current_events)
        if event.get("event") == "zero_confirmation_outcome_version_forward"
        and event.get("attempt") == ACTIVE_ATTEMPT
    ]
    _require(
        type(post_count) is int
        and post_count > 0
        and type(post_bytes) is int
        and post_bytes > 0
        and len(active_edges) == 1
        and post_count == active_edges[0] + 1
        and len(lines) == len(current_events)
        == current_state.get("ledger_event_count")
        and post_count <= len(lines)
        and b"".join(lines[:post_count]) == ledger_payload[:post_bytes]
        and post_bytes == len(b"".join(lines[:post_count]))
        and hashlib.sha256(ledger_payload[:post_bytes]).hexdigest()
        == post_ledger.get("sha256")
        and post_ledger.get("head_sha256")
        == current_events[post_count - 1].get("record_sha256")
        and post_state.get("ledger_event_count") == post_count
        and post_state.get("ledger_head_sha256")
        == post_ledger.get("head_sha256"),
        "independent descendant adapter ledger boundary drift",
    )
    expected_base = deep_copy(post_state)
    if post_count == len(lines):
        _require(
            compact_bytes(current_state) == compact_bytes(expected_base),
            "independent current initial adapter state differs from receipt",
        )
        return {"group_count": 0, "event_count": 0, "passed": True}

    field = "v008_durable_adapter_transaction"
    kind = "receipt_anchored_v008_descendant_adapter_transaction"
    cursor = post_count
    group_count = 0
    descendant_event_count = 0
    while cursor < len(current_events):
        first = current_events[cursor]
        envelope = first.get(field)
        first_keys = {
            "schema_version", "authorization_kind", "transaction_id",
            "event_index", "event_count", "base_state_sha256",
            "proposed_state_sha256", "base_state",
        }
        _require(
            isinstance(envelope, Mapping)
            and set(envelope) == first_keys
            and type(envelope.get("schema_version")) is int
            and envelope.get("schema_version") == 1
            and envelope.get("authorization_kind") == kind
            and is_sha256(envelope.get("transaction_id"))
            and type(envelope.get("event_index")) is int
            and envelope.get("event_index") == 0
            and type(envelope.get("event_count")) is int
            and envelope.get("event_count") > 0
            and is_sha256(envelope.get("base_state_sha256"))
            and is_sha256(envelope.get("proposed_state_sha256"))
            and isinstance(envelope.get("base_state"), Mapping),
            "independent descendant adapter first-envelope drift",
        )
        count = int(envelope["event_count"])
        end = cursor + count
        _require(
            end <= len(current_events),
            "independent descendant adapter group exceeds ledger",
        )
        base_state = dict(envelope["base_state"])
        _require(
            canonical_object_sha256(base_state)
            == envelope.get("base_state_sha256")
            and compact_bytes(base_state) == compact_bytes(expected_base)
            and base_state.get("active_attempt") == ACTIVE_ATTEMPT
            and type(base_state.get("ledger_event_count")) is int
            and base_state.get("ledger_event_count") == cursor
            and base_state.get("ledger_head_sha256")
            == current_events[cursor - 1].get("record_sha256"),
            "independent descendant adapter base-state drift",
        )
        stripped: list[dict[str, Any]] = []
        transaction_id = str(envelope["transaction_id"])
        for offset in range(count):
            record = current_events[cursor + offset]
            item = record.get(field)
            keys = {
                "schema_version", "authorization_kind", "transaction_id",
                "event_index", "event_count", "base_state_sha256",
                "proposed_state_sha256",
            }
            if offset == 0:
                keys.add("base_state")
            record_without_hash = dict(record)
            observed_record_sha256 = record_without_hash.pop(
                "record_sha256", None
            )
            _require(
                lines[cursor + offset] == compact_bytes(record) + b"\n"
                and isinstance(item, Mapping)
                and set(item) == keys
                and type(item.get("schema_version")) is int
                and item.get("schema_version") == 1
                and item.get("authorization_kind") == kind
                and item.get("transaction_id") == transaction_id
                and type(item.get("event_index")) is int
                and item.get("event_index") == offset
                and type(item.get("event_count")) is int
                and item.get("event_count") == count
                and item.get("base_state_sha256")
                == envelope.get("base_state_sha256")
                and item.get("proposed_state_sha256")
                == envelope.get("proposed_state_sha256")
                and type(record.get("seq")) is int
                and record.get("seq") == cursor + offset + 1
                and record.get("prev_sha256")
                == current_events[cursor + offset - 1].get("record_sha256")
                and is_sha256(observed_record_sha256)
                and observed_record_sha256
                == canonical_object_sha256(record_without_hash)
                and type(record.get("created_unix_ns")) is int
                and record.get("created_unix_ns") > 0
                and record.get("attempt") == ACTIVE_ATTEMPT,
                "independent descendant adapter envelope/record drift",
            )
            stripped_event = dict(record)
            for owned in (field, "seq", "prev_sha256", "record_sha256"):
                stripped_event.pop(owned)
            stripped.append(stripped_event)
        sequence = [event.get("event") for event in stripped]
        _require(
            (
                len(sequence) == 1
                and sequence[0]
                in {
                    "state_completed", "outcome_counts_updated",
                    "preregistered_early_scientific_failure_staged",
                    "scientific_terminal_decision_recorded",
                }
            )
            or sequence
            == [
                "state_completed", "scientific_terminal_decision_recorded",
                "state_completed",
            ],
            "independent descendant adapter event sequence drift",
        )
        transaction_projection = {
            "schema_version": 1,
            "authorization_kind": kind,
            "base_state_sha256": canonical_object_sha256(base_state),
            "proposed_state_sha256": envelope.get("proposed_state_sha256"),
            "events": stripped,
        }
        _require(
            canonical_object_sha256(transaction_projection) == transaction_id,
            "independent descendant adapter transaction id drift",
        )
        if end < len(current_events):
            next_envelope = current_events[end].get(field)
            _require(
                isinstance(next_envelope, Mapping)
                and next_envelope.get("event_index") == 0
                and isinstance(next_envelope.get("base_state"), Mapping),
                "independent descendant adapter grouping gap",
            )
            result_state = dict(next_envelope["base_state"])
        else:
            result_state = dict(current_state)
        _require(
            result_state.get("active_attempt") == ACTIVE_ATTEMPT
            and type(result_state.get("ledger_event_count")) is int
            and result_state.get("ledger_event_count") == end
            and result_state.get("ledger_head_sha256")
            == current_events[end - 1].get("record_sha256"),
            "independent descendant adapter target-state drift",
        )
        without_marker = deep_copy(result_state)
        without_marker.pop("v008_durable_controller_adapter", None)
        proposal = deep_copy(without_marker)
        proposal["ledger_event_count"] = base_state["ledger_event_count"]
        proposal["ledger_head_sha256"] = base_state["ledger_head_sha256"]
        _require(
            canonical_object_sha256(proposal)
            == envelope.get("proposed_state_sha256"),
            "independent descendant adapter proposed-state commitment drift",
        )
        verify_controller_transition(base_state, proposal, stripped)
        operation = canonical_object_sha256(
            {
                "base_state_object_sha256": canonical_object_sha256(base_state),
                "base_ledger_sha256": hashlib.sha256(
                    b"".join(lines[:cursor])
                ).hexdigest(),
                "expected_suffix_sha256": hashlib.sha256(
                    b"".join(lines[cursor:end])
                ).hexdigest(),
                "intended_state_object_sha256": canonical_object_sha256(
                    without_marker
                ),
            }
        )
        verified_marker = _verify_durable_controller_adapter_marker(
            result_state,
            ledger_event_count=end,
            ledger_head_sha256=str(current_events[end - 1]["record_sha256"]),
            policy=policy,
        )
        _require(
            verified_marker["operation_sha256"] == operation,
            "independent descendant adapter operation digest drift",
        )
        count_events = [
            event
            for event in stripped
            if event.get("event") == "outcome_counts_updated"
        ]
        _require(
            (
                not count_events
                and all(
                    "prior_controller_adapter_marker" not in event
                    for event in stripped
                )
            )
            or (
                len(stripped) == len(count_events) == 1
                and compact_bytes(
                    count_events[0].get("prior_controller_adapter_marker")
                )
                == compact_bytes(
                    base_state.get("v008_durable_controller_adapter")
                )
            ),
            "independent descendant count prior-marker drift",
        )
        expected_base = result_state
        cursor = end
        group_count += 1
        descendant_event_count += count
    return {
        "group_count": group_count,
        "event_count": descendant_event_count,
        "passed": True,
    }


def _verify_version_forward_transaction_receipt(
    *,
    policy: PathPolicy,
    current_state: Mapping[str, Any],
    current_events: Sequence[Mapping[str, Any]],
    expected_seal_sha256: str,
    expected_invalidity_sha256: str,
    expected_edge: Mapping[str, Any],
    expected_source_marker: Mapping[str, Any],
    expected_equivalence: Mapping[str, Any],
    expected_active_sealed_file_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay the durable schema-v2 version-forward receipt.

    This verifier intentionally does not import or execute the transaction
    adapter, root controller, inheritance verifier, or receipt producer.  It
    reconstructs the one permitted root transition, ledger suffix, adapter
    operation, and historical STATE bytes directly from the receipt and the
    current immutable byte prefixes.
    """

    receipt_relative = (
        f"{policy.attempt_relative}/audit/version_forward_transaction_receipt.json"
    )
    seal_relative = (
        f"{policy.attempt_relative}/audit/pre_data_inheritance_seal.json"
    )
    invalidity_relative = INVALIDITY_RELATIVE
    invalidity_draft_relative = INVALIDITY_DRAFT_RELATIVE
    prior_receipt_relative = SOURCE_RECEIPT_RELATIVE
    state_relative = f"{policy.study_relative}/STATE.json"
    ledger_relative = f"{policy.study_relative}/RESEARCH_LEDGER.jsonl"
    genesis_relative = f"{policy.study_relative}/LEDGER_CHAIN_GENESIS.json"
    root_program_relative = f"{policy.study_relative}/program.py"
    adapter_relative = f"{policy.attempt_relative}/version_forward_transaction.py"
    zero_outcomes: dict[str, int | bool] = {
        "fit_outcome_episodes": 0,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    numeric_outcomes = tuple(
        key for key, value in zero_outcomes.items() if type(value) is int
    )
    partition_names = {
        "exact_hash",
        "normalized_ast",
        "canonical_contracts",
        "procedural_only",
        "new_lineage_support",
        "lineage_support",
    }

    def is_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def deep_copy(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise VerificationError("receipt contains a non-JSON value") from error

    def compact_bytes(value: Any) -> bytes:
        try:
            return json.dumps(
                value, allow_nan=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise VerificationError("receipt contains non-canonical JSON") from error

    def pretty_bytes(value: Any) -> bytes:
        try:
            return (
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise VerificationError("receipt contains non-canonical JSON") from error

    def object_sha256(value: Any) -> str:
        return hashlib.sha256(compact_bytes(value)).hexdigest()

    def exact_mapping(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
        _require(
            isinstance(value, Mapping) and set(value) == keys,
            f"{label} closed schema drift",
        )
        return value

    def stable_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise VerificationError(f"cannot open {label}") from error
        try:
            before = os.fstat(descriptor)
            _require(
                stat.S_ISREG(before.st_mode)
                and not stat.S_ISLNK(before.st_mode)
                and int(before.st_nlink) == 1,
                f"{label} is linked, aliased, or non-regular",
            )
            chunks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(descriptor)
            _require(
                (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_nlink,
                )
                == (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_nlink,
                ),
                f"{label} changed while read",
            )
            payload = b"".join(chunks)
            return {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "device": int(before.st_dev),
                "inode": int(before.st_ino),
                "nlink": int(before.st_nlink),
            }, payload
        finally:
            os.close(descriptor)

    def json_object(payload: bytes, label: str) -> dict[str, Any]:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError(f"invalid JSON in {label}") from error
        _require(type(value) is dict, f"{label} is not a JSON object")
        return value

    def live_rich_record(
        relative: str, *, label: str, include_ast: bool
    ) -> tuple[dict[str, Any], bytes]:
        path = policy.path(relative, scope="study")
        metadata, payload = stable_file(path, label)
        record: dict[str, Any] = {"path": relative, **metadata}
        if include_ast:
            try:
                source = payload.decode("utf-8")
                tree = ast.parse(source, filename=str(path))
            except (UnicodeDecodeError, SyntaxError) as error:
                raise VerificationError(f"cannot parse {label}") from error
            record["ast_sha256"] = hashlib.sha256(
                ast.dump(tree, include_attributes=False).encode("utf-8")
            ).hexdigest()
        return record, payload

    def validate_rich_record(
        value: Any,
        *,
        relative: str,
        label: str,
        include_ast: bool,
    ) -> tuple[dict[str, Any], bytes]:
        keys = {"path", "sha256", "bytes", "device", "inode", "nlink"}
        if include_ast:
            keys.add("ast_sha256")
        exact_mapping(value, keys, label)
        for key in ("bytes", "device", "inode", "nlink"):
            _require(
                type(value.get(key)) is int and int(value[key]) >= 0,
                f"{label} metadata type drift: {key}",
            )
        _require(value.get("nlink") == 1, f"{label} no-replace identity drift")
        _require(is_sha256(value.get("sha256")), f"{label} hash syntax drift")
        if include_ast:
            _require(
                is_sha256(value.get("ast_sha256")),
                f"{label} AST hash syntax drift",
            )
        observed, payload = live_rich_record(
            relative, label=label, include_ast=include_ast
        )
        _require(dict(value) == observed, f"{label} live identity drift")
        return observed, payload

    def ledger_summary(
        payload: bytes, genesis_payload: bytes, label: str
    ) -> dict[str, Any]:
        genesis = json_object(genesis_payload, "ledger genesis")
        prefix_bytes = genesis.get("legacy_prefix_bytes")
        legacy_count = genesis.get("legacy_prefix_event_count")
        legacy_sha256 = genesis.get("legacy_prefix_sha256")
        _require(
            type(prefix_bytes) is int
            and prefix_bytes > 0
            and type(legacy_count) is int
            and legacy_count > 0
            and is_sha256(legacy_sha256),
            "ledger genesis digest schema drift",
        )
        _require(payload.endswith(b"\n"), f"{label} lacks terminal newline")
        prefix = payload[:prefix_bytes]
        _require(
            len(prefix) == prefix_bytes
            and hashlib.sha256(prefix).hexdigest() == legacy_sha256
            and prefix.endswith(b"\n"),
            f"{label} legacy prefix drift",
        )
        lines = payload.splitlines(keepends=True)
        _require(len(lines) >= legacy_count, f"{label} lost legacy events")
        _require(
            b"".join(lines[:legacy_count]) == prefix,
            f"{label} legacy event/byte boundary drift",
        )
        previous = f"legacy:{legacy_sha256}"
        events: list[dict[str, Any]] = []
        last_created = -1
        for sequence, encoded in enumerate(lines, start=1):
            _require(encoded.endswith(b"\n"), f"{label} line framing drift")
            event = json_object(encoded, f"{label} event {sequence}")
            created = event.get("created_unix_ns")
            _require(
                type(created) is int and created >= 0 and created >= last_created,
                f"{label} timestamp/type drift at event {sequence}",
            )
            last_created = created
            if sequence > legacy_count:
                _require(
                    encoded == compact_bytes(event) + b"\n",
                    f"{label} event {sequence} is not canonical compact JSON",
                )
                unhashed = dict(event)
                observed_hash = unhashed.pop("record_sha256", None)
                _require(
                    is_sha256(observed_hash)
                    and observed_hash == object_sha256(unhashed)
                    and type(unhashed.get("seq")) is int
                    and unhashed.get("seq") == sequence
                    and unhashed.get("prev_sha256") == previous,
                    f"{label} chain drift at event {sequence}",
                )
                previous = str(observed_hash)
            events.append(event)
        return {
            "event_count": len(events),
            "head_sha256": previous,
            "ledger_sha256": hashlib.sha256(payload).hexdigest(),
            "events": events,
            "last_created_unix_ns": last_created,
        }

    def validate_snapshot(
        value: Any,
        *,
        label: str,
        current_ledger_payload: bytes,
        live_genesis_record: Mapping[str, Any],
        genesis_payload: bytes,
    ) -> dict[str, Any]:
        exact_mapping(value, {"state", "ledger", "genesis"}, f"{label} snapshot")
        state_record = exact_mapping(
            value.get("state"),
            {"path", "sha256", "bytes", "object_sha256", "object"},
            f"{label} STATE digest",
        )
        ledger_record = exact_mapping(
            value.get("ledger"),
            {
                "path",
                "sha256",
                "bytes",
                "event_count",
                "head_sha256",
                "ledger_sha256",
            },
            f"{label} ledger digest",
        )
        genesis_record = exact_mapping(
            value.get("genesis"),
            {"path", "sha256", "bytes"},
            f"{label} genesis digest",
        )
        _require(
            state_record.get("path") == state_relative
            and is_sha256(state_record.get("sha256"))
            and type(state_record.get("bytes")) is int
            and state_record["bytes"] > 0
            and is_sha256(state_record.get("object_sha256"))
            and type(state_record.get("object")) is dict,
            f"{label} STATE digest schema drift",
        )
        encoded_state = pretty_bytes(state_record["object"])
        _require(
            len(encoded_state) == state_record["bytes"]
            and hashlib.sha256(encoded_state).hexdigest() == state_record["sha256"]
            and object_sha256(state_record["object"])
            == state_record["object_sha256"],
            f"{label} STATE byte/object digest drift",
        )
        _require(
            ledger_record.get("path") == ledger_relative
            and is_sha256(ledger_record.get("sha256"))
            and ledger_record.get("ledger_sha256") == ledger_record.get("sha256")
            and type(ledger_record.get("bytes")) is int
            and ledger_record["bytes"] > 0
            and type(ledger_record.get("event_count")) is int
            and ledger_record["event_count"] > 0
            and isinstance(ledger_record.get("head_sha256"), str),
            f"{label} ledger digest schema drift",
        )
        prefix = current_ledger_payload[: ledger_record["bytes"]]
        _require(
            len(prefix) == ledger_record["bytes"]
            and hashlib.sha256(prefix).hexdigest() == ledger_record["sha256"],
            f"{label} ledger is not an authenticated current prefix",
        )
        summary = ledger_summary(prefix, genesis_payload, f"{label} ledger")
        _require(
            summary["event_count"] == ledger_record["event_count"]
            and summary["head_sha256"] == ledger_record["head_sha256"]
            and summary["ledger_sha256"] == ledger_record["ledger_sha256"],
            f"{label} ledger digest recomputation drift",
        )
        expected_genesis = {
            "path": genesis_relative,
            "sha256": live_genesis_record["sha256"],
            "bytes": live_genesis_record["bytes"],
        }
        _require(
            dict(genesis_record) == expected_genesis,
            f"{label} ledger genesis live binding drift",
        )
        state_object = state_record["object"]
        _require(
            type(state_object.get("ledger_event_count")) is int
            and state_object.get("ledger_event_count") == ledger_record["event_count"]
            and state_object.get("ledger_head_sha256") == ledger_record["head_sha256"],
            f"{label} STATE/ledger digest binding drift",
        )
        return {
            "state": state_object,
            "ledger": ledger_record,
            "events": summary["events"],
            "payload": prefix,
            "last_created_unix_ns": summary["last_created_unix_ns"],
        }

    def validate_independent_partition_result(value: Any, label: str) -> None:
        exact_mapping(
            value,
            {"source_path_count", "target_path_count", "partition_counts"},
            label,
        )
        counts = value.get("partition_counts")
        _require(
            type(value.get("source_path_count")) is int
            and value["source_path_count"] > 0
            and type(value.get("target_path_count")) is int
            and value["target_path_count"] > 0
            and isinstance(counts, Mapping)
            and set(counts) == partition_names
            and all(type(count) is int and count >= 0 for count in counts.values())
            and sum(counts.values()) == value["target_path_count"],
            f"{label} type/count drift",
        )
        _require(
            dict(value) == dict(expected_equivalence),
            f"{label} differs from fresh independent equivalence",
        )

    def validate_verifiers(
        value: Any,
        *,
        phase: str,
        pre_state_sha256: str,
        pre_ledger_sha256: str,
    ) -> None:
        wrapper_keys = (
            {"producer", "standalone", "independent_partitions"}
            if phase == "pre-forward"
            else {"standalone", "independent_partitions"}
        )
        exact_mapping(value, wrapper_keys, f"{phase} verifier wrapper")
        independent = value.get("independent_partitions")
        validate_independent_partition_result(
            independent, f"{phase} independent partitions"
        )
        if phase == "pre-forward":
            producer = exact_mapping(
                value.get("producer"),
                {
                    "passed",
                    "phase",
                    "attempt",
                    "seal_path",
                    "seal_sha256",
                    "state_sha256",
                    "ledger_sha256",
                    "outcome_arrays_opened",
                    "read_only",
                },
                "pre-forward producer verifier",
            )
            _require(
                producer.get("passed") is True
                and producer.get("phase") == phase
                and producer.get("attempt") == ACTIVE_ATTEMPT
                and producer.get("seal_path") == seal_relative
                and producer.get("seal_sha256") == expected_seal_sha256
                and producer.get("state_sha256") == pre_state_sha256
                and producer.get("ledger_sha256") == pre_ledger_sha256
                and producer.get("outcome_arrays_opened") is False
                and producer.get("read_only") is True,
                "pre-forward producer verifier binding drift",
            )
        standalone = exact_mapping(
            value.get("standalone"),
            {
                "passed",
                "phase",
                "attempt",
                "active_attempt",
                "science_attempt",
                "seal_path",
                "seal_sha256",
                "partition_file_count",
                "sealed_file_count",
                "authorized_early_verifier_state",
                "authorized_role_count_recovery",
                "outcome_arrays_opened",
                "output_paths_created",
                "read_only",
            },
            f"{phase} standalone verifier",
        )
        _require(
            standalone.get("passed") is True
            and standalone.get("phase") == phase
            and standalone.get("attempt") == ACTIVE_ATTEMPT
            and standalone.get("active_attempt") == ACTIVE_ATTEMPT
            and standalone.get("science_attempt") == SCIENCE_ATTEMPT
            and standalone.get("seal_path") == seal_relative
            and standalone.get("seal_sha256") == expected_seal_sha256
            and type(standalone.get("partition_file_count")) is int
            and standalone.get("partition_file_count")
            == independent["target_path_count"]
            and type(standalone.get("sealed_file_count")) is int
            and standalone["sealed_file_count"]
            == expected_active_sealed_file_summary["file_count"]
            and standalone.get("authorized_early_verifier_state") is None
            and standalone.get("authorized_role_count_recovery") is None
            and standalone.get("outcome_arrays_opened") is False
            and type(standalone.get("output_paths_created")) is int
            and standalone.get("output_paths_created") == 0
            and standalone.get("read_only") is True,
            f"{phase} standalone verifier binding drift",
        )

    exact_mapping(
        expected_active_sealed_file_summary,
        {"file_count", "total_bytes"},
        "fresh active sealed-file summary",
    )
    _require(
        type(expected_active_sealed_file_summary.get("file_count")) is int
        and expected_active_sealed_file_summary.get("file_count") > 0
        and type(expected_active_sealed_file_summary.get("total_bytes")) is int
        and expected_active_sealed_file_summary.get("total_bytes") > 0,
        "fresh active sealed-file summary/count drift",
    )
    _require(
        isinstance(expected_equivalence, Mapping)
        and set(expected_equivalence)
        == {"source_path_count", "target_path_count", "partition_counts"}
        and type(expected_equivalence.get("source_path_count")) is int
        and expected_equivalence.get("source_path_count") > 0
        and type(expected_equivalence.get("target_path_count")) is int
        and expected_equivalence.get("target_path_count") > 0
        and isinstance(expected_equivalence.get("partition_counts"), Mapping)
        and set(expected_equivalence["partition_counts"]) == partition_names
        and all(
            type(count) is int and count >= 0
            for count in expected_equivalence["partition_counts"].values()
        )
        and sum(expected_equivalence["partition_counts"].values())
        == expected_equivalence["target_path_count"],
        "fresh independent source-partition projection drift",
    )

    receipt_path = policy.path(receipt_relative, scope="attempt")
    receipt_file_record, receipt_payload = stable_file(
        receipt_path, "version-forward transaction receipt"
    )
    receipt = json_object(receipt_payload, "version-forward transaction receipt")
    _require(
        receipt_payload == pretty_bytes(receipt),
        "version-forward transaction receipt is not canonical pretty JSON",
    )
    top_keys = {
        "schema_version",
        "artifact_type",
        "authorization_kind",
        "source_attempt",
        "target_attempt",
        "resume_state",
        "journal_created_unix_ns",
        "forward_created_unix_ns",
        "created_unix_ns",
        "receipt_context_sha256",
        "receipt_context",
        "seal",
        "invalidity",
        "invalidity_draft",
        "prior_receipt",
        "root_program",
        "transaction_source",
        "pre_snapshot",
        "pre_verifiers",
        "controller_return",
        "controller_return_sha256",
        "post_snapshot",
        "post_verifiers",
        "outcome_counts",
        "state_transaction_absent",
        "receipt_journal_absent",
        "passed",
    }
    exact_mapping(receipt, top_keys, "version-forward receipt")
    journal_created = receipt.get("journal_created_unix_ns")
    forward_created = receipt.get("forward_created_unix_ns")
    receipt_created = receipt.get("created_unix_ns")
    _require(
        type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 2
        and receipt.get("artifact_type")
        == "atomic_zero_outcome_version_forward_transaction_receipt"
        and receipt.get("authorization_kind")
        == "locked_v005_to_v008_inherited_pre_data_forward"
        and receipt.get("source_attempt") == SOURCE_ATTEMPT
        and receipt.get("target_attempt") == ACTIVE_ATTEMPT
        and receipt.get("resume_state") == VERSION_FORWARD_RESUME_STATE
        and type(journal_created) is int
        and journal_created > 0
        and type(forward_created) is int
        and forward_created == journal_created + 1
        and type(receipt_created) is int
        and receipt_created == journal_created + 2
        and is_sha256(receipt.get("receipt_context_sha256"))
        and type(receipt.get("receipt_context")) is dict
        and type(receipt.get("controller_return")) is dict
        and is_sha256(receipt.get("controller_return_sha256"))
        and receipt.get("state_transaction_absent") is True
        and receipt.get("receipt_journal_absent") is True
        and receipt.get("passed") is True,
        "version-forward receipt header/type/timestamp drift",
    )
    outcome_counts = exact_mapping(
        receipt.get("outcome_counts"), set(zero_outcomes), "receipt outcome counts"
    )
    _require(
        all(type(outcome_counts.get(key)) is int for key in numeric_outcomes)
        and type(outcome_counts.get("confirmation_outcomes_opened_for_analysis"))
        is bool
        and dict(outcome_counts) == zero_outcomes,
        "receipt outcome counts are not strict exact zero",
    )

    context = exact_mapping(
        receipt["receipt_context"],
        {
            "schema_version",
            "source_attempt",
            "target_attempt",
            "resume_state",
            "journal_created_unix_ns",
            "forward_created_unix_ns",
            "receipt_created_unix_ns",
            "fixed_inputs",
            "pre_snapshot",
            "pre_verifiers",
            "controller_proposal",
            "predicted_post_verifiers",
        },
        "receipt precommit context",
    )
    _require(
        type(context.get("schema_version")) is int
        and context.get("schema_version") == 1
        and context.get("source_attempt") == SOURCE_ATTEMPT
        and context.get("target_attempt") == ACTIVE_ATTEMPT
        and context.get("resume_state") == VERSION_FORWARD_RESUME_STATE
        and type(context.get("journal_created_unix_ns")) is int
        and context.get("journal_created_unix_ns") == journal_created
        and type(context.get("forward_created_unix_ns")) is int
        and context.get("forward_created_unix_ns") == forward_created
        and type(context.get("receipt_created_unix_ns")) is int
        and context.get("receipt_created_unix_ns") == receipt_created
        and object_sha256(context) == receipt["receipt_context_sha256"],
        "receipt precommit context digest/timestamp drift",
    )
    fixed_inputs = exact_mapping(
        context.get("fixed_inputs"),
        {
            "seal",
            "invalidity",
            "invalidity_draft",
            "prior_receipt",
            "root_program",
            "transaction_source",
        },
        "receipt fixed inputs",
    )
    _require(
        fixed_inputs.get("seal") == receipt.get("seal")
        and fixed_inputs.get("invalidity") == receipt.get("invalidity")
        and fixed_inputs.get("invalidity_draft")
        == receipt.get("invalidity_draft")
        and fixed_inputs.get("prior_receipt") == receipt.get("prior_receipt")
        and fixed_inputs.get("root_program") == receipt.get("root_program")
        and fixed_inputs.get("transaction_source")
        == receipt.get("transaction_source"),
        "receipt fixed-input projection drift",
    )
    fixed_specs = {
        "seal": (seal_relative, False),
        "invalidity": (invalidity_relative, False),
        "invalidity_draft": (invalidity_draft_relative, False),
        "prior_receipt": (prior_receipt_relative, False),
        "root_program": (root_program_relative, True),
        "transaction_source": (adapter_relative, True),
    }
    fixed_records: dict[str, dict[str, Any]] = {}
    fixed_payloads: dict[str, bytes] = {}
    for name, (relative, include_ast) in fixed_specs.items():
        record, payload = validate_rich_record(
            receipt[name],
            relative=relative,
            label=f"receipt {name}",
            include_ast=include_ast,
        )
        fixed_records[name] = record
        fixed_payloads[name] = payload
    _require(
        is_sha256(expected_seal_sha256)
        and fixed_records["seal"]["sha256"] == expected_seal_sha256,
        "receipt inheritance-seal hash drift",
    )
    _require(
        is_sha256(expected_invalidity_sha256)
        and fixed_records["invalidity"]["sha256"]
        == expected_invalidity_sha256,
        "receipt invalidity hash drift",
    )

    state_path = policy.path(state_relative, scope="study")
    ledger_path = policy.path(ledger_relative, scope="study")
    current_state_file, current_state_payload = stable_file(
        state_path, "current controller STATE"
    )
    current_ledger_file, current_ledger_payload = stable_file(
        ledger_path, "current research ledger"
    )
    live_genesis_record, genesis_payload = live_rich_record(
        genesis_relative, label="ledger genesis", include_ast=False
    )
    current_state_object = json_object(current_state_payload, "current controller STATE")
    _require(
        compact_bytes(current_state_object) == compact_bytes(dict(current_state)),
        "current controller STATE differs from independently supplied state",
    )
    current_ledger = ledger_summary(
        current_ledger_payload, genesis_payload, "current research ledger"
    )
    _require(
        compact_bytes(current_ledger["events"])
        == compact_bytes([dict(event) for event in current_events]),
        "current research ledger differs from independently supplied events",
    )
    _require(
        type(current_state_object.get("ledger_event_count")) is int
        and current_state_object.get("ledger_event_count")
        == current_ledger["event_count"]
        and current_state_object.get("ledger_head_sha256")
        == current_ledger["head_sha256"],
        "current controller STATE/ledger drift",
    )
    pre = validate_snapshot(
        receipt.get("pre_snapshot"),
        label="pre-forward",
        current_ledger_payload=current_ledger_payload,
        live_genesis_record=live_genesis_record,
        genesis_payload=genesis_payload,
    )
    post = validate_snapshot(
        receipt.get("post_snapshot"),
        label="post-forward",
        current_ledger_payload=current_ledger_payload,
        live_genesis_record=live_genesis_record,
        genesis_payload=genesis_payload,
    )
    _require(
        compact_bytes(context.get("pre_snapshot"))
        == compact_bytes(receipt.get("pre_snapshot"))
        and compact_bytes(context.get("pre_verifiers"))
        == compact_bytes(receipt.get("pre_verifiers"))
        and compact_bytes(context.get("predicted_post_verifiers"))
        == compact_bytes(receipt.get("post_verifiers")),
        "receipt context/top-level projection drift",
    )
    validate_verifiers(
        receipt.get("pre_verifiers"),
        phase="pre-forward",
        pre_state_sha256=receipt["pre_snapshot"]["state"]["sha256"],
        pre_ledger_sha256=receipt["pre_snapshot"]["ledger"]["sha256"],
    )
    validate_verifiers(
        receipt.get("post_verifiers"),
        phase="post-forward",
        pre_state_sha256=receipt["pre_snapshot"]["state"]["sha256"],
        pre_ledger_sha256=receipt["pre_snapshot"]["ledger"]["sha256"],
    )
    expected_post_standalone = deep_copy(receipt["pre_verifiers"]["standalone"])
    expected_post_standalone["phase"] = "post-forward"
    _require(
        compact_bytes(receipt["post_verifiers"]["standalone"])
        == compact_bytes(expected_post_standalone)
        and compact_bytes(receipt["post_verifiers"]["independent_partitions"])
        == compact_bytes(receipt["pre_verifiers"]["independent_partitions"]),
        "receipt predicted post-verifier projection drift",
    )

    pre_state = pre["state"]
    for key in numeric_outcomes:
        _require(
            type(pre_state.get(key)) is int and pre_state.get(key) == 0,
            f"pre-forward outcome counter type/value drift: {key}",
        )
    _require(
        type(pre_state.get("confirmation_outcomes_opened_for_analysis")) is bool
        and pre_state.get("confirmation_outcomes_opened_for_analysis") is False
        and type(pre_state.get("schema_version")) is int
        and pre_state.get("schema_version") == 1
        and pre_state.get("active_attempt") == SOURCE_ATTEMPT
        and pre_state.get("active_attempt_path")
        == ATTEMPT_ROOTS[SOURCE_ATTEMPT]
        and pre_state.get("current_state") == VERSION_FORWARD_RESUME_STATE
        and pre_state.get("confirmation_terminal") is False
        and pre_state.get("early_scientific_failure") is None
        and "v008_durable_controller_adapter" not in pre_state
        and type(pre_state.get("updated_unix_ns")) is int
        and 0 < pre_state["updated_unix_ns"] <= journal_created
        and pre["last_created_unix_ns"] <= journal_created,
        "pre-forward STATE identity/type/chronology drift",
    )
    history = pre_state.get("attempt_history")
    checkpoints = pre_state.get("verified_checkpoints")
    _require(
        isinstance(history, list)
        and len(history) == 5
        and all(isinstance(item, Mapping) for item in history)
        and [item.get("version") for item in history]
        == [
            SCIENCE_ATTEMPT,
            PRIOR_ATTEMPT,
            FIT_SOURCE_ATTEMPT,
            INTERMEDIATE_ATTEMPT,
            SOURCE_ATTEMPT,
        ]
        and history[0].get("path") == ATTEMPT_ROOTS[SCIENCE_ATTEMPT]
        and history[0].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[1].get("path") == ATTEMPT_ROOTS[PRIOR_ATTEMPT]
        and history[1].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[2].get("path") == ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]
        and history[2].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[3].get("path") == ATTEMPT_ROOTS[INTERMEDIATE_ATTEMPT]
        and history[3].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[4].get("path") == ATTEMPT_ROOTS[SOURCE_ATTEMPT]
        and history[4].get("status")
        == "active_zero_confirmation_outcome_version_forward"
        and isinstance(checkpoints, list)
        and bool(checkpoints)
        and pre_state.get("last_verified_checkpoint") == checkpoints[-1],
        "pre-forward attempt/checkpoint lineage drift",
    )

    prior_lineage = pre_state.get("version_forward_lineage")
    _require(
        isinstance(prior_lineage, list)
        and len(prior_lineage) == 4
        and all(isinstance(item, Mapping) for item in prior_lineage)
        and prior_lineage[0].get("old_attempt") == SCIENCE_ATTEMPT
        and prior_lineage[0].get("new_attempt") == PRIOR_ATTEMPT
        and prior_lineage[0].get("resume_state")
        == VERSION_FORWARD_RESUME_STATE,
        "pre-forward v001->v002 lineage edge drift",
    )
    _require(
        prior_lineage[1].get("old_attempt") == PRIOR_ATTEMPT
        and prior_lineage[1].get("new_attempt") == FIT_SOURCE_ATTEMPT
        and prior_lineage[1].get("resume_state")
        == VERSION_FORWARD_RESUME_STATE,
        "pre-forward v002->v003 lineage edge drift",
    )
    _require(
        prior_lineage[2].get("old_attempt") == FIT_SOURCE_ATTEMPT
        and prior_lineage[2].get("new_attempt") == INTERMEDIATE_ATTEMPT
        and prior_lineage[2].get("resume_state")
        == VERSION_FORWARD_RESUME_STATE,
        "pre-forward v003->v004 lineage edge drift",
    )
    _require(
        prior_lineage[3].get("old_attempt") == INTERMEDIATE_ATTEMPT
        and prior_lineage[3].get("new_attempt") == SOURCE_ATTEMPT
        and prior_lineage[3].get("resume_state")
        == VERSION_FORWARD_RESUME_STATE,
        "pre-forward v004->v005 lineage edge drift",
    )

    source_marker = pre_state.get("v005_durable_controller_adapter")
    marker_keys = {
        "schema_version",
        "authorization_kind",
        "adapter_source_path",
        "adapter_source_sha256",
        "adapter_source_ast_sha256",
        "root_program_path",
        "root_program_sha256",
        "root_program_ast_sha256",
        "ledger_event_count",
        "ledger_head_sha256",
        "operation_sha256",
        "state_binding_sha256",
    }
    exact_mapping(source_marker, marker_keys, "source v005 adapter marker")
    _require(
        dict(source_marker) == dict(expected_source_marker),
        "receipt pre-state source marker differs from inheritance seal",
    )
    source_adapter_relative = (
        f"{ATTEMPT_ROOTS[SOURCE_ATTEMPT]}/version_forward_transaction.py"
    )
    source_adapter, _source_adapter_payload = live_rich_record(
        source_adapter_relative,
        label="source v005 adapter",
        include_ast=True,
    )
    source_root, _source_root_payload = live_rich_record(
        root_program_relative,
        label="source v005 root program",
        include_ast=True,
    )
    source_marker_without_binding = dict(source_marker)
    source_state_binding = source_marker_without_binding.pop(
        "state_binding_sha256", None
    )
    pre_without_source_marker = deep_copy(pre_state)
    pre_without_source_marker.pop("v005_durable_controller_adapter", None)
    _require(
        type(source_marker.get("schema_version")) is int
        and source_marker.get("schema_version") == 2
        and source_marker.get("authorization_kind")
        == "receipt_bound_v005_root_controller_adapter"
        and source_marker.get("adapter_source_path")
        == source_adapter_relative
        and source_marker.get("adapter_source_sha256")
        == source_adapter["sha256"]
        and source_marker.get("adapter_source_ast_sha256")
        == source_adapter["ast_sha256"]
        and source_marker.get("root_program_path") == root_program_relative
        and source_marker.get("root_program_sha256")
        == source_root["sha256"]
        and source_marker.get("root_program_ast_sha256")
        == source_root["ast_sha256"]
        and source_marker.get("ledger_event_count")
        == pre["ledger"]["event_count"]
        and source_marker.get("ledger_head_sha256")
        == pre["ledger"]["head_sha256"]
        and is_sha256(source_marker.get("operation_sha256"))
        and source_state_binding
        == object_sha256(
            {
                "marker_without_state_binding": source_marker_without_binding,
                "state_without_marker": pre_without_source_marker,
            }
        ),
        "source v005 adapter marker content/state binding drift",
    )
    prior_receipt = json_object(
        fixed_payloads["prior_receipt"], "historical v005 receipt"
    )
    prior_post = prior_receipt.get("post_snapshot")
    prior_post_state = (
        prior_post.get("state") if isinstance(prior_post, Mapping) else None
    )
    prior_transaction_source = prior_receipt.get("transaction_source")
    _require(
        prior_receipt.get("source_attempt") == INTERMEDIATE_ATTEMPT
        and prior_receipt.get("target_attempt") == SOURCE_ATTEMPT
        and prior_receipt.get("passed") is True
        and isinstance(prior_post_state, Mapping)
        and compact_bytes(prior_post_state.get("object"))
        == compact_bytes(pre_state)
        and prior_post_state.get("object_sha256") == object_sha256(pre_state)
        and compact_bytes(prior_receipt.get("controller_return"))
        == compact_bytes(pre_state)
        and prior_receipt.get("controller_return_sha256")
        == object_sha256(pre_state)
        and isinstance(prior_transaction_source, Mapping)
        and prior_transaction_source.get("sha256")
        == source_marker.get("adapter_source_sha256"),
        "historical v005 receipt/live source marker cross-link drift",
    )

    expected_proposal_state = deep_copy(pre_state)
    expected_proposal_state.pop("v005_durable_controller_adapter", None)
    inherited_projection: list[dict[str, Any]] = []
    first_annotation = {
        "attempt": PRIOR_ATTEMPT,
        "equivalence_evidence_path": prior_lineage[0]["equivalence_path"],
        "equivalence_evidence_sha256": prior_lineage[0][
            "equivalence_sha256"
        ],
    }
    fit_annotation = {
        "attempt": FIT_SOURCE_ATTEMPT,
        "equivalence_evidence_path": prior_lineage[1]["equivalence_path"],
        "equivalence_evidence_sha256": prior_lineage[1][
            "equivalence_sha256"
        ],
    }
    intermediate_annotation = {
        "attempt": INTERMEDIATE_ATTEMPT,
        "equivalence_evidence_path": prior_lineage[2]["equivalence_path"],
        "equivalence_evidence_sha256": prior_lineage[2][
            "equivalence_sha256"
        ],
    }
    source_annotation = {
        "attempt": SOURCE_ATTEMPT,
        "equivalence_evidence_path": SOURCE_PRE_DATA_RELATIVE,
        "equivalence_evidence_sha256": prior_lineage[3][
            "equivalence_sha256"
        ],
    }
    active_annotation = {
        "attempt": ACTIVE_ATTEMPT,
        "equivalence_evidence_path": seal_relative,
        "equivalence_evidence_sha256": expected_seal_sha256,
    }
    for index, (original, replayed) in enumerate(
        zip(checkpoints, expected_proposal_state["verified_checkpoints"], strict=True)
    ):
        _require(isinstance(original, Mapping), f"pre-forward checkpoint {index} drift")
        _require(
            isinstance(original.get("name"), str)
            and bool(original["name"])
            and isinstance(original.get("evidence_path"), str)
            and is_sha256(original.get("evidence_sha256"))
            and type(original.get("created_unix_ns")) is int
            and original["created_unix_ns"] > 0
            and original.get("source_attempt") == SCIENCE_ATTEMPT
            and original.get("verification_lineage") == "direct_checkpoint"
            and original.get("inherited_into_attempts")
            == [
                first_annotation,
                fit_annotation,
                intermediate_annotation,
                source_annotation,
            ],
            f"pre-forward checkpoint {index} schema/lineage drift",
        )
        evidence_path = policy.path(original["evidence_path"], scope="study")
        _require(
            evidence_path.is_relative_to(
                policy.repository_root
                / f"{policy.study_relative}/attempts/{SCIENCE_ATTEMPT}"
            ),
            f"pre-forward checkpoint {index} evidence escapes v001",
        )
        evidence_record, _evidence_payload = stable_file(
            evidence_path, f"pre-forward checkpoint {index} evidence"
        )
        _require(
            evidence_record["sha256"] == original["evidence_sha256"],
            f"pre-forward checkpoint {index} evidence hash drift",
        )
        replayed["inherited_into_attempts"] = [
            deep_copy(first_annotation),
            deep_copy(fit_annotation),
            deep_copy(intermediate_annotation),
            deep_copy(source_annotation),
            deep_copy(active_annotation),
        ]
        inherited_projection.append(
            {
                "checkpoint_name": original["name"],
                "source_attempt": SCIENCE_ATTEMPT,
                "evidence_path": original["evidence_path"],
                "evidence_sha256": original["evidence_sha256"],
            }
        )
    expected_proposal_state["last_verified_checkpoint"] = expected_proposal_state[
        "verified_checkpoints"
    ][-1]
    expected_proposal_state["attempt_history"][4][
        "status"
    ] = "invalid_zero_confirmation_outcome_procedural"
    expected_proposal_state["attempt_history"][4][
        "invalidity_evidence_path"
    ] = invalidity_relative
    expected_proposal_state["attempt_history"][4][
        "invalidity_evidence_sha256"
    ] = expected_invalidity_sha256
    expected_proposal_state["attempt_history"].append(
        {
            "version": ACTIVE_ATTEMPT,
            "path": f"{policy.study_relative}/attempts/{ACTIVE_ATTEMPT}",
            "status": "active_zero_confirmation_outcome_version_forward",
            "created_unix_ns": forward_created,
            "version_forward_evidence_path": seal_relative,
            "version_forward_evidence_sha256": expected_seal_sha256,
            "attempt_parameterization_verified": True,
        }
    )
    expected_proposal_state["active_attempt"] = ACTIVE_ATTEMPT
    expected_proposal_state[
        "active_attempt_path"
    ] = f"{policy.study_relative}/attempts/{ACTIVE_ATTEMPT}"
    lineage_edge = {
        "old_attempt": SOURCE_ATTEMPT,
        "new_attempt": ACTIVE_ATTEMPT,
        "resume_state": VERSION_FORWARD_RESUME_STATE,
        "invalidity_path": invalidity_relative,
        "invalidity_sha256": expected_invalidity_sha256,
        "equivalence_path": seal_relative,
        "equivalence_sha256": expected_seal_sha256,
        "inherited_verified_checkpoints": inherited_projection,
        "attempt_parameterization_verified": True,
    }
    _require(
        compact_bytes(dict(expected_edge)) == compact_bytes(lineage_edge),
        "receipt declarative edge differs from authenticated lineage",
    )
    expected_proposal_state["version_forward_lineage"] = deep_copy(
        prior_lineage
    ) + [deep_copy(lineage_edge)]
    expected_proposal_state["updated_unix_ns"] = forward_created
    expected_proposal_event = {
        "event": "zero_confirmation_outcome_version_forward",
        "attempt": ACTIVE_ATTEMPT,
        **lineage_edge,
        "source_controller_adapter_marker": deep_copy(source_marker),
        "created_unix_ns": forward_created,
    }
    proposal = exact_mapping(
        context.get("controller_proposal"),
        {"state", "events"},
        "receipt controller proposal",
    )
    _require(
        type(proposal.get("state")) is dict
        and isinstance(proposal.get("events"), list)
        and len(proposal["events"]) == 1
        and type(proposal["events"][0]) is dict
        and compact_bytes(proposal["state"])
        == compact_bytes(expected_proposal_state)
        and compact_bytes(proposal["events"][0])
        == compact_bytes(expected_proposal_event),
        "receipt controller proposal is not the exact declarative root transition",
    )

    context_sha256 = receipt["receipt_context_sha256"]
    pre_count = pre["ledger"]["event_count"]
    expected_record_without_hash = deep_copy(expected_proposal_event)
    expected_record_without_hash[
        "version_forward_receipt_context_sha256"
    ] = context_sha256
    expected_record_without_hash["seq"] = pre_count + 1
    expected_record_without_hash["prev_sha256"] = pre["ledger"]["head_sha256"]
    record_sha256 = object_sha256(expected_record_without_hash)
    expected_record = {
        **expected_record_without_hash,
        "record_sha256": record_sha256,
    }
    expected_suffix = compact_bytes(expected_record) + b"\n"
    _require(
        post["ledger"]["event_count"] == pre_count + 1
        and post["ledger"]["bytes"] == pre["ledger"]["bytes"] + len(expected_suffix)
        and post["ledger"]["head_sha256"] == record_sha256
        and post["payload"] == pre["payload"] + expected_suffix
        and post["events"] == pre["events"] + [expected_record]
        and post["last_created_unix_ns"] == forward_created,
        "receipt predicted pre/post ledger linkage drift",
    )
    _require(
        current_ledger["event_count"] >= post["ledger"]["event_count"]
        and current_ledger_payload[: post["ledger"]["bytes"]] == post["payload"],
        "receipt post-ledger prefix is not durable in the current ledger",
    )
    current_forward = [
        event
        for event in current_ledger["events"]
        if event.get("event") == "zero_confirmation_outcome_version_forward"
    ]
    context_bound = [
        event
        for event in current_ledger["events"]
        if "version_forward_receipt_context_sha256" in event
    ]
    _require(
        len(current_forward) == 5
        and current_forward[-1] == expected_record
        and len(context_bound) == 5
        and context_bound[-1] == expected_record
        and context_bound == current_forward,
        "current ledger lacks the exact fifth receipt-context-bound forward edge",
    )

    intended_without_marker = deep_copy(expected_proposal_state)
    intended_without_marker["ledger_event_count"] = pre_count + 1
    intended_without_marker["ledger_head_sha256"] = record_sha256
    intended_without_marker.pop("v008_durable_controller_adapter", None)
    operation_sha256 = object_sha256(
        {
            "base_state_object_sha256": object_sha256(pre_state),
            "base_ledger_sha256": hashlib.sha256(pre["payload"]).hexdigest(),
            "expected_suffix_sha256": hashlib.sha256(expected_suffix).hexdigest(),
            "intended_state_object_sha256": object_sha256(intended_without_marker),
        }
    )
    marker_without_binding = {
        "schema_version": 2,
        "authorization_kind": "receipt_bound_v008_root_controller_adapter",
        "adapter_source_path": fixed_records["transaction_source"]["path"],
        "adapter_source_sha256": fixed_records["transaction_source"]["sha256"],
        "adapter_source_ast_sha256": fixed_records["transaction_source"][
            "ast_sha256"
        ],
        "root_program_path": fixed_records["root_program"]["path"],
        "root_program_sha256": fixed_records["root_program"]["sha256"],
        "root_program_ast_sha256": fixed_records["root_program"]["ast_sha256"],
        "ledger_event_count": pre_count + 1,
        "ledger_head_sha256": record_sha256,
        "operation_sha256": operation_sha256,
    }
    expected_marker = dict(marker_without_binding)
    expected_marker["state_binding_sha256"] = object_sha256(
        {
            "marker_without_state_binding": marker_without_binding,
            "state_without_marker": intended_without_marker,
        }
    )
    expected_post_state = deep_copy(intended_without_marker)
    expected_post_state["v008_durable_controller_adapter"] = expected_marker
    post_state = post["state"]
    _require(
        compact_bytes(post_state) == compact_bytes(expected_post_state)
        and compact_bytes(receipt.get("controller_return"))
        == compact_bytes(expected_post_state)
        and receipt.get("controller_return_sha256")
        == object_sha256(expected_post_state)
        and {
            key: receipt["controller_return"].get(key) for key in zero_outcomes
        }
        == zero_outcomes,
        "receipt controller return/post STATE/operation linkage drift",
    )
    historical_marker = _verify_durable_controller_adapter_marker(
        post_state,
        ledger_event_count=pre_count + 1,
        ledger_head_sha256=record_sha256,
        policy=policy,
    )
    _require(
        historical_marker["operation_sha256"] == operation_sha256,
        "historical adapter marker operation digest drift",
    )
    _require(
        current_state_object.get("active_attempt") == ACTIVE_ATTEMPT
        and current_state_object.get("active_attempt_path")
        == f"{policy.study_relative}/attempts/{ACTIVE_ATTEMPT}"
        and current_state_object.get("version_forward_lineage")
        == expected_post_state["version_forward_lineage"]
        and current_state_object.get("attempt_history", [
        ])[ : len(expected_post_state["attempt_history"])]
        == expected_post_state["attempt_history"]
        and current_state_object.get("verified_checkpoints", [])[
            : len(expected_post_state["verified_checkpoints"])
        ]
        == expected_post_state["verified_checkpoints"]
        and "v005_durable_controller_adapter" not in current_state_object,
        "current STATE is not a descendant of the receipt-bound post STATE",
    )
    current_marker = _verify_durable_controller_adapter_marker(
        current_state_object,
        ledger_event_count=current_ledger["event_count"],
        ledger_head_sha256=current_ledger["head_sha256"],
        policy=policy,
    )
    descendant_adapter_chain = _verify_descendant_adapter_transaction_chain(
        post_state=post_state,
        post_ledger=post["ledger"],
        current_state=current_state_object,
        current_events=current_ledger["events"],
        ledger_payload=current_ledger_payload,
        policy=policy,
    )

    residue_relatives = (
        f"{policy.study_relative}/STATE_TRANSACTION.json",
        f"{policy.study_relative}/.STATE_TRANSACTION.json.v008-staging",
        f"{policy.study_relative}/.STATE.json.v008-staging",
        f"{policy.study_relative}/VERSION_FORWARD_TRANSACTION.json",
        f"{policy.study_relative}/.VERSION_FORWARD_TRANSACTION.json.v008-staging",
        f"{policy.attempt_relative}/audit/.version_forward_transaction_receipt.json.v008-staging",
    )
    residues = [
        relative
        for relative in residue_relatives
        if os.path.lexists(policy.repository_root / relative)
    ]
    _require(not residues, f"version-forward journal/transaction residue present: {residues}")
    _require(
        receipt_file_record["nlink"] == 1,
        "version-forward receipt no-replace identity drift",
    )
    return {
        "schema_version": 2,
        "receipt_path": receipt_relative,
        "receipt_sha256": receipt_file_record["sha256"],
        "receipt_bytes": receipt_file_record["bytes"],
        "receipt_device": receipt_file_record["device"],
        "receipt_inode": receipt_file_record["inode"],
        "receipt_nlink": receipt_file_record["nlink"],
        "receipt_context_sha256": context_sha256,
        "pre_event_count": pre_count,
        "post_event_count": pre_count + 1,
        "forward_record_sha256": record_sha256,
        "operation_sha256": operation_sha256,
        "current_event_count": current_ledger["event_count"],
        "current_state_sha256": current_state_file["sha256"],
        "current_ledger_sha256": current_ledger_file["sha256"],
        "active_sealed_file_count": expected_active_sealed_file_summary[
            "file_count"
        ],
        "partition_target_path_count": expected_equivalence["target_path_count"],
        "journal_resolved": True,
        "state_transaction_absent": True,
        "receipt_staging_absent": True,
        "historical_adapter": historical_marker,
        "current_adapter": current_marker,
        "descendant_adapter_chain": descendant_adapter_chain,
        "passed": True,
    }


def _verify_version_forward_transaction_receipt(
    *,
    policy: PathPolicy,
    current_state: Mapping[str, Any],
    current_events: Sequence[Mapping[str, Any]],
    expected_seal_sha256: str,
    expected_invalidity_sha256: str,
    expected_edge: Mapping[str, Any],
    expected_source_marker: Mapping[str, Any],
    expected_equivalence: Mapping[str, Any],
    expected_active_sealed_file_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently reconstruct the v007-to-v008 durable transaction."""

    def require_exact_mapping(
        value: Any,
        expected: Mapping[str, Any],
        label: str,
    ) -> None:
        _require(
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(
                type(value.get(key)) is type(item)
                and value.get(key) == item
                for key, item in expected.items()
            ),
            f"{label} drift",
        )

    receipt_relative = (
        f"{policy.attempt_relative}/audit/version_forward_transaction_receipt.json"
    )
    seal_relative = (
        f"{policy.attempt_relative}/audit/pre_data_inheritance_seal.json"
    )
    state_relative = f"{policy.study_relative}/STATE.json"
    ledger_relative = f"{policy.study_relative}/RESEARCH_LEDGER.jsonl"
    genesis_relative = f"{policy.study_relative}/LEDGER_CHAIN_GENESIS.json"
    root_program_relative = f"{policy.study_relative}/program.py"
    adapter_relative = f"{policy.attempt_relative}/version_forward_transaction.py"
    receipt_path = policy.path(receipt_relative, scope="study")
    receipt_metadata = receipt_path.lstat()
    _require(
        stat.S_ISREG(receipt_metadata.st_mode)
        and not stat.S_ISLNK(receipt_metadata.st_mode)
        and receipt_metadata.st_nlink == 1,
        "version-forward receipt is linked, aliased, or non-regular",
    )
    receipt = read_json(receipt_path)
    receipt_payload = receipt_path.read_bytes()
    _require(
        receipt_payload
        == (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
        "version-forward receipt is noncanonical",
    )
    receipt_file_record = {
        "path": receipt_relative,
        "sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "bytes": len(receipt_payload),
        "device": receipt_metadata.st_dev,
        "inode": receipt_metadata.st_ino,
        "nlink": receipt_metadata.st_nlink,
    }

    outcomes: dict[str, int | bool] = {
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    top_keys = {
        "schema_version", "artifact_type", "authorization_kind",
        "source_attempt", "target_attempt", "resume_state",
        "journal_created_unix_ns", "forward_created_unix_ns",
        "created_unix_ns", "receipt_context_sha256", "receipt_context",
        "seal", "invalidity", "invalidity_draft", "prior_receipt",
        "root_program", "transaction_source", "pre_snapshot",
        "pre_verifiers", "controller_return", "controller_return_sha256",
        "post_snapshot", "post_verifiers", "outcome_counts",
        "state_transaction_absent", "receipt_journal_absent", "passed",
    }
    journal_created = receipt.get("journal_created_unix_ns")
    forward_created = receipt.get("forward_created_unix_ns")
    receipt_created = receipt.get("created_unix_ns")
    context = receipt.get("receipt_context")
    context_sha256 = receipt.get("receipt_context_sha256")
    _require(
        set(receipt) == top_keys
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 2
        and receipt.get("artifact_type")
        == "atomic_zero_confirmation_outcome_version_forward_transaction_receipt"
        and receipt.get("authorization_kind")
        == "locked_v007_to_v008_inherited_pre_selection_forward"
        and receipt.get("source_attempt") == ACTIVATION_SOURCE_ATTEMPT
        and receipt.get("target_attempt") == ACTIVE_ATTEMPT
        and receipt.get("resume_state") == VERSION_FORWARD_RESUME_STATE
        and type(journal_created) is int
        and journal_created > 0
        and type(forward_created) is int
        and forward_created == journal_created + 1
        and type(receipt_created) is int
        and receipt_created == journal_created + 2
        and isinstance(context, Mapping)
        and context_sha256 == canonical_object_sha256(context)
        and require_exact_mapping(
            receipt.get("outcome_counts"), outcomes, "receipt outcome counts"
        )
        is None
        and receipt.get("state_transaction_absent") is True
        and receipt.get("receipt_journal_absent") is True
        and receipt.get("passed") is True,
        "version-forward receipt header/type drift",
    )

    def source_ast_sha256(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode()
        ).hexdigest()

    def validate_current_record(
        value: Any, relative: str, *, ast_bound: bool = False
    ) -> dict[str, Any]:
        path = policy.path(relative, scope="study")
        metadata = path.lstat()
        expected_keys = {"path", "sha256", "bytes", "device", "inode", "nlink"}
        if ast_bound:
            expected_keys.add("ast_sha256")
        observed = {
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": metadata.st_size,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "nlink": metadata.st_nlink,
        }
        if ast_bound:
            observed["ast_sha256"] = source_ast_sha256(path)
        _require(
            isinstance(value, Mapping)
            and set(value) == expected_keys
            and dict(value) == observed
            and stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_nlink == 1,
            f"version-forward fixed live identity drift: {relative}",
        )
        return observed

    fixed_relatives = {
        "seal": seal_relative,
        "invalidity": INVALIDITY_RELATIVE,
        "invalidity_draft": INVALIDITY_DRAFT_RELATIVE,
        "prior_receipt": SOURCE_RECEIPT_RELATIVE,
        "root_program": root_program_relative,
        "transaction_source": adapter_relative,
    }
    fixed_records = {
        name: validate_current_record(
            receipt.get(name),
            relative,
            ast_bound=name in {"root_program", "transaction_source"},
        )
        for name, relative in fixed_relatives.items()
    }
    _require(
        fixed_records["seal"]["sha256"] == expected_seal_sha256
        and fixed_records["invalidity"]["sha256"]
        == expected_invalidity_sha256
        and fixed_records["prior_receipt"]["sha256"]
        == EXPECTED_SOURCE_RECEIPT_SHA256,
        "version-forward fixed hash provenance drift",
    )

    context_keys = {
        "schema_version", "source_attempt", "target_attempt", "resume_state",
        "journal_created_unix_ns", "forward_created_unix_ns",
        "receipt_created_unix_ns", "fixed_inputs", "pre_snapshot",
        "pre_verifiers", "controller_proposal", "predicted_post_verifiers",
    }
    _require(
        set(context) == context_keys
        and type(context.get("schema_version")) is int
        and context.get("schema_version") == 1
        and context.get("source_attempt") == ACTIVATION_SOURCE_ATTEMPT
        and context.get("target_attempt") == ACTIVE_ATTEMPT
        and context.get("resume_state") == VERSION_FORWARD_RESUME_STATE
        and context.get("journal_created_unix_ns") == journal_created
        and context.get("forward_created_unix_ns") == forward_created
        and context.get("receipt_created_unix_ns") == receipt_created
        and context.get("fixed_inputs")
        == {name: receipt[name] for name in fixed_relatives}
        and context.get("pre_snapshot") == receipt.get("pre_snapshot")
        and context.get("pre_verifiers") == receipt.get("pre_verifiers")
        and context.get("predicted_post_verifiers")
        == receipt.get("post_verifiers"),
        "version-forward precommit context binding drift",
    )

    pre_verifiers = receipt.get("pre_verifiers")
    post_verifiers = receipt.get("post_verifiers")
    _require(
        isinstance(pre_verifiers, Mapping)
        and set(pre_verifiers)
        == {"producer", "standalone", "independent_partitions"}
        and isinstance(post_verifiers, Mapping)
        and set(post_verifiers) == {"standalone", "independent_partitions"},
        "version-forward verifier record census drift",
    )
    producer = pre_verifiers.get("producer")
    expected_producer_keys = {
        "passed", "phase", "attempt", "seal_path", "seal_sha256",
        "state_sha256", "ledger_sha256", "outcome_arrays_opened", "read_only",
    }
    _require(
        isinstance(producer, Mapping)
        and set(producer) == expected_producer_keys
        and producer.get("passed") is True
        and producer.get("phase") == "pre-forward"
        and producer.get("attempt") == ACTIVE_ATTEMPT
        and producer.get("seal_path") == seal_relative
        and producer.get("seal_sha256") == expected_seal_sha256
        and producer.get("state_sha256") == EXPECTED_SOURCE_STATE_SHA256
        and producer.get("ledger_sha256") == EXPECTED_SOURCE_LEDGER_SHA256
        and producer.get("outcome_arrays_opened") is False
        and producer.get("read_only") is True,
        "version-forward producer verifier projection drift",
    )

    def verify_standalone_projection(value: Any, phase: str) -> None:
        expected_keys = {
            "passed", "phase", "attempt", "active_attempt", "science_attempt",
            "seal_path", "seal_sha256", "partition_file_count",
            "sealed_file_count", "authorized_early_verifier_state",
            "authorized_role_count_recovery", "outcome_arrays_opened",
            "output_paths_created", "read_only",
        }
        _require(
            isinstance(value, Mapping)
            and set(value) == expected_keys
            and value.get("passed") is True
            and value.get("phase") == phase
            and value.get("attempt") == ACTIVE_ATTEMPT
            and value.get("active_attempt") == ACTIVE_ATTEMPT
            and value.get("science_attempt") == SCIENCE_ATTEMPT
            and value.get("seal_path") == seal_relative
            and value.get("seal_sha256") == expected_seal_sha256
            and type(value.get("partition_file_count")) is int
            and value.get("partition_file_count")
            == expected_equivalence.get("target_path_count")
            and type(value.get("sealed_file_count")) is int
            and value.get("sealed_file_count")
            == expected_active_sealed_file_summary.get("file_count")
            and value.get("authorized_early_verifier_state") is None
            and value.get("authorized_role_count_recovery") is None
            and value.get("outcome_arrays_opened") is False
            and type(value.get("output_paths_created")) is int
            and value.get("output_paths_created") == 0
            and value.get("read_only") is True,
            f"version-forward {phase} standalone verifier projection drift",
        )

    verify_standalone_projection(pre_verifiers.get("standalone"), "pre-forward")
    verify_standalone_projection(post_verifiers.get("standalone"), "post-forward")
    _require(
        pre_verifiers.get("independent_partitions") == expected_equivalence
        and post_verifiers.get("independent_partitions") == expected_equivalence,
        "version-forward independent equivalence projection drift",
    )

    pre = receipt.get("pre_snapshot")
    post = receipt.get("post_snapshot")
    _require(
        isinstance(pre, Mapping)
        and set(pre) == {"state", "ledger", "genesis"}
        and isinstance(post, Mapping)
        and set(post) == {"state", "ledger", "genesis"},
        "version-forward snapshot schema drift",
    )

    def validate_state_record(value: Any, label: str) -> dict[str, Any]:
        _require(
            isinstance(value, Mapping)
            and set(value) == {"path", "sha256", "bytes", "object_sha256", "object"}
            and value.get("path") == state_relative
            and type(value.get("bytes")) is int
            and isinstance(value.get("object"), Mapping),
            f"version-forward {label} state-record drift",
        )
        state_object = dict(value["object"])
        pretty = (json.dumps(state_object, indent=2, sort_keys=True) + "\n").encode()
        _require(
            value.get("sha256") == hashlib.sha256(pretty).hexdigest()
            and value.get("bytes") == len(pretty)
            and value.get("object_sha256")
            == canonical_object_sha256(state_object),
            f"version-forward {label} state content/digest drift",
        )
        return state_object

    pre_state = validate_state_record(pre.get("state"), "pre")
    post_state = validate_state_record(post.get("state"), "post")
    _require(
        pre["state"].get("sha256") == EXPECTED_SOURCE_STATE_SHA256
        and pre_state.get("active_attempt") == ACTIVATION_SOURCE_ATTEMPT
        and pre_state.get("active_attempt_path")
        == ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]
        and pre_state.get("current_state") == VERSION_FORWARD_RESUME_STATE
        and require_exact_mapping(
            {key: pre_state.get(key) for key in outcomes},
            outcomes,
            "pre-forward outcomes",
        )
        is None
        and pre_state.get("completed_states") == list(INHERITED_COMPLETED_STATES)
        and isinstance(pre_state.get("verified_checkpoints"), list)
        and len(pre_state["verified_checkpoints"]) == len(INHERITED_COMPLETED_STATES)
        and isinstance(pre_state.get("attempt_history"), list)
        and [item.get("version") for item in pre_state["attempt_history"]]
        == list(ALLOWED_ATTEMPTS[:-1])
        and isinstance(pre_state.get("version_forward_lineage"), list)
        and len(pre_state["version_forward_lineage"]) == 6,
        "v007 pre-forward selection-boundary drift",
    )

    ledger_keys = {
        "path", "sha256", "bytes", "event_count", "head_sha256",
        "ledger_sha256",
    }
    pre_ledger = pre.get("ledger")
    post_ledger = post.get("ledger")
    _require(
        isinstance(pre_ledger, Mapping)
        and set(pre_ledger) == ledger_keys
        and isinstance(post_ledger, Mapping)
        and set(post_ledger) == ledger_keys
        and pre_ledger.get("path") == post_ledger.get("path") == ledger_relative
        and pre_ledger.get("sha256") == EXPECTED_SOURCE_LEDGER_SHA256
        and pre_ledger.get("ledger_sha256") == EXPECTED_SOURCE_LEDGER_SHA256
        and type(pre_ledger.get("event_count")) is int
        and type(post_ledger.get("event_count")) is int
        and post_ledger.get("event_count") == pre_ledger.get("event_count") + 1,
        "version-forward pre/post ledger record drift",
    )
    ledger_path = policy.path(ledger_relative, scope="study")
    ledger_payload = ledger_path.read_bytes()
    lines = ledger_payload.splitlines(keepends=True)
    pre_count = int(pre_ledger["event_count"])
    post_count = int(post_ledger["event_count"])
    _require(
        len(lines) == len(current_events)
        and all(line.endswith(b"\n") for line in lines)
        and pre_ledger.get("bytes") == len(b"".join(lines[:pre_count]))
        and post_ledger.get("bytes") == len(b"".join(lines[:post_count]))
        and hashlib.sha256(b"".join(lines[:pre_count])).hexdigest()
        == pre_ledger.get("sha256")
        and hashlib.sha256(b"".join(lines[:post_count])).hexdigest()
        == post_ledger.get("sha256")
        and post_ledger.get("ledger_sha256") == post_ledger.get("sha256")
        and pre_state.get("ledger_event_count") == pre_count
        and pre_state.get("ledger_head_sha256") == pre_ledger.get("head_sha256")
        and post_state.get("ledger_event_count") == post_count
        and post_state.get("ledger_head_sha256") == post_ledger.get("head_sha256"),
        "version-forward ledger-prefix byte drift",
    )
    for snapshot in (pre, post):
        genesis = snapshot.get("genesis")
        _require(
            isinstance(genesis, Mapping)
            and set(genesis) == {"path", "sha256", "bytes"}
            and genesis.get("path") == genesis_relative
            and genesis.get("sha256")
            == sha256_file(policy.path(genesis_relative, scope="study")),
            "version-forward genesis record drift",
        )
    _require(pre.get("genesis") == post.get("genesis"), "genesis identity drift")

    forward = dict(current_events[post_count - 1])
    raw_forward = dict(forward)
    for key in ("seq", "prev_sha256", "record_sha256"):
        raw_forward.pop(key, None)
    _require(
        forward.get("event") == "zero_confirmation_outcome_version_forward"
        and forward.get("attempt") == ACTIVE_ATTEMPT
        and forward.get("old_attempt") == ACTIVATION_SOURCE_ATTEMPT
        and forward.get("new_attempt") == ACTIVE_ATTEMPT
        and forward.get("resume_state") == VERSION_FORWARD_RESUME_STATE
        and forward.get("created_unix_ns") == forward_created
        and forward.get("seq") == post_count
        and forward.get("prev_sha256") == pre_ledger.get("head_sha256")
        and forward.get("record_sha256") == post_ledger.get("head_sha256")
        and forward.get("version_forward_receipt_context_sha256")
        == context_sha256,
        "version-forward ledger edge drift",
    )
    record_projection = dict(forward)
    record_sha256 = record_projection.pop("record_sha256")
    _require(
        record_sha256 == canonical_object_sha256(record_projection)
        and lines[post_count - 1]
        == canonical_json_bytes(forward) + b"\n",
        "version-forward ledger edge hash drift",
    )

    source_marker = pre_state.get("v007_durable_controller_adapter")
    source_marker_core = dict(source_marker) if isinstance(source_marker, Mapping) else {}
    source_binding = source_marker_core.pop("state_binding_sha256", None)
    pre_without_source_marker = json.loads(json.dumps(pre_state))
    pre_without_source_marker.pop("v007_durable_controller_adapter", None)
    source_adapter_path = policy.path(
        f"{ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]}/version_forward_transaction.py",
        scope="study",
    )
    _require(
        isinstance(source_marker, Mapping)
        and source_marker == expected_source_marker
        and forward.get("source_controller_adapter_marker") == source_marker
        and source_marker.get("authorization_kind")
        == "receipt_bound_v007_root_controller_adapter"
        and source_marker.get("adapter_source_path")
        == f"{ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]}/version_forward_transaction.py"
        and source_marker.get("adapter_source_sha256")
        == sha256_file(source_adapter_path)
        and source_marker.get("adapter_source_ast_sha256")
        == source_ast_sha256(source_adapter_path)
        and source_marker.get("ledger_event_count") == pre_count
        and source_marker.get("ledger_head_sha256") == pre_ledger.get("head_sha256")
        and source_binding
        == canonical_object_sha256(
            {
                "marker_without_state_binding": source_marker_core,
                "state_without_marker": pre_without_source_marker,
            }
        ),
        "v007 source adapter marker drift",
    )

    edge = {
        key: forward[key]
        for key in (
            "old_attempt", "new_attempt", "resume_state", "invalidity_path",
            "invalidity_sha256", "equivalence_path", "equivalence_sha256",
            "inherited_verified_checkpoints", "attempt_parameterization_verified",
        )
    }
    _require(
        edge == dict(expected_edge)
        and edge.get("invalidity_path") == INVALIDITY_RELATIVE
        and edge.get("invalidity_sha256") == expected_invalidity_sha256
        and edge.get("equivalence_path") == seal_relative
        and edge.get("equivalence_sha256") == expected_seal_sha256,
        "version-forward expected-edge provenance drift",
    )
    proposal_state = json.loads(json.dumps(pre_state))
    proposal_state.pop("v007_durable_controller_adapter", None)
    annotation = {
        "attempt": ACTIVE_ATTEMPT,
        "equivalence_evidence_path": seal_relative,
        "equivalence_evidence_sha256": expected_seal_sha256,
    }
    for checkpoint in proposal_state.get("verified_checkpoints", []):
        inherited = checkpoint.setdefault("inherited_into_attempts", [])
        _require(isinstance(inherited, list), "checkpoint inheritance schema drift")
        if annotation not in inherited:
            inherited.append(dict(annotation))
    proposal_state["last_verified_checkpoint"] = proposal_state[
        "verified_checkpoints"
    ][-1]
    history = proposal_state.get("attempt_history")
    _require(isinstance(history, list) and len(history) == 7, "source history drift")
    history[-1]["status"] = "invalid_zero_confirmation_outcome_procedural"
    history[-1]["invalidity_evidence_path"] = INVALIDITY_RELATIVE
    history[-1]["invalidity_evidence_sha256"] = expected_invalidity_sha256
    history.append(
        {
            "version": ACTIVE_ATTEMPT,
            "path": policy.attempt_relative,
            "status": "active_zero_confirmation_outcome_version_forward",
            "created_unix_ns": forward_created,
            "version_forward_evidence_path": seal_relative,
            "version_forward_evidence_sha256": expected_seal_sha256,
            "attempt_parameterization_verified": True,
        }
    )
    proposal_state["active_attempt"] = ACTIVE_ATTEMPT
    proposal_state["active_attempt_path"] = policy.attempt_relative
    proposal_state["version_forward_lineage"] = [
        *proposal_state["version_forward_lineage"],
        edge,
    ]
    proposal_state["updated_unix_ns"] = forward_created
    proposal = context.get("controller_proposal")
    _require(
        isinstance(proposal, Mapping)
        and proposal.get("state") == proposal_state
        and proposal.get("events") == [
            {
                key: value
                for key, value in raw_forward.items()
                if key != "version_forward_receipt_context_sha256"
            }
        ],
        "version-forward declarative root transition/controller proposal reconstruction drift",
    )

    intended_without_marker = json.loads(json.dumps(proposal_state))
    intended_without_marker["ledger_event_count"] = post_count
    intended_without_marker["ledger_head_sha256"] = record_sha256
    post_without_marker = json.loads(json.dumps(post_state))
    post_marker = post_without_marker.pop("v008_durable_controller_adapter", None)
    _require(
        post_without_marker == intended_without_marker,
        "version-forward post-state reconstruction drift",
    )
    historical_marker = _verify_durable_controller_adapter_marker(
        post_state,
        ledger_event_count=post_count,
        ledger_head_sha256=str(record_sha256),
        policy=policy,
    )
    operation_sha256 = canonical_object_sha256(
        {
            "base_state_object_sha256": canonical_object_sha256(pre_state),
            "base_ledger_sha256": pre_ledger["sha256"],
            "expected_suffix_sha256": hashlib.sha256(
                lines[post_count - 1]
            ).hexdigest(),
            "intended_state_object_sha256": canonical_object_sha256(
                post_without_marker
            ),
        }
    )
    _require(
        post_marker.get("operation_sha256") == operation_sha256
        and receipt.get("controller_return") == post_state
        and receipt.get("controller_return_sha256")
        == canonical_object_sha256(post_state),
        "version-forward post marker/controller return operation linkage drift",
    )

    current_state_path = policy.path(state_relative, scope="study")
    current_state_payload = current_state_path.read_bytes()
    current_state_object = read_json(current_state_path)
    _require(
        dict(current_state) == current_state_object
        and current_state_object.get("active_attempt") == ACTIVE_ATTEMPT
        and "v007_durable_controller_adapter" not in current_state_object
        and current_state_object.get("version_forward_lineage")
        == post_state.get("version_forward_lineage")
        and current_state_object.get("attempt_history", [])[:8]
        == post_state.get("attempt_history")
        and current_state_object.get("verified_checkpoints", [])[:9]
        == post_state.get("verified_checkpoints"),
        "current STATE is not a receipt-bound descendant",
    )
    current_ledger_file = {
        "sha256": sha256_file(ledger_path),
        "bytes": ledger_path.stat().st_size,
    }
    current_marker = _verify_durable_controller_adapter_marker(
        current_state_object,
        ledger_event_count=len(current_events),
        ledger_head_sha256=str(current_state_object.get("ledger_head_sha256")),
        policy=policy,
    )
    descendant_adapter_chain = _verify_descendant_adapter_transaction_chain(
        post_state=post_state,
        post_ledger=post_ledger,
        current_state=current_state_object,
        current_events=current_events,
        ledger_payload=ledger_payload,
        policy=policy,
    )
    residues = [
        relative
        for relative in (
            f"{policy.study_relative}/STATE_TRANSACTION.json",
            f"{policy.study_relative}/.STATE_TRANSACTION.json.v008-staging",
            f"{policy.study_relative}/.STATE.json.v008-staging",
            f"{policy.study_relative}/VERSION_FORWARD_TRANSACTION.json",
            f"{policy.study_relative}/.VERSION_FORWARD_TRANSACTION.json.v008-staging",
            f"{policy.attempt_relative}/audit/.version_forward_transaction_receipt.json.v008-staging",
        )
        if os.path.lexists(policy.repository_root / relative)
    ]
    _require(not residues, f"version-forward transaction residue present: {residues}")
    return {
        "schema_version": 2,
        "receipt_path": receipt_relative,
        "receipt_sha256": receipt_file_record["sha256"],
        "receipt_bytes": receipt_file_record["bytes"],
        "receipt_device": receipt_file_record["device"],
        "receipt_inode": receipt_file_record["inode"],
        "receipt_nlink": receipt_file_record["nlink"],
        "receipt_context_sha256": context_sha256,
        "pre_event_count": pre_count,
        "post_event_count": post_count,
        "forward_record_sha256": record_sha256,
        "operation_sha256": operation_sha256,
        "current_event_count": len(current_events),
        "current_state_sha256": hashlib.sha256(current_state_payload).hexdigest(),
        "current_ledger_sha256": current_ledger_file["sha256"],
        "active_sealed_file_count": expected_active_sealed_file_summary[
            "file_count"
        ],
        "partition_target_path_count": expected_equivalence["target_path_count"],
        "journal_resolved": True,
        "state_transaction_absent": True,
        "receipt_staging_absent": True,
        "historical_adapter": historical_marker,
        "current_adapter": current_marker,
        "descendant_adapter_chain": descendant_adapter_chain,
        "passed": True,
    }


def verify_ledger_and_state(
    contract: Mapping[str, Any], policy: PathPolicy
) -> dict[str, Any]:
    paths = contract["paths"]
    genesis = read_json(policy.path(paths["ledger_genesis"], scope="study"))
    ledger_path = policy.path(paths["ledger"], scope="study")
    raw = ledger_path.read_bytes()
    prefix_bytes = int(genesis["legacy_prefix_bytes"])
    legacy_count = int(genesis["legacy_prefix_event_count"])
    prefix = raw[:prefix_bytes]
    _require(len(prefix) == prefix_bytes, "ledger shorter than legacy prefix")
    _require(hashlib.sha256(prefix).hexdigest() == genesis["legacy_prefix_sha256"], "legacy ledger prefix drift")
    _require(prefix.endswith(b"\n"), "legacy prefix is not line complete")
    line_bytes = raw.splitlines(keepends=True)
    _require(len(line_bytes) >= legacy_count, "ledger lost legacy events")
    _require(b"".join(line_bytes[:legacy_count]) == prefix, "legacy byte/event boundary mismatch")
    events: list[dict[str, Any]] = []
    previous = f"legacy:{genesis['legacy_prefix_sha256']}"
    last_created = -1
    for index, encoded in enumerate(line_bytes, start=1):
        _require(encoded.endswith(b"\n"), f"ledger event {index} lacks newline")
        value = json.loads(encoded)
        _require(isinstance(value, dict), f"ledger event {index} is not an object")
        created = int(value.get("created_unix_ns", -1))
        _require(created >= last_created, f"ledger timestamp moved backward at {index}")
        last_created = created
        if index > legacy_count:
            record = dict(value)
            observed_hash = record.pop("record_sha256", None)
            expected_hash = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
            _require(observed_hash == expected_hash, f"ledger hash drift at sequence {index}")
            _require(record.get("seq") == index, f"ledger sequence drift at {index}")
            _require(record.get("prev_sha256") == previous, f"ledger predecessor drift at {index}")
            previous = str(observed_hash)
        events.append(value)
    _require(events and events[0].get("event") == "program_initialized", "ledger genesis event drift")
    edge_positions = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "zero_confirmation_outcome_version_forward"
    ]
    expected_edge_pairs = list(zip(ALLOWED_ATTEMPTS[:-1], ALLOWED_ATTEMPTS[1:]))
    _require(
        len(edge_positions) == len(expected_edge_pairs) == 7,
        "ledger version-forward edge census drift",
    )
    for edge_number, (edge_position, pair) in enumerate(
        zip(edge_positions, expected_edge_pairs, strict=True)
    ):
        _require(
            edge_position >= legacy_count,
            f"version-forward edge {edge_number} is inside the legacy prefix",
        )
        edge_record = events[edge_position]
        expected_resume = (
            VERSION_FORWARD_RESUME_STATE if edge_number >= 5 else "FIT_COHORTS"
        )
        _require(
            edge_record.get("attempt") == pair[1]
            and edge_record.get("old_attempt") == pair[0]
            and edge_record.get("new_attempt") == pair[1]
            and edge_record.get("resume_state") == expected_resume,
            f"version-forward ledger edge identity drift: {edge_number}",
        )
    prior_edge_index, edge_index = edge_positions[-2:]
    prior_edge_event = events[prior_edge_index]
    edge_event = events[edge_index]
    _require(
        all(
            event.get("attempt") == SCIENCE_ATTEMPT
            for event in events[: edge_positions[0]]
        ),
        "pre-forward ledger segment is not contiguous v001",
    )
    for segment_number, (start, attempt) in enumerate(
        zip(edge_positions, ALLOWED_ATTEMPTS[1:], strict=True)
    ):
        end = (
            edge_positions[segment_number + 1]
            if segment_number + 1 < len(edge_positions)
            else len(events)
        )
        _require(
            all(event.get("attempt") == attempt for event in events[start:end]),
            f"ledger attempt segment is not contiguous: {attempt}",
        )
    pre_data_completed = STATE_MACHINE[: STATE_MACHINE.index("FIT_COHORTS")]
    pre_edge_completions = [
        event.get("completed_state")
        for event in events[: edge_positions[0]]
        if event.get("event") == "state_completed"
    ]
    _require(
        pre_edge_completions == list(pre_data_completed),
        "v001 ledger segment does not end at the inherited pre-data checkpoint",
    )
    post_edge_completions = [
        event.get("completed_state")
        for event in events[edge_index + 1 :]
        if event.get("event") == "state_completed"
    ]
    _require(
        bool(post_edge_completions) and post_edge_completions[0] == VERSION_FORWARD_RESUME_STATE,
        "first direct v008 completion is not SELECTION_COHORTS",
    )
    _require(
        not any(state in INHERITED_COMPLETED_STATES for state in post_edge_completions),
        "v008 replayed an inherited pre-data checkpoint",
    )

    state = read_json(policy.path(paths["state"], scope="study"))
    _require(state.get("schema_version") == 1, "state schema drift")
    _require(state.get("active_attempt") == ATTEMPT, "state active attempt drift")
    _require(state.get("active_attempt_path") == contract["attempt_root"], "state attempt path drift")
    _require(tuple(state.get("state_machine", ())) == STATE_MACHINE, "state machine drift")
    expected_git = contract.get("expected_git_head")
    _require(isinstance(expected_git, str) and state.get("expected_git_head") == expected_git, "expected Git head drift")
    _require(state.get("current_state") == "INDEPENDENT_VERIFICATION", "verifier invoked in illegal state")
    mode = str(contract["mode"])
    analysis_execution_invalid = bool(contract.get("_analysis_execution_invalid_branch"))
    early_triggers = {
        "no_candidate": "CANDIDATE_SELECTION",
        "power_infeasible": "CONFIRMATION_POWER_AND_COHORT_FREEZE",
    }
    if mode == "confirmation":
        _require(state.get("early_scientific_failure") in (None, {}), "confirmation mode carries an early-failure branch")
        if analysis_execution_invalid:
            trigger = "SEALED_ANALYSIS"
            trigger_index = STATE_MACHINE.index(trigger)
            independent_index = STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
            expected_completed = list(STATE_MACHINE[: trigger_index + 1])
            expected_skipped = list(STATE_MACHINE[trigger_index + 1 : independent_index])
            staged = state.get("postconfirmation_integrity_failure")
            _require(isinstance(staged, Mapping), "analysis execution-invalid staging record absent")
            _require(
                staged.get("status") == "awaiting_independent_verification"
                and staged.get("source") == "analysis_execution"
                and staged.get("trigger_state") == trigger,
                "analysis execution-invalid staging identity drift",
            )
            _require(staged.get("source_path") == contract["_analysis_execution_invalid_relative"], "analysis failure source path drift")
            failure_path = policy.path(staged["source_path"], scope="attempt")
            _require(staged.get("source_sha256") == sha256_file(failure_path), "analysis failure source hash drift")
            _require(staged.get("skipped_states") == expected_skipped, "analysis failure skipped-state list drift")
            audit_keys = staged.get("audit_hash_keys")
            _require(isinstance(audit_keys, list) and "analysis_execution_invalid" in audit_keys, "analysis failure audit hash key absent")
            skipped_records = state.get("skipped_states")
            _require(
                isinstance(skipped_records, list)
                and [item.get("state") for item in skipped_records] == expected_skipped,
                "analysis failure skipped-state transparency drift",
            )
            _require(
                all(
                    item.get("source") == "analysis_execution"
                    and item.get("reason") == "postconfirmation_integrity_failure_short_circuit"
                    and item.get("source_path") == staged.get("source_path")
                    and item.get("source_sha256") == staged.get("source_sha256")
                    for item in skipped_records
                ),
                "analysis failure skipped-state record drift",
            )
        else:
            current_index = STATE_MACHINE.index(state["current_state"])
            expected_completed = list(STATE_MACHINE[:current_index])
            expected_skipped = []
            _require(state.get("postconfirmation_integrity_failure") in (None, {}), "ordinary confirmation carries an integrity-failure branch")
    else:
        trigger = early_triggers[mode]
        trigger_index = STATE_MACHINE.index(trigger)
        independent_index = STATE_MACHINE.index("INDEPENDENT_VERIFICATION")
        expected_completed = list(STATE_MACHINE[:trigger_index])
        expected_skipped = list(STATE_MACHINE[trigger_index + 1 : independent_index])
        early = state.get("early_scientific_failure")
        _require(isinstance(early, Mapping), "early-failure state record absent")
        _require(early.get("mode") == mode and early.get("trigger_state") == trigger, "early-failure branch identity drift")
        _require(early.get("skipped_states") == expected_skipped, "early-failure skipped-state list drift")
        skipped_records = state.get("skipped_states")
        _require(isinstance(skipped_records, list) and [item.get("state") for item in skipped_records] == expected_skipped, "controller skipped-state records drift")
        _require(all(item.get("mode") == mode for item in skipped_records), "skipped-state mode drift")
    _require(state.get("completed_states") == expected_completed, "completed-state prefix drift")
    checkpoints = state.get("verified_checkpoints")
    _require(isinstance(checkpoints, list) and len(checkpoints) == len(expected_completed), "checkpoint count drift")
    checkpoint_by_state: dict[str, Mapping[str, Any]] = {}
    last_checkpoint_created = -1
    for completed_state, checkpoint in zip(expected_completed, checkpoints, strict=True):
        _require(isinstance(checkpoint, Mapping), "checkpoint is not an object")
        created = int(checkpoint["created_unix_ns"])
        _require(created >= last_checkpoint_created, "checkpoint chronology moved backward")
        last_checkpoint_created = created
        evidence_path = policy.path(checkpoint["evidence_path"], scope="study")
        _require(sha256_file(evidence_path) == checkpoint["evidence_sha256"], "checkpoint evidence hash drift")
        evidence = read_json(evidence_path)
        is_analysis_failure_checkpoint = bool(
            analysis_execution_invalid and completed_state == "SEALED_ANALYSIS"
        )
        if is_analysis_failure_checkpoint:
            _require(
                checkpoint["evidence_path"] == contract["_analysis_execution_invalid_relative"],
                "analysis-failure checkpoint path drift",
            )
            _require(
                evidence.get("passed") is False
                and evidence.get("execution_invalid") is True
                and evidence.get("integrity_failure") is True
                and evidence.get("process_valid") is False,
                "analysis-failure checkpoint does not prove integrity failure",
            )
        else:
            _require(evidence.get("passed") is True, f"checkpoint {completed_state} did not pass")
        expected_source_attempt = (
            SCIENCE_ATTEMPT
            if completed_state in pre_data_completed
            else (
                SOURCE_ATTEMPT
                if completed_state in INHERITED_COMPLETED_STATES
                else ACTIVE_ATTEMPT
            )
        )
        _require(
            checkpoint.get("source_attempt") == expected_source_attempt,
            f"checkpoint {completed_state} source-attempt lineage drift",
        )
        _require(
            evidence.get("attempt") == expected_source_attempt,
            f"checkpoint {completed_state} attempt drift",
        )
        _require(evidence.get("checkpoint_state") == completed_state, f"checkpoint {completed_state} state drift")
        _require(int(evidence.get("created_unix_ns", 0)) <= created, "checkpoint predates its evidence chronology")
        checkpoint_by_state[completed_state] = checkpoint
    _require(state.get("last_verified_checkpoint") == checkpoints[-1], "last checkpoint pointer drift")
    _require(state.get("ledger_event_count") == len(events), "state/ledger event-count mismatch")
    _require(state.get("ledger_head_sha256") == previous, "state/ledger head mismatch")
    _require(int(state.get("updated_unix_ns", 0)) >= last_created, "state timestamp predates ledger head")
    controller_adapter = _verify_durable_controller_adapter_marker(
        state,
        ledger_event_count=len(events),
        ledger_head_sha256=previous,
        policy=policy,
    )

    state_events = [event for event in events if event.get("event") == "state_completed"]
    _require(len(state_events) == len(expected_completed), "ledger state-completion count drift")
    for completed_state, event in zip(expected_completed, state_events, strict=True):
        checkpoint = checkpoint_by_state[completed_state]
        _require(event.get("completed_state") == completed_state, "ledger completed state order drift")
        expected_event_attempt = (
            SCIENCE_ATTEMPT
            if completed_state in pre_data_completed
            else (
                SOURCE_ATTEMPT
                if completed_state in INHERITED_COMPLETED_STATES
                else ACTIVE_ATTEMPT
            )
        )
        _require(event.get("attempt") == expected_event_attempt, "ledger completed-state attempt drift")
        next_state = (
            "INDEPENDENT_VERIFICATION"
            if analysis_execution_invalid and completed_state == "SEALED_ANALYSIS"
            else STATE_MACHINE[STATE_MACHINE.index(completed_state) + 1]
        )
        _require(event.get("next_state") == next_state, "ledger next-state drift")
        _require(event.get("evidence_path") == checkpoint["evidence_path"], "ledger checkpoint path drift")
        _require(event.get("evidence_sha256") == checkpoint["evidence_sha256"], "ledger checkpoint hash drift")
        if analysis_execution_invalid and completed_state == "SEALED_ANALYSIS":
            _require(
                event.get("postconfirmation_integrity_failure") is True
                and event.get("integrity_source") == "analysis_execution"
                and event.get("skipped_states") == expected_skipped,
                "analysis-failure ledger staging event drift",
            )
    if mode != "confirmation":
        staged = [event for event in events if event.get("event") == "preregistered_early_scientific_failure_staged"]
        _require(len(staged) == 1, "early-failure ledger staging event drift")
        _require(staged[0].get("mode") == mode and staged[0].get("trigger_state") == early_triggers[mode], "early-failure ledger identity drift")
        _require(staged[0].get("skipped_states") == expected_skipped, "early-failure ledger skipped-state drift")

    current = STATE_MACHINE[0]
    reconstructed_counts: dict[str, Any] = {
        "fit_outcome_episodes": 0,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    allowed_states = {
        "fit_outcome_episodes": {"FIT_COHORTS"},
        "selection_outcome_episodes": {"SELECTION_COHORTS"},
        "smoke_outcome_episodes": {"EXCLUDED_MECHANICAL_SMOKE"},
        "confirmation_outcome_episodes_generated": {"CONFIRMATION_GENERATION", "CONFIRMATION_EXECUTION"},
        "confirmation_outcome_episodes_executed": {"CONFIRMATION_EXECUTION", "CONFIRMATION_INPUT_SEAL"},
        "confirmation_outcomes_opened_for_analysis": {"SEALED_ANALYSIS"},
    }
    for event in events:
        if event.get("event") == "state_completed":
            _require(event.get("completed_state") == current, "ledger transition is out of order")
            current = str(event["next_state"])
        elif event.get("event") == "outcome_counts_updated":
            fields = event.get("fields")
            _require(isinstance(fields, Mapping) and bool(fields), "count event lacks fields")
            for key, value in fields.items():
                _require(key in allowed_states and current in allowed_states[key], f"counter {key} changed in {current}")
                old = reconstructed_counts[key]
                if isinstance(old, bool):
                    _require(not (old is True and value is False), f"counter {key} moved backward")
                else:
                    _require(int(value) >= int(old), f"counter {key} decreased")
                reconstructed_counts[key] = value
    for key, expected in reconstructed_counts.items():
        _require(state.get(key) == expected, f"state counter {key} differs from ledger")
    return {
        "event_count": len(events),
        "legacy_event_count": legacy_count,
        "ledger_head_sha256": previous,
        "completed_state_count": len(expected_completed),
        "state_updated_unix_ns": int(state["updated_unix_ns"]),
        "state": state,
        "events": events,
        "version_forward_edge_index": edge_index,
        "version_forward_edge": edge_event,
        "prior_version_forward_edge_index": prior_edge_index,
        "prior_version_forward_edge": prior_edge_event,
        "durable_controller_adapter": controller_adapter,
        "ledger_segments": {
            "v001_event_count": edge_positions[0],
            "version_forward_event_count": len(edge_positions),
            "v008_event_count_after_edge": len(events) - edge_index - 1,
        },
    }


class _AdministrativeAstNormalizer(ast.NodeTransformer):
    """Independent v001/v008 administrative-parameter normalizer."""

    _attempt_names = {"ATTEMPT", "ACTIVE_ATTEMPT"}

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802 - ast API
        if isinstance(node.ctx, ast.Load) and node.id in self._attempt_names:
            return ast.copy_location(ast.Constant(value="<ATTEMPT>"), node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802 - ast API
        if not isinstance(node.value, str):
            return node
        value = node.value
        parts = value.split("/")
        if (
            len(parts) == 4
            and parts[0] == "runs"
            and parts[1].startswith("lewm_v5_")
            and parts[2] == "attempts"
            and parts[3] == "v005"
        ):
            value = "v008"
        value = value.replace(
            "audit/pre_data_inheritance_seal.json", "<PRE_DATA_SEAL>"
        )
        value = re.sub(r"v00[234567]", "<ACTIVE_ATTEMPT>", value)
        if value == node.value:
            return node
        return ast.copy_location(ast.Constant(value=value), node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:  # noqa: N802 - ast API
        visited = self.generic_visit(node)
        pieces: list[str] = []
        for item in visited.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                pieces.append(item.value)
            elif isinstance(item, ast.FormattedValue) and item.conversion == -1 and item.format_spec is None and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                pieces.append(item.value.value)
            else:
                return visited
        return ast.copy_location(ast.Constant("".join(pieces)), node)


def _assignment_names(node: ast.stmt) -> set[str]:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    result: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            result.add(target.id)
    return result


def _is_active_attempt_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or node.orelse or not node.body:
        return False
    test = node.test
    return bool(
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "ACTIVE_ATTEMPT"
        and len(test.ops) == 1
        and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value
        in {ACTIVATION_SOURCE_ATTEMPT, ACTIVE_ATTEMPT}
        and all(isinstance(item, ast.Raise) for item in node.body)
    )


def _normalize_normative_sources(node: ast.stmt) -> ast.stmt:
    names = _assignment_names(node)
    if names not in ({"NORMATIVE_SOURCES"}, {"REQUIRED_IMPLEMENTATION_FILES"}) or not isinstance(node, ast.Assign):
        return node
    collection = node.value
    if isinstance(collection, ast.Call) and collection.args:
        collection = collection.args[0]
    if not isinstance(collection, (ast.List, ast.Tuple, ast.Set)):
        return node
    collection.elts = [
        item for item in collection.elts
        if not (
            isinstance(item, ast.Constant)
            and isinstance(item.value, str)
            and item.value in NEW_LINEAGE_SUPPORT_PATHS
        )
    ]
    return node


def normalized_administrative_ast_sha256(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    tree.body = [
        _normalize_normative_sources(node)
        for node in tree.body
        if not (_assignment_names(node) & _AdministrativeAstNormalizer._attempt_names)
        and not _is_active_attempt_guard(node)
    ]
    normalized = _AdministrativeAstNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    payload = ast.dump(normalized, annotate_fields=True, include_attributes=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonicalize_contract(value: Any) -> Any:
    _require(isinstance(value, Mapping), "contract canonicalization input drift")
    attempt = value.get("attempt")
    lineage = value.get("lineage")
    _require(
        attempt in {ACTIVATION_SOURCE_ATTEMPT, ACTIVE_ATTEMPT}
        and isinstance(lineage, Mapping),
        "contract canonicalization identity drift",
    )
    common = {
        "science_attempt": SCIENCE_ATTEMPT,
        "invalidity_evidence": SCIENCE_INVALIDITY_RELATIVE,
        "source_pre_data_seal": SCIENCE_PRE_DATA_RELATIVE,
    }
    source_attempts = [
        SCIENCE_ATTEMPT,
        PRIOR_ATTEMPT,
        FIT_SOURCE_ATTEMPT,
        INTERMEDIATE_ATTEMPT,
        SOURCE_PARENT_ATTEMPT,
        SOURCE_ATTEMPT,
        ACTIVATION_SOURCE_ATTEMPT,
    ]
    source_lineage = {
        **common,
        "active_attempt": ACTIVATION_SOURCE_ATTEMPT,
        "attempts": source_attempts,
        "source_roots": {
            item: ATTEMPT_ROOTS[item] for item in source_attempts
        },
        "pre_data_inheritance_seal": SOURCE_PRE_DATA_RELATIVE,
        "prior_pre_data_inheritance_seal": (
            f"{ATTEMPT_ROOTS[SOURCE_ATTEMPT]}/audit/"
            "pre_data_inheritance_seal.json"
        ),
        "source_invalidity_evidence": (
            f"{ATTEMPT_ROOTS[SOURCE_ATTEMPT]}/audit/"
            "v006_procedural_invalidity.json"
        ),
        "superseded_source_invalidity_draft": (
            f"{ATTEMPT_ROOTS[SOURCE_ATTEMPT]}/audit/"
            "v006_procedural_invalidity_draft.json"
        ),
        "prior_version_forward_receipt": (
            f"{ATTEMPT_ROOTS[SOURCE_ATTEMPT]}/audit/"
            "version_forward_transaction_receipt.json"
        ),
        "version_forward_receipt": SOURCE_RECEIPT_RELATIVE,
    }
    target_lineage = {
        **common,
        "active_attempt": ACTIVE_ATTEMPT,
        "attempts": [*source_attempts, ACTIVE_ATTEMPT],
        "source_roots": ATTEMPT_ROOTS,
        "pre_data_inheritance_seal": INHERITANCE_SEAL_RELATIVE,
        "prior_pre_data_inheritance_seal": SOURCE_PRE_DATA_RELATIVE,
        "source_invalidity_evidence": INVALIDITY_RELATIVE,
        "superseded_source_invalidity_draft": INVALIDITY_DRAFT_RELATIVE,
        "prior_version_forward_receipt": SOURCE_RECEIPT_RELATIVE,
        "version_forward_receipt": (
            f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/audit/"
            "version_forward_transaction_receipt.json"
        ),
    }
    expected = (
        source_lineage if attempt == ACTIVATION_SOURCE_ATTEMPT else target_lineage
    )
    _require(dict(lineage) == expected, "contract transitive lineage drift")

    def project(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): project(child) for key, child in item.items()}
        if isinstance(item, list):
            return [project(child) for child in item]
        if isinstance(item, str):
            result = item.replace(
                f"/attempts/{ACTIVE_ATTEMPT}/",
                f"/attempts/{ACTIVATION_SOURCE_ATTEMPT}/",
            ).replace(
                f"/attempts/{ACTIVE_ATTEMPT}",
                f"/attempts/{ACTIVATION_SOURCE_ATTEMPT}",
            )
            return result
        return item

    result = project(value)
    result["attempt"] = ACTIVATION_SOURCE_ATTEMPT
    result["attempt_root"] = ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]
    result["lineage"] = source_lineage
    return result


def canonical_contract_sha256(path: Path) -> str:
    value = read_json(path)
    canonical = _canonicalize_contract(value)
    return canonical_object_sha256(canonical)


def _lineage_file_link(
    value: Any,
    *,
    expected_relative: str,
    policy: PathPolicy,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, "lineage file-link schema drift")
    _require(value.get("path") == expected_relative, "lineage file-link path drift")
    path = policy.path(expected_relative, scope="study")
    digest = sha256_file(path)
    _require(value.get("sha256") == digest, "lineage file-link hash drift")
    return {"path": expected_relative, "sha256": digest, "object": read_json(path)}


def _top_level_symbol_hashes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, str] = {}
    occurrences: dict[str, int] = {}

    def target_identity(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{target_identity(node.value)}.{node.attr}"
        if isinstance(node, (ast.Tuple, ast.List)):
            return "[" + ",".join(target_identity(item) for item in node.elts) + "]"
        return ast.dump(node, annotate_fields=True, include_attributes=False)

    def base_label(node: ast.AST, index: int) -> str:
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return "docstring"
        if isinstance(node, ast.Import):
            identity = ",".join(
                item.name + (f" as {item.asname}" if item.asname else "")
                for item in node.names
            )
            return f"import:{identity}"
        if isinstance(node, ast.ImportFrom):
            identity = ",".join(
                item.name + (f" as {item.asname}" if item.asname else "")
                for item in node.names
            )
            return f"import_from:{'.' * node.level}{node.module or ''}:{identity}"
        if isinstance(node, ast.FunctionDef):
            return f"function:{node.name}"
        if isinstance(node, ast.AsyncFunctionDef):
            return f"async_function:{node.name}"
        if isinstance(node, ast.ClassDef):
            return f"class:{node.name}"
        if isinstance(node, ast.Assign):
            return "assign:" + ",".join(
                target_identity(item) for item in node.targets
            )
        if isinstance(node, ast.AnnAssign):
            return f"annassign:{target_identity(node.target)}"
        if isinstance(node, ast.AugAssign):
            return f"augassign:{target_identity(node.target)}"
        return type(node).__name__.lower()

    for index, node in enumerate(tree.body):
        base = base_label(node, index)
        occurrences[base] = occurrences.get(base, 0) + 1
        label = f"{base}#{occurrences[base]:03d}"
        _require(label not in result, f"duplicate top-level unit label: {path}: {label}")
        result[label] = hashlib.sha256(
            ast.dump(
                node, annotate_fields=True, include_attributes=False
            ).encode()
        ).hexdigest()
    _require(len(result) == len(tree.body), f"top-level unit census incomplete: {path}")
    return result


def _independent_lineage_support_scope(
    source: Path, target: Path, relative: str
) -> dict[str, Any]:
    policy = LINEAGE_SUPPORT_UNIT_POLICY.get(relative)
    policy_keys = {
        "source_top_level_unit_count", "target_top_level_unit_count",
        "source_top_level_unit_sequence_sha256",
        "target_top_level_unit_sequence_sha256",
        "required_changed_top_level_units", "allowed_changed_top_level_units",
        "inserted_top_level_units", "deleted_top_level_units",
    }
    _require(
        isinstance(policy, dict) and set(policy) == policy_keys,
        f"independent lineage unit policy absent or malformed: {relative}",
    )
    left = _top_level_symbol_hashes(source)
    right = _top_level_symbol_hashes(target)
    source_sequence_sha256 = canonical_object_sha256(list(left))
    target_sequence_sha256 = canonical_object_sha256(list(right))
    changed = sorted(
        label
        for label in set(left) | set(right)
        if left.get(label) != right.get(label)
    )
    inserted = sorted(set(right) - set(left))
    deleted = sorted(set(left) - set(right))
    required = list(policy["required_changed_top_level_units"])
    allowed = list(policy["allowed_changed_top_level_units"])
    _require(
        set(LINEAGE_SUPPORT_UNIT_POLICY) == set(LINEAGE_SUPPORT_PATHS)
        and policy["source_top_level_unit_count"] == len(left)
        and policy["target_top_level_unit_count"] == len(right)
        and policy["source_top_level_unit_sequence_sha256"]
        == source_sequence_sha256
        and policy["target_top_level_unit_sequence_sha256"]
        == target_sequence_sha256
        and required == sorted(required)
        and allowed == sorted(allowed)
        and required == allowed == changed
        and list(policy["inserted_top_level_units"]) == inserted
        and list(policy["deleted_top_level_units"]) == deleted
        and not any("*" in label for label in allowed),
        f"independent lineage top-level unit policy drift: {relative}",
    )
    allowed_set = set(allowed)
    _require(
        all(
            left[label] == right[label]
            for label in set(left) & set(right)
            if label not in allowed_set
        ),
        f"independent lineage nonallowed unit changed: {relative}",
    )
    unchanged = [
        [label, digest]
        for label, digest in left.items()
        if label in right and digest == right[label]
    ]
    return {
        "unit_policy_id": "ordered_occurrence_qualified_module_body_ast_v1",
        "source_top_level_unit_count": len(left),
        "target_top_level_unit_count": len(right),
        "source_top_level_unit_sequence_sha256": source_sequence_sha256,
        "target_top_level_unit_sequence_sha256": target_sequence_sha256,
        "required_changed_top_level_units": required,
        "allowed_changed_top_level_units": allowed,
        "observed_changed_top_level_units": changed,
        "inserted_top_level_units": inserted,
        "deleted_top_level_units": deleted,
        "unchanged_top_level_unit_count": len(unchanged),
        "unchanged_top_level_units_sha256": canonical_object_sha256(unchanged),
    }


def _closure_literal_set(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        _require(node.func.id == "frozenset" and len(node.args) == 1 and not node.keywords, "nonliteral closure call")
        return _closure_literal_set(node.args[0])
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise VerificationError("nonliteral manifest closure") from error
    _require(isinstance(value, (set, frozenset, list, tuple)) and all(isinstance(item, str) for item in value), "invalid manifest closure collection")
    return set(value)


def _independent_manifest_policy(target: Path) -> dict[str, set[str]]:
    tree = ast.parse((target / "build_manifest.py").read_text(encoding="utf-8"))
    names = {"ALLOWED_SOURCE_SUFFIXES", "PRE_DATA_STATIC_RELATIVE_PATHS", "PRE_DATA_REPOSITORY_RELATIVE_PATHS", "PRE_DATA_AUDIT_INPUTS", "GENERATED_AUDIT_NAMES", "GENERATED_ATTEMPT_PRODUCTS", "GENERATED_ROLES", "GENERATED_REGIMES", "MUTABLE_STUDY_ROOT_RELATIVE_PATHS"}
    observed: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            observed[node.targets[0].id] = _closure_literal_set(node.value)
    expected = {
        "ALLOWED_SOURCE_SUFFIXES": set(EXPECTED_ALLOWED_SOURCE_SUFFIXES),
        "PRE_DATA_STATIC_RELATIVE_PATHS": set(EXPECTED_PRE_DATA_STATIC_PATHS),
        "PRE_DATA_REPOSITORY_RELATIVE_PATHS": set(EXPECTED_REPOSITORY_PATHS),
        "PRE_DATA_AUDIT_INPUTS": set(EXPECTED_PRE_DATA_AUDIT_INPUTS),
        "GENERATED_AUDIT_NAMES": set(EXPECTED_GENERATED_AUDIT_PATHS),
        "GENERATED_ATTEMPT_PRODUCTS": set(EXPECTED_GENERATED_ATTEMPT_PRODUCTS),
        "GENERATED_ROLES": set(EXPECTED_GENERATED_ROLES),
        "GENERATED_REGIMES": set(EXPECTED_GENERATED_REGIMES),
        "MUTABLE_STUDY_ROOT_RELATIVE_PATHS": set(EXPECTED_MUTABLE_REPOSITORY_PATHS),
    }
    _require(observed == expected, f"independent manifest closure policy drift: {sorted(name for name in names if observed.get(name) != expected[name])}")
    return observed


def _independent_cache_or_near_miss(relative: str) -> bool:
    for part in Path(relative).parts:
        lowered = part.casefold()
        if "cache" in lowered or lowered.endswith((".pyc", ".pyo")):
            return True
        if lowered.endswith(("~", ".bak", ".orig", ".rej", ".swp", ".swo", ".tmp", ".temp")):
            return True
        if any(f"{suffix}." in lowered for suffix in EXPECTED_ALLOWED_SOURCE_SUFFIXES):
            return True
    return False


def _independent_scope_audit(study: Path) -> dict[str, Any]:
    linked: list[str] = []; invalid: list[str] = []; cache: list[str] = []
    multi: list[str] = []; aliases: list[tuple[str, str]] = []
    seen: dict[tuple[int, int], str] = {}
    for root, directories, filenames in os.walk(study, topdown=True, followlinks=False):
        base = Path(root); retained: list[str] = []
        for name in directories:
            path = base / name; relative = path.relative_to(study).as_posix()
            try: metadata = path.lstat()
            except FileNotFoundError: invalid.append(relative); continue
            if stat.S_ISLNK(metadata.st_mode): linked.append(relative); continue
            if not stat.S_ISDIR(metadata.st_mode): invalid.append(relative); continue
            if _independent_cache_or_near_miss(relative): cache.append(relative); continue
            retained.append(name)
        directories[:] = retained
        for name in filenames:
            path = base / name; relative = path.relative_to(study).as_posix()
            try: metadata = path.lstat()
            except FileNotFoundError: invalid.append(relative); continue
            if stat.S_ISLNK(metadata.st_mode): linked.append(relative); continue
            if not stat.S_ISREG(metadata.st_mode): invalid.append(relative); continue
            if _independent_cache_or_near_miss(relative): cache.append(relative)
            if metadata.st_nlink != 1: multi.append(relative)
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            if identity in seen: aliases.append((seen[identity], relative))
            else: seen[identity] = relative
    _require(not (linked or invalid or cache or multi or aliases), f"independent complete sealed-scope filesystem audit failed: linked={sorted(linked)}, invalid={sorted(invalid)}, cache_or_near_miss={sorted(cache)}, multi_link={sorted(multi)}, inode_aliases={sorted(aliases)}")
    return {"complete_scope_live_walk": True, "linked": [], "invalid_entries": [], "cache_or_near_miss": [], "multi_link_files": [], "inode_aliases": []}


def _independent_root_namespace(policy: PathPolicy, constants: Mapping[str, set[str]]) -> dict[str, Any]:
    study = policy.study_root
    required = {Path(raw).name for raw in constants["PRE_DATA_REPOSITORY_RELATIVE_PATHS"]}
    mutable = {Path(raw).name for raw in constants["MUTABLE_STUDY_ROOT_RELATIVE_PATHS"]}
    operational = set(EXPECTED_STUDY_ROOT_OPERATIONAL_NAMES); directories = set(EXPECTED_STUDY_ROOT_DIRECTORY_NAMES)
    actual = {path.name for path in study.iterdir()}
    unexpected = sorted(actual - required - mutable - operational - directories)
    missing = sorted((required | {"STATE.json", "RESEARCH_LEDGER.jsonl"} | directories) - actual)
    attempts = study / "attempts"; attempt_names = {path.name for path in attempts.iterdir()} if attempts.is_dir() else set()
    _require(not unexpected and not missing and "STATE_TRANSACTION.json" not in actual and attempt_names == set(ALLOWED_ATTEMPTS), f"independent study-root namespace drift: missing={missing}, unexpected={unexpected}, transaction_residue={'STATE_TRANSACTION.json' in actual}, attempts={sorted(attempt_names)}")
    return {"declared_mutable_repository_paths": sorted(constants["MUTABLE_STUDY_ROOT_RELATIVE_PATHS"]), "declared_operational_repository_names": sorted(operational), "declared_study_root_directories": sorted(directories), "study_root_namespace_complete": True, "study_root_unexpected": [], "state_transaction_present": False}


def _independent_episode_ids(target: Path) -> dict[tuple[str, str], set[str]]:
    ledger = read_json(target / "cohort_seed_ledger.json"); regimes = ledger.get("regimes")
    _require(isinstance(regimes, Mapping), "independent cohort regimes absent")
    result: dict[tuple[str, str], set[str]] = {}
    for regime in EXPECTED_GENERATED_REGIMES:
        regime_value = regimes.get(regime); roles = regime_value.get("roles") if isinstance(regime_value, Mapping) else None
        _require(isinstance(roles, Mapping), f"independent cohort roles absent: {regime}")
        for role in EXPECTED_GENERATED_ROLES:
            value = roles.get(role); _require(isinstance(value, Mapping), f"independent cohort role absent: {role}/{regime}")
            records = [*(value.get("primary") or ()), *(value.get("replacements") or ())]
            identifiers = {str(item.get("episode_id")) for item in records if isinstance(item, Mapping)}
            _require(bool(identifiers) and len(identifiers) == len(records), f"independent cohort IDs invalid: {role}/{regime}")
            result[(role, regime)] = identifiers
    return result


def _independent_data_json(relative: str, identifiers: Mapping[tuple[str, str], set[str]]) -> tuple[bool, str | None]:
    parts = Path(relative).parts
    if Path(relative).suffix.lower() != ".json" or not parts or parts[0] != "data": return False, None
    if parts == ("data", "replacement_registry.json"): return True, None
    if len(parts) == 3 and parts[:2] == ("data", "replacement_claims"):
        stem = Path(parts[2]).stem
        return Path(parts[2]).suffix == ".json" and len(stem) == 6 and all(character in "0123456789" for character in stem), None
    if len(parts) == 4 and parts[1] in EXPECTED_GENERATED_ROLES and parts[2] in EXPECTED_GENERATED_REGIMES: return parts[3] in {"raw_manifest.json", "execution_manifest.json"}, parts[1]
    if len(parts) == 5 and parts[1] in EXPECTED_GENERATED_ROLES and parts[2] in EXPECTED_GENERATED_REGIMES and parts[3] in {"raw", "execution"}: return Path(parts[4]).stem in identifiers.get((parts[1], parts[2]), set()), parts[1]
    if len(parts) == 5 and parts[:2] == ("data", "persistence_intents") and parts[2] in EXPECTED_GENERATED_ROLES and parts[3] in EXPECTED_GENERATED_REGIMES: return Path(parts[4]).stem in identifiers.get((parts[2], parts[3]), set()), parts[2]
    return False, None


def _independent_data_path(relative: str, identifiers: Mapping[tuple[str, str], set[str]]) -> tuple[bool, str | None]:
    data_json, role = _independent_data_json(relative, identifiers)
    if data_json: return True, role
    parts = Path(relative).parts
    if parts == ("data", "replacement_registry.lock"): return True, None
    if len(parts) == 4 and parts[0] == "data" and parts[1] in EXPECTED_GENERATED_ROLES and parts[2] in EXPECTED_GENERATED_REGIMES and parts[3] == "role.npz": return True, parts[1]
    if len(parts) == 5 and parts[0] == "data" and parts[1] in EXPECTED_GENERATED_ROLES and parts[2] in EXPECTED_GENERATED_REGIMES and parts[3] in {"raw", "execution"} and Path(parts[4]).suffix == ".npz":
        return Path(parts[4]).stem in identifiers.get((parts[1], parts[2]), set()), parts[1]
    match = re.fullmatch(r"\.materialization-(fit|selection|smoke|confirmation)-(native_plan|markov_oracle|plan_action_noise_0p2|plan_random_action_0p1)\.lock", relative)
    return (True, match.group(1)) if match else (False, None)


def _independent_generated_policy(state: Mapping[str, Any], phase: str) -> tuple[set[str], set[str]]:
    base = {
        "audit/inherited_fit_inventory.json",
        "audit/pre_data_inheritance_seal.json",
    }
    if phase == "pre-forward": return set(), base
    base.add("audit/version_forward_transaction_receipt.json")
    current = state.get("current_state")
    fit = {"audit/fit_cohorts.json", "audit/fit_lock_checkpoint.json", "fit/fitted_candidates.npz", "fit/fit_lock.json", "metrics/fit_power_summary.json"}
    preselection = {"audit/pre_selection_manifest.json", "audit/pre_selection_seal.json"}
    selection = {"audit/selection_cohorts.json", "audit/candidate_selection.json", "selection/selection_ledger.json", "metrics/selection_power_summary.json"}
    freeze = {"audit/gate_freeze_checkpoint.json", "freeze/gate_fit.npz", "freeze/compiled_gate.npz", "freeze/compiled_gate_manifest.json", "freeze/gate_freeze.json"}
    power = {"audit/confirmation_power_and_cohort_freeze.json", "power_analysis.json"}
    if current == "FIT_COHORTS": return {"fit"}, base | {"audit/fit_cohorts.json"}
    if current == "SELECTION_COHORTS": return {"fit", "selection"}, base | fit | preselection | {"audit/selection_cohorts.json"}
    if current == "CANDIDATE_SELECTION": return {"fit", "selection"}, base | fit | preselection | selection
    if current == "CONFIRMATION_POWER_AND_COHORT_FREEZE": return {"fit", "selection"}, base | fit | preselection | selection | freeze | power
    raise VerificationError(f"independent closure has no policy for state: {current}")


def _independent_generated_path(relative: str, identifiers: Mapping[tuple[str, str], set[str]], state: Mapping[str, Any], phase: str) -> bool:
    roles, exact = _independent_generated_policy(state, phase)
    if relative in exact: return True
    if phase == "pre-forward": return False
    data, role = _independent_data_path(relative, identifiers)
    if data: return role is None or role in roles
    match = re.fullmatch(r"audit/(fit|selection|smoke|confirmation)_(native_plan|markov_oracle|plan_action_noise_0p2|plan_random_action_0p1)_execution_failure\.json", relative)
    return bool(match and match.group(1) in roles)


def _independent_fit_inventory(policy: PathPolicy) -> dict[str, Any]:
    """Rehash the inherited fit namespace without decoding an array."""

    path = policy.path(FIT_INVENTORY_RELATIVE, scope="study")
    value = read_json(path)
    expected_keys = {
        "schema_version", "artifact_type", "attempt", "source_attempt",
        "source_state", "created_unix_ns", "episode_count",
        "episodes_per_regime", "regime_order", "regimes", "file_count",
        "files", "files_canonical_sha256", "source_invalidity",
        "source_pre_data_seal", "source_version_forward_receipt",
        "source_cohort_seed_ledger", "source_replacement_registry",
        "source_controller_state_sha256", "source_research_ledger_sha256",
        "controller_fit_outcome_episodes_at_inventory",
        "later_role_artifact_file_count", "replacement_record_count",
        "dataset_role_separation_verified", "source_bytes_rehashed",
        "outcome_arrays_opened", "passed",
    }
    _require(set(value) == expected_keys, "independent fit inventory schema drift")
    _require(
        path.read_bytes()
        == (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
        "independent fit inventory canonical encoding drift",
    )
    records = value.get("files")
    regime_order = [
        "native_plan", "markov_oracle", "plan_action_noise_0p2",
        "plan_random_action_0p1",
    ]
    _require(
        type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("artifact_type") == "immutable_fit_role_inheritance_inventory"
        and value.get("attempt") == ACTIVE_ATTEMPT
        and value.get("source_attempt") == FIT_SOURCE_ATTEMPT
        and value.get("source_state") == "FIT_COHORTS"
        and type(value.get("created_unix_ns")) is int
        and value.get("created_unix_ns") > 0
        and type(value.get("file_count")) is int
        and value.get("file_count") == EXPECTED_FIT_INVENTORY_FILE_COUNT
        and type(value.get("episode_count")) is int
        and value.get("episode_count") == EXPECTED_FIT_EPISODE_COUNT
        and type(value.get("episodes_per_regime")) is int
        and value.get("episodes_per_regime") == 300
        and value.get("regime_order") == regime_order
        and value.get("source_controller_state_sha256")
        == EXPECTED_FIT_SOURCE_STATE_SHA256
        and value.get("source_research_ledger_sha256")
        == EXPECTED_FIT_SOURCE_LEDGER_SHA256
        and value.get("controller_fit_outcome_episodes_at_inventory") == 0
        and value.get("later_role_artifact_file_count") == 0
        and value.get("replacement_record_count") == 0
        and value.get("dataset_role_separation_verified") is True
        and value.get("source_bytes_rehashed") is True
        and value.get("outcome_arrays_opened") is False
        and value.get("passed") is True
        and isinstance(records, list)
        and len(records) == EXPECTED_FIT_INVENTORY_FILE_COUNT,
        "independent fit inventory header/census drift",
    )
    regimes = value.get("regimes")
    _require(
        isinstance(regimes, Mapping) and set(regimes) == set(regime_order),
        "independent fit inventory regime census drift",
    )
    source_prefix = f"{ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]}/"
    allowed_prefixes = (
        source_prefix + "data/fit/",
        source_prefix + "data/persistence_intents/fit/",
    )
    singleton = source_prefix + "data/replacement_registry.json"
    paths: list[str] = []
    inodes: set[tuple[int, int]] = set()
    for record in records:
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "bytes", "sha256"},
            "independent fit inventory record schema drift",
        )
        raw = record.get("path")
        _require(
            isinstance(raw, str)
            and (raw == singleton or raw.startswith(allowed_prefixes))
            and type(record.get("bytes")) is int
            and record.get("bytes") >= 0
            and isinstance(record.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
            f"independent inherited fit record value drift: {raw}",
        )
        candidate = policy.path(raw, scope="study")
        metadata = candidate.lstat()
        _require(
            record.get("bytes") == metadata.st_size
            and record.get("sha256") == sha256_file(candidate)
            and stat.S_ISREG(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_nlink == 1,
            f"independent inherited fit file drift: {raw}",
        )
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        _require(identity not in inodes, "independent inherited fit inode alias")
        inodes.add(identity)
        paths.append(raw)
    _require(
        paths == sorted(paths)
        and len(paths) == len(set(paths))
        and canonical_object_sha256(records)
        == value.get("files_canonical_sha256"),
        "independent fit inventory order/digest drift",
    )
    source_root = policy.repository_root / ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]
    namespace_paths: list[str] = []
    for namespace_root in (
        source_root / "data/fit",
        source_root / "data/persistence_intents/fit",
    ):
        root_metadata = namespace_root.lstat()
        _require(
            not stat.S_ISLNK(root_metadata.st_mode)
            and stat.S_ISDIR(root_metadata.st_mode),
            "independent inherited fit namespace root drift",
        )
        for candidate in namespace_root.rglob("*"):
            metadata = candidate.lstat()
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                "independent inherited fit namespace symlink drift",
            )
            if stat.S_ISDIR(metadata.st_mode):
                continue
            _require(
                stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                "independent inherited fit namespace entry drift",
            )
            namespace_paths.append(
                candidate.relative_to(policy.repository_root).as_posix()
            )
    registry_path = source_root / "data/replacement_registry.json"
    registry_metadata = registry_path.lstat()
    _require(
        not stat.S_ISLNK(registry_metadata.st_mode)
        and stat.S_ISREG(registry_metadata.st_mode)
        and registry_metadata.st_nlink == 1,
        "independent inherited replacement registry identity drift",
    )
    namespace_paths.append(
        registry_path.relative_to(policy.repository_root).as_posix()
    )
    _require(
        sorted(namespace_paths) == paths,
        "independent inherited fit namespace file census drift",
    )
    links = {
        "source_invalidity": EXPECTED_FIT_SOURCE_INVALIDITY_SHA256,
        "source_pre_data_seal": EXPECTED_FIT_SOURCE_SEAL_SHA256,
        "source_version_forward_receipt": EXPECTED_FIT_SOURCE_RECEIPT_SHA256,
        "source_cohort_seed_ledger": EXPECTED_SOURCE_COHORT_LEDGER_SHA256,
        "source_replacement_registry": (
            EXPECTED_FIT_SOURCE_REPLACEMENT_REGISTRY_SHA256
        ),
    }
    for key, digest in links.items():
        record = value.get(key)
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "bytes", "sha256"}
            and record.get("sha256") == digest
            and isinstance(record.get("path"), str)
            and type(record.get("bytes")) is int,
            f"independent fit inventory lineage schema drift: {key}",
        )
        linked = policy.path(record["path"], scope="study")
        _require(
            linked.stat().st_size == record["bytes"]
            and sha256_file(linked) == digest,
            f"independent fit inventory lineage drift: {key}",
        )
    file_records = {record["path"]: record for record in records}
    for regime in regime_order:
        record = regimes[regime]
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {
                "episode_count", "episode_ids_sha256", "raw_manifest",
                "execution_manifest", "aggregate",
            }
            and type(record.get("episode_count")) is int
            and record.get("episode_count") == 300
            and isinstance(record.get("episode_ids_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", record["episode_ids_sha256"])
            is not None,
            f"independent fit inventory regime schema drift: {regime}",
        )
        for label, suffix in (
            ("raw_manifest", "raw_manifest.json"),
            ("execution_manifest", "execution_manifest.json"),
            ("aggregate", "role.npz"),
        ):
            link = record[label]
            expected_path = source_prefix + f"data/fit/{regime}/{suffix}"
            _require(
                isinstance(link, Mapping)
                and set(link) == {"path", "bytes", "sha256"}
                and link.get("path") == expected_path
                and file_records.get(expected_path) == link,
                f"independent fit inventory regime link drift: {regime}/{label}",
            )
    return {
        "path": FIT_INVENTORY_RELATIVE,
        "sha256": sha256_file(path),
        "file_count": EXPECTED_FIT_INVENTORY_FILE_COUNT,
        "episode_count": EXPECTED_FIT_EPISODE_COUNT,
        "source_paths": paths,
        "outcome_arrays_opened": False,
        "passed": True,
    }


def _independent_manifest_closure(policy: PathPolicy) -> dict[str, Any]:
    source = policy.repository_root / ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]; target = policy.attempt_root
    constants = _independent_manifest_policy(target)
    inventory = _independent_fit_inventory(policy)
    state = read_json(policy.study_root / "STATE.json")
    phase = "pre-forward" if state.get("active_attempt") == ACTIVATION_SOURCE_ATTEMPT else "post-forward"
    source_seal = read_json(source / "audit/pre_data_inheritance_seal.json")
    scope = _independent_scope_audit(policy.study_root); root = _independent_root_namespace(policy, constants)
    complete = source_seal.get("complete_source_partition")
    source_static = complete.get("target_paths") if isinstance(complete, Mapping) else None
    _require(
        isinstance(source_static, list)
        and len(source_static) == len(set(source_static))
        and set(source_static)
        == (
            constants["PRE_DATA_STATIC_RELATIVE_PATHS"]
            | constants["PRE_DATA_AUDIT_INPUTS"]
        ),
        "independent v007 static closure absent",
    )
    inventory_prefix = f"{ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]}/"
    _require(
        all(raw.startswith(inventory_prefix) for raw in inventory["source_paths"]),
        "independent inherited fit inventory escaped v003",
    )
    expected_source = set(source_static) | set(SOURCE_SELECTION_BOUNDARY_PRODUCTS) | {
        "audit/pre_data_inheritance_seal.json",
        "audit/version_forward_transaction_receipt.json",
        "audit/inherited_fit_inventory.json",
        "audit/v007_procedural_invalidity_draft.json",
        "audit/v007_procedural_invalidity.json",
    }
    observed_source = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
    expected_source_dirs = {parent.as_posix() for relative in expected_source for parent in Path(relative).parents if parent != Path(".")}; observed_source_dirs = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_dir()}
    _require(
        observed_source == expected_source
        and observed_source_dirs == expected_source_dirs
        and len(observed_source) == 66,
        "independent v007 namespace drift",
    )
    science = policy.repository_root / ATTEMPT_ROOTS[SCIENCE_ATTEMPT]
    science_seal = read_json(science / "audit/pre_data_seal.json")
    science_files = science_seal.get("sealed_files")
    _require(isinstance(science_files, Mapping), "independent v001 sealed_files absent")
    science_prefix = f"{ATTEMPT_ROOTS[SCIENCE_ATTEMPT]}/"
    expected_science = {
        str(raw)[len(science_prefix):]
        for raw in science_files
        if str(raw).startswith(science_prefix)
    } | {"audit/pre_data_seal.json", "audit/v001_procedural_invalidity.json"}
    observed_science = {path.relative_to(science).as_posix() for path in science.rglob("*") if path.is_file()}
    _require(observed_science == expected_science and len(observed_science) == 53, "independent v001 namespace drift")
    identifiers = _independent_episode_ids(target); declared = constants["PRE_DATA_STATIC_RELATIVE_PATHS"] | constants["PRE_DATA_AUDIT_INPUTS"]
    observed: set[str] = set(); generated: set[str] = set()
    for path in target.rglob("*"):
        if not path.is_file(): continue
        relative = path.relative_to(target).as_posix()
        if relative in declared: observed.add(relative)
        elif _independent_generated_path(relative, identifiers, state, phase): generated.add(relative)
        else: raise VerificationError(f"independent unclassified/wrong-phase v008 path: {relative}")
    _require(observed == declared, "independent v008 required namespace drift")
    expected_dirs = {parent.as_posix() for relative in declared | generated for parent in Path(relative).parents if parent != Path(".")}; observed_dirs = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_dir()}
    _require(observed_dirs == expected_dirs, "independent v008 directory namespace drift")
    return {"declared_static_paths": sorted(constants["PRE_DATA_STATIC_RELATIVE_PATHS"]), "declared_audit_inputs": sorted(constants["PRE_DATA_AUDIT_INPUTS"]), "declared_generated_audits": sorted(constants["GENERATED_AUDIT_NAMES"]), "declared_generated_attempt_products": sorted(constants["GENERATED_ATTEMPT_PRODUCTS"]), "declared_generated_roles": sorted(constants["GENERATED_ROLES"]), "declared_generated_regimes": sorted(constants["GENERATED_REGIMES"]), "allowed_source_suffixes": sorted(constants["ALLOWED_SOURCE_SUFFIXES"]), "declared_repository_paths": sorted(constants["PRE_DATA_REPOSITORY_RELATIVE_PATHS"]), "source_attempt_allowed_suffix_path_count": len(observed_source), "inherited_fit_inventory": {key: inventory[key] for key in ("path", "sha256", "file_count", "episode_count", "outcome_arrays_opened", "passed")}, "science_attempt_exact_path_count": len(observed_science), "discovered_paths": sorted(observed), "missing": [], "unexpected": [], **scope, **root, "passed": True}


def _source_manifest_relative_paths(policy: PathPolicy) -> set[str]:
    source_seal = read_json(
        policy.path(SOURCE_PRE_DATA_RELATIVE, scope="study")
    )
    complete = source_seal.get("complete_source_partition")
    target_paths = complete.get("target_paths") if isinstance(complete, Mapping) else None
    _require(
        isinstance(target_paths, list)
        and len(target_paths) == 61
        and target_paths == sorted(target_paths)
        and len(target_paths) == len(set(target_paths)),
        "v007 source static closure drift",
    )
    constants = _independent_manifest_policy(policy.attempt_root)
    _require(
        set(target_paths)
        == (
            constants["PRE_DATA_STATIC_RELATIVE_PATHS"]
            | constants["PRE_DATA_AUDIT_INPUTS"]
        ),
        "v007 source static partition membership drift",
    )
    return set(target_paths)


def verify_equivalence_partitions(seal: Mapping[str, Any], policy: PathPolicy) -> dict[str, Any]:
    _validate_inheritance_seal_scalar_types(seal)
    partitions = seal.get("source_partitions")
    expected_modes = {"exact_hash", "normalized_ast", "canonical_contracts", "procedural_only", "new_lineage_support", "lineage_support"}
    _require(isinstance(partitions, Mapping) and set(partitions) == expected_modes, "source partition schema drift")
    members: dict[str, list[str]] = {mode: [] for mode in expected_modes}
    source_prefix = f"{ATTEMPT_ROOTS[ACTIVATION_SOURCE_ATTEMPT]}/"
    target_prefix = f"{ATTEMPT_ROOTS[ACTIVE_ATTEMPT]}/"

    def literal_policy(path: Path) -> Any:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        values = [
            node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "LINEAGE_SUPPORT_UNIT_POLICY"
            and node.value is not None
        ]
        _require(
            len(values) == 1,
            f"independent lineage unit policy literal count drift: {path}",
        )
        try:
            return ast.literal_eval(values[0])
        except (ValueError, TypeError, SyntaxError) as exc:
            raise VerificationError(
                f"independent lineage unit policy is not literal: {path}"
            ) from exc

    target_root = policy.repository_root / ATTEMPT_ROOTS[ACTIVE_ATTEMPT]
    for owner in (
        "version_forward_prepare.py",
        "verify_version_forward.py",
        "independent_verify.py",
    ):
        _require(
            literal_policy(target_root / owner) == LINEAGE_SUPPORT_UNIT_POLICY,
            f"independent lineage unit policy transcription drift: {owner}",
        )
    for record in partitions["exact_hash"]:
        _require(set(record) == {"relative_path", "source_path", "target_path", "sha256", "bytes", "classification"}, "exact-file record schema drift")
        relative = record["relative_path"]; source = policy.path(record["source_path"], scope="study"); target = policy.path(record["target_path"], scope="study")
        digest = sha256_file(source)
        _require(record["source_path"] == source_prefix + relative and record["target_path"] == target_prefix + relative, "exact-file owner drift")
        _require(digest == sha256_file(target) == record["sha256"] and source.stat().st_size == target.stat().st_size == record["bytes"], "exact-file equivalence drift")
        expected_class = "scientific_object_exact" if relative in SCIENTIFIC_OBJECT_PATHS else "administrative_or_test_exact"
        _require(record["classification"] == expected_class, "exact-file classification drift")
        members["exact_hash"].append(relative)
    for record in partitions["normalized_ast"]:
        _require(set(record) == {"relative_path", "source_path", "target_path", "source_sha256", "target_sha256", "normalized_ast_sha256", "transform_id"}, "normalized-AST record schema drift")
        relative = record["relative_path"]; source = policy.path(record["source_path"], scope="study"); target = policy.path(record["target_path"], scope="study")
        _require(relative in NORMALIZED_AST_PATHS and record["transform_id"] == "v007_to_v008_active_attempt_ast_v1", "normalized-AST allowlist drift")
        left = normalized_administrative_ast_sha256(source); right = normalized_administrative_ast_sha256(target)
        _require(
            record["source_sha256"] == sha256_file(source)
            and record["target_sha256"] == sha256_file(target)
            and left == right == record["normalized_ast_sha256"],
            "normalized-AST equivalence drift",
        )
        members["normalized_ast"].append(relative)
    for record in partitions["canonical_contracts"]:
        _require(set(record) == {"relative_path", "source_path", "target_path", "source_sha256", "target_sha256", "canonical_sha256", "transform_id"}, "canonical-contract record schema drift")
        relative = record["relative_path"]; source = policy.path(record["source_path"], scope="study"); target = policy.path(record["target_path"], scope="study")
        _require(relative in CANONICAL_CONTRACT_PATHS and record["transform_id"] == "v008_transitive_lineage_contract_v1", "canonical-contract allowlist drift")
        left = canonical_contract_sha256(source); right = canonical_contract_sha256(target)
        _require(record["source_sha256"] == sha256_file(source) and record["target_sha256"] == sha256_file(target) and left == right == record["canonical_sha256"], "canonical-contract equivalence drift")
        members["canonical_contracts"].append(relative)
    for record in partitions["procedural_only"]:
        expected_keys = {"relative_path", "source_path", "target_path", "source_sha256", "target_sha256", "change_id", "changed_top_level_symbols", "allowed_top_level_symbols", "unchanged_top_level_symbol_count", "diff_sha256"}
        _require(set(record) == expected_keys, "procedural record schema drift")
        relative = record["relative_path"]; source = policy.path(record["source_path"], scope="study"); target = policy.path(record["target_path"], scope="study")
        _require(relative == "generator.py" and record["change_id"] == "zero_outcome_v008_generator_root_parent_repair_v1", "procedural allowlist drift")
        left = _top_level_symbol_hashes(source); right = _top_level_symbol_hashes(target)
        changed = sorted(name for name in set(left) | set(right) if left.get(name) != right.get(name))
        diff = "".join(difflib.unified_diff(source.read_text().splitlines(keepends=True), target.read_text().splitlines(keepends=True), fromfile=relative, tofile=relative)).encode()
        _require(record["source_sha256"] == sha256_file(source) and record["target_sha256"] == sha256_file(target), "procedural raw hash drift")
        allowed_symbols = set(record["allowed_top_level_symbols"])
        expected_allowed_symbols = frozenset(
            {"function:_assert_safe_directory_chain#001"}
        )
        expected_changed_symbols = frozenset(
            {"function:_assert_safe_directory_chain#001"}
        )
        _require(expected_allowed_symbols is not None, "procedural static symbol allowlist absent")
        _require(expected_changed_symbols is not None, "procedural static required-symbol set absent")
        _require("*" not in allowed_symbols, "procedural wildcard symbol allowlist forbidden")
        _require(
            allowed_symbols == set(expected_allowed_symbols),
            "procedural static symbol allowlist drift",
        )
        _require(
            set(record["changed_top_level_symbols"])
            == set(expected_changed_symbols),
            "procedural required changed-symbol set drift",
        )
        _require(
            record["changed_top_level_symbols"] == changed
            and set(changed) <= allowed_symbols,
            "procedural AST scope drift",
        )
        _require(
            set(changed) == set(expected_changed_symbols),
            "procedural required changed-symbol set drift",
        )
        _require(record["unchanged_top_level_symbol_count"] == sum(name in right and left[name] == right[name] for name in left) and record["diff_sha256"] == hashlib.sha256(diff).hexdigest(), "procedural diff drift")
        members["procedural_only"].append(relative)
    for record in partitions["new_lineage_support"]:
        _require(set(record) == {"relative_path", "target_path", "target_sha256", "bytes", "purpose"}, "new-lineage record schema drift")
        relative = record["relative_path"]; target = policy.path(record["target_path"], scope="study")
        _require(relative in NEW_LINEAGE_SUPPORT_PATHS and record["target_path"] == target_prefix + relative and record["target_sha256"] == sha256_file(target) and record["bytes"] == target.stat().st_size and record["purpose"] == "immutable_v004_fit_role_provenance_and_read_only_path_map", "new-lineage support drift")
        members["new_lineage_support"].append(relative)
    lineage_paths = set(LINEAGE_SUPPORT_PATHS)
    for record in partitions["lineage_support"]:
        expected_record_keys = {
            "relative_path", "source_path", "target_path", "source_sha256",
            "target_sha256", "purpose", "unit_policy_id",
            "source_top_level_unit_count", "target_top_level_unit_count",
            "source_top_level_unit_sequence_sha256",
            "target_top_level_unit_sequence_sha256",
            "required_changed_top_level_units",
            "allowed_changed_top_level_units",
            "observed_changed_top_level_units", "inserted_top_level_units",
            "deleted_top_level_units", "unchanged_top_level_unit_count",
            "unchanged_top_level_units_sha256",
        }
        _require(
            set(record) == expected_record_keys,
            "lineage-support record schema drift",
        )
        relative = record["relative_path"]
        source = policy.path(record["source_path"], scope="study")
        target = policy.path(record["target_path"], scope="study")
        _require(
            relative in lineage_paths
            and record["source_path"] == source_prefix + relative
            and record["target_path"] == target_prefix + relative
            and record["source_sha256"] == sha256_file(source)
            and record["target_sha256"] == sha256_file(target)
            and record["source_sha256"] != record["target_sha256"]
            and record["purpose"]
            == "v008_transitive_lineage_authorization_or_test_only",
            "lineage-support raw binding drift",
        )
        scope = _independent_lineage_support_scope(source, target, relative)
        _require(
            all(record.get(key) == value for key, value in scope.items()),
            f"independent lineage-support unit scope record drift: {relative}",
        )
        members["lineage_support"].append(relative)
    _require(set(members["normalized_ast"]) == NORMALIZED_AST_PATHS, "normalized-AST partition membership drift")
    _require(set(members["canonical_contracts"]) == CANONICAL_CONTRACT_PATHS, "canonical-contract partition membership drift")
    _require(not members["procedural_only"], "procedural partition membership drift")
    _require(
        set(members["new_lineage_support"]) == set(NEW_LINEAGE_SUPPORT_PATHS),
        "target-only lineage support membership drift",
    )
    _require(
        lineage_paths == set(LINEAGE_SUPPORT_UNIT_POLICY)
        == set(members["lineage_support"]),
        "lineage-support partition membership drift",
    )
    complete = seal.get("complete_source_partition")
    source_paths = _source_manifest_relative_paths(policy)
    closure = seal.get("v008_manifest_closure")
    _require(
        isinstance(closure, Mapping)
        and closure == _independent_manifest_closure(policy),
        "v008 manifest closure drift",
    )
    target_paths = set(closure["discovered_paths"])
    flattened = [item for values in members.values() for item in values]
    _require(isinstance(complete, Mapping) and complete.get("source_paths") == sorted(source_paths) and complete.get("target_paths") == sorted(target_paths) and complete.get("partition_members") == members, "complete partition projection drift")
    _require(len(flattened) == len(set(flattened)) and set(flattened) == target_paths and source_paths <= set(flattened) and complete.get("overlap") == complete.get("unpartitioned_source") == complete.get("unpartitioned_target") == [] and complete.get("passed") is True, "source partition is not complete/disjoint")
    _require(seal.get("equivalent_files") == partitions["exact_hash"], "equivalent-file compatibility projection drift")
    return {"source_path_count": len(source_paths), "target_path_count": len(target_paths), "partition_counts": {key: len(value) for key, value in members.items()}}


def _verify_version_forward_lineage_legacy(
    contract: Mapping[str, Any],
    policy: PathPolicy,
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate the v001 -> v002 -> v003 -> v004 -> v005 -> v008 authorization."""

    lineage = contract["lineage"]
    seal_path = policy.path(lineage["pre_data_inheritance_seal"], scope="attempt")
    seal_hash = sha256_file(seal_path)
    seal = read_json(seal_path)
    _validate_inheritance_seal_scalar_types(seal)
    required_equivalence = {
        "source_hash_equivalent": True,
        "normalized_ast_equivalent": True,
        "scientific_object_hash_equivalent": True,
        "configuration_hash_equivalent": True,
        "scientific_changes": False,
        "attempt_parameterization_verified": True,
        "runtime_modules_accept_active_attempt": True,
        "independent_verifier_accepts_active_attempt": True,
    }
    header_checks = {
        "schema": type(seal.get("schema_version")) is int
        and seal.get("schema_version") == 1,
        "artifact_type": seal.get("artifact_type") == "pre_data_inheritance_and_version_forward_equivalence",
        "attempt": seal.get("attempt") == ACTIVE_ATTEMPT,
        "source_attempt": seal.get("source_attempt") == SOURCE_ATTEMPT,
        "target_attempt": seal.get("target_attempt") == ACTIVE_ATTEMPT,
        "science_attempt": seal.get("science_attempt") == SCIENCE_ATTEMPT,
        "checkpoint_state": seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL",
        "resume_state": seal.get("resume_state") == VERSION_FORWARD_RESUME_STATE,
        "authorization_kind": seal.get("authorization_kind") == "zero_outcome_version_forward_inherited_pre_data",
        "procedural_invalidity": seal.get("procedural_invalidity_confirmed") is True,
        "zero_confirmation": seal.get("zero_confirmation_outcomes_at_version_forward") is True,
        "passed": seal.get("passed") is True,
        **{key: seal.get(key) is expected for key, expected in required_equivalence.items()},
    }
    _require(all(header_checks.values()), f"inheritance-seal header drift: {[key for key, value in header_checks.items() if not value]}")
    zero_counts = {
        "fit_outcome_episodes": 0,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    _require(
        isinstance(seal.get("outcome_counts_at_seal"), Mapping)
        and set(seal["outcome_counts_at_seal"]) == set(zero_counts)
        and all(
            type(seal["outcome_counts_at_seal"].get(key)) is type(expected)
            and seal["outcome_counts_at_seal"].get(key) == expected
            for key, expected in zero_counts.items()
        ),
        "inheritance seal outcome counts are not exact zero",
    )
    scan = seal.get("confirmation_artifact_scan")
    _require(scan == {"file_count": 0, "paths": []}, "inheritance confirmation scan is not exact empty")

    invalidity_link = _lineage_file_link(
        seal.get("invalidity_evidence"), expected_relative=INVALIDITY_RELATIVE, policy=policy
    )
    invalidity = invalidity_link["object"]
    controller_counts = invalidity.get("controller_outcome_counts")
    fit_evidence = invalidity.get("fit_evidence")
    _require(
        invalidity.get("attempt") == SOURCE_ATTEMPT
        and invalidity.get("procedural_invalidity") is True
        and invalidity.get("authoritative") is True
        and invalidity.get("scientific_outcomes_conditioned_on") is False
        and invalidity.get("selection_or_later_outcomes_opened") is False
        and invalidity.get("confirmation_outcome_episodes_generated") == 0
        and invalidity.get("confirmation_outcome_episodes_executed") == 0
        and invalidity.get("confirmation_outcomes_opened_for_analysis") is False
        and isinstance(controller_counts, Mapping)
        and dict(controller_counts) == zero_counts
        and isinstance(fit_evidence, Mapping)
        and fit_evidence.get("inherited_episode_count") == 1200
        and fit_evidence.get("inherited_file_count")
        == EXPECTED_FIT_INVENTORY_FILE_COUNT
        and fit_evidence.get("source_attempt") == FIT_SOURCE_ATTEMPT
        and fit_evidence.get("outcome_arrays_opened") is False
        and fit_evidence.get(
            "development_part_arrays_opened_by_failed_checkpoint_verification"
        ) is False
        and fit_evidence.get("controller_count_transition_completed") is False
        and fit_evidence.get(
            "aggregate_arrays_opened_by_failed_checkpoint_verification"
        ) is False,
        "authoritative v005 procedural-invalidity header drift",
    )
    _require(
        invalidity.get("scientific_criterion_changed") is False,
        "v005 invalidity reports scientific criterion changes",
    )
    raw_draft_link = seal.get("superseded_invalidity_draft")
    _require(
        isinstance(raw_draft_link, Mapping)
        and set(raw_draft_link) == {"path", "sha256", "authoritative"}
        and raw_draft_link.get("path") == INVALIDITY_DRAFT_RELATIVE
        and raw_draft_link.get("sha256")
        == sha256_file(policy.path(INVALIDITY_DRAFT_RELATIVE, scope="study"))
        and raw_draft_link.get("authoritative") is False,
        "v005 invalidity draft link drift",
    )
    _require(
        isinstance(invalidity.get("superseded_draft"), Mapping)
        and invalidity["superseded_draft"].get("path")
        == raw_draft_link["path"]
        and invalidity["superseded_draft"].get("sha256")
        == raw_draft_link["sha256"],
        "v005 invalidity does not bind its superseded draft",
    )
    _require(
        seal.get("outcome_arrays_opened") is False
        and seal.get("hdf5_contents_opened") is False
        and seal.get("v3_targets_opened") is False,
        "inheritance preparation opened a forbidden input class",
    )
    source_link = _lineage_file_link(
        seal.get("source_pre_data_seal"), expected_relative=SOURCE_PRE_DATA_RELATIVE, policy=policy
    )
    source_seal = source_link["object"]
    _require(
        source_seal.get("attempt") == SOURCE_ATTEMPT
        and source_seal.get("source_attempt") == INTERMEDIATE_ATTEMPT
        and source_seal.get("target_attempt") == SOURCE_ATTEMPT
        and source_seal.get("science_attempt") == SCIENCE_ATTEMPT
        and source_seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL"
        and source_seal.get("passed") is True
        and source_seal.get("outcome_counts_at_seal") == zero_counts,
        "v005 source inheritance seal drift",
    )
    prior_receipt_link = _lineage_file_link(
        seal.get("source_version_forward_transaction_receipt"),
        expected_relative=SOURCE_RECEIPT_RELATIVE,
        policy=policy,
    )
    prior_receipt = prior_receipt_link["object"]
    _require(
        prior_receipt.get("source_attempt") == INTERMEDIATE_ATTEMPT
        and prior_receipt.get("target_attempt") == SOURCE_ATTEMPT
        and prior_receipt.get("passed") is True,
        "v005 source version-forward receipt header drift",
    )
    intermediate_seal_link = _lineage_file_link(
        source_seal.get("source_pre_data_seal"),
        expected_relative=INTERMEDIATE_PRE_DATA_RELATIVE,
        policy=policy,
    )
    intermediate_seal = intermediate_seal_link["object"]
    _require(
        intermediate_seal.get("attempt") == INTERMEDIATE_ATTEMPT
        and intermediate_seal.get("source_attempt") == FIT_SOURCE_ATTEMPT
        and intermediate_seal.get("target_attempt") == INTERMEDIATE_ATTEMPT
        and intermediate_seal.get("science_attempt") == SCIENCE_ATTEMPT
        and intermediate_seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL"
        and intermediate_seal.get("passed") is True
        and intermediate_seal.get("outcome_counts_at_seal") == zero_counts,
        "v004 intermediate inheritance seal drift",
    )
    intermediate_receipt_link = _lineage_file_link(
        source_seal.get("source_version_forward_transaction_receipt"),
        expected_relative=INTERMEDIATE_RECEIPT_RELATIVE,
        policy=policy,
    )
    _require(
        intermediate_receipt_link["object"].get("source_attempt")
        == FIT_SOURCE_ATTEMPT
        and intermediate_receipt_link["object"].get("target_attempt")
        == INTERMEDIATE_ATTEMPT
        and intermediate_receipt_link["object"].get("passed") is True,
        "v004 version-forward receipt header drift",
    )
    intermediate_invalidity_link = _lineage_file_link(
        source_seal.get("invalidity_evidence"),
        expected_relative=INTERMEDIATE_INVALIDITY_RELATIVE,
        policy=policy,
    )
    fit_seal_link = _lineage_file_link(
        intermediate_seal.get("source_pre_data_seal"),
        expected_relative=FIT_PRE_DATA_RELATIVE,
        policy=policy,
    )
    fit_seal = fit_seal_link["object"]
    _require(
        fit_seal.get("attempt") == FIT_SOURCE_ATTEMPT
        and fit_seal.get("source_attempt") == PRIOR_ATTEMPT
        and fit_seal.get("target_attempt") == FIT_SOURCE_ATTEMPT
        and fit_seal.get("science_attempt") == SCIENCE_ATTEMPT
        and fit_seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL"
        and fit_seal.get("passed") is True
        and fit_seal.get("outcome_counts_at_seal") == zero_counts,
        "v003 fit-source inheritance seal drift",
    )
    fit_receipt_link = _lineage_file_link(
        intermediate_seal.get("source_version_forward_transaction_receipt"),
        expected_relative=FIT_RECEIPT_RELATIVE,
        policy=policy,
    )
    _require(
        fit_receipt_link["object"].get("source_attempt") == PRIOR_ATTEMPT
        and fit_receipt_link["object"].get("target_attempt")
        == FIT_SOURCE_ATTEMPT
        and fit_receipt_link["object"].get("passed") is True,
        "v003 version-forward receipt header drift",
    )
    fit_invalidity_link = _lineage_file_link(
        intermediate_seal.get("invalidity_evidence"),
        expected_relative=FIT_INVALIDITY_RELATIVE,
        policy=policy,
    )
    prior_seal_link = _lineage_file_link(
        fit_seal.get("source_pre_data_seal"),
        expected_relative=PRIOR_PRE_DATA_RELATIVE,
        policy=policy,
    )
    prior_seal = prior_seal_link["object"]
    _require(
        prior_seal.get("attempt") == PRIOR_ATTEMPT
        and prior_seal.get("source_attempt") == SCIENCE_ATTEMPT
        and prior_seal.get("target_attempt") == PRIOR_ATTEMPT
        and prior_seal.get("science_attempt") == SCIENCE_ATTEMPT
        and prior_seal.get("checkpoint_state") == "PRE_OUTCOME_SEAL"
        and prior_seal.get("passed") is True
        and prior_seal.get("outcome_counts_at_seal") == zero_counts,
        "v002 prior inheritance seal drift",
    )
    first_receipt_link = _lineage_file_link(
        fit_seal.get("source_version_forward_transaction_receipt"),
        expected_relative=PRIOR_RECEIPT_RELATIVE,
        policy=policy,
    )
    _require(
        first_receipt_link["object"].get("source_attempt") == SCIENCE_ATTEMPT
        and first_receipt_link["object"].get("target_attempt") == PRIOR_ATTEMPT
        and first_receipt_link["object"].get("passed") is True,
        "v002 version-forward receipt header drift",
    )
    prior_invalidity_link = _lineage_file_link(
        fit_seal.get("invalidity_evidence"),
        expected_relative=PRIOR_INVALIDITY_RELATIVE,
        policy=policy,
    )
    science_pre_data_link = _lineage_file_link(
        prior_seal.get("source_pre_data_seal"),
        expected_relative=SCIENCE_PRE_DATA_RELATIVE,
        policy=policy,
    )
    science_invalidity_link = _lineage_file_link(
        prior_seal.get("invalidity_evidence"),
        expected_relative=SCIENCE_INVALIDITY_RELATIVE,
        policy=policy,
    )
    source_pre_data = {
        "attempt": SOURCE_ATTEMPT,
        "sha256": source_link["sha256"],
        "science_pre_data_sha256": science_pre_data_link["sha256"],
        "passed": True,
    }
    source_files_record = seal.get("source_sealed_files")
    _require(isinstance(source_files_record, Mapping), "source sealed-file provenance absent")
    _require(set(source_files_record) == {"file_count", "files", "all_rehashed"}, "source sealed-file schema drift")
    source_files = _manifest_files(source_link["object"])
    _require(source_files_record.get("files") == source_files, "source sealed-file map differs from v005 seal")
    _require(source_files_record.get("file_count") == len(source_files) and source_files_record.get("all_rehashed") is True, "source sealed-file summary drift")
    verify_file_map(source_files, policy, scope="repository")
    active_files = seal.get("sealed_files")
    active_summary = verify_file_map(active_files, policy, scope="repository")

    expected_root_paths = {"program.py", "README.md", "LEDGER_CHAIN_GENESIS.json"}
    root_controls = seal.get("root_controls")
    _require(isinstance(root_controls, Mapping) and set(root_controls) == expected_root_paths, "root-control closure drift")
    for name, record in root_controls.items():
        _require(isinstance(record, Mapping) and set(record) == {"path", "sha256", "bytes", "source_seal_bound"}, "root-control record schema drift")
        raw = f"{STUDY_RELATIVE}/{name}"
        _require(record.get("path") == raw, "root-control path drift")
        path = policy.path(raw, scope="study")
        _require(record.get("sha256") == sha256_file(path) and record.get("bytes") == path.stat().st_size, "root-control hash/size drift")
        _require(record.get("source_seal_bound") is (raw in source_files), "root-control source-seal binding drift")

    checkpoints = state.get("verified_checkpoints")
    completed = state.get("completed_states")
    _require(isinstance(checkpoints, list) and isinstance(completed, list), "controller checkpoint lineage absent")
    _require(completed[: len(INHERITED_COMPLETED_STATES)] == list(INHERITED_COMPLETED_STATES), "inherited completed-state prefix drift")
    inherited = checkpoints[: len(INHERITED_COMPLETED_STATES)]
    projection = [
        {
            "checkpoint_name": item.get("name"),
            "source_attempt": item.get("source_attempt"),
            "evidence_path": item.get("evidence_path"),
            "evidence_sha256": item.get("evidence_sha256"),
        }
        for item in inherited
    ]
    _require(seal.get("inherited_verified_checkpoints") == projection, "inheritance-seal checkpoint projection drift")
    first_annotation = {
        "attempt": PRIOR_ATTEMPT,
        "equivalence_evidence_path": PRIOR_PRE_DATA_RELATIVE,
        "equivalence_evidence_sha256": prior_seal_link["sha256"],
    }
    fit_annotation = {
        "attempt": FIT_SOURCE_ATTEMPT,
        "equivalence_evidence_path": FIT_PRE_DATA_RELATIVE,
        "equivalence_evidence_sha256": fit_seal_link["sha256"],
    }
    intermediate_annotation = {
        "attempt": INTERMEDIATE_ATTEMPT,
        "equivalence_evidence_path": INTERMEDIATE_PRE_DATA_RELATIVE,
        "equivalence_evidence_sha256": intermediate_seal_link["sha256"],
    }
    source_annotation = {
        "attempt": SOURCE_ATTEMPT,
        "equivalence_evidence_path": SOURCE_PRE_DATA_RELATIVE,
        "equivalence_evidence_sha256": source_link["sha256"],
    }
    annotation = {
        "attempt": ACTIVE_ATTEMPT,
        "equivalence_evidence_path": INHERITANCE_SEAL_RELATIVE,
        "equivalence_evidence_sha256": seal_hash,
    }
    for checkpoint in inherited:
        _require(
            checkpoint.get("source_attempt") == SCIENCE_ATTEMPT
            and checkpoint.get("verification_lineage") == "direct_checkpoint"
            and checkpoint.get("inherited_into_attempts")
            == [
                first_annotation,
                fit_annotation,
                intermediate_annotation,
                source_annotation,
                annotation,
            ],
            "inherited checkpoint annotation drift",
        )
        evidence = policy.path(checkpoint["evidence_path"], scope="study")
        _require(evidence.is_relative_to(policy.repository_root / ATTEMPT_ROOTS[SCIENCE_ATTEMPT]), "inherited evidence is outside v001")
    _require(
        projection[-1]["evidence_path"] == SCIENCE_PRE_DATA_RELATIVE
        and projection[-1]["evidence_sha256"]
        == science_pre_data_link["sha256"],
        "inherited PRE_OUTCOME_SEAL checkpoint drift",
    )

    history = state.get("attempt_history")
    _require(isinstance(history, list) and [item.get("version") for item in history] == list(ALLOWED_ATTEMPTS), "state attempt-history lineage drift")
    _require(
        history[0].get("path") == ATTEMPT_ROOTS[SCIENCE_ATTEMPT]
        and history[0].get("status") == "invalid_zero_confirmation_outcome_procedural"
        and history[0].get("invalidity_evidence_path")
        == SCIENCE_INVALIDITY_RELATIVE
        and history[0].get("invalidity_evidence_sha256")
        == science_invalidity_link["sha256"],
        "v001 attempt-history invalidity binding drift",
    )
    _require(
        history[1].get("path") == ATTEMPT_ROOTS[PRIOR_ATTEMPT]
        and history[1].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[1].get("invalidity_evidence_path")
        == PRIOR_INVALIDITY_RELATIVE
        and history[1].get("invalidity_evidence_sha256")
        == prior_invalidity_link["sha256"]
        and history[1].get("version_forward_evidence_path")
        == PRIOR_PRE_DATA_RELATIVE
        and history[1].get("version_forward_evidence_sha256")
        == prior_seal_link["sha256"]
        and history[1].get("attempt_parameterization_verified") is True,
        "v002 attempt-history invalidity binding drift",
    )
    _require(
        history[2].get("path") == ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]
        and history[2].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[2].get("invalidity_evidence_path")
        == FIT_INVALIDITY_RELATIVE
        and history[2].get("invalidity_evidence_sha256")
        == fit_invalidity_link["sha256"]
        and history[2].get("version_forward_evidence_path")
        == FIT_PRE_DATA_RELATIVE
        and history[2].get("version_forward_evidence_sha256")
        == fit_seal_link["sha256"]
        and history[2].get("attempt_parameterization_verified") is True,
        "v003 attempt-history invalidity binding drift",
    )
    _require(
        history[3].get("path") == ATTEMPT_ROOTS[INTERMEDIATE_ATTEMPT]
        and history[3].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[3].get("invalidity_evidence_path")
        == INTERMEDIATE_INVALIDITY_RELATIVE
        and history[3].get("invalidity_evidence_sha256")
        == intermediate_invalidity_link["sha256"]
        and history[3].get("version_forward_evidence_path")
        == INTERMEDIATE_PRE_DATA_RELATIVE
        and history[3].get("version_forward_evidence_sha256")
        == intermediate_seal_link["sha256"]
        and history[3].get("attempt_parameterization_verified") is True,
        "v004 attempt-history invalidity binding drift",
    )
    _require(
        history[4].get("path") == ATTEMPT_ROOTS[SOURCE_ATTEMPT]
        and history[4].get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and history[4].get("invalidity_evidence_path") == INVALIDITY_RELATIVE
        and history[4].get("invalidity_evidence_sha256")
        == invalidity_link["sha256"]
        and history[4].get("version_forward_evidence_path")
        == SOURCE_PRE_DATA_RELATIVE
        and history[4].get("version_forward_evidence_sha256")
        == source_link["sha256"]
        and history[4].get("attempt_parameterization_verified") is True,
        "v005 attempt-history invalidity binding drift",
    )
    _require(
        history[5].get("path") == ATTEMPT_ROOTS[ACTIVE_ATTEMPT]
        and history[5].get("status")
        == "active_zero_confirmation_outcome_version_forward"
        and history[5].get("version_forward_evidence_path")
        == INHERITANCE_SEAL_RELATIVE
        and history[5].get("version_forward_evidence_sha256") == seal_hash
        and history[5].get("attempt_parameterization_verified") is True,
        "v008 attempt-history equivalence binding drift",
    )
    state_lineage = state.get("version_forward_lineage")
    prior_lineage = seal.get("prior_version_forward_lineage")
    _require(
        isinstance(state_lineage, list)
        and len(state_lineage) == 5
        and isinstance(prior_lineage, list)
        and len(prior_lineage) == 4
        and state_lineage[:-1] == prior_lineage,
        "STATE must preserve exactly four prior edges before v008",
    )
    first_state_edge = state_lineage[0]
    first_edge_common = {
        "old_attempt": SCIENCE_ATTEMPT,
        "new_attempt": PRIOR_ATTEMPT,
        "resume_state": VERSION_FORWARD_RESUME_STATE,
        "invalidity_path": SCIENCE_INVALIDITY_RELATIVE,
        "invalidity_sha256": science_invalidity_link["sha256"],
        "equivalence_path": PRIOR_PRE_DATA_RELATIVE,
        "equivalence_sha256": prior_seal_link["sha256"],
        "inherited_verified_checkpoints": projection,
        "attempt_parameterization_verified": True,
    }
    _require(
        all(
            first_state_edge.get(key) == value
            for key, value in first_edge_common.items()
        ),
        "first STATE version-forward edge binding drift",
    )
    fit_state_edge = state_lineage[1]
    fit_edge_common = {
        "old_attempt": PRIOR_ATTEMPT,
        "new_attempt": FIT_SOURCE_ATTEMPT,
        "resume_state": VERSION_FORWARD_RESUME_STATE,
        "invalidity_path": PRIOR_INVALIDITY_RELATIVE,
        "invalidity_sha256": prior_invalidity_link["sha256"],
        "equivalence_path": FIT_PRE_DATA_RELATIVE,
        "equivalence_sha256": fit_seal_link["sha256"],
        "inherited_verified_checkpoints": projection,
        "attempt_parameterization_verified": True,
    }
    _require(
        all(
            fit_state_edge.get(key) == value
            for key, value in fit_edge_common.items()
        ),
        "fit-source STATE version-forward edge binding drift",
    )
    intermediate_state_edge = state_lineage[2]
    intermediate_edge_common = {
        "old_attempt": FIT_SOURCE_ATTEMPT,
        "new_attempt": INTERMEDIATE_ATTEMPT,
        "resume_state": VERSION_FORWARD_RESUME_STATE,
        "invalidity_path": FIT_INVALIDITY_RELATIVE,
        "invalidity_sha256": fit_invalidity_link["sha256"],
        "equivalence_path": INTERMEDIATE_PRE_DATA_RELATIVE,
        "equivalence_sha256": intermediate_seal_link["sha256"],
        "inherited_verified_checkpoints": projection,
        "attempt_parameterization_verified": True,
    }
    _require(
        all(
            intermediate_state_edge.get(key) == value
            for key, value in intermediate_edge_common.items()
        ),
        "intermediate STATE version-forward edge binding drift",
    )
    source_state_edge = state_lineage[3]
    source_edge_common = {
        "old_attempt": INTERMEDIATE_ATTEMPT,
        "new_attempt": SOURCE_ATTEMPT,
        "resume_state": VERSION_FORWARD_RESUME_STATE,
        "invalidity_path": INTERMEDIATE_INVALIDITY_RELATIVE,
        "invalidity_sha256": intermediate_invalidity_link["sha256"],
        "equivalence_path": SOURCE_PRE_DATA_RELATIVE,
        "equivalence_sha256": source_link["sha256"],
        "inherited_verified_checkpoints": projection,
        "attempt_parameterization_verified": True,
    }
    _require(
        all(
            source_state_edge.get(key) == value
            for key, value in source_edge_common.items()
        ),
        "source STATE version-forward edge binding drift",
    )
    state_edge = state_lineage[-1]
    edge_common = {
        "old_attempt": SOURCE_ATTEMPT,
        "new_attempt": ACTIVE_ATTEMPT,
        "resume_state": VERSION_FORWARD_RESUME_STATE,
        "invalidity_path": INVALIDITY_RELATIVE,
        "invalidity_sha256": invalidity_link["sha256"],
        "equivalence_path": INHERITANCE_SEAL_RELATIVE,
        "equivalence_sha256": seal_hash,
        "inherited_verified_checkpoints": projection,
        "attempt_parameterization_verified": True,
    }
    _require(all(state_edge.get(key) == value for key, value in edge_common.items()), "STATE version-forward edge binding drift")
    edge_events = [event for event in events if event.get("event") == "zero_confirmation_outcome_version_forward"]
    _require(len(edge_events) == 5, "ledger version-forward edge count drift")
    (
        first_ledger_edge,
        fit_ledger_edge,
        intermediate_ledger_edge,
        source_ledger_edge,
        ledger_edge,
    ) = edge_events
    _require(all(first_ledger_edge.get(key) == value for key, value in first_edge_common.items()), "first ledger version-forward edge binding drift")
    _require(all(fit_ledger_edge.get(key) == value for key, value in fit_edge_common.items()), "fit-source ledger version-forward edge binding drift")
    _require(all(intermediate_ledger_edge.get(key) == value for key, value in intermediate_edge_common.items()), "intermediate ledger version-forward edge binding drift")
    _require(all(source_ledger_edge.get(key) == value for key, value in source_edge_common.items()), "source ledger version-forward edge binding drift")
    _require(all(ledger_edge.get(key) == value for key, value in edge_common.items()), "ledger version-forward edge binding drift")
    edge_index = list(events).index(ledger_edge)
    snapshot = seal.get("controller_snapshot")
    frozen = invalidity.get("frozen_evidence", {})
    _require(
        isinstance(snapshot, Mapping)
        and snapshot.get("state_path") == f"{STUDY_RELATIVE}/STATE.json"
        and snapshot.get("ledger_path") == f"{STUDY_RELATIVE}/RESEARCH_LEDGER.jsonl"
        and snapshot.get("event_count") == edge_index
        and snapshot.get("head_sha256") == ledger_edge.get("prev_sha256")
        and snapshot.get("state_sha256")
        == frozen.get("controller_state_sha256_at_failure")
        and snapshot.get("ledger_sha256")
        == frozen.get("research_ledger_sha256_at_failure")
        and snapshot.get("genesis_sha256") == frozen.get("ledger_chain_genesis_sha256"),
        "pre-forward controller snapshot does not meet the authenticated ledger edge",
    )

    equivalence = verify_equivalence_partitions(seal, policy)
    checks = seal.get("checks")
    _require(isinstance(checks, Mapping) and bool(checks) and all(value is True for value in checks.values()), "inheritance producer checks report failure")
    source_marker = seal.get("source_controller_adapter_marker")
    _require(
        isinstance(source_marker, Mapping),
        "inheritance seal source controller marker absent",
    )
    transaction_receipt = _verify_version_forward_transaction_receipt(
        policy=policy,
        current_state=state,
        current_events=events,
        expected_seal_sha256=seal_hash,
        expected_invalidity_sha256=invalidity_link["sha256"],
        expected_edge=edge_common,
        expected_source_marker=source_marker,
        expected_equivalence=equivalence,
        expected_active_sealed_file_summary=active_summary,
    )
    return {
        "attempts": list(ALLOWED_ATTEMPTS),
        "version_forward_edges": 5,
        "science_attempt": SCIENCE_ATTEMPT,
        "active_attempt": ACTIVE_ATTEMPT,
        "seal_sha256": seal_hash,
        "invalidity_sha256": invalidity_link["sha256"],
        "source_pre_data_sha256": source_link["sha256"],
        "inherited_checkpoint_count": len(inherited),
        "active_sealed_file_count": active_summary["file_count"],
        "source_pre_data": source_pre_data,
        "equivalence": equivalence,
        "transaction_receipt": transaction_receipt,
    }


def verify_version_forward_lineage(
    contract: Mapping[str, Any],
    policy: PathPolicy,
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently authenticate the v001-to-v008 selection-boundary lineage."""

    counts: dict[str, int | bool] = {
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    zero_counts: dict[str, int | bool] = {
        **counts,
        "fit_outcome_episodes": 0,
    }

    def exact_typed(value: Any, expected: Mapping[str, Any], label: str) -> None:
        _require(
            isinstance(value, Mapping)
            and set(value) == set(expected)
            and all(
                type(value.get(key)) is type(item)
                and value.get(key) == item
                for key, item in expected.items()
            ),
            f"{label} drift",
        )

    def link(value: Any, relative: str, label: str) -> dict[str, Any]:
        _require(
            isinstance(value, Mapping)
            and set(value) == {"path", "sha256"}
            and value.get("path") == relative,
            f"{label} link schema drift",
        )
        path = policy.path(relative, scope="study")
        digest = sha256_file(path)
        _require(value.get("sha256") == digest, f"{label} link hash drift")
        return {"path": path, "sha256": digest, "object": read_json(path)}

    lineage = contract.get("lineage")
    _require(isinstance(lineage, Mapping), "contract lineage absent")
    seal_relative = str(lineage.get("pre_data_inheritance_seal"))
    _require(
        seal_relative == INHERITANCE_SEAL_RELATIVE,
        "active inheritance-seal path drift",
    )
    seal_path = policy.path(seal_relative, scope="study")
    seal = read_json(seal_path)
    seal_hash = sha256_file(seal_path)
    _validate_inheritance_seal_scalar_types(seal)
    header = {
        "schema_version": 1,
        "artifact_type": "pre_data_inheritance_and_version_forward_equivalence",
        "authorization_kind": (
            "zero_confirmation_outcome_version_forward_inherited_pre_selection"
        ),
        "attempt": ACTIVE_ATTEMPT,
        "source_attempt": ACTIVATION_SOURCE_ATTEMPT,
        "science_attempt": SCIENCE_ATTEMPT,
        "target_attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "PRE_OUTCOME_SEAL",
        "resume_state": VERSION_FORWARD_RESUME_STATE,
        "procedural_invalidity_confirmed": True,
        "zero_confirmation_outcomes_at_version_forward": True,
        "source_hash_equivalent": True,
        "normalized_ast_equivalent": True,
        "scientific_object_hash_equivalent": True,
        "configuration_hash_equivalent": True,
        "scientific_changes": False,
        "attempt_parameterization_verified": True,
        "runtime_modules_accept_active_attempt": True,
        "independent_verifier_accepts_active_attempt": True,
        "source_equivalent": True,
        "ast_equivalent": True,
        "scientific_objects_equivalent": True,
        "hashes_equivalent": True,
        "outcome_arrays_opened": False,
        "hdf5_contents_opened": False,
        "v3_targets_opened": False,
        "passed": True,
    }
    _require(
        all(
            type(seal.get(key)) is type(value)
            and seal.get(key) == value
            for key, value in header.items()
        )
        and type(seal.get("created_unix_ns")) is int
        and seal.get("created_unix_ns") > 0,
        "inheritance-seal header/type drift",
    )
    exact_typed(seal.get("outcome_counts_at_seal"), counts, "sealed outcome counts")
    _require(
        seal.get("confirmation_artifact_scan") == {"file_count": 0, "paths": []},
        "inheritance confirmation-artifact census drift",
    )

    invalidity_link = link(
        seal.get("invalidity_evidence"), INVALIDITY_RELATIVE, "v007 invalidity"
    )
    source_seal_link = link(
        seal.get("source_pre_data_seal"),
        SOURCE_PRE_DATA_RELATIVE,
        "v007 pre-data seal",
    )
    source_selection_link = link(
        seal.get("source_pre_selection_seal"),
        SOURCE_PRE_SELECTION_RELATIVE,
        "v006 pre-selection seal",
    )
    source_receipt_link = link(
        seal.get("source_version_forward_transaction_receipt"),
        SOURCE_RECEIPT_RELATIVE,
        "v007 activation receipt",
    )
    draft = seal.get("superseded_invalidity_draft")
    draft_path = policy.path(INVALIDITY_DRAFT_RELATIVE, scope="study")
    _require(
        isinstance(draft, Mapping)
        and set(draft) == {"path", "sha256", "authoritative"}
        and draft.get("path") == INVALIDITY_DRAFT_RELATIVE
        and draft.get("sha256") == sha256_file(draft_path)
        and draft.get("authoritative") is False,
        "v007 superseded invalidity-draft provenance drift",
    )
    _require(
        invalidity_link["sha256"] == EXPECTED_SOURCE_INVALIDITY_SHA256
        and source_seal_link["sha256"] == EXPECTED_SOURCE_SEAL_SHA256
        and source_selection_link["sha256"]
        == EXPECTED_SOURCE_PRE_SELECTION_SHA256
        and source_receipt_link["sha256"] == EXPECTED_SOURCE_RECEIPT_SHA256
        and draft.get("sha256") == EXPECTED_SOURCE_INVALIDITY_DRAFT_SHA256,
        "v007 immutable-record hash drift",
    )

    invalidity = invalidity_link["object"]
    exact_typed(
        invalidity.get("controller_outcome_counts"),
        counts,
        "v007 invalidity controller counts",
    )
    fit_evidence = invalidity.get("fit_evidence")
    selection_evidence = invalidity.get("selection_evidence")
    frozen = invalidity.get("frozen_evidence")
    superseded = invalidity.get("superseded_draft")
    _require(
        invalidity.get("attempt") == ACTIVATION_SOURCE_ATTEMPT
        and invalidity.get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and invalidity.get("current_state") == VERSION_FORWARD_RESUME_STATE
        and invalidity.get("procedural_invalidity") is True
        and invalidity.get("authoritative") is True
        and invalidity.get("scientific_outcomes_conditioned_on") is False
        and invalidity.get("scientific_criterion_changed") is False
        and invalidity.get("selection_or_later_outcomes_opened") is False
        and invalidity.get("confirmation_artifact_file_count") == 0
        and invalidity.get("confirmation_outcome_episodes_generated") == 0
        and invalidity.get("confirmation_outcome_episodes_executed") == 0
        and invalidity.get("confirmation_outcomes_opened_for_analysis") is False
        and invalidity.get("later_role_artifact_file_counts")
        == {"selection": 0, "smoke": 0, "confirmation": 0}
        and isinstance(fit_evidence, Mapping)
        and fit_evidence.get("fit_outcome_episode_count") == 1200
        and fit_evidence.get("selection_outcomes_used_for_fit") is False
        and fit_evidence.get("immutable_and_reusable_under_authenticated_lineage")
        is True
        and fit_evidence.get("independent_scientific_replay_completed") is True
        and isinstance(selection_evidence, Mapping)
        and selection_evidence
        == {
            "controller_selection_outcome_episode_count": 0,
            "generation_output_file_count": 0,
            "seed_tuples_consumed": 0,
            "selection_outcome_arrays_opened": False,
            "worlds_constructed": 0,
        }
        and isinstance(frozen, Mapping)
        and frozen.get("controller_state_sha256_at_failure")
        == EXPECTED_SOURCE_STATE_SHA256
        and frozen.get("research_ledger_sha256_at_failure")
        == EXPECTED_SOURCE_LEDGER_SHA256
        and frozen.get("pre_data_inheritance_seal_sha256")
        == EXPECTED_SOURCE_SEAL_SHA256
        and frozen.get("pre_selection_seal_sha256")
        == EXPECTED_SOURCE_PRE_SELECTION_SHA256
        and frozen.get("version_forward_receipt_sha256")
        == EXPECTED_SOURCE_RECEIPT_SHA256
        and isinstance(superseded, Mapping)
        and superseded.get("path") == INVALIDITY_DRAFT_RELATIVE
        and superseded.get("sha256") == EXPECTED_SOURCE_INVALIDITY_DRAFT_SHA256,
        "v007 authoritative procedural-invalidity provenance drift",
    )
    defect = invalidity.get("defect")
    _require(
        isinstance(defect, Mapping)
        and defect.get("classification")
        == "independent_verifier_contract_schema_rejects_authenticated_inherited_selection_boundary_paths"
        and defect.get("exception_type") == "VerificationError"
        and defect.get("reproduced_without_outcome_arrays") is True
        and defect.get("contract_schema_only") is True
        and isinstance(defect.get("reproduction_input"), Mapping)
        and defect["reproduction_input"].get("inherited_pre_selection_seal_path")
        == SOURCE_PRE_SELECTION_RELATIVE,
        "v007 provenance-validation defect record drift",
    )

    source_seal = source_seal_link["object"]
    source_receipt = source_receipt_link["object"]
    source_selection = source_selection_link["object"]
    exact_typed(
        source_seal.get("outcome_counts_at_seal"),
        counts,
        "v007 activation-seal counts",
    )
    _require(
        source_seal.get("attempt") == ACTIVATION_SOURCE_ATTEMPT
        and source_seal.get("source_attempt") == SOURCE_ATTEMPT
        and source_seal.get("target_attempt") == ACTIVATION_SOURCE_ATTEMPT
        and source_seal.get("science_attempt") == SCIENCE_ATTEMPT
        and source_seal.get("resume_state") == "SELECTION_COHORTS"
        and source_seal.get("passed") is True
        and source_seal.get("confirmation_artifact_scan")
        == {"file_count": 0, "paths": []}
        and source_receipt.get("source_attempt") == SOURCE_ATTEMPT
        and source_receipt.get("target_attempt") == ACTIVATION_SOURCE_ATTEMPT
        and source_receipt.get("resume_state") == "SELECTION_COHORTS"
        and source_receipt.get("passed") is True
        and source_receipt.get("post_snapshot", {}).get("state", {}).get("sha256")
        == EXPECTED_SOURCE_STATE_SHA256
        and source_receipt.get("post_snapshot", {}).get("ledger", {}).get("sha256")
        == EXPECTED_SOURCE_LEDGER_SHA256,
        "v007 activation provenance drift",
    )
    exact_typed(
        source_selection.get("outcome_counts_at_seal"),
        counts,
        "v006 pre-selection counts",
    )
    _require(
        source_selection.get("attempt") == SOURCE_ATTEMPT
        and source_selection.get("checkpoint_state") == "PRE_SELECTION_SEAL"
        and source_selection.get("selection_arrays_opened_by_sealer") is False
        and source_selection.get("selected_head_refit_permitted") is False
        and source_selection.get("passed") is True
        and source_selection.get("pre_data_seal_sha256")
        == EXPECTED_SELECTION_SOURCE_SEAL_SHA256,
        "v006 pre-selection seal drift",
    )

    source_files = _manifest_files(source_seal)
    source_record = seal.get("source_sealed_files")
    _require(
        isinstance(source_record, Mapping)
        and set(source_record) == {"file_count", "files", "all_rehashed"}
        and source_record.get("file_count") == 6475
        and source_record.get("files") == source_files
        and source_record.get("all_rehashed") is True,
        "v007 activation sealed-file census drift",
    )
    _require(
        verify_file_map(source_files, policy, scope="repository")["file_count"]
        == 6475,
        "v007 activation sealed-file rehash census drift",
    )
    selection_files = _manifest_files(source_selection)
    selection_record = seal.get("source_pre_selection_sealed_files")
    _require(
        isinstance(selection_record, Mapping)
        and set(selection_record) == {"file_count", "files", "all_rehashed"}
        and selection_record.get("file_count") == 6408
        and selection_record.get("files") == selection_files
        and selection_record.get("all_rehashed") is True,
        "v006 pre-selection sealed-file census drift",
    )
    _require(
        verify_file_map(selection_files, policy, scope="repository")["file_count"]
        == 6408,
        "v006 pre-selection sealed-file rehash census drift",
    )

    equivalence = verify_equivalence_partitions(seal, policy)
    closure = seal.get("v008_manifest_closure")
    _require(
        isinstance(closure, Mapping)
        and closure == _independent_manifest_closure(policy),
        "v008 manifest-closure reconstruction drift",
    )
    expected_sealed = dict(source_files)
    for relative in closure["discovered_paths"]:
        candidate = policy.attempt_root / str(relative)
        expected_sealed[candidate.relative_to(policy.repository_root).as_posix()] = (
            sha256_file(candidate)
        )
    for relative in closure["declared_repository_paths"]:
        candidate = policy.path(str(relative), scope="repository")
        expected_sealed[str(relative)] = sha256_file(candidate)
    for relative in (
        INVALIDITY_RELATIVE,
        INVALIDITY_DRAFT_RELATIVE,
        SOURCE_PRE_DATA_RELATIVE,
        SOURCE_RECEIPT_RELATIVE,
        SOURCE_PRE_SELECTION_RELATIVE,
    ):
        expected_sealed[relative] = sha256_file(
            policy.path(relative, scope="study")
        )
    inventory_path = policy.path(FIT_INVENTORY_RELATIVE, scope="study")
    inventory = read_json(inventory_path)
    for record in inventory.get("files", []):
        _require(
            isinstance(record, Mapping)
            and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str),
            "fit-inventory file record drift",
        )
        expected_sealed[str(record["path"])] = str(record["sha256"])
    for relative in SOURCE_OPERATIONAL_LOCKS:
        candidate = (
            policy.repository_root
            / ATTEMPT_ROOTS[FIT_SOURCE_ATTEMPT]
            / relative
        )
        _require(candidate.stat().st_size == 0, f"source lock drift: {relative}")
        expected_sealed[candidate.relative_to(policy.repository_root).as_posix()] = (
            sha256_file(candidate)
        )
    expected_sealed[FIT_INVENTORY_RELATIVE] = sha256_file(inventory_path)
    active_files = seal.get("sealed_files")
    _require(
        isinstance(active_files, Mapping)
        and len(expected_sealed) == EXPECTED_TRANSITIVE_SEALED_FILE_COUNT
        and dict(active_files) == dict(sorted(expected_sealed.items())),
        "v008 transitive sealed-file reconstruction drift",
    )
    active_summary = verify_file_map(active_files, policy, scope="repository")

    root_controls = seal.get("root_controls")
    _require(
        isinstance(root_controls, Mapping)
        and set(root_controls) == {"program.py", "README.md", "LEDGER_CHAIN_GENESIS.json"},
        "root-control census drift",
    )
    for name, record in root_controls.items():
        relative = f"{STUDY_RELATIVE}/{name}"
        candidate = policy.path(relative, scope="study")
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "sha256", "bytes", "source_seal_bound"}
            and record.get("path") == relative
            and record.get("sha256") == sha256_file(candidate)
            and record.get("bytes") == candidate.stat().st_size
            and record.get("source_seal_bound")
            is (source_files.get(relative) == sha256_file(candidate)),
            f"root-control provenance drift: {name}",
        )

    checkpoints = state.get("verified_checkpoints")
    completed = state.get("completed_states")
    _require(
        isinstance(checkpoints, list)
        and isinstance(completed, list)
        and completed[: len(INHERITED_COMPLETED_STATES)]
        == list(INHERITED_COMPLETED_STATES)
        and len(checkpoints) >= len(INHERITED_COMPLETED_STATES),
        "inherited checkpoint chronology drift",
    )
    inherited = checkpoints[: len(INHERITED_COMPLETED_STATES)]
    projection: list[dict[str, Any]] = []
    active_annotation = {
        "attempt": ACTIVE_ATTEMPT,
        "equivalence_evidence_path": INHERITANCE_SEAL_RELATIVE,
        "equivalence_evidence_sha256": seal_hash,
    }
    for index, checkpoint in enumerate(inherited):
        _require(
            isinstance(checkpoint, Mapping)
            and checkpoint.get("verification_lineage") == "direct_checkpoint",
            "inherited checkpoint schema drift",
        )
        evidence_relative = str(checkpoint.get("evidence_path"))
        evidence = policy.path(evidence_relative, scope="study")
        _require(
            checkpoint.get("evidence_sha256") == sha256_file(evidence),
            f"inherited checkpoint evidence drift: {index}",
        )
        expected_source = SCIENCE_ATTEMPT if index < 6 else SOURCE_ATTEMPT
        expected_root = ATTEMPT_ROOTS[expected_source] + "/"
        _require(
            checkpoint.get("source_attempt") == expected_source
            and evidence_relative.startswith(expected_root)
            and active_annotation in checkpoint.get("inherited_into_attempts", []),
            f"inherited checkpoint dataset-role provenance drift: {index}",
        )
        projection.append(
            {
                "checkpoint_name": checkpoint.get("name"),
                "source_attempt": checkpoint.get("source_attempt"),
                "evidence_path": evidence_relative,
                "evidence_sha256": checkpoint.get("evidence_sha256"),
            }
        )
    _require(
        projection == seal.get("inherited_verified_checkpoints")
        and projection[-1]["evidence_path"] == SOURCE_PRE_SELECTION_RELATIVE
        and projection[-1]["evidence_sha256"]
        == EXPECTED_SOURCE_PRE_SELECTION_SHA256,
        "inheritance-seal checkpoint projection drift",
    )

    prior_lineage = seal.get("prior_version_forward_lineage")
    state_lineage = state.get("version_forward_lineage")
    _require(
        isinstance(prior_lineage, list)
        and len(prior_lineage) == 6
        and isinstance(state_lineage, list)
        and len(state_lineage) == 7
        and state_lineage[:-1] == prior_lineage,
        "seven-edge deterministic lineage drift",
    )
    expected_pairs = list(zip(ALLOWED_ATTEMPTS[:-2], ALLOWED_ATTEMPTS[1:-1]))
    for index, (edge, pair) in enumerate(zip(prior_lineage, expected_pairs)):
        _require(
            isinstance(edge, Mapping)
            and edge.get("old_attempt") == pair[0]
            and edge.get("new_attempt") == pair[1]
            and edge.get("resume_state")
            == (
                VERSION_FORWARD_RESUME_STATE
                if index == len(prior_lineage) - 1
                else "FIT_COHORTS"
            )
            and edge.get("attempt_parameterization_verified") is True
            and isinstance(edge.get("inherited_verified_checkpoints"), list)
            and len(edge["inherited_verified_checkpoints"])
            == (9 if index == len(prior_lineage) - 1 else 6),
            f"prior lineage edge drift: {index}",
        )
        for field in ("invalidity_path", "equivalence_path"):
            candidate = policy.path(str(edge.get(field)), scope="study")
            _require(
                sha256_file(candidate) == edge.get(field.replace("path", "sha256")),
                f"prior lineage evidence drift: {index}/{field}",
            )
    edge = state_lineage[-1]
    edge_expected = {
        "old_attempt": ACTIVATION_SOURCE_ATTEMPT,
        "new_attempt": ACTIVE_ATTEMPT,
        "resume_state": VERSION_FORWARD_RESUME_STATE,
        "invalidity_path": INVALIDITY_RELATIVE,
        "invalidity_sha256": invalidity_link["sha256"],
        "equivalence_path": INHERITANCE_SEAL_RELATIVE,
        "equivalence_sha256": seal_hash,
        "inherited_verified_checkpoints": projection,
        "attempt_parameterization_verified": True,
    }
    _require(
        isinstance(edge, Mapping)
        and set(edge) == set(edge_expected)
        and dict(edge) == edge_expected,
        "v007-to-v008 lineage edge drift",
    )

    history = state.get("attempt_history")
    _require(
        isinstance(history, list)
        and len(history) == 8
        and [item.get("version") for item in history] == list(ALLOWED_ATTEMPTS),
        "attempt-history chronology drift",
    )
    for index, item in enumerate(history[:7]):
        _require(
            item.get("path") == ATTEMPT_ROOTS[ALLOWED_ATTEMPTS[index]],
            f"attempt-history path drift: {index}",
        )
        _require(
            item.get("status") == "invalid_zero_confirmation_outcome_procedural",
            f"attempt-history terminal status drift: {index}",
        )
        for path_key, hash_key in (
            ("invalidity_evidence_path", "invalidity_evidence_sha256"),
            ("version_forward_evidence_path", "version_forward_evidence_sha256"),
        ):
            if path_key in item:
                candidate = policy.path(str(item[path_key]), scope="study")
                _require(
                    sha256_file(candidate) == item.get(hash_key),
                    f"attempt-history evidence drift: {index}/{path_key}",
                )
    _require(
        history[6].get("status") == "invalid_zero_confirmation_outcome_procedural"
        and history[6].get("invalidity_evidence_path") == INVALIDITY_RELATIVE
        and history[6].get("invalidity_evidence_sha256")
        == invalidity_link["sha256"]
        and history[7].get("path") == ATTEMPT_ROOTS[ACTIVE_ATTEMPT]
        and history[7].get("status")
        == "active_zero_confirmation_outcome_version_forward"
        and history[7].get("version_forward_evidence_path")
        == INHERITANCE_SEAL_RELATIVE
        and history[7].get("version_forward_evidence_sha256") == seal_hash
        and history[7].get("attempt_parameterization_verified") is True,
        "v007/v008 attempt-history boundary drift",
    )

    edge_events = [
        event
        for event in events
        if event.get("event") == "zero_confirmation_outcome_version_forward"
    ]
    _require(len(edge_events) == 7, "ledger version-forward edge census drift")
    for index, (ledger_edge, state_edge) in enumerate(zip(edge_events, state_lineage)):
        _require(
            all(ledger_edge.get(key) == value for key, value in state_edge.items()),
            f"ledger/state lineage edge drift: {index}",
        )
    _require(
        edge_events[-1].get("attempt") == ACTIVE_ATTEMPT
        and edge_events[-1].get("source_controller_adapter_marker")
        == seal.get("source_controller_adapter_marker")
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(edge_events[-1].get("version_forward_receipt_context_sha256")),
        )
        is not None,
        "v008 ledger authorization binding drift",
    )

    snapshot = seal.get("controller_snapshot")
    _require(
        isinstance(snapshot, Mapping)
        and snapshot.get("state_path") == f"{STUDY_RELATIVE}/STATE.json"
        and snapshot.get("state_sha256") == EXPECTED_SOURCE_STATE_SHA256
        and snapshot.get("ledger_path")
        == f"{STUDY_RELATIVE}/RESEARCH_LEDGER.jsonl"
        and snapshot.get("ledger_sha256") == EXPECTED_SOURCE_LEDGER_SHA256
        and snapshot.get("event_count") == 17
        and snapshot.get("head_sha256")
        == "7486d4096773da9450cf145ca321e2eaabb907e4cc50cee460dc19d6f8714d45"
        and snapshot.get("genesis_sha256")
        == frozen.get("ledger_chain_genesis_sha256"),
        "v007 selection-boundary controller snapshot drift",
    )
    checks = {
        "expected_boundary_outcome_counts": True,
        "no_confirmation_artifacts": True,
        "invalidity_authenticated": True,
        "source_pre_data_seal_authenticated": True,
        "source_pre_selection_seal_authenticated": True,
        "all_inherited_checkpoints_authenticated": True,
        "source_sealed_files_rehashed": True,
        "complete_source_partition": True,
        "v008_manifest_closure": True,
        "root_controls_authenticated": True,
        "scientific_objects_equivalent": True,
        "contracts_canonically_equivalent": True,
    }
    exact_typed(seal.get("checks"), checks, "inheritance producer checks")
    transaction_receipt = _verify_version_forward_transaction_receipt(
        policy=policy,
        current_state=state,
        current_events=events,
        expected_seal_sha256=seal_hash,
        expected_invalidity_sha256=invalidity_link["sha256"],
        expected_edge=edge_expected,
        expected_source_marker=seal.get("source_controller_adapter_marker"),
        expected_equivalence=equivalence,
        expected_active_sealed_file_summary=active_summary,
    )
    return {
        "attempts": list(ALLOWED_ATTEMPTS),
        "version_forward_edges": 7,
        "science_attempt": SCIENCE_ATTEMPT,
        "active_attempt": ACTIVE_ATTEMPT,
        "seal_sha256": seal_hash,
        "invalidity_sha256": invalidity_link["sha256"],
        "source_pre_data_sha256": source_seal_link["sha256"],
        "source_pre_selection_sha256": source_selection_link["sha256"],
        "inherited_checkpoint_count": len(inherited),
        "active_sealed_file_count": active_summary["file_count"],
        "source_pre_data": {
            "attempt": ACTIVATION_SOURCE_ATTEMPT,
            "sha256": source_seal_link["sha256"],
            "pre_selection_sha256": source_selection_link["sha256"],
            "passed": True,
        },
        "equivalence": equivalence,
        "transaction_receipt": transaction_receipt,
    }


def _walk_json(value: Any) -> Iterator[Any]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)
    else:
        yield value


def scan_prior_identifiers(
    policy: PathPolicy, contract: Mapping[str, Any]
) -> tuple[set[int], set[str], dict[str, Any]]:
    scan = contract.get("prior_scan", {})
    runs_raw = scan.get("runs_root", "runs")
    runs_root = policy.path(runs_raw, scope="repository", file_only=False)
    forbidden = set(scan.get("forbidden_tree_names", ["lewm_adaptive_compute_v3"]))
    excluded_raw = scan.get("excluded_roots", [contract["study_root"]])
    excluded = {policy.path(raw, scope="repository", file_only=False) for raw in excluded_raw}
    numbers: set[int] = set()
    strings: set[str] = set()
    metadata_paths: list[str] = []
    filename_paths: list[str] = []
    for root, directories, filenames in os.walk(runs_root, topdown=True, followlinks=False):
        root_path = Path(root)
        kept = []
        for directory in directories:
            candidate = root_path / directory
            if candidate.is_symlink() or directory in forbidden:
                continue
            if any(candidate == excluded_root or candidate.is_relative_to(excluded_root) for excluded_root in excluded):
                continue
            kept.append(directory)
        directories[:] = sorted(kept)
        for filename in sorted(filenames):
            path = root_path / filename
            if path.is_symlink() or any(part in forbidden for part in path.parts):
                continue
            if any(path.is_relative_to(excluded_root) for excluded_root in excluded):
                continue
            relative = path.relative_to(runs_root).as_posix()
            filename_paths.append(relative)
            strings.update((path.name, path.stem))
            if path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            payload = path.read_text(encoding="utf-8")
            documents = (
                [json.loads(payload)]
                if path.suffix.lower() == ".json"
                else [json.loads(line) for line in payload.splitlines() if line.strip()]
            )
            for document in documents:
                for item in _walk_json(document):
                    if isinstance(item, bool):
                        continue
                    if isinstance(item, int):
                        numbers.add(item)
                    elif isinstance(item, str):
                        strings.add(item)
            metadata_paths.append(relative)
    safe_numbers: set[int] = set()
    safe_strings: set[str] = set()
    safe_hashes: dict[str, str] = {}
    for relative in SAFE_V3_RELATIVE_PATHS:
        path = runs_root / relative
        _require(path.is_file() and not path.is_symlink(), f"safe V3 audit file absent: {relative}")
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        local_numbers: set[int] = set()
        local_strings: set[str] = {path.name, path.stem}
        if path.suffix == ".json":
            for item in _walk_json(json.loads(text)):
                if isinstance(item, bool):
                    continue
                if isinstance(item, int):
                    local_numbers.add(item)
                elif isinstance(item, str):
                    local_strings.add(item)
        elif path.suffix == ".py":
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant):
                    if type(node.value) is int:
                        local_numbers.add(node.value)
                    elif isinstance(node.value, str):
                        local_strings.add(node.value)
                elif isinstance(node, ast.Name):
                    local_strings.add(node.id)
        else:
            local_numbers.update(int(token.replace("_", "")) for token in re.findall(r"(?<![\w.])-?\d[\d_]*(?![\w.])", text))
            local_strings.update(re.findall(r"[A-Za-z][A-Za-z0-9_.:/-]{2,}", text))
        safe_numbers.update(local_numbers)
        safe_strings.update(local_strings)
        safe_hashes[relative] = hashlib.sha256(payload).hexdigest()
        numbers.update(local_numbers)
        strings.update(local_strings)
        filename_paths.append(relative)
        metadata_paths.append(relative)
    return numbers, strings, {
        "metadata_file_count": len(metadata_paths),
        "filename_count": len(filename_paths),
        "numeric_identifier_set_sha256": _set_digest(numbers),
        "string_identifier_set_sha256": _set_digest(strings),
        "metadata_path_set_sha256": _set_digest(metadata_paths),
        "filename_path_set_sha256": _set_digest(filename_paths),
        "safe_v3_source_config_plan_files": safe_hashes,
        "safe_v3_numeric_literal_set_sha256": _set_digest(safe_numbers),
        "safe_v3_string_identifier_set_sha256": _set_digest(safe_strings),
        "v3_safe_source_config_plan_opened": True,
        "v3_test_target_cache_split_metrics_artifacts_opened": False,
        "binary_contents_opened": False,
        "hdf5_contents_opened": False,
    }


def _iter_rng_records(analysis_rng_ids: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for group in analysis_rng_ids.values():
        if isinstance(group, Mapping):
            records = group.get("rng_ids", [])
            _require(isinstance(records, list), "RNG record group is not a list")
            for record in records:
                _require(isinstance(record, Mapping), "RNG record is not an object")
                yield record


def verify_identifier_ledger(
    contract: Mapping[str, Any], policy: PathPolicy
) -> dict[str, Any]:
    ledger = read_json(policy.path(contract["paths"]["cohort_seed_ledger"], scope="study"))
    _require(ledger.get("schema_version") == 1 and ledger.get("attempt") == SCIENCE_ATTEMPT, "cohort ledger identity drift")
    _require(tuple(ledger.get("regime_order", ())) == DGP_ORDER, "cohort DGP order drift")
    _require(tuple(ledger.get("role_order", ())) == ROLE_ORDER, "cohort role order drift")
    records: list[tuple[str, str, str, Mapping[str, Any]]] = []
    for dgp in DGP_ORDER:
        role_map = ledger["regimes"][dgp]["roles"]
        for role in ROLE_ORDER:
            primary = role_map[role]["primary"]
            replacements = role_map[role]["replacements"]
            expected_primary = 4500 if role == "confirmation" else ROLE_COUNTS[role]
            _require(len(primary) == expected_primary, f"{dgp}/{role} primary count drift")
            _require(len(replacements) == 200, f"{dgp}/{role} replacement count drift")
            for slot, record in enumerate(primary):
                _require(int(record["slot"]) == slot, f"{dgp}/{role} primary slot drift")
                records.append((dgp, role, "primary", record))
            for slot, record in enumerate(replacements):
                _require(int(record["slot"]) == slot, f"{dgp}/{role} replacement slot drift")
                records.append((dgp, role, "replacements", record))
    episode_ids = [str(record[3]["episode_id"]) for record in records]
    numeric = [
        int(record[3][key])
        for record in records
        for key in ("env_seed", "policy_seed", "oracle_np_seed", "action_space_seed")
    ]
    rng_records = list(_iter_rng_records(ledger["analysis_rng_ids"]))
    rng_ids = [int(record["rng_id"]) for record in rng_records]
    rng_labels = [str(record["label"]) for record in rng_records]
    _require(len(set(episode_ids)) == len(episode_ids), "episode identifiers overlap globally")
    _require(len(set(numeric + rng_ids)) == len(numeric) + len(rng_ids), "numeric identifiers overlap globally")
    _require(len(set(rng_labels)) == len(rng_labels), "RNG labels overlap globally")
    _require(not ledger.get("numeric_overlap"), "cohort ledger reports prior numeric overlap")
    _require(not ledger.get("string_overlap"), "cohort ledger reports prior string overlap")
    prior_numbers, prior_strings, snapshot = scan_prior_identifiers(policy, contract)
    _require(not (set(numeric + rng_ids) & prior_numbers), "independent scan found prior numeric overlap")
    _require(not (set(episode_ids + rng_labels) & prior_strings), "independent scan found prior string overlap")
    recorded_snapshot = ledger["prior_identifier_snapshot"]
    for key in (
        "metadata_file_count",
        "filename_count",
        "numeric_identifier_set_sha256",
        "string_identifier_set_sha256",
        "metadata_path_set_sha256",
        "filename_path_set_sha256",
    ):
        _require(recorded_snapshot.get(key) == snapshot[key], f"prior identifier snapshot drift: {key}")
    _require(all(bool(value) for value in ledger.get("checks", {}).values()), "cohort ledger builder check failed")
    return {
        "episode_identifier_count": len(episode_ids),
        "numeric_identifier_count": len(numeric) + len(rng_ids),
        "rng_identifier_count": len(rng_ids),
        "prior_snapshot": snapshot,
        "ledger": ledger,
    }


def _independent_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA-256 digest",
    )
    return str(value)


def _independent_canonical_json_record(path: Path, label: str) -> dict[str, Any]:
    metadata = path.lstat()
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and int(metadata.st_nlink) == 1,
        f"{label} is linked, aliased, or non-regular",
    )
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise VerificationError(f"invalid JSON in {label}") from error
    _require(
        isinstance(value, dict)
        and raw == (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        f"noncanonical {label}",
    )
    return value


def _independent_seed_tuple(value: Mapping[str, Any]) -> tuple[int, int, int, int]:
    output: list[int] = []
    for field in SEED_FIELDS:
        item = value.get(field)
        _require(type(item) is int, f"non-integer seed field: {field}")
        output.append(int(item))
    return tuple(output)  # type: ignore[return-value]


def _independent_registry_contract(
    policy: PathPolicy,
    cohort_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    attempt = policy.attempt_relative
    registry_relative = f"{attempt}/data/replacement_registry.json"
    claims_relative = f"{attempt}/data/replacement_claims"
    dgp_relative = f"{attempt}/DGP_MATRIX.json"
    ledger_relative = f"{attempt}/cohort_seed_ledger.json"
    inheritance_relative = f"{attempt}/audit/pre_data_inheritance_seal.json"
    preconfirmation_relative = (
        f"{attempt}/audit/pre_confirmation_package_seal.json"
    )
    registry_path = policy.path(registry_relative, scope="attempt")
    genesis = _independent_canonical_json_record(
        registry_path, "replacement registry genesis"
    )
    genesis_keys = {
        "schema_version", "record_type", "attempt", "science_attempt", "scope",
        "created_unix_ns", "dgp_matrix_path", "dgp_matrix_sha256",
        "cohort_seed_ledger_path", "cohort_seed_ledger_sha256", "regime_order",
        "role_order", "seed_field_order", "replacement_count_per_regime_per_role",
        "replacement_rule", "claims_directory", "claim_filename_format",
        "authorization_policy", "claim_contract", "record_sha256",
    }
    _require(set(genesis) == genesis_keys, "replacement genesis schema drift")
    genesis_body = dict(genesis)
    genesis_hash = genesis_body.pop("record_sha256", None)
    expected_authorization_policy = {
        role: {
            "path": (
                inheritance_relative
                if role in ("fit", "selection")
                else preconfirmation_relative
            ),
            "checkpoint_state": (
                "PRE_OUTCOME_SEAL"
                if role in ("fit", "selection")
                else "PRE_CONFIRMATION_PACKAGE_SEAL"
            ),
        }
        for role in ROLE_ORDER
    }
    _require(
        genesis.get("schema_version") == 2
        and genesis.get("record_type") == "replacement_registry_genesis"
        and genesis.get("attempt") == ACTIVE_ATTEMPT
        and genesis.get("science_attempt") == SCIENCE_ATTEMPT
        and genesis.get("scope") == "append_only_all_roles_and_all_dgps"
        and type(genesis.get("created_unix_ns")) is int
        and int(genesis["created_unix_ns"]) > 0
        and genesis.get("dgp_matrix_path") == dgp_relative
        and genesis.get("dgp_matrix_sha256")
        == sha256_file(policy.path(dgp_relative, scope="attempt"))
        and genesis.get("cohort_seed_ledger_path") == ledger_relative
        and genesis.get("cohort_seed_ledger_sha256")
        == sha256_file(policy.path(ledger_relative, scope="attempt"))
        and genesis.get("regime_order") == list(DGP_ORDER)
        and genesis.get("role_order") == list(ROLE_ORDER)
        and genesis.get("seed_field_order") == list(SEED_FIELDS)
        and genesis.get("replacement_count_per_regime_per_role") == 200
        and genesis.get("replacement_rule") == cohort_ledger.get("replacement_rule")
        and genesis.get("claims_directory") == claims_relative
        and genesis.get("claim_filename_format") == "{claim_index:06d}.json"
        and genesis.get("authorization_policy") == expected_authorization_policy
        and genesis.get("claim_contract")
        == (
            "exclusive immutable canonical JSON segments; contiguous global indexes; "
            "SHA-256 predecessor chain; exact sealed-ledger tuple; exact role seal; "
            "authenticated mechanical-failure trigger; global source single-use"
        )
        and genesis_hash == canonical_object_sha256(genesis_body),
        "replacement registry genesis authentication drift",
    )

    claims_root = policy.path(
        claims_relative, scope="attempt", file_only=False
    )
    root_info = claims_root.lstat()
    _require(
        stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
        "replacement claims root is linked or non-directory",
    )
    entries = sorted(claims_root.iterdir(), key=lambda item: item.name)
    _require(
        [item.name for item in entries]
        == [f"{index:06d}.json" for index in range(len(entries))],
        "replacement claim segments are not one contiguous exact prefix",
    )
    claim_keys = {
        "schema_version", "record_type", "attempt", "science_attempt", "claim_index",
        "claimed_unix_ns", "prev_sha256", "registry_genesis_path",
        "registry_genesis_file_sha256", "registry_genesis_record_sha256", "role",
        "regime", "slot", "episode_id", "replacement_slot",
        "replacement_episode_id", *SEED_FIELDS, "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256", "authorization_seal",
        "trigger_failure_log_path", "trigger_failure_seq",
        "trigger_failure_record_sha256", "failed_seed_source_episode_id",
        "record_sha256",
    }
    previous = str(genesis_hash)
    claims: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    source_seeds: set[tuple[int, int, int, int]] = set()
    trigger_hashes: set[str] = set()
    for index, claim_path in enumerate(entries):
        policy.path(
            f"{claims_relative}/{claim_path.name}", scope="attempt"
        )
        claim = _independent_canonical_json_record(
            claim_path, f"replacement claim {index}"
        )
        _require(set(claim) == claim_keys, f"replacement claim {index} schema drift")
        body = dict(claim)
        record_hash = body.pop("record_sha256", None)
        role = claim.get("role")
        regime = claim.get("regime")
        slot = claim.get("slot")
        replacement_slot = claim.get("replacement_slot")
        _require(role in ROLE_ORDER and regime in DGP_ORDER, "replacement claim role/DGP drift")
        _require(type(slot) is int and type(replacement_slot) is int, "replacement claim slot type drift")
        role_payload = cohort_ledger["regimes"][str(regime)]["roles"][str(role)]
        primary = role_payload["primary"]
        replacements = role_payload["replacements"]
        _require(0 <= int(slot) < len(primary), "replacement destination slot outside ledger")
        _require(0 <= int(replacement_slot) < len(replacements), "replacement source slot outside ledger")
        destination = primary[int(slot)]
        replacement = replacements[int(replacement_slot)]
        authorization_relative = (
            inheritance_relative
            if role in ("fit", "selection")
            else preconfirmation_relative
        )
        authorization_state = (
            "PRE_OUTCOME_SEAL"
            if role in ("fit", "selection")
            else "PRE_CONFIRMATION_PACKAGE_SEAL"
        )
        authorization_path = policy.path(authorization_relative, scope="attempt")
        expected_authorization = {
            "path": authorization_relative,
            "sha256": sha256_file(authorization_path),
            "checkpoint_state": authorization_state,
        }
        _require(
            type(claim.get("schema_version")) is int
            and claim.get("schema_version") == 2
            and claim.get("record_type") == "replacement_claim"
            and claim.get("attempt") == ACTIVE_ATTEMPT
            and claim.get("science_attempt") == SCIENCE_ATTEMPT
            and type(claim.get("claim_index")) is int
            and claim.get("claim_index") == index
            and type(claim.get("claimed_unix_ns")) is int
            and int(claim["claimed_unix_ns"]) > 0
            and claim.get("prev_sha256") == previous
            and claim.get("registry_genesis_path") == registry_relative
            and claim.get("registry_genesis_file_sha256") == sha256_file(registry_path)
            and claim.get("registry_genesis_record_sha256") == genesis_hash
            and claim.get("episode_id") == destination.get("episode_id")
            and claim.get("replacement_episode_id") == replacement.get("episode_id")
            and _independent_seed_tuple(claim) == _independent_seed_tuple(replacement)
            and claim.get("cohort_seed_ledger_path") == ledger_relative
            and claim.get("cohort_seed_ledger_sha256") == sha256_file(policy.path(ledger_relative, scope="attempt"))
            and claim.get("authorization_seal") == expected_authorization
            and claim.get("trigger_failure_log_path")
            == f"{attempt}/data/{role}/{regime}/rollout_failures.jsonl"
            and type(claim.get("trigger_failure_seq")) is int
            and int(claim["trigger_failure_seq"]) > 0
            and _independent_sha256(
                claim.get("trigger_failure_record_sha256"),
                "replacement trigger hash",
            )
            == claim.get("trigger_failure_record_sha256")
            and isinstance(claim.get("failed_seed_source_episode_id"), str)
            and bool(claim.get("failed_seed_source_episode_id"))
            and record_hash == canonical_object_sha256(body),
            f"replacement claim {index} authentication drift",
        )
        source_id = str(claim["replacement_episode_id"])
        seeds = _independent_seed_tuple(claim)
        trigger = str(claim["trigger_failure_record_sha256"])
        _require(
            source_id not in source_ids
            and seeds not in source_seeds
            and trigger not in trigger_hashes,
            "replacement claim reuses source, seeds, or failure trigger",
        )
        source_ids.add(source_id)
        source_seeds.add(seeds)
        trigger_hashes.add(trigger)
        previous = str(record_hash)
        claims.append(claim)
    return {
        "genesis": genesis,
        "claims": claims,
        "head_record_sha256": previous,
        "registry_path": registry_relative,
        "claims_directory": claims_relative,
    }


def _independent_failure_contract(
    policy: PathPolicy,
    *,
    role: str,
    dgp: str,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    attempt = policy.attempt_relative
    relative = f"{attempt}/data/{role}/{dgp}/rollout_failures.jsonl"
    candidate_path = policy.repository_root / relative
    if not os.path.lexists(candidate_path):
        return []
    path = policy.path(relative, scope="attempt")
    metadata = path.lstat()
    _require(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and int(metadata.st_nlink) == 1,
        "rollout failure log is linked, aliased, or non-regular",
    )
    raw = path.read_bytes()
    _require(bool(raw) and raw.endswith(b"\n"), "present failure log is empty or partial")
    authorization_relative = (
        f"{attempt}/audit/pre_data_inheritance_seal.json"
        if role in ("fit", "selection")
        else f"{attempt}/audit/pre_confirmation_package_seal.json"
    )
    authorization_state = (
        "PRE_OUTCOME_SEAL"
        if role in ("fit", "selection")
        else "PRE_CONFIRMATION_PACKAGE_SEAL"
    )
    authorization_path = policy.path(authorization_relative, scope="attempt")
    authorization = {
        "path": authorization_relative,
        "sha256": sha256_file(authorization_path),
        "checkpoint_state": authorization_state,
    }
    primary_by_destination = {
        (str(item["episode_id"]), int(item["slot"])): item for item in primary
    }
    replacements_by_id = {str(item["episode_id"]): item for item in replacements}
    expected_keys = {
        "schema_version", "attempt", "created_unix_ns", "classification", "role",
        "regime", "slot", "episode_id", "attempted_seed_source_episode_id",
        "replacement_used", "replacement_claim_index", "replacement_claim_sha256",
        *SEED_FIELDS, "exception_type", "exception_message", "traceback",
        "replacement_permitted", "retention_contract", "authorization_seal", "seq",
        "prev_sha256", "record_sha256",
    }
    previous = f"authorization:{authorization['sha256']}"
    observed_attempts: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for sequence, encoded in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise VerificationError(f"invalid failure record {sequence}") from error
        _require(
            isinstance(value, dict)
            and encoded
            == json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"),
            f"noncanonical failure record {sequence}",
        )
        _require(set(value) == expected_keys, f"failure record {sequence} schema drift")
        body = dict(value)
        record_hash = body.pop("record_sha256", None)
        replacement_used = value.get("replacement_used")
        _require(type(replacement_used) is bool, "failure replacement_used is not exact boolean")
        destination = primary_by_destination.get(
            (str(value.get("episode_id")), value.get("slot"))
        )
        source_id = str(value.get("attempted_seed_source_episode_id"))
        source = replacements_by_id.get(source_id) if replacement_used else destination
        _require(destination is not None and source is not None, "failure seed source is unsealed")
        _require(
            type(value.get("schema_version")) is int
            and value.get("schema_version") == 2
            and value.get("attempt") == ACTIVE_ATTEMPT
            and type(value.get("created_unix_ns")) is int
            and int(value["created_unix_ns"]) > 0
            and value.get("classification") == "mechanical_rollout_exception"
            and value.get("role") == role
            and value.get("regime") == dgp
            and type(value.get("slot")) is int
            and source_id == source.get("episode_id")
            and _independent_seed_tuple(value) == _independent_seed_tuple(source)
            and isinstance(value.get("exception_type"), str)
            and bool(value.get("exception_type"))
            and isinstance(value.get("exception_message"), str)
            and isinstance(value.get("traceback"), str)
            and value.get("replacement_permitted") is True
            and value.get("retention_contract")
            == "pixels_action_shapes_finiteness_and_local_step_count_only"
            and value.get("authorization_seal") == authorization
            and type(value.get("seq")) is int
            and value.get("seq") == sequence
            and value.get("prev_sha256") == previous
            and record_hash == canonical_object_sha256(body),
            f"failure record {sequence} authentication drift",
        )
        if replacement_used:
            claim_index = value.get("replacement_claim_index")
            _require(
                type(claim_index) is int and 0 <= int(claim_index) < len(claims),
                "replacement failure claim index drift",
            )
            claim = claims[int(claim_index)]
            _require(
                value.get("replacement_claim_sha256") == claim.get("record_sha256")
                and claim.get("role") == role
                and claim.get("regime") == dgp
                and claim.get("slot") == value.get("slot")
                and claim.get("episode_id") == value.get("episode_id")
                and claim.get("replacement_episode_id") == source_id,
                "replacement failure/claim binding drift",
            )
        else:
            _require(
                source_id == value.get("episode_id")
                and value.get("replacement_claim_index") is None
                and value.get("replacement_claim_sha256") is None,
                "primary failure claim/source drift",
            )
        attempt_key = (str(value["episode_id"]), source_id)
        _require(attempt_key not in observed_attempts, "failure source attempt reused")
        observed_attempts.add(attempt_key)
        previous = str(record_hash)
        result.append(value)
    return result


def _verify_replacement_registry_and_failures(
    policy: PathPolicy,
    cohort_ledger: Mapping[str, Any],
    retained: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    roles: Sequence[str],
) -> dict[str, Any]:
    registry = _independent_registry_contract(policy, cohort_ledger)
    claims = registry["claims"]
    retained_sources: set[str] = set()
    failed_sources_global: set[str] = set()
    dispositions: dict[int, str] = {}
    failure_count = 0
    for role in roles:
        for dgp in DGP_ORDER:
            records = list(retained[role][dgp])
            primary = cohort_ledger["regimes"][dgp]["roles"][role]["primary"][
                : len(records)
            ]
            replacements = cohort_ledger["regimes"][dgp]["roles"][role][
                "replacements"
            ]
            failures = _independent_failure_contract(
                policy,
                role=role,
                dgp=dgp,
                primary=primary,
                replacements=replacements,
                claims=claims,
            )
            failure_count += len(failures)
            failures_by_destination: dict[str, list[Mapping[str, Any]]] = {}
            for failure in failures:
                failures_by_destination.setdefault(
                    str(failure["episode_id"]), []
                ).append(failure)
                failed_sources_global.add(
                    str(failure["attempted_seed_source_episode_id"])
                )
            claims_by_destination: dict[str, list[Mapping[str, Any]]] = {}
            for claim in claims:
                if claim.get("role") == role and claim.get("regime") == dgp:
                    claims_by_destination.setdefault(
                        str(claim["episode_id"]), []
                    ).append(claim)
            for values in claims_by_destination.values():
                values.sort(key=lambda item: int(item["claim_index"]))
            record_ids: set[str] = set()
            for record, destination in zip(records, primary, strict=True):
                episode_id = str(destination["episode_id"])
                _require(str(record.get("episode_id")) == episode_id, "retained destination drift")
                _require(type(record.get("replacement_used")) is bool, "retained replacement_used is not exact boolean")
                retained_source = str(record.get("seed_source_episode_id"))
                retained_sources.add(retained_source)
                record_ids.add(episode_id)
                destination_failures = failures_by_destination.get(episode_id, [])
                destination_claims = claims_by_destination.get(episode_id, [])
                failed_sources = [
                    str(item["attempted_seed_source_episode_id"])
                    for item in destination_failures
                ]
                _require(
                    retained_source not in failed_sources,
                    "retained source has an authenticated failure",
                )
                if not destination_failures:
                    _require(not destination_claims, "replacement claim exists without failure")
                    _require(
                        record.get("replacement_used") is False
                        and retained_source == episode_id
                        and record.get("replacement_claim_index") is None
                        and record.get("replacement_claim_sha256") is None
                        and _independent_seed_tuple(record)
                        == _independent_seed_tuple(destination),
                        "failure-free destination did not retain exact primary",
                    )
                    continue
                _require(
                    record.get("replacement_used") is True
                    and len(destination_claims) == len(destination_failures),
                    "failed destination lacks exact replacement progression",
                )
                expected_failed_source = episode_id
                for failure, claim in zip(
                    destination_failures, destination_claims, strict=True
                ):
                    _require(
                        failure.get("attempted_seed_source_episode_id")
                        == expected_failed_source
                        and claim.get("failed_seed_source_episode_id")
                        == expected_failed_source
                        and claim.get("trigger_failure_record_sha256")
                        == failure.get("record_sha256")
                        and claim.get("trigger_failure_seq") == failure.get("seq"),
                        "independent failure/claim progression drift",
                    )
                    expected_failed_source = str(claim["replacement_episode_id"])
                    dispositions[int(claim["claim_index"])] = (
                        "failed"
                        if expected_failed_source in failed_sources
                        else "retained"
                    )
                terminal = destination_claims[-1]
                _require(
                    retained_source == expected_failed_source
                    and type(record.get("replacement_claim_index")) is int
                    and record.get("replacement_claim_index")
                    == terminal.get("claim_index")
                    and record.get("replacement_claim_sha256")
                    == terminal.get("record_sha256")
                    and _independent_seed_tuple(record)
                    == _independent_seed_tuple(terminal)
                    and dispositions[int(terminal["claim_index"])] == "retained",
                    "retained replacement is not terminal failure-free claim",
                )
            _require(
                not (set(failures_by_destination) - record_ids),
                "failure targets an unretained destination",
            )
            _require(
                not (set(claims_by_destination) - record_ids),
                "claim targets an unretained destination",
            )
    role_claim_indexes = {
        int(claim["claim_index"])
        for claim in claims
        if claim.get("role") in roles
    }
    _require(
        set(dispositions) == role_claim_indexes,
        "verified role claims are not the exact failed-or-retained closure",
    )
    _require(
        not (retained_sources & failed_sources_global),
        "a globally retained source appears in authenticated failures",
    )
    return {
        "registry_claim_count": len(claims),
        "verified_role_claim_count": len(role_claim_indexes),
        "authenticated_failure_count": failure_count,
        "retained_sources_failure_free": True,
        "claim_dispositions": {
            str(index): dispositions[index] for index in sorted(dispositions)
        },
        "registry_head_record_sha256": registry["head_record_sha256"],
    }


def verify_role_manifests(
    contract: Mapping[str, Any],
    policy: PathPolicy,
    cohort_ledger: Mapping[str, Any],
    confirmation_n: int,
    roles_to_verify: Sequence[str] = ROLE_ORDER,
) -> dict[str, Any]:
    manifests = contract.get("role_manifests")
    _require(isinstance(manifests, Mapping), "role_manifests contract absent")
    used_sources: set[str] = set()
    roles = tuple(roles_to_verify)
    _require(all(role in ROLE_ORDER for role in roles), "unknown manifest role")
    created: dict[str, dict[str, int]] = {role: {} for role in roles}
    retained: dict[str, dict[str, Sequence[Mapping[str, Any]]]] = {
        role: {} for role in roles
    }
    totals: dict[str, int] = {}
    for role in roles:
        _require(tuple(manifests.get(role, {})) == DGP_ORDER, f"{role} role-manifest DGP order drift")
        expected_count = confirmation_n if role == "confirmation" else ROLE_COUNTS[role]
        totals[role] = 0
        for dgp in DGP_ORDER:
            path = policy.path(manifests[role][dgp], scope="study")
            manifest = read_json(path)
            _require(manifest.get("attempt") == ACTIVE_ATTEMPT, "role manifest attempt drift")
            _require(manifest.get("role") == role, f"role manifest role drift: {dgp}/{role}")
            _require(manifest.get("regime", manifest.get("dgp_id")) == dgp, "role manifest DGP drift")
            episodes = manifest.get("episodes")
            _require(isinstance(episodes, list) and len(episodes) == expected_count, f"role manifest count drift: {dgp}/{role}")
            retained[role][dgp] = episodes
            primary = cohort_ledger["regimes"][dgp]["roles"][role]["primary"]
            replacements = cohort_ledger["regimes"][dgp]["roles"][role]["replacements"]
            replacement_by_id = {str(item["episode_id"]): item for item in replacements}
            maximum_created = 0
            for slot, record in enumerate(episodes):
                _require(
                    type(record.get("slot")) is int and record["slot"] == slot,
                    f"role manifest slot drift: {dgp}/{role}",
                )
                expected_primary = primary[slot]
                _require(str(record["episode_id"]) == str(expected_primary["episode_id"]), "role manifest primary ID drift")
                replacement_used = record.get("replacement_used")
                _require(
                    type(replacement_used) is bool,
                    "role manifest replacement_used is not exact boolean",
                )
                source_id = str(record.get("seed_source_episode_id", record["episode_id"]))
                source = replacement_by_id.get(source_id) if replacement_used else expected_primary
                _require(source is not None, "role manifest replacement source is unassigned")
                _require(source_id == str(source["episode_id"]), "role manifest seed-source ID drift")
                _require(source_id not in used_sources, "episode seed source reused across roles/DGPs")
                used_sources.add(source_id)
                for key in ("env_seed", "policy_seed", "oracle_np_seed", "action_space_seed"):
                    _require(int(record[key]) == int(source[key]), f"role manifest seed drift: {key}")
                artifact_raw = record.get("path", record.get("raw_path"))
                artifact_hash = record.get("sha256", record.get("raw_sha256"))
                _require(isinstance(artifact_raw, str) and isinstance(artifact_hash, str), "role episode artifact link absent")
                artifact = policy.path(artifact_raw, scope="study")
                _require(sha256_file(artifact) == artifact_hash, "role episode artifact hash drift")
                maximum_created = max(maximum_created, int(record.get("created_unix_ns", 0)))
            created[role][dgp] = max(maximum_created, int(manifest.get("created_unix_ns", 0)))
            totals[role] += len(episodes)
    expected_totals = {
        role: (4 * confirmation_n if role == "confirmation" else 4 * ROLE_COUNTS[role])
        for role in roles
    }
    _require(totals == expected_totals, "global role counts drift")
    replacement_evidence = _verify_replacement_registry_and_failures(
        policy, cohort_ledger, retained, roles
    )
    return {
        "counts": totals,
        "maximum_created_unix_ns": created,
        "replacement_registry_and_failures": replacement_evidence,
    }


def build_causal_features_numpy(
    history: np.ndarray,
    action_history: np.ndarray,
    current_prediction: np.ndarray,
    last_update: np.ndarray,
) -> np.ndarray:
    """Independent NumPy transcription of the frozen 1,046-feature graph."""

    history_array = np.asarray(history)
    action_array = np.asarray(action_history)
    current = np.asarray(current_prediction)
    update = np.asarray(last_update)
    _require(history_array.ndim == 3 and history_array.shape[1:] == (HISTORY_LEN, LATENT_DIM), "history primitive shape drift")
    _require(action_array.shape == (len(history_array), HISTORY_LEN, ACTION_DIM), "action-history primitive shape drift")
    _require(current.shape == (len(history_array), LATENT_DIM), "current primitive shape drift")
    _require(update.shape == current.shape, "update primitive shape drift")
    _require(all(np.issubdtype(value.dtype, np.floating) for value in (history_array, action_array, current, update)), "feature primitives must be floating")
    dtype = current.dtype
    epsilon = np.finfo(dtype).eps
    last_history = history_array[:, -1]
    gap = current - last_history
    norm = lambda value: np.sqrt(np.sum(np.square(value), axis=-1, keepdims=True, dtype=dtype), dtype=dtype)
    current_norm = norm(current)
    update_norm = norm(update)
    last_history_norm = norm(last_history)
    gap_norm = norm(gap)
    relative_update = update_norm / np.maximum(current_norm, epsilon)
    update_current_dot = np.sum(update * current, axis=-1, keepdims=True, dtype=dtype)
    update_current_cosine = update_current_dot / np.maximum(update_norm * current_norm, epsilon)
    update_gap_dot = np.sum(update * gap, axis=-1, keepdims=True, dtype=dtype)
    update_gap_cosine = update_gap_dot / np.maximum(update_norm * gap_norm, epsilon)
    history_change = history_array[:, 1:] - history_array[:, :-1]
    action_change = action_array[:, 1:] - action_array[:, :-1]
    history_change_norm = np.sqrt(np.sum(np.square(history_change), axis=2, dtype=dtype), dtype=dtype)
    action_change_norm = np.sqrt(np.sum(np.square(action_change), axis=2, dtype=dtype), dtype=dtype)
    features = np.concatenate(
        (
            history_array.reshape(len(history_array), -1),
            action_array.reshape(len(history_array), -1),
            current,
            update,
            current_norm,
            update_norm,
            relative_update,
            last_history_norm,
            gap_norm,
            update_current_cosine,
            update_gap_cosine,
            history_change_norm,
            action_change_norm,
        ),
        axis=1,
    )
    _require(features.shape == (len(history_array), FEATURE_DIM), "causal feature width drift")
    _require(np.isfinite(features).all(), "reconstructed causal features are nonfinite")
    return features


def reconstruct_scores_calls(
    arrays: Mapping[str, np.ndarray],
    weights: np.ndarray,
    biases: np.ndarray,
    thresholds: np.ndarray,
    *,
    feature_rtol: float = 2e-6,
    feature_atol: float = 2e-7,
    score_rtol: float = 2e-6,
    score_atol: float = 2e-7,
) -> dict[str, Any]:
    calls_recorded = np.asarray(arrays["calls"], dtype=np.int64)
    rows = len(calls_recorded)
    production_features = np.asarray(arrays["production_features"])
    recorded_scores = np.asarray(arrays["scores"])
    _require(production_features.shape == (rows, STAGE_COUNT, FEATURE_DIM), "production feature shape drift")
    _require(recorded_scores.shape == (rows, STAGE_COUNT), "recorded score shape drift")
    head_count = weights.shape[1]
    _require(weights.shape == (STAGE_COUNT, head_count, FEATURE_DIM), "compiled weight shape drift")
    _require(biases.shape == (STAGE_COUNT, head_count), "compiled bias shape drift")
    _require(thresholds.shape == (STAGE_COUNT,), "compiled threshold shape drift")
    history = np.asarray(arrays["history"])
    action = np.asarray(arrays["action_history"])
    stage_current = np.asarray(arrays["stage_current"])
    stage_update = np.asarray(arrays["stage_update"])
    if history.ndim == 3:
        _require(history.shape == (rows, HISTORY_LEN, LATENT_DIM), "history shape drift")
    else:
        _require(history.shape == (rows, STAGE_COUNT, HISTORY_LEN, LATENT_DIM), "stage history shape drift")
    if action.ndim == 3:
        _require(action.shape == (rows, HISTORY_LEN, ACTION_DIM), "action history shape drift")
    else:
        _require(action.shape == (rows, STAGE_COUNT, HISTORY_LEN, ACTION_DIM), "stage action history shape drift")
    _require(stage_current.shape == (rows, STAGE_COUNT, LATENT_DIM), "stage-current shape drift")
    _require(stage_update.shape == stage_current.shape, "stage-update shape drift")
    active = np.ones(rows, dtype=bool)
    reconstructed_calls = np.ones(rows, dtype=np.int64)
    reconstructed_scores = np.full((rows, STAGE_COUNT), np.nan, dtype=np.float64)
    maximum_feature_delta = 0.0
    maximum_score_delta = 0.0
    reached_counts = []
    for stage in range(STAGE_COUNT):
        reached = active.copy()
        reached_counts.append(int(reached.sum()))
        local_history = history[reached] if history.ndim == 3 else history[reached, stage]
        local_action = action[reached] if action.ndim == 3 else action[reached, stage]
        reconstructed_features = build_causal_features_numpy(
            local_history,
            local_action,
            stage_current[reached, stage],
            stage_update[reached, stage],
        )
        observed_features = production_features[reached, stage]
        _require(np.allclose(reconstructed_features, observed_features, rtol=feature_rtol, atol=feature_atol), f"causal feature reconstruction mismatch at stage {stage+1}")
        if reconstructed_features.size:
            maximum_feature_delta = max(maximum_feature_delta, float(np.max(np.abs(reconstructed_features.astype(np.float64) - observed_features.astype(np.float64)))))
        _require(np.isnan(production_features[~reached, stage]).all(), "unreached production features must be NaN")
        # Runtime gate tensors are float32.  NumPy matrix multiplication here is
        # deliberately independent of the production Torch implementation.
        feature_runtime = np.asarray(reconstructed_features, dtype=np.float32)
        head_scores = feature_runtime @ np.asarray(weights[stage], dtype=np.float32).T
        head_scores = head_scores + np.asarray(biases[stage], dtype=np.float32)[None, :]
        score = np.min(head_scores, axis=1).astype(np.float64)
        reconstructed_scores[reached, stage] = score
        observed_score = recorded_scores[reached, stage].astype(np.float64)
        _require(np.allclose(score, observed_score, rtol=score_rtol, atol=score_atol), f"gate score reconstruction mismatch at stage {stage+1}")
        if score.size:
            maximum_score_delta = max(maximum_score_delta, float(np.max(np.abs(score - observed_score))))
        _require(np.isnan(recorded_scores[~reached, stage]).all(), "unreached gate scores must be NaN")
        active = reached.copy()
        active[reached] = score > float(thresholds[stage])
        reconstructed_calls[active] += 1
    _require(np.array_equal(reconstructed_calls, calls_recorded), "calls do not reconstruct from primitives and frozen gate")
    return {
        "calls": reconstructed_calls,
        "scores": reconstructed_scores,
        "maximum_feature_absolute_delta": maximum_feature_delta,
        "maximum_score_absolute_delta": maximum_score_delta,
        "reached_counts": reached_counts,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    order = np.argsort(vector, kind="mergesort")
    ranks = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and vector[order[stop]] == vector[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denominator) if denominator else float("nan")


def episode_means(values: np.ndarray, slots: np.ndarray, rows_per_episode: int = ROWS_PER_EPISODE) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    episode = np.asarray(slots, dtype=np.int64)
    unique = np.unique(episode)
    _require(np.array_equal(unique, np.arange(len(unique))), "episode slots are not contiguous")
    result = np.empty(len(unique), dtype=np.float64)
    for slot in unique:
        local = vector[episode == slot]
        _require(len(local) == rows_per_episode, f"episode {slot} row count drift")
        result[int(slot)] = local.sum(dtype=np.float64) / rows_per_episode
    return result


def exact_compute(calls: np.ndarray, head_count: int) -> dict[str, Any]:
    values = np.asarray(calls, dtype=np.int64)
    _require(values.ndim == 1 and len(values) > 0, "calls must be a nonempty vector")
    _require(np.isin(values, (1, 2, 3, 4)).all(), "call depth outside 1..4")
    _require(head_count in (2, 8), "head count outside candidate family")
    rows = len(values)
    solver_calls = int(values.sum(dtype=np.int64))
    gate_evaluations = int(np.minimum(values, 3).sum(dtype=np.int64))
    feature = gate_evaluations * FEATURE_FLOPS
    head = gate_evaluations * head_count * AFFINE_HEAD_FLOPS
    additional = (solver_calls - rows) * ADDITIONAL_REFINER_FLOPS
    total = rows * (BASE_FLOPS + MANDATORY_DEPTH1_FLOPS) + additional + feature + head
    return {
        "rows": rows,
        "base_model_calls": rows,
        "refiner_model_calls": solver_calls,
        "mean_refiner_calls": float(solver_calls / rows),
        "gate_evaluations": gate_evaluations,
        "base_flops": rows * BASE_FLOPS,
        "mandatory_depth1_flops": rows * MANDATORY_DEPTH1_FLOPS,
        "additional_refiner_flops": additional,
        "gate_feature_flops": feature,
        "gate_head_flops": head,
        "gate_total_flops": feature + head,
        "gate_nonflop_operations": gate_evaluations * GATE_NONFLOPS[head_count],
        "adaptive_total_counted_flops": total,
        "analytic_equivalent_total_refiner_calls": float(solver_calls + (feature + head) / ADDITIONAL_REFINER_FLOPS),
        "call_histogram": np.bincount(values, minlength=5)[1:].astype(int).tolist(),
    }


def endpoint_losses(
    target: np.ndarray,
    exits: np.ndarray,
    selected: np.ndarray,
    whitening: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    target64 = np.asarray(target, dtype=np.float64)
    exits64 = np.asarray(exits, dtype=np.float64)
    selected64 = np.asarray(selected, dtype=np.float64)
    matrix = np.asarray(whitening, dtype=np.float64)
    _require(target64.ndim == 2 and target64.shape[1] == LATENT_DIM, "target shape drift")
    _require(exits64.shape == (len(target64), EXIT_COUNT, LATENT_DIM), "dense exit shape drift")
    _require(selected64.shape == target64.shape, "selected output shape drift")
    _require(matrix.shape == (LATENT_DIM, LATENT_DIM), "fixed whitening shape drift")
    _require(all(np.isfinite(value).all() for value in (target64, exits64, selected64, matrix)), "nonfinite endpoint input")
    dense_delta = exits64 - target64[:, None, :]
    adaptive_delta = selected64 - target64
    dense_white = np.einsum("nkd,df->nkf", dense_delta, matrix, optimize=False)
    adaptive_white = np.einsum("nd,df->nf", adaptive_delta, matrix, optimize=False)
    return (
        {
            "raw": np.mean(np.square(dense_delta), axis=2, dtype=np.float64),
            "fixed_whitened": np.mean(np.square(dense_white), axis=2, dtype=np.float64),
        },
        {
            "raw": np.mean(np.square(adaptive_delta), axis=1, dtype=np.float64),
            "fixed_whitened": np.mean(np.square(adaptive_white), axis=1, dtype=np.float64),
        },
    )


def strongest_analytic(losses: np.ndarray, total_flops: int) -> dict[str, Any]:
    values = np.asarray(losses, dtype=np.float64)
    rows = len(values)
    _require(values.shape == (rows, EXIT_COUNT) and rows > 0, "analytic loss shape drift")
    depth_costs = np.asarray(
        [BASE_FLOPS + MANDATORY_DEPTH1_FLOPS + (depth - 1) * ADDITIONAL_REFINER_FLOPS for depth in range(1, 5)],
        dtype=np.float64,
    )
    candidates: list[tuple[float, int, int, Fraction, np.ndarray]] = []
    means = values.mean(axis=0, dtype=np.float64)
    for lower in range(EXIT_COUNT):
        for upper in range(lower, EXIT_COUNT):
            if lower == upper:
                if int(total_flops) != rows * int(depth_costs[lower]):
                    continue
                rational = Fraction(0, 1)
            else:
                low_total = rows * int(depth_costs[lower])
                high_total = rows * int(depth_costs[upper])
                if not low_total <= int(total_flops) <= high_total:
                    continue
                rational = Fraction(int(total_flops) - low_total, high_total - low_total)
            weight = float(rational)
            mixture = values[:, lower] + weight * (values[:, upper] - values[:, lower])
            objective = float(means[lower] + weight * (means[upper] - means[lower]))
            candidates.append((objective, lower + 1, upper + 1, rational, mixture))
    _require(bool(candidates), "no exact-compute analytic allocation")
    objective, lower, upper, rational, mixture = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    low_total = rows * int(depth_costs[lower - 1])
    delta_total = rows * (int(depth_costs[upper - 1]) - int(depth_costs[lower - 1]))
    _require(
        (int(total_flops) - low_total) * rational.denominator
        == delta_total * rational.numerator,
        "analytic integer cross-multiplication mismatch",
    )
    allocation_numerator = rows * (
        int(depth_costs[lower - 1]) * rational.denominator
        + rational.numerator * (int(depth_costs[upper - 1]) - int(depth_costs[lower - 1]))
    )
    return {
        "loss": mixture,
        "mean_loss": objective,
        "depth_lower": lower,
        "depth_upper": upper,
        "weight_upper": float(rational),
        "weight_upper_numerator": int(rational.numerator),
        "weight_upper_denominator": int(rational.denominator),
        "allocation_total_counted_flops": int(total_flops),
        "allocation_compute_numerator": int(allocation_numerator),
        "allocation_compute_denominator": int(rational.denominator),
        "integer_cross_multiplication_equality": allocation_numerator == int(total_flops) * rational.denominator,
        "exact_total_compute_match": True,
    }


def seeded_control(
    losses: np.ndarray,
    allocation: Mapping[str, Any],
    adaptive_flops: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(losses, dtype=np.float64)
    rows = len(values)
    lower = int(allocation["depth_lower"])
    upper = int(allocation["depth_upper"])
    cost = lambda depth: BASE_FLOPS + MANDATORY_DEPTH1_FLOPS + (depth - 1) * ADDITIONAL_REFINER_FLOPS
    lower_total = rows * cost(lower)
    if lower == upper:
        _require(lower_total >= adaptive_flops, "degenerate seeded allocation under budget")
        count_upper = 0
    else:
        count_upper = int(math.ceil((adaptive_flops - lower_total) / (cost(upper) - cost(lower))))
        count_upper = min(rows, max(0, count_upper))
    calls = np.full(rows, lower, dtype=np.int64)
    if count_upper:
        calls[np.random.default_rng(int(seed)).permutation(rows)[:count_upper]] = upper
    selected = values[np.arange(rows), calls - 1]
    total = int(sum(cost(int(depth)) for depth in calls))
    _require(total >= adaptive_flops, "seeded comparator is not weakly-more-compute")
    return {"loss": selected, "calls": calls, "total_counted_flops": total, "weakly_more_compute": True}


def randomized_histogram_calls(calls: np.ndarray, slots: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(calls, dtype=np.int64)
    episode = np.asarray(slots, dtype=np.int64)
    output = np.empty_like(values)
    rng = np.random.default_rng(int(seed))
    for slot in np.unique(episode):
        indices = np.flatnonzero(episode == slot)
        output[indices] = values[indices][rng.permutation(len(indices))]
        _require(np.array_equal(np.bincount(output[indices], minlength=5), np.bincount(values[indices], minlength=5)), "within-episode histogram drift")
    return output


def analyze_confirmation_regime(
    arrays: Mapping[str, np.ndarray],
    whitening: np.ndarray,
    gate: Mapping[str, np.ndarray],
    *,
    seeded_seeds: Mapping[str, int],
    histogram_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reconstruction = reconstruct_scores_calls(
        arrays,
        np.asarray(gate["weights"]),
        np.asarray(gate["biases"]),
        np.asarray(gate["thresholds"]),
    )
    calls = reconstruction["calls"]
    slots = np.asarray(arrays["episode_slot"], dtype=np.int64)
    steps = np.asarray(arrays["model_step"], dtype=np.int64)
    episode_count = len(np.unique(slots))
    _require(
        np.array_equal(steps, np.tile(np.arange(3, 3 + ROWS_PER_EPISODE), episode_count)),
        "confirmation model-step order drift",
    )
    exits = np.asarray(arrays["exits"])
    selected = np.asarray(arrays["selected"])
    selected_dense = exits[np.arange(len(calls)), calls - 1]
    _require(np.allclose(selected, selected_dense, rtol=1e-6, atol=1e-7), "sparse/dense selected-output mismatch")
    fixed, adaptive = endpoint_losses(arrays["target"], exits, selected, whitening)
    head_count = int(np.asarray(gate["weights"]).shape[1])
    compute = exact_compute(calls, head_count)
    histogram_calls = randomized_histogram_calls(calls, slots, int(histogram_seed))
    metrics: dict[str, np.ndarray] = {
        "episode_slot": np.arange(episode_count, dtype=np.int32),
        "adaptive_raw": episode_means(adaptive["raw"], slots),
        "adaptive_fixed_whitened": episode_means(adaptive["fixed_whitened"], slots),
    }
    allocations: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    positions = np.arange(len(calls))
    for endpoint in ENDPOINTS:
        allocation = strongest_analytic(fixed[endpoint], compute["adaptive_total_counted_flops"])
        seeded = seeded_control(
            fixed[endpoint], allocation, compute["adaptive_total_counted_flops"], int(seeded_seeds[endpoint])
        )
        row_contrasts = {
            f"{endpoint}_vs_analytic": allocation["loss"] - adaptive[endpoint],
            f"{endpoint}_vs_seeded_weakly_more_compute": seeded["loss"] - adaptive[endpoint],
            f"{endpoint}_vs_fixed_depth_1": fixed[endpoint][:, 0] - adaptive[endpoint],
            f"{endpoint}_vs_within_episode_histogram": fixed[endpoint][positions, histogram_calls - 1] - adaptive[endpoint],
        }
        metrics.update({name: episode_means(value, slots) for name, value in row_contrasts.items()})
        allocations[endpoint] = {key: value for key, value in allocation.items() if key != "loss"}
        controls[endpoint] = {
            "seeded_total_counted_flops": seeded["total_counted_flops"],
            "seeded_weakly_more_compute": seeded["weakly_more_compute"],
        }
    return {
        "episode_count": episode_count,
        "row_count": len(calls),
        "compute": compute,
        "allocations": allocations,
        "controls": controls,
        "reconstruction": {key: value for key, value in reconstruction.items() if key not in ("calls", "scores")},
    }, metrics


def bootstrap_all(
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    seed: int,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    chunk_size: int = BOOTSTRAP_CHUNK,
) -> dict[str, dict[str, np.ndarray]]:
    _require(tuple(metrics) == DGP_ORDER, "bootstrap DGP order drift")
    output = {
        dgp: {name: np.empty(replicates, dtype=np.float64) for name in ALL_CONTRASTS}
        for dgp in DGP_ORDER
    }
    rng = np.random.default_rng(int(seed))
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        for dgp in DGP_ORDER:
            episode_count = len(metrics[dgp][ALL_CONTRASTS[0]])
            sampled = rng.integers(0, episode_count, size=(stop - start, episode_count), dtype=np.int32)
            for name in ALL_CONTRASTS:
                values = np.asarray(metrics[dgp][name], dtype=np.float64)
                output[dgp][name][start:stop] = values[sampled].sum(axis=1, dtype=np.float64) / episode_count
    return output


def linear_quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    _require(ordered.ndim == 1 and len(ordered) > 0, "quantile input is empty")
    location = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def terminal_mapping(bounds: Mapping[str, Mapping[str, float]], integrity_valid: bool) -> tuple[str, int]:
    supported = sum(float(bounds[dgp][endpoint]) > 0.0 for dgp in DGP_ORDER for endpoint in ENDPOINTS)
    if not integrity_valid:
        return TERMINAL_INVALID, supported
    if supported == FAMILY_SIZE:
        return TERMINAL_CONFIRMED, supported
    if supported:
        return TERMINAL_PARTIAL, supported
    return TERMINAL_FAILED, supported


def fit_array_sha256(value: np.ndarray) -> str:
    """Digest used by the fit artifact (intentionally distinct from analysis)."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def candidate_object_sha256(
    candidate_id: str,
    architecture: str,
    ridge: float,
    quantile: float,
    weights: np.ndarray,
    biases: np.ndarray,
    thresholds: np.ndarray,
    head_names: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "architecture": architecture,
                "ridge": float(ridge),
                "fit_quantile": float(quantile),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for value in (weights, biases, thresholds, head_names):
        digest.update(fit_array_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _replay_strict_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} SHA-256 drift",
    )
    return value


def _replay_exact_link(
    value: Any,
    *,
    policy: PathPolicy,
    label: str,
    expected_path: str | None = None,
) -> dict[str, str]:
    _require(
        isinstance(value, Mapping) and set(value) == {"path", "sha256"},
        f"{label} link schema drift",
    )
    raw = _canonical_relative(value.get("path"))
    if expected_path is not None:
        _require(raw == expected_path, f"{label} path drift")
    digest = _replay_strict_sha256(value.get("sha256"), label)
    path = policy.path(raw, scope="repository")
    metadata = path.lstat()
    _require(
        not path.is_symlink()
        and stat.S_ISREG(metadata.st_mode)
        and int(metadata.st_nlink) == 1
        and sha256_file(path) == digest,
        f"{label} live identity drift",
    )
    return {"path": raw, "sha256": digest}


def _read_canonical_replay_json(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    value = read_json(path)
    expected = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _require(raw == expected, f"{label} is not canonical pretty JSON")
    return value


def _replay_roles(mode: str) -> tuple[str, ...]:
    if mode == "confirmation":
        return REPLAY_ROLE_ORDER
    _require(mode in ("no_candidate", "power_infeasible"), "unknown replay mode")
    return ("fit", "selection")


def _replay_episodes_per_dgp(
    role: str, *, state: Mapping[str, Any], confirmation_n: int
) -> int:
    if role in REPLAY_FIXED_EPISODES_PER_DGP:
        return REPLAY_FIXED_EPISODES_PER_DGP[role]
    _require(
        role == "confirmation"
        and type(confirmation_n) is int
        and confirmation_n > 0
        and type(state.get("expected_confirmation_episode_count")) is int
        and state.get("expected_confirmation_episode_count")
        == confirmation_n * len(DGP_ORDER),
        "scientific replay confirmation count drift",
    )
    return confirmation_n


def _replay_array_keys(role: str) -> frozenset[str]:
    if role in ("fit", "selection"):
        return REPLAY_DEVELOPMENT_ARRAY_KEYS
    if role == "smoke":
        return REPLAY_ROBUST_ARRAY_KEYS
    _require(role == "confirmation", "unknown replay array role")
    return REPLAY_ROBUST_ARRAY_KEYS | {"target"}


def _verify_replay_regime_result(
    result: Mapping[str, Any],
    *,
    role: str,
    dgp: str,
    episodes_per_dgp: int,
    policy: PathPolicy,
    terminal_files: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(result) == REPLAY_RESULT_KEYS, f"replay schema drift: {role}/{dgp}")
    _require(
        type(result.get("schema_version")) is int
        and result.get("schema_version") == 1
        and result.get("artifact_type")
        == "v008_independent_scientific_replay_regime"
        and result.get("attempt") == ACTIVE_ATTEMPT
        and result.get("role") == role
        and result.get("regime") == dgp
        and type(result.get("episode_count")) is int
        and result.get("episode_count") == episodes_per_dgp
        and type(result.get("row_count")) is int
        and result.get("row_count") == episodes_per_dgp * ROWS_PER_EPISODE,
        f"replay identity/count drift: {role}/{dgp}",
    )
    exact_flags = {
        "input_loader_agreement_exact": True,
        "every_persisted_tensor_exact": True,
        "target_loss_computed": False,
        "loss_or_effect_used_for_acceptance": False,
        "contact_or_privileged_materialized": False,
        "no_gradients": True,
        "artifact_tree_unchanged": True,
        "read_only": True,
        "passed": True,
    }
    _require(
        all(
            type(result.get(name)) is bool and result.get(name) is expected
            for name, expected in exact_flags.items()
        ),
        f"replay exact assertion drift: {role}/{dgp}",
    )
    raw_relative = f"{policy.attempt_relative}/data/{role}/{dgp}/raw_manifest.json"
    execution_relative = (
        f"{policy.attempt_relative}/data/{role}/{dgp}/execution_manifest.json"
    )
    raw_link = _replay_exact_link(
        result.get("raw_manifest"),
        policy=policy,
        label=f"{role}/{dgp} replay raw manifest",
        expected_path=raw_relative,
    )
    execution_link = _replay_exact_link(
        result.get("execution_manifest"),
        policy=policy,
        label=f"{role}/{dgp} replay execution manifest",
        expected_path=execution_relative,
    )
    for link in (raw_link, execution_link):
        _require(
            link["path"] in terminal_files
            and _record_hash(terminal_files[link["path"]]) == link["sha256"],
            f"terminal manifest omits replay input: {link['path']}",
        )
    raw_manifest = _read_canonical_replay_json(
        policy.path(raw_relative, scope="attempt"), f"{role}/{dgp} raw manifest"
    )
    execution_manifest = _read_canonical_replay_json(
        policy.path(execution_relative, scope="attempt"),
        f"{role}/{dgp} execution manifest",
    )
    for manifest, label in ((raw_manifest, "raw"), (execution_manifest, "execution")):
        _require(
            manifest.get("attempt") == ACTIVE_ATTEMPT
            and manifest.get("role") == role
            and manifest.get("regime") == dgp
            and manifest.get("complete") is True
            and type(manifest.get("episode_count")) is int
            and manifest.get("episode_count") == episodes_per_dgp,
            f"{role}/{dgp} {label} replay-manifest drift",
        )
    _require(
        type(execution_manifest.get("row_count")) is int
        and execution_manifest.get("row_count")
        == episodes_per_dgp * ROWS_PER_EPISODE,
        f"{role}/{dgp} replay execution row-count drift",
    )
    authorization = execution_manifest.get("authorization")
    _require(isinstance(authorization, Mapping), "replay authorization absent")
    authorization_hash = _replay_strict_sha256(
        authorization.get("state_sha256"), f"{role}/{dgp} replay authorization"
    )

    runtime = result.get("runtime_audit")
    runtime_checks = runtime.get("checks") if isinstance(runtime, Mapping) else None
    mps = runtime.get("mps") if isinstance(runtime, Mapping) else None
    _require(
        isinstance(runtime, Mapping)
        and type(runtime.get("schema_version")) is int
        and runtime.get("schema_version") == 1
        and runtime.get("passed") is True
        and runtime.get("requested_role") == "independent_verification"
        and runtime.get("runtime_role") == "evaluation"
        and isinstance(runtime_checks, Mapping)
        and bool(runtime_checks)
        and all(type(value) is bool and value for value in runtime_checks.values())
        and isinstance(mps, Mapping)
        and set(mps) == {"required", "built", "available"}
        and all(type(mps[name]) is bool and mps[name] for name in mps)
        and runtime.get("read_only_preflight") is True
        and type(runtime.get("seed_tuples_consumed")) is int
        and runtime.get("seed_tuples_consumed") == 0
        and type(runtime.get("output_paths_created")) is int
        and runtime.get("output_paths_created") == 0,
        f"replay runtime/MPS drift: {role}/{dgp}",
    )
    module_before = result.get("module_before")
    _require(
        isinstance(module_before, Mapping)
        and module_before.get("passed") is True
        and result.get("module_after") == module_before,
        f"replay module drift: {role}/{dgp}",
    )
    sources = result.get("source_hashes")
    _require(isinstance(sources, Mapping) and bool(sources), "replay sources absent")
    verified_sources: dict[str, str] = {}
    for raw, digest in sources.items():
        relative = _canonical_relative(raw)
        observed = _replay_strict_sha256(digest, f"replay source {relative}")
        path = policy.path(relative, scope="repository")
        metadata = path.lstat()
        _require(
            not path.is_symlink()
            and stat.S_ISREG(metadata.st_mode)
            and int(metadata.st_nlink) == 1
            and sha256_file(path) == observed,
            f"replay source live drift: {relative}",
        )
        if relative.startswith(f"{policy.attempt_relative}/"):
            _require(
                relative in terminal_files
                and _record_hash(terminal_files[relative]) == observed,
                f"terminal manifest omits local replay source: {relative}",
            )
        verified_sources[relative] = observed
    _require(
        {
            f"{policy.attempt_relative}/scientific_replay.py",
            f"{policy.attempt_relative}/scientific_replay_launcher.py",
        }
        <= set(verified_sources),
        "replay source audit omits its implementation",
    )

    raw_episodes = raw_manifest.get("episodes")
    execution_episodes = execution_manifest.get("episodes")
    replay_episodes = result.get("episodes")
    _require(
        type(raw_episodes) is list
        and type(execution_episodes) is list
        and type(replay_episodes) is list
        and len(raw_episodes) == len(execution_episodes) == len(replay_episodes)
        == episodes_per_dgp,
        f"replay episode count/list drift: {role}/{dgp}",
    )
    expected_array_keys = _replay_array_keys(role)
    episode_ids: list[str] = []
    file_index: list[dict[str, Any]] = []
    for slot, (raw_record, execution_record, replay_record) in enumerate(
        zip(raw_episodes, execution_episodes, replay_episodes, strict=True)
    ):
        _require(
            isinstance(raw_record, Mapping)
            and isinstance(execution_record, Mapping)
            and isinstance(replay_record, Mapping)
            and set(replay_record) == REPLAY_EPISODE_KEYS
            and type(raw_record.get("slot")) is int
            and raw_record.get("slot") == slot
            and type(execution_record.get("slot")) is int
            and execution_record.get("slot") == slot
            and type(replay_record.get("slot")) is int
            and replay_record.get("slot") == slot
            and isinstance(replay_record.get("episode_id"), str)
            and bool(replay_record["episode_id"])
            and replay_record.get("episode_id") == raw_record.get("episode_id")
            == execution_record.get("episode_id")
            and replay_record.get("every_persisted_tensor_exact") is True,
            f"replay episode identity drift: {role}/{dgp}/{slot}",
        )
        _replay_strict_sha256(
            replay_record.get("input_loader_audit_sha256"),
            f"{role}/{dgp}/{slot} loader audit",
        )
        arrays = replay_record.get("array_sha256")
        _require(
            isinstance(arrays, Mapping)
            and set(arrays) == expected_array_keys
            and all(
                bool(_replay_strict_sha256(value, f"{role}/{dgp}/{slot}/{name}"))
                for name, value in arrays.items()
            ),
            f"replay array-hash schema drift: {role}/{dgp}/{slot}",
        )
        raw_path = _canonical_relative(raw_record.get("raw_path"))
        raw_sidecar = PurePosixPath(raw_path).with_suffix(".json").as_posix()
        expected_links = {
            "raw": {"path": raw_path, "sha256": raw_record.get("raw_sha256")},
            "raw_sidecar": {
                "path": raw_sidecar,
                "sha256": execution_record.get("source_raw_sidecar_sha256"),
            },
            "execution_part": {
                "path": execution_record.get("path"),
                "sha256": execution_record.get("sha256"),
            },
            "execution_sidecar": {
                "path": execution_record.get("sidecar_path"),
                "sha256": execution_record.get("sidecar_sha256"),
            },
        }
        verified_links: dict[str, Any] = {}
        for name, expected in expected_links.items():
            _require(
                replay_record.get(name) == expected,
                f"replay episode link drift: {role}/{dgp}/{slot}/{name}",
            )
            link = _replay_exact_link(
                replay_record[name],
                policy=policy,
                label=f"{role}/{dgp}/{slot} {name}",
            )
            _require(
                link["path"] in terminal_files
                and _record_hash(terminal_files[link["path"]]) == link["sha256"],
                f"terminal manifest omits replay episode file: {link['path']}",
            )
            verified_links[name] = link
        episode_id = str(replay_record["episode_id"])
        episode_ids.append(episode_id)
        file_index.append({"episode_id": episode_id, **verified_links})

    aggregate = result.get("aggregate")
    _require(
        isinstance(aggregate, Mapping)
        and set(aggregate)
        == {"applicable", "path", "sha256", "arrays", "exact"},
        f"replay aggregate schema drift: {role}/{dgp}",
    )
    if role in ("fit", "selection"):
        aggregate_arrays = aggregate.get("arrays")
        _require(
            aggregate.get("applicable") is True
            and aggregate.get("path") == execution_manifest.get("aggregate_role_path")
            and aggregate.get("sha256")
            == execution_manifest.get("aggregate_role_sha256")
            and aggregate.get("exact") is True
            and result.get("aggregate_exact") is True
            and isinstance(aggregate_arrays, Mapping)
            and set(aggregate_arrays) == DEVELOPMENT_AGGREGATE_KEYS
            and all(
                bool(_replay_strict_sha256(value, f"{role}/{dgp} aggregate {name}"))
                for name, value in aggregate_arrays.items()
            )
            and result.get("compiled_gate") is None,
            f"development replay aggregate drift: {role}/{dgp}",
        )
        aggregate_link = _replay_exact_link(
            {"path": aggregate["path"], "sha256": aggregate["sha256"]},
            policy=policy,
            label=f"{role}/{dgp} replay aggregate",
        )
        _require(
            aggregate_link["path"] in terminal_files
            and _record_hash(terminal_files[aggregate_link["path"]])
            == aggregate_link["sha256"],
            f"terminal manifest omits replay aggregate: {role}/{dgp}",
        )
    else:
        _require(
            dict(aggregate)
            == {
                "applicable": False,
                "path": None,
                "sha256": None,
                "arrays": None,
                "exact": None,
            }
            and result.get("aggregate_exact") is None,
            f"robust replay aggregate drift: {role}/{dgp}",
        )
        gate = _replay_exact_link(
            result.get("compiled_gate"),
            policy=policy,
            label=f"{role}/{dgp} replay gate",
        )
        _require(
            execution_manifest.get("compiled_gate_path") == gate["path"]
            and execution_manifest.get("compiled_gate_sha256") == gate["sha256"],
            f"replay gate/manifest drift: {role}/{dgp}",
        )
    return {
        "result": dict(result),
        "raw_manifest": raw_link,
        "execution_manifest": execution_link,
        "aggregate": dict(aggregate),
        "episode_count": episodes_per_dgp,
        "row_count": episodes_per_dgp * ROWS_PER_EPISODE,
        "authorization_state_sha256": authorization_hash,
        "episode_ids_sha256": canonical_object_sha256(episode_ids),
        "verified_file_index_sha256": canonical_object_sha256(file_index),
        "source_hashes": dict(sorted(verified_sources.items())),
    }


def _independent_development_replay_binding(
    role: str, policy: PathPolicy
) -> dict[str, Any]:
    name = {"fit": "fit_cohorts.json", "selection": "selection_cohorts.json"}.get(role)
    _require(name is not None, f"no development replay checkpoint for {role}")
    path = policy.path(
        f"{policy.attempt_relative}/audit/{name}", scope="attempt"
    )
    checkpoint = _read_canonical_replay_json(path, f"{role} role checkpoint")
    evidence = checkpoint.get("evidence")
    verification = evidence.get("role_verification") if isinstance(evidence, Mapping) else None
    _require(
        isinstance(verification, Mapping) and verification.get("role") == role,
        f"{role} stable role verification absent",
    )
    regimes = verification.get("regimes")
    _require(
        isinstance(regimes, Mapping) and set(regimes) == set(DGP_ORDER),
        f"{role} stable role DGP drift",
    )
    stable_regimes: dict[str, Any] = {}
    for dgp in DGP_ORDER:
        item = regimes[dgp]
        _require(isinstance(item, Mapping), f"{role}/{dgp} stable item drift")
        raw = item.get("raw_manifest")
        execution = item.get("execution_manifest")
        aggregate = item.get("aggregate")
        _require(
            isinstance(raw, Mapping)
            and set(raw) == {"path", "sha256"}
            and isinstance(execution, Mapping)
            and set(execution) == {"path", "sha256"}
            and isinstance(aggregate, Mapping)
            and set(aggregate) == {"path", "sha256", "bytes"}
            and type(aggregate.get("bytes")) is int
            and aggregate["bytes"] > 0
            and type(item.get("episode_count")) is int
            and type(item.get("row_count")) is int,
            f"{role}/{dgp} stable manifest schema drift",
        )
        for label, link in (("raw", raw), ("execution", execution)):
            _canonical_relative(link.get("path"))
            _replay_strict_sha256(link.get("sha256"), f"{role}/{dgp} {label}")
        _canonical_relative(aggregate.get("path"))
        _replay_strict_sha256(aggregate.get("sha256"), f"{role}/{dgp} aggregate")
        stable_regimes[dgp] = {
            "raw_manifest": dict(raw),
            "execution_manifest": dict(execution),
            "aggregate": dict(aggregate),
            "episode_count": item["episode_count"],
            "row_count": item["row_count"],
            "episode_ids_sha256": _replay_strict_sha256(
                item.get("episode_ids_sha256"), f"{role}/{dgp} IDs"
            ),
            "verified_file_index_sha256": _replay_strict_sha256(
                item.get("verified_file_index_sha256"), f"{role}/{dgp} file index"
            ),
        }
    stable = {
        "role": role,
        "state": REPLAY_ROLE_STATES[role],
        "regime_order": list(DGP_ORDER),
        "episodes_per_regime": verification.get("episodes_per_regime"),
        "episode_count": verification.get("episode_count"),
        "row_count": verification.get("row_count"),
        "authorization_state_sha256": _replay_strict_sha256(
            verification.get("authorization_state_sha256"),
            f"{role} authorization",
        ),
        "regimes": stable_regimes,
    }
    qualification = verification.get("scientific_replay_qualification")
    _require(
        isinstance(qualification, Mapping)
        and qualification.get("passed") is True
        and qualification.get("role") == role
        and qualification.get("state") == REPLAY_ROLE_STATES[role]
        and qualification.get("manifest_verification_sha256")
        == canonical_object_sha256(stable),
        f"{role} checkpoint replay qualification drift",
    )
    return stable


def _independent_robust_replay_binding(
    role: str,
    *,
    episodes_per_dgp: int,
    authorization_state_sha256: str,
    regimes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "role": role,
        "state": REPLAY_ROLE_STATES[role],
        "regime_order": list(DGP_ORDER),
        "episodes_per_regime": episodes_per_dgp,
        "episode_count": episodes_per_dgp * len(DGP_ORDER),
        "row_count": episodes_per_dgp * len(DGP_ORDER) * ROWS_PER_EPISODE,
        "authorization_state_sha256": authorization_state_sha256,
        "regimes": {
            dgp: {
                "raw_manifest": regimes[dgp]["raw_manifest"],
                "execution_manifest": regimes[dgp]["execution_manifest"],
                "aggregate": regimes[dgp]["aggregate"],
                "episode_count": regimes[dgp]["episode_count"],
                "row_count": regimes[dgp]["row_count"],
                "episode_ids_sha256": regimes[dgp]["episode_ids_sha256"],
                "verified_file_index_sha256": regimes[dgp][
                    "verified_file_index_sha256"
                ],
            }
            for dgp in DGP_ORDER
        },
    }


def verify_scientific_replay_qualification(
    contract: Mapping[str, Any],
    policy: PathPolicy,
    state: Mapping[str, Any],
    *,
    confirmation_n: int,
) -> dict[str, Any]:
    """Authenticate terminal-rerun MPS evidence before any scientific NPZ."""

    mode = str(contract["mode"])
    roles = _replay_roles(mode)
    terminal_path = policy.path(
        contract["paths"]["terminal_manifest"], scope="attempt"
    )
    terminal = read_json(terminal_path)
    terminal_files = _manifest_files(terminal)
    audit_relative = (
        f"{policy.attempt_relative}/{SCIENTIFIC_REPLAY_QUALIFICATION_RELATIVE}"
    )
    audit_path = policy.path(audit_relative, scope="attempt")
    _require(
        audit_relative in terminal_files
        and _record_hash(terminal_files[audit_relative]) == sha256_file(audit_path),
        "terminal manifest omits scientific replay qualification",
    )
    audit = _read_canonical_replay_json(
        audit_path, "terminal scientific replay qualification"
    )
    expected_audit_keys = {
        "schema_version", "artifact_type", "attempt", "mode",
        "checkpoint_state", "role_order", "regime_order", "role_count",
        "qualification_count", "episode_count", "row_count", "role_audits",
        "source_hashes", "controller_snapshot",
        "controller_unchanged_during_replay", "input_loader_agreement_exact",
        "every_persisted_tensor_exact",
        "all_applicable_development_aggregates_exact", "all_runtime_mps_exact",
        "target_loss_computed", "loss_or_effect_used_for_acceptance",
        "contact_or_privileged_materialized", "no_gradients",
        "artifact_trees_unchanged", "read_only",
        "terminal_manifest_created_only_after_replay_qualification", "passed",
    }
    _require(set(audit) == expected_audit_keys, "terminal replay-audit schema drift")
    for field in (
        "schema_version", "role_count", "qualification_count", "episode_count",
        "row_count",
    ):
        _require(type(audit.get(field)) is int, f"terminal replay integer type drift: {field}")
    exact_flags = {
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
    _require(
        all(
            type(audit.get(name)) is bool and audit.get(name) is expected
            for name, expected in exact_flags.items()
        ),
        "terminal replay exact assertions drift",
    )
    _require(
        audit.get("schema_version") == 1
        and audit.get("artifact_type")
        == "v008_terminal_scientific_replay_qualification"
        and audit.get("attempt") == ACTIVE_ATTEMPT
        and audit.get("mode") == mode
        and audit.get("checkpoint_state") == "INDEPENDENT_VERIFICATION"
        and type(audit.get("role_order")) is list
        and audit.get("role_order") == list(roles)
        and type(audit.get("regime_order")) is list
        and audit.get("regime_order") == list(DGP_ORDER)
        and audit.get("role_count") == len(roles)
        and audit.get("qualification_count") == len(roles) * len(DGP_ORDER),
        "terminal replay identity/order drift",
    )

    snapshot = audit.get("controller_snapshot")
    _require(
        isinstance(snapshot, Mapping)
        and set(snapshot)
        == {"active_attempt", "current_state", "state", "ledger", "outcome_counts"},
        "terminal replay controller snapshot schema drift",
    )
    state_record = snapshot.get("state")
    ledger_record = snapshot.get("ledger")
    counts = snapshot.get("outcome_counts")
    _require(
        isinstance(state_record, Mapping)
        and set(state_record) == {"path", "sha256", "bytes", "object_sha256"}
        and isinstance(ledger_record, Mapping)
        and set(ledger_record)
        == {"path", "sha256", "bytes", "event_count", "head_sha256"}
        and isinstance(counts, Mapping)
        and set(counts)
        == {
            "fit_outcome_episodes", "selection_outcome_episodes",
            "smoke_outcome_episodes", "confirmation_outcome_episodes_generated",
            "confirmation_outcome_episodes_executed",
            "confirmation_outcomes_opened_for_analysis",
        },
        "terminal replay controller record schema drift",
    )
    state_path = policy.path(contract["paths"]["state"], scope="study")
    ledger_path = policy.path(contract["paths"]["ledger"], scope="study")
    _require(
        snapshot.get("active_attempt") == ACTIVE_ATTEMPT
        and snapshot.get("current_state") == "INDEPENDENT_VERIFICATION"
        and state_record.get("path") == contract["paths"]["state"]
        and _replay_strict_sha256(state_record.get("sha256"), "replay STATE")
        == sha256_file(state_path)
        and type(state_record.get("bytes")) is int
        and state_record.get("bytes") == state_path.stat().st_size
        and _replay_strict_sha256(
            state_record.get("object_sha256"), "replay STATE object"
        )
        == canonical_object_sha256(state)
        and ledger_record.get("path") == contract["paths"]["ledger"]
        and _replay_strict_sha256(ledger_record.get("sha256"), "replay ledger")
        == sha256_file(ledger_path)
        and type(ledger_record.get("bytes")) is int
        and ledger_record.get("bytes") == ledger_path.stat().st_size
        and type(ledger_record.get("event_count")) is int
        and ledger_record.get("event_count") == state.get("ledger_event_count")
        and ledger_record.get("head_sha256") == state.get("ledger_head_sha256"),
        "terminal replay controller/live chronology drift",
    )
    for field, value in counts.items():
        expected = state.get(field)
        if field == "confirmation_outcomes_opened_for_analysis":
            _require(type(value) is bool, f"replay count type drift: {field}")
        else:
            _require(type(value) is int and value >= 0, f"replay count type drift: {field}")
        _require(value == expected, f"replay controller count drift: {field}")

    role_audits = audit.get("role_audits")
    _require(
        isinstance(role_audits, Mapping) and set(role_audits) == set(roles),
        "terminal replay role-audit membership drift",
    )
    common_sources: dict[str, str] | None = None
    verified_roles: dict[str, Any] = {}
    total_episodes = 0
    total_rows = 0
    role_record_keys = {
        "path", "sha256", "role", "state", "authorization_state_sha256",
        "manifest_verification_sha256", "episode_count", "row_count", "passed",
    }
    for role in roles:
        episodes_per_dgp = _replay_episodes_per_dgp(
            role, state=state, confirmation_n=confirmation_n
        )
        role_record = role_audits[role]
        _require(
            isinstance(role_record, Mapping) and set(role_record) == role_record_keys,
            f"{role} terminal replay role-link schema drift",
        )
        role_relative = f"{policy.attempt_relative}/audit/{role}_scientific_replay.json"
        role_link = _replay_exact_link(
            {"path": role_record.get("path"), "sha256": role_record.get("sha256")},
            policy=policy,
            label=f"{role} replay role audit",
            expected_path=role_relative,
        )
        _require(
            role_relative in terminal_files
            and _record_hash(terminal_files[role_relative]) == role_link["sha256"],
            f"terminal manifest omits {role} replay role audit",
        )
        role_audit = _read_canonical_replay_json(
            policy.path(role_relative, scope="attempt"), f"{role} replay role audit"
        )
        _require(
            set(role_audit) == REPLAY_ROLE_AUDIT_KEYS,
            f"{role} replay role-audit closed schema drift",
        )
        for field in ("schema_version", "episodes_per_regime", "episode_count", "row_count"):
            _require(
                type(role_audit.get(field)) is int,
                f"{role} replay role integer type drift: {field}",
            )
        role_flags = {
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
        _require(
            all(
                type(role_audit.get(name)) is bool
                and role_audit.get(name) is expected
                for name, expected in role_flags.items()
            ),
            f"{role} replay role exact assertions drift",
        )
        _require(
            role_audit.get("schema_version") == 1
            and role_audit.get("artifact_type")
            == "v008_independent_scientific_replay_role"
            and role_audit.get("attempt") == ACTIVE_ATTEMPT
            and role_audit.get("role") == role
            and role_audit.get("state") == REPLAY_ROLE_STATES[role]
            and type(role_audit.get("regime_order")) is list
            and role_audit.get("regime_order") == list(DGP_ORDER)
            and role_audit.get("episodes_per_regime") == episodes_per_dgp
            and role_audit.get("episode_count") == episodes_per_dgp * len(DGP_ORDER)
            and role_audit.get("row_count")
            == episodes_per_dgp * len(DGP_ORDER) * ROWS_PER_EPISODE,
            f"{role} replay role identity/count drift",
        )
        regimes_raw = role_audit.get("regimes")
        _require(
            isinstance(regimes_raw, Mapping) and set(regimes_raw) == set(DGP_ORDER),
            f"{role} replay role DGP membership drift",
        )
        regimes: dict[str, Any] = {}
        authorization_hash: str | None = None
        role_sources: dict[str, str] | None = None
        for dgp in DGP_ORDER:
            item = regimes_raw[dgp]
            _require(
                isinstance(item, Mapping) and set(item) == REPLAY_ROLE_REGIME_KEYS,
                f"{role}/{dgp} replay role record drift",
            )
            result_relative = (
                f"{policy.attempt_relative}/audit/scientific_replay/{role}_{dgp}.json"
            )
            result_link = _replay_exact_link(
                item.get("result"),
                policy=policy,
                label=f"{role}/{dgp} replay result",
                expected_path=result_relative,
            )
            _require(
                result_relative in terminal_files
                and _record_hash(terminal_files[result_relative])
                == result_link["sha256"],
                f"terminal manifest omits replay result: {role}/{dgp}",
            )
            result = _read_canonical_replay_json(
                policy.path(result_relative, scope="attempt"),
                f"{role}/{dgp} replay result",
            )
            verified = _verify_replay_regime_result(
                result,
                role=role,
                dgp=dgp,
                episodes_per_dgp=episodes_per_dgp,
                policy=policy,
                terminal_files=terminal_files,
            )
            expected_item = {
                "result": result_link,
                "raw_manifest": verified["raw_manifest"],
                "execution_manifest": verified["execution_manifest"],
                "aggregate": verified["aggregate"],
                "episode_count": verified["episode_count"],
                "row_count": verified["row_count"],
            }
            _require(dict(item) == expected_item, f"{role}/{dgp} role/result drift")
            observed_authorization = verified["authorization_state_sha256"]
            if authorization_hash is None:
                authorization_hash = observed_authorization
            else:
                _require(
                    authorization_hash == observed_authorization,
                    f"{role} replay authorization differs across DGPs",
                )
            observed_sources = verified["source_hashes"]
            if role_sources is None:
                role_sources = observed_sources
            else:
                _require(
                    role_sources == observed_sources,
                    f"{role} replay sources differ across DGPs",
                )
            regimes[dgp] = verified
        _require(
            authorization_hash is not None and role_sources is not None,
            f"empty replay role: {role}",
        )
        if role in ("fit", "selection"):
            stable = _independent_development_replay_binding(role, policy)
            _require(
                stable["authorization_state_sha256"] == authorization_hash
                and stable["episodes_per_regime"] == episodes_per_dgp
                and stable["episode_count"] == episodes_per_dgp * len(DGP_ORDER)
                and stable["row_count"]
                == episodes_per_dgp * len(DGP_ORDER) * ROWS_PER_EPISODE,
                f"{role} replay/development binding drift",
            )
            for dgp in DGP_ORDER:
                expected = stable["regimes"][dgp]
                observed = regimes[dgp]
                _require(
                    expected["raw_manifest"] == observed["raw_manifest"]
                    and expected["execution_manifest"]
                    == observed["execution_manifest"]
                    and expected["aggregate"]["path"]
                    == observed["aggregate"]["path"]
                    and expected["aggregate"]["sha256"]
                    == observed["aggregate"]["sha256"]
                    and expected["episode_count"] == observed["episode_count"]
                    and expected["row_count"] == observed["row_count"],
                    f"{role}/{dgp} replay/checkpoint projection drift",
                )
        else:
            stable = _independent_robust_replay_binding(
                role,
                episodes_per_dgp=episodes_per_dgp,
                authorization_state_sha256=authorization_hash,
                regimes=regimes,
            )
        manifest_digest = canonical_object_sha256(stable)
        _require(
            role_audit.get("authorization_state_sha256") == authorization_hash
            and role_audit.get("manifest_verification_sha256") == manifest_digest
            and role_audit.get("source_hashes") == role_sources,
            f"{role} replay role binding/source drift",
        )
        expected_role_record = {
            "path": role_relative,
            "sha256": sha256_file(policy.path(role_relative, scope="attempt")),
            "role": role,
            "state": REPLAY_ROLE_STATES[role],
            "authorization_state_sha256": authorization_hash,
            "manifest_verification_sha256": manifest_digest,
            "episode_count": episodes_per_dgp * len(DGP_ORDER),
            "row_count": episodes_per_dgp * len(DGP_ORDER) * ROWS_PER_EPISODE,
            "passed": True,
        }
        _require(
            dict(role_record) == expected_role_record,
            f"{role} terminal replay role link drift",
        )
        if common_sources is None:
            common_sources = role_sources
        else:
            _require(common_sources == role_sources, "replay sources differ across roles")
        total_episodes += expected_role_record["episode_count"]
        total_rows += expected_role_record["row_count"]
        verified_roles[role] = {
            "role_audit": role_audit,
            "regimes": regimes,
            "manifest_binding": stable,
        }

    _require(
        common_sources is not None
        and audit.get("source_hashes") == common_sources
        and audit.get("episode_count") == total_episodes
        and audit.get("row_count") == total_rows,
        "terminal replay global source/count drift",
    )
    if mode in ("no_candidate", "power_infeasible"):
        for role in ("smoke", "confirmation"):
            role_path = policy.path(
                f"{policy.attempt_relative}/audit/{role}_scientific_replay.json",
                scope="attempt",
                must_exist=False,
            )
            _require(
                not role_path.exists() and not role_path.is_symlink(),
                f"early replay contains later role audit: {role}",
            )
    return {
        "path": audit_relative,
        "sha256": sha256_file(audit_path),
        "mode": mode,
        "role_order": list(roles),
        "qualification_count": len(roles) * len(DGP_ORDER),
        "episode_count": total_episodes,
        "row_count": total_rows,
        "roles": verified_roles,
        "controller_snapshot": dict(snapshot),
        "passed": True,
    }


def _load_npz(path: Path, *, exact_keys: frozenset[str] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        names = frozenset(stored.files)
        forbidden = sorted(name for name in names if any(token in name.lower() for token in FORBIDDEN_TOKENS))
        _require(not forbidden, f"forbidden array names in {path}: {forbidden}")
        if exact_keys is not None:
            _require(names == exact_keys, f"array schema drift in {path}: missing={sorted(exact_keys-names)} extra={sorted(names-exact_keys)}")
        return {name: stored[name].copy() for name in stored.files}


def _verify_dense_causal_features(arrays: Mapping[str, np.ndarray], label: str) -> float:
    observed = np.asarray(arrays["production_features"])
    rows = len(observed)
    _require(observed.shape == (rows, STAGE_COUNT, FEATURE_DIM), f"{label} feature shape drift")
    history = np.asarray(arrays["history"])
    action = np.asarray(arrays["action_history"])
    current = np.asarray(arrays["stage_current"])
    update = np.asarray(arrays["stage_update"])
    _require(history.shape == (rows, HISTORY_LEN, LATENT_DIM), f"{label} history shape drift")
    _require(action.shape == (rows, HISTORY_LEN, ACTION_DIM), f"{label} action-history shape drift")
    _require(current.shape == (rows, STAGE_COUNT, LATENT_DIM), f"{label} stage-current shape drift")
    _require(update.shape == current.shape, f"{label} stage-update shape drift")
    maximum = 0.0
    for stage in range(STAGE_COUNT):
        rebuilt = build_causal_features_numpy(history, action, current[:, stage], update[:, stage])
        _require(np.allclose(rebuilt, observed[:, stage], rtol=2e-6, atol=2e-7), f"{label} causal feature mismatch at stage {stage+1}")
        maximum = max(maximum, float(np.max(np.abs(rebuilt.astype(np.float64) - observed[:, stage].astype(np.float64)), initial=0.0)))
    return maximum


def load_development_role(
    raw_paths: Mapping[str, Any], manifest_paths: Mapping[str, Any], policy: PathPolicy,
    *, episodes_per_dgp: int, role: str, replay_role: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _require(
        type(episodes_per_dgp) is int and episodes_per_dgp > 0,
        f"{role} development episode-count type drift",
    )
    _require(role in {"fit", "selection"}, "unsupported development role")
    _require(
        isinstance(replay_role, Mapping)
        and replay_role.get("role_audit", {}).get("passed") is True
        and isinstance(replay_role.get("regimes"), Mapping)
        and tuple(replay_role["regimes"]) == DGP_ORDER,
        f"{role} exact scientific replay evidence is absent",
    )
    _require(tuple(raw_paths) == DGP_ORDER, f"{role} input DGP order drift")
    _require(tuple(manifest_paths) == DGP_ORDER, f"{role} development-manifest DGP order drift")
    manifest_keys = {
        "schema_version", "attempt", "created_unix_ns", "role", "regime",
        "complete", "episode_count", "row_count", "episodes",
        "aggregate_role_path", "aggregate_role_sha256", "aggregate_recovery",
        "aggregate_role_keys", "raw_manifest_path", "raw_manifest_sha256",
        "source_raw_manifest_sha256", "authorization", "source_bindings",
        "loaded_input_keys", "contact_or_privileged_materialized",
        "target_role_isolation", "causal_primitive_contract", "module_before",
        "module_after", "base_provenance", "frozen_facade", "no_gradients",
    }
    episode_record_keys = {
        "slot", "episode_id", "path", "sha256", "sidecar_path",
        "sidecar_sha256", "source_raw_path", "source_raw_sha256",
        "source_raw_sidecar_path", "source_raw_sidecar_sha256",
        "call_histogram",
    }
    part_contract = {
        "episode_slot": ((ROWS_PER_EPISODE,), "int32"),
        "model_step": ((ROWS_PER_EPISODE,), "int16"),
        "target": ((ROWS_PER_EPISODE, LATENT_DIM), "float32"),
        "exits": ((ROWS_PER_EPISODE, EXIT_COUNT, LATENT_DIM), "float32"),
        "production_features": (
            (ROWS_PER_EPISODE, STAGE_COUNT, FEATURE_DIM), "float32"
        ),
        "history": ((ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM), "float32"),
        "action_history": (
            (ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM), "float32"
        ),
        "stage_current": (
            (ROWS_PER_EPISODE, STAGE_COUNT, LATENT_DIM), "float32"
        ),
        "stage_update": (
            (ROWS_PER_EPISODE, STAGE_COUNT, LATENT_DIM), "float32"
        ),
    }
    causal_contract = {
        "history": [ROWS_PER_EPISODE, HISTORY_LEN, LATENT_DIM],
        "action_history": [ROWS_PER_EPISODE, HISTORY_LEN, ACTION_DIM],
        "stage_current": [ROWS_PER_EPISODE, STAGE_COUNT, LATENT_DIM],
        "stage_update": [ROWS_PER_EPISODE, STAGE_COUNT, LATENT_DIM],
        "sufficient_for_independent_feature_reconstruction": True,
    }

    def strict_sha256(value: Any, label: str) -> str:
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"{label} SHA-256 drift",
        )
        return value

    def strict_array_contract(
        arrays: Mapping[str, np.ndarray],
        contract: Mapping[str, tuple[tuple[int, ...], str]],
        label: str,
    ) -> None:
        _require(set(arrays) == set(contract), f"{label} member-key drift")
        for name, (shape, dtype) in contract.items():
            value = np.asarray(arrays[name])
            _require(
                value.shape == shape and str(value.dtype) == dtype,
                f"{label} shape/dtype drift: {name}",
            )

    parts: dict[str, list[np.ndarray]] = {
        "features": [], "exits": [], "target": [], "episode": [], "step": [], "dgp": []
    }
    hashes: dict[str, str] = {}
    maximum_delta = 0.0
    for dgp_index, dgp in enumerate(DGP_ORDER):
        replay_regime = replay_role["regimes"][dgp]
        replay_result = replay_regime.get("result") if isinstance(replay_regime, Mapping) else None
        _require(
            isinstance(replay_result, Mapping)
            and replay_result.get("passed") is True
            and replay_result.get("role") == role
            and replay_result.get("regime") == dgp
            and type(replay_result.get("episode_count")) is int
            and replay_result.get("episode_count") == episodes_per_dgp,
            f"{role}/{dgp} replay result binding drift",
        )
        path = policy.path(raw_paths[dgp], scope="study")
        replay_aggregate = replay_result.get("aggregate")
        _require(
            isinstance(replay_aggregate, Mapping)
            and replay_aggregate.get("applicable") is True
            and replay_aggregate.get("exact") is True
            and replay_aggregate.get("path") == raw_paths[dgp]
            and replay_aggregate.get("sha256") == sha256_file(path)
            and isinstance(replay_aggregate.get("arrays"), Mapping)
            and set(replay_aggregate["arrays"]) == DEVELOPMENT_AGGREGATE_KEYS,
            f"{role}/{dgp} replay aggregate binding drift",
        )
        arrays = _load_npz(path, exact_keys=DEVELOPMENT_AGGREGATE_KEYS)
        rows = episodes_per_dgp * ROWS_PER_EPISODE
        aggregate_contract = {
            name: ((rows, *shape[1:]), dtype)
            for name, (shape, dtype) in part_contract.items()
            if name in DEVELOPMENT_AGGREGATE_KEYS
        }
        strict_array_contract(arrays, aggregate_contract, f"{role}/{dgp} aggregate")
        for name in DEVELOPMENT_AGGREGATE_KEYS:
            _require(
                replay_array_sha256(arrays[name])
                == replay_aggregate["arrays"][name],
                f"{role}/{dgp} aggregate differs from scientific replay: {name}",
            )
        hashes[dgp] = sha256_file(path)
        manifest_path = policy.path(manifest_paths[dgp], scope="study")
        manifest = read_json(manifest_path)
        _require(
            set(manifest) == manifest_keys,
            f"{role}/{dgp} development manifest closed schema drift",
        )
        _require(
            type(manifest.get("schema_version")) is int
            and manifest.get("schema_version") == 1
            and manifest.get("attempt") == ACTIVE_ATTEMPT
            and type(manifest.get("created_unix_ns")) is int
            and manifest["created_unix_ns"] > 0
            and manifest.get("role") == role
            and manifest.get("regime") == dgp
            and manifest.get("complete") is True
            and type(manifest.get("episode_count")) is int
            and manifest.get("episode_count") == episodes_per_dgp
            and type(manifest.get("row_count")) is int
            and manifest.get("row_count") == rows,
            f"{role}/{dgp} development manifest header/type/count drift",
        )
        _require(
            manifest.get("aggregate_role_path") == raw_paths[dgp]
            and strict_sha256(
                manifest.get("aggregate_role_sha256"),
                f"{role}/{dgp} aggregate manifest",
            )
            == hashes[dgp]
            and isinstance(manifest.get("aggregate_recovery"), Mapping)
            and type(manifest.get("aggregate_role_keys")) is list
            and manifest.get("aggregate_role_keys")
            == [
                "episode_slot", "model_step", "target", "exits",
                "production_features",
            ]
            and isinstance(manifest.get("authorization"), Mapping)
            and isinstance(manifest.get("source_bindings"), Mapping)
            and type(manifest.get("loaded_input_keys")) is list
            and manifest.get("loaded_input_keys") == ["action", "pixels"]
            and manifest.get("contact_or_privileged_materialized") is False
            and manifest.get("target_role_isolation") == role
            and isinstance(manifest.get("module_before"), Mapping)
            and isinstance(manifest.get("module_after"), Mapping)
            and isinstance(manifest.get("base_provenance"), Mapping)
            and isinstance(manifest.get("frozen_facade"), Mapping)
            and manifest.get("no_gradients") is True,
            f"{role}/{dgp} development manifest structure/list drift",
        )
        observed_causal = manifest.get("causal_primitive_contract")
        _require(
            isinstance(observed_causal, Mapping)
            and set(observed_causal) == set(causal_contract)
            and observed_causal == causal_contract
            and all(
                type(observed_causal[name]) is list
                and all(type(value) is int for value in observed_causal[name])
                for name in ("history", "action_history", "stage_current", "stage_update")
            ),
            f"{role}/{dgp} causal numeric/list contract drift",
        )
        strict_sha256(
            manifest.get("raw_manifest_sha256"),
            f"{role}/{dgp} raw manifest",
        )
        strict_sha256(
            manifest.get("source_raw_manifest_sha256"),
            f"{role}/{dgp} source raw manifest",
        )
        episodes = manifest.get("episodes")
        _require(
            type(episodes) is list and len(episodes) == episodes_per_dgp,
            f"{role}/{dgp} development manifest count/list drift",
        )
        replay_episodes = replay_result.get("episodes")
        _require(
            type(replay_episodes) is list
            and len(replay_episodes) == episodes_per_dgp,
            f"{role}/{dgp} replay episode-list drift",
        )
        reconstructed_parts: dict[str, list[np.ndarray]] = {name: [] for name in DEVELOPMENT_AGGREGATE_KEYS}
        feature_part_paths: list[Path] = []
        for slot, record in enumerate(episodes):
            replay_episode = replay_episodes[slot]
            _require(
                type(record) is dict and set(record) == episode_record_keys,
                f"{role}/{dgp} development episode record schema drift",
            )
            _require(
                type(record.get("slot")) is int
                and record.get("slot") == slot
                and isinstance(record.get("episode_id"), str)
                and bool(record["episode_id"])
                and record.get("call_histogram") is None,
                f"{role}/{dgp} development episode identity/type drift",
            )
            for name in (
                "sha256", "sidecar_sha256", "source_raw_sha256",
                "source_raw_sidecar_sha256",
            ):
                strict_sha256(record.get(name), f"{role}/{dgp}/{slot} {name}")
            for name in (
                "path", "sidecar_path", "source_raw_path",
                "source_raw_sidecar_path",
            ):
                _require(
                    isinstance(record.get(name), str) and bool(record[name]),
                    f"{role}/{dgp}/{slot} {name} type drift",
                )
            part_path = policy.path(record["path"], scope="study")
            _require(sha256_file(part_path) == record["sha256"], f"{role}/{dgp} development part hash drift")
            sidecar_path = policy.path(record["sidecar_path"], scope="study")
            _require(sha256_file(sidecar_path) == record["sidecar_sha256"], f"{role}/{dgp} development sidecar hash drift")
            _require(
                isinstance(replay_episode, Mapping)
                and replay_episode.get("slot") == slot
                and replay_episode.get("episode_id") == record["episode_id"]
                and replay_episode.get("execution_part")
                == {"path": record["path"], "sha256": record["sha256"]}
                and replay_episode.get("execution_sidecar")
                == {
                    "path": record["sidecar_path"],
                    "sha256": record["sidecar_sha256"],
                }
                and replay_episode.get("every_persisted_tensor_exact") is True
                and isinstance(replay_episode.get("array_sha256"), Mapping)
                and set(replay_episode["array_sha256"]) == DEVELOPMENT_KEYS,
                f"{role}/{dgp}/{slot} replay/part binding drift",
            )
            part = _load_npz(part_path, exact_keys=DEVELOPMENT_KEYS)
            strict_array_contract(part, part_contract, f"{role}/{dgp}/{slot} part")
            for name in DEVELOPMENT_KEYS:
                _require(
                    replay_array_sha256(part[name])
                    == replay_episode["array_sha256"][name],
                    f"{role}/{dgp}/{slot} part differs from scientific replay: {name}",
                )
            _require(
                np.array_equal(
                    part["episode_slot"],
                    np.full(ROWS_PER_EPISODE, slot, dtype=np.int32),
                ),
                f"{role}/{dgp} part slot drift",
            )
            _require(
                np.array_equal(
                    part["model_step"],
                    np.arange(3, 3 + ROWS_PER_EPISODE, dtype=np.int16),
                ),
                f"{role}/{dgp} part step drift",
            )
            _require(
                np.array_equal(part["stage_current"], part["exits"][:, :STAGE_COUNT]),
                f"{role}/{dgp} persisted exits/stage-current drift",
            )
            for stage in range(1, STAGE_COUNT):
                _require(
                    np.array_equal(
                        part["stage_current"][:, stage],
                        part["stage_current"][:, stage - 1]
                        + part["stage_update"][:, stage],
                    ),
                    f"{role}/{dgp} stage update/output drift at {stage}",
                )
            for name in DEVELOPMENT_AGGREGATE_KEYS:
                reconstructed_parts[name].append(part[name])
            feature_part_paths.append(part_path)
        for name in DEVELOPMENT_AGGREGATE_KEYS:
            rebuilt = np.concatenate(reconstructed_parts[name], axis=0)
            _require(np.array_equal(rebuilt, arrays[name]), f"{role}/{dgp} aggregate differs from sealed primitive parts: {name}")
        # Only after replay hashes and the complete part-to-aggregate projection
        # are exact may this verifier perform its own causal feature arithmetic.
        for slot, part_path in enumerate(feature_part_paths):
            feature_part = _load_npz(part_path, exact_keys=DEVELOPMENT_KEYS)
            maximum_delta = max(
                maximum_delta,
                _verify_dense_causal_features(
                    feature_part, f"{role}/{dgp}/{slot}"
                ),
            )
        _require(len(arrays["episode_slot"]) == rows, f"{role}/{dgp} row count drift")
        expected_episode = np.repeat(
            np.arange(episodes_per_dgp, dtype=np.int32), ROWS_PER_EPISODE
        )
        expected_step = np.tile(
            np.arange(3, 3 + ROWS_PER_EPISODE, dtype=np.int16), episodes_per_dgp
        )
        _require(np.array_equal(arrays["episode_slot"], expected_episode), f"{role}/{dgp} episode order drift")
        _require(np.array_equal(arrays["model_step"], expected_step), f"{role}/{dgp} step order drift")
        target = np.asarray(arrays["target"], dtype=np.float64)
        exits = np.asarray(arrays["exits"], dtype=np.float64)
        features = np.asarray(arrays["production_features"], dtype=np.float64)
        _require(target.shape == (rows, LATENT_DIM), f"{role}/{dgp} target shape drift")
        _require(exits.shape == (rows, EXIT_COUNT, LATENT_DIM), f"{role}/{dgp} exits shape drift")
        _require(all(np.isfinite(value).all() for value in (target, exits, features)), f"{role}/{dgp} nonfinite data")
        parts["features"].append(features)
        parts["exits"].append(exits)
        parts["target"].append(target)
        parts["episode"].append(expected_episode.astype(np.int64))
        parts["step"].append(expected_step.astype(np.int64))
        parts["dgp"].append(np.full(rows, dgp_index, dtype=np.int64))
    role_data = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    return role_data, {
        "input_sha256": hashes,
        "development_manifest_sha256": {
            dgp: sha256_file(policy.path(manifest_paths[dgp], scope="study")) for dgp in DGP_ORDER
        },
        "maximum_feature_absolute_delta": maximum_delta,
    }


def _fit_endpoint_losses(role: Mapping[str, np.ndarray], whitening: np.ndarray) -> np.ndarray:
    difference = role["exits"] - role["target"][:, None, :]
    raw = np.mean(np.square(difference), axis=2, dtype=np.float64)
    transformed = np.einsum("nkd,df->nkf", difference, whitening, optimize=True)
    fixed = np.mean(np.square(transformed), axis=2, dtype=np.float64)
    result = np.stack((raw, fixed), axis=2)
    _require(np.isfinite(result).all(), "development endpoint loss is nonfinite")
    return result


def _balanced_mean_std(values: np.ndarray, domains: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([values[domains == dgp].mean(axis=0) for dgp in range(4)])
    mean = means.mean(axis=0)
    second = np.stack([np.square(values[domains == dgp] - mean).mean(axis=0) for dgp in range(4)]).mean(axis=0)
    return mean, np.maximum(np.sqrt(np.maximum(second, 0.0)), floor)


def _domain_mean_std(values: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), np.maximum(values.std(axis=0), floor)


def _ridge_path(features: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    root = np.ones(len(features), dtype=np.float64) if weights is None else np.sqrt(weights)
    x = features * root[:, None]
    y = targets * root[:, None]
    output = np.empty((len(RIDGES), features.shape[1], targets.shape[1]), dtype=np.float64)
    if len(x) <= x.shape[1]:
        gram = x @ x.T
        identity = np.eye(len(x), dtype=np.float64)
        for index, ridge in enumerate(RIDGES):
            output[index] = x.T @ np.linalg.solve(gram + ridge * identity, y)
    else:
        gram = x.T @ x
        cross = x.T @ y
        identity = np.eye(x.shape[1], dtype=np.float64)
        for index, ridge in enumerate(RIDGES):
            output[index] = np.linalg.solve(gram + ridge * identity, cross)
    _require(np.isfinite(output).all(), "independent ridge result nonfinite")
    return output


def _compile_affine(weight: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, float]:
    raw = weight / std
    return raw, float(-np.dot(mean / std, weight))


def _fit_scores(features: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.min(features[:, stage] @ weights[stage].T + biases[stage][None, :], axis=1) for stage in range(STAGE_COUNT)],
        axis=1,
    )


def _calls_from_dense_scores(scores: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calls = np.ones(len(scores), dtype=np.int64)
    active = np.ones(len(scores), dtype=bool)
    reached = np.empty((len(scores), STAGE_COUNT), dtype=bool)
    for stage in range(STAGE_COUNT):
        reached[:, stage] = active
        active &= scores[:, stage] > thresholds[stage]
        calls += active.astype(np.int64)
    return calls, reached


def _candidate_grid() -> list[tuple[str, float, float, str]]:
    rows = []
    for architecture in ARCHITECTURES:
        for ridge in RIDGES:
            for quantile in FIT_QUANTILES:
                ridge_label = f"{ridge:g}".replace(".", "p")
                quantile_label = f"{quantile:.2f}".replace(".", "p")
                rows.append((architecture, ridge, quantile, f"{architecture}__ridge_{ridge_label}__q_{quantile_label}"))
    return rows


def recompute_fitted_candidates(
    role: Mapping[str, np.ndarray], fixed_whitening: np.ndarray
) -> dict[str, np.ndarray]:
    domains = role["dgp"]
    primary = _fit_endpoint_losses(role, fixed_whitening)
    gains = primary[:, :STAGE_COUNT] - primary[:, 1:]
    centered_target = role["target"] - role["target"].mean(axis=0)
    covariance = np.einsum("ni,nj->ij", centered_target, centered_target, optimize=False) / max(len(centered_target) - 1, 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    robust_floor = max(max(float(eigenvalues[-1]), 0.0) * 1e-6, 1e-12)
    robust_whitening = np.einsum("ik,k,jk->ij", eigenvectors, np.reciprocal(np.sqrt(np.maximum(eigenvalues, robust_floor))), eigenvectors, optimize=False)
    pooled_mean = np.empty((3, FEATURE_DIM)); pooled_std = np.empty_like(pooled_mean)
    domain_mean = np.empty((4, 3, FEATURE_DIM)); domain_std = np.empty_like(domain_mean)
    gain_mean = np.empty((4, 3, 2)); gain_std = np.empty_like(gain_mean)
    standardized = np.empty_like(gains)
    for stage in range(3):
        pooled_mean[stage], pooled_std[stage] = _balanced_mean_std(role["features"][:, stage], domains, 1e-6)
        for dgp in range(4):
            mask = domains == dgp
            domain_mean[dgp, stage], domain_std[dgp, stage] = _domain_mean_std(role["features"][mask, stage], 1e-6)
            gain_mean[dgp, stage], gain_std[dgp, stage] = _domain_mean_std(gains[mask, stage], 1e-12)
            standardized[mask, stage] = (gains[mask, stage] - gain_mean[dgp, stage]) / gain_std[dgp, stage]
    pooled_w = np.empty((3, 3, 2, FEATURE_DIM)); pooled_b = np.empty((3, 3, 2))
    counts = np.bincount(domains, minlength=4)
    balanced_weights = np.asarray([len(domains) / (4 * counts[value]) for value in domains])
    for stage in range(3):
        normalized = (role["features"][:, stage] - pooled_mean[stage]) / pooled_std[stage]
        path = _ridge_path(normalized, standardized[:, stage], balanced_weights)
        for ridge_index in range(3):
            for endpoint in range(2):
                pooled_w[ridge_index, stage, endpoint], pooled_b[ridge_index, stage, endpoint] = _compile_affine(path[ridge_index, :, endpoint], pooled_mean[stage], pooled_std[stage])
    envelope_w = np.empty((3, 3, 8, FEATURE_DIM)); envelope_b = np.empty((3, 3, 8))
    for stage in range(3):
        for dgp in range(4):
            mask = domains == dgp
            normalized = (role["features"][mask, stage] - domain_mean[dgp, stage]) / domain_std[dgp, stage]
            path = _ridge_path(normalized, standardized[mask, stage])
            for ridge_index in range(3):
                for endpoint in range(2):
                    head = 2 * dgp + endpoint
                    envelope_w[ridge_index, stage, head], envelope_b[ridge_index, stage, head] = _compile_affine(path[ridge_index, :, endpoint], domain_mean[dgp, stage], domain_std[dgp, stage])
    grid = _candidate_grid()
    weights = np.zeros((24, 3, 8, FEATURE_DIM)); biases = np.zeros((24, 3, 8)); thresholds = np.empty((24, 3))
    ids = np.asarray([row[3] for row in grid]); architectures = np.asarray([row[0] for row in grid])
    ridges = np.asarray([row[1] for row in grid]); quantiles = np.asarray([row[2] for row in grid]); heads = np.asarray([ARCHITECTURE_HEADS[row[0]] for row in grid])
    head_names = np.full((24, 8), "", dtype="<U64")
    fit_mean = np.empty((24, 4, 2)); fit_sd = np.empty_like(fit_mean); fit_count = np.empty((24, 4, 2), dtype=np.int64)
    fit_calls = np.empty((24, 4)); fit_reached = np.empty((24, 4, 3))
    score_cache: dict[tuple[str, float], np.ndarray] = {}
    for index, (architecture, ridge, quantile, _) in enumerate(grid):
        ridge_index = RIDGES.index(ridge); head_count = ARCHITECTURE_HEADS[architecture]
        source_w, source_b = (pooled_w[ridge_index], pooled_b[ridge_index]) if head_count == 2 else (envelope_w[ridge_index], envelope_b[ridge_index])
        weights[index, :, :head_count] = source_w; biases[index, :, :head_count] = source_b
        names = ENDPOINTS if head_count == 2 else tuple(f"{dgp}/{endpoint}" for dgp in DGP_ORDER for endpoint in ENDPOINTS)
        head_names[index, :head_count] = names
        cache_key = (architecture, ridge)
        if cache_key not in score_cache:
            score_cache[cache_key] = _fit_scores(role["features"], source_w, source_b)
        scores = score_cache[cache_key]
        active = np.ones(len(scores), dtype=bool); calls = np.ones(len(scores), dtype=np.int64); reached = np.empty((len(scores), 3), dtype=bool)
        for stage in range(3):
            reached[:, stage] = active
            thresholds[index, stage] = float(np.quantile(scores[active, stage], quantile, method="linear"))
            active &= scores[:, stage] > thresholds[index, stage]
            calls += active.astype(np.int64)
        for dgp in range(4):
            mask = domains == dgp; local_calls = calls[mask]; local = primary[mask]; slots = role["episode"][mask]
            compute = exact_compute(local_calls, head_count)
            for endpoint in range(2):
                analytic = strongest_analytic(local[:, :, endpoint], compute["adaptive_total_counted_flops"])
                adaptive = local[np.arange(len(local_calls)), local_calls - 1, endpoint]
                episodes = episode_means(analytic["loss"] - adaptive, slots)
                fit_mean[index, dgp, endpoint] = episodes.mean(); fit_sd[index, dgp, endpoint] = episodes.std(ddof=1); fit_count[index, dgp, endpoint] = len(episodes)
            fit_calls[index, dgp] = local_calls.mean(); fit_reached[index, dgp] = reached[mask].mean(axis=0)
    objects = np.asarray([
        candidate_object_sha256(str(ids[i]), str(architectures[i]), float(ridges[i]), float(quantiles[i]), weights[i, :, :int(heads[i])], biases[i, :, :int(heads[i])], thresholds[i], head_names[i, :int(heads[i])])
        for i in range(24)
    ])
    return {
        "dgp_ids": np.asarray(DGP_ORDER), "candidate_ids": ids, "candidate_object_sha256": objects,
        "architectures": architectures, "ridges": ridges, "fit_quantiles": quantiles, "head_count": heads,
        "head_names": head_names, "compiled_weights": weights, "compiled_biases": biases, "thresholds": thresholds,
        "fit_contrast_episode_mean": fit_mean, "fit_contrast_episode_sd": fit_sd, "fit_contrast_episode_count": fit_count,
        "fit_mean_calls": fit_calls, "fit_reached_fraction": fit_reached, "pooled_feature_mean": pooled_mean,
        "pooled_feature_std": pooled_std, "per_dgp_feature_mean": domain_mean, "per_dgp_feature_std": domain_std,
        "per_dgp_stage_endpoint_gain_mean": gain_mean, "per_dgp_stage_endpoint_gain_std": gain_std,
        "fixed_whitening_matrix": fixed_whitening, "robust_fit_whitening_matrix": robust_whitening,
        "robust_fit_target_mean": role["target"].mean(axis=0), "robust_fit_eigenvalue_floor": np.asarray(robust_floor),
        "feature_dim": np.asarray(FEATURE_DIM), "stage_count": np.asarray(STAGE_COUNT), "fit_row_count": np.asarray(len(domains)),
    }


def compare_fitted_artifact(recomputed: Mapping[str, np.ndarray], observed: Mapping[str, np.ndarray]) -> dict[str, Any]:
    required = set(recomputed) | {"schema_version"}
    _require(required.issubset(observed), f"fitted artifact missing arrays: {sorted(required-set(observed))}")
    maximum_score = 0.0
    maximum_primitive = 0.0
    for name, expected in recomputed.items():
        actual = np.asarray(observed[name])
        _require(actual.shape == expected.shape, f"fitted array shape drift: {name}")
        if np.issubdtype(expected.dtype, np.number):
            _require(np.allclose(actual, expected, rtol=2e-9, atol=2e-10, equal_nan=True), f"fitted numeric mismatch: {name}")
            if actual.size:
                finite = np.isfinite(actual) & np.isfinite(expected)
                if finite.any():
                    maximum = max(maximum, float(np.max(np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64)))))
        else:
            _require(np.array_equal(actual.astype(str), expected.astype(str)), f"fitted text mismatch: {name}")
    # Independently verify every stored candidate object's internal digest.
    for index in range(CANDIDATE_COUNT):
        head_count = int(observed["head_count"][index])
        digest = candidate_object_sha256(
            str(observed["candidate_ids"][index]), str(observed["architectures"][index]),
            float(observed["ridges"][index]), float(observed["fit_quantiles"][index]),
            observed["compiled_weights"][index, :, :head_count], observed["compiled_biases"][index, :, :head_count],
            observed["thresholds"][index], observed["head_names"][index, :head_count],
        )
        _require(digest == str(observed["candidate_object_sha256"][index]), f"candidate object hash drift at {index}")
    return {"candidate_count": CANDIDATE_COUNT, "maximum_numeric_absolute_delta": maximum, "all_candidate_objects_rehashed": True}


def canonical_analysis_seeds(cohort: Mapping[str, Any]) -> dict[str, Any]:
    groups = cohort["analysis_rng_ids"]
    bootstrap = groups["bootstrap"]
    records = groups["comparator"]["rng_ids"]
    lookup = {(str(item["regime"]), str(item["purpose"])): int(item["rng_id"]) for item in records}
    result = {
        "joint_bootstrap_seed": int(bootstrap["rng_ids"][0]["rng_id"]),
        "bootstrap_replicates": int(bootstrap["replicate_count"]),
        "seeded_comparator_seeds": {
            dgp: {
                "raw": lookup[(dgp, "seeded_weak_more_raw")],
                "fixed_whitened": lookup[(dgp, "seeded_weak_more_fixed_whitened")],
            }
            for dgp in DGP_ORDER
        },
        "histogram_seeds": {dgp: lookup[(dgp, "within_episode_call_histogram")] for dgp in DGP_ORDER},
        "aliases_only_no_additional_rng_ids": True,
    }
    _require(cohort.get("analysis_seeds") == result, "analysis seed aliases differ from assigned RNG identifiers")
    _require(result["bootstrap_replicates"] == BOOTSTRAP_REPLICATES, "bootstrap seed contract drift")
    return result


def _selection_candidate_core(
    role: Mapping[str, np.ndarray], fitted: Mapping[str, np.ndarray], candidate_index: int,
    seeds: Mapping[str, Any], fixed_losses: np.ndarray,
) -> dict[str, Any]:
    head_count = int(fitted["head_count"][candidate_index])
    architecture = str(fitted["architectures"][candidate_index])
    scores = _fit_scores(
        role["features"], fitted["compiled_weights"][candidate_index, :, :head_count],
        fitted["compiled_biases"][candidate_index, :, :head_count],
    )
    calls, reached = _calls_from_dense_scores(scores, fitted["thresholds"][candidate_index])
    standardized: list[float] = []
    rank_values: list[float] = []
    dgp_rows: list[dict[str, Any]] = []
    all_valid = True
    for dgp_index, dgp in enumerate(DGP_ORDER):
        mask = role["dgp"] == dgp_index
        local_calls = calls[mask]; local_reached = reached[mask]; local_scores = scores[mask]
        local_loss = fixed_losses[mask]; slots = role["episode"][mask]; positions = np.arange(len(local_calls))
        adaptive = local_loss[positions, local_calls - 1]
        compute = exact_compute(local_calls, head_count)
        analytic_records = []
        contrasts = []
        comparator_valid = True
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            analytic = strongest_analytic(local_loss[:, :, endpoint_index], compute["adaptive_total_counted_flops"])
            episode_contrast = episode_means(analytic["loss"] - adaptive[:, endpoint_index], slots)
            contrasts.append(float(episode_contrast.mean()))
            denominator = max(float(fitted["fit_contrast_episode_sd"][candidate_index, dgp_index, endpoint_index]), 1e-12)
            standardized.append(float(episode_contrast.mean() / denominator))
            seeded = seeded_control(local_loss[:, :, endpoint_index], analytic, compute["adaptive_total_counted_flops"], int(seeds["seeded_comparator_seeds"][dgp][endpoint]))
            comparator_valid &= bool(seeded["weakly_more_compute"])
            analytic_records.append({
                "depth_lower": analytic["depth_lower"], "depth_upper": analytic["depth_upper"],
                "weight_upper": analytic["weight_upper"], "mean": float(episode_contrast.mean()),
                "sd_ddof1": float(episode_contrast.std(ddof=1)), "episode_count": len(episode_contrast),
            })
        shuffled = randomized_histogram_calls(local_calls, slots, int(seeds["histogram_seeds"][dgp]))
        comparator_valid &= all(
            np.array_equal(np.bincount(shuffled[slots == slot], minlength=5), np.bincount(local_calls[slots == slot], minlength=5))
            for slot in np.unique(slots)
        )
        stage_rows = []
        for stage in range(3):
            gain = local_loss[:, stage] - local_loss[:, stage + 1]
            gain = (gain - fitted["per_dgp_stage_endpoint_gain_mean"][dgp_index, stage]) / fitted["per_dgp_stage_endpoint_gain_std"][dgp_index, stage]
            combined = gain.mean(axis=1)
            stage_mask = local_reached[:, stage]
            rho = spearman(local_scores[stage_mask, stage], combined[stage_mask])
            rank_values.append(rho)
            stage_rows.append({"reached_rows": int(stage_mask.sum()), "reached_fraction": float(stage_mask.mean()), "spearman": rho})
        finite = bool(np.isfinite(contrasts).all() and np.isfinite([item["spearman"] for item in stage_rows]).all())
        constraints = bool(1.05 <= local_calls.mean() <= 2.5 and stage_rows[1]["reached_fraction"] >= .05 and stage_rows[2]["reached_fraction"] >= .01)
        valid = bool(finite and comparator_valid and constraints)
        all_valid &= valid
        dgp_rows.append({
            "dgp_id": dgp, "mean_calls": float(local_calls.mean()),
            "call_histogram": np.bincount(local_calls, minlength=5)[1:].astype(int).tolist(),
            "stagewise_rank": stage_rows, "contrasts": dict(zip(ENDPOINTS, contrasts, strict=True)),
            "analytic": analytic_records, "flops_per_row": compute["adaptive_total_counted_flops"] / len(local_calls),
            "mechanically_valid": valid,
        })
    return {
        "candidate_id": str(fitted["candidate_ids"][candidate_index]), "eligible": bool(all_valid),
        "worst_standardized_exact_compute_contrast": min(standardized) if standardized else float("nan"),
        "worst_dgp_stage_spearman": min(rank_values) if rank_values else float("nan"),
        "worst_flops_per_row": max(row["flops_per_row"] for row in dgp_rows), "dgps": dgp_rows,
    }


def _close_float(left: Any, right: Any, label: str, *, tolerance: float = 5e-10) -> None:
    if left is None or right is None:
        other = right if left is None else left
        _require(other is None or not math.isfinite(float(other)), f"missing numeric selection field: {label}")
        return
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        _require(not math.isfinite(float(left)) and not math.isfinite(float(right)), f"nonfinite numeric mismatch: {label}")
        return
    _require(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance), f"numeric selection mismatch: {label}")


def verify_selection(
    role: Mapping[str, np.ndarray], fitted: Mapping[str, np.ndarray], ledger: Mapping[str, Any], seeds: Mapping[str, Any]
) -> dict[str, Any]:
    _require(ledger.get("candidate_count") == CANDIDATE_COUNT, "selection candidate count drift")
    _require(ledger.get("selected_head_refit_after_selection") is False, "selection permits a refit")
    expected_seed_map = {
        dgp: {
            "fixed_whitened": int(seeds["seeded_comparator_seeds"][dgp]["fixed_whitened"]),
            "histogram": int(seeds["histogram_seeds"][dgp]),
            "raw": int(seeds["seeded_comparator_seeds"][dgp]["raw"]),
        }
        for dgp in DGP_ORDER
    }
    _require(ledger.get("comparator_rng_ids") == expected_seed_map, "selection comparator RNG assignment drift")
    rows = ledger.get("candidates")
    _require(isinstance(rows, list) and len(rows) == CANDIDATE_COUNT, "selection ledger candidate rows drift")
    fixed_losses = _fit_endpoint_losses(role, np.asarray(fitted["fixed_whitening_matrix"], dtype=np.float64))
    recomputed = []
    for index, observed in enumerate(rows):
        core = _selection_candidate_core(role, fitted, index, seeds, fixed_losses)
        recomputed.append(core)
        _require(observed.get("candidate_index") == index and observed.get("candidate_id") == core["candidate_id"], "selection candidate order drift")
        _require(bool(observed.get("eligible")) == core["eligible"], f"selection eligibility mismatch: {core['candidate_id']}")
        for field in ("worst_standardized_exact_compute_contrast", "worst_dgp_stage_spearman", "worst_flops_per_row"):
            _close_float(observed.get(field), core[field], f"{core['candidate_id']}/{field}")
        observed_dgps = observed.get("dgps")
        _require(isinstance(observed_dgps, list) and len(observed_dgps) == 4, "selection per-DGP rows absent")
        for observed_dgp, expected_dgp in zip(observed_dgps, core["dgps"], strict=True):
            _require(observed_dgp.get("dgp_id") == expected_dgp["dgp_id"], "selection DGP order drift")
            _close_float(observed_dgp.get("mean_calls"), expected_dgp["mean_calls"], "mean_calls")
            _require(observed_dgp.get("call_histogram") == expected_dgp["call_histogram"], "selection call histogram mismatch")
            for stage_index, (observed_stage, expected_stage) in enumerate(zip(observed_dgp["stagewise_rank"], expected_dgp["stagewise_rank"], strict=True)):
                _require(int(observed_stage["reached_rows"]) == expected_stage["reached_rows"], "selection reached count mismatch")
                _close_float(observed_stage["spearman"], expected_stage["spearman"], f"selection Spearman stage {stage_index+1}")
            for endpoint in ENDPOINTS:
                _close_float(observed_dgp["exact_compute_contrast"][endpoint], expected_dgp["contrasts"][endpoint], f"selection contrast {endpoint}")
    eligible = [
        (index, row) for index, row in enumerate(recomputed) if row["eligible"]
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item[1]["worst_standardized_exact_compute_contrast"], -item[1]["worst_dgp_stage_spearman"],
            item[1]["worst_flops_per_row"], int(fitted["head_count"][item[0]]), float(fitted["ridges"][item[0]]),
            float(fitted["fit_quantiles"][item[0]]), item[1]["candidate_id"],
        ),
    )
    ranked_ids = [row["candidate_id"] for _, row in ranked]
    _require(ledger.get("ranked_eligible_candidate_ids") == ranked_ids, "eligible-candidate ranking drift")
    selected_index = ranked[0][0] if ranked else None
    _require(ledger.get("selected_candidate_index") == selected_index, "selected candidate index drift")
    selected_id = None if selected_index is None else str(fitted["candidate_ids"][selected_index])
    _require(ledger.get("selected_candidate_id") == selected_id, "selected candidate identity drift")
    return {"eligible_count": len(ranked), "selected_candidate_index": selected_index, "selected_candidate_id": selected_id, "all_candidates_independently_evaluated": True}


def verify_no_refit_and_freeze(
    contract: Mapping[str, Any], policy: PathPolicy, fitted: Mapping[str, np.ndarray], selection: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = contract["paths"]
    fit_path = policy.path(paths["fitted_candidates"], scope="study")
    lock_path = policy.path(paths["fit_lock"], scope="study")
    selection_path = policy.path(paths["selection_ledger"], scope="study")
    lock = read_json(lock_path)
    _require(lock.get("fitted_candidates_sha256") == sha256_file(fit_path), "fit lock fitted hash drift")
    _require(lock.get("fixed_whitening_sha256") == V5_WHITENING_SHA256, "fit lock fixed whitening drift")
    _require(lock.get("selection_input_opened_before_lock") is False and lock.get("selection_outcomes_used_for_fit") is False, "fit/selection role isolation drift")
    _require(lock.get("selected_head_refit_permitted") is False, "fit lock permits selected refit")
    fit_hashes = {dgp: sha256_file(policy.path(raw, scope="study")) for dgp, raw in contract["fit_inputs"].items()}
    _require(lock.get("fit_input_hashes") == dict(sorted(fit_hashes.items())), "fit lock input hash drift")
    if selection.get("selected_candidate_index") is None:
        return {}, {"fit_lock_created_unix_ns": int(lock["created_unix_ns"]), "selected": False}
    index = int(selection["selected_candidate_index"]); heads = int(fitted["head_count"][index])
    gate_fit_path = policy.path(paths["gate_fit"], scope="study")
    gate_fit = _load_npz(gate_fit_path)
    expected_arrays = {
        "weights": fitted["compiled_weights"][index, :, :heads], "biases": fitted["compiled_biases"][index, :, :heads],
        "thresholds": fitted["thresholds"][index], "head_names": fitted["head_names"][index, :heads],
    }
    for name, expected in expected_arrays.items():
        actual = gate_fit[name]
        _require(np.array_equal(actual, expected), f"selected gate refit/mutation detected: {name}")
    _require(str(np.asarray(gate_fit["candidate_id"]).item()) == selection["selected_candidate_id"], "gate-fit selected identity drift")
    _require(str(np.asarray(gate_fit["source_candidate_object_sha256"]).item()) == str(fitted["candidate_object_sha256"][index]), "gate-fit source object drift")
    compiled_path = policy.path(paths["compiled_gate"], scope="study")
    compiled = _load_npz(compiled_path)
    for name in ("weights", "biases", "thresholds"):
        _require(compiled[name].dtype == np.float32, f"compiled {name} is not float32")
        _require(np.array_equal(compiled[name], np.asarray(gate_fit[name], dtype=np.float32)), f"compiled gate changed science: {name}")
    _require(np.array_equal(compiled["head_names"].astype(str), gate_fit["head_names"].astype(str)), "compiled head names drift")
    compiler_manifest_path = policy.path(paths["compiled_gate_manifest"], scope="study")
    compiler = read_json(compiler_manifest_path)
    _require(compiler.get("source_gate_fit_sha256") == sha256_file(gate_fit_path), "compiler source hash drift")
    _require(compiler.get("compiled_gate_sha256") == sha256_file(compiled_path), "compiler output hash drift")
    _require(compiler.get("passed") is True, "compiler manifest failed")
    freeze_path = policy.path(paths["gate_freeze"], scope="study")
    freeze = read_json(freeze_path)
    expected_hash_links = {
        "fitted_candidates_sha256": sha256_file(fit_path), "fit_lock_sha256": sha256_file(lock_path),
        "selection_ledger_sha256": sha256_file(selection_path), "gate_fit_sha256": sha256_file(gate_fit_path),
        "compiled_gate_sha256": sha256_file(compiled_path),
        "compiled_gate_manifest_sha256": sha256_file(compiler_manifest_path),
    }
    for key, value in expected_hash_links.items():
        _require(freeze.get(key) == value, f"gate-freeze hash drift: {key}")
    _require(freeze.get("selected_head_refit_after_selection") is False and freeze.get("contact_or_privileged_gate_inputs") is False, "gate freeze scientific contract drift")
    return compiled, {
        "selected": True, "head_count": heads, "fit_lock_created_unix_ns": int(lock["created_unix_ns"]),
        "selection_created_unix_ns": int(selection["created_unix_ns"]), "gate_freeze_created_unix_ns": int(freeze["created_unix_ns"]),
        "compiler_manifest_sha256": sha256_file(compiler_manifest_path),
    }


def verify_confirmation_input_bindings(
    contract: Mapping[str, Any], policy: PathPolicy, confirmation_n: int
) -> dict[str, Any]:
    """Verify the exhaustive post-execution seal before opening any NPZ."""

    seal_path = policy.path(contract["paths"]["confirmation_input_seal"], scope="attempt")
    result = verify_seal(
        seal_path,
        policy,
        checkpoint_state="CONFIRMATION_INPUT_SEAL",
        expected_counts={
            "confirmation_generated": 4 * confirmation_n,
            "confirmation_executed": 4 * confirmation_n,
            "confirmation_opened": False,
        },
    )
    seal_files = set(_manifest_files(result["object"]))
    required = {
        contract["paths"][name]
        for name in (
            "fixed_whitening", "compiled_gate", "compiled_gate_manifest", "gate_freeze",
            "cohort_seed_ledger", "dgp_matrix", "candidate_grid", "power_rule", "power_freeze",
            "outcome_mapping", "pre_confirmation_seal",
        )
    }
    power_document = read_json(policy.path(contract["paths"]["power_freeze"], scope="study"))
    power_inputs = power_document.get("inputs")
    _require(isinstance(power_inputs, Mapping), "binding power input links absent")
    for record in power_inputs.values():
        _require(isinstance(record, Mapping) and isinstance(record.get("path"), str), "binding power input record invalid")
        input_path = policy.path(record["path"])
        _require(sha256_file(input_path) == record.get("sha256"), "binding power input hash drift")
        required.add(str(record["path"]))
    verifier_source = Path(__file__).resolve()
    required.update(
        {
            verifier_source.relative_to(policy.repository_root).as_posix(),
            verifier_source.with_name("capture_verifier.py").relative_to(policy.repository_root).as_posix(),
            verifier_source.with_name("analysis.py").relative_to(policy.repository_root).as_posix(),
            str(contract["_loaded_contract_relative"]),
        }
    )
    for raw in contract.get("confirmation_required_bindings", []):
        required.add(_canonical_relative(raw))
    for raw in contract.get("producer_sources", []):
        required.add(_canonical_relative(raw))
    for mapping_name in ("confirmation_manifests",):
        mapping = contract.get(mapping_name)
        _require(isinstance(mapping, Mapping) and tuple(mapping) == DGP_ORDER, f"{mapping_name} contract drift")
        required.update(_canonical_relative(raw) for raw in mapping.values())
    role_confirmation = contract.get("role_manifests", {}).get("confirmation", {})
    _require(tuple(role_confirmation) == DGP_ORDER, "confirmation raw manifests absent from contract")
    required.update(_canonical_relative(raw) for raw in role_confirmation.values())
    # Every regular file beneath the prospectively declared raw/execution roots
    # must be present in the seal; directory patterns cannot silently omit parts.
    roots = contract.get("confirmation_artifact_roots")
    _require(isinstance(roots, list) and bool(roots), "confirmation artifact roots absent")
    for raw_root in roots:
        root = policy.path(raw_root, scope="study", file_only=False)
        for walk_root, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = sorted(directories)
            for directory in directories:
                _require(not (Path(walk_root) / directory).is_symlink(), "symlink in confirmation artifact root")
            for filename in sorted(filenames):
                path = Path(walk_root) / filename
                _require(not path.is_symlink() and path.is_file(), "nonregular confirmation artifact")
                required.add(path.relative_to(policy.repository_root).as_posix())
    missing = sorted(required - seal_files)
    _require(not missing, f"confirmation input seal omits required/transitive files: {missing}")
    # Reject hard-link aliases across the complete sealed set, including files
    # outside the confirmation roots.
    inodes: dict[tuple[int, int], str] = {}
    for raw in seal_files:
        path = policy.path(raw)
        inode = (path.stat().st_dev, path.stat().st_ino)
        _require(inode not in inodes, f"confirmation seal hard-link alias: {raw}/{inodes.get(inode)}")
        inodes[inode] = raw
    return {
        "created_unix_ns": result["created_unix_ns"], "sealed_file_count": len(seal_files),
        "required_binding_count": len(required), "exhaustive_roots": list(roots), "passed": True,
    }


def load_confirmation_regime(
    dgp: str,
    manifest_raw: Any,
    policy: PathPolicy,
    cohort: Mapping[str, Any],
    confirmation_n: int,
    replay_regime: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    replay_result = replay_regime.get("result") if isinstance(replay_regime, Mapping) else None
    _require(
        isinstance(replay_result, Mapping)
        and replay_result.get("passed") is True
        and replay_result.get("role") == "confirmation"
        and replay_result.get("regime") == dgp
        and type(replay_result.get("episode_count")) is int
        and replay_result.get("episode_count") == confirmation_n,
        f"confirmation/{dgp} scientific replay evidence is absent",
    )
    manifest_path = policy.path(manifest_raw, scope="study")
    manifest = read_json(manifest_path)
    _require(manifest.get("regime", manifest.get("dgp_id")) == dgp, "execution manifest DGP drift")
    episodes = manifest.get("episodes")
    _require(isinstance(episodes, list) and len(episodes) == confirmation_n, "execution manifest episode count drift")
    parts: dict[str, list[np.ndarray]] = {name: [] for name in EXECUTION_KEYS}
    primary = cohort["regimes"][dgp]["roles"]["confirmation"]["primary"]
    ids: list[str] = []
    replay_episodes = replay_result.get("episodes")
    _require(
        type(replay_episodes) is list and len(replay_episodes) == confirmation_n,
        "confirmation replay episode list drift",
    )
    for slot, record in enumerate(episodes):
        _require(int(record["slot"]) == slot, "execution manifest slot drift")
        _require(str(record["episode_id"]) == str(primary[slot]["episode_id"]), "execution episode ID drift")
        path_raw = record.get("path", record.get("execution_path"))
        hash_raw = record.get("sha256", record.get("execution_sha256"))
        _require(isinstance(path_raw, str) and isinstance(hash_raw, str), "execution part link absent")
        path = policy.path(path_raw, scope="study")
        _require(sha256_file(path) == hash_raw, "execution part hash drift")
        sidecar_path = policy.path(record["sidecar_path"], scope="study")
        _require(sha256_file(sidecar_path) == record["sidecar_sha256"], "execution sidecar hash drift")
        replay_episode = replay_episodes[slot]
        _require(
            isinstance(replay_episode, Mapping)
            and replay_episode.get("slot") == slot
            and replay_episode.get("episode_id") == record["episode_id"]
            and replay_episode.get("execution_part")
            == {"path": path_raw, "sha256": hash_raw}
            and replay_episode.get("execution_sidecar")
            == {
                "path": record["sidecar_path"],
                "sha256": record["sidecar_sha256"],
            }
            and replay_episode.get("every_persisted_tensor_exact") is True
            and isinstance(replay_episode.get("array_sha256"), Mapping)
            and set(replay_episode["array_sha256"]) == EXECUTION_KEYS,
            f"confirmation/{dgp}/{slot} replay/part binding drift",
        )
        arrays = _load_npz(path, exact_keys=EXECUTION_KEYS)
        for name in EXECUTION_KEYS:
            _require(
                replay_array_sha256(arrays[name])
                == replay_episode["array_sha256"][name],
                f"confirmation/{dgp}/{slot} differs from scientific replay: {name}",
            )
        _require(len(arrays["calls"]) == ROWS_PER_EPISODE, "execution episode row count drift")
        _require(np.array_equal(np.asarray(arrays["episode_slot"], dtype=np.int64), np.full(ROWS_PER_EPISODE, slot)), "execution part slot vector drift")
        _require(np.array_equal(np.asarray(arrays["model_step"], dtype=np.int64), np.arange(3, 3 + ROWS_PER_EPISODE)), "execution part step vector drift")
        for name in EXECUTION_KEYS:
            parts[name].append(arrays[name])
        ids.append(str(record["episode_id"]))
    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}, {
        "manifest_sha256": sha256_file(manifest_path), "episode_ids_sha256": _set_digest(ids),
        "episode_count": confirmation_n,
    }


def validate_execution_dense_contract(
    arrays: Mapping[str, np.ndarray], compiled: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    calls = np.asarray(arrays["calls"], dtype=np.int64)
    rows = len(calls); heads = int(np.asarray(compiled["weights"]).shape[1])
    expected_reached = np.column_stack([calls >= stage + 1 for stage in range(3)])
    _require(np.array_equal(np.asarray(arrays["reached"], dtype=bool), expected_reached), "recorded reached mask drift")
    dense_current = np.asarray(arrays["dense_stage_current"])
    dense_update = np.asarray(arrays["dense_stage_update"])
    history = np.asarray(arrays["history"]); action = np.asarray(arrays["action_history"])
    dense_scores = np.asarray(arrays["dense_scores"]); dense_heads = np.asarray(arrays["dense_head_scores"])
    sparse_heads = np.asarray(arrays["head_scores"])
    _require(dense_current.shape == (rows, 3, LATENT_DIM) and dense_update.shape == dense_current.shape, "dense primitive shape drift")
    _require(dense_scores.shape == (rows, 3) and dense_heads.shape == (rows, 3, heads), "dense score/head shape drift")
    _require(sparse_heads.shape == (rows, 3, heads), "sparse head score shape drift")
    maximum_score = 0.0
    maximum_primitive = 0.0
    for stage in range(3):
        features = build_causal_features_numpy(history, action, dense_current[:, stage], dense_update[:, stage])
        head_values = np.asarray(features, dtype=np.float32) @ np.asarray(compiled["weights"][stage], dtype=np.float32).T
        head_values += np.asarray(compiled["biases"][stage], dtype=np.float32)[None, :]
        score = np.min(head_values, axis=1)
        _require(np.allclose(head_values, dense_heads[:, stage], rtol=2e-6, atol=2e-7), "dense head-score mismatch")
        _require(np.allclose(score, dense_scores[:, stage], rtol=2e-6, atol=2e-7), "dense minimum-score mismatch")
        reached = expected_reached[:, stage]
        _require(np.allclose(head_values[reached], sparse_heads[reached, stage], rtol=2e-6, atol=2e-7), "sparse head-score mismatch")
        _require(np.isnan(sparse_heads[~reached, stage]).all(), "unreached sparse heads are not NaN")
        _require(np.allclose(dense_current[reached, stage], arrays["stage_current"][reached, stage], rtol=2e-6, atol=2e-7), "sparse current mismatch")
        _require(np.allclose(dense_update[reached, stage], arrays["stage_update"][reached, stage], rtol=2e-6, atol=2e-7), "sparse update mismatch")
        _require(np.isnan(arrays["stage_current"][~reached, stage]).all() and np.isnan(arrays["stage_update"][~reached, stage]).all(), "unreached sparse primitives are not NaN")
        maximum_score = max(
            maximum_score,
            float(np.max(np.abs(score.astype(np.float64) - dense_scores[:, stage].astype(np.float64)), initial=0.0)),
        )
        maximum_primitive = max(
            maximum_primitive,
            float(np.max(np.abs(dense_current[reached, stage].astype(np.float64) - arrays["stage_current"][reached, stage].astype(np.float64)), initial=0.0)),
            float(np.max(np.abs(dense_update[reached, stage].astype(np.float64) - arrays["stage_update"][reached, stage].astype(np.float64)), initial=0.0)),
        )
    _require(maximum_primitive <= 4.768e-7, "sparse/dense primitive numerical ceiling exceeded")
    return {
        "dense_scores_reconstructed": True,
        "maximum_dense_score_absolute_delta": maximum_score,
        "maximum_sparse_dense_primitive_absolute_delta": maximum_primitive,
        "frozen_primitive_max_abs_ceiling": 4.768e-7,
    }


def _compare_json_science(expected: Any, observed: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        _require(isinstance(observed, Mapping), f"JSON type mismatch: {label}")
        for key, value in expected.items():
            _require(key in observed, f"JSON field absent: {label}/{key}")
            _compare_json_science(value, observed[key], f"{label}/{key}")
    elif isinstance(expected, list):
        _require(isinstance(observed, list) and len(expected) == len(observed), f"JSON list mismatch: {label}")
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            _compare_json_science(left, right, f"{label}/{index}")
    elif isinstance(expected, float):
        _close_float(expected, observed, label, tolerance=2e-11)
    else:
        _require(expected == observed, f"JSON value mismatch: {label}")


def independent_bootstrap_summary(
    metrics: Mapping[str, Mapping[str, np.ndarray]], bootstrap: Mapping[str, Mapping[str, np.ndarray]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    individual: dict[str, Any] = {}; simultaneous: dict[str, Any] = {}
    for dgp in DGP_ORDER:
        individual[dgp] = {}; simultaneous[dgp] = {}
        for name in ALL_CONTRASTS:
            values = np.asarray(metrics[dgp][name], dtype=np.float64); replicates = bootstrap[dgp][name]
            individual[dgp][name] = {
                "estimate": float(values.mean()), "lower": linear_quantile(replicates, .025),
                "upper": linear_quantile(replicates, .975), "confidence": .95, "sidedness": "two_sided",
                "method": "episode_bootstrap_percentile_numpy_linear_quantile", "terminal": name in PRIMARY_CONTRASTS,
            }
        for endpoint, name in zip(ENDPOINTS, PRIMARY_CONTRASTS, strict=True):
            lower = linear_quantile(bootstrap[dgp][name], PER_CLAIM_ALPHA)
            simultaneous[dgp][endpoint] = {
                "contrast": name, "estimate": float(np.mean(metrics[dgp][name])), "lower": lower,
                "supported": bool(lower > 0), "familywise_alpha": FAMILYWISE_ALPHA, "family_size": FAMILY_SIZE,
                "per_claim_one_sided_alpha": PER_CLAIM_ALPHA,
                "method": "bonferroni_one_sided_episode_bootstrap_percentile_numpy_linear_quantile",
            }
    heterogeneity: dict[str, Any] = {"pairwise": {}, "range": {}}
    for endpoint, name in zip(ENDPOINTS, PRIMARY_CONTRASTS, strict=True):
        estimates = {dgp: float(np.mean(metrics[dgp][name])) for dgp in DGP_ORDER}
        heterogeneity["range"][endpoint] = {
            "minimum_regime": min(estimates, key=estimates.get), "maximum_regime": max(estimates, key=estimates.get),
            "max_minus_min": max(estimates.values()) - min(estimates.values()), "descriptive_not_terminal": True,
        }
        for left_index, left in enumerate(DGP_ORDER):
            for right in DGP_ORDER[left_index + 1:]:
                difference = bootstrap[left][name] - bootstrap[right][name]
                heterogeneity["pairwise"].setdefault(f"{left}_minus_{right}", {})[endpoint] = {
                    "estimate": estimates[left] - estimates[right], "lower": linear_quantile(difference, .025),
                    "upper": linear_quantile(difference, .975), "descriptive_not_terminal": True,
                }
    return individual, simultaneous, heterogeneity


def independent_rank_calibration(
    scores: np.ndarray, calls: np.ndarray, losses: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranks: list[dict[str, Any]] = []; calibration: list[dict[str, Any]] = []
    for stage in range(3):
        reached = np.flatnonzero(np.isfinite(scores[:, stage]))
        _require(np.array_equal(np.isfinite(scores[:, stage]), calls >= stage + 1), "score reached mask drift")
        endpoint_gains = {endpoint: losses[endpoint][:, stage] - losses[endpoint][:, stage + 1] for endpoint in ENDPOINTS}
        endpoints: dict[str, Any] = {}
        for endpoint in ENDPOINTS:
            rho = spearman(scores[reached, stage], endpoint_gains[endpoint][reached])
            endpoints[endpoint] = {
                "spearman_score_next_stage_gain_rho": float(rho) if np.isfinite(rho) else None,
                "rank_sign": "positive" if np.isfinite(rho) and rho > 0 else "negative" if np.isfinite(rho) and rho < 0 else "zero_or_undefined",
                "positive_sign": bool(np.isfinite(rho) and rho > 0),
                "mean_next_stage_gain": float(endpoint_gains[endpoint][reached].mean()) if len(reached) else None,
            }
        ranks.append({
            "stage": stage + 1, "reached_rows": len(reached),
            "continued_rows": int(np.sum(calls[reached] > stage + 1)),
            "continuation_rate_among_reached": float(np.mean(calls[reached] > stage + 1)) if len(reached) else None,
            "endpoints": endpoints,
        })
        order = reached[np.argsort(scores[reached, stage], kind="mergesort")]
        bins = []
        for bin_index, indices in enumerate(np.array_split(order, 10)):
            if not len(indices):
                continue
            record: dict[str, Any] = {
                "bin": bin_index + 1, "rows": len(indices), "score_mean": float(scores[indices, stage].mean()),
                "continuation_rate": float(np.mean(calls[indices] > stage + 1)),
            }
            for endpoint in ENDPOINTS:
                gain = endpoint_gains[endpoint][indices]
                record[f"{endpoint}_next_stage_gain_mean"] = float(gain.mean())
                record[f"{endpoint}_score_minus_gain_mean"] = float((scores[indices, stage] - gain).mean())
            bins.append(record)
        calibration.append({"stage": stage + 1, "reached_rows": len(reached), "equal_count_score_bins": bins, "descriptive_only": True})
    return ranks, calibration


def verify_power_result(power: Mapping[str, Any], *, mode: str) -> int:
    _require(power.get("family_size") == FAMILY_SIZE and float(power.get("per_claim_alpha")) == PER_CLAIM_ALPHA, "binding power family drift")
    _require(power.get("no_sequential_expansion") is True and power.get("fresh_confirmation_outcomes_opened") == 0, "binding power timing drift")
    feasible = bool(power.get("feasible"))
    if mode == "confirmation":
        _require(feasible and power.get("decision") == "confirmation_size_fixed", "confirmation lacks feasible fixed power")
        _require(power.get("confirmation_generation_authorized_by_power") is True, "confirmation was not power-authorized")
        n = int(power["selected_confirmation_episodes_per_regime"])
        _require(n in range(500, 4501, 500), "confirmation N outside frozen grid")
        _require(power.get("confirmation_episode_count_per_regime") == {dgp: n for dgp in DGP_ORDER}, "power per-regime N drift")
        _require(power.get("selected_confirmation_slots_per_regime") == [0, n - 1], "confirmation prefix drift")
        return n
    _require(mode == "power_infeasible", "power verifier invoked in illegal mode")
    _require(not feasible and power.get("passed") is False, "power-infeasible result became feasible")
    _require(power.get("decision") == "power_infeasible_no_confirmation", "power-infeasible decision drift")
    _require(power.get("confirmation_generation_authorized_by_power") is False, "infeasible power authorized generation")
    _require(power.get("selected_confirmation_episodes_per_regime") is None, "infeasible power assigned N")
    _require(power.get("terminal_label_if_infeasible") == TERMINAL_FAILED, "infeasible power terminal mapping drift")
    return 0


def _regularized_gamma_p(shape: float, value: float) -> float:
    _require(shape > 0 and value >= 0, "gamma CDF domain error")
    if value == 0:
        return 0.0
    logarithm = -value + shape * math.log(value) - math.lgamma(shape)
    epsilon = 2e-15
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        denominator = shape
        for _ in range(10000):
            denominator += 1.0
            term *= value / denominator
            total += term
            if abs(term) <= abs(total) * epsilon:
                return float(total * math.exp(logarithm))
        raise VerificationError("gamma-series convergence failure")
    tiny = 1e-300
    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(b, tiny)
    fraction = d
    for index in range(1, 10000):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            return float(1.0 - math.exp(logarithm) * fraction)
    raise VerificationError("gamma-fraction convergence failure")


def _chi_square_quantile(probability: float, degrees: int) -> float:
    _require(0 < probability < 1 and degrees > 0, "chi-square quantile domain error")
    lower = 0.0
    upper = max(float(degrees), 1.0)
    while _regularized_gamma_p(degrees / 2.0, upper / 2.0) < probability:
        upper *= 2.0
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if _regularized_gamma_p(degrees / 2.0, middle / 2.0) < probability:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _upper_95_sd(sd: float, episodes: int) -> float:
    if sd == 0:
        return 0.0
    degrees = episodes - 1
    return math.sqrt(degrees * sd * sd / _chi_square_quantile(.05, degrees))


def verify_power_input_provenance(
    power: Mapping[str, Any], contract: Mapping[str, Any], policy: PathPolicy
) -> dict[str, Any]:
    """Authenticate local-only moments and selected-gate identity links."""

    inputs = power.get("inputs")
    expected_keys = {
        "fit_summary", "selection_summary", "power_rule", "fit_lock",
        "selection_ledger", "gate_freeze",
    }
    _require(isinstance(inputs, Mapping) and set(inputs) == expected_keys, "binding power input schema drift")
    expected_paths = {
        "fit_summary": f"{contract['attempt_root']}/metrics/fit_power_summary.json",
        "selection_summary": f"{contract['attempt_root']}/metrics/selection_power_summary.json",
        "power_rule": contract["paths"]["power_rule"],
        "fit_lock": contract["paths"]["fit_lock"],
        "selection_ledger": contract["paths"]["selection_ledger"],
        "gate_freeze": contract["paths"]["gate_freeze"],
    }
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name in sorted(expected_keys):
        record = inputs[name]
        _require(isinstance(record, Mapping), f"binding power input record invalid: {name}")
        _require(record.get("path") == expected_paths[name], f"binding power input path drift: {name}")
        path = policy.path(record["path"])
        observed_hash = sha256_file(path)
        _require(record.get("sha256") == observed_hash, f"binding power input hash drift: {name}")
        documents[name] = read_json(path)
        hashes[name] = observed_hash

    rule_record = inputs["power_rule"]
    _require(
        hashes["power_rule"] == rule_record.get("expected_sha256") == POWER_RULE_SHA256,
        "frozen local power-rule hash drift",
    )
    moments = documents["power_rule"].get("consumed_v5_endpoint_moments")
    _require(isinstance(moments, Mapping) and set(moments) == set(PRIMARY_CONTRASTS), "local power-rule moment schema drift")
    _require(
        canonical_object_sha256(moments)
        == rule_record.get("endpoint_moments_object_sha256")
        == POWER_RULE_MOMENTS_SHA256,
        "local power-rule moment object hash drift",
    )
    for endpoint in PRIMARY_CONTRASTS:
        item = moments[endpoint]
        _require(isinstance(item, Mapping), f"local power-rule moment absent: {endpoint}")
        _require(
            int(item.get("episode_count", 0)) > 1
            and math.isfinite(float(item.get("mean", float("nan"))))
            and float(item.get("mean", 0.0)) > 0
            and math.isfinite(float(item.get("sd_ddof1", float("nan"))))
            and float(item.get("sd_ddof1", -1.0)) >= 0,
            f"local power-rule moment invalid: {endpoint}",
        )

    identity = power.get("selected_gate_identity")
    identity_keys = {
        "selected_candidate_id", "selected_candidate_object_sha256",
        "fit_lock_sha256", "selection_ledger_sha256", "gate_freeze_sha256",
    }
    _require(isinstance(identity, Mapping) and set(identity) == identity_keys, "binding power selected-gate identity drift")
    candidate_id = identity["selected_candidate_id"]
    _require(isinstance(candidate_id, str) and bool(candidate_id), "binding power candidate identity absent")
    for field in identity_keys - {"selected_candidate_id"}:
        value = identity[field]
        _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"binding power identity hash invalid: {field}")
    _require(identity["fit_lock_sha256"] == hashes["fit_lock"], "power fit-lock identity hash drift")
    _require(identity["selection_ledger_sha256"] == hashes["selection_ledger"], "power selection-ledger identity hash drift")
    _require(identity["gate_freeze_sha256"] == hashes["gate_freeze"], "power gate-freeze identity hash drift")

    summaries = {"fit": documents["fit_summary"], "selection": documents["selection_summary"]}
    for role, summary in summaries.items():
        _require(summary.get("schema_version") == 1 and summary.get("attempt") == ATTEMPT, f"{role} power-summary identity drift")
        _require(summary.get("role") == role, f"{role} power-summary role drift")
        _require(summary.get("co_primary_only") is True, f"{role} power summary is not co-primary-only")
        _require(summary.get("auxiliary_robust_whitening_excluded") is True, f"{role} power summary includes auxiliary endpoint")
        for field in identity_keys:
            _require(summary.get(field) == identity[field], f"{role} power-summary selected identity drift: {field}")

    fit_lock = documents["fit_lock"]
    selection = documents["selection_ledger"]
    freeze = documents["gate_freeze"]
    selected_row = next(
        (
            row for row in selection.get("candidates", [])
            if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
        ),
        None,
    )
    crosslinks = {
        "fit_lock_status": fit_lock.get("status") == "all_24_candidates_fit_compiled_and_locked_before_selection_open",
        "fit_lock_no_selection": fit_lock.get("selection_input_opened_before_lock") is False,
        "selection_status": selection.get("status") == "selection_complete_no_selected_head_refit",
        "selection_candidate": selection.get("selected_candidate_id") == candidate_id,
        "selection_no_refit": selection.get("selected_head_refit_after_selection") is False,
        "selection_fit_lock": selection.get("fit_lock_sha256") == identity["fit_lock_sha256"],
        "selection_candidate_object": isinstance(selected_row, Mapping) and selected_row.get("candidate_object_sha256") == identity["selected_candidate_object_sha256"],
        "selection_candidate_eligible": isinstance(selected_row, Mapping) and selected_row.get("eligible") is True,
        "freeze_status": freeze.get("status") == "selected_gate_frozen_before_smoke_or_confirmation",
        "freeze_candidate": freeze.get("selected_candidate_id") == candidate_id,
        "freeze_candidate_object": freeze.get("selected_candidate_object_sha256") == identity["selected_candidate_object_sha256"],
        "freeze_fit_lock": freeze.get("fit_lock_sha256") == identity["fit_lock_sha256"],
        "freeze_selection": freeze.get("selection_ledger_sha256") == identity["selection_ledger_sha256"],
        "freeze_no_refit": freeze.get("selected_head_refit_after_selection") is False,
        "freeze_zero_confirmation": freeze.get("confirmation_episodes_at_freeze") == 0,
    }
    _require(all(crosslinks.values()), f"binding power identity cross-link failure: {[name for name, passed in crosslinks.items() if not passed]}")
    return {
        "moments": moments,
        "fit_summary": summaries["fit"],
        "selection_summary": summaries["selection"],
        "identity": dict(identity),
        "input_hashes": hashes,
        "identity_crosslinks": crosslinks,
        "local_power_rule_only": True,
    }


def verify_binding_power_recomputation(
    power: Mapping[str, Any], fitted: Mapping[str, np.ndarray], selection: Mapping[str, Any],
    contract: Mapping[str, Any], policy: PathPolicy,
) -> dict[str, Any]:
    provenance = verify_power_input_provenance(power, contract, policy)
    index = selection.get("selected_candidate_index")
    _require(index is not None, "binding power exists without selected candidate")
    index = int(index)
    selected_id = str(fitted["candidate_ids"][index])
    selected_row = next(row for row in selection["candidates"] if row["candidate_id"] == selected_id)
    selected_dgps = {row["dgp_id"]: row for row in selected_row["dgps"]}
    identity = provenance["identity"]
    _require(identity["selected_candidate_id"] == selected_id, "binding power selected candidate drift")
    _require(identity["selected_candidate_object_sha256"] == str(fitted["candidate_object_sha256"][index]), "binding power selected candidate object drift")
    consumed = provenance["moments"]
    fit_summary = provenance["fit_summary"]
    selection_summary = provenance["selection_summary"]
    endpoint_keys = ("raw_vs_analytic", "fixed_whitened_vs_analytic")
    reconstructed_claims: dict[str, Any] = {}
    for dgp_index, dgp in enumerate(DGP_ORDER):
        reconstructed_claims[dgp] = {}
        for endpoint_index, endpoint_key in enumerate(endpoint_keys):
            fit_item = {
                "episode_count": int(fitted["fit_contrast_episode_count"][index, dgp_index, endpoint_index]),
                "mean": float(fitted["fit_contrast_episode_mean"][index, dgp_index, endpoint_index]),
                "sd_ddof1": float(fitted["fit_contrast_episode_sd"][index, dgp_index, endpoint_index]),
            }
            selection_item = selected_dgps[dgp]["exact_compute_episode_summary"][endpoint_key]
            selection_item = {
                "episode_count": int(selection_item["episode_count"]), "mean": float(selection_item["mean"]),
                "sd_ddof1": float(selection_item["sd_ddof1"]),
            }
            record = power["claim_design_inputs"][dgp][endpoint_key]
            _compare_json_science(fit_item, fit_summary["claims"][dgp][endpoint_key], f"fit power summary/{dgp}/{endpoint_key}")
            _compare_json_science(selection_item, selection_summary["claims"][dgp][endpoint_key], f"selection power summary/{dgp}/{endpoint_key}")
            _compare_json_science(fit_item, record["fit"], f"power/{dgp}/{endpoint_key}/fit")
            _compare_json_science(selection_item, record["selection"], f"power/{dgp}/{endpoint_key}/selection")
            positive = fit_item["mean"] > 0 and selection_item["mean"] > 0
            delta = .35 * min(fit_item["mean"], selection_item["mean"]) if positive else None
            fit_upper = _upper_95_sd(fit_item["sd_ddof1"], fit_item["episode_count"])
            selection_upper = _upper_95_sd(selection_item["sd_ddof1"], selection_item["episode_count"])
            consumed_sd = float(consumed[endpoint_key]["sd_ddof1"])
            sigma = max(fit_upper, selection_upper, consumed_sd)
            _close_float(record["fit_sd_upper_95"], fit_upper, "fit SD upper", tolerance=2e-10)
            _close_float(record["selection_sd_upper_95"], selection_upper, "selection SD upper", tolerance=2e-10)
            _close_float(record["consumed_v5_endpoint_sd"], consumed_sd, "consumed SD")
            _close_float(record["sigma"], sigma, "design sigma", tolerance=2e-10)
            _require(record["both_means_strictly_positive"] is positive, "power positive-mean rule drift")
            if delta is None:
                _require(record["delta"] is None, "power assigned effect to nonpositive claim")
            else:
                _close_float(record["delta"], delta, "retained effect")
            reconstructed_claims[dgp][endpoint_key] = {"delta": delta, "sigma": sigma}
    critical = NormalDist().inv_cdf(1.0 - PER_CLAIM_ALPHA)
    grid = []
    for episodes in range(500, 4501, 500):
        marginal: dict[str, Any] = {}; failure = 0.0
        for dgp in DGP_ORDER:
            marginal[dgp] = {}
            for endpoint_key in endpoint_keys:
                item = reconstructed_claims[dgp][endpoint_key]
                power_value = 0.0 if item["delta"] is None else NormalDist().cdf(float(item["delta"]) * math.sqrt(episodes) / float(item["sigma"]) - critical)
                marginal[dgp][endpoint_key] = power_value
                failure += 1.0 - power_value
        family = max(0.0, 1.0 - failure)
        grid.append({"episodes_per_regime": episodes, "marginal_power": marginal, "union_bound_family_power_lower": family, "eligible": family >= .95})
    _compare_json_science(grid, power["power_grid"], "binding power grid")
    first = next((row for row in grid if row["eligible"]), None)
    expected_n = None if first is None or any(item[endpoint]["delta"] is None for item in reconstructed_claims.values() for endpoint in endpoint_keys) else int(first["episodes_per_regime"])
    _require(power["selected_confirmation_episodes_per_regime"] == expected_n, "binding power selected N drift")
    return {
        "claim_count": 8,
        "grid_count": len(grid),
        "selected_confirmation_episodes_per_regime": expected_n,
        "chi_square_quantiles_independently_recomputed": True,
        "local_hash_pinned_power_rule_only": True,
        "selected_gate_identity_crosslinks": provenance["identity_crosslinks"],
    }


def verify_confirmation_analysis(
    contract: Mapping[str, Any], policy: PathPolicy, cohort: Mapping[str, Any], compiled: Mapping[str, np.ndarray],
    seeds: Mapping[str, Any], confirmation_n: int, replay_role: Mapping[str, Any],
) -> dict[str, Any]:
    whitening_path = policy.path(contract["paths"]["fixed_whitening"])
    _require(sha256_file(whitening_path) == V5_WHITENING_SHA256, "unchanged V5 whitening file hash drift")
    whitening_archive = _load_npz(whitening_path)
    _require("whitening_matrix" in whitening_archive, "fixed whitening matrix key absent")
    whitening = np.asarray(whitening_archive["whitening_matrix"], dtype=np.float64)
    _require(whitening.shape == (LATENT_DIM, LATENT_DIM) and np.isfinite(whitening).all(), "fixed whitening matrix invalid")
    analysis_path = policy.path(contract["paths"]["analysis_result"], scope="attempt")
    analysis = read_json(analysis_path)
    _require(analysis.get("attempt") == ACTIVE_ATTEMPT and analysis.get("checkpoint_state") == "SEALED_ANALYSIS", "analysis identity drift")
    metrics_by_dgp: dict[str, dict[str, np.ndarray]] = {}
    regime_checks: dict[str, Any] = {}
    for dgp in DGP_ORDER:
        arrays, input_record = load_confirmation_regime(
            dgp,
            contract["confirmation_manifests"][dgp],
            policy,
            cohort,
            confirmation_n,
            replay_role["regimes"][dgp],
        )
        dense_check = validate_execution_dense_contract(arrays, compiled)
        summary, metrics = analyze_confirmation_regime(
            arrays, whitening, compiled,
            seeded_seeds=seeds["seeded_comparator_seeds"][dgp], histogram_seed=int(seeds["histogram_seeds"][dgp]),
        )
        _require(summary["episode_count"] == confirmation_n, "independent confirmation N drift")
        producer_regime = analysis["regimes"][dgp]
        metric_record = producer_regime["episode_metrics"]
        metric_path = policy.path(metric_record["path"], scope="study")
        _require(sha256_file(metric_path) == metric_record["sha256"], "producer episode-metric file hash drift")
        producer_metrics = _load_npz(metric_path)
        _require(set(producer_metrics) == set(metrics), "episode metric array schema drift")
        for name, expected in metrics.items():
            _require(np.array_equal(producer_metrics[name], expected), f"episode metrics differ elementwise: {dgp}/{name}")
            _require(metric_record["arrays"][name] == array_sha256(expected), f"episode metric array hash drift: {dgp}/{name}")
        # Independently reproduce the terminal diagnostics that can catch a
        # sign inversion even when aggregate contrasts look favorable.
        fixed, _ = endpoint_losses(arrays["target"], arrays["exits"], arrays["selected"], whitening)
        ranks, calibration = independent_rank_calibration(np.asarray(arrays["scores"], dtype=np.float64), np.asarray(arrays["calls"], dtype=np.int64), fixed)
        _compare_json_science(ranks, producer_regime["stagewise_gate_score_next_stage_gain_rank"], f"{dgp}/rank")
        _compare_json_science(calibration, producer_regime["routing_calibration"], f"{dgp}/calibration")
        for key, value in summary["compute"].items():
            if key in producer_regime["compute"]:
                _compare_json_science(value, producer_regime["compute"][key], f"{dgp}/compute/{key}")
        _require(analysis["inputs"][dgp]["manifest_sha256"] == input_record["manifest_sha256"], "analysis execution-manifest hash drift")
        metrics_by_dgp[dgp] = metrics
        regime_checks[dgp] = summary | dense_check | {"input": input_record}
    bootstrap = bootstrap_all(metrics_by_dgp, int(seeds["joint_bootstrap_seed"]))
    flat_expected = {f"{dgp}__{name}": bootstrap[dgp][name] for dgp in DGP_ORDER for name in ALL_CONTRASTS}
    bootstrap_path = policy.path(contract["paths"]["bootstrap_replicates"], scope="study")
    producer_bootstrap = _load_npz(bootstrap_path)
    _require(set(producer_bootstrap) == set(flat_expected), "bootstrap array schema drift")
    manifest = analysis["bootstrap"]["artifact"]
    _require(manifest["sha256"] == sha256_file(bootstrap_path), "bootstrap NPZ file hash drift")
    _require(manifest["seed"] == int(seeds["joint_bootstrap_seed"]) and manifest["replicates_per_array"] == BOOTSTRAP_REPLICATES, "bootstrap seed/count drift")
    for name, expected in flat_expected.items():
        actual = producer_bootstrap[name]
        _require(np.array_equal(actual, expected), f"bootstrap differs elementwise: {name}")
        _require(manifest["arrays"][name]["sha256"] == array_sha256(expected), f"bootstrap array hash drift: {name}")
    individual, simultaneous, heterogeneity = independent_bootstrap_summary(metrics_by_dgp, bootstrap)
    summary_path = policy.path(contract["paths"]["bootstrap_summary"], scope="study")
    producer_summary = read_json(summary_path)
    _compare_json_science(individual, producer_summary["individual"], "bootstrap/individual")
    _compare_json_science(simultaneous, producer_summary["simultaneous_co_primary"], "bootstrap/simultaneous")
    _compare_json_science(heterogeneity, producer_summary["heterogeneity"], "bootstrap/heterogeneity")
    _compare_json_science(simultaneous, analysis["simultaneous_co_primary"], "analysis/simultaneous")
    bounds = {dgp: {endpoint: float(simultaneous[dgp][endpoint]["lower"]) for endpoint in ENDPOINTS} for dgp in DGP_ORDER}
    label, supported = terminal_mapping(bounds, True)
    _require(analysis.get("process_valid") is True and analysis.get("passed") is True, "producer analysis is not process-valid")
    _require(analysis.get("proposed_terminal_label") == label and analysis.get("supported_co_primary_claim_count") == supported, "terminal mapping drift")
    _require(analysis["fixed_whitening"]["sha256"] == V5_WHITENING_SHA256 and analysis["fixed_whitening"]["unchanged_v5_endpoint"] is True, "analysis changed fixed endpoint")
    return {
        "terminal_label": label, "supported_co_primary_claim_count": supported,
        "confirmation_episode_count_per_regime": confirmation_n, "regimes": regime_checks,
        "bootstrap_arrays_reproduced_elementwise": len(flat_expected), "bootstrap_replicates_per_array": BOOTSTRAP_REPLICATES,
        "analysis_result_sha256": sha256_file(analysis_path), "bootstrap_replicates_sha256": sha256_file(bootstrap_path),
        "bootstrap_summary_sha256": sha256_file(summary_path), "fixed_whitening_sha256": sha256_file(whitening_path),
    }


def _require_no_role_artifacts(policy: PathPolicy, contract: Mapping[str, Any]) -> None:
    defaults = [
        f"{contract['attempt_root']}/data/smoke",
        f"{contract['attempt_root']}/data/confirmation",
        f"{contract['attempt_root']}/metrics/bootstrap_replicates.npz",
        f"{contract['attempt_root']}/analysis_result.json",
    ]
    for raw in contract.get("zero_confirmation_paths", defaults):
        path = policy.path(raw, scope="attempt", must_exist=False, file_only=False)
        if path.is_dir():
            _require(not any(path.iterdir()), f"early-terminal role directory is not empty: {raw}")
        else:
            _require(not path.exists(), f"early-terminal confirmation artifact exists: {raw}")


def verify_chronology(
    mode: str, seals: Mapping[str, Any], role_manifests: Mapping[str, Any], freeze: Mapping[str, Any],
    power: Mapping[str, Any] | None, analysis: Mapping[str, Any] | None,
) -> dict[str, Any]:
    points = [int(seals["pre_data"]["created_unix_ns"])]
    labels = ["pre_data"]
    fit_max = max(role_manifests["maximum_created_unix_ns"]["fit"].values())
    points.append(fit_max); labels.append("fit_data")
    points.append(int(seals["pre_selection"]["created_unix_ns"])); labels.append("pre_selection")
    selection_max = max(role_manifests["maximum_created_unix_ns"]["selection"].values())
    points.append(selection_max); labels.append("selection_data")
    if freeze.get("selected"):
        _require(int(freeze["fit_lock_created_unix_ns"]) <= int(seals["pre_selection"]["created_unix_ns"]), "fit lock did not precede pre-selection seal")
        _require(int(freeze["selection_created_unix_ns"]) >= int(seals["pre_selection"]["created_unix_ns"]), "selection predates selection seal")
        points.append(int(freeze["gate_freeze_created_unix_ns"])); labels.append("gate_freeze")
    if power is not None:
        points.append(int(power["created_unix_ns"])); labels.append("power_freeze")
    if mode == "confirmation":
        points.append(int(seals["pre_confirmation"]["created_unix_ns"])); labels.append("pre_confirmation")
        points.append(max(role_manifests["maximum_created_unix_ns"]["smoke"].values())); labels.append("smoke")
        points.append(max(role_manifests["maximum_created_unix_ns"]["confirmation"].values())); labels.append("confirmation")
        points.append(int(seals["confirmation_input"]["created_unix_ns"])); labels.append("confirmation_input_seal")
        points.append(int(analysis["created_unix_ns"])); labels.append("analysis")
    _require(all(right >= left for left, right in zip(points, points[1:])), f"scientific chronology drift: {list(zip(labels, points))}")
    return {"ordered_events": list(zip(labels, points)), "strict_role_isolation_chronology": True}


def verify_analysis_execution_invalid_branch(
    contract: Mapping[str, Any], policy: PathPolicy
) -> dict[str, Any]:
    """Authenticate a staged post-open analysis failure without loading NPZs."""

    _require(contract.get("mode") == "confirmation", "analysis-failure branch requires confirmation mode")
    failure_relative = contract["_analysis_execution_invalid_relative"]
    failure_path = policy.path(failure_relative, scope="attempt")
    analysis_path = policy.path(
        contract["paths"]["analysis_result"], scope="attempt", must_exist=False
    )
    if analysis_path.exists() or analysis_path.is_symlink():
        analysis_path = policy.path(
            contract["paths"]["analysis_result"], scope="attempt"
        )

    # Close the complete byte/path set before trusting the failure claim.  No
    # function reachable from this branch calls ``_load_npz``.
    terminal_manifest = verify_terminal_manifest(contract, policy)
    normative = verify_normative_contracts(contract, policy)
    ledger_state = verify_ledger_and_state(contract, policy)
    state = ledger_state["state"]
    lineage = verify_version_forward_lineage(contract, policy, state, ledger_state["events"])
    identifier = verify_identifier_ledger(contract, policy)
    paths = contract["paths"]
    pre_data = verify_seal(
        policy.path(paths["pre_data_seal"], scope="study"), policy,
        checkpoint_state="PRE_OUTCOME_SEAL",
        expected_counts={"fit": 0, "selection": 0, "smoke": 0, "confirmation_generated": 0, "confirmation_executed": 0, "confirmation_opened": False},
        expected_attempt=SCIENCE_ATTEMPT,
    )
    pre_selection = verify_seal(
        policy.path(paths["pre_selection_seal"], scope="study"), policy,
        checkpoint_state="PRE_SELECTION_SEAL",
        expected_counts={"fit": 1200, "selection": 0, "smoke": 0, "confirmation_generated": 0, "confirmation_executed": 0, "confirmation_opened": False},
        expected_pre_data_seal_sha256=lineage["seal_sha256"],
    )
    power = read_json(policy.path(paths["power_freeze"], scope="study"))
    confirmation_n = verify_power_result(power, mode="confirmation")
    power_provenance = verify_power_input_provenance(power, contract, policy)
    pre_confirmation = verify_seal(
        policy.path(paths["pre_confirmation_seal"], scope="study"), policy,
        checkpoint_state="PRE_CONFIRMATION_PACKAGE_SEAL",
        expected_counts={"fit": 1200, "selection": 2000, "smoke": 0, "confirmation_generated": 0, "confirmation_executed": 0, "confirmation_opened": False},
    )
    confirmation_input = verify_confirmation_input_bindings(contract, policy, confirmation_n)
    replay_check = verify_scientific_replay_qualification(
        contract,
        policy,
        state,
        confirmation_n=confirmation_n,
    )

    failure = read_json(failure_path)
    total_confirmation = 4 * confirmation_n
    result_present = analysis_path.is_file()
    header_checks = {
        "schema": failure.get("schema_version") == 1,
        "attempt": failure.get("attempt") == ATTEMPT,
        "checkpoint": failure.get("checkpoint_state") == "SEALED_ANALYSIS",
        "status": failure.get("status") == "irreversible_post_open_analysis_execution_invalid",
        "producer_failed": failure.get("passed") is False,
        "integrity_failure": failure.get("integrity_failure") is True,
        "integrity_not_passed": failure.get("integrity_passed") is False,
        "process_invalid": failure.get("process_valid") is False,
        "execution_invalid": failure.get("execution_invalid") is True,
        "terminal_mapping": failure.get("proposed_terminal_label") == TERMINAL_INVALID,
        "opened": failure.get("confirmation_opened") is True and failure.get("confirmation_outcomes_opened_for_analysis") is True,
        "generated_fixed": failure.get("confirmation_outcome_episodes_generated") == total_confirmation,
        "executed_fixed": failure.get("confirmation_outcome_episodes_executed") == total_confirmation,
        "result_presence_exact": failure.get("analysis_result_present") is result_present,
        "science_unchanged": failure.get("scientific_objects_changed") is False and failure.get("scientific_objects_changed_after_open") is False,
        "no_outcomes_in_failure_capture": failure.get("failure_capture_opened_no_arrays") is True and failure.get("outcome_values_recorded") is False,
        "forbidden_inputs_closed": failure.get("contact_motion_phase_reward_success_opened") is False,
        "no_retry": failure.get("retry_permitted") is False,
        "independent_required": failure.get("independent_verification_required") is True,
    }
    _require(all(header_checks.values()), f"analysis execution-invalid header drift: {[name for name, passed in header_checks.items() if not passed]}")
    _require(int(failure.get("created_unix_ns", 0)) > 0, "analysis failure lacks chronology timestamp")
    _require(isinstance(failure.get("error_type"), str) and bool(failure["error_type"]), "analysis failure error type absent")
    _require(isinstance(failure.get("error"), str) and bool(failure["error"]), "analysis failure message absent")
    result_record = failure.get("analysis_result")
    if result_present:
        _require(isinstance(result_record, Mapping), "present failed analysis result lacks an explicit binding")
        _require(result_record.get("path") == contract["paths"]["analysis_result"], "failed analysis-result path drift")
        _require(result_record.get("sha256") == sha256_file(analysis_path), "failed analysis-result hash drift")
        _require(int(result_record.get("bytes", -1)) == analysis_path.stat().st_size, "failed analysis-result byte-size drift")
    else:
        _require(result_record is None, "absent failed analysis result has a spurious binding")
    _require(
        failure.get("execution_root") == f"{contract['attempt_root']}/data/confirmation",
        "analysis failure execution-root drift",
    )
    partial_candidates = [
        f"{contract['attempt_root']}/metrics/{dgp}_episode_metrics.npz"
        for dgp in DGP_ORDER
    ] + [
        f"{contract['attempt_root']}/metrics/bootstrap_replicates.npz",
        f"{contract['attempt_root']}/metrics/compute_ledger.json",
        f"{contract['attempt_root']}/metrics/bootstrap_summary.json",
    ]
    expected_partial: dict[str, dict[str, Any]] = {}
    for raw in partial_candidates:
        candidate = policy.path(raw, scope="attempt", must_exist=False)
        if candidate.exists() or candidate.is_symlink():
            candidate = policy.path(raw, scope="attempt")
            expected_partial[raw] = {
                "sha256": sha256_file(candidate),
                "bytes": candidate.stat().st_size,
            }
    _require(
        failure.get("partial_derived_outputs") == expected_partial,
        "analysis failure partial-output closure drift",
    )
    failure_detail = failure.get("failure")
    _require(isinstance(failure_detail, Mapping), "analysis failure detail absent")
    _require(
        failure_detail.get("exception_type") == failure["error_type"]
        and failure_detail.get("exception_message") == failure["error"]
        and failure_detail.get("traceback_includes_locals") is False,
        "analysis failure detail drift",
    )

    expected_inputs = {
        "confirmation_input_seal": paths["confirmation_input_seal"],
        "pre_confirmation_package_seal": paths["pre_confirmation_seal"],
        "analysis_source": f"{contract['attempt_root']}/analysis.py",
        "fixed_whitening": paths["fixed_whitening"],
        "compiled_gate_manifest": paths["compiled_gate_manifest"],
        "cohort_seed_ledger": paths["cohort_seed_ledger"],
        "compiled_gate": paths["compiled_gate"],
    }
    failure_inputs = failure.get("inputs")
    _require(isinstance(failure_inputs, Mapping) and set(failure_inputs) == set(expected_inputs), "analysis failure input schema drift")
    for name, expected_path in expected_inputs.items():
        record = failure_inputs[name]
        _require(isinstance(record, Mapping) and record.get("path") == expected_path, f"analysis failure input path drift: {name}")
        source = policy.path(record["path"])
        _require(record.get("sha256") == sha256_file(source), f"analysis failure input hash drift: {name}")
        _require(int(record.get("bytes", -1)) == source.stat().st_size, f"analysis failure input byte-size drift: {name}")
    _require(
        failure.get("confirmation_input_seal_sha256") == failure_inputs["confirmation_input_seal"]["sha256"],
        "analysis failure input-seal shortcut hash drift",
    )

    staged = state["postconfirmation_integrity_failure"]
    _require(staged["source_sha256"] == sha256_file(failure_path), "controller does not bind analysis failure hash")
    state_counts = {
        "smoke": state.get("smoke_outcome_episodes") == 24,
        "generated": state.get("confirmation_outcome_episodes_generated") == total_confirmation,
        "executed": state.get("confirmation_outcome_episodes_executed") == total_confirmation,
        "opened": state.get("confirmation_outcomes_opened_for_analysis") is True,
        "expected_fixed": state.get("expected_confirmation_episode_count") == total_confirmation,
        "no_terminal_before_verification": state.get("terminal_label") is None,
    }
    _require(all(state_counts.values()), f"analysis failure controller counts drift: {[name for name, passed in state_counts.items() if not passed]}")
    controller = failure.get("controller_chronology")
    _require(isinstance(controller, Mapping), "analysis failure controller chronology absent")
    _require(
        controller.get("state_path") == paths["state"]
        and controller.get("research_ledger_path") == paths["ledger"]
        and controller.get("controller_status_verified") is True,
        "analysis failure controller path/status drift",
    )
    events = ledger_state["events"]
    staging_event = events[-1]
    _require(
        staging_event.get("event") == "state_completed"
        and staging_event.get("completed_state") == "SEALED_ANALYSIS"
        and staging_event.get("postconfirmation_integrity_failure") is True,
        "analysis failure is not the terminal ledger staging event",
    )
    _require(
        controller.get("ledger_event_count") == len(events) - 1
        and staging_event.get("prev_sha256") == controller.get("ledger_head_sha256"),
        "analysis failure pre-staging ledger chronology drift",
    )
    checkpoints = state["verified_checkpoints"]
    _require(len(checkpoints) >= 2 and controller.get("last_verified_checkpoint") == checkpoints[-2], "analysis failure prior checkpoint pointer drift")

    chronology_labels = ("pre_data", "pre_selection", "pre_confirmation", "confirmation_input", "analysis_failure", "failure_staged")
    chronology_points = (
        int(pre_data["created_unix_ns"]), int(pre_selection["created_unix_ns"]),
        int(pre_confirmation["created_unix_ns"]), int(confirmation_input["created_unix_ns"]),
        int(failure["created_unix_ns"]), int(staged["staged_unix_ns"]),
    )
    _require(all(left <= right for left, right in zip(chronology_points, chronology_points[1:])), "analysis failure chronology inversion")
    source_record = {"path": failure_relative, "sha256": sha256_file(failure_path), "bytes": failure_path.stat().st_size}
    return {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "mode": "confirmation",
        "checkpoint_state": "INDEPENDENT_VERIFICATION",
        "passed": True,
        "terminal_label": TERMINAL_INVALID,
        "process_valid": False,
        "integrity_failure": True,
        "verifier_contract": {
            "path": contract["_loaded_contract_relative"],
            "sha256": sha256_file(policy.path(contract["_loaded_contract_relative"], scope="attempt")),
        },
        "source_hashes": {
            "analysis_execution_invalid": source_record,
            "confirmation_input_seal": sha256_file(policy.path(paths["confirmation_input_seal"], scope="attempt")),
            "power_freeze": sha256_file(policy.path(paths["power_freeze"], scope="attempt")),
        },
        "checks": {
            "terminal_manifest_authenticated": True,
            "controller_chronology_and_open_counts": True,
            "confirmation_input_seal_authenticated_without_array_open": True,
            "mps_scientific_replay_authenticated_without_array_open": True,
            "scientific_objects_unchanged": True,
            "analysis_integrity_failure_authenticated": True,
            "analysis_result_presence_matches_failure_record": True,
            "local_hash_pinned_power_rule_only": True,
        },
        "evidence": {
            "terminal_path_set": terminal_manifest,
            "normative_contracts": normative,
            "ledger_state": {key: ledger_state[key] for key in ("event_count", "legacy_event_count", "ledger_head_sha256", "completed_state_count")},
            "version_forward_lineage": lineage,
            "identifier_role_isolation": {key: identifier[key] for key in ("episode_identifier_count", "numeric_identifier_count", "rng_identifier_count")},
            "power": power,
            "power_provenance": {key: power_provenance[key] for key in ("identity", "input_hashes", "identity_crosslinks", "local_power_rule_only")},
            "confirmation_input_seal": confirmation_input,
            "scientific_replay": replay_check,
            "analysis_execution_invalid": source_record,
            "analysis_result_present": result_present,
            "chronology": {"labels": chronology_labels, "created_unix_ns": chronology_points, "monotone": True},
        },
        "read_only_verifier": True,
        "local_production_modules_imported": False,
        "outcome_arrays_loaded": False,
        "stdout_json_only": True,
    }


def verify(contract_path: Path) -> dict[str, Any]:
    contract, policy = load_contract(contract_path)
    if contract.get("_analysis_execution_invalid_branch"):
        return verify_analysis_execution_invalid_branch(contract, policy)
    mode = str(contract["mode"])
    normative = verify_normative_contracts(contract, policy)
    ledger_state = verify_ledger_and_state(contract, policy)
    terminal_manifest = verify_terminal_manifest(contract, policy)
    lineage_check = verify_version_forward_lineage(
        contract, policy, ledger_state["state"], ledger_state["events"]
    )
    identifier = verify_identifier_ledger(contract, policy)
    cohort = identifier["ledger"]
    seeds = canonical_analysis_seeds(cohort)
    paths = contract["paths"]
    state = ledger_state["state"]
    seals: dict[str, Any] = {}
    seals["pre_data"] = verify_seal(
        policy.path(paths["pre_data_seal"], scope="study"), policy,
        checkpoint_state="PRE_OUTCOME_SEAL",
        expected_counts={"fit": 0, "selection": 0, "smoke": 0, "confirmation_generated": 0, "confirmation_executed": 0, "confirmation_opened": False},
        expected_attempt=SCIENCE_ATTEMPT,
    )
    seals["pre_selection"] = verify_seal(
        policy.path(paths["pre_selection_seal"], scope="study"), policy,
        checkpoint_state="PRE_SELECTION_SEAL",
        expected_counts={"fit": 1200, "selection": 0, "smoke": 0, "confirmation_generated": 0, "confirmation_executed": 0, "confirmation_opened": False},
        expected_pre_data_seal_sha256=lineage_check["seal_sha256"],
    )
    prechecked_power: dict[str, Any] | None = None
    prechecked_confirmation_n = 0
    if mode == "confirmation":
        # Fail closed on the complete confirmation byte closure before *any*
        # NPZ is loaded, including otherwise-benign fit and frozen-gate NPZs.
        prechecked_power = read_json(policy.path(paths["power_freeze"], scope="study"))
        prechecked_confirmation_n = verify_power_result(prechecked_power, mode="confirmation")
        seals["pre_confirmation"] = verify_seal(
            policy.path(paths["pre_confirmation_seal"], scope="study"), policy,
            checkpoint_state="PRE_CONFIRMATION_PACKAGE_SEAL",
            expected_counts={"fit": 1200, "selection": 2000, "smoke": 0, "confirmation_generated": 0, "confirmation_executed": 0, "confirmation_opened": False},
        )
        seals["confirmation_input"] = verify_confirmation_input_bindings(
            contract, policy, prechecked_confirmation_n
        )
    replay_check = verify_scientific_replay_qualification(
        contract,
        policy,
        state,
        confirmation_n=prechecked_confirmation_n,
    )
    base_roles = verify_role_manifests(contract, policy, cohort, 0, roles_to_verify=("fit", "selection"))
    # All development fitting is independently repeated from primitives.  No
    # production module is imported, and selection is opened only after this.
    whitening_path = policy.path(paths["fixed_whitening"])
    _require(sha256_file(whitening_path) == V5_WHITENING_SHA256, "V5 whitening archive hash drift")
    whitening_archive = _load_npz(whitening_path)
    _require("whitening_matrix" in whitening_archive, "V5 whitening matrix absent")
    fixed_whitening = np.asarray(whitening_archive["whitening_matrix"], dtype=np.float64)
    development_manifests = contract.get("development_manifests")
    _require(isinstance(development_manifests, Mapping), "development manifest contract absent")
    fit_role, fit_inputs = load_development_role(
        contract["fit_inputs"], development_manifests["fit"], policy,
        episodes_per_dgp=300, role="fit",
        replay_role=replay_check["roles"]["fit"],
    )
    recomputed = recompute_fitted_candidates(fit_role, fixed_whitening)
    fitted_path = policy.path(paths["fitted_candidates"], scope="study")
    fitted = _load_npz(fitted_path)
    fit_check = compare_fitted_artifact(recomputed, fitted)
    fit_role = {}  # release the large fit arrays before opening selection
    recomputed = {}
    selection_ledger_path = policy.path(paths["selection_ledger"], scope="study")
    selection_ledger = read_json(selection_ledger_path)
    selection_role, selection_inputs = load_development_role(
        contract["selection_inputs"], development_manifests["selection"], policy,
        episodes_per_dgp=500, role="selection",
        replay_role=replay_check["roles"]["selection"],
    )
    selection_check = verify_selection(selection_role, fitted, selection_ledger, seeds)
    selection_role = {}
    compiled: dict[str, np.ndarray] = {}
    freeze: dict[str, Any]
    power_object: dict[str, Any] | None = None
    power_recomputation: dict[str, Any] | None = None
    confirmation_check: dict[str, Any] | None = None
    terminal_label = TERMINAL_FAILED
    confirmation_n = 0
    if mode == "no_candidate":
        _require(selection_check["eligible_count"] == 0 and selection_check["selected_candidate_id"] is None, "no-candidate mode has an eligible gate")
        _require_no_role_artifacts(policy, contract)
        freeze = {"selected": False, "fit_lock_created_unix_ns": int(read_json(policy.path(paths["fit_lock"], scope="study"))["created_unix_ns"])}
        _require(state["smoke_outcome_episodes"] == 0 and state["confirmation_outcome_episodes_generated"] == 0 and state["confirmation_outcome_episodes_executed"] == 0 and state["confirmation_outcomes_opened_for_analysis"] is False, "no-candidate state opened later roles")
    else:
        compiled, freeze = verify_no_refit_and_freeze(contract, policy, fitted, selection_ledger)
        _require(freeze.get("selected") is True, "post-selection mode lacks a frozen gate")
        power_path = policy.path(paths["power_freeze"], scope="study")
        power_object = prechecked_power or read_json(power_path)
        confirmation_n = prechecked_confirmation_n or verify_power_result(power_object, mode=mode)
        power_recomputation = verify_binding_power_recomputation(
            power_object, fitted, selection_ledger, contract, policy
        )
        if mode == "power_infeasible":
            _require_no_role_artifacts(policy, contract)
            _require(state["smoke_outcome_episodes"] == 0 and state["confirmation_outcome_episodes_generated"] == 0 and state["confirmation_outcome_episodes_executed"] == 0 and state["confirmation_outcomes_opened_for_analysis"] is False, "power-infeasible state opened later roles")
        else:
            all_roles = verify_role_manifests(contract, policy, cohort, confirmation_n, roles_to_verify=ROLE_ORDER)
            confirmation_check = verify_confirmation_analysis(
                contract,
                policy,
                cohort,
                compiled,
                seeds,
                confirmation_n,
                replay_check["roles"]["confirmation"],
            )
            terminal_label = confirmation_check["terminal_label"]
            base_roles = all_roles
            analysis_object = read_json(policy.path(paths["analysis_result"], scope="attempt"))
            _require(state["smoke_outcome_episodes"] == 24 and state["confirmation_outcome_episodes_generated"] == 4 * confirmation_n and state["confirmation_outcome_episodes_executed"] == 4 * confirmation_n and state["confirmation_outcomes_opened_for_analysis"] is True, "confirmation state counters drift")
    analysis_object = read_json(policy.path(paths["analysis_result"], scope="attempt")) if mode == "confirmation" else None
    chronology = verify_chronology(mode, seals, base_roles, freeze, power_object, analysis_object)
    source_hashes = {
        "selection_ledger": sha256_file(selection_ledger_path),
        "fitted_candidates": sha256_file(fitted_path),
    }
    if power_object is not None:
        source_hashes["power_freeze"] = sha256_file(policy.path(paths["power_freeze"], scope="study"))
    if confirmation_check is not None:
        source_hashes.update({
            "analysis_result": confirmation_check["analysis_result_sha256"],
            "bootstrap_replicates": confirmation_check["bootstrap_replicates_sha256"],
            "confirmation_input_seal": sha256_file(policy.path(paths["confirmation_input_seal"], scope="attempt")),
        })
    return {
        "schema_version": 1, "attempt": ATTEMPT, "mode": mode,
        "checkpoint_state": "INDEPENDENT_VERIFICATION", "passed": True,
        "terminal_label": terminal_label,
        "verifier_contract": {
            "path": contract_path.resolve().relative_to(policy.repository_root).as_posix(),
            "sha256": sha256_file(contract_path),
        },
        "source_hashes": source_hashes,
        "checks": {
            "normative_contracts": True,
            "ledger_state_and_chronology": True,
            "version_forward_lineage": True,
            "mps_scientific_replay_before_numpy": True,
            "global_identifier_and_role_isolation": True,
            "fit_recomputed_from_primitive_causal_features": True,
            "selection_recomputed_without_refit": True,
            "terminal_path_set_complete": True,
            "mode_specific_terminal_mapping": True,
            "confirmation_recomputed_elementwise_if_applicable": confirmation_check is not None or mode != "confirmation",
        },
        "evidence": {
            "normative_contracts": normative, "ledger_state": {key: ledger_state[key] for key in ("event_count", "legacy_event_count", "ledger_head_sha256", "completed_state_count")},
            "version_forward_lineage": lineage_check,
            "scientific_replay": replay_check,
            "identifier_role_isolation": {key: identifier[key] for key in ("episode_identifier_count", "numeric_identifier_count", "rng_identifier_count")},
            "role_manifests": base_roles, "fit_inputs": fit_inputs, "fit_recomputation": fit_check,
            "selection_inputs": selection_inputs, "selection_recomputation": selection_check,
            "freeze_no_refit": freeze, "power": power_object,
            "power_recomputation": power_recomputation, "confirmation": confirmation_check,
            "chronology": chronology, "terminal_path_set": terminal_manifest,
        },
        "read_only_verifier": True,
        "local_production_modules_imported": False,
        "stdout_json_only": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("contract", type=Path)
    try:
        args = parser.parse_args(argv)
        result = verify(args.contract)
        print(json.dumps(_jsonable(result), sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1, "attempt": ATTEMPT,
            "checkpoint_state": "INDEPENDENT_VERIFICATION", "passed": False,
            "error_type": type(exc).__name__, "error": str(exc),
            "read_only_verifier": True, "local_production_modules_imported": False,
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
