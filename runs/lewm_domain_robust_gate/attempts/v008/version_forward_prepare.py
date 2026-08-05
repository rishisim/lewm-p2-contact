#!/usr/bin/env python3
"""Prepare the immutable v007 -> v008 checkpoint-recovery authorization.

This program is deliberately stdlib-only and outcome-blind.  It reads source,
configuration, controller, and seal bytes; it never imports an attempt module
and never decodes an array.  Merely importing or running ``check`` creates no
artifact.  ``write`` is the sole mutating command and uses exclusive creation.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True


SCIENCE_ATTEMPT = "v001"
INTERMEDIATE_ATTEMPT = "v004"
SOURCE_PARENT_ATTEMPT = "v005"
SELECTION_SOURCE_ATTEMPT = "v006"
SOURCE_ATTEMPT = "v007"
TARGET_ATTEMPT = "v008"
FIT_SOURCE_ATTEMPT = "v003"
RESUME_STATE = "SELECTION_COHORTS"
AUTHORIZATION_KIND = "zero_confirmation_outcome_version_forward_inherited_pre_selection"
OUTPUT_RELATIVE = "audit/pre_data_inheritance_seal.json"
INVALIDITY_RELATIVE = "audit/v007_procedural_invalidity.json"
INVALIDITY_DRAFT_RELATIVE = "audit/v007_procedural_invalidity_draft.json"
SOURCE_SEAL_RELATIVE = "audit/pre_data_inheritance_seal.json"
SOURCE_RECEIPT_RELATIVE = "audit/version_forward_transaction_receipt.json"
SOURCE_PRE_SELECTION_SEAL_RELATIVE = "audit/pre_selection_seal.json"
FIT_INVENTORY_RELATIVE = "audit/inherited_fit_inventory.json"
EXPECTED_INVALIDITY_SHA256 = "adddaa7ebf470866365da414d74f6094f4aefaeb02480e81c5b75961204d584e"
EXPECTED_INVALIDITY_DRAFT_SHA256 = "83014917878d8fc226add312284f6c34795ee88d6f49dab80079f04bacea75b5"
EXPECTED_SOURCE_SEAL_SHA256 = "8868038420c049f649b2e37c4b17dd87dbbbc4d8042b43fb003b4efa8e288190"
EXPECTED_SOURCE_RECEIPT_SHA256 = "f038c25974c3ddd1bf7ea4ae9e9be68d1230a359653d6ba3c90558e9034b0818"
EXPECTED_SOURCE_PRE_SELECTION_SEAL_SHA256 = "3ebff0d56f515b399a4000df2d20f39314cd2e2aadb31bd271ab75f42e58327c"
EXPECTED_SOURCE_FIT_INVENTORY_SHA256 = "b70f6ca1d63b92e4e512a063359fd990d0263b5f1b83d6e51ce9ddd6487582c3"
EXPECTED_SOURCE_PRE_FORWARD_STATE_SHA256 = "0493b0f1bcb4c2d1b91fdc51548d64af20d568709ee3aa33611c1d234dd030c5"
EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256 = "a6a9374dddddfc20254d8fd532252c045cec8a7c4ab31a06ea7a703de542d0db"
EXPECTED_SOURCE_STATE_SHA256 = "0493b0f1bcb4c2d1b91fdc51548d64af20d568709ee3aa33611c1d234dd030c5"
EXPECTED_SOURCE_LEDGER_SHA256 = "a6a9374dddddfc20254d8fd532252c045cec8a7c4ab31a06ea7a703de542d0db"
EXPECTED_COHORT_LEDGER_SHA256 = "1b7f524ed11cf787e67e4887590d985cfa4eb928e333e7adb7667e48a0b8785d"
EXPECTED_SOURCE_GENERATOR_SHA256 = "a36d532fcc2ac37d076e7826cf1506cd5e7eda9366c4e136e7defff301f89d83"
EXPECTED_SOURCE_LAUNCHER_SHA256 = "4f0a6d185d1cbe15378e58d8c51a98271f6a583fc11d49e1204f4c826264d5a0"
EXPECTED_SOURCE_ADAPTER_SHA256 = "28ae15486800ca0d17c85028b7011e7a3a88a1cfee6f53241f75136dbfc9b752"
EXPECTED_FIT_SOURCE_INVALIDITY_SHA256 = "25be74578dc22eb2c1b4a3cde6b0dbd7e5a788d1bcb146f251b76fa40bc95165"
EXPECTED_FIT_SOURCE_SEAL_SHA256 = "74018164e9e208babd7a3b420053a58e665e98694e4116d5a4f1cf70f143d943"
EXPECTED_FIT_SOURCE_RECEIPT_SHA256 = "8cdd1a24aacdaecf4eda85fb5b9119536d88bec92e0cc64db6a3bcc7d9ecb262"
EXPECTED_FIT_SOURCE_STATE_SHA256 = "53788a9e7c1633f95c66af682d69b32bfcc73baee4db4e4c3176e0fb9456d821"
EXPECTED_FIT_SOURCE_LEDGER_SHA256 = "611be25f22716de7b593dea523904fc871b854962a8fe9e936029e4ee0dd237b"
EXPECTED_FIT_FILE_COUNT = 6013
EXPECTED_FIT_EPISODE_COUNT = 1200
EXPECTED_TRANSITIVE_SEALED_FILE_COUNT = 6541
SOURCE_OPERATIONAL_LOCKS = frozenset()
SOURCE_SELECTION_BOUNDARY_PRODUCTS = frozenset()

OUTCOME_KEYS = (
    "fit_outcome_episodes",
    "selection_outcome_episodes",
    "smoke_outcome_episodes",
    "confirmation_outcome_episodes_generated",
    "confirmation_outcome_episodes_executed",
    "confirmation_outcomes_opened_for_analysis",
)
EXPECTED_OUTCOMES: dict[str, int | bool] = {
    "fit_outcome_episodes": 1200,
    "selection_outcome_episodes": 0,
    "smoke_outcome_episodes": 0,
    "confirmation_outcome_episodes_generated": 0,
    "confirmation_outcome_episodes_executed": 0,
    "confirmation_outcomes_opened_for_analysis": False,
}
SOURCE_ACTIVATION_OUTCOMES: dict[str, int | bool] = {
    **EXPECTED_OUTCOMES,
}

EXPECTED_ALLOWED_SOURCE_SUFFIXES = frozenset({".py", ".json", ".md"})
EXPECTED_PRE_DATA_STATIC_PATHS = frozenset(
    {
        "DGP_MATRIX.json", "DIAGNOSTIC_ACCOUNT.md", "PREREGISTRATION.md",
        "analysis.py", "build_manifest.py", "build_seed_ledger.py",
        "candidate_grid.json", "capture_verifier.py", "checkpoints.py",
        "cohort_seed_ledger.json", "compile_gate.py", "counted_features.py",
        "fit_inheritance.py", "fit_select.py", "flops.py", "generator.py", "inherited_authorization.py",
        "independent_verify.py", "input_loader.py", "latency.py", "launcher.py",
        "outcome_mapping.json", "power_analysis.py", "power_baseline.json",
        "power_rule.json", "preseal.py", "runner.py", "runtime_contract.py",
        "scientific_replay.py", "scientific_replay_launcher.py",
        "study_common.py", "terminal_workflow.py",
        "tests/test_analysis.py", "tests/test_attempt_parameterization_v002.py",
        "tests/test_checkpoint_hardening.py", "tests/test_fit_select.py",
        "tests/test_generator_contract.py", "tests/test_independent_verify.py",
        "tests/test_inherited_authorization_v002.py",
        "tests/test_manifest_closure_v002.py", "tests/test_preseal.py",
        "tests/test_program_controller.py", "tests/test_runner_gate.py",
        "tests/test_scientific_replay.py", "tests/test_seed_power.py",
        "tests/test_terminal_workflow.py",
        "tests/test_version_forward_v002.py", "tests/test_version_forward_transaction.py",
        "tests/test_workflow.py",
        "verifier_contract.json", "verifier_contract_no_candidate.json",
        "verifier_contract_power_infeasible.json", "verify_identifier_freshness.py",
        "verify_version_forward.py", "version_forward_prepare.py",
        "version_forward_transaction.py", "workflow.py",
    }
)
EXPECTED_PRE_DATA_AUDIT_INPUTS = frozenset(
    {
        "audit/bootstrap_audit.json", "audit/diagnostic_account.json",
        "audit/identifier_freshness_verification.json",
        "audit/preregistration_and_power.json",
    }
)
EXPECTED_REPOSITORY_PATHS = frozenset(
    {
        "runs/lewm_domain_robust_gate/LEDGER_CHAIN_GENESIS.json",
        "runs/lewm_domain_robust_gate/README.md",
        "runs/lewm_domain_robust_gate/program.py",
    }
)
EXPECTED_MUTABLE_REPOSITORY_PATHS = frozenset(
    {
        "runs/lewm_domain_robust_gate/STATE.json",
        "runs/lewm_domain_robust_gate/STATE_TRANSACTION.json",
        "runs/lewm_domain_robust_gate/VERSION_FORWARD_TRANSACTION.json",
    }
)
EXPECTED_GENERATED_AUDIT_PATHS = frozenset(
    {
        "audit/analysis_execution_invalid.json", "audit/candidate_selection.json",
        "audit/confirmation_execution_complete.json",
        "audit/confirmation_generation_complete.json",
        "audit/confirmation_input_seal.json",
        "audit/confirmation_power_and_cohort_freeze.json",
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
    }
)
EXPECTED_GENERATED_ATTEMPT_PRODUCTS = frozenset(
    {
        "FOLLOW_ON_TASK.md", "INDEPENDENT_AUDIT.md", "LIMITATIONS.md", "REPORT.md",
        "ROBUSTNESS_MAP.json", "analysis_result.json", "decision.json",
        "fit/fit_lock.json", "freeze/compiled_gate_manifest.json",
        "freeze/gate_freeze.json", "metrics/bootstrap_summary.json",
        "metrics/compute_ledger.json", "metrics/fit_power_summary.json",
        "metrics/latency_and_resources.json", "metrics/selection_power_summary.json",
        "power_analysis.json", "selection/selection_ledger.json",
    }
)
EXPECTED_GENERATED_ROLES = frozenset({"fit", "selection", "smoke", "confirmation"})
EXPECTED_GENERATED_REGIMES = frozenset(
    {"native_plan", "markov_oracle", "plan_action_noise_0p2", "plan_random_action_0p1"}
)
EXPECTED_STUDY_ROOT_OPERATIONAL_NAMES = frozenset(
    {".program.lock", "RESEARCH_LEDGER.jsonl"}
)
EXPECTED_STUDY_ROOT_DIRECTORY_NAMES = frozenset({"attempts"})

# These files differ only in active-attempt plumbing.  Their complete ASTs,
# after the narrow transform implemented below, must be identical.
NORMALIZED_AST_PATHS = frozenset(
    {
        "power_analysis.py",
    }
)
CANONICAL_CONTRACT_PATHS = frozenset(
    {
        "verifier_contract.json",
        "verifier_contract_no_candidate.json",
        "verifier_contract_power_infeasible.json",
    }
)
PROCEDURAL_EXISTING_PATHS = frozenset(
    {
        "analysis.py",
        "build_manifest.py",
        "capture_verifier.py",
        "checkpoints.py",
        "compile_gate.py",
        "generator.py",
        "independent_verify.py",
        "latency.py",
        "launcher.py",
        "preseal.py",
        "runner.py",
        "study_common.py",
        "tests/test_analysis.py",
        "tests/test_checkpoint_hardening.py",
        "tests/test_generator_contract.py",
        "tests/test_independent_verify.py",
        "tests/test_preseal.py",
        "tests/test_runner_gate.py",
        "tests/test_seed_power.py",
        "tests/test_terminal_workflow.py",
        "tests/test_workflow.py",
        "terminal_workflow.py",
        "workflow.py",
    }
)
NEW_LINEAGE_SUPPORT_PATHS = frozenset(
    {
        "inherited_authorization.py",
        "scientific_replay.py",
        "scientific_replay_launcher.py",
        "tests/test_attempt_parameterization_v002.py",
        "tests/test_inherited_authorization_v002.py",
        "tests/test_manifest_closure_v002.py",
        "tests/test_scientific_replay.py",
        "tests/test_version_forward_v002.py",
        "tests/test_version_forward_transaction.py",
        "verify_version_forward.py",
        "version_forward_prepare.py",
        "version_forward_transaction.py",
    }
)

# Existing procedural modules must change exactly these top-level definitions
# or bindings.  The separately transcribed ceiling and required sets are
# intentionally explicit: neither missing hardening nor extra deltas can be
# approved by observing the target dynamically.
PROCEDURAL_ALLOWED_SYMBOLS: dict[str, frozenset[str]] = {
    "analysis.py": frozenset(
        {
            "ACTIVE_ATTEMPT",
            "ATTEMPT",
            "_fsync_parent_directory",
            "_publish_temporary",
            "atomic_json",
            "atomic_npz",
            "capture_post_open_analysis_failure",
            "run_sealed_analysis",
            "validate_sealed_analysis_authorization",
        }
    ),
    "build_manifest.py": frozenset(
        {
            "<docstring>",
            "ACTIVE_ATTEMPT",
            "EXPECTED_ACTIVE_ATTEMPT",
            "GENERATED_ATTEMPT_PRODUCTS",
            "GENERATED_AUDIT_NAMES",
            "GENERATED_EXECUTION_FAILURE_NAMES",
            "GENERATED_REGIMES",
            "GENERATED_ROLES",
            "MUTABLE_STUDY_ROOT_RELATIVE_PATHS",
            "PRE_DATA_LABEL",
            "PRE_DATA_REPOSITORY_RELATIVE_PATHS",
            "PRE_DATA_STATIC_RELATIVE_PATHS",
            "SCIENCE_ATTEMPT",
            "_allowed_suffix",
            "_assert_discovered_integrity",
            "_is_generated_data_json",
            "_manifest_for_paths",
            "_study_root",
            "_walk_attempt_candidates",
            "_walk_study_root_candidates",
            "build_pre_data_manifest",
            "classify_pre_data_source_paths",
            "collect_pre_data_source_paths",
            "discovered_python_source_paths",
            "is_generated_attempt_product",
            "source_closure_policy",
            "unexpected_pre_data_source_paths",
            "unexpected_python_source_paths",
            "verify_manifest",
        }
    ),
    "capture_verifier.py": frozenset(),
    "checkpoints.py": frozenset(
        {
            "<imports>",
            "ATTEMPT",
            "CONFIRMATION_MAX_EPISODES_PER_DGP",
            "CONFIRMATION_REPLACEMENTS_PER_DGP",
            "REGIME_SLUGS",
            "ROLE_ORDER",
            "ROLE_PRIMARY_COUNTS",
            "SEED_FIELDS",
            "V5_FIXED_WHITENING",
            "_assert_exact_confirmation_namespace",
            "_attempt_root_from_manifest",
            "_canonical_relative",
            "_confirmation_contract",
            "_confirmation_files",
            "_exact_compute_from_histograms",
            "_exact_keys",
            "_exact_seal_link",
            "_expected_episode_id",
            "_observed_file_link",
            "_seed_tuple",
            "_strict_file_link",
            "_strict_int",
            "_strict_sha256",
            "_validate_raw_array_metadata",
            "_validate_rollout_failure_log",
            "_validate_seed_record",
            "_validate_source_binding",
            "_validate_source_raw_manifest_binding",
            "_verify_confirmation_execution_manifest",
            "_verify_confirmation_raw_manifest",
            "atomic_json",
            "confirmation_execution_checkpoint",
            "confirmation_generation_checkpoint",
            "confirmation_input_seal",
            "verify_execution_manifest",
            "verify_raw_manifest",
        }
    ),
    "compile_gate.py": frozenset(
        {
            "_atomic_json",
            "_atomic_npz",
            "_fsync_parent_directory",
            "_publish_exclusive_temporary",
        }
    ),
    "generator.py": frozenset(
        {
            "<imports>",
            "ATTEMPT",
            "EPISODE_ID_PATTERN",
            "PRE_DATA_SEAL_PATH",
            "REGIME_SLUGS",
            "ROLE_SLUGS",
            "_append_jsonl",
            "_assert_safe_directory_chain",
            "_assert_safe_output_leaf",
            "_canonical_output_file",
            "_expected_episode_id",
            "_failure_descriptor",
            "_failure_genesis",
            "_failure_hash",
            "_generate_locked",
            "_lexical_attempt_relative",
            "_strict_failure_records",
            "_validate_destination_episode_id",
            "_validate_episode_id",
            "_validate_seed_record",
            "episode_paths",
            "generate",
            "load_generation_inputs",
            "materialization_lock",
            "materialization_lock_path",
            "preflight_output_namespace",
            "raw_directory",
            "read_rollout_failures",
            "record_rollout_failure",
            "validate_dgp_matrix",
            "validate_existing_manifest",
            "validate_raw_manifest_contract",
            "validate_seed_ledger",
            "verify_authorization_seal",
            "verify_episode_record",
        }
    ),
    "independent_verify.py": frozenset(
        {
            "<imports>",
            "ACTIVE_ATTEMPT",
            "ALLOWED_ATTEMPTS",
            "ATTEMPT",
            "ATTEMPT_ROOTS",
            "CANONICAL_CONTRACT_PATHS",
            "INHERITANCE_SEAL_RELATIVE",
            "INHERITED_COMPLETED_STATES",
            "INVALIDITY_RELATIVE",
            "NEW_LINEAGE_SUPPORT_PATHS",
            "NORMALIZED_AST_PATHS",
            "PROCEDURAL_ALLOWED_SYMBOLS",
            "PROCEDURAL_EXPECTED_CHANGED_SYMBOLS",
            "PROCEDURAL_EXISTING_PATHS",
            "SCIENTIFIC_OBJECT_PATHS",
            "SOURCE_PRE_DATA_RELATIVE",
            "STUDY_RELATIVE",
            "VERSION_FORWARD_RESUME_STATE",
            "_AdministrativeAstNormalizer",
            "_assignment_names",
            "_canonicalize_contract",
            "_is_active_attempt_guard",
            "_lineage_file_link",
            "_normalize_normative_sources",
            "_source_manifest_relative_paths",
            "_top_level_symbol_hashes",
            "canonical_contract_sha256",
            "load_contract",
            "normalized_administrative_ast_sha256",
            "verify",
            "verify_analysis_execution_invalid_branch",
            "verify_confirmation_analysis",
            "verify_equivalence_partitions",
            "verify_ledger_and_state",
            "verify_role_manifests",
            "verify_seal",
            "verify_version_forward_lineage",
        }
    ),
    "latency.py": frozenset(
        {
            "ACTIVE_ATTEMPT",
            "_fsync_parent_directory",
            "_publish_temporary",
            "atomic_json",
            "run_latency_suite",
            "validate_latency_result",
        }
    ),
    "preseal.py": frozenset(
        {
            "<imports>",
            "ACTIVE_ATTEMPT",
            "PRE_DATA_SEAL_PATH",
            "REQUIRED_IMPLEMENTATION_FILES",
            "SCIENCE_ATTEMPT",
            "_files_under",
            "_state",
            "_verified_checkpoint",
            "implementation_complete",
            "preseal_qualification",
            "seal_pre_confirmation",
            "seal_pre_data",
            "seal_pre_selection",
            "validate_power_input_coherence",
            "validate_seed_freshness_evidence",
            "validate_verifier_contracts",
        }
    ),
    "runner.py": frozenset(
        {
            "<imports>",
            "ATTEMPT_VERSION",
            "_assert_regular_unlinked",
            "_enforce_execution_runtime",
            "_execute_confirmation_locked",
            "_execute_development_locked",
            "_execution_record_if_valid",
            "_execution_source_bindings",
            "_load_raw_manifest",
            "_preflight_execution_namespace",
            "_state_authorization",
            "_validate_existing_execution_manifest",
            "execute_confirmation",
            "execute_development",
        }
    ),
    "study_common.py": frozenset(
        {"_fsync_parent_directory", "_publish_temporary", "atomic_json", "atomic_npz"}
    ),
    "tests/test_independent_verify.py": frozenset(
        {
            "_policy",
            "test_analysis_execution_invalid_branch_is_preload_and_binds_result_presence",
            "test_power_provenance_is_local_identity_bound_and_tamper_evident",
            "test_v002_equivalence_partitions_are_independently_recomputed_and_tamper_evident",
            "test_v002_equivalence_partitions_reject_omitted_required_symbol",
            "test_v002_equivalence_partitions_reject_wildcard_procedural_scope",
            "test_v002_lineage_normalization_and_contract_canonicalization_are_tamper_evident",
            "test_verifier_imports_only_stdlib_and_numpy_and_failure_stdout_is_one_json_line",
            "test_zero_confirmation_version_forward_requires_explicit_equivalence",
        }
    ),
    "tests/test_analysis.py": frozenset(
        {
            "_authorization_fixture",
            "_latency_binding_and_result",
            "test_post_open_failure_capture_is_outcome_free_and_never_overwrites",
        }
    ),
    "tests/test_checkpoint_hardening.py": frozenset(
        {
            "<imports>",
            "JSON_EXCLUSIVE_WRITERS",
            "NPZ_EXCLUSIVE_WRITERS",
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
        }
    ),
    "tests/test_generator_contract.py": frozenset(
        {
            "test_intent_only_partial_artifact_stops_without_reroll",
            "test_paths_are_role_and_regime_isolated",
            "test_replacement_registry_is_global_across_roles_and_dgps",
        }
    ),
    "tests/test_preseal.py": frozenset(
        {
            "<docstring>",
            "test_compile_and_power_are_state_gated_inside_exact_eval_worker",
            "test_power_input_paths_resolve_from_repository_not_cwd",
            "test_power_provenance_binds_rule_locks_freeze_and_selected_identity",
            "test_pre_data_manifest_ignores_later_stage_top_level_products",
            "test_pre_data_manifest_rejects_unlisted_python_source",
            "test_preseal_state_uses_verified_controller_not_direct_state_read",
            "test_python_cache_hygiene_fails_closed",
            "test_stage_file_collection_rejects_supplied_directory_symlink_before_rglob",
        }
    ),
    "tests/test_runner_gate.py": frozenset(
        {
            "test_complete_confirmation_part_is_idempotently_resumed",
            "test_execution_handlers_do_not_capture_process_interruptions",
        }
    ),
    "tests/test_seed_power.py": frozenset(
        {"_claim_summary", "test_power_cli_binds_gate_locks_and_persists_only_repo_relative_paths"}
    ),
    "tests/test_terminal_workflow.py": frozenset(
        {
            "_analysis",
            "_attempt",
            "_captured_audit",
            "_contract",
            "_state",
            "test_early_decision_has_zero_later_roles_and_reports_are_claim_bounded",
            "test_report_artifact_mutation_is_not_repaired",
            "test_result_present_analysis_integrity_failure_is_manifested_but_not_called",
        }
    ),
    "tests/test_workflow.py": frozenset(
        {
            "<docstring>",
            "FakeController",
            "_configure_power_paths",
            "_controller_json",
            "_install_fit_count_crash",
            "_ledger_record",
            "_power_result",
            "_prior_selection_audit",
            "_synthetic_role_manifests",
            "test_count_crash_recovery_rejects_state_ledger_audit_and_manifest_tamper",
            "test_default_freeze_writes_identity_summaries_only_after_gate_freeze",
            "test_exact_role_count_is_resume_idempotent",
            "test_exact_role_count_without_authenticated_crash_record_fails_closed",
            "test_fit_count_commit_crash_reauthenticates_manifests_and_advances_once",
            "test_inherited_early_stop_adapter_rejects_unsealed_active_contract",
            "test_loaded_workflow_controller_adapts_both_inherited_early_stop_contracts",
            "test_sealed_analysis_integrity_failure_is_staged_not_advanced",
        }
    ),
    "workflow.py": frozenset(
        {
            "<imports>",
            "ATTEMPT",
            "DEFAULT_EVIDENCE_PATHS",
            "LEDGER_PATH",
            "ROLE_COUNTER_FIELDS",
            "ROLE_EXECUTION_SEALS",
            "ROLE_RAW_SEALS",
            "ROLE_SEALS",
            "Workflow",
            "_common_role_authorization_state_sha256",
            "_controller_state_object_sha256",
            "_load_controller",
            "_require_inherited_presealed_verifier_contract",
            "_verify_role_count_update_recovery",
            "_verify_role_regime",
            "verify_role_manifests",
        }
    ),
}

PROCEDURAL_EXPECTED_CHANGED_SYMBOLS: dict[str, frozenset[str]] = {
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
    "tests/test_independent_verify.py": frozenset({"_policy", "test_analysis_execution_invalid_branch_is_preload_and_binds_result_presence", "test_power_provenance_is_local_identity_bound_and_tamper_evident", "test_v002_equivalence_partitions_are_independently_recomputed_and_tamper_evident", "test_v002_equivalence_partitions_reject_omitted_required_symbol", "test_v002_equivalence_partitions_reject_wildcard_procedural_scope", "test_v002_lineage_normalization_and_contract_canonicalization_are_tamper_evident", "test_verifier_imports_only_stdlib_and_numpy_and_failure_stdout_is_one_json_line", "test_zero_confirmation_version_forward_requires_explicit_equivalence"}),
    "tests/test_preseal.py": frozenset({"<docstring>", "test_compile_and_power_are_state_gated_inside_exact_eval_worker", "test_power_input_paths_resolve_from_repository_not_cwd", "test_power_provenance_binds_rule_locks_freeze_and_selected_identity", "test_pre_data_manifest_ignores_later_stage_top_level_products", "test_pre_data_manifest_rejects_unlisted_python_source", "test_preseal_state_uses_verified_controller_not_direct_state_read", "test_python_cache_hygiene_fails_closed", "test_stage_file_collection_rejects_supplied_directory_symlink_before_rglob"}),
    "tests/test_runner_gate.py": frozenset({"test_complete_confirmation_part_is_idempotently_resumed", "test_execution_handlers_do_not_capture_process_interruptions"}),
    "tests/test_seed_power.py": frozenset({"_claim_summary", "test_power_cli_binds_gate_locks_and_persists_only_repo_relative_paths"}),
    "tests/test_terminal_workflow.py": frozenset({"_analysis", "_attempt", "_captured_audit", "_contract", "_state", "test_early_decision_has_zero_later_roles_and_reports_are_claim_bounded", "test_report_artifact_mutation_is_not_repaired", "test_result_present_analysis_integrity_failure_is_manifested_but_not_called"}),
    "tests/test_workflow.py": frozenset({"<docstring>", "FakeController", "_configure_power_paths", "_controller_json", "_install_fit_count_crash", "_ledger_record", "_power_result", "_prior_selection_audit", "_synthetic_role_manifests", "test_count_crash_recovery_rejects_state_ledger_audit_and_manifest_tamper", "test_default_freeze_writes_identity_summaries_only_after_gate_freeze", "test_exact_role_count_is_resume_idempotent", "test_exact_role_count_without_authenticated_crash_record_fails_closed", "test_fit_count_commit_crash_reauthenticates_manifests_and_advances_once", "test_inherited_early_stop_adapter_rejects_unsealed_active_contract", "test_loaded_workflow_controller_adapts_both_inherited_early_stop_contracts", "test_sealed_analysis_integrity_failure_is_staged_not_advanced"}),
    "terminal_workflow.py": frozenset({"<docstring>", "ACTIVE_ATTEMPT", "ATTEMPT", "CONTROLLER", "_analysis_for_decision", "_audit_common", "_manifest_contract", "_robustness_map", "_validate_manifest_state", "build_postterminal_reports", "build_terminal_decision", "build_terminal_input_manifest", "complete_terminal_workflow", "finalize_terminal", "verify_terminal_input_manifest"}),
    "workflow.py": frozenset({"<imports>", "ATTEMPT", "DEFAULT_EVIDENCE_PATHS", "LEDGER_PATH", "ROLE_COUNTER_FIELDS", "ROLE_EXECUTION_SEALS", "ROLE_RAW_SEALS", "ROLE_SEALS", "Workflow", "_common_role_authorization_state_sha256", "_controller_state_object_sha256", "_load_controller", "_require_inherited_presealed_verifier_contract", "_verify_role_count_update_recovery", "_verify_role_regime", "verify_role_manifests"}),
}

# Final, independently transcribed additions from the frozen v002 implementation
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
        "test_confirmation_consumer_accepts_only_the_closed_v002_intent_contract",
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

# Final live policy for the narrow v007 -> v008 recovery edge.  Every target
# static path is explicitly classified exactly once.  Scientific objects are
# either byte-identical or normalized only for active-attempt administration;
# recovery and lineage changes receive explicit top-level-unit census policy.
EXACT_HASH_PATHS = frozenset(
    {
        'DGP_MATRIX.json',
        'DIAGNOSTIC_ACCOUNT.md',
        'PREREGISTRATION.md',
        'audit/bootstrap_audit.json',
        'audit/diagnostic_account.json',
        'audit/identifier_freshness_verification.json',
        'audit/preregistration_and_power.json',
        'build_seed_ledger.py',
        'candidate_grid.json',
        'capture_verifier.py',
        'cohort_seed_ledger.json',
        'compile_gate.py',
        'counted_features.py',
        'fit_select.py',
        'flops.py',
        'generator.py',
        'input_loader.py',
        'outcome_mapping.json',
        'power_baseline.json',
        'power_rule.json',
        'runtime_contract.py',
        'runner.py',
        'study_common.py',
        'tests/test_fit_select.py',
        'tests/test_generator_contract.py',
        'tests/test_program_controller.py',
        'verify_identifier_freshness.py',
    }
)
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
PROCEDURAL_EXISTING_PATHS = frozenset()
PROCEDURAL_EXPECTED_CHANGED_SYMBOLS: dict[str, frozenset[str]] = {}
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
STATIC_PARTITION_PATHS = frozenset().union(
    EXACT_HASH_PATHS,
    NORMALIZED_AST_PATHS,
    CANONICAL_CONTRACT_PATHS,
    PROCEDURAL_EXISTING_PATHS,
    NEW_LINEAGE_SUPPORT_PATHS,
    LINEAGE_SUPPORT_PATHS,
)


SCIENTIFIC_OBJECT_PATHS = frozenset(
    {
        "DGP_MATRIX.json",
        "DIAGNOSTIC_ACCOUNT.md",
        "PREREGISTRATION.md",
        "analysis.py",
        "audit/bootstrap_audit.json",
        "audit/diagnostic_account.json",
        "audit/identifier_freshness_verification.json",
        "audit/preregistration_and_power.json",
        "candidate_grid.json",
        "cohort_seed_ledger.json",
        "compile_gate.py",
        "counted_features.py",
        "fit_select.py",
        "flops.py",
        "input_loader.py",
        "outcome_mapping.json",
        "power_analysis.py",
        "power_baseline.json",
        "power_rule.json",
    }
)


class PreparationError(RuntimeError):
    """The proposed inheritance edge is incomplete or non-equivalent."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"JSON value is not an object: {path}")
    return value


def _canonical_json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(repo_root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise PreparationError(f"path is outside repository: {path}") from exc


def _resolve_relative(repo_root: Path, raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreparationError(f"unsafe repository-relative path: {raw}")
    path = (repo_root / relative).resolve(strict=False)
    if not path.is_relative_to(repo_root.resolve(strict=True)):
        raise PreparationError(f"path escapes repository: {raw}")
    return path


def _attempt_roots(study_root: Path) -> tuple[Path, Path, Path]:
    lexical = Path(study_root).absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise PreparationError(f"study/attempt ancestor symlink forbidden: {current}")
    study = lexical.resolve(strict=True)
    repo = study.parents[1]
    source = study / "attempts" / SOURCE_ATTEMPT
    target = study / "attempts" / TARGET_ATTEMPT
    if not source.is_dir() or not target.is_dir():
        raise PreparationError("v002 and v008 attempt roots must both exist")
    return repo, source, target


def _literal_collection(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id != "frozenset" or len(node.args) != 1 or node.keywords:
            raise PreparationError("unsupported manifest collection expression")
        return _literal_collection(node.args[0])
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise PreparationError("manifest closure constant is not literal") from exc
    if not isinstance(value, (set, frozenset, tuple, list)):
        raise PreparationError("manifest closure constant is not a collection")
    if not all(isinstance(item, str) for item in value):
        raise PreparationError("manifest closure collection contains non-string")
    return set(value)


def _manifest_constants(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {
        "ALLOWED_SOURCE_SUFFIXES",
        "PRE_DATA_STATIC_RELATIVE_PATHS",
        "PRE_DATA_REPOSITORY_RELATIVE_PATHS",
        "PRE_DATA_AUDIT_INPUTS",
        "GENERATED_AUDIT_NAMES",
        "GENERATED_ATTEMPT_PRODUCTS",
        "GENERATED_ROLES",
        "GENERATED_REGIMES",
        "MUTABLE_STUDY_ROOT_RELATIVE_PATHS",
    }
    result: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            result[target.id] = _literal_collection(node.value)
    if set(result) != wanted:
        raise PreparationError(f"manifest closure constants incomplete: {set(result)}")
    expected = {
        "ALLOWED_SOURCE_SUFFIXES": set(EXPECTED_ALLOWED_SOURCE_SUFFIXES),
        "PRE_DATA_STATIC_RELATIVE_PATHS": set(EXPECTED_PRE_DATA_STATIC_PATHS),
        "PRE_DATA_REPOSITORY_RELATIVE_PATHS": set(EXPECTED_REPOSITORY_PATHS),
        "PRE_DATA_AUDIT_INPUTS": set(EXPECTED_PRE_DATA_AUDIT_INPUTS),
        "GENERATED_AUDIT_NAMES": set(EXPECTED_GENERATED_AUDIT_PATHS),
        "GENERATED_ATTEMPT_PRODUCTS": set(EXPECTED_GENERATED_ATTEMPT_PRODUCTS),
        "GENERATED_ROLES": set(EXPECTED_GENERATED_ROLES),
        "GENERATED_REGIMES": set(EXPECTED_GENERATED_REGIMES),
        "MUTABLE_STUDY_ROOT_RELATIVE_PATHS": set(
            EXPECTED_MUTABLE_REPOSITORY_PATHS
        ),
    }
    if result != expected:
        drift = sorted(name for name in wanted if result.get(name) != expected[name])
        raise PreparationError(f"manifest closure policy drift: {drift}")
    return result


def _is_cache_or_near_miss(relative: str) -> bool:
    for part in Path(relative).parts:
        lowered = part.casefold()
        if "cache" in lowered or lowered.endswith((".pyc", ".pyo")):
            return True
        if lowered.endswith(("~", ".bak", ".orig", ".rej", ".swp", ".swo", ".tmp", ".temp")):
            return True
        if any(f"{suffix}." in lowered for suffix in EXPECTED_ALLOWED_SOURCE_SUFFIXES):
            return True
    return False


def _audit_complete_sealed_scope(study: Path) -> dict[str, Any]:
    linked: list[str] = []
    invalid: list[str] = []
    cache_or_near_miss: list[str] = []
    multi_link: list[str] = []
    aliases: list[tuple[str, str]] = []
    seen: dict[tuple[int, int], str] = {}
    for directory, directory_names, file_names in os.walk(
        study, topdown=True, followlinks=False
    ):
        base = Path(directory)
        retained: list[str] = []
        for name in directory_names:
            path = base / name
            relative = path.relative_to(study).as_posix()
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                invalid.append(relative)
                continue
            if stat.S_ISLNK(metadata.st_mode):
                linked.append(relative)
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                invalid.append(relative)
                continue
            if _is_cache_or_near_miss(relative):
                cache_or_near_miss.append(relative)
                continue
            retained.append(name)
        directory_names[:] = retained
        for name in file_names:
            path = base / name
            relative = path.relative_to(study).as_posix()
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                invalid.append(relative)
                continue
            if stat.S_ISLNK(metadata.st_mode):
                linked.append(relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                invalid.append(relative)
                continue
            if _is_cache_or_near_miss(relative):
                cache_or_near_miss.append(relative)
            if metadata.st_nlink != 1:
                multi_link.append(relative)
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            if identity in seen:
                aliases.append((seen[identity], relative))
            else:
                seen[identity] = relative
    if linked or invalid or cache_or_near_miss or multi_link or aliases:
        raise PreparationError(
            "complete sealed-scope filesystem audit failed: "
            f"linked={sorted(linked)}, invalid={sorted(invalid)}, "
            f"cache_or_near_miss={sorted(cache_or_near_miss)}, "
            f"multi_link={sorted(multi_link)}, inode_aliases={sorted(aliases)}"
        )
    return {
        "complete_scope_live_walk": True,
        "linked": [],
        "invalid_entries": [],
        "cache_or_near_miss": [],
        "multi_link_files": [],
        "inode_aliases": [],
    }


def _audit_study_root_namespace(
    study: Path,
    repo: Path,
    repository_paths: set[str],
    mutable_paths: set[str],
) -> dict[str, Any]:
    def local_names(raw_paths: set[str]) -> set[str]:
        names: set[str] = set()
        for raw in raw_paths:
            path = _resolve_relative(repo, raw)
            if path.parent != study:
                raise PreparationError(f"study-root policy path escaped root: {raw}")
            names.add(path.name)
        return names

    required = local_names(repository_paths)
    mutable = local_names(mutable_paths)
    operational = set(EXPECTED_STUDY_ROOT_OPERATIONAL_NAMES)
    directories = set(EXPECTED_STUDY_ROOT_DIRECTORY_NAMES)
    actual = {path.name for path in study.iterdir()}
    expected = required | mutable | operational | directories
    unexpected = sorted(actual - expected)
    missing = sorted((required | {"STATE.json", "RESEARCH_LEDGER.jsonl"} | directories) - actual)
    transaction_residue = "STATE_TRANSACTION.json" in actual
    attempts = study / "attempts"
    attempt_names = {path.name for path in attempts.iterdir()} if attempts.is_dir() else set()
    expected_attempts = {
        SCIENCE_ATTEMPT,
        "v002",
        FIT_SOURCE_ATTEMPT,
        INTERMEDIATE_ATTEMPT,
        SOURCE_PARENT_ATTEMPT,
        SELECTION_SOURCE_ATTEMPT,
        SOURCE_ATTEMPT,
        TARGET_ATTEMPT,
    }
    if unexpected or missing or transaction_residue or attempt_names != expected_attempts:
        raise PreparationError(
            "study-root namespace closure failed: "
            f"missing={missing}, unexpected={unexpected}, "
            f"transaction_residue={transaction_residue}, attempts={sorted(attempt_names)}"
        )
    return {
        "declared_mutable_repository_paths": sorted(mutable_paths),
        "declared_operational_repository_names": sorted(operational),
        "declared_study_root_directories": sorted(directories),
        "study_root_namespace_complete": True,
        "study_root_unexpected": [],
        "state_transaction_present": False,
    }


def _is_generated_data_json(
    relative: str,
    roles: set[str],
    regimes: set[str],
    episode_ids: Mapping[tuple[str, str], set[str]],
) -> bool:
    parts = Path(relative).parts
    if Path(relative).suffix.lower() != ".json" or not parts or parts[0] != "data":
        return False
    if parts == ("data", "replacement_registry.json"):
        return True
    if len(parts) == 3 and parts[:2] == ("data", "replacement_claims"):
        stem = Path(parts[2]).stem
        return (
            Path(parts[2]).suffix == ".json"
            and len(stem) == 6
            and all(character in "0123456789" for character in stem)
        )
    if len(parts) == 4 and parts[1] in roles and parts[2] in regimes:
        return parts[3] in {"raw_manifest.json", "execution_manifest.json"}
    if len(parts) == 5 and parts[1] in roles and parts[2] in regimes:
        return (
            parts[3] in {"raw", "execution"}
            and Path(parts[4]).stem in episode_ids.get((parts[1], parts[2]), set())
        )
    if len(parts) == 5 and parts[:2] == ("data", "persistence_intents"):
        return (
            parts[2] in roles
            and parts[3] in regimes
            and Path(parts[4]).stem in episode_ids.get((parts[2], parts[3]), set())
        )
    return False


def _is_generated_attempt_product(
    relative: str,
    constants: Mapping[str, set[str]],
    episode_ids: Mapping[tuple[str, str], set[str]],
) -> bool:
    execution_failures = {
        f"audit/{role}_{regime}_execution_failure.json"
        for role in constants["GENERATED_ROLES"]
        for regime in constants["GENERATED_REGIMES"]
    }
    return (
        relative in constants["GENERATED_AUDIT_NAMES"]
        or relative in constants["GENERATED_ATTEMPT_PRODUCTS"]
        or relative in execution_failures
        or _is_generated_data_json(
            relative,
            constants["GENERATED_ROLES"],
            constants["GENERATED_REGIMES"],
            episode_ids,
        )
    )


def _prospective_episode_ids(target: Path) -> dict[tuple[str, str], set[str]]:
    ledger = _read_object(target / "cohort_seed_ledger.json")
    regimes = ledger.get("regimes")
    if not isinstance(regimes, Mapping):
        raise PreparationError("cohort seed ledger regimes are absent")
    result: dict[tuple[str, str], set[str]] = {}
    for regime in EXPECTED_GENERATED_REGIMES:
        regime_value = regimes.get(regime)
        roles = regime_value.get("roles") if isinstance(regime_value, Mapping) else None
        if not isinstance(roles, Mapping):
            raise PreparationError(f"cohort seed ledger roles absent: {regime}")
        for role in EXPECTED_GENERATED_ROLES:
            role_value = roles.get(role)
            if not isinstance(role_value, Mapping):
                raise PreparationError(f"cohort seed ledger role absent: {role}/{regime}")
            records = [
                *(role_value.get("primary") or ()),
                *(role_value.get("replacements") or ()),
            ]
            identifiers = {
                str(record.get("episode_id"))
                for record in records
                if isinstance(record, Mapping)
            }
            if len(identifiers) != len(records) or not identifiers:
                raise PreparationError(f"cohort seed ledger episode IDs invalid: {role}/{regime}")
            result[(role, regime)] = identifiers
    return result


def _fit_inventory(target: Path, repo: Path) -> dict[str, Any]:
    """Independently rehash the closed inherited-fit inventory."""

    path = target / FIT_INVENTORY_RELATIVE
    value = _read_object(path)
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
    if set(value) != expected_keys:
        raise PreparationError("inherited fit inventory schema drift")
    created = value.get("created_unix_ns")
    header = {
        "schema": type(value.get("schema_version")) is int
        and value.get("schema_version") == 1,
        "artifact": value.get("artifact_type")
        == "immutable_fit_role_inheritance_inventory",
        "attempt": value.get("attempt") == TARGET_ATTEMPT,
        "source": value.get("source_attempt") == FIT_SOURCE_ATTEMPT,
        "state": value.get("source_state") == "FIT_COHORTS",
        "created": type(created) is int and created > 0,
        "episodes": type(value.get("episode_count")) is int
        and value.get("episode_count") == EXPECTED_FIT_EPISODE_COUNT,
        "per_regime": type(value.get("episodes_per_regime")) is int
        and value.get("episodes_per_regime") == 300,
        "regimes": value.get("regime_order")
        == [
            "native_plan", "markov_oracle", "plan_action_noise_0p2",
            "plan_random_action_0p1",
        ],
        "file_count": type(value.get("file_count")) is int
        and value.get("file_count") == EXPECTED_FIT_FILE_COUNT,
        "state_hash": value.get("source_controller_state_sha256")
        == EXPECTED_FIT_SOURCE_STATE_SHA256,
        "ledger_hash": value.get("source_research_ledger_sha256")
        == EXPECTED_FIT_SOURCE_LEDGER_SHA256,
        "controller_count": type(
            value.get("controller_fit_outcome_episodes_at_inventory")
        ) is int
        and value.get("controller_fit_outcome_episodes_at_inventory") == 0,
        "later_roles": type(value.get("later_role_artifact_file_count")) is int
        and value.get("later_role_artifact_file_count") == 0,
        "replacements": type(value.get("replacement_record_count")) is int
        and value.get("replacement_record_count") == 0,
        "separation": value.get("dataset_role_separation_verified") is True,
        "rehashed": value.get("source_bytes_rehashed") is True,
        "unopened": value.get("outcome_arrays_opened") is False,
        "passed": value.get("passed") is True,
    }
    if not all(header.values()):
        raise PreparationError(f"inherited fit inventory header drift: {header}")
    canonical = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.read_bytes() != canonical:
        raise PreparationError("inherited fit inventory is not canonical JSON")

    records = value.get("files")
    if not isinstance(records, list) or len(records) != EXPECTED_FIT_FILE_COUNT:
        raise PreparationError("inherited fit inventory file census drift")
    source_prefix = _repo_relative(
        target.parents[0] / FIT_SOURCE_ATTEMPT, repo
    ) + "/"
    allowed_prefixes = (
        source_prefix + "data/fit/",
        source_prefix + "data/persistence_intents/fit/",
    )
    allowed_singleton = source_prefix + "data/replacement_registry.json"
    observed_paths: list[str] = []
    observed_inodes: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise PreparationError("inherited fit inventory file-record schema drift")
        raw = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (
            not isinstance(raw, str)
            or not (raw == allowed_singleton or raw.startswith(allowed_prefixes))
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise PreparationError("inherited fit inventory file-record value drift")
        candidate = _resolve_relative(repo, raw)
        metadata = candidate.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != size
            or sha256_file(candidate) != digest
        ):
            raise PreparationError(f"inherited fit file provenance drift: {raw}")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in observed_inodes:
            raise PreparationError("inherited fit inventory contains inode aliases")
        observed_inodes.add(identity)
        observed_paths.append(raw)
    if observed_paths != sorted(observed_paths) or len(set(observed_paths)) != len(
        observed_paths
    ):
        raise PreparationError("inherited fit inventory path ordering drift")
    if _canonical_json_sha(records) != value.get("files_canonical_sha256"):
        raise PreparationError("inherited fit inventory canonical digest drift")

    # Reconstruct the complete inherited role namespace from directory entries;
    # listed-record rehashing alone cannot detect an unrecorded scientific file.
    source_root = target.parent / FIT_SOURCE_ATTEMPT
    namespace_roots = (
        source_root / "data/fit",
        source_root / "data/persistence_intents/fit",
    )
    namespace_paths: list[str] = []
    for namespace_root in namespace_roots:
        root_metadata = namespace_root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise PreparationError("inherited fit namespace root drift")
        for candidate in namespace_root.rglob("*"):
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise PreparationError("inherited fit namespace contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PreparationError("inherited fit namespace entry type drift")
            namespace_paths.append(_repo_relative(candidate, repo))
    registry_path = source_root / "data/replacement_registry.json"
    registry_metadata = registry_path.lstat()
    if (
        stat.S_ISLNK(registry_metadata.st_mode)
        or not stat.S_ISREG(registry_metadata.st_mode)
        or registry_metadata.st_nlink != 1
    ):
        raise PreparationError("inherited replacement registry identity drift")
    namespace_paths.append(_repo_relative(registry_path, repo))
    if sorted(namespace_paths) != observed_paths:
        raise PreparationError("inherited fit namespace file census drift")

    links = {
        "source_invalidity": EXPECTED_FIT_SOURCE_INVALIDITY_SHA256,
        "source_pre_data_seal": EXPECTED_FIT_SOURCE_SEAL_SHA256,
        "source_version_forward_receipt": EXPECTED_FIT_SOURCE_RECEIPT_SHA256,
        "source_cohort_seed_ledger": EXPECTED_COHORT_LEDGER_SHA256,
        "source_replacement_registry": (
            "7aa18ee9ab56e45a3f3aac8ff95ba75978debb2e72368e5ee9e78a1f27c2b871"
        ),
    }
    for key, expected_hash in links.items():
        record = value.get(key)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "bytes", "sha256"}
            or record.get("sha256") != expected_hash
        ):
            raise PreparationError(f"inherited fit inventory lineage link drift: {key}")
        linked = _resolve_relative(repo, str(record.get("path")))
        if (
            type(record.get("bytes")) is not int
            or linked.stat().st_size != record.get("bytes")
            or sha256_file(linked) != expected_hash
        ):
            raise PreparationError(f"inherited fit inventory linked file drift: {key}")
    return {
        "path": _repo_relative(path, repo),
        "sha256": sha256_file(path),
        "file_count": EXPECTED_FIT_FILE_COUNT,
        "episode_count": EXPECTED_FIT_EPISODE_COUNT,
        "source_paths": observed_paths,
        "outcome_arrays_opened": False,
        "passed": True,
    }


def _audit_source_attempt_namespace(
    source: Path,
    repo: Path,
    source_seal: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> int:
    complete = source_seal.get("complete_source_partition")
    source_static = complete.get("target_paths") if isinstance(complete, Mapping) else None
    if (
        not isinstance(source_static, list)
        or set(source_static) != STATIC_PARTITION_PATHS
        or len(source_static) != len(STATIC_PARTITION_PATHS)
    ):
        raise PreparationError("v007 source seal static closure is absent or drifted")
    expected = set(source_static) | set(SOURCE_SELECTION_BOUNDARY_PRODUCTS) | {
        SOURCE_SEAL_RELATIVE,
        SOURCE_RECEIPT_RELATIVE,
        INVALIDITY_DRAFT_RELATIVE,
        INVALIDITY_RELATIVE,
        FIT_INVENTORY_RELATIVE,
    }
    observed = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    observed_directories = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_dir()
    }
    if observed != expected or observed_directories != expected_directories:
        raise PreparationError(
            "v007 complete namespace drift: "
            f"missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}, "
            f"missing_directories={sorted(expected_directories-observed_directories)}, "
            f"unexpected_directories={sorted(observed_directories-expected_directories)}"
        )
    return len(observed)


def _audit_science_attempt_namespace(
    study: Path,
    repo: Path,
    source_seal: Mapping[str, Any],
) -> int:
    """Require the preserved v001 attempt to remain its exact 53-file tree."""

    science = study / "attempts" / SCIENCE_ATTEMPT
    current = source_seal
    visited: set[str] = set()
    seal_path: Path | None = None
    invalidity_path: Path | None = None
    while True:
        tip = current.get("source_pre_data_seal")
        prior_invalidity = current.get("invalidity_evidence")
        if not isinstance(tip, Mapping) or not isinstance(prior_invalidity, Mapping):
            raise PreparationError("transitive seal lost its source lineage")
        candidate = _resolve_relative(repo, str(tip.get("path")))
        if sha256_file(candidate) != tip.get("sha256"):
            raise PreparationError("transitive lineage seal hash drift")
        raw = _repo_relative(candidate, repo)
        if raw in visited:
            raise PreparationError("transitive lineage seal cycle")
        visited.add(raw)
        if candidate == science / "audit/pre_data_seal.json":
            seal_path = candidate
            invalidity_path = _resolve_relative(
                repo, str(prior_invalidity.get("path"))
            )
            break
        current = _read_object(candidate)
    if seal_path is None or invalidity_path is None:
        raise PreparationError("v001 lineage tip is absent")
    if (
        seal_path != science / "audit/pre_data_seal.json"
        or invalidity_path != science / "audit/v001_procedural_invalidity.json"
        or sha256_file(seal_path) != tip.get("sha256")
        or sha256_file(invalidity_path) != prior_invalidity.get("sha256")
    ):
        raise PreparationError("v001 seal/invalidity lineage hash drift")
    science_seal = _read_object(seal_path)
    sealed = science_seal.get("sealed_files")
    if not isinstance(sealed, Mapping):
        raise PreparationError("v001 seal has no sealed-files map")
    prefix = _repo_relative(science, repo) + "/"
    expected = {
        str(raw)[len(prefix):]
        for raw in sealed
        if isinstance(raw, str) and raw.startswith(prefix)
    } | {"audit/pre_data_seal.json", "audit/v001_procedural_invalidity.json"}
    if len(expected) != 53:
        raise PreparationError(f"v001 expected namespace count drift: {len(expected)}")
    observed = {
        path.relative_to(science).as_posix()
        for path in science.rglob("*")
        if path.is_file()
    }
    expected_directories = {
        parent.as_posix()
        for relative in expected
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    observed_directories = {
        path.relative_to(science).as_posix()
        for path in science.rglob("*")
        if path.is_dir()
    }
    if observed != expected or observed_directories != expected_directories:
        raise PreparationError(
            "v001 complete namespace drift: "
            f"missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}"
        )
    for raw, expected_hash in sealed.items():
        path = _resolve_relative(repo, str(raw))
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise PreparationError(f"v001 sealed-file hash drift: {raw}")
    return len(observed)


def _manifest_closure(
    source: Path,
    target: Path,
    repo: Path,
    source_seal: Mapping[str, Any],
    *,
    allow_inheritance_output: bool = False,
) -> dict[str, Any]:
    constants = _manifest_constants(target / "build_manifest.py")
    inventory = _fit_inventory(target, repo)
    static = constants["PRE_DATA_STATIC_RELATIVE_PATHS"]
    audit_inputs = constants["PRE_DATA_AUDIT_INPUTS"]
    generated = constants["GENERATED_AUDIT_NAMES"]
    repository_paths = constants["PRE_DATA_REPOSITORY_RELATIVE_PATHS"]
    if OUTPUT_RELATIVE not in generated:
        raise PreparationError("inheritance seal is not an exact generated-audit exclusion")
    declared = static | audit_inputs
    scope_audit = _audit_complete_sealed_scope(target.parents[1])
    root_audit = _audit_study_root_namespace(
        target.parents[1],
        repo,
        repository_paths,
        constants["MUTABLE_STUDY_ROOT_RELATIVE_PATHS"],
    )
    source_count = _audit_source_attempt_namespace(
        source, repo, source_seal, inventory
    )
    science_count = _audit_science_attempt_namespace(
        target.parents[1], repo, source_seal
    )
    episode_ids = _prospective_episode_ids(target)
    discovered: set[str] = set()
    generated_present: list[str] = []
    for path in target.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        if relative in declared:
            discovered.add(relative)
        elif _is_generated_attempt_product(relative, constants, episode_ids):
            generated_present.append(relative)
        else:
            raise PreparationError(f"unclassified v008 path: {relative}")
    permitted_generated = {FIT_INVENTORY_RELATIVE}
    if allow_inheritance_output:
        permitted_generated.add(OUTPUT_RELATIVE)
    if set(generated_present) - permitted_generated:
        raise PreparationError(
            "pre-forward generated artifact present outside exact authorization: "
            f"{sorted(set(generated_present)-permitted_generated)}"
        )
    if FIT_INVENTORY_RELATIVE not in generated_present:
        raise PreparationError("inherited fit inventory is absent from live closure")
    if allow_inheritance_output and OUTPUT_RELATIVE not in generated_present:
        raise PreparationError("existing inheritance seal is absent from live closure")
    expected_directories = {
        parent.as_posix()
        for relative in declared | set(generated_present)
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    observed_directories = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise PreparationError(
            "v008 directory namespace drift: "
            f"missing={sorted(expected_directories-observed_directories)}, "
            f"unexpected={sorted(observed_directories-expected_directories)}"
        )
    missing = sorted(declared - discovered)
    unexpected = sorted(discovered - declared)
    if missing or unexpected:
        raise PreparationError(
            f"v008 source closure failed: missing={missing}, unexpected={unexpected}"
        )
    missing_repository = sorted(
        raw for raw in repository_paths if not _resolve_relative(repo, raw).is_file()
    )
    if repository_paths != set(EXPECTED_REPOSITORY_PATHS) or missing_repository:
        raise PreparationError(
            "v008 repository source closure is not exact: "
            f"declared={sorted(repository_paths)}, missing={missing_repository}"
        )
    return {
        "declared_static_paths": sorted(static),
        "declared_audit_inputs": sorted(audit_inputs),
        "declared_generated_audits": sorted(generated),
        "declared_generated_attempt_products": sorted(
            constants["GENERATED_ATTEMPT_PRODUCTS"]
        ),
        "declared_generated_roles": sorted(constants["GENERATED_ROLES"]),
        "declared_generated_regimes": sorted(constants["GENERATED_REGIMES"]),
        "allowed_source_suffixes": sorted(constants["ALLOWED_SOURCE_SUFFIXES"]),
        "declared_repository_paths": sorted(repository_paths),
        "source_attempt_allowed_suffix_path_count": source_count,
        "inherited_fit_inventory": {
            key: inventory[key]
            for key in (
                "path", "sha256", "file_count", "episode_count",
                "outcome_arrays_opened", "passed",
            )
        },
        "science_attempt_exact_path_count": science_count,
        "discovered_paths": sorted(discovered),
        "missing": [],
        "unexpected": [],
        **scope_audit,
        **root_audit,
        "passed": True,
    }


class _AttemptNormalizer(ast.NodeTransformer):
    """Erase only the preregistered active-attempt administrative delta."""

    _bindings = {"ATTEMPT", "ACTIVE_ATTEMPT"}

    @staticmethod
    def _binding_names(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Assign):
            return {item.id for item in node.targets if isinstance(item, ast.Name)}
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return {node.target.id}
        return set()

    @staticmethod
    def _guard(node: ast.AST) -> bool:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            return False
        test = node.test
        return (
            isinstance(test.left, ast.Name)
            and test.left.id == "ACTIVE_ATTEMPT"
            and len(test.ops) == 1
            and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value in {SOURCE_ATTEMPT, TARGET_ATTEMPT}
            and bool(node.body)
            and all(isinstance(item, ast.Raise) for item in node.body)
            and not node.orelse
        )

    def visit_Module(self, node: ast.Module) -> ast.Module:  # noqa: N802
        body: list[ast.stmt] = []
        for item in node.body:
            names = self._binding_names(item)
            if names & self._bindings:
                continue
            if self._guard(item):
                continue
            if isinstance(item, ast.Assign) and names in (
                {"NORMATIVE_SOURCES"},
                {"REQUIRED_IMPLEMENTATION_FILES"},
            ):
                item = self._normalize_normative_sources(item)
            transformed = self.visit(item)
            if transformed is not None:
                body.append(transformed)
        node.body = body
        return node

    @staticmethod
    def _normalize_normative_sources(node: ast.Assign) -> ast.Assign:
        collection = node.value
        if isinstance(collection, ast.Call) and collection.args:
            collection = collection.args[0]
        if not isinstance(collection, (ast.List, ast.Tuple, ast.Set)):
            return node
        collection.elts = [
            item
            for item in collection.elts
            if not (
                isinstance(item, ast.Constant)
                and isinstance(item.value, str)
                and item.value in NEW_LINEAGE_SUPPORT_PATHS
            )
        ]
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in self._bindings:
            return ast.copy_location(ast.Constant("<ATTEMPT>"), node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
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
        value = re.sub(r"v00[2345678]", "<ACTIVE_ATTEMPT>", value)
        return ast.copy_location(ast.Constant(value), node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:  # noqa: N802
        visited = self.generic_visit(node)
        pieces: list[str] = []
        for item in visited.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                pieces.append(item.value)
            elif (
                isinstance(item, ast.FormattedValue)
                and item.conversion == -1
                and item.format_spec is None
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            ):
                pieces.append(item.value.value)
            else:
                return visited
        return ast.copy_location(ast.Constant("".join(pieces)), node)


def _validate_attempt_bindings(path: Path, *, target: bool) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0]
            if isinstance(name, ast.Name):
                assignments[name.id] = node.value
    science = assignments.get("SCIENCE_ATTEMPT")
    if science is not None and not (
        isinstance(science, ast.Constant) and science.value == SCIENCE_ATTEMPT
    ):
        raise PreparationError(f"science-attempt binding drift: {path}")
    if target:
        active = assignments.get("ACTIVE_ATTEMPT")
        if active is not None and not (
            isinstance(active, ast.Attribute)
            and isinstance(active.value, ast.Name)
            and active.value.id == "ATTEMPT_ROOT"
            and active.attr == "name"
        ):
            raise PreparationError(f"active-attempt binding is not fail-closed: {path}")


def _normalized_ast(path: Path, *, target: bool) -> tuple[str, str]:
    _validate_attempt_bindings(path, target=target)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    normalized = _AttemptNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    dump = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return dump, _sha256_bytes(dump.encode("utf-8"))


def _normalize_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    attempt = value.get("attempt")
    lineage_input = value.get("lineage")
    if not isinstance(lineage_input, Mapping):
        raise PreparationError("verifier contract lineage is absent")
    common = {
        "science_attempt": SCIENCE_ATTEMPT,
        "invalidity_evidence": (
            "runs/lewm_domain_robust_gate/attempts/v001/"
            "audit/v001_procedural_invalidity.json"
        ),
        "source_pre_data_seal": (
            "runs/lewm_domain_robust_gate/attempts/v001/audit/pre_data_seal.json"
        ),
    }
    source_attempts = [
        SCIENCE_ATTEMPT,
        "v002",
        FIT_SOURCE_ATTEMPT,
        INTERMEDIATE_ATTEMPT,
        SOURCE_PARENT_ATTEMPT,
        SELECTION_SOURCE_ATTEMPT,
        SOURCE_ATTEMPT,
    ]
    source_roots = {
        item: f"runs/lewm_domain_robust_gate/attempts/{item}"
        for item in source_attempts
    }
    expected_source = {
        **common,
        "active_attempt": SOURCE_ATTEMPT,
        "attempts": source_attempts,
        "source_roots": source_roots,
        "pre_data_inheritance_seal": (
            "runs/lewm_domain_robust_gate/attempts/v007/"
            "audit/pre_data_inheritance_seal.json"
        ),
        "prior_pre_data_inheritance_seal": (
            "runs/lewm_domain_robust_gate/attempts/v006/"
            "audit/pre_data_inheritance_seal.json"
        ),
        "source_invalidity_evidence": (
            "runs/lewm_domain_robust_gate/attempts/v006/"
            "audit/v006_procedural_invalidity.json"
        ),
        "superseded_source_invalidity_draft": (
            "runs/lewm_domain_robust_gate/attempts/v006/"
            "audit/v006_procedural_invalidity_draft.json"
        ),
        "prior_version_forward_receipt": (
            "runs/lewm_domain_robust_gate/attempts/v006/"
            "audit/version_forward_transaction_receipt.json"
        ),
        "version_forward_receipt": (
            "runs/lewm_domain_robust_gate/attempts/v007/"
            "audit/version_forward_transaction_receipt.json"
        ),
    }
    expected_target = {
        **common,
        "active_attempt": TARGET_ATTEMPT,
        "attempts": [*source_attempts, TARGET_ATTEMPT],
        "source_roots": {
            **source_roots,
            TARGET_ATTEMPT: "runs/lewm_domain_robust_gate/attempts/v008",
        },
        "pre_data_inheritance_seal": (
            "runs/lewm_domain_robust_gate/attempts/v008/"
            "audit/pre_data_inheritance_seal.json"
        ),
        "prior_pre_data_inheritance_seal": (
            "runs/lewm_domain_robust_gate/attempts/v007/"
            "audit/pre_data_inheritance_seal.json"
        ),
        "source_invalidity_evidence": (
            "runs/lewm_domain_robust_gate/attempts/v007/"
            "audit/v007_procedural_invalidity.json"
        ),
        "superseded_source_invalidity_draft": (
            "runs/lewm_domain_robust_gate/attempts/v007/"
            "audit/v007_procedural_invalidity_draft.json"
        ),
        "prior_version_forward_receipt": (
            "runs/lewm_domain_robust_gate/attempts/v007/"
            "audit/version_forward_transaction_receipt.json"
        ),
        "version_forward_receipt": (
            "runs/lewm_domain_robust_gate/attempts/v008/"
            "audit/version_forward_transaction_receipt.json"
        ),
    }
    expected = expected_source if attempt == SOURCE_ATTEMPT else expected_target
    if dict(lineage_input) != expected:
        raise PreparationError(
            f"verifier contract transitive lineage drift: {attempt}"
        )

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str):
            result = item.replace("/attempts/v008/", "/attempts/v007/")
            return result.replace("/attempts/v008", "/attempts/v007")
        return item

    normalized = visit(dict(value))
    normalized["attempt"] = SOURCE_ATTEMPT
    normalized["attempt_root"] = "runs/lewm_domain_robust_gate/attempts/v007"
    if isinstance(normalized.get("lineage"), dict):
        normalized["lineage"] = expected_source
    return normalized


def _top_level_symbols(path: Path) -> dict[str, str]:
    """Hash every module-body node under an occurrence-stable ordered label."""

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
        if label in result:
            raise PreparationError(f"duplicate top-level unit label: {path}: {label}")
        result[label] = _sha256_bytes(
            ast.dump(
                node, annotate_fields=True, include_attributes=False
            ).encode()
        )
    if len(result) != len(tree.body):
        raise PreparationError(f"top-level unit census incomplete: {path}")
    return result


def _procedural_scope(source: Path, target: Path, relative: str) -> dict[str, Any]:
    left = _top_level_symbols(source)
    right = _top_level_symbols(target)
    changed = sorted(
        name for name in set(left) | set(right) if left.get(name) != right.get(name)
    )
    allowed = PROCEDURAL_ALLOWED_SYMBOLS[relative]
    expected = PROCEDURAL_EXPECTED_CHANGED_SYMBOLS[relative]
    if "*" in allowed or not set(changed).issubset(allowed):
        raise PreparationError(
            f"procedural AST scope escaped allowlist for {relative}: {changed}"
        )
    if set(changed) != expected:
        raise PreparationError(
            f"procedural required changed-symbol set drift for {relative}: "
            f"expected={sorted(expected)}, observed={changed}"
        )
    source_lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    target_lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(source_lines, target_lines, fromfile=relative, tofile=relative)
    ).encode("utf-8")
    if not diff:
        raise PreparationError(f"procedural file is unexpectedly byte-exact: {relative}")
    return {
        "changed_top_level_symbols": changed,
        "allowed_top_level_symbols": sorted(allowed),
        "unchanged_top_level_symbol_count": len(
            [name for name in set(left) & set(right) if left[name] == right[name]]
        ),
        "diff_sha256": _sha256_bytes(diff),
    }


def _lineage_support_scope(
    source: Path, target: Path, relative: str
) -> dict[str, Any]:
    policy = LINEAGE_SUPPORT_UNIT_POLICY.get(relative)
    policy_keys = {
        "source_top_level_unit_count",
        "target_top_level_unit_count",
        "source_top_level_unit_sequence_sha256",
        "target_top_level_unit_sequence_sha256",
        "required_changed_top_level_units",
        "allowed_changed_top_level_units",
        "inserted_top_level_units",
        "deleted_top_level_units",
    }
    if not isinstance(policy, dict) or set(policy) != policy_keys:
        raise PreparationError(f"lineage unit policy absent or malformed: {relative}")
    left = _top_level_symbols(source)
    right = _top_level_symbols(target)
    source_sequence_sha256 = _canonical_json_sha(list(left))
    target_sequence_sha256 = _canonical_json_sha(list(right))
    changed = sorted(
        label
        for label in set(left) | set(right)
        if left.get(label) != right.get(label)
    )
    inserted = sorted(set(right) - set(left))
    deleted = sorted(set(left) - set(right))
    required = list(policy["required_changed_top_level_units"])
    allowed = list(policy["allowed_changed_top_level_units"])
    if (
        set(LINEAGE_SUPPORT_UNIT_POLICY) != set(LINEAGE_SUPPORT_PATHS)
        or policy["source_top_level_unit_count"] != len(left)
        or policy["target_top_level_unit_count"] != len(right)
        or policy["source_top_level_unit_sequence_sha256"]
        != source_sequence_sha256
        or policy["target_top_level_unit_sequence_sha256"]
        != target_sequence_sha256
        or required != sorted(required)
        or allowed != sorted(allowed)
        or required != allowed
        or changed != required
        or list(policy["inserted_top_level_units"]) != inserted
        or list(policy["deleted_top_level_units"]) != deleted
        or any("*" in label for label in allowed)
    ):
        raise PreparationError(f"lineage top-level unit policy drift: {relative}")
    allowed_set = set(allowed)
    for label in set(left) & set(right):
        if label not in allowed_set and left[label] != right[label]:
            raise PreparationError(
                f"lineage nonallowed unit changed: {relative}: {label}"
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
        "unchanged_top_level_units_sha256": _canonical_json_sha(unchanged),
    }


def _source_manifest_paths(source: Path, repo: Path) -> tuple[dict[str, Any], set[str]]:
    """Recover v004's sealed static closure without outcome decoding."""

    del repo  # kept in the signature so producer/verifier call sites stay parallel
    source_seal = _read_object(source / SOURCE_SEAL_RELATIVE)
    complete = source_seal.get("complete_source_partition")
    relative = complete.get("target_paths") if isinstance(complete, Mapping) else None
    if (
        not isinstance(relative, list)
        or len(relative) != len(set(relative))
        or set(relative) != STATIC_PARTITION_PATHS - NEW_LINEAGE_SUPPORT_PATHS
    ):
        raise PreparationError("v004 inheritance seal static partition drift")
    return source_seal, set(relative)


def _partition_sources(source: Path, target: Path, repo: Path, closure: Mapping[str, Any]) -> dict[str, Any]:
    _manifest, source_paths = _source_manifest_paths(source, repo)
    target_paths = set(closure["discovered_paths"])
    exact: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    procedural: list[dict[str, Any]] = []
    new_support: list[dict[str, Any]] = []
    lineage_support: list[dict[str, Any]] = []

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
        if len(values) != 1:
            raise PreparationError(f"lineage unit policy literal count drift: {path}")
        try:
            return ast.literal_eval(values[0])
        except (ValueError, TypeError, SyntaxError) as exc:
            raise PreparationError(f"lineage unit policy is not literal: {path}") from exc

    for owner in (
        "version_forward_prepare.py",
        "verify_version_forward.py",
        "independent_verify.py",
    ):
        if literal_policy(target / owner) != LINEAGE_SUPPORT_UNIT_POLICY:
            raise PreparationError(f"lineage unit policy transcription drift: {owner}")
    source_only = sorted(source_paths - target_paths)
    target_only = sorted(target_paths - source_paths)
    if source_only or set(target_only) != set(NEW_LINEAGE_SUPPORT_PATHS):
        raise PreparationError(
            f"attempt source-set delta is not exact: source_only={source_only}, target_only={target_only}"
        )
    if (
        source_paths != STATIC_PARTITION_PATHS - NEW_LINEAGE_SUPPORT_PATHS
        or target_paths != STATIC_PARTITION_PATHS
    ):
        raise PreparationError("v004/v008 static path census drift")
    for relative in sorted(source_paths):
        left = source / relative
        right = target / relative
        left_hash = sha256_file(left)
        right_hash = sha256_file(right)
        source_path = _repo_relative(left, repo)
        target_path = _repo_relative(right, repo)
        if relative in EXACT_HASH_PATHS:
            if left_hash != right_hash:
                raise PreparationError(f"exact-hash object differs: {relative}")
            exact.append(
                {
                    "relative_path": relative,
                    "source_path": source_path,
                    "target_path": target_path,
                    "sha256": left_hash,
                    "bytes": left.stat().st_size,
                    "classification": (
                        "scientific_object_exact"
                        if relative in SCIENTIFIC_OBJECT_PATHS
                        else "administrative_or_test_exact"
                    ),
                }
            )
        elif relative in PROCEDURAL_EXISTING_PATHS:
            if left_hash == right_hash:
                raise PreparationError(
                    f"required procedural module reverted byte-exact: {relative}"
                )
            scope = _procedural_scope(left, right, relative)
            procedural.append(
                {
                    "relative_path": relative,
                    "source_path": source_path,
                    "target_path": target_path,
                    "source_sha256": left_hash,
                    "target_sha256": right_hash,
                    "change_id": "zero_outcome_v008_generator_root_parent_repair_v1",
                    **scope,
                }
            )
        elif relative in NORMALIZED_AST_PATHS:
            if left_hash == right_hash:
                raise PreparationError(f"administrative path unexpectedly byte-exact: {relative}")
            left_dump, left_normalized = _normalized_ast(left, target=False)
            right_dump, right_normalized = _normalized_ast(right, target=True)
            if left_dump != right_dump:
                raise PreparationError(f"normalized AST differs: {relative}")
            normalized.append(
                {
                    "relative_path": relative,
                    "source_path": source_path,
                    "target_path": target_path,
                    "source_sha256": left_hash,
                    "target_sha256": right_hash,
                    "normalized_ast_sha256": left_normalized,
                    "transform_id": "v007_to_v008_active_attempt_ast_v1",
                }
            )
        elif relative in CANONICAL_CONTRACT_PATHS:
            left_value = _read_object(left)
            right_value = _read_object(right)
            left_canonical = _normalize_contract(left_value)
            right_canonical = _normalize_contract(right_value)
            if left_canonical != right_canonical:
                raise PreparationError(f"canonical verifier contract differs: {relative}")
            contracts.append(
                {
                    "relative_path": relative,
                    "source_path": source_path,
                    "target_path": target_path,
                    "source_sha256": left_hash,
                    "target_sha256": right_hash,
                    "canonical_sha256": _canonical_json_sha(left_canonical),
                    "transform_id": "v008_transitive_lineage_contract_v1",
                }
            )
        elif relative in LINEAGE_SUPPORT_PATHS:
            if left_hash == right_hash:
                raise PreparationError(f"v008 lineage support unexpectedly byte-exact: {relative}")
            lineage_support.append(
                {
                    "relative_path": relative,
                    "source_path": source_path,
                    "target_path": target_path,
                    "source_sha256": left_hash,
                    "target_sha256": right_hash,
                    "purpose": "v008_transitive_lineage_authorization_or_test_only",
                    **_lineage_support_scope(left, right, relative),
                }
            )
        else:
            raise PreparationError(f"unallowlisted v004/v008 source difference: {relative}")
    for relative in sorted(NEW_LINEAGE_SUPPORT_PATHS):
        right = target / relative
        new_support.append(
            {
                "relative_path": relative,
                "target_path": _repo_relative(right, repo),
                "target_sha256": sha256_file(right),
                "bytes": right.stat().st_size,
                "purpose": "immutable_v004_fit_role_provenance_and_read_only_path_map",
            }
        )
    members = {
        "exact_hash": [item["relative_path"] for item in exact],
        "normalized_ast": [item["relative_path"] for item in normalized],
        "canonical_contracts": [item["relative_path"] for item in contracts],
        "procedural_only": [item["relative_path"] for item in procedural],
        "new_lineage_support": [item["relative_path"] for item in new_support],
        "lineage_support": [item["relative_path"] for item in lineage_support],
    }
    flattened = [name for values in members.values() for name in values]
    overlap = sorted({name for name in flattened if flattened.count(name) > 1})
    unpartitioned_source = sorted(source_paths - set(flattened))
    unpartitioned_target = sorted(target_paths - set(flattened))
    if overlap or unpartitioned_source or unpartitioned_target:
        raise PreparationError("source partition is not complete and disjoint")
    return {
        "exact_hash": exact,
        "normalized_ast": normalized,
        "canonical_contracts": contracts,
        "procedural_only": procedural,
        "new_lineage_support": new_support,
        "lineage_support": lineage_support,
        "complete_source_partition": {
            "source_paths": sorted(source_paths),
            "target_paths": sorted(target_paths),
            "partition_members": members,
            "overlap": [],
            "unpartitioned_source": [],
            "unpartitioned_target": [],
            "passed": True,
        },
    }


def _verify_ledger(study: Path) -> dict[str, Any]:
    genesis_path = study / "LEDGER_CHAIN_GENESIS.json"
    ledger_path = study / "RESEARCH_LEDGER.jsonl"
    genesis = _read_object(genesis_path)
    raw = ledger_path.read_bytes()
    prefix_bytes = genesis.get("legacy_prefix_bytes")
    legacy_count = genesis.get("legacy_prefix_event_count")
    if type(prefix_bytes) is not int or type(legacy_count) is not int:
        raise PreparationError("ledger genesis count types are invalid")
    if _sha256_bytes(raw[:prefix_bytes]) != genesis.get("legacy_prefix_sha256"):
        raise PreparationError("ledger immutable prefix hash drift")
    lines = raw.splitlines()
    if len(lines) < legacy_count:
        raise PreparationError("ledger is shorter than its legacy prefix")
    previous = f"legacy:{genesis['legacy_prefix_sha256']}"
    for sequence, line in enumerate(lines[legacy_count:], start=legacy_count + 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise PreparationError("ledger record is not an object")
        observed = value.pop("record_sha256", None)
        expected = _canonical_json_sha(value)
        if observed != expected or value.get("seq") != sequence or value.get("prev_sha256") != previous:
            raise PreparationError(f"ledger hash-chain drift at sequence {sequence}")
        previous = str(observed)
    return {
        "event_count": len(lines),
        "head_sha256": previous,
        "ledger_sha256": sha256_file(ledger_path),
        "genesis_sha256": sha256_file(genesis_path),
    }


def _state_and_lineage(
    study: Path, source: Path, repo: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate the exact v006 selection-boundary failure and lineage."""

    state_path = study / "STATE.json"
    ledger_path = study / "RESEARCH_LEDGER.jsonl"
    state = _read_object(state_path)
    chain = _verify_ledger(study)
    counts = {key: state.get(key) for key in OUTCOME_KEYS}
    if (
        state.get("active_attempt") != SOURCE_ATTEMPT
        or state.get("active_attempt_path") != _repo_relative(source, repo)
        or state.get("current_state") != RESUME_STATE
        or state.get("confirmation_terminal") is not False
        or state.get("ledger_event_count") != chain["event_count"]
        or state.get("ledger_head_sha256") != chain["head_sha256"]
        or counts != EXPECTED_OUTCOMES
        or sha256_file(state_path) != EXPECTED_SOURCE_STATE_SHA256
        or chain["ledger_sha256"] != EXPECTED_SOURCE_LEDGER_SHA256
    ):
        raise PreparationError("v007 controller checkpoint/ledger binding drift")

    invalidity_path = source / INVALIDITY_RELATIVE
    draft_path = source / INVALIDITY_DRAFT_RELATIVE
    source_seal_path = source / SOURCE_SEAL_RELATIVE
    source_receipt_path = source / SOURCE_RECEIPT_RELATIVE
    selection_source = source.parent / SELECTION_SOURCE_ATTEMPT
    pre_selection_path = selection_source / SOURCE_PRE_SELECTION_SEAL_RELATIVE
    invalidity = _read_object(invalidity_path)
    draft = _read_object(draft_path)
    source_seal = _read_object(source_seal_path)
    source_receipt = _read_object(source_receipt_path)
    pre_selection = _read_object(pre_selection_path)
    fixed_hashes = {
        invalidity_path: EXPECTED_INVALIDITY_SHA256,
        draft_path: EXPECTED_INVALIDITY_DRAFT_SHA256,
        source_seal_path: EXPECTED_SOURCE_SEAL_SHA256,
        source_receipt_path: EXPECTED_SOURCE_RECEIPT_SHA256,
        pre_selection_path: EXPECTED_SOURCE_PRE_SELECTION_SEAL_SHA256,
    }
    if any(sha256_file(path) != digest for path, digest in fixed_hashes.items()):
        raise PreparationError("v007 immutable lineage record hash drift")

    defect = invalidity.get("defect")
    fit_evidence = invalidity.get("fit_evidence")
    selection_evidence = invalidity.get("selection_evidence")
    frozen = invalidity.get("frozen_evidence")
    if (
        invalidity.get("attempt") != SOURCE_ATTEMPT
        or invalidity.get("procedural_invalidity") is not True
        or invalidity.get("authoritative") is not True
        or invalidity.get("status")
        != "invalid_zero_confirmation_outcome_procedural"
        or invalidity.get("current_state") != RESUME_STATE
        or invalidity.get("scientific_criterion_changed") is not False
        or invalidity.get("scientific_outcomes_conditioned_on") is not False
        or invalidity.get("selection_or_later_outcomes_opened") is not False
        or invalidity.get("immutable_fit_evidence_preserved") is not True
        or invalidity.get("confirmation_artifact_file_count") != 0
        or invalidity.get("controller_outcome_counts") != EXPECTED_OUTCOMES
        or invalidity.get("later_role_artifact_file_counts")
        != {"confirmation": 0, "selection": 0, "smoke": 0}
        or not isinstance(defect, Mapping)
        or defect.get("classification")
        != "independent_verifier_contract_schema_rejects_authenticated_inherited_selection_boundary_paths"
        or defect.get("reproduced_without_outcome_arrays") is not True
        or defect.get("exception_type") != "VerificationError"
        or defect.get("contract_schema_only") is not True
        or "pre_selection_seal" not in str(defect.get("failure_message"))
        or defect.get("workflow_sha256") != sha256_file(source / "workflow.py")
        or defect.get("independent_verifier_sha256")
        != sha256_file(source / "independent_verify.py")
        or defect.get("standalone_version_forward_verifier_sha256")
        != sha256_file(source / "verify_version_forward.py")
        or not isinstance(fit_evidence, Mapping)
        or fit_evidence.get("fit_outcome_episode_count")
        != EXPECTED_FIT_EPISODE_COUNT
        or fit_evidence.get("independent_scientific_replay_completed") is not True
        or fit_evidence.get("selection_outcomes_used_for_fit") is not False
        or fit_evidence.get("immutable_and_reusable_under_authenticated_lineage")
        is not True
        or not isinstance(selection_evidence, Mapping)
        or selection_evidence
        != {
            "controller_selection_outcome_episode_count": 0,
            "generation_output_file_count": 0,
            "seed_tuples_consumed": 0,
            "selection_outcome_arrays_opened": False,
            "worlds_constructed": 0,
        }
    ):
        raise PreparationError("v007 procedural-invalidity classification drift")

    prior_draft = invalidity.get("superseded_draft")
    if (
        not isinstance(prior_draft, Mapping)
        or prior_draft.get("authoritative") is not False
        or prior_draft.get("path") != _repo_relative(draft_path, repo)
        or prior_draft.get("sha256") != EXPECTED_INVALIDITY_DRAFT_SHA256
        or draft.get("attempt") != SOURCE_ATTEMPT
        or draft.get("procedural_invalidity") is not True
        or draft.get("authoritative") is not False
        or draft.get("confirmation_outcome_episodes_generated") != 0
        or draft.get("confirmation_outcome_episodes_executed") != 0
        or draft.get("confirmation_outcomes_opened_for_analysis") is not False
    ):
        raise PreparationError("v007 invalidity draft provenance drift")

    frozen_checks = {
        "controller": isinstance(frozen, Mapping)
        and frozen.get("controller_state_sha256_at_failure")
        == EXPECTED_SOURCE_STATE_SHA256,
        "ledger": isinstance(frozen, Mapping)
        and frozen.get("research_ledger_sha256_at_failure")
        == EXPECTED_SOURCE_LEDGER_SHA256,
        "genesis": isinstance(frozen, Mapping)
        and frozen.get("ledger_chain_genesis_sha256") == chain["genesis_sha256"],
        "seal": isinstance(frozen, Mapping)
        and frozen.get("pre_data_inheritance_seal_sha256")
        == EXPECTED_SOURCE_SEAL_SHA256,
        "pre_selection": isinstance(frozen, Mapping)
        and frozen.get("pre_selection_seal_sha256")
        == EXPECTED_SOURCE_PRE_SELECTION_SEAL_SHA256,
        "receipt": isinstance(frozen, Mapping)
        and frozen.get("version_forward_receipt_sha256")
        == EXPECTED_SOURCE_RECEIPT_SHA256,
        "program": isinstance(frozen, Mapping)
        and frozen.get("root_program_sha256") == sha256_file(study / "program.py"),
        "cohort": sha256_file(source / "cohort_seed_ledger.json")
        == EXPECTED_COHORT_LEDGER_SHA256,
        "generator": sha256_file(source / "generator.py")
        == EXPECTED_SOURCE_GENERATOR_SHA256,
        "launcher": sha256_file(source / "launcher.py")
        == EXPECTED_SOURCE_LAUNCHER_SHA256,
    }
    if not all(frozen_checks.values()):
        raise PreparationError(f"v007 frozen checkpoint drift: {frozen_checks}")

    inherited_fit = source_seal.get("inherited_fit_role")
    if (
        source_seal.get("attempt") != SOURCE_ATTEMPT
        or source_seal.get("source_attempt") != SELECTION_SOURCE_ATTEMPT
        or source_seal.get("target_attempt") != SOURCE_ATTEMPT
        or source_seal.get("science_attempt") != SCIENCE_ATTEMPT
        or source_seal.get("resume_state") != "SELECTION_COHORTS"
        or source_seal.get("passed") is not True
        or source_seal.get("outcome_counts_at_seal")
        != SOURCE_ACTIVATION_OUTCOMES
        or not isinstance(inherited_fit, Mapping)
        or inherited_fit.get("episode_count") != EXPECTED_FIT_EPISODE_COUNT
        or inherited_fit.get("file_count") != EXPECTED_FIT_FILE_COUNT
        or inherited_fit.get("inventory_sha256")
        != EXPECTED_SOURCE_FIT_INVENTORY_SHA256
        or source_receipt.get("source_attempt") != SELECTION_SOURCE_ATTEMPT
        or source_receipt.get("target_attempt") != SOURCE_ATTEMPT
        or source_receipt.get("resume_state") != "SELECTION_COHORTS"
        or source_receipt.get("passed") is not True
    ):
        raise PreparationError("v007 source seal/receipt header drift")

    receipt_seal = source_receipt.get("seal")
    post_snapshot = source_receipt.get("post_snapshot")
    post_state = post_snapshot.get("state") if isinstance(post_snapshot, Mapping) else None
    post_ledger = post_snapshot.get("ledger") if isinstance(post_snapshot, Mapping) else None
    transaction_source = source_receipt.get("transaction_source")
    activation_state = post_state.get("object") if isinstance(post_state, Mapping) else None
    if (
        not isinstance(receipt_seal, Mapping)
        or receipt_seal.get("sha256") != EXPECTED_SOURCE_SEAL_SHA256
        or not isinstance(transaction_source, Mapping)
        or transaction_source.get("sha256") != EXPECTED_SOURCE_ADAPTER_SHA256
        or not isinstance(activation_state, Mapping)
        or activation_state.get("active_attempt") != SOURCE_ATTEMPT
        or activation_state.get("current_state") != "SELECTION_COHORTS"
        or {key: activation_state.get(key) for key in OUTCOME_KEYS}
        != SOURCE_ACTIVATION_OUTCOMES
        or post_state.get("sha256") != EXPECTED_SOURCE_PRE_FORWARD_STATE_SHA256
        or post_state.get("object_sha256") != _canonical_json_sha(activation_state)
        or source_receipt.get("controller_return") != activation_state
        or source_receipt.get("controller_return_sha256")
        != _canonical_json_sha(activation_state)
        or not isinstance(post_ledger, Mapping)
        or post_ledger.get("sha256") != EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256
        or post_ledger.get("ledger_sha256")
        != EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256
        or type(post_ledger.get("bytes")) is not int
    ):
        raise PreparationError("v007 activation receipt binding drift")
    current_ledger = ledger_path.read_bytes()
    prefix = current_ledger[: int(post_ledger["bytes"])]
    if (
        len(prefix) != int(post_ledger["bytes"])
        or _sha256_bytes(prefix) != EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256
        or chain["event_count"] != int(post_ledger.get("event_count", -1))
    ):
        raise PreparationError("v007 ledger-prefix drift")

    histories = state.get("attempt_history")
    lineage = state.get("version_forward_lineage")
    prior_lineage = source_seal.get("prior_version_forward_lineage")
    expected_versions = [
        SCIENCE_ATTEMPT,
        "v002",
        FIT_SOURCE_ATTEMPT,
        INTERMEDIATE_ATTEMPT,
        SOURCE_PARENT_ATTEMPT,
        SELECTION_SOURCE_ATTEMPT,
        SOURCE_ATTEMPT,
    ]
    if (
        not isinstance(histories, list)
        or [item.get("version") for item in histories if isinstance(item, Mapping)]
        != expected_versions
        or not isinstance(lineage, list)
        or len(lineage) != 6
        or not isinstance(prior_lineage, list)
        or lineage[:-1] != prior_lineage
        or lineage[-1].get("old_attempt") != SELECTION_SOURCE_ATTEMPT
        or lineage[-1].get("new_attempt") != SOURCE_ATTEMPT
        or lineage[-1].get("equivalence_sha256")
        != EXPECTED_SOURCE_SEAL_SHA256
        or lineage[-1].get("resume_state") != "SELECTION_COHORTS"
    ):
        raise PreparationError("v001-through-v007 controller lineage drift")

    marker = state.get("v007_durable_controller_adapter")
    if (
        not isinstance(marker, Mapping)
        or marker.get("authorization_kind")
        != "receipt_bound_v007_root_controller_adapter"
        or marker.get("adapter_source_sha256") != EXPECTED_SOURCE_ADAPTER_SHA256
        or marker.get("ledger_event_count") != chain["event_count"]
        or marker.get("ledger_head_sha256") != chain["head_sha256"]
    ):
        raise PreparationError("live v007 durable adapter marker drift")
    marker_core = dict(marker)
    binding = marker_core.pop("state_binding_sha256", None)
    state_without_marker = json.loads(json.dumps(state))
    state_without_marker.pop("v007_durable_controller_adapter", None)
    if binding != _canonical_json_sha(
        {
            "marker_without_state_binding": marker_core,
            "state_without_marker": state_without_marker,
        }
    ):
        raise PreparationError("live v007 durable adapter state binding drift")

    completed = state.get("completed_states")
    checkpoints_raw = state.get("verified_checkpoints")
    expected_completed = [
        "BOOTSTRAP_AUDIT",
        "DIAGNOSTIC_ACCOUNT",
        "PREREGISTRATION_AND_POWER",
        "IMPLEMENTATION_COMPLETE",
        "PRESEAL_QUALIFICATION",
        "PRE_OUTCOME_SEAL",
        "FIT_COHORTS",
        "FIT_LOCK",
        "PRE_SELECTION_SEAL",
    ]
    if (
        completed != expected_completed
        or not isinstance(checkpoints_raw, list)
        or len(checkpoints_raw) != len(expected_completed)
    ):
        raise PreparationError("v007 inherited checkpoint chronology is incomplete")
    checkpoints: list[dict[str, Any]] = []
    for index, item in enumerate(checkpoints_raw):
        expected_source = (
            SCIENCE_ATTEMPT if index < 6 else SELECTION_SOURCE_ATTEMPT
        )
        if (
            not isinstance(item, dict)
            or item.get("source_attempt") != expected_source
        ):
            raise PreparationError("v007 inherited checkpoint record drift")
        path = _resolve_relative(repo, str(item.get("evidence_path")))
        if not path.is_file() or sha256_file(path) != item.get("evidence_sha256"):
            raise PreparationError(f"inherited checkpoint hash drift: {path}")
        checkpoints.append(
            {
                "checkpoint_name": item.get("name"),
                "source_attempt": expected_source,
                "evidence_path": _repo_relative(path, repo),
                "evidence_sha256": item.get("evidence_sha256"),
            }
        )

    pre_counts = pre_selection.get("outcome_counts_at_seal")
    if (
        pre_selection.get("attempt") != SELECTION_SOURCE_ATTEMPT
        or pre_selection.get("checkpoint_state") != "PRE_SELECTION_SEAL"
        or pre_selection.get("passed") is not True
        or pre_selection.get("pre_data_seal_sha256")
        != "8f5c24d0903317bacbd3722fdd13abc8697a4ca98d1f6e9456541d620609b499"
        or pre_selection.get("selection_arrays_opened_by_sealer") != 0
        or pre_selection.get("selected_head_refit_permitted") is not False
        or pre_counts != EXPECTED_OUTCOMES
        or state.get("last_verified_checkpoint", {}).get("evidence_path")
        != _repo_relative(pre_selection_path, repo)
        or state.get("last_verified_checkpoint", {}).get("evidence_sha256")
        != EXPECTED_SOURCE_PRE_SELECTION_SEAL_SHA256
    ):
        raise PreparationError("v006 pre-selection checkpoint binding drift")
    sealed = pre_selection.get("sealed_files")
    expected_fit_files = {
        _repo_relative(selection_source / "fit/fitted_candidates.npz", repo):
        sha256_file(selection_source / "fit/fitted_candidates.npz"),
        _repo_relative(selection_source / "fit/fit_lock.json", repo):
        sha256_file(selection_source / "fit/fit_lock.json"),
    }
    if (
        not isinstance(sealed, Mapping)
        or any(sealed.get(path) != digest for path, digest in expected_fit_files.items())
    ):
        raise PreparationError("v006 pre-selection seal lost fitted-candidate bytes")

    return state, checkpoints, invalidity, source_seal


def _verify_source_sealed_files(source_seal: Mapping[str, Any], repo: Path) -> dict[str, str]:
    files = source_seal.get("sealed_files")
    if not isinstance(files, dict) or not files:
        raise PreparationError("v001 source pre-data seal has no sealed_files")
    result: dict[str, str] = {}
    for raw, expected in sorted(files.items()):
        if not isinstance(raw, str) or not isinstance(expected, str):
            raise PreparationError("v001 sealed_files record type drift")
        path = _resolve_relative(repo, raw)
        if not path.is_file() or sha256_file(path) != expected:
            raise PreparationError(f"v001 sealed file hash drift: {raw}")
        result[raw] = expected
    return result


def _scientific_records(partitions: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_path: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for mode in ("exact_hash", "normalized_ast", "procedural_only"):
        for item in partitions[mode]:
            by_path[str(item["relative_path"])] = (mode, item)
    missing = sorted(SCIENTIFIC_OBJECT_PATHS - set(by_path))
    if missing:
        raise PreparationError(f"scientific-object paths are unpartitioned: {missing}")
    records: list[dict[str, Any]] = []
    for relative in sorted(SCIENTIFIC_OBJECT_PATHS):
        mode, item = by_path[relative]
        record = {
            "relative_path": relative,
            "source_path": item["source_path"],
            "target_path": item["target_path"],
            "equivalence_mode": mode,
            "source_sha256": item.get("source_sha256", item.get("sha256")),
            "target_sha256": item.get("target_sha256", item.get("sha256")),
            "classification": (
                "seed_identifier_or_configuration"
                if Path(relative).suffix in {".json", ".md"}
                else "numerical_or_endpoint_source"
            ),
        }
        if mode == "normalized_ast":
            record["normalized_ast_sha256"] = item["normalized_ast_sha256"]
        records.append(record)
    return records


def build_inheritance_seal(
    study_root: Path,
    *,
    _existing_created_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Derive, but do not write, the complete immutable authorization."""

    study = Path(study_root).resolve(strict=True)
    repo, source, target = _attempt_roots(study)
    output = target / OUTPUT_RELATIVE
    existing = os.path.lexists(output)
    if existing and _existing_created_unix_ns is None:
        raise FileExistsError(f"immutable inheritance seal already exists: {output}")
    if not existing and _existing_created_unix_ns is not None:
        raise FileNotFoundError(f"inheritance seal disappeared during verification: {output}")
    state, checkpoints, invalidity, source_seal = _state_and_lineage(study, source, repo)
    closure = _manifest_closure(
        source,
        target,
        repo,
        source_seal,
        allow_inheritance_output=existing,
    )
    partitions = _partition_sources(source, target, repo, closure)
    source_sealed = _verify_source_sealed_files(source_seal, repo)
    selection_source = source.parent / SELECTION_SOURCE_ATTEMPT
    source_pre_selection_path = (
        selection_source / SOURCE_PRE_SELECTION_SEAL_RELATIVE
    )
    source_pre_selection = _read_object(source_pre_selection_path)
    source_checkpoint_sealed = _verify_source_sealed_files(
        source_pre_selection, repo
    )

    confirmation_files = sorted(
        _repo_relative(path, repo)
        for root in (source / "data/confirmation", target / "data/confirmation")
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if confirmation_files:
        raise PreparationError(f"confirmation artifacts exist: {confirmation_files}")

    invalidity_path = source / INVALIDITY_RELATIVE
    invalidity_draft_path = source / INVALIDITY_DRAFT_RELATIVE
    source_seal_path = source / SOURCE_SEAL_RELATIVE
    source_receipt_path = source / SOURCE_RECEIPT_RELATIVE
    root_paths = {
        "program.py": study / "program.py",
        "README.md": study / "README.md",
        "LEDGER_CHAIN_GENESIS.json": study / "LEDGER_CHAIN_GENESIS.json",
    }
    root_controls: dict[str, Any] = {}
    for name, path in root_paths.items():
        raw = _repo_relative(path, repo)
        observed = sha256_file(path)
        if name == "program.py" and source_sealed.get(raw) != observed:
            raise PreparationError("root program.py is not bound by the v001 source seal")
        if name == "LEDGER_CHAIN_GENESIS.json" and invalidity["frozen_evidence"].get(
            "ledger_chain_genesis_sha256"
        ) != observed:
            raise PreparationError("ledger genesis is not bound by the invalidity evidence")
        root_controls[name] = {
            "path": raw,
            "sha256": observed,
            "bytes": path.stat().st_size,
            "source_seal_bound": source_sealed.get(raw) == observed,
        }

    target_sealed: dict[str, str] = dict(source_sealed)
    for relative in closure["discovered_paths"]:
        path = target / relative
        target_sealed[_repo_relative(path, repo)] = sha256_file(path)
    for raw in closure["declared_repository_paths"]:
        path = _resolve_relative(repo, raw)
        target_sealed[raw] = sha256_file(path)
    target_sealed[_repo_relative(invalidity_path, repo)] = sha256_file(invalidity_path)
    target_sealed[_repo_relative(invalidity_draft_path, repo)] = sha256_file(
        invalidity_draft_path
    )
    target_sealed[_repo_relative(source_seal_path, repo)] = sha256_file(source_seal_path)
    target_sealed[_repo_relative(source_receipt_path, repo)] = sha256_file(
        source_receipt_path
    )
    target_sealed[_repo_relative(source_pre_selection_path, repo)] = sha256_file(
        source_pre_selection_path
    )
    fit_inventory_path = target / FIT_INVENTORY_RELATIVE
    fit_inventory_value = _read_object(fit_inventory_path)
    for record in fit_inventory_value["files"]:
        target_sealed[str(record["path"])] = str(record["sha256"])
    fit_source = study / "attempts" / FIT_SOURCE_ATTEMPT
    for relative in SOURCE_OPERATIONAL_LOCKS:
        lock_path = fit_source / relative
        if lock_path.stat().st_size != 0:
            raise PreparationError(f"source operational lock is nonempty: {relative}")
        target_sealed[_repo_relative(lock_path, repo)] = sha256_file(lock_path)
    target_sealed[_repo_relative(fit_inventory_path, repo)] = sha256_file(
        fit_inventory_path
    )
    if len(target_sealed) != EXPECTED_TRANSITIVE_SEALED_FILE_COUNT:
        raise PreparationError(
            f"transitive sealed-file union is not exact: {len(target_sealed)}"
        )

    exact_files = partitions["exact_hash"]
    scientific = _scientific_records(partitions)
    counts = {key: state[key] for key in OUTCOME_KEYS}
    checks = {
        "expected_boundary_outcome_counts": counts == EXPECTED_OUTCOMES,
        "no_confirmation_artifacts": not confirmation_files,
        "invalidity_authenticated": True,
        "source_pre_data_seal_authenticated": True,
        "source_pre_selection_seal_authenticated": True,
        "all_inherited_checkpoints_authenticated": bool(checkpoints),
        "source_sealed_files_rehashed": bool(source_sealed),
        "complete_source_partition": partitions["complete_source_partition"]["passed"],
        "v008_manifest_closure": closure["passed"],
        "root_controls_authenticated": True,
        "scientific_objects_equivalent": len(scientific) == len(SCIENTIFIC_OBJECT_PATHS),
        "contracts_canonically_equivalent": len(partitions["canonical_contracts"])
        == len(CANONICAL_CONTRACT_PATHS),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "artifact_type": "pre_data_inheritance_and_version_forward_equivalence",
        "authorization_kind": AUTHORIZATION_KIND,
        "attempt": TARGET_ATTEMPT,
        "source_attempt": SOURCE_ATTEMPT,
        "science_attempt": SCIENCE_ATTEMPT,
        "target_attempt": TARGET_ATTEMPT,
        "checkpoint_state": "PRE_OUTCOME_SEAL",
        "resume_state": RESUME_STATE,
        "created_unix_ns": (
            time.time_ns()
            if _existing_created_unix_ns is None
            else _existing_created_unix_ns
        ),
        "procedural_invalidity_confirmed": True,
        "zero_confirmation_outcomes_at_version_forward": True,
        "invalidity_evidence": {
            "path": _repo_relative(invalidity_path, repo),
            "sha256": sha256_file(invalidity_path),
        },
        "superseded_invalidity_draft": {
            "path": _repo_relative(invalidity_draft_path, repo),
            "sha256": sha256_file(invalidity_draft_path),
            "authoritative": False,
        },
        "source_pre_data_seal": {
            "path": _repo_relative(source_seal_path, repo),
            "sha256": sha256_file(source_seal_path),
        },
        "source_pre_selection_seal": {
            "path": _repo_relative(source_pre_selection_path, repo),
            "sha256": sha256_file(source_pre_selection_path),
        },
        "source_version_forward_transaction_receipt": {
            "path": _repo_relative(source_receipt_path, repo),
            "sha256": sha256_file(source_receipt_path),
        },
        "inherited_fit_role": {
            "source_attempt": FIT_SOURCE_ATTEMPT,
            "inventory_path": closure["inherited_fit_inventory"]["path"],
            "inventory_sha256": closure["inherited_fit_inventory"]["sha256"],
            "file_count": closure["inherited_fit_inventory"]["file_count"],
            "episode_count": closure["inherited_fit_inventory"]["episode_count"],
            "outcome_arrays_opened": False,
            "passed": True,
        },
        "prior_version_forward_lineage": json.loads(
            json.dumps(state["version_forward_lineage"])
        ),
        "source_controller_adapter_marker": json.loads(
            json.dumps(state["v007_durable_controller_adapter"])
        ),
        "outcome_counts_at_seal": counts,
        "confirmation_artifact_scan": {"file_count": 0, "paths": []},
        "inherited_verified_checkpoints": checkpoints,
        "root_controls": root_controls,
        "controller_snapshot": {
            "state_path": _repo_relative(study / "STATE.json", repo),
            "state_sha256": sha256_file(study / "STATE.json"),
            "ledger_path": _repo_relative(study / "RESEARCH_LEDGER.jsonl", repo),
            **_verify_ledger(study),
        },
        "source_sealed_files": {
            "file_count": len(source_sealed),
            "files": dict(sorted(source_sealed.items())),
            "all_rehashed": True,
        },
        "source_pre_selection_sealed_files": {
            "file_count": len(source_checkpoint_sealed),
            "files": dict(sorted(source_checkpoint_sealed.items())),
            "all_rehashed": True,
        },
        "sealed_files": dict(sorted(target_sealed.items())),
        "source_partitions": {
            "exact_hash": exact_files,
            "normalized_ast": partitions["normalized_ast"],
            "canonical_contracts": partitions["canonical_contracts"],
            "procedural_only": partitions["procedural_only"],
            "new_lineage_support": partitions["new_lineage_support"],
            "lineage_support": partitions["lineage_support"],
        },
        "equivalent_files": exact_files,
        "scientific_objects": scientific,
        "complete_source_partition": partitions["complete_source_partition"],
        "v008_manifest_closure": closure,
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
        "checks": checks,
        "outcome_arrays_opened": False,
        "hdf5_contents_opened": False,
        "v3_targets_opened": False,
        "passed": passed,
    }


def verify_existing_inheritance_seal(
    seal_path: Path,
    study_root: Path,
) -> dict[str, Any]:
    study = Path(study_root).resolve(strict=True)
    expected_path = study / "attempts" / TARGET_ATTEMPT / OUTPUT_RELATIVE
    if Path(seal_path).absolute() != expected_path.absolute():
        raise PreparationError("inheritance seal path is not exact")
    observed = _read_object(expected_path)
    if (
        type(observed.get("schema_version")) is not int
        or observed.get("schema_version") != 1
    ):
        raise PreparationError("inheritance seal schema type is invalid")
    created = observed.get("created_unix_ns")
    if type(created) is not int or created <= 0:
        raise PreparationError("inheritance seal creation timestamp is invalid")
    expected = build_inheritance_seal(
        study,
        _existing_created_unix_ns=created,
    )
    expected_bytes = (
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        expected_path.read_bytes() != expected_bytes
        or _canonical_json_sha(observed) != _canonical_json_sha(expected)
    ):
        raise PreparationError("existing inheritance seal differs from producer recomputation")
    return {
        "passed": True,
        "phase": "pre-forward",
        "attempt": TARGET_ATTEMPT,
        "seal_path": _repo_relative(expected_path, study.parents[1]),
        "seal_sha256": sha256_file(expected_path),
        "state_sha256": sha256_file(study / "STATE.json"),
        "ledger_sha256": sha256_file(study / "RESEARCH_LEDGER.jsonl"),
        "outcome_arrays_opened": False,
        "read_only": True,
    }


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Durably create an immutable JSON artifact without replacement."""

    output = Path(path)
    if os.path.lexists(output):
        raise FileExistsError(f"immutable inheritance seal already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "write"))
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    arguments = parser.parse_args(argv)
    value = build_inheritance_seal(arguments.study_root)
    output = (
        arguments.study_root.resolve()
        / "attempts"
        / TARGET_ATTEMPT
        / OUTPUT_RELATIVE
    )
    if arguments.command == "write":
        write_exclusive(output, value)
        result = {
            "passed": True,
            "output": str(output),
            "sha256": sha256_file(output),
            "sealed_file_count": len(value["sealed_files"]),
        }
    else:
        result = {
            "passed": value["passed"],
            "would_write": str(output),
            "artifact_created": False,
            "sealed_file_count": len(value["sealed_files"]),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
