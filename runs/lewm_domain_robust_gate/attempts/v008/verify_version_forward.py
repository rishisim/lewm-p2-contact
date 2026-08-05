#!/usr/bin/env python3
"""Independent stdlib verifier for the v007 -> v008 inheritance seal.

No code is imported from ``version_forward_prepare.py`` or from either attempt.
All file sets, transformations, hashes, controller chronology, and post-forward
links are recomputed here from bytes.
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
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True


SCIENCE_ATTEMPT = "v001"
INTERMEDIATE = "v004"
SOURCE_PARENT = "v005"
SELECTION_SOURCE = "v006"
SOURCE = "v007"
TARGET = "v008"
FIT_SOURCE = "v003"
RESUME = "SELECTION_COHORTS"
OUTPUT = "audit/pre_data_inheritance_seal.json"
INVALIDITY = "audit/v007_procedural_invalidity.json"
INVALIDITY_DRAFT = "audit/v007_procedural_invalidity_draft.json"
SOURCE_SEAL = "audit/pre_data_inheritance_seal.json"
SOURCE_RECEIPT = "audit/version_forward_transaction_receipt.json"
SOURCE_PRE_SELECTION_SEAL = "audit/pre_selection_seal.json"
FIT_INVENTORY = "audit/inherited_fit_inventory.json"
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
EXPECTED_SELECTION_SOURCE_ADAPTER_SHA256 = "c73b9c44037898c943614c7c4eb20efae04e0969f6a12f75ae1d0982e6985998"
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
AUTH_KIND = "zero_confirmation_outcome_version_forward_inherited_pre_selection"
EXPECTED_COUNTS = {
    "fit_outcome_episodes": 1200,
    "selection_outcome_episodes": 0,
    "smoke_outcome_episodes": 0,
    "confirmation_outcome_episodes_generated": 0,
    "confirmation_outcome_episodes_executed": 0,
    "confirmation_outcomes_opened_for_analysis": False,
}
SOURCE_ACTIVATION_COUNTS = {**EXPECTED_COUNTS}
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
NORMALIZED = frozenset(
    {"power_analysis.py"}
)
CONTRACTS = frozenset(
    {"verifier_contract.json", "verifier_contract_no_candidate.json", "verifier_contract_power_infeasible.json"}
)
PROCEDURAL = frozenset(
    {"analysis.py", "build_manifest.py", "capture_verifier.py", "checkpoints.py", "compile_gate.py", "generator.py", "independent_verify.py", "latency.py", "launcher.py", "preseal.py", "runner.py", "study_common.py", "terminal_workflow.py", "tests/test_analysis.py", "tests/test_checkpoint_hardening.py", "tests/test_generator_contract.py", "tests/test_independent_verify.py", "tests/test_preseal.py", "tests/test_runner_gate.py", "tests/test_seed_power.py", "tests/test_terminal_workflow.py", "tests/test_workflow.py", "workflow.py"}
)
NEW_SUPPORT = frozenset(
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
SCIENCE = frozenset(
    {
        "DGP_MATRIX.json", "DIAGNOSTIC_ACCOUNT.md", "PREREGISTRATION.md", "analysis.py",
        "audit/bootstrap_audit.json", "audit/diagnostic_account.json",
        "audit/identifier_freshness_verification.json", "audit/preregistration_and_power.json",
        "candidate_grid.json", "cohort_seed_ledger.json", "compile_gate.py", "counted_features.py",
        "fit_select.py", "flops.py", "input_loader.py", "outcome_mapping.json",
        "power_analysis.py", "power_baseline.json", "power_rule.json",
    }
)
PROCEDURAL_ALLOWED_SYMBOLS: dict[str, frozenset[str]] = {
    "analysis.py": frozenset(
        {"ACTIVE_ATTEMPT", "ATTEMPT", "_fsync_parent_directory", "_publish_temporary",
         "atomic_json", "atomic_npz", "capture_post_open_analysis_failure",
         "run_sealed_analysis", "validate_sealed_analysis_authorization"}
    ),
    "build_manifest.py": frozenset(
        {
            "<docstring>", "ACTIVE_ATTEMPT", "EXPECTED_ACTIVE_ATTEMPT", "SCIENCE_ATTEMPT",
            "PRE_DATA_STATIC_RELATIVE_PATHS", "PRE_DATA_REPOSITORY_RELATIVE_PATHS", "GENERATED_AUDIT_NAMES",
            "GENERATED_ATTEMPT_PRODUCTS", "GENERATED_ROLES", "GENERATED_REGIMES",
            "GENERATED_EXECUTION_FAILURE_NAMES", "MUTABLE_STUDY_ROOT_RELATIVE_PATHS", "PRE_DATA_LABEL",
            "_study_root", "_allowed_suffix", "_walk_attempt_candidates", "_walk_study_root_candidates",
            "_assert_discovered_integrity", "_is_generated_data_json",
            "classify_pre_data_source_paths", "collect_pre_data_source_paths", "is_generated_attempt_product",
            "source_closure_policy", "discovered_python_source_paths",
            "unexpected_pre_data_source_paths", "unexpected_python_source_paths",
            "build_pre_data_manifest", "verify_manifest", "_manifest_for_paths",
        }
    ),
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
    "compile_gate.py": frozenset(
        {"_atomic_json", "_atomic_npz", "_fsync_parent_directory", "_publish_exclusive_temporary"}
    ),
    "generator.py": frozenset(
        {"<imports>", "ATTEMPT", "EPISODE_ID_PATTERN", "PRE_DATA_SEAL_PATH",
         "REGIME_SLUGS", "ROLE_SLUGS", "_append_jsonl", "_assert_safe_directory_chain",
         "_assert_safe_output_leaf", "_canonical_output_file", "_expected_episode_id",
         "_failure_descriptor", "_failure_genesis", "_failure_hash", "_generate_locked",
         "_lexical_attempt_relative", "_strict_failure_records",
         "_validate_destination_episode_id", "_validate_episode_id", "_validate_seed_record",
         "episode_paths", "generate", "load_generation_inputs", "materialization_lock",
         "materialization_lock_path", "preflight_output_namespace", "raw_directory",
         "read_rollout_failures", "record_rollout_failure", "validate_dgp_matrix",
         "validate_existing_manifest", "validate_raw_manifest_contract", "validate_seed_ledger",
         "verify_authorization_seal", "verify_episode_record"}
    ),
    "independent_verify.py": frozenset(
        {
            "<imports>", "ACTIVE_ATTEMPT", "ALLOWED_ATTEMPTS", "ATTEMPT", "ATTEMPT_ROOTS",
            "CANONICAL_CONTRACT_PATHS", "INHERITANCE_SEAL_RELATIVE", "INHERITED_COMPLETED_STATES",
            "INVALIDITY_RELATIVE", "NEW_LINEAGE_SUPPORT_PATHS", "NORMALIZED_AST_PATHS",
            "PROCEDURAL_ALLOWED_SYMBOLS", "PROCEDURAL_EXPECTED_CHANGED_SYMBOLS",
            "PROCEDURAL_EXISTING_PATHS", "SCIENTIFIC_OBJECT_PATHS",
            "SOURCE_PRE_DATA_RELATIVE", "STUDY_RELATIVE", "VERSION_FORWARD_RESUME_STATE",
            "_AdministrativeAstNormalizer", "_assignment_names", "_canonicalize_contract",
            "_is_active_attempt_guard", "_lineage_file_link", "_normalize_normative_sources",
            "_source_manifest_relative_paths", "_top_level_symbol_hashes", "canonical_contract_sha256",
            "load_contract", "normalized_administrative_ast_sha256", "verify",
            "verify_analysis_execution_invalid_branch", "verify_confirmation_analysis",
            "verify_equivalence_partitions", "verify_ledger_and_state", "verify_role_manifests",
            "verify_seal", "verify_version_forward_lineage",
        }
    ),
    "latency.py": frozenset(
        {"ACTIVE_ATTEMPT", "_fsync_parent_directory", "_publish_temporary", "atomic_json",
         "run_latency_suite", "validate_latency_result"}
    ),
    "preseal.py": frozenset({
        "<imports>", "ACTIVE_ATTEMPT", "PRE_DATA_SEAL_PATH", "REQUIRED_IMPLEMENTATION_FILES",
        "SCIENCE_ATTEMPT", "_files_under", "_state", "_verified_checkpoint",
        "implementation_complete", "preseal_qualification", "seal_pre_confirmation",
        "seal_pre_data", "seal_pre_selection", "validate_power_input_coherence",
        "validate_seed_freshness_evidence", "validate_verifier_contracts",
    }),
    "runner.py": frozenset({"<imports>", "ATTEMPT_VERSION", "_assert_regular_unlinked",
        "_enforce_execution_runtime", "_execute_confirmation_locked", "_execute_development_locked",
        "_execution_record_if_valid", "_execution_source_bindings", "_load_raw_manifest",
        "_preflight_execution_namespace", "_state_authorization",
        "_validate_existing_execution_manifest", "execute_confirmation", "execute_development"}),
    "study_common.py": frozenset(
        {"_fsync_parent_directory", "_publish_temporary", "atomic_json", "atomic_npz"}
    ),
    "tests/test_independent_verify.py": frozenset(
        {
            "_policy", "test_analysis_execution_invalid_branch_is_preload_and_binds_result_presence",
            "test_power_provenance_is_local_identity_bound_and_tamper_evident",
            "test_v002_equivalence_partitions_are_independently_recomputed_and_tamper_evident",
            "test_v002_equivalence_partitions_reject_omitted_required_symbol",
            "test_v002_equivalence_partitions_reject_wildcard_procedural_scope",
            "test_v002_lineage_normalization_and_contract_canonicalization_are_tamper_evident",
            "test_verifier_imports_only_stdlib_and_numpy_and_failure_stdout_is_one_json_line",
            "test_zero_confirmation_version_forward_requires_explicit_equivalence",
        }
    ),
    "tests/test_analysis.py": frozenset({"_authorization_fixture", "_latency_binding_and_result", "test_post_open_failure_capture_is_outcome_free_and_never_overwrites"}),
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
    "tests/test_preseal.py": frozenset(
        {
            "<docstring>", "test_compile_and_power_are_state_gated_inside_exact_eval_worker",
            "test_power_input_paths_resolve_from_repository_not_cwd",
            "test_power_provenance_binds_rule_locks_freeze_and_selected_identity",
            "test_pre_data_manifest_ignores_later_stage_top_level_products",
            "test_pre_data_manifest_rejects_unlisted_python_source",
            "test_preseal_state_uses_verified_controller_not_direct_state_read",
            "test_python_cache_hygiene_fails_closed",
            "test_stage_file_collection_rejects_supplied_directory_symlink_before_rglob",
        }
    ),
    "tests/test_runner_gate.py": frozenset({"test_complete_confirmation_part_is_idempotently_resumed", "test_execution_handlers_do_not_capture_process_interruptions"}),
    "tests/test_seed_power.py": frozenset({"_claim_summary", "test_power_cli_binds_gate_locks_and_persists_only_repo_relative_paths"}),
    "tests/test_terminal_workflow.py": frozenset(
        {"_analysis", "_attempt", "_captured_audit", "_contract", "_state",
         "test_early_decision_has_zero_later_roles_and_reports_are_claim_bounded",
         "test_report_artifact_mutation_is_not_repaired",
         "test_result_present_analysis_integrity_failure_is_manifested_but_not_called"}
    ),
    "tests/test_workflow.py": frozenset(
        {"<docstring>", "FakeController", "_configure_power_paths", "_controller_json",
         "_install_fit_count_crash", "_ledger_record", "_power_result",
         "_prior_selection_audit", "_synthetic_role_manifests",
         "test_count_crash_recovery_rejects_state_ledger_audit_and_manifest_tamper",
         "test_default_freeze_writes_identity_summaries_only_after_gate_freeze",
         "test_exact_role_count_is_resume_idempotent",
         "test_exact_role_count_without_authenticated_crash_record_fails_closed",
         "test_fit_count_commit_crash_reauthenticates_manifests_and_advances_once",
         "test_inherited_early_stop_adapter_rejects_unsealed_active_contract",
         "test_loaded_workflow_controller_adapts_both_inherited_early_stop_contracts",
         "test_sealed_analysis_integrity_failure_is_staged_not_advanced"}
    ),
    "workflow.py": frozenset(
        {"<imports>", "ATTEMPT", "DEFAULT_EVIDENCE_PATHS", "LEDGER_PATH",
         "ROLE_COUNTER_FIELDS", "ROLE_EXECUTION_SEALS", "ROLE_RAW_SEALS", "ROLE_SEALS",
         "Workflow", "_common_role_authorization_state_sha256",
         "_controller_state_object_sha256", "_load_controller",
         "_require_inherited_presealed_verifier_contract", "_verify_role_count_update_recovery",
         "_verify_role_regime", "verify_role_manifests"}
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

# Independently transcribed live v006 -> v008 recovery partition policy.
EXACT = frozenset(
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
NORMALIZED = frozenset(
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
CONTRACTS = frozenset(
    {
        'verifier_contract.json',
        'verifier_contract_no_candidate.json',
        'verifier_contract_power_infeasible.json',
    }
)
PROCEDURAL = frozenset()
PROCEDURAL_EXPECTED_CHANGED_SYMBOLS: dict[str, frozenset[str]] = {}
PROCEDURAL_ALLOWED_SYMBOLS = dict(PROCEDURAL_EXPECTED_CHANGED_SYMBOLS)
NEW_SUPPORT = frozenset()
LINEAGE_SUPPORT = frozenset(
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
STATIC = frozenset().union(
    EXACT, NORMALIZED, CONTRACTS, PROCEDURAL, NEW_SUPPORT, LINEAGE_SUPPORT
)


class VerificationError(RuntimeError):
    """An independently recomputed inheritance invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"unreadable JSON object: {path}") from exc
    _require(isinstance(value, dict), f"non-object JSON: {path}")
    return value


def _json_hash(value: Any) -> str:
    return _hash_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _exact_typed_mapping(value: Any, expected: Mapping[str, Any]) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(expected)
        and all(
            type(value.get(key)) is type(item) and value.get(key) == item
            for key, item in expected.items()
        )
    )


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


def _roots(study_root: Path) -> tuple[Path, Path, Path, Path]:
    lexical = Path(study_root).absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        _require(
            not (
                os.path.lexists(current)
                and stat.S_ISLNK(current.lstat().st_mode)
            ),
            f"study/attempt ancestor symlink forbidden: {current}",
        )
    study = lexical.resolve(strict=True)
    repo = study.parents[1]
    science = study / "attempts" / SCIENCE_ATTEMPT
    source = study / "attempts" / SOURCE
    target = study / "attempts" / TARGET
    _require(
        science.is_dir() and source.is_dir() and target.is_dir(),
        "v001/v005/v008 attempt roots absent",
    )
    return study, repo, source, target


def _relative(path: Path, repo: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(repo.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise VerificationError(f"path outside repository: {path}") from exc


def _resolve(repo: Path, raw: str) -> Path:
    rel = Path(raw)
    _require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe path: {raw}")
    result = (repo / rel).resolve(strict=False)
    _require(result.is_relative_to(repo.resolve(strict=True)), f"path escape: {raw}")
    return result


def _literal_set(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        _require(node.func.id == "frozenset" and len(node.args) == 1 and not node.keywords, "nonliteral closure call")
        return _literal_set(node.args[0])
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise VerificationError("nonliteral manifest closure") from exc
    _require(isinstance(value, (set, frozenset, list, tuple)), "closure is not a collection")
    _require(all(isinstance(item, str) for item in value), "closure has non-string value")
    return set(value)


def _manifest_policy(target: Path) -> dict[str, set[str]]:
    tree = ast.parse((target / "build_manifest.py").read_text(encoding="utf-8"))
    names = {
        "ALLOWED_SOURCE_SUFFIXES", "PRE_DATA_STATIC_RELATIVE_PATHS",
        "PRE_DATA_REPOSITORY_RELATIVE_PATHS", "PRE_DATA_AUDIT_INPUTS",
        "GENERATED_AUDIT_NAMES", "GENERATED_ATTEMPT_PRODUCTS", "GENERATED_ROLES",
        "GENERATED_REGIMES", "MUTABLE_STUDY_ROOT_RELATIVE_PATHS",
    }
    constants: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                constants[name] = _literal_set(node.value)
    _require(set(constants) == names, "manifest closure constants incomplete")
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
    _require(constants == expected, f"manifest closure policy drift: {sorted(name for name in names if constants.get(name) != expected[name])}")
    return constants


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
    linked: list[str] = []; invalid: list[str] = []; cache: list[str] = []
    multi: list[str] = []; aliases: list[tuple[str, str]] = []
    seen: dict[tuple[int, int], str] = {}
    for directory, directory_names, file_names in os.walk(study, topdown=True, followlinks=False):
        base = Path(directory); retained: list[str] = []
        for name in directory_names:
            path = base / name; relative = path.relative_to(study).as_posix()
            try: metadata = path.lstat()
            except FileNotFoundError: invalid.append(relative); continue
            if stat.S_ISLNK(metadata.st_mode): linked.append(relative); continue
            if not stat.S_ISDIR(metadata.st_mode): invalid.append(relative); continue
            if _is_cache_or_near_miss(relative): cache.append(relative); continue
            retained.append(name)
        directory_names[:] = retained
        for name in file_names:
            path = base / name; relative = path.relative_to(study).as_posix()
            try: metadata = path.lstat()
            except FileNotFoundError: invalid.append(relative); continue
            if stat.S_ISLNK(metadata.st_mode): linked.append(relative); continue
            if not stat.S_ISREG(metadata.st_mode): invalid.append(relative); continue
            if _is_cache_or_near_miss(relative): cache.append(relative)
            if metadata.st_nlink != 1: multi.append(relative)
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            if identity in seen: aliases.append((seen[identity], relative))
            else: seen[identity] = relative
    _require(not (linked or invalid or cache or multi or aliases), f"complete sealed-scope filesystem audit failed: linked={sorted(linked)}, invalid={sorted(invalid)}, cache_or_near_miss={sorted(cache)}, multi_link={sorted(multi)}, inode_aliases={sorted(aliases)}")
    return {"complete_scope_live_walk": True, "linked": [], "invalid_entries": [], "cache_or_near_miss": [], "multi_link_files": [], "inode_aliases": []}


def _audit_study_root_namespace(study: Path, repo: Path, repositories: set[str], mutable: set[str]) -> dict[str, Any]:
    def names(raws: set[str]) -> set[str]:
        result: set[str] = set()
        for raw in raws:
            path = _resolve(repo, raw)
            _require(path.parent == study, f"study-root policy path escaped root: {raw}")
            result.add(path.name)
        return result
    required = names(repositories); mutable_names = names(mutable)
    operational = set(EXPECTED_STUDY_ROOT_OPERATIONAL_NAMES); directories = set(EXPECTED_STUDY_ROOT_DIRECTORY_NAMES)
    actual = {path.name for path in study.iterdir()}
    unexpected = sorted(actual - required - mutable_names - operational - directories)
    missing = sorted((required | {"STATE.json", "RESEARCH_LEDGER.jsonl"} | directories) - actual)
    attempts = study / "attempts"
    attempt_names = {path.name for path in attempts.iterdir()} if attempts.is_dir() else set()
    _require(not unexpected and not missing and "STATE_TRANSACTION.json" not in actual and attempt_names == {SCIENCE_ATTEMPT, "v002", FIT_SOURCE, INTERMEDIATE, SOURCE_PARENT, SELECTION_SOURCE, SOURCE, TARGET}, f"study-root namespace closure failed: missing={missing}, unexpected={unexpected}, transaction_residue={'STATE_TRANSACTION.json' in actual}, attempts={sorted(attempt_names)}")
    return {"declared_mutable_repository_paths": sorted(mutable), "declared_operational_repository_names": sorted(operational), "declared_study_root_directories": sorted(directories), "study_root_namespace_complete": True, "study_root_unexpected": [], "state_transaction_present": False}


def _prospective_episode_ids(target: Path) -> dict[tuple[str, str], set[str]]:
    ledger = _object(target / "cohort_seed_ledger.json"); regimes = ledger.get("regimes")
    _require(isinstance(regimes, dict), "cohort seed ledger regimes absent")
    result: dict[tuple[str, str], set[str]] = {}
    for regime in EXPECTED_GENERATED_REGIMES:
        regime_value = regimes.get(regime); roles = regime_value.get("roles") if isinstance(regime_value, dict) else None
        _require(isinstance(roles, dict), f"cohort roles absent: {regime}")
        for role in EXPECTED_GENERATED_ROLES:
            value = roles.get(role); _require(isinstance(value, dict), f"cohort role absent: {role}/{regime}")
            records = [*(value.get("primary") or ()), *(value.get("replacements") or ())]
            identifiers = {str(item.get("episode_id")) for item in records if isinstance(item, dict)}
            _require(bool(identifiers) and len(identifiers) == len(records), f"cohort IDs invalid: {role}/{regime}")
            result[(role, regime)] = identifiers
    return result


def _fit_inventory(target: Path, repo: Path) -> dict[str, Any]:
    """Recompute inherited fit provenance without importing producer code."""

    path = target / FIT_INVENTORY
    value = _object(path)
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
    _require(set(value) == expected_keys, "inherited fit inventory schema drift")
    checks = {
        "schema": type(value.get("schema_version")) is int
        and value.get("schema_version") == 1,
        "artifact": value.get("artifact_type")
        == "immutable_fit_role_inheritance_inventory",
        "attempt": value.get("attempt") == TARGET,
        "source": value.get("source_attempt") == FIT_SOURCE,
        "state": value.get("source_state") == "FIT_COHORTS",
        "created": type(value.get("created_unix_ns")) is int
        and value.get("created_unix_ns") > 0,
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
    _require(all(checks.values()), f"inherited fit inventory header drift: {checks}")
    _require(
        path.read_bytes()
        == (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
        "inherited fit inventory is not canonical JSON",
    )
    records = value.get("files")
    _require(
        isinstance(records, list) and len(records) == EXPECTED_FIT_FILE_COUNT,
        "inherited fit inventory file census drift",
    )
    source_root = target.parent / FIT_SOURCE
    source_prefix = _relative(source_root, repo) + "/"
    allowed_prefixes = (
        source_prefix + "data/fit/",
        source_prefix + "data/persistence_intents/fit/",
    )
    allowed_singleton = source_prefix + "data/replacement_registry.json"
    paths: list[str] = []
    inodes: set[tuple[int, int]] = set()
    for record in records:
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "bytes", "sha256"},
            "inherited fit inventory file-record schema drift",
        )
        raw = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        _require(
            isinstance(raw, str)
            and (raw == allowed_singleton or raw.startswith(allowed_prefixes))
            and type(size) is int
            and size >= 0
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "inherited fit inventory file-record value drift",
        )
        candidate = _resolve(repo, raw)
        metadata = candidate.lstat()
        _require(
            not stat.S_ISLNK(metadata.st_mode)
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and metadata.st_size == size
            and file_hash(candidate) == digest,
            f"inherited fit file provenance drift: {raw}",
        )
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        _require(identity not in inodes, "inherited fit inventory inode alias")
        inodes.add(identity)
        paths.append(raw)
    _require(
        paths == sorted(paths)
        and len(set(paths)) == len(paths)
        and _json_hash(records) == value.get("files_canonical_sha256"),
        "inherited fit inventory ordering or canonical digest drift",
    )
    namespace_paths: list[str] = []
    for namespace_root in (
        source_root / "data/fit",
        source_root / "data/persistence_intents/fit",
    ):
        root_metadata = namespace_root.lstat()
        _require(
            not stat.S_ISLNK(root_metadata.st_mode)
            and stat.S_ISDIR(root_metadata.st_mode),
            "inherited fit namespace root drift",
        )
        for candidate in namespace_root.rglob("*"):
            metadata = candidate.lstat()
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                "inherited fit namespace contains a symlink",
            )
            if stat.S_ISDIR(metadata.st_mode):
                continue
            _require(
                stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
                "inherited fit namespace entry type drift",
            )
            namespace_paths.append(_relative(candidate, repo))
    registry_path = source_root / "data/replacement_registry.json"
    registry_metadata = registry_path.lstat()
    _require(
        not stat.S_ISLNK(registry_metadata.st_mode)
        and stat.S_ISREG(registry_metadata.st_mode)
        and registry_metadata.st_nlink == 1,
        "inherited replacement registry identity drift",
    )
    namespace_paths.append(_relative(registry_path, repo))
    _require(
        sorted(namespace_paths) == paths,
        "inherited fit namespace file census drift",
    )
    links = {
        "source_invalidity": EXPECTED_FIT_SOURCE_INVALIDITY_SHA256,
        "source_pre_data_seal": EXPECTED_FIT_SOURCE_SEAL_SHA256,
        "source_version_forward_receipt": EXPECTED_FIT_SOURCE_RECEIPT_SHA256,
        "source_cohort_seed_ledger": EXPECTED_COHORT_LEDGER_SHA256,
        "source_replacement_registry": (
            "7aa18ee9ab56e45a3f3aac8ff95ba75978debb2e72368e5ee9e78a1f27c2b871"
        ),
    }
    for key, digest in links.items():
        record = value.get(key)
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "bytes", "sha256"}
            and record.get("sha256") == digest,
            f"inherited fit lineage link schema drift: {key}",
        )
        linked = _resolve(repo, str(record.get("path")))
        _require(
            type(record.get("bytes")) is int
            and linked.stat().st_size == record.get("bytes")
            and file_hash(linked) == digest,
            f"inherited fit lineage link drift: {key}",
        )
    return {
        "path": _relative(path, repo),
        "sha256": file_hash(path),
        "file_count": EXPECTED_FIT_FILE_COUNT,
        "episode_count": EXPECTED_FIT_EPISODE_COUNT,
        "source_paths": paths,
        "outcome_arrays_opened": False,
        "passed": True,
    }


def _is_generated_data_json(relative: str, episode_ids: Mapping[tuple[str, str], set[str]]) -> tuple[bool, str | None]:
    parts = Path(relative).parts
    if Path(relative).suffix.lower() != ".json" or not parts or parts[0] != "data": return False, None
    if parts == ("data", "replacement_registry.json"): return True, None
    if len(parts) == 3 and parts[:2] == ("data", "replacement_claims"):
        stem = Path(parts[2]).stem
        return Path(parts[2]).suffix == ".json" and len(stem) == 6 and all(character in "0123456789" for character in stem), None
    if len(parts) == 4 and parts[1] in EXPECTED_GENERATED_ROLES and parts[2] in EXPECTED_GENERATED_REGIMES:
        return parts[3] in {"raw_manifest.json", "execution_manifest.json"}, parts[1]
    if len(parts) == 5 and parts[1] in EXPECTED_GENERATED_ROLES and parts[2] in EXPECTED_GENERATED_REGIMES and parts[3] in {"raw", "execution"}:
        return Path(parts[4]).stem in episode_ids.get((parts[1], parts[2]), set()), parts[1]
    if len(parts) == 5 and parts[:2] == ("data", "persistence_intents") and parts[2] in EXPECTED_GENERATED_ROLES and parts[3] in EXPECTED_GENERATED_REGIMES:
        return Path(parts[4]).stem in episode_ids.get((parts[2], parts[3]), set()), parts[2]
    return False, None


def _is_generated_data_path(relative: str, episode_ids: Mapping[tuple[str, str], set[str]]) -> tuple[bool, str | None]:
    data_json, role = _is_generated_data_json(relative, episode_ids)
    if data_json: return True, role
    parts = Path(relative).parts
    if parts == ("data", "replacement_registry.lock"): return True, None
    if len(parts) == 4 and parts[0] == "data" and parts[1] in EXPECTED_GENERATED_ROLES and parts[2] in EXPECTED_GENERATED_REGIMES and parts[3] == "role.npz": return True, parts[1]
    if len(parts) == 5 and parts[0] == "data" and parts[1] in EXPECTED_GENERATED_ROLES and parts[2] in EXPECTED_GENERATED_REGIMES and parts[3] in {"raw", "execution"} and Path(parts[4]).suffix == ".npz":
        return Path(parts[4]).stem in episode_ids.get((parts[1], parts[2]), set()), parts[1]
    match = re.fullmatch(r"\.materialization-(fit|selection|smoke|confirmation)-(native_plan|markov_oracle|plan_action_noise_0p2|plan_random_action_0p1)\.lock", relative)
    return (True, match.group(1)) if match else (False, None)


def _state_generated_policy(state: Mapping[str, Any], phase: str) -> tuple[set[str], set[str]]:
    base = {OUTPUT, FIT_INVENTORY}
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
    raise VerificationError(f"closure has no generated-object policy for state: {current}")


def _generated_source_product(relative: str, constants: Mapping[str, set[str]], episode_ids: Mapping[tuple[str, str], set[str]], state: Mapping[str, Any], phase: str) -> bool:
    allowed_roles, exact = _state_generated_policy(state, phase)
    if relative in exact: return True
    if phase == "pre-forward": return False
    data, role = _is_generated_data_path(relative, episode_ids)
    if data: return role is None or role in allowed_roles
    match = re.fullmatch(r"audit/(fit|selection|smoke|confirmation)_(native_plan|markov_oracle|plan_action_noise_0p2|plan_random_action_0p1)_execution_failure\.json", relative)
    return bool(match and match.group(1) in allowed_roles)


def _audit_source_attempt_namespace(
    source: Path,
    repo: Path,
    source_seal: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> int:
    complete = source_seal.get("complete_source_partition")
    source_static = complete.get("target_paths") if isinstance(complete, Mapping) else None
    _require(isinstance(source_static, list) and len(source_static) == len(set(source_static)) and set(source_static) == STATIC, "v007 source seal static closure drift")
    expected = set(source_static) | set(SOURCE_SELECTION_BOUNDARY_PRODUCTS) | {
        SOURCE_SEAL,
        SOURCE_RECEIPT,
        FIT_INVENTORY,
        INVALIDITY_DRAFT,
        INVALIDITY,
    }
    observed = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
    expected_dirs = {parent.as_posix() for relative in expected for parent in Path(relative).parents if parent != Path(".")}
    observed_dirs = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_dir()}
    _require(
        len(expected) == 66
        and observed == expected
        and observed_dirs == expected_dirs,
        f"v007 namespace drift: count={len(observed)}, missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}, missing_directories={sorted(expected_dirs-observed_dirs)}, unexpected_directories={sorted(observed_dirs-expected_dirs)}",
    )
    return len(observed)


def _audit_science_attempt_namespace(study: Path, repo: Path, source_seal: Mapping[str, Any]) -> int:
    science = study / "attempts" / SCIENCE_ATTEMPT
    current = source_seal
    visited: set[str] = set()
    while True:
        tip = current.get("source_pre_data_seal")
        invalidity = current.get("invalidity_evidence")
        _require(
            isinstance(tip, Mapping) and isinstance(invalidity, Mapping),
            "transitive seal lost source lineage",
        )
        seal_path = _resolve(repo, str(tip.get("path")))
        _require(file_hash(seal_path) == tip.get("sha256"), "transitive seal hash drift")
        relative = _relative(seal_path, repo)
        _require(relative not in visited, "transitive seal lineage cycle")
        visited.add(relative)
        if seal_path == science / "audit/pre_data_seal.json":
            invalidity_path = _resolve(repo, str(invalidity.get("path")))
            break
        current = _object(seal_path)
    _require(seal_path == science / "audit/pre_data_seal.json" and invalidity_path == science / "audit/v001_procedural_invalidity.json", "v001 lineage paths drift")
    _require(file_hash(seal_path) == tip.get("sha256") and file_hash(invalidity_path) == invalidity.get("sha256"), "v001 lineage hashes drift")
    seal = _object(seal_path); sealed = seal.get("sealed_files")
    _require(isinstance(sealed, Mapping), "v001 sealed_files absent")
    prefix = _relative(science, repo) + "/"
    expected = {str(raw)[len(prefix):] for raw in sealed if isinstance(raw, str) and raw.startswith(prefix)} | {"audit/pre_data_seal.json", "audit/v001_procedural_invalidity.json"}
    observed = {path.relative_to(science).as_posix() for path in science.rglob("*") if path.is_file()}
    expected_dirs = {parent.as_posix() for relative in expected for parent in Path(relative).parents if parent != Path(".")}
    observed_dirs = {path.relative_to(science).as_posix() for path in science.rglob("*") if path.is_dir()}
    _require(len(expected) == 53 and observed == expected and observed_dirs == expected_dirs, "v001 exact namespace drift")
    for raw, digest in sealed.items():
        path = _resolve(repo, str(raw)); _require(path.is_file() and file_hash(path) == digest, f"v001 sealed hash drift: {raw}")
    return len(observed)


def _closure(source: Path, target: Path, repo: Path, source_seal: Mapping[str, Any], state: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    constants = _manifest_policy(target)
    inventory = _fit_inventory(target, repo)
    static = constants["PRE_DATA_STATIC_RELATIVE_PATHS"]
    audits = constants["PRE_DATA_AUDIT_INPUTS"]
    generated = constants["GENERATED_AUDIT_NAMES"]
    repositories = constants["PRE_DATA_REPOSITORY_RELATIVE_PATHS"]
    _require(OUTPUT in generated, "inheritance output is not a generated-audit exclusion")
    scope = _audit_complete_sealed_scope(target.parents[1])
    root = _audit_study_root_namespace(target.parents[1], repo, repositories, constants["MUTABLE_STUDY_ROOT_RELATIVE_PATHS"])
    source_count = _audit_source_attempt_namespace(source, repo, source_seal, inventory)
    science_count = _audit_science_attempt_namespace(target.parents[1], repo, source_seal)
    episode_ids = _prospective_episode_ids(target)
    found: set[str] = set(); generated_present: set[str] = set()
    for path in target.rglob("*"):
        if not path.is_file(): continue
        relative = path.relative_to(target).as_posix()
        if relative in static | audits: found.add(relative)
        elif _generated_source_product(relative, constants, episode_ids, state, phase): generated_present.add(relative)
        else: raise VerificationError(f"unclassified or wrong-phase v008 path: {relative}")
    declared = static | audits
    _require(
        len(declared) == 61 and found == declared,
        f"v008 closure mismatch: count={len(found)}, missing={sorted(declared-found)}, unexpected={sorted(found-declared)}",
    )
    expected_dirs = {parent.as_posix() for relative in declared | generated_present for parent in Path(relative).parents if parent != Path(".")}
    observed_dirs = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_dir()}
    _require(observed_dirs == expected_dirs, f"v008 directory namespace drift: missing={sorted(expected_dirs-observed_dirs)}, unexpected={sorted(observed_dirs-expected_dirs)}")
    _require(repositories == set(EXPECTED_REPOSITORY_PATHS), "repository closure paths drift")
    _require(all(_resolve(repo, raw).is_file() for raw in repositories), "repository closure input absent")
    return {
        "declared_static_paths": sorted(static), "declared_audit_inputs": sorted(audits),
        "declared_generated_audits": sorted(generated),
        "declared_generated_attempt_products": sorted(constants["GENERATED_ATTEMPT_PRODUCTS"]),
        "declared_generated_roles": sorted(constants["GENERATED_ROLES"]),
        "declared_generated_regimes": sorted(constants["GENERATED_REGIMES"]),
        "allowed_source_suffixes": sorted(constants["ALLOWED_SOURCE_SUFFIXES"]),
        "declared_repository_paths": sorted(repositories),
        "source_attempt_allowed_suffix_path_count": source_count,
        "inherited_fit_inventory": {
            key: inventory[key]
            for key in (
                "path", "sha256", "file_count", "episode_count",
                "outcome_arrays_opened", "passed",
            )
        },
        "science_attempt_exact_path_count": science_count,
        "discovered_paths": sorted(found), "missing": [], "unexpected": [],
        **scope, **root, "passed": True,
    }


class _Normalize(ast.NodeTransformer):
    bindings = {"ATTEMPT", "ACTIVE_ATTEMPT"}

    @staticmethod
    def _names(node: ast.AST) -> set[str]:
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
            isinstance(test.left, ast.Name) and test.left.id == "ACTIVE_ATTEMPT"
            and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.NotEq))
            and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value in {SOURCE, TARGET} and bool(node.body)
            and all(isinstance(item, ast.Raise) for item in node.body) and not node.orelse
        )

    def visit_Module(self, node: ast.Module) -> ast.Module:  # noqa: N802
        body: list[ast.stmt] = []
        for item in node.body:
            names = self._names(item)
            if names & self.bindings or self._guard(item):
                continue
            if isinstance(item, ast.Assign) and names in ({"NORMATIVE_SOURCES"}, {"REQUIRED_IMPLEMENTATION_FILES"}):
                collection = item.value
                if isinstance(collection, ast.Call) and collection.args:
                    collection = collection.args[0]
                if isinstance(collection, (ast.List, ast.Tuple, ast.Set)):
                    collection.elts = [
                        child for child in collection.elts
                        if not (isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value in NEW_SUPPORT)
                    ]
            transformed = self.visit(item)
            if transformed is not None:
                body.append(transformed)
        node.body = body
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id in self.bindings:
            return ast.copy_location(ast.Constant("<ATTEMPT>"), node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str):
            value = node.value
            parts = value.split("/")
            if len(parts) == 4 and parts[0] == "runs" and parts[1].startswith("lewm_v5_") and parts[2] == "attempts" and parts[3] == "v005":
                value = "v008"
            value = value.replace("audit/pre_data_inheritance_seal.json", "<PRE_DATA_SEAL>")
            value = re.sub(r"v00[2345678]", "<ACTIVE_ATTEMPT>", value)
            return ast.copy_location(ast.Constant(value), node)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:  # noqa: N802
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


def _active_binding_is_exact(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value
    active = values.get("ACTIVE_ATTEMPT")
    if active is not None and not (isinstance(active, ast.Attribute) and isinstance(active.value, ast.Name) and active.value.id == "ATTEMPT_ROOT" and active.attr == "name"):
        return False
    science = values.get("SCIENCE_ATTEMPT")
    return science is None or (isinstance(science, ast.Constant) and science.value == SCIENCE_ATTEMPT)


def _normalized(path: Path, target: bool) -> tuple[str, str]:
    if target:
        _require(_active_binding_is_exact(path), f"active/science attempt binding drift: {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value = _Normalize().visit(tree)
    ast.fix_missing_locations(value)
    dump = ast.dump(value, annotate_fields=True, include_attributes=False)
    return dump, _hash_bytes(dump.encode())


def _contract(value: Mapping[str, Any]) -> dict[str, Any]:
    attempt = value.get("attempt")
    lineage = value.get("lineage")
    _require(isinstance(lineage, Mapping), "contract lineage absent")
    common = {
        "science_attempt": SCIENCE_ATTEMPT,
        "invalidity_evidence": "runs/lewm_domain_robust_gate/attempts/v001/audit/v001_procedural_invalidity.json",
        "source_pre_data_seal": "runs/lewm_domain_robust_gate/attempts/v001/audit/pre_data_seal.json",
    }
    source_lineage = {
        **common,
        "active_attempt": SOURCE,
        "attempts": [
            SCIENCE_ATTEMPT,
            "v002",
            FIT_SOURCE,
            INTERMEDIATE,
            SOURCE_PARENT,
            SELECTION_SOURCE,
            SOURCE,
        ],
        "source_roots": {
            SCIENCE_ATTEMPT: "runs/lewm_domain_robust_gate/attempts/v001",
            "v002": "runs/lewm_domain_robust_gate/attempts/v002",
            FIT_SOURCE: "runs/lewm_domain_robust_gate/attempts/v003",
            INTERMEDIATE: "runs/lewm_domain_robust_gate/attempts/v004",
            SOURCE_PARENT: "runs/lewm_domain_robust_gate/attempts/v005",
            SELECTION_SOURCE: "runs/lewm_domain_robust_gate/attempts/v006",
            SOURCE: "runs/lewm_domain_robust_gate/attempts/v007",
        },
        "pre_data_inheritance_seal": "runs/lewm_domain_robust_gate/attempts/v007/audit/pre_data_inheritance_seal.json",
        "prior_pre_data_inheritance_seal": "runs/lewm_domain_robust_gate/attempts/v006/audit/pre_data_inheritance_seal.json",
        "source_invalidity_evidence": "runs/lewm_domain_robust_gate/attempts/v006/audit/v006_procedural_invalidity.json",
        "superseded_source_invalidity_draft": "runs/lewm_domain_robust_gate/attempts/v006/audit/v006_procedural_invalidity_draft.json",
        "prior_version_forward_receipt": "runs/lewm_domain_robust_gate/attempts/v006/audit/version_forward_transaction_receipt.json",
        "version_forward_receipt": "runs/lewm_domain_robust_gate/attempts/v007/audit/version_forward_transaction_receipt.json",
    }
    target_lineage = {
        **common,
        "active_attempt": TARGET,
        "attempts": [
            SCIENCE_ATTEMPT,
            "v002",
            FIT_SOURCE,
            INTERMEDIATE,
            SOURCE_PARENT,
            SELECTION_SOURCE,
            SOURCE,
            TARGET,
        ],
        "source_roots": {
            SCIENCE_ATTEMPT: "runs/lewm_domain_robust_gate/attempts/v001",
            "v002": "runs/lewm_domain_robust_gate/attempts/v002",
            FIT_SOURCE: "runs/lewm_domain_robust_gate/attempts/v003",
            INTERMEDIATE: "runs/lewm_domain_robust_gate/attempts/v004",
            SOURCE_PARENT: "runs/lewm_domain_robust_gate/attempts/v005",
            SELECTION_SOURCE: "runs/lewm_domain_robust_gate/attempts/v006",
            SOURCE: "runs/lewm_domain_robust_gate/attempts/v007",
            TARGET: "runs/lewm_domain_robust_gate/attempts/v008",
        },
        "pre_data_inheritance_seal": "runs/lewm_domain_robust_gate/attempts/v008/audit/pre_data_inheritance_seal.json",
        "prior_pre_data_inheritance_seal": "runs/lewm_domain_robust_gate/attempts/v007/audit/pre_data_inheritance_seal.json",
        "source_invalidity_evidence": "runs/lewm_domain_robust_gate/attempts/v007/audit/v007_procedural_invalidity.json",
        "superseded_source_invalidity_draft": "runs/lewm_domain_robust_gate/attempts/v007/audit/v007_procedural_invalidity_draft.json",
        "prior_version_forward_receipt": "runs/lewm_domain_robust_gate/attempts/v007/audit/version_forward_transaction_receipt.json",
        "version_forward_receipt": "runs/lewm_domain_robust_gate/attempts/v008/audit/version_forward_transaction_receipt.json",
    }
    _require(dict(lineage) == (source_lineage if attempt == SOURCE else target_lineage), f"contract transitive lineage drift: {attempt}")

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str):
            return item.replace("/attempts/v008/", "/attempts/v007/").replace(
                "/attempts/v008", "/attempts/v007"
            )
        return item
    result = visit(dict(value))
    result["attempt"] = SOURCE
    result["attempt_root"] = "runs/lewm_domain_robust_gate/attempts/v007"
    if isinstance(result.get("lineage"), dict):
        result["lineage"] = source_lineage
    return result


def _symbols(path: Path) -> dict[str, str]:
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
        result[label] = _hash_bytes(
            ast.dump(
                node, annotate_fields=True, include_attributes=False
            ).encode()
        )
    _require(len(result) == len(tree.body), f"top-level unit census incomplete: {path}")
    return result


def _scope(left: Path, right: Path, relative: str) -> dict[str, Any]:
    a, b = _symbols(left), _symbols(right)
    changed = sorted(name for name in set(a) | set(b) if a.get(name) != b.get(name))
    allowed = PROCEDURAL_ALLOWED_SYMBOLS[relative]
    expected = PROCEDURAL_EXPECTED_CHANGED_SYMBOLS[relative]
    _require("*" not in allowed and set(changed).issubset(allowed), f"procedural AST scope escape: {relative}: {changed}")
    _require(set(changed) == expected, f"procedural required changed-symbol set drift: {relative}: expected={sorted(expected)}, observed={changed}")
    diff = "".join(difflib.unified_diff(left.read_text(encoding="utf-8").splitlines(keepends=True), right.read_text(encoding="utf-8").splitlines(keepends=True), fromfile=relative, tofile=relative)).encode()
    _require(bool(diff), f"procedural file unexpectedly exact: {relative}")
    return {
        "changed_top_level_symbols": changed, "allowed_top_level_symbols": sorted(allowed),
        "unchanged_top_level_symbol_count": len([name for name in set(a) & set(b) if a[name] == b[name]]),
        "diff_sha256": _hash_bytes(diff),
    }


def _lineage_scope(left_path: Path, right_path: Path, relative: str) -> dict[str, Any]:
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
        f"lineage unit policy absent or malformed: {relative}",
    )
    left, right = _symbols(left_path), _symbols(right_path)
    source_sequence_sha256 = _json_hash(list(left))
    target_sequence_sha256 = _json_hash(list(right))
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
        set(LINEAGE_SUPPORT_UNIT_POLICY) == set(LINEAGE_SUPPORT)
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
        f"lineage top-level unit policy drift: {relative}",
    )
    allowed_set = set(allowed)
    _require(
        all(
            left[label] == right[label]
            for label in set(left) & set(right)
            if label not in allowed_set
        ),
        f"lineage nonallowed unit changed: {relative}",
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
        "unchanged_top_level_units_sha256": _json_hash(unchanged),
    }


def _source_paths(source: Path, repo: Path) -> tuple[dict[str, Any], set[str]]:
    del repo
    seal = _object(source / SOURCE_SEAL)
    complete = seal.get("complete_source_partition")
    paths = complete.get("target_paths") if isinstance(complete, Mapping) else None
    _require(
        isinstance(paths, list)
        and len(paths) == len(set(paths))
        and set(paths) == STATIC - NEW_SUPPORT,
        "v006 source static partition drift",
    )
    return seal, set(paths)


def _verify_records(artifact: Mapping[str, Any], source: Path, target: Path, repo: Path, closure: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    _manifest, source_paths = _source_paths(source, repo)
    target_paths = set(closure["discovered_paths"])
    _require(
        source_paths == STATIC - NEW_SUPPORT and target_paths == STATIC,
        "v006/v008 static path census drift",
    )
    raw = artifact.get("source_partitions")
    _require(isinstance(raw, dict), "source_partitions absent")
    expected_keys = {"exact_hash", "normalized_ast", "canonical_contracts", "procedural_only", "new_lineage_support", "lineage_support"}
    _require(set(raw) == expected_keys, "source partition key set drift")
    _require(
        all(isinstance(raw[name], list) for name in expected_keys),
        "source partition member collection is not a list",
    )
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in expected_keys}
    members: dict[str, list[str]] = {key: [] for key in expected_keys}

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
        _require(len(values) == 1, f"lineage unit policy literal count drift: {path}")
        try:
            return ast.literal_eval(values[0])
        except (ValueError, TypeError, SyntaxError) as exc:
            raise VerificationError(f"lineage unit policy is not literal: {path}") from exc

    for owner in (
        "version_forward_prepare.py",
        "verify_version_forward.py",
        "independent_verify.py",
    ):
        _require(
            literal_policy(target / owner) == LINEAGE_SUPPORT_UNIT_POLICY,
            f"lineage unit policy transcription drift: {owner}",
        )

    def record_path(item: Mapping[str, Any]) -> tuple[str, Path, Path]:
        relative = item.get("relative_path")
        _require(isinstance(relative, str), "partition relative path invalid")
        left, right = source / relative, target / relative
        _require(item.get("source_path") == _relative(left, repo) and item.get("target_path") == _relative(right, repo), f"partition path binding drift: {relative}")
        return relative, left, right

    for item in raw["exact_hash"]:
        _require(isinstance(item, dict), "exact record invalid")
        relative, left, right = record_path(item)
        digest = file_hash(left)
        _require(digest == file_hash(right) == item.get("sha256"), f"exact hash drift: {relative}")
        _require(item.get("bytes") == left.stat().st_size, f"exact byte count drift: {relative}")
        result["exact_hash"].append(dict(item)); members["exact_hash"].append(relative)
    for item in raw["normalized_ast"]:
        _require(isinstance(item, dict), "AST record invalid")
        relative, left, right = record_path(item)
        _require(relative in NORMALIZED and item.get("transform_id") == "v006_to_v008_active_attempt_ast_v1", f"AST allowlist drift: {relative}")
        a, ah = _normalized(left, False); b, bh = _normalized(right, True)
        _require(a == b and ah == bh == item.get("normalized_ast_sha256"), f"normalized AST drift: {relative}")
        _require(file_hash(left) == item.get("source_sha256") and file_hash(right) == item.get("target_sha256"), f"AST raw hash drift: {relative}")
        result["normalized_ast"].append(dict(item)); members["normalized_ast"].append(relative)
    for item in raw["canonical_contracts"]:
        _require(isinstance(item, dict), "contract record invalid")
        relative, left, right = record_path(item)
        _require(relative in CONTRACTS and item.get("transform_id") == "v008_transitive_lineage_contract_v1", f"contract allowlist drift: {relative}")
        a, b = _contract(_object(left)), _contract(_object(right))
        _require(a == b and _json_hash(a) == item.get("canonical_sha256"), f"contract canonical drift: {relative}")
        _require(file_hash(left) == item.get("source_sha256") and file_hash(right) == item.get("target_sha256"), f"contract raw hash drift: {relative}")
        result["canonical_contracts"].append(dict(item)); members["canonical_contracts"].append(relative)
    for item in raw["procedural_only"]:
        _require(isinstance(item, dict), "procedural record invalid")
        relative, left, right = record_path(item)
        _require(relative in PROCEDURAL and item.get("change_id") == "zero_outcome_v008_generator_root_parent_repair_v1", f"procedural allowlist drift: {relative}")
        scope = _scope(left, right, relative)
        _require(all(item.get(key) == value for key, value in scope.items()), f"procedural scope record drift: {relative}")
        _require(file_hash(left) == item.get("source_sha256") and file_hash(right) == item.get("target_sha256"), f"procedural raw hash drift: {relative}")
        result["procedural_only"].append(dict(item)); members["procedural_only"].append(relative)
    for item in raw["new_lineage_support"]:
        _require(isinstance(item, dict), "new-support record invalid")
        relative = item.get("relative_path")
        _require(
            isinstance(relative, str)
            and relative in NEW_SUPPORT
            and set(item)
            == {"relative_path", "target_path", "target_sha256", "bytes", "purpose"},
            f"target-only support schema drift: {relative}",
        )
        right = target / relative
        _require(
            item.get("target_path") == _relative(right, repo)
            and item.get("target_sha256") == file_hash(right)
            and item.get("bytes") == right.stat().st_size
            and item.get("purpose")
            == "immutable_v004_fit_role_provenance_and_read_only_path_map",
            f"target-only support provenance drift: {relative}",
        )
        result["new_lineage_support"].append(dict(item)); members["new_lineage_support"].append(relative)
    for item in raw["lineage_support"]:
        _require(isinstance(item, dict), "lineage-support record invalid")
        relative, left, right = record_path(item)
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
        _require(set(item) == expected_record_keys, f"lineage-support record schema drift: {relative}")
        _require(relative in LINEAGE_SUPPORT, f"lineage-support allowlist drift: {relative}")
        _require(item.get("purpose") == "v008_transitive_lineage_authorization_or_test_only", f"lineage-support purpose drift: {relative}")
        _require(file_hash(left) == item.get("source_sha256") and file_hash(right) == item.get("target_sha256"), f"lineage-support hash drift: {relative}")
        scope = _lineage_scope(left, right, relative)
        _require(
            all(item.get(key) == value for key, value in scope.items()),
            f"lineage-support unit scope record drift: {relative}",
        )
        result["lineage_support"].append(dict(item)); members["lineage_support"].append(relative)
    all_members = [name for values in members.values() for name in values]
    _require(len(all_members) == len(set(all_members)), "partition overlap")
    expected_members = {
        "exact_hash": EXACT,
        "normalized_ast": NORMALIZED,
        "canonical_contracts": CONTRACTS,
        "procedural_only": PROCEDURAL,
        "new_lineage_support": NEW_SUPPORT,
        "lineage_support": LINEAGE_SUPPORT,
    }
    _require(
        all(
            members[name] == sorted(expected_members[name])
            for name in expected_keys
        ),
        "v006->v008 partition membership/order drift",
    )
    required_procedural = {
        relative
        for relative, symbols in PROCEDURAL_EXPECTED_CHANGED_SYMBOLS.items()
        if symbols
    }
    _require(
        required_procedural <= set(members["procedural_only"])
        and not (required_procedural & set(members["exact_hash"])),
        "required procedural module reverted into exact partition",
    )
    _require(
        len(source_paths) == 61
        and len(target_paths) == 61
        and set(all_members) == target_paths
        and source_paths == target_paths,
        "partition coverage is not the exact v006/v008 namespace",
    )
    expected_complete = {
        "source_paths": sorted(source_paths), "target_paths": sorted(target_paths),
        "partition_members": members, "overlap": [], "unpartitioned_source": [], "unpartitioned_target": [], "passed": True,
    }
    _require(artifact.get("complete_source_partition") == expected_complete, "complete partition summary drift")
    return result, members


def _ledger(study: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    genesis = _object(study / "LEDGER_CHAIN_GENESIS.json")
    raw = (study / "RESEARCH_LEDGER.jsonl").read_bytes()
    prefix_bytes, legacy = genesis.get("legacy_prefix_bytes"), genesis.get("legacy_prefix_event_count")
    _require(type(prefix_bytes) is int and type(legacy) is int, "genesis count type drift")
    _require(_hash_bytes(raw[:prefix_bytes]) == genesis.get("legacy_prefix_sha256"), "legacy prefix drift")
    lines = raw.splitlines(); _require(len(lines) >= legacy, "ledger prefix truncated")
    previous = f"legacy:{genesis['legacy_prefix_sha256']}"; events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        value = json.loads(line); _require(isinstance(value, dict), "ledger record non-object")
        events.append(dict(value))
        if index < legacy:
            continue
        record = dict(value); observed = record.pop("record_sha256", None)
        _require(observed == _json_hash(record), f"ledger record hash drift: {index+1}")
        _require(
            type(record.get("seq")) is int
            and record.get("seq") == index + 1
            and isinstance(record.get("prev_sha256"), str)
            and record.get("prev_sha256") == previous,
            f"ledger chain drift: {index+1}",
        )
        previous = str(observed)
    return {
        "event_count": len(lines), "head_sha256": previous,
        "ledger_sha256": file_hash(study / "RESEARCH_LEDGER.jsonl"),
        "genesis_sha256": file_hash(study / "LEDGER_CHAIN_GENESIS.json"),
    }, events


def _source_adapter_marker(
    value: Any,
    *,
    source_state: Mapping[str, Any],
    source: Path,
    study: Path,
    repo: Path,
    event_count: int,
    head_sha256: str,
) -> dict[str, Any]:
    """Authenticate the durable v007 marker from bytes and state binding."""

    expected_keys = {
        "schema_version", "authorization_kind", "adapter_source_path",
        "adapter_source_sha256", "adapter_source_ast_sha256",
        "root_program_path", "root_program_sha256",
        "root_program_ast_sha256", "ledger_event_count",
        "ledger_head_sha256", "operation_sha256", "state_binding_sha256",
    }
    adapter = source / "version_forward_transaction.py"
    root_program = study / "program.py"
    _require(isinstance(value, Mapping), "v007 durable adapter marker absent")
    marker = dict(value)
    marker_without_binding = dict(marker)
    state_binding = marker_without_binding.pop("state_binding_sha256", None)
    state_without_marker = json.loads(json.dumps(dict(source_state)))
    state_without_marker.pop("v007_durable_controller_adapter", None)
    _require(
        set(marker) == expected_keys
        and type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 2
        and marker.get("authorization_kind")
        == "receipt_bound_v007_root_controller_adapter"
        and marker.get("adapter_source_path") == _relative(adapter, repo)
        and marker.get("adapter_source_sha256")
        == file_hash(adapter) == EXPECTED_SOURCE_ADAPTER_SHA256
        and marker.get("adapter_source_ast_sha256")
        == _source_ast_sha256(adapter)
        and marker.get("root_program_path") == _relative(root_program, repo)
        and marker.get("root_program_sha256") == file_hash(root_program)
        and marker.get("root_program_ast_sha256")
        == _source_ast_sha256(root_program)
        and type(marker.get("ledger_event_count")) is int
        and marker.get("ledger_event_count") == event_count
        and marker.get("ledger_head_sha256") == head_sha256
        and isinstance(marker.get("operation_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(marker["operation_sha256"]))
        is not None
        and isinstance(state_binding, str)
        and state_binding
        == _json_hash(
            {
                "marker_without_state_binding": marker_without_binding,
                "state_without_marker": state_without_marker,
            }
        ),
        "v007 durable adapter marker/hash/state binding drift",
    )
    return marker


def _authenticate_immediate_source_legacy(
    *,
    study: Path,
    repo: Path,
    source: Path,
    source_seal: Mapping[str, Any],
    invalidity: Mapping[str, Any],
    invalidity_draft: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    live_state: Mapping[str, Any],
    live_chain: Mapping[str, Any],
    live_events: Sequence[Mapping[str, Any]],
    phase: str,
) -> dict[str, Any]:
    """Legacy implementation retained only for source-level comparison."""

    invalidity_path = source / INVALIDITY
    draft_path = source / INVALIDITY_DRAFT
    seal_path = source / SOURCE_SEAL
    receipt_path = source / SOURCE_RECEIPT
    _require(
        file_hash(invalidity_path) == EXPECTED_INVALIDITY_SHA256
        and file_hash(draft_path) == EXPECTED_INVALIDITY_DRAFT_SHA256
        and file_hash(seal_path) == EXPECTED_SOURCE_SEAL_SHA256
        and file_hash(receipt_path) == EXPECTED_SOURCE_RECEIPT_SHA256,
        "fixed v005 invalidity/seal/receipt hash drift",
    )
    prior_draft = invalidity.get("superseded_draft")
    fit_evidence = invalidity.get("fit_evidence")
    defect = invalidity.get("defect")
    _require(
        invalidity.get("attempt") == SOURCE
        and invalidity.get("procedural_invalidity") is True
        and invalidity.get("authoritative") is True
        and invalidity.get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and invalidity.get("current_state") == RESUME
        and invalidity.get("scientific_criterion_changed") is False
        and invalidity.get("scientific_outcomes_conditioned_on") is False
        and invalidity.get("selection_or_later_outcomes_opened") is False
        and invalidity.get("immutable_fit_evidence_preserved") is True
        and _exact_typed_mapping(
            invalidity.get("controller_outcome_counts"), EXPECTED_COUNTS
        )
        and invalidity.get("later_role_artifact_file_counts")
        == {"confirmation": 0, "selection": 0, "smoke": 0}
        and invalidity.get("confirmation_artifact_file_count") == 0
        and isinstance(fit_evidence, Mapping)
        and fit_evidence.get("inherited_episode_count")
        == EXPECTED_FIT_EPISODE_COUNT
        and fit_evidence.get("inherited_file_count")
        == EXPECTED_FIT_FILE_COUNT
        and fit_evidence.get("source_attempt") == FIT_SOURCE
        and fit_evidence.get("outcome_arrays_opened") is False
        and fit_evidence.get(
            "development_part_arrays_opened_by_failed_checkpoint_verification"
        ) is False
        and fit_evidence.get(
            "aggregate_arrays_opened_by_failed_checkpoint_verification"
        ) is False
        and fit_evidence.get("controller_count_transition_completed") is False
        and fit_evidence.get("independent_scientific_replay_started") is False
        and isinstance(defect, Mapping)
        and defect.get("classification")
        == "inherited_fit_version_forward_source_role_conflation"
        and defect.get("exception_type") == "WorkflowError"
        and defect.get("failure_location")
        == "workflow._verify_inherited_fit_manifests.inheritance.source_attempt"
        and defect.get("failure_message")
        == "fit-role inventory is not bound by the active inheritance seal"
        and defect.get("reproduced_without_outcome_arrays") is True
        and defect.get("reproduction_input")
        == {
            "inheritance_seal_path": _relative(seal_path, repo),
            "inherited_fit_role_source_attempt": FIT_SOURCE,
            "sealed_version_forward_source_attempt": INTERMEDIATE,
            "workflow_comparison_source_attempt": FIT_SOURCE,
        }
        and defect.get("independent_verifier_sha256")
        == file_hash(source / "independent_verify.py")
        and defect.get("standalone_version_forward_verifier_sha256")
        == file_hash(source / "verify_version_forward.py")
        and defect.get("workflow_sha256") == file_hash(source / "workflow.py")
        and isinstance(prior_draft, Mapping)
        and prior_draft.get("authoritative") is False
        and prior_draft.get("path") == _relative(draft_path, repo)
        and prior_draft.get("sha256") == EXPECTED_INVALIDITY_DRAFT_SHA256
        and invalidity_draft.get("attempt") == SOURCE
        and invalidity_draft.get("procedural_invalidity") is True
        and invalidity_draft.get("authoritative") is False,
        "v005 authoritative invalidity/draft binding drift",
    )
    frozen = invalidity.get("frozen_evidence")
    _require(
        isinstance(frozen, Mapping)
        and frozen.get("controller_state_sha256_at_failure")
        == EXPECTED_SOURCE_STATE_SHA256
        and frozen.get("research_ledger_sha256_at_failure")
        == EXPECTED_SOURCE_LEDGER_SHA256
        and frozen.get("pre_data_inheritance_seal_sha256")
        == EXPECTED_SOURCE_SEAL_SHA256
        and frozen.get("version_forward_receipt_sha256")
        == EXPECTED_SOURCE_RECEIPT_SHA256
        and frozen.get("ledger_chain_genesis_sha256")
        == file_hash(study / "LEDGER_CHAIN_GENESIS.json")
        and frozen.get("root_program_sha256") == file_hash(study / "program.py")
        and file_hash(source / "cohort_seed_ledger.json")
        == EXPECTED_COHORT_LEDGER_SHA256
        and file_hash(source / "generator.py") == EXPECTED_SOURCE_GENERATOR_SHA256
        and file_hash(source / "launcher.py") == EXPECTED_SOURCE_LAUNCHER_SHA256,
        "v005 frozen checkpoint drift",
    )
    _require(
        source_seal.get("attempt") == SOURCE
        and source_seal.get("source_attempt") == INTERMEDIATE
        and source_seal.get("science_attempt") == SCIENCE_ATTEMPT
        and source_seal.get("target_attempt") == SOURCE
        and source_seal.get("passed") is True
        and _exact_typed_mapping(source_seal.get("outcome_counts_at_seal"), EXPECTED_COUNTS),
        "v005 inheritance-seal lineage header drift",
    )
    receipt_context = source_receipt.get("receipt_context")
    post_snapshot = source_receipt.get("post_snapshot")
    post_state_record = (
        post_snapshot.get("state") if isinstance(post_snapshot, Mapping) else None
    )
    post_ledger = (
        post_snapshot.get("ledger") if isinstance(post_snapshot, Mapping) else None
    )
    post_genesis = (
        post_snapshot.get("genesis") if isinstance(post_snapshot, Mapping) else None
    )
    source_state = (
        post_state_record.get("object")
        if isinstance(post_state_record, Mapping)
        else None
    )
    transaction_source = source_receipt.get("transaction_source")
    receipt_seal = source_receipt.get("seal")
    _require(
        source_receipt.get("source_attempt") == INTERMEDIATE
        and source_receipt.get("target_attempt") == SOURCE
        and source_receipt.get("passed") is True
        and isinstance(receipt_context, Mapping)
        and _json_hash(receipt_context)
        == source_receipt.get("receipt_context_sha256")
        and isinstance(source_state, Mapping)
        and isinstance(post_state_record, Mapping)
        and post_state_record.get("sha256") == EXPECTED_SOURCE_STATE_SHA256
        and _hash_bytes(
            (json.dumps(dict(source_state), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        ) == EXPECTED_SOURCE_STATE_SHA256
        and post_state_record.get("object_sha256") == _json_hash(source_state)
        and source_receipt.get("controller_return") == source_state
        and source_receipt.get("controller_return_sha256")
        == _json_hash(source_state)
        and isinstance(post_ledger, Mapping)
        and post_ledger.get("sha256") == EXPECTED_SOURCE_LEDGER_SHA256
        and post_ledger.get("ledger_sha256") == EXPECTED_SOURCE_LEDGER_SHA256
        and type(post_ledger.get("event_count")) is int
        and post_ledger.get("event_count") == 11
        and isinstance(post_ledger.get("head_sha256"), str)
        and isinstance(post_genesis, Mapping)
        and post_genesis.get("sha256")
        == file_hash(study / "LEDGER_CHAIN_GENESIS.json")
        and isinstance(transaction_source, Mapping)
        and transaction_source.get("sha256") == EXPECTED_SOURCE_ADAPTER_SHA256
        and isinstance(receipt_seal, Mapping)
        and receipt_seal.get("sha256") == EXPECTED_SOURCE_SEAL_SHA256,
        "v005 prior transaction receipt/live-prestate cross-link drift",
    )
    source_event_count = int(post_ledger["event_count"])
    _require(
        len(live_events) >= source_event_count,
        "research ledger truncated before the v005 prefix",
    )
    ledger_lines = (study / "RESEARCH_LEDGER.jsonl").read_bytes().splitlines(
        keepends=True
    )
    source_prefix = b"".join(ledger_lines[:source_event_count])
    _require(
        _hash_bytes(source_prefix) == EXPECTED_SOURCE_LEDGER_SHA256
        and live_events[source_event_count - 1].get("record_sha256")
        == post_ledger.get("head_sha256")
        and source_state.get("ledger_event_count") == source_event_count
        and source_state.get("ledger_head_sha256")
        == post_ledger.get("head_sha256"),
        "live ledger does not preserve the exact v005 prefix",
    )
    lineage = source_state.get("version_forward_lineage")
    history = source_state.get("attempt_history")
    _require(
        source_state.get("active_attempt") == SOURCE
        and source_state.get("active_attempt_path") == _relative(source, repo)
        and "v008_durable_controller_adapter" not in source_state
        and source_state.get("current_state") == RESUME
        and _exact_typed_mapping(
            {key: source_state.get(key) for key in EXPECTED_COUNTS}, EXPECTED_COUNTS
        )
        and isinstance(lineage, list)
        and len(lineage) == 4
        and isinstance(lineage[-1], Mapping)
        and lineage[-1].get("old_attempt") == INTERMEDIATE
        and lineage[-1].get("new_attempt") == SOURCE
        and lineage[-1].get("equivalence_sha256")
        == EXPECTED_SOURCE_SEAL_SHA256
        and isinstance(history, list)
        and [item.get("version") for item in history if isinstance(item, Mapping)]
        == [SCIENCE_ATTEMPT, "v002", FIT_SOURCE, INTERMEDIATE, SOURCE],
        "v001->v002->v003->v004->v005 source controller lineage/history drift",
    )
    marker = _source_adapter_marker(
        source_state.get("v005_durable_controller_adapter"),
        source_state=source_state,
        source=source,
        study=study,
        repo=repo,
        event_count=source_event_count,
        head_sha256=str(post_ledger["head_sha256"]),
    )
    if phase == "pre-forward":
        _require(
            dict(live_state) == dict(source_state)
            and live_chain.get("event_count") == source_event_count
            and live_chain.get("head_sha256") == post_ledger.get("head_sha256")
            and live_chain.get("ledger_sha256") == EXPECTED_SOURCE_LEDGER_SHA256
            and file_hash(study / "STATE.json") == EXPECTED_SOURCE_STATE_SHA256,
            "pre-forward live controller differs from fixed v005 checkpoint",
        )
    return {
        "state": dict(source_state),
        "lineage": json.loads(json.dumps(lineage)),
        "history": json.loads(json.dumps(history)),
        "marker": marker,
        "ledger": dict(post_ledger),
        "genesis": dict(post_genesis),
    }


def _authenticate_immediate_source(
    *,
    study: Path,
    repo: Path,
    source: Path,
    source_seal: Mapping[str, Any],
    invalidity: Mapping[str, Any],
    invalidity_draft: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    live_state: Mapping[str, Any],
    live_chain: Mapping[str, Any],
    live_events: Sequence[Mapping[str, Any]],
    phase: str,
    authorized_transaction_context_sha256: str | None = None,
) -> dict[str, Any]:
    """Reconstruct and authenticate the immutable v007 selection boundary."""

    invalidity_path = source / INVALIDITY
    draft_path = source / INVALIDITY_DRAFT
    source_seal_path = source / SOURCE_SEAL
    source_receipt_path = source / SOURCE_RECEIPT
    selection_source = source.parent / SELECTION_SOURCE
    pre_selection_path = selection_source / SOURCE_PRE_SELECTION_SEAL
    fixed = {
        invalidity_path: EXPECTED_INVALIDITY_SHA256,
        draft_path: EXPECTED_INVALIDITY_DRAFT_SHA256,
        source_seal_path: EXPECTED_SOURCE_SEAL_SHA256,
        source_receipt_path: EXPECTED_SOURCE_RECEIPT_SHA256,
        pre_selection_path: EXPECTED_SOURCE_PRE_SELECTION_SEAL_SHA256,
    }
    _require(
        all(path.is_file() and file_hash(path) == digest for path, digest in fixed.items()),
        "v007 immutable lineage record hash drift",
    )

    defect = invalidity.get("defect")
    fit_evidence = invalidity.get("fit_evidence")
    selection_evidence = invalidity.get("selection_evidence")
    frozen = invalidity.get("frozen_evidence")
    _require(
        invalidity.get("attempt") == SOURCE
        and invalidity.get("procedural_invalidity") is True
        and invalidity.get("authoritative") is True
        and invalidity.get("status")
        == "invalid_zero_confirmation_outcome_procedural"
        and invalidity.get("current_state") == RESUME
        and invalidity.get("scientific_criterion_changed") is False
        and invalidity.get("scientific_outcomes_conditioned_on") is False
        and invalidity.get("selection_or_later_outcomes_opened") is False
        and invalidity.get("immutable_fit_evidence_preserved") is True
        and _exact_typed_mapping(
            invalidity.get("controller_outcome_counts"), EXPECTED_COUNTS
        )
        and invalidity.get("later_role_artifact_file_counts")
        == {"confirmation": 0, "selection": 0, "smoke": 0}
        and invalidity.get("confirmation_artifact_file_count") == 0
        and isinstance(defect, Mapping)
        and defect.get("classification")
        == "independent_verifier_contract_schema_rejects_authenticated_inherited_selection_boundary_paths"
        and defect.get("reproduced_without_outcome_arrays") is True
        and defect.get("exception_type") == "VerificationError"
        and defect.get("contract_schema_only") is True
        and "pre_selection_seal" in str(defect.get("failure_message"))
        and isinstance(fit_evidence, Mapping)
        and fit_evidence.get("fit_outcome_episode_count")
        == EXPECTED_FIT_EPISODE_COUNT
        and fit_evidence.get("independent_scientific_replay_completed") is True
        and fit_evidence.get("selection_outcomes_used_for_fit") is False
        and fit_evidence.get("immutable_and_reusable_under_authenticated_lineage")
        is True
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
        and frozen.get("pre_selection_seal_sha256")
        == EXPECTED_SOURCE_PRE_SELECTION_SEAL_SHA256
        and frozen.get("version_forward_receipt_sha256")
        == EXPECTED_SOURCE_RECEIPT_SHA256,
        "v007 procedural-invalidity provenance drift",
    )
    prior_draft = invalidity.get("superseded_draft")
    _require(
        isinstance(prior_draft, Mapping)
        and prior_draft.get("path") == _relative(draft_path, repo)
        and prior_draft.get("sha256") == EXPECTED_INVALIDITY_DRAFT_SHA256
        and prior_draft.get("authoritative") is False
        and invalidity_draft.get("attempt") == SOURCE
        and invalidity_draft.get("procedural_invalidity") is True
        and invalidity_draft.get("authoritative") is False,
        "v007 invalidity-draft provenance drift",
    )

    inherited_fit = source_seal.get("inherited_fit_role")
    _require(
        source_seal.get("attempt") == SOURCE
        and source_seal.get("source_attempt") == SELECTION_SOURCE
        and source_seal.get("target_attempt") == SOURCE
        and source_seal.get("science_attempt") == SCIENCE_ATTEMPT
        and source_seal.get("resume_state") == "SELECTION_COHORTS"
        and source_seal.get("passed") is True
        and _exact_typed_mapping(
            source_seal.get("outcome_counts_at_seal"), SOURCE_ACTIVATION_COUNTS
        )
        and isinstance(inherited_fit, Mapping)
        and inherited_fit.get("episode_count") == EXPECTED_FIT_EPISODE_COUNT
        and inherited_fit.get("file_count") == EXPECTED_FIT_FILE_COUNT
        and inherited_fit.get("inventory_sha256")
        == EXPECTED_SOURCE_FIT_INVENTORY_SHA256,
        "v007 source-seal provenance drift",
    )
    post_snapshot = source_receipt.get("post_snapshot")
    activation_state_record = (
        post_snapshot.get("state") if isinstance(post_snapshot, Mapping) else None
    )
    activation_ledger = (
        post_snapshot.get("ledger") if isinstance(post_snapshot, Mapping) else None
    )
    activation_state = (
        activation_state_record.get("object")
        if isinstance(activation_state_record, Mapping)
        else None
    )
    _require(
        source_receipt.get("source_attempt") == SELECTION_SOURCE
        and source_receipt.get("target_attempt") == SOURCE
        and source_receipt.get("resume_state") == "SELECTION_COHORTS"
        and source_receipt.get("passed") is True
        and isinstance(activation_state, Mapping)
        and activation_state_record.get("sha256")
        == EXPECTED_SOURCE_PRE_FORWARD_STATE_SHA256
        and activation_state_record.get("object_sha256")
        == _json_hash(activation_state)
        and activation_state.get("active_attempt") == SOURCE
        and activation_state.get("current_state") == "SELECTION_COHORTS"
        and _exact_typed_mapping(
            {key: activation_state.get(key) for key in EXPECTED_COUNTS},
            SOURCE_ACTIVATION_COUNTS,
        )
        and isinstance(activation_ledger, Mapping)
        and activation_ledger.get("sha256")
        == EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256
        and activation_ledger.get("ledger_sha256")
        == EXPECTED_SOURCE_PRE_FORWARD_LEDGER_SHA256,
        "v007 activation receipt binding drift",
    )

    if phase == "pre-forward":
        source_state = dict(live_state)
        source_ledger = {
            "path": _relative(study / "RESEARCH_LEDGER.jsonl", repo),
            "sha256": live_chain.get("ledger_sha256"),
            "ledger_sha256": live_chain.get("ledger_sha256"),
            "bytes": (study / "RESEARCH_LEDGER.jsonl").stat().st_size,
            "event_count": live_chain.get("event_count"),
            "head_sha256": live_chain.get("head_sha256"),
        }
    else:
        target_receipt_path = (
            source.parent / TARGET / "audit/version_forward_transaction_receipt.json"
        )
        journal_path = study / "VERSION_FORWARD_TRANSACTION.json"
        if target_receipt_path.is_file():
            context_source = _object(target_receipt_path)
            snapshot = context_source.get("pre_snapshot")
            _require(
                context_source.get("source_attempt") == SOURCE
                and context_source.get("target_attempt") == TARGET
                and context_source.get("resume_state") == RESUME,
                "v008 receipt source-boundary identity drift",
            )
        else:
            _require(
                journal_path.is_file()
                and isinstance(authorized_transaction_context_sha256, str),
                "post-forward source snapshot lacks transaction provenance",
            )
            journal = _object(journal_path)
            context = journal.get("receipt_context")
            _require(
                isinstance(context, Mapping)
                and _json_hash(context) == authorized_transaction_context_sha256
                == journal.get("receipt_context_sha256")
                and context.get("source_attempt") == SOURCE
                and context.get("target_attempt") == TARGET
                and context.get("resume_state") == RESUME,
                "transient source snapshot context drift",
            )
            snapshot = context.get("pre_snapshot")
        _require(isinstance(snapshot, Mapping), "source boundary snapshot absent")
        state_record = snapshot.get("state")
        source_ledger = snapshot.get("ledger")
        _require(
            isinstance(state_record, Mapping)
            and isinstance(state_record.get("object"), Mapping)
            and isinstance(source_ledger, Mapping),
            "source boundary snapshot schema drift",
        )
        source_state = dict(state_record["object"])

    _require(
        _hash_bytes(
            (json.dumps(source_state, indent=2, sort_keys=True) + "\n").encode()
        )
        == EXPECTED_SOURCE_STATE_SHA256
        and source_ledger.get("sha256") == EXPECTED_SOURCE_LEDGER_SHA256
        and source_ledger.get("ledger_sha256") == EXPECTED_SOURCE_LEDGER_SHA256
        and source_state.get("active_attempt") == SOURCE
        and source_state.get("active_attempt_path") == _relative(source, repo)
        and source_state.get("current_state") == RESUME
        and source_state.get("confirmation_terminal") is False
        and _exact_typed_mapping(
            {key: source_state.get(key) for key in EXPECTED_COUNTS},
            EXPECTED_COUNTS,
        )
        and source_state.get("ledger_event_count")
        == source_ledger.get("event_count")
        and source_state.get("ledger_head_sha256")
        == source_ledger.get("head_sha256"),
        "v007 selection-boundary state/ledger drift",
    )
    source_count = int(source_ledger["event_count"])
    ledger_lines = (study / "RESEARCH_LEDGER.jsonl").read_bytes().splitlines(
        keepends=True
    )
    prefix = b"".join(ledger_lines[:source_count])
    _require(
        len(live_events) >= source_count
        and _hash_bytes(prefix) == EXPECTED_SOURCE_LEDGER_SHA256
        and live_events[source_count - 1].get("record_sha256")
        == source_ledger.get("head_sha256"),
        "v007 ledger prefix provenance drift",
    )

    lineage = source_state.get("version_forward_lineage")
    history = source_state.get("attempt_history")
    expected_versions = [
        SCIENCE_ATTEMPT,
        "v002",
        FIT_SOURCE,
        INTERMEDIATE,
        SOURCE_PARENT,
        SELECTION_SOURCE,
        SOURCE,
    ]
    _require(
        isinstance(lineage, list)
        and len(lineage) == 6
        and lineage[-1].get("old_attempt") == SELECTION_SOURCE
        and lineage[-1].get("new_attempt") == SOURCE
        and lineage[-1].get("resume_state") == "SELECTION_COHORTS"
        and lineage[-1].get("equivalence_sha256")
        == EXPECTED_SOURCE_SEAL_SHA256
        and isinstance(history, list)
        and [item.get("version") for item in history if isinstance(item, Mapping)]
        == expected_versions
        and source_state.get("completed_states")
        == [
            "BOOTSTRAP_AUDIT",
            "DIAGNOSTIC_ACCOUNT",
            "PREREGISTRATION_AND_POWER",
            "IMPLEMENTATION_COMPLETE",
            "PRESEAL_QUALIFICATION",
            "PRE_OUTCOME_SEAL",
            "FIT_COHORTS",
            "FIT_LOCK",
            "PRE_SELECTION_SEAL",
        ],
        "v001-through-v007 deterministic chronology drift",
    )
    checkpoints = source_state.get("verified_checkpoints")
    _require(
        isinstance(checkpoints, list) and len(checkpoints) == 9,
        "v007 checkpoint census drift",
    )
    for checkpoint in checkpoints:
        _require(
            isinstance(checkpoint, Mapping)
            and isinstance(checkpoint.get("evidence_path"), str)
            and _resolve(repo, checkpoint["evidence_path"]).is_file()
            and file_hash(_resolve(repo, checkpoint["evidence_path"]))
            == checkpoint.get("evidence_sha256"),
            "v007 checkpoint evidence hash drift",
        )
    pre_selection = _object(pre_selection_path)
    _require(
        pre_selection.get("attempt") == SELECTION_SOURCE
        and pre_selection.get("checkpoint_state") == "PRE_SELECTION_SEAL"
        and pre_selection.get("passed") is True
        and _exact_typed_mapping(
            pre_selection.get("outcome_counts_at_seal"), EXPECTED_COUNTS
        )
        and source_state.get("last_verified_checkpoint", {}).get(
            "evidence_sha256"
        )
        == EXPECTED_SOURCE_PRE_SELECTION_SEAL_SHA256,
        "v006 pre-selection checkpoint drift",
    )
    marker = _source_adapter_marker(
        source_state.get("v007_durable_controller_adapter"),
        source_state=source_state,
        source=source,
        study=study,
        repo=repo,
        event_count=source_count,
        head_sha256=str(source_ledger["head_sha256"]),
    )
    genesis = {
        "path": _relative(study / "LEDGER_CHAIN_GENESIS.json", repo),
        "sha256": file_hash(study / "LEDGER_CHAIN_GENESIS.json"),
        "bytes": (study / "LEDGER_CHAIN_GENESIS.json").stat().st_size,
    }
    return {
        "state": source_state,
        "lineage": json.loads(json.dumps(lineage)),
        "history": json.loads(json.dumps(history)),
        "marker": marker,
        "ledger": dict(source_ledger),
        "genesis": genesis,
    }


def _validate_post_forward_raw_authorization_state(
    state: Mapping[str, Any],
    *,
    authorized_early_verifier_state: str | None = None,
    authorized_role_count_recovery: str | None = None,
) -> None:
    """Accept raw states plus one explicitly selected exceptional read-only mode."""

    _require(state.get("active_attempt") == TARGET, "post-forward active attempt drift")
    current = state.get("current_state")
    raw_states = {"SELECTION_COHORTS"}
    early_states = {
        "CANDIDATE_SELECTION",
        "CONFIRMATION_POWER_AND_COHORT_FREEZE",
    }
    recovery_states = {"selection": "SELECTION_COHORTS"}
    _require(
        not (
            authorized_early_verifier_state is not None
            and authorized_role_count_recovery is not None
        ),
        "post-forward exceptional authorization modes are mutually exclusive",
    )
    if authorized_early_verifier_state is None and authorized_role_count_recovery is None:
        _require(
            current in raw_states,
            "post-forward inherited authorization state is not selection raw",
        )
    elif authorized_early_verifier_state is not None:
        _require(
            authorized_early_verifier_state in early_states
            and current == authorized_early_verifier_state,
            "post-forward authorized early-verifier state drift",
        )
    else:
        _require(
            authorized_role_count_recovery in recovery_states
            and current == recovery_states[authorized_role_count_recovery],
            "post-forward authorized role-count recovery state drift",
        )

    expected_fit = 1200
    expected_selection = (
        2000
        if current in early_states or authorized_role_count_recovery == "selection"
        else 0
    )
    _require(
        type(state.get("fit_outcome_episodes")) is int
        and state.get("fit_outcome_episodes") == expected_fit
        and type(state.get("selection_outcome_episodes")) is int
        and state.get("selection_outcome_episodes") == expected_selection
        and type(state.get("smoke_outcome_episodes")) is int
        and state.get("smoke_outcome_episodes") == 0
        and type(state.get("confirmation_outcome_episodes_generated")) is int
        and state.get("confirmation_outcome_episodes_generated") == 0
        and type(state.get("confirmation_outcome_episodes_executed")) is int
        and state.get("confirmation_outcome_episodes_executed") == 0
        and state.get("confirmation_outcomes_opened_for_analysis") is False,
        "post-forward raw-authorization counters drift",
    )
    _require(
        type(state.get("expected_fit_episode_count")) is int
        and state.get("expected_fit_episode_count") == 1200,
        "post-fit authorization lacks the fixed completed fit cohort",
    )
    if current in early_states or authorized_role_count_recovery == "selection":
        _require(
            type(state.get("expected_selection_episode_count")) is int
            and state.get("expected_selection_episode_count") == 2000,
            "post-selection authorization lacks the fixed completed selection cohort",
        )
    if current in early_states:
        state_machine = (
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
        _require(
            tuple(state.get("state_machine", ())) == state_machine,
            "early-verifier state-machine definition drift",
        )
        current_index = state_machine.index(str(current))
        _require(
            state.get("completed_states") == list(state_machine[:current_index]),
            "early-verifier completed-state prefix drift",
        )
        _require(
            state.get("early_scientific_failure") is None
            and state.get("skipped_states") in (None, []),
            "early-verifier guard was invoked after early-path mutation",
        )


def _source_ast_sha256(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _hash_bytes(ast.dump(tree, include_attributes=False).encode("utf-8"))


def _verify_recovery_adapter_marker(
    value: Any,
    *,
    event_count: int,
    head_sha256: str,
    target: Path,
    repo: Path,
    label: str,
    state_without_marker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version", "authorization_kind", "adapter_source_path",
        "adapter_source_sha256", "adapter_source_ast_sha256",
        "root_program_path", "root_program_sha256",
        "root_program_ast_sha256", "ledger_event_count",
        "ledger_head_sha256", "operation_sha256", "state_binding_sha256",
    }
    adapter = target / "version_forward_transaction.py"
    root_program = target.parents[1] / "program.py"
    operation = value.get("operation_sha256") if isinstance(value, Mapping) else None
    state_binding = (
        value.get("state_binding_sha256") if isinstance(value, Mapping) else None
    )
    marker_without_state_binding = dict(value) if isinstance(value, Mapping) else {}
    marker_without_state_binding.pop("state_binding_sha256", None)
    expected_state_binding = (
        _json_hash(
            {
                "marker_without_state_binding": marker_without_state_binding,
                "state_without_marker": dict(state_without_marker),
            }
        )
        if state_without_marker is not None
        else None
    )
    _require(
        isinstance(value, Mapping)
        and set(value) == expected_keys
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 2
        and value.get("authorization_kind")
        == "receipt_bound_v008_root_controller_adapter"
        and value.get("adapter_source_path") == _relative(adapter, repo)
        and value.get("adapter_source_sha256") == file_hash(adapter)
        and value.get("adapter_source_ast_sha256") == _source_ast_sha256(adapter)
        and value.get("root_program_path") == _relative(root_program, repo)
        and value.get("root_program_sha256") == file_hash(root_program)
        and value.get("root_program_ast_sha256")
        == _source_ast_sha256(root_program)
        and type(value.get("ledger_event_count")) is int
        and value.get("ledger_event_count") == event_count
        and value.get("ledger_head_sha256") == head_sha256
        and isinstance(operation, str)
        and len(operation) == 64
        and all(character in "0123456789abcdef" for character in operation)
        and isinstance(state_binding, str)
        and len(state_binding) == 64
        and all(
            character in "0123456789abcdef" for character in state_binding
        )
        and (
            expected_state_binding is None
            or state_binding == expected_state_binding
        ),
        f"{label} v008 controller adapter marker drift",
    )
    return dict(value)


def _load_active_transaction_receipt(
    *,
    target: Path,
    authorized_transaction_context_sha256: str | None,
) -> tuple[dict[str, Any], bool]:
    """Load either the published receipt or its exact durable prepublication journal."""

    receipt_path = target / "audit/version_forward_transaction_receipt.json"
    if os.path.lexists(receipt_path):
        receipt = _object(receipt_path)
        _require(
            receipt_path.read_bytes()
            == (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
            "descendant adapter receipt is noncanonical",
        )
        if authorized_transaction_context_sha256 is not None:
            _require(
                receipt.get("receipt_context_sha256")
                == authorized_transaction_context_sha256,
                "published receipt differs from transient transaction authorization",
            )
        return receipt, True

    _require(
        isinstance(authorized_transaction_context_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", authorized_transaction_context_sha256)
        is not None,
        "post-forward verification lacks a published receipt or transient authorization",
    )
    journal_path = target.parents[1] / "VERSION_FORWARD_TRANSACTION.json"
    _require(os.path.lexists(journal_path), "transient receipt journal is absent")
    journal = _object(journal_path)
    _require(
        journal_path.read_bytes()
        == (json.dumps(journal, indent=2, sort_keys=True) + "\n").encode(),
        "transient receipt journal is noncanonical",
    )
    expected_keys = {
        "schema_version", "artifact_type", "authorization_kind",
        "receipt_context", "receipt_context_sha256", "controller_proposal",
        "predicted_controller_transaction", "predicted_post_snapshot",
        "predicted_post_verifiers", "predicted_receipt",
        "predicted_receipt_sha256", "transaction_sha256",
    }
    digest_projection = dict(journal)
    transaction_sha256 = digest_projection.pop("transaction_sha256", None)
    receipt = journal.get("predicted_receipt")
    _require(
        set(journal) == expected_keys
        and type(journal.get("schema_version")) is int
        and journal.get("schema_version") == 1
        and journal.get("artifact_type")
        == "v008_version_forward_receipt_context_journal"
        and journal.get("authorization_kind")
        == "receipt_bound_v008_root_controller_adapter"
        and journal.get("receipt_context_sha256")
        == authorized_transaction_context_sha256
        and _json_hash(journal.get("receipt_context"))
        == authorized_transaction_context_sha256
        and transaction_sha256 == _json_hash(digest_projection)
        and isinstance(receipt, Mapping)
        and journal.get("predicted_receipt_sha256")
        == _hash_bytes(
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
        )
        and receipt.get("receipt_context") == journal.get("receipt_context")
        and receipt.get("receipt_context_sha256")
        == authorized_transaction_context_sha256
        and receipt.get("post_snapshot")
        == journal.get("predicted_post_snapshot")
        and receipt.get("post_verifiers")
        == journal.get("predicted_post_verifiers")
        and journal.get("controller_proposal")
        == journal.get("receipt_context", {}).get("controller_proposal")
        and journal.get("predicted_post_verifiers")
        == journal.get("receipt_context", {}).get("predicted_post_verifiers"),
        "transient receipt journal projection drift",
    )
    return dict(receipt), False


def _verify_initial_transaction_receipt_anchor(
    *,
    receipt: Mapping[str, Any],
    published: bool,
    source_state: Mapping[str, Any],
    source_ledger: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    ledger_payload: bytes,
    target: Path,
    repo: Path,
    seal_path: Path,
    allow_transaction_journal: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the exact initial forward and return its authenticated post tip."""

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
    context = receipt.get("receipt_context")
    journal_created = receipt.get("journal_created_unix_ns")
    forward_created = receipt.get("forward_created_unix_ns")
    receipt_created = receipt.get("created_unix_ns")
    zero = {
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    _require(
        set(receipt) == top_keys
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 2
        and receipt.get("artifact_type")
        == "atomic_zero_confirmation_outcome_version_forward_transaction_receipt"
        and receipt.get("authorization_kind")
        == "locked_v007_to_v008_inherited_pre_selection_forward"
        and receipt.get("source_attempt") == SOURCE
        and receipt.get("target_attempt") == TARGET
        and receipt.get("resume_state") == RESUME
        and type(journal_created) is int
        and journal_created > 0
        and type(forward_created) is int
        and forward_created == journal_created + 1
        and type(receipt_created) is int
        and receipt_created == journal_created + 2
        and isinstance(context, Mapping)
        and receipt.get("receipt_context_sha256") == _json_hash(context)
        and _exact_typed_mapping(receipt.get("outcome_counts"), zero)
        and receipt.get("state_transaction_absent") is True
        and receipt.get("receipt_journal_absent") is True
        and receipt.get("passed") is True,
        "active transaction receipt header/type drift",
    )
    context_keys = {
        "schema_version", "source_attempt", "target_attempt", "resume_state",
        "journal_created_unix_ns", "forward_created_unix_ns",
        "receipt_created_unix_ns", "fixed_inputs", "pre_snapshot",
        "pre_verifiers", "controller_proposal", "predicted_post_verifiers",
    }
    fixed_names = {
        "seal", "invalidity", "invalidity_draft", "prior_receipt",
        "root_program", "transaction_source",
    }
    fixed_inputs = context.get("fixed_inputs") if isinstance(context, Mapping) else None
    _require(
        set(context) == context_keys
        and type(context.get("schema_version")) is int
        and context.get("schema_version") == 1
        and context.get("source_attempt") == SOURCE
        and context.get("target_attempt") == TARGET
        and context.get("resume_state") == RESUME
        and context.get("journal_created_unix_ns") == journal_created
        and context.get("forward_created_unix_ns") == forward_created
        and context.get("receipt_created_unix_ns") == receipt_created
        and isinstance(fixed_inputs, Mapping)
        and set(fixed_inputs) == fixed_names
        and all(
            fixed_inputs.get(name) == receipt.get(name) for name in fixed_names
        )
        and context.get("pre_snapshot") == receipt.get("pre_snapshot")
        and context.get("pre_verifiers") == receipt.get("pre_verifiers")
        and context.get("predicted_post_verifiers")
        == receipt.get("post_verifiers"),
        "active transaction receipt context projection drift",
    )

    basic_record_keys = {"path", "sha256", "bytes", "device", "inode", "nlink"}

    def validate_record(
        value: Any, *, expected_path: Path, ast_bound: bool = False
    ) -> Mapping[str, Any]:
        keys = set(basic_record_keys)
        if ast_bound:
            keys.add("ast_sha256")
        _require(
            isinstance(value, Mapping)
            and set(value) == keys
            and value.get("path") == _relative(expected_path, repo)
            and isinstance(value.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
            and all(
                type(value.get(key)) is int and value[key] >= 0
                for key in ("bytes", "device", "inode", "nlink")
            )
            and value.get("nlink") == 1
            and (
                not ast_bound
                or value.get("ast_sha256") == _source_ast_sha256(expected_path)
            ),
            f"active transaction receipt file record drift: {expected_path.name}",
        )
        return value

    validate_record(receipt["seal"], expected_path=seal_path)
    _require(
        receipt["seal"].get("sha256") == file_hash(seal_path),
        "active receipt inheritance-seal hash drift",
    )
    root_program = target.parents[1] / "program.py"
    transaction_source = target / "version_forward_transaction.py"
    for name, path in (
        ("root_program", root_program),
        ("transaction_source", transaction_source),
    ):
        record = validate_record(receipt[name], expected_path=path, ast_bound=True)
        _require(
            record.get("sha256") == file_hash(path)
            and record.get("bytes") == path.stat().st_size,
            f"active receipt live source drift: {name}",
        )

    pre = receipt.get("pre_snapshot")
    post = receipt.get("post_snapshot")
    _require(
        isinstance(pre, Mapping)
        and set(pre) == {"state", "ledger", "genesis"}
        and isinstance(post, Mapping)
        and set(post) == {"state", "ledger", "genesis"},
        "active receipt snapshot schema drift",
    )

    def validate_state_record(
        value: Any, expected_state: Mapping[str, Any], label: str
    ) -> dict[str, Any]:
        _require(
            isinstance(value, Mapping)
            and set(value)
            == {"path", "sha256", "bytes", "object_sha256", "object"}
            and value.get("path") == _relative(target.parents[1] / "STATE.json", repo)
            and type(value.get("bytes")) is int
            and value["bytes"] >= 0
            and isinstance(value.get("object"), Mapping),
            f"active receipt {label} STATE record drift",
        )
        state_object = dict(value["object"])
        pretty = (json.dumps(state_object, indent=2, sort_keys=True) + "\n").encode()
        _require(
            state_object == dict(expected_state)
            and value.get("sha256") == _hash_bytes(pretty)
            and value.get("bytes") == len(pretty)
            and value.get("object_sha256") == _json_hash(state_object),
            f"active receipt {label} STATE content drift",
        )
        return state_object

    pre_state = validate_state_record(pre["state"], source_state, "pre-forward")
    pre_ledger = pre.get("ledger")
    post_ledger = post.get("ledger")
    ledger_keys = {
        "path", "sha256", "bytes", "event_count", "head_sha256",
        "ledger_sha256",
    }
    _require(
        isinstance(pre_ledger, Mapping)
        and set(pre_ledger) == ledger_keys
        and isinstance(post_ledger, Mapping)
        and set(post_ledger) == ledger_keys
        and all(
            type(record.get(key)) is int and record[key] >= 0
            for record in (pre_ledger, post_ledger)
            for key in ("bytes", "event_count")
        )
        and pre_ledger.get("path")
        == post_ledger.get("path")
        == _relative(target.parents[1] / "RESEARCH_LEDGER.jsonl", repo)
        and pre_ledger.get("event_count") == source_ledger.get("event_count")
        and pre_ledger.get("head_sha256") == source_ledger.get("head_sha256")
        and pre_ledger.get("sha256") == source_ledger.get("sha256")
        and pre_ledger.get("bytes") == source_ledger.get("bytes")
        and pre_ledger.get("ledger_sha256") == pre_ledger.get("sha256"),
        "active receipt pre-ledger/source-tip drift",
    )
    pre_count = int(pre_ledger["event_count"])
    post_count = post_ledger.get("event_count")
    lines = ledger_payload.splitlines(keepends=True)
    _require(
        type(post_count) is int
        and post_count == pre_count + 1
        and len(lines) == len(events)
        and post_count <= len(lines)
        and all(line.endswith(b"\n") for line in lines)
        and pre_ledger.get("bytes") == len(b"".join(lines[:pre_count]))
        and post_ledger.get("bytes") == len(b"".join(lines[:post_count]))
        and pre_ledger.get("sha256") == _hash_bytes(b"".join(lines[:pre_count]))
        and post_ledger.get("sha256") == _hash_bytes(b"".join(lines[:post_count]))
        and post_ledger.get("ledger_sha256") == post_ledger.get("sha256")
        and post_ledger.get("head_sha256")
        == events[post_count - 1].get("record_sha256"),
        "active receipt pre/post ledger prefix drift",
    )
    for snapshot in (pre, post):
        genesis = snapshot.get("genesis")
        _require(
            isinstance(genesis, Mapping)
            and set(genesis) == {"path", "sha256", "bytes"}
            and genesis.get("path")
            == _relative(target.parents[1] / "LEDGER_CHAIN_GENESIS.json", repo)
            and type(genesis.get("bytes")) is int
            and genesis["bytes"] >= 0
            and genesis.get("sha256")
            == file_hash(target.parents[1] / "LEDGER_CHAIN_GENESIS.json"),
            "active receipt genesis drift",
        )
    _require(pre.get("genesis") == post.get("genesis"), "receipt genesis identity drift")

    forward = dict(events[post_count - 1])
    expected_record_keys = {
        "event", "attempt", "old_attempt", "new_attempt", "resume_state",
        "invalidity_path", "invalidity_sha256", "equivalence_path",
        "equivalence_sha256", "inherited_verified_checkpoints",
        "attempt_parameterization_verified", "source_controller_adapter_marker",
        "created_unix_ns", "version_forward_receipt_context_sha256",
        "seq", "prev_sha256", "record_sha256",
    }
    _require(
        set(forward) == expected_record_keys
        and forward.get("event") == "zero_confirmation_outcome_version_forward"
        and forward.get("attempt") == TARGET
        and forward.get("old_attempt") == SOURCE
        and forward.get("new_attempt") == TARGET
        and forward.get("resume_state") == RESUME
        and forward.get("attempt_parameterization_verified") is True
        and type(forward.get("created_unix_ns")) is int
        and forward.get("created_unix_ns") == forward_created
        and type(forward.get("seq")) is int
        and forward.get("seq") == post_count
        and forward.get("prev_sha256") == pre_ledger.get("head_sha256")
        and forward.get("version_forward_receipt_context_sha256")
        == receipt.get("receipt_context_sha256"),
        "active receipt forward record schema/content drift",
    )
    record_without_hash = dict(forward)
    record_sha256 = record_without_hash.pop("record_sha256")
    _require(
        record_sha256 == _json_hash(record_without_hash)
        and lines[post_count - 1]
        == json.dumps(forward, sort_keys=True, separators=(",", ":")).encode()
        + b"\n",
        "active receipt forward record byte/hash drift",
    )
    edge = {
        key: forward[key]
        for key in (
            "old_attempt", "new_attempt", "resume_state", "invalidity_path",
            "invalidity_sha256", "equivalence_path", "equivalence_sha256",
            "inherited_verified_checkpoints", "attempt_parameterization_verified",
        )
    }
    expected_proposal_state = json.loads(json.dumps(pre_state))
    source_marker = expected_proposal_state.pop(
        "v007_durable_controller_adapter", None
    )
    _require(
        source_marker == forward.get("source_controller_adapter_marker"),
        "active receipt source adapter marker drift",
    )
    active_annotation = {
        "attempt": TARGET,
        "equivalence_evidence_path": _relative(seal_path, repo),
        "equivalence_evidence_sha256": file_hash(seal_path),
    }
    checkpoints = expected_proposal_state.get("verified_checkpoints")
    _require(isinstance(checkpoints, list) and bool(checkpoints), "source checkpoints absent")
    for checkpoint in checkpoints:
        _require(isinstance(checkpoint, Mapping), "source checkpoint schema drift")
        inherited_into = checkpoint.setdefault("inherited_into_attempts", [])
        _require(isinstance(inherited_into, list), "checkpoint inheritance list drift")
        inherited_into.append(json.loads(json.dumps(active_annotation)))
    expected_proposal_state["last_verified_checkpoint"] = checkpoints[-1]
    history = expected_proposal_state.get("attempt_history")
    _require(isinstance(history, list) and len(history) == 7, "source history drift")
    history[6]["status"] = "invalid_zero_confirmation_outcome_procedural"
    history[6]["invalidity_evidence_path"] = forward["invalidity_path"]
    history[6]["invalidity_evidence_sha256"] = forward["invalidity_sha256"]
    history.append(
        {
            "version": TARGET,
            "path": _relative(target, repo),
            "status": "active_zero_confirmation_outcome_version_forward",
            "created_unix_ns": forward_created,
            "version_forward_evidence_path": _relative(seal_path, repo),
            "version_forward_evidence_sha256": file_hash(seal_path),
            "attempt_parameterization_verified": True,
        }
    )
    expected_proposal_state["active_attempt"] = TARGET
    expected_proposal_state["active_attempt_path"] = _relative(target, repo)
    prior_lineage = expected_proposal_state.get("version_forward_lineage")
    _require(isinstance(prior_lineage, list), "source version-forward lineage drift")
    expected_proposal_state["version_forward_lineage"] = prior_lineage + [edge]
    expected_proposal_state["updated_unix_ns"] = forward_created
    expected_raw_event = dict(forward)
    for key in (
        "version_forward_receipt_context_sha256", "seq", "prev_sha256",
        "record_sha256",
    ):
        expected_raw_event.pop(key)
    proposal = context.get("controller_proposal")
    _require(
        isinstance(proposal, Mapping)
        and set(proposal) == {"state", "events"}
        and proposal.get("state") == expected_proposal_state
        and proposal.get("events") == [expected_raw_event],
        "active receipt exact controller proposal drift",
    )

    intended_without_marker = json.loads(json.dumps(expected_proposal_state))
    intended_without_marker["ledger_event_count"] = post_count
    intended_without_marker["ledger_head_sha256"] = record_sha256
    observed_post = post.get("state")
    _require(
        isinstance(observed_post, Mapping)
        and isinstance(observed_post.get("object"), Mapping),
        "active receipt post STATE object absent",
    )
    post_state = dict(observed_post["object"])
    post_without_marker = json.loads(json.dumps(post_state))
    marker = post_without_marker.pop("v008_durable_controller_adapter", None)
    _require(
        post_without_marker == intended_without_marker,
        "active receipt post STATE differs from exact proposal transition",
    )
    verified_marker = _verify_recovery_adapter_marker(
        marker,
        event_count=post_count,
        head_sha256=str(record_sha256),
        target=target,
        repo=repo,
        label="receipt post STATE",
        state_without_marker=post_without_marker,
    )
    expected_operation = _json_hash(
        {
            "base_state_object_sha256": _json_hash(pre_state),
            "base_ledger_sha256": pre_ledger["sha256"],
            "expected_suffix_sha256": _hash_bytes(lines[post_count - 1]),
            "intended_state_object_sha256": _json_hash(post_without_marker),
        }
    )
    _require(
        verified_marker.get("operation_sha256") == expected_operation,
        "active receipt post marker operation drift",
    )
    validate_state_record(observed_post, post_state, "post-forward")
    _require(
        receipt.get("controller_return") == post_state
        and receipt.get("controller_return_sha256") == _json_hash(post_state),
        "active receipt controller-return binding drift",
    )
    if published and not allow_transaction_journal:
        _require(
            not os.path.lexists(target.parents[1] / "VERSION_FORWARD_TRANSACTION.json"),
            "published receipt retains a transaction journal",
        )
    return post_state, dict(post_ledger)


def _verify_descendant_controller_transition(
    *,
    base_state: Mapping[str, Any],
    proposed_state: Mapping[str, Any],
    controller_events: Sequence[Mapping[str, Any]],
    repo: Path,
) -> None:
    """Recheck the bounded root-controller transition behind one adapter group."""

    def active_json_object(relative: Any, label: str) -> tuple[Path, dict[str, Any]]:
        _require(isinstance(relative, str), f"descendant {label} path is absent")
        try:
            path = (repo / str(relative)).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise VerificationError(f"descendant {label} is absent") from error
        active_root = (
            repo / f"runs/lewm_domain_robust_gate/attempts/{TARGET}"
        ).resolve()
        _require(
            path.is_relative_to(active_root),
            f"descendant {label} escapes the active attempt",
        )
        try:
            value = _object(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError(f"descendant {label} is unreadable") from error
        return path, value

    def proves_integrity_failure(value: Any) -> bool:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered in {
                    "execution_invalid", "integrity_failure", "integrity_failed",
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

    base = json.loads(json.dumps(dict(base_state)))
    base_marker = base.pop("v008_durable_controller_adapter", None)
    proposed = json.loads(json.dumps(dict(proposed_state)))
    sequence = [event.get("event") for event in controller_events]
    state_machine = (
        "BOOTSTRAP_AUDIT", "DIAGNOSTIC_ACCOUNT",
        "PREREGISTRATION_AND_POWER", "IMPLEMENTATION_COMPLETE",
        "PRESEAL_QUALIFICATION", "PRE_OUTCOME_SEAL", "FIT_COHORTS",
        "FIT_LOCK", "PRE_SELECTION_SEAL", "SELECTION_COHORTS",
        "CANDIDATE_SELECTION", "GATE_FREEZE",
        "CONFIRMATION_POWER_AND_COHORT_FREEZE",
        "PRE_CONFIRMATION_PACKAGE_SEAL", "EXCLUDED_MECHANICAL_SMOKE",
        "CONFIRMATION_GENERATION", "CONFIRMATION_EXECUTION",
        "CONFIRMATION_INPUT_SEAL", "SEALED_ANALYSIS",
        "LATENCY_AND_RESOURCE_REPORTING", "INDEPENDENT_VERIFICATION",
        "TERMINAL", "POST_TERMINAL_REPORTING",
    )
    early_specs = {
        "no_candidate": (
            "CANDIDATE_SELECTION", "selection/selection_ledger.json",
            "verifier_contract_no_candidate.json",
        ),
        "power_infeasible": (
            "CONFIRMATION_POWER_AND_COHORT_FREEZE", "power_analysis.json",
            "verifier_contract_power_infeasible.json",
        ),
    }
    integrity_specs = {
        "SEALED_ANALYSIS": (
            "analysis_execution", "audit/analysis_execution_invalid.json",
            ["analysis_execution_invalid", "analysis_execution", "analysis_failure"],
        ),
        "LATENCY_AND_RESOURCE_REPORTING": (
            "latency_resource", "metrics/latency_and_resources.json",
            ["latency_resource", "latency_and_resources", "latency_report"],
        ),
    }
    _require(
        controller_events
        and all(
            type(event.get("created_unix_ns")) is int
            and event.get("created_unix_ns") > 0
            and event.get("attempt") == TARGET
            for event in controller_events
        ),
        "descendant controller event identity/type drift",
    )

    if sequence == ["outcome_counts_updated"]:
        event = controller_events[0]
        fields = event.get("fields")
        allowed = {
            "fit_outcome_episodes", "selection_outcome_episodes",
            "smoke_outcome_episodes", "confirmation_outcome_episodes_generated",
            "confirmation_outcome_episodes_executed",
            "confirmation_outcomes_opened_for_analysis",
        }
        _require(
            set(event)
            == {
                "event", "attempt", "fields", "prior_controller_adapter_marker",
                "created_unix_ns",
            }
            and isinstance(fields, Mapping)
            and len(fields) == 1
            and set(fields).issubset(allowed)
            and event.get("prior_controller_adapter_marker") == base_marker,
            "descendant count event closed schema drift",
        )
        field, value = next(iter(fields.items()))
        _require(
            (
                type(value) is bool
                if field == "confirmation_outcomes_opened_for_analysis"
                else type(value) is int and value >= 0
            ),
            "descendant count value type drift",
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
                (type(value) is bool and type(old) is bool and not (old and not value))
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
            "descendant count monotonicity/state/maximum drift",
        )
        expected = json.loads(json.dumps(base))
        expected[field] = value
        expected["updated_unix_ns"] = event["created_unix_ns"]
        _require(
            proposed == expected,
            "descendant count base-to-target transition drift",
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
            and base.get("early_scientific_failure") is None
            and base.get("terminal_label") is None
            and isinstance(event.get("terminal_label"), str)
            and event.get("terminal_label")
            in {
                "domain_robust_gate_confirmed", "domain_robust_gate_partial",
                "domain_robust_gate_failed",
            }
            and type(event.get("process_valid")) is bool
            and isinstance(event.get("decision_path"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(event.get("decision_sha256")))
            is not None,
            "descendant terminal-decision event closed schema drift",
        )
        decision, decision_object = active_json_object(
            event["decision_path"], "terminal decision"
        )
        _require(
            file_hash(decision) == event["decision_sha256"]
            and decision_object.get("attempt") == TARGET
            and decision_object.get("terminal_label")
            == event["terminal_label"]
            and type(decision_object.get("process_valid")) is bool
            and decision_object.get("process_valid") is event["process_valid"],
            "descendant terminal decision evidence drift",
        )
        expected = json.loads(json.dumps(base))
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
            proposed == expected,
            "descendant terminal-decision base-to-target transition drift",
        )
        return

    if sequence == ["preregistered_early_scientific_failure_staged"]:
        event = controller_events[0]
        keys = {
            "event", "attempt", "mode", "trigger_state",
            "trigger_evidence_path", "trigger_evidence_sha256",
            "verifier_contract_path", "verifier_contract_sha256",
            "skipped_states", "smoke_outcome_episodes",
            "confirmation_outcome_episodes_generated",
            "confirmation_outcome_episodes_executed",
            "confirmation_outcomes_opened_for_analysis", "created_unix_ns",
        }
        mode_spec = early_specs.get(event.get("mode"))
        trigger, source_suffix, contract_suffix = (
            mode_spec if mode_spec is not None else (None, None, None)
        )
        exact_skipped = (
            list(
                state_machine[
                    state_machine.index(str(trigger)) + 1:
                    state_machine.index("INDEPENDENT_VERIFICATION")
                ]
            )
            if trigger in state_machine
            else None
        )
        expected_source_path = (
            f"runs/lewm_domain_robust_gate/attempts/{TARGET}/{source_suffix}"
            if source_suffix is not None
            else None
        )
        expected_contract_path = (
            f"runs/lewm_domain_robust_gate/attempts/{TARGET}/{contract_suffix}"
            if contract_suffix is not None
            else None
        )
        _require(
            set(event) == keys
            and mode_spec is not None
            and event.get("trigger_state") == trigger == base.get("current_state")
            and event.get("trigger_evidence_path") == expected_source_path
            and event.get("verifier_contract_path") == expected_contract_path
            and event.get("skipped_states") == exact_skipped
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
            and type(base.get("confirmation_outcome_episodes_generated")) is int
            and base.get("confirmation_outcome_episodes_generated") == 0
            and type(base.get("confirmation_outcome_episodes_executed")) is int
            and base.get("confirmation_outcome_episodes_executed") == 0
            and base.get("confirmation_outcomes_opened_for_analysis") is False,
            "descendant early-failure event closed schema drift",
        )
        source_file, source_object = active_json_object(
            event["trigger_evidence_path"], "early-failure source"
        )
        contract_file, contract_object = active_json_object(
            event["verifier_contract_path"], "early-failure verifier contract"
        )
        contract_path_key = (
            "selection_ledger"
            if event["mode"] == "no_candidate"
            else "power_freeze"
        )
        _require(
            file_hash(source_file) == event["trigger_evidence_sha256"]
            and file_hash(contract_file) == event["verifier_contract_sha256"]
            and type(contract_object.get("schema_version")) is int
            and contract_object.get("schema_version") == 1
            and contract_object.get("attempt") == TARGET
            and contract_object.get("attempt_root")
            == f"runs/lewm_domain_robust_gate/attempts/{TARGET}"
            and contract_object.get("mode") == event["mode"]
            and isinstance(contract_object.get("paths"), Mapping)
            and contract_object["paths"].get(contract_path_key)
            == event["trigger_evidence_path"],
            "descendant early-failure contract binding drift",
        )
        if event["mode"] == "no_candidate":
            source_semantics = (
                source_object.get("status")
                == "selection_complete_no_selected_head_refit"
                and type(source_object.get("candidate_count")) is int
                and source_object.get("candidate_count") == 24
                and type(source_object.get("eligible_count")) is int
                and source_object.get("eligible_count") == 0
                and source_object.get("selected_candidate_id") is None
                and source_object.get("selected_candidate_index") is None
                and source_object.get("selected_head_refit_after_selection") is False
                and type(
                    source_object.get(
                        "prior_confirmation_outcome_episodes_used", 0
                    )
                ) is int
                and source_object.get(
                    "prior_confirmation_outcome_episodes_used", 0
                ) == 0
            )
        else:
            source_semantics = (
                source_object.get("status")
                == "binding_post_selection_power_result"
                and source_object.get("decision")
                == "power_infeasible_no_confirmation"
                and source_object.get("feasible") is False
                and source_object.get("passed") is False
                and source_object.get(
                    "confirmation_generation_authorized_by_power"
                ) is False
                and source_object.get(
                    "selected_confirmation_episodes_per_regime"
                ) is None
                and source_object.get("confirmation_episode_count_per_regime")
                is None
                and source_object.get("terminal_label_if_infeasible")
                == "domain_robust_gate_failed"
                and type(
                    source_object.get("fresh_confirmation_outcomes_opened")
                ) is int
                and source_object.get("fresh_confirmation_outcomes_opened") == 0
            )
        _require(
            source_semantics,
            "descendant early-failure source eligibility drift",
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
            "descendant early-failure role completeness drift",
        )
        active_root = repo / f"runs/lewm_domain_robust_gate/attempts/{TARGET}"
        forbidden_roots = (
            "data/smoke", "data/confirmation", "execution/smoke",
            "execution/confirmation", "metrics/smoke", "metrics/confirmation",
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
            and all(not (active_root / relative).exists() for relative in forbidden_files),
            "descendant early-failure later-role artifact drift",
        )
        created = event["created_unix_ns"]
        mode = event["mode"]
        expected = json.loads(json.dumps(base))
        expected["early_scientific_failure"] = {
            "mode": mode,
            "status": "awaiting_independent_verification",
            "trigger_state": event["trigger_state"],
            "trigger_evidence_path": event["trigger_evidence_path"],
            "trigger_evidence_sha256": event["trigger_evidence_sha256"],
            "verifier_contract_path": event["verifier_contract_path"],
            "verifier_contract_sha256": event["verifier_contract_sha256"],
            "skipped_states": event["skipped_states"],
            "staged_unix_ns": created,
            "required_terminal_label": "domain_robust_gate_failed",
            "required_process_valid": True,
        }
        expected["skipped_states"] = [
            {
                "state": skipped,
                "mode": mode,
                "reason": "preregistered_process_valid_early_scientific_failure",
                "trigger_evidence_path": event["trigger_evidence_path"],
                "trigger_evidence_sha256": event["trigger_evidence_sha256"],
                "recorded_unix_ns": created,
            }
            for skipped in event["skipped_states"]
        ]
        expected["current_state"] = "INDEPENDENT_VERIFICATION"
        expected["next_action"] = (
            "run the standalone read-only verifier in the exact staged mode and "
            "capture audit/independent_verification.json before finalizing"
        )
        expected["updated_unix_ns"] = created
        _require(
            proposed == expected,
            "descendant early-failure base-to-target transition drift",
        )
        return

    if sequence == ["state_completed"]:
        event = controller_events[0]
        ordinary_keys = {
            "event", "attempt", "completed_state", "next_state",
            "checkpoint_name", "evidence_path", "evidence_sha256",
            "created_unix_ns",
        }
        integrity_keys = ordinary_keys | {
            "postconfirmation_integrity_failure", "integrity_source",
            "skipped_states",
        }
        _require(
            set(event) in (ordinary_keys, integrity_keys)
            and event.get("completed_state") == base.get("current_state")
            and isinstance(event.get("checkpoint_name"), str)
            and isinstance(event.get("evidence_path"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(event.get("evidence_sha256")))
            is not None,
            "descendant state-completed event closed schema drift",
        )
        evidence = (repo / str(event["evidence_path"])).resolve(strict=True)
        _require(
            evidence.is_relative_to(
                (
                    repo
                    / f"runs/lewm_domain_robust_gate/attempts/{TARGET}"
                ).resolve()
            )
            and file_hash(evidence) == event["evidence_sha256"],
            "descendant state-completed evidence drift",
        )
        completed_before = base.get("completed_states")
        completed_after = proposed.get("completed_states")
        checkpoints_before = base.get("verified_checkpoints")
        checkpoints_after = proposed.get("verified_checkpoints")
        _require(
            isinstance(completed_before, list)
            and isinstance(completed_after, list)
            and completed_after == completed_before + [event["completed_state"]]
            and isinstance(checkpoints_before, list)
            and isinstance(checkpoints_after, list)
            and len(checkpoints_after) == len(checkpoints_before) + 1
            and checkpoints_after[:-1] == checkpoints_before
            and proposed.get("last_verified_checkpoint") == checkpoints_after[-1]
            and proposed.get("current_state")
            == (
                "POST_TERMINAL_REPORTING"
                if event.get("completed_state") == "POST_TERMINAL_REPORTING"
                else event.get("next_state")
            )
            and proposed.get("updated_unix_ns") == event["created_unix_ns"],
            "descendant state-completed core transition drift",
        )
        checkpoint = checkpoints_after[-1]
        _require(
            isinstance(checkpoint, Mapping)
            and checkpoint.get("name") == event["checkpoint_name"]
            and checkpoint.get("created_unix_ns") == event["created_unix_ns"]
            and checkpoint.get("evidence_path") == event["evidence_path"]
            and checkpoint.get("evidence_sha256") == event["evidence_sha256"]
            and checkpoint.get("source_attempt") == TARGET
            and checkpoint.get("verification_lineage")
            in {
                "direct_checkpoint",
                "direct_postconfirmation_integrity_checkpoint",
            },
            "descendant state-completed checkpoint projection drift",
        )
        evidence_object = _object(evidence)
        completed_state = str(event["completed_state"])
        created = int(event["created_unix_ns"])
        expected = json.loads(json.dumps(base))
        if set(event) == ordinary_keys:
            _require(
                completed_state in state_machine,
                "descendant ordinary state-machine drift",
            )
            state_index = state_machine.index(completed_state)
            final = completed_state == "POST_TERMINAL_REPORTING"
            exact_next = None if final else state_machine[state_index + 1]
            _require(
                event.get("next_state") == exact_next
                and evidence_object.get("passed") is True
                and evidence_object.get("attempt") == TARGET
                and evidence_object.get("checkpoint_state") == completed_state
                and not (
                    completed_state == "INDEPENDENT_VERIFICATION"
                    and evidence_object.get("terminal_label")
                    == "domain_robust_gate_execution_invalid"
                )
                and isinstance(proposed.get("next_action"), str)
                and not (
                    base.get("early_scientific_failure") is not None
                    and not final
                )
                and not (
                    base.get("postconfirmation_integrity_failure") is not None
                    and not final
                )
                and not (
                    base.get("terminal_label") is not None
                    and completed_state
                    not in {"TERMINAL", "POST_TERMINAL_REPORTING"}
                ),
                "descendant ordinary state-completed authorization drift",
            )
            exact_checkpoint = {
                "name": event["checkpoint_name"],
                "created_unix_ns": created,
                "evidence_path": event["evidence_path"],
                "evidence_sha256": event["evidence_sha256"],
                "source_attempt": TARGET,
                "verification_lineage": "direct_checkpoint",
            }
            expected["completed_states"] = completed_before + [completed_state]
            expected["current_state"] = (
                "POST_TERMINAL_REPORTING" if final else exact_next
            )
            expected["last_verified_checkpoint"] = exact_checkpoint
            expected["verified_checkpoints"] = checkpoints_before + [
                exact_checkpoint
            ]
            expected["next_action"] = proposed["next_action"]
            if final:
                _require(
                    base.get("terminal_label") is not None
                    and base.get("process_valid") is not None
                    and base.get("post_terminal_reporting_complete") is not True
                    and proposed.get("next_action")
                    == "program_complete_no_further_scientific_or_reporting_transition",
                    "descendant post-terminal completion drift",
                )
                expected["post_terminal_reporting_complete"] = True
            if completed_state == "PREREGISTRATION_AND_POWER":
                fixed_counts = {
                    "expected_fit_episode_count": 1200,
                    "expected_selection_episode_count": 2000,
                    "expected_smoke_episode_count": 24,
                    "maximum_confirmation_episode_count": 18000,
                }
                registered = evidence_object.get("registered_counts")
                _require(
                    registered == fixed_counts
                    and all(type(registered[key]) is int for key in fixed_counts),
                    "descendant preregistered-count projection drift",
                )
                expected.update(fixed_counts)
            if completed_state == "CONFIRMATION_POWER_AND_COHORT_FREEZE":
                fixed_each = evidence_object.get(
                    "fixed_confirmation_episodes_per_regime"
                )
                fixed_total = evidence_object.get(
                    "fixed_confirmation_episode_count"
                )
                _require(
                    type(fixed_each) is int
                    and fixed_each in range(500, 4501, 500)
                    and type(fixed_total) is int
                    and fixed_total == 4 * fixed_each,
                    "descendant confirmation-size projection drift",
                )
                expected["expected_confirmation_episodes_per_regime"] = fixed_each
                expected["expected_confirmation_episode_count"] = fixed_total
        else:
            integrity_spec = integrity_specs.get(completed_state)
            exact_skipped = (
                list(
                    state_machine[
                        state_machine.index(completed_state) + 1:
                        state_machine.index("INDEPENDENT_VERIFICATION")
                    ]
                )
                if integrity_spec is not None
                else None
            )
            source_name, source_suffix, hash_keys = (
                integrity_spec if integrity_spec is not None else (None, None, None)
            )
            exact_path = (
                f"runs/lewm_domain_robust_gate/attempts/{TARGET}/{source_suffix}"
                if source_suffix is not None
                else None
            )
            _require(
                integrity_spec is not None
                and event.get("next_state") == "INDEPENDENT_VERIFICATION"
                and event.get("postconfirmation_integrity_failure") is True
                and event.get("integrity_source") == source_name
                and event.get("skipped_states") == exact_skipped
                and event.get("evidence_path") == exact_path
                and event.get("checkpoint_name")
                == f"{TARGET}_{source_name}_integrity_failure"
                and evidence_object.get("attempt") == TARGET
                and evidence_object.get("checkpoint_state") == completed_state
                and evidence_object.get("passed") is False
                and base.get("early_scientific_failure") is None
                and base.get("terminal_label") is None,
                "descendant integrity-failure staging drift",
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
                and type(base.get("confirmation_outcome_episodes_generated"))
                is int
                and base.get("confirmation_outcome_episodes_generated")
                == expected_confirmation
                and type(base.get("confirmation_outcome_episodes_executed"))
                is int
                and base.get("confirmation_outcome_episodes_executed")
                == expected_confirmation
                and base.get("confirmation_outcomes_opened_for_analysis") is True
                and proves_integrity_failure(evidence_object),
                "descendant integrity fixed-open proof drift",
            )
            if completed_state == "SEALED_ANALYSIS":
                analysis_result = (
                    repo
                    / f"runs/lewm_domain_robust_gate/attempts/{TARGET}"
                    / "analysis_result.json"
                )
                _require(
                    type(
                        evidence_object.get(
                            "confirmation_outcome_episodes_generated"
                        )
                    ) is int
                    and evidence_object.get(
                        "confirmation_outcome_episodes_generated"
                    ) == expected_confirmation
                    and type(
                        evidence_object.get(
                            "confirmation_outcome_episodes_executed"
                        )
                    ) is int
                    and evidence_object.get(
                        "confirmation_outcome_episodes_executed"
                    ) == expected_confirmation
                    and evidence_object.get(
                        "confirmation_outcomes_opened_for_analysis"
                    ) is True
                    and type(evidence_object.get("analysis_result_present")) is bool
                    and evidence_object.get("analysis_result_present")
                    is analysis_result.is_file()
                    and isinstance(evidence_object.get("error_type"), str)
                    and bool(evidence_object["error_type"])
                    and isinstance(evidence_object.get("error"), str)
                    and bool(evidence_object["error"])
                    and evidence_object.get("scientific_objects_changed") is False,
                    "descendant analysis-invalid evidence drift",
                )
            exact_checkpoint = {
                "name": event["checkpoint_name"],
                "created_unix_ns": created,
                "evidence_path": event["evidence_path"],
                "evidence_sha256": event["evidence_sha256"],
                "source_attempt": TARGET,
                "verification_lineage": (
                    "direct_postconfirmation_integrity_checkpoint"
                ),
            }
            expected["completed_states"] = completed_before + [completed_state]
            expected["verified_checkpoints"] = checkpoints_before + [
                exact_checkpoint
            ]
            expected["last_verified_checkpoint"] = exact_checkpoint
            expected["current_state"] = "INDEPENDENT_VERIFICATION"
            expected["postconfirmation_integrity_failure"] = {
                "status": "awaiting_independent_verification",
                "source": source_name,
                "trigger_state": completed_state,
                "source_path": event["evidence_path"],
                "source_sha256": event["evidence_sha256"],
                "audit_hash_keys": hash_keys,
                "skipped_states": exact_skipped,
                "staged_unix_ns": created,
            }
            expected["skipped_states"] = [
                {
                    "state": name,
                    "source": source_name,
                    "reason": "postconfirmation_integrity_failure_short_circuit",
                    "source_path": event["evidence_path"],
                    "source_sha256": event["evidence_sha256"],
                    "recorded_unix_ns": created,
                }
                for name in exact_skipped
            ]
            expected["next_action"] = (
                "run the read-only confirmation-mode independent verifier and "
                "capture the execution-invalid audit; do not retry or alter "
                "confirmation evidence"
            )
        expected["updated_unix_ns"] = created
        _require(
            proposed == expected,
            "descendant state-completed unrelated STATE drift",
        )
        return

    if sequence == [
        "state_completed", "scientific_terminal_decision_recorded",
        "state_completed",
    ]:
        first, decision, last = controller_events
        common = first["created_unix_ns"]
        ordinary = {
            "event", "attempt", "completed_state", "next_state",
            "checkpoint_name", "evidence_path", "evidence_sha256",
            "created_unix_ns",
        }
        terminal = {
            "event", "attempt", "terminal_label", "process_valid",
            "decision_path", "decision_sha256", "created_unix_ns",
        }
        early = decision.get("terminal_label") == "domain_robust_gate_failed"
        expected_first = ordinary | (
            {"early_failure_mode"}
            if early
            else {"postconfirmation_integrity_failure"}
        )
        expected_decision = terminal | (
            {"confirmation_terminal", "early_failure_mode"} if early else set()
        )
        expected_last = ordinary | (
            {"early_failure_mode", "skipped_states"}
            if early
            else {"postconfirmation_integrity_failure"}
        )
        _require(
            set(first) == expected_first
            and set(decision) == expected_decision
            and set(last) == expected_last
            and decision.get("created_unix_ns") == last.get("created_unix_ns") == common
            and first.get("completed_state") == "INDEPENDENT_VERIFICATION"
            and first.get("next_state") == "TERMINAL"
            and last.get("completed_state") == "TERMINAL"
            and last.get("next_state") == "POST_TERMINAL_REPORTING"
            and decision.get("terminal_label")
            in {"domain_robust_gate_failed", "domain_robust_gate_execution_invalid"}
            and type(decision.get("process_valid")) is bool
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
                    and decision.get("process_valid") is False
                    and first.get("postconfirmation_integrity_failure") is True
                    and last.get("postconfirmation_integrity_failure") is True
                )
            )
        ,
            "descendant three-event terminal event schema drift",
        )
        _require(
            proposed.get("terminal_label") == decision.get("terminal_label")
            and proposed.get("process_valid") == decision.get("process_valid")
            and proposed.get("scientific_terminal") is True
            and proposed.get("current_state") == "POST_TERMINAL_REPORTING"
            and proposed.get("updated_unix_ns") == common
            and proposed.get("completed_states")
            == base.get("completed_states", [])
            + ["INDEPENDENT_VERIFICATION", "TERMINAL"],
            "descendant three-event terminal transition drift",
        )
        for event in (first, last):
            evidence = (repo / str(event["evidence_path"])).resolve(strict=True)
            _require(
                evidence.is_relative_to(
                    (repo / f"runs/lewm_domain_robust_gate/attempts/{TARGET}").resolve()
                )
                and file_hash(evidence) == event["evidence_sha256"],
                "descendant three-event terminal evidence drift",
            )
        exact_audit_path = (
            f"runs/lewm_domain_robust_gate/attempts/{TARGET}/"
            "audit/independent_verification.json"
        )
        exact_decision_path = (
            f"runs/lewm_domain_robust_gate/attempts/{TARGET}/decision.json"
        )
        _require(
            first.get("evidence_path") == exact_audit_path
            and last.get("evidence_path") == exact_decision_path,
            "descendant three-event terminal evidence path drift",
        )
        audit_file, audit_object = active_json_object(
            first["evidence_path"], "three-event independent audit"
        )
        decision_file, decision_object = active_json_object(
            last["evidence_path"], "three-event terminal decision"
        )
        _require(
            file_hash(audit_file) == first["evidence_sha256"]
            and file_hash(decision_file) == last["evidence_sha256"]
            and audit_object.get("attempt") == TARGET
            and decision_object.get("attempt") == TARGET
            and decision_object.get("terminal_label")
            == decision["terminal_label"]
            and type(decision_object.get("process_valid")) is bool
            and decision_object.get("process_valid")
            is decision["process_valid"],
            "descendant three-event terminal JSON binding drift",
        )
        before = base.get("verified_checkpoints")
        after = proposed.get("verified_checkpoints")
        _require(
            isinstance(before, list)
            and isinstance(after, list)
            and after[:-2] == before
            and proposed.get("last_verified_checkpoint") == after[-1]
            and [item.get("evidence_path") for item in after[-2:]]
            == [first.get("evidence_path"), last.get("evidence_path")]
            and [item.get("evidence_sha256") for item in after[-2:]]
            == [first.get("evidence_sha256"), last.get("evidence_sha256")]
            and decision.get("decision_path") == last.get("evidence_path")
            and decision.get("decision_sha256") == last.get("evidence_sha256")
            and proposed.get("terminal_decision_path") == decision.get("decision_path")
            and proposed.get("terminal_decision_sha256")
            == decision.get("decision_sha256"),
            "descendant three-event terminal checkpoint drift",
        )
        _require(
            base.get("current_state") == "INDEPENDENT_VERIFICATION"
            and base.get("terminal_label") is None,
            "descendant three-event terminal base-state drift",
        )
        if early:
            mode = decision["early_failure_mode"]
            staged = base.get("early_scientific_failure")
            contract_file, contract_object = active_json_object(
                staged.get("verifier_contract_path")
                if isinstance(staged, Mapping)
                else None,
                "early-terminal verifier contract",
            )
            contract_record = audit_object.get("verifier_contract")
            source_hashes = audit_object.get("source_hashes")
            source_key = (
                "selection_ledger" if mode == "no_candidate" else "power_freeze"
            )
            _require(
                mode in early_specs
                and isinstance(staged, Mapping)
                and staged.get("mode") == mode
                and staged.get("status") == "awaiting_independent_verification"
                and base.get("postconfirmation_integrity_failure") is None
                and type(audit_object.get("schema_version")) is int
                and audit_object.get("schema_version") == 1
                and audit_object.get("mode") == mode
                and audit_object.get("passed") is True
                and audit_object.get("terminal_label")
                == "domain_robust_gate_failed"
                and audit_object.get("read_only_verifier") is True
                and isinstance(audit_object.get("checks"), Mapping)
                and bool(audit_object["checks"])
                and isinstance(contract_record, Mapping)
                and contract_record.get("path")
                == staged.get("verifier_contract_path")
                and contract_record.get("sha256")
                == file_hash(contract_file)
                == staged.get("verifier_contract_sha256")
                and type(contract_object.get("schema_version")) is int
                and contract_object.get("schema_version") == 1
                and contract_object.get("attempt") == TARGET
                and contract_object.get("attempt_root")
                == f"runs/lewm_domain_robust_gate/attempts/{TARGET}"
                and contract_object.get("mode") == mode
                and isinstance(source_hashes, Mapping)
                and linked_sha256(source_hashes.get(source_key))
                == staged.get("trigger_evidence_sha256")
                and type(decision_object.get("schema_version")) is int
                and decision_object.get("schema_version") == 1
                and decision_object.get("passed") is True
                and decision_object.get("checkpoint_state") == "TERMINAL"
                and decision_object.get("early_failure_mode") == mode
                and decision_object.get("trigger_evidence_path")
                == staged.get("trigger_evidence_path")
                and decision_object.get("trigger_evidence_sha256")
                == staged.get("trigger_evidence_sha256")
                and decision_object.get("independent_verification_path")
                == first.get("evidence_path")
                and decision_object.get("independent_verification_sha256")
                == first.get("evidence_sha256")
                and type(decision_object.get("smoke_outcome_episodes")) is int
                and decision_object.get("smoke_outcome_episodes") == 0
                and type(
                    decision_object.get(
                        "confirmation_outcome_episodes_generated"
                    )
                ) is int
                and decision_object.get(
                    "confirmation_outcome_episodes_generated"
                ) == 0
                and type(
                    decision_object.get(
                        "confirmation_outcome_episodes_executed"
                    )
                ) is int
                and decision_object.get(
                    "confirmation_outcome_episodes_executed"
                ) == 0
                and decision_object.get(
                    "confirmation_outcomes_opened_for_analysis"
                ) is False
                and last.get("skipped_states") == staged.get("skipped_states")
                and first.get("checkpoint_name")
                == f"{TARGET}_{mode}_independent_verification_passed"
                and last.get("checkpoint_name")
                == f"{TARGET}_{mode}_terminal_decision_recorded",
                "descendant early terminal authorization drift",
            )
            active_root = (
                repo / f"runs/lewm_domain_robust_gate/attempts/{TARGET}"
            )
            forbidden_roots = (
                "data/smoke", "data/confirmation", "execution/smoke",
                "execution/confirmation", "metrics/smoke", "metrics/confirmation",
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
                type(base.get("expected_fit_episode_count")) is int
                and type(base.get("fit_outcome_episodes")) is int
                and base.get("fit_outcome_episodes")
                == base.get("expected_fit_episode_count")
                and type(base.get("expected_selection_episode_count")) is int
                and type(base.get("selection_outcome_episodes")) is int
                and base.get("selection_outcome_episodes")
                == base.get("expected_selection_episode_count")
                and type(base.get("smoke_outcome_episodes")) is int
                and base.get("smoke_outcome_episodes") == 0
                and type(base.get("confirmation_outcome_episodes_generated")) is int
                and base.get("confirmation_outcome_episodes_generated") == 0
                and type(base.get("confirmation_outcome_episodes_executed")) is int
                and base.get("confirmation_outcome_episodes_executed") == 0
                and base.get("confirmation_outcomes_opened_for_analysis") is False
                and all(
                    not root.exists()
                    or not any(path.is_file() for path in root.rglob("*"))
                    for root in (
                        active_root / relative for relative in forbidden_roots
                    )
                )
                and all(
                    not (active_root / relative).exists()
                    for relative in forbidden_files
                ),
                "descendant early terminal later-role isolation drift",
            )
            checkpoint_lineage = "direct_early_scientific_failure_checkpoint"
        else:
            expected_confirmation = base.get(
                "expected_confirmation_episode_count"
            )
            contract_path = (
                f"runs/lewm_domain_robust_gate/attempts/{TARGET}/"
                "verifier_contract.json"
            )
            contract_file, contract_object = active_json_object(
                contract_path, "execution-invalid verifier contract"
            )
            contract_record = audit_object.get("verifier_contract")
            capture_record = audit_object.get("capture")
            source_hashes = audit_object.get("source_hashes")
            staged_integrity = base.get("postconfirmation_integrity_failure")
            _require(
                base.get("early_scientific_failure") is None
                and type(expected_confirmation) is int
                and expected_confirmation > 0
                and type(base.get("expected_smoke_episode_count")) is int
                and type(base.get("smoke_outcome_episodes")) is int
                and base.get("smoke_outcome_episodes")
                == base.get("expected_smoke_episode_count")
                and type(base.get("confirmation_outcome_episodes_generated")) is int
                and base.get("confirmation_outcome_episodes_generated")
                == expected_confirmation
                and type(base.get("confirmation_outcome_episodes_executed")) is int
                and base.get("confirmation_outcome_episodes_executed")
                == expected_confirmation
                and base.get("confirmation_outcomes_opened_for_analysis") is True
                and type(audit_object.get("schema_version")) is int
                and audit_object.get("schema_version") == 1
                and audit_object.get("mode") == "confirmation"
                and audit_object.get("read_only_verifier") is True
                and audit_object.get("execution_invalid") is True
                and audit_object.get("terminal_label")
                == "domain_robust_gate_execution_invalid"
                and proves_integrity_failure(audit_object)
                and isinstance(contract_record, Mapping)
                and contract_record.get("path") == contract_path
                and contract_record.get("sha256") == file_hash(contract_file)
                and type(contract_object.get("schema_version")) is int
                and contract_object.get("schema_version") == 1
                and contract_object.get("attempt") == TARGET
                and contract_object.get("attempt_root")
                == f"runs/lewm_domain_robust_gate/attempts/{TARGET}"
                and contract_object.get("mode") == "confirmation"
                and isinstance(capture_record, Mapping)
                and capture_record.get("captured_exclusively") is True
                and capture_record.get("wrapper_integrity_passed") is True
                and type(decision_object.get("schema_version")) is int
                and decision_object.get("schema_version") == 1
                and decision_object.get("passed") is True
                and decision_object.get("checkpoint_state") == "TERMINAL"
                and decision_object.get("terminal_basis")
                == "postconfirmation_integrity_failure"
                and decision_object.get("independent_verification_path")
                == first.get("evidence_path")
                and decision_object.get("independent_verification_sha256")
                == first.get("evidence_sha256")
                and type(
                    decision_object.get(
                        "confirmation_outcome_episodes_generated"
                    )
                ) is int
                and decision_object.get(
                    "confirmation_outcome_episodes_generated"
                ) == expected_confirmation
                and type(
                    decision_object.get(
                        "confirmation_outcome_episodes_executed"
                    )
                ) is int
                and decision_object.get(
                    "confirmation_outcome_episodes_executed"
                ) == expected_confirmation
                and decision_object.get(
                    "confirmation_outcomes_opened_for_analysis"
                ) is True
                and first.get("checkpoint_name")
                == f"{TARGET}_execution_invalid_independent_audit"
                and last.get("checkpoint_name")
                == f"{TARGET}_execution_invalid_terminal_decision",
                "descendant execution-invalid base/evidence drift",
            )
            if isinstance(staged_integrity, Mapping):
                candidates = staged_integrity.get("audit_hash_keys")
                _require(
                    audit_object.get("passed") is True
                    and isinstance(source_hashes, Mapping)
                    and isinstance(candidates, list)
                    and bool(candidates)
                    and any(
                        linked_sha256(source_hashes.get(name))
                        == staged_integrity.get("source_sha256")
                        for name in candidates
                    ),
                    "descendant execution-invalid staged-source drift",
                )
            else:
                _require(
                    staged_integrity is None
                    and audit_object.get("passed") is False
                    and capture_record.get("scientific_verifier_passed") is False
                    and type(capture_record.get("verifier_returncode")) is int
                    and capture_record.get("verifier_returncode") != 0
                    and isinstance(audit_object.get("error_type"), str)
                    and bool(audit_object["error_type"])
                    and isinstance(audit_object.get("error"), str)
                    and bool(audit_object["error"]),
                    "descendant execution-invalid verifier-failure drift",
                )
            checkpoint_lineage = (
                "direct_postconfirmation_integrity_checkpoint"
            )
        audit_checkpoint = {
            "name": first["checkpoint_name"],
            "created_unix_ns": common,
            "evidence_path": first["evidence_path"],
            "evidence_sha256": first["evidence_sha256"],
            "source_attempt": TARGET,
            "verification_lineage": checkpoint_lineage,
        }
        decision_checkpoint = {
            "name": last["checkpoint_name"],
            "created_unix_ns": common,
            "evidence_path": last["evidence_path"],
            "evidence_sha256": last["evidence_sha256"],
            "source_attempt": TARGET,
            "verification_lineage": checkpoint_lineage,
        }
        exact = json.loads(json.dumps(base))
        exact["completed_states"] = base.get("completed_states", []) + [
            "INDEPENDENT_VERIFICATION", "TERMINAL"
        ]
        exact["verified_checkpoints"] = before + [
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
                    "terminal_recorded_unix_ns": common,
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
                    "terminal_recorded_unix_ns": common,
                }
            )
            exact["next_action"] = (
                "report the immutable execution-invalid result and integrity "
                "diagnosis; do not retry or reinterpret it as a scientific "
                "partial/failure"
            )
        exact["updated_unix_ns"] = common
        _require(
            proposed == exact,
            "descendant three-event terminal unrelated STATE drift",
        )
        return

    raise VerificationError("descendant adapter event sequence drift")


def _verify_descendant_adapter_transaction_chain(
    *,
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    ledger_payload: bytes,
    target: Path,
    repo: Path,
    seal_path: Path,
    source_state: Mapping[str, Any],
    source_ledger: Mapping[str, Any],
    authorized_transaction_context_sha256: str | None,
) -> int:
    """Independently replay every receipt-anchored v008 adapter transaction."""

    receipt, published = _load_active_transaction_receipt(
        target=target,
        authorized_transaction_context_sha256=authorized_transaction_context_sha256,
    )
    post_state, post_ledger = _verify_initial_transaction_receipt_anchor(
        receipt=receipt,
        published=published,
        source_state=source_state,
        source_ledger=source_ledger,
        events=events,
        ledger_payload=ledger_payload,
        target=target,
        repo=repo,
        seal_path=seal_path,
        allow_transaction_journal=(
            authorized_transaction_context_sha256 is not None
        ),
    )
    post_count = int(post_ledger["event_count"])
    post_bytes = int(post_ledger["bytes"])
    lines = ledger_payload.splitlines(keepends=True)
    active_edges = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "zero_confirmation_outcome_version_forward"
        and event.get("attempt") == TARGET
    ]
    _require(
        post_count > 0
        and post_bytes > 0
        and len(active_edges) == 1
        and post_count == active_edges[0] + 1
        and len(lines) == len(events) == state.get("ledger_event_count")
        and post_count <= len(lines)
        and b"".join(lines[:post_count]) == ledger_payload[:post_bytes]
        and post_bytes == len(b"".join(lines[:post_count]))
        and _hash_bytes(ledger_payload[:post_bytes])
        == post_ledger.get("sha256")
        and post_ledger.get("head_sha256")
        == events[post_count - 1].get("record_sha256")
        and post_state.get("ledger_event_count") == post_count
        and post_state.get("ledger_head_sha256")
        == post_ledger.get("head_sha256")
        and receipt.get("controller_return") == post_state,
        "descendant adapter ledger boundary drift",
    )
    expected_base = json.loads(json.dumps(dict(post_state)))
    if post_count == len(lines):
        _require(
            _json_hash(state) == _json_hash(expected_base),
            "current initial adapter state differs from receipt",
        )
        return 0

    field = "v008_durable_adapter_transaction"
    kind = "receipt_anchored_v008_descendant_adapter_transaction"
    cursor = post_count
    groups = 0
    while cursor < len(lines):
        first = events[cursor]
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
            and isinstance(envelope.get("transaction_id"), str)
            and re.fullmatch(r"[0-9a-f]{64}", envelope["transaction_id"])
            is not None
            and type(envelope.get("event_index")) is int
            and envelope.get("event_index") == 0
            and type(envelope.get("event_count")) is int
            and envelope.get("event_count") > 0
            and isinstance(envelope.get("base_state_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", envelope["base_state_sha256"])
            is not None
            and isinstance(envelope.get("proposed_state_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", envelope["proposed_state_sha256"])
            is not None
            and isinstance(envelope.get("base_state"), Mapping),
            "descendant adapter first-envelope drift",
        )
        count = int(envelope["event_count"])
        end = cursor + count
        _require(end <= len(events), "descendant adapter group exceeds ledger")
        base_state = dict(envelope["base_state"])
        _require(
            _json_hash(base_state) == envelope.get("base_state_sha256")
            and _json_hash(base_state) == _json_hash(expected_base)
            and base_state.get("active_attempt") == TARGET
            and type(base_state.get("ledger_event_count")) is int
            and base_state.get("ledger_event_count") == cursor
            and base_state.get("ledger_head_sha256")
            == events[cursor - 1].get("record_sha256"),
            "descendant adapter base-state drift",
        )
        stripped: list[dict[str, Any]] = []
        transaction_id = str(envelope["transaction_id"])
        for offset in range(count):
            record = events[cursor + offset]
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
                lines[cursor + offset]
                == json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
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
                == events[cursor + offset - 1].get("record_sha256")
                and isinstance(observed_record_sha256, str)
                and re.fullmatch(r"[0-9a-f]{64}", observed_record_sha256)
                is not None
                and observed_record_sha256 == _json_hash(record_without_hash)
                and type(record.get("created_unix_ns")) is int
                and record.get("created_unix_ns") > 0
                and record.get("attempt") == TARGET,
                "descendant adapter envelope/record drift",
            )
            body = dict(record)
            body.pop(field)
            body.pop("seq")
            body.pop("prev_sha256")
            body.pop("record_sha256")
            stripped.append(body)
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
            "descendant adapter event sequence drift",
        )
        transaction_projection = {
            "schema_version": 1,
            "authorization_kind": kind,
            "base_state_sha256": _json_hash(base_state),
            "proposed_state_sha256": envelope.get("proposed_state_sha256"),
            "events": stripped,
        }
        _require(
            _json_hash(transaction_projection) == transaction_id,
            "descendant adapter transaction id drift",
        )
        if end < len(events):
            next_envelope = events[end].get(field)
            _require(
                isinstance(next_envelope, Mapping)
                and next_envelope.get("event_index") == 0
                and isinstance(next_envelope.get("base_state"), Mapping),
                "descendant adapter grouping gap",
            )
            result_state = dict(next_envelope["base_state"])
        else:
            result_state = dict(state)
        _require(
            result_state.get("active_attempt") == TARGET
            and type(result_state.get("ledger_event_count")) is int
            and result_state.get("ledger_event_count") == end
            and result_state.get("ledger_head_sha256")
            == events[end - 1].get("record_sha256"),
            "descendant adapter target-state drift",
        )
        without_marker = json.loads(json.dumps(result_state))
        marker = without_marker.pop("v008_durable_controller_adapter", None)
        proposal = json.loads(json.dumps(without_marker))
        proposal["ledger_event_count"] = base_state["ledger_event_count"]
        proposal["ledger_head_sha256"] = base_state["ledger_head_sha256"]
        _require(
            _json_hash(proposal) == envelope.get("proposed_state_sha256"),
            "descendant adapter proposed-state commitment drift",
        )
        _verify_descendant_controller_transition(
            base_state=base_state,
            proposed_state=proposal,
            controller_events=stripped,
            repo=repo,
        )
        operation = _json_hash(
            {
                "base_state_object_sha256": _json_hash(base_state),
                "base_ledger_sha256": _hash_bytes(b"".join(lines[:cursor])),
                "expected_suffix_sha256": _hash_bytes(
                    b"".join(lines[cursor:end])
                ),
                "intended_state_object_sha256": _json_hash(without_marker),
            }
        )
        verified_marker = _verify_recovery_adapter_marker(
            marker,
            event_count=end,
            head_sha256=str(events[end - 1]["record_sha256"]),
            target=target,
            repo=repo,
            label="descendant adapter target",
            state_without_marker=without_marker,
        )
        _require(
            verified_marker.get("operation_sha256") == operation,
            "descendant adapter operation digest drift",
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
                and count_events[0].get("prior_controller_adapter_marker")
                == base_state.get("v008_durable_controller_adapter")
            ),
            "descendant count prior-marker drift",
        )
        expected_base = result_state
        cursor = end
        groups += 1
    return groups


def _verify_adapter_count_event_bindings(
    events: Sequence[Mapping[str, Any]],
    *,
    target: Path,
    repo: Path,
) -> int:
    field = "prior_controller_adapter_marker"
    verified = 0
    for event in events:
        is_active_count = (
            event.get("attempt") == TARGET
            and event.get("event") == "outcome_counts_updated"
        )
        if field in event and not is_active_count:
            marker = event.get(field)
            source = target.parent / SELECTION_SOURCE
            adapter = source / "version_forward_transaction.py"
            program = target.parents[1] / "program.py"
            keys = {
                "schema_version", "authorization_kind", "adapter_source_path",
                "adapter_source_sha256", "adapter_source_ast_sha256",
                "root_program_path", "root_program_sha256",
                "root_program_ast_sha256", "ledger_event_count",
                "ledger_head_sha256", "operation_sha256",
                "state_binding_sha256",
            }
            sequence = event.get("seq")
            previous = event.get("prev_sha256")
            _require(
                event.get("attempt") == SELECTION_SOURCE
                and event.get("event") == "outcome_counts_updated"
                and isinstance(marker, Mapping)
                and set(marker) == keys
                and type(marker.get("schema_version")) is int
                and marker.get("schema_version") == 2
                and marker.get("authorization_kind")
                == "receipt_bound_v006_root_controller_adapter"
                and marker.get("adapter_source_path") == _relative(adapter, repo)
                and marker.get("adapter_source_sha256")
                == file_hash(adapter) == EXPECTED_SELECTION_SOURCE_ADAPTER_SHA256
                and marker.get("adapter_source_ast_sha256")
                == _source_ast_sha256(adapter)
                and marker.get("root_program_path") == _relative(program, repo)
                and marker.get("root_program_sha256") == file_hash(program)
                and marker.get("root_program_ast_sha256")
                == _source_ast_sha256(program)
                and type(sequence) is int
                and isinstance(previous, str)
                and marker.get("ledger_event_count") == sequence - 1
                and marker.get("ledger_head_sha256") == previous
                and all(
                    isinstance(marker.get(key), str)
                    and re.fullmatch(r"[0-9a-f]{64}", marker[key]) is not None
                    for key in ("operation_sha256", "state_binding_sha256")
                ),
                "historical v006 adapter marker drift",
            )
        if not is_active_count:
            continue
        sequence = event.get("seq")
        previous = event.get("prev_sha256")
        _require(
            field in event
            and type(sequence) is int
            and isinstance(previous, str),
            "v008 count event predecessor identity drift",
        )
        _verify_recovery_adapter_marker(
            event[field],
            event_count=sequence - 1,
            head_sha256=previous,
            target=target,
            repo=repo,
            label="v008 count-event predecessor",
        )
        verified += 1
    return verified


def _verify_role_count_recovery_authorization(
    state: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    role: str,
    target: Path,
    repo: Path,
) -> dict[str, Any]:
    """Independently authenticate the sole post-count/pre-checkpoint crash state."""

    specifications = {
        "fit": {
            "state": "FIT_COHORTS",
            "counter": "fit_outcome_episodes",
            "expected": 1200,
            "episodes_per_regime": 300,
            "audit": "audit/fit_cohorts.json",
            "later": (
                "audit/fit_lock_checkpoint.json",
                "audit/pre_selection_manifest.json",
                "audit/pre_selection_seal.json",
                "audit/selection_cohorts.json",
                "audit/candidate_selection.json",
                "data/selection",
                "data/smoke",
                "data/confirmation",
            ),
        },
        "selection": {
            "state": "SELECTION_COHORTS",
            "counter": "selection_outcome_episodes",
            "expected": 2000,
            "episodes_per_regime": 500,
            "audit": "audit/selection_cohorts.json",
            "later": (
                "audit/candidate_selection.json",
                "audit/gate_freeze_checkpoint.json",
                "audit/confirmation_power_and_cohort_freeze.json",
                "audit/pre_confirmation_package_seal.json",
                "data/smoke",
                "data/confirmation",
            ),
        },
    }
    _require(role in specifications, "unknown role-count recovery role")
    specification = specifications[role]
    state_name = str(specification["state"])
    counter = str(specification["counter"])
    expected = int(specification["expected"])
    audit_path = target / str(specification["audit"])
    audit_relative = (
        target.resolve(strict=True).relative_to(repo.resolve(strict=True))
        / str(specification["audit"])
    ).as_posix()
    expected_counters = {
        "fit_outcome_episodes": 1200,
        "selection_outcome_episodes": expected if role == "selection" else 0,
        "smoke_outcome_episodes": 0,
        "confirmation_outcome_episodes_generated": 0,
        "confirmation_outcome_episodes_executed": 0,
        "confirmation_outcomes_opened_for_analysis": False,
    }
    _require(
        state.get("current_state") == state_name
        and type(state.get(counter)) is int
        and state.get(counter) == expected
        and all(
            type(state.get(name)) is type(value)
            and state.get(name) == value
            for name, value in expected_counters.items()
        ),
        "role-count recovery state/counters drift",
    )
    state_machine = (
        "BOOTSTRAP_AUDIT", "DIAGNOSTIC_ACCOUNT", "PREREGISTRATION_AND_POWER",
        "IMPLEMENTATION_COMPLETE", "PRESEAL_QUALIFICATION", "PRE_OUTCOME_SEAL",
        "FIT_COHORTS", "FIT_LOCK", "PRE_SELECTION_SEAL", "SELECTION_COHORTS",
        "CANDIDATE_SELECTION", "GATE_FREEZE", "CONFIRMATION_POWER_AND_COHORT_FREEZE",
        "PRE_CONFIRMATION_PACKAGE_SEAL", "EXCLUDED_MECHANICAL_SMOKE",
        "CONFIRMATION_GENERATION", "CONFIRMATION_EXECUTION", "CONFIRMATION_INPUT_SEAL",
        "SEALED_ANALYSIS", "LATENCY_AND_RESOURCE_REPORTING", "INDEPENDENT_VERIFICATION",
        "TERMINAL", "POST_TERMINAL_REPORTING",
    )
    current_index = state_machine.index(state_name)
    completed = state.get("completed_states")
    checkpoints = state.get("verified_checkpoints")
    _require(tuple(state.get("state_machine", ())) == state_machine, "role-count recovery state-machine drift")
    _require(completed == list(state_machine[:current_index]), "role-count recovery completed-state prefix drift")
    _require(isinstance(checkpoints, list) and len(checkpoints) == current_index, "role-count recovery checkpoint prefix drift")
    _require(
        all(
            isinstance(item, Mapping)
            and item.get("evidence_path") != audit_relative
            and item.get("name") != f"{TARGET}_{role}_cohorts_verified"
            for item in checkpoints
        ),
        "role-count recovery already has a role checkpoint",
    )
    _require(
        state.get("early_scientific_failure") is None
        and state.get("postconfirmation_integrity_failure") is None
        and state.get("terminal_label") is None
        and state.get("confirmation_terminal") is False
        and state.get("skipped_states") in (None, []),
        "role-count recovery follows a branch/terminal mutation",
    )

    _require(bool(events), "role-count recovery ledger is empty")
    event = dict(events[-1])
    event_body = dict(event)
    observed_hash = event_body.pop("record_sha256", None)
    expected_event_keys = {
        "event", "attempt", "fields", "created_unix_ns", "seq",
        "prev_sha256", "record_sha256", "prior_controller_adapter_marker",
        "v008_durable_adapter_transaction",
    }
    _require(
        set(event) == expected_event_keys
        and event.get("event") == "outcome_counts_updated"
        and event.get("attempt") == TARGET
        and event.get("fields") == {counter: expected}
        and type(event.get("seq")) is int
        and type(state.get("ledger_event_count")) is int
        and event.get("seq") == len(events) == state.get("ledger_event_count")
        and type(event.get("created_unix_ns")) is int
        and type(state.get("updated_unix_ns")) is int
        and event.get("created_unix_ns") == state.get("updated_unix_ns")
        and observed_hash == state.get("ledger_head_sha256")
        and observed_hash == _json_hash(event_body),
        "role-count recovery final ledger event drift",
    )
    _require(len(events) >= 2, "role-count recovery lacks a predecessor event")
    previous = events[-2]
    previous_hash = previous.get("record_sha256")
    _require(
        isinstance(previous_hash, str)
        and event.get("prev_sha256") == previous_hash
        and type(previous.get("created_unix_ns")) is int,
        "role-count recovery predecessor event drift",
    )

    ledger_payload = (target.parents[1] / "RESEARCH_LEDGER.jsonl").read_bytes()
    ledger_lines = ledger_payload.splitlines(keepends=True)
    expected_event_suffix = (
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    _require(
        len(ledger_lines) == len(events)
        and ledger_lines[-1] == expected_event_suffix,
        "role-count recovery final ledger suffix is not canonical",
    )
    base_ledger_payload = b"".join(ledger_lines[:-1])

    current_state_without_marker = json.loads(json.dumps(dict(state)))
    current_state_without_marker.pop("v008_durable_controller_adapter", None)
    current_adapter_marker = _verify_recovery_adapter_marker(
        state.get("v008_durable_controller_adapter"),
        event_count=len(events),
        head_sha256=str(observed_hash),
        target=target,
        repo=repo,
        label="current role-count recovery",
        state_without_marker=current_state_without_marker,
    )

    pre_count_state_without_marker = dict(current_state_without_marker)
    pre_count_state_without_marker[counter] = 0
    pre_count_state_without_marker["updated_unix_ns"] = previous[
        "created_unix_ns"
    ]
    pre_count_state_without_marker["ledger_event_count"] = len(events) - 1
    pre_count_state_without_marker["ledger_head_sha256"] = previous_hash
    prior_adapter_marker = _verify_recovery_adapter_marker(
        event.get("prior_controller_adapter_marker"),
        event_count=len(events) - 1,
        head_sha256=str(previous_hash),
        target=target,
        repo=repo,
        label="prior role-count recovery",
        state_without_marker=pre_count_state_without_marker,
    )
    pre_count_state = dict(pre_count_state_without_marker)
    pre_count_state["v008_durable_controller_adapter"] = prior_adapter_marker
    operation_context = {
        "base_state_object_sha256": _json_hash(pre_count_state),
        "base_ledger_sha256": _hash_bytes(base_ledger_payload),
        "expected_suffix_sha256": _hash_bytes(expected_event_suffix),
        "intended_state_object_sha256": _json_hash(
            current_state_without_marker
        ),
    }
    expected_operation_sha256 = _json_hash(operation_context)
    _require(
        current_adapter_marker.get("operation_sha256")
        == expected_operation_sha256,
        "current role-count recovery adapter operation/context drift",
    )
    pre_count_state_sha256 = _hash_bytes(
        (json.dumps(pre_count_state, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    regimes = (
        "native_plan", "markov_oracle", "plan_action_noise_0p2",
        "plan_random_action_0p1",
    )
    manifest_hashes: list[str] = []
    for regime in regimes:
        manifest_path = target / "data" / role / regime / "execution_manifest.json"
        info = manifest_path.lstat() if os.path.lexists(manifest_path) else None
        _require(
            info is not None and stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            and int(info.st_nlink) == 1,
            f"role-count recovery execution manifest is absent/linked: {role}/{regime}",
        )
        manifest = _object(manifest_path)
        authorization = manifest.get("authorization")
        _require(
            manifest.get("attempt") == TARGET
            and manifest.get("role") == role
            and manifest.get("regime") == regime
            and manifest.get("complete") is True
            and manifest.get("episode_count") == int(specification["episodes_per_regime"])
            and isinstance(authorization, Mapping)
            and authorization.get("active_attempt") == TARGET
            and authorization.get("state") == state_name
            and authorization.get("state_sha256") == pre_count_state_sha256,
            f"role-count recovery execution authorization drift: {role}/{regime}",
        )
        manifest_hashes.append(file_hash(manifest_path))
    _require(len(manifest_hashes) == len(regimes), "role-count recovery manifest count drift")

    audit_sha256: str | None = None
    audit_created_unix_ns: int | None = None
    if os.path.lexists(audit_path):
        info = audit_path.lstat()
        _require(
            stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            and int(info.st_nlink) == 1,
            "role-count recovery audit is linked/non-regular",
        )
        audit = _object(audit_path)
        _require(
            audit_path.read_bytes()
            == (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            "role-count recovery audit is not canonical JSON",
        )
        _require(
            type(audit.get("created_unix_ns")) is int
            and int(audit["created_unix_ns"])
            >= int(event["created_unix_ns"]),
            "role-count recovery audit chronology drift",
        )
        audit_created_unix_ns = int(audit["created_unix_ns"])
        audit_sha256 = file_hash(audit_path)
    conflicting = [
        relative for relative in specification["later"]
        if os.path.lexists(target / str(relative))
    ]
    _require(not conflicting, f"role-count recovery has later output conflicts: {conflicting}")
    return {
        "passed": True,
        "role": role,
        "state": state_name,
        "counter_field": counter,
        "counter_value": expected,
        "count_update_event_seq": len(events),
        "count_update_event_sha256": observed_hash,
        "pre_count_state_sha256": pre_count_state_sha256,
        "prior_controller_adapter_marker_sha256": _json_hash(
            prior_adapter_marker
        ),
        "prior_state_binding_sha256": prior_adapter_marker[
            "state_binding_sha256"
        ],
        "current_state_binding_sha256": current_adapter_marker[
            "state_binding_sha256"
        ],
        "count_operation_sha256": expected_operation_sha256,
        "count_operation_context": operation_context,
        "execution_manifest_sha256": manifest_hashes,
        "role_audit_path": audit_relative,
        "role_audit_sha256": audit_sha256,
        "role_audit_created_unix_ns": audit_created_unix_ns,
        "no_later_output_conflicts": True,
    }


def verify(
    path: Path,
    study_root: Path,
    *,
    phase: str,
    authorized_early_verifier_state: str | None = None,
    authorized_role_count_recovery: str | None = None,
    authorized_transaction_context_sha256: str | None = None,
) -> dict[str, Any]:
    _require(phase in {"pre-forward", "post-forward"}, "invalid verifier phase")
    _require(
        phase == "post-forward" or authorized_transaction_context_sha256 is None,
        "transient transaction authorization is post-forward only",
    )
    requested = Path(path).absolute()
    lexical_expected = (
        Path(study_root).absolute() / "attempts" / TARGET / OUTPUT
    )
    _require(
        requested == lexical_expected,
        "inheritance seal path is not exact",
    )
    study, repo, source, target = _roots(study_root)
    seal_path = Path(path).resolve(strict=True)
    _require(
        seal_path == (target / OUTPUT).resolve(strict=False),
        "inheritance seal path is not exact",
    )
    artifact = _object(seal_path)
    _validate_inheritance_seal_scalar_types(artifact)
    header = {
        "schema_version": 1,
        "artifact_type": "pre_data_inheritance_and_version_forward_equivalence",
        "authorization_kind": AUTH_KIND,
        "attempt": TARGET, "source_attempt": SOURCE,
        "science_attempt": SCIENCE_ATTEMPT, "target_attempt": TARGET,
        "checkpoint_state": "PRE_OUTCOME_SEAL", "resume_state": RESUME,
        "procedural_invalidity_confirmed": True, "zero_confirmation_outcomes_at_version_forward": True,
        "source_hash_equivalent": True, "normalized_ast_equivalent": True,
        "scientific_object_hash_equivalent": True, "configuration_hash_equivalent": True,
        "scientific_changes": False, "attempt_parameterization_verified": True,
        "runtime_modules_accept_active_attempt": True, "independent_verifier_accepts_active_attempt": True,
        "source_equivalent": True, "ast_equivalent": True, "scientific_objects_equivalent": True,
        "hashes_equivalent": True, "outcome_arrays_opened": False,
        "hdf5_contents_opened": False, "v3_targets_opened": False, "passed": True,
    }
    _require(
        all(
            type(artifact.get(key)) is type(value)
            and artifact.get(key) == value
            for key, value in header.items()
        ),
        "inheritance seal header/boolean drift",
    )
    _require(type(artifact.get("created_unix_ns")) is int and artifact["created_unix_ns"] > 0, "creation timestamp invalid")
    invalid_path = source / INVALIDITY
    invalidity_draft_path = source / INVALIDITY_DRAFT
    source_seal_path = source / SOURCE_SEAL
    source_receipt_path = source / SOURCE_RECEIPT
    source_pre_selection_path = source.parent / SELECTION_SOURCE / SOURCE_PRE_SELECTION_SEAL
    _require(artifact.get("invalidity_evidence") == {"path": _relative(invalid_path, repo), "sha256": file_hash(invalid_path)}, "invalidity link drift")
    _require(
        artifact.get("superseded_invalidity_draft")
        == {
            "path": _relative(invalidity_draft_path, repo),
            "sha256": file_hash(invalidity_draft_path),
            "authoritative": False,
        },
        "superseded invalidity draft link drift",
    )
    _require(artifact.get("source_pre_data_seal") == {"path": _relative(source_seal_path, repo), "sha256": file_hash(source_seal_path)}, "source seal link drift")
    _require(
        artifact.get("source_pre_selection_seal")
        == {
            "path": _relative(source_pre_selection_path, repo),
            "sha256": file_hash(source_pre_selection_path),
        },
        "source pre-selection seal link drift",
    )
    _require(
        artifact.get("source_version_forward_transaction_receipt")
        == {
            "path": _relative(source_receipt_path, repo),
            "sha256": file_hash(source_receipt_path),
        },
        "prior version-forward receipt link drift",
    )
    invalidity = _object(invalid_path)
    invalidity_draft = _object(invalidity_draft_path)
    source_seal = _object(source_seal_path)
    source_receipt = _object(source_receipt_path)
    source_pre_selection = _object(source_pre_selection_path)
    _require(
        _exact_typed_mapping(artifact.get("outcome_counts_at_seal"), EXPECTED_COUNTS),
        "sealed outcome counts drift",
    )
    confirmation = [
        _relative(item, repo) for root in (source / "data/confirmation", target / "data/confirmation") if root.exists()
        for item in root.rglob("*") if item.is_file() or item.is_symlink()
    ]
    _require(not confirmation and artifact.get("confirmation_artifact_scan") == {"file_count": 0, "paths": []}, "confirmation artifact scan drift")
    state = _object(study / "STATE.json")
    chain, events = _ledger(study)
    source_authorization = _authenticate_immediate_source(
        study=study,
        repo=repo,
        source=source,
        source_seal=source_seal,
        invalidity=invalidity,
        invalidity_draft=invalidity_draft,
        source_receipt=source_receipt,
        live_state=state,
        live_chain=chain,
        live_events=events,
        phase=phase,
        authorized_transaction_context_sha256=(
            authorized_transaction_context_sha256
        ),
    )
    _require(
        artifact.get("prior_version_forward_lineage")
        == source_authorization["lineage"],
        "sealed prior version-forward lineage drift",
    )
    _require(
        artifact.get("source_controller_adapter_marker")
        == source_authorization["marker"],
        "sealed v006 controller adapter marker drift",
    )
    closure = _closure(
        source,
        target,
        repo,
        source_seal,
        state,
        phase=phase,
    )
    _require(artifact.get("v008_manifest_closure") == closure, "manifest closure record drift")
    partitions, _members = _verify_records(artifact, source, target, repo, closure)
    _require(artifact.get("equivalent_files") == partitions["exact_hash"], "equivalent_files alias drift")

    source_files = source_seal.get("sealed_files")
    _require(isinstance(source_files, dict) and bool(source_files), "source sealed_files absent")
    for raw, digest in source_files.items():
        _require(isinstance(raw, str) and isinstance(digest, str), "source sealed file record type drift")
        candidate = _resolve(repo, raw)
        _require(candidate.is_file() and file_hash(candidate) == digest, f"source sealed file drift: {raw}")
    _require(
        len(source_files) == 6475
        and artifact.get("source_sealed_files")
        == {
            "file_count": len(source_files),
            "files": dict(sorted(source_files.items())),
            "all_rehashed": True,
        },
        "v007 source sealed_files provenance drift",
    )

    checkpoint_source_files = source_pre_selection.get("sealed_files")
    _require(
        isinstance(checkpoint_source_files, Mapping)
        and len(checkpoint_source_files) == 6408,
        "v006 pre-selection sealed-file census drift",
    )
    for raw, digest in checkpoint_source_files.items():
        _require(
            isinstance(raw, str)
            and isinstance(digest, str)
            and _resolve(repo, raw).is_file()
            and file_hash(_resolve(repo, raw)) == digest,
            f"v006 pre-selection sealed file drift: {raw}",
        )
    _require(
        artifact.get("source_pre_selection_sealed_files")
        == {
            "file_count": len(checkpoint_source_files),
            "files": dict(sorted(checkpoint_source_files.items())),
            "all_rehashed": True,
        },
        "v006 pre-selection sealed-file provenance drift",
    )

    expected_sealed = dict(source_files)
    for relative in closure["discovered_paths"]:
        candidate = target / relative; expected_sealed[_relative(candidate, repo)] = file_hash(candidate)
    for raw in closure["declared_repository_paths"]:
        expected_sealed[raw] = file_hash(_resolve(repo, raw))
    expected_sealed[_relative(invalid_path, repo)] = file_hash(invalid_path)
    expected_sealed[_relative(invalidity_draft_path, repo)] = file_hash(
        invalidity_draft_path
    )
    expected_sealed[_relative(source_seal_path, repo)] = file_hash(source_seal_path)
    expected_sealed[_relative(source_receipt_path, repo)] = file_hash(
        source_receipt_path
    )
    expected_sealed[_relative(source_pre_selection_path, repo)] = file_hash(
        source_pre_selection_path
    )
    fit_inventory_path = target / FIT_INVENTORY
    fit_inventory_value = _object(fit_inventory_path)
    for record in fit_inventory_value["files"]:
        expected_sealed[str(record["path"])] = str(record["sha256"])
    for relative in SOURCE_OPERATIONAL_LOCKS:
        lock_path = target.parent / FIT_SOURCE / relative
        _require(lock_path.stat().st_size == 0, f"source lock is nonempty: {relative}")
        expected_sealed[_relative(lock_path, repo)] = file_hash(lock_path)
    expected_sealed[_relative(fit_inventory_path, repo)] = file_hash(
        fit_inventory_path
    )
    _require(
        len(expected_sealed) == EXPECTED_TRANSITIVE_SEALED_FILE_COUNT
        and artifact.get("sealed_files") == dict(sorted(expected_sealed.items())),
        "v008 transitive authorization sealed_files drift",
    )
    inventory_link = closure["inherited_fit_inventory"]
    _require(
        artifact.get("inherited_fit_role")
        == {
            "source_attempt": FIT_SOURCE,
            "inventory_path": inventory_link["path"],
            "inventory_sha256": inventory_link["sha256"],
            "file_count": inventory_link["file_count"],
            "episode_count": inventory_link["episode_count"],
            "outcome_arrays_opened": False,
            "passed": True,
        },
        "inherited fit-role seal binding drift",
    )

    frozen = invalidity.get("frozen_evidence"); _require(isinstance(frozen, dict), "frozen evidence absent")
    _require(
        frozen.get("pre_data_inheritance_seal_sha256")
        == file_hash(source_seal_path)
        == EXPECTED_SOURCE_SEAL_SHA256
        and frozen.get("version_forward_receipt_sha256")
        == file_hash(source_receipt_path)
        == EXPECTED_SOURCE_RECEIPT_SHA256,
        "frozen v007 seal/receipt drift",
    )
    root_paths = {"program.py": study/"program.py", "README.md": study/"README.md", "LEDGER_CHAIN_GENESIS.json": study/"LEDGER_CHAIN_GENESIS.json"}
    root_controls: dict[str, Any] = {}
    for name, candidate in root_paths.items():
        raw = _relative(candidate, repo); digest = file_hash(candidate)
        root_controls[name] = {"path": raw, "sha256": digest, "bytes": candidate.stat().st_size, "source_seal_bound": source_files.get(raw) == digest}
    _require(root_controls["program.py"]["source_seal_bound"] is True, "program.py not source-seal bound")
    _require(root_controls["LEDGER_CHAIN_GENESIS.json"]["sha256"] == frozen.get("ledger_chain_genesis_sha256"), "genesis invalidity binding drift")
    _require(artifact.get("root_controls") == root_controls, "root control records drift")

    _require(state.get("ledger_event_count") == chain["event_count"] and state.get("ledger_head_sha256") == chain["head_sha256"], "state/ledger binding drift")
    source_state = source_authorization["state"]
    checkpoints = source_state.get("verified_checkpoints")
    _require(
        isinstance(checkpoints, list)
        and len(checkpoints) == len(source_state.get("completed_states", [])) == 9,
        "inherited checkpoint prefix absent or non-exact",
    )
    inherited: list[dict[str, Any]] = []
    for item in checkpoints:
        _require(isinstance(item, dict), "checkpoint record invalid")
        candidate = _resolve(repo, str(item.get("evidence_path")))
        _require(candidate.is_file() and file_hash(candidate) == item.get("evidence_sha256"), "checkpoint evidence drift")
        inherited.append(
            {
                "checkpoint_name": item.get("name"),
                "source_attempt": item.get("source_attempt"),
                "evidence_path": _relative(candidate, repo),
                "evidence_sha256": item.get("evidence_sha256"),
            }
        )
    prior_seal_link = source_seal.get("source_pre_data_seal")
    _require(isinstance(prior_seal_link, Mapping), "v007 seal lost transitive link")
    prior_seal_path = _resolve(repo, str(prior_seal_link.get("path")))
    _require(file_hash(prior_seal_path) == prior_seal_link.get("sha256"), "transitive seal link drift")
    current_seal = _object(prior_seal_path)
    while True:
        direct_science_tip = current_seal.get("source_pre_data_seal")
        _require(isinstance(direct_science_tip, Mapping), "transitive checkpoint-tip link absent")
        direct_path = _resolve(repo, str(direct_science_tip.get("path")))
        _require(file_hash(direct_path) == direct_science_tip.get("sha256"), "transitive checkpoint-tip hash drift")
        if direct_path == target.parent / SCIENCE_ATTEMPT / "audit/pre_data_seal.json":
            break
        current_seal = _object(direct_path)
    _require(
        inherited[5]["evidence_sha256"] == direct_science_tip.get("sha256"),
        "inherited checkpoint tip lost the v001 pre-data seal",
    )
    _require(artifact.get("inherited_verified_checkpoints") == inherited, "inherited checkpoint projection drift")

    snapshot = artifact.get("controller_snapshot"); _require(isinstance(snapshot, dict), "controller snapshot absent")
    source_ledger = source_authorization["ledger"]
    source_genesis = source_authorization["genesis"]
    expected_snapshot = {
        "state_path": _relative(study/"STATE.json", repo),
        "state_sha256": EXPECTED_SOURCE_STATE_SHA256,
        "ledger_path": _relative(study/"RESEARCH_LEDGER.jsonl", repo),
        "event_count": source_ledger["event_count"],
        "head_sha256": source_ledger["head_sha256"],
        "ledger_sha256": EXPECTED_SOURCE_LEDGER_SHA256,
        "genesis_sha256": source_genesis["sha256"],
    }
    _require(snapshot == expected_snapshot, "pre-forward controller snapshot drift")
    if phase == "pre-forward":
        _require(
            authorized_early_verifier_state is None
            and authorized_role_count_recovery is None,
            "exceptional authorization is invalid during pre-forward verification",
        )
        _require(
            dict(state) == source_state
            and state.get("active_attempt") == SOURCE
            and state.get("active_attempt_path") == _relative(source, repo)
            and state.get("current_state") == RESUME,
            "pre-forward state identity drift",
        )
        _require(
            _exact_typed_mapping({key: state.get(key) for key in EXPECTED_COUNTS}, EXPECTED_COUNTS),
            "pre-forward outcome counters drift",
        )
        _require(file_hash(study/"STATE.json") == snapshot["state_sha256"] and chain["ledger_sha256"] == snapshot["ledger_sha256"], "pre-forward snapshot is not current")
    else:
        _validate_post_forward_raw_authorization_state(
            state,
            authorized_early_verifier_state=authorized_early_verifier_state,
            authorized_role_count_recovery=authorized_role_count_recovery,
        )
        _require(
            state.get("active_attempt_path") == _relative(target, repo)
            and "v007_durable_controller_adapter" not in state,
            "post-forward active root or retired v007 adapter marker drift",
        )
        ordinary_role = {
            "FIT_COHORTS": "fit",
            "SELECTION_COHORTS": "selection",
        }.get(str(state.get("current_state")))
        if (
            authorized_early_verifier_state is None
            and authorized_role_count_recovery is None
            and ordinary_role is not None
        ):
            _require(
                not os.path.lexists(
                    target / "audit" / f"{ordinary_role}_cohorts.json"
                ),
                "ordinary role authorization found a pre-count role audit",
            )
        current_state_without_marker = json.loads(json.dumps(dict(state)))
        current_state_without_marker.pop(
            "v008_durable_controller_adapter", None
        )
        _verify_recovery_adapter_marker(
            state.get("v008_durable_controller_adapter"),
            event_count=len(events),
            head_sha256=str(state.get("ledger_head_sha256")),
            target=target,
            repo=repo,
            label="current STATE",
            state_without_marker=current_state_without_marker,
        )
        _verify_descendant_adapter_transaction_chain(
            state=state,
            events=events,
            ledger_payload=(study / "RESEARCH_LEDGER.jsonl").read_bytes(),
            target=target,
            repo=repo,
            seal_path=seal_path,
            source_state=source_authorization["state"],
            source_ledger=source_authorization["ledger"],
            authorized_transaction_context_sha256=(
                authorized_transaction_context_sha256
            ),
        )
        _verify_adapter_count_event_bindings(
            events,
            target=target,
            repo=repo,
        )
        if authorized_role_count_recovery is not None:
            _verify_role_count_recovery_authorization(
                state,
                events,
                role=authorized_role_count_recovery,
                target=target,
                repo=repo,
            )
        lineage = state.get("version_forward_lineage")
        prior_lineage = source_authorization["lineage"]
        _require(
            isinstance(lineage, list)
            and len(lineage) == 7
            and lineage[:-1] == prior_lineage,
            "post-forward lineage did not preserve the exact transitive prefix",
        )
        edge = lineage[-1]
        seal_rel = _relative(seal_path, repo)
        seal_digest = file_hash(seal_path)
        _require(
            isinstance(edge, Mapping)
            and set(edge) == {
                "old_attempt", "new_attempt", "resume_state",
                "invalidity_path", "invalidity_sha256", "equivalence_path",
                "equivalence_sha256", "inherited_verified_checkpoints",
                "attempt_parameterization_verified",
            }
            and edge.get("old_attempt") == SOURCE
            and edge.get("new_attempt") == TARGET
            and edge.get("resume_state") == RESUME
            and edge.get("inherited_verified_checkpoints") == inherited
            and edge.get("attempt_parameterization_verified") is True,
            "state version-forward edge identity/schema drift",
        )
        _require(edge.get("invalidity_path") == _relative(invalid_path, repo) and edge.get("invalidity_sha256") == file_hash(invalid_path), "state invalidity edge drift")
        _require(edge.get("equivalence_path") == seal_rel and edge.get("equivalence_sha256") == seal_digest, "state equivalence edge drift")
        history = state.get("attempt_history")
        source_history = source_authorization["history"]
        _require(
            isinstance(history, list)
            and len(history) == 8
            and [
                item.get("version") for item in history
                if isinstance(item, Mapping)
            ] == [
                SCIENCE_ATTEMPT,
                "v002",
                FIT_SOURCE,
                INTERMEDIATE,
                SOURCE_PARENT,
                SELECTION_SOURCE,
                SOURCE,
                TARGET,
            ]
            and history[0] == source_history[0]
            and history[1] == source_history[1]
            and history[2] == source_history[2]
            and history[3] == source_history[3]
            and history[4] == source_history[4]
            and history[5] == source_history[5]
            and {
                key: value
                for key, value in history[6].items()
                if key not in {
                    "status", "invalidity_evidence_path",
                    "invalidity_evidence_sha256",
                }
            }
            == {
                key: value
                for key, value in source_history[6].items()
                if key not in {
                    "status", "invalidity_evidence_path",
                    "invalidity_evidence_sha256",
                }
            }
            and history[6].get("status")
            == "invalid_zero_confirmation_outcome_procedural"
            and history[6].get("invalidity_evidence_path")
            == _relative(invalid_path, repo)
            and history[6].get("invalidity_evidence_sha256")
            == file_hash(invalid_path)
            and history[7].get("path") == _relative(target, repo)
            and set(history[7]) == {
                "version", "path", "status", "created_unix_ns",
                "version_forward_evidence_path",
                "version_forward_evidence_sha256",
                "attempt_parameterization_verified",
            }
            and type(history[7].get("created_unix_ns")) is int
            and history[7].get("status")
            == "active_zero_confirmation_outcome_version_forward"
            and history[7].get("version_forward_evidence_path") == seal_rel
            and history[7].get("version_forward_evidence_sha256") == seal_digest
            and history[7].get("attempt_parameterization_verified") is True,
            "post-forward attempt-history transitive lineage drift",
        )
        vf_events = [item for item in events if item.get("event") == "zero_confirmation_outcome_version_forward"]
        source_vf_events = [
            item
            for item in events[: int(source_ledger["event_count"])]
            if item.get("event") == "zero_confirmation_outcome_version_forward"
        ]
        _require(
            len(vf_events) == 7
            and len(source_vf_events) == 6
            and vf_events[:-1] == source_vf_events
            and source_vf_events[-1].get("old_attempt") == SELECTION_SOURCE
            and source_vf_events[-1].get("new_attempt") == SOURCE
            and source_vf_events[-1].get("equivalence_sha256")
            == EXPECTED_SOURCE_SEAL_SHA256
            and vf_events[-1].get("attempt") == TARGET
            and vf_events[-1].get("old_attempt") == SOURCE
            and vf_events[-1].get("new_attempt") == TARGET
            and vf_events[-1].get("invalidity_path")
            == _relative(invalid_path, repo)
            and vf_events[-1].get("invalidity_sha256")
            == file_hash(invalid_path)
            and vf_events[-1].get("equivalence_path") == seal_rel
            and vf_events[-1].get("equivalence_sha256") == seal_digest
            and vf_events[-1].get("source_controller_adapter_marker")
            == source_authorization["marker"]
            and set(vf_events[-1]) == {
                "event", "attempt", "old_attempt", "new_attempt",
                "invalidity_path", "invalidity_sha256", "equivalence_path",
                "equivalence_sha256", "resume_state",
                "inherited_verified_checkpoints",
                "attempt_parameterization_verified", "created_unix_ns",
                "source_controller_adapter_marker",
                "version_forward_receipt_context_sha256", "seq",
                "prev_sha256", "record_sha256",
            }
            and vf_events[-1].get("inherited_verified_checkpoints") == inherited
            and vf_events[-1].get("attempt_parameterization_verified") is True
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(vf_events[-1].get("version_forward_receipt_context_sha256")),
            ) is not None,
            "seven-edge ledger version-forward lineage drift",
        )

    by_path: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for mode in ("exact_hash", "normalized_ast", "procedural_only"):
        for item in partitions[mode]:
            by_path[str(item["relative_path"])] = (mode, item)
    _require(SCIENCE.issubset(by_path), "scientific objects absent from partition")
    expected_science: list[dict[str, Any]] = []
    for relative in sorted(SCIENCE):
        mode, item = by_path[relative]
        record = {
            "relative_path": relative, "source_path": item["source_path"], "target_path": item["target_path"],
            "equivalence_mode": mode, "source_sha256": item.get("source_sha256", item.get("sha256")),
            "target_sha256": item.get("target_sha256", item.get("sha256")),
            "classification": "seed_identifier_or_configuration" if Path(relative).suffix in {".json", ".md"} else "numerical_or_endpoint_source",
        }
        if mode == "normalized_ast": record["normalized_ast_sha256"] = item["normalized_ast_sha256"]
        expected_science.append(record)
    _require(artifact.get("scientific_objects") == expected_science, "scientific object records drift")
    expected_checks = {
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
    _require(
        _exact_typed_mapping(artifact.get("checks"), expected_checks),
        "check summary is not exact",
    )
    return {
        "passed": True, "phase": phase, "attempt": TARGET,
        "active_attempt": TARGET, "science_attempt": SCIENCE_ATTEMPT,
        "seal_path": _relative(seal_path, repo), "seal_sha256": file_hash(seal_path),
        "partition_file_count": len(closure["discovered_paths"]), "sealed_file_count": len(expected_sealed),
        "authorized_early_verifier_state": authorized_early_verifier_state,
        "authorized_role_count_recovery": authorized_role_count_recovery,
        "outcome_arrays_opened": False, "output_paths_created": 0, "read_only": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pre-forward", "post-forward"))
    parser.add_argument("--study-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seal", type=Path)
    parser.add_argument(
        "--authorized-early-verifier-state",
        choices=(
            "CANDIDATE_SELECTION",
            "CONFIRMATION_POWER_AND_COHORT_FREEZE",
        ),
    )
    parser.add_argument(
        "--authorized-role-count-recovery",
        choices=("selection",),
    )
    args = parser.parse_args(argv)
    seal = args.seal or (args.study_root.resolve()/"attempts"/TARGET/OUTPUT)
    print(
        json.dumps(
            verify(
                seal,
                args.study_root,
                phase=args.phase,
                authorized_early_verifier_state=args.authorized_early_verifier_state,
                authorized_role_count_recovery=args.authorized_role_count_recovery,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
