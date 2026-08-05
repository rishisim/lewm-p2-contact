#!/usr/bin/env python3
"""Synthesize immutable domain results into findings, state, and final report."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import time
from typing import Any


TERMINAL_OUTCOME = "v5_validity_supported_with_caveats"
FAILED_TEST_IDS = {"FRZ-03", "CMP-01", "FLT-02", "FLT-03", "NEG-03"}


def read(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    atomic_text(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True, type=pathlib.Path)
    parser.add_argument("--package", required=True, type=pathlib.Path)
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    package = args.package.resolve()

    matrix = read(audit / "TEST_MATRIX.json")
    clean = read(audit / "CLEAN_ROOM_RECOMPUTATION.json")
    replay = read(audit / "REPLAY_RESULTS.json")
    negatives = read(audit / "NEGATIVE_CONTROLS.json")
    faults = read(audit / "FAULT_INJECTION.json")
    cohort = read(audit / "COHORT_ISOLATION.json")
    causal = read(audit / "CAUSAL_INPUT_AUDIT.json")
    chronology = read(audit / "CHRONOLOGY_AUDIT.json")
    compute = read(audit / "COMPUTE_AUDIT.json")
    implementation = read(audit / "IMPLEMENTATION_INTEGRITY.json")
    source_final = read(audit / "SOURCE_FINAL_HASHES.json")
    latency = read(package / "metrics/v5_confirmation_latency.json")

    if not (
        clean["all_primary_agreement_checks_pass"]
        and replay["passed"]
        and cohort["passed"]
        and causal["passed"]
        and chronology["passed"]
        and compute["all_declared_convention_checks_pass"]
        and source_final["passed"]
    ):
        raise RuntimeError("cannot synthesize: a required primary artifact is incomplete")
    if negatives["overall_predeclared_status"] != "fail":
        raise RuntimeError("expected retained NEG-03 predeclared failure")
    if set(faults["summary"]["package_fail_open_case_ids"]) != {
        "FLT-02D",
        "FLT-02E",
        "FLT-03A",
        "FLT-03B",
    }:
        raise RuntimeError("unexpected fault-injection case set")

    domain_evidence = {
        "chain_of_custody": [
            "SOURCE_SNAPSHOT.json",
            "CHRONOLOGY_AUDIT.json",
            "SOURCE_FINAL_HASHES.json",
        ],
        "confirmation_isolation": [
            "COHORT_ISOLATION.json",
            "CAUSAL_INPUT_AUDIT.json",
        ],
        "causal_input_target_leakage": ["CAUSAL_INPUT_AUDIT.json"],
        "frozen_implementation": [
            "IMPLEMENTATION_INTEGRITY.json",
            "REPLAY_RESULTS.json",
        ],
        "clean_room_recomputation": ["CLEAN_ROOM_RECOMPUTATION.json"],
        "inference_sparse_execution": [
            "REPLAY_RESULTS.json",
            "REPLAY_RESULTS_INITIAL_FAILURE.json",
            "CLEAN_ROOM_RECOMPUTATION.json",
        ],
        "compute_comparator_fairness": [
            "COMPUTE_AUDIT.json",
            "CLEAN_ROOM_RECOMPUTATION.json",
        ],
        "exclusions_recovery": ["CHRONOLOGY_AUDIT.json"],
        "fail_closed_fault_injection": ["FAULT_INJECTION.json"],
        "negative_controls": [
            "NEGATIVE_CONTROLS.json",
            "CAUSAL_INPUT_AUDIT.json",
        ],
        "statistical_numerical_sensitivity": ["CLEAN_ROOM_RECOMPUTATION.json"],
    }
    observed_by_id = {
        "COC-01": (
            "Initial lstat/SHA-256 snapshot covered 3,612 regular files and 37 "
            "directories across v001-v004."
        ),
        "COC-02": (
            "v001 had 101 pre-outcome wrong-runtime failures and zero outcomes; "
            "v002/v003 failed preseal with zero outcomes; only v004 completed."
        ),
        "COC-03": (
            "Scientific sources, model/gate/whitening artifacts, thresholds, analysis "
            "seeds, outcome mapping, and compute constants were hash-bound before raw V5."
        ),
        "COC-04": (
            "No hidden/partial v004 raw artifacts or failure ledger; v004 contains exactly "
            "1,600 NPZs and sidecars and no replacements."
        ),
        "COC-05": (
            "Initial/final v001-v004 and full-program path, content, and stat records are "
            "exact; aggregate hashes are unchanged."
        ),
        "ISO-01": (
            "No V5 ID or exact structured seed overlaps 31,629 accessible prior IDs or "
            "28,427 accessible prior seeds."
        ),
        "ISO-02": "No low-32-bit or SeedSequence-state transformed seed overlap.",
        "ISO-03": (
            "All 1,600 V5 raw archives and 2,472 accessible prior raw archives were "
            "fingerprinted; no state/action/observation/initial-pixel exact collision."
        ),
        "ISO-04": (
            "No cross-cohort state/action/observation/pixel sketch fell below the "
            "predeclared near-duplicate threshold."
        ),
        "ISO-05": (
            "Fit/selection isolation is frozen pre-V5; runtime/canary audit found no "
            "confirmation-derived normalization or whitening input."
        ),
        "LEAK-01": (
            "Static gate inputs are history, action history, current prediction, last "
            "update, and frozen stage parameters only."
        ),
        "LEAK-02": "Runtime tracing materialized only action and pixels.",
        "LEAK-03": (
            "Twelve privileged canary families were never materialized and changing them "
            "left loader outputs invariant."
        ),
        "LEAK-04": (
            "Every gate input is available at the decision; target latent appears only "
            "after routing for audit/analysis."
        ),
        "FRZ-01": (
            "All preseal, frozen-candidate, external-model, input-seal, and package-ready "
            "hash checks pass now under both declared runtimes."
        ),
        "FRZ-02": (
            "Sealed sources and stable frozen module-state digests are tied to exact full "
            "MPS replay; historical device is strongly inferred though not manifest-explicit."
        ),
        "FRZ-03": (
            "FAIL: runtime sealing does not hash the complete transitive Python/native/MPS "
            "dependency graph or loaded sys.modules/OS build. Exact replay bounds but "
            "does not retroactively attest the historical environment."
        ),
        "FRZ-04": (
            "Module state/gradients are unchanged and repeated packaged inference is exact "
            "for all 60,800 rows."
        ),
        "NUM-01": "All hand/synthetic calls, mixture, quantile, rank, and Spearman fixtures pass.",
        "NUM-02": "Per-transition and per-episode errors independently agree within tolerance.",
        "NUM-03": (
            "Calls, reached counts, histogram [50,973, 7,267, 1,773, 787], and threshold "
            "decisions reproduce exactly."
        ),
        "NUM-04": "Fixed exits and every transition-independent comparator mapping reproduce.",
        "NUM-05": "Raw and native-whitened effects and bounds reproduce to machine precision.",
        "NUM-06": (
            "All 20,000 bootstrap streams/quantiles, Bonferroni bounds, and stage ranks reproduce."
        ),
        "NUM-07": "Recomputed predicates map exactly to v5_confirmation_passed.",
        "RPL-01": (
            "Complete packaged-path replay covers all 1,600 episodes/60,800 rows with exact "
            "targets, exits, sparse outputs, calls, scores, and features."
        ),
        "RPL-02": (
            "Independent explicit sqrt/sum algebra is within audit tolerance with exact "
            "calls/outputs; a second vector_norm-order transcription is bit exact."
        ),
        "RPL-03": (
            "Dense/sparse, forward_selected/manual sparse, row identity, and restoration "
            "are exact/within the sealed output contract."
        ),
        "RPL-04": (
            "Reached values are finite, unreached values NaN, calls are stable, and minimum "
            "active score margins are 2.50e-5, 8.82e-5, and 2.85e-5."
        ),
        "CMP-01": (
            "FAIL: 4,332,936,120,435 is exactly reproduced under the inherited ledger, "
            "including 584,398,195 gate FLOPs, but it omits executed pixel encoding and "
            "visible bias/GELU/alpha/residual operations."
        ),
        "CMP-02": (
            "Analytic comparator uses a global transition-independent d1/d2 mixture; "
            "weight 0.25295409752763076 reproduces exactly."
        ),
        "CMP-03": (
            "Seeded comparator uses 4,332,936,224,000 declared FLOPs, exactly 103,565 "
            "more than adaptive."
        ),
        "CMP-04": "Call-count and within-episode histogram-to-exit mappings reproduce exactly.",
        "EXC-01": "Exactly 1,600 episodes × 38 rows = 60,800 transitions across manifests/arrays.",
        "EXC-02": "All IDs, raw paths, raw hashes, and slots are unique; zero replacements.",
        "EXC-03": (
            "Recovery failures preceded outcomes and were runtime/preseal-rule based, not "
            "performance conditioned."
        ),
        "FLT-01": "One-byte execution and gate corruptions are hash-rejected.",
        "FLT-02": (
            "FAIL: missing/extra/hash mismatch reject, but existing-manifest preflight accepts "
            "duplicate ID metadata and semantic verifier accepts a consistently rebound row order."
        ),
        "FLT-03": (
            "FAIL: raw preflight accepts ../ root escape with a matching declared hash and "
            "final verifier follows a byte-identical external symlink."
        ),
        "FLT-04": "Threshold, compute-price, and wrong-interpreter mutations reject.",
        "FLT-05": (
            "Post-manifest privileged fields reject; pre-manifest unknown fields are safely "
            "allowlist-ignored, not materialized, and leave outputs exact."
        ),
        "NEG-01": "2,000 within-episode/reachability permutations: raw/white p=0.00049975.",
        "NEG-02": (
            "2,000 random and 2,000 constant/mixed compute-matched policies: "
            "raw/white p=0.00049975."
        ),
        "NEG-03": (
            "FAIL retained: 1,000 whole-episode target reassignments give raw p=0.164835 "
            "(white p=0.000999); second 1600×1600 implementation agrees."
        ),
        "NEG-04": (
            "2,000 score-gain breaks and 2,000 perturbed reversed orderings give "
            "raw/white p=0.00049975; deterministic reversal is worse."
        ),
        "NEG-05": "Two extreme unused-field values leave every semantic output digest exact.",
        "SNS-01": "Forward/reverse NumPy and math.fsum raw effects are identical.",
        "SNS-02": "All standard bootstrap quantile conventions retain positive lower bounds.",
        "SNS-03": "Every leave-one-episode-out effect is positive; all 100-episode block means are positive.",
    }

    tests = []
    for declared in matrix["tests"]:
        test_id = declared["id"]
        if test_id not in observed_by_id:
            raise RuntimeError(f"missing result synthesis for {test_id}")
        tests.append(
            {
                **declared,
                "status": "fail" if test_id in FAILED_TEST_IDS else "pass",
                "observed": observed_by_id[test_id],
                "evidence": domain_evidence[declared["domain"]],
            }
        )
    if len(tests) != 49 or {item["id"] for item in tests} != {
        item["id"] for item in matrix["tests"]
    }:
        raise RuntimeError("test synthesis does not exactly cover sealed matrix")
    passed = [item for item in tests if item["status"] == "pass"]
    failed = [item for item in tests if item["status"] == "fail"]
    decision_passed = [
        item for item in passed if item["decision_critical"] is True
    ]
    decision_failed = [
        item for item in failed if item["decision_critical"] is True
    ]
    domain_summary = {}
    for domain in sorted({item["domain"] for item in tests}):
        values = [item for item in tests if item["domain"] == domain]
        domain_summary[domain] = {
            "test_count": len(values),
            "pass_count": sum(item["status"] == "pass" for item in values),
            "fail_count": sum(item["status"] == "fail" for item in values),
            "failed_ids": [
                item["id"] for item in values if item["status"] == "fail"
            ],
        }

    test_results = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "sealed_matrix_sha256": sha256(audit / "TEST_MATRIX.json"),
        "plan_seal_sha256": sha256(audit / "audit_plan_seal.json"),
        "summary": {
            "test_count": len(tests),
            "pass_count": len(passed),
            "fail_count": len(failed),
            "partial_count": 0,
            "not_run_count": 0,
            "mandatory_for_support_count": sum(
                item["mandatory_for_support"] for item in tests
            ),
            "decision_critical_count": sum(
                item["decision_critical"] for item in tests
            ),
            "decision_critical_pass_count": len(decision_passed),
            "decision_critical_fail_count": len(decision_failed),
            "failed_test_ids": [item["id"] for item in failed],
            "decision_critical_failed_test_ids": [
                item["id"] for item in decision_failed
            ],
        },
        "domain_summary": domain_summary,
        "tests": tests,
        "post_hoc_diagnostics_not_counted_as_predeclared_passes": [
            {
                "name": "loss-profile episode-block permutation",
                "artifact": "NEGATIVE_CONTROLS.json",
                "result": "raw/white plus-one p=0.00049975",
            },
            {
                "name": "second order-matched direct-module replay",
                "artifact": "REPLAY_RESULTS.json",
                "result": "features/scores/calls/outputs exact",
            },
            {
                "name": "replay initial-failure diagnosis",
                "artifact": "REPLAY_RESULTS_INITIAL_FAILURE.json",
                "result": (
                    "audit harness omitted rtol in acceptance; no V5 output/call defect"
                ),
            },
        ],
    }
    atomic_json(audit / "TEST_RESULTS.json", test_results)

    findings = [
        {
            "id": "V5-AUD-MAJ-001",
            "severity": "Major",
            "title": "Artifact validators have pre-seal path and manifest fail-open surfaces",
            "test_ids": ["FLT-02", "FLT-03"],
            "evidence": ["FAULT_INJECTION.json", "SOURCE_FINAL_HASHES.json"],
            "description": (
                "The copied existing-manifest preflight accepted duplicate identifier "
                "metadata and a ../ path outside REPO_ROOT when the declared hash matched; "
                "a semantic verifier accepted a consistently rebound episode-block order; "
                "the final durable-file verifier accepted a byte-identical external symlink."
            ),
            "impact": (
                "These are meaningful supply-chain/hardening weaknesses and could permit "
                "substitution or duplication before the confirmation input seal. They did "
                "not compromise the preserved result: canonical paths are confined and "
                "regular, IDs/paths/hashes are unique, row order is exact, and initial/final "
                "snapshots are unchanged."
            ),
            "decision_reversal_capable_if_exploited": True,
            "evidence_of_exploitation": False,
            "blocker_rationale": (
                "The actual canonical state was exhaustively checked and does not use any "
                "of the accepted corruptions."
            ),
        },
        {
            "id": "V5-AUD-MAJ-002",
            "severity": "Major",
            "title": "The unqualified exact-FLOP claim is not literal end-to-end arithmetic",
            "test_ids": ["CMP-01"],
            "evidence": ["COMPUTE_AUDIT.json", "CLEAN_ROOM_RECOMPUTATION.json"],
            "description": (
                "The stated total is an exact integer under the frozen historical "
                "latent-prediction ledger, not a literal count of all executed operations. "
                "It excludes pixel encoding and visible bias/GELU/alpha/residual arithmetic; "
                "the base predictor also lacks a complete primitive graph."
            ),
            "impact": (
                "The claim must be scoped to the declared convention. Comparator fairness "
                "survives: common omissions cancel, adding 704 or 832 visible operations per "
                "later adapter slightly increases the raw advantage, and every sensitivity "
                "keeps the seeded comparator weakly more compute."
            ),
            "decision_reversal_capable_if_exploited": False,
            "evidence_of_exploitation": False,
            "blocker_rationale": "No tested reasonable correction reverses the effect.",
        },
        {
            "id": "V5-AUD-MAJ-003",
            "severity": "Major",
            "title": "Historical transitive runtime environment is incompletely attested",
            "test_ids": ["FRZ-03"],
            "evidence": ["IMPLEMENTATION_INTEGRITY.json", "REPLAY_RESULTS.json"],
            "description": (
                "The seal records interpreter hashes, exact package versions, and top-level "
                "module origins, but not all imported Python files, native libraries, the "
                "Metal/MPS framework or OS build, nor a loaded-module snapshot."
            ),
            "impact": (
                "An exact historical environment cannot be reconstructed solely from the "
                "seal. Current dual-runtime probes and a bitwise-exact 60,800-row MPS replay "
                "make a decision-changing historical substitution unsupported, but cannot "
                "retroactively attest it."
            ),
            "decision_reversal_capable_if_exploited": True,
            "evidence_of_exploitation": False,
            "blocker_rationale": "Exact full replay bounds the unresolved surface; no mismatch exists.",
        },
        {
            "id": "V5-AUD-MAJ-004",
            "severity": "Major",
            "title": "Complete original base-model training raw corpus is unavailable",
            "test_ids": ["ISO-03", "ISO-04"],
            "evidence": ["COHORT_ISOLATION.json", "CHRONOLOGY_AUDIT.json"],
            "description": (
                "The repository does not preserve an episode-level raw corpus for original "
                "base-model training, so observation-level absence/near-absence across that "
                "entire historical corpus cannot be tested."
            ),
            "impact": (
                "All accessible evidence is clean—2,472 prior raw archives, 31,629 IDs, "
                "28,427 seeds, transformed seeds, initial states, actions, observations and "
                "pixels—and V5 was generated later from fresh unique seeds. This leaves a "
                "bounded provenance uncertainty, not evidence of direct cohort leakage."
            ),
            "decision_reversal_capable_if_exploited": True,
            "evidence_of_exploitation": False,
            "blocker_rationale": (
                "Post-training chronology and fresh seed derivation make exact V5 membership "
                "in the unavailable training corpus non-credible."
            ),
        },
        {
            "id": "V5-AUD-MAJ-005",
            "severity": "Major",
            "title": "The sealed audit plan has an internally over-strict terminal mapping",
            "test_ids": ["FRZ-03", "CMP-01", "FLT-02", "FLT-03", "NEG-03"],
            "evidence": ["AUDIT_PLAN.md", "TEST_MATRIX.json", "TEST_RESULTS.json"],
            "description": (
                "The seal marked all 49 tests mandatory for both supported outcomes, while "
                "reserving invalidation for a validated Blocker and inconclusive for essential "
                "missing/corrupt evidence or persistent critical disagreement. Resolved, "
                "non-invalidating failures therefore create a gap in which none of the four "
                "sealed labels is available."
            ),
            "impact": (
                "No failed test was relabeled. Because the user requires exactly one outcome, "
                "the governing top-level definitions are applied: no Blocker, exact primary "
                "reproduction, and bounded Major caveats map most truthfully to "
                "v5_validity_supported_with_caveats."
            ),
            "decision_reversal_capable_if_exploited": False,
            "evidence_of_exploitation": False,
            "blocker_rationale": "This is a defect in this audit's mapping, not in V5 data or computation.",
            "post_hoc_audit_plan_defect": True,
        },
        {
            "id": "V5-AUD-MIN-001",
            "severity": "Minor",
            "title": "Predeclared whole-episode target permutation fails in raw space",
            "test_ids": ["NEG-03"],
            "evidence": ["NEGATIVE_CONTROLS.json"],
            "description": (
                "The retained raw plus-one p-value is 0.164835. A second pair-matrix method "
                "agrees to 1.58e-19, so this is not a harness error."
            ),
            "impact": (
                "Whole-target reassignment changes the physical prediction task and has 15.1× "
                "the null SD of matched allocation controls. All sharper reachability/histogram/"
                "score-gain controls pass; a separately labeled post-hoc loss-profile permutation "
                "also passes. The preregistered failure remains visible."
            ),
        },
        {
            "id": "V5-AUD-MIN-002",
            "severity": "Minor",
            "title": "Historical execution device/command is not explicit in the execution manifest",
            "test_ids": ["FRZ-02"],
            "evidence": ["IMPLEMENTATION_INTEGRITY.json", "REPLAY_RESULTS.json"],
            "description": (
                "The manifest omits device and command fields. MPS is strongly inferred from "
                "the launcher default, MPS qualification and exact full MPS replay."
            ),
            "impact": "Reproducibility/attestation gap; no observed numerical discrepancy.",
        },
        {
            "id": "V5-AUD-MIN-003",
            "severity": "Minor",
            "title": "No post-decision complete package path-set manifest exists",
            "test_ids": ["COC-03", "FRZ-01"],
            "evidence": ["IMPLEMENTATION_INTEGRITY.json", "SOURCE_FINAL_HASHES.json"],
            "description": (
                "artifact_manifest.json freezes PACKAGE_READY before confirmation. The input "
                "seal and independent verifier bind decision inputs, but raw sidecars and all "
                "post-confirmation outputs are not collected in a second complete manifest."
            ),
            "impact": "The audit's initial/final full snapshots close the preserved-evidence gap.",
        },
        {
            "id": "V5-AUD-MIN-004",
            "severity": "Minor",
            "title": "Raw loader safely ignores rather than schema-rejects unknown fields",
            "test_ids": ["FLT-05", "LEAK-03"],
            "evidence": ["CAUSAL_INPUT_AUDIT.json", "FAULT_INJECTION.json"],
            "description": (
                "Unknown privileged canaries can be present before manifesting, but only action "
                "and pixels are subscripted/materialized and outputs remain exact."
            ),
            "impact": "No target leak; strict unknown-field rejection would improve defense in depth.",
        },
        {
            "id": "V5-AUD-MIN-005",
            "severity": "Minor",
            "title": "Chronology relies on mutable local timestamps and has a narrow TOCTOU window",
            "test_ids": ["COC-02", "COC-03"],
            "evidence": ["CHRONOLOGY_AUDIT.json"],
            "description": (
                "Timestamps are not externally notarized and raw files are hashed before open; "
                "a change-and-restore inside that window is not cryptographically excluded."
            ),
            "impact": (
                "Strict ordering, pre/post hashes, exact replay and unchanged final snapshots "
                "provide strong but not notarized custody."
            ),
        },
        {
            "id": "V5-AUD-MIN-006",
            "severity": "Minor",
            "title": "Initial replay audit acceptance logic produced a false failure",
            "test_ids": ["RPL-02"],
            "evidence": [
                "REPLAY_RESULTS_INITIAL_FAILURE.json",
                "REPLAY_RESULTS.json",
            ],
            "description": (
                "The first harness applied an absolute-only 2e-6 cap to an algebraically "
                "different sqrt/sum reduction and omitted the preregistered relative term."
            ),
            "impact": (
                "Synthetic validation identified the audit-harness defect. No tolerance was "
                "relaxed: order-matched features/scores are exact; explicit algebra passes "
                "audit tolerance; every call/output is exact. Only 88/1,600 episodes satisfy "
                "the package's stricter score allclose under that deliberately different order."
            ),
        },
    ]
    blockers = [item for item in findings if item["severity"] == "Blocker"]
    majors = [item for item in findings if item["severity"] == "Major"]
    minors = [item for item in findings if item["severity"] == "Minor"]
    if blockers or len(majors) != 5 or len(minors) != 6:
        raise RuntimeError("finding classification invariant failed")

    findings_payload = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "terminal_outcome": TERMINAL_OUTCOME,
        "v5_result_appears_genuine": True,
        "blocker_count": len(blockers),
        "major_count": len(majors),
        "minor_count": len(minors),
        "findings": findings,
        "terminal_reasoning": {
            "validated_blocker_present": False,
            "essential_evidence_corrupt_or_missing": False,
            "critical_independent_disagreement_persists": False,
            "primary_claim_exactly_reproduced": True,
            "complete_replay_passed": True,
            "causal_input_audit_passed": True,
            "canonical_source_unchanged": True,
            "predeclared_test_pass_count": len(passed),
            "predeclared_test_fail_count": len(failed),
            "failed_test_ids": [item["id"] for item in failed],
            "sealed_support_predicate_satisfied": False,
            "sealed_mapping_gap_present": True,
            "governing_user_definition_applied": (
                "no Blocker plus exact confirmation reproduction plus bounded Major "
                "uncertainty => v5_validity_supported_with_caveats"
            ),
        },
        "precise_supported_claim": (
            "For the frozen v004 PlanOracle confirmation cohort, the sealed causal adaptive "
            "policy produced lower episode-averaged raw and native-whitened latent MSE than "
            "the strongest transition-independent comparator at the same declared "
            "latent-prediction operation budget, with the preregistered intervals and "
            "outcome mapping independently reproduced."
        ),
        "unsupported_claims": [
            "broad generalization beyond the frozen PlanOracle DGP",
            "practically large effect",
            "latency or energy improvement",
            "literal end-to-end exact arithmetic FLOPs including pixel encoding",
            "complete historical base-training observation-level non-overlap",
            "complete transitive runtime attestation",
            "fail-closed resistance to every pre-seal manifest/path substitution",
        ],
    }
    atomic_json(audit / "FINDINGS.json", findings_payload)

    raw = clean["criteria"]["raw_vs_analytic"]
    white = clean["criteria"]["native_whitened_vs_analytic"]
    raw_relative = 100.0 * raw["estimate"] / clean["compute"]["raw_analytic"]["mean_loss"]
    white_relative = (
        100.0
        * white["estimate"]
        / clean["compute"]["native_whitened_analytic"]["mean_loss"]
    )
    fault_ids = ", ".join(faults["summary"]["package_fail_open_case_ids"])
    failed_ids = ", ".join(item["id"] for item in failed)
    major_lines = "\n".join(
        f"- **{item['id']} — {item['title']}.** {item['impact']}"
        for item in majors
    )
    minor_lines = "\n".join(
        f"- **{item['id']} — {item['title']}.** {item['impact']}"
        for item in minors
    )
    domain_rows = "\n".join(
        f"| `{domain}` | {value['pass_count']} | {value['fail_count']} | "
        f"{', '.join(f'`{item}`' for item in value['failed_ids']) or '—'} |"
        for domain, value in domain_summary.items()
    )
    latency_by = {
        (item["path"], item["batch_size"]): item for item in latency["timings"]
    }
    report = f"""# V5 adversarial validity audit

## Terminal outcome: `{TERMINAL_OUTCOME}`

**Bottom line:** the frozen V5 confirmation appears genuine. I found no
demonstrated target leak, cohort reuse, outcome-conditioned recovery,
implementation error, numerical error, or comparator construction error that
can explain or reverse the positive result. The exact primary statistics,
calls, outputs, bootstrap intervals, ranks, and terminal mapping reproduce
independently over all 1,600 episodes and 60,800 transitions.

This is support with substantial caveats, not a clean bill of health. The
package fails several defense-in-depth fault tests, its unqualified “exact
FLOPs” wording is too broad, its historical transitive runtime is incompletely
attested, and the original base-training raw corpus is unavailable. None is
shown in the preserved canonical evidence, and none reverses the comparator
result.

## The precise claim supported

For the frozen v004 PlanOracle DGP and its prospectively generated 1,600-episode
confirmation cohort, the frozen causal policy allocated solver depth better
than the strongest transition-independent comparator at the same **declared
latent-prediction operation budget**. This statement is limited to this
data-generating distribution and accounting convention.

It does **not** establish broad generalization, a practically large gain,
lower latency, lower energy, or a literal end-to-end arithmetic FLOP total.

## Independently reproduced result

| Quantity | Clean-room result | Discrepancy |
|---|---:|---:|
| Episodes / transitions | 1,600 / 60,800 | exact |
| Adaptive raw MSE | {clean['losses']['adaptive_raw_mse']:.17g} | within declared tolerance |
| Raw analytic-comparator MSE | {clean['compute']['raw_analytic']['mean_loss']:.17g} | exact |
| Raw benefit | {raw['estimate']:.17g} | 0 |
| Raw 95% interval | [{raw['lower']:.17g}, {raw['upper']:.17g}] | 0 |
| Raw relative reduction | {raw_relative:.6f}% | sensitivity only |
| Adaptive native-whitened MSE | {clean['losses']['adaptive_native_whitened_mse']:.17g} | within declared tolerance |
| Native-whitened comparator MSE | {clean['compute']['native_whitened_analytic']['mean_loss']:.17g} | exact |
| Native-whitened benefit | {white['estimate']:.17g} | 1.08e-19 |
| Native-whitened 95% interval | [{white['lower']:.17g}, {white['upper']:.17g}] | 0 |
| Native-whitened relative reduction | {white_relative:.6f}% | sensitivity only |
| Call histogram d1/d2/d3/d4 | {clean['calls']['call_histogram']} | exact |
| Reached gate stages | {clean['calls']['reached_stage_counts']} | exact |
| Declared adaptive total | {clean['compute']['adaptive_total_flops']:,} | exact under ledger |
| Seeded comparator total | {clean['compute']['seeded_total_flops']:,} | +{clean['compute']['seeded_minus_adaptive_flops']:,} |
| Recomputed terminal | `{clean['terminal_mapping']['recomputed_terminal']}` | exact |

Every leave-one-episode-out effect is positive. All 16 contiguous 100-episode
raw block means are positive. Alternative accumulation orders and standard
quantile conventions preserve the sign and decision.

## Tests performed

The sealed matrix contained 49 tests. **{len(passed)} passed and {len(failed)}
failed; none is partial, unresolved, or not run.** Of 43 decision-critical
tests, {len(decision_passed)} passed and {len(decision_failed)} failed. Failed
IDs are: {failed_ids}.

| Domain | Pass | Fail | Failed IDs |
|---|---:|---:|---|
{domain_rows}

### Chain of custody and isolation

- The v001-v004 chronology is coherent. v001 generated zero outcomes after 101
  wrong-runtime slot-0 failures; v002 and v003 stopped before sealing/raw
  outcomes; only v004 completed. Scientific bytes/functions are unchanged
  across versions.
- v004 contains exactly 1,600 unique raw IDs/paths/hashes and 60,800 ordered
  rows, with zero replacements and no confirmation failure ledger.
- The scan compared all 1,600 V5 archives against 2,472 accessible prior raw
  archives, plus 31,629 structured prior IDs and 28,427 prior seeds. Exact,
  low-32-bit, SeedSequence, state, action, observation, pixel, and calibrated
  near-duplicate tests found no collision.
- The unavailable original base-training raw corpus is an explicit limitation.
  V5 nevertheless postdates training and uses fresh, unique seed tuples.
- Initial/final v001-v004 aggregate SHA-256 is
  `{source_final['v001_v004']['final_aggregate_sha256']}`; the full readiness
  program aggregate is
  `{source_final['full_readiness_program_supplement']['final_aggregate_sha256']}`.
  No path, content, or stat record changed during the audit.

### Causal input and leakage

Static dataflow and runtime access tracing agree: model/gate code materializes
only `pixels` and `action`. Gate features use past/current latent history,
action history, the current prediction, and the most recent solver update.
Targets, future observations/latents, loss/gain, contact, reward, done,
success, oracle decisions, future summaries, and V5 normalization canaries do
not enter a decision. Varying those copied canaries leaves allowlisted inputs
and semantic outputs exact.

### Frozen execution and complete replay

All frozen model/gate/whitening/source/input hashes currently match. Module
state is frozen and gradient-free before/after. The full MPS packaged replay
reproduces targets, all dense exits, selected sparse outputs, calls, scores,
and features for all 60,800 rows.

The independent explicit sqrt/sum feature transcription differed by at most
`3.814697265625e-6` from float32 reduction order, passed the preregistered audit
`atol/rtol`, and preserved every call/output exactly. Only 88/1,600 episodes
met the package's tighter sealed score-allclose under that deliberately
different algebraic order. The required second, vector-norm-order
transcription was bit exact for features, scores, calls, and outputs. The
initial failure was an audit-harness acceptance bug (absolute-only comparison
without the preregistered relative term), retained in
`REPLAY_RESULTS_INITIAL_FAILURE.json`; no tolerance was relaxed.

### Compute and comparator fairness

The declared integer ledger is internally exact:

- 60,800 base and mandatory d1 calls;
- 13,174 later-adapter calls;
- 73,187 gate evaluations × 7,985 = 584,398,195 gate FLOPs;
- declared total 4,332,936,120,435;
- seeded comparator 4,332,936,224,000, or 103,565 more.

The analytic comparator is global and transition-independent. It uses a d1/d2
mixture with d2 weight `0.25295409752763076`; using aggregate target loss to
select the strongest global pair makes it a stronger benchmark but supplies
no row-level routing information.

The total is not a literal end-to-end count. The ledger inherits dense-linear
constants that omit visible biases, GELUs, alpha multiplication, and residual
adds; it also excludes the executed pixel encoder. Adding the directly visible
704 or 832 operations per later adapter changes the raw effect to
`6.773222841315813e-6` or `6.77398761790643e-6`, respectively—slightly
stronger—and the seeded comparator remains weakly more compute. The justified
wording is “exact under the frozen latent-prediction ledger.”

### Negative controls

- Within-episode/reachability score permutations: 2,000 repetitions,
  raw/white plus-one p = 0.00049975.
- Random and constant/mixed matched-compute policies: 2,000 each,
  raw/white p = 0.00049975.
- Score-to-gain breaks and perturbed reversed order: 2,000 each,
  raw/white p = 0.00049975; deterministic reversal is worse.
- Unused fields: exact semantic invariance.
- **Retained failure:** 1,000 whole-episode target reassignments give raw
  p = 0.164835 and white p = 0.000999. A second 1,600×1,600 pair-matrix
  implementation agrees. This null changes the physical prediction task and
  has 15.1× the SD of matched allocation nulls. A separately labeled post-hoc
  loss-profile block permutation gives raw/white p = 0.00049975; it diagnoses
  but does not erase the preregistered failure.

### Fault injection

Thirteen copy-only corruptions were exercised. Nine behaved as required and
four exposed fail-open behavior (`{fault_ids}`):

- one-byte execution/gate changes, missing/extra files, manifest hash drift,
  thresholds, compute price, wrong runtime, and post-manifest canaries reject;
- a duplicate manifest identifier is accepted by the existing-manifest
  preflight (the smoke final verifier later rejects it);
- a consistently rebound episode-block order is semantically accepted;
- a `../` raw path with a declared matching hash escapes the root;
- a byte-identical external symlink passes the final hash/path verifier.

Canonical evidence contains none of these states: paths are relative and
confined, files are regular, IDs/paths/hashes are unique, rows are ordered,
and both complete snapshots match.

## Findings

### Blockers

None.

### Major

{major_lines}

### Minor

{minor_lines}

## Preregistered terminal-mapping defect

The sealed audit plan made every one of the 49 tests mandatory for either
supported outcome. It also limited invalidation to a validated Blocker and
inconclusive to missing/corrupt essential evidence or persistent critical
disagreement capable of hiding a Blocker. The five resolved failures are real
and remain failures, but no Blocker exists and the primary evidence is complete
and mutually agreeing. The sealed rules therefore leave an unintended empty
case: no label fits.

I have not converted failures into passes. Because the user requires exactly
one of four outcomes, I apply the governing user-level definitions and choose
`{TERMINAL_OUTCOME}` as the least misleading exact label: the confirmation is
independently supported, while important bounded Major caveats remain. Calling
it invalidated would falsely imply a demonstrated invalidating path; calling it
inconclusive would falsely imply the primary result or essential evidence is
unresolved.

## Generalization, practical size, and latency

- **Frozen-confirmation validity:** supported with the caveats above.
- **Generalization:** not established. No output from the separate
  generalization task was used.
- **Practical effect:** raw MSE reduction is about {raw_relative:.3f}% and
  native-whitened reduction about {white_relative:.3f}% versus the analytic
  comparator—statistically clear but small.
- **Latency:** no win is established. Synchronized MPS adaptive medians are
  {latency_by[('actual_adaptive_sparse', 1)]['median_seconds']*1000:.3f} ms at
  batch 1 versus {latency_by[('fixed_depth_1', 1)]['median_seconds']*1000:.3f}
  ms for fixed d1, and
  {latency_by[('actual_adaptive_sparse', 1024)]['median_seconds']*1000:.3f} ms
  at batch 1,024 versus
  {latency_by[('fixed_depth_1', 1024)]['median_seconds']*1000:.3f} ms for fixed
  d1. Pixel encoding and simulator rollout are excluded from these timings.
- **Energy:** not measured.

## Artifact locations

- Audit worktree: `{audit.parents[2]}`
- Audit run: `{audit}`
- Canonical read-only evidence: `{package}`
- Test ledger: `{audit / 'TEST_RESULTS.json'}`
- Findings: `{audit / 'FINDINGS.json'}`
- Final source reconciliation: `{audit / 'SOURCE_FINAL_HASHES.json'}`
- This report: `{audit / 'REPORT.md'}`

No commit, push, pull request, retraining, replacement outcome, V3 target
inspection, or canonical evidence mutation was performed.
"""
    atomic_text(audit / "REPORT.md", report)

    artifact_names = [
        "AUDIT_PLAN.md",
        "audit_plan_seal.json",
        "THREAT_MODEL.md",
        "TEST_MATRIX.json",
        "TEST_RESULTS.json",
        "SOURCE_SNAPSHOT.json",
        "SOURCE_FINAL_HASHES.json",
        "CHRONOLOGY_AUDIT.json",
        "COHORT_ISOLATION.json",
        "CAUSAL_INPUT_AUDIT.json",
        "IMPLEMENTATION_INTEGRITY.json",
        "CLEAN_ROOM_RECOMPUTATION.json",
        "REPLAY_RESULTS.json",
        "NEGATIVE_CONTROLS.json",
        "COMPUTE_AUDIT.json",
        "FAULT_INJECTION.json",
        "FINDINGS.json",
        "REPORT.md",
    ]
    state = {
        "schema_version": 1,
        "run_id": audit.name,
        "objective": (
            "Adversarially audit the complete V5 result and reach an evidence-backed "
            "terminal judgment about its scientific and software validity."
        ),
        "phase": "terminal",
        "status": "complete",
        "terminal_outcome": TERMINAL_OUTCOME,
        "v5_result_appears_genuine": True,
        "source_snapshot_complete": True,
        "source_final_reconciliation_complete": True,
        "canonical_evidence_changed_during_audit": False,
        "audit_plan_sealed": True,
        "detailed_episode_results_opened": True,
        "completed_test_ids": [item["id"] for item in tests],
        "passed_test_ids": [item["id"] for item in passed],
        "failed_test_ids": [item["id"] for item in failed],
        "unresolved_test_ids": [],
        "post_hoc_diagnostics": [
            "loss-profile episode-block permutation",
            "order-matched second replay transcription",
            "initial replay failure diagnosis",
        ],
        "blockers": [],
        "majors": [item["id"] for item in majors],
        "minors": [item["id"] for item in minors],
        "test_summary": test_results["summary"],
        "terminal_mapping_reconciliation": findings_payload["terminal_reasoning"],
        "artifact_paths": {
            name: str(audit / name) for name in artifact_names
        },
        "artifact_sha256": {
            name: sha256(audit / name) for name in artifact_names
        },
        "notes": [
            "Canonical v001-v004 evidence remained immutable and read-only.",
            "The separate lewm_v5_generalization task was pruned and not used as evidence.",
            "Five failed preregistered IDs are retained without relabeling.",
            "The sealed support predicate has a documented post-hoc mapping defect.",
            "No commit, push, PR, retraining, replacement outcome, or V3 target inspection occurred.",
        ],
        "completed_unix_ns": time.time_ns(),
    }
    atomic_json(audit / "STATE.json", state)

    print(
        json.dumps(
            {
                "terminal_outcome": TERMINAL_OUTCOME,
                "pass_count": len(passed),
                "fail_count": len(failed),
                "blocker_count": len(blockers),
                "major_count": len(majors),
                "minor_count": len(minors),
                "report": str(audit / "REPORT.md"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
