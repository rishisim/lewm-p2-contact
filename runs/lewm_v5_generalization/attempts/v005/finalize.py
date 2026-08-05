#!/usr/bin/env python3
"""Create the honest robustness map, report, and project-scoped follow-on task."""

from __future__ import annotations

import json
import time

from study_common import (
    ATTEMPT_ROOT,
    REGIMES,
    append_ledger,
    atomic_json,
    finalize_follow_on,
    read_json,
    relative_to_repo,
    sha256_file,
)


def fmt(value: float) -> str:
    return f"{value:.8g}"


def main() -> None:
    map_path = ATTEMPT_ROOT / "ROBUSTNESS_MAP.json"
    report_path = ATTEMPT_ROOT / "REPORT.md"
    follow_path = ATTEMPT_ROOT / "FOLLOW_ON_TASK.md"
    if map_path.exists() and report_path.exists() and follow_path.exists():
        finalize_follow_on(follow_path)
        print(json.dumps(read_json(map_path), sort_keys=True))
        return
    if any(path.exists() for path in (map_path, report_path, follow_path)):
        raise RuntimeError("partial final reporting artifact set")
    decision_path = ATTEMPT_ROOT / "decision.json"
    audit_path = ATTEMPT_ROOT / "audit/independent_verification.json"
    analysis_path = ATTEMPT_ROOT / "analysis_result.json"
    posthoc_path = ATTEMPT_ROOT / "metrics/posthoc_interpretation.json"
    decision = read_json(decision_path)
    audit = read_json(audit_path)
    analysis = read_json(analysis_path)
    posthoc = read_json(posthoc_path)
    if (
        not decision.get("confirmation_terminal")
        or not audit.get("passed")
        or not posthoc.get("decision_sha256_unchanged_after_posthoc")
    ):
        raise RuntimeError("final reporting requires verified terminal evidence")
    regimes = {}
    for regime in REGIMES:
        simultaneous = decision["simultaneous_co_primary"][regime]
        regimes[regime] = {
            "label": decision["regime_labels"][regime],
            "co_primary": simultaneous,
            "supported_claim_count": sum(
                item["lower"] > 0 for item in simultaneous.values()
            ),
            "call_depth_shift_vs_v5": analysis["regimes"][regime][
                "distribution_shift_vs_v5"
            ]["call_depth"],
            "stagewise_rank": analysis["regimes"][regime][
                "stagewise_rank"
            ],
            "compute": analysis["regimes"][regime]["compute"],
            "posthoc_interpretation": posthoc["regimes"][regime]["summary"],
        }
    robustness_map = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "terminal_outcome": decision["terminal_outcome"],
        "process_valid": decision["process_valid"],
        "independent_verification_passed": decision[
            "independent_verification_passed"
        ],
        "supported_co_primary_claim_count": decision[
            "supported_co_primary_claim_count"
        ],
        "regimes": regimes,
        "heterogeneity": analysis["heterogeneity"],
        "claim_boundary": (
            "This map concerns zero-shot latent-prediction adaptive-compute "
            "generalization only. It does not establish downstream control "
            "improvement, contact-aware routing, wall-clock acceleration, "
            "universal validity, or a priority claim."
        ),
        "v5_not_rerun": True,
        "generalization_not_retried": True,
        "decision_sha256": sha256_file(decision_path),
        "independent_audit_sha256": sha256_file(audit_path),
        "analysis_sha256": sha256_file(analysis_path),
        "posthoc_sha256": sha256_file(posthoc_path),
    }
    atomic_json(map_path, robustness_map, exclusive=True)

    rows = []
    for regime in REGIMES:
        raw = decision["simultaneous_co_primary"][regime][
            "raw_vs_analytic"
        ]
        white = decision["simultaneous_co_primary"][regime][
            "fixed_whitened_vs_analytic"
        ]
        rows.append(
            "| "
            + " | ".join(
                (
                    regime,
                    decision["regime_labels"][regime],
                    f"{fmt(raw['estimate'])} [{fmt(raw['lower'])}, +∞)",
                    f"{fmt(white['estimate'])} [{fmt(white['lower'])}, +∞)",
                    str(
                        analysis["regimes"][regime]["compute"][
                            "call_histogram"
                        ]
                    ),
                )
            )
            + " |"
        )
    report = f"""# LeWM V5 zero-shot generalization report

Terminal result: `{decision['terminal_outcome']}`.

The fixed V5 policy was evaluated once on exactly 3,000 fresh episodes in each
of three preregistered one-factor DGP shifts. All 9,000 cohorts and 342,000
modeled rows were hash-sealed before analysis. The independent verifier
reproduced the terminal evidence and passed.

| Regime | Regime map | Raw effect and simultaneous lower bound | Fixed-whitened effect and simultaneous lower bound | Depth 1–4 histogram |
|---|---|---:|---:|---|
{chr(10).join(rows)}

The simultaneous family is the fixed set of six regime-by-endpoint claims,
using 20,000 episode bootstrap replicates and Bonferroni one-sided alpha
0.05/6. Positive values favor the frozen adaptive policy over the strongest
transition-independent analytic mixture at exactly matched total counted
compute, including gate feature and dual-head overhead.

Overall, {decision['supported_co_primary_claim_count']} of six co-primary
claims have strictly positive simultaneous lower bounds. The preregistered
mapping therefore yields `{decision['terminal_outcome']}`. Auxiliary seeded,
fixed-depth-1, and within-episode-histogram controls, stagewise rank signs,
heterogeneity, latency, energy availability, and post-hoc interpretation did
not affect that mapping.

Latency is reported separately from FLOPs at
`metrics/latency_and_resources.json`. Energy was not measured because this
runtime has no reliable resettable per-path energy interface. Contact, motion,
phase, reward, and success were opened only after the immutable decision and
are reported only in `metrics/posthoc_interpretation.json`.

The confirmed V5 claim remains restricted to the fixed Cube PlanOracle DGP:
lower episode-averaged raw and PlanOracle-native-whitened latent MSE than the
strongest transition-independent analytic allocation at exactly matched
counted compute. This generalization study does not establish universal
generalization, downstream control improvement, contact-aware routing,
wall-clock acceleration, or “first adaptive world model.” LoopWM remains
relevant related work.
"""
    report_path.write_text(report, encoding="utf-8")

    terminal = decision["terminal_outcome"]
    if terminal == "zero_shot_generalization_supported":
        follow = """# Next project-scoped Codex task

Create a paper-ready evidence-synthesis project that consumes, but never
reruns, the V5 confirmation and v005 generalization cohorts. Produce a
claim-bounded methods/results package, robustness figures, related-work
positioning including LoopWM, and an independent citation/evidence audit.

In a separately scoped deployment-efficiency study, preregister latency,
memory, batching, and energy measurement before any deployment outcomes. Keep
deployment evidence separate from latent-MSE confirmation and do not imply
wall-clock acceleration from FLOPs.
"""
    elif terminal in (
        "zero_shot_generalization_partial",
        "zero_shot_generalization_failed",
    ):
        follow = """# Next project-scoped Codex task

Launch domain-robust gate discovery using only the consumed v005
generalization cohorts for diagnosis of frozen feature, score, gain,
call-price, and policy-distribution shifts. Preserve the v005 result as
terminal evidence.

Preregister isolated multi-DGP roles: training DGPs for gate fitting, disjoint
selection DGPs for candidate choice, excluded mechanical smoke, and entirely
fresh future confirmation cohorts. Never train, select, calibrate, whiten, or
confirm on the same episode. Keep the base model, solver causality, gradient
boundaries, contact exclusion, exact-total-compute comparator, simultaneous
inference, and independent verification explicit. Do not reuse V5 or v005
confirmation episodes for a future confirmatory claim.
"""
    else:
        follow = """# Next project-scoped Codex task

Preserve the invalid v005 attempt and diagnose only the recorded integrity
failure. A future attempt is allowed only if it can be versioned forward
without changing the sealed scientific design and without reusing any smoke
or target identifier. Do not interpret scientific outcomes from an invalid
execution.
"""
    follow_path.write_text(follow, encoding="utf-8")
    append_ledger(
        "robustness_map_and_report_complete",
        map_path=relative_to_repo(map_path),
        map_sha256=sha256_file(map_path),
        report_path=relative_to_repo(report_path),
        report_sha256=sha256_file(report_path),
        follow_on_path=relative_to_repo(follow_path),
        follow_on_sha256=sha256_file(follow_path),
        terminal_outcome=terminal,
    )
    finalize_follow_on(follow_path)
    print(json.dumps(robustness_map, sort_keys=True))


if __name__ == "__main__":
    main()
