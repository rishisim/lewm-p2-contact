#!/usr/bin/env python3
"""Write the narrative report and artifact guide from frozen machine results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import common


def _f(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}g}"


def _role_table(summaries: dict[str, Any]) -> str:
    labels = {
        "offline_discovery": "Offline discovery",
        "offline_calibration": "Offline calibration",
        "plan_oracle": "Fresh PlanOracle",
        "markov_oracle": "Paired MarkovOracle",
        "v4_markov": "Consumed V4 MarkovOracle",
    }
    rows = [
        "| Dataset | Episodes | |a|≥.99 | Action RMS | a[4] mean | |z|>3 | d0 MSE | d1→d2 gain | Contact | Calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role, label in labels.items():
        value = summaries[role]
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(value["episodes"]),
                    _f(value["raw_action_abs_ge_0_99_rate"]),
                    _f(value["raw_action_rms"]),
                    _f(value["raw_action_coord4_mean"]),
                    _f(value["normalized_action_abs_gt3_rate"]),
                    _f(value["d0_raw_mse"]),
                    _f(value["d1_to_d2_gain"]),
                    _f(value["contact_row_rate"]),
                    _f(value["mean_calls"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def run() -> dict[str, Any]:
    decision_path = common.STUDY_ROOT / "decision.json"
    comparison_path = common.STUDY_ROOT / "metrics/distribution_comparison.json"
    gate_path = common.STUDY_ROOT / "metrics/gate_prospective.json"
    latency_path = common.STUDY_ROOT / "metrics/latency.json"
    decision = common.study_json(decision_path)
    comparison = common.study_json(comparison_path)
    gate = common.study_json(gate_path)
    latency = common.study_json(latency_path) if latency_path.exists() else None
    verdict = decision["decision"]
    v5 = decision["future_entirely_fresh_v5_confirmation_authorized"]
    bounded_path = common.STUDY_ROOT / "audit/bounded_reconstruction_check.json"
    bounded = common.study_json(bounded_path) if bounded_path.exists() else None
    if verdict == "generator_reconstruction_failed" and bounded is None:
        raise RuntimeError(
            "generator reconstruction failure requires its preregistered smallest bounded check"
        )
    if verdict == "distribution_contract_passed":
        interpretation = (
            "The intended Cube play-policy distribution was reconstructed: fresh PlanOracle "
            "entered the frozen offline core region, was materially closer than the paired "
            "MarkovOracle control, restored positive d1→d2 gain, and the unchanged gate passed "
            "all prospective discovery criteria. This supports the causal diagnosis that V4's "
            "MarkovOracle generator was out of distribution for the released Cube play data."
        )
    elif verdict == "generator_reconstruction_failed":
        interpretation = (
            "Fresh PlanOracle did not satisfy the frozen offline distribution contract. The "
            "study therefore does not isolate V4's policy choice as the complete cause; only "
            "the smallest bounded version/action-semantics or replay check is justified."
        )
    elif verdict == "distribution_matched_gate_failed":
        interpretation = (
            "Fresh PlanOracle matched the frozen offline/base distribution, but the unchanged "
            "gate failed its prospective discovery criteria. V4's MarkovOracle mismatch remains "
            "diagnostic, but distribution reconstruction alone does not rescue adaptive routing."
        )
    else:
        interpretation = "A validity audit failed, so no scientific distribution or gate conclusion is licensed."
    exact = gate["comparisons"]["raw_vs_exact_total_flop_analytic"]
    seeded = gate["comparisons"]["raw_vs_exact_total_flop_seeded"]
    histogram = gate["comparisons"]["raw_vs_histogram"]
    fixed = gate["comparisons"]["raw_vs_fixed_d1"]
    latency_sentence = (
        f"The separate synchronized MPS latency audit passed ({len(latency['timings'])} timed path/batch rows); "
        "latency did not enter the statistical or FLOP verdict."
        if latency is not None
        else "The separate latency audit had not yet been materialized when this report was written."
    )
    bounded_sentence = ""
    if bounded is not None:
        bounded_sentence = (
            "\n\nThe authorized post-decision diagnostic replayed exactly one existing "
            f"PlanOracle trajectory with zero new seeds or policy trajectories. Exact required "
            f"transition replay was {'successful' if bounded['transition_replay']['required_arrays_exact'] else 'unsuccessful'}; "
            "the upstream 1,001-step versus released-local 200-action horizon mismatch remains. "
            "No model or gate was tuned."
        )
    report = f"""# LeWM Cube policy distribution-contract study

## Outcome

Mechanical decision: **`{verdict}`**.

{interpretation}

A future, entirely fresh V5 confirmation is **{'authorized' if v5 else 'not authorized'}**.
No V5 episode was launched here. This result is discovery evidence, not confirmation.

## Generator provenance and design

The controlling upstream source is OGBench v1.2.1, tag commit
`1d4140997f60c52c6fb0702ec100dc988b18c548`. Its official
`cube-single-play-v0` recipe uses `dataset_type=play`, and
`generate_manipspace.py` maps play to the non-Markovian `CubePlanOracle`;
the noisy recipe maps to `CubeMarkovOracle`. Installed source bytes, package
versions, the 95 GB offline HDF5 SHA-256, constructors, preprocessing, frozen
model objects, seeds, bootstrap estimands, metrics, and decision rules were
sealed before any new rollout.

The bounded design was executed exactly: 12 physically separate smoke episodes
per policy (permanently excluded), 90 PlanOracle discovery episodes, and 30
MarkovOracle controls paired on identical environment seeds and exact initial
states. Wrapper and NumPy-global oracle RNG streams were separate. There was no
sequential expansion and no confirmatory episode.

## Distribution readout

Fresh PlanOracle's joint max-T statistic was
`{_f(comparison['plan_joint_max_abs_z'])}` against the frozen 95% threshold
`{_f(comparison['joint_95_threshold'])}`. Its paired standardized distance to
the offline center was `{_f(comparison['plan_standardized_distance'])}`, versus
`{_f(comparison['markov_standardized_distance'])}` for MarkovOracle (ratio
`{_f(comparison['plan_to_markov_distance_ratio'])}`; Plan was closer on
{comparison['plan_closer_coordinate_count']}/7 core coordinates).

{_role_table(comparison['role_summaries'])}

Raw pixel/observation summaries, latent histories/targets, raw and normalized
actions, d0 and fixed d1–d4 raw/whitened losses, updates, gains, every gate
feature block and normalized-z summary, scores, rankings, price exceedance,
calls, and post-hoc physical composition are retained in machine-readable
artifacts. Contact, impact, state, motion, phase, task, and success did not enter
training, inference, policy selection, or the primary contract thresholds.

## Frozen gate on 90 PlanOracle episodes

The unchanged gate used mean calls `{_f(gate['mean_calls'])}` and raw MSE
`{_f(gate['adaptive_raw_mse'])}`. Episode-bootstrap benefits (baseline minus
adaptive) were:

- exact-total-FLOP analytic: `{_f(exact['mean_benefit'])}`,
  95% CI [`{_f(exact['ci_low'])}`, `{_f(exact['ci_high'])}`];
- conservative exact-total-FLOP seeded: `{_f(seeded['mean_benefit'])}`,
  95% CI [`{_f(seeded['ci_low'])}`, `{_f(seeded['ci_high'])}`];
- fixed d1: `{_f(fixed['mean_benefit'])}`,
  95% CI [`{_f(fixed['ci_low'])}`, `{_f(fixed['ci_high'])}`];
- exact call-histogram randomization: `{_f(histogram['mean_benefit'])}`,
  95% CI [`{_f(histogram['ci_low'])}`, `{_f(histogram['ci_high'])}`].

The exact-total comparator counted the 8,196-FLOP gate overhead at every
evaluated decision; fixed base/V1 and later-adapter costs use the unchanged
historical convention. Analytic exact matching, conservative integer matching,
exact histogram preservation, whitening, and call/frontier audits are recorded.
{latency_sentence}

## Validity and claim boundary

Frozen-object byte hashes, no-gradient state, causal feature signatures, d0/d1
identity, dense/reference/optimized sparse equivalence, row counts, paired
initial states, raw-hash freshness, seed nonmembership, exact calls/FLOPs,
bootstrap intervals, and decision mapping were audited. Every new cache proves
empty V3-test intersection; the combined V3 cache was never opened with NumPy,
and V3 test targets remain untouched.

V4 is retained as a valid negative result for its actual MarkovOracle
distribution and was used only as a labeled diagnostic comparator. It defined
no band, threshold, model, normalization, stopping rule, or selection. This
study makes no confirmatory or contact-aware claim and no “first adaptive world
model” claim. LoopWM remains relevant adaptive-computation related work in text
environments.

## Strongest caveats and next step

- The released LeWM checkpoint lacks a complete original pretraining episode
  manifest; exact nonduplication was checked against every accessible recorded
  episode outside the opaque V3 test set, but full pretraining nonmembership
  cannot be mechanically proven.
- The public upstream command uses a 1,001-step collection horizon, while the
  released local LeWM artifact has 201 states/200 actions; this reconstruction
  therefore pins the local 200-action contract and tests it empirically.
- This is one environment, one frozen solver/gate, and a 90-episode discovery
  judge. Actual latency is hardware-specific and separate from FLOPs.
{bounded_sentence}

Next step: {decision['next_step']}
"""
    readme = """# Cube distribution-contract artifact

This directory contains a bounded PlanOracle-versus-MarkovOracle distribution
contract study. `PREREGISTRATION.md` and `reference/reference_bands.json` were
frozen before new rollouts; `decision.json` is the mechanical verdict;
`REPORT.md` is the scientific interpretation. `metrics/` contains the complete
machine-readable comparisons, bootstrap replicates, FLOPs, calls, feature-block
summaries, and latency. `audit/` contains isolation, seed, pairing, freshness,
test, hash, and independent recomputation evidence.

Smoke data are physically separate and permanently excluded. V3 test targets
were never opened. V4 is diagnostic only. No V5 confirmation was launched.
"""
    report_path = common.STUDY_ROOT / "REPORT.md"
    readme_path = common.STUDY_ROOT / "README.md"
    if report_path.exists() or readme_path.exists():
        raise RuntimeError("report artifacts already exist")
    report_path.write_text(report, encoding="utf-8")
    readme_path.write_text(readme, encoding="utf-8")
    result = {
        "report_sha256": common.sha256_file(report_path),
        "readme_sha256": common.sha256_file(readme_path),
        "decision": verdict,
        "v5_authorized": v5,
    }
    common.write_study_json(common.STUDY_ROOT / "audit/report_manifest.json", result, exclusive=True)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, sort_keys=True))
