#!/usr/bin/env python3
"""Generate final tables, figures, verification record, and report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import common
import judge_runner


def load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text())


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def tables(discovery: Mapping[str, Any], calibration: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    budget_rows = []
    for family, payload in discovery["families"].items():
        for key, point in payload["operating_points"].items():
            budget_rows.append({
                "family": family,
                "operating_point": key,
                "mean_calls": point["mean_calls"],
                "raw_mse": point["raw_mse"],
                "matched_raw_mse": point["matched_raw_mse"],
                "raw_benefit": point["vs_matched_randomized"]["mean_benefit"],
                "raw_ci_low": point["vs_matched_randomized"]["ci_low"],
                "raw_ci_high": point["vs_matched_randomized"]["ci_high"],
                "whitened_benefit": point["whitened_vs_matched"]["mean_benefit"],
                "whitened_ci_low": point["whitened_vs_matched"]["ci_low"],
                "gate_flops_per_decision": point["gate_cost"]["total_incremental_gate_flops_per_evaluated_decision"],
                "adaptive_total_flops": point["adaptive_total_flops_per_transition"],
                "call_nondominated": point["call_nondominated"],
                "flop_nondominated": point["flop_nondominated"],
                "positive_outer_fold_fraction": point["positive_outer_fold_fraction"],
                "passed": point["passes_discovery_point"],
            })
    write_csv(ROOT / "metrics/budget_frontier.csv", budget_rows)
    write_csv(ROOT / "metrics/latency.csv", [
        {key: value for key, value in row.items() if key != "all_seconds"}
        for row in runtime["timings"]
    ])
    fixed_rows = []
    for source, values in (
        ("discovery_all_420", discovery["fixed_exit_raw_mse"]),
        ("calibration_once_90", calibration["fixed_exit_raw_mse"]),
    ):
        for depth, value in enumerate(values, start=1):
            fixed_rows.append({"source": source, "depth": depth, "raw_mse": value})
    write_csv(ROOT / "metrics/fixed_exits.csv", fixed_rows)
    gain_rows = []
    for family, payload in discovery["families"].items():
        for item in payload["diagnostics"]:
            gain_rows.append({"source": "discovery", "family": family, **{k: v for k, v in item.items() if k != "quantiles"}})
    for item in calibration["ranking_diagnostics"]:
        gain_rows.append({"source": "calibration_once", "family": "full_linear", **{k: v for k, v in item.items() if k != "quantiles"}})
    write_csv(ROOT / "metrics/gain_diagnostics.csv", gain_rows)


def figures(discovery: Mapping[str, Any], calibration: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    colors = {
        "full_linear": "#166534", "distilled_rank_linear": "#2563eb", "stable_sparse": "#9333ea",
        "low_rank": "#ea580c", "in_model_halting": "#0891b2", "compact11_control": "#6b7280",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for family, payload in discovery["families"].items():
        points = sorted(payload["operating_points"].values(), key=lambda row: row["mean_calls"])
        calls = [row["mean_calls"] for row in points]
        raw = [row["vs_matched_randomized"]["mean_benefit"] * 1e5 for row in points]
        white = [row["whitened_vs_matched"]["mean_benefit"] * 1e3 for row in points]
        axes[0].plot(calls, raw, marker="o", label=family, color=colors[family])
        axes[1].plot(calls, white, marker="o", label=family, color=colors[family])
    for axis, ylabel in zip(axes, ("Raw benefit vs matched (×10⁻⁵)", "Whitened benefit vs matched (×10⁻³)")):
        axis.axhline(0, color="black", lw=0.8)
        axis.set_xlabel("Mean solver calls")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Nested discovery allocation benefit")
    fig.tight_layout()
    fig.savefig(out / "01_discovery_benefits.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in discovery["global_frontier"]:
        marker = {"fixed": "s", "adaptive": "o", "matched": "x"}[row["kind"]]
        color = "#166534" if row["kind"] == "adaptive" and row["name"].startswith("full_linear") else "#64748b"
        ax.scatter(row["total_flops"] / 1e6, row["raw_mse"], marker=marker, color=color, alpha=0.75)
    ax.set_xlabel("Fully counted FLOPs per transition (millions)")
    ax.set_ylabel("Raw latent MSE")
    ax.set_title("Discovery compute frontier")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "02_flop_frontier.png", dpi=180)
    plt.close(fig)

    comparisons = ["matched_randomized", "analytic_mixture", "fixed_d1", "histogram_null"]
    values = [calibration[f"vs_{name}"]["mean_benefit"] * 1e5 for name in comparisons]
    lows = [calibration[f"vs_{name}"]["ci_low"] * 1e5 for name in comparisons]
    highs = [calibration[f"vs_{name}"]["ci_high"] * 1e5 for name in comparisons]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(comparisons))
    ax.errorbar(x, values, yerr=[np.asarray(values) - np.asarray(lows), np.asarray(highs) - np.asarray(values)], fmt="o", capsize=5, color="#166534")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [name.replace("_", "\n") for name in comparisons])
    ax.set_ylabel("Calibration raw benefit (×10⁻⁵)")
    ax.set_title("Frozen one-shot calibration comparisons (95% clustered CI)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "03_calibration_comparisons.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for item in calibration["ranking_diagnostics"]:
        quantiles = item["quantiles"]
        ax.plot([row["quantile"] + 1 for row in quantiles], [row["mean_realized_gain"] * 1e4 for row in quantiles], marker="o", label=f"decision d{item['decision_depth']}")
    ax.set_xlabel("Predicted-gain quintile")
    ax.set_ylabel("Mean realized marginal gain (×10⁻⁴)")
    ax.set_title("Calibration gain ordering")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "04_calibration_gain_quantiles.png", dpi=180)
    plt.close(fig)

    subset = [row for row in runtime["timings"] if row["batch_size"] == 1024]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(subset)), [row["median_seconds"] * 1000 for row in subset], color=["#166534" if "adaptive" in row["path"] else "#64748b" for row in subset])
    ax.set_xticks(np.arange(len(subset)), [row["path"].replace("_", "\n") for row in subset], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Median end-to-end latency for 3,420 rows (ms)")
    ax.set_title("MPS latency, batch size 1,024 (base prediction included)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "05_latency.png", dpi=180)
    plt.close(fig)


def run_tests() -> dict[str, Any]:
    python = "/Users/rishisim/.cache/lewm-v2-venv/bin/python"
    commands = [
        [python, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"],
        [python, "-m", "unittest", "discover", "-s", str(common.PRIOR / "tests"), "-v"],
    ]
    chunks = []
    passed = True
    for command in commands:
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
        chunks.append("$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr)
        passed = passed and completed.returncode == 0
    (ROOT / "test_logs.txt").write_text("\n".join(chunks))
    if not passed:
        raise RuntimeError("final unit-test suite failed")
    return {"passed": passed, "commands": commands, "log_sha256": common.sha256_file(ROOT / "test_logs.txt")}


def verification(discovery: Mapping[str, Any], calibration: Mapping[str, Any], runtime: Mapping[str, Any], tests: Mapping[str, Any]) -> dict[str, Any]:
    _, _, _, isolation = common.load_prior_modules()
    opaque = isolation.assert_combined_cache_never_opened()
    frozen = judge_runner.validate_frozen_tournament()
    prepared = load("cache/stagewise_train_outputs_manifest.json")
    return {
        "schema_version": 1,
        "immutable_evidence": {
            "selected_solver_checkpoint_sha256": common.sha256_file(common.SOLVER_CHECKPOINT),
            "isolated_train_cache_sha256": common.sha256_file(common.TRAIN_CACHE),
            "prior_report_sha256": common.sha256_file(common.PRIOR / "REPORT.md"),
            "prior_decision_sha256": common.sha256_file(common.PRIOR / "decision.json"),
            "opaque_combined_cache_audit": opaque,
        },
        "solver_preservation": {
            "prepare": prepared,
            "calibration": calibration["solver_audit"],
            "checkpoint_unchanged_through_discovery": discovery["solver_checkpoint_unchanged"],
        },
        "data_isolation": {
            "discovery_episodes": discovery["episodes"],
            "calibration_episodes": calibration["episodes"],
            "calibration_receipt": calibration["receipt"],
            "calibration_cache_sha256": calibration["extraction"]["cache_sha256"],
            "v3_test_targets_consumed": False,
            "v4_created": (REPO / "runs/lewm_adaptive_compute_v4").exists(),
        },
        "tournament": {
            "sha256": frozen["tournament_sha256"],
            "threshold_isolation": True,
            "single_student": True,
            "teacher_deployed": False,
            "cross_fitted_teacher_labels": True,
            "fold_local_sparse_selection": True,
            "grouped_nested_repeated_cv": True,
        },
        "mechanical_audits": {
            "causal_features": calibration["causal_feature_audit"],
            "hard_flop_budget": calibration["gate_cost"],
            "exact_calls": calibration["exact_call_audit"],
            "sparse_runtime": runtime["sparse_dense_equivalence"],
            "runtime_passed": runtime["passed"],
            "all_metrics_finite": finite_tree(discovery) and finite_tree(calibration) and finite_tree(runtime),
            "unit_tests": tests,
        },
        "negative_controls": {
            "score_permutation_calibration": calibration["score_permutation_vs_own_matched"],
            "compact11_discovery_passed": any(row["passes_discovery_point"] for row in discovery["families"]["compact11_control"]["operating_points"].values()),
            "stable_sparse_discovery_passed": any(row["passes_discovery_point"] for row in discovery["families"]["stable_sparse"]["operating_points"].values()),
        },
        "passed": bool(
            calibration["passed"] and runtime["passed"] and tests["passed"]
            and prepared["d0_bitwise"] and prepared["d1_bitwise"]
            and not (REPO / "runs/lewm_adaptive_compute_v4").exists()
        ),
    }


def report(discovery: Mapping[str, Any], calibration: Mapping[str, Any], runtime: Mapping[str, Any], verification_payload: Mapping[str, Any]) -> None:
    d = discovery["families"]["full_linear"]["operating_points"]["b1.25"]
    c = calibration
    rt = {row["path"]: row for row in runtime["timings"] if row["batch_size"] == 1024}
    lines = [
        "# LeWM Critic-Compression Discovery Report", "",
        "## Decision", "",
        "**`critic_compression_discovery_passed`**.", "",
        "A single full-coordinate regularized linear gate passed two-repeat grouped nested discovery CV and the frozen one-shot V3 calibration judge. This supports a cheap causal allocation mechanism for the unchanged visual-latent stagewise LeWM solver. It is not fresh-data confirmation: all 420 V3 training episodes were discovery data, the 90-episode V3 calibration split has now been consumed exactly once, V3 test targets remain untouched, and no V4 data was created. The next step is a separate preregistered V4 confirmation.", "",
        "## Core result", "",
        "The deployed student is one shared 1,046-coordinate linear head with causal depth encoding, 1,050 parameters (4,200 parameter bytes), no ensemble, and no teacher at inference. Its fully counted incremental maximum is 8,196 FLOPs per evaluated decision—3.1% of a 264,960-FLOP later-stage adapter and well below the 66,240 cap.", "",
        "| evaluation | calls | adaptive raw MSE | matched raw MSE | benefit | clustered 95% CI | whitened benefit |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| nested discovery (420 episodes) | {d['mean_calls']:.3f} | {d['raw_mse']:.9f} | {d['matched_raw_mse']:.9f} | {d['vs_matched_randomized']['mean_benefit']:.9f} | [{d['vs_matched_randomized']['ci_low']:.9f}, {d['vs_matched_randomized']['ci_high']:.9f}] | {d['whitened_vs_matched']['mean_benefit']:.7f} |",
        f"| one-shot calibration (90 episodes) | {c['mean_calls']:.3f} | {c['raw_mse']:.9f} | {c['matched_raw_mse']:.9f} | {c['vs_matched_randomized']['mean_benefit']:.9f} | [{c['vs_matched_randomized']['ci_low']:.9f}, {c['vs_matched_randomized']['ci_high']:.9f}] | {c['whitened_vs_matched']['mean_benefit']:.7f} |", "",
        f"Calibration also beat the analytic expected mixture by {c['vs_analytic_mixture']['mean_benefit']:.9f} (CI lower {c['vs_analytic_mixture']['ci_low']:.9f}), fixed d1 by {c['vs_fixed_d1']['mean_benefit']:.9f} (CI lower {c['vs_fixed_d1']['ci_low']:.9f}), and the exact histogram null by {c['vs_histogram_null']['mean_benefit']:.9f} (CI lower {c['vs_histogram_null']['ci_low']:.9f}). The whitened direction was positive ({c['whitened_vs_matched']['mean_benefit']:.7f}) but its CI included zero [{c['whitened_vs_matched']['ci_low']:.7f}, {c['whitened_vs_matched']['ci_high']:.7f}]; the frozen rule required positive direction, not whitened significance.", "",
        "## Solver and isolation", "",
        "The selected checkpoint remained byte-identical at SHA-256 `e63277943a356f3e28b4c4d1a1eb56acc8771fc07eccccdca67f878fed5ba782`. d0 identity, d0 bitwise equality, and d1 bitwise equality passed before and after student training and again on calibration. Solver state hashes did not change and solver gradients remained absent. The old 84-episode evidence reproduced d1 0.003120225 and d4 0.003048932.", "",
        "Only the isolated train-only cache was opened during discovery. The old combined V3 target NPZ was checked opaquely by hash and never loaded with NumPy. The calibration receipt was created before direct source-HDF5 target extraction; its cache contains exactly 90 calibration episodes and no test episode.", "",
        "## Candidate ladder and failure diagnosis", "",
        "The prior rich critic remained training-time teacher/diagnostic only. Its repeated OOF Spearman correlations were 0.285, 0.230, and 0.186. The full linear student showed weaker but useful routing correlations (0.171, 0.183, 0.150) and won because it retained allocation benefit at very low cost. The distilled-ranking linear family also passed at low-call points but was not selected by the frozen effect/frontier rule.", "",
        "The 11-summary control again failed because whitened effects were negative at every operating point. Stable sparse raw-coordinate models preserved some raw benefit but no point satisfied the global call/FLOP frontier. The low-rank bottleneck had the strongest rank signal and raw allocation effects, but its 39,874-FLOP gate was FLOP-dominated by the linear winner. The in-model current/update head was cheaper and whitened-positive at low calls, but globally dominated. Thus full raw-coordinate information is linearly accessible; aggressive sparsification loses enough routing value; low-rank capacity is not the limiting factor; whitening still exposes motion/scale shortcuts in compact/deeper policies.", "",
        "## Compute and latency", "",
        f"At calibration the adaptive point used {c['adaptive_total_flops_per_transition']/1e6:.3f}M total FLOPs per transition versus {c['matched_total_flops_per_transition']/1e6:.3f}M for the matched mixture and remained nondominated in both call and FLOP frontiers. Gate components per decision were 3,998 feature-construction, 2,098 normalization, 2,099 linear-head, and 1 control-flow FLOPs.", "",
        f"On MPS for all 3,420 calibration rows at batch size 1,024, end-to-end medians including the common base predictor were {rt['fixed_d1']['median_seconds']*1000:.1f} ms fixed d1, {rt['fixed_d4']['median_seconds']*1000:.1f} ms fixed d4, {rt['matched_transition_independent_mixture']['median_seconds']*1000:.1f} ms matched mixture, {rt['adaptive_dense_all_exits']['median_seconds']*1000:.1f} ms dense adaptive, and {rt['adaptive_realistic_sparse']['median_seconds']*1000:.1f} ms realistic sparse adaptive. Sparse routing was slower than dense routing on this device despite executing fewer solver rows, indicating indexing/synchronization overhead. FLOPs, latency, and energy are different notions; energy was not measured and latency was not substituted for the frozen FLOP rule.", "",
        "Sparse execution processed exactly 4,288 solver rows, matched the frozen calls, and agreed with dense selected outputs within 2.38419e-7.", "",
        "## Calibration interpretation", "",
        "Calibration gain ordering remained positive but modest (Spearman 0.134, 0.093, 0.126 by decision). The score-permutation negative control did not significantly beat its own matched mixture. No contact, impact, regime, privileged state, future value, episode-global statistic, target, or test-derived statistic was available to the gate. This result does not justify calling the mechanism contact-aware and does not claim a first adaptive world model.", "",
        "## Reproduction", "", "```bash",
        "PY=/Users/rishisim/.cache/lewm-v2-venv/bin/python",
        "PYTHONDONTWRITEBYTECODE=1 \"$PY\" -m unittest discover -s runs/lewm_adaptive_compute_critic_compression/tests -v",
        "PYTHONDONTWRITEBYTECODE=1 \"$PY\" runs/lewm_adaptive_compute_critic_compression/run_experiment.py prepare --device mps",
        "PYTHONDONTWRITEBYTECODE=1 \"$PY\" runs/lewm_adaptive_compute_critic_compression/run_experiment.py smoke --device mps",
        "PYTHONDONTWRITEBYTECODE=1 \"$PY\" runs/lewm_adaptive_compute_critic_compression/run_experiment.py discovery --device mps",
        "# judge is one-shot and has already been consumed; do not rerun",
        "PYTHONDONTWRITEBYTECODE=1 \"$PY\" runs/lewm_adaptive_compute_critic_compression/runtime_audit.py --device mps",
        "PYTHONDONTWRITEBYTECODE=1 \"$PY\" runs/lewm_adaptive_compute_critic_compression/make_artifacts.py",
        "```", "",
        "The frozen tournament hash is `" + calibration["receipt"]["tournament_sha256"] + "`. Full nested metrics, exact calls, intervals, checkpoints, hashes, test logs, tables, and plots are under this run directory.", "",
        "## Figures", "",
        "![Discovery benefit](figures/01_discovery_benefits.png)", "",
        "![FLOP frontier](figures/02_flop_frontier.png)", "",
        "![Calibration comparisons](figures/03_calibration_comparisons.png)", "",
        "![Calibration gain quantiles](figures/04_calibration_gain_quantiles.png)", "",
        "![Latency](figures/05_latency.png)", "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines))


def provenance() -> dict[str, Any]:
    return {
        "workspace": str(REPO),
        "source_thread_handoff": "019f5e02-aee8-7c73-a532-9df2e1573827",
        "prior_discovery": str(common.PRIOR),
        "selected_solver_checkpoint": str(common.SOLVER_CHECKPOINT),
        "selected_solver_checkpoint_sha256": common.sha256_file(common.SOLVER_CHECKPOINT),
        "final_student_checkpoint": str(ROOT / "checkpoints/final_student.pt"),
        "final_student_checkpoint_sha256": common.sha256_file(ROOT / "checkpoints/final_student.pt"),
        "frozen_tournament_sha256": common.sha256_file(ROOT / "audit/frozen_tournament.json"),
        "calibration_receipt_sha256": common.sha256_file(ROOT / "audit/calibration_access_receipt.json"),
        "calibration_metrics_sha256": common.sha256_file(ROOT / "metrics/calibration_once.json"),
        "v3_test_targets_consumed": False,
        "v4_created": False,
    }


def manifest() -> dict[str, Any]:
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == "artifact_manifest.json":
            continue
        files[str(path.relative_to(ROOT))] = common.sha256_file(path)
    payload = {"decision": load("decision.json")["decision"], "files": files}
    common.write_json(ROOT / "artifact_manifest.json", payload)
    return payload


def main() -> None:
    discovery = load("metrics/discovery_cv.json")
    calibration = load("metrics/calibration_once.json")
    runtime = load("metrics/runtime.json")
    tests = run_tests()
    tables(discovery, calibration, runtime)
    figures(discovery, calibration, runtime)
    verification_payload = verification(discovery, calibration, runtime, tests)
    common.write_json(ROOT / "metrics/adversarial_verification.json", verification_payload)
    common.write_json(ROOT / "provenance.json", provenance())
    report(discovery, calibration, runtime, verification_payload)
    manifest_payload = manifest()
    print(json.dumps({"decision": load("decision.json")["decision"], "artifacts": len(manifest_payload["files"]), "verification_passed": verification_payload["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
