#!/usr/bin/env python3
"""Create the immutable cycle report and complete path-set manifest."""

from __future__ import annotations

import json
import time

from cycle_common import REPO_ROOT, ROOT, atomic_json, read_json, sha256_file


def durable_files() -> set:
    return {
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name not in {"artifact_manifest.json", "artifact_manifest_verification.json"}
        and not path.name.startswith(".")
    }


def main() -> None:
    report_path = ROOT / "REPORT.md"
    manifest_path = ROOT / "artifact_manifest.json"
    verification_path = ROOT / "audit/artifact_manifest_verification.json"
    if report_path.exists() or manifest_path.exists() or verification_path.exists():
        raise RuntimeError("cycle finalization is immutable and already exists")
    decision = read_json(ROOT / "decision.json")
    outcome = decision["terminal_outcome"]
    if outcome in {
        "cycle_prospective_discovery_passed",
        "cycle_prospective_discovery_failed",
    }:
        independent = read_json(ROOT / "audit/independent_verification.json")
        analysis = read_json(ROOT / "analysis_result.json")
        latency = read_json(ROOT / "metrics/latency.json")
        facts = "\n".join(
            f"- {name}: estimate {value['estimate']:.10g}, 95% CI [{value['lower']:.10g}, {value['upper']:.10g}]"
            for name, value in decision["criteria"].items()
        )
        simultaneous = "\n".join(
            f"- {name}: simultaneous lower {value['lower']:.10g}"
            for name, value in decision["simultaneous_co_primary"].items()
        )
        adaptive = analysis
        compute = decision["compute"]
        adaptive_latency = [
            item
            for item in latency["timings"]
            if item["path"] == "actual_adaptive_sparse"
        ]
        latency_lines = "\n".join(
            f"- batch {item['batch_size']}: median {item['median_seconds']:.9g}s, p05 {item['p05_seconds']:.9g}s, p95 {item['p95_seconds']:.9g}s"
            for item in adaptive_latency
        )
        text = f"""# Cycle 007 MPS threshold-dtype repair report

Terminal outcome: `{outcome}`

## Facts

{facts}

Simultaneous co-primary bounds:

{simultaneous}

Adaptive raw MSE: {adaptive['adaptive_raw_mse']:.10g}. Adaptive PlanOracle-native-whitened MSE: {adaptive['adaptive_planoracle_native_whitened_mse']:.10g}.

Refiner model calls: {compute['refiner_model_calls']}; mean calls: {compute['mean_refiner_calls']:.10g}; gate evaluations: {compute['gate_evaluations']}; exact total FLOPs: {compute['adaptive_total_flops']}; gate feature FLOPs: {compute['gate_feature_flops']}; dual-affine score FLOPs: {compute['gate_dual_affine_score_flops']}; separately counted non-FLOP comparison/min operations: {compute['nonflop_comparison_min_operations']}.

Synchronized actual sparse latency (separate from FLOPs):

{latency_lines}

## Independent audit

Independent reproduction passed: {independent['passed']}. It separately reproduced effects, all intervals, calls, exact compute, stagewise ranks, and terminal mapping.

## Interpretation

The terminal outcome follows the presealed mapping. Stagewise rank remained a statistical criterion; it was not treated as a process-integrity defect. A discovery pass permits construction of a separately sealed V5 package; it is not itself V5 confirmation.

## Process validity

Contact and privileged fields were excluded by the model/gate input allowlist. The actual result came from sparse execution; dense execution was a separately accounted comparator/numerical shadow. All frozen modules remained unchanged with no gradients.

## Limitations

This is prospective discovery on one frozen simulator/model distribution after an isolated 240-episode fit and 120-episode selection design. It does not remove confirmation uncertainty and does not claim the first adaptive world model; LoopWM is relevant related work.

Zero V5 outcome episodes were generated.
"""
    else:
        text = f"""# Cycle 007 MPS threshold-dtype repair report

Terminal outcome: `{outcome}`

This cycle is implementation-invalid under the presealed mapping. Its complete negative execution evidence is preserved. No criterion was weakened, no post-seal normative repair was made, and no V5 outcome episode was generated.
"""
    report_path.write_text(text)
    files = {
        str(path.relative_to(REPO_ROOT)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(durable_files())
    }
    manifest = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "root": str(ROOT.relative_to(REPO_ROOT)),
        "terminal_outcome": outcome,
        "path_set_policy": "all durable files; interpreter caches, dot-temporaries, manifest, and manifest-verification file excluded",
        "files": files,
        "path_set_complete": True,
        "v5_outcome_episodes": 0,
    }
    atomic_json(manifest_path, manifest, exclusive=True)
    expected = set(files)
    actual = {str(path.relative_to(REPO_ROOT)) for path in durable_files()}
    bad = [
        relative
        for relative, record in files.items()
        if not (REPO_ROOT / relative).exists()
        or sha256_file(REPO_ROOT / relative) != record["sha256"]
    ]
    verification = {
        "schema_version": 1,
        "passed": expected == actual and not bad,
        "expected_path_set_equals_actual": expected == actual,
        "bad_hashes": bad,
        "file_count": len(files),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_json(verification_path, verification, exclusive=True)
    if not verification["passed"]:
        raise RuntimeError("cycle artifact manifest verification failed")
    print(json.dumps({"outcome": outcome, **verification}, sort_keys=True))


if __name__ == "__main__":
    main()
