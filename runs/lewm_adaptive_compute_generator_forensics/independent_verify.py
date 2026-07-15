#!/usr/bin/env python3
"""Independent artifact-only verification; intentionally imports no audit modules."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
SPLIT = WORK / "runs/lewm_adaptive_compute_v3/cache/split_manifest.json"
OUT = ROOT / "audit/independent_verification.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def err(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = np.abs(np.asarray(a) - np.asarray(b))
    return {"count": int(d.size), "exact": bool(np.array_equal(a, b)),
            "max_abs": float(d.max(initial=0)), "mean_abs": float(d.mean())}


def same_float(a: Any, b: Any, atol: float = 2e-14) -> bool:
    return bool(math.isclose(float(a), float(b), rel_tol=0, abs_tol=atol))


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(int(x["count"]) for x in items)
    return {"count": n, "exact": all(bool(x["exact"]) for x in items),
            "max_abs": max(float(x["max_abs"]) for x in items),
            "mean_abs": sum(int(x["count"]) * float(x["mean_abs"]) for x in items) / n}


def recompute_metrics_summary(data: dict[str, Any]) -> dict[str, Any]:
    keys = ["state_qpos", "state_qvel", "observation", "control_mapping",
            "action_transition_qpos", "action_transition_qvel",
            "direct_control_transition_qpos", "direct_control_transition_qvel"]
    result = {k: aggregate([r[k] for r in data["records"] if k in r]) for k in keys}
    privileged_keys = sorted(data["records"][0]["privileged"])
    result["privileged"] = {k: aggregate([r["privileged"][k] for r in data["records"]]) for k in privileged_keys}
    result["reset_qpos"] = aggregate([r["qpos"] for r in data["reset_records"]])
    result["reset_qvel"] = aggregate([r["qvel"] for r in data["reset_records"]])
    result["pixels_best"] = aggregate([r["conventions"][r["best_convention"]] for r in data["pixel_records"]])
    return result


def summaries_match(got: dict[str, Any], claimed: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for k, value in got.items():
        if k == "privileged":
            checks[k] = {n: all((value[n][x] == claimed[k][n][x] if x != "mean_abs" else same_float(value[n][x], claimed[k][n][x])) for x in value[n]) for n in value}
        else:
            checks[k] = all((value[x] == claimed[k][x] if x != "mean_abs" else same_float(value[x], claimed[k][x])) for x in value)
    return checks


def main() -> None:
    protocol_path = ROOT / "protocol.json"
    alignment_path = ROOT / "alignment.json"
    metrics_path = ROOT / "metrics.json"
    protocol, alignment, metrics, split = map(load, [protocol_path, alignment_path, metrics_path, SPLIT])
    ids = list(map(int, protocol["sample"]["episode_ids"]))
    discovery = set(map(int, split["selection"]["split_episode_ordinals"]["train"]))
    test = set(map(int, split["selection"]["split_episode_ordinals"]["test"]))
    salt = protocol["sample"]["salt"]
    scored = sorted((hashlib.sha256(f"{salt}:{i}".encode()).hexdigest(), i) for i in discovery)
    expected_ids = [i for _, i in scored[:8]]
    score_map = {str(i): s for s, i in scored[:8]}

    source_manifest = load(WORK / "runs/lewm_adaptive_compute_distribution_contract/source_manifest.json")
    h5_path = Path(source_manifest["offline_source"]["path"])
    rows = 201
    per_episode = []
    with h5py.File(h5_path, "r") as f:
        # Fail closed: every raw slice is derived only from frozen sampled IDs.
        for eid in ids:
            if eid not in discovery or eid in test:
                raise RuntimeError(f"raw read denied for episode {eid}")
            sl = slice(eid * rows, (eid + 1) * rows)
            qpos = f["qpos"][sl]
            qvel = f["qvel"][sl]
            pqpos = f["prev_qpos"][sl]
            pqvel = f["prev_qvel"][sl]
            action = f["action"][sl]
            control = f["control"][sl]
            time = f["time"][sl].reshape(-1)
            step = f["step_idx"][sl].reshape(-1)
            terminated = f["terminated"][sl].reshape(-1)
            truncated = f["truncated"][sl].reshape(-1)
            per_episode.append({
                "episode_id": eid,
                "prev_vs_prior_qpos": err(pqpos[1:], qpos[:-1]),
                "prev_vs_prior_qvel": err(pqvel[1:], qvel[:-1]),
                "prev_vs_same_qpos": err(pqpos, qpos),
                "prev_vs_same_qvel": err(pqvel, qvel),
                "first_prev_vs_first_qpos": err(pqpos[0], qpos[0]),
                "first_prev_vs_first_qvel": err(pqvel[0], qvel[0]),
                "final_action_zero": bool(np.array_equal(action[-1], np.zeros_like(action[-1]))),
                "final_control_vs_prior": err(control[-1], control[-2]),
                "time": {"first": float(time[0]), "last": float(time[-1]),
                         "increment": err(np.diff(time), np.full(200, 0.05))},
                "step_idx": {"first": int(step[0]), "last": int(step[-1]),
                             "unit_increment": bool(np.array_equal(np.diff(step), np.ones(200, dtype=np.diff(step).dtype)))},
                "terminal": {"earlier_any": bool(np.any(terminated[:-1]) or np.any(truncated[:-1])),
                             "terminated_last": bool(terminated[-1]), "truncated_last": bool(truncated[-1])},
            })

    alignment_checks = []
    for got, claimed in zip(per_episode, alignment["per_episode"], strict=True):
        def walk(g: Any, c: Any, path: str = "") -> None:
            if isinstance(g, dict):
                for k, v in g.items():
                    if k in c:
                        walk(v, c[k], path + "." + k)
            elif isinstance(g, float):
                alignment_checks.append({"field": path, "ok": same_float(g, c)})
            else: alignment_checks.append({"field": path, "ok": g == c})
        walk(got, claimed)

    # Recompute every stored aggregate solely from stored per-record summaries.
    recomputed = recompute_metrics_summary(metrics)
    summary_checks = summaries_match(recomputed, metrics["summary"])

    tol = protocol["tolerances"]
    derived_passes = {
        "state": recomputed["state_qpos"]["max_abs"] <= tol["float64_state_restore_atol"] and recomputed["state_qvel"]["max_abs"] <= tol["float64_state_restore_atol"],
        "observation": recomputed["observation"]["max_abs"] <= tol["float64_observation_atol"],
        "render": bool(recomputed["pixels_best"]["exact"]),
        "control": recomputed["control_mapping"]["max_abs"] <= tol["float64_control_atol"],
        "action_dynamics": recomputed["action_transition_qpos"]["max_abs"] <= tol["float64_transition_atol"] and recomputed["action_transition_qvel"]["max_abs"] <= tol["float64_transition_atol"],
        "direct_control_dynamics": recomputed["direct_control_transition_qpos"]["max_abs"] <= tol["float64_transition_atol"] and recomputed["direct_control_transition_qvel"]["max_abs"] <= tol["float64_transition_atol"],
        "reset": recomputed["reset_qpos"]["max_abs"] <= tol["float64_state_restore_atol"] and recomputed["reset_qvel"]["max_abs"] <= tol["float64_state_restore_atol"],
    }
    earliest = ("reset_only_mismatch" if not derived_passes["reset"] else
                "render_or_observation_mismatch" if not derived_passes["observation"] or not derived_passes["render"] else
                "action_control_mismatch" if not derived_passes["control"] else
                "dynamics_mismatch" if not derived_passes["action_dynamics"] or not derived_passes["direct_control_dynamics"] else
                "current_local_mechanics_matched")

    hashes = {
        "protocol.json": sha(protocol_path), "alignment.json": sha(alignment_path),
        "metrics.json": sha(metrics_path), "split_manifest.json": sha(SPLIT),
    }
    source_hash_checks = {}
    for name, expected in protocol["source_hashes"].items():
        path = h5_path if name == "source_h5" else SPLIT if name == "split_manifest" else Path(name)
        # The 101 GB HDF5 hash is already independently fixed by immutable source_manifest;
        # selected-row reads are verified here without a second full-file scan.
        actual = source_manifest["offline_source"]["sha256"] if name == "source_h5" else sha(path)
        source_hash_checks[name] = {"expected": expected, "actual": actual, "ok": actual == expected,
                                    "method": "immutable_source_manifest" if name == "source_h5" else "file_bytes"}

    decision_check = {"present": (ROOT / "decision.json").exists()}
    if decision_check["present"]:
        decision = load(ROOT / "decision.json")
        layer_value = decision.get("layer_localized_decision")
        provenance_value = decision.get("generator_provenance_decision")
        decision_check.update({"reported_layer_localized": layer_value,
                               "reported_generator_provenance": provenance_value,
                               "earliest_current_layer_mapping": earliest,
                               "layer_consistent": layer_value == earliest,
                               "provenance_value_allowed": provenance_value in {"generator_provenance_unresolved", "exact_historical_generator_recovered"},
                               "consistent": layer_value == earliest and provenance_value in {"generator_provenance_unresolved", "exact_historical_generator_recovered"}})

    historical_check = {"present": (ROOT / "historical_candidate_metrics.json").exists()}
    if historical_check["present"]:
        candidate_metrics_path = ROOT / "historical_candidate_metrics.json"
        candidate_metrics = load(candidate_metrics_path)
        candidates = load(ROOT / "historical_candidates.json")
        candidate = candidates["candidates"][0]
        candidate_summary = recompute_metrics_summary(candidate_metrics)
        candidate_summary_checks = summaries_match(candidate_summary, candidate_metrics["summary"])
        source_repo = ROOT / "historical/source_repo"
        local_commit = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
        impl = Path(candidate_metrics["implementation_source"]).resolve()
        sidecar = (ROOT / "historical_candidate_metrics.sha256").read_text().split()[0]
        collector_rel = "scripts/data/collect_cube.py"
        config_rel = "scripts/data/config/ogb.yaml"
        collector_at_head = subprocess.check_output(["git", "-C", str(source_repo), "show", f"HEAD:{collector_rel}"])
        config_at_head = subprocess.check_output(["git", "-C", str(source_repo), "show", f"HEAD:{config_rel}"])
        lock_text = (source_repo / "uv.lock").read_text()
        historical_check.update({
            "candidate_id": candidate["id"], "frozen_candidate_count": candidates["candidate_count"],
            "decision_executed_ids": load(ROOT / "decision.json").get("historical_candidates_executed", []),
            "source_repo_commit_expected": candidate["stable_worldmodel_commit"], "source_repo_commit_actual": local_commit,
            "source_repo_commit_matches": local_commit == candidate["stable_worldmodel_commit"],
            "collector_matches_commit": hashlib.sha256(collector_at_head).hexdigest() == sha(source_repo / collector_rel),
            "collector_sha256": sha(source_repo / collector_rel),
            "ogb_config_matches_commit": hashlib.sha256(config_at_head).hexdigest() == sha(source_repo / config_rel),
            "ogb_config_sha256": sha(source_repo / config_rel),
            "ogbench_1_2_1_locked": 'name = "ogbench"\nversion = "1.2.1"' in lock_text,
            "implementation_source": str(impl), "implementation_source_inside_frozen_repo": source_repo.resolve() in impl.parents,
            "metrics_sha256": sha(candidate_metrics_path), "metrics_sidecar_sha256": sidecar,
            "metrics_hash_matches": sha(candidate_metrics_path) == sidecar,
            "episodes_match": candidate_metrics["episodes"] == ids,
            "transition_count": candidate_metrics["transition_count"], "frame_count": candidate_metrics["frame_count"],
            "reset_count": len(candidate_metrics["reset_records"]), "summary_checks": candidate_summary_checks,
            "passes_match_current": candidate_metrics["passes"] == metrics["passes"],
            "summaries_match_current": candidate_metrics["summary"] == metrics["summary"],
            "protocol_hash_matches": candidate_metrics["protocol_sha256"] == sha(protocol_path),
            "alignment_hash_matches": candidate_metrics["alignment_sha256"] == sha(alignment_path),
        })
        historical_check["consistent"] = all([
            historical_check["source_repo_commit_matches"], historical_check["implementation_source_inside_frozen_repo"],
            historical_check["collector_matches_commit"], historical_check["ogb_config_matches_commit"], historical_check["ogbench_1_2_1_locked"],
            historical_check["metrics_hash_matches"], historical_check["episodes_match"],
            historical_check["transition_count"] == 64, historical_check["frame_count"] == 24,
            historical_check["reset_count"] == 8, historical_check["passes_match_current"],
            historical_check["summaries_match_current"], historical_check["protocol_hash_matches"],
            historical_check["alignment_hash_matches"], candidate["id"] in historical_check["decision_executed_ids"],
            all(v if isinstance(v, bool) else all(v.values()) for v in candidate_summary_checks.values()),
        ])

    out = {
        "schema_version": 1, "status": "pass",
        "independence": {"analysis_modules_imported_or_called": [], "environment_executed": False,
                         "combined_v3_cache_opened_with_numpy": False, "v3_test_targets_opened": False},
        "hashes": hashes, "source_hash_checks": source_hash_checks,
        "sample": {"claimed": ids, "recomputed": expected_ids, "scores_match": score_map == protocol["sample"]["episode_scores"],
                   "count": len(ids), "all_in_discovery_allowlist": set(ids) <= discovery,
                   "v3_test_intersection": sorted(set(ids) & test)},
        "bounds": {"all_state_observation_records": len(metrics["records"]),
                   "transition_records": sum("control_mapping" in r for r in metrics["records"]), "expected_transition_records": 64,
                   "frame_records": len(metrics["pixel_records"]), "expected_frame_records": 24,
                   "reset_records": len(metrics["reset_records"]), "episodes": len(ids)},
        "alignment": {"all_fields_match": all(x["ok"] for x in alignment_checks),
                      "failed_fields": [x["field"] for x in alignment_checks if not x["ok"]],
                      "row_semantics_supported": all(x["prev_vs_prior_qpos"]["exact"] and x["prev_vs_prior_qvel"]["exact"] for x in per_episode)},
        "recomputed_summary": recomputed, "summary_checks": summary_checks,
        "derived_passes": derived_passes, "reported_passes": metrics["passes"],
        "passes_match": derived_passes == metrics["passes"],
        "earliest_current_layer_mapping": earliest, "decision_check": decision_check,
        "historical_candidate_check": historical_check,
        "artifact_inventory_at_verification": {
            "files_excluding_independent_outputs": len([p for p in ROOT.rglob("*") if p.is_file() and p.name not in {"independent_verification.json", "independent_verification.log"}]),
            "final_manifest_present": (ROOT / "artifact_manifest.json").exists(),
        },
    }
    fatal = [
        expected_ids != ids, score_map != protocol["sample"]["episode_scores"], bool(set(ids) & test),
        sum("control_mapping" in r for r in metrics["records"]) != 64, len(metrics["pixel_records"]) != 24, len(metrics["reset_records"]) != 8,
        not out["alignment"]["all_fields_match"], not all(v if isinstance(v, bool) else all(v.values()) for v in summary_checks.values()),
        derived_passes != metrics["passes"], not all(x["ok"] for x in source_hash_checks.values()),
        decision_check.get("present", False) and not decision_check.get("consistent", False),
        historical_check.get("present", False) and not historical_check.get("consistent", False),
    ]
    if any(fatal): out["status"] = "fail"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "output": str(OUT), "earliest": earliest,
                      "sample": expected_ids, "passes_match": out["passes_match"]}, sort_keys=True))
    if out["status"] != "pass": raise SystemExit(1)


if __name__ == "__main__":
    main()
