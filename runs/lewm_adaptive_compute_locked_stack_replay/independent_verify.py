#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def load(n): return json.loads((ROOT / n).read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p, e, d = load("protocol.json"), load("environment_provenance.json"), load("decision.json")
    lock = (ROOT / "source_repo/uv.lock").read_text()
    project = (ROOT / "source_repo/pyproject.toml").read_text()
    ids = p["sample"]["episode_ids"]
    checks = {
        "protocol_seal": sha(ROOT / "protocol.json") == "9d6fd38e9a1be045dc7cc0012099795cc2d18839e650db572fb7f6b866cd0993",
        "head": subprocess.check_output(["git", "-C", str(ROOT / "source_repo"), "rev-parse", "HEAD"], text=True).strip() == p["candidate"]["source_commit"],
        "lock_hash": sha(ROOT / "source_repo/uv.lock") == p["candidate"]["uv_lock_sha256"],
        "lock_root_version": 'name = "stable-worldmodel"\nversion = "0.0.3"' in lock,
        "source_version": 'name = "stable-worldmodel"\nversion = "0.0.5"' in project,
        "installed_version": e["installed_packages"]["stable-worldmodel"] == "0.0.5",
        "dependency_versions": all(e["installed_packages"][k] == v for k,v in p["candidate"]["required_packages"].items() if k != "stable-worldmodel"),
        "binding": e["checks"]["run_local_interpreter"] and e["checks"]["run_local_source_import"] and e["checks"]["no_import_from_current_project_environment"],
        "counts_frozen": len(ids) == 8 and p["indices"]["transition_count"] == 64 and p["indices"]["frame_count"] == 24,
        "isolation": p["sample"]["v3_test_episode_intersection"] == [] and not e["hdf5_opened"] and not e["v3_test_targets_opened"] and not e["combined_v3_cache_opened_with_numpy"],
        "zero_prohibited_work": e["new_policy_trajectories"] == e["model_or_gate_rows"] == e["v5_episodes"] == 0,
        "decision": d["outcome"] == "locked_stack_unavailable" and d["scientific_replay_executed"] is False
    }
    out = {"schema_version": 1, "status": "pass" if all(checks.values()) else "fail", "checks": checks,
           "independence": {"analysis_implementation_imported": False, "environment_imported": False, "hdf5_opened": False},
           "verified_blocker": "uv.lock root 0.0.3 conflicts with immutable source pyproject 0.0.5; frozen sync installed source metadata 0.0.5"}
    (ROOT / "independent_verification.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "checks": checks}, sort_keys=True))
    if out["status"] != "pass": raise SystemExit(1)
if __name__ == "__main__": main()
