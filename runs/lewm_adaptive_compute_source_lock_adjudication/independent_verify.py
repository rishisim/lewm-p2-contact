#!/usr/bin/env python3
"""Independent recomputation; deliberately imports no run analysis module."""
import hashlib, json, re, subprocess, tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / "runs/lewm_adaptive_compute_generator_forensics/historical/source_repo"

def raw(c, p):
    q = subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True)
    return q.stdout if q.returncode == 0 else None

def names(reqs):
    return sorted(re.sub(r"[-_.]+", "-", re.match(r"[A-Za-z0-9_.-]+", x).group()).lower() for x in reqs)

def main():
    proto = json.loads((HERE / "protocol.json").read_text())
    inv = json.loads((HERE / "candidate_inventory.json").read_text())
    dec = json.loads((HERE / "decision.json").read_text())
    met = json.loads((HERE / "metrics.json").read_text())
    commits = [x["commit"] for x in proto["candidate_discovery"]["candidates"]]
    recomputed = []
    for c in commits:
        py = tomllib.loads(raw(c, "pyproject.toml").decode())
        lock_bytes = raw(c, "uv.lock")
        lock = tomllib.loads(lock_bytes.decode())
        root = next(x for x in lock["package"] if x["name"] == "stable-worldmodel" and x.get("source", {}).get("editable") == ".")
        md = root["metadata"]["requires-dist"]
        src_base = names(py["project"]["dependencies"])
        lock_base = sorted(x["name"] for x in md if "extra ==" not in x.get("marker", ""))
        src_env = names(py["project"]["optional-dependencies"]["env"])
        lock_env = sorted(x["name"] for x in md if "extra == 'env'" in x.get("marker", ""))
        required = {p: raw(c, p) is not None for p in proto["eligibility"]["required_files"]}
        script = raw(c, "scripts/data/collect_cube.py").decode()
        config = raw(c, "scripts/data/config/ogb.yaml").decode()
        multiview = "multiview=True" in script
        collector = all([
            "'swm/OGBCube-v0'" in script, "env_type='single'" in script,
            "ExpertPolicy(" in script, "terminate_at_goal=False" in script,
            multiview, "max_episode_steps: 200" in config,
            "image_shape: [224, 224]" in config, "seed: 3072" in config,
            "num_traj: 10000" in config,
        ])
        base_ok = src_base == lock_base
        # Validate exact known differences independently, avoiding the analysis normalizer.
        if c.startswith("089"):
            base_diff_ok = set(lock_base) - set(src_base) == {"stable-baselines3"} and set(src_base) - set(lock_base) == set()
        else:
            base_diff_ok = set(lock_base) - set(src_base) == {"stable-baselines3"} and set(src_base) - set(lock_base) == {"rich", "typer"}
        recomputed.append({
            "commit": c,
            "version_ok": py["project"]["version"] == root["version"],
            "source_version": py["project"]["version"], "lock_version": root["version"],
            "base_ok": base_ok, "base_diff_ok": base_diff_ok,
            "env_name_multiset_ok": src_env == lock_env,
            "required_files_ok": all(required.values()), "collector_ok": collector,
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        })
    seal = lambda side: subprocess.run(["shasum", "-a", "256", "-c", side], cwd=HERE, capture_output=True).returncode == 0
    prior_manifest = ROOT / "runs/lewm_adaptive_compute_locked_stack_replay/artifact_manifest.json"
    prior_hash = hashlib.sha256(prior_manifest.read_bytes()).hexdigest()
    checks = {
        "protocol_seal": seal("protocol.sha256"), "preregistration_seal": seal("preregistration.sha256"),
        "candidate_count_and_ids": commits == ["089efbb87746e153e9ee051bf228b9c12aa3889a", "a265229cb29688715651cbd831a3b4c10b8f98b4"] and inv["candidate_count"] == 2,
        "tag_peels_exactly": subprocess.check_output(["git", "-C", str(REPO), "rev-list", "-n", "1", "0.0.4"], text=True).strip() == commits[0],
        "versions_incoherent": [(x["source_version"], x["lock_version"]) for x in recomputed] == [("0.0.4", "0.0.3"), ("0.0.5", "0.0.3")],
        "base_drift_recomputed": all(x["base_diff_ok"] and not x["base_ok"] for x in recomputed),
        "env_requirements_recomputed": all(x["env_name_multiset_ok"] for x in recomputed),
        "eligibility_mapping": not recomputed[0]["required_files_ok"] and not recomputed[0]["collector_ok"] and recomputed[1]["required_files_ok"] and recomputed[1]["collector_ok"] and inv["eligible_candidates"] == [],
        "allowlist_and_counts": proto["sample"]["episode_ids"] == [3316,5355,6405,2420,6763,8240,9767,8510] and len(proto["sample"]["transition_rows"]) == 8 and len(proto["sample"]["frame_rows"]) == 3 and met["transition_count_executed"] == 0,
        "decision_mapping": dec["outcome"] == "coherent_generator_candidate_not_identified" and dec["eligible_candidate_count"] == 0,
        "zero_prohibited_work": all([not dec["scientific_replay_executed"], not dec["candidate_environment_constructed"], not dec["hdf5_opened"], dec["new_policy_trajectories"] == 0, dec["model_or_gate_rows"] == 0, dec["v5_episodes"] == 0, not dec["v3_test_targets_opened"], not dec["combined_v3_cache_numpy_loaded"]]),
        "prior_run_manifest_preserved": prior_hash == "1bf76a5547f948f24f07c393548736d562e2169ef33cb6ac62494a93a6b0166d",
        "network_fetch_absent": inv["network_fetch_performed"] is False,
    }
    out = {"schema_version": 1, "status": "pass" if all(checks.values()) else "fail", "checks": checks, "recomputed": recomputed, "independence": {"candidate_selection_implementation_imported": False, "mechanics_implementation_imported": False, "environment_imported": False, "hdf5_opened": False}}
    (HERE / "independent_verification.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, sort_keys=True))
    raise SystemExit(0 if out["status"] == "pass" else 1)

if __name__ == "__main__": main()
