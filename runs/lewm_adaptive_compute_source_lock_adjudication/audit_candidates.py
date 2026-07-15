#!/usr/bin/env python3
"""Read-only audit of the two preregistered immutable git objects."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
REPO = ROOT / "runs/lewm_adaptive_compute_generator_forensics/historical/source_repo"
CANDIDATES = [
    ("089efbb87746e153e9ee051bf228b9c12aa3889a", "0.0.4"),
    ("a265229cb29688715651cbd831a3b4c10b8f98b4", None),
]
FILES = [
    "scripts/data/collect_cube.py",
    "scripts/data/config/ogb.yaml",
    "stable_worldmodel/envs/ogbench/cube_env.py",
    "stable_worldmodel/envs/ogbench/expert_policy.py",
    "stable_worldmodel/envs/ogbench/__init__.py",
]
LEGACY_FILES = [
    "stable_worldmodel/envs/ogbench_manip/cube_env.py",
    "stable_worldmodel/envs/ogbench_manip/expert_policy.py",
    "stable_worldmodel/envs/ogbench_manip/__init__.py",
]


def git(*args: str, binary: bool = False):
    p = subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True)
    return p.stdout if binary else p.stdout.decode()


def blob(commit: str, path: str) -> bytes | None:
    p = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True)
    return p.stdout if p.returncode == 0 else None


def canon_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def req_text(req: str) -> dict:
    # The candidate declarations are simple PEP 508 strings. Preserve the full
    # suffix while canonicalizing name/extras so equality is deterministic.
    m = re.match(r"^([A-Za-z0-9_.-]+)(\[[^]]+\])?(.*)$", req.strip())
    assert m, req
    extras = sorted(canon_name(x.strip()) for x in (m.group(2) or "")[1:-1].split(",") if x.strip())
    return {"name": canon_name(m.group(1)), "extras": extras, "suffix": m.group(3).strip()}


def lock_req(item: dict) -> dict:
    suffix = ""
    if item.get("specifier"):
        suffix += item["specifier"]
    if item.get("marker"):
        suffix += "; " + item["marker"]
    if item.get("url"):
        suffix += " @ " + item["url"]
    extras = item.get("extras", item.get("extra", []))
    return {"name": canon_name(item["name"]), "extras": sorted(canon_name(x) for x in extras), "suffix": suffix}


def sorted_reqs(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda x: (x["name"], x["extras"], x["suffix"]))


def diff_names(source: list[dict], locked: list[dict]) -> dict:
    ss = {json.dumps(x, sort_keys=True) for x in source}
    ls = {json.dumps(x, sort_keys=True) for x in locked}
    return {
        "source_only": [json.loads(x) for x in sorted(ss - ls)],
        "lock_only": [json.loads(x) for x in sorted(ls - ss)],
    }


def facts(script: str, config: str) -> dict:
    return {
        "environment_id": "swm/OGBCube-v0" if "'swm/OGBCube-v0'" in script else None,
        "env_type": "single" if "env_type='single'" in script else None,
        "policy": "ExpertPolicy" if "ExpertPolicy(" in script else None,
        "trajectory_count": int(re.search(r"num_traj:\s*(\d+)", config).group(1)),
        "action_horizon": int(re.search(r"max_episode_steps:\s*(\d+)", config).group(1)),
        "seed": int(re.search(r"^seed:\s*(\d+)", config, re.M).group(1)),
        "terminate_at_goal": False if "terminate_at_goal=False" in script else None,
        "multiview": True if "multiview=True" in script else (False if "multiview=False" in script else None),
        "image_shape": [int(x) for x in re.search(r"image_shape:\s*\[(\d+),\s*(\d+)\]", config).groups()],
        "dataset_name": re.search(r"world\.record_dataset\(\s*'([^']+)'", script).group(1),
        "explicit_plan_oracle": "policy_type='plan_oracle'" in script,
    }


def main() -> None:
    protocol = json.loads((RUN / "protocol.json").read_text())
    wanted = protocol["eligibility"]["collector_contract"]
    records = []
    commands = []
    for commit, tag in CANDIDATES:
        commands += [f"git cat-file -t {commit}", f"git show {commit}:pyproject.toml", f"git show {commit}:uv.lock"]
        ident = git("show", "-s", "--format=%H%n%aI%n%cI%n%s%n%P", commit).splitlines()
        py_b = blob(commit, "pyproject.toml")
        lock_b = blob(commit, "uv.lock")
        assert py_b and lock_b
        py = tomllib.loads(py_b.decode())
        lock = tomllib.loads(lock_b.decode())
        root = next(p for p in lock["package"] if p["name"] == "stable-worldmodel" and p.get("source", {}).get("editable") == ".")
        metadata = root.get("metadata", {})
        source_base = sorted_reqs([req_text(x) for x in py["project"].get("dependencies", [])])
        source_env = sorted_reqs([req_text(x) for x in py["project"].get("optional-dependencies", {}).get("env", [])])
        lock_meta_base = sorted_reqs([lock_req(x) for x in metadata.get("requires-dist", [])])
        lock_meta_env = sorted_reqs([lock_req(x) for x in metadata.get("requires-dev", {}).get("env", [])])
        # uv lock versions in this history represent extras in requires-dist
        # entries carrying marker "extra == 'env'" rather than requires-dev.
        if not lock_meta_env:
            all_meta = metadata.get("requires-dist", [])
            lock_meta_env = sorted_reqs([lock_req({k: v for k, v in x.items() if k != "marker"}) for x in all_meta if "extra == 'env'" in x.get("marker", "")])
            lock_meta_base = sorted_reqs([lock_req(x) for x in all_meta if "extra ==" not in x.get("marker", "")])
        file_inventory = {}
        for path in FILES + LEGACY_FILES:
            data = blob(commit, path)
            file_inventory[path] = None if data is None else {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": git("rev-parse", f"{commit}:{path}").strip()}
        script = blob(commit, FILES[0]).decode()
        config = blob(commit, FILES[1]).decode()
        found = facts(script, config)
        collector_match = all(found[k] == v for k, v in wanted.items())
        version_ok = py["project"]["version"] == root["version"]
        base_ok = source_base == lock_meta_base
        env_ok = source_env == lock_meta_env
        files_ok = all(file_inventory[x] is not None for x in protocol["eligibility"]["required_files"])
        temporal_ok = ident[1] < "2026-03-26T20:49:37Z"
        tag_identity = None
        if tag:
            tag_identity = {"ref": f"refs/tags/{tag}", "peeled_commit": git("rev-list", "-n", "1", tag).strip(), "matches": git("rev-list", "-n", "1", tag).strip() == commit}
        last_lock = git("log", "-1", "--format=%H%n%aI%n%s", commit, "--", "uv.lock").splitlines()
        checks = {"version_coherent": version_ok, "base_requirements_coherent": base_ok, "env_requirements_coherent": env_ok, "required_files_exist": files_ok, "collector_contract_matches": collector_match, "temporal_outer_bound": temporal_ok}
        records.append({
            "commit": commit, "tag_identity": tag_identity,
            "identity": {"commit": ident[0], "author_date": ident[1], "committer_date": ident[2], "message": ident[3], "parents": ident[4].split()},
            "hashes": {"pyproject.toml": hashlib.sha256(py_b).hexdigest(), "uv.lock": hashlib.sha256(lock_b).hexdigest()},
            "lock_last_change": {"commit": last_lock[0], "date": last_lock[1], "message": last_lock[2]},
            "source": {"project_version": py["project"]["version"], "base": source_base, "env": source_env},
            "lock_root": {"project_version": root["version"], "base_metadata": lock_meta_base, "env_metadata": lock_meta_env},
            "contract_diffs": {"base": diff_names(source_base, lock_meta_base), "env": diff_names(source_env, lock_meta_env)},
            "files": file_inventory, "collector_facts": found, "checks": checks,
            "static_eligible": all(checks.values()),
            "mtime_caveat": "commit predates preserved HDF5 mtime" if ident[1] < "2026-02-18T20:12:16-07:00" else "commit postdates preserved HDF5 mtime; mtime is not proof of generation time",
        })
    eligible = [r["commit"] for r in records if r["static_eligible"]]
    out = {"schema_version": 1, "repository": str(REPO), "repository_remote": git("remote", "get-url", "origin").strip(), "network_fetch_performed": False, "commands": commands, "candidate_count": len(records), "eligible_candidates": eligible, "candidates": records}
    (RUN / "candidate_inventory.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (RUN / "source_lock_contract_diffs.json").write_text(json.dumps({"schema_version": 1, "candidates": [{"commit": r["commit"], "checks": r["checks"], "diffs": r["contract_diffs"]} for r in records]}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
