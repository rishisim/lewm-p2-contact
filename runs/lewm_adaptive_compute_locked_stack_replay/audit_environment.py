#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib, importlib.metadata, inspect, json, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "source_repo"

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1 << 20): h.update(b)
    return h.hexdigest()

def main() -> None:
    names = ["stable-worldmodel", "ogbench", "mujoco", "dm-control", "gymnasium", "numpy", "h5py", "hdf5plugin"]
    modules = ["stable_worldmodel", "ogbench", "mujoco", "dm_control", "gymnasium", "numpy", "h5py", "hdf5plugin"]
    imports = {}
    for name in modules:
        mod = importlib.import_module(name)
        imports[name] = str(Path(inspect.getfile(mod)).resolve())
    installed = {name: importlib.metadata.version(name) for name in names}
    lock = (REPO / "uv.lock").read_text()
    pyproject = (REPO / "pyproject.toml").read_text()
    expected = json.loads((ROOT / "protocol.json").read_text())["candidate"]["required_packages"]
    checks = {
        "head_matches": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() == "a265229cb29688715651cbd831a3b4c10b8f98b4",
        "lock_hash_matches": sha(REPO / "uv.lock") == "0d1659e36b8cb4b054b95ae69cfabb174cc15193dbba126e0c8cb5da5751724d",
        "python_matches": platform.python_version() == "3.10.20",
        "run_local_interpreter": Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        "run_local_source_import": Path(imports["stable_worldmodel"]).resolve().is_relative_to(REPO.resolve()),
        "no_import_from_current_project_environment": all("/World Models/le-wm/.venv/" not in p and "/le-wm/.venv/" not in p for p in imports.values()),
        "locked_dependencies_match": all(installed[k] == v for k, v in expected.items() if k != "stable-worldmodel"),
        "stable_worldmodel_distribution_matches_lock": installed["stable-worldmodel"] == expected["stable-worldmodel"],
        "lock_declares_root_0_0_3": 'name = "stable-worldmodel"\nversion = "0.0.3"' in lock,
        "source_declares_0_0_5": 'name = "stable-worldmodel"\nversion = "0.0.5"' in pyproject,
    }
    out = {
        "schema_version": 1, "status": "blocked_exact_environment",
        "interpreter": sys.executable, "python": platform.python_version(), "platform": platform.platform(),
        "git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "hashes": {"uv.lock": sha(REPO / "uv.lock"), "pyproject.toml": sha(REPO / "pyproject.toml"), "stable_worldmodel/__init__.py": sha(REPO / "stable_worldmodel/__init__.py")},
        "installed_packages": installed, "import_locations": imports, "checks": checks,
        "blocker": "Immutable commit pyproject.toml declares stable-worldmodel 0.0.5 while its frozen uv.lock root package entry declares 0.0.3; uv sync --frozen installs the run-local source with 0.0.5 distribution metadata.",
        "scientific_environment_executed": False, "hdf5_opened": False,
        "v3_test_targets_opened": False, "combined_v3_cache_opened_with_numpy": False,
        "new_policy_trajectories": 0, "model_or_gate_rows": 0, "v5_episodes": 0
    }
    (ROOT / "environment_provenance.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    if checks["stable_worldmodel_distribution_matches_lock"]: raise SystemExit("expected frozen inconsistency not observed")
    print(json.dumps({"status": out["status"], "blocker": out["blocker"]}, sort_keys=True))

if __name__ == "__main__": main()
