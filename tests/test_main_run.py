"""Resume/provenance and orchestration failure checks, without model evaluation."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from lewm_research import main
from lewm_research.probe.conditions import BaseScene
from lewm_research.probe.report import compute_report, markdown_report


def test_bases_resume_rejects_config_and_content_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    calls = []
    def generate(n, seed, **kwargs):
        calls.append((n, seed, kwargs))
        return [BaseScene(f"{seed}:0", 1, [0, 0], [0, 0, 0], [0, 0], [0, 0, 0], [0, 0], {})]
    monkeypatch.setattr(main, "generate_bases", generate)
    root = tmp_path / "runs" / "construction"
    path = main.prepare_bases(1, 300000000, 55, root)
    assert main.prepare_bases(1, 300000000, 55, root) == path
    assert len(calls) == 1
    with pytest.raises(ValueError, match="configuration differs"):
        main.prepare_bases(2, 300000000, 55, root)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum differs"):
        main.prepare_bases(1, 300000000, 55, root)


def test_frozen_normalization_rejected_before_evaluation(tmp_path):
    path = tmp_path / "normalization.json"
    path.write_text('{}')
    with pytest.raises(ValueError, match="frozen W4d"):
        main.check_normalization(path)


def test_coverage_power_limitation_in_report():
    result = compute_report([], [], [])
    result["coverage_underpowered"] = True
    assert "10-point cross-arm coverage comparison is underpowered" in markdown_report(result)


@pytest.mark.parametrize("failure", ["", "reference-F", "lewm-pusht"])
def test_driver_resume_and_failure_markers(tmp_path, failure):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys, time
args = sys.argv[1:]
if args[:2] == ["run", "python"]:
    if "prepare_output_root" in args[3]: print(args[4])
    sys.exit(0)
args = args[2:]
root = pathlib.Path(os.environ["LEWM_WORK_ROOT"]) / "runs" / "main"
with (root / "calls.jsonl").open("a") as f: f.write(json.dumps(args) + "\\n")
out = pathlib.Path(args[args.index("--run-dir") + 1])
if out.name == os.environ.get("FAIL_STAGE"): sys.exit(9)
time.sleep(.05)
out.mkdir(parents=True, exist_ok=True)
''')
    uv.chmod(0o755)
    env = {**os.environ, "LEWM_WORK_ROOT": str(tmp_path), "FAIL_STAGE": failure,
           "PATH": str(bindir) + os.pathsep + os.environ["PATH"]}
    script = Path(__file__).resolve().parents[1] / "experiments/role_swap/run_main.sh"
    run = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=15)
    root = tmp_path / "runs/main"
    if failure:
        assert run.returncode != 0, run.stdout + run.stderr
        assert (root / "FAILED").exists()
        assert not (root / "DONE").exists()
        assert not (root / ".lock").exists()
    else:
        assert run.returncode == 0, run.stdout + run.stderr
        assert (root / "DONE").exists()
        calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
        abc = [c for c in calls if c[0] == "probe-abc"]
        assert {c[c.index("--conditions") + 1] for c in abc} == {"off_path,near_path", "move_peg,move_T_matched"}
        assert all(c[c.index("--n") + 1] == "50" for c in abc)
        report = next(c for c in calls if c[0] == "probe-report")
        assert "--publish" in report and "--coverage-underpowered" in report
        before = (root / "calls.jsonl").read_text()
        assert subprocess.run(["bash", str(script)], env=env, capture_output=True, timeout=15).returncode == 0
        assert (root / "calls.jsonl").read_text() == before
