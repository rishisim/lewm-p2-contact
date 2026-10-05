"""Fast tests for the local research CLI and storage paths."""

import json

from typer.testing import CliRunner

from lewm_research import device, paths, runs
from lewm_research.cli import app


def test_paths_override(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    assert paths.work_root() == tmp_path
    assert paths.stablewm_home() == tmp_path / "stable-worldmodel"
    assert paths.dataset_path("data.h5") == tmp_path / "stable-worldmodel/datasets/data.h5"
    assert paths.checkpoint_dir("model") == tmp_path / "stable-worldmodel/checkpoints/model"
    assert paths.runs_root() == tmp_path / "runs"


def test_device_fallback(monkeypatch):
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device.torch.backends.mps, "is_available", lambda: False)
    assert str(device.resolve_device("auto")) == "cpu"
    assert str(device.resolve_device("cuda")) == "cpu"
    assert str(device.resolve_device("mps")) == "cpu"
    assert device.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_run_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    run_dir = runs.create_run("unit", {"seed": 42})
    runs.write_metrics(run_dir, {"ok": True})
    assert json.loads((run_dir / "config.json").read_text()) == {"seed": 42}
    assert json.loads((run_dir / "metrics.json").read_text()) == {"ok": True}
    assert "python" in json.loads((run_dir / "env.json").read_text())


def test_cli_help():
    runner = CliRunner()
    for args in ([], ["smoke-eval", "--help"]):
        result = runner.invoke(app, args or ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output


def test_smoke_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("LEWM_WORK_ROOT", str(tmp_path))
    result = CliRunner().invoke(app, ["smoke-eval"])
    assert result.exit_code == 1
    assert "PushT dataset missing" in result.output
