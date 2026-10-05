"""Create run records outside the repository."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from importlib.metadata import version

from .device import resolve_device
from .paths import runs_root


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def create_run(stage: str, config: dict) -> Path:
    """Create a timestamped run with configuration and environment metadata."""
    if not stage or Path(stage).name != stage or stage in {".", ".."}:
        raise ValueError("stage must be one directory name")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = runs_root() / stage / f"{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    configured_device = config.get("device", config.get("solver", {}).get("device", "auto"))
    env = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "stable_worldmodel": version("stable-worldmodel"),
        "device": str(resolve_device(configured_device)),
        "platform": platform.platform(),
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    return run_dir


def write_metrics(run_dir: Path, metrics: dict) -> None:
    """Write aggregate run metrics."""
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
