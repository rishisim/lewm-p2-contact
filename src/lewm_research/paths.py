"""Resolve local research storage paths."""

import os
from pathlib import Path


def work_root() -> Path:
    """Return the internal SSD work root."""
    return Path(os.environ.get("LEWM_WORK_ROOT", "~/lewm-work")).expanduser().resolve()


def stablewm_home() -> Path:
    """Set and return the Stable WorldModel cache root."""
    home = work_root() / "stable-worldmodel"
    os.environ["STABLEWM_HOME"] = str(home)
    return home


def runs_root() -> Path:
    """Return the disposable run root."""
    return work_root() / "runs"


def checkpoint_dir(name: str) -> Path:
    """Return a named checkpoint directory."""
    return stablewm_home() / "checkpoints" / name


def dataset_path(name: str) -> Path:
    """Return a named dataset path."""
    return stablewm_home() / "datasets" / name
