"""Small shared persistence/statistics helpers for Step-4 analyses."""

from pathlib import Path
import hashlib
import json

import numpy as np

from ..runs import create_run
from ..paths import runs_root
from .rollout_eval import prepare_output_root
from .stats import paired_cluster_bootstrap


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_names(names):
    names = list(names)
    if (not names or len(names) != len(set(names))
            or any(Path(n).name != n or n in ("", ".", "..") for n in names)):
        raise ValueError("unique checkpoint directory names required")
    return names


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(file, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def open_run(stage, config, run_dir):
    prepare_output_root(runs_root())
    root = prepare_output_root(run_dir) if run_dir else create_run(stage, config)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("resume configuration differs")
    write_json(path, config)
    return root


def cluster_mean(values, groups, seed=0, samples=2000):
    values, groups = np.asarray(values, float), np.asarray(groups)
    valid = np.isfinite(values)
    if not valid.any():
        return {"mean": None, "ci": None, "bases": 0, "rows": 0}
    result = paired_cluster_bootstrap(values[valid], np.zeros(valid.sum()), groups[valid], seed, samples)
    result["mean"] = result.pop("difference")
    result["rows"] = int(valid.sum())
    result["undefined_rows"] = int((~valid).sum())
    return result


def contrasts():
    # Canonical export order is D control/hard, G hard/control. Names are data;
    # in particular this also follows a concurrent on_path -> near_path change.
    from .conditions import NAMES
    return {"D": (NAMES[1], NAMES[0]), "G": (NAMES[2], NAMES[3])}
