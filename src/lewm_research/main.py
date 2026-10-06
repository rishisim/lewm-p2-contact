"""Small persistence/preflight helpers for the frozen main-run shell driver."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .paths import checkpoint_dir
from .probe.analysis import fingerprint, open_run, write_json
from .probe.conditions import generate_bases, load_bases
from .probe.readout import DEFAULT_CHECKPOINTS

FROZEN_NORMALIZATION_SHA256 = "839239731fb7c7734fa66057cce9408af8efa043d9045fee5c1c980f058c41f3"


def check_normalization(path):
    """Ensure evaluation, readout, and ABC all use the same frozen statistics."""
    def digest(p):
        return hashlib.sha256(json.dumps(json.loads(Path(p).read_text()), sort_keys=True).encode()).hexdigest()
    if digest(path) != FROZEN_NORMALIZATION_SHA256:
        raise ValueError("normalization differs from frozen W4d protocol")
    for name in DEFAULT_CHECKPOINTS:
        if digest(checkpoint_dir(name) / "normalization.json") != FROZEN_NORMALIZATION_SHA256:
            raise ValueError(f"{name} normalization differs from frozen W4d protocol")
        if not (checkpoint_dir(name) / "weights.pt").is_file():
            raise FileNotFoundError(checkpoint_dir(name) / "weights.pt")


def prepare_bases(n, seed, near_path_distance, run_dir):
    config = {"stage": "probe-bases", "n": n, "seed": seed,
              "near_path_distance": near_path_distance, "construction": "W4d"}
    root = open_run("probe-bases", config, run_dir)
    path = root / "bases.json"
    manifest = root / "manifest.json"
    if path.exists():
        bases = load_bases(path)
        if len(bases) != n or any(not b.id.startswith(f"{seed}:") for b in bases):
            raise ValueError("persisted bases differ from requested construction")
        if manifest.exists() and json.loads(manifest.read_text())["sha256"] != fingerprint(path):
            raise ValueError("persisted bases checksum differs")
    else:
        bases = generate_bases(n, seed, near_path_distance=near_path_distance)
        write_json(path, [asdict(b) for b in bases])
    write_json(manifest, {"sha256": fingerprint(path), "base_ids": [b.id for b in bases]})
    return path
