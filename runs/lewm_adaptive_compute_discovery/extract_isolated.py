#!/usr/bin/env python3
"""Extract physically isolated train or one-shot calibration latent caches."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
import data_isolation

SOURCE = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")
BASE_CONFIG = REPO / "runs/lewm_transfer/cube/cache/model/config.json"
BASE_WEIGHTS = REPO / "runs/lewm_transfer/cube/cache/model/weights.pt"


def load_model_io():
    path = REPO / "runs/lewm_adaptive_compute_v2/model_io.py"
    spec = importlib.util.spec_from_file_location("discovery_model_io", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load V2 model I/O")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_isolated(path: Path, role: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    audit = data_isolation.assert_isolated_cache(arrays, role)
    return arrays, audit


def extract(role: str, device_name: str) -> dict[str, object]:
    cfg = json.loads((ROOT / "config.json").read_text())
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    cache_name = "v3_train_only.npz" if role == "train" else "v3_calibration_only.npz"
    manifest_name = "v3_train_only_manifest.json" if role == "train" else "v3_calibration_only_manifest.json"
    cache = ROOT / "cache" / cache_name
    manifest = ROOT / "cache" / manifest_name
    if cache.exists() or manifest.exists():
        if not (cache.exists() and manifest.exists()):
            raise RuntimeError("partial isolated cache artifact")
        arrays, audit = load_isolated(cache, role)
        meta = json.loads(manifest.read_text())
        if meta["cache"]["sha256"] != data_isolation.sha256_file(cache):
            raise RuntimeError("isolated cache hash mismatch")
        return {"cache": str(cache), "manifest": str(manifest), "reused": True, "audit": audit}

    sets = data_isolation.load_pinned_v3_episode_sets()
    selected = data_isolation.extraction_splits(role)
    forbidden = np.concatenate((sets["calibration"], sets["test"])) if role == "train" else np.concatenate((sets["train"], sets["test"]))
    model_io = load_model_io()
    model_io.extract_fresh_cube_cache(
        source_h5=SOURCE,
        config_path=BASE_CONFIG,
        weights_path=BASE_WEIGHTS,
        output_npz=cache,
        split_manifest_path=manifest,
        split_episodes=selected,
        selection_seed=int(cfg["selection_seed"]),
        excluded_episode_ordinals=forbidden,
        reserved_episode_ordinals=(),
        prior_manifest_provenance={
            "v3_split_manifest": data_isolation.PINNED["manifest_sha256"],
            "strict_role": role,
            "v3_test_targets": "forbidden",
        },
        device=device,
        encode_batch_size=int(cfg["encode_batch_size"]),
        predict_batch_size=int(cfg["predict_batch_size"]),
    )
    arrays, audit = load_isolated(cache, role)
    return {"cache": str(cache), "manifest": str(manifest), "reused": False, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("train", "calibration_once"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.role == "calibration_once":
        raise RuntimeError("calibration extraction is sealed inside run_discovery.py judge")
    data_isolation.assert_combined_cache_never_opened()
    print(json.dumps(extract(args.role, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
