"""Episode-held-out, nested grouped ridge readouts of object position."""

from pathlib import Path
import hashlib
import json
import shutil
import time

import cv2
import h5py
import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import dgemm
from threadpoolctl import threadpool_limits
import torch

from ..lewm import encode
from ..paths import checkpoint_dir, dataset_path
from .analysis import checkpoint_names, cluster_mean, fingerprint, open_run, write_json, write_npz
from .conditions import NAMES, load_bases, make_conditions
from .rollout_eval import Planner
from ..envs.pusht_peg import PushTPeg

DEFAULT_CHECKPOINTS = ("lewm-pusht", "ft_block_s0", "ft_block_s1", "ft_mixed_s0", "ft_mixed_s1")
SPLIT_CHECKPOINTS = DEFAULT_CHECKPOINTS[1:]


def grouped_folds(groups, folds=5, seed=0):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if folds < 2 or len(unique) < folds:
        raise ValueError("at least folds >= 2 distinct episodes required")
    shuffled = np.random.default_rng(seed).permutation(unique)
    for held in np.array_split(shuffled, folds):
        test = np.flatnonzero(np.isin(groups, held))
        train = np.flatnonzero(~np.isin(groups, held))
        yield train, test


def _ridge_path(x, y, test, alphas):
    """Train-only standardization/intercept; one eigensolve for all penalties."""
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    x, test = (x - mean) / scale, (test - mean) / scale
    ym = y.mean(0)
    y = y - ym
    if len(x) < x.shape[1]:
        eigen, vectors = eigh(dgemm(1, x, x, trans_b=True), check_finite=False)
        left = dgemm(1, dgemm(1, test, x, trans_b=True), vectors)
        right = dgemm(1, vectors, y, trans_a=True)
    else:
        eigen, vectors = eigh(dgemm(1, x, x, trans_a=True), check_finite=False)
        left = dgemm(1, test, vectors)
        right = dgemm(1, vectors, dgemm(1, x, y, trans_a=True), trans_a=True)
    eigen = np.maximum(eigen, 0)
    predictions = np.stack([dgemm(1, left / (eigen + a), right) + ym for a in alphas])
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite ridge predictions")
    return predictions


def nested_readout(features, targets, groups, seed=0, outer_folds=5, inner_folds=3,
                   alphas=(0.01, 0.1, 1, 10, 100, 1000), samples=2000):
    """OOF Euclidean pixel error for peg xy (first pair), T xy (second)."""
    x, y, groups = np.asarray(features, float), np.asarray(targets, float), np.asarray(groups)
    if x.ndim != 2 or y.shape != (len(x), 4) or groups.shape != (len(x),):
        raise ValueError("aligned features, four targets, and episode groups required")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or any(a <= 0 for a in alphas):
        raise ValueError("finite inputs and positive ridge penalties required")
    prediction, mean_prediction = np.empty_like(y), np.empty_like(y)
    assignments = np.empty(len(y), int)
    selected = []
    with threadpool_limits(limits=2):
        for fold, (train, test) in enumerate(grouped_folds(groups, outer_folds, seed)):
            losses = np.zeros((len(alphas), 2))
            count = 0
            for itrain, ival in grouped_folds(groups[train], inner_folds, seed + fold + 1):
                a, b = train[itrain], train[ival]
                candidates = _ridge_path(x[a], y[a], x[b], alphas)
                error = (candidates - y[b]).reshape(len(alphas), len(b), 2, 2)
                losses += np.linalg.norm(error, axis=-1).sum(axis=1)
                count += len(b)
            indices = np.argmin(losses / count, axis=0)
            candidates = _ridge_path(x[train], y[train], x[test], alphas)
            for target, idx in enumerate(indices):
                prediction[test, 2*target:2*target+2] = candidates[idx, :, 2*target:2*target+2]
            mean_prediction[test] = y[train].mean(0)
            assignments[test] = fold
            selected.append({"fold": fold, "peg_alpha": alphas[indices[0]], "T_alpha": alphas[indices[1]]})
    errors = np.linalg.norm((prediction-y).reshape(-1, 2, 2), axis=-1)
    mean_errors = np.linalg.norm((mean_prediction-y).reshape(-1, 2, 2), axis=-1)
    def summaries(e):
        return {name: cluster_mean(e[:, i], groups, seed, samples) for i, name in enumerate(("peg", "T"))}
    return {"ridge": summaries(errors), "mean": summaries(mean_errors), "alphas": selected,
            "episode_errors": {str(g): {name: {"sum": float(errors[groups==g, i].sum()),
                "n": int((groups==g).sum())} for i, name in enumerate(("peg", "T"))} for g in np.unique(groups)},
            "mean_episode_errors": {str(g): {name: {"sum": float(mean_errors[groups==g, i].sum()),
                "n": int((groups==g).sum())} for i, name in enumerate(("peg", "T"))} for g in np.unique(groups)},
            "predictions": prediction, "errors": errors, "folds": assignments}


def pixel_features(pixels):
    """32x32 RGB baseline, unnormalized [0,1] pixels."""
    return np.stack([cv2.resize(p, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1)
                     for p in pixels]).astype(np.float32) / 255


def validation_episodes(split_checkpoints=SPLIT_CHECKPOINTS):
    """Intersect recorded validation sets within each training dataset."""
    sets, provenance = {}, {}
    for name in split_checkpoints:
        path = checkpoint_dir(name) / "trainer_state.pth"
        metadata = torch.load(path, weights_only=False, map_location="cpu")["metadata"]
        dataset = metadata["dataset"]["name"]
        val, train = set(metadata["val_episodes"]), set(metadata["train_episodes"])
        if not val or val & train:
            raise ValueError(f"invalid episode split: {name}")
        sets.setdefault(dataset, []).append(val)
        provenance[name] = {"dataset": metadata["dataset"], "seed": metadata["seed"],
                            "val_episodes": sorted(val), "train_episodes": sorted(train)}
    if len(sets) != 2:
        raise ValueError("both training datasets need recorded validation splits")
    common = {name: sorted(set.intersection(*vals)) for name, vals in sets.items()}
    if any(not v for v in common.values()):
        raise ValueError("no common held-out episodes across checkpoints")
    return common, provenance


def build_pool(bases_path, frames_per_dataset, seed, heldout):
    states, groups, rows = [], [], []
    rng = np.random.default_rng(seed)
    counts = {}
    for name, episodes in sorted(heldout.items()):
        with h5py.File(dataset_path(name), "r") as file:
            available = np.flatnonzero(np.isin(file["episode_idx"][:], episodes))
            selected = np.sort(rng.choice(available, min(frames_per_dataset, len(available)), replace=False))
            ids = file["episode_idx"][selected]
            states.extend(file["state"][selected])
            groups.extend(f"{name}:{int(ep)}" for ep in ids)
            rows.extend({"dataset": name, "row": int(row)} for row in selected)
            counts[name] = {"available": len(available), "selected": len(selected), "episodes": episodes}
    for base in load_bases(bases_path):
        for name, condition in make_conditions(base, render_images=False).items():
            if name not in NAMES:
                continue
            bodies = condition.start_snapshot["bodies"]
            start = [*bodies["agent"]["position"], *bodies["block"]["position"],
                     bodies["block"]["angle"], *bodies["agent"]["velocity"], *bodies["peg"]["position"]]
            for endpoint, state in (("start", start), ("goal", condition.goal_state)):
                states.append(state)
                # All matched condition endpoints from one base stay together.
                groups.append(f"condition:{base.id}")
                rows.append({"base_id": base.id, "condition": name, "endpoint": endpoint, "peg_radius": base.peg_radius})
    return np.asarray(states), np.asarray(groups), {"rows": rows, "datasets": counts}


def render_pool(states, manifest, with_target):
    """Use original dataset pixels without decoration; re-render the other arm."""
    env = PushTPeg(resolution=224, with_target=with_target, render_target_pose=(256,256,np.pi/4))
    files = {}
    try:
        env.reset(seed=0)
        pixels = []
        for state, row in zip(states, manifest["rows"]):
            if not with_target and "dataset" in row:
                name = row["dataset"]
                if name not in files:
                    files[name] = h5py.File(dataset_path(name), "r")
                pixels.append(files[name]["pixels"][row["row"]])
            else:
                radius = row.get("peg_radius", 15)
                if env.peg_radius != radius:
                    env.close()
                    env = PushTPeg(resolution=224, with_target=with_target,
                                   render_target_pose=(256,256,np.pi/4), peg_radius=radius)
                    env.reset(seed=0)
                env._set_state(state)
                pixels.append(env._render_frame("rgb_array").copy())
        return np.stack(pixels)
    finally:
        env.close()
        for file in files.values():
            file.close()


def encode_pool(planner, pixels, batch_size=64):
    cls, projected = [], []
    with torch.inference_mode():
        for offset in range(0, len(pixels), batch_size):
            tensor = torch.stack([planner.transform(torch.from_numpy(p.copy()).permute(2,0,1))
                                  for p in pixels[offset:offset+batch_size]]).to(planner.device)
            c, p = encode(planner.model, tensor)
            cls.append(c.cpu().numpy()); projected.append(p.cpu().numpy())
    return {"cls": np.concatenate(cls), "projected": np.concatenate(projected)}


def run_readout(bases_path, checkpoints=DEFAULT_CHECKPOINTS, frames_per_dataset=4000,
                seed=42, run_dir=None, device="auto", batch_size=64, outer_folds=5,
                inner_folds=3, samples=2000, split_checkpoints=SPLIT_CHECKPOINTS):
    if frames_per_dataset < 1 or batch_size < 1:
        raise ValueError("positive frame count and batch size required")
    checkpoints = checkpoint_names(checkpoints)
    if not set(checkpoints).difference(("lewm-pusht",)).issubset(split_checkpoints):
        raise ValueError("all fine-tuned readout checkpoints need their held-out split in split_checkpoints")
    heldout, provenance = validation_episodes(split_checkpoints)
    bases = load_bases(bases_path)
    if len({b.peg_radius for b in bases}) > 1:
        raise ValueError("one peg radius required per readout run")
    dataset_hashes = {name: fingerprint(dataset_path(name)) for name in heldout}
    if any(dataset_hashes[p["dataset"]["name"]] != p["dataset"]["sha256"] for p in provenance.values()):
        raise ValueError("training dataset content differs from recorded split metadata")
    config = {"stage": "probe-readout", "seed": seed, "checkpoints": list(checkpoints),
              "checkpoint_sha256": {n: fingerprint(checkpoint_dir(n)/"weights.pt") for n in checkpoints},
              "bases_sha256": fingerprint(bases_path), "conditions": list(NAMES), "peg_radius": bases[0].peg_radius if bases else 15,
              "frames_per_dataset": frames_per_dataset, "splits": provenance,
              "dataset_sha256": dataset_hashes,
              "device": device, "batch_size": batch_size, "outer_folds": outer_folds,
              "inner_folds": inner_folds, "bootstrap_samples": samples}
    root = open_run("probe-readout", config, run_dir)
    start = time.perf_counter()
    if not (root / "pool.npz").exists():
        states, groups, manifest = build_pool(bases_path, frames_per_dataset, seed, heldout)
        write_json(root / "pool.json", manifest)
        write_npz(root / "pool.npz", states=states, groups=groups)
    with np.load(root / "pool.npz") as pool:
        states, groups = pool["states"], pool["groups"]
    manifest = json.loads((root / "pool.json").read_text())
    if len(states) != len(groups) or len(states) != len(manifest["rows"]):
        raise ValueError("incomplete persisted evaluation pool")
    targets = states[:, [7, 8, 2, 3]]
    for name in checkpoints:
        output = root / f"{name}.json"
        if output.exists():
            continue
        arm_start = time.perf_counter()
        planner = Planner(name, device=device)
        try:
            render_path = root / f"pixels_{int(planner.with_target)}.npz"
            if not render_path.exists():
                write_npz(render_path, pixels=render_pool(states, manifest, planner.with_target))
            with np.load(render_path) as images:
                pixels = images["pixels"]
            feature_path = root / f"{name}_features.npz"
            if not feature_path.exists():
                write_npz(feature_path, **encode_pool(planner, pixels, batch_size))
            result = {"checkpoint": name, "with_target": planner.with_target, "features": {}}
            pixel_result = root / f"pixel_{int(planner.with_target)}.json"
            for kind in ("cls", "projected", "pixel"):
                if kind == "pixel" and pixel_result.exists():
                    result["features"][kind] = json.loads(pixel_result.read_text())
                    continue
                if kind == "pixel":
                    features = pixel_features(pixels)
                else:
                    with np.load(feature_path) as cache:
                        features = cache[kind]
                metrics = nested_readout(features, targets, groups, seed, outer_folds, inner_folds, samples=samples)
                write_npz(root / f"{name}_{kind}_oof.npz", groups=groups, targets=targets,
                    **{k: metrics.pop(k) for k in ("predictions", "errors", "folds")})
                result["features"][kind] = metrics
                if kind == "pixel":
                    write_json(pixel_result, metrics)
            result["wall_s"] = time.perf_counter()-arm_start
            write_json(output, result)
        finally:
            planner.close()
    result = {"stage": "probe-readout", "run_dir": str(root), "seed": seed,
              "pool_identity": hashlib.sha256(states.tobytes()+json.dumps(groups.tolist()).encode()).hexdigest(),
              "pool": {"frames": len(states), "episodes": len(np.unique(groups)), "datasets": manifest["datasets"]},
              "checkpoints": {n: json.loads((root/f"{n}.json").read_text()) for n in checkpoints},
              "invocation_wall_s": time.perf_counter()-start}
    write_json(root / "readout.json", result)
    return result


def run_pool_readout(source_readout, checkpoints, run_dir, device="auto", seed=42):
    """Read out new checkpoints on an unchanged, persisted evaluation pool."""
    source = Path(source_readout).expanduser().resolve()
    destination = Path(run_dir).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError("output must be separate from the source readout")
    checkpoints = checkpoint_names(checkpoints)
    source_config = json.loads((source / "config.json").read_text())
    if seed != source_config["seed"]:
        raise ValueError("seed must match the source readout folds/bootstrap")
    with np.load(source / "pool.npz") as pool:
        states, groups = pool["states"], pool["groups"]
    manifest = json.loads((source / "pool.json").read_text())
    if len(states) != len(groups) or len(states) != len(manifest["rows"]):
        raise ValueError("incomplete persisted evaluation pool")
    identity = hashlib.sha256(states.tobytes()+json.dumps(groups.tolist()).encode()).hexdigest()
    source_result = json.loads((source / "readout.json").read_text())
    expected = source_config.get("pool_identity", source_result["pool_identity"])
    if identity != expected or identity != source_result["pool_identity"]:
        raise ValueError("source evaluation pool identity differs")
    provenance = {}
    for name in checkpoints:
        state_path = checkpoint_dir(name) / "trainer_state.pth"
        if not state_path.exists() and name == "lewm-pusht":
            continue
        metadata = torch.load(state_path, map_location="cpu", weights_only=False)["metadata"]
        dataset = metadata["dataset"]["name"]
        episodes = {int(str(g).rsplit(":", 1)[1]) for g in groups
                    if str(g).rsplit(":", 1)[0] == dataset}
        if episodes & set(metadata["train_episodes"]):
            raise ValueError(f"evaluation pool overlaps training episodes: {name}")
        if source_config["dataset_sha256"].get(dataset) != metadata["dataset"]["sha256"]:
            raise ValueError(f"training dataset identity differs from source pool: {name}")
        provenance[name] = metadata
    config = {**source_config, "stage": "probe-pool-readout", "source_readout": str(source),
              "source_config_sha256": fingerprint(source / "config.json"),
              "source_pool_sha256": {f: fingerprint(source / f) for f in ("pool.npz", "pool.json")},
              "pool_identity": identity, "checkpoints": checkpoints, "splits": provenance,
              "checkpoint_sha256": {n: fingerprint(checkpoint_dir(n)/"weights.pt") for n in checkpoints},
              "device": device}
    root = open_run("probe-pool-readout", config, destination)
    def copy_input(filename):
        target = root / filename
        if target.exists():
            if fingerprint(target) != fingerprint(source / filename):
                raise ValueError(f"persisted input differs: {filename}")
        else:
            shutil.copyfile(source / filename, target)
    copy_input("pool.npz")
    copy_input("pool.json")
    targets = states[:, [7, 8, 2, 3]]
    start = time.perf_counter()
    for name in checkpoints:
        planner = Planner(name, device=device)
        try:
            filename = f"pixels_{int(planner.with_target)}.npz"
            copy_input(filename)
            output = root / f"{name}.json"
            if output.exists():
                continue
            arm_start = time.perf_counter()
            with np.load(root / filename) as images:
                pixels = images["pixels"]
            if len(pixels) != len(states):
                raise ValueError("persisted pixels and states are misaligned")
            features = encode_pool(planner, pixels, source_config["batch_size"])
            write_npz(root / f"{name}_features.npz", **features)
            result = {"checkpoint": name, "with_target": planner.with_target, "features": {}}
            for kind in ("cls", "projected"):
                metrics = nested_readout(features[kind], targets, groups, seed,
                    source_config["outer_folds"], source_config["inner_folds"],
                    samples=source_config["bootstrap_samples"])
                write_npz(root / f"{name}_{kind}_oof.npz", groups=groups, targets=targets,
                    **{k: metrics.pop(k) for k in ("predictions", "errors", "folds")})
                result["features"][kind] = metrics
            # The rendering-matched pixel baseline is already measured on this pool.
            pixel_path = source / f"pixel_{int(planner.with_target)}.json"
            if pixel_path.exists():
                result["features"]["pixel"] = json.loads(pixel_path.read_text())
            result["wall_s"] = time.perf_counter()-arm_start
            write_json(output, result)
        finally:
            planner.close()
    result = {"stage": "probe-pool-readout", "run_dir": str(root), "seed": seed,
              "pool_identity": identity,
              "pool": {"frames": len(states), "episodes": len(np.unique(groups)), "datasets": manifest["datasets"]},
              "checkpoints": {n: json.loads((root/f"{n}.json").read_text()) for n in checkpoints},
              "invocation_wall_s": time.perf_counter()-start}
    write_json(root / "readout.json", result)
    return result
