"""Frozen Phase A: paired test-time peg-size readouts."""

import hashlib
import json
import time

import h5py
import numpy as np

from ..envs.pusht_peg import PushTPeg
from ..paths import checkpoint_dir, dataset_path, runs_root
from .analysis import checkpoint_names, fingerprint, open_run, write_json, write_npz
from .readout import DEFAULT_CHECKPOINTS, Planner, encode_pool, nested_readout, pixel_features, validation_episodes
from .report import _readout_summary

RADII = (15, 30, 45)


def resample_pegs(states, seed=42):
    states = np.asarray(states, dtype=float).copy()
    rng = np.random.default_rng([seed, 45])
    env = PushTPeg(peg_radius=45, with_target=False)
    try:
        env.reset(seed=0)
        for state in states:
            env._set_state(state)
            for _ in range(10000):
                xy = rng.uniform(75, 437, size=2)
                if not env.peg_overlaps(xy):
                    state[7:9] = xy
                    break
            else:
                raise RuntimeError("Could not place radius-45 peg")
    finally:
        env.close()
    return states


def build_pool(heldout, frames_per_dataset=4000, seed=42):
    states, groups, rows, counts = [], [], [], {}
    rng = np.random.default_rng(seed)
    for name, episodes in sorted(heldout.items()):
        with h5py.File(dataset_path(name), "r") as file:
            available = np.flatnonzero(np.isin(file["episode_idx"][:], episodes))
            selected = np.sort(rng.choice(available, min(frames_per_dataset, len(available)), replace=False))
            ids = file["episode_idx"][selected]
            states.extend(file["state"][selected])
            groups.extend(f"{name}:{int(ep)}" for ep in ids)
            rows.extend({"dataset": name, "row": int(row)} for row in selected)
            counts[name] = {"available": len(available), "selected": len(selected), "episodes": episodes}
    return resample_pegs(states, seed), np.asarray(groups), {"rows": rows, "datasets": counts}


def render_pool(states, radius, with_target):
    env = PushTPeg(resolution=224, with_target=with_target,
                   render_target_pose=(256,256,np.pi/4), peg_radius=radius)
    try:
        env.reset(seed=0)
        pixels = []
        for state in states:
            env._set_state(state)
            pixels.append(env._render_frame("rgb_array").copy())
        return np.stack(pixels)
    finally:
        env.close()


def _peg_bootstrap(episodes, seed, samples):
    ids = sorted(episodes)
    sums = np.array([episodes[e]["peg"]["sum"] for e in ids])
    counts = np.array([episodes[e]["peg"]["n"] for e in ids])
    draws = np.random.default_rng(seed).integers(len(ids), size=(samples,len(ids)))
    return float(sums.sum()/counts.sum()), sums[draws].sum(1)/counts[draws].sum(1)


def _estimate(mean, bootstrap):
    return {"mean": mean, "ci": np.quantile(bootstrap,[.025,.975]).tolist() if len(bootstrap) else None}


def summarize(readouts, seed=42, samples=2000):
    result = {}
    for name, radii in readouts.items():
        result[name] = {}
        for radius, arm in radii.items():
            summary = _readout_summary([arm], seed, samples)
            cells = {}
            for kind in ("cls", "projected"):
                metrics = arm["features"][kind]
                peg, boot = _peg_bootstrap(metrics["episode_errors"], seed, samples)
                mean, meanboot = _peg_bootstrap(metrics["mean_episode_errors"], seed, samples)
                valid = meanboot > 0
                cells[kind] = {**summary[kind], "pixel_peg": summary["pixel"]["peg"],
                               "mean_peg": summary["mean"]["peg"],
                               "peg_over_mean": _estimate(peg/mean if mean > 0 else None,
                                                          boot[valid]/meanboot[valid])}
                if radius == "45" and "15" in radii:
                    small = radii["15"]["features"][kind]["episode_errors"]
                    if sorted(small) != sorted(metrics["episode_errors"]) or any(
                            small[e]["peg"]["n"] != metrics["episode_errors"][e]["peg"]["n"] for e in small):
                        raise ValueError("radius comparison requires identical episode groups and counts")
                    smallpeg, smallboot = _peg_bootstrap(small, seed, samples)
                    cells[kind]["peg_reduction_15_to_45"] = _estimate(smallpeg-peg, smallboot-boot)
            result[name][radius] = cells
    return result


def apply_reading(checkpoints):
    cells = []
    for name in DEFAULT_CHECKPOINTS[1:]:
        for kind in ("cls", "projected"):
            radii = checkpoints.get(name, {})
            large, small = radii.get("45", {}).get(kind), radii.get("15", {}).get(kind)
            flags, label, sanity = [], "unavailable", None
            if small is not None:
                sanity = small["peg_over_mean"]["mean"] > .8
                ci = small["peg_over_mean"]["ci"]
                if ci is not None and ci[0] <= .8 <= ci[1]:
                    flags.append("radius-15 peg/mean straddles 0.8")
            if large is not None:
                for key, threshold in (("peg_over_mean",.8),("peg_over_T",2),("peg_over_pixel",1.5),
                                       ("peg_reduction_15_to_45",0)):
                    ci = large.get(key, {}).get("ci")
                    if ci is not None and ci[0] <= threshold <= ci[1]:
                        flags.append(f"{key} straddles {threshold:g}")
                peg = large["peg"]["mean"]
                reduction = large.get("peg_reduction_15_to_45")
                if peg > .8*large["mean_peg"]["mean"]:
                    label = "size-unexplained"
                elif reduction is not None:
                    label = ("size-explained" if peg <= 2*large["T"]["mean"]
                             and peg <= 1.5*large["pixel_peg"]["mean"] and reduction["ci"][0] > 0
                             else "unlabeled")
            cells.append({"checkpoint": name, "kind": kind, "label": label,
                          "radius_15_deficit": sanity, "flags": flags})
    labels = [c["label"] for c in cells]
    overall = ("size substantially explains the readout deficit" if all(l=="size-explained" for l in labels)
               else "size alone does not explain it" if all(l=="size-unexplained" for l in labels)
               else "partial/ambiguous")
    return {"cells": cells, "overall": overall, "complete": "unavailable" not in labels and all(c["radius_15_deficit"] is not None for c in cells),
            "radius_15_sanity_passed": all(c["radius_15_deficit"] is True for c in cells),
            "pool_only": any(c["radius_15_deficit"] is False for c in cells)}


def markdown_summary(result):
    def value(metric):
        if metric["mean"] is None:
            return "undefined"
        lo, hi = metric["ci"]
        return f'{metric["mean"]:.2f} [{lo:.2f}, {hi:.2f}]'
    lines = ["# Size control: Phase A", "", "Estimates [episode-bootstrap 95% CI].", "",
             "| Checkpoint | Radius | Kind | Peg | T | Pixel peg | Mean peg | Peg/T | Peg/pixel | Peg/mean | 15→45 reduction |",
             "|---|---:|---|---|---|---|---|---|---|---|---|"]
    for name, radii in result["checkpoints"].items():
        for radius, kinds in sorted(radii.items(), key=lambda item:int(item[0])):
            for kind, cell in kinds.items():
                metrics = [value(cell[k]) for k in ("peg","T","pixel_peg","mean_peg","peg_over_T",
                                                    "peg_over_pixel","peg_over_mean")]
                reduction = cell.get("peg_reduction_15_to_45")
                lines.append(f'| {name} | {radius} | {kind} | '+" | ".join(metrics)+
                             f' | {value(reduction) if reduction else "—"} |')
    reading = result["reading"]
    lines += ["", "| Checkpoint | Kind | Label | Radius-15 deficit | Threshold flags |",
              "|---|---|---|---|---|"]
    for cell in reading["cells"]:
        lines.append(f'| {cell["checkpoint"]} | {cell["kind"]} | {cell["label"]} | '
                     f'{cell["radius_15_deficit"]} | {"; ".join(cell["flags"]) or "—"} |')
    lines += ["", f'Overall reading: **{reading["overall"]}**.',
              f'Complete: {reading["complete"]}. Radius-15 sanity passed: {reading["radius_15_sanity_passed"]}.']
    if reading["pool_only"]:
        lines += ["Radius-15 sanity failed: the result concerns this pool only."]
    lines += ["A negative result bounds test-time enlargement of frozen-encoder readout; it cannot exclude a training-time size effect and says nothing directly about planning."]
    return "\n".join(lines)+"\n"


def run_size_control(checkpoints=DEFAULT_CHECKPOINTS, radii=RADII, frames_per_dataset=4000,
                     seed=42, run_dir=None, device="auto"):
    checkpoints = checkpoint_names(checkpoints)
    radii = tuple(radii)
    if not set(checkpoints).issubset(DEFAULT_CHECKPOINTS):
        raise ValueError("Phase A checkpoints required")
    if not radii or len(set(radii)) != len(radii) or not set(radii).issubset(RADII):
        raise ValueError("unique Phase A radii (15, 30, 45) required")
    if frames_per_dataset < 1 or seed < 0:
        raise ValueError("positive frame count and nonnegative seed required")
    heldout, provenance = validation_episodes()
    hashes = {name: fingerprint(dataset_path(name)) for name in heldout}
    if any(hashes[p["dataset"]["name"]] != p["dataset"]["sha256"] for p in provenance.values()):
        raise ValueError("training dataset content differs from recorded split metadata")
    config = {"stage": "probe-size-control", "seed": seed, "checkpoints": checkpoints,
              "radii": list(radii), "frames_per_dataset": frames_per_dataset, "splits": provenance,
              "dataset_sha256": hashes, "device": device, "batch_size": 64,
              "outer_folds": 5, "inner_folds": 3, "bootstrap_samples": 2000,
              "checkpoint_sha256": {n: fingerprint(checkpoint_dir(n)/"weights.pt") for n in checkpoints}}
    root = open_run("probe-size-control", config, run_dir or runs_root()/"size-control/phaseA")
    start = time.perf_counter()
    if not (root/"pool.npz").exists():
        states, groups, manifest = build_pool(heldout, frames_per_dataset, seed)
        write_json(root/"pool.json", manifest)
        write_npz(root/"pool.npz", states=states, groups=groups)
    with np.load(root/"pool.npz") as pool:
        states, groups = pool["states"], pool["groups"]
    manifest = json.loads((root/"pool.json").read_text())
    if len(states) != len(groups) or len(states) != len(manifest["rows"]):
        raise ValueError("incomplete persisted evaluation pool")
    targets = states[:, [7,8,2,3]]
    readouts = {}
    for name in checkpoints:
        readouts[name] = {}
        pending = [r for r in radii if not (root/f"{name}_r{r}.json").exists()]
        planner = Planner(name, device=device) if pending else None
        try:
            for radius in radii:
                output = root/f"{name}_r{radius}.json"
                if not output.exists():
                    arm = {"checkpoint": name, "radius": radius, "with_target": planner.with_target, "features": {}}
                    pixels_path = root/f"pixels_r{radius}_{int(planner.with_target)}.npz"
                    if not pixels_path.exists():
                        write_npz(pixels_path, pixels=render_pool(states, radius, planner.with_target))
                    with np.load(pixels_path) as cache:
                        pixels = cache["pixels"]
                    features_path = root/f"{name}_r{radius}_features.npz"
                    if not features_path.exists():
                        write_npz(features_path, **encode_pool(planner, pixels))
                    for kind in ("cls", "projected", "pixel"):
                        stem = (f"pixel_r{radius}_{int(planner.with_target)}" if kind=="pixel"
                                else f"{name}_r{radius}_{kind}")
                        path = root/f"{stem}.json"
                        if not path.exists():
                            if kind == "pixel":
                                features = pixel_features(pixels)
                            else:
                                with np.load(features_path) as cache:
                                    features = cache[kind]
                            metrics = nested_readout(features, targets, groups, seed=seed, samples=2000)
                            write_npz(root/f"{stem}_oof.npz", groups=groups, targets=targets,
                                      **{k: metrics.pop(k) for k in ("predictions","errors","folds")})
                            write_json(path, metrics)
                        arm["features"][kind] = json.loads(path.read_text())
                    write_json(output, arm)
                readouts[name][str(radius)] = json.loads(output.read_text())
        finally:
            if planner is not None:
                planner.close()
    summary = summarize(readouts, seed)
    result = {"stage": "probe-size-control", "run_dir": str(root), "seed": seed,
              "pool_identity": hashlib.sha256(states.tobytes()+json.dumps(groups.tolist()).encode()).hexdigest(),
              "pool": {"frames": len(states), "episodes": len(np.unique(groups)), "datasets": manifest["datasets"]},
              "checkpoints": summary, "reading": apply_reading(summary),
              "invocation_wall_s": time.perf_counter()-start}
    write_json(root/"summary.json", result)
    (root/"summary.md").write_text(markdown_summary(result))
    return result
