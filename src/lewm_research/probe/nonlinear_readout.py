"""Cached, CPU-only nonlinear position readouts for the frozen pin check."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from ..paths import runs_root
from .analysis import cluster_mean, fingerprint, write_json, write_npz
from .readout import DEFAULT_CHECKPOINTS, grouped_folds, pixel_features
from .rollout_eval import prepare_output_root

PHASE_B = ("ft45_block_s0", "ft45_mixed_s0")
TARGETS = ("peg", "T")


def inner_split(groups, val_frac=0.2, seed=0):
    """Return row indices for disjoint inner-train/validation episode sets."""
    groups = np.asarray(groups)
    episodes = np.unique(groups)
    if groups.ndim != 1 or len(episodes) < 2 or not 0 < val_frac < 1:
        raise ValueError("at least two episodes and 0 < val_frac < 1 required")
    nval = min(len(episodes)-1, max(1, round(len(episodes)*val_frac)))
    held = np.random.default_rng(seed).permutation(episodes)[:nval]
    return np.flatnonzero(~np.isin(groups, held)), np.flatnonzero(np.isin(groups, held))


def _inputs(features, targets, groups):
    x, y, groups = np.asarray(features, float), np.asarray(targets, float), np.asarray(groups)
    if x.ndim != 2 or x.shape[1] < 1 or y.shape != (len(x), 4) or groups.shape != (len(x),):
        raise ValueError("aligned features, four targets, and episode groups required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite inputs required")
    return x, y, groups


def _scale(x):
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1
    return mean, scale


def _result(kind, prediction, mean_prediction, y, groups, folds, seed):
    errors = np.linalg.norm((prediction-y).reshape(-1, 2, 2), axis=-1)
    mean_errors = np.linalg.norm((mean_prediction-y).reshape(-1, 2, 2), axis=-1)
    def summaries(e):
        return {name: cluster_mean(e[:, i], groups, seed, 2000) for i, name in enumerate(TARGETS)}
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite predictions")
    return {kind: summaries(errors), "mean": summaries(mean_errors),
            "predictions": prediction, "errors": errors, "folds": folds}


def mlp_readout(features, targets, groups, seed=0, outer_folds=5, hidden=(256,256),
                lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=200,
                patience=20, inner_val_frac=0.2, threads=2):
    """One early-stopped CPU network per target pair; no outer-train refit.

    Scalers are fitted on inner-train only, keeping validation and outer-test
    observations out of preprocessing as well as optimizer updates.
    """
    x, y, groups = _inputs(features, targets, groups)
    if (not hidden or any(h < 1 for h in hidden) or lr <= 0 or weight_decay < 0
            or min(batch_size, max_epochs, patience, threads) < 1):
        raise ValueError("positive network/training settings required")
    torch.set_num_threads(threads)
    prediction, mean_prediction = np.empty_like(y), np.empty_like(y)
    assignments, selected = np.empty(len(y), int), []
    with threadpool_limits(limits=threads):
        for fold, (train, test) in enumerate(grouped_folds(groups, outer_folds, seed)):
            itrain, ival = inner_split(groups[train], inner_val_frac, seed+fold+1)
            a, b = train[itrain], train[ival]
            xm, xs = _scale(x[a])
            xt = torch.tensor((x[a]-xm)/xs, dtype=torch.float32, device="cpu")
            xv = torch.tensor((x[b]-xm)/xs, dtype=torch.float32, device="cpu")
            xe = torch.tensor((x[test]-xm)/xs, dtype=torch.float32, device="cpu")
            record = {"fold": fold}
            torch.manual_seed(seed+fold)
            for i, name in enumerate(TARGETS):
                columns = slice(2*i, 2*i+2)
                ym, ys = _scale(y[a, columns])
                yt = torch.tensor((y[a, columns]-ym)/ys, dtype=torch.float32, device="cpu")
                yv = torch.tensor((y[b, columns]-ym)/ys, dtype=torch.float32, device="cpu")
                layers, width = [], x.shape[1]
                for h in hidden:
                    layers.extend((nn.Linear(width, h, device="cpu"), nn.ReLU()))
                    width = h
                model = nn.Sequential(*layers, nn.Linear(width, 2, device="cpu"))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                best, stale, weights = float("inf"), 0, None
                for epoch in range(max_epochs):
                    model.train()
                    order = torch.randperm(len(a), device="cpu")
                    for offset in range(0, len(a), batch_size):
                        rows = order[offset:offset+batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        loss = (model(xt[rows])-yt[rows]).square().mean()
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        val = (model(xv)-yv).square().mean().item()
                    if not np.isfinite(val):
                        raise ValueError("nonfinite validation loss")
                    if val < best:
                        best, stale = val, 0
                        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
                        record[f"{name}_best_epoch"] = epoch+1
                    else:
                        stale += 1
                    if stale >= patience:
                        break
                model.load_state_dict(weights)
                with torch.no_grad():
                    prediction[test, columns] = model(xe).numpy()*ys+ym
                record[f"{name}_epochs"] = epoch+1
                record[f"{name}_val_mse"] = best
            selected.append(record)
            mean_prediction[test] = y[train].mean(0)
            assignments[test] = fold
    return {**_result("mlp", prediction, mean_prediction, y, groups, assignments, seed),
            "early_stopping": selected}


def knn_readout(features, targets, groups, seed=0, outer_folds=5, k=10, threads=2):
    """Uniform Euclidean neighbors on outer-train-standardized features."""
    x, y, groups = _inputs(features, targets, groups)
    if k < 1 or threads < 1:
        raise ValueError("positive k and threads required")
    prediction, mean_prediction = np.empty_like(y), np.empty_like(y)
    assignments = np.empty(len(y), int)
    with threadpool_limits(limits=threads):
        for fold, (train, test) in enumerate(grouped_folds(groups, outer_folds, seed)):
            if k > len(train):
                raise ValueError("k exceeds outer-training rows")
            mean, scale = _scale(x[train])
            neighbors = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=1)
            neighbors.fit((x[train]-mean)/scale)
            for offset in range(0, len(test), 256):
                rows = test[offset:offset+256]
                indices = neighbors.kneighbors((x[rows]-mean)/scale, return_distance=False)
                prediction[rows] = y[train][indices].mean(1)
            mean_prediction[test] = y[train].mean(0)
            assignments[test] = fold
    return _result("knn", prediction, mean_prediction, y, groups, assignments, seed)


def pin_label(mlp_peg, mlp_t, pixel_mlp_peg, mean_peg, knn_peg):
    """Apply the pre-registered point-estimate rule, with present precedence."""
    present = mlp_peg <= 2*mlp_t and mlp_peg <= 1.5*pixel_mlp_peg
    absent = mlp_peg > .8*mean_peg and knn_peg > .8*mean_peg
    return {"label": "present" if present else "absent" if absent else "partial",
            "conflict": bool(present and absent)}


def _output_root(out_dir):
    root = prepare_output_root(out_dir or runs_root()/"pin-mlp")
    if not root.is_relative_to((runs_root()/"pin-mlp").resolve()):
        raise ValueError("pin readout outputs must stay under runs/pin-mlp")
    return root


def _pool(source):
    with np.load(source/"pool.npz") as pool:
        states, groups = pool["states"], pool["groups"]
    # size_control resamples peg coordinates once when building this shared
    # pool, then renders these exact states at every radius. Never resample.
    targets = states[:, [7,8,2,3]]
    identity = hashlib.sha256(states.tobytes()+json.dumps(groups.tolist()).encode()).hexdigest()
    return targets, groups, identity


def _check_ridge(source, stem, targets, groups):
    folds = np.empty(len(groups), int)
    for fold, (_, test) in enumerate(grouped_folds(groups, 5, 42)):
        folds[test] = fold
    with np.load(source/f"{stem}_oof.npz") as oof:
        np.testing.assert_array_equal(oof["groups"], groups, err_msg="ridge pool groups differ")
        np.testing.assert_array_equal(oof["targets"], targets, err_msg="ridge pool targets differ")
        np.testing.assert_array_equal(oof["folds"], folds, err_msg="cached ridge folds differ")
    return json.loads((source/f"{stem}.json").read_text())


def _run_unit(job):
    start = time.perf_counter()
    source, root = Path(job["source"]), Path(job["out_dir"])
    targets, groups, _ = _pool(source)
    if job["kind"] == "pixel":
        with np.load(source/job["cache"]) as cache:
            features = pixel_features(cache["pixels"])
    else:
        with np.load(source/job["cache"]) as cache:
            features = cache[job["kind"]]
    result = {**job, "stage": "pin-mlp-unit", "frames": len(groups), "metrics": {}}
    for method, readout in (("mlp", mlp_readout), ("knn", knn_readout)):
        arm_start = time.perf_counter()
        metrics = readout(features, targets, groups, seed=42, threads=job["threads"])
        write_npz(root/f'{job["unit"]}_{method}_oof.npz', groups=groups, targets=targets,
                  **{k: metrics.pop(k) for k in ("predictions", "errors", "folds")})
        metrics["wall_s"] = time.perf_counter()-arm_start
        result["metrics"][method] = metrics
    result["wall_s"] = time.perf_counter()-start
    write_json(root/f'{job["unit"]}.json', result)
    return {"unit": job["unit"], "wall_s": result["wall_s"]}


def run_pin_mlp(out_dir=None, checkpoints=(*DEFAULT_CHECKPOINTS, *PHASE_B),
                radii=(15,45), kinds=("cls", "projected"), workers=4, threads=2,
                phase_a=None, phase_b=None, include_pixels=True):
    """Resume selected cached units; defaults execute the full frozen protocol."""
    root = _output_root(out_dir)
    if (not checkpoints or len(set(checkpoints)) != len(checkpoints)
            or not set(checkpoints).issubset((*DEFAULT_CHECKPOINTS, *PHASE_B))
            or not radii or len(set(radii)) != len(radii) or not set(radii).issubset((15,45))
            or not kinds or len(set(kinds)) != len(kinds) or not set(kinds).issubset(("cls", "projected"))
            or min(workers, threads) < 1):
        raise ValueError("unique frozen checkpoints, radii 15/45, kinds cls/projected and positive workers/threads required")
    sources = {"phaseA": Path(phase_a or runs_root()/"size-control/phaseA").expanduser().resolve(),
               "phaseB": Path(phase_b or runs_root()/"size-control/phaseB/readout").expanduser().resolve()}
    config = {"stage": "probe-pin-mlp", "seed": 42, "outer_folds": 5, "hidden": [256,256],
              "lr": .001, "weight_decay": .0001, "batch_size": 256, "max_epochs": 200,
              "patience": 20, "inner_val_frac": .2, "standardization": "inner-train only for MLP; outer-train for kNN",
              "k": 10, "bootstrap_samples": 2000, "device": "cpu", "threads": threads,
              "sources": {k: str(v) for k, v in sources.items()}}
    if (root/"config.json").exists() and json.loads((root/"config.json").read_text()) != config:
        raise ValueError("resume configuration differs")
    # Preflight all cached ridge folds before creating any output or fitting.
    jobs, pixel_jobs, pools = [], {}, {}
    for name in checkpoints:
        phase = "phaseB" if name in PHASE_B else "phaseA"
        source = sources[phase]
        if phase not in pools:
            pools[phase] = _pool(source)
            source_config = json.loads((source/"config.json").read_text())
            if source_config["seed"] != 42 or source_config["outer_folds"] != 5:
                raise ValueError("cached pool must use seed 42 and five outer folds")
        targets, groups, identity = pools[phase]
        for radius in radii:
            arm = json.loads((source/f"{name}_r{radius}.json").read_text())
            wt = int(arm["with_target"])
            pixel_stem = f"pixel_r{radius}_{wt}"
            pixel_cache = f"pixels_r{radius}_{wt}.npz"
            key = (identity, fingerprint(source/pixel_cache))
            if key not in pixel_jobs:
                # Content-derived names also share pixels when a resumed run
                # expands from Phase B alone to both phases (or vice versa).
                digest = hashlib.sha256((identity+key[1]).encode()).hexdigest()[:16]
                unit = f"{pixel_stem}_{digest}"
                pixel_jobs[key] = {"unit": unit, "phase": phase, "kind": "pixel", "radius": radius,
                    "with_target": bool(wt), "source": str(source), "cache": pixel_cache,
                    "out_dir": str(root), "threads": threads, "pool_identity": identity,
                    "linear": _check_ridge(source, pixel_stem, targets, groups)}
            else:
                _check_ridge(source, pixel_stem, targets, groups)
            for kind in kinds:
                stem = f"{name}_r{radius}_{kind}"
                jobs.append({"unit": f"{phase}_{stem}", "checkpoint": name, "phase": phase,
                    "radius": radius, "kind": kind, "with_target": bool(wt), "source": str(source),
                    "cache": f"{name}_r{radius}_features.npz", "out_dir": str(root), "threads": threads,
                    "pool_identity": identity, "pixel_unit": pixel_jobs[key]["unit"],
                    "linear": _check_ridge(source, stem, targets, groups)})
    if len(pools) == 2 and pools["phaseA"][2] != pools["phaseB"][2]:
        raise ValueError("Phase B pool identity differs from Phase A")
    if include_pixels:
        jobs += list(pixel_jobs.values())
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"config.json", config)
    pending = [j for j in jobs if not (root/f'{j["unit"]}.json').exists()]
    start, completed = time.perf_counter(), []
    print(f"pin-mlp: {len(pending)} pending, {len(jobs)-len(pending)} cached; {workers} workers x {threads} CPU threads", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_run_unit, job): job["unit"] for job in pending}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f'pin-mlp: {result["unit"]} completed in {result["wall_s"]:.2f}s ({len(completed)}/{len(pending)})', flush=True)
    return {"out_dir": str(root), "completed": completed, "skipped": len(jobs)-len(pending),
            "invocation_wall_s": time.perf_counter()-start}


def pin_mlp_report(out_dir=None):
    """Write point-estimate labels and episode-cluster CIs from finished units."""
    root = _output_root(out_dir)
    units = {}
    for path in sorted(root.glob("*.json")):
        unit = json.loads(path.read_text())
        if unit.get("stage") == "pin-mlp-unit":
            units[unit["unit"]] = unit
    rows, pixels, missing = [], [], []
    for unit in units.values():
        linear, metrics = unit["linear"], unit["metrics"]
        row = {k: unit[k] for k in ("unit", "phase", "radius", "kind")}
        row.update(linear=linear["ridge"], mlp=metrics["mlp"]["mlp"],
                   knn=metrics["knn"]["knn"], mean=linear["mean"])
        if unit["kind"] == "pixel":
            row["with_target"] = unit["with_target"]
            pixels.append(row)
            continue
        pixel = units.get(unit["pixel_unit"])
        if pixel is None:
            missing.append({"unit": unit["unit"], "pixel_unit": unit["pixel_unit"]})
            continue
        if pixel["pool_identity"] != unit["pool_identity"]:
            raise ValueError("pixel baseline pool differs")
        row["checkpoint"] = unit["checkpoint"]
        row["pixel_unit"] = unit["pixel_unit"]
        row["pixel_mlp"] = pixel["metrics"]["mlp"]["mlp"]
        peg, t = row["mlp"]["peg"]["mean"], row["mlp"]["T"]["mean"]
        pixel_peg, mean, knn = row["pixel_mlp"]["peg"]["mean"], row["mean"]["peg"]["mean"], row["knn"]["peg"]["mean"]
        row.update(pin_label(peg, t, pixel_peg, mean, knn))
        row["ratios"] = {key: numerator/denominator if denominator > 0 else None
            for key, numerator, denominator in (("MLP-peg/MLP-T", peg, t),
                ("MLP-peg/pixel-MLP-peg", peg, pixel_peg), ("MLP-peg/mean-peg", peg, mean),
                ("kNN-peg/mean-peg", knn, mean))}
        rows.append(row)
    expected = {(name, r, k) for name in (*DEFAULT_CHECKPOINTS, *PHASE_B) for r in (15,45) for k in ("cls", "projected")}
    absent = sorted(expected-{(r["checkpoint"], r["radius"], r["kind"]) for r in rows})
    result = {"stage": "pin-mlp-report", "rows": rows, "pixels": pixels,
              "missing_pixel_units": missing, "missing_embedding_units": absent,
              "complete": not missing and not absent, "primary_embedding": "projected"}
    def value(metric):
        lo, hi = metric["ci"]
        return f'{metric["mean"]:.2f} [{lo:.2f}, {hi:.2f}]'
    lines = ["# Pin nonlinear readout", "", "Euclidean pixel error [episode-cluster bootstrap 95% CI, 2000 draws].",
             "Labels use point estimates; projected is primary. Incomplete units are listed in summary.json.", "",
             "| Checkpoint | Radius | Embedding | Linear peg | Linear T | MLP peg | MLP T | kNN peg | kNN T | Pixel-MLP peg | Pixel-MLP T | Mean peg | Mean T | MLP-peg/MLP-T | MLP-peg/pixel-MLP-peg | MLP-peg/mean-peg | kNN-peg/mean-peg | Label | Conflict |",
             "|"+"|".join(["---"]*19)+"|"]
    for row in rows:
        cells = [row["checkpoint"], str(row["radius"]), row["kind"]]
        cells += [value(row[method][target]) for method in ("linear", "mlp", "knn", "pixel_mlp", "mean") for target in TARGETS]
        cells += [f"{ratio:.3f}" if ratio is not None else "undefined" for ratio in row["ratios"].values()]
        cells += [row["label"], str(row["conflict"])]
        lines.append("| "+" | ".join(cells)+" |")
    lines += ["", "| Pixel cache | Radius | With target | Linear peg | Linear T | MLP peg | MLP T | kNN peg | kNN T | Mean peg | Mean T |",
              "|"+"|".join(["---"]*11)+"|"]
    for row in pixels:
        cells = [row["unit"], str(row["radius"]), str(row["with_target"])]
        cells += [value(row[method][target]) for method in ("linear", "mlp", "knn", "mean") for target in TARGETS]
        lines.append("| "+" | ".join(cells)+" |")
    lines += ["", f'Complete: {result["complete"]}. Missing embedding rows: {len(absent)}.']
    write_json(root/"summary.json", result)
    (root/"summary.md").write_text("\n".join(lines)+"\n")
    return result
