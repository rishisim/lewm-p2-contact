#!/usr/bin/env python3
"""Execute the frozen LeWM adaptive-compute V3 protocol end to end."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from model import (AdaptiveLeWM, LocalGainGate, SharedResidualRefiner,
                   build_causal_features, causal_feature_names,
                   gate_flops_per_decision, joint_loss, refiner_flops_per_call)
import policy


def _load_v2_model_io():
    path = REPO / "runs/lewm_adaptive_compute_v2/model_io.py"
    spec = importlib.util.spec_from_file_location("v3_reused_model_io", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


model_io = _load_v2_model_io()
SOURCE = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")
BASE_CONFIG = REPO / "runs/lewm_transfer/cube/cache/model/config.json"
BASE_WEIGHTS = REPO / "runs/lewm_transfer/cube/cache/model/weights.pt"
V1_REFINER = REPO / "runs/lewm_adaptive_compute_v1/checkpoints/refiner_seed_260713.pt"
PRIOR_MANIFESTS = (
    REPO / "runs/lewm_adaptive_compute_v1/cache/split_manifest.json",
    REPO / "runs/lewm_adaptive_compute_v1/smoke_run/cache/split_manifest.json",
    REPO / "runs/lewm_adaptive_compute_v2/cache/split_manifest.json",
    REPO / "runs/lewm_adaptive_compute_v2/smoke_run/cache/split_manifest.json",
)
EXPECTED = {
    "base_config": "4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999",
    "base_weights": "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89",
    "v1_refiner": "388a82fc30c96921083bfa4f2578cb296e6c3544510953d17578cf766cdf10f1",
    "prior": (
        "3cc9bd4d7df229b0a4111a49933e98c5b72e916b5b5f98d7d5e023fc30ed544d",
        "f5a4da6c0f37d6222e83085327e30a1b70f0ace6f96e9c97d37c1067d7b65940",
        "ce34fa665fa0b7c0237a211e12c1f44e442408a404bf34eba2010c15dc83c1c0",
        "ad1d2e1cf42deb48518009bffbd5262925db2ff202a01eb65f1909125535ffb3",
    ),
    "plan": "21efb81487e7eee4506f20318791a7d54fdfca7c33c5c356374e344089dce9bf",
    "full_config": "63a83582217666eed01d2d5382d68c88b59e4aed3704792968aba2b5c57c8744",
    "smoke_config": "bdd50bef8cdeb367b38086d004bf0ab73ff244071f10cf18b9e9a08b13041448",
}
SPLITS = {"train": 0, "calibration": 1, "test": 2}


def log(message: str) -> None:
    print(f"[lewm-v3] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite strict JSON value")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [jsonable(dict(row)) for row in rows]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader(); writer.writerows(rows)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "mps": torch.mps.synchronize()
    elif device.type == "cuda": torch.cuda.synchronize(device)


def validate_preregistration(cfg_path: Path) -> dict[str, str]:
    observed = {
        "plan": sha256(ROOT / "PLAN.md"),
        "full_config": sha256(ROOT / "full_config.json"),
        "smoke_config": sha256(ROOT / "smoke_config.json"),
        "base_config": sha256(BASE_CONFIG), "base_weights": sha256(BASE_WEIGHTS),
        "v1_refiner": sha256(V1_REFINER),
    }
    for key, expected in EXPECTED.items():
        if key == "prior": continue
        if observed[key] != expected:
            raise RuntimeError(f"frozen hash mismatch for {key}: {observed[key]}")
    cfg = json.loads(cfg_path.read_text())
    expected_config_name = "full_config" if cfg["mode"] == "full" else "smoke_config"
    if cfg_path.resolve() != (ROOT / f"{cfg['mode']}_config.json").resolve():
        raise RuntimeError("configuration must be the frozen V3 config file")
    if observed[expected_config_name] != EXPECTED[expected_config_name]:
        raise RuntimeError("active config hash mismatch")
    if SOURCE.stat().st_size != 101_942_558_720:
        raise RuntimeError("Cube source size changed")
    return observed


def prior_exclusions() -> tuple[np.ndarray, dict[str, Any]]:
    used = set(range(30)); records = []
    for path, expected in zip(PRIOR_MANIFESTS, EXPECTED["prior"]):
        actual = sha256(path)
        if actual != expected: raise RuntimeError(f"prior manifest drift: {path}")
        payload = json.loads(path.read_text())
        split = payload["selection"]["split_episode_ordinals"]
        values = {int(v) for name in SPLITS for v in split[name]}
        used.update(values)
        records.append({"path": str(path.resolve()), "sha256": actual, "episode_count": len(values)})
    values = np.asarray(sorted(used), dtype=np.int64)
    digest = hashlib.sha256(np.asarray(values, dtype="<i4").tobytes()).hexdigest()
    if len(values) != 1242 or digest != "ac71dbe795efdd0217983cd615a55ed81244303ae1243d559932d2a5def62915":
        raise RuntimeError("prior exclusion union mismatch")
    return values, {"manifests": records, "union_count": len(values), "union_sorted_int32le_sha256": digest}


def make_split(cfg: Mapping[str, Any], excluded: np.ndarray) -> dict[str, np.ndarray]:
    reserved = np.asarray([42, 43, 44, 45, 46, 47], dtype=np.int64)
    if cfg["mode"] == "smoke":
        split = {name: np.asarray(cfg[f"{name}_episode_ordinals"], dtype=np.int64) for name in SPLITS}
        if set(np.concatenate(list(split.values()))) != set(reserved):
            raise RuntimeError("smoke split drift")
    else:
        eligible = np.asarray(sorted(set(range(10_000)) - set(excluded) - set(reserved)), dtype=np.int64)
        selected = np.random.default_rng(int(cfg["selection_seed"])).choice(
            eligible, size=600, replace=False
        )
        a, b = int(cfg["train_episodes"]), int(cfg["calibration_episodes"])
        split = {"train": selected[:a], "calibration": selected[a:a+b], "test": selected[a+b:]}
    sets = {k: set(v.tolist()) for k, v in split.items()}
    if any(sets[a] & sets[b] for a, b in (("train","calibration"),("train","test"),("calibration","test"))):
        raise RuntimeError("V3 split overlap")
    if set(np.concatenate(list(split.values()))) & set(excluded):
        raise RuntimeError("V3 split intersects prior work")
    if cfg["mode"] == "full" and set(np.concatenate(list(split.values()))) & set(reserved):
        raise RuntimeError("full split intersects V3 smoke")
    return split


def extract(cfg: Mapping[str, Any], output: Path, split: Mapping[str, np.ndarray],
            excluded: np.ndarray, provenance: Mapping[str, Any], device: torch.device,
            force: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache = output / "cache/cube_inputs.npz"; manifest = output / "cache/split_manifest.json"
    if force or not cache.exists() or not manifest.exists():
        log("extracting strictly isolated Cube latent cache")
        model_io.extract_fresh_cube_cache(
            source_h5=SOURCE, config_path=BASE_CONFIG, weights_path=BASE_WEIGHTS,
            output_npz=cache, split_manifest_path=manifest, split_episodes=split,
            selection_seed=int(cfg.get("selection_seed", 260913)),
            excluded_episode_ordinals=excluded,
            reserved_episode_ordinals=(range(42,48) if cfg["mode"] == "full" else ()),
            prior_manifest_provenance=provenance, device=device,
            encode_batch_size=int(cfg["encode_batch_size"]),
            predict_batch_size=int(cfg["predict_batch_size"]),
        )
    with np.load(cache, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    required = {"history","action","base_pred","target","episode_id","model_step","split"}
    if required - set(arrays): raise RuntimeError("V3 cache incomplete")
    if any(not np.isfinite(arrays[k]).all() for k in ("history","action","base_pred","target")):
        raise RuntimeError("nonfinite V3 cache")
    for name, code in SPLITS.items():
        observed = set(np.unique(arrays["episode_id"][arrays["split"] == code]).tolist())
        if observed != set(split[name].tolist()): raise RuntimeError(f"cache {name} split mismatch")
        if int((arrays["split"] == code).sum()) != 38 * len(split[name]):
            raise RuntimeError(f"cache {name} row count mismatch")
    meta = json.loads(manifest.read_text())
    if meta["cache"]["sha256"] != sha256(cache): raise RuntimeError("cache hash mismatch")
    return arrays, meta


def new_model(seed: int, cfg: Mapping[str, Any], device: torch.device) -> AdaptiveLeWM:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    refiner = SharedResidualRefiner(hidden_dim=int(cfg["hidden_dim"]), iteration_dim=int(cfg["iteration_dim"]))
    payload = torch.load(V1_REFINER, map_location="cpu", weights_only=True)
    result = refiner.load_state_dict(payload["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys: raise RuntimeError("V1 refiner strict load failed")
    gate = LocalGainGate(len(causal_feature_names()), tuple(cfg["gate_hidden_dims"]), int(cfg["gate_iteration_dim"]))
    return AdaptiveLeWM(refiner, gate).to(device)


def batches(indices: np.ndarray, size: int):
    for start in range(0, len(indices), size): yield indices[start:start+size]


def tensors(arrays: Mapping[str, np.ndarray], index: np.ndarray, device: torch.device,
            include_target: bool = True):
    out = [torch.as_tensor(np.ascontiguousarray(arrays[k][index]), device=device) for k in ("history","action","base_pred")]
    if include_target: out.append(torch.as_tensor(np.ascontiguousarray(arrays["target"][index]), device=device))
    return out


@torch.no_grad()
def fit_gate_normalization(model: AdaptiveLeWM, arrays: Mapping[str, np.ndarray],
                           indices: np.ndarray, cfg: Mapping[str, Any], device: torch.device) -> None:
    features, gains = [], []
    model.eval()
    for index in batches(indices, int(cfg["predict_batch_size"])):
        h, a, z, y = tensors(arrays, index, device)
        exits, updates, _ = model.refiner.forward_all(h, a, z)
        mse = torch.stack([(value-y).square().mean(1) for value in exits], 1)
        for depth in (1,2,3):
            features.append(build_causal_features(h,a,exits[depth],updates[depth-1]).cpu())
            gains.append((mse[:,depth]-mse[:,depth+1]).cpu())
    model.gate.fit_normalization(torch.cat(features).to(device), torch.cat(gains).to(device))


@torch.no_grad()
def objective(model: AdaptiveLeWM, arrays: Mapping[str, np.ndarray], indices: np.ndarray,
              cfg: Mapping[str, Any], device: torch.device) -> tuple[float, dict[str,float]]:
    model.eval(); sums = {}; count = 0
    for index in batches(indices, int(cfg["train_batch_size"])):
        loss, parts = joint_loss(model, *tensors(arrays,index,device), 4, cfg)
        n = len(index); count += n
        for key,value in parts.items(): sums[key] = sums.get(key,0.0) + value*n
    return sums["total"]/count, {k:v/count for k,v in sums.items()}


def train_seed(seed: int, cfg: Mapping[str, Any], arrays: Mapping[str,np.ndarray],
               train_idx: np.ndarray, cal_idx: np.ndarray, device: torch.device,
               checkpoint: Path) -> tuple[AdaptiveLeWM, dict[str,Any], list[dict[str,Any]]]:
    model = new_model(seed,cfg,device)
    fit_gate_normalization(model,arrays,train_idx,cfg,device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    rng = np.random.default_rng(seed); depth_rng = np.random.default_rng(seed+int(cfg["stochastic_depth_seed_offset"]))
    best = math.inf; best_state=None; best_epoch=-1; stale=0; history=[]
    for epoch in range(int(cfg["max_epochs"])):
        model.train(); order=rng.permutation(train_idx); train_sum=0.0; seen=0
        for index in batches(order,int(cfg["train_batch_size"])):
            cap=int(depth_rng.integers(1,5)); optimizer.zero_grad(set_to_none=True)
            loss,_=joint_loss(model,*tensors(arrays,index,device),cap,cfg)
            if not bool(torch.isfinite(loss)): raise RuntimeError("nonfinite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),float(cfg["gradient_clip"])); optimizer.step()
            train_sum += float(loss.detach())*len(index); seen += len(index)
        cal, parts=objective(model,arrays,cal_idx,cfg,device)
        history.append({"seed":seed,"epoch":epoch+1,"train_objective":train_sum/seen,
                        "calibration_objective":cal,**{f"cal_{k}":v for k,v in parts.items()}})
        if cal < best-float(cfg["min_delta"]):
            best=cal; best_epoch=epoch+1; best_state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if epoch==0 or (epoch+1)%10==0: log(f"seed={seed} epoch={epoch+1} cal={cal:.7g} best={best:.7g}")
        if stale>=int(cfg["patience"]): break
    if best_state is None: raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state,strict=True); model.eval().requires_grad_(False)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":best_state,"seed":seed,"best_epoch":best_epoch,
                "calibration_objective":best,"config_sha256":sha256(ROOT/f"{cfg['mode']}_config.json")},checkpoint)
    return model,{"seed":seed,"best_epoch":best_epoch,"calibration_objective":best,
                  "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha256(checkpoint)},history


@torch.no_grad()
def dense_predict(model: AdaptiveLeWM, arrays: Mapping[str,np.ndarray], indices: np.ndarray,
                  cfg: Mapping[str,Any], device: torch.device) -> tuple[np.ndarray,np.ndarray]:
    exits=np.empty((len(indices),5,192),np.float32); scores=np.empty((len(indices),3),np.float32)
    model.eval(); cursor=0
    for index in batches(indices,int(cfg["predict_batch_size"])):
        h,a,z=tensors(arrays,index,device,False); values,_,gate=model.dense(h,a,z)
        n=len(index); exits[cursor:cursor+n]=torch.stack(values,1).cpu().numpy()
        scores[cursor:cursor+n]=torch.stack(gate,1).cpu().numpy(); cursor+=n
    if not np.isfinite(exits).all() or not np.isfinite(scores).all(): raise RuntimeError("nonfinite dense outputs")
    return exits,scores


def raw_losses(target: np.ndarray, exits: np.ndarray) -> np.ndarray:
    return np.einsum("nkd,nkd->nk",exits-target[:,None,:],exits-target[:,None,:],optimize=True)/target.shape[1]


def fit_whitening(target: np.ndarray) -> dict[str,np.ndarray|float]:
    mean=target.mean(0,dtype=np.float64); centered=target.astype(np.float64)-mean
    cov=np.einsum("ni,nj->ij",centered,centered,optimize=True)/max(len(target)-1,1)
    values,vectors=np.linalg.eigh(cov); floor=max(float(values.max())*1e-6,1e-12); used=np.maximum(values,floor)
    # Avoid Accelerate/BLAS nonfinite warnings observed for ill-conditioned
    # smoke covariance matrices; this is the finite einsum path retained from V2.
    matrix=np.einsum("ik,k,jk->ij",vectors,1/np.sqrt(used),vectors,optimize=False)
    return {"mean":mean,"matrix":matrix,"eigenvalues":values,"floor":floor}


def white_losses(target: np.ndarray, exits: np.ndarray, whitening: Mapping[str,Any]) -> np.ndarray:
    diff=np.einsum("nkd,df->nkf",exits.astype(np.float64)-target[:,None,:],
                   whitening["matrix"],optimize=False)
    result=np.einsum("nkd,nkd->nk",diff,diff,optimize=True)/target.shape[1]
    if not np.isfinite(result).all(): raise RuntimeError("nonfinite whitened loss")
    return result


def freeze_policies(seed_models: Mapping[int,AdaptiveLeWM], seed_summaries: Sequence[Mapping[str,Any]],
                    arrays: Mapping[str,np.ndarray], idx: Mapping[str,np.ndarray], cfg: Mapping[str,Any],
                    device: torch.device, output: Path):
    records={}; calibration_cache={}; test_cache={}; thresholds=[]
    for summary in seed_summaries:
        seed=int(summary["seed"]); model=seed_models[seed]
        cal_exits,cal_scores=dense_predict(model,arrays,idx["calibration"],cfg,device)
        cal_target=arrays["target"][idx["calibration"]]  # calibration-only fitting
        cal_raw=raw_losses(cal_target,cal_exits); calibration_cache[seed]=(cal_exits,cal_scores,cal_raw)
        test_exits,test_scores=dense_predict(model,arrays,idx["test"],cfg,device)  # target-free boundary
        test_cache[seed]=(test_exits,test_scores)
        seed_record={}
        for target_mean in cfg["target_mean_depths"]:
            selected=policy.calibrate_threshold(cal_scores,cal_raw,float(target_mean),int(cfg["threshold_grid_size"]))
            test_depth=policy.local_depths(test_scores,float(selected["threshold"]))
            key=f"{float(target_mean):.2f}"; seed_record[key]={"calibration":selected,"adaptive":test_depth}
            thresholds.append({"seed":seed,"operating_point":key,**selected,
                               "test_mean_depth_target_free":float(test_depth.mean()),
                               "test_total_calls_target_free":int(test_depth.sum())})
        records[seed]=seed_record
    selected=min(seed_summaries,key=lambda x:(float(x["calibration_objective"]),list(cfg["training_seeds"]).index(int(x["seed"]))))
    primary_seed=int(selected["seed"])
    payload={"selected_seed":np.asarray(primary_seed),"selected_rule":np.asarray("smallest calibration objective")}
    for seed,by_op in records.items():
        for op,data in by_op.items(): payload[f"seed_{seed}_op_{op}_adaptive"]=data["adaptive"]
    path=output/"metrics/test_allocations_frozen_pre_outcome.npz"; path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**payload)
    write_csv(output/"metrics/calibrated_thresholds.csv",thresholds)
    return primary_seed,records,calibration_cache,test_cache,{"path":str(path.resolve()),"sha256":sha256(path)}


def comparison(name: str, left: np.ndarray, right: np.ndarray, episodes: np.ndarray,
               cfg: Mapping[str,Any], metric: str) -> dict[str,Any]:
    return {"comparison":name,"metric":metric,**policy.clustered_ci(
        np.asarray(left)-np.asarray(right),episodes,int(cfg["bootstrap_samples"]),int(cfg["bootstrap_seed"]))}


def latent_predict_flops() -> int:
    # Frozen analytic contract for B=1,T=3; LayerNorm/activations are omitted.
    t=3; d=192; heads=16; dh=64; inner=heads*dh; mlp=2048
    action=2*t*(25*25+25*768+768*192)
    block=2*t*(d*(3*inner)+inner*d+d*mlp+mlp*d+d*(6*d)) + 4*heads*t*t*dh
    predictor=6*block
    pred_proj=2*t*(d*2048+2048*d)
    return int(action+predictor+pred_proj)


@torch.no_grad()
def online_adaptive(model: AdaptiveLeWM, h: torch.Tensor,a: torch.Tensor,z: torch.Tensor,
                    threshold: float) -> tuple[torch.Tensor,torch.Tensor,int]:
    current=z; depths=torch.ones(len(z),dtype=torch.long,device=z.device); active=torch.arange(len(z),device=z.device); gate_rows=0
    last=model.refiner.update(h,a,current,0); current=current+last
    for decision_depth in (1,2,3):
        if not len(active): break
        score=model.gate(build_causal_features(h[active],a[active],current[active],last[active]),decision_depth)
        gate_rows+=len(active); keep=active[score>threshold]
        if not len(keep): break
        update=model.refiner.update(h[keep],a[keep],current[keep],decision_depth)
        current=current.index_copy(0,keep,current[keep]+update); depths[keep]+=1
        replacement=torch.zeros_like(last); replacement[keep]=update; last=replacement; active=keep
    return current,depths,gate_rows


@torch.no_grad()
def latency(model: AdaptiveLeWM, arrays: Mapping[str,np.ndarray], test_idx: np.ndarray,
            threshold: float, cfg: Mapping[str,Any], device: torch.device) -> dict[str,Any]:
    base,_,_=model_io.load_frozen_lewm(BASE_CONFIG,BASE_WEIGHTS,device,
        expected_config_sha256=EXPECTED["base_config"],expected_weights_sha256=EXPECTED["base_weights"])
    h=torch.as_tensor(np.ascontiguousarray(arrays["history"][test_idx]),device=device)
    a=torch.as_tensor(np.ascontiguousarray(arrays["action"][test_idx]),device=device)
    repeats=[]; observed=None
    for _ in range(int(cfg["latency_repeats"])):
        synchronize(device); start=time.perf_counter()
        z=base.predict(h,base.action_encoder(a))[:,-1]
        _,depths,gate_rows=online_adaptive(model,h,a,z,threshold)
        synchronize(device); repeats.append(time.perf_counter()-start); observed=(depths,gate_rows)
    depths,gate_rows=observed
    base.eval().requires_grad_(False)
    if any(p.grad is not None or p.requires_grad for p in base.parameters()): raise RuntimeError("base freeze failed in timing")
    return {"device":str(device),"samples":len(test_idx),"repeats_seconds":repeats,
            "median_seconds":float(np.median(repeats)),"total_calls":int(depths.sum()),
            "mean_depth":float(depths.float().mean()),"gate_decisions":int(gate_rows)}


def make_figures(output: Path, policy_rows: list[dict[str,Any]], threshold_rows: list[dict[str,Any]],
                 seed_rows: list[dict[str,Any]], regime_rows: list[dict[str,Any]],
                 frontier: list[dict[str,Any]] | None = None) -> dict[str,Any]:
    import matplotlib.pyplot as plt
    figures=output/"figures"; figures.mkdir(parents=True,exist_ok=True); paths={}
    adaptive=[r for r in policy_rows if r["policy"]=="adaptive"]
    baselines=[r for r in policy_rows if r["policy"] in ("fixed_d1","fixed_d2","fixed_d4","matched_mixture")]
    fig,ax=plt.subplots(figsize=(7,5));
    if frontier:
        baselines=[r for r in frontier if r["policy"]!="adaptive"]
        adaptive_frontier=[r for r in frontier if r["policy"]=="adaptive"]
    else:
        adaptive_frontier=adaptive
    for rows,label,marker in ((baselines,"fixed/mixed","o"),(adaptive_frontier,"adaptive","s")):
        ax.scatter([r["mean_calls"] for r in rows],[r["raw_mse"] for r in rows],label=label,marker=marker,s=55)
    ax.set(xlabel="Mean refiner calls",ylabel="Raw latent MSE",title="V3 compute-error frontier"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    paths["pareto"]=figures/"01_compute_error_pareto.png"; fig.savefig(paths["pareto"],dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ops=sorted({r["operating_point"] for r in threshold_rows}); x=np.arange(len(ops)); width=.18
    for depth in (1,2,3,4):
        counts=[]
        for op in ops:
            row=next(r for r in adaptive if r["operating_point"]==op); counts.append(row.get(f"depth_{depth}_fraction",0))
        ax.bar(x+(depth-2.5)*width,counts,width,label=f"depth {depth}")
    ax.set_xticks(x,ops); ax.set(xlabel="Target mean depth",ylabel="Fraction",title="Local adaptive allocations"); ax.legend(); fig.tight_layout()
    paths["allocation"]=figures/"02_allocation.png"; fig.savefig(paths["allocation"],dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); selected=[r for r in threshold_rows if r.get("selected_seed")]
    ax.plot([r["target_mean_depth"] for r in selected],[r["calibration_mean_depth"] for r in selected],"o-",label="calibration")
    ax.plot([r["target_mean_depth"] for r in selected],[r["test_mean_depth_target_free"] for r in selected],"s-",label="test realized")
    ax.plot([1.2,2.6],[1.2,2.6],"k--",alpha=.4); ax.set(xlabel="Target",ylabel="Mean depth",title="Calibration-to-test compute transfer"); ax.legend(); fig.tight_layout()
    paths["calibration"]=figures/"03_calibration_transfer.png"; fig.savefig(paths["calibration"],dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); labels=[str(int(r["seed"])) for r in seed_rows]; x=np.arange(len(labels)); width=.36
    fixed_values=[r["benefit_vs_fixed_d1"] for r in seed_rows]; matched_values=[r["benefit_vs_matched"] for r in seed_rows]
    ax.bar(x-width/2,fixed_values,width,label="vs fixed depth 1",color="#2a9d8f")
    ax.bar(x+width/2,matched_values,width,label="vs matched mixture",color="#e76f51")
    ax.axhline(0,color="black",lw=1); ax.set_xticks(x,labels); ax.set(xlabel="Training seed",ylabel="Raw benefit",title="Primary operating-point seed stability"); ax.legend(); fig.tight_layout()
    paths["seeds"]=figures/"04_seed_stability.png"; fig.savefig(paths["seeds"],dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); names=[r["regime"] for r in regime_rows]; vals=[r["mean_depth"] for r in regime_rows]
    ax.bar(names,vals,color="#457b9d"); ax.set(xlabel="Post-hoc regime",ylabel="Mean adaptive depth",title="Allocation by physical regime"); ax.tick_params(axis="x",rotation=20); fig.tight_layout()
    paths["regimes"]=figures/"05_posthoc_regimes.png"; fig.savefig(paths["regimes"],dpi=220); plt.close(fig)
    from PIL import Image
    audit={}
    for name,path in paths.items():
        with Image.open(path) as image: audit[name]={"path":str(path.resolve()),"sha256":sha256(path),"width":image.width,"height":image.height}
    return audit


def run(args: argparse.Namespace) -> dict[str,Any]:
    cfg=json.loads(args.config.read_text()); frozen_hashes=validate_preregistration(args.config)
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    if ROOT.resolve() not in output.parents and output!=ROOT.resolve(): raise RuntimeError("V3 outputs must remain in V3 directory")
    for name in ("cache","checkpoints","metrics","figures","logs"): (output/name).mkdir(exist_ok=True)
    device=choose_device(args.device); log(f"mode={cfg['mode']} device={device}")
    excluded,exclusion_provenance=prior_exclusions(); split=make_split(cfg,excluded)
    planned={"selection_seed":cfg.get("selection_seed"),"split_episode_ordinals":{k:v.tolist() for k,v in split.items()},
             "excluded":exclusion_provenance,"smoke_reserved":[42,43,44,45,46,47],"pretraining_membership":"unknown"}
    write_json(output/"cache/planned_split.json",planned)
    arrays,manifest=extract(cfg,output,split,excluded,exclusion_provenance,device,args.force_extract)
    idx={name:np.flatnonzero(arrays["split"]==code) for name,code in SPLITS.items()}
    base_hash=hashlib.sha256(np.ascontiguousarray(arrays["base_pred"]).tobytes()).hexdigest()
    histories=[]; models={}; summaries=[]
    for seed in cfg["training_seeds"]:
        checkpoint=output/f"checkpoints/joint_seed_{seed}.pt"
        if checkpoint.exists() and not args.force_train:
            payload=torch.load(checkpoint,map_location=device,weights_only=True); model=new_model(int(seed),cfg,device)
            model.load_state_dict(payload["state_dict"],strict=True); model.eval().requires_grad_(False)
            summary={k:payload[k] for k in ("seed","best_epoch","calibration_objective")}
            summary.update({"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha256(checkpoint)})
            history=[]
        else:
            log(f"training joint refiner/gate seed {seed}")
            model,summary,history=train_seed(int(seed),cfg,arrays,idx["train"],idx["calibration"],device,checkpoint)
        models[int(seed)]=model; summaries.append(summary); histories.extend(history)
    if histories: write_csv(output/"metrics/training_history.csv",histories)
    write_json(output/"metrics/seed_selection.json",{"seeds":summaries,"rule":"smallest calibration objective; ties seed order"})
    primary_seed,records,cal_cache,test_cache,allocation_freeze=freeze_policies(models,summaries,arrays,idx,cfg,device,output)
    threshold_rows=[]
    for seed,ops in records.items():
        for op,data in ops.items(): threshold_rows.append({"seed":seed,"operating_point":op,
            **data["calibration"],"test_mean_depth_target_free":float(data["adaptive"].mean()),
            "test_total_calls_target_free":int(data["adaptive"].sum()),"selected_seed":seed==primary_seed})

    # Final-test outcome boundary: all models, thresholds, seed selection, and
    # adaptive allocations above are frozen before this target view.
    test_target=arrays["target"][idx["test"]]
    test_episodes=arrays["episode_id"][idx["test"]]
    cal_target=arrays["target"][idx["calibration"]]
    whitening=fit_whitening(cal_target)
    np.savez_compressed(output/"cache/calibration_whitening.npz",**whitening)
    seed_eval={}; seed_rows=[]; all_policy_rows=[]; all_comparisons=[]
    primary_op=f"{float(cfg['primary_target_mean_depth']):.2f}"
    for seed in cfg["training_seeds"]:
        seed=int(seed); cal_exits,_,cal_raw=cal_cache[seed]; exits,_=test_cache[seed]
        raw=raw_losses(test_target,exits); white=white_losses(test_target,exits,whitening)
        per_op={}
        for op,data in records[seed].items():
            adaptive=data["adaptive"]; calls=int(adaptive.sum()); n=len(adaptive)
            means=cal_raw.mean(0)
            matched=policy.strongest_mixture(n,calls,means,int(cfg["mixed_baseline_seed"])+seed+int(float(op)*100))
            random_alloc=policy.histogram_randomized(adaptive,int(cfg["random_seed"])+seed+int(float(op)*100))
            permutation=policy.histogram_permutation(adaptive,test_episodes,int(cfg["permutation_seed"])+seed+int(float(op)*100))
            allocations={"adaptive":adaptive,"matched_mixture":matched,"random_histogram":random_alloc,"permutation":permutation,
                         "fixed_d1":np.ones(n,dtype=np.int64),"fixed_d2":np.full(n,2),"fixed_d4":np.full(n,4)}
            if any(int(v.sum())!=calls for k,v in allocations.items() if k in ("adaptive","matched_mixture","random_histogram","permutation")):
                raise RuntimeError("matched policy call audit failed")
            oracle=policy.oracle_exact(raw,calls); allocations["oracle"]=oracle
            losses={name:policy.gather(raw,depths) for name,depths in allocations.items()}
            whites={name:policy.gather(white,depths) for name,depths in allocations.items()}
            per_op[op]={"allocations":allocations,"losses":losses,"white":whites}
            if seed==primary_seed:
                for name,depths in allocations.items():
                    hist=policy.histogram(depths); row={"seed":seed,"operating_point":op,"policy":name,
                        "raw_mse":float(losses[name].mean()),"whitened_mse":float(whites[name].mean()),
                        "total_calls":int(depths.sum()),"mean_calls":float(depths.mean()),"diagnostic_only":name=="oracle"}
                    for d in range(1,5): row[f"depth_{d}_fraction"]=hist.get(d,0)/len(depths)
                    all_policy_rows.append(row)
                for metric,values in (("raw",losses),("whitened",whites)):
                    for name in ("matched_mixture","fixed_d1","random_histogram","permutation"):
                        all_comparisons.append(comparison(f"adaptive_vs_{name}",values[name],values["adaptive"],test_episodes,cfg,metric)|{"seed":seed,"operating_point":op})
                    all_comparisons.append(comparison("oracle_advantage_vs_matched",values["matched_mixture"],values["oracle"],test_episodes,cfg,metric)|{"seed":seed,"operating_point":op,"diagnostic_only":True})
        seed_eval[seed]={"raw":raw,"white":white,"per_op":per_op}
        primary=per_op[primary_op]["losses"]
        seed_rows.append({"seed":seed,"benefit_vs_matched":float((primary["matched_mixture"]-primary["adaptive"]).mean()),
                          "benefit_vs_fixed_d1":float((primary["fixed_d1"]-primary["adaptive"]).mean()),
                          "primary_mean_depth":float(records[seed][primary_op]["adaptive"].mean())})

    write_csv(output/"metrics/policy_metrics.csv",all_policy_rows); write_csv(output/"metrics/paired_comparisons.csv",all_comparisons)
    write_csv(output/"metrics/seed_stability.csv",seed_rows)
    selected_eval=seed_eval[primary_seed]; primary_values=selected_eval["per_op"][primary_op]["losses"]
    aggregate_matched=np.mean([seed_eval[int(s)]["per_op"][primary_op]["losses"]["matched_mixture"]-seed_eval[int(s)]["per_op"][primary_op]["losses"]["adaptive"] for s in cfg["training_seeds"]],axis=0)
    aggregate_fixed=np.mean([seed_eval[int(s)]["per_op"][primary_op]["losses"]["fixed_d1"]-seed_eval[int(s)]["per_op"][primary_op]["losses"]["adaptive"] for s in cfg["training_seeds"]],axis=0)
    seed_aggregate={"vs_matched":policy.clustered_ci(aggregate_matched,test_episodes,int(cfg["bootstrap_samples"]),int(cfg["bootstrap_seed"])),
                    "vs_fixed_d1":policy.clustered_ci(aggregate_fixed,test_episodes,int(cfg["bootstrap_samples"]),int(cfg["bootstrap_seed"])),
                    "positive_seed_counts":{"vs_matched":sum(r["benefit_vs_matched"]>0 for r in seed_rows),"vs_fixed_d1":sum(r["benefit_vs_fixed_d1"]>0 for r in seed_rows)}}
    write_json(output/"metrics/seed_aggregation.json",seed_aggregate)

    # Frontier audit uses selected-seed fixed baselines once plus adaptive and
    # calibration-selected mixtures at each operating point.
    frontier=[]
    raw_selected=selected_eval["raw"]
    for depth in (0,1,2,4): frontier.append({"policy":f"fixed_d{depth}","operating_point":"fixed","mean_calls":float(depth),"raw_mse":float(raw_selected[:,depth].mean())})
    for op in records[primary_seed]:
        item=selected_eval["per_op"][op]
        for name in ("adaptive","matched_mixture"):
            depths=item["allocations"][name]; frontier.append({"policy":name,"operating_point":op,"mean_calls":float(depths.mean()),"raw_mse":float(item["losses"][name].mean())})
    flags=policy.nondominated(frontier)
    for row,flag in zip(frontier,flags): row["nondominated"]=flag
    write_csv(output/"metrics/frontier.csv",frontier)

    # Post-hoc physical labels are first read after every allocation is frozen.
    labels=model_io.extract_post_prediction_labels(SOURCE,arrays["episode_id"],arrays["model_step"])
    calmask=arrays["split"]==SPLITS["calibration"]; testmask=arrays["split"]==SPLITS["test"]
    static_threshold=float(np.quantile(labels["block_disp"][calmask],.25)); transport_threshold=float(np.quantile(labels["block_disp"][calmask],.75))
    test_labels={k:v[testmask] for k,v in labels.items()}; regimes=np.full(len(test_target),"free",dtype="U20")
    regimes[test_labels["block_disp"]<=static_threshold]="static"
    regimes[(~test_labels["interaction"])&(test_labels["block_disp"]>transport_threshold)]="transport_free"
    regimes[test_labels["interaction"]]="contact"; regimes[test_labels["impact"]]="impact"
    primary_depth=records[primary_seed][primary_op]["adaptive"]; primary_loss=primary_values["adaptive"]; fixed_loss=primary_values["fixed_d1"]
    regime_rows=[]
    for name in ("impact","contact","transport_free","free","static"):
        mask=regimes==name
        if mask.any(): regime_rows.append({"regime":name,"n":int(mask.sum()),"mean_depth":float(primary_depth[mask].mean()),
            "benefit_vs_fixed_d1":float((fixed_loss[mask]-primary_loss[mask]).mean())})
    write_csv(output/"metrics/posthoc_regimes.csv",regime_rows)
    write_json(output/"metrics/posthoc_thresholds.json",{"static_block_displacement_q25":static_threshold,"transport_block_displacement_q75":transport_threshold})

    selected_threshold=float(records[primary_seed][primary_op]["calibration"]["threshold"])
    latency_result=latency(models[primary_seed],arrays,idx["test"],selected_threshold,cfg,device)
    ref_flops=refiner_flops_per_call(models[primary_seed].refiner); gate_flops=gate_flops_per_decision(models[primary_seed].gate); base_flops=latent_predict_flops()
    adaptive_calls=int(primary_depth.sum()); gate_decisions=latency_result["gate_decisions"]
    compute={"base_latent_predict_flops_per_transition":base_flops,"refiner_flops_per_call":ref_flops,
             "gate_flops_per_decision":gate_flops,"primary_refiner_calls":adaptive_calls,
             "primary_gate_decisions":gate_decisions,"primary_incremental_flops":adaptive_calls*ref_flops+gate_decisions*gate_flops,
             "primary_total_latent_inference_flops":len(primary_depth)*base_flops+adaptive_calls*ref_flops+gate_decisions*gate_flops,
             "latency":latency_result,"flop_convention":"2 FLOPs per multiply-add; norms/activations omitted"}

    # Primary decision from the exact preregistered rule.
    by={(r["comparison"],r["metric"],r["operating_point"]):r for r in all_comparisons}
    p_matched=by[("adaptive_vs_matched_mixture","raw",primary_op)]; p_fixed=by[("adaptive_vs_fixed_d1","raw",primary_op)]
    p_oracle=by[("oracle_advantage_vs_matched","raw",primary_op)]
    depth0=raw_selected[:,0]; depth1=raw_selected[:,1]
    response=policy.clustered_ci(depth0-depth1,test_episodes,int(cfg["bootstrap_samples"]),int(cfg["bootstrap_seed"]))
    adaptive_frontier=any(r["policy"]=="adaptive" and r["nondominated"] for r in frontier)
    exact_primary=all(int(v.sum())==int(primary_depth.sum()) for k,v in selected_eval["per_op"][primary_op]["allocations"].items() if k in ("adaptive","matched_mixture","random_histogram","permutation","oracle"))
    base_unchanged=hashlib.sha256(np.ascontiguousarray(arrays["base_pred"]).tobytes()).hexdigest()==base_hash
    validity={"preregistration_hashes":True,"prior_and_split_isolation":True,"causal_local_features_only":True,
              "test_allocation_frozen_before_outcome":True,"frozen_backbone":True,"base_prediction_cache_unchanged":base_unchanged,
              "finite_metrics":all(math.isfinite(float(r["raw_mse"])) for r in all_policy_rows),"primary_exact_budget":exact_primary,
              "deterministic_policy_replay":np.array_equal(primary_depth,policy.local_depths(test_cache[primary_seed][1],selected_threshold)),
              "posthoc_labels_after_allocation":True}
    if not all(validity.values()): verdict="validity_or_budget_failed"
    elif response["ci_low"]<=0 or p_oracle["ci_low"]<=0: verdict="adaptive_refiner_headroom_failed"
    elif (p_matched["ci_low"]>0 and p_fixed["ci_low"]>0 and adaptive_frontier
          and seed_aggregate["positive_seed_counts"]["vs_matched"]>=2
          and seed_aggregate["positive_seed_counts"]["vs_fixed_d1"]>=2): verdict="adaptive_pareto_gate_passed"
    else: verdict="local_halting_gate_failed"
    wm=by[("adaptive_vs_matched_mixture","whitened",primary_op)]; wf=by[("adaptive_vs_fixed_d1","whitened",primary_op)]
    whitened_strength="strong" if wm["ci_low"]>0 and wf["ci_low"]>0 else "directional" if wm["mean_benefit"]>0 and wf["mean_benefit"]>0 else "failed"
    pm=seed_aggregate["positive_seed_counts"]["vs_matched"]; pf=seed_aggregate["positive_seed_counts"]["vs_fixed_d1"]
    seed_strength="strong" if min(pm,pf)==3 else "majority" if min(pm,pf)>=2 else "failed"

    figure_audit=make_figures(output,all_policy_rows,threshold_rows,seed_rows,regime_rows,frontier)
    source_hashes={p.name:sha256(p) for p in (ROOT/"model.py",ROOT/"policy.py",ROOT/"run_experiment.py")}
    decision={"verdict":verdict,"confirmatory":cfg["mode"]=="full","primary_seed":primary_seed,"primary_operating_point":primary_op,
              "primary":{"adaptive_raw_mse":float(primary_loss.mean()),"matched_raw_mse":float(primary_values["matched_mixture"].mean()),
                         "fixed_d1_raw_mse":float(primary_values["fixed_d1"].mean()),"adaptive_whitened_mse":float(selected_eval["per_op"][primary_op]["white"]["adaptive"].mean()),
                         "adaptive_vs_matched":p_matched,"adaptive_vs_fixed_d1":p_fixed,"oracle_advantage":p_oracle,
                         "fixed_d1_vs_depth0":response,"nondominated_adaptive_point":adaptive_frontier},
              "robustness":{"whitened":whitened_strength,"seed":seed_strength,"seed_aggregation":seed_aggregate},
              "compute":compute,"validity":validity,"allocation_freeze":allocation_freeze,
              "multi_step_rollout":{"status":"unsupported","reason":"existing rollout evaluator does not expose a frozen one-step V3 refiner insertion with unambiguous future action alignment"},
              "provenance":{"frozen_hashes":frozen_hashes,"source_hashes":source_hashes,"cache_sha256":manifest["cache"]["sha256"],"figures":figure_audit}}
    write_json(output/"decision.json",decision); write_json(output/"provenance.json",decision["provenance"])
    write_report(output,decision,all_policy_rows,frontier,seed_rows,regime_rows)
    log(f"complete verdict={verdict}")
    return decision


def fmt_ci(row: Mapping[str,Any]) -> str:
    return f"{row['mean_benefit']:.8g} [{row['ci_low']:.8g}, {row['ci_high']:.8g}]"


def table(rows: Sequence[Mapping[str,Any]], columns: Sequence[str]) -> str:
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for row in rows: lines.append("| "+" | ".join(str(row.get(c,"")) for c in columns)+" |")
    return "\n".join(lines)


def write_report(output: Path, decision: Mapping[str,Any], policy_rows, frontier, seed_rows, regime_rows) -> None:
    p=decision["primary"]; c=decision["compute"]
    report=f"""# LeWM Adaptive Compute V3 Report

## Preregistered verdict

**`{decision['verdict']}`**.

At the primary target mean depth {decision['primary_operating_point']}, the local
adaptive policy realized raw MSE `{p['adaptive_raw_mse']:.9g}` versus
`{p['matched_raw_mse']:.9g}` for the strongest calibration-selected
transition-independent matched-call mixture and `{p['fixed_d1_raw_mse']:.9g}`
for fixed depth 1. Adaptive benefit versus matched was {fmt_ci(p['adaptive_vs_matched'])};
versus fixed depth 1 it was {fmt_ci(p['adaptive_vs_fixed_d1'])}. The oracle
matched-call advantage was {fmt_ci(p['oracle_advantage'])}.

At least one adaptive point nondominated: **{p['nondominated_adaptive_point']}**.
Whitened robustness: **{decision['robustness']['whitened']}**. Seed robustness:
**{decision['robustness']['seed']}**.

## Compute and latency

- Primary realized calls: `{c['primary_refiner_calls']}`.
- Refiner FLOPs/call: `{c['refiner_flops_per_call']}`; gate FLOPs/decision:
  `{c['gate_flops_per_decision']}`.
- Total latent-inference FLOPs including the released LeWM predictor estimate:
  `{c['primary_total_latent_inference_flops']}`.
- MPS end-to-end median latency including released base prediction, gate, and
  refiner: `{c['latency']['median_seconds']:.6g}` seconds for
  `{c['latency']['samples']}` transitions.

## Compute-error frontier

{table(frontier,['policy','operating_point','mean_calls','raw_mse','nondominated'])}

## Primary seed stability

{table(seed_rows,['seed','primary_mean_depth','benefit_vs_matched','benefit_vs_fixed_d1'])}

## Post-hoc physical regimes

These labels were attached after every allocation froze. They are descriptive
correlations and do not make the gate contact-aware.

{table(regime_rows,['regime','n','mean_depth','benefit_vs_fixed_d1'])}

## Validity and caveats

All preregistration/hash, prior-exclusion, split-isolation, local-causality,
frozen-backbone, pre-outcome allocation, finite-metric, deterministic replay,
and primary exact-budget audits recorded in `decision.json` passed unless the
verdict explicitly says `validity_or_budget_failed`.

Original LeWM pretraining episode membership is unavailable. The 95-GB source
is bound by asserted layout/size and selected-cache hashes rather than a
whole-file digest. FLOPs omit normalization/activation scalar operations.
Existing rollout code could not insert V3 without an ambiguous protocol, so
multi-step error is explicitly unsupported. Physical analyses are post hoc.

## Figures

![Pareto](figures/01_compute_error_pareto.png)

![Allocation](figures/02_allocation.png)

![Calibration transfer](figures/03_calibration_transfer.png)

![Seed stability](figures/04_seed_stability.png)

![Regimes](figures/05_posthoc_regimes.png)
"""
    (output/"REPORT.md").write_text(report)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"full_config.json")
    parser.add_argument("--output-dir",type=Path,default=ROOT)
    parser.add_argument("--device",choices=("auto","mps","cpu","cuda"),default="auto")
    parser.add_argument("--force-extract",action="store_true"); parser.add_argument("--force-train",action="store_true")
    return parser.parse_args()


if __name__=="__main__": run(parse_args())
