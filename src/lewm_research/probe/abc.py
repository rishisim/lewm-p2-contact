"""Shared physical candidate banks and predicted/observed/role cost decomposition."""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.stats import spearmanr
import torch

from ..envs.pusht_peg import PEG_RADIUS, PushTPeg
from ..paths import checkpoint_dir
from .analysis import checkpoint_names, cluster_mean, fingerprint, open_run, write_json, write_npz
from .conditions import NAMES, load_bases, make_conditions, score_trajectory
from .readout import DEFAULT_CHECKPOINTS
from .rollout_eval import Planner, episode_seed, run_episode


def bank_candidates(population, costs, normalization, seed, low=-1, high=1):
    """16 cost-ordered unique reference members + 16 perturbed + 32 uniform.

    Reference members retain the wheel's unbounded Gaussian coordinates.
    Perturbations are clipped to the env range; uniform members lie in it.
    sigma * normalized range converts to sigma * physical range, per axis.
    PushT uses relative env controls in [-1,1] (100 px per action unit),
    not absolute screen coordinates. Normalization is a separate transform.
    """
    population = np.asarray(population, dtype=np.float32).reshape(-1, 25, 2)
    spec = normalization["columns"]["action"]
    std, mean = np.asarray(spec["std"]), np.asarray(spec["mean"])
    physical = (population * std + mean).astype(np.float32)
    members, seen = [], set()
    for i in np.argsort(np.asarray(costs).reshape(-1), kind="stable"):
        key = physical[i].tobytes()
        if key not in seen:
            seen.add(key); members.append(physical[i])
        if len(members) == 16:
            break
    if len(members) < 16:
        raise ValueError("reference final population has fewer than 16 unique candidates")
    rng = np.random.default_rng(seed)
    reference = np.stack(members)
    normalized = (reference - mean) / std
    sigmas = np.resize(np.array([.1, .3, .6]), 16)
    noise = rng.normal(size=reference.shape) * sigmas[:, None, None] * ((high-low) / std)
    perturbed = np.clip((normalized + noise) * std + mean, low, high).astype(np.float32)
    uniform = rng.uniform(low, high, (32, 25, 2)).astype(np.float32)
    return np.concatenate([reference, perturbed, uniform])


def execute_bank(actions, condition):
    env = PushTPeg(with_target=False, render_target_pose=(256,256,np.pi/4), peg_radius=condition.peg_radius)
    try:
        env.reset(seed=0)
        trajectories, scores = [], []
        for sequence in actions:
            env._setup()
            env.restore_snapshot(condition.start_snapshot)
            states = [env._get_obs().copy()]
            for action in sequence:
                states.append(env.step(action)[0]["state"].copy())
            trajectories.append(states)
            scores.append(score_trajectory(states, condition))
        return {"actions": np.asarray(actions), "trajectories": np.asarray(trajectories),
                "C": np.array([s["role_cost"] for s in scores]),
                "C_end": np.array([s["role_cost_endpoint"] for s in scores]),
                "success": np.array([s["success"] for s in scores])}
    finally:
        env.close()


def bank_exclusions(bank):
    reasons = []
    if not np.any(bank["success"]):
        reasons.append("no_successful_candidate")
    if np.all(bank["success"]):
        reasons.append("no_failed_candidate")
    if np.ptp(bank["C"]) <= 20:
        reasons.append("C_range_not_above_20")
    return reasons


def abc_metrics(a, b, c):
    from .stats import top1_regret
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    if a.ndim != 1 or a.shape != b.shape or a.shape != c.shape or len(a) < 2:
        raise ValueError("at least two aligned candidate costs required")
    if not all(np.isfinite(v).all() for v in (a, b, c)):
        raise ValueError("candidate costs must be finite")
    def rho(x, y):
        return float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else None
    return {"AC": rho(a, c), "BC": rho(b, c), "AB": rho(a, b),
            "A_regret": top1_regret(a, c), "B_regret": top1_regret(b, c)}


def load_bank(path):
    # No checkpoint argument: physical bank identity is independent of models.
    with np.load(path, allow_pickle=False) as file:
        return {k: file[k].copy() for k in file.files}


def model_costs(planner, bank, condition, batch_size=16):
    """A uses wheel get_cost; B uses wheel criterion on real 15/20/25 frames.

    Planning starts with history_len=1 and grows up to predictor.num_frames.
    Real history is sampled every five env steps, matching that latent history.
    LeWM's encoder is framewise, so B's criterion selects the real last frame.
    """
    env = PushTPeg(resolution=224, with_target=planner.with_target,
                   render_target_pose=(256,256,np.pi/4), peg_radius=condition.peg_radius)
    try:
        env.reset(seed=0)
        def image(state):
            env._set_state(state)
            return planner.transform(torch.from_numpy(env._render_frame("rgb_array").copy()).permute(2,0,1))
        initial = image(bank["trajectories"][0,0]).to(planner.device)
        goal = image(condition.goal_state).to(planner.device)
        spec = planner.stats["columns"]["action"]
        normalized = (bank["actions"] - np.asarray(spec["mean"])) / np.asarray(spec["std"])
        a, b = [], []
        with torch.inference_mode():
            for offset in range(0, len(normalized), batch_size):
                count = len(normalized[offset:offset+batch_size])
                action = torch.as_tensor(normalized[offset:offset+batch_size], dtype=torch.float32,
                                         device=planner.device).reshape(1,count,5,10)
                info = {"pixels": initial[None,None,None].expand(1,count,1,-1,-1,-1),
                        "goal": goal[None,None,None].expand(1,count,1,-1,-1,-1),
                        "action": torch.zeros(1,count,1,10,device=planner.device)}
                # get_cost/criterion intentionally sum elementwise MSE, exactly
                # as the pinned wheel does (do not substitute a feature mean).
                a.extend(planner.model.get_cost(info, action)[0].cpu().tolist())
                history = torch.stack([torch.stack([image(state) for state in trajectory[[15,20,25]]])
                    for trajectory in bank["trajectories"][offset:offset+count]]).to(planner.device)
                real = planner.model.encode({"pixels": history})["emb"]
                cost_info = {"predicted_emb": real[None], "goal_emb": info["goal_emb"]}
                b.extend(planner.model.criterion(cost_info)[0].cpu().tolist())
        return np.asarray(a), np.asarray(b)
    finally:
        env.close()


def aggregate_metrics(records, seed=0, samples=2000):
    result = {}
    for name in sorted({r["condition"] for r in records}):
        rows = [r for r in records if r["condition"] == name]
        result[name] = {key: cluster_mean([np.nan if r["metrics"][key] is None else r["metrics"][key]
                                         for r in rows], [r["base_id"] for r in rows], seed, samples)
                        for key in ("AC", "BC", "AB", "A_regret", "B_regret")}
    return result


def run_abc(bases_path, checkpoints=DEFAULT_CHECKPOINTS, conditions="all", n=50,
            seed=42, run_dir=None, device="auto", workers=4, population=300,
            iterations=30, topk=30, approach_weight=.1, batch_size=16,
            feasibility_run=None, samples=2000):
    names = list(NAMES) if conditions == "all" else conditions.split(",")
    checkpoints = checkpoint_names(checkpoints)
    if (n < 1 or population < 16 or batch_size < 1 or not names or len(names)!=len(set(names))
            or any(name not in NAMES for name in names)):
        raise ValueError("valid conditions, n/batch > 0 and population >= 16 required")
    bases = load_bases(bases_path)
    if len({b.peg_radius for b in bases}) > 1:
        raise ValueError("one peg radius required per ABC run")
    peg_radius = bases[0].peg_radius if bases else PEG_RADIUS
    eligible = None
    if feasibility_run:
        froot = Path(feasibility_run)
        fc = json.loads((froot/"config.json").read_text())
        if fc["arm"] != "reference" or fc["bases_sha256"] != fingerprint(bases_path) or fc.get("peg_radius", PEG_RADIUS) != peg_radius:
            raise ValueError("feasibility must be reference on identical bases")
        rows = [json.loads(line) for line in (froot/"episodes.jsonl").read_text().splitlines()]
        solved = [{r["base_id"] for r in rows if r["condition"] == name and r["score"]["success"]} for name in names]
        eligible = set.intersection(*solved)
        bases = [b for b in bases if b.id in eligible]
    bases = bases[:n]
    action_env = PushTPeg(peg_radius=peg_radius)
    low, high = action_env.action_space.low.copy(), action_env.action_space.high.copy()
    action_env.close()
    config = {"stage": "probe-abc", "seed": seed, "checkpoints": list(checkpoints),
              "checkpoint_sha256": {name: fingerprint(checkpoint_dir(name)/"weights.pt") for name in checkpoints},
              "checkpoint_normalization_sha256": {name: fingerprint(checkpoint_dir(name)/"normalization.json") for name in checkpoints},
              "bases_sha256": fingerprint(bases_path), "base_ids": [b.id for b in bases],
              "conditions": names, "n": n, "K": 64, "horizon_blocks": 5, "peg_radius": peg_radius,
              "device": device, "workers": workers, "population": population,
              "iterations": iterations, "topk": topk, "approach_weight": approach_weight,
              "batch_size": batch_size, "bootstrap_samples": samples,
              "physical_action_low": low.tolist(), "physical_action_high": high.tolist(),
              "action_semantics": "relative env controls; 100 px per unit",
              "normalization_sha256": fingerprint(checkpoint_dir("lewm-pusht")/"normalization.json"),
              "feasibility_run": str(feasibility_run) if feasibility_run else None,
              "feasibility_sha256": fingerprint(Path(feasibility_run)/"episodes.jsonl") if feasibility_run else None}
    root = open_run("probe-abc", config, run_dir)
    bank_root = root / "banks"
    bank_root.mkdir(exist_ok=True)
    started = time.perf_counter()
    reference = Planner("reference", workers=workers, population=population,
                        iterations=iterations, topk=topk, approach_weight=approach_weight)
    bank_info = []
    try:
        for base in bases:
            matched = make_conditions(base, render_images=False)
            if missing := set(names)-set(matched):
                raise ValueError(f"base {base.id} lacks requested conditions: {sorted(missing)}")
            for name in names:
                condition = matched[name]
                stem = hashlib.sha256(f"{base.id}:{name}".encode()).hexdigest()[:16]
                path, meta_path = bank_root/f"{stem}.npz", bank_root/f"{stem}.json"
                if not meta_path.exists():
                    before = time.perf_counter()
                    bank_seed = episode_seed(seed, base.id, name)
                    record = run_episode(base, condition, reference, bank_seed,
                                         population_path=bank_root/f"{stem}_reference.npz", budget=25)
                    population_record = reference.recorder.population
                    actions = bank_candidates(population_record["candidates"].numpy(),
                        population_record["costs"].numpy(), reference.stats, bank_seed, low, high)
                    bank = execute_bank(actions, condition)
                    write_npz(path, **bank)
                    info = {"base_id": base.id, "condition": name, "seed": bank_seed,
                            "bank": str(path), "sha256": fingerprint(path),
                            "C_range": float(np.ptp(bank["C"])), "successful_candidates": int(bank["success"].sum()),
                            "exclusions": bank_exclusions(bank), "wall_s": time.perf_counter()-before,
                            "reference_planning_s": sum(record["planning_times_s"])}
                    write_json(meta_path, info)
                saved = json.loads(meta_path.read_text())
                if saved["base_id"] != base.id or saved["condition"] != name or fingerprint(path) != saved["sha256"]:
                    raise ValueError("persisted bank identity changed")
                bank_info.append(saved)
    finally:
        reference.close()
    records = []
    for name in checkpoints:
        output = root / f"{name}.json"
        existing = json.loads(output.read_text()) if output.exists() else []
        completed = {(r["base_id"],r["condition"]) for r in existing}
        todo = [info for info in bank_info if not info["exclusions"]
                and (info["base_id"],info["condition"]) not in completed]
        if todo:
            planner = Planner(name, device=device)
            try:
                base_map = {b.id: b for b in bases}
                for info in todo:
                    before = time.perf_counter()
                    bank = load_bank(info["bank"])
                    if fingerprint(info["bank"]) != info["sha256"]:
                        raise ValueError("persisted physical bank changed")
                    condition = make_conditions(base_map[info["base_id"]], render_images=False,
                                                with_target=planner.with_target)[info["condition"]]
                    a, b = model_costs(planner, bank, condition, batch_size)
                    stem = Path(info["bank"]).stem
                    write_npz(root/f"{stem}_{name}_costs.npz", A=a, B=b, C=bank["C"], C_end=bank["C_end"])
                    existing.append({"checkpoint": name, "base_id": info["base_id"], "condition": info["condition"],
                        "bank_sha256": info["sha256"], "with_target": planner.with_target,
                        "metrics": abc_metrics(a, b, bank["C"]), "endpoint_metrics": abc_metrics(a,b,bank["C_end"]),
                        "wall_s": time.perf_counter()-before})
                    write_json(output, existing)
            finally:
                planner.close()
        if not output.exists():
            write_json(output, existing)
        records.extend(existing)
    result = {"stage": "probe-abc", "run_dir": str(root), "seed": seed, "banks": bank_info,
              "requested_bases": n, "selected_bases": len(bases),
              "feasibility_selected": feasibility_run is not None,
              "feasibility_eligible_bases": len(eligible) if eligible is not None else None,
              "included_banks": sum(not b["exclusions"] for b in bank_info),
              "excluded_banks": sum(bool(b["exclusions"]) for b in bank_info),
              "records": records, "checkpoints": {name: aggregate_metrics(
                  [r for r in records if r["checkpoint"] == name], seed, samples) for name in checkpoints},
              "invocation_wall_s": time.perf_counter()-started}
    write_json(root/"abc.json", result)
    return result
