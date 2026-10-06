"""Fixed-budget, step-wise evaluation using the wheel policy and wrappers."""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

from ..device import resolve_device
from ..envs.pusht_peg import PEG_RADIUS
from ..lewm import load_lewm
from ..normalization import load_normalization
from ..paths import checkpoint_dir, runs_root
from ..runs import create_run
from .conditions import ALL_NAMES, LEGACY_NAMES, PRIMARY_NAMES, jsonable, load_bases, make_conditions, score_trajectory, save_conditions
from .reference import PopulationRecorder, SimulatorCost
from .stats import paired_cluster_bootstrap, wilson_ci


def prepare_output_root(path):
    """Keep complete runs outside Git, including user-specified run roots."""
    root = Path(path).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if root.is_relative_to(repo):
        raise ValueError("probe output cannot be inside the repository")
    try:
        root.relative_to(runs_root().resolve())
    except ValueError:
        raise ValueError("probe output must be under paths.runs_root()") from None
    return root


class PersistedScaler:
    """Reversible wheel Transformable using frozen checkpoint statistics."""

    def __init__(self, stats, column):
        spec = stats["columns"][column]
        self.mean = np.asarray(spec["mean"])
        self.std = np.asarray(spec["std"])

    def transform(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x):
        return (x * self.std + self.mean).astype(np.float32)


class Planner:
    """Reusable model/pool; fresh seeded solver and policy for every episode."""

    def __init__(self, arm, workers=1, population=None, iterations=None, topk=None, device="auto", normalization=None, approach_weight=0, with_target=None):
        if arm != "reference" and approach_weight != 0:
            raise ValueError("approach shaping is reference-only")
        self.approach_weight = approach_weight
        self.with_target = (arm in ("reference", "lewm-pusht")) if with_target is None else with_target
        self.arm = arm
        self.stats = normalization if normalization is not None else load_normalization(
            checkpoint_dir("lewm-pusht" if arm == "reference" else arm))
        self.population = population or (100 if arm == "reference" else 300)
        self.iterations = iterations or (10 if arm == "reference" else 30)
        self.topk = topk or min(10 if arm == "reference" else 30, self.population)
        if self.population < 2 or not 2 <= self.topk <= self.population or self.iterations < 1:
            raise ValueError("CEM needs population >= topk >= 2 and iterations >= 1")
        self.device = torch.device("cpu") if arm == "reference" else resolve_device(device)
        self.model = None if arm == "reference" else load_lewm(checkpoint_dir(arm), self.device)
        self.workers = workers
        self.cost = None
        self.recorder = PopulationRecorder()
        self.transform = transforms.Compose([
            transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size=224),
        ])

    def start(self, condition, envs, seed):
        if self.arm == "reference":
            if self.cost is None:
                self.cost = SimulatorCost(condition, self.stats, self.workers, self.approach_weight)
            self.cost.condition = condition
            self.cost.snapshot = condition.start_snapshot
            self.cost.prior_max = 0
        solver = swm.solver.CEMSolver(self.cost if self.arm == "reference" else self.model,
                                      num_samples=self.population, n_steps=self.iterations,
                                      topk=self.topk, device=self.device, seed=seed,
                                      callbacks=[self.recorder])
        process = {k: PersistedScaler(self.stats, "proprio" if "proprio" in k else "action")
                   for k in ("action", "proprio", "goal_proprio")}
        # This is upstream pusht.yaml's history_len=1. LeWM grows its
        # latent context up to predictor.num_frames=3 inside rollout().
        self.policy = swm.policy.WorldModelPolicy(
            solver, swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
            process=process, transform={"pixels": self.transform, "goal": self.transform})
        self.policy.set_env(envs)

    def close(self):
        if self.cost:
            self.cost.close()


def episode_seed(seed, base_id, condition):
    digest = hashlib.sha256(f"{seed}:{base_id}:{condition}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def controlled_steps(world, planner, budget, before_plan=None, fixed_goal=None, start_step=0):
    """Shared full-budget loop for probe episodes and the no-peg fidelity gate."""
    for step in range(start_step, budget):
        planning = step % 25 == 0
        if planning and before_plan:
            before_plan()
        before = time.perf_counter()
        action = planner.policy.get_action(world.infos)
        if planning and planner.device.type == "mps":
            torch.mps.synchronize()
        planning_s = time.perf_counter() - before if planning else None
        _, _, _, _, world.infos = world.envs.step(action)
        if fixed_goal:
            world.infos.update({k: v.copy() for k, v in fixed_goal.items()})
        yield step, action, planning_s


def run_episode(base, condition, planner, seed, population_path=None, budget=50, prefix_record=None, prefix_population=None):
    """Execute the fixed budget, irrespective of env success/termination."""
    started = time.perf_counter()
    world = swm.World("swm/PushTPeg-v1", num_envs=1, image_shape=(224, 224),
                      max_episode_steps=max(100, budget + 1), with_target=planner.with_target,
                      render_target_pose=(256, 256, np.pi / 4), peg_radius=base.peg_radius)
    try:
        # Wheel MegaWrapper + EnvPool construct the exact (env,time,...) info
        # layout and action-history convention consumed by WorldModelPolicy.
        bodies = condition.start_snapshot["bodies"]
        state = [*bodies["agent"]["position"], *bodies["block"]["position"],
                 bodies["block"]["angle"], *bodies["agent"]["velocity"], *bodies["peg"]["position"]]
        world.envs.single_action_space.seed(seed)
        world.reset(seed=base.seed, options={"state": state, "goal_state": condition.goal_state})
        raw = world.envs.envs[0].unwrapped
        raw.restore_snapshot(condition.start_snapshot)
        raw.set_goal(condition.goal_state)
        planner.start(condition, world.envs, seed)
        trajectory = [raw._get_obs().copy()]
        actions, planning_times, populations = [], [], []
        checkpoints = {}
        start_step, prefix_wall = 0, 0.0
        if prefix_record is not None:
            if planner.arm != "reference" or budget != 100 or len(prefix_record["actions"]) != 50:
                raise ValueError("only reference 50-to-100 extensions are supported")
            # Replaying physical actions retains live contact-cache state. A
            # snapshot at step 50 would not. Full horizon execution leaves no
            # warm-start tail; only the wheel CEM generator must be advanced.
            for action in prefix_record["actions"]:
                _, _, _, _, world.infos = world.envs.step(np.asarray([action], dtype=np.float32))
                trajectory.append(raw._get_obs().copy())
                actions.append(action)
            if not np.array_equal(trajectory, prefix_record["trajectory"]):
                raise ValueError("reference prefix replay differs")
            for _ in range(2 * planner.iterations):
                torch.randn(1, planner.population, 5, 10,
                            generator=planner.policy.solver.torch_gen, dtype=torch.float32)
            planning_times = list(prefix_record["planning_times_s"])
            checkpoints = dict(prefix_record["budget_checkpoints"])
            prefix_wall = prefix_record["wall_s"]
            start_step = 50
            if prefix_population is not None:
                with np.load(prefix_population) as bank:
                    populations = [{k: torch.from_numpy(bank[f"call{i}_{k}"].copy())
                        for k in ("candidates", "costs", "topk_candidates", "topk_vals")} for i in range(2)]
        def before_plan():
            if planner.cost:
                planner.cost.snapshot = jsonable(raw.get_snapshot())
                planner.cost.prior_max = max(np.linalg.norm(s[7:9] - condition.scoring_spec["peg_start"])
                                            for s in trajectory)

        for step, action, planning_s in controlled_steps(world, planner, budget, before_plan, start_step=start_step):
            if planning_s is not None:
                planning_times.append(planning_s)
                populations.append(planner.recorder.population)
            trajectory.append(raw._get_obs().copy())
            actions.append(action[0].tolist())
            if step + 1 in (50, 100):
                checkpoints[str(step + 1)] = {"wall_s": prefix_wall + time.perf_counter() - started,
                                            "score": score_trajectory(trajectory, condition)}
        if population_path is not None:
            np.savez_compressed(population_path, **{f"call{i}_{k}": v.numpy()
                                                   for i, p in enumerate(populations) for k, v in p.items()})
        return {"base_id": base.id, "condition": condition.name, "arm": planner.arm,
                "seed": seed, "trajectory": jsonable(trajectory), "actions": actions,
                "budget_checkpoints": checkpoints,
                "planning_times_s": planning_times, "wall_s": prefix_wall + time.perf_counter() - started,
                "score": score_trajectory(trajectory, condition)}
    finally:
        world.close()


def summarize(records, seed):
    result = {"episodes": len(records), "conditions": {}, "contrasts": {}}
    for name in ALL_NAMES:
        rows = [r for r in records if r["condition"] == name]
        if not rows:
            continue
        successes = sum(r["score"]["success"] for r in rows)
        result["conditions"][name] = {"n": len(rows), "success_rate": successes / len(rows),
                                     "wilson_ci": wilson_ci(successes, len(rows)),
                                     "mean_wall_s": float(np.mean([r["wall_s"] for r in rows])),
                                     "t_success_rate": float(np.mean([r["score"]["t_success"] for r in rows]))}
    for label, hard, control in (("D", "on_path", "off_path"), ("D_prime", "near_path", "off_path"), ("G", "move_peg", "move_T_matched")):
        a = {r["base_id"]: r for r in records if r["condition"] == hard}
        b = {r["base_id"]: r for r in records if r["condition"] == control}
        ids = sorted(a.keys() & b.keys())
        if ids:
            result["contrasts"][label] = paired_cluster_bootstrap(
                [a[i]["score"]["success"] for i in ids], [b[i]["score"]["success"] for i in ids], ids, seed)
            result["contrasts"][label]["discordance"] = float(np.mean([
                a[i]["score"]["success"] != b[i]["score"]["success"] for i in ids]))
            if label in ("D", "D_prime"):
                result["contrasts"][label + "_disturbance_allowed"] = paired_cluster_bootstrap(
                    [a[i]["score"]["t_success"] for i in ids], [b[i]["score"]["t_success"] for i in ids], ids, seed)
    return result


def evaluate(arm, bases_path, conditions="all", seed=42, n=100, run_dir=None,
             feasibility_run=None, normalization=None, budget=50, displacement_range=(40, 100), prefix_run=None, **planner_options):
    prepare_output_root(runs_root())
    if run_dir is not None:
        run_dir = prepare_output_root(run_dir)
    if budget not in (50, 100):
        raise ValueError("budget must be 50 or 100")
    bases = load_bases(bases_path)[:n]
    defaults = PRIMARY_NAMES if bases and all("near_path" in (b.rollout_tasks or {}) for b in bases) else LEGACY_NAMES
    names = list(defaults) if conditions == "all" else conditions.split(",")
    if not bases or any(name not in ALL_NAMES for name in names):
        raise ValueError("nonempty bases and valid conditions required")
    if len({b.peg_radius for b in bases}) != 1:
        raise ValueError("one peg radius required per evaluation run")
    with_target = planner_options.get("with_target")
    if with_target is None:
        with_target = arm in ("reference", "lewm-pusht")
    planner_options["with_target"] = with_target
    config = {"arm": arm, "bases_sha256": hashlib.sha256(Path(bases_path).read_bytes()).hexdigest(),
              "seed": seed, "n": n, "conditions": names, "budget": budget, "peg_radius": bases[0].peg_radius,
              "displacement_range": list(displacement_range),
              "observation_rendering": "upstream-fixed-target" if with_target else "no-target",
              "feasibility_run": str(feasibility_run) if feasibility_run else None, **planner_options}
    if normalization is None:
        normalization = load_normalization(checkpoint_dir("lewm-pusht" if arm == "reference" else arm))
    config["normalization_sha256"] = hashlib.sha256(
        json.dumps(normalization, sort_keys=True).encode()).hexdigest()
    if arm != "reference":
        config["checkpoint_weights_sha256"] = hashlib.sha256(
            (checkpoint_dir(arm) / "weights.pt").read_bytes()).hexdigest()
    feasibility_config, feasibility = None, None
    if feasibility_run:
        feasibility_config = json.loads((Path(feasibility_run) / "config.json").read_text())
        if (feasibility_config["arm"] != "reference" or feasibility_config["seed"] == seed
                or feasibility_config["bases_sha256"] != config["bases_sha256"]
                or feasibility_config.get("peg_radius", PEG_RADIUS) != config["peg_radius"]
                or any(feasibility_config.get(k) != config[k] for k in
                       ("budget", "displacement_range", "normalization_sha256"))):
            raise ValueError("feasibility must use reference, the same bases and protocol, and a separate seed F")
        if arm == "reference" and any(feasibility_config.get(k, 0 if k == "approach_weight" else None)
                != config.get(k, 0 if k == "approach_weight" else None)
                for k in ("population", "iterations", "topk", "approach_weight")):
            raise ValueError("reference evaluation must match feasibility planner and shaping")
        feasibility = [json.loads(line) for line in
                       (Path(feasibility_run) / "episodes.jsonl").read_text().splitlines()]
        present = {(r["base_id"], r["condition"]) for r in feasibility}
        if any((b.id, name) not in present for b in bases for name in names):
            raise ValueError("feasibility reference run must cover all selected bases/conditions")
    prefixes = {}
    if prefix_run:
        prefix_config = json.loads((Path(prefix_run)/"config.json").read_text())
        prefix_config["peg_radius"] = prefix_config.get("peg_radius", PEG_RADIUS)
        if arm != "reference" or budget != 100 or prefix_config["budget"] != 50:
            raise ValueError("only reference 50-to-100 extensions are supported")
        if any(prefix_config.get(k) != v for k,v in config.items() if k not in ("budget", "feasibility_run")):
            raise ValueError("reference prefix config differs")
        prefixes = {(r["base_id"],r["condition"]):r for r in
                    map(json.loads,(Path(prefix_run)/"episodes.jsonl").read_text().splitlines())}
        config["prefix_run"] = str(prefix_run)
    run_dir = Path(run_dir) if run_dir else create_run("probe-eval", config)
    run_dir = prepare_output_root(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        previous["peg_radius"] = previous.get("peg_radius", PEG_RADIUS)
        if previous != config:
            raise ValueError("resume configuration differs")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (run_dir / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    path = run_dir / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed = {(r["base_id"], r["condition"]) for r in records}
    planner = Planner(arm, normalization=normalization, **planner_options)
    try:
        for base in bases:
            matched = make_conditions(base, displacement_range, with_target=planner.with_target)
            for name in names:
                if (base.id, name) in completed:
                    continue
                stem = hashlib.sha256(f"{base.id}:{name}".encode()).hexdigest()[:16]
                save_conditions({name: matched[name]}, run_dir / f"{stem}_condition.json")
                record = run_episode(base, matched[name], planner, episode_seed(seed, base.id, name),
                                     run_dir / f"{stem}_population.npz", budget=budget,
                                     prefix_record=prefixes.get((base.id,name)),
                                     prefix_population=Path(prefix_run)/f"{stem}_population.npz" if prefix_run else None)
                with path.open("a") as file:
                    file.write(json.dumps(record) + "\n")
                records.append(record)
    finally:
        planner.close()
    summary = {"run_dir": str(run_dir), "unconditional": summarize(records, seed)}
    if feasibility_run:
        summary["common_feasible"] = {}
        for label, pair in (("D", ("off_path", "on_path")), ("D_prime", ("off_path", "near_path")), ("G", ("move_peg", "move_T_matched"))):
            if not all(name in names for name in pair):
                continue
            solved = [{r["base_id"] for r in feasibility
                       if r["condition"] == name and r["score"]["success"]} for name in pair]
            common = set.intersection(*solved) & {b.id for b in bases}
            summary["common_feasible"][label] = {
                "included_base_ids": sorted(common), "excluded": len(bases) - len(common),
                "feasibility_seed": feasibility_config["seed"],
                **summarize([r for r in records if r["base_id"] in common and r["condition"] in pair], seed)}
    else:
        summary["common_feasible"] = None
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
