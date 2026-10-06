"""Independent timing/feasibility pilot; does not choose evaluation N."""

import json
import hashlib
from pathlib import Path
import numpy as np

from ..runs import create_run
from ..normalization import load_normalization
from ..paths import checkpoint_dir, runs_root
from .conditions import generate_bases, save_bases
from .rollout_eval import evaluate, prepare_output_root
from .stats import paired_cluster_bootstrap

PILOT_SEED = 1_000_000_000


def pilot(checkpoint="lewm-pusht", workers=4, seed=PILOT_SEED, n=10,
          run_dir=None, reference_population=100, reference_iterations=10,
          lewm_population=300, lewm_iterations=30, device="auto"):
    if seed < PILOT_SEED:
        raise ValueError("pilot seed must be in reserved range >= 1000000000")
    prepare_output_root(runs_root())
    config = dict(checkpoint=checkpoint, workers=workers, seed=seed, n=n,
                  reference_population=reference_population, reference_iterations=reference_iterations,
                  lewm_population=lewm_population, lewm_iterations=lewm_iterations, device=device)
    root = Path(run_dir) if run_dir else create_run("probe-pilot", config)
    root = prepare_output_root(root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "pilot_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("pilot resume configuration differs")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    bases_path = root / "bases.json"
    if not bases_path.exists():
        save_bases(generate_bases(n, seed), bases_path)
    frozen_path = root / "normalization.json"
    if not frozen_path.exists():
        prior = root / "reference" / "normalization.json"
        frozen = json.loads(prior.read_text()) if prior.exists() else load_normalization(checkpoint_dir("lewm-pusht"))
        frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    frozen = json.loads(frozen_path.read_text())
    results = {}
    for arm, population, iterations in (("reference", reference_population, reference_iterations),
                                         (checkpoint, lewm_population, lewm_iterations)):
        results[arm] = evaluate(arm, bases_path, seed=seed, n=n, run_dir=root / arm,
                                normalization=frozen, workers=workers, population=population, iterations=iterations, device=device)
    records = {arm: [json.loads(line) for line in (root / arm / "episodes.jsonl").read_text().splitlines()]
               for arm in results}
    for arm, rows in records.items():
        wall = [r["wall_s"] for r in rows]
        results[arm]["episode_wall_times"] = [
            {"base_id": r["base_id"], "condition": r["condition"],
             "wall_s": r["wall_s"], "planning_times_s": r["planning_times_s"]} for r in rows]
        results[arm]["wall_time_summary_s"] = {
            "mean": float(np.mean(wall)), "median": float(np.median(wall)),
            "p90": float(np.quantile(wall, 0.9)), "max": float(np.max(wall))}
    paired = {}
    for condition in results["reference"]["unconditional"]["conditions"]:
        reference = {r["base_id"]: r for r in records["reference"] if r["condition"] == condition}
        learned = {r["base_id"]: r for r in records[checkpoint] if r["condition"] == condition}
        ids = sorted(reference.keys() & learned.keys())
        a = [learned[i]["score"]["success"] for i in ids]
        b = [reference[i]["score"]["success"] for i in ids]
        paired[condition] = {**paired_cluster_bootstrap(a, b, ids, seed),
                             "discordance": float(np.mean(np.asarray(a) != b))}
    common = {}
    for contrast, names in (("D", ("off_path", "on_path")), ("G", ("move_peg", "move_T_matched"))):
        feasible = set.intersection(*[{r["base_id"] for r in records["reference"]
                                      if r["condition"] == name and r["score"]["success"]} for name in names])
        common[contrast] = {"base_ids": sorted(feasible), "n": len(feasible), "excluded": n - len(feasible)}
    result = {"run_dir": str(root), "config": config, "arms": results,
              "normalization": {"path": str(frozen_path), "method": frozen["method"],
                                "sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()},
              "paired_arms": paired, "reference_common_feasible": common,
              "evaluation_n_frozen": False}
    (root / "pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
