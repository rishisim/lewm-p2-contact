#!/usr/bin/env python3
"""Freeze Task E development starts and provenance before diagnostic outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "le-wm"))
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
import eval as lewm_eval

from task_d import canonical_bytes, file_sha256, freeze_cohort, validate_cohort


def main() -> None:
    config_path = ROOT / "task_e_config.json"
    protocol_path = ROOT / "TASK_E_PROTOCOL.md"
    config = json.loads(config_path.read_text())
    task_a = json.loads((ROOT / "config.json").read_text())["planner_qualification"]
    eval_cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    dataset = lewm_eval.get_dataset(eval_cfg, task_a["dataset"]["name"])
    selection = config["execution"]
    oversized = freeze_cohort(
        dataset,
        lewm_eval,
        {
            "task_a": task_a,
            "cohort_seed": selection["cohort_seed"],
            "pilot_starts": selection["pilot_starts"],
            "sealed_starts": selection["sealed_starts"] + 64,
        },
    )
    task_d = json.loads((ROOT / "task_d_cohort.json").read_text())
    prior_d_episodes = {
        int(row["episode_id"]) for role in ("pilot", "sealed") for row in task_d[role]
    }
    chosen = [
        row
        for row in oversized["pilot"] + oversized["sealed"]
        if int(row["episode_id"]) not in prior_d_episodes
    ][: selection["pilot_starts"] + selection["sealed_starts"]]
    if len(chosen) != selection["pilot_starts"] + selection["sealed_starts"]:
        raise RuntimeError("insufficient Task-D-disjoint starts")
    payload = {
        "schema_version": 1,
        "status": "frozen_before_task_e_outcomes",
        "selection": {
            "unit": "unique_source_episode",
            "cohort_seed": selection["cohort_seed"],
            "goal_offset_steps": 25,
            "no_outcome_replacement": True,
        },
        "exclusions": oversized["exclusions"]
        | {"task_d_episodes": sorted(prior_d_episodes)},
        "pilot": chosen[: selection["pilot_starts"]],
        "sealed": chosen[selection["pilot_starts"] :],
        "candidate_banks": {
            "pilot_seeds": selection["pilot_candidate_seeds"],
            "sealed_seeds": selection["sealed_candidate_seeds"],
            "call_seed_contract": "task_c_population.call_seed(candidate_seed,row_id,0)",
            "primary": config["candidate_bank"],
            "secondary": config["secondary_bank"],
        },
        "reference_replay": {
            "subset_per_pilot_bank": selection["replay_subset_per_pilot_bank"],
            "subset_seed": selection["replay_subset_seed"],
            "action_equality": "exact",
            "state_and_cost_absolute_tolerance": 1e-7,
        },
        "provenance": {
            "source_commit": "3436fff1daee93aee590acfaa3a543eebdc58dce",
            "dataset": task_a["dataset"]["name"],
            "dataset_path": str(dataset.uri if hasattr(dataset, "uri") else dataset),
            "task_a_config_sha256": file_sha256(ROOT / "config.json"),
            "task_b_results_sha256": file_sha256(ROOT / "task_b_results.json"),
            "task_c_config_sha256": file_sha256(ROOT / "task_c_config.json"),
            "task_d_cohort_sha256": file_sha256(ROOT / "task_d_cohort.json"),
            "task_d_results_sha256": file_sha256(ROOT / "task_d_results.json"),
            "task_e_config_sha256": file_sha256(config_path),
            "task_e_protocol_sha256": file_sha256(protocol_path),
        },
    }
    payload["episode_set_sha256"] = __import__("hashlib").sha256(
        np.asarray(
            [row["episode_id"] for row in payload["pilot"] + payload["sealed"]],
            dtype="<i8",
        ).tobytes()
    ).hexdigest()
    validate_cohort(
        payload, selection["pilot_starts"], selection["sealed_starts"]
    )
    output = ROOT / "task_e_manifest.json"
    if output.exists() and json.loads(output.read_text()) != payload:
        raise RuntimeError("refusing to overwrite a different frozen Task E manifest")
    if not output.exists():
        output.write_bytes(canonical_bytes(payload))
    print(json.dumps({"manifest": str(output), "sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
