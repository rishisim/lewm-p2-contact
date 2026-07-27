#!/usr/bin/env python3
"""Freeze Task D cohort and provenance before any Task D outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "le-wm"))
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
import eval as lewm_eval

from task_d import canonical_bytes, file_sha256, freeze_cohort, validate_cohort


def main() -> None:
    config_path = ROOT / "task_d_config.json"
    config = json.loads(config_path.read_text())
    task_a = json.loads((ROOT / "config.json").read_text())["planner_qualification"]
    eval_cfg = OmegaConf.load(REPO / "le-wm/config/eval/pusht.yaml")
    dataset = lewm_eval.get_dataset(eval_cfg, task_a["dataset"]["name"])
    selection = config["execution"]
    cohort = freeze_cohort(
        dataset,
        lewm_eval,
        {
            "task_a": task_a,
            "cohort_seed": selection["cohort_seed"],
            "pilot_starts": selection["pilot_starts"],
            "sealed_starts": selection["sealed_starts"],
        },
    )
    validate_cohort(cohort, selection["pilot_starts"], selection["sealed_starts"])
    cohort["provenance"] = {
        "dataset": task_a["dataset"]["name"],
        "dataset_path": str(dataset.uri if hasattr(dataset, "uri") else dataset),
        "source_commit": "f5f1cf1a03c0c3f302b159e253181bcc1923e0a7",
        "task_a_config_sha256": file_sha256(ROOT / "config.json"),
        "task_a_results_sha256": file_sha256(ROOT / "results.json"),
        "task_b_results_sha256": file_sha256(ROOT / "task_b_results.json"),
        "task_d_config_sha256": file_sha256(config_path),
        "task_d_protocol_sha256": file_sha256(ROOT / "TASK_D_PROTOCOL.md"),
    }
    output = ROOT / "task_d_cohort.json"
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != cohort:
            raise RuntimeError("refusing to overwrite a different frozen Task D cohort")
    else:
        output.write_bytes(canonical_bytes(cohort))
    print(json.dumps({"cohort": str(output), "sha256": file_sha256(output)}))


if __name__ == "__main__":
    main()
