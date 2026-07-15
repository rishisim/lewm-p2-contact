#!/usr/bin/env python3
"""Freeze the generator-forensics protocol without reading selected HDF5 rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SPLIT = REPO / "runs/lewm_adaptive_compute_v3/cache/split_manifest.json"
H5 = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")
SALT = "lewm-generator-forensics-v1-exactly-8-discovery"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def score(ep: int) -> str:
    return hashlib.sha256(f"{SALT}:{ep}".encode()).hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / "protocol.json"
    if out.exists():
        raise SystemExit("protocol already frozen")
    split_hash = sha256(SPLIT)
    if split_hash != "4b9344b6ff2acce4c58c3b296ba2017b0122f8f8158f45088239712ce94b7030":
        raise RuntimeError("split manifest drift")
    payload = json.loads(SPLIT.read_text())
    discovery = [int(x) for x in payload["selection"]["split_episode_ordinals"]["train"]]
    test = {int(x) for x in payload["selection"]["split_episode_ordinals"]["test"]}
    chosen = sorted(discovery, key=lambda x: (score(x), x))[:8]
    if len(chosen) != 8 or set(chosen) & test:
        raise RuntimeError("invalid isolated sample")
    installed = Path("/Users/rishisim/Documents/research/World Models/le-wm/.venv/lib/python3.10/site-packages")
    sources = [
        installed / "stable_worldmodel/envs/ogbench/cube_env.py",
        installed / "ogbench/manipspace/envs/manipspace_env.py",
        installed / "ogbench/manipspace/envs/cube_env.py",
    ]
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_selected_hdf5_outcomes_or_environment_execution",
        "sample": {
            "population": "isolated V3 discovery allowlist only",
            "salt": SALT,
            "rule": "ascending SHA256(salt + ':' + decimal episode_id), tie episode_id",
            "episode_ids": chosen,
            "episode_scores": {str(x): score(x) for x in chosen},
            "count": 8,
            "v3_test_episode_intersection": [],
        },
        "indices": {
            "transition_rows": [0, 1, 2, 5, 10, 25, 50, 100],
            "frame_rows": [0, 50, 200],
            "initial_row": 0,
            "transitions_per_episode": 8,
            "frames_per_episode": 3,
            "no_sequential_expansion": True,
        },
        "alignment": {
            "hypotheses_in_order": [
                "row_t_is_pre_action_state_and_action_t_produces_row_t_plus_1",
                "row_t_is_post_action_state_and_prev_state_is_pre_action_state",
            ],
            "tests": [
                "prev_qpos[t] == qpos[t-1] and prev_qvel[t] == qvel[t-1] for t=1..200",
                "prev_qpos[t] == qpos[t] and prev_qvel[t] == qvel[t]",
                "step_idx/time increments and row-200 terminal sentinel",
                "control/action association by same-row versus one-row shift",
            ],
            "freeze_output": "alignment.json must be written before environment comparison",
        },
        "transforms": {
            "state": "inject exact float64 qpos/qvel then mj_forward; preserve stored target mocap/task",
            "observation": "installed compute_observation, plus independently assembled 28-D vector from stored privileged/proprio fields",
            "pixels": ["identity", "vertical_flip", "horizontal_flip", "vertical_horizontal_flip", "rgb_to_bgr_after_each_orientation"],
            "action": "installed set_control on stored normalized 5-D action; compare resulting float64 7-D ctrl",
            "transition_action": "restore pre-state and target, invoke installed step(action)",
            "transition_control": "restore pre-state and target, assign stored ctrl, run mj_step(nstep=installed _n_steps), rnePostConstraint/post_step",
            "psnr": "20*log10(255/rmse), infinity for exact",
            "ssim": "skimage.metrics.structural_similarity(channel_axis=-1,data_range=255) when installed; otherwise explicitly unavailable",
        },
        "tolerances": {
            "exact": 0.0,
            "float64_state_restore_atol": 1e-12,
            "float64_observation_atol": 1e-10,
            "float64_control_atol": 1e-10,
            "float64_transition_atol": 1e-9,
            "float32_action_atol": 1e-6,
            "pixels_pass": "bit equality after one predeclared convention; metrics descriptive otherwise",
            "rtol": 0.0,
        },
        "decision_tree": {
            "all_layers_pass": "current_local_mechanics_matched",
            "reset_fails_first": "reset_only_mismatch",
            "observation_or_render_fails_first": "render_or_observation_mismatch",
            "action_control_fails_first": "action_control_mismatch",
            "dynamics_fails_first": "dynamics_mismatch",
            "historical_direct_evidence_and_all_checks_pass": "exact_historical_generator_recovered",
            "otherwise_after_bounded_candidates": "generator_provenance_unresolved",
        },
        "historical_candidates": {"maximum": 3, "requires_direct_repository_or_release_evidence": True},
        "bounds": {"new_policy_trajectories": 0, "model_or_gate_rows": 0, "v5_episodes": 0},
        "isolation": {
            "combined_v3_cache_opened_with_numpy": False,
            "v3_test_targets_opened": False,
            "raw_reads_fail_closed_to_sample_episode_ids": chosen,
        },
        "source_hashes": {
            "split_manifest": split_hash,
            "source_h5": sha256(H5),
            **{str(p): sha256(p) for p in sources},
        },
    }
    serialized = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    out.write_text(serialized)
    (ROOT / "protocol.sha256").write_text(hashlib.sha256(serialized.encode()).hexdigest() + "  protocol.json\n")
    print(json.dumps({"protocol_sha256": hashlib.sha256(serialized.encode()).hexdigest(), "episodes": chosen}))


if __name__ == "__main__":
    main()
