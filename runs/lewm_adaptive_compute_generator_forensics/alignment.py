#!/usr/bin/env python3
"""Seal internal HDF5 row alignment before any environment comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parent
H5 = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")


def stats(x: np.ndarray) -> dict:
    x = np.abs(np.asarray(x, dtype=np.float64))
    return {"max_abs": float(x.max(initial=0)), "mean_abs": float(x.mean()) if x.size else 0.0, "exact": bool(np.all(x == 0))}


def main() -> None:
    protocol_bytes = (ROOT / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    expected = (ROOT / "protocol.sha256").read_text().split()[0]
    if hashlib.sha256(protocol_bytes).hexdigest() != expected:
        raise RuntimeError("protocol drift")
    out = ROOT / "alignment.json"
    if out.exists():
        raise RuntimeError("alignment already sealed")
    episodes = [int(x) for x in protocol["sample"]["episode_ids"]]
    allowed = set(protocol["isolation"]["raw_reads_fail_closed_to_sample_episode_ids"])
    if set(episodes) != allowed or len(episodes) != 8 or protocol["sample"]["v3_test_episode_intersection"]:
        raise RuntimeError("sample isolation failure")
    per_ep = []
    with h5py.File(H5, "r", swmr=True) as h:
        for ep in episodes:
            if ep not in allowed:
                raise RuntimeError("nonallowlisted raw read")
            off, n = int(h["ep_offset"][ep]), int(h["ep_len"][ep])
            if n != 201:
                raise RuntimeError("unexpected episode length")
            qpos = np.asarray(h["qpos"][off:off+n])
            qvel = np.asarray(h["qvel"][off:off+n])
            prev_qpos = np.asarray(h["prev_qpos"][off:off+n])
            prev_qvel = np.asarray(h["prev_qvel"][off:off+n])
            step = np.asarray(h["step_idx"][off:off+n])
            time = np.asarray(h["time"][off:off+n, 0])
            control = np.asarray(h["control"][off:off+n])
            action = np.asarray(h["action"][off:off+n])
            per_ep.append({
                "episode_id": ep,
                "prev_vs_prior_qpos": stats(prev_qpos[1:] - qpos[:-1]),
                "prev_vs_prior_qvel": stats(prev_qvel[1:] - qvel[:-1]),
                "prev_vs_same_qpos": stats(prev_qpos - qpos),
                "prev_vs_same_qvel": stats(prev_qvel - qvel),
                "first_prev_vs_first_qpos": stats(prev_qpos[:1] - qpos[:1]),
                "first_prev_vs_first_qvel": stats(prev_qvel[:1] - qvel[:1]),
                "step_idx": {"first": int(step[0]), "last": int(step[-1]), "unit_increment": bool(np.all(np.diff(step) == 1))},
                "time": {"first": float(time[0]), "last": float(time[-1]), "increment": stats(np.diff(time) - 0.05)},
                "terminal": {"terminated_last": bool(h["terminated"][off+n-1]), "truncated_last": bool(h["truncated"][off+n-1]), "earlier_any": bool(np.any(h["terminated"][off:off+n-1]) or np.any(h["truncated"][off:off+n-1]))},
                "final_action_zero": bool(np.all(action[-1] == 0)),
                "final_control_vs_prior": stats(control[-1] - control[-2]),
            })
    prior_exact = all(x["prev_vs_prior_qpos"]["exact"] and x["prev_vs_prior_qvel"]["exact"] for x in per_ep)
    sentinel = all(x["step_idx"]["first"] == 0 and x["step_idx"]["last"] == 200 and x["step_idx"]["unit_increment"] and x["terminal"]["truncated_last"] and not x["terminal"]["earlier_any"] for x in per_ep)
    if not prior_exact or not sentinel:
        raise RuntimeError("frozen alignment criteria failed")
    result = {
        "schema_version": 1,
        "status": "sealed_before_environment_comparison",
        "protocol_sha256": expected,
        "episode_ids": episodes,
        "episode_count": 8,
        "rows_per_episode": 201,
        "v3_test_episode_intersection": [],
        "v3_test_targets_opened": False,
        "combined_v3_cache_opened_with_numpy": False,
        "resolved_alignment": {
            "row_semantics": "row t stores post-step state for action[t-1]; prev_qpos/prev_qvel at row t are exactly qpos/qvel at row t-1 for t>=1",
            "transition_probe": "for frozen transition index t, inject qpos[t]/qvel[t], apply action[t], compare against qpos[t+1]/qvel[t+1]; compare generated control to control[t+1] because get_step_info is captured post-step on row t+1",
            "frame_observation_probe": "row t state/observation/pixels are co-indexed",
            "row_200": "terminal sentinel after action[199]; action[200] is padding and is never executed",
        },
        "per_episode": per_ep,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    out.write_text(serialized)
    (ROOT / "alignment.sha256").write_text(hashlib.sha256(serialized.encode()).hexdigest() + "  alignment.json\n")
    print(json.dumps(result["resolved_alignment"], indent=2))


if __name__ == "__main__":
    main()
