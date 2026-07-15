#!/usr/bin/env python3
"""Execute the frozen offline-to-local mechanics reconstruction audit."""
from __future__ import annotations

import hashlib, json, math, os, platform, time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import h5py
import hdf5plugin  # noqa: F401
import mujoco
import numpy as np
import stable_worldmodel as swm
from ogbench.manipspace import lie

ROOT = Path(__file__).resolve().parent
H5 = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")


def err(a: Any, b: Any) -> dict[str, Any]:
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"exact": bool(np.array_equal(a, b)), "max_abs": float(d.max(initial=0)), "mean_abs": float(d.mean()) if d.size else 0.0, "count": int(d.size)}


def combine(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = sum(x["count"] for x in items)
    return {"exact": all(x["exact"] for x in items), "max_abs": max((x["max_abs"] for x in items), default=0.0), "mean_abs": sum(x["mean_abs"] * x["count"] for x in items) / n if n else 0.0, "count": n}


def conventions(x: np.ndarray) -> dict[str, np.ndarray]:
    oriented = {"identity": x, "vertical_flip": x[::-1], "horizontal_flip": x[:, ::-1], "vertical_horizontal_flip": x[::-1, ::-1]}
    return {**oriented, **{k + "_bgr": v[..., ::-1] for k, v in oriented.items()}}


def set_target(env: Any, pos: np.ndarray, yaw: float, block: int = 0) -> None:
    env._target_block = int(block)
    env._target_task = "cube"
    mid = env._cube_target_mocap_ids[int(block)]
    env._data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64)
    env._data.mocap_quat[mid] = lie.SO3.from_z_radians(float(yaw)).wxyz


def restore(env: Any, row: dict[str, np.ndarray]) -> None:
    set_target(env, row["privileged_target_block_pos"], float(row["privileged_target_block_yaw"][0]), int(row["privileged_target_block"]))
    env._data.qpos[:] = row["qpos"]
    env._data.qvel[:] = row["qvel"]
    env._data.time = float(row["time"][0])
    env._data.ctrl[:] = row["control"]
    mujoco.mj_forward(env._model, env._data)
    env._prev_qpos = np.asarray(row["prev_qpos"]).copy()
    env._prev_qvel = np.asarray(row["prev_qvel"]).copy()
    env._reset_next_step = False


def load_row(h: h5py.File, i: int) -> dict[str, np.ndarray]:
    names = ["qpos","qvel","prev_qpos","prev_qvel","control","action","time","observation","pixels","proprio_joint_pos","proprio_joint_vel","proprio_effector_pos","proprio_effector_yaw","proprio_gripper_opening","proprio_gripper_vel","proprio_gripper_contact","privileged_block_0_pos","privileged_block_0_quat","privileged_block_0_yaw","privileged_target_block","privileged_target_block_pos","privileged_target_block_yaw"]
    return {k: np.asarray(h[k][i]) for k in names}


def main() -> None:
    protocol_b = (ROOT / "protocol.json").read_bytes(); alignment_b = (ROOT / "alignment.json").read_bytes()
    protocol = json.loads(protocol_b); alignment = json.loads(alignment_b)
    if hashlib.sha256(protocol_b).hexdigest() != alignment["protocol_sha256"] or alignment["status"] != "sealed_before_environment_comparison":
        raise RuntimeError("unsealed protocol/alignment")
    output_name = os.environ.get("LEWM_FORENSICS_OUTPUT", "metrics.json")
    if "/" in output_name or output_name.startswith("."):
        raise RuntimeError("invalid output name")
    out = ROOT / output_name
    if out.exists(): raise RuntimeError("metrics already exist")
    episodes = protocol["sample"]["episode_ids"]; tids = protocol["indices"]["transition_rows"]; fids = protocol["indices"]["frame_rows"]
    world = swm.World("swm/OGBCube-v0", num_envs=1, max_episode_steps=200, image_shape=(224,224), env_type="single", multiview=False, width=224, height=224, visualize_info=False, terminate_at_goal=False, mode="data_collection")
    world.reset(seed=0, options=None)
    env = world.envs.envs[0].unwrapped
    records, pixels = [], []
    reset_records = []
    started = time.time()
    try:
      with h5py.File(H5, "r", swmr=True) as h:
        for ep in episodes:
          if ep not in protocol["isolation"]["raw_reads_fail_closed_to_sample_episode_ids"]: raise RuntimeError("raw allowlist failure")
          off = int(h["ep_offset"][ep]); n = int(h["ep_len"][ep])
          if n != 201 or not np.all(np.asarray(h["ep_idx"][off:off+n]) == ep): raise RuntimeError("layout failure")
          first = load_row(h, off)
          # Frozen reset/IK reconstruction: stored task variations, fixed seed 0, no state injection.
          start_yaw = float(first["privileged_block_0_yaw"][0]); goal_yaw = float(first["privileged_target_block_yaw"][0])
          vals = {"cube.start_position": first["privileged_block_0_pos"][:2][None], "cube.start_yaw": np.array([start_yaw % (2*np.pi)]), "cube.goal_position": first["privileged_target_block_pos"][:2][None], "cube.goal_yaw": np.array([goal_yaw % (2*np.pi)])}
          env.np_random = np.random.default_rng(0)
          env.reset(seed=0, options={"variation": [], "variation_values": vals})
          ri = env.compute_ob_info()
          reset_records.append({"episode_id": ep, "qpos": err(ri["qpos"], first["qpos"]), "qvel": err(ri["qvel"], first["qvel"]), "block_pos": err(ri["privileged/block_0_pos"], first["privileged_block_0_pos"]), "target_pos": err(ri["privileged/target_block_pos"], first["privileged_target_block_pos"]), "known_seed": False, "seed_used": 0})
          for t in sorted(set(tids + fids)):
            row = load_row(h, off+t); restore(env, row)
            state_qpos, state_qvel = err(env._data.qpos, row["qpos"]), err(env._data.qvel, row["qvel"])
            oi = env.compute_ob_info(); obs = env.compute_observation()
            priv = {"joint_pos": err(oi["proprio/joint_pos"], row["proprio_joint_pos"]), "joint_vel": err(oi["proprio/joint_vel"], row["proprio_joint_vel"]), "effector_pos": err(oi["proprio/effector_pos"], row["proprio_effector_pos"]), "effector_yaw": err(oi["proprio/effector_yaw"], row["proprio_effector_yaw"]), "gripper_opening": err(oi["proprio/gripper_opening"], row["proprio_gripper_opening"]), "gripper_vel": err(oi["proprio/gripper_vel"], row["proprio_gripper_vel"]), "gripper_contact": err(oi["proprio/gripper_contact"], row["proprio_gripper_contact"]), "block_pos": err(oi["privileged/block_0_pos"], row["privileged_block_0_pos"]), "block_quat": err(oi["privileged/block_0_quat"], row["privileged_block_0_quat"]), "block_yaw": err(oi["privileged/block_0_yaw"], row["privileged_block_0_yaw"])}
            rec = {"episode_id": ep, "row": t, "state_qpos": state_qpos, "state_qvel": state_qvel, "observation": err(obs, row["observation"]), "privileged": priv}
            if t in fids:
              frame = env.render(camera="front_pixels")
              choices = {k: err(v, row["pixels"]) for k,v in conventions(frame).items()}
              best = min(choices, key=lambda k:(choices[k]["max_abs"], choices[k]["mean_abs"], k)); d = frame.astype(float)-row["pixels"].astype(float); mse=float(np.mean(d*d))
              pixels.append({"episode_id":ep,"row":t,"best_convention":best,"conventions":choices,"mismatch_rate":float(np.mean(conventions(frame)[best] != row["pixels"])),"per_channel_mean_abs":np.mean(np.abs(conventions(frame)[best].astype(float)-row["pixels"].astype(float)),axis=(0,1)).tolist(),"psnr":None if mse==0 else float(20*math.log10(255/math.sqrt(mse))),"ssim":None})
            if t in tids:
              nxt = load_row(h, off+t+1)
              restore(env,row); env.set_control(row["action"]); control_e=err(env._data.ctrl,nxt["control"])
              restore(env,row); env.step(row["action"]); aqp, aqv = err(env._data.qpos,nxt["qpos"]),err(env._data.qvel,nxt["qvel"])
              restore(env,row); env._data.ctrl[:]=nxt["control"]; env.pre_step(); mujoco.mj_step(env._model,env._data,nstep=env._n_steps); mujoco.mj_rnePostConstraint(env._model,env._data); env.post_step(); cqp,cqv=err(env._data.qpos,nxt["qpos"]),err(env._data.qvel,nxt["qvel"])
              rec["control_mapping"],rec["action_transition_qpos"],rec["action_transition_qvel"],rec["direct_control_transition_qpos"],rec["direct_control_transition_qvel"] = control_e,aqp,aqv,cqp,cqv
            records.append(rec)
    finally: world.close()
    def collect(path: str) -> list[dict]:
      out=[]
      for r in records:
        x=r
        try:
          for p in path.split("."): x=x[p]
        except KeyError:
          continue
        out.append(x)
      return out
    tol=protocol["tolerances"]
    summary={k:combine(collect(k)) for k in ["state_qpos","state_qvel","observation","control_mapping","action_transition_qpos","action_transition_qvel","direct_control_transition_qpos","direct_control_transition_qvel"]}
    summary["privileged"]={k:combine([r["privileged"][k] for r in records]) for k in records[0]["privileged"]}
    summary["reset_qpos"]=combine([r["qpos"] for r in reset_records]); summary["reset_qvel"]=combine([r["qvel"] for r in reset_records]); summary["pixels_best"]=combine([x["conventions"][x["best_convention"]] for x in pixels])
    passes={"reset":summary["reset_qpos"]["max_abs"]<=tol["float64_state_restore_atol"] and summary["reset_qvel"]["max_abs"]<=tol["float64_state_restore_atol"],"state":summary["state_qpos"]["max_abs"]<=tol["float64_state_restore_atol"] and summary["state_qvel"]["max_abs"]<=tol["float64_state_restore_atol"],"observation":summary["observation"]["max_abs"]<=tol["float64_observation_atol"],"render":summary["pixels_best"]["exact"],"control":summary["control_mapping"]["max_abs"]<=tol["float64_control_atol"],"action_dynamics":summary["action_transition_qpos"]["max_abs"]<=tol["float64_transition_atol"] and summary["action_transition_qvel"]["max_abs"]<=tol["float64_transition_atol"],"direct_control_dynamics":summary["direct_control_transition_qpos"]["max_abs"]<=tol["float64_transition_atol"] and summary["direct_control_transition_qvel"]["max_abs"]<=tol["float64_transition_atol"]}
    result={"schema_version":1,"status":"complete","protocol_sha256":alignment["protocol_sha256"],"alignment_sha256":hashlib.sha256(alignment_b).hexdigest(),"episodes":episodes,"episode_count":8,"transition_count":len(episodes)*len(tids),"frame_count":len(episodes)*len(fids),"new_policy_trajectories":0,"model_or_gate_evaluation_rows":0,"v5_confirmation_episodes":0,"v3_test_episode_intersection":[],"v3_test_targets_opened":False,"combined_v3_cache_opened_with_numpy":False,"environment":{"python":platform.python_version(),"platform":platform.platform()},"summary":summary,"passes":passes,"reset_records":reset_records,"pixel_records":pixels,"records":records,"elapsed_seconds":time.time()-started}
    result["implementation_source"] = str(Path(swm.__file__).resolve())
    text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"; out.write_text(text); (ROOT/(out.stem+".sha256")).write_text(hashlib.sha256(text.encode()).hexdigest()+f"  {out.name}\n"); print(json.dumps({"summary":summary,"passes":passes},indent=2))

if __name__ == "__main__": main()
