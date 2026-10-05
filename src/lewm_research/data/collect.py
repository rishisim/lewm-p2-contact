"""Fixed-length PushT-Peg collection in the wheel's HDF5 episode layout."""

from concurrent.futures import ProcessPoolExecutor
import json
import time
from pathlib import Path

import h5py
import numpy as np

from lewm_research.envs.pusht_peg import PushTPeg
from lewm_research.paths import dataset_path
from lewm_research.policies.weak import BlockWeakPolicy, MixedPolicy


def _episode(task):
    episode_idx, steps, policy_name, seed, placement = task
    episode_seed = int(np.random.SeedSequence([seed, episode_idx]).generate_state(1)[0])
    env = PushTPeg(with_target=False, terminate_on_success=False)
    obs, info = env.reset(seed=episode_seed, options={"peg_placement": placement})
    policy = BlockWeakPolicy(seed=episode_seed) if policy_name == "block" else MixedPolicy(seed=episode_seed)
    # The raw environment has no Gym spec; registration gives the same class a spec.
    policy.env = env
    policy.discrete = False
    choice = policy.begin_episode(episode_idx) if isinstance(policy, MixedPolicy) else "block"
    initial_peg = np.asarray(env.peg.position)
    max_displacement = 0.0
    contact_frames = 0
    columns = {k: [] for k in ("pixels", "action", "proprio", "state", "episode_idx", "ep_idx", "step_idx")}
    try:
        # Row t = observation before acting and the action taken from it (expert-data convention).
        for step_idx in range(steps):
            action = policy.get_action()[0]
            columns["pixels"].append(env.render().astype(np.uint8, copy=False))
            columns["action"].append(action)
            columns["proprio"].append(obs["proprio"].astype(np.float32))
            columns["state"].append(obs["state"].astype(np.float32))
            columns["episode_idx"].append(np.int32(episode_idx))
            columns["ep_idx"].append(np.int32(episode_idx))
            columns["step_idx"].append(np.int32(step_idx))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError("PushTPeg ended before fixed episode length")
            max_displacement = max(max_displacement, float(np.linalg.norm(info["peg_pos"] - initial_peg)))
            contact_frames += int(info["peg_contact"])
    finally:
        env.close()
    return ({key: np.stack(values) for key, values in columns.items()},
            {"episode_idx": episode_idx, "seed": episode_seed, "choice": choice,
             "peg_displaced": max_displacement > 5, "peg_contact_frames": contact_frames})


def _write_episode(file, episode, ep_idx, offset):
    length = len(episode["action"])
    if "ep_len" not in file:
        file.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype="i4")
        file.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype="i8")
        for name, values in episode.items():
            file.create_dataset(name, shape=(0, *values.shape[1:]),
                                maxshape=(None, *values.shape[1:]), dtype=values.dtype,
                                chunks=(1, *values.shape[1:]), compression="lzf")
    for name, values in episode.items():
        dataset = file[name]
        dataset.resize(offset + length, axis=0)
        dataset[offset:offset + length] = values
    for name, value in (("ep_len", length), ("ep_offset", offset)):
        dataset = file[name]
        dataset.resize(ep_idx + 1, axis=0)
        dataset[ep_idx] = value


def collect(name, episodes, steps=100, policy="block", seed=0, workers=1, placement="clutter"):
    """Collect one named dataset, failing if it already exists."""
    if episodes < 1 or steps < 1 or workers < 1:
        raise ValueError("episodes, steps, and workers must be positive")
    if policy not in {"block", "mixed"}:
        raise ValueError("policy must be block or mixed")
    if Path(name).name != name or not name or name in {".", ".."}:
        raise ValueError("name must be a single dataset name")
    path = dataset_path(name if name.endswith(".h5") else f"{name}.h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(path)
    if placement not in {"uniform", "clutter"}:
        raise ValueError("placement must be uniform or clutter")
    tasks = ((i, steps, policy, seed, placement) for i in range(episodes))
    start = time.perf_counter()
    records = []
    try:
        with h5py.File(path, "x", libver="latest") as file:
            if workers == 1:
                results = map(_episode, tasks)
                for idx, (episode, record) in enumerate(results):
                    _write_episode(file, episode, idx, idx * steps)
                    records.append(record)
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    for idx, (episode, record) in enumerate(pool.map(_episode, tasks)):
                        _write_episode(file, episode, idx, idx * steps)
                        records.append(record)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - start
    frames = episodes * steps
    metadata = {
        "dataset": str(path), "policy": policy, "seed": seed, "placement": placement,
        "episodes": episodes, "steps_per_episode": steps, "frames": frames,
        "workers": workers,
        "fraction_episodes_peg_displacement_gt_5px": sum(r["peg_displaced"] for r in records) / episodes,
        "fraction_frames_peg_contact": sum(r["peg_contact_frames"] for r in records) / frames,
        "bytes_per_frame": path.stat().st_size / frames,
        "frames_per_second": frames / elapsed,
        "elapsed_seconds": elapsed,
        "episode_records": records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
