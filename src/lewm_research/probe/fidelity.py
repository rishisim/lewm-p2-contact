"""Dataset-paired no-peg fidelity gate for the controlled evaluation loop."""

import json
import hashlib
import time
import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.world.world import _extract_init_goal

from ..paths import dataset_path, runs_root
from .rollout_eval import Planner, prepare_output_root, controlled_steps


def expert_pairs(n=50, seed=42):
    dataset = swm.data.HDF5Dataset(path=dataset_path('pusht_expert_train.h5'),
                                  keys_to_cache=['action', 'proprio', 'state'])
    index = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    episode = dataset.get_col_data(index)
    step = dataset.get_col_data('step_idx')
    episodes, inverse = np.unique(episode, return_inverse=True)
    lengths = np.zeros(len(episodes), dtype=np.int64)
    np.maximum.at(lengths, inverse, step + 1)
    valid = np.flatnonzero(step <= (lengths - 26)[inverse])
    chosen = np.sort(np.random.default_rng(seed).choice(valid, n, replace=False))
    rows = dataset.get_row_data(chosen)
    initial, goal, _ = _extract_init_goal(dataset, rows[index].tolist(), rows['step_idx'].tolist(), 25)
    return chosen, initial, goal


def fidelity(n=50, seed=42, device='auto', run_dir=None, dataset_images=False):
    """Full controlled budget, upstream ever-success (includes agent error).

    The optional dataset-images diagnostic duplicates smoke-eval's overwritten
    observations; the primary gate renders both states as Step 4 does.
    """
    root = prepare_output_root(run_dir or runs_root() / 'w4b-fidelity')
    root.mkdir(parents=True, exist_ok=True)
    chosen, initial, goal = expert_pairs(n, seed)
    planner = Planner('lewm-pusht', device=device, topk=30)
    world = swm.World('swm/PushTPeg-v1', num_envs=n, image_shape=(224,224),
                      with_target=True, render_target_pose=(256,256,np.pi/4), peg_enabled=False, max_episode_steps=100)
    started = time.perf_counter()
    try:
        options = [{'state': np.r_[s, -1000., -1000.],
                    'goal_state': np.r_[g, -1000., -1000.]}
                   for s, g in zip(initial['state'], goal['goal_state'])]
        world.reset(seed=initial.get('seed'), options=options)
        if dataset_images:
            shape = world.infos['pixels'].shape[:2]
            for source in (initial, goal):
                for k, v in source.items():
                    if k in world.infos or k in goal:
                        world.infos[k] = np.broadcast_to(v[:,None,...], shape + v.shape[1:]).copy()
        pixel_mae = {"start": float(np.abs(world.infos['pixels'][:,0].astype(float) - initial['pixels']).mean()),
                     "goal": float(np.abs(world.infos['goal'][:,0].astype(float) - goal['goal']).mean())}
        (root / 'normalization.json').write_text(json.dumps(planner.stats, indent=2))
        fixed_goal = {k: v.copy() for k,v in world.infos.items() if k.startswith('goal')}
        planner.start(None, world.envs, seed)
        success = np.zeros(n, bool)
        trajectories = [[] for _ in range(n)]
        for _, _, _ in controlled_steps(world, planner, 50, fixed_goal=fixed_goal):
            for i, wrapped in enumerate(world.envs.envs):
                raw = wrapped.unwrapped
                state = raw._get_obs()
                trajectories[i].append(state.tolist())
                success[i] |= raw.eval_state(raw.goal_state, state)[0]
        result = {'n': n, 'seed': seed, 'chosen_rows': chosen.tolist(),
                  'dataset_images': dataset_images, 'render_target_pose': [256,256,float(np.pi/4)],
                  'pixel_mae': pixel_mae,
                  'normalization_sha256': hashlib.sha256(json.dumps(planner.stats,sort_keys=True).encode()).hexdigest(),
                  'successes': success.tolist(),
                  'success_count': int(success.sum()), 'elapsed_s': time.perf_counter()-started,
                  'trajectories': trajectories}
        (root / ('dataset.json' if dataset_images else 'rendered.json')).write_text(json.dumps(result))
        return {k: v for k,v in result.items() if k != 'trajectories'}
    finally:
        world.close()
        planner.close()
