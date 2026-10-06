"""Deterministic matched scenes and trajectory-aware physical scoring."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import gymnasium as gym
import numpy as np
from shapely.geometry import LineString

import lewm_research.envs  # noqa: F401
from ..envs.pusht_peg import PushTPeg
from ..policies.weak import BlockWeakPolicy, PegWeakPolicy

NAMES = ("off_path", "on_path", "move_peg", "move_T_matched")


@dataclass
class BaseScene:
    id: str
    seed: int
    agent_xy: list
    block_pose: list
    peg_xy: list
    t_goal_pose: list
    goal_agent_xy: list
    snapshot: dict
    rollout_tasks: dict | None = None


@dataclass
class Condition:
    name: str
    base_id: str
    start_snapshot: dict
    goal_state: list
    scoring_spec: dict
    goal_image: np.ndarray


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def segment_distance(xy, start, goal):
    from shapely.geometry import Point
    return LineString([start, goal]).distance(Point(*xy))


def _free(env, state):
    env._set_state(state)
    shapes = env._shape_geometry(env.block)
    agent = env._shape_geometry(env.agent)[0]
    return (all(not agent.intersects(s) for s in shapes)
            and not env.peg_overlaps(state[7:9])
            and all(s.bounds[0] >= 30 and s.bounds[1] >= 30
                    and s.bounds[2] <= 482 and s.bounds[3] <= 482 for s in shapes)
            and np.all((state[:2] >= 50) & (state[:2] <= 462))
            and np.all((state[7:9] >= 45) & (state[7:9] <= 467)))


def make_conditions(base, displacement_range=(40,100), render_images=True, with_target=True):
    """Construct only the canonical W4c tasks; legacy scenes remain readable."""
    if base.rollout_tasks is None:
        raise ValueError("archived pre-W4c scenes require their original Git revision")
    if tuple(displacement_range) != (40,100):
        raise ValueError("W4c fixes displacement range=(40,100)")
    return _rollout_conditions(base, render_images, with_target)


def _rollout_conditions(base, render_images, with_target):
    """Stored physical endpoints; preserve all start velocities."""
    env = PushTPeg(with_target=with_target, render_target_pose=(256,256,np.pi/4))
    result = {}
    try:
        env.reset(seed=base.seed)
        for name, task in base.rollout_tasks.items():
            env.restore_snapshot(task["snapshot"])
            env.set_goal(task["goal_state"])
            result[name] = Condition(name, base.id, jsonable(env.get_snapshot()), task["goal_state"],
                {"peg":"target" if name=="move_peg" else "preserve", "peg_start":list(env.peg.position)},
                env.render_state(task["goal_state"]) if render_images else None)
        return result
    finally:
        env.close()


def _weak_trace(env, snapshot, policy, steps=300):
    raw = env.unwrapped
    raw._setup()
    raw.restore_snapshot(snapshot)
    policy.set_env(env)
    states, snapshots, actions = [raw._get_obs().copy()], [jsonable(raw.get_snapshot())], []
    for _ in range(steps):
        action = policy.get_action()[0]
        env.step(action)
        states.append(raw._get_obs().copy())
        snapshots.append(jsonable(raw.get_snapshot()))
        snapshots[-1]["goal"] = None
        actions.append(action.tolist())
    return np.asarray(states), snapshots, actions


def _window(states, object_indices, stationary_indices, maximum=None, bin_index=None, small_rotation=False):
    for t in range(len(states)-25):
        start, goal = states[t], states[t+25]
        length = np.linalg.norm(goal[object_indices]-start[object_indices])
        fixed = np.max(np.abs(states[t:t+26, stationary_indices]-start[stationary_indices])) < 1e-6
        angle = abs((goal[4]-start[4]+np.pi)%(2*np.pi)-np.pi)
        if (np.linalg.norm(start[:2]-start[object_indices])<=40 and length>=40
                and (maximum is None or length<=maximum) and fixed
                and (bin_index is None or min(2,int((length-40)//20))==bin_index)
                and (not small_rotation or angle<np.pi/9)):
            return t
    return None


def generate_bases(n, seed, min_t_displacement=40, displacement_range=(40,100)):
    """W4c paired D/G rollout windows from the same clutter scene.

    The first qualifying D window is used; invalid midpoint geometry discards
    the scene. Witness actions establish reachability, never initialize CEM.
    """
    from copy import deepcopy
    if n<1 or seed<0 or min_t_displacement!=40 or tuple(displacement_range)!=(40,100):
        raise ValueError("W4c fixes minimum=40 and displacement range=(40,100)")
    env = gym.make("swm/PushTPeg-v1",with_target=False,max_episode_steps=10000)
    bases = []
    try:
        for attempt in range(10000*n):
            scene_seed = int(np.random.SeedSequence([seed,attempt]).generate_state(1)[0])
            obs,_ = env.reset(seed=scene_seed,options={"peg_placement":"clutter"})
            raw = env.unwrapped
            if not _free(raw,obs["state"]):
                continue
            original = jsonable(raw.get_snapshot()); original["goal"] = None
            ts,snaps,actions = _weak_trace(env,original,BlockWeakPolicy(seed=scene_seed))
            d = _window(ts,slice(2,4),slice(7,9))
            if d is None:
                continue
            start,goal = ts[d],ts[d+25]
            if segment_distance(start[7:9],start[2:4],goal[2:4])<80:
                continue
            on = (start[2:4]+goal[2:4])/2
            a,b = start.copy(),goal.copy(); a[7:9]=b[7:9]=on
            raw._set_state(a); valid = not raw.peg_overlaps(on)
            raw._set_state(b)
            if not valid or raw.peg_overlaps(on) or not np.all((on>=45)&(on<=467)):
                continue
            ps,psnaps,pactions = _weak_trace(env,original,PegWeakPolicy(dist_constraint=30,seed=scene_seed))
            g = _window(ps,slice(7,9),slice(2,5),maximum=100)
            if g is None:
                continue
            length = np.linalg.norm(ps[g+25,7:9]-ps[g,7:9]); bin_index = min(2,int((length-40)//20))
            m = _window(ts,slice(2,4),slice(7,9),maximum=100,bin_index=bin_index,small_rotation=True)
            if m is None:
                continue
            tasks = {}
            for name,states,snapshots,acts,t in (("off_path",ts,snaps,actions,d),("on_path",ts,snaps,actions,d),
                    ("move_peg",ps,psnaps,pactions,g),("move_T_matched",ts,snaps,actions,m)):
                snap,end = deepcopy(snapshots[t]),states[t+25].copy()
                if name=="on_path":
                    snap["bodies"]["peg"]["position"]=on.tolist()
                    snap["bodies"]["peg"]["velocity"]=[0,0]; end[7:9]=on
                tasks[name] = {"snapshot":snap,"goal_state":end.tolist(),"rollout_t":t,
                    "witness_actions":acts[t:t+25],"displacement_bin":bin_index if name.startswith("move_") else None,
                    "displacement_px":float(np.linalg.norm(end[7:9]-states[t,7:9]) if name=="move_peg"
                                            else np.linalg.norm(end[2:4]-states[t,2:4]))}
            # Snapshot restores cannot retain Chipmunk's cached contact impulses.
            # Reject a scene if the original action witness cannot solve its
            # unchanged goal after a fresh restore, as evaluation will start it.
            witnessed = True
            for name in ("off_path", "move_peg", "move_T_matched"):
                task = tasks[name]
                raw._setup()
                raw.restore_snapshot(task["snapshot"])
                trajectory = [raw._get_obs().copy()]
                for action in task["witness_actions"]:
                    trajectory.append(raw.step(np.asarray(action, dtype=np.float32))[0]["state"].copy())
                c = Condition(name, "construction", task["snapshot"], task["goal_state"],
                    {"peg":"target" if name=="move_peg" else "preserve",
                     "peg_start":trajectory[0][7:9].tolist()}, None)
                task["witness_replay_score"] = score_trajectory(trajectory, c)
                task["witness_replay_max_state_error"] = float(np.max(np.abs(trajectory[-1]-task["goal_state"])))
                witnessed &= task["witness_replay_score"]["success"]
            if not witnessed:
                continue
            bases.append(BaseScene(f"{seed}:{attempt}",scene_seed,start[:2].tolist(),start[2:5].tolist(),
                start[7:9].tolist(),goal[2:5].tolist(),goal[:2].tolist(),snaps[d],tasks))
            print(f"accepted {len(bases)}/{n} scene attempt={attempt}",flush=True)
            if len(bases)==n:
                return bases
        raise RuntimeError(f"Only generated {len(bases)}/{n} feasible bases")
    finally:
        env.close()


def _errors(trajectory, condition):
    states = np.asarray(trajectory, dtype=float)
    if states.ndim != 2 or states.shape[1] != 9 or not len(states) or not np.isfinite(states).all():
        raise ValueError("trajectory must contain finite nine-value states")
    goal = np.asarray(condition.goal_state)
    position = np.linalg.norm(states[:, 2:4] - goal[2:4], axis=1)
    angle = np.abs((states[:, 4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
    displacement = np.linalg.norm(states[:, 7:9] - condition.scoring_spec["peg_start"], axis=1)
    peg_error = np.linalg.norm(states[:, 7:9] - goal[7:9], axis=1)
    return position, angle, displacement, peg_error


def score_trajectory(trajectory, condition):
    pos, angle, displacement, peg_error = _errors(trajectory, condition)
    t = (pos < 20) & (angle < np.pi / 9)
    preserved = np.maximum.accumulate(displacement) < 10
    peg = peg_error < 15
    success = t & (peg if condition.scoring_spec["peg"] == "target" else preserved)
    return {"t_success": bool(t[-1]), "peg_preserved": bool(preserved[-1]),
            "peg_target": bool(peg[-1]), "success": bool(success[-1]),
            "disturbance_allowed_t_success": bool(t[-1]),
            "ever_disturbance_allowed_t_success": bool(t.any()),
            "ever_t_success": bool(t.any()), "ever_peg_preserved": bool(preserved.any()),
            "ever_peg_target": bool(peg.any()), "ever_success": bool(success.any()),
            "max_peg_displacement": float(displacement.max()),
            "role_cost": role_cost(trajectory, condition),
            "role_cost_endpoint": role_cost_endpoint(trajectory, condition)}


def role_cost(trajectory, condition):
    pos, angle, displacement, peg_error = _errors(trajectory, condition)
    peg = peg_error[-1] if condition.scoring_spec["peg"] == "target" else 2 * displacement.max()
    return float(pos[-1] + 100 * angle[-1] + peg)


def role_cost_endpoint(trajectory, condition):
    return role_cost(np.asarray(trajectory)[-1:], condition)


def save_bases(bases, path):
    Path(path).write_text(json.dumps([asdict(b) for b in bases], indent=2) + "\n")


def load_bases(path):
    return [BaseScene(**b) for b in json.loads(Path(path).read_text())]


def save_conditions(conditions, path):
    Path(path).write_text(json.dumps({k: jsonable(asdict(v)) for k, v in conditions.items()}) + "\n")


def load_conditions(path):
    result = {}
    for name, values in json.loads(Path(path).read_text()).items():
        values["goal_image"] = np.asarray(values["goal_image"], dtype=np.uint8)
        result[name] = Condition(**values)
    return result
