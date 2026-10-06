"""Deterministic matched scenes and trajectory-aware physical scoring."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import gymnasium as gym
import numpy as np
from shapely.geometry import LineString

import lewm_research.envs  # noqa: F401
from ..envs.pusht_peg import PushTPeg
from ..policies.weak import BlockWeakPolicy

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


def make_conditions(base, displacement_range=(60, 100), render_images=True):
    """Return all four conditions or raise ValueError for infeasible geometry."""
    if len(displacement_range) != 2 or not 0 < displacement_range[0] < displacement_range[1]:
        raise ValueError("displacement range must have positive increasing endpoints")
    env = PushTPeg(with_target=True, render_target_pose=(256, 256, np.pi / 4))
    try:
        env.reset(seed=base.seed)
        env.restore_snapshot(base.snapshot)
        start = env._get_obs()
        target = start.copy()
        target[:2] = base.goal_agent_xy
        target[2:5] = base.t_goal_pose
        rng = np.random.default_rng(base.seed ^ 0x5744)
        if not _free(env, start) or not _free(env, target):
            raise ValueError("start or goal geometry invalid")
        if segment_distance(start[7:9], start[2:4], target[2:4]) < 80:
            raise ValueError("off-path peg too close")
        on = None
        # Midpoint first, then deterministic jitter restricted to the segment.
        for fraction in [0.5, *rng.uniform(0.05, 0.95, 200)]:
            xy = start[2:4] + fraction * (target[2:4] - start[2:4])
            a, b = start.copy(), target.copy()
            a[7:9] = b[7:9] = xy
            if _free(env, a) and _free(env, b):
                on = xy
                break
        if on is None:
            raise ValueError("no non-overlapping on-path location")
        vector = None
        for _ in range(2000):
            angle, length = rng.uniform(0, 2 * np.pi), rng.uniform(*displacement_range)
            v = length * np.array([np.cos(angle), np.sin(angle)])
            peg_goal, t_goal = start.copy(), start.copy()
            peg_goal[7:9] += v
            t_goal[2:4] += v
            if (_free(env, peg_goal) and _free(env, t_goal)
                    and segment_distance(start[7:9], start[2:4], t_goal[2:4]) >= 80):
                vector = v
                break
        if vector is None:
            raise ValueError("no matched translation")
        result = {}
        for name in NAMES:
            initial, goal = start.copy(), target.copy()
            if name == "on_path":
                initial[7:9] = goal[7:9] = on
            elif name == "move_peg":
                goal = start.copy()
                goal[7:9] += vector
            elif name == "move_T_matched":
                goal = start.copy()
                goal[2:4] += vector
            env.restore_snapshot(base.snapshot)
            env._set_state(initial)
            env.set_goal(goal)
            result[name] = Condition(name, base.id, jsonable(env.get_snapshot()), goal.tolist(),
                                     {"peg": "target" if name == "move_peg" else "preserve",
                                      "peg_start": initial[7:9].tolist()}, env.render_state(goal) if render_images else None)
        return result
    finally:
        env.close()


def generate_bases(n, seed, min_t_displacement=60, displacement_range=(60, 100)):
    """Reject invalid scenes; goals come from a verified peg-parked rollout."""
    if n < 1 or seed < 0 or min_t_displacement <= 0:
        raise ValueError("n must be positive and seed nonnegative")
    env = gym.make("swm/PushTPeg-v1", with_target=False)
    bases = []
    try:
        for attempt in range(10000 * n):
            scene_seed = int(np.random.SeedSequence([seed, attempt]).generate_state(1)[0])
            obs, _ = env.reset(seed=scene_seed, options={"peg_placement": "clutter"})
            raw = env.unwrapped
            start = obs["state"].copy()
            if not _free(raw, start):
                continue
            snap = raw.get_snapshot()
            parked = start.copy()
            parked[7:9] = (-1000, -1000)
            raw._set_state(parked)
            policy = BlockWeakPolicy(seed=scene_seed)
            policy.set_env(env)
            for _ in range(25):
                env.step(policy.get_action()[0])
            goal = raw._get_obs()
            if not np.array_equal(goal[7:9], parked[7:9]):
                raise RuntimeError("parked peg moved")
            if np.linalg.norm(goal[2:4] - start[2:4]) < min_t_displacement:
                continue
            base = BaseScene(f"{seed}:{attempt}", scene_seed, start[:2].tolist(),
                             start[2:5].tolist(), start[7:9].tolist(), goal[2:5].tolist(),
                             goal[:2].tolist(), jsonable(snap))
            try:
                make_conditions(base, displacement_range, render_images=False)
            except ValueError:
                continue
            bases.append(base)
            if len(bases) == n:
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
