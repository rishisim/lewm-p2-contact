"""PushT with a movable, visible peg and exact simulator snapshots."""

from copy import deepcopy

import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import spaces
from shapely.geometry import Point, Polygon
from stable_worldmodel.envs.pusht.env import PushT


PEG_COLOR = (230, 126, 34)
PEG_RADIUS = 15
CLUTTER_MIN_BLOCK_DIST = 180
CLUTTER_MIN_AGENT_DIST = 100


class PushTPeg(PushT):
    def __init__(self, *args, terminate_on_success=False, peg_enabled=True, render_target_pose=None, peg_radius=PEG_RADIUS, **kwargs):
        self.peg_radius = peg_radius
        self.render_target_pose = render_target_pose
        self.peg_enabled = peg_enabled
        super().__init__(*args, **kwargs)
        self.terminate_on_success = terminate_on_success
        self.env_name = "PushTPeg"
        self.observation_space = spaces.Dict({
            "proprio": self.observation_space["proprio"],
            "state": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, -512, -512, -1000, -1000]),
                high=np.array([512, 512, 512, 512, 2 * np.pi, 512, 512, 512, 512]),
                dtype=np.float64,
            ),
        })

    def _setup(self):
        super()._setup()
        self.n_peg_contacts = 0
        self.peg = None
        if not self.peg_enabled:
            return
        self.peg = pymunk.Body(1, pymunk.moment_for_circle(1, 0, self.peg_radius))
        self.peg.position = (256, 256)
        shape = pymunk.Circle(self.peg, self.peg_radius)
        shape.friction = next(iter(self.block.shapes)).friction
        shape.color = pygame.Color(*PEG_COLOR)
        shape.collision_type = 2
        self.space.add(self.peg, shape)
        self.space.on_collision(2, 0, post_solve=self._handle_peg_collision)
        self.n_peg_contacts = 0

    def _handle_peg_collision(self, arbiter, space, data):
        self.n_peg_contacts += len(arbiter.contact_point_set.points)

    def _render_frame(self, mode):
        # Upstream dataset decoration is fixed, independent of the task goal.
        goal_pose = self.goal_pose
        if self.render_target_pose is not None:
            self.goal_pose = np.asarray(self.render_target_pose)
        try:
            return self._render_with_peg(mode)
        finally:
            self.goal_pose = goal_pose

    def _render_with_peg(self, mode):
        if not self.peg_enabled:
            return super()._render_frame(mode)
        self._set_body_color(self.peg, PEG_COLOR)
        frame = super()._render_frame(mode)
        # Pymunk's debug renderer brightens fills; preserve the specified RGB
        # in observation pixels after the upstream resize.
        center = pymunk.pygame_util.to_pygame(self.peg.position, self.screen)
        center = tuple(round(v * self.render_size / self.window_size) for v in center)
        radius = round(self.peg_radius * self.render_size / self.window_size)
        cv2.circle(frame, center, radius, PEG_COLOR, -1)
        cv2.circle(frame, center, radius, (120, 60, 10), 1)
        return frame

    @staticmethod
    def _proprio(state):
        return np.asarray(state)[[0, 1, 5, 6]].copy()

    def _get_obs(self):
        return np.array((*self.agent.position, *self.block.position,
                         self.block.angle % (2 * np.pi), *self.agent.velocity,
                         *(self.peg.position if self.peg_enabled else (-1000, -1000))), dtype=np.float64)

    def _set_state(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (9,):
            raise ValueError("PushTPeg state must have nine values")
        self.agent.angle = 0
        self.agent.position = tuple(state[:2])
        self.agent.velocity = tuple(state[5:7])
        self.agent.angular_velocity = 0
        self.block.angle = float(state[4])
        self.block.position = tuple(state[2:4])
        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0
        self.space.reindex_shapes_for_body(self.agent)
        self.space.reindex_shapes_for_body(self.block)
        if self.peg_enabled:
            self.peg.angle = 0
            self.peg.position = tuple(state[7:9])
            self.peg.velocity = (0, 0)
            self.peg.angular_velocity = 0
            self.space.reindex_shapes_for_body(self.peg)

    def get_snapshot(self):
        bodies = {}
        for name in (("agent", "block", "peg") if self.peg_enabled else ("agent", "block")):
            body = getattr(self, name)
            bodies[name] = {
                "position": tuple(body.position), "angle": body.angle,
                "velocity": tuple(body.velocity),
                "angular_velocity": body.angular_velocity,
            }
        return {
            "bodies": bodies,
            "goal_state": None if self.goal_state is None else np.array(self.goal_state).copy(),
            "goal_pose": np.array(self.goal_pose).copy(),
            "goal": None if not hasattr(self, "_goal") else np.array(self._goal).copy(),
            "latest_action": deepcopy(self.latest_action),
            "n_contact_points": self.n_contact_points,
            "n_peg_contacts": self.n_peg_contacts,
        }

    def restore_snapshot(self, snap):
        # Removing shapes clears Chipmunk's cached contact arbiters. Those
        # impulses otherwise survive a pose rewind and alter the next step.
        for name in (("agent", "block", "peg") if self.peg_enabled else ("agent", "block")):
            body = getattr(self, name)
            self.space.remove(*body.shapes, body)
        for name, values in snap["bodies"].items():
            body = getattr(self, name)
            body.angle = values["angle"]
            body.position = values["position"]
            body.velocity = values["velocity"]
            body.angular_velocity = values["angular_velocity"]
            self.space.add(body, *body.shapes)
        self.goal_state = None if snap["goal_state"] is None else np.array(snap["goal_state"]).copy()
        self.goal_pose = np.array(snap["goal_pose"]).copy()
        self._goal = None if snap["goal"] is None else np.array(snap["goal"]).copy()
        self.latest_action = deepcopy(snap["latest_action"])
        self.n_contact_points = snap["n_contact_points"]
        self.n_peg_contacts = snap["n_peg_contacts"]

    def render_state(self, state):
        # Rendering in an independent Pymunk space preserves live contact
        # arbiters as well as body fields, including during a collision.
        renderer = type(self)(resolution=self.render_size, with_target=self.with_target,
                              render_action=False, render_mode="rgb_array", peg_enabled=self.peg_enabled, render_target_pose=self.render_target_pose, peg_radius=self.peg_radius)
        try:
            renderer.variation_space = self.variation_space
            renderer._setup()
            renderer.goal_pose = np.array(self.goal_pose).copy()
            renderer._set_state(state)
            return renderer._render_frame("rgb_array").copy()
        finally:
            renderer.close()

    def set_goal(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (9,):
            raise ValueError("PushTPeg goal must have nine values")
        state = state.copy()
        if not self.peg_enabled:
            state[7:9] = -1000
        self.goal_state = state
        self.goal_pose = state[2:5].copy()
        self._goal = self.render_state(state)

    def _get_info(self):
        info = super()._get_info()
        info["goal_proprio"] = self._proprio(self.goal_state)
        info["peg_pos"] = np.array(self.peg.position if self.peg_enabled else (-1000, -1000))
        info["peg_contact"] = self.n_peg_contacts > 0
        return info

    def step(self, action):
        self.n_peg_contacts = 0
        obs, reward, terminated, truncated, info = super().step(np.asarray(action))
        obs["proprio"] = self._proprio(obs["state"])
        return obs, reward, bool(terminated and self.terminate_on_success), truncated, info

    def _shape_geometry(self, body):
        geometries = []
        for shape in body.shapes:
            if isinstance(shape, pymunk.Circle):
                geometries.append(Point(*body.local_to_world(shape.offset)).buffer(shape.radius))
            elif isinstance(shape, pymunk.Poly):
                geometries.append(Polygon([tuple(body.local_to_world(v)) for v in shape.get_vertices()]))
        return geometries

    def peg_overlaps(self, xy):
        disc = Point(*xy).buffer(self.peg_radius)
        return any(disc.intersects(geometry) for body in (self.agent, self.block)
                   for geometry in self._shape_geometry(body))

    def _sample_peg(self, placement="uniform"):
        """Sample a non-overlapping peg; ``clutter`` keeps it away from the block and agent."""
        block, agent = np.asarray(self.block.position), np.asarray(self.agent.position)
        for _ in range(10000):
            xy = self.np_random.uniform(30 + self.peg_radius, 512 - 30 - self.peg_radius, size=2)
            if self.peg_overlaps(xy):
                continue
            if placement == "clutter" and (np.linalg.norm(xy - block) < CLUTTER_MIN_BLOCK_DIST
                                           or np.linalg.norm(xy - agent) < CLUTTER_MIN_AGENT_DIST):
                continue
            return xy
        raise RuntimeError(f"Could not place peg ({placement})")

    def reset(self, seed=None, options=None):
        # Let the upstream reset initialize variations and the Pymunk space.
        options = dict(options or {})
        if "peg_xy" in options and "peg_placement" in options:
            raise ValueError("Use peg_xy or peg_placement, not both")
        placement = options.pop("peg_placement", None)
        peg_xy = options.pop("peg_xy", (256, 256))
        agent_xy = options.pop("agent_xy", None)
        block_pose = options.pop("block_pose", None)
        rng = np.random.default_rng(seed)
        if "state" in options:
            start = np.asarray(options["state"], dtype=np.float64)
        else:
            agent_xy = rng.uniform(50, 450, 2) if agent_xy is None else agent_xy
            block_pose = (*rng.uniform(100, 400, 2), rng.uniform(0, 2 * np.pi)) if block_pose is None else block_pose
            start = np.array([*agent_xy, *block_pose, 0, 0, *peg_xy], dtype=np.float64)
        goal = np.asarray(options.get("goal_state", start), dtype=np.float64)
        upstream_options = {**options, "state": start, "goal_state": goal}
        super().reset(seed=seed, options=upstream_options)
        if placement is not None and self.peg_enabled:
            if placement not in {"uniform", "clutter"}:
                raise ValueError("peg_placement must be 'uniform' or 'clutter'")
            start = start.copy()
            start[7:9] = self._sample_peg(placement)
            self._set_state(start)
            if "goal_state" not in options:
                goal = start.copy()
        self.set_goal(goal)
        state = self._get_obs()
        return {"state": state, "proprio": self._proprio(state)}, self._get_info()
