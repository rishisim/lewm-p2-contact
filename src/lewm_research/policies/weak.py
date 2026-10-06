"""Weak PushT collection policies with episode-level target selection."""

import numpy as np


class _WeakPolicy:
    def __init__(self, dist_constraint=100, seed=None):
        if dist_constraint <= 0:
            raise ValueError("dist_constraint must be positive")
        self.dist_constraint = dist_constraint
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env):
        spec = getattr(env, "spec", None)
        if spec is None and hasattr(env, "envs"):
            spec = env.envs[0].spec
        if spec is None or not spec.id.startswith("swm/PushT"):
            raise ValueError("Weak policy requires a PushT-family environment")
        self.env = env
        self.discrete = "Discrete" in spec.id

    def _envs(self):
        if hasattr(self.env, "envs"):
            return [env.unwrapped for env in self.env.envs]
        return [self.env.unwrapped]

    def _target(self, env, index):
        raise NotImplementedError

    def _limit(self, env, index):
        return self.dist_constraint

    def get_action(self, info_dict=None, **kwargs):
        if not hasattr(self, "env"):
            raise RuntimeError("Call set_env first")
        actions = []
        for i, env in enumerate(self._envs()):
            target = np.asarray(self._target(env, i))
            desired = np.asarray(env.agent.position) + self.rng.uniform(-1, 1, 2) * env.action_scale
            limit = self._limit(env, i)
            desired = np.clip(desired, target - limit, target + limit)
            action = np.clip((desired - np.asarray(env.agent.position)) / env.action_scale, -1, 1)
            if self.discrete:
                action = env.quantizer.quantize(action)
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)


class BlockWeakPolicy(_WeakPolicy):
    def _target(self, env, index):
        return env.block.position


class PegWeakPolicy(_WeakPolicy):
    def _target(self, env, index):
        return env.peg.position


class MixedPolicy(_WeakPolicy):
    """Per-episode block- or peg-centered actions; peg episodes use a tighter box to make contact."""

    def __init__(self, dist_constraint=100, seed=None, p_peg=0.5, peg_dist_constraint=None):
        super().__init__(dist_constraint, seed)
        if peg_dist_constraint is not None and peg_dist_constraint <= 0:
            raise ValueError("peg_dist_constraint must be positive")
        self.p_peg = p_peg
        self.peg_dist_constraint = peg_dist_constraint
        self.choices = {}

    def begin_episode(self, episode_idx):
        choice = "peg" if self.rng.random() < self.p_peg else "block"
        self.choices[episode_idx] = choice
        self.current_choice = choice
        return choice

    def _target(self, env, index):
        if not hasattr(self, "current_choice"):
            self.begin_episode(0)
        return env.peg.position if self.current_choice == "peg" else env.block.position

    def _limit(self, env, index):
        if self.current_choice == "peg":
            return env.peg_radius + 15 if self.peg_dist_constraint is None else self.peg_dist_constraint
        return self.dist_constraint
