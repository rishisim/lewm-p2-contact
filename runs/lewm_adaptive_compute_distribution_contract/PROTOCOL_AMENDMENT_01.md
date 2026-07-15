# Protocol amendment 01: installed Cube reset seed forwarding

Status: frozen after the permanently excluded 12+12 smoke episodes and before
any main PlanOracle or MarkovOracle rollout.

The smoke pairing audit failed because the installed
`stable_worldmodel.envs.ogbench.CubeEnv.reset(seed=...)` resets its variation
space with `seed=None` and calls `super().reset(...)` without forwarding its
explicit `seed`. Thus `World.reset(seed)` records the seed in `EnvPool` but does
not control Cube start/goal variations or the physical-state RNG. The failed
audit is preserved byte-for-byte at `smoke_pairing_pre_amendment_failed.json`.

The smallest equivalent correction samples the four normal Cube default
variations with the predeclared environment seed, supplies those exact values
to the normal reset body, and installs `default_rng(env_seed)` as the unwrapped
environment RNG before reset. Policy-wrapper and oracle RNG streams remain
separate and unchanged. Capture, preprocessing, horizon, sample sizes, metrics,
bands, gate, and decision rules are unchanged.

An initial-state-only probe reused excluded smoke seed `1640100000` and
produced the same digest for both policy constructors:
`6a03206e2c327e185e7c0c8d3e371948c02fea594d88edab426943d7312d7583`. It generated zero additional policy
episodes. Every one of the 30 main pairs must still pass exact seed and initial
state hashing. No main target had been opened when this amendment was frozen.
