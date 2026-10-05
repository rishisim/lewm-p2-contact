"""Research environments registered with Gymnasium."""

from gymnasium.envs.registration import register, registry


if "swm/PushTPeg-v1" not in registry:
    register(
        id="swm/PushTPeg-v1",
        entry_point="lewm_research.envs.pusht_peg:PushTPeg",
    )
