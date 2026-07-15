#!/usr/bin/env python3
"""Run the authorized one-trajectory replay with the amended reset path."""

from __future__ import annotations

import json
from typing import Any

import bounded_check
import generator
import generator_seedfix


_ORIGINAL_MAKE_WORLD = generator.make_world


def make_world_seedfixed(policy_type: str) -> tuple[Any, Any]:
    world, policy = _ORIGINAL_MAKE_WORLD(policy_type)
    original_reset = world.reset

    def fixed_reset(seed: int | None = None, options: Any = None) -> None:
        if seed is None or options is not None:
            raise RuntimeError("bounded replay requires its recorded explicit seed")
        generator_seedfix.deterministic_environment_reset(
            world, original_reset, int(seed)
        )

    world.reset = fixed_reset
    return world, policy


def run() -> dict:
    generator_seedfix.assert_amendment_seal()
    generator.make_world = make_world_seedfixed
    return bounded_check.run()


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
