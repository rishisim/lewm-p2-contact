#!/usr/bin/env python3
"""Qualify the correct generation runtime without a cohort identifier or outcome."""

from __future__ import annotations

import importlib
import json
import sys
import time

from cycle_common import ROOT, assert_runtime_contract, atomic_json
from runner import distribution_modules
from verify_runtime_contract import verify as verify_runtime_contract


def main() -> None:
    output = ROOT / "audit/generation_runtime_qualification.json"
    if output.exists():
        raise RuntimeError("generation runtime qualification is already immutable")
    runtime = assert_runtime_contract("generation")
    contract = verify_runtime_contract()
    imports = {}
    for name in ("stable_worldmodel", "ogbench", "mujoco", "gymnasium"):
        module = importlib.import_module(name)
        imports[name] = str(getattr(module, "__file__", None))
    common, generator, seedfix = distribution_modules()
    del common, seedfix
    world = policy = None
    constructed = False
    cleanup_completed = False
    try:
        world, policy = generator.make_world("plan_oracle")
        constructed = bool(
            world is not None
            and policy is not None
            and callable(getattr(world, "close", None))
        )
        if not constructed:
            raise RuntimeError("global PlanOracle DGP construction failed qualification")
    finally:
        if world is not None:
            world.close()
            cleanup_completed = True
    checks = {
        "runtime_contract": contract["passed"],
        "exact_generation_interpreter": runtime["sys_executable"] == sys.executable,
        "required_imports": all(imports.values()),
        "planoracle_dgp_constructed": constructed,
        "planoracle_dgp_cleanup": cleanup_completed,
        "no_episode_identifier_used": True,
        "no_seed_tuple_used": True,
        "no_rollout_generated": True,
        "no_outcome_loss_contact_reward_or_success_inspected": True,
    }
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "runtime": runtime,
        "import_origins": imports,
        "checks": checks,
        "passed": all(checks.values()),
        "qualification_input": "global imports and DGP constructor/cleanup only",
        "package_smoke_episodes": 0,
        "v5_outcome_episodes": 0,
    }
    atomic_json(output, result, exclusive=True)
    if not result["passed"]:
        raise RuntimeError(f"generation runtime qualification failed: {result}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
