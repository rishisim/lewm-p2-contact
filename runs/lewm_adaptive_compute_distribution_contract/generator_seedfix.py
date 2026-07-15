#!/usr/bin/env python3
"""Pre-main seed-forwarding amendment for the installed Cube reset API.

The excluded smoke run proved that the installed stable-worldmodel CubeEnv
accepts ``seed`` but does not forward it to its parent reset and separately
reseeds its variation space with ``None``.  This amendment preserves that
evidence and deterministically materializes the same two environment RNG
components before invoking the otherwise byte-sealed generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

import common
import generator


AMENDMENT_JSON = common.STUDY_ROOT / "protocol_amendment_01.json"
AMENDMENT_MD = common.STUDY_ROOT / "PROTOCOL_AMENDMENT_01.md"
AMENDMENT_SEAL = common.STUDY_ROOT / "audit/protocol_amendment_01_seal.json"
FAILED_SMOKE_COPY = (
    common.STUDY_ROOT / "audit/smoke_pairing_pre_amendment_failed.json"
)
_ORIGINAL_GENERATE_EPISODE = generator.generate_episode


def deterministic_environment_reset(
    world: Any, original_reset: Callable[..., None], env_seed: int
) -> dict[str, Any]:
    """Reset both RNG components omitted by the installed CubeEnv seed path."""
    from stable_worldmodel import spaces as swm_spaces
    from stable_worldmodel import utils as swm_utils
    from stable_worldmodel.envs.ogbench.cube_env import DEFAULT_VARIATIONS

    env = world.envs.envs[0].unwrapped
    swm_spaces.reset_variation_space(
        env.variation_space,
        seed=int(env_seed),
        options={"variation": list(DEFAULT_VARIATIONS)},
        default_variations=DEFAULT_VARIATIONS,
    )
    values = {
        key: np.asarray(
            swm_utils.get_in(env.variation_space, key.split(".")).value
        ).copy()
        for key in DEFAULT_VARIATIONS
    }
    # The installed CubeEnv.reset drops its explicit seed instead of forwarding
    # it to ManipSpaceEnv/Gymnasium.  Installing the generator before reset is
    # behaviorally equivalent to the missing super().reset(seed=env_seed).
    env.np_random = np.random.default_rng(int(env_seed))
    # CubeEnv also calls variation_space.seed(None).  Disable that resampling
    # and supply the already seeded values, preserving its normal reset body.
    original_reset(
        seed=int(env_seed),
        options={"variation": [], "variation_values": values},
    )
    return {
        "environment_seed": int(env_seed),
        "variation_seed": int(env_seed),
        "physical_state_seed": int(env_seed),
        "variation_values": {key: value.tolist() for key, value in values.items()},
        "variation_values_sha256": common.combined_array_sha256(
            [values[key] for key in sorted(values)]
        ),
    }


def generate_episode_seedfixed(*args: Any, **kwargs: Any) -> tuple[dict, dict]:
    world = args[0] if args else kwargs["world"]
    original_reset = world.reset
    reset_metadata: dict[str, Any] = {}

    def fixed_reset(seed: int | None = None, options: Any = None) -> None:
        if seed is None or options is not None:
            raise RuntimeError("amended generator requires one explicit environment seed")
        reset_metadata.update(
            deterministic_environment_reset(world, original_reset, int(seed))
        )

    world.reset = fixed_reset
    try:
        arrays, audit = _ORIGINAL_GENERATE_EPISODE(*args, **kwargs)
    finally:
        world.reset = original_reset
    audit.update(
        {
            "environment_reset_seed_forwarding_bug_corrected": True,
            "environment_reset_amendment": reset_metadata,
            "rng_activation_order": (
                "policy.set_seed; deterministic variation-space seed/value freeze; "
                "install env.np_random from env_seed; world.reset with explicit values; "
                "numpy.random.seed(oracle_np_seed); first policy action"
            ),
            "protocol_amendment_01_sha256": common.sha256_file(AMENDMENT_JSON),
        }
    )
    return arrays, audit


def _probe(policy_type: str, env_seed: int) -> dict[str, Any]:
    world, _ = generator.make_world(policy_type)
    original_reset = world.reset
    try:
        metadata = deterministic_environment_reset(world, original_reset, env_seed)
        state = generator._v4_generator._capture(world.infos)
        digest = generator._initial_state_digest(state)
    finally:
        world.close()
    return {
        "policy_constructor": policy_type,
        "initial_state_sha256": digest,
        "reset_metadata": metadata,
    }


def seal_amendment() -> dict[str, Any]:
    common.assert_pre_generation_seal()
    if AMENDMENT_SEAL.exists():
        raise RuntimeError("protocol amendment is already sealed")
    if any((common.STUDY_ROOT / "data").glob("main_*_raw/*")) or any(
        (common.STUDY_ROOT / "data").glob("main_*_manifest.json")
    ):
        raise RuntimeError("cannot amend after any main rollout artifact exists")
    smoke_path = common.STUDY_ROOT / "audit/smoke_pairing.json"
    failed = common.study_json(smoke_path)
    if failed.get("passed") or failed["checks"].get("identical_initial_states"):
        raise RuntimeError("expected the preserved smoke seed-forwarding failure")
    common.write_study_json(FAILED_SMOKE_COPY, failed, exclusive=True)

    smoke_seed = int(
        common.study_json(common.STUDY_ROOT / "seed_manifest.json")["roles"]
        ["smoke"]["plan_oracle"][0]["env_seed"]
    )
    probes = [_probe(policy, smoke_seed) for policy in ("plan_oracle", "markov_oracle")]
    probe_passed = (
        probes[0]["initial_state_sha256"] == probes[1]["initial_state_sha256"]
        and probes[0]["reset_metadata"]["variation_values_sha256"]
        == probes[1]["reset_metadata"]["variation_values_sha256"]
    )
    source = common.study_json(common.STUDY_ROOT / "source_manifest.json")
    installed = source["installed_controlling_sources"]
    cube_path = next(path for path in installed if path.endswith("/cube_env.py") and "stable_worldmodel" in path)
    world_path = next(path for path in installed if path.endswith("/world/world.py"))
    amendment = {
        "schema_version": 1,
        "status": "frozen_after_excluded_smoke_and_before_any_main_rollout",
        "trigger": (
            "world.reset(seed) reached installed CubeEnv.reset, but CubeEnv reset its "
            "variation space with seed=None and omitted seed when calling super().reset"
        ),
        "failed_smoke_pairing_sha256": common.sha256_file(FAILED_SMOKE_COPY),
        "failed_smoke_initial_states_equal": False,
        "main_targets_opened_before_amendment": False,
        "correction": {
            "variation_rng": (
                "sample DEFAULT_VARIATIONS with env_seed; pass values back with variation=[]"
            ),
            "physical_state_rng": (
                "set unwrapped env.np_random=default_rng(env_seed) before normal reset"
            ),
            "policy_rng": "unchanged separate ExpertPolicy wrapper stream",
            "oracle_rng": "unchanged separate NumPy-global stream activated after reset",
            "capture_preprocessing_horizon": "unchanged",
        },
        "smallest_equivalent_design": (
            "retain the exactly 12+12 excluded smoke episodes and their failure; "
            "apply corrected deterministic reset only to the predeclared 90+30 main episodes"
        ),
        "corrected_initial_only_probe": {
            "reused_excluded_smoke_env_seed": smoke_seed,
            "new_policy_episodes": 0,
            "probes": probes,
            "passed": probe_passed,
        },
        "controlling_source_evidence": {
            "stable_cube_env_path": cube_path,
            "stable_cube_env_sha256": installed[cube_path]["sha256"],
            "world_path": world_path,
            "world_sha256": installed[world_path]["sha256"],
            "cube_reset_lines": "776-818: seed parameter is not forwarded; variation seed is None",
            "world_reset_lines": "179-185: seed is forwarded into EnvPool",
        },
        "main_pairing_requirement": "unchanged: all 30 paired initial-state hashes must match exactly",
        "sample_sizes_changed": False,
        "metrics_or_thresholds_changed": False,
        "gate_or_model_changed": False,
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    if not probe_passed:
        raise RuntimeError("corrected initial-only pairing probe failed")
    common.write_study_json(AMENDMENT_JSON, amendment, exclusive=True)
    markdown = f"""# Protocol amendment 01: installed Cube reset seed forwarding

Status: frozen after the permanently excluded 12+12 smoke episodes and before
any main PlanOracle or MarkovOracle rollout.

The smoke pairing audit failed because the installed
`stable_worldmodel.envs.ogbench.CubeEnv.reset(seed=...)` resets its variation
space with `seed=None` and calls `super().reset(...)` without forwarding its
explicit `seed`. Thus `World.reset(seed)` records the seed in `EnvPool` but does
not control Cube start/goal variations or the physical-state RNG. The failed
audit is preserved byte-for-byte at `{FAILED_SMOKE_COPY.name}`.

The smallest equivalent correction samples the four normal Cube default
variations with the predeclared environment seed, supplies those exact values
to the normal reset body, and installs `default_rng(env_seed)` as the unwrapped
environment RNG before reset. Policy-wrapper and oracle RNG streams remain
separate and unchanged. Capture, preprocessing, horizon, sample sizes, metrics,
bands, gate, and decision rules are unchanged.

An initial-state-only probe reused excluded smoke seed `{smoke_seed}` and
produced the same digest for both policy constructors:
`{probes[0]['initial_state_sha256']}`. It generated zero additional policy
episodes. Every one of the 30 main pairs must still pass exact seed and initial
state hashing. No main target had been opened when this amendment was frozen.
"""
    AMENDMENT_MD.write_text(markdown, encoding="utf-8")

    amended_smoke = {
        "schema_version": 2,
        "phase": "smoke",
        "classification": "excluded_api_compatibility_smoke_after_pre-main_amendment",
        "passed": True,
        "passed_meaning": (
            "exact bounded counts, physical separation, exclusion, seed pairing, and "
            "successful pre-main reset correction; original initial-state mismatch retained"
        ),
        "original_pairing_passed": False,
        "original_initial_states_equal": False,
        "checks": {
            "paired_count_exact": failed["checks"]["paired_count_exact"],
            "identical_environment_seed_identifiers": failed["checks"]["identical_environment_seeds"],
            "separate_wrapper_rng": failed["checks"]["separate_wrapper_rng"],
            "separate_oracle_np_rng": failed["checks"]["separate_oracle_np_rng"],
            "physically_separate_paths": failed["checks"]["physically_separate_paths"],
            "distinct_trajectory_ids": failed["checks"]["distinct_trajectory_ids"],
            "permanently_excluded_from_encoding_and_decision": True,
            "seed_forwarding_incompatibility_diagnosed": True,
            "corrected_initial_only_probe_passed": probe_passed,
            "main_pairing_requirement_unchanged": True,
            "v3_test_targets_opened": False,
        },
        "pairs": failed["pairs"],
        "pre_amendment_failed_audit_path": str(FAILED_SMOKE_COPY.relative_to(common.REPO_ROOT)),
        "pre_amendment_failed_audit_sha256": common.sha256_file(FAILED_SMOKE_COPY),
        "protocol_amendment_path": str(AMENDMENT_JSON.relative_to(common.REPO_ROOT)),
        "protocol_amendment_sha256": common.sha256_file(AMENDMENT_JSON),
        "v5_confirmation_episodes": 0,
    }
    common.write_study_json(smoke_path, amended_smoke)

    sealed_paths = [
        Path(__file__).resolve(),
        common.STUDY_ROOT / "report_with_amendment.py",
        common.STUDY_ROOT / "bounded_check_with_amendment.py",
        AMENDMENT_JSON,
        AMENDMENT_MD,
        FAILED_SMOKE_COPY,
        smoke_path,
        common.STUDY_ROOT / "audit/pre_generation_seal.json",
    ]
    missing = [str(path) for path in sealed_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"amendment support files missing: {missing}")
    seal = {
        "schema_version": 1,
        "status": "frozen_before_any_main_rollout",
        "sealed_files": {
            str(path.relative_to(common.REPO_ROOT)): common.sha256_file(path)
            for path in sealed_paths
        },
        "main_targets_opened": False,
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    common.write_study_json(AMENDMENT_SEAL, seal, exclusive=True)
    return seal


def assert_amendment_seal() -> dict[str, Any]:
    seal = common.study_json(AMENDMENT_SEAL)
    if seal.get("status") != "frozen_before_any_main_rollout":
        raise RuntimeError("invalid amendment seal")
    for relative, expected in seal["sealed_files"].items():
        if common.sha256_file(common.REPO_ROOT / relative) != expected:
            raise RuntimeError(f"protocol amendment drift: {relative}")
    return seal


def generate_main(policy_type: str) -> dict[str, Any]:
    assert_amendment_seal()
    generator.generate_episode = generate_episode_seedfixed
    result = generator.run("main", policy_type)
    link_path = common.STUDY_ROOT / "audit" / f"main_{policy_type}_amendment_link.json"
    manifest_path = common.role_raw_paths("main", policy_type)[1]
    link = {
        "schema_version": 1,
        "policy_type": policy_type,
        "manifest_sha256": common.sha256_file(manifest_path),
        "protocol_amendment_01_seal_sha256": common.sha256_file(AMENDMENT_SEAL),
        "all_episode_audits_record_seed_forwarding_correction": all(
            item["generation_audit"].get(
                "environment_reset_seed_forwarding_bug_corrected", False
            )
            for item in result["episodes"]
        ),
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    if not link["all_episode_audits_record_seed_forwarding_correction"]:
        raise RuntimeError("main manifest did not retain seed correction audit")
    common.write_study_json(link_path, link, exclusive=not link_path.exists())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seal-amendment")
    generate = sub.add_parser("generate-main")
    generate.add_argument("policy", choices=("plan_oracle", "markov_oracle"))
    sub.add_parser("audit-main")
    args = parser.parse_args()
    if args.command == "seal-amendment":
        result = seal_amendment()
    elif args.command == "generate-main":
        result = generate_main(args.policy)
    else:
        assert_amendment_seal()
        result = generator.audit_pairing("main")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
