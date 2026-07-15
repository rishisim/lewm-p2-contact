#!/usr/bin/env python3
"""Freeze the distribution-contract protocol before any new rollout."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import common


UPSTREAM_COMMIT = "1d4140997f60c52c6fb0702ec100dc988b18c548"
UPSTREAM_GENERATOR_SHA256 = "120e84e88c046175301ce11c12e9fb0d64d3dece4b03855ec445b328f5b9a2fc"
UPSTREAM_COMMANDS_SHA256 = "ce14c1b1a1b3b6738ba0a42baa27af776b2ee91e025517a752061e4dd0c74cb3"

CORE_METRICS = [
    "raw_action_abs_ge_0_99_rate",
    "raw_action_rms",
    "raw_action_coord4_mean",
    "normalized_action_abs_gt3_rate",
    "history_temporal_change_norm",
    "target_norm",
    "d0_raw_mse",
]


def config_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "lewm_adaptive_compute_distribution_contract",
        "classification": "bounded_discovery_and_diagnostic_control_not_confirmation",
        "sample": {
            "smoke_episodes_per_policy": 12,
            "plan_discovery_episodes": 90,
            "paired_markov_diagnostic_episodes": 30,
            "examples_per_episode": 38,
            "sequential_expansion_forbidden": True,
            "v5_confirmation_episodes": 0,
        },
        "environment_constructor": {
            "callable": "stable_worldmodel.World",
            "environment": "swm/OGBCube-v0",
            "num_envs": 1,
            "max_episode_steps": 200,
            "image_shape": [224, 224],
            "env_type": "single",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
            "terminate_at_goal": False,
            "mode": "data_collection",
        },
        "policy_constructors": {
            policy: {
                "callable": "stable_worldmodel.envs.ogbench.ExpertPolicy",
                "policy_type": policy,
                "action_noise": 0.1,
                "p_random_action": 0.0,
                "noise_smoothing": 0.5,
                "min_norm": 0.4,
                "seed": None,
            }
            for policy in ("plan_oracle", "markov_oracle")
        },
        "rng_contract": {
            "environment": "world.reset(seed=env_seed) before policy RNG activation",
            "wrapper": "ExpertPolicy.set_seed(policy_seed)",
            "oracle": (
                "numpy.random.seed(oracle_np_seed) after environment reset and before first action; "
                "required because OGBench 1.2.1 PlanOracle and MarkovOracle use NumPy global RNG"
            ),
            "pairing": "PlanOracle and MarkovOracle use identical environment seeds and separate wrapper/oracle streams",
        },
        "preprocessing": {
            "pixels": {
                "source": "uint8 RGB pixels",
                "resolution": [224, 224],
                "transform": "V2 model_io.pixel_transform",
                "imagenet_mean": [0.485, 0.456, 0.406],
                "imagenet_std": [0.229, 0.224, 0.225],
            },
            "frameskip": 5,
            "history": 3,
            "latent_dim": 192,
            "raw_action_dim": 5,
            "blocked_action_dim": 25,
            "target_alignment": "model steps 3..40; target latent at exactly the next model step",
            "action_mean": common.FROZEN_ACTION_MEAN.tolist(),
            "action_std": common.FROZEN_ACTION_STD.tolist(),
            "action_statistics_frozen_rows": 2_000_000,
        },
        "frozen_adaptive_contract": {
            "base_weights_sha256": common.EXPECTED["base_weights_sha256"],
            "stagewise_solver_sha256": common.EXPECTED["solver_checkpoint_sha256"],
            "linear_student_sha256": common.EXPECTED["student_checkpoint_sha256"],
            "tournament_sha256": common.EXPECTED["tournament_sha256"],
            "discovery_whitening_sha256": common.EXPECTED["whitening_sha256"],
            "compute_price": common.COMPUTE_PRICE,
            "gate_parameters": common.GATE_PARAMETERS,
            "gate_flops_per_evaluated_decision": common.GATE_FLOPS_PER_DECISION,
            "stopping_rule": "start at d1; continue at each of d1,d2,d3 iff frozen score > frozen compute price",
            "training_or_tuning_from_new_data": False,
        },
        "reference_region": {
            "sources": ["offline_discovery_420", "consumed_offline_calibration_90"],
            "episode_estimand": "equal-weight episode mean; every episode has 38 prediction rows",
            "bootstrap_samples": 10_000,
            "fresh_sample_size": 90,
            "marginal_interval": [0.025, 0.975],
            "joint_region": "95% max-|studentized mean deviation| region over frozen core metrics",
            "core_metrics": CORE_METRICS,
            "contact_or_state_in_primary_region": False,
        },
        "comparison_metrics": {
            "raw_inputs": "pixel RGB moments/temporal change; observation moments; raw actions",
            "latent_inputs": "history/target norms, moments, temporal changes, normalized actions",
            "model": "d0 and d1-d4 raw/discovery-whitened losses; update norms; gains; positive-gain rates",
            "gate": "every causal feature block, frozen-normalized z summaries, scores, rankings, price exceedance, call histograms",
            "composition": "contact, impact, effector/block motion, phase, target task, final success",
            "composition_role": "post-hoc only; excluded from training, inference, policy selection, primary thresholds",
        },
        "runtime_benchmark": {
            "device": "mps",
            "batch_sizes": [1, 32, 256, 1024],
            "warmups": 2,
            "repetitions": 5,
            "measurement": "synchronized end-to-end base prediction plus solver/gate path",
            "statistical_and_flop_verdict_independent_of_latency": True,
        },
        "decision_rules": {
            "distribution_contract_passed": {
                "all_required": [
                    "PlanOracle 90-episode core mean lies inside frozen joint offline max-T region",
                    "PlanOracle is materially closer than paired MarkovOracle: standardized Euclidean distance ratio <= 0.70 and closer on >=5/7 core coordinates",
                    "PlanOracle d1-to-d2 mean marginal gain is >0 (the two offline split-specific means and their frozen pooled band are reported descriptively)",
                    "frozen gate prospective discovery criteria all pass",
                    "all validity, isolation, hash, exact-call, FLOP, and dense/sparse audits pass",
                ]
            },
            "gate_prospective_discovery": {
                "all_required": [
                    "95% episode-bootstrap CI lower bound >0 versus analytic strongest transition-independent exact-total-FLOP mixture",
                    "95% episode-bootstrap CI lower bound >0 versus conservative seeded integer exact-total-FLOP mixture",
                    "95% episode-bootstrap CI lower bound >0 versus fixed d1",
                    "95% episode-bootstrap CI lower bound >0 versus exact call-histogram randomization",
                    "discovery-whitened point benefit >0 versus analytic exact-total-FLOP mixture",
                    "score/gain Spearman is >0 at all three decisions",
                    "adaptive point is nondominated in both call and fully-counted-FLOP frontiers",
                ]
            },
            "generator_reconstruction_failed": (
                "PlanOracle misses the core offline region or is not materially closer than paired MarkovOracle; "
                "permit only the smallest bounded version/action-semantics/replay check, never gate tuning"
            ),
            "distribution_matched_gate_failed": (
                "PlanOracle matches the core/base distribution but frozen gate criteria fail; no V5; "
                "recommend smallest domain-robust gate discovery study"
            ),
            "claims_forbidden": [
                "V5 confirmation",
                "confirmatory claim",
                "contact-aware adaptive claim",
                "first adaptive world model claim",
            ],
        },
        "related_work_note": "LoopWM is relevant adaptive-computation work in text environments.",
    }


def _episode_seed_records() -> dict[str, Any]:
    smoke_env = np.arange(1_640_100_000, 1_640_100_012, dtype=np.int64)
    main_env = np.arange(1_640_200_000, 1_640_200_090, dtype=np.int64)

    def records(
        env: np.ndarray, wrapper_start: int, oracle_start: int, prefix: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "slot": int(slot),
                "trajectory_id": f"{prefix}-{slot:03d}",
                "env_seed": int(seed),
                "policy_seed": int(wrapper_start + slot),
                "oracle_np_seed": int(oracle_start + slot),
            }
            for slot, seed in enumerate(env)
        ]

    return {
        "smoke": {
            "plan_oracle": records(smoke_env, 1_641_100_000, 1_642_100_000, "dc-smoke-plan"),
            "markov_oracle": records(smoke_env, 1_643_100_000, 1_644_100_000, "dc-smoke-markov"),
        },
        "main": {
            "plan_oracle": records(main_env, 1_641_200_000, 1_642_200_000, "dc-plan-discovery"),
            "markov_oracle": records(main_env[:30], 1_643_200_000, 1_644_200_000, "dc-markov-control"),
        },
        "replacement": {
            "plan_oracle": records(
                np.arange(1_640_900_000, 1_640_900_024, dtype=np.int64),
                1_641_900_000,
                1_642_900_000,
                "dc-plan-replacement",
            ),
            "markov_oracle": records(
                np.arange(1_640_910_000, 1_640_910_024, dtype=np.int64),
                1_643_900_000,
                1_644_900_000,
                "dc-markov-replacement",
            ),
        },
    }


STATISTICAL_SEEDS = {
    "reference_joint_max_t": 1_646_000_001,
    "reference_marginal_bands": 1_646_000_002,
    "plan_d1_d2_gain_ci": 1_646_000_003,
    "gate_exact_analytic": 1_646_000_101,
    "gate_exact_seeded": 1_646_000_102,
    "gate_fixed_d1": 1_646_000_103,
    "gate_histogram": 1_646_000_104,
    "gate_white_exact": 1_646_000_105,
    "exact_total_assignment": 1_646_000_201,
    "histogram_randomization": 1_646_000_202,
    "latency_order": 1_646_000_301,
    "independent_bootstrap": 1_646_000_401,
}


def seed_manifest() -> dict[str, Any]:
    roles = _episode_seed_records()
    candidate: set[int] = set(STATISTICAL_SEEDS.values())
    intentional_env_reuse: list[int] = []
    for phase in ("smoke", "main", "replacement"):
        by_policy = roles[phase]
        for entries in by_policy.values():
            for entry in entries:
                candidate.update(
                    (entry["env_seed"], entry["policy_seed"], entry["oracle_np_seed"])
                )
        if "plan_oracle" in by_policy and "markov_oracle" in by_policy:
            plan_env = {item["env_seed"] for item in by_policy["plan_oracle"]}
            markov_env = {item["env_seed"] for item in by_policy["markov_oracle"]}
            intentional_env_reuse.extend(sorted(plan_env & markov_env))

    prior_inventory_path = common.V4_ROOT / "audit/prior_identifier_inventory.json"
    prior = common.study_json(prior_inventory_path)
    prior_ids = {int(value) for value in prior["unique_numeric_ids"]}
    v4_seed_path = common.V4_ROOT / "seed_manifest.json"
    prior_ids.update(common.recursive_integers(common.study_json(v4_seed_path)))
    overlap = sorted(candidate & prior_ids)
    if overlap:
        raise RuntimeError(f"new seed block overlaps recorded prior identifiers: {overlap}")
    return {
        "schema_version": 1,
        "status": "frozen_before_any_new_rollout",
        "selection_rule": (
            "Preallocated contiguous blocks beginning at 1,640,100,000, chosen without "
            "rollout access and accepted only after exact nonmembership in the frozen prior "
            "identifier inventory and V4 seed manifest."
        ),
        "roles": roles,
        "statistical_seeds": STATISTICAL_SEEDS,
        "unique_seed_count": len(candidate),
        "unique_seeds_sha256": common.canonical_int_digest(candidate),
        "intentional_paired_environment_seed_reuse": sorted(set(intentional_env_reuse)),
        "policy_and_oracle_rng_streams_are_never_reused_across_policies": True,
        "overlap_with_recorded_prior_identifiers": overlap,
        "prior_identifier_inventory": {
            "path": str(prior_inventory_path.relative_to(common.REPO_ROOT)),
            "sha256": common.sha256_file(prior_inventory_path),
            "unique_numeric_id_count": int(prior["unique_numeric_id_count"]),
        },
        "v4_seed_manifest_sha256": common.sha256_file(v4_seed_path),
        "failure_replacement_rule": (
            "Only an exception or mechanically malformed trajectory may consume the next "
            "unused policy-specific replacement triple. Outcomes, success, loss, calls, "
            "contact, motion, or any comparison metric may never trigger replacement."
        ),
    }


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def source_manifest(source_h5_sha256: str) -> dict[str, Any]:
    site = Path(
        "/Users/rishisim/Documents/research/World Models/le-wm/.venv/lib/"
        "python3.10/site-packages"
    )
    installed = [
        site / "stable_worldmodel/envs/ogbench/expert_policy.py",
        site / "stable_worldmodel/envs/ogbench/cube_env.py",
        site / "stable_worldmodel/world/world.py",
        site / "ogbench/manipspace/oracles/plan/plan_oracle.py",
        site / "ogbench/manipspace/oracles/plan/cube_plan.py",
        site / "ogbench/manipspace/oracles/markov/markov_oracle.py",
        site / "ogbench/manipspace/oracles/markov/cube_markov.py",
        site / "ogbench/manipspace/envs/manipspace_env.py",
    ]
    missing = [str(path) for path in installed if not path.exists()]
    if missing:
        raise RuntimeError(f"missing controlling installed sources: {missing}")
    stable_repo = Path("/Users/rishisim/Documents/research/World Models/le-wm")
    return {
        "schema_version": 1,
        "generation_interpreter": str(Path(sys.executable).resolve()),
        "evaluation_interpreter": str(common.EVALUATION_PYTHON),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: _version(name)
            for name in (
                "stable-worldmodel",
                "ogbench",
                "gymnasium",
                "mujoco",
                "dm-control",
                "numpy",
                "scipy",
                "torch",
                "transformers",
                "h5py",
                "hdf5plugin",
            )
        },
        "installed_controlling_sources": {
            str(path): {"sha256": common.sha256_file(path), "bytes": path.stat().st_size}
            for path in installed
        },
        "immutable_local_scientific_sources": {
            str(path.relative_to(common.REPO_ROOT)): common.sha256_file(path)
            for path in (
                common.V4_ROOT / "common.py",
                common.V4_ROOT / "runtime.py",
                common.V4_ROOT / "generator.py",
                common.REPO_ROOT / "runs/lewm_adaptive_compute_v2/model_io.py",
                common.DISCOVERY_ROOT / "data_isolation.py",
                common.DISCOVERY_ROOT / "models.py",
                common.DISCOVERY_ROOT / "policy.py",
                common.DISCOVERY_ROOT / "run_discovery.py",
            )
        },
        "stable_worldmodel_local_repo": {
            "path": str(stable_repo),
            "head": _git(stable_repo, "rev-parse", "HEAD"),
            "worktree_status_sha256": common.sha256_bytes(
                _git(stable_repo, "status", "--short").encode("utf-8")
            ),
            "installed_files_pinned_by_bytes_not_cleanliness": True,
        },
        "upstream_ogbench_recipe": {
            "repository": "https://github.com/seohongpark/ogbench",
            "tag": "v1.2.1",
            "commit": UPSTREAM_COMMIT,
            "generate_manipspace_sha256": UPSTREAM_GENERATOR_SHA256,
            "commands_sh_sha256": UPSTREAM_COMMANDS_SHA256,
            "controlling_mapping": {
                "cube-single-play-v0": "dataset_type=play -> CubePlanOracle",
                "cube-single-noisy-v0": "dataset_type=noisy -> CubeMarkovOracle",
            },
            "official_command_difference_retained_as_caveat": (
                "upstream public command uses max_episode_steps=1001; the released local "
                "LeWM artifact and frozen model contract contain 201 states/200 actions, so "
                "this bounded reconstruction uses the local 200-action horizon"
            ),
        },
        "offline_source": {
            "path": str(common.SOURCE_H5),
            "bytes": common.SOURCE_H5.stat().st_size,
            "sha256": source_h5_sha256,
            "huggingface_dataset": "quentinll/lewm-cube",
            "huggingface_revision": "02a19a67a0dc8c9d6215f89c19e0a597691e152a",
            "dataset_card_generation_flags_present": False,
            "policy_equivalence_is_empirically_tested_not_assumed": True,
        },
        "frozen_objects": common.verify_frozen_objects(),
        "v3_combined_cache": {
            "path": str(common.V3 / "cache/cube_inputs.npz"),
            "sha256_opaque_only": common.EXPECTED["v3_combined_cache_opaque_sha256"],
            "opened_with_numpy": False,
            "test_targets_opened": False,
        },
    }


def preregistration_markdown() -> str:
    cfg = config_payload()
    rules = cfg["decision_rules"]
    return f"""# Preregistration: Cube policy distribution contract

Status: frozen before every new smoke or main rollout. This is a bounded
discovery/diagnostic study, not confirmation. No V5 episodes may be generated.

## Controlling generator contract

OGBench v1.2.1 at `{UPSTREAM_COMMIT}` maps `dataset_type=play` to the
non-Markovian `CubePlanOracle` and `dataset_type=noisy` to `CubeMarkovOracle`.
The paired local generator uses the exact same `swm/OGBCube-v0` constructor,
200-action horizon, capture path, and preprocessing for both policies. The
first 30 main environment seeds are identical between policies. Wrapper RNG
and NumPy-global oracle RNG streams are separate and recorded.

## Fixed design

- 12 PlanOracle and 12 MarkovOracle smoke episodes, permanently excluded.
- 90 fresh PlanOracle discovery-judge episodes.
- 30 paired MarkovOracle diagnostic-control episodes on the first 30 Plan seeds.
- No sequential expansion and zero V5 confirmation episodes.

## Isolation

Only the physically isolated 420-episode V3 discovery cache and already-consumed
90-episode calibration cache can supply offline targets. The combined V3 cache
is hash-checked opaquely and never loaded. Raw HDF5 reads are limited by those
two cache allowlists and are used only for raw-input and post-hoc composition
descriptors. Every derived cache proves empty intersection with the pinned V3
test episode set. Contact, impact, state, motion, phase, task, and success are
never solver/gate inputs or primary decision thresholds.

## Frozen reference and decision

The core metrics are: {', '.join(CORE_METRICS)}. A 10,000-replicate,
episode-resampled 90-episode reference distribution supplies marginal 95%
bands and a joint 95% max-T region. It is frozen before new target generation.

`distribution_contract_passed` requires every item in the predeclared JSON
rule, including core-region membership, material paired closeness to Plan over
Markov, restored positive d1-to-d2 gain, all frozen-gate discovery criteria,
and all validity audits. If reconstruction fails, only a smallest bounded
version/action-semantics or replay check is allowed; the gate cannot be tuned.
If distribution matches but the gate fails, V5 remains unauthorized and the
next study is domain-robust gate discovery.

Before any fresh rollout, the offline-only recomputation found that discovery
and calibration had positive but materially different d1-to-d2 means
(`1.65336e-4` and `4.08993e-5`). The required restoration rule is therefore the
directional rule stated in the task (`PlanOracle mean > 0`); the two split means
and pooled episode-bootstrap band are frozen and reported descriptively, not
used as an added equivalence threshold. The seven-metric joint distribution
region remains unchanged and includes policy/action and base-model d0 metrics,
not d1-to-d2 gain.

## Claim limits

This study cannot support a confirmatory, contact-aware, or “first adaptive
world model” claim. V4 remains a valid negative result for its actual
MarkovOracle distribution and is diagnostic only. LoopWM is relevant related
work in text environments.
"""


def initialize(source_h5_sha256: str) -> None:
    if any(common.STUDY_ROOT.glob("data/*_raw/*")):
        raise RuntimeError("cannot initialize after rollout files exist")
    outputs = {
        common.STUDY_ROOT / "config.json": config_payload(),
        common.STUDY_ROOT / "seed_manifest.json": seed_manifest(),
        common.STUDY_ROOT / "source_manifest.json": source_manifest(source_h5_sha256),
    }
    for path, payload in outputs.items():
        if path.exists():
            raise RuntimeError(f"initial protocol artifact already exists: {path}")
        common.write_study_json(path, payload, exclusive=True)
    prereg = common.STUDY_ROOT / "PREREGISTRATION.md"
    if prereg.exists():
        raise RuntimeError("PREREGISTRATION.md already exists")
    prereg.write_text(preregistration_markdown(), encoding="utf-8")
    print(json.dumps({"initialized": True, "v3_test_targets_opened": False}, sort_keys=True))


def seal() -> None:
    seal_path = common.STUDY_ROOT / "audit/pre_generation_seal.json"
    if seal_path.exists():
        raise RuntimeError("pre-generation seal already exists")
    forbidden = list((common.STUDY_ROOT / "data").glob("*_oracle_raw")) + list(
        (common.STUDY_ROOT / "data").glob("*_oracle_manifest.json")
    )
    if forbidden:
        raise RuntimeError(f"rollout artifacts predate seal: {forbidden}")
    required = [
        common.STUDY_ROOT / "config.json",
        common.STUDY_ROOT / "seed_manifest.json",
        common.STUDY_ROOT / "source_manifest.json",
        common.STUDY_ROOT / "PREREGISTRATION.md",
        common.STUDY_ROOT / "reference/reference_bands.json",
        common.STUDY_ROOT / "reference/offline_episode_metrics.npz",
        common.STUDY_ROOT / "audit/offline_isolation.json",
        common.STUDY_ROOT / "audit/raw_reference_access.json",
    ]
    required.extend(sorted(common.STUDY_ROOT.glob("*.py")))
    required.extend(sorted((common.STUDY_ROOT / "tests").glob("*.py")))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot seal missing artifacts: {missing}")
    source = common.study_json(common.STUDY_ROOT / "source_manifest.json")
    if source["v3_combined_cache"]["opened_with_numpy"] or source["v3_combined_cache"]["test_targets_opened"]:
        raise RuntimeError("V3 combined/test isolation contract failed")
    payload = {
        "schema_version": 1,
        "status": "frozen_before_any_smoke_or_main_rollout",
        "sealed_files": {
            str(path.relative_to(common.REPO_ROOT)): common.sha256_file(path)
            for path in sorted(set(required))
        },
        "reference_bands_sha256": common.sha256_file(
            common.STUDY_ROOT / "reference/reference_bands.json"
        ),
        "v3_test_targets_opened": False,
        "v5_confirmation_episodes": 0,
    }
    common.write_study_json(seal_path, payload, exclusive=True)
    print(json.dumps({"sealed": True, "files": len(payload["sealed_files"])}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    initial = sub.add_parser("initialize")
    initial.add_argument("--source-h5-sha256", required=True)
    sub.add_parser("seal")
    args = parser.parse_args()
    if args.command == "initialize":
        if len(args.source_h5_sha256) != 64:
            raise ValueError("source HDF5 SHA-256 must be 64 hex characters")
        initialize(args.source_h5_sha256)
    else:
        seal()


if __name__ == "__main__":
    main()
