#!/usr/bin/env python3
"""Fresh, role-isolated Cube rollout generation for the v001 gate study.

The generator intentionally persists only raw pixels and actions.  Episode
retention depends solely on the sealed shapes/dtypes/finiteness contract and
locally counted reset/step bookkeeping.  A mechanical rollout exception may
consume a prospectively assigned replacement; any persistence error stops the
process on the same identifier.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from inherited_authorization import (
    ACTIVE_ATTEMPT,
    INHERITANCE_SEAL_PATH,
    SCIENCE_ATTEMPT,
    InheritedAuthorizationError,
    verify_inherited_pre_data_authorization,
)
from input_loader import (
    ACTION_SHAPE,
    INPUT_ALLOWLIST,
    PIXELS_SHAPE,
    array_sha256,
    load_model_gate_inputs,
    validate_model_gate_inputs,
)
from runtime_contract import verify_here
from study_common import (
    ATTEMPT_ROOT,
    REPO_ROOT,
    atomic_json,
    atomic_npz,
    read_json,
    read_verified_controller,
    sha256_file,
)


ATTEMPT = ACTIVE_ATTEMPT
ROLES = ("fit", "selection", "smoke", "confirmation")
EXPECTED_REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
RAW_ROWS = 201
ROLLOUT_STEPS = 200
RAW_ACTION_DIM = 5

DGP_MATRIX_PATH = ATTEMPT_ROOT / "DGP_MATRIX.json"
SEED_LEDGER_PATH = ATTEMPT_ROOT / "cohort_seed_ledger.json"
PRE_DATA_SEAL_PATH = INHERITANCE_SEAL_PATH
PRE_CONFIRMATION_SEAL_PATH = (
    ATTEMPT_ROOT / "audit/pre_confirmation_package_seal.json"
)
REPLACEMENT_REGISTRY_PATH = ATTEMPT_ROOT / "data/replacement_registry.json"
REPLACEMENT_CLAIMS_ROOT = ATTEMPT_ROOT / "data/replacement_claims"
REPLACEMENT_REGISTRY_LOCK_PATH = (
    ATTEMPT_ROOT / "data/replacement_registry.lock"
)
RESET_REFERENCE_PATH = (
    REPO_ROOT
    / "runs/lewm_adaptive_compute_distribution_contract/generator_seedfix.py"
)
SEED_FIELDS = (
    "env_seed",
    "policy_seed",
    "oracle_np_seed",
    "action_space_seed",
)
RECORD_FIELDS = ("slot", "episode_id", *SEED_FIELDS)
REGIME_SLUGS = {
    "native_plan": "np",
    "markov_oracle": "mo",
    "plan_action_noise_0p2": "n2",
    "plan_random_action_0p1": "r1",
}
ROLE_SLUGS = {
    "fit": "ft",
    "selection": "sl",
    "smoke": "sm",
    "confirmation": "cf",
}
EPISODE_ID_PATTERN = re.compile(
    r"drgv001-(?:np|mo|n2|r1)-(?:ft|sl|sm|cf)-(?:p|r)-[0-9]{3,4}"
)
PERSISTENCE_INTENT_SCHEMA_VERSION = 2
PERSISTENCE_INTENT_STATUS = "valid_in_memory_episode_closed_for_raw_persistence"
ORPHAN_ADOPTION_RULE = (
    "adopt only after exact closed-intent, current authorization, prospective "
    "ledger, canonical path, nonlink identity, and fully recomputed two-array "
    "verification"
)
RAW_MANIFEST_ORPHAN_POLICY = (
    "adopt raw-only archive only after exact closed intent, current authorization, "
    "prospective ledger, canonical nonlink paths, and full two-array recomputation; "
    "otherwise stop"
)
RNG_ACTIVATION_ORDER = (
    "policy.set_seed; deterministic V5 seed-forwarded reset; action-space seed; "
    "numpy oracle seed; first policy action"
)
PERSISTENCE_INTENT_KEYS = frozenset(
    {
        "schema_version",
        "attempt",
        "science_attempt",
        "created_unix_ns",
        "status",
        "role",
        "regime",
        "slot",
        "episode_id",
        "raw_path",
        "raw_sidecar_path",
        "persistence_intent_path",
        "dgp_matrix_path",
        "dgp_matrix_sha256",
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "destination_seed_record",
        "seed_source_pool",
        "seed_source_slot",
        "seed_source_record",
        "seed_source_episode_id",
        "replacement_used",
        "replacement_claim_index",
        "replacement_claim_sha256",
        *SEED_FIELDS,
        "arrays",
        "initial_pixels_sha256",
        "generation_audit",
        "authorization_seal",
        "orphan_adoption_rule",
    }
)
GENERATION_AUDIT_KEYS = frozenset(
    {
        "rows",
        "rollout_steps_expected",
        "rollout_steps_completed",
        "pixel_frames_captured",
        "initial_pixels_sha256",
        "input_validation",
        "reset_metadata",
        "action_space_seed",
        "action_spaces_seeded",
        "rng_activation_order",
        "only_pixels_and_actions_captured",
        "retention_uses_only_input_contract_and_local_step_bookkeeping",
    }
)
RESET_METADATA_KEYS = frozenset(
    {
        "environment_seed",
        "variation_seed",
        "physical_state_seed",
        "variation_values_sha256",
        "reset_reference_path",
        "reset_reference_sha256",
        "reset_values_persisted",
    }
)
_PERSISTENCE_SOURCE_HASH_CACHE: dict[tuple[Any, ...], str] = {}


class GenerationContractError(RuntimeError):
    """A sealed generation input or existing artifact violates the contract."""


class GenerationPreflightError(RuntimeError):
    """A package-level failure occurred before an episode identifier was used."""


class EpisodeRolloutError(RuntimeError):
    """A mechanical rollout failed before a valid in-memory episode existed."""


class EpisodePersistenceError(RuntimeError):
    """A valid in-memory episode could not be durably persisted or resumed."""


class ReplacementPoolExhausted(EpisodeRolloutError):
    """Every sealed replacement tuple available to a slot has been consumed."""


@dataclass(frozen=True)
class SealLink:
    path: Path
    relative_path: str
    sha256: str
    checkpoint_state: str
    payload: Mapping[str, Any]

    def as_json(self) -> dict[str, str]:
        return {
            "path": self.relative_path,
            "sha256": self.sha256,
            "checkpoint_state": self.checkpoint_state,
        }


@dataclass(frozen=True)
class EpisodePaths:
    raw: Path
    sidecar: Path
    intent: Path


@dataclass(frozen=True)
class GenerationInputs:
    dgp_matrix: Mapping[str, Any]
    seed_ledger: Mapping[str, Any]
    seal: SealLink
    regime_specification: Mapping[str, Any]
    primary: tuple[dict[str, Any], ...]
    replacements: tuple[dict[str, Any], ...]


def _relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical_json(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise GenerationContractError("nonfinite value cannot enter JSON mechanics")
    return value


def _seed_tuple(record: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(record[name]) for name in SEED_FIELDS)  # type: ignore[return-value]


def _source_identity(record: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(record["episode_id"]), _seed_tuple(record)


def _expected_episode_id(
    *, regime: str, role: str, pool_name: str, index: int
) -> str:
    if regime not in REGIME_SLUGS or role not in ROLE_SLUGS:
        raise GenerationContractError(f"unknown episode identity context: {regime}/{role}")
    if pool_name not in {"primary", "replacements"} or index < 0:
        raise GenerationContractError("invalid episode identity pool/index")
    kind = "p" if pool_name == "primary" else "r"
    width = 4 if role == "confirmation" and pool_name == "primary" else 3
    return (
        f"drgv001-{REGIME_SLUGS[regime]}-{ROLE_SLUGS[role]}-"
        f"{kind}-{index:0{width}d}"
    )


def _validate_episode_id(value: Any, *, label: str) -> str:
    episode_id = str(value)
    if (
        EPISODE_ID_PATTERN.fullmatch(episode_id) is None
        or Path(episode_id).name != episode_id
        or any(token in episode_id for token in ("/", "\\", ".", ".."))
    ):
        raise GenerationContractError(f"noncanonical or unsafe episode_id in {label}")
    return episode_id


def _validate_seed_record(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_episode_id: str | None = None,
) -> dict[str, Any]:
    missing = [name for name in RECORD_FIELDS if name not in record]
    if missing:
        raise GenerationContractError(f"{label} is missing fields: {missing}")
    normalized = dict(record)
    normalized["slot"] = int(record["slot"])
    normalized["episode_id"] = _validate_episode_id(
        record["episode_id"], label=label
    )
    if normalized["slot"] < 0 or not normalized["episode_id"]:
        raise GenerationContractError(f"invalid slot or episode_id in {label}")
    if (
        expected_episode_id is not None
        and normalized["episode_id"] != expected_episode_id
    ):
        raise GenerationContractError(
            f"episode_id does not match its sealed role/regime/slot in {label}"
        )
    for field in SEED_FIELDS:
        normalized[field] = int(record[field])
        if normalized[field] < 0:
            raise GenerationContractError(f"negative {field} in {label}")
    return normalized


def validate_dgp_matrix(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("attempt") != SCIENCE_ATTEMPT:
        raise GenerationContractError("DGP matrix attempt mismatch")
    order = tuple(payload.get("regime_order", ()))
    regimes = payload.get("regimes")
    if order != EXPECTED_REGIMES or not isinstance(regimes, Mapping):
        raise GenerationContractError("DGP matrix is not the sealed four-regime order")
    if tuple(regimes) != EXPECTED_REGIMES:
        raise GenerationContractError("DGP mapping order or membership drift")
    required = {
        "policy_type",
        "action_noise",
        "p_random_action",
        "single_changed_factor",
    }
    expected_policy_values = {
        "native_plan": ("plan_oracle", 0.1, 0.0),
        "markov_oracle": ("markov_oracle", 0.1, 0.0),
        "plan_action_noise_0p2": ("plan_oracle", 0.2, 0.0),
        "plan_random_action_0p1": ("plan_oracle", 0.1, 0.1),
    }
    for name in EXPECTED_REGIMES:
        specification = regimes[name]
        if not isinstance(specification, Mapping) or not required.issubset(
            specification
        ):
            raise GenerationContractError(f"incomplete DGP specification: {name}")
        if specification["policy_type"] not in ("plan_oracle", "markov_oracle"):
            raise GenerationContractError(f"unsupported policy type in {name}")
        for key in ("action_noise", "p_random_action"):
            value = float(specification[key])
            if not np.isfinite(value) or value < 0:
                raise GenerationContractError(f"invalid {key} in {name}")
        observed_policy_values = (
            str(specification["policy_type"]),
            float(specification["action_noise"]),
            float(specification["p_random_action"]),
        )
        if observed_policy_values != expected_policy_values[name]:
            raise GenerationContractError(
                f"one-factor DGP values drift in {name}: {observed_policy_values}"
            )
    common = payload.get("environment_common")
    expected_common = {
        "environment": "swm/OGBCube-v0",
        "env_type": "single",
        "num_envs": 1,
        "mode": "data_collection",
        "height": 224,
        "width": 224,
        "image_shape": [224, 224],
        "max_episode_steps": 200,
        "terminate_at_goal": False,
        "visualize_info": False,
        "noise_smoothing": 0.5,
        "min_norm": 0.4,
        "deterministic_reset": "exact V5 seed-forwarding correction",
    }
    if not isinstance(common, Mapping) or any(
        common.get(key) != value for key, value in expected_common.items()
    ):
        raise GenerationContractError("environment constructor contract drift")
    return dict(payload)


def validate_seed_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("attempt") != SCIENCE_ATTEMPT:
        raise GenerationContractError("cohort seed ledger attempt mismatch")
    if tuple(payload.get("regime_order", ())) != EXPECTED_REGIMES or tuple(
        payload.get("role_order", ())
    ) != ROLES:
        raise GenerationContractError("cohort seed ledger order drift")
    if payload.get("role_primary_counts_per_regime") != {
        "fit": 300,
        "selection": 500,
        "smoke": 6,
        "confirmation": 4_500,
    } or payload.get("replacement_count_per_regime_per_role") != 200:
        raise GenerationContractError("cohort seed ledger count contract drift")
    freshness_checks = payload.get("checks")
    if not isinstance(freshness_checks, Mapping) or not freshness_checks or any(
        value is not True for value in freshness_checks.values()
    ):
        raise GenerationContractError("cohort seed ledger freshness checks do not pass")
    if (
        payload.get("fresh_outcome_episodes_opened") != 0
        or payload.get("prior_outcome_arrays_opened") != 0
    ):
        raise GenerationContractError("cohort ledger was not built outcome-blind")
    regimes = payload.get("regimes")
    if not isinstance(regimes, Mapping) or set(regimes) != set(EXPECTED_REGIMES):
        raise GenerationContractError("cohort seed ledger regime membership drift")
    episode_ids: set[str] = set()
    rng_identifiers: dict[str, set[int]] = {name: set() for name in SEED_FIELDS}
    for regime in EXPECTED_REGIMES:
        regime_payload = regimes[regime]
        roles = regime_payload.get("roles") if isinstance(regime_payload, Mapping) else None
        if not isinstance(roles, Mapping) or set(roles) != set(ROLES):
            raise GenerationContractError(f"role membership drift in {regime}")
        for role in ROLES:
            role_payload = roles[role]
            if not isinstance(role_payload, Mapping) or set(role_payload) != {
                "primary",
                "replacements",
            }:
                raise GenerationContractError(
                    f"seed role schema drift in {regime}/{role}"
                )
            for pool_name in ("primary", "replacements"):
                pool = role_payload[pool_name]
                if not isinstance(pool, list):
                    raise GenerationContractError(
                        f"seed pool is not a list: {regime}/{role}/{pool_name}"
                    )
                for index, raw_record in enumerate(pool):
                    if not isinstance(raw_record, Mapping):
                        raise GenerationContractError("seed record is not an object")
                    record = _validate_seed_record(
                        raw_record,
                        label=f"{regime}/{role}/{pool_name}/{index}",
                        expected_episode_id=_expected_episode_id(
                            regime=regime,
                            role=role,
                            pool_name=pool_name,
                            index=index,
                        ),
                    )
                    if record["episode_id"] in episode_ids:
                        raise GenerationContractError(
                            f"duplicate episode identifier: {record['episode_id']}"
                        )
                    episode_ids.add(record["episode_id"])
                    for field in SEED_FIELDS:
                        if record[field] in rng_identifiers[field]:
                            raise GenerationContractError(
                                f"duplicate {field} identifier: {record[field]}"
                            )
                        rng_identifiers[field].add(record[field])
                if pool_name == "primary":
                    observed_slots = [int(item["slot"]) for item in pool]
                    if observed_slots != list(range(len(pool))):
                        raise GenerationContractError(
                            f"noncontiguous primary slots: {regime}/{role}"
                        )
    return dict(payload)


def _zero_count(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) == 0


def verify_authorization_seal(role: str) -> SealLink:
    if role not in ROLES:
        raise GenerationContractError(f"unknown generation role: {role}")
    if role in ("fit", "selection"):
        path = PRE_DATA_SEAL_PATH
        expected_state = "PRE_OUTCOME_SEAL"
        zero_integer_fields = (
            "fit_outcome_episodes",
            "selection_outcome_episodes",
            "smoke_outcome_episodes",
            "confirmation_outcome_episodes_generated",
            "confirmation_outcome_episodes_executed",
        )
    else:
        path = PRE_CONFIRMATION_SEAL_PATH
        expected_state = "PRE_CONFIRMATION_PACKAGE_SEAL"
        zero_integer_fields = (
            "smoke_outcome_episodes",
            "confirmation_outcome_episodes_generated",
            "confirmation_outcome_episodes_executed",
        )
    if role in ("fit", "selection"):
        expected_current_state = {
            "fit": "FIT_COHORTS",
            "selection": "SELECTION_COHORTS",
        }[role]
        try:
            lineage = verify_inherited_pre_data_authorization(
                expected_current_state=expected_current_state
            )
        except InheritedAuthorizationError as error:
            raise GenerationPreflightError(
                "inherited pre-data authorization failed before seed or output use"
            ) from error
        if (
            lineage.get("seal_path") != _relative(path)
            or lineage.get("seal_sha256") != sha256_file(path)
        ):
            raise GenerationPreflightError(
                "inherited authorization returned the wrong seal identity"
            )
    if not path.is_file():
        raise GenerationPreflightError(f"authorization seal is missing: {path}")
    payload = read_json(path)
    if (
        payload.get("passed") is not True
        or payload.get("attempt") != ATTEMPT
        or payload.get("checkpoint_state") != expected_state
    ):
        raise GenerationPreflightError(
            f"authorization seal header is invalid for {role}"
        )
    counts = payload.get("outcome_counts_at_seal")
    if not isinstance(counts, Mapping):
        raise GenerationPreflightError("authorization seal lacks outcome counts")
    if any(not _zero_count(counts.get(field)) for field in zero_integer_fields):
        raise GenerationPreflightError(
            f"authorization seal has a nonzero later-role count for {role}"
        )
    if counts.get("confirmation_outcomes_opened_for_analysis") is not False:
        raise GenerationPreflightError(
            "authorization seal was written after confirmation analysis opened"
        )
    if role in ("smoke", "confirmation") and (
        counts.get("fit_outcome_episodes") != 4 * 300
        or counts.get("selection_outcome_episodes") != 4 * 500
    ):
        raise GenerationPreflightError(
            "pre-confirmation seal does not record the exact completed fit/selection cohorts"
        )
    sealed_files = payload.get("sealed_files")
    if not isinstance(sealed_files, Mapping) or not sealed_files:
        raise GenerationPreflightError("authorization seal has no sealed_files map")
    bad: list[str] = []
    normalized_sealed: dict[str, str] = {}
    for raw_relative, raw_expected in sealed_files.items():
        relative = str(raw_relative)
        expected = str(raw_expected)
        candidate_relative = Path(relative)
        if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
            bad.append(relative)
            continue
        candidate = (REPO_ROOT / candidate_relative).resolve()
        if not candidate.is_relative_to(REPO_ROOT.resolve()):
            bad.append(relative)
            continue
        if not candidate.is_file() or sha256_file(candidate) != expected:
            bad.append(relative)
            continue
        normalized_sealed[_relative(candidate)] = expected
    required_files = (
        DGP_MATRIX_PATH,
        SEED_LEDGER_PATH,
        Path(__file__).resolve(),
        ATTEMPT_ROOT / "input_loader.py",
        ATTEMPT_ROOT / "study_common.py",
    )
    missing_required = [
        _relative(required)
        for required in required_files
        if normalized_sealed.get(_relative(required))
        != sha256_file(required)
    ]
    if bad or missing_required:
        raise GenerationPreflightError(
            f"authorization seal hash failure: bad={bad} missing={missing_required}"
        )
    return SealLink(
        path=path,
        relative_path=_relative(path),
        sha256=sha256_file(path),
        checkpoint_state=expected_state,
        payload=payload,
    )


def _confirmation_prefix_length(
    seal: SealLink, regime: str, maximum: int
) -> int:
    counts = seal.payload.get("confirmation_episode_count_per_regime")
    if not isinstance(counts, Mapping) or set(counts) != set(EXPECTED_REGIMES):
        raise GenerationPreflightError(
            "pre-confirmation seal lacks the fixed common confirmation counts"
        )
    integer_counts = {name: int(counts[name]) for name in EXPECTED_REGIMES}
    if len(set(integer_counts.values())) != 1:
        raise GenerationPreflightError("confirmation prefix is not common across DGPs")
    count = integer_counts[regime]
    if count <= 0 or count > maximum or count % 500 != 0:
        raise GenerationPreflightError(
            f"invalid frozen confirmation prefix length: {count}"
        )
    return count


def load_generation_inputs(role: str, regime: str) -> GenerationInputs:
    if role not in ROLES or regime not in EXPECTED_REGIMES:
        raise GenerationContractError(f"invalid role/regime: {role}/{regime}")
    seal = verify_authorization_seal(role)
    dgp_matrix = validate_dgp_matrix(read_json(DGP_MATRIX_PATH))
    seed_ledger = validate_seed_ledger(read_json(SEED_LEDGER_PATH))
    sealed_files = seal.payload["sealed_files"]
    for path in (DGP_MATRIX_PATH, SEED_LEDGER_PATH):
        relative = _relative(path)
        if sealed_files.get(relative) != sha256_file(path):
            raise GenerationPreflightError(
                f"authorization seal does not cross-link {relative}"
            )
    role_payload = seed_ledger["regimes"][regime]["roles"][role]
    primary = tuple(
        _validate_seed_record(
            item,
            label=f"{regime}/{role}/primary/{index}",
            expected_episode_id=_expected_episode_id(
                regime=regime,
                role=role,
                pool_name="primary",
                index=index,
            ),
        )
        for index, item in enumerate(role_payload["primary"])
    )
    replacements = tuple(
        _validate_seed_record(
            item,
            label=f"{regime}/{role}/replacement/{index}",
            expected_episode_id=_expected_episode_id(
                regime=regime,
                role=role,
                pool_name="replacements",
                index=index,
            ),
        )
        for index, item in enumerate(role_payload["replacements"])
    )
    expected_count = int(
        dgp_matrix["roles_per_regime"][
            "confirmation_maximum_assigned" if role == "confirmation" else role
        ]["episodes"]
    )
    if len(primary) != expected_count:
        raise GenerationPreflightError(
            f"sealed primary count mismatch for {role}/{regime}"
        )
    expected_replacements = int(
        dgp_matrix["replacement_tuples_per_regime_per_role"]
    )
    if len(replacements) != expected_replacements:
        raise GenerationPreflightError(
            f"sealed replacement count mismatch for {role}/{regime}"
        )
    if role == "confirmation":
        prefix = _confirmation_prefix_length(seal, regime, len(primary))
        primary = primary[:prefix]
    return GenerationInputs(
        dgp_matrix=dgp_matrix,
        seed_ledger=seed_ledger,
        seal=seal,
        regime_specification=dgp_matrix["regimes"][regime],
        primary=primary,
        replacements=replacements,
    )


def verify_controller_permission(role: str) -> dict[str, Any]:
    expected_states = {
        "fit": "FIT_COHORTS",
        "selection": "SELECTION_COHORTS",
        "smoke": "EXCLUDED_MECHANICAL_SMOKE",
        "confirmation": "CONFIRMATION_GENERATION",
    }
    state = read_verified_controller()
    checks = {
        "active_attempt": state.get("active_attempt") == ATTEMPT,
        "current_state": state.get("current_state") == expected_states[role],
        "terminal_unset": state.get("terminal_label") is None,
        "confirmation_not_terminal": state.get("confirmation_terminal") is False,
    }
    if not all(checks.values()):
        raise GenerationPreflightError(
            f"controller does not authorize {role} generation: {checks}"
        )
    return state


def make_world(regime_specification: Mapping[str, Any]) -> tuple[Any, Any]:
    """Construct the exact Cube world and selected one-factor policy DGP."""
    import stable_worldmodel as swm
    from stable_worldmodel.envs.ogbench import ExpertPolicy

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=ROLLOUT_STEPS,
        image_shape=(224, 224),
        env_type="single",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
    )
    policy = ExpertPolicy(
        policy_type=str(regime_specification["policy_type"]),
        action_noise=float(regime_specification["action_noise"]),
        p_random_action=float(regime_specification["p_random_action"]),
        noise_smoothing=0.5,
        min_norm=0.4,
        seed=None,
    )
    world.set_policy(policy)
    observed = {
        "policy_type": policy.type,
        "action_noise": float(policy.action_noise),
        "p_random_action": float(policy.p_random_action),
        "noise_smoothing": float(policy.noise_smoothing),
        "min_norm": float(policy.min_norm),
    }
    expected = {
        "policy_type": str(regime_specification["policy_type"]),
        "action_noise": float(regime_specification["action_noise"]),
        "p_random_action": float(regime_specification["p_random_action"]),
        "noise_smoothing": 0.5,
        "min_norm": 0.4,
    }
    if observed != expected:
        world.close()
        raise GenerationPreflightError(
            f"constructed policy does not match sealed DGP: {observed} != {expected}"
        )
    return world, policy


def deterministic_environment_reset(
    world: Any, original_reset: Callable[..., None], env_seed: int
) -> dict[str, Any]:
    """Apply the exact V5 deterministic Cube reset mechanics.

    The reset values are used only to drive the installed reset API.  The
    sidecar records their combined digest, never the values themselves.
    """
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
    env.np_random = np.random.default_rng(int(env_seed))
    original_reset(
        seed=int(env_seed),
        options={"variation": [], "variation_values": values},
    )
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        digest.update(array_sha256(values[key]).encode("ascii"))
    return {
        "environment_seed": int(env_seed),
        "variation_seed": int(env_seed),
        "physical_state_seed": int(env_seed),
        "variation_values_sha256": digest.hexdigest(),
        "reset_reference_path": _relative(RESET_REFERENCE_PATH),
        "reset_reference_sha256": sha256_file(RESET_REFERENCE_PATH),
        "reset_values_persisted": False,
    }


def seed_action_spaces(world: Any, seed: int) -> list[str]:
    candidates = [
        ("vector_action_space", getattr(world.envs, "action_space", None)),
        (
            "unwrapped_action_space",
            getattr(world.envs.envs[0].unwrapped, "action_space", None),
        ),
    ]
    seeded: list[str] = []
    seen: set[int] = set()
    for name, action_space in candidates:
        if action_space is None or id(action_space) in seen:
            continue
        seen.add(id(action_space))
        seed_method = getattr(action_space, "seed", None)
        if not callable(seed_method):
            raise EpisodeRolloutError(f"{name} has no deterministic seed method")
        seed_method(int(seed))
        seeded.append(name)
    if not seeded:
        raise EpisodeRolloutError("no action-space RNG was explicitly seeded")
    return seeded


def _capture_pixels(infos: Mapping[str, Any]) -> np.ndarray:
    """Read the sole environment channel retained by the generator."""
    try:
        value = infos["pixels"]
    except (KeyError, TypeError) as error:
        raise EpisodeRolloutError("environment lacks the required pixels channel") from error
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != PIXELS_SHAPE[1:] or array.dtype != np.uint8:
        raise EpisodeRolloutError(
            f"unexpected captured pixels: shape={array.shape} dtype={array.dtype}"
        )
    if not np.isfinite(array).all():
        raise EpisodeRolloutError("captured pixels contain a nonfinite value")
    return array.copy()


def generate_episode(
    world: Any,
    policy: Any,
    *,
    role: str,
    regime: str,
    slot: int,
    episode_id: str,
    env_seed: int,
    policy_seed: int,
    oracle_np_seed: int,
    action_space_seed: int,
    reset_environment: Callable[[Any, Callable[..., None], int], Mapping[str, Any]] = deterministic_environment_reset,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Roll out and mechanically validate one exactly 200-step episode."""
    del role, regime, slot, episode_id  # labels never influence trajectory retention
    try:
        policy.set_seed(int(policy_seed))
        reset_metadata = dict(
            reset_environment(world, world.reset, int(env_seed))
        )
        action_spaces_seeded = seed_action_spaces(world, int(action_space_seed))
        np.random.seed(int(oracle_np_seed))
        pixel_frames = [_capture_pixels(world.infos)]
        actions: list[np.ndarray] = []
        steps_completed = 0
        for step_index in range(ROLLOUT_STEPS):
            action = np.asarray(policy.get_action(world.infos), dtype=np.float32)
            if action.shape != (1, RAW_ACTION_DIM) or not np.isfinite(action).all():
                raise EpisodeRolloutError(
                    f"invalid expert action at local step {step_index}: {action.shape}"
                )
            actions.append(action[0].copy())
            step_result = world.envs.step(action)
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                raise EpisodeRolloutError(
                    f"environment step API drift at local step {step_index}"
                )
            infos = step_result[4]
            if not isinstance(infos, Mapping):
                raise EpisodeRolloutError("environment info container is not a mapping")
            world.infos = infos
            pixel_frames.append(_capture_pixels(infos))
            steps_completed += 1
        arrays = {
            "pixels": np.stack(pixel_frames, axis=0),
            "action": np.vstack(
                (
                    np.asarray(actions, dtype=np.float32),
                    np.full((1, RAW_ACTION_DIM), np.nan, dtype=np.float32),
                )
            ),
        }
        validation = validate_model_gate_inputs(arrays)
        if steps_completed != ROLLOUT_STEPS or len(pixel_frames) != RAW_ROWS:
            raise EpisodeRolloutError(
                "local non-outcome step bookkeeping is not exactly 200/201"
            )
        audit = {
            "rows": RAW_ROWS,
            "rollout_steps_expected": ROLLOUT_STEPS,
            "rollout_steps_completed": steps_completed,
            "pixel_frames_captured": len(pixel_frames),
            "initial_pixels_sha256": array_sha256(arrays["pixels"][0]),
            "input_validation": validation,
            "reset_metadata": _canonical_json(reset_metadata),
            "action_space_seed": int(action_space_seed),
            "action_spaces_seeded": action_spaces_seeded,
            "rng_activation_order": RNG_ACTIVATION_ORDER,
            "only_pixels_and_actions_captured": True,
            "retention_uses_only_input_contract_and_local_step_bookkeeping": True,
        }
        return arrays, audit
    except EpisodeRolloutError:
        raise
    except (
        FloatingPointError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise EpisodeRolloutError(
            f"mechanical episode rollout exception: {type(error).__name__}: {error}"
        ) from error


def raw_directory(role: str, regime: str) -> Path:
    if role not in ROLES or regime not in EXPECTED_REGIMES:
        raise GenerationContractError(f"invalid output role/regime: {role}/{regime}")
    return ATTEMPT_ROOT / "data" / role / regime / "raw"


def raw_manifest_path(role: str, regime: str) -> Path:
    return ATTEMPT_ROOT / "data" / role / regime / "raw_manifest.json"


def failure_log_path(role: str, regime: str) -> Path:
    return ATTEMPT_ROOT / "data" / role / regime / "rollout_failures.jsonl"


def materialization_lock_path(role: str, regime: str) -> Path:
    if role not in ROLES or regime not in EXPECTED_REGIMES:
        raise GenerationContractError(f"invalid lock role/regime: {role}/{regime}")
    return ATTEMPT_ROOT / f".materialization-{role}-{regime}.lock"


@contextmanager
def materialization_lock(role: str, regime: str) -> Iterator[None]:
    """Admit exactly one generator/runner writer for a role-by-DGP cell."""

    path = materialization_lock_path(role, regime)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    locked = False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise GenerationPreflightError(
                f"materialization lock is not a regular file: {path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as error:
            raise GenerationPreflightError(
                f"concurrent materialization already holds {role}/{regime}"
            ) from error
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_destination_episode_id(episode_id: Any, *, role: str, regime: str) -> str:
    value = _validate_episode_id(
        episode_id, label=f"output/{regime}/{role}"
    )
    prefix = f"drgv001-{REGIME_SLUGS[regime]}-{ROLE_SLUGS[role]}-p-"
    digits = value.removeprefix(prefix)
    expected_width = 4 if role == "confirmation" else 3
    if (
        not value.startswith(prefix)
        or len(digits) != expected_width
        or not digits.isdigit()
    ):
        raise GenerationContractError(
            "output episode_id does not match its canonical role/regime primary slug"
        )
    return value


def _lexical_attempt_relative(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ATTEMPT_ROOT / path
    try:
        relative = path.relative_to(ATTEMPT_ROOT)
    except ValueError as error:
        raise GenerationPreflightError(f"output path escapes active attempt: {path}") from error
    if not relative.parts or ".." in relative.parts:
        raise GenerationPreflightError(f"noncanonical output path: {path}")
    return relative


def _assert_safe_directory_chain(
    directory: Path,
    *,
    seen_inodes: dict[tuple[int, int], Path],
) -> None:
    # A root-level output leaf (the per-cell materialization lock) has the
    # attempt root itself as its parent.  That parent is a valid empty
    # ancestry chain, but it must be admitted only here: the general lexical
    # path helper and output-leaf checker continue to reject ATTEMPT_ROOT as a
    # leaf.  Validate and register the root before returning so a linked,
    # non-directory, or inode-aliased attempt root still fails closed.
    directory = Path(directory)
    if directory == ATTEMPT_ROOT:
        if not os.path.lexists(directory):
            raise GenerationPreflightError(
                f"active attempt root is absent: {directory}"
            )
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise GenerationPreflightError(
                f"active attempt root is linked or not a directory: {directory}"
            )
        if directory.resolve(strict=True) != ATTEMPT_ROOT.resolve(strict=True):
            raise GenerationPreflightError(
                f"active attempt root aliases another path: {directory}"
            )
        identity = (int(info.st_dev), int(info.st_ino))
        previous = seen_inodes.get(identity)
        if previous is not None and previous != directory:
            raise GenerationPreflightError(
                f"distinct output parents alias one inode: {previous}, {directory}"
            )
        seen_inodes[identity] = directory
        return
    relative = _lexical_attempt_relative(directory)
    current = ATTEMPT_ROOT
    for part in relative.parts:
        current = current / part
        if not os.path.lexists(current):
            continue
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise GenerationPreflightError(
                f"existing output parent is linked or not a directory: {current}"
            )
        canonical = (ATTEMPT_ROOT.resolve(strict=True) / current.relative_to(ATTEMPT_ROOT))
        if current.resolve(strict=True) != canonical.resolve(strict=False):
            raise GenerationPreflightError(f"existing output parent aliases another path: {current}")
        identity = (int(info.st_dev), int(info.st_ino))
        previous = seen_inodes.get(identity)
        if previous is not None and previous != current:
            raise GenerationPreflightError(
                f"distinct output parents alias one inode: {previous}, {current}"
            )
        seen_inodes[identity] = current


def _assert_safe_output_leaf(
    path: Path,
    *,
    seen_directories: dict[tuple[int, int], Path],
    seen_files: dict[tuple[int, int], Path],
) -> None:
    _lexical_attempt_relative(path)
    _assert_safe_directory_chain(path.parent, seen_inodes=seen_directories)
    if not os.path.lexists(path):
        return
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
    ):
        raise GenerationPreflightError(
            f"existing output leaf is linked/aliased or not a regular file: {path}"
        )
    identity = (int(info.st_dev), int(info.st_ino))
    previous = seen_files.get(identity)
    if previous is not None and previous != path:
        raise GenerationPreflightError(
            f"distinct output files alias one inode: {previous}, {path}"
        )
    seen_files[identity] = path


def _canonical_output_file(root: Path, filename: str) -> Path:
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise GenerationContractError(f"unsafe output filename: {filename}")
    _lexical_attempt_relative(root)
    candidate = root / filename
    if candidate.parent != root or not candidate.is_relative_to(ATTEMPT_ROOT):
        raise GenerationPreflightError(f"output path is not root-contained: {candidate}")
    # Existing linked ancestry is rejected here as well as by the package-wide
    # preflight so standalone path construction can never silently traverse it.
    _assert_safe_directory_chain(root, seen_inodes={})
    resolved_root = root.resolve(strict=False)
    if not candidate.resolve(strict=False).is_relative_to(resolved_root):
        raise GenerationPreflightError(f"resolved output path escapes canonical root: {candidate}")
    return candidate


def episode_paths(role: str, regime: str, episode_id: str) -> EpisodePaths:
    episode_id = _validate_destination_episode_id(
        episode_id, role=role, regime=regime
    )
    directory = raw_directory(role, regime)
    intent_directory = (
        ATTEMPT_ROOT / "data/persistence_intents" / role / regime
    )
    return EpisodePaths(
        raw=_canonical_output_file(directory, f"{episode_id}.npz"),
        sidecar=_canonical_output_file(directory, f"{episode_id}.json"),
        intent=_canonical_output_file(
            intent_directory, f"{episode_id}.json"
        ),
    )


def preflight_output_namespace(
    role: str,
    regime: str,
    primary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject linked, noncanonical, or aliased outputs before any mutation."""

    directories: dict[tuple[int, int], Path] = {}
    files: dict[tuple[int, int], Path] = {}
    raw_root = raw_directory(role, regime)
    intent_root = ATTEMPT_ROOT / "data/persistence_intents" / role / regime
    for root in (raw_root, intent_root, REPLACEMENT_CLAIMS_ROOT):
        _assert_safe_directory_chain(root, seen_inodes=directories)
    fixed_leaves = (
        raw_manifest_path(role, regime),
        failure_log_path(role, regime),
        materialization_lock_path(role, regime),
        REPLACEMENT_REGISTRY_PATH,
        REPLACEMENT_REGISTRY_LOCK_PATH,
    )
    for path in fixed_leaves:
        _assert_safe_output_leaf(
            path, seen_directories=directories, seen_files=files
        )
    if os.path.lexists(REPLACEMENT_CLAIMS_ROOT):
        for claim_path in sorted(REPLACEMENT_CLAIMS_ROOT.iterdir()):
            _assert_safe_output_leaf(
                claim_path, seen_directories=directories, seen_files=files
            )
    observed_ids: list[str] = []
    for destination in primary:
        episode_id = _validate_destination_episode_id(
            destination.get("episode_id"), role=role, regime=regime
        )
        paths = episode_paths(role, regime, episode_id)
        for path in (paths.raw, paths.sidecar, paths.intent):
            _assert_safe_output_leaf(
                path, seen_directories=directories, seen_files=files
            )
        observed_ids.append(episode_id)
    if len(observed_ids) != len(set(observed_ids)):
        raise GenerationPreflightError("output namespace contains duplicate episode IDs")
    return {
        "passed": True,
        "role": role,
        "regime": regime,
        "episode_count": len(observed_ids),
        "existing_directory_count": len(directories),
        "existing_file_count": len(files),
        "filesystem_mutations": 0,
    }


def _failure_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _canonical_json(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _failure_genesis(seal: SealLink) -> str:
    return f"authorization:{seal.sha256}"


def _strict_failure_records(
    raw: bytes,
    *,
    role: str,
    regime: str,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    seal: SealLink,
) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise EpisodePersistenceError("rollout failure chain has a partial final line")
    primary_by_destination = {
        (str(item["episode_id"]), int(item["slot"])): dict(item)
        for item in primary
    }
    replacement_by_source = {
        str(item["episode_id"]): dict(item) for item in replacements
    }
    previous = _failure_genesis(seal)
    observed_attempts: set[tuple[str, str]] = set()
    failures: list[dict[str, Any]] = []
    for sequence, encoded in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise EpisodePersistenceError(
                f"invalid rollout failure log line {sequence}"
            ) from error
        if not isinstance(value, dict):
            raise EpisodePersistenceError("rollout failure record is not an object")
        canonical_line = json.dumps(
            value, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        if encoded != canonical_line:
            raise EpisodePersistenceError(
                f"rollout failure line {sequence} is not canonical JSON"
            )
        payload = dict(value)
        observed_hash = payload.pop("record_sha256", None)
        slot_value = payload.get("slot")
        destination_key = (
            str(payload.get("episode_id", "")),
            slot_value if type(slot_value) is int else -1,
        )
        destination = primary_by_destination.get(destination_key)
        source_id = str(payload.get("attempted_seed_source_episode_id", ""))
        replacement_used = payload.get("replacement_used")
        candidate = (
            replacement_by_source.get(source_id)
            if replacement_used is True
            else destination
        )
        expected_keys = {
            "schema_version",
            "attempt",
            "created_unix_ns",
            "classification",
            "role",
            "regime",
            "slot",
            "episode_id",
            "attempted_seed_source_episode_id",
            "replacement_used",
            "replacement_claim_index",
            "replacement_claim_sha256",
            *SEED_FIELDS,
            "exception_type",
            "exception_message",
            "traceback",
            "replacement_permitted",
            "retention_contract",
            "authorization_seal",
            "seq",
            "prev_sha256",
        }
        checks = {
            "keys": set(payload) == expected_keys,
            "schema": type(payload.get("schema_version")) is int
            and payload.get("schema_version") == 2,
            "attempt": payload.get("attempt") == ATTEMPT,
            "role": payload.get("role") == role,
            "regime": payload.get("regime") == regime,
            "classification": payload.get("classification")
            == "mechanical_rollout_exception",
            "slot": type(slot_value) is int,
            "sequence": type(payload.get("seq")) is int
            and payload.get("seq") == sequence,
            "previous": payload.get("prev_sha256") == previous,
            "authorization": payload.get("authorization_seal") == seal.as_json(),
            "destination": destination is not None,
            "replacement_flag": type(replacement_used) is bool,
            "candidate": candidate is not None,
            "timestamp": type(payload.get("created_unix_ns")) is int
            and int(payload["created_unix_ns"]) > 0,
            "exception": isinstance(payload.get("exception_type"), str)
            and bool(payload.get("exception_type"))
            and isinstance(payload.get("exception_message"), str)
            and isinstance(payload.get("traceback"), str),
            "replacement_permitted": payload.get("replacement_permitted") is True,
            "retention": payload.get("retention_contract")
            == "pixels_action_shapes_finiteness_and_local_step_count_only",
        }
        if destination is not None and replacement_used is False:
            checks["primary_source"] = source_id == destination["episode_id"]
            checks["primary_claim"] = payload.get("replacement_claim_index") is None
            checks["primary_claim_hash"] = (
                payload.get("replacement_claim_sha256") is None
            )
        if candidate is not None:
            checks["candidate_seeds"] = _seed_tuple(payload) == _seed_tuple(candidate)
        if replacement_used is True:
            checks["replacement_claim"] = type(
                payload.get("replacement_claim_index")
            ) is int and int(payload["replacement_claim_index"]) >= 0
            checks["replacement_claim_hash"] = isinstance(
                payload.get("replacement_claim_sha256"), str
            ) and _is_sha256(payload.get("replacement_claim_sha256"))
        attempt_key = (destination_key[0], source_id)
        checks["attempt_unique"] = attempt_key not in observed_attempts
        checks["hash"] = observed_hash == _failure_hash(payload)
        if not all(checks.values()):
            raise EpisodePersistenceError(
                f"rollout failure authentication drift at line {sequence}: {checks}"
            )
        observed_attempts.add(attempt_key)
        previous = str(observed_hash)
        failures.append(value)
    return failures


def _failure_descriptor(path: Path, *, create: bool) -> int:
    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise EpisodePersistenceError(f"failure log is not a regular file: {path}")
    return descriptor


def read_rollout_failures(
    role: str,
    regime: str,
    *,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    seal: SealLink,
) -> list[dict[str, Any]]:
    path = failure_log_path(role, regime)
    if not os.path.lexists(path):
        return []
    descriptor = _failure_descriptor(path, create=False)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = b""
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            raw += block
        if not raw:
            raise EpisodePersistenceError(
                "present rollout failure log is empty; zero failures require absence"
            )
        return _strict_failure_records(
            raw,
            role=role,
            regime=regime,
            primary=primary,
            replacements=replacements,
            seal=seal,
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def record_rollout_failure(
    *,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replacement_used: bool,
    error: EpisodeRolloutError,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    seal: SealLink,
) -> dict[str, Any]:
    if type(replacement_used) is not bool:
        raise EpisodePersistenceError("replacement_used must be an exact boolean")
    base = {
        "schema_version": 2,
        "attempt": ATTEMPT,
        "created_unix_ns": time.time_ns(),
        "classification": "mechanical_rollout_exception",
        "role": role,
        "regime": regime,
        "slot": int(destination["slot"]),
        "episode_id": str(destination["episode_id"]),
        "attempted_seed_source_episode_id": str(candidate["episode_id"]),
        "replacement_used": replacement_used,
        "replacement_claim_index": candidate.get("replacement_claim_index"),
        "replacement_claim_sha256": candidate.get("replacement_claim_sha256"),
        **{field: int(candidate[field]) for field in SEED_FIELDS},
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "replacement_permitted": True,
        "retention_contract": "pixels_action_shapes_finiteness_and_local_step_count_only",
        "authorization_seal": seal.as_json(),
    }
    path = failure_log_path(role, regime)
    try:
        preexisting = os.path.lexists(path)
        descriptor = _failure_descriptor(path, create=True)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.lseek(descriptor, 0, os.SEEK_SET)
            raw = b""
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                raw += block
            if preexisting and not raw:
                raise EpisodePersistenceError(
                    "present rollout failure log is empty; refusing append"
                )
            existing = _strict_failure_records(
                raw,
                role=role,
                regime=regime,
                primary=primary,
                replacements=replacements,
                seal=seal,
            )
            record = {
                **base,
                "seq": len(existing) + 1,
                "prev_sha256": (
                    existing[-1]["record_sha256"]
                    if existing
                    else _failure_genesis(seal)
                ),
            }
            record["record_sha256"] = _failure_hash(record)
            encoded = (
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError("short append to rollout failure log")
            os.fsync(descriptor)
            verified = _strict_failure_records(
                raw + encoded,
                role=role,
                regime=regime,
                primary=primary,
                replacements=replacements,
                seal=seal,
            )
            if verified[-1] != record:
                raise EpisodePersistenceError(
                    "persisted rollout failure did not verify exactly"
                )
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    except Exception as persistence_error:
        raise EpisodePersistenceError(
            "rollout failed but its failure record could not be persisted; "
            "the seed cannot advance"
        ) from persistence_error
    return dict(record)


@contextmanager
def _registry_lock(path: Path = REPLACEMENT_REGISTRY_LOCK_PATH) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _registry_authorization_policy() -> dict[str, dict[str, str]]:
    return {
        role: {
            "path": _relative(
                PRE_DATA_SEAL_PATH
                if role in ("fit", "selection")
                else PRE_CONFIRMATION_SEAL_PATH
            ),
            "checkpoint_state": (
                "PRE_OUTCOME_SEAL"
                if role in ("fit", "selection")
                else "PRE_CONFIRMATION_PACKAGE_SEAL"
            ),
        }
        for role in ROLES
    }


def _registry_genesis() -> dict[str, Any]:
    ledger = read_json(SEED_LEDGER_PATH)
    payload = {
        "schema_version": 2,
        "record_type": "replacement_registry_genesis",
        "attempt": ATTEMPT,
        "science_attempt": SCIENCE_ATTEMPT,
        "scope": "append_only_all_roles_and_all_dgps",
        "created_unix_ns": time.time_ns(),
        "dgp_matrix_path": _relative(DGP_MATRIX_PATH),
        "dgp_matrix_sha256": sha256_file(DGP_MATRIX_PATH),
        "cohort_seed_ledger_path": _relative(SEED_LEDGER_PATH),
        "cohort_seed_ledger_sha256": sha256_file(SEED_LEDGER_PATH),
        "regime_order": list(EXPECTED_REGIMES),
        "role_order": list(ROLES),
        "seed_field_order": list(SEED_FIELDS),
        "replacement_count_per_regime_per_role": 200,
        "replacement_rule": _canonical_json(ledger.get("replacement_rule")),
        "claims_directory": _relative(REPLACEMENT_CLAIMS_ROOT),
        "claim_filename_format": "{claim_index:06d}.json",
        "authorization_policy": _registry_authorization_policy(),
        "claim_contract": (
            "exclusive immutable canonical JSON segments; contiguous global indexes; "
            "SHA-256 predecessor chain; exact sealed-ledger tuple; exact role seal; "
            "authenticated mechanical-failure trigger; global source single-use"
        ),
    }
    payload["record_sha256"] = _failure_hash(payload)
    return payload


def _validate_registry_genesis(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "record_type",
        "attempt",
        "science_attempt",
        "scope",
        "created_unix_ns",
        "dgp_matrix_path",
        "dgp_matrix_sha256",
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "regime_order",
        "role_order",
        "seed_field_order",
        "replacement_count_per_regime_per_role",
        "replacement_rule",
        "claims_directory",
        "claim_filename_format",
        "authorization_policy",
        "claim_contract",
        "record_sha256",
    }
    body = dict(payload)
    observed_hash = body.pop("record_sha256", None)
    ledger = read_json(SEED_LEDGER_PATH)
    checks = {
        "keys": set(payload) == expected_keys,
        "schema": payload.get("schema_version") == 2,
        "record_type": payload.get("record_type")
        == "replacement_registry_genesis",
        "attempt": payload.get("attempt") == ATTEMPT,
        "science_attempt": payload.get("science_attempt") == SCIENCE_ATTEMPT,
        "scope": payload.get("scope") == "append_only_all_roles_and_all_dgps",
        "timestamp": type(payload.get("created_unix_ns")) is int
        and int(payload["created_unix_ns"]) > 0,
        "dgp_path": payload.get("dgp_matrix_path") == _relative(DGP_MATRIX_PATH),
        "dgp_hash": payload.get("dgp_matrix_sha256")
        == sha256_file(DGP_MATRIX_PATH),
        "ledger_path": payload.get("cohort_seed_ledger_path")
        == _relative(SEED_LEDGER_PATH),
        "ledger_hash": payload.get("cohort_seed_ledger_sha256")
        == sha256_file(SEED_LEDGER_PATH),
        "regime_order": payload.get("regime_order") == list(EXPECTED_REGIMES),
        "role_order": payload.get("role_order") == list(ROLES),
        "seed_fields": payload.get("seed_field_order") == list(SEED_FIELDS),
        "replacement_count": payload.get("replacement_count_per_regime_per_role")
        == 200,
        "replacement_rule": payload.get("replacement_rule")
        == _canonical_json(ledger.get("replacement_rule")),
        "claims_directory": payload.get("claims_directory")
        == _relative(REPLACEMENT_CLAIMS_ROOT),
        "filename_format": payload.get("claim_filename_format")
        == "{claim_index:06d}.json",
        "authorization_policy": payload.get("authorization_policy")
        == _registry_authorization_policy(),
        "claim_contract": payload.get("claim_contract")
        == (
            "exclusive immutable canonical JSON segments; contiguous global indexes; "
            "SHA-256 predecessor chain; exact sealed-ledger tuple; exact role seal; "
            "authenticated mechanical-failure trigger; global source single-use"
        ),
        "record_hash": observed_hash == _failure_hash(body),
    }
    if not all(checks.values()):
        raise EpisodePersistenceError(
            f"replacement registry genesis drift: {checks}"
        )
    return dict(payload)


def _canonical_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise EpisodePersistenceError(f"{label} escapes repository: {path}") from error
    current = REPO_ROOT
    for part in relative.parts:
        current = current / part
        try:
            ancestor_info = current.lstat()
        except FileNotFoundError as error:
            raise EpisodePersistenceError(f"missing {label}: {path}") from error
        if stat.S_ISLNK(ancestor_info.st_mode):
            raise EpisodePersistenceError(f"{label} has a symlink path component: {path}")
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise EpisodePersistenceError(f"missing {label}: {path}") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
    ):
        raise EpisodePersistenceError(f"{label} is linked or non-regular: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise EpisodePersistenceError(f"invalid JSON in {label}: {path}") from error
    if not isinstance(value, dict) or raw != (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"):
        raise EpisodePersistenceError(f"noncanonical {label}: {path}")
    return value


def _validate_registry_claim(
    value: Mapping[str, Any],
    *,
    index: int,
    previous: str,
    genesis: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "record_type",
        "attempt",
        "science_attempt",
        "claim_index",
        "claimed_unix_ns",
        "prev_sha256",
        "registry_genesis_path",
        "registry_genesis_file_sha256",
        "registry_genesis_record_sha256",
        "role",
        "regime",
        "slot",
        "episode_id",
        "replacement_slot",
        "replacement_episode_id",
        *SEED_FIELDS,
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "authorization_seal",
        "trigger_failure_log_path",
        "trigger_failure_seq",
        "trigger_failure_record_sha256",
        "failed_seed_source_episode_id",
        "record_sha256",
    }
    body = dict(value)
    observed_hash = body.pop("record_sha256", None)
    role = value.get("role")
    regime = value.get("regime")
    slot = value.get("slot")
    replacement_slot = value.get("replacement_slot")
    authorization = value.get("authorization_seal")
    policy = genesis.get("authorization_policy", {}).get(role, {})
    checks = {
        "keys": set(value) == expected_keys,
        "schema": value.get("schema_version") == 2,
        "record_type": value.get("record_type") == "replacement_claim",
        "attempt": value.get("attempt") == ATTEMPT,
        "science_attempt": value.get("science_attempt") == SCIENCE_ATTEMPT,
        "index": type(value.get("claim_index")) is int
        and value.get("claim_index") == index,
        "timestamp": type(value.get("claimed_unix_ns")) is int
        and int(value["claimed_unix_ns"]) > 0,
        "previous": value.get("prev_sha256") == previous,
        "genesis_path": value.get("registry_genesis_path")
        == _relative(REPLACEMENT_REGISTRY_PATH),
        "genesis_file_hash": value.get("registry_genesis_file_sha256")
        == sha256_file(REPLACEMENT_REGISTRY_PATH),
        "genesis_record_hash": value.get("registry_genesis_record_sha256")
        == genesis.get("record_sha256"),
        "role": role in ROLES,
        "regime": regime in EXPECTED_REGIMES,
        "slot": type(slot) is int and int(slot) >= 0,
        "replacement_slot": type(replacement_slot) is int
        and 0 <= int(replacement_slot) < 200,
        "ledger_path": value.get("cohort_seed_ledger_path")
        == _relative(SEED_LEDGER_PATH),
        "ledger_hash": value.get("cohort_seed_ledger_sha256")
        == sha256_file(SEED_LEDGER_PATH),
        "authorization_schema": isinstance(authorization, Mapping)
        and set(authorization) == {"path", "sha256", "checkpoint_state"},
        "authorization_path": isinstance(authorization, Mapping)
        and authorization.get("path") == policy.get("path"),
        "authorization_state": isinstance(authorization, Mapping)
        and authorization.get("checkpoint_state")
        == policy.get("checkpoint_state"),
        "authorization_hash": isinstance(authorization, Mapping)
        and _is_sha256(authorization.get("sha256")),
        "failure_path": role in ROLES
        and regime in EXPECTED_REGIMES
        and value.get("trigger_failure_log_path")
        == _relative(failure_log_path(str(role), str(regime))),
        "failure_sequence": type(value.get("trigger_failure_seq")) is int
        and int(value["trigger_failure_seq"]) > 0,
        "failure_hash": _is_sha256(value.get("trigger_failure_record_sha256")),
        "failed_source": isinstance(value.get("failed_seed_source_episode_id"), str)
        and bool(value.get("failed_seed_source_episode_id")),
        "record_hash": observed_hash == _failure_hash(body),
    }
    if isinstance(authorization, Mapping) and isinstance(
        authorization.get("path"), str
    ):
        relative_authorization = Path(str(authorization["path"]))
        authorization_path = REPO_ROOT / relative_authorization
        checks["authorization_file"] = (
            not relative_authorization.is_absolute()
            and ".." not in relative_authorization.parts
            and relative_authorization.as_posix() == authorization["path"]
            and authorization_path.is_file()
            and not authorization_path.is_symlink()
            and authorization.get("sha256") == sha256_file(authorization_path)
        )
    else:
        checks["authorization_file"] = False
    if role in ROLES and regime in EXPECTED_REGIMES and type(slot) is int:
        checks["destination_id"] = value.get("episode_id") == _expected_episode_id(
            regime=str(regime), role=str(role), pool_name="primary", index=int(slot)
        )
    if (
        role in ROLES
        and regime in EXPECTED_REGIMES
        and type(replacement_slot) is int
        and 0 <= int(replacement_slot) < 200
    ):
        checks["replacement_id"] = value.get(
            "replacement_episode_id"
        ) == _expected_episode_id(
            regime=str(regime),
            role=str(role),
            pool_name="replacements",
            index=int(replacement_slot),
        )
    try:
        _seed_tuple(value)
    except (KeyError, TypeError, ValueError):
        checks["seeds"] = False
    else:
        checks["seeds"] = all(type(value.get(field)) is int for field in SEED_FIELDS)
    try:
        ledger = read_json(SEED_LEDGER_PATH)
        role_payload = ledger["regimes"][str(regime)]["roles"][str(role)]
        ledger_destination = _validate_seed_record(
            role_payload["primary"][int(slot)],
            label=f"{regime}/{role}/primary/{slot}",
            expected_episode_id=_expected_episode_id(
                regime=str(regime),
                role=str(role),
                pool_name="primary",
                index=int(slot),
            ),
        )
        ledger_replacement = _validate_seed_record(
            role_payload["replacements"][int(replacement_slot)],
            label=f"{regime}/{role}/replacement/{replacement_slot}",
            expected_episode_id=_expected_episode_id(
                regime=str(regime),
                role=str(role),
                pool_name="replacements",
                index=int(replacement_slot),
            ),
        )
    except (KeyError, IndexError, TypeError, ValueError, GenerationContractError):
        checks["ledger_destination"] = False
        checks["ledger_replacement"] = False
    else:
        checks["ledger_destination"] = (
            value.get("episode_id") == ledger_destination["episode_id"]
            and int(value.get("slot", -1)) == ledger_destination["slot"]
        )
        checks["ledger_replacement"] = (
            value.get("replacement_episode_id")
            == ledger_replacement["episode_id"]
            and int(value.get("replacement_slot", -1))
            == ledger_replacement["slot"]
            and _seed_tuple(value) == _seed_tuple(ledger_replacement)
        )
    if not all(checks.values()):
        raise EpisodePersistenceError(
            f"replacement claim authentication drift at index {index}: {checks}"
        )
    return dict(value)


def _load_registry(
    path: Path = REPLACEMENT_REGISTRY_PATH,
    claims_root: Path = REPLACEMENT_CLAIMS_ROOT,
) -> dict[str, Any]:
    genesis = _validate_registry_genesis(_canonical_json_file(path, "registry genesis"))
    if os.path.lexists(claims_root):
        info = claims_root.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise EpisodePersistenceError("replacement claims root is linked or non-directory")
        entries = sorted(claims_root.iterdir(), key=lambda item: item.name)
    else:
        entries = []
    expected_names = [f"{index:06d}.json" for index in range(len(entries))]
    if [entry.name for entry in entries] != expected_names:
        raise EpisodePersistenceError("replacement claim segments are not one contiguous prefix")
    claims: list[dict[str, Any]] = []
    previous = str(genesis["record_sha256"])
    source_ids: set[str] = set()
    source_seeds: set[tuple[int, int, int, int]] = set()
    trigger_hashes: set[str] = set()
    for index, claim_path in enumerate(entries):
        claim = _validate_registry_claim(
            _canonical_json_file(claim_path, "replacement claim segment"),
            index=index,
            previous=previous,
            genesis=genesis,
        )
        source_id = str(claim["replacement_episode_id"])
        seeds = _seed_tuple(claim)
        trigger = str(claim["trigger_failure_record_sha256"])
        if source_id in source_ids or seeds in source_seeds or trigger in trigger_hashes:
            raise EpisodePersistenceError(
                "replacement claim chain reuses a source, seed tuple, or failure trigger"
            )
        source_ids.add(source_id)
        source_seeds.add(seeds)
        trigger_hashes.add(trigger)
        previous = str(claim["record_sha256"])
        claims.append(claim)
    return {
        "schema_version": 2,
        "attempt": ATTEMPT,
        "scope": "append_only_all_roles_and_all_dgps",
        "genesis": genesis,
        "claims": claims,
        "head_record_sha256": previous,
    }


def replacement_registry_prefix(
    *,
    registry_path: Path = REPLACEMENT_REGISTRY_PATH,
    claims_root: Path = REPLACEMENT_CLAIMS_ROOT,
    permitted_existing_roles: Sequence[str] = ROLES,
) -> dict[str, Any]:
    """Describe one immutable genesis/segment prefix for a stage seal."""

    registry = _load_registry(registry_path, claims_root)
    permitted = tuple(permitted_existing_roles)
    if (
        not permitted
        or any(role not in ROLES for role in permitted)
        or any(claim["role"] not in permitted for claim in registry["claims"])
    ):
        raise EpisodePersistenceError(
            "replacement registry contains a role outside the prospective stage prefix"
        )
    claim_files = [
        {
            "path": _relative(claims_root / f"{index:06d}.json"),
            "sha256": sha256_file(claims_root / f"{index:06d}.json"),
            "record_sha256": claim["record_sha256"],
        }
        for index, claim in enumerate(registry["claims"])
    ]
    return {
        "schema_version": 1,
        "storage": "immutable_genesis_and_exclusive_hash_chained_claim_segments",
        "genesis": {
            "path": _relative(registry_path),
            "sha256": sha256_file(registry_path),
            "record_sha256": registry["genesis"]["record_sha256"],
        },
        "claims_directory": _relative(claims_root),
        "claim_count": len(claim_files),
        "head_record_sha256": registry["head_record_sha256"],
        "claim_files": claim_files,
        "permitted_existing_roles": list(permitted),
        "cohort_seed_ledger_path": _relative(SEED_LEDGER_PATH),
        "cohort_seed_ledger_sha256": sha256_file(SEED_LEDGER_PATH),
        "dgp_matrix_path": _relative(DGP_MATRIX_PATH),
        "dgp_matrix_sha256": sha256_file(DGP_MATRIX_PATH),
    }


def ensure_replacement_registry(
    registry_path: Path = REPLACEMENT_REGISTRY_PATH,
    lock_path: Path = REPLACEMENT_REGISTRY_LOCK_PATH,
    claims_root: Path = REPLACEMENT_CLAIMS_ROOT,
) -> dict[str, Any]:
    """Create immutable genesis or validate its exclusive claim-segment chain."""
    try:
        with _registry_lock(lock_path):
            if not registry_path.exists():
                atomic_json(registry_path, _registry_genesis(), exclusive=True)
            claims_root.mkdir(parents=True, exist_ok=True)
            return _load_registry(registry_path, claims_root)
    except EpisodePersistenceError:
        raise
    except Exception as error:
        raise EpisodePersistenceError(
            "global replacement registry could not be initialized"
        ) from error


def _destination_matches(
    claim: Mapping[str, Any], *, role: str, regime: str, destination: Mapping[str, Any]
) -> bool:
    return (
        claim.get("role") == role
        and claim.get("regime") == regime
        and int(claim.get("slot", -1)) == int(destination["slot"])
        and claim.get("episode_id") == str(destination["episode_id"])
    )


def existing_replacement_claims(
    *,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    registry_path: Path = REPLACEMENT_REGISTRY_PATH,
    lock_path: Path = REPLACEMENT_REGISTRY_LOCK_PATH,
    claims_root: Path = REPLACEMENT_CLAIMS_ROOT,
) -> list[dict[str, Any]]:
    with _registry_lock(lock_path):
        registry = _load_registry(registry_path, claims_root)
        return [
            dict(claim)
            for claim in registry["claims"]
            if _destination_matches(
                claim, role=role, regime=regime, destination=destination
            )
        ]


def claim_next_replacement(
    replacements: Sequence[Mapping[str, Any]],
    *,
    primary: Sequence[Mapping[str, Any]],
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    trigger_failure: Mapping[str, Any],
    seal: SealLink,
    failed_source_ids: set[str] | None = None,
    registry_path: Path = REPLACEMENT_REGISTRY_PATH,
    lock_path: Path = REPLACEMENT_REGISTRY_LOCK_PATH,
    claims_root: Path = REPLACEMENT_CLAIMS_ROOT,
) -> dict[str, Any]:
    """Atomically publish the next ledger tuple as one immutable claim segment."""
    try:
        with _registry_lock(lock_path):
            registry = _load_registry(registry_path, claims_root)
            failed = failed_source_ids or set()
            for existing in registry["claims"]:
                if not _destination_matches(
                    existing,
                    role=role,
                    regime=regime,
                    destination=destination,
                ):
                    continue
                if str(existing["replacement_episode_id"]) in failed:
                    continue
                if existing.get("trigger_failure_record_sha256") != trigger_failure.get(
                    "record_sha256"
                ):
                    continue
                return _candidate_from_claim(existing, replacements)
            used_ids = {
                str(claim["replacement_episode_id"])
                for claim in registry["claims"]
            }
            used_seed_tuples = {
                _seed_tuple(claim) for claim in registry["claims"]
            }
            trigger_body = dict(trigger_failure)
            trigger_hash = trigger_body.pop("record_sha256", None)
            if not (
                trigger_failure.get("schema_version") == 2
                and trigger_hash == _failure_hash(trigger_body)
                and trigger_failure.get("authorization_seal") == seal.as_json()
                and trigger_failure.get("role") == role
                and trigger_failure.get("regime") == regime
                and trigger_failure.get("slot") == destination.get("slot")
                and trigger_failure.get("episode_id") == destination.get("episode_id")
                and trigger_failure.get("attempted_seed_source_episode_id") in failed
            ):
                raise EpisodePersistenceError(
                    "replacement claim lacks its exact authenticated mechanical failure"
                )
            authenticated_failures = read_rollout_failures(
                role,
                regime,
                primary=primary,
                replacements=replacements,
                seal=seal,
            )
            if not any(item == dict(trigger_failure) for item in authenticated_failures):
                raise EpisodePersistenceError(
                    "replacement trigger is not in the authenticated failure chain"
                )
            for replacement_slot, raw_candidate in enumerate(replacements):
                candidate = _validate_seed_record(
                    raw_candidate, label=f"{regime}/{role}/replacement"
                )
                if (
                    candidate["episode_id"] in used_ids
                    or _seed_tuple(candidate) in used_seed_tuples
                ):
                    continue
                claim: dict[str, Any] = {
                    "schema_version": 2,
                    "record_type": "replacement_claim",
                    "attempt": ATTEMPT,
                    "science_attempt": SCIENCE_ATTEMPT,
                    "claim_index": len(registry["claims"]),
                    "claimed_unix_ns": time.time_ns(),
                    "prev_sha256": registry["head_record_sha256"],
                    "registry_genesis_path": _relative(registry_path),
                    "registry_genesis_file_sha256": sha256_file(registry_path),
                    "registry_genesis_record_sha256": registry["genesis"][
                        "record_sha256"
                    ],
                    "role": role,
                    "regime": regime,
                    "slot": int(destination["slot"]),
                    "episode_id": str(destination["episode_id"]),
                    "replacement_slot": replacement_slot,
                    "replacement_episode_id": str(candidate["episode_id"]),
                    **{field: int(candidate[field]) for field in SEED_FIELDS},
                    "cohort_seed_ledger_path": _relative(SEED_LEDGER_PATH),
                    "cohort_seed_ledger_sha256": sha256_file(SEED_LEDGER_PATH),
                    "authorization_seal": seal.as_json(),
                    "trigger_failure_log_path": _relative(
                        failure_log_path(role, regime)
                    ),
                    "trigger_failure_seq": int(trigger_failure["seq"]),
                    "trigger_failure_record_sha256": str(trigger_hash),
                    "failed_seed_source_episode_id": str(
                        trigger_failure["attempted_seed_source_episode_id"]
                    ),
                }
                claim["record_sha256"] = _failure_hash(claim)
                claim_path = claims_root / f"{claim['claim_index']:06d}.json"
                atomic_json(claim_path, claim, exclusive=True)
                verified = _load_registry(registry_path, claims_root)
                if verified["claims"][-1] != claim:
                    raise EpisodePersistenceError(
                        "global replacement claim did not verify after persistence"
                    )
                return {
                    **candidate,
                    "replacement_claim_index": claim["claim_index"],
                    "replacement_claim_sha256": claim["record_sha256"],
                }
    except ReplacementPoolExhausted:
        raise
    except Exception as error:
        raise EpisodePersistenceError(
            "global replacement claim could not be persisted"
        ) from error
    raise ReplacementPoolExhausted(
        f"replacement pool exhausted for destination {destination['episode_id']}"
    )


def _candidate_from_claim(
    claim: Mapping[str, Any], replacements: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    replacement_id = str(claim["replacement_episode_id"])
    matches = [
        item for item in replacements if str(item["episode_id"]) == replacement_id
    ]
    if len(matches) != 1 or _seed_tuple(matches[0]) != _seed_tuple(claim):
        raise EpisodePersistenceError(
            "replacement registry claim does not match the sealed seed ledger"
        )
    return {
        **dict(matches[0]),
        "replacement_claim_index": int(claim["claim_index"]),
        "replacement_claim_sha256": str(claim["record_sha256"]),
    }


def _failed_source_ids(
    failures: Sequence[Mapping[str, Any]], destination: Mapping[str, Any]
) -> set[str]:
    return {
        str(item["attempted_seed_source_episode_id"])
        for item in failures
        if int(item.get("slot", -1)) == int(destination["slot"])
        and item.get("episode_id") == str(destination["episode_id"])
    }


def _validate_retained_failure_partition(
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    role: str,
    regime: str,
    primary: Sequence[Mapping[str, Any]],
    require_complete: bool,
) -> dict[str, Any]:
    """Prove each destination retains exactly its first failure-free source."""

    primary_by_id = {str(item["episode_id"]): item for item in primary}
    record_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        episode_id = str(record.get("episode_id"))
        if episode_id in record_by_id or episode_id not in primary_by_id:
            raise EpisodePersistenceError("retained episode identity is duplicate or unsealed")
        if type(record.get("replacement_used")) is not bool:
            raise EpisodePersistenceError("retained replacement_used is not exact boolean")
        record_by_id[episode_id] = record

    failures_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for failure in failures:
        episode_id = str(failure.get("episode_id"))
        if episode_id not in primary_by_id:
            raise EpisodePersistenceError("failure references an unsealed destination")
        failures_by_id.setdefault(episode_id, []).append(failure)

    with _registry_lock(REPLACEMENT_REGISTRY_LOCK_PATH):
        claims = [
            claim
            for claim in _load_registry(
                REPLACEMENT_REGISTRY_PATH, REPLACEMENT_CLAIMS_ROOT
            )["claims"]
            if claim.get("role") == role and claim.get("regime") == regime
        ]
    claims_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for claim in claims:
        episode_id = str(claim.get("episode_id"))
        if episode_id not in primary_by_id:
            raise EpisodePersistenceError("replacement claim targets an unsealed destination")
        claims_by_id.setdefault(episode_id, []).append(claim)
    for destination_claims in claims_by_id.values():
        destination_claims.sort(key=lambda item: int(item["claim_index"]))

    if require_complete:
        expected_ids = set(primary_by_id)
        if set(record_by_id) != expected_ids:
            raise EpisodePersistenceError("retained cohort is not the exact destination set")
        if not set(failures_by_id).issubset(expected_ids) or not set(
            claims_by_id
        ).issubset(expected_ids):
            raise EpisodePersistenceError("failure/claim destination closure drift")

    failed_count = 0
    for episode_id, record in record_by_id.items():
        destination = primary_by_id[episode_id]
        destination_failures = failures_by_id.get(episode_id, [])
        destination_claims = claims_by_id.get(episode_id, [])
        retained_source = str(record.get("seed_source_episode_id"))
        failed_sources = [
            str(item.get("attempted_seed_source_episode_id"))
            for item in destination_failures
        ]
        if retained_source in failed_sources:
            raise EpisodePersistenceError(
                "retained source has an authenticated mechanical failure"
            )
        failed_count += len(failed_sources)
        if not destination_failures:
            if destination_claims:
                raise EpisodePersistenceError("replacement claim exists without a failure")
            if not (
                record.get("replacement_used") is False
                and retained_source == episode_id
                and record.get("replacement_claim_index") is None
                and record.get("replacement_claim_sha256") is None
                and _seed_tuple(record) == _seed_tuple(destination)
            ):
                raise EpisodePersistenceError("failure-free destination did not retain primary")
            continue

        if len(destination_claims) != len(destination_failures):
            raise EpisodePersistenceError(
                "each authenticated failure must have exactly one prospective next claim"
            )
        expected_failed_source = episode_id
        for failure, claim in zip(
            destination_failures, destination_claims, strict=True
        ):
            if not (
                failure.get("attempted_seed_source_episode_id")
                == expected_failed_source
                and claim.get("trigger_failure_record_sha256")
                == failure.get("record_sha256")
                and claim.get("trigger_failure_seq") == failure.get("seq")
                and claim.get("failed_seed_source_episode_id")
                == expected_failed_source
                and _destination_matches(
                    claim,
                    role=role,
                    regime=regime,
                    destination=destination,
                )
            ):
                raise EpisodePersistenceError(
                    "failure/replacement claim progression is not exact"
                )
            expected_failed_source = str(claim["replacement_episode_id"])
        terminal_claim = destination_claims[-1]
        if not (
            record.get("replacement_used") is True
            and retained_source == expected_failed_source
            and type(record.get("replacement_claim_index")) is int
            and record.get("replacement_claim_index")
            == terminal_claim.get("claim_index")
            and record.get("replacement_claim_sha256")
            == terminal_claim.get("record_sha256")
            and _seed_tuple(record) == _seed_tuple(terminal_claim)
        ):
            raise EpisodePersistenceError(
                "retained replacement is not the terminal failure-free claim"
            )

    return {
        "retained_count": len(record_by_id),
        "authenticated_failed_source_count": failed_count,
        "retained_sources_failure_free": True,
        "every_failure_has_exact_next_claim": True,
    }


def next_candidate(
    *,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    seal: SealLink,
) -> tuple[dict[str, Any], bool]:
    failed_ids = _failed_source_ids(failures, destination)
    if str(destination["episode_id"]) not in failed_ids:
        return dict(destination), False
    claims = existing_replacement_claims(
        role=role, regime=regime, destination=destination
    )
    destination_failures = [
        item
        for item in failures
        if item.get("episode_id") == str(destination["episode_id"])
        and item.get("slot") == int(destination["slot"])
    ]
    authenticated_failure_hashes = {
        str(item.get("record_sha256")) for item in destination_failures
    }
    for claim in claims:
        candidate = _candidate_from_claim(claim, replacements)
        if (
            str(candidate["episode_id"]) not in failed_ids
            and claim.get("trigger_failure_record_sha256")
            in authenticated_failure_hashes
        ):
            return candidate, True
    if not destination_failures:
        raise EpisodePersistenceError(
            "replacement selection lacks an authenticated destination failure"
        )
    trigger_failure = destination_failures[-1]
    candidate = claim_next_replacement(
        replacements,
        primary=primary,
        role=role,
        regime=regime,
        destination=destination,
        trigger_failure=trigger_failure,
        seal=seal,
        failed_source_ids=failed_ids,
    )
    return candidate, True


def _intent_invariants(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_unix_ns"}


def _exact_positive_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise EpisodePersistenceError(f"{label} must be an exact integer >= {minimum}")
    return int(value)


def _exact_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EpisodePersistenceError(f"{label} must be an exact lowercase SHA-256")
    return value


def _canonical_seed_record_for_persistence(
    value: Mapping[str, Any], *, label: str, expected_episode_id: str
) -> dict[str, Any]:
    if set(value) != set(RECORD_FIELDS):
        raise EpisodePersistenceError(f"{label} seed-record schema is not exact")
    if (
        type(value.get("slot")) is not int
        or not isinstance(value.get("episode_id"), str)
        or any(type(value.get(field)) is not int for field in SEED_FIELDS)
    ):
        raise EpisodePersistenceError(f"{label} seed-record scalar types are not exact")
    try:
        normalized = _validate_seed_record(
            value, label=label, expected_episode_id=expected_episode_id
        )
    except GenerationContractError as error:
        raise EpisodePersistenceError(f"{label} seed record is invalid") from error
    return {key: normalized[key] for key in RECORD_FIELDS}


def _prospective_assignment(
    *,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replacement_used: bool,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str, int, int | None, str | None]:
    if role not in ROLES or regime not in EXPECTED_REGIMES:
        raise EpisodePersistenceError("persistence assignment role/DGP is invalid")
    if type(replacement_used) is not bool:
        raise EpisodePersistenceError("replacement_used must be an exact boolean")
    slot = _exact_positive_int(destination.get("slot"), "destination slot")
    if slot >= len(primary):
        raise EpisodePersistenceError("destination slot is outside the prospective primary pool")
    expected_destination_id = _expected_episode_id(
        regime=regime, role=role, pool_name="primary", index=slot
    )
    destination_record = _canonical_seed_record_for_persistence(
        destination,
        label="destination",
        expected_episode_id=expected_destination_id,
    )
    primary_record = _canonical_seed_record_for_persistence(
        primary[slot],
        label="prospective primary",
        expected_episode_id=expected_destination_id,
    )
    if destination_record != primary_record:
        raise EpisodePersistenceError("destination tuple does not match the prospective ledger")

    claim_keys = {"replacement_claim_index", "replacement_claim_sha256"}
    expected_candidate_keys = set(RECORD_FIELDS) | (claim_keys if replacement_used else set())
    if set(candidate) != expected_candidate_keys:
        raise EpisodePersistenceError("seed-source record schema is not exact")
    if not replacement_used:
        source_record = _canonical_seed_record_for_persistence(
            candidate,
            label="primary seed source",
            expected_episode_id=expected_destination_id,
        )
        if source_record != destination_record:
            raise EpisodePersistenceError("primary seed source differs from destination tuple")
        return destination_record, source_record, "primary", slot, None, None

    source_slot = _exact_positive_int(candidate.get("slot"), "replacement source slot")
    if source_slot >= len(replacements):
        raise EpisodePersistenceError("replacement source slot is outside the prospective pool")
    expected_source_id = _expected_episode_id(
        regime=regime, role=role, pool_name="replacements", index=source_slot
    )
    source_record = _canonical_seed_record_for_persistence(
        {key: candidate[key] for key in RECORD_FIELDS},
        label="replacement seed source",
        expected_episode_id=expected_source_id,
    )
    ledger_source = _canonical_seed_record_for_persistence(
        replacements[source_slot],
        label="prospective replacement",
        expected_episode_id=expected_source_id,
    )
    if source_record != ledger_source:
        raise EpisodePersistenceError("replacement tuple does not match the prospective ledger")
    claim_index = _exact_positive_int(
        candidate.get("replacement_claim_index"), "replacement claim index"
    )
    claim_sha256 = _exact_sha256(
        candidate.get("replacement_claim_sha256"), "replacement claim SHA-256"
    )
    return (
        destination_record,
        source_record,
        "replacements",
        source_slot,
        claim_index,
        claim_sha256,
    )


def _regular_file_snapshot(path: Path, label: str) -> tuple[int, ...]:
    if not os.path.lexists(path):
        raise EpisodePersistenceError(f"{label} is missing: {path}")
    try:
        info = Path(path).lstat()
    except OSError as error:
        raise EpisodePersistenceError(f"could not lstat {label}: {path}") from error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
    ):
        raise EpisodePersistenceError(
            f"{label} is linked, aliased, or non-regular: {path}"
        )
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _assert_repo_file_chain(path: Path, label: str) -> tuple[int, ...]:
    path = Path(path)
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise EpisodePersistenceError(f"{label} escapes the repository: {path}") from error
    current = REPO_ROOT
    for part in relative.parts[:-1]:
        current = current / part
        if not os.path.lexists(current):
            raise EpisodePersistenceError(f"{label} parent is missing: {current}")
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise EpisodePersistenceError(f"{label} parent is linked or non-directory")
    return _regular_file_snapshot(path, label)


def _stable_source_sha256(path: Path, label: str) -> str:
    before = _assert_repo_file_chain(path, label)
    key = (str(Path(path)), *before)
    cached = _PERSISTENCE_SOURCE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    observed = sha256_file(path)
    after = _assert_repo_file_chain(path, label)
    if before != after:
        raise EpisodePersistenceError(f"{label} changed while it was hashed")
    _PERSISTENCE_SOURCE_HASH_CACHE[key] = observed
    return observed


def _stable_artifact_sha256(path: Path, label: str) -> str:
    before = _regular_file_snapshot(path, label)
    observed = sha256_file(path)
    after = _regular_file_snapshot(path, label)
    if before != after:
        raise EpisodePersistenceError(f"{label} changed while it was hashed")
    return observed


def _read_canonical_json_object(path: Path, label: str) -> dict[str, Any]:
    before = _regular_file_snapshot(path, label)
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EpisodePersistenceError(f"{label} is not valid UTF-8 JSON") from error
    after = _regular_file_snapshot(path, label)
    if before != after:
        raise EpisodePersistenceError(f"{label} changed while it was read")
    if not isinstance(value, dict):
        raise EpisodePersistenceError(f"{label} is not a JSON object")
    expected = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if raw != expected:
        raise EpisodePersistenceError(f"{label} is not canonical JSON")
    return value


def _expected_episode_paths(
    *, role: str, regime: str, destination: Mapping[str, Any], paths: EpisodePaths
) -> EpisodePaths:
    expected = episode_paths(role, regime, str(destination.get("episode_id")))
    if paths != expected:
        raise EpisodePersistenceError("episode persistence paths are not canonical")
    return expected


def _validate_current_crosslinks(
    intent: Mapping[str, Any], *, role: str, seal: SealLink
) -> None:
    expected_seal_path = (
        PRE_DATA_SEAL_PATH if role in ("fit", "selection") else PRE_CONFIRMATION_SEAL_PATH
    )
    expected_checkpoint = (
        "PRE_OUTCOME_SEAL"
        if role in ("fit", "selection")
        else "PRE_CONFIRMATION_PACKAGE_SEAL"
    )
    if Path(seal.path) != expected_seal_path:
        raise EpisodePersistenceError("authorization seal path is not current for the role")
    source_paths = (
        (DGP_MATRIX_PATH, "DGP matrix"),
        (SEED_LEDGER_PATH, "cohort seed ledger"),
        (expected_seal_path, "authorization seal"),
        (RESET_REFERENCE_PATH, "reset reference"),
    )
    identities: dict[tuple[int, int], Path] = {}
    for path, label in source_paths:
        snapshot = _assert_repo_file_chain(path, label)
        identity = (snapshot[0], snapshot[1])
        if identity in identities and identities[identity] != path:
            raise EpisodePersistenceError("persistence support files alias one inode")
        identities[identity] = path
    expected_seal = {
        "path": _relative(expected_seal_path),
        "sha256": _stable_source_sha256(expected_seal_path, "authorization seal"),
        "checkpoint_state": expected_checkpoint,
    }
    if (
        seal.relative_path != expected_seal["path"]
        or seal.sha256 != expected_seal["sha256"]
        or seal.checkpoint_state != expected_checkpoint
        or intent.get("authorization_seal") != expected_seal
    ):
        raise EpisodePersistenceError("persistence intent authorization crosslink drift")
    source_checks = {
        "dgp_path": intent.get("dgp_matrix_path") == _relative(DGP_MATRIX_PATH),
        "dgp_sha256": intent.get("dgp_matrix_sha256")
        == _stable_source_sha256(DGP_MATRIX_PATH, "DGP matrix"),
        "ledger_path": intent.get("cohort_seed_ledger_path")
        == _relative(SEED_LEDGER_PATH),
        "ledger_sha256": intent.get("cohort_seed_ledger_sha256")
        == _stable_source_sha256(SEED_LEDGER_PATH, "cohort seed ledger"),
    }
    if not all(source_checks.values()):
        raise EpisodePersistenceError(
            f"persistence intent DGP/ledger crosslink drift: {source_checks}"
        )


def _expected_array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validation = validate_model_gate_inputs(arrays)
    return {
        key: {
            "shape": list(np.asarray(arrays[key]).shape),
            "dtype": str(np.asarray(arrays[key]).dtype),
            "sha256": validation["array_sha256"][key],
        }
        for key in sorted(INPUT_ALLOWLIST)
    }


def _validate_generation_audit(
    value: Any,
    *,
    arrays: Mapping[str, np.ndarray],
    source_record: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != GENERATION_AUDIT_KEYS:
        raise EpisodePersistenceError("generation audit schema is not exact")
    audit = _canonical_json(value)
    validation = validate_model_gate_inputs(arrays)
    initial_sha256 = array_sha256(np.asarray(arrays["pixels"])[0])
    exact_counts = {
        "rows": RAW_ROWS,
        "rollout_steps_expected": ROLLOUT_STEPS,
        "rollout_steps_completed": ROLLOUT_STEPS,
        "pixel_frames_captured": RAW_ROWS,
        "action_space_seed": int(source_record["action_space_seed"]),
    }
    for key, expected in exact_counts.items():
        if type(audit.get(key)) is not int or audit.get(key) != expected:
            raise EpisodePersistenceError(f"generation audit {key} drift")
    if (
        audit.get("initial_pixels_sha256") != initial_sha256
        or audit.get("input_validation") != validation
        or audit.get("rng_activation_order") != RNG_ACTIVATION_ORDER
        or audit.get("only_pixels_and_actions_captured") is not True
        or audit.get(
            "retention_uses_only_input_contract_and_local_step_bookkeeping"
        )
        is not True
    ):
        raise EpisodePersistenceError("generation audit content drift")
    seeded = audit.get("action_spaces_seeded")
    allowed_seeded = ["vector_action_space", "unwrapped_action_space"]
    if (
        not isinstance(seeded, list)
        or not seeded
        or len(seeded) != len(set(seeded))
        or any(name not in allowed_seeded for name in seeded)
        or seeded != [name for name in allowed_seeded if name in seeded]
    ):
        raise EpisodePersistenceError("generation audit action-space seed list drift")
    reset = audit.get("reset_metadata")
    if not isinstance(reset, Mapping) or set(reset) != RESET_METADATA_KEYS:
        raise EpisodePersistenceError("generation reset metadata schema is not exact")
    for key in ("environment_seed", "variation_seed", "physical_state_seed"):
        if type(reset.get(key)) is not int or reset.get(key) != int(source_record["env_seed"]):
            raise EpisodePersistenceError(f"generation reset metadata {key} drift")
    _exact_sha256(reset.get("variation_values_sha256"), "variation values SHA-256")
    if (
        reset.get("reset_reference_path") != _relative(RESET_REFERENCE_PATH)
        or reset.get("reset_reference_sha256")
        != _stable_source_sha256(RESET_REFERENCE_PATH, "reset reference")
        or reset.get("reset_values_persisted") is not False
    ):
        raise EpisodePersistenceError("generation reset reference contract drift")
    return dict(audit)


def _validate_persistence_intent(
    intent: Mapping[str, Any],
    *,
    paths: EpisodePaths,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    seal: SealLink,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if set(intent) != PERSISTENCE_INTENT_KEYS:
        raise EpisodePersistenceError("persistence intent schema is not exact")
    _expected_episode_paths(
        role=role, regime=regime, destination=destination, paths=paths
    )
    _exact_positive_int(intent.get("created_unix_ns"), "persistence intent timestamp", minimum=1)
    if not (
        type(intent.get("schema_version")) is int
        and intent.get("schema_version") == PERSISTENCE_INTENT_SCHEMA_VERSION
        and intent.get("attempt") == ATTEMPT
        and intent.get("science_attempt") == SCIENCE_ATTEMPT
        and intent.get("status") == PERSISTENCE_INTENT_STATUS
        and intent.get("role") == role
        and intent.get("regime") == regime
        and intent.get("orphan_adoption_rule") == ORPHAN_ADOPTION_RULE
    ):
        raise EpisodePersistenceError("persistence intent header drift")
    if not isinstance(intent.get("destination_seed_record"), Mapping) or not isinstance(
        intent.get("seed_source_record"), Mapping
    ):
        raise EpisodePersistenceError("persistence intent seed records are not objects")
    candidate: dict[str, Any] = dict(intent["seed_source_record"])
    replacement_used = intent.get("replacement_used")
    if type(replacement_used) is not bool:
        raise EpisodePersistenceError("persistence intent replacement flag is not exact")
    if replacement_used:
        candidate.update(
            {
                "replacement_claim_index": intent.get("replacement_claim_index"),
                "replacement_claim_sha256": intent.get("replacement_claim_sha256"),
            }
        )
    destination_record, source_record, source_pool, source_slot, claim_index, claim_sha256 = (
        _prospective_assignment(
            role=role,
            regime=regime,
            destination=intent["destination_seed_record"],
            candidate=candidate,
            replacement_used=replacement_used,
            primary=primary,
            replacements=replacements,
        )
    )
    supplied_destination = _canonical_seed_record_for_persistence(
        destination,
        label="supplied destination",
        expected_episode_id=str(destination_record["episode_id"]),
    )
    identity_checks = {
        "destination": supplied_destination == destination_record,
        "slot": type(intent.get("slot")) is int
        and intent.get("slot") == destination_record["slot"],
        "episode_id": intent.get("episode_id") == destination_record["episode_id"],
        "source_pool": intent.get("seed_source_pool") == source_pool,
        "source_slot": type(intent.get("seed_source_slot")) is int
        and intent.get("seed_source_slot") == source_slot,
        "source_record": intent.get("seed_source_record") == source_record,
        "source_id": intent.get("seed_source_episode_id") == source_record["episode_id"],
        "claim_index": intent.get("replacement_claim_index") == claim_index,
        "claim_sha256": intent.get("replacement_claim_sha256") == claim_sha256,
        "source_seeds": all(intent.get(field) == source_record[field] for field in SEED_FIELDS),
        "raw_path": intent.get("raw_path") == _relative(paths.raw),
        "sidecar_path": intent.get("raw_sidecar_path") == _relative(paths.sidecar),
        "intent_path": intent.get("persistence_intent_path") == _relative(paths.intent),
    }
    if not all(identity_checks.values()):
        raise EpisodePersistenceError(
            f"persistence intent assignment/path drift: {identity_checks}"
        )
    expected_metadata = _expected_array_metadata(arrays)
    if intent.get("arrays") != expected_metadata:
        raise EpisodePersistenceError("persistence intent array metadata/hash drift")
    audit = _validate_generation_audit(
        intent.get("generation_audit"), arrays=arrays, source_record=source_record
    )
    if intent.get("initial_pixels_sha256") != audit["initial_pixels_sha256"]:
        raise EpisodePersistenceError("persistence intent initial-frame hash drift")
    _validate_current_crosslinks(intent, role=role, seal=seal)
    return dict(intent)


def build_persistence_intent(
    *,
    paths: EpisodePaths,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replacement_used: bool,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    generation_audit: Mapping[str, Any],
    seal: SealLink,
) -> dict[str, Any]:
    destination_record, source_record, source_pool, source_slot, claim_index, claim_sha256 = (
        _prospective_assignment(
            role=role,
            regime=regime,
            destination=destination,
            candidate=candidate,
            replacement_used=replacement_used,
            primary=primary,
            replacements=replacements,
        )
    )
    _expected_episode_paths(
        role=role, regime=regime, destination=destination_record, paths=paths
    )
    audit = _validate_generation_audit(
        generation_audit, arrays=arrays, source_record=source_record
    )
    intended = {
        "schema_version": PERSISTENCE_INTENT_SCHEMA_VERSION,
        "attempt": ATTEMPT,
        "science_attempt": SCIENCE_ATTEMPT,
        "created_unix_ns": time.time_ns(),
        "status": PERSISTENCE_INTENT_STATUS,
        "role": role,
        "regime": regime,
        "slot": destination_record["slot"],
        "episode_id": destination_record["episode_id"],
        "raw_path": _relative(paths.raw),
        "raw_sidecar_path": _relative(paths.sidecar),
        "persistence_intent_path": _relative(paths.intent),
        "dgp_matrix_path": _relative(DGP_MATRIX_PATH),
        "dgp_matrix_sha256": _stable_source_sha256(DGP_MATRIX_PATH, "DGP matrix"),
        "cohort_seed_ledger_path": _relative(SEED_LEDGER_PATH),
        "cohort_seed_ledger_sha256": _stable_source_sha256(
            SEED_LEDGER_PATH, "cohort seed ledger"
        ),
        "destination_seed_record": destination_record,
        "seed_source_pool": source_pool,
        "seed_source_slot": source_slot,
        "seed_source_record": source_record,
        "seed_source_episode_id": source_record["episode_id"],
        "replacement_used": replacement_used,
        "replacement_claim_index": claim_index,
        "replacement_claim_sha256": claim_sha256,
        **{field: source_record[field] for field in SEED_FIELDS},
        "arrays": _expected_array_metadata(arrays),
        "initial_pixels_sha256": audit["initial_pixels_sha256"],
        "generation_audit": audit,
        "authorization_seal": seal.as_json(),
        "orphan_adoption_rule": ORPHAN_ADOPTION_RULE,
    }
    return _validate_persistence_intent(
        intended,
        paths=paths,
        role=role,
        regime=regime,
        destination=destination_record,
        primary=primary,
        replacements=replacements,
        seal=seal,
        arrays=arrays,
    )


def _write_or_verify_intent(path: Path, intended: Mapping[str, Any]) -> dict[str, Any]:
    try:
        if set(intended) != PERSISTENCE_INTENT_KEYS:
            raise EpisodePersistenceError("intended persistence intent schema is not exact")
        if os.path.lexists(path):
            existing = _read_canonical_json_object(path, "persistence intent")
            if _intent_invariants(existing) != _intent_invariants(intended):
                raise EpisodePersistenceError(
                    f"persistence intent drift for {intended['episode_id']}"
                )
            _exact_positive_int(
                existing.get("created_unix_ns"), "persistence intent timestamp", minimum=1
            )
            return existing
        try:
            atomic_json(path, intended, exclusive=True)
            written = _read_canonical_json_object(path, "persistence intent")
            if written != dict(intended):
                raise EpisodePersistenceError("new persistence intent byte/object drift")
            return written
        except Exception:
            if os.path.lexists(path):
                raced = _read_canonical_json_object(path, "persistence intent")
                if _intent_invariants(raced) == _intent_invariants(intended):
                    return raced
            raise
    except EpisodePersistenceError:
        raise
    except Exception as error:
        raise EpisodePersistenceError(
            f"could not persist episode intent: {path}"
        ) from error


def _verify_raw_against_intent(
    raw_path: Path, intent: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    before = _regular_file_snapshot(raw_path, "raw episode archive")
    try:
        arrays, audit = load_model_gate_inputs(raw_path)
    except Exception as error:
        raise EpisodePersistenceError(
            f"raw orphan or episode violates the input contract: {raw_path}"
        ) from error
    after = _regular_file_snapshot(raw_path, "raw episode archive")
    if before != after:
        raise EpisodePersistenceError("raw episode archive changed during recomputation")
    if intent.get("arrays") != _expected_array_metadata(arrays):
        raise EpisodePersistenceError(
            f"raw content does not match persistence intent: {raw_path}"
        )
    if intent.get("initial_pixels_sha256") != array_sha256(arrays["pixels"][0]):
        raise EpisodePersistenceError("raw initial frame does not match persistence intent")
    return arrays, audit


def _record_from_intent(
    paths: EpisodePaths, intent: Mapping[str, Any], raw_audit: Mapping[str, Any]
) -> dict[str, Any]:
    if type(intent.get("replacement_used")) is not bool:
        raise EpisodePersistenceError(
            "persistence intent replacement_used must be exact boolean"
        )
    return {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "created_unix_ns": time.time_ns(),
        "complete": True,
        "role": intent["role"],
        "regime": intent["regime"],
        "slot": int(intent["slot"]),
        "episode_id": intent["episode_id"],
        "seed_source_episode_id": intent["seed_source_episode_id"],
        "replacement_used": intent["replacement_used"],
        "replacement_claim_index": intent.get("replacement_claim_index"),
        "replacement_claim_sha256": intent.get("replacement_claim_sha256"),
        **{field: int(intent[field]) for field in SEED_FIELDS},
        "raw_path": _relative(paths.raw),
        "raw_sha256": sha256_file(paths.raw),
        "raw_bytes": paths.raw.stat().st_size,
        "arrays": intent["arrays"],
        "initial_pixels_sha256": intent["initial_pixels_sha256"],
        "generation_audit": intent["generation_audit"],
        "input_loader_audit": _canonical_json(raw_audit),
        "persistence_intent_path": _relative(paths.intent),
        "persistence_intent_sha256": sha256_file(paths.intent),
        "authorization_seal": intent["authorization_seal"],
        "raw_archive_members": sorted(INPUT_ALLOWLIST),
        "orphan_adoption_safe": True,
    }


def _write_or_verify_sidecar(
    paths: EpisodePaths, record: Mapping[str, Any]
) -> dict[str, Any]:
    invariants = {
        key: value for key, value in record.items() if key != "created_unix_ns"
    }
    try:
        if os.path.lexists(paths.sidecar):
            existing = _read_canonical_json_object(paths.sidecar, "raw sidecar")
            existing_invariants = {
                key: value
                for key, value in existing.items()
                if key != "created_unix_ns"
            }
            if existing_invariants != invariants:
                raise EpisodePersistenceError(
                    f"existing raw sidecar drift: {paths.sidecar}"
                )
            return existing
        try:
            atomic_json(paths.sidecar, record, exclusive=True)
            written = _read_canonical_json_object(paths.sidecar, "raw sidecar")
            if written != dict(record):
                raise EpisodePersistenceError("new raw sidecar byte/object drift")
            return written
        except Exception:
            if os.path.lexists(paths.sidecar):
                raced = _read_canonical_json_object(paths.sidecar, "raw sidecar")
                raced_invariants = {
                    key: value
                    for key, value in raced.items()
                    if key != "created_unix_ns"
                }
                if raced_invariants == invariants:
                    return raced
            raise
    except EpisodePersistenceError:
        raise
    except Exception as error:
        raise EpisodePersistenceError(
            f"could not persist raw sidecar: {paths.sidecar}"
        ) from error


def adopt_orphan(
    *,
    paths: EpisodePaths,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    seal: SealLink,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Adopt a raw-only artifact iff its prior durable intent proves completeness."""
    _expected_episode_paths(
        role=role, regime=regime, destination=destination, paths=paths
    )
    if not os.path.lexists(paths.raw) or os.path.lexists(paths.sidecar):
        raise EpisodePersistenceError("orphan adoption requires raw-only final paths")
    if not os.path.lexists(paths.intent):
        raise EpisodePersistenceError(
            f"unsafe raw orphan has no durable content intent: {paths.raw}"
        )
    directories: dict[tuple[int, int], Path] = {}
    files: dict[tuple[int, int], Path] = {}
    try:
        for path in (paths.raw, paths.intent, paths.sidecar):
            _assert_safe_output_leaf(
                path, seen_directories=directories, seen_files=files
            )
    except GenerationPreflightError as error:
        raise EpisodePersistenceError("raw orphan output namespace is unsafe") from error
    intent = _read_canonical_json_object(paths.intent, "persistence intent")
    intent_sha256 = _stable_artifact_sha256(paths.intent, "persistence intent")
    arrays, audit = _verify_raw_against_intent(paths.raw, intent)
    _validate_persistence_intent(
        intent,
        paths=paths,
        role=role,
        regime=regime,
        destination=destination,
        primary=primary,
        replacements=replacements,
        seal=seal,
        arrays=arrays,
    )
    _validate_retained_failure_partition(
        [intent],
        failures,
        role=role,
        regime=regime,
        primary=primary,
        require_complete=False,
    )
    record = _record_from_intent(paths, intent, audit)
    if (
        record["raw_sha256"] != audit.get("archive_sha256")
        or record["raw_sha256"]
        != _stable_artifact_sha256(paths.raw, "raw episode archive")
        or record["persistence_intent_sha256"] != intent_sha256
        or intent_sha256
        != _stable_artifact_sha256(paths.intent, "persistence intent")
    ):
        raise EpisodePersistenceError("orphan changed before sidecar construction")
    return _write_or_verify_sidecar(paths, record)


def persist_episode(
    *,
    paths: EpisodePaths,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replacement_used: bool,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    generation_audit: Mapping[str, Any],
    seal: SealLink,
) -> dict[str, Any]:
    """Persist a valid episode without ever converting I/O failure to replacement."""
    intent = build_persistence_intent(
        paths=paths,
        role=role,
        regime=regime,
        destination=destination,
        candidate=candidate,
        replacement_used=replacement_used,
        primary=primary,
        replacements=replacements,
        arrays=arrays,
        generation_audit=generation_audit,
        seal=seal,
    )
    intent = _write_or_verify_intent(paths.intent, intent)
    try:
        if not os.path.lexists(paths.raw):
            atomic_npz(
                paths.raw,
                {key: arrays[key] for key in sorted(INPUT_ALLOWLIST)},
                exclusive=True,
            )
    except Exception as error:
        if os.path.lexists(paths.raw):
            # A concurrent identical writer may have won the exclusive create.
            # Exact intent/content verification is the only adoption path.
            _verify_raw_against_intent(paths.raw, intent)
        else:
            raise EpisodePersistenceError(
                f"could not persist raw episode; replacement is forbidden: {paths.raw}"
            ) from error
    materialized, raw_audit = _verify_raw_against_intent(paths.raw, intent)
    _validate_persistence_intent(
        intent,
        paths=paths,
        role=role,
        regime=regime,
        destination=destination,
        primary=primary,
        replacements=replacements,
        seal=seal,
        arrays=materialized,
    )
    record = _record_from_intent(paths, intent, raw_audit)
    return _write_or_verify_sidecar(paths, record)


def verify_episode_record(
    record: Mapping[str, Any],
    *,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    seal: SealLink,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paths = episode_paths(role, regime, str(destination["episode_id"]))
    directories: dict[tuple[int, int], Path] = {}
    files: dict[tuple[int, int], Path] = {}
    for path in (paths.raw, paths.sidecar, paths.intent):
        _assert_safe_output_leaf(
            path, seen_directories=directories, seen_files=files
        )
        try:
            info = path.lstat()
        except FileNotFoundError as error:
            raise EpisodePersistenceError(
                f"partial episode artifact: {paths.raw}"
            ) from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise EpisodePersistenceError(
                f"episode artifact is linked or non-regular: {path}"
            )
    sidecar_bytes = paths.sidecar.read_bytes()
    expected_sidecar_bytes = (
        json.dumps(dict(record), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        sidecar_object = json.loads(sidecar_bytes)
    except json.JSONDecodeError as error:
        raise EpisodePersistenceError(
            f"raw sidecar is invalid JSON: {paths.sidecar}"
        ) from error
    if sidecar_object != dict(record) or sidecar_bytes != expected_sidecar_bytes:
        raise EpisodePersistenceError(
            f"raw sidecar byte/object mismatch: {paths.sidecar}"
        )
    checks = {
        "attempt": record.get("attempt") == ATTEMPT,
        "complete": record.get("complete") is True,
        "role": record.get("role") == role,
        "regime": record.get("regime") == regime,
        "slot": int(record.get("slot", -1)) == int(destination["slot"]),
        "episode_id": record.get("episode_id") == str(destination["episode_id"]),
        "raw_path": record.get("raw_path") == _relative(paths.raw),
        "raw_sha256": record.get("raw_sha256") == sha256_file(paths.raw),
        "intent_path": record.get("persistence_intent_path")
        == _relative(paths.intent),
        "intent_sha256": record.get("persistence_intent_sha256")
        == sha256_file(paths.intent),
        "seal": record.get("authorization_seal") == seal.as_json(),
    }
    if not all(checks.values()):
        raise EpisodePersistenceError(
            f"existing episode sidecar failed verification: {checks}"
        )
    if record.get("replacement_used") is True:
        matches = [
            candidate
            for candidate in replacements
            if candidate.get("episode_id") == record.get("seed_source_episode_id")
            and _seed_tuple(candidate) == _seed_tuple(record)
        ]
        claim_index = record.get("replacement_claim_index")
        if len(matches) != 1 or not isinstance(claim_index, int):
            raise EpisodePersistenceError(
                "replacement sidecar does not match the sealed replacement pool"
            )
        with _registry_lock():
            registry = _load_registry()
            if claim_index < 0 or claim_index >= len(registry["claims"]):
                raise EpisodePersistenceError(
                    "replacement sidecar claim index is outside the global registry"
                )
            claim = registry["claims"][claim_index]
        claim_checks = {
            "destination": _destination_matches(
                claim, role=role, regime=regime, destination=destination
            ),
            "source_id": claim.get("replacement_episode_id")
            == record.get("seed_source_episode_id"),
            "seeds": _seed_tuple(claim) == _seed_tuple(record),
            "claim_sha256": claim.get("record_sha256")
            == record.get("replacement_claim_sha256"),
        }
        if not all(claim_checks.values()):
            raise EpisodePersistenceError(
                f"replacement sidecar/global-registry drift: {claim_checks}"
            )
    else:
        if (
            record.get("replacement_used") is not False
            or record.get("replacement_claim_index") is not None
            or record.get("replacement_claim_sha256") is not None
            or record.get("seed_source_episode_id")
            != str(destination["episode_id"])
            or _seed_tuple(record) != _seed_tuple(destination)
        ):
            raise EpisodePersistenceError("primary sidecar seed assignment drift")
    intent = _read_canonical_json_object(paths.intent, "persistence intent")
    arrays, raw_audit = _verify_raw_against_intent(paths.raw, intent)
    _validate_persistence_intent(
        intent,
        paths=paths,
        role=role,
        regime=regime,
        destination=destination,
        primary=primary,
        replacements=replacements,
        seal=seal,
        arrays=arrays,
    )
    expected = _record_from_intent(paths, intent, raw_audit)
    invariant_record = {
        key: value for key, value in record.items() if key != "created_unix_ns"
    }
    invariant_expected = {
        key: value for key, value in expected.items() if key != "created_unix_ns"
    }
    if invariant_record != invariant_expected:
        raise EpisodePersistenceError(f"episode record/content drift: {paths.raw}")
    return dict(record)


def resume_episode(
    *,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    seal: SealLink,
    primary: Sequence[Mapping[str, Any]],
    replacements: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    paths = episode_paths(role, regime, str(destination["episode_id"]))
    raw_exists = os.path.lexists(paths.raw)
    sidecar_exists = os.path.lexists(paths.sidecar)
    intent_exists = os.path.lexists(paths.intent)
    if not raw_exists and not sidecar_exists:
        if intent_exists:
            raise EpisodePersistenceError(
                f"intent-only partial artifact cannot be regenerated: {paths.intent}"
            )
        return None
    if raw_exists and not sidecar_exists:
        record = adopt_orphan(
            paths=paths,
            role=role,
            regime=regime,
            destination=destination,
            seal=seal,
            primary=primary,
            replacements=replacements,
            failures=failures,
        )
        verified = verify_episode_record(
            record,
            role=role,
            regime=regime,
            destination=destination,
            seal=seal,
            primary=primary,
            replacements=replacements,
        )
        _validate_retained_failure_partition(
            [verified],
            failures,
            role=role,
            regime=regime,
            primary=primary,
            require_complete=False,
        )
        return verified
    if sidecar_exists and not raw_exists:
        raise EpisodePersistenceError(
            f"sidecar-only partial artifact cannot be replaced: {paths.sidecar}"
        )
    if not intent_exists:
        raise EpisodePersistenceError(
            f"complete paths lack the durable content intent: {paths.raw}"
        )
    record = read_json(paths.sidecar)
    _validate_retained_failure_partition(
        [record],
        failures,
        role=role,
        regime=regime,
        primary=primary,
        require_complete=False,
    )
    return verify_episode_record(
        record,
        role=role,
        regime=regime,
        destination=destination,
        seal=seal,
        primary=primary,
        replacements=replacements,
    )


def validate_raw_manifest_contract(
    role: str,
    regime: str,
    inputs: GenerationInputs,
    path: Path | None = None,
) -> dict[str, Any] | None:
    expected_path = raw_manifest_path(role, regime)
    path = expected_path if path is None else Path(path)
    if path != expected_path:
        raise EpisodePersistenceError(
            f"raw manifest is not at its canonical role/regime path: {path}"
        )
    if not path.exists():
        return None
    directories: dict[tuple[int, int], Path] = {}
    files: dict[tuple[int, int], Path] = {}
    _assert_safe_output_leaf(
        path, seen_directories=directories, seen_files=files
    )
    payload = read_json(path)
    expected_fields = {
        "schema_version",
        "attempt",
        "created_unix_ns",
        "complete",
        "role",
        "regime",
        "regime_specification",
        "episode_count",
        "episodes",
        "replacements_used",
        "raw_archive_members",
        "raw_pixels_contract",
        "raw_action_contract",
        "role_isolation",
        "smoke_permanently_excluded",
        "retention_contract",
        "dgp_matrix_path",
        "dgp_matrix_sha256",
        "cohort_seed_ledger_path",
        "cohort_seed_ledger_sha256",
        "authorization_seal",
        "replacement_registry_path",
        "replacement_registry_scope",
        "orphan_policy",
    }
    checks = {
        "fields": set(payload) == expected_fields,
        "schema": payload.get("schema_version") == 1,
        "attempt": payload.get("attempt") == ATTEMPT,
        "timestamp": type(payload.get("created_unix_ns")) is int
        and int(payload["created_unix_ns"]) > 0,
        "complete": payload.get("complete") is True,
        "role": payload.get("role") == role,
        "regime": payload.get("regime") == regime,
        "episode_count": payload.get("episode_count") == len(inputs.primary),
        "seal": payload.get("authorization_seal") == inputs.seal.as_json(),
        "dgp_sha256": payload.get("dgp_matrix_sha256")
        == sha256_file(DGP_MATRIX_PATH),
        "dgp_path": payload.get("dgp_matrix_path") == _relative(DGP_MATRIX_PATH),
        "ledger_sha256": payload.get("cohort_seed_ledger_sha256")
        == sha256_file(SEED_LEDGER_PATH),
        "ledger_path": payload.get("cohort_seed_ledger_path")
        == _relative(SEED_LEDGER_PATH),
        "regime_specification": payload.get("regime_specification")
        == inputs.regime_specification,
        "raw_archive_members": payload.get("raw_archive_members")
        == sorted(INPUT_ALLOWLIST),
        "raw_pixels_contract": payload.get("raw_pixels_contract")
        == "uint8 [201,224,224,3]",
        "raw_action_contract": payload.get("raw_action_contract")
        == "float32 [201,5]; 200 finite plus terminal NaN",
        "role_isolation": payload.get("role_isolation") is True,
        "smoke_exclusion": payload.get("smoke_permanently_excluded")
        is (role == "smoke"),
        "retention_contract": payload.get("retention_contract")
        == "pixels_action_shapes_finiteness_and_local_step_count_only",
        "replacement_registry": payload.get("replacement_registry_path")
        == _relative(REPLACEMENT_REGISTRY_PATH),
        "replacement_registry_exists": REPLACEMENT_REGISTRY_PATH.is_file(),
        "replacement_registry_scope": payload.get("replacement_registry_scope")
        == "append_only_all_roles_and_all_dgps",
        "orphan_policy": payload.get("orphan_policy")
        == RAW_MANIFEST_ORPHAN_POLICY,
    }
    episodes = payload.get("episodes")
    if not all(checks.values()) or not isinstance(episodes, list):
        raise EpisodePersistenceError(f"existing raw manifest drift: {checks}")
    if len(episodes) != len(inputs.primary):
        raise EpisodePersistenceError("raw manifest episode list count drift")
    verified = [
        verify_episode_record(
            record,
            role=role,
            regime=regime,
            destination=destination,
            seal=inputs.seal,
            primary=inputs.primary,
            replacements=inputs.replacements,
        )
        for record, destination in zip(episodes, inputs.primary, strict=True)
    ]
    if verified != episodes:
        raise EpisodePersistenceError("raw manifest record normalization drift")
    if payload.get("replacements_used") != sum(
        bool(item["replacement_used"]) for item in episodes
    ):
        raise EpisodePersistenceError("raw manifest replacement count drift")
    failures = read_rollout_failures(
        role,
        regime,
        primary=inputs.primary,
        replacements=inputs.replacements,
        seal=inputs.seal,
    )
    _validate_retained_failure_partition(
        verified,
        failures,
        role=role,
        regime=regime,
        primary=inputs.primary,
        require_complete=True,
    )
    return payload


def validate_existing_manifest(
    role: str,
    regime: str,
    inputs: GenerationInputs,
) -> dict[str, Any] | None:
    """Compatibility name for the single strict raw-manifest validator."""

    return validate_raw_manifest_contract(role, regime, inputs)


def _close_world(world: Any) -> None:
    if world is None:
        return
    try:
        world.close()
    except Exception:
        pass


def _generate_locked(
    role: str, regime: str, inputs: GenerationInputs
) -> dict[str, Any]:
    """Generate/resume only while the role-by-DGP writer lock is held."""

    existing_manifest = validate_existing_manifest(role, regime, inputs)
    if existing_manifest is not None:
        return existing_manifest
    verify_controller_permission(role)
    failures = read_rollout_failures(
        role,
        regime,
        primary=inputs.primary,
        replacements=inputs.replacements,
        seal=inputs.seal,
    )
    ensure_replacement_registry()

    raw_directory(role, regime).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    world = policy = None
    try:
        for index, destination in enumerate(inputs.primary):
            resumed = resume_episode(
                role=role,
                regime=regime,
                destination=destination,
                seal=inputs.seal,
                primary=inputs.primary,
                replacements=inputs.replacements,
                failures=failures,
            )
            if resumed is not None:
                records.append(resumed)
                continue
            paths = episode_paths(role, regime, str(destination["episode_id"]))
            while True:
                candidate, replacement_used = next_candidate(
                    role=role,
                    regime=regime,
                    destination=destination,
                    primary=inputs.primary,
                    replacements=inputs.replacements,
                    failures=failures,
                    seal=inputs.seal,
                )
                if world is None:
                    try:
                        world, policy = make_world(inputs.regime_specification)
                    except Exception as error:
                        raise GenerationPreflightError(
                            f"world construction failed before episode use: {error}"
                        ) from error
                try:
                    arrays, generation_audit = generate_episode(
                        world,
                        policy,
                        role=role,
                        regime=regime,
                        slot=int(destination["slot"]),
                        episode_id=str(destination["episode_id"]),
                        env_seed=int(candidate["env_seed"]),
                        policy_seed=int(candidate["policy_seed"]),
                        oracle_np_seed=int(candidate["oracle_np_seed"]),
                        action_space_seed=int(candidate["action_space_seed"]),
                    )
                except EpisodeRolloutError as error:
                    failure = record_rollout_failure(
                        role=role,
                        regime=regime,
                        destination=destination,
                        candidate=candidate,
                        replacement_used=replacement_used,
                        error=error,
                        primary=inputs.primary,
                        replacements=inputs.replacements,
                        seal=inputs.seal,
                    )
                    failures.append(failure)
                    _close_world(world)
                    world = policy = None
                    continue

                # Persistence is intentionally outside the rollout exception
                # handler.  No error below this line may advance a seed tuple.
                _validate_retained_failure_partition(
                    [
                        {
                            "episode_id": str(destination["episode_id"]),
                            "seed_source_episode_id": str(candidate["episode_id"]),
                            "replacement_used": replacement_used,
                            "replacement_claim_index": candidate.get(
                                "replacement_claim_index"
                            ),
                            "replacement_claim_sha256": candidate.get(
                                "replacement_claim_sha256"
                            ),
                            **{field: int(candidate[field]) for field in SEED_FIELDS},
                        }
                    ],
                    failures,
                    role=role,
                    regime=regime,
                    primary=inputs.primary,
                    require_complete=False,
                )
                record = persist_episode(
                    paths=paths,
                    role=role,
                    regime=regime,
                    destination=destination,
                    candidate=candidate,
                    replacement_used=replacement_used,
                    primary=inputs.primary,
                    replacements=inputs.replacements,
                    arrays=arrays,
                    generation_audit=generation_audit,
                    seal=inputs.seal,
                )
                records.append(record)
                print(
                    f"generated {role} {regime} {index + 1}/{len(inputs.primary)}",
                    flush=True,
                )
                break
    finally:
        _close_world(world)

    if len(records) != len(inputs.primary):
        raise GenerationContractError("raw cohort did not reach its exact sealed count")
    if [item["episode_id"] for item in records] != [
        item["episode_id"] for item in inputs.primary
    ]:
        raise GenerationContractError("raw cohort order or identity drift")
    _validate_retained_failure_partition(
        records,
        failures,
        role=role,
        regime=regime,
        primary=inputs.primary,
        require_complete=True,
    )
    manifest = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "created_unix_ns": time.time_ns(),
        "complete": True,
        "role": role,
        "regime": regime,
        "regime_specification": inputs.regime_specification,
        "episode_count": len(records),
        "episodes": records,
        "replacements_used": sum(bool(item["replacement_used"]) for item in records),
        "raw_archive_members": sorted(INPUT_ALLOWLIST),
        "raw_pixels_contract": "uint8 [201,224,224,3]",
        "raw_action_contract": "float32 [201,5]; 200 finite plus terminal NaN",
        "role_isolation": True,
        "smoke_permanently_excluded": role == "smoke",
        "retention_contract": "pixels_action_shapes_finiteness_and_local_step_count_only",
        "dgp_matrix_path": _relative(DGP_MATRIX_PATH),
        "dgp_matrix_sha256": sha256_file(DGP_MATRIX_PATH),
        "cohort_seed_ledger_path": _relative(SEED_LEDGER_PATH),
        "cohort_seed_ledger_sha256": sha256_file(SEED_LEDGER_PATH),
        "authorization_seal": inputs.seal.as_json(),
        "replacement_registry_path": _relative(REPLACEMENT_REGISTRY_PATH),
        "replacement_registry_scope": "append_only_all_roles_and_all_dgps",
        "orphan_policy": RAW_MANIFEST_ORPHAN_POLICY,
    }
    path = raw_manifest_path(role, regime)
    try:
        atomic_json(path, manifest, exclusive=True)
    except Exception as error:
        if path.exists():
            raced = validate_existing_manifest(role, regime, inputs)
            if raced is not None:
                return raced
        raise EpisodePersistenceError(
            f"could not persist raw role manifest: {path}"
        ) from error
    return manifest


def generate(role: str, regime: str) -> dict[str, Any]:
    """Generate or resume one sealed role-by-DGP raw cohort."""

    runtime = verify_here("generation", include_external_hashes=True)
    if runtime.get("passed") is not True:
        raise GenerationPreflightError("generation runtime preflight did not pass")
    inputs = load_generation_inputs(role, regime)
    preflight_output_namespace(role, regime, inputs.primary)
    with materialization_lock(role, regime):
        return _generate_locked(role, regime, inputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=ROLES)
    parser.add_argument("regime", choices=EXPECTED_REGIMES)
    arguments = parser.parse_args()
    result = generate(arguments.role, arguments.regime)
    print(
        json.dumps(
            {
                "attempt": result["attempt"],
                "role": result["role"],
                "regime": result["regime"],
                "episode_count": result["episode_count"],
                "replacements_used": result["replacements_used"],
                "manifest_path": _relative(raw_manifest_path(arguments.role, arguments.regime)),
                "manifest_sha256": sha256_file(
                    raw_manifest_path(arguments.role, arguments.regime)
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
