from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pytest


ATTEMPT_ROOT = Path(__file__).resolve().parents[1]
if str(ATTEMPT_ROOT) not in sys.path:
    sys.path.insert(0, str(ATTEMPT_ROOT))

import generator as generation  # noqa: E402
import input_loader  # noqa: E402
import checkpoints  # noqa: E402
import independent_verify as independent  # noqa: E402
import launcher  # noqa: E402


@pytest.fixture(scope="module")
def valid_arrays() -> dict[str, np.ndarray]:
    action = np.zeros(input_loader.ACTION_SHAPE, dtype=np.float32)
    action[-1] = np.nan
    return {
        "pixels": np.zeros(input_loader.PIXELS_SHAPE, dtype=np.uint8),
        "action": action,
    }


@pytest.mark.parametrize(
    "arguments",
    (
        ["generate", "fit", "native_plan"],
        ["generate-role", "fit"],
        ["development", "fit", "native_plan"],
        ["development-role", "fit"],
    ),
)
def test_inherited_fit_role_rejects_generation_and_execution(
    arguments: list[str],
) -> None:
    with pytest.raises(launcher.DispatchError, match="inherited read-only"):
        launcher.main(arguments)


def test_input_loader_materializes_exact_two_array_contract(
    tmp_path: Path, valid_arrays: Mapping[str, np.ndarray]
) -> None:
    path = tmp_path / "episode.npz"
    np.savez_compressed(path, **valid_arrays)

    loaded, audit = input_loader.load_model_gate_inputs(
        path, expected_sha256=input_loader.sha256_file(path)
    )

    assert set(loaded) == {"pixels", "action"}
    assert loaded["pixels"].shape == (201, 224, 224, 3)
    assert loaded["pixels"].dtype == np.uint8
    assert loaded["action"].shape == (201, 5)
    assert loaded["action"].dtype == np.float32
    assert audit["arrays_materialized"] == ["action", "pixels"]
    assert audit["non_input_arrays_materialized"] is False


def test_input_loader_rejects_any_third_archive_member_before_loading_it(
    tmp_path: Path, valid_arrays: Mapping[str, np.ndarray]
) -> None:
    path = tmp_path / "extra.npz"
    np.savez_compressed(path, **valid_arrays, extra=np.asarray([1]))

    with pytest.raises(input_loader.InputContractError, match="members must be exactly"):
        input_loader.load_model_gate_inputs(path)


def test_retention_validation_is_only_shape_dtype_and_finiteness(
    valid_arrays: Mapping[str, np.ndarray]
) -> None:
    audit = input_loader.validate_model_gate_inputs(valid_arrays)
    assert audit["pixels_finite"] is True
    assert audit["modeled_actions_finite"] is True
    assert audit["terminal_action_nan_sentinel"] is True

    bad_action = valid_arrays["action"].copy()
    bad_action[17, 2] = np.inf
    with pytest.raises(input_loader.InputContractError, match="modeled action"):
        input_loader.validate_model_gate_inputs(
            {"pixels": valid_arrays["pixels"], "action": bad_action}
        )

    wrong_pixels = np.zeros((200, 224, 224, 3), dtype=np.uint8)
    with pytest.raises(input_loader.InputContractError, match="unexpected pixels"):
        input_loader.validate_model_gate_inputs(
            {"pixels": wrong_pixels, "action": valid_arrays["action"]}
        )


class PixelOnlyMapping(Mapping[str, Any]):
    def __init__(self, pixels: np.ndarray, accesses: list[str]) -> None:
        self._pixels = pixels
        self._accesses = accesses

    def __getitem__(self, key: str) -> Any:
        self._accesses.append(key)
        if key != "pixels":
            raise AssertionError(f"unexpected environment channel access: {key}")
        return self._pixels

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("generator must not enumerate environment channels")

    def __len__(self) -> int:
        raise AssertionError("generator must not count environment channels")


class UntouchableStepValue:
    def __getattribute__(self, name: str) -> Any:
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError(f"non-info step value was inspected: {name}")

    def __bool__(self) -> bool:
        raise AssertionError("non-info step value was inspected")


class FakeActionSpace:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def seed(self, seed: int) -> None:
        self.seeds.append(seed)


class FakeUnwrapped:
    def __init__(self) -> None:
        self.action_space = FakeActionSpace()


class FakeInnerEnv:
    def __init__(self) -> None:
        self.unwrapped = FakeUnwrapped()


class FakeEnvPool:
    def __init__(self, infos: Mapping[str, Any]) -> None:
        self.action_space = FakeActionSpace()
        self.envs = [FakeInnerEnv()]
        self._infos = infos
        self.step_calls = 0

    def step(self, action: np.ndarray) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        assert action.shape == (1, 5)
        self.step_calls += 1
        untouched = UntouchableStepValue()
        return untouched, untouched, untouched, untouched, self._infos


class FakeWorld:
    def __init__(self, infos: Mapping[str, Any]) -> None:
        self.infos = infos
        self.envs = FakeEnvPool(infos)

    def reset(self, *_: Any, **__: Any) -> None:
        raise AssertionError("test reset adapter must be used")


class FakePolicy:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def get_action(self, _: Mapping[str, Any]) -> np.ndarray:
        return np.zeros((1, 5), dtype=np.float32)


def test_rollout_reads_pixels_only_and_never_touches_other_step_values() -> None:
    accesses: list[str] = []
    frame = np.zeros((1, 1, 224, 224, 3), dtype=np.uint8)
    infos = PixelOnlyMapping(frame, accesses)
    world = FakeWorld(infos)
    policy = FakePolicy()

    def reset_adapter(
        received_world: Any, original_reset: Any, env_seed: int
    ) -> Mapping[str, Any]:
        assert received_world is world
        assert original_reset == world.reset
        assert env_seed == 101
        return {"reset_seed": env_seed, "values_persisted": False}

    arrays, audit = generation.generate_episode(
        world,
        policy,
        role="fit",
        regime="native_plan",
        slot=0,
        episode_id="synthetic-contract-episode",
        env_seed=101,
        policy_seed=102,
        oracle_np_seed=103,
        action_space_seed=104,
        reset_environment=reset_adapter,
    )

    assert set(arrays) == {"pixels", "action"}
    assert arrays["pixels"].shape == (201, 224, 224, 3)
    assert arrays["action"].shape == (201, 5)
    assert accesses == ["pixels"] * 201
    assert world.envs.step_calls == 200
    assert policy.seeds == [102]
    assert world.envs.action_space.seeds == [104]
    assert world.envs.envs[0].unwrapped.action_space.seeds == [104]
    assert audit["rollout_steps_completed"] == 200
    assert audit["pixel_frames_captured"] == 201
    assert audit["only_pixels_and_actions_captured"] is True


def seed_record(index: int, prefix: str = "replacement") -> dict[str, Any]:
    base = 10_000 + index * 10
    return {
        "slot": index,
        "episode_id": f"{prefix}-{index}",
        "env_seed": base + 1,
        "policy_seed": base + 2,
        "oracle_np_seed": base + 3,
        "action_space_seed": base + 4,
    }


def _canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture_seed_base(role: str, regime: str, pool: str) -> int:
    return (
        10_000
        + list(generation.EXPECTED_REGIMES).index(regime) * 1_000
        + list(generation.ROLES).index(role) * 100
        + (50 if pool == "replacements" else 0)
    )


def _registry_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    attempt = tmp_path / "runs/study/attempts/v004"
    dgp_path = attempt / "DGP_MATRIX.json"
    ledger_path = attempt / "cohort_seed_ledger.json"
    pre_data = attempt / "audit/pre_data_inheritance_seal.json"
    pre_confirmation = attempt / "audit/pre_confirmation_package_seal.json"
    registry_path = attempt / "data/replacement_registry.json"
    claims_root = attempt / "data/replacement_claims"
    lock_path = attempt / "data/replacement_registry.lock"
    reset_reference = (
        tmp_path
        / "runs/lewm_adaptive_compute_distribution_contract/generator_seedfix.py"
    )
    _canonical_json(dgp_path, {"fixture": "dgp"})
    ledger_regimes = {
        regime: {
            "roles": {
                role: {
                    "primary": [
                        _ledger_tuple(
                            role=role,
                            regime=regime,
                            pool="primary",
                            slot=0,
                            seed_base=_fixture_seed_base(role, regime, "primary"),
                        )
                    ],
                    "replacements": [
                        _ledger_tuple(
                            role=role,
                            regime=regime,
                            pool="replacements",
                            slot=slot,
                            seed_base=_fixture_seed_base(
                                role, regime, "replacements"
                            )
                            + 10 * slot,
                        )
                        for slot in range(2)
                    ],
                }
                for role in generation.ROLES
            }
        }
        for regime in generation.EXPECTED_REGIMES
    }
    _canonical_json(
        ledger_path,
        {
            "replacement_rule": {"fixture": True},
            "regimes": ledger_regimes,
        },
    )
    _canonical_json(pre_data, {"fixture": "pre-data"})
    _canonical_json(pre_confirmation, {"fixture": "pre-confirmation"})
    reset_reference.parent.mkdir(parents=True, exist_ok=True)
    reset_reference.write_text("# synthetic reset reference\n", encoding="utf-8")
    for name, value in {
        "REPO_ROOT": tmp_path,
        "ATTEMPT_ROOT": attempt,
        "DGP_MATRIX_PATH": dgp_path,
        "SEED_LEDGER_PATH": ledger_path,
        "PRE_DATA_SEAL_PATH": pre_data,
        "PRE_CONFIRMATION_SEAL_PATH": pre_confirmation,
        "REPLACEMENT_REGISTRY_PATH": registry_path,
        "REPLACEMENT_CLAIMS_ROOT": claims_root,
        "REPLACEMENT_REGISTRY_LOCK_PATH": lock_path,
        "RESET_REFERENCE_PATH": reset_reference,
    }.items():
        monkeypatch.setattr(generation, name, value)
    generation.ensure_replacement_registry(
        registry_path=registry_path,
        lock_path=lock_path,
        claims_root=claims_root,
    )
    return {
        "attempt": attempt,
        "registry": registry_path,
        "claims": claims_root,
        "lock": lock_path,
        "fit_seal": generation.SealLink(
            path=pre_data,
            relative_path=generation._relative(pre_data),
            sha256=input_loader.sha256_file(pre_data),
            checkpoint_state="PRE_OUTCOME_SEAL",
            payload={},
        ),
        "post_seal": generation.SealLink(
            path=pre_confirmation,
            relative_path=generation._relative(pre_confirmation),
            sha256=input_loader.sha256_file(pre_confirmation),
            checkpoint_state="PRE_CONFIRMATION_PACKAGE_SEAL",
            payload={},
        ),
    }


def _ledger_tuple(
    *, role: str, regime: str, pool: str, slot: int, seed_base: int
) -> dict[str, Any]:
    role_slug = {
        "fit": "ft",
        "selection": "sl",
        "smoke": "sm",
        "confirmation": "cf",
    }[role]
    regime_slug = {
        "native_plan": "np",
        "markov_oracle": "mo",
        "plan_action_noise_0p2": "n2",
        "plan_random_action_0p1": "ra",
    }[regime]
    kind = "p" if pool == "primary" else "r"
    width = 4 if role == "confirmation" and pool == "primary" else 3
    return {
        "slot": slot,
        "episode_id": f"drgv001-{regime_slug}-{role_slug}-{kind}-{slot:0{width}d}",
        "env_seed": seed_base + 1,
        "policy_seed": seed_base + 2,
        "oracle_np_seed": seed_base + 3,
        "action_space_seed": seed_base + 4,
    }


def _independent_policy(fixture: Mapping[str, Any]) -> independent.PathPolicy:
    return independent.PathPolicy(
        fixture["attempt"].parents[3],
        "runs/study/attempts/v004",
        "runs/study",
    )


def _issue_claim(
    fixture: Mapping[str, Any],
    *,
    role: str,
    regime: str,
    destination: Mapping[str, Any],
    replacements: list[dict[str, Any]],
    seal: generation.SealLink,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = (dict(destination),)
    generation.failure_log_path(role, regime).parent.mkdir(
        parents=True, exist_ok=True
    )
    failure = generation.record_rollout_failure(
        role=role,
        regime=regime,
        destination=destination,
        candidate=destination,
        replacement_used=False,
        error=generation.EpisodeRolloutError("synthetic mechanical failure"),
        primary=primary,
        replacements=replacements,
        seal=seal,
    )
    candidate = generation.claim_next_replacement(
        replacements,
        primary=primary,
        role=role,
        regime=regime,
        destination=destination,
        trigger_failure=failure,
        seal=seal,
        failed_source_ids={str(destination["episode_id"])},
        registry_path=fixture["registry"],
        lock_path=fixture["lock"],
        claims_root=fixture["claims"],
    )
    return candidate, failure


def _multi_failure_case(
    fixture: Mapping[str, Any],
    *,
    role: str,
    regime: str,
    seal: generation.SealLink,
) -> dict[str, Any]:
    ledger = json.loads(
        (fixture["attempt"] / "cohort_seed_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    role_payload = ledger["regimes"][regime]["roles"][role]
    destination = dict(role_payload["primary"][0])
    replacements = [dict(item) for item in role_payload["replacements"]]
    first, primary_failure = _issue_claim(
        fixture,
        role=role,
        regime=regime,
        destination=destination,
        replacements=replacements,
        seal=seal,
    )
    replacement_failure = generation.record_rollout_failure(
        role=role,
        regime=regime,
        destination=destination,
        candidate=first,
        replacement_used=True,
        error=generation.EpisodeRolloutError("synthetic first replacement failure"),
        primary=(destination,),
        replacements=replacements,
        seal=seal,
    )
    second = generation.claim_next_replacement(
        replacements,
        primary=(destination,),
        role=role,
        regime=regime,
        destination=destination,
        trigger_failure=replacement_failure,
        seal=seal,
        failed_source_ids={
            str(destination["episode_id"]),
            str(first["episode_id"]),
        },
        registry_path=fixture["registry"],
        lock_path=fixture["lock"],
        claims_root=fixture["claims"],
    )
    failures = generation.read_rollout_failures(
        role,
        regime,
        primary=(destination,),
        replacements=replacements,
        seal=seal,
    )
    final = {
        "slot": destination["slot"],
        "episode_id": destination["episode_id"],
        "seed_source_episode_id": second["episode_id"],
        "replacement_used": True,
        "replacement_claim_index": second["replacement_claim_index"],
        "replacement_claim_sha256": second["replacement_claim_sha256"],
        **{field: second[field] for field in generation.SEED_FIELDS},
    }
    return {
        "destination": destination,
        "replacements": replacements,
        "first": first,
        "second": second,
        "failures": failures,
        "primary_failure": primary_failure,
        "replacement_failure": replacement_failure,
        "final": final,
    }


def test_replacement_registry_is_global_across_roles_and_dgps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    first_destination = _ledger_tuple(
        role="fit",
        regime="native_plan",
        pool="primary",
        slot=0,
        seed_base=_fixture_seed_base("fit", "native_plan", "primary"),
    )
    first_replacements = [
        _ledger_tuple(
            role="fit",
            regime="native_plan",
            pool="replacements",
            slot=0,
            seed_base=_fixture_seed_base("fit", "native_plan", "replacements"),
        )
    ]
    first, failure = _issue_claim(
        fixture,
        role="fit",
        regime="native_plan",
        destination=first_destination,
        replacements=first_replacements,
        seal=fixture["fit_seal"],
    )
    retry = generation.claim_next_replacement(
        first_replacements,
        primary=(first_destination,),
        role="fit",
        regime="native_plan",
        destination=first_destination,
        trigger_failure=failure,
        seal=fixture["fit_seal"],
        failed_source_ids={str(first_destination["episode_id"])},
        registry_path=fixture["registry"],
        lock_path=fixture["lock"],
        claims_root=fixture["claims"],
    )
    second_destination = _ledger_tuple(
        role="selection",
        regime="markov_oracle",
        pool="primary",
        slot=0,
        seed_base=_fixture_seed_base("selection", "markov_oracle", "primary"),
    )
    second_replacements = [
        _ledger_tuple(
            role="selection",
            regime="markov_oracle",
            pool="replacements",
            slot=0,
            seed_base=_fixture_seed_base(
                "selection", "markov_oracle", "replacements"
            ),
        )
    ]
    second, _ = _issue_claim(
        fixture,
        role="selection",
        regime="markov_oracle",
        destination=second_destination,
        replacements=second_replacements,
        seal=fixture["fit_seal"],
    )

    assert first["episode_id"] == "drgv001-np-ft-r-000"
    assert retry == first
    assert second["episode_id"] == "drgv001-mo-sl-r-000"
    registry = generation._load_registry(fixture["registry"], fixture["claims"])
    assert registry["scope"] == "append_only_all_roles_and_all_dgps"
    assert [claim["role"] for claim in registry["claims"]] == ["fit", "selection"]
    assert [path.name for path in fixture["claims"].iterdir()] == [
        "000000.json",
        "000001.json",
    ]


def test_replacement_registry_rejects_extra_field_and_reorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    for role, regime in (
        ("fit", "native_plan"),
        ("selection", "markov_oracle"),
    ):
        _issue_claim(
            fixture,
            role=role,
            regime=regime,
            destination=_ledger_tuple(
                role=role,
                regime=regime,
                pool="primary",
                slot=0,
                seed_base=_fixture_seed_base(role, regime, "primary"),
            ),
            replacements=[
                _ledger_tuple(
                    role=role,
                    regime=regime,
                    pool="replacements",
                    slot=0,
                    seed_base=_fixture_seed_base(role, regime, "replacements"),
                )
            ],
            seal=fixture["fit_seal"],
        )
    first_path = fixture["claims"] / "000000.json"
    second_path = fixture["claims"] / "000001.json"
    policy = _independent_policy(fixture)
    ledger = json.loads(
        (fixture["attempt"] / "cohort_seed_ledger.json").read_text()
    )
    assert len(independent._independent_registry_contract(policy, ledger)["claims"]) == 2
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    tampered = {**first, "unexpected": True}
    tampered["record_sha256"] = generation._failure_hash(
        {key: value for key, value in tampered.items() if key != "record_sha256"}
    )
    _canonical_json(first_path, tampered)
    with pytest.raises(generation.EpisodePersistenceError, match="authentication drift"):
        generation._load_registry(fixture["registry"], fixture["claims"])
    with pytest.raises(independent.VerificationError, match="schema drift"):
        independent._independent_registry_contract(policy, ledger)
    _canonical_json(first_path, first)
    _canonical_json(first_path, second)
    _canonical_json(second_path, first)
    with pytest.raises(generation.EpisodePersistenceError, match="authentication drift"):
        generation._load_registry(fixture["registry"], fixture["claims"])
    with pytest.raises(independent.VerificationError, match="authentication drift"):
        independent._independent_registry_contract(policy, ledger)


def test_replacement_registry_rejects_source_reuse_and_segment_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    destination = _ledger_tuple(
        role="fit",
        regime="native_plan",
        pool="primary",
        slot=0,
        seed_base=_fixture_seed_base("fit", "native_plan", "primary"),
    )
    _issue_claim(
        fixture,
        role="fit",
        regime="native_plan",
        destination=destination,
        replacements=[
            _ledger_tuple(
                role="fit",
                regime="native_plan",
                pool="replacements",
                slot=0,
                seed_base=_fixture_seed_base(
                    "fit", "native_plan", "replacements"
                ),
            )
        ],
        seal=fixture["fit_seal"],
    )
    first_path = fixture["claims"] / "000000.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    reused = dict(first)
    reused.update(
        {
            "claim_index": 1,
            "claimed_unix_ns": int(first["claimed_unix_ns"]) + 1,
            "prev_sha256": first["record_sha256"],
            "trigger_failure_seq": 2,
            "trigger_failure_record_sha256": "f" * 64,
            "failed_seed_source_episode_id": first["replacement_episode_id"],
        }
    )
    reused["record_sha256"] = generation._failure_hash(
        {key: value for key, value in reused.items() if key != "record_sha256"}
    )
    _canonical_json(fixture["claims"] / "000001.json", reused)
    policy = _independent_policy(fixture)
    ledger = json.loads(
        (fixture["attempt"] / "cohort_seed_ledger.json").read_text()
    )
    with pytest.raises(generation.EpisodePersistenceError, match="reuses a source"):
        generation._load_registry(fixture["registry"], fixture["claims"])
    with pytest.raises(independent.VerificationError, match="reuses source"):
        independent._independent_registry_contract(policy, ledger)

    (fixture["claims"] / "000001.json").rename(
        fixture["claims"] / "000002.json"
    )
    with pytest.raises(generation.EpisodePersistenceError, match="contiguous prefix"):
        generation._load_registry(fixture["registry"], fixture["claims"])
    with pytest.raises(independent.VerificationError, match="contiguous exact prefix"):
        independent._independent_registry_contract(policy, ledger)


def test_checkpoint_failure_chain_v2_rejects_tamper_reorder_and_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    destination = _ledger_tuple(
        role="fit",
        regime="native_plan",
        pool="primary",
        slot=0,
        seed_base=_fixture_seed_base("fit", "native_plan", "primary"),
    )
    replacements = [
        _ledger_tuple(
            role="fit",
            regime="native_plan",
            pool="replacements",
            slot=0,
            seed_base=_fixture_seed_base("fit", "native_plan", "replacements"),
        )
    ]
    claimed, _ = _issue_claim(
        fixture,
        role="fit",
        regime="native_plan",
        destination=destination,
        replacements=replacements,
        seal=fixture["fit_seal"],
    )
    generation.record_rollout_failure(
        role="fit",
        regime="native_plan",
        destination=destination,
        candidate=claimed,
        replacement_used=True,
        error=generation.EpisodeRolloutError("synthetic replacement failure"),
        primary=(destination,),
        replacements=replacements,
        seal=fixture["fit_seal"],
    )
    path = generation.failure_log_path("fit", "native_plan")
    claims = generation._load_registry(fixture["registry"], fixture["claims"])[
        "claims"
    ]

    def verify() -> list[dict[str, Any]]:
        return checkpoints._failure_chain_v2(
            path,
            role="fit",
            regime="native_plan",
            primary=(destination,),
            replacements=replacements,
            authorization=fixture["fit_seal"].as_json(),
            registry_claims=claims,
        )

    def write_records(records: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
        )

    original = [json.loads(line) for line in path.read_text().splitlines()]
    assert verify() == original

    attacks: list[list[dict[str, Any]]] = []
    attacks.append([dict(original[1]), dict(original[0])])
    extra = [dict(item) for item in original]
    extra[0]["unexpected"] = True
    extra[0]["record_sha256"] = generation._failure_hash(
        {key: value for key, value in extra[0].items() if key != "record_sha256"}
    )
    attacks.append(extra)
    wrong_claim = [dict(item) for item in original]
    wrong_claim[1]["replacement_claim_sha256"] = "e" * 64
    wrong_claim[1]["record_sha256"] = generation._failure_hash(
        {
            key: value
            for key, value in wrong_claim[1].items()
            if key != "record_sha256"
        }
    )
    attacks.append(wrong_claim)
    reused = dict(original[1])
    reused["seq"] = 3
    reused["created_unix_ns"] = int(reused["created_unix_ns"]) + 1
    reused["prev_sha256"] = original[1]["record_sha256"]
    reused["record_sha256"] = generation._failure_hash(
        {key: value for key, value in reused.items() if key != "record_sha256"}
    )
    attacks.append([*original, reused])

    for attack in attacks:
        write_records(attack)
        with pytest.raises(RuntimeError, match="failure"):
            verify()
    write_records(original)
    assert verify() == original


def test_retained_partition_accepts_zero_and_multiple_failures_but_rejects_failed_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    ledger = json.loads(
        (fixture["attempt"] / "cohort_seed_ledger.json").read_text()
    )
    zero_primary = ledger["regimes"]["markov_oracle"]["roles"]["fit"][
        "primary"
    ][0]
    zero_record = {
        **zero_primary,
        "seed_source_episode_id": zero_primary["episode_id"],
        "replacement_used": False,
        "replacement_claim_index": None,
        "replacement_claim_sha256": None,
    }
    zero = generation._validate_retained_failure_partition(
        [zero_record],
        [],
        role="fit",
        regime="markov_oracle",
        primary=(zero_primary,),
        require_complete=True,
    )
    assert zero["authenticated_failed_source_count"] == 0

    case = _multi_failure_case(
        fixture,
        role="fit",
        regime="native_plan",
        seal=fixture["fit_seal"],
    )
    evidence = generation._validate_retained_failure_partition(
        [case["final"]],
        case["failures"],
        role="fit",
        regime="native_plan",
        primary=(case["destination"],),
        require_complete=True,
    )
    assert evidence["authenticated_failed_source_count"] == 2

    failed_retained = {
        **case["final"],
        "seed_source_episode_id": case["first"]["episode_id"],
        "replacement_claim_index": case["first"]["replacement_claim_index"],
        "replacement_claim_sha256": case["first"]["replacement_claim_sha256"],
        **{
            field: case["first"][field]
            for field in generation.SEED_FIELDS
        },
    }
    with pytest.raises(
        generation.EpisodePersistenceError,
        match="retained source has an authenticated",
    ):
        generation._validate_retained_failure_partition(
            [failed_retained],
            case["failures"],
            role="fit",
            regime="native_plan",
            primary=(case["destination"],),
            require_complete=True,
        )
    wrong_type = {**case["final"], "replacement_used": 1}
    with pytest.raises(generation.EpisodePersistenceError, match="exact boolean"):
        generation._validate_retained_failure_partition(
            [wrong_type],
            case["failures"],
            role="fit",
            regime="native_plan",
            primary=(case["destination"],),
            require_complete=True,
        )


def test_present_empty_or_malformed_failure_log_stops_before_generation_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    ledger = json.loads(
        (fixture["attempt"] / "cohort_seed_ledger.json").read_text()
    )
    role_payload = ledger["regimes"]["native_plan"]["roles"]["fit"]
    primary = tuple(role_payload["primary"])
    replacements = tuple(role_payload["replacements"])
    failure_path = generation.failure_log_path("fit", "native_plan")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_bytes(b"")
    with pytest.raises(generation.EpisodePersistenceError, match="empty"):
        generation.read_rollout_failures(
            "fit",
            "native_plan",
            primary=primary,
            replacements=replacements,
            seal=fixture["fit_seal"],
        )

    monkeypatch.setattr(generation, "validate_existing_manifest", lambda *args: None)
    monkeypatch.setattr(generation, "verify_controller_permission", lambda role: {})
    registry_called = False

    def forbidden_registry(*args: Any, **kwargs: Any) -> None:
        nonlocal registry_called
        registry_called = True
        raise AssertionError("registry/output initialization followed empty log")

    monkeypatch.setattr(generation, "ensure_replacement_registry", forbidden_registry)
    inputs = generation.GenerationInputs(
        dgp_matrix={"fixture": True},
        seed_ledger=ledger,
        seal=fixture["fit_seal"],
        regime_specification={"fixture": True},
        primary=primary,
        replacements=replacements,
    )
    with pytest.raises(generation.EpisodePersistenceError, match="empty"):
        generation._generate_locked("fit", "native_plan", inputs)
    assert registry_called is False
    assert not generation.raw_directory("fit", "native_plan").exists()

    failure_path.write_bytes(b"{malformed\n")
    with pytest.raises(generation.EpisodePersistenceError, match="invalid"):
        generation.read_rollout_failures(
            "fit",
            "native_plan",
            primary=primary,
            replacements=replacements,
            seal=fixture["fit_seal"],
        )


def test_confirmation_checkpoint_requires_failure_free_terminal_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    case = _multi_failure_case(
        fixture,
        role="confirmation",
        regime="native_plan",
        seal=fixture["post_seal"],
    )
    claims = generation._load_registry(fixture["registry"], fixture["claims"])[
        "claims"
    ]
    path = generation.failure_log_path("confirmation", "native_plan")
    verified = checkpoints._validate_rollout_failure_log(
        path,
        regime="native_plan",
        primary=(case["destination"],),
        replacements=case["replacements"],
        final_records={case["destination"]["episode_id"]: case["final"]},
        authorization=fixture["post_seal"].as_json(),
        registry_claims=claims,
    )
    assert len(verified) == 2
    failed_retained = {
        **case["final"],
        "seed_source_episode_id": case["first"]["episode_id"],
        "replacement_claim_index": case["first"]["replacement_claim_index"],
        "replacement_claim_sha256": case["first"]["replacement_claim_sha256"],
        **{
            field: case["first"][field]
            for field in generation.SEED_FIELDS
        },
    }
    with pytest.raises(RuntimeError, match="retained source has"):
        checkpoints._validate_rollout_failure_log(
            path,
            regime="native_plan",
            primary=(case["destination"],),
            replacements=case["replacements"],
            final_records={
                case["destination"]["episode_id"]: failed_retained
            },
            authorization=fixture["post_seal"].as_json(),
            registry_claims=claims,
        )


def test_independent_verifier_authenticates_multi_failure_partition_and_adversaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    case = _multi_failure_case(
        fixture,
        role="fit",
        regime="native_plan",
        seal=fixture["fit_seal"],
    )
    ledger = json.loads(
        (fixture["attempt"] / "cohort_seed_ledger.json").read_text()
    )
    retained: dict[str, dict[str, list[dict[str, Any]]]] = {"fit": {}}
    for regime in generation.EXPECTED_REGIMES:
        primary = ledger["regimes"][regime]["roles"]["fit"]["primary"][0]
        retained["fit"][regime] = [
            {
                **primary,
                "seed_source_episode_id": primary["episode_id"],
                "replacement_used": False,
                "replacement_claim_index": None,
                "replacement_claim_sha256": None,
            }
        ]
    retained["fit"]["native_plan"] = [dict(case["final"])]
    policy = _independent_policy(fixture)
    evidence = independent._verify_replacement_registry_and_failures(
        policy, ledger, retained, ("fit",)
    )
    assert evidence["authenticated_failure_count"] == 2
    assert evidence["claim_dispositions"] == {"0": "failed", "1": "retained"}

    wrong_type = {
        role: {dgp: [dict(records[0])] for dgp, records in regimes.items()}
        for role, regimes in retained.items()
    }
    wrong_type["fit"]["native_plan"][0]["replacement_used"] = 1
    with pytest.raises(independent.VerificationError, match="exact boolean"):
        independent._verify_replacement_registry_and_failures(
            policy, ledger, wrong_type, ("fit",)
        )

    failed_retained = {
        role: {dgp: [dict(records[0])] for dgp, records in regimes.items()}
        for role, regimes in retained.items()
    }
    failed_retained["fit"]["native_plan"][0].update(
        {
            "seed_source_episode_id": case["first"]["episode_id"],
            "replacement_claim_index": case["first"]["replacement_claim_index"],
            "replacement_claim_sha256": case["first"]["replacement_claim_sha256"],
            **{
                field: case["first"][field]
                for field in generation.SEED_FIELDS
            },
        }
    )
    with pytest.raises(independent.VerificationError, match="retained source has"):
        independent._verify_replacement_registry_and_failures(
            policy, ledger, failed_retained, ("fit",)
        )

    failure_path = generation.failure_log_path("fit", "native_plan")
    original = failure_path.read_bytes()
    lines = original.splitlines()
    failure_path.write_bytes(b"\n".join(reversed(lines)) + b"\n")
    with pytest.raises(independent.VerificationError, match="authentication drift"):
        independent._verify_replacement_registry_and_failures(
            policy, ledger, retained, ("fit",)
        )
    failure_path.write_bytes(original)

    empty_path = generation.failure_log_path("fit", "markov_oracle")
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    empty_path.write_bytes(b"")
    with pytest.raises(independent.VerificationError, match="empty or partial"):
        independent._verify_replacement_registry_and_failures(
            policy, ledger, retained, ("fit",)
        )


def fake_seal(tmp_path: Path) -> generation.SealLink:
    path = tmp_path / "seal.json"
    path.write_text("{}\n", encoding="utf-8")
    return generation.SealLink(
        path=path,
        relative_path="seal.json",
        sha256=input_loader.sha256_file(path),
        checkpoint_state="PRE_OUTCOME_SEAL",
        payload={},
    )


def generation_audit(
    arrays: Mapping[str, np.ndarray], source: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "rows": 201,
        "initial_pixels_sha256": input_loader.array_sha256(arrays["pixels"][0]),
        "rollout_steps_expected": 200,
        "rollout_steps_completed": 200,
        "pixel_frames_captured": 201,
        "input_validation": input_loader.validate_model_gate_inputs(arrays),
        "reset_metadata": {
            "environment_seed": int(source["env_seed"]),
            "variation_seed": int(source["env_seed"]),
            "physical_state_seed": int(source["env_seed"]),
            "variation_values_sha256": "a" * 64,
            "reset_reference_path": generation._relative(
                generation.RESET_REFERENCE_PATH
            ),
            "reset_reference_sha256": input_loader.sha256_file(
                generation.RESET_REFERENCE_PATH
            ),
            "reset_values_persisted": False,
        },
        "action_space_seed": int(source["action_space_seed"]),
        "action_spaces_seeded": [
            "vector_action_space",
            "unwrapped_action_space",
        ],
        "rng_activation_order": generation.RNG_ACTIVATION_ORDER,
        "only_pixels_and_actions_captured": True,
        "retention_uses_only_input_contract_and_local_step_bookkeeping": True,
    }


def _raw_orphan_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
    *,
    role: str = "fit",
    replacement: bool = False,
) -> dict[str, Any]:
    fixture = _registry_fixture(tmp_path, monkeypatch)
    regime = "native_plan"
    destination = _ledger_tuple(
        role=role,
        regime=regime,
        pool="primary",
        slot=0,
        seed_base=_fixture_seed_base(role, regime, "primary"),
    )
    replacements = [
        _ledger_tuple(
            role=role,
            regime=regime,
            pool="replacements",
            slot=slot,
            seed_base=_fixture_seed_base(role, regime, "replacements")
            + 10 * slot,
        )
        for slot in range(2)
    ]
    seal = fixture["fit_seal"] if role in ("fit", "selection") else fixture["post_seal"]
    failures: list[dict[str, Any]] = []
    candidate: dict[str, Any] = dict(destination)
    if replacement:
        candidate, failure = _issue_claim(
            fixture,
            role=role,
            regime=regime,
            destination=destination,
            replacements=replacements,
            seal=seal,
        )
        failures.append(failure)
    paths = generation.episode_paths(role, regime, destination["episode_id"])
    intent = generation.build_persistence_intent(
        paths=paths,
        role=role,
        regime=regime,
        destination=destination,
        candidate=candidate,
        replacement_used=replacement,
        primary=(destination,),
        replacements=replacements,
        arrays=valid_arrays,
        generation_audit=generation_audit(valid_arrays, candidate),
        seal=seal,
    )
    generation.atomic_json(paths.intent, intent, exclusive=True)
    generation.atomic_npz(paths.raw, valid_arrays, exclusive=True)
    return {
        **fixture,
        "role": role,
        "regime": regime,
        "destination": destination,
        "candidate": candidate,
        "primary": (destination,),
        "replacements": replacements,
        "failures": failures,
        "seal": seal,
        "paths": paths,
        "intent": intent,
    }


def _adopt_fixture(case: Mapping[str, Any]) -> dict[str, Any]:
    return generation.adopt_orphan(
        paths=case["paths"],
        role=case["role"],
        regime=case["regime"],
        destination=case["destination"],
        seal=case["seal"],
        primary=case["primary"],
        replacements=case["replacements"],
        failures=case["failures"],
    )


@pytest.mark.parametrize("role", ["fit", "confirmation"])
def test_safe_raw_orphan_is_adopted_only_through_closed_matching_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
    role: str,
) -> None:
    case = _raw_orphan_fixture(
        tmp_path, monkeypatch, valid_arrays, role=role
    )
    record = generation.resume_episode(
        role=case["role"],
        regime=case["regime"],
        destination=case["destination"],
        seal=case["seal"],
        primary=case["primary"],
        replacements=case["replacements"],
        failures=case["failures"],
    )
    assert record is not None
    paths = case["paths"]

    assert paths.sidecar.is_file()
    assert record["orphan_adoption_safe"] is True
    assert record["raw_sha256"] == input_loader.sha256_file(paths.raw)
    assert record["persistence_intent_sha256"] == input_loader.sha256_file(
        paths.intent
    )
    sidecar_bytes = paths.sidecar.read_bytes()
    repeated = generation.resume_episode(
        role=case["role"],
        regime=case["regime"],
        destination=case["destination"],
        seal=case["seal"],
        primary=case["primary"],
        replacements=case["replacements"],
        failures=case["failures"],
    )
    assert repeated == record
    with pytest.raises(generation.EpisodePersistenceError, match="raw-only"):
        _adopt_fixture(case)
    assert paths.sidecar.read_bytes() == sidecar_bytes


def test_replacement_raw_orphan_resume_binds_claim_failure_and_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
) -> None:
    case = _raw_orphan_fixture(
        tmp_path, monkeypatch, valid_arrays, replacement=True
    )

    record = _adopt_fixture(case)

    assert record["replacement_used"] is True
    assert record["seed_source_episode_id"] == case["candidate"]["episode_id"]
    assert record["replacement_claim_sha256"] == case["candidate"][
        "replacement_claim_sha256"
    ]


def test_confirmation_consumer_accepts_only_the_closed_v004_intent_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
) -> None:
    case = _raw_orphan_fixture(
        tmp_path, monkeypatch, valid_arrays, role="confirmation"
    )
    record = _adopt_fixture(case)
    monkeypatch.setattr(checkpoints, "REPO_ROOT", tmp_path)
    intent = json.loads(case["paths"].intent.read_text(encoding="utf-8"))

    checkpoints._validate_generation_audit_contract(
        record["generation_audit"],
        arrays=record["arrays"],
        initial_pixels_sha256=record["initial_pixels_sha256"],
        source=case["destination"],
        label="synthetic confirmation generation audit",
    )
    accepted = checkpoints._validate_closed_persistence_intent_contract(
        intent,
        record=record,
        destination=case["destination"],
        source=case["destination"],
        source_pool="primary",
        source_slot=0,
        arrays=record["arrays"],
        raw_path=case["paths"].raw,
        sidecar_path=case["paths"].sidecar,
        intent_path=case["paths"].intent,
        dgp_path=generation.DGP_MATRIX_PATH,
        ledger_path=generation.SEED_LEDGER_PATH,
        label="synthetic confirmation intent",
    )
    assert accepted == intent

    for tampered in (
        {**intent, "contact": 0},
        {**intent, "seed_source_slot": True},
        {**intent, "raw_path": intent["raw_path"] + ".alternate"},
    ):
        with pytest.raises(RuntimeError):
            checkpoints._validate_closed_persistence_intent_contract(
                tampered,
                record=record,
                destination=case["destination"],
                source=case["destination"],
                source_pool="primary",
                source_slot=0,
                arrays=record["arrays"],
                raw_path=case["paths"].raw,
                sidecar_path=case["paths"].sidecar,
                intent_path=case["paths"].intent,
                dgp_path=generation.DGP_MATRIX_PATH,
                ledger_path=generation.SEED_LEDGER_PATH,
                label="tampered confirmation intent",
            )


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "extra",
        "contact_like",
        "attempt",
        "science_attempt",
        "role",
        "dgp",
        "authorization",
        "seed",
        "source_record",
        "source_pool",
        "path",
        "dgp_hash",
        "ledger_hash",
        "array_hash",
        "generation_extra",
        "reset_contact_like",
        "initial_hash",
    ],
)
def test_raw_orphan_intent_adversaries_stop_before_sidecar_or_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
    mutation: str,
) -> None:
    case = _raw_orphan_fixture(tmp_path, monkeypatch, valid_arrays)
    intent = json.loads(json.dumps(case["intent"]))
    if mutation == "schema":
        intent["schema_version"] = 3
    elif mutation == "extra":
        intent["unexpected"] = True
    elif mutation == "contact_like":
        intent["contact"] = 0
    elif mutation == "attempt":
        intent["attempt"] = "v001"
    elif mutation == "science_attempt":
        intent["science_attempt"] = "v004"
    elif mutation == "role":
        intent["role"] = "selection"
    elif mutation == "dgp":
        intent["regime"] = "markov_oracle"
    elif mutation == "authorization":
        intent["authorization_seal"]["sha256"] = "b" * 64
    elif mutation == "seed":
        intent["env_seed"] += 1
    elif mutation == "source_record":
        intent["seed_source_record"]["policy_seed"] += 1
    elif mutation == "source_pool":
        intent["seed_source_pool"] = "replacements"
    elif mutation == "path":
        intent["raw_path"] += ".alternate"
    elif mutation == "dgp_hash":
        intent["dgp_matrix_sha256"] = "c" * 64
    elif mutation == "ledger_hash":
        intent["cohort_seed_ledger_sha256"] = "d" * 64
    elif mutation == "array_hash":
        intent["arrays"]["action"]["sha256"] = "e" * 64
    elif mutation == "generation_extra":
        intent["generation_audit"]["unexpected"] = True
    elif mutation == "reset_contact_like":
        intent["generation_audit"]["reset_metadata"]["phase"] = 1
    elif mutation == "initial_hash":
        intent["initial_pixels_sha256"] = "f" * 64
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError(mutation)
    _canonical_json(case["paths"].intent, intent)

    with pytest.raises(generation.EpisodePersistenceError):
        _adopt_fixture(case)

    assert not case["paths"].sidecar.exists()
    assert not generation.failure_log_path("fit", "native_plan").exists()
    assert generation._load_registry(case["registry"], case["claims"])["claims"] == []


@pytest.mark.parametrize("partial", ["malformed", "valid_subset", "noncanonical"])
def test_raw_orphan_malformed_or_partial_intent_never_writes_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
    partial: str,
) -> None:
    case = _raw_orphan_fixture(tmp_path, monkeypatch, valid_arrays)
    if partial == "malformed":
        case["paths"].intent.write_bytes(b'{"schema_version": 2')
    elif partial == "valid_subset":
        _canonical_json(case["paths"].intent, {"schema_version": 2})
    else:
        case["paths"].intent.write_text(
            json.dumps(case["intent"], sort_keys=True), encoding="utf-8"
        )

    with pytest.raises(generation.EpisodePersistenceError):
        _adopt_fixture(case)

    assert not case["paths"].sidecar.exists()
    assert generation._load_registry(case["registry"], case["claims"])["claims"] == []


@pytest.mark.parametrize(
    "link_attack", ["raw_symlink", "raw_hardlink", "intent_symlink", "sidecar_dangling"]
)
def test_raw_orphan_link_and_alias_attacks_stop_before_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
    link_attack: str,
) -> None:
    case = _raw_orphan_fixture(tmp_path, monkeypatch, valid_arrays)
    paths = case["paths"]
    if link_attack == "raw_symlink":
        target = paths.raw.with_name("raw-symlink-target.npz")
        paths.raw.rename(target)
        paths.raw.symlink_to(target)
    elif link_attack == "raw_hardlink":
        paths.raw.with_name("raw-hardlink-alias.npz").hardlink_to(paths.raw)
    elif link_attack == "intent_symlink":
        target = paths.intent.with_name("intent-symlink-target.json")
        paths.intent.rename(target)
        paths.intent.symlink_to(target)
    else:
        paths.sidecar.symlink_to(paths.sidecar.with_name("missing-sidecar-target"))

    with pytest.raises(generation.EpisodePersistenceError):
        _adopt_fixture(case)

    assert not paths.sidecar.is_file()


def test_raw_orphan_recomputes_raw_arrays_and_rejects_post_intent_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
) -> None:
    case = _raw_orphan_fixture(tmp_path, monkeypatch, valid_arrays)
    changed_action = valid_arrays["action"].copy()
    changed_action[0, 0] = 1.0
    np.savez_compressed(
        case["paths"].raw,
        pixels=valid_arrays["pixels"],
        action=changed_action,
    )

    with pytest.raises(generation.EpisodePersistenceError, match="raw content"):
        _adopt_fixture(case)

    assert not case["paths"].sidecar.exists()


@pytest.mark.parametrize("source", ["authorization", "dgp", "ledger", "reset"])
def test_raw_orphan_rejects_live_crosslink_change_after_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
    source: str,
) -> None:
    case = _raw_orphan_fixture(tmp_path, monkeypatch, valid_arrays)
    path = {
        "authorization": case["seal"].path,
        "dgp": generation.DGP_MATRIX_PATH,
        "ledger": generation.SEED_LEDGER_PATH,
        "reset": generation.RESET_REFERENCE_PATH,
    }[source]
    path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    with pytest.raises(generation.EpisodePersistenceError):
        _adopt_fixture(case)

    assert not case["paths"].sidecar.exists()


def test_raw_orphan_without_intent_stops_instead_of_replacing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
) -> None:
    case = _raw_orphan_fixture(tmp_path, monkeypatch, valid_arrays)
    paths = case["paths"]
    paths.intent.unlink()

    with pytest.raises(generation.EpisodePersistenceError, match="no durable"):
        _adopt_fixture(case)
    assert not paths.sidecar.exists()


def test_persistence_error_is_not_a_rollout_error_and_does_not_claim_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_arrays: Mapping[str, np.ndarray],
) -> None:
    case = _raw_orphan_fixture(tmp_path, monkeypatch, valid_arrays)
    paths = case["paths"]
    paths.raw.unlink()
    paths.intent.unlink()

    def fail_write(*_: Any, **__: Any) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(generation, "atomic_npz", fail_write)
    with pytest.raises(
        generation.EpisodePersistenceError, match="replacement is forbidden"
    ) as caught:
        generation.persist_episode(
            paths=paths,
            role="fit",
            regime="native_plan",
            destination=case["destination"],
            candidate=case["destination"],
            replacement_used=False,
            primary=case["primary"],
            replacements=case["replacements"],
            arrays=valid_arrays,
            generation_audit=generation_audit(
                valid_arrays, case["destination"]
            ),
            seal=case["seal"],
        )

    assert not isinstance(caught.value, generation.EpisodeRolloutError)
    assert not (tmp_path / "replacement_registry.json").exists()
    assert not paths.sidecar.exists()
    assert paths.intent.exists()


def test_intent_only_partial_artifact_stops_without_reroll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generation, "ATTEMPT_ROOT", tmp_path)
    destination = {
        **seed_record(31),
        "episode_id": "drgv001-np-ft-p-031",
    }
    paths = generation.episode_paths(
        "fit", "native_plan", destination["episode_id"]
    )
    paths.intent.parent.mkdir(parents=True)
    paths.intent.write_text("{}\n", encoding="utf-8")

    with pytest.raises(generation.EpisodePersistenceError, match="cannot be regenerated"):
        generation.resume_episode(
            role="fit",
            regime="native_plan",
            destination=destination,
            seal=fake_seal(tmp_path),
            primary=(destination,),
            replacements=(),
            failures=(),
        )


def test_paths_are_role_and_regime_isolated() -> None:
    fit = generation.episode_paths(
        "fit", "native_plan", "drgv001-np-ft-p-001"
    )
    selection = generation.episode_paths(
        "selection", "native_plan", "drgv001-np-sl-p-001"
    )
    shifted = generation.episode_paths(
        "fit", "markov_oracle", "drgv001-mo-ft-p-001"
    )
    assert fit.raw != selection.raw != shifted.raw
    assert generation.raw_manifest_path("confirmation", "native_plan").as_posix().endswith(
        "data/confirmation/native_plan/raw_manifest.json"
    )


def _patch_empty_output_namespace(
    attempt_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generation, "ATTEMPT_ROOT", attempt_root)
    monkeypatch.setattr(
        generation,
        "REPLACEMENT_REGISTRY_PATH",
        attempt_root / "data/replacement_registry.json",
    )
    monkeypatch.setattr(
        generation,
        "REPLACEMENT_CLAIMS_ROOT",
        attempt_root / "data/replacement_claims",
    )
    monkeypatch.setattr(
        generation,
        "REPLACEMENT_REGISTRY_LOCK_PATH",
        attempt_root / "data/replacement_registry.lock",
    )


@pytest.mark.parametrize("role", generation.ROLES)
@pytest.mark.parametrize("regime", generation.EXPECTED_REGIMES)
def test_empty_attempt_preflight_accepts_all_sixteen_role_dgp_cells_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    regime: str,
) -> None:
    """The root-level lock parent is valid for every sealed role-by-DGP cell."""

    attempt_root = tmp_path / "attempts/v004"
    attempt_root.mkdir(parents=True)
    _patch_empty_output_namespace(attempt_root, monkeypatch)
    width = 4 if role == "confirmation" else 3
    episode_id = (
        f"drgv001-{generation.REGIME_SLUGS[regime]}-"
        f"{generation.ROLE_SLUGS[role]}-p-{0:0{width}d}"
    )
    before = tuple(attempt_root.rglob("*"))

    result = generation.preflight_output_namespace(
        role,
        regime,
        ({"episode_id": episode_id},),
    )

    assert result["passed"] is True
    assert result["role"] == role
    assert result["regime"] == regime
    assert result["episode_count"] == 1
    assert tuple(attempt_root.rglob("*")) == before == ()


def test_only_directory_chain_admits_exact_attempt_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path / "attempts/v004"
    attempt_root.mkdir(parents=True)
    _patch_empty_output_namespace(attempt_root, monkeypatch)
    seen: dict[tuple[int, int], Path] = {}

    generation._assert_safe_directory_chain(attempt_root, seen_inodes=seen)

    info = attempt_root.lstat()
    assert seen == {(int(info.st_dev), int(info.st_ino)): attempt_root}
    with pytest.raises(generation.GenerationPreflightError, match="noncanonical"):
        generation._lexical_attempt_relative(attempt_root)
    with pytest.raises(generation.GenerationPreflightError, match="noncanonical"):
        generation._assert_safe_output_leaf(
            attempt_root,
            seen_directories={},
            seen_files={},
        )
    with pytest.raises(generation.GenerationPreflightError, match="noncanonical"):
        generation._assert_safe_directory_chain(Path("."), seen_inodes={})


def test_attempt_root_directory_chain_rejects_symlink_non_directory_and_inode_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_root = tmp_path / "real/v004"
    real_root.mkdir(parents=True)
    linked_root = tmp_path / "linked-v004"
    linked_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(generation, "ATTEMPT_ROOT", linked_root)
    with pytest.raises(generation.GenerationPreflightError, match="linked"):
        generation._assert_safe_directory_chain(linked_root, seen_inodes={})

    regular_root = tmp_path / "regular-v004"
    regular_root.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setattr(generation, "ATTEMPT_ROOT", regular_root)
    with pytest.raises(generation.GenerationPreflightError, match="not a directory"):
        generation._assert_safe_directory_chain(regular_root, seen_inodes={})

    monkeypatch.setattr(generation, "ATTEMPT_ROOT", real_root)
    info = real_root.lstat()
    with pytest.raises(generation.GenerationPreflightError, match="alias one inode"):
        generation._assert_safe_directory_chain(
            real_root,
            seen_inodes={(int(info.st_dev), int(info.st_ino)): tmp_path / "other"},
        )
