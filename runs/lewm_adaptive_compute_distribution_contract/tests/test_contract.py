from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze  # noqa: E402
import common  # noqa: E402
import evaluate  # noqa: E402
import prepare  # noqa: E402


def test_preregistered_design_is_exact_and_nonconfirmatory() -> None:
    cfg = prepare.config_payload()
    assert cfg["sample"] == {
        "smoke_episodes_per_policy": 12,
        "plan_discovery_episodes": 90,
        "paired_markov_diagnostic_episodes": 30,
        "examples_per_episode": 38,
        "sequential_expansion_forbidden": True,
        "v5_confirmation_episodes": 0,
    }
    assert cfg["classification"].endswith("not_confirmation")
    assert cfg["policy_constructors"]["plan_oracle"]["policy_type"] == "plan_oracle"
    assert cfg["policy_constructors"]["markov_oracle"]["policy_type"] == "markov_oracle"
    assert cfg["reference_region"]["core_metrics"] == prepare.CORE_METRICS


def test_seed_blocks_are_prior_disjoint_and_pair_only_environment_rng() -> None:
    manifest = prepare.seed_manifest()
    assert manifest["overlap_with_recorded_prior_identifiers"] == []
    plan = manifest["roles"]["main"]["plan_oracle"][:30]
    markov = manifest["roles"]["main"]["markov_oracle"]
    assert [item["env_seed"] for item in plan] == [item["env_seed"] for item in markov]
    assert not ({item["policy_seed"] for item in plan} & {item["policy_seed"] for item in markov})
    assert not ({item["oracle_np_seed"] for item in plan} & {item["oracle_np_seed"] for item in markov})
    assert len(manifest["roles"]["smoke"]["plan_oracle"]) == 12
    assert len(manifest["roles"]["smoke"]["markov_oracle"]) == 12


def test_action_window_reconstruction_is_exact() -> None:
    rng = np.random.default_rng(123)
    raw = rng.uniform(-1, 1, size=(200, 5)).astype(np.float32)
    normalized = (raw - common.FROZEN_ACTION_MEAN) / common.FROZEN_ACTION_STD
    blocks = normalized.reshape(40, 25)
    windows = np.stack([blocks[index : index + 3] for index in range(38)])
    reconstructed = evaluate._reconstruct_raw_actions(windows)
    np.testing.assert_allclose(reconstructed, raw, rtol=2e-6, atol=2e-7)


def test_feature_block_partition_and_gate_cost() -> None:
    assert common.FEATURE_BLOCKS[0] == ("history_latents", 0, 576)
    assert common.FEATURE_BLOCKS[-1] == ("depth_one_hot", 1046, 1049)
    assert sum(end - start for _, start, end in common.FEATURE_BLOCKS) == 1049
    adapter = 2 * (843 * 128 + 128 * 192)
    gate = 192 + 3806 + 2 * 1049 + (2 * 1049 + 1) + 1
    assert adapter == 264_960 == common.STAGE_ADAPTER_FLOPS
    assert gate == 8_196 == common.GATE_FLOPS_PER_DECISION
    assert 1049 + 1 == 1_050 == common.GATE_PARAMETERS


@pytest.mark.parametrize(
    ("valid", "matched", "gate", "expected"),
    [
        (False, False, False, "distribution_contract_invalid"),
        (True, False, True, "generator_reconstruction_failed"),
        (True, True, False, "distribution_matched_gate_failed"),
        (True, True, True, "distribution_contract_passed"),
    ],
)
def test_decision_tree(valid: bool, matched: bool, gate: bool, expected: str) -> None:
    assert analyze.map_decision(valid, matched, gate) == expected


def test_causal_feature_builder_has_no_target_argument() -> None:
    runtime = common.load_runtime()
    _, models, _ = runtime.load_discovery_modules()
    assert "target" not in models.build_causal_features.__code__.co_varnames
    names = models.causal_feature_names(latent_dim=192, action_dim=25, history_len=3)
    assert len(names) == 1046
    assert len(set(names)) == 1046


def test_v3_isolation_module_has_no_test_extraction_role() -> None:
    isolation = common.load_isolation()
    with pytest.raises(ValueError):
        isolation.extraction_splits("test")
    sets = isolation.load_pinned_v3_episode_sets()
    assert not (set(sets["train"].tolist()) & set(sets["test"].tolist()))
    assert not (set(sets["calibration"].tolist()) & set(sets["test"].tolist()))


def test_frozen_hash_contract() -> None:
    objects = common.verify_frozen_objects()
    assert objects["base_weights"]["sha256"] == common.EXPECTED["base_weights_sha256"]
    assert objects["solver_checkpoint"]["sha256"] == common.EXPECTED["solver_checkpoint_sha256"]
    assert objects["student_checkpoint"]["sha256"] == common.EXPECTED["student_checkpoint_sha256"]
    assert objects["whitening"]["sha256"] == common.EXPECTED["whitening_sha256"]


def test_reference_artifacts_if_present_are_test_clear() -> None:
    path = ROOT / "reference/offline_episode_metrics.npz"
    if not path.exists():
        pytest.skip("pre-freeze unit phase")
    with np.load(path, allow_pickle=False) as stored:
        ids = set(stored["episode_id"].astype(int).tolist())
        assert len(ids) == 510
    assert not (ids & common.v3_test_set())
    bands = json.loads((ROOT / "reference/reference_bands.json").read_text())
    assert bands["status"] == "frozen_before_any_new_smoke_or_main_rollout"
    assert bands["v4_used_to_define_any_band_or_rule"] is False


def test_final_artifacts_if_present_have_exact_calls_and_no_v5() -> None:
    decision_path = ROOT / "decision.json"
    if not decision_path.exists():
        pytest.skip("pre-final unit phase")
    with np.load(ROOT / "data/plan_oracle_evaluation.npz", allow_pickle=False) as stored:
        calls = np.ones(len(stored["scores"]), dtype=np.int64)
        active = np.ones(len(calls), dtype=bool)
        for stage in range(3):
            active &= stored["scores"][:, stage] > common.COMPUTE_PRICE
            calls += active
        assert np.array_equal(calls, stored["calls"])
    decision = json.loads(decision_path.read_text())
    assert decision["v5_launched"] is False
    assert decision["confirmatory_claim"] is False
    assert decision["validity_checks"]["v3_test_targets_opened"] is False

