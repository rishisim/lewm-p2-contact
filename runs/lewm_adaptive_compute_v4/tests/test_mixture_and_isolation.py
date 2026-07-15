from __future__ import annotations

import numpy as np

import common
import runtime


def test_exact_integer_mixture_hits_conservative_total_and_is_seeded() -> None:
    _, _, policy = runtime.load_discovery_modules()
    means = np.asarray([4.0, 3.0, 2.6, 2.5])
    n = 100
    target = 127
    first = policy.strongest_transition_independent_baseline(
        means, common.SUPPORTED_CALLS, n=n, target_total_calls=target, seed=123
    )
    second = policy.strongest_transition_independent_baseline(
        means, common.SUPPORTED_CALLS, n=n, target_total_calls=target, seed=123
    )
    assert first["audit"]["exact_total_match"]
    assert first["selected_calls"].sum() == target
    assert np.array_equal(first["selected_calls"], second["selected_calls"])


def test_analytic_mixture_has_exact_fractional_compute() -> None:
    _, _, policy = runtime.load_discovery_modules()
    means = np.asarray([4.0, 3.0, 2.6, 2.5])
    result = policy.optimal_expected_mixture(means, common.SUPPORTED_CALLS, 1.234567)
    assert np.isclose(result["probabilities"] @ common.SUPPORTED_CALLS, 1.234567, atol=1e-10)


def test_frozen_seed_roles_are_disjoint_when_manifest_exists() -> None:
    path = common.ROOT / "seed_manifest.json"
    if not path.exists():
        return
    payload = common.read_json(path)
    roles = {}
    for role in ("smoke", "confirmation"):
        roles[role] = {
            value
            for item in payload["roles"][role]["episodes"]
            for value in item["primary"].values()
        }
    assert roles["smoke"].isdisjoint(roles["confirmation"])
    assert len(payload["roles"]["smoke"]["episodes"]) == 12
    assert len(payload["roles"]["confirmation"]["episodes"]) == 300


def test_v3_test_cache_is_never_a_v4_input() -> None:
    for path in common.ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "runs/lewm_adaptive_compute_v3/cache/cube_inputs.npz" not in text
