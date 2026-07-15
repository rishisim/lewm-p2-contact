#!/usr/bin/env python3
"""Derivation B: independent closed-form count with no graph-code import."""

from __future__ import annotations

import json

from cycle_common import ROOT, atomic_json


def main() -> None:
    latent = 192
    action = 25
    history = 3
    changes = history - 1
    gap_subtractions = latent
    four_reused_norms = 4 * (latent + (latent - 1) + 1)
    relative_ratio = 1
    two_cosines = 2 * (latent + (latent - 1) + 1 + 1)
    history_change_norms = changes * (latent + latent + (latent - 1) + 1)
    action_change_norms = changes * (action + action + (action - 1) + 1)
    feature = (
        gap_subtractions
        + four_reused_norms
        + relative_ratio
        + two_cosines
        + history_change_norms
        + action_change_norms
    )
    affine = 2 * (1046 + 1045 + 1)
    nonflop = 3 + 1 + 1
    totals = {
        "feature_flops_per_reached_evaluation": feature,
        "dual_affine_score_flops_per_reached_evaluation": affine,
        "total_flops_per_reached_evaluation": feature + affine,
        "nonflop_comparison_min_per_reached_evaluation": nonflop,
    }
    if totals != {
        "feature_flops_per_reached_evaluation": 3801,
        "dual_affine_score_flops_per_reached_evaluation": 4184,
        "total_flops_per_reached_evaluation": 7985,
        "nonflop_comparison_min_per_reached_evaluation": 5,
    }:
        raise RuntimeError(f"unexpected symbolic totals: {totals}")
    result = {
        "schema_version": 1,
        "method": "closed-form count independently transcribed from the mathematical feature specification",
        "terms": {
            "gap_subtractions": gap_subtractions,
            "four_reused_norms": four_reused_norms,
            "relative_ratio": relative_ratio,
            "two_cosines": two_cosines,
            "history_change_norms": history_change_norms,
            "action_change_norms": action_change_norms,
            "two_affine_heads": affine,
            "three_clamps_plus_head_min_plus_threshold_comparison": nonflop,
        },
        "totals": totals,
        "passed": True,
    }
    atomic_json(ROOT / "audit/flop_derivation_symbolic.json", result, exclusive=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
