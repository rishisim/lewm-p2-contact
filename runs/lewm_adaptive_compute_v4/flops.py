#!/usr/bin/env python3
"""First-principles FLOP accounting under the frozen historical convention."""

from __future__ import annotations

import json

import common


def recompute() -> dict[str, object]:
    # Stagewise later adapter input: 3*192 latent history + 3*25 action
    # history + 192 preceding exit = 843.  The hidden width is 128 and the
    # output is 192.  Multiply/add pairs are counted as two FLOPs; norms and
    # activations in the solver are omitted consistently with the immutable
    # discovery convention.
    adapter = 2 * (843 * 128 + 128 * 192)
    # Frozen full-feature linear gate: form d1 update, compute the exact 11-tail
    # summaries, normalize 1046 features + 3 depth coordinates, apply the
    # 1049-to-1 affine head, and make one stopping comparison.
    feature_update = 192
    feature_summaries = 3806
    feature = feature_update + feature_summaries
    normalization = 2 * (common.FEATURE_DIM + 3)
    linear = 2 * (common.FEATURE_DIM + 3) + 1
    control = 1
    gate = feature + normalization + linear + control
    values = {
        "schema_version": 1,
        "convention": (
            "historical LeWM discovery arithmetic convention: multiply/add pair=2; "
            "common base and V1 anchors retained; solver norms/activations omitted; gate "
            "summary/normalization/head/control fully incremental-counted"
        ),
        "common_base_prediction_flops": common.BASE_PREDICT_FLOPS,
        "v1_call_flops": common.V1_CALL_FLOPS,
        "later_adapter": {
            "input_dim": 843,
            "hidden_dim": 128,
            "output_dim": 192,
            "formula": "2 * (843*128 + 128*192)",
            "flops": adapter,
        },
        "gate_per_evaluated_decision": {
            "d1_update_arithmetic": feature_update,
            "summary_arithmetic": feature_summaries,
            "feature_construction_flops": feature,
            "normalization_flops": normalization,
            "linear_head_flops": linear,
            "control_flow_flops": control,
            "total_flops": gate,
            "parameters": (common.FEATURE_DIM + 3) + 1,
        },
        "fixed_depth_total_flops": {
            f"d{depth}": common.BASE_PREDICT_FLOPS
            + common.V1_CALL_FLOPS
            + (depth - 1) * adapter
            for depth in range(1, 5)
        },
        "checks": {
            "adapter_matches_frozen": adapter == common.STAGE_ADAPTER_FLOPS,
            "gate_matches_frozen": gate == common.GATE_FLOPS_PER_DECISION,
            "gate_parameters_match_frozen": (common.FEATURE_DIM + 3) + 1
            == common.GATE_PARAMETERS,
        },
    }
    values["passed"] = all(values["checks"].values())
    if not values["passed"]:
        raise RuntimeError(f"FLOP recomputation failed: {values}")
    return values


if __name__ == "__main__":
    result = recompute()
    common.write_json(common.ROOT / "metrics/flops.json", result)
    print(json.dumps({"passed": result["passed"]}, sort_keys=True))
