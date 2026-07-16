#!/usr/bin/env python3
"""Derivation A: sum the nodes emitted by the executable feature graph."""

from __future__ import annotations

import json

from counted_features import executable_operation_graph
from cycle_common import ROOT, atomic_json


def main() -> None:
    graph = executable_operation_graph()
    totals = graph["totals"]
    if totals != {
        "feature_flops_per_reached_evaluation": 3801,
        "dual_affine_score_flops_per_reached_evaluation": 4184,
        "total_flops_per_reached_evaluation": 7985,
        "nonflop_comparison_min_per_reached_evaluation": 5,
    }:
        raise RuntimeError(f"unexpected executable graph totals: {totals}")
    atomic_json(ROOT / "operation_graph.json", graph, exclusive=True)
    result = {
        "schema_version": 1,
        "method": "independent sum over executable primitive nodes",
        "totals": totals,
        "passed": True,
    }
    atomic_json(ROOT / "audit/flop_derivation_graph.json", result, exclusive=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
