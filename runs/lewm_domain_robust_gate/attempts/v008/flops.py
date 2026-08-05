#!/usr/bin/env python3
"""Exact counted-compute contract for the domain-robust gate study."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any, Iterable

import numpy as np

from counted_features import executable_operation_graph


BASE_FLOPS = 70_529_190
DEPTH1_FLOPS = 669_184
ADAPTER_FLOPS = 264_960
FEATURE_FLOPS = 3_801
AFFINE_HEAD_FLOPS = 2_092

POOLED_HEAD_COUNT = 2
ENVELOPE_HEAD_COUNT = 8
POOLED_GATE_FLOPS = 7_985
ENVELOPE_GATE_FLOPS = 20_537
POOLED_GATE_NONFLOPS = 5
ENVELOPE_GATE_NONFLOPS = 11


def _validate_head_count(head_count: int) -> int:
    value = int(head_count)
    if value not in (POOLED_HEAD_COUNT, ENVELOPE_HEAD_COUNT):
        raise ValueError("head_count must be 2 (pooled) or 8 (DGP envelope)")
    return value


def gate_cost(head_count: int) -> dict[str, int]:
    """Return exact per-reached-row gate FLOPs and non-FLOP operations."""

    heads = _validate_head_count(head_count)
    graph = executable_operation_graph(head_count=heads)
    totals = graph["totals"]
    result = {
        "head_count": heads,
        "feature_flops": int(totals["feature_flops_per_reached_evaluation"]),
        "affine_head_flops_each": int(
            totals["affine_score_flops_per_head_per_reached_evaluation"]
        ),
        "all_head_flops": int(totals["all_affine_score_flops_per_reached_evaluation"]),
        "total_gate_flops": int(totals["total_flops_per_reached_evaluation"]),
        "nonflop_operations": int(totals["nonflop_operations_per_reached_evaluation"]),
    }
    expected = {
        POOLED_HEAD_COUNT: {
            "head_count": 2,
            "feature_flops": FEATURE_FLOPS,
            "affine_head_flops_each": AFFINE_HEAD_FLOPS,
            "all_head_flops": 4_184,
            "total_gate_flops": POOLED_GATE_FLOPS,
            "nonflop_operations": POOLED_GATE_NONFLOPS,
        },
        ENVELOPE_HEAD_COUNT: {
            "head_count": 8,
            "feature_flops": FEATURE_FLOPS,
            "affine_head_flops_each": AFFINE_HEAD_FLOPS,
            "all_head_flops": 16_736,
            "total_gate_flops": ENVELOPE_GATE_FLOPS,
            "nonflop_operations": ENVELOPE_GATE_NONFLOPS,
        },
    }[heads]
    if result != expected:
        raise RuntimeError(f"executable operation graph cost drift: {result}")
    return result


def _validated_calls(calls: Iterable[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(calls)
    if values.ndim != 1 or not len(values):
        raise ValueError("calls must be a nonempty rank-one vector")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("calls must contain exact integers")
    values = values.astype(np.int64, copy=False)
    if ((values < 1) | (values > 4)).any():
        raise ValueError("every row must use one through four solver calls")
    return values


def reached_gate_evaluations(calls: Iterable[int] | np.ndarray) -> int:
    """Count reached gate stages under sequential early exit."""

    values = _validated_calls(calls)
    return int(np.minimum(values, 3).sum(dtype=np.int64))


def exact_compute(calls: Iterable[int] | np.ndarray, head_count: int) -> dict[str, Any]:
    """Price one execution vector using the immutable historical convention."""

    values = _validated_calls(calls)
    cost = gate_cost(head_count)
    rows = int(len(values))
    solver_calls = int(values.sum(dtype=np.int64))
    later_adapter_calls = solver_calls - rows
    gate_evaluations = reached_gate_evaluations(values)
    base_flops = rows * BASE_FLOPS
    depth1_flops = rows * DEPTH1_FLOPS
    adapter_flops = later_adapter_calls * ADAPTER_FLOPS
    feature_flops = gate_evaluations * FEATURE_FLOPS
    head_flops = gate_evaluations * cost["all_head_flops"]
    gate_flops = feature_flops + head_flops
    total = base_flops + depth1_flops + adapter_flops + gate_flops
    equivalent = Fraction(gate_flops, ADAPTER_FLOPS)
    return {
        "row_count": rows,
        "base_calls": rows,
        "solver_calls": solver_calls,
        "mandatory_depth1_calls": rows,
        "later_adapter_calls": later_adapter_calls,
        "gate_evaluations": gate_evaluations,
        "head_count": int(head_count),
        "base_flops": base_flops,
        "depth1_flops": depth1_flops,
        "adapter_flops": adapter_flops,
        "gate_feature_flops": feature_flops,
        "gate_head_flops": head_flops,
        "gate_total_flops": gate_flops,
        "total_counted_flops": total,
        "gate_nonflop_operations": gate_evaluations * cost["nonflop_operations"],
        "exact_equivalent_transition_independent_solver_calls_numerator": (
            solver_calls * equivalent.denominator + equivalent.numerator
        ),
        "exact_equivalent_transition_independent_solver_calls_denominator": (
            equivalent.denominator
        ),
        "exact_equivalent_transition_independent_solver_calls": (
            float(solver_calls + equivalent)
        ),
        "historical_counting_convention": {
            "base_per_row": BASE_FLOPS,
            "mandatory_depth1_per_row": DEPTH1_FLOPS,
            "each_later_adapter": ADAPTER_FLOPS,
            "feature_per_reached_gate": FEATURE_FLOPS,
            "affine_head_per_reached_gate": AFFINE_HEAD_FLOPS,
            "activations_and_normalizations_inside_frozen_solver_omitted_as_in_V5": True,
        },
    }


def derive_exact_costs() -> dict[str, Any]:
    """Expose both independent symbolic totals and executable-graph totals."""

    latent = 192
    action = 25
    changes = 2
    feature_symbolic = (
        latent
        + 4 * (latent + (latent - 1) + 1)
        + 1
        + 2 * (latent + (latent - 1) + 1 + 1)
        + changes * (latent + latent + (latent - 1) + 1)
        + changes * (action + action + (action - 1) + 1)
    )
    affine_symbolic = 1046 + 1045 + 1
    if feature_symbolic != FEATURE_FLOPS or affine_symbolic != AFFINE_HEAD_FLOPS:
        raise RuntimeError("closed-form gate derivation drift")
    pooled = gate_cost(POOLED_HEAD_COUNT)
    envelope = gate_cost(ENVELOPE_HEAD_COUNT)
    return {
        "schema_version": 1,
        "passed": True,
        "base_flops_per_row": BASE_FLOPS,
        "depth1_flops_per_row": DEPTH1_FLOPS,
        "later_adapter_flops_per_call": ADAPTER_FLOPS,
        "closed_form": {
            "feature_flops": feature_symbolic,
            "affine_head_flops_each": affine_symbolic,
            "feature_nonflop_clamps": 3,
        },
        "pooled_dual": pooled,
        "dgp_envelope": envelope,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", nargs="*", type=int)
    parser.add_argument("--head-count", type=int, choices=(2, 8), default=8)
    arguments = parser.parse_args()
    result = (
        exact_compute(arguments.calls, arguments.head_count)
        if arguments.calls
        else derive_exact_costs()
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
