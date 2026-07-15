#!/usr/bin/env python3
"""Fold fit-only normalization into the frozen stage-specific dual heads."""

from __future__ import annotations

import json

import numpy as np

from cycle_common import FEATURE_DIM, ROOT, atomic_npz, sha256_file


def main() -> None:
    source = ROOT / "freeze/gate_fit.npz"
    output = ROOT / "freeze/compiled_gate.npz"
    if output.exists():
        raise RuntimeError("compiled gate already exists")
    with np.load(source, allow_pickle=False) as stored:
        mean = stored["feature_mean"].astype(np.float64)
        standard_deviation = stored["feature_std"].astype(np.float64)
        raw_weight = stored["wr"].astype(np.float64)
        white_weight = stored["ww"].astype(np.float64)
        thresholds = stored["thresholds"].astype(np.float64)
    expected = (3, FEATURE_DIM)
    if any(value.shape != expected for value in (mean, standard_deviation, raw_weight, white_weight)):
        raise RuntimeError("unexpected stage-specific gate shape")
    if thresholds.shape != (3,) or not np.isfinite(thresholds).all():
        raise RuntimeError("invalid stage-specific thresholds")
    if np.any(standard_deviation <= 0) or not np.isfinite(standard_deviation).all():
        raise RuntimeError("invalid fit-only feature standard deviation")
    raw_affine = (raw_weight / standard_deviation).astype(np.float32)
    white_affine = (white_weight / standard_deviation).astype(np.float32)
    centered = mean / standard_deviation
    raw_bias = (-np.einsum("sd,sd->s", centered, raw_weight, optimize=False)).astype(np.float32)
    white_bias = (-np.einsum("sd,sd->s", centered, white_weight, optimize=False)).astype(np.float32)
    if not all(
        np.isfinite(value).all()
        for value in (raw_affine, white_affine, raw_bias, white_bias, thresholds)
    ):
        raise RuntimeError("compiled gate contains nonfinite values")
    atomic_npz(
        output,
        {
            "a_raw": raw_affine,
            "b_raw": raw_bias,
            "a_white": white_affine,
            "b_white": white_bias,
            "thresholds": thresholds,
        },
    )
    print(
        json.dumps(
            {
                "compiled_gate_sha256": sha256_file(output),
                "gate_width": FEATURE_DIM,
                "stage_count": 3,
                "thresholds": thresholds.tolist(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
