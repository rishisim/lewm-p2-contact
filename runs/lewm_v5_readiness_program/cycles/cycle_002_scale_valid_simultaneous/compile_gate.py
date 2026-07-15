#!/usr/bin/env python3
"""Fold the frozen normalization into the two frozen affine gate heads."""

from __future__ import annotations

import json

import numpy as np

from cycle_common import ROOT, SOURCE_DISCOVERY, THRESHOLD, atomic_npz, sha256_file


def main() -> None:
    output = ROOT / "freeze/compiled_gate.npz"
    if output.exists():
        raise RuntimeError("compiled gate already exists")
    contract_path = SOURCE_DISCOVERY / "freeze/gate_contract.npz"
    weight_path = SOURCE_DISCOVERY / "freeze/gate_weights.npz"
    with np.load(contract_path, allow_pickle=False) as contract, np.load(
        weight_path, allow_pickle=False
    ) as weights:
        mean = contract["feature_mean"].astype(np.float64)
        standard_deviation = contract["feature_std"].astype(np.float64)
        raw_weight = weights["wr"].astype(np.float64)
        white_weight = weights["ww"].astype(np.float64)
    if mean.shape != standard_deviation.shape or mean.shape != (1049,):
        raise RuntimeError("unexpected frozen gate contract shape")
    if np.any(standard_deviation <= 0) or not np.isfinite(standard_deviation).all():
        raise RuntimeError("invalid frozen feature standard deviation")
    raw_affine = (raw_weight / standard_deviation).astype(np.float32)
    white_affine = (white_weight / standard_deviation).astype(np.float32)
    centered = mean / standard_deviation
    # Avoid the local Accelerate/BLAS dot path, which can emit spurious
    # nonfinite values for this ill-conditioned vector.  The explicit scalar
    # einsum is the finite reduction path already used elsewhere in the repo.
    raw_bias = np.float32(-np.einsum("i,i->", centered, raw_weight, optimize=False))
    white_bias = np.float32(-np.einsum("i,i->", centered, white_weight, optimize=False))
    if not all(
        np.isfinite(value).all()
        for value in (raw_affine, white_affine, raw_bias, white_bias)
    ):
        raise RuntimeError("compiled gate contains nonfinite values")
    atomic_npz(
        output,
        {
            "a_raw": raw_affine,
            "b_raw": raw_bias,
            "a_white": white_affine,
            "b_white": white_bias,
            "threshold": np.float64(THRESHOLD),
        },
    )
    print(
        json.dumps(
            {
                "compiled_gate_sha256": sha256_file(output),
                "gate_width": len(raw_affine),
                "threshold": THRESHOLD,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
