#!/usr/bin/env python3
"""Fail-closed verification of the marginal-value exploratory artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
NUMERIC = ROOT / "numeric"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(name: str) -> list[dict[str, str]]:
    with (NUMERIC / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    audit = json.loads((NUMERIC / "independent_audit.json").read_text())
    cube_source = REPO / "runs/lewm_v5_readiness_program/v5_package_versions/v004/data/v5_confirmation_execution.npz"
    cube_manifest = json.loads((cube_source.with_name("v5_confirmation_execution_manifest.json")).read_text())
    pilot_source = REPO / "runs/lewm_pusht_replication_pilot/EVALUATION_ARRAYS.npz"
    binary_source = REPO / "runs/lewm_pusht_binary_confirmation/EVALUATION_ARRAYS.npz"
    pilot_hashes = json.loads((pilot_source.parent / "ARTIFACT_HASHES.json").read_text())
    binary_hashes = json.loads((binary_source.parent / "ARTIFACT_HASHES.json").read_text())

    checks: dict[str, bool] = {}
    checks["audit_validation_all_true"] = all(audit["validation"].values())
    checks["cube_source_hash_manifest"] = file_hash(cube_source) == cube_manifest["sha256"] == audit["cube"]["source_sha256"]
    checks["pilot_source_hash_manifest"] = file_hash(pilot_source) == pilot_hashes["files"]["EVALUATION_ARRAYS.npz"] == audit["pusht_four_depth_pilot"]["source_sha256"]
    checks["binary_source_hash_manifest"] = file_hash(binary_source) == binary_hashes["files"]["EVALUATION_ARRAYS.npz"] == audit["pusht_binary"]["source_sha256"]

    cube = rows("cube_frontier.csv")
    binary = rows("pusht_binary_frontier.csv")
    checks["cube_grid_exact"] = [float(row["retention_fraction"]) for row in cube] == [0, .125, .25, .375, .5, .625, .75, .875, 1]
    checks["cube_frozen_unique"] = sum(row["frozen_operating_point"] == "True" for row in cube) == 1
    checks["cube_both_endpoints_positive_at_eight_nonzero_settings"] = sum(
        float(row["raw_benefit_vs_ti_envelope"]) > 0 and float(row["whitened_benefit_vs_ti_envelope"]) > 0 for row in cube
    ) == 8
    frozen_cube = next(row for row in cube if row["frozen_operating_point"] == "True")
    checks["cube_frozen_benefit_recomputed"] = close(float(frozen_cube["raw_benefit_vs_ti_envelope"]), 6.7690034329932584e-6) and close(
        float(frozen_cube["whitened_benefit_vs_ti_envelope"]), 0.000608622054686106
    )

    checks["binary_frozen_unique"] = sum(row["frozen_operating_point"] == "True" for row in binary) == 1
    frozen_binary = next(row for row in binary if row["frozen_operating_point"] == "True")
    checks["binary_frozen_k"] = int(frozen_binary["optional_depth2_rows"]) == 1055
    checks["binary_frozen_benefits"] = close(float(frozen_binary["raw_benefit_vs_ti_envelope"]), 0.002733782040519925) and close(
        float(frozen_binary["whitened_benefit_vs_ti_envelope"]), 0.0015692368266779222
    )
    checks["binary_both_positive_10_through_90_percent"] = all(
        float(row["raw_benefit_vs_ti_envelope"]) > 0 and float(row["whitened_benefit_vs_ti_envelope"]) > 0
        for row in binary if 0.1 <= float(row["allocation_fraction"]) <= 0.9
    )
    checks["binary_oracle_capture_frozen"] = close(
        float(frozen_binary["raw_fraction_oracle_allocation_uplift_captured"]), 0.6199412926137653
    ) and close(float(frozen_binary["whitened_fraction_oracle_allocation_uplift_captured"]), 0.529851003141007)

    with np.load(NUMERIC / "cube_frontier_episode_arrays.npz", allow_pickle=False) as stored:
        checks["cube_lossless_frontier_shape"] = stored["learned_raw"].shape == (9, 1600) and stored["ti_whitened"].shape == (9, 1600)
    with np.load(NUMERIC / "pusht_binary_frontier_episode_arrays.npz", allow_pickle=False) as stored:
        checks["binary_lossless_frontier_shape"] = stored["learned_raw"].shape == (len(binary), 240) and stored["oracle_whitened"].shape == (len(binary), 240)
    with np.load(NUMERIC / "finite_schedule_arrays.npz", allow_pickle=False) as stored:
        checks["finite_schedules_lossless_and_finite"] = len(stored.files) == 24 and all(np.isfinite(stored[key]).all() for key in stored.files)

    for stem in ("quality_compute_frontiers", "allocation_and_calibration", "gate_overhead_sensitivity"):
        png = ROOT / "figures" / f"{stem}.png"
        pdf = ROOT / "figures" / f"{stem}.pdf"
        with Image.open(png) as image:
            checks[f"{stem}_render_dimensions"] = image.width >= 1600 and image.height >= 700
        checks[f"{stem}_pdf_nonempty"] = pdf.stat().st_size > 10_000

    required = (
        "cube_frontier.csv", "pusht_binary_frontier.csv", "fixed_depth_points.csv",
        "calibration_deciles.csv", "rank_and_calibration_summary.csv", "oracle_headroom.csv",
        "gate_overhead_sensitivity.csv", "finite_integer_scheduling.csv",
        "pusht_four_depth_pilot_point.json", "independent_audit.json",
    )
    checks["required_numeric_artifacts_present"] = all((NUMERIC / name).is_file() for name in required)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"validation failed: {failed}")
    report = {"status": "all_checks_passed", "check_count": len(checks), "checks": checks}
    (NUMERIC / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
