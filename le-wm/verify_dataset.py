#!/usr/bin/env python3
"""Verify stable-worldmodel datasets and optional source/destination parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm


INDEX_CANDIDATES = ("episode_idx", "ep_idx")
IMAGE_COLUMN_HINTS = ("pixels", "image", "frame", "goal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset name or path accepted by swm.data.load_dataset.")
    parser.add_argument("--reference", help="Optional reference dataset for parity checks.")
    parser.add_argument("--cache-dir", help="Override STABLEWM_HOME for dataset resolution.")
    parser.add_argument("--expected-rows", type=int, help="Expected row count with num_steps=1.")
    parser.add_argument("--expected-episodes", type=int, help="Expected episode count.")
    parser.add_argument(
        "--required-column",
        action="append",
        default=[],
        help="Column that must be present or index-accessible; repeat as needed.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        action="append",
        default=[],
        help="Flat row index to sample for parity; defaults to first/middle/last.",
    )
    parser.add_argument(
        "--compare-column",
        action="append",
        default=[],
        help="Tabular column to compare against --reference; defaults to common non-image columns.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    return parser.parse_args()


def load_dataset(name: str, cache_dir: str | None):
    return swm.data.load_dataset(name, cache_dir=cache_dir, frameskip=1, num_steps=1)


def dataset_summary(name: str, dataset) -> dict[str, Any]:
    return {
        "name": name,
        "format": type(dataset).__name__,
        "rows": int(len(dataset)),
        "episodes": int(len(dataset.lengths)),
        "columns": list(dataset.column_names),
        "index_columns": available_index_columns(dataset),
    }


def available_index_columns(dataset) -> list[str]:
    found = []
    for column in (*INDEX_CANDIDATES, "step_idx"):
        try:
            dataset.get_col_data(column)
            found.append(column)
        except (KeyError, ValueError):
            continue
    return found


def has_column(dataset, column: str) -> bool:
    if column in dataset.column_names:
        return True
    try:
        dataset.get_col_data(column)
        return True
    except (KeyError, ValueError):
        return False


def default_sample_indices(rows: int) -> list[int]:
    if rows <= 0:
        return []
    values = [0, rows // 2, rows - 1]
    return sorted(set(int(v) for v in values))


def is_image_column(name: str, value: Any) -> bool:
    lowered = name.lower()
    if any(hint in lowered for hint in IMAGE_COLUMN_HINTS):
        return True
    arr = np.asarray(value)
    return arr.ndim >= 3


def sample_rows(dataset, indices: list[int]) -> dict[str, Any]:
    if not indices:
        return {}
    rows = dataset.get_row_data(indices)
    out: dict[str, Any] = {}
    for key, value in rows.items():
        out[key] = np.asarray(value)
    for column in (*INDEX_CANDIDATES, "step_idx"):
        if column not in out:
            try:
                out[column] = np.asarray(dataset.get_col_data(column))[indices]
            except (KeyError, ValueError):
                continue
    return out


def compare_values(
    column: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if is_image_column(column, left) or is_image_column(column, right):
        return {
            "column": column,
            "status": "skipped",
            "reason": "image column; use bounded pixel/model-output parity instead of byte equality",
        }
    if left.shape != right.shape:
        return {
            "column": column,
            "status": "failed",
            "reason": f"shape mismatch {left.shape} != {right.shape}",
        }
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        left_num = np.nan_to_num(left.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        right_num = np.nan_to_num(right.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = float(np.max(np.abs(left_num - right_num))) if left_num.size else 0.0
        ok = bool(np.allclose(left_num, right_num, atol=atol, rtol=rtol))
        return {
            "column": column,
            "status": "passed" if ok else "failed",
            "max_abs_diff": max_abs,
            "atol": atol,
            "rtol": rtol,
        }
    ok = bool(np.array_equal(left, right))
    return {"column": column, "status": "passed" if ok else "failed"}


def parity_report(dataset, reference, sample_indices, compare_columns, atol, rtol) -> dict[str, Any]:
    left_rows = sample_rows(dataset, sample_indices)
    right_rows = sample_rows(reference, sample_indices)
    if compare_columns:
        columns = compare_columns
    else:
        columns = sorted(set(left_rows) & set(right_rows) - {"episode_idx", "ep_idx", "step_idx"})
    checks = []
    for column in columns:
        if column not in left_rows or column not in right_rows:
            checks.append({"column": column, "status": "failed", "reason": "missing from sampled rows"})
            continue
        checks.append(compare_values(column, left_rows[column], right_rows[column], atol=atol, rtol=rtol))
    return {"sample_indices": sample_indices, "checks": checks}


def write_reports(report: dict[str, Any], json_path: Path | None, md_path: Path | None) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Dataset Verification",
            "",
            f"- Dataset: `{report['dataset']['name']}`",
            f"- Format: `{report['dataset']['format']}`",
            f"- Rows: `{report['dataset']['rows']}`",
            f"- Episodes: `{report['dataset']['episodes']}`",
            f"- Required columns passed: `{report['required_columns_passed']}`",
        ]
        if report.get("reference"):
            lines.extend(
                [
                    f"- Reference: `{report['reference']['name']}`",
                    f"- Reference rows: `{report['reference']['rows']}`",
                    f"- Reference episodes: `{report['reference']['episodes']}`",
                ]
            )
        if report.get("parity"):
            lines.extend(["", "## Sampled Parity"])
            for check in report["parity"]["checks"]:
                detail = check.get("reason") or f"max_abs_diff={check.get('max_abs_diff', 0.0)}"
                lines.append(f"- `{check['column']}`: {check['status']} ({detail})")
        md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.cache_dir)
    summary = dataset_summary(args.dataset, dataset)
    failures = []

    if args.expected_rows is not None and summary["rows"] != args.expected_rows:
        failures.append(f"row count {summary['rows']} != expected {args.expected_rows}")
    if args.expected_episodes is not None and summary["episodes"] != args.expected_episodes:
        failures.append(f"episode count {summary['episodes']} != expected {args.expected_episodes}")

    missing_required = [column for column in args.required_column if not has_column(dataset, column)]
    if missing_required:
        failures.append(f"missing required columns: {missing_required}")

    report: dict[str, Any] = {
        "dataset": summary,
        "required_columns": args.required_column,
        "required_columns_passed": not missing_required,
        "image_parity_policy": "Skip byte equality for image columns; use bounded pixel/model-output parity.",
    }

    if args.reference:
        reference = load_dataset(args.reference, args.cache_dir)
        ref_summary = dataset_summary(args.reference, reference)
        report["reference"] = ref_summary
        if summary["rows"] != ref_summary["rows"]:
            failures.append(f"row count differs from reference: {summary['rows']} != {ref_summary['rows']}")
        if summary["episodes"] != ref_summary["episodes"]:
            failures.append(
                f"episode count differs from reference: {summary['episodes']} != {ref_summary['episodes']}"
            )
        sample_indices = args.sample_index or default_sample_indices(min(summary["rows"], ref_summary["rows"]))
        report["parity"] = parity_report(
            dataset,
            reference,
            sample_indices,
            args.compare_column,
            args.atol,
            args.rtol,
        )
        failed_checks = [check for check in report["parity"]["checks"] if check["status"] == "failed"]
        if failed_checks:
            failures.append(f"sampled parity failures: {[check['column'] for check in failed_checks]}")

    report["status"] = "failed" if failures else "passed"
    report["failures"] = failures
    write_reports(report, args.report_json, args.report_md)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
