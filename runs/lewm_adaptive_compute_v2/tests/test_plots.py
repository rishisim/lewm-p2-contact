from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg


MODULE_PATH = Path(__file__).resolve().parents[1] / "plots.py"
SPEC = importlib.util.spec_from_file_location("lewm_v2_plots", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def _figure_rows() -> dict[str, list[dict[str, object]]]:
    policies = ["adaptive", "uniform_d2", "fixed_d1", "random", "permutation", "oracle"]
    policy_rows = [
        {
            "policy": policy,
            "raw_mean_loss": 0.0030 + index * 0.00005,
            "whitened_mean_loss": 0.98 + index * 0.01,
        }
        for index, policy in enumerate(policies)
    ]
    comparison_rows = []
    for index, comparison in enumerate(
        ("uniform_d2 - adaptive", "fixed_d1 - adaptive", "random - adaptive", "adaptive - oracle")
    ):
        raw = (index + 1) * 1e-5
        white = (index - 1) * 0.002
        comparison_rows.append(
            {
                "comparison": comparison,
                "raw_mean_difference": raw,
                "raw_ci_lower": raw - 5e-6,
                "raw_ci_upper": raw + 5e-6,
                "whitened_mean_difference": white,
                "whitened_ci_lower": white - 0.001,
                "whitened_ci_upper": white + 0.001,
            }
        )
    calibration_rows = [
        {"benefit": "b12", "pearson": 0.2, "spearman": 0.25, "rmse": 0.8, "mae": 0.6},
        {"benefit": "b14", "pearson": 0.3, "spearman": 0.35, "rmse": 0.9, "mae": 0.7},
    ]
    quantile_rows = [
        {
            "benefit": benefit,
            "quantile": quantile,
            "mean_predicted": -0.3 + quantile * 0.1 + benefit_index * 0.01,
            "mean_actual": -1e-5 + quantile * 5e-6 + benefit_index * 1e-6,
            "count": 684,
        }
        for benefit_index, benefit in enumerate(("b12", "b14"))
        for quantile in range(1, 6)
    ]
    allocation_counts = {
        "adaptive": (1500, 960, 960),
        "uniform_d2": (0, 3420, 0),
        "fixed_d1": (3420, 0, 0),
        "random": (1500, 960, 960),
        "permutation": (1500, 960, 960),
        "oracle": (1500, 960, 960),
    }
    allocation_rows = []
    for policy, counts in allocation_counts.items():
        calls = sum(depth * count for depth, count in zip(plots.DEPTHS, counts))
        allocation_rows.extend(
            {"policy": policy, "depth": depth, "count": count, "total_calls": calls}
            for depth, count in zip(plots.DEPTHS, counts)
        )
    regime_rows = []
    for regime_index, regime in enumerate(("impact", "contact", "transport_free", "static")):
        counts = (34, 33, 33)
        regime_rows.extend(
            {
                "regime": regime,
                "depth": depth,
                "count": count,
                "fraction": count / 100,
                "raw_benefit_vs_uniform": (regime_index - 1) * 1e-5,
                "n": 100,
            }
            for depth, count in zip(plots.DEPTHS, counts)
        )
    return {
        "policy_rows": policy_rows,
        "comparison_rows": comparison_rows,
        "calibration_rows": calibration_rows,
        "quantile_rows": quantile_rows,
        "allocation_rows": allocation_rows,
        "regime_rows": regime_rows,
    }


class PlotTests(unittest.TestCase):
    def test_source_integrity_is_stable_order_sensitive_and_strict(self) -> None:
        rows = [{"a": 1, "b": 2.5}, {"a": 2, "b": 3.5}]
        first = plots.source_data_integrity(rows)
        second = plots.source_data_integrity(rows)
        self.assertEqual(first, second)
        self.assertEqual(first["row_count"], 2)
        self.assertEqual(first["columns"], ["a", "b"])
        self.assertEqual(len(first["sha256"]), 64)
        self.assertNotEqual(plots.source_data_integrity(list(reversed(rows)))["sha256"], first["sha256"])
        with self.assertRaisesRegex(ValueError, "finite"):
            plots.source_data_integrity([{"a": float("nan")}])

    def test_write_all_figures_decodes_pngs_and_records_dimensions(self) -> None:
        rows = _figure_rows()
        with tempfile.TemporaryDirectory() as directory:
            result = plots.write_all_figures(**rows, output_dir=Path(directory))
            self.assertEqual(
                set(result["paths"]),
                {
                    "policy_performance",
                    "gate_diagnostics",
                    "exact_allocation",
                    "posthoc_regimes",
                },
            )
            for name, value in result["paths"].items():
                path = Path(value)
                image = mpimg.imread(path)
                self.assertGreaterEqual(image.shape[0], 400, name)
                self.assertGreaterEqual(image.shape[1], 600, name)
                self.assertEqual(result["pngs"][name]["width_px"], image.shape[1])
                self.assertEqual(result["pngs"][name]["height_px"], image.shape[0])
                self.assertEqual(
                    result["pngs"][name]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
                )

            manifest_path = Path(result["manifest"])

            def reject_constant(value: str) -> None:
                raise AssertionError(f"Nonfinite JSON constant: {value}")

            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8"), parse_constant=reject_constant
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["sources"], result["sources"])
            self.assertEqual(set(manifest["figures"]), set(result["paths"]))

    def test_allocation_figure_rejects_inexact_call_accounting(self) -> None:
        rows = _figure_rows()["allocation_rows"]
        rows[0] = {**rows[0], "total_calls": int(rows[0]["total_calls"]) + 1}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "total_calls"):
                plots.plot_exact_allocation(rows, Path(directory) / "bad.png")

    def test_gate_quantiles_must_be_score_ordered(self) -> None:
        rows = _figure_rows()
        rows["quantile_rows"][1]["mean_predicted"] = -99.0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "nondecreasing"):
                plots.plot_gate_diagnostics(
                    rows["calibration_rows"],
                    rows["quantile_rows"],
                    Path(directory) / "bad.png",
                )


if __name__ == "__main__":
    unittest.main()
