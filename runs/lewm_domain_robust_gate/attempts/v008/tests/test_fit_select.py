from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ATTEMPT = Path(__file__).resolve().parents[1]
if str(ATTEMPT) not in sys.path:
    sys.path.insert(0, str(ATTEMPT))

import fit_select as subject  # noqa: E402


def synthetic_comparator_seeds() -> dict[str, dict[str, int]]:
    cursor = 1_000
    result: dict[str, dict[str, int]] = {}
    for dgp in subject.DGP_IDS:
        result[dgp] = {
            "raw": cursor,
            "fixed_whitened": cursor + 1,
            "histogram": cursor + 2,
        }
        cursor += 3
    return result


def synthetic_per_dgp(
    seed: int = 11, *, episodes: int = 4, steps: int = 3, latent_dim: int = 4
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, np.ndarray]] = {}
    rows = episodes * steps
    for dgp_index, dgp_id in enumerate(subject.DGP_IDS):
        features = rng.normal(
            loc=0.15 * dgp_index,
            scale=1.0 + 0.05 * dgp_index,
            size=(rows, subject.STAGE_COUNT, subject.FEATURE_DIM),
        )
        target = rng.normal(size=(rows, latent_dim))
        direction = rng.normal(size=(rows, latent_dim))
        direction += 0.15 * features[:, 0, :latent_dim]
        residuals = []
        for depth, scale in enumerate((1.0, 0.82, 0.61, 0.43)):
            stage_signal = 0.025 * features[:, min(depth, 2), :latent_dim]
            residuals.append(scale * direction + stage_signal)
        exits = target[:, None, :] + np.stack(residuals, axis=1)
        result[dgp_id] = {
            "production_features": features.astype(np.float32),
            "target": target.astype(np.float32),
            "exits": exits.astype(np.float32),
            "episode_slot": np.repeat(np.arange(episodes, dtype=np.int64), steps),
            "model_step": np.tile(np.arange(steps, dtype=np.int64), episodes),
            # Permitted causal source tensors may coexist but are never read by
            # the fit/selection criteria once production_features are frozen.
            "history": np.zeros((rows, 3, latent_dim), dtype=np.float32),
            "action_history": np.zeros((rows, 3, 25), dtype=np.float32),
        }
    return result


class FitSelectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.per_dgp = synthetic_per_dgp()
        cls.role = subject.prepare_role(cls.per_dgp)
        cls.fixed_whitening = np.eye(cls.role.latent_dim, dtype=np.float64)
        cls.fitted = subject.fit_candidate_family(cls.role, cls.fixed_whitening)

    def test_role_contract_is_causal_and_fail_closed(self) -> None:
        self.assertEqual(
            self.role.features.shape,
            (4 * 4 * 3, subject.STAGE_COUNT, subject.FEATURE_DIM),
        )
        self.assertEqual(self.role.dgp_ids, subject.DGP_IDS)
        contaminated = synthetic_per_dgp(seed=17)
        contaminated[subject.DGP_IDS[0]]["contact"] = np.zeros(12)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            subject.prepare_role(contaminated)

    def test_bounded_family_has_exactly_24_compiled_candidates(self) -> None:
        fitted = self.fitted
        grid = json.loads((ATTEMPT / "candidate_grid.json").read_text())
        self.assertEqual(
            tuple(item["id"] for item in grid["architectures"]),
            subject.ARCHITECTURES,
        )
        self.assertEqual(tuple(grid["ridge_penalties"]), subject.RIDGES)
        self.assertEqual(
            tuple(grid["sequential_fit_score_quantiles"]), subject.FIT_QUANTILES
        )
        self.assertEqual(len(fitted["candidate_ids"]), 24)
        self.assertEqual(len(set(fitted["candidate_ids"].astype(str))), 24)
        self.assertEqual(
            fitted["compiled_weights"].shape,
            (24, 3, 8, subject.FEATURE_DIM),
        )
        self.assertEqual(fitted["compiled_biases"].shape, (24, 3, 8))
        self.assertEqual(fitted["thresholds"].shape, (24, 3))
        self.assertEqual(fitted["compiled_weights"].dtype, np.dtype(np.float32))
        self.assertEqual(fitted["compiled_biases"].dtype, np.dtype(np.float32))
        self.assertEqual(fitted["thresholds"].dtype, np.dtype(np.float32))
        self.assertTrue(fitted["compiled_weights"].flags.c_contiguous)
        self.assertTrue(fitted["compiled_biases"].flags.c_contiguous)
        self.assertTrue(fitted["thresholds"].flags.c_contiguous)
        self.assertTrue(fitted["fit_runtime_call_equivalence"].all())
        self.assertEqual(fitted["fit_contrast_episode_sd"].shape, (24, 4, 2))
        self.assertEqual(
            np.bincount(fitted["head_count"], minlength=9)[[2, 8]].tolist(),
            [12, 12],
        )
        self.assertEqual(
            set(fitted["ridges"].tolist()), set(subject.RIDGES)
        )
        self.assertEqual(
            set(fitted["fit_quantiles"].tolist()), set(subject.FIT_QUANTILES)
        )
        # Quantiles select thresholds only; they cannot refit the heads.
        for start in range(0, 24, 4):
            for index in range(start + 1, start + 4):
                np.testing.assert_array_equal(
                    fitted["compiled_weights"][start],
                    fitted["compiled_weights"][index],
                )
                np.testing.assert_array_equal(
                    fitted["compiled_biases"][start],
                    fitted["compiled_biases"][index],
                )
        np.testing.assert_array_equal(
            fitted["fixed_whitening_matrix"], self.fixed_whitening
        )
        self.assertEqual(fitted["per_dgp_feature_mean"].shape, (4, 3, 1046))
        self.assertEqual(
            fitted["per_dgp_stage_endpoint_gain_std"].shape, (4, 3, 2)
        )

    def test_affine_compilation_matches_normalized_head(self) -> None:
        rng = np.random.default_rng(19)
        mean = rng.normal(size=subject.FEATURE_DIM)
        std = rng.uniform(0.1, 2.0, size=subject.FEATURE_DIM)
        weight = rng.normal(size=subject.FEATURE_DIM)
        bias = -0.37
        raw_weight, raw_bias = subject.compile_affine_head(
            weight, mean, std, normalized_bias=bias
        )
        features = rng.normal(size=(7, subject.FEATURE_DIM))
        normalized_score = ((features - mean) / std) @ weight + bias
        compiled_score = features @ raw_weight + raw_bias
        np.testing.assert_allclose(compiled_score, normalized_score, rtol=2e-13, atol=2e-12)

    def test_sequential_quantile_uses_only_reached_rows(self) -> None:
        # Unequal row counts guard the frozen ordinary reached-row quantile.
        domains = np.repeat(np.arange(4), (3, 5, 7, 9)).astype(np.int64)
        values = np.arange(len(domains), dtype=np.float32)
        scores = np.column_stack((values, values[::-1], np.sin(values)))
        thresholds, calls, reached = subject.sequential_fit_thresholds(
            scores, domains, 0.55
        )
        self.assertEqual(thresholds.dtype, np.dtype(np.float32))
        self.assertEqual(
            thresholds[0],
            np.float32(np.quantile(scores[:, 0], 0.55, method="linear")),
        )
        self.assertEqual(
            thresholds[1],
            np.float32(
                np.quantile(scores[reached[:, 1], 1], 0.55, method="linear")
            ),
        )
        replay_calls, replay_reached = subject.calls_from_scores(scores, thresholds)
        np.testing.assert_array_equal(calls, replay_calls)
        np.testing.assert_array_equal(reached, replay_reached)
        np.testing.assert_array_equal(reached[:, 1], scores[:, 0] > thresholds[0])
        np.testing.assert_array_equal(
            reached[:, 2],
            (scores[:, 0] > thresholds[0]) & (scores[:, 1] > thresholds[1]),
        )

    def test_float32_boundary_ties_stop_and_all_call_depths_reconstruct(self) -> None:
        zero = np.float32(0.0)
        above = np.nextafter(zero, np.float32(np.inf), dtype=np.float32)
        scores = np.asarray(
            [
                [zero, zero, zero],
                [above, zero, zero],
                [above, above, zero],
                [above, above, above],
            ],
            dtype=np.float32,
        )
        thresholds = np.zeros(3, dtype=np.float32)
        audit = subject.audit_dense_sparse_call_equivalence(scores, thresholds)
        np.testing.assert_array_equal(audit["calls"], np.arange(1, 5))
        self.assertTrue(audit["threshold_ties_stop"])
        self.assertEqual(audit["runtime_dtype"], "float32")
        expected_finite = np.asarray(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
                [True, True, True],
            ]
        )
        np.testing.assert_array_equal(
            np.isfinite(audit["sparse_scores"]), expected_finite
        )
        corrupted = audit["sparse_scores"].copy()
        corrupted[0, 1] = zero
        with self.assertRaisesRegex(RuntimeError, "unreached runtime score"):
            subject.calls_from_sparse_runtime_scores(corrupted, thresholds)

    def test_score_path_is_exact_float32_runtime_arithmetic(self) -> None:
        rng = np.random.default_rng(121)
        features = rng.normal(size=(5, 3, subject.FEATURE_DIM)).astype(np.float32)
        weights = rng.normal(size=(3, 2, subject.FEATURE_DIM)).astype(np.float32)
        biases = rng.normal(size=(3, 2)).astype(np.float32)
        scores = subject.score_compiled_gate(features, weights, biases)
        self.assertEqual(scores.dtype, np.dtype(np.float32))
        manual = np.empty((5, 3), dtype=np.float32)
        for stage in range(3):
            heads = (
                np.einsum(
                    "nd,hd->nh",
                    features[:, stage],
                    weights[stage],
                    optimize=False,
                )
                + biases[stage]
            )
            manual[:, stage] = heads.min(axis=1)
        np.testing.assert_array_equal(scores, manual)
        with self.assertRaisesRegex(TypeError, "exactly float32"):
            subject.score_compiled_gate(
                features, weights.astype(np.float64), biases.astype(np.float64)
            )

    def test_ranking_does_not_exclude_negative_scientific_results(self) -> None:
        def row(candidate: str, contrast: float, rho: float, **updates: float):
            value = {
                "candidate_id": candidate,
                "eligible": True,
                "worst_standardized_exact_compute_contrast": contrast,
                "worst_dgp_stage_spearman": rho,
                "worst_flops_per_row": 10.0,
                "head_count": 2,
                "ridge": 1.0,
                "fit_quantile": 0.65,
            }
            value.update(updates)
            return value

        rows = [
            row("negative_but_best", -0.1, 0.2),
            row("more_negative", -0.3, 0.9),
            row("mechanically_invalid_positive", 4.0, 1.0, eligible=False),
        ]
        ranked = subject.rank_eligible_candidates(rows)
        self.assertEqual([item["candidate_id"] for item in ranked], [
            "negative_but_best", "more_negative"
        ])

    def test_selection_reports_all_dgps_endpoints_and_stages(self) -> None:
        ledger = subject.evaluate_selection(
            self.fitted,
            self.role,
            comparator_seeds=synthetic_comparator_seeds(),
        )
        self.assertEqual(ledger["candidate_count"], 24)
        self.assertEqual(len(ledger["candidates"]), 24)
        self.assertTrue(ledger["all_24_selection_call_traces_reconstruct_exactly"])
        self.assertEqual(ledger["selection_runtime_dtype"], "float32")
        first = ledger["candidates"][0]
        self.assertTrue(first["eligibility_is_mechanical_only"])
        self.assertEqual(len(first["dgps"]), 4)
        self.assertEqual(
            sum(len(record["exact_compute_contrast"]) for record in first["dgps"]),
            8,
        )
        self.assertEqual(
            sum(len(record["stagewise_rank"]) for record in first["dgps"]),
            12,
        )
        self.assertFalse(ledger["robust_fit_whitening_used_for_selection"])
        fit_power, selection_power = subject.selected_power_summaries(
            self.fitted, ledger
        )
        self.assertEqual(set(fit_power["claims"]), set(subject.DGP_IDS))
        self.assertEqual(
            set(selection_power["claims"][subject.DGP_IDS[0]]),
            {"raw_vs_analytic", "fixed_whitened_vs_analytic"},
        )
        self.assertEqual(
            fit_power["claims"][subject.DGP_IDS[0]]["raw_vs_analytic"][
                "episode_count"
            ],
            4,
        )
        # Ledger output is strict JSON: invalid mechanical metrics use null,
        # never nonstandard NaN tokens.
        json.dumps(ledger, allow_nan=False)

    def test_exact_architecture_costs_include_all_heads(self) -> None:
        calls = np.asarray([1, 2, 3, 4], dtype=np.int64)
        pooled = subject._compute_accounting(calls, "balanced_pooled_dual")
        envelope = subject._compute_accounting(calls, "domain_envelope_eight")
        self.assertEqual(pooled["gate_flops_per_evaluation"], 7_985)
        self.assertEqual(envelope["gate_flops_per_evaluation"], 20_537)
        self.assertEqual(pooled["gate_nonflops_per_evaluation"], 5)
        self.assertEqual(envelope["gate_nonflops_per_evaluation"], 11)
        self.assertGreater(envelope["total_flops"], pooled["total_flops"])
        self.assertTrue(pooled["equivalent_call_exact_integer_identity"])

    def test_fractional_analytic_budget_is_exact_for_adversarial_integers(self) -> None:
        rng = np.random.default_rng(313)
        rows = 7
        losses = rng.uniform(size=(rows, 4))
        fixed1 = rows * (
            subject.BASE_FLOPS_PER_ROW + subject.DEPTH1_FLOPS_PER_ROW
        )
        fixed4 = fixed1 + rows * 3 * subject.ADAPTER_FLOPS_PER_ADDITIONAL_CALL
        for budget in (fixed1, fixed1 + 1, fixed1 + 7_985, fixed4 - 1, fixed4):
            allocation = subject._strongest_analytic_mixture(losses, budget)
            numerator = allocation["allocation_total_counted_flops_numerator"]
            denominator = allocation["allocation_total_counted_flops_denominator"]
            self.assertEqual(numerator, budget * denominator)
            self.assertTrue(allocation["exact_integer_cross_product_identity"])
            weight_numerator = allocation["weight_upper_numerator"]
            weight_denominator = allocation["weight_upper_denominator"]
            self.assertGreaterEqual(weight_numerator, 0)
            self.assertLessEqual(weight_numerator, weight_denominator)
        with self.assertRaisesRegex(ValueError, "outside"):
            subject._strongest_analytic_mixture(losses, fixed1 - 1)

    def test_comparator_rngs_are_exact_ledger_assignments(self) -> None:
        records = []
        purpose = {
            "raw": "seeded_weak_more_raw",
            "fixed_whitened": "seeded_weak_more_fixed_whitened",
            "histogram": "within_episode_call_histogram",
        }
        expected = synthetic_comparator_seeds()
        for dgp in subject.DGP_IDS:
            for key, assigned in expected[dgp].items():
                records.append(
                    {"regime": dgp, "purpose": purpose[key], "rng_id": assigned}
                )
        ledger = {
            "analysis_rng_ids": {"comparator": {"rng_ids": records}}
        }
        self.assertEqual(subject.comparator_seeds_from_ledger(ledger), expected)

    def test_lock_is_verified_before_selection_loader_and_gate_is_not_refit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted_path = root / "fitted_candidates.npz"
            lock_path = root / "fit_lock.json"
            subject.save_fitted_candidates(fitted_path, self.fitted)
            subject.seal_fit_lock(
                fitted_path,
                lock_path,
                fit_input_hashes={"synthetic": "a" * 64},
                fixed_whitening_sha256="b" * 64,
            )
            # Hash drift must stop execution before the callback can expose any
            # selection outcome.
            with fitted_path.open("ab") as handle:
                handle.write(b"drift")
            opened = False

            def loader():
                nonlocal opened
                opened = True
                return self.per_dgp

            with self.assertRaisesRegex(RuntimeError, "hash drift"):
                subject.select_after_fit_lock(
                    fitted_path=fitted_path,
                    fit_lock_path=lock_path,
                    selection_loader=loader,
                    comparator_seeds=synthetic_comparator_seeds(),
                    selection_ledger_path=root / "selection_ledger.json",
                    gate_fit_path=root / "gate_fit.npz",
                )
            self.assertFalse(opened)

        fake_ledger = {
            "selected_candidate_index": 0,
            "selected_candidate_id": str(self.fitted["candidate_ids"][0]),
        }
        gate = subject.selected_gate_arrays(self.fitted, fake_ledger)
        self.assertEqual(gate["weights"].dtype, np.dtype(np.float32))
        self.assertEqual(gate["biases"].dtype, np.dtype(np.float32))
        self.assertEqual(gate["thresholds"].dtype, np.dtype(np.float32))
        self.assertEqual(str(gate["runtime_dtype"]), "float32")
        np.testing.assert_array_equal(
            gate["weights"], self.fitted["compiled_weights"][0, :, :2]
        )
        np.testing.assert_array_equal(
            gate["biases"], self.fitted["compiled_biases"][0, :, :2]
        )
        np.testing.assert_array_equal(
            gate["thresholds"], self.fitted["thresholds"][0]
        )

    def test_gate_freeze_binds_exact_compiler_semantics_and_rejects_tamper(
        self,
    ) -> None:
        ledger = subject.evaluate_selection(
            self.fitted,
            self.role,
            comparator_seeds=synthetic_comparator_seeds(),
        )
        self.assertIsNotNone(ledger["selected_candidate_index"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fitted_path = root / "fitted_candidates.npz"
            lock_path = root / "fit_lock.json"
            selection_path = root / "selection_ledger.json"
            gate_fit_path = root / "gate_fit.npz"
            compiled_path = root / "compiled_gate.npz"
            manifest_path = root / "compiled_gate_manifest.json"
            freeze_path = root / "gate_freeze.json"
            subject.save_fitted_candidates(fitted_path, self.fitted)
            subject.seal_fit_lock(
                fitted_path,
                lock_path,
                fit_input_hashes={"synthetic": "a" * 64},
                fixed_whitening_sha256="b" * 64,
            )
            ledger["fit_lock_sha256"] = subject.sha256_file(lock_path)
            ledger["fitted_candidates_sha256"] = subject.sha256_file(fitted_path)
            selection_path.write_text(
                json.dumps(ledger, sort_keys=True, allow_nan=False) + "\n"
            )
            gate = subject.selected_gate_arrays(self.fitted, ledger)
            np.savez_compressed(gate_fit_path, **gate)
            compiled = {
                "schema_version": np.asarray(1, dtype=np.int16),
                "architecture": gate["architecture"],
                "candidate_id": gate["candidate_id"],
                "weights": gate["weights"].copy(),
                "biases": gate["biases"].copy(),
                "thresholds": gate["thresholds"].copy(),
                "head_names": gate["head_names"].copy(),
            }
            np.savez_compressed(compiled_path, **compiled)
            architecture = str(gate["architecture"].item())
            manifest = {
                "passed": True,
                "source_gate_fit_sha256": subject.sha256_file(gate_fit_path),
                "compiled_gate_sha256": subject.sha256_file(compiled_path),
                "target_or_contact_arrays_opened": False,
                "prior_outcome_arrays_opened": False,
                "metadata": {
                    "candidate_id": str(gate["candidate_id"].item()),
                    "architecture": architecture,
                    "head_count": int(gate["head_count"].item()),
                    "runtime_dtype": "float32",
                    "compiler_performed_fitting_or_recentering": False,
                    "float32_cast_max_abs": {
                        "weights": 0.0,
                        "biases": 0.0,
                        "thresholds": 0.0,
                    },
                    "inference_score": (
                        "minimum_over_all_stage_specific_affine_heads"
                    ),
                    "continue_rule": (
                        "score_strictly_greater_than_stage_threshold"
                    ),
                    "gate_cost": {
                        "total_gate_flops": (
                            subject.ARCHITECTURE_GATE_FLOPS[architecture]
                        ),
                        "nonflop_operations": (
                            subject.ARCHITECTURE_GATE_NONFLOPS[architecture]
                        ),
                    },
                },
                "complete_compute_derivation": {"passed": True},
            }
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n"
            )
            self.assertEqual(
                manifest["metadata"]["float32_cast_max_abs"],
                {"weights": 0.0, "biases": 0.0, "thresholds": 0.0},
            )
            freeze = subject.seal_gate_freeze(
                fitted_path=fitted_path,
                fit_lock_path=lock_path,
                selection_ledger_path=selection_path,
                gate_fit_path=gate_fit_path,
                compiled_gate_path=compiled_path,
                compiled_gate_manifest_path=manifest_path,
                gate_freeze_path=freeze_path,
            )
            self.assertTrue(
                freeze["compiled_runtime_arrays_bitwise_equal_selected_gate"]
            )
            self.assertTrue(
                all(freeze["compiled_gate_manifest_semantic_checks"].values())
            )

            tampered = json.loads(manifest_path.read_text())
            tampered["metadata"]["continue_rule"] = "greater_than_or_equal"
            tampered_path = root / "tampered_manifest.json"
            tampered_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RuntimeError, "semantic cross-link"):
                subject.seal_gate_freeze(
                    fitted_path=fitted_path,
                    fit_lock_path=lock_path,
                    selection_ledger_path=selection_path,
                    gate_fit_path=gate_fit_path,
                    compiled_gate_path=compiled_path,
                    compiled_gate_manifest_path=tampered_path,
                    gate_freeze_path=root / "tampered_freeze.json",
                )


if __name__ == "__main__":
    unittest.main()
