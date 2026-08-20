# Manuscript source map

This map resolves every headline claim, figure, and table in `main.tex` to the
local terminal artifact from which it was transcribed or recomputed. Paths are
relative to the repository root
`/Users/rishisim/Documents/research/lewm-p2-contact`. The authoritative catalog
of full SHA-256 values is
`runs/lewm_paper_evidence_synthesis/EVIDENCE_INDEX.json`; all catalog hashes
used by the figure script are checked before any figure is written.

For compactness, the following path prefixes are used below. Each expansion is
literal and relative to the repository root:

- `CUBE/` = `runs/lewm_v5_readiness_program/v5_package_versions/v004/`
- `SHIFT/` = `runs/lewm_v5_generalization/attempts/v005/`
- `PILOT/` = `runs/lewm_pusht_replication_pilot/`
- `BINARY/` = `runs/lewm_pusht_binary_confirmation/`
- `BRIDGE/` = `runs/lewm_frozen_gate_planning_bridge/`
- `DECOMP/` = `runs/lewm_planning_bridge_decomposition/`
- `SYNTH/` = `runs/lewm_paper_evidence_synthesis/`

No experiment artifact is generated or modified by the manuscript workflow.

## Headline claims

| ID | Manuscript claim | Exact local evidence | Reconstruction or check |
|---|---|---|---|
| H1 | Frozen Cube PlanOracle allocation improves both raw and native-whitened next-latent MSE at matched counted FLOPs. | `CUBE/decision.json`; `CUBE/metrics/v5_confirmation_episode_metrics.npz`; `CUBE/metrics/bootstrap_replicates.npz`; `CUBE/metrics/compute_ledger_realized.json` | Episode effects and 20,000 bootstrap quantiles are reconstructed by `SYNTH/evidence_check.py` and asserted again by `manuscript/generate_figures.py`. Exact effects: `6.769003361969305e-06` raw and `0.0006086220971507569` whitened. Exact adaptive total: `4,332,936,120,435` FLOPs. |
| H2 | The frozen Cube shift study is partially robust: four of six simultaneous claims pass. | `SHIFT/decision.json`; `SHIFT/analysis_result.json`; `SHIFT/metrics/bootstrap_replicates.npz`; `SHIFT/metrics/markov_oracle_episode_metrics.npz`; `SHIFT/metrics/plan_action_noise_0p2_episode_metrics.npz`; `SHIFT/metrics/plan_random_action_0p1_episode_metrics.npz` | `SHIFT/decision.json/supported_co_primary_claim_count = 4`; individual and simultaneous quantiles are independently recomputed by both checkers. The raw Markov and raw random-action simultaneous lower bounds are nonpositive and remain visible in Figure 2. |
| H3 | The four-depth PushT pilot is formally negative. | `PILOT/PILOT_DECISION.json`; `PILOT/EVALUATION_ARRAYS.npz`; `PILOT/INDEPENDENT_VERIFICATION.json` | The raw effect is `0.003909796985389183`; the fit-whitened effect is `-0.00030990012922212844`. Because both co-primary point effects had to favor adaptive, the locked label is `pusht_replication_pilot_not_supported`. |
| H4 | The PushT binary rule was derived from consumed pilot arrays and then confirmed on a disjoint fresh cohort. | Discovery chronology: `SYNTH/PAPER_STATUS.md`, `SYNTH/NEXT_STEP_DECISION.md`, and `PILOT/PILOT_DECISION.json`. Protocol and outcome: `BINARY/CONFIG.json`, `BINARY/DECISION.json`, `BINARY/EVALUATION_ARRAYS.npz`, `BINARY/INDEPENDENT_CHECK.json` | Protocol lock: `2026-07-18T23:30:23.485504Z`; 240 fixed fresh episodes; no exclusions or replacements. Raw effect `0.002733782040519924`; fit-whitened effect `0.0015692368266779146`; both simultaneous lower bounds positive. |
| H5 | All PushT WeakPolicy episodes across the reported fit, selection, pilot, and confirmation roles have zero task success. | `PILOT/DESCRIPTIVE_CONTEXT.json`; `BINARY/CONFIG.json` (locked pilot-context success fraction); `SYNTH/PAPER_STATUS.md` | The synthesis checker reconstructs fit, selection, pilot-evaluation, and confirmation success fractions and reports zero for each. |
| H6 | Five-step prediction is exploratory and directionally favorable, but the planning bridge is unsupported. | `BRIDGE/RESULTS.json`; `BRIDGE/phase_b_metrics.npz`; `BRIDGE/phase_c_metrics.npz`; `DECOMP/RESULTS.json`; `DECOMP/decomposition_metrics.npz` | At K=5, adaptive-minus-matched is `-6.597279931162478e-05` raw and `-0.0011870892398969934` whitened. Candidate intervals cross zero; `informative_start_count = 0`; selections differ at 1/20 starts and zero informative starts. |
| H7 | Counted FLOPs are not latency; synchronized timing does not favor the adaptive implementations. | `CUBE/metrics/v5_confirmation_latency.json`; `PILOT/LATENCY.json`; `BINARY/DECISION.json` key `synchronized_latency` | Table 7 transcribes all medians. Operation totals are separately reconstructed from the ledgers. Timing does not enter any terminal scientific criterion. |
| H8 | The PushT fit-whitened confirmation depends on the frozen `1e-3` covariance floor; smaller post-outcome refit floors weaken or reverse the effect. | `BINARY/EVALUATION_ARRAYS.npz`; method in `SYNTH/evidence_check.py`; indexed summary in `SYNTH/EVIDENCE_INDEX.json` key `posthoc_whitening_sensitivity` | Post-outcome target-refit effects are `0.0013365069884080866`, `0.0008799668239503079`, and `-0.0001742822988118231` for floors `1e-3`, `3e-4`, and `1e-4`. This is mapped only to the limitation and Appendix table. |

## Figures

All figures are produced by the single read-only script
`manuscript/generate_figures.py`. Its `verify_sources()` function validates the
indexed SHA-256 values and independently reconstructs effect means, bootstrap
quantiles, call histograms, compute totals, sample counts, and planning
adequacy counts before rendering.

| Figure/panel | Displayed evidence | Exact local source |
|---|---|---|
| Figure 1A | Cube raw and native-whitened percent benefits, individual 95% intervals, simultaneous lower caps, sample count, exact FLOPs | `CUBE/decision.json`; `CUBE/metrics/bootstrap_replicates.npz`; `CUBE/metrics/v5_confirmation_episode_metrics.npz` |
| Figure 1B | Negative PushT pilot followed by pilot-derived binary fresh confirmation; percent normalization and appropriate uncertainty labels | `PILOT/PILOT_DECISION.json`; `PILOT/EVALUATION_ARRAYS.npz`; `BINARY/DECISION.json`; `BINARY/EVALUATION_ARRAYS.npz` |
| Figure 1C | Cube and binary depth proportions plus routing Spearman summaries | `CUBE/decision.json` key `compute/call_histogram`; `CUBE/metrics/stagewise_ranking.json`; `BINARY/DECISION.json` keys `compute/call_histogram_depth_1_to_2` and `stage1_score_gain_rank` |
| Figure 2A | All three raw Cube shift claims, individual intervals, and simultaneous lower caps | `SHIFT/decision.json` keys `individual_intervals` and `simultaneous_co_primary`; `SHIFT/metrics/bootstrap_replicates.npz` |
| Figure 2B | All three fixed-whitened Cube shift claims with the same uncertainty distinction | Same shift artifacts as Figure 2A |
| Figure 2C | Native and shifted four-depth allocation proportions | `CUBE/decision.json`; `SHIFT/decision.json` key `compute/*/call_histogram` |
| Figure 3A | Raw and whitened adaptive-minus-matched effects for horizons 1–5, explicitly exploratory | `BRIDGE/phase_b_metrics.npz` arrays `raw_mse` and `whitened_mse`, conditions `adaptive` and `matched`; checked against `BRIDGE/RESULTS.json` |
| Figure 3B | Raw and whitened 20-start candidate effects with start-cluster intervals on separate axes | `DECOMP/RESULTS.json` key `candidate_rollout_fidelity`; `DECOMP/decomposition_metrics.npz` |
| Figure 3C | Physical outcome ranges, exactly constant starts, tolerance-distinct groups, and the zero-informative-start conclusion | `DECOMP/decomposition_metrics.npz` arrays `physical_outcome_range`, `meaningfully_distinct_outcomes`, and `informative_start`; `DECOMP/RESULTS.json` key `candidate_set_informativeness` |

## Main table

| Table | Contents | Exact local source |
|---|---|---|
| Table 1 (`tab:confirmations`) | Consolidated Cube and PushT adaptive/comparator MSE, absolute and relative benefits, and confidence intervals | Cube sources in H1 and PushT sources in H4 |

## Equations and supplementary tables

| Item | Contents | Exact local source |
|---|---|---|
| Equations 1–2 | Stagewise refiner and non-anticipatory dual-head gate | `CUBE/runner.py`; `CUBE/operation_ledger.json`; `PILOT/run_pilot.py`; `PILOT/pusht_core.py`; `PILOT/FROZEN_GATE.npz`; `BINARY/run_confirmation.py` |
| Equations 3–5 | Episode raw loss, benefit sign, and counted-compute ledger | `CUBE/analysis.py`; `CUBE/operation_ledger.json`; `PILOT/run_pilot.py`; `BINARY/run_confirmation.py`; `SYNTH/CLAIM_LEDGER.md` |
| Supplementary table `tab:operations` | Primitive Cube and PushT operation charges | `CUBE/operation_ledger.json`; `PILOT/PILOT_DECISION.json` key `compute`; `BINARY/DECISION.json` key `compute` |
| Supplementary table `tab:shiftfull` | Individual intervals and simultaneous shift bounds | `SHIFT/decision.json`; `SHIFT/metrics/bootstrap_replicates.npz` |
| Supplementary table `tab:hashes` | SHA-256 values for principal numerical arrays | `SYNTH/EVIDENCE_INDEX.json` key `source_catalog` |

## Principal source hashes

| Source ID | SHA-256 |
|---|---|
| `cube_episode_metrics` | `4eb5cc464f2526a9ccd9ab0aaf84f3d789439120390dd95df0fef224d54e9391` |
| `cube_bootstrap_replicates` | `b1a4a8c29d18430e6deb4a60cb19d7147c4d2125dfececc6f05479440657f469` |
| `cube_compute_ledger` | `5e29a09b6345a2c1d776402b7a725313e1d73c8bd7d9d78c60863bed613c9c04` |
| `shift_bootstrap_replicates` | `d1ace0696cee85d10b4472110d4515dc42825487922a4366e0247d65290d3b81` |
| `planning_phase_b_metrics` | `56325873fcbecc01f4ee931473d225faaf1b7ee7e56225510084e7afec3e51f5` |
| `planning_decomposition_metrics` | `ba734b92c2cba7546201693349e331dd06a8e2233429134a71cdc8009b274788` |
| `pusht_pilot_arrays` | `30ec62250b99de0755dc1a9460491007af8445f152596615aad1b561a5f721c8` |
| `pusht_binary_arrays` | `080c052a582c5d20650b1569fd8425c986533e1afbe616a83e2e7e5f9ce89cc9` |

## Related-work provenance

Bibliographic metadata in `references.bib` was checked against primary
proceedings, OpenReview, official conference, or arXiv pages. In particular,
the scope statement for LoopWM comes from arXiv:2606.18208; the nearby fixed
depth-regime study comes from arXiv:2607.10203; and planning-compute comparisons
come from the official ICLR 2026 Sparse Imagination page and arXiv:2206.02380.
The focused search record and URLs are retained in `README.md`.
