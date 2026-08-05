# Venue-neutral paper outline

## Working title

**Learned allocation improves latent prediction at matched counted compute in two visual world-model domains**

Avoid “speedup,” “planning,” “general world-model efficiency,” “zero-shot transfer,” and priority language in the title.

## One-paragraph argumentative spine

Additional recurrent latent refinement has heterogeneous value across transitions. A causal router can learn that heterogeneity and allocate refinement selectively. The cleanest evidence is a frozen, fresh Cube confirmation at matched counted FLOPs. Robustness of that same frozen Cube policy is real but partial across one-factor shifts. A second domain provides a harder and more honest story: the first four-depth PushT pilot failed its formal dual-endpoint rule; its consumed outcomes suggested a simpler binary mechanism; and a separately locked fresh cohort confirmed that binary rule after PushT-specific fitting. Five-step prediction fidelity persists weakly, but a direct decomposition does not support downstream planning. Therefore the contribution is predictive quality at matched counted compute, not control or latency.

## 1. Introduction

### Role

Frame fixed inference depth as a potentially wasteful inductive choice when the marginal value of refinement varies by transition. State the scientific question: can a causal learned router improve prediction relative to the strongest transition-independent fixed-depth mixture when all counted operations are charged?

### Content

1. Fixed compute ignores heterogeneous refinement value.
2. Adaptive routing is only meaningful if comparisons include gate/feature overhead and preserve a strict causal boundary.
3. The paper tests the idea through a confirmatory Cube study, a frozen-policy shift map, and an environment-specific PushT replication with an explicit negative-to-confirmation chronology.
4. Contributions:
   - fresh Cube confirmation on raw and preregistered native-whitened endpoints;
   - partial zero-shot robustness map within Cube;
   - transparent negative four-depth PushT pilot followed by fresh confirmation of a pilot-derived binary rule;
   - negative planning bridge and latency results that delimit the claim.

End the introduction with the strongest bounded claim from `PAPER_STATUS.md`.

## 2. Related work

### Role

Position the work without novelty priority. Distinguish adaptive computation in world models from early exiting, token routing, and generic dynamic-depth inference.

### Required focused pass before submission

- Include **LoopWM** as known relevant adaptive-computation work in text environments.
- Compare what is adapted (rollout/model depth, token/transition, planning budget), what compute is counted, and whether evaluation concerns prediction or control.
- Search specifically for adaptive-depth visual world models, dynamic latent refinement, early-exit video prediction, and learned computation allocation under exact compute constraints.
- Do not write “first adaptive world model” unless an independent, current evidence audit later supports it; the present manuscript does not need that claim.

This synthesis intentionally does not attempt the literature review.

## 3. Problem formulation and method

### 3.1 Frozen world-model prediction and recurrent refinement

Define the base next-latent predictor, the shared/stagewise residual refinement steps, and fixed depths 1–4. State that the target is next-step latent prediction, not task reward.

### 3.2 Causal routing

Define the feature boundary: history latents, normalized actions, current prediction, and current update only. State that contact, reward, success, future target, simulator state, and geometry are excluded from the model/gate inputs. Explain sequential thresholds and reached-stage evaluation.

### 3.3 Counted compute and comparators

Give the exact FLOP ledger for the base predictor, mandatory/optional refiner calls, feature construction, gate heads, action normalization, and reached gate evaluations. Define:

- strongest transition-independent analytic fixed-depth mixture at the adaptive total counted FLOPs;
- conservative seeded runnable mixture with weakly more compute (Cube);
- within-episode call-histogram randomization/permutation.

Explicitly distinguish expectation-level analytic mixtures from executable integer schedules, and counted FLOPs from measured latency.

### 3.4 Metrics and inference

Define raw latent MSE, the Cube PlanOracle-native whitening, the PushT fit-derived whitening with fixed covariance floor ratio `1e-3`, episode averaging, bootstrap unit, multiplicity families, and sign conventions. State that whitening is a coordinate weighting, not a control metric.

### 3.5 Study hierarchy

Provide the discovery/pilot-to-confirmation chronology:

1. Cube candidate selected in prior discovery and sealed in V5 v004.
2. Fresh Cube V5 confirmation.
3. Frozen Cube zero-shot one-factor shifts.
4. Exploratory consumed Cube multi-step/planning bridge and its negative decomposition.
5. PushT-specific refiner fitting and gate selection, then four-depth pilot.
6. Outcome-informed binary hypothesis derived from the consumed pilot.
7. Binary protocol locked before a disjoint 240-episode PushT confirmation.

This section is essential to prevent pilot and confirmation evidence from being conflated.

## 4. Experimental design

## 4.1 Cube PlanOracle confirmation

### Argumentative role

Primary internal-validity test of the full four-depth learned allocator. This is the anchor experiment.

### Design to report

Fresh 1,600 episodes; 38 transitions each; frozen base, refiner, gate, thresholds, normalization, and whitening; dual raw/native-whitened co-primary endpoints; 20,000 episode bootstraps; exact counted-compute comparator and mechanism controls.

## 4.2 Frozen Cube distribution shifts

### Argumentative role

Map the external-validity envelope of the already-confirmed frozen policy without recalibration.

### Design to report

Three one-factor changes, 3,000 episodes each: PlanOracle→MarkovOracle, action noise `0.1→0.2`, and random-action probability `0→0.1`. Six simultaneous claims with familywise alpha `0.05`. This experiment supports partial, not universal, robustness.

## 4.3 PushT four-depth replication pilot

### Argumentative role

Test whether the method can be instantiated in a second visual domain after environment-specific fitting. Preserve it as a formal negative result and use it to motivate, not validate, the later binary hypothesis.

### Design to report

PushT-specific fit/selection/evaluation roles of 120/40/80 episodes; six gate candidates; WeakPolicy; four depths; raw and fixed fit-whitened pilot rule; exploratory intervals; exact compute. Report raw/routing evidence only after the negative label.

## 4.4 Fresh PushT binary confirmation

### Argumentative role

Test the precise outcome-informed hypothesis generated by the consumed pilot: mandatory depth 1 plus selectively allocated depth 2, with stages 2–3 never consulted.

### Design to report

Protocol lock, disjoint seed range, 240 fresh episodes, no replacements/exclusions, 20,000 episode bootstraps, two simultaneous co-primary endpoints, strongest pairwise fixed-depth mixture over depths 1–4, and exact integer counted-FLOP equality.

## 4.5 Five-step prediction and planning decomposition

### Argumentative role

Test the chain from one-step prediction to multi-step fidelity to decision utility, and show exactly where the evidence stops.

### Design to report

- Consumed Cube Phase B: 100 episodes, 3,400 overlapping starts, horizons 1–5, matched depth histograms and compute.
- Consumed 20-start, 64-candidate planning pilot and decomposition: off-policy rollout fidelity, candidate informativeness, actual-latent-cost/physical-error alignment, and selected-candidate utility.

The nominal earlier `planner_ranking_promising` label must be subordinated to the decomposition's conclusion that downstream planning evidence is unsupported.

## 4.6 Latency diagnostics

### Argumentative role

Demonstrate that FLOP accounting was not silently converted into a speed claim. Report the slower sparse-routing timings and measurement exclusions.

## 5. Results

Use the following order and the prose in `RESULTS_NARRATIVE.md`:

1. Cube confirmation passes both co-primary endpoints.
2. Frozen Cube robustness is supported for increased action noise and mixed for the other two shifts.
3. Four-depth PushT pilot fails formally despite positive raw and routing evidence.
4. Pilot-derived binary hypothesis passes a fresh PushT confirmation.
5. Five-step predictive advantage is small and exploratory; downstream planning remains unsupported.
6. Counted-FLOP parity does not produce wall-clock speedup.

Keeping the negative pilot immediately before the binary confirmation is non-negotiable.

## 6. Discussion

### 6.1 Scientific interpretation

Argue that transition-specific computation allocation can improve prediction beyond what is available to a transition-independent depth distribution at the same counted cost. Emphasize that two domains support the mechanism under different study structures.

### 6.2 What transferred and what did not

- Within Cube: the same fully frozen policy transferred partially across one-factor DGP shifts.
- Across Cube to PushT: the method was replicated after PushT-specific fitting; the Cube gate did not transfer zero-shot.
- Across horizons: a small prediction advantage persisted through five steps on consumed data.
- Into decisions: the evidence did not transfer to useful physical ranking or control.

### 6.3 Why the binary PushT result matters

The pilot exposed that depths 3–4 were not required for the useful routing signal in this DGP. Fresh confirmation shows that the simplified stage-1 decision was reproducible, while the chronology prevents claiming it was an untouched first-shot hypothesis.

### 6.4 Metric and comparator interpretation

Discuss raw versus whitened endpoints, the `1e-3` floor, post-hoc floor sensitivity, and the expectation-level nature of analytic mixtures. Point to runnable/histogram controls without overstating equivalence.

## 7. Limitations

Use `LIMITATIONS_AND_REVIEWER_RISKS.md` as the section source. Lead with prediction/control separation, zero-success PushT trajectories, and environment-specific PushT retraining. Then cover metric sensitivity, analytic comparators, modest Cube effect, partial shifts, and latency.

## 8. Conclusion

Conclude only that causal learned allocation improved next-latent prediction at matched counted FLOPs in the confirmed Cube setting and in a fresh confirmation of a pilot-derived PushT binary mechanism. State that practical speed and downstream control remain open.

## Main-text tables and figures

- Tables: use `TABLES.md`, retaining core Tables 1–6 in the main paper unless venue limits require moving detailed latency to the supplement.
- Figures: use only the three figures in `FIGURE_PLAN.md`.

## Supplement

1. Exact operation ledgers and symbolic FLOP derivations.
2. Frozen-object hashes and chronology.
3. Full bootstrap and comparator definitions.
4. Cube shift per-regime call histograms and secondary comparators.
5. PushT fit/selection details and all pilot endpoints.
6. Post-hoc whitening-floor sensitivity, clearly marked non-confirmatory.
7. Five-step per-horizon tables and planning-decomposition diagnostics.
8. Latency protocol, raw repetitions, and exclusions.
