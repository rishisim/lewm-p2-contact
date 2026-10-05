# LeWM refresh: literature and feasibility decision

Reviewed 2026-10-05. This is a project-selection record, not an experimental result or a committed training protocol. No models were trained or benchmark results reproduced during this review.

## Decision

**A general LeWM architecture/training contribution remains viable, but no particular new method has cleared the novelty and evidence gates.** Stop presenting information/data/compute diagnosis, physical-event subgoals, or training-only physical supervision as contributions by themselves. The review found direct prior work for all three.

The recommended next investment is a bounded qualification of a public, stricter evaluation and its strongest relevant baselines. This is preparation for selecting a method, not the paper contribution. Use CLEAR-LeWM v0.8 as the first candidate; use ARC-style ranking audits only as supplementary diagnosis. Keep the completed adaptive-prediction paper closed.

A modest architecture change can support a research paper if it explains a repeatable failure, outperforms strong simple alternatives with matched resources, and transfers beyond its development setting. A leaderboard increase alone does not establish those properties. Neither a wholly untouched topic nor a dramatic architectural invention is required.

## What the closest papers rule out

| Proposed broad idea | Closest fully read work | Implication |
| --- | --- | --- |
| Make latents preserve actions | [SMWM](https://arxiv.org/abs/2606.20104), [Delta-JEPA](https://arxiv.org/abs/2606.31232), [AD-WM](https://arxiv.org/abs/2609.30264) | Inverse dynamics, latent-difference decoding, residual prediction and mutual-information auxiliaries are existing baselines. |
| Make planning geometry better | [TD-JEPA](https://arxiv.org/abs/2607.25337), [CGS](https://arxiv.org/abs/2609.35603), [Temporal Straightening](https://arxiv.org/abs/2603.12231) | Temporal distance, action/transition geometry and straightening are already tested. CGS includes a privileged transition-geometry variant. |
| Learn candidate ranking | [D-JEPA](https://arxiv.org/abs/2609.24749) | Outcome-supervised candidate-set reranking already exists; simulator counterfactual labels must be budgeted explicitly. |
| Add physical training supervision | [JEPA-x](https://arxiv.org/abs/2608.24044), [PSG-JEPA](https://arxiv.org/abs/2608.06799), [Scaffolder](https://arxiv.org/abs/2405.14853) | Training-only sensing is not a gap. These methods use different planning/policy interfaces and cannot be compared by headline scores. |
| Separate motion, action and background | [MotionJEPA](https://arxiv.org/abs/2609.23881), [DWM](https://arxiv.org/abs/2607.18715), [StarWM](https://arxiv.org/abs/2609.30667) | Motion losses and action/world separation already have direct precedents; DWM modifies LeWM itself. |
| Factorize the representation | [JEPA-Anything](https://arxiv.org/abs/2609.20800) | Generic latent factorization is not enough; its broad prediction results do not establish broad control gains. |
| Use physical dynamics structure | [H-JEPA](https://arxiv.org/abs/2609.33497) | A particularly important compact architectural comparator, with public code and manipulation results. |
| Plan through nearer goals / variable horizons | [Aim Short](https://arxiv.org/abs/2609.30036v4), [Beyond the Next Step](https://arxiv.org/abs/2606.21775) | Retrieved subgoals and variable-length prediction are strong alternatives to event-based temporal abstraction. |

These overlaps reject generic novelty claims, not every possible extension. The important distinction is the exact mechanism and the evidence supporting it.

## Findings that change the decision

**Action sensitivity is not sufficient.** [The Intervention Gap](https://arxiv.org/abs/2608.29998) separates representation of real effects from propagation of those effects through the predictor. It reports direction/scale errors despite substantial action sensitivity. This supports investigating calibrated effects, but the paper does not establish a new architecture or a closed-loop planning remedy.

**H-JEPA is a substantive architecture baseline.** Its orthonormal action port constrains instantaneous full-latent action gain; its port-inverse consistency loss is closely related to a projected rollout loss. Reported Cube gains are substantial, but the effect is not attributable to inverse consistency alone. A variable-gain port is not inherently novel: learned state-dependent input maps exist in [port-Hamiltonian control work](https://arxiv.org/abs/2401.09520). Also, no object movement before contact does not refute H-JEPA: the agent can move and the latent action direction can rotate. [Paper and appendices](https://arxiv.org/abs/2609.33497).

**Offline rank improvements can fail online.** ARC-Bench reports a learned terminal probe that improves fixed-set ranking but fails its small online test. Aim Short shows that target selection materially changes performance; its main PushT query episodes are excluded from retrieval but drawn from the source training split, so model-pretraining exposure needs separate scrutiny. Neither finding proves our historical planning failure had the same cause. [ARC-Bench](https://arxiv.org/abs/2609.05461), [Aim Short](https://arxiv.org/abs/2609.30036v4).

**Representation robustness is real but crowded.** [The Obsessed Encoder](https://www.enigma.inc/posts/obsessed-encoder) provides reproducible nuisance-feature failures. MotionJEPA already addresses a motion/static distinction; StarWM documents failures involving coherent backgrounds and passive objects. A generic second stream or motion weighting would need a sharper justification than these examples alone.

**Paper numbers can be stale or incomparable.** The [Temporal Straightening repository](https://github.com/agentic-learning-ai-lab/temporal-straightening) documents evaluation/training corrections and reruns. [INTACT's repository](https://github.com/zju3dv/INTACT-JEPA) documents a corrected actor-evaluation path. INTACT's search-free interface uses expert action supervision and a goal-conditioned action operator: its success cannot be credited entirely to improved predictive dynamics. Preserve direct-policy and actor-disabled planning comparisons separately.

## Benchmark choice

1. **CLEAR-LeWM v0.8: first qualification target.** This independent community evaluation provides frozen manifests, repaired evaluation, Moderate/Strict modes, reference results and public code. Strict evaluates more precise task completion. It already includes INTACT comparisons, so the task is not simply to beat vanilla LeWM. Pin the version, manifests and inference mode; batch size changes can alter CEM randomness. Public availability is verified; local execution is not. [Repository and current results](https://github.com/DavidSunok/CLEAR-LeWM).
2. **Aim Short: complementary long-range test.** Public controller code and per-episode results exist, but fresh runs need upstream datasets, weights, caches and simulators. Match search budgets and establish fresh model-disjoint test episodes. Use it if the proposed method claims long-range planning benefit. [Reproduction instructions](https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far/blob/main/REPRODUCING.md).
3. **ARC-Bench: diagnostic design, not presently verified turnkey infrastructure.** No official runnable package was located; its appendix describes a future release. Do not call a home-built adaptation an official benchmark run. Ranking errors must be assessed against chance, regret and candidate quality, not just the fraction of nonoptimal top picks. [Paper](https://arxiv.org/abs/2609.05461).
4. **OGBench: later external transfer.** A public, broader goal-conditioned suite, including manipulation beyond the single Cube task. Adapter and training costs make it a second-stage test, not the first smoke test. [Official repository](https://github.com/seohongpark/ogbench).

Audited source snapshots: CLEAR `14acb4924f6cb9a658b62c9be63efb78e59f1987`, INTACT `653ee22266a34a74efca21b0b03dfc1fd6fa37ff`, Temporal Straightening `2c3c7666a69a730042590d548c6731259c4183ac`. Recheck upstream changes before execution rather than silently mixing versions.

## One conditional research lead, not a selected architecture

Working hypothesis: **a model can retain action information while misrepresenting which physically different outcomes nearby actions produce; correcting that error could improve planning across changing interaction regimes.**

This is more general than detecting a secure grasp, but novelty remains unproven. AD-WM, H-JEPA, D-JEPA and CGS are essential nearby work. Do not rebrand state-dependent dynamics or another inverse loss as a new principle.

The smallest useful test is to compare nearby action alternatives from the same held-out starts, using realized simulator outcomes. Separate endpoint representation error, predicted effect direction/scale, target-cost error, and elite-selection regret. Stratify interaction states for diagnosis, while keeping the proposed method free of task-specific event labels if claiming generality. Use physical task features as well as each model's latent coordinates, because different encoders can rescale the comparison.

Proceed toward a module only if a goal-relevant prediction failure remains under a strong baseline and a simple flexible control fixes it. Reject this lead if adequate target selection eliminates the gap, candidates contain no useful action, a standard inverse/residual baseline solves it, or improvements require privileged test-time information outside the intended setting.

## Bounded next step

This review recommends, but does not execute, the following qualification:

- Pin and smoke-test CLEAR v0.8 on one manipulation task; verify dataset/checkpoint access, reset semantics and resource needs before scheduling training.
- Establish vanilla LeWM plus one strong compatible method under the same planner. Prefer H-JEPA for architectural structure or AD-WM for action representation, according to available artifacts and the observed failure; do not begin with eight retrained baselines.
- Inspect a small, fresh validation set of failures. Separate model errors from target/objective, search and success-rule issues. Previously consumed local evaluation episodes are development evidence, not fresh confirmation.
- Produce one page stating the repeated failure, nearest existing remedy, proposed mechanism and smallest falsifying experiment. If no such page is defensible after the bounded qualification, stop this branch rather than launch another training campaign.

For an eventual paper, require multiple training seeds, paired held-out evaluations, appropriate uncertainty, matched data/compute/tuning budgets, causal ablations, and a held-out task or second backbone commensurate with the generality claim. Performance on four tasks with one backbone supports that scope; it does not establish universal world-model improvement.

## Reading coverage and remaining boundaries

Seven Sol research agents plus the parent reviewed the material. **25 distinct papers were read through their main text and all available appendices**:

- Action/prediction: AD-WM, D-JEPA, DA-LeWM (2608.18746), SMWM, Delta-JEPA, TD-JEPA, CAER (2608.30897), H-JEPA.
- Evaluation: ARC-Bench, Aim Short.
- Sensing/policy: Scaffolder, JEPA-x, PSG-JEPA, ContactWorld (2606.13877), INTACT (2607.26056).
- Representation: JEPA-Anything, Planning Limits (2609.39235), MotionJEPA, StarWM, DWM, Contrastive World Models (2609.22175), Temporal Straightening, CGS.
- Parent full reads: The Intervention Gap, Beyond the Next Step.

The Obsessed Encoder blog was read fully. LeWM evaluation sections, TRM, PhyLatent, PIGDreamer and broader bookmarked work received targeted or partial reading; they are not counted as full reads. CLEAR received a documentation/code audit, not an independent replication. Recent sparse-motion work [Keeping JEPA World Models Plannable When Little of the Frame Moves](https://arxiv.org/abs/2610.03137) was screened only; it must be read fully before pursuing a sparse-motion method.

The bookmark inventory and detailed reading notes are retained locally under the ignored `tmp/research-refresh/` directory. Account-derived bookmark selections are excluded from this public research record. A visible-timeline audit cannot guarantee recovery of deleted or unavailable X entries. Literature search is bounded, not proof that no other relevant prior exists.
