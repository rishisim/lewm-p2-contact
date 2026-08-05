# Limitations and reviewer risks

The risks are ranked by their ability to change the scientific interpretation, not by ease of response.

## 1. Prediction quality is not downstream control utility

**Risk.** A reviewer may argue that lower latent MSE is scientifically unimportant unless it improves planning or task success.

**Evidence.** The dedicated Cube decomposition does not support the bridge. Off-policy adaptive-minus-matched prediction effects were small and had 95% start-cluster intervals crossing zero. Zero of 20 candidate starts met the fixed informativeness rule, actual-terminal raw latent goal cost had mean Spearman `0.0237205` with physical error on nine rank-defined starts, and top-5 overlap was `0.040` versus `0.078125` chance. No closed-loop experiment was run.

**Draft response.** “We agree that prediction and control are distinct. We therefore make prediction at matched counted compute the endpoint, report the negative planning decomposition in the main text, and make no control claim. The decomposition is informative precisely because it shows that a confirmed predictive advantage should not be promoted into decision utility without an independently valid bridge.”

**Manuscript action.** Put the negative bridge in the main Results and Figure 3, not only in limitations.

## 2. PushT external validity is limited by zero-success WeakPolicy trajectories

**Risk.** Every PushT episode came from `WeakPolicy(dist_constraint=100)`, and success fraction was `0.0` in all 240 pilot-role episodes and all 240 fresh binary-confirmation episodes. The confirmed prediction effect may characterize failure-region trajectories rather than behavior relevant to successful manipulation.

**Evidence.** Pilot role summaries: fit 120, selection 40, evaluation 80, each with zero success. Binary fixed-cohort summary: 240 episodes, mean reward `-393.4329`, success fraction `0.0`. Privileged reward/success context was summarized only after the policy decision and was not a model/gate input.

**Draft response.** “The PushT result establishes predictive allocation on a fixed weak-policy DGP, not improved manipulation or coverage of successful behavior. We now state the zero-success rate explicitly in the main text. A competent-policy cohort is a high-value extension, but it tests a broader external-validity claim and is not retroactively required for the bounded prediction result.”

**Manuscript action.** Report `success_fraction=0.0` beside the PushT population definition, not as a buried diagnostic.

## 3. PushT is environment-specific retraining, not zero-shot cross-environment transfer

**Risk.** Readers may infer that the Cube gate or refiner transferred to PushT.

**Evidence.** The PushT program fitted a PushT-specific refiner on 120 episodes, fitted whitening on PushT targets, and selected a PushT gate on 40 episodes before the 80-episode pilot. The binary confirmation froze those PushT-specific objects. The pilot's own scope states `cube_gate_zero_shot_transfer: false`.

**Draft response.** “We have revised the framing to ‘method replication after environment-specific fitting.’ The only zero-shot experiment is the within-Cube one-factor shift study, where the complete V5 Cube mechanism was frozen. We do not describe PushT as transfer.”

**Manuscript action.** Use a study-hierarchy diagram or prose divider that labels Cube shifts “zero-shot frozen policy” and PushT “environment-specific fit, then fresh confirmation.”

## 4. The primary analytic comparators are expectation-level mixtures

**Risk.** A fractional fixed-depth mixture can match total compute analytically but is not itself a single executable integer schedule. A reviewer may view it as weaker or operationally artificial.

**Evidence.** Cube converts gate overhead into an analytic-equivalent mean depth (`1.252954098`) and linearly interpolates fixed-depth losses; a seeded executable mixture used `103,565` more FLOPs and was also beaten. Cube additionally beat within-episode histogram randomization. PushT totals match exactly as integers (`102,905,055,196` pilot; `307,585,049,440` binary), but the primary comparator loss is still the expectation of a transition-independent pairwise mixture. PushT binary also beat within-episode call randomization.

**Draft response.** “The analytic comparator answers the clean allocation question: what prediction is available to the strongest transition-independent depth distribution at the same counted budget? We agree it is not a deployed schedule, so we label it expectation-level and report executable seeded/randomized controls where available. The claim is relative to transition-independent allocation, not to every possible static batching implementation.”

**Manuscript action.** Define the comparator equation and its operational status in Methods; do not call it simply a ‘fixed baseline.’

## 5. PushT whitening is sensitive to covariance-floor regularization

**Risk.** The fixed fit-whitened endpoint may look arbitrary because its covariance eigenvalue floor ratio was `1e-3`, and smaller ratios can weaken or reverse the effect.

**Evidence.** The formal binary benefit under the pilot-frozen transform was `0.001569236827` with simultaneous lower bound `0.001268753083`. In a clearly post-hoc outcome-cohort metric redefinition, benefits were `0.001336506988` at floor `1e-3`, `0.0008799668240` at `3e-4`, and `-0.0001742822988` at `1e-4`. This sensitivity uses confirmation targets to re-estimate covariance and therefore cannot replace the frozen endpoint, but it demonstrates limited metric invariance. Raw benefit remained independently positive under the formal analysis.

**Draft response.** “The `1e-3` floor and transform were frozen from the PushT fit role before evaluation, so the confirmatory result is not selected post hoc. We nevertheless agree that whitened conclusions are metric-dependent. We report the floor sensitivity explicitly, avoid metric-general language, and place greater interpretive weight on the separately positive raw endpoint.”

**Manuscript action.** Include the clearly labeled sensitivity note in main limitations and the exact curve/table in the supplement.

## 6. The binary PushT hypothesis was outcome-informed

**Risk.** Simplifying to depth 1 plus optional depth 2 after seeing the negative pilot can look like post-hoc rescue.

**Evidence.** It was post-hoc hypothesis generation. On consumed pilot arrays, the binary simplification yielded raw/fit-whitened point benefits `0.002435240881`/`0.001233341903`. Crucially, the binary protocol and threshold were then locked before a disjoint contiguous 240-episode cohort; no replacements or exclusions occurred, and the fresh result was independently reproduced.

**Draft response.** “We do not present the binary policy as an untouched first-shot PushT hypothesis. The manuscript reports the negative four-depth pilot first, identifies the consumed pilot as the source of the binary hypothesis, and treats only the disjoint 240-episode cohort as confirmation. This is standard discovery-to-confirmation sequencing, with the discovery result fully disclosed.”

**Manuscript action.** Preserve the order negative pilot → derived hypothesis → protocol lock → fresh confirmation in the Abstract and Results.

## 7. The confirmed Cube effect is modest

**Risk.** Raw benefit is only `6.769e-6`, or `0.1968%` of comparator MSE; the whitened benefit is `0.6125%`. Statistical precision may not imply practical importance.

**Evidence.** The fresh cohort is large (1,600 episodes), both simultaneous lower bounds are positive, and within-episode/seeded controls agree. Measured latency did not improve, and no downstream utility was found.

**Draft response.** “We characterize the Cube effect as small and precise. Its scientific role is to establish that transition-specific learned allocation can outperform the strongest transition-independent compute distribution under a strict fully counted budget, not to claim a large practical gain. PushT raw reduction is larger (`2.2521%`), but we do not use that to erase the Cube magnitude.”

**Manuscript action.** Report absolute losses, absolute effects, and relative effects together.

## 8. Counted-FLOP efficiency did not produce wall-clock speedup

**Risk.** Readers may interpret “adaptive compute” or “efficiency” as faster inference.

**Evidence.** Cube adaptive latency exceeded fixed depth 1 at every measured batch size. In PushT binary, cached-base adaptive selection took `16.9063 ms` versus `1.6025 ms` for fixed depth 1 and `2.7835 ms` for fixed depth 2. Pixel encoding was excluded and energy was not measured.

**Draft response.** “Our efficiency axis is predictive quality at matched counted FLOPs. We report the negative latency diagnostics and do not claim speed or energy savings. Realizing speed would require implementation work—compiled routing, batching, and end-to-end measurement—that is outside the present scientific result.”

**Manuscript action.** Prefer “matched counted compute” to “compute efficient” when ambiguity is possible.

## 9. Frozen Cube robustness is partial and metric-dependent

**Risk.** Three one-factor shifts may be overread as general robustness.

**Evidence.** Only increased action noise passed both endpoints. MarkovOracle and random-action contamination passed fixed-whitened but not raw. Four of six simultaneous lower bounds were positive; terminal label `zero_shot_generalization_partial`.

**Draft response.** “We agree and use the study as a robustness map rather than a universal claim. The mixed regimes are shown in the main figure and table, and the raw failures are named explicitly.”

**Manuscript action.** Never summarize the study as “the gate generalized across all shifts.”

## 10. Five-step and planning studies are consumed, small, and tie-heavy

**Risk.** The nominal `planner_ranking_promising` label could be cherry-picked, while the later decomposition reveals that the candidate pool was uninformative.

**Evidence.** The five-step study uses 3,400 overlapping starts from 100 consumed episodes. The ranking pilot has 20 starts; 10 had physical outcome range at most `1e-12 m`, and zero met the informativeness rule. Both off-policy prediction intervals cross zero.

**Draft response.** “We treat the five-step result as exploratory prediction evidence and the downstream bridge as unsupported. The decomposition, not the earlier nominal label, controls the paper-level conclusion. No planning claim is made.”

**Manuscript action.** Do not use `planner_ranking_promising` in the Abstract, title, or contribution list.

## 11. Novelty and literature positioning remain incomplete

**Risk.** A priority claim could be contradicted by existing adaptive-computation world-model work.

**Evidence.** LoopWM is already known relevant work in text environments. This task intentionally did not perform a current literature review.

**Draft response.** “We make no ‘first’ claim. Before submission we will run a focused related-work pass covering LoopWM, adaptive-depth visual/video prediction, dynamic world-model rollouts, and learned routing under compute constraints.”

**Manuscript action.** Complete a focused literature pass during drafting; this is not an experimental blocker.

## Overall reviewer posture

The strongest response is scope discipline, not defensive expansion. The paper's contribution is a carefully confirmed existence claim for transition-specific prediction gains at matched counted FLOPs, accompanied by unusually explicit negative evidence showing where that claim does not extend.
