# World-model refresh: problem discovery and literature boundaries

Reviewed 2026-10-05. Rounds one and two of problem discovery are complete. The round-two adjudication below supersedes the historical round-one ranking. This is a project-selection record, not an experimental result, proof of novelty, or a committed training protocol. No models were trained or benchmark results reproduced.

## Decision

**A general LeWM architecture/training contribution remains viable, but no particular new method has cleared the novelty and evidence gates.** Stop presenting information/data/compute diagnosis, physical-event subgoals, or training-only physical supervision as contributions by themselves. The review found direct prior work for all three.

The user has redirected selection toward significant unresolved problems before methods or benchmarks. This supersedes the earlier recommendation to qualify CLEAR-LeWM first. LeWM is a possible experimental platform, not the boundary of the research question. Keep the completed adaptive-prediction paper closed.

A modest architecture change can support a research paper if it explains a repeatable failure, outperforms strong simple alternatives with matched resources, and transfers beyond its development setting. A leaderboard increase alone does not establish those properties. Neither a wholly untouched topic nor a dramatic architectural invention is required.

## Round-two decision

**P5 is the strongest problem to investigate for a generalizable research direction; P2 is the cheapest diagnostic; P1 is a conditional alternate. None yet warrants a new-method training campaign.** This separates scientific value from implementation convenience. A problem being known does not disqualify it: the question is whether existing solutions leave an important capability unresolved. Conversely, an untested combination of nuisance, task, and backbone is not automatically a significant gap.

Seven Sol agents handled three primary reviews, two independent prior-work challenges, feasibility, and a comparative audit. The parent independently read a component-forgetting study and reconciled conflicting interpretations. The strongest new comparisons substantially weakened our earlier broad claims.

| Candidate | Round-two verdict | Strongest existing answers | What remains to establish |
| --- | --- | --- | --- |
| P5: sequential perception and dynamics reuse | Narrow and prioritize as a research question | Future-blind replay, frozen generic features, latent alignment | A decision-relevant failure in reusable transitions, rather than poor visual pretraining or incompatible old readouts |
| P2: effect accuracy into better decisions | Narrow; use as a diagnostic, with P3 as a possible mechanism | GAP, D-JEPA, AD-WM, rollout-preserving updates | Residual endpoint error on useful planner queries after cost, support, preservation and candidate controls |
| P1: information for later goals | Narrow; lower confidence than round one | DINO-WM, TC-WM, forward-backward representations, reconstruction/object features | A substantive decision-sufficiency tradeoff beyond standard spatial features, not an artificially small latent or one distractor vignette |

### P5 brief — Learn new visual situations without relearning known dynamics

**Research question.** When the observation mapping changes but the underlying interaction rules stay the same, can a sequentially trained world model learn to see the new situation while retaining and reusing its existing action-conditioned dynamics, without access to future-task data?

**Why this survives.** Reusing physical knowledge across new appearances is a meaningful capability, and future-blind learning is a realistic requirement. The compositional benchmark's modular advantage requires an encoder pretrained on all tasks and frozen. Its sequential comparison changes both pretraining and encoder updates; model capacity also differs. Consequently, it motivates this question but does not prove that encoder drift is the cause. [Main paper and supplement](https://arxiv.org/abs/2609.22055).

**Closest answer and remaining boundary.** [WMAR](https://arxiv.org/abs/2401.16650) provides a substantive future-blind replay remedy. [DINO-WM](https://arxiv.org/abs/2411.04983) may avoid adaptation for many appearance/layout shifts. [Adapt & Align](https://arxiv.org/abs/2312.13699) makes latent alignment an existing control, not an invention to claim. None of these comparisons alone settles whether old controlled transitions remain reusable when perceptual adaptation is genuinely necessary. Return loss is not enough: a separate [component-forgetting study](https://arxiv.org/abs/2607.19749) shows that model knowledge can remain usable while the actor fails, in its small replay-maintained setting.

**Simplest alternative explanation.** The new encoder is simply less informative than the all-task-pretrained one, or old heads cannot read its changed coordinates. Neither implies the dynamics knowledge was lost.

**Decisive experiment.** Use an A→B sequence with paired renders of the same physical states/actions and unchanged physics. B introduces a new observation mapping. First compare natural-size frozen generic features and ordinary fixed-memory replay. Only if perceptual adaptation is actually needed, compare an adapted encoder with a simple map into the frozen A dynamics against equal-budget joint updates. Joint A+B pretraining is a labeled ceiling, not a deployable comparator. Evaluate physical transition accuracy, realized candidate-choice regret, and old/new control; evaluate old and refreshed readouts separately. Fit all deployable maps/readouts using permitted training data only. A small state decoder is a diagnostic instrument, not a physical oracle.

**Stop rule.** Drop the proposed mechanism if generic features, replay or simple alignment suffice; if only actor/readout compatibility fails; or if physics changed inadvertently. Advance only after a repeated transition/decision failure survives these controls and an intervention localizes its cause. A positive two-task test motivates further work; it does not establish general continual robot learning.

### P2 brief — Decision-relevant prediction across goals and planner queries

**Research question.** Can an existing visual world model improve the consequences that matter for selecting actions across held-out goals, while preserving its existing rollout behavior and remaining reliable as the planner changes its candidate distribution?

**Correction to the original motivation.** Do-JEPA's 50-episode comparison is base 94%, ordinary augmentation fine-tuning 58%, paired fine-tuning 54%. Thus the collapse does not isolate the paired loss as its cause. The preservation adapter's separate 600-episode comparison is approximately 87.7% versus 87.8% base: compatibility, not a planning gain. Cohorts must not be mixed. [Do-JEPA Appendix F](https://arxiv.org/abs/2609.37378).

**Closest answer and remaining boundary.** [Goal-Aware Prediction (GAP, ICML 2020)](https://proceedings.mlr.press/v119/nair20a.html) already trains reward-free image-goal-aware prediction to improve useful trajectories and planning. [D-JEPA](https://arxiv.org/abs/2609.24749) already improves candidate selection with executed-outcome labels; [AD-WM](https://arxiv.org/abs/2609.30264) provides strong action-aware controls. The problem is therefore not discovering that MSE and decisions differ. A remaining opportunity would require a failure these solutions do not fix under fair goal/data/planning interfaces. Goal conditioning must not be excluded merely to preserve a gap; whether a goal-independent reusable predictor is needed requires a concrete use case.

**Simplest alternative explanation.** Effects improve on irrelevant actions or directions, while the goal score, useful-candidate supply, multi-step preservation or action-support distribution remains wrong.

**Decisive experiment.** On a predeclared non-ceiling population, replay the same action bank from each held-out simulator start. Compare A: predicted endpoint/native latent goal cost; B: realized endpoint encoded by the same model/same cost; C: realized physical endpoint/physical task cost. A→B estimates endpoint-prediction headroom; B→C combines representation and cost limitations, not information loss alone. Physical task cost cannot be applied to a predicted latent without an additional readout. Include one executable strong action/residual baseline before expanding comparisons. Repeat on early and late CEM candidates and then paired closed-loop runs only if the fixed-bank result warrants it.

**Stop rule.** Do not pursue a dynamics remedy if realized endpoints do not help the native score, if useful actions are absent, or if ordinary target/preservation/GAP-style/AD-WM controls resolve the deficit. Improvements confined to mean effect error or a fixed bank are insufficient. P3 is a possible explanation within this test, not a separate method commitment.

### P1 brief — Preserve decision-sufficient state for a specified future task family

**Research question.** Across a declared family of later goals, when do visual nuisance suppression and state compression discard information needed for actual choices, after accounting for strong spatial representations and their resource costs?

**Closest answer and remaining boundary.** DINO-WM already succeeds on unseen wall/door layouts. [TC-WM](https://arxiv.org/abs/2605.25620) supplies compact task-centric states using foundation features and proprioceptive supervision. Reward-free [forward-backward representations](https://proceedings.neurips.cc/paper_files/paper/2021/file/003dd617c12d444ff9c80f717c3fa982-Paper.pdf) address rewards specified after training; [recent rank analysis](https://arxiv.org/abs/2602.11399) limits universal guarantees at low rank. These are substantial answers. Coherent-video plus passive-object failure across strong goal-planning methods is not yet established. Arbitrary future rewards cannot be promised under unrestricted lossy compression.

**Simplest alternative explanation.** The information is present but the predictor, goal geometry or candidate generator fails; a weak linear probe cannot establish deletion. Current observations or valid goal images may already supply the needed state.

**Decisive experiment.** Start with natural-size DINO spatial features and a compact baseline in paired fully visible scenes with independent nuisance changes and held-out task combinations. Keep legitimate goal-image information available. Compare actual action-choice flips and regret on identical useful candidates; localize score versus endpoint errors as in P2. Only if these controls fail should TC-WM, reconstruction and object-centric alternatives and resource curves be expanded. Declare their extra training information rather than removing it to handicap them.

**Stop rule.** Drop the opportunity if ordinary spatial features or data/capacity resolve it, or if a legitimate goal/current observation supplies what the test tried to hide. A single synthetic obstacle or probe failure is not enough. Any eventual claim needs a meaningful sufficiency-versus-resource improvement in more than the development family.

### Recommended next research action and feasibility

Prioritize the **P5 causal discriminator** if choosing the next research direction for the user's generalization objective. Its implementation is larger, but its question is broader than improving one planner's score. Treat P2's frozen-bank audit as the lower-cost alternative if continuity with existing LeWM infrastructure takes priority, not as proof of greater scientific importance. Keep P1 as a fallback rather than forcing its untested conjunction into a flagship claim. No parallel training campaigns are recommended.

This round authorizes and completes review, not pilot implementation. The P5 benchmark's public page contains a Code placeholder; a runnable release was not verified. Local LeWM has state restoration and scoring, but lacks the proposed fixed-bank diagnostic and a continual-learning protocol. The configured ChildLens volume was absent during the read-only audit, the stable-worldmodel cache link was dangling, and no local model/dataset readiness was established. [AD-WM simulation artifacts](https://github.com/ad-wm/ad-wm-code), [DINO-WM source](https://github.com/gaoyuezhou/dino_wm), and [TC-WM source](https://github.com/MinghaoFu/TC-WM) are public; their availability is not a local reproduction. TC-WM's README describes checkpoint release as future work. Verify artifact hashes and time a smoke run before promising GPU hours.

### Round-two reading and completion evidence

New complete reads reported by the reviewing agents: TC-WM main plus Appendices A–E; Denoised MDPs main plus Appendices A–B; Do-JEPA main plus Appendices A–G; AVL-JEPA main plus Appendices A–D; the compositional continual-learning paper plus official project Appendices A–C and learning curves; WMAR main and supplement. The parent read The World Model Remembers, the Actor Forgets including Appendices A–C. Other adversarial sources have targeted reading scope recorded in the local packet; do not count all cited papers as new full reads. Previous StarWM/MotionJEPA/AD-WM/D-JEPA/H-JEPA full reads were reused with decisive passages rechecked.

For each of P1/P2/P5, the review now records an exact question, strongest known answers, remaining evidence gap, alternative explanation, falsifying experiment and verdict. Comparative and feasibility reviews were reconciled. Full notes and coverage are retained under ignored `tmp/research-refresh/adversarial/`; no account-derived selections are published here. This completes round two, while leaving experimental validation and method selection explicitly unperformed.

## Historical round-one result: nine candidate problems

Seven Sol agents covered recent limitations, frontier capabilities, cross-area connections, apparently contradictory findings, stressed assumptions, bookmark omissions, and a skeptical filter. The parent added continual-reuse research and consolidated overlaps. This round reused the previous full reads and made targeted new primary-source reads; it was not a second cover-to-cover audit of every paper.

The nine candidates below are not nine established novel gaps. “Evidence” means an observed failure in the cited setting; “boundary” states what still needs verification. No claim is made that these are the field's universally agreed most important problems. Priority balances broader consequence, concrete evidence, open questions, and a feasible test.

| ID | Research question | Evidence for failure | Fit for modest resources | Round-two priority |
| --- | --- | --- | --- | --- |
| P1 | Can a compact model discard nuisance while retaining information needed by future goals? | Direct in an adjacent control family; transfer to LeWM unproven | Good controlled-simulation fit | Shortlist |
| P2 | When can better action-effect prediction improve decisions without damaging other model capabilities? | Direct LeWM evidence | Good, if compatible artifacts exist | Shortlist |
| P3 | Can a model remain reliable on actions selected by increasingly strong search? | Direct hallucination evidence; mechanism needs isolation | Good frozen-model diagnostic | Alternate; linked to P2 |
| P4 | What interaction data make alternative action outcomes identifiable? | Theory under restricted assumptions; nonlinear boundary unresolved | Good simulator reset test | Fundamental, but highly occupied |
| P5 | Can learned dynamics remain reusable while perception learns new situations? | Direct continual-control study; unmatched capacity caveat | Moderate; new sequential-task setup | Shortlist, larger scope |
| P6 | Can models compose dependent physical events beyond trained chain lengths? | Direct video-model evidence; not established for LeWM | Small mechanism test possible | Exploratory |
| P7 | Can a model preserve and update hidden object state for later decisions? | Direct video persistence failures; compact-control boundary unproven | Moderate controlled occlusion setup | Exploratory |
| P8 | When should an uncertain model act to obtain information? | Strong theoretical motivation; recent probabilistic methods already exist | Moderate; requires careful observability setup | Crowded, conditional |
| P9 | Can adaptation revise changed knowledge while retaining what remains valid? | Recent position argument plus adaptation literature | Moderate; online-feedback assumptions matter | Deprioritized pending sharper evidence |

### P1 — Preserve information before its future use is known

**Question and importance.** A static obstacle, target, or currently untouched object can determine a later successful action. Can reward-free predictive compression suppress distracting variation without deleting such information? This matters for reusing a model across goals, beyond one grasp or contact detector.

**Evidence and boundary.** StarWM's coherent-background experiment loses passive-object position and its reward-aware variant performs better. MotionJEPA succeeds on static-texture distractions; these are different protocols, not contradictory head-to-head results. The sparse-motion study also reports restored action sensitivity without reliable hard pushing. These establish failure examples, not a universal representation defect. [StarWM Appendix F.4](https://arxiv.org/abs/2609.30667), [MotionJEPA](https://arxiv.org/abs/2609.23881), [sparse-motion study](https://arxiv.org/abs/2610.03137).

**Small test.** Hold dynamics and nuisance video fixed; change a visible passive obstacle/target that changes the correct later action. Compare strong inverse/motion objectives, ordinary reconstruction, and goal-conditioned controls on identical candidate actions. Measure actual decision changes and control, not only linear probes. **Reject** if existing methods retain the needed state, routine data/capacity controls solve the failure, or the information was never visible. Full arbitrary-future-task sufficiency is impossible under unrestricted compression; specify a task family and capacity budget.

### P2 — Improve physical effects without breaking the decisions built on the model

**Question and importance.** How can correcting local action consequences yield improved control while preserving factual rollouts and the goal relationships the planner already uses?

**Evidence and boundary.** Do-JEPA directly improves paired effects, yet full fine-tuning reduces PushT planning from 94% to 54% in its small test. Its preservation adapter maintains planning in a larger equivalence test, without a gain. This does not prove accurate effects harm control: intervention distribution, rollout stability, task ceiling and cost geometry remain possible explanations. AVL-JEPA also occupies generic action-grounded robustness; its counterfactual regret selects displacement magnitude, not goal-conditioned progress. [Do-JEPA Appendix F](https://arxiv.org/abs/2609.37378), [AVL-JEPA Appendix B.2](https://arxiv.org/abs/2610.03587).

**Small test.** On a non-ceiling task, compare predicted versus simulator-realized outcomes for the same useful alternatives, while separately replacing the dynamics and goal score with oracles. Include AD-WM/H-JEPA and preservation controls. **Reject** if an existing remedy fixes the relevant decisions, remaining errors lie in irrelevant directions, or target/search failure explains the null. A new effect loss alone is already preempted.

### P3 — Reliability under the planner's own changing queries

**Question and importance.** Can a model know when the action sequence selected by optimization is trustworthy? Search actively selects unusual errors; accuracy on random or logged candidates can conceal this.

**Evidence and boundary.** RP1 reports missed-grasp attachment hallucinations and partial improvement from collecting planner-induced failures. D-JEPA's fixed-set simulation successes and ARC's offline-to-online reversal use different methods and information, so they are not an efficacy contradiction; D-JEPA also has live robot reranking. [RP1 §7.4](https://arxiv.org/abs/2608.18669), [D-JEPA](https://arxiv.org/abs/2609.24749), [ARC-Bench](https://arxiv.org/abs/2609.05461).

**Small test.** Track prediction-versus-realized cost for selected and unselected actions over successive CEM refinements, with fixed goals and adequate candidates. Compare ordinary ensembles/support penalties and planner-collected corrective data. **Reject** if cost alignment, proposal quality, or those existing controls explain the failure. This is the familiar model-exploitation problem; the possible opportunity is calibrated reliability under adaptive visual-model queries, not discovering exploitation itself. Merge with P2 if the same mechanism explains both.

### P4 — Identify counterfactual effects from limited interaction

**Question and importance.** Which additional interactions teach a model the action alternatives needed for decisions, rather than merely improving prediction along an expert's path?

**Evidence and boundary.** Controlled-world-model identifiability already relates counterfactual error to conditional action excitation under linear-Gaussian dynamics and invertible observations. Weakly excited directions cannot be recovered universally from fixed data. Plan2Explore and offline uncertainty methods already address exploration and support. [Identifiability §§3–5](https://arxiv.org/abs/2607.22430), [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html).

**Small test.** Match state visitation, marginal action scale and data volume while varying action diversity conditional on state. Test actual alternative actions from held-out simulator resets; compare equal-budget random and targeted new interactions. Recovery states offer one application, but WM-DAgger/SimDist are existing recovery-data approaches. **Reject** if ordinary perturbation data closes the gap or the planner never needs unsupported actions. The theory is already a contribution by others; a visual extension needs a substantive acquisition or partial-identification question, not another loss claiming to infer missing information.

### P5 — Reuse dynamics while the visual representation changes

**Question and importance.** As a robot encounters new appearances and tasks, how can it improve perception without making previously learned transitions unusable? The issue is broader than retaining old-task scores: the coordinate system consumed by the dynamics can change.

**Evidence and boundary.** A recent compositional continual-learning study finds its modular advantage only with a frozen encoder pretrained on all future tasks. Without that privilege it matches the monolithic comparison. Capacity is not fully controlled, so modularity is not isolated as the causal remedy. [Study §§III-E, IV-A and limitations](https://arxiv.org/abs/2609.22055).

**Small test.** Use a short sequence with separable visual and dynamics changes, no future-task access, and fixed replay/capacity budgets. Compare frozen pretrained features, replay, joint-training upper bounds and latent alignment; test old transition validity and new-task learning. **Reject** if standard alignment or stable features close the gap. Continual learning and [latent alignment](https://arxiv.org/abs/2312.13699) are established; the question is preserving reusable controlled dynamics under genuinely sequential visual learning. This requires a new task protocol, not an immediate LeWM patch.

### P6 — Compose dependent physical events

**Question and importance.** Can a predictor generalize to longer chains of dependent interactions when each individual interaction is familiar?

**Evidence and boundary.** The Seriality Gap isolates collision-chain degradation in bidirectional video diffusion using length-matched controls; autoregressive/blockwise alternatives already help. Its theory and large training campaign do not establish the same bottleneck in autoregressive LeWM. [Seriality Gap §§3–5](https://arxiv.org/abs/2607.13031).

**Small test.** Compare independent and causally chained contacts while matching time, object count, total motion, data, and inference work. Measure event order and action-choice accuracy. **Reject** if ordinary recurrence/variable horizons resolve it or if errors track time/occlusion rather than dependency depth. A compact latent experiment can test the mechanism, but cannot claim to solve frontier video generation. Reintroducing adaptive depth or event-based planning without this evidence would repeat the old project's mistake.

### P7 — Maintain hidden state through unobserved interactions

**Question and importance.** Can the model retain object identity and update its possible state while it is hidden, so it chooses a sensible later action?

**Evidence and boundary.** WROP reports object-count and reappearance failures; PlayWorld finds weak out-of-sight evolution. These use video-generation/judge-based evaluations, not LeWM planning. TrackEverything tracks observed history; that does not by itself solve prospective hidden interactions. [WROP](https://arxiv.org/abs/2609.28654), [PlayWorld](https://arxiv.org/abs/2608.13552), [TrackEverything](https://arxiv.org/abs/2609.30222).

**Small test.** Separate remembering a stationary hidden object from predicting its unseen collision, using visible-history and full-state diagnostic controls. Compare recurrent/object-track/belief baselines on reappearance and later action choice. **Reject** if ordinary memory solves it, task choices are unaffected, or the setup demands a unique answer where observations permit several futures. A distribution over hidden states is then the appropriate target. Generic object permanence and object tokens are already occupied research areas.

### P8 — Turn uncertainty into useful information-gathering actions

**Question and importance.** When should the agent spend an action resolving ambiguity instead of immediately pursuing its goal?

**Evidence and boundary.** Branch-JEPA explores multiple latent futures, while probabilistic latent MPC and recurrent counterfactual models already exist. Their distinct results do not establish that compact reward-free visual models reliably choose useful probes. This is a provisional boundary, not an absence claim. [Branch-JEPA](https://arxiv.org/abs/2607.05238), [VJEPA](https://arxiv.org/abs/2601.14354), [CLWM](https://arxiv.org/abs/2609.05834).

**Small test.** Pair visually aliased hidden modes with revealable and unrevealable controls. Evaluate whether a diagnostic action improves expected task return under a fixed action budget. Compare ordinary longer history, belief planning, and active-sensing methods. **Reject** if short history resolves the ambiguity, probing has no value, or established methods already achieve the ceiling. This must survive a serious POMDP/dual-control review before being shortlisted. P7 concerns maintaining hidden state; P8 concerns choosing actions to reduce its uncertainty.

### P9 — Revise changed facts without discarding valid knowledge

**Question and importance.** During deployment, can adaptation update the aspects of dynamics that changed while retaining performance where the old model remains valid?

**Evidence and boundary.** A recent position paper distinguishes obsolete knowledge from catastrophic forgetting; it provides a framing, not a demonstrated new algorithmic failure. SimDist, ICWM, ReDRAW, DALI, Sandwich-Residuals and SPREAD already address substantial parts of adaptation. [Retention position paper](https://arxiv.org/abs/2610.03713), [Sandwich-Residuals](https://arxiv.org/abs/2609.21740), [ICWM](https://arxiv.org/abs/2606.26025).

**Small test.** Separate changed physics from changed appearance with identical feedback budgets; measure adaptation latency and performance on still-valid old regimes. **Reject** if context inference or simple residual updates suffice, or the environment provides too little information to detect change. Do not assume gravity, embodiment or action effects are universally invariant: define what stays fixed in the experimental world. This remains deprioritized because the generic problem and many remedies are already well developed.

## Recommendation after round one

Advance **P1, P2 and P5** to adversarial literature review, not implementation. P1 offers a broad representation-sufficiency question with a manageable test; P2 has the clearest direct LeWM negative result; P5 offers a distinct generalization problem beyond the original contact project, at greater setup cost. These are judgments about investigation value, not publication forecasts. P3 is the strongest alternate and may become part of P2's explanation.

For each shortlisted problem, round two should produce the exact unsolved boundary against the strongest existing method, the simplest alternative explanation, and one experiment capable of rejecting the hypothesis. If a standard method answers it, remove it. No benchmark, architecture, training run, or hardware allocation is selected yet.

The supplemental agent rescreened all 75 previously ambiguous bookmark records and cross-checked the 58 retained posts; no extra distinct candidate cleared screening. Do-JEPA and AVL-JEPA were important new prior-work constraints. The previous 405-post inventory was reused; X was not rescanned this round. Private selections and detailed notes remain in ignored `tmp/research-refresh/`.

## Earlier method and benchmark audit (reference, not the current next step)

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

## Earlier action-effect lead, now narrowed by P2

Working hypothesis: **a model can retain action information while misrepresenting which physically different outcomes nearby actions produce; correcting that error could improve planning across changing interaction regimes.**

This is more general than detecting a secure grasp, but novelty remains unproven. AD-WM, H-JEPA, D-JEPA and CGS are essential nearby work. Do not rebrand state-dependent dynamics or another inverse loss as a new principle.

The smallest useful test is to compare nearby action alternatives from the same held-out starts, using realized simulator outcomes. Separate endpoint representation error, predicted effect direction/scale, target-cost error, and elite-selection regret. Stratify interaction states for diagnosis, while keeping the proposed method free of task-specific event labels if claiming generality. Use physical task features as well as each model's latent coordinates, because different encoders can rescale the comparison.

Proceed toward a module only if a goal-relevant prediction failure remains under a strong baseline and a simple flexible control fixes it. Reject this lead if adequate target selection eliminates the gap, candidates contain no useful action, a standard inverse/residual baseline solves it, or improvements require privileged test-time information outside the intended setting.

## Conditional benchmark qualification after a problem is selected

If the selected problem concerns manipulation planning, the following qualification remains useful. It is deferred until after problem selection, and was not executed:

- Pin and smoke-test CLEAR v0.8 on one manipulation task; verify dataset/checkpoint access, reset semantics and resource needs before scheduling training.
- Establish vanilla LeWM plus one strong compatible method under the same planner. Prefer H-JEPA for architectural structure or AD-WM for action representation, according to available artifacts and the observed failure; do not begin with eight retrained baselines.
- Inspect a small, fresh validation set of failures. Separate model errors from target/objective, search and success-rule issues. Previously consumed local evaluation episodes are development evidence, not fresh confirmation.
- Produce one page stating the repeated failure, nearest existing remedy, proposed mechanism and smallest falsifying experiment. If no such page is defensible after the bounded qualification, stop this branch rather than launch another training campaign.

For an eventual paper, require multiple training seeds, paired held-out evaluations, appropriate uncertainty, matched data/compute/tuning budgets, causal ablations, and a held-out task or second backbone commensurate with the generality claim. Performance on four tasks with one backbone supports that scope; it does not establish universal world-model improvement.

## Earlier full-reading coverage and remaining boundaries

In the preceding method audit, seven Sol research agents plus the parent reviewed the material. **25 distinct papers were read through their main text and all available appendices**. This historical count does not describe the targeted new discovery reads:

- Action/prediction: AD-WM, D-JEPA, DA-LeWM (2608.18746), SMWM, Delta-JEPA, TD-JEPA, CAER (2608.30897), H-JEPA.
- Evaluation: ARC-Bench, Aim Short.
- Sensing/policy: Scaffolder, JEPA-x, PSG-JEPA, ContactWorld (2606.13877), INTACT (2607.26056).
- Representation: JEPA-Anything, Planning Limits (2609.39235), MotionJEPA, StarWM, DWM, Contrastive World Models (2609.22175), Temporal Straightening, CGS.
- Parent full reads: The Intervention Gap, Beyond the Next Step.

The Obsessed Encoder blog was read fully. LeWM evaluation sections, TRM, PhyLatent, PIGDreamer and broader bookmarked work received targeted or partial reading; they are not counted as full reads. CLEAR received a documentation/code audit, not an independent replication. The new discovery round examined main results and selected appendices of [Keeping JEPA World Models Plannable When Little of the Frame Moves](https://arxiv.org/abs/2610.03137), upgrading its previous abstract screen without claiming a complete appendix audit. Do-JEPA, AVL-JEPA and other new sources likewise have their exact reading scope in the local discovery notes. Current hardware availability, reproduction cost and checkpoint compatibility remain unverified.

The bookmark inventory and detailed reading notes are retained locally under the ignored `tmp/research-refresh/` directory. Account-derived bookmark selections are excluded from this public research record. A visible-timeline audit cannot guarantee recovery of deleted or unavailable X entries. Literature search is bounded, not proof that no other relevant prior exists.
