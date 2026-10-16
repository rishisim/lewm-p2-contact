# World-model problem search: three candidates and their novelty boundaries

Reviewed 2026-10-16. This is the current project-selection record, replacing the October 5 ranking; earlier assessments remain in Git history. It is a literature assessment and proposed diagnostic design, not a new experiment, a novelty certification, or a commitment to an architecture. No training or benchmark reproduction was performed in this review.

## Recommendation

Keep three candidates:

1. **P1: preserving usefulness when goals change.** Best fit to the original scientific interest; substantial direct overlap with existing work. Retain the question, withdraw the current compression-causation claim.
2. **Reliability of plans after optimization** (historical P3). Strongest immediate empirical research bet, provided the study goes beyond rediscovering model exploitation or applying an uncertainty penalty.
3. **Composing familiar interactions into unfamiliar causal chains** (historical P6). Ambitious and distinct, but the least established opportunity in action-conditioned latent planning.

These are three directions worth testing, not three demonstrated open gaps. The central selection criterion is whether a failure survives the strongest reasonable existing answer at an acceptable data, supervision, and inference cost. The absence of a particular benchmark experiment is not a research gap.

A useful common question is: **When does a learned model support new decisions beyond the behavior that supplied its training data?** The candidates isolate different changes: the goal, the search procedure, and the composition of physical events. Do not combine all three into one initial experiment.

| Candidate | Why it matters | Main novelty threat | Current evidence strength | First rejection condition |
| --- | --- | --- | --- | --- |
| P1: changing goals | Reusing learned physics should not require relearning every newly relevant object | Query-dependent sufficiency, task-aware abstraction, spatial/object representations already address much of this | Strong local LeWM failure; general mechanism unresolved | Adequate familiar-object training plus a standard representation/scorer solves it cheaply |
| Plan reliability | More computation is valuable only if selected actions work in the environment | Pessimistic MBRL, outcome verification, uncertainty calibration, conformal control | Direct exploitation and uncertainty failures in specific systems; positive remedies also exist | Standard calibration, ensembles, or replanning recover useful search without material extra cost |
| Causal composition | Learning every complete interaction sequence would limit reuse | Learned physics simulators, object-oriented models, autoregression, compositional planners | Strong adjacent evidence; weak direct evidence for the proposed latent-planning boundary | Structured or autoregressive baselines compose adequately, or failures reduce to ordinary local error |

## 1. P1: what must survive when the agent's goal changes?

### Precise question and significance

Can a visual world model learn from interactions concentrated on one set of goals, then support different goals over the **same familiar objects and physics**, while remaining robust to genuine nuisance and affordable to plan with?

A mug can be background during one task and the target of the next. A useful representation must distinguish nuisance that never changes a decision from information that is merely irrelevant to the behavior policy that generated the data. That distinction motivates the research, but does not establish a new phenomenon.

Specify a finite, meaningful family of tasks. Requiring a small representation to support every conceivable future objective is neither necessary nor a fair empirical target. If objects or physical mechanisms were never observed, their acquisition is a different problem.

### What the current proposal establishes

The supplied proposal documents a large asymmetric failure in the evaluated LeWM pipeline. It does not isolate why that failure occurs. Its strongest causal claims should not guide project selection yet.

- All evaluated models use the same 192-dimensional representation. There is no controlled compression intervention.
- Fine-tuning a PushT model on peg interactions does not eliminate inherited specialization. The weak scratch run is not a competent control for this.
- Position supervision makes the peg decodable, but that does not establish accurate action-conditioned peg forecasts or useful goal-distance geometry.
- The reference changes both dynamics and the scoring interface. It uses role-dependent physical costs and optional approach shaping; it is evidence of physical feasibility under that interface.
- The candidate correlations cited for the second layer come from ordinary models, not the privileged checkpoint.
- Endpoint pictures do not uniquely specify trajectory requirements such as never disturbing an object.

Sources: [proposal source](proposal/p1_proposal.tex), [reference implementation](../src/lewm_research/probe/reference.py), [task costs](../src/lewm_research/probe/conditions.py). These files were inspected; no new numerical results are claimed here.

A DINO-WM control is useful but cannot identify compression causally: representation structure, pretraining, trainability and prediction differ. A dimensionality sweep within a matched family would be needed before blaming latent size; competence and optimization must also be controlled.

### Closest prior work and the boundary it leaves

| Work | Existing answer | Implication for P1 |
| --- | --- | --- |
| [What Must a World Model Distinguish for Planning?](https://arxiv.org/html/2609.33030v1) | Directly analyzes information needed by different queries and search stages; studies held-out objective compositions and modular query-aware proposals | The broad question and the idea of reusing physical predictions across objectives are already addressed. Its one-bit experiment compresses candidate messages after prediction, not a visual world state. A possible remaining empirical question concerns how interaction-biased visual training produces useful or harmful invariance before queries arrive. |
| [JEPA-Bisim, v2](https://arxiv.org/html/2602.18639v2) | Learns compact nuisance-robust features; its 1,000-sample PushT comparison reports native success 48% to 36% while relative shift degradation improves 47.6% to 7.8% | A measured tradeoff motivates study, but is not an impossibility result. Its non-colliding distractors do not establish behavior when a familiar object becomes an interaction target. |
| [DINO-WM](https://arxiv.org/abs/2411.04983) | Spatial pretrained features support visual planning and transfer to new configurations | A necessary strong positive control. If it solves the task at acceptable cost, the broad capability claim weakens substantially. |
| [Task-Centric World Models](https://arxiv.org/abs/2605.25620) | Compresses foundation features with reconstruction and physical alignment | Compactness alone is not the problem. Its measurement and planning interfaces must be accounted for in a comparison. |
| [Causal-JEPA](https://proceedings.mlr.press/v306/nam26c.html) | Object-level masking improves relational reasoning and supports efficient planning | Object-centric latent planning is an existing answer, not a proposed novelty. Compare a competent object representation rather than a weak slot baseline. |
| [ATLAS](https://arxiv.org/html/2609.36333) | Preserves relational geometry between encoder and planning representations, with LeWM planning gains | “Information is decodable but the planning latent has bad geometry” also has direct prior work. Whitening and geometric alignment are serious controls. |

Related negative findings in [StarWM](https://arxiv.org/html/2609.30667) and [SLIM](https://arxiv.org/abs/2610.03137) further weaken a claim to first discover neglected passive/small moving factors.

**Unresolved question, not established failure:** Under adequate coverage of the relevant local dynamics, does training-induced invariance prevent reliable decisions across changing object roles, even with strong spatial/object features, legitimate query conditioning and calibrated scoring? If so, what determines the tradeoff among task coverage, precision, robustness and resources?

### Smallest decisive study

First use existing checkpoints for diagnosis: counterfactually move only one object; inspect encoded geometry; decode actual and predicted states separately; compare candidate rankings with realized endpoints. Whitening must be fit on training data with regularization, not tuned on the evaluation goals.

Then construct a role-change setting in which all objects and required interaction types are familiar during training. Keep appearances and physics fixed at evaluation. Change which object or trajectory property matters. Include genuine visual nuisance separately.

Use a small number of well-qualified baselines: current LeWM, a spatial model, and a strong structured or task-aware alternative. First compare defaults; only then make matched resource claims. Do not deliberately cripple the stronger baseline's representation.

For a fixed bank with useful actions, compare native model predictions and costs, real endpoints encoded with the same cost, and real physical endpoints with the intended task cost. The first substitution measures prediction headroom; the second changes both representation and scoring and must not be called a pure encoder test. A privileged state head is diagnostic, not an equal-information deployment method.

**Advance** if a substantial role-change deficit survives adequate acquisition, scoring fixes and strong existing representations, or if the only working remedy has a consequential measured cost.

**Reject the broad P1 story** if ordinary training coverage, whitening, goal specification or a standard model resolves the deficit cheaply. A pipeline diagnosis could still be useful, but would not justify the proposed fundamental compression claim.

## 2. Can a planner trust the plan it has optimized?

### Precise question and significance

Can a learned world model provide useful estimates of **decision error on the actions its planner selects**, as search effort and goals change, without requiring excessive corrective interactions or becoming too conservative to act?

Prediction tests average over some distribution of actions. Planning actively seeks unusually favorable predictions. Its chosen action may therefore expose errors that average accuracy or confidence does not reveal. This is the established model-exploitation problem; rediscovering that principle is not a contribution.

The possible research value lies in identifying when modern latent planning and its strongest remedies retain useful reliability under this selection process, and what feedback or representational independence is required.

### Closest work: both negative evidence and working answers

| Work | Evidence or answer | Remaining boundary |
| --- | --- | --- |
| [Biased Dreams](https://arxiv.org/html/2604.25416) | RSSM experiments find that ensemble disagreement can decrease while rollout error increases, associated with attraction toward familiar latent regions | Direct mechanism evidence, but not a general result for every latent architecture or a sweep of CEM optimization intensity. Its external latent ensemble and partly qualitative attractor analysis matter. |
| [RWM-U](https://arxiv.org/html/2504.16680) | Uncertainty-aware autoregressive modeling plus pessimistic policy learning enables real robotic control | Strong positive counterevidence to “ensembles do not work.” Its proprioceptive setting differs; both insufficient and excessive penalties hurt behavior. |
| [RP1](https://arxiv.org/html/2608.18669) | Documents exploitation of imagined successful grasps and improves performance through planner-collected corrective data | Demonstrates both a failure and a working repair. Repeated-feedback cost and recurrence under changed planning remain relevant measurements. |
| [World-Coherent Decoding](https://arxiv.org/html/2609.02159v1) | Uses internal signals and executed imagination–reality mismatch to improve future selection | Directly occupies reliability-aware candidate selection and online feedback. Gains under its limited-randomization setting are not evidence of universal transfer across search procedures. |
| [HAUWM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad222e6e38e26e3f00d85829b9e02f43-Abstract-Conference.html) | Trains horizon-dependent predictive uncertainty and reports downstream control benefits | “Uncertainty should reflect horizon” is already an explicit learning objective. Monotone variance alone is not a selected-action calibration test. |
| [SufficientPlan](https://arxiv.org/html/2610.08350) | Certifies reduced planning budgets against a full-budget baseline on a fixed development distribution | Budget selection already has a practical answer. Matching the full-budget system does not establish that its predictions are physically correct; distribution changes may require recalibration. |
| [CERT-Replan](https://arxiv.org/html/2610.09302) | Uses calibrated obstacle-prediction risk to trigger changes of planning mode | Adaptive risk-triggered replanning is also occupied. Its structured navigation interface differs from generic image-goal latent planning. |

Older foundations include [PETS](https://arxiv.org/abs/1805.12114), [MOPO](https://arxiv.org/abs/2005.13239), [MOReL](https://arxiv.org/abs/2005.05951), and [COMBO](https://arxiv.org/abs/2102.08363). Statistical precedents include [action-conditional conformal decision guarantees](https://arxiv.org/abs/2606.05551) and [online conformal selection with limited feedback](https://arxiv.org/abs/2605.14953). Consequently, neither selection-aware uncertainty nor reducing feedback is intrinsically novel. Their assumptions and guarantees must be examined before proposing a theorem or method.

**Unresolved question, not a new general principle:** Is there a reproducible, practically costly failure of these remedies when search selects actions using learned visual representations—particularly errors shared by predictor and verifier—and can it be explained beyond ordinary lack of action coverage?

### Smallest decisive study

Use an already competent model and simulator. Hold rollout horizon fixed while varying candidate count and search rounds; investigate horizon separately. Record random candidates, elites and selected actions. Execute audited alternatives from identical simulator states.

Measure physical task regret, realized progress, success and prediction optimism. If a score claims to be a probability, measure calibration on selected actions; do not call arbitrary ensemble variance a probability. Separate factual one-step error from multistep and selection errors.

Use nested fixed candidate banks for one diagnostic: the true best candidate cannot worsen when the bank grows, so a worse selected outcome reveals a selection failure. Adaptive CEM is a separate test because its banks need not be nested.

Compare shared-encoder head ensembles with independently trained representations, simple support penalties, held-out residual calibration, shorter replanning intervals and a small matched corrective-data budget. Budget every executed alternative used as a label; distinguish privileged audit access from deployable feedback. Examine both frozen calibration transfer and matched recalibration, rather than banning adaptation to manufacture a gap.

Shared blindness is a hypothesis. If the relevant evidence is absent from every input to a verifier, no calibration method can reconstruct it by assumption alone; that elementary impossibility would not be a new contribution.

**Advance** if inexpensive standard remedies fail in a repeatable, mechanistically identifiable way, while useful candidate actions exist, and reliability can be improved without sacrificing most task progress.

**Reject** if ordinary calibration or replanning resolves the problem, if the effect occurs only under an unusably weak model, or if it is wholly explained by known coverage failure without a meaningful new cost or mechanism result.

This is the best immediate research bet because the experiment can separate prediction, selection, confidence and feedback costs. It is not the least crowded literature.

## 3. Can familiar interactions compose into unfamiliar causal chains?

### Precise question and significance

After learning each required local interaction, can a model predict and choose actions for **new sequences of those interactions**, without training on every complete sequence?

For example, pushing A makes it contact B, which moves C out of a passage. The interest is reuse of learned mechanisms. Showing that three events are harder than one is insufficient: ordinary accumulated error and missing contact configurations are obvious explanations.

### Closest prior work

| Work | Existing answer or evidence | Implication |
| --- | --- | --- |
| [Graph Network-based Simulators](https://arxiv.org/abs/2002.09405) | Learn local physical dynamics and generalize to longer rollouts and larger particle systems; noise training reduces accumulated error | Local physical composition and graph-based remedies are longstanding. Structured-state access is a powerful diagnostic control. |
| [HOWM](https://proceedings.mlr.press/v162/zhao22b.html) | Formalizes object-replacement compositional generalization and learns object/action binding | Novel object combinations are already directly studied. This differs from, but constrains, a claim about dependent contact chains. |
| [The Seriality Gap in Video Diffusion Models](https://arxiv.org/html/2607.13031) | Isolates degradation across dependent collision events and finds benefits from autoregressive generation; theory concerns specified deterministic prediction assumptions | The causal-chain framing and serial-computation explanation are already theirs. LeWM is autoregressive, so this result cannot simply be transferred to explain its failures. |
| [Causal-JEPA](https://proceedings.mlr.press/v306/nam26c.html) | Learns interaction-dependent object representations | Merely adding objects, masks or relations is not enough to establish novelty. |
| [World Action Planner](https://arxiv.org/abs/2607.27599) | Reports generalization through world-model-based search on compositional robot tasks | Compositional task planning already has positive evidence; investigate a specific mechanism rather than claiming general absence. |
| [RoleWorld](https://arxiv.org/html/2610.13172) | Reuses symbolic interaction rules through functional role abstractions and updates role assignments from feedback | Even functional-role transfer has a recent direct answer. Its symbolic LLM environment is not a visual contact-dynamics test. |

**Unresolved question:** Under verified local interaction coverage, what causes useful decision quality to deteriorate with dependency structure in learned visual dynamics? Does the remaining limitation survive standard autoregressive, object-relational and rollout-training remedies?

The reviewed papers do not establish this failure in LeWM or DINO-WM. This candidate has the highest risk of turning into a rediscovery of accumulated error or an application of an existing graph model.

### Smallest decisive study

Construct controlled scenes crossing causal dependency depth with elapsed rollout time. Compare equal-duration chains and independent contacts, and equal-depth sequences with extra waiting. Match contact count, state precision, object count and action difficulty where possible; disclose factors that cannot be matched.

Verify that every necessary pairwise contact type, relative velocity and approach geometry is supported by training. Observing each object separately is insufficient. Include direct interaction tests and perturbation tests near contact boundaries.

Compare a visual autoregressive baseline with a learned relational model receiving privileged object state, then a visually inferred relational counterpart if the first comparison reveals useful headroom. The state model locates perception/representation costs; it is not an equal-input competitor. Include rollout noise/multistep training before attributing failures to missing compositional structure.

Test intermediate state resets: if supplying the true state after each event removes the failure, ordinary error accumulation is implicated. Quantify whether measured local errors already predict the observed failure rate; resets alone do not establish a novel mechanism. Evaluate action choice with a fixed candidate bank before adding search.

**Advance** only if a consistent dependence-specific deficit remains after these controls and affects decisions, with evidence beyond one brittle contact setup.

**Reject** if standard structured dynamics solve the setting, if local interactions were actually out of distribution, or if the results add no explanation beyond familiar numerical sensitivity and rollout drift.

## Why other directions did not enter the top three

- **Generic action-effect correction (historical P2):** [The Intervention Gap](https://arxiv.org/abs/2608.29998) is a diagnostic precedent, while [AD-WM](https://arxiv.org/abs/2609.30264), [D-JEPA](https://arxiv.org/abs/2609.24749), and [RC-aux](https://arxiv.org/abs/2605.07278) already provide direct repairs. Retain them as controls. A new inverse head is not a problem statement.
- **Continual dynamics reuse:** [Benchmarking World Models for Continual Learning on Compositional Tasks](https://arxiv.org/abs/2609.22055) already separates adaptation from knowledge reuse. This remains important but needs a stronger specific residual question before displacing the shortlist.
- **Memory and useful probing:** POMDP belief estimation and dual control are mature. [Belief-based interfaces for LLM agents](https://arxiv.org/abs/2609.00455) and [budgeted environment probing](https://arxiv.org/abs/2606.31422) add recent overlap. No sharper feasible visual-control gap was established in this review.
- **Long-horizon image-goal failure:** [The Planning Limits of Latent World Models](https://arxiv.org/abs/2609.39235) already implicates the relationship between goal distance and planning horizon, including exact-dynamics controls. This is essential differential diagnosis, not a fourth new problem.

## Recommended next decision

Give P1 a cheap diagnostic gate using the existing work, then make plan reliability the first new empirical screen. Keep causal composition as an independent, higher-risk alternative. Do not launch three large training campaigns.

Before selecting a contribution, require one page that states the capability, strongest existing answer, residual failure, simplest explanation, and a test that can reject the claim. A mechanism study could itself be a contribution; architecture is warranted only when evidence localizes a limitation that simpler alternatives cannot meet economically.

A single LeWM failure does not establish a field-wide limit. Require competent baselines, independent evaluation scenes, multiple seeds when fitting models, legitimate goal information, measured resources and a second setting proportional to the eventual generality claim. The amount of evidence should grow with the claim, not with a preselected paper title.

## Reading and evidence scope

This review used the supplied proposal and relevant local implementation, the previous canonical review as a search index, and fresh primary-source searches through October 16, 2026. It did not rescan private X bookmarks. Claims are author-reported unless explicitly identified as local code observations or proposed experiments.

Targeted main-text/method/limitation reads covered the query-sufficiency study, Seriality Gap, Biased Dreams, World-Coherent Decoding, RWM-U, ATLAS and SufficientPlan; HOWM's formal setting was checked in its PDF. Causal-JEPA's proceedings abstract was accessible, but its linked PDF failed in this pass. HAUWM, World Action Planner, the two conformal-selection papers and some rejected-direction references were screened at the primary abstract/proceedings level; their complete assumptions have not been audited. These are novelty warnings and baseline leads, not reproduced results or assurances that implementations are ready.

An omitted evaluation in any of these works is an unknown. The proposed residual questions remain hypotheses until matched tests show that strong existing methods leave a consequential problem.
