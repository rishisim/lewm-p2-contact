# V5 adaptive-computation novelty review

Date of review: 2026-07-16

## Executive conclusion

The completed research program has a plausible and compelling novelty claim, but it should not be framed as the first use of adaptive computation in a world model.

The broad ingredients already exist separately:

- adaptive depth and early exit are established in neural networks;
- fixed-budget dynamic routing is established in language models;
- learned systems can predict whether more inference-time computation will help;
- world-model systems already adapt rollout length, imagination frequency, or token sparsity;
- LoopWM explicitly proposes adaptive inner-loop depth for a world model;
- adaptive neural PDE solvers allocate iterative refinement non-uniformly at equal compute.

The likely novel combination is:

> Predicting the transition-conditional marginal value of another refinement step in an action-conditioned learned world model, then demonstrating that a non-anticipatory gate uses this signal to outperform the best transition-independent allocation over the same refinement depths at exactly matched total compute.

If this is generalized across environments and architectures, survives multi-step rollouts, improves planning or control, and produces a practical accuracy-compute or accuracy-latency benefit, the stronger paper-level contribution becomes:

> A general empirical demonstration that computation in learned world models is a transition-dependent resource whose marginal value is predictable from causally available information and can be exploited to improve prediction and decision making under a fixed inference budget.

The sharpest distinctive test is not merely that an adaptive gate works. It is the intervention represented by the histogram-randomized comparator:

> Holding the amount and distribution of refinement computation fixed, changing which transitions receive it changes performance.

That establishes that transition-conditioned allocation itself carries value.

## What V5 currently establishes

V5 evaluates a frozen visual latent world model and four-exit residual refiner under a fixed Cube PlanOracle data-generating process.

For each transition and reached refinement stage, a stage-specific dual linear gate receives 1,046 non-anticipatory features constructed from pixel-derived history latents, normalized actions, the current prediction, the most recent update, and scalar summaries. It never receives the prediction target, future observations, contact labels, reward, success, or privileged simulator state.

The gate was fit to predict the next-stage reduction in two losses:

- raw latent mean-squared error;
- a frozen, PlanOracle-native whitened latent mean-squared error.

Each head predicts standardized marginal gain in one metric, and the deployed score is the minimum of the two predicted gains. Thus, the policy is a conservative multi-metric estimate of the benefit of another refinement.

The V5 confirmation used:

- 1,600 fresh episodes;
- 60,800 modeled transitions;
- a preregistered, frozen gate and analysis;
- exact sparse execution of only the selected later refiners;
- complete gate-overhead FLOP accounting;
- 20,000 paired episode bootstrap replicates;
- an independent reimplementation and audit.

At a mean of 1.2167 refiner calls per transition, the adaptive policy beat:

- the best transition-independent analytic mixture at exactly equal total FLOPs;
- a seeded integer mixture using weakly more FLOPs;
- fixed depth 1;
- a within-episode random reassignment preserving every episode's exact call histogram.

Against the exact-FLOP analytic comparator, the confirmed improvements were:

- raw latent MSE: `6.769e-6`, about `0.197%` of comparator loss;
- native-whitened latent MSE: `6.086e-4`, about `0.612%` of comparator loss.

The effects are statistically convincing but modest. The synchronized MPS measurements do not currently show a wall-clock speedup over fixed depth 1; routing and sparse-execution overhead make the adaptive path slower in the measured batches. V5 therefore supports an allocation-quality claim, not yet a deployment-efficiency claim.

Local evidence:

- [V5 preregistration](../runs/lewm_v5_readiness_program/v5_package_versions/v004/PREREGISTRATION.md)
- [V5 terminal decision](../runs/lewm_v5_readiness_program/v5_package_versions/v004/decision.json)
- [V5 latency measurements](../runs/lewm_v5_readiness_program/v5_package_versions/v004/metrics/v5_confirmation_latency.json)
- [Gate fit implementation](../runs/lewm_v5_readiness_program/cycles/cycle_006_stage_specific_gate_refit/fit_select_gate.py)
- [Gate fit/selection preregistration](../runs/lewm_v5_readiness_program/cycles/cycle_006_stage_specific_gate_refit/FIT_SELECTION_PREREGISTRATION.md)

## Closest prior work

### 1. Adaptive computation and early exit

Adaptive Computation Time introduced learned variable computation within recurrent models. Universal Transformers added dynamic per-position halting to a recurrent-depth transformer. PonderNet learned a distribution over computation steps balancing accuracy and cost. BranchyNet and SkipNet showed that inputs can exit early or skip blocks based on confidence or learned gates.

These works establish that input-dependent depth is not itself novel.

Primary sources:

- [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
- [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407)
- [BranchyNet](https://arxiv.org/abs/1709.01686)
- [SkipNet](https://arxiv.org/abs/1711.09485)

### 2. Fixed-budget compute routing

Mixture-of-Depths routes a fixed number of tokens through transformer blocks, making total computation predictable while assigning it contextually. This is an important conceptual predecessor to V5's claim that a fixed compute budget should be allocated non-uniformly.

The distinction is the allocation unit and objective: tokens and language-model loss versus physical transitions and next-state refinement gain.

- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)

### 3. Predicting the value of more computation

Several language-model papers learn whether additional inference will improve an answer. Manvi et al. predict whether restarting or sampling again is likely to yield a better response. Re-FORC forecasts reward as a function of additional reasoning compute. Constrained allocation work learns an oracle-derived per-input compute policy and beats uniform allocation under matched budgets.

These papers mean that "predicting whether more computation will help" is not a domain-general novelty claim.

The V5 distinction is applying this idea to the marginal error reduction of successive internal world-model refinements for individual action-conditioned transitions.

- [Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better](https://arxiv.org/abs/2410.02725)
- [Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization](https://arxiv.org/abs/2604.14853)
- [Re-FORC OpenReview PDF](https://openreview.net/pdf/895e9b349dbe75c3f24511cd3cf86eed67affa3d.pdf)

### 4. Looped World Models

LoopWM is the most direct overlap. It introduces a parameter-shared recurrent transformer dynamics core for world modeling and proposes a learned hidden-state exit gate that can terminate inner refinement early for simpler transitions.

This prevents any claim that V5 is the first adaptive-depth world model or the first world model with a learned halting gate.

However, the current LoopWM paper and V5 answer different scientific questions:

- LoopWM primarily proposes a looped architecture, spectral stability, parameter efficiency, and deferred decoding.
- Its exit gate is trained jointly with the model using entropy regularization and is used as a learned hidden-state halting probability.
- The paper describes large possible FLOP savings, but its reported results section does not provide V5's controlled test against the best transition-independent exact-total-compute mixture.
- It does not appear to include a call-histogram reassignment test demonstrating that transition identity, rather than only average depth, creates the gain.
- It evaluates text-based ScienceWorld and ALFWorld tasks, whereas V5 uses pixel/action-derived latent physical transitions.

Thus LoopWM substantially narrows, but does not eliminate, the proposed contribution.

- [Looped World Models](https://arxiv.org/pdf/2606.18208)

### 5. Compute-quality regimes in latent world models

The July 2026 preprint *Adaptive Compute in Latent World Models: When Depth Helps, Hurts, or Doesn't Matter* is highly relevant. It evaluates fixed early exits across nine DeepMind Control tasks and shows that deeper prediction helps rollouts on some tasks, hurts on others, and is neutral on another. It also connects the depth regime to CEM planning.

This paper directly studies whether predictor depth has useful variation across tasks and training regimes. It does not, however, demonstrate a learned per-transition router or show that transition-conditioned allocation beats transition-independent allocation at the same total compute.

The two projects are complementary:

- that work asks whether a task/model pair has a usable depth-quality tradeoff under composition;
- the proposed V5 program asks whether within-task marginal refinement value varies across transitions, can be predicted online, and can be exploited under a budget.

- [Adaptive Compute in Latent World Models](https://arxiv.org/pdf/2607.10203)

### 6. Sparse Imagination

Sparse Imagination reduces the number of visual patch tokens processed during world-model planning and demonstrates substantial planning-time savings while maintaining control performance. It supports resource-adjustable world-model inference, but the sparsity is primarily a configured resource level with randomized token selection rather than a learned decision about which physical transitions deserve more refinement depth.

- [Sparse Imagination for Efficient Visual World Model Planning](https://arxiv.org/abs/2506.01392)
- [ICLR 2026 paper](https://openreview.net/pdf?id=faxcxKINBC)

### 7. Prioritized imagination

Dynamic World Simulation uses "prioritized imagination" to sample more rollout starting states with high TD loss. This is an important example of concentrating world-model resources on valuable states rather than using them uniformly.

It allocates the frequency of imagined rollouts for policy learning, not internal computation within an individual transition prediction. Its value signal is downstream TD loss rather than the marginal predictive gain of another refinement step.

- [Pre-Trained Video Generative Models as World Simulators](https://arxiv.org/abs/2502.07825)

### 8. Adaptive world-model rollout length

Several MBRL methods adapt how far the model is trusted or rolled forward:

- Adaptive Rollout Length treats rollout horizon as a meta-level control problem.
- CRLA truncates imagined rollouts when actions become unreliable.
- MACURA uses local model uncertainty to terminate rollouts.
- TATU truncates uncertain trajectories in offline MBRL.

These demonstrate transition- or state-dependent control of the number of world-model calls. They do not vary the internal computational depth used to produce a single next-state prediction.

- [Adaptive Rollout Length for Model-Based RL](https://arxiv.org/abs/2206.02380)
- [Conservative Rollout Length Adaptation](https://openreview.net/forum?id=cYksYKbf6K)
- [MACURA](https://arxiv.org/abs/2405.19014)
- [TATU](https://arxiv.org/abs/2304.04660)

### 9. Adaptive neural PDE refinement

Adaptive Test-Time Compute Allocation for Neural PDE Solvers is the closest result outside conventional world models. It trains a difficulty estimator to predict local solution error, allocates variable iterative refinement to difficult spatial regions, and reports lower error at equal compute across four PDE families.

This rules out broad claims such as:

> We are the first to show that learned physical predictors can allocate iterative refinement adaptively.

The remaining distinction is substantial but domain-specific:

- spatial regions of a PDE field versus sequential action-conditioned transitions;
- local error estimation versus next-refinement marginal gain;
- neural operator solution refinement versus learned world-model dynamics;
- no world-model rollout/planning/control claim versus decision-making with an internal environment model.

- [Adaptive Test-Time Compute Allocation for Neural PDE Solvers](https://openreview.net/forum?id=xLrA907jVi)
- [ICLR 2026 paper PDF](https://openreview.net/pdf/9e77aaae209219ffc8f86c9fd0e0cddbdc16aedc.pdf)

### 10. Dynamic physical simulators and graph computation

Adaptive Message Passing learns variable message-passing depth, while EvoMesh learns input-conditioned graph hierarchies for physical simulation. These adapt spatial information propagation or graph structure, not refinement depth across sequential world-model transitions.

- [Adaptive Message Passing](https://openreview.net/forum?id=YWLWUTtVF3)
- [EvoMesh](https://openreview.net/forum?id=ZZvTc92dYQ)

### 11. Sparse latent-state updates

Variational Sparse Gating selectively updates components of a world model's latent recurrent state. It is relevant because it introduces gating into world-model dynamics, but its gate controls which latent state components update, not how many refinement computations a transition receives.

- [Learning Robust Dynamics through Variational Sparse Gating](https://openreview.net/forum?id=460hxFeWzyr)

### 12. Convergence-controlled iterative models

Deep Equilibrium Models and Neural ODE solvers naturally use variable numbers of iterations or function evaluations based on numerical convergence and error tolerances. They demonstrate that input-dependent numerical effort is well established.

V5 differs by learning a predictor of task loss improvement rather than following a generic convergence or integration-error rule.

- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

### 13. Rational metareasoning

The theoretical idea behind the project is closely related to value of computation: an internal computation should be performed when its expected improvement exceeds its cost. This has a long history in rational metareasoning.

The proposed paper can use this as motivation, but should not present marginal value of computation as a new concept. Its contribution is an empirical and algorithmic instantiation for learned world-model transition refinement.

- [Principles of Metareasoning](https://doi.org/10.1016/0004-3702(91)90015-C)
- [Learning to Select Computations](https://arxiv.org/abs/1711.06892)
- [Computing the Value of Computation for Planning](https://arxiv.org/abs/1811.03035)

## Feature comparison

| Work | Action-conditioned world model | Variable internal depth per transition | Predicts marginal benefit of next refinement | Exact matched-budget comparison | Proves transition identity matters | Multi-environment | Planning/control consequence | Actual latency gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACT / PonderNet / early exit | No | Yes | Usually no | Sometimes cost-aware | No | Yes | No | Sometimes |
| Mixture-of-Depths | No | Token-level | No | Yes | Token routing matters | Yes | No | Yes |
| Adaptive LLM compute | No | Query/reasoning-level | Yes in closest work | Yes in some work | Query identity matters | Yes | No | Sometimes |
| Adaptive PDE allocation | Physical solver, not conventional WM | Spatial refinement | Predicts error/difficulty | Yes | Spatial allocation matters | Four PDEs | No agent control | FLOP savings |
| Sparse Imagination | Yes | Token sparsity, not depth | No | Resource curves | Mostly random token subsets | Several tasks | Yes | Yes |
| Prioritized imagination | Yes | Number of imagined rollouts | TD-based priority | Not the same test | Starting-state value matters | Several tasks | Yes | Not central |
| MACURA / adaptive rollout work | Yes | Rollout horizon, not predictor depth | Uncertainty proxy | Not predictor-depth parity | State-dependent trust matters | Yes | Yes | Not central |
| LoopWM | Yes | Yes | Learned halt probability, not explicitly marginal gain | Not demonstrated in the V5 form | Not isolated by reassignment | Two text environments | Limited | Claimed potential |
| Depth-regime study | Yes | Evaluates fixed exits | No learned router | Fixed-depth compute curves | No | Nine DMC tasks | Partial CEM evidence | No |
| Current V5 | Yes | Yes | Yes | Yes | Yes | No | No | No |
| Intended full paper | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Targeted |

## Novelty boundary

### Claims that are not safe

- "We introduce adaptive computation."
- "We are the first model to learn when to think longer."
- "We introduce the first adaptive world model."
- "We are the first to use a gate in a world model."
- "We are the first to allocate computation non-uniformly for physical prediction."
- "We are the first to vary world-model computation according to state or transition difficulty."

### Claims that may be safe after a final exhaustive citation check

Current controlled claim:

> To our knowledge, this is the first exact-compute-controlled demonstration that a non-anticipatory, transition-conditioned gate can improve a learned visual world model by predicting the marginal value of additional refinement.

Stronger claim emphasizing the decisive comparator:

> To our knowledge, we provide the first evidence in a learned world model that, while holding the total amount and per-episode distribution of refinement computation fixed, assigning that computation to different transitions measurably changes predictive performance.

Completed generalization claim:

> We show across diverse action-conditioned world models and physical environments that the marginal value of iterative refinement is heterogeneous across transitions, predictable from information available at inference time, and exploitable to improve prediction and control under fixed compute budgets.

Value-of-computation framing:

> We formulate adaptive world-model inference as amortized value-of-computation estimation: at each refinement stage, a lightweight gate predicts the loss reduction expected from one more computation and continues only where that computation has high marginal value.

Retrofitting claim, if validated across pretrained models:

> Unlike jointly trained halting architectures, our method retrofits a frozen multi-exit world model with a lightweight gain predictor, separating the scientific question of compute allocation from changes to the underlying predictor.

## Recommended headline contribution

The strongest paper should not lead with "adaptive computation for world models," because LoopWM already occupies that phrase.

It should lead with one of:

- **marginal value of refinement**;
- **value-of-computation gating**;
- **transition-conditioned compute allocation**;
- **fixed-budget world-model refinement**.

Recommended central statement:

> Existing adaptive-depth systems generally halt using confidence, uncertainty, or latent convergence. We instead estimate the marginal predictive value of the next computation. Across action-conditioned world models and physical environments, this value varies across transitions, can be predicted without future information, and supports better fixed-budget allocation than fixed depth, random allocation, uncertainty-based routing, and convergence-based early exit. The resulting gains persist in multi-step rollouts and improve planning or control.

## Experiments needed for the full claim

### A. Establish heterogeneous value robustly

- Measure the full per-transition gain distribution at every exit.
- Report the fraction of positive, near-zero, and negative gains.
- Show heterogeneity within the same environment, not only differences between environments.
- Separate contact, near-contact, free motion, occlusion, rapid acceleration, and other regimes post hoc.
- Demonstrate that heterogeneity is not created only by early-exit supervision.
- Evaluate whether gain survives autoregressive composition, following the caution raised by the depth-regime study.

### B. Establish predictability

- Use entirely held-out episodes, seeds, objects, and dynamics.
- Report ranking, calibration, and decision utility, not only regression error.
- Compare marginal-gain prediction with:
  - model confidence;
  - ensemble uncertainty;
  - prediction/update norm;
  - latent convergence;
  - reconstruction residual;
  - action magnitude;
  - contact heuristics;
  - oracle future gain.
- Clarify that "causal" means non-anticipatory information flow unless a formal causal-inference design is added.

### C. Establish useful allocation

- Compare against the best fixed depth at every budget.
- Compare against the best transition-independent mixture over all available depths.
- Preserve and randomize exact call histograms globally and within episodes.
- Include learned uncertainty and convergence routers.
- Include an oracle gain allocator as an upper bound.
- Sweep several budgets and report the accuracy-compute Pareto frontier.
- Count encoder, gate, routing, memory movement, and synchronization costs consistently.

### D. Establish generalization

At minimum, vary three axes rather than only accumulating similar tasks:

- environments: Cube, PushT/contact manipulation, locomotion or navigation;
- dynamics: unseen mass, friction, damping, action noise, objects, and contact geometry;
- architectures: stacked residual exits, shared recurrent/looped refinement, and a second latent world-model family.

Strong evidence would include both transfer settings:

- refit the small gate in each new domain while keeping the allocation principle fixed;
- zero-shot or lightly adapted transfer of one gate across related domains.

The first establishes mechanism generality; the second establishes gate generality. They should not be conflated.

### E. Establish meaningful consequences

- Multi-step latent and observation-space rollout error.
- Planning-quality metrics using identical planners and candidate budgets.
- Downstream task return or success rate.
- Failure analysis showing when lower prediction error does and does not affect decisions.
- Tests at planning horizons where compounding error can reverse shallow/deep rankings.

### F. Establish practical efficiency

- Wall-clock benchmarks on GPU and CPU with batch sizes representative of online control and batched planning.
- A grouped or bucketed sparse kernel to reduce dynamic-routing overhead.
- End-to-end latency including encoder and planner.
- Energy measurement where a reliable interface exists.
- Accuracy versus FLOPs, latency, and energy as separate Pareto curves.
- A result that is useful in either direction:
  - same quality with less real cost; or
  - better quality at the same real cost.

## Recommended paper claims by evidence level

### If only the current V5 result is available

> We provide a preregistered proof of concept that transition-conditioned marginal-gain gating can improve one-step latent prediction over transition-independent allocation at exact FLOP parity in one controlled visual world-model setting.

### If cross-environment prediction generalizes

> We establish that marginal refinement value is a recurring property of action-conditioned world models and that learned transition-level allocation improves their prediction-compute frontier across diverse physical environments.

### If rollouts and planning improve

> We establish transition-conditioned computation as a useful world-model inference mechanism: it improves one-step prediction, preserves the advantage under autoregressive composition, and yields better planning or control at matched compute.

### If real latency or energy also improves

> We demonstrate a practically efficient adaptive world model that improves both predictive or control performance and real inference cost by allocating refinement according to its predicted marginal value.

## Overall assessment

Assuming all six planned pillars are convincingly demonstrated, this is novel enough to support a serious ICLR, ICML, or NeurIPS submission.

The contribution would not be the invention of adaptive computation, iterative refinement, or adaptive world models. It would be the first strong, controlled, general evidence for a particular principle:

> In learned world models, computation has transition-specific marginal value, and estimating that value is a better allocation rule than treating depth, uncertainty, confidence, or convergence alone as the decision variable.

The result becomes especially compelling if the paper shows that:

1. the effect is present within environments and transfers across them;
2. the gate predicts actual counterfactual next-stage gain;
3. exact-budget and histogram-preserving controls isolate allocation from compute quantity;
4. the advantage survives multi-step composition;
5. it improves decisions;
6. an implementation converts the FLOP advantage into latency or energy benefit.

That complete chain is not supplied by any single work found in this review.

## Review limitation

This is a research novelty audit, not a legal priority search. It covers primary papers and public preprints located through targeted searches up to 2026-07-16. Because several directly relevant papers appeared in June and July 2026, the literature should be searched again immediately before submission, and any "first" wording should remain qualified with "to our knowledge."
