# Focused literature positioning

Reviewed 2026-07-18. This is a focused primary-source review of the claim
boundary, not a systematic review. It supports no priority or “first” claim.

## What the literature already establishes

| Primary source | Relevant mechanism | Consequence for LeWM positioning |
|---|---|---|
| Graves, [*Adaptive Computation Time for Recurrent Neural Networks*](https://arxiv.org/abs/1603.08983) (2016) | Input-dependent recurrent updates with a differentiable halting mechanism and ponder cost. | Adaptive recurrent depth and a quality/compute penalty are longstanding, not LeWM contributions. |
| Banino, Balaguer, and Blundell, [*PonderNet: Learning to Ponder*](https://arxiv.org/abs/2107.05407) (2021) | A probabilistic distribution over halting steps, trained with a task loss and a prior over computation. | Learned instance-wise halting is occupied; PonderNet does not by itself supply an exact finite-cohort global budget. |
| Bolukbasi et al., [*Adaptive Neural Networks for Efficient Inference*](https://proceedings.mlr.press/v70/bolukbasi17a.html) (ICML 2017) | A global objective trading prediction utility against computation and a learned decision based on expected future reward. | The Lagrangian “gain minus compute price” view is prior art. It can clarify LeWM but is not a new formulation in the broad adaptive-inference literature. |
| Huang et al., [*Multi-Scale Dense Networks for Resource Efficient Image Classification*](https://arxiv.org/abs/1703.09844) (2017) | Anytime prediction and budgeted-batch classification, where a global budget is distributed unevenly across examples using confidence. | A quality-versus-compute frontier and shared finite batch budget are not distinctive by themselves. |
| Raposo et al., [*Mixture-of-Depths*](https://arxiv.org/abs/2404.02258) (2024) | Per-layer token routing with a fixed top-k capacity and predictable aggregate compute. | Exact top-k resource allocation is a close fixed-budget analogue, although its units are tokens at transformer layers rather than latent-dynamics transitions. |
| Valade et al., [*Early-Exit with Reject Option for Efficient Classification with Limited Budget*](https://proceedings.mlr.press/v286/valade25a.html) (UAI 2025) | Cost-aware early exit with a reject option, Bayesian risk, head costs, and a fixed-budget guarantee. | Modern early-exit work already treats heterogeneous head costs and explicit budgets; LeWM should not claim a new general budget theory. |
| Geiping et al., [*Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*](https://arxiv.org/abs/2502.05171) (2025) | A parameter-shared recurrent block that supports flexible test-time depth and continued quality gains with more recurrence. | Recurrent depth as a test-time-compute axis is occupied; this work is primarily about scaling depth, not matched-budget allocation across transitions. |
| Bae et al., [*Mixture of Recursions*](https://papers.nips.cc/paper_files/paper/2025/hash/8b08bbf8b420faa6eeb4020720582ec7-Abstract-Conference.html) (NeurIPS 2025) | Lightweight routing of token-specific recursive depths through a shared stack. | Dynamic recursive depth is not new; the relevant distinction is the routed object and the matched-compute empirical test. |
| Jeddi et al., [*LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation*](https://arxiv.org/abs/2602.11451) (2026) | Elastic recurrent depth and budget-conditioned latent reasoning trajectories. | Flexible loop depth and budget-conditioned recurrence are current active topics, again outside visual next-latent allocation. |

## World-model and imagination neighbors

| Primary source | Relevant mechanism | Boundary relative to this study |
|---|---|---|
| Lu et al., [*Looped World Models*](https://arxiv.org/abs/2606.18208) (LoopWM, 2026) | A parameter-shared looped world model with a learned exit gate and adaptive numbers of world-model loops. | This directly occupies broad “adaptive-depth world model” territory. The apparent distinction here is the matched counted-compute allocation test on visual latent transitions, not priority for adaptive world models. |
| Sivasankar, [*Adaptive Compute in Latent World Models: When Depth Helps, Hurts, or Doesn't Matter*](https://arxiv.org/abs/2607.10203) (2026) | Fixed exits and depth regimes across latent world models in nine DeepMind Control tasks, emphasizing when added depth helps or hurts. | Very close evidence about latent-world-model depth; according to the primary paper’s described design it does not test a learned transition-specific router. |
| Hamrick et al., [*Metacontrol for Adaptive Imagination-Based Optimization*](https://arxiv.org/abs/1705.02670) (2017) | A metacontroller allocates imagined simulations and model choices using reliability and computational cost. | Cost-aware adaptive imagination predates LeWM. It allocates planning simulations rather than one-step latent refinement. |
| Chun, Jeong, and Kim, [*Sparse Imagination in World Models*](https://arxiv.org/abs/2506.01392) (2025/ICLR 2026) | Varies token sparsity and imagination compute for world-model planning. | It allocates planning/imagination resources, not the depth of a single causal transition predictor. |
| Yu et al., [*When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning*](https://arxiv.org/abs/2602.08236) (2026) | Selective world-model invocation and adaptive imagination length with computation penalties. | Current work already occupies selective world-model-call allocation. Its setting is external world-model use for visual-spatial reasoning, not matched-FLOP latent-transition refinement. |

## Precisely defensible distinctiveness

The strongest defensible distinction is a **combination of empirical design
choices**, not a new resource-allocation theorem:

1. the routed unit is a visual, action-conditioned next-latent transition;
2. the allocator is non-anticipatory at the prediction boundary;
3. the arithmetic ledger charges base prediction, every executed refinement,
   gate-feature construction, and dual affine scoring;
4. the primary comparator is the endpoint-specific lower convex envelope of
   transition-independent fixed-depth arms at the same counted total;
5. transition-specific value is tested on disjoint frozen confirmations, while
   the failed four-depth PushT pilot remains in the chronology; and
6. the paper explicitly separates predictive fidelity from latency, planning,
   control, and cross-environment transfer.

This combination is cleaner than saying merely “we add an early-exit gate.” It
is still an empirical matched-budget study rather than a general adaptive-world-
model framework.

## Claims that are not supportable

- first adaptive world model;
- first recurrent or elastic-depth world model;
- first compute-price or marginal-value stopping rule;
- first exact global-budget or top-k adaptive-compute method;
- first quality-versus-compute frontier;
- first cost-aware adaptive imagination or selective world-model call policy;
- a general theorem that the deployed Cube or PushT score is optimal; or
- a domain-general router.

## Recommended related-work positioning

> Adaptive depth, compute-regularized halting, budgeted early exit, and top-k
> capacity allocation are established in adaptive inference. Recent work also
> applies looped depth and selective imagination to world models. Our narrower
> question is empirical: whether a causal transition-level score reallocates a
> fully counted refinement budget better than the strongest transition-
> independent fixed-depth envelope on frozen visual latent-dynamics cohorts.

That wording makes no priority claim and states the actual comparison that the
accepted evidence supports.
