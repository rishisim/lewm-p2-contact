# Planner-aware value-of-computation controller

## Decision

Treat inference compute as a hard-budget portfolio over two coupled axes.  At
each closed-loop planning decision, choose an action

\[
a=(d,n),
\]

where \(d\) is the terminal world-model refinement depth and \(n\) is the
CEM/MPPI population.  The controller predicts the downstream value of each
joint action, charges its measured end-to-end cost, and selects a
prefix-feasible schedule without exceeding the episode budget.  It does not
optimize next-latent MSE.

The first implementation is deliberately an offline/executable scheduling
kernel.  It accepts frozen value and trust estimates plus measured integer
costs, solves the multiple-choice hard-budget problem exactly, and emits an
auditable schedule.  Model adapters, data collection, and training remain
outside this package until competent PushT planning inputs are verified.

## Causal information

At decision time the shared controller may use only information already
available to the planner:

- dimension-invariant summaries of latent history, actions, candidate costs,
  refinement updates, ensemble/route disagreement, and rollout dispersion;
- planner diagnostics from completed iterations (elite-value improvement,
  elite variance, action entropy, effective sample size, and warm-start
  agreement);
- remaining FLOPs and latency budget, elapsed horizon, and the proposed action
  descriptor \((d,n)\).

Raw coordinates, environment IDs, targets from the future, simulator state,
future rewards/success, and counterfactual deeper outputs are forbidden
features.  Raw latent/action vectors may be used by the world model and planner,
but not by the shared budget controller.  Feature names, dimensions, and
normalization statistics must be frozen before evaluation.  “Causal” means
non-anticipatory information flow, not causal identification.

## Computational actions and costs

The action grid is the Cartesian product of supported refinement depths and
planner populations.  Depth is prefix-feasible: selecting depth \(d\) charges
all refinements through \(d\).  Population cost includes every candidate
rollout, CEM/MPPI update, goal-cost evaluation, and fixed per-step overhead.
The controller, feature construction, routing, synchronization, and transfers
are also charged.

Two independent ledgers are required:

1. integer counted FLOPs derived from the executed graph;
2. synchronized end-to-end wall-clock nanoseconds measured on the locked
   hardware/software stack.

The FLOP experiment and latency experiment each solve their own hard-budget
problem.  FLOPs must never be converted into predicted latency.  The budget is
an upper bound, not an expectation; unused budget is reported.

## Value target

For a planning state \(s_t\), define the full-budget reference action
\(a^\star\) using the largest verified depth and population.  The primary
supervised target for compute action \(a\) is negative downstream planning
regret,

\[
V(s_t,a) =
-\left[J_{\mathrm{env}}(s_t,\pi_a)-J_{\mathrm{env}}(s_t,\pi_{a^\star})\right],
\]

estimated from common-random-number closed-loop rollouts.  Task success is the
co-primary binary outcome; regret is primary when success ties or is dense
enough to rank.  A conservative lower-confidence value (mean minus a frozen
uncertainty penalty) drives allocation.  Multi-step rollout error,
reachability error, latent MSE, calibration, and candidate-rank correlation are
diagnostics only.

Training examples must contain every joint action for the same planning state
or a preregistered unbiased missing-action estimator.  Splits are by episode
and environment.  Leave-one-environment-out evaluation refits neither feature
normalization nor the controller.

## Hard-budget policy

Given frozen conservative values \(\widehat V^-_{i,a}\), measured costs
\(c_a\), and budget \(B\), schedule one action per decision:

\[
\max_{a_1,\ldots,a_T}\sum_i \widehat V^-_{i,a_i}
\quad\text{s.t.}\quad
\sum_i c_{a_i}\le B.
\]

This is a multiple-choice knapsack, not a ratio-greedy rule.  The scaffold
solves it exactly with deterministic ties.  Online receding-horizon use must
reserve the minimum fallback cost for unseen decisions and may re-solve after
each observation.  Any future approximate solver must be checked against the
exact kernel on bounded instances.

## Trust, failure, and abstention

An action is ineligible when its inputs are nonfinite, its uncertainty exceeds
the frozen bound, route/ensemble disagreement exceeds the frozen trust bound,
or it is outside calibration support.  If no optional action is trusted, the
controller abstains to the mandatory coarse fallback action.  The fallback is
always executable, included in the budget reserve, and may invoke a separately
declared architecture/observation path; silently spending more refinement is
not abstention.  Budget infeasibility fails closed before environment
execution.

Report abstention rate, reason histogram, budget left unused, and outcomes on
abstained versus routed decisions.  Compare against both always-fallback and
always-full-compute policies.

## Baselines and matching

Evaluate **every** fixed depth × population pair, not only frontier points.
For each adaptive FLOP budget and each adaptive latency budget:

- report all fixed pairs with their measured cost and outcomes;
- form the strongest transition-independent randomized mixture of fixed pairs
  at the identical aggregate budget when integer execution permits;
- otherwise use the strongest non-exceeding executable schedule and report
  slack, plus the analytic convex-envelope value as a non-executable reference;
- include depth-only adaptation with fixed population, population-only
  adaptation with fixed depth, shuffled joint actions preserving the adaptive
  histogram, a cost-only controller, and an oracle schedule labeled diagnostic.

All arms share starts, goals, environment seeds, action bounds, horizon,
replanning frequency, warm starts, terminal handling, and common random
numbers.  Controller overhead is charged only to arms that execute it, while
the matched fixed comparator receives the same total budget.

## First evaluation order

1. Establish nonzero-success PushT trajectories and a competent fixed CEM
   controller using the existing LeWM checkpoint.
2. Freeze the joint action grid and measure per-action FLOPs and synchronized
   latency.
3. Run bounded replay/smoke tests and verify the ledger, causal boundary, and
   exact scheduler.
4. Only then collect disjoint training/calibration/evaluation cohorts.
5. Add Reacher and TwoRooms, followed by leave-one-environment-out transfer.

Cube remains a mechanism/diagnostic environment unless it supplies an
informative closed-loop planner population.

## Falsification criteria

The planner-aware claim fails if any of the following occurs on the locked
evaluation:

- no competent fixed planner produces a preregistered minimum success rate or
  sufficiently variable regret population;
- the joint controller fails to improve success or planning regret over the
  strongest matched-cost fixed-pair/mixed baseline;
- gains disappear under either exact FLOP matching or measured-latency
  matching;
- either one-axis ablation matches the joint controller within the
  preregistered noninferiority margin;
- action identity randomization performs equivalently;
- leave-one-environment-out performance is worse than the strongest fixed
  baseline beyond the safety margin;
- budget, causal-feature, or common-random-number audits fail.

Prediction improvements without planning improvement are a negative planning
result, not partial confirmation.
