# Marginal value of refinement: formal scope

## Setup

For transition $i$, let $\ell_{i,d}$ be its loss after executed depth $d$,
and let

\[
G_{i,d}=\ell_{i,d}-\ell_{i,d+1}
\]

be the realized gain from the next refinement. At the decision following depth
$d$, the router observes only the causal information set
$\mathcal I_{i,d}$: past/current latents and actions, the current prediction,
the most recent update, and derived summaries. It does not contain the target,
future action, reward, contact, simulator state, or later refinement output.
Write

\[
\mu_{i,d}=\mathbb E[G_{i,d}\mid\mathcal I_{i,d}].
\]

Let $c_d$ be the counted cost of executing refinement $d+1$, and $g_d$
the counted cost of evaluating the gate at depth $d$. Decisions obey the
prefix constraint: depth $d+1$ can be executed only after depths $1{:}d$.
The complete routed-system cost includes every reached gate evaluation,
including a gate that decides to stop.

## One clean exact result: binary, equal-cost allocation

**Proposition (finite-cohort top-k optimum).** Consider $n$ transitions that
have all completed mandatory depth 1. Suppose:

1. each transition has exactly one optional depth-2 refinement;
2. its optional counted cost is the same constant $c>0$;
3. every transition’s gate cost has already been paid and is therefore constant
   across allocations;
4. $\mu_i=\mathbb E[\ell_{i,1}-\ell_{i,2}\mid\mathcal I_i]$ is known; and
5. the loss is additive across transitions.

If the remaining budget permits at most $K$ optional refinements, an optimal
allocation selects the $K$ largest positive $\mu_i$ (or all positive values
if fewer than $K$). If exactly $K$ refinements must be spent, select the
$K$ largest $\mu_i$, even if some are negative. Equivalently, a threshold
$\mu_i>\lambda$, with a deterministic or randomized tie rule, implements the
corresponding top-k solution.

*Proof.* For any feasible allocation containing $j$ but excluding $i$ with
$\mu_i>\mu_j$, exchanging $j$ for $i$ preserves cost and increases expected
total gain by $\mu_i-\mu_j>0$. Repeating exchanges yields the sorted allocation.
With an at-most budget, spending on a nonpositive value cannot reduce expected
loss. $\square$

This proposition is deliberately narrow. If optional costs differ, the finite
problem is a 0-1 knapsack; sorting $\mu_i/c_i$ is exact only for the fractional
relaxation or under additional special structure.

## Expectation budget and compute price

For a randomized policy $\pi$, the expectation-budget problem is

\[
\min_{\pi}\;\mathbb E_\pi[\ell_{i,D_i}]
\quad\text{s.t.}\quad
\mathbb E_\pi[C_i]\le B.
\]

Allowing policy randomization makes the attainable expectation-level
loss/cost set convex. Under feasibility and the usual strong-duality/supporting-
hyperplane conditions, some $\lambda\ge0$ supports a budget-optimal policy that
minimizes

\[
\mathbb E_\pi[\ell_{i,D_i}+\lambda C_i].
\]

In the binary equal-cost setting this gives the familiar last-step rule

\[
\text{continue iff}\quad
\frac{\mathbb E[\ell_{i,1}-\ell_{i,2}\mid\mathcal I_{i,1}]}{c_1}>\lambda.
\]

If no single $\lambda$ hits $B$ exactly, an expectation-level optimum may
randomize between adjacent policies. This result does not itself provide an
exact integer schedule for a realized finite cohort.

## Why the myopic rule is not generally correct for multistage Cube

At an intermediate depth, continuing both earns the immediate gain and unlocks
later choices. Let $V_{d+1}$ denote optimal future surplus after reaching the
next decision. Once the current gate has been evaluated, its cost is sunk. A
Bellman form is

\[
V_d(\mathcal I_d;\lambda)
=\max\!\left\{0,
\mathbb E\!\left[
G_d-\lambda(c_d+g_{d+1})
+V_{d+1}(\mathcal I_{d+1};\lambda)
\mid\mathcal I_d
\right]\right\},
\]

where $g_{d+1}=0$ after the last possible refinement. Thus the simple rule

\[
\mu_d/\Delta C_d>\lambda
\]

is exact at the last optional step, but is not generally exact earlier: a weak
immediate gain can be worth buying because it unlocks a high-value later state.
The myopic rule becomes valid only with additional assumptions, for example:

- there is no later optional decision;
- the score already estimates the full Bellman continuation value rather than
  only the immediate gain; or
- all marginal values are known before allocation, refinement costs are equal,
  and each transition has a nonincreasing marginal-gain sequence. In that
  static discrete-concave case, the globally largest $K$ marginal gains form
  a prefix-feasible optimum (with a prefix-respecting tie rule).

Without such conditions, a multistage fixed-budget problem is a stochastic
dynamic program or a prefix-constrained knapsack, not a generic ratio-greedy
problem.

## Three budget settings that must remain separate

| Setting | Exact object | What is optimal/feasible |
|---|---|---|
| Expectation budget | A distribution over policies with $\mathbb E[C]\le B$ | A Lagrange price can support the convex frontier; randomization may be needed between prices. |
| Exact finite-cohort budget | Integer decisions for a fixed set of transitions | Binary equal-cost top-k is exact; unequal costs give knapsack; multistage choices add prefix constraints. |
| Executable scheduling | Decisions must be produced in the available runtime order | Batch binary routing can score all rows and execute exact top-k. Streaming needs a threshold/quota policy. Multistage batch routing can operate in rounds, but globally optimal scheduling must account for later option value and gate costs. |

## Gate overhead

Gate cost has two roles:

1. At a decision whose features and score have already been computed, that
   gate cost is sunk, so the local binary continue/stop denominator contains the
   optional refinement cost, not the already-paid gate.
2. At the system and comparator level, every reached gate must be charged. An
   earlier continuation may also force payment of the next gate. A router can
   therefore have useful rankings yet lose to a gate-free fixed-depth mixture
   when overhead is too large.

For binary batch top-k, one gate per row is constant across routed allocations:
subtract $n g_1$ from the budget and allocate the remaining integer optional
calls. For deciding whether routing should exist at all, however, $n g_1$
cannot be ignored.

## Transition-independent lower envelope

Let $(C_d,L_d)$ be the counted cost and mean loss of fixed depth $d$. The
strongest transition-independent expectation comparator at cost $B$ is the
lower convex envelope

\[
L_{\mathrm{TI}}(B)=
\min_{p_d\ge0}
\left\{\sum_d p_d L_d:
\sum_d p_d=1,\;\sum_d p_d C_d=B\right\}.
\]

With one scalar budget and finitely many depths, an optimum uses at most two
depths. This is the analytic pairwise mixture already used in the accepted
Cube and PushT analyses. It is an expectation over transition-independent
assignments, not a literal fractional transition. An executable finite schedule
uses integer depth counts, an outcome-independent assignment, and either exact
integer equality when available or clearly reported floor/ceiling slack.

## Are the existing gates instances of the rule?

### PushT binary

Structurally, yes **as a rank-based one-optional-step allocator**: all stage-1
scores are available, optional costs are equal, and thresholding the frozen
score is equivalent on a fixed cohort to selecting a top-k set. But the deployed
score is

\[
q_i=\min\{\widehat z^{\rm raw}_i,\widehat z^{\rm white}_i\},
\]

where the two heads predict differently standardized gains. It is not the
conditional marginal reduction of one declared scalar loss, is not divided by
cost, and is not calibrated in $\lambda$-compatible units. Therefore the
accepted binary method should not be renamed “marginal-value routing.” A safe
description is *a causal score ranking that admits a top-k budget
interpretation*.

### PushT four-depth pilot

The same standardized dual-head score and quantile-selected thresholds are
used at three stages. Later immediate gains and optional values are not combined
into a Bellman continuation value. The formally negative whitened pilot also
shows why positive rank association is insufficient for multistage optimality.

### Cube four-depth confirmation

The relationship is weaker. Cube uses stage-specific standardized dual heads,
stage-specific quantile thresholds, and sparse execution. Its saved later-stage
scores exist only for rows reached by the frozen policy. The gate predicts
immediate next-refinement gains, not future option value. The accepted method is
therefore an empirically validated causal allocation heuristic, not an instance
of the myopic compute-price optimum.

## Framework verdict

The marginal-value lens is useful for explaining **what an ideal allocator
would estimate**, why binary top-k is clean, why gate overhead belongs in the
global ledger, and why expectation and executable budgets differ. It does not
unify the deployed Cube and PushT gates as one optimal policy without changing
their meaning. The honest contribution remains the matched-compute empirical
test of causal allocation.
