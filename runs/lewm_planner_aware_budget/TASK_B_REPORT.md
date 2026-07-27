# Task B gated report

## Outcome

B1 is frozen and covered by executable contract tests. B2 stopped at the
preregistered readiness gate. No refiner was trained, no checkpoint was
exported, and CEM integration was not attempted.

This is the required negative bounded-pilot outcome rather than a completed
positive-depth planner result. Existing local evidence cannot support the
requested planner-independent claim: the prior real-transition WeakPolicy
export contains only length-3 histories, the expert Lance source does not cover
mixed/off-policy or executed CEM action distributions by itself, and Task A did
not retain real next-transition targets for CEM candidates.

## Retained changes

- `pusht_refiner.py` defines the one canonical action/mask/depth API, exact
  depth-zero semantics, fail-closed positive-depth availability, a zero-started
  residual design, and call/row accounting.
- `TASK_B_PROTOCOL.md` freezes data roles, episode/source splits, seeds,
  selection, metrics, thresholds, causal boundary, export requirements, and
  the smallest collection expansion.
- `task_b_readiness.json` is the compact machine-readable source audit.
- `tests/test_pusht_refiner.py` proves action round trips and range rejection,
  legal prefixes, masked-padding invariance, exact depth zero, invalid-mask
  rejection, unavailable-export rejection, and stage ledger counts.

The original checkout, installed package, Task A planner/checkpoint/config, and
untracked prior research artifacts were read only. No run/output directory was
created.

## Gate and next decision

Proceeding requires explicit authorization for a small real-transition
collection. The frozen minimum is 20 fit episodes per source and 10
selection/evaluation episodes per source, with at least 100 examples in each
role-by-length cell. It would execute a fixed mixture of expert-policy, Task A
CEM-selected, WeakPolicy, and perturbed/off-policy physical actions in PushT,
writing raw records only to the already ignored
`runs/lewm_planner_aware_budget/work/` root.

Only if schema, target alignment, masks, provenance, source/episode isolation,
coverage, and canonical action checks pass would training begin. B3/B4 and the
requested measurable CEM effect remain deliberately incomplete.
