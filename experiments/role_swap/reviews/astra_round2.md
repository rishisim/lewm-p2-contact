**Revise before implementation.** V2 resolves the wheel/API dispute and much of the execution design, but causal localization and preservation scoring remain blockers. Static review only; no files modified.

Below, **P** = [PLAN.md](/Users/rishisim/Documents/research/lewm-p2-contact/experiments/role_swap/PLAN.md), **W** = the supplied `whl/x/stable_worldmodel` directory. Statuses refer to resolution **in the plan**, not tested implementation.

1. **R1-6 — Not resolved in the decision rule; highest priority.** P:203–207 still labels weak ridge readout “representation-limited,” using a peg/T error ratio despite different visibility and predictability. Weak linear accessibility does not establish information loss. Likewise, B–C correlations within 0.1 of control could both be poor; that cannot establish predictor limitation. Keep the improved CLS/projected probes, but rename findings to *linear-readout deficit*, *observed-state cost mismatch*, and *prediction-associated degradation*. Require adequate absolute B–C agreement and directly assess A–B disagreement before attributing failure to prediction.

2. **R1-3 — Partially resolved.** Controlled evaluation, fixed-length collection, terminal scoring, and maximum disturbance are good fixes (P:107,115,151–175). **However, the revised protocol is internally inconsistent:** preservation success depends on the entire trajectory, while C explicitly uses only the final state (P:194), and reference displacement is unspecified (P:164). Moving the peg away and back can receive perfect C yet fail preservation. Define trajectory-aware reference/C costs; distinguish endpoint-cost diagnostics from preservation diagnostics.

3. **R1-1 — Partially resolved.** Shared training placement distributions and the narrower claim are improvements. But on/off-path still changes geometry, and translating different objects by the same vector does not match manipulation difficulty (P:140–160). The disturbance-allowed metric does not, by itself, “separate” physical difficulty from preservation. Treat these as descriptive task-generalization contrasts; remove causal separation language or add genuinely matched objective interventions.

4. **R1-2 — Partially resolved.** Snapshot tests, indexed proprio, body creation, and separate goal rendering address most concerns. Explicitly include `_get_obs()` and `_set_state()` overrides or bypasses: W:`envs/pusht/env.py:360–368` returns seven values; `:514–527` ignores peg coordinates, zeros agent velocity for nine-value input, and advances physics. The reset override must also construct consistent nine-value defaults and goal fields.

5. **R1-4 — Resolved at plan level.** The pinned wheel genuinely exposes `solver.CEMSolver(model=...)` (W:`solver/cem.py:31–41`), `WorldModelPolicy`, and `wm.utils.load_pretrained`. **My round-1 Git-HEAD incompatibility and missing `[data]` extra objections do not apply:** wheel METADATA:17–19 includes Lance dependencies directly. Explicit pretrained initialization, parity checks, and episode splits address the remaining points. Runtime compatibility still needs the planned smoke tests.

6. **R1-5 — Resolved.** P:70–79 correctly specifies 25 executed steps and shared normalization. W:`policy.py:326–328,407–421` confirms the execution semantics.

7. **R1-7 — Mostly resolved.** Independent feasibility selection, unconditional reporting, clustering, shared banks, and outcome diversity are improvements. Remove the now-unused oracle-normalization concept. Freeze the actual sigma sweep and C definition before evaluation. The new reference-gap criterion also needs clarification: use a separate evaluation seed, otherwise feasibility selection makes its near-zero gap largely tautological (P:167–201).

8. **R1-8 — Partially resolved.** Separate benchmarks and per-checkpoint conclusions help. Move tiny collection/backward/reload/planning gates **before** the 50-episode smoke evaluation (P:83–91); freeze N using an independent pilot.

**Cut remaining over-engineering:** defer submodule replacement and broad package/CLI restructuring. Drop custom CEM solely for population access: the wheel already supplies `candidates`, `costs`, and elites through callbacks (W:`solver/cem.py:247–259`).

