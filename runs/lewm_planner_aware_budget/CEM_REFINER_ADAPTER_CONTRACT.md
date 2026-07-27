# CEM–refiner adapter contract and compatibility decision

Status: blocked before adapter implementation. This record was written after
interface/artifact audit and before any planner-model bridge was implemented.

## Intended public contract

The project-owned cost-model adapter would accept one integer
`refinement_depth` in `{0, 1, 2, 4}`. Booleans, non-integers, and other values
would fail before a rollout. Depth would be uniform across every candidate,
predicted horizon transition, CEM iteration, and closed-loop replan. Depth zero
would delegate exactly to the qualified `AutoCostModel` behavior.

The adapter, not CEM, would own latent encoding, autoregressive history updates,
action-block conversion, and terminal goal cost. It would preserve the solver's
`[environment, candidate, horizon, 10]` candidate shape and return
`[environment, candidate]` costs. The base model and refiner would be
hash-verified, frozen, in evaluation mode, and called under inference mode.
No target latent, future observation, simulator state, reward, contact, or
environment geometry would be accepted at the refinement boundary.

The operation ledger would count:

- candidate rollouts: one per evaluated environment/candidate pair;
- predicted transition steps: one per candidate per model horizon step;
- base calls and rows;
- refiner invocations and rows at each executed stage;
- selected fixed depth and synchronized elapsed time.

For a uniform depth `d`, each predicted transition row would have one base
prediction and exactly `d` prefix-feasible refiner stages. Explicit per-stage
FLOP constants could be reported as configured estimates. They would not be
called a complete planner FLOP ledger because encoding, goal encoding, CEM
sampling/top-k/statistics, tensor movement, and policy overhead are not all
covered.

Missing or hash-incompatible artifacts, shape/normalization disagreement,
nonfinite tensors, trainable parameters, training mode, or accounting mismatch
would fail closed. There would be no silent fallback from a requested positive
depth to depth zero.

## Audited recurrence

Installed `CEMSolver` expands observations to
`[B, population, ...]`, samples candidates shaped
`[B, population, horizon, raw_action_dim * action_block]`, and calls
`model.get_cost` once per CEM iteration. The frozen Task A settings therefore
request `[1, 128, 5, 10]` candidates twenty times per replan.

Tracked `JEPA.rollout` takes the observation time length `H` from
`info["pixels"].shape[2]`. It encodes those frames once, splits the first `H`
candidate action blocks into the initial action history, and then predicts
autoregressively. Each prediction uses only the last three available latents
and action blocks. `criterion` is terminal latent squared error summed over 192
coordinates. `WorldModelPolicy` executes five raw 2-D actions from each chosen
10-D block before replanning.

`World` supplies current observations with one time row,
`[num_envs, 1, ...]`; Task A does not install a frame-stack wrapper.
Consequently a five-step CEM rollout presents predictor histories of lengths
`1, 2, 3, 3, 3`.

The available PushT refiner checkpoint has SHA-256
`29be92d39a2efba992c8e2dcfd289956d04f057d63d8ed1042de25c305641a6f`.
It was trained for the same released PushT weight set, latent dimension 192,
blocked action dimension 10, and exits 1–4. Its input layer width is 814:

```text
3 * 192 latent history
+ 3 * 10 action history
+ 192 current prediction
+ 16 iteration embedding
= 814
```

It has no variable-length mask or history-length input. Its training records
contain only three-step histories. It also consumes actions normalized by ten
pilot-fit coordinate means/scales from a WeakPolicy transition distribution;
Task A's CEM candidates inhabit the qualified policy/model action-processing
space. The artifact does not export a validated mapping between those two
spaces for planner rollouts.

## Decision

The adapter is not implemented. Repeating or zero-padding the initial latent
and action rows would make the tensor width fit, but no available artifact was
trained or validated under either convention. Changing Task A to frame-stack
three observations would also change the qualified depth-zero planner path.
Either choice would fabricate the missing scientific interface and violate the
requirement that every CEM transition execute the specified verified refiner.

`compatibility_check.py` proves the mismatch from the tracked planner contract
and the read-only PushT artifact. No untracked research artifact is copied,
edited, or deleted.

## Smallest unblocker

Freeze the exact Task A PushT object checkpoint and action preprocessing, then
generate one-step training examples from the actual CEM rollout recurrence,
including prefix lengths one and two. Define one causal history policy
(preferably an explicit mask/length-conditioned refiner; padding is acceptable
only if declared before training), train/evaluate depths 1, 2, and 4 under that
policy, and export:

1. a hash-pinned checkpoint with the history/padding or mask contract;
2. planner-action conversion constants and provenance;
3. dense/fixed-depth reference outputs for prefix lengths 1, 2, and 3;
4. exact stage call/row reference ledgers; and
5. frozen/eval/no-gradient and no-leakage validation evidence.

With that export, the intended adapter contract above can be implemented
without changing `stable_worldmodel` or the qualified depth-zero planner.
