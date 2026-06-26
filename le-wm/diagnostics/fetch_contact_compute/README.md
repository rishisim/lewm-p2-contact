# Fetch Interaction-Error Pilot

This diagnostic tests the narrow descriptive claim that held-out one-step
forward-model error concentrates at active gripper-object
interaction/manipulation contact steps. It deliberately does not implement or
evaluate an adaptive-compute mechanism.

Use "interaction/manipulation contact" for the headline claim, not broad
"contact." Support-only or resting object contact is tracked separately and can
show the opposite pattern.

## Quick Smoke

Use the LeWM environment that has StableWM, Gymnasium Robotics, MuJoCo, and
Torch installed. On this machine that is currently:

```bash
../lewm-p1-inversedynamics/le-wm/.venv/bin/python
```

Run a tiny non-pixel smoke collection:

```bash
env -u MUJOCO_GL ../lewm-p1-inversedynamics/le-wm/.venv/bin/python \
  le-wm/diagnostics/fetch_contact_compute/collect_fetch_contact_dataset.py \
  --num-trajectories 4 --max-steps 12 --skip-pixels --force
```

Train a small single-depth forward-error probe:

```bash
../lewm-p1-inversedynamics/le-wm/.venv/bin/python \
  le-wm/diagnostics/fetch_contact_compute/train_forward_error_probe.py \
  --epochs 5 --force
```

Analyze the held-out split:

```bash
../lewm-p1-inversedynamics/le-wm/.venv/bin/python \
  le-wm/diagnostics/fetch_contact_compute/analyze_interaction_error.py --force
```

## Full Pilot Default

The collector defaults to `500` trajectories, `50` steps, `96x96` RGB frames,
and both `swm/FetchSlideDense-v3` and `swm/FetchPushDense-v3`. Outputs are
written under `le-wm/diagnostics/fetch_contact_compute/data/` and are intended
as bounded local pilot artifacts rather than final benchmark data. `data/` and
`runs/` are ignored locally so smoke tests and full pilots do not accidentally
enter git.

## Regime Precedence

`primary_regime` is assigned with this precedence:

1. `impact_onset`
2. `post_impact_response`
3. `sustained_contact_dynamics`
4. `gripper_object_contact`
5. `free_motion`
6. `boundary_or_invalid`

For proposal text, keep the claim source-faithful: this pilot can motivate an
adaptive architecture by showing where error concentrates, but it is not itself
evidence that extra computation reduces that error.
