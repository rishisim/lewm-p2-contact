# LeWM Adaptive Compute V1

The frozen scientific protocol is in `PLAN.md`. The implementation operates on
the released Cube LeWM latent predictor, trains only one shared residual
refinement block, and keeps all generated artifacts inside this directory.

Known-good local interpreter (no install required):

```bash
PY='/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python'
```

Unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover \
  -s runs/lewm_adaptive_compute_v1/tests -p 'test_*.py'
```

Reproducible six-episode smoke run:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_v1/run_experiment.py \
  --config runs/lewm_adaptive_compute_v1/smoke_config.json \
  --output-dir runs/lewm_adaptive_compute_v1/smoke_run
```

Full preregistered pilot:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_v1/run_experiment.py \
  --config runs/lewm_adaptive_compute_v1/full_config.json \
  --output-dir runs/lewm_adaptive_compute_v1
```

Add `--force-extract` or `--force-train` only to deliberately regenerate the
corresponding cache or refiner checkpoints.
