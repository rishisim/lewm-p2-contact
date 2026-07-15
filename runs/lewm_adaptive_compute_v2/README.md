# LeWM Adaptive Compute V2

V2 is the frozen sequential causal-gate experiment specified in `PLAN.md`.
It reuses the released Cube LeWM base and V1-selected shared refiner without
retraining either model. All V2 artifacts remain inside this directory; V1 is
read-only provenance.

Known-good interpreter recovered entirely from the local package cache after
the V1 checkout's external `.venv` was concurrently removed:

```bash
PY='/Users/rishisim/.cache/lewm-v2-venv/bin/python'
```

Focused unit tests:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover \
  -s runs/lewm_adaptive_compute_v2/tests -p 'test_*.py'
```

Disjoint six-episode engineering smoke run:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_v2/run_experiment.py \
  --config runs/lewm_adaptive_compute_v2/smoke_config.json \
  --output-dir runs/lewm_adaptive_compute_v2/smoke_run
```

Full preregistered 600-episode run:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_v2/run_experiment.py \
  --config runs/lewm_adaptive_compute_v2/full_config.json \
  --output-dir runs/lewm_adaptive_compute_v2
```

`--force-extract`, `--force-features`, and `--force-gate` deliberately
regenerate the corresponding V2-only artifacts. They never modify V1.
