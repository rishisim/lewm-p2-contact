# LeWM Adaptive Compute V3

This directory contains the preregistered joint iterative-refiner/local-halting
experiment. `PLAN.md` and `full_config.json` were frozen and hashed before the
V3 split or any V3 outcome was generated. The confirmatory verdict is
`adaptive_refiner_headroom_failed`.

Known-good local interpreter:

```bash
PY='/Users/rishisim/.cache/lewm-v2-venv/bin/python'
```

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover \
  -s runs/lewm_adaptive_compute_v3/tests -v
```

Isolated engineering smoke reproduction:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_v3/run_experiment.py \
  --config runs/lewm_adaptive_compute_v3/smoke_config.json \
  --output-dir runs/lewm_adaptive_compute_v3/smoke_run --device mps
```

Full preregistered reproduction (regenerates no V1/V2 artifact):

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_v3/run_experiment.py \
  --config runs/lewm_adaptive_compute_v3/full_config.json \
  --output-dir runs/lewm_adaptive_compute_v3 --device mps
```

Use `--force-extract` or `--force-train` only to deliberately regenerate the
corresponding V3 artifacts. The exact split is recorded in
`cache/planned_split.json`; strict model/cache hashes are in `provenance.json`.
