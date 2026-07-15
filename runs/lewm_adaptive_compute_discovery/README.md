# LeWM Adaptive Compute Discovery

This directory contains the isolated First-Proof search following the negative
V3 joint-training result. It never opens the V3 combined latent cache and never
loads V3 test targets.

Final verdict: `discovery_gate_failed`. The calibration judge command below is
documented as protocol only and was intentionally not executed because no
candidate passed the internal discovery gate.

Interpreter:

```bash
PY=/Users/rishisim/.cache/lewm-v2-venv/bin/python
```

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover \
  -s runs/lewm_adaptive_compute_discovery/tests -v
```

Discovery-fit/internal-validation phase:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_discovery/run_discovery.py fit --device mps
```

One-shot V3 calibration tournament (only after fit freezes the tournament):

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_discovery/run_discovery.py judge --device mps
```

All generated artifacts remain under this directory. A failing discovery gate
forbids V4 creation and fresh confirmatory data.
