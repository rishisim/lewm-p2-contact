# LeWM Adaptive-Compute Critic Compression

This directory contains a discovery-only critic-compression tournament for the
immutable selected stagewise solver from `../lewm_adaptive_compute_discovery`.
No solver retraining, fresh episodes, V4 data, or V3 test-target access is
permitted.

Commands (the report records exact completed invocations):

```bash
PY=/Users/rishisim/.cache/lewm-v2-venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest discover \
  -s runs/lewm_adaptive_compute_critic_compression/tests -v
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_critic_compression/run_experiment.py prepare --device mps
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_critic_compression/run_experiment.py smoke --device mps
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_critic_compression/run_experiment.py discovery --device mps
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_critic_compression/runtime_audit.py --device mps
PYTHONDONTWRITEBYTECODE=1 "$PY" \
  runs/lewm_adaptive_compute_critic_compression/make_artifacts.py
```

The sealed `judge` command was run once after discovery passed. Its receipt now
prevents reuse; do not rerun it. See `REPORT.md` and `decision.json` for the
result and interpretation.
