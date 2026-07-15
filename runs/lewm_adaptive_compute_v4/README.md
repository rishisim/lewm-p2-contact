# V4 confirmatory artifact

This directory contains the preregistered one-shot LeWM V4 confirmation. Prior experiment trees are immutable; all new files live here. `PREREGISTRATION.md` and `decision_rule.json` define the frozen analysis, `decision.json` is the mechanical verdict, `REPORT.md` is the narrative readout, and `artifact_manifest.json` hashes the complete deliverable.

## Exact execution order

1. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/audit_prior.py`
2. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/prepare_preregistration.py`
3. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/run_tests.py phase0`
4. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/seal.py phase0`
5. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/freshness.py index`
6. `/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python runs/lewm_adaptive_compute_v4/generator.py smoke`
7. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/evaluate.py smoke --device mps`
8. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/run_tests.py preconfirmation`
9. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/seal.py preconfirmation`
10. `/Users/rishisim/Documents/research/World Models/le-wm/.venv/bin/python runs/lewm_adaptive_compute_v4/generator.py confirmation`
11. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/receipt.py`
12. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/freshness.py audit`
13. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/evaluate.py confirmation --device mps`
14. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/statistics.py`
15. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/secondary.py`
16. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/flops.py`
17. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/benchmark_runtime.py`
18. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/report.py`
19. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/run_tests.py final`
20. `/Users/rishisim/.cache/lewm-v2-venv/bin/python runs/lewm_adaptive_compute_v4/finalize.py`
