# World Models

Research workspace snapshot for local world-model experiments.

## Contents

- `le-wm/`: local source snapshot based on `lucas-maes/le-wm`, including the current local code/config edits and helper scripts.
- `prelim-p1/`: Proposal 1 staged feasibility scripts, reports, logs, and compact result artifacts.
- `stable-pretraining-readonly/`: local source snapshot based on `galilai-group/stable-pretraining`.
- `stable-worldmodel-readonly/`: local source snapshot based on `galilai-group/stable-worldmodel`.
- `LeWM_recon_report.md`: local reconstruction notes/report.

## Excluded

This repository intentionally excludes local virtual environments, Python caches, large dataset caches, checkpoints, and transient Hydra/output folders. In particular, the local `le-wm/.venv`, `le-wm/.cache`, and `le-wm/outputs` directories are not tracked.
