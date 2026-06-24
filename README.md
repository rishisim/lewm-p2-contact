# LeWM P2 Contact

Proposal 2 workspace for contact-event temporal abstraction and event-indexed latent prediction.

## Contents

- `le-wm/`: local source snapshot based on `lucas-maes/le-wm`, including contact-event diagnostics and geometric contact fallback utilities.
- `le-wm/diagnostics/`: tracked diagnostic summaries, CSVs, and plots for PushT contact analyses.
- `LeWM_recon_report.md`: local reconstruction notes/report.

## Dataset Policy

This workspace delegates dataset loading and conversion to `stable-worldmodel`.
Use Lance as the canonical working format for repeated training, evaluation
sampling, and large pixel reads. Keep HDF5 as an import/provenance format when
it is the original published artifact.

All new training or evaluation loaders should go through
`swm.data.load_dataset(...)` so the `stable-worldmodel` format registry chooses
the backend from the dataset path. Use `le-wm/verify_dataset.py` before relying
on a converted Lance artifact.

## Excluded

This repository intentionally excludes local virtual environments, Python caches, large dataset caches, checkpoints, videos, and transient Hydra/output folders.
