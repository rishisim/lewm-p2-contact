# LeWM P2 Contact

Proposal 2 workspace for contact-event temporal abstraction and event-indexed latent prediction.

## Contents

- `le-wm/`: local source snapshot based on `lucas-maes/le-wm`, including
  project-specific training/eval config and contact-event diagnostics.
- `le-wm/diagnostics/`: compact tracked diagnostic scripts, summaries, CSVs,
  plots, and dataset-verification reports.
- `docs/`: project notes that are not part of the upstream LeWM source tree.

## Current Status

- Dataset loading has been migrated to `stable-worldmodel` via
  `swm.data.load_dataset(...)`.
- Lance is the working format for PushT, Reacher eval, TwoRoom, and Cube.
- HDF5 source artifacts remain provenance/import artifacts and live outside git
  under `$STABLEWM_HOME` or the local source-download cache.
- Upstream Stable WorldModel public-HF-URI cleanup is tracked in
  [galilai-group/stable-worldmodel#270](https://github.com/galilai-group/stable-worldmodel/pull/270).

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

This repository intentionally excludes local virtual environments, Python caches,
large dataset caches, Lance tables, checkpoints, videos, and transient
Hydra/output folders. Keep compact diagnostic reports in git; keep heavyweight
working data in `$STABLEWM_HOME`.
