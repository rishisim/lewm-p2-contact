# Diagnostics

This folder contains compact project-specific diagnostics and dataset
verification reports. Keep scripts, summaries, CSVs, and plots in git when they
are small enough to review. Keep generated Lance tables, videos, checkpoints,
and full datasets out of git.

## Layout

- `dataset_verification/`: Lance/HDF5 parity reports for working dataset
  artifacts used by configs.
- `pusht_latent_contacts/`: PushT contact and alternative-event diagnostics.
- `cube_event_localization/`: Cube pixel-event diagnostic script and compact
  outputs. The local `cube_first30_pixel_subset.lance` cache is intentionally
  ignored because it is about 870M and reproducible from the Cube source HDF5.
- `fetch_contact_compute/`: StableWM Fetch pilot for active manipulation
  contact versus non-interaction prediction-error diagnostics.
- `stage4_dryrun/`: early PushT dry-run outputs retained for provenance.

## Current Dataset Policy

HDF5 remains source/provenance when that is the published artifact. Lance is the
working/cache/training/eval format. New loaders should go through
`swm.data.load_dataset(...)` and new converted artifacts should have a report in
`dataset_verification/` before configs depend on them.
