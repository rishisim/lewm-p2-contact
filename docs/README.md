# Project Docs

Project-specific notes live here so the upstream `le-wm/` source tree stays
easy to compare with LeWM and Stable WorldModel.

## Index

- `reconnaissance.md`: initial read-only LeWM/source/data reconnaissance.
- `../README.md`: repo-level status, dataset policy, and cleanup boundaries.
- `../le-wm/README.md`: runnable LeWM usage notes and per-dataset Lance policy.
- `../le-wm/diagnostics/README.md`: tracked diagnostics and local-heavy-artifact
  policy.

## Local Data Anchors

The current verified working artifacts are outside git:

- `$STABLEWM_HOME/datasets/pusht_expert_train.lance`
- `$STABLEWM_HOME/datasets/reacher.lance`
- `$STABLEWM_HOME/datasets/reacher_eval.lance`
- `$STABLEWM_HOME/datasets/tworoom.lance`
- `$STABLEWM_HOME/datasets/cube_single_expert.lance`

Cube provenance is also retained locally at:

- `$STABLEWM_HOME/source_downloads/lewm-cube/cube_single_expert.tar.zst`
- `$STABLEWM_HOME/source_downloads/lewm-cube/extracted/cube_single_expert.h5`

The tarball is reproducible from Hugging Face and can be moved or deleted if
disk pressure matters; the extracted HDF5 should remain the local provenance
source unless a replacement source is explicitly verified.
