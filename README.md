# LeWM research workspace

A minimal working base for LeWM world-model research: the LeWM source
snapshot, its training/eval configs, and the governed external-storage guard.

## Earlier work

The first research chapter in this repository is complete and archived.
**History-Conditioned Refinement for Visual Manipulation World Models at
Matched Counted Compute** found that history-conditioned allocation modestly
improved next-latent prediction at matched counted compute (Cube and PushT
confirmations). Planning and control benefit remained unestablished.

| Reference | Contents |
|---|---|
| [`adaptive-prediction-paper-v1`](https://github.com/rishisim/lewm-p2-contact/tree/adaptive-prediction-paper-v1) | Full study: manuscript source, curated evidence, diagnostics, research status |
| [Release with compiled paper](https://github.com/rishisim/lewm-p2-contact/releases/tag/adaptive-prediction-paper-v1) | Paper PDF |
| [`archive/planner-aware-follow-on`](https://github.com/rishisim/lewm-p2-contact/tree/archive/planner-aware-follow-on) | Completed negative stop: offline prediction gains did not yield stable planning value |
| [`archive/domain-robust-gate`](https://github.com/rishisim/lewm-p2-contact/tree/archive/domain-robust-gate) | Paused before selection; no scientific result |

To inspect any of them without disturbing this checkout:

```bash
git fetch origin --tags
git worktree add --detach ../lewm-paper-v1 adaptive-prediction-paper-v1
```

Evaluation data inspected during that study counts as consumed; new
confirmations need fresh data.

## Contents

- `le-wm/`: source snapshot based on
  [`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm) with
  project-specific training/eval config.
- `lewm_storage.py`, `storage.json`, `storage_manifest.json`: storage guard,
  contract, and checksummed inventory.
- `docs/external_storage.md`: storage layout and recovery policy.
- `tests/`: storage-guard tests.

## Datasets

Dataset loading goes through `stable-worldmodel` via
`swm.data.load_dataset(...)`. Lance is the working format for repeated
training and evaluation; HDF5 is kept as the import/provenance format when it
is the original published artifact. Run `le-wm/verify_dataset.py` before
relying on a converted Lance table.

## External storage

Source datasets and locally unique checkpoints live on the governed USB volume
at `/Volumes/ChildLens_Governed/lewm-storage`. Run `python lewm_storage.py` as
a preflight; the training, evaluation, and dataset-verification entrypoints run
it automatically and stop when the volume is absent. Remote machines set
`LEWM_STORAGE_ROOT` to a prepared root containing the matching marker. See
[docs/external_storage.md](docs/external_storage.md).

Virtual environments, caches, Lance tables, checkpoints, videos, and run
outputs stay out of git.
