# LeWM adaptive prediction: completed study

This repository preserves the completed research study **History-Conditioned
Refinement for Visual Manipulation World Models at Matched Counted Compute**.
It began as a contact-event temporal-abstraction project and developed into a
bounded study of adaptive allocation for next-latent prediction. The paper,
curated evidence, negative results, and unfinished follow-on record form the
archive of this research chapter.

## Result and paper

History-conditioned allocation modestly improved next-latent prediction at
matched counted compute in a fresh Cube confirmation and a separately fitted,
pilot-derived PushT binary-policy confirmation. This does not establish
planning or control improvement, practical speed or energy savings, universal
robustness, or zero-shot Cube-to-PushT transfer. The original four-depth PushT
pilot was negative, and the planning bridge did not support a downstream claim.

- [Paper source and build instructions](manuscript/README.md)
- [Compiled paper and archival release](https://github.com/rishisim/lewm-p2-contact/releases/tag/adaptive-prediction-paper-v1)
- [Research status](RESEARCH_STATUS.md)
- [Evidence and claim boundaries](runs/lewm_paper_evidence_synthesis/PAPER_STATUS.md)
- [Claim-to-artifact source map](manuscript/SOURCE_MAP.md)

The annotated tag `adaptive-prediction-paper-v1` freezes the consolidated paper
state. The release PDF is a research record; its availability does not assert
workshop acceptance or publication.

## Other research lines

| Preserved tag | Outcome |
|---|---|
| [`archive/planner-aware-follow-on`](https://github.com/rishisim/lewm-p2-contact/tree/archive/planner-aware-follow-on) | Completed negative stop. Offline prediction improvements did not yield stable planning value; deeper refinement and the bounded critic were harmful. Includes the complete Tasks A--F history and all eight former remote task branches. |
| [`archive/domain-robust-gate`](https://github.com/rishisim/lewm-p2-contact/tree/archive/domain-robust-gate) | Paused before selection, with no scientific result. Fit candidates were sealed; a verifier/inheritance reconciliation is required before resuming selection. |

Each tag retains that line's own `RESEARCH_STATUS.md`, implementation, and
provenance. They preserve historical work without folding unfinished or stopped
implementations into the canonical paper state.

To recover a line in a separate checkout:

```bash
git fetch origin --tags
git worktree add --detach ../lewm-planner-review archive/planner-aware-follow-on
git worktree add --detach ../lewm-domain-review archive/domain-robust-gate
```

## Revisiting the question

Begin with a one-page research brief: the new question, what the archived
result establishes, what remains unknown, and the smallest experiment whose
outcome would change the next decision. Specify the baseline, compute axis,
evaluation population, success criterion, and stop rule before running it.
Treat previously inspected evaluation data as consumed; use fresh data for a
new confirmation. Then choose a branch for a direct extension or a separate
repository for a substantially different question. The archival tag remains
the reference point for the completed study.

## Artifact preservation

Git contains source and curated evidence. The release retains the compiled
paper; it does not contain datasets, unique checkpoints, or full run arrays.
The governed dataset/checkpoint root is
`/Volumes/ChildLens_Governed/lewm-storage`; see
[the storage contract and recovery policy](docs/external_storage.md).
Additional full experiment products are retained locally in
`/Users/rishisim/Documents/research/lewm-p2-contact-local-artifacts`.

At archival consolidation on October 5, 2026, the mounted drive passed the
storage preflight and its manifest matched the Git inventory. All listed
artifacts existed with the recorded sizes; all 38 unique checkpoint files
matched the recorded tree SHA-256, and the split manifest hash matched. Full
hashes of the large HDF5 datasets were not rerun. A second independent backup
was not verified. The sibling local artifact directory is not evidence of an
independent backup. Preserve unique checkpoints and full experiment evidence
at a second governed location and verify their manifest hashes before
considering heavyweight preservation complete.

## Contents

- `le-wm/`: local source snapshot based on `lucas-maes/le-wm`, including
  project-specific training/eval config and contact-event diagnostics.
- `le-wm/diagnostics/`: compact tracked diagnostic scripts, summaries, CSVs,
  plots, and dataset-verification reports.
- `docs/`: project notes that are not part of the upstream LeWM source tree.

## Implementation and dataset status

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

## External Storage

Authoritative source datasets and locally unique checkpoints live on the
governed USB volume at `/Volumes/ChildLens_Governed/lewm-storage`. The canonical
layout, checksums, and recovery procedure are recorded in
`docs/external_storage.md`, `storage.json`, and `storage_manifest.json`.

Run `python lewm_storage.py` as a preflight. The training, evaluation, and
dataset-verification entrypoints run the same preflight automatically and stop
with a clear message when the USB is not connected. Remote machines must point
`LEWM_STORAGE_ROOT` at a prepared root containing the matching marker.

## Excluded

This repository intentionally excludes local virtual environments, Python caches,
large dataset caches, Lance tables, checkpoints, videos, and transient
Hydra/output folders. Keep compact diagnostic reports in git; keep heavyweight
working data in `$STABLEWM_HOME`.
