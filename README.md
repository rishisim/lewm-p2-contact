# LeWM research workspace

A working base for LeWM world-model research with a pinned upstream submodule,
local probe tools, and the governed external-storage guard.

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

## Setup and commands

```bash
git submodule update --init
uv sync
uv run lewm --help
uv run lewm gates
uv run lewm bench-sim
uv run lewm smoke-eval --no-video
uv run pytest -q
```

On macOS arm64, uv excludes `decord` because PyPI has no compatible build.
The HDF5 workflow is available; the optional video dataset reader is not.

`third_party/le-wm/` is pristine upstream source pinned to the role-swap probe
commit. `src/lewm_research/` contains local paths, device, run, model, and CLI
code. `src/lewm_research/data/verify.py` preserves the dataset verification
tool and uses the governed USB guard as before; run it with
`uv run python -m lewm_research.data.verify DATASET` when that volume is ready.
`lewm_storage.py`, `storage.json`, and `storage_manifest.json` remain the
governed USB storage contract for earlier Cube, Reacher, and TwoRoom data.

## Datasets

The PushT smoke evaluation reads `pusht_expert_train.h5` with the pinned
`stable-worldmodel` HDF5 reader. Place it at
`$LEWM_WORK_ROOT/stable-worldmodel/datasets/` before running `smoke-eval`.

## External storage

New PushT downloads, checkpoints, and runs go under `LEWM_WORK_ROOT` (default
`~/lewm-work`) on the internal SSD. The CLI sets `STABLEWM_HOME` to its
`stable-worldmodel/` child. Earlier governed USB datasets remain under
`LEWM_STORAGE_ROOT`; run `python lewm_storage.py` to check that volume. See
[docs/external_storage.md](docs/external_storage.md).

Virtual environments, caches, Lance tables, checkpoints, videos, and run
outputs stay out of git.
