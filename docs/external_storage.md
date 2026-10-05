# External LeWM Storage

Earlier Cube, Reacher, and TwoRoom source datasets live on the governed USB root:

`/Volumes/ChildLens_Governed/lewm-storage`

The machine-readable contract is `../storage.json`; the checksummed inventory
is `../storage_manifest.json`. The USB copy carries the same manifest and a
storage-identity marker. Git remains the source of truth for code and compact
provenance records; the USB contains heavyweight artifacts only.

## Preflight

```bash
python lewm_storage.py
```

The legacy storage guard fails if the USB is absent or its marker is wrong.
For a prepared remote storage root:

```bash
export LEWM_STORAGE_ROOT=/path/to/lewm-storage
python lewm_storage.py
```

The guard exports `STABLEWM_HOME` as
`$LEWM_STORAGE_ROOT/stable-worldmodel` for the current Python process.

## Role-swap work root

The governed USB has too little free space for the new PushT dataset. New,
regenerable role-swap artifacts go to `LEWM_WORK_ROOT` on the internal SSD
(default `~/lewm-work`):

- `stable-worldmodel/datasets/` for the PushT HDF5 source and later datasets;
- `stable-worldmodel/checkpoints/` for pretrained and fine-tuned weights;
- `runs/` for complete disposable run records, media, and metrics.

The `lewm` CLI sets `STABLEWM_HOME` to
`$LEWM_WORK_ROOT/stable-worldmodel` for its process. Do not point the new
role-swap workflow at the governed USB. Record SHA-256 for downloaded weights
and PushT source. Keep only compact JSON, CSV, Markdown, and PNG summaries in
Git. Delete a downloaded `.zst` only after verified decompression and only if
space requires it. The pilot will measure expanded dataset size and remaining
budget before more collection.

## Recovery policy

- Preserve the HDF5 files listed as `provenance_source` in the manifest.
- Re-download public Hugging Face artifacts using the source records in the
  project documentation.
- Rebuild Lance working tables from the preserved HDF5 sources and validate
  them with `uv run python -m lewm_research.data.verify` and the tracked
  verification reports.
- Do not treat derived Lance tables, extracted caches, or Hugging Face caches as
  authoritative records.

The USB is a storage location, not a backup. Everything on it can be
re-downloaded from the sources in the manifest. If new work produces unique
checkpoints, replicate them to a second location before relying on them.
