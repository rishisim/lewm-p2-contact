# External LeWM Storage

The local canonical LeWM source datasets and unique checkpoints were migrated
to the governed USB root:

`/Volumes/ChildLens_Governed/lewm-storage`

The machine-readable contract is `../storage.json`; the checksummed inventory
is `../storage_manifest.json`. The USB copy carries the same manifest and a
storage-identity marker. Git remains the source of truth for code and compact
provenance records; the USB contains heavyweight artifacts only.

## Preflight

```bash
python lewm_storage.py
```

Local entrypoints fail before importing or loading Stable WorldModel data if
the USB is absent or the marker is wrong. For a prepared remote storage root:

```bash
export LEWM_STORAGE_ROOT=/path/to/lewm-storage
python lewm_storage.py
```

The guard exports `STABLEWM_HOME` as
`$LEWM_STORAGE_ROOT/stable-worldmodel` for the current Python process.

## Recovery policy

- Preserve the HDF5 files listed as `provenance_source` in the manifest.
- Preserve locally unique checkpoints.
- Re-download public Hugging Face artifacts using the source records in the
  project documentation.
- Rebuild Lance working tables from the preserved HDF5 sources and validate
  them with `le-wm/verify_dataset.py` and the tracked verification reports.
- Do not treat derived Lance tables, extracted caches, or Hugging Face caches as
  authoritative records.

The USB is a storage location, not a backup. Replicate irreplaceable checkpoints
to a second governed location before retiring this device.
