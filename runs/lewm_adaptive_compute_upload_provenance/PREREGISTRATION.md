# LeWM upload-adjacent provenance preregistration

Status: sealed before every network/API/git request and archive-byte fetch in
this run. `protocol.json` is the machine-controlling protocol. SHA-256 sidecars
bind both files.

## Frozen scope

This final read-only provenance diagnostic inspects only immutable public
metadata adjacent to Hugging Face dataset `quentinll/lewm-cube` and archive
`cube_single_expert.tar.zst`. The complete frozen revision sequence is
`de75ff3cef0851c208b849c58570d765a2ae0637`,
`f6fe469578297a910bd4c88b9857f572908d7a34`,
`116b5467ac69094027b3e2cfda48c79f482356d3`,
`63c01ba443b5ebdfd363b46ead9912b612218d96`,
`54324364f20a0002c075958c49c7027ffa2ffd71`, and
`02a19a67a0dc8c9d6215f89c19e0a597691e152a`. No mutable branch head is
evidence.

Allowed sources are Hugging Face primary API and immutable revision/resolve
URLs; immutable Git commit/tree/blob endpoints for that dataset; and directly
associated GitHub release/workflow artifacts only when they explicitly name
the archive filename, exact LFS/Xet/HDF5 hash, or upload revision. There is no
broad source-version or mechanics search. Maximum network requests are 48.
Total newly fetched archive payload is capped at 16 MiB (16,777,216 bytes).

## Metadata and bounded archive rules

For every revision, record commit identity, parent, author/committer timestamps,
signature fields if present, message, immutable tree, sibling files, README
bytes/diffs, `.gitattributes`, and archive pointer/storage metadata and action.
Raw bodies and response headers are retained with SHA-256, URL, status,
redirect chain, ETag, Content-Range and Content-Length. Git blob IDs, LFS
SHA-256, Xet hashes, HTTP ETags, and content digests remain distinct namespaces.

Archive recovery begins with the smallest useful prefix and may expand only
within the total cap. Every response must be HTTP 206 with a valid
`Content-Range` matching the request and a body no larger than the requested
range. HTTP 200, malformed/missing range metadata, unexpected byte count, or a
redirect target that cannot preserve Range aborts archive work immediately;
the body is not retained. Streaming decompression stops after the first
tar/PAX header. Only member name, size, mtime, mode, uid/gid, uname/gname, PAX
keys, zstd frame metadata, and embedded comment/digest/source fields may be
recorded. HDF5 payload is never opened or interpreted.

Local extracted-file inspection is restricted to stat and xattr-style
filesystem metadata. Its 95-GiB SHA-256 is not recomputed; only the immutable
prior-manifest digest is compared. V3 targets/caches, Cube environments,
policies, models, gates, trajectories, and V5 are untouched.

## Evidence and decisions

Binding evidence is an exact source commit with a coherent same-commit lock, or
an immutable container digest with build/source record, explicitly tied to the
exact archive OID/hash/upload revision. Narrowing evidence is an exact
collector command/config, dependency versions, environment variables, seed,
renderer/backend, or source tag explicitly tied to the exact archive but short
of a coherent binding. Timestamps, names, uploader identity, sizes, storage
hashes, generic project links, and archive-member metadata without source
linkage are descriptive only.

The exclusive outcomes are `upload_metadata_binds_coherent_generator`,
`upload_metadata_narrows_generator_without_binding`,
`upload_metadata_contains_no_generator_binding`, and
`upload_metadata_unavailable`. Descriptive or temporal evidence is never
promoted. If sufficient immutable metadata is accessible and no binding or
narrowing evidence exists, choose the third outcome. If primary immutable
metadata cannot be reconstructed within caps, choose unavailable. Stop after
all six revisions, upload-tree/pointer metadata, eligible directly tied
artifacts, one bounded archive-header attempt, and local stat are adjudicated.
Even a binding does not authorize mechanics replay.

Independent verification must not import the collector. It checks seals,
revisions, response hashes, request/payload caps, range behavior, hash
namespaces, tar parsing if present, evidence mapping, zero prohibited work,
prior-manifest preservation, and a complete final manifest. If no binding is
found, further local generator archaeology is closed unless new immutable
external evidence appears; the smallest next step is a fully declared
PlanOracle-native discovery DGP, explicitly non-confirmatory and not claimed to
match the released HDF5, with no contact gate input, exact total compute
including gate overhead, separate FLOPs/latency, and no V5 before a prospective
discovery gate passes.
