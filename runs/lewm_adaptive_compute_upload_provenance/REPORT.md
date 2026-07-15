# LeWM upload-adjacent provenance report

## Outcome

The preregistered outcome is **`upload_metadata_contains_no_generator_binding`**.
All six immutable revisions and their trees were reconstructed, the upload
pointer and storage namespaces agree, and a bounded archive-header recovery
succeeded. None contains an exact source commit plus coherent same-commit lock,
an immutable container digest plus build/source record, or even archive-tied
collector/config/dependency/seed/renderer metadata. Binding and narrowing
evidence counts are both zero.

## Immutable repository history

The complete [Hugging Face commit history](https://huggingface.co/api/datasets/quentinll/lewm-cube/commits/main)
is a linear six-commit chain from `de75ff3…` through `02a19a6…`. The upload
commit [`f6fe469…`](https://huggingface.co/api/datasets/quentinll/lewm-cube/revision/f6fe469578297a910bd4c88b9857f572908d7a34)
adds exactly `cube_single_expert.tar.zst`; later commits modify only `README.md`.
The archived cards progress from MIT-license front matter to generic
LeWorldModel project/paper links. They never state a generator command, source
revision/tag, lock, dependency version, seed, environment variable,
renderer/backend, or container.

The upload Git object is an LFS pointer with Git blob ID
`01e3f03328086b91750f474872e03428f4d63e9a`, LFS SHA-256
`3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600`,
Xet hash `ffe07ef65b18ea16455f304e1d22fc9e617381cd874576103419dea1064503bf`,
and size 46,184,624,478 bytes, as independently returned by the immutable
[upload tree API](https://huggingface.co/api/datasets/quentinll/lewm-cube/tree/f6fe469578297a910bd4c88b9857f572908d7a34?recursive=true&expand=true).
These are deliberately separate namespaces; the final Xet response ETag is the
Xet hash, not a content SHA-256 assertion.

The initial through penultimate commits contain Hugging Face system RSA
signatures using key ID `6A528E38E0733467`; the local verifier lacks the public
key and reports them as unverified. The final card commit is unsigned. Even if
the signatures were cryptographically verified, their signed commit contents
contain no generator binding. There are no dataset tags, releases, workflow
files, or sibling artifacts beyond `.gitattributes`, `README.md`, and the
archive. A bounded exact-string inspection found no directly tied LeWM workflow
artifact that names the archive or its immutable identifiers.

## Bounded archive and local packaging metadata

Exactly 1,048,576 compressed archive bytes were fetched—6.25% of the 16-MiB
cap. The immutable resolve URL returned 302 with `X-Repo-Commit`,
`X-Linked-ETag`, and `X-Xet-Hash`, then the Xet object returned HTTP 206,
`Content-Length: 1048576`, and
`Content-Range: bytes 0-1048575/46184624478`. No full-body response was
accepted. Conservative accounting is 28 HTTP requests against the frozen cap
of 48, including Git protocol requests and the Range redirect.

Streaming decompression stopped after the first 512-byte tar block. The valid
ustar header records member `cube_single_expert.h5`, size 101,942,558,720,
mtime `2026-02-19T03:12:16Z`, mode 0664, numeric uid/gid 1,500,000,921, and
uname/gname `lucas.maes`. It has no PAX keys or embedded source, digest, or
comment field. Local `stat` reproduces the member size and mtime exactly; the
prior immutable HDF5 digest remains
`0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625`
and was not recomputed. The username, timestamp, and exact stat match are useful
packaging provenance but descriptive only; they do not attribute a generator.
No HDF5 content, attributes, groups, datasets, or rows were opened.

## Verification and scientific boundary

Focused tests pass 4/4. The independent verifier imports none of the collection
implementation and passes all checks: preregistration seals, exact revisions
and parents, raw body/header hashes, request and byte caps, redirect/Range
behavior, pointer/hash separation, tar checksum/parser result, local packaging
match, evidence/decision mapping, zero prohibited execution, and preservation
of the prior adjudication manifest SHA-256
`8b91cd6dafebda232b26c82d4d9cc6cb42603e2ba273d31aaa8512aad5224896`.

The current installed-stack mechanics remain incompatible with sampled stored
observations/controls. No coherent historical public source/lock pair has been
identified. Reset and cross-platform pixel mismatch alone do not prove source
mismatch, and small direct-dynamics residuals may remain non-identifiable from
partial simulator state. Contact remains post-hoc only; V4 remains a consumed
MarkovOracle diagnostic; no “first adaptive world model” claim is made and
LoopWM remains related work.

## Closure and smallest research pivot

Further local generator archaeology is now closed as low-value unless new
external immutable evidence appears. The remaining local metadata is
descriptive and cannot be upgraded by more timestamp, username, storage, or
source-version inference.

The smallest non-confirmatory pivot is a fully declared PlanOracle-native
discovery DGP with an explicit statement that it does not match or reconstruct
the released HDF5. Contact must not enter the gate. Total compute must include
gate overhead exactly, with FLOPs and latency reported separately. V5 remains
forbidden until a prospective discovery gate passes. This is a new scientific
design, not a provenance confirmation or a mechanics replay.
