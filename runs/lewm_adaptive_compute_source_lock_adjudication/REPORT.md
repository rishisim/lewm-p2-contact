# LeWM adaptive-computation source/lock adjudication report

## Outcome

The preregistered outcome is **`coherent_generator_candidate_not_identified`**.
No candidate passes every source/lock-coherence and collector-relevance rule.
Accordingly, no candidate clone or environment was constructed and no
scientific HDF5 row or Cube environment was opened.

## Candidate adjudication

The immutable `0.0.4` tag target
`089efbb87746e153e9ee051bf228b9c12aa3889a` predates the preserved HDF5 mtime,
but its source project version is 0.0.4 while its same-commit editable lock root
is 0.0.3. Its lock root also adds `stable-baselines3>=2.7.1` as a base dependency
that the source declares only under `env`. The script uses the older
`ogbench_manip` wrapper path and single-view capture (`multiview=false`), so the
preregistered wrapper files do not exist and the released multiview collector
contract does not match. Its `env` extra metadata itself normalizes exactly.

Commit `a265229cb29688715651cbd831a3b4c10b8f98b4` is the strongest direct
collector evidence. Its script and config match `swm/OGBCube-v0`, single Cube,
`ExpertPolicy`, 10,000 trajectories, 200 actions, seed 3072,
`terminate_at_goal=false`, and 224x224 multiview capture. The required current
wrapper files exist. However, its source version is 0.0.5 while its editable
lock root remains 0.0.3. Its normalized base declarations also disagree: source
adds `typer` and `rich`, while the lock omits those and instead declares
`stable-baselines3>=2.7.1` as base. Its `env` extra normalizes exactly. The lock
hash is recorded in the inventory and its last change is immutable commit
`2ef4e90ac3228317e3793946cce83ed8d22149ac` (2026-02-11).

Both objects were already present in the prior detached partial clone, so no
network fetch occurred. Candidate inventory records git object IDs, dates,
parents, tag peeling, SHA-256 and git blob hashes, lock last-change identity,
normalized declarations, file presence, and collector facts. No mutable branch
head is evidence. The February 18 preserved HDF5 mtime remains supportive only;
`a265229` postdates it and lacks upload-side attribution.

## Execution boundary and verification

The sealed rule requires exactly one static pass before frozen installation.
There were zero. Counts are therefore zero for environment construction, HDF5
rows, transitions, frames, new policy trajectories, model/gate rows, and V5.
V3 test targets were not opened and the combined V3 cache was not NumPy-loaded.
No restore-mode or candidate/current mechanics comparison exists, and
`transition_nonidentifiable_from_partial_state` remains undetermined.

Independent verification parses both pyprojects and locks directly from the
immutable objects and does not import the candidate-selection implementation.
It checks seals, exact candidates/tag, normalized base/env contracts, required
files and collector facts, allowlists/counts, decision mapping, zero prohibited
work, prior-run manifest integrity, and preservation. The final read-only
verifier also rejects unrecorded eligible files.

## Scientific handoff and claim boundaries

No coherent historically plausible public source/lock candidate exists within
the sealed two-object search. The collector match at `a265229` is strong but
cannot rescue a hybrid package contract; the timestamp advantage at `0.0.4`
cannot rescue version, dependency, wrapper, and multiview mismatch. Therefore
another local mechanics replay would not identify provenance and is not
scientifically justified.

Further local provenance archaeology is lower-value because the narrow public
history already separates the best timestamp candidate from the best collector
candidate, while neither carries a coherent lock and the released artifact/card
contains no generator commit, lock, seed, or container. The smallest next step
without author contact or V5 is a read-only search for immutable upload-adjacent
metadata already embedded in the public dataset repository—such as LFS/Xet
pointer history, archive member metadata, or signed release/workflow artifacts—
that explicitly binds the upload to a source commit or container. Do not replay
mechanics unless such evidence identifies one coherent same-commit contract.

The current installed stack remains incompatible with exact stored observations
and controls on sampled rows. Reset and cross-platform pixel mismatches alone do
not prove source mismatch. Small direct-dynamics residuals may be
non-identifiable without full simulator state, but this run did not test that
hypothesis. There is no policy, distribution, solver/gate, FLOP/latency,
contact-aware performance, V5, or “first adaptive world model” claim. LoopWM
remains related work.
