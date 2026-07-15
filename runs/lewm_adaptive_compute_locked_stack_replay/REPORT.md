# LeWM Cube locked-stack replay report

## Outcome

The preregistered outcome is **`locked_stack_unavailable`**. The exact frozen
environment cannot be constructed and proven without modifying immutable input:
the required commit's `uv.lock` root entry says stable-worldmodel 0.0.3, while
the same commit's `pyproject.toml` says 0.0.5. `/opt/homebrew/bin/uv sync
--frozen` installed the local project source with distribution metadata 0.0.5.
All other required versions matched: MuJoCo 3.4.0, dm-control 1.0.36,
Gymnasium 1.2.3, OGBench 1.2.1, and NumPy 2.2.6 under CPython 3.10.20.

No version was substituted, neither immutable file was edited, and no unlocked
resolution was attempted. The scientific replay therefore stopped before the
HDF5 was opened or the Cube environment was constructed.

## Frozen scope and execution boundary

`PREREGISTRATION.md` and `protocol.json` were sealed read-only before the new
clone or environment existed. They freeze source commit
`a265229cb29688715651cbd831a3b4c10b8f98b4`, lock SHA-256
`0d1659e36b8cb4b054b95ae69cfabb174cc15193dbba126e0c8cb5da5751724d`,
Python 3.10.20, the same eight discovery episode IDs, 64 transition rows, and
24 frame rows. They also freeze cold, cold-zero-warmstart, and reused-data
restore modes, grouped observation metrics, tolerances, comparison rules, and
the decision tree.

Executed scientific counts are all zero: zero HDF5 rows, transitions, frames,
policy trajectories, model/gate rows, and V5 episodes. V3 test targets were not
opened and the combined V3 cache was not NumPy-loaded. Consequently no locked
mechanics error ratios, EGL/GLFW rendering results, MuJoCo-state inventory, or
restore-mode non-identifiability result exists. `transition_nonidentifiable_from_partial_state`
remains undetermined rather than falsely inferred.

## Package and source audit

The clone is detached at the required commit and its lock hash matches. Imports
for all audited dependencies came from this run-local virtual environment;
stable-worldmodel imported from this run-local source clone. No import resolved
to the current project environment. The exact dependency versions other than
the root project match the lock. The blocker is narrowly package identity:
frozen sync treats the source tree as 0.0.5 despite the stale 0.0.3 lock entry.

This is not safely repairable inside the experiment. Editing `pyproject.toml`
to 0.0.3, regenerating `uv.lock` to 0.0.5, or installing a registry 0.0.3 wheel
would each change the frozen candidate or abandon the required run-local source.

## Scientific handoff

The earlier claims survive unchanged:

- The current installed stack is incompatible with exact stored observations
  and controls on the sampled rows.
- Reset failure alone does not prove a source mismatch because seed and
  end-effector start variation are absent.
- Cross-platform pixel mismatch alone does not prove a source mismatch.
- Small direct-dynamics residuals may be non-identifiable without complete
  simulator state, but this run could not test that hypothesis.

This diagnostic does **not** establish whether the locked dependencies explain
any non-render mismatch and does not prove the released generator. The smallest
next research step is to obtain an immutable clarification of which root-project
identity is intended: a source commit whose project metadata and lock agree, or
an author-provided corrected lock/container for this exact collector. Only then
replay these already-frozen rows. No V5 is authorized.

## Verification and manifest policy

Independent verification imports neither analysis nor environment code. It
checks the protocol seal, commit and lock, conflicting version declarations,
installed binding, exact dependency versions, counts, isolation, decision, and
zero prohibited work. All mutable logs are finished before manifest creation.

The final artifact manifest excludes `.venv/`, `source_repo/.git/`, and the
materialized `source_repo/` files. The source tree is represented by commit,
lock hash, and selected source hashes; the environment is represented by the
frozen requirements export, installed/import provenance, and sync log. The
read-only final verifier prints to stdout only and does not append to any
manifested file.
