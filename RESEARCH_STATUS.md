# Domain-robust gate status

Status: `paused_pre_selection_no_scientific_result`

This branch is the canonical record for the unfinished domain-robust,
contact-free adaptive-compute gate program.

## What is complete

- The scientific objective, DGP matrix, preregistration, power rules, and
  controller workflow were defined.
- The fit role completed 1,200 episodes in v006.
- Candidate gates, the fit lock, and the pre-selection boundary were sealed.
- No selection or confirmation outcome was used to fit the candidates.

## Why it is paused

Attempt v007 stopped before constructing a selection world because its
independent verifier applied an active-attempt path-root rule to authenticated
artifacts inherited from v006. The authoritative v007 invalidity record shows
zero selection seeds consumed, zero selection outcomes, and zero confirmation
outcomes.

The v008 directory contains a prepared verifier repair and test package, but no
root-ledger transaction activated v008. The sealed root `STATE.json` therefore
remains preserved rather than rewritten after the fact. Resumption requires a
new, independently authenticated version-forward transaction that reconciles
the v007 invalidity and activates the repaired attempt before any selection
seed is consumed.

## Scientific interpretation

There is no positive or negative generalization result from this program yet.
It is a fitted-but-unselected experiment paused for procedural reconciliation.

## Artifact policy

This branch retains canonical source, contracts, ledgers, fit/pre-selection
audit records, the v007 invalidity, and the prepared v008 repair. Raw episode
roles, fitted arrays, future selection/confirmation products, caches, and logs
remain external and ignored.
