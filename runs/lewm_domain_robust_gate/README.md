# LeWM domain-robust gate study

This is the sole root for the gate-only study authorized after the terminal V5
v004 confirmation and terminal v005 zero-shot generalization result. `STATE.json`
and `RESEARCH_LEDGER.jsonl` are durable recovery controls; every scientific
artifact belongs to an immutable versioned directory under `attempts/`.

The program is fail-closed. Resume by running `python3 program.py status`,
verifying the last checkpoint hash, and continuing only the `next_action`.
V5 and v005 are consumed read-only evidence and must never be edited, repaired,
regenerated, or rerun.
