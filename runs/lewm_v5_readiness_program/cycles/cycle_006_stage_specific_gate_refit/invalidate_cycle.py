#!/usr/bin/env python3
"""Seal an implementation-invalid cycle without changing normative code."""

from __future__ import annotations

import argparse
import json
import time

from cycle_common import ROOT, atomic_json, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--evidence", required=True)
    arguments = parser.parse_args()
    decision_path = ROOT / "decision.json"
    if decision_path.exists():
        raise RuntimeError("terminal outcome is immutable and already exists")
    evidence_path = ROOT / arguments.evidence
    if not evidence_path.exists():
        raise RuntimeError("invalidation evidence is missing")
    evidence = read_json(evidence_path)
    prospective_manifest = ROOT / "data/prospective_raw_manifest.json"
    prospective_episodes = (
        int(read_json(prospective_manifest)["episode_count"])
        if prospective_manifest.exists()
        else 0
    )
    decision = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        "terminal_outcome": "cycle_execution_invalid",
        "failure_stage": arguments.stage,
        "evidence_path": arguments.evidence,
        "evidence": evidence,
        "prospective_episodes_consumed": prospective_episodes,
        "prospective_statistical_pass_claimed": False,
        "normative_code_patched_after_seal": False,
        "v5_outcome_episodes": 0,
    }
    atomic_json(decision_path, decision, exclusive=True)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()

