#!/usr/bin/env python3
"""Persist the result of the otherwise read-only package-source verifier."""

from __future__ import annotations

import json
import time

from cycle_common import ROOT, assert_runtime_contract, atomic_json
from verify_package_sources import verify


def main() -> None:
    assert_runtime_contract("evaluation")
    output = ROOT / "audit/package_source_qualification.json"
    if output.exists():
        raise RuntimeError("package-source qualification is already immutable")
    result = {
        "schema_version": 1,
        "created_unix_ns": time.time_ns(),
        **verify(),
    }
    atomic_json(output, result, exclusive=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
