#!/usr/bin/env python3
"""Create the irreversible one-shot V4 confirmation access receipt."""

from __future__ import annotations

import json

import common


RECEIPT = common.ROOT / "audit/confirmation_access_receipt.json"


def create() -> dict[str, object]:
    if RECEIPT.exists():
        raise RuntimeError("the one-shot V4 confirmation receipt already exists")
    required = {
        "phase0_seal": common.ROOT / "audit/phase0_seal.json",
        "pre_confirmation_seal": common.ROOT / "audit/pre_confirmation_seal.json",
        "config": common.ROOT / "config.json",
        "decision_rule": common.ROOT / "decision_rule.json",
        "seed_manifest": common.ROOT / "seed_manifest.json",
        "confirmation_data_manifest": common.ROOT / "data/confirmation_data_manifest.json",
        "smoke_completion": common.ROOT / "audit/smoke_completion.json",
    }
    for path in required.values():
        if not path.exists():
            raise RuntimeError(f"cannot consume confirmation before required artifact exists: {path}")
    common.verify_seal(required["pre_confirmation_seal"])
    confirmation = common.read_json(required["confirmation_data_manifest"])
    if not confirmation.get("complete") or int(confirmation.get("episode_count", -1)) != 300:
        raise RuntimeError("confirmation generation is not complete at exactly 300 episodes")
    payload = {
        "schema_version": 1,
        "status": "consumed_before_any_confirmatory_target_or_next_latent_io",
        "role": "v4_confirmation_once",
        "confirmation_episode_count": 300,
        "v3_test_targets": "forbidden_and_unopened",
        "partial_metric_inspection": "forbidden",
        "rerun_with_changed_choices": "forbidden",
        "files": {
            name: {"path": str(path.relative_to(common.REPO)), "sha256": common.sha256_file(path)}
            for name, path in required.items()
        },
        "frozen_objects": common.verify_frozen_objects(),
    }
    common.write_json(RECEIPT, payload, exclusive=True)
    common.file_mode_read_only(RECEIPT)
    return payload


if __name__ == "__main__":
    result = create()
    print(json.dumps({"status": result["status"], "receipt": str(RECEIPT)}, sort_keys=True))
