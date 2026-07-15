#!/usr/bin/env python3
"""Generate the sealed report, then apply the pre-main reset-amendment note."""

from __future__ import annotations

import json

import common
import generator_seedfix
import report as sealed_report


def run() -> dict:
    generator_seedfix.assert_amendment_seal()
    result = sealed_report.run()
    report_path = common.STUDY_ROOT / "REPORT.md"
    readme_path = common.STUDY_ROOT / "README.md"
    text = report_path.read_text(encoding="utf-8")
    old = """The bounded design was executed exactly: 12 physically separate smoke episodes
per policy (permanently excluded), 90 PlanOracle discovery episodes, and 30
MarkovOracle controls paired on identical environment seeds and exact initial
states. Wrapper and NumPy-global oracle RNG streams were separate. There was no
sequential expansion and no confirmatory episode.
"""
    new = """The bounded counts were executed exactly: 12 physically separate smoke episodes
per policy (permanently excluded), 90 PlanOracle discovery episodes, and 30
MarkovOracle controls. The excluded smoke exposed an installed Cube reset bug:
`seed` was accepted but not forwarded, and the variation space was reseeded from
entropy. That failed smoke audit is preserved. Protocol Amendment 01 was frozen
before any main rollout; it explicitly seeded the normal Cube variation and
physical-state RNGs. All 30 main controls were then paired on identical
environment seeds and exact initial states. Wrapper and NumPy-global oracle RNG
streams remained separate. There was no sequential expansion and no
confirmatory episode.
"""
    if text.count(old) != 1:
        raise RuntimeError("sealed report design paragraph changed unexpectedly")
    text = text.replace(old, new)
    marker = "## Distribution readout\n"
    amendment = """## Pre-main protocol amendment

The local incompatibility and its correction were discovered using only the
excluded smoke data. The original failed pairing audit, installed source hashes,
corrected initial-state-only probe, corrected RNG values for every main episode,
and amendment seal are retained in `audit/`. Sample sizes, preprocessing,
reference bands, metrics, gate, and decision rules were unchanged.

"""
    if text.count(marker) != 1:
        raise RuntimeError("sealed report insertion marker changed unexpectedly")
    text = text.replace(marker, amendment + marker)
    report_path.write_text(text, encoding="utf-8")

    readme = readme_path.read_text(encoding="utf-8")
    readme += (
        "\n`PROTOCOL_AMENDMENT_01.md` records the excluded-smoke reset seed "
        "incompatibility and the correction frozen before all main rollouts.\n"
    )
    readme_path.write_text(readme, encoding="utf-8")
    result.update(
        {
            "report_sha256": common.sha256_file(report_path),
            "readme_sha256": common.sha256_file(readme_path),
            "protocol_amendment_01_sha256": common.sha256_file(
                generator_seedfix.AMENDMENT_JSON
            ),
            "amendment_applied_to_report": True,
        }
    )
    common.write_study_json(
        common.STUDY_ROOT / "audit/report_manifest.json", result
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
