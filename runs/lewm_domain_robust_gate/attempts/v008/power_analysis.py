#!/usr/bin/env python3
"""Prospective family-8 power rule for fixed confirmation size.

The binding calculation is run once after fit and selection summaries exist and
before any confirmation outcome is generated.  It consumes summary moments,
not episode arrays.  An infeasible grid is a valid explicit result and never an
invitation to generate, expand, or retry confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from scipy.stats import chi2


ATTEMPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ATTEMPT_ROOT.parents[3]
ACTIVE_ATTEMPT = ATTEMPT_ROOT.name
if ACTIVE_ATTEMPT != "v008":
    raise RuntimeError(
        f"power_analysis.py must run from v008, got {ACTIVE_ATTEMPT!r}"
    )
SCIENCE_ATTEMPT = "v001"
POWER_RULE_PATH = ATTEMPT_ROOT / "power_rule.json"
POWER_RULE_SHA256 = "de64698680b3060181120ce40a07f447f2746af4c3f9c6aab882965a639e467a"
POWER_RULE_MOMENTS_SHA256 = (
    "5b32f9a48dc6d2df07f19d452d9031359eb5bb9a30778eb26f47709f4d20e36f"
)
FIT_LOCK_PATH = ATTEMPT_ROOT / "fit/fit_lock.json"
SELECTION_LEDGER_PATH = ATTEMPT_ROOT / "selection/selection_ledger.json"
GATE_FREEZE_PATH = ATTEMPT_ROOT / "freeze/gate_freeze.json"

REGIMES = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
ENDPOINTS = ("raw_vs_analytic", "fixed_whitened_vs_analytic")
FIT_EPISODES_PER_REGIME = 300
SELECTION_EPISODES_PER_REGIME = 500
FAMILY_SIZE = len(REGIMES) * len(ENDPOINTS)
FAMILYWISE_ALPHA = 0.05
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / FAMILY_SIZE
FAMILY_POWER_TARGET = 0.95
EFFECT_RETAINED_FRACTION = 0.35
GRID = tuple(range(500, 4_501, 500))
SD_UPPER_CONFIDENCE = 0.95
INHERITED_EXPECTED_MINIMUM = 3_500
NORMAL = NormalDist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regular_file(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError(f"required regular non-symlink file is absent: {candidate}")
    return candidate


def _repo_relative(path: Path, *, must_exist: bool = True) -> str:
    candidate = Path(path)
    resolved = (
        _regular_file(candidate).resolve()
        if must_exist
        else candidate.resolve(strict=False)
    )
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"power artifact escapes the repository: {path}") from exc


def load_frozen_power_rule_moments(
    path: Path = POWER_RULE_PATH,
) -> dict[str, dict[str, float]]:
    """Load endpoint moments only from the exact frozen local power rule."""

    path = _regular_file(path)
    observed_hash = _sha256(path)
    if observed_hash != POWER_RULE_SHA256:
        raise RuntimeError(f"frozen local power-rule hash drift: {observed_hash}")
    document = json.loads(path.read_text(encoding="utf-8"))
    rule_checks = {
        "schema": document.get("schema_version") == 1,
        "attempt": document.get("attempt") == SCIENCE_ATTEMPT,
        "status": document.get("status")
        == "prospective_rule_frozen_before_fit_outcomes",
        "family_size": document.get("family_size") == FAMILY_SIZE,
        "familywise_alpha": document.get("familywise_alpha") == FAMILYWISE_ALPHA,
        "per_claim_alpha": document.get("per_claim_alpha") == PER_CLAIM_ALPHA,
        "family_power_target": document.get("family_power_target")
        == FAMILY_POWER_TARGET,
        "grid": document.get("common_episode_grid_per_regime") == list(GRID),
        "same_n": document.get("same_n_every_dgp") is True,
        "no_sequential_expansion": document.get("sequential_expansion") is False,
    }
    if not all(rule_checks.values()):
        raise RuntimeError(
            "frozen local power-rule object drift: "
            f"{[name for name, passed in rule_checks.items() if not passed]}"
        )
    source = document.get("consumed_v5_endpoint_moments")
    if not isinstance(source, Mapping):
        raise RuntimeError("frozen power rule lacks inherited endpoint moments")
    if _canonical_sha256(source) != POWER_RULE_MOMENTS_SHA256:
        raise RuntimeError("frozen power-rule endpoint-moment object hash drift")
    result: dict[str, dict[str, float]] = {}
    for endpoint in ENDPOINTS:
        item = source.get(endpoint)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"frozen power rule lacks endpoint {endpoint}")
        mean = float(item["mean"])
        sd = float(item["sd_ddof1"])
        count = int(item["episode_count"])
        if count != 1_600 or not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"invalid consumed V5 moment for {endpoint}: {item}")
        result[endpoint] = {
            "mean": mean,
            "sd_ddof1": sd,
            "episode_count": count,
        }
    return result


def one_sided_mean_power(n: int, delta: float, sigma: float) -> float:
    if n <= 0 or delta <= 0 or sigma <= 0:
        return 0.0
    critical = NORMAL.inv_cdf(1.0 - PER_CLAIM_ALPHA)
    noncentral = delta * math.sqrt(n) / sigma
    return float(NORMAL.cdf(noncentral - critical))


def upper_95_sd(sd_ddof1: float, episode_count: int) -> float:
    """One-sided normal-theory 95% upper confidence bound for sigma."""

    sd = float(sd_ddof1)
    n = int(episode_count)
    if n < 2 or not math.isfinite(sd) or sd < 0:
        raise ValueError(f"invalid SD summary: sd={sd!r}, n={n!r}")
    if sd == 0:
        return 0.0
    df = n - 1
    denominator = float(chi2.ppf(1.0 - SD_UPPER_CONFIDENCE, df))
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("chi-square SD upper-bound quantile is invalid")
    return float(math.sqrt(df * sd * sd / denominator))


def _grid_for_claims(claims: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for n in GRID:
        marginal: dict[str, dict[str, float]] = {}
        failure_sum = 0.0
        for regime in REGIMES:
            marginal[regime] = {}
            for endpoint in ENDPOINTS:
                item = claims[regime][endpoint]
                power = one_sided_mean_power(
                    n,
                    float(item["delta"] or 0.0),
                    float(item["sigma"]),
                )
                marginal[regime][endpoint] = power
                failure_sum += 1.0 - power
        family_lower = max(0.0, 1.0 - failure_sum)
        grid.append(
            {
                "episodes_per_regime": n,
                "marginal_power": marginal,
                "union_bound_family_power_lower": float(family_lower),
                "eligible": bool(family_lower >= FAMILY_POWER_TARGET),
            }
        )
    return grid


def inherited_family8_yardstick(
    frozen_endpoint_moments: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Reproduce the nonbinding V5-moment family-8 planning yardstick."""

    claims = {
        regime: {
            endpoint: {
                "delta": EFFECT_RETAINED_FRACTION
                * float(frozen_endpoint_moments[endpoint]["mean"]),
                "sigma": float(frozen_endpoint_moments[endpoint]["sd_ddof1"]),
            }
            for endpoint in ENDPOINTS
        }
        for regime in REGIMES
    }
    grid = _grid_for_claims(claims)
    selected = next((item for item in grid if item["eligible"]), None)
    selected_n = None if selected is None else int(selected["episodes_per_regime"])
    return {
        "binding": False,
        "purpose": (
            "inherited planning yardstick only; it cannot authorize confirmation "
            "and is superseded by the binding fit/selection rule"
        ),
        "source": "exact hash-pinned endpoint moments in local power_rule.json",
        "effect_fraction": EFFECT_RETAINED_FRACTION,
        "family_size": FAMILY_SIZE,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "target_union_bound_family_power": FAMILY_POWER_TARGET,
        "grid": grid,
        "first_qualifying_episodes_per_regime": selected_n,
        "expected_first_qualifying_episodes_per_regime": INHERITED_EXPECTED_MINIMUM,
        "reproduces_family8_inherited_yardstick": selected_n
        == INHERITED_EXPECTED_MINIMUM,
    }


IDENTITY_HASH_FIELDS = (
    "selected_candidate_object_sha256",
    "fit_lock_sha256",
    "selection_ledger_sha256",
    "gate_freeze_sha256",
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_summary_identity(
    fit_summary: Mapping[str, Any], selection_summary: Mapping[str, Any]
) -> dict[str, str]:
    """Bind both role summaries to one already-frozen selected gate."""

    summaries = {"fit": fit_summary, "selection": selection_summary}
    for role, summary in summaries.items():
        checks = {
            "schema": summary.get("schema_version") == 1,
            "attempt": summary.get("attempt") == ACTIVE_ATTEMPT,
            "role": summary.get("role") == role,
            "co_primary_only": summary.get("co_primary_only") is True,
            "auxiliary_excluded": summary.get(
                "auxiliary_robust_whitening_excluded"
            )
            is True,
            "candidate_id": isinstance(summary.get("selected_candidate_id"), str)
            and bool(summary.get("selected_candidate_id")),
            **{
                field: _is_sha256(summary.get(field))
                for field in IDENTITY_HASH_FIELDS
            },
        }
        if not all(checks.values()):
            raise ValueError(
                f"{role} power-summary identity contract failed: "
                f"{[name for name, passed in checks.items() if not passed]}"
            )
    identity_fields = ("selected_candidate_id", *IDENTITY_HASH_FIELDS)
    identity = {field: str(fit_summary[field]) for field in identity_fields}
    drift = [
        field for field in identity_fields if selection_summary.get(field) != identity[field]
    ]
    if drift:
        raise ValueError(f"fit/selection power-summary identity drift: {drift}")
    return identity


def validate_identity_artifacts(
    identity: Mapping[str, str],
    *,
    fit_lock_path: Path,
    selection_ledger_path: Path,
    gate_freeze_path: Path,
) -> dict[str, dict[str, Any]]:
    """Independently hash and cross-check the three selected-gate locks."""

    paths = {
        "fit_lock": _regular_file(fit_lock_path),
        "selection_ledger": _regular_file(selection_ledger_path),
        "gate_freeze": _regular_file(gate_freeze_path),
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    expected = {
        "fit_lock": identity["fit_lock_sha256"],
        "selection_ledger": identity["selection_ledger_sha256"],
        "gate_freeze": identity["gate_freeze_sha256"],
    }
    if observed != expected:
        raise RuntimeError(
            "selected-gate identity artifact hash drift: "
            f"observed={observed}, expected={expected}"
        )
    fit_lock = json.loads(paths["fit_lock"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection_ledger"].read_text(encoding="utf-8"))
    freeze = json.loads(paths["gate_freeze"].read_text(encoding="utf-8"))
    if not all(isinstance(item, Mapping) for item in (fit_lock, selection, freeze)):
        raise RuntimeError("selected-gate identity artifact is not a JSON object")
    selected_row = next(
        (
            row
            for row in selection.get("candidates", [])
            if isinstance(row, Mapping)
            and row.get("candidate_id") == identity["selected_candidate_id"]
        ),
        None,
    )
    checks = {
        "fit_lock_status": fit_lock.get("status")
        == "all_24_candidates_fit_compiled_and_locked_before_selection_open",
        "fit_lock_no_selection": fit_lock.get("selection_input_opened_before_lock")
        is False,
        "selection_status": selection.get("status")
        == "selection_complete_no_selected_head_refit",
        "selection_candidate": selection.get("selected_candidate_id")
        == identity["selected_candidate_id"],
        "selection_no_refit": selection.get("selected_head_refit_after_selection")
        is False,
        "selection_fit_lock": selection.get("fit_lock_sha256")
        == identity["fit_lock_sha256"],
        "selection_candidate_object": isinstance(selected_row, Mapping)
        and selected_row.get("candidate_object_sha256")
        == identity["selected_candidate_object_sha256"],
        "selection_candidate_eligible": isinstance(selected_row, Mapping)
        and selected_row.get("eligible") is True,
        "freeze_status": freeze.get("status")
        == "selected_gate_frozen_before_smoke_or_confirmation",
        "freeze_candidate": freeze.get("selected_candidate_id")
        == identity["selected_candidate_id"],
        "freeze_candidate_object": freeze.get("selected_candidate_object_sha256")
        == identity["selected_candidate_object_sha256"],
        "freeze_fit_lock": freeze.get("fit_lock_sha256")
        == identity["fit_lock_sha256"],
        "freeze_selection": freeze.get("selection_ledger_sha256")
        == identity["selection_ledger_sha256"],
        "freeze_no_refit": freeze.get("selected_head_refit_after_selection") is False,
        "freeze_zero_confirmation": freeze.get("confirmation_episodes_at_freeze")
        == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "selected-gate identity cross-link failure: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    return {
        name: {
            "path": _repo_relative(path),
            "sha256": observed[name],
        }
        for name, path in paths.items()
    }


def normalize_claim_summary(
    document: Mapping[str, Any],
    *,
    expected_episode_count: int,
    role: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    claims = document.get("claims")
    if not isinstance(claims, Mapping) or set(claims) != set(REGIMES):
        raise ValueError(f"{role} claims must contain exactly the four frozen regimes")
    normalized: dict[str, dict[str, dict[str, float | int]]] = {}
    for regime in REGIMES:
        by_endpoint = claims[regime]
        if not isinstance(by_endpoint, Mapping) or set(by_endpoint) != set(ENDPOINTS):
            raise ValueError(
                f"{role} {regime} must contain exactly the two co-primary endpoints"
            )
        normalized[regime] = {}
        for endpoint in ENDPOINTS:
            item = by_endpoint[endpoint]
            if not isinstance(item, Mapping):
                raise ValueError(f"{role} {regime}/{endpoint} is not a summary mapping")
            count = item.get("episode_count")
            raw_mean = item.get("mean")
            raw_sd = item.get("sd_ddof1")
            if type(count) is not int:
                raise ValueError(
                    f"{role} {regime}/{endpoint} episode_count must be an exact JSON integer"
                )
            if type(raw_mean) not in (int, float) or type(raw_sd) not in (int, float):
                raise ValueError(
                    f"{role} {regime}/{endpoint} mean and sd_ddof1 must be non-bool JSON numbers"
                )
            try:
                mean = float(raw_mean)
                sd = float(raw_sd)
            except OverflowError as exc:
                raise ValueError(
                    f"nonfinite/negative {role} moment for {regime}/{endpoint}"
                ) from exc
            if count != expected_episode_count:
                raise ValueError(
                    f"{role} {regime}/{endpoint} count {count} != {expected_episode_count}"
                )
            if not math.isfinite(mean) or not math.isfinite(sd) or sd < 0:
                raise ValueError(f"nonfinite/negative {role} moment for {regime}/{endpoint}")
            normalized[regime][endpoint] = {
                "episode_count": count,
                "mean": mean,
                "sd_ddof1": sd,
            }
    return normalized


def compute_binding_power(
    fit_summary: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
    frozen_endpoint_moments: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    identity = validate_summary_identity(fit_summary, selection_summary)
    fit = normalize_claim_summary(
        fit_summary,
        expected_episode_count=FIT_EPISODES_PER_REGIME,
        role="fit",
    )
    selection = normalize_claim_summary(
        selection_summary,
        expected_episode_count=SELECTION_EPISODES_PER_REGIME,
        role="selection",
    )
    yardstick = inherited_family8_yardstick(frozen_endpoint_moments)
    if not yardstick["reproduces_family8_inherited_yardstick"]:
        raise RuntimeError("sealed consumed moments no longer reproduce family-8 N=3500")

    design_claims: dict[str, dict[str, dict[str, Any]]] = {}
    all_positive = True
    for regime in REGIMES:
        design_claims[regime] = {}
        for endpoint in ENDPOINTS:
            fit_item = fit[regime][endpoint]
            selection_item = selection[regime][endpoint]
            fit_mean = float(fit_item["mean"])
            selection_mean = float(selection_item["mean"])
            positive = fit_mean > 0.0 and selection_mean > 0.0
            all_positive = all_positive and positive
            delta = (
                EFFECT_RETAINED_FRACTION * min(fit_mean, selection_mean)
                if positive
                else None
            )
            fit_upper = upper_95_sd(
                float(fit_item["sd_ddof1"]), int(fit_item["episode_count"])
            )
            selection_upper = upper_95_sd(
                float(selection_item["sd_ddof1"]),
                int(selection_item["episode_count"]),
            )
            consumed_sd = float(frozen_endpoint_moments[endpoint]["sd_ddof1"])
            sigma = max(fit_upper, selection_upper, consumed_sd)
            design_claims[regime][endpoint] = {
                "fit": dict(fit_item),
                "selection": dict(selection_item),
                "both_means_strictly_positive": positive,
                "delta": delta,
                "delta_rule": (
                    ".35 * min(fit mean, selection mean), only when both are positive"
                ),
                "fit_sd_upper_95": fit_upper,
                "selection_sd_upper_95": selection_upper,
                "consumed_v5_endpoint_sd": consumed_sd,
                "sigma": sigma,
                "sigma_rule": (
                    "max(fit one-sided 95% SD upper bound, selection one-sided "
                    "95% SD upper bound, consumed V5 endpoint SD)"
                ),
            }

    grid = _grid_for_claims(design_claims)
    selected = next((item for item in grid if item["eligible"]), None)
    feasible = selected is not None and all_positive
    selected_n = int(selected["episodes_per_regime"]) if feasible else None
    result = {
        "schema_version": 1,
        "study": "lewm_domain_robust_gate",
        "attempt": ACTIVE_ATTEMPT,
        "selected_gate_identity": identity,
        "status": "binding_post_selection_power_result",
        "binding": True,
        "timing": (
            "computed after the selected gate and fit/selection summaries, and "
            "before opening any confirmation outcome"
        ),
        "method": (
            "one-sided normal paired-episode mean power at alpha .05/8; family "
            "power lower bounded by one minus the sum of eight marginal failure "
            "probabilities"
        ),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "family_size": FAMILY_SIZE,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "family_power_target": FAMILY_POWER_TARGET,
        "effect_retained_fraction": EFFECT_RETAINED_FRACTION,
        "sd_upper_confidence": SD_UPPER_CONFIDENCE,
        "grid": list(GRID),
        "claim_design_inputs": design_claims,
        "power_grid": grid,
        "selected_grid_record": selected if feasible else None,
        "selected_confirmation_episodes_per_regime": selected_n,
        "confirmation_episode_count_per_regime": (
            None
            if selected_n is None
            else {regime: selected_n for regime in REGIMES}
        ),
        "selected_confirmation_slots_per_regime": (
            None if selected_n is None else [0, selected_n - 1]
        ),
        "fixed_prefix_only": selected_n is not None,
        "all_eight_claims_have_positive_fit_and_selection_means": all_positive,
        "feasible": feasible,
        "passed": feasible,
        "decision": (
            "confirmation_size_fixed"
            if feasible
            else "power_infeasible_no_confirmation"
        ),
        "terminal_label_if_infeasible": (
            None if feasible else "domain_robust_gate_failed"
        ),
        "confirmation_generation_authorized_by_power": feasible,
        "no_sequential_expansion": True,
        "if_infeasible": (
            "generate zero confirmation outcomes; preserve this explicit result "
            "and stop without weakening, expansion, or retry"
        ),
        "inherited_family8_yardstick": yardstick,
        "inherited_yardstick_is_nonbinding": True,
        "fresh_confirmation_outcomes_opened": 0,
        "prior_outcome_arrays_opened_by_this_analysis": 0,
    }
    return result


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_binding_power(
    *,
    fit_summary_path: Path,
    selection_summary_path: Path,
    fit_lock_path: Path,
    selection_ledger_path: Path,
    gate_freeze_path: Path,
    power_rule_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate all frozen identities, compute once, and write immutably."""

    fit_path = _regular_file(fit_summary_path)
    selection_path = _regular_file(selection_summary_path)
    rule_path = _regular_file(power_rule_path)
    output_path = Path(output_path)
    _repo_relative(output_path, must_exist=False)
    if output_path.exists():
        raise RuntimeError(f"immutable power result already exists: {output_path}")
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(fit, Mapping) or not isinstance(selection, Mapping):
        raise RuntimeError("power summaries must be JSON objects")
    moments = load_frozen_power_rule_moments(rule_path)
    result = compute_binding_power(fit, selection, moments)
    identity_artifacts = validate_identity_artifacts(
        result["selected_gate_identity"],
        fit_lock_path=fit_lock_path,
        selection_ledger_path=selection_ledger_path,
        gate_freeze_path=gate_freeze_path,
    )
    gate_freeze = json.loads(_regular_file(gate_freeze_path).read_text(encoding="utf-8"))
    if not isinstance(gate_freeze, Mapping) or not isinstance(
        gate_freeze.get("created_unix_ns"), int
    ):
        raise RuntimeError("gate freeze lacks its immutable chronology timestamp")
    result["created_unix_ns"] = max(
        time.time_ns(), int(gate_freeze["created_unix_ns"]) + 1
    )
    result["inputs"] = {
        "fit_summary": {
            "path": _repo_relative(fit_path),
            "sha256": _sha256(fit_path),
        },
        "selection_summary": {
            "path": _repo_relative(selection_path),
            "sha256": _sha256(selection_path),
        },
        "power_rule": {
            "path": _repo_relative(rule_path),
            "sha256": _sha256(rule_path),
            "expected_sha256": POWER_RULE_SHA256,
            "endpoint_moments_object_sha256": POWER_RULE_MOMENTS_SHA256,
        },
        **identity_artifacts,
    }
    _write_exclusive(output_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-summary",
        type=Path,
        default=ATTEMPT_ROOT / "metrics/fit_power_summary.json",
    )
    parser.add_argument(
        "--selection-summary",
        type=Path,
        default=ATTEMPT_ROOT / "metrics/selection_power_summary.json",
    )
    parser.add_argument(
        "--power-rule", type=Path, default=POWER_RULE_PATH
    )
    parser.add_argument(
        "--fit-lock", type=Path, default=FIT_LOCK_PATH
    )
    parser.add_argument(
        "--selection-ledger", type=Path, default=SELECTION_LEDGER_PATH
    )
    parser.add_argument(
        "--gate-freeze", type=Path, default=GATE_FREEZE_PATH
    )
    parser.add_argument(
        "--output", type=Path, default=ATTEMPT_ROOT / "power_analysis.json"
    )
    args = parser.parse_args(argv)
    result = run_binding_power(
        fit_summary_path=args.fit_summary,
        selection_summary_path=args.selection_summary,
        fit_lock_path=args.fit_lock,
        selection_ledger_path=args.selection_ledger,
        gate_freeze_path=args.gate_freeze,
        power_rule_path=args.power_rule,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": _repo_relative(args.output),
                "decision": result["decision"],
                "selected_confirmation_episodes_per_regime": result[
                    "selected_confirmation_episodes_per_regime"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
