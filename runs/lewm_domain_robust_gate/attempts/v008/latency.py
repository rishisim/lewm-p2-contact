#!/usr/bin/env python3
"""Synchronized latency and resource reporting, separate from FLOP inference.

The benchmark measures the complete latent/action-input prediction path for
fixed depths 1--4, the actual adaptive sparse path, and a dense all-exit shadow.
It never substitutes latency for counted FLOPs or for a statistical endpoint.
The runner supplies a small adapter (``make_latency_adapter``) so this module
does not duplicate the frozen model or gate implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np


ATTEMPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ATTEMPT_ROOT.parents[3]
ACTIVE_ATTEMPT = ATTEMPT_ROOT.name
if ACTIVE_ATTEMPT != "v008":
    raise RuntimeError(f"latency.py must run from v008, got {ACTIVE_ATTEMPT!r}")
DGP_ORDER = (
    "native_plan",
    "markov_oracle",
    "plan_action_noise_0p2",
    "plan_random_action_0p1",
)
BATCH_SIZES = (1, 32, 256, 1024)
PATH_ORDER = (
    "fixed_depth_1",
    "fixed_depth_2",
    "fixed_depth_3",
    "fixed_depth_4",
    "actual_adaptive_sparse",
    "dense_all_exit_shadow",
)
NUMERICAL_RTOL = 1e-5
NUMERICAL_ATOL = 1e-6
NUMERICAL_MAX_ABS = 1e-5


class LatencyAdapter(Protocol):
    """Narrow protocol implemented by the frozen runner."""

    device: Any
    provenance: Mapping[str, Any]

    def synchronize(self) -> None: ...

    def module_audit(self) -> Mapping[str, Any]: ...

    def expected_calls(self, regime: str, batch_size: int) -> Any: ...

    def fixed_depth(self, regime: str, batch_size: int, depth: int) -> tuple[Any, Any]: ...

    def adaptive_sparse(self, regime: str, batch_size: int) -> tuple[Any, Any]: ...

    def dense_shadow(self, regime: str, batch_size: int) -> tuple[Any, Any]: ...


def _to_numpy(value: Any) -> np.ndarray:
    current = value
    for method in ("detach", "cpu"):
        function = getattr(current, method, None)
        if callable(function):
            current = function()
    return np.asarray(current)


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temporary(
    temporary: Path, destination: Path, *, exclusive: bool
) -> None:
    """Publish complete latency JSON with true exclusive creation."""

    if exclusive:
        os.link(temporary, destination, follow_symlinks=False)
    else:
        os.replace(temporary, destination)
    _fsync_parent_directory(destination)


def atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(temporary, path, exclusive=exclusive)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def build_input_binding(
    *,
    analysis_path: Path,
    analysis_bytes: bytes,
    runner_module_name: str,
    runner_module: Any,
    adapter: LatencyAdapter,
    requested_device: str,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Bind a latency artifact to every input that can alter its result."""

    source_name = getattr(runner_module, "__file__", None)
    if not source_name:
        raise RuntimeError("runner module lacks a source path")
    source_path = Path(source_name).resolve()
    if not source_path.is_file():
        raise RuntimeError("runner module source is not a regular file")
    # Round-trip now so non-JSON or nonfinite provenance fails before timing.
    provenance = json.loads(
        json.dumps(
            dict(adapter.provenance), allow_nan=False, sort_keys=True
        )
    )
    return {
        "analysis_result_path": _stable_path(analysis_path),
        "analysis_result_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
        "analysis_result_bytes": len(analysis_bytes),
        "runner_module": runner_module_name,
        "runner_source_path": _stable_path(source_path),
        "runner_source_sha256": sha256_file(source_path),
        "runner_provenance": provenance,
        "requested_device": requested_device,
        "resolved_device": str(adapter.device),
        "warmups": int(warmups),
        "repetitions": int(repetitions),
    }


def validate_latency_result(
    result: Mapping[str, Any], input_binding: Mapping[str, Any]
) -> dict[str, bool]:
    """Validate an immutable result before accepting it as resumable output."""

    regimes = result.get("regimes")
    regime_checks: list[bool] = []
    regime_passes: list[bool] = []
    if isinstance(regimes, Mapping) and tuple(regimes) == DGP_ORDER:
        for regime in DGP_ORDER:
            record = regimes[regime]
            if not isinstance(record, Mapping):
                regime_checks.append(False)
                regime_passes.append(False)
                continue
            equivalence = record.get("equivalence")
            expected_batches = (
                [item.get("batch_size") for item in equivalence]
                if isinstance(equivalence, list)
                and all(isinstance(item, Mapping) for item in equivalence)
                else []
            )
            equivalence_passed = bool(equivalence) and all(
                item.get("passed") is True for item in equivalence
            )
            regime_checks.append(
                expected_batches == list(BATCH_SIZES)
                and record.get("passed") is equivalence_passed
            )
            regime_passes.append(equivalence_passed)
    binding = dict(input_binding)
    expected_integrity = bool(
        result.get("module_before") == result.get("module_after")
        and len(regime_checks) == len(DGP_ORDER)
        and all(regime_checks)
        and all(regime_passes)
    )
    checks = {
        "schema": result.get("schema_version") == 1,
        "attempt": result.get("attempt") == ACTIVE_ATTEMPT,
        "checkpoint_state": result.get("checkpoint_state")
        == "LATENCY_AND_RESOURCE_REPORTING",
        "input_binding_exact": result.get("input_binding") == binding,
        "input_binding_hash": result.get("input_binding_sha256")
        == _canonical_sha256(binding),
        "runner_provenance_exact": result.get("runner_provenance")
        == binding.get("runner_provenance"),
        "device_exact": result.get("device") == binding.get("resolved_device"),
        "regimes_internally_consistent": len(regime_checks) == len(DGP_ORDER)
        and all(regime_checks),
        "integrity_exact": result.get("integrity_passed") is expected_integrity,
        "passed_exact": result.get("passed") is expected_integrity,
        "passed_implies_integrity": result.get("passed") is not True
        or result.get("integrity_passed") is True,
        "forbidden_outcomes_unopened": result.get(
            "contact_motion_phase_reward_success_opened"
        )
        is False,
    }
    return checks


def load_existing_result(
    path: Path, input_binding: Mapping[str, Any]
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise RuntimeError("existing latency artifact is not a JSON object")
    checks = validate_latency_result(result, input_binding)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            f"existing latency artifact is unbound or internally invalid: {failed}"
        )
    return result


def benchmark_path(
    synchronize: Callable[[], None],
    function: Callable[[], tuple[Any, Any]],
    *,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    """Benchmark one path with synchronization immediately around each trial."""

    if warmups < 0 or repetitions <= 0:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    output = calls = None
    for _ in range(warmups):
        output, calls = function()
        synchronize()
    measurements: list[float] = []
    for _ in range(repetitions):
        synchronize()
        started = time.perf_counter_ns()
        output, calls = function()
        synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e9
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("invalid synchronized latency observation")
        measurements.append(elapsed)
    if output is None or calls is None:
        raise RuntimeError("latency path produced no output")
    values = np.asarray(measurements, dtype=np.float64)
    return (
        {
            "warmups": int(warmups),
            "repetitions": int(repetitions),
            "all_seconds": values.tolist(),
            "median_seconds": float(np.median(values)),
            "p05_seconds": float(np.quantile(values, 0.05, method="linear")),
            "p95_seconds": float(np.quantile(values, 0.95, method="linear")),
            "minimum_seconds": float(values.min()),
            "maximum_seconds": float(values.max()),
            "synchronized": True,
        },
        (output, calls),
    )


def path_equivalence(
    sparse: tuple[Any, Any],
    dense: tuple[Any, Any],
    expected_calls: Any,
) -> dict[str, Any]:
    sparse_output, sparse_calls = map(_to_numpy, sparse)
    dense_output, dense_calls = map(_to_numpy, dense)
    expected = _to_numpy(expected_calls)
    sparse_output_finite = bool(np.isfinite(sparse_output).all())
    dense_output_finite = bool(np.isfinite(dense_output).all())
    if sparse_output.shape != dense_output.shape:
        maximum = None
        allclose = False
    elif not sparse_output_finite or not dense_output_finite:
        maximum = None
        allclose = False
    else:
        difference = np.abs(
            sparse_output.astype(np.float64) - dense_output.astype(np.float64)
        )
        maximum = float(difference.max(initial=0.0))
        allclose = bool(
            np.allclose(
                sparse_output,
                dense_output,
                rtol=NUMERICAL_RTOL,
                atol=NUMERICAL_ATOL,
            )
        )
    result = {
        "sparse_output_finite": sparse_output_finite,
        "dense_output_finite": dense_output_finite,
        "sparse_calls_match_execution_prefix": bool(np.array_equal(sparse_calls, expected)),
        "dense_calls_match_execution_prefix": bool(np.array_equal(dense_calls, expected)),
        "sparse_dense_calls_exact": bool(np.array_equal(sparse_calls, dense_calls)),
        "sparse_dense_output_allclose": allclose,
        "sparse_dense_max_abs": maximum,
    }
    result["passed"] = bool(
        result["sparse_calls_match_execution_prefix"]
        and result["dense_calls_match_execution_prefix"]
        and result["sparse_dense_calls_exact"]
        and result["sparse_dense_output_allclose"]
        and maximum is not None
        and maximum <= NUMERICAL_MAX_ABS
    )
    return result


def expected_analytic_mixture_latency(
    timings: Sequence[Mapping[str, Any]],
    allocation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lower_depth = int(allocation["depth_lower"])
    upper_depth = int(allocation["depth_upper"])
    weight = float(allocation["weight_upper"])
    output: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        lower = next(
            item
            for item in timings
            if item["path"] == f"fixed_depth_{lower_depth}"
            and item["batch_size"] == batch_size
        )
        upper = next(
            item
            for item in timings
            if item["path"] == f"fixed_depth_{upper_depth}"
            and item["batch_size"] == batch_size
        )
        seconds = (1.0 - weight) * float(lower["median_seconds"]) + weight * float(
            upper["median_seconds"]
        )
        output.append(
            {
                "batch_size": batch_size,
                "depth_lower": lower_depth,
                "depth_upper": upper_depth,
                "weight_upper": weight,
                "expected_median_seconds": seconds,
                "expected_throughput_rows_per_second": batch_size / seconds,
                "interpretation": (
                    "transition-independent expected fixed-depth-mixture latency; "
                    "descriptive only and not a FLOP substitution"
                ),
            }
        )
    return output


def energy_availability(adapter: Any) -> dict[str, Any]:
    """Report availability; never invent an energy estimate."""

    meter = getattr(adapter, "energy_meter", None)
    reliable = bool(getattr(meter, "reliable", False)) if meter is not None else False
    resettable = bool(getattr(meter, "resettable", False)) if meter is not None else False
    if meter is None or not reliable or not resettable:
        return {
            "available": False,
            "measured": False,
            "reliable_resettable_per_path_interface": False,
            "reason": (
                "no reliable resettable per-path energy interface is available "
                "in the frozen runtime"
            ),
        }
    # The adapter may expose a separately sealed implementation in a future
    # efficiency study.  This confirmation benchmark records availability but
    # intentionally does not let energy alter scientific inference.
    return {
        "available": True,
        "measured": False,
        "reliable_resettable_per_path_interface": True,
        "reason": "interface available; no energy observation requested by this runner",
    }


def run_latency_suite(
    adapter: LatencyAdapter,
    analysis_result: Mapping[str, Any],
    *,
    warmups: int = 2,
    repetitions: int = 7,
    input_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    before = dict(adapter.module_audit())
    if before.get("passed") is not True:
        raise RuntimeError("module audit failed before latency measurement")
    regimes: dict[str, Any] = {}
    for regime in DGP_ORDER:
        timings: list[dict[str, Any]] = []
        equivalence: list[dict[str, Any]] = []
        for batch_size in BATCH_SIZES:
            functions: list[tuple[str, Callable[[], tuple[Any, Any]]]] = [
                (
                    f"fixed_depth_{depth}",
                    lambda depth=depth: adapter.fixed_depth(regime, batch_size, depth),
                )
                for depth in range(1, 5)
            ] + [
                (
                    "actual_adaptive_sparse",
                    lambda: adapter.adaptive_sparse(regime, batch_size),
                ),
                (
                    "dense_all_exit_shadow",
                    lambda: adapter.dense_shadow(regime, batch_size),
                ),
            ]
            observed: dict[str, tuple[Any, Any]] = {}
            for path_name, function in functions:
                measurement, outputs = benchmark_path(
                    adapter.synchronize,
                    function,
                    warmups=warmups,
                    repetitions=repetitions,
                )
                measurement.update(
                    {
                        "path": path_name,
                        "batch_size": batch_size,
                        "throughput_rows_per_second_at_median": batch_size
                        / measurement["median_seconds"],
                    }
                )
                timings.append(measurement)
                observed[path_name] = outputs
            record = path_equivalence(
                observed["actual_adaptive_sparse"],
                observed["dense_all_exit_shadow"],
                adapter.expected_calls(regime, batch_size),
            )
            record["batch_size"] = batch_size
            equivalence.append(record)
        expected_mixtures = {
            endpoint: expected_analytic_mixture_latency(
                timings,
                analysis_result["regimes"][regime]["analytic_allocations"][endpoint],
            )
            for endpoint in ("raw", "fixed_whitened")
        }
        regimes[regime] = {
            "timings": timings,
            "equivalence": equivalence,
            "analytic_expected_latency": expected_mixtures,
            "compute_resources": analysis_result["regimes"][regime]["compute"],
            "passed": all(item["passed"] for item in equivalence),
        }
    adapter.synchronize()
    after = dict(adapter.module_audit())
    energy = energy_availability(adapter)
    passed = before == after and all(item["passed"] for item in regimes.values())
    binding = dict(input_binding or {})
    return {
        "schema_version": 1,
        "attempt": ACTIVE_ATTEMPT,
        "checkpoint_state": "LATENCY_AND_RESOURCE_REPORTING",
        "created_unix_ns": time.time_ns(),
        "passed": passed,
        "integrity_passed": passed,
        "input_binding": binding,
        "input_binding_sha256": _canonical_sha256(binding),
        "device": str(adapter.device),
        "measurement": (
            "synchronized latent/action-input through frozen base prediction, "
            "solver, counted causal features, selected gate heads, and routing"
        ),
        "includes": [
            "frozen base prediction",
            "frozen stagewise solver",
            "counted causal feature graph",
            "selected compiled affine gate heads",
            "routing and indexing",
            "device synchronization",
        ],
        "excludes": [
            "simulator rollout",
            "pixel encoder",
            "FLOP-based terminal inference",
            "statistical terminal mapping",
        ],
        "regimes": regimes,
        "module_before": before,
        "module_after": after,
        "runner_provenance": dict(adapter.provenance),
        "energy": energy,
        "latency_reported_separately_from_fully_counted_flops": True,
        "latency_and_energy_excluded_from_terminal_mapping": True,
        "contact_motion_phase_reward_success_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--runner-module", default="runner")
    arguments = parser.parse_args()
    output = ATTEMPT_ROOT / "metrics/latency_and_resources.json"
    analysis_path = ATTEMPT_ROOT / "analysis_result.json"
    analysis_bytes = analysis_path.read_bytes()
    analysis_result = json.loads(analysis_bytes)
    if not isinstance(analysis_result, dict):
        raise RuntimeError("analysis result is not a JSON object")
    module = importlib.import_module(arguments.runner_module)
    factory = getattr(module, "make_latency_adapter", None)
    if not callable(factory):
        raise RuntimeError(
            f"{arguments.runner_module} lacks frozen make_latency_adapter(device)"
        )
    adapter = factory(arguments.device)
    binding = build_input_binding(
        analysis_path=analysis_path,
        analysis_bytes=analysis_bytes,
        runner_module_name=arguments.runner_module,
        runner_module=module,
        adapter=adapter,
        requested_device=arguments.device,
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
    )
    if output.exists():
        result = load_existing_result(output, binding)
        print(output.read_text(encoding="utf-8"), end="")
        if not result["passed"]:
            raise SystemExit(2)
        return
    result = run_latency_suite(
        adapter,
        analysis_result,
        warmups=arguments.warmups,
        repetitions=arguments.repetitions,
        input_binding=binding,
    )
    generated_checks = validate_latency_result(result, binding)
    if not all(generated_checks.values()):
        failed = sorted(
            name for name, passed in generated_checks.items() if not passed
        )
        raise RuntimeError(f"generated latency artifact failed validation: {failed}")
    atomic_json(output, result, exclusive=True)
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
