"""Sealed one-shot V3 calibration judge for the frozen compression winner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

import common
import protocol
import students


SOURCE_H5 = Path("/Users/rishisim/.stable_worldmodel/source_downloads/lewm-cube/extracted/cube_single_expert.h5")
BASE_CONFIG = common.REPO / "runs/lewm_transfer/cube/cache/model/config.json"
BASE_WEIGHTS = common.REPO / "runs/lewm_transfer/cube/cache/model/weights.pt"
CALIBRATION_CACHE = common.ROOT / "cache/v3_calibration_only.npz"
CALIBRATION_MANIFEST = common.ROOT / "cache/v3_calibration_only_manifest.json"
RECEIPT = common.ROOT / "audit/calibration_access_receipt.json"
WHITENING = common.ROOT / "checkpoints/discovery_whitening.npz"


def load_model_io() -> Any:
    path = common.REPO / "runs/lewm_adaptive_compute_v2/model_io.py"
    spec = importlib.util.spec_from_file_location("critic_compression_model_io", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen model I/O")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_whitening_artifact() -> dict[str, Any]:
    """Freeze all-discovery whitening before any calibration receipt exists."""
    if RECEIPT.exists():
        raise RuntimeError("cannot create whitening after calibration access")
    if WHITENING.exists():
        with np.load(WHITENING, allow_pickle=False) as stored:
            required = {"mean", "matrix", "eigenvalues", "floor"}
            if set(stored.files) != required:
                raise RuntimeError("frozen whitening artifact is malformed")
        return {"path": str(WHITENING.resolve()), "sha256": common.sha256_file(WHITENING)}
    arrays = common.load_train_arrays()
    whitening = common.fit_whitening(arrays["target"])
    WHITENING.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        WHITENING,
        mean=whitening["mean"],
        matrix=whitening["matrix"],
        eigenvalues=whitening["eigenvalues"],
        floor=np.asarray(whitening["floor"], dtype=np.float64),
    )
    return {"path": str(WHITENING.resolve()), "sha256": common.sha256_file(WHITENING)}


def validate_frozen_tournament() -> dict[str, Any]:
    path = common.ROOT / "audit/frozen_tournament.json"
    digest_path = common.ROOT / "audit/frozen_tournament.sha256.json"
    if not path.exists() or not digest_path.exists():
        raise RuntimeError("no frozen passing tournament")
    expected = json.loads(digest_path.read_text())["sha256"]
    observed = common.sha256_file(path)
    if observed != expected:
        raise RuntimeError("frozen tournament hash drift")
    tournament = json.loads(path.read_text())
    if tournament["status"] != "frozen_before_v3_calibration_target_access":
        raise RuntimeError("invalid tournament status")
    checks = {
        common.ROOT / "config.json": tournament["config_sha256"],
        common.ROOT / "PLAN.md": tournament["plan_sha256"],
        Path(tournament["student_checkpoint"]): tournament["student_checkpoint_sha256"],
        common.SOLVER_CHECKPOINT: tournament["solver_checkpoint_sha256"],
        common.TRAIN_CACHE: tournament["train_cache_sha256"],
        common.PREPARED_CACHE: tournament["prepared_cache_sha256"],
        common.ROOT / "metrics/discovery_cv.json": tournament["discovery_metrics_sha256"],
        Path(tournament["whitening_path"]): tournament["whitening_sha256"],
    }
    for artifact, artifact_hash in checks.items():
        if common.sha256_file(artifact) != artifact_hash:
            raise RuntimeError(f"frozen artifact drift: {artifact}")
    for name, source_hash in tournament["source_sha256"].items():
        if common.sha256_file(common.ROOT / name) != source_hash:
            raise RuntimeError(f"frozen judge source drift: {name}")
    tournament["tournament_sha256"] = observed
    return tournament


def create_receipt(tournament_sha256: str) -> dict[str, Any]:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "consumed_before_target_io",
        "tournament_sha256": str(tournament_sha256),
        "calibration_role": "v3_calibration_once",
        "v3_test_targets": "forbidden",
    }
    try:
        with RECEIPT.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError("V3 calibration access has already been consumed") from exc
    return payload


def extract_calibration_once(device: torch.device) -> dict[str, Any]:
    if CALIBRATION_CACHE.exists() or CALIBRATION_MANIFEST.exists():
        raise RuntimeError("calibration artifacts existed before the one-shot judge")
    _, _, _, isolation = common.load_prior_modules()
    sets = isolation.load_pinned_v3_episode_sets()
    selected = isolation.extraction_splits("calibration_once")
    forbidden = np.concatenate((sets["train"], sets["test"]))
    model_io = load_model_io()
    prior_cfg = json.loads((common.PRIOR / "config.json").read_text())
    model_io.extract_fresh_cube_cache(
        source_h5=SOURCE_H5,
        config_path=BASE_CONFIG,
        weights_path=BASE_WEIGHTS,
        output_npz=CALIBRATION_CACHE,
        split_manifest_path=CALIBRATION_MANIFEST,
        split_episodes=selected,
        selection_seed=int(prior_cfg["selection_seed"]),
        excluded_episode_ordinals=forbidden,
        reserved_episode_ordinals=(),
        prior_manifest_provenance={
            "v3_split_manifest": isolation.PINNED["manifest_sha256"],
            "strict_role": "calibration_once",
            "critic_compression_tournament": json.loads(RECEIPT.read_text())["tournament_sha256"],
            "v3_test_targets": "forbidden",
        },
        device=device,
        encode_batch_size=int(prior_cfg["encode_batch_size"]),
        predict_batch_size=int(prior_cfg["predict_batch_size"]),
    )
    with np.load(CALIBRATION_CACHE, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    audit = isolation.assert_isolated_cache(arrays, "calibration_once")
    if audit["episodes"] != 90 or audit["test_episodes_present"]:
        raise RuntimeError("calibration isolation audit failed")
    return {
        "cache_sha256": common.sha256_file(CALIBRATION_CACHE),
        "manifest_sha256": common.sha256_file(CALIBRATION_MANIFEST),
        "audit": audit,
    }


def load_calibration_arrays() -> dict[str, np.ndarray]:
    with np.load(CALIBRATION_CACHE, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    _, _, _, isolation = common.load_prior_modules()
    isolation.assert_isolated_cache(arrays, "calibration_once")
    return arrays


@torch.inference_mode()
def solver_outputs(arrays: Mapping[str, np.ndarray], device: torch.device) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model, v1, _, model_module = common.load_frozen_solver(device)
    solver_hash_before = common.state_hash(model)
    n = len(arrays["episode_id"])
    exits = np.empty((n, 4, 192), dtype=np.float32)
    features = np.empty((n, 3, 1046), dtype=np.float32)
    identity = d0_equal = d1_equal = True
    cursor = 0
    for part in common.iter_batches(np.arange(n), 1024):
        h = torch.as_tensor(np.ascontiguousarray(arrays["history"][part]), device=device)
        a = torch.as_tensor(np.ascontiguousarray(arrays["action"][part]), device=device)
        z = torch.as_tensor(np.ascontiguousarray(arrays["base_pred"][part]), device=device)
        output, updates = model(h, a, z, max_depth=4, return_updates=True)
        expected = v1(h, a, z, depths=(0, 1))
        identity = identity and output[0] is z
        d0_equal = d0_equal and bool(torch.equal(output[0], expected[0]))
        d1_equal = d1_equal and bool(torch.equal(output[1], expected[1]))
        b = len(part)
        exits[cursor : cursor + b] = torch.stack([output[d] for d in (1, 2, 3, 4)], 1).cpu().numpy()
        for stage, depth in enumerate((1, 2, 3)):
            features[cursor : cursor + b, stage] = model_module.build_causal_features(
                h, a, output[depth], updates[depth]
            ).cpu().numpy()
        cursor += b
    audit = {
        "d0_identity": identity,
        "d0_bitwise": d0_equal,
        "d1_bitwise": d1_equal,
        "solver_state_sha256_before": solver_hash_before,
        "solver_state_sha256_after": common.state_hash(model),
        "all_solver_gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
    }
    audit["passed"] = bool(
        identity and d0_equal and d1_equal
        and audit["solver_state_sha256_before"] == audit["solver_state_sha256_after"]
        and audit["all_solver_gradients_absent"]
    )
    if not audit["passed"]:
        raise RuntimeError("calibration solver preservation audit failed")
    return exits, features, audit


def load_student(tournament: Mapping[str, Any], device: torch.device) -> students.FrozenStudent:
    checkpoint = torch.load(tournament["student_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["family"] != tournament["family"] or checkpoint["operating_point"] != tournament["operating_point"]:
        raise RuntimeError("frozen student metadata mismatch")
    if float(checkpoint["compute_price"]) != float(tournament["compute_price"]):
        raise RuntimeError("frozen compute price mismatch")
    student = students.FrozenStudent.from_payload(checkpoint["student"])
    if not student.build(device).training is False:
        raise RuntimeError("student did not load in evaluation mode")
    return student


def judge(device: torch.device) -> dict[str, Any]:
    tournament = validate_frozen_tournament()
    receipt = create_receipt(tournament["tournament_sha256"])
    extraction = extract_calibration_once(device)
    arrays = load_calibration_arrays()
    exits, features, solver_audit = solver_outputs(arrays, device)
    target = np.asarray(arrays["target"], dtype=np.float32)
    episodes = np.asarray(arrays["episode_id"], dtype=np.int64)
    losses = np.square(exits - target[:, None, :]).mean(2).astype(np.float64)
    with np.load(tournament["whitening_path"], allow_pickle=False) as stored:
        whitening = {name: stored[name] for name in stored.files}
    white_losses = common.whitened_losses(target, exits, whitening)
    student = load_student(tournament, device)
    scores = students.predict_student(student, features, device)
    calls = protocol.sequential_calls(scores, float(tournament["compute_price"]))
    policy = protocol.prior_policy()
    discovery_means = np.asarray(tournament["discovery_fixed_exit_raw_mse"], dtype=np.float64)
    exact = policy.strongest_transition_independent_baseline(
        discovery_means, protocol.SUPPORTED_CALLS,
        n=len(calls), target_total_calls=int(calls.sum()), seed=int(tournament["baseline_seeds"]["matched"]),
    )
    analytic = policy.optimal_expected_mixture(discovery_means, protocol.SUPPORTED_CALLS, float(calls.mean()))
    histogram_calls = policy.randomized_histogram_control(calls, int(tournament["baseline_seeds"]["histogram"]))
    adaptive = protocol.selected_loss(losses, calls)
    matched = protocol.selected_loss(losses, exact["selected_calls"])
    analytic_loss = np.einsum("nd,d->n", losses, analytic["probabilities"], optimize=False)
    histogram = protocol.selected_loss(losses, histogram_calls)
    white_adaptive = protocol.selected_loss(white_losses, calls)
    white_matched = protocol.selected_loss(white_losses, exact["selected_calls"])
    permuted_scores = policy.permute_critic_scores(scores, int(tournament["baseline_seeds"]["score_permutation"]), episode_ids=episodes)
    permuted_calls = protocol.sequential_calls(permuted_scores, float(tournament["compute_price"]))
    permuted = protocol.selected_loss(losses, permuted_calls)
    permuted_baseline = policy.strongest_transition_independent_baseline(
        discovery_means, protocol.SUPPORTED_CALLS,
        n=len(permuted_calls), target_total_calls=int(permuted_calls.sum()), seed=int(tournament["baseline_seeds"]["permuted_matched"]),
    )
    permuted_matched = protocol.selected_loss(losses, permuted_baseline["selected_calls"])
    bootstrap = int(tournament["bootstrap_samples"])
    seed = int(tournament["bootstrap_seed"])
    def ci(candidate: np.ndarray, baseline: np.ndarray, offset: int) -> dict[str, Any]:
        return policy.clustered_paired_loss_ci(candidate, baseline, episodes, samples=bootstrap, seed=seed + offset)
    gate = tournament["gate_cost"]
    cfg = common.load_config()
    mean_calls = float(calls.mean())
    mean_gate_decisions = float(np.minimum(calls, 3).mean())
    adaptive_flops = float(
        cfg["base_predict_flops"] + cfg["v1_call_flops"]
        + (mean_calls - 1.0) * cfg["stage_adapter_flops"]
        + mean_gate_decisions * gate["total_incremental_gate_flops_per_evaluated_decision"]
    )
    matched_flops = float(
        cfg["base_predict_flops"] + cfg["v1_call_flops"]
        + (mean_calls - 1.0) * cfg["stage_adapter_flops"]
    )
    frontier = []
    for depth in range(1, 5):
        frontier.append({
            "name": f"fixed_d{depth}", "mean_calls": float(depth),
            "raw_mse": float(losses[:, depth - 1].mean()),
            "total_flops": float(cfg["base_predict_flops"] + cfg["v1_call_flops"] + (depth - 1) * cfg["stage_adapter_flops"]),
        })
    frontier.extend((
        {"name": "adaptive", "mean_calls": mean_calls, "raw_mse": float(adaptive.mean()), "total_flops": adaptive_flops},
        {"name": "matched", "mean_calls": mean_calls, "raw_mse": float(matched.mean()), "total_flops": matched_flops},
    ))
    call_mask = policy.nondominated_mask([row["mean_calls"] for row in frontier], [row["raw_mse"] for row in frontier])
    flop_mask = policy.nondominated_mask([row["total_flops"] for row in frontier], [row["raw_mse"] for row in frontier])
    for row, call_ok, flop_ok in zip(frontier, call_mask, flop_mask, strict=True):
        row["call_nondominated"] = bool(call_ok)
        row["flop_nondominated"] = bool(flop_ok)
    adaptive_frontier = next(row for row in frontier if row["name"] == "adaptive")
    result = {
        "schema_version": 1,
        "family": tournament["family"],
        "operating_point": tournament["operating_point"],
        "compute_price": float(tournament["compute_price"]),
        "rows": int(len(calls)),
        "episodes": int(len(np.unique(episodes))),
        "mean_calls": mean_calls,
        "total_calls": int(calls.sum()),
        "call_histogram": {str(int(value)): int((calls == value).sum()) for value in protocol.SUPPORTED_CALLS},
        "raw_mse": float(adaptive.mean()),
        "matched_raw_mse": float(matched.mean()),
        "analytic_raw_mse": float(analytic_loss.mean()),
        "histogram_raw_mse": float(histogram.mean()),
        "fixed_exit_raw_mse": losses.mean(0),
        "whitened_mse": float(white_adaptive.mean()),
        "vs_matched_randomized": ci(adaptive, matched, 0),
        "vs_analytic_mixture": ci(adaptive, analytic_loss, 11),
        "vs_fixed_d1": ci(adaptive, losses[:, 0], 17),
        "vs_histogram_null": ci(adaptive, histogram, 23),
        "whitened_vs_matched": ci(white_adaptive, white_matched, 31),
        "score_permutation_vs_own_matched": ci(permuted, permuted_matched, 37),
        "exact_call_audit": exact["audit"],
        "analytic_probabilities": analytic["probabilities"],
        "ranking_diagnostics": protocol.ranking_diagnostics(scores, protocol.gains_from_losses(losses)),
        "solver_audit": solver_audit,
        "causal_feature_audit": common.causal_feature_audit(),
        "gate_cost": gate,
        "adaptive_total_flops_per_transition": adaptive_flops,
        "matched_total_flops_per_transition": matched_flops,
        "frontier": frontier,
        "call_nondominated": adaptive_frontier["call_nondominated"],
        "flop_nondominated": adaptive_frontier["flop_nondominated"],
        "receipt": receipt,
        "extraction": extraction,
        "v3_test_targets_consumed": False,
    }
    result["passed"] = bool(
        all(result[name]["ci_low"] > 0 for name in (
            "vs_matched_randomized", "vs_analytic_mixture", "vs_fixed_d1", "vs_histogram_null"
        ))
        and result["whitened_vs_matched"]["mean_benefit"] > 0
        and result["call_nondominated"] and result["flop_nondominated"]
        and result["exact_call_audit"]["exact_total_match"]
        and result["solver_audit"]["passed"]
        and result["causal_feature_audit"]["passed"]
        and result["gate_cost"]["within_budget"]
    )
    common.write_json(common.ROOT / "metrics/calibration_once.json", result)
    decision = {
        "decision": "critic_compression_discovery_passed" if result["passed"] else "critic_compression_calibration_failed",
        "phase": "one_shot_v3_calibration",
        "winner": tournament["family"],
        "operating_point": tournament["operating_point"],
        "calibration_consumed_once": True,
        "calibration_passed": bool(result["passed"]),
        "v3_test_targets_consumed": False,
        "v4_created": False,
        "recommendation": "separate preregistered V4 confirmation" if result["passed"] else "do not proceed to V4 with this gate",
        "tournament_sha256": tournament["tournament_sha256"],
    }
    common.write_json(common.ROOT / "decision.json", decision)
    return {"decision": decision, "calibration": result}
