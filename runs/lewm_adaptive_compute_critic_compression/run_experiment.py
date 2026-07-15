#!/usr/bin/env python3
"""Run the nested critic-compression tournament and sealed one-shot judge."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import common
import protocol
import students


def log(message: str) -> None:
    print(f"[critic-compression] {message}", flush=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [common.jsonable(dict(row)) for row in rows]
    fields = sorted({key for row in normalized for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def config_key(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def strict_teacher_maps(
    features: np.ndarray,
    gains: np.ndarray,
    episodes: np.ndarray,
    assignment: np.ndarray,
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    epoch_override: int | None = None,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    """Teacher labels for an inner split with the held fold excluded entirely."""
    folds = int(cfg["inner_folds"])
    maps: dict[int, np.ndarray] = {}
    audits = []
    for held in range(folds):
        labels = np.full_like(gains, np.nan, dtype=np.float64)
        remaining = [fold for fold in range(folds) if fold != held]
        for index, target_fold in enumerate(remaining):
            source_folds = [fold for fold in remaining if fold != target_fold]
            source = np.isin(assignment, source_folds)
            target = assignment == target_fold
            if set(np.unique(episodes[source])) & set(np.unique(episodes[target])):
                raise RuntimeError("teacher target/source episode overlap")
            if np.any(assignment[source] == held):
                raise RuntimeError("inner held fold entered teacher training")
            teacher = students.fit_teacher(
                features[source], gains[source],
                seed=int(cfg["teacher_seeds"][index % len(cfg["teacher_seeds"])]) + held * 10007 + target_fold * 101,
                device=device,
                hidden_dims=cfg["teacher_hidden_dims"],
                epochs=int(epoch_override or cfg["teacher_epochs"]),
                batch_size=int(cfg["batch_size"]),
                learning_rate=float(cfg["learning_rate"]),
            )
            labels[target] = students.predict_student(teacher, features[target], device)
            audits.append({
                "held_inner_fold": held,
                "teacher_target_fold": target_fold,
                "teacher_source_folds": source_folds,
                "source_episodes": int(len(np.unique(episodes[source]))),
                "target_episodes": int(len(np.unique(episodes[target]))),
                "held_episodes_absent_from_source": True,
            })
        train = assignment != held
        if not np.isfinite(labels[train]).all() or np.isfinite(labels[~train]).any():
            raise RuntimeError("strict teacher-map coverage failed")
        maps[held] = labels
    return maps, audits


def fit_fold_student(
    features: np.ndarray,
    gains: np.ndarray,
    teacher_labels: np.ndarray,
    episodes: np.ndarray,
    *,
    family: str,
    candidate: Mapping[str, Any],
    seed: int,
    cfg: Mapping[str, Any],
    device: torch.device,
    epoch_override: int | None = None,
) -> tuple[students.FrozenStudent, dict[str, Any] | None]:
    selection = None
    selection_audit = None
    if family == "stable_sparse":
        weight = float(candidate.get("teacher_weight", 0.7))
        selection_target = (1.0 - weight) * gains + weight * teacher_labels
        selection, selection_audit = students.stable_feature_selection(
            features, selection_target, episodes,
            k=int(candidate["sparse_k"]), folds=int(cfg["inner_folds"]),
            seed=int(cfg["inner_seed"]) + int(seed),
        )
    fitted = students.fit_student(
        features, gains, teacher_labels,
        family=family, config=candidate, seed=int(seed), device=device,
        feature_selection=selection,
        epochs=int(epoch_override or cfg["student_epochs"]),
        batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
    )
    return fitted, selection_audit


def tune_family_inner(
    features: np.ndarray,
    gains: np.ndarray,
    losses: np.ndarray,
    episodes: np.ndarray,
    assignment: np.ndarray,
    teacher_maps: Mapping[int, np.ndarray],
    *,
    family: str,
    candidates: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    device: torch.device,
    seed_offset: int,
    epoch_override: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    candidate_rows = []
    oof_by_key: dict[str, np.ndarray] = {}
    for candidate_index, candidate in enumerate(candidates):
        key = config_key(candidate)
        oof = np.full_like(gains, np.nan, dtype=np.float64)
        sparse_audits = []
        for held in range(int(cfg["inner_folds"])):
            train = assignment != held
            evaluate = ~train
            teacher_labels = teacher_maps[held]
            if not np.isfinite(teacher_labels[train]).all():
                raise RuntimeError("inner student received missing teacher labels")
            model, sparse_audit = fit_fold_student(
                features[train], gains[train], teacher_labels[train], episodes[train],
                family=family, candidate=candidate,
                seed=int(cfg["student_seeds"][held % len(cfg["student_seeds"])]) + seed_offset + candidate_index * 1009,
                cfg=cfg, device=device, epoch_override=epoch_override,
            )
            oof[evaluate] = students.predict_student(model, features[evaluate], device)
            if sparse_audit is not None:
                selected = set(int(value) for value in model.feature_indices)
                if any(value >= 1035 for value in selected):
                    raise RuntimeError("sparse selector used a derived summary")
                sparse_audits.append({"held_inner_fold": held, **sparse_audit})
        if not np.isfinite(oof).all():
            raise RuntimeError("student inner cross-fit left missing predictions")
        utility = protocol.inner_routing_utility(oof, losses, cfg["target_mean_calls"])
        candidate_rows.append({
            "config_key": key,
            "config": dict(candidate),
            "inner_routing": utility,
            "sparse_selection_audits": sparse_audits,
        })
        oof_by_key[key] = oof
    chosen = max(candidate_rows, key=lambda row: (row["inner_routing"]["utility"], row["config_key"]))
    return dict(chosen["config"]), oof_by_key[chosen["config_key"]], {
        "selected_config_key": chosen["config_key"],
        "selected_config": chosen["config"],
        "candidates": candidate_rows,
    }


def empty_repeat_payload(n: int) -> dict[str, np.ndarray | list[Any]]:
    names_float = (
        "adaptive_loss", "matched_loss", "analytic_loss", "histogram_loss",
        "permuted_loss", "permuted_matched_loss", "oracle_loss",
        "white_adaptive_loss", "white_matched_loss", "white_analytic_loss",
        "white_histogram_loss",
    )
    result: dict[str, Any] = {name: np.full(n, np.nan, dtype=np.float64) for name in names_float}
    for name in ("calls", "matched_calls", "histogram_calls", "permuted_calls", "oracle_calls"):
        result[name] = np.zeros(n, dtype=np.int64)
    result["fold_audits"] = []
    result["fold_benefits"] = []
    return result


def validate_repeat_payload(payload: Mapping[str, Any]) -> None:
    for name, values in payload.items():
        if name in ("fold_audits", "fold_benefits", "exact_call_audit"):
            continue
        array = np.asarray(values)
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise RuntimeError(f"repeat payload has missing {name}")
        if array.dtype.kind in "iu" and np.any(array <= 0):
            raise RuntimeError(f"repeat payload has invalid {name}")


def run_nested_cv(device: torch.device, *, smoke: bool = False) -> dict[str, Any]:
    cfg = common.load_config()
    prepared = common.load_prepared()
    train_arrays = common.load_train_arrays()
    features = prepared["features"]
    losses = prepared["losses"].astype(np.float64)
    gains = protocol.gains_from_losses(losses)
    episodes = prepared["episode_id"]
    targets = np.asarray(train_arrays["target"], dtype=np.float32)
    exits = np.asarray(prepared["exits"], dtype=np.float32)
    if smoke:
        chosen_episodes = np.unique(episodes)[:12]
        keep = np.isin(episodes, chosen_episodes)
        features, losses, gains, episodes, targets, exits = (
            value[keep] for value in (features, losses, gains, episodes, targets, exits)
        )
    n = len(episodes)
    repeats_count = 1 if smoke else int(cfg["outer_repeats"])
    families_cfg = {
        family: candidates[:1] if smoke else candidates
        for family, candidates in cfg["families"].items()
    }
    family_repeats: dict[str, dict[str, list[dict[str, Any]]]] = {
        family: {f"b{float(budget):.2f}": [] for budget in cfg["target_mean_calls"]}
        for family in families_cfg
    }
    family_scores = {family: [] for family in families_cfg}
    teacher_scores_repeats = []
    white_loss_repeats = []
    selection_records = []
    solver_hash_before = common.sha256_file(common.SOLVER_CHECKPOINT)
    for repeat in range(repeats_count):
        outer_assignment = common.episode_assignment(
            episodes, int(cfg["outer_folds"]), int(cfg["outer_seed"]) + repeat * 100003
        )
        repeat_scores = {family: np.full_like(gains, np.nan, dtype=np.float64) for family in families_cfg}
        repeat_teacher_scores = np.full_like(gains, np.nan, dtype=np.float64)
        repeat_white = np.full_like(losses, np.nan, dtype=np.float64)
        repeat_points = {
            family: {key: empty_repeat_payload(n) for key in family_repeats[family]}
            for family in families_cfg
        }
        for outer_fold in range(int(cfg["outer_folds"])):
            train = outer_assignment != outer_fold
            evaluate = ~train
            train_episodes = episodes[train]
            eval_episodes = episodes[evaluate]
            if set(np.unique(train_episodes)) & set(np.unique(eval_episodes)):
                raise RuntimeError("outer episode isolation failed")
            log(f"{'smoke ' if smoke else ''}repeat={repeat} outer_fold={outer_fold} train_rows={train.sum()} eval_rows={evaluate.sum()}")
            whitening = common.fit_whitening(targets[train])
            eval_white = common.whitened_losses(targets[evaluate], exits[evaluate], whitening)
            repeat_white[evaluate] = eval_white
            inner_assignment = common.episode_assignment(
                train_episodes,
                int(cfg["inner_folds"]),
                int(cfg["inner_seed"]) + repeat * 10007 + outer_fold * 1009,
            )
            teacher_maps, teacher_map_audits = strict_teacher_maps(
                features[train], gains[train], train_episodes, inner_assignment,
                cfg, device, epoch_override=1 if smoke else None,
            )
            # Ordinary outer-train cross-fitting is safe for final outer model
            # labels because no outer evaluation episode enters any teacher.
            teacher_oof, teacher_oof_provenance = students.crossfit_teacher(
                features[train], gains[train], train_episodes,
                folds=int(cfg["inner_folds"]),
                seed=int(cfg["inner_seed"]) + repeat * 7919 + outer_fold,
                teacher_seeds=cfg["teacher_seeds"], device=device,
                hidden_dims=cfg["teacher_hidden_dims"],
                epochs=1 if smoke else int(cfg["teacher_epochs"]),
                batch_size=int(cfg["batch_size"]), learning_rate=float(cfg["learning_rate"]),
            )
            teacher_final = students.fit_teacher(
                features[train], gains[train],
                seed=int(cfg["teacher_seeds"][0]) + repeat * 1009 + outer_fold,
                device=device, hidden_dims=cfg["teacher_hidden_dims"],
                epochs=1 if smoke else int(cfg["teacher_epochs"]),
                batch_size=int(cfg["batch_size"]), learning_rate=float(cfg["learning_rate"]),
            )
            teacher_eval = students.predict_student(teacher_final, features[evaluate], device)
            repeat_teacher_scores[evaluate] = teacher_eval
            fold_record = {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "train_episode_count": int(len(np.unique(train_episodes))),
                "eval_episode_count": int(len(np.unique(eval_episodes))),
                "teacher_map_audits": teacher_map_audits,
                "teacher_oof_provenance": teacher_oof_provenance,
                "families": {},
            }
            for family_index, (family, candidates) in enumerate(families_cfg.items()):
                chosen, train_scores_oof, tuning = tune_family_inner(
                    features[train], gains[train], losses[train], train_episodes,
                    inner_assignment, teacher_maps,
                    family=family, candidates=candidates, cfg=cfg, device=device,
                    seed_offset=repeat * 100003 + outer_fold * 10007 + family_index * 101,
                    epoch_override=1 if smoke else None,
                )
                model, final_sparse_audit = fit_fold_student(
                    features[train], gains[train], teacher_oof, train_episodes,
                    family=family, candidate=chosen,
                    seed=int(cfg["student_seeds"][0]) + repeat * 10007 + outer_fold * 101 + family_index,
                    cfg=cfg, device=device, epoch_override=1 if smoke else None,
                )
                eval_scores = students.predict_student(model, features[evaluate], device)
                repeat_scores[family][evaluate] = eval_scores
                evaluated = protocol.evaluate_outer_fold(
                    train_scores_oof=train_scores_oof,
                    eval_scores=eval_scores,
                    train_losses=losses[train],
                    eval_losses=losses[evaluate],
                    eval_white_losses=eval_white,
                    eval_episodes=eval_episodes,
                    budgets=cfg["target_mean_calls"],
                    seed=int(cfg["outer_seed"]) + repeat * 100003 + outer_fold * 1009 + family_index * 17,
                )
                for key, local in evaluated.items():
                    destination = repeat_points[family][key]
                    for name in destination:
                        if name in ("fold_audits", "fold_benefits"):
                            continue
                        destination[name][evaluate] = local[name]
                    destination["fold_audits"].append(local["exact_call_audit"])
                    destination["fold_benefits"].append(float((local["matched_loss"] - local["adaptive_loss"]).mean()))
                fold_record["families"][family] = {
                    "tuning": tuning,
                    "selected_config": chosen,
                    "selected_config_key": config_key(chosen),
                    "outer_model_feature_indices": model.feature_indices,
                    "outer_sparse_selection_audit": final_sparse_audit,
                }
            selection_records.append(fold_record)
        if not np.isfinite(repeat_white).all() or not np.isfinite(repeat_teacher_scores).all():
            raise RuntimeError("repeat fold stitching failed")
        white_loss_repeats.append(repeat_white)
        teacher_scores_repeats.append(repeat_teacher_scores)
        for family in families_cfg:
            if not np.isfinite(repeat_scores[family]).all():
                raise RuntimeError(f"repeat score stitching failed for {family}")
            family_scores[family].append(repeat_scores[family])
            for key, payload in repeat_points[family].items():
                payload["exact_call_audit"] = {
                    "exact_total_match": bool(all(item["exact_total_match"] for item in payload["fold_audits"])),
                    "folds": payload["fold_audits"],
                }
                validate_repeat_payload(payload)
                family_repeats[family][key].append(payload)
    solver_hash_after = common.sha256_file(common.SOLVER_CHECKPOINT)
    if solver_hash_before != solver_hash_after:
        raise RuntimeError("solver checkpoint changed during student training")
    if smoke:
        return {
            "status": "smoke_passed",
            "rows": n,
            "episodes": int(len(np.unique(episodes))),
            "families": list(families_cfg),
            "finite": True,
            "solver_checkpoint_unchanged": True,
            "selection_records": selection_records,
        }

    average_white = np.stack(white_loss_repeats).mean(0)
    families = {}
    for family, points in family_repeats.items():
        selected_configs = [record["families"][family]["selected_config"] for record in selection_records]
        max_features = max(
            int(candidate.get("sparse_k", len(common.feature_indices(family)) if family != "stable_sparse" else 1))
            for candidate in selected_configs
        )
        max_hidden = max((tuple(candidate.get("hidden_dims", ())) for candidate in selected_configs), key=lambda dims: (sum(dims), dims))
        operating_points = {}
        for key, repeat_payloads in points.items():
            point = protocol.aggregate_repeated_point(
                repeat_payloads,
                episode_ids=episodes,
                fixed_losses=losses,
                fixed_white_losses=average_white,
                family=family,
                feature_count=max_features,
                hidden_dims=max_hidden,
                bootstrap_samples=int(cfg["bootstrap_samples_discovery"]),
                bootstrap_seed=int(cfg["bootstrap_seed"]) + sum(map(ord, family + key)),
            )
            point["outer_fold_matched_benefits"] = [float(value) for payload in repeat_payloads for value in payload["fold_benefits"]]
            point["positive_outer_fold_fraction"] = float(np.mean(np.asarray(point["outer_fold_matched_benefits"]) > 0))
            operating_points[key] = point
        averaged_scores = np.stack(family_scores[family]).mean(0)
        families[family] = {
            "selected_configs_by_outer_fold": selected_configs,
            "conservative_oof_feature_count": max_features,
            "conservative_oof_hidden_dims": max_hidden,
            "diagnostics": protocol.ranking_diagnostics(averaged_scores, gains),
            "operating_points": operating_points,
        }
    teacher_average = np.stack(teacher_scores_repeats).mean(0)
    teacher_diagnostics = protocol.ranking_diagnostics(teacher_average, gains)
    fixed_raw = losses.mean(0)
    frontier = protocol.mark_global_frontier(families, fixed_raw, cfg)
    for payload in families.values():
        for point in payload["operating_points"].values():
            point["passes_discovery_point"] = bool(
                protocol.discovery_point_passes(point)
                and point["positive_outer_fold_fraction"] >= 2.0 / 3.0
            )
    advancing = []
    for family, payload in families.items():
        for key, point in payload["operating_points"].items():
            if point["passes_discovery_point"]:
                advancing.append({"family": family, "operating_point": key, "point": point})
    result = {
        "schema_version": 1,
        "protocol": "two-repeat three-fold grouped nested CV; three grouped inner folds",
        "rows": n,
        "episodes": int(len(np.unique(episodes))),
        "solver_checkpoint_sha256_before": solver_hash_before,
        "solver_checkpoint_sha256_after": solver_hash_after,
        "solver_checkpoint_unchanged": solver_hash_before == solver_hash_after,
        "causal_feature_audit": common.causal_feature_audit(),
        "candidate_cost_audit": students.audit_candidate_costs(cfg),
        "fixed_exit_raw_mse": fixed_raw,
        "teacher_diagnostics": teacher_diagnostics,
        "families": families,
        "global_frontier": frontier,
        "selection_records": selection_records,
        "advancing_points": advancing,
        "discovery_pass": bool(advancing),
        "calibration_touched": False,
    }
    return result


def choose_final_configuration(discovery: Mapping[str, Any], family: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [dict(value) for value in cfg["families"][family]]
    counts = {config_key(candidate): 0 for candidate in candidates}
    utilities = {key: [] for key in counts}
    by_key = {config_key(candidate): candidate for candidate in candidates}
    for record in discovery["selection_records"]:
        family_record = record["families"][family]
        counts[family_record["selected_config_key"]] += 1
        for row in family_record["tuning"]["candidates"]:
            utilities[row["config_key"]].append(float(row["inner_routing"]["utility"]))
    winner = max(counts, key=lambda key: (counts[key], float(np.mean(utilities[key])), key))
    return {
        "config": by_key[winner],
        "config_key": winner,
        "outer_selection_count": counts[winner],
        "outer_selection_counts": counts,
        "mean_inner_utilities": {key: float(np.mean(values)) for key, values in utilities.items()},
    }


def fit_discovery_student(
    discovery: Mapping[str, Any],
    family: str,
    operating_point: str,
    device: torch.device,
) -> tuple[students.FrozenStudent, dict[str, Any]]:
    cfg = common.load_config()
    prepared = common.load_prepared()
    features = prepared["features"]
    losses = prepared["losses"].astype(np.float64)
    gains = protocol.gains_from_losses(losses)
    episodes = prepared["episode_id"]
    final_choice = choose_final_configuration(discovery, family, cfg)
    candidate = final_choice["config"]
    assignment = common.episode_assignment(episodes, int(cfg["inner_folds"]), int(cfg["inner_seed"]) + 700001)
    teacher_maps, strict_audits = strict_teacher_maps(features, gains, episodes, assignment, cfg, device)
    # Strict OOF student scores for threshold freezing.
    oof = np.full_like(gains, np.nan, dtype=np.float64)
    oof_sparse = []
    for held in range(int(cfg["inner_folds"])):
        train = assignment != held
        model, sparse = fit_fold_student(
            features[train], gains[train], teacher_maps[held][train], episodes[train],
            family=family, candidate=candidate,
            seed=int(cfg["student_seeds"][held]) + 800003,
            cfg=cfg, device=device,
        )
        oof[~train] = students.predict_student(model, features[~train], device)
        if sparse is not None:
            oof_sparse.append({"held_fold": held, **sparse})
    if not np.isfinite(oof).all():
        raise RuntimeError("final strict student OOF coverage failed")
    teacher_oof, teacher_provenance = students.crossfit_teacher(
        features, gains, episodes,
        folds=int(cfg["inner_folds"]), seed=int(cfg["inner_seed"]) + 900001,
        teacher_seeds=cfg["teacher_seeds"], device=device,
        hidden_dims=cfg["teacher_hidden_dims"], epochs=int(cfg["teacher_epochs"]),
        batch_size=int(cfg["batch_size"]), learning_rate=float(cfg["learning_rate"]),
    )
    final_model, final_sparse = fit_fold_student(
        features, gains, teacher_oof, episodes,
        family=family, candidate=candidate,
        seed=int(cfg["student_seeds"][0]) + 900007,
        cfg=cfg, device=device,
    )
    budget = float(operating_point.removeprefix("b"))
    price_record = protocol.calibrate_price(oof, budget)
    checkpoint_path = ROOT / "checkpoints/final_student.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "student": final_model.payload(),
        "family": family,
        "operating_point": operating_point,
        "compute_price": float(price_record["compute_price"]),
        "price_record": price_record,
        "configuration_selection": final_choice,
        "strict_oof_scores": oof,
        "teacher_oof_scores": teacher_oof,
        "strict_teacher_audits": strict_audits,
        "teacher_provenance": teacher_provenance,
        "oof_sparse_audits": oof_sparse,
        "final_sparse_audit": final_sparse,
    }, checkpoint_path)
    meta = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": common.sha256_file(checkpoint_path),
        "family": family,
        "operating_point": operating_point,
        "compute_price": float(price_record["compute_price"]),
        "configuration_selection": final_choice,
        "feature_indices": final_model.feature_indices,
        "cost": students.gate_cost(family, len(final_model.feature_indices), final_model.hidden_dims),
        "strict_oof_diagnostics": protocol.ranking_diagnostics(oof, gains),
        "final_sparse_audit": final_sparse,
    }
    return final_model, meta


def freeze_tournament(discovery: Mapping[str, Any], final_meta: Mapping[str, Any]) -> dict[str, Any]:
    cfg = common.load_config()
    import judge_runner
    whitening = judge_runner.create_whitening_artifact()
    source_names = ("common.py", "students.py", "protocol.py", "run_experiment.py", "judge_runner.py")
    tournament = {
        "schema_version": 1,
        "status": "frozen_before_v3_calibration_target_access",
        "scientific_scope": "critic compression only; selected stagewise solver immutable",
        "family": final_meta["family"],
        "operating_point": final_meta["operating_point"],
        "compute_price": final_meta["compute_price"],
        "student_checkpoint": final_meta["checkpoint"],
        "student_checkpoint_sha256": final_meta["checkpoint_sha256"],
        "configuration_selection": final_meta["configuration_selection"],
        "feature_indices": final_meta["feature_indices"],
        "gate_cost": final_meta["cost"],
        "calibration_pass_rule": cfg["calibration_pass_rule"],
        "bootstrap_samples": int(cfg["bootstrap_samples_calibration"]),
        "bootstrap_seed": int(cfg["bootstrap_seed"]),
        "config_sha256": common.sha256_file(ROOT / "config.json"),
        "plan_sha256": common.sha256_file(ROOT / "PLAN.md"),
        "source_sha256": {name: common.sha256_file(ROOT / name) for name in source_names},
        "solver_checkpoint_sha256": common.sha256_file(common.SOLVER_CHECKPOINT),
        "train_cache_sha256": common.sha256_file(common.TRAIN_CACHE),
        "prepared_cache_sha256": common.sha256_file(common.PREPARED_CACHE),
        "discovery_metrics_sha256": common.sha256_file(ROOT / "metrics/discovery_cv.json"),
        "discovery_fixed_exit_raw_mse": discovery["fixed_exit_raw_mse"],
        "whitening_path": whitening["path"],
        "whitening_sha256": whitening["sha256"],
        "baseline_seeds": {
            "matched": 272101,
            "histogram": 272102,
            "score_permutation": 272103,
            "permuted_matched": 272104
        },
        "v3_test_targets": "forbidden",
        "calibration_reselection": "forbidden",
    }
    path = ROOT / "audit/frozen_tournament.json"
    common.write_json(path, tournament)
    digest = common.sha256_file(path)
    common.write_json(ROOT / "audit/frozen_tournament.sha256.json", {"sha256": digest})
    return {**tournament, "tournament_sha256": digest}


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "git_head": _git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git_output(["git", "status", "--short"]),
    }


def _git_output(command: Sequence[str]) -> str:
    import subprocess
    return subprocess.run(command, cwd=common.REPO, check=True, text=True, capture_output=True).stdout.strip()


def run_discovery(device: torch.device) -> dict[str, Any]:
    if (ROOT / "audit/calibration_access_receipt.json").exists():
        raise RuntimeError("discovery cannot rerun after calibration access")
    common.write_json(ROOT / "environment.json", environment_record(device))
    result = run_nested_cv(device, smoke=False)
    common.write_json(ROOT / "metrics/discovery_cv.json", result)
    advancing = result["advancing_points"]
    if advancing:
        winner = max(
            advancing,
            key=lambda row: (
                row["point"]["vs_matched_randomized"]["ci_low"],
                -row["point"]["adaptive_total_flops_per_transition"],
                -row["point"]["raw_mse"],
                row["family"], row["operating_point"],
            ),
        )
    else:
        # Retain and benchmark the best negative diagnostic without authorizing calibration.
        candidates = [
            {"family": family, "operating_point": key, "point": point}
            for family, payload in result["families"].items()
            for key, point in payload["operating_points"].items()
        ]
        winner = max(candidates, key=lambda row: (row["point"]["vs_matched_randomized"]["mean_benefit"], -row["point"]["raw_mse"]))
    _, final_meta = fit_discovery_student(result, winner["family"], winner["operating_point"], device)
    common.write_json(ROOT / "metrics/final_student.json", final_meta)
    if advancing:
        frozen = freeze_tournament(result, final_meta)
        decision = {
            "decision": "critic_compression_discovery_passed",
            "phase": "discovery_cv",
            "winner": winner["family"],
            "operating_point": winner["operating_point"],
            "calibration_consumed": False,
            "v3_test_targets_consumed": False,
            "v4_created": False,
            "tournament_sha256": frozen["tournament_sha256"],
            "next_action": "sealed one-shot V3 calibration judge is authorized",
        }
    else:
        decision = {
            "decision": "critic_compression_internal_failed",
            "phase": "discovery_cv",
            "best_negative_diagnostic": winner["family"],
            "operating_point": winner["operating_point"],
            "calibration_consumed": False,
            "calibration_cache_created": False,
            "v3_test_targets_consumed": False,
            "v4_created": False,
            "reason": "no nested-CV student operating point passed every raw, whitened, exact-call, FLOP-frontier, and consistency gate",
        }
    common.write_json(ROOT / "decision.json", decision)
    return {"decision": decision, "winner": winner, "final_student": final_meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "smoke", "discovery", "judge", "report"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = common.choose_device(args.device)
    if args.phase == "prepare":
        result = common.prepare_solver_cache(device)
    elif args.phase == "smoke":
        result = run_nested_cv(device, smoke=True)
        common.write_json(ROOT / "metrics/smoke.json", result)
    elif args.phase == "discovery":
        result = run_discovery(device)
    elif args.phase == "judge":
        import judge_runner
        result = judge_runner.judge(device)
    else:
        raise RuntimeError("report generation is enabled after discovery metrics exist")
    log(f"complete phase={args.phase} result={common.jsonable(result.get('decision', result.get('status', 'ok')) if isinstance(result, dict) else 'ok')}")


if __name__ == "__main__":
    main()
