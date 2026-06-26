"""Shared utilities for Fetch interaction-error diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_REGIMES = [
    "impact_onset",
    "post_impact_response",
    "sustained_contact_dynamics",
    "gripper_object_contact",
    "free_motion",
    "boundary_or_invalid",
]

HARD_REGIMES = {
    "impact_onset",
    "post_impact_response",
    "sustained_contact_dynamics",
}

OBJECT_GEOM = "object0"
GRIPPER_GEOMS = {
    "robot0:gripper_link",
    "robot0:r_gripper_finger_link",
    "robot0:l_gripper_finger_link",
}
STATIC_BACKGROUND_PREFIXES = ("floor", "robot0:base")


@dataclass(frozen=True)
class TaxonomyThresholds:
    impulse: float
    impulse_delta: float
    velocity_delta: float
    object_speed: float

    def as_dict(self) -> dict[str, float]:
        return {
            "high_contact_impulse_threshold": float(self.impulse),
            "contact_impulse_delta_threshold": float(self.impulse_delta),
            "object_velocity_delta_threshold": float(self.velocity_delta),
            "object_speed_threshold": float(self.object_speed),
        }


def pair_key(name_a: str, name_b: str) -> str:
    return "::".join(sorted((str(name_a), str(name_b))))


def parse_pairs(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {part for part in text.split(";") if part}


def pairs_to_text(pairs: Iterable[str]) -> str:
    return ";".join(sorted(set(pairs)))


def is_static_background_pair(pair: str) -> bool:
    names = pair.split("::")
    return any(any(name.startswith(prefix) for prefix in STATIC_BACKGROUND_PREFIXES) for name in names)


def is_gripper_object_pair(pair: str) -> bool:
    names = set(pair.split("::"))
    return OBJECT_GEOM in names and bool(names & GRIPPER_GEOMS)


def is_object_support_pair(pair: str) -> bool:
    names = set(pair.split("::"))
    if OBJECT_GEOM not in names:
        return False
    other_names = names - {OBJECT_GEOM}
    if other_names & GRIPPER_GEOMS:
        return False
    if any(name.startswith("floor") for name in other_names):
        return False
    return True


def is_task_object_pair(pair: str) -> bool:
    if is_static_background_pair(pair):
        return False
    return is_gripper_object_pair(pair) or is_object_support_pair(pair)


def finite_percentile(values: pd.Series | np.ndarray, percentile: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(default)
    return float(np.percentile(arr, percentile))


def positive_percentile(values: pd.Series | np.ndarray, percentile: float, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return float(default)
    return float(np.percentile(arr, percentile))


def add_pair_transition_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["task_contact_pairs"] = out.get("task_contact_pairs", "").fillna("").astype(str)
    out["gripper_object_contact"] = out.get("gripper_object_contact", False).astype(bool)
    out["object_support_contact"] = out.get("object_support_contact", False).astype(bool)

    prev_pairs = pd.Series("", index=out.index, dtype=object)
    new_pairs = pd.Series("", index=out.index, dtype=object)
    new_gripper = pd.Series(False, index=out.index, dtype=bool)
    new_support = pd.Series(False, index=out.index, dtype=bool)
    prev_gripper = pd.Series(False, index=out.index, dtype=bool)
    prev_support = pd.Series(False, index=out.index, dtype=bool)

    for _, group in out.groupby("episode_id", sort=False):
        last_pairs: set[str] = set()
        last_gripper = False
        last_support = False
        for idx in group.index:
            current_pairs = parse_pairs(out.at[idx, "task_contact_pairs"])
            current_gripper = bool(out.at[idx, "gripper_object_contact"])
            current_support = bool(out.at[idx, "object_support_contact"])
            added = current_pairs - last_pairs

            prev_pairs.at[idx] = pairs_to_text(last_pairs)
            new_pairs.at[idx] = pairs_to_text(added)
            prev_gripper.at[idx] = last_gripper
            prev_support.at[idx] = last_support
            new_gripper.at[idx] = current_gripper and not last_gripper
            new_support.at[idx] = current_support and not last_support

            last_pairs = current_pairs
            last_gripper = current_gripper
            last_support = current_support

    out["prev_task_contact_pairs"] = prev_pairs
    out["new_task_contact_pairs"] = new_pairs
    out["new_task_contact_pair"] = out["new_task_contact_pairs"].astype(bool)
    out["prev_gripper_object_contact"] = prev_gripper
    out["prev_object_support_contact"] = prev_support
    out["gripper_object_contact_onset"] = new_gripper
    out["object_support_contact_onset"] = new_support
    return out


def add_motion_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "object_speed" not in out:
        velocity_cols = [f"object_vel_{axis}" for axis in ("x", "y", "z")]
        if all(col in out for col in velocity_cols):
            out["object_speed"] = np.linalg.norm(out[velocity_cols].to_numpy(dtype=np.float64), axis=1)
        else:
            out["object_speed"] = 0.0

    if "object_velocity_delta" not in out:
        deltas = np.zeros(len(out), dtype=np.float64)
        velocity_cols = [f"object_vel_{axis}" for axis in ("x", "y", "z")]
        if all(col in out for col in velocity_cols):
            for _, group in out.groupby("episode_id", sort=False):
                values = group[velocity_cols].to_numpy(dtype=np.float64)
                diff = np.vstack([np.zeros((1, values.shape[1])), np.diff(values, axis=0)])
                deltas[group.index.to_numpy()] = np.linalg.norm(diff, axis=1)
        else:
            for _, group in out.groupby("episode_id", sort=False):
                values = group["object_speed"].to_numpy(dtype=np.float64)
                diff = np.r_[0.0, np.diff(values)]
                deltas[group.index.to_numpy()] = np.abs(diff)
        out["object_velocity_delta"] = deltas

    if "contact_impulse_delta" not in out:
        deltas = np.zeros(len(out), dtype=np.float64)
        if "max_contact_impulse_proxy" in out:
            for _, group in out.groupby("episode_id", sort=False):
                values = group["max_contact_impulse_proxy"].to_numpy(dtype=np.float64)
                diff = np.r_[0.0, np.diff(values)]
                deltas[group.index.to_numpy()] = np.abs(diff)
        out["contact_impulse_delta"] = deltas
    return out


def infer_taxonomy_thresholds(
    df: pd.DataFrame,
    *,
    impulse_percentile: float = 90.0,
    velocity_percentile: float = 90.0,
    speed_percentile: float = 60.0,
) -> TaxonomyThresholds:
    contact_mask = (
        df.get("gripper_object_contact", False).astype(bool)
        | df.get("object_support_contact", False).astype(bool)
    )
    impulse_values = df.loc[contact_mask, "max_contact_impulse_proxy"] if "max_contact_impulse_proxy" in df else []
    impulse_delta_values = df.loc[contact_mask, "contact_impulse_delta"] if "contact_impulse_delta" in df else []
    velocity_delta_values = df["object_velocity_delta"] if "object_velocity_delta" in df else []
    speed_values = df.loc[contact_mask, "object_speed"] if "object_speed" in df else []
    return TaxonomyThresholds(
        impulse=positive_percentile(impulse_values, impulse_percentile),
        impulse_delta=positive_percentile(impulse_delta_values, impulse_percentile),
        velocity_delta=positive_percentile(velocity_delta_values, velocity_percentile),
        object_speed=positive_percentile(speed_values, speed_percentile),
    )


def add_post_impact_window(df: pd.DataFrame, *, response_window: int) -> pd.Series:
    result = pd.Series(False, index=df.index)
    for _, group in df.groupby("episode_id", sort=False):
        impact_steps = set(group.loc[group["impact_onset"], "step_idx"].astype(int))
        if not impact_steps:
            continue
        for idx, step in group["step_idx"].astype(int).items():
            result.at[idx] = any(1 <= step - impact <= response_window for impact in impact_steps)
    return result


def assign_event_taxonomy(
    df: pd.DataFrame,
    *,
    history_size: int = 3,
    response_window: int = 3,
    thresholds: TaxonomyThresholds | None = None,
) -> tuple[pd.DataFrame, TaxonomyThresholds]:
    out = add_motion_columns(add_pair_transition_columns(df))
    if thresholds is None:
        thresholds = infer_taxonomy_thresholds(out)

    invalid = pd.Series(False, index=out.index)
    if "step_idx" in out:
        invalid |= out["step_idx"].astype(int) < int(history_size)
    invalid |= ~np.isfinite(out.get("state_norm", pd.Series(0.0, index=out.index)).astype(float))
    invalid |= out.get("reset_artifact", False).astype(bool)

    out["high_contact_impulse"] = (
        (out.get("max_contact_impulse_proxy", 0.0).astype(float) >= thresholds.impulse)
        & (out.get("max_contact_impulse_proxy", 0.0).astype(float) > 0.0)
    )
    out["high_contact_impulse_delta"] = (
        (out.get("contact_impulse_delta", 0.0).astype(float) >= thresholds.impulse_delta)
        & (out.get("contact_impulse_delta", 0.0).astype(float) > 0.0)
    )
    out["object_velocity_discontinuity"] = (
        (out["object_velocity_delta"].astype(float) >= thresholds.velocity_delta)
        & (out["object_velocity_delta"].astype(float) > 0.0)
    )
    out["sliding_contact"] = (
        out["object_support_contact"].astype(bool)
        & (out["object_speed"].astype(float) >= thresholds.object_speed)
        & (out["object_speed"].astype(float) > 0.0)
    )
    out["object_table_impact_onset"] = (
        out["object_support_contact"].astype(bool)
        & (out["object_support_contact_onset"].astype(bool) | out["high_contact_impulse_delta"])
        & (out["object_velocity_discontinuity"] | out["high_contact_impulse"])
    )
    out["impact_onset"] = (
        out["object_table_impact_onset"]
        | (
            out["new_task_contact_pair"].astype(bool)
            & (out["high_contact_impulse"] | out["object_velocity_discontinuity"])
        )
        | (
            out["gripper_object_contact_onset"].astype(bool)
            & out["object_velocity_discontinuity"]
        )
    )
    out["post_contact_response_window"] = add_post_impact_window(out, response_window=response_window)

    sustained = (
        (out["gripper_object_contact"].astype(bool) | out["object_support_contact"].astype(bool))
        & (out["sliding_contact"] | (out["object_speed"].astype(float) >= thresholds.object_speed))
    )

    regimes = np.full(len(out), "free_motion", dtype=object)
    regimes[invalid.to_numpy()] = "boundary_or_invalid"
    mask = ~invalid.to_numpy()
    impact = (out["impact_onset"].to_numpy() & mask)
    post = (out["post_contact_response_window"].to_numpy() & mask & ~impact)
    sustained_mask = sustained.to_numpy() & mask & ~impact & ~post
    gripper = (
        out["gripper_object_contact"].to_numpy()
        & mask
        & ~impact
        & ~post
        & ~sustained_mask
    )

    regimes[impact] = "impact_onset"
    regimes[post] = "post_impact_response"
    regimes[sustained_mask] = "sustained_contact_dynamics"
    regimes[gripper] = "gripper_object_contact"
    out["primary_regime"] = regimes
    out["is_valid_model_step"] = out["primary_regime"] != "boundary_or_invalid"

    unknown = set(out["primary_regime"]) - set(PRIMARY_REGIMES)
    if unknown:
        raise ValueError(f"Unknown primary regimes assigned: {sorted(unknown)}")
    if out["primary_regime"].isna().any():
        raise ValueError("Primary regime assignment produced null values")
    return out, thresholds
