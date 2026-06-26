#!/usr/bin/env python3
"""Unit tests for Fetch interaction-error taxonomy."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    TaxonomyThresholds,
    assign_event_taxonomy,
    is_gripper_object_pair,
    is_object_support_pair,
    is_task_object_pair,
)


class FetchTaxonomyTest(unittest.TestCase):
    def test_pair_classification(self) -> None:
        self.assertTrue(is_gripper_object_pair("object0::robot0:r_gripper_finger_link"))
        self.assertTrue(is_object_support_pair("object0::table0"))
        self.assertTrue(is_object_support_pair("geom_22::object0"))
        self.assertFalse(is_task_object_pair("floor0::robot0:base_link"))

    def test_primary_regimes_are_mece_with_precedence(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "episode_id": 0,
                    "step_idx": 0,
                    "task_contact_pairs": "",
                    "gripper_object_contact": False,
                    "object_support_contact": False,
                    "max_contact_impulse_proxy": 0.0,
                    "contact_impulse_delta": 0.0,
                    "object_velocity_delta": 0.0,
                    "object_speed": 0.0,
                    "state_norm": 1.0,
                    "reset_artifact": True,
                },
                {
                    "episode_id": 0,
                    "step_idx": 1,
                    "task_contact_pairs": "object0::robot0:r_gripper_finger_link",
                    "gripper_object_contact": True,
                    "object_support_contact": False,
                    "max_contact_impulse_proxy": 10.0,
                    "contact_impulse_delta": 10.0,
                    "object_velocity_delta": 1.0,
                    "object_speed": 0.2,
                    "state_norm": 1.0,
                    "reset_artifact": False,
                },
                {
                    "episode_id": 0,
                    "step_idx": 2,
                    "task_contact_pairs": "object0::robot0:r_gripper_finger_link",
                    "gripper_object_contact": True,
                    "object_support_contact": False,
                    "max_contact_impulse_proxy": 1.0,
                    "contact_impulse_delta": 0.0,
                    "object_velocity_delta": 0.0,
                    "object_speed": 0.2,
                    "state_norm": 1.0,
                    "reset_artifact": False,
                },
                {
                    "episode_id": 0,
                    "step_idx": 3,
                    "task_contact_pairs": "object0::table0",
                    "gripper_object_contact": False,
                    "object_support_contact": True,
                    "max_contact_impulse_proxy": 1.0,
                    "contact_impulse_delta": 0.0,
                    "object_velocity_delta": 0.0,
                    "object_speed": 0.2,
                    "state_norm": 1.0,
                    "reset_artifact": False,
                },
                {
                    "episode_id": 0,
                    "step_idx": 4,
                    "task_contact_pairs": "object0::robot0:l_gripper_finger_link",
                    "gripper_object_contact": True,
                    "object_support_contact": False,
                    "max_contact_impulse_proxy": 1.0,
                    "contact_impulse_delta": 0.0,
                    "object_velocity_delta": 0.0,
                    "object_speed": 0.0,
                    "state_norm": 1.0,
                    "reset_artifact": False,
                },
                {
                    "episode_id": 0,
                    "step_idx": 5,
                    "task_contact_pairs": "",
                    "gripper_object_contact": False,
                    "object_support_contact": False,
                    "max_contact_impulse_proxy": 0.0,
                    "contact_impulse_delta": 0.0,
                    "object_velocity_delta": 0.0,
                    "object_speed": 0.0,
                    "state_norm": 1.0,
                    "reset_artifact": False,
                },
            ]
        )
        labeled, _ = assign_event_taxonomy(
            df,
            history_size=1,
            response_window=1,
            thresholds=TaxonomyThresholds(
                impulse=5.0,
                impulse_delta=5.0,
                velocity_delta=0.5,
                object_speed=0.1,
            ),
        )
        self.assertEqual(
            labeled["primary_regime"].tolist(),
            [
                "boundary_or_invalid",
                "impact_onset",
                "post_impact_response",
                "sustained_contact_dynamics",
                "gripper_object_contact",
                "free_motion",
            ],
        )
        self.assertFalse(labeled["primary_regime"].isna().any())
        self.assertEqual(len(labeled), labeled["primary_regime"].notna().sum())


if __name__ == "__main__":
    unittest.main()
