import hashlib, json, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


class AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((ROOT / "protocol.json").read_text())
        cls.a = json.loads((ROOT / "alignment.json").read_text())
        cls.m = json.loads((ROOT / "metrics.json").read_text())

    def test_protocol_hash_and_bounds(self):
        raw = (ROOT / "protocol.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), (ROOT / "protocol.sha256").read_text().split()[0])
        self.assertEqual(len(self.p["sample"]["episode_ids"]), 8)
        self.assertEqual(len(set(self.p["sample"]["episode_ids"])), 8)
        self.assertEqual(self.p["indices"]["transitions_per_episode"], 8)
        self.assertEqual(self.p["indices"]["frames_per_episode"], 3)

    def test_hash_sample_and_isolation(self):
        split = json.loads((REPO / "runs/lewm_adaptive_compute_v3/cache/split_manifest.json").read_text())
        train = split["selection"]["split_episode_ordinals"]["train"]
        test = set(split["selection"]["split_episode_ordinals"]["test"])
        salt = self.p["sample"]["salt"]
        expected = sorted(train, key=lambda x: (hashlib.sha256(f"{salt}:{x}".encode()).hexdigest(), x))[:8]
        self.assertEqual(expected, self.p["sample"]["episode_ids"])
        self.assertFalse(set(expected) & test)

    def test_alignment_and_counts(self):
        self.assertEqual(self.a["status"], "sealed_before_environment_comparison")
        self.assertTrue(all(x["prev_vs_prior_qpos"]["exact"] and x["prev_vs_prior_qvel"]["exact"] for x in self.a["per_episode"]))
        self.assertEqual(self.m["transition_count"], 64)
        self.assertEqual(self.m["frame_count"], 24)
        self.assertEqual(self.m["new_policy_trajectories"], 0)
        self.assertEqual(self.m["model_or_gate_evaluation_rows"], 0)

    def test_summary_recomputes_from_records(self):
        keys = ["state_qpos","state_qvel","observation","control_mapping","action_transition_qpos","action_transition_qvel","direct_control_transition_qpos","direct_control_transition_qvel"]
        for key in keys:
            xs = [r[key] for r in self.m["records"] if key in r]
            n = sum(x["count"] for x in xs)
            self.assertEqual(n, self.m["summary"][key]["count"])
            self.assertAlmostEqual(max(x["max_abs"] for x in xs), self.m["summary"][key]["max_abs"], places=15)
            self.assertAlmostEqual(sum(x["mean_abs"]*x["count"] for x in xs)/n, self.m["summary"][key]["mean_abs"], places=15)

    def test_fail_closed_claim_flags(self):
        self.assertEqual(self.m["v3_test_episode_intersection"], [])
        self.assertFalse(self.m["v3_test_targets_opened"])
        self.assertFalse(self.m["combined_v3_cache_opened_with_numpy"])
        self.assertEqual(self.m["v5_confirmation_episodes"], 0)

    def test_decision_and_historical_candidate(self):
        d = json.loads((ROOT / "decision.json").read_text())
        hm = json.loads((ROOT / "historical_candidate_metrics.json").read_text())
        self.assertEqual(d["layer_localized_decision"], "reset_only_mismatch")
        self.assertEqual(d["generator_provenance_decision"], "generator_provenance_unresolved")
        self.assertEqual(d["historical_candidate_count"], 1)
        self.assertEqual(hm["transition_count"], 64)
        self.assertEqual(hm["frame_count"], 24)
        self.assertEqual(hm["passes"], self.m["passes"])
        for key in self.m["summary"]:
            self.assertEqual(hm["summary"][key], self.m["summary"][key])


if __name__ == "__main__": unittest.main()
