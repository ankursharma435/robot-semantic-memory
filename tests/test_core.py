"""
Unit tests for the memory/gate/metrics core.

Deliberately hermetic: NO model load, NO network, NO camera. Every embedding
here is a hand-built synthetic vector, so the suite runs in well under a
second and gives the same answer on any machine — including the Jetson, once
it arrives. Stdlib unittest only, so it adds no dependency.

    python -m unittest discover -s tests -v
    python tests/test_core.py            # same thing

The integration counterpart is verify_offline.py, which DOES load the real
encoder and (with --live) hits the NIM/Riva services.
"""

import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from encoder import truncate, cosine, TIER_DIMS, FULL_DIM
from memory_store import MemoryStore
import gate
from gate import decide, Decision, load_calibration
from metrics import MetricsLogger


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_vector(head, tail_seed, head_dim=64):
    """Build a 1024-dim unit vector with a controlled first `head_dim` block.

    This is the whole trick that lets us test the truncation-bias bug without
    the real encoder: the first 64 dims (the SHORT tier) are set explicitly,
    and the remaining dims are filled deterministically from tail_seed. Two
    vectors can therefore be made to agree strongly at 64 dims while
    disagreeing at 768 — exactly the pattern that made max()-across-tiers
    pick the short tier.
    """
    rng = np.random.default_rng(tail_seed)
    v = np.zeros(FULL_DIM, dtype=np.float32)
    v[:head_dim] = head
    v[head_dim:] = rng.normal(0, 1, FULL_DIM - head_dim)
    return unit(v)


class TestTruncate(unittest.TestCase):
    def test_renormalizes_to_unit_length(self):
        v = unit(np.random.default_rng(0).normal(0, 1, FULL_DIM))
        for dim in (64, 256, 768):
            self.assertAlmostEqual(float(np.linalg.norm(truncate(v, dim))), 1.0, places=5)

    def test_is_a_prefix_up_to_scale(self):
        v = unit(np.random.default_rng(1).normal(0, 1, FULL_DIM))
        t = truncate(v, 64)
        # direction of the prefix must be preserved
        ratio = t / v[:64]
        self.assertTrue(np.allclose(ratio, ratio[0], rtol=1e-4))

    def test_zero_vector_does_not_divide_by_zero(self):
        z = np.zeros(FULL_DIM, dtype=np.float32)
        out = truncate(z, 64)
        self.assertEqual(len(out), 64)
        self.assertFalse(np.any(np.isnan(out)))


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(short_ttl_seconds=30.0)
        self.vec = unit(np.random.default_rng(2).normal(0, 1, FULL_DIM))

    def test_stores_at_tier_dimension(self):
        for tier, dim in TIER_DIMS.items():
            e = self.store.add(tier, self.vec, 1.0, 2.0, label="x")
            self.assertEqual(e.embedding.shape[0], dim,
                              f"{tier} should store {dim} dims")

    def test_stats_counts_entries(self):
        self.store.add("short", self.vec, 0, 0)
        self.store.add("medium", self.vec, 0, 0, label="a")
        self.assertEqual(self.store.stats(),
                          {"short": 1, "medium": 1, "long": 0})

    def test_prune_short_term_drops_expired_only(self):
        store = MemoryStore(short_ttl_seconds=0.05)
        store.add("short", self.vec, 0, 0)
        store.add("medium", self.vec, 0, 0, label="keep")
        time.sleep(0.08)
        store.add("short", self.vec, 9, 9)   # fresh
        store.prune_short_term()
        self.assertEqual(len(store._entries["short"]), 1, "expired entry should go")
        self.assertEqual(store._entries["short"][0].x, 9)
        self.assertEqual(len(store._entries["medium"]), 1, "medium must be untouched")

    def test_query_returns_scores_descending(self):
        rng = np.random.default_rng(3)
        for i in range(5):
            self.store.add("long", unit(rng.normal(0, 1, FULL_DIM)), i, 0, label=str(i))
        got = self.store.query(self.vec, "long", top_k=5)
        scores = [s for s, _ in got]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_query_empty_tier_returns_empty(self):
        self.assertEqual(self.store.query(self.vec, "long", top_k=3), [])


class TestBestMatchRegression(unittest.TestCase):
    """The bug this project actually had: max() across tiers always chose the
    unlabeled short tier, because narrow MRL prefixes inflate cosine scores."""

    def setUp(self):
        self.store = MemoryStore(short_ttl_seconds=30.0)

        # query and the RIGHT answer: moderate agreement across all dims
        self.query = make_vector(head=0.05, tail_seed=100)
        right = make_vector(head=0.05, tail_seed=100)  # same seed => identical
        # a DECOY that agrees strongly in the first 64 dims but not beyond
        decoy = make_vector(head=0.05, tail_seed=999)

        self.store.add("long", right, 1.0, 1.0, label="right answer")
        self.store.add("medium", right, 1.0, 1.0, label="right answer")
        self.store.add("short", decoy, 7.0, 7.0)      # unlabeled, by design

    def test_short_tier_score_is_inflated(self):
        """Sanity-check the fixture reproduces the real phenomenon."""
        per_tier = self.store.query_all_tiers(self.query, top_k=1)
        self.assertGreater(per_tier["short"][0][0], per_tier["long"][0][0],
                            "fixture must reproduce short-tier score inflation")

    def test_best_match_does_not_answer_from_unlabeled_short_tier(self):
        thresholds = {"long": 0.5, "medium": 0.5, "short": 0.5}
        tier, score, entry = self.store.best_match(self.query, thresholds)
        self.assertEqual(tier, "long")
        self.assertEqual(entry.label, "right answer")

    def test_naive_max_would_have_picked_short(self):
        """Documents the old behaviour so nobody reintroduces it."""
        per_tier = self.store.query_all_tiers(self.query, top_k=1)
        naive_tier = max(per_tier, key=lambda t: per_tier[t][0][0]
                          if per_tier[t] else -1.0)
        self.assertEqual(naive_tier, "short",
                          "if this fails the fixture no longer covers the bug")

    def test_falls_back_to_most_durable_when_nothing_clears(self):
        impossible = {"long": 2.0, "medium": 2.0, "short": 2.0}
        tier, score, entry = self.store.best_match(self.query, impossible)
        self.assertEqual(tier, "long", "fallback must be the most durable tier")
        self.assertIsNotNone(entry)

    def test_skips_empty_tiers_in_priority_order(self):
        store = MemoryStore()
        v = make_vector(head=0.05, tail_seed=7)
        store.add("medium", v, 3.0, 4.0, label="only medium")
        tier, score, entry = store.best_match(v, {"long": 0.1, "medium": 0.1, "short": 0.1})
        self.assertEqual(tier, "medium")
        self.assertEqual(entry.label, "only medium")

    def test_object_in_view_is_not_mistaken_for_a_learned_object(self):
        """The always-watching loop writes every Nth frame into the short
        tier, so an object held up to the camera is in memory within a
        second. That must NOT make the gate think it has learned it —
        otherwise a robot never escalates on anything it can see."""
        store = MemoryStore()
        learned = make_vector(head=0.05, tail_seed=11)
        store.add("long", learned, 1.0, 1.0, label="learned thing")
        store.add("medium", learned, 1.0, 1.0, label="learned thing")

        # a novel object, in frame right now: short tier only, unlabelled
        novel = make_vector(head=0.05, tail_seed=77)
        store.add("short", novel, 9.0, 9.0)

        # Thresholds low enough that the short tier WOULD clear if consulted.
        thresholds = {"long": 0.9, "medium": 0.9, "short": 0.0}
        tier, score, entry = store.best_match(novel, thresholds)
        self.assertNotEqual(tier, "short",
                             "a novel object in view must not answer as learned")
        self.assertTrue(entry is None or entry.label is not None,
                        "an answer must never come from an unlabelled entry")

    def test_learned_only_false_can_still_see_the_short_tier(self):
        """'Did I just see this?' is a different question from 'do I know
        this?' — the short tier is the right answer to the first one."""
        store = MemoryStore()
        seen = make_vector(head=0.05, tail_seed=21)
        store.add("short", seen, 4.0, 5.0)
        tier, score, entry = store.best_match(
            seen, {"long": 0.1, "medium": 0.1, "short": 0.1},
            learned_only=False)
        self.assertEqual(tier, "short")
        self.assertIsNotNone(entry)

    def test_short_tier_is_not_a_learned_tier(self):
        self.assertNotIn("short", MemoryStore.LEARNED_TIERS)
        self.assertIn("long", MemoryStore.LEARNED_TIERS)
        self.assertIn("medium", MemoryStore.LEARNED_TIERS)

    def test_empty_store_returns_none(self):
        tier, score, entry = MemoryStore().best_match(
            self.query, {"long": 0.1, "medium": 0.1, "short": 0.1})
        self.assertIsNone(entry)
        self.assertIsNone(tier)
        self.assertEqual(score, -1.0)


class TestGate(unittest.TestCase):
    def test_manipulation_always_escalates_even_on_perfect_match(self):
        r = decide(1.0, requires_manipulation=True, tier="long")
        self.assertEqual(r.decision, Decision.ESCALATE)

    def test_high_similarity_is_cheap(self):
        r = decide(0.99, requires_manipulation=False, tier="long")
        self.assertEqual(r.decision, Decision.CHEAP)

    def test_zero_similarity_escalates_as_novel(self):
        r = decide(0.0, requires_manipulation=False, tier="long")
        self.assertEqual(r.decision, Decision.ESCALATE)
        self.assertIn("novel", r.reason)

    def test_thresholds_are_ordered_by_tier_width(self):
        """Narrower tiers must demand MORE similarity, since truncation
        inflates scores. If this inverts, novelty detection breaks."""
        t = gate.TIER_CONFIDENCE_THRESHOLDS
        self.assertGreaterEqual(t["short"], t["medium"])
        self.assertGreaterEqual(t["medium"], t["long"])

    def test_same_score_can_differ_by_tier(self):
        """A score that is confident at 768 dims should not automatically be
        confident at 64 dims."""
        score = gate.TIER_CONFIDENCE_THRESHOLDS["long"] + 0.001
        self.assertEqual(decide(score, tier="long").decision, Decision.CHEAP)
        if gate.TIER_CONFIDENCE_THRESHOLDS["short"] > score:
            self.assertEqual(decide(score, tier="short").decision, Decision.ESCALATE)

    def test_unknown_tier_falls_back_to_base_threshold(self):
        r = decide(0.99, tier="not_a_tier")
        self.assertEqual(r.decision, Decision.CHEAP)

    def test_reason_reports_the_threshold_used(self):
        r = decide(0.0, tier="medium")
        self.assertIn("medium", r.reason)


class TestCalibrationLoading(unittest.TestCase):
    def _conf(self):
        return {"long": 0.35, "medium": 0.38, "short": 0.43}

    def _nov(self):
        return {"long": 0.25, "medium": 0.28, "short": 0.33}

    def test_valid_file_overrides_defaults(self):
        conf, nov = self._conf(), self._nov()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"source": "unit test",
                        "confidence": {"long": 0.2, "medium": 0.22, "short": 0.3},
                        "novelty": {"long": 0.15}}, f)
            path = f.name
        try:
            src = load_calibration(path, conf, nov)
            self.assertEqual(src, "unit test")
            self.assertAlmostEqual(conf["long"], 0.2)
            self.assertAlmostEqual(conf["short"], 0.3)
            self.assertAlmostEqual(nov["long"], 0.15)
            self.assertAlmostEqual(nov["medium"], 0.28, msg="unspecified keys keep defaults")
        finally:
            os.unlink(path)

    def test_missing_file_is_a_no_op(self):
        conf = self._conf()
        self.assertIsNone(load_calibration("/nonexistent/thresholds.json", conf, self._nov()))
        self.assertAlmostEqual(conf["long"], 0.35)

    def test_malformed_file_does_not_raise_and_keeps_defaults(self):
        """A corrupt calibration file must not take the demo down."""
        conf = self._conf()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{ this is not json")
            path = f.name
        try:
            self.assertIsNone(load_calibration(path, conf, self._nov()))
            self.assertAlmostEqual(conf["long"], 0.35)
        finally:
            os.unlink(path)

    def test_non_numeric_value_does_not_raise(self):
        conf = self._conf()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"confidence": {"long": "banana"}}, f)
            path = f.name
        try:
            load_calibration(path, conf, self._nov())   # must not raise
        finally:
            os.unlink(path)

    def test_summary_states_whether_calibrated(self):
        s = gate.threshold_summary()
        self.assertTrue("calibrated from" in s or "PROVISIONAL" in s)


class TestMetrics(unittest.TestCase):
    def test_escalation_rate(self):
        m = MetricsLogger()
        m.log_decision("a", "long", 0.9, "cheap", 10.0)
        m.log_decision("b", "long", 0.1, "escalate", 10.0, escalate_latency_ms=1000.0)
        m.log_decision("c", "long", 0.1, "escalate", 10.0, escalate_latency_ms=1000.0)
        self.assertAlmostEqual(m.escalation_rate(), 2 / 3)

    def test_escalation_rate_empty_session(self):
        self.assertEqual(MetricsLogger().escalation_rate(), 0.0)

    def test_blended_cost_math(self):
        m = MetricsLogger()
        # 3 cheap @ 10ms, 1 escalate @ 1000ms -> actual avg = (30+1000)/4 = 257.5
        for i in range(3):
            m.log_decision(f"c{i}", "long", 0.9, "cheap", 10.0)
        m.log_decision("e", "long", 0.1, "escalate", 10.0, escalate_latency_ms=1000.0)
        b = m.blended_cost_vs_always_escalate()
        self.assertAlmostEqual(b["actual_avg_ms"], 257.5)
        self.assertAlmostEqual(b["always_escalate_avg_ms"], 1000.0)
        self.assertAlmostEqual(b["speedup_factor"], 1000.0 / 257.5)

    def test_blended_cost_with_no_escalations_is_undefined_not_a_crash(self):
        m = MetricsLogger()
        m.log_decision("c", "long", 0.9, "cheap", 10.0)
        b = m.blended_cost_vs_always_escalate()
        self.assertIsNone(b["speedup_factor"])

    def test_storage_footprint_favours_tiering(self):
        m = MetricsLogger()
        counts = {"short": 100, "medium": 10, "long": 1}
        s = m.storage_footprint(counts, TIER_DIMS, fixed_width_dim=1024)
        expected_tiered = (100 * 64 + 10 * 256 + 1 * 768) * 4
        expected_fixed = 111 * 1024 * 4
        self.assertEqual(s["tiered_bytes"], expected_tiered)
        self.assertEqual(s["fixed_width_bytes"], expected_fixed)
        self.assertGreater(s["savings_factor"], 1.0)

    def test_storage_footprint_empty_store(self):
        s = MetricsLogger().storage_footprint(
            {"short": 0, "medium": 0, "long": 0}, TIER_DIMS)
        self.assertIsNone(s["savings_factor"])

    def test_save_json_round_trip(self):
        m = MetricsLogger()
        m.log_decision("q", "long", 0.5, "cheap", 12.5)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            m.save_json(p)
            with open(p) as f:
                data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["query"], "q")


class TestFrameSource(unittest.TestCase):
    def setUp(self):
        from frame_source import FrameSource
        self.FrameSource = FrameSource
        self.dir = tempfile.mkdtemp()
        import cv2
        for i in range(3):
            img = np.full((16, 16, 3), i * 40, dtype=np.uint8)
            cv2.imwrite(os.path.join(self.dir, f"img_{i}.jpg"), img)

    def test_reads_and_loops_forever(self):
        src = self.FrameSource.open(images=self.dir)
        shapes = []
        for _ in range(7):   # more reads than files
            ok, frame = src.read()
            self.assertTrue(ok)
            shapes.append(frame.shape)
        self.assertEqual(len(shapes), 7)
        self.assertTrue(all(s == (16, 16, 3) for s in shapes))

    def test_current_name_tracks_served_frame(self):
        src = self.FrameSource.open(images=self.dir)
        src.read()
        self.assertEqual(src.current_name(), "img_0.jpg")
        src.read()
        self.assertEqual(src.current_name(), "img_1.jpg")

    def test_empty_directory_raises_clearly(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(RuntimeError) as ctx:
                self.FrameSource.open(images=empty)
            self.assertIn("No .jpg", str(ctx.exception))

    def test_camera_failure_message_is_actionable(self):
        """A blocked camera must explain the fix, not just say 'failed'."""
        with self.assertRaises(RuntimeError) as ctx:
            self.FrameSource.open(camera=99)   # no such device
        msg = str(ctx.exception)
        self.assertIn("Privacy & Security", msg)
        self.assertIn("--images", msg)

    def test_watch_reads_a_live_updating_file(self):
        """The camera-bridge path: same path, changing contents."""
        import cv2
        path = os.path.join(self.dir, "latest.jpg")
        cv2.imwrite(path, np.full((8, 8, 3), 10, dtype=np.uint8))
        src = self.FrameSource.open(watch=path)
        self.assertTrue(src.is_live)
        ok, first = src.read()
        self.assertTrue(ok)
        # publisher replaces the file
        cv2.imwrite(path, np.full((8, 8, 3), 200, dtype=np.uint8))
        ok, second = src.read()
        self.assertTrue(ok)
        self.assertGreater(float(second.mean()), float(first.mean()),
                            "watch mode must re-read the file, not cache it")

    def test_watch_missing_file_explains_the_bridge(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.FrameSource._open_watch(
                os.path.join(self.dir, "nope.jpg"), timeout_s=0.2)
        self.assertIn("camera_bridge.py", str(ctx.exception))

    def test_images_mode_is_not_live(self):
        self.assertFalse(self.FrameSource.open(images=self.dir).is_live)


class TestBaselineEmbedder(unittest.TestCase):
    """No network here — just that the NV-CLIP retirement is wired correctly."""

    def test_default_is_not_the_discontinued_nvclip(self):
        import nvidia_nim
        # Only meaningful when .env doesn't pin an override.
        if not os.environ.get("NVCLIP_MODEL"):
            self.assertNotEqual(nvidia_nim.BASELINE_EMBED_MODEL, "nvidia/nvclip",
                                 "nvidia/nvclip is discontinued and always 404s")

    def test_deprecated_alias_still_exists(self):
        import nvidia_nim
        self.assertIs(nvidia_nim.embed_image_nvclip, nvidia_nim.embed_image_baseline)

    def test_baseline_dim_matches_storage_default(self):
        """metrics.storage_footprint's default must match the real baseline
        width, or the storage comparison silently misreports."""
        import inspect
        import nvidia_nim
        from metrics import MetricsLogger
        default = inspect.signature(
            MetricsLogger.storage_footprint).parameters["fixed_width_dim"].default
        self.assertEqual(default, nvidia_nim.BASELINE_EMBED_DIM)

    def test_robot_prompt_frames_the_query(self):
        import nvidia_nim
        p = nvidia_nim.robot_perception_prompt("where is the mug")
        self.assertIn("where is the mug", p)
        self.assertIn("robot", p.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
