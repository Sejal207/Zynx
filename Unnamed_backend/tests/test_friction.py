"""Tests for Phase 1: deterministic behavioural friction analysis built on
top of the Phase 0 timestamped event stream. Pure unit tests over synthetic
event lists (no browser/LLM involved) plus one integration test that reuses
the Phase 0 test harness to reproduce the exact MyGov "stuck dropdown" loop
pattern end to end.
"""
import itertools
import os
import sys
import unittest
from unittest.mock import patch

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from orchestrator.friction import (
    analyze_friction, detect_friction_signals, group_into_episodes,
    CLICK_REPEAT_WINDOW_MS, HESITATION_THRESHOLD_MS, EPISODE_GAP_MS,
)

_counter = itertools.count()


def ev(event_type, t_ms, url="https://x.test/", index=None, text=None, tag=None,
       state_changed=None, had_error=None, metadata=None):
    return {
        "event_id": f"e{next(_counter)}_{event_type}",
        "timestamp_ms": float(t_ms),
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "event_type": event_type,
        "action": {"type": event_type, "index": index},
        "current_url": url,
        "element_index": index,
        "element_text": text,
        "element_tag": tag,
        "state_changed": state_changed,
        "had_error": had_error,
        "source": "backend",
        "metadata": metadata or {},
    }


class FrictionRuleTests(unittest.TestCase):

    def test_repeated_click_detection(self):
        events = [
            ev("task_start", 0),
            ev("click", 1000, index=5, text="Quiz", state_changed=True, had_error=False),
            ev("click", 4000, index=5, text="Quiz", state_changed=True, had_error=False),
            ev("click", 7000, index=5, text="Quiz", state_changed=False, had_error=False),
        ]
        signals = detect_friction_signals(events)
        repeated = [s for s in signals if s["signal_type"] == "repeated_click"]
        self.assertEqual(len(repeated), 1)
        self.assertEqual(repeated[0]["metadata"]["click_count"], 3)
        self.assertEqual(repeated[0]["severity"], "medium")
        self.assertEqual(len(repeated[0]["evidence_event_ids"]), 3)

    def test_repeated_click_outside_window_is_not_grouped(self):
        events = [
            ev("click", 0, index=5, state_changed=True, had_error=False),
            ev("click", CLICK_REPEAT_WINDOW_MS + 5000, index=5, state_changed=True, had_error=False),
        ]
        signals = detect_friction_signals(events)
        self.assertEqual([s for s in signals if s["signal_type"] == "repeated_click"], [])

    def test_dead_click_detection(self):
        events = [
            ev("click", 0, index=7, text="Quiz 33", state_changed=False, had_error=False),
            ev("click", 3000, index=23, text="Take Quiz", state_changed=False, had_error=False),
        ]
        signals = detect_friction_signals(events)
        dead = [s for s in signals if s["signal_type"] == "dead_click"]
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["metadata"]["count"], 2)

    def test_single_isolated_dead_click_is_not_flagged(self):
        """One harmless/no-effect click with nothing else nearby must not
        become a finding -- ordinary exploration, not a signal."""
        events = [
            ev("navigation", 0),
            ev("click", 5000, index=2, state_changed=False, had_error=False),
            ev("navigation", 10000, url="https://x.test/other"),
        ]
        signals = detect_friction_signals(events)
        self.assertEqual(signals, [])

    def test_excessive_scrolling(self):
        events = [
            ev("scroll", 0), ev("scroll", 1000), ev("scroll", 2000), ev("scroll", 3000),
            ev("click", 4000, index=1, state_changed=True, had_error=False),
        ]
        signals = detect_friction_signals(events)
        scrolls = [s for s in signals if s["signal_type"] == "excessive_scroll"]
        self.assertEqual(len(scrolls), 1)
        self.assertEqual(scrolls[0]["metadata"]["scroll_count"], 4)
        self.assertEqual(scrolls[0]["severity"], "high")

    def test_scroll_reversal(self):
        events = [
            ev("scroll", 0, metadata={"direction": "down"}),
            ev("scroll", 1000, metadata={"direction": "up"}),
        ]
        signals = detect_friction_signals(events)
        reversals = [s for s in signals if s["signal_type"] == "scroll_reversal"]
        self.assertEqual(len(reversals), 1)

    def test_scroll_reversal_absent_without_direction_metadata(self):
        """Documented limitation: current browser layer never records scroll
        direction, so this rule must find nothing on realistic data."""
        events = [ev("scroll", 0), ev("scroll", 1000), ev("scroll", 2000)]
        signals = detect_friction_signals(events)
        self.assertEqual([s for s in signals if s["signal_type"] == "scroll_reversal"], [])

    def test_backtracking(self):
        events = [
            ev("navigation", 0, url="https://x.test/a"),
            ev("navigation", 1000, url="https://x.test/b"),
            ev("navigation", 2000, url="https://x.test/a"),  # back to a previously visited page
        ]
        signals = detect_friction_signals(events)
        back = [s for s in signals if s["signal_type"] == "backtracking"]
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["affected_url"], "https://x.test/a")

    def test_navigation_loop(self):
        events = [
            ev("navigation", 0, url="https://x.test/a"),
            ev("navigation", 1000, url="https://x.test/b"),
            ev("navigation", 2000, url="https://x.test/a"),
            ev("navigation", 3000, url="https://x.test/b"),
            ev("navigation", 4000, url="https://x.test/a"),
        ]
        signals = detect_friction_signals(events)
        loops = [s for s in signals if s["signal_type"] == "navigation_loop"]
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0]["affected_url"], "https://x.test/a")
        self.assertEqual(loops[0]["metadata"]["visit_count"], 3)

    def test_hesitation(self):
        events = [
            ev("click", 0, index=1, state_changed=True, had_error=False),
            ev("click", HESITATION_THRESHOLD_MS + 1000, index=2, state_changed=True, had_error=False),
        ]
        signals = detect_friction_signals(events)
        hes = [s for s in signals if s["signal_type"] == "hesitation"]
        self.assertEqual(len(hes), 1)
        self.assertIn("gap_ms", hes[0]["metadata"])
        self.assertLess(hes[0]["confidence"], 1.0)  # never asserted with full certainty

    def test_repeated_action_loop_is_not_duplicated(self):
        events = [
            ev("repeated_action", 0, index=16, url="https://x.test/"),
            ev("repeated_action", 3000, index=16, url="https://x.test/"),
        ]
        signals = detect_friction_signals(events)
        loop_signals = [s for s in signals if s["signal_type"] == "repeated_action_loop"]
        self.assertEqual(len(loop_signals), 1, "both guard events for the same element must merge into ONE signal")
        self.assertEqual(loop_signals[0]["metadata"]["guard_interventions"], 2)


class EpisodeGroupingTests(unittest.TestCase):

    def test_episode_groups_nearby_signals(self):
        events = [
            ev("click", 0, index=5, state_changed=True, had_error=False),
            ev("click", 3000, index=5, state_changed=True, had_error=False),
            ev("click", 6000, index=5, state_changed=False, had_error=False),  # repeated_click ends ~6000
            ev("click", 6000 + EPISODE_GAP_MS - 2000, index=9, state_changed=False, had_error=False),
            ev("click", 6000 + EPISODE_GAP_MS - 1000, index=12, state_changed=False, had_error=False),  # dead_click nearby
        ]
        result = analyze_friction(events)
        self.assertEqual(result["episode_count"], 1)
        ep = result["episodes"][0]
        self.assertIn("repeated_click", ep["signal_types"])
        self.assertIn("dead_click", ep["signal_types"])
        # The ~10s gap between the two clusters also legitimately exceeds the
        # hesitation threshold, so a real hesitation signal is correctly
        # detected too and folds into this same episode -- not a bug.
        self.assertGreaterEqual(ep["signal_count"], 2)

    def test_distant_signals_form_separate_episodes(self):
        far_gap = EPISODE_GAP_MS + 20000
        events = [
            ev("click", 0, index=5, state_changed=True, had_error=False),
            ev("click", 3000, index=5, state_changed=True, had_error=False),
            ev("click", 3000 + far_gap, index=9, state_changed=True, had_error=False),
            ev("click", 6000 + far_gap, index=9, state_changed=True, had_error=False),
        ]
        result = analyze_friction(events)
        self.assertEqual(result["episode_count"], 2)

    def test_overlapping_signal_time_ranges_merge_cleanly(self):
        """Two signals whose evidence event timestamps overlap in range must
        still merge into a single episode without corrupting evidence."""
        events = [
            ev("click", 0, index=1, state_changed=True, had_error=False),
            ev("click", 2000, index=1, state_changed=True, had_error=False),
            ev("click", 1000, index=2, state_changed=False, had_error=False),  # interleaved in time
            ev("click", 1500, index=3, state_changed=False, had_error=False),
        ]
        result = analyze_friction(events)
        self.assertEqual(result["episode_count"], 1)
        ep = result["episodes"][0]
        all_evidence = ep["evidence_event_ids"]
        self.assertEqual(len(all_evidence), len(set(all_evidence)), "no duplicated evidence ids")

    def test_no_false_episode_from_one_normal_click(self):
        events = [
            ev("navigation", 0),
            ev("click", 2000, index=0, text="Home", state_changed=True, had_error=False),
            ev("navigation", 4000, url="https://x.test/home"),
        ]
        result = analyze_friction(events)
        self.assertEqual(result["episode_count"], 0)
        self.assertEqual(result["signal_count"], 0)

    def test_timestamp_ordering_of_episodes(self):
        events = [
            ev("click", 50000, index=1, state_changed=True, had_error=False),
            ev("click", 52000, index=1, state_changed=True, had_error=False),
            ev("click", 0, index=9, state_changed=True, had_error=False),
            ev("click", 2000, index=9, state_changed=True, had_error=False),
        ]
        result = analyze_friction(events)
        starts = [ep["start_time_ms"] for ep in result["episodes"]]
        self.assertEqual(starts, sorted(starts))

    def test_evidence_preservation(self):
        """The exact event_ids that produced a signal must be exactly what
        the episode reports as evidence -- nothing added, nothing dropped."""
        e1 = ev("click", 0, index=4, state_changed=True, had_error=False)
        e2 = ev("click", 3000, index=4, state_changed=True, had_error=False)
        e3 = ev("click", 6000, index=4, state_changed=False, had_error=False)
        signals = detect_friction_signals([e1, e2, e3])
        repeated = [s for s in signals if s["signal_type"] == "repeated_click"][0]
        self.assertEqual(set(repeated["evidence_event_ids"]), {e1["event_id"], e2["event_id"], e3["event_id"]})
        episodes = group_into_episodes(signals)
        self.assertEqual(set(episodes[0]["evidence_event_ids"]), {e1["event_id"], e2["event_id"], e3["event_id"]})


class FrictionIntegrationTest(unittest.TestCase):
    """Reuses the Phase 0 fake-browser harness to reproduce the exact
    real-world MyGov pattern (stuck dropdown -> loop guard fires -> agent
    recovers) end to end through run_evaluation_stream, and verifies the
    loop guard's own events surface as behavioural friction evidence."""

    def test_baseline_loop_is_represented_as_friction_evidence(self):
        import asyncio
        import orchestrator.runner as runner

        class FakeBrowser:
            def __init__(self, headless=False):
                self._url = "https://x.test/"
                self._click_count = 0

            async def start(self): pass
            async def stop(self): pass

            @property
            def current_url(self):
                return self._url

            async def navigate(self, url):
                self._url = url
                return "", [{"index": 0, "tag": "a", "text": "Quiz", "aria_label": "", "id": ""}], "Home"

            async def execute_action(self, action_type, index=None, selector=None, value=None):
                if action_type == "click" and index == 0:
                    self._click_count += 1
                # Stuck dropdown: nothing ever changes for this element/url.
                return "", [{"index": 0, "tag": "a", "text": "Quiz", "aria_label": "", "id": ""}], "Home"

        def fake_decide(state):
            clicks_idx0 = sum(
                1 for a in state["interaction_history"]
                if a.get("type") == "click" and a.get("index") == 0
            )
            if clicks_idx0 < 3:
                return {"thought_trace": "clicking the dropdown", "action": {"type": "click", "index": 0, "value": None}}
            return {"thought_trace": "giving up", "action": {"type": "complete", "index": None, "value": None}}

        with patch.object(runner, "MultimodalVisionBrowser", return_value=FakeBrowser()), \
             patch.object(runner._navigator, "decide_next_action", side_effect=fake_decide), \
             patch.object(runner._evaluator, "evaluate", return_value={
                 "effectiveness": {"score": 3}, "efficiency": {"score": 3}, "learnability": {"score": 3},
                 "cognitive_load": {"score": 5}, "system1_system2_ratio": {"system1_percent": 50, "system2_percent": 50},
                 "overall_ux_score": 3, "top_issues": [], "top_positives": [], "friction_points": [],
                 "psychological_biases": {},
             }), \
             patch.object(runner._reporter, "generate_report", return_value={
                 "executive_summary": "ok", "psychological_friction_analysis": "ok",
                 "key_recommendations": [], "markdown_report": "# ok",
             }):

            async def _drain():
                out = []
                async for item in runner.run_evaluation_stream(
                    run_id="fric-test", target_url="https://x.test/",
                    task_description="find quiz", persona={},
                ):
                    out.append(item)
                return out

            payloads = asyncio.run(_drain())

        friction_payloads = [p for p in payloads if p.get("event") == "friction"]
        self.assertEqual(len(friction_payloads), 1)
        episodes = friction_payloads[0]["episodes"]
        self.assertGreaterEqual(len(episodes), 1, "the stuck-dropdown loop must produce at least one friction episode")

        all_signal_types = set()
        for ep in episodes:
            all_signal_types.update(ep["signal_types"])
        self.assertIn("repeated_action_loop", all_signal_types,
                       "the navigator's own loop-guard events must appear as friction evidence")

        # existing report/scorecard flow must still work unchanged
        report_payloads = [p for p in payloads if p.get("event") == "report"]
        self.assertEqual(len(report_payloads), 1)
        self.assertIn("friction", report_payloads[0])


if __name__ == "__main__":
    unittest.main()
