"""Tests for Phase 0 of the multimodal UX extension: the timestamped UX event
model layered onto the existing browser-evaluation loop.

Uses only the standard library (unittest + unittest.mock) -- no new
dependency was added for this. The real Playwright browser and the real LLM
calls are replaced with lightweight fakes so these tests are fast,
deterministic, and need no network/API key/browser binary.
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import patch

# Make `orchestrator` importable regardless of the current working directory
# this is invoked from (mirrors how run_test.py already assumes cwd is the
# Unnamed_backend directory, but doesn't require it).
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import orchestrator.runner as runner
from orchestrator.events import EVENT_TYPES, SessionClock, make_event


class FakeVisionBrowser:
    """Stands in for MultimodalVisionBrowser: no Playwright, no network.

    `script` maps an action_type ("navigate" is handled by `navigate()`
    itself) to a callable(call_index) -> (elements, url) so a test can
    control exactly what the "page" looks like after each action.
    """

    def __init__(self, headless=False, script=None, start_url="https://example.test/"):
        self.headless = headless
        self._url = start_url
        self._script = script or {}
        self._call_counts = {}

    async def start(self):
        pass

    async def stop(self):
        pass

    @property
    def current_url(self):
        return self._url

    async def navigate(self, url):
        self._url = url
        elements = self._script.get("_initial_elements", [
            {"index": 0, "tag": "a", "text": "Quiz", "aria_label": "", "id": ""},
            {"index": 1, "tag": "a", "text": "About", "aria_label": "", "id": ""},
        ])
        return "", elements, "Fake Home Page"

    async def execute_action(self, action_type, index=None, selector=None, value=None):
        n = self._call_counts.get(action_type, 0)
        self._call_counts[action_type] = n + 1
        handler = self._script.get(action_type)
        if handler is not None:
            elements, url = handler(n)
            self._url = url
            return "", elements, "Fake Page"
        # default: nothing changes
        return "", self._script.get("_initial_elements", []), "Fake Page"


def _run(coro_gen):
    """Drain an async generator into a plain list, synchronously."""
    async def _drain():
        out = []
        async for item in coro_gen:
            out.append(item)
        return out
    return asyncio.run(_drain())


class TimestampedEventModelTests(unittest.TestCase):
    """Covers items 1-9 from the Phase 0 test plan."""

    def setUp(self):
        # Evaluator/reporter never make real LLM calls in these tests.
        self.evaluate_patch = patch.object(
            runner._evaluator, "evaluate",
            return_value={
                "effectiveness": {"score": 5}, "efficiency": {"score": 5},
                "learnability": {"score": 5}, "cognitive_load": {"score": 5},
                "system1_system2_ratio": {"system1_percent": 50, "system2_percent": 50},
                "overall_ux_score": 5, "top_issues": [], "top_positives": [],
                "friction_points": [], "psychological_biases": {},
            },
        )
        self.report_patch = patch.object(
            runner._reporter, "generate_report",
            return_value={
                "executive_summary": "ok", "psychological_friction_analysis": "ok",
                "key_recommendations": [], "markdown_report": "# ok",
            },
        )
        self.evaluate_patch.start()
        self.report_patch.start()
        self.addCleanup(self.evaluate_patch.stop)
        self.addCleanup(self.report_patch.stop)

    def _run_stream(self, browser, decisions):
        """Run run_evaluation_stream with a scripted browser and a scripted
        sequence of navigator decisions, and return all yielded payloads."""
        with patch.object(runner, "MultimodalVisionBrowser", return_value=browser), \
             patch.object(runner._navigator, "decide_next_action", side_effect=decisions):
            gen = runner.run_evaluation_stream(
                run_id="test-run", target_url="https://example.test/",
                task_description="find the quiz", persona={"persona_type": "novice", "tech_literacy": 4},
            )
            return _run(gen)

    def test_full_run_produces_timestamped_events_of_expected_types(self):
        browser = FakeVisionBrowser(script={
            "click": lambda n: (
                [{"index": 0, "tag": "a", "text": "Take Quiz", "aria_label": "", "id": ""}],
                "https://example.test/quiz",
            ),
            "scroll": lambda n: (
                [{"index": 0, "tag": "a", "text": "Take Quiz", "aria_label": "", "id": ""}],
                "https://example.test/quiz",
            ),
        })
        decisions = [
            {"thought_trace": "clicking quiz link", "action": {"type": "click", "index": 0, "value": None}},
            {"thought_trace": "scrolling to look for more", "action": {"type": "scroll", "index": None, "value": None}},
            {"thought_trace": "done here", "action": {"type": "complete", "index": None, "value": None}},
        ]
        payloads = self._run_stream(browser, decisions)

        done = [p for p in payloads if p.get("event") == "done"]
        self.assertEqual(len(done), 1)
        events = done[0]["events"]
        types_seen = {e["event_type"] for e in events}

        # 1: task_start has a timestamp
        task_start_events = [e for e in events if e["event_type"] == "task_start"]
        self.assertEqual(len(task_start_events), 1)
        self.assertIsInstance(task_start_events[0]["timestamp_ms"], float)
        self.assertGreaterEqual(task_start_events[0]["timestamp_ms"], 0.0)

        # 2: navigation has a timestamp
        nav_events = [e for e in events if e["event_type"] == "navigation"]
        self.assertGreaterEqual(len(nav_events), 1)
        for e in nav_events:
            self.assertIsInstance(e["timestamp_ms"], float)

        # 3: click has a timestamp
        click_events = [e for e in events if e["event_type"] == "click"]
        self.assertEqual(len(click_events), 1)
        self.assertIsInstance(click_events[0]["timestamp_ms"], float)
        self.assertEqual(click_events[0]["element_index"], 0)
        self.assertEqual(click_events[0]["element_text"], "Quiz")  # from pre-action DOM

        # 4: scroll has a timestamp
        scroll_events = [e for e in events if e["event_type"] == "scroll"]
        self.assertEqual(len(scroll_events), 1)
        self.assertIsInstance(scroll_events[0]["timestamp_ms"], float)

        # task_end recorded (agent completed)
        task_end_events = [e for e in events if e["event_type"] == "task_end"]
        self.assertEqual(len(task_end_events), 1)
        self.assertEqual(task_end_events[0]["metadata"]["reason"], "agent_completed")

        # 5: chronologically ordered (monotonic, non-decreasing)
        timestamps = [e["timestamp_ms"] for e in events]
        self.assertEqual(timestamps, sorted(timestamps))

        # 6: distinguishable even if two timestamps tie at this resolution
        event_ids = [e["event_id"] for e in events]
        self.assertEqual(len(event_ids), len(set(event_ids)), "event_ids must be unique")

        # every event type actually used is a known, declared type
        self.assertTrue(types_seen.issubset(set(EVENT_TYPES)))

        # 8: existing interaction_history still has its original fields
        # (retrieved indirectly via the execute_action node_complete payloads,
        # since state isn't returned directly by the generator)
        exec_payloads = [p for p in payloads if p.get("event") == "node_complete" and p.get("node") == "execute_action"]
        self.assertGreaterEqual(len(exec_payloads), 1)
        last_action = exec_payloads[-1]["action"]
        self.assertIn("type", last_action)          # original field preserved
        self.assertIn("timestamp_ms", last_action)   # new field added
        self.assertIn("event_id", last_action)        # new field added

        # 9: existing evaluator/report flow still works (mocked output passed through unchanged)
        scorecard_payloads = [p for p in payloads if p.get("event") == "scorecard"]
        report_payloads = [p for p in payloads if p.get("event") == "report"]
        self.assertEqual(len(scorecard_payloads), 1)
        self.assertEqual(len(report_payloads), 1)
        self.assertEqual(scorecard_payloads[0]["scorecard"]["overall_ux_score"], 5)
        self.assertEqual(report_payloads[0]["report"]["executive_summary"], "ok")

    def test_timestamps_are_not_based_on_sse_delivery_time(self):
        """run_evaluation_stream is called directly here, with no SSE/HTTP
        layer in between at all (that layer -- main.py's event_stream(), with
        its asyncio.sleep(0.02) pacing -- is entirely bypassed in this test).
        So any timestamp recorded here could not have come from SSE arrival
        time; it can only have come from the backend/browser action boundary
        where _record() is actually called, which is what we assert on
        directly: timestamps are already fully formed at yield time, and
        artificially delaying *consumption* of the generator changes nothing
        about the values already recorded.
        """
        browser = FakeVisionBrowser()
        decisions = [
            {"thought_trace": "t", "action": {"type": "scroll", "index": None, "value": None}},
            {"thought_trace": "t2", "action": {"type": "complete", "index": None, "value": None}},
        ]

        async def _drain_with_artificial_delay():
            out = []
            with patch.object(runner, "MultimodalVisionBrowser", return_value=browser), \
                 patch.object(runner._navigator, "decide_next_action", side_effect=decisions):
                gen = runner.run_evaluation_stream(
                    run_id="test-run-2", target_url="https://example.test/",
                    task_description="x", persona={},
                )
                async for item in gen:
                    await asyncio.sleep(0.05)  # simulate slow SSE consumption
                    out.append(item)
            return out

        payloads = asyncio.run(_drain_with_artificial_delay())
        done = [p for p in payloads if p.get("event") == "done"][0]
        events = done["events"]
        # All recorded timestamps must be small (recorded during the fast,
        # un-delayed generator body), even though total consumption above
        # took several times 0.05s per event due to the artificial delay.
        max_ts = max(e["timestamp_ms"] for e in events)
        self.assertLess(max_ts, 500.0,
                         "timestamps should reflect fast in-generator execution, "
                         "not the artificially slow consumption loop")

    def test_repeated_action_guard_still_works_and_is_recorded_as_an_event(self):
        """Same click, same element, same URL, page never changes -- the
        exact real-world symptom this guard was built for."""
        browser = FakeVisionBrowser(script={
            "click": lambda n: (
                [{"index": 0, "tag": "a", "text": "Quiz", "aria_label": "", "id": ""}],
                "https://example.test/",  # URL never changes -- stuck dropdown
            ),
            "scroll": lambda n: (
                [{"index": 0, "tag": "a", "text": "Quiz", "aria_label": "", "id": ""}],
                "https://example.test/",
            ),
        })

        def fake_decide(state):
            clicks_idx0 = sum(
                1 for a in state["interaction_history"]
                if a.get("type") == "click" and a.get("index") == 0
            )
            if clicks_idx0 < 3:
                return {"thought_trace": "clicking the dropdown", "action": {"type": "click", "index": 0, "value": None}}
            return {"thought_trace": "giving up", "action": {"type": "complete", "index": None, "value": None}}

        with patch.object(runner, "MultimodalVisionBrowser", return_value=browser), \
             patch.object(runner._navigator, "decide_next_action", side_effect=fake_decide):
            gen = runner.run_evaluation_stream(
                run_id="test-run-3", target_url="https://example.test/",
                task_description="find the quiz", persona={},
            )
            payloads = _run(gen)

        done = [p for p in payloads if p.get("event") == "done"][0]
        events = done["events"]

        repeated_events = [e for e in events if e["event_type"] == "repeated_action"]
        self.assertGreaterEqual(len(repeated_events), 1,
                                 "expected at least one repeated_action event to be recorded")

        exec_payloads = [p for p in payloads if p.get("event") == "node_complete" and p.get("node") == "execute_action"]
        self.assertTrue(any(len(p["errors"]) > 0 for p in exec_payloads),
                         "repeated-action loop should be recorded in errors, visible to the evaluator")

        # the loop must never have run past MAX_STEPS
        self.assertLessEqual(len(exec_payloads), runner.MAX_STEPS)


class SessionClockUnitTests(unittest.TestCase):
    """Direct unit tests of the event model itself, independent of the runner."""

    def test_make_event_has_monotonic_and_absolute_timestamps(self):
        clock = SessionClock()
        ev = make_event(clock, "task_start", current_url="https://x.test/")
        self.assertIsInstance(ev["timestamp_ms"], float)
        self.assertGreaterEqual(ev["timestamp_ms"], 0.0)
        self.assertIsInstance(ev["timestamp_utc"], str)
        # must parse as a real ISO8601 timestamp
        from datetime import datetime
        datetime.fromisoformat(ev["timestamp_utc"])

    def test_events_from_same_clock_are_ordered(self):
        clock = SessionClock()
        e1 = make_event(clock, "click")
        e2 = make_event(clock, "scroll")
        self.assertLessEqual(e1["timestamp_ms"], e2["timestamp_ms"])
        self.assertNotEqual(e1["event_id"], e2["event_id"])


if __name__ == "__main__":
    unittest.main()
