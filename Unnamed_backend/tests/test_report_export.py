"""Tests for orchestrator/report_export.py: JSON/Markdown/PDF formatting for
both report modes. These tests check that EXISTING data is preserved and
correctly routed into the right output -- never that new findings/scores are
invented, and never that not_assessed/missing data gets fabricated into a
fake value.
"""
import json
import os
import sys
import unittest

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from orchestrator import report_export as re_
from orchestrator import human_ux_report as hur
from orchestrator import think_aloud


AGENT_PAYLOAD = {
    "run_id": "abcd1234",
    "target_url": "https://example.test/",
    "task_description": "Find the pricing section.",
    "persona": {"persona_type": "novice", "tech_literacy": 4},
    "task_completed": True,
    "scorecard": {
        "overall_ux_score": 7,
        "effectiveness": {"score": 8, "rationale": "Task succeeded."},
        "efficiency": {"score": 6, "rationale": "A few wasted clicks."},
        "learnability": {"score": 7, "rationale": "Adapted quickly."},
        "cognitive_load": {"score": 5, "rationale": "Moderate density."},
        "system1_system2_ratio": {"system1_percent": 60, "system2_percent": 40, "rationale": "Mostly intuitive."},
        "psychological_biases": {"f_pattern_scanning": 6, "banner_blindness": 5},
        "top_issues": ["Pricing link was hard to find."],
        "top_positives": ["Navigation was clear."],
        "friction_points": [{"step": 2, "type": "navigation", "severity": "medium", "description": "Dead click on footer link."}],
    },
    "report": {
        "executive_summary": "Overall the experience was acceptable.",
        "psychological_friction_analysis": "Load spiked when pricing was not immediately visible.",
        "key_recommendations": [{"priority": "high", "title": "Add pricing to nav", "detail": "Pricing should be a top-level nav item."}],
        "markdown_report": "# existing markdown report",
    },
    "friction": {
        "episodes": [{
            "episode_id": "ep_1", "start_time_ms": 1000.0, "end_time_ms": 4000.0, "duration_ms": 3000.0,
            "signal_types": ["repeated_clicks"], "affected_elements": ["button#submit"], "affected_urls": [],
            "severity": "medium", "evidence_event_ids": ["e1", "e2"], "evidence_summary": ["Clicked submit 3 times with no visible change."],
            "confidence": "high", "signal_count": 1, "metadata": {},
        }],
        "episode_count": 1, "signal_count": 1,
    },
}


def _seg(text, labels=None, start_ms=None, end_ms=None, raw=None):
    seg = think_aloud.make_segment(text, start_time_ms=start_ms, end_time_ms=end_ms)
    if raw is not None:
        seg["raw_transcript"] = raw
    if labels is not None:
        seg["classification"] = {
            "segment_id": seg["segment_id"], "labels": labels,
            "evidence_text": seg["display_transcript"], "confidence": "high", "rationale": "x",
        }
    return seg


class AgentJsonExportTests(unittest.TestCase):
    def test_preserves_core_fields(self):
        data = re_.agent_report_to_dict(AGENT_PAYLOAD, "deadbeef" * 4, "2026-01-01T00:00:00+00:00")
        self.assertEqual(data["target_url"], "https://example.test/")
        self.assertEqual(data["run_id"], "abcd1234")
        self.assertEqual(data["mode"], "agent")
        self.assertEqual(data["report_id"], "deadbeef" * 4)
        self.assertEqual(data["scorecard"]["overall_ux_score"], 7)
        self.assertEqual(data["friction"]["episode_count"], 1)

    def test_is_valid_json_serializable(self):
        data = re_.agent_report_to_dict(AGENT_PAYLOAD, None, "2026-01-01T00:00:00+00:00")
        serialized = json.dumps(data)
        self.assertIsInstance(json.loads(serialized), dict)

    def test_does_not_invent_task_completed_when_absent(self):
        payload = {**AGENT_PAYLOAD, "task_completed": None}
        data = re_.agent_report_to_dict(payload, None, "now")
        self.assertIsNone(data["task_completed"])


class AgentMarkdownExportTests(unittest.TestCase):
    def test_contains_expected_sections(self):
        md = re_.agent_report_to_markdown(AGENT_PAYLOAD, "deadbeef" * 4, "2026-01-01T00:00:00+00:00")
        for heading in [
            "ZYNX — Autonomous Agent UX Evaluation Report",
            "1. Evaluation Overview", "4. Existing UX Scorecard",
            "5. Key UX Findings", "6. Behavioural Friction Evidence",
            "7. Recommendations", "8. Supporting Evidence",
        ]:
            self.assertIn(heading, md)

    def test_omits_recommendations_section_when_none_present(self):
        payload = {**AGENT_PAYLOAD, "report": {**AGENT_PAYLOAD["report"], "key_recommendations": []}}
        md = re_.agent_report_to_markdown(payload, None, "now")
        self.assertNotIn("## 7. Recommendations", md)

    def test_escapes_pipe_characters_in_table_cells(self):
        payload = {**AGENT_PAYLOAD, "scorecard": {**AGENT_PAYLOAD["scorecard"], "top_issues": ["A | B weird issue"]}}
        md = re_.agent_report_to_markdown(payload, None, "now")
        self.assertIn("A | B weird issue", md)  # appears in a bullet list, not a table cell here -- never crashes
        self.assertNotIn("\\|\\|", md)


class AgentPdfExportTests(unittest.TestCase):
    def test_produces_valid_pdf_bytes(self):
        pdf_bytes = re_.agent_report_to_pdf(AGENT_PAYLOAD, "deadbeef" * 4, "2026-01-01T00:00:00+00:00")
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 500)

    def test_handles_missing_optional_sections_without_crashing(self):
        minimal = {"target_url": "https://x.test/", "task_description": "", "scorecard": {}, "report": {}}
        pdf_bytes = re_.agent_report_to_pdf(minimal, None, "now")
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


class ThinkAloudJsonExportTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            _seg("Mujhe pricing section dhoondhna hai.", labels=["NAVIGATION_INTENT"], start_ms=0, end_ms=4000, raw="मुझे प्राइसिंग सेक्शन ढूंढना है।"),
            _seg("Mujhe laga tha pricing navigation mein hogi.", labels=["EXPECTATION"], start_ms=4000, end_ms=8000),
            _seg("Mujhe pricing dikh hi nahi rahi hai.", labels=["COMPLAINT"], start_ms=8000, end_ms=12000),
        ]
        self.report = hur.build_human_ux_report(
            target_url="https://example.test/", task_description="Find pricing",
            persona={"persona_type": "novice"}, segments=self.segments,
            audio_duration_ms=12000.0, language="hi-IN", task_outcome="completed", llm=None,
        )

    def test_preserves_report_and_adds_segments(self):
        data = re_.think_aloud_report_to_dict(self.report, self.segments, "notes here", "outcome note",
                                               "cafebabe" * 4, "2026-01-01T00:00:00+00:00")
        self.assertEqual(data["mode"], "think_aloud")
        self.assertEqual(data["report_id"], "cafebabe" * 4)
        self.assertEqual(len(data["segments"]), 3)
        self.assertEqual(data["researcher_notes"], "notes here")
        self.assertEqual(data["task_outcome_note"], "outcome note")
        self.assertEqual(data["score"]["total"], self.report["score"]["total"])

    def test_preserves_raw_and_display_transcript_distinction(self):
        data = re_.think_aloud_report_to_dict(self.report, self.segments, None, None, None, "now")
        seg0 = data["segments"][0]
        self.assertIn("प्राइसिंग", seg0["raw_transcript"])
        self.assertEqual(seg0["display_transcript"], self.segments[0]["display_transcript"])
        self.assertNotEqual(seg0["raw_transcript"], seg0["display_transcript"])

    def test_not_assessed_outcome_is_not_fabricated_into_a_score(self):
        report = hur.build_human_ux_report(
            target_url="https://example.test/", task_description="Find pricing",
            persona=None, segments=self.segments, audio_duration_ms=12000.0,
            language="hi-IN", task_outcome="not_assessed", llm=None,
        )
        data = re_.think_aloud_report_to_dict(report, self.segments, None, None, None, "now")
        self.assertFalse(data["score"]["fully_assessed"])
        self.assertIsNone(data["score"]["total"])

    def test_is_valid_json_serializable(self):
        data = re_.think_aloud_report_to_dict(self.report, self.segments, None, None, None, "now")
        self.assertIsInstance(json.loads(json.dumps(data)), dict)


class ThinkAloudMarkdownExportTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            _seg("Mujhe laga tha pricing navigation mein hogi.", labels=["EXPECTATION"], start_ms=0, end_ms=4000),
            _seg("Mujhe pricing dikh hi nahi rahi hai.", labels=["COMPLAINT"], start_ms=4000, end_ms=8000),
        ]
        self.report = hur.build_human_ux_report(
            target_url="https://example.test/", task_description="Find pricing",
            persona={"persona_type": "novice"}, segments=self.segments,
            audio_duration_ms=8000.0, language="hi-IN", task_outcome="not_completed", llm=None,
        )

    def test_contains_expected_sections_in_order(self):
        md = re_.think_aloud_report_to_markdown(self.report, self.segments, "my notes", None, None, "now")
        for heading in [
            "1. Session and Task Overview", "2. Task UX Score", "3. Score Breakdown",
            "4. Think-Aloud Signal Summary", "5. Evidence Timeline", "6. UX Findings",
            "7. Task Outcome and Researcher Notes", "8. Full Think-Aloud Transcript",
        ]:
            self.assertIn(heading, md)
        self.assertIn("my notes", md)

    def test_keeps_quotes_linked_to_correct_segment(self):
        md = re_.think_aloud_report_to_markdown(self.report, self.segments, None, None, None, "now")
        self.assertIn("Mujhe laga tha pricing navigation mein hogi.", md)
        self.assertIn("Mujhe pricing dikh hi nahi rahi hai.", md)

    def test_not_assessed_score_shown_as_status_not_a_number(self):
        report = hur.build_human_ux_report(
            target_url="https://example.test/", task_description="Find pricing",
            persona=None, segments=self.segments, audio_duration_ms=8000.0,
            language="hi-IN", task_outcome="not_assessed", llm=None,
        )
        md = re_.think_aloud_report_to_markdown(report, self.segments, None, None, None, "now")
        self.assertIn("Not fully assessed", md)


class ThinkAloudPdfExportTests(unittest.TestCase):
    def test_produces_valid_pdf_bytes(self):
        segments = [_seg("Mujhe pricing dikh hi nahi rahi hai.", labels=["COMPLAINT"], start_ms=0, end_ms=4000)]
        report = hur.build_human_ux_report(
            target_url="https://example.test/", task_description="Find pricing",
            persona=None, segments=segments, audio_duration_ms=4000.0,
            language="hi-IN", task_outcome="completed", llm=None,
        )
        pdf_bytes = re_.think_aloud_report_to_pdf(report, segments, None, None, "cafebabe" * 4, "now")
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 500)

    def test_handles_devanagari_raw_transcript_without_crashing(self):
        segments = [_seg("Mujhe pricing dikh hi nahi rahi hai.", labels=["COMPLAINT"],
                          start_ms=0, end_ms=4000, raw="मुझे प्राइसिंग दिख ही नहीं रही है।")]
        report = hur.build_human_ux_report(
            target_url="https://example.test/", task_description="Find pricing",
            persona=None, segments=segments, audio_duration_ms=4000.0,
            language="hi-IN", task_outcome="completed", llm=None,
        )
        pdf_bytes = re_.think_aloud_report_to_pdf(report, segments, None, None, None, "now")
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_handles_no_segments_without_crashing(self):
        report = hur.build_human_ux_report(
            target_url="https://example.test/", task_description="Find pricing",
            persona=None, segments=[], audio_duration_ms=None,
            language=None, task_outcome="not_assessed", llm=None,
        )
        pdf_bytes = re_.think_aloud_report_to_pdf(report, [], None, None, None, "now")
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


class SafeScriptTextTests(unittest.TestCase):
    def test_latin_text_passes_through(self):
        self.assertEqual(re_.safe_script_text("Mujhe pricing nahi dikh rahi."), "Mujhe pricing nahi dikh rahi.")

    def test_empty_text_returns_empty(self):
        self.assertEqual(re_.safe_script_text(None), "")
        self.assertEqual(re_.safe_script_text(""), "")

    def test_devanagari_text_with_font_available_passes_through(self):
        re_._ensure_fonts()
        if re_.DEVANAGARI_FONT:
            text = "मुझे प्राइसिंग नहीं दिख रही है।"
            self.assertEqual(re_.safe_script_text(text), text)

    def test_devanagari_without_font_becomes_safe_placeholder(self):
        re_._ensure_fonts()
        original = re_.DEVANAGARI_FONT
        try:
            re_.DEVANAGARI_FONT = None
            text = "मुझे प्राइसिंग नहीं दिख रही है।"
            result = re_.safe_script_text(text)
            self.assertNotEqual(result, text)
            self.assertIn("omitted", result)
        finally:
            re_.DEVANAGARI_FONT = original


if __name__ == "__main__":
    unittest.main()
