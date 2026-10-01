"""Tests for the Human + Think-Aloud UX Evaluation Report (deterministic
Task UX Score, signal summary, evidence timeline, findings,
expectation-vs-experience). The LLM (used only for the optional prose
summary, never for the score) is always mocked -- no real Groq/OpenAI call.
"""
import os
import sys
import unittest

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from orchestrator import think_aloud, human_ux_report as hur


def _seg(text, labels=None, start_ms=None, end_ms=None):
    seg = think_aloud.make_segment(text, start_time_ms=start_ms, end_time_ms=end_ms)
    if labels is not None:
        seg["classification"] = {
            "segment_id": seg["segment_id"], "labels": labels,
            "evidence_text": seg["display_transcript"], "confidence": "high", "rationale": "x",
        }
    return seg


class TaskCompletionScoreTests(unittest.TestCase):
    def test_completed_gives_full_40(self):
        self.assertEqual(hur.calculate_task_completion_score("completed"), 40)

    def test_partially_completed_gives_20(self):
        self.assertEqual(hur.calculate_task_completion_score("partially_completed"), 20)

    def test_not_completed_gives_zero(self):
        self.assertEqual(hur.calculate_task_completion_score("not_completed"), 0)

    def test_not_assessed_gives_none_not_a_guessed_score(self):
        self.assertIsNone(hur.calculate_task_completion_score("not_assessed"))


class VerbalSignalsScoreTests(unittest.TestCase):
    def test_confusion_deducts_configured_amount(self):
        segs = [_seg("a", labels=["CONFUSION"])]
        result = hur.calculate_verbal_signals_score(segs)
        self.assertEqual(result["score"], hur.VERBAL_SIGNALS_MAX - 2)

    def test_uncertainty_deducts_configured_amount(self):
        segs = [_seg("a", labels=["UNCERTAINTY"])]
        result = hur.calculate_verbal_signals_score(segs)
        self.assertEqual(result["score"], hur.VERBAL_SIGNALS_MAX - 1)

    def test_complaint_deducts_configured_amount(self):
        segs = [_seg("a", labels=["COMPLAINT"])]
        result = hur.calculate_verbal_signals_score(segs)
        self.assertEqual(result["score"], hur.VERBAL_SIGNALS_MAX - 2)

    def test_expectation_causes_no_deduction(self):
        segs = [_seg("a", labels=["EXPECTATION"])]
        result = hur.calculate_verbal_signals_score(segs)
        self.assertEqual(result["score"], hur.VERBAL_SIGNALS_MAX)

    def test_navigation_intent_causes_no_deduction(self):
        segs = [_seg("a", labels=["NAVIGATION_INTENT"])]
        result = hur.calculate_verbal_signals_score(segs)
        self.assertEqual(result["score"], hur.VERBAL_SIGNALS_MAX)

    def test_confirmation_causes_no_deduction(self):
        segs = [_seg("a", labels=["CONFIRMATION"])]
        result = hur.calculate_verbal_signals_score(segs)
        self.assertEqual(result["score"], hur.VERBAL_SIGNALS_MAX)

    def test_multiple_labels_in_same_segment_count_once_per_label(self):
        # CONFUSION + COMPLAINT in ONE segment -> both deductions apply once each,
        # not doubled, and not skipped.
        segs = [_seg("Button samajh nahi aa raha.", labels=["CONFUSION", "COMPLAINT"])]
        result = hur.calculate_verbal_signals_score(segs)
        self.assertEqual(result["score"], hur.VERBAL_SIGNALS_MAX - 2 - 2)
        self.assertEqual(len(result["evidence_segment_ids"]["CONFUSION"]), 1)
        self.assertEqual(len(result["evidence_segment_ids"]["COMPLAINT"]), 1)

    def test_score_cannot_go_below_zero(self):
        segs = [_seg(f"seg{i}", labels=["CONFUSION", "COMPLAINT"]) for i in range(20)]
        result = hur.calculate_verbal_signals_score(segs)
        self.assertEqual(result["score"], 0)

    def test_score_operates_on_evidence_segments_not_source_chunks(self):
        """Documents the Part 6 design decision (see the comment above
        calculate_verbal_signals_score): the score operates on whatever
        `segments` list it receives -- i.e. on the finer EVIDENCE segments
        produced by think_aloud.py's sentence splitting, not on Sarvam's
        original coarse source chunk. Two distinct COMPLAINT sentences that
        happened to come from the SAME 24s Sarvam chunk (same shared
        start/end, timing_source="derived") must each deduct separately,
        exactly as if they'd been two genuinely separate Sarvam chunks."""
        coarse_chunk_two_complaints = [
            _seg("Pricing nahi dikh raha.", labels=["COMPLAINT"], start_ms=0, end_ms=24000),
            _seg("Button bhi samajh nahi aa raha.", labels=["COMPLAINT"], start_ms=0, end_ms=24000),
        ]
        for seg in coarse_chunk_two_complaints:
            seg["timing_source"] = "derived"  # both from the same parent chunk
        result = hur.calculate_verbal_signals_score(coarse_chunk_two_complaints)
        self.assertEqual(result["score"], hur.VERBAL_SIGNALS_MAX - 2 - 2,
                          "two distinct evidence segments must each deduct, even sharing one source range")


class RecoveryScoreTests(unittest.TestCase):
    def test_difficulty_then_confirmation_then_completed_gives_full_20(self):
        segs = [
            _seg("problem", labels=["COMPLAINT"], start_ms=0, end_ms=1000),
            _seg("confirm", labels=["CONFIRMATION"], start_ms=2000, end_ms=3000),
        ]
        result = hur.calculate_recovery_score(segs, "completed")
        self.assertEqual(result["score"], 20)

    def test_no_difficulty_signal_gives_15_not_perfect(self):
        """Explicitly NOT 20 -- no evidence of a difficulty episode means
        nothing to recover from, not "perfect recovery"."""
        segs = [_seg("all fine", labels=["EXPECTATION"], start_ms=0, end_ms=1000)]
        result = hur.calculate_recovery_score(segs, "completed")
        self.assertEqual(result["score"], 15)

    def test_difficulty_with_no_confirmation_and_not_completed_gives_zero(self):
        segs = [_seg("problem", labels=["CONFUSION"], start_ms=0, end_ms=1000)]
        result = hur.calculate_recovery_score(segs, "not_completed")
        self.assertEqual(result["score"], 0)

    def test_difficulty_then_confirmation_partially_completed_gives_12(self):
        segs = [
            _seg("problem", labels=["UNCERTAINTY"], start_ms=0, end_ms=1000),
            _seg("confirm", labels=["CONFIRMATION"], start_ms=2000, end_ms=3000),
        ]
        result = hur.calculate_recovery_score(segs, "partially_completed")
        self.assertEqual(result["score"], 12)

    def test_not_assessed_task_outcome_gives_not_fully_assessed(self):
        segs = [_seg("problem", labels=["COMPLAINT"], start_ms=0, end_ms=1000)]
        result = hur.calculate_recovery_score(segs, "not_assessed")
        self.assertIsNone(result["score"])
        self.assertEqual(result["status"], "Not fully assessed")


class EvidenceCompletenessTests(unittest.TestCase):
    def test_full_evidence_gives_15(self):
        segs = [_seg("a", labels=["EXPECTATION"], start_ms=0, end_ms=1000)]
        result = hur.calculate_evidence_completeness_score(segs)
        self.assertEqual(result["score"], 15)

    def test_incomplete_timing_gives_10(self):
        segs = [_seg("a", labels=["EXPECTATION"])]  # no timestamps
        result = hur.calculate_evidence_completeness_score(segs)
        self.assertEqual(result["score"], 10)

    def test_no_classification_gives_5(self):
        segs = [think_aloud.make_segment("a")]  # no .classification attached at all
        result = hur.calculate_evidence_completeness_score(segs)
        self.assertEqual(result["score"], 5)

    def test_no_transcript_gives_zero(self):
        result = hur.calculate_evidence_completeness_score([])
        self.assertEqual(result["score"], 0)

    def test_deterministic_repeat_calls_match(self):
        segs = [_seg("a", labels=["EXPECTATION"], start_ms=0, end_ms=1000)]
        r1 = hur.calculate_evidence_completeness_score(segs)
        r2 = hur.calculate_evidence_completeness_score(segs)
        self.assertEqual(r1, r2)


class OverallScoreTests(unittest.TestCase):
    def test_not_assessed_never_produces_a_misleading_total(self):
        segs = [_seg("a", labels=["EXPECTATION"], start_ms=0, end_ms=1000)]
        score = hur.calculate_task_ux_score(segs, "not_assessed")
        self.assertIsNone(score["total"])
        self.assertFalse(score["fully_assessed"])
        self.assertEqual(score["status"], "Not fully assessed")
        # component scores that ARE available must still be shown
        self.assertIsNotNone(score["dimensions"]["verbal_ux_signals"]["score"])
        self.assertIsNotNone(score["dimensions"]["evidence_completeness"]["score"])

    def test_fully_assessed_totals_sum_correctly(self):
        segs = [
            _seg("expect", labels=["EXPECTATION"], start_ms=0, end_ms=1000),
            _seg("complaint", labels=["COMPLAINT"], start_ms=2000, end_ms=3000),
            _seg("confirm", labels=["CONFIRMATION"], start_ms=4000, end_ms=5000),
        ]
        score = hur.calculate_task_ux_score(segs, "completed")
        self.assertTrue(score["fully_assessed"])
        d = score["dimensions"]
        expected_total = d["task_completion"]["score"] + d["verbal_ux_signals"]["score"] \
            + d["recovery"]["score"] + d["evidence_completeness"]["score"]
        self.assertEqual(score["total"], expected_total)
        self.assertLessEqual(score["total"], 100)

    def test_disclaimer_present_and_no_universal_quality_language(self):
        score = hur.calculate_task_ux_score([], "not_assessed")
        self.assertIn("should not be interpreted as a universal website quality score", score["disclaimer"])
        for banned in ("excellent", "poor", "bad website", "good website"):
            self.assertNotIn(banned, score["disclaimer"].lower())
            self.assertNotIn(banned, score["method_note"].lower())


class SignalSummaryTests(unittest.TestCase):
    def test_signal_counts_are_deterministic(self):
        segs = [
            _seg("a", labels=["EXPECTATION"]),
            _seg("b", labels=["CONFUSION"]),
            _seg("c", labels=["NAVIGATION_INTENT", "UNCERTAINTY"]),
            _seg("d", labels=["CONFIRMATION"]),
            _seg("e", labels=["COMPLAINT"]),
        ]
        result = hur.build_signal_summary(segs)
        self.assertEqual(result["EXPECTATION"]["count"], 1)
        self.assertEqual(result["CONFUSION"]["count"], 1)
        self.assertEqual(result["NAVIGATION_INTENT"]["count"], 1)
        self.assertEqual(result["UNCERTAINTY"]["count"], 1)
        self.assertEqual(result["CONFIRMATION"]["count"], 1)
        self.assertEqual(result["COMPLAINT"]["count"], 1)
        # repeat call gives identical result
        self.assertEqual(result, hur.build_signal_summary(segs))


class TimelineTests(unittest.TestCase):
    def test_timeline_preserves_original_timestamps(self):
        segs = [
            _seg("second", labels=["COMPLAINT"], start_ms=5000, end_ms=6000),
            _seg("first", labels=["EXPECTATION"], start_ms=1000, end_ms=2000),
        ]
        timeline = hur.build_evidence_timeline(segs)
        self.assertEqual(timeline[0]["start_time_ms"], 1000)
        self.assertEqual(timeline[1]["start_time_ms"], 5000)


class FindingsTests(unittest.TestCase):
    def test_findings_preserve_original_evidence_text(self):
        text = "Yahan mujhe pricing nahi dikh rahi."
        segs = [_seg(text, labels=["COMPLAINT"], start_ms=0, end_ms=1000)]
        findings = hur.build_findings(segs)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["evidence_text"], text)

    def test_no_finding_without_supporting_labels(self):
        segs = [think_aloud.make_segment("Okay.")]  # no classification at all
        segs[0]["classification"] = {"labels": [], "evidence_text": "Okay.", "confidence": "low", "rationale": "x"}
        findings = hur.build_findings(segs)
        self.assertEqual(findings, [])


class ExpectationExtractionWithFineSegmentsTests(unittest.TestCase):
    """Part 4: with properly split evidence segments (the Part 1 fix), the
    'Expectation vs Experience' section must show only the actual
    EXPECTATION-labelled sentence, never the entire multi-sentence transcript
    it used to show when everything was bundled into one giant segment."""

    def test_expectation_extracts_only_the_expectation_sentence(self):
        segs = [
            _seg("Mujhe pricing section dhoondhna hai.", labels=["NAVIGATION_INTENT"], start_ms=0, end_ms=4000),
            _seg("Mujhe laga tha pricing navigation mein hogi.", labels=["EXPECTATION"], start_ms=4000, end_ms=8000),
            _seg("Mujhe pricing dikh hi nahi rahi hai.", labels=["COMPLAINT"], start_ms=8000, end_ms=12000),
            _seg("Haan mil gayi mujhe relevant information.", labels=["CONFIRMATION"], start_ms=12000, end_ms=16000),
        ]
        result = hur.build_expectation_vs_experience(segs)
        self.assertIsNotNone(result["expectation"])
        self.assertEqual(result["expectation"]["text"], "Mujhe laga tha pricing navigation mein hogi.")
        # must NOT be the full concatenated transcript
        self.assertNotIn("dhoondhna", result["expectation"]["text"])
        self.assertNotIn("relevant information", result["expectation"]["text"])

    def test_observed_experience_uses_complaint_or_confusion_evidence(self):
        segs = [
            _seg("Mujhe laga tha pricing navigation mein hogi.", labels=["EXPECTATION"], start_ms=0, end_ms=1000),
            _seg("Mujhe pricing dikh hi nahi rahi hai.", labels=["COMPLAINT"], start_ms=1000, end_ms=2000),
        ]
        result = hur.build_expectation_vs_experience(segs)
        self.assertEqual(result["observed_experience"]["text"], "Mujhe pricing dikh hi nahi rahi hai.")

    def test_confirmation_slot_uses_confirmation_evidence_only(self):
        segs = [
            _seg("Mujhe laga tha pricing navigation mein hogi.", labels=["EXPECTATION"], start_ms=0, end_ms=1000),
            _seg("Haan mil gayi mujhe relevant information.", labels=["CONFIRMATION"], start_ms=1000, end_ms=2000),
        ]
        result = hur.build_expectation_vs_experience(segs)
        self.assertEqual(result["confirmation"]["text"], "Haan mil gayi mujhe relevant information.")

    def test_this_was_the_actual_reported_bug_a_single_giant_segment(self):
        """Before Part 1's segmentation fix, Sarvam's whole 24s recording
        arrived as ONE segment carrying all six labels -- so 'expectation'
        would show the ENTIRE transcript. This test documents that if that
        happens again upstream, the report at least behaves consistently
        (shows the one available segment) rather than crashing; the real fix
        is in think_aloud.py's segmentation, verified separately."""
        full_text = ("Mujhe pricing section dhoondhna hai. Mujhe laga tha pricing "
                      "navigation mein hogi. Mujhe pricing dikh hi nahi rahi hai.")
        segs = [_seg(full_text, labels=["EXPECTATION", "COMPLAINT", "NAVIGATION_INTENT"], start_ms=0, end_ms=24000)]
        result = hur.build_expectation_vs_experience(segs)
        # With only one giant segment, expectation and observed_experience
        # necessarily point to the SAME segment -- this is the exact symptom
        # that motivated Part 1's fix, reproduced here as a regression anchor.
        self.assertEqual(result["expectation"]["text"], full_text)


class SegmentFieldPreservationTests(unittest.TestCase):
    def test_raw_transcript_unchanged_by_report_generation(self):
        raw = "यह टेस्ट है"
        seg = think_aloud.make_segment(raw, language="hi-IN")
        original_raw = seg["raw_transcript"]
        hur.build_human_ux_report("https://x.test", "task", {}, [seg], None, "hi-IN", "not_assessed")
        self.assertEqual(seg["raw_transcript"], original_raw)

    def test_display_transcript_unchanged_by_report_generation(self):
        seg = think_aloud.make_segment("Mujhe laga pricing yahan hogi.")
        original_display = seg["display_transcript"]
        hur.build_human_ux_report("https://x.test", "task", {}, [seg], None, "en-IN", "not_assessed")
        self.assertEqual(seg["display_transcript"], original_display)


class FullReportIntegrationTests(unittest.TestCase):
    """Uses a fake LLM (never real Groq/OpenAI) for the optional summary
    only -- confirms the summary call never influences the score."""

    class _FakeLLM:
        def invoke(self, messages):
            class R:
                content = "Deterministic-safe mocked summary sentence."
            return R()

    def test_full_report_builds_without_llm(self):
        segs = [
            _seg("Mujhe laga pricing navigation mein hogi.", labels=["EXPECTATION"], start_ms=0, end_ms=1000),
            _seg("Yahan mujhe pricing nahi dikh rahi.", labels=["COMPLAINT"], start_ms=2000, end_ms=3000),
            _seg("Haan, mujhe mil gaya.", labels=["CONFIRMATION"], start_ms=4000, end_ms=5000),
        ]
        report = hur.build_human_ux_report(
            "https://example.com", "Find pricing", {"persona_type": "novice"},
            segs, 6000.0, "en-IN", "completed", llm=None,
        )
        self.assertEqual(report["task_outcome"], "completed")
        self.assertTrue(report["score"]["fully_assessed"])
        self.assertIsInstance(report["summary"], str)
        self.assertTrue(len(report["summary"]) > 0)

    def test_full_report_with_mocked_llm_summary(self):
        segs = [_seg("Mujhe laga pricing navigation mein hogi.", labels=["EXPECTATION"], start_ms=0, end_ms=1000)]
        report = hur.build_human_ux_report(
            "https://example.com", "Find pricing", {}, segs, 1000.0, "en-IN",
            "not_assessed", llm=self._FakeLLM(),
        )
        self.assertEqual(report["summary"], "Deterministic-safe mocked summary sentence.")

    def test_report_does_not_use_phase1_friction_fields(self):
        """No Phase 1 field names appear anywhere in a Human report's output."""
        segs = [_seg("test", labels=["COMPLAINT"], start_ms=0, end_ms=1000)]
        report = hur.build_human_ux_report("https://x.test", "t", {}, segs, 1000.0, "en-IN", "completed")
        report_str = str(report)
        for forbidden in ("repeated_click", "dead_click", "excessive_scroll",
                          "navigation_loop", "repeated_action_loop", "friction_episode"):
            self.assertNotIn(forbidden, report_str)

    def test_backward_compatibility_missing_classification_does_not_crash(self):
        """An older-shaped segment with no .classification key at all must
        not crash report generation."""
        seg = think_aloud.make_segment("Some old transcript with no classification.")
        report = hur.build_human_ux_report("https://x.test", "t", {}, [seg], 1000.0, "en-IN", "not_assessed")
        self.assertEqual(report["findings"], [])

    def test_empty_segments_list_does_not_crash(self):
        report = hur.build_human_ux_report("https://x.test", "t", {}, [], None, None, "not_assessed")
        self.assertEqual(report["score"]["dimensions"]["evidence_completeness"]["score"], 0)


if __name__ == "__main__":
    unittest.main()
