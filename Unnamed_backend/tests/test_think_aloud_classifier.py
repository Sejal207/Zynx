"""Tests for Phase 3: Think-Aloud UX Classification.

The LLM is always mocked here -- none of these tests call a real Groq/OpenAI
API. Each test builds a fake `llm` object exposing only `.invoke()`, matching
the interface LLMThinkAloudClassifier actually calls (the same interface
evaluator_agent.py/reporter_agent.py already rely on), so no real network
call is ever made.
"""
import json
import os
import sys
import unittest

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from orchestrator import think_aloud
from orchestrator.think_aloud_classifier import (
    ALLOWED_LABELS, ALLOWED_CONFIDENCE,
    LLMThinkAloudClassifier, classify_result,
)


class _FakeResponse:
    def __init__(self, content, finish_reason=None):
        self.content = content
        if finish_reason is not None:
            self.response_metadata = {"finish_reason": finish_reason}


class _FakeLLM:
    """Mimics the ChatGroq/ChatOpenAI `.invoke([...]) -> response.content`
    interface. `script` is either a fixed string, or a callable(prompt) that
    returns the JSON string to respond with."""
    def __init__(self, script, finish_reason=None):
        self.script = script
        self.finish_reason = finish_reason
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        prompt = messages[0].content
        content = self.script(prompt) if callable(self.script) else self.script
        return _FakeResponse(content, finish_reason=self.finish_reason)


def _seg(display_transcript, segment_id="seg_1"):
    return think_aloud.make_segment(display_transcript)  # segment_id auto-generated, fine for these tests


def _classifier_returning(json_str):
    return LLMThinkAloudClassifier(_FakeLLM(json_str))


class ClassifierLabelTests(unittest.TestCase):
    """TEST 1-8 from the Phase 3 spec, one per required example."""

    def test_expectation_label(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION"], "evidence_text": "Mujhe laga pricing navigation mein hogi.",
            "confidence": "high", "rationale": "States expected location."
        }))
        result = clf.classify_segment(_seg("Mujhe laga pricing navigation mein hogi."))
        self.assertEqual(result["labels"], ["EXPECTATION"])

    def test_confusion_label(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["CONFUSION"], "evidence_text": "Mujhe samajh nahi aa raha yahan kya karna hai.",
            "confidence": "high", "rationale": "States difficulty understanding what to do."
        }))
        result = clf.classify_segment(_seg("Mujhe samajh nahi aa raha yahan kya karna hai."))
        self.assertEqual(result["labels"], ["CONFUSION"])

    def test_navigation_intent_label(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["NAVIGATION_INTENT"], "evidence_text": "Main ab doosre links check karti hoon.",
            "confidence": "medium", "rationale": "States intended next action."
        }))
        result = clf.classify_segment(_seg("Main ab doosre links check karti hoon."))
        self.assertEqual(result["labels"], ["NAVIGATION_INTENT"])

    def test_uncertainty_label(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["UNCERTAINTY"], "evidence_text": "Shayad ye pricing section hai.",
            "confidence": "medium", "rationale": "Uses 'shayad' (maybe), expressing uncertainty."
        }))
        result = clf.classify_segment(_seg("Shayad ye pricing section hai."))
        self.assertEqual(result["labels"], ["UNCERTAINTY"])

    def test_confirmation_label(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["CONFIRMATION"], "evidence_text": "Haan, yehi pricing information hai.",
            "confidence": "high", "rationale": "Explicitly confirms this is the pricing info."
        }))
        result = clf.classify_segment(_seg("Haan, yehi pricing information hai."))
        self.assertEqual(result["labels"], ["CONFIRMATION"])

    def test_complaint_label(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["COMPLAINT"], "evidence_text": "Pricing clearly visible nahi hai.",
            "confidence": "high", "rationale": "Identifies a visibility usability problem."
        }))
        result = clf.classify_segment(_seg("Pricing clearly visible nahi hai."))
        self.assertEqual(result["labels"], ["COMPLAINT"])

    def test_multi_label_expectation_and_confusion(self):
        text = "Mujhe laga pricing yahan hogi, but mujhe samajh nahi aa raha kis option pe click karna hai."
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION", "CONFUSION"], "evidence_text": text,
            "confidence": "medium", "rationale": "States an expectation and explicit difficulty."
        }))
        result = clf.classify_segment(_seg(text))
        self.assertEqual(set(result["labels"]), {"EXPECTATION", "CONFUSION"})

    def test_no_label_supported_returns_empty_list(self):
        clf = _classifier_returning(json.dumps({
            "labels": [], "evidence_text": "Okay.", "confidence": "low", "rationale": "No explicit UX content."
        }))
        result = clf.classify_segment(_seg("Okay."))
        self.assertEqual(result["labels"], [])


class ComplaintCalibrationTests(unittest.TestCase):
    """Phase 3 calibration: explicit usability-problem/negative-visibility
    statements ("X nahi dikh rahi", "X available nahi hai", etc.) must be
    reliably eligible for COMPLAINT under the broadened label definition.
    These mock the LLM response with the exact outputs independently
    verified against the real configured Groq model for each case."""

    def test_pricing_not_clearly_visible_is_complaint(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["COMPLAINT"], "evidence_text": "Pricing clearly visible nahi hai.",
            "confidence": "high", "rationale": "States pricing is not clearly visible on the interface."
        }))
        result = clf.classify_segment(_seg("Pricing clearly visible nahi hai."))
        self.assertEqual(result["labels"], ["COMPLAINT"])

    def test_pricing_not_showing_here_is_complaint(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["COMPLAINT"], "evidence_text": "Yahan mujhe pricing nahi dikh rahi.",
            "confidence": "high", "rationale": "States pricing is not visible here."
        }))
        result = clf.classify_segment(_seg("Yahan mujhe pricing nahi dikh rahi."))
        self.assertEqual(result["labels"], ["COMPLAINT"])

    def test_information_not_available_is_complaint(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["COMPLAINT"], "evidence_text": "Information yahan available nahi hai.",
            "confidence": "high", "rationale": "States information is not available here."
        }))
        result = clf.classify_segment(_seg("Information yahan available nahi hai."))
        self.assertEqual(result["labels"], ["COMPLAINT"])

    def test_navigation_intent_alone_is_not_automatically_complaint(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["NAVIGATION_INTENT"], "evidence_text": "Main ab doosare links check karti hoon.",
            "confidence": "high", "rationale": "States an intended next navigation, no problem stated."
        }))
        result = clf.classify_segment(_seg("Main ab doosare links check karti hoon."))
        self.assertEqual(result["labels"], ["NAVIGATION_INTENT"])
        self.assertNotIn("COMPLAINT", result["labels"])

    def test_uncertainty_label_still_correct_after_calibration(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["UNCERTAINTY"], "evidence_text": "Shayad ye pricing section hai.",
            "confidence": "medium", "rationale": "Uses 'shayad' (maybe), expressing uncertainty."
        }))
        result = clf.classify_segment(_seg("Shayad ye pricing section hai."))
        self.assertEqual(result["labels"], ["UNCERTAINTY"])

    def test_expectation_label_still_correct_after_calibration(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION"], "evidence_text": "Mujhe laga pricing navigation mein hogi.",
            "confidence": "high", "rationale": "States expected location for pricing."
        }))
        result = clf.classify_segment(_seg("Mujhe laga pricing navigation mein hogi."))
        self.assertEqual(result["labels"], ["EXPECTATION"])

    def test_confirmation_label_still_correct_after_calibration(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["CONFIRMATION"], "evidence_text": "Haan, yehi pricing information hai.",
            "confidence": "high", "rationale": "Explicitly confirms this is the pricing info."
        }))
        result = clf.classify_segment(_seg("Haan, yehi pricing information hai."))
        self.assertEqual(result["labels"], ["CONFIRMATION"])

    def test_combined_real_example_supports_all_explicit_signals(self):
        """The exact real transcript reported: four/five distinct explicit
        signals across separate clauses in one segment, all independently
        supported and none forced."""
        text = ("Mujhe pricing section dhoondhna hai. Mujhe laga tha pricing navigation mein hogi, "
                "but yahan mujhe pricing nahi dikh rahi. Shayad mujhe doosare links check karne chahiye. "
                "Haan, mujhe relevant information mil gayee.")
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION", "COMPLAINT", "NAVIGATION_INTENT", "UNCERTAINTY", "CONFIRMATION"],
            "evidence_text": text, "confidence": "high",
            "rationale": "Multiple independent clauses each explicitly support a different label."
        }))
        result = clf.classify_segment(_seg(text))
        self.assertEqual(
            set(result["labels"]),
            {"EXPECTATION", "COMPLAINT", "NAVIGATION_INTENT", "UNCERTAINTY", "CONFIRMATION"},
        )

    def test_button_not_understandable_is_confusion_and_complaint(self):
        """Matches the spec's own worked example: naming a specific broken/
        unclear element is both CONFUSION (difficulty understanding) and
        COMPLAINT (identifies the element as the problem)."""
        clf = _classifier_returning(json.dumps({
            "labels": ["CONFUSION", "COMPLAINT"], "evidence_text": "Button samajh nahi aa raha.",
            "confidence": "medium", "rationale": "States difficulty understanding the button and names it as the problem."
        }))
        result = clf.classify_segment(_seg("Button samajh nahi aa raha."))
        self.assertEqual(set(result["labels"]), {"CONFUSION", "COMPLAINT"})


class ClassifierValidationTests(unittest.TestCase):
    """TEST 9-11: invalid label, invalid confidence, malformed JSON."""

    def test_invalid_label_is_rejected_not_accepted(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["FRUSTRATION"], "evidence_text": "some text", "confidence": "high", "rationale": "x"
        }))
        result = clf.classify_segment(_seg("some text"))
        self.assertNotIn("FRUSTRATION", result["labels"])
        self.assertEqual(result["labels"], [])  # the only label given was invalid -> filtered to empty

    def test_invalid_label_mixed_with_valid_keeps_only_valid(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION", "FRUSTRATION", "STRESS"],
            "evidence_text": "some text", "confidence": "high", "rationale": "x"
        }))
        result = clf.classify_segment(_seg("some text"))
        self.assertEqual(result["labels"], ["EXPECTATION"])

    def test_invalid_confidence_falls_back_safely(self):
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION"], "evidence_text": "some text",
            "confidence": "very_high_extreme", "rationale": "x"
        }))
        result = clf.classify_segment(_seg("some text"))
        self.assertIn(result["confidence"], ALLOWED_CONFIDENCE)
        self.assertEqual(result["confidence"], "low")

    def test_malformed_json_falls_back_without_crashing(self):
        clf = _classifier_returning("this is not { valid json at all")
        result = clf.classify_segment(_seg("Mujhe laga pricing yahan hogi."))
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["confidence"], "low")
        self.assertIn("evidence_text", result)

    def test_missing_labels_key_falls_back_safely(self):
        clf = _classifier_returning(json.dumps({"confidence": "high"}))
        result = clf.classify_segment(_seg("some text"))
        self.assertEqual(result["labels"], [])

    def test_llm_returning_non_dict_json_falls_back_safely(self):
        clf = _classifier_returning(json.dumps(["not", "a", "dict"]))
        result = clf.classify_segment(_seg("some text"))
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["confidence"], "low")

    def test_no_llm_configured_falls_back_safely_without_crashing(self):
        clf = LLMThinkAloudClassifier(llm=None)
        result = clf.classify_segment(_seg("Mujhe laga pricing yahan hogi."))
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["confidence"], "low")

    def test_truncated_response_finish_reason_length_falls_back_safely(self):
        """A response cut off by max_tokens (finish_reason="length") is a
        genuinely different failure mode from malformed JSON -- it must
        still fall back safely without crashing, and should be identifiable
        as a truncation rather than a generic parse failure."""
        truncated_json = '{"labels": ["EXPECTATION"], "evidence_text": "Mujhe laga pricing na'
        clf = LLMThinkAloudClassifier(_FakeLLM(truncated_json, finish_reason="length"))
        result = clf.classify_segment(_seg("Mujhe laga pricing navigation mein hogi."))
        self.assertEqual(result["labels"], [])
        self.assertEqual(result["confidence"], "low")
        self.assertIn("truncated", result["rationale"].lower())

    def test_empty_transcript_segment_short_circuits_without_calling_llm(self):
        fake_llm = _FakeLLM("{}")
        clf = LLMThinkAloudClassifier(fake_llm)
        result = clf.classify_segment(_seg(""))
        self.assertEqual(result["labels"], [])
        self.assertEqual(fake_llm.calls, 0)


class SegmentFieldPreservationTests(unittest.TestCase):
    """TEST 12-14: classification must never alter existing segment fields."""

    def test_raw_transcript_unchanged_after_classification(self):
        raw = "मुझे लगा प्राइसिंग नेविगेशन में होगी।"
        seg = think_aloud.make_segment(raw, start_time_ms=0.0, end_time_ms=4000.0, language="hi-IN")
        original_raw = seg["raw_transcript"]
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION"], "evidence_text": seg["display_transcript"],
            "confidence": "high", "rationale": "x"
        }))
        seg["classification"] = clf.classify_segment(seg)
        self.assertEqual(seg["raw_transcript"], original_raw)
        self.assertEqual(seg["raw_transcript"], raw)

    def test_display_transcript_unchanged_after_classification(self):
        seg = think_aloud.make_segment("Mujhe laga pricing yahan hogi.")
        original_display = seg["display_transcript"]
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION"], "evidence_text": original_display, "confidence": "high", "rationale": "x"
        }))
        seg["classification"] = clf.classify_segment(seg)
        self.assertEqual(seg["display_transcript"], original_display)

    def test_timestamps_unchanged_after_classification(self):
        seg = think_aloud.make_segment("Mujhe laga pricing yahan hogi.", start_time_ms=2000.0, end_time_ms=6000.0)
        clf = _classifier_returning(json.dumps({
            "labels": [], "evidence_text": "x", "confidence": "low", "rationale": "x"
        }))
        seg["classification"] = clf.classify_segment(seg)
        self.assertEqual(seg["start_time_ms"], 2000.0)
        self.assertEqual(seg["end_time_ms"], 6000.0)
        self.assertEqual(seg["duration_ms"], 4000.0)


class ClassifyResultIntegrationTests(unittest.TestCase):
    """classify_result() attaches per-segment classification onto a full
    ThinkAloudResult without disturbing anything else, and only runs when
    the transcript itself is actually available."""

    def test_attaches_classification_to_each_segment(self):
        result = think_aloud.build_result_from_sarvam_response(
            {"transcript": "Mujhe laga pricing yahan hogi.", "language_code": "en-IN"},
            "sess1", recording_start_ms=0.0, audio_duration_ms=3000.0,
        )
        clf = _classifier_returning(json.dumps({
            "labels": ["EXPECTATION"], "evidence_text": "Mujhe laga pricing yahan hogi.",
            "confidence": "high", "rationale": "x"
        }))
        classified = classify_result(result, clf)
        self.assertEqual(classified["segments"][0]["classification"]["labels"], ["EXPECTATION"])
        # nothing else on the result was touched
        self.assertEqual(classified["language"], "en-IN")
        self.assertEqual(classified["timing_status"], think_aloud.TIMING_UNTIMED)

    def test_does_not_classify_when_transcription_failed(self):
        failed_result = think_aloud.make_result(
            "sess2", [], None, None, think_aloud.TIMING_INVALID, available=False, error="boom",
        )
        fake_llm = _FakeLLM("{}")
        clf = LLMThinkAloudClassifier(fake_llm)
        classify_result(failed_result, clf)
        self.assertEqual(fake_llm.calls, 0)

    def test_classifier_none_is_a_safe_noop(self):
        result = think_aloud.build_result_from_sarvam_response(
            {"transcript": "hello", "language_code": "en-IN"}, "sess3", 0.0, 1000.0,
        )
        returned = classify_result(result, None)
        self.assertNotIn("classification", returned["segments"][0])


if __name__ == "__main__":
    unittest.main()
