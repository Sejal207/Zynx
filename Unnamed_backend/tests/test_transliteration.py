"""Tests for the Devanagari -> casual Roman script conversion (script
conversion only, never translation) and its integration into
ThinkAloudSegment's raw_transcript/display_transcript fields.
"""
import os
import sys
import unittest

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from orchestrator.transliteration import romanize
from orchestrator import think_aloud


class RomanizeUnitTests(unittest.TestCase):

    def test_english_transcript_remains_unchanged(self):
        text = "I expected the quizzes section to be under Participate."
        self.assertEqual(romanize(text), text)

    def test_already_romanized_hinglish_remains_unchanged(self):
        text = "Mujhe laga quizzes Participate ke andar honge, but yahan nahi mil raha."
        self.assertEqual(romanize(text), text)

    def test_regression_mixed_script_candra_o_no_longer_leaks_devanagari(self):
        """Exact real-run bug report: 'information' respelled phonetically in
        Devanagari (using the candra-O vowel sign U+0949, e.g. फॉर्म-style
        loanwords) previously left the ॉ character untouched mid-word,
        producing mixed-script output like 'inphaॉrmeshana'. No Devanagari
        codepoint may appear anywhere in the output."""
        variants = ["इन्फॉर्मेशन", "इन्फ़ॉर्मेशन", "इनफॉर्मेशन"]
        for raw in variants:
            result = romanize(raw)
            devanagari_chars = [c for c in result if "ऀ" <= c <= "ॿ"]
            self.assertEqual(devanagari_chars, [], f"leaked Devanagari chars in {result!r} from {raw!r}")

    def test_information_resolves_to_clean_english_word(self):
        for raw in ["इन्फॉर्मेशन", "इन्फ़ॉर्मेशन", "इनफॉर्मेशन", "इंफॉर्मेशन"]:
            self.assertEqual(romanize(raw), "Information")

    def test_candra_o_and_candra_e_matras_are_mapped(self):
        # Standalone sanity check on the underlying character gap, independent
        # of the ENGLISH_LOANWORDS dictionary entry for "information".
        result = romanize("डॉक्टर")  # "doctor", uses candra-O matra
        devanagari_chars = [c for c in result if "ऀ" <= c <= "ॿ"]
        self.assertEqual(devanagari_chars, [])

    def test_decomposed_nukta_form_matches_precomposed_form(self):
        """फ़ (precomposed nukta consonant) and फ + combining nukta (U+093C,
        decomposed form) must romanize to the same sound."""
        precomposed = romanize("फ़ॉर्म")   # form, precomposed फ़
        decomposed = romanize("फ़ॉर्म")  # form, फ + combining nukta
        self.assertEqual(precomposed, decomposed)

    def test_full_pricing_sentence_with_information_word(self):
        """The specific sentence pattern from the real bug report."""
        raw = "मुझे relevant इन्फॉर्मेशन नहीं मिल रही"
        result = romanize(raw)
        self.assertIn("information", result.lower())
        devanagari_chars = [c for c in result if "ऀ" <= c <= "ॿ"]
        self.assertEqual(devanagari_chars, [])

    def test_hindi_devanagari_becomes_romanized(self):
        raw = "मुझे तो यहीं पे प्राइसिंग सेक्शन ढूंढना था।"
        self.assertEqual(romanize(raw), "Mujhe to yahi pe pricing section dhoondhna tha.")

    def test_second_provided_example_matches_exactly(self):
        raw = "मुझे यहाँ प्राइसिंग नहीं दिख रही है"
        self.assertEqual(romanize(raw), "Mujhe yahan pricing nahi dikh rahi hai")

    def test_mixed_hindi_and_english_keeps_english_words_as_english(self):
        raw = "यह वेबसाइट बहुत अच्छी है"
        result = romanize(raw)
        self.assertIn("website", result.lower())
        self.assertNotIn("वेबसाइट", result)

    def test_script_conversion_not_translation(self):
        """Meaning must be preserved untranslated -- "pricing" stays
        "pricing", not rendered as an English translation of a Hindi word."""
        raw = "प्राइसिंग सेक्शन"
        result = romanize(raw)
        self.assertEqual(result.lower(), "pricing section")

    def test_unrecognized_characters_are_preserved_not_invented(self):
        # An emoji / unrelated symbol embedded in Devanagari text must survive
        # untouched rather than being dropped or guessed at.
        raw = "नमस्ते 🙂 दुनिया"
        result = romanize(raw)
        self.assertIn("🙂", result)

    def test_empty_and_none_handled_safely(self):
        self.assertEqual(romanize(""), "")
        self.assertIsNone(romanize(None))

    def test_pure_digits_and_punctuation_unchanged(self):
        self.assertEqual(romanize("12345 !@#$%"), "12345 !@#$%")


class SegmentRawAndDisplayTests(unittest.TestCase):
    """Confirms make_segment keeps raw_transcript untouched while adding a
    separate, Romanized display_transcript."""

    def test_raw_transcript_is_never_altered(self):
        raw = "मुझे यहाँ प्राइसिंग नहीं दिख रही है"
        seg = think_aloud.make_segment(raw, language="hi-IN")
        self.assertEqual(seg["raw_transcript"], raw)

    def test_display_transcript_is_separate_and_romanized(self):
        raw = "मुझे यहाँ प्राइसिंग नहीं दिख रही है"
        seg = think_aloud.make_segment(raw, language="hi-IN")
        self.assertEqual(seg["display_transcript"], "Mujhe yahan pricing nahi dikh rahi hai")
        self.assertNotEqual(seg["display_transcript"], seg["raw_transcript"])

    def test_transcript_alias_matches_display_transcript(self):
        raw = "नहीं"
        seg = think_aloud.make_segment(raw)
        self.assertEqual(seg["transcript"], seg["display_transcript"])

    def test_english_segment_raw_and_display_are_identical(self):
        raw = "I am looking for the quizzes section."
        seg = think_aloud.make_segment(raw, language="en-IN")
        self.assertEqual(seg["raw_transcript"], raw)
        self.assertEqual(seg["display_transcript"], raw)

    def test_timestamps_unchanged_by_romanization(self):
        raw = "मुझे यहाँ प्राइसिंग नहीं दिख रही है"
        seg = think_aloud.make_segment(raw, start_time_ms=4000.0, end_time_ms=8000.0)
        self.assertEqual(seg["start_time_ms"], 4000.0)
        self.assertEqual(seg["end_time_ms"], 8000.0)
        self.assertEqual(seg["duration_ms"], 4000.0)

    def test_language_metadata_unchanged_by_romanization(self):
        seg = think_aloud.make_segment("मुझे यहाँ है", language="hi-IN", confidence=0.91)
        self.assertEqual(seg["language"], "hi-IN")
        self.assertEqual(seg["confidence"], 0.91)

    def test_missing_or_empty_transcript_handled_safely(self):
        seg = think_aloud.make_segment("")
        self.assertEqual(seg["raw_transcript"], "")
        self.assertEqual(seg["display_transcript"], "")
        self.assertEqual(seg["transcript"], "")


class FullPipelineRomanizationTests(unittest.TestCase):
    """End-to-end through build_result_from_sarvam_response, matching how a
    real Sarvam response actually arrives."""

    def test_sarvam_response_with_devanagari_is_romanized_for_display(self):
        raw_response = {
            "transcript": "मुझे यहाँ प्राइसिंग नहीं दिख रही है",
            "language_code": "hi-IN",
            "language_probability": 0.93,
        }
        result = think_aloud.build_result_from_sarvam_response(
            raw_response, "sess1", recording_start_ms=0.0, audio_duration_ms=5000.0,
        )
        self.assertTrue(result["available"])
        seg = result["segments"][0]
        self.assertEqual(seg["raw_transcript"], raw_response["transcript"])
        self.assertEqual(seg["display_transcript"], "Mujhe yahan pricing nahi dikh rahi hai")
        # language metadata and timing_status must be unaffected by romanization
        self.assertEqual(result["language"], "hi-IN")

    def test_sarvam_response_with_timed_devanagari_segments(self):
        raw_response = {
            "transcript": "ignored when per-chunk timestamps exist",
            "language_code": "hi-IN",
            "timestamps": {
                "words": ["मुझे यहाँ", "प्राइसिंग नहीं दिख रही है"],
                "start_time_seconds": [0.0, 3.0],
                "end_time_seconds": [3.0, 7.0],
            },
        }
        result = think_aloud.build_result_from_sarvam_response(
            raw_response, "sess2", recording_start_ms=1000.0, audio_duration_ms=7000.0,
        )
        self.assertEqual(result["timing_status"], think_aloud.TIMING_TIMED)
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["raw_transcript"], "मुझे यहाँ")
        self.assertEqual(result["segments"][0]["display_transcript"], "Mujhe yahan")
        self.assertEqual(result["segments"][0]["start_time_ms"], 1000.0)  # unchanged by romanization
        self.assertEqual(result["segments"][1]["display_transcript"], "Pricing nahi dikh rahi hai")


class BaselineReportRegressionTests(unittest.TestCase):
    """Regression tests built from the ACTUAL raw_transcript strings found in
    a real saved baseline report (output/evaluations/think_aloud/, GitHub
    pricing-plans session) that exposed unreadable character-level-fallback
    respellings -- e.g. "gitahab", "phichar", "kanpeyar". Each test pins the
    exact input string and the now-correct output, so a future dictionary
    change can't silently regress these specific words."""

    def test_github_is_not_phonetically_mangled(self):
        raw = "सो मुझे गिटहब के प्राइसिंग प्लांस देखने थे।"
        out = romanize(raw)
        self.assertIn("GitHub", out)
        self.assertNotIn("gitahab", out.lower())

    def test_option_and_main_navigation_are_readable(self):
        raw = "मुझे लग रहा था कि प्राइसिंग का ऑप्शन मेन नेविगेशन में मिल जाएगा बट यहां तो नहीं है।"
        out = romanize(raw)
        self.assertIn("option", out.lower())
        self.assertIn("main navigation", out.lower())

    def test_differences_is_not_phonetically_mangled(self):
        raw = "अभी मेन प्लांस डिफरेंसेस समझने की कोशिश कर रही हूं।"
        out = romanize(raw)
        self.assertIn("differences", out.lower())
        self.assertNotIn("dipharenses", out.lower())

    def test_feature_and_clear_are_readable(self):
        raw = "मुझे क्लियर नहीं है कि इस फीचर के लिए पेड प्लान चाहिए या फ्री प्लान में भी मिल जाए।"
        out = romanize(raw)
        self.assertIn("clear", out.lower())
        self.assertNotIn("kliyar", out.lower())
        self.assertIn("feature", out.lower())
        self.assertNotIn("phichar", out.lower())
        self.assertIn("free", out.lower())
        self.assertNotIn("phri", out.lower())

    def test_relevant_spelling_variant_is_covered(self):
        """The dictionary previously only covered the "रिलेवेंट" spelling;
        the real baseline used the "रेलेवेंट" variant and fell through to
        the phonetic fallback, which rendered it as "relevent" (typo-like,
        not the real spelling)."""
        raw = "अच्छा मुझे रेलेवेंट प्राइसिंग इंफॉर्मेशन मिले।"
        out = romanize(raw)
        self.assertIn("relevant", out.lower())
        self.assertIn("information", out.lower())

    def test_compare_is_not_phonetically_mangled(self):
        raw = "मेन प्लांस कंपेयर मैं प्लांस कंपेयर कर सकती हूं।"
        out = romanize(raw)
        self.assertIn("compare", out.lower())
        self.assertNotIn("kanpeyar", out.lower())

    def test_paid_tree_ambiguity_left_unresolved_not_fabricated(self):
        """Documented known limitation: "पेड" is genuinely ambiguous
        (paid/tree) and must NOT be silently resolved to either meaning."""
        out = romanize("पेड प्लान")
        self.assertNotIn("paid", out.lower())  # never confidently fabricated
        self.assertNotIn("tree", out.lower())  # nor the other reading

    def test_english_technical_tokens_pass_through_untouched_mid_sentence(self):
        """Protects pure-English tokens embedded in a Devanagari sentence
        from any destructive character-level transliteration -- they are
        never Devanagari in the first place, so romanize() must leave them
        byte-for-byte unchanged."""
        raw = "मुझे GitHub pricing page पर जाना है।"
        out = romanize(raw)
        self.assertIn("GitHub pricing page", out)

    def test_documentation_search_button_scroll_confirmation_confusion(self):
        """Additional required-term coverage beyond the GitHub-session
        examples: documentation, search, button, scroll, confusion,
        confirmation."""
        cases = {
            "मुझे डॉक्यूमेंटेशन चाहिए।": "documentation",
            "मैंने सर्च बार में टाइप किया।": "search",
            "मैंने बटन दबाया।": "button",
            "मुझे स्क्रॉल करना पड़ा।": "scroll",
            "मुझे कन्फ्यूजन हो रहा है।": "confusion",
            "यह कन्फर्मेशन मिल गया।": "confirmation",
        }
        for raw, expected_word in cases.items():
            with self.subTest(raw=raw):
                out = romanize(raw).lower()
                self.assertIn(expected_word, out)

    def test_raw_transcript_is_never_mutated_by_the_dictionary_additions(self):
        """Confirms the new dictionary entries only affect display_transcript
        -- raw_transcript remains the exact Sarvam-returned string."""
        raw = "सो मुझे गिटहब के प्राइसिंग प्लांस देखने थे।"
        seg = think_aloud.make_segment(raw)
        self.assertEqual(seg["raw_transcript"], raw)
        self.assertIn("GitHub", seg["display_transcript"])

    def test_display_transcript_has_no_unexpected_devanagari_for_these_sentences(self):
        """For sentences fully covered by the dictionaries, the Romanized
        output should contain no leftover Devanagari characters at all."""
        import re
        from orchestrator.transliteration import DEVANAGARI_RANGE
        raw_sentences = [
            "सो मुझे गिटहब के प्राइसिंग प्लांस देखने थे।",
            "अभी मेन प्लांस डिफरेंसेस समझने की कोशिश कर रही हूं।",
            "अच्छा मुझे रेलेवेंट प्राइसिंग इंफॉर्मेशन मिले।",
        ]
        for raw in raw_sentences:
            with self.subTest(raw=raw):
                out = romanize(raw)
                self.assertFalse(DEVANAGARI_RANGE.search(out), f"unexpected Devanagari left in: {out!r}")


if __name__ == "__main__":
    unittest.main()
