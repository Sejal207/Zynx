"""Tests for Phase 2: Human Think-Aloud data model, Sarvam STT integration,
and the /api/think-aloud/* endpoints.

Backend-only: browser microphone permission handling, MediaRecorder state,
and the "record again" UI flow live in Unnamed_frontend/src/App.jsx and are
not exercised by these Python tests (no JS test runner exists in this
project) -- they are covered by code review and are called out explicitly
in the final report rather than silently assumed to be tested.
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from orchestrator import think_aloud
from tools import sarvam_stt


class ThinkAloudSegmentTests(unittest.TestCase):
    """Items 1-6, 8-9 of the Phase 2 test plan: segment creation & fields."""

    def test_segment_creation_basic_fields(self):
        seg = think_aloud.make_segment("hello world")
        self.assertTrue(seg["segment_id"].startswith("seg_"))
        self.assertEqual(seg["transcript"], "hello world")
        self.assertEqual(seg["source"], "think_aloud")

    def test_timestamp_preservation(self):
        seg = think_aloud.make_segment("hi", start_time_ms=1000.0, end_time_ms=3500.0)
        self.assertEqual(seg["start_time_ms"], 1000.0)
        self.assertEqual(seg["end_time_ms"], 3500.0)

    def test_duration_calculation(self):
        seg = think_aloud.make_segment("hi", start_time_ms=1000.0, end_time_ms=3500.0)
        self.assertEqual(seg["duration_ms"], 2500.0)

    def test_untimed_segment_has_no_fabricated_timing(self):
        seg = think_aloud.make_segment("hi")  # no timestamps given
        self.assertIsNone(seg["start_time_ms"])
        self.assertIsNone(seg["end_time_ms"])
        self.assertIsNone(seg["duration_ms"])

    def test_language_metadata_preservation(self):
        seg = think_aloud.make_segment("namaste", language="hi-IN")
        self.assertEqual(seg["language"], "hi-IN")

    def test_confidence_preservation(self):
        seg = think_aloud.make_segment("hi", confidence=0.87)
        self.assertEqual(seg["confidence"], 0.87)

    def test_empty_transcript_segment_still_constructible_but_flagged_upstream(self):
        # make_segment itself doesn't reject empty text -- the *result*
        # builder (build_result_from_sarvam_response) is what decides
        # availability; tested below.
        seg = think_aloud.make_segment("")
        self.assertEqual(seg["transcript"], "")


class SarvamResponseParsingTests(unittest.TestCase):
    """Items 3, 7, 8, 9: untimed handling, malformed response, missing
    timestamps, empty transcript -- all via the real parsing function."""

    def test_full_response_with_timestamps_produces_timed_segments(self):
        raw = {
            "transcript": "I expected the quizzes section to be under Participate.",
            "language_code": "en-IN",
            "language_probability": 0.95,
            "timestamps": {
                "words": ["I expected the quizzes section", "to be under Participate."],
                "start_time_seconds": [0.0, 2.5],
                "end_time_seconds": [2.5, 5.0],
            },
        }
        result = think_aloud.build_result_from_sarvam_response(raw, "sess1", recording_start_ms=1000.0, audio_duration_ms=5000.0)
        self.assertTrue(result["available"])
        self.assertEqual(result["timing_status"], think_aloud.TIMING_TIMED)
        self.assertEqual(len(result["segments"]), 2)
        # timestamps must be offset by the recording's session-relative start
        self.assertEqual(result["segments"][0]["start_time_ms"], 1000.0)
        self.assertEqual(result["segments"][0]["end_time_ms"], 3500.0)
        self.assertEqual(result["language"], "en-IN")

    def test_response_without_timestamps_is_marked_untimed_not_fabricated(self):
        raw = {"transcript": "Mujhe laga quizzes yahan milenge.", "language_code": "hi-IN"}
        result = think_aloud.build_result_from_sarvam_response(raw, "sess2", recording_start_ms=0.0, audio_duration_ms=3000.0)
        self.assertTrue(result["available"])
        self.assertEqual(result["timing_status"], think_aloud.TIMING_UNTIMED)
        self.assertEqual(len(result["segments"]), 1)
        self.assertIsNone(result["segments"][0]["start_time_ms"])
        self.assertIsNone(result["segments"][0]["end_time_ms"])

    def test_single_large_timed_chunk_with_multiple_sentences_is_split(self):
        """Exact real-run scenario: Sarvam returned the ENTIRE 24s recording
        as ONE timestamped chunk containing six sentences. Evidence
        granularity must improve (multiple segments), but the timestamps
        must never be invented per-sentence -- each derived segment shares
        the parent chunk's real range and is explicitly flagged 'derived'."""
        raw = {
            "transcript": "ignored when per-chunk timestamps exist",
            "language_code": "en-IN",
            "timestamps": {
                "words": [
                    "Mujhe pricing section dhoondhna hai. Mujhe laga tha pricing "
                    "navigation mein hogi. Haan mil gayi mujhe relevant information."
                ],
                "start_time_seconds": [0.0],
                "end_time_seconds": [24.0],
            },
        }
        result = think_aloud.build_result_from_sarvam_response(raw, "sess-big", recording_start_ms=0.0, audio_duration_ms=24000.0)
        self.assertEqual(result["timing_status"], think_aloud.TIMING_TIMED)
        self.assertEqual(len(result["segments"]), 3, "one large chunk with 3 sentences must split into 3 segments")
        for seg in result["segments"]:
            # No fabricated per-sentence timing: every derived segment shares
            # the PARENT chunk's exact real range.
            self.assertEqual(seg["start_time_ms"], 0.0)
            self.assertEqual(seg["end_time_ms"], 24000.0)
            self.assertEqual(seg["timing_source"], think_aloud.TIMING_SOURCE_DERIVED)
        self.assertIn("dhoondhna", result["segments"][0]["display_transcript"].lower())
        self.assertIn("navigation", result["segments"][1]["display_transcript"].lower())
        self.assertIn("information", result["segments"][2]["display_transcript"].lower())

    def test_single_sentence_chunk_keeps_source_timing_not_derived(self):
        """A chunk that is ALREADY exactly one sentence must not be marked
        'derived' -- its timestamp is Sarvam's own real source timing."""
        raw = {
            "transcript": "ignored",
            "language_code": "en-IN",
            "timestamps": {
                "words": ["I am looking for the quizzes section."],
                "start_time_seconds": [2.0],
                "end_time_seconds": [5.0],
            },
        }
        result = think_aloud.build_result_from_sarvam_response(raw, "sess-single", recording_start_ms=0.0, audio_duration_ms=5000.0)
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["timing_source"], think_aloud.TIMING_SOURCE_ORIGINAL)

    def test_untimed_multi_sentence_transcript_still_split_for_evidence(self):
        """No timestamps at all (Case C) -- segments stay untimed, but
        splitting into sentences still improves evidence/classification
        granularity; None start/end is never replaced with a guess."""
        raw = {
            "transcript": "Mujhe pricing chahiye. Yeh sahi jagah nahi hai.",
            "language_code": "en-IN",
        }
        result = think_aloud.build_result_from_sarvam_response(raw, "sess-untimed", recording_start_ms=0.0, audio_duration_ms=None)
        self.assertEqual(result["timing_status"], think_aloud.TIMING_UNTIMED)
        self.assertEqual(len(result["segments"]), 2)
        for seg in result["segments"]:
            self.assertIsNone(seg["start_time_ms"])
            self.assertIsNone(seg["end_time_ms"])
            self.assertEqual(seg["timing_source"], think_aloud.TIMING_SOURCE_UNTIMED)

    def test_single_sentence_untimed_transcript_not_over_split(self):
        raw = {"transcript": "Mujhe laga quizzes yahan milenge.", "language_code": "hi-IN"}
        result = think_aloud.build_result_from_sarvam_response(raw, "sess-single-untimed", 0.0, None)
        self.assertEqual(len(result["segments"]), 1)

    def test_unknown_language_is_not_invented(self):
        raw = {"transcript": "hello", "language_code": "unknown"}
        result = think_aloud.build_result_from_sarvam_response(raw, "sess3", 0.0, None)
        self.assertIsNone(result["language"])

    def test_empty_transcript_marks_unavailable(self):
        raw = {"transcript": "", "language_code": "en-IN"}
        result = think_aloud.build_result_from_sarvam_response(raw, "sess4", 0.0, None)
        self.assertFalse(result["available"])
        self.assertEqual(result["timing_status"], think_aloud.TIMING_INVALID)
        self.assertIsNotNone(result["error"])

    def test_malformed_response_does_not_crash(self):
        result = think_aloud.build_result_from_sarvam_response("not a dict", "sess5", 0.0, None)
        self.assertFalse(result["available"])
        self.assertEqual(result["timing_status"], think_aloud.TIMING_INVALID)

    def test_mismatched_timestamp_arrays_fall_back_to_untimed(self):
        raw = {
            "transcript": "some text",
            "language_code": "en-IN",
            "timestamps": {"words": ["a", "b"], "start_time_seconds": [0.0], "end_time_seconds": [1.0, 2.0]},
        }
        result = think_aloud.build_result_from_sarvam_response(raw, "sess6", 0.0, None)
        self.assertEqual(result["timing_status"], think_aloud.TIMING_UNTIMED)
        self.assertEqual(result["segments"][0]["transcript"], "some text")

    def test_hinglish_transcript_wording_is_preserved_verbatim(self):
        """The transcript builder must never alter/translate returned text."""
        raw = {"transcript": "Mujhe laga quizzes Participate ke andar honge, but yahan nahi mil raha.",
               "language_code": "hi-IN"}
        result = think_aloud.build_result_from_sarvam_response(raw, "sess7", 0.0, None)
        self.assertEqual(result["segments"][0]["transcript"], raw["transcript"])


class SessionLifecycleTests(unittest.TestCase):
    """Item 10: recording state transitions, using real session-relative
    timing (SessionClock), not frontend/SSE time."""

    def test_session_lifecycle_and_duration(self):
        session = think_aloud.create_session("https://x.test/", "find the quiz", {"persona_type": "novice"})
        sid = session["session_id"]
        self.assertEqual(session["events"][0]["event_type"], "think_aloud_session_start")
        self.assertEqual(session["events"][0]["source"], "think_aloud")

        start_ev = think_aloud.record_recording_start(sid)
        self.assertEqual(start_ev["event_type"], "think_aloud_recording_start")
        self.assertGreaterEqual(start_ev["timestamp_ms"], session["events"][0]["timestamp_ms"])

        stop_ev = think_aloud.record_recording_stop(sid)
        self.assertEqual(stop_ev["event_type"], "think_aloud_recording_stop")
        self.assertIsNotNone(stop_ev["metadata"]["duration_ms"])
        self.assertGreaterEqual(stop_ev["metadata"]["duration_ms"], 0.0)

    def test_unknown_session_raises(self):
        with self.assertRaises(think_aloud.SessionNotFound):
            think_aloud.get_session("does-not-exist")


class SarvamClientErrorTests(unittest.TestCase):
    """Items 12-13: Sarvam API failure and quota error handling."""

    def test_not_configured_raises_config_error(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SARVAM_API_KEY", None)
            with self.assertRaises(sarvam_stt.SarvamConfigError):
                asyncio.run(sarvam_stt.transcribe(b"fakeaudiobytes"))

    def test_empty_audio_rejected_before_network_call(self):
        with patch.dict(os.environ, {"SARVAM_API_KEY": "fake-key-for-test"}):
            with self.assertRaises(ValueError):
                asyncio.run(sarvam_stt.transcribe(b""))

    def test_quota_error_maps_to_sarvam_api_error(self):
        class FakeResponse:
            status_code = 429
            text = "rate limited"
            def json(self): return {}

        async def fake_post(*args, **kwargs):
            return FakeResponse()

        with patch.dict(os.environ, {"SARVAM_API_KEY": "fake-key-for-test"}):
            with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=fake_post)):
                with self.assertRaises(sarvam_stt.SarvamAPIError) as ctx:
                    asyncio.run(sarvam_stt.transcribe(b"fakeaudiobytes"))
                self.assertEqual(ctx.exception.status_code, 429)

    def test_auth_error_maps_to_sarvam_api_error(self):
        class FakeResponse:
            status_code = 401
            text = "unauthorized"
            def json(self): return {}

        async def fake_post(*args, **kwargs):
            return FakeResponse()

        with patch.dict(os.environ, {"SARVAM_API_KEY": "fake-key-for-test"}):
            with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=fake_post)):
                with self.assertRaises(sarvam_stt.SarvamAPIError) as ctx:
                    asyncio.run(sarvam_stt.transcribe(b"fakeaudiobytes"))
                self.assertEqual(ctx.exception.status_code, 401)


class EndpointAndModeIsolationTests(unittest.TestCase):
    """Item 14 (Autonomous Agent unchanged) and item 15 (Human + Think-Aloud
    never invokes the autonomous navigator), exercised through the real
    FastAPI app and real ASGI HTTP calls, with only the Sarvam network call
    itself mocked out.

    Uses httpx.AsyncClient(transport=ASGITransport(...)) directly rather than
    fastapi.testclient.TestClient: the installed starlette (0.36.3, bundled
    by the pinned fastapi==0.110.0) predates httpx 0.28's removal of
    TestClient's `app=` shortcut, so TestClient itself raises a TypeError in
    this environment. This is a test-harness-only workaround -- it does not
    touch the application or any dependency version.
    """

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SARVAM_API_KEY", "fake-key-for-test")
        import main as backend_main
        cls.backend_main = backend_main

    def _client(self):
        import httpx
        transport = httpx.ASGITransport(app=self.backend_main.app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    def test_autonomous_agent_endpoint_still_exists_and_is_unaffected(self):
        """Mode A's route must still be present and untouched by Phase 2."""
        routes = {r.path for r in self.backend_main.app.routes}
        self.assertIn("/api/evaluate", routes)

    def test_think_aloud_endpoints_exist(self):
        routes = {r.path for r in self.backend_main.app.routes}
        for p in ["/api/think-aloud/start", "/api/think-aloud/recording/start",
                  "/api/think-aloud/recording/stop", "/api/think-aloud/transcribe"]:
            self.assertIn(p, routes)

    def test_human_mode_never_invokes_autonomous_navigator(self):
        import orchestrator.runner as runner

        async def _run():
            async with self._client() as client:
                start_res = await client.post("/api/think-aloud/start", json={
                    "target_url": "https://x.test/", "task_description": "find the quiz",
                    "persona": {"persona_type": "novice"},
                })
                self.assertEqual(start_res.status_code, 200)
                session_id = start_res.json()["session_id"]

                await client.post("/api/think-aloud/recording/start", data={"session_id": session_id})
                await client.post("/api/think-aloud/recording/stop", data={"session_id": session_id})

                files = {"file": ("recording.webm", b"fake-audio-bytes", "audio/webm")}
                data = {"session_id": session_id, "audio_duration_ms": "4000"}
                transcribe_res = await client.post("/api/think-aloud/transcribe", data=data, files=files)
                self.assertEqual(transcribe_res.status_code, 200)
                result = transcribe_res.json()
                self.assertTrue(result["available"])
                self.assertEqual(result["segments"][0]["transcript"],
                                  "I expected the quizzes section to be under Participate.")

        with patch.object(runner._navigator, "decide_next_action") as fake_decide:
            with patch("tools.sarvam_stt.transcribe", new=AsyncMock(return_value={
                "transcript": "I expected the quizzes section to be under Participate.",
                "language_code": "en-IN",
            })):
                asyncio.run(_run())
            fake_decide.assert_not_called()

    def test_transcribe_with_empty_audio_returns_clean_error_not_crash(self):
        async def _run():
            async with self._client() as client:
                start_res = await client.post("/api/think-aloud/start", json={
                    "target_url": "https://x.test/", "task_description": "t", "persona": {},
                })
                session_id = start_res.json()["session_id"]
                files = {"file": ("recording.webm", b"", "audio/webm")}
                res = await client.post("/api/think-aloud/transcribe", data={"session_id": session_id}, files=files)
                self.assertEqual(res.status_code, 200)  # never a raw crash / never hangs
                body = res.json()
                self.assertFalse(body["available"])
                self.assertIsNotNone(body["error"])
        asyncio.run(_run())

    def test_transcribe_without_sarvam_key_configured_returns_clean_error(self):
        async def _run():
            async with self._client() as client:
                start_res = await client.post("/api/think-aloud/start", json={
                    "target_url": "https://x.test/", "task_description": "t", "persona": {},
                })
                session_id = start_res.json()["session_id"]
                files = {"file": ("recording.webm", b"fake-bytes", "audio/webm")}
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("SARVAM_API_KEY", None)
                    res = await client.post("/api/think-aloud/transcribe", data={"session_id": session_id}, files=files)
                self.assertEqual(res.status_code, 200)
                body = res.json()
                self.assertFalse(body["available"])
                self.assertIn("SARVAM_API_KEY", body["error"])
        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
