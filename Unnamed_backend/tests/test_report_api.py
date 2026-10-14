"""Endpoint-level tests for the Save Report / Export JSON / Export Markdown /
Export PDF API (main.py), exercised through the real FastAPI app over real
ASGI HTTP calls -- see EndpointAndModeIsolationTests in test_think_aloud.py
for why httpx.ASGITransport is used directly instead of TestClient in this
environment.

Every test redirects report storage into a temporary directory via
ZYNX_REPORT_ARCHIVE_DIR -- never touches the real output/evaluations/ folder.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

AGENT_PAYLOAD = {
    "run_id": "abcd1234",
    "target_url": "https://example.test/",
    "task_description": "Find the pricing section.",
    "persona": {"persona_type": "novice"},
    "task_completed": True,
    "scorecard": {"overall_ux_score": 7, "effectiveness": {"score": 8, "rationale": "ok"}},
    "report": {
        "executive_summary": "Fine overall.",
        "psychological_friction_analysis": "Some load near pricing.",
        "key_recommendations": [],
        "markdown_report": "# existing",
    },
    "friction": {"episodes": [], "episode_count": 0, "signal_count": 0},
}

THINK_ALOUD_REPORT = {
    "session": {"target_url": "https://example.test/", "task_description": "Find pricing",
                "persona": {"persona_type": "novice"}, "audio_duration_ms": 8000.0,
                "language": "hi-IN", "segment_count": 1},
    "task_outcome": "completed",
    "score": {
        "total": 85, "max": 100, "fully_assessed": True, "status": None,
        "disclaimer": "Task UX Score reflects evidence observed in this session...",
        "method_note": "note",
        "dimensions": {
            "task_completion": {"score": 40, "max": 40, "status": None},
            "verbal_ux_signals": {"score": 23, "max": 25, "evidence_segment_ids": {}},
            "recovery": {"score": 15, "max": 20, "rationale": "x"},
            "evidence_completeness": {"score": 7, "max": 15, "rationale": "x"},
        },
    },
    "signal_summary": {"EXPECTATION": {"count": 1, "segment_ids": ["s1"]}},
    "timeline": [{"segment_id": "s1", "start_time_ms": 0.0, "end_time_ms": 2000.0,
                  "timing_source": "original", "labels": ["EXPECTATION"],
                  "text": "Mujhe laga tha pricing yahan hogi.", "language": "hi-IN"}],
    "findings": [{"title": "Stated expectation", "evidence_text": "Mujhe laga tha pricing yahan hogi.",
                  "labels": ["EXPECTATION"], "segment_id": "s1", "start_time_ms": 0.0, "end_time_ms": 2000.0}],
    "expectation_experience": {"expectation": None, "observed_experience": None,
                                "what_happened_next": None, "confirmation": None},
    "summary": "The participant stated an expectation about pricing.",
}

THINK_ALOUD_SEGMENTS = [{
    "segment_id": "s1", "transcript": "Mujhe laga tha pricing yahan hogi.",
    "raw_transcript": "मुझे लगा था प्राइसिंग यहां होगी।",
    "display_transcript": "Mujhe laga tha pricing yahan hogi.",
    "start_time_ms": 0.0, "end_time_ms": 2000.0, "duration_ms": 2000.0,
    "timing_source": "original", "language": "hi-IN", "confidence": 0.9, "source": "sarvam",
    "classification": {"segment_id": "s1", "labels": ["EXPECTATION"],
                        "evidence_text": "Mujhe laga tha pricing yahan hogi.",
                        "confidence": "high", "rationale": "x"},
}]


class ReportApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SARVAM_API_KEY", "fake-key-for-test")
        import main as backend_main
        cls.backend_main = backend_main

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env_patch = patch.dict(os.environ, {"ZYNX_REPORT_ARCHIVE_DIR": self._tmp.name})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmp.cleanup()

    def _client(self):
        import httpx
        transport = httpx.ASGITransport(app=self.backend_main.app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")


class AgentSaveEndpointTests(ReportApiTestCase):
    def test_save_returns_report_id_and_saved_at(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/agent/save", json={"payload": AGENT_PAYLOAD})
                self.assertEqual(res.status_code, 200)
                body = res.json()
                self.assertEqual(body["mode"], "agent")
                self.assertIn("report_id", body)
                self.assertIn("saved_at", body)
                self.assertFalse(body["already_existed"])
        import asyncio
        asyncio.run(_run())

    def test_retry_with_same_report_id_is_idempotent(self):
        async def _run():
            async with self._client() as client:
                first = await client.post("/api/reports/agent/save", json={"payload": AGENT_PAYLOAD})
                rid = first.json()["report_id"]
                second = await client.post("/api/reports/agent/save",
                                            json={"report_id": rid, "payload": AGENT_PAYLOAD})
                self.assertEqual(second.status_code, 200)
                self.assertTrue(second.json()["already_existed"])
                self.assertEqual(second.json()["report_id"], rid)
        import asyncio
        asyncio.run(_run())

    def test_missing_required_field_returns_422_not_500(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/agent/save", json={"payload": {}})
                self.assertEqual(res.status_code, 422)
        import asyncio
        asyncio.run(_run())


class ThinkAloudSaveEndpointTests(ReportApiTestCase):
    def test_save_returns_report_id(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/think-aloud/save", json={
                    "payload": {"report": THINK_ALOUD_REPORT, "segments": THINK_ALOUD_SEGMENTS,
                                "researcher_notes": "n", "task_outcome_note": "o"},
                })
                self.assertEqual(res.status_code, 200)
                body = res.json()
                self.assertEqual(body["mode"], "think_aloud")
                self.assertIn("report_id", body)
        import asyncio
        asyncio.run(_run())

    def test_agent_and_think_aloud_saves_never_collide(self):
        async def _run():
            async with self._client() as client:
                agent_res = await client.post("/api/reports/agent/save", json={"payload": AGENT_PAYLOAD})
                ta_res = await client.post("/api/reports/think-aloud/save", json={
                    "payload": {"report": THINK_ALOUD_REPORT, "segments": THINK_ALOUD_SEGMENTS},
                })
                self.assertNotEqual(agent_res.json()["report_id"], ta_res.json()["report_id"])
        import asyncio
        asyncio.run(_run())


class ExportEndpointTests(ReportApiTestCase):
    def test_agent_json_export_has_correct_content_type_and_filename(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/agent/export/json", json={"payload": AGENT_PAYLOAD})
                self.assertEqual(res.status_code, 200)
                self.assertIn("application/json", res.headers["content-type"])
                self.assertIn("ZYNX_Agent_Report_", res.headers["content-disposition"])
                self.assertIn(".json", res.headers["content-disposition"])
                body = res.json()
                self.assertEqual(body["target_url"], AGENT_PAYLOAD["target_url"])
        import asyncio
        asyncio.run(_run())

    def test_agent_markdown_export_has_correct_content_type(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/agent/export/markdown", json={"payload": AGENT_PAYLOAD})
                self.assertEqual(res.status_code, 200)
                self.assertIn("text/markdown", res.headers["content-type"])
                self.assertIn("# ZYNX — Autonomous Agent UX Evaluation Report", res.text)
        import asyncio
        asyncio.run(_run())

    def test_agent_pdf_export_is_a_valid_pdf(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/agent/export/pdf", json={"payload": AGENT_PAYLOAD})
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.headers["content-type"], "application/pdf")
                self.assertTrue(res.content.startswith(b"%PDF"))
        import asyncio
        asyncio.run(_run())

    def test_think_aloud_json_export_preserves_segments(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/think-aloud/export/json", json={
                    "payload": {"report": THINK_ALOUD_REPORT, "segments": THINK_ALOUD_SEGMENTS},
                })
                self.assertEqual(res.status_code, 200)
                body = res.json()
                self.assertEqual(len(body["segments"]), 1)
                self.assertEqual(body["segments"][0]["display_transcript"], "Mujhe laga tha pricing yahan hogi.")
        import asyncio
        asyncio.run(_run())

    def test_think_aloud_pdf_export_is_a_valid_pdf(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/think-aloud/export/pdf", json={
                    "payload": {"report": THINK_ALOUD_REPORT, "segments": THINK_ALOUD_SEGMENTS},
                })
                self.assertEqual(res.status_code, 200)
                self.assertTrue(res.content.startswith(b"%PDF"))
        import asyncio
        asyncio.run(_run())

    def test_invalid_client_report_id_does_not_crash_export(self):
        async def _run():
            async with self._client() as client:
                res = await client.post("/api/reports/agent/export/json", json={
                    "report_id": "../../etc/passwd", "payload": AGENT_PAYLOAD,
                })
                self.assertEqual(res.status_code, 200)
                self.assertIsNone(res.json()["report_id"])
                self.assertIn("draft", res.headers["content-disposition"])
        import asyncio
        asyncio.run(_run())

    def test_export_does_not_persist_anything_to_disk(self):
        """Exports are downloads only -- generating/downloading must never
        silently archive a report."""
        async def _run():
            async with self._client() as client:
                await client.post("/api/reports/agent/export/json", json={"payload": AGENT_PAYLOAD})
                await client.post("/api/reports/agent/export/pdf", json={"payload": AGENT_PAYLOAD})
        import asyncio
        asyncio.run(_run())
        agent_archive = os.path.join(self._tmp.name, "agent")
        self.assertFalse(os.path.exists(agent_archive))


class ExistingEndpointsRegressionTests(ReportApiTestCase):
    """Confirms the new routes were purely additive."""

    def test_evaluate_endpoint_still_registered(self):
        routes = {r.path for r in self.backend_main.app.routes}
        self.assertIn("/api/evaluate", routes)

    def test_think_aloud_endpoints_still_registered(self):
        routes = {r.path for r in self.backend_main.app.routes}
        for p in ["/api/think-aloud/start", "/api/think-aloud/transcribe", "/api/think-aloud/report"]:
            self.assertIn(p, routes)

    def test_new_report_routes_registered(self):
        routes = {r.path for r in self.backend_main.app.routes}
        for p in ["/api/reports/agent/save", "/api/reports/agent/export/json",
                  "/api/reports/agent/export/markdown", "/api/reports/agent/export/pdf",
                  "/api/reports/think-aloud/save", "/api/reports/think-aloud/export/json",
                  "/api/reports/think-aloud/export/markdown", "/api/reports/think-aloud/export/pdf"]:
            self.assertIn(p, routes)


if __name__ == "__main__":
    unittest.main()
