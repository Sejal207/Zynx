"""Tests for orchestrator/report_store.py: report_id validation/path safety,
save/no-overwrite/idempotent-retry behaviour, secret scrubbing, and
filesystem error handling.

Every test redirects the archive base directory into a temporary directory
via the ZYNX_REPORT_ARCHIVE_DIR env var -- never touches the real
output/evaluations/ folder.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from orchestrator import report_store as rs


class TempArchiveTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env_patch = patch.dict(os.environ, {"ZYNX_REPORT_ARCHIVE_DIR": self._tmp.name})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmp.cleanup()


class ReportIdValidationTests(unittest.TestCase):
    def test_generated_id_is_valid_shape(self):
        rid = rs.generate_report_id()
        self.assertTrue(rs.is_valid_report_id(rid))
        self.assertEqual(len(rid), 32)

    def test_rejects_non_hex_id(self):
        self.assertFalse(rs.is_valid_report_id("not-a-valid-id"))

    def test_rejects_path_traversal_attempt(self):
        self.assertFalse(rs.is_valid_report_id("../../etc/passwd"))
        self.assertFalse(rs.is_valid_report_id("..%2f..%2fsecret"))

    def test_rejects_short_or_long_ids(self):
        self.assertFalse(rs.is_valid_report_id("abc123"))
        self.assertFalse(rs.is_valid_report_id("a" * 33))

    def test_rejects_none_and_empty(self):
        self.assertFalse(rs.is_valid_report_id(None))
        self.assertFalse(rs.is_valid_report_id(""))

    def test_resolve_report_id_falls_back_to_new_id_when_invalid(self):
        resolved = rs.resolve_report_id("../malicious")
        self.assertTrue(rs.is_valid_report_id(resolved))

    def test_resolve_report_id_keeps_valid_client_id(self):
        valid = rs.generate_report_id()
        self.assertEqual(rs.resolve_report_id(valid), valid)


class ReportDirPathSafetyTests(TempArchiveTestCase):
    def test_report_dir_stays_inside_mode_directory(self):
        rid = rs.generate_report_id()
        d = rs.report_dir("agent", rid)
        base = os.path.abspath(os.path.join(rs.archive_base_dir(), "agent"))
        self.assertTrue(os.path.abspath(d).startswith(base))

    def test_report_dir_rejects_invalid_id(self):
        with self.assertRaises(rs.InvalidReportId):
            rs.report_dir("agent", "../../escape")

    def test_report_dir_rejects_invalid_mode(self):
        rid = rs.generate_report_id()
        with self.assertRaises(ValueError):
            rs.report_dir("not_a_real_mode", rid)


class SecretScrubbingTests(unittest.TestCase):
    def test_strips_api_key_fields(self):
        data = {"target_url": "https://x.test", "api_key": "sk-secret", "nested": {"authorization": "Bearer xyz"}}
        clean = rs.scrub_secrets(data)
        self.assertNotIn("api_key", clean)
        self.assertNotIn("authorization", clean["nested"])
        self.assertEqual(clean["target_url"], "https://x.test")

    def test_strips_secrets_inside_lists(self):
        data = {"segments": [{"text": "hi", "auth_header": "Bearer xyz"}]}
        clean = rs.scrub_secrets(data)
        self.assertNotIn("auth_header", clean["segments"][0])
        self.assertEqual(clean["segments"][0]["text"], "hi")

    def test_leaves_ordinary_fields_untouched(self):
        data = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
        self.assertEqual(rs.scrub_secrets(data), data)


class SaveReportPersistenceTests(TempArchiveTestCase):
    def _builders(self, marker="v1"):
        return (
            lambda rid, gen_at: {"report_id": rid, "mode": "agent", "generated_at": gen_at, "marker": marker},
            lambda rid, gen_at: f"# report {marker}",
            lambda rid, gen_at: f"PDFBYTES-{marker}".encode("utf-8"),
        )

    def test_save_writes_all_three_files(self):
        build_json, build_md, build_pdf = self._builders()
        result = rs.save_report("agent", None, build_json, build_md, build_pdf)
        self.assertFalse(result["already_existed"])
        d = rs.report_dir("agent", result["report_id"])
        self.assertTrue(os.path.exists(os.path.join(d, "report.json")))
        self.assertTrue(os.path.exists(os.path.join(d, "report.md")))
        self.assertTrue(os.path.exists(os.path.join(d, "report.pdf")))

    def test_save_generates_a_fresh_id_when_none_given(self):
        build_json, build_md, build_pdf = self._builders()
        result = rs.save_report("agent", None, build_json, build_md, build_pdf)
        self.assertTrue(rs.is_valid_report_id(result["report_id"]))

    def test_retry_with_same_id_is_idempotent_and_does_not_rebuild(self):
        build_json, build_md, build_pdf = self._builders("first")
        first = rs.save_report("agent", None, build_json, build_md, build_pdf)

        rebuild_called = {"json": 0, "md": 0, "pdf": 0}

        def tracking_json(rid, gen_at):
            rebuild_called["json"] += 1
            return {"report_id": rid, "mode": "agent", "generated_at": gen_at, "marker": "second"}

        def tracking_md(rid, gen_at):
            rebuild_called["md"] += 1
            return "# second"

        def tracking_pdf(rid, gen_at):
            rebuild_called["pdf"] += 1
            return b"PDFBYTES-second"

        second = rs.save_report("agent", first["report_id"], tracking_json, tracking_md, tracking_pdf)

        self.assertTrue(second["already_existed"])
        self.assertEqual(second["report_id"], first["report_id"])
        self.assertEqual(second["saved_at"], first["saved_at"])
        self.assertEqual(rebuild_called, {"json": 0, "md": 0, "pdf": 0})

    def test_retry_never_overwrites_original_content(self):
        build_json, build_md, build_pdf = self._builders("original")
        first = rs.save_report("agent", None, build_json, build_md, build_pdf)

        other_json, other_md, other_pdf = self._builders("replacement-attempt")
        rs.save_report("agent", first["report_id"], other_json, other_md, other_pdf)

        d = rs.report_dir("agent", first["report_id"])
        with open(os.path.join(d, "report.json"), encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["marker"], "original")
        with open(os.path.join(d, "report.md"), encoding="utf-8") as f:
            self.assertIn("original", f.read())

    def test_agent_and_think_aloud_are_stored_in_separate_directories(self):
        build_json, build_md, build_pdf = self._builders()
        agent_result = rs.save_report("agent", None, build_json, build_md, build_pdf)
        ta_result = rs.save_report("think_aloud", None, build_json, build_md, build_pdf)

        self.assertNotEqual(agent_result["report_id"], ta_result["report_id"])
        agent_dir = rs.report_dir("agent", agent_result["report_id"])
        ta_dir = rs.report_dir("think_aloud", ta_result["report_id"])
        self.assertNotEqual(agent_dir, ta_dir)
        self.assertIn(os.sep + "agent" + os.sep, agent_dir)
        self.assertIn(os.sep + "think_aloud" + os.sep, ta_dir)

    def test_saving_one_mode_never_overwrites_the_other(self):
        """A report_id that happens to collide between modes (practically
        impossible with uuid4, but defend the invariant explicitly) must
        still resolve to two independent files, since mode is part of the
        storage path, not just the id."""
        shared_id = rs.generate_report_id()
        agent_json, agent_md, agent_pdf = self._builders("agent-content")
        ta_json, ta_md, ta_pdf = self._builders("think-aloud-content")

        rs.save_report("agent", shared_id, agent_json, agent_md, agent_pdf)
        rs.save_report("think_aloud", shared_id, ta_json, ta_md, ta_pdf)

        with open(os.path.join(rs.report_dir("agent", shared_id), "report.json"), encoding="utf-8") as f:
            agent_saved = json.load(f)
        with open(os.path.join(rs.report_dir("think_aloud", shared_id), "report.json"), encoding="utf-8") as f:
            ta_saved = json.load(f)
        self.assertEqual(agent_saved["marker"], "agent-content")
        self.assertEqual(ta_saved["marker"], "think-aloud-content")

    def test_multiple_distinct_saves_never_collide(self):
        ids = set()
        for i in range(5):
            build_json, build_md, build_pdf = self._builders(f"v{i}")
            result = rs.save_report("agent", None, build_json, build_md, build_pdf)
            ids.add(result["report_id"])
        self.assertEqual(len(ids), 5)

    def test_saved_json_never_contains_secret_like_fields(self):
        def build_json(rid, gen_at):
            return {"report_id": rid, "mode": "agent", "generated_at": gen_at, "api_key": "sk-should-not-be-saved"}
        result = rs.save_report("agent", None, build_json, lambda r, g: "", lambda r, g: b"")
        d = rs.report_dir("agent", result["report_id"])
        with open(os.path.join(d, "report.json"), encoding="utf-8") as f:
            saved = json.load(f)
        self.assertNotIn("api_key", saved)


class FilesystemErrorHandlingTests(TempArchiveTestCase):
    def test_write_failure_raises_report_store_error_not_a_crash(self):
        rid = rs.generate_report_id()
        with patch("builtins.open", side_effect=OSError("disk full")):
            with self.assertRaises(rs.ReportStoreError):
                rs.write_report_files("agent", rid, {"report_id": rid, "mode": "agent", "generated_at": "now"}, "md", b"pdf")


if __name__ == "__main__":
    unittest.main()
