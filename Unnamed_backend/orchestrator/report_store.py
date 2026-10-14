"""Persistent local archive for saved Agent and Human Think-Aloud reports.

No database: each saved report is a directory of three files (report.json,
report.md, report.pdf) under output/evaluations/<mode>/<report_id>/, mirroring
the repo's existing output/ convention (see agents/reporter_agent.py's
_save_markdown, which already writes to <repo_root>/output/).

Design rules this module enforces (see the Phase spec):
- report_id is always a 32-char lowercase hex string (uuid4().hex) -- never
  taken from client input without being re-validated against that shape, so
  it can never be used for path traversal.
- Saving never overwrites an earlier archived report. Re-saving the same
  report_id is a no-op that returns the original save's metadata
  (idempotent retries).
- Nothing resembling a secret (api keys, tokens, passwords, cookies, auth
  headers) is ever written to disk -- scrub_secrets() strips any such key
  recursively before serialization, defensively, even though the report
  data this module receives should never contain one in the first place.
"""
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("ux-report-store")

REPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
VALID_MODES = ("agent", "think_aloud")

_SECRET_KEY_MARKERS = (
    "api_key", "apikey", "authorization", "password", "secret",
    "token", "cookie", "auth_header", "access_token", "bearer",
)


class InvalidReportId(ValueError):
    pass


class ReportStoreError(Exception):
    """Wraps filesystem failures so callers can show a safe message without
    leaking paths/stack traces to the client."""


def _repo_root() -> str:
    # this file: Unnamed_backend/orchestrator/report_store.py -> up 2 = Unnamed_backend -> up 1 = repo root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def archive_base_dir() -> str:
    """Base directory for all saved reports. Overridable via the
    ZYNX_REPORT_ARCHIVE_DIR env var so tests can redirect into a tempdir
    without ever touching the real output/ folder."""
    override = os.environ.get("ZYNX_REPORT_ARCHIVE_DIR")
    if override:
        return override
    return os.path.join(_repo_root(), "output", "evaluations")


def generate_report_id() -> str:
    return uuid.uuid4().hex


def is_valid_report_id(report_id: Optional[str]) -> bool:
    return bool(report_id) and bool(REPORT_ID_RE.match(report_id))


def resolve_report_id(client_supplied: Optional[str]) -> str:
    """Use the client's report_id if it's a valid retry of a previous save;
    otherwise mint a fresh one. Never trusts an invalid-shaped id."""
    if client_supplied and is_valid_report_id(client_supplied):
        return client_supplied
    return generate_report_id()


def _mode_dir(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid report mode: {mode!r}")
    return os.path.join(archive_base_dir(), mode)


def report_dir(mode: str, report_id: str) -> str:
    """Resolve the on-disk directory for one report, with defense-in-depth
    against path traversal: report_id is shape-validated AND the resolved
    path is re-checked to still be inside the mode directory."""
    if not is_valid_report_id(report_id):
        raise InvalidReportId(f"Invalid report_id: {report_id!r}")
    base = os.path.abspath(_mode_dir(mode))
    target = os.path.abspath(os.path.join(base, report_id))
    if os.path.commonpath([base, target]) != base:
        raise InvalidReportId("report_id resolved outside the archive directory")
    return target


def scrub_secrets(obj: Any) -> Any:
    """Recursively drop any dict key that looks like a secret. Applied to
    every payload before it is written to disk or returned as an export."""
    if isinstance(obj, dict):
        return {
            k: scrub_secrets(v) for k, v in obj.items()
            if not any(marker in str(k).lower() for marker in _SECRET_KEY_MARKERS)
        }
    if isinstance(obj, list):
        return [scrub_secrets(v) for v in obj]
    return obj


def _atomic_write(path: str, data, is_bytes: bool) -> None:
    tmp_path = path + ".tmp"
    mode = "wb" if is_bytes else "w"
    kwargs = {} if is_bytes else {"encoding": "utf-8"}
    with open(tmp_path, mode, **kwargs) as f:
        f.write(data)
    os.replace(tmp_path, path)


def existing_save_metadata(mode: str, report_id: str) -> Optional[Dict[str, Any]]:
    """If this report_id is already archived, return its top-level
    report_id/mode/generated_at; otherwise None."""
    target_dir = report_dir(mode, report_id)
    json_path = os.path.join(target_dir, "report.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "report_id": data.get("report_id", report_id),
            "mode": data.get("mode", mode),
            "generated_at": data.get("generated_at"),
        }
    except Exception:
        logger.warning(f"Archived report {mode}/{report_id} exists but report.json could not be read back.")
        return {"report_id": report_id, "mode": mode, "generated_at": None}


def write_report_files(mode: str, report_id: str, json_dict: Dict[str, Any], md_text: str, pdf_bytes: bytes) -> None:
    """Writes report.json/.md/.pdf for a NEW report_id. Callers must check
    existing_save_metadata() first -- this function does not itself guard
    against overwriting; it always writes."""
    target_dir = report_dir(mode, report_id)
    try:
        os.makedirs(target_dir, exist_ok=True)
        clean = scrub_secrets(json_dict)
        json_bytes = json.dumps(clean, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        _atomic_write(os.path.join(target_dir, "report.json"), json_bytes, is_bytes=True)
        _atomic_write(os.path.join(target_dir, "report.md"), md_text, is_bytes=False)
        _atomic_write(os.path.join(target_dir, "report.pdf"), pdf_bytes, is_bytes=True)
    except OSError as e:
        logger.exception(f"Failed to persist report {mode}/{report_id}")
        raise ReportStoreError("Could not save the report to disk.") from e


def save_report(mode: str, report_id: Optional[str], build_json: Any, build_md: Any, build_pdf: Any) -> Dict[str, Any]:
    """Orchestrates one Save Report action.

    `build_json`/`build_md`/`build_pdf` are callables taking (report_id,
    generated_at) and returning the respective document -- so the
    (relatively expensive) PDF render only happens when this is genuinely a
    new report, never on an idempotent retry of an already-saved report_id).
    Returns {"report_id", "mode", "saved_at", "already_existed"}.
    """
    resolved_id = resolve_report_id(report_id)
    existing = existing_save_metadata(mode, resolved_id)
    if existing is not None:
        return {
            "report_id": resolved_id, "mode": mode,
            "saved_at": existing.get("generated_at"), "already_existed": True,
        }

    saved_at = datetime.now(timezone.utc).isoformat()
    json_dict = build_json(resolved_id, saved_at)
    md_text = build_md(resolved_id, saved_at)
    pdf_bytes = build_pdf(resolved_id, saved_at)
    write_report_files(mode, resolved_id, json_dict, md_text, pdf_bytes)
    return {"report_id": resolved_id, "mode": mode, "saved_at": saved_at, "already_existed": False}
