import os
import sys
import platform
import asyncio
import logging
from dotenv import load_dotenv

# WINDOWS + PLAYWRIGHT FIX: Playwright's async driver needs to spawn a
# subprocess (the browser), which requires a ProactorEventLoop on Windows.
# Must run before uvicorn/FastAPI create the event loop that will serve
# requests, so it sits at the very top of module import.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

_diag_logger = logging.getLogger("ux-backend.eventloop")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_diag_logger.info("EVENT LOOP DIAGNOSTICS (at module import time)")
_diag_logger.info(f"platform: {sys.platform} ({platform.platform()})")
_diag_logger.info(f"python version: {sys.version}")
_diag_logger.info(f"event loop policy class (after set): {type(asyncio.get_event_loop_policy()).__name__}")

# Load environment variables FIRST. find_dotenv walks up to the repo-root .env.
from dotenv import find_dotenv
load_dotenv(find_dotenv())

# Defensive: a GROQ_API_BASE ending in /openai/v1 makes the groq SDK double the
# path (/openai/v1/openai/v1) and 404. Strip it so the SDK uses its default.
_gb = os.getenv("GROQ_API_BASE", "")
if _gb.rstrip("/").endswith("/openai/v1"):
    os.environ.pop("GROQ_API_BASE", None)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import uuid

logger = logging.getLogger("ux-backend")

app = FastAPI(
    title="Zynx Multimodal UX Intelligence Platform",
    description="API for managing evaluations, telemetry, and reporting.",
    version="0.1.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PersonaModel(BaseModel):
    """Rich persona profile used to simulate a real human user."""
    persona_type: str = "novice"          # novice | intermediate | expert
    name: Optional[str] = None
    age_range: Optional[str] = None       # e.g. "25-34"
    tech_literacy: Optional[int] = None   # 1-10
    primary_goal: Optional[str] = None    # what the user wants to accomplish
    device: Optional[str] = "desktop"     # desktop | mobile | tablet
    domain_familiarity: Optional[str] = None  # e.g. "first time", "occasional user"
    frustration_tolerance: Optional[str] = None  # low | medium | high


class EvaluationRequest(BaseModel):
    target_url: str
    task_description: str = "Explore the website and find the main purpose."
    persona: PersonaModel = PersonaModel()


class ThinkAloudStartRequest(BaseModel):
    """Starts a MODE B (Human + Think-Aloud) session. This does NOT launch
    the autonomous navigator -- it only opens a session-relative clock and
    records a think_aloud_session_start event, exactly like Mode A's
    run_evaluation_stream does for its own clock, but for a human-driven,
    multi-request session instead of one streamed run."""
    target_url: str
    task_description: str = "Explore the website and find the main purpose."
    persona: PersonaModel = PersonaModel()



@app.on_event("startup")
async def _log_actual_event_loop():
    """Diagnostic only: confirms what loop uvicorn actually ends up running
    requests on, since the policy set at import time does not necessarily
    change a loop that uvicorn already created before importing this module."""
    loop = asyncio.get_running_loop()
    policy = asyncio.get_event_loop_policy()
    _diag_logger.info("EVENT LOOP DIAGNOSTICS (actual running loop at FastAPI startup)")
    _diag_logger.info(f"running event-loop class: {type(loop).__name__}")
    _diag_logger.info(f"event loop policy class: {type(policy).__name__}")


@app.get("/")
def read_root():
    return {"message": "Zynx Backend Running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/evaluate")
async def evaluate(request: EvaluationRequest):
    """
    Streams a LangGraph UX evaluation run as Server-Sent Events (SSE).
    Each event is a JSON payload representing a completed graph node.
    """
    logger.info(f"Received evaluation request for URL: {request.target_url}")

    try:
        from .orchestrator.runner import run_evaluation_stream
    except ImportError:
        from orchestrator.runner import run_evaluation_stream

    run_id = str(uuid.uuid4())[:8]

    async def event_stream():
        try:
            async for payload in run_evaluation_stream(
                run_id=run_id,
                target_url=request.target_url,
                task_description=request.task_description,
                persona=request.persona.model_dump(),
            ):
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.02)
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# MODE B -- Human + Think-Aloud (Phase 2)
#
# Distinct from /api/evaluate above: the autonomous navigator is never
# imported or invoked anywhere in this section. A human drives the browser
# themselves; these endpoints only manage the think-aloud recording/session
# lifecycle and Sarvam transcription.
# ============================================================================

try:
    from .orchestrator import think_aloud
    from .tools import sarvam_stt
except ImportError:
    from orchestrator import think_aloud
    from tools import sarvam_stt

ta_logger = logging.getLogger("ux-think-aloud-api")


@app.post("/api/think-aloud/start")
async def think_aloud_start(request: ThinkAloudStartRequest):
    """Open a new Human + Think-Aloud session. Returns a session_id the
    frontend uses for every subsequent call in this session."""
    session = think_aloud.create_session(
        target_url=request.target_url,
        task_description=request.task_description,
        persona=request.persona.model_dump(),
    )
    ta_logger.info(f"Think-Aloud session started: {session['session_id']} ({request.target_url})")
    return {
        "session_id": session["session_id"],
        "event": session["events"][-1],
    }


@app.post("/api/think-aloud/recording/start")
async def think_aloud_recording_start(session_id: str = Form(...)):
    """Marks the actual moment recording starts, on the session's own
    monotonic clock -- not frontend/SSE delivery time."""
    try:
        ev = think_aloud.record_recording_start(session_id)
    except think_aloud.SessionNotFound as e:
        return {"error": str(e)}
    return {"event": ev}


@app.post("/api/think-aloud/recording/stop")
async def think_aloud_recording_stop(session_id: str = Form(...)):
    """Marks the actual moment recording stops, and computes the real
    recording duration from the session clock (never trusts client-reported
    duration as the timestamp of record)."""
    try:
        ev = think_aloud.record_recording_stop(session_id)
    except think_aloud.SessionNotFound as e:
        return {"error": str(e)}
    return {"event": ev}


@app.post("/api/think-aloud/transcribe")
async def think_aloud_transcribe(
    session_id: str = Form(...),
    audio_duration_ms: Optional[float] = Form(None),
    file: UploadFile = File(...),
):
    """Send the recorded audio to Sarvam STT and return a ThinkAloudResult.

    Always returns HTTP 200 with a ThinkAloudResult-shaped body -- even on
    failure -- so the frontend has one simple contract: check
    result["available"]; if False, show result["error"]. This avoids the UI
    ever getting stuck on "Processing..." with no way to distinguish
    "still working" from "failed silently".
    """
    try:
        session = think_aloud.get_session(session_id)
    except think_aloud.SessionNotFound as e:
        return think_aloud.make_result(session_id, [], audio_duration_ms, None,
                                        think_aloud.TIMING_INVALID, available=False, error=str(e))

    audio_bytes = await file.read()

    if not audio_bytes:
        reason = "Empty recording -- no audio data received"
        think_aloud.record_error(session_id, reason)
        return think_aloud.make_result(session_id, [], audio_duration_ms, None,
                                        think_aloud.TIMING_INVALID, available=False, error=reason)

    if not sarvam_stt.is_configured():
        reason = "Sarvam is not configured on this server (SARVAM_API_KEY missing)"
        ta_logger.error(reason)
        think_aloud.record_error(session_id, reason)
        return think_aloud.make_result(session_id, [], audio_duration_ms, None,
                                        think_aloud.TIMING_INVALID, available=False, error=reason)

    try:
        raw = await sarvam_stt.transcribe(audio_bytes, filename=file.filename or "recording.webm")
    except sarvam_stt.SarvamConfigError as e:
        reason = str(e)
        think_aloud.record_error(session_id, reason)
        return think_aloud.make_result(session_id, [], audio_duration_ms, None,
                                        think_aloud.TIMING_INVALID, available=False, error=reason)
    except sarvam_stt.SarvamAPIError as e:
        reason = str(e)
        ta_logger.error(f"Sarvam STT failed for session {session_id}: {reason}")
        think_aloud.record_error(session_id, reason)
        return think_aloud.make_result(session_id, [], audio_duration_ms, None,
                                        think_aloud.TIMING_INVALID, available=False, error=reason)
    except Exception as e:
        reason = f"Unexpected error calling Sarvam: {e}"
        ta_logger.exception(f"Unexpected Sarvam failure for session {session_id}")
        think_aloud.record_error(session_id, reason)
        return think_aloud.make_result(session_id, [], audio_duration_ms, None,
                                        think_aloud.TIMING_INVALID, available=False, error=reason)

    recording_start_ms = think_aloud.get_recording_start_ms(session_id)
    result = think_aloud.build_result_from_sarvam_response(
        raw, session_id, recording_start_ms, audio_duration_ms,
    )
    think_aloud.record_transcription_complete(session_id, result)
    ta_logger.info(
        f"Think-Aloud transcription complete for {session_id}: "
        f"{len(result['segments'])} segment(s), timing={result['timing_status']}, language={result['language']}"
    )
    return result


@app.get("/api/think-aloud/session/{session_id}")
async def think_aloud_get_session(session_id: str):
    """Returns the session's recorded events and last transcription result,
    for debugging/review."""
    try:
        session = think_aloud.get_session(session_id)
    except think_aloud.SessionNotFound as e:
        return {"error": str(e)}
    return {
        "session_id": session_id,
        "target_url": session["target_url"],
        "task_description": session["task_description"],
        "events": session["events"],
        "result": session["result"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
