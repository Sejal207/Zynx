"""Phase 2: Human Think-Aloud data model and session management.

Human manually performs a UX task, optionally speaking their thoughts aloud;
that audio is transcribed (Sarvam STT) into timestamped, reviewable segments.

This module deliberately does NOT:
- classify segments (expectation/confusion/intent/etc. -- Phase 3)
- infer emotion, sentiment, or any psychological state
- merge with Phase 1 friction episodes (Phase 3)
- modify Phase 0's event model (events.py) -- it only reuses SessionClock
  and make_event(), which already accept arbitrary event_type strings

Think-Aloud events use the SAME timestamp_ms/timestamp_utc/event_id/
event_type/source conventions as Phase 0 browser events, with source
"think_aloud" instead of "backend", so the two timelines are alignable in a
later phase without needing a different shape today.
"""
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from .events import SessionClock, make_event
from .transliteration import romanize

logger = logging.getLogger("ux-think-aloud")

# ---- timing_status values ---------------------------------------------
TIMING_TIMED = "TIMED"       # every segment has real start/end timestamps
TIMING_UNTIMED = "UNTIMED"   # STT returned text but no usable per-segment timing
TIMING_INVALID = "INVALID"   # no usable transcript at all

# ---- per-segment timing_source values ----------------------------------
# Distinguishes a REAL Sarvam-provided timestamp from a timestamp merely
# shared by several sentences that were split out of one larger Sarvam
# chunk. Never conflate the two -- a "derived" segment's start/end are the
# PARENT chunk's real range, not that sentence's own exact timing, which
# Sarvam never provided and this module never invents.
TIMING_SOURCE_ORIGINAL = "source"    # Sarvam's own timestamp, one sentence per chunk
TIMING_SOURCE_DERIVED = "derived"    # sentence split from a larger timed chunk; shares its range
TIMING_SOURCE_UNTIMED = "untimed"    # no usable timestamp at all

# Best-effort sentence/phrase boundary: '.', '!', '?', or the Hindi danda/
# double-danda ('।', '॥'), followed by whitespace. This is a heuristic, not a
# linguistic sentence tokenizer -- it can over-split on abbreviations or
# decimal numbers, which is an accepted, documented limitation for
# think-aloud speech transcripts (rare in practice here).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentence/phrase units for evidence granularity only --
    never used to invent timing. A single-sentence input returns a
    single-item list unchanged."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

# Think-Aloud-specific event types (kept local to this module -- Phase 0's
# events.py EVENT_TYPES tuple for browser events is intentionally untouched).
THINK_ALOUD_EVENT_TYPES = (
    "think_aloud_session_start",
    "think_aloud_recording_start",
    "think_aloud_recording_stop",
    "think_aloud_transcription_complete",
    "think_aloud_error",
)


def new_segment_id() -> str:
    return f"seg_{uuid.uuid4().hex[:10]}"


def new_session_id() -> str:
    return f"ta_{uuid.uuid4().hex[:10]}"


# ------------------------------------------------------------ data model --

def make_segment(
    raw_transcript: str,
    start_time_ms: Optional[float] = None,
    end_time_ms: Optional[float] = None,
    language: Optional[str] = None,
    confidence: Optional[float] = None,
    timing_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one ThinkAloudSegment.

    Timestamps are only ever populated from real data (Sarvam's returned
    chunk timestamps, offset by the recording's own session-relative start
    time) -- never invented from a segment's position in the list. A segment
    with no real timing has start_time_ms/end_time_ms/duration_ms all None.

    Two text representations are kept, per the research requirement that the
    original Sarvam output must never be discarded or overwritten:
      - raw_transcript: exactly what Sarvam returned, original script intact.
      - display_transcript: the same words with any Devanagari script
        converted to casual Roman/Latin (romanize() is a script conversion
        only -- it never translates meaning, and leaves English/Latin text
        byte-for-byte unchanged).
    `transcript` is kept as an alias of display_transcript for backward
    compatibility with existing frontend/consumer code that already reads
    `segment.transcript` -- it is the field the UI shows and lets the user
    edit; raw_transcript is untouched by that editing.

    `timing_source` records WHERE start_time_ms/end_time_ms actually came
    from -- TIMING_SOURCE_ORIGINAL (Sarvam's own timestamp for exactly this
    text), TIMING_SOURCE_DERIVED (this text is one sentence split out of a
    larger Sarvam chunk, and the timestamps shown are that PARENT chunk's
    shared range, not this sentence's own exact timing), or
    TIMING_SOURCE_UNTIMED (no usable timestamp at all). If not given
    explicitly, it's inferred from whether real timestamps were passed, which
    keeps existing callers (that never mention timing_source) working exactly
    as before.
    """
    duration_ms = None
    if start_time_ms is not None and end_time_ms is not None:
        duration_ms = round(end_time_ms - start_time_ms, 1)
    if timing_source is None:
        timing_source = TIMING_SOURCE_ORIGINAL if duration_ms is not None else TIMING_SOURCE_UNTIMED
    display_transcript = romanize(raw_transcript)
    return {
        "segment_id": new_segment_id(),
        "transcript": display_transcript,
        "raw_transcript": raw_transcript,
        "display_transcript": display_transcript,
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "duration_ms": duration_ms,
        "timing_source": timing_source,
        "language": language,      # None = genuinely unknown, never guessed
        "confidence": confidence,  # None = not provided by STT
        "source": "think_aloud",
    }


def make_result(
    session_id: str,
    segments: List[Dict[str, Any]],
    audio_duration_ms: Optional[float],
    language: Optional[str],
    timing_status: str,
    available: bool,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one ThinkAloudResult."""
    return {
        "available": available,
        "session_id": session_id,
        "audio_duration_ms": audio_duration_ms,
        "segments": segments,
        "language": language,
        "timing_status": timing_status,
        "error": error,
        "source": "think_aloud",
    }


def build_result_from_sarvam_response(
    raw: Dict[str, Any],
    session_id: str,
    recording_start_ms: float,
    audio_duration_ms: Optional[float],
) -> Dict[str, Any]:
    """Convert Sarvam's raw STT JSON into a ThinkAloudResult.

    Sarvam response shape (per api-subscription-key STT API):
      {"transcript": str, "language_code": str | "unknown",
       "language_probability": float | None,
       "timestamps": {"words": [chunk_text, ...],
                       "start_time_seconds": [float, ...],
                       "end_time_seconds": [float, ...]}}
    "words" here are chunk/phrase-level segments per Sarvam's own docs, not
    individual words -- used directly as ThinkAloudSegments. Timestamps from
    Sarvam are relative to the start of the audio FILE; they are offset by
    recording_start_ms (this recording's own session-relative start, captured
    at the real backend/recording boundary) to land on the same
    session-relative timeline Phase 0 already uses.
    """
    if not isinstance(raw, dict):
        return make_result(session_id, [], audio_duration_ms, None, TIMING_INVALID,
                            available=False, error="Malformed STT response (not a JSON object)")

    transcript = (raw.get("transcript") or "").strip()
    language = raw.get("language_code")
    if not language or language == "unknown":
        language = None  # do not invent a language label
    confidence = raw.get("language_probability")
    if not isinstance(confidence, (int, float)):
        confidence = None

    timestamps = raw.get("timestamps") or {}
    words = timestamps.get("words") if isinstance(timestamps, dict) else None
    starts = timestamps.get("start_time_seconds") if isinstance(timestamps, dict) else None
    ends = timestamps.get("end_time_seconds") if isinstance(timestamps, dict) else None

    segments: List[Dict[str, Any]] = []

    have_aligned_timestamps = (
        isinstance(words, list) and isinstance(starts, list) and isinstance(ends, list)
        and len(words) > 0 and len(words) == len(starts) == len(ends)
    )

    if have_aligned_timestamps:
        for text, s, e in zip(words, starts, ends):
            text = (text or "").strip()
            if not text:
                continue
            try:
                s_ms = round(recording_start_ms + float(s) * 1000, 1)
                e_ms = round(recording_start_ms + float(e) * 1000, 1)
            except (TypeError, ValueError):
                # A single malformed timestamp pair -- keep the text, drop timing
                # for this segment only, rather than discarding it entirely.
                s_ms = e_ms = None

            # Sarvam sometimes returns ONE chunk spanning many sentences (a
            # whole 24s recording as a single "word"/timestamp pair). Split
            # for evidence/classification granularity, but never pretend a
            # split-out sentence has its own exact timestamp: every sentence
            # derived from the same chunk shares that chunk's real [s_ms,
            # e_ms] range and is explicitly flagged "derived", never "source".
            sentences = _split_into_sentences(text)
            if not sentences:
                continue
            if len(sentences) == 1:
                timing_source = TIMING_SOURCE_ORIGINAL if s_ms is not None else TIMING_SOURCE_UNTIMED
                segments.append(make_segment(sentences[0], s_ms, e_ms, language, confidence,
                                              timing_source=timing_source))
            else:
                timing_source = TIMING_SOURCE_DERIVED if s_ms is not None else TIMING_SOURCE_UNTIMED
                for sentence in sentences:
                    segments.append(make_segment(sentence, s_ms, e_ms, language, confidence,
                                                  timing_source=timing_source))
        timing_status = TIMING_TIMED if segments else TIMING_INVALID
    elif transcript:
        # STT gave us text but no usable per-segment timestamps at all. Still
        # split into sentences for evidence/classification granularity --
        # timing stays None either way, so this never fabricates timing, it
        # only improves which text unit each classification attaches to.
        sentences = _split_into_sentences(transcript) or [transcript]
        segments = [
            make_segment(sentence, None, None, language, confidence, timing_source=TIMING_SOURCE_UNTIMED)
            for sentence in sentences
        ]
        timing_status = TIMING_UNTIMED
    else:
        segments = []
        timing_status = TIMING_INVALID

    available = any(s["transcript"] for s in segments)
    error = None if available else "Transcript has no usable text"
    return make_result(session_id, segments, audio_duration_ms, language, timing_status, available, error)


# ------------------------------------------------------ session management --
# In-memory only, mirroring the rest of this project's "no database" design.
# A Human + Think-Aloud session spans several separate HTTP calls (start ->
# recording start -> recording stop -> transcribe), unlike the single
# streaming call used for Autonomous Agent runs, so it needs its own small
# server-side state keyed by session_id between those calls.

_SESSIONS: Dict[str, Dict[str, Any]] = {}


class SessionNotFound(Exception):
    pass


def create_session(target_url: str, task_description: str, persona: Dict[str, Any]) -> Dict[str, Any]:
    session_id = new_session_id()
    clock = SessionClock()
    session = {
        "session_id": session_id,
        "clock": clock,
        "target_url": target_url,
        "task_description": task_description,
        "persona": persona,
        "events": [],
        "recording_start_ms": None,
        "recording_stop_ms": None,
        "result": None,
    }
    _SESSIONS[session_id] = session
    ev = make_event(
        clock, "think_aloud_session_start", current_url=target_url,
        metadata={"task_description": task_description, "persona": persona},
    )
    ev["source"] = "think_aloud"
    session["events"].append(ev)
    return session


def get_session(session_id: str) -> Dict[str, Any]:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise SessionNotFound(f"No think-aloud session found for id {session_id}")
    return session


def record_recording_start(session_id: str) -> Dict[str, Any]:
    session = get_session(session_id)
    ev = make_event(session["clock"], "think_aloud_recording_start", current_url=session["target_url"])
    ev["source"] = "think_aloud"
    session["events"].append(ev)
    session["recording_start_ms"] = ev["timestamp_ms"]
    return ev


def record_recording_stop(session_id: str) -> Dict[str, Any]:
    session = get_session(session_id)
    ev = make_event(session["clock"], "think_aloud_recording_stop", current_url=session["target_url"])
    ev["source"] = "think_aloud"
    duration_ms = None
    if session["recording_start_ms"] is not None:
        duration_ms = round(ev["timestamp_ms"] - session["recording_start_ms"], 1)
    ev["metadata"] = {"duration_ms": duration_ms}
    session["events"].append(ev)
    session["recording_stop_ms"] = ev["timestamp_ms"]
    return ev


def record_transcription_complete(session_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    session = get_session(session_id)
    ev = make_event(
        session["clock"], "think_aloud_transcription_complete", current_url=session["target_url"],
        metadata={"segment_count": len(result.get("segments", [])), "timing_status": result.get("timing_status")},
    )
    ev["source"] = "think_aloud"
    session["events"].append(ev)
    session["result"] = result
    return ev


def record_error(session_id: str, reason: str) -> Dict[str, Any]:
    session = get_session(session_id)
    ev = make_event(session["clock"], "think_aloud_error", current_url=session["target_url"],
                     metadata={"reason": reason})
    ev["source"] = "think_aloud"
    session["events"].append(ev)
    return ev


def get_recording_start_ms(session_id: str) -> float:
    session = get_session(session_id)
    if session["recording_start_ms"] is None:
        # No recording/start was ever registered for this session -- fall
        # back to 0.0 (start of the session clock) rather than crash;
        # timing will still be internally consistent, just not anchored to
        # an actual recording-start boundary.
        return 0.0
    return session["recording_start_ms"]
