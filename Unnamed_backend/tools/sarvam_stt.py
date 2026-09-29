"""Minimal Sarvam Speech-to-Text client for Phase 2 Think-Aloud.

Single purpose: send one audio file to Sarvam's STT API and return the raw
parsed JSON response. Contains no UX-specific logic -- turning that raw
response into ThinkAloudSegment/ThinkAloudResult objects is
orchestrator/think_aloud.py's job, not this module's.

API reference (verified against Sarvam's public docs):
  POST https://api.sarvam.ai/speech-to-text
  header: api-subscription-key: <SARVAM_API_KEY>
  multipart/form-data: file=<audio>, language_code, model, with_timestamps
  language_code="unknown" requests auto language detection across the
  ~22 supported Indic languages + English, and is what lets Hindi, English,
  and Hinglish/code-mixed speech all be sent through the same call without
  the caller having to pre-declare a language.
"""
import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("sarvam-stt")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_MODEL = "saaras:v4"
REQUEST_TIMEOUT_S = 60.0


class SarvamConfigError(Exception):
    """SARVAM_API_KEY is not configured on this server."""


class SarvamAPIError(Exception):
    """Any Sarvam API-level failure: auth, quota, network, timeout, bad response."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def is_configured() -> bool:
    return bool(os.getenv("SARVAM_API_KEY"))


async def transcribe(audio_bytes: bytes, filename: str = "recording.webm") -> Dict[str, Any]:
    """Send audio to Sarvam STT and return the raw response JSON.

    Raises SarvamConfigError if no API key is set, ValueError for an empty
    payload, SarvamAPIError for any network/HTTP/quota/parse failure. Never
    logs the API key or the audio content itself.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SarvamConfigError("SARVAM_API_KEY is not configured on this server")
    if not audio_bytes:
        raise ValueError("Empty audio payload")

    headers = {"api-subscription-key": api_key}
    files = {"file": (filename, audio_bytes)}
    data = {
        "language_code": "unknown",  # auto-detect: Hindi / English / code-mixed
        "model": DEFAULT_MODEL,
        "with_timestamps": "true",
    }

    logger.info(f"Sending {len(audio_bytes)} bytes to Sarvam STT (model={DEFAULT_MODEL})")

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            resp = await client.post(SARVAM_STT_URL, headers=headers, files=files, data=data)
    except httpx.TimeoutException as e:
        raise SarvamAPIError(f"Sarvam request timed out after {REQUEST_TIMEOUT_S}s") from e
    except httpx.RequestError as e:
        raise SarvamAPIError(f"Network error calling Sarvam: {e}") from e

    if resp.status_code in (401, 403):
        raise SarvamAPIError("Sarvam authentication failed -- check SARVAM_API_KEY", resp.status_code)
    if resp.status_code == 429:
        raise SarvamAPIError("Sarvam quota/rate limit exceeded", resp.status_code)
    if resp.status_code >= 400:
        raise SarvamAPIError(f"Sarvam API error {resp.status_code}: {resp.text[:300]}", resp.status_code)

    try:
        return resp.json()
    except ValueError as e:
        raise SarvamAPIError(f"Sarvam returned a non-JSON response: {e}") from e
