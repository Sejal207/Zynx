"""Shared helpers for extracting a JSON object from an LLM's raw text response.

LLM responses often wrap the intended JSON in markdown code fences, or add
prose before/after it. This module isolates that JSON safely -- using a
string-aware brace scanner rather than regex -- so nested braces/quotes in
the payload (e.g. inside a "rationale" or "markdown_report" string) can never
corrupt the extracted boundary.
"""
import json
import logging
from typing import Any, Dict, Optional


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline == -1:
        return text
    body = text[first_newline + 1:]
    stripped = body.rstrip()
    if stripped.endswith("```"):
        body = stripped[:-3]
    return body.strip()


def _find_balanced_object(text: str) -> Optional[str]:
    """Return the first top-level {...} object in `text`, respecting string
    quoting/escaping so braces inside string values never break the scan.
    Returns None if no object is found, or if it never closes (i.e. the
    response was truncated before the object finished)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Best-effort extraction of a single JSON object from an LLM response.

    Raises ValueError if no balanced object can be found (most commonly a
    truncated response), or json.JSONDecodeError if the object is balanced
    but not valid JSON. Callers should fall back to a heuristic rather than
    fabricate data in either case.
    """
    if not raw_text:
        raise ValueError("Empty LLM response")
    cleaned = _strip_code_fences(raw_text)
    candidate = _find_balanced_object(cleaned)
    if candidate is None:
        raise ValueError("No balanced JSON object found in LLM response (likely truncated)")
    return json.loads(candidate)


def log_raw_response(logger_: logging.Logger, label: str, raw_text: str, limit: int = 800) -> None:
    """Log a truncated preview of a raw LLM response for local debugging.
    This only ever logs the model's own text output, never request headers
    or API keys."""
    if raw_text is None:
        logger_.debug(f"{label} raw response: <empty>")
        return
    if len(raw_text) <= limit:
        preview = raw_text
    else:
        preview = raw_text[:limit] + f"...[truncated, {len(raw_text)} chars total]"
    logger_.debug(f"{label} raw response: {preview}")
