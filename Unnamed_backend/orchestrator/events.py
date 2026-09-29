"""Structured, timestamped UX event model (Phase 0 of the multimodal extension).

Every backend/browser action gets a real timestamp captured at the moment it
actually happened -- not when an SSE message was emitted, not when the
frontend rendered it. This is additive: it does not replace or restructure
interaction_history, it only attaches timestamped structured events alongside
it, so existing evaluator/reporter/frontend code keeps working unchanged.
"""
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# The full set of event_types this module emits. Kept here so callers/tests
# have one source of truth for what's expected to exist.
EVENT_TYPES = (
    "task_start", "navigation", "click", "type", "scroll",
    "repeated_action", "error", "task_end",
)


class SessionClock:
    """A single time origin for one evaluation run.

    time.monotonic() is immune to system clock adjustments (NTP corrections,
    DST, manual clock changes) and is what behavioral ordering/duration
    should always be computed from -- it's what "monotonic elapsed-session
    timestamp" means here. time.time() (as an absolute UTC ISO8601 string) is
    captured alongside it purely for human/log correlation; it is never used
    for ordering or duration math.
    """

    def __init__(self):
        self._monotonic_origin = time.monotonic()

    def elapsed_ms(self) -> float:
        return round((time.monotonic() - self._monotonic_origin) * 1000, 1)

    @staticmethod
    def wall_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def new_event_id() -> str:
    return uuid.uuid4().hex[:12]


def make_event(
    clock: SessionClock,
    event_type: str,
    *,
    action: Optional[Dict[str, Any]] = None,
    current_url: Optional[str] = None,
    element_index: Optional[int] = None,
    element_text: Optional[str] = None,
    element_tag: Optional[str] = None,
    state_changed: Optional[bool] = None,
    had_error: Optional[bool] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one structured UX event.

    IMPORTANT: this is timestamped at the instant it is called. Callers MUST
    call this at the actual action boundary -- i.e. right after an
    `await browser.execute_action(...)` (or `.navigate(...)`) call returns,
    never before dispatching it and never at SSE-yield/delivery time.
    """
    return {
        "event_id": new_event_id(),
        "timestamp_ms": clock.elapsed_ms(),
        "timestamp_utc": clock.wall_iso(),
        "event_type": event_type,
        "action": action,
        "current_url": current_url,
        "element_index": element_index,
        "element_text": element_text,
        "element_tag": element_tag,
        "state_changed": state_changed,
        "had_error": had_error,
        "source": "backend",
        "metadata": metadata or {},
    }
