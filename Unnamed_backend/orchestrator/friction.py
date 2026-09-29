"""Phase 1: Behavioural Friction Analysis.

Converts the timestamped UX events from Phase 0 (orchestrator/events.py) into
structured FrictionSignal / FrictionEpisode objects, using only deterministic,
rule-based detection over observable event data (timestamps, URLs, element
indices, state_changed/had_error flags). No LLM is involved in deciding
whether something is a friction episode -- the underlying events remain the
evidence of record and are always carried along (evidence_event_ids).

Every signal/episode describes OBSERVED BEHAVIOUR only
("Repeated ineffective clicks were observed.") -- never a claimed
psychological state ("the user was frustrated/confused"). That claim can only
ever come from a separate evidence channel (e.g. future audio/facial signals),
which does not exist yet in this codebase.
"""
import uuid
from typing import Any, Dict, List, Optional

# ---- deterministic thresholds (documented, not tuned against any dataset) --
CLICK_REPEAT_WINDOW_MS = 20_000        # same element+url clicked again within this window
DEAD_CLICK_WINDOW_MS = 15_000          # distinct no-effect clicks clustered within this window
EXCESSIVE_SCROLL_MIN_COUNT = 3         # scrolls in a row (ignoring clicks/nav) to count as "excessive"
FAILED_ACTION_WINDOW_MS = 15_000       # action errors clustered within this window
HESITATION_THRESHOLD_MS = 8_000        # gap between one action completing and the next starting
EPISODE_GAP_MS = 12_000                # signals within this gap of each other merge into one episode

SEVERITY_ORDER = ("low", "medium", "high")


def _severity_rank(sev: str) -> int:
    return SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else 0


def _severity_from_count(n: int) -> str:
    if n >= 4:
        return "high"
    if n == 3:
        return "medium"
    return "low"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _make_signal(
    signal_type: str,
    events: List[Dict[str, Any]],
    evidence_summary: str,
    severity: str,
    confidence: float,
    affected_element: Optional[str] = None,
    affected_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one FrictionSignal. Field names follow the requested shape
    (signal_type, start_time, end_time, duration, affected_element,
    affected_url, severity, evidence_event_ids, evidence_summary, confidence,
    metadata); timestamps are suffixed _ms to stay consistent with the
    existing Phase 0 event convention (events.py's timestamp_ms)."""
    times = [e["timestamp_ms"] for e in events]
    start_time = min(times)
    end_time = max(times)
    return {
        "signal_id": _new_id("sig"),
        "signal_type": signal_type,
        "start_time_ms": start_time,
        "end_time_ms": end_time,
        "duration_ms": round(end_time - start_time, 1),
        "affected_element": affected_element,
        "affected_url": affected_url,
        "severity": severity,
        "evidence_event_ids": [e["event_id"] for e in events],
        "evidence_summary": evidence_summary,
        "confidence": confidence,
        "metadata": metadata or {},
    }


def _element_label(event: Dict[str, Any]) -> Optional[str]:
    text = event.get("element_text")
    if text:
        return text.strip().splitlines()[0][:60]
    idx = event.get("element_index")
    return f"element #{idx}" if idx is not None else None


# ------------------------------------------------------------------ rules --

def _detect_repeated_clicks(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 1: the same element on the same URL clicked >=2 times within
    CLICK_REPEAT_WINDOW_MS of each other."""
    signals = []
    clicks = [e for e in events if e["event_type"] == "click"]
    used = set()
    for i, e in enumerate(clicks):
        if e["event_id"] in used:
            continue
        key = (e.get("element_index"), e.get("current_url"))
        group = [e]
        for other in clicks[i + 1:]:
            if other["event_id"] in used:
                continue
            same_key = (other.get("element_index"), other.get("current_url")) == key
            within_window = other["timestamp_ms"] - group[-1]["timestamp_ms"] <= CLICK_REPEAT_WINDOW_MS
            if same_key and within_window:
                group.append(other)
            elif not same_key:
                continue
            else:
                break  # same key but window exceeded -- stop extending this group
        if len(group) >= 2:
            for g in group:
                used.add(g["event_id"])
            signals.append(_make_signal(
                "repeated_click", group,
                evidence_summary=f"{len(group)} repeated clicks on the same element with no progress.",
                severity=_severity_from_count(len(group)),
                confidence=1.0,
                affected_element=_element_label(group[0]),
                affected_url=group[0].get("current_url"),
                metadata={"click_count": len(group)},
            ))
    return signals


def _detect_dead_clicks(events: List[Dict[str, Any]], already_covered: set) -> List[Dict[str, Any]]:
    """Rule 2: clicks that produced no visible state change and no error,
    on DIFFERENT elements, clustered close together in time (i.e. the agent
    tried several things in a row and none of them worked). A single
    isolated no-effect click with nothing else nearby is NOT flagged -- that
    is ordinary exploration, not a finding."""
    dead = [
        e for e in events
        if e["event_type"] in ("click", "type") and e.get("state_changed") is False
        and not e.get("had_error") and e["event_id"] not in already_covered
    ]
    signals = []
    i = 0
    while i < len(dead):
        group = [dead[i]]
        j = i + 1
        while j < len(dead) and dead[j]["timestamp_ms"] - group[-1]["timestamp_ms"] <= DEAD_CLICK_WINDOW_MS:
            group.append(dead[j])
            j += 1
        if len(group) >= 2:
            signals.append(_make_signal(
                "dead_click", group,
                evidence_summary=f"{len(group)} ineffective clicks observed with no resulting page change.",
                severity=_severity_from_count(len(group)),
                confidence=0.9,
                affected_element=_element_label(group[0]),
                affected_url=group[0].get("current_url"),
                metadata={"count": len(group)},
            ))
        i = j if j > i + 1 else i + 1
    return signals


def _detect_excessive_scrolling(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 3: EXCESSIVE_SCROLL_MIN_COUNT+ scroll events in a row (no click/
    navigation between them, i.e. scrolling without acting on anything)."""
    signals = []
    run: List[Dict[str, Any]] = []
    for e in events:
        if e["event_type"] == "scroll":
            run.append(e)
        elif e["event_type"] in ("click", "type", "navigation"):
            if len(run) >= EXCESSIVE_SCROLL_MIN_COUNT:
                signals.append(_make_signal(
                    "excessive_scroll", run,
                    evidence_summary=f"{len(run)} consecutive scrolls with no intervening click.",
                    severity=_severity_from_count(len(run)),
                    confidence=1.0,
                    affected_url=run[0].get("current_url"),
                    metadata={"scroll_count": len(run)},
                ))
            run = []
    if len(run) >= EXCESSIVE_SCROLL_MIN_COUNT:
        signals.append(_make_signal(
            "excessive_scroll", run,
            evidence_summary=f"{len(run)} consecutive scrolls with no intervening click.",
            severity=_severity_from_count(len(run)),
            confidence=1.0,
            affected_url=run[0].get("current_url"),
            metadata={"scroll_count": len(run)},
        ))
    return signals


def _detect_scroll_reversal(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 4: scrolling back the opposite direction shortly after scrolling
    one way. LIMITATION: the current browser layer (vision_browser.py) only
    ever issues a fixed downward scroll -- there is no upward-scroll action
    and no `direction` field recorded on scroll events. This rule reads
    metadata.direction defensively so it activates automatically the moment
    directional scrolling is added, but on the current data it will always
    find zero matches (this is documented, not hidden)."""
    signals = []
    scrolls = [e for e in events if e["event_type"] == "scroll" and e.get("metadata", {}).get("direction")]
    for i in range(1, len(scrolls)):
        prev, cur = scrolls[i - 1], scrolls[i]
        if prev["metadata"]["direction"] != cur["metadata"]["direction"]:
            signals.append(_make_signal(
                "scroll_reversal", [prev, cur],
                evidence_summary="Scroll direction reversed shortly after scrolling the other way.",
                severity="low",
                confidence=0.8,
                affected_url=cur.get("current_url"),
            ))
    return signals


def _detect_hesitation(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 5: a long gap between one action completing and the next action
    completing. NOTE: in this system the "actor" is an LLM agent, not a live
    human, so this measures decision/network latency between actions, not
    human deliberation -- it is still a legitimate, observable behavioural-
    timing signal (and the only kind of "hesitation" this architecture can
    honestly report), but is flagged with reduced confidence and the
    provenance is recorded in metadata rather than asserted as certain."""
    signals = []
    actionable = [e for e in events if e["event_type"] in ("click", "scroll", "type", "navigation")]
    for i in range(1, len(actionable)):
        gap = actionable[i]["timestamp_ms"] - actionable[i - 1]["timestamp_ms"]
        if gap >= HESITATION_THRESHOLD_MS:
            signals.append(_make_signal(
                "hesitation", [actionable[i - 1], actionable[i]],
                evidence_summary=f"Extended gap of {round(gap / 1000, 1)}s observed between actions.",
                severity="low" if gap < HESITATION_THRESHOLD_MS * 2 else "medium",
                confidence=0.6,
                affected_url=actionable[i].get("current_url"),
                metadata={"gap_ms": gap, "note": "reflects agent decision/network latency, not confirmed human deliberation"},
            ))
    return signals


def _detect_repeated_hover(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 6: repeated hovering over the same element. LIMITATION: hover is
    not currently captured anywhere in this codebase (confirmed: no `hover`
    event_type is ever emitted by orchestrator/runner.py or
    tools/vision_browser.py). This rule is implemented and will activate
    automatically if hover telemetry is added later; on current data it
    always returns no signals."""
    hovers = [e for e in events if e["event_type"] == "hover"]
    if not hovers:
        return []
    signals = []
    used = set()
    for i, e in enumerate(hovers):
        if e["event_id"] in used:
            continue
        key = (e.get("element_index"), e.get("current_url"))
        group = [e]
        for other in hovers[i + 1:]:
            if (other.get("element_index"), other.get("current_url")) == key:
                group.append(other)
        if len(group) >= 2:
            for g in group:
                used.add(g["event_id"])
            signals.append(_make_signal(
                "repeated_hover", group,
                evidence_summary=f"{len(group)} repeated hovers over the same element.",
                severity=_severity_from_count(len(group)),
                confidence=1.0,
                affected_element=_element_label(group[0]),
                affected_url=group[0].get("current_url"),
            ))
    return signals


def _detect_backtracking(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 7: navigating back to a URL that was already visited earlier in
    the session (there is no dedicated "back" action in this codebase, so
    this is detected purely from the URL sequence)."""
    signals = []
    seen_urls: List[str] = []
    nav_events = [e for e in events if e["event_type"] == "navigation"]
    for e in nav_events:
        url = e.get("current_url")
        if url in seen_urls:
            signals.append(_make_signal(
                "backtracking", [e],
                evidence_summary=f"Returned to a previously visited page ({url}).",
                severity="low",
                confidence=1.0,
                affected_url=url,
            ))
        seen_urls.append(url)
    return signals


def _detect_navigation_loops(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 8: the same URL visited 3+ times over the session -- a cycling
    pattern rather than a single incidental return."""
    signals = []
    nav_events = [e for e in events if e["event_type"] == "navigation"]
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    for e in nav_events:
        by_url.setdefault(e.get("current_url"), []).append(e)
    for url, group in by_url.items():
        if len(group) >= 3:
            signals.append(_make_signal(
                "navigation_loop", group,
                evidence_summary=f"The same page was visited {len(group)} times, suggesting a navigation loop.",
                severity=_severity_from_count(len(group)),
                confidence=1.0,
                affected_url=url,
                metadata={"visit_count": len(group)},
            ))
    return signals


def _detect_repeated_failed_actions(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 9: >=2 action errors (had_error=True) clustered within
    FAILED_ACTION_WINDOW_MS -- distinct from a "dead click" (which means the
    action ran but changed nothing); had_error specifically means the
    browser layer found no interactive elements at all afterward."""
    signals = []
    failed = [e for e in events if e.get("had_error") is True]
    i = 0
    while i < len(failed):
        group = [failed[i]]
        j = i + 1
        while j < len(failed) and failed[j]["timestamp_ms"] - group[-1]["timestamp_ms"] <= FAILED_ACTION_WINDOW_MS:
            group.append(failed[j])
            j += 1
        if len(group) >= 2:
            signals.append(_make_signal(
                "repeated_failed_action", group,
                evidence_summary=f"{len(group)} actions in a row failed to find any interactive elements.",
                severity=_severity_from_count(len(group)),
                confidence=1.0,
                affected_url=group[0].get("current_url"),
                metadata={"count": len(group)},
            ))
        i = j if j > i + 1 else i + 1
    return signals


def _detect_repeated_action_loops(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rule 10: fold the navigator's own loop-guard events (Phase 0,
    event_type "repeated_action") into friction evidence. Per spec, these
    must NOT be duplicated -- all repeated_action events referencing the
    same (element_index, url) are merged into ONE signal."""
    loop_events = [e for e in events if e["event_type"] == "repeated_action"]
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for e in loop_events:
        key = (e.get("element_index"), e.get("current_url"))
        grouped.setdefault(key, []).append(e)
    signals = []
    for (idx, url), group in grouped.items():
        signals.append(_make_signal(
            "repeated_action_loop", group,
            evidence_summary=(
                f"The navigator's own loop guard intervened {len(group)} time(s) "
                f"on element #{idx} at this page -- the same action was being "
                "repeated with no effect."
            ),
            severity=_severity_from_count(len(group) + 1),  # the guard already means real struggle
            confidence=1.0,
            affected_element=f"element #{idx}" if idx is not None else None,
            affected_url=url,
            metadata={"guard_interventions": len(group)},
        ))
    return signals


_RULES = (
    # _detect_repeated_clicks is run separately in detect_friction_signals()
    # (its output feeds covered_click_ids for dead-click dedup) -- not
    # listed here to avoid running it, and double-counting its signals, twice.
    _detect_excessive_scrolling,
    _detect_scroll_reversal,
    _detect_hesitation,
    _detect_repeated_hover,
    _detect_backtracking,
    _detect_navigation_loops,
    _detect_repeated_failed_actions,
    _detect_repeated_action_loops,
)


def detect_friction_signals(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run every deterministic rule over the chronological event list and
    return all FrictionSignals found, sorted by start time."""
    events = sorted(events, key=lambda e: e["timestamp_ms"])
    signals: List[Dict[str, Any]] = []
    covered_click_ids: set = set()

    repeated_click_signals = _detect_repeated_clicks(events)
    signals.extend(repeated_click_signals)
    for s in repeated_click_signals:
        covered_click_ids.update(s["evidence_event_ids"])

    signals.extend(_detect_dead_clicks(events, covered_click_ids))
    for rule in _RULES:
        signals.extend(rule(events))

    signals.sort(key=lambda s: s["start_time_ms"])
    return signals


def _episode_from_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    start = min(s["start_time_ms"] for s in group)
    end = max(s["end_time_ms"] for s in group)
    signal_types = sorted({s["signal_type"] for s in group})
    affected_elements = sorted({s["affected_element"] for s in group if s["affected_element"]})
    affected_urls = sorted({s["affected_url"] for s in group if s["affected_url"]})
    evidence_ids = []
    for s in group:
        evidence_ids.extend(s["evidence_event_ids"])
    worst_severity = max((s["severity"] for s in group), key=_severity_rank)
    min_confidence = min(s["confidence"] for s in group)
    summary_bullets = [s["evidence_summary"] for s in group]
    return {
        "episode_id": _new_id("ep"),
        "start_time_ms": start,
        "end_time_ms": end,
        "duration_ms": round(end - start, 1),
        "signal_types": signal_types,
        "affected_elements": affected_elements,
        "affected_urls": affected_urls,
        "severity": worst_severity,
        "evidence_event_ids": sorted(set(evidence_ids)),
        "evidence_summary": summary_bullets,
        "confidence": min_confidence,
        "signal_count": len(group),
        "metadata": {"signal_ids": [s["signal_id"] for s in group]},
    }


def group_into_episodes(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge temporally-close signals into FrictionEpisodes.

    A single qualifying signal (e.g. one repeated_click of 3+, one
    navigation_loop) can be its own episode -- it already represents a
    non-trivial pattern, not "one harmless click" (an ordinary click never
    produces a signal in the first place, so it can never reach this stage).

    "hesitation" signals are handled separately from the main chaining pass:
    by construction a hesitation signal's time range spans from one action to
    the next, so if it were allowed to chain normally, a single long pause
    between two otherwise-unrelated struggle clusters would fuse them into
    one giant episode purely because the pause "touches" both. Instead, each
    hesitation signal is attached to whichever single nearby episode it's
    closest to (if any); it never bridges two separate episodes together.
    """
    if not signals:
        return []

    structural = [s for s in signals if s["signal_type"] != "hesitation"]
    hesitations = [s for s in signals if s["signal_type"] == "hesitation"]

    groups: List[List[Dict[str, Any]]] = []
    if structural:
        structural = sorted(structural, key=lambda s: s["start_time_ms"])
        current = [structural[0]]
        for s in structural[1:]:
            gap = s["start_time_ms"] - current[-1]["end_time_ms"]
            if gap <= EPISODE_GAP_MS:
                current.append(s)
            else:
                groups.append(current)
                current = [s]
        groups.append(current)

    for h in hesitations:
        best_idx, best_dist = None, None
        for i, group in enumerate(groups):
            gstart = min(s["start_time_ms"] for s in group)
            gend = max(s["end_time_ms"] for s in group)
            touches = h["end_time_ms"] >= gstart - EPISODE_GAP_MS and h["start_time_ms"] <= gend + EPISODE_GAP_MS
            if touches:
                dist = min(abs(h["start_time_ms"] - gend), abs(gstart - h["end_time_ms"]))
                if best_dist is None or dist < best_dist:
                    best_dist, best_idx = dist, i
        if best_idx is not None:
            groups[best_idx].append(h)
        else:
            groups.append([h])

    episodes = [_episode_from_group(g) for g in groups]
    episodes.sort(key=lambda e: e["start_time_ms"])
    return episodes


def analyze_friction(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Entry point: raw timestamped events in, structured friction summary out."""
    signals = detect_friction_signals(events)
    episodes = group_into_episodes(signals)
    return {
        "episodes": episodes,
        "signals": signals,
        "episode_count": len(episodes),
        "signal_count": len(signals),
    }
