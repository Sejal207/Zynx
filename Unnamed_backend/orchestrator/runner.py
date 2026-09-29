
"""
Single persistent-browser evaluation loop.

This replaces the old LangGraph orchestration whose nodes opened and closed a
fresh browser on every step (and never advanced current_url), which left the
agent stuck on the homepage doing nothing. Here ONE browser stays open for the
whole run, so clicks/scrolls/navigation actually persist, and every real step
is streamed to the frontend as it happens.

After navigation: UXEvaluatorAgent scores the session, UXReporterAgent writes
the final report, and both are streamed as SSE events.
"""
import logging
import json
import os
from typing import AsyncGenerator, Dict, Any, Optional

try:
    from ..agents.multimodal_agent import MultimodalNavigatorAgent
    from ..agents.evaluator_agent import UXEvaluatorAgent
    from ..agents.reporter_agent import UXReporterAgent
    from ..tools.vision_browser import MultimodalVisionBrowser
except ImportError:
    from agents.multimodal_agent import MultimodalNavigatorAgent
    from agents.evaluator_agent import UXEvaluatorAgent
    from agents.reporter_agent import UXReporterAgent
    from tools.vision_browser import MultimodalVisionBrowser

from .events import SessionClock, make_event
from .friction import analyze_friction

logger = logging.getLogger("ux-runner")

MAX_STEPS = 10

# One agent (LLM client) is fine to share across runs.
_navigator = MultimodalNavigatorAgent()
# Evaluator and reporter get their own client with a much larger max_tokens
# budget than the navigator's -- their JSON payload (rationales, friction
# points, a full markdown report) is far bigger than a one-line navigation
# decision and was being truncated (invalid/unterminated JSON) when it
# shared the navigator's 800-token client.
_analysis_llm = _navigator.build_analysis_llm(2500) if _navigator.llm is not None else None
_evaluator = UXEvaluatorAgent(_analysis_llm)
_reporter = UXReporterAgent(_analysis_llm)


def _action_signature(action: Dict[str, Any], url: str) -> tuple:
    return (
        action.get("type"),
        action.get("index"),
        (action.get("value") or "")[:50],
        url,
    )


def _dom_fingerprint(elements: list) -> tuple:
    """A cheap fingerprint of visible page state, used only to tell whether
    the last action actually changed anything -- not to identify elements."""
    return tuple(
        (el.get("tag"), (el.get("text") or "")[:40])
        for el in (elements or [])[:25]
    )


def _persona_load_modifier(persona: Dict[str, Any]) -> float:
    """
    Returns a multiplier that adjusts how quickly cognitive load builds.
    Low tech literacy → loads faster (1.3x); high literacy → loads slower (0.7x).
    """
    tech = persona.get("tech_literacy")
    if tech is None:
        return 1.0
    level = int(tech)
    if level <= 3:
        return 1.4
    elif level <= 5:
        return 1.15
    elif level <= 7:
        return 1.0
    else:
        return 0.75


def _update_cognitive_metrics(state: Dict[str, Any], action: Dict[str, Any], dom: list, had_error: bool):
    """Lightweight heuristic so the frontend gauges actually move.

    Cognitive load rises when the page is dense or an action fails; it eases
    when the user acts decisively. Frustration accumulates on errors/scrolling.
    Persona tech_literacy adjusts how fast load builds.
    """
    load = state.get("cognitive_load", 0.2)
    frustration = state.get("emotional_frustration", 0.0)
    persona = state.get("persona", {})
    modifier = _persona_load_modifier(persona)

    density = min(len(dom) / 40.0, 1.0)  # busy pages cost more attention
    load = 0.6 * load + 0.4 * (0.3 + 0.5 * density) * modifier

    if had_error:
        load = min(1.0, load + 0.2 * modifier)
        frustration = min(1.0, frustration + 0.25)
    elif action.get("type") == "scroll":
        frustration = min(1.0, frustration + 0.05 * modifier)
    elif action.get("type") == "click":
        load = max(0.0, load - 0.05)

    state["cognitive_load"] = round(min(1.0, load), 2)
    state["emotional_frustration"] = round(frustration, 2)


async def run_evaluation_stream(
    run_id: str, target_url: str, task_description: str, persona: Dict[str, Any]
) -> AsyncGenerator[Dict[str, Any], None]:

    clock = SessionClock()
    ux_events: list = []

    def _record(event_type: str, **kwargs) -> Dict[str, Any]:
        """Record one structured UX event, timestamped at the moment this is
        called. Callers must call this at the actual action boundary."""
        ev = make_event(clock, event_type, **kwargs)
        ux_events.append(ev)
        return ev

    state: Dict[str, Any] = {
        "run_id": run_id,
        "target_url": target_url,
        "persona": persona,
        "current_url": target_url,
        "task_description": task_description,
        "dom_elements": [],
        "working_memory": [],
        "interaction_history": [],
        "cognitive_load": 0.2,
        "emotional_frustration": 0.0,
        "task_completed": False,
        "errors": [],
        "ux_events": ux_events,
    }

    _record("task_start", current_url=target_url,
            metadata={"task_description": task_description, "persona": persona})

    yield {"event": "start", "run_id": run_id, "target_url": target_url, "persona": persona}

    browser = MultimodalVisionBrowser(headless=False)
    try:
        await browser.start()

        # --- Step 0: land on the page -------------------------------------
        screenshot, elements, title = await browser.navigate(target_url)
        # Timestamp captured here, right after the browser action actually
        # completed -- not before dispatch, not at SSE-yield time.
        nav_event = _record("navigation", action={"type": "navigate"},
                             current_url=browser.current_url,
                             metadata={"title": title, "dom_count_after": len(elements)})
        state["current_url"] = browser.current_url
        state["dom_elements"] = elements
        yield _payload("navigate", state, screenshot,
                       thought=f"Landed on '{title or target_url}'. Scanning the page...",
                       action={"type": "navigate"}, events=[nav_event])

        # --- Reason -> act loop -------------------------------------------
        last_signature = None
        last_action_effective = True  # nothing to compare yet
        consecutive_repeats = 0  # how many times the last EXECUTED signature repeated in a row
        pre_fingerprint = _dom_fingerprint(state["dom_elements"])

        for step in range(MAX_STEPS):
            decision = _navigator.decide_next_action(state)
            thought = decision.get("thought_trace", "")
            action = decision.get("action", {}) or {}
            atype = action.get("type", "scroll")
            signature = _action_signature(action, state["current_url"])

            # Loop guard: intervene when the agent proposes the EXACT same
            # action (type+index/value+URL) it just tried, and either (a) that
            # previous attempt produced no visible page change at all, or (b)
            # it's now the 3rd time in a row for this exact signature -- which
            # catches an oscillating case a single before/after fingerprint
            # comparison misses (e.g. a dropdown whose DOM genuinely toggles
            # open/closed on every click, so each individual click "changes
            # something," yet the agent never advances past it). A repeated
            # action against a genuinely progressing page (e.g. "Next page"
            # where the URL changes each time) never matches this signature
            # in the first place, so it's left alone -- not flagged.
            step_events = []

            if atype != "complete" and signature == last_signature and (
                not last_action_effective or consecutive_repeats >= 2
            ):
                logger.warning(
                    f"Repeated-action loop detected at step {step}: "
                    f"{atype} idx={action.get('index')} on {state['current_url']} "
                    f"had no effect last time -- forcing a re-plan."
                )
                state["errors"].append(
                    f"step {step}: repeated-action loop detected "
                    f"({atype} idx={action.get('index')}) -- no state change after last attempt"
                )
                step_events.append(_record(
                    "repeated_action", action=action, current_url=state["current_url"],
                    element_index=action.get("index"),
                    metadata={"step": step, "consecutive_repeats": consecutive_repeats},
                ))
                hint = (
                    f"Your last action ({atype} on element {action.get('index')}) had no "
                    "visible effect on the page. Do NOT repeat it -- choose a different "
                    "element, scroll to see more options, or try a different action."
                )
                mem = state["working_memory"]
                mem.append(hint)
                state["working_memory"] = mem[-3:]

                decision = _navigator.decide_next_action(state)
                thought = decision.get("thought_trace", "")
                action = decision.get("action", {}) or {}
                atype = action.get("type", "scroll")
                signature = _action_signature(action, state["current_url"])

                if atype != "complete" and signature == last_signature:
                    # Still stuck even after the re-plan hint -- force a
                    # distinct action instead of executing the no-op forever.
                    logger.warning(
                        f"Step {step}: navigator repeated the same stuck action even "
                        "after re-plan hint -- forcing a scroll instead."
                    )
                    action = {"type": "scroll", "index": None, "value": None}
                    atype = "scroll"
                    thought = f"{thought} (anti-loop: forced scroll after repeated no-op action)"
                    signature = _action_signature(action, state["current_url"])

            mem = state["working_memory"]
            mem.append(f"{atype} -> {thought[:80]}")
            state["working_memory"] = mem[-3:]

            if atype == "complete":
                # No browser action to wait for -- the decision itself is the
                # event; timestamp it now, at the moment it was decided.
                complete_event = _record("task_end", action=action, current_url=state["current_url"],
                                          metadata={"reason": "agent_completed", "step": step})
                step_events.append(complete_event)
                action["timestamp_ms"] = complete_event["timestamp_ms"]
                action["timestamp_utc"] = complete_event["timestamp_utc"]
                action["event_id"] = complete_event["event_id"]
                state["interaction_history"].append(action)
                state["task_completed"] = True
                yield _payload("agent_decision", state, _last_shot(state),
                               thought=thought, action=action, events=step_events)
                break

            # Look up the target element's text/tag from the DOM snapshot the
            # navigator actually reasoned over, for the structured event.
            # Best-effort only: vision_browser re-reads the live DOM right
            # before clicking, so on a fast-changing page this can lag by one
            # capture -- documented as a known limitation.
            idx = action.get("index")
            dom_before = state.get("dom_elements", [])
            target_el = dom_before[idx] if (isinstance(idx, int) and 0 <= idx < len(dom_before)) else None
            element_text = target_el.get("text") if target_el else None
            element_tag = target_el.get("tag") if target_el else None
            url_before_action = state["current_url"]

            screenshot, elements, title = await browser.execute_action(
                action_type=atype,
                index=action.get("index"),
                selector=action.get("selector"),
                value=action.get("value"),
            )
            new_url = browser.current_url
            new_fingerprint = _dom_fingerprint(elements)
            last_action_effective = (new_fingerprint != pre_fingerprint) or (new_url != url_before_action)
            consecutive_repeats = consecutive_repeats + 1 if signature == last_signature else 0
            last_signature = signature
            pre_fingerprint = new_fingerprint

            had_error = not elements

            # Timestamp captured here, right after the browser action
            # actually completed -- this is the real action boundary.
            action_event = _record(
                atype, action=action, current_url=new_url,
                element_index=action.get("index"), element_text=element_text, element_tag=element_tag,
                state_changed=last_action_effective, had_error=had_error,
                metadata={"step": step, "dom_count_after": len(elements)},
            )
            step_events.append(action_event)
            action["timestamp_ms"] = action_event["timestamp_ms"]
            action["timestamp_utc"] = action_event["timestamp_utc"]
            action["event_id"] = action_event["event_id"]
            state["interaction_history"].append(action)

            if new_url != url_before_action:
                step_events.append(_record(
                    "navigation", action=action, current_url=new_url,
                    metadata={"from_url": url_before_action, "to_url": new_url,
                              "via_action": atype, "step": step},
                ))

            if had_error:
                state["errors"].append(f"step {step}: no interactive elements after {atype}")
                step_events.append(_record(
                    "error", action=action, current_url=new_url,
                    metadata={"reason": "no_interactive_elements_after_action", "step": step},
                ))

            _update_cognitive_metrics(state, action, elements, had_error)

            state["current_url"] = new_url
            state["dom_elements"] = elements
            state["_last_shot"] = screenshot

            yield _payload("execute_action", state, screenshot, thought=thought, action=action, events=step_events)

        if not state["task_completed"]:
            # Loop exhausted MAX_STEPS without the agent deciding "complete".
            _record("task_end", current_url=state["current_url"],
                    metadata={"reason": "max_steps_reached", "steps": len(state["interaction_history"])})

        # --- Raw metrics event -------------------------------------------
        history = state["interaction_history"]
        raw_metrics = {
            "task_success": state["task_completed"],
            "steps": len(history),
            "click_count": sum(1 for a in history if a.get("type") == "click"),
            "scroll_count": sum(1 for a in history if a.get("type") == "scroll"),
            "cognitive_load": state["cognitive_load"],
            "frustration": state["emotional_frustration"],
            "error_count": len(state["errors"]),
        }
        yield {"event": "metrics", "run_id": run_id, "metrics": raw_metrics}

        # --- Behavioural friction analysis (Phase 1) ----------------------
        # Deterministic, rule-based over the timestamped events from Phase 0 --
        # no LLM involved in deciding what counts as a friction episode.
        try:
            friction = analyze_friction(ux_events)
        except Exception as e:
            logger.exception("Friction analysis failed")
            friction = {"episodes": [], "signals": [], "episode_count": 0, "signal_count": 0}
        state["friction_episodes"] = friction["episodes"]
        yield {
            "event": "friction", "run_id": run_id,
            "episodes": friction["episodes"],
            "episode_count": friction["episode_count"],
            "signal_count": friction["signal_count"],
        }

        # --- LLM Evaluation (replaces heuristic scores) ------------------
        yield {"event": "evaluating", "run_id": run_id, "message": "Running UX evaluation..."}
        try:
            scorecard = _evaluator.evaluate(state)
            yield {"event": "scorecard", "run_id": run_id, "scorecard": scorecard}
        except Exception as e:
            logger.exception("Evaluation failed")
            yield {"event": "error", "run_id": run_id, "message": f"Evaluation error: {e}"}
            scorecard = {}

        # --- LLM Report generation ----------------------------------------
        yield {"event": "reporting", "run_id": run_id, "message": "Generating UX report..."}
        try:
            report = _reporter.generate_report(state, scorecard)
            yield {"event": "report", "run_id": run_id, "report": report, "scorecard": scorecard,
                   "friction": friction}
        except Exception as e:
            logger.exception("Report generation failed")
            yield {"event": "error", "run_id": run_id, "message": f"Report error: {e}"}

    except Exception as e:
        logger.exception("Run failed")
        _record("error", current_url=state.get("current_url", target_url), metadata={"reason": str(e)})
        yield {"event": "error", "run_id": run_id, "message": str(e)}
    finally:
        await browser.stop()
        # Full structured event list for this run, in chronological order --
        # additive only; existing "done" consumers only check d.event=="done".
        yield {"event": "done", "run_id": run_id, "events": ux_events}


def _last_shot(state: Dict[str, Any]) -> str:
    return state.get("_last_shot", "")


def _payload(node: str, state: Dict[str, Any], screenshot: str, thought: str, action: dict,
             events: Optional[list] = None) -> Dict[str, Any]:
    return {
        "event": "node_complete",
        "node": node,
        "current_url": state.get("current_url", ""),
        "task_completed": state.get("task_completed", False),
        "cognitive_load": state.get("cognitive_load", 0.0),
        "emotional_frustration": state.get("emotional_frustration", 0.0),
        "working_memory": state.get("working_memory", []),
        "thought_trace": thought,
        "action": action,
        "dom_count": len(state.get("dom_elements", [])),
        "screenshot_base64": screenshot,
        "errors": state.get("errors", []),
        "events": events or [],
    }
