
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
from typing import AsyncGenerator, Dict, Any

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

logger = logging.getLogger("ux-runner")

MAX_STEPS = 10

# One agent (LLM client) is fine to share across runs.
_navigator = MultimodalNavigatorAgent()
# Evaluator and reporter share the same LLM instance
_evaluator = UXEvaluatorAgent(_navigator.llm)
_reporter = UXReporterAgent(_navigator.llm)


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
    }

    yield {"event": "start", "run_id": run_id, "target_url": target_url, "persona": persona}

    browser = MultimodalVisionBrowser(headless=False)
    try:
        await browser.start()

        # --- Step 0: land on the page -------------------------------------
        screenshot, elements, title = await browser.navigate(target_url)
        state["current_url"] = browser.current_url
        state["dom_elements"] = elements
        yield _payload("navigate", state, screenshot,
                       thought=f"Landed on '{title or target_url}'. Scanning the page...",
                       action={"type": "navigate"})

        # --- Reason -> act loop -------------------------------------------
        for step in range(MAX_STEPS):
            decision = _navigator.decide_next_action(state)
            thought = decision.get("thought_trace", "")
            action = decision.get("action", {}) or {}
            atype = action.get("type", "scroll")

            mem = state["working_memory"]
            mem.append(f"{atype} -> {thought[:80]}")
            state["working_memory"] = mem[-3:]
            state["interaction_history"].append(action)

            if atype == "complete":
                state["task_completed"] = True
                yield _payload("agent_decision", state, _last_shot(state),
                               thought=thought, action=action)
                break

            errors_before = len(state["errors"])
            screenshot, elements, title = await browser.execute_action(
                action_type=atype,
                index=action.get("index"),
                selector=action.get("selector"),
                value=action.get("value"),
            )
            state["current_url"] = browser.current_url
            state["dom_elements"] = elements
            state["_last_shot"] = screenshot

            had_error = not elements
            if had_error:
                state["errors"].append(f"step {step}: no interactive elements after {atype}")
            _update_cognitive_metrics(state, action, elements, had_error)

            yield _payload("execute_action", state, screenshot, thought=thought, action=action)

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
            yield {"event": "report", "run_id": run_id, "report": report, "scorecard": scorecard}
        except Exception as e:
            logger.exception("Report generation failed")
            yield {"event": "error", "run_id": run_id, "message": f"Report error: {e}"}

    except Exception as e:
        logger.exception("Run failed")
        yield {"event": "error", "run_id": run_id, "message": str(e)}
    finally:
        await browser.stop()
        yield {"event": "done", "run_id": run_id}


def _last_shot(state: Dict[str, Any]) -> str:
    return state.get("_last_shot", "")


def _payload(node: str, state: Dict[str, Any], screenshot: str, thought: str, action: dict) -> Dict[str, Any]:
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
    }
