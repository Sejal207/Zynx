"""
UX Evaluator Agent
------------------
Runs once after the navigation loop is complete.
Takes the full interaction history + persona profile and produces
a structured JSON scorecard with all UX dimensions.
"""
import json
import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage

logger = logging.getLogger("ux-evaluator-agent")


EVALUATOR_PROMPT = """\
You are a senior UX researcher and cognitive psychologist.
You have just observed an AI agent simulate a real human user navigating a website.
Your job is to evaluate the quality of the user experience based on the evidence below.

=== PERSONA ===
{persona_summary}

=== TASK ===
{task_description}

=== NAVIGATION LOG ===
{navigation_log}

=== RAW METRICS ===
- Total steps taken: {steps}
- Clicks: {click_count}
- Scrolls: {scroll_count}
- Errors encountered: {error_count}
- Final cognitive load (heuristic): {cognitive_load:.0%}
- Final frustration (heuristic): {frustration:.0%}
- Task completed: {task_success}

=== YOUR SCORING TASK ===
Score each dimension 1-10 with evidence from the navigation log.
Think carefully about how the persona's characteristics interact with what you observed.

Respond with ONLY valid JSON (no markdown, no prose), matching this exact schema:
{{
  "effectiveness": {{
    "score": <1-10>,
    "task_success_rate": <0.0-1.0>,
    "error_rate": <0.0-1.0>,
    "rationale": "<2-3 sentences>"
  }},
  "efficiency": {{
    "score": <1-10>,
    "action_count": <int>,
    "wasted_actions": <int>,
    "rationale": "<2-3 sentences>"
  }},
  "learnability": {{
    "score": <1-10>,
    "adaptation_speed": "<fast|medium|slow>",
    "rationale": "<2-3 sentences>"
  }},
  "cognitive_load": {{
    "score": <1-10>,
    "peak_load_moment": "<describe when load was highest>",
    "rationale": "<2-3 sentences>"
  }},
  "system1_system2_ratio": {{
    "system1_percent": <0-100>,
    "system2_percent": <0-100>,
    "rationale": "<2-3 sentences>"
  }},
  "friction_points": [
    {{
      "step": <int or null>,
      "description": "<what happened>",
      "type": "<navigation|cognitive|visual|information_density|error>",
      "severity": "<low|medium|high>"
    }}
  ],
  "psychological_biases": {{
    "f_pattern_scanning": <1-10>,
    "banner_blindness": <1-10>,
    "confirmation_bias": <1-10>,
    "anchoring_bias": <1-10>,
    "availability_heuristic": <1-10>
  }},
  "overall_ux_score": <1-10>,
  "top_issues": ["<issue 1>", "<issue 2>", "<issue 3>"],
  "top_positives": ["<positive 1>", "<positive 2>"]
}}
"""


class UXEvaluatorAgent:
    """
    Evaluates a completed UX session and produces a structured scorecard.
    Uses the same LLM as the navigator for consistency.
    """

    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the evaluation. Returns a structured scorecard dict.
        Falls back to heuristic scores if LLM is unavailable.
        """
        if self.llm is None:
            return self._heuristic_evaluation(state)

        history = state.get("interaction_history", [])
        memory = state.get("working_memory", [])
        errors = state.get("errors", [])

        # Build a readable navigation log
        nav_log_lines = []
        for i, action in enumerate(history):
            atype = action.get("type", "unknown")
            mem_entry = memory[i] if i < len(memory) else ""
            nav_log_lines.append(f"  Step {i+1}: [{atype}] {mem_entry}")
        if errors:
            for e in errors:
                nav_log_lines.append(f"  ERROR: {e}")
        navigation_log = "\n".join(nav_log_lines) or "No actions recorded."

        # Build persona summary string
        persona = state.get("persona", {})
        persona_parts = []
        if persona.get("name"):
            persona_parts.append(f"Name: {persona['name']}")
        if persona.get("persona_type"):
            persona_parts.append(f"Type: {persona['persona_type']}")
        if persona.get("age_range"):
            persona_parts.append(f"Age: {persona['age_range']}")
        if persona.get("tech_literacy") is not None:
            persona_parts.append(f"Tech literacy: {persona['tech_literacy']}/10")
        if persona.get("primary_goal"):
            persona_parts.append(f"Goal: {persona['primary_goal']}")
        if persona.get("device"):
            persona_parts.append(f"Device: {persona['device']}")
        if persona.get("domain_familiarity"):
            persona_parts.append(f"Domain familiarity: {persona['domain_familiarity']}")
        if persona.get("frustration_tolerance"):
            persona_parts.append(f"Frustration tolerance: {persona['frustration_tolerance']}")
        if persona.get("expanded_profile"):
            persona_parts.append(f"\nFull psychological profile:\n{persona['expanded_profile']}")
        persona_summary = "\n".join(persona_parts) if persona_parts else "Generic user (no persona specified)"

        click_count = sum(1 for a in history if a.get("type") == "click")
        scroll_count = sum(1 for a in history if a.get("type") == "scroll")

        prompt = EVALUATOR_PROMPT.format(
            persona_summary=persona_summary,
            task_description=state.get("task_description", "Explore the website"),
            navigation_log=navigation_log,
            steps=len(history),
            click_count=click_count,
            scroll_count=scroll_count,
            error_count=len(errors),
            cognitive_load=state.get("cognitive_load", 0.0),
            frustration=state.get("emotional_frustration", 0.0),
            task_success=state.get("task_completed", False),
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end + 1]
            result = json.loads(content)
            logger.info(f"Evaluation complete. Overall UX score: {result.get('overall_ux_score')}")
            return result
        except Exception as e:
            logger.error(f"Evaluator LLM call failed: {e}. Using heuristic fallback.")
            return self._heuristic_evaluation(state)

    def _heuristic_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback when LLM is unavailable — compute everything from raw metrics."""
        history = state.get("interaction_history", [])
        errors = state.get("errors", [])
        task_done = state.get("task_completed", False)
        steps = len(history)
        click_count = sum(1 for a in history if a.get("type") == "click")
        scroll_count = sum(1 for a in history if a.get("type") == "scroll")
        error_count = len(errors)
        cog_load = state.get("cognitive_load", 0.5)

        effectiveness_score = round(10 * (1 if task_done else 0.4) * max(0.4, 1 - error_count * 0.15))
        efficiency_score = round(max(1, 10 - max(0, steps - 5) * 0.8))
        learnability_score = round(max(1, 10 - scroll_count * 0.5 - error_count * 1))
        cog_score = round(max(1, 10 - cog_load * 9))
        s1_pct = max(10, min(90, round(60 - scroll_count * 5 - error_count * 5)))
        s2_pct = 100 - s1_pct
        overall = round((effectiveness_score + efficiency_score + learnability_score + cog_score) / 4)

        return {
            "effectiveness": {
                "score": effectiveness_score,
                "task_success_rate": 1.0 if task_done else 0.4,
                "error_rate": min(1.0, error_count / max(1, steps)),
                "rationale": "Heuristic estimate based on task completion and error count."
            },
            "efficiency": {
                "score": efficiency_score,
                "action_count": steps,
                "wasted_actions": max(0, steps - 5),
                "rationale": "Heuristic estimate based on total steps taken."
            },
            "learnability": {
                "score": learnability_score,
                "adaptation_speed": "fast" if scroll_count < 3 else "medium",
                "rationale": "Heuristic estimate based on scrolling and error patterns."
            },
            "cognitive_load": {
                "score": cog_score,
                "peak_load_moment": "During dense DOM interactions",
                "rationale": "Derived from DOM density heuristic formula."
            },
            "system1_system2_ratio": {
                "system1_percent": s1_pct,
                "system2_percent": s2_pct,
                "rationale": "Heuristic: more scrolling and errors indicate more System 2 thinking."
            },
            "friction_points": [
                {"step": None, "description": e, "type": "error", "severity": "medium"}
                for e in errors
            ],
            "psychological_biases": {
                "f_pattern_scanning": 6,
                "banner_blindness": 5,
                "confirmation_bias": 5,
                "anchoring_bias": 4,
                "availability_heuristic": 5
            },
            "overall_ux_score": overall,
            "top_issues": [
                "Unable to accurately assess without LLM evaluation",
                f"{error_count} action error(s) detected",
                f"{scroll_count} scroll(s) suggest navigation difficulty"
            ],
            "top_positives": [
                "Task was completed" if task_done else "Session ran to completion",
                f"{click_count} decisive click(s) recorded"
            ]
        }
