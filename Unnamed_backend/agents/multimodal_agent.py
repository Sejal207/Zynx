import os
import json
import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

try:
    from ..orchestrator.state import UXEvaluationState
except ImportError:
    from orchestrator.state import UXEvaluationState

logger = logging.getLogger("ux-multimodal-agent")


class BaseMultimodalNavigatorAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        logger.info("MULTIMODAL AGENT CONFIGURATION")

        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        logger.info(f"GROQ_API_KEY present: {bool(groq_api_key)}")
        logger.info(f"OPENAI_API_KEY present: {bool(openai_api_key)}")

        if groq_api_key:
            logger.info(f"Using Groq provider with model: {model_name}")
            # NOTE: do NOT pass base_url here -- a stray GROQ_API_BASE pointing at
            # .../openai/v1 makes the client double the path and 404. We let the
            # SDK use its built-in default.
            self.llm = ChatGroq(model=model_name, max_tokens=800, api_key=groq_api_key)
        elif openai_api_key:
            logger.info("Using OpenAI provider with model: gpt-4o-mini")
            self.llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=800, openai_api_key=openai_api_key)
        else:
            logger.warning("No API key found! Operating in simulation mode.")
            self.llm = None

    # ----- simulation fallback (used only if the LLM is unavailable) ---------
    def _get_simulation_action(self, state: "UXEvaluationState", error_msg: str = None) -> Dict[str, Any]:
        history = state.get("interaction_history", [])
        dom = state.get("dom_elements", [])
        suffix = f" (sim: {error_msg})" if error_msg else ""

        if len(history) >= 6:
            return {
                "thought_trace": f"I've explored enough to judge this page.{suffix}",
                "action": {"type": "complete", "index": None, "value": None},
            }

        keywords = ["pricing", "sign up", "sign in", "login", "get started",
                    "learn more", "about", "contact", "product", "service"]
        for el in dom:
            text = el.get("text", "").lower()
            if any(k in text for k in keywords):
                return {
                    "thought_trace": f"'{el.get('text')}' looks relevant to my goal -- clicking it.{suffix}",
                    "action": {"type": "click", "index": el["index"], "value": None},
                }
        return {
            "thought_trace": f"Nothing obvious above the fold -- scrolling to see more.{suffix}",
            "action": {"type": "scroll", "index": None, "value": None},
        }

    def _build_persona_summary(self, persona: Dict[str, Any]) -> str:
        """Build a rich, psychologically detailed persona description from the persona dict."""
        if not persona:
            return "A generic website visitor with no specific profile."

        parts = []
        name = persona.get("name")
        age_range = persona.get("age_range")
        persona_type = persona.get("persona_type", persona.get("type", ""))
        tech_literacy = persona.get("tech_literacy")
        primary_goal = persona.get("primary_goal")
        device = persona.get("device")
        domain_familiarity = persona.get("domain_familiarity")
        frustration_tolerance = persona.get("frustration_tolerance")
        expanded_profile = persona.get("expanded_profile")

        intro = f"You are simulating {name or 'a user'}"
        if age_range:
            intro += f", aged {age_range}"
        if persona_type:
            intro += f", a {persona_type}-level technology user"
        parts.append(intro + ".")

        if tech_literacy is not None:
            level = int(tech_literacy)
            if level <= 3:
                parts.append(
                    f"Tech literacy: {level}/10 — you struggle with unfamiliar interfaces, "
                    "rely on obvious labels and buttons, get confused by jargon, and may click "
                    "wrong things. You need clear visual cues to navigate."
                )
            elif level <= 6:
                parts.append(
                    f"Tech literacy: {level}/10 — you are comfortable with standard web patterns "
                    "but may hesitate on unfamiliar or ambiguous UI. You sometimes scroll to look "
                    "for more obvious options before committing to a click."
                )
            else:
                parts.append(
                    f"Tech literacy: {level}/10 — you are highly proficient. You scan pages "
                    "efficiently using F-pattern reading, quickly identify navigation structures, "
                    "and notice UX flaws like unclear CTAs or broken flows immediately."
                )

        if primary_goal:
            parts.append(f"Your specific goal on this website: {primary_goal}")

        if device:
            if "mobile" in device.lower():
                parts.append(
                    "You are on a mobile device — you prefer large tap targets, "
                    "avoid hover-dependent interactions, and have a narrower field of view."
                )
            else:
                parts.append(f"You are using a {device} device.")

        if domain_familiarity:
            parts.append(f"Your familiarity with this type of website/domain: {domain_familiarity}.")

        if frustration_tolerance:
            tol = frustration_tolerance.lower()
            if "low" in tol:
                parts.append(
                    "Frustration tolerance: LOW — you give up quickly if you can't find "
                    "what you need within 2-3 actions. Express this in your thought trace."
                )
            elif "high" in tol:
                parts.append(
                    "Frustration tolerance: HIGH — you are persistent and methodical, "
                    "willing to dig deep even when the site is confusing."
                )
            else:
                parts.append(f"Frustration tolerance: {frustration_tolerance}.")

        if expanded_profile:
            parts.append(f"\nDetailed psychological profile:\n{expanded_profile}")

        return " ".join(parts)

    def decide_next_action(self, state: "UXEvaluationState") -> Dict[str, Any]:
        if self.llm is None:
            return self._get_simulation_action(state, "No API key configured")

        dom = state.get("dom_elements", [])
        dom_context = "\n".join(
            f"[{el['index']}] <{el['tag']}> \"{el.get('text', '')}\""
            + (f" (label: {el['aria_label']})" if el.get("aria_label") else "")
            for el in dom
        ) or "No interactive elements detected."

        working_memory = "\n".join(state.get("working_memory", [])) or "None"
        persona = state.get("persona", {})
        persona_summary = self._build_persona_summary(persona)

        prompt_text = f"""You are simulating a real human user testing a website's usability.

=== YOUR PERSONA ===
{persona_summary}

=== CURRENT TASK ===
{state.get('task_description')}

=== CURRENT URL ===
{state.get('current_url')}

=== WHAT YOU JUST DID (recent memory) ===
{working_memory}

=== INTERACTIVE ELEMENTS VISIBLE ON SCREEN (act by index) ===
{dom_context}

Behave authentically as this specific persona. Express genuine reactions, hesitations,
and biases that match the persona's tech literacy and goals. Use "complete" only once
the goal is satisfied or clearly unreachable.

Respond with ONLY valid JSON, no prose:
{{"thought_trace": "first-person reasoning as this persona about what you see and why you are making this choice",
  "action": {{"type": "click|scroll|type|complete", "index": <int or null>, "value": "<text for type, else null>"}}}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            # Be tolerant of trailing prose after the JSON object.
            start, end = content.find("{"), content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end + 1]
            result = json.loads(content)
            result.setdefault("action", {}).setdefault("type", "scroll")
            logger.info(f"Parsed action: {result['action'].get('type')} idx={result['action'].get('index')}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return self._get_simulation_action(state, "parse error")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._get_simulation_action(state, str(e)[:80])


class MultimodalNavigatorAgent(BaseMultimodalNavigatorAgent):
    """Agent that reasons over the DOM element list to pick the next interaction."""
    pass
