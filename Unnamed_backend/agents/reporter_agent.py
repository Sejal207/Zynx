"""
UX Reporter Agent
-----------------
Takes the structured evaluation scorecard and synthesises it into:
1. A rich JSON report payload for the frontend
2. A markdown file saved to disk (mirrors the CrewAI terminal output)
"""
import json
import logging
import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage

logger = logging.getLogger("ux-reporter-agent")

REPORTER_PROMPT = """\
You are a professional UX report writer specialising in cognitive psychology and usability research.
You have been given a structured UX evaluation scorecard. Write a polished, stakeholder-ready report.

=== TARGET URL ===
{target_url}

=== PERSONA ===
{persona_summary}

=== TASK ===
{task_description}

=== EVALUATION SCORECARD ===
{scorecard_json}

Write the report as valid JSON with this EXACT schema (no markdown, no prose outside the JSON):
{{
  "executive_summary": "<3-5 sentence paragraph summarising overall UX health>",
  "psychological_friction_analysis": "<3-5 sentence paragraph on where cognitive load spiked and why>",
  "key_recommendations": [
    {{"priority": "high|medium|low", "title": "<short title>", "detail": "<1-2 sentences>"}},
    ...
  ],
  "markdown_report": "<complete markdown report as a single escaped string with \\n for newlines>"
}}

The markdown_report field must be a complete, polished Markdown document with:
- # Zynx UX Evaluation Report header
- ## Executive Summary
- ## Psychological Friction Analysis  
- ## UX Scorecard (table with all dimensions)
- ## Psychological Bias Analysis (table)
- ## Key Findings & Recommendations (numbered, prioritised)
"""


class UXReporterAgent:
    """
    Synthesises evaluation scores into a human-readable report.
    """

    def __init__(self, llm):
        self.llm = llm

    def generate_report(self, state: Dict[str, Any], scorecard: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the final report. Returns a report dict.
        Always saves a markdown file to output/ux_report.md.
        """
        persona = state.get("persona", {})
        persona_parts = []
        for field, label in [
            ("name", "Name"), ("persona_type", "Type"), ("age_range", "Age"),
            ("tech_literacy", "Tech literacy"), ("primary_goal", "Goal"),
            ("device", "Device"), ("domain_familiarity", "Domain familiarity"),
        ]:
            val = persona.get(field)
            if val is not None:
                suffix = "/10" if field == "tech_literacy" else ""
                persona_parts.append(f"{label}: {val}{suffix}")
        persona_summary = "\n".join(persona_parts) if persona_parts else "Generic user"

        if self.llm is not None:
            prompt = REPORTER_PROMPT.format(
                target_url=state.get("target_url", ""),
                persona_summary=persona_summary,
                task_description=state.get("task_description", ""),
                scorecard_json=json.dumps(scorecard, indent=2),
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
                report = json.loads(content)
                logger.info("Report generated successfully by LLM.")
                self._save_markdown(report.get("markdown_report", ""), state)
                return report
            except Exception as e:
                logger.error(f"Reporter LLM call failed: {e}. Using heuristic fallback.")

        # Heuristic fallback
        return self._heuristic_report(state, scorecard, persona_summary)

    def _heuristic_report(self, state: Dict[str, Any], scorecard: Dict[str, Any], persona_summary: str) -> Dict[str, Any]:
        overall = scorecard.get("overall_ux_score", 5)
        issues = scorecard.get("top_issues", [])
        positives = scorecard.get("top_positives", [])
        friction_points = scorecard.get("friction_points", [])
        eff = scorecard.get("effectiveness", {})
        effi = scorecard.get("efficiency", {})
        learn = scorecard.get("learnability", {})
        cog = scorecard.get("cognitive_load", {})
        s1s2 = scorecard.get("system1_system2_ratio", {})
        biases = scorecard.get("psychological_biases", {})

        executive_summary = (
            f"The overall UX health of the site scores {overall}/10. "
            f"{'Task was completed successfully.' if state.get('task_completed') else 'The task was not completed.'} "
            f"Key strengths include: {'; '.join(positives)}. "
            f"Primary areas for improvement: {'; '.join(issues[:2])}."
        )

        friction_desc = "; ".join(fp.get("description", "") for fp in friction_points) or "No major friction points detected."
        psychological_friction = (
            f"Cognitive friction was observed at the following moments: {friction_desc}. "
            f"The user relied on System 1 (intuitive) thinking {s1s2.get('system1_percent', 60)}% of the time "
            f"and System 2 (analytical) thinking {s1s2.get('system2_percent', 40)}% of the time."
        )

        recommendations = []
        for issue in issues:
            recommendations.append({"priority": "high", "title": issue[:50], "detail": issue})

        md = f"""# Zynx UX Evaluation Report

**URL:** {state.get('target_url', 'N/A')}
**Persona:** {persona_summary}
**Task:** {state.get('task_description', 'N/A')}

## Executive Summary
{executive_summary}

## Psychological Friction Analysis
{psychological_friction}

## UX Scorecard

| Metric | Score | Notes |
|--------|-------|-------|
| Effectiveness | {eff.get('score', 'N/A')}/10 | Task success: {eff.get('task_success_rate', 'N/A')} |
| Efficiency | {effi.get('score', 'N/A')}/10 | Actions: {effi.get('action_count', 'N/A')} |
| Learnability | {learn.get('score', 'N/A')}/10 | Adaptation: {learn.get('adaptation_speed', 'N/A')} |
| Cognitive Load | {cog.get('score', 'N/A')}/10 | {cog.get('peak_load_moment', 'N/A')} |
| System 1/2 Ratio | {s1s2.get('system1_percent', 'N/A')}/{s1s2.get('system2_percent', 'N/A')} | |
| **Overall UX Score** | **{overall}/10** | |

## Psychological Bias Analysis

| Bias | Score |
|------|-------|
| F-Pattern Scanning | {biases.get('f_pattern_scanning', 'N/A')}/10 |
| Banner Blindness | {biases.get('banner_blindness', 'N/A')}/10 |
| Confirmation Bias | {biases.get('confirmation_bias', 'N/A')}/10 |
| Anchoring Bias | {biases.get('anchoring_bias', 'N/A')}/10 |
| Availability Heuristic | {biases.get('availability_heuristic', 'N/A')}/10 |

## Key Findings & Recommendations

{chr(10).join(f'{i+1}. **{r["title"]}** — {r["detail"]}' for i, r in enumerate(recommendations))}
"""
        self._save_markdown(md, state)
        return {
            "executive_summary": executive_summary,
            "psychological_friction_analysis": psychological_friction,
            "key_recommendations": recommendations,
            "markdown_report": md,
        }

    def _save_markdown(self, md: str, state: Dict[str, Any]):
        """Save report to output/ux_report.md relative to project root."""
        try:
            # Try to save relative to the repo root (two levels up from this file)
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base, "output")
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "ux_report.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            logger.info(f"Report saved to {path}")
        except Exception as e:
            logger.warning(f"Could not save report to disk: {e}")
