from crewai import Task

class UXTasks:

    def create_persona_task(self, agent, audience_description):
        return Task(
            description=(
                f"Analyze this audience: '{audience_description}'.\n"
                "Create a detailed profile including:\n"
                "1. Core Demographics & Context.\n"
                "2. Technical Literacy (1-10 scale).\n"
                "3. Cognitive Load Tolerance (High/Medium/Low).\n"
                "4. Emotional Drivers (What makes them happy/frustrated?).\n"
                "5. Common Biases (e.g., F-Pattern scanning, Banner blindness).\n"
                "# PSYCHOLOGY HOOK: Define their likely 'System 1' triggers (what they do without thinking) "
                "and 'System 2' barriers (what makes them stop and analyze)."
            ),
            expected_output="A comprehensive psychological and behavioral profile in structured JSON-like format.",
            agent=agent,
        )

    def navigate_site_task(self, agent, url, persona_task):
        return Task(
            description=(
                f"1. Navigate to {url}.\n"
                "2. Based on the provided persona, act realistically.\n"
                "3. Your goal is to find 'Pricing', 'Sign Up', or 'Contact' information.\n"
                "4. Perform at least 5-7 actions.\n"
                "5. **MANDATORY**: Before every action, log your 'Internal Thought Process'.\n"
                "   - State if the action is System 1 (instinctive click) or System 2 (calculated search).\n"
                "   - Note if your current 'Cognitive Load' feels high or low."
            ),
            expected_output="A chronological log of actions, each accompanied by a System 1/2 reasoning and a Cognitive Load self-assessment.",
            agent=agent,
            context=[persona_task]
        )

    def score_ux_performance_task(self, agent, navigation_task, persona_task):
        return Task(
            description=(
                "Review the navigation logs and the user persona.\n"
                "Score the following UX parameters:\n"
                "1. Effectiveness: Task Success Rate and Error Rate.\n"
                "2. Efficiency: Action count and Interaction Delay.\n"
                "3. Learnability: How quickly the user adapted.\n"
                "4. **Cognitive Load Score**: (1-10) How much mental effort was required?\n"
                "5. **System 1/2 Ratio**: Did the user rely more on intuition or analysis?\n"
                "# PSYCHOLOGY HOOK: Identify 'Friction Points' where the user was forced into slow System 2 thinking."
            ),
            expected_output="Structured UX scores (1-10) with detailed psychological analysis of friction points.",
            agent=agent,
            context=[navigation_task, persona_task]
        )

    def create_final_report_task(self, agent, context_tasks):
        return Task(
            description=(
                "Synthesize all findings into a professional UX Research Report.\n"
                "1. Executive Summary: What is the overall health of the site UX?\n"
                "2. The 'Psychological Friction' Analysis: Where did cognitive load spike?\n"
                "3. Key Findings & Actionable Recommendations.\n"
                "4. UX Scorecard (including Psychological metrics)."
            ),
            expected_output="A polished, stakeholder-ready Markdown report.",
            agent=agent,
            context=context_tasks,
            output_file="output/ux_report.md"
        )