import os
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE importing agents
# agents.py reads env vars at module import time
load_dotenv()

import logging
from crewai import Crew, Process

from agents import get_all_agents
from tasks import UXTasks

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ux-main")

def run_ux_test(url: str, audience_description: str):
    # Initialize agents (tools are handled inside agents.py now)
    agents_dict = get_all_agents()
    
    persona_agent   = agents_dict["persona"]
    navigator_agent = agents_dict["navigator"]
    evaluator_agent = agents_dict["evaluator"]
    reporter_agent  = agents_dict["reporter"]

    tasks_obj = UXTasks()

    # 1. Create Persona
    persona_task = tasks_obj.create_persona_task(persona_agent, audience_description)

    # 2. Simulate Navigation
    navigation_task = tasks_obj.navigate_site_task(
        navigator_agent, 
        url, 
        persona_task
    )

    # 3. Evaluate UX
    evaluation_task = tasks_obj.score_ux_performance_task(
        agent=evaluator_agent,
        navigation_task=navigation_task,
        persona_task=persona_task
    )

    # 4. Generate Report
    report_task = tasks_obj.create_final_report_task(
        agent=reporter_agent,
        context_tasks=[persona_task, navigation_task, evaluation_task]
    )

    # Create Crew
    ux_crew = Crew(
        agents=[persona_agent, navigator_agent, evaluator_agent, reporter_agent],
        tasks=[persona_task, navigation_task, evaluation_task, report_task],
        process=Process.sequential,
        verbose=True,
        memory=False # Disabled for local Ollama compatibility
    )

    logger.info(f"Starting UX Test - URL: {url}, Audience: {audience_description}")
    print(f"\n Starting UX Test\nURL: {url}\nAudience: {audience_description}\n")

    try:
        result = ux_crew.kickoff()
        logger.info("Crew execution completed successfully")
        return result
    except Exception as e:
        logger.error(f"Crew execution failed: {e}")
        logger.error("Troubleshooting tips:")
        logger.error("  1. If using Ollama: ensure it's running with `ollama serve`")
        logger.error("  2. If using OpenAI: ensure OPENAI_API_KEY is set in .env")
        logger.error("  3. Check the logs above to see which LLM provider was selected")
        raise


if __name__ == "__main__":
    print("="*50)
    print("      ZYNX UX INTELLIGENCE PLATFORM       ")
    print("="*50)
    
    target_url = input("Enter Website URL (e.g., google.com): ").strip()
    target_audience = input("Enter Audience Description: ").strip()

    if not target_url or not target_audience:
        print("❌ Error: URL and Audience are required!")
    else:
        try:
            report = run_ux_test(target_url, target_audience)
            print("\n" + "="*70)
            print("FINAL UX TEST REPORT GENERATED")
            print("="*70)
            print(f"Report saved to: output/ux_report.md")
        except Exception as e:
            print(f"❌ An error occurred: {e}")