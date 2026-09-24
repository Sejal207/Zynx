import asyncio
import os
from dotenv import load_dotenv

# Load API keys from the root .env file
load_dotenv(dotenv_path="../.env")

from orchestrator.graph import evaluation_app
from orchestrator.state import UXEvaluationState

async def main():
    print("Initializing Zynx Multimodal Agent Test Run...")
    
    initial_state = UXEvaluationState(
        run_id="test-run-001",
        target_url="https://example.com",
        persona={"type": "novice", "goal": "Find more information about this domain."},
        current_url="",
        dom_elements=[],
        screenshot_base64="",
        working_memory=[],
        interaction_history=[],
        cognitive_load=0.0,
        emotional_frustration=0.0,
        task_description="Determine the purpose of this website and try to click the 'More information' link.",
        task_completed=False,
        errors=[]
    )
    
    print(f"Targeting: {initial_state['target_url']}")
    
    # We use stream so we can see the outputs as nodes complete
    async for output in evaluation_app.astream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n--- Node Completed: {node_name} ---")
            if "_pending_action" in state_update and state_update["_pending_action"]:
                print(f"Pending Action: {state_update['_pending_action']}")
            if "task_completed" in state_update and state_update["task_completed"]:
                print("\n✅ Task Completed by Agent!")

if __name__ == "__main__":
    asyncio.run(main())
