import asyncio
import logging
from langgraph.graph import StateGraph, END
from .state import UXEvaluationState

# Use relative imports for sibling packages
try:
    from ..agents.multimodal_agent import MultimodalNavigatorAgent
    from ..tools.vision_browser import MultimodalVisionBrowser
except ImportError:
    # Fallback for different execution contexts
    from agents.multimodal_agent import MultimodalNavigatorAgent
    from tools.vision_browser import MultimodalVisionBrowser

logger = logging.getLogger("ux-graph")

# Initialize components
agent = MultimodalNavigatorAgent()
# In a real deployed app, the browser instance would be managed carefully per-run
# For this graph definition, we'll instantiate it dynamically in the node.

async def capture_initial_state(state: UXEvaluationState) -> UXEvaluationState:
    logger.info(f"Capturing initial state for URL: {state['target_url']}")
    try:
        browser = MultimodalVisionBrowser()
        await browser.start()
        screenshot, elements, title = await browser.navigate(state['target_url'])
        await browser.stop()
        logger.info(f"Successfully captured {len(elements)} DOM elements from {title}")
    except Exception as e:
        logger.error(f"Browser error: {e}. Using simulation fallback.")
        # Simulated placeholder screenshot and empty DOM elements
        screenshot = ""
        elements = []
        title = ""
    return {
        **state,
        "current_url": state['target_url'],
        "screenshot_base64": screenshot,
        "dom_elements": elements,
    }

async def agent_decision_step(state: UXEvaluationState) -> UXEvaluationState:
    logger.info("Agent is reasoning about the multimodal state...")
    decision = agent.decide_next_action(state)

    thought = decision.get("thought_trace", "")
    action = decision.get("action", {})
    logger.info(f"Agent Thought: {thought}")
    logger.info(f"Agent Action: {action}")
    
    # Update working memory (sliding window of 3)
    memory = state.get('working_memory', [])
    memory.append(f"Thought: {thought} | Action: {action.get('type')} on {action.get('selector')}")
    if len(memory) > 3:
        memory.pop(0)
        
    history = state.get('interaction_history', [])
    history.append(action)
    
    # Check if agent decided to complete the task
    is_complete = action.get('type') == 'complete'
    
    return {
        **state,
        "working_memory": memory,
        "interaction_history": history,
        "task_completed": is_complete,
        # We temporarily store the pending action in state so the next node can execute it
        "_pending_action": action 
    }

async def execute_action_step(state: UXEvaluationState) -> UXEvaluationState:
    action = state.get("_pending_action")
    if not action or action.get('type') == 'complete':
        logger.info("No action to execute or task complete")
        return state

    logger.info(f"Executing action: {action}")
    browser = MultimodalVisionBrowser()
    await browser.start()
    # Recover state by navigating back to current URL (simplified for prototype)
    await browser.navigate(state['current_url']) 
    
    screenshot, elements, title = await browser.execute_action(
        action_type=action.get('type'),
        selector=action.get('selector'),
        value=action.get('value')
    )
    
    await browser.stop()
    
    return {
        **state,
        "screenshot_base64": screenshot,
        "dom_elements": elements,
        "_pending_action": None # clear pending
    }

def should_continue(state: UXEvaluationState) -> str:
    """Conditional edge router"""
    if state.get("task_completed", False):
        logger.info("Task completed, ending workflow")
        return "end"
    if len(state.get("interaction_history", [])) > 10:  # Max steps
        logger.info("Max steps reached, ending workflow")
        return "end"
    return "continue"


async def compute_metrics(state: UXEvaluationState) -> UXEvaluationState:
    """Aggregate simple telemetry metrics for the frontend.
    This runs after each action execution and stores a `metrics` dict in the state.
    """
    logger.info("Computing metrics...")
    # Basic counts
    interaction_history = state.get('interaction_history', [])
    click_count = sum(1 for a in interaction_history if a.get('type') == 'click')
    scroll_count = sum(1 for a in interaction_history if a.get('type') == 'scroll')
    # Time metrics could be added later (e.g., timestamps in history)
    metrics = {
        'task_success': state.get('task_completed', False),
        'click_count': click_count,
        'scroll_count': scroll_count,
        'cognitive_load': state.get('cognitive_load', 0.0),
        'frustration': state.get('emotional_frustration', 0.0),
    }
    return {
        **state,
        'metrics': metrics
    }


# Define the LangGraph State Machine
workflow = StateGraph(UXEvaluationState)

# Add ALL nodes BEFORE adding edges
workflow.add_node("capture_initial", capture_initial_state)
workflow.add_node("agent_decision", agent_decision_step)
workflow.add_node("execute_action", execute_action_step)
workflow.add_node("compute_metrics", compute_metrics)

workflow.set_entry_point("capture_initial")
workflow.add_edge("capture_initial", "agent_decision")
workflow.add_conditional_edges(
    "agent_decision",
    should_continue,
    {
        "continue": "execute_action",
        "end": END
    }
)
workflow.add_edge("execute_action", "compute_metrics")
workflow.add_edge("compute_metrics", "agent_decision")

evaluation_app = workflow.compile()
