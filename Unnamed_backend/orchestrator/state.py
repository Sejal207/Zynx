from typing import TypedDict, List, Optional, Dict, Any

class AgentAction(TypedDict):
    type: str # 'click', 'type', 'scroll', 'navigate'
    selector: Optional[str]
    value: Optional[str]
    x: Optional[int]
    y: Optional[int]

class UXEvaluationState(TypedDict):
    """
    The shared state for the LangGraph Orchestrator.
    This replaces the sequential CrewAI passing.
    """
    run_id: str
    target_url: str
    persona: Dict[str, Any]
    
    # Current UI State
    current_url: str
    dom_elements: List[Dict[str, Any]] # Extracted interactive elements
    screenshot_base64: str
    
    # Cognitive & Interaction State
    working_memory: List[str] # Limited buffer of what the agent remembers
    interaction_history: List[AgentAction]
    cognitive_load: float
    emotional_frustration: float
    
    # Task Status
    task_description: str
    task_completed: bool
    errors: List[str]
    _pending_action: Optional[AgentAction]

