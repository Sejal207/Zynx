import os
import logging
from crewai import Agent, LLM
from crewai.tools import BaseTool
from typing import List
from tools.browser_tool import PlaywrightBrowserTool

# Setup logging
logger = logging.getLogger("ux-agents")

# ====================== LLM CONFIGURATION ======================
# Strategy: Use Groq if available (fast & reliable tool calling), 
# otherwise fall back to Ollama qwen2.5:7b.

# Get API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_llm(provider="groq"):
    """
    Returns the appropriate LLM instance.
    Options: 'groq', 'openai', 'ollama'
    """
    logger.info("=" * 50)
    logger.info("LLM CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Requested provider: {provider}")
    logger.info(f"GROQ_API_KEY present: {bool(GROQ_API_KEY)}")
    logger.info(f"OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")

    if provider == "groq" and GROQ_API_KEY:
        logger.info("Using Groq provider (llama-3.3-70b-versatile)")
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.3,
        )
    elif provider == "openai" and OPENAI_API_KEY:
        logger.info("Using OpenAI provider (gpt-4o-mini)")
        return LLM(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=OPENAI_API_KEY
        )
    else:
        logger.info("Using Ollama provider (localhost:11434)")
        logger.info("Model: ollama/qwen2.5:7b")

        # Check Ollama connectivity
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get('models', [])
                model_names = [m.get('name', 'unknown') for m in models]
                logger.info(f"Ollama server is reachable. Available models: {model_names}")
                if not any('qwen2.5' in name for name in model_names):
                    logger.warning("qwen2.5:7b not found in Ollama. Run: ollama pull qwen2.5:7b")
            else:
                logger.warning(f"Ollama returned status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at localhost:11434")
            logger.error("Please ensure Ollama is running: `ollama serve`")
        except Exception as e:
            logger.error(f"Error checking Ollama connectivity: {e}")

        # Default to local Ollama (works out of the box)
        return LLM(
            model="ollama/qwen2.5:7b",
            base_url="http://localhost:11434",
            temperature=0.4,
            api_key=None
        )

# Select your preferred provider here
active_llm = get_llm(provider="groq")  # Using Groq (fast, reliable tool calling)

# ====================== AGENTS ======================

def persona_generator() -> Agent:
    """Creates detailed human-like persona based on audience description"""
    logger.info("Creating Persona Generator Agent")
    return Agent(
        role="Psychological Persona Architect",
        goal="Create a realistic, detailed psychological profile of the target audience for UX testing purposes.",
        backstory=(
            "You are an expert in human psychology, user behavior, and UX research. "
            "You deeply understand cognitive load, attention patterns, emotional triggers, "
            "and decision-making biases. Your personas are practical and directly usable for simulation. "
            "Focus on System 1 (fast/intuitive) vs System 2 (slow/analytical) tendencies."
        ),
        llm=active_llm,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )


def navigator_agent(tools: List[BaseTool] = None) -> Agent:
    """Main agent that acts like a real human user on the website"""
    logger.info("Creating Navigator Agent")
    if tools is None:
        tools = [PlaywrightBrowserTool()]
        logger.info("Initialized with default PlaywrightBrowserTool")

    return Agent(
        role="Simulated Human User",
        goal=(
            "Navigate the website realistically as the defined persona. "
            "Perform tasks naturally, show hesitation, make mistakes, and express emotions."
        ),
        backstory=(
            "You are a real human user with specific psychological traits. "
            "You scan pages quickly using F-patterns, get frustrated with confusing layouts, "
            "and have limited cognitive bandwidth. You think aloud (using log) before every action. "
            "# PSYCHOLOGY RULE: Before any click, explain if it's a System 1 (instinctive) or System 2 (calculated) decision."
        ),
        llm=active_llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
        memory=False,
        max_iter=15,
    )


def evaluator_agent() -> Agent:
    """Evaluates the navigation and scores UX parameters"""
    logger.info("Creating Evaluator Agent")
    return Agent(
        role="UX Evaluator & Analyst",
        goal="Analyze the user's journey and accurately score Effectiveness, Efficiency, Learnability, Engagement, and Emotional signals.",
        backstory=(
            "You are a senior UX researcher. You analyze navigation logs for 'Cognitive Load' spikes "
            "and 'Interaction Friction'. You are objective and provide actionable insights."
        ),
        llm=active_llm,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )


def reporter_agent() -> Agent:
    """Generates the final readable report"""
    logger.info("Creating Reporter Agent")
    return Agent(
        role="Professional UX Report Writer",
        goal="Create a clear, structured, and insightful usability testing report.",
        backstory=(
            "You write professional, concise, and actionable UX reports. "
            "You highlight both positive findings and pain points with clear reasoning."
        ),
        llm=active_llm,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )


# Helper function
def get_all_agents(tools: List[BaseTool] = None):
    logger.info("Initializing all agents...")
    agents = {
        "persona": persona_generator(),
        "navigator": navigator_agent(tools),
        "evaluator": evaluator_agent(),
        "reporter": reporter_agent()
    }
    logger.info("All agents initialized successfully")
    return agents