import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables FIRST. find_dotenv walks up to the repo-root .env.
from dotenv import find_dotenv
load_dotenv(find_dotenv())

# Defensive: a GROQ_API_BASE ending in /openai/v1 makes the groq SDK double the
# path (/openai/v1/openai/v1) and 404. Strip it so the SDK uses its default.
_gb = os.getenv("GROQ_API_BASE", "")
if _gb.rstrip("/").endswith("/openai/v1"):
    os.environ.pop("GROQ_API_BASE", None)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import json
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ux-backend")

app = FastAPI(
    title="Zynx Multimodal UX Intelligence Platform",
    description="API for managing evaluations, telemetry, and reporting.",
    version="0.1.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PersonaModel(BaseModel):
    """Rich persona profile used to simulate a real human user."""
    persona_type: str = "novice"          # novice | intermediate | expert
    name: Optional[str] = None
    age_range: Optional[str] = None       # e.g. "25-34"
    tech_literacy: Optional[int] = None   # 1-10
    primary_goal: Optional[str] = None    # what the user wants to accomplish
    device: Optional[str] = "desktop"     # desktop | mobile | tablet
    domain_familiarity: Optional[str] = None  # e.g. "first time", "occasional user"
    frustration_tolerance: Optional[str] = None  # low | medium | high


class EvaluationRequest(BaseModel):
    target_url: str
    task_description: str = "Explore the website and find the main purpose."
    persona: PersonaModel = PersonaModel()



@app.get("/")
def read_root():
    return {"message": "Zynx Backend Running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/evaluate")
async def evaluate(request: EvaluationRequest):
    """
    Streams a LangGraph UX evaluation run as Server-Sent Events (SSE).
    Each event is a JSON payload representing a completed graph node.
    """
    logger.info(f"Received evaluation request for URL: {request.target_url}")

    try:
        from .orchestrator.runner import run_evaluation_stream
    except ImportError:
        from orchestrator.runner import run_evaluation_stream

    run_id = str(uuid.uuid4())[:8]

    async def event_stream():
        try:
            async for payload in run_evaluation_stream(
                run_id=run_id,
                target_url=request.target_url,
                task_description=request.task_description,
                persona=request.persona.model_dump(),
            ):
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.02)
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
