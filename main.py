"""
FastAPI application with /assistant endpoint.

Exposes the research assistant as a simple REST API.

Usage:
    python main.py
    
    Then POST to: http://localhost:8000/assistant
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn

from graph.graph import create_research_graph
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("fastapi_server")


class AssistantRequest(BaseModel):
    """Request model for /assistant endpoint."""
    query: str = Field(..., min_length=10, description="Research query")
    context: str = Field(default="", description="Additional context")


class AssistantResponse(BaseModel):
    """Response model for /assistant endpoint."""
    query: str
    final_report: str
    executive_summary: str
    confidence_score: float
    citations_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting FastAPI Research Assistant...")
    yield
    logger.info("Shutting down FastAPI Research Assistant...")


# Create FastAPI app
app = FastAPI(
    title="Research Assistant API",
    description="Multi-agent research assistant powered by LangGraph 1.0",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Research Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "assistant": "POST /assistant",
            "health": "GET /health",
            "docs": "/docs"
        },
        "mcp_server": "Use: python -m mcp.server"
    }


@app.post("/assistant", response_model=AssistantResponse)
async def assistant_endpoint(request: AssistantRequest):
    """
    Main assistant endpoint.

    Executes a research query using the multi-agent workflow.

    Example:
        POST /assistant
        {
            "query": "What are the latest trends in AI safety?",
            "context": "Focus on 2024-2025"
        }
    """
    logger.info(f"Assistant request received: {request.query}")

    try:
        # Create LangGraph workflow
        graph = create_research_graph()

        # Initial state
        initial_state = {
            "query": request.query,
            "context": request.context,
            "workflow_id": f"api_{id(request)}",
            "max_iterations": settings.max_iterations,
            "messages": [],
            "current_step": "coordinator"
        }

        # Execute workflow
        config = {"recursion_limit": settings.recursion_limit}
        result = await graph.ainvoke(initial_state, config)

        return AssistantResponse(
            query=request.query,
            final_report=result.get("final_report", ""),
            executive_summary=result.get("executive_summary", ""),
            confidence_score=result.get("confidence_score", 0.0),
            citations_count=len(result.get("citations", []))
        )

    except Exception as e:
        logger.error(f"Error in assistant endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        from database.connection import get_db_connection
        conn = get_db_connection()
        conn.close()
        db_status = "connected"
    except:
        db_status = "disconnected"

    return {
        "status": "healthy",
        "database": db_status,
        "openai_configured": bool(settings.openai_api_key)
    }


if __name__ == "__main__":
    logger.info(
        f"Starting FastAPI server on {settings.api_host}:{settings.api_port}")
    logger.info(
        f"API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info(
        f"Assistant endpoint: POST http://{settings.api_host}:{settings.api_port}/assistant")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
