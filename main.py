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
from database.connection import test_connection
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
    logger.info(" Starting FastAPI Research Assistant...")

    # Test database connection at startup
    logger.info(" Testing database connection...")
    if test_connection():
        logger.info(" Database connected successfully!")
    else:
        logger.error(" Database connection failed!")
        logger.error(
            "️  Make sure pgvector is running: podman start pgvector")

    # Check OpenAI configuration
    if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
        logger.info(" OpenAI API key configured")
    else:
        logger.warning("️  OpenAI API key not configured in .env")

    yield

    logger.info(" Shutting down FastAPI Research Assistant...")


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
        "status": "running",
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
    logger.info(f" Assistant request received: {request.query}")

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

        logger.info(
            f" Research completed with confidence: {result.get('confidence_score', 0):.2%}")

        return AssistantResponse(
            query=request.query,
            final_report=result.get("final_report", ""),
            executive_summary=result.get("executive_summary", ""),
            confidence_score=result.get("confidence_score", 0.0),
            citations_count=len(result.get("citations", []))
        )

    except Exception as e:
        logger.error(f" Error in assistant endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check database connection
    db_connected = test_connection()

    # Check OpenAI configuration
    openai_configured = bool(
        settings.openai_api_key and
        settings.openai_api_key != "your_openai_api_key_here"
    )

    status = "healthy" if (db_connected and openai_configured) else "degraded"

    return {
        "status": status,
        "database": "connected" if db_connected else "disconnected",
        "openai": "configured" if openai_configured else "not_configured",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    logger.info(
        f" Starting FastAPI server on {settings.api_host}:{settings.api_port}")
    logger.info(
        f" API Docs: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info(
        f" Assistant endpoint: POST http://{settings.api_host}:{settings.api_port}/assistant")
    logger.info(
        f" Health check: GET http://{settings.api_host}:{settings.api_port}/health")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
