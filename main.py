"""
Main FastAPI app with integrated MCP protocol server.

- HTTP REST API available for /assistant
- MCP server mounted under /mcp path
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
from mcp.server import Server
import asyncio

from graph.graph import create_research_graph
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("fastapi_mcp_app")

# Create MCP server instance
mcp_server = Server("research-assistant")


@mcp_server.call_tool()
async def research_query(query: str, context: str = "") -> str:
    logger.info(f"[MCP TOOL] research_query called with query: {query}")
    try:
        graph = create_research_graph()
        initial_state = {
            "query": query,
            "context": context,
            "workflow_id": f"mcp_{id(query)}",
            "max_iterations": settings.max_iterations,
            "messages": [],
            "current_step": "coordinator"
        }
        config = {"recursion_limit": settings.recursion_limit}
        result = await graph.ainvoke(initial_state, config)
        return result.get("final_report", "No report generated")
    except Exception as e:
        logger.error(f"[MCP TOOL] Error in research_query: {str(e)}")
        return f"Error: {str(e)}"


@mcp_server.list_tools()
async def list_tools():
    return [
        {"name": "research_query",
         "description": "Run a research query, return full report."}
    ]


# Create main FastAPI app
app = FastAPI(
    title="Research Assistant API with MCP Protocol",
    description="Multi-agent research assistant via FastAPI with integrated MCP support",
    version="1.0.0",
)

# CORS middleware for MCP Inspector and browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount MCP server under /mcp path
app.mount("/mcp", mcp_server.asgi_app())


class Citation(BaseModel):
    id: int
    title: str
    url: str
    preview: str


class AssistantRequest(BaseModel):
    query: str = Field(..., min_length=10, description="Research query")
    context: str = Field("", description="Additional context or constraints")


class AssistantResponse(BaseModel):
    query: str
    final_report: str
    citations: List[Citation] = []


@app.post("/assistant", response_model=AssistantResponse)
async def assistant_endpoint(request: AssistantRequest):
    logger.info(f"[HTTP API] Research query received: {request.query}")
    try:
        graph = create_research_graph()
        initial_state = {
            "query": request.query,
            "context": request.context,
            "workflow_id": f"http_{id(request)}",
            "max_iterations": settings.max_iterations,
            "messages": [],
            "current_step": "coordinator"
        }
        config = {"recursion_limit": settings.recursion_limit}
        result = await graph.ainvoke(initial_state, config)

        documents = result.get("retrieved_documents", [])
        citations = []
        for i, doc in enumerate(documents, 1):
            citations.append({
                "id": i,
                "title": doc.get("title", "Untitled Document"),
                "url": doc.get("source_url", "N/A"),
                "preview": (doc.get("content", "")[:150] + "...") if doc.get("content") else ""
            })

        return AssistantResponse(
            query=request.query,
            final_report=result.get("final_report", ""),
            citations=citations
        )
    except Exception as e:
        logger.error(f"[HTTP API] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    logger.info(
        f"Starting FastAPI + MCP server on {settings.api_host}:{settings.api_port}")
    uvicorn.run("main:app", host=settings.api_host,
                port=settings.api_port, reload=True)
