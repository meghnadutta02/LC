"""
Main FastAPI app with integrated MCP protocol server.

- HTTP REST API available for /assistant
- MCP server mounted under /mcp path
"""

from graph.nodes import create_llm
from agents.coordinator import CoordinatorAgent
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import WebSocket, APIRouter

from mcp.server import Server
import asyncio

from graph.graph import create_research_graph
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("fastapi_mcp_app")

# Create MCP server instance
mcp_server = Server("research-assistant")

ws_router = APIRouter()

active_websockets = set()


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


class AssistantRequest(BaseModel):
    query: str = Field(..., min_length=10, description="Research query")
    context: str = Field("", description="Additional context or constraints")


class AssistantResponse(BaseModel):
    query: str
    final_report: str


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
        return AssistantResponse(query=request.query, final_report=result.get("final_report", ""))
    except Exception as e:
        logger.error(f"[HTTP API] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@ws_router.websocket("/ws/agent-updates")
async def agent_updates_ws(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive (no-op, for now)
    except Exception:
        pass
    finally:
        active_websockets.discard(websocket)
coordinator_agent = CoordinatorAgent(create_llm())


@ws_router.websocket("/ws/agent-updates")
async def agent_updates_ws(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("action") == "human_feedback":
                # Process human feedback asynchronously here
                coordinator_agent.receive_human_feedback(message)
                logger.info(f"Human feedback received: {message}")
    except Exception:
        pass
    finally:
        active_websockets.discard(websocket)


async def broadcast_agent_update(msg: dict):
    # Use 'active_websockets' if using the router version, else inject set from global app context
    dead_websockets = set()
    for ws in active_websockets:
        try:
            await ws.send_json(msg)
        except Exception:
            dead_websockets.add(ws)
    for ws in dead_websockets:
        active_websockets.discard(ws)


if __name__ == "__main__":
    import uvicorn
    logger.info(
        f"Starting FastAPI + MCP server on {settings.api_host}:{settings.api_port}")
    uvicorn.run("main:app", host=settings.api_host,
                port=settings.api_port, reload=True)
