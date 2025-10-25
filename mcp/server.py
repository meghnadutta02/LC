"""
MCP Server for Research Assistant.

Exposes the assistant as MCP tools for use with MCP Inspector.

Usage:
    python -m mcp.server
    
    Or with MCP Inspector:
    npx @modelcontextprotocol/inspector python -m mcp.server
"""

import asyncio
from typing import Any, Sequence
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.stdio import stdio_server

from graph.graph import create_research_graph
from database.vector_store import VectorStore
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("mcp_server")

# Create MCP server
app = Server("research-assistant-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="research_query",
            description="Execute a research query using multi-agent workflow (5 specialized agents)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research question to investigate",
                        "minLength": 10
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context or constraints",
                        "default": ""
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_documents",
            description="Semantic search using vector embeddings in pgvector",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                        "minLength": 3
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (1-50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool execution."""
    logger.info(f"MCP Tool called: {name}")

    try:
        if name == "research_query":
            return await execute_research(arguments)
        elif name == "search_documents":
            return await search_documents(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in MCP tool {name}: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def execute_research(arguments: dict) -> Sequence[TextContent]:
    """Execute research query."""
    query = arguments.get("query")
    context = arguments.get("context", "")

    logger.info(f"Executing research: {query}")

    graph = create_research_graph()

    initial_state = {
        "query": query,
        "context": context,
        "workflow_id": f"mcp_{asyncio.get_event_loop().time()}",
        "max_iterations": settings.max_iterations,
        "messages": [],
        "current_step": "coordinator"
    }

    config = {"recursion_limit": settings.recursion_limit}
    result = await graph.ainvoke(initial_state, config)

    report = result.get("final_report", "No report generated")
    summary = result.get("executive_summary", "")

    response = f"""# Research Report

## Query
{query}

## Summary
{summary}

## Full Report
{report}

## Confidence
{result.get('confidence_score', 0):.2%}
"""

    return [TextContent(type="text", text=response)]


async def search_documents(arguments: dict) -> Sequence[TextContent]:
    """Search documents using vector similarity."""
    query = arguments.get("query")
    max_results = arguments.get("max_results", 10)

    logger.info(f"Searching documents: {query}")

    vector_store = VectorStore()
    results = await vector_store.search(query=query, max_results=max_results)

    response = f"# Search Results: {query}\n\nFound {len(results)} documents\n\n"

    for i, result in enumerate(results, 1):
        response += f"## {i}. {result.get('title', 'Untitled')}\n"
        response += f"**Similarity:** {result.get('similarity', 0):.2%}\n"
        response += f"{result.get('content', '')[:300]}...\n\n"

    return [TextContent(type="text", text=response)]


async def main():
    """Run MCP server."""
    logger.info("Starting MCP Server...")
    logger.info("Use: npx @modelcontextprotocol/inspector python -m mcp.server")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
