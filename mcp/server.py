import asyncio
from typing import Sequence
from mcp.fastmcp import FastMCP
from mcp.types import TextContent
from mcp.server.stdio import stdio_server

from graph.graph import create_research_graph
from database.vector_store import VectorStore
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("mcp_server")

mcp = FastMCP("research-assistant-mcp")


@mcp.tool()
async def research_query(query: str, context: str = "") -> str:
    """
    Execute a research query using a multi-agent workflow.

    Args:
        query (str): Research question to investigate. Minimum 10 characters.
        context (str, optional): Additional context or constraints to narrow the search. Defaults to "".

    Returns:
        str: A detailed research report as a string.
    """
    logger.info(f"Executing research query: {query}")
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
    return result.get("final_report", "No report generated")


@mcp.tool()
async def search_documents(query: str, max_results: int = 10) -> str:
    """
    Perform semantic search in the vector store using vector embeddings.

    Args:
        query (str): Search query. Minimum 3 characters.
        max_results (int, optional): Maximum number of results to return. Defaults to 10.

    Returns:
        str: A formatted string listing matching documents with titles, similarity scores, and excerpts.
    """
    logger.info(f"Searching documents: {query}")
    vector_store = VectorStore()
    results = await vector_store.search(query=query, max_results=max_results)

    response = f"# Search Results: {query}\n\nFound {len(results)} documents\n\n"

    for i, result in enumerate(results, 1):
        response += f"## {i}. {result.get('title', 'Untitled')}\n"
        response += f"**Similarity:** {result.get('similarity', 0):.2%}\n"
        response += f"{result.get('content', '')[:300]}...\n\n"

    return response


async def main():
    """Start the MCP server using stdio transport."""
    logger.info("Starting MCP Server with FastMCP...")
    logger.info(
        "Run Inspector with: npx @modelcontextprotocol/inspector python -m mcp.server")

    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
