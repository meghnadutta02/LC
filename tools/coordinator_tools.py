"""
Coordinator Agent Tools.

Tools for query decomposition and result synthesis.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.logger import setup_logger

logger = setup_logger("coordinator_tools")


class QueryDecompositionInput(BaseModel):
    """Input schema for query decomposition."""
    query: str = Field(description="Research query to decompose")
    context: str = Field(default="", description="Additional context")


class ResultSynthesisInput(BaseModel):
    """Input schema for result synthesis."""
    results: List[Dict[str, Any]] = Field(
        description="Results from all agents")
    query: str = Field(description="Original query")


@tool(args_schema=QueryDecompositionInput)
def decompose_query_tool(query: str, context: str = "") -> Dict[str, Any]:
    """
    Decompose a complex research query into sub-tasks.

    Analyzes the query and determines which agents are needed.

    Args:
        query: Research query to decompose
        context: Additional context

    Returns:
        Dictionary with sub-tasks and execution plan
    """
    logger.info(f"Decomposing query: {query}")

    # Analyze query
    query_lower = query.lower()

    tasks = []

    # Always need retrieval
    tasks.append({
        "id": "retrieval",
        "agent": "retrieval",
        "description": f"Retrieve information for: {query}",
        "priority": 1
    })

    # Check for analysis keywords
    analysis_keywords = ["analyze", "compare",
                         "trend", "pattern", "relationship"]
    if any(kw in query_lower for kw in analysis_keywords):
        tasks.append({
            "id": "analysis",
            "agent": "analysis",
            "description": f"Analyze information for: {query}",
            "priority": 2
        })

    # Always validate
    tasks.append({
        "id": "validation",
        "agent": "validation",
        "description": f"Validate findings for: {query}",
        "priority": 3
    })

    # Always format
    tasks.append({
        "id": "formatting",
        "agent": "formatting",
        "description": f"Format report for: {query}",
        "priority": 4
    })

    return {
        "tasks": tasks,
        "total_tasks": len(tasks),
        "execution_order": [t["id"] for t in tasks]
    }


@tool(args_schema=ResultSynthesisInput)
def synthesize_results_tool(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Synthesize results from all agents into a coherent summary.

    Args:
        results: List of results from different agents
        query: Original research query

    Returns:
        Synthesized summary
    """
    logger.info(f"Synthesizing {len(results)} results")

    # Combine insights
    all_insights = []
    for result in results:
        if "insights" in result:
            all_insights.extend(result["insights"])

    return {
        "query": query,
        "total_results": len(results),
        "combined_insights": all_insights,
        "summary": f"Synthesized {len(results)} agent results with {len(all_insights)} total insights"
    }
