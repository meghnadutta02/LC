"""
LangGraph Workflow Definition.

Creates the multi-agent research workflow using LangGraph 1.0.
"""

from langgraph.graph import StateGraph, END
from typing import Literal

from graph.state import ResearchState
from graph.nodes import (
    coordinator_node,
    retrieval_node,
    analysis_node,
    validation_node,
    formatting_node
)
from utils.logger import setup_logger

logger = setup_logger("workflow_graph")


def should_continue(state: ResearchState) -> Literal["retrieval", "end"]:
    """
    Conditional edge: determine if workflow should continue.

    Args:
        state: Current workflow state

    Returns:
        Next node name or "end"
    """
    if state["current_step"] == "completed":
        return "end"

    if state["iteration"] >= state["max_iterations"]:
        logger.warning(f"Max iterations ({state['max_iterations']}) reached")
        return "end"

    return "retrieval"


def create_research_graph() -> StateGraph:
    """
    Create the LangGraph workflow for the research assistant.

    Workflow:
        START → Coordinator → Retrieval → Analysis → Validation → Formatting → END

    Returns:
        Compiled StateGraph
    """
    logger.info("Creating research workflow graph")

    # Create graph
    workflow = StateGraph(ResearchState)

    # Add nodes (agents)
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("formatting", formatting_node)

    # Define edges (workflow flow)
    workflow.set_entry_point("coordinator")

    workflow.add_edge("coordinator", "retrieval")
    workflow.add_edge("retrieval", "analysis")
    workflow.add_edge("analysis", "validation")
    workflow.add_edge("validation", "formatting")
    workflow.add_edge("formatting", END)

    # Compile graph
    app = workflow.compile()

    logger.info("Research workflow graph compiled successfully")

    return app
