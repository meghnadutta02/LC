"""
LangGraph State Schema for Research Assistant.

Using TypedDict with Annotated for proper state management in LangGraph 1.0.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from datetime import datetime
import operator


class ResearchState(TypedDict):
    """
    State schema for the multi-agent research workflow.

    This state is shared across all agent nodes in the LangGraph.
    Annotated fields use operators to define how state updates are merged.
    """

    # === Input ===
    query: str  # Original research query
    context: str  # Additional context

    # === Workflow Control ===
    workflow_id: str  # Unique workflow identifier
    current_step: str  # Current agent step
    iteration: int  # Current iteration count
    max_iterations: int  # Maximum allowed iterations

    # === Coordinator Agent ===
    decomposed_tasks: List[Dict[str, Any]]  # Sub-tasks from decomposition
    execution_plan: Dict[str, Any]  # Task execution order

    # === Retrieval Agent ===
    # Retrieved docs
    retrieved_documents: Annotated[List[Dict[str, Any]], operator.add]
    search_queries: List[str]  # Search queries executed
    document_ids: List[int]  # IDs of stored documents in pgvector

    # === Analysis Agent ===
    analysis_results: Dict[str, Any]  # Analysis outputs
    comparative_findings: List[Dict[str, Any]]  # Comparative analysis
    trends: List[Dict[str, Any]]  # Identified trends
    insights: Annotated[List[str], operator.add]  # Key insights

    # === Validation Agent ===
    validation_results: Dict[str, Any]  # Validation outcomes
    credibility_scores: Dict[str, float]  # Source credibility
    confidence_score: float  # Overall confidence (0-1)
    contradictions: List[Dict[str, Any]]  # Identified contradictions

    # === Formatting Agent ===
    final_report: str  # Formatted final output
    executive_summary: str  # Executive summary
    citations: List[Dict[str, Any]]  # Citation list

    # === Messages & Logs ===
    messages: Annotated[List[str], operator.add]  # Workflow messages
    errors: Annotated[List[str], operator.add]  # Error messages

    # === Metadata ===
    started_at: Optional[datetime]  # Workflow start time
    completed_at: Optional[datetime]  # Workflow completion time


def create_initial_state(
    query: str,
    context: str = "",
    workflow_id: str = "",
    max_iterations: int = 25
) -> ResearchState:
    """
    Create initial state for a new research workflow.

    Args:
        query: Research query
        context: Additional context
        workflow_id: Unique workflow ID
        max_iterations: Maximum iterations

    Returns:
        Initial ResearchState
    """
    return ResearchState(
        # Input
        query=query,
        context=context,

        # Workflow Control
        workflow_id=workflow_id,
        current_step="coordinator",
        iteration=0,
        max_iterations=max_iterations,

        # Coordinator
        decomposed_tasks=[],
        execution_plan={},

        # Retrieval
        retrieved_documents=[],
        search_queries=[],
        document_ids=[],

        # Analysis
        analysis_results={},
        comparative_findings=[],
        trends=[],
        insights=[],

        # Validation
        validation_results={},
        credibility_scores={},
        confidence_score=0.0,
        contradictions=[],

        # Formatting
        final_report="",
        executive_summary="",
        citations=[],

        # Messages & Logs
        messages=[],
        errors=[],

        # Metadata
        started_at=datetime.now(),
        completed_at=None
    )
