"""
PostgreSQL checkpointing for LangGraph.

This allows workflow state to be persisted and recovered.
"""

from langgraph.checkpoint.postgres import PostgresSaver
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("checkpoints")


def create_checkpoint_saver() -> PostgresSaver:
    """
    Create PostgreSQL checkpoint saver for LangGraph.

    This enables:
    - Workflow state persistence
    - Recovery from failures
    - Time-travel debugging

    Returns:
        PostgresSaver instance
    """
    logger.info("Creating PostgreSQL checkpoint saver")

    saver = PostgresSaver.from_conn_string(settings.db_url)

    logger.info("Checkpoint saver created successfully")

    return saver


def create_research_graph_with_checkpoints():
    """
    Create research graph with PostgreSQL checkpointing enabled.

    Usage:
        graph = create_research_graph_with_checkpoints()
        result = await graph.ainvoke(state, config={"thread_id": "my-thread"})
    """
    from graph.graph import create_research_graph

    graph = create_research_graph()
    checkpointer = create_checkpoint_saver()

    # Compile with checkpointing
    app = graph.compile(checkpointer=checkpointer)

    logger.info("Research graph with checkpointing ready")

    return app
