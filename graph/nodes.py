"""
LangGraph Node Functions.

Each function represents an agent in the workflow.
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI

from graph.state import ResearchState
from agents.coordinator import CoordinatorAgent
from agents.retrieval import RetrievalAgent
from agents.analysis import AnalysisAgent
from agents.validation import ValidationAgent
from agents.formatting import FormattingAgent
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("graph_nodes")


def create_llm() -> ChatOpenAI:
    """Create OpenAI LLM instance."""
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens,
        api_key=settings.openai_api_key
    )


async def coordinator_node(state: ResearchState) -> Dict[str, Any]:
    """
    Coordinator Agent Node.

    Decomposes the query and plans execution.
    """
    logger.info(f"[Coordinator] Processing query: {state['query']}")

    llm = create_llm()
    agent = CoordinatorAgent(llm)

    result = await agent.process(state)

    return {
        "current_step": "retrieval",
        "decomposed_tasks": result["decomposed_tasks"],
        "execution_plan": result["execution_plan"],
        "messages": [f"Coordinator: Decomposed into {len(result['decomposed_tasks'])} tasks"]
    }


async def retrieval_node(state: ResearchState) -> Dict[str, Any]:
    """
    Retrieval Agent Node.

    Searches for and retrieves relevant documents, stores them with embeddings.
    """
    logger.info(f"[Retrieval] Searching for: {state['query']}")

    llm = create_llm()
    agent = RetrievalAgent(llm, state["workflow_id"])

    result = await agent.process(state)

    return {
        "current_step": "analysis",
        "retrieved_documents": result["documents"],
        "search_queries": result["queries"],
        "document_ids": result["document_ids"],
        "messages": [f"Retrieval: Found {len(result['documents'])} documents"]
    }


async def analysis_node(state: ResearchState) -> Dict[str, Any]:
    """
    Analysis Agent Node.

    Performs deep analysis on retrieved documents.
    """
    logger.info(
        f"[Analysis] Analyzing {len(state['retrieved_documents'])} documents")

    llm = create_llm()
    agent = AnalysisAgent(llm)

    result = await agent.process(state)

    return {
        "current_step": "validation",
        "analysis_results": result["analysis"],
        "comparative_findings": result["comparisons"],
        "trends": result["trends"],
        "insights": result["insights"],
        "messages": [f"Analysis: Generated {len(result['insights'])} insights"]
    }


async def validation_node(state: ResearchState) -> Dict[str, Any]:
    """
    Validation Agent Node.

    Validates findings and checks credibility.
    """
    logger.info("[Validation] Validating findings")

    llm = create_llm()
    agent = ValidationAgent(llm)

    result = await agent.process(state)

    return {
        "current_step": "formatting",
        "validation_results": result["validation"],
        "credibility_scores": result["credibility"],
        "confidence_score": result["confidence"],
        "contradictions": result["contradictions"],
        "messages": [f"Validation: Confidence score {result['confidence']:.2%}"]
    }


async def formatting_node(state: ResearchState) -> Dict[str, Any]:
    """
    Formatting Agent Node.

    Formats the final report with citations.
    """
    logger.info("[Formatting] Creating final report")

    llm = create_llm()
    agent = FormattingAgent(llm)

    result = await agent.process(state)

    return {
        "current_step": "completed",
        "final_report": result["report"],
        "executive_summary": result["summary"],
        "citations": result["citations"],
        "messages": [f"Formatting: Report generated with {len(result['citations'])} citations"]
    }
