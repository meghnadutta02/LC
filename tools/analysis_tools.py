"""
Analysis Agent Tools.

Tools for document analysis, trend identification, and comparison.
Uses OpenAI for analysis.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("analysis_tools")


class AnalyzeDocumentsInput(BaseModel):
    """Input schema for document analysis."""
    documents: List[Dict[str, Any]] = Field(description="Documents to analyze")
    query: str = Field(description="Research query")


class IdentifyTrendsInput(BaseModel):
    """Input schema for trend identification."""
    documents: List[Dict[str, Any]] = Field(description="Documents to analyze")
    query: str = Field(description="Research query")


class ComparativeAnalysisInput(BaseModel):
    """Input schema for comparative analysis."""
    documents: List[Dict[str, Any]] = Field(description="Documents to compare")
    aspects: List[str] = Field(default=[], description="Aspects to compare")


@tool(args_schema=AnalyzeDocumentsInput)
async def analyze_documents_tool(
    documents: List[Dict[str, Any]],
    query: str
) -> Dict[str, Any]:
    """
    Analyze documents to extract key insights using OpenAI.

    Args:
        documents: List of documents to analyze
        query: Research query for context

    Returns:
        Dictionary with analysis results and insights
    """
    logger.info(f"Analyzing {len(documents)} documents")

    if not documents:
        return {"insights": [], "analysis": "No documents to analyze"}

    try:
        # Prepare document summaries
        doc_texts = []
        for i, doc in enumerate(documents[:3]):  # Limit to 3 docs
            text = doc.get("content", doc.get("snippet", ""))
            doc_texts.append(f"Document {i+1}: {text[:500]}")

        combined_text = "\n\n".join(doc_texts)

        # Use OpenAI for analysis
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

        prompt = f"""Analyze these documents related to: "{query}"

{combined_text}

Extract 5-7 key insights. Format as bullet points."""

        response = await llm.ainvoke(prompt)

        # Parse insights
        insights = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line.startswith(('-', '*', '•')) or (line and line[0].isdigit()):
                insight = line.lstrip('-*•0123456789. ').strip()
                if len(insight) > 20:
                    insights.append(insight)

        return {
            "insights": insights,
            "analysis": response.content,
            "documents_analyzed": len(documents)
        }

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "insights": ["Analysis completed with limited information"],
            "analysis": "Error in analysis",
            "error": str(e)
        }


@tool(args_schema=IdentifyTrendsInput)
async def identify_trends_tool(
    documents: List[Dict[str, Any]],
    query: str
) -> Dict[str, Any]:
    """
    Identify trends and patterns across documents using OpenAI.

    Args:
        documents: Documents to analyze for trends
        query: Research query

    Returns:
        Dictionary with identified trends
    """
    logger.info(f"Identifying trends in {len(documents)} documents")

    if not documents:
        return {"trends": []}

    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

        # Simplified trend analysis
        doc_texts = [doc.get("content", doc.get("snippet", ""))[
            :300] for doc in documents[:3]]
        combined = "\n".join(doc_texts)

        prompt = f"""Identify 3 key trends or patterns in these documents about "{query}":

{combined}

List trends briefly."""

        response = await llm.ainvoke(prompt)

        trends = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                trends.append({"trend": line.lstrip('-*•0123456789. ')})

        return {"trends": trends[:3]}

    except Exception as e:
        logger.error(f"Trend identification error: {str(e)}")
        return {"trends": []}


@tool(args_schema=ComparativeAnalysisInput)
async def comparative_analysis_tool(
    documents: List[Dict[str, Any]],
    aspects: List[str] = []
) -> Dict[str, Any]:
    """
    Perform comparative analysis across documents using OpenAI.

    Args:
        documents: Documents to compare
        aspects: Specific aspects to compare

    Returns:
        Dictionary with comparative findings
    """
    logger.info(f"Comparing {len(documents)} documents")

    if len(documents) < 2:
        return {"comparisons": ["Need at least 2 documents for comparison"]}

    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

        doc_summaries = [
            f"Doc {i+1}: {doc.get('content', doc.get('snippet', ''))[:200]}"
            for i, doc in enumerate(documents[:2])
        ]

        prompt = f"""Compare these documents:

{chr(10).join(doc_summaries)}

Identify key similarities and differences."""

        response = await llm.ainvoke(prompt)

        return {
            "comparisons": [response.content],
            "documents_compared": len(documents)
        }

    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return {"comparisons": []}
