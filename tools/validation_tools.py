"""
Validation Agent Tools.

Tools for source validation, contradiction detection, and confidence scoring.
Uses OpenAI for validation analysis.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger("validation_tools")


class ValidateSourcesInput(BaseModel):
    """Input schema for source validation."""
    documents: List[Dict[str, Any]] = Field(
        description="Documents to validate")
    insights: List[str] = Field(
        description="Insights to validate against sources")


class CheckContradictionsInput(BaseModel):
    """Input schema for contradiction detection."""
    documents: List[Dict[str, Any]] = Field(description="Documents to check")
    insights: List[str] = Field(
        description="Insights to check for contradictions")


class CalculateConfidenceInput(BaseModel):
    """Input schema for confidence calculation."""
    num_sources: int = Field(description="Number of sources")
    validation_results: Dict[str, Any] = Field(
        description="Validation results")


@tool(args_schema=ValidateSourcesInput)
async def validate_sources_tool(
    documents: List[Dict[str, Any]],
    insights: List[str]
) -> Dict[str, Any]:
    """
    Validate sources and assess their credibility using OpenAI.

    Args:
        documents: List of source documents
        insights: Insights to validate

    Returns:
        Dictionary with credibility scores and validation results
    """
    logger.info(f"Validating {len(documents)} sources")

    if not documents:
        return {
            "credibility_scores": {},
            "validation_summary": "No sources to validate"
        }

    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,  # Lower temperature for validation
            api_key=settings.openai_api_key
        )

        # Prepare source info
        sources_text = "\n".join([
            f"- Source {i+1}: {doc.get('title', 'Untitled')} ({doc.get('source_url', 'N/A')})"
            for i, doc in enumerate(documents[:5])
        ])

        insights_text = "\n".join([f"- {insight}" for insight in insights[:5]])

        prompt = f"""Validate these sources and insights:

SOURCES:
{sources_text}

INSIGHTS TO VALIDATE:
{insights_text}

Assess:
1. Are the sources credible? (mention any concerns)
2. Do the insights align with the sources?
3. Any red flags or limitations?

Provide a brief validation summary."""

        response = await llm.ainvoke(prompt)

        # Calculate credibility scores
        credibility_scores = {}
        for i, doc in enumerate(documents):
            source_id = doc.get('source_url', f'source_{i}')
            # Base credibility (can be enhanced with actual checks)
            credibility_scores[source_id] = 0.8

        return {
            "credibility_scores": credibility_scores,
            "validation_summary": response.content,
            "sources_validated": len(documents)
        }

    except Exception as e:
        logger.error(f"Source validation error: {str(e)}")
        return {
            "credibility_scores": {},
            "validation_summary": "Validation error",
            "error": str(e)
        }


@tool(args_schema=CheckContradictionsInput)
async def check_contradictions_tool(
    documents: List[Dict[str, Any]],
    insights: List[str]
) -> Dict[str, Any]:
    """
    Check for contradictions or inconsistencies using OpenAI.

    Args:
        documents: Documents to check
        insights: Insights to check for contradictions

    Returns:
        Dictionary with detected contradictions
    """
    logger.info(f"Checking for contradictions in {len(insights)} insights")

    if not insights or not documents:
        return {"contradictions": []}

    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,
            api_key=settings.openai_api_key
        )

        insights_text = "\n".join([f"- {insight}" for insight in insights[:5]])
        doc_snippets = "\n".join([
            f"Doc {i+1}: {doc.get('content', doc.get('snippet', ''))[:200]}"
            for i, doc in enumerate(documents[:3])
        ])

        prompt = f"""Check for contradictions or inconsistencies:

INSIGHTS:
{insights_text}

DOCUMENTS:
{doc_snippets}

Are there any:
1. Contradictions between insights?
2. Contradictions between insights and documents?
3. Logical inconsistencies?

List any contradictions found (or say "None detected")."""

        response = await llm.ainvoke(prompt)

        # Parse contradictions
        contradictions = []
        content_lower = response.content.lower()

        if "none" not in content_lower and "no contradiction" not in content_lower:
            # Found contradictions
            for line in response.content.split('\n'):
                line = line.strip()
                if line and len(line) > 20:
                    contradictions.append({
                        "description": line.lstrip('-*•0123456789. '),
                        "severity": "low"
                    })

        return {
            "contradictions": contradictions,
            "contradiction_check_summary": response.content
        }

    except Exception as e:
        logger.error(f"Contradiction check error: {str(e)}")
        return {"contradictions": []}


@tool(args_schema=CalculateConfidenceInput)
def calculate_confidence_tool(
    num_sources: int,
    validation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate overall confidence score based on validation results.

    Args:
        num_sources: Number of sources analyzed
        validation_results: Results from validation checks

    Returns:
        Dictionary with confidence score and explanation
    """
    logger.info("Calculating confidence score")

    # Base confidence on number of sources
    base_confidence = min(0.5 + (num_sources * 0.08), 0.9)

    # Adjust based on contradictions
    contradictions = validation_results.get("contradictions", [])
    if contradictions:
        penalty = min(len(contradictions) * 0.1, 0.3)
        base_confidence -= penalty

    # Adjust based on credibility scores
    credibility_scores = validation_results.get("credibility_scores", {})
    if credibility_scores:
        avg_credibility = sum(credibility_scores.values()
                              ) / len(credibility_scores)
        base_confidence = (base_confidence + avg_credibility) / 2

    # Clamp between 0 and 1
    confidence = max(0.0, min(1.0, base_confidence))

    explanation = f"Confidence based on {num_sources} sources"
    if contradictions:
        explanation += f", {len(contradictions)} contradictions detected"

    return {
        "confidence_score": confidence,
        "explanation": explanation,
        "sources_analyzed": num_sources
    }
