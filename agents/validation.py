"""
Validation Agent - Agent 4

Responsible for:
- Fact-checking
- Source credibility assessment
- Contradiction detection
- Confidence scoring
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState
from utils.logger import setup_logger

logger = setup_logger("validation_agent")


class ValidationAgent:
    """
    Validation Agent ensures accuracy and credibility of findings.
    """

    def __init__(self, llm: ChatOpenAI):
        """Initialize validation agent."""
        self.llm = llm
        self.system_prompt = """You are a Research Validation AI Agent.

Your responsibilities:
1. Assess the credibility of sources
2. Identify contradictions or inconsistencies
3. Cross-reference claims
4. Assign confidence scores

Be critical and thorough. Flag any questionable claims or weak sources."""

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        Validate research findings.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with validation results
        """
        query = state["query"]
        documents = state.get("retrieved_documents", [])
        insights = state.get("insights", [])

        logger.info(f"Validation agent validating {len(insights)} insights")

        if not documents or not insights:
            return self._default_validation()

        # Perform validation
        validation_results = await self._validate_findings(
            query, documents, insights
        )

        logger.info(
            f"Validation complete with confidence: {validation_results['confidence']:.2%}")

        return validation_results

    async def _validate_findings(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        insights: List[str]
    ) -> Dict[str, Any]:
        """Validate findings against documents."""

        # Prepare content
        insights_text = "\n".join([f"- {insight}" for insight in insights])
        sources_text = "\n".join([
            f"Source {i+1}: {doc.get('title', 'Untitled')} ({doc.get('source_url', 'N/A')})"
            for i, doc in enumerate(documents[:5])
        ])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Research Query: {query}

Sources:
{sources_text}

Key Insights to Validate:
{insights_text}

Please:
1. Assess the credibility of each source (score 0-1)
2. Identify any contradictions or inconsistencies
3. Provide an overall confidence score (0-1) for the findings
4. List any concerns or limitations

Be thorough and critical.""")
        ]

        response = await self.llm.ainvoke(messages)

        # Parse validation results
        credibility = self._assess_credibility(documents, response.content)
        contradictions = self._detect_contradictions(response.content)
        confidence = self._calculate_confidence(
            response.content, len(documents))

        return {
            "validation": {
                "summary": response.content,
                "sources_checked": len(documents)
            },
            "credibility": credibility,
            "contradictions": contradictions,
            "confidence": confidence
        }

    def _assess_credibility(
        self,
        documents: List[Dict[str, Any]],
        validation_text: str
    ) -> Dict[str, float]:
        """Assess source credibility."""
        credibility = {}

        for i, doc in enumerate(documents):
            source = doc.get("source_url", f"source_{i}")
            # Simple heuristic: all sources get 0.8 unless flagged
            if "questionable" in validation_text.lower() or "unreliable" in validation_text.lower():
                credibility[source] = 0.6
            else:
                credibility[source] = 0.8

        return credibility

    def _detect_contradictions(self, validation_text: str) -> List[Dict[str, Any]]:
        """Detect contradictions from validation."""
        contradictions = []

        # Look for contradiction keywords
        if any(word in validation_text.lower() for word in ["contradict", "inconsistent", "conflict"]):
            contradictions.append({
                "type": "content_inconsistency",
                "description": "Potential contradictions detected in validation"
            })

        return contradictions

    def _calculate_confidence(self, validation_text: str, num_sources: int) -> float:
        """Calculate overall confidence score."""

        # Base confidence on number of sources
        base_confidence = min(0.5 + (num_sources * 0.1), 0.9)

        # Adjust based on validation text
        text_lower = validation_text.lower()

        if "high confidence" in text_lower or "reliable" in text_lower:
            base_confidence += 0.1
        elif "low confidence" in text_lower or "unreliable" in text_lower:
            base_confidence -= 0.2

        if "concerns" in text_lower or "limitations" in text_lower:
            base_confidence -= 0.1

        # Clamp between 0 and 1
        return max(0.0, min(1.0, base_confidence))

    def _default_validation(self) -> Dict[str, Any]:
        """Return default validation when no data."""
        return {
            "validation": {"summary": "No findings to validate"},
            "credibility": {},
            "contradictions": [],
            "confidence": 0.5
        }
