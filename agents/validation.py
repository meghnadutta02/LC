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
from main import broadcast_agent_update
from graph.state import ResearchState
from utils.logger import setup_logger

logger = setup_logger("validation_agent")


class ValidationAgent:
    """Validation Agent ensures accuracy and credibility."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize validation agent."""
        self.llm = llm
        self.system_prompt = """You are a Research Validation AI Agent.

Your responsibilities:
1. Assess source credibility thoroughly
2. Identify contradictions or inconsistencies
3. Cross-reference claims
4. Assign confidence scores based on evidence quality

Be critical, thorough, and objective."""

        logger.info(" Validation Agent initialized")

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """Validate research findings."""

        query = state["query"]
        documents = state.get("retrieved_documents", [])
        insights = state.get("insights", [])

        logger.info("=" * 80)
        logger.info(" VALIDATION AGENT PROCESSING")
        logger.info(f" Query: {query}")
        logger.info(f" Sources to validate: {len(documents)}")
        logger.info(f" Insights to verify: {len(insights)}")
        logger.info("=" * 80)

        # Real-time: broadcast start
        await broadcast_agent_update({
            "phase": "validation",
            "status": "started",
            "query": query,
            "documents_count": len(documents),
            "insights_count": len(insights)
        })

        if not documents or not insights:
            logger.warning("️  Insufficient data for validation")

            # Real-time: broadcast finish for insufficient data
            await broadcast_agent_update({
                "phase": "validation",
                "status": "finished",
                "result": "insufficient_data"
            })

            return self._default_validation()

        # Log what's being validated
        logger.info(" Validating Insights:")
        for i, insight in enumerate(insights[:5], 1):
            logger.info(f"   {i}. {insight[:80]}...")
        if len(insights) > 5:
            logger.info(f"   ... and {len(insights) - 5} more insights")

        logger.info(" Against Sources:")
        for i, doc in enumerate(documents[:5], 1):
            logger.info(f"   {i}. {doc.get('title', 'Untitled')[:60]}")
            logger.info(f"      URL: {doc.get('source_url', 'N/A')}")

        # Perform validation
        logger.info("🤖 STEP 1: Performing validation analysis...")
        validation_results = await self._validate_findings(query, documents, insights)

        logger.info(f"    Validation complete")
        logger.info(
            f"    Confidence Score: {validation_results['confidence']:.2%}")
        logger.info(
            f"     Contradictions Found: {len(validation_results['contradictions'])}")

        # Log credibility assessment
        logger.info(" Source Credibility Assessment:")
        for source, score in list(validation_results['credibility'].items())[:5]:
            logger.info(f"   • {source[:60]}: {score:.2%}")

        logger.info("=" * 80)
        logger.info(
            f" VALIDATION COMPLETED: Confidence {validation_results['confidence']:.2%}")
        logger.info("=" * 80)

        # Real-time: broadcast finish
        await broadcast_agent_update({
            "phase": "validation",
            "status": "finished",
            "result": {
                "confidence": validation_results['confidence'],
                "contradictions_count": len(validation_results['contradictions']),
                "credibility_summary": validation_results['credibility']
            }
        })

        return validation_results

    async def _validate_findings(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        insights: List[str]
    ) -> Dict[str, Any]:
        """Validate findings against documents."""

        insights_text = "\n".join([f"- {insight}" for insight in insights[:8]])
        sources_text = "\n".join([
            f"Source {i+1}: {doc.get('title', 'Untitled')}\n  URL: {doc.get('source_url', 'N/A')}\n  Preview: {doc.get('content', '')[:150]}..."
            for i, doc in enumerate(documents[:5])
        ])

        logger.info("   🤖 Running validation with LLM...")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Research Query: {query}

SOURCES TO VALIDATE:
{sources_text}

INSIGHTS TO VERIFY:
{insights_text}

Please:
1. Assess each source's credibility (note any concerns)
2. Verify if insights are supported by sources
3. Identify any contradictions or inconsistencies
4. Provide overall confidence score (0-1)
5. List any limitations or concerns

Be thorough and critical in your assessment.""")
        ]

        response = await self.llm.ainvoke(messages)

        logger.info(
            f"    Validation response: {len(response.content)} characters")

        # Parse validation results
        credibility = self._assess_credibility(documents, response.content)
        contradictions = self._detect_contradictions(response.content)
        confidence = self._calculate_confidence(
            response.content, len(documents), credibility)

        logger.info(f"    Assessed {len(credibility)} sources")
        logger.info(f"    Found {len(contradictions)} contradictions")
        logger.info(f"    Calculated confidence: {confidence:.2%}")

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

        validation_lower = validation_text.lower()

        for i, doc in enumerate(documents):
            source = doc.get("source_url", f"source_{i}")

            # Default credibility
            score = 0.8

            # Adjust based on validation text
            if "unreliable" in validation_lower or "questionable" in validation_lower:
                score = 0.6
            elif "reliable" in validation_lower or "credible" in validation_lower:
                score = 0.9

            credibility[source] = score
            logger.debug(f"      → {source[:40]}: {score:.2%}")

        return credibility

    def _detect_contradictions(self, validation_text: str) -> List[Dict[str, Any]]:
        """Detect contradictions from validation."""
        contradictions = []

        content_lower = validation_text.lower()

        contradiction_keywords = ["contradict",
                                  "inconsistent", "conflict", "disagree"]

        if any(word in content_lower for word in contradiction_keywords):
            logger.warning("   ️  Contradictions detected in validation")

            # Extract contradiction details
            lines = validation_text.split('\n')
            for line in lines:
                if any(word in line.lower() for word in contradiction_keywords):
                    contradictions.append({
                        "description": line.strip(),
                        "severity": "medium"
                    })
                    logger.warning(f"      → {line.strip()[:80]}...")
        else:
            logger.info("    No major contradictions detected")

        return contradictions

    def _calculate_confidence(
        self,
        validation_text: str,
        num_sources: int,
        credibility: Dict[str, float]
    ) -> float:
        """Calculate overall confidence score."""

        # Base confidence on sources
        base_confidence = min(0.5 + (num_sources * 0.08), 0.9)
        logger.debug(f"      Base confidence (sources): {base_confidence:.2%}")

        # Adjust for credibility
        if credibility:
            avg_credibility = sum(credibility.values()) / len(credibility)
            base_confidence = (base_confidence + avg_credibility) / 2
            logger.debug(
                f"      After credibility adjustment: {base_confidence:.2%}")

        # Adjust based on validation text
        text_lower = validation_text.lower()

        if "high confidence" in text_lower:
            base_confidence += 0.1
            logger.debug("      Bonus: High confidence mentioned (+0.1)")
        elif "low confidence" in text_lower:
            base_confidence -= 0.2
            logger.debug("      Penalty: Low confidence mentioned (-0.2)")

        if "limitations" in text_lower or "concerns" in text_lower:
            base_confidence -= 0.05
            logger.debug("      Penalty: Limitations noted (-0.05)")

        confidence = max(0.0, min(1.0, base_confidence))

        return confidence

    def _default_validation(self) -> Dict[str, Any]:
        """Return default validation."""
        logger.warning("️  Using default validation (no data)")
        return {
            "validation": {"summary": "Insufficient data for validation"},
            "credibility": {},
            "contradictions": [],
            "confidence": 0.5
        }
