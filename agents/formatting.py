"""
Formatting Agent - Agent 5

Responsible for:
- Report structuring
- Citation formatting
- Executive summary generation
- Final output formatting
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime

from graph.state import ResearchState
from utils.logger import setup_logger

logger = setup_logger("formatting_agent")


class FormattingAgent:
    """
    Formatting Agent creates the final research report.
    """

    def __init__(self, llm: ChatOpenAI):
        """Initialize formatting agent."""
        self.llm = llm
        self.system_prompt = """You are a Research Formatting AI Agent.

Your responsibilities:
1. Create well-structured research reports
2. Write executive summaries
3. Format citations properly
4. Ensure clarity and professionalism

Output should be:
- Well-organized with clear sections
- Professional and academic tone
- Properly cited
- Easy to read and understand"""

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        Format the final research report.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with formatted report and citations
        """
        query = state["query"]
        documents = state.get("retrieved_documents", [])
        insights = state.get("insights", [])
        analysis = state.get("analysis_results", {})
        confidence = state.get("confidence_score", 0.0)

        logger.info("Formatting final research report")

        # Generate report
        report = await self._generate_report(
            query, documents, insights, analysis, confidence
        )

        # Generate executive summary
        summary = await self._generate_summary(query, insights, confidence)

        # Format citations
        citations = self._format_citations(documents)

        logger.info("Report formatting completed")

        return {
            "report": report,
            "summary": summary,
            "citations": citations
        }

    async def _generate_report(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        insights: List[str],
        analysis: Dict[str, Any],
        confidence: float
    ) -> str:
        """Generate the full research report."""

        insights_text = "\n".join([f"- {insight}" for insight in insights])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Create a comprehensive research report for:

Research Query: {query}

Key Insights:
{insights_text}

Number of Sources: {len(documents)}
Confidence Score: {confidence:.2%}

Create a well-structured report with:
1. Introduction
2. Methodology (brief)
3. Key Findings
4. Analysis
5. Conclusions
6. Recommendations (if applicable)

Use professional, academic language. Make it comprehensive but concise.""")
        ]

        response = await self.llm.ainvoke(messages)

        # Add metadata
        report = f"""# Research Report: {query}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Confidence Score:** {confidence:.2%}
**Sources Analyzed:** {len(documents)}

---

{response.content}

---

**Note:** This report was generated using AI-powered multi-agent research assistant.
"""

        return report

    async def _generate_summary(
        self,
        query: str,
        insights: List[str],
        confidence: float
    ) -> str:
        """Generate executive summary."""

        insights_text = "\n".join([f"- {insight}" for insight in insights[:5]])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Create a concise executive summary (2-3 paragraphs) for:

Research Query: {query}

Top Insights:
{insights_text}

Confidence: {confidence:.2%}

Write a clear, concise executive summary that captures the key findings.""")
        ]

        response = await self.llm.ainvoke(messages)

        return response.content

    def _format_citations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format citations from documents."""

        citations = []

        for i, doc in enumerate(documents, 1):
            citation = {
                "id": i,
                "title": doc.get("title", "Untitled"),
                "source_url": doc.get("source_url", "N/A"),
                "accessed": datetime.now().strftime('%Y-%m-%d'),
                "relevance": doc.get("metadata", {}).get("relevance", "medium")
            }
            citations.append(citation)

        return citations
