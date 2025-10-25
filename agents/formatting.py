"""
Formatting Agent - Agent 5

Enhanced with:
- Longer, more detailed reports
- Complete citation lists with URLs
- Structured sections
- Comprehensive logging
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
    Formatting Agent creates comprehensive research reports.
    """

    def __init__(self, llm: ChatOpenAI):
        """Initialize formatting agent."""
        self.llm = llm
        self.system_prompt = """You are a Research Report Writing AI Agent.

Your responsibilities:
1. Create comprehensive, detailed research reports (1500-2000 words)
2. Write professional executive summaries
3. Include proper citations with URLs
4. Ensure academic tone and structure

The report should be:
- Thorough and detailed
- Well-organized with clear sections
- Evidence-based with citations
- Professional and academic
- At least 1500 words"""

        logger.info(" Formatting Agent initialized")

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        Format the final comprehensive research report.

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

        logger.info("=" * 80)
        logger.info(" FORMATTING AGENT PROCESSING")
        logger.info(f" Query: {query}")
        logger.info(f" Documents: {len(documents)}")
        logger.info(f" Insights: {len(insights)}")
        logger.info(f" Confidence: {confidence:.2%}")
        logger.info("=" * 80)

        # Generate comprehensive report
        logger.info(" STEP 1: Generating comprehensive report...")
        report = await self._generate_detailed_report(
            query, documents, insights, analysis, confidence
        )
        logger.info(f"    Report generated ({len(report.split())} words)")

        # Generate executive summary
        logger.info(" STEP 2: Generating executive summary...")
        summary = await self._generate_summary(query, insights, confidence)
        logger.info(f"    Summary generated ({len(summary.split())} words)")

        # Format complete citations
        logger.info(" STEP 3: Formatting citations...")
        citations = self._format_detailed_citations(
            documents, query)  # Pass query here
        logger.info(f"    {len(citations)} citations formatted")

        logger.info("=" * 80)
        logger.info(" FORMATTING COMPLETED")
        logger.info("=" * 80)

        return {
            "report": report,
            "summary": summary,
            "citations": citations
        }

    async def _generate_detailed_report(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        insights: List[str],
        analysis: Dict[str, Any],
        confidence: float
    ) -> str:
        """Generate comprehensive detailed report."""

        # Prepare content
        insights_text = "\n".join([f"- {insight}" for insight in insights])

        doc_summaries = []
        for i, doc in enumerate(documents[:10], 1):
            doc_summaries.append(f"""
Document {i}: {doc.get('title', 'Untitled')}
Source: {doc.get('source_url', 'N/A')}
Content: {doc.get('content', '')[:300]}...
""")

        docs_text = "\n".join(doc_summaries)

        logger.info(
            "   🤖 Generating report with LLM (expecting 1500+ words)...")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Create a comprehensive research report (1500-2000 words) for:

RESEARCH QUERY: {query}

KEY INSIGHTS:
{insights_text}

SOURCE DOCUMENTS:
{docs_text}

ANALYSIS SUMMARY:
{analysis.get('summary', 'Analysis included in findings')}

Create a detailed report with these sections:

1. INTRODUCTION (200 words)
   - Background and context
   - Research objectives
   - Scope

2. METHODOLOGY (150 words)
   - Research approach
   - Sources analyzed
   - Data collection methods

3. KEY FINDINGS (600 words)
   - Detailed findings with evidence
   - Reference specific sources
   - Multiple perspectives

4. DETAILED ANALYSIS (400 words)
   - In-depth analysis
   - Connections and patterns
   - Implications

5. CONCLUSIONS (200 words)
   - Summary of findings
   - Main takeaways
   - Significance

6. RECOMMENDATIONS (150 words)
   - Future research directions
   - Practical applications

Make it comprehensive, detailed, and professional. Use evidence from the sources.""")
        ]

        response = await self.llm.ainvoke(messages)

        # Build final report
        report = f"""# Comprehensive Research Report: {query}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Confidence Score:** {confidence:.1%}  
**Sources Analyzed:** {len(documents)}  
**Research Method:** Multi-Agent AI Analysis

---

{response.content}

---

## Citations

This report references {len(documents)} sources. See full citation list below.

---

**Methodology Note:** This research was conducted using a multi-agent AI system that:
1. Retrieved relevant documents from database and web sources
2. Performed semantic analysis and fact-checking
3. Validated findings across multiple sources
4. Synthesized information into this comprehensive report

**Confidence Assessment:** {confidence:.1%}
- Based on {len(documents)} independent sources
- Cross-referenced and validated
- Analysis performed by specialized AI agents

---

*Generated by Multi-Agent Research Assistant - Powered by LangGraph 1.0 & OpenAI*
"""

        logger.info(f"    Final report: {len(report.split())} words")
        return report

    async def _generate_summary(
        self,
        query: str,
        insights: List[str],
        confidence: float
    ) -> str:
        """Generate executive summary."""

        insights_text = "\n".join([f"- {insight}" for insight in insights[:7]])

        logger.info("   🤖 Generating executive summary...")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Create a detailed executive summary (300-400 words) for:

RESEARCH QUERY: {query}

KEY INSIGHTS:
{insights_text}

CONFIDENCE: {confidence:.1%}

Write a comprehensive 3-4 paragraph executive summary that captures:
- Main findings
- Key takeaways
- Significance
- Implications

Be thorough and professional.""")
        ]

        response = await self.llm.ainvoke(messages)
        logger.info(f"    Summary: {len(response.content.split())} words")

        return response.content

    def _format_detailed_citations(
        self,
        documents: List[Dict[str, Any]],
        query: str  # Added query parameter
    ) -> List[Dict[str, Any]]:
        """Format complete citations with all metadata."""

        citations = []

        for i, doc in enumerate(documents, 1):
            citation = {
                "id": i,
                "title": doc.get("title", "Untitled Document"),
                "url": doc.get("source_url", "N/A"),
                "accessed": datetime.now().strftime('%Y-%m-%d'),
                #  Fixed - now query is defined
                "search_query": doc.get("search_query", query),
                "relevance": doc.get("metadata", {}).get("relevance", "medium"),
                "source_type": doc.get("metadata", {}).get("source", "web"),
                "formatted_citation": f"[{i}] {doc.get('title', 'Untitled')}. Retrieved from {doc.get('source_url', 'N/A')}. Accessed {datetime.now().strftime('%Y-%m-%d')}.",
                "preview": doc.get("content", "")[:150] + "..."
            }
            citations.append(citation)

            logger.info(f"      [{i}] {citation['title'][:60]}...")
            logger.info(f"          URL: {citation['url']}")

        return citations
