"""
Analysis Agent - Agent 3

 detailed logging of analysis process.
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState
from utils.logger import setup_logger

logger = setup_logger("analysis_agent")


class AnalysisAgent:
    """Analysis Agent performs deep analysis on retrieved documents."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize analysis agent."""
        self.llm = llm
        self.system_prompt = """You are a Research Analysis AI Agent.

Your responsibilities:
1. Perform comparative analysis across documents
2. Identify trends and patterns
3. Detect causal relationships
4. Generate detailed insights (aim for 8-10 insights)

Be analytical, objective, and evidence-based. Provide thorough analysis."""

        logger.info("🔬 Analysis Agent initialized")

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """Analyze retrieved documents."""

        query = state["query"]
        documents = state.get("retrieved_documents", [])

        logger.info("=" * 80)
        logger.info("🔬 ANALYSIS AGENT PROCESSING")
        logger.info(f"📋 Query: {query}")
        logger.info(f"📚 Documents to analyze: {len(documents)}")
        logger.info("=" * 80)

        if not documents:
            logger.warning("⚠️  No documents to analyze")
            return self._empty_analysis()

        # Log document summary
        logger.info("📄 Document Summary:")
        for i, doc in enumerate(documents[:5], 1):
            logger.info(f"   {i}. {doc.get('title', 'Untitled')[:60]}...")
            logger.info(f"      Source: {doc.get('source_url', 'N/A')}")

        if len(documents) > 5:
            logger.info(f"   ... and {len(documents) - 5} more documents")

        # Perform analysis
        logger.info("🤖 STEP 1: Performing deep analysis with LLM...")
        analysis_results = await self._analyze_documents(query, documents)

        logger.info(f"   ✅ Analysis complete")
        logger.info(
            f"   💡 Generated {len(analysis_results['insights'])} insights")

        # Log insights
        logger.info("💡 Key Insights:")
        for i, insight in enumerate(analysis_results['insights'], 1):
            logger.info(f"   {i}. {insight[:100]}...")

        logger.info("=" * 80)
        logger.info(
            f"✅ ANALYSIS COMPLETED: {len(analysis_results['insights'])} insights generated")
        logger.info("=" * 80)

        return analysis_results

    async def _analyze_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform detailed analysis on documents."""

        # Prepare document summaries (use more documents for analysis)
        doc_summaries = "\n\n".join([
            f"Document {i+1}: {doc.get('title', 'Untitled')}\nSource: {doc.get('source_url', 'N/A')}\nContent: {doc['content'][:600]}..."
            for i, doc in enumerate(documents[:8])  # Increased from 5 to 8
        ])

        logger.info(f"   📊 Analyzing {min(len(documents), 8)} documents...")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Research Query: {query}

Documents:
{doc_summaries}

Perform a comprehensive analysis:
1. Compare and contrast information across documents
2. Identify key trends, patterns, or themes
3. Note any causal relationships or correlations
4. Generate 8-10 detailed, specific insights

Provide thorough, evidence-based analysis with specific references to the documents.""")
        ]

        response = await self.llm.ainvoke(messages)

        logger.info(
            f"   📝 LLM response length: {len(response.content)} characters")

        # Parse insights
        insights = self._extract_insights(response.content)

        return {
            "analysis": {"summary": response.content},
            "comparisons": self._extract_comparisons(response.content),
            "trends": self._extract_trends(response.content),
            "insights": insights
        }

    def _extract_insights(self, content: str) -> List[str]:
        """Extract key insights from LLM response."""
        insights = []

        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered lists
            if line.startswith(("- ", "* ", "• ")) or (line and line[0].isdigit() and ". " in line):
                insight = line.lstrip("-*•0123456789. ").strip()
                if len(insight) > 30:  # More substantial insights
                    insights.append(insight)
                    logger.debug(
                        f"      → Extracted insight: {insight[:80]}...")

        logger.info(f"   ✓ Extracted {len(insights)} insights from analysis")
        return insights[:10]  # Limit to 10 best insights

    def _extract_comparisons(self, content: str) -> List[Dict[str, Any]]:
        """Extract comparative findings."""
        return [{"comparison": "Comparative analysis included in main analysis"}]

    def _extract_trends(self, content: str) -> List[Dict[str, Any]]:
        """Extract identified trends."""
        return [{"trend": "Trends identified in main analysis"}]

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        logger.warning("⚠️  Returning empty analysis")
        return {
            "analysis": {},
            "comparisons": [],
            "trends": [],
            "insights": []
        }
