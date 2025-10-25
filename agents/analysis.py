"""
Analysis Agent - Agent 3

Responsible for:
- Comparative analysis
- Trend detection
- Causal reasoning
- Statistical analysis
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState
from utils.logger import setup_logger

logger = setup_logger("analysis_agent")


class AnalysisAgent:
    """
    Analysis Agent performs deep analysis on retrieved documents.
    """

    def __init__(self, llm: ChatOpenAI):
        """Initialize analysis agent."""
        self.llm = llm
        self.system_prompt = """You are a Research Analysis AI Agent.

Your responsibilities:
1. Perform comparative analysis across multiple documents
2. Identify trends and patterns
3. Detect causal relationships
4. Generate key insights

Be analytical, objective, and evidence-based. Support all claims with references to the documents."""

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        Analyze retrieved documents.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with analysis results
        """
        query = state["query"]
        documents = state.get("retrieved_documents", [])

        logger.info(f"Analysis agent analyzing {len(documents)} documents")

        if not documents:
            logger.warning("No documents to analyze")
            return self._empty_analysis()

        # Perform analysis
        analysis_results = await self._analyze_documents(query, documents)

        logger.info("Analysis completed")

        return analysis_results

    async def _analyze_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform analysis on documents."""

        # Prepare document summaries
        doc_summaries = "\n\n".join([
            f"Document {i+1} ({doc.get('title', 'Untitled')}):\n{doc['content'][:500]}..."
            for i, doc in enumerate(documents[:5])  # Limit to 5 docs
        ])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Research Query: {query}

Documents:
{doc_summaries}

Perform a comprehensive analysis:
1. Compare and contrast the information across documents
2. Identify key trends or patterns
3. Note any causal relationships
4. Generate 5-7 key insights

Provide structured, evidence-based analysis.""")
        ]

        response = await self.llm.ainvoke(messages)

        # Parse insights from response
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
                if len(insight) > 20:  # Meaningful insights
                    insights.append(insight)

        return insights[:7]  # Limit to 7 insights

    def _extract_comparisons(self, content: str) -> List[Dict[str, Any]]:
        """Extract comparative findings."""
        return [{"comparison": "Comparative analysis included in main analysis"}]

    def _extract_trends(self, content: str) -> List[Dict[str, Any]]:
        """Extract identified trends."""
        return [{"trend": "Trends identified in main analysis"}]

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "analysis": {},
            "comparisons": [],
            "trends": [],
            "insights": []
        }
