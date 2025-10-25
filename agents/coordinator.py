"""
Coordinator Agent - Agent 1

Responsible for:
- Query decomposition
- Task prioritization
- Result synthesis
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState
from utils.logger import setup_logger

logger = setup_logger("coordinator_agent")


class CoordinatorAgent:
    """
    Coordinator Agent orchestrates the research workflow.

    Decomposes complex queries into sub-tasks and plans execution.
    """

    def __init__(self, llm: ChatOpenAI):
        """Initialize coordinator agent with LLM."""
        self.llm = llm
        self.system_prompt = """You are a Research Coordinator AI Agent.

Your responsibilities:
1. Analyze research queries and break them into manageable sub-tasks
2. Determine which specialized agents are needed (Retrieval, Analysis, Validation, Formatting)
3. Plan the execution order

Output Format:
Return a JSON object with:
- "needs_retrieval": boolean
- "needs_analysis": boolean  
- "needs_validation": boolean
- "tasks": list of task descriptions
- "focus_areas": list of specific areas to investigate

Be concise and specific. Focus on what information is needed."""

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        Process the query and decompose it into tasks.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with decomposed tasks and execution plan
        """
        query = state["query"]
        context = state.get("context", "")

        logger.info(f"Coordinator processing query: {query}")

        # Create messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Research Query: {query}

Context: {context}

Analyze this query and determine:
1. What information needs to be retrieved?
2. What analysis needs to be performed?
3. What validation is required?

Provide a structured plan.""")
        ]

        # Get LLM response
        response = await self.llm.ainvoke(messages)
        content = response.content

        # Parse response and create tasks
        tasks = self._parse_tasks(content, query)
        execution_plan = self._create_execution_plan(tasks)

        logger.info(f"Coordinator created {len(tasks)} tasks")

        return {
            "decomposed_tasks": tasks,
            "execution_plan": execution_plan
        }

    def _parse_tasks(self, llm_response: str, query: str) -> List[Dict[str, Any]]:
        """Parse LLM response and create structured tasks."""

        # Default tasks based on query analysis
        tasks = []

        # Analyze query keywords
        query_lower = query.lower()

        # Always include retrieval
        tasks.append({
            "id": "retrieval_task",
            "agent": "retrieval",
            "description": f"Gather relevant information for: {query}",
            "priority": 1
        })

        # Check if analysis is needed
        analysis_keywords = ["analyze", "compare",
                             "trend", "pattern", "relationship", "impact"]
        if any(kw in query_lower for kw in analysis_keywords):
            tasks.append({
                "id": "analysis_task",
                "agent": "analysis",
                "description": f"Perform deep analysis on: {query}",
                "priority": 2
            })

        # Always include validation
        tasks.append({
            "id": "validation_task",
            "agent": "validation",
            "description": f"Validate findings for: {query}",
            "priority": 3
        })

        # Always include formatting
        tasks.append({
            "id": "formatting_task",
            "agent": "formatting",
            "description": f"Format final report for: {query}",
            "priority": 4
        })

        return tasks

    def _create_execution_plan(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create execution plan from tasks."""

        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda x: x["priority"])

        return {
            "total_tasks": len(tasks),
            "execution_order": [t["id"] for t in sorted_tasks],
            "sequential": True  # Execute sequentially for now
        }
