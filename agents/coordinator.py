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
    """Coordinator Agent orchestrates the research workflow."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize coordinator agent with LLM."""
        self.llm = llm
        self.system_prompt = """You are a Research Coordinator AI Agent.

Your responsibilities:
1. Analyze research queries and break them into sub-tasks
2. Determine which agents are needed
3. Plan execution order

Be concise and specific. Focus on what information is needed."""

        logger.info(" Coordinator Agent initialized")

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """Process the query and decompose it into tasks."""

        query = state["query"]
        context = state.get("context", "")

        logger.info("=" * 80)
        logger.info(" COORDINATOR AGENT PROCESSING")
        logger.info(f" Query: {query}")
        logger.info(f" Context: {context}")
        logger.info("=" * 80)

        logger.info("🤖 STEP 1: Analyzing query with LLM...")

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
        logger.info(f"    LLM Analysis complete")
        logger.info(f"    Response preview: {response.content[:200]}...")

        # Parse response and create tasks
        logger.info(" STEP 2: Creating execution tasks...")
        tasks = self._parse_tasks(response.content, query)

        for i, task in enumerate(tasks, 1):
            logger.info(
                f"   Task {i}: [{task['agent']}] {task['description']}")

        logger.info("️  STEP 3: Creating execution plan...")
        execution_plan = self._create_execution_plan(tasks)
        logger.info(
            f"    Execution plan: {len(tasks)} tasks in {execution_plan.get('total_tasks', 0)} steps")

        logger.info("=" * 80)
        logger.info(f" COORDINATOR COMPLETED: {len(tasks)} tasks planned")
        logger.info("=" * 80)

        return {
            "decomposed_tasks": tasks,
            "execution_plan": execution_plan
        }

    def _parse_tasks(self, llm_response: str, query: str) -> List[Dict[str, Any]]:
        """Parse LLM response and create structured tasks."""

        tasks = []
        query_lower = query.lower()

        # Always include retrieval
        tasks.append({
            "id": "retrieval_task",
            "agent": "retrieval",
            "description": f"Retrieve relevant information for: {query}",
            "priority": 1
        })
        logger.info(f"       Added retrieval task")

        # Check if analysis is needed
        analysis_keywords = ["analyze", "compare",
                             "trend", "pattern", "relationship", "impact"]
        if any(kw in query_lower for kw in analysis_keywords):
            tasks.append({
                "id": "analysis_task",
                "agent": "analysis",
                "description": f"Perform analysis on: {query}",
                "priority": 2
            })
            logger.info(f"       Added analysis task (keywords detected)")

        # Always validate
        tasks.append({
            "id": "validation_task",
            "agent": "validation",
            "description": f"Validate findings for: {query}",
            "priority": 3
        })
        logger.info(f"       Added validation task")

        # Always format
        tasks.append({
            "id": "formatting_task",
            "agent": "formatting",
            "description": f"Format comprehensive report for: {query}",
            "priority": 4
        })
        logger.info(f"       Added formatting task")

        return tasks

    def _create_execution_plan(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create execution plan from tasks."""

        sorted_tasks = sorted(tasks, key=lambda x: x["priority"])

        plan = {
            "total_tasks": len(tasks),
            "execution_order": [t["id"] for t in sorted_tasks],
            "sequential": True
        }

        logger.info(
            f"      → Execution order: {' → '.join([t['agent'] for t in sorted_tasks])}")

        return plan
