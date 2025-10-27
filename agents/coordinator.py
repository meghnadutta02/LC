from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState
from utils.logger import setup_logger
from main import broadcast_agent_update  # Your websocket broadcast utility

import asyncio

logger = setup_logger("coordinator_agent")


class CoordinatorAgent:
    """Coordinator Agent orchestrates the research workflow with HITL support."""

    def __init__(self, llm: ChatOpenAI):
        """Initialize coordinator agent with LLM and async feedback control."""
        self.llm = llm
        self.system_prompt = """You are a Research Coordinator AI Agent.

Your responsibilities:
1. Analyze research queries and break them into sub-tasks
2. Determine which agents are needed
3. Plan execution order

Be concise and specific. Focus on what information is needed."""

        # Async event and data for human-in-the-loop feedback
        self.human_feedback_event = asyncio.Event()
        self.human_feedback_data = None

        logger.info("Coordinator Agent initialized")

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """Process the query and decompose it into tasks, with HITL checkpoint."""

        query = state["query"]
        context = state.get("context", "")

        logger.info("=" * 80)
        logger.info("COORDINATOR AGENT PROCESSING")
        logger.info(f"Query: {query}")
        logger.info(f"Context: {context}")
        logger.info("=" * 80)

        # Broadcast start of coordinator phase
        await broadcast_agent_update({
            "phase": "coordinator",
            "status": "started",
            "query": query,
            "context": context
        })

        logger.info("🤖 STEP 1: Analyzing query with LLM...")

        # Create messages to ask LLM to analyze query decomposition
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
        logger.info(f"   LLM Analysis complete")
        logger.info(f"   Response preview: {response.content[:200]}...")

        logger.info("STEP 2: Creating execution tasks...")
        tasks = self._parse_tasks(response.content, query)

        for i, task in enumerate(tasks, 1):
            logger.info(
                f"   Task {i}: [{task['agent']}] {task['description']}")

        logger.info("STEP 3: Creating execution plan...")
        execution_plan = self._create_execution_plan(tasks)
        logger.info(
            f"    Execution plan: {len(tasks)} tasks in {execution_plan.get('total_tasks', 0)} steps")

        # Broadcast HITL checkpoint - awaiting human approval/modification
        await broadcast_agent_update({
            "phase": "coordinator",
            "status": "awaiting_human_approval",
            "decomposed_tasks": tasks,
            "execution_plan": execution_plan
        })

        # Wait asynchronously for human feedback, with timeout (e.g. 5 minutes)
        try:
            await asyncio.wait_for(self.human_feedback_event.wait(), timeout=300)
        except asyncio.TimeoutError:
            logger.warning(
                "Human approval timeout reached, proceeding automatically")

        # Process human feedback if any
        if self.human_feedback_data:
            if not self.human_feedback_data.get("approved", False):
                logger.warning("Human approval denied, aborting workflow")
                return {"error": "Human approval denied"}

            if "modified_tasks" in self.human_feedback_data:
                tasks = self.human_feedback_data["modified_tasks"]
                execution_plan = self._create_execution_plan(tasks)

        # Reset event and feedback data for next run
        self.human_feedback_event.clear()
        self.human_feedback_data = None

        logger.info("=" * 80)
        logger.info(f"COORDINATOR COMPLETED: {len(tasks)} tasks planned")
        logger.info("=" * 80)

        # Broadcast finish
        await broadcast_agent_update({
            "phase": "coordinator",
            "status": "finished",
            "decomposed_tasks": tasks,
            "execution_plan": execution_plan
        })

        return {
            "decomposed_tasks": tasks,
            "execution_plan": execution_plan
        }

    def receive_human_feedback(self, feedback: dict):
        """Receive human feedback asynchronously from WebSocket handler."""
        logger.info(f"Received human feedback: {feedback}")
        self.human_feedback_data = feedback
        self.human_feedback_event.set()

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
        logger.info(f"      Added retrieval task")

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
            logger.info(f"      Added analysis task (keywords detected)")

        # Always validate
        tasks.append({
            "id": "validation_task",
            "agent": "validation",
            "description": f"Validate findings for: {query}",
            "priority": 3
        })
        logger.info(f"      Added validation task")

        # Always format
        tasks.append({
            "id": "formatting_task",
            "agent": "formatting",
            "description": f"Format comprehensive report for: {query}",
            "priority": 4
        })
        logger.info(f"      Added formatting task")

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
            f"       → Execution order: {' → '.join([t['agent'] for t in sorted_tasks])}")

        return plan
