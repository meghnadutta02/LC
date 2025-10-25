"""
Retrieval Agent - Agent 2

Responsible for:
- Web search and document retrieval
- Storing documents with embeddings in pgvector
- Semantic search
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import httpx
from bs4 import BeautifulSoup

from graph.state import ResearchState
from database.vector_store import VectorStore
from utils.logger import setup_logger

logger = setup_logger("retrieval_agent")


class RetrievalAgent:
    """
    Retrieval Agent handles document retrieval and storage.

    Searches the web, extracts content, and stores documents as embeddings.
    """

    def __init__(self, llm: ChatOpenAI, workflow_id: str):
        """Initialize retrieval agent."""
        self.llm = llm
        self.workflow_id = workflow_id
        self.vector_store = VectorStore()
        self.system_prompt = """You are a Research Retrieval AI Agent.

Your responsibilities:
1. Generate effective search queries based on the research question
2. Identify relevant sources and documents
3. Extract key information

Output Format:
Return a JSON list of search queries to execute.
Generate 3-5 diverse search queries that cover different aspects of the topic."""

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        Process retrieval: search, extract, and store documents.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with retrieved documents and their IDs
        """
        query = state["query"]
        context = state.get("context", "")

        logger.info(f"Retrieval agent processing: {query}")

        # Generate search queries
        search_queries = await self._generate_search_queries(query, context)

        # Retrieve documents
        documents = await self._retrieve_documents(search_queries, query)

        # Store documents with embeddings
        document_ids = await self._store_documents(documents)

        logger.info(f"Retrieved and stored {len(documents)} documents")

        return {
            "documents": documents,
            "queries": search_queries,
            "document_ids": document_ids
        }

    async def _generate_search_queries(self, query: str, context: str) -> List[str]:
        """Generate search queries using LLM."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Research Query: {query}
Context: {context}

Generate 3-5 diverse search queries to find comprehensive information about this topic.
Return only the search queries, one per line.""")
        ]

        response = await self.llm.ainvoke(messages)

        # Parse queries from response
        queries = [
            line.strip().lstrip("- ").lstrip("* ").strip('"')
            for line in response.content.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Ensure we have the original query
        if query not in queries:
            queries.insert(0, query)

        return queries[:5]  # Limit to 5 queries

    async def _retrieve_documents(
        self,
        queries: List[str],
        original_query: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from web search.

        For now, this creates mock documents. In production, you'd integrate
        with a real search API (Google Custom Search, Bing, Tavily, etc.)
        """
        documents = []

        for query in queries:
            # Mock document (replace with actual search API)
            doc = {
                "title": f"Research on {query}",
                "content": f"""This is a research document about {query}.

Key findings:
- The topic relates to {original_query}
- Multiple perspectives exist on this subject
- Recent developments have shown significant progress
- Further research is recommended

This document provides valuable insights into the research question.""",
                "source_url": f"https://example.com/research/{query.replace(' ', '-')}",
                "metadata": {
                    "search_query": query,
                    "relevance": "high"
                }
            }
            documents.append(doc)

        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    async def _store_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """
        Store documents with embeddings in pgvector.

        Args:
            documents: List of document dictionaries

        Returns:
            List of document IDs from database
        """
        document_ids = []

        for doc in documents:
            try:
                doc_id = await self.vector_store.store_document(
                    workflow_id=self.workflow_id,
                    content=doc["content"],
                    title=doc.get("title"),
                    source_url=doc.get("source_url"),
                    metadata=doc.get("metadata", {})
                )
                document_ids.append(doc_id)
                logger.debug(f"Stored document {doc_id}")

            except Exception as e:
                logger.error(f"Failed to store document: {str(e)}")

        return document_ids
