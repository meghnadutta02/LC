"""
Retrieval Agent - Agent 2

Responsible for:
- Check database first before web search
- Comprehensive logging
- Store all retrieved documents
- Semantic search
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState
from database.vector_store import VectorStore
from tools.retrieval_tools import web_search_tool, extract_text_tool
from utils.logger import setup_logger

logger = setup_logger("retrieval_agent")


class RetrievalAgent:
    """
    Retrieval Agent with database-first strategy.

    Flow:
    1. Check existing documents in database
    2. If insufficient, perform web search
    3. Store all new documents with embeddings
    """

    def __init__(self, llm: ChatOpenAI, workflow_id: str):
        """Initialize retrieval agent."""
        self.llm = llm
        self.workflow_id = workflow_id
        self.vector_store = VectorStore()
        self.system_prompt = """You are a Research Retrieval AI Agent.

Your responsibilities:
1. Search for relevant information
2. Identify high-quality sources
3. Extract key information

Generate 3-5 diverse search queries that cover different aspects of the topic."""

        logger.info(
            f" Retrieval Agent initialized for workflow {workflow_id}")

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        """
        Process retrieval with database-first approach.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with retrieved documents and their IDs
        """
        query = state["query"]
        context = state.get("context", "")

        logger.info("=" * 80)
        logger.info(f" RETRIEVAL AGENT PROCESSING")
        logger.info(f" Query: {query}")
        logger.info(f" Context: {context}")
        logger.info("=" * 80)

        # Step 1: Check database first
        logger.info("️  STEP 1: Checking database for existing documents...")
        db_documents = await self._check_database(query)

        # Step 2: If insufficient, search web
        web_documents = []
        if len(db_documents) < 3:
            logger.info(
                f" STEP 2: Insufficient documents in DB ({len(db_documents)}), performing web search...")
            web_documents = await self._search_web(query, context)
        else:
            logger.info(
                f" STEP 2: Found {len(db_documents)} documents in database, skipping web search")

        # Combine documents
        all_documents = db_documents + web_documents
        logger.info(
            f" Total documents: {len(all_documents)} ({len(db_documents)} from DB + {len(web_documents)} from web)")

        # Step 3: Store new documents
        logger.info(" STEP 3: Storing new documents in database...")
        document_ids = await self._store_documents(web_documents)

        logger.info("=" * 80)
        logger.info(
            f" RETRIEVAL COMPLETED: {len(all_documents)} documents retrieved")
        logger.info("=" * 80)

        return {
            "documents": all_documents,
            "queries": [query],
            "document_ids": document_ids
        }

    async def _check_database(self, query: str) -> List[Dict[str, Any]]:
        """Check database for existing relevant documents."""
        logger.info(f"    Searching database for: '{query}'")

        try:
            results = await self.vector_store.search(
                query=query,
                max_results=5,
                similarity_threshold=0.75
            )

            logger.info(
                f"    Found {len(results)} existing documents in database")

            for i, doc in enumerate(results, 1):
                logger.info(
                    f"      #{i}: {doc.get('title', 'Untitled')[:50]}... (similarity: {doc.get('similarity', 0):.2%})")

            return results

        except Exception as e:
            logger.error(f"    Database search error: {str(e)}")
            return []

    async def _search_web(self, query: str, context: str) -> List[Dict[str, Any]]:
        """Search web for new documents."""
        logger.info(f"    Web search initiated for: '{query}'")

        # Generate search queries
        search_queries = await self._generate_search_queries(query, context)
        logger.info(f"    Generated {len(search_queries)} search queries:")
        for i, sq in enumerate(search_queries, 1):
            logger.info(f"      {i}. {sq}")

        # Search web
        all_documents = []
        for i, search_query in enumerate(search_queries, 1):
            logger.info(
                f"    Executing search {i}/{len(search_queries)}: '{search_query}'")

            try:
                result = await web_search_tool.ainvoke({"query": search_query, "max_results": 3})
                documents = result.get("results", [])

                logger.info(f"       Found {len(documents)} results")

                for doc in documents:
                    # Create detailed document
                    detailed_doc = {
                        "title": doc.get("title", "Untitled"),
                        "content": doc.get("snippet", ""),
                        "source_url": doc.get("url", "N/A"),
                        "search_query": search_query,
                        "metadata": {
                            "relevance": "high",
                            "source": doc.get("source", "web"),
                            "search_index": i
                        }
                    }
                    all_documents.append(detailed_doc)
                    logger.info(f"         • {detailed_doc['title'][:60]}...")

            except Exception as e:
                logger.error(f"       Search error: {str(e)}")

        logger.info(
            f"    Web search complete: {len(all_documents)} documents retrieved")
        return all_documents

    async def _generate_search_queries(self, query: str, context: str) -> List[str]:
        """Generate diverse search queries."""
        logger.info(f"   🤖 Generating search queries using LLM...")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Research Query: {query}
Context: {context}

Generate 4 diverse search queries to find comprehensive information.
Return only the queries, one per line.""")
        ]

        response = await self.llm.ainvoke(messages)

        queries = [
            line.strip().lstrip("- ").lstrip("* ").strip('"')
            for line in response.content.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        if query not in queries:
            queries.insert(0, query)

        return queries[:4]

    async def _store_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """Store documents with embeddings."""
        if not documents:
            logger.info("   ℹ️  No new documents to store")
            return []

        logger.info(
            f"    Storing {len(documents)} documents with embeddings...")
        document_ids = []

        for i, doc in enumerate(documents, 1):
            try:
                logger.info(
                    f"      {i}/{len(documents)}: Storing '{doc.get('title', 'Untitled')[:40]}...'")

                doc_id = await self.vector_store.store_document(
                    workflow_id=self.workflow_id,
                    content=doc["content"],
                    title=doc.get("title"),
                    source_url=doc.get("source_url"),
                    metadata=doc.get("metadata", {})
                )
                document_ids.append(doc_id)
                logger.info(f"          Stored with ID: {doc_id}")

            except Exception as e:
                logger.error(f"          Storage error: {str(e)}")

        logger.info(f"    Stored {len(document_ids)} documents successfully")
        return document_ids
