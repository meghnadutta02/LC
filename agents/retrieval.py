from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState
from database.vector_store import VectorStore
from tools.retrieval_tools import web_search_tool
from utils.logger import setup_logger
# Adjust import to your broadcast util location
from main import broadcast_agent_update

logger = setup_logger("retrieval_agent")


class RetrievalAgent:
    """
    Retrieval Agent with database-first strategy.

    Flow:
    1. Semantic+keyword search in database
    2. If insufficient, perform web search
    3. Store all new documents with embeddings
    4. Track and log all sources
    5. Broadcast real-time updates
    """

    def __init__(self, llm: ChatOpenAI, workflow_id: str):
        self.llm = llm
        self.workflow_id = workflow_id
        self.vector_store = VectorStore()
        self.system_prompt = """You are a Research Retrieval AI Agent.

Your responsibilities:
1. Search for relevant information
2. Identify high-quality sources
3. Extract key information

Generate 3-5 diverse search queries covering different topics."""

        logger.info(
            f"[RETRIEVAL] Retrieval Agent initialized for workflow {workflow_id}")

    async def process(self, state: ResearchState) -> Dict[str, Any]:
        query = state["query"]
        context = state.get("context", "")

        logger.info("=" * 80)
        logger.info(f"[RETRIEVAL] Starting retrieval for query: {query}")
        await broadcast_agent_update({
            "phase": "retrieval", "status": "started", "query": query
        })

        # Step 1: Semantic search
        logger.info("[RETRIEVAL] STEP 1: Semantic search in database...")
        db_semantic_docs = await self._check_database(query)

        # Step 2: Keyword fallback search if needed
        db_keyword_docs = []
        if len(db_semantic_docs) < 3:
            logger.info(
                "[RETRIEVAL] STEP 2: Insufficient semantic results, performing keyword search...")
            db_keyword_docs = await self.vector_store.keyword_search(query, max_results=10)

        # Merge and deduplicate documents
        combined_docs_dict = {}
        for doc in db_semantic_docs + db_keyword_docs:
            key = doc.get("id") or doc.get("content")[:200]
            if key not in combined_docs_dict:
                combined_docs_dict[key] = doc
        combined_docs = list(combined_docs_dict.values())
        logger.info(f"[RETRIEVAL] Combined docs count: {len(combined_docs)}")

        await broadcast_agent_update({
            "phase": "retrieval", "status": "query_db_finished", "documents_found": len(combined_docs)
        })

        # Step 3: Web search if still insufficient
        web_documents = []
        if len(combined_docs) < 3:
            logger.info(
                "[RETRIEVAL] STEP 3: Performing web search due to insufficient docs")
            web_documents = await self._search_web(query, context)
            combined_docs.extend(web_documents)

        await broadcast_agent_update({
            "phase": "retrieval", "status": "web_search_finished", "web_documents_found": len(web_documents)
        })

        # Step 4: Store new web docs
        logger.info("[RETRIEVAL] STEP 4: Storing new documents to DB...")
        document_ids = await self._store_documents(web_documents)

        # Log all sources
        logger.info("=" * 80)
        logger.info("[RETRIEVAL] Sources summary:")
        for i, doc in enumerate(combined_docs, 1):
            logger.info(
                f"[SOURCE {i}] {doc.get('title', 'Untitled')[:60]} - {doc.get('source_url', 'N/A')}")
        logger.info("=" * 80)
        logger.info(
            f"[RETRIEVAL] Retrieval complete with {len(combined_docs)} documents")

        await broadcast_agent_update({
            "phase": "retrieval", "status": "finished", "total_documents": len(combined_docs)
        })

        return {
            "documents": combined_docs,
            "queries": [query],
            "document_ids": document_ids
        }

    async def _check_database(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"[RETRIEVAL] Semantic DB search for '{query}'")
        try:
            results = await self.vector_store.search(query=query, max_results=5, similarity_threshold=0.75)
            logger.info(f"[RETRIEVAL] Found {len(results)} semantic results")
            return results
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error during semantic search: {e}")
            return []

    async def _search_web(self, query: str, context: str) -> List[Dict[str, Any]]:
        logger.info(f"[RETRIEVAL] Web search started for '{query}'")
        search_queries = await self._generate_search_queries(query, context)
        logger.info(f"[RETRIEVAL] Generated {len(search_queries)} queries")

        all_docs = []
        all_sources = []

        for i, sq in enumerate(search_queries, 1):
            logger.info(f"[RETRIEVAL] Executing web search query {i}: {sq}")
            try:
                result = await web_search_tool.ainvoke({"query": sq, "max_results": 3})
                documents = result.get("results", [])
                sources = result.get("sources", [])
                all_sources.extend(sources)
                for doc in documents:
                    detailed_doc = {
                        "title": doc.get("title", "Untitled"),
                        "content": doc.get("snippet", ""),
                        "source_url": doc.get("url", "N/A"),
                        "search_query": sq,
                        "metadata": {
                            "relevance": "high",
                            "source": doc.get("source", "web"),
                            "search_index": i
                        }
                    }
                    all_docs.append(detailed_doc)
                    logger.info(
                        f"[RETRIEVAL] Found doc: {detailed_doc['title'][:60]}")
            except Exception as e:
                logger.error(f"[RETRIEVAL] Web search error: {e}")

        logger.info(
            f"[RETRIEVAL] Web search completed: {len(all_docs)} documents found")
        return all_docs

    async def _generate_search_queries(self, query: str, context: str) -> List[str]:
        logger.info("[RETRIEVAL] Generating search queries using LLM")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"Research Query: {query}\nContext: {context}\n\nGenerate 4 diverse search queries one per line.")
        ]
        response = await self.llm.ainvoke(messages)
        queries = [
            line.strip().lstrip("-* ").strip('"') for line in response.content.splitlines() if line.strip()
        ]
        if query not in queries:
            queries.insert(0, query)
        return queries[:4]

    async def _store_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        if not documents:
            logger.info("[RETRIEVAL] No documents to store")
            return []
        logger.info(
            f"[RETRIEVAL] Storing {len(documents)} documents with embeddings")
        ids = []
        for i, doc in enumerate(documents, 1):
            try:
                logger.info(
                    f"[RETRIEVAL] Storing doc [{i}/{len(documents)}]: {doc.get('title', 'Untitled')[:40]}...")
                doc_id = await self.vector_store.store_document(
                    workflow_id=self.workflow_id,
                    content=doc["content"],
                    title=doc.get("title"),
                    source_url=doc.get("source_url"),
                    metadata=doc.get("metadata", {}),
                    # Store original search query
                    query=doc.get("search_query", None)
                )
                ids.append(doc_id)
                logger.info(f"[RETRIEVAL] Stored doc id: {doc_id}")
            except Exception as e:
                logger.error(f"[RETRIEVAL] Error storing document: {e}")
        return ids
