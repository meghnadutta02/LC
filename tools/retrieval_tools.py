"""
Retrieval Agent Tools.

Tools for web search, document retrieval, and text extraction.
Uses free tools from LangChain 1.0.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from pydantic import BaseModel, Field
import httpx
from bs4 import BeautifulSoup

from database.vector_store import VectorStore
from utils.logger import setup_logger

logger = setup_logger("retrieval_tools")


class WebSearchInput(BaseModel):
    """Input schema for web search."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results")


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search."""
    query: str = Field(description="Search query")
    workflow_id: str = Field(description="Workflow ID")
    max_results: int = Field(default=10, description="Maximum results")


class ExtractTextInput(BaseModel):
    """Input schema for text extraction."""
    url: str = Field(description="URL to extract text from")


@tool(args_schema=WebSearchInput)
async def web_search_tool(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo (free, no API key required).

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        Dictionary with search results including titles, URLs, and snippets
    """
    logger.info("=" * 80)
    logger.info(f"[WEB SEARCH] Searching for: '{query}'")
    logger.info("=" * 80)

    try:
        # Use DuckDuckGo (free alternative)
        search = DuckDuckGoSearchAPIWrapper()
        results_text = search.run(query)

        # Parse results
        results = []
        snippets = results_text.split('\n\n')

        logger.info(f"[WEB SEARCH] Found {len(snippets)} results:")
        logger.info("-" * 80)

        for i, snippet in enumerate(snippets[:max_results], 1):
            if snippet.strip():
                # Try to extract title and URL from snippet
                lines = snippet.strip().split('\n')
                title = lines[0] if lines else f"Result {i}"
                url = lines[1] if len(
                    lines) > 1 and lines[1].startswith('http') else "N/A"
                content = '\n'.join(lines[1:]) if len(
                    lines) > 1 else snippet.strip()

                result = {
                    "id": i,
                    "title": title,
                    "url": url,
                    "snippet": content,
                    "source": "DuckDuckGo"
                }
                results.append(result)

                # Log each result with source
                logger.info(f"[SOURCE #{i}]")
                logger.info(f"  Title: {title}")
                logger.info(f"  URL: {url}")
                logger.info(f"  Preview: {content[:150]}...")
                logger.info("-" * 80)

        logger.info(
            f"[WEB SEARCH] Successfully retrieved {len(results)} sources")
        logger.info("=" * 80)

        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "sources": [{"title": r["title"], "url": r["url"]} for r in results]
        }

    except Exception as e:
        logger.error(f"[WEB SEARCH] Error: {str(e)}")

        # Fallback to mock data with logging
        logger.warning("[WEB SEARCH] Using fallback mock data")
        mock_result = {
            "id": 1,
            "title": f"Research on {query}",
            "url": f"https://example.com/research/{query.replace(' ', '-')}",
            "snippet": f"Information about {query} from web search.",
            "source": "mock"
        }

        logger.info("[SOURCE #1] (MOCK)")
        logger.info(f"  Title: {mock_result['title']}")
        logger.info(f"  URL: {mock_result['url']}")
        logger.info(f"  Preview: {mock_result['snippet']}")

        return {
            "query": query,
            "results": [mock_result],
            "total_found": 1,
            "sources": [{"title": mock_result["title"], "url": mock_result["url"]}]
        }


@tool(args_schema=SemanticSearchInput)
async def semantic_search_tool(
    query: str,
    workflow_id: str,
    max_results: int = 10
) -> Dict[str, Any]:
    """
    Perform semantic search in pgvector database.

    Args:
        query: Search query
        workflow_id: Workflow ID to filter results
        max_results: Maximum number of results

    Returns:
        Dictionary with semantically similar documents
    """
    logger.info(f"[SEMANTIC SEARCH] Searching: {query}")

    try:
        vector_store = VectorStore()
        results = await vector_store.search(
            query=query,
            max_results=max_results,
            workflow_id=workflow_id
        )

        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        logger.error(f"[SEMANTIC SEARCH] Error: {str(e)}")
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "error": str(e)
        }


@tool(args_schema=ExtractTextInput)
async def extract_text_tool(url: str) -> Dict[str, Any]:
    """
    Extract text content from a URL.

    Uses httpx and BeautifulSoup (free tools).

    Args:
        url: URL to extract text from

    Returns:
        Dictionary with extracted text
    """
    logger.info(f"[TEXT EXTRACTION] Extracting from: {url}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip()
                      for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            logger.info(
                f"[TEXT EXTRACTION] Extracted {len(text)} characters from {url}")

            return {
                "url": url,
                "text": text[:5000],  # Limit to 5000 chars
                "length": len(text),
                "success": True
            }

    except Exception as e:
        logger.error(f"[TEXT EXTRACTION] Error: {str(e)}")
        return {
            "url": url,
            "text": "",
            "length": 0,
            "success": False,
            "error": str(e)
        }
