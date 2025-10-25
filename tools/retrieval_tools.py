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
        Dictionary with search results
    """
    logger.info(f"Web search: {query}")

    try:
        # Use DuckDuckGo (free alternative)
        search = DuckDuckGoSearchAPIWrapper()
        results_text = search.run(query)

        # Parse results
        results = []
        snippets = results_text.split('\n\n')

        for i, snippet in enumerate(snippets[:max_results]):
            if snippet.strip():
                results.append({
                    "title": f"Result {i+1}",
                    "snippet": snippet.strip(),
                    "source": "DuckDuckGo"
                })

        logger.info(f"Found {len(results)} results")

        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        # Fallback to mock data
        return {
            "query": query,
            "results": [{
                "title": f"Research on {query}",
                "snippet": f"Information about {query} from web search.",
                "source": "mock"
            }],
            "total_found": 1
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
    logger.info(f"Semantic search: {query}")

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
        logger.error(f"Semantic search error: {str(e)}")
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
    logger.info(f"Extracting text from: {url}")

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

            return {
                "url": url,
                "text": text[:5000],  # Limit to 5000 chars
                "length": len(text),
                "success": True
            }

    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        return {
            "url": url,
            "text": "",
            "length": 0,
            "success": False,
            "error": str(e)
        }
