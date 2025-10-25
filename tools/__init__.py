"""
Tools package for the research assistant.

All tools are MCP-compatible and use LangChain 1.0 syntax.
"""

from .coordinator_tools import (
    decompose_query_tool,
    synthesize_results_tool
)
from .retrieval_tools import (
    web_search_tool,
    semantic_search_tool,
    extract_text_tool
)
from .analysis_tools import (
    analyze_documents_tool,
    identify_trends_tool,
    comparative_analysis_tool
)
from .validation_tools import (
    validate_sources_tool,
    check_contradictions_tool,
    calculate_confidence_tool
)
from .formatting_tools import (
    format_report_tool,
    generate_citations_tool,
    create_summary_tool
)

__all__ = [
    # Coordinator
    "decompose_query_tool",
    "synthesize_results_tool",
    # Retrieval
    "web_search_tool",
    "semantic_search_tool",
    "extract_text_tool",
    # Analysis
    "analyze_documents_tool",
    "identify_trends_tool",
    "comparative_analysis_tool",
    # Validation
    "validate_sources_tool",
    "check_contradictions_tool",
    "calculate_confidence_tool",
    # Formatting
    "format_report_tool",
    "generate_citations_tool",
    "create_summary_tool",
]
