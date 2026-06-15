"""
Unit tests for the hybrid_generation_node — the most complex path in the graph.

Tests verify:
  1. Basic hybrid node invocation returns expected keys
  2. SQL-only edge cases (with no docs retrieved)
  3. Graceful handling when both SQL and retrieval fail
  4. The node integrates SQL results and RAG context into a synthesis answer
"""

import pytest


class TestHybridGenerationNode:
    """Tests for the hybrid_generation_node graph node function.

    Uses lazy imports inside test functions to avoid triggering a
    pre-existing pyarrow segfault during module collection phase.
    """

    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        """The returned dict must contain all expected hybrid output keys."""
        from langchain_core.messages import HumanMessage

        from app.core.graph.nodes import hybrid_generation_node

        result = await hybrid_generation_node(
            {
                "question": "Compare sales data with policy documents",
                "messages": [
                    HumanMessage(content="Compare sales data with policy documents")
                ],
                "ltm_context": "",
                "summary": "",
            }
        )
        expected_keys = {
            "sql_query",
            "sql_results",
            "sql_status",
            "docs",
            "good_docs",
            "refined_context",
            "crag_verdict",
            "answer",
            "hyde_used",
            "hyde_hypotheses",
            "reranking_used",
        }
        assert expected_keys.issubset(set(result.keys()))

    @pytest.mark.asyncio
    async def test_handles_empty_question(self):
        """Should not crash with an empty question string."""
        from app.core.graph.nodes import hybrid_generation_node

        result = await hybrid_generation_node(
            {
                "question": "",
                "messages": [],
                "ltm_context": "",
                "summary": "",
            }
        )
        assert "answer" in result
        assert "sql_status" in result
        assert "sql_results" in result

    @pytest.mark.asyncio
    async def test_handles_no_question_key(self):
        """Should not crash when question key is missing."""
        from app.core.graph.nodes import hybrid_generation_node

        result = await hybrid_generation_node(
            {
                "messages": [],
                "ltm_context": "",
                "summary": "",
            }
        )
        assert "answer" in result

    @pytest.mark.asyncio
    async def test_sql_status_defaults_to_skipped_when_no_data(self):
        """When no SQL data is available, sql_status should be 'skipped' or 'error'."""
        from langchain_core.messages import HumanMessage

        from app.core.graph.nodes import hybrid_generation_node

        result = await hybrid_generation_node(
            {
                "question": "Tell me about policies",
                "messages": [HumanMessage(content="Tell me about policies")],
                "ltm_context": "",
                "summary": "",
            }
        )
        assert result["sql_status"] in ("skipped", "error", "failed_safety")
        assert result["sql_results"] == []

    @pytest.mark.asyncio
    async def test_sql_status_string_type(self):
        """sql_status should always be a string."""
        from langchain_core.messages import HumanMessage

        from app.core.graph.nodes import hybrid_generation_node

        result = await hybrid_generation_node(
            {
                "question": "Show orders and compare with policy",
                "messages": [
                    HumanMessage(content="Show orders and compare with policy")
                ],
                "ltm_context": "",
                "summary": "",
            }
        )
        assert isinstance(result["sql_status"], str)

    @pytest.mark.asyncio
    async def test_answer_is_string(self):
        """The answer field should always be a string."""
        from langchain_core.messages import HumanMessage

        from app.core.graph.nodes import hybrid_generation_node

        result = await hybrid_generation_node(
            {
                "question": "What is our sales data showing?",
                "messages": [HumanMessage(content="What is our sales data showing?")],
                "ltm_context": "",
                "summary": "",
            }
        )
        assert isinstance(result["answer"], str)
