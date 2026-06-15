"""
Unit tests for the IDOP 5-path semantic query router.

Tests all five classification paths (SQL, MUTATION, RAG, CHAT, HYBRID)
using mocked LangChain LLM chain, plus fallback behavior on API failure.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.graph.router import QueryRouter, RouteDecision

# ═══════════════════════════════════════════════════════════════════════
# Tests for QueryRouter
# ═══════════════════════════════════════════════════════════════════════


class TestQueryRouter:
    """Tests for the 5-class LLM semantic router."""

    @pytest.fixture
    def router(self):
        """Create a QueryRouter instance with mocked LLM chain."""
        with patch("app.core.graph.router.get_chat_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_chain = AsyncMock()
            # with_structured_output returns a new chain
            mock_llm.with_structured_output.return_value = mock_chain
            mock_get_llm.return_value = mock_llm
            r = QueryRouter()
            r._chain = mock_chain
            yield r

    # ----- Classification Tests -----

    @pytest.mark.asyncio
    async def test_classify_sql_query(self, router):
        """Test that analytical database questions are classified as SQL."""
        router._chain.ainvoke.return_value = RouteDecision(
            query_type="SQL",
            reason="The user wants to query database tables for product information.",
        )
        result = await router.route_query("Which products have never been ordered?")
        assert result == "SQL"

    @pytest.mark.asyncio
    async def test_classify_mutation_query(self, router):
        """Test that data modification requests are classified as MUTATION."""
        router._chain.ainvoke.return_value = RouteDecision(
            query_type="MUTATION",
            reason="The user wants to insert new rows from a file.",
        )
        result = await router.route_query("Insert new products from this Excel file")
        assert result == "MUTATION"

    @pytest.mark.asyncio
    async def test_classify_rag_query(self, router):
        """Test that knowledge search requests are classified as RAG."""
        router._chain.ainvoke.return_value = RouteDecision(
            query_type="RAG",
            reason="The user is asking about a policy document.",
        )
        result = await router.route_query("What is our company's refund policy?")
        assert result == "RAG"

    @pytest.mark.asyncio
    async def test_classify_chat_query(self, router):
        """Test that general conversational queries are classified as CHAT."""
        router._chain.ainvoke.return_value = RouteDecision(
            query_type="CHAT",
            reason="The user is greeting the system.",
        )
        result = await router.route_query("Hello, how are you?")
        assert result == "CHAT"

    @pytest.mark.asyncio
    async def test_classify_hybrid_query(self, router):
        """Test that combined SQL + document queries are classified as HYBRID."""
        router._chain.ainvoke.return_value = RouteDecision(
            query_type="HYBRID",
            reason="The user wants database data compared against document guidelines.",
        )
        result = await router.route_query(
            "Get sales data for customer X and compare it against the sales strategy in our PDF guidelines."
        )
        assert result == "HYBRID"

    # ----- Edge Cases -----

    @pytest.mark.asyncio
    async def test_invalid_query_type_falls_back_to_chat(self, router):
        """Test that an unrecognized query_type from the LLM falls back to CHAT."""
        router._chain.ainvoke.return_value = RouteDecision(
            query_type="UNKNOWN_TYPE",
            reason="Unrecognized classification.",
        )
        result = await router.route_query("Something weird and unclassifiable")
        assert result == "CHAT"

    @pytest.mark.asyncio
    async def test_api_failure_falls_back_to_chat(self, router):
        """Test that an LLM API error causes graceful fallback to CHAT."""
        router._chain.ainvoke.side_effect = Exception("API rate limit exceeded")
        result = await router.route_query("Show total revenue by region")
        assert result == "CHAT"

    @pytest.mark.asyncio
    async def test_lowercase_query_type_normalized_to_uppercase(self, router):
        """Test that lowercase query_type from LLM is normalized to uppercase."""
        router._chain.ainvoke.return_value = RouteDecision(
            query_type="sql",
            reason="Lowercase classification.",
        )
        result = await router.route_query("Show me all customers")
        assert result == "SQL"

    @pytest.mark.asyncio
    async def test_mixed_case_query_type_normalized(self, router):
        """Test that mixed-case query_type is normalized correctly."""
        router._chain.ainvoke.return_value = RouteDecision(
            query_type="Hybrid",
            reason="Mixed case classification.",
        )
        result = await router.route_query("Compare data with docs")
        assert result == "HYBRID"

    # ----- RouteDecision Model Tests -----

    def test_route_decision_model_accepts_all_types(self):
        """Test that the RouteDecision Pydantic model accepts all 5 query types."""
        for qtype in ["SQL", "MUTATION", "RAG", "CHAT", "HYBRID"]:
            decision = RouteDecision(query_type=qtype, reason=f"Test {qtype}")
            assert decision.query_type == qtype

    def test_route_decision_requires_reason(self):
        """Test that RouteDecision requires a reason field."""
        with pytest.raises(Exception):
            RouteDecision(query_type="SQL")  # Missing 'reason'


# ═══════════════════════════════════════════════════════════════════════
# Tests for Graph Routing Function (route_after_router)
# ═══════════════════════════════════════════════════════════════════════


class TestRouteAfterRouter:
    """Tests for the route_after_router conditional edge function."""

    def test_sql_routes_to_sql_gen(self):
        from app.core.graph.nodes import route_after_router

        state = {"query_type": "SQL"}
        assert route_after_router(state) == "sql_gen"

    def test_mutation_routes_to_mutation(self):
        from app.core.graph.nodes import route_after_router

        state = {"query_type": "MUTATION"}
        assert route_after_router(state) == "mutation"

    def test_rag_routes_to_ltm_remember(self):
        from app.core.graph.nodes import route_after_router

        state = {"query_type": "RAG"}
        assert route_after_router(state) == "ltm_remember"

    def test_chat_routes_to_chat(self):
        from app.core.graph.nodes import route_after_router

        state = {"query_type": "CHAT"}
        assert route_after_router(state) == "chat"

    def test_hybrid_routes_to_hybrid(self):
        from app.core.graph.nodes import route_after_router

        state = {"query_type": "HYBRID"}
        assert route_after_router(state) == "hybrid"

    def test_empty_type_routes_to_chat(self):
        from app.core.graph.nodes import route_after_router

        state = {"query_type": ""}
        assert route_after_router(state) == "chat"

    def test_missing_type_routes_to_chat(self):
        from app.core.graph.nodes import route_after_router

        state = {}
        assert route_after_router(state) == "chat"
